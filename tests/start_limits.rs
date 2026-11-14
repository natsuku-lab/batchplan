//! Opt-in correction of measured starts at joint limits, before the usual planning checks.

use std::collections::HashMap;

use batchplan::{
    Device, Error, HandConstraint, Handover, Held, Obstacle, Pick, Place, PlanFailure, PlanOptions,
    PlanProblem, Pose, Robot, RobotOptions, StartMotion, TaskOptions, TaskStep, ToolPath,
    ToolPathOptions, ToolPathStop, World,
};
use glam::{Quat, Vec3};

const TOLERANCE: f32 = 0.001;

fn robot() -> Robot {
    Robot::from_urdf(
        r#"<robot name="limits">
          <link name="base"/><link name="arm"/><link name="carriage"/>
          <link name="tip"><collision><geometry><sphere radius="0.02"/></geometry></collision></link>
          <link name="other_tip"/>
          <joint name="hinge" type="revolute">
            <parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
            <limit lower="-1" upper="1" effort="10" velocity="1"/>
          </joint>
          <joint name="slide" type="prismatic">
            <parent link="arm"/><child link="carriage"/><origin xyz="1 0 0"/><axis xyz="1 0 0"/>
            <limit lower="-1" upper="1" effort="10" velocity="1"/>
          </joint>
          <joint name="spin" type="continuous">
            <parent link="carriage"/><child link="tip"/><axis xyz="0 0 1"/>
            <limit effort="10" velocity="1"/>
          </joint>
          <joint name="other" type="prismatic">
            <parent link="base"/><child link="other_tip"/><origin xyz="0 0 2"/><axis xyz="0 1 0"/>
            <limit lower="-1" upper="1" effort="10" velocity="1"/>
          </joint>
        </robot>"#,
        &RobotOptions {
            ee_links: vec!["tip".into(), "other_tip".into()],
            ..RobotOptions::default()
        },
    )
    .unwrap()
}

fn joint(robot: &Robot, name: &str) -> usize {
    robot.joint_names().iter().position(|n| n == name).unwrap()
}

fn start(robot: &Robot) -> Vec<f32> {
    let mut q = robot.default_q().to_vec();
    q[joint(robot, "hinge")] = 1.0;
    q[joint(robot, "slide")] = 0.2;
    q[joint(robot, "spin")] = 7.0;
    q
}

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot).unwrap()];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

fn input<T>(result: Result<T, Error>) -> String {
    match result {
        Err(Error::Input(message)) => message,
        Err(error) => panic!("expected an input error, got {error}"),
        Ok(_) => panic!("invalid input was accepted"),
    }
}

fn options() -> PlanOptions {
    PlanOptions {
        start_tolerance: TOLERANCE,
        ..PlanOptions::default()
    }
}

fn path(robot: &Robot, q: Vec<f32>) -> ToolPath {
    let mut goal = start(robot);
    goal[joint(robot, "slide")] = 0.3;
    ToolPath {
        world: 0,
        start: q,
        hand: 0,
        waypoints: vec![robot.ee_pose(&goal, 0)],
        hold: vec![
            joint(robot, "hinge"),
            joint(robot, "spin"),
            joint(robot, "other"),
        ],
        holding: None,
    }
}

#[test]
fn corrected_batches_match_explicit_starts_and_keep_continuous_and_held_joints() {
    let robot = robot();
    let (hinge, slide, spin) = (
        joint(&robot, "hinge"),
        joint(&robot, "slide"),
        joint(&robot, "spin"),
    );
    let corrected = start(&robot);
    assert!(
        corrected[spin] > robot.upper()[spin],
        "the continuous joint must exercise its unbounded range"
    );
    let mut measured = corrected.clone();
    measured[hinge] = 1.0f32.next_up();
    let mut goal = corrected.clone();
    goal[slide] = 0.4;
    let first = PlanProblem {
        hold: vec![hinge, spin],
        ..PlanProblem::new(0, measured.clone(), goal.clone())
    };
    let mut second = PlanProblem::new(0, corrected.clone(), goal.clone());
    second.start[slide] = (-1.0f32).next_down();
    let problems = [first, second, PlanProblem::new(0, corrected.clone(), goal)];
    let mut explicit = problems.clone();
    explicit[0].start[hinge] = 1.0;
    explicit[1].start[slide] = -1.0;
    for device in devices(&robot) {
        let worlds = device.upload(&[World::default()]).unwrap();
        assert!(
            input(batchplan::plan(
                &device,
                &worlds,
                &problems,
                &PlanOptions::default()
            ))
            .contains("start")
        );
        let result = batchplan::plan(&device, &worlds, &problems, &options()).unwrap();
        let expected =
            batchplan::plan(&device, &worlds, &explicit, &PlanOptions::default()).unwrap();
        assert_eq!(
            result.solved().count(),
            problems.len(),
            "all corrected free-space paths must solve"
        );
        assert_eq!(
            result.paths, expected.paths,
            "correction must precede seeding and planning"
        );
        for (actual, expected) in result.problems.iter().zip(&explicit) {
            assert_eq!(
                actual.start, expected.start,
                "results retain the configuration actually planned"
            );
            assert_eq!(
                actual.start[spin], 7.0,
                "continuous joints must not be wrapped or clamped"
            );
        }
        let paths = [path(&robot, measured.clone())];
        assert!(
            input(batchplan::follow(
                &device,
                &worlds,
                &paths,
                &ToolPathOptions::default()
            ))
            .contains("start")
        );
        let followed = batchplan::follow(
            &device,
            &worlds,
            &paths,
            &ToolPathOptions {
                start_tolerance: TOLERANCE,
                ..ToolPathOptions::default()
            },
        )
        .unwrap();
        let manual = batchplan::follow(
            &device,
            &worlds,
            &[path(&robot, corrected.clone())],
            &ToolPathOptions::default(),
        )
        .unwrap();
        assert_eq!(
            followed.fraction,
            vec![1.0],
            "the fixture must follow a real path"
        );
        assert_eq!(followed.paths, manual.paths);
        assert_eq!(followed.problems[0].start, corrected);
    }
    assert_eq!(
        problems[0].start[hinge],
        1.0f32.next_up(),
        "caller-owned inputs stay intact"
    );
}

#[test]
fn correction_rejects_large_nonfinite_and_malformed_starts_and_never_repairs_goals() {
    let robot = robot();
    let q = start(&robot);
    let (hinge, spin) = (joint(&robot, "hinge"), joint(&robot, "spin"));
    for device in devices(&robot) {
        let worlds = device.upload(&[World::default()]).unwrap();
        for (index, value, name) in [
            (hinge, 1.002, "hinge"),
            (hinge, -1.002, "hinge"),
            (spin, f32::NAN, "spin"),
            (spin, f32::INFINITY, "spin"),
        ] {
            let mut problem = PlanProblem::new(0, q.clone(), q.clone());
            problem.start[index] = value;
            let message = input(batchplan::plan(&device, &worlds, &[problem], &options()));
            assert!(
                message.contains(name) && message.contains(&value.to_string()),
                "joint and rejected value missing: {message}"
            );
        }
        for size in [robot.dof() - 1, robot.dof() + 1] {
            let problem = PlanProblem::new(0, vec![0.0; size], q.clone());
            assert!(
                input(batchplan::plan(&device, &worlds, &[problem], &options()))
                    .contains("start has")
            );
        }
        for tolerance in [-1.0, f32::NAN, f32::INFINITY] {
            let o = PlanOptions {
                start_tolerance: tolerance,
                ..options()
            };
            assert!(input(batchplan::plan(&device, &worlds, &[], &o)).contains("start_tolerance"));
            let o = ToolPathOptions {
                start_tolerance: tolerance,
                ..ToolPathOptions::default()
            };
            assert!(
                input(batchplan::follow(&device, &worlds, &[], &o)).contains("start_tolerance")
            );
        }
        let mut problem = PlanProblem::new(0, q.clone(), q.clone());
        let mut bad_goal = q.clone();
        bad_goal[hinge] = 1.0f32.next_up();
        problem.goal = bad_goal.into();
        assert!(input(batchplan::plan(&device, &worlds, &[problem], &options())).contains("goal"));
    }
}

#[test]
fn exact_tolerance_limits_are_inclusive_and_held_goals_are_not_repaired() {
    let robot = robot();
    let hinge = joint(&robot, "hinge");
    let q = start(&robot);
    let exact = 0.125;
    for device in devices(&robot) {
        let worlds = device.upload(&[World::default()]).unwrap();
        let o = PlanOptions {
            start_tolerance: exact,
            ..options()
        };
        for measured in [1.125f32, -1.125] {
            let mut problem = PlanProblem::new(0, q.clone(), q.clone());
            problem.start[hinge] = measured;
            let result = batchplan::plan(&device, &worlds, &[problem.clone()], &o).unwrap();
            assert_eq!(result.problems[0].start[hinge], measured.signum());
            problem.start[hinge] = measured + measured.signum() * f32::EPSILON;
            assert!(input(batchplan::plan(&device, &worlds, &[problem], &o)).contains("hinge"));
        }
        let mut problem = PlanProblem::new(0, q.clone(), q.clone());
        problem.start[hinge] = 1.0f32.next_up();
        problem.hold.push(hinge);
        let mut goal = q.clone();
        goal[hinge] = 1.0f32.next_down();
        problem.goal = goal.into();
        assert!(
            input(batchplan::plan(&device, &worlds, &[problem], &options()))
                .contains("held joints")
        );
        for tolerance in [0.0, TOLERANCE] {
            for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
                let mut invalid = q.clone();
                invalid[joint(&robot, "spin")] = value;
                let o = ToolPathOptions {
                    start_tolerance: tolerance,
                    ..ToolPathOptions::default()
                };
                assert!(
                    input(batchplan::follow(
                        &device,
                        &worlds,
                        &[path(&robot, invalid)],
                        &o
                    ))
                    .contains("path")
                );
            }
        }
    }
}

#[test]
fn correction_preserves_a_moving_starts_velocity_and_acceleration() {
    let robot = robot();
    let hinge = joint(&robot, "hinge");
    let corrected = start(&robot);
    let mut measured = corrected.clone();
    measured[hinge] = 1.0f32.next_up();
    let mut goal = corrected.clone();
    goal[hinge] = 0.5;
    let mut motion = StartMotion {
        velocity: vec![0.0; robot.dof()],
        acceleration: vec![0.0; robot.dof()],
    };
    motion.velocity[hinge] = -0.2;
    motion.acceleration[hinge] = 0.05;
    let problem = PlanProblem {
        start_motion: Some(motion.clone()),
        ..PlanProblem::new(0, measured, goal)
    };
    let mut explicit = problem.clone();
    explicit.start = corrected;
    for device in devices(&robot) {
        let worlds = device.upload(&[World::default()]).unwrap();
        let result =
            batchplan::plan(&device, &worlds, std::slice::from_ref(&problem), &options()).unwrap();
        let expected = batchplan::plan(
            &device,
            &worlds,
            std::slice::from_ref(&explicit),
            &PlanOptions::default(),
        )
        .unwrap();
        assert!(
            result.best(0).is_some(),
            "the inward moving start must have a path: {:?}",
            result.failures
        );
        assert_eq!(result.paths, expected.paths);
        let actual = result.problems[0].start_motion.as_ref().unwrap();
        assert_eq!(actual.velocity, motion.velocity);
        assert_eq!(actual.acceleration, motion.acceleration);
    }
}

#[test]
fn corrected_starts_still_undergo_collision_and_constraint_checks() {
    let robot = robot();
    let corrected = start(&robot);
    let mut measured = corrected.clone();
    measured[joint(&robot, "hinge")] = 1.0f32.next_up();
    let pose = robot.ee_pose(&corrected, 0);
    let obstacle = Obstacle::Sphere {
        center: pose.position,
        radius: 0.1,
    };
    for device in devices(&robot) {
        let worlds = device
            .upload(&[World::new(vec![obstacle.clone()]), World::default()])
            .unwrap();
        let mut problem = PlanProblem::new(0, measured.clone(), corrected.clone());
        let result = batchplan::plan(&device, &worlds, &[problem.clone()], &options()).unwrap();
        assert!(
            matches!(&result.failures[0], Some(PlanFailure::StartCollides { contacts }) if !contacts.is_empty()),
            "a corrected colliding start must still fail"
        );
        let followed = batchplan::follow(
            &device,
            &worlds,
            &[path(&robot, measured.clone())],
            &ToolPathOptions {
                start_tolerance: TOLERANCE,
                ..ToolPathOptions::default()
            },
        )
        .unwrap();
        assert!(
            matches!(followed.stopped[0], Some(ToolPathStop::Collision { .. })),
            "follow must still check collision"
        );
        problem.world = 1;
        let mut away = pose;
        away.position += Vec3::Z;
        problem.constraints.push(HandConstraint {
            position: Vec3::splat(0.01),
            ..HandConstraint::new(0, away)
        });
        assert!(
            input(batchplan::plan(&device, &worlds, &[problem], &options()))
                .contains("start breaks its constraints")
        );
    }
}

#[test]
fn every_task_corrects_its_start_and_preserves_the_start_collision_failure() {
    let robot = robot();
    let corrected = start(&robot);
    let mut measured = corrected.clone();
    measured[joint(&robot, "hinge")] = 1.0f32.next_up();
    let pose = robot.ee_pose(&corrected, 0);
    let object = Obstacle::Sphere {
        center: Vec3::ZERO,
        radius: 0.01,
    };
    let held = Held {
        hand: 0,
        object: object.clone(),
        grasp: Pose::IDENTITY,
        locked: HashMap::new(),
    };
    let pick = Pick::new(0, measured.clone(), 1, vec![Pose::IDENTITY]);
    let place = Place::new(1, measured.clone(), held.clone(), vec![pose]);
    let handover = Handover {
        world: 1,
        start: measured,
        held,
        to: 1,
        meetings: vec![pose],
        grasps: vec![Pose::IDENTITY],
        approach: 0.1,
        retreat: 0.1,
        constraints: vec![],
    };
    let o = TaskOptions {
        plan: options(),
        ..TaskOptions::default()
    };
    for device in devices(&robot) {
        let barrier = Obstacle::Sphere {
            center: pose.position,
            radius: 0.1,
        };
        let worlds = device
            .upload(&[
                World::new(vec![
                    barrier.clone(),
                    object.placed(Pose::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY)),
                ]),
                World::new(vec![barrier]),
            ])
            .unwrap();
        assert!(
            input(batchplan::pick(
                &device,
                &worlds,
                std::slice::from_ref(&pick),
                &TaskOptions::default()
            ))
            .contains("start")
        );
        assert!(
            input(batchplan::place(
                &device,
                &worlds,
                std::slice::from_ref(&place),
                &TaskOptions::default()
            ))
            .contains("start")
        );
        assert!(
            input(batchplan::handover(
                &device,
                &worlds,
                std::slice::from_ref(&handover),
                &TaskOptions::default()
            ))
            .contains("start")
        );
        let picked = batchplan::pick(&device, &worlds, std::slice::from_ref(&pick), &o).unwrap();
        let placed = batchplan::place(&device, &worlds, std::slice::from_ref(&place), &o).unwrap();
        let passed =
            batchplan::handover(&device, &worlds, std::slice::from_ref(&handover), &o).unwrap();
        for actual in [
            &picked.problems[0].start,
            &placed.problems[0].start,
            &passed.problems[0].start,
        ] {
            assert_eq!(
                actual, &corrected,
                "task results retain their corrected starts"
            );
        }
        for outcome in [
            &picked.outcomes[0],
            &placed.outcomes[0],
            &passed.outcomes[0],
        ] {
            let failure = outcome.as_ref().unwrap_err();
            assert_eq!(
                failure.at(TaskStep::Start),
                failure.candidates.len(),
                "all candidates must still fail at their colliding start"
            );
            assert!(
                !failure.candidates.is_empty(),
                "the fixture must attempt a candidate"
            );
        }
        let alone_pick = Pick {
            transit: false,
            ..pick.clone()
        };
        let mut alone_place = Place {
            transit: false,
            ..place.clone()
        };
        alone_place.poses[0].position += Vec3::Z;
        let picked = batchplan::pick(&device, &worlds, &[alone_pick], &o).unwrap();
        let placed = batchplan::place(&device, &worlds, &[alone_place], &o).unwrap();
        assert_eq!(picked.problems[0].start, corrected);
        assert_eq!(placed.problems[0].start, corrected);
        for outcome in [&picked.outcomes[0], &placed.outcomes[0]] {
            let failure = outcome.as_ref().unwrap_err();
            assert_eq!(
                failure.at(TaskStep::Start),
                0,
                "a task without transit skips the start collision check"
            );
        }
    }
}
