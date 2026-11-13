//! Pose goals in `plan`: solved by IK into the configurations a path may end at, more seeds where
//! the first find nothing collision-free, and failures that say whether the poses are out of reach
//! or reached only in collision.

#[path = "../examples/common/mod.rs"]
mod common;

use std::time::Duration;

use batchplan::glam::{Quat, Vec3};
use batchplan::{
    Device, Goal, Held, IkOptions, IkProblem, Obstacle, PlanFailure, PlanOptions, PlanProblem,
    Pose, Robot, RobotOptions, World,
};

fn panda() -> Robot {
    common::panda().unwrap()
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

/// The hand pointing down, `position` in front of the robot.
fn down(position: Vec3) -> Pose {
    Pose::new(position, Quat::from_rotation_x(std::f32::consts::PI))
}

/// How far the path's end puts the hand from `target`: meters, radians.
fn missed(robot: &Robot, path: &[f32], target: Pose) -> (f32, f32) {
    let end = robot.ee_pose(&path[path.len() - robot.dof()..], 0);
    (
        end.position.distance(target.position),
        end.rotation.angle_between(target.rotation),
    )
}

/// Targets in front of, beside and behind the robot, all reachable with the hand down.
const TARGETS: [Vec3; 5] = [
    Vec3::new(0.3, 0.5, 0.3),
    Vec3::new(-0.3, 0.4, 0.5),
    Vec3::new(0.2, -0.6, 0.2),
    Vec3::new(-0.4, -0.3, 0.3),
    Vec3::new(0.6, 0.0, 0.1),
];

#[test]
fn a_pose_goal_ends_with_the_hand_at_the_pose_on_every_device() {
    let robot = panda();
    let start = robot.default_q().to_vec();
    let problems: Vec<PlanProblem> = TARGETS
        .iter()
        .map(|&p| PlanProblem::new(0, start.clone(), down(p)))
        .collect();
    let mut first = None;
    for device in devices(&robot) {
        let worlds = device.upload(&[World::default()]).unwrap();
        let result = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap();
        for (i, &target) in TARGETS.iter().enumerate() {
            let path = result
                .best(i)
                .unwrap_or_else(|| panic!("target {i}: {:?}", result.failures[i]));
            let (meters, radians) = missed(&robot, path, down(target));
            assert!(
                meters < 1.5e-3 && radians < 1.5e-2,
                "target {i}: {meters} m, {radians} rad"
            );
            assert_eq!(&path[..robot.dof()], &start[..]);
        }
        assert!(
            matches!(result.problems[0].goal, Goal::Poses(_)),
            "the problems as given"
        );
        let paths: Vec<Option<Vec<f32>>> = (0..TARGETS.len())
            .map(|i| result.best(i).map(<[f32]>::to_vec))
            .collect();
        // IK on a GPU rounds unlike the CPU's, so the goals planned to agree to rounding.
        match &first {
            None => first = Some(paths),
            Some(cpu) => {
                for (a, b) in cpu.iter().zip(&paths) {
                    let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
                    let apart = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0, f32::max);
                    assert!(
                        a.len() == b.len() && apart < 1e-4,
                        "{} plans {apart} rad apart",
                        device.name()
                    );
                }
            }
        }
    }
}

/// Joint-space distance.
fn apart(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn a_pose_goal_ends_at_the_solution_nearest_the_start() {
    // The start is not the robot's default configuration, which unseeded IK starts from.
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let mut start = robot.default_q().to_vec();
    start[0] += 1.2;
    start[5] -= 0.5;
    let target = down(TARGETS[4]);
    let ik = batchplan::IkProblem {
        seed: Some(start.clone()),
        ..batchplan::IkProblem::new(0, target)
    };
    let o = PlanOptions::default();
    let ik = batchplan::solve_ik(&device, &worlds, &[ik], &o.goal_ik).unwrap();
    let solutions = (0..o.goal_ik.seeds).filter(|&i| ik.success[i]);
    let nearest = solutions
        .min_by(|&a, &b| apart(ik.solution(a), &start).total_cmp(&apart(ik.solution(b), &start)))
        .unwrap();
    assert!(
        nearest != 0,
        "the seed from the start is not the nearest solution"
    );
    let result = batchplan::plan(&device, &worlds, &[PlanProblem::new(0, start, target)], &o);
    let path = result.unwrap().best(0).unwrap().to_vec();
    assert_eq!(&path[path.len() - robot.dof()..], ik.solution(nearest));
}

#[test]
fn sixteen_times_the_seeds_solve_what_four_times_miss() {
    // One seed of ten iterations, then four, then sixteen: only the last reaches this pose.
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    let target = down(Vec3::new(-0.3, -0.28, 0.39));
    let few = IkOptions {
        seeds: 1,
        iterations: 10,
        ..IkOptions::default()
    };
    let solves = |seeds: usize| {
        let problem = IkProblem {
            seed: Some(start.clone()),
            ..IkProblem::new(0, target)
        };
        let o = IkOptions { seeds, ..few };
        let ik = batchplan::solve_ik(&device, &worlds, &[problem], &o).unwrap();
        ik.success.iter().any(|&s| s)
    };
    assert!(
        !solves(4) && solves(16),
        "four times the seeds miss, sixteen reach"
    );
    let o = PlanOptions {
        goal_ik: few,
        ..PlanOptions::default()
    };
    let problem = PlanProblem::new(0, start.clone(), target);
    let result = batchplan::plan(&device, &worlds, &[problem], &o).unwrap();
    assert!(result.best(0).is_some(), "{:?}", result.failures[0]);
}

/// Each joint of `q` at its value nearest `from` among whole turns within its limits.
fn nearest_turn(robot: &Robot, from: &[f32], q: &[f32]) -> Vec<f32> {
    let turns = |j: usize| (-2..=2).map(move |k| q[j] + k as f32 * std::f32::consts::TAU);
    (0..q.len())
        .map(|j| {
            let within = turns(j).filter(|v| (robot.lower()[j]..=robot.upper()[j]).contains(v));
            within
                .min_by(|a, b| (a - from[j]).abs().total_cmp(&(b - from[j]).abs()))
                .unwrap_or(q[j])
        })
        .collect()
}

#[test]
fn a_pose_goal_ends_at_the_nearest_turn_of_its_solutions() {
    // The UR5e's joints span two turns, and the path ends at a solution's nearest turn: the
    // solution nearest the start counts there, not where IK left it.
    let robot = Robot::load(
        common::asset("ur5e/ur_description/urdf/ur5e.urdf"),
        &RobotOptions::default(),
    )
    .unwrap();
    let device = Device::cpu(&robot).unwrap();
    let table = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.02), Vec3::new(1.0, 1.0, 0.02));
    let worlds = device.upload(&[World::new(vec![table])]).unwrap();
    let start = robot.default_q().to_vec();
    let target = down(Vec3::new(0.4, 0.2, 0.3));
    let o = PlanOptions::default();
    let problem = IkProblem {
        seed: Some(start.clone()),
        ..IkProblem::new(0, target)
    };
    let ik = batchplan::solve_ik(&device, &worlds, &[problem], &o.goal_ik).unwrap();
    let solutions: Vec<&[f32]> = (0..o.goal_ik.seeds)
        .filter(|&i| ik.success[i])
        .map(|i| ik.solution(i))
        .collect();
    let turned = |q: &[f32]| apart(&nearest_turn(&robot, &start, q), &start);
    let nearest = |of: &dyn Fn(&[f32]) -> f32| {
        (0..solutions.len())
            .min_by(|&a, &b| of(solutions[a]).total_cmp(&of(solutions[b])))
            .unwrap()
    };
    let (as_left, as_turned) = (nearest(&|q| apart(q, &start)), nearest(&turned));
    assert!(
        as_left != as_turned,
        "the solution IK leaves nearest is not the nearest turned"
    );
    let result = batchplan::plan(
        &device,
        &worlds,
        &[PlanProblem::new(0, start.clone(), target)],
        &o,
    );
    let path = result.unwrap().best(0).unwrap().to_vec();
    let end = &path[path.len() - robot.dof()..];
    let (ends, nearest_end) = (apart(end, &start), turned(solutions[as_turned]));
    assert!(
        (ends - nearest_end).abs() < 1e-4,
        "ends {ends} rad away, against {nearest_end}"
    );
}

#[test]
fn pose_goals_keep_what_the_hand_holds() {
    // IK that leaves the held rod out reaches this pose where the rod hits the arm; plan's IK
    // holds it and finds a pose that keeps it clear.
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    let rod = Held {
        locked: Default::default(),
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.3, 0.02)),
        grasp: Pose::IDENTITY,
    };
    let rotation = Quat::from_xyzw(0.1089, -0.5573, -0.5139, 0.6430).normalize();
    let target = Pose::new(Vec3::new(0.1056, -0.1596, 0.4082), rotation);
    let o = PlanOptions::default();
    let bare = IkProblem {
        seed: Some(start.clone()),
        ..IkProblem::new(0, target)
    };
    let ik = batchplan::solve_ik(&device, &worlds, &[bare], &o.goal_ik).unwrap();
    let holding = |goal: Goal| PlanProblem {
        holding: Some(rod.clone()),
        ..PlanProblem::new(0, start.clone(), goal)
    };
    let without = holding(ik.nearest(0, &start).unwrap().into());
    let without = batchplan::plan(&device, &worlds, &[without], &o).unwrap();
    assert!(
        matches!(without.failures[0], Some(PlanFailure::GoalCollides { .. })),
        "the rod collides where IK without it ends: {:?}",
        without.failures[0]
    );
    let result = batchplan::plan(&device, &worlds, &[holding(target.into())], &o).unwrap();
    assert!(result.best(0).is_some(), "{:?}", result.failures[0]);
}

#[test]
fn goal_ik_rounds_stop_once_the_time_budget_passes() {
    // With no time at all, the first round runs and the pose out of reach is not solved again.
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let far = PlanProblem::new(0, robot.default_q(), down(Vec3::new(2.0, 0.0, 0.3)));
    let at_once = PlanOptions {
        time_budget: Some(Duration::ZERO),
        ..PlanOptions::default()
    };
    let result = batchplan::plan(&device, &worlds, std::slice::from_ref(&far), &at_once).unwrap();
    assert_eq!(result.failures[0], Some(PlanFailure::OutOfTime));
    let result = batchplan::plan(&device, &worlds, &[far], &PlanOptions::default()).unwrap();
    assert_eq!(result.failures[0], Some(PlanFailure::GoalUnreachable));
}

#[test]
fn pose_goals_out_of_reach_or_inside_an_obstacle_say_so() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let inside = TARGETS[0];
    let worlds = device
        .upload(&[World::new(vec![Obstacle::sphere(inside, 0.1)])])
        .unwrap();
    let start = robot.default_q().to_vec();
    let problems = [
        PlanProblem::new(0, start.clone(), down(TARGETS[2])),
        PlanProblem::new(0, start.clone(), down(Vec3::new(2.0, 0.0, 0.3))),
        PlanProblem::new(0, start.clone(), down(inside)),
        PlanProblem::new(0, start.clone(), start),
    ];
    let result = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap();
    assert!(
        result.best(0).is_some() && result.best(3).is_some(),
        "{:?}",
        result.failures
    );
    assert_eq!(result.failures[1], Some(PlanFailure::GoalUnreachable));
    assert!(
        matches!(&result.failures[2], Some(PlanFailure::GoalCollides { contacts }) if !contacts.is_empty()),
        "{:?}",
        result.failures[2]
    );
    assert!(result.best(1).is_none() && result.best(2).is_none());
    // The joint goal planned beside them plans as it does alone.
    let alone = batchplan::plan(&device, &worlds, &problems[3..], &PlanOptions::default()).unwrap();
    assert_eq!(result.best(3), alone.best(0));
}

#[test]
fn pose_goals_keep_held_joints_and_constraints() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    let mut moved = start.clone();
    moved[1] += 0.3;
    moved[3] += 0.3;
    let target = robot.ee_pose(&moved, 0);
    let held = PlanProblem {
        hold: vec![0],
        ..PlanProblem::new(0, start.clone(), target)
    };
    let result = batchplan::plan(&device, &worlds, &[held], &PlanOptions::default()).unwrap();
    let path = result.best(0).unwrap();
    assert!(
        path.chunks(robot.dof()).all(|q| q[0] == start[0]),
        "the base stays"
    );
    // Pinned upright, the hand can't end turned on its side: every solution breaks it.
    let upright = batchplan::HandConstraint {
        tilt: 0.1,
        ..batchplan::HandConstraint::new(0, robot.ee_pose(&start, 0))
    };
    let sideways = Pose::new(
        TARGETS[0],
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
    );
    let pinned = PlanProblem {
        constraints: vec![upright],
        ..PlanProblem::new(0, start, sideways)
    };
    let result = batchplan::plan(&device, &worlds, &[pinned], &PlanOptions::default()).unwrap();
    assert_eq!(result.failures[0], Some(PlanFailure::GoalUnreachable));
}

#[test]
fn malformed_pose_goals_are_input_errors() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    let target = down(TARGETS[0]);
    let two_hands = PlanProblem {
        goal: Goal::Poses(vec![Some(target), None]),
        ..PlanProblem::new(0, start.clone(), target)
    };
    let skewed = Pose::new(target.position, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0));
    let no_poses = PlanProblem {
        goal: Goal::Poses(vec![None]),
        ..PlanProblem::new(0, start.clone(), target)
    };
    // Behind a joint goal, so the error names the problem's index in the batch, not in goal IK's.
    let fine = PlanProblem::new(0, start.clone(), start.clone());
    for (problem, why) in [
        (two_hands, "hands"),
        (PlanProblem::new(0, start, skewed), "unit"),
        (no_poses, "at least one pose"),
    ] {
        let problems = [fine.clone(), problem];
        let e = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap_err();
        let e = e.to_string();
        assert!(e.contains("problem 1") && e.contains(why), "{e}");
    }
    let nowhere = Pose::new(Vec3::NAN, Quat::IDENTITY);
    let problems = [PlanProblem::new(0, fine.start.clone(), nowhere)];
    let e = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap_err();
    assert!(e.to_string().contains("finite position"), "{e}");
    // Goal IK's options are checked with the plan's, whatever the goals.
    let no_seeds = PlanOptions {
        goal_ik: IkOptions {
            seeds: 0,
            ..IkOptions::default()
        },
        ..PlanOptions::default()
    };
    let e = batchplan::plan(&device, &worlds, &[fine], &no_seeds).unwrap_err();
    assert!(
        e.to_string().contains("goal_ik needs at least one seed"),
        "{e}"
    );
}
