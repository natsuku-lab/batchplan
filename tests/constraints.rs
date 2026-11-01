//! Hand constraints through `plan` and `rrt::connect`: a gripper carried upright past a pillar,
//! kept inside a box, turned within a twist, beside an arm that waits; ends that break their
//! constraints, and constraints that cannot be kept, rejected; constrained plans the same alone
//! and in a batch, on every device. GPU parts are skipped when no adapter is available unless
//! `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::{
    CollisionWeights, Device, Error, HandConstraint, IkProblem, PlanOptions, PlanProblem, Pose,
    Robot, RrtOptions, RrtProblem, Trajectory, Worlds,
};
use glam::{Quat, Vec3};

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot).unwrap()];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

/// A path's configurations at 1 kHz once timed, far denser than the planner checks it; the
/// trajectory must pass `Trajectory::check`, within every joint range and limit.
fn samples(robot: &Robot, path: &[f32]) -> Vec<Vec<f32>> {
    let trajectory = Trajectory::try_new(robot, path, 1.0).unwrap();
    trajectory.check(robot).unwrap();
    let timed = trajectory.sample(1000.0).unwrap();
    timed
        .positions
        .chunks(timed.dof)
        .map(<[f32]>::to_vec)
        .collect()
}

/// How far hand `hand`'s z axis is from `rotation`'s, and how far it turns about it (radians).
fn tilt_and_twist(robot: &Robot, q: &[f32], hand: usize, rotation: Quat) -> (f32, f32) {
    let relative = rotation.inverse() * robot.ee_pose(q, hand).rotation;
    let tilt = (relative * Vec3::Z).angle_between(Vec3::Z);
    let m = glam::Mat3::from_quat(relative);
    let twist = (m.x_axis.y - m.y_axis.x).atan2(m.x_axis.x + m.y_axis.y);
    (tilt, twist.abs())
}

/// Whether every configuration is collision-free in world 0.
fn free(device: &Device, worlds: &Worlds, q: &[Vec<f32>]) -> bool {
    let flat: Vec<f32> = q.concat();
    let e = device
        .evaluate(worlds, &vec![0; q.len()], &flat, &CollisionWeights::NONE)
        .unwrap();
    (0..q.len()).all(|i| e.collision_free(i))
}

/// The gripper carried upright, within 5°, past a pillar to every goal: each valid path, searched
/// or optimized, sampled densely once timed, keeps the bound and the scene, where most of the same
/// problems planned without the constraint tilt past it. Optimization keeps some seeds valid and
/// shortens some paths.
#[test]
fn a_gripper_carried_past_a_pillar_stays_upright() {
    let robot = common::panda().unwrap();
    let bound = 5f32.to_radians();
    let down = common::upright(0.0);
    for device in devices(&robot) {
        let worlds = device.upload(&[common::pillar_scene()]).unwrap();
        let (start, goals) =
            common::upright_reaches(&device, &worlds, 16, &mut Rng::new(5)).unwrap();
        assert!(goals.len() >= 12, "only {} goals in reach", goals.len());
        let upright = HandConstraint {
            tilt: bound,
            ..HandConstraint::new(0, Pose::new(Vec3::ZERO, down))
        };
        let problems: Vec<PlanProblem> = goals
            .iter()
            .map(|g| PlanProblem {
                constraints: vec![upright],
                ..PlanProblem::new(0, start.clone(), g.clone())
            })
            .collect();
        let o = PlanOptions {
            seeds: 2,
            ..PlanOptions::default()
        };
        let result = batchplan::plan(&device, &worlds, &problems, &o).unwrap();
        assert_eq!(
            result.solved().count(),
            problems.len(),
            "{}: every upright reach should be planned",
            device.name()
        );
        let (mut optimized, mut shorter) = (0, 0);
        for p in 0..problems.len() {
            let base = p * result.seeds;
            for seed in (0..result.seeds).filter(|&k| result.valid[base + k]) {
                let q = samples(&robot, result.paths.path(base + seed));
                let most = q
                    .iter()
                    .map(|q| tilt_and_twist(&robot, q, 0, down).0)
                    .fold(0.0, f32::max);
                assert!(
                    most <= bound + 1e-4,
                    "{}: problem {p}, seed {seed} tilts {:.3}° along its path",
                    device.name(),
                    most.to_degrees()
                );
                assert!(
                    free(&device, &worlds, &q),
                    "{}: problem {p}, seed {seed} collides",
                    device.name()
                );
                if seed > 0 {
                    optimized += 1;
                    shorter += usize::from(result.length[base + seed] < result.length[base]);
                }
            }
        }
        assert!(
            optimized >= problems.len() / 2 && shorter > 0,
            "{}: {optimized} optimized seeds valid, {shorter} shorter than their search",
            device.name()
        );
        let unconstrained: Vec<PlanProblem> = problems
            .iter()
            .map(|p| PlanProblem {
                constraints: vec![],
                ..p.clone()
            })
            .collect();
        let free_result = batchplan::plan(&device, &worlds, &unconstrained, &o).unwrap();
        let tilts: Vec<f32> = free_result
            .solved()
            .map(|s| {
                samples(&robot, s.solution)
                    .iter()
                    .map(|q| tilt_and_twist(&robot, q, 0, down).0.to_degrees())
                    .fold(0.0, f32::max)
            })
            .collect();
        let past = tilts.iter().filter(|&&t| t > bound.to_degrees()).count();
        assert!(
            2 * past > tilts.len(),
            "{}: most unconstrained paths should tilt past the bound, not {past} of {}: {tilts:?}",
            device.name(),
            tilts.len()
        );
    }
}

/// Optimized seeds that break their constraints are not valid: stopped after one step, the
/// straight lines optimization starts from still tilt the gripper past its bound, and every seed
/// still valid keeps it at 1 kHz.
#[test]
fn optimized_seeds_that_break_their_constraints_are_invalid() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::pillar_scene()]).unwrap();
    let (start, goals) = common::upright_reaches(&device, &worlds, 16, &mut Rng::new(5)).unwrap();
    let down = common::upright(0.0);
    let bound = 5f32.to_radians();
    let upright = HandConstraint {
        tilt: bound,
        ..HandConstraint::new(0, Pose::new(Vec3::ZERO, down))
    };
    let problems: Vec<PlanProblem> = goals
        .iter()
        .map(|g| PlanProblem {
            constraints: vec![upright],
            ..PlanProblem::new(0, start.clone(), g.clone())
        })
        .collect();
    let o = PlanOptions {
        seeds: 4,
        iterations: 1,
        ..PlanOptions::default()
    };
    let result = batchplan::plan(&device, &worlds, &problems, &o).unwrap();
    let (mut kept, mut refused) = (0, 0);
    for p in 0..problems.len() {
        for seed in 1..result.seeds {
            let item = p * result.seeds + seed;
            if !result.valid[item] {
                refused += 1;
                continue;
            }
            let most = samples(&robot, result.paths.path(item))
                .iter()
                .map(|q| tilt_and_twist(&robot, q, 0, down).0)
                .fold(0.0, f32::max);
            assert!(
                most <= bound + 1e-4,
                "problem {p}, seed {seed} is valid and tilts {:.2}°",
                most.to_degrees()
            );
            kept += 1;
        }
    }
    assert!(kept > 0 && refused > 0, "{kept} kept, {refused} refused");
}

/// A spline too short to follow its path within 1° is repaired keeping the bound: with 10 control
/// points, every upright reach plans, and keeps the bound at 1 kHz. Reaches whose searched path is
/// over 4 rad are left out: one winds 5.8 rad on the M4 Pro, twice the others' 2.2 on average, past
/// what 10 control points follow.
#[test]
fn short_splines_are_repaired_within_their_constraints() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::pillar_scene()]).unwrap();
    let (start, goals) =
        common::upright_reaches(&device, &worlds, 64, &mut Rng::new(1234)).unwrap();
    let down = common::upright(0.0);
    let bound = 1f32.to_radians();
    let upright = HandConstraint {
        tilt: bound,
        ..HandConstraint::new(0, Pose::new(Vec3::ZERO, down))
    };
    let problems: Vec<PlanProblem> = goals
        .iter()
        .map(|g| PlanProblem {
            constraints: vec![upright],
            ..PlanProblem::new(0, start.clone(), g.clone())
        })
        .collect();
    let searched = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap();
    let problems: Vec<PlanProblem> = problems
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| searched.length[i] < 4.0)
        .map(|(_, p)| p)
        .collect();
    assert!(
        problems.len() >= 50,
        "{} reaches under 4 rad",
        problems.len()
    );
    let o = PlanOptions {
        control_points: 10,
        ..PlanOptions::default()
    };
    let result = batchplan::plan(&device, &worlds, &problems, &o).unwrap();
    assert_eq!(result.solved().count(), problems.len(), "every reach plans");
    for s in result.solved() {
        let most = samples(&robot, s.solution)
            .iter()
            .map(|q| tilt_and_twist(&robot, q, 0, down).0)
            .fold(0.0, f32::max);
        assert!(
            most <= bound + 1e-4,
            "problem {} tilts {:.3}°",
            s.index,
            most.to_degrees()
        );
    }
}

/// A spline fitted through a constrained path stays within the joints' ranges, where a start at a
/// joint's limit drew the fit's control points past it: here the start's third joint sits at its
/// limit, within 1° of upright.
#[test]
fn constrained_splines_stay_within_the_joint_ranges() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::pillar_scene()]).unwrap();
    // The IK solution with the most clearance in the zone's corner, not the one nearest the ready
    // pose: its third joint at its limit.
    let corner = IkProblem::new(0, Pose::new(common::zone_corner(), common::upright(0.0)));
    let start = batchplan::solve_ik(&device, &worlds, &[corner], &Default::default())
        .unwrap()
        .best(0)
        .expect("the corner is in reach")
        .to_vec();
    let goals = common::upright_goals(&device, &worlds, &start, 32, &mut Rng::new(109)).unwrap();
    assert_eq!(
        start[2],
        robot.upper()[2],
        "the start should sit at joint 2's limit"
    );
    let upright = HandConstraint {
        tilt: 1f32.to_radians(),
        ..HandConstraint::new(0, Pose::new(Vec3::ZERO, common::upright(0.0)))
    };
    let problem = PlanProblem {
        constraints: vec![upright],
        ..PlanProblem::new(0, start, goals[6].clone())
    };
    let result = batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default()).unwrap();
    let path = result.best(0).expect("a path");
    for point in path.chunks(robot.dof()) {
        for (j, v) in point.iter().enumerate() {
            assert!(
                (robot.lower()[j]..=robot.upper()[j]).contains(v),
                "a control point leaves joint {j}'s range: {v}"
            );
        }
    }
    samples(&robot, path);
}

/// A hand kept in a box 10 cm around the line between its ends, tilted within 0.5 rad and turned
/// within a little more than the goal's twist, keeps all three at every sample, where
/// unconstrained paths leave them.
#[test]
fn hands_stay_in_their_box_and_twist() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::pillar_scene()]).unwrap();
    let (start, goals) = common::upright_reaches(&device, &worlds, 24, &mut Rng::new(8)).unwrap();
    let from = robot.ee_pose(&start, 0);
    let mut checked = [0; 2];
    for goal in &goals {
        let to = robot.ee_pose(goal, 0);
        let middle = 0.5 * (from.position + to.position);
        let half = 0.5 * (to.position - from.position).abs() + Vec3::splat(0.1);
        let twist = tilt_and_twist(&robot, goal, 0, from.rotation).1 + 0.15;
        let constraint = HandConstraint {
            position: half,
            tilt: 0.5,
            twist,
            ..HandConstraint::new(0, Pose::new(middle, from.rotation))
        };
        let problem = PlanProblem {
            constraints: vec![constraint],
            ..PlanProblem::new(0, start.clone(), goal.clone())
        };
        let plain = PlanProblem::new(0, start.clone(), goal.clone());
        // Problems the box leaves no way through give up sooner.
        let o = PlanOptions {
            rrt: RrtOptions {
                max_iterations: 2000,
                ..PlanOptions::default().rrt
            },
            ..PlanOptions::default()
        };
        let result = batchplan::plan(&device, &worlds, &[problem, plain], &o).unwrap();
        let outside =
            |q: &[f32]| ((robot.ee_pose(q, 0).position - middle).abs() - half).max_element() > 1e-4;
        let twisted = |q: &[f32]| {
            let (tilt, turn) = tilt_and_twist(&robot, q, 0, from.rotation);
            tilt > 0.5 + 1e-4 || turn > twist + 1e-4
        };
        if let Some(path) = result.best(0) {
            let q = samples(&robot, path);
            assert!(
                !q.iter().any(|q| outside(q)),
                "a constrained path leaves its box"
            );
            assert!(
                !q.iter().any(|q| twisted(q)),
                "a constrained path tilts or turns past its bound"
            );
            checked[0] += 1;
        }
        let q = samples(&robot, result.best(1).expect("an unconstrained path"));
        checked[1] += usize::from(q.iter().any(|q| outside(q) || twisted(q)));
    }
    assert!(
        checked[0] >= 3 && checked[1] >= 3,
        "{} constrained paths checked, {} unconstrained ones leave the bounds",
        checked[0],
        checked[1]
    );
}

/// On two arms, the left one carries its gripper upright while the right one waits: the held
/// joints keep the start's values bit for bit at every control point, and the left hand keeps
/// its bound.
#[test]
fn an_arm_carries_upright_while_the_other_waits() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[batchplan::World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    let (left, right) = (robot.hand_joints(0), robot.hand_joints(1));
    let rotation = robot.ee_pose(&start, 0).rotation;
    let bound = 0.1;
    let mut rng = Rng::new(3);
    let mut planned = 0;
    for _ in 0..8 {
        // A goal for the left arm by IK at an upright pose nearby.
        let target = robot.ee_pose(&start, 0).position
            + Vec3::new(
                rng.range(-0.15, 0.15),
                rng.range(-0.2, 0.2),
                rng.range(-0.15, 0.1),
            );
        let ik = batchplan::IkProblem {
            targets: vec![Some(Pose::new(target, rotation)), None],
            seed: Some(start.clone()),
            ..batchplan::IkProblem::new(0, Pose::new(target, rotation))
        };
        let solved = batchplan::solve_ik(&device, &worlds, &[ik], &Default::default()).unwrap();
        let Some(goal) = solved.nearest(0, &start) else {
            continue;
        };
        let problem = PlanProblem {
            hold: right.clone(),
            constraints: vec![HandConstraint {
                tilt: bound,
                ..HandConstraint::new(0, Pose::new(Vec3::ZERO, rotation))
            }],
            ..PlanProblem::new(0, start.clone(), goal.to_vec())
        };
        let result =
            batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default()).unwrap();
        let Some(path) = result.best(0) else { continue };
        for point in path.chunks(robot.dof()) {
            assert!(
                right.iter().all(|&j| point[j] == start[j]),
                "a held joint moved"
            );
        }
        let moved = path
            .chunks(robot.dof())
            .any(|p| left.iter().any(|&j| p[j] != start[j]));
        assert!(moved, "the left arm should move");
        for q in samples(&robot, path) {
            assert!(tilt_and_twist(&robot, &q, 0, rotation).0 <= bound + 1e-4);
        }
        planned += 1;
    }
    assert!(planned >= 5, "{planned} of 8 planned");
}

/// How far hand 1 is from its place `relative` in hand 0's frame: along each of that frame's axes
/// at most (meters), and its tilt and twist (radians).
fn off_its_place(robot: &Robot, q: &[f32], relative: Pose) -> (f32, f32, f32) {
    let (a, b) = (robot.ee_pose(q, 0), robot.ee_pose(q, 1));
    let now = a.inverse().mul_pose(b);
    let away = (now.position - relative.position).abs().max_element();
    let (tilt, twist) = {
        let m = relative.rotation.inverse() * now.rotation;
        let r = glam::Mat3::from_quat(m);
        let tilt = (m * Vec3::Z).angle_between(Vec3::Z);
        let twist = (r.x_axis.y - r.y_axis.x).atan2(r.x_axis.x + r.y_axis.y);
        (tilt, twist.abs())
    };
    (away, tilt, twist)
}

/// Two arms carry a tray between them over a block, the tray kept level: the right hand pinned to
/// its place in the left hand's frame, the left within 5° of straight down. Every path, sampled
/// densely once timed, keeps the right hand within 2 mm and 0.02 rad of its place, the floors of
/// a constraint relative to another hand, and the left hand level, and is collision-free; the same
/// problems planned without the constraints pull the hands centimeters apart. A goal 1.5 mm off its
/// place plans, within the floor; one 4 mm off is an input error.
#[test]
fn two_hands_carry_a_tray_level_between_them() {
    let robot = common::two_pandas().unwrap();
    let bound = 5f32.to_radians();
    for device in devices(&robot) {
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let common::TrayCarries {
            start,
            place,
            goals,
        } = common::tray_carries(&device, &worlds).unwrap();
        assert!(goals.len() >= 4, "only {} goals in reach", goals.len());
        let carried = common::carried_level(place, bound).to_vec();
        let problems: Vec<PlanProblem> = goals
            .iter()
            .map(|g| PlanProblem {
                constraints: carried.clone(),
                ..PlanProblem::new(0, start.clone(), g.clone())
            })
            .collect();
        let o = PlanOptions::default();
        let result = batchplan::plan(&device, &worlds, &problems, &o).unwrap();
        assert_eq!(
            result.solved().count(),
            problems.len(),
            "{}: every carry should be planned",
            device.name()
        );
        for s in result.solved() {
            let q = samples(&robot, s.solution);
            let (mut away, mut turned, mut tilted) = (0.0f32, 0.0f32, 0.0f32);
            for q in &q {
                let (a, tilt, twist) = off_its_place(&robot, q, place);
                away = away.max(a);
                turned = turned.max(tilt.max(twist));
                tilted = tilted.max(tilt_and_twist(&robot, q, 0, common::upright(0.0)).0);
            }
            assert!(
                away <= 2e-3 + 1e-5 && turned <= 0.02 + 1e-4 && tilted <= bound + 1e-4,
                "{}: carry {} moves the right hand {:.2} mm and {:.4} rad off its place, and \
                 tilts the left {:.2}°",
                device.name(),
                s.index,
                away * 1e3,
                turned,
                tilted.to_degrees()
            );
            assert!(
                free(&device, &worlds, &q),
                "{}: carry {} collides",
                device.name(),
                s.index
            );
        }
        let unconstrained: Vec<PlanProblem> = problems
            .iter()
            .map(|p| PlanProblem {
                constraints: vec![],
                ..p.clone()
            })
            .collect();
        let loose = batchplan::plan(&device, &worlds, &unconstrained, &o).unwrap();
        let apart = loose
            .solved()
            .filter(|s| {
                let q = samples(&robot, s.solution);
                q.iter().any(|q| off_its_place(&robot, q, place).0 > 0.01)
            })
            .count();
        assert!(
            2 * apart > problems.len(),
            "{}: most unconstrained carries should pull the hands apart, not {apart} of {}",
            device.name(),
            problems.len()
        );
        // The goal IK solves with the right hand off its place: within the floor, and past it.
        let carry = |off: f32| {
            let shifted = Pose::new(place.position + Vec3::X * off, place.rotation);
            let at = common::tray(Vec3::new(0.62, 0.0, 0.32), 0.0);
            let goal = common::tray_grip(&device, &worlds, at, Some(shifted), &start)
                .unwrap()
                .expect("the shifted goal is in reach");
            let problem = PlanProblem {
                constraints: carried.clone(),
                ..PlanProblem::new(0, start.clone(), goal)
            };
            batchplan::plan(&device, &worlds, &[problem], &o).map(|r| r.best(0).is_some())
        };
        assert!(
            carry(1.5e-3).unwrap(),
            "{}: 1.5 mm off its place plans",
            device.name()
        );
        match carry(4e-3) {
            Err(Error::Input(m)) => assert!(m.contains("the goal breaks its constraints"), "{m}"),
            other => panic!("4 mm off its place should be an input error, got {other:?}"),
        }
    }
}

/// A start or goal that breaks its constraints, a hand the robot does not have, a constraint
/// relative to its own hand or a missing one, and a negative bound are input errors that say what
/// was wrong; so is an end of a search that breaks them.
#[test]
fn ends_and_bounds_that_cannot_be_kept_are_input_errors() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[batchplan::World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    // The last joint turns the hand about its own z axis: no tilt.
    let mut goal = start.clone();
    goal[6] -= 0.5;
    let level = robot.ee_pose(&start, 0);
    let tipped = Pose::new(level.position, level.rotation * Quat::from_rotation_x(0.3));
    let o = PlanOptions::default();
    let plan = |constraint: HandConstraint| {
        let problem = PlanProblem {
            constraints: vec![constraint],
            ..PlanProblem::new(0, start.clone(), goal.clone())
        };
        batchplan::plan(&device, &worlds, &[problem], &o).map(|_| ())
    };
    let message = |r: Result<(), Error>| match r {
        Err(Error::Input(m)) => m,
        other => panic!("expected an input error, got {other:?}"),
    };
    let tilt = |pose| HandConstraint {
        tilt: 0.1,
        ..HandConstraint::new(0, pose)
    };
    assert!(plan(tilt(level)).is_ok(), "both ends keep a level hand");
    assert!(message(plan(tilt(tipped))).contains("the start breaks its constraints"));
    let twisted = HandConstraint {
        tilt: 0.1,
        twist: 0.2,
        ..HandConstraint::new(0, level)
    };
    assert!(message(plan(twisted)).contains("the goal breaks its constraints"));
    let free_tilt = HandConstraint {
        twist: 0.2,
        ..HandConstraint::new(0, level)
    };
    assert!(message(plan(free_tilt)).contains("its tilt must be bounded"));
    assert!(message(plan(HandConstraint::new(1, level))).contains("names hand 1 of 1"));
    for other in [0, 1] {
        let relative = HandConstraint {
            relative_to: Some(other),
            ..HandConstraint::new(0, level)
        };
        assert!(message(plan(relative)).contains(&format!("is relative to hand {other}")));
    }
    let negative = HandConstraint {
        tilt: -0.1,
        ..HandConstraint::new(0, level)
    };
    assert!(message(plan(negative)).contains("must not be negative"));
    let search = RrtProblem {
        world: 0,
        start: start.clone(),
        goals: vec![goal.clone()],
        hold: vec![],
        constraints: vec![tilt(tipped)],
    };
    let r = batchplan::rrt::connect(&device, &worlds, &[search], &RrtOptions::default());
    assert!(message(r.map(|_| ())).contains("an end breaks its constraints"));
}

/// A constrained problem plans the same alone, in a batch with others, and on every device. The
/// gripper stays upright inside the zone grown by 5 cm, which leaves the arm little room past the
/// pillar, so the plans come from sampling.
#[test]
fn constrained_plans_do_not_depend_on_the_batch_or_device() {
    let robot = common::panda().unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let worlds = cpu.upload(&[common::pillar_scene()]).unwrap();
    let (start, goals) = common::upright_reaches(&cpu, &worlds, 12, &mut Rng::new(21)).unwrap();
    let middle = 0.5 * (common::ZONE_MIN + common::ZONE_MAX);
    let upright = HandConstraint {
        position: 0.5 * (common::ZONE_MAX - common::ZONE_MIN) + Vec3::splat(0.05),
        tilt: 0.1,
        ..HandConstraint::new(0, Pose::new(middle, common::upright(0.0)))
    };
    let problems: Vec<PlanProblem> = goals
        .iter()
        .enumerate()
        .map(|(i, g)| PlanProblem {
            // Every other problem unconstrained, so constrained ones share batches with others.
            constraints: if i % 2 == 0 { vec![upright] } else { vec![] },
            ..PlanProblem::new(0, start.clone(), g.clone())
        })
        .collect();
    let o = PlanOptions {
        rrt: RrtOptions {
            max_iterations: 1000,
            ..PlanOptions::default().rrt
        },
        ..PlanOptions::default()
    };
    let mut runs = vec![];
    for device in devices(&robot) {
        let worlds = device.upload(&[common::pillar_scene()]).unwrap();
        let batch = batchplan::plan(&device, &worlds, &problems, &o).unwrap();
        let alone: Vec<Option<Vec<f32>>> = problems
            .iter()
            .map(|p| {
                let r = batchplan::plan(&device, &worlds, std::slice::from_ref(p), &o).unwrap();
                r.best(0).map(<[f32]>::to_vec)
            })
            .collect();
        let batched: Vec<Option<Vec<f32>>> = (0..problems.len())
            .map(|i| batch.best(i).map(<[f32]>::to_vec))
            .collect();
        assert_eq!(
            batched,
            alone,
            "{}: a batch plans differently",
            device.name()
        );
        assert!(
            batched.iter().step_by(2).filter(|p| p.is_some()).count() >= 2,
            "too few constrained plans"
        );
        runs.push(batched);
    }
    if let [cpu, gpu] = &runs[..] {
        assert_eq!(cpu, gpu, "the devices plan differently");
    }
    // The constrained plans come from sampling, which the batch could disturb: another seed plans
    // some of them differently.
    let reseeded = PlanOptions {
        rrt: RrtOptions {
            rng_seed: 99,
            ..o.rrt
        },
        ..o
    };
    let other = batchplan::plan(&cpu, &worlds, &problems, &reseeded).unwrap();
    let changed = (0..problems.len())
        .step_by(2)
        .filter(|&i| other.best(i).map(<[f32]>::to_vec) != runs[0][i])
        .count();
    assert!(changed > 0, "no constrained plan depends on its samples");
}
