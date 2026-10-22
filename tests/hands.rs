//! Robots with several hands: two arms assembled into one robot, IK toward both hands' targets at
//! once, and an arm without a target held where it is.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::{
    CollisionWeights, Device, IkOptions, IkProblem, Obstacle, Part, PlanOptions, PlanProblem, Pose,
    Robot, RobotOptions, RrtOptions, RrtProblem, Trajectory, World, Worlds,
};
use glam::{Quat, Vec3};

const UR5E: &str = "ur5e/ur_description/urdf/ur5e.urdf";

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot).unwrap()];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

/// Two UR5e arms a metre apart, facing the same way.
fn two_arms() -> Robot {
    let part = |prefix: &str, y: f32| Part {
        path: common::asset(UR5E).into(),
        prefix: prefix.into(),
        base: Pose::new(Vec3::new(0.0, y, 0.0), Quat::IDENTITY),
    };
    let options = RobotOptions {
        ee_links: vec!["left_tool0".into(), "right_tool0".into()],
        ..Default::default()
    };
    Robot::assemble(&[part("left_", 0.5), part("right_", -0.5)], &options).unwrap()
}

/// Collision-free configurations of both arms near `robot`'s default.
fn configurations(robot: &Robot, cpu: &Device, worlds: &Worlds, count: usize) -> Vec<Vec<f32>> {
    let n = robot.dof();
    let mut rng = Rng::new(3);
    let mut out = vec![];
    while out.len() < count {
        let q: Vec<f32> = (0..n)
            .map(|j| {
                (robot.default_q()[j] + rng.range(-1.0, 1.0))
                    .clamp(robot.lower()[j], robot.upper()[j])
            })
            .collect();
        if cpu
            .evaluate(worlds, &[0], &q, &CollisionWeights::NONE)
            .unwrap()
            .collision_free(0)
        {
            out.push(q);
        }
    }
    out
}

#[test]
fn two_arms_assemble_into_one_robot() {
    let robot = two_arms();
    let single = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    assert_eq!((robot.dof(), robot.hands()), (12, 2));
    assert!(
        robot
            .joint_names()
            .iter()
            .all(|n| n.starts_with("left_") || n.starts_with("right_"))
    );
    let (left, right) = (robot.hand_joints(0), robot.hand_joints(1));
    assert_eq!((left.len(), right.len()), (6, 6));
    assert!(
        left.iter().all(|j| !right.contains(j)),
        "the arms share joints"
    );
    // Each hand sits where the arm alone puts it, moved to its base.
    let q: Vec<f32> = (0..12).map(|j| 0.1 * j as f32 - 0.5).collect();
    for (hand, joints, y) in [(0, &left, 0.5), (1, &right, -0.5)] {
        let arm: Vec<f32> = joints.iter().map(|&j| q[j]).collect();
        let alone = single.ee_pose(&arm, 0);
        let assembled = robot.ee_pose(&q, hand);
        assert!((assembled.position - alone.position - Vec3::new(0.0, y, 0.0)).length() < 1e-5);
        assert!(
            assembled.rotation.dot(alone.rotation).abs() > 1.0 - 1e-6,
            "hand {hand} turned"
        );
    }
    assert_eq!(robot.parts().len(), 2);
}

#[test]
fn ik_reaches_both_hands_targets_at_once() {
    let robot = two_arms();
    let cpu = Device::cpu(&robot).unwrap();
    let scene = [World::default()];
    let on_cpu = cpu.upload(&scene).unwrap();
    // Targets both arms reach at once, without touching each other.
    let problems: Vec<IkProblem> = configurations(&robot, &cpu, &on_cpu, 32)
        .iter()
        .map(|q| IkProblem {
            world: 0,
            targets: vec![Some(robot.ee_pose(q, 0)), Some(robot.ee_pose(q, 1))],
            seed: None,
        })
        .collect();
    let o = IkOptions::default();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let ik = batchplan::solve_ik(&d, &worlds, &problems, &o).unwrap();
        let solved: Vec<_> = ik.solved().collect();
        eprintln!(
            "{}: {} of {} two-hand problems",
            d.name(),
            solved.len(),
            problems.len()
        );
        assert!(
            solved.len() >= 28,
            "{}: solved {} of {}",
            d.name(),
            solved.len(),
            problems.len()
        );
        for s in &solved {
            for (hand, target) in s.problem.targets.iter().enumerate() {
                let reached = robot.ee_pose(s.solution, hand);
                let target = target.unwrap();
                assert!(
                    (reached.position - target.position).length() < o.position_tolerance,
                    "{}: hand {hand} misses",
                    d.name()
                );
            }
            let e = cpu
                .evaluate(&on_cpu, &[0], s.solution, &CollisionWeights::NONE)
                .unwrap();
            assert!(e.collision_free(0), "{}: the arms collide", d.name());
        }
    }
}

#[test]
fn a_hand_without_a_target_holds_its_arm() {
    let robot = two_arms();
    let cpu = Device::cpu(&robot).unwrap();
    let scene = [World::default()];
    let on_cpu = cpu.upload(&scene).unwrap();
    let qs = configurations(&robot, &cpu, &on_cpu, 16);
    let left = robot.hand_joints(0);
    // The left hand moves to where another configuration has it; the right arm waits.
    let problems: Vec<IkProblem> = qs
        .windows(2)
        .map(|w| IkProblem {
            world: 0,
            targets: vec![Some(robot.ee_pose(&w[1], 0)), None],
            seed: Some(w[0].clone()),
        })
        .collect();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let ik = batchplan::solve_ik(&d, &worlds, &problems, &IkOptions::default()).unwrap();
        assert!(
            ik.solved().count() >= 12,
            "{}: solved {} of {}",
            d.name(),
            ik.solved().count(),
            problems.len()
        );
        // Every seed, successful or not, keeps the right arm exactly where the seed has it.
        for (p, problem) in problems.iter().enumerate() {
            let seed = problem.seed.as_ref().unwrap();
            for s in 0..ik.seeds {
                let q = ik.solution(p * ik.seeds + s);
                for j in (0..robot.dof()).filter(|j| !left.contains(j)) {
                    assert_eq!(
                        q[j],
                        seed[j],
                        "{}: problem {p} seed {s} moves held joint {j}",
                        d.name()
                    );
                }
            }
        }
    }
}

/// A wall in front of the left arm, and problems that carry the left hand from one side of it to
/// the other while the right arm stays where it starts.
fn over_the_wall(robot: &Robot, cpu: &Device) -> (World, Vec<PlanProblem>) {
    let n = robot.dof();
    let left = robot.hand_joints(0);
    let right: Vec<usize> = (0..n).filter(|j| !left.contains(j)).collect();
    let wall = World {
        obstacles: vec![Obstacle::cuboid(
            Vec3::new(0.47, 0.5, 0.25),
            Vec3::new(0.02, 0.35, 0.25),
        )],
    };
    let worlds = cpu.upload(std::slice::from_ref(&wall)).unwrap();
    let mut rng = Rng::new(9);
    let (mut near, mut far) = (vec![], vec![]);
    while near.len() < 24 || far.len() < 24 {
        let q: Vec<f32> = (0..n)
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect();
        let hand = robot.ee_pose(&q, 0).position;
        // Below the wall's top and across from its middle, on either side.
        let (side_near, side_far) = (
            hand.x > 0.15 && hand.x < 0.35,
            hand.x > 0.6 && hand.x < 0.85,
        );
        if hand.z > 0.05 && hand.z < 0.45 && (hand.y - 0.5).abs() < 0.3 && (side_near || side_far) {
            let free = cpu
                .evaluate(&worlds, &[0], &q, &CollisionWeights::NONE)
                .unwrap()
                .collision_free(0);
            if free && side_near && near.len() < 24 {
                near.push(q);
            } else if free && side_far && far.len() < 24 {
                far.push(q);
            }
        }
    }
    let problems = near
        .iter()
        .zip(&far)
        .map(|(a, b)| {
            let goal = (0..n)
                .map(|j| if left.contains(&j) { b[j] } else { a[j] })
                .collect();
            PlanProblem {
                world: 0,
                start: a.clone(),
                goal,
                start_motion: None,
                hold: right.clone(),
            }
        })
        .collect();
    (wall, problems)
}

/// RRT-Connect samples, extends and bends only the joints it may: every waypoint of every path over
/// the wall keeps the right arm where it starts.
#[test]
fn searches_hold_an_arm_still() {
    let robot = two_arms();
    let cpu = Device::cpu(&robot).unwrap();
    let (wall, problems) = over_the_wall(&robot, &cpu);
    let worlds = cpu.upload(&[wall]).unwrap();
    let searches: Vec<RrtProblem> = problems
        .iter()
        .map(|p| RrtProblem {
            world: 0,
            start: p.start.clone(),
            goals: vec![p.goal.clone()],
            hold: p.hold.clone(),
        })
        .collect();
    let found = batchplan::rrt::connect(&cpu, &worlds, &searches, &RrtOptions::default()).unwrap();
    let n = robot.dof();
    let paths: Vec<&Vec<f32>> = found.paths.iter().flatten().collect();
    let searched = paths.iter().filter(|p| p.len() > 2 * n).count();
    eprintln!(
        "{} of {} found, {searched} with waypoints",
        paths.len(),
        problems.len()
    );
    assert!(
        paths.len() >= 6 && searched >= 5,
        "the wall should need searching: {searched} of {}",
        paths.len()
    );
    for (path, problem) in found.paths.iter().zip(&problems) {
        for q in path.iter().flat_map(|p| p.chunks(n)) {
            assert!(
                problem.hold.iter().all(|&j| q[j] == problem.start[j]),
                "a waypoint moves the held arm"
            );
        }
    }
}

/// Plans move only what they may: over the wall, every valid seed of every plan (searched or
/// optimized) and every timed sample keeps the right arm where it starts.
#[test]
fn plans_hold_an_arm_still() {
    let robot = two_arms();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let (wall, problems) = over_the_wall(&robot, &cpu);
    let scene = [wall];
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let o = PlanOptions {
            seeds: 2,
            ..Default::default()
        };
        let result = batchplan::plan(&d, &worlds, &problems, &o).unwrap();
        let solved = result.solved().count();
        eprintln!("{}: planned {solved} of {}", d.name(), problems.len());
        assert!(
            solved >= 20,
            "{}: planned {solved} of {}",
            d.name(),
            problems.len()
        );
        let mut optimized = 0;
        for (p, problem) in problems.iter().enumerate() {
            for s in (0..result.seeds).filter(|&s| result.valid[p * result.seeds + s]) {
                optimized += usize::from(s > 0);
                for q in result.paths.path(p * result.seeds + s).chunks(n) {
                    assert!(
                        problem.hold.iter().all(|&j| q[j] == problem.start[j]),
                        "{}: seed {s} moves the held arm",
                        d.name()
                    );
                }
            }
            if let Some(path) = result.best(p) {
                // Blending equal control points is exact up to rounding.
                for q in Trajectory::new(&robot, path, 1.0)
                    .unwrap()
                    .sample(100.0)
                    .unwrap()
                    .positions
                    .chunks(n)
                {
                    let moved = problem
                        .hold
                        .iter()
                        .map(|&j| (q[j] - problem.start[j]).abs())
                        .fold(0.0, f32::max);
                    assert!(
                        moved < 1e-5,
                        "{}: the timed path moves the held arm {moved} rad",
                        d.name()
                    );
                }
            }
        }
        assert!(
            optimized >= 10,
            "{}: only {optimized} optimized seeds are valid",
            d.name()
        );
    }
}
