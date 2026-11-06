//! Why a request failed: plans say which end collides and with what, or whether the search, the
//! spline or the time ran out; IK tells a pose out of reach from one reached only in collision.
//! GPU parts are skipped when no adapter is available unless `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use std::time::Duration;

use batchplan::rng::Rng;
use batchplan::{
    CollisionWeights, Contact, Device, IkFailure, IkOptions, IkProblem, Obstacle, PlanFailure,
    PlanOptions, PlanProblem, Pose, Robot, RobotOptions, RrtOptions, StartMotion, World,
};
use glam::Vec3;

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

/// A box around the hand at configuration `q`.
fn around_hand(robot: &Robot, q: &[f32]) -> Obstacle {
    Obstacle::cuboid(robot.ee_pose(q, 0).position, Vec3::splat(0.05))
}

/// Whether `contacts` all collide and one of them is the hand's `panda_hand` link in `obstacle`.
fn hand_in(contacts: &[Contact], obstacle: usize) -> bool {
    contacts.iter().all(|c| c.distance() < 0.0)
        && contacts.iter().any(|c| {
            matches!(c, Contact::World { link, obstacle: o, .. } if link == "panda_hand" && *o == obstacle)
        })
}

#[test]
fn plans_say_which_end_collides_and_with_what() {
    let robot = panda();
    let start = robot.default_q().to_vec();
    let mut goal = start.clone();
    goal[0] += 1.2;
    let scene = vec![
        World::new(vec![around_hand(&robot, &start)]),
        World::new(vec![around_hand(&robot, &goal)]),
        World::default(),
    ];
    let problems = [
        PlanProblem::new(0, start.clone(), goal.clone()),
        PlanProblem::new(1, start.clone(), goal.clone()),
        PlanProblem::new(2, start, goal),
    ];
    let mut failures = None;
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let result = batchplan::plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        let name = d.name();
        assert!(
            matches!(&result.failures[0], Some(PlanFailure::StartCollides { contacts }) if hand_in(contacts, 0)),
            "{name}: {:?}",
            result.failures[0]
        );
        assert!(
            matches!(&result.failures[1], Some(PlanFailure::GoalCollides { contacts }) if hand_in(contacts, 0)),
            "{name}: {:?}",
            result.failures[1]
        );
        assert!(result.best(2).is_some(), "{name}: the free reach is solved");
        assert_eq!(result.failures[2], None, "{name}");
        match &failures {
            None => failures = Some(result.failures),
            Some(first) => assert_eq!(*first, result.failures, "{name} says otherwise"),
        }
    }
}

#[test]
fn plans_say_whether_the_search_the_spline_or_the_time_ran_out() {
    let robot = panda();
    let (scene, problems) = common::blocked_reaches(&robot);
    assert!(problems.len() > 20, "{} blocked reaches", problems.len());
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&scene).unwrap();
    // A ball across every straight line leaves a search that may not sample nothing to find.
    let no_samples = PlanOptions {
        rrt: RrtOptions {
            max_iterations: 0,
            ..PlanOptions::default().rrt
        },
        ..PlanOptions::default()
    };
    let result = batchplan::plan(&device, &worlds, &problems, &no_samples).unwrap();
    let failed: Vec<&PlanFailure> = result.failures.iter().flatten().collect();
    assert!(
        !failed.is_empty() && failed.iter().all(|f| **f == PlanFailure::NoPath),
        "{failed:?}"
    );
    assert!(
        (0..problems.len()).all(|p| result.best(p).is_some() == result.failures[p].is_none()),
        "a failure for every problem without a path, and none for the rest"
    );
    // No time at all: nothing is searched.
    let no_time = PlanOptions {
        time_budget: Some(Duration::ZERO),
        ..PlanOptions::default()
    };
    let result = batchplan::plan(&device, &worlds, &problems, &no_time).unwrap();
    assert!(
        result
            .failures
            .iter()
            .all(|f| *f == Some(PlanFailure::OutOfTime)),
        "{:?}",
        result.failures
    );
    // Moving fast toward a ball it cannot brake short of, on a Panda slowed to a fifth of its
    // acceleration: the way back is free, but no spline that continues the motion misses the ball.
    let slowed = Robot::load(
        common::asset("franka/franka_panda.urdf"),
        &RobotOptions {
            acceleration_scale: 0.2,
            jerk_scale: 0.2,
            ..common::panda_options()
        },
    )
    .unwrap();
    let device = Device::cpu(&slowed).unwrap();
    let start = slowed.default_q().to_vec();
    let mut ahead = start.clone();
    ahead[0] += 1.0;
    let ball = Obstacle::sphere(
        slowed.link_pose(&ahead, "panda_hand").unwrap().position,
        0.04,
    );
    let worlds = device.upload(&[World::new(vec![ball])]).unwrap();
    let free = |q: &[f32]| {
        device
            .evaluate(&worlds, &[0], q, &CollisionWeights::NONE)
            .unwrap()
            .collision_free(0)
    };
    let mut goal = start.clone();
    goal[0] -= 1.0;
    assert!(
        free(&start) && free(&goal) && !free(&ahead),
        "the ball is ahead"
    );
    let mut velocity = vec![0.0; slowed.dof()];
    velocity[0] = 0.9 * slowed.max_velocity()[0];
    let moving = PlanProblem {
        start_motion: Some(StartMotion {
            velocity,
            acceleration: vec![0.0; slowed.dof()],
        }),
        ..PlanProblem::new(0, start, goal)
    };
    let result = batchplan::plan(&device, &worlds, &[moving], &PlanOptions::default()).unwrap();
    assert_eq!(result.failures[0], Some(PlanFailure::NoSpline));
}

#[test]
fn ik_tells_a_pose_out_of_reach_from_one_reached_only_in_collision() {
    let robot = panda();
    let reachable = robot.ee_pose(robot.default_q(), 0);
    let far = Pose::new(Vec3::new(2.0, 0.0, 0.5), reachable.rotation);
    let mut rng = Rng::new(9);
    // Balls in the hand's way at random reachable poses: some poses collide deep enough that the
    // push away from obstacles holds every seed off them.
    let mut scene = vec![
        World::default(),
        World::new(vec![around_hand(&robot, robot.default_q())]),
    ];
    let mut problems = vec![
        IkProblem::new(0, reachable),
        IkProblem::new(0, far),
        IkProblem::new(1, reachable),
    ];
    for _ in 0..200 {
        let world = common::tabletop(&mut rng);
        let target = common::grasp_target(&world, &mut rng);
        let ball = Obstacle::sphere(target.position + Vec3::Z * rng.range(0.0, 0.15), 0.06);
        problems.push(IkProblem::new(scene.len() as u32, target));
        scene.push(World::new(vec![ball]));
    }
    let o = IkOptions::default();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let result = batchplan::solve_ik(&d, &worlds, &problems, &o).unwrap();
        let failures = batchplan::ik::failures(&d, &worlds, &result, &o).unwrap();
        let name = d.name();
        assert_eq!(failures[0], None, "{name}: solved");
        assert_eq!(failures[1], Some(IkFailure::Unreachable), "{name}");
        assert!(
            matches!(&failures[2], Some(IkFailure::Collides { contacts }) if hand_in(contacts, 0)),
            "{name}: {:?}",
            failures[2]
        );
        let tolerances = |p: usize| {
            (p * o.seeds..(p + 1) * o.seeds).any(|k| {
                result.position_error[k] < o.position_tolerance
                    && result.rotation_error[k] < o.rotation_tolerance
            })
        };
        // Reached only once the push is off: solving again is what tells.
        let held_off = (3..problems.len())
            .filter(|&p| !tolerances(p) && matches!(failures[p], Some(IkFailure::Collides { .. })))
            .count();
        assert!(
            held_off > 5,
            "{name}: {held_off} poses held off by the push"
        );
        assert!(
            (0..problems.len()).all(|p| result.best(p).is_some() == failures[p].is_none()),
            "{name}: a failure for every unsolved problem, and none for the rest"
        );
    }
}
