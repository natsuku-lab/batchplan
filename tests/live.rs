//! Live motion through its interface: a follower replans around a box moved into its path and
//! takes over from the state the robot is in then, stops on its path when there is no time or its
//! replan collides, and says so when no stop is short enough; replans keep the trajectory's
//! constraints, and stop where they cannot start inside them; servo steps slow and halt before a
//! wall without entering it, reach a free target and brake joints before their ends, within the
//! limits.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::glam::{Quat, Vec3};
use batchplan::rng::Rng;
use batchplan::{
    CollisionWeights, Device, Error, FollowStatus, Follower, FollowerOptions, HandConstraint,
    JointState, Obstacle, PlanOptions, PlanProblem, Pose, Robot, RobotOptions, RrtOptions,
    ServoOptions, ServoStatus, Trajectory, World, Worlds,
};

const DT: f32 = 0.02;

/// A sweep across the front of the Panda.
const FROM: [f32; 7] = [-1.2, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8];
const TO: [f32; 7] = [1.2, -0.4, 0.3, -2.0, 0.3, 1.9, 0.5];

/// The sweep at half speed: fast enough that a box ahead needs a replan, slow enough that a way
/// around exists (at full speed the first joint turns at 2 rad/s, and stopping is all it can do).
fn sweep(robot: &Robot, device: &Device) -> Trajectory {
    sweep_at(robot, device, 0.5)
}

fn sweep_at(robot: &Robot, device: &Device, speed: f32) -> Trajectory {
    let worlds = device.upload(&[World::default()]).unwrap();
    let problem = PlanProblem::new(0, FROM.to_vec(), TO.to_vec());
    let plans = batchplan::plan(device, &worlds, &[problem], &PlanOptions::default()).unwrap();
    Trajectory::try_new(robot, plans.best(0).expect("a free sweep plans"), speed).unwrap()
}

fn state_at(trajectory: &Trajectory, t: f32) -> JointState {
    let mut state = JointState::new(trajectory.dof);
    trajectory.at(t, &mut state);
    state
}

/// A box over where the hand is at time `t` of `trajectory`.
fn box_at(robot: &Robot, trajectory: &Trajectory, t: f32) -> Obstacle {
    let mut state = JointState::new(robot.dof());
    trajectory.at(t, &mut state);
    Obstacle::cuboid(
        robot.ee_pose(&state.position, 0).position,
        Vec3::splat(0.05),
    )
}

fn clear(device: &Device, worlds: &Worlds, q: &[f32]) -> bool {
    let e = device
        .evaluate(worlds, &[0], q, &CollisionWeights::NONE)
        .unwrap();
    e.collision_free(0)
}

#[test]
fn a_box_in_the_path_is_replanned_around_from_the_predicted_state() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let first = sweep(&robot, &device);
    let duration = first.duration();
    let mut follower =
        Follower::try_new(&robot, 0, first.clone(), 0.0, FollowerOptions::default()).unwrap();
    let mut worlds = device.upload(&[World::default()]).unwrap();
    let inserted = 0.3 * duration;
    let (mut replanned, mut statuses) = (None, vec![]);
    let mut t = 0.0;
    while t < 4.0 * duration {
        if (t - inserted).abs() < DT / 2.0 {
            worlds = device
                .upload(&[World::new(vec![box_at(&robot, &first, 0.6 * duration)])])
                .unwrap();
            assert!(
                !clear(&device, &worlds, &{
                    let mut s = JointState::new(robot.dof());
                    first.at(0.6 * duration, &mut s);
                    s.position
                }),
                "the box blocks the old path"
            );
        }
        let (before, began) = (follower.trajectory().clone(), follower.began());
        let tick = follower.tick(&device, &worlds, t).unwrap();
        assert!(
            clear(&device, &worlds, &tick.state.position),
            "at {t:.2} s the robot collides ({:?})",
            tick.status
        );
        if let FollowStatus::Replanned { at } = tick.status {
            replanned = Some((before, began, at));
        }
        statuses.push(tick.status.clone());
        if matches!(tick.status, FollowStatus::Arrived | FollowStatus::Blocked) {
            break;
        }
        t += DT;
    }
    assert_eq!(
        statuses.last(),
        Some(&FollowStatus::Arrived),
        "{statuses:?}"
    );
    let (old, began, at) = replanned.expect("the box made it replan");
    // The replan takes over from the state the old trajectory reaches when it does.
    let (new, new_began) = (follower.trajectory(), follower.began());
    assert!((new_began - at).abs() < 1e-6, "{new_began} against {at}");
    let n = robot.dof();
    let (mut was, mut is) = (JointState::new(n), JointState::new(n));
    old.at(at - began, &mut was);
    new.at(0.0, &mut is);
    for j in 0..n {
        assert!((was.position[j] - is.position[j]).abs() < 1e-4, "joint {j}");
        assert!((was.velocity[j] - is.velocity[j]).abs() < 1e-2, "joint {j}");
    }
}

#[test]
fn with_no_time_to_replan_the_robot_stops_before_the_box() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let first = sweep(&robot, &device);
    let duration = first.duration();
    // A replan would take longer than the motion has left.
    let o = FollowerOptions {
        budget: duration,
        ..Default::default()
    };
    let mut follower = Follower::try_new(&robot, 0, first.clone(), 0.0, o).unwrap();
    let mut worlds = device.upload(&[World::default()]).unwrap();
    let inserted = 0.2 * duration;
    let (mut t, mut statuses, mut stop) = (0.0, vec![], None);
    while t < 4.0 * duration {
        if (t - inserted).abs() < DT / 2.0 {
            worlds = device
                .upload(&[World::new(vec![box_at(&robot, &first, 0.8 * duration)])])
                .unwrap();
        }
        let tick = follower.tick(&device, &worlds, t).unwrap();
        assert!(
            clear(&device, &worlds, &tick.state.position),
            "at {t:.2} s the robot collides ({:?})",
            tick.status
        );
        if tick.status == FollowStatus::Stopping && stop.is_none() {
            stop = Some((follower.trajectory().clone(), tick.state.clone()));
        }
        statuses.push(tick.status.clone());
        if matches!(tick.status, FollowStatus::Arrived | FollowStatus::Blocked) {
            break;
        }
        t += DT;
    }
    assert_eq!(
        statuses.last(),
        Some(&FollowStatus::Blocked),
        "{statuses:?}"
    );
    let (stop, from) = stop.expect("it stopped");
    stop.check_from(&robot, &from).unwrap();
}

/// Servo steps from `start` toward `target` until it arrives or rests, at most `steps` of them:
/// the states commanded, and each step's status.
fn servo_until(
    device: &Device,
    worlds: &Worlds,
    start: &[f32],
    target: Pose,
    steps: usize,
) -> (Vec<JointState>, Vec<ServoStatus>) {
    let o = ServoOptions::default();
    let mut state = JointState {
        position: start.to_vec(),
        ..JointState::new(start.len())
    };
    let (mut states, mut statuses) = (vec![state.clone()], vec![]);
    let mut still = 0;
    for _ in 0..steps {
        let step = batchplan::servo(device, worlds, 0, &state, target, &o).unwrap();
        state = step.state;
        states.push(state.clone());
        statuses.push(step.status.clone());
        let moving = state.velocity.iter().any(|v| v.abs() > 1e-4);
        still = if moving { 0 } else { still + 1 };
        if step.status == ServoStatus::Arrived || still > 50 {
            break;
        }
    }
    (states, statuses)
}

/// Whether every step keeps the joints' ranges and their velocity, acceleration and jerk limits,
/// its acceleration the one its change of velocity takes.
fn within_limits(robot: &Robot, states: &[JointState], dt: f32) {
    for (k, w) in states.windows(2).enumerate() {
        for j in 0..robot.dof() {
            let range = robot.lower()[j]..=robot.upper()[j];
            assert!(range.contains(&w[1].position[j]), "step {k}: joint {j}");
            assert!(
                w[1].velocity[j].abs() <= robot.max_velocity()[j] * 1.0001,
                "step {k}"
            );
            let change = (w[1].velocity[j] - w[0].velocity[j]) / dt;
            assert!(
                (change - w[1].acceleration[j]).abs() <= 1e-3 * robot.max_acceleration()[j],
                "step {k}: joint {j}'s velocity changes at {change}, its acceleration {}",
                w[1].acceleration[j]
            );
            assert!(
                w[1].acceleration[j].abs() <= robot.max_acceleration()[j] * 1.0001,
                "step {k}"
            );
            let jerk = (w[1].acceleration[j] - w[0].acceleration[j]) / dt;
            assert!(
                jerk.abs() <= robot.max_jerk()[j] * 1.0001,
                "step {k}: jerk {jerk}"
            );
        }
    }
}

/// Servoing the hand 40 cm along x toward a wall 25 cm ahead of it, clear of the rest of the arm
/// (the wrist reaches about 12 cm ahead of the grasp point here).
fn servo_toward_a_wall(robot: &Robot) {
    let device = Device::cpu(robot).unwrap();
    let start = robot.default_q().to_vec();
    let hand = robot.ee_pose(&start, 0);
    let target = Pose::new(hand.position + Vec3::new(0.4, 0.0, 0.0), hand.rotation);
    let wall = Obstacle::cuboid(
        hand.position + Vec3::new(0.25, 0.0, 0.0),
        Vec3::new(0.01, 0.12, 0.12),
    );
    let worlds = device.upload(&[World::new(vec![wall])]).unwrap();
    let e = device
        .evaluate(&worlds, &[0], &start, &CollisionWeights::NONE)
        .unwrap();
    assert!(
        e.world_clearance[0] > 0.05,
        "the arm starts {} m from the wall",
        e.world_clearance[0]
    );
    let (states, statuses) = servo_until(&device, &worlds, &start, target, 3000);
    let status = statuses.last().unwrap();
    assert!(
        matches!(status, ServoStatus::Halted { .. }),
        "it ends {status:?}"
    );
    // A wall that stays put keeps it from speeding up again once it has halted.
    let halted = statuses
        .iter()
        .position(|s| matches!(s, ServoStatus::Halted { .. }))
        .unwrap();
    assert!(
        !statuses[halted..].contains(&ServoStatus::Normal),
        "{:?}",
        &statuses[halted..]
    );
    for (k, s) in states.iter().enumerate() {
        assert!(clear(&device, &worlds, &s.position), "step {k} collides");
    }
    let reached = robot.ee_pose(&states.last().unwrap().position, 0).position;
    assert!(
        reached.x > hand.position.x + 0.03,
        "it moved toward the wall: {reached}"
    );
    within_limits(robot, &states, ServoOptions::default().dt);
}

#[test]
fn servo_slows_and_halts_before_a_wall_without_entering_it() {
    servo_toward_a_wall(&common::panda().unwrap());
}

#[test]
fn servo_brakes_in_time_on_a_robot_that_brakes_slowly() {
    // A tenth of the acceleration: braking from full speed takes centimeters, not millimeters.
    let options = RobotOptions {
        acceleration_scale: 0.1,
        ..common::panda_options()
    };
    let robot = Robot::load(common::asset("franka/franka_panda.urdf"), &options).unwrap();
    servo_toward_a_wall(&robot);
}

#[test]
fn servo_reaches_a_free_target_within_the_limits() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    let hand = robot.ee_pose(&start, 0);
    let target = Pose::new(
        hand.position + Vec3::new(0.06, -0.05, -0.04),
        Quat::from_rotation_z(0.2) * hand.rotation,
    );
    let (states, statuses) = servo_until(&device, &worlds, &start, target, 4000);
    assert_eq!(statuses.last(), Some(&ServoStatus::Arrived));
    let reached = robot.ee_pose(&states.last().unwrap().position, 0);
    assert!(reached.position.distance(target.position) < 1e-3);
    within_limits(&robot, &states, ServoOptions::default().dt);
}

#[test]
fn servo_turns_the_shorter_way_whichever_sign_the_target_quaternion_has() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    let hand = robot.ee_pose(&start, 0);
    let turn = Quat::from_rotation_z(0.2) * hand.rotation;
    // The same rotation as `turn`, written as the quaternion of the other sign.
    let target = Pose::new(hand.position, -turn);
    assert!(
        (target.rotation * hand.rotation.inverse()).w < 0.0,
        "the target's quaternion is on the far side of the hand's"
    );
    let (states, statuses) = servo_until(&device, &worlds, &start, target, 4000);
    assert_eq!(statuses.last(), Some(&ServoStatus::Arrived));
    for (k, s) in states.iter().enumerate() {
        let angle = robot
            .ee_pose(&s.position, 0)
            .rotation
            .angle_between(hand.rotation);
        assert!(
            angle < 0.25,
            "step {k}: the hand turned {angle} rad for 0.2"
        );
    }
}

#[test]
fn servo_brakes_a_joint_before_the_end_of_its_range() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    let hand = robot.ee_pose(&start, 0);
    // Three radians about the hand's own axis: past the end of the last joint's range.
    let target = Pose::new(hand.position, hand.rotation * Quat::from_rotation_z(3.0));
    let (states, _) = servo_until(&device, &worlds, &start, target, 1500);
    let last = states.last().unwrap();
    let ends = (0..robot.dof()).filter(|&j| {
        robot.upper()[j] - last.position[j] < 0.05 || last.position[j] - robot.lower()[j] < 0.05
    });
    assert!(
        ends.count() > 0,
        "a joint reaches the end of its range: {last:?}"
    );
    within_limits(&robot, &states, ServoOptions::default().dt);
}

#[test]
fn a_wall_too_close_to_stop_for_is_reported_as_a_halt_into_it() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let free = device.upload(&[World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    let hand = robot.ee_pose(&start, 0);
    let target = Pose::new(hand.position + Vec3::new(0.4, 0.0, 0.0), hand.rotation);
    let o = ServoOptions::default();
    let mut state = JointState {
        position: start,
        ..JointState::new(robot.dof())
    };
    for _ in 0..100 {
        state = batchplan::servo(&device, &free, 0, &state, target, &o)
            .unwrap()
            .state;
    }
    // A thin wall just ahead of the arm, moving at full speed: 3 mm away, closer than it can stop.
    let wall_at = |gap: f32| {
        let at = robot.ee_pose(&state.position, 0).position + Vec3::new(gap, 0.0, 0.0);
        World::new(vec![Obstacle::cuboid(at, Vec3::new(0.005, 0.15, 0.15))])
    };
    // Exact clearances, as far out as the wall goes.
    let exact = CollisionWeights {
        margin: 0.5,
        ..CollisionWeights::NONE
    };
    let clearance = |world: World| {
        let worlds = device.upload(&[world]).unwrap();
        let e = device
            .evaluate(&worlds, &[0], &state.position, &exact)
            .unwrap();
        e.world_clearance[0]
    };
    let gap = 0.15 - (clearance(wall_at(0.15)) - 0.003);
    let worlds = device.upload(&[wall_at(gap)]).unwrap();
    let speed = state.velocity.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(speed > 0.05, "the arm moves at {speed} rad/s");
    let mut collided = false;
    for k in 0..200 {
        let step = batchplan::servo(&device, &worlds, 0, &state, target, &o).unwrap();
        if !clear(&device, &worlds, &step.state.position) {
            collided = true;
            assert!(
                matches!(step.status, ServoStatus::Halted { clearance } if clearance <= 0.0),
                "step {k} collides as {:?}",
                step.status
            );
        }
        state = step.state;
    }
    assert!(collided, "the wall is closer than the arm can stop");
    assert!(state.velocity.iter().all(|v| v.abs() < 1e-3), "it rests");
}

#[test]
fn a_box_ahead_of_the_takeover_stops_the_robot_before_it() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let first = sweep(&robot, &device);
    let duration = first.duration();
    // A long budget leaves 0.6 s between the replan and its takeover.
    let o = FollowerOptions {
        budget: 0.6,
        ..Default::default()
    };
    let mut follower = Follower::try_new(&robot, 0, first.clone(), 0.0, o).unwrap();
    let big = box_at(&robot, &first, 0.6 * duration);
    let mut worlds = device.upload(&[World::default()]).unwrap();
    let (mut t, mut takeover, mut statuses) = (0.0, None, vec![]);
    while t < 4.0 * duration {
        if (t - 0.1 * duration).abs() < DT / 2.0 {
            worlds = device.upload(&[World::new(vec![big.clone()])]).unwrap();
        }
        if takeover.is_some_and(|at: f32| (t - (at - 0.6 + 2.0 * DT)).abs() < DT / 2.0) {
            // A small box where the hand is 0.4 s before the takeover: clear of the robot now, of
            // where the replan takes over and of the replan, but not of the old trajectory between.
            let at = takeover.unwrap();
            let hand = robot
                .ee_pose(&state_at(&first, at - 0.4).position, 0)
                .position;
            let small = Obstacle::cuboid(hand, Vec3::splat(0.005));
            worlds = device
                .upload(&[World::new(vec![big.clone(), small])])
                .unwrap();
            assert!(clear(&device, &worlds, &state_at(&first, t).position));
            assert!(clear(&device, &worlds, &state_at(&first, at).position));
            let between = (1..10).map(|k| t + (at - t) * k as f32 / 10.0);
            let blocked = between
                .filter(|&s| !clear(&device, &worlds, &state_at(&first, s).position))
                .count();
            assert!(
                blocked > 0,
                "the box blocks the old trajectory before the takeover"
            );
            let mut ahead = follower.clone();
            ahead.tick(&device, &worlds, at).unwrap();
            let replan = ahead.trajectory();
            let along = (0..=100).map(|k| replan.duration() * k as f32 / 100.0);
            for s in along {
                assert!(clear(&device, &worlds, &state_at(replan, s).position));
            }
        }
        let tick = follower.tick(&device, &worlds, t).unwrap();
        assert!(
            clear(&device, &worlds, &tick.state.position),
            "at {t:.2} s the robot collides ({:?})",
            tick.status
        );
        if let FollowStatus::Replanned { at } = tick.status {
            takeover.get_or_insert(at);
        }
        statuses.push(tick.status.clone());
        if matches!(tick.status, FollowStatus::Arrived | FollowStatus::Blocked) {
            break;
        }
        t += DT;
    }
    assert!(
        takeover.is_some(),
        "the big box made it replan: {statuses:?}"
    );
    assert_eq!(
        statuses.last(),
        Some(&FollowStatus::Blocked),
        "{statuses:?}"
    );
}

#[test]
fn a_pending_replan_that_collides_is_dropped_for_a_stop() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let first = sweep(&robot, &device);
    let duration = first.duration();
    let o = FollowerOptions {
        budget: 0.3,
        ..Default::default()
    };
    let mut follower = Follower::try_new(&robot, 0, first.clone(), 0.0, o).unwrap();
    let big = box_at(&robot, &first, 0.6 * duration);
    let mut worlds = device.upload(&[World::new(vec![big.clone()])]).unwrap();
    let mut t = 0.1 * duration;
    let at = loop {
        let tick = follower.tick(&device, &worlds, t).unwrap();
        if let FollowStatus::Replanned { at } = tick.status {
            break at;
        }
        assert_eq!(tick.status, FollowStatus::Following);
        t += DT;
    };
    // The replan, read from a copy that has switched to it.
    let mut ahead = follower.clone();
    ahead.tick(&device, &worlds, at).unwrap();
    let replan = ahead.trajectory().clone();
    // A box on the replan a second after it takes over, clear of the old trajectory before.
    let hand = robot.ee_pose(&state_at(&replan, 1.0).position, 0).position;
    let small = Obstacle::cuboid(hand, Vec3::splat(0.04));
    worlds = device.upload(&[World::new(vec![big, small])]).unwrap();
    assert!(!clear(&device, &worlds, &state_at(&replan, 1.0).position));
    let mut s = t;
    while s <= at {
        assert!(clear(&device, &worlds, &state_at(&first, s).position));
        s += 0.01;
    }
    let tick = follower.tick(&device, &worlds, t + DT).unwrap();
    assert_eq!(tick.status, FollowStatus::Stopping);
}

#[test]
fn a_stop_that_cannot_end_short_of_a_box_reports_the_collision() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    // At full speed, with a box 0.2 s ahead: too close to stop for.
    let first = sweep_at(&robot, &device, 1.0);
    let duration = first.duration();
    let mut follower =
        Follower::try_new(&robot, 0, first.clone(), 0.0, FollowerOptions::default()).unwrap();
    let mut worlds = device.upload(&[World::default()]).unwrap();
    let (mut t, mut reported, mut collided) = (0.0, None, None);
    while t < 4.0 * duration {
        if (t - 0.4 * duration).abs() < 0.005 {
            let hand = robot
                .ee_pose(&state_at(&first, t + 0.2).position, 0)
                .position;
            worlds = device
                .upload(&[World::new(vec![Obstacle::cuboid(hand, Vec3::splat(0.01))])])
                .unwrap();
            assert!(clear(&device, &worlds, &state_at(&first, t).position));
        }
        let tick = follower.tick(&device, &worlds, t).unwrap();
        if let FollowStatus::Collides { at } = tick.status {
            reported.get_or_insert(at);
        }
        if !clear(&device, &worlds, &tick.state.position) {
            collided.get_or_insert(t);
        }
        if matches!(tick.status, FollowStatus::Arrived | FollowStatus::Blocked) {
            break;
        }
        t += 0.01;
    }
    let collided = collided.expect("no stop ends short of the box");
    let at = reported.expect("the follower reports the collision");
    assert!(
        (at - collided).abs() <= FollowerOptions::default().check_every + 0.01,
        "it reports a collision at {at} s; the robot collides at {collided} s"
    );
}

#[test]
fn non_finite_and_degenerate_inputs_are_errors() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let first = sweep(&robot, &device);
    let mut follower =
        Follower::try_new(&robot, 0, first, 0.0, FollowerOptions::default()).unwrap();
    fn input<T>(r: Result<T, Error>) -> bool {
        matches!(r, Err(Error::Input(_)))
    }
    assert!(input(follower.tick(&device, &worlds, f32::NAN)));
    // Checks closer than f32 can step through a motion's times would never end.
    let fine = FollowerOptions {
        check_every: 1e-8,
        ..Default::default()
    };
    let trajectory = follower.trajectory().clone();
    assert!(input(Follower::try_new(
        &robot,
        0,
        trajectory.clone(),
        0.0,
        fine
    )));
    // Options `plan` refuses would fail the first replan, with the robot moving.
    let defaults = PlanOptions::default();
    let unplannable = FollowerOptions {
        plan: PlanOptions {
            rrt: RrtOptions {
                resolution: 0.0,
                ..defaults.rrt
            },
            ..defaults
        },
        ..Default::default()
    };
    assert!(input(Follower::try_new(
        &robot,
        0,
        trajectory.clone(),
        0.0,
        unplannable
    )));
    // Infinitely far apart, the first check would come at no time at all.
    let sparse = FollowerOptions {
        check_every: f32::INFINITY,
        ..Default::default()
    };
    assert!(input(Follower::try_new(
        &robot,
        0,
        trajectory.clone(),
        0.0,
        sparse
    )));
    // A replan's goal is the trajectory's: constraints it breaks would refuse every replan, and a
    // hand the robot lacks every plan.
    let far = HandConstraint {
        position: Vec3::splat(0.01),
        ..HandConstraint::new(0, Pose::new(Vec3::new(0.0, 0.0, 2.0), Quat::IDENTITY))
    };
    for constraint in [far, HandConstraint::new(1, Pose::IDENTITY)] {
        let kept = FollowerOptions {
            constraints: vec![constraint],
            ..Default::default()
        };
        assert!(input(Follower::try_new(
            &robot,
            0,
            trajectory.clone(),
            0.0,
            kept
        )));
    }
    let state = JointState {
        position: robot.default_q().to_vec(),
        ..JointState::new(robot.dof())
    };
    let hand = robot.ee_pose(&state.position, 0);
    let o = ServoOptions::default();
    let servo = |state: &JointState, target: Pose, o: &ServoOptions| {
        batchplan::servo(&device, &worlds, 0, state, target, o)
    };
    let nan_target = Pose::new(Vec3::NAN, hand.rotation);
    assert!(input(servo(&state, nan_target, &o)));
    let mut moving = state.clone();
    moving.velocity[0] = f32::NAN;
    assert!(input(servo(&moving, hand, &o)));
    let forever = ServoOptions {
        dt: f32::INFINITY,
        ..o
    };
    assert!(input(servo(&state, hand, &forever)));
    let backward = ServoOptions { linear: -0.1, ..o };
    assert!(input(servo(&state, hand, &backward)));
}

/// The gripper's tilt from straight down at `q`.
fn tilt(robot: &Robot, q: &[f32]) -> f32 {
    (robot.ee_pose(q, 0).rotation * Vec3::Z).angle_between(Vec3::NEG_Z)
}

/// A follower carrying the gripper upright replans around a box put in its way within the same
/// bound: every state it commands, and its trajectory sampled at 1 kHz once it arrives, keep the
/// gripper within 5° of straight down, where a replan without the constraint tilts it past. Kept
/// near its goal, where it is not yet, it stops instead of replanning.
#[test]
fn replans_keep_the_constraints_the_trajectory_was_planned_with() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let bound = 5f32.to_radians();
    let upright = HandConstraint {
        tilt: bound,
        ..HandConstraint::new(0, Pose::new(Vec3::ZERO, common::upright(0.0)))
    };
    let scene = common::pillar_scene();
    let empty = device.upload(std::slice::from_ref(&scene)).unwrap();
    let (start, goals) = common::upright_reaches(&device, &empty, 16, &mut Rng::new(2)).unwrap();
    let far = |g: &&Vec<f32>| {
        let d: f32 = g.iter().zip(&start).map(|(a, b)| (a - b) * (a - b)).sum();
        d
    };
    let goal = goals
        .iter()
        .max_by(|a, b| far(a).total_cmp(&far(b)))
        .unwrap()
        .clone();
    let problem = PlanProblem {
        constraints: vec![upright],
        ..PlanProblem::new(0, start, goal)
    };
    let plans = batchplan::plan(&device, &empty, &[problem], &PlanOptions::default()).unwrap();
    let first = Trajectory::try_new(&robot, plans.best(0).expect("an upright reach"), 0.5).unwrap();
    let duration = first.duration();
    let mut blocked = scene;
    blocked
        .obstacles
        .push(box_at(&robot, &first, 0.6 * duration));
    let blocked = device.upload(&[blocked]).unwrap();
    // Within a centimeter of where the goal has the hand: the goal keeps it, a replan's start not.
    let points = &first.control_points;
    let end = robot.ee_pose(&points[points.len() - robot.dof()..], 0);
    let elsewhere = HandConstraint {
        position: Vec3::splat(0.01),
        ..HandConstraint::new(0, end)
    };
    for constraints in [vec![upright], vec![], vec![elsewhere]] {
        let o = FollowerOptions {
            constraints: constraints.clone(),
            ..FollowerOptions::default()
        };
        let mut follower = Follower::try_new(&robot, 0, first.clone(), 0.0, o).unwrap();
        let (mut t, mut statuses, mut most) = (0.0, vec![], 0.0f32);
        while t < 4.0 * duration {
            let worlds = if t < 0.3 * duration { &empty } else { &blocked };
            let tick = follower.tick(&device, worlds, t).unwrap();
            assert!(
                clear(&device, worlds, &tick.state.position),
                "at {t:.2} s it collides"
            );
            most = most.max(tilt(&robot, &tick.state.position));
            statuses.push(tick.status.clone());
            if matches!(tick.status, FollowStatus::Arrived | FollowStatus::Blocked) {
                break;
            }
            t += DT;
        }
        let timed = follower.trajectory().sample(1000.0).unwrap();
        let replan = timed
            .positions
            .chunks(robot.dof())
            .map(|q| tilt(&robot, q))
            .fold(0.0, f32::max);
        if constraints == [elsewhere] {
            assert!(
                statuses.contains(&FollowStatus::Stopping)
                    && statuses.last() == Some(&FollowStatus::Blocked),
                "{statuses:?}"
            );
            continue;
        }
        assert!(
            statuses
                .iter()
                .any(|s| matches!(s, FollowStatus::Replanned { .. }))
                && statuses.last() == Some(&FollowStatus::Arrived),
            "{statuses:?}"
        );
        if constraints.is_empty() {
            assert!(
                replan > 2.0 * bound,
                "the free replan tilts only {:.2}°",
                replan.to_degrees()
            );
        } else {
            assert!(
                most <= bound + 1e-4 && replan <= bound + 1e-4,
                "the follower tilts {:.2}° at its ticks, {:.2}° along its replan",
                most.to_degrees(),
                replan.to_degrees()
            );
        }
    }
}
