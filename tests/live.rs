//! Live motion through its interface: a follower replans around a box moved into its path and
//! takes over from the state the robot is in then, stops on its path when there is no time or its
//! replan collides, and says so when no stop is short enough; one that keeps its path, as along a
//! tool path, stops on it and holds a robot held back where it is; replans keep the trajectory's
//! constraints, and stop where they cannot start inside them; servo steps slow and halt before a
//! wall without entering it, reach a free target and brake joints before their ends, within the
//! limits.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::glam::{Quat, Vec3};
use batchplan::rng::Rng;
use batchplan::{
    CollisionWeights, Contact, Device, Error, FollowStatus, Follower, FollowerOptions,
    HandConstraint, HandSpeed, JointState, Obstacle, PlanOptions, PlanProblem, Pose, Robot,
    RobotOptions, RrtOptions, ServoOptions, ServoStatus, ToolPath, ToolPathOptions, Trajectory,
    World, Worlds,
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

/// Whether `contacts` name a link against obstacle `obstacle`, and nothing else.
fn names(contacts: &[Contact], obstacle: usize) -> bool {
    let named = |c: &Contact| match c {
        Contact::World {
            obstacle: o,
            distance,
            ..
        } => *o == obstacle && *distance <= 0.0,
        _ => false,
    };
    !contacts.is_empty() && contacts.iter().all(named)
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

fn clear(device: &Device, worlds: &Worlds, q: &[f32]) -> bool {
    let e = device
        .evaluate(worlds, &[0], q, &CollisionWeights::NONE)
        .unwrap();
    e.collision_free(0)
}

#[test]
fn a_box_in_the_path_is_replanned_around_from_the_predicted_state() {
    replans_around_a_box(0.0);
}

/// A lead moves the takeover later by itself, and the plan from where the robot is then.
#[test]
fn a_lead_gives_a_replan_time_to_travel_before_it_takes_over() {
    replans_around_a_box(0.1);
}

/// The sweep with a box dropped ahead of the hand: the follower replans `budget + lead` ahead of
/// the tick that sees it, hands the replan over in its status, and arrives on it.
fn replans_around_a_box(lead: f32) {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let first = sweep(&robot, &device);
    let duration = first.duration();
    let o = FollowerOptions {
        lead,
        ..Default::default()
    };
    let budget = o.budget;
    let mut follower = Follower::try_new(&robot, 0, first.clone(), 0.0, o).unwrap();
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
        let tick = follower.tick(&device, &worlds, t, None).unwrap();
        assert!(
            clear(&device, &worlds, &tick.state.position),
            "at {t:.2} s the robot collides ({:?})",
            tick.status
        );
        if let FollowStatus::Replanned { at, trajectory } = &tick.status {
            assert!(
                (at - (t + budget + lead)).abs() < 1e-5,
                "a replan seen at {t} takes over at {at}, not {}",
                t + budget + lead
            );
            assert_eq!(
                *follower.trajectory(),
                before,
                "the old trajectory runs until {at}"
            );
            replanned = Some((before, began, *at, trajectory.clone()));
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
    let (old, began, at, handed) = replanned.expect("the box made it replan");
    // The replan its status handed over takes over, from the state the old trajectory reaches
    // when it does.
    let (new, new_began) = (follower.trajectory(), follower.began());
    assert_eq!(
        *new, handed,
        "the replan that took over is the one handed over"
    );
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
    // The checks run on the device, and what they found is named on its CPU threads.
    for device in devices(&robot) {
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
            let tick = follower.tick(&device, &worlds, t, None).unwrap();
            assert!(
                clear(&device, &worlds, &tick.state.position),
                "at {t:.2} s the robot collides ({:?})",
                tick.status
            );
            // Every tick of the stop names the box it is for.
            if let FollowStatus::Stopping { contacts } = &tick.status {
                assert!(names(contacts, 0), "{contacts:?}");
                stop.get_or_insert_with(|| (follower.trajectory().clone(), tick.state.clone()));
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
    let mut replan: Option<Trajectory> = None;
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
            let replan = replan.as_ref().expect("the replan was handed over");
            let along = (0..=100).map(|k| replan.duration() * k as f32 / 100.0);
            for s in along {
                assert!(clear(&device, &worlds, &state_at(replan, s).position));
            }
        }
        let tick = follower.tick(&device, &worlds, t, None).unwrap();
        assert!(
            clear(&device, &worlds, &tick.state.position),
            "at {t:.2} s the robot collides ({:?})",
            tick.status
        );
        if let FollowStatus::Replanned { at, trajectory } = &tick.status
            && takeover.is_none()
        {
            takeover = Some(*at);
            replan = Some(trajectory.clone());
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
    let (at, replan) = loop {
        let tick = follower.tick(&device, &worlds, t, None).unwrap();
        if let FollowStatus::Replanned { at, trajectory } = tick.status {
            break (at, trajectory);
        }
        assert_eq!(tick.status, FollowStatus::Following);
        t += DT;
    };
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
    let tick = follower.tick(&device, &worlds, t + DT, None).unwrap();
    assert!(
        matches!(&tick.status, FollowStatus::Stopping { contacts } if names(contacts, 1)),
        "the stop is for the small box: {:?}",
        tick.status
    );
}

#[test]
fn a_stop_that_runs_into_another_box_names_it() {
    let robot = common::panda().unwrap();
    for device in devices(&robot) {
        let first = sweep(&robot, &device);
        let duration = first.duration();
        let o = FollowerOptions {
            budget: duration,
            ..Default::default()
        };
        let mut follower = Follower::try_new(&robot, 0, first.clone(), 0.0, o).unwrap();
        let ahead = box_at(&robot, &first, 0.8 * duration);
        let worlds = device.upload(&[World::new(vec![ahead.clone()])]).unwrap();
        let t = 0.2 * duration;
        let tick = follower.tick(&device, &worlds, t, None).unwrap();
        assert!(
            matches!(&tick.status, FollowStatus::Stopping { contacts } if names(contacts, 0)),
            "{:?}",
            tick.status
        );
        // A second box where the stop comes to rest: the stop runs into it, not the first.
        let stop = follower.trajectory().clone();
        let rest = box_at(&robot, &stop, stop.duration());
        let worlds = device.upload(&[World::new(vec![ahead, rest])]).unwrap();
        let tick = follower.tick(&device, &worlds, t + DT, None).unwrap();
        assert!(
            matches!(&tick.status, FollowStatus::Collides { contacts, .. } if names(contacts, 1)),
            "{}: {:?}",
            device.name(),
            tick.status
        );
    }
}

#[test]
fn a_motion_too_near_its_end_to_stop_names_what_it_collides_with() {
    let robot = common::panda().unwrap();
    for device in devices(&robot) {
        // At full speed, 0.2 s from the end of the path, with a box over the hand at the goal:
        // the path ends sooner than any stop would, so it runs to its end, into the box.
        let first = sweep_at(&robot, &device, 1.0);
        let duration = first.duration();
        let mut follower =
            Follower::try_new(&robot, 0, first.clone(), 0.0, FollowerOptions::default()).unwrap();
        let worlds = device
            .upload(&[World::new(vec![box_at(&robot, &first, duration)])])
            .unwrap();
        let tick = follower
            .tick(&device, &worlds, duration - 0.2, None)
            .unwrap();
        assert!(
            matches!(&tick.status, FollowStatus::Collides { contacts, .. } if names(contacts, 0)),
            "{}: {:?}",
            device.name(),
            tick.status
        );
    }
}

/// On every device, a ball stepped 10 nm at a time through the radius at which it first touches
/// the rest of the motion: each stop names the ball, though a GPU device's check may find a graze
/// its CPU threads' contacts put a hair outside.
#[test]
fn a_graze_is_named_on_every_device() {
    let robot = common::panda().unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let first = sweep(&robot, &cpu);
    let duration = first.duration();
    let o = FollowerOptions {
        budget: duration,
        ..Default::default()
    };
    let t = 0.2 * duration;
    let ahead = robot
        .ee_pose(&state_at(&first, 0.6 * duration).position, 0)
        .position;
    let center = ahead - Vec3::new(0.0, 0.0, 0.12);
    // The radius at which a ball there first touches the samples the follower checks.
    let point = cpu
        .upload(&[World::new(vec![Obstacle::sphere(center, 0.0)])])
        .unwrap();
    let mut touch = f32::INFINITY;
    for k in 0.. {
        let at = (t + k as f32 * o.check_every).min(duration);
        let near = cpu
            .contacts(&point, &[0], &state_at(&first, at).position, 1.0)
            .unwrap();
        let ball = near[0]
            .iter()
            .filter(|c| matches!(c, Contact::World { .. }))
            .map(Contact::distance);
        touch = ball.fold(touch, f32::min);
        if at >= duration {
            break;
        }
    }
    assert!(touch.is_finite() && touch > 0.01, "{touch}");
    for device in devices(&robot) {
        let mut stopped = 0;
        for step in -50..50 {
            let radius = touch + step as f32 * 1e-8;
            let ball = World::new(vec![Obstacle::sphere(center, radius)]);
            let worlds = device.upload(&[ball]).unwrap();
            let mut follower = Follower::try_new(&robot, 0, first.clone(), 0.0, o.clone()).unwrap();
            match follower.tick(&device, &worlds, t, None).unwrap().status {
                FollowStatus::Stopping { contacts } | FollowStatus::Collides { contacts, .. } => {
                    assert!(!contacts.is_empty(), "{}: radius {radius}", device.name());
                    stopped += 1;
                }
                FollowStatus::Following => {}
                status => panic!("{status:?}"),
            }
        }
        assert!(
            stopped > 10 && stopped < 90,
            "{}: {stopped} of 100",
            device.name()
        );
    }
}

#[test]
fn a_stop_that_cannot_end_short_of_a_box_reports_the_collision() {
    let robot = common::panda().unwrap();
    for device in devices(&robot) {
        let name = device.name();
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
            let tick = follower.tick(&device, &worlds, t, None).unwrap();
            if let FollowStatus::Collides { at, contacts } = &tick.status {
                // What the motion it is on meets at `at`, where its stop samples on its own clock.
                let q = state_at(follower.trajectory(), at - follower.began()).position;
                let there = device.contacts(&worlds, &[0], &q, 0.0).unwrap().remove(0);
                let same = there.len() == contacts.len()
                    && there
                        .iter()
                        .zip(contacts)
                        .all(|(a, b)| (a.distance() - b.distance()).abs() < 1e-5);
                assert!(
                    names(contacts, 0) && same,
                    "{name}: {contacts:?} against {there:?}"
                );
                reported.get_or_insert(*at);
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
            "{name}: it reports a collision at {at} s; the robot collides at {collided} s"
        );
    }
}

#[test]
fn a_robot_held_far_back_is_followed_from_where_it_is() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let first = sweep(&robot, &device);
    let duration = first.duration();
    let mut follower =
        Follower::try_new(&robot, 0, first.clone(), 0.0, FollowerOptions::default()).unwrap();
    // A servo a little short of its command: the trajectory goes on as it was.
    let t = 0.4 * duration;
    let mut short = state_at(&first, t);
    short.position[0] -= 0.1;
    let tick = follower.tick(&device, &worlds, t, Some(&short)).unwrap();
    assert_eq!(tick.status, FollowStatus::Following);
    assert_eq!(tick.state, state_at(&first, t));
    // Held back where it was a third of the way earlier: a new trajectory from there.
    let held = state_at(&first, 0.1 * duration);
    let tick = follower.tick(&device, &worlds, t, Some(&held)).unwrap();
    let FollowStatus::Reanchored { off } = tick.status else {
        panic!("{:?}", tick.status);
    };
    let farthest = (0..robot.dof())
        .map(|j| (held.position[j] - state_at(&first, t).position[j]).abs())
        .fold(0.0, f32::max);
    assert!((off - farthest).abs() < 1e-5 && off > 0.2, "{off}");
    let from = (0..robot.dof()).map(|j| (tick.state.position[j] - held.position[j]).abs());
    assert!(from.fold(0.0, f32::max) < 1e-5, "{:?}", tick.state);
    let speed = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let gap = (0..robot.dof()).map(|j| (tick.state.velocity[j] - held.velocity[j]).abs());
    assert!(gap.fold(0.0, f32::max) < 1e-3 && speed(&held.velocity) > 0.1);
    assert_eq!(follower.began(), t);
    // Then followed to the goal.
    let mut now = t;
    let status = loop {
        now += DT;
        let tick = follower.tick(&device, &worlds, now, None).unwrap();
        if !matches!(tick.status, FollowStatus::Following) || now > t + 4.0 * duration {
            break tick;
        }
    };
    assert_eq!(status.status, FollowStatus::Arrived);
    assert_eq!(status.state.position, TO);
    // Held inside a box, it cannot plan from there: held at rest there, blocked.
    let mut follower =
        Follower::try_new(&robot, 0, first.clone(), 0.0, FollowerOptions::default()).unwrap();
    let boxed = device
        .upload(&[World::new(vec![box_at(&robot, &first, 0.1 * duration)])])
        .unwrap();
    let tick = follower.tick(&device, &boxed, t, Some(&held)).unwrap();
    assert_eq!(tick.status, FollowStatus::Blocked);
    assert_eq!(tick.state.position, held.position);
    assert!(speed(&tick.state.velocity) == 0.0);
    let tick = follower.tick(&device, &boxed, t + DT, None).unwrap();
    assert_eq!(tick.status, FollowStatus::Blocked);
    assert_eq!(tick.state.position, held.position);
    // A measured state of the wrong size is refused.
    let wrong = JointState::new(robot.dof() - 1);
    assert!(matches!(
        follower.tick(&device, &worlds, t, Some(&wrong)),
        Err(Error::Input(_))
    ));
}

/// The hand moved 30 cm forward from the Panda's default configuration at 10 cm/s, as `follow`
/// moves it: the trajectory and the line's ends.
fn line(robot: &Robot, device: &Device) -> (Trajectory, Vec3, Vec3) {
    let worlds = device.upload(&[World::default()]).unwrap();
    let from = robot.ee_pose(robot.default_q(), 0);
    let to = Pose::new(from.position + Vec3::new(0.3, 0.0, 0.0), from.rotation);
    let path = ToolPath {
        world: 0,
        start: robot.default_q().to_vec(),
        hand: 0,
        waypoints: vec![to],
        holding: None,
    };
    let result = batchplan::follow(device, &worlds, &[path], &ToolPathOptions::default()).unwrap();
    assert_eq!(result.fraction[0], 1.0, "{:?}", result.stopped[0]);
    let speed = HandSpeed {
        hand: 0,
        linear: 0.1,
        angular: f32::INFINITY,
    };
    let trajectory = Trajectory::with_hand_speed(robot, &result.paths[0], &speed).unwrap();
    (trajectory, from.position, to.position)
}

/// The distance from `p` to the segment from `a` to `b`.
fn off_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
    p.distance(a + (b - a) * t)
}

#[test]
fn a_follower_that_keeps_its_path_stops_on_it_where_the_rest_collides() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let (line, from, to) = line(&robot, &device);
    let duration = line.duration();
    let free = device.upload(&[World::default()]).unwrap();
    let middle = (from + to) / 2.0;
    let blocked = device
        .upload(&[World::new(vec![Obstacle::cuboid(
            middle,
            Vec3::splat(0.04),
        )])])
        .unwrap();
    assert!(
        !clear(&device, &blocked, &state_at(&line, duration / 2.0).position)
            && clear(&device, &blocked, &line.start())
            && clear(&device, &blocked, &line.end()),
        "the box blocks the line, and not its ends"
    );
    // The line with the box dropped onto its middle a tenth of the way along: the statuses, and
    // the hand's positions.
    let run = |replan: bool| {
        let o = FollowerOptions {
            replan,
            ..Default::default()
        };
        let mut follower = Follower::try_new(&robot, 0, line.clone(), 0.0, o).unwrap();
        let (mut t, mut statuses, mut hand) = (0.0, vec![], vec![]);
        while t < 4.0 * duration {
            let worlds = if t < duration / 10.0 { &free } else { &blocked };
            let tick = follower.tick(&device, worlds, t, None).unwrap();
            assert!(
                clear(&device, worlds, &tick.state.position),
                "at {t:.2} s the robot collides ({:?})",
                tick.status
            );
            hand.push(robot.ee_pose(&tick.state.position, 0).position);
            statuses.push(tick.status.clone());
            if matches!(tick.status, FollowStatus::Arrived | FollowStatus::Blocked) {
                break;
            }
            t += DT;
        }
        (statuses, hand)
    };
    // Replanning, it would leave the line for a way around the box.
    let (statuses, _) = run(true);
    assert!(
        statuses
            .iter()
            .any(|s| matches!(s, FollowStatus::Replanned { .. })),
        "{statuses:?}"
    );
    // Keeping its path, it stops on the line short of the box, at rest.
    let (statuses, hand) = run(false);
    let stopping =
        |s: &FollowStatus| matches!(s, FollowStatus::Stopping { contacts } if names(contacts, 0));
    assert!(
        statuses.iter().any(stopping)
            && statuses.iter().all(|s| {
                matches!(s, FollowStatus::Following | FollowStatus::Blocked) || stopping(s)
            }),
        "{statuses:?}"
    );
    assert_eq!(statuses.last(), Some(&FollowStatus::Blocked));
    let off = hand
        .iter()
        .map(|&p| off_segment(p, from, to))
        .fold(0.0, f32::max);
    assert!(off < 2e-3, "the hand strays {off} m from the line");
}

#[test]
fn a_follower_that_keeps_its_path_holds_a_robot_held_far_back_where_it_is() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let (line, ..) = line(&robot, &device);
    let t = line.duration() / 2.0;
    let mut held = state_at(&line, t);
    held.position[0] -= 0.3;
    assert!(held.velocity.iter().any(|&v| v.abs() > 0.01), "it moves");
    // Replanning, it would follow from there on a new motion.
    let mut follower =
        Follower::try_new(&robot, 0, line.clone(), 0.0, FollowerOptions::default()).unwrap();
    let tick = follower.tick(&device, &worlds, t, Some(&held)).unwrap();
    assert!(
        matches!(tick.status, FollowStatus::Reanchored { .. }),
        "{:?}",
        tick.status
    );
    // Keeping its path, it holds the robot there at rest.
    let o = FollowerOptions {
        replan: false,
        ..Default::default()
    };
    let mut follower = Follower::try_new(&robot, 0, line, 0.0, o).unwrap();
    for (now, measured) in [(t, Some(&held)), (t + DT, None)] {
        let tick = follower.tick(&device, &worlds, now, measured).unwrap();
        assert_eq!(tick.status, FollowStatus::Blocked);
        assert_eq!(tick.state.position, held.position);
        assert!(tick.state.velocity.iter().all(|&v| v == 0.0));
    }
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
    assert!(input(follower.tick(&device, &worlds, f32::NAN, None)));
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
    // A lead that is negative or not a number would take a replan over before it was planned.
    for lead in [-0.01, f32::NAN] {
        let early = FollowerOptions {
            lead,
            ..Default::default()
        };
        assert!(input(Follower::try_new(
            &robot,
            0,
            trajectory.clone(),
            0.0,
            early
        )));
    }
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
    // Without replans, the goal may lie outside them.
    let unplanned = FollowerOptions {
        constraints: vec![far],
        replan: false,
        ..Default::default()
    };
    Follower::try_new(&robot, 0, trajectory, 0.0, unplanned).unwrap();
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
    // The farthest goal whose reach the box, where the hand will be at 60%, appears clear of: the
    // robot is 30% along when it does, and a reach that turns back on itself would be inside it.
    let mut by_distance: Vec<&Vec<f32>> = goals.iter().collect();
    by_distance.sort_by(|a, b| far(b).total_cmp(&far(a)));
    let mut found = None;
    for goal in by_distance {
        let problem = PlanProblem {
            constraints: vec![upright],
            ..PlanProblem::new(0, start.clone(), goal.clone())
        };
        let plans = batchplan::plan(&device, &empty, &[problem], &PlanOptions::default()).unwrap();
        let Some(path) = plans.best(0) else { continue };
        let first = Trajectory::try_new(&robot, path, 0.5).unwrap();
        let mut blocked = scene.clone();
        blocked
            .obstacles
            .push(box_at(&robot, &first, 0.6 * first.duration()));
        let blocked = device.upload(&[blocked]).unwrap();
        let mut state = JointState::new(robot.dof());
        first.at(0.3 * first.duration() + DT, &mut state);
        if clear(&device, &blocked, &state.position) {
            found = Some((first, blocked));
            break;
        }
    }
    let (first, blocked) = found.expect("an upright reach the box appears clear of");
    let duration = first.duration();
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
            let tick = follower.tick(&device, worlds, t, None).unwrap();
            assert!(
                clear(&device, worlds, &tick.state.position),
                "at {t:.2} s of {duration:.2} it collides, {} constraints: {:?} after {:?}, at {:?}",
                constraints.len(),
                tick.status,
                statuses.last(),
                tick.state.position
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
                statuses
                    .iter()
                    .any(|s| matches!(s, FollowStatus::Stopping { .. }))
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
