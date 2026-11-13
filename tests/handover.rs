//! One hand passing what it holds to another (`handover`), as a pick toward one end of a bar
//! leaves it: every candidate that gets through comes back, quickest first, each hand's straight
//! move holding the other arm still; hands that cannot meet say where and why; `grasps` holds an
//! object toward its ends, and `handover_grasps` and `handover_poses` propose the exchange.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::FRAC_PI_2;

use batchplan::{
    CollisionWeights, Contact, Device, Error, GraspOptions, Handover, Held, Obstacle, Pick, Pose,
    Robot, TaskOptions, TaskPlan, TaskStep, Trajectory, World, Worlds,
};
use glam::{Quat, Vec2, Vec3};

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot).unwrap()];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

/// A table with its top at z = 0, and a bar of half sizes `half` lying on it along y, 20 cm in
/// front of the left Panda's hand at its ready pose.
fn table_and_bar(robot: &Robot, half: Vec3) -> (Obstacle, Obstacle) {
    let hand = robot.ee_pose(robot.default_q(), 0).position;
    let table = Obstacle::cuboid(Vec3::new(0.3, 0.0, -0.02), Vec3::new(0.6, 0.9, 0.02));
    let bar = Obstacle::Cuboid {
        center: Vec3::new(hand.x + 0.2, hand.y, half.z),
        half_extents: half,
        rotation: Quat::from_rotation_z(FRAC_PI_2),
    };
    (table, bar)
}

/// The left Panda picks the bar up toward one end; then where it holds it, and how. Worlds: the
/// table and the bar, then the table alone.
fn picked(device: &Device, half: Vec3) -> (Worlds, Vec<f32>, Held) {
    let robot = device.robot();
    let (table, bar) = table_and_bar(robot, half);
    let worlds = device
        .upload(&[
            World::new(vec![table.clone(), bar.clone()]),
            World::new(vec![table]),
        ])
        .unwrap();
    let toward_ends = GraspOptions {
        along: 0.75,
        ..GraspOptions::default()
    };
    let grasps = batchplan::grasps(&bar, &toward_ends).unwrap();
    let pick = Pick {
        support: vec![0],
        ..Pick::new(0, robot.default_q(), 1, grasps)
    };
    let picked = batchplan::pick(device, &worlds, &[pick], &TaskOptions::default()).unwrap();
    let plan = &picked.outcomes[0].as_ref().expect("the bar is picked up")[0];
    (worlds, plan.end().to_vec(), plan.held.clone().unwrap())
}

fn configurations(t: &Trajectory) -> impl Iterator<Item = &[f32]> {
    t.control_points.chunks(t.dof)
}

/// The robot holding `held` collides nowhere along `t`, sampled at 100 Hz.
fn clear_holding(robot: &Robot, worlds: &Worlds, world: u32, held: &Held, t: &Trajectory) -> bool {
    let holding = robot.attach(&held.attachment(robot).unwrap()).unwrap();
    let device = Device::cpu(&holding).unwrap();
    let uploaded = device.upload(&[worlds[world as usize].clone()]).unwrap();
    let q = t.sample(100.0).unwrap().positions;
    let items = vec![0; q.len() / robot.dof()];
    let e = device
        .evaluate(&uploaded, &items, &q, &CollisionWeights::NONE)
        .unwrap();
    (0..items.len()).all(|i| e.collision_free(i))
}

#[test]
fn a_held_bar_is_passed_to_the_other_arm_each_hand_holding_the_other_still() {
    let robot = common::two_pandas().unwrap();
    let half = Vec3::new(0.12, 0.02, 0.02);
    let (giving, taking) = (robot.hand_joints(0), robot.hand_joints(1));
    for device in devices(&robot) {
        let name = device.name();
        let (worlds, start, held) = picked(&device, half);
        let task = Handover::new(&robot, 1, start.clone(), held.clone(), 1);
        assert!(task.meetings.len() > 10 && task.grasps.len() == 2, "{name}");
        let result = batchplan::handover(
            &device,
            &worlds,
            std::slice::from_ref(&task),
            &TaskOptions::default(),
        );
        let plans = result.unwrap().outcomes.remove(0);
        let plans = plans.unwrap_or_else(|f| panic!("{name}: {f}"));
        assert!(plans.len() > 1, "{name}: {} plans", plans.len());
        let durations: Vec<f32> = plans.iter().map(TaskPlan::duration).collect();
        assert!(
            durations.is_sorted(),
            "{name}: quickest first, {durations:?}"
        );
        for plan in &plans {
            let (meeting, grasp) = (
                task.meetings[plan.candidate / task.grasps.len()],
                task.grasps[plan.candidate % task.grasps.len()],
            );
            assert_eq!(plan.meeting, Some(meeting), "{name}");
            let taken = plan.held.as_ref().unwrap();
            assert_eq!((taken.hand, taken.grasp), (1, grasp), "{name}");
            assert_eq!(taken.object, held.object, "{name}");
            // The way there from the start, holding the bar, clear of everything.
            let meet = plan.transit.as_ref().unwrap();
            assert_eq!(configurations(meet).next().unwrap(), &start[..], "{name}");
            assert_eq!(configurations(meet).last().unwrap(), plan.begin(), "{name}");
            assert!(clear_holding(&robot, &worlds, 1, &held, meet), "{name}");
            // Moving in, the giver holds the bar at the meeting and keeps still to the bit.
            let take_start = configurations(&plan.approach).next().unwrap();
            for q in configurations(&plan.approach) {
                assert!(giving.iter().all(|&j| q[j] == take_start[j]), "{name}");
            }
            let gives = meeting.mul_pose(held.grasp);
            let at = robot.ee_pose(take_start, 0);
            assert!(at.position.distance(gives.position) < 2e-3, "{name}");
            let took = configurations(&plan.approach).last().unwrap();
            let takes = robot.ee_pose(took, 1);
            assert!(
                takes.position.distance(meeting.mul_pose(grasp).position) < 2e-3,
                "{name}"
            );
            // Moving away, the taker keeps still, and the giver backs off along its z axis.
            for q in configurations(&plan.depart) {
                assert!(taking.iter().all(|&j| q[j] == took[j]), "{name}");
            }
            let left = robot.ee_pose(plan.end(), 0).position;
            let back = gives.position - gives.rotation * Vec3::Z * task.retreat;
            assert!(left.distance(back) < 2e-3, "{name}: {left} for {back}");
        }
    }
}

/// A bar too short for two hands: every meeting puts them in each other's way, and the failure says
/// so, naming links of both arms.
#[test]
fn hands_that_cannot_meet_say_where_and_why() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let (worlds, start, held) = picked(&device, Vec3::new(0.03, 0.02, 0.02));
    let task = Handover::new(&robot, 1, start, held, 1);
    let result = batchplan::handover(&device, &worlds, &[task], &TaskOptions::default()).unwrap();
    let failure = match &result.outcomes[0] {
        Ok(plans) => panic!("{} plans share a 6 cm bar", plans.len()),
        Err(failure) => failure,
    };
    assert_eq!(failure.step(), TaskStep::Reach, "{failure}");
    let both = |c: &Contact| match c {
        Contact::SelfCollision { links, .. } => {
            links[0].starts_with("left_") != links[1].starts_with("left_")
        }
        _ => false,
    };
    let between = failure.candidates.iter().flat_map(|c| &c.contacts);
    assert!(between.filter(|c| both(c)).count() > 0, "{failure}");
}

#[test]
fn malformed_handovers_are_input_errors() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let (worlds, start, held) = picked(&device, Vec3::new(0.12, 0.02, 0.02));
    let good = Handover::new(&robot, 1, start, held, 1);
    let cases = [
        (
            Handover {
                to: 0,
                ..good.clone()
            },
            "takes what it holds",
        ),
        (
            Handover {
                to: 2,
                ..good.clone()
            },
            "hand 2",
        ),
        (
            Handover {
                world: 2,
                ..good.clone()
            },
            "world 2",
        ),
        (
            Handover {
                meetings: vec![],
                ..good.clone()
            },
            "no meetings",
        ),
        (
            Handover {
                grasps: vec![],
                ..good.clone()
            },
            "no grasps",
        ),
        (
            Handover {
                approach: 0.0,
                ..good.clone()
            },
            "positive approach",
        ),
        (
            Handover {
                retreat: f32::NAN,
                ..good
            },
            "positive approach",
        ),
    ];
    for (task, what) in cases {
        match batchplan::handover(&device, &worlds, &[task], &TaskOptions::default()) {
            Err(Error::Input(message)) => assert!(message.contains(what), "{what}: {message}"),
            other => panic!("{what}: {other:?}"),
        }
    }
    let bad = GraspOptions {
        along: 1.0,
        ..GraspOptions::default()
    };
    let bar = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.12, 0.02, 0.02));
    assert!(matches!(
        batchplan::grasps(&bar, &bad),
        Err(Error::Input(_))
    ));
}

/// Grasps toward the ends hold a bar at three quarters of its half length either side, along the
/// side the fingers close on and not across it, and the same grasps as the middle's otherwise.
#[test]
fn grasps_toward_the_ends_slide_along_the_side_the_fingers_close_on() {
    let half = Vec3::new(0.12, 0.02, 0.03);
    let bar = Obstacle::cuboid(Vec3::ZERO, half);
    let middle = batchplan::grasps(&bar, &GraspOptions::default()).unwrap();
    let ends = GraspOptions {
        along: 0.75,
        ..GraspOptions::default()
    };
    let ends = batchplan::grasps(&bar, &ends).unwrap();
    assert_eq!(ends.len(), 2 * middle.len());
    for (k, centered) in middle.iter().enumerate() {
        let pair = &ends[2 * k..2 * k + 2];
        let shift = pair[0].position - centered.position;
        assert_eq!(pair[1].position - centered.position, -shift, "grasp {k}");
        assert!(
            pair.iter().all(|g| g.rotation == centered.rotation),
            "grasp {k}"
        );
        // Along the one axis neither the approach nor the closing runs on, 3/4 of the way out.
        let free = centered.rotation * Vec3::X;
        let axis = (0..3)
            .find(|&a| free[a].abs() > 0.5)
            .expect("an axis of the box");
        assert!(
            (shift.length() - 0.75 * half[axis]).abs() < 1e-6,
            "grasp {k}: {shift}"
        );
        assert!(
            shift.normalize().dot(free).abs() > 1.0 - 1e-6,
            "grasp {k}: {shift}"
        );
    }
}

/// The taker's grasps hold the other end: the giver's grasp turned half about the bar's vertical
/// axis keeps its approach from above, and turned half about the axis across both comes from below.
/// The meetings lie between the arms and in front of them, the bar's held end toward the giver,
/// tilted as the hand holds it.
#[test]
fn the_exchange_proposed_puts_each_hand_toward_its_own_end() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let (_, start, held) = picked(&device, Vec3::new(0.12, 0.02, 0.02));
    let grasps = batchplan::handover_grasps(&held);
    assert_eq!(grasps.len(), 2);
    let approach = |g: &Pose| g.rotation * Vec3::Z;
    for g in &grasps {
        assert!(
            (g.position + held.grasp.position).length() < 0.03,
            "the other end: {g:?}"
        );
    }
    assert!(
        approach(&grasps[0]).dot(approach(&held.grasp)) > 0.99,
        "from the same side"
    );
    assert!(
        approach(&grasps[1]).dot(approach(&held.grasp)) < -0.99,
        "from the other side"
    );
    let meetings = batchplan::handover_poses(&robot, &start, &held, 1);
    assert!(!meetings.is_empty());
    let hand = robot.ee_pose(&start, 0);
    let tilt = hand.mul_pose(held.grasp.inverse()).rotation * Vec3::Z;
    for m in &meetings {
        assert!(m.position.y.abs() < 1e-3 && m.position.x > 0.1, "{m:?}");
        let held_end = m.rotation * held.grasp.position;
        assert!(
            held_end.y > 0.0,
            "the held end faces the giver, at +y: {m:?}"
        );
        let up = m.rotation * Vec3::Z;
        assert!(
            (up.z - tilt.z).abs() < 1e-4,
            "the tilt kept: {up} for {tilt}"
        );
    }
    assert!(batchplan::handover_poses(&robot, &start, &held, 0).is_empty());
    let lost = vec![f32::NAN; start.len()];
    assert!(
        batchplan::handover_poses(&robot, &lost, &held, 1).is_empty(),
        "no start"
    );
    assert!(batchplan::handover_poses(&robot, &start[1..], &held, 1).is_empty());
}

/// Two tasks in one call, the second at some of the first's meetings in the other order, each
/// planned as it is alone, its own meetings reported; and the giver backs off under `tool_speed`,
/// the cap on the hand that moves away.
#[test]
fn tasks_together_plan_as_each_alone_and_the_giver_leaves_under_the_cap() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let (worlds, start, held) = picked(&device, Vec3::new(0.12, 0.02, 0.02));
    let first = Handover::new(&robot, 1, start, held, 1);
    let second = Handover {
        meetings: first.meetings.iter().rev().take(20).copied().collect(),
        ..first.clone()
    };
    let o = TaskOptions::default();
    let alone = |task: &Handover| {
        batchplan::handover(&device, &worlds, std::slice::from_ref(task), &o).unwrap()
    };
    let both = batchplan::handover(&device, &worlds, &[first.clone(), second.clone()], &o).unwrap();
    for (k, task) in [first, second].iter().enumerate() {
        let one = alone(task);
        let (together, alone) = (both.outcomes[k].as_ref(), one.outcomes[0].as_ref());
        let (together, alone) = (together.unwrap(), alone.unwrap());
        assert_eq!(together.len(), alone.len(), "task {k}");
        for (a, b) in together.iter().zip(alone) {
            assert_eq!(
                (a.candidate, a.meeting),
                (b.candidate, b.meeting),
                "task {k}"
            );
            assert_eq!(
                a.approach.control_points, b.approach.control_points,
                "task {k}"
            );
            assert_eq!(a.depart.control_points, b.depart.control_points, "task {k}");
            let meeting = task.meetings[a.candidate / task.grasps.len()];
            assert_eq!(a.meeting, Some(meeting), "task {k}");
        }
    }
    let plan = &both.outcomes[0].as_ref().unwrap()[0];
    let samples = plan.depart.sample(500.0).unwrap();
    let positions: Vec<Vec3> = samples
        .positions
        .chunks(robot.dof())
        .map(|q| robot.ee_pose(q, 0).position)
        .collect();
    let fastest = positions
        .windows(2)
        .map(|w| w[0].distance(w[1]) * 500.0)
        .fold(0.0, f32::max);
    assert!(
        fastest > 0.5 * o.tool_speed,
        "the giver moves away: {fastest} m/s"
    );
    assert!(
        fastest <= 1.02 * o.tool_speed,
        "the giver leaves at {fastest} m/s"
    );
}

/// Toward the ends, a cylinder's grasps around its side slide along its axis and stay on its
/// surface, and over a floor no grasp's frame goes below it.
#[test]
fn round_objects_are_grasped_toward_their_ends_above_the_floor() {
    let can = Obstacle::Cylinder {
        center: Vec3::new(0.4, 0.0, 0.1),
        radius: 0.02,
        half_height: 0.1,
        rotation: Quat::IDENTITY,
    };
    let middle = batchplan::grasps(&can, &GraspOptions::default()).unwrap();
    let ends = GraspOptions {
        along: 0.75,
        ..GraspOptions::default()
    };
    let ends = batchplan::grasps(&can, &ends).unwrap();
    assert!(ends.len() > middle.len());
    // From the side (the hand's approach level) and slid along the axis.
    let level = |g: &&Pose| (g.rotation * Vec3::Z).z.abs() < 1e-3;
    let around = ends
        .iter()
        .filter(level)
        .filter(|g| g.position.z.abs() > 1e-3);
    assert!(around.clone().count() > 0, "side grasps slide");
    for g in around {
        let off_axis = Vec2::new(g.position.x, g.position.y).length();
        assert!(off_axis <= 0.02 + 1e-6, "on the can: {g:?}");
        assert!(
            (g.position.z.abs() - 0.075).abs() < 1e-6,
            "3/4 along: {g:?}"
        );
    }
    let floor = GraspOptions {
        along: 0.97,
        floor: Some(0.005),
        ..GraspOptions::default()
    };
    for g in batchplan::grasps(&can, &floor).unwrap() {
        let frame = can.pose().mul_pose(g).position;
        assert!(frame.z >= 0.005 - 1e-5, "above the floor: {frame}");
    }
}
