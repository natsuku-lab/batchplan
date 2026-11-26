//! Moving the hand along a set path (`follow`), on every device: the motion stays within the
//! path's precision, stops where the path collides, leaves reach or makes the joints jump, and
//! says how far it got.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::{FRAC_PI_2, PI};

use batchplan::rng::Rng;
use batchplan::{
    CollisionModel, CollisionWeights, Contact, Device, Error, HandSpeed, Obstacle, Pose, Robot,
    RobotOptions, Screw, ToolPath, ToolPathOptions, ToolPathResult, ToolPathStop, Touch,
    Trajectory, World,
};
use glam::{Quat, Vec3};

fn follow(device: &Device, world: World, paths: &[ToolPath]) -> ToolPathResult {
    let worlds = device.upload(&[world]).unwrap();
    batchplan::follow(device, &worlds, paths, &ToolPathOptions::default()).unwrap()
}

fn path(robot: &Robot, waypoints: Vec<Pose>) -> ToolPath {
    ToolPath {
        world: 0,
        start: robot.default_q().to_vec(),
        hand: 0,
        waypoints,
        hold: vec![],
        holding: None,
    }
}

/// The hand's poses along a followed path, timed and sampled at 250 Hz.
fn hand_poses(robot: &Robot, control_points: &[f32]) -> Vec<Pose> {
    let trajectory = Trajectory::try_new(robot, control_points, 1.0).unwrap();
    let samples = trajectory.sample(250.0).unwrap();
    samples
        .positions
        .chunks(robot.dof())
        .map(|q| robot.ee_pose(q, 0))
        .collect()
}

/// The distance from `p` to the segment from `a` to `b`.
fn off_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
    p.distance(a + (b - a) * t)
}

#[test]
fn a_straight_line_is_followed_within_its_precision() {
    let robot = common::panda().unwrap();
    let start = robot.ee_pose(robot.default_q(), 0);
    let end = Pose::new(start.position - Vec3::new(0.0, 0.0, 0.1), start.rotation);
    let o = ToolPathOptions::default();
    for device in common::devices(&robot) {
        let name = device.name();
        let result = follow(&device, World::default(), &[path(&robot, vec![end])]);
        assert_eq!(result.fraction[0], 1.0, "{name}: {:?}", result.stopped[0]);
        assert_eq!(result.stopped[0], None, "{name}");
        let poses = hand_poses(&robot, &result.paths[0]);
        assert!(poses.len() > 50, "{name}: {} samples", poses.len());
        for pose in &poses {
            let off = off_segment(pose.position, start.position, end.position);
            assert!(off <= o.precision + 1e-4, "{name}: {off} m off the line");
            let turned = pose.rotation.angle_between(start.rotation);
            assert!(
                turned <= o.rotation_precision + 1e-3,
                "{name}: turned {turned} rad"
            );
        }
        let last = poses.last().unwrap();
        assert!(last.position.distance(end.position) < 1e-3, "{name}");
    }
}

#[test]
fn a_hand_speed_cap_moves_the_hand_at_that_speed() {
    let robot = common::panda().unwrap();
    let start = robot.ee_pose(robot.default_q(), 0);
    let end = Pose::new(start.position - Vec3::new(0.0, 0.0, 0.1), start.rotation);
    let cpu = Device::cpu(&robot).unwrap();
    let result = follow(&cpu, World::default(), &[path(&robot, vec![end])]);
    let speed = HandSpeed {
        hand: 0,
        linear: 0.05,
        angular: f32::INFINITY,
        near: None,
    };
    let trajectory = Trajectory::with_hand_speed(&robot, &result.paths[0], &[speed]).unwrap();
    // 2 s at the cap, and a little more to start and stop.
    let at_cap = 0.1 / speed.linear;
    assert!(
        trajectory.duration() >= at_cap && trajectory.duration() < 1.15 * at_cap,
        "{} s for {at_cap} s at the cap",
        trajectory.duration()
    );
}

#[test]
fn an_obstacle_stops_the_motion_where_it_collides() {
    let robot = common::panda().unwrap();
    let q = robot.default_q();
    let start = robot.ee_pose(q, 0);
    let end = Pose::new(start.position - Vec3::new(0.0, 0.0, 0.2), start.rotation);
    // A slab across the hand's way down, 10 cm below the tool frame.
    let slab = Obstacle::cuboid(
        start.position - Vec3::new(0.0, 0.0, 0.12),
        Vec3::new(0.15, 0.15, 0.01),
    );
    let world = World::new(vec![slab]);
    for device in common::devices(&robot) {
        let name = device.name();
        let result = follow(&device, world.clone(), &[path(&robot, vec![end])]);
        let fraction = result.fraction[0];
        assert!(
            (0.1..0.6).contains(&fraction),
            "{name}: covers {fraction} of the way down"
        );
        assert!(
            matches!(result.stopped[0], Some(ToolPathStop::Collision { .. })),
            "{name}: {:?}",
            result.stopped[0]
        );
        // What the next step collides with: the slab, and nothing else.
        let hits_slab = |c: &Contact| matches!(c, Contact::World { obstacle: 0, .. });
        assert!(
            !result.contacts[0].is_empty()
                && result.contacts[0]
                    .iter()
                    .all(|c| hits_slab(c) && c.distance() < 0.0),
            "{name}: {:?}",
            result.contacts[0]
        );
        // The motion it returns keeps clear of the slab.
        let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
        let samples = Trajectory::try_new(&robot, &result.paths[0], 1.0)
            .unwrap()
            .sample(250.0)
            .unwrap();
        let items = vec![0; samples.positions.len() / robot.dof()];
        let clearance = device
            .evaluate(&worlds, &items, &samples.positions, &CollisionWeights::NONE)
            .unwrap()
            .world_clearance;
        assert!(
            clearance.iter().all(|&c| c >= 0.0),
            "{name}: the motion collides"
        );
    }
}

#[test]
fn poses_out_of_reach_stop_the_motion_before_them() {
    let robot = common::panda().unwrap();
    let start = robot.ee_pose(robot.default_q(), 0);
    let far = Pose::new(start.position + Vec3::new(1.5, 0.0, 0.0), start.rotation);
    for device in common::devices(&robot) {
        let name = device.name();
        let result = follow(&device, World::default(), &[path(&robot, vec![far])]);
        let fraction = result.fraction[0];
        assert!(
            fraction > 0.0 && fraction < 1.0,
            "{name}: covers {fraction} of the way"
        );
        assert!(
            matches!(result.stopped[0], Some(ToolPathStop::Unreachable { .. })),
            "{name}: {:?}",
            result.stopped[0]
        );
        assert!(result.contacts[0].is_empty(), "{name}: nothing collides");
        // As far as it went, it went straight.
        let poses = hand_poses(&robot, &result.paths[0]);
        let reached = poses.last().unwrap().position;
        assert!(
            reached.x > start.position.x + 0.1,
            "{name}: reached {reached}"
        );
        for pose in &poses {
            assert!(off_segment(pose.position, start.position, far.position) < 1.1e-3);
        }
    }
}

#[test]
fn a_turn_in_place_keeps_the_hand_where_it_is() {
    let robot = common::panda().unwrap();
    let start = robot.ee_pose(robot.default_q(), 0);
    let turned = Pose::new(start.position, start.rotation * Quat::from_rotation_z(1.2));
    for device in common::devices(&robot) {
        let name = device.name();
        let result = follow(&device, World::default(), &[path(&robot, vec![turned])]);
        assert_eq!(result.fraction[0], 1.0, "{name}: {:?}", result.stopped[0]);
        let poses = hand_poses(&robot, &result.paths[0]);
        for pose in &poses {
            assert!(pose.position.distance(start.position) <= 1.1e-3, "{name}");
        }
        let last = poses.last().unwrap();
        assert!(
            last.rotation.angle_between(turned.rotation) < 2e-3,
            "{name}"
        );
    }
}

#[test]
fn sharp_turns_keep_the_precision_at_any_step() {
    let robot = common::panda().unwrap();
    let start = robot.ee_pose(robot.default_q(), 0);
    let corner = Pose::new(start.position - Vec3::new(0.0, 0.0, 0.06), start.rotation);
    let end = Pose::new(corner.position + Vec3::new(0.0, 0.08, 0.0), start.rotation);
    // At the default 5 mm steps and at 5 cm ones, which leave the spline far between its
    // configurations around the corner.
    let coarse = ToolPathOptions {
        step: 0.05,
        rotation_step: 0.5,
        ..ToolPathOptions::default()
    };
    for o in [ToolPathOptions::default(), coarse] {
        for device in common::devices(&robot) {
            let name = format!("{}, {} m steps", device.name(), o.step);
            let worlds = device.upload(&[World::default()]).unwrap();
            let tool_path = path(&robot, vec![corner, end]);
            let result = batchplan::follow(&device, &worlds, &[tool_path], &o).unwrap();
            assert_eq!(result.fraction[0], 1.0, "{name}: {:?}", result.stopped[0]);
            let poses = hand_poses(&robot, &result.paths[0]);
            let nearest = poses
                .iter()
                .map(|p| p.position.distance(corner.position))
                .fold(f32::INFINITY, f32::min);
            assert!(
                nearest <= o.precision,
                "{name}: passes {nearest} m from the corner"
            );
            for pose in &poses {
                let off = off_segment(pose.position, start.position, corner.position)
                    .min(off_segment(pose.position, corner.position, end.position));
                assert!(off <= o.precision + 1e-4, "{name}: {off} m off the path");
            }
        }
    }
}

#[test]
fn the_motion_through_a_singularity_stays_on_the_line() {
    // The UR5e's wrist lines up where its fifth joint passes zero, and a straight move whose ends
    // lie on either side of it needs the wrist to swing about half a turn. IK held to the
    // precision gives up short of it; given more iterations, it swings the wrist on the way
    // through. Either way the hand keeps to the line.
    let options = RobotOptions {
        collision_model: Some(CollisionModel::default()),
        ..Default::default()
    };
    let robot = Robot::load(
        common::asset("ur5e/ur_description/urdf/ur5e.urdf"),
        &options,
    )
    .unwrap();
    let start = vec![0.0, -1.2, 1.4, -1.8, 0.3, 0.4];
    let mut mirrored = start.clone();
    mirrored[4] = -0.3;
    let (from, to) = (robot.ee_pose(&start, 0), robot.ee_pose(&mirrored, 0));
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[World::default()]).unwrap();
        let tool_path = ToolPath {
            world: 0,
            start: start.clone(),
            hand: 0,
            waypoints: vec![to],
            hold: vec![],
            holding: None,
        };
        for (iterations, through) in [(300, false), (1000, true)] {
            let o = ToolPathOptions {
                iterations,
                ..ToolPathOptions::default()
            };
            let result =
                batchplan::follow(&device, &worlds, std::slice::from_ref(&tool_path), &o).unwrap();
            assert_eq!(
                result.fraction[0] == 1.0,
                through,
                "{name}, {iterations} iterations: {:?}",
                result.stopped[0]
            );
            let samples = Trajectory::try_new(&robot, &result.paths[0], 1.0)
                .unwrap()
                .sample(250.0)
                .unwrap();
            let wrist: Vec<f32> = samples
                .positions
                .chunks(robot.dof())
                .map(|q| q[5])
                .collect();
            let swing = wrist
                .iter()
                .fold(0.0f32, |m, &w| m.max((w - wrist[0]).abs()));
            assert!(
                !through || swing > 2.5,
                "{name}: the wrist swings only {swing} rad"
            );
            for q in samples.positions.chunks(robot.dof()) {
                let off = off_segment(robot.ee_pose(q, 0).position, from.position, to.position);
                assert!(off <= o.precision + 1e-4, "{name}: {off} m off the line");
            }
        }
    }
}

#[test]
fn the_hand_moves_into_a_grasp_only_where_it_may_touch() {
    let robot = common::panda().unwrap();
    let q = robot.default_q();
    let start = robot.ee_pose(q, 0);
    // The approach: 8 cm along the tool's own z, ending around a bar through the open fingers.
    let grasp = start.mul_pose(Pose::new(Vec3::new(0.0, 0.0, 0.08), Quat::IDENTITY));
    let bar = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.01, 0.06, 0.01)).placed(grasp);
    let touching = World {
        touch: vec![Touch::new(0, robot.hand_links(0))],
        ..World::new(vec![bar.clone()])
    };
    for device in common::devices(&robot) {
        let name = device.name();
        let blocked = follow(
            &device,
            World::new(vec![bar.clone()]),
            &[path(&robot, vec![grasp])],
        );
        assert!(
            blocked.fraction[0] < 1.0
                && matches!(blocked.stopped[0], Some(ToolPathStop::Collision { .. })),
            "{name}: the fingers may not enter the bar"
        );
        let allowed = follow(&device, touching.clone(), &[path(&robot, vec![grasp])]);
        assert_eq!(allowed.fraction[0], 1.0, "{name}: {:?}", allowed.stopped[0]);
    }
}

#[test]
fn every_device_follows_paths_alike() {
    let robot = common::panda().unwrap();
    let start = robot.ee_pose(robot.default_q(), 0);
    let paths: Vec<ToolPath> = (0..6)
        .map(|k| {
            let offset = Vec3::new(0.02 * k as f32, -0.03, -0.05 - 0.01 * k as f32);
            let turn = Quat::from_rotation_z(0.2 * k as f32);
            path(
                &robot,
                vec![Pose::new(start.position + offset, start.rotation * turn)],
            )
        })
        .collect();
    let results: Vec<ToolPathResult> = common::devices(&robot)
        .iter()
        .map(|d| follow(d, World::default(), &paths))
        .collect();
    for other in &results[1..] {
        assert_eq!(other.fraction, results[0].fraction);
        assert_eq!(other.stopped, results[0].stopped);
        assert_eq!(other.paths, results[0].paths);
    }
    assert!(results[0].fraction.iter().all(|&f| f == 1.0));
}

#[test]
fn paths_that_do_not_move_the_hand_are_covered() {
    let robot = common::panda().unwrap();
    let start = robot.ee_pose(robot.default_q(), 0);
    let device = Device::cpu(&robot).unwrap();
    let result = follow(&device, World::default(), &[path(&robot, vec![start])]);
    assert_eq!(result.fraction[0], 1.0);
    assert_eq!(result.stopped[0], None);
    assert!(result.paths[0].is_empty());
}

#[test]
fn motions_keep_the_joint_limits() {
    // Starts near the joints' limits and moves in every direction: joints run into their limits
    // and stay there while the others carry on.
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let device = Device::cpu(&robot).unwrap();
    let mut rng = Rng::new(9);
    let paths: Vec<ToolPath> = (0..400)
        .map(|_| {
            let start: Vec<f32> = (0..n)
                .map(|j| {
                    let (lo, hi) = (robot.lower()[j], robot.upper()[j]);
                    let near = (hi - lo) * rng.range(0.0, 0.08);
                    if rng.uniform() < 0.5 {
                        lo + near
                    } else {
                        hi - near
                    }
                })
                .collect();
            let hand = robot.ee_pose(&start, 0);
            let direction = Vec3::new(rng.normal(), rng.normal(), rng.normal()).normalize();
            ToolPath {
                start,
                ..path(
                    &robot,
                    vec![Pose::new(hand.position + direction * 0.15, hand.rotation)],
                )
            }
        })
        .collect();
    let result = follow(&device, World::default(), &paths);
    let moved: Vec<&Vec<f32>> = result.paths.iter().filter(|p| !p.is_empty()).collect();
    assert!(moved.len() > 100, "only {} motions", moved.len());
    for control_points in moved {
        Trajectory::try_new(&robot, control_points, 1.0)
            .unwrap()
            .check(&robot)
            .unwrap();
    }
}

#[test]
fn turning_while_moving_a_little_keeps_the_precision_without_stopping() {
    // Pours: up to a radian of turn about any axis while the hand moves half a millimetre to six,
    // from configurations around the default.
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let o = ToolPathOptions::default();
    let device = Device::cpu(&robot).unwrap();
    let mut rng = Rng::new(4);
    let paths: Vec<ToolPath> = (0..200)
        .map(|_| {
            let start: Vec<f32> = robot
                .default_q()
                .iter()
                .enumerate()
                .map(|(j, &v)| (v + 0.4 * rng.normal()).clamp(robot.lower()[j], robot.upper()[j]))
                .collect();
            let hand = robot.ee_pose(&start, 0);
            let random = |rng: &mut Rng| Vec3::new(rng.normal(), rng.normal(), rng.normal());
            let axis = random(&mut rng).normalize();
            let shift = random(&mut rng).normalize() * rng.range(0.0005, 0.006);
            let turn = Quat::from_axis_angle(axis, rng.range(0.2, 1.0));
            ToolPath {
                start,
                ..path(
                    &robot,
                    vec![Pose::new(hand.position + shift, turn * hand.rotation)],
                )
            }
        })
        .collect();
    let result = follow(&device, World::default(), &paths);
    let covered: Vec<usize> = (0..paths.len())
        .filter(|&i| result.fraction[i] == 1.0)
        .collect();
    assert!(covered.len() > 100, "only {} pours covered", covered.len());
    // Each motion rests at its two ends; a stop on the way rests a third time.
    let mut stops = 0;
    for &i in &covered {
        let control: Vec<&[f32]> = result.paths[i].chunks(n).collect();
        let rests = control
            .windows(3)
            .filter(|w| w[0] == w[1] && w[1] == w[2])
            .count();
        stops += rests - 2;
    }
    assert!(
        stops <= covered.len() / 50,
        "{stops} stops on the way in {} pours",
        covered.len()
    );
    for &i in covered.iter().take(20) {
        let (from, to) = (robot.ee_pose(&paths[i].start, 0), paths[i].waypoints[0]);
        for pose in hand_poses(&robot, &result.paths[i]) {
            let off = off_segment(pose.position, from.position, to.position);
            assert!(off <= o.precision + 1e-4, "pour {i}: {off} m off the line");
        }
    }
}

/// Joints held stay where the start has them, to the bit, along the whole motion, and the hand
/// follows its path with the joints left; where those cannot keep it on the path, the motion
/// stops there.
#[test]
fn held_joints_stay_where_they_start() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    // The base turned to a value a solve would round, unlike `default_q`'s 0.
    let mut at = robot.default_q().to_vec();
    at[0] = 0.4;
    let start = robot.ee_pose(&at, 0);
    let across = Pose::new(start.position + Vec3::new(0.0, 0.1, 0.0), start.rotation);
    let along = |hold: Vec<usize>| ToolPath {
        start: at.clone(),
        hold,
        ..path(&robot, vec![across])
    };
    let o = ToolPathOptions::default();
    let largest_turn = |control_points: &[f32], joints: &[usize]| {
        let turns = control_points.chunks(n).flat_map(|q| {
            joints
                .iter()
                .map(|&j| (q[j] - at[j]).abs())
                .collect::<Vec<_>>()
        });
        turns.fold(0.0, f32::max)
    };
    for device in common::devices(&robot) {
        let name = device.name();
        // Free, the base turns for a move across.
        let free = follow(&device, World::default(), &[along(vec![])]);
        assert_eq!(free.fraction[0], 1.0, "{name}: {:?}", free.stopped[0]);
        let turn = largest_turn(&free.paths[0], &[0]);
        assert!(turn > 0.01, "{name}: free, the base turns {turn} rad");
        let result = follow(&device, World::default(), &[along(vec![0])]);
        assert_eq!(result.fraction[0], 1.0, "{name}: {:?}", result.stopped[0]);
        assert_eq!(
            largest_turn(&result.paths[0], &[0]),
            0.0,
            "{name}: the base turns"
        );
        for pose in hand_poses(&robot, &result.paths[0]) {
            let off = off_segment(pose.position, start.position, across.position);
            assert!(off <= o.precision + 1e-4, "{name}: {off} m off the line");
        }
        // The wrist alone cannot keep the hand's pose along the line.
        let result = follow(&device, World::default(), &[along(vec![0, 1, 2, 3])]);
        assert!(
            result.fraction[0] < 0.1
                && matches!(result.stopped[0], Some(ToolPathStop::Unreachable { .. })),
            "{name}: {} of the way, {:?}",
            result.fraction[0],
            result.stopped[0]
        );
        let turn = largest_turn(&result.paths[0], &[0, 1, 2, 3]);
        assert_eq!(turn, 0.0, "{name}: a held joint turns");
    }
}

#[test]
fn malformed_paths_are_input_errors() {
    let robot = common::panda().unwrap();
    let start = robot.ee_pose(robot.default_q(), 0);
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let good = path(&robot, vec![start]);
    let o = ToolPathOptions::default();
    let cases = [
        (
            ToolPath {
                hand: 3,
                ..good.clone()
            },
            o.clone(),
            "hand 3",
        ),
        (
            ToolPath {
                start: vec![0.0; 3],
                ..good.clone()
            },
            o.clone(),
            "a start of 3",
        ),
        (
            ToolPath {
                world: 2,
                ..good.clone()
            },
            o.clone(),
            "world 2",
        ),
        (
            ToolPath {
                waypoints: vec![],
                ..good.clone()
            },
            o.clone(),
            "no waypoints",
        ),
        (
            ToolPath {
                hold: vec![2, 7],
                ..good.clone()
            },
            o.clone(),
            "holds joint 7 of a robot with 7",
        ),
        (
            ToolPath {
                start: robot.upper().iter().map(|v| v + 0.1).collect(),
                ..good.clone()
            },
            o.clone(),
            "leaves the joint limits",
        ),
        (
            good.clone(),
            ToolPathOptions {
                resolution: f32::INFINITY,
                ..o.clone()
            },
            "resolution",
        ),
        (
            good.clone(),
            ToolPathOptions {
                resolution: 1e-30,
                ..o.clone()
            },
            "resolution",
        ),
        (
            good.clone(),
            ToolPathOptions {
                step: f32::INFINITY,
                ..o.clone()
            },
            "steps",
        ),
        (
            good.clone(),
            ToolPathOptions {
                step: 1e-30,
                ..o.clone()
            },
            "steps",
        ),
        (
            good.clone(),
            ToolPathOptions {
                rotation_step: 1e-9,
                ..o.clone()
            },
            "steps",
        ),
        (
            good,
            ToolPathOptions {
                precision: 0.0,
                ..o
            },
            "positive",
        ),
    ];
    for (bad, options, says) in cases {
        let err = batchplan::follow(&device, &worlds, &[bad], &options).unwrap_err();
        assert!(
            matches!(&err, Error::Input(m) if m.contains(says)),
            "{err:?}"
        );
    }
}

/// How far `p` lies from the line through `point` along the unit `direction`, and how far along it.
fn about_line(p: Vec3, point: Vec3, direction: Vec3) -> (f32, f32) {
    let along = (p - point).dot(direction);
    ((p - point - direction * along).length(), along)
}

#[test]
fn a_frame_carried_about_a_screw_ends_where_trigonometry_says() {
    // A frame 10 cm from an upright line and 5 cm above its point, turned a quarter turn.
    let point = Vec3::new(0.5, -0.1, 0.2);
    let screw = Screw::about(point, Vec3::Z);
    let rotation = Quat::from_rotation_x(0.7) * Quat::from_rotation_y(-0.3);
    let frame = Pose::new(point + Vec3::new(0.1, 0.0, 0.05), rotation);
    let precision = 0.001;
    let path = screw.path(frame, FRAC_PI_2, precision).unwrap();
    let count = path.len();
    // A quarter turn right-handed about +z takes +x to +y, and the frame's axes with it, as the
    // last waypoint and `carry` have them; every waypoint before it is a turn of its own share.
    let end = point + Vec3::new(0.0, 0.1, 0.05);
    for (k, pose) in path.iter().enumerate() {
        let angle = FRAC_PI_2 * (k + 1) as f32 / count as f32;
        let at = point + Vec3::new(0.1 * angle.cos(), 0.1 * angle.sin(), 0.05);
        assert!(pose.position.distance(at) < 1e-6, "{k}: {pose:?}");
        let turned = Quat::from_rotation_z(angle) * rotation;
        assert!(pose.rotation.angle_between(turned) < 1e-5, "{k}: {pose:?}");
        let (off, along) = about_line(pose.position, point, Vec3::Z);
        assert!((off - 0.1).abs() < 1e-6 && (along - 0.05).abs() < 1e-6);
    }
    assert!(path[count - 1].position.distance(end) < 1e-6);
    assert_eq!(path[count - 1], screw.carry(frame, FRAC_PI_2));
    // The straight moves between them sag from the circle by at most the precision, at their
    // middles; one fewer would sag more: the count follows from the sag, not from a step.
    let sag = |pieces: usize| 0.1 * (1.0 - (FRAC_PI_2 / pieces as f32 / 2.0).cos());
    let ends = std::iter::once(frame).chain(path.iter().copied());
    let ends: Vec<Pose> = ends.collect();
    for pair in ends.windows(2) {
        let middle = (pair[0].position + pair[1].position) / 2.0;
        let (off, _) = about_line(middle, point, Vec3::Z);
        assert!(0.1 - off <= precision, "sags {} m", 0.1 - off);
    }
    assert!(
        sag(count) <= precision && sag(count - 1) > precision,
        "{count}"
    );
    // The other way round, it ends as far the other way.
    let back = screw.path(frame, -FRAC_PI_2, precision).unwrap();
    assert_eq!(back.len(), count);
    let other = point + Vec3::new(0.0, -0.1, 0.05);
    assert!(back[count - 1].position.distance(other) < 1e-6);
    // A frame on the line needs only what its rotation does: a quarter turn at most between
    // waypoints, so that the shortest rotation between them is the screw's.
    let on = Pose::new(point + Vec3::Z * 0.3, rotation);
    assert_eq!(screw.path(on, PI, precision).unwrap().len(), 2);
    assert_eq!(screw.path(on, -1.0, precision).unwrap().len(), 1);
    assert_eq!(screw.path(frame, 0.0, precision).unwrap(), vec![frame]);
}

#[test]
fn a_screw_with_a_pitch_advances_along_its_line_as_it_turns() {
    // A tilted line, 3 mm along it a radian, and a frame 4 cm from it turned back 3/4 of a turn.
    let direction = Vec3::new(0.2, -0.3, 1.0).normalize();
    let screw = Screw {
        point: Vec3::new(0.4, 0.1, 0.3),
        direction,
        pitch: 0.003,
    };
    let frame = Pose::new(
        screw.point + direction.any_orthonormal_vector() * 0.04,
        Quat::from_rotation_z(0.4),
    );
    let (angle, precision) = (-1.5 * PI, 2e-4);
    let path = screw.path(frame, angle, precision).unwrap();
    let count = path.len();
    let helix = |share: f32| screw.carry(frame, angle * share);
    for (k, pose) in path.iter().enumerate() {
        let share = (k + 1) as f32 / count as f32;
        let (off, along) = about_line(pose.position, screw.point, direction);
        assert!((off - 0.04).abs() < 1e-6, "{k}: {off} m from the line");
        assert!(
            (along - screw.pitch * angle * share).abs() < 1e-6,
            "{k}: {along} m along it"
        );
        assert!(pose.position.distance(helix(share).position) < 1e-6);
    }
    let (_, advanced) = about_line(path[count - 1].position, screw.point, direction);
    assert!((advanced + 0.003 * 1.5 * PI).abs() < 1e-6, "{advanced}");
    // A straight move between two waypoints strays from the helix by at most the precision, a
    // point along it measured from the helix's at the same share of the turn.
    let ends = std::iter::once(frame).chain(path.iter().copied());
    let ends: Vec<Pose> = ends.collect();
    for (k, pair) in ends.windows(2).enumerate() {
        for s in 0..=16 {
            let t = s as f32 / 16.0;
            let on_chord = pair[0].position.lerp(pair[1].position, t);
            let share = (k as f32 + t) / count as f32;
            let off = on_chord.distance(helix(share).position);
            assert!(off <= precision * 1.001, "{k} at {t}: {off} m");
        }
    }
}

#[test]
fn a_hand_follows_a_screws_waypoints_within_the_precision_of_its_arc() {
    // An upright hinge 10 cm beside the hand: the hand swings a quarter turn about it, as round a
    // lever's. Half the precision is the waypoints' sag, half how far `follow` strays from them.
    let robot = common::panda().unwrap();
    let start = robot.ee_pose(robot.default_q(), 0);
    let hinge = start.position + Vec3::new(0.1, 0.0, 0.0);
    let screw = Screw::about(hinge, Vec3::Z);
    let precision = 0.001;
    let o = ToolPathOptions {
        precision: precision / 2.0,
        ..ToolPathOptions::default()
    };
    let waypoints = screw.path(start, FRAC_PI_2, precision / 2.0).unwrap();
    let end = screw.carry(start, FRAC_PI_2);
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[World::default()]).unwrap();
        let swing = path(&robot, waypoints.clone());
        let result = batchplan::follow(&device, &worlds, &[swing], &o).unwrap();
        assert_eq!(result.fraction[0], 1.0, "{name}: {:?}", result.stopped[0]);
        let poses = hand_poses(&robot, &result.paths[0]);
        assert!(poses.len() > 50, "{name}: {} samples", poses.len());
        let mut farthest = 0.0f32;
        for pose in &poses {
            // Off the arc: its distance from the circle, and the rotation the arc has where the
            // hand is about the hinge.
            let (off, along) = about_line(pose.position, hinge, Vec3::Z);
            let along = along - (start.position - hinge).dot(Vec3::Z);
            let strays = ((off - 0.1).powi(2) + along.powi(2)).sqrt();
            assert!(strays <= precision, "{name}: {strays} m off the arc");
            let from = pose.position - hinge;
            let angle = from.y.atan2(from.x) - PI;
            let angle = (angle + PI).rem_euclid(2.0 * PI) - PI;
            let turned = Quat::from_rotation_z(angle) * start.rotation;
            let twisted = pose.rotation.angle_between(turned);
            assert!(
                twisted <= o.rotation_precision + 1e-3,
                "{name}: {twisted} rad"
            );
            farthest = farthest.max(angle);
        }
        // It swings all the way, and ends where the screw carries the hand.
        assert!(
            (farthest - FRAC_PI_2).abs() < 1e-2,
            "{name}: {farthest} rad"
        );
        let last = poses[poses.len() - 1];
        assert!(last.position.distance(end.position) <= precision, "{name}");
    }
}

#[test]
fn malformed_screws_are_input_errors() {
    let frame = Pose::new(Vec3::new(0.1, 0.0, 0.0), Quat::IDENTITY);
    let good = Screw::about(Vec3::ZERO, Vec3::Z);
    let cases = [
        (
            Screw {
                direction: Vec3::new(0.0, 0.0, 2.0),
                ..good
            },
            frame,
            1.0,
            1e-3,
            "[0, 0, 2]",
        ),
        (
            Screw {
                point: Vec3::new(f32::NAN, 0.0, 0.0),
                ..good
            },
            frame,
            1.0,
            1e-3,
            "NaN",
        ),
        (
            Screw {
                pitch: f32::INFINITY,
                ..good
            },
            frame,
            1.0,
            1e-3,
            "inf",
        ),
        (good, frame, f32::NAN, 1e-3, "angle, got NaN"),
        (
            Screw::about(Vec3::ZERO, Vec3::ZERO),
            frame,
            1.0,
            1e-3,
            "unit, got [0, 0, 0] of length 0",
        ),
        // More waypoints than can be held: a turn of an angle too large, or a precision too fine.
        (
            good,
            frame,
            1e13,
            1e-3,
            "a turn of 10000000000000 rad at a precision of 0.001 m takes",
        ),
        (good, frame, 1.0, 1e-30, "waypoints, more than can be held"),
        (good, frame, 1.0, 0.0, "got 0 m"),
        (good, frame, 1.0, -1e-3, "got -0.001 m"),
        (
            good,
            Pose::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
            1.0,
            1e-3,
            "unit rotation",
        ),
    ];
    for (screw, pose, angle, precision, says) in cases {
        let err = screw.path(pose, angle, precision).unwrap_err();
        assert!(
            matches!(&err, Error::Input(m) if m.contains(says)),
            "{err:?}"
        );
    }
}
