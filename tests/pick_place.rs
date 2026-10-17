//! Pick-and-place demonstrations: the gripper closes once and opens again, the object goes where
//! it was asked to and follows the gripper while held, and nothing collides on the way.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::datagen::{
    Handover, Origin, PickPlace, PickPlaceOptions, RecoveryOptions, handover, pick_and_place, reset_states,
};
use batchplan::rng::Rng;
use batchplan::*;
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

/// A table with its top at z = 0, a tall box beside the path, and a 4 x 6 x 6 cm box to move;
/// the box last, its target across the table.
fn scenes(count: usize) -> (Vec<World>, Vec<PickPlace>) {
    let mut rng = Rng::new(5);
    let (mut worlds, mut tasks) = (vec![], vec![]);
    for w in 0..count {
        let table = Obstacle::Cuboid {
            center: Vec3::new(0.4, 0.0, -0.02),
            half_extents: Vec3::new(0.7, 0.8, 0.02),
            rotation: Quat::IDENTITY,
        };
        let clutter = Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.55, 0.65), rng.range(-0.05, 0.05), 0.15),
            half_extents: Vec3::new(0.04, 0.04, 0.15),
            rotation: Quat::from_rotation_z(rng.range(0.0, 3.0)),
        };
        let half = Vec3::new(0.02, 0.03, 0.03);
        let object = Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.35, 0.5), rng.range(0.15, 0.3), half.z),
            half_extents: half,
            rotation: Quat::from_rotation_z(rng.range(-1.0, 1.0)),
        };
        let place = Pose {
            position: Vec3::new(rng.range(0.35, 0.5), rng.range(-0.3, -0.15), half.z),
            rotation: Quat::from_rotation_z(rng.range(-1.0, 1.0)),
        };
        worlds.push(World { obstacles: vec![table, clutter, object] });
        tasks.push(PickPlace { world: w as u32, object: 2, place });
    }
    (worlds, tasks)
}

#[test]
fn boxes_are_picked_up_carried_and_set_down_where_asked() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let (worlds, tasks) = scenes(6);
    for d in devices(&robot) {
        let demos = pick_and_place(&d, &worlds, &tasks, &PickPlaceOptions::default()).unwrap();
        assert!(demos.len() >= 4, "{}: {} of {} tasks done", d.name(), demos.len(), tasks.len());
        eprintln!("{}: {} of {} tasks done", d.name(), demos.len(), tasks.len());
        let cpu = Device::cpu(&robot).unwrap();
        for demo in &demos {
            let task = tasks.iter().find(|t| t.world == demo.world).unwrap();
            let world = &worlds[demo.world as usize];
            let frames = demo.trajectory.len();
            let q = |f: usize| &demo.trajectory.positions[f * n..(f + 1) * n];
            // The gripper starts and ends open and closes once.
            let g = &demo.gripper;
            assert!(g[0] == 1.0 && g[frames - 1] == 1.0 && g.contains(&0.0), "{}: gripper {g:?}", d.name());
            let closings = g.windows(2).filter(|w| w[0] > 0.0 && w[1] == 0.0).count();
            assert_eq!(closings, 1, "{}: the gripper closes once", d.name());
            // It moves over the default half second, never in a jump.
            let step = demo.trajectory.dt / 0.5 + 1e-5;
            assert!(g.windows(2).all(|w| (w[1] - w[0]).abs() <= step), "{}: the gripper jumps", d.name());
            // The object starts where it is and ends where it was asked to go.
            let carried = demo.carried.as_ref().unwrap();
            assert_eq!(carried.obstacle, task.object);
            assert_eq!(carried.poses[0], world.obstacles[task.object].pose());
            let end = carried.poses[frames - 1];
            assert!((end.position - task.place.position).length() < 3e-3, "{}: set down at {}", d.name(), end.position);
            assert!(end.rotation.dot(task.place.rotation).abs() > 1.0 - 1e-4, "{}: set down turned", d.name());
            // While held, it keeps its place in the gripper.
            let held: Vec<usize> = (0..frames).filter(|&f| g[f] == 0.0).collect();
            let grip = |f: usize| robot.ee_pose(q(f), 0).inverse().mul_pose(carried.poses[f]);
            let first = grip(held[0]);
            for &f in &held {
                let now = grip(f);
                assert!((now.position - first.position).length() < 1e-4, "{}: frame {f} slips", d.name());
            }
            // The arm never touches the table or the clutter, and the box, once lifted 2 cm,
            // touches nothing either.
            let mut without = world.clone();
            without.obstacles.remove(task.object);
            let uploaded = cpu.upload(std::slice::from_ref(&without)).unwrap();
            let clear =
                cpu.evaluate(&uploaded, &vec![0; frames], &demo.trajectory.positions, &CollisionWeights::NONE).unwrap();
            assert!((0..frames).all(|f| clear.collision_free(f)), "{}: the arm collides", d.name());
            let mut lifted = without.clone();
            let start_z = carried.poses[0].position.z;
            let up: Vec<usize> =
                held.iter().copied().filter(|&f| carried.poses[f].position.z > start_z + 0.02).collect();
            assert!(!up.is_empty(), "{}: the box should be lifted", d.name());
            for &f in &up {
                lifted.obstacles = without.obstacles.clone();
                lifted.obstacles.push(world.obstacles[task.object].placed(carried.poses[f]));
                let box_now = lifted.obstacles.last().unwrap();
                let nearest = without
                    .obstacles
                    .iter()
                    .map(|o| {
                        // The box's corners against each obstacle: a cheap check that it clears them.
                        (0..8)
                            .map(|c| {
                                let Obstacle::Cuboid { center, half_extents: h, rotation } = *box_now else {
                                    unreachable!()
                                };
                                let corner =
                                    Vec3::new([-h.x, h.x][c & 1], [-h.y, h.y][c >> 1 & 1], [-h.z, h.z][c >> 2 & 1]);
                                o.distance(center + rotation * corner).0
                            })
                            .fold(f32::INFINITY, f32::min)
                    })
                    .fold(f32::INFINITY, f32::min);
                assert!(nearest > 0.0, "{}: frame {f}: the carried box touches the scene ({nearest})", d.name());
            }
            // The grasp point is inside the box, 2 cm below its top.
            let Obstacle::Cuboid { half_extents: half, .. } = world.obstacles[task.object] else { unreachable!() };
            let tool = first.inverse().position;
            assert!(
                tool.x.abs() < 1e-3 && tool.y.abs() < 1e-3 && (tool.z - (half.z - 0.02)).abs() < 1e-3,
                "{}: held at {tool}",
                d.name()
            );
            // The fingers close across the box's narrower side, its x axis.
            let closing = robot.ee_pose(q(held[0]), 0).rotation * Vec3::Y;
            let narrow = world.obstacles[task.object].pose().rotation * Vec3::X;
            assert!(closing.dot(narrow).abs() > 0.999, "{}: closing along {closing}", d.name());
            // Velocities point the way the positions go, played backwards or not.
            let v = |f: usize| &demo.trajectory.velocities[f * n..(f + 1) * n];
            for f in 1..frames - 1 {
                for j in 0..n {
                    let moved = (q(f + 1)[j] - q(f - 1)[j]) / (2.0 * demo.trajectory.dt);
                    assert!(
                        moved.abs() < 0.05 || moved.signum() == v(f)[j].signum(),
                        "{}: frame {f} joint {j} moves {moved} at {}",
                        d.name(),
                        v(f)[j]
                    );
                }
            }
            // Speeds stay within the robot's limits.
            for f in 1..frames {
                for j in 0..n {
                    let speed = (q(f)[j] - q(f - 1)[j]).abs() / demo.trajectory.dt;
                    assert!(speed <= robot.max_velocity()[j] * 1.05, "{}: joint {j} at {speed} rad/s", d.name());
                }
            }
            assert!(
                demo.task.starts_with("Pick up the box and put it") && demo.task.ends_with("to the right."),
                "{}",
                demo.task
            );
        }
    }
}

#[test]
fn pick_and_place_tasks_are_checked() {
    let robot = common::panda().unwrap();
    let d = Device::cpu(&robot).unwrap();
    let (worlds, tasks) = scenes(1);
    let wrong = |t: PickPlace| pick_and_place(&d, &worlds, &[t], &PickPlaceOptions::default()).is_err();
    assert!(wrong(PickPlace { world: 1, ..tasks[0].clone() }), "a world past the end");
    assert!(wrong(PickPlace { object: 0, ..tasks[0].clone() }), "the table is too wide to grasp");
    let tilted = Pose { rotation: Quat::from_rotation_x(0.5), ..tasks[0].place };
    assert!(wrong(PickPlace { place: tilted, ..tasks[0].clone() }), "a tilted place pose");
}

#[test]
fn episodes_stay_on_one_arm_configuration() {
    // The pose above each box starts every later IK, so a far solution there (joints near
    // opposite limits) makes the arm swing around for moves of a few centimetres.
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let cpu = Device::cpu(&robot).unwrap();
    let (worlds, tasks) = scenes(64);
    let demos = pick_and_place(&cpu, &worlds, &tasks, &PickPlaceOptions::default()).unwrap();
    assert!(demos.len() >= 56, "{} of 64 episodes", demos.len());
    let mut travel: Vec<f32> = demos
        .iter()
        .map(|d| {
            let q: Vec<&[f32]> = d.trajectory.positions.chunks(n).collect();
            q.windows(2).map(|w| w[0].iter().zip(w[1]).map(|(a, b)| (a - b).abs()).sum::<f32>()).sum()
        })
        .collect();
    travel.sort_by(f32::total_cmp);
    let median = travel[travel.len() / 2];
    eprintln!("{} of 64 episodes; the joints travel {median:.1} rad in the median one", demos.len());
    // 11.3 rad on an M4 Pro; 15.8 when the pose above the box took the solution with the most
    // clearance. The threshold leaves room for other machines.
    assert!(median < 13.5, "the joints travel {median} rad in the median episode");
}

/// A table, a 4 x 6 x 6 cm box on one side and its target on the other, and across the carry
/// between them a wall, three posts, an arch or a cluster of boxes.
fn walled(count: usize) -> (Vec<World>, Vec<PickPlace>) {
    let mut rng = Rng::new(7);
    let (mut worlds, mut tasks) = (vec![], vec![]);
    for w in 0..count {
        let table = Obstacle::Cuboid {
            center: Vec3::new(0.4, 0.0, -0.02),
            half_extents: Vec3::new(0.7, 0.8, 0.02),
            rotation: Quat::IDENTITY,
        };
        let half = Vec3::new(0.02, 0.03, 0.03);
        let (pick, place) = (
            Vec3::new(rng.range(0.35, 0.6), rng.range(0.2, 0.38), half.z),
            Vec3::new(rng.range(0.35, 0.6), rng.range(-0.38, -0.2), half.z),
        );
        // Where the straight carry crosses y = 0.
        let x = pick.x + (place.x - pick.x) * pick.y / (pick.y - place.y);
        let upright = |center: Vec3, half_extents: Vec3, yaw: f32| Obstacle::Cuboid {
            center,
            half_extents,
            rotation: Quat::from_rotation_z(yaw),
        };
        let mut obstacles = vec![table];
        match w % 4 {
            0 => {
                let height = rng.range(0.08, 0.2);
                let wall = Vec3::new(rng.range(0.12, 0.2), 0.015, height);
                obstacles.push(upright(Vec3::new(x, 0.0, height), wall, rng.range(-0.3, 0.3)));
            }
            1 => {
                for k in 0..3 {
                    let height = rng.range(0.2, 0.4);
                    let at = Vec3::new(x + (k as f32 - 1.0) * 0.1, rng.range(-0.03, 0.03), height);
                    obstacles.push(upright(at, Vec3::new(0.025, 0.025, height), 0.0));
                }
            }
            2 => {
                let (height, span) = (rng.range(0.25, 0.35), rng.range(0.1, 0.16));
                for side in [-1.0, 1.0] {
                    obstacles.push(upright(
                        Vec3::new(x + side * span, 0.0, height),
                        Vec3::new(0.02, 0.02, height),
                        0.0,
                    ));
                }
                let top = Vec3::new(x, 0.0, 2.0 * height + 0.02);
                obstacles.push(upright(top, Vec3::new(span + 0.03, 0.03, 0.02), 0.0));
            }
            _ => {
                for _ in 0..4 {
                    let size = Vec3::new(rng.range(0.03, 0.06), rng.range(0.03, 0.06), rng.range(0.05, 0.18));
                    let at = Vec3::new(x + rng.range(-0.1, 0.1), rng.range(-0.06, 0.06), size.z);
                    obstacles.push(upright(at, size, rng.range(0.0, 3.0)));
                }
            }
        }
        obstacles.push(upright(pick, half, rng.range(-1.0, 1.0)));
        tasks.push(PickPlace {
            world: w as u32,
            object: obstacles.len() - 1,
            place: Pose { position: place, rotation: Quat::from_rotation_z(rng.range(-1.0, 1.0)) },
        });
        worlds.push(World { obstacles });
    }
    (worlds, tasks)
}

/// Carries bent around obstacles keep moving: a planned spline stops only at a corner no blend
/// around it clears.
#[test]
fn carries_keep_moving_around_obstacles() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let cpu = Device::cpu(&robot).unwrap();
    let (worlds, tasks) = walled(512);
    let demos = pick_and_place(&cpu, &worlds, &tasks, &PickPlaceOptions::default()).unwrap();
    assert!(demos.len() >= 480, "{} of 512 episodes", demos.len());
    // The lift, the carry and the lowering each end at rest while the gripper holds the box; any
    // more rests stop the carry on its way.
    let stopped = demos
        .iter()
        .filter(|d| {
            let closed = d.gripper.iter().position(|&g| g < 1.0).unwrap();
            let opened = d.gripper.iter().rposition(|&g| g < 1.0).unwrap();
            let speed: Vec<f32> =
                d.trajectory.velocities.chunks(n).map(|v| v.iter().fold(0.0f32, |m, x| m.max(x.abs()))).collect();
            (closed + 1..=opened).filter(|&f| speed[f] < 1e-2 && speed[f - 1] >= 1e-2).count() > 3
        })
        .count();
    // The arm keeps clear of the scene at every frame (the box, which it holds, left out).
    let scenes: Vec<World> = tasks
        .iter()
        .map(|t| {
            let mut scene = worlds[t.world as usize].clone();
            scene.obstacles.remove(t.object);
            scene
        })
        .collect();
    let uploaded = cpu.upload(&scenes).unwrap();
    let frames: Vec<u32> = demos.iter().flat_map(|d| std::iter::repeat_n(d.world, d.trajectory.len())).collect();
    let q: Vec<f32> = demos.iter().flat_map(|d| d.trajectory.positions.iter().copied()).collect();
    let clear = cpu.evaluate(&uploaded, &frames, &q, &CollisionWeights::NONE).unwrap();
    let touching = (0..frames.len()).filter(|&f| !clear.collision_free(f)).count();
    assert_eq!(touching, 0, "the arm touches the scene in {touching} frames");
    eprintln!("{} of 512 episodes; {stopped} carries stop on their way", demos.len());
    // 2 on an M4 Pro, at corners no blend clears; 43 when splines keep every stop the fitting
    // pinned.
    assert!(stopped <= 20, "{stopped} carries stop on their way");
}

/// A table, a 20-24 cm block in front of the left Panda beyond a post, a meeting pose between the
/// arms and a place in front of the right Panda, the block's long side across the arms there.
fn handovers(count: usize) -> (Vec<World>, Vec<Handover>) {
    use std::f32::consts::FRAC_PI_2;
    let mut rng = Rng::new(13);
    let (mut worlds, mut tasks) = (vec![], vec![]);
    for w in 0..count {
        let table = Obstacle::Cuboid {
            center: Vec3::new(0.45, 0.0, -0.02),
            half_extents: Vec3::new(0.5, 0.8, 0.02),
            rotation: Quat::IDENTITY,
        };
        let half = Vec3::new(rng.range(0.1, 0.12), rng.range(0.015, 0.02), rng.range(0.02, 0.03));
        let yaw = |rng: &mut Rng| Quat::from_rotation_z(FRAC_PI_2 + rng.range(-0.5, 0.5));
        let object = Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.4, 0.55), rng.range(0.25, 0.4), half.z),
            half_extents: half,
            rotation: yaw(&mut rng),
        };
        let meet =
            Pose { position: Vec3::new(0.5, rng.range(-0.05, 0.05), rng.range(0.25, 0.35)), rotation: yaw(&mut rng) };
        let place =
            Pose { position: Vec3::new(rng.range(0.4, 0.55), rng.range(-0.4, -0.25), half.z), rotation: yaw(&mut rng) };
        // A post where the left hand passes on its way down to the block, so reaching it takes a
        // search.
        let wall = Obstacle::Cuboid {
            center: Vec3::new(0.32, 0.35, 0.225),
            half_extents: Vec3::new(0.01, 0.01, 0.225),
            rotation: Quat::IDENTITY,
        };
        worlds.push(World { obstacles: vec![table, object, wall] });
        tasks.push(Handover { world: w as u32, object: 1, from: 0, to: 1, meet, place });
    }
    (worlds, tasks)
}

/// One hand picks a block up, the other takes it where they meet and sets it down: each gripper
/// closes once and opens once, in that order; the block rides in one hand and then the other
/// without slipping and ends where it goes; the waiting arm stays put; and neither arm touches
/// the scene or the other.
#[test]
fn blocks_are_handed_from_one_hand_to_the_other() {
    let robot = common::two_pandas().unwrap();
    let n = robot.dof();
    let (left, right) = (robot.hand_joints(0), robot.hand_joints(1));
    let (worlds, tasks) = handovers(8);
    for d in devices(&robot) {
        let demos = handover(&d, &worlds, &tasks, &PickPlaceOptions::default()).unwrap();
        eprintln!("{}: {} of {} handovers", d.name(), demos.len(), tasks.len());
        assert!(demos.len() >= 6, "{}: {} of {} handovers", d.name(), demos.len(), tasks.len());
        for demo in &demos {
            let task = &tasks[demo.world as usize];
            let frames = demo.trajectory.len();
            let q = |f: usize| &demo.trajectory.positions[f * n..(f + 1) * n];
            let g = |f: usize, h: usize| demo.gripper[f * 2 + h];
            // Each gripper closes once and opens once: giver, taker, giver, taker.
            let first = |h: usize, value: f32, after: usize| (after..frames).find(|&f| g(f, h) == value).unwrap();
            let giver_closed = first(0, 0.0, 0);
            let taker_closed = first(1, 0.0, giver_closed);
            let giver_open = first(0, 1.0, taker_closed);
            let taker_open = first(1, 1.0, giver_open);
            for h in 0..2 {
                let changes = (1..frames).filter(|&f| (g(f, h) < 1.0) != (g(f - 1, h) < 1.0)).count();
                assert_eq!(changes, 2, "{}: hand {h} should close and open once", d.name());
            }
            assert!(giver_closed < taker_closed && taker_closed < giver_open && giver_open < taker_open);
            // The block rides without slipping, in the giver's hand and then in the taker's.
            let carried = &demo.carried.as_ref().unwrap().poses;
            let grip = |f: usize, h: usize| robot.ee_pose(q(f), h).inverse().mul_pose(carried[f]);
            // (The block moves with a hand from the frame after the one its gripper closes on, or
            // after the other gripper opens.)
            for (h, frames) in [(0, giver_closed + 1..taker_closed), (1, giver_open + 1..taker_open)] {
                let at = grip(frames.start, h);
                for f in frames {
                    let now = grip(f, h);
                    assert!((now.position - at.position).length() < 1e-4, "{}: the block slips in hand {h}", d.name());
                }
            }
            let end = carried[frames - 1];
            assert!((end.position - task.place.position).length() < 3e-3, "{}: the block ends off its place", d.name());
            // The right arm waits until the left has picked the block up; the left waits from the
            // time the block is set down.
            // (Timed samples blend equal control points, exact up to rounding.)
            let still = |joints: &[usize], frames: std::ops::Range<usize>, at: usize| {
                frames.into_iter().all(|f| joints.iter().all(|&j| (q(f)[j] - q(at)[j]).abs() < 1e-5))
            };
            assert!(still(&right, 0..giver_closed, 0), "{}: the right arm moves early", d.name());
            assert!(still(&left, taker_open..frames, taker_open), "{}: the left arm moves late", d.name());
            // Neither arm touches the scene (the block left out) or the other.
            let mut scene = worlds[demo.world as usize].clone();
            scene.obstacles.remove(task.object);
            let cpu = Device::cpu(&robot).unwrap();
            let uploaded = cpu.upload(std::slice::from_ref(&scene)).unwrap();
            let clear =
                cpu.evaluate(&uploaded, &vec![0; frames], &demo.trajectory.positions, &CollisionWeights::NONE).unwrap();
            assert!((0..frames).all(|f| clear.collision_free(f)), "{}: an arm touches something", d.name());
        }
    }
}

/// The smallest distance from a corner of the cuboid `block` to any of `obstacles`: a cheap check
/// that it clears them.
fn corner_clearance(block: &Obstacle, obstacles: &[Obstacle]) -> f32 {
    let Obstacle::Cuboid { center, half_extents: h, rotation } = *block else { unreachable!() };
    (0..8)
        .map(|c| center + rotation * Vec3::new([-h.x, h.x][c & 1], [-h.y, h.y][c >> 1 & 1], [-h.z, h.z][c >> 2 & 1]))
        .flat_map(|corner| obstacles.iter().map(move |o| o.distance(corner).0))
        .fold(f32::INFINITY, f32::min)
}

/// Reset states are a handover's samples with joint noise, at rest: a block in one closed hand
/// keeps its grip in that hand, a block no hand holds stays where the episode has it, no state has
/// both hands on the block, and neither the arms nor a block in a hand touch the scene.
#[test]
fn reset_states_carry_the_block_in_the_hand_holding_it() {
    let robot = common::two_pandas().unwrap();
    let n = robot.dof();
    let (worlds, tasks) = handovers(4);
    let cpu = Device::cpu(&robot).unwrap();
    let demos = handover(&cpu, &worlds, &tasks, &PickPlaceOptions::default()).unwrap();
    let options = RecoveryOptions { per_trajectory: 64, phase: (0.0, 1.0), ..Default::default() };
    for d in devices(&robot) {
        let states = reset_states(&d, &worlds, &demos, &options).unwrap();
        let (mut in_hand, mut loose, mut moved) = ([0; 2], 0, 0);
        for s in &states {
            let Origin::Recovery { parent, phase } = s.origin else { panic!("{}: a nominal state", d.name()) };
            let demo = &demos[parent];
            let t = &demo.trajectory;
            assert_eq!(s.world, demo.world);
            assert!(s.trajectory.len() == 1 && s.trajectory.velocities.iter().all(|&v| v == 0.0));
            let q = &s.trajectory.positions;
            let block = s.carried.as_ref().unwrap().poses[0];
            let poses = &demo.carried.as_ref().unwrap().poses;
            let task = &tasks[demo.world as usize];
            let mut scene = worlds[demo.world as usize].clone();
            let object = scene.obstacles.remove(task.object).placed(block);
            match (s.gripper[0] == 0.0, s.gripper[1] == 0.0) {
                (true, true) => panic!("{}: both hands hold the block", d.name()),
                (false, false) => {
                    loose += 1;
                    assert!(block == poses[0] || block == poses[poses.len() - 1], "{}: the block moved", d.name());
                    scene.obstacles.push(object);
                }
                (left, _) => {
                    let h = usize::from(!left);
                    in_hand[h] += 1;
                    // The grip where the hand last holds the block alone, held since its gripper closed.
                    let alone = |f: usize| (demo.gripper[f * 2 + h] == 0.0) && (demo.gripper[f * 2 + 1 - h] != 0.0);
                    let f = (0..t.len()).rev().find(|&f| alone(f)).unwrap();
                    let grip = robot.ee_pose(&t.positions[f * n..(f + 1) * n], h).inverse().mul_pose(poses[f]);
                    let now = robot.ee_pose(q, h).inverse().mul_pose(block);
                    assert!((now.position - grip.position).length() < 1e-4, "{}: the block slips", d.name());
                    let nominal = poses[(phase * (t.len() - 1) as f32).round() as usize];
                    moved += usize::from((block.position - nominal.position).length() > 1e-3);
                    let clear = corner_clearance(&object, &scene.obstacles);
                    assert!(clear > 0.0, "{}: the block in hand {h} touches the scene ({clear})", d.name());
                }
            }
            let uploaded = cpu.upload(std::slice::from_ref(&scene)).unwrap();
            let clear = cpu.evaluate(&uploaded, &[0], q, &CollisionWeights::NONE).unwrap();
            assert!(clear.collision_free(0), "{}: an arm touches something", d.name());
        }
        eprintln!("{}: {} states: {in_hand:?} in each hand, {loose} loose", d.name(), states.len());
        assert!(in_hand[0] > 0 && in_hand[1] > 0 && loose > 0, "{}: {in_hand:?} in hand, {loose} loose", d.name());
        assert!(moved > 0, "{}: the noise should move the blocks in hand", d.name());
        // None where both hands hold the block, though the same episode held in one hand gives some.
        let held_by = |hands: [f32; 2]| {
            let mut demo = demos[0].clone();
            demo.gripper = hands.repeat(demo.trajectory.len());
            reset_states(&d, &worlds, &[demo], &options).unwrap().len()
        };
        let (one, both) = (held_by([0.0, 1.0]), held_by([0.0, 0.0]));
        assert!(one > 0 && both == 0, "{}: {one} states held in one hand, {both} in both", d.name());
    }
}
