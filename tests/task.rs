//! Whole tasks in one call (`pick`, `place`): every candidate that gets through comes back, quickest
//! first; a task none of whose candidates does says where and why each stopped; `grasps` and
//! `place_poses` propose candidates.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::PI;

use batchplan::rng::Rng;
use batchplan::{
    Device, Error, GraspOptions, Held, IkOptions, IkProblem, Obstacle, Pick, Place, Pose, Robot,
    RobotOptions, SdfGrid, TaskFailure, TaskOptions, TaskPlan, TaskStep, ToolPathStop, Touch,
    Trajectory, World,
};
use glam::{Mat3, Quat, Vec3};

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

const TABLE: usize = 0;
const BOX: usize = 1;
const HALF: Vec3 = Vec3::new(0.02, 0.03, 0.03);

/// A table with its top at z = 0 and a 4 x 6 x 6 cm box on it, then `more`.
fn scene(more: Vec<Obstacle>) -> World {
    let mut obstacles = vec![
        Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02)),
        Obstacle::cuboid(Vec3::new(0.45, 0.2, HALF.z), HALF),
    ];
    obstacles.extend(more);
    World::new(obstacles)
}

/// A grasp facing `toward`, closing along `closing`, 2 cm in from the box's face.
fn grasp(toward: Vec3, closing: Vec3) -> Pose {
    let rotation = Quat::from_mat3(&Mat3::from_cols(closing.cross(toward), closing, toward));
    let surface = HALF.dot(toward.abs());
    Pose::new(-toward * (surface - 0.02), rotation)
}

/// From above, closing along the box's x axis, along it the other way round, and along its y axis.
fn top() -> Pose {
    grasp(-Vec3::Z, Vec3::X)
}
fn top_reversed() -> Pose {
    grasp(-Vec3::Z, -Vec3::X)
}
fn top_across() -> Pose {
    grasp(-Vec3::Z, Vec3::Y)
}

/// A post as tall as the box, 2 cm off its +x face, where an open finger closing along x comes
/// down and below the palm of a grasp closing along y.
fn post() -> Obstacle {
    Obstacle::cuboid(
        Vec3::new(0.45 + HALF.x + 0.02 + 0.005, 0.2, HALF.z),
        Vec3::new(0.005, 0.02, HALF.z),
    )
}

/// A slab over the box, its middle at height `z`.
fn slab(z: f32) -> Obstacle {
    Obstacle::cuboid(Vec3::new(0.45, 0.2, z), Vec3::new(0.15, 0.15, 0.02))
}

fn pick_from(robot: &Robot, grasps: Vec<Pose>) -> Pick {
    Pick {
        support: vec![TABLE],
        ..Pick::new(0, robot.default_q(), BOX, grasps)
    }
}

/// The plan's motions keep the limits and join: transit from the start, approach from the
/// transit's end to the grasp, departure from there by `lift`.
fn check_pick(robot: &Robot, start: &[f32], plan: &TaskPlan, at: Pose, lift: Vec3) {
    let n = robot.dof();
    let first = |t: &Trajectory| t.control_points[..n].to_vec();
    let last = |t: &Trajectory| t.control_points[t.control_points.len() - n..].to_vec();
    for t in [&plan.transit, &plan.approach, &plan.depart] {
        t.check(robot).unwrap();
    }
    assert_eq!(first(&plan.transit), start);
    assert_eq!(last(&plan.transit), first(&plan.approach));
    assert_eq!(last(&plan.approach), first(&plan.depart));
    let grasped = robot.ee_pose(&last(&plan.approach), 0);
    assert!(
        grasped.position.distance(at.position) < 2e-3,
        "{grasped:?} for {at:?}"
    );
    assert!(grasped.rotation.angle_between(at.rotation) < 2e-2);
    let lifted = robot.ee_pose(&last(&plan.depart), 0);
    assert!(lifted.position.distance(at.position + lift) < 2e-3);
}

fn pick_in(
    device: &Device,
    world: World,
    task: Pick,
    o: &TaskOptions,
) -> Result<Vec<TaskPlan>, TaskFailure> {
    let worlds = device.upload(&[world]).unwrap();
    batchplan::pick(device, &worlds, &[task], o)
        .unwrap()
        .outcomes
        .remove(0)
}

#[test]
fn the_quickest_grasp_that_gets_through_wins() {
    let robot = panda();
    let start = robot.default_q().to_vec();
    let o = TaskOptions::default();
    for device in devices(&robot) {
        let name = device.name();
        // The post is where a finger of either grasp closing along x goes; the one closing along
        // y gets through, alone.
        let task = pick_from(&robot, vec![top(), top_reversed(), top_across()]);
        let plans = pick_in(&device, scene(vec![post()]), task.clone(), &o).unwrap();
        assert_eq!(plans.len(), 1, "{name}");
        let plan = &plans[0];
        assert_eq!(plan.candidate, 2, "{name}");
        let at = Pose::new(Vec3::new(0.45, 0.2, HALF.z), Quat::IDENTITY).mul_pose(top_across());
        check_pick(&robot, &start, plan, at, task.lift);
        let held = plan
            .held
            .as_ref()
            .expect("a pick leaves the hand holding the box");
        assert_eq!(held.grasp, top_across());
        assert_eq!(held.object, Obstacle::cuboid(Vec3::ZERO, HALF));
        // Without the post every grasp gets through, quickest first, and the first is not the
        // quickest.
        let grasps = vec![top_across(), top(), top_reversed()];
        let all = pick_in(
            &device,
            scene(vec![]),
            pick_from(&robot, grasps.clone()),
            &o,
        )
        .unwrap();
        assert_eq!(all.len(), 3, "{name}");
        assert!(
            all.windows(2).all(|w| w[0].duration() <= w[1].duration()),
            "{name}"
        );
        assert_ne!(all[0].candidate, 0, "{name}");
        // Each grasp alone takes as long, but for IK's seeds, which each problem draws after
        // those before it in the batch.
        for plan in &all {
            let alone = pick_in(
                &device,
                scene(vec![]),
                pick_from(&robot, vec![grasps[plan.candidate]]),
                &o,
            )
            .unwrap()[0]
                .duration();
            assert!(
                (alone - plan.duration()).abs() < 0.02 * alone,
                "{name}: grasp {}: {} batched, {alone} alone",
                plan.candidate,
                plan.duration()
            );
        }
    }
}

#[test]
fn a_task_none_of_whose_grasps_gets_through_says_where_and_why_each_stopped() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    let tops = || pick_from(&robot, vec![top(), top_reversed(), top_across()]);
    let stops = |f: &TaskFailure, step: TaskStep, why: fn(&ToolPathStop) -> bool| {
        f.candidates
            .iter()
            .all(|c| c.step == step && c.why.as_ref().is_some_and(why))
    };
    let collides = |s: &ToolPathStop| matches!(s, ToolPathStop::Collision { .. });
    // A slab where the palm would be at the grasp.
    let f = pick_in(&device, scene(vec![slab(0.09)]), tops(), &o).unwrap_err();
    assert!(stops(&f, TaskStep::Reach, collides), "{f}");
    assert_eq!(
        f.to_string(),
        "none of 3 candidates got through: 3 at the grasp or release (3 colliding)"
    );
    // A box beyond the arm's reach.
    let mut far = scene(vec![]);
    far.obstacles[BOX] = Obstacle::cuboid(Vec3::new(1.4, 0.2, HALF.z), HALF);
    let f = pick_in(&device, far, tops(), &o).unwrap_err();
    let out = |s: &ToolPathStop| matches!(s, ToolPathStop::Unreachable { .. });
    assert!(stops(&f, TaskStep::Reach, out), "{f}");
    // From 40 cm up, the hand passes a small block high over the box that the arm clears at the
    // grasp: the moves in stop at it, unless they may stop short.
    let block = Obstacle::cuboid(Vec3::new(0.45, 0.2, 0.38), Vec3::new(0.02, 0.02, 0.01));
    let from_high = || Pick {
        approach: 0.4,
        ..tops()
    };
    let f = pick_in(&device, scene(vec![block.clone()]), from_high(), &o).unwrap_err();
    assert!(stops(&f, TaskStep::Approach, collides), "{f}");
    let shorter = TaskOptions {
        min_fraction: 0.05,
        ..TaskOptions::default()
    };
    let plans = pick_in(&device, scene(vec![block]), from_high(), &shorter).unwrap();
    let n = robot.dof();
    let begins = robot.ee_pose(&plans[0].approach.control_points[..n], 0);
    let transit = &plans[0].transit.control_points;
    let arrives = robot.ee_pose(&transit[transit.len() - n..], 0);
    assert!(begins.position.distance(arrives.position) < 1e-5);
    let grasp_height = HALF.z * 2.0 - 0.02;
    let moved = begins.position.z - grasp_height;
    assert!(moved > 0.02 && moved < 0.38, "the move in is {moved} m");
    // Lifting the box off a table it may not touch, and into a small lid just over the box,
    // between the fingers and below the palm.
    let unsupported = Pick {
        support: vec![],
        ..tops()
    };
    let f = pick_in(&device, scene(vec![]), unsupported, &o).unwrap_err();
    assert!(stops(&f, TaskStep::Depart, collides), "{f}");
    let lid = Obstacle::cuboid(Vec3::new(0.45, 0.2, 0.068), Vec3::new(0.01, 0.015, 0.004));
    let f = pick_in(&device, scene(vec![lid.clone()]), tops(), &o).unwrap_err();
    assert!(stops(&f, TaskStep::Depart, collides), "{f}");
    // Every grasp of the box: those from its sides stop at the grasp, those from above lifting.
    let all = batchplan::grasps(
        &Obstacle::cuboid(Vec3::ZERO, HALF),
        &GraspOptions::default(),
    )
    .unwrap();
    let f = pick_in(&device, scene(vec![lid]), pick_from(&robot, all), &o).unwrap_err();
    assert!(
        f.at(TaskStep::Reach) > 0 && f.at(TaskStep::Depart) > 0,
        "{f}"
    );
    assert_eq!(f.step(), TaskStep::Depart);
    // A start inside a box.
    let hand = robot.ee_pose(robot.default_q(), 0).position;
    let cage = Obstacle::cuboid(hand, Vec3::splat(0.05));
    let f = pick_in(&device, scene(vec![cage]), tops(), &o).unwrap_err();
    assert!(stops(&f, TaskStep::Start, collides), "{f}");
    assert_eq!(f.to_string(), "the start collides");
}

#[test]
fn a_held_object_sits_in_the_hand_as_its_grasp_holds_it() {
    let robot = panda();
    let object = Pose::new(Vec3::new(0.4, 0.1, 0.2), Quat::from_rotation_z(0.7));
    // From above, and from two sides: grasps that are not their own inverse.
    for grasp in [top(), grasp(Vec3::X, Vec3::Y), grasp(-Vec3::Y, Vec3::Z)] {
        let held = Held {
            hand: 0,
            object: Obstacle::cuboid(Vec3::ZERO, HALF),
            grasp,
        };
        let attached = held.attachment(&robot).unwrap();
        assert_eq!(attached.link, "ee_link");
        assert_eq!(attached.touch_links, robot.hand_links(0));
        let Obstacle::Cuboid {
            center,
            half_extents,
            rotation,
        } = attached.shapes[0]
        else {
            panic!("a box stays a box: {:?}", attached.shapes[0]);
        };
        assert_eq!(half_extents, HALF);
        // With the hand where the grasp holds the object, the shape is where the object is.
        let at = object.mul_pose(grasp).mul_pose(Pose::new(center, rotation));
        assert!(at.position.distance(object.position) < 1e-5, "{at:?}");
        assert!(at.rotation.angle_between(object.rotation) < 1e-4, "{at:?}");
    }
}

#[test]
fn the_fingers_may_touch_what_they_grasp_and_the_palm_may_not() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    // A box as wide as the fingers open, which they touch closing across it.
    let wide = Vec3::new(HALF.x, 0.04, HALF.z);
    let mut world = scene(vec![]);
    world.obstacles[BOX] = Obstacle::cuboid(Vec3::new(0.45, 0.2, wide.z), wide);
    let plans = pick_in(&device, world, pick_from(&robot, vec![top_across()]), &o).unwrap();
    plans[0].approach.check(&robot).unwrap();
    // A tall box held 2 cm into its top, and 5.5 cm in, deeper than the fingers reach below the
    // palm.
    let tall = Vec3::new(HALF.x, HALF.y, 0.08);
    let mut world = scene(vec![]);
    world.obstacles[BOX] = Obstacle::cuboid(Vec3::new(0.45, 0.2, tall.z), tall);
    let at_depth = |depth: f32| Pose::new(Vec3::Z * (tall.z - depth), top_across().rotation);
    let task = |depth| pick_from(&robot, vec![at_depth(depth)]);
    pick_in(&device, world.clone(), task(0.02), &o).unwrap();
    let f = pick_in(&device, world, task(0.055), &o).unwrap_err();
    assert_eq!(f.step(), TaskStep::Reach, "{f}");
}

#[test]
fn a_picked_box_is_set_down_where_there_is_room() {
    let robot = panda();
    for device in devices(&robot) {
        let name = device.name();
        let o = TaskOptions::default();
        // Another box stands where the first pose would set it down.
        let first = Pose::new(Vec3::new(0.45, -0.2, HALF.z), Quat::from_rotation_z(0.3));
        let second = Pose::new(Vec3::new(0.5, -0.05, HALF.z), Quat::from_rotation_z(-0.3));
        let other = Obstacle::cuboid(first.position + Vec3::Z * 0.05, Vec3::splat(0.05));
        let world = scene(vec![other]);
        // From above, a centimeter off the middle: a grasp that is not its own inverse.
        let off_middle = Pose::new(top().position + Vec3::X * 0.01, top().rotation);
        let picked = pick_in(
            &device,
            world.clone(),
            pick_from(&robot, vec![off_middle]),
            &o,
        )
        .unwrap();
        let picked = &picked[0];
        let held = picked.held.clone().unwrap();
        // The box goes with the hand: the world it is set down in no longer holds it.
        let mut after = world;
        after.remove(BOX).unwrap();
        let after = device.upload(&[after]).unwrap();
        let task = Place {
            support: vec![TABLE],
            ..Place::new(0, picked.end(), held.clone(), vec![first, second])
        };
        let placed = batchplan::place(&device, &after, std::slice::from_ref(&task), &o).unwrap();
        let plans = placed.outcomes[0].as_ref().unwrap();
        assert_eq!(plans.len(), 1, "{name}");
        let plan = &plans[0];
        assert_eq!(plan.candidate, 1, "{name}");
        assert_eq!(
            plan.placed,
            Some(Obstacle::Cuboid {
                center: second.position,
                half_extents: HALF,
                rotation: second.rotation,
            })
        );
        assert!(plan.held.is_none());
        let n = robot.dof();
        let last = |t: &Trajectory| t.control_points[t.control_points.len() - n..].to_vec();
        let at = second.mul_pose(held.grasp);
        let released = robot.ee_pose(&last(&plan.approach), 0);
        assert!(released.position.distance(at.position) < 2e-3, "{name}");
        let lowered = robot.ee_pose(&plan.approach.control_points[..n], 0);
        assert!(
            lowered.position.distance(at.position - task.lower) < 2e-3,
            "{name}"
        );
        let backed = robot.ee_pose(&last(&plan.depart), 0);
        let back = at.position - at.rotation * Vec3::Z * task.retreat;
        assert!(backed.position.distance(back) < 2e-3, "{name}");
        for t in [&plan.transit, &plan.approach, &plan.depart] {
            t.check(&robot).unwrap();
        }
        // Without the table to set it on, it cannot be let go of there.
        let unsupported = Place {
            support: vec![],
            ..task
        };
        let failure = batchplan::place(&device, &after, &[unsupported], &o).unwrap();
        let failure = failure.outcomes[0].as_ref().unwrap_err();
        assert_eq!(failure.step(), TaskStep::Reach, "{name}: {failure}");
    }
}

#[test]
fn every_grasp_that_picks_the_box_up_can_be_tried_for_the_place() {
    // The quickest pick and place together: one place task per grasp that got through, batched.
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    let world = scene(vec![]);
    let grasps = vec![top(), top_reversed(), top_across()];
    let picks = pick_in(&device, world.clone(), pick_from(&robot, grasps), &o).unwrap();
    assert_eq!(picks.len(), 3);
    let mut after = world;
    after.remove(BOX).unwrap();
    let after = device.upload(&[after]).unwrap();
    let to = Pose::new(Vec3::new(0.45, -0.15, HALF.z), Quat::IDENTITY);
    let object = Obstacle::cuboid(Vec3::ZERO, HALF);
    let places: Vec<Place> = picks
        .iter()
        .map(|p| Place {
            support: vec![TABLE],
            ..Place::new(
                0,
                p.end(),
                p.held.clone().unwrap(),
                batchplan::place_poses(&object, to, 8),
            )
        })
        .collect();
    let placed = batchplan::place(&device, &after, &places, &o).unwrap();
    let totals: Vec<f32> = picks
        .iter()
        .zip(&placed.outcomes)
        .map(|(p, placed)| p.duration() + placed.as_ref().unwrap()[0].duration())
        .collect();
    assert_eq!(totals.len(), 3);
    eprintln!("pick and place, by grasp: {totals:?} s");
}

#[test]
fn each_hand_lifts_and_sets_down_what_it_holds() {
    // Two Pandas, each picking up the box in front of it from the table, in one call, then
    // setting both down in another.
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    let boxes = [
        Vec3::new(0.45, 0.35, HALF.z),
        Vec3::new(0.45, -0.35, HALF.z),
    ];
    let world = World::new(vec![
        Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.9, 0.02)),
        Obstacle::cuboid(boxes[0], HALF),
        Obstacle::cuboid(boxes[1], HALF),
    ]);
    let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
    let tasks: Vec<Pick> = (0..2)
        .map(|hand| Pick {
            hand,
            support: vec![TABLE],
            ..Pick::new(0, robot.default_q(), 1 + hand, vec![top(), top_across()])
        })
        .collect();
    let picked = batchplan::pick(&device, &worlds, &tasks, &o).unwrap();
    for (hand, outcome) in picked.outcomes.iter().enumerate() {
        let plan = &outcome.as_ref().unwrap()[0];
        assert_eq!(plan.held.as_ref().unwrap().hand, hand);
    }
    // Each sets its box down 15 cm further out, from where its own pick left the robot.
    let mut after = world;
    after.remove(2).unwrap();
    after.remove(1).unwrap();
    let after = device.upload(&[after]).unwrap();
    let places: Vec<Place> = picked
        .outcomes
        .iter()
        .enumerate()
        .map(|(hand, outcome)| {
            let plan = &outcome.as_ref().unwrap()[0];
            let to = Pose::new(boxes[hand] + Vec3::X * 0.15, Quat::IDENTITY);
            Place {
                support: vec![TABLE],
                ..Place::new(0, plan.end(), plan.held.clone().unwrap(), vec![to])
            }
        })
        .collect();
    let placed = batchplan::place(&device, &after, &places, &o).unwrap();
    for outcome in &placed.outcomes {
        outcome.as_ref().unwrap();
    }
}

#[test]
fn a_hand_without_fingers_touches_what_it_grasps_with_the_whole_hand() {
    // The UR5e's tool frames below its wrist have no spheres: the wrist is the hand, and holding
    // the tool frame a centimeter into a box puts the wrist's spheres into it.
    let options = RobotOptions {
        default_q: Some(vec![0.0, -1.57, 1.57, -1.57, -1.57, 0.0]),
        ..Default::default()
    };
    let robot = Robot::load(
        common::asset("ur5e/ur_description/urdf/ur5e.urdf"),
        &options,
    )
    .unwrap();
    assert!(
        robot.finger_links(0).is_empty(),
        "{:?}",
        robot.finger_links(0)
    );
    let device = Device::cpu(&robot).unwrap();
    let half = Vec3::new(0.03, 0.03, 0.04);
    let world = World::new(vec![
        Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02)),
        Obstacle::cuboid(Vec3::new(0.4, 0.2, half.z), half),
    ]);
    // The tool frame a centimeter into the box's top, its z axis pointing down.
    let into_top = Pose::new(Vec3::Z * (half.z - 0.01), Quat::from_rotation_x(PI));
    let task = Pick {
        support: vec![TABLE],
        ..Pick::new(0, robot.default_q(), BOX, vec![into_top])
    };
    pick_in(&device, world, task, &TaskOptions::default()).unwrap();
}

#[test]
fn a_support_listed_after_the_object_is_renumbered_with_it() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    // The box first, the table after it.
    let world = World::new(vec![
        Obstacle::cuboid(Vec3::new(0.45, 0.2, HALF.z), HALF),
        Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02)),
    ]);
    let task = Pick {
        support: vec![1],
        ..Pick::new(0, robot.default_q(), 0, vec![top()])
    };
    pick_in(&device, world, task, &TaskOptions::default()).unwrap();
}

#[test]
fn the_fingers_may_touch_what_they_let_go_of() {
    // A box as wide as the fingers open, picked and set down 20 cm along: backing off, the
    // fingers slide along it.
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    let wide = Vec3::new(HALF.x, 0.04, HALF.z);
    let mut world = scene(vec![]);
    world.obstacles[BOX] = Obstacle::cuboid(Vec3::new(0.45, 0.2, wide.z), wide);
    let picked = pick_in(
        &device,
        world.clone(),
        pick_from(&robot, vec![top_across()]),
        &o,
    )
    .unwrap();
    let mut after = world;
    after.remove(BOX).unwrap();
    let after = device.upload(&[after]).unwrap();
    let to = Pose::new(Vec3::new(0.45, 0.0, wide.z), Quat::IDENTITY);
    let place = Place {
        support: vec![TABLE],
        ..Place::new(
            0,
            picked[0].end(),
            picked[0].held.clone().unwrap(),
            vec![to],
        )
    };
    let placed = batchplan::place(&device, &after, &[place], &o).unwrap();
    placed.outcomes[0].as_ref().unwrap();
}

#[test]
fn ik_takes_the_solution_nearest_the_start() {
    // With one candidate, a task's IK is the same batch as solving its pose alone. From starts
    // around the ready pose, the first whose first successful seed is not the nearest: the task
    // keeps the nearest.
    let robot = panda();
    let n = robot.dof();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    let world = scene(vec![]);
    let mut rng = Rng::new(17);
    let mut around = |q: &[f32]| -> Vec<f32> {
        (0..n)
            .map(|j| (q[j] + rng.range(-0.6, 0.6)).clamp(robot.lower()[j], robot.upper()[j]))
            .collect()
    };
    // The solution nearest `start` of `at` alone, if the first successful seed is another.
    let nearest_not_first = |worlds: &[World], at: Pose, start: &[f32]| {
        let uploaded = device.upload(worlds).unwrap();
        let problem = IkProblem {
            world: 0,
            targets: vec![Some(at)],
            seed: Some(start.to_vec()),
        };
        let solved =
            batchplan::solve_ik(&device, &uploaded, &[problem], &IkOptions::default()).unwrap();
        let first = (0..solved.seeds).find(|&k| solved.success[k])?;
        let nearest = solved.nearest(0, start)?.to_vec();
        (solved.solution(first) != nearest.as_slice()).then_some(nearest)
    };
    let last = |t: &Trajectory| t.control_points[t.control_points.len() - n..].to_vec();
    let at = world.obstacles[BOX].pose().mul_pose(top());
    let mut touching = world.clone();
    touching.touch.push(Touch::new(BOX, robot.finger_links(0)));
    let (picked, nearest) = (0..100)
        .find_map(|_| {
            let start = around(robot.default_q());
            let nearest = nearest_not_first(std::slice::from_ref(&touching), at, &start)?;
            let task = Pick {
                start,
                ..pick_from(&robot, vec![top()])
            };
            let plans = pick_in(&device, world.clone(), task, &o).ok()?;
            Some((plans[0].clone(), nearest))
        })
        .expect("a start whose first successful seed is not the nearest");
    assert_eq!(last(&picked.approach), nearest);
    // And where a place lets go.
    let mut after = world;
    after.remove(BOX).unwrap();
    let uploaded = device.upload(std::slice::from_ref(&after)).unwrap();
    let to = Pose::new(Vec3::new(0.45, -0.1, HALF.z), Quat::IDENTITY);
    let held = picked.held.clone().unwrap();
    let release = to.mul_pose(held.grasp);
    let (released, nearest) = (0..100)
        .find_map(|_| {
            let start = around(picked.end());
            let nearest = nearest_not_first(std::slice::from_ref(&after), release, &start)?;
            let place = Place {
                support: vec![TABLE],
                ..Place::new(0, start, held.clone(), vec![to])
            };
            let placed = batchplan::place(&device, &uploaded, &[place], &o).unwrap();
            let plans = placed.outcomes[0].as_ref().ok()?;
            Some((last(&plans[0].approach), nearest))
        })
        .expect("a start whose first successful seed is not the nearest");
    assert_eq!(released, nearest);
}

#[test]
fn grasps_fit_between_the_fingers_and_face_into_the_object() {
    let o = GraspOptions::default();
    let shapes = [
        (Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.03, 0.1)), 16),
        (Obstacle::sphere(Vec3::ZERO, 0.03), 24),
        // From 8 sides, both ways round, closing around it (not along: 20 cm long), and from 8
        // turns at each end.
        (Obstacle::cylinder(Vec3::ZERO, 0.03, 0.1), 32),
        // The same, and closing along it too (7 cm long).
        (Obstacle::capsule(Vec3::ZERO, 0.025, 0.01), 48),
    ];
    for (shape, count) in shapes {
        let grasps = batchplan::grasps(&shape, &o).unwrap();
        assert_eq!(grasps.len(), count, "{shape:?}");
        for g in grasps {
            let (y, z) = (g.rotation * Vec3::Y, g.rotation * Vec3::Z);
            // The IK frame is the grasp depth inside the surface, which lies just behind it.
            let inside = shape.distance(g.position).value;
            assert!(
                (inside + o.depth).abs() < 1e-4,
                "{shape:?}: {inside} at {g:?}"
            );
            assert!(shape.distance(g.position - z * (o.depth + 1e-3)).value > 0.0);
            // The fingers close on something no wider than they open.
            let width = |direction: Vec3| {
                let mut far = 0.0;
                while shape.distance(g.position + direction * far).value < 0.0 {
                    far += 1e-4;
                }
                far
            };
            let across = width(y) + width(-y);
            assert!(
                across <= o.opening + 1e-3,
                "{shape:?}: {across} across at {g:?}"
            );
        }
    }
    let grid = SdfGrid::try_new([2, 2, 2], 0.1, Vec3::ZERO, &[1.0; 8]).unwrap();
    let grid = Obstacle::from_grid(grid);
    assert!(matches!(batchplan::grasps(&grid, &o), Err(Error::Input(_))));
}

#[test]
fn place_poses_turn_an_object_onto_itself() {
    let at = Pose::new(Vec3::new(0.4, 0.1, 0.03), Quat::from_rotation_x(0.2));
    let cases = [
        (Obstacle::cuboid(Vec3::ZERO, HALF), 2),
        (Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.05)), 4),
        (Obstacle::cylinder(Vec3::ZERO, 0.03, 0.05), 6),
        (
            Obstacle::from_grid(SdfGrid::try_new([2, 2, 2], 0.1, Vec3::ZERO, &[1.0; 8]).unwrap()),
            1,
        ),
    ];
    for (object, count) in cases {
        let poses = batchplan::place_poses(&object, at, 6);
        assert_eq!(poses.len(), count, "{object:?}");
        for p in &poses {
            assert_eq!(p.position, at.position);
            // Turned about the object's own z axis, which stays where it was.
            let axis = p.rotation * Vec3::Z;
            assert!(axis.distance(at.rotation * Vec3::Z) < 1e-5, "{object:?}");
        }
        // The shape lands on itself: a corner of its box goes to a corner.
        if let Obstacle::Cuboid { half_extents, .. } = object {
            for p in &poses {
                let corner = p.rotation * half_extents;
                let local = at.rotation.inverse() * corner;
                assert!((local.abs() - half_extents).length() < 1e-5, "{local}");
            }
        }
    }
}

#[test]
fn malformed_tasks_are_input_errors() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[scene(vec![])]).unwrap();
    let o = TaskOptions::default();
    let good = pick_from(&robot, vec![top()]);
    let unnormalized = Pose::new(top().position, top().rotation * 1.5);
    let bad = [
        Pick {
            world: 1,
            ..good.clone()
        },
        Pick {
            object: 2,
            ..good.clone()
        },
        Pick {
            grasps: vec![],
            ..good.clone()
        },
        Pick {
            grasps: vec![unnormalized],
            ..good.clone()
        },
        Pick {
            approach: 0.0,
            ..good.clone()
        },
        Pick {
            lift: Vec3::ZERO,
            ..good.clone()
        },
        Pick {
            support: vec![BOX],
            ..good.clone()
        },
        Pick {
            start: vec![0.0; 3],
            ..good.clone()
        },
        Pick {
            start: robot.upper().iter().map(|v| v + 0.1).collect(),
            ..good.clone()
        },
        Pick {
            hand: 1,
            ..good.clone()
        },
    ];
    for task in bad {
        let result = batchplan::pick(&device, &worlds, std::slice::from_ref(&task), &o);
        assert!(matches!(result, Err(Error::Input(_))), "{task:?}");
    }
    for options in [
        TaskOptions {
            tool_speed: 0.0,
            ..TaskOptions::default()
        },
        TaskOptions {
            min_fraction: 0.0,
            ..TaskOptions::default()
        },
        TaskOptions {
            min_fraction: 1.5,
            ..TaskOptions::default()
        },
    ] {
        let result = batchplan::pick(&device, &worlds, std::slice::from_ref(&good), &options);
        assert!(matches!(result, Err(Error::Input(_))), "{options:?}");
    }
    // A place on a device that already holds the object.
    let held = Held {
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, HALF),
        grasp: top(),
    };
    let holding = device
        .with_robot(&robot.attach(&held.attachment(&robot).unwrap()).unwrap())
        .unwrap();
    let place = Place::new(0, robot.default_q(), held, vec![Pose::IDENTITY]);
    let result = batchplan::place(&holding, &worlds, &[place], &o);
    assert!(matches!(result, Err(Error::Input(_))), "{result:?}");
}
