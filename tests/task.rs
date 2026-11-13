//! Whole tasks in one call (`pick`, `place`): every candidate that gets through comes back, quickest
//! first; a task none of whose candidates does says where and why each stopped; `grasps` and
//! `place_poses` propose candidates.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::PI;

use batchplan::rng::Rng;
use batchplan::{
    CandidateStop, Contact, Device, Error, GraspOptions, HandConstraint, Held, IkOptions,
    IkProblem, Obstacle, Pick, Place, Pose, Robot, RobotOptions, SdfGrid, TaskFailure, TaskOptions,
    TaskPlan, TaskStep, ToolPathStop, Touch, Trajectory, World,
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
    for t in plan.motions() {
        t.check(robot).unwrap();
    }
    assert_eq!(first(transit(plan)), start);
    assert_eq!(last(transit(plan)), plan.begin());
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

/// Where hand `hand` is along `t`, 256 times per knot interval, and how fast it moves there: from
/// the joint velocities, the hand's positions 5 ms of that motion either side.
fn hand_speeds(robot: &Robot, t: &Trajectory, hand: usize) -> Vec<(Vec3, f32)> {
    let n = robot.dof();
    let samples = t.sample(256.0 / t.knot_interval).unwrap();
    let states = samples
        .positions
        .chunks(n)
        .zip(samples.velocities.chunks(n));
    let h = 0.005;
    let each = states.map(|(q, dq)| {
        let moved = |by: f32| -> Vec3 {
            let q: Vec<f32> = q.iter().zip(dq).map(|(q, dq)| q + by * dq).collect();
            robot.ee_pose(&q, hand).position
        };
        (moved(0.0), moved(h).distance(moved(-h)) / (2.0 * h))
    });
    each.collect()
}

/// The transit of a task that plans it.
fn transit(plan: &TaskPlan) -> &Trajectory {
    plan.transit.as_ref().expect("the task plans its transit")
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
        // Each grasp alone moves alike.
        for plan in &all {
            let alone = pick_in(
                &device,
                scene(vec![]),
                pick_from(&robot, vec![grasps[plan.candidate]]),
                &o,
            )
            .unwrap()
            .remove(0);
            let motions = |p: &TaskPlan| p.motions().cloned().collect::<Vec<_>>();
            assert_eq!(
                motions(&alone),
                motions(plan),
                "{name}: grasp {}",
                plan.candidate
            );
        }
    }
}

/// A task's straight moves capped near where hand `hand` grasps or lets go (`near`) keep the
/// paths they take capped all along (`along`): the hand keeps under `cap` within `radius` of
/// there, on the way in and on the way out, moves faster farther off, and the moves take less
/// time than capped all along.
fn check_capped_near(robot: &Robot, hand: usize, near: &TaskPlan, along: &TaskPlan, what: &str) {
    let (cap, radius) = (NEAR_CAP, NEAR_RADIUS);
    let n = robot.dof();
    let cp = &near.approach.control_points;
    let contact = robot.ee_pose(&cp[cp.len() - n..], hand).position;
    for (kind, moved, capped) in [
        ("in", &near.approach, &along.approach),
        ("away", &near.depart, &along.depart),
    ] {
        assert_eq!(
            moved.control_points, capped.control_points,
            "{what}: {kind}"
        );
        let speeds = hand_speeds(robot, moved, hand);
        let fastest = |inside: bool| {
            let of = speeds
                .iter()
                .filter(|(p, _)| (p.distance(contact) <= radius) == inside);
            of.map(|(_, v)| *v).fold(0.0, f32::max)
        };
        let (within, beyond) = (fastest(true), fastest(false));
        assert!(
            within <= cap * (1.0 + 1e-3) && within > 0.9 * cap,
            "{what}: {within} m/s moving {kind} near the contact under a cap of {cap}"
        );
        assert!(
            beyond > 2.0 * cap,
            "{what}: {beyond} m/s moving {kind} farther off"
        );
        assert!(
            moved.duration() < 0.7 * capped.duration(),
            "{what}: {} s moving {kind}, {} s capped all along",
            moved.duration(),
            capped.duration()
        );
    }
}

/// The straight moves' cap and where it holds, in the tests of `tool_speed_near`.
const NEAR_CAP: f32 = 0.05;
const NEAR_RADIUS: f32 = 0.02;

/// `tool_speed` (`NEAR_CAP`) held near the contact alone, or all along.
fn capped(near: bool) -> TaskOptions {
    TaskOptions {
        tool_speed: NEAR_CAP,
        tool_speed_near: near.then_some(NEAR_RADIUS),
        ..TaskOptions::default()
    }
}

/// With `tool_speed_near`, the straight moves are capped near the grasp alone.
#[test]
fn the_straight_moves_can_be_capped_near_the_grasp_alone() {
    let robot = panda();
    let task = pick_from(&robot, vec![top_across()]);
    for device in devices(&robot) {
        let pick = |near| {
            let plans = pick_in(&device, scene(vec![]), task.clone(), &capped(near)).unwrap();
            plans.into_iter().next().expect("the grasp gets through")
        };
        check_capped_near(&robot, 0, &pick(true), &pick(false), &device.name());
    }
}

/// On two arms, the cap holds near where the hand that moves grasps or lets go: the second hand
/// picks a box up and sets it down capped near its own grasp and release.
#[test]
fn on_two_arms_the_cap_holds_near_the_hand_that_moves() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let at = Vec3::new(0.45, -0.35, HALF.z);
    let world = World::new(vec![
        Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.9, 0.02)),
        Obstacle::cuboid(at, HALF),
    ]);
    let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
    let task = Pick {
        hand: 1,
        support: vec![TABLE],
        ..Pick::new(0, robot.default_q(), 1, vec![top_across()])
    };
    let pick = |near: bool| {
        let picked = batchplan::pick(&device, &worlds, std::slice::from_ref(&task), &capped(near));
        let plans = picked
            .unwrap()
            .outcomes
            .remove(0)
            .expect("the grasp gets through");
        plans.into_iter().next().expect("a plan")
    };
    let (near, along) = (pick(true), pick(false));
    check_capped_near(&robot, 1, &near, &along, "pick");
    let mut after = world;
    after.remove(1).unwrap();
    let after = device.upload(&[after]).unwrap();
    let place = Place {
        support: vec![TABLE],
        ..Place::new(
            0,
            near.end(),
            near.held.clone().unwrap(),
            vec![Pose::new(at + Vec3::X * 0.15, Quat::IDENTITY)],
        )
    };
    let set_down = |near: bool| {
        let placed = batchplan::place(&device, &after, std::slice::from_ref(&place), &capped(near));
        let plans = placed
            .unwrap()
            .outcomes
            .remove(0)
            .expect("the release gets through");
        plans.into_iter().next().expect("a plan")
    };
    check_capped_near(&robot, 1, &set_down(true), &set_down(false), "place");
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
    // What each candidate collides with: obstacle `k`, among others.
    let names = |f: &TaskFailure, k: usize| {
        f.candidates.iter().all(|c| {
            c.contacts
                .iter()
                .any(|t| matches!(t, Contact::World { obstacle, .. } if *obstacle == k))
        })
    };
    // A slab where the palm would be at the grasp.
    let f = pick_in(&device, scene(vec![slab(0.09)]), tops(), &o).unwrap_err();
    assert!(stops(&f, TaskStep::Reach, collides), "{f}");
    assert!(names(&f, 2), "{:?}", f.candidates);
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
    assert!(f.candidates.iter().all(|c| c.contacts.is_empty()));
    // From 40 cm up, the hand passes a small block high over the box that the arm clears at the
    // grasp: the moves in stop at it, unless they may stop short.
    let block = Obstacle::cuboid(Vec3::new(0.45, 0.2, 0.38), Vec3::new(0.02, 0.02, 0.01));
    let from_high = || Pick {
        approach: 0.4,
        ..tops()
    };
    let f = pick_in(&device, scene(vec![block.clone()]), from_high(), &o).unwrap_err();
    assert!(stops(&f, TaskStep::Approach, collides), "{f}");
    assert!(names(&f, 2), "{:?}", f.candidates);
    let shorter = TaskOptions {
        min_fraction: 0.05,
        ..TaskOptions::default()
    };
    let plans = pick_in(&device, scene(vec![block]), from_high(), &shorter).unwrap();
    let n = robot.dof();
    let begins = robot.ee_pose(&plans[0].approach.control_points[..n], 0);
    let transit = &transit(&plans[0]).control_points;
    let arrives = robot.ee_pose(&transit[transit.len() - n..], 0);
    assert!(begins.position.distance(arrives.position) < 1e-5);
    let grasp_height = HALF.z * 2.0 - 0.02;
    let moved = begins.position.z - grasp_height;
    assert!(moved > 0.02 && moved < 0.38, "the move in is {moved} m");
    // Only an obstacle shortens a move: a lift out of reach stops the pick all the same.
    let too_high = Pick {
        lift: Vec3::Z * 1.5,
        ..tops()
    };
    let f = pick_in(&device, scene(vec![]), too_high, &shorter).unwrap_err();
    assert!(stops(&f, TaskStep::Depart, out), "{f}");
    // Lifting the box off a table it may not touch, and into a small lid just over the box,
    // between the fingers and below the palm.
    let unsupported = Pick {
        support: vec![],
        ..tops()
    };
    let f = pick_in(&device, scene(vec![]), unsupported, &o).unwrap_err();
    assert!(stops(&f, TaskStep::Depart, collides), "{f}");
    assert!(names(&f, TABLE), "{:?}", f.candidates);
    let lid = Obstacle::cuboid(Vec3::new(0.45, 0.2, 0.068), Vec3::new(0.01, 0.015, 0.004));
    let f = pick_in(&device, scene(vec![lid.clone()]), tops(), &o).unwrap_err();
    assert!(stops(&f, TaskStep::Depart, collides), "{f}");
    // Named as the task's world numbers it, where the lid comes after the box.
    assert!(names(&f, 2), "{:?}", f.candidates);
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
    assert!(names(&f, 2), "{:?}", f.candidates);
    assert_eq!(f.to_string(), "the start collides");
}

#[test]
fn a_held_object_sits_in_the_hand_as_its_grasp_holds_it() {
    let robot = panda();
    let object = Pose::new(Vec3::new(0.4, 0.1, 0.2), Quat::from_rotation_z(0.7));
    // From above, and from two sides: grasps that are not their own inverse.
    for grasp in [top(), grasp(Vec3::X, Vec3::Y), grasp(-Vec3::Y, Vec3::Z)] {
        let held = Held {
            locked: Default::default(),
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
        for t in plan.motions() {
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
        // At the free pose, the object it holds meets the table; the other collides as before.
        let on_table = |c: &Contact| matches!(c, Contact::World { link, obstacle: TABLE, .. } if link == "held by hand 0");
        let held_on_table = &failure.candidates[1].contacts;
        assert!(
            !held_on_table.is_empty() && held_on_table.iter().all(on_table),
            "{name}: {held_on_table:?}"
        );
        assert!(!failure.candidates[0].contacts.is_empty(), "{name}");
    }
}

/// The gripper's largest tilt from straight down along a trajectory, sampled at 1 kHz.
fn most_tilt(robot: &Robot, t: &Trajectory) -> f32 {
    let samples = t.sample(1000.0).unwrap();
    samples
        .positions
        .chunks(robot.dof())
        .map(|q| (robot.ee_pose(q, 0).rotation * Vec3::Z).angle_between(Vec3::NEG_Z))
        .fold(0.0, f32::max)
}

/// A place carries the box with the gripper within 5° of straight down along its transit, over a
/// wall it would otherwise tip the box past; a pose the hand reaches tilted stops at the transit,
/// and a start that breaks the constraint is an input error. A pick whose hand must stay near its
/// start stops at the transit too.
#[test]
fn a_place_carries_the_box_upright() {
    let robot = panda();
    let bound = 5f32.to_radians();
    let upright = HandConstraint {
        tilt: bound,
        ..HandConstraint::new(0, Pose::new(Vec3::ZERO, Quat::from_rotation_x(PI)))
    };
    for device in devices(&robot) {
        let name = device.name();
        let o = TaskOptions::default();
        let wall = Obstacle::cuboid(Vec3::new(0.45, 0.0, 0.15), Vec3::new(0.15, 0.01, 0.15));
        let world = scene(vec![wall]);
        let picked = pick_in(&device, world.clone(), pick_from(&robot, vec![top()]), &o).unwrap();
        // Kept within 20 cm of where it starts, the hand cannot get down to the box.
        let near = HandConstraint {
            position: Vec3::splat(0.2),
            ..HandConstraint::new(0, robot.ee_pose(robot.default_q(), 0))
        };
        let confined = Pick {
            constraints: vec![near],
            ..pick_from(&robot, vec![top()])
        };
        let failure = pick_in(&device, world.clone(), confined, &o).unwrap_err();
        assert_eq!(failure.step(), TaskStep::Transit, "{name}: {failure}");
        let held = picked[0].held.clone().unwrap();
        let mut after = world;
        after.remove(BOX).unwrap();
        let after = device.upload(&[after]).unwrap();
        let level = Pose::new(Vec3::new(0.45, -0.2, HALF.z), Quat::from_rotation_z(0.3));
        let tipped = Pose::new(Vec3::new(0.5, -0.15, 0.15), Quat::from_rotation_x(0.35));
        let place = |poses: Vec<Pose>, constraints: Vec<HandConstraint>| Place {
            support: vec![TABLE],
            constraints,
            ..Place::new(0, picked[0].end(), held.clone(), poses)
        };
        let carried = batchplan::place(&device, &after, &[place(vec![level], vec![upright])], &o);
        let plans = carried.unwrap().outcomes.remove(0).unwrap();
        let free = batchplan::place(&device, &after, &[place(vec![level], vec![])], &o);
        let loose = free.unwrap().outcomes.remove(0).unwrap();
        transit(&plans[0]).check(&robot).unwrap();
        let (kept, tipping) = (
            most_tilt(&robot, transit(&plans[0])),
            most_tilt(&robot, transit(&loose[0])),
        );
        assert!(
            kept <= bound + 1e-4 && tipping > 2.0 * bound,
            "{name}: the transit tilts {:.2}° with the constraint and {:.2}° without",
            kept.to_degrees(),
            tipping.to_degrees()
        );
        let tipping = batchplan::place(&device, &after, &[place(vec![tipped], vec![upright])], &o);
        let failure = tipping.unwrap().outcomes.remove(0).unwrap_err();
        assert_eq!(failure.step(), TaskStep::Transit, "{name}: {failure}");
        // Behind a candidate that stops at the transit, the next one's transit is its own.
        let both = place(vec![tipped, level], vec![upright]);
        let plans = batchplan::place(&device, &after, &[both], &o).unwrap();
        let plans = plans.outcomes[0].as_ref().unwrap();
        let n = robot.dof();
        assert!(
            plans.iter().all(|p| p.candidate == 1),
            "{name}: only the level pose"
        );
        for p in plans {
            let points = &transit(p).control_points;
            assert_eq!(
                &points[points.len() - n..],
                p.begin(),
                "{name}: a transit ends elsewhere"
            );
        }
        let anyhow = batchplan::place(&device, &after, &[place(vec![tipped], vec![])], &o);
        assert!(
            anyhow.unwrap().outcomes[0].is_ok(),
            "{name}: the tipped pose plans without the constraint"
        );
        let tilted_start = Place {
            start: robot.default_q().to_vec(),
            ..place(vec![level], vec![upright])
        };
        match batchplan::place(&device, &after, std::slice::from_ref(&tilted_start), &o) {
            Err(Error::Input(m)) => {
                assert!(
                    m.contains("task 0: the start breaks its constraints"),
                    "{m}"
                );
            }
            other => panic!("{name}: a tilted start should be an input error, got {other:?}"),
        }
        // Without its transit a task has no path to keep them along, and its start is a seed.
        let alone = Place {
            transit: false,
            ..tilted_start
        };
        let moves = batchplan::place(&device, &after, &[alone], &o);
        assert!(moves.is_ok(), "{name}: {moves:?}");
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
fn a_place_takes_the_other_arms_where_they_wait() {
    // The left Panda picks its box up and sets it down 15 cm further out, the right one waiting
    // where the start has it, or turning away on the transit and waiting there.
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    let at = Vec3::new(0.45, 0.35, HALF.z);
    let mut world = World::new(vec![
        Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.9, 0.02)),
        Obstacle::cuboid(at, HALF),
    ]);
    let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
    let pick = Pick {
        support: vec![TABLE],
        ..Pick::new(0, robot.default_q(), BOX, vec![top(), top_across()])
    };
    let picked = batchplan::pick(&device, &worlds, &[pick], &o)
        .unwrap()
        .outcomes[0]
        .as_ref()
        .unwrap()[0]
        .clone();
    world.remove(BOX).unwrap();
    let worlds = device.upload(&[world]).unwrap();
    let start = picked.end().to_vec();
    let (mine, others) = (robot.hand_joints(0), robot.hand_joints(1));
    let mut idle = start.clone();
    idle[others[0]] -= 0.5;
    // The left arm's joints in it do not count.
    let mut ignored = idle.clone();
    for &j in &mine {
        ignored[j] = 0.5 * (robot.lower()[j] + robot.upper()[j]);
    }
    let place = |idle: Option<Vec<f32>>| Place {
        support: vec![TABLE],
        idle,
        ..Place::new(
            0,
            start.clone(),
            picked.held.clone().unwrap(),
            vec![Pose::new(at + Vec3::X * 0.15, Quat::IDENTITY)],
        )
    };
    let tasks = [place(None), place(Some(idle.clone())), place(Some(ignored))];
    let placed = batchplan::place(&device, &worlds, &tasks, &o).unwrap();
    let plans: Vec<&TaskPlan> = placed
        .outcomes
        .iter()
        .map(|outcome| &outcome.as_ref().unwrap()[0])
        .collect();
    let n = robot.dof();
    let other_arm = |q: &[f32]| -> Vec<f32> { others.iter().map(|&j| q[j]).collect() };
    let rows =
        |t: &Trajectory| -> Vec<Vec<f32>> { t.control_points.chunks(n).map(other_arm).collect() };
    // The straight moves' splines interpolate a joint that waits to within rounding.
    let at = |r: &[f32], q: &[f32]| {
        r.iter()
            .zip(other_arm(q))
            .all(|(a, b)| (a - b).abs() < 1e-5)
    };
    let (waiting, turned) = (plans[0], plans[1]);
    for t in waiting.motions() {
        assert!(rows(t).iter().all(|r| at(r, &start)));
    }
    let transit = rows(transit(turned));
    assert!(at(&transit[0], &start));
    assert!(at(transit.last().unwrap(), &idle));
    for t in [&turned.approach, &turned.depart] {
        assert!(rows(t).iter().all(|r| at(r, &idle)));
    }
    let motions = |p: &TaskPlan| p.motions().cloned().collect::<Vec<_>>();
    assert_eq!(motions(plans[2]), motions(turned));
    // A configuration of the wrong length, or with an idle joint past its limits or not a number,
    // is refused; the left arm's joints may be anything.
    let mut past = idle.clone();
    past[others[1]] = robot.upper()[others[1]] + 0.1;
    let mut nan = idle.clone();
    nan[others[2]] = f32::NAN;
    for bad in [vec![0.0; 3], past, nan] {
        let result = batchplan::place(&device, &worlds, &[place(Some(bad))], &o);
        assert!(matches!(result, Err(Error::Input(_))), "{result:?}");
    }
    let mut anything = idle.clone();
    for &j in &mine {
        anything[j] = f32::NAN;
    }
    let placed = batchplan::place(&device, &worlds, &[place(Some(anything))], &o).unwrap();
    assert_eq!(
        motions(&placed.outcomes[0].as_ref().unwrap()[0]),
        motions(turned)
    );
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

/// A place's contacts number the obstacles as its world does, and the object it sets down after
/// them: backing off from a box held 5 cm deep, the palm still touches it.
#[test]
fn a_place_names_the_object_it_sets_down_after_its_worlds_obstacles() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let world = World::new(vec![scene(vec![]).obstacles[TABLE].clone()]);
    let tall = Vec3::new(HALF.x, HALF.y, 0.06);
    let held = Held {
        locked: Default::default(),
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, tall),
        grasp: Pose::new(Vec3::Z * 0.01, top_across().rotation),
    };
    let to = Pose::new(Vec3::new(0.45, 0.0, tall.z), Quat::IDENTITY);
    let place = Place {
        support: vec![TABLE],
        ..Place::new(0, robot.default_q(), held, vec![to])
    };
    let set_down = world.obstacles.len();
    let worlds = device.upload(&[world]).unwrap();
    let f = batchplan::place(&device, &worlds, &[place], &TaskOptions::default())
        .unwrap()
        .outcomes
        .remove(0)
        .unwrap_err();
    let palm = |t: &Contact| matches!(t, Contact::World { link, obstacle, .. } if link == "panda_hand" && *obstacle == set_down);
    assert!(
        f.candidates
            .iter()
            .all(|c| c.step == TaskStep::Depart && c.contacts.iter().any(palm)),
        "{:?}",
        f.candidates
    );
}

/// A held object collides where its grasp holds it, whatever else the call holds: a bar taken from
/// the side near its foot meets a slab over its top as it lifts, in the same call as a cube taken
/// from above, and taken near its top too, it clears a higher slab the foot's grasp would lift it
/// into.
#[test]
fn a_held_object_collides_where_its_grasp_holds_it() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    let half = Vec3::new(0.02, 0.02, 0.15);
    let bar = Obstacle::cuboid(Vec3::new(0.45, 0.2, half.z), half);
    // From -x, closing along y, `height` above the bar's foot.
    let (toward, closing) = (Vec3::X, Vec3::Y);
    let side = |height: f32| {
        Pose::new(
            Vec3::new(0.0, 0.0, height - half.z),
            Quat::from_mat3(&Mat3::from_cols(closing.cross(toward), closing, toward)),
        )
    };
    let slab = Obstacle::cuboid(Vec3::new(0.45, 0.2, 0.35), Vec3::new(0.05, 0.05, 0.01));
    let cube = scene(vec![]);
    let table = cube.obstacles[TABLE].clone();
    let lift = |more: Vec<Obstacle>| {
        let mut obstacles = vec![table.clone(), bar.clone()];
        obstacles.extend(more);
        World::new(obstacles)
    };
    let tasks = [
        pick_from(&robot, vec![top()]),
        Pick {
            world: 1,
            ..pick_from(&robot, vec![side(0.08)])
        },
    ];
    let outcomes = |bar_world: World| {
        let worlds = device.upload(&[cube.clone(), bar_world]).unwrap();
        batchplan::pick(&device, &worlds, &tasks, &o)
            .unwrap()
            .outcomes
    };
    let free = outcomes(lift(vec![]));
    assert!(free.iter().all(Result::is_ok), "{free:?}");
    let blocked = outcomes(lift(vec![slab]));
    assert!(blocked[0].is_ok(), "{:?}", blocked[0]);
    let f = blocked[1].as_ref().unwrap_err();
    let held_at_slab = |t: &Contact| matches!(t, Contact::World { link, obstacle, .. } if link == "held by hand 0" && *obstacle == 2);
    assert!(
        f.candidates
            .iter()
            .all(|c| c.step == TaskStep::Depart && c.contacts.iter().any(held_at_slab)),
        "{:?}",
        f.candidates
    );
    // Both grasps of the bar under a slab from 0.46 m: lifted 10 cm, its top reaches 0.40 m from
    // either, and 0.50 m where the foot's grasp held it from 18 cm up.
    let high = Obstacle::cuboid(Vec3::new(0.45, 0.2, 0.47), Vec3::new(0.05, 0.05, 0.01));
    let worlds = device.upload(&[lift(vec![high])]).unwrap();
    let both = pick_from(&robot, vec![side(0.08), side(0.18)]);
    let plans = batchplan::pick(&device, &worlds, &[both], &o)
        .unwrap()
        .outcomes
        .remove(0);
    assert_eq!(plans.map(|p| p.len()), Ok(2));
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
            hold: vec![],
            holding: None,
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
    // Refused as a grid however it is placed, without its values in the message.
    let skewed = grid.placed(Pose::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)));
    let err = batchplan::grasps(&skewed, &o).unwrap_err().to_string();
    assert_eq!(err, "a distance grid has no grasps");
}

/// The lowest of a grasp's frame and its open fingertips, `o.tip` past the frame and
/// `o.opening` apart, with `object` placed as it stands.
fn lowest(object: &Obstacle, grasp: &Pose, o: &GraspOptions) -> f32 {
    let at = object.pose().mul_pose(*grasp);
    let (out, closing) = (at.rotation * Vec3::Z, at.rotation * Vec3::Y);
    let fingers =
        [1.0, -1.0].map(|side| at.position + out * o.tip + closing * side * o.opening / 2.0);
    fingers.iter().map(|p| p.z).fold(at.position.z, f32::min)
}

#[test]
fn grasps_keep_the_hand_above_the_floor_or_say_why_there_are_none() {
    // A 2.5 cm cube on a table whose top is at z = 0, fingertips 12 mm past the frame.
    let cube = Obstacle::cuboid(Vec3::new(0.4, 0.0, 0.0125), Vec3::splat(0.0125));
    let o = GraspOptions {
        tip: 0.012,
        floor: Some(0.002),
        ..GraspOptions::default()
    };
    let grasps = batchplan::grasps(&cube, &o).unwrap();
    let at = cube.pose();
    let height = |g: &Pose, reach: f32| {
        at.mul_pose(*g).position.z + (at.rotation * g.rotation * Vec3::Z).z * reach
    };
    let down = |g: &Pose| (at.rotation * g.rotation * Vec3::Z).z;
    assert!(grasps.iter().all(|g| lowest(&cube, g, &o) >= 0.002 - 1e-5));
    // From above, as deep as keeps the fingertips at the floor; from the sides, the full depth,
    // closing level: closing upright, the lower finger would be in the table. From below, none.
    let from_above: Vec<&Pose> = grasps.iter().filter(|g| down(g) < -0.5).collect();
    assert_eq!(from_above.len(), 4);
    assert!(
        from_above
            .iter()
            .all(|g| (height(g, o.tip) - 0.002).abs() < 1e-5)
    );
    let from_sides: Vec<&Pose> = grasps.iter().filter(|g| down(g).abs() < 0.5).collect();
    assert_eq!(from_sides.len(), 8);
    let held = o.depth.min(0.0125);
    let level = |g: &&Pose| (at.rotation * g.rotation * Vec3::Y).z.abs() < 1e-6;
    assert!(
        from_sides
            .iter()
            .all(|g| level(g)
                && (cube.distance(at.mul_pose(**g).position).value + held).abs() < 1e-4)
    );
    assert_eq!(grasps.len(), 12, "none from below");
    // With no floor, every face's grasps at the full depth, the fingertips in the table.
    let anywhere = batchplan::grasps(&cube, &GraspOptions { floor: None, ..o }).unwrap();
    assert_eq!(anywhere.len(), 24);
    // A floor over the cube leaves none, and says why; so does a box too wide every way.
    let err = batchplan::grasps(
        &cube,
        &GraspOptions {
            floor: Some(0.03),
            ..o
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::Input(_)) && err.to_string().contains("floor"),
        "{err}"
    );
    let wide = Obstacle::cuboid(Vec3::ZERO, Vec3::splat(0.05));
    let err = batchplan::grasps(&wide, &GraspOptions::default()).unwrap_err();
    assert!(
        err.to_string().contains("10.0 cm across") && err.to_string().contains("8.0 cm"),
        "{err}"
    );
}

/// An 8.8 x 4.8 cm package `height` tall, standing at `x, y` on a table whose top is at z = 0.
fn package(x: f32, y: f32, height: f32) -> Obstacle {
    let half = Vec3::new(0.044, 0.024, height / 2.0);
    Obstacle::cuboid(Vec3::new(x, y, half.z), half)
}

#[test]
fn a_flat_box_is_grasped_from_above_by_as_much_of_the_fingers_as_reaches_in() {
    // A package 1.85 cm tall, fingertips 2 cm past the frame and a floor 5 mm over the table: at
    // the package's top the frame would put them 6.5 mm below the floor, and above it they reach
    // 13.5 mm in.
    let object = package(0.4, 0.0, 0.0185);
    let o = GraspOptions {
        tip: 0.02,
        floor: Some(0.005),
        ..GraspOptions::default()
    };
    let top = 0.0185;
    assert!(top - o.tip < 0.005 && top - 0.005 > o.overlap);
    let grasps = batchplan::grasps(&object, &o).unwrap();
    let at = object.pose();
    let height = |g: &Pose, reach: f32| {
        at.mul_pose(*g).position.z + (at.rotation * g.rotation * Vec3::Z).z * reach
    };
    let down = |g: &Pose| (at.rotation * g.rotation * Vec3::Z).z;
    // From above, closing across its 4.8 cm both ways round: the fingertips at the floor, the
    // frame over the top.
    let from_above: Vec<&Pose> = grasps.iter().filter(|g| down(g) < -0.5).collect();
    assert_eq!(from_above.len(), 2, "{grasps:?}");
    assert!(
        from_above
            .iter()
            .all(|g| (height(g, o.tip) - 0.005).abs() < 1e-5 && height(g, 0.0) > top)
    );
    assert_eq!(grasps.len(), 6, "4 from the sides, closing level");
    // Asked to hold more of the fingers than reach in, none from above.
    let deeper = GraspOptions {
        overlap: top - 0.005 + 1e-3,
        ..o
    };
    let grasps = batchplan::grasps(&object, &deeper).unwrap();
    assert!(grasps.iter().all(|g| down(g) > -0.5), "{grasps:?}");
    assert_eq!(grasps.len(), 4);
    // An overlap and an opening must be positive and finite, and the object one that can be
    // checked.
    let wrong = [-0.01, 0.0, f32::NAN, f32::INFINITY].map(|bad| [(bad, 0.08), (0.01, bad)]);
    for (overlap, opening) in wrong.concat() {
        let bad = GraspOptions {
            overlap,
            opening,
            ..o
        };
        assert!(matches!(
            batchplan::grasps(&object, &bad),
            Err(Error::Input(_))
        ));
    }
    let inside_out = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, -0.02, 0.02));
    let err = batchplan::grasps(&inside_out, &o).unwrap_err();
    assert!(
        matches!(err, Error::Input(_)) && err.to_string().contains("object to grasp"),
        "{err}"
    );
}

#[test]
fn every_grasp_from_above_over_a_floor_keeps_the_overlap_in_the_object() {
    let from_above = |grasps: &[Pose]| -> Vec<Pose> {
        grasps
            .iter()
            .copied()
            .filter(|g| (g.rotation * Vec3::Z).z < -0.5)
            .collect()
    };
    // An 8 mm package, the fingertips at the frame: at most 8 mm of them can ever reach in, short
    // of the 1 cm overlap, whether the floor holds the grasp less deep (5 mm) or not (2 mm).
    let flat = package(0.4, 0.0, 0.008);
    for floor in [0.002, 0.005] {
        let o = GraspOptions {
            floor: Some(floor),
            ..GraspOptions::default()
        };
        // Over the high floor the side grasps' frames, at its middle, are below the floor too.
        let grasps = batchplan::grasps(&flat, &o).unwrap_or_default();
        assert!(from_above(&grasps).is_empty(), "floor {floor}: {grasps:?}");
        // With a 2.5 mm overlap: from above at the full depth over the low floor, the middle, and
        // over the high one 3 mm in, the fingertips at the floor.
        let shallow = GraspOptions {
            overlap: 0.0025,
            ..o
        };
        let grasps = from_above(&batchplan::grasps(&flat, &shallow).unwrap());
        let depth = if floor < 0.004 { 0.004 } else { 0.003 };
        assert_eq!(grasps.len(), 2, "floor {floor}");
        assert!(
            grasps
                .iter()
                .all(|g| (flat.pose().mul_pose(*g).position.z - (0.008 - depth)).abs() < 1e-5),
            "floor {floor}: {grasps:?}"
        );
    }
    // Fingertips 2 mm short of the frame, a 2.4 cm package and a floor at 13 mm: the frame stops
    // at the floor, 11 mm in, and the fingers reach only 9 mm in.
    let o = GraspOptions {
        tip: -0.002,
        floor: Some(0.013),
        ..GraspOptions::default()
    };
    let grasps = batchplan::grasps(&package(0.4, 0.0, 0.024), &o).unwrap_or_default();
    assert!(from_above(&grasps).is_empty(), "{grasps:?}");
    let grasps = batchplan::grasps(
        &package(0.4, 0.0, 0.024),
        &GraspOptions {
            overlap: 0.009,
            ..o
        },
    )
    .unwrap();
    assert_eq!(from_above(&grasps).len(), 2);
    // Tilted a hair, as a simulator or perception reports it, the package is grasped as level:
    // with the floor over its middle, nowhere.
    let tilted = package(0.4, 0.0, 0.0185).placed(Pose::new(
        Vec3::new(0.4, 0.0, 0.00925),
        Quat::from_rotation_y(-5e-7),
    ));
    let o = GraspOptions {
        tip: 0.0124,
        floor: Some(0.01),
        ..GraspOptions::default()
    };
    let grasps = batchplan::grasps(&tilted, &o).unwrap_or_default();
    let at = tilted.pose();
    let far = grasps
        .iter()
        .map(|g| at.mul_pose(*g).position.distance(at.position))
        .fold(0.0, f32::max);
    assert!(far < 0.05, "a grasp {far} m from the package");
}

#[test]
fn a_flat_package_on_a_table_is_picked_from_above_by_its_top() {
    let robot = panda();
    let (tip, floor, top) = (robot.tip(0), 0.005, 0.016);
    let object = package(0.45, 0.1, top);
    let o = GraspOptions {
        tip,
        floor: Some(floor),
        ..GraspOptions::default()
    };
    // Lower than the fingertips reach past the frame and the floor, but taller than the floor and
    // the overlap.
    assert!(
        top < tip + floor && top - floor > o.overlap,
        "the fingertips reach {tip} m past the frame"
    );
    let (from_above, from_sides): (Vec<Pose>, Vec<Pose>) = batchplan::grasps(&object, &o)
        .unwrap()
        .into_iter()
        .partition(|g| (g.rotation * Vec3::Z).z < -0.5);
    let on_table = |c: &CandidateStop| {
        c.contacts.iter().any(|t| {
            matches!(
                t,
                Contact::World {
                    obstacle: TABLE,
                    ..
                }
            )
        })
    };
    let world = World::new(vec![scene(vec![]).obstacles[TABLE].clone(), object]);
    for device in devices(&robot) {
        let name = device.name();
        let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
        let pick = |grasps: &[Pose]| {
            let task = Pick {
                support: vec![TABLE],
                ..Pick::new(0, robot.default_q(), 1, grasps.to_vec())
            };
            batchplan::pick(&device, &worlds, &[task], &TaskOptions::default())
                .unwrap()
                .outcomes
                .remove(0)
        };
        // From the sides, the hand meets the table where it reaches.
        let failure = pick(&from_sides).unwrap_err();
        let colliding: Vec<&CandidateStop> = failure
            .candidates
            .iter()
            .filter(|c| !c.contacts.is_empty())
            .collect();
        assert!(
            !colliding.is_empty() && colliding.iter().all(|c| on_table(c)),
            "{name}: {failure}"
        );
        assert!(pick(&from_above).is_ok(), "{name}");
    }
}

#[test]
fn a_bar_over_a_plate_is_gripped_tilted_with_the_open_fingers_kept_off_the_plate() {
    // A bar 1 cm in radius lying along y, its top 3.7 cm over the plate, as a drawer's handle out
    // of its front. Tilted about the bar, an open finger swings down toward the plate.
    let robot = panda();
    let (tip, floor) = (robot.tip(0), 0.005);
    let middle = Vec3::new(0.5, 0.1, 0.027);
    let bar = Obstacle::cylinder(middle, 0.01, 0.06)
        .placed(Pose::new(middle, Quat::from_rotation_x(PI / 2.0)));
    let o = GraspOptions {
        tip,
        floor: Some(floor),
        turns: 16,
        ..GraspOptions::default()
    };
    let grasps = batchplan::grasps(&bar, &o).unwrap();
    assert!(grasps.iter().all(|g| lowest(&bar, g, &o) >= floor - 1e-5));
    let down = |g: &Pose| (bar.pose().rotation * g.rotation * Vec3::Z).z;
    let tilted: Vec<Pose> = grasps
        .iter()
        .copied()
        .filter(|g| (-0.99..-0.01).contains(&down(g)))
        .collect();
    assert!(!tilted.is_empty(), "{grasps:?}");
    // Held as far out as keeps the lower finger at the floor, and no farther.
    assert!(
        tilted
            .iter()
            .all(|g| (lowest(&bar, g, &o) - floor).abs() < 1e-5)
    );
    let world = World::new(vec![scene(vec![]).obstacles[TABLE].clone(), bar]);
    for device in devices(&robot) {
        let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
        let task = Pick {
            support: vec![TABLE],
            ..Pick::new(0, robot.default_q(), 1, tilted.clone())
        };
        let outcome = batchplan::pick(&device, &worlds, &[task], &TaskOptions::default())
            .unwrap()
            .outcomes
            .remove(0);
        let through = outcome.unwrap_or_else(|f| panic!("{}: {f}", device.name()));
        assert_eq!(through.len(), tilted.len(), "{}", device.name());
    }
}

#[test]
fn a_small_cube_on_a_table_is_picked_with_the_fingertips_kept_off_the_table() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let (tip, floor) = (robot.tip(0), 0.002);
    // Narrower than the fingers reach past the frame, so a grasp at its middle puts them in the
    // table, but tall enough to hold above them.
    let half = 0.8 * tip;
    assert!(
        2.0 * half > tip + floor,
        "the fingertips reach {tip} m past the frame"
    );
    let cube = Obstacle::cuboid(Vec3::new(0.45, 0.1, half), Vec3::splat(half));
    let world = World::new(vec![scene(vec![]).obstacles[TABLE].clone(), cube.clone()]);
    let worlds = device.upload(&[world]).unwrap();
    let pick = |grasps: Vec<Pose>| {
        let task = Pick {
            support: vec![TABLE],
            ..Pick::new(0, robot.default_q(), 1, grasps)
        };
        batchplan::pick(&device, &worlds, &[task], &TaskOptions::default())
            .unwrap()
            .outcomes
            .remove(0)
    };
    let from_above = |g: &Pose| (g.rotation * Vec3::Z).z < -0.5;
    let deep = batchplan::grasps(&cube, &GraspOptions::default()).unwrap();
    let deep: Vec<Pose> = deep.into_iter().filter(from_above).collect();
    let failure = pick(deep).unwrap_err();
    let on_table = |c: &CandidateStop| {
        c.contacts.iter().any(|t| {
            matches!(
                t,
                Contact::World {
                    obstacle: TABLE,
                    ..
                }
            )
        })
    };
    assert!(failure.candidates.iter().all(on_table), "{failure}");
    let o = GraspOptions {
        tip,
        floor: Some(floor),
        ..GraspOptions::default()
    };
    let kept = batchplan::grasps(&cube, &o).unwrap();
    let kept: Vec<Pose> = kept.into_iter().filter(from_above).collect();
    assert!(pick(kept).is_ok());
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

/// Whether every candidate of `f` stopped at `step`, colliding with obstacle `with` among others,
/// or for no stated reason where `with` is none.
fn all_stop(f: &TaskFailure, step: TaskStep, with: Option<usize>) -> bool {
    f.candidates.iter().all(|c| {
        let named = match with {
            Some(k) => {
                let hits =
                    |t: &Contact| matches!(t, Contact::World { obstacle, .. } if *obstacle == k);
                matches!(c.why, Some(ToolPathStop::Collision { .. })) && c.contacts.iter().any(hits)
            }
            None => c.why.is_none(),
        };
        c.step == step && named
    })
}

/// `base` with a ball at a finger of the hand at `start`, clear of what the fingers hold, three
/// times: no link may touch anything, the hand's links may touch obstacle `post` and the ball, and
/// they may touch the ball alone.
fn stepped(robot: &Robot, mut base: World, start: &[f32], post: usize) -> [World; 3] {
    let hand = robot.ee_pose(start, 0);
    base.obstacles.push(Obstacle::sphere(
        hand.position + hand.rotation * Vec3::Y * 0.05,
        0.015,
    ));
    let ball = base.obstacles.len() - 1;
    let links = robot.hand_links(0);
    let mut loose = base.clone();
    loose.touch = vec![
        Touch::new(post, links.clone()),
        Touch::new(ball, links.clone()),
    ];
    let mut past_the_ball = base.clone();
    past_the_ball.touch = vec![Touch::new(ball, links)];
    [base, loose, past_the_ball]
}

const STRICT: u32 = 0;
const LOOSE: u32 = 1;
const PAST_THE_BALL: u32 = 2;

#[test]
fn each_step_runs_in_the_world_its_task_names() {
    let robot = panda();
    let o = TaskOptions::default();
    let home = Pose::new(Vec3::new(0.45, 0.2, HALF.z), Quat::IDENTITY);
    for device in devices(&robot) {
        let name = device.name();
        // A finger of the grasp comes down on the post, which only the loose world lets it touch,
        // and a finger of the start is in the ball, which only the strict world keeps it from.
        let (the_post, ball) = (2, 3);
        let worlds = stepped(&robot, scene(vec![post()]), robot.default_q(), the_post);
        let worlds = device.upload(&worlds).unwrap();
        let loose = Pick {
            world: LOOSE,
            ..pick_from(&robot, vec![top()])
        };
        // Begun a centimeter back, the move in begins with the finger on the post already.
        let near = Pick {
            approach: 0.01,
            ..loose.clone()
        };
        let tasks = [
            loose.clone(),
            Pick {
                depart_world: Some(STRICT),
                ..loose.clone()
            },
            Pick {
                world: STRICT,
                transit_world: Some(LOOSE),
                depart_world: Some(LOOSE),
                ..loose.clone()
            },
            Pick {
                transit_world: Some(STRICT),
                ..loose.clone()
            },
            near.clone(),
            Pick {
                transit_world: Some(PAST_THE_BALL),
                ..near
            },
        ];
        let picked = batchplan::pick(&device, &worlds, &tasks, &o)
            .unwrap()
            .outcomes;
        let stopped = |k: usize| {
            picked[k]
                .as_ref()
                .expect_err("a step in a stricter world stops")
        };
        assert!(picked[0].is_ok(), "{name}: {:?}", picked[0]);
        assert!(
            all_stop(stopped(1), TaskStep::Depart, Some(the_post)),
            "{name}: {:?}",
            picked[1]
        );
        assert!(
            all_stop(stopped(2), TaskStep::Reach, Some(the_post)),
            "{name}: {:?}",
            picked[2]
        );
        assert!(
            all_stop(stopped(3), TaskStep::Start, Some(ball)),
            "{name}: {:?}",
            picked[3]
        );
        assert!(picked[4].is_ok(), "{name}: {:?}", picked[4]);
        assert!(
            all_stop(stopped(5), TaskStep::Transit, None),
            "{name}: {:?}",
            picked[5]
        );

        // Set down again where it stood, from the same start: the same finger on the same post
        // and in the same ball, which the box in the hand keeps clear of.
        let lifted = &picked[0].as_ref().unwrap()[0];
        let (the_post, ball) = (1, 2);
        let mut after = scene(vec![post()]);
        after.remove(BOX).unwrap();
        let worlds = stepped(&robot, after, robot.default_q(), the_post);
        let worlds = device.upload(&worlds).unwrap();
        let loose = Place {
            support: vec![TABLE],
            ..Place::new(
                LOOSE,
                robot.default_q(),
                lifted.held.clone().unwrap(),
                vec![home],
            )
        };
        let near = Place {
            lower: Vec3::Z * -0.01,
            ..loose.clone()
        };
        let tasks = [
            loose.clone(),
            Place {
                depart_world: Some(STRICT),
                ..loose.clone()
            },
            Place {
                world: STRICT,
                transit_world: Some(LOOSE),
                depart_world: Some(LOOSE),
                ..loose.clone()
            },
            Place {
                transit_world: Some(STRICT),
                ..loose.clone()
            },
            near.clone(),
            Place {
                transit_world: Some(PAST_THE_BALL),
                ..near
            },
        ];
        let placed = batchplan::place(&device, &worlds, &tasks, &o)
            .unwrap()
            .outcomes;
        let stopped = |k: usize| {
            placed[k]
                .as_ref()
                .expect_err("a step in a stricter world stops")
        };
        assert!(placed[0].is_ok(), "{name}: {:?}", placed[0]);
        assert!(
            all_stop(stopped(1), TaskStep::Depart, Some(the_post)),
            "{name}: {:?}",
            placed[1]
        );
        assert!(
            all_stop(stopped(2), TaskStep::Reach, Some(the_post)),
            "{name}: {:?}",
            placed[2]
        );
        assert!(
            all_stop(stopped(3), TaskStep::Start, Some(ball)),
            "{name}: {:?}",
            placed[3]
        );
        assert!(placed[4].is_ok(), "{name}: {:?}", placed[4]);
        assert!(
            all_stop(stopped(5), TaskStep::Transit, None),
            "{name}: {:?}",
            placed[5]
        );
    }
}

#[test]
fn a_task_without_its_transit_gives_its_moves_alone() {
    let robot = panda();
    let o = TaskOptions::default();
    let n = robot.dof();
    // The moves of `alone`, a task without its transit, are those of `whole`, the same task
    // with it, candidate by candidate, and it keeps every candidate `whole` does.
    let same_moves = |whole: &[TaskPlan], alone: &[TaskPlan], name: &str| {
        assert!(!whole.is_empty(), "{name}");
        for plan in whole {
            let moves = alone.iter().find(|m| m.candidate == plan.candidate);
            let moves = moves.unwrap_or_else(|| panic!("{name}: candidate {}", plan.candidate));
            assert_eq!(
                moves.approach.control_points, plan.approach.control_points,
                "{name}"
            );
            assert_eq!(
                moves.depart.control_points, plan.depart.control_points,
                "{name}"
            );
            let points = &transit(plan).control_points;
            assert_eq!(moves.begin(), &points[points.len() - n..], "{name}");
        }
        for moves in alone {
            assert!(moves.transit.is_none(), "{name}");
            assert_eq!(moves.motions().count(), 2, "{name}");
            let both = moves.approach.duration() + moves.depart.duration();
            assert_eq!(moves.duration(), both, "{name}");
        }
        let quickest_first = alone.windows(2).all(|w| w[0].duration() <= w[1].duration());
        assert!(quickest_first, "{name}");
    };
    for device in devices(&robot) {
        let name = device.name();
        let whole = pick_from(&robot, vec![top(), top_reversed(), top_across()]);
        let alone = Pick {
            transit: false,
            ..whole.clone()
        };
        let picked = pick_in(&device, scene(vec![]), whole.clone(), &o).unwrap();
        let moves = pick_in(&device, scene(vec![]), alone.clone(), &o).unwrap();
        same_moves(&picked, &moves, &name);

        // A start in a ball stops the task that leaves it, and only seeds the other's grasps;
        // tasks of both kinds share a call.
        let [blocked, ..] = stepped(&robot, scene(vec![]), &whole.start, 0);
        let worlds = device.upload(&[blocked]).unwrap();
        let outcomes = batchplan::pick(&device, &worlds, &[whole, alone], &o)
            .unwrap()
            .outcomes;
        let f = outcomes[0].as_ref().expect_err("the start is in the ball");
        assert!(all_stop(f, TaskStep::Start, Some(2)), "{name}: {f:?}");
        let moves = outcomes[1].as_ref().unwrap();
        assert!(
            !moves.is_empty() && moves.iter().all(|m| m.transit.is_none()),
            "{name}"
        );

        // A place alike, from where the lift left the box.
        let lifted = &picked[0];
        let mut after = scene(vec![]);
        after.remove(BOX).unwrap();
        let after = device.upload(&[after]).unwrap();
        let there = Pose::new(Vec3::new(0.45, -0.2, HALF.z), Quat::IDENTITY);
        let whole = Place {
            support: vec![TABLE],
            ..Place::new(0, lifted.end(), lifted.held.clone().unwrap(), vec![there])
        };
        let alone = Place {
            transit: false,
            ..whole.clone()
        };
        let place = |task: Place| {
            let result = batchplan::place(&device, &after, &[task], &o).unwrap();
            result.outcomes.into_iter().next().unwrap().unwrap()
        };
        same_moves(&place(whole), &place(alone), &name);
    }
}

#[test]
fn malformed_tasks_are_input_errors() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device
        .upload(&[scene(vec![]), scene(vec![post()])])
        .unwrap();
    let o = TaskOptions::default();
    let good = pick_from(&robot, vec![top()]);
    let unnormalized = Pose::new(top().position, top().rotation * 1.5);
    let bad = [
        Pick {
            world: 2,
            ..good.clone()
        },
        Pick {
            transit_world: Some(2),
            ..good.clone()
        },
        // A step's world holds the task's obstacles: this one has one more.
        Pick {
            depart_world: Some(1),
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
            tool_speed: f32::NAN,
            ..TaskOptions::default()
        },
        TaskOptions {
            min_fraction: 0.0,
            ..TaskOptions::default()
        },
        TaskOptions {
            tool_speed_near: Some(0.0),
            ..TaskOptions::default()
        },
        TaskOptions {
            tool_speed_near: Some(-0.02),
            ..TaskOptions::default()
        },
        TaskOptions {
            tool_speed_near: Some(f32::NAN),
            ..TaskOptions::default()
        },
        TaskOptions {
            tool_speed_near: Some(f32::INFINITY),
            ..TaskOptions::default()
        },
        TaskOptions {
            min_fraction: 1.5,
            ..TaskOptions::default()
        },
    ] {
        // Refused as options, by name, not later as the motions are timed.
        let named = if options.tool_speed_near.is_some() {
            "tool_speed_near"
        } else if options.min_fraction == 1.0 {
            "tool speed"
        } else {
            "min_fraction"
        };
        let result = batchplan::pick(&device, &worlds, std::slice::from_ref(&good), &options);
        assert!(
            matches!(&result, Err(Error::Input(why)) if why.contains(named)),
            "{options:?}: {result:?}"
        );
    }
    // A place on a device that already holds the object.
    let held = Held {
        locked: Default::default(),
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, HALF),
        grasp: top(),
    };
    let holding = device
        .with_robot(&robot.attach(&held.attachment(&robot).unwrap()).unwrap())
        .unwrap();
    let place = Place::new(0, robot.default_q(), held.clone(), vec![Pose::IDENTITY]);
    let result = batchplan::place(&holding, &worlds, &[place], &o);
    assert!(matches!(result, Err(Error::Input(_))), "{result:?}");
    // Two places of an object whose size is not finite, so no two copies of it compare equal.
    let not_finite = Held {
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::new(f32::NAN, HALF.y, HALF.z)),
        ..held
    };
    let place = Place::new(0, robot.default_q(), not_finite, vec![Pose::IDENTITY]);
    let result = batchplan::place(&device, &worlds, &[place.clone(), place], &o);
    assert!(matches!(result, Err(Error::Input(_))), "{result:?}");
}

/// A pick may keep the fingers narrower than loaded (`Pick::locked`): beside a post 2 cm off the
/// box, the open fingers closing along x come down on it and the pick fails, while fingers opened
/// 1 cm each, where they close on the box, pass between them. In one call with an open pick between
/// two narrow ones, each narrow pick plans as it does alone on a device for the robot relocked so,
/// what it leaves held says where the fingers are, and a bad task is named by its place in the call.
#[test]
fn a_pick_keeps_the_fingers_as_narrow_as_it_says() {
    let robot = panda();
    let narrow: std::collections::HashMap<String, f32> = [
        ("panda_finger_joint1".to_string(), 0.01),
        ("panda_finger_joint2".to_string(), 0.01),
    ]
    .into();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[scene(vec![post()])]).unwrap();
    let o = TaskOptions::default();
    let open = pick_from(&robot, vec![top()]);
    let narrowed = Pick {
        locked: narrow.clone(),
        ..open.clone()
    };
    let calls = [narrowed.clone(), open.clone(), narrowed.clone()];
    let both = batchplan::pick(&device, &worlds, &calls, &o).unwrap();
    assert_eq!(
        format!("{:?}", both.outcomes[0]),
        format!("{:?}", both.outcomes[2]),
        "the narrow picks alike, around the open one"
    );
    let failure = both.outcomes[1]
        .as_ref()
        .expect_err("the open fingers come down on the post");
    assert!(
        failure
            .candidates
            .iter()
            .flat_map(|c| &c.contacts)
            .any(|c| matches!(c, Contact::World { obstacle: 2, .. })),
        "{failure}"
    );
    let plans = both.outcomes[0]
        .as_ref()
        .unwrap_or_else(|f| panic!("narrow: {f}: {:?}", f.candidates[0].contacts));
    let relocked = device.with_robot(&robot.relock(&narrow).unwrap()).unwrap();
    let alone = batchplan::pick(&relocked, &worlds, std::slice::from_ref(&open), &o).unwrap();
    let alone = alone.outcomes[0].as_ref().unwrap();
    assert_eq!(plans.len(), alone.len());
    for (a, b) in plans.iter().zip(alone) {
        assert_eq!(a.approach.control_points, b.approach.control_points);
        assert_eq!(a.depart.control_points, b.depart.control_points);
        assert_eq!(
            a.held.as_ref().unwrap().locked,
            narrow,
            "held says where the fingers are"
        );
    }
    let bad = Pick {
        locked: [("panda_joint1".to_string(), 0.0)].into(),
        ..narrowed.clone()
    };
    match batchplan::pick(&device, &worlds, &[narrowed.clone(), open, bad], &o) {
        Err(Error::Input(message)) => assert!(message.contains("task 2"), "{message}"),
        other => panic!("a joint not locked is refused by the task's place in the call: {other:?}"),
    }
    let far = Pick {
        object: 9,
        ..narrowed.clone()
    };
    match batchplan::pick(&device, &worlds, &[narrowed.clone(), narrowed, far], &o) {
        Err(Error::Input(message)) => assert!(message.contains("task 2"), "{message}"),
        other => panic!("{other:?}"),
    }
}
