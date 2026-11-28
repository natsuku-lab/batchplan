//! Whole tasks in one call (`pick`, `place`): every candidate that gets through comes back, quickest
//! first; a task none of whose candidates does says where and why each stopped; `grasps` and
//! `place_poses` propose candidates.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::LazyLock;

use batchplan::rng::Rng;
use batchplan::{
    CandidateStop, CollisionModel, Contact, Device, Error, Gripper, GripperCalibration,
    HandConstraint, Held, Hold, IkOptions, IkProblem, Obstacle, Pick, Place, Pose, Release, Robot,
    RobotOptions, SdfGrid, ShapeGraspOptions, Surface, SurfaceGraspOptions, TaskFailure,
    TaskOptions, TaskPlan, TaskStep, ToolPathStop, Touch, Trajectory, World, Worlds,
};
use glam::{Mat3, Quat, Vec3};

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

/// `holds`, each with the fingers where `locks` has them, one each.
fn locked_at(holds: &[Hold], locks: impl IntoIterator<Item = HashMap<String, f32>>) -> Vec<Hold> {
    let each = holds.iter().zip(locks);
    each.map(|(h, locked)| Hold {
        locked,
        ..h.clone()
    })
    .collect()
}

/// `releases`, each dropping as far as `drops` has it, one each.
fn dropping(releases: &[Release], drops: impl IntoIterator<Item = f32>) -> Vec<Release> {
    let each = releases.iter().zip(drops);
    each.map(|(r, drop)| Release { drop, ..*r }).collect()
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
    let robot = common::panda().unwrap();
    let start = robot.default_q().to_vec();
    let o = TaskOptions::default();
    for device in common::devices(&robot) {
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
    let robot = common::panda().unwrap();
    let task = pick_from(&robot, vec![top_across()]);
    for device in common::devices(&robot) {
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
    let robot = common::panda().unwrap();
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
        hand(),
        &ShapeGraspOptions::default(),
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
    let robot = common::panda().unwrap();
    let object = Pose::new(Vec3::new(0.4, 0.1, 0.2), Quat::from_rotation_z(0.7));
    // From above, and from two sides: grasps that are not their own inverse.
    for grasp in [top(), grasp(Vec3::X, Vec3::Y), grasp(-Vec3::Y, Vec3::Z)] {
        let held = Held {
            carries: vec![],
            touch_links: vec![],
            also: vec![],
            locked: Default::default(),
            spheres: Default::default(),
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
    let robot = common::panda().unwrap();
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
    let robot = common::panda().unwrap();
    for device in common::devices(&robot) {
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
    let robot = common::panda().unwrap();
    let bound = 5f32.to_radians();
    let upright = HandConstraint {
        tilt: bound,
        ..HandConstraint::new(0, Pose::new(Vec3::ZERO, Quat::from_rotation_x(PI)))
    };
    for device in common::devices(&robot) {
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
    let robot = common::panda().unwrap();
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
    let robot = common::panda().unwrap();
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
    let robot = common::panda().unwrap();
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
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let world = World::new(vec![scene(vec![]).obstacles[TABLE].clone()]);
    let tall = Vec3::new(HALF.x, HALF.y, 0.06);
    let held = Held {
        carries: vec![],
        touch_links: vec![],
        also: vec![],
        locked: Default::default(),
        spheres: Default::default(),
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

/// A release stays at the commanded height while the predicted scene and the departure
/// collision enclosure account for the selected drop, including candidates in later waves.
#[test]
fn placing_above_a_rest_returns_its_landing_and_the_checked_fall() {
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let rest = Pose::new(Vec3::new(0.45, -0.2, HALF.z), Quat::IDENTITY);
    let taken = Pose::new(Vec3::new(0.3, -0.25, HALF.z), Quat::IDENTITY);
    let before = scene(vec![Obstacle::cuboid(taken.position, Vec3::splat(0.03))]);
    let mut after = before.clone();
    after.remove(BOX).unwrap();
    for device in common::devices(&robot) {
        let worlds = device.upload(&[before.clone(), after.clone()]).unwrap();
        let picked =
            batchplan::pick(&device, &worlds, &[pick_from(&robot, vec![top()])], &o).unwrap();
        let picked = &picked.outcomes[0].as_ref().unwrap()[0];
        let drop = 0.008;
        let release = Pose::new(rest.position + Vec3::Z * drop, rest.rotation);
        let mut poses = vec![taken; 40];
        poses.push(release);
        let mut falls = vec![0.002; 40];
        falls.push(drop);
        let releases = dropping(
            &poses.into_iter().map(Release::from).collect::<Vec<_>>(),
            falls,
        );
        let task = Place {
            support: vec![TABLE],
            enough: Some(1),
            ..Place::new(1, picked.end(), picked.held.clone().unwrap(), releases)
        };
        let run = |t: &Place| batchplan::place(&device, &worlds, std::slice::from_ref(t), &o);
        let plan = run(&task).unwrap().outcomes.remove(0).unwrap().remove(0);
        assert_eq!(plan.candidate, 40);
        let landed = plan.placed.as_ref().unwrap();
        assert!(landed.pose().position.distance(rest.position) < 1e-6);
        let swept = plan.falling.unwrap();
        let top = release.position + Vec3::Z * HALF.z;
        let bottom = rest.position - Vec3::Z * HALF.z;
        assert!(swept.distance(top).value < 1e-6 && swept.distance(bottom).value < 1e-6);
        let on = Place {
            releases: dropping(&task.releases, [0.0; 41]),
            ..task.clone()
        };
        let exact = run(&on).unwrap().outcomes.remove(0).unwrap().remove(0);
        assert!(exact.falling.is_none());
        assert_eq!(exact.placed.unwrap().pose(), release);
        assert_eq!(exact.approach.control_points, plan.approach.control_points);
        for invalid in [f32::NAN, -drop] {
            assert!(matches!(
                run(&Place {
                    releases: dropping(&task.releases, [invalid; 41]),
                    ..task.clone()
                }),
                Err(Error::Input(_))
            ));
        }
    }
}

/// A held object collides where its grasp holds it, whatever else the call holds: a bar taken from
/// the side near its foot meets a slab over its top as it lifts, in the same call as a cube taken
/// from above, and taken near its top too, it clears a higher slab the foot's grasp would lift it
/// into.
#[test]
fn a_held_object_collides_where_its_grasp_holds_it() {
    let robot = common::panda().unwrap();
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
    let robot = common::panda().unwrap();
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
    let o = ShapeGraspOptions::default();
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
        let grasps = batchplan::grasps(&shape, hand(), &o).unwrap();
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
                across <= hand().capacity() + 1e-3,
                "{shape:?}: {across} across at {g:?}"
            );
        }
    }
    let grid = SdfGrid::try_new([2, 2, 2], 0.1, Vec3::ZERO, &[1.0; 8]).unwrap();
    let grid = Obstacle::from_grid(grid);
    assert!(matches!(
        batchplan::grasps(&grid, hand(), &o),
        Err(Error::Input(_))
    ));
    // Refused as a grid however it is placed, without its values in the message.
    let skewed = grid.placed(Pose::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)));
    let err = batchplan::grasps(&skewed, hand(), &o)
        .unwrap_err()
        .to_string();
    assert_eq!(err, "a distance grid has no grasps");
}

/// The Panda's hand as `common::panda` holds it, its pads 8 cm apart, for grasps of a shape, which
/// need no fingers that slide.
fn hand() -> &'static Gripper {
    static HAND: LazyLock<Gripper> = LazyLock::new(|| {
        let calibration = GripperCalibration::new(0.08, 0.0, HashMap::new());
        common::panda().unwrap().gripper(0, &calibration).unwrap()
    });
    &HAND
}

/// How far `o`'s fingertips reach past the frame: its `tip`, or the hand's.
fn reach(o: &ShapeGraspOptions) -> f32 {
    o.tip.unwrap_or_else(|| hand().tip())
}

/// The lowest of a grasp's frame and its open fingertips, [`reach`] past the frame and the hand's
/// loaded gap apart, with `object` placed as it stands.
fn lowest(object: &Obstacle, grasp: &Pose, o: &ShapeGraspOptions) -> f32 {
    let at = object.pose().mul_pose(*grasp);
    let (out, closing) = (at.rotation * Vec3::Z, at.rotation * Vec3::Y);
    let open = hand().opening();
    let fingers =
        [1.0, -1.0].map(|side| at.position + out * reach(o) + closing * side * open / 2.0);
    fingers.iter().map(|p| p.z).fold(at.position.z, f32::min)
}

#[test]
fn grasps_keep_the_hand_above_the_floor_or_say_why_there_are_none() {
    // A 2.5 cm cube on a table whose top is at z = 0, fingertips 12 mm past the frame.
    let cube = Obstacle::cuboid(Vec3::new(0.4, 0.0, 0.0125), Vec3::splat(0.0125));
    let o = ShapeGraspOptions {
        tip: Some(0.012),
        floor: Some(0.002),
        ..ShapeGraspOptions::default()
    };
    let grasps = batchplan::grasps(&cube, hand(), &o).unwrap();
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
            .all(|g| (height(g, reach(&o)) - 0.002).abs() < 1e-5)
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
    let anywhere =
        batchplan::grasps(&cube, hand(), &ShapeGraspOptions { floor: None, ..o }).unwrap();
    assert_eq!(anywhere.len(), 24);
    // A floor over the cube leaves none, and says why; so does a box too wide every way.
    let err = batchplan::grasps(
        &cube,
        hand(),
        &ShapeGraspOptions {
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
    let err = batchplan::grasps(&wide, hand(), &ShapeGraspOptions::default()).unwrap_err();
    assert!(
        err.to_string().contains("10.0 cm across") && err.to_string().contains("8.0 cm"),
        "{err}"
    );
}

/// A gripper whose pads stand 8 cm apart as loaded and that grasps 6 cm.
fn grasping_six_cm() -> Gripper {
    let calibration = GripperCalibration {
        capacity: Some(0.06),
        ..GripperCalibration::new(0.08, 0.0, HashMap::new())
    };
    common::panda().unwrap().gripper(0, &calibration).unwrap()
}

#[test]
fn a_shapes_grasps_close_across_no_more_than_the_gripper_grasps() {
    // A 7 x 4 x 4 cm box, one side between the 6 cm capacity and the 8 cm loaded gap: closed
    // across its 4 cm sides alone.
    let object = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.035, 0.02, 0.02));
    let o = ShapeGraspOptions::default();
    let across = |g: &Pose| (g.rotation * Vec3::Y).x.abs();
    let open = batchplan::grasps(&object, hand(), &o).unwrap();
    assert!(
        open.iter().any(|g| across(g) > 0.99),
        "8 cm closes across its 7 cm"
    );
    let within = batchplan::grasps(&object, &grasping_six_cm(), &o).unwrap();
    assert!(
        !within.is_empty() && within.iter().all(|g| across(g) < 0.01),
        "{within:?}"
    );
}

#[test]
fn what_the_gripper_grasps_and_how_far_it_opens_are_what_a_refusal_names() {
    // The gripper grasps 6 cm and opens 8: a ball 7 cm across fits no grasp, and the message
    // names what it grasps; the 2.5 cm cube under a floor over its top has no grasp the floor
    // allows, and the message names how far apart the open fingers stand.
    let gripper = grasping_six_cm();
    let ball = Obstacle::sphere(Vec3::ZERO, 0.035);
    let err = batchplan::grasps(&ball, &gripper, &ShapeGraspOptions::default()).unwrap_err();
    assert!(
        err.to_string().contains("7.0 cm across") && err.to_string().contains("grasps 6.0 cm"),
        "{err}"
    );
    let cube = Obstacle::cuboid(Vec3::new(0.4, 0.0, 0.0125), Vec3::splat(0.0125));
    let o = ShapeGraspOptions {
        tip: Some(0.012),
        floor: Some(0.03),
        ..ShapeGraspOptions::default()
    };
    let err = batchplan::grasps(&cube, &gripper, &o).unwrap_err();
    assert!(err.to_string().contains("8.0 cm apart"), "{err}");
}

#[test]
fn a_hand_without_spheres_has_no_reach_for_a_floor_to_keep() {
    // Loaded without collision spheres, the hand reaches nowhere past its frame: over a floor, a
    // shape's grasps refuse the gripper's tip, naming it and why, and take the options' own; with
    // no floor, no tip is read.
    let options = RobotOptions {
        collision_model: Some(CollisionModel::default()),
        ..common::panda_options()
    };
    let robot = Robot::load(common::asset("franka/franka_panda.urdf"), &options).unwrap();
    let calibration = GripperCalibration::new(0.08, 0.0, HashMap::new());
    let bare = robot.gripper(0, &calibration).unwrap();
    assert_eq!(bare.tip(), f32::NEG_INFINITY);
    let cube = Obstacle::cuboid(Vec3::new(0.4, 0.0, 0.0125), Vec3::splat(0.0125));
    let floored = ShapeGraspOptions {
        floor: Some(0.002),
        ..ShapeGraspOptions::default()
    };
    match batchplan::grasps(&cube, &bare, &floored) {
        Err(Error::Input(message)) => {
            assert!(
                message.contains("-inf m") && message.contains("no spheres"),
                "{message}"
            )
        }
        other => panic!("a hand without spheres keeps no floor: {other:?}"),
    }
    let given = ShapeGraspOptions {
        tip: Some(0.012),
        ..floored
    };
    assert!(!batchplan::grasps(&cube, &bare, &given).unwrap().is_empty());
    let anywhere = batchplan::grasps(&cube, &bare, &ShapeGraspOptions::default()).unwrap();
    assert_eq!(
        anywhere,
        batchplan::grasps(&cube, hand(), &ShapeGraspOptions::default()).unwrap()
    );
}

/// The open fingers stand the gripper's loaded gap apart, wider than what it grasps: over a floor,
/// a box tilted 20 degrees about its long side, grasped from above across its 4 cm, is held as
/// deep as keeps the lower open fingertip, half the 8 cm loaded gap out along the tilted closing
/// line, at the floor, where half its 6 cm capacity would leave it 3.4 mm over it.
#[test]
fn over_a_floor_the_open_fingers_stand_the_loaded_gap_apart() {
    let gripper = grasping_six_cm();
    let tilted = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.05, 0.02, 0.02)).placed(Pose::new(
        Vec3::new(0.4, 0.0, 0.05),
        Quat::from_rotation_x(20f32.to_radians()),
    ));
    let (tip, floor) = (0.01, 0.03);
    let o = ShapeGraspOptions {
        tip: Some(tip),
        floor: Some(floor),
        ..ShapeGraspOptions::default()
    };
    let found = batchplan::grasps(&tilted, &gripper, &o).unwrap();
    // The lowest of a grasp's frame and its open fingertips, the fingers `apart`.
    let lowest = |g: &Pose, apart: f32| {
        let at = tilted.pose().mul_pose(*g);
        let (out, closing) = (at.rotation * Vec3::Z, at.rotation * Vec3::Y);
        let tips = [1.0, -1.0].map(|side| at.position + out * tip + closing * side * apart / 2.0);
        tips.iter().map(|p| p.z).fold(at.position.z, f32::min)
    };
    let open = gripper.opening();
    assert!(
        found.iter().all(|g| lowest(g, open) >= floor - 1e-5),
        "{found:?}"
    );
    let at_the_floor = found.iter().filter(|g| {
        let down = (tilted.pose().rotation * g.rotation * Vec3::Z).z;
        down < -0.5
            && (lowest(g, open) - floor).abs() < 1e-5
            && lowest(g, gripper.capacity()) > floor + 1e-3
    });
    assert!(at_the_floor.count() > 0, "{found:?}");
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
    let o = ShapeGraspOptions {
        tip: Some(0.02),
        floor: Some(0.005),
        ..ShapeGraspOptions::default()
    };
    let top = 0.0185;
    assert!(top - reach(&o) < 0.005 && top - 0.005 > o.overlap);
    let grasps = batchplan::grasps(&object, hand(), &o).unwrap();
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
            .all(|g| (height(g, reach(&o)) - 0.005).abs() < 1e-5 && height(g, 0.0) > top)
    );
    assert_eq!(grasps.len(), 6, "4 from the sides, closing level");
    // Asked to hold more of the fingers than reach in, none from above.
    let deeper = ShapeGraspOptions {
        overlap: top - 0.005 + 1e-3,
        ..o
    };
    let grasps = batchplan::grasps(&object, hand(), &deeper).unwrap();
    assert!(grasps.iter().all(|g| down(g) > -0.5), "{grasps:?}");
    assert_eq!(grasps.len(), 4);
    // An overlap and an opening must be positive and finite, and the object one that can be
    // checked: the overlap the options', the opening the gripper's, which refuses it.
    for bad in [-0.01, 0.0, f32::NAN, f32::INFINITY] {
        let wrong = ShapeGraspOptions { overlap: bad, ..o };
        assert!(matches!(
            batchplan::grasps(&object, hand(), &wrong),
            Err(Error::Input(_))
        ));
        let calibration = GripperCalibration::new(bad, 0.0, HashMap::new());
        assert!(matches!(
            common::panda().unwrap().gripper(0, &calibration),
            Err(Error::Input(_))
        ));
    }
    let inside_out = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, -0.02, 0.02));
    let err = batchplan::grasps(&inside_out, hand(), &o).unwrap_err();
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
        let o = ShapeGraspOptions {
            tip: Some(0.0),
            floor: Some(floor),
            ..ShapeGraspOptions::default()
        };
        // Over the high floor the side grasps' frames, at its middle, are below the floor too.
        let grasps = batchplan::grasps(&flat, hand(), &o).unwrap_or_default();
        assert!(from_above(&grasps).is_empty(), "floor {floor}: {grasps:?}");
        // With a 2.5 mm overlap: from above at the full depth over the low floor, the middle, and
        // over the high one 3 mm in, the fingertips at the floor.
        let shallow = ShapeGraspOptions {
            overlap: 0.0025,
            ..o
        };
        let grasps = from_above(&batchplan::grasps(&flat, hand(), &shallow).unwrap());
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
    let o = ShapeGraspOptions {
        tip: Some(-0.002),
        floor: Some(0.013),
        ..ShapeGraspOptions::default()
    };
    let grasps = batchplan::grasps(&package(0.4, 0.0, 0.024), hand(), &o).unwrap_or_default();
    assert!(from_above(&grasps).is_empty(), "{grasps:?}");
    let grasps = batchplan::grasps(
        &package(0.4, 0.0, 0.024),
        hand(),
        &ShapeGraspOptions {
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
    let o = ShapeGraspOptions {
        tip: Some(0.0124),
        floor: Some(0.01),
        ..ShapeGraspOptions::default()
    };
    let grasps = batchplan::grasps(&tilted, hand(), &o).unwrap_or_default();
    let at = tilted.pose();
    let far = grasps
        .iter()
        .map(|g| at.mul_pose(*g).position.distance(at.position))
        .fold(0.0, f32::max);
    assert!(far < 0.05, "a grasp {far} m from the package");
}

#[test]
fn a_flat_package_on_a_table_is_picked_from_above_by_its_top() {
    let robot = common::panda().unwrap();
    let (tip, floor, top) = (robot.tip(0), 0.005, 0.016);
    let object = package(0.45, 0.1, top);
    let o = ShapeGraspOptions {
        tip: Some(tip),
        floor: Some(floor),
        ..ShapeGraspOptions::default()
    };
    // Lower than the fingertips reach past the frame and the floor, but taller than the floor and
    // the overlap.
    assert!(
        top < tip + floor && top - floor > o.overlap,
        "the fingertips reach {tip} m past the frame"
    );
    let (from_above, from_sides): (Vec<Pose>, Vec<Pose>) = batchplan::grasps(&object, hand(), &o)
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
    for device in common::devices(&robot) {
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
fn a_shapes_grasps_reach_the_grippers_tip_where_their_options_name_none() {
    // The flat package above, lower than the Panda's fingertips reach past the frame and the
    // floor: how far they reach moves its grasps from above.
    let robot = common::panda().unwrap();
    assert_eq!(hand().tip(), robot.tip(0));
    let object = package(0.45, 0.1, 0.016);
    let grasps = |tip: Option<f32>| {
        let o = ShapeGraspOptions {
            tip,
            floor: Some(0.005),
            ..ShapeGraspOptions::default()
        };
        batchplan::grasps(&object, hand(), &o).unwrap()
    };
    assert_eq!(grasps(None), grasps(Some(robot.tip(0))));
    assert_ne!(
        grasps(None),
        grasps(Some(0.0)),
        "the reach moves the grasps"
    );
}

#[test]
fn a_bar_over_a_plate_is_gripped_tilted_with_the_open_fingers_kept_off_the_plate() {
    // A bar 1 cm in radius lying along y, its top 3.7 cm over the plate, as a drawer's handle out
    // of its front. Tilted about the bar, an open finger swings down toward the plate.
    let robot = common::panda().unwrap();
    let (tip, floor) = (robot.tip(0), 0.005);
    let middle = Vec3::new(0.5, 0.1, 0.027);
    let bar = Obstacle::cylinder(middle, 0.01, 0.06)
        .placed(Pose::new(middle, Quat::from_rotation_x(PI / 2.0)));
    let o = ShapeGraspOptions {
        tip: Some(tip),
        floor: Some(floor),
        turns: 16,
        ..ShapeGraspOptions::default()
    };
    let grasps = batchplan::grasps(&bar, hand(), &o).unwrap();
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
    for device in common::devices(&robot) {
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
    let robot = common::panda().unwrap();
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
    let deep = batchplan::grasps(&cube, hand(), &ShapeGraspOptions::default()).unwrap();
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
    let o = ShapeGraspOptions {
        tip: Some(tip),
        floor: Some(floor),
        ..ShapeGraspOptions::default()
    };
    let kept = batchplan::grasps(&cube, hand(), &o).unwrap();
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
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let home = Pose::new(Vec3::new(0.45, 0.2, HALF.z), Quat::IDENTITY);
    for device in common::devices(&robot) {
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
    let robot = common::panda().unwrap();
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
    for device in common::devices(&robot) {
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
    let robot = common::panda().unwrap();
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
            grasps: vec![unnormalized.into()],
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
        carries: vec![],
        touch_links: vec![],
        also: vec![],
        locked: Default::default(),
        spheres: Default::default(),
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

/// A grasp may keep the fingers narrower than loaded (`Hold::locked`): beside a post 2 cm off the
/// box, the open fingers closing along x come down on it and the pick fails, while fingers opened
/// 1 cm each, where they close on the box, pass between them. In one call with an open pick between
/// two narrow ones, each narrow pick plans as it does alone on a device for the robot relocked so,
/// what it leaves held says where the fingers are, and a bad task is named by its place in the call.
/// One pick whose grasps keep the fingers each their own way plans by each as a pick of that
/// grasp alone does, its plans numbered as its grasps are.
#[test]
fn a_pick_keeps_the_fingers_as_narrow_as_it_says() {
    let robot = common::panda().unwrap();
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
        grasps: locked_at(&open.grasps, [narrow.clone()]),
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
    // The same grasp three times, the fingers narrow for the second alone: the one plan is its.
    let mixed = Pick {
        grasps: locked_at(
            &vec![Hold::from(top()); 3],
            [Default::default(), narrow.clone(), Default::default()],
        ),
        ..open.clone()
    };
    let each = batchplan::pick(&device, &worlds, &[open.clone(), mixed.clone()], &o).unwrap();
    let by_one = each.outcomes[1].as_ref().expect("the narrow grasp plans");
    assert_eq!(by_one.len(), plans.len());
    for (a, b) in by_one.iter().zip(plans) {
        assert_eq!(a.candidate, 1, "numbered as the pick's grasps are");
        assert_eq!(a.approach.control_points, b.approach.control_points);
        assert_eq!(a.depart.control_points, b.depart.control_points);
        assert_eq!(a.held.as_ref().unwrap().locked, narrow);
    }
    // With every grasp's fingers open, each stops where the open pick's one does, in order.
    let open_thrice = Pick {
        grasps: vec![Hold::from(top()); 3],
        ..mixed.clone()
    };
    let stopped = batchplan::pick(&device, &worlds, &[open_thrice], &o).unwrap();
    let stopped = stopped.outcomes[0].as_ref().unwrap_err();
    assert_eq!(stopped.candidates, vec![failure.candidates[0].clone(); 3]);
    // And where the narrow one is stopped too, by a slab over the box, every grasp's stop comes
    // in the pick's order: the narrow one's between the open ones'.
    let covered = device.upload(&[scene(vec![post(), slab(0.1)])]).unwrap();
    let all = batchplan::pick(&device, &covered, std::slice::from_ref(&mixed), &o).unwrap();
    let all = all.outcomes[0].as_ref().unwrap_err();
    let alone = |task: &Pick| {
        let result = batchplan::pick(&device, &covered, std::slice::from_ref(task), &o).unwrap();
        result.outcomes[0].as_ref().unwrap_err().candidates[0].clone()
    };
    assert_eq!(
        all.candidates,
        [alone(&open), alone(&narrowed), alone(&open)]
    );
    assert_ne!(all.candidates[0], all.candidates[1]);
    let bad = Pick {
        grasps: locked_at(
            &narrowed.grasps,
            [[("panda_joint1".to_string(), 0.0)].into()],
        ),
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

/// A pick that wants only some plans (`Pick::enough`) gives the first that many of its grasps, in
/// their order, that get through, quickest first, each as the pick that tries every grasp gives
/// it: of a hundred grasps of which the sixth, the 41st, the 42nd and the 91st get through, one
/// plan is the sixth's, two are the sixth's and the 41st's (the 42nd, tried with it, left out),
/// and ten are all four. A pick none gets through says where each of the hundred stopped, as
/// without a count; among grasps with fingers of their own the count takes the first that plans;
/// and no plans is not enough.
#[test]
fn a_pick_stops_at_as_many_plans_as_will_do_in_its_grasps_order() {
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    // Beside the post, the grasps closing along x stop and those closing along y get through.
    let through = [5, 40, 41, 90];
    let grasps = (0..100).map(|k| match through.contains(&k) {
        true => top_across(),
        false => top(),
    });
    let every = pick_from(&robot, grasps.collect());
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[scene(vec![post()])]).unwrap();
        let plans_of = |task: &Pick| {
            let result = batchplan::pick(&device, &worlds, std::slice::from_ref(task), &o);
            result.unwrap().outcomes.remove(0)
        };
        let all = plans_of(&every).unwrap();
        let mut by_grasp: Vec<usize> = all.iter().map(|plan| plan.candidate).collect();
        by_grasp.sort_unstable();
        assert_eq!(by_grasp, through, "{name}");
        for (enough, first) in [(1, &through[..1]), (2, &through[..2]), (10, &through[..])] {
            let some = Pick {
                enough: Some(enough),
                ..every.clone()
            };
            let found = plans_of(&some).unwrap();
            // Those of them the whole pick gives, in its order: quickest first.
            let wanted = all.iter().filter(|plan| first.contains(&plan.candidate));
            let wanted: Vec<&TaskPlan> = wanted.collect();
            assert_eq!(found.len(), wanted.len(), "{name}: {enough} will do");
            for (a, b) in found.iter().zip(wanted) {
                assert_eq!(a.candidate, b.candidate, "{name}: {enough} will do");
                // Searches run alike on either device; a GPU's IK rounds by its batch.
                if name.starts_with("cpu") {
                    assert_eq!(a.approach.control_points, b.approach.control_points);
                    assert_eq!(a.depart.control_points, b.depart.control_points);
                }
            }
        }
        let none = Pick {
            grasps: vec![Hold::from(top()); 100],
            ..every.clone()
        };
        let stopped = plans_of(&none).unwrap_err();
        let one = Pick {
            enough: Some(1),
            ..none
        };
        let tried = plans_of(&one).unwrap_err();
        assert_eq!(tried.candidates.len(), 100, "{name}");
        if name.starts_with("cpu") {
            assert_eq!(tried, stopped, "{name}");
        }
        // Fingers narrow for the second of three grasps alone: the first plan is its.
        let narrow: std::collections::HashMap<String, f32> = [
            ("panda_finger_joint1".to_string(), 0.01),
            ("panda_finger_joint2".to_string(), 0.01),
        ]
        .into();
        let mixed = Pick {
            grasps: locked_at(
                &vec![Hold::from(top()); 3],
                [Default::default(), narrow, Default::default()],
            ),
            enough: Some(1),
            ..every.clone()
        };
        let found = plans_of(&mixed).unwrap();
        assert_eq!(found.len(), 1, "{name}");
        assert_eq!(found[0].candidate, 1, "{name}");
        let nothing = Pick {
            enough: Some(0),
            ..every.clone()
        };
        match batchplan::pick(&device, &worlds, &[every.clone(), nothing], &o) {
            Err(Error::Input(message)) => assert!(message.contains("task 1"), "{message}"),
            other => panic!("{name}: no plans being enough is refused: {other:?}"),
        }
    }
}

/// A hold at the gripper's loaded gap, or wider, keeps the fingers as loaded, so that a pick of
/// such holds runs as one of poses; narrower, it locks them where the gripper says.
#[test]
fn a_hold_at_the_loaded_gap_keeps_the_fingers_as_loaded() {
    let robot = common::panda().unwrap();
    let calibration = GripperCalibration::new(0.08, 0.0, common::panda_closed());
    let gripper = robot.gripper(0, &calibration).unwrap();
    for open in [gripper.opening(), 0.09] {
        assert_eq!(Hold::new(top(), open, &gripper), Hold::from(top()));
    }
    let narrow = Hold::new(top(), 0.02, &gripper);
    assert_eq!((narrow.pose, narrow.cost), (top(), 0.0));
    assert_eq!(narrow.locked, gripper.locked_at(0.02));
    assert!(!narrow.locked.is_empty());
}

/// A pick's plans come least cost first, the quickest among equals, and a cost changes nothing of
/// what is tried: of the hundred grasps of which the sixth, the 41st, the 42nd and the 91st get
/// through, every cost alike gives the plans and their order that costs of 0 give, quickest first
/// as before costs; the 91st costing least comes first, the rest quickest first; and with two
/// that will do, they are the first two in the grasps' order to get through, the cheaper first,
/// whatever a later grasp costs. A cost that is not finite is refused, naming the grasp.
#[test]
fn a_picks_plans_come_least_cost_first_and_the_quickest_among_equals() {
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let through = [5, 40, 41, 90];
    let grasps = (0..100).map(|k| match through.contains(&k) {
        true => top_across(),
        false => top(),
    });
    let every = pick_from(&robot, grasps.collect());
    let costing = |cost: &dyn Fn(usize) -> f32| {
        let each = every.grasps.iter().enumerate();
        Pick {
            grasps: each
                .map(|(k, h)| Hold {
                    cost: cost(k),
                    ..h.clone()
                })
                .collect(),
            ..every.clone()
        }
    };
    let candidates = |plans: &[TaskPlan]| plans.iter().map(|p| p.candidate).collect::<Vec<_>>();
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[scene(vec![post()])]).unwrap();
        let plans_of = |task: &Pick| {
            let result = batchplan::pick(&device, &worlds, std::slice::from_ref(task), &o);
            result.unwrap().outcomes.remove(0).unwrap()
        };
        let free = plans_of(&every);
        let quickest_first = free.windows(2).all(|w| {
            let (a, b) = (w[0].duration(), w[1].duration());
            a < b || (a == b && w[0].candidate < w[1].candidate)
        });
        assert!(
            quickest_first && free.iter().all(|p| p.cost == 0.0),
            "{name}"
        );
        let alike = plans_of(&costing(&|_| 2.5));
        assert_eq!(candidates(&alike), candidates(&free), "{name}");
        for (a, b) in alike.iter().zip(&free) {
            assert_eq!(a.cost, 2.5, "{name}");
            assert_eq!(
                a.approach.control_points, b.approach.control_points,
                "{name}"
            );
            assert_eq!(a.depart.control_points, b.depart.control_points, "{name}");
        }
        let last_first = plans_of(&costing(&|k| if k == 90 { 0.0 } else { 1.0 }));
        let rest: Vec<usize> = candidates(&free).into_iter().filter(|&k| k != 90).collect();
        assert_eq!(candidates(&last_first)[0], 90, "{name}");
        assert_eq!(candidates(&last_first)[1..], rest[..], "{name}");
        for (cost, wanted) in [(40, [40, 5]), (90, [5, 40])] {
            let two = Pick {
                enough: Some(2),
                ..costing(&|k| if k == cost { 0.0 } else { 1.0 })
            };
            let found = plans_of(&two);
            let mut first_two = candidates(&found);
            if cost == 90 {
                first_two.sort_unstable();
            }
            assert_eq!(
                first_two, wanted,
                "{name}: the 91st costing least is not tried first"
            );
        }
        let unbounded = costing(&|k| if k == 3 { f32::NAN } else { 0.0 });
        match batchplan::pick(&device, &worlds, &[every.clone(), unbounded], &o) {
            Err(Error::Input(message)) => {
                assert!(
                    message.contains("task 1: grasp 3") && message.contains("NaN"),
                    "{message}"
                )
            }
            other => panic!("{name}: a cost that is not finite is refused: {other:?}"),
        }
    }
}

/// Points 2 mm apart on the top and sides of a box at `at`, as cameras around it would see
/// them, with the normals out of them: the object's, or what stands beside it.
fn seen(at: Vec3, half: Vec3, object: bool, s: &mut Surface) {
    let faces = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z];
    for normal in faces {
        let axis = normal.abs().max_position();
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        let steps = |h: f32| (2.0 * h / 0.002).round() as i32;
        for i in 0..=steps(half[a]) {
            for j in 0..=steps(half[b]) {
                let across = Vec3::AXES[a] * (i as f32 * 0.002 - half[a])
                    + Vec3::AXES[b] * (j as f32 * 0.002 - half[b]);
                s.points.push(at + normal * half[axis] + across);
                s.normals.push(normal);
                s.object.push(object);
            }
        }
    }
}

/// Grasps whose fingers come in narrower (`SurfaceGraspOptions::narrow`) are picked by them: a box
/// too long to close along stands 3 cm from a wall as tall as it, which the open fingers closing
/// across it would be in, so seen as it is it has no grasp. With fingers that may come in
/// narrower it has, each with its opening, a surface's noise wider than the box to each finger,
/// and one pick whose grasps lock the fingers there
/// (`Hold::locked`) plans by them on every device; the same grasps with the fingers as loaded
/// stop at the wall.
#[test]
fn grasps_whose_fingers_come_in_narrower_are_picked_beside_a_wall() {
    let robot = common::panda().unwrap();
    let half = Vec3::new(0.02, 0.07, 0.03);
    let at = Vec3::new(0.45, 0.2, half.z);
    let wall_half = Vec3::new(0.005, 0.09, half.z);
    let wall_at = at + Vec3::X * (half.x + 0.03 + wall_half.x);
    const WALL: usize = 2;
    let world = World::new(vec![
        Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02)),
        Obstacle::cuboid(at, half),
        Obstacle::cuboid(wall_at, wall_half),
    ]);
    let mut surface = Surface::default();
    seen(at, half, /* object */ true, &mut surface);
    seen(wall_at, wall_half, /* object */ false, &mut surface);
    let open = SurfaceGraspOptions {
        floor: Some(0.005),
        ..Default::default()
    };
    let narrow = SurfaceGraspOptions {
        narrow: true,
        ..open
    };
    let calibration = GripperCalibration::new(0.08, 0.0, common::panda_closed());
    let gripper = robot.gripper(0, &calibration).unwrap();
    for device in common::devices(&robot) {
        let name = device.name();
        let on = |o: &SurfaceGraspOptions| {
            let found = batchplan::grasps_on(&device, &gripper, std::slice::from_ref(&surface), o);
            found.unwrap().remove(0)
        };
        assert!(
            on(&open).is_empty(),
            "{name}: the open fingers are in the wall"
        );
        let found = on(&narrow);
        assert!(!found.is_empty(), "{name}");
        // As far from the wall as they can stand: a surface's noise off the box, each finger.
        for g in &found {
            let nearest = 2.0 * half.x + 2.0 * 0.0025;
            assert!((g.opening - nearest).abs() < 1e-6, "{name}: {g:?}");
        }
        let fingers = |opening: f32| -> std::collections::HashMap<String, f32> {
            ["panda_finger_joint1", "panda_finger_joint2"]
                .iter()
                .map(|joint| (joint.to_string(), opening / 2.0))
                .collect()
        };
        let in_box = Pose::new(at, Quat::IDENTITY).inverse();
        let holds = found.iter().map(|g| Hold {
            pose: in_box.mul_pose(g.pose),
            locked: fingers(g.opening),
            cost: 0.0,
        });
        let task = Pick {
            support: vec![TABLE],
            ..Pick::new(0, robot.default_q(), BOX, holds)
        };
        let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
        let o = TaskOptions::default();
        let result = batchplan::pick(&device, &worlds, std::slice::from_ref(&task), &o).unwrap();
        let plans = result.outcomes[0]
            .as_ref()
            .unwrap_or_else(|f| panic!("{name}: {f}"));
        for plan in plans {
            assert_eq!(
                plan.held.as_ref().unwrap().locked,
                task.grasps[plan.candidate].locked,
                "{name}: held as its grasp came in"
            );
        }
        let as_loaded = Pick {
            grasps: task.grasps.iter().map(|h| Hold::from(h.pose)).collect(),
            ..task
        };
        let result = batchplan::pick(&device, &worlds, &[as_loaded], &o).unwrap();
        let failure = result.outcomes[0]
            .as_ref()
            .expect_err("the open fingers come down on the wall");
        // Every grasp the hand can reach, and so every one that planned, stops in the wall.
        let at_wall = |c: &CandidateStop| {
            let walls =
                |contact: &Contact| matches!(contact, Contact::World { obstacle: WALL, .. });
            c.contacts.iter().any(walls)
        };
        for (k, stop) in failure.candidates.iter().enumerate() {
            let reached = !matches!(stop.why, Some(ToolPathStop::Unreachable { .. }));
            assert!(at_wall(stop) || !reached, "{name}: grasp {k}: {stop:?}");
        }
        for plan in plans {
            assert!(at_wall(&failure.candidates[plan.candidate]), "{name}");
        }
    }
}

/// A call that holds objects in many ways has a device for each, and what runs on one waits for
/// work it hands the device's threads, where a thread that waits takes up another's on top of
/// its own: all at once, fifteen hundred ways to hold a box, each set down at either of two
/// turns, stack a thread's calls deeper than its stack. They are set down in one place of them
/// all, each as it is alone.
#[test]
fn as_many_holds_as_fill_a_stack_at_once_are_placed_in_one_call() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let mut after = scene(vec![]);
    let object = after.remove(BOX).unwrap().placed(Pose::IDENTITY);
    let worlds = device.upload(&[after]).unwrap();
    let o = TaskOptions::default();
    // The box held from above a hair deeper each time: no two ways alike.
    const WAYS: usize = 1500;
    let to = Pose::new(Vec3::new(0.45, -0.2, HALF.z), Quat::IDENTITY);
    let places: Vec<Place> = (0..WAYS)
        .map(|k| {
            let held = Held {
                carries: vec![],
                touch_links: vec![],
                also: vec![],
                locked: Default::default(),
                spheres: Default::default(),
                hand: 0,
                object: object.clone(),
                grasp: Pose::new(top().position - Vec3::Z * k as f32 * 4e-6, top().rotation),
            };
            Place {
                support: vec![TABLE],
                ..Place::new(
                    0,
                    robot.default_q(),
                    held,
                    batchplan::place_poses(&object, to, 2),
                )
            }
        })
        .collect();
    let placed = batchplan::place(&device, &worlds, &places, &o).unwrap();
    for (k, outcome) in placed.outcomes.iter().enumerate() {
        let plans = outcome.as_ref().unwrap_or_else(|f| panic!("way {k}: {f}"));
        assert_eq!(plans.len(), 2, "way {k}: either turn");
    }
    let last = &places[WAYS - 1..];
    let alone = batchplan::place(&device, &worlds, last, &o).unwrap();
    let (among, alone) = (&placed.outcomes[WAYS - 1], &alone.outcomes[0]);
    for (a, b) in among.as_ref().unwrap().iter().zip(alone.as_ref().unwrap()) {
        assert_eq!(a.candidate, b.candidate);
        assert_eq!(a.approach.control_points, b.approach.control_points);
        assert_eq!(
            a.transit.as_ref().unwrap().control_points,
            b.transit.as_ref().unwrap().control_points
        );
    }
}

/// A pick whose grasps lock the fingers each their own way has a relocked robot for each, and
/// what runs on one waits for work it hands the device's threads as a held object's does: five
/// thousand openings, two grasps at each, are picked by in one call, the last as it is alone.
#[test]
fn as_many_finger_openings_as_fill_a_stack_at_once_are_picked_by_in_one_call() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[scene(vec![])]).unwrap();
    let o = TaskOptions::default();
    const WAYS: usize = 5000;
    // From as loaded, 8 cm, to 6 cm: every one wider than the box.
    let fingers = |way: usize| -> std::collections::HashMap<String, f32> {
        let at = 0.04 - way as f32 * 2e-6;
        ["panda_finger_joint1", "panda_finger_joint2"]
            .iter()
            .map(|joint| (joint.to_string(), at))
            .collect()
    };
    let every = Pick {
        grasps: locked_at(
            &vec![Hold::from(top()); 2 * WAYS],
            (0..2 * WAYS).map(|grasp| fingers(grasp / 2)),
        ),
        ..pick_from(&robot, vec![])
    };
    let picked = batchplan::pick(&device, &worlds, std::slice::from_ref(&every), &o).unwrap();
    let plans = picked.outcomes[0].as_ref().unwrap();
    assert_eq!(plans.len(), 2 * WAYS);
    let last = plans.iter().find(|plan| plan.candidate == 2 * WAYS - 1);
    let alone = Pick {
        grasps: locked_at(&[Hold::from(top())], [fingers(WAYS - 1)]),
        ..pick_from(&robot, vec![])
    };
    let alone = batchplan::pick(&device, &worlds, &[alone], &o).unwrap();
    let alone = &alone.outcomes[0].as_ref().unwrap()[0];
    let last = last.expect("the last grasp's plan");
    assert_eq!(last.approach.control_points, alone.approach.control_points);
    assert_eq!(last.held.as_ref().unwrap().locked, fingers(WAYS - 1));
}

/// A place may say how far its hand opens where it lets go (`Place::open`): between two posts 3 cm
/// off the box's sides, where the fingers as loaded come down on the posts, a place that names
/// no opening stops at its release, and one that names some lets go at the first of them clear of
/// the posts, says which, and backs off with the fingers there. Around a place that names none,
/// in one call, two that do plan alike. With only the loaded opening to choose from it gets to
/// where it lets go and stops there, at the posts; one that then finds no transit says nothing
/// of the openings that stopped; an opening that names an arm's joint is refused by its task and
/// place in the list.
#[test]
fn a_place_lets_go_as_wide_as_it_has_room_to() {
    let robot = common::panda().unwrap();
    let fingers = |at: f32| -> std::collections::HashMap<String, f32> {
        [
            ("panda_finger_joint1".to_string(), at),
            ("panda_finger_joint2".to_string(), at),
        ]
        .into()
    };
    let spot = Vec3::new(0.45, -0.2, HALF.z);
    let post = |side: f32| {
        Obstacle::cuboid(
            spot + Vec3::X * side * (HALF.x + 0.03 + 0.005),
            Vec3::new(0.005, 0.02, HALF.z),
        )
    };
    let before = scene(vec![post(1.0), post(-1.0)]);
    let mut after = before.clone();
    let object = after.remove(BOX).unwrap().placed(Pose::IDENTITY);
    // Without the box, as the places see it: the table, then the posts.
    let posts = [1, 2];
    // Something in the hand's way where it lets go: in place of a post, for a move away in a
    // world of its own, and beside both, for a release that cannot be reached.
    let block = Obstacle::cuboid(spot + Vec3::Z * 0.16, Vec3::splat(0.03));
    let mut backing = after.clone();
    backing.obstacles[posts[1]] = block.clone();
    let mut taken = after.clone();
    taken.obstacles.push(block);
    let the_block = taken.obstacles.len() - 1;
    let o = TaskOptions::default();
    let open = vec![fingers(0.04), fingers(0.03), fingers(0.02), fingers(0.01)];
    for device in common::devices(&robot) {
        let scenes = [
            before.clone(),
            after.clone(),
            backing.clone(),
            taken.clone(),
        ];
        let worlds = device.upload(&scenes).unwrap();
        let held = Pick {
            grasps: locked_at(&[Hold::from(top())], [fingers(0.01)]),
            ..pick_from(&robot, vec![])
        };
        let picked = batchplan::pick(&device, &worlds, &[held], &o).unwrap();
        let picked = &picked.outcomes[0].as_ref().unwrap()[0];
        let unnamed = Place {
            support: vec![TABLE],
            ..Place::new(
                1,
                picked.end(),
                picked.held.clone().unwrap(),
                vec![Pose::new(spot, Quat::IDENTITY)],
            )
        };
        let named = Place {
            open: open.clone(),
            ..unnamed.clone()
        };
        let tasks = [named.clone(), unnamed.clone(), named.clone()];
        let placed = batchplan::place(&device, &worlds, &tasks, &o).unwrap();
        let of_posts =
            |c: &Contact| matches!(c, Contact::World { obstacle, .. } if posts.contains(obstacle));
        let stopped = placed.outcomes[1]
            .as_ref()
            .expect_err("the fingers as loaded come down on the posts");
        assert_eq!(stopped.candidates[0].step, TaskStep::Reach, "{stopped}");
        assert!(
            stopped.candidates[0].contacts.iter().any(of_posts),
            "{stopped}"
        );
        assert_eq!(
            format!("{:?}", placed.outcomes[0]),
            format!("{:?}", placed.outcomes[2]),
            "those that name openings alike, around the one that names none"
        );
        let plan = &placed.outcomes[0]
            .as_ref()
            .unwrap_or_else(|f| panic!("{f}: {:?}", f.candidates[0].contacts))[0];
        // The first opening clear of the posts where the hand lets go is the plan's, and not
        // the first named.
        let n = robot.dof();
        let release = &plan.depart.control_points[..n];
        let clear = |open: &std::collections::HashMap<String, f32>| {
            let relocked = device.with_robot(&robot.relock(open).unwrap()).unwrap();
            let contacts = relocked.contacts(&worlds, &[1], release, 0.0).unwrap();
            !contacts[0].iter().any(of_posts)
        };
        let first = open.iter().position(clear).expect("an opening is clear");
        assert!(
            first > 0 && first + 1 < open.len(),
            "opening {first} of {open:?}"
        );
        assert_eq!(plan.opening, Some(first));
        for t in plan.motions() {
            t.check(&robot).unwrap();
        }
        assert_eq!(
            plan.approach.control_points[plan.approach.control_points.len() - n..],
            *release
        );
        let at = robot.ee_pose(release, 0);
        let backed = robot.ee_pose(plan.end(), 0);
        let behind = at.position - at.rotation * Vec3::Z * named.retreat;
        assert!(
            backed.position.distance(behind) < 2e-3,
            "{backed:?} from {at:?}"
        );
        assert_eq!(
            plan.placed,
            Some(object.placed(Pose::new(spot, Quat::IDENTITY)))
        );

        // Each place backs off in its own world: after one set down in the open, one whose
        // hand is in a block where it backs off stops there, though the first's world is clear.
        let aside = Place {
            releases: vec![Pose::new(spot - Vec3::X * 0.15, Quat::IDENTITY).into()],
            ..unnamed.clone()
        };
        let blocked = Place {
            open: vec![open[first].clone()],
            depart_world: Some(2),
            ..unnamed.clone()
        };
        let placed = batchplan::place(&device, &worlds, &[aside, blocked], &o).unwrap();
        assert!(placed.outcomes[0].is_ok(), "{:?}", placed.outcomes[0]);
        let stopped = placed.outcomes[1]
            .as_ref()
            .expect_err("the hand backs off into the block");
        assert!(
            all_stop(stopped, TaskStep::Depart, Some(posts[1])),
            "{stopped:?}"
        );

        // A release the arm cannot reach says what the arm meets, the fingers where they hold
        // the box: the block, and neither post.
        let unreached = Place {
            world: 3,
            ..named.clone()
        };
        let placed = batchplan::place(&device, &worlds, &[unreached], &o).unwrap();
        let stopped = placed.outcomes[0].as_ref().unwrap_err();
        assert!(
            all_stop(stopped, TaskStep::Reach, Some(the_block)),
            "{stopped:?}"
        );
        let contacts = &stopped.candidates[0].contacts;
        assert!(!contacts.iter().any(of_posts), "{contacts:?}");

        // With the loaded opening alone to choose from, the release is reached, the fingers
        // where they hold the box, and the hand cannot open there.
        let loaded = Place {
            open: vec![Default::default()],
            ..unnamed.clone()
        };
        let placed = batchplan::place(&device, &worlds, &[loaded], &o).unwrap();
        let stopped = placed.outcomes[0].as_ref().unwrap_err();
        assert_eq!(stopped.candidates[0].step, TaskStep::Depart, "{stopped}");
        assert!(
            stopped.candidates[0].contacts.iter().any(of_posts),
            "{stopped}"
        );

        // Kept within 20 cm of where the pick ends, the hand cannot get to the posts: the place
        // stops at its transit, and says nothing of the openings it could not let go at.
        let near = HandConstraint {
            position: Vec3::splat(0.2),
            ..HandConstraint::new(0, robot.ee_pose(picked.end(), 0))
        };
        let confined = Place {
            constraints: vec![near],
            ..named.clone()
        };
        let placed = batchplan::place(&device, &worlds, &[confined], &o).unwrap();
        let stopped = placed.outcomes[0].as_ref().unwrap_err();
        assert!(all_stop(stopped, TaskStep::Transit, None), "{stopped:?}");

        let arm = Place {
            open: vec![fingers(0.02), [("panda_joint1".to_string(), 0.0)].into()],
            ..unnamed.clone()
        };
        match batchplan::place(&device, &worlds, &[unnamed, arm], &o) {
            Err(Error::Input(e)) => assert!(e.contains("task 1: opening 1"), "{e}"),
            other => panic!("an opening that names an arm's joint is refused: {other:?}"),
        }
    }
}

/// A place that wants only some plans (`Place::enough`) gives the first that many of its poses, in
/// their order, that get through, quickest first, each as the place that tries every pose gives
/// it: of a hundred poses of which the sixth, the 41st, the 42nd and the 91st are clear of a block,
/// one plan is the sixth's, two are the sixth's and the 41st's (the 42nd, tried with it, left out),
/// and ten are all four. In one call a place that one plan will do for and one that wants ten
/// each get theirs. A place none gets through says where each of the hundred stopped, as without
/// a count; and no plans is not enough.
#[test]
fn a_place_stops_at_as_many_plans_as_will_do_in_its_poses_order() {
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let free = Pose::new(Vec3::new(0.45, -0.2, HALF.z), Quat::IDENTITY);
    let taken = Pose::new(Vec3::new(0.3, -0.25, HALF.z), Quat::IDENTITY);
    let before = scene(vec![Obstacle::cuboid(taken.position, Vec3::splat(0.03))]);
    let mut after = before.clone();
    after.remove(BOX).unwrap();
    let through = [5, 40, 41, 90];
    let poses = (0..100).map(|k| match through.contains(&k) {
        true => free,
        false => taken,
    });
    let poses: Vec<Pose> = poses.collect();
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[before.clone(), after.clone()]).unwrap();
        let picked = batchplan::pick(&device, &worlds, &[pick_from(&robot, vec![top()])], &o);
        let picked = picked.unwrap();
        let picked = &picked.outcomes[0].as_ref().unwrap()[0];
        let every = Place {
            support: vec![TABLE],
            ..Place::new(1, picked.end(), picked.held.clone().unwrap(), poses.clone())
        };
        let plans_of = |tasks: &[Place]| batchplan::place(&device, &worlds, tasks, &o).unwrap();
        let all = plans_of(std::slice::from_ref(&every))
            .outcomes
            .remove(0)
            .unwrap();
        let mut by_pose: Vec<usize> = all.iter().map(|plan| plan.candidate).collect();
        by_pose.sort_unstable();
        assert_eq!(by_pose, through, "{name}");
        let some = |enough: usize| Place {
            enough: Some(enough),
            ..every.clone()
        };
        let same = |found: &[TaskPlan], first: &[usize], what: &str| {
            // Those of them the whole place gives, in its order: quickest first.
            let wanted = all.iter().filter(|plan| first.contains(&plan.candidate));
            let wanted: Vec<&TaskPlan> = wanted.collect();
            assert_eq!(found.len(), wanted.len(), "{name}: {what}");
            for (a, b) in found.iter().zip(wanted) {
                assert_eq!(a.candidate, b.candidate, "{name}: {what}");
                assert_eq!(a.placed, b.placed, "{name}: {what}");
                // Searches run alike on either device; a GPU's IK rounds by its batch.
                if name.starts_with("cpu") {
                    assert_eq!(a.approach.control_points, b.approach.control_points);
                    assert_eq!(a.depart.control_points, b.depart.control_points);
                }
            }
        };
        for (enough, first) in [(1, &through[..1]), (2, &through[..2]), (10, &through[..])] {
            let found = plans_of(&[some(enough)]).outcomes.remove(0).unwrap();
            same(&found, first, &format!("{enough} will do"));
        }
        // One that is done after its first poses beside one that goes on to its last.
        let both = plans_of(&[some(1), some(10)]).outcomes;
        same(both[0].as_ref().unwrap(), &through[..1], "one, beside ten");
        same(both[1].as_ref().unwrap(), &through, "ten, beside one");

        let none = Place {
            releases: vec![taken.into(); 100],
            ..every.clone()
        };
        let stopped = plans_of(std::slice::from_ref(&none)).outcomes.remove(0);
        let stopped = stopped.unwrap_err();
        let one = Place {
            enough: Some(1),
            ..none
        };
        let tried = plans_of(&[one]).outcomes.remove(0).unwrap_err();
        assert_eq!(tried.candidates.len(), 100, "{name}");
        assert!(
            all_stop(&tried, TaskStep::Reach, Some(1)),
            "{name}: {tried}"
        );
        if name.starts_with("cpu") {
            assert_eq!(tried, stopped, "{name}");
        }
        match batchplan::place(&device, &worlds, &[every.clone(), some(0)], &o) {
            Err(Error::Input(message)) => assert!(message.contains("task 1"), "{message}"),
            other => panic!("{name}: no plans being enough is refused: {other:?}"),
        }
    }
}

#[test]
fn gripping_one_part_lifts_the_whole_body_and_keeps_its_neighbors() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let body = scene(vec![]).obstacles[BOX].clone();
    let center = body.pose().position;
    let patch =
        Obstacle::cuboid(center + 0.015 * Vec3::Z, Vec3::new(HALF.x, HALF.y, 0.015)).placed(
            Pose::new(center + Vec3::Z * 0.015, Quat::from_rotation_z(PI)),
        );
    let rest = Obstacle::cuboid(center - 0.015 * Vec3::Z, Vec3::new(HALF.x, HALF.y, 0.015));
    let table = scene(vec![]).obstacles[TABLE].clone();
    let grasp = body.pose().mul_pose(top());
    let local = patch.pose().inverse().mul_pose(grasp);
    let mut patch_pose = patch.pose();
    patch_pose.rotation *= 1.0008;
    let patch = patch.placed(patch_pose);
    let local = Pose::new(local.position, local.rotation * 1.0008);
    assert!(patch_pose.is_unit() && local.is_unit());
    assert!(
        !patch_pose.mul_pose(local).is_unit(),
        "the two rounding errors accumulate"
    );
    let task = Pick {
        body: Some(batchplan::PickBody {
            shape: body.clone(),
            other_parts: vec![0],
        }),
        support: vec![2],
        ..Pick::new(0, robot.default_q(), 1, vec![local])
    };
    let mut world = World::new(vec![rest, patch, table.clone()]);
    let after = task.lift_world(&world).unwrap();
    assert_eq!(after.obstacles, [table]);
    assert_eq!(after.touch, [Touch::held(0, 0)]);
    // Overlapping indices erase collidable material or leave a support pointing at another
    // obstacle. Check the step world on its own and before planning.
    let uploaded = device.upload(std::slice::from_ref(&world)).unwrap();
    for invalid in [vec![0, 0], vec![1], vec![2], vec![3]] {
        let mut bad = task.clone();
        bad.body.as_mut().unwrap().other_parts = invalid;
        assert!(matches!(bad.lift_world(&world), Err(Error::Input(_))));
        assert!(matches!(
            batchplan::pick(&device, &uploaded, &[bad], &TaskOptions::default()),
            Err(Error::Input(_))
        ));
    }
    let plans = pick_in(
        &device,
        world.clone(),
        task.clone(),
        &TaskOptions::default(),
    )
    .unwrap();
    let held = plans[0].held.as_ref().unwrap();
    assert_eq!(held.object, body.placed(Pose::IDENTITY));
    assert!(held.grasp.position.distance(top().position) < 1e-6);
    assert!(held.grasp.rotation.dot(top().rotation).abs() > 0.99999);

    // A lid over the contact part stops the lifted body; it is still obstacle 3 in diagnostics.
    world.obstacles.push(Obstacle::cuboid(
        Vec3::new(0.45, 0.2, 0.068),
        Vec3::new(0.01, 0.015, 0.004),
    ));
    let failure = pick_in(&device, world, task, &TaskOptions::default()).unwrap_err();
    assert_eq!(failure.candidates[0].step, TaskStep::Depart);
    assert!(
        failure.candidates[0]
            .contacts
            .iter()
            .any(|contact| matches!(contact, Contact::World { obstacle: 3, .. })),
        "{failure:?}"
    );
}

#[test]
fn a_pick_can_fit_its_carried_cylinder_beside_a_wall() {
    let robot = common::panda().unwrap();
    let device = Device::cpu_threads(&robot, 4).unwrap();
    let center = Vec3::new(0.45, 0.2, 0.023);
    let radius = 0.056;
    let mut w = scene(vec![]);
    w.obstacles[BOX] = Obstacle::cylinder(center, radius, 0.023);
    let wall = |gap| {
        Obstacle::cuboid(
            center + Vec3::X * (radius + gap + 0.005),
            Vec3::new(0.005, 0.09, 0.055),
        )
    };
    let side = w.obstacles.len();
    w.obstacles.push(wall(0.0035));
    let grasp = Pose::new(Vec3::new(-0.03, 0.0, 0.022), top_across().rotation);
    let task = pick_from(&robot, vec![grasp]);
    let ordinary = TaskOptions::default();
    let f = pick_in(&device, w.clone(), task.clone(), &ordinary).unwrap_err();
    assert_eq!(f.candidates[0].step, TaskStep::Depart);
    let precise = TaskOptions {
        held_spheres: batchplan::SphereOptions {
            budget: 256,
            tolerance: 0.001,
            ..Default::default()
        },
        ..ordinary
    };
    let plans = pick_in(&device, w.clone(), task.clone(), &precise)
        .expect("the finer cylinder clears the wall");
    assert_eq!(
        plans[0].held.as_ref().unwrap().spheres,
        precise.held_spheres
    );
    w.obstacles[side] = wall(-0.0035);
    let failure = pick_in(&device, w, task, &precise).unwrap_err();
    assert_eq!(failure.candidates[0].step, TaskStep::Depart);
    assert!(failure.candidates[0].contacts.iter().any(|contact|
        matches!(contact, Contact::World { link, obstacle, .. } if link == "held by hand 0" && *obstacle == side)
    ));
}

#[test]
fn the_release_sweep_stops_a_palm_below_the_falling_object() {
    let robot = common::panda().unwrap();
    let held = Held {
        carries: vec![],
        touch_links: vec![],
        also: vec![],
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, HALF),
        grasp: grasp(Vec3::Z, Vec3::X),
        locked: Default::default(),
        spheres: Default::default(),
    };
    let release = Pose::new(Vec3::new(0.5, -0.12, 0.55), Quat::IDENTITY);
    let task = Place {
        transit: false,
        ..Place::new(0, robot.default_q(), held, vec![release])
    };
    for device in common::devices(&robot) {
        let worlds = device.upload(&[World::default()]).unwrap();
        let plan = |task: &Place| {
            batchplan::place(
                &device,
                &worlds,
                std::slice::from_ref(task),
                &TaskOptions::default(),
            )
            .unwrap()
            .outcomes
            .remove(0)
        };
        assert!(plan(&task).is_ok(), "the palm is clear at release");
        let falling = Place {
            releases: dropping(&task.releases, [0.1]),
            ..task.clone()
        };
        let stopped = plan(&falling).expect_err("the drop passes through the palm");
        let stop = &stopped.candidates[0];
        assert_eq!(stop.step, TaskStep::Depart);
        assert!(stop.contacts.iter().any(|c| matches!(c, Contact::World { link, obstacle: 0, distance } if link == "panda_hand" && *distance < 0.0)), "{stop:?}");
    }
}

/// Holds costing their grasps' unseen span (`Grasp::unseen_span`) pick those closing on two seen
/// patches first, then those of the least span: of seven holds alike but for what their grasps
/// saw, the four seen on both sides, then the two one-sided across 2.5 cm, then the one across
/// 6.5 cm. A span that is not finite is refused as the hold's cost.
#[test]
fn completed_picks_prefer_tightly_measured_contacts_by_their_cost() {
    let robot = common::panda().unwrap();
    let device = Device::cpu_threads(&robot, 1).unwrap();
    let world = scene(vec![]);
    let mut surface = Surface::default();
    seen(
        world.obstacles[BOX].pose().position,
        HALF,
        true,
        &mut surface,
    );
    let calibration = GripperCalibration::new(0.08, 0.0, std::collections::HashMap::new());
    let gripper = robot.gripper(0, &calibration).unwrap();
    let mut found = batchplan::grasps_on(
        &device,
        &gripper,
        &[surface],
        &SurfaceGraspOptions::default(),
    )
    .unwrap();
    let observed = found[0].pop().expect("the box has opposed measured faces");
    assert!(!observed.one_sided);
    assert_eq!(observed.unseen_span(), 0.0);

    let mut grasps = [observed; 7];
    grasps[6].contacts = [Vec3::ZERO, Vec3::Y * 0.075];
    grasps[2].contacts = [Vec3::ZERO, Vec3::Y * 0.010];
    for (index, width) in [(0, 0.065), (3, 0.025), (5, 0.025)] {
        grasps[index].one_sided = true;
        grasps[index].contacts = [Vec3::ZERO, Vec3::Y * width];
    }
    let spans = [0.065, 0.0, 0.0, 0.025, 0.0, 0.025, 0.0];
    for (g, span) in grasps.iter().zip(spans) {
        assert!((g.unseen_span() - span).abs() < 1e-7, "{g:?}");
    }
    let worlds = device.upload(&[world]).unwrap();
    let held = |grasps: &[batchplan::Grasp]| {
        let each = grasps.iter().map(|g| Hold {
            cost: g.unseen_span(),
            ..Hold::from(top_across())
        });
        Pick {
            grasps: each.collect(),
            ..pick_from(&robot, vec![])
        }
    };
    let o = TaskOptions::default();
    let mut result = batchplan::pick(&device, &worlds, &[held(&grasps)], &o).unwrap();
    let plans = result.outcomes.remove(0).unwrap();
    let lot = |range: std::ops::Range<usize>| {
        let mut lot: Vec<usize> = plans[range].iter().map(|p| p.candidate).collect();
        lot.sort_unstable();
        lot
    };
    assert_eq!(
        (lot(0..4), lot(4..6), lot(6..7)),
        (vec![1, 2, 4, 6], vec![3, 5], vec![0])
    );
    let mut invalid = grasps;
    invalid[3].contacts[1].y = f32::NAN;
    match batchplan::pick(&device, &worlds, &[held(&invalid)], &o) {
        Err(Error::Input(message)) => assert!(message.contains("grasp 3"), "{message}"),
        other => panic!("a span that is not finite is refused: {other:?}"),
    }
}

/// Where the box is set down in the tests of releases' costs: beside a post, and in the open.
const BESIDE: Vec3 = Vec3::new(0.45, -0.2, HALF.z);
const OPEN: Vec3 = Vec3::new(0.5, -0.05, HALF.z);

/// The table without the box, and a post off the +x face of a box set down at `BESIDE`, as
/// `post` stands off it where it is picked: an open finger of a hold closing along x comes down on
/// it, and one closing along y passes it.
fn place_scene() -> World {
    let post = Obstacle::cuboid(
        BESIDE + Vec3::X * (HALF.x + 0.02 + 0.005),
        Vec3::new(0.005, 0.02, HALF.z),
    );
    let mut world = scene(vec![post]);
    world.remove(BOX).unwrap();
    world
}

/// A place in world 1, onto the table, per pick of the box from world 0 by one of `grasps`: from
/// where that pick leaves the hand, holding the box as it does. The poses are left to give.
fn holds(device: &Device, worlds: &Worlds, robot: &Robot, grasps: &[Pose]) -> Vec<Place> {
    let picks: Vec<Pick> = grasps.iter().map(|&g| pick_from(robot, vec![g])).collect();
    let picked = batchplan::pick(device, worlds, &picks, &TaskOptions::default()).unwrap();
    let each = picked.outcomes.into_iter().map(|outcome| {
        let plan = outcome.unwrap().remove(0);
        Place {
            support: vec![TABLE],
            ..Place::new(
                1,
                plan.end(),
                plan.held.clone().unwrap(),
                Vec::<Release>::new(),
            )
        }
    });
    each.collect()
}

/// How far from `near` in x and y the plan sets the object down.
fn off(plan: &TaskPlan, near: Vec3) -> f32 {
    let at = plan.placed.as_ref().unwrap().pose().position;
    at.truncate().distance(near.truncate())
}

/// `poses` as releases, each costing its distance in x and y from `near`.
fn costing_from(near: Vec3, poses: &[Pose]) -> Vec<Release> {
    let each = poses.iter().map(|&pose| Release {
        cost: pose.position.truncate().distance(near.truncate()),
        ..Release::from(pose)
    });
    each.collect()
}

/// Of places of the box from two holds, each releasing it beside the post or in the open, each
/// release costing how far it is from the spot beside the post and one plan doing: the hold
/// closing along x stops beside the post and gives the one in the open, the one closing along y
/// sets it down beside the post, and the least of their first plans' costs is that one's.
#[test]
fn of_several_holds_the_least_first_plan_cost_sets_the_box_down_nearest_the_spot() {
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let poses = [
        Pose::new(BESIDE, Quat::IDENTITY),
        Pose::new(OPEN, Quat::IDENTITY),
    ];
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[scene(vec![]), place_scene()]).unwrap();
        // Closing along x, then along y: the first's finger meets the post beside the first pose.
        let tasks = holds(&device, &worlds, &robot, &[top(), top_across()])
            .into_iter()
            .map(|t| Place {
                releases: costing_from(BESIDE, &poses),
                enough: Some(1),
                ..t
            });
        let tasks: Vec<Place> = tasks.collect();
        let each = batchplan::place(&device, &worlds, &tasks, &o).unwrap();
        let first = |t: usize| &each.outcomes[t].as_ref().unwrap()[0];
        assert_eq!((first(0).candidate, first(1).candidate), (1, 0), "{name}");
        assert!(
            (first(0).cost - off(first(0), BESIDE)).abs() < 1e-6,
            "{name}"
        );
        assert_eq!(first(1).cost, 0.0, "{name}");
        let least = (0..tasks.len()).min_by(|&a, &b| first(a).cost.total_cmp(&first(b).cost));
        assert_eq!(least, Some(1), "{name}");
    }
}

/// Releases alike in cost come quickest first, not in their order: the box at the same spot both
/// ways round, from either hold, each pair costing alike and given the slower first.
#[test]
fn releases_alike_in_cost_come_quickest_first() {
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let near = OPEN + Vec3::new(0.03, 0.04, 0.0);
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[scene(vec![]), place_scene()]).unwrap();
        let tasks = holds(&device, &worlds, &robot, &[top_across(), top()])
            .into_iter()
            .map(|t| {
                let both =
                    batchplan::place_poses(&t.held.object, Pose::new(OPEN, Quat::IDENTITY), 1);
                Place {
                    releases: costing_from(near, &both),
                    ..t
                }
            });
        let tasks: Vec<Place> = tasks.collect();
        let each = batchplan::place(&device, &worlds, &tasks, &o).unwrap();
        // Each task's releases again, the quicker of the two last.
        let slower_first = tasks.iter().zip(&each.outcomes).map(|(t, outcome)| {
            let plans = outcome.as_ref().unwrap();
            assert_eq!(plans.len(), 2, "{name}: both ways");
            assert!(
                plans[0].duration() < plans[1].duration(),
                "{name}: one is quicker"
            );
            let order = [plans[1].candidate, plans[0].candidate];
            Place {
                releases: order.map(|k| t.releases[k]).to_vec(),
                ..t.clone()
            }
        });
        let slower_first: Vec<Place> = slower_first.collect();
        let again = batchplan::place(&device, &worlds, &slower_first, &o).unwrap();
        for outcome in &again.outcomes {
            let plans = outcome.as_ref().unwrap();
            assert_eq!(plans[0].cost, plans[1].cost, "{name}");
            assert_eq!(
                (plans[0].candidate, plans[1].candidate),
                (1, 0),
                "{name}: the later, quicker first"
            );
        }
    }
}

/// A place that wants one plan tries its releases in their order, whatever they cost: given
/// beside the post and then in the open, it gives the one beside the post; given nearest the open
/// spot first, as `rests_on` gives them about a point, the one in the open, let go of 2 cm over
/// the table and falling there, numbered as given.
#[test]
fn a_place_that_wants_one_plan_tries_its_releases_in_their_order() {
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let beside = Release::from(Pose::new(BESIDE, Quat::IDENTITY));
    let open = Release {
        drop: 0.02,
        ..Release::from(Pose::new(OPEN + Vec3::Z * 0.02, Quat::IDENTITY))
    };
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[scene(vec![]), place_scene()]).unwrap();
        // Closing along y, the box goes down beside the post, and in the open.
        let from = holds(&device, &worlds, &robot, &[top_across()]).remove(0);
        let first_of = |releases: Vec<Release>| {
            let task = Place {
                releases,
                enough: Some(1),
                ..from.clone()
            };
            let each = batchplan::place(&device, &worlds, std::slice::from_ref(&task), &o).unwrap();
            let mut plans = each.outcomes.into_iter().next().unwrap().unwrap();
            assert_eq!(plans.len(), 1, "{name}");
            plans.remove(0)
        };
        let given = first_of(costing_from(OPEN, &[beside.pose, open.pose]));
        assert!(
            given.candidate == 0 && off(&given, OPEN) > 0.1,
            "{name}: in the releases' order, the one beside the post"
        );
        let nearest = first_of(vec![
            Release { cost: 0.0, ..open },
            Release {
                cost: off(&given, OPEN),
                ..beside
            },
        ]);
        assert_eq!(nearest.candidate, 0, "{name}: numbered as given");
        let landing = nearest.placed.as_ref().unwrap().pose().position;
        assert!(
            landing.distance(OPEN) < 1e-3,
            "{name}: on the table at the spot, {} m off",
            landing.distance(OPEN)
        );
        assert!(nearest.falling.is_some(), "{name}");
    }
}

#[test]
fn a_place_checks_which_device_prepared_its_worlds_with_no_tasks_too() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let other = Device::cpu(&robot).unwrap();
    let worlds = other.upload(&[place_scene()]).unwrap();
    let o = TaskOptions::default();
    let error = batchplan::place(&device, &worlds, &[], &o)
        .map(drop)
        .unwrap_err();
    assert!(
        matches!(error, Error::Input(_)) && error.to_string().contains("different device"),
        "{error}"
    );
}
