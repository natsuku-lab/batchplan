//! Turning a thing about a line in one call (`turn`): a knob in place, a lever about its hinge, a
//! cap along its thread; a turn the configuration nearest the start cannot make is made by
//! another; one that meets an obstacle says how far it got; a batch gives each task what it gives
//! alone.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::f32::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_3, PI, TAU};

use batchplan::rng::Rng;
use batchplan::{
    Contact, Device, Error, Held, IkProblem, Obstacle, Pose, Robot, Screw, SdfGrid, TaskFailure,
    TaskOptions, TaskPlan, TaskStep, ToolPath, ToolPathOptions, ToolPathStop, Touch, Trajectory,
    Turn, World,
};
use glam::{Mat3, Quat, Vec2, Vec3};

const TABLE: usize = 0;
const KNOB: usize = 1;
/// A knob's bar: 9 cm long, 2.5 cm wide, 3 cm tall.
const BAR: Vec3 = Vec3::new(0.045, 0.0125, 0.015);

fn table() -> Obstacle {
    Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02))
}

/// A knob standing on the table at `at`, turned `yaw` about the vertical.
fn knob(at: Vec2, yaw: f32) -> Obstacle {
    Obstacle::cuboid(Vec3::ZERO, BAR)
        .placed(Pose::new(at.extend(BAR.z), Quat::from_rotation_z(yaw)))
}

/// A grasp facing `toward` and closing along `closing`, in the object's frame, `depth` in from
/// the face it comes to, `shift` from the middle.
fn grasp(toward: Vec3, closing: Vec3, half: Vec3, depth: f32, shift: Vec3) -> Pose {
    let rotation = Quat::from_mat3(&Mat3::from_cols(closing.cross(toward), closing, toward));
    Pose::new(shift - toward * (half.dot(toward.abs()) - depth), rotation)
}

/// From above at the knob's middle, closing across it either way round, 1 cm under its top.
fn from_above(closing: f32) -> Pose {
    grasp(-Vec3::Z, Vec3::Y * closing, BAR, 0.01, Vec3::ZERO)
}

/// Hand 0 turns the knob, which stands on the table, by any of `grasps`, `angle` in place.
fn on_table(robot: &Robot, grasps: Vec<Pose>, angle: f32) -> Turn {
    Turn {
        support: vec![TABLE],
        ..Turn::new(0, robot.default_q(), KNOB, grasps, angle)
    }
}

fn turn_in(
    device: &Device,
    world: World,
    task: Turn,
    o: &TaskOptions,
) -> Result<Vec<TaskPlan>, TaskFailure> {
    let worlds = device.upload(&[world]).unwrap();
    batchplan::turn(device, &worlds, &[task], o)
        .unwrap()
        .outcomes
        .remove(0)
}

fn first(t: &Trajectory) -> Vec<f32> {
    t.control_points[..t.dof].to_vec()
}

fn last(t: &Trajectory) -> Vec<f32> {
    t.control_points[t.control_points.len() - t.dof..].to_vec()
}

/// The hand's poses along `t`, 64 times per knot interval.
fn hand_poses(robot: &Robot, t: &Trajectory) -> Vec<Pose> {
    let samples = t.sample(64.0 / t.knot_interval).unwrap();
    let poses = samples.positions.chunks(robot.dof());
    poses.map(|q| robot.ee_pose(q, 0)).collect()
}

/// The plan's motions keep the limits and join, from the transit to the move away, and its turn
/// starts at the grasp `at`.
fn check_joined(robot: &Robot, plan: &TaskPlan, at: Pose, name: &str) -> Trajectory {
    for t in plan.motions() {
        t.check(robot).unwrap();
    }
    let turn = plan.turn.clone().expect("a turn's plan turns");
    let transit = plan.transit.as_ref().expect("the task plans its transit");
    assert_eq!(last(transit), first(&plan.approach), "{name}");
    assert_eq!(last(&plan.approach), first(&turn), "{name}");
    assert_eq!(last(&turn), first(&plan.depart), "{name}");
    let grasped = robot.ee_pose(&first(&turn), 0);
    assert!(grasped.position.distance(at.position) < 1e-3, "{name}");
    assert!(grasped.rotation.angle_between(at.rotation) < 1e-2, "{name}");
    turn
}

/// A turn's plans come least cost first, the quickest among equals: of the knob's two grasps from
/// above, given the quicker first and costing more, the later, slower one comes first, and alike
/// in cost the quicker does.
#[test]
fn a_turns_plans_come_least_cost_first() {
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let world = World::new(vec![table(), knob(Vec2::new(0.5, 0.05), 0.3)]);
    for device in common::devices(&robot) {
        let name = device.name();
        let both = vec![from_above(1.0), from_above(-1.0)];
        let free = turn_in(
            &device,
            world.clone(),
            on_table(&robot, both.clone(), FRAC_PI_2),
            &o,
        );
        let free = free.unwrap();
        assert!(
            free[0].duration() < free[1].duration(),
            "{name}: one is quicker"
        );
        // The quicker first in the list, costing 1, the slower after it, costing 0.
        let order = [free[0].candidate, free[1].candidate];
        let costing = |costs: [f32; 2]| {
            let holds = (order.iter().zip(costs)).map(|(&k, cost)| batchplan::Hold {
                cost,
                ..both[k].into()
            });
            Turn {
                grasps: holds.collect(),
                ..on_table(&robot, vec![], FRAC_PI_2)
            }
        };
        let plans = turn_in(&device, world.clone(), costing([1.0, 0.0]), &o).unwrap();
        let firsts = plans
            .iter()
            .map(|p| (p.candidate, p.cost))
            .collect::<Vec<_>>();
        assert_eq!(
            firsts,
            [(1, 0.0), (0, 1.0)],
            "{name}: the cheaper, later, slower first"
        );
        let alike = turn_in(&device, world.clone(), costing([2.0, 2.0]), &o).unwrap();
        let firsts = alike.iter().map(|p| p.candidate).collect::<Vec<_>>();
        assert_eq!(firsts, [0, 1], "{name}: alike in cost, the quicker first");
    }
}

#[test]
fn a_knob_turned_in_place_ends_with_the_hand_and_the_knob_turned_about_the_hands_axis() {
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let precision = o.tool_path.precision;
    let bar = knob(Vec2::new(0.5, 0.05), 0.3);
    let world = World::new(vec![table(), bar.clone()]);
    for device in common::devices(&robot) {
        let name = device.name();
        let grasps = vec![from_above(1.0), from_above(-1.0)];
        let task = on_table(&robot, grasps.clone(), FRAC_PI_2);
        let plans = turn_in(&device, world.clone(), task, &o).unwrap();
        assert_eq!(plans.len(), 2, "{name}: either way round");
        for plan in &plans {
            let at = bar.pose().mul_pose(grasps[plan.candidate]);
            let turn = check_joined(&robot, plan, at, &name);
            // In place: the hand stays where it grips, turning about its own z axis as far as
            // the turn has gone; it ends turned by the angle.
            let poses = hand_poses(&robot, &turn);
            for pose in &poses {
                assert!(
                    pose.position.distance(at.position) <= precision,
                    "{name}: {} m off",
                    pose.position.distance(at.position)
                );
                let about = (at.rotation.inverse() * pose.rotation).to_axis_angle();
                let off_axis = about.0.cross(Vec3::Z).length() * about.1.abs();
                assert!(off_axis < 2e-2, "{name}: turns about {about:?}");
            }
            let end = poses[poses.len() - 1];
            let turned = at.rotation * Quat::from_rotation_z(FRAC_PI_2);
            assert!(end.rotation.angle_between(turned) < 1e-2, "{name}");
            // The knob turns with it, about the hand's axis, which points down through its middle.
            let placed = plan
                .placed
                .clone()
                .expect("a turn leaves the object turned");
            let expected = Quat::from_rotation_z(-FRAC_PI_2) * bar.pose().rotation;
            assert!(
                placed.pose().rotation.angle_between(expected) < 1e-5,
                "{name}"
            );
            assert!(placed.pose().position.distance(bar.pose().position) < 1e-5);
            assert!(plan.held.is_none(), "{name}: the hand lets go");
            // The move away goes back up along the hand's z axis from where the turn ends.
            let away = robot.ee_pose(&last(&plan.depart), 0);
            let up = end.position - end.rotation * Vec3::Z * 0.1;
            assert!(away.position.distance(up) < 2e-3, "{name}");
        }
        // Quickest first, and each grasp alone moves alike.
        assert!(plans[0].duration() <= plans[1].duration(), "{name}");
        for plan in &plans {
            let alone = on_table(&robot, vec![grasps[plan.candidate]], FRAC_PI_2);
            let alone = turn_in(&device, world.clone(), alone, &o)
                .unwrap()
                .remove(0);
            let motions = |p: &TaskPlan| p.motions().cloned().collect::<Vec<_>>();
            assert_eq!(motions(&alone), motions(plan), "{name}");
        }
    }
}

/// A lever on a post, its hinge upright through the post: the post, an arm 15 cm long, 2 cm wide
/// and 1.5 cm thick resting on it, and the line it swings about.
fn lever() -> (Obstacle, Obstacle, Screw) {
    let hinge = Vec3::new(0.45, -0.2, 0.15);
    let post = Obstacle::cylinder(Vec3::new(hinge.x, hinge.y, 0.07), 0.015, 0.07);
    let half = Vec3::new(0.075, 0.01, 0.0075);
    let arm = Obstacle::cuboid(hinge + Vec3::new(0.06, 0.0, half.z), half);
    (post, arm, Screw::about(hinge, Vec3::Z))
}

/// The lever's arm gripped from above 10 cm from its hinge, closing across it.
fn lever_grasp() -> Pose {
    let half = Vec3::new(0.075, 0.01, 0.0075);
    grasp(-Vec3::Z, Vec3::Y, half, 0.01, Vec3::X * 0.04)
}

#[test]
fn a_lever_turned_about_its_hinge_follows_the_arc() {
    // A lever on a post, its hinge upright through the post: an arm 15 cm long and 2 cm wide,
    // gripped from above 10 cm from the hinge and swung 60 degrees.
    let robot = common::panda().unwrap();
    let o = TaskOptions {
        tool_speed: 0.03,
        ..TaskOptions::default()
    };
    let precision = o.tool_path.precision;
    let (post, arm, screw) = lever();
    let hinge = screw.point;
    let world = World::new(vec![table(), post, arm.clone()]);
    for device in common::devices(&robot) {
        let name = device.name();
        let grasps = vec![lever_grasp()];
        let at = arm.pose().mul_pose(grasps[0]);
        let radius = (at.position - hinge).truncate().length();
        assert!((radius - 0.1).abs() < 1e-6, "{radius}");
        let task = Turn {
            screw: Some(screw),
            support: vec![1],
            ..Turn::new(0, robot.default_q(), 2, grasps, FRAC_PI_3)
        };
        let plans = turn_in(&device, world.clone(), task, &o).unwrap();
        let turn = check_joined(&robot, &plans[0], at, &name);
        let poses = hand_poses(&robot, &turn);
        let mut farthest = 0.0f32;
        for pose in &poses {
            let from = pose.position - hinge;
            let off = (from.truncate().length() - radius).hypot(pose.position.z - at.position.z);
            // Half the precision for the waypoints' sag, and the follower holds each
            // configuration it steps to within a tenth of its own half: 0.5 + 0.05.
            assert!(off <= 0.55 * precision, "{name}: {off} m off the arc");
            let swung = (at.position - hinge).truncate().angle_to(from.truncate());
            farthest = farthest.max(swung);
        }
        assert!((farthest - FRAC_PI_3).abs() < 1e-2, "{name}: {farthest}");
        // Under the tool speed all along: 10.5 cm at 3 cm/s takes three and a half seconds, where
        // the joints alone take a second and a half.
        let gaps = poses
            .windows(2)
            .map(|w| w[0].position.distance(w[1].position));
        let fastest = gaps.fold(0.0f32, f32::max) / (turn.knot_interval / 64.0);
        assert!(fastest <= o.tool_speed * 1.02, "{name}: {fastest} m/s");
        assert!(turn.duration() >= 0.1 * FRAC_PI_3 / o.tool_speed, "{name}");
        let end = screw.carry(at, FRAC_PI_3);
        let ended = poses[poses.len() - 1];
        assert!(ended.position.distance(end.position) < 1e-3, "{name}");
        assert!(ended.rotation.angle_between(end.rotation) < 1e-2, "{name}");
        let placed = plans[0].placed.clone().unwrap();
        assert!(
            placed
                .pose()
                .position
                .distance(screw.carry(arm.pose(), FRAC_PI_3).position)
                < 1e-5
        );
    }
}

#[test]
fn a_cap_with_a_pitch_rises_as_it_turns() {
    // A cap 4 cm across on a bottle, its thread 5 mm a turn: unscrewed half a turn, anticlockwise
    // from above, it rises 2.5 mm.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let bottle = Obstacle::cylinder(Vec3::new(0.45, 0.2, 0.075), 0.035, 0.075);
    let cap = Obstacle::cylinder(Vec3::new(0.45, 0.2, 0.1625), 0.02, 0.0125);
    let world = World::new(vec![table(), bottle, cap.clone()]);
    let screw = Screw {
        point: Vec3::new(0.45, 0.2, 0.15),
        direction: Vec3::Z,
        pitch: 0.005 / TAU,
    };
    // From above across it, at eight turns about its axis: half a turn about the hand's axis is
    // more than the last joint has room for from some.
    let across = grasp(-Vec3::Z, Vec3::Y, Vec3::splat(0.0125), 0.008, Vec3::ZERO);
    let turns = (0..8).map(|k| {
        Pose::new(
            across.position,
            Quat::from_rotation_z(k as f32 * TAU / 8.0) * across.rotation,
        )
    });
    let grasps: Vec<Pose> = turns.collect();
    for device in common::devices(&robot) {
        let name = device.name();
        let task = Turn {
            screw: Some(screw),
            support: vec![1],
            ..Turn::new(0, robot.default_q(), 2, grasps.clone(), PI)
        };
        let plans = turn_in(&device, world.clone(), task, &o).unwrap();
        let at = cap.pose().mul_pose(grasps[plans[0].candidate]);
        let turn = check_joined(&robot, &plans[0], at, &name);
        // As far up as it has turned: the hand on the helix, the turn's share of 2.5 mm up. How
        // far it has turned is how far its level x axis has, between just under 0 and just over a
        // half turn.
        let heading = |r: Quat| {
            let x = r * Vec3::X;
            x.y.atan2(x.x)
        };
        for pose in hand_poses(&robot, &turn) {
            let angle = (heading(pose.rotation) - heading(at.rotation) + PI).rem_euclid(TAU) - PI;
            let angle = if angle < -FRAC_PI_2 {
                angle + TAU
            } else {
                angle
            };
            let risen = pose.position.z - at.position.z;
            let expected = screw.pitch * angle;
            assert!(
                (risen - expected).abs() <= o.tool_path.precision,
                "{name}: {risen} m up at {angle} rad"
            );
        }
        let ended = robot.ee_pose(&last(&turn), 0);
        let end = screw.carry(at, PI);
        assert!((end.position.z - at.position.z - 0.0025).abs() < 1e-6);
        assert!(ended.position.distance(end.position) < 2e-4, "{name}");
        let placed = plans[0].placed.clone().unwrap();
        let risen = placed.pose().position.z - cap.pose().position.z;
        assert!((risen - 0.0025).abs() < 1e-6, "{name}: {risen}");
    }
}

/// Every collision-free configuration IK finds at `at` in `worlds`' first world, nearest `start`
/// first, as `turn` takes them.
fn solutions(
    device: &Device,
    worlds: &batchplan::Worlds,
    start: &[f32],
    at: Pose,
) -> Vec<Vec<f32>> {
    let problem = IkProblem {
        world: 0,
        targets: vec![Some(at)],
        seed: Some(start.to_vec()),
        hold: vec![],
        holding: None,
    };
    let o = TaskOptions::default();
    let solved = batchplan::solve_ik(device, worlds, &[problem], &o.ik).unwrap();
    let distance = |q: &[f32]| -> f32 { q.iter().zip(start).map(|(a, b)| (a - b).powi(2)).sum() };
    let mut found: Vec<Vec<f32>> = (0..solved.seeds)
        .filter(|&i| solved.success[i])
        .map(|i| solved.solution(i).to_vec())
        .collect();
    found.sort_by(|a, b| distance(a).total_cmp(&distance(b)));
    found
}

#[test]
fn a_long_turn_the_nearest_configuration_cannot_make_is_made_by_another() {
    // 170 degrees about the hand's axis: from the configuration nearest the start, the last joint
    // runs into its limit a quarter of the way round; another configuration at the same grasp,
    // its wrist the other way over, has the room.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let angle = 170f32.to_radians();
    let bar = knob(Vec2::new(0.5, 0.05), 1.3);
    let mut world = World::new(vec![table(), bar.clone()]);
    let task = on_table(&robot, vec![from_above(1.0)], angle);
    let at = bar.pose().mul_pose(from_above(1.0));
    let start = robot.default_q().to_vec();
    let fingers: Vec<String> = robot
        .finger_links(0)
        .iter()
        .map(|l| l.to_string())
        .collect();
    // The knob held as the grasp holds it, in the world without it, touching the table.
    let held = Held {
        hand: 0,
        object: bar.placed(Pose::IDENTITY),
        grasp: from_above(1.0),
        locked: Default::default(),
        spheres: Default::default(),
    };
    let without = World {
        touch: vec![Touch::held(TABLE, 0)],
        ..World::new(vec![table()])
    };
    for device in common::devices(&robot) {
        let name = device.name();
        let mut touching = world.clone();
        touching
            .touch
            .push(Touch::new(KNOB, fingers.iter().map(String::as_str)));
        let worlds = device.upload(&[touching]).unwrap();
        let found = solutions(&device, &worlds, &start, at);
        assert!(found.len() > 1, "{name}: {} solutions", found.len());
        // Each configuration's turn, as `turn` follows it: holding the knob.
        let worlds = device.upload(std::slice::from_ref(&without)).unwrap();
        let along = ToolPathOptions {
            precision: o.tool_path.precision / 2.0,
            ..o.tool_path.clone()
        };
        let waypoints = Screw::about(at.position, at.rotation * Vec3::Z)
            .path(at, angle, along.precision)
            .unwrap();
        let paths: Vec<ToolPath> = found
            .iter()
            .map(|q| ToolPath {
                world: 0,
                start: q.clone(),
                hand: 0,
                waypoints: waypoints.clone(),
                hold: vec![],
                holding: Some(held.clone()),
            })
            .collect();
        let followed = batchplan::follow(&device, &worlds, &paths, &along).unwrap();
        // The nearest stops short, at a joint at its limit.
        assert!(
            matches!(followed.stopped[0], Some(ToolPathStop::Unreachable { .. })),
            "{name}: {:?}",
            followed.stopped[0]
        );
        assert!(
            followed.fraction[0] > 0.1,
            "{name}: {}",
            followed.fraction[0]
        );
        let n = robot.dof();
        let reached = &followed.paths[0][followed.paths[0].len() - n..];
        let at_limit = (0..n).any(|j| {
            let (lo, hi) = (robot.lower()[j], robot.upper()[j]);
            reached[j] - lo < 0.05 || hi - reached[j] < 0.05
        });
        assert!(at_limit, "{name}: stops at {reached:?}");
        // The turn is made from the first configuration, nearest first, that makes it whole.
        let whole = followed.fraction.iter().position(|&f| f == 1.0);
        let whole = whole.unwrap_or_else(|| panic!("{name}: no configuration turns it"));
        let plans = turn_in(&device, world.clone(), task.clone(), &o).unwrap();
        let turn = plans[0].turn.as_ref().unwrap();
        assert_eq!(first(turn), found[whole], "{name}");
        let ended = robot.ee_pose(&last(turn), 0);
        let end = at.rotation * Quat::from_rotation_z(angle);
        assert!(ended.rotation.angle_between(end) < 1e-2, "{name}");
    }
    // A post beside the knob, where the fingers come to two thirds of the way round: no
    // configuration gets past it.
    world.obstacles.push(post_at(&bar, 2.0 / 3.0 * angle));
    for device in common::devices(&robot) {
        let failure = turn_in(&device, world.clone(), task.clone(), &o).unwrap_err();
        assert_eq!(failure.step(), TaskStep::Turn, "{}", device.name());
    }
}

/// A post as tall as the knob `bar` standing beside it, where a finger of a grasp from above
/// across its middle comes to as the hand turns it by `angle` about the vertical (either way: the
/// fingers lie across the middle from each other).
fn post_at(bar: &Obstacle, angle: f32) -> Obstacle {
    let middle = bar.pose().position;
    let across = bar.pose().rotation * Vec3::Y;
    let at = Quat::from_rotation_z(-angle) * across * 0.06;
    Obstacle::cuboid(
        Vec3::new(middle.x + at.x, middle.y + at.y, BAR.z),
        Vec3::new(0.008, 0.008, BAR.z),
    )
}

#[test]
fn a_turn_into_an_obstacle_stops_at_the_turn_and_says_how_far_it_got() {
    // A post where the fingers come to halfway round a quarter turn.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let bar = knob(Vec2::new(0.5, 0.05), 0.3);
    let post = post_at(&bar, FRAC_PI_2 / 2.0);
    let world = World::new(vec![table(), bar, post]);
    let grasps = vec![from_above(1.0), from_above(-1.0)];
    for device in common::devices(&robot) {
        let name = device.name();
        let task = on_table(&robot, grasps.clone(), FRAC_PI_2);
        let failure = turn_in(&device, world.clone(), task, &o).unwrap_err();
        assert_eq!(failure.step(), TaskStep::Turn, "{name}: {failure}");
        assert!(
            failure.to_string().contains("2 turning (2 colliding)"),
            "{name}: {failure}"
        );
        for stop in &failure.candidates {
            assert_eq!(stop.step, TaskStep::Turn, "{name}");
            assert!(
                matches!(stop.why, Some(ToolPathStop::Collision { .. })),
                "{name}"
            );
            // It got some of the way, short of the post's side, which is a little under halfway.
            let turned = stop.turned.expect("a stop at the turn says how far it got");
            assert!(
                turned > 0.1 && turned < FRAC_PI_2 / 2.0,
                "{name}: {turned} rad"
            );
            let mut fingers = robot.finger_links(0);
            fingers.push("held by hand 0");
            let post_hit = stop.contacts.iter().any(|c| {
                matches!(c, Contact::World { obstacle: 2, link, .. } if fingers.contains(&link.as_str()))
            });
            assert!(post_hit, "{name}: {:?}", stop.contacts);
        }
    }
}

#[test]
fn a_batch_of_worlds_gives_each_the_result_it_gives_alone() {
    // 64 knobs at random places and turns, a quarter turn each, by either grasp from above, and
    // beside each a post where a finger comes to some way round, which stops the turns one way.
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    let mut rng = Rng::new(3);
    let mut worlds = vec![];
    let mut tasks = vec![];
    for w in 0..64 {
        let at = Vec2::new(rng.range(0.35, 0.65), rng.range(-0.35, 0.35));
        let bar = knob(at, rng.range(-PI, PI));
        let post = post_at(&bar, rng.range(0.3, 1.2));
        worlds.push(World::new(vec![table(), bar, post]));
        let grasps = vec![from_above(1.0), from_above(-1.0)];
        let angle = if rng.uniform() < 0.5 {
            FRAC_PI_2
        } else {
            -FRAC_PI_2
        };
        tasks.push(Turn {
            world: w,
            ..on_table(&robot, grasps, angle)
        });
    }
    let uploaded = device.upload(&worlds).unwrap();
    let all = batchplan::turn(&device, &uploaded, &tasks, &o).unwrap();
    let through = all.outcomes.iter().filter(|o| o.is_ok()).count();
    let turning = |f: &TaskFailure| f.at(TaskStep::Turn) > 0;
    let stopped = all
        .outcomes
        .iter()
        .filter(|o| o.as_ref().is_err_and(turning));
    let stopped = stopped.count();
    // 15 through and 49 stopped since a held bar takes its medial cover, which reaches the
    // tolerance past it nearly everywhere (17 and 47 with the cells it took before).
    assert!(
        through > 8 && stopped > 8,
        "{through} through, {stopped} stopped turning"
    );
    for (w, outcome) in all.outcomes.iter().enumerate() {
        let task = Turn {
            world: 0,
            ..tasks[w].clone()
        };
        let alone = turn_in(&device, worlds[w].clone(), task, &o);
        match (outcome, &alone) {
            (Ok(plans), Ok(lone)) => {
                let motions = |p: &[TaskPlan]| {
                    let each = p
                        .iter()
                        .map(|p| (p.candidate, p.motions().cloned().collect()));
                    each.collect::<Vec<(usize, Vec<Trajectory>)>>()
                };
                assert_eq!(motions(plans), motions(lone), "world {w}");
            }
            (Err(f), Err(lone)) => assert_eq!(f, lone, "world {w}"),
            _ => panic!("world {w}: {outcome:?} in the batch, {alone:?} alone"),
        }
    }
}

#[test]
fn malformed_turns_are_input_errors() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let grid = SdfGrid::try_new([4, 4, 4], 0.01, Vec3::new(0.3, 0.3, 0.0), &[0.01; 64]).unwrap();
    let world = World::new(vec![table(), knob(Vec2::new(0.5, 0.0), 0.0), grid.into()]);
    let worlds = device.upload(&[world]).unwrap();
    let good = on_table(&robot, vec![from_above(1.0)], FRAC_PI_2);
    let cases = [
        (
            Turn {
                angle: 0.0,
                ..good.clone()
            },
            "other than 0, got 0 rad",
        ),
        (
            Turn {
                angle: f32::NAN,
                ..good.clone()
            },
            "got NaN rad",
        ),
        (
            Turn {
                screw: Some(Screw::about(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.5))),
                ..good.clone()
            },
            "task 0: a screw's direction must be unit, got [0, 0, 0.5]",
        ),
        (
            Turn {
                screw: Some(Screw {
                    pitch: f32::NAN,
                    ..Screw::about(Vec3::ZERO, Vec3::Z)
                }),
                ..good.clone()
            },
            "a screw must be finite",
        ),
        (
            Turn {
                object: 2,
                ..good.clone()
            },
            "obstacle 2 is a distance grid, which cannot be turned",
        ),
        (
            Turn {
                object: 3,
                ..good.clone()
            },
            "obstacle 3 of 3",
        ),
        (
            Turn {
                approach: 0.0,
                ..good.clone()
            },
            "positive approach, got 0 m",
        ),
        (
            Turn {
                grasps: vec![],
                ..good.clone()
            },
            "has no grasps",
        ),
        (
            Turn {
                enough: Some(0),
                ..good.clone()
            },
            "at least one plan",
        ),
        (
            Turn {
                support: vec![KNOB],
                ..good.clone()
            },
            "support 1 must be another of the world's 3 obstacles",
        ),
        (
            Turn {
                screw: Some(Screw::about(Vec3::ZERO, Vec3::ZERO)),
                ..good.clone()
            },
            "task 0: a screw's direction must be unit, got [0, 0, 0]",
        ),
        (
            Turn {
                angle: 1e13,
                ..good.clone()
            },
            "task 0, grasp 0: a turn of 10000000000000 rad at a precision of 0.0005 m takes",
        ),
    ];
    for (bad, says) in cases {
        let err = batchplan::turn(&device, &worlds, &[bad], &TaskOptions::default()).unwrap_err();
        assert!(
            matches!(&err, Error::Input(m) if m.contains(says)),
            "{err:?}"
        );
    }
    // A precision so fine that a turn 4 cm about a line takes more waypoints than can be held.
    let mut fine = TaskOptions::default();
    fine.tool_path.precision = 1e-30;
    let aside = Turn {
        screw: Some(Screw::about(Vec3::new(0.54, 0.0, 0.0), Vec3::Z)),
        ..good
    };
    let err = batchplan::turn(&device, &worlds, &[aside], &fine).unwrap_err();
    let says = |m: &str| {
        m.starts_with("task 0, grasp 0: a turn of 1.5707964 rad at a precision of")
            && m.ends_with("waypoints, more than can be held")
    };
    assert!(matches!(&err, Error::Input(m) if says(m)), "{err:?}");
}

#[test]
fn what_a_turned_lever_sweeps_is_checked_against_all_but_its_support() {
    // The lever swung 60 degrees past a post 8 mm square reaching 1 cm over the arm's underside,
    // 12.8 cm out from the hinge at 30 degrees: halfway round, the arm is in the post, where the
    // hand and the arm of the robot are not.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let (mount, arm, screw) = lever();
    let hinge = screw.point;
    let out = Vec2::from_angle(30f32.to_radians()) * 0.128;
    let at = Vec3::new(hinge.x + out.x, hinge.y + out.y, 0.08);
    let post = Obstacle::cuboid(at, Vec3::new(0.004, 0.004, 0.08));
    let world = World::new(vec![table(), mount, arm, post]);
    let task = Turn {
        screw: Some(screw),
        support: vec![1],
        ..Turn::new(0, robot.default_q(), 2, vec![lever_grasp()], FRAC_PI_3)
    };
    for device in common::devices(&robot) {
        let name = device.name();
        let failure = turn_in(&device, world.clone(), task.clone(), &o).unwrap_err();
        assert_eq!(failure.step(), TaskStep::Turn, "{name}: {failure}");
        let stop = &failure.candidates[0];
        let turned = stop.turned.expect("a stop at the turn says how far it got");
        assert!(
            turned > 0.1 && turned < FRAC_PI_3 / 2.0,
            "{name}: {turned} rad"
        );
        let met = stop.contacts.iter().any(
            |c| matches!(c, Contact::World { link, obstacle: 3, .. } if link == "held by hand 0"),
        );
        assert!(met, "{name}: {:?}", stop.contacts);
        // Named as what the lever may touch, the post lets it by.
        let past = Turn {
            support: vec![1, 3],
            ..task.clone()
        };
        assert!(turn_in(&device, world.clone(), past, &o).is_ok(), "{name}");
    }
}

#[test]
fn a_cap_turns_on_its_bottle_only_where_the_bottle_is_its_support() {
    // The cap sits on the bottle all the way round as it turns: what it turns on must be named.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let bottle = Obstacle::cylinder(Vec3::new(0.45, 0.2, 0.075), 0.035, 0.075);
    let cap = Obstacle::cylinder(Vec3::new(0.45, 0.2, 0.1625), 0.02, 0.0125);
    let world = World::new(vec![table(), bottle, cap]);
    let across = grasp(-Vec3::Z, Vec3::Y, Vec3::splat(0.0125), 0.008, Vec3::ZERO);
    let turns = (0..8).map(|k| {
        let turn = Quat::from_rotation_z(k as f32 * TAU / 8.0);
        Pose::new(across.position, turn * across.rotation)
    });
    let task = Turn {
        screw: Some(Screw::about(Vec3::new(0.45, 0.2, 0.15), Vec3::Z)),
        ..Turn::new(0, robot.default_q(), 2, turns, FRAC_PI_2)
    };
    for device in common::devices(&robot) {
        let name = device.name();
        let failure = turn_in(&device, world.clone(), task.clone(), &o).unwrap_err();
        assert_eq!(failure.step(), TaskStep::Turn, "{name}: {failure}");
        for stop in failure
            .candidates
            .iter()
            .filter(|c| c.step == TaskStep::Turn)
        {
            let on_bottle = stop.contacts.iter().any(|c| {
                matches!(c, Contact::World { link, obstacle: 1, .. } if link == "held by hand 0")
            });
            assert!(on_bottle, "{name}: {:?}", stop.contacts);
        }
        let on = Turn {
            support: vec![1],
            ..task.clone()
        };
        assert!(turn_in(&device, world.clone(), on, &o).is_ok(), "{name}");
    }
}

#[test]
fn an_obstacle_that_stays_is_turned_in_by_the_fingers_and_its_sweep_is_not_checked() {
    // A bar 30 cm tall standing on the table, gripped from above at its top and turned a quarter
    // turn in place, and a post beside its foot where its end comes halfway round. Turned whole,
    // the bar meets the post; staying, as a knob a camera could not tell from its panel, the
    // fingers turn at its top and nothing checks the bar's sweep.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let half = Vec3::new(0.045, 0.0125, 0.15);
    let yaw = 0.3;
    let middle = Vec3::new(0.5, 0.05, half.z);
    let tall =
        Obstacle::cuboid(Vec3::ZERO, half).placed(Pose::new(middle, Quat::from_rotation_z(yaw)));
    // The hand's z axis points down: a quarter turn about it is a quarter turn back about up.
    let halfway = Vec2::from_angle(yaw - FRAC_PI_2 / 2.0) * half.x;
    let post = Obstacle::cuboid(
        Vec3::new(middle.x + halfway.x, middle.y + halfway.y, 0.025),
        Vec3::new(0.005, 0.005, 0.025),
    );
    let world = World::new(vec![table(), tall.clone(), post]);
    let task = Turn {
        support: vec![TABLE],
        ..Turn::new(
            0,
            robot.default_q(),
            1,
            vec![grasp(-Vec3::Z, Vec3::Y, half, 0.01, Vec3::ZERO)],
            FRAC_PI_2,
        )
    };
    for device in common::devices(&robot) {
        let name = device.name();
        let failure = turn_in(&device, world.clone(), task.clone(), &o).unwrap_err();
        assert_eq!(failure.step(), TaskStep::Turn, "{name}: {failure}");
        let met = failure.candidates[0].contacts.iter().any(
            |c| matches!(c, Contact::World { link, obstacle: 2, .. } if link == "held by hand 0"),
        );
        assert!(met, "{name}: {:?}", failure.candidates[0].contacts);
        let staying = Turn {
            stays: true,
            ..task.clone()
        };
        let plans = turn_in(&device, world.clone(), staying, &o).unwrap();
        assert_eq!(
            plans[0].placed,
            Some(tall.clone()),
            "{name}: it stands where it stood"
        );
    }
}

#[test]
fn a_turn_stops_at_as_many_plans_as_will_do_in_its_grasps_order() {
    // 36 grasps 2 m off, out of reach, then the knob's two from above: one plan will do, and it
    // is the first of the two, found among the second lot of grasps tried.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let world = World::new(vec![table(), knob(Vec2::new(0.5, 0.05), 0.3)]);
    let far = Pose::new(Vec3::new(2.0, 0.0, 0.0), from_above(1.0).rotation);
    let mut grasps = vec![far; 36];
    grasps.extend([from_above(1.0), from_above(-1.0)]);
    for device in common::devices(&robot) {
        let name = device.name();
        let every = turn_in(
            &device,
            world.clone(),
            on_table(&robot, grasps.clone(), FRAC_PI_2),
            &o,
        );
        let every = every.unwrap();
        let mut got: Vec<usize> = every.iter().map(|p| p.candidate).collect();
        got.sort_unstable();
        assert_eq!(got, vec![36, 37], "{name}");
        let one = Turn {
            enough: Some(1),
            ..on_table(&robot, grasps.clone(), FRAC_PI_2)
        };
        let one = turn_in(&device, world.clone(), one, &o).unwrap();
        assert_eq!(one.len(), 1, "{name}");
        assert_eq!(one[0].candidate, 36, "{name}");
        let same = every
            .iter()
            .find(|p| p.candidate == 36)
            .expect("grasp 36 gets through");
        let motions = |p: &TaskPlan| p.motions().cloned().collect::<Vec<_>>();
        assert_eq!(motions(&one[0]), motions(same), "{name}");
    }
}

#[test]
fn a_turn_keeps_the_fingers_as_narrow_as_it_says() {
    // A round knob 2.5 cm across and 3 cm tall, and a post 5 cm from its axis where an open
    // finger comes down: gripped with the fingers open, the grasp is in the post; with them 3 cm
    // apart it turns, the fingers passing inside the post.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let round = Obstacle::cylinder(Vec3::new(0.5, 0.05, 0.015), 0.0125, 0.015);
    let post = Obstacle::cuboid(Vec3::new(0.5, 0.1, 0.02), Vec3::new(0.004, 0.004, 0.02));
    let world = World::new(vec![table(), round, post]);
    let open = on_table(
        &robot,
        vec![grasp(
            -Vec3::Z,
            Vec3::Y,
            Vec3::splat(0.015),
            0.01,
            Vec3::ZERO,
        )],
        FRAC_PI_2,
    );
    let narrow: HashMap<String, f32> = [
        ("panda_finger_joint1", 0.015),
        ("panda_finger_joint2", 0.015),
    ]
    .into_iter()
    .map(|(j, v)| (j.to_string(), v))
    .collect();
    for device in common::devices(&robot) {
        let name = device.name();
        let failure = turn_in(&device, world.clone(), open.clone(), &o).unwrap_err();
        assert_eq!(failure.step(), TaskStep::Reach, "{name}: {failure}");
        let locked = Turn {
            grasps: vec![batchplan::Hold {
                locked: narrow.clone(),
                ..open.grasps[0].clone()
            }],
            ..open.clone()
        };
        assert!(
            turn_in(&device, world.clone(), locked, &o).is_ok(),
            "{name}"
        );
    }
}

#[test]
fn each_grasp_turns_the_object_about_its_own_line() {
    // The knob gripped from above 2 cm either side of its middle and turned a quarter turn in
    // place: each grasp turns it about its own z axis, and each plan leaves it where its turn does.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let bar = knob(Vec2::new(0.5, 0.05), 0.3);
    let world = World::new(vec![table(), bar.clone()]);
    let grasps = vec![
        grasp(-Vec3::Z, Vec3::Y, BAR, 0.01, Vec3::X * 0.02),
        grasp(-Vec3::Z, Vec3::Y, BAR, 0.01, Vec3::X * -0.02),
    ];
    for device in common::devices(&robot) {
        let name = device.name();
        let plans = turn_in(
            &device,
            world.clone(),
            on_table(&robot, grasps.clone(), FRAC_PI_2),
            &o,
        );
        let plans = plans.unwrap();
        assert_eq!(plans.len(), 2, "{name}");
        let mut middles = vec![];
        for plan in &plans {
            let at = bar.pose().mul_pose(grasps[plan.candidate]);
            let about = Screw::about(at.position, at.rotation * Vec3::Z);
            let expected = about.carry(bar.pose(), FRAC_PI_2);
            let placed = plan
                .placed
                .clone()
                .expect("a turn leaves the object turned")
                .pose();
            assert!(placed.position.distance(expected.position) < 1e-5, "{name}");
            assert!(
                placed.rotation.angle_between(expected.rotation) < 1e-5,
                "{name}"
            );
            middles.push(placed.position);
        }
        assert!(
            middles[0].distance(middles[1]) > 0.03,
            "{name}: {middles:?}"
        );
    }
}

#[test]
fn the_move_away_leaves_the_object_where_the_turn_left_it() {
    // A door 25 cm wide, 40 cm tall and 1 cm thick standing on the table, hinged upright at
    // (0.45, 0.25) and reaching along -x, gripped at its free edge and swung 60 degrees. The hand
    // retraces its move in, in the door's frame; the arm comes back where the door stood before
    // the turn, so a move away checked against the door unturned would meet it.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let hinge = Vec3::new(0.45, 0.25, 0.0);
    let door = Obstacle::cuboid(
        Vec3::new(hinge.x - 0.125, hinge.y, 0.2),
        Vec3::new(0.125, 0.005, 0.2),
    );
    let world = World::new(vec![table(), door]);
    let edge = Pose::new(
        Vec3::new(-0.105, 0.0, 0.0),
        Quat::from_xyzw(0.0, FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2),
    );
    let task = Turn {
        screw: Some(Screw::about(hinge, Vec3::Z)),
        support: vec![TABLE],
        ..Turn::new(0, robot.default_q(), 1, vec![edge], -FRAC_PI_3)
    };
    let mut stood = world.clone();
    stood.touch.push(Touch::new(1, robot.finger_links(0)));
    for device in common::devices(&robot) {
        let name = device.name();
        let plans = turn_in(&device, world.clone(), task.clone(), &o).unwrap();
        let samples = plans[0].depart.sample(250.0).unwrap();
        let worlds = device.upload(std::slice::from_ref(&stood)).unwrap();
        let items = vec![0; samples.len()];
        let contacts = device
            .contacts(&worlds, &items, &samples.positions, 0.0)
            .unwrap();
        let met = contacts.iter().flatten().any(
            |c| matches!(c, Contact::World { link, obstacle: 1, .. } if link == "panda_link4"),
        );
        assert!(
            met,
            "{name}: the move away meets the door where it stood only at the arm"
        );
    }
}

#[test]
fn a_grasp_a_little_off_unit_is_turned_in_place() {
    // A grasp's rotation scaled by 1.0008, within the tolerance every task allows: the hand's own
    // axis there is a line of unit direction, and the knob turns.
    let robot = common::panda().unwrap();
    let o = TaskOptions::default();
    let world = World::new(vec![table(), knob(Vec2::new(0.5, 0.05), 0.3)]);
    let grip = from_above(1.0);
    let scaled = Pose::new(grip.position, grip.rotation * 1.0008);
    let device = Device::cpu(&robot).unwrap();
    assert!(
        turn_in(
            &device,
            world,
            on_table(&robot, vec![scaled], FRAC_PI_2),
            &o
        )
        .is_ok()
    );
}

/// A horizontal rail and a handle whose frame is turned relative to it. Each task pulls from
/// above with open fingers, through a different world in the same batch.
#[test]
fn drawer_draws_are_signed_world_translations_and_stops_report_meters() {
    let robot = common::panda().unwrap();
    let handle = knob(Vec2::new(0.50, 0.05), 0.42);
    let clear = World::new(vec![table(), handle.clone()]);
    let direction = Vec3::NEG_X;
    let grasp = from_above(1.0);
    let at = handle.pose().mul_pose(grasp);
    // The stop meets the hand on its way in the positive x direction. A negative distance
    // along the world's -x axis goes that way even though the handle's frame is turned.
    let mut blocked = clear.clone();
    let wall = Obstacle::cuboid(Vec3::new(0.62, 0.05, 0.08), Vec3::new(0.008, 0.10, 0.08));
    blocked.obstacles.push(wall);
    let tasks = [0.08, -0.08, -0.18].map(|distance| {
        let world = u32::from(distance < -0.1);
        batchplan::Pull::new(
            world,
            robot.default_q(),
            KNOB,
            vec![grasp],
            direction,
            distance,
        )
    });
    let o = TaskOptions::default();
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[clear.clone(), blocked.clone()]).unwrap();
        let result = batchplan::pull(&device, &worlds, &tasks, &o).unwrap();
        for (t, outcome) in result.problems.iter().zip(&result.outcomes[..2]) {
            let plans = outcome
                .as_ref()
                .unwrap_or_else(|why| panic!("{name}: {why}"));
            let plan = &plans[0];
            let draw = plan.pull.as_ref().expect("a pull plan has a draw");
            assert_eq!(plan.motions().count(), 4, "{name}: includes draw");
            assert!(plan.turn.is_none() && plan.placed.is_none() && plan.held.is_none());
            assert_eq!(last(&plan.approach), first(draw), "{name}");
            assert_eq!(last(draw), first(&plan.depart), "{name}");
            for motion in plan.motions() {
                motion.check(&robot).unwrap();
            }
            let poses = hand_poses(&robot, draw);
            let end = at.position + direction * t.distance;
            assert!(
                poses.last().unwrap().position.distance(end) < o.tool_path.precision,
                "{name}"
            );
            for pose in poses {
                let off = (pose.position - at.position).reject_from_normalized(direction);
                assert!(off.length() <= o.tool_path.precision, "{name}: {off}");
                assert!(
                    pose.rotation.angle_between(at.rotation) < o.tool_path.rotation_precision,
                    "{name}"
                );
            }
            let away = robot.ee_pose(&last(&plan.depart), 0);
            assert!(
                away.position
                    .distance(end - at.rotation * Vec3::Z * t.approach)
                    < 2e-3,
                "{name}"
            );
        }
        let failure = result.outcomes[2]
            .as_ref()
            .expect_err("the wall stops the hand");
        assert_eq!(failure.step(), TaskStep::Pull, "{name}: {failure}");
        let stop = &failure.candidates[0];
        let drawn = stop.pulled.expect("a stopped draw reports a distance");
        assert!((-0.18..-0.01).contains(&drawn), "{name}: {drawn}");
        assert!(stop.turned.is_none(), "{name}: metres belong to a pull");
        assert!(
            stop.contacts
                .iter()
                .any(|c| matches!(c, Contact::World { obstacle: 2, .. })),
            "{name}: {:?}",
            stop.contacts
        );
        let alone = batchplan::pull(&device, &worlds, &tasks[2..], &o).unwrap();
        assert_eq!(alone.outcomes[0].as_ref().unwrap_err(), failure, "{name}");
    }
}

#[test]
fn a_placed_grid_is_a_contact_target_for_a_fixed_fixture() {
    let robot = common::panda().unwrap();
    // One occupied grid cell at the center of a handle with a rotated local frame.
    let pose = Pose::new(Vec3::new(0.51, 0.07, 0.14), Quat::from_rotation_z(0.55));
    let cell = SdfGrid::try_new([3; 3], 0.01, -Vec3::splat(0.01), &[0.0; 27]).unwrap();
    let supplied = Pose::new(pose.position, pose.rotation * 1.0008);
    let grid = Obstacle::from_grid(cell).placed(supplied);
    let grasp = from_above(1.0);
    let supplied_grasp = Pose::new(grasp.position, grasp.rotation * 1.0008);
    assert!(supplied.is_unit() && supplied_grasp.is_unit());
    assert!(!supplied.mul_pose(supplied_grasp).is_unit());
    let world = World::new(vec![table(), grid]);
    let axis = Vec3::NEG_Y;
    let o = TaskOptions::default();
    let pull = batchplan::Pull::new(0, robot.default_q(), 1, vec![supplied_grasp], axis, 0.05);
    let turn = Turn {
        stays: true,
        ..Turn::new(0, robot.default_q(), 1, vec![supplied_grasp], 0.3)
    };
    let at = pose.mul_pose(grasp);
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[world]).unwrap();
    let pulled = batchplan::pull(&device, &worlds, &[pull], &o).unwrap();
    let drawn = pulled.outcomes[0].as_ref().unwrap()[0]
        .pull
        .as_ref()
        .unwrap();
    let end = robot.ee_pose(&last(drawn), 0);
    assert!(end.position.distance(at.position + axis * 0.05) < o.tool_path.precision);
    let turned = batchplan::turn(&device, &worlds, &[turn], &o).unwrap();
    let plan = &turned.outcomes[0].as_ref().unwrap()[0];
    assert_eq!(plan.placed.as_ref().unwrap().pose(), supplied);
    let end = robot.ee_pose(&last(plan.turn.as_ref().unwrap()), 0);
    assert!(end.position.distance(at.position) < o.tool_path.precision);
    assert!(
        end.rotation
            .angle_between(at.rotation * Quat::from_rotation_z(0.3))
            < o.tool_path.rotation_precision
    );
}

#[test]
fn pulls_reject_bad_directions_and_distances_before_searching() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::new(vec![table()])]).unwrap();
    for (direction, distance, message) in [
        (Vec3::ZERO, 0.1, "direction"),
        (Vec3::Y * 2.0, 0.1, "direction"),
        (Vec3::NAN, 0.1, "direction"),
        (Vec3::Z, 0.0, "distance"),
        (Vec3::Z, f32::NAN, "distance"),
        (Vec3::Z, f32::MAX, "poses"),
    ] {
        let task = batchplan::Pull::new(
            0,
            robot.default_q(),
            0,
            vec![Pose::IDENTITY],
            direction,
            distance,
        );
        let error =
            batchplan::pull(&device, &worlds, &[task], &TaskOptions::default()).unwrap_err();
        assert!(
            matches!(error, Error::Input(_)) && error.to_string().contains(message),
            "{error}"
        );
    }
    // The step count alone fits, but the tool path cannot measure the displacement in f32.
    let options = TaskOptions {
        tool_path: batchplan::ToolPathOptions {
            step: 1e24,
            ..Default::default()
        },
        ..Default::default()
    };
    let too_far =
        batchplan::Pull::new(0, robot.default_q(), 0, vec![Pose::IDENTITY], Vec3::Y, 1e24);
    let error = batchplan::pull(&device, &worlds, &[too_far], &options).unwrap_err();
    assert!(matches!(error, Error::Input(_)), "{error}");
}

#[test]
fn a_hand_without_finger_links_cannot_touch_the_contact_grid() {
    let reference = common::panda().unwrap();
    let robot = Robot::load(
        common::asset("franka/franka_panda.urdf"),
        &batchplan::RobotOptions {
            finger_links: [("ee_link".into(), vec![])].into(),
            ..common::panda_options()
        },
    )
    .unwrap();
    assert!(robot.finger_links(0).is_empty() && !robot.hand_links(0).is_empty());
    let at = reference.ee_pose(reference.default_q(), 0);
    let finger = reference
        .hand_spheres(0)
        .into_iter()
        .find(|s| s.finger)
        .unwrap();
    let center = at.position + at.rotation * finger.center;
    let grid = SdfGrid::try_new([3; 3], 0.01, -Vec3::splat(0.01), &[0.0; 27]).unwrap();
    let target = Obstacle::from_grid(grid).placed(Pose::new(center, Quat::IDENTITY));
    let grasp = target.pose().inverse().mul_pose(at);
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::new(vec![target])]).unwrap();
    let o = TaskOptions::default();
    let pull = batchplan::Pull {
        transit: false,
        ..batchplan::Pull::new(0, robot.default_q(), 0, vec![grasp], Vec3::X, 0.03)
    };
    let turn = Turn {
        transit: false,
        stays: true,
        ..Turn::new(0, robot.default_q(), 0, vec![grasp], 0.2)
    };
    let pulls = batchplan::pull(&device, &worlds, &[pull], &o).unwrap();
    let turns = batchplan::turn(&device, &worlds, &[turn], &o).unwrap();
    for outcome in [pulls.outcomes, turns.outcomes] {
        let failure = outcome[0]
            .as_ref()
            .expect_err("no finger has permission to touch the target");
        assert_eq!(failure.step(), TaskStep::Reach);
        assert!(
            failure.candidates[0]
                .contacts
                .iter()
                .any(|c| matches!(c, Contact::World { obstacle: 0, .. }))
        );
    }
}

#[test]
fn a_draw_without_a_transit_checks_which_device_prepared_its_worlds() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let other = Device::cpu(&robot).unwrap();
    let world = World::new(vec![table(), knob(Vec2::new(0.5, 0.05), 0.3)]);
    let worlds = other.upload(&[world]).unwrap();
    let task = batchplan::Pull {
        transit: false,
        ..batchplan::Pull::new(
            0,
            robot.default_q(),
            KNOB,
            vec![from_above(1.0)],
            Vec3::Y,
            0.05,
        )
    };
    let error = batchplan::pull(&device, &worlds, &[task], &TaskOptions::default()).unwrap_err();
    assert!(
        matches!(error, Error::Input(_)) && error.to_string().contains("different device"),
        "{error}"
    );
}
