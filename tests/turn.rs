//! Turning a thing about a line in one call (`turn`): a knob in place, a lever about its hinge, a
//! cap along its thread; a turn the configuration nearest the start cannot make is made by
//! another; one that meets an obstacle says how far it got; a batch gives each task what it gives
//! alone.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, PI, TAU};

use batchplan::rng::Rng;
use batchplan::{
    Contact, Device, Error, IkProblem, Obstacle, Pose, Robot, Screw, SdfGrid, TaskFailure,
    TaskOptions, TaskPlan, TaskStep, ToolPath, ToolPathOptions, ToolPathStop, Touch, Trajectory,
    Turn, World,
};
use glam::{Mat3, Quat, Vec2, Vec3};

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

#[test]
fn a_knob_turned_in_place_ends_with_the_hand_and_the_knob_turned_about_the_hands_axis() {
    let robot = panda();
    let o = TaskOptions::default();
    let precision = o.tool_path.precision;
    let bar = knob(Vec2::new(0.5, 0.05), 0.3);
    let world = World::new(vec![table(), bar.clone()]);
    for device in devices(&robot) {
        let name = device.name();
        let grasps = vec![from_above(1.0), from_above(-1.0)];
        let task = Turn::new(0, robot.default_q(), KNOB, grasps.clone(), FRAC_PI_2);
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
            let alone = Turn::new(
                0,
                robot.default_q(),
                KNOB,
                vec![grasps[plan.candidate]],
                FRAC_PI_2,
            );
            let alone = turn_in(&device, world.clone(), alone, &o)
                .unwrap()
                .remove(0);
            let motions = |p: &TaskPlan| p.motions().cloned().collect::<Vec<_>>();
            assert_eq!(motions(&alone), motions(plan), "{name}");
        }
    }
}

#[test]
fn a_lever_turned_about_its_hinge_follows_the_arc() {
    // A lever on a post, its hinge upright through the post: an arm 15 cm long and 2 cm wide,
    // gripped from above 10 cm from the hinge and swung 60 degrees.
    let robot = panda();
    let o = TaskOptions {
        tool_speed: 0.03,
        ..TaskOptions::default()
    };
    let precision = o.tool_path.precision;
    let hinge = Vec3::new(0.45, -0.2, 0.15);
    let post = Obstacle::cylinder(Vec3::new(hinge.x, hinge.y, 0.07), 0.015, 0.07);
    let half = Vec3::new(0.075, 0.01, 0.0075);
    let arm = Obstacle::cuboid(hinge + Vec3::new(0.06, 0.0, half.z), half);
    let world = World::new(vec![table(), post, arm.clone()]);
    let screw = Screw::about(hinge, Vec3::Z);
    for device in devices(&robot) {
        let name = device.name();
        let toward_end = Vec3::X * 0.04;
        let grasps = vec![grasp(-Vec3::Z, Vec3::Y, half, 0.01, toward_end)];
        let at = arm.pose().mul_pose(grasps[0]);
        let radius = (at.position - hinge).truncate().length();
        assert!((radius - 0.1).abs() < 1e-6, "{radius}");
        let task = Turn {
            screw: Some(screw),
            ..Turn::new(0, robot.default_q(), 2, grasps, FRAC_PI_3)
        };
        let plans = turn_in(&device, world.clone(), task, &o).unwrap();
        let turn = check_joined(&robot, &plans[0], at, &name);
        let poses = hand_poses(&robot, &turn);
        let mut farthest = 0.0f32;
        for pose in &poses {
            let from = pose.position - hinge;
            let off = (from.truncate().length() - radius).hypot(pose.position.z - at.position.z);
            assert!(off <= precision, "{name}: {off} m off the arc");
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
    let robot = panda();
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
    for device in devices(&robot) {
        let name = device.name();
        let task = Turn {
            screw: Some(screw),
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
    let robot = panda();
    let o = TaskOptions::default();
    let angle = 170f32.to_radians();
    let bar = knob(Vec2::new(0.5, 0.05), 1.3);
    let mut world = World::new(vec![table(), bar.clone()]);
    let task = Turn::new(0, robot.default_q(), KNOB, vec![from_above(1.0)], angle);
    let at = bar.pose().mul_pose(from_above(1.0));
    let start = robot.default_q().to_vec();
    let fingers: Vec<String> = robot
        .finger_links(0)
        .iter()
        .map(|l| l.to_string())
        .collect();
    for device in devices(&robot) {
        let name = device.name();
        let mut touching = world.clone();
        touching
            .touch
            .push(Touch::new(KNOB, fingers.iter().map(String::as_str)));
        let worlds = device.upload(&[touching]).unwrap();
        let found = solutions(&device, &worlds, &start, at);
        assert!(found.len() > 1, "{name}: {} solutions", found.len());
        // Each configuration's turn, as `turn` follows it.
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
                holding: None,
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
    for device in devices(&robot) {
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
    let robot = panda();
    let o = TaskOptions::default();
    let bar = knob(Vec2::new(0.5, 0.05), 0.3);
    let post = post_at(&bar, FRAC_PI_2 / 2.0);
    let world = World::new(vec![table(), bar, post]);
    let grasps = vec![from_above(1.0), from_above(-1.0)];
    for device in devices(&robot) {
        let name = device.name();
        let task = Turn::new(0, robot.default_q(), KNOB, grasps.clone(), FRAC_PI_2);
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
            let fingers = robot.finger_links(0);
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
    let robot = panda();
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
        tasks.push(Turn::new(w, robot.default_q(), KNOB, grasps, angle));
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
    assert!(
        through > 16 && stopped > 8,
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
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let grid = SdfGrid::try_new([4, 4, 4], 0.01, Vec3::new(0.3, 0.3, 0.0), &[0.01; 64]).unwrap();
    let world = World::new(vec![table(), knob(Vec2::new(0.5, 0.0), 0.0), grid.into()]);
    let worlds = device.upload(&[world]).unwrap();
    let good = Turn::new(0, robot.default_q(), KNOB, vec![from_above(1.0)], FRAC_PI_2);
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
                ..good
            },
            "at least one plan",
        ),
    ];
    for (bad, says) in cases {
        let err = batchplan::turn(&device, &worlds, &[bad], &TaskOptions::default()).unwrap_err();
        assert!(
            matches!(&err, Error::Input(m) if m.contains(says)),
            "{err:?}"
        );
    }
}
