//! One object held by two hands (`Held::also`): every call that holds it keeps both at their
//! grasps, the object rigid between them, where each hand's fingers may touch it. GPU parts are
//! skipped when no adapter is available unless `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;
use std::path::PathBuf;

use batchplan::live::{Follower, FollowerOptions};
use batchplan::{
    Contact, Device, Error, Goal, Grip, GripperCalibration, HandGrasps, Handover, Held, Hold,
    IkProblem, JoinOptions, Obstacle, Pick, Place, PlanOptions, PlanProblem, Pose, Release, Robot,
    Sequence, ShapeGraspOptions, TaskFailure, TaskOptions, TaskPlan, TaskStep, ToolPath,
    ToolPathOptions, ToolPathStop, Trajectory, World, Worlds,
};
use glam::{Quat, Vec3};

/// A bar 32 cm long, 3 cm thick, held from above by the left hand 12 cm to one side of its
/// middle and by the right hand 12 cm to the other, each closing across it, turned half a turn
/// from the other.
fn bar() -> Held {
    let grasp = |side: f32| {
        Pose::new(
            Vec3::new(0.0, 0.12 * side, 0.0),
            common::upright(FRAC_PI_2 * side),
        )
    };
    Held {
        carries: vec![],
        touch_links: vec![],
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.015, 0.16, 0.015)),
        grasp: grasp(1.0),
        also: vec![Grip {
            hand: 1,
            grasp: grasp(-1.0),
        }],
        locked: Default::default(),
        spheres: Default::default(),
    }
}

/// The left hand's pose where the bar lies level with its middle at `at`, turned `yaw` about the
/// vertical.
fn left_at(at: Vec3, yaw: f32) -> Pose {
    common::tray(at, yaw).mul_pose(bar().grasp)
}

/// How far the right hand is from its grasp of the bar the left hand holds: meters, radians.
fn off_its_grasp(robot: &Robot, q: &[f32]) -> (f32, f32) {
    off_grasp(robot, &bar(), q)
}

/// How far `held`'s other hand is from its grasp of what its hand holds: meters, radians.
fn off_grasp(robot: &Robot, held: &Held, q: &[f32]) -> (f32, f32) {
    let there = robot
        .ee_pose(q, held.hand)
        .mul_pose(held.grasp.inverse())
        .mul_pose(held.also[0].grasp);
    let other = robot.ee_pose(q, held.also[0].hand);
    (
        other.position.distance(there.position),
        other.rotation.angle_between(there.rotation),
    )
}

/// The farthest the right hand strays from its grasp over a path, sampled at 1 kHz once timed.
fn strays(robot: &Robot, path: &[f32]) -> (f32, f32) {
    strays_from(robot, &bar(), path)
}

/// The farthest `held`'s other hand strays from its grasp over a path, sampled at 1 kHz once
/// timed.
fn strays_from(robot: &Robot, held: &Held, path: &[f32]) -> (f32, f32) {
    let trajectory = Trajectory::try_new(robot, path, 1.0).unwrap();
    trajectory.check(robot).unwrap();
    let timed = trajectory.sample(1000.0).unwrap();
    let off = timed.positions.chunks(timed.dof);
    off.map(|q| off_grasp(robot, held, q))
        .fold((0.0, 0.0), |(a, b), (c, d)| {
            (f32::max(a, c), f32::max(b, d))
        })
}

/// Both hands on the bar where it lies at `at`, by IK for the left hand's pose alone: the
/// collision-free solution nearest `seed`.
fn holding_at(device: &Device, worlds: &Worlds, at: Vec3, seed: &[f32]) -> Vec<f32> {
    let problem = IkProblem {
        seed: Some(seed.to_vec()),
        holding: Some(bar()),
        ..IkProblem {
            targets: vec![Some(left_at(at, 0.0)), None],
            ..IkProblem::new(0, Pose::IDENTITY)
        }
    };
    let solved = batchplan::solve_ik(device, worlds, &[problem], &common::fixture_ik()).unwrap();
    solved
        .nearest(0, seed)
        .expect("the bar is in reach")
        .to_vec()
}

fn input<T: std::fmt::Debug>(result: Result<T, Error>, names: &str) {
    match result {
        Err(Error::Input(message)) => assert!(message.contains(names), "{message}"),
        other => panic!("an input error naming {names:?}, got {other:?}"),
    }
}

const LOW: Vec3 = Vec3::new(0.3, 0.0, 0.2);

/// IK given the left hand's pose puts the right hand at its grasp, and says where in its
/// problem, the left hand's pose as given; a pose of its own that keeps its pin is taken as its
/// grasp, and one that breaks it is refused. The bar may touch both hands: held by the left alone
/// it collides with the right there.
#[test]
fn ik_places_the_other_hand_at_its_grasp() {
    let robot = common::two_pandas().unwrap();
    let ready = robot.default_q().to_vec();
    assert!(
        off_its_grasp(&robot, &ready).0 > 0.05,
        "the hands start off the bar, so IK has to place the right one"
    );
    for device in common::devices(&robot) {
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let held = bar();
        let left = left_at(LOW, 0.0);
        let right = left
            .mul_pose(held.grasp.inverse())
            .mul_pose(held.also[0].grasp);
        let problem = |second: Option<Pose>| IkProblem {
            seed: Some(ready.clone()),
            holding: Some(bar()),
            ..IkProblem {
                targets: vec![Some(left), second],
                ..IkProblem::new(0, Pose::IDENTITY)
            }
        };
        let o = common::fixture_ik();
        let solved = batchplan::solve_ik(&device, &worlds, &[problem(None)], &o).unwrap();
        let placed = solved.problems[0].targets[1].expect("the right hand has a pose");
        assert!(placed.position.distance(right.position) < 1e-6);
        assert_eq!(solved.problems[0].targets[0], Some(left), "as given");
        let q = solved.nearest(0, &ready).expect("the bar is in reach");
        let (away, turned) = off_its_grasp(&robot, q);
        assert!(
            away <= 2e-3 && turned <= 0.02,
            "{}: the right hand is {:.2} mm and {turned:.4} rad off its grasp",
            device.name(),
            away * 1e3
        );
        // Both hands' links may touch it; with the left alone holding it, the right is in it.
        let alone = Held {
            also: vec![],
            ..bar()
        };
        let touches = |held: &Held| {
            let contacts = device.contacts_holding(&worlds, &[0], q, 0.0, &[Some(held)]);
            contacts.unwrap().remove(0)
        };
        assert!(touches(&held).is_empty(), "{:?}", touches(&held));
        assert!(
            !touches(&alone).is_empty(),
            "held by the left hand alone, the bar collides with the right"
        );
        // A pose of its own is measured as its pin measures a configuration: 1.6 mm along each
        // axis, 2.8 mm away in all, keeps it; 2.5 mm along one axis does not.
        let off = |by: Vec3| Pose::new(right.position + by, right.rotation);
        let near = problem(Some(off(Vec3::splat(1.6e-3))));
        let near = batchplan::solve_ik(&device, &worlds, &[near], &o);
        assert_eq!(near.unwrap().problems[0].targets[1], Some(placed));
        let far = problem(Some(off(Vec3::X * 2.5e-3)));
        let far = batchplan::solve_ik(&device, &worlds, &[far], &o);
        input(
            far,
            "IK problem 0: the pose given hand 1 is off its grasp of the object hand 0 holds",
        );
    }
}

/// Plans holding the bar with both hands, given the left hand's goal pose alone, carry it over a
/// block: every path, sampled at 1 kHz once timed, keeps the right hand within 2 mm and 0.02 rad
/// of its grasp, the floors of a pin, and is collision-free with the bar in both hands. The same
/// ends planned holding nothing pull the hands centimeters apart. The result's problems are as
/// given, without constraints. A start with the right hand off the bar is refused.
#[test]
fn plans_carry_what_two_hands_hold_rigid_between_them() {
    let robot = common::two_pandas().unwrap();
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let start = holding_at(&device, &worlds, LOW, robot.default_q());
        let beyond = [
            (Vec3::new(0.62, 0.0, 0.32), 0.0),
            (Vec3::new(0.62, 0.05, 0.36), 0.3),
            (Vec3::new(0.6, -0.05, 0.34), -0.3),
            (Vec3::new(0.65, 0.0, 0.4), 0.15),
        ];
        let problems: Vec<PlanProblem> = beyond
            .iter()
            .map(|&(at, yaw)| PlanProblem {
                holding: Some(bar()),
                ..PlanProblem::new(
                    0,
                    start.clone(),
                    Goal::Poses(vec![Some(left_at(at, yaw)), None]),
                )
            })
            .collect();
        let o = PlanOptions::default();
        let result = batchplan::plan(&device, &worlds, &problems, &o).unwrap();
        assert!(
            result.solved().count() >= 3,
            "{name}: {} of {} carries planned: {:?}",
            result.solved().count(),
            problems.len(),
            result.failures
        );
        let held = bar();
        let mut apart = 0;
        for s in result.solved() {
            assert!(s.problem.constraints.is_empty());
            assert!(matches!(&s.problem.goal, Goal::Poses(targets) if targets[1].is_none()));
            let (away, turned) = strays(&robot, s.solution);
            assert!(
                away <= 2e-3 + 1e-5 && turned <= 0.02 + 1e-4,
                "{name}: carry {} moves the right hand {:.2} mm and {turned:.4} rad off its grasp",
                s.index,
                away * 1e3
            );
            let timed = Trajectory::try_new(&robot, s.solution, 1.0).unwrap();
            let timed = timed.sample(1000.0).unwrap();
            let samples = timed.positions.len() / timed.dof;
            let contacts = device.contacts_holding(
                &worlds,
                &vec![0; samples],
                &timed.positions,
                0.0,
                &vec![Some(&held); samples],
            );
            let colliding = contacts.unwrap().iter().filter(|c| !c.is_empty()).count();
            assert_eq!(colliding, 0, "{name}: carry {} collides", s.index);
            // The same ends, holding nothing.
            let n = robot.dof();
            let goal = s.solution[s.solution.len() - n..].to_vec();
            let loose = [PlanProblem::new(0, start.clone(), goal)];
            let loose = batchplan::plan(&device, &worlds, &loose, &o).unwrap();
            if let Some(path) = loose.best(0)
                && strays(&robot, path).0 > 0.01
            {
                apart += 1;
            }
        }
        assert!(
            2 * apart > result.solved().count(),
            "{name}: most carries planned holding nothing should pull the hands apart, not {apart}"
        );
        let off = PlanProblem {
            start: robot.default_q().to_vec(),
            ..problems[0].clone()
        };
        input(
            batchplan::plan(&device, &worlds, &[off], &o),
            "at the start, hand 1 is not at its grasp of the object hand 0 holds",
        );
        // A goal pose of the right hand's own, 4 mm from its grasp, is the problem's to answer for.
        let left = left_at(beyond[0].0, beyond[0].1);
        let right = left
            .mul_pose(held.grasp.inverse())
            .mul_pose(held.also[0].grasp);
        let shifted = Pose::new(right.position + Vec3::X * 4e-3, right.rotation);
        let apart = PlanProblem {
            goal: Goal::Poses(vec![Some(left), Some(shifted)]),
            ..problems[0].clone()
        };
        input(
            batchplan::plan(&device, &worlds, &[problems[0].clone(), apart], &o),
            "problem 1: the pose given hand 1 is off its grasp of the object hand 0 holds",
        );
    }
}

/// A hold names each other hand once, a hand of the robot's other than its own, at a finite
/// grasp with a unit rotation: anything else is an input error of the problem that holds so,
/// from IK and from a plan.
#[test]
fn a_hold_another_hand_cannot_share_is_an_input_error() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let start = holding_at(&device, &worlds, LOW, robot.default_q());
    let second = bar().also[0];
    let with = |also: Vec<Grip>| Held { also, ..bar() };
    let scaled = Pose::new(second.grasp.position, second.grasp.rotation * 2.0);
    let not_finite = Pose::new(Vec3::new(f32::NAN, 0.0, 0.0), second.grasp.rotation);
    let cases = [
        (
            vec![Grip { hand: 2, ..second }],
            "hand 2 holds the object too",
        ),
        (
            vec![Grip { hand: 0, ..second }],
            "hand 0 holds the object too",
        ),
        (vec![second, second], "hand 1 holds the object too"),
        (
            vec![Grip {
                grasp: scaled,
                ..second
            }],
            "hand 1's grasp of the held object must be finite with a unit rotation",
        ),
        (
            vec![Grip {
                grasp: not_finite,
                ..second
            }],
            "hand 1's grasp of the held object must be finite with a unit rotation",
        ),
    ];
    // A device that holds something in a hand already cannot take a hold that hand shares.
    let cube = Held {
        carries: vec![],
        touch_links: vec![],
        hand: 1,
        also: vec![],
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::splat(0.02)),
        grasp: Pose::IDENTITY,
        locked: Default::default(),
        spheres: Default::default(),
    };
    let busy = robot.attach(&cube.attachment(&robot).unwrap()).unwrap();
    let busy = device.with_robot(&busy).unwrap();
    let ik = IkProblem {
        seed: Some(start.clone()),
        holding: Some(bar()),
        ..IkProblem {
            targets: vec![Some(left_at(LOW, 0.0)), None],
            ..IkProblem::new(0, Pose::IDENTITY)
        }
    };
    input(
        batchplan::solve_ik(&busy, &worlds, &[ik], &common::fixture_ik()),
        "hand 1 holds the object too, and the robot already has 'held by hand 1'",
    );
    for (also, names) in cases {
        let held = with(also);
        let ik = IkProblem {
            seed: Some(start.clone()),
            holding: Some(held.clone()),
            ..IkProblem {
                targets: vec![Some(left_at(LOW, 0.0)), None],
                ..IkProblem::new(0, Pose::IDENTITY)
            }
        };
        let solved = batchplan::solve_ik(&device, &worlds, &[ik], &common::fixture_ik());
        input(solved, names);
        let carry = PlanProblem {
            holding: Some(held),
            ..PlanProblem::new(0, start.clone(), start.clone())
        };
        let planned = batchplan::plan(&device, &worlds, &[carry], &PlanOptions::default());
        input(planned, names);
    }
}

/// A join's blends keep both hands on what they hold: eight carries that zigzag across the
/// block's near side, turning the bar each way, joined 30 cm wide. Holding nothing, the blends
/// pull the right hand a centimeter off the bar (11.9 mm on the M4 Pro); holding it, the joins
/// still blend, and the right hand keeps its grasp. A blend is checked at the join's resolution,
/// as a plan's spline is, and between checks it strays past a pin's floors (2 mm, 0.02 rad) by a
/// hair: at 1 kHz to 2.44 mm and 0.0144 rad on the Ryzen and 1.93 mm on the M4 Pro (2026-11-27),
/// so the bound here is one and a half floors.
#[test]
fn joins_keep_both_hands_on_what_they_hold() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let n = robot.dof();
    let stops = [
        (Vec3::new(0.0, 0.15, 0.25), 0.5),
        (Vec3::new(0.02, -0.15, 0.05), -0.5),
        (Vec3::new(-0.05, 0.12, 0.3), 0.6),
        (Vec3::new(0.0, -0.12, 0.1), -0.6),
        (Vec3::new(0.02, 0.1, 0.2), 0.3),
        (Vec3::new(-0.04, -0.1, 0.3), -0.4),
        (Vec3::new(0.0, 0.15, 0.1), 0.5),
        (Vec3::ZERO, 0.0),
    ];
    let mut from = holding_at(&device, &worlds, LOW, robot.default_q());
    let mut paths = vec![];
    for (at, yaw) in stops {
        let carry = PlanProblem {
            holding: Some(bar()),
            ..PlanProblem::new(
                0,
                from.clone(),
                Goal::Poses(vec![Some(left_at(LOW + at, yaw)), None]),
            )
        };
        let planned = batchplan::plan(&device, &worlds, &[carry], &PlanOptions::default());
        let path = planned.unwrap().best(0).expect("the carry plans").to_vec();
        from = path[path.len() - n..].to_vec();
        paths.push(path);
    }
    // The farthest the right hand strays from its grasp over the joined motion, in meters and
    // radians, and how many joins blend.
    let joined = |holding: Option<Held>| {
        let sequence = Sequence {
            world: 0,
            blends: vec![0.3; paths.len() - 1],
            paths: paths.clone(),
            constraints: vec![],
            holding,
        };
        let joined = batchplan::join(&device, &worlds, &[sequence], &JoinOptions::default());
        let joined = joined.unwrap();
        let off = joined.paths[0].iter().map(|path| strays(&robot, path));
        let (away, turned) = off.fold((0.0, 0.0), |(a, b), (c, d)| {
            (f32::max(a, c), f32::max(b, d))
        });
        (away, turned, joined.blends[0].iter().flatten().count())
    };
    let (free, _, _) = joined(None);
    assert!(
        free > 4e-3,
        "holding nothing, the blends pull the right hand {:.2} mm off the bar: not past its pin",
        free * 1e3
    );
    let (away, turned, blended) = joined(Some(bar()));
    assert!(
        away <= 3e-3 && turned <= 0.03,
        "holding the bar, the right hand strays {:.2} mm and {turned:.4} rad from its grasp",
        away * 1e3
    );
    assert!(blended > 0, "holding the bar, no join blends");
}

/// A follower holding the bar with both hands keeps them on it as a constraint of its replans: a
/// trajectory that ends with the right hand off the bar is refused where it would replan.
#[test]
fn a_follower_keeps_both_hands_on_what_it_holds() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let start = holding_at(&device, &worlds, LOW, robot.default_q());
    let away = [PlanProblem::new(0, start, robot.default_q())];
    let away = batchplan::plan(&device, &worlds, &away, &PlanOptions::default()).unwrap();
    let path = away.best(0).expect("the hands leave the bar");
    let trajectory = Trajectory::try_new(&robot, path, 1.0).unwrap();
    let options = |holding: Option<Held>| FollowerOptions {
        holding,
        ..FollowerOptions::default()
    };
    assert!(Follower::try_new(&robot, 0, trajectory.clone(), 0.0, options(None)).is_ok());
    let one_hand = Held {
        also: vec![],
        ..bar()
    };
    let alone = Follower::try_new(&robot, 0, trajectory.clone(), 0.0, options(Some(one_hand)));
    assert!(alone.is_ok());
    let refused = Follower::try_new(&robot, 0, trajectory, 0.0, options(Some(bar())));
    input(
        refused.map(|_| ()),
        "the trajectory's goal breaks the follower's constraints",
    );
}

/// Hand `hand`'s straight move from `start` through `waypoints`, holding `holding`.
fn straight(start: &[f32], hand: usize, waypoints: Vec<Pose>, holding: Option<Held>) -> ToolPath {
    ToolPath {
        world: 0,
        start: start.to_vec(),
        hand,
        waypoints,
        hold: vec![],
        holding,
    }
}

/// A straight move of one hand takes the other along on what both hold: lifting the bar 15 cm
/// while turning it 0.3 rad, the right hand stays within the path's precisions of its grasp at
/// every step and within a pin's floors at 1 kHz, whichever hand the path names, where the same
/// path holding nothing leaves the right hand behind. A plan holding the bar starts where the
/// move ends. A start with the right hand off the bar is refused.
#[test]
fn a_straight_move_takes_the_other_hand_along() {
    let robot = common::two_pandas().unwrap();
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let start = holding_at(&device, &worlds, LOW, robot.default_q());
        let held = bar();
        let lifted = left_at(LOW + Vec3::Z * 0.15, 0.3);
        let right = |left: Pose| {
            left.mul_pose(held.grasp.inverse())
                .mul_pose(held.also[0].grasp)
        };
        let paths = [
            straight(&start, 0, vec![lifted], Some(bar())),
            straight(&start, 1, vec![right(lifted)], Some(bar())),
            straight(&start, 0, vec![lifted], None),
        ];
        let o = Default::default();
        let followed = batchplan::follow(&device, &worlds, &paths, &o).unwrap();
        assert_eq!(
            followed.fraction, [1.0; 3],
            "{name}: {:?}",
            followed.stopped
        );
        let n = robot.dof();
        for (k, path) in followed.paths[..2].iter().enumerate() {
            let (away, turned) = strays(&robot, path);
            assert!(
                away <= 2e-3 && turned <= 0.02,
                "{name}: path {k} moves the right hand {:.2} mm and {turned:.4} rad off its grasp",
                away * 1e3
            );
            let end = &path[path.len() - n..];
            let up = robot.ee_pose(end, 1).position.z - robot.ee_pose(&start, 1).position.z;
            assert!(
                (up - 0.15).abs() < 2e-3,
                "{name}: the right hand rose {up} m"
            );
            let on = PlanProblem {
                holding: Some(bar()),
                ..PlanProblem::new(
                    0,
                    end.to_vec(),
                    Goal::Poses(vec![Some(left_at(LOW, 0.0)), None]),
                )
            };
            let planned = batchplan::plan(&device, &worlds, &[on], &PlanOptions::default());
            assert!(
                planned.unwrap().best(0).is_some(),
                "{name}: a carry follows"
            );
        }
        let (left_behind, _) = strays(&robot, &followed.paths[2]);
        assert!(
            left_behind > 0.1,
            "holding nothing, the right hand stays where it was: {left_behind} m"
        );
        let off = straight(robot.default_q(), 0, vec![lifted], Some(bar()));
        input(
            batchplan::follow(&device, &worlds, &[off], &o),
            "path 0: at the start, hand 1 is not at its grasp of the object hand 0 holds",
        );
    }
}

/// The other hand's place in the path's hand's frame, and the farthest it moves from `place`
/// there over a path sampled at 1 kHz once timed: meters, radians.
fn place_of_the_right(robot: &Robot, q: &[f32]) -> Pose {
    robot.ee_pose(q, 0).inverse().mul_pose(robot.ee_pose(q, 1))
}

fn moves_from(robot: &Robot, place: Pose, path: &[f32]) -> (f32, f32) {
    let trajectory = Trajectory::try_new(robot, path, 1.0).unwrap();
    let timed = trajectory.sample(1000.0).unwrap();
    let off = timed.positions.chunks(timed.dof).map(|q| {
        let now = place_of_the_right(robot, q);
        (
            now.position.distance(place.position),
            now.rotation.angle_between(place.rotation),
        )
    });
    off.fold((0.0, 0.0), |(a, b), (c, d)| {
        (f32::max(a, c), f32::max(b, d))
    })
}

/// A straight move keeps the hands as the start has them. A start that holds the bar to what a
/// pin allows, the right hand 1.5 mm along two axes from the grasp the hold names, as two IK
/// solutions leave it, is followed whole at the default precision and at half of it, the right
/// hand keeping its place in the left's frame to the path's precisions: held to the grasp as
/// named, 2.1 mm away, it would be off the path before its first step.
#[test]
fn a_straight_move_keeps_the_hands_as_the_start_has_them() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let start = holding_at(&device, &worlds, LOW, robot.default_q());
    let named = bar().also[0];
    let shifted = Pose::new(
        named.grasp.position + Vec3::new(1.5e-3, 1.5e-3, 0.0),
        named.grasp.rotation,
    );
    let held = Held {
        also: vec![Grip {
            grasp: shifted,
            ..named
        }],
        ..bar()
    };
    let place = place_of_the_right(&robot, &start);
    let lifted = left_at(LOW + Vec3::Z * 0.15, 0.3);
    for precision in [1e-3, 5e-4] {
        let o = ToolPathOptions {
            precision,
            ..Default::default()
        };
        let path = straight(&start, 0, vec![lifted], Some(held.clone()));
        let followed = batchplan::follow(&device, &worlds, &[path], &o).unwrap();
        assert_eq!(followed.fraction[0], 1.0, "{:?}", followed.stopped[0]);
        let (away, turned) = moves_from(&robot, place, &followed.paths[0]);
        assert!(
            away <= 1.5 * precision && turned <= 1.5 * o.rotation_precision,
            "held to {precision} m, the right hand moves {:.3} mm and {turned:.4} rad from its \
             place at the start",
            away * 1e3
        );
    }
}

/// A hand that does not hold the object follows no path on joints that move a hand that does: on
/// two Pandas with a third hand on the left arm's fourth link, a path of the third while the
/// first two hold the bar is an input error, as it would turn the left arm under the bar; with
/// the joints it shares held, it is taken, and in one call with a path of a hand that holds the
/// bar each comes to what it does alone. A place sets down what two hands hold, not three.
#[test]
fn a_hand_that_does_not_hold_leaves_those_that_do_alone() {
    let two = common::two_pandas().unwrap();
    let on_two = Device::cpu(&two).unwrap();
    let worlds = on_two.upload(&[common::tray_scene()]).unwrap();
    let start = holding_at(&on_two, &worlds, LOW, two.default_q());
    let links = ["left_ee_link", "right_ee_link", "left_panda_link4"];
    let robot = common::two_pandas_with(&links).unwrap();
    assert_eq!((robot.hands(), robot.dof()), (3, two.dof()));
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let elbow = robot.ee_pose(&start, 2);
    let raised = Pose::new(elbow.position + Vec3::Z * 0.02, elbow.rotation);
    let path = straight(&start, 2, vec![raised], Some(bar()));
    input(
        batchplan::follow(
            &device,
            &worlds,
            std::slice::from_ref(&path),
            &Default::default(),
        ),
        "path 0: hand 2 follows a path on joints that move hand 0",
    );
    let still = ToolPath {
        hold: robot.hand_joints(2),
        ..path
    };
    // It has no joints left to move on. In one call with a path of a hand that holds, each comes
    // to what it does alone.
    let lifted = straight(
        &start,
        0,
        vec![left_at(LOW + Vec3::Z * 0.1, 0.0)],
        Some(bar()),
    );
    let o = ToolPathOptions::default();
    let both = [still.clone(), lifted.clone(), still];
    let followed = batchplan::follow(&device, &worlds, &both, &o).unwrap();
    assert_eq!(followed.fraction, [0.0, 1.0, 0.0], "{:?}", followed.stopped);
    let alone = batchplan::follow(&device, &worlds, &[lifted], &o).unwrap();
    assert_eq!(followed.paths[1], alone.paths[0]);
    assert!(followed.paths[0].is_empty() && followed.paths[2].is_empty());
    let three = Held {
        also: vec![
            bar().also[0],
            Grip {
                hand: 2,
                grasp: Pose::IDENTITY,
            },
        ],
        ..bar()
    };
    let place = Place::new(0, start, three, [common::tray(LOW, 0.0)]);
    input(
        batchplan::place(&device, &worlds, &[place], &Default::default()),
        "2 other hands hold the object with hand 0",
    );
}

/// A hand that has let go of what the other still holds backs off it: with both hands at the
/// bar, the left holding it alone, the right hand's straight move back and its plan away stop
/// where they start, its hand against the held bar; with the right hand's links among those the
/// bar may touch (`Held::touch_links`), both go through, and the links are let touch it, not
/// kept at it. A link the robot lacks is an input error that names it and its problem.
#[test]
fn a_hand_that_let_go_backs_off_what_the_other_holds() {
    let robot = common::two_pandas().unwrap();
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let start = holding_at(&device, &worlds, LOW, robot.default_q());
        let alone = Held {
            also: vec![],
            ..bar()
        };
        let touching = Held {
            touch_links: robot.hand_links(1).into_iter().map(String::from).collect(),
            ..alone.clone()
        };
        let at = robot.ee_pose(&start, 1);
        let back = Pose::new(at.position - at.rotation * Vec3::Z * 0.1, at.rotation);
        let off = |held: &Held| straight(&start, 1, vec![back], Some(held.clone()));
        let o = ToolPathOptions::default();
        let followed = batchplan::follow(&device, &worlds, &[off(&alone), off(&touching)], &o);
        let followed = followed.unwrap();
        assert!(
            matches!(followed.stopped[0], Some(ToolPathStop::Collision { .. })),
            "{name}: the hand that let go is against the held bar: {:?}",
            followed.stopped[0]
        );
        assert_eq!(
            followed.fraction,
            [0.0, 1.0],
            "{name}: {:?}",
            followed.stopped
        );
        // The left hand, which holds the bar, has not moved, and the right ends 10 cm back.
        let n = robot.dof();
        let end = &followed.paths[1][followed.paths[1].len() - n..];
        assert!(
            robot
                .ee_pose(end, 0)
                .position
                .distance(robot.ee_pose(&start, 0).position)
                < 1e-6
        );
        assert!((robot.ee_pose(end, 1).position.distance(at.position) - 0.1).abs() < 2e-3);
        // A plan from there, the right arm going home while the left holds still.
        let home: Vec<f32> = (0..n)
            .map(|j| match robot.hand_joints(1).contains(&j) {
                true => robot.default_q()[j],
                false => start[j],
            })
            .collect();
        let away = |held: &Held| PlanProblem {
            holding: Some(held.clone()),
            ..PlanProblem::new(0, start.clone(), home.clone())
        };
        let plans = [away(&alone), away(&touching)];
        let planned = batchplan::plan(&device, &worlds, &plans, &PlanOptions::default()).unwrap();
        assert!(
            planned.best(0).is_none() && planned.best(1).is_some(),
            "{name}: {:?}",
            planned.failures
        );
        let unknown = Held {
            touch_links: vec!["right_panda_elbow".into()],
            ..alone.clone()
        };
        input(
            batchplan::follow(&device, &worlds, &[off(&alone), off(&unknown)], &o),
            "path 1: the held object may touch 'right_panda_elbow', which is no link of the robot",
        );
    }
}

/// The other hand is held to the path's precisions between steps too. Turning the left hand half a
/// radian about its own axis in one step is one joint's motion, which keeps the left hand on its
/// path, and moves the right hand along an arc 12 cm long, which the joints' straight move from
/// one end to the other leaves by a third of a millimeter: held to 0.2 mm, the step is halved
/// until the right hand keeps its grasp, where the same turn holding nothing takes the one step.
/// With the right arm's joints held it cannot keep up, and the motion stops at the start.
#[test]
fn the_other_hand_keeps_its_grasp_between_steps() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let start = holding_at(&device, &worlds, LOW + Vec3::Z * 0.1, robot.default_q());
    let left = robot.ee_pose(&start, 0);
    let turned = Pose::new(
        left.position,
        left.rotation * glam::Quat::from_rotation_z(0.5),
    );
    let coarse = ToolPathOptions {
        step: 1.0,
        rotation_step: 1.0,
        precision: 2e-4,
        ..Default::default()
    };
    let turns = [
        straight(&start, 0, vec![turned], Some(bar())),
        straight(&start, 0, vec![turned], None),
    ];
    let followed = batchplan::follow(&device, &worlds, &turns, &coarse).unwrap();
    assert_eq!(followed.fraction, [1.0; 2], "{:?}", followed.stopped);
    let n = robot.dof();
    let [together, alone] = [0, 1].map(|k| followed.paths[k].len() / n);
    assert!(
        together > alone,
        "holding the bar the turn takes more steps than the {alone} control points of one, not \
         {together}"
    );
    let (away, turned_off) = strays(&robot, &followed.paths[0]);
    assert!(
        away <= 2e-4 && turned_off <= 0.01,
        "the right hand is {:.3} mm and {turned_off:.4} rad off its grasp",
        away * 1e3
    );
    let [turn, _] = turns;
    let held_still = ToolPath {
        hold: robot.hand_joints(1),
        ..turn
    };
    let stopped = batchplan::follow(&device, &worlds, &[held_still], &coarse).unwrap();
    assert_eq!(stopped.fraction[0], 0.0);
    assert!(
        matches!(stopped.stopped[0], Some(ToolPathStop::Unreachable { .. })),
        "{:?}",
        stopped.stopped[0]
    );
}

/// A handover moves what one hand holds, and refuses what two hold together.
#[test]
fn a_handover_refuses_what_two_hands_hold() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let start = holding_at(&device, &worlds, LOW, robot.default_q());
    let handover = Handover::new(&robot, 0, start, bar(), 1);
    input(
        batchplan::handover(&device, &worlds, &[handover], &Default::default()),
        "hands 0 and 1 hold the object together",
    );
}

/// Where the bar lies across the top of the tray scene's block, along y.
const ON_BLOCK: Vec3 = Vec3::new(0.5, 0.0, 0.265);

/// The tray scene's block, obstacle 0, with the bar lying on it at `ON_BLOCK`, obstacle 1.
fn bar_on_the_block() -> World {
    let mut world = common::tray_scene();
    let lying = bar().object.placed(common::tray(ON_BLOCK, 0.0));
    world.obstacles.push(lying);
    world
}

/// Grasps of the bar on the block from above toward hand `hand`'s end of it (the left's, +y, for
/// hand 0), as `grasps` holds it toward its ends: the two that close across it, each way round.
fn from_above(hand: usize) -> Vec<Hold> {
    let calibration = GripperCalibration::new(0.08, 0.0, HashMap::new());
    let gripper = common::panda().unwrap().gripper(0, &calibration).unwrap();
    let options = ShapeGraspOptions {
        along: 0.75,
        floor: Some(0.255),
        ..ShapeGraspOptions::default()
    };
    let all = batchplan::grasps(&bar_on_the_block().obstacles[1], &gripper, &options).unwrap();
    let side = if hand == 0 { 1.0 } else { -1.0 };
    let mine = all
        .into_iter()
        .filter(|g| (g.rotation * Vec3::Z).z < -0.9 && g.position.y * side > 0.0);
    let mine: Vec<Hold> = mine.map(Hold::from).collect();
    assert_eq!(mine.len(), 2, "hand {hand}'s end has two grasps from above");
    mine
}

/// The left hand picking the bar up off the block with the right, by pairs of their grasps from
/// above at their ends: each of the left's with each of the right's, the left's second costing 1,
/// and the right's second costing a half and closing the right fingers on the bar, 3 cm apart, so
/// that its pairs are a group of their own and touch the bar.
fn pick_by_two(robot: &Robot) -> Pick {
    let mut firsts = from_above(0);
    firsts[1].cost = 1.0;
    let mut seconds = from_above(1);
    seconds[1].cost = 0.5;
    let narrower = ["right_panda_finger_joint1", "right_panda_finger_joint2"];
    seconds[1].locked = HashMap::from(narrower.map(|j| (j.to_string(), 0.015)));
    let pairs = firsts
        .iter()
        .flat_map(|first| seconds.iter().map(move |second| (first, second)));
    let (firsts, seconds): (Vec<Hold>, Vec<Hold>) =
        pairs.map(|(a, b)| (a.clone(), b.clone())).unzip();
    Pick {
        also: Some(HandGrasps {
            hand: 1,
            grasps: seconds,
        }),
        support: vec![0],
        ..Pick::new(0, robot.default_q(), 1, firsts)
    }
}

/// Where the first plan of the pick by two hands leaves the robot, and what it holds.
fn picked_by_two(device: &Device) -> (Vec<f32>, Held) {
    let worlds = device.upload(&[bar_on_the_block()]).unwrap();
    let task = pick_by_two(device.robot());
    let picked = batchplan::pick(device, &worlds, &[task], &TaskOptions::default()).unwrap();
    let plans = picked.outcomes[0]
        .as_ref()
        .expect("the hands pick the bar up");
    (plans[0].end().to_vec(), plans[0].held.clone().unwrap())
}

/// That along `t`, sampled at 1 kHz, hand `second` stays where `t` starts it until hand `first`
/// is where `t` ends it, and `first` stays there after: each moves on a leg of its own, `first`'s
/// first. Each moves 5 cm or more.
fn in_turn(robot: &Robot, t: &Trajectory, [first, second]: [usize; 2], what: &str) {
    let n = robot.dof();
    let points = &t.control_points;
    let (begin, end) = (&points[..n], &points[points.len() - n..]);
    let waiting = robot.ee_pose(begin, second).position;
    let done = robot.ee_pose(end, first).position;
    let timed = t.sample(1000.0).unwrap();
    for (k, q) in timed.positions.chunks(n).enumerate() {
        let waits = robot.ee_pose(q, second).position.distance(waiting) < 1e-4;
        let there = robot.ee_pose(q, first).position.distance(done) < 1e-4;
        assert!(
            waits || there,
            "{what}: at {k} ms hand {first} moves while hand {second} does"
        );
    }
    for hand in [first, second] {
        let moved = robot.ee_pose(begin, hand).position;
        let moved = moved.distance(robot.ee_pose(end, hand).position);
        assert!(moved > 0.05, "{what}: hand {hand} moves {moved} m");
    }
}

/// The fastest hand `hand` moves along `t`, sampled at 1 kHz: from the joint velocities, the
/// hand's positions 5 ms of that motion either side.
fn fastest(robot: &Robot, t: &Trajectory, hand: usize) -> f32 {
    let n = robot.dof();
    let timed = t.sample(1000.0).unwrap();
    let states = timed.positions.chunks(n).zip(timed.velocities.chunks(n));
    let h = 0.005;
    let speeds = states.map(|(q, dq)| {
        let moved = |by: f32| {
            let q: Vec<f32> = q.iter().zip(dq).map(|(q, dq)| q + by * dq).collect();
            robot.ee_pose(&q, hand).position
        };
        moved(h).distance(moved(-h)) / (2.0 * h)
    });
    speeds.fold(0.0, f32::max)
}

/// How many of `t`'s states, at 1 kHz, collide holding `held` in world 0 of `worlds`.
fn colliding(device: &Device, worlds: &Worlds, held: &Held, t: &Trajectory) -> usize {
    let timed = t.sample(1000.0).unwrap();
    let samples = timed.positions.len() / timed.dof;
    let contacts = device.contacts_holding(
        worlds,
        &vec![0; samples],
        &timed.positions,
        0.0,
        &vec![Some(held); samples],
    );
    contacts.unwrap().iter().filter(|c| !c.is_empty()).count()
}

/// Two hands pick the bar up off the block, each by grasps of its own at its end, every pair
/// getting through, those whose right fingers close on the bar too: candidates are the pairs as
/// given, each costing its two holds' sum and keeping both holds' fingers, and the plans hold the
/// bar with the right hand at the pair's grasp. Sampled at 1 kHz, the left hand moves in while
/// the right waits, then the right while the left waits at
/// its grasp, each under the tool speed, and both end at their grasps; the lift raises the right
/// hand with the left, within 2 mm and 0.02 rad of its grasp, clear of everything holding the bar.
/// With a count of plans that will do, the first that many pairs in the given order come back, as
/// trying every pair gives them.
#[test]
fn two_hands_pick_up_the_bar_by_its_ends() {
    let robot = common::two_pandas().unwrap();
    let n = robot.dof();
    let task = pick_by_two(&robot);
    let (firsts, seconds) = (&task.grasps, &task.also.as_ref().unwrap().grasps);
    let lying = common::tray(ON_BLOCK, 0.0);
    let o = TaskOptions::default();
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[bar_on_the_block()]).unwrap();
        let lifting = task.lift_world(&bar_on_the_block()).unwrap();
        let lifting = device.upload(&[lifting]).unwrap();
        let plans_of = |task: &Pick| {
            let result = batchplan::pick(&device, &worlds, std::slice::from_ref(task), &o);
            let mut result = result.unwrap();
            assert_eq!(
                result.problems[0].grasps, task.grasps,
                "{name}: the problem as given"
            );
            result.outcomes.remove(0)
        };
        let all = plans_of(&task).unwrap_or_else(|failure| panic!("{name}: {failure}"));
        assert_eq!(all.len(), 4, "{name}: every pair gets through");
        for plan in &all {
            let what = format!("{name}: pair {}", plan.candidate);
            let k = plan.candidate;
            assert!(k < firsts.len(), "{what}");
            let held = plan
                .held
                .clone()
                .expect("a pick leaves the hands holding the bar");
            let mut locked = firsts[k].locked.clone();
            locked.extend(seconds[k].locked.clone());
            assert_eq!(
                (held.hand, held.grasp, &held.locked),
                (0, firsts[k].pose, &locked),
                "{what}"
            );
            let second = Grip {
                hand: 1,
                grasp: seconds[k].pose,
            };
            assert_eq!(held.also, [second], "{what}");
            assert_eq!(plan.cost, firsts[k].cost + seconds[k].cost, "{what}");
            in_turn(
                &robot,
                &plan.approach,
                [0, 1],
                &format!("{what}, moving in"),
            );
            let points = &plan.approach.control_points;
            let grasped = &points[points.len() - n..];
            for (hand, grasp) in [(0, firsts[k].pose), (1, seconds[k].pose)] {
                let (at, wanted) = (robot.ee_pose(grasped, hand), lying.mul_pose(grasp));
                assert!(
                    at.position.distance(wanted.position) < 2e-3
                        && at.rotation.angle_between(wanted.rotation) < 0.02,
                    "{what}: hand {hand} at {at:?}, not its grasp {wanted:?}"
                );
            }
            for (kind, t) in [("in", &plan.approach), ("away", &plan.depart)] {
                for hand in [0, 1] {
                    let fastest = fastest(&robot, t, hand);
                    assert!(
                        fastest <= o.tool_speed * (1.0 + 1e-3),
                        "{what}: hand {hand} moves {kind} at {fastest} m/s"
                    );
                }
            }
            let (away, turned) = strays_from(&robot, &held, &plan.depart.control_points);
            assert!(
                away <= 2e-3 && turned <= 0.02,
                "{what}: the lift moves the right hand {:.2} mm and {turned:.4} rad off its grasp",
                away * 1e3
            );
            let height = |q: &[f32]| robot.ee_pose(q, 1).position.z;
            let rose = height(plan.end()) - height(grasped);
            assert!(
                (rose - task.lift.z).abs() < 2e-3,
                "{what}: the right hand rises {rose} m with the left"
            );
            let collides = colliding(&device, &lifting, &held, &plan.depart);
            assert_eq!(collides, 0, "{what}: the lift collides");
        }
        let mut through: Vec<usize> = all.iter().map(|plan| plan.candidate).collect();
        through.sort_unstable();
        for enough in [1, 2] {
            let some = Pick {
                enough: Some(enough),
                ..task.clone()
            };
            let found = plans_of(&some).unwrap();
            let first = &through[..enough.min(through.len())];
            let wanted = all.iter().filter(|plan| first.contains(&plan.candidate));
            let wanted: Vec<usize> = wanted.map(|plan| plan.candidate).collect();
            let found: Vec<usize> = found.iter().map(|plan| plan.candidate).collect();
            assert_eq!(found, wanted, "{name}: {enough} will do");
        }
    }
}

/// A pick by two hands needs another hand of the robot's than its own, with a grasp for each of
/// the pick's, checked as a pick's own are, and pairs whose holds keep each locked joint at one
/// value and whose costs add up to a finite one; a place of what two hands hold needs both at
/// their grasps at its start. Anything else is an input error that names it.
#[test]
fn two_hands_refuse_what_they_cannot_share() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[bar_on_the_block()]).unwrap();
    let o = TaskOptions::default();
    let task = pick_by_two(&robot);
    let second = |change: &dyn Fn(&mut HandGrasps)| {
        let mut changed = task.clone();
        change(changed.also.as_mut().unwrap());
        changed
    };
    let finger = |value: f32| HashMap::from([("right_panda_finger_joint1".to_string(), value)]);
    let mut disagree = task.clone();
    disagree.grasps[1].locked = finger(0.03);
    let costly = |hold: &Hold| Hold {
        cost: f32::MAX,
        ..hold.clone()
    };
    let past_any_cost = Pick {
        grasps: task.grasps.iter().map(costly).collect(),
        ..second(&|also| also.grasps = also.grasps.iter().map(costly).collect())
    };
    let cases = [
        (
            second(&|also| also.hand = 2),
            "task 0: hand 2 takes hold with hand 0",
        ),
        (
            second(&|also| also.hand = 0),
            "task 0: hand 0 takes hold with hand 0",
        ),
        (
            second(&|also| also.grasps.truncate(3)),
            "task 0 has 4 grasps and 3 of the second hand's",
        ),
        (
            second(&|also| also.grasps.clear()),
            "task 0 has 4 grasps and 0 of the second hand's",
        ),
        (
            second(&|also| also.grasps[1].pose.rotation *= 2.0),
            "task 0: second hand's grasp 1 must be finite with a unit rotation",
        ),
        (
            second(&|also| also.grasps[0].cost = f32::NAN),
            "task 0: second hand's grasp 0 must cost a finite amount",
        ),
        (
            second(&|also| also.grasps[0].locked = HashMap::from([("no_joint".into(), 0.0)])),
            "task 0, second hand's grasp 0",
        ),
        (
            disagree,
            "task 0: pair 1 locks joint 'right_panda_finger_joint1' at 0.03 by the first hand's \
             grasp and at 0.015 by the second's",
        ),
        (past_any_cost, "task 0: pair 0's grasps cost"),
    ];
    for (pick, names) in cases {
        input(batchplan::pick(&device, &worlds, &[pick], &o), names);
    }
    let bare = device.upload(&[common::tray_scene()]).unwrap();
    let off = Place::new(0, robot.default_q(), bar(), [common::tray(ON_BLOCK, 0.0)]);
    input(
        batchplan::place(&device, &bare, &[off], &o),
        "task 0: at the start, hand 1 is not at its grasp of the object hand 0 holds",
    );
}

/// Two hands set down the bar they picked up, 4 cm along the block: the transit and the lower keep
/// the right hand within 2 mm and 0.02 rad of its grasp at 1 kHz; the left hand backs off 10 cm
/// along its z axis while the right waits, then the right, its fingers opened only to the bar's
/// width, touching it; the bar is placed where it was let go
/// of; and a plan holding nothing starts where the place ends.
#[test]
fn two_hands_set_the_bar_down_and_back_off_in_turn() {
    let robot = common::two_pandas().unwrap();
    let n = robot.dof();
    let o = TaskOptions::default();
    let release = common::tray(ON_BLOCK + Vec3::Y * 0.04, 0.0);
    for device in common::devices(&robot) {
        let name = device.name();
        let (start, held) = picked_by_two(&device);
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let on_the_bar = ["right_panda_finger_joint1", "right_panda_finger_joint2"];
        let place = Place {
            support: vec![0],
            open: vec![HashMap::from(on_the_bar.map(|j| (j.to_string(), 0.015)))],
            ..Place::new(0, start.clone(), held.clone(), [release])
        };
        let result = batchplan::place(&device, &worlds, std::slice::from_ref(&place), &o).unwrap();
        let plans = result.outcomes[0].as_ref();
        let plans = plans.unwrap_or_else(|failure| panic!("{name}: {failure}"));
        for plan in plans {
            let transit = plan.transit.as_ref().expect("the place plans its transit");
            assert_eq!(&transit.control_points[..n], &start[..], "{name}");
            for (kind, t) in [("transit", transit), ("lower", &plan.approach)] {
                let (away, turned) = strays_from(&robot, &held, &t.control_points);
                assert!(
                    away <= 2e-3 + 1e-5 && turned <= 0.02 + 1e-4,
                    "{name}: the {kind} moves the right hand {:.2} mm and {turned:.4} rad off its \
                     grasp",
                    away * 1e3
                );
            }
            let points = &plan.approach.control_points;
            let let_go = &points[points.len() - n..];
            let height = |q: &[f32]| robot.ee_pose(q, 1).position.z;
            let lowered = height(&points[..n]) - height(let_go);
            assert!(
                (lowered - 0.1).abs() < 2e-3,
                "{name}: the right hand lowers {lowered} m"
            );
            assert_eq!(plan.opening, Some(0), "{name}");
            in_turn(
                &robot,
                &plan.depart,
                [0, 1],
                &format!("{name}, backing off"),
            );
            let grasps = [(0, held.grasp), (1, held.also[0].grasp)];
            for (hand, grasp) in grasps {
                let wanted = release.mul_pose(grasp);
                let back = wanted.position - wanted.rotation * Vec3::Z * place.retreat;
                let at = robot.ee_pose(plan.end(), hand).position;
                assert!(
                    at.distance(back) < 2e-3,
                    "{name}: hand {hand} backs off to {at}, not {back}"
                );
            }
            let set_down = held.object.placed(release);
            assert_eq!(plan.placed.as_ref(), Some(&set_down), "{name}");
            let mut there = common::tray_scene();
            there.obstacles.push(set_down);
            let there = device.upload(&[there]).unwrap();
            let away = [PlanProblem::new(0, plan.end().to_vec(), robot.default_q())];
            let away = batchplan::plan(&device, &there, &away, &PlanOptions::default()).unwrap();
            assert!(away.best(0).is_some(), "{name}: {:?}", away.failures);
        }
    }
}

/// Pick, carry and place on one device, each starting where the one before ends: the hands pick
/// the bar up off the block, a plan holding it carries it up in front of the block, and they set
/// it down on the block again, the place's transit keeping the right hand at its grasp.
#[test]
fn two_hands_pick_carry_and_place_the_bar() {
    let robot = common::two_pandas().unwrap();
    let n = robot.dof();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    let worlds = device
        .upload(&[bar_on_the_block(), common::tray_scene()])
        .unwrap();
    let picked = batchplan::pick(&device, &worlds, &[pick_by_two(&robot)], &o).unwrap();
    let picked = &picked.outcomes[0]
        .as_ref()
        .expect("the hands pick the bar up")[0];
    let pick_transit = picked.transit.as_ref().expect("the pick plans its transit");
    assert_eq!(&pick_transit.control_points[..n], robot.default_q());
    let held = picked.held.clone().unwrap();
    let up = common::tray(Vec3::new(0.3, 0.0, 0.45), 0.3).mul_pose(held.grasp);
    let carry = PlanProblem {
        holding: Some(held.clone()),
        ..PlanProblem::new(1, picked.end().to_vec(), Goal::Poses(vec![Some(up), None]))
    };
    let carried = batchplan::plan(&device, &worlds, &[carry], &PlanOptions::default()).unwrap();
    let carried = carried.best(0).expect("the hands carry the bar").to_vec();
    assert_eq!(&carried[..n], picked.end());
    let (away, turned) = strays_from(&robot, &held, &carried);
    assert!(away <= 2e-3 + 1e-5 && turned <= 0.02 + 1e-4);
    let end = &carried[carried.len() - n..];
    let releases =
        [-0.04, 0.0, 0.04].map(|y| Release::from(common::tray(ON_BLOCK + Vec3::Y * y, 0.0)));
    let place = Place {
        support: vec![0],
        ..Place::new(1, end.to_vec(), held.clone(), releases)
    };
    let placed = batchplan::place(&device, &worlds, &[place], &o).unwrap();
    let placed = &placed.outcomes[0]
        .as_ref()
        .expect("the hands set the bar down")[0];
    let transit = placed
        .transit
        .as_ref()
        .expect("the place plans its transit");
    assert_eq!(&transit.control_points[..n], end);
    let (away, turned) = strays_from(&robot, &held, &transit.control_points);
    assert!(
        away <= 2e-3 + 1e-5 && turned <= 0.02 + 1e-4,
        "the place's transit moves the right hand {:.2} mm and {turned:.4} rad off its grasp",
        away * 1e3
    );
    let release = releases[placed.candidate].pose;
    assert_eq!(placed.placed, Some(held.object.placed(release)));
}

/// What a task's plans come to, to tell two calls' apart: each plan's candidate, cost and
/// motions' control points.
fn plans_of(outcome: &Result<Vec<TaskPlan>, TaskFailure>) -> Vec<(usize, f32, Vec<Vec<f32>>)> {
    let plans = outcome
        .as_ref()
        .unwrap_or_else(|failure| panic!("{failure}"));
    let motions = |plan: &TaskPlan| plan.motions().map(|t| t.control_points.clone()).collect();
    plans
        .iter()
        .map(|plan| (plan.candidate, plan.cost, motions(plan)))
        .collect()
}

/// One call takes tasks by one hand and by two: a pick of the bar by the left hand alone and one
/// by both, and places of what each then holds, come to what each comes to in a call of its own.
/// A place by two hands takes an idle configuration for the joints neither hand moves alone: on
/// two arms that is none, so one that puts the right arm's joints past their limits changes
/// nothing.
#[test]
fn one_call_takes_tasks_by_one_hand_and_by_two() {
    let robot = common::two_pandas().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let o = TaskOptions::default();
    let worlds = device.upload(&[bar_on_the_block()]).unwrap();
    let picks = [
        pick_by_two(&robot),
        Pick {
            support: vec![0],
            ..Pick::new(0, robot.default_q(), 1, from_above(0))
        },
    ];
    let together = batchplan::pick(&device, &worlds, &picks, &o).unwrap();
    let mut places = vec![];
    for (task, outcome) in picks.iter().zip(&together.outcomes) {
        let alone = batchplan::pick(&device, &worlds, std::slice::from_ref(task), &o).unwrap();
        assert_eq!(plans_of(outcome), plans_of(&alone.outcomes[0]));
        let first = &outcome.as_ref().unwrap()[0];
        let held = first.held.clone().expect("a pick holds what it picked up");
        assert_eq!(held.also.len(), usize::from(task.also.is_some()));
        places.push(Place {
            support: vec![0],
            ..Place::new(
                0,
                first.end().to_vec(),
                held,
                [common::tray(ON_BLOCK + Vec3::Y * 0.04, 0.0)],
            )
        });
    }
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let together = batchplan::place(&device, &worlds, &places, &o).unwrap();
    for (task, outcome) in places.iter().zip(&together.outcomes) {
        let alone = batchplan::place(&device, &worlds, std::slice::from_ref(task), &o).unwrap();
        assert_eq!(plans_of(outcome), plans_of(&alone.outcomes[0]));
    }
    let mut idle = places[0].start.clone();
    for j in robot.hand_joints(1) {
        idle[j] = 100.0;
    }
    let idling = Place {
        idle: Some(idle),
        ..places[0].clone()
    };
    let idled = batchplan::place(&device, &worlds, &[idling], &o).unwrap();
    assert_eq!(
        plans_of(&idled.outcomes[0]),
        plans_of(&together.outcomes[0])
    );
}

/// A bar the left hand holds alone, by one end.
fn bar_in_the_left() -> Held {
    Held {
        also: vec![],
        ..bar()
    }
}

/// A 4 cm cube the right hand holds from above, its fingers closing across the bar's length.
fn cube_in_the_right() -> Held {
    Held {
        carries: vec![],
        touch_links: vec![],
        hand: 1,
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::splat(0.02)),
        grasp: Pose::new(Vec3::ZERO, common::upright(-FRAC_PI_2)),
        also: vec![],
        locked: Default::default(),
        spheres: Default::default(),
    }
}

/// Where the left hand holds the bar level in the air, and the cube where it lies on the bar's
/// far end.
const BAR_AT: Vec3 = Vec3::new(0.35, 0.0, 0.45);

fn cube_on_the_bar() -> Pose {
    common::tray(BAR_AT + Vec3::new(0.0, -0.10, 0.035), 0.0)
}

/// The robot with the left hand at its grasp of the bar where it is held, and the right at its
/// grasp of the cube at `cube`: the solution nearest the ready pose, holding neither.
fn each_holding(device: &Device, worlds: &Worlds, cube: Pose) -> Vec<f32> {
    let robot = device.robot();
    let problem = IkProblem {
        seed: Some(robot.default_q().to_vec()),
        ..IkProblem {
            targets: (0..robot.hands())
                .map(|hand| match hand {
                    0 => Some(common::tray(BAR_AT, 0.0).mul_pose(bar_in_the_left().grasp)),
                    1 => Some(cube.mul_pose(cube_in_the_right().grasp)),
                    _ => None,
                })
                .collect(),
            ..IkProblem::new(0, Pose::IDENTITY)
        }
    };
    let solved = batchplan::solve_ik(device, worlds, &[problem], &common::fixture_ik()).unwrap();
    let nearest = solved.nearest(0, robot.default_q());
    nearest.expect("both hands reach").to_vec()
}

/// Whether the joints `joints` stay where `start` has them at every control point of `plan`'s
/// motions.
fn stay(plan: &TaskPlan, start: &[f32], joints: &[usize]) -> bool {
    let n = start.len();
    plan.motions().into_iter().all(|t| {
        let mut points = t.control_points.chunks(n);
        points.all(|q| joints.iter().all(|&j| q[j] == start[j]))
    })
}

/// The right hand sets a cube on the far end of a bar the left hand holds in the air
/// (`Place::onto`), on a device that holds neither. On a device that holds the bar, the same
/// place without it stops where the hand lets go, the cube against the bar, so the release does
/// touch it. With it, the plan lowers the cube onto the bar and backs off, the left arm where the
/// start has it throughout, whatever `idle` says of its joints. The cube may touch the bar only
/// where it arrives: from below and beyond the bar's end, the transit, sampled at 1 kHz, goes
/// around the bar holding the cube, and a start with the cube against the bar is refused. From
/// the plan's end, with the cube in the world where it lies, the right arm goes home past the bar
/// the left still holds.
#[test]
fn a_hand_sets_its_object_on_what_the_other_holds() {
    let robot = common::two_pandas().unwrap();
    let o = TaskOptions::default();
    let (bar, cube) = (bar_in_the_left(), cube_in_the_right());
    let release = cube_on_the_bar();
    let left = robot.hand_joints(0);
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let above = common::tray(Vec3::new(0.35, -0.3, 0.6), 0.0);
        let start = each_holding(&device, &worlds, above);
        let holding_the_bar = robot.attach(&bar.attachment(&robot).unwrap()).unwrap();
        let holder = device.with_robot(&holding_the_bar).unwrap();

        let unsupported = Place::new(0, start.clone(), cube.clone(), [release]);
        let stopped = batchplan::place(&holder, &worlds, &[unsupported], &o).unwrap();
        let failure = stopped.outcomes[0]
            .as_ref()
            .expect_err("the cube meets the bar");
        assert_eq!(failure.at(TaskStep::Reach), 1, "{name}: {failure}");
        assert!(
            matches!(
                failure.candidates[0].why,
                Some(ToolPathStop::Collision { .. })
            ),
            "{name}: {failure}"
        );

        // An idle configuration counts for no joint that moves the support: not a number there.
        let idle: Vec<f32> = (0..robot.dof())
            .map(|j| match left.contains(&j) {
                true => f32::NAN,
                false => robot.default_q()[j],
            })
            .collect();
        let place = Place {
            onto: Some(bar.clone()),
            idle: Some(idle),
            ..Place::new(0, start.clone(), cube.clone(), [release])
        };
        let result = batchplan::place(&device, &worlds, std::slice::from_ref(&place), &o).unwrap();
        let plans = result.outcomes[0].as_ref();
        let plans = plans.unwrap_or_else(|failure| panic!("{name}: {failure}"));
        for plan in plans {
            assert!(stay(plan, &start, &left), "{name}: the left arm moves");
            let transit = plan.transit.as_ref().expect("the place plans its transit");
            assert_eq!(colliding(&holder, &worlds, &cube, transit), 0, "{name}");
            let n = robot.dof();
            let points = &plan.approach.control_points;
            let let_go = robot.ee_pose(&points[points.len() - n..], 1);
            let wanted = release.mul_pose(cube.grasp);
            assert!(
                let_go.position.distance(wanted.position) < 2e-3,
                "{name}: the hand lets go at {}, not {}",
                let_go.position,
                wanted.position
            );
            let set_down = cube.object.placed(release);
            assert_eq!(plan.placed.as_ref(), Some(&set_down), "{name}");
            // The cube lies in the world against the bar the left hand still holds.
            let mut there = common::tray_scene();
            there.obstacles.push(set_down);
            there.touch = vec![batchplan::Touch::held(1, 0)];
            let there = holder.upload(&[there]).unwrap();
            let home: Vec<f32> = (0..n)
                .map(|j| match left.contains(&j) {
                    true => start[j],
                    false => robot.default_q()[j],
                })
                .collect();
            let away = [PlanProblem::new(0, plan.end().to_vec(), home)];
            let away = batchplan::plan(&holder, &there, &away, &PlanOptions::default()).unwrap();
            assert!(away.best(0).is_some(), "{name}: {:?}", away.failures);
        }

        // From below and beyond the bar's end, the joints' straight way to where the move in
        // begins takes the cube through the bar.
        let beyond = common::tray(BAR_AT + Vec3::new(0.0, -0.24, -0.09), 0.0);
        let beyond = each_holding(&device, &worlds, beyond);
        let around = Place {
            onto: Some(bar.clone()),
            ..Place::new(0, beyond.clone(), cube.clone(), [release])
        };
        let result = batchplan::place(&device, &worlds, &[around], &o).unwrap();
        let plans = result.outcomes[0].as_ref();
        let plans =
            plans.unwrap_or_else(|failure| panic!("{name}, from beyond the bar: {failure}"));
        for plan in plans {
            let transit = plan.transit.as_ref().expect("the place plans its transit");
            let n = robot.dof();
            let begins = &transit.control_points[transit.control_points.len() - n..];
            let direct: Vec<f32> = (0..=50)
                .flat_map(|k| {
                    let ends = beyond.iter().zip(begins);
                    ends.map(move |(a, b)| a + (b - a) * k as f32 / 50.0)
                })
                .collect();
            let holding = vec![Some(&cube); 51];
            let met = holder.contacts_holding(&worlds, &[0; 51], &direct, 0.0, &holding);
            assert!(
                met.unwrap().iter().any(|contacts| !contacts.is_empty()),
                "{name}: the straight way is through the bar"
            );
            assert_eq!(colliding(&holder, &worlds, &cube, transit), 0, "{name}");
        }

        let against = each_holding(&device, &worlds, release);
        let touching = Place {
            onto: Some(bar.clone()),
            ..Place::new(0, against, cube.clone(), [release])
        };
        let refused = batchplan::place(&device, &worlds, &[touching], &o).unwrap();
        let failure = refused.outcomes[0]
            .as_ref()
            .expect_err("the start collides");
        assert_eq!(failure.at(TaskStep::Start), 1, "{name}: {failure}");
    }
}

/// What an object is set on is what other hands hold, by a hold as good as any: the placing
/// hand's own, a hand the robot lacks, an object two hands hold, and worlds that let the placing
/// hand's object touch obstacles by its link are each an input error naming it.
#[test]
fn what_an_object_is_set_on_is_another_hands() {
    let robot = common::two_pandas().unwrap();
    let device = &common::devices(&robot)[0];
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let o = TaskOptions::default();
    let start = each_holding(
        device,
        &worlds,
        common::tray(Vec3::new(0.35, -0.3, 0.6), 0.0),
    );
    let onto = |onto: Held, held: Held, start: Vec<f32>| Place {
        onto: Some(onto),
        ..Place::new(0, start, held, [cube_on_the_bar()])
    };
    let own = onto(cube_in_the_right(), cube_in_the_right(), start.clone());
    input(
        batchplan::place(device, &worlds, &[own], &o),
        "task 0: hand 1 holds the object and what it is set on (`onto`)",
    );
    let lacking = Held {
        hand: 2,
        ..bar_in_the_left()
    };
    let lacking = onto(lacking, cube_in_the_right(), start.clone());
    input(
        batchplan::place(device, &worlds, &[lacking], &o),
        "task 0: what the object is set on (`onto`): holds an object in hand 2 of 2",
    );
    let both = holding_at(device, &worlds, LOW, robot.default_q());
    let by_two = onto(cube_in_the_right(), bar(), both);
    input(
        batchplan::place(device, &worlds, &[by_two], &o),
        "task 0: an object 2 hands hold cannot be set on what another holds (`onto`)",
    );
    let mut touched = common::tray_scene();
    touched.touch = vec![batchplan::Touch::held(0, 1)];
    let touched = device.upload(&[touched]).unwrap();
    let named = onto(bar_in_the_left(), cube_in_the_right(), start);
    input(
        batchplan::place(device, &touched, &[named], &o),
        "task 0: these worlds let what hand 1 holds touch obstacles (`Touch::held`)",
    );
}

/// A support two hands hold must have both at their grasps at the start: with a third hand on the
/// left arm named as holding the bar 5 cm from where it is, the place is refused.
#[test]
fn a_support_two_hands_hold_starts_with_both_on_it() {
    let links = ["left_ee_link", "right_ee_link", "left_panda_link7"];
    let robot = common::two_pandas_with(&links).unwrap();
    let device = &common::devices(&robot)[0];
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let start = each_holding(
        device,
        &worlds,
        common::tray(Vec3::new(0.35, -0.3, 0.6), 0.0),
    );
    let bar_at = robot
        .ee_pose(&start, 0)
        .mul_pose(bar_in_the_left().grasp.inverse());
    let third = bar_at.inverse().mul_pose(robot.ee_pose(&start, 2));
    let by = |grasp: Pose| Held {
        also: vec![Grip { hand: 2, grasp }],
        ..bar_in_the_left()
    };
    let onto = |onto: Held| Place {
        onto: Some(onto),
        ..Place::new(0, start.clone(), cube_in_the_right(), [cube_on_the_bar()])
    };
    let o = TaskOptions::default();
    let held = batchplan::place(device, &worlds, &[onto(by(third))], &o).unwrap();
    assert!(held.outcomes[0].is_ok(), "{:?}", held.outcomes[0]);
    let off = Pose::new(third.position + Vec3::X * 0.05, third.rotation);
    input(
        batchplan::place(device, &worlds, &[onto(by(off))], &o),
        "task 0: at the start, what the object is set on (`onto`): hand 2 is not at its grasp",
    );
}

/// One call takes places onto a held thing beside places onto none, each coming to what it comes
/// to in a call of its own, in the tasks' order; and a place onto a held thing that says one plan
/// will do gives the first of its releases to get through.
#[test]
fn one_call_sets_objects_on_a_held_thing_and_on_none() {
    let robot = common::two_pandas().unwrap();
    let device = &common::devices(&robot)[0];
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let o = TaskOptions::default();
    let start = each_holding(
        device,
        &worlds,
        common::tray(Vec3::new(0.35, -0.3, 0.6), 0.0),
    );
    let cube = cube_in_the_right();
    let aside = common::tray(Vec3::new(0.35, -0.35, 0.45), 0.0);
    let plain = Place::new(0, start.clone(), cube.clone(), [aside]);
    let further = common::tray(BAR_AT + Vec3::new(0.0, -0.13, 0.035), 0.0);
    let onto = Place {
        onto: Some(bar_in_the_left()),
        ..Place::new(0, start.clone(), cube, [cube_on_the_bar(), further])
    };
    let alone = |task: &Place| {
        let result = batchplan::place(device, &worlds, std::slice::from_ref(task), &o).unwrap();
        plans_of(&result.outcomes[0])
    };
    let nearer = common::tray(Vec3::new(0.3, -0.35, 0.45), 0.0);
    let other = Place::new(0, start, cube_in_the_right(), [nearer]);
    assert_ne!(alone(&plain), alone(&other), "the two onto none differ");
    let tasks = [plain.clone(), onto.clone(), other];
    let together = batchplan::place(device, &worlds, &tasks, &o).unwrap();
    for (task, outcome) in tasks.iter().zip(&together.outcomes) {
        assert_eq!(plans_of(outcome), alone(task));
    }
    let every = alone(&onto);
    assert_eq!(every.len(), 2, "both releases on the bar get through");
    assert_ne!(alone(&plain), every, "the two tasks differ");
    let one = Place {
        enough: Some(1),
        ..onto
    };
    let first = alone(&one);
    let in_order = every.iter().find(|(candidate, ..)| *candidate == 0);
    assert_eq!(first.len(), 1);
    assert_eq!(
        Some(&first[0]),
        in_order,
        "the first release in the task's order"
    );
}

/// Two Pandas 70 cm apart on a torso that turns about the vertical, loaded from a description
/// written here: both hands move on the torso's joint.
fn pandas_on_a_torso() -> Robot {
    let panda = std::fs::read_to_string(common::asset("franka/franka_panda.urdf")).unwrap();
    let links = &panda[panda.find("<link").unwrap()..panda.rfind("</robot>").unwrap()];
    let meshes = common::asset("franka/meshes");
    let arm = |prefix: &str, y: f32| {
        let arm = links
            .replace("name=\"", &format!("name=\"{prefix}"))
            .replace("link=\"", &format!("link=\"{prefix}"))
            .replace("filename=\"meshes", &format!("filename=\"{meshes}"));
        format!(
            "{arm}<joint name=\"{prefix}mount\" type=\"fixed\"><parent link=\"torso\"/>\
             <child link=\"{prefix}base_link\"/><origin xyz=\"0 {y} 0\"/></joint>"
        )
    };
    let description = format!(
        "<robot name=\"pandas_on_a_torso\"><link name=\"world\"/><link name=\"torso\"/>\
         <joint name=\"torso_joint\" type=\"revolute\"><parent link=\"world\"/>\
         <child link=\"torso\"/><axis xyz=\"0 0 1\"/>\
         <limit effort=\"87\" lower=\"-1\" upper=\"1\" velocity=\"1\"/></joint>{}{}</robot>",
        arm("left_", 0.35),
        arm("right_", -0.35)
    );
    let file = format!("batchplan-pandas-on-a-torso-{}.urdf", std::process::id());
    let path: PathBuf = std::env::temp_dir().join(file);
    std::fs::write(&path, description).unwrap();
    let one = common::panda_options();
    let spheres = one.collision_model.as_ref().expect("Panda spheres");
    let locked = ["left_", "right_"].into_iter().flat_map(|prefix| {
        let fingers = one.lock_joints.iter();
        fingers.map(move |(joint, value)| (format!("{prefix}{joint}"), *value))
    });
    let mut options = batchplan::RobotOptions {
        ee_links: vec!["left_ee_link".into(), "right_ee_link".into()],
        lock_joints: locked.collect(),
        collision_model: Some(batchplan::CollisionModel::assemble(&[
            ("left_", spheres),
            ("right_", spheres),
        ])),
        ..Default::default()
    };
    let ready = one.default_q.as_ref().expect("the Panda's ready pose");
    let names = Robot::load(&path, &options).unwrap().joint_names().to_vec();
    let at = |name: &String| match name.rsplit_once("panda_joint") {
        Some((_, k)) => ready[k.parse::<usize>().unwrap() - 1],
        None => 0.0,
    };
    options.default_q = Some(names.iter().map(at).collect());
    let robot = Robot::load(&path, &options).unwrap();
    std::fs::remove_file(&path).unwrap();
    robot
}

/// On a robot whose arms hang from one torso, the place's hand moves on a joint that moves what
/// its object is set on, and that joint stays where the start has it with the supporting arm's:
/// the bar the left hand holds does not move while the right sets the cube on it, on a transit
/// searched around a post that stands between the cube and the bar.
#[test]
fn what_an_object_is_set_on_stays_where_it_is() {
    let robot = pandas_on_a_torso();
    let torso = robot.joint_names().iter().position(|j| j == "torso_joint");
    let torso = torso.expect("the torso's joint");
    let (left, right) = (robot.hand_joints(0), robot.hand_joints(1));
    assert!(
        left.contains(&torso) && right.contains(&torso),
        "both hands move on the torso"
    );
    let o = TaskOptions::default();
    let cube = cube_in_the_right();
    for device in common::devices(&robot) {
        let name = device.name();
        let mut scene = common::tray_scene();
        let post = Obstacle::cuboid(Vec3::new(0.35, -0.22, 0.46), Vec3::new(0.05, 0.01, 0.16));
        scene.obstacles.push(post);
        let worlds = device.upload(&[scene]).unwrap();
        let beyond = common::tray(Vec3::new(0.35, -0.36, 0.5), 0.0);
        let start = each_holding(&device, &worlds, beyond);
        let place = Place {
            onto: Some(bar_in_the_left()),
            ..Place::new(0, start.clone(), cube.clone(), [cube_on_the_bar()])
        };
        let result = batchplan::place(&device, &worlds, &[place], &o).unwrap();
        let plans = result.outcomes[0].as_ref();
        let plans = plans.unwrap_or_else(|failure| panic!("{name}: {failure}"));
        for plan in plans {
            assert!(
                stay(plan, &start, &left),
                "{name}: a joint that moves the left hand moves"
            );
            let transit = plan.transit.as_ref().expect("the place plans its transit");
            let n = robot.dof();
            // The joints' straight way to where the move in begins is through the post, so the
            // transit is a searched one.
            let begins = &transit.control_points[transit.control_points.len() - n..];
            let direct: Vec<f32> = (0..=50)
                .flat_map(|k| {
                    let ends = start.iter().zip(begins);
                    ends.map(move |(a, b)| a + (b - a) * k as f32 / 50.0)
                })
                .collect();
            let holding = vec![Some(&cube); 51];
            let met = device.contacts_holding(&worlds, &[0; 51], &direct, 0.0, &holding);
            assert!(
                met.unwrap().iter().any(|contacts| !contacts.is_empty()),
                "{name}: the straight way is through the post"
            );
        }
    }
}

/// The left hand at `pose` holding `held`, the right arm at the ready pose: the collision-free
/// solution nearest the ready pose.
fn left_holding(device: &Device, worlds: &Worlds, held: &Held, pose: Pose) -> Vec<f32> {
    let robot = device.robot();
    let problem = IkProblem {
        seed: Some(robot.default_q().to_vec()),
        hold: robot.hand_joints(1),
        holding: Some(held.clone()),
        targets: vec![Some(pose), None],
        ..IkProblem::new(0, Pose::IDENTITY)
    };
    let solved = batchplan::solve_ik(device, worlds, &[problem], &common::fixture_ik()).unwrap();
    let nearest = solved.nearest(0, robot.default_q());
    nearest.expect("the left hand reaches").to_vec()
}

/// Whether the box that rides first on what `held` holds meets `post` anywhere along `t`, sampled
/// at 1 kHz: on the shapes, at points 1 cm apart through the box, not on any spheres.
fn rider_meets(robot: &Robot, held: &Held, post: &Obstacle, t: &Trajectory) -> bool {
    let rider = &held.carries[0];
    let Obstacle::Cuboid { half_extents, .. } = *rider else {
        panic!("a box rides, not {rider:?}");
    };
    let steps = (2.0 * half_extents.max_element() / 0.01).ceil() as usize;
    let along = |k: usize| k as f32 / steps as f32 * 2.0 - 1.0;
    let grid = (0..=steps)
        .flat_map(|i| (0..=steps).flat_map(move |j| (0..=steps).map(move |k| (i, j, k))));
    let points: Vec<Vec3> = grid
        .map(|(i, j, k)| half_extents * Vec3::new(along(i), along(j), along(k)))
        .collect();
    let timed = t.sample(1000.0).unwrap();
    let mut states = timed.positions.chunks(timed.dof);
    states.any(|q| {
        let object = robot.ee_pose(q, held.hand).mul_pose(held.grasp.inverse());
        let at = object.mul_pose(rider.pose());
        let inside = |p: &Vec3| post.distance(at.position + at.rotation * *p).value < 0.0;
        points.iter().any(inside)
    })
}

/// The cube the right hand sets on the bar the left holds rides on it: the place's plan gives the
/// bar's hold carrying the cube where the plan sets it down, in the bar's frame. From the plan's
/// end, the left hand carries the bar 10 cm forward past a plate hung 15 mm over it, across the
/// cube's way: planned holding that hold, the path, sampled at 1 kHz, is clear holding it, and the
/// cube's own box keeps clear of the plate, where the same ends planned holding the bar alone take
/// the cube's box through it.
#[test]
fn what_is_set_on_a_held_thing_rides_on_it() {
    let robot = common::two_pandas().unwrap();
    let o = TaskOptions::default();
    let (bar, cube) = (bar_in_the_left(), cube_in_the_right());
    let release = cube_on_the_bar();
    let right = robot.hand_joints(1);
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let above = common::tray(Vec3::new(0.35, -0.3, 0.6), 0.0);
        let start = each_holding(&device, &worlds, above);
        let place = Place {
            onto: Some(bar.clone()),
            ..Place::new(0, start.clone(), cube.clone(), [release])
        };
        let result = batchplan::place(&device, &worlds, &[place], &o).unwrap();
        let plans = result.outcomes[0].as_ref();
        let plans = plans.unwrap_or_else(|failure| panic!("{name}: {failure}"));
        let plan = &plans[0];
        assert!(plan.carried.is_empty(), "{name}: nothing rode on the cube");

        // The cube in the bar's frame, where the plan sets it down.
        let bar_at = robot.ee_pose(&start, 0).mul_pose(bar.grasp.inverse());
        let placed = plan.placed.as_ref().expect("a place sets its object down");
        let wanted = bar_at.inverse().mul_pose(placed.pose());
        let on_its_end = Vec3::new(0.0, -0.10, 0.035);
        assert!(
            wanted.position.distance(on_its_end) < 2e-3,
            "{name}: the cube lies on the bar's far end, at {}",
            wanted.position
        );
        let carrying = plan.held.as_ref().expect("the place gives the bar's hold");
        let bare = Held {
            carries: vec![],
            ..carrying.clone()
        };
        assert_eq!(bare, bar, "{name}: the bar's hold, but for what it carries");
        let [rider] = &carrying.carries[..] else {
            panic!("{name}: the bar carries {:?}", carrying.carries);
        };
        assert_eq!(rider.placed(Pose::IDENTITY), cube.object, "{name}");
        let got = rider.pose();
        assert!(
            got.position.distance(wanted.position) < 1e-5
                && got.rotation.dot(wanted.rotation).abs() > 1.0 - 1e-6,
            "{name}: the cube rides at {got:?}, not {wanted:?}"
        );

        // A plate 15 mm over the bar, across the cube's way as the bar goes 10 cm forward.
        let mut scene = common::tray_scene();
        let plate = Obstacle::cuboid(Vec3::new(0.40, -0.10, 0.49), Vec3::new(0.005, 0.03, 0.01));
        scene.obstacles.push(plate.clone());
        let worlds = device.upload(&[scene]).unwrap();
        let from = plan.end().to_vec();
        let forward = common::tray(BAR_AT + Vec3::X * 0.1, 0.0).mul_pose(bar.grasp);
        let goal = IkProblem {
            seed: Some(from.clone()),
            hold: right.clone(),
            holding: Some(carrying.clone()),
            targets: vec![Some(forward), None],
            ..IkProblem::new(0, Pose::IDENTITY)
        };
        let solved = batchplan::solve_ik(&device, &worlds, &[goal], &common::fixture_ik()).unwrap();
        let goal = solved
            .nearest(0, &from)
            .expect("the bar goes forward")
            .to_vec();
        let planned = |holding: &Held| {
            let problem = PlanProblem {
                hold: right.clone(),
                holding: Some(holding.clone()),
                ..PlanProblem::new(0, from.clone(), goal.clone())
            };
            let result = batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default());
            let result = result.unwrap();
            let path = result.best(0);
            let path = path.unwrap_or_else(|| panic!("{name}: {:?}", result.failures));
            Trajectory::try_new(&robot, path, 1.0).unwrap()
        };
        let riding = planned(carrying);
        assert_eq!(
            colliding(&device, &worlds, carrying, &riding),
            0,
            "{name}: holding the bar and the cube, the path meets something"
        );
        assert!(
            !rider_meets(&robot, carrying, &plate, &riding),
            "{name}: the cube on the bar meets the plate"
        );
        let alone = planned(&bar);
        assert!(
            rider_meets(&robot, carrying, &plate, &alone),
            "{name}: holding the bar alone, the path keeps the cube clear of the plate too"
        );
    }
}

/// The bar the left hand holds, the cube riding on its far end, set down on the block: the plan's
/// `placed` is the bar where it lies, its `carried` the cube where it then stands on it, and the
/// left hand backs off with its fingers clear of the cube, sampled at 1 kHz. A second rider where
/// the left hand's palm stands as it lets go, which the hand may touch while it holds the bar,
/// stops the move away: it stands in that world, the obstacle after the cube.
#[test]
fn what_rides_on_a_thing_set_down_is_set_down_with_it() {
    let robot = common::two_pandas().unwrap();
    let o = TaskOptions::default();
    let cube = cube_in_the_right().object;
    let on_its_end = cube.placed(common::tray(Vec3::new(0.0, -0.10, 0.035), 0.0));
    let carrying = Held {
        carries: vec![on_its_end.clone()],
        ..bar_in_the_left()
    };
    let at_the_palm = Obstacle::cuboid(Vec3::new(0.0, 0.12, 0.07), Vec3::new(0.06, 0.015, 0.005));
    let crowded = Held {
        carries: vec![on_its_end, at_the_palm],
        ..carrying.clone()
    };
    let lying = common::tray(ON_BLOCK, 0.0);
    let above = common::tray(ON_BLOCK + Vec3::Z * 0.15, 0.0).mul_pose(carrying.grasp);
    let fingers = robot.finger_links(0);
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let start = left_holding(&device, &worlds, &crowded, above);
        let set_down = |held: &Held| Place {
            support: vec![0],
            ..Place::new(0, start.clone(), held.clone(), [lying])
        };
        let tasks = [set_down(&carrying), set_down(&crowded)];
        let result = batchplan::place(&device, &worlds, &tasks, &o).unwrap();
        let plans = result.outcomes[0].as_ref();
        let plans = plans.unwrap_or_else(|failure| panic!("{name}: {failure}"));
        for plan in plans {
            let bar = carrying.object.placed(lying);
            assert_eq!(plan.placed.as_ref(), Some(&bar), "{name}: the bar alone");
            let [stands] = &plan.carried[..] else {
                panic!("{name}: carried {:?}", plan.carried);
            };
            let wanted = ON_BLOCK + Vec3::new(0.0, -0.10, 0.035);
            assert!(
                stands.pose().position.distance(wanted) < 1e-5,
                "{name}: the cube stands at {}, not {wanted}",
                stands.pose().position
            );
            assert_eq!(stands.placed(Pose::IDENTITY), cube, "{name}");
            // The move away, the fingers touching the bar and nothing touching the cube.
            let mut there = common::tray_scene();
            there.obstacles.extend([bar, stands.clone()]);
            there.touch = vec![batchplan::Touch::new(1, fingers.iter().copied())];
            let there = device.upload(&[there]).unwrap();
            let timed = plan.depart.sample(1000.0).unwrap();
            let samples = timed.positions.len() / timed.dof;
            let met = device.contacts(&there, &vec![0; samples], &timed.positions, 0.0);
            assert!(
                met.unwrap().iter().all(Vec::is_empty),
                "{name}: the hand backs off into the cube"
            );
        }
        let failure = result.outcomes[1].as_ref();
        let failure = failure.expect_err("the palm meets what rides where it stands");
        assert_eq!(failure.at(TaskStep::Depart), 1, "{name}: {failure}");
        let against = (failure.candidates[0].contacts.iter())
            .any(|contact| matches!(contact, Contact::World { obstacle: 3, .. }));
        assert!(against, "{name}: {failure}");
    }
}

/// An object that carries others is not set on what another hand holds: an input error naming
/// how many it carries.
#[test]
fn what_carries_others_is_not_set_on_a_held_thing() {
    let robot = common::two_pandas().unwrap();
    let device = &common::devices(&robot)[0];
    let worlds = device.upload(&[common::tray_scene()]).unwrap();
    let start = each_holding(
        device,
        &worlds,
        common::tray(Vec3::new(0.35, -0.3, 0.6), 0.0),
    );
    let pebble = Obstacle::sphere(Vec3::new(0.0, 0.0, 0.03), 0.01);
    let carrying = Held {
        carries: vec![pebble],
        ..cube_in_the_right()
    };
    let place = Place {
        onto: Some(bar_in_the_left()),
        ..Place::new(0, start, carrying, [cube_on_the_bar()])
    };
    input(
        batchplan::place(device, &worlds, &[place], &TaskOptions::default()),
        "task 0: an object that carries 1 others (`carries`) cannot be set on what another hand \
         holds (`onto`)",
    );
}

/// The bar's pose where `q` has the left hand at its grasp of it.
fn bar_where(robot: &Robot, q: &[f32]) -> Pose {
    robot
        .ee_pose(q, 0)
        .mul_pose(bar_in_the_left().grasp.inverse())
}

/// A rod the right hand sets on the bar the left holds, a block riding on the bar already, may
/// touch the bar but not the block: where the rod's end would cover the block, the place stops
/// where the hand lets go, against the block, the obstacle after the world's own. Set beside the
/// block, a shorter rod turned 0.6 rad about the vertical is taken, and the plan's hold of the bar
/// carries both, the rod turned so in the bar's frame.
#[test]
fn what_rides_on_a_support_is_not_passed_through() {
    let robot = common::two_pandas().unwrap();
    let o = TaskOptions::default();
    let block = Obstacle::cuboid(Vec3::new(0.0, -0.03, 0.032), Vec3::splat(0.01));
    let bar = Held {
        carries: vec![block.clone()],
        ..bar_in_the_left()
    };
    let rod = |half_length: f32| Held {
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.01, half_length, 0.01)),
        ..cube_in_the_right()
    };
    let end = BAR_AT + Vec3::new(0.0, -0.10, 0.025);
    let (through, beside) = (common::tray(end, 0.0), common::tray(end, 0.6));
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let above = common::tray(Vec3::new(0.35, -0.3, 0.6), 0.0);
        let start = each_holding(&device, &worlds, above);
        let bar_at = bar_where(&robot, &start);
        let block_at = bar_at.mul_pose(block.pose()).position;
        let covering = rod(0.08).object.placed(through).distance(block_at).value;
        assert!(
            covering < 0.0,
            "{name}: the rod's end covers the block: {covering}"
        );
        let onto = |rod: Held, at: Pose| Place {
            onto: Some(bar.clone()),
            ..Place::new(0, start.clone(), rod, [at])
        };
        let tasks = [onto(rod(0.08), through), onto(rod(0.04), beside)];
        let result = batchplan::place(&device, &worlds, &tasks, &o).unwrap();
        let failure = result.outcomes[0].as_ref();
        let failure = failure.expect_err("the rod passes through the block");
        assert_eq!(failure.at(TaskStep::Reach), 1, "{name}: {failure}");
        let against = (failure.candidates[0].contacts.iter())
            .any(|contact| matches!(contact, Contact::World { obstacle: 1, .. }));
        assert!(against, "{name}: {failure}");

        let plans = result.outcomes[1].as_ref();
        let plans = plans.unwrap_or_else(|failure| panic!("{name}: beside the block, {failure}"));
        for plan in plans {
            let carrying = plan.held.as_ref().expect("the bar's hold");
            let [first, second] = &carrying.carries[..] else {
                panic!("{name}: the bar carries {:?}", carrying.carries);
            };
            assert_eq!(first, &block, "{name}: the block first, as it was");
            let wanted = bar_at.inverse().mul_pose(beside);
            let turn = second.pose().rotation;
            assert!(
                turn.dot(wanted.rotation).abs() > 1.0 - 1e-6
                    && turn.angle_between(Quat::IDENTITY) > 0.5,
                "{name}: the rod rides turned {turn}, not {}",
                wanted.rotation
            );
        }
    }
}

/// A capsule riding on the bar lies along the bar's x axis, turned so in the bar's frame. Held,
/// its spheres lie along that axis: a ball at its end meets what the left hand holds, and one
/// where the capsule would stand unturned does not. Set down on the block with the bar, it stands
/// turned so (`TaskPlan::carried`).
#[test]
fn a_rider_turned_in_its_carriers_frame_rides_turned() {
    let robot = common::two_pandas().unwrap();
    let o = TaskOptions::default();
    let lying = Pose::new(
        Vec3::new(0.0, -0.10, 0.03),
        Quat::from_rotation_y(FRAC_PI_2),
    );
    let capsule = Obstacle::capsule(Vec3::ZERO, 0.01, 0.05).placed(lying);
    let carrying = Held {
        carries: vec![capsule],
        ..bar_in_the_left()
    };
    let on_block = common::tray(ON_BLOCK, 0.0);
    let above = common::tray(ON_BLOCK + Vec3::Z * 0.15, 0.0).mul_pose(carrying.grasp);
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[common::tray_scene()]).unwrap();
        let start = each_holding(
            &device,
            &worlds,
            common::tray(Vec3::new(0.35, -0.3, 0.6), 0.0),
        );
        let bar_at = bar_where(&robot, &start);
        // A ball by the capsule's end as it lies, and one where it would stand unturned.
        let balls = [Vec3::new(0.055, -0.10, 0.03), Vec3::new(0.0, -0.10, 0.085)];
        let probes: Vec<World> = (balls.iter())
            .map(|&at| {
                let mut world = common::tray_scene();
                world.obstacles.push(Obstacle::sphere(
                    bar_at.mul_pose(Pose::new(at, Quat::IDENTITY)).position,
                    0.008,
                ));
                world
            })
            .collect();
        let probes = device.upload(&probes).unwrap();
        let q: Vec<f32> = [start.clone(), start.clone()].concat();
        let holding = [Some(&carrying); 2];
        let met = device
            .contacts_holding(&probes, &[0, 1], &q, 0.0, &holding)
            .unwrap();
        let meets = |contacts: &[Contact]| {
            (contacts.iter()).any(|contact| matches!(contact, Contact::World { obstacle: 1, .. }))
        };
        assert!(
            meets(&met[0]),
            "{name}: the capsule's end is clear of the ball: {:?}",
            met[0]
        );
        assert!(
            !meets(&met[1]),
            "{name}: the capsule stands unturned: {:?}",
            met[1]
        );

        let start = left_holding(&device, &worlds, &carrying, above);
        let set_down = Place {
            support: vec![0],
            ..Place::new(0, start, carrying.clone(), [on_block])
        };
        let result = batchplan::place(&device, &worlds, &[set_down], &o).unwrap();
        let plans = result.outcomes[0].as_ref();
        let plans = plans.unwrap_or_else(|failure| panic!("{name}: {failure}"));
        let wanted = on_block.mul_pose(lying);
        for plan in plans {
            let [stands] = &plan.carried[..] else {
                panic!("{name}: carried {:?}", plan.carried);
            };
            let at = stands.pose();
            assert!(
                at.position.distance(wanted.position) < 1e-5
                    && at.rotation.dot(wanted.rotation).abs() > 1.0 - 1e-6,
                "{name}: the capsule stands at {at:?}, not {wanted:?}"
            );
        }
    }
}
