//! Behaviour every `Device` must share, checked on each available device (the GPU is skipped when
//! absent unless `BATCHPLAN_REQUIRE_GPU=1`).

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::time::Duration;

use batchplan::rng::Rng;
use batchplan::{
    CollisionWeights, Device, Error, IkOptions, IkProblem, JointState, Obstacle, PlanOptions,
    PlanProblem, Pose, Robot, RobotOptions, RrtOptions, SdfGrid, ShortcutOptions, StartMotion,
    Trajectory, World,
};
use glam::{Quat, Vec3};

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

/// A rotation a thousandth off a unit quaternion is accepted, and turns its obstacle as the unit
/// one does: nothing is stretched. Each kind of obstacle that turns, long enough for a stretch of
/// 0.4% to move its end by millimetres, measures the same from the same points and gives every
/// configuration the same clearance and cost, on every device.
#[test]
fn an_obstacle_turned_by_a_rotation_not_quite_unit_is_not_stretched() {
    let robot = panda();
    let n = robot.dof();
    let turn = Quat::from_rotation_z(3.0) * Quat::from_rotation_x(0.4);
    let grid = {
        let dims = [41u32, 3, 3];
        let distances: Vec<f32> = (0..41 * 9).map(|k| 0.01 * ((k % 7) as f32 - 3.0)).collect();
        let origin = Vec3::new(-0.6, -0.03, -0.03);
        SdfGrid::try_new(dims, 0.03, origin, &distances).unwrap()
    };
    let at = Vec3::new(0.5, 0.1, 0.4);
    let turned = |rotation: Quat| {
        let long = Vec3::new(0.6, 0.02, 0.02);
        World::new(vec![
            Obstacle::cuboid(at, long).placed(Pose::new(at, rotation)),
            Obstacle::cylinder(Vec3::ZERO, 0.02, 0.6)
                .placed(Pose::new(at + Vec3::Y * 0.3, rotation)),
            Obstacle::capsule(Vec3::ZERO, 0.02, 0.6)
                .placed(Pose::new(at - Vec3::Y * 0.3, rotation)),
            Obstacle::from_grid(grid.clone()).placed(Pose::new(at + Vec3::Z * 0.3, rotation)),
        ])
    };
    for norm in [0.9991f32, 1.0009] {
        let off = Quat::from_xyzw(turn.x * norm, turn.y * norm, turn.z * norm, turn.w * norm);
        let (unit, scaled) = (turned(turn), turned(off));
        let mut rng = Rng::new(9);
        for (a, b) in unit.obstacles.iter().zip(&scaled.obstacles) {
            for _ in 0..200 {
                let p = at
                    + Vec3::new(
                        rng.range(-1.0, 1.0),
                        rng.range(-1.0, 1.0),
                        rng.range(-0.5, 0.8),
                    );
                let (da, db) = (a.distance(p).value, b.distance(p).value);
                assert!((da - db).abs() < 2e-5, "{da} and {db} from {p}");
            }
        }
        let q: Vec<f32> = (0..300 * n)
            .map(|k| rng.range(robot.lower()[k % n], robot.upper()[k % n]))
            .collect();
        let weights = CollisionWeights {
            margin: 0.05,
            ..CollisionWeights::default()
        };
        for device in devices(&robot) {
            let name = device.name();
            let worlds = device.upload(&[unit.clone(), scaled.clone()]).unwrap();
            let of = |world: u32| {
                device
                    .evaluate(&worlds, &vec![world; 300], &q, &weights)
                    .unwrap()
            };
            let (as_unit, as_scaled) = (of(0), of(1));
            let mut near = 0;
            for i in 0..300 {
                let (a, b) = (as_unit.world_clearance[i], as_scaled.world_clearance[i]);
                assert!((a - b).abs() < 2e-5, "{name}: clearances {a} and {b}");
                let (a, b) = (as_unit.cost[i], as_scaled.cost[i]);
                assert!(
                    (a - b).abs() <= 1e-4 * a.abs().max(1.0),
                    "{name}: costs {a} and {b}"
                );
                near += usize::from(as_unit.world_clearance[i] < 0.05);
            }
            assert!(
                near > 30,
                "{name}: {near} of 300 within the margin of an obstacle"
            );
        }
    }
}

/// A device made for a relocked robot with `with_robot` checks collisions as one made for a robot
/// loaded with those values, against worlds uploaded before.
#[test]
fn a_relocked_robot_collides_where_one_loaded_there_does() {
    let panda = |joint1: f32| {
        let mut o = common::panda_options();
        o.lock_joints.insert("panda_joint1".into(), joint1);
        o.default_q = None;
        Robot::load(common::asset("franka/franka_panda.urdf"), &o).unwrap()
    };
    let robot = panda(0.0);
    let relocked = robot
        .relock(&HashMap::from([("panda_joint1".into(), 0.9)]))
        .unwrap();
    let loaded = panda(0.9);
    // A box beside the arm turned 0.9 rad, clear of it unturned.
    let q = [0.3, 0.0, -2.0, 0.0, 1.8, 0.8];
    let hand = loaded.ee_pose(&q, 0).position;
    let world = World::new(vec![Obstacle::cuboid(hand, Vec3::splat(0.05))]);
    let w = CollisionWeights {
        world: 10.0,
        self_collision: 10.0,
        margin: 0.02,
        self_margin: 0.01,
    };
    for device in devices(&robot) {
        let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
        let unturned = device.evaluate(&worlds, &[0], &q, &w).unwrap();
        assert!(unturned.collision_free(0), "{device:?}: clear before");
        let turned = device.with_robot(&relocked).unwrap();
        let fresh = Device::cpu(&loaded).unwrap();
        let fresh_worlds = fresh.upload(std::slice::from_ref(&world)).unwrap();
        let (a, b) = (
            turned.evaluate(&worlds, &[0], &q, &w).unwrap(),
            fresh.evaluate(&fresh_worlds, &[0], &q, &w).unwrap(),
        );
        assert!(!a.collision_free(0), "{device:?}: the box is in the way");
        let close = |x: f32, y: f32| (x - y).abs() <= 1e-4 * (1.0 + y.abs());
        assert!(
            close(a.world_clearance[0], b.world_clearance[0])
                && close(a.self_clearance[0], b.self_clearance[0])
                && close(a.cost[0], b.cost[0]),
            "{device:?}: {a:?} against {b:?}"
        );
        for (ga, gb) in a.gradient.iter().zip(&b.gradient) {
            assert!(
                close(*ga, *gb),
                "{device:?}: {:?} against {:?}",
                a.gradient,
                b.gradient
            );
        }
    }
}

#[test]
fn malformed_batches_are_errors_on_every_device() {
    let robot = panda();
    let q = robot.default_q().to_vec();
    let scene = vec![World::new(vec![Obstacle::sphere(
        Vec3::new(2.0, 0.0, 0.0),
        0.1,
    )])];
    let none = CollisionWeights::NONE;
    for d in devices(&robot) {
        let name = d.name();
        let worlds = d.upload(&scene).unwrap();
        let elsewhere = Device::cpu(&robot).unwrap().upload(&scene).unwrap();
        let err = d.evaluate(&elsewhere, &[0], &q, &none).unwrap_err();
        assert!(
            matches!(err, Error::Input(_)),
            "{name}: worlds uploaded to another device give {err:?}"
        );
        assert!(
            d.evaluate(&worlds, &[0, 0], &q, &none).is_err(),
            "{name}: two items, one configuration"
        );
        assert!(
            d.evaluate(&worlds, &[0], &q[..6], &none).is_err(),
            "{name}: configuration missing a joint"
        );
        assert!(
            d.evaluate(&worlds, &[1], &q, &none).is_err(),
            "{name}: world index past the end"
        );
        let none_kept = IkOptions {
            kept: 0,
            ..IkOptions::default()
        };
        let reachable = [IkProblem::new(0, robot.ee_pose(&q, 0))];
        assert!(
            matches!(
                batchplan::solve_ik(&d, &worlds, &reachable, &none_kept),
                Err(Error::Input(_))
            ),
            "{name}: IK that keeps no seed past its screening"
        );
        let mut unknown = q.clone();
        unknown[0] = f32::NAN;
        assert!(
            matches!(
                d.contacts(&worlds, &[0], &unknown, 0.0),
                Err(Error::Input(_))
            ),
            "{name}: contacts of a configuration that is not finite"
        );
        assert!(
            matches!(
                d.evaluate(&worlds, &[0], &unknown, &none),
                Err(Error::Input(_))
            ),
            "{name}: a configuration that is not finite"
        );
        let target = robot.ee_pose(&q, 0);
        assert!(
            batchplan::solve_ik(
                &d,
                &worlds,
                &[IkProblem::new(1, target)],
                &IkOptions::default()
            )
            .is_err(),
            "{name}: IK"
        );
        let problem = PlanProblem::new(1, q.clone(), q.clone());
        assert!(
            batchplan::plan(&d, &worlds, &[problem], &PlanOptions::default()).is_err(),
            "{name}: plan"
        );
        // Inputs no device could check consistently are refused before any work.
        let input = |r: Result<_, Error>| matches!(r, Err(Error::Input(_)));
        let mut beyond = q.clone();
        beyond[3] = robot.upper()[3] + 0.5;
        let past_limits = PlanProblem::new(0, q.clone(), beyond);
        assert!(
            input(
                batchplan::plan(&d, &worlds, &[past_limits], &PlanOptions::default()).map(|_| ())
            ),
            "{name}: goal past limits"
        );
        let skewed = Pose {
            rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 2.0),
            ..target
        };
        let ik = batchplan::solve_ik(
            &d,
            &worlds,
            &[IkProblem::new(0, skewed)],
            &IkOptions::default(),
        );
        assert!(input(ik.map(|_| ())), "{name}: a non-unit target rotation");
        let rotated = |rotation| Obstacle::Cuboid {
            center: Vec3::ZERO,
            half_extents: Vec3::ONE,
            rotation,
        };
        for bad in [
            Obstacle::sphere(Vec3::new(f32::NAN, 0.0, 0.0), 0.1),
            Obstacle::sphere(Vec3::ZERO, -0.1),
            rotated(Quat::from_xyzw(0.0, 0.0, 0.0, 0.5)),
        ] {
            let err = d.upload(&[World::new(vec![bad.clone()])]).map(|_| ());
            assert!(input(err), "{name}: {bad:?} uploaded");
        }
    }
}

#[test]
fn clearances_are_exact_up_to_the_margins() {
    let robot = panda();
    let (n, items) = (robot.dof(), 4000);
    let mut rng = Rng::new(5);
    let q: Vec<f32> = (0..items * n)
        .map(|i| rng.range(robot.lower()[i % n], robot.upper()[i % n]))
        .collect();
    let item_world = vec![0; items];
    let scene = [common::tabletop(&mut Rng::new(4))];
    let margin = 0.05;
    let near = CollisionWeights {
        world: 1.0,
        self_collision: 1.0,
        margin,
        self_margin: margin,
    };
    // Margins wider than the robot leave nothing far enough apart to skip.
    let everything = CollisionWeights {
        margin: 10.0,
        self_margin: 10.0,
        ..CollisionWeights::NONE
    };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let exact = d.evaluate(&worlds, &item_world, &q, &everything).unwrap();
        let gated = d.evaluate(&worlds, &item_world, &q, &near).unwrap();
        let kinds = [
            ("world", exact.world_clearance, gated.world_clearance),
            ("self", exact.self_clearance, gated.self_clearance),
        ];
        for (kind, exact, gated) in kinds {
            let (mut close, mut bounded) = (0, 0);
            for (&e, &g) in exact.iter().zip(&gated) {
                if e <= margin {
                    assert_eq!(g, e, "{}: a close {kind} pair was skipped", d.name());
                    close += 1;
                } else {
                    assert!(
                        g > margin && g <= e + 1e-5,
                        "{}: {kind} {g} does not bound {e}",
                        d.name()
                    );
                    bounded += usize::from(g < e);
                }
            }
            assert!(
                close > 100 && bounded > 20,
                "{}: {kind}: {close} close, {bounded} bounded",
                d.name()
            );
        }
    }
}

#[test]
fn contacts_name_what_each_clearance_measures() {
    let robot = panda();
    let (n, items) = (robot.dof(), 2000);
    let mut rng = Rng::new(6);
    let q: Vec<f32> = (0..items * n)
        .map(|i| rng.range(robot.lower()[i % n], robot.upper()[i % n]))
        .collect();
    let item_world = vec![0; items];
    let scene = common::tabletop(&mut Rng::new(4));
    // The table may touch every link: it names no contact.
    let mut touching = scene.clone();
    let links: Vec<String> = robot.collision_model().spheres.keys().cloned().collect();
    touching.touch.push(batchplan::Touch::new(0, links));
    // A table the base stands in, which no joint moves it out of: not a contact either.
    let mounted = World::new(vec![Obstacle::cuboid(
        Vec3::new(0.0, 0.0, -0.47),
        Vec3::new(0.3, 0.3, 0.5),
    )]);
    let scenes = [scene, touching, mounted];
    let everything = CollisionWeights {
        margin: 10.0,
        self_margin: 10.0,
        ..CollisionWeights::NONE
    };
    let within = 0.05;
    let cpu = Device::cpu(&robot).unwrap();
    let cpu_worlds = cpu.upload(&scenes).unwrap();
    let mut listed = None;
    for d in devices(&robot) {
        let worlds = d.upload(&scenes).unwrap();
        let contacts = d.contacts(&worlds, &item_world, &q, within).unwrap();
        let on_mount = vec![2; items];
        let mounted = d.contacts(&worlds, &on_mount, &q, within).unwrap();
        let exact = cpu
            .evaluate(&cpu_worlds, &item_world, &q, &everything)
            .unwrap();
        let exact_mounted = cpu
            .evaluate(&cpu_worlds, &on_mount, &q, &everything)
            .unwrap();
        for (found, exact) in mounted.iter().zip(&exact_mounted.world_clearance) {
            let nearest = found
                .iter()
                .find(|c| matches!(c, batchplan::Contact::World { .. }))
                .map(batchplan::Contact::distance);
            assert_eq!(nearest, (*exact < within).then_some(*exact), "{}", d.name());
        }
        let (mut world, mut own) = (0, 0);
        for (item, found) in contacts.iter().enumerate() {
            assert!(
                found.windows(2).all(|w| w[0].distance() <= w[1].distance()),
                "{}: nearest first",
                d.name()
            );
            let nearest = |self_collision: bool| {
                found
                    .iter()
                    .filter(|c| {
                        matches!(c, batchplan::Contact::SelfCollision { .. }) == self_collision
                    })
                    .map(batchplan::Contact::distance)
                    .next()
            };
            let expected = |clearance: f32| (clearance < within).then_some(clearance);
            assert_eq!(
                nearest(false),
                expected(exact.world_clearance[item]),
                "{}: item {item}",
                d.name()
            );
            assert_eq!(
                nearest(true),
                expected(exact.self_clearance[item]),
                "{}: item {item}",
                d.name()
            );
            world += usize::from(exact.world_clearance[item] < 0.0);
            own += usize::from(exact.self_clearance[item] < 0.0);
        }
        assert!(
            world > 50 && own > 50,
            "{world} in the world, {own} in themselves"
        );
        // Allowed to touch the table, the links name only the boxes.
        let touching = d.contacts(&worlds, &vec![1; items], &q, within).unwrap();
        let on_table = touching
            .iter()
            .flatten()
            .any(|c| matches!(c, batchplan::Contact::World { obstacle: 0, .. }));
        assert!(!on_table, "{}: a touch is a contact", d.name());
        let named_table = contacts
            .iter()
            .flatten()
            .any(|c| matches!(c, batchplan::Contact::World { obstacle: 0, .. }));
        assert!(named_table, "the table is near");
        match &listed {
            None => listed = Some(contacts),
            Some(first) => assert_eq!(*first, contacts, "{} lists other contacts", d.name()),
        }
    }
}

#[test]
fn warming_up_builds_what_first_calls_would_and_changes_no_result() {
    let robot = panda();
    let (scene, problems) = common::blocked_reaches(&robot);
    let problems = &problems[..4];
    let o = PlanOptions {
        seeds: 2,
        ..PlanOptions::default()
    };
    let goals: Vec<IkProblem> = problems
        .iter()
        .map(|p| IkProblem::new(p.world, robot.ee_pose(p.goal.joints().unwrap(), 0)))
        .collect();
    let run = |d: &Device| {
        let worlds = d.upload(&scene).unwrap();
        let plans = batchplan::plan(d, &worlds, problems, &o).unwrap();
        let ik = batchplan::solve_ik(d, &worlds, &goals, &IkOptions::default()).unwrap();
        (plans.paths, plans.valid, ik.q)
    };
    let mut fresh = devices(&robot);
    for (cold, warm) in fresh.drain(..).zip(devices(&robot)) {
        warm.warmup().unwrap();
        warm.with_robot(&robot).unwrap().warmup().unwrap();
        assert_eq!(run(&cold), run(&warm), "{}", warm.name());
    }
}

#[test]
fn ik_reports_the_clearances_evaluate_finds() {
    let robot = panda();
    let scene = [common::tabletop(&mut Rng::new(4))];
    let mut rng = Rng::new(9);
    let problems: Vec<IkProblem> = (0..48)
        .map(|_| IkProblem::new(0, common::grasp_target(&scene[0], &mut rng)))
        .collect();
    let unpushed = IkOptions {
        collision_step: 0.0,
        ..IkOptions::default()
    };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        // With the push and without it, which finds the clearances once at the end; a few
        // problems after a call that pushed, whose buffers a GPU uses again, and many.
        for (o, count) in [
            (IkOptions::default(), 16),
            (unpushed, 16),
            (unpushed, 48),
            (IkOptions::default(), 48),
        ] {
            let ik = batchplan::solve_ik(&d, &worlds, &problems[..count], &o).unwrap();
            let items = ik.q.len() / robot.dof();
            let e = d
                .evaluate(&worlds, &vec![0; items], &ik.q, &CollisionWeights::NONE)
                .unwrap();
            let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-6);
            let call = format!("{}, {count} problems, step {}", d.name(), o.collision_step);
            assert!(
                close(&ik.world_clearance, &e.world_clearance),
                "{call}: world clearance"
            );
            assert!(
                close(&ik.self_clearance, &e.self_clearance),
                "{call}: self clearance"
            );
            assert!(
                !close(&ik.world_clearance, &ik.self_clearance),
                "world and self clearance should differ"
            );
        }
    }
}

#[test]
fn evaluation_does_not_depend_on_the_batch() {
    // Small batches split each configuration's collision work across several GPU invocations,
    // large ones give it one; the results must agree.
    let robot = panda();
    let n = robot.dof();
    let mut rng = Rng::new(13);
    let q: Vec<f32> = (0..40_000 * n)
        .map(|i| rng.range(robot.lower()[i % n], robot.upper()[i % n]))
        .collect();
    let scene = [common::tabletop(&mut Rng::new(4))];
    let w = CollisionWeights {
        world: 1.0,
        self_collision: 1.0,
        margin: 0.05,
        self_margin: 0.02,
    };
    let few = 64;
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let alone = d
            .evaluate(&worlds, &vec![0; few], &q[..few * n], &w)
            .unwrap();
        let batched = d.evaluate(&worlds, &vec![0; q.len() / n], &q, &w).unwrap();
        assert!(
            alone.cost.iter().filter(|&&c| c > 0.0).count() > 10,
            "too few configurations carry cost"
        );
        let close = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-4 * x.abs().max(1.0))
        };
        assert!(
            close(&alone.world_clearance, &batched.world_clearance[..few]),
            "{}: world clearance",
            d.name()
        );
        assert!(
            close(&alone.self_clearance, &batched.self_clearance[..few]),
            "{}: self clearance",
            d.name()
        );
        assert!(
            close(&alone.cost, &batched.cost[..few]),
            "{}: cost",
            d.name()
        );
        assert!(
            close(&alone.gradient, &batched.gradient[..few * n]),
            "{}: gradient",
            d.name()
        );
    }
}

/// An IK step moves a seed by more than a few of an `f32`'s steps or not at all, on either device:
/// one that would move no joint by more than 1e-6 rad is not taken, so a seed that comes to rest
/// stays exactly there (and the CPU skips its remaining steps).
#[test]
fn ik_steps_move_a_seed_or_leave_it_at_rest() {
    let robot = panda();
    let n = robot.dof();
    let scene = [common::tabletop(&mut Rng::new(4))];
    let mut rng = Rng::new(9);
    let problems: Vec<IkProblem> = (0..8)
        .map(|_| IkProblem::new(0, common::grasp_target(&scene[0], &mut rng)))
        .collect();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        // Every seed takes every step: none is left behind after a screening.
        let after = |iterations: u32| {
            let o = IkOptions {
                iterations,
                screening: 0,
                ..IkOptions::default()
            };
            batchplan::solve_ik(&d, &worlds, &problems, &o).unwrap().q
        };
        let (mut rested, mut moved) = (0, 0);
        for steps in [10, 30, 60] {
            let (before, one_more) = (after(steps), after(steps + 1));
            for (seed, (a, b)) in before.chunks(n).zip(one_more.chunks(n)).enumerate() {
                let step = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(
                    step == 0.0 || step > 1e-6,
                    "{}: step {} moves seed {seed} by {step} rad",
                    d.name(),
                    steps + 1
                );
                rested += usize::from(step == 0.0);
                moved += usize::from(step > 0.0);
            }
        }
        // A seed the CPU stopped stepping is one a further step leaves exactly where it is: solved
        // again from there, it comes back bit for bit. (On the CPU alone: a GPU batch of another
        // size can take another lane variant, which rounds differently.)
        if d.name().starts_with("cpu") {
            let (ended, one_more) = (after(60), after(61));
            let mut again = 0;
            for (seed, (a, b)) in ended.chunks(n).zip(one_more.chunks(n)).enumerate() {
                if a != b {
                    continue;
                }
                let from = IkProblem {
                    seed: Some(a.to_vec()),
                    ..problems[seed / IkOptions::default().seeds].clone()
                };
                let one = IkOptions {
                    seeds: 1,
                    iterations: 1,
                    ..IkOptions::default()
                };
                let stepped = batchplan::solve_ik(&d, &worlds, &[from], &one).unwrap();
                assert_eq!(
                    stepped.q, a,
                    "a step moves seed {seed}, which the CPU stopped"
                );
                again += 1;
            }
            assert!(again > 100, "only {again} seeds at rest after 60 steps");
        }
        assert!(
            rested > 100 && moved > 100,
            "{}: {rested} steps left a seed at rest, {moved} moved one",
            d.name()
        );
    }
}

/// Past the screening only each problem's nearest seeds go on: the others stay as the screening
/// left them, those that go on were the nearest then, and what every seed taking every step
/// solves is still solved.
#[test]
fn ik_refines_the_seeds_nearest_after_the_screening() {
    refines_the_nearest(IkOptions::default());
}

/// Without the push away from collisions the seeds that go on are the nearest by their errors
/// alone, in collision or not.
#[test]
fn ik_without_the_push_refines_the_seeds_nearest_by_error_alone() {
    refines_the_nearest(IkOptions {
        collision_step: 0.0,
        ..IkOptions::default()
    });
}

/// A small call's buffers are the last call's on a GPU: IK without the push solves alike after a
/// call that pushed, whose clearances are still in them, and on a device that has solved nothing.
#[test]
fn ik_without_the_push_reads_nothing_of_the_call_before() {
    let robot = panda();
    let scene = [common::tabletop(&mut Rng::new(4))];
    let mut rng = Rng::new(9);
    let problems: Vec<IkProblem> = (0..16)
        .map(|_| IkProblem::new(0, common::grasp_target(&scene[0], &mut rng)))
        .collect();
    let unpushed = IkOptions {
        collision_step: 0.0,
        ..IkOptions::default()
    };
    for (used, fresh) in devices(&robot).into_iter().zip(devices(&robot)) {
        let solve = |d: &Device, o: &IkOptions| {
            let worlds = d.upload(&scene).unwrap();
            batchplan::solve_ik(d, &worlds, &problems, o).unwrap()
        };
        let pushed = solve(&used, &IkOptions::default());
        // Collisions the choice would count, were it to read them.
        let colliding = (0..pushed.success.len())
            .filter(|&i| pushed.world_clearance[i] < 0.0 || pushed.self_clearance[i] < 0.0);
        assert!(colliding.count() > problems.len(), "{}", used.name());
        let (after, alone) = (solve(&used, &unpushed), solve(&fresh, &unpushed));
        assert_eq!(after.q, alone.q, "{}", used.name());
    }
}

/// Solves grasps on cluttered tabletops with `o` on every device and checks which seeds went on
/// past the screening.
fn refines_the_nearest(o: IkOptions) {
    let robot = panda();
    let n = robot.dof();
    let mut rng = Rng::new(4);
    let scene: Vec<World> = (0..48).map(|_| common::tabletop(&mut rng)).collect();
    let problems: Vec<IkProblem> = scene
        .iter()
        .enumerate()
        .map(|(i, world)| IkProblem::new(i as u32, common::grasp_target(world, &mut rng)))
        .collect();
    assert!(o.screening > 0 && o.screening < o.iterations && o.kept < o.seeds);
    let pushes = o.collision_step > 0.0;
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let solve = |o: &IkOptions| batchplan::solve_ik(&d, &worlds, &problems, o).unwrap();
        let every = |iterations: u32| IkOptions {
            iterations,
            screening: 0,
            ..o
        };
        let (screened, staged, full) = (
            solve(&every(o.screening)),
            solve(&o),
            solve(&every(o.iterations)),
        );
        let clear =
            |i: usize| screened.world_clearance[i] >= 0.0 && screened.self_clearance[i] >= 0.0;
        let error =
            |i: usize| screened.position_error[i] + o.rotation_weight * screened.rotation_error[i];
        // How near a seed that had not arrived was after the screening, as the choice ranks them:
        // where IK pushes, any collision-free seed before any that is not.
        let cost = |i: usize| {
            if pushes && !clear(i) {
                1e6 + error(i)
            } else {
                error(i)
            }
        };
        // Arrived, as the choice sees it: collisions count where IK pushes away from them.
        let arrived = |i: usize| {
            screened.position_error[i] < o.position_tolerance
                && screened.rotation_error[i] < o.rotation_tolerance
                && (clear(i) || !pushes)
        };
        let (mut went_on, mut colliding) = (0, 0);
        for p in 0..problems.len() {
            let seeds = p * o.seeds..(p + 1) * o.seeds;
            let moved = |i: &usize| staged.q[i * n..(i + 1) * n] != screened.q[i * n..(i + 1) * n];
            let refined: Vec<usize> = seeds.clone().filter(moved).collect();
            assert!(
                refined.len() <= o.kept,
                "{}: problem {p} refined {refined:?}",
                d.name()
            );
            assert!(
                refined.iter().all(|&i| !screened.success[i]),
                "{}: problem {p} sends on a seed that had arrived",
                d.name()
            );
            let farthest = refined.iter().map(|&i| cost(i)).fold(0.0, f32::max);
            let left = seeds.filter(|i| !refined.contains(i) && !arrived(*i));
            // A seed stuck short of its target does not move when sent on either: only seeds left
            // behind while nearer than one that went on would be a wrong choice.
            let nearer = left.filter(|&i| cost(i) < farthest).count();
            assert!(
                nearer + refined.len() <= o.kept,
                "{}: problem {p} left {nearer} seeds nearer than one it refined",
                d.name()
            );
            went_on += refined.len();
            colliding += refined.iter().filter(|&&i| !clear(i)).count();
        }
        assert!(
            went_on > problems.len(),
            "{}: only {went_on} seeds went on",
            d.name()
        );
        // Without the push, seeds in collision go on too: counting collisions would send others.
        assert!(
            pushes || colliding > 4,
            "{}: only {colliding} of the seeds that went on were in collision",
            d.name()
        );
        // The seeds left behind rarely cost a problem its solution.
        let (staged, full) = (staged.solved().count(), full.solved().count());
        assert!(
            staged + 1 >= full && staged > problems.len() / 2,
            "{}: {staged} problems solved, {full} when every seed takes every step",
            d.name()
        );
    }
}

/// A problem's IK solves alike alone and in a batch: each problem draws its random seeds alike.
#[test]
fn ik_does_not_depend_on_the_batch() {
    let robot = panda();
    let n = robot.dof();
    let scene = [common::tabletop(&mut Rng::new(4))];
    let mut rng = Rng::new(9);
    let problems: Vec<IkProblem> = (0..8)
        .map(|_| IkProblem::new(0, common::grasp_target(&scene[0], &mut rng)))
        .collect();
    let o = IkOptions::default();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let batch = batchplan::solve_ik(&d, &worlds, &problems, &o).unwrap();
        for (i, p) in problems.iter().enumerate() {
            let alone = batchplan::solve_ik(&d, &worlds, std::slice::from_ref(p), &o).unwrap();
            let items = i * o.seeds..(i + 1) * o.seeds;
            assert_eq!(
                alone.q,
                batch.q[items.start * n..items.end * n],
                "{}: problem {i}",
                d.name()
            );
            assert_eq!(alone.success, batch.success[items], "{}", d.name());
        }
        // Unmoved, each seed is where it starts: problems that differ start from random seeds of
        // their own, and a problem given twice from the same ones.
        let unmoved = IkOptions { iterations: 0, ..o };
        let twice = [
            problems[0].clone(),
            problems[1].clone(),
            problems[0].clone(),
        ];
        let starts = batchplan::solve_ik(&d, &worlds, &twice, &unmoved).unwrap();
        let random = |problem: usize| starts.solution(problem * o.seeds + 1).to_vec();
        assert_ne!(random(0), random(1), "{}", d.name());
        assert_eq!(random(0), random(2), "{}", d.name());
    }
}

/// A problem's plan is the same wherever it is in its batch, optimization seeds included: each
/// problem bends its seeds through random via points alike.
#[test]
fn plans_do_not_depend_on_the_batch() {
    let robot = panda();
    let (scene, problems) = common::blocked_reaches(&robot);
    let o = PlanOptions {
        seeds: 6,
        ..Default::default()
    };
    let reversed: Vec<PlanProblem> = problems.iter().rev().cloned().collect();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let a = batchplan::plan(&d, &worlds, &problems, &o).unwrap();
        let b = batchplan::plan(&d, &worlds, &reversed, &o).unwrap();
        let seeds = a.seeds;
        // The last seed bends through a random via point; problems searched straight have none.
        let bent = (0..problems.len())
            .filter(|&i| !a.paths[i * seeds + seeds - 1].is_empty())
            .count();
        assert!(bent > 2, "{}: {bent} problems optimized", d.name());
        for i in 0..problems.len() {
            let j = problems.len() - 1 - i;
            for s in 0..seeds {
                let (x, y) = (i * seeds + s, j * seeds + s);
                assert_eq!(
                    a.paths[x],
                    b.paths[y],
                    "{}: problem {i}, seed {s}",
                    d.name()
                );
                assert_eq!((a.valid[x], a.length[x]), (b.valid[y], b.length[y]));
            }
        }
    }
}

#[test]
fn obstacle_free_worlds_work_on_every_device() {
    let robot = panda();
    let start = robot.default_q().to_vec();
    let mut elsewhere = start.clone();
    elsewhere[0] += 0.8;
    elsewhere[3] += 0.4;
    let target = robot.ee_pose(&elsewhere, 0);
    for d in devices(&robot) {
        let name = d.name();
        let worlds = d.upload(&[World::default()]).unwrap();
        let e = d
            .evaluate(&worlds, &[0], &start, &CollisionWeights::NONE)
            .unwrap();
        assert!(
            e.world_clearance[0] > 1e29,
            "{name}: an empty world has nothing to hit"
        );
        let ik = batchplan::solve_ik(
            &d,
            &worlds,
            &[IkProblem::new(0, target)],
            &IkOptions::default(),
        )
        .unwrap();
        let goal = ik
            .best(0)
            .unwrap_or_else(|| panic!("{name}: reachable target unsolved"))
            .to_vec();
        let result = batchplan::plan(
            &d,
            &worlds,
            &[PlanProblem::new(0, start.clone(), goal)],
            &PlanOptions::default(),
        )
        .unwrap();
        assert!(
            result.best(0).is_some(),
            "{name}: no valid plan in an empty world"
        );
    }
}

#[test]
fn results_keep_the_worlds_of_their_problems() {
    // Two targets in world 1 and one in world 0: problem index != world index.
    let robot = panda();
    let mut rng = Rng::new(21);
    let scene: Vec<World> = (0..2).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = [1u32, 1, 0]
        .iter()
        .map(|&w| IkProblem::new(w, common::grasp_target(&scene[w as usize], &mut rng)))
        .collect();
    let cpu = Device::cpu(&robot).unwrap();
    let on_cpu = cpu.upload(&scene).unwrap();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let ik = batchplan::solve_ik(&d, &worlds, &goals, &IkOptions::default()).unwrap();
        assert_eq!(ik.problems.len(), goals.len());
        let problems: Vec<PlanProblem> = ik
            .solved()
            .map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution))
            .collect();
        assert!(
            problems.len() >= 2,
            "{}: too few IK solutions to test",
            d.name()
        );
        let result = batchplan::plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        assert!(
            result.solved().any(|s| s.index as u32 != s.problem.world),
            "{}: no solved problem whose index differs from its world",
            d.name()
        );
        for s in result.solved() {
            assert_eq!(s.problem.world, problems[s.index].world);
            // Every reported path is collision-free in its own world along its whole length.
            let n = robot.dof();
            let trajectory = Trajectory::try_new(&robot, s.solution, 1.0).unwrap();
            let q = trajectory
                .sample(32.0 / trajectory.knot_interval)
                .unwrap()
                .positions;
            let item_world = vec![s.problem.world; q.len() / n];
            let e = cpu
                .evaluate(&on_cpu, &item_world, &q, &CollisionWeights::NONE)
                .unwrap();
            assert!(
                (0..item_world.len()).all(|i| e.collision_free(i)),
                "{}: path collides in world {}",
                d.name(),
                s.problem.world
            );
        }
    }
}

#[test]
fn trajopt_smoothness_gradient_matches_its_cost() {
    // Collision off: the trajectory cost is pure smoothness,
    //   J = acceleration_weight * sum |q[t+1] - 2 q[t] + q[t-1]|^2 + velocity_weight * sum |q[t+1] - q[t]|^2,
    // whose gradient sets the direction of the first L-BFGS step (see tests/gpu.rs).
    let robot = panda();
    let n = robot.dof();
    let mut goal = robot.default_q().to_vec();
    goal[0] += 1.0;
    goal[2] -= 0.7;
    // A ball on the straight line, so the searched path detours and the problem is optimized.
    let middle: Vec<f32> = robot
        .default_q()
        .iter()
        .zip(&goal)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let ball = Obstacle::sphere(robot.ee_pose(&middle, 0).position, 0.05);
    let problems = vec![PlanProblem::new(0, robot.default_q(), goal)];
    let scene = vec![World::new(vec![ball])];
    let o = PlanOptions {
        collision: CollisionWeights::NONE,
        seeds: 8,
        ..Default::default()
    };
    let cost = |path: &[f64]| {
        let t_count = path.len() / n;
        let q = |t: usize, j: usize| path[t * n + j];
        let mut c = 0.0;
        for j in 0..n {
            for t in 0..t_count - 1 {
                c += o.velocity_weight as f64 * (q(t + 1, j) - q(t, j)).powi(2);
            }
            for t in 1..t_count - 1 {
                c += o.acceleration_weight as f64
                    * (q(t + 1, j) - 2.0 * q(t, j) + q(t - 1, j)).powi(2);
            }
        }
        c
    };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let seed = batchplan::plan(&d, &worlds, &problems, &PlanOptions { iterations: 0, ..o })
            .unwrap()
            .paths;
        let stepped = batchplan::plan(&d, &worlds, &problems, &PlanOptions { iterations: 1, ..o })
            .unwrap()
            .paths;
        // Optimized seed 2's path, bent through a via point (seed 0 is the searched path, seed 1
        // the straight line), after the searched path. Only control points 3..points - 3 move.
        let points = o.control_points;
        let (seed, stepped) = (&seed[3], &stepped[3]);
        let path: Vec<f64> = seed.iter().map(|&v| v as f64).collect();
        let free = 3 * n..(points - 3) * n;
        let fd: Vec<f64> = free
            .clone()
            .map(|i| {
                let h = 1e-5;
                let (mut up, mut down) = (path.clone(), path.clone());
                up[i] += h;
                down[i] -= h;
                (cost(&up) - cost(&down)) / (2.0 * h)
            })
            .collect();
        // Without history the step is steepest descent: -gradient times a positive scale.
        let step: Vec<f64> = free.map(|i| (seed[i] - stepped[i]) as f64).collect();
        let largest = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let (fd_max, step_max) = (largest(&fd), largest(&step));
        assert!(
            fd_max > 0.1,
            "{}: test path is too straight to exercise the gradient ({fd_max})",
            d.name()
        );
        assert!(
            step_max > 0.0,
            "{}: the step did not move the path",
            d.name()
        );
        let worst = fd
            .iter()
            .zip(&step)
            .map(|(f, s)| (f / fd_max - s / step_max).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1e-2,
            "{}: the step's direction is off the smoothness gradient by {worst:.2e}",
            d.name()
        );
    }
}

/// Weights of a uniform cubic B-spline span's control points at `u` (same as the planner's).
fn basis(u: f64) -> [f64; 4] {
    let v = 1.0 - u;
    [
        v * v * v / 6.0,
        (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0,
        (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0,
        u * u * u / 6.0,
    ]
}

#[test]
fn trajopt_collision_gradient_matches_its_cost() {
    // The full trajectory cost: smoothness on the control points plus the collision cost at
    // `samples_per_span` points of every span, at u = (s + 0.5) / samples_per_span.
    let robot = panda();
    let n = robot.dof();
    let mut goal = robot.default_q().to_vec();
    goal[0] += 1.2;
    let start = robot.default_q().to_vec();
    // A post between start and goal that the straight seed sweeps through.
    let middle: Vec<f32> = start
        .iter()
        .zip(&goal)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let center = robot.ee_pose(&middle, 0).position;
    let post = Obstacle::Cylinder {
        center,
        rotation: Quat::IDENTITY,
        radius: 0.05,
        half_height: 0.3,
    };
    let scene = vec![World::new(vec![post])];
    let problems = vec![PlanProblem::new(0, start, goal)];
    // No search beyond the straight line, which the post blocks, so optimized seed 0 is that line.
    let o = PlanOptions {
        seeds: 8,
        rrt: RrtOptions {
            max_iterations: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let (points, k) = (o.control_points, o.samples_per_span);
    let samples_of = |cp: &[f64]| -> Vec<f32> {
        let mut out = vec![];
        for span in 0..points - 3 {
            for s in 0..k {
                let w = basis((s as f64 + 0.5) / k as f64);
                out.extend(
                    (0..n)
                        .map(|j| (0..4).map(|i| w[i] * cp[(span + i) * n + j]).sum::<f64>() as f32),
                );
            }
        }
        out
    };
    let smoothness = |cp: &[f64]| {
        let q = |t: usize, j: usize| cp[t * n + j];
        let mut c = 0.0;
        for j in 0..n {
            for t in 0..points - 1 {
                c += o.velocity_weight as f64 * (q(t + 1, j) - q(t, j)).powi(2);
            }
            for t in 1..points - 1 {
                c += o.acceleration_weight as f64
                    * (q(t + 1, j) - 2.0 * q(t, j) + q(t - 1, j)).powi(2);
            }
        }
        c
    };
    let cpu = Device::cpu(&robot).unwrap();
    let on_cpu = cpu.upload(&scene).unwrap();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let seed = batchplan::plan(&d, &worlds, &problems, &PlanOptions { iterations: 0, ..o })
            .unwrap()
            .paths;
        let stepped = batchplan::plan(&d, &worlds, &problems, &PlanOptions { iterations: 1, ..o })
            .unwrap()
            .paths;
        // Optimized seed 0, the straight line through the post, after the (unsolved) searched path.
        let (seed, stepped) = (&seed[1], &stepped[1]);
        let path: Vec<f64> = seed.iter().map(|&v| v as f64).collect();
        let free: Vec<usize> = (3 * n..(points - 3) * n).collect();
        // Central differences at two step sizes: the cost is only piecewise smooth (obstacle
        // edges, hinge activation) and evaluated in f32, so take the closer of the two.
        let fd_at = |h: f64| -> Vec<f64> {
            let mut perturbed = vec![];
            for &i in &free {
                for sign in [1.0, -1.0] {
                    let mut p = path.clone();
                    p[i] += sign * h;
                    perturbed.extend(samples_of(&p));
                }
            }
            let per_path = (points - 3) * k;
            let items = perturbed.len() / n;
            let costs = cpu
                .evaluate(&on_cpu, &vec![0; items], &perturbed, &o.collision)
                .unwrap()
                .cost;
            let collision = |p: usize| {
                costs[p * per_path..(p + 1) * per_path]
                    .iter()
                    .map(|&c| c as f64)
                    .sum::<f64>()
            };
            free.iter()
                .enumerate()
                .map(|(f, &i)| {
                    let (mut up, mut down) = (path.clone(), path.clone());
                    up[i] += h;
                    down[i] -= h;
                    (collision(2 * f) - collision(2 * f + 1) + smoothness(&up) - smoothness(&down))
                        / (2.0 * h)
                })
                .collect()
        };
        let (fine, coarse) = (fd_at(2e-4), fd_at(1e-3));
        // Without history the step is steepest descent: -gradient times a positive scale, so
        // gradients are compared scaled to their largest entries.
        let step: Vec<f64> = free
            .iter()
            .map(|&i| (seed[i] - stepped[i]) as f64)
            .collect();
        let largest = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let (fine_max, coarse_max, step_max) = (largest(&fine), largest(&coarse), largest(&step));
        assert!(
            step_max > 0.0,
            "{}: the step did not move the path",
            d.name()
        );
        let mut worst = 0.0f64;
        for f in 0..free.len() {
            let s = step[f] / step_max;
            let err = (fine[f] / fine_max - s)
                .abs()
                .min((coarse[f] / coarse_max - s).abs());
            worst = worst.max(err / (fine[f].abs() / fine_max).max(0.05));
        }
        eprintln!(
            "{}: largest gradient {fine_max:.1}, worst relative error {worst:.2e}",
            d.name()
        );
        let seed_samples = samples_of(&path);
        let items = seed_samples.len() / n;
        let touching = cpu
            .evaluate(&on_cpu, &vec![0; items], &seed_samples, &o.collision)
            .unwrap()
            .cost;
        assert!(
            touching.iter().filter(|&&c| c > 1.0).count() > 3,
            "{}: the straight seed misses the post",
            d.name()
        );
        assert!(
            worst < 2e-2,
            "{}: trajectory gradient off by {worst:.2e} relative",
            d.name()
        );
    }
}

#[test]
fn planning_without_a_budget_gives_the_same_result_every_run() {
    let robot = panda();
    let (scene, problems) = common::reaches(&robot);
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let a = batchplan::plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        let b = batchplan::plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        assert_eq!(a.paths, b.paths, "{}: paths differ between runs", d.name());
        assert_eq!((a.valid, a.length), (b.valid, b.length), "{}", d.name());
    }
}

#[test]
fn plans_from_a_moving_start_continue_its_motion() {
    let robot = panda();
    let n = robot.dof();
    let (scene, problems) = common::reaches(&robot);
    let motion = StartMotion {
        velocity: vec![0.5, -0.3, 0.2, 0.4, 0.0, 0.3, 0.0],
        acceleration: vec![1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.5],
    };
    let moving: Vec<PlanProblem> = problems
        .iter()
        .take(8)
        .map(|p| PlanProblem {
            start_motion: Some(motion.clone()),
            ..p.clone()
        })
        .collect();
    let start = JointState {
        position: robot.default_q().to_vec(),
        velocity: motion.velocity.clone(),
        acceleration: motion.acceleration.clone(),
    };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let result = batchplan::plan(&d, &worlds, &moving, &PlanOptions::default()).unwrap();
        let solved: Vec<_> = result.solved().collect();
        assert!(
            solved.len() >= 6,
            "{}: {} of 8 planned",
            d.name(),
            solved.len()
        );
        for s in solved {
            let trajectory = Trajectory::moving(&robot, s.solution, &motion).unwrap();
            trajectory.check_from(&robot, &start).unwrap();
            assert!(
                matches!(trajectory.check(&robot), Err(Error::Unsafe(_))),
                "{}: a moving start passes as rest",
                d.name()
            );
            let mut end = JointState::new(n);
            trajectory.at(trajectory.duration(), &mut end);
            assert_eq!(
                end.position,
                s.problem.goal.joints().unwrap(),
                "{}",
                d.name()
            );
        }
    }
}

#[test]
fn reaches_replan_mid_motion_at_full_speed() {
    // Partway through full-speed reaches, replanned to the same goals: the old motions' remainders
    // show that paths exist.
    let robot = panda();
    let n = robot.dof();
    let (scene, problems) = common::reaches(&robot);
    let cpu = Device::cpu(&robot).unwrap();
    let rest = batchplan::plan(
        &cpu,
        &cpu.upload(&scene).unwrap(),
        &problems[..24],
        &PlanOptions::default(),
    )
    .unwrap();
    let mut state = JointState::new(n);
    let (mut moving, mut states) = (vec![], vec![]);
    for s in rest.solved() {
        let trajectory = Trajectory::try_new(&robot, s.solution, 1.0).unwrap();
        trajectory.at(0.4 * trajectory.duration(), &mut state);
        if state.velocity.iter().fold(0.0f32, |m, v| m.max(v.abs())) < 1.5 {
            continue;
        }
        let motion = StartMotion {
            velocity: state.velocity.clone(),
            acceleration: state.acceleration.clone(),
        };
        moving.push(PlanProblem {
            start: state.position.clone(),
            start_motion: Some(motion),
            ..s.problem.clone()
        });
        states.push(state.clone());
    }
    assert!(moving.len() >= 8, "only {} reaches move fast", moving.len());
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let result = batchplan::plan(&d, &worlds, &moving, &PlanOptions::default()).unwrap();
        let solved: Vec<_> = result.solved().collect();
        eprintln!(
            "{}: {} of {} replanned at full speed",
            d.name(),
            solved.len(),
            moving.len()
        );
        // 24 of 24 on an M4 Pro and on the Ryzen; 18 when timing gave up on a moving start whose
        // smoothed map overshot, and 0 with the start knot interval sized for rest.
        assert!(
            10 * solved.len() >= 9 * moving.len(),
            "{}: {} of {} replanned",
            d.name(),
            solved.len(),
            moving.len()
        );
        for s in solved {
            let motion = s.problem.start_motion.as_ref().unwrap();
            let trajectory = Trajectory::moving(&robot, s.solution, motion).unwrap();
            trajectory.check_from(&robot, &states[s.index]).unwrap();
        }
    }
}

#[test]
fn a_robot_mounted_on_a_table_is_not_in_collision_with_it() {
    // Spheres fitted to the Panda's meshes reach below its base, into the table it stands on.
    let robot = Robot::load(
        common::asset("franka/franka_panda.urdf"),
        &RobotOptions {
            collision_model: None,
            ..common::panda_options()
        },
    )
    .unwrap();
    let base = robot.link_pose(robot.default_q(), "panda_link0").unwrap();
    let lowest = robot.collision_model().spheres["panda_link0"]
        .iter()
        .map(|s| (base.position + base.rotation * Vec3::new(s[0], s[1], s[2])).z - s[3])
        .fold(f32::INFINITY, f32::min);
    assert!(
        lowest < 0.0,
        "the base's spheres should reach below the table: {lowest}"
    );
    let table = Obstacle::Cuboid {
        center: Vec3::new(0.4, 0.0, -0.02),
        half_extents: Vec3::new(0.7, 0.8, 0.02),
        rotation: Quat::IDENTITY,
    };
    let mounted = World::new(vec![table.clone()]);
    // A pin through the elbow still collides.
    let elbow = robot
        .link_pose(robot.default_q(), "panda_link4")
        .unwrap()
        .position;
    let pin = Obstacle::Cuboid {
        center: elbow,
        half_extents: Vec3::new(0.3, 0.005, 0.005),
        rotation: Quat::IDENTITY,
    };
    let pinned = World::new(vec![table, pin]);
    let goal = vec![0.5, -0.8, 0.0, -2.2, 0.0, 1.6, 0.8];
    for d in devices(&robot) {
        let worlds = d.upload(&[mounted.clone(), pinned.clone()]).unwrap();
        let q: Vec<f32> = robot
            .default_q()
            .iter()
            .chain(robot.default_q())
            .copied()
            .collect();
        let e = d
            .evaluate(&worlds, &[0, 1], &q, &CollisionWeights::NONE)
            .unwrap();
        assert!(
            e.collision_free(0),
            "{}: on its table, world clearance {}",
            d.name(),
            e.world_clearance[0]
        );
        assert!(!e.collision_free(1), "{}: the pin should collide", d.name());
        let problem = PlanProblem::new(0, robot.default_q(), goal.clone());
        let result = batchplan::plan(&d, &worlds, &[problem], &PlanOptions::default()).unwrap();
        assert!(
            result.best(0).is_some(),
            "{}: no plan on the table",
            d.name()
        );
    }
}

#[test]
fn seeded_ik_stays_on_the_branch_it_starts_from() {
    let robot = panda();
    // A configuration far from the default one, and a target 2 cm above its IK frame.
    let here = vec![1.2, 0.3, -0.8, -1.6, 0.9, 2.0, -0.4];
    let far: f32 = here
        .iter()
        .zip(robot.default_q())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!(
        far > 1.5,
        "the start should be far from the default configuration: {far}"
    );
    let mut target = robot.ee_pose(&here, 0);
    target.position.z += 0.02;
    let one_seed = IkOptions {
        seeds: 1,
        ..IkOptions::default()
    };
    for d in devices(&robot) {
        let worlds = d.upload(&[World::default()]).unwrap();
        let seeded = IkProblem {
            world: 0,
            targets: vec![Some(target)],
            seed: Some(here.clone()),
            hold: vec![],
            holding: None,
        };
        let ik = batchplan::solve_ik(&d, &worlds, &[seeded], &one_seed).unwrap();
        let q = ik
            .nearest(0, &here)
            .unwrap_or_else(|| panic!("{}: no solution from the seed", d.name()));
        let moved: f32 = q
            .iter()
            .zip(&here)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(moved < 0.2, "{}: moved {moved} rad from the seed", d.name());
        // Unseeded, the one seed starts from the default configuration and ends elsewhere.
        let ik = batchplan::solve_ik(&d, &worlds, &[IkProblem::new(0, target)], &one_seed).unwrap();
        let elsewhere = ik
            .solution(0)
            .iter()
            .zip(&here)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            elsewhere > 0.5,
            "{}: from the default it lands {elsewhere} rad away",
            d.name()
        );
    }
}

/// Optimization starts from the searched path's spline, then the straight line to the goal: with no
/// steps taken, seeds 0 and 1 are those.
#[test]
fn optimization_seeds_start_with_the_searched_path_and_the_straight_line() {
    let robot = panda();
    let n = robot.dof();
    let (scene, problems) = common::blocked_reaches(&robot);
    let o = PlanOptions {
        seeds: 2,
        iterations: 0,
        ..Default::default()
    };
    let (points, per_path) = (o.control_points, o.control_points * n);
    for d in devices(&robot) {
        let result = batchplan::plan(&d, &d.upload(&scene).unwrap(), &problems, &o).unwrap();
        let mut seeded = 0;
        for (pi, p) in problems.iter().enumerate() {
            let path = |k: usize| &result.paths[pi * 3 + k];
            if path(1).is_empty() {
                continue;
            }
            seeded += 1;
            assert_eq!(
                path(1),
                path(0),
                "{}: problem {pi}'s first seed is not its searched spline",
                d.name()
            );
            for (t, q) in path(2).chunks(n).enumerate() {
                let u = (t.saturating_sub(2) as f32 / (points - 5) as f32).min(1.0);
                let off = (0..n)
                    .map(|j| {
                        (q[j] - (p.start[j] + u * (p.goal.joints().unwrap()[j] - p.start[j]))).abs()
                    })
                    .fold(0.0, f32::max);
                assert!(
                    off < 1e-5,
                    "{}: problem {pi}'s second seed leaves the straight line by {off}",
                    d.name()
                );
            }
        }
        assert!(
            seeded * 3 >= problems.len(),
            "{}: only {seeded} of {} problems seeded",
            d.name(),
            problems.len()
        );
        assert_eq!(result.paths.len(), problems.len() * 3);
        assert!(
            result
                .paths
                .iter()
                .all(|p| p.is_empty() || p.len() == per_path),
            "{}: a path has other than the options' control points",
            d.name()
        );
    }
}

/// Reaches under a slab from the Panda's default configuration, as MotionBenchMaker's picks under a
/// table: the slab's world and the problems.
fn reaches_under_a_slab(robot: &Robot) -> (World, Vec<PlanProblem>) {
    let slab = Obstacle::cuboid(Vec3::new(0.6, 0.0, 0.35), Vec3::new(0.25, 0.5, 0.02));
    let floor = Obstacle::cuboid(Vec3::new(0.5, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let start = robot.default_q().to_vec();
    let mut rng = Rng::new(3);
    let side = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let under = (0..64)
        .map(|_| {
            let at = Vec3::new(
                rng.range(0.45, 0.75),
                rng.range(-0.35, 0.35),
                rng.range(0.12, 0.23),
            );
            PlanProblem::new(0, start.clone(), Pose::new(at, side))
        })
        .collect();
    (World::new(vec![slab, floor]), under)
}

/// An optimized seed no step moves is valid where the searched spline it copies is, and as long:
/// optimized seeds are validated at the samples searched splines are checked at, keeping the margin
/// they keep (2 mm, or none with `shortcut.clearance` 0), ramped from the curve's ends (where a
/// moving start is, not its first control point).
#[test]
fn unmoved_optimized_seeds_are_valid_where_their_searched_splines_are() {
    let robot = panda();
    let n = robot.dof();
    let (slab, under) = reaches_under_a_slab(&robot);
    let start = robot.default_q().to_vec();
    let mut velocity = vec![0.0; n];
    velocity[0] = 2.0;
    // A sphere 0.3 mm behind the hand as it starts moving away.
    let hand = robot.ee_pose(&start, 0).position;
    let ahead: Vec<f32> = start
        .iter()
        .zip(&velocity)
        .map(|(q, v)| q + 1e-3 * v)
        .collect();
    let away = (robot.ee_pose(&ahead, 0).position - hand).normalize();
    let behind = |gap: f32| World::new(vec![Obstacle::sphere(hand - away * (0.03 + gap), 0.03)]);
    let cpu = Device::cpu(&robot).unwrap();
    let near = CollisionWeights {
        margin: 0.01,
        ..CollisionWeights::NONE
    };
    let clearance = |gap: f32| {
        let worlds = cpu.upload(&[behind(gap)]).unwrap();
        cpu.evaluate(&worlds, &[0], &start, &near)
            .unwrap()
            .world_clearance[0]
    };
    let (mut lo, mut hi) = (-0.1, 0.2);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if clearance(mid) < 3e-4 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    assert!(
        clearance(hi) < 1e-3,
        "the start is {} m from the sphere",
        clearance(hi)
    );
    let sphere = behind(hi);
    let mut goal = start.clone();
    goal[0] += 1.5;
    goal[3] += 0.3;
    let moving = PlanProblem {
        start_motion: Some(StartMotion {
            velocity,
            acceleration: vec![0.0; n],
        }),
        ..PlanProblem::new(0, start, goal)
    };
    let unmoved = PlanOptions {
        seeds: 1,
        iterations: 0,
        ..Default::default()
    };
    let loose = PlanOptions {
        shortcut: ShortcutOptions {
            clearance: 0.0,
            ..Default::default()
        },
        ..unmoved
    };
    let cases = [
        ("under a slab", &slab, &under[..], &unmoved),
        ("under a slab, no clearance", &slab, &under[..], &loose),
        (
            "moving away from a sphere",
            &sphere,
            std::slice::from_ref(&moving),
            &unmoved,
        ),
    ];
    for d in devices(&robot) {
        for (case, world, problems, o) in cases {
            let worlds = d.upload(std::slice::from_ref(world)).unwrap();
            let result = batchplan::plan(&d, &worlds, problems, o).unwrap();
            let mut compared = 0;
            for p in 0..problems.len() {
                let (spline, seed) = (2 * p, 2 * p + 1);
                // Problems searched straight are not optimized.
                if !result.valid[spline] || result.paths[seed].is_empty() {
                    continue;
                }
                compared += 1;
                assert!(
                    result.paths[seed] == result.paths[spline]
                        && result.valid[seed]
                        && result.length[seed] == result.length[spline],
                    "{}, {case}: problem {p}'s searched spline is valid at {} rad; its unmoved copy \
                     valid {} at {} rad",
                    d.name(),
                    result.length[spline],
                    result.valid[seed],
                    result.length[seed]
                );
            }
            assert!(compared > 0, "{}, {case}: nothing optimized", d.name());
        }
    }
}

/// Joint-space distance.
fn apart(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// The spline through control points `cp` (`[points, n]`) sampled as searched splines are checked:
/// every span at `per_span` evenly spaced points, or at more where it moves farther, so that
/// neighbouring samples lie at most `spacing` apart; then its end.
fn checked_samples(cp: &[f32], n: usize, per_span: usize, spacing: f32) -> Vec<f32> {
    let point = |i: usize| &cp[i * n..(i + 1) * n];
    let mut out = vec![];
    for span in 0..cp.len() / n - 3 {
        // At most three times the span's Bezier control polygon's longest edge per unit of u.
        let speed = (0.5 * apart(point(span), point(span + 2)))
            .max(apart(point(span + 1), point(span + 2)))
            .max(0.5 * apart(point(span + 1), point(span + 3)));
        let count = per_span.max((speed / spacing).ceil() as usize);
        for s in 0..count {
            let w = basis(s as f64 / count as f64);
            let q = (0..n).map(|j| {
                (0..4)
                    .map(|i| w[i] as f32 * cp[(span + i) * n + j])
                    .sum::<f32>()
            });
            out.extend(q);
        }
    }
    out.extend_from_slice(point(cp.len() / n - 1));
    out
}

/// Optimized seeds are valid exactly where they keep the spline margin from the world (2 mm, ramped
/// from the curve's ends) and are clear of themselves, at the samples searched splines are checked
/// at. Optimization pulls seeds to reaches under a slab taut along its edge, and the margin alone
/// rejects some.
#[test]
fn optimized_seeds_are_valid_where_they_keep_the_spline_margin() {
    const MARGIN: f32 = 0.002;
    let robot = panda();
    let n = robot.dof();
    let (slab, under) = reaches_under_a_slab(&robot);
    let o = PlanOptions {
        seeds: 8,
        // Part way, more of them sit within the margin.
        iterations: 30,
        ..Default::default()
    };
    let seeds = 1 + o.seeds;
    let (k, spacing) = (o.validate_substeps, o.shortcut.resolution);
    let near = CollisionWeights {
        margin: 0.01,
        ..CollisionWeights::NONE
    };
    for d in devices(&robot) {
        let worlds = d.upload(std::slice::from_ref(&slab)).unwrap();
        let result = batchplan::plan(&d, &worlds, &under, &o).unwrap();
        let (mut kept, mut short) = (0, 0);
        let optimized = (0..result.paths.len()).filter(|&i| i % seeds > 0);
        for item in optimized.filter(|&i| !result.paths[i].is_empty()) {
            let q = checked_samples(&result.paths[item], n, k, spacing);
            let items = q.len() / n;
            let e = d.evaluate(&worlds, &vec![0; items], &q, &near).unwrap();
            let ends = [&q[..n], &q[q.len() - n..]];
            let (mut excess, mut clear) = (f32::INFINITY, true);
            for (i, s) in q.chunks(n).enumerate() {
                let from_end = apart(s, ends[0]).min(apart(s, ends[1]));
                let margin = MARGIN * (from_end / o.shortcut.clearance_ramp).min(1.0);
                let (world, own) = (e.world_clearance[i], e.self_clearance[i]);
                excess = excess.min((world - margin).min(own));
                clear &= world >= 0.0 && own >= 0.0;
            }
            // Rounding decides within a hundredth of a millimeter.
            if excess.abs() < 1e-5 {
                continue;
            }
            assert_eq!(
                result.valid[item],
                excess > 0.0,
                "{}: problem {}'s seed {} keeps {excess} m past the margin",
                d.name(),
                item / seeds,
                item % seeds
            );
            kept += usize::from(excess > 0.0);
            short += usize::from(excess < 0.0 && clear);
        }
        assert!(
            kept > 0 && short > 0,
            "{}: {kept} seeds keep the margin, {short} clear of the world do not",
            d.name()
        );
    }
}

/// A problem searched to its straight line is not optimized: no seed could be shorter.
#[test]
fn problems_searched_straight_are_not_optimized() {
    let robot = panda();
    let (_, reaches) = common::reaches(&robot);
    let scene = vec![World::default()];
    let problems: Vec<PlanProblem> = reaches
        .into_iter()
        .take(8)
        .map(|p| PlanProblem { world: 0, ..p })
        .collect();
    let o = PlanOptions {
        seeds: 4,
        ..Default::default()
    };
    for d in devices(&robot) {
        let result = batchplan::plan(&d, &d.upload(&scene).unwrap(), &problems, &o).unwrap();
        for pi in 0..problems.len() {
            assert!(
                result.valid[pi * 5],
                "{}: problem {pi} has no searched path",
                d.name()
            );
            assert!(
                (1..5).all(|k| !result.valid[pi * 5 + k]),
                "{}: problem {pi}'s straight path was optimized",
                d.name()
            );
        }
    }
}

/// A budget spent while searching leaves optimization unstarted.
#[test]
fn a_spent_budget_starts_no_optimization() {
    let robot = panda();
    let (_, reaches) = common::reaches(&robot);
    let scene = vec![World::default()];
    let problems: Vec<PlanProblem> = reaches
        .into_iter()
        .take(8)
        .map(|p| PlanProblem { world: 0, ..p })
        .collect();
    let o = PlanOptions {
        seeds: 4,
        time_budget: Some(Duration::ZERO),
        ..Default::default()
    };
    for d in devices(&robot) {
        let result = batchplan::plan(&d, &d.upload(&scene).unwrap(), &problems, &o).unwrap();
        assert!(
            result.valid.iter().all(|&v| !v),
            "{}: a seed was planned or optimized",
            d.name()
        );
    }
}

/// Repairs keep their own weights: without optimization seeds, plans do not depend on the
/// optimization's.
#[test]
fn plans_without_seeds_ignore_the_optimization_weights() {
    let robot = panda();
    let n = robot.dof();
    let (scene, problems) = common::reaches(&robot);
    let cpu = Device::cpu(&robot).unwrap();
    let worlds = cpu.upload(&scene).unwrap();
    let rest = batchplan::plan(&cpu, &worlds, &problems, &PlanOptions::default()).unwrap();
    let mut state = JointState::new(n);
    let moving: Vec<PlanProblem> = rest
        .solved()
        .map(|s| {
            let trajectory = Trajectory::try_new(&robot, s.solution, 1.0).unwrap();
            trajectory.at(0.4 * trajectory.duration(), &mut state);
            let motion = StartMotion {
                velocity: state.velocity.clone(),
                acceleration: state.acceleration.clone(),
            };
            PlanProblem {
                start: state.position.clone(),
                start_motion: Some(motion),
                ..s.problem.clone()
            }
        })
        .collect();
    let weighed = PlanOptions {
        acceleration_weight: 0.5,
        velocity_weight: 40.0,
        collision: CollisionWeights {
            margin: 0.003,
            ..PlanOptions::default().collision
        },
        ..Default::default()
    };
    let a = batchplan::plan(&cpu, &worlds, &moving, &PlanOptions::default()).unwrap();
    let b = batchplan::plan(&cpu, &worlds, &moving, &weighed).unwrap();
    assert_eq!(
        a.paths, b.paths,
        "the optimization's weights changed plans without seeds"
    );
}
