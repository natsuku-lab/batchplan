//! A held object per problem (`PlanProblem::holding`, `IkProblem::holding`, `ToolPath::holding`):
//! one call takes problems holding nothing, different objects and one object by different grasps,
//! and gives each what a device holding its object alone gives.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::FRAC_PI_2;

use batchplan::{
    Contact, Device, Error, Held, IkOptions, IkProblem, Obstacle, PlanFailure, PlanOptions,
    PlanProblem, Pose, Robot, SdfGrid, ToolPath, ToolPathOptions, ToolPathStop, Touch, World,
    Worlds,
};
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

const SLAB: usize = 0;
const HELD: &str = "held by hand 0";

/// A 10 cm cube held with the IK frame `inset` from its middle toward the hand.
fn cube(inset: f32) -> Held {
    Held {
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::splat(0.05)),
        grasp: Pose::new(Vec3::new(0.0, 0.0, -inset), Quat::IDENTITY),
    }
}

/// A 24 cm bar held across the hand, level, by its middle.
fn bar() -> Held {
    Held {
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.02, 0.12)),
        grasp: Pose::new(Vec3::ZERO, Quat::from_rotation_y(FRAC_PI_2)),
    }
}

/// The Panda at its ready pose over a slab 4 cm under its IK frame: clear of the hand, in the way
/// of a cube held there. `up` is 15 cm higher and `aside` 20 cm to the side of that.
struct Scene {
    robot: Robot,
    world: World,
    ready: Vec<f32>,
    up: Vec<f32>,
    aside: Vec<f32>,
}

impl Scene {
    fn new() -> Self {
        let robot = common::panda().unwrap();
        let ready = robot.default_q().to_vec();
        let hand = robot.ee_pose(&ready, 0);
        let slab = Obstacle::cuboid(
            hand.position - Vec3::Z * 0.045,
            Vec3::new(0.15, 0.15, 0.005),
        );
        let world = World::new(vec![slab]);
        let device = Device::cpu(&robot).unwrap();
        let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
        let reach = |by: Vec3| {
            let target = Pose::new(hand.position + by, hand.rotation);
            let problem = IkProblem {
                seed: Some(ready.clone()),
                ..IkProblem::new(0, target)
            };
            let solved =
                batchplan::solve_ik(&device, &worlds, &[problem], &IkOptions::default()).unwrap();
            solved.nearest(0, &ready).expect("reachable").to_vec()
        };
        let up = reach(Vec3::Z * 0.15);
        let aside = reach(Vec3::new(0.0, 0.2, 0.15));
        Self {
            robot,
            world,
            ready,
            up,
            aside,
        }
    }

    /// Where problem `i` of a batch of [`holds`] goes: the first three to the ready pose, the rest
    /// aside, so the problems holding alike differ.
    fn goal(&self, i: usize) -> &[f32] {
        if i < 3 { &self.ready } else { &self.aside }
    }

    /// A device of `device`'s holding `held`, as attaching it gives one.
    fn holder(&self, device: &Device, held: Option<&Held>) -> Device {
        let robot = match held {
            Some(held) => self
                .robot
                .attach(&held.attachment(&self.robot).unwrap())
                .unwrap(),
            None => self.robot.clone(),
        };
        device.with_robot(&robot).unwrap()
    }
}

/// Holding nothing, a cube by two grasps and a bar, interleaved, the first cube twice.
fn holds() -> Vec<Option<Held>> {
    vec![
        None,
        Some(cube(0.0)),
        Some(cube(0.02)),
        Some(bar()),
        None,
        Some(cube(0.0)),
    ]
}

/// [`holds`], then as many problems all holding the cube by its first grasp, one batch each: in
/// each, the problem first holding the cube goes down to the ready pose.
fn batches() -> [(Vec<Option<Held>>, usize); 2] {
    [(holds(), 1), (vec![Some(cube(0.0)); 6], 0)]
}

fn held_at_slab(contacts: &[Contact]) -> bool {
    contacts.iter().any(|c| {
        matches!(c, Contact::World { link, obstacle: SLAB, distance } if link == HELD && *distance < 0.0)
    })
}

#[test]
fn each_plan_holds_its_own_object() {
    let s = Scene::new();
    let o = PlanOptions::default();
    for device in devices(&s.robot) {
        let name = device.name();
        let worlds = device.upload(std::slice::from_ref(&s.world)).unwrap();
        for (holds, cube) in batches() {
            // The first three down to the ready pose, the rest aside, so each group's problems differ.
            let problems: Vec<PlanProblem> = holds
                .into_iter()
                .enumerate()
                .map(|(i, holding)| PlanProblem {
                    holding,
                    ..PlanProblem::new(0, s.up.clone(), s.goal(i).to_vec())
                })
                .collect();
            let all = batchplan::plan(&device, &worlds, &problems, &o).unwrap();
            for (i, p) in problems.iter().enumerate() {
                assert_eq!(all.problems[i].holding, p.holding, "{name}: problem {i}");
                let holder = s.holder(&device, p.holding.as_ref());
                let alone = PlanProblem {
                    holding: None,
                    ..p.clone()
                };
                let one = batchplan::plan(&holder, &worlds, &[alone], &o).unwrap();
                let items = i * all.seeds..(i + 1) * all.seeds;
                assert_eq!(
                    all.paths[items.clone()],
                    one.paths[..],
                    "{name}: problem {i}"
                );
                assert_eq!(
                    all.valid[items.clone()],
                    one.valid[..],
                    "{name}: problem {i}"
                );
                assert_eq!(all.length[items], one.length[..], "{name}: problem {i}");
                assert_eq!(all.failures[i], one.failures[0], "{name}: problem {i}");
            }
            // Holding nothing, the hand reaches the ready pose over the slab; holding the cube, it
            // would put the cube into it.
            if cube > 0 {
                assert!(all.best(0).is_some(), "{name}: {:?}", all.failures[0]);
            }
            assert!(
                matches!(&all.failures[cube], Some(PlanFailure::GoalCollides { contacts }) if held_at_slab(contacts)),
                "{name}: {:?}",
                all.failures[cube]
            );
        }
    }
}

#[test]
fn each_ik_problem_holds_its_own_object() {
    let s = Scene::new();
    let o = IkOptions::default();
    for device in devices(&s.robot) {
        let name = device.name();
        let worlds = device.upload(std::slice::from_ref(&s.world)).unwrap();
        for (holds, cube) in batches() {
            let problems: Vec<IkProblem> = holds
                .into_iter()
                .enumerate()
                .map(|(i, holding)| IkProblem {
                    seed: Some(s.up.clone()),
                    holding,
                    ..IkProblem::new(0, s.robot.ee_pose(s.goal(i), 0))
                })
                .collect();
            let all = batchplan::solve_ik(&device, &worlds, &problems, &o).unwrap();
            let why = batchplan::ik::failures(&device, &worlds, &all, &o).unwrap();
            for (i, p) in problems.iter().enumerate() {
                assert_eq!(all.problems[i].holding, p.holding, "{name}: problem {i}");
                let holder = s.holder(&device, p.holding.as_ref());
                let alone = IkProblem {
                    holding: None,
                    ..p.clone()
                };
                let one = batchplan::solve_ik(&holder, &worlds, &[alone], &o).unwrap();
                let items = i * all.seeds..(i + 1) * all.seeds;
                let n = all.dof;
                assert_eq!(
                    all.q[items.start * n..items.end * n],
                    one.q[..],
                    "{name}: problem {i}"
                );
                assert_eq!(
                    all.success[items.clone()],
                    one.success[..],
                    "{name}: problem {i}"
                );
                assert_eq!(
                    all.world_clearance[items.clone()],
                    one.world_clearance[..],
                    "{name}: problem {i}"
                );
                assert_eq!(
                    all.self_clearance[items],
                    one.self_clearance[..],
                    "{name}: problem {i}"
                );
                let alone_why = batchplan::ik::failures(&holder, &worlds, &one, &o).unwrap();
                assert_eq!(why[i], alone_why[0], "{name}: problem {i}");
            }
            // The hand alone reaches the ready pose; the cube held there meets the slab, and its
            // failure says so.
            if cube > 0 {
                assert!(all.best(0).is_some(), "{name}");
            }
            assert!(
                matches!(&why[cube], Some(batchplan::IkFailure::Collides { contacts }) if held_at_slab(contacts)),
                "{name}: {:?}",
                why[cube]
            );
        }
    }
}

#[test]
fn each_tool_path_holds_its_own_object() {
    let s = Scene::new();
    let o = ToolPathOptions::default();
    for device in devices(&s.robot) {
        let name = device.name();
        let worlds = device.upload(std::slice::from_ref(&s.world)).unwrap();
        for (holds, cube) in batches() {
            let paths: Vec<ToolPath> = holds
                .into_iter()
                .enumerate()
                .map(|(i, holding)| ToolPath {
                    world: 0,
                    start: s.up.clone(),
                    hand: 0,
                    waypoints: vec![s.robot.ee_pose(s.goal(i), 0)],
                    holding,
                })
                .collect();
            let all = batchplan::follow(&device, &worlds, &paths, &o).unwrap();
            for (i, p) in paths.iter().enumerate() {
                assert_eq!(all.problems[i].holding, p.holding, "{name}: path {i}");
                let holder = s.holder(&device, p.holding.as_ref());
                let alone = ToolPath {
                    holding: None,
                    ..p.clone()
                };
                let one = batchplan::follow(&holder, &worlds, &[alone], &o).unwrap();
                assert_eq!(all.fraction[i], one.fraction[0], "{name}: path {i}");
                assert_eq!(all.stopped[i], one.stopped[0], "{name}: path {i}");
                assert_eq!(all.paths[i], one.paths[0], "{name}: path {i}");
                assert_eq!(all.contacts[i], one.contacts[0], "{name}: path {i}");
            }
            // Down to the ready pose: the hand alone gets there; the cube stops on the slab.
            if cube > 0 {
                assert_eq!(all.fraction[0], 1.0, "{name}: {:?}", all.stopped[0]);
            }
            assert!(
                matches!(all.stopped[cube], Some(ToolPathStop::Collision { .. }))
                    && held_at_slab(&all.contacts[cube]),
                "{name}: {:?} {:?}",
                all.stopped[cube],
                all.contacts[cube]
            );
        }
    }
}

#[test]
fn each_plan_holds_its_own_object_through_optimization() {
    // Reaches around a ball, so searches detour and optimized seeds have work to do.
    let robot = common::panda().unwrap();
    let (scene, reaches) = common::blocked_reaches(&robot);
    let o = PlanOptions {
        seeds: 2,
        ..PlanOptions::default()
    };
    let cycle = holds();
    let problems: Vec<PlanProblem> = reaches
        .into_iter()
        .take(12)
        .enumerate()
        .map(|(i, p)| PlanProblem {
            holding: cycle[i % cycle.len()].clone(),
            ..p
        })
        .collect();
    let s = Scene::new();
    for device in devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&scene).unwrap();
        let all = batchplan::plan(&device, &worlds, &problems, &o).unwrap();
        for (i, p) in problems.iter().enumerate() {
            let holder = s.holder(&device, p.holding.as_ref());
            let alone = PlanProblem {
                holding: None,
                ..p.clone()
            };
            let one = batchplan::plan(&holder, &worlds, &[alone], &o).unwrap();
            let items = i * all.seeds..(i + 1) * all.seeds;
            assert_eq!(
                all.paths[items.clone()],
                one.paths[..],
                "{name}: problem {i}"
            );
            assert_eq!(
                all.valid[items.clone()],
                one.valid[..],
                "{name}: problem {i}"
            );
            assert_eq!(all.length[items], one.length[..], "{name}: problem {i}");
            assert_eq!(all.failures[i], one.failures[0], "{name}: problem {i}");
        }
        let optimized = |i: usize| (1..all.seeds).any(|k| all.valid[i * all.seeds + k]);
        assert!(
            (0..problems.len()).any(|i| problems[i].holding.is_some() && optimized(i)),
            "{name}: no held problem has a valid optimized seed"
        );
    }
}

#[test]
fn a_hold_the_device_cannot_take_is_an_input_error_naming_its_problem() {
    let s = Scene::new();
    let device = Device::cpu(&s.robot).unwrap();
    let worlds = device.upload(std::slice::from_ref(&s.world)).unwrap();
    let plan = |device: &Device, worlds: &Worlds, holding: Held| {
        let problems = [
            PlanProblem::new(0, s.up.clone(), s.aside.clone()),
            PlanProblem {
                holding: Some(holding),
                ..PlanProblem::new(0, s.up.clone(), s.aside.clone())
            },
        ];
        batchplan::plan(device, worlds, &problems, &PlanOptions::default()).unwrap_err()
    };
    let input = |e: Error, says: &str| {
        assert!(
            matches!(e, Error::Input(_))
                && e.to_string().contains("problem 1")
                && e.to_string().contains(says),
            "{e}"
        );
    };
    // A device already holding one in that hand.
    let holder = s.holder(&device, Some(&cube(0.0)));
    input(plan(&holder, &worlds, bar()), HELD);
    // A hand the robot lacks, a distance grid, and a grasp that is not a pose.
    input(plan(&device, &worlds, Held { hand: 1, ..bar() }), "hand 1");
    let grid = SdfGrid::try_new([2, 2, 2], 0.1, Vec3::ZERO, &[1.0; 8]).unwrap();
    let held_grid = Held {
        object: Obstacle::from_grid(grid),
        ..bar()
    };
    input(plan(&device, &worlds, held_grid), "distance grid");
    let skewed = Held {
        grasp: Pose::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
        ..bar()
    };
    input(plan(&device, &worlds, skewed), "grasp");
    // A world out of range is named by its problem, however the batch is grouped.
    let ik = [
        IkProblem {
            holding: Some(cube(0.0)),
            ..IkProblem::new(0, s.robot.ee_pose(&s.up, 0))
        },
        IkProblem::new(0, s.robot.ee_pose(&s.up, 0)),
        IkProblem::new(7, s.robot.ee_pose(&s.up, 0)),
    ];
    let e = batchplan::solve_ik(&device, &worlds, &ik, &IkOptions::default()).unwrap_err();
    assert!(e.to_string().contains("IK problem 2"), "{e}");
    // Worlds whose touches were resolved for a robot holding something in hand 0 serve no hold in
    // hand 1.
    let two = common::two_pandas().unwrap();
    let both = Device::cpu(&two).unwrap();
    let left = cube(0.0);
    let holding_left = both
        .with_robot(&two.attach(&left.attachment(&two).unwrap()).unwrap())
        .unwrap();
    let mut world = s.world.clone();
    world.touch.push(Touch::new(SLAB, [HELD]));
    let touching = holding_left.upload(&[world]).unwrap();
    let q = two.default_q().to_vec();
    let problems = [
        PlanProblem {
            holding: Some(left),
            ..PlanProblem::new(0, q.clone(), q.clone())
        },
        PlanProblem {
            holding: Some(Held {
                hand: 1,
                ..cube(0.0)
            }),
            ..PlanProblem::new(0, q.clone(), q)
        },
    ];
    let e = batchplan::plan(&both, &touching, &problems, &PlanOptions::default()).unwrap_err();
    input(e, "uploaded for a robot holding");
}
