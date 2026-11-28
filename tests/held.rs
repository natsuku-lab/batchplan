//! A held object per problem (`PlanProblem::holding`, `IkProblem::holding`, `ToolPath::holding`):
//! one call takes problems holding nothing, different objects and one object by different grasps,
//! and gives each what a device holding its object alone gives.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::FRAC_PI_2;

use batchplan::{
    AttachedObject, Contact, Device, Error, Held, IkOptions, IkProblem, Obstacle, PlanFailure,
    PlanOptions, PlanProblem, Pose, Robot, RobotOptions, SdfGrid, SphereOptions, ToolPath,
    ToolPathOptions, ToolPathStop, Touch, World, Worlds,
};
use glam::{Quat, Vec3};

const SLAB: usize = 0;
const HELD: &str = "held by hand 0";

/// A 10 cm cube held with the IK frame `inset` from its middle toward the hand.
fn cube(inset: f32) -> Held {
    Held {
        touch_links: vec![],
        also: vec![],
        locked: Default::default(),
        spheres: Default::default(),
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::splat(0.05)),
        grasp: Pose::new(Vec3::new(0.0, 0.0, -inset), Quat::IDENTITY),
    }
}

/// A 24 cm bar held across the hand, level, by its middle.
fn bar() -> Held {
    Held {
        touch_links: vec![],
        also: vec![],
        locked: Default::default(),
        spheres: Default::default(),
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
    for device in common::devices(&s.robot) {
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
    for device in common::devices(&s.robot) {
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
    for device in common::devices(&s.robot) {
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
                    hold: vec![],
                    holding,
                })
                .collect();
            let all = batchplan::follow(&device, &worlds, &paths, &o).unwrap();
            for (i, p) in paths.iter().enumerate() {
                assert_eq!(all.problems[i].holding, p.holding, "{name}: path {i}");
                let holder = s.holder(&device, p.holding.as_ref());
                let alone = ToolPath {
                    hold: vec![],
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
    for device in common::devices(&robot) {
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

/// A world uploaded to the device that holds nothing lets what a hand comes to hold touch an
/// obstacle (`Touch::held`): a problem holding there is checked as on a device holding its object,
/// with the world uploaded to that one letting the held link touch it.
#[test]
fn a_world_lets_what_a_hand_comes_to_hold_touch_an_obstacle() {
    let s = Scene::new();
    let o = IkOptions::default();
    let held = cube(0.0);
    let resting = World {
        touch: vec![Touch::held(SLAB, 0)],
        ..s.world.clone()
    };
    for device in common::devices(&s.robot) {
        let name = device.name();
        // World 0 as it was, world 1 with the cube free to rest on the slab.
        let worlds = device.upload(&[s.world.clone(), resting.clone()]).unwrap();
        let at_ready = |world: u32, holding: Option<Held>| IkProblem {
            seed: Some(s.up.clone()),
            holding,
            ..IkProblem::new(world, s.robot.ee_pose(&s.ready, 0))
        };
        let problems = [
            at_ready(0, Some(held.clone())),
            at_ready(1, Some(held.clone())),
            at_ready(1, None),
            at_ready(0, None),
        ];
        let all = batchplan::solve_ik(&device, &worlds, &problems, &o).unwrap();
        assert!(all.best(0).is_none(), "{name}: the cube meets the slab");
        assert!(
            all.best(1).is_some(),
            "{name}: the cube may rest on the slab"
        );
        // Nothing is held where nothing is: the two worlds are one to the bare hand.
        let (n, seeds) = (all.dof, all.seeds);
        let of = |i: usize| &all.q[i * seeds * n..(i + 1) * seeds * n];
        assert!(all.best(2).is_some(), "{name}");
        assert_eq!(of(2), of(3), "{name}");

        // As the device holding the cube solves it, in the world uploaded to that one.
        let holder = s.holder(&device, Some(&held));
        let named = World {
            touch: vec![Touch::new(SLAB, [HELD])],
            ..s.world.clone()
        };
        let on_holder = holder.upload(&[s.world.clone(), named]).unwrap();
        let one = batchplan::solve_ik(&holder, &on_holder, &[at_ready(1, None)], &o).unwrap();
        assert_eq!(of(1), &one.q[..], "{name}");
        let items = seeds..2 * seeds;
        assert_eq!(all.success[items.clone()], one.success[..], "{name}");
        assert_eq!(
            all.world_clearance[items],
            one.world_clearance[..],
            "{name}"
        );

        // A plan and a straight move holding the cube end where it rests, in that world alone.
        let down = |world: u32| PlanProblem {
            holding: Some(held.clone()),
            ..PlanProblem::new(world, s.up.clone(), s.ready.clone())
        };
        let plans = [down(0), down(1)];
        let planned = batchplan::plan(&device, &worlds, &plans, &PlanOptions::default()).unwrap();
        let solved: Vec<usize> = planned.solved().map(|p| p.index).collect();
        assert_eq!(solved, [1], "{name}");
        let lower = |world: u32| ToolPath {
            world,
            start: s.up.clone(),
            hand: 0,
            waypoints: vec![s.robot.ee_pose(&s.ready, 0)],
            hold: vec![],
            holding: Some(held.clone()),
        };
        let paths = [lower(0), lower(1)];
        let o = ToolPathOptions::default();
        let followed = batchplan::follow(&device, &worlds, &paths, &o).unwrap();
        assert!(
            followed.fraction[0] < 1.0,
            "{name}: {:?}",
            followed.fraction
        );
        assert_eq!(
            followed.fraction[1], 1.0,
            "{name}: {:?}",
            followed.stopped[1]
        );
    }
}

/// One hand's held object per upload, and a problem that holds in another is refused.
#[test]
fn worlds_let_one_hands_held_object_touch_obstacles() {
    let two_hands = RobotOptions {
        ee_links: vec!["panda_hand".into(), "panda_link4".into()],
        ..common::panda_options()
    };
    let robot = Robot::load(common::asset("franka/franka_panda.urdf"), &two_hands).unwrap();
    let device = Device::cpu(&robot).unwrap();
    let slab = Obstacle::cuboid(Vec3::new(0.5, 0.0, 0.2), Vec3::new(0.15, 0.15, 0.005));
    let resting = |hand: usize| World {
        touch: vec![Touch::held(0, hand)],
        ..World::new(vec![slab.clone()])
    };
    let input = |e: Error, says: &str| {
        assert!(
            matches!(e, Error::Input(_)) && e.to_string().contains(says),
            "{e}"
        );
    };
    // Each hand's alone uploads; both in one upload, or in one world, do not.
    for hand in 0..2 {
        device.upload(&[resting(hand), resting(hand)]).unwrap();
    }
    input(
        device.upload(&[resting(0), resting(1)]).unwrap_err(),
        "world 1",
    );
    let both = World {
        touch: vec![Touch::held(0, 0), Touch::held(0, 1)],
        ..World::new(vec![slab.clone()])
    };
    input(device.upload(&[both]).unwrap_err(), "hands 0 and 1");
    // A hand the robot lacks names no link.
    input(device.upload(&[resting(2)]).unwrap_err(), "held by hand 2");
    // Worlds for what hand 0 holds refuse a problem holding in hand 1.
    let worlds = device.upload(&[resting(0)]).unwrap();
    let q = robot.default_q().to_vec();
    let problem = PlanProblem {
        holding: Some(Held { hand: 1, ..bar() }),
        ..PlanProblem::new(0, q.clone(), q)
    };
    let refused = batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default());
    input(refused.unwrap_err(), "held by hand 0");
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
    let no_spheres = Held {
        spheres: SphereOptions {
            budget: 0,
            ..Default::default()
        },
        ..bar()
    };
    input(plan(&device, &worlds, no_spheres), "spheres");
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
    input(e, "let 'held by hand 0' touch obstacles");
}

/// A hold says where the fingers closed (`Held::locked`), and every call that holds the object
/// checks the robot so: a ball where the open left finger is, and the closed one is not, stops a
/// path that holds nothing narrower, lets one held closed through, and the closed hold gives what a
/// device for the robot relocked there gives.
#[test]
fn a_hold_checks_the_fingers_where_they_closed() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let closed: std::collections::HashMap<String, f32> = [
        ("panda_finger_joint1".to_string(), 0.005),
        ("panda_finger_joint2".to_string(), 0.005),
    ]
    .into();
    let narrow = robot.relock(&closed).unwrap();
    // Balls on a grid around the open left finger; the first the open robot touches and the
    // closed one does not.
    let finger = robot.link_pose(&q, "panda_leftfinger").unwrap().position;
    let grid = (0..343).map(|i| {
        let at = |k: usize| (k as f32 - 3.0) * 0.01;
        finger + Vec3::new(at(i % 7), at(i / 7 % 7), at(i / 49))
    });
    let balls: Vec<Vec3> = grid.collect();
    let worlds_of: Vec<World> = balls
        .iter()
        .map(|&b| World::new(vec![Obstacle::sphere(b, 0.006)]))
        .collect();
    let free = |robot: &Robot| {
        let device = Device::cpu(robot).unwrap();
        let worlds = device.upload(&worlds_of).unwrap();
        let items: Vec<u32> = (0..balls.len() as u32).collect();
        let configurations: Vec<f32> = items.iter().flat_map(|_| q.iter().copied()).collect();
        let e = device
            .evaluate(
                &worlds,
                &items,
                &configurations,
                &batchplan::CollisionWeights::NONE,
            )
            .unwrap();
        (0..balls.len())
            .map(|i| e.collision_free(i))
            .collect::<Vec<bool>>()
    };
    let (open, shut) = (free(&robot), free(&narrow));
    let ball = (0..balls.len())
        .find(|&i| !open[i] && shut[i])
        .map(|i| balls[i])
        .expect("a ball only the open finger touches");
    let small = Held {
        touch_links: vec![],
        also: vec![],
        locked: Default::default(),
        spheres: Default::default(),
        hand: 0,
        object: Obstacle::sphere(Vec3::ZERO, 0.004),
        grasp: Pose::IDENTITY,
    };
    let held_closed = Held {
        locked: closed,
        ..small.clone()
    };
    let hand = robot.ee_pose(&q, 0);
    let up = Pose::new(hand.position + Vec3::Z * 0.02, hand.rotation);
    let path = |holding: Held| ToolPath {
        world: 0,
        start: q.clone(),
        hand: 0,
        waypoints: vec![up],
        hold: vec![],
        holding: Some(holding),
    };
    let o = ToolPathOptions::default();
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device
            .upload(&[World::new(vec![Obstacle::sphere(ball, 0.006)])])
            .unwrap();
        let both = [path(small.clone()), path(held_closed.clone())];
        let result = batchplan::follow(&device, &worlds, &both, &o).unwrap();
        assert!(
            matches!(result.stopped[0], Some(ToolPathStop::Collision { .. })),
            "{name}: open, {:?}",
            result.stopped[0]
        );
        assert_eq!(
            result.fraction[1], 1.0,
            "{name}: closed, {:?}",
            result.stopped[1]
        );
        let relocked = device.with_robot(&narrow).unwrap();
        let alone = batchplan::follow(&relocked, &worlds, &[path(small.clone())], &o).unwrap();
        assert_eq!(
            result.paths[1], alone.paths[0],
            "{name}: as the relocked robot"
        );
        let wrong = Held {
            locked: [("panda_joint1".to_string(), 0.0)].into(),
            ..small.clone()
        };
        match batchplan::follow(&device, &worlds, &[path(wrong)], &o) {
            Err(Error::Input(message)) => {
                assert!(
                    message.contains("path 0") && message.contains("locked"),
                    "{name}: {message}"
                );
            }
            other => panic!("{name}: a joint not locked is refused, got {other:?}"),
        }
    }
}

/// A joint that carries a hand (a torso, a turntable) is refused in a hold: the calls that hold an
/// object solve some steps on the robot as loaded.
#[test]
fn a_hold_cannot_move_a_joint_that_carries_the_hand() {
    let mut options = common::panda_options();
    options.lock_joints.insert("panda_joint1".to_string(), 0.0);
    options.default_q = options.default_q.map(|q| q[1..].to_vec());
    let robot = Robot::load(common::asset("franka/franka_panda.urdf"), &options).unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let held = Held {
        locked: [("panda_joint1".to_string(), 0.8)].into(),
        ..cube(0.0)
    };
    let q = robot.default_q().to_vec();
    let problem = PlanProblem {
        holding: Some(held),
        ..PlanProblem::new(0, q.clone(), q)
    };
    match batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default()) {
        Err(Error::Input(message)) => assert!(
            message.contains("panda_joint1") && message.contains("carries hand 0"),
            "{message}"
        ),
        other => panic!("a torso is refused: {other:?}"),
    }
}

/// A plate slid over a table away from a ball 6 mm from its side, where its spheres for a
/// tolerance of 12 mm reach farthest past it: held with those, which reach farther than the gap,
/// it reads as against the ball and does not move; held with spheres that stay within the gap
/// (`Held::spheres`), the default's within 5 mm whatever their budget or a limit on how far they
/// reach, it goes.
#[test]
fn a_held_plate_passes_a_neighbor_a_few_millimeters_away_with_finer_spheres() {
    let robot = common::panda().unwrap();
    let ready = robot.default_q().to_vec();
    let down = robot.ee_pose(&ready, 0).rotation;
    let (radius, half_height, top) = (0.0695, 0.009, 0.2);
    let at = Vec3::new(0.45, 0.0, top + half_height);
    let table = Obstacle::cuboid(Vec3::new(0.5, 0.0, top - 0.02), Vec3::new(0.4, 0.4, 0.02));
    let cylinder = Obstacle::Cylinder {
        center: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        radius,
        half_height,
    };
    let coarse = SphereOptions {
        tolerance: 0.012,
        ..Default::default()
    };
    // Where the coarse spheres reach farthest past the plate's side, which a hand holds turned
    // as it lies.
    let attached = robot
        .attach(&AttachedObject {
            name: "plate".into(),
            link: "panda_hand".into(),
            shapes: vec![cylinder.clone()],
            touch_links: vec![],
            spheres: coarse,
        })
        .unwrap();
    let past_side = |s: &[f32; 4]| Vec3::new(s[0], s[1], 0.0).length() + s[3] - radius;
    let farthest = attached.collision_model().spheres["plate"]
        .iter()
        .max_by(|a, b| past_side(a).total_cmp(&past_side(b)))
        .copied()
        .expect("the plate has spheres");
    assert!(past_side(&farthest) > 0.006, "{farthest:?}");
    let toward = Vec3::new(farthest[0], farthest[1], 0.0).normalize();
    let beside = Obstacle::sphere(at + toward * (radius + 0.006 + 0.03), 0.03);
    let world = World {
        touch: vec![Touch::held(0, 0)],
        ..World::new(vec![table, beside])
    };
    let hand = Pose::new(at + Vec3::Z * 0.06, down);
    let plate = |spheres: SphereOptions| Held {
        touch_links: vec![],
        also: vec![],
        hand: 0,
        object: cylinder.clone(),
        grasp: Pose::new(hand.position - at, hand.rotation),
        locked: Default::default(),
        spheres,
    };
    // Within the 5 mm the default asks for, which leave room for where IK puts the hand (to 1 mm
    // and 0.01 rad).
    let fine = SphereOptions::default();
    for device in common::devices(&robot) {
        let name = device.name();
        let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
        let reach = IkProblem {
            seed: Some(ready.clone()),
            holding: Some(plate(fine)),
            ..IkProblem::new(0, hand)
        };
        let solved =
            batchplan::solve_ik(&device, &worlds, &[reach], &IkOptions::default()).unwrap();
        let start = solved.nearest(0, &ready).expect("over the plate").to_vec();
        let slide = |spheres: SphereOptions| ToolPath {
            world: 0,
            start: start.clone(),
            hand: 0,
            waypoints: vec![Pose::new(hand.position - toward * 0.05, hand.rotation)],
            hold: vec![],
            holding: Some(plate(spheres)),
        };
        // Or held within 4 mm of it, which leaves some of it outside them.
        let held_in = SphereOptions {
            max_overhang: Some(0.004),
            ..Default::default()
        };
        let paths = [slide(coarse), slide(fine), slide(held_in)];
        let followed =
            batchplan::follow(&device, &worlds, &paths, &ToolPathOptions::default()).unwrap();
        assert_eq!(followed.fraction[0], 0.0, "{name}");
        let against =
            |c: &Contact| matches!(c, Contact::World { link, obstacle: 1, .. } if link == HELD);
        assert!(
            followed.contacts[0].iter().any(against),
            "{name}: {:?}",
            followed.contacts[0]
        );
        for path in [1, 2] {
            assert_eq!(
                followed.fraction[path], 1.0,
                "{name}: {:?} {:?}",
                followed.stopped[path], followed.contacts[path]
            );
        }
    }
}

/// What a configuration touches holding an object, asked of the device that holds nothing: what
/// a device built by attaching the object lists, each item with its own hold, in the items'
/// order; and what the world lets the held object touch is not a contact.
#[test]
fn contacts_are_listed_for_what_each_configuration_holds() {
    let s = Scene::new();
    let (cube, bar) = (cube(0.0), bar());
    let resting = World {
        touch: vec![Touch::held(SLAB, 0)],
        ..s.world.clone()
    };
    for device in common::devices(&s.robot) {
        let name = device.name();
        let worlds = device.upload(&[s.world.clone(), resting.clone()]).unwrap();
        // At the ready pose the hand is clear of the slab and a cube held there is in it.
        let holds = [
            None,
            Some(&cube),
            Some(&bar),
            Some(&cube),
            None,
            Some(&cube),
        ];
        let q: Vec<f32> = [&s.ready, &s.ready, &s.ready, &s.up, &s.up, &s.ready]
            .iter()
            .flat_map(|q| q.iter().copied())
            .collect();
        let item_world = [0, 0, 0, 0, 0, 1];
        let found = device
            .contacts_holding(&worlds, &item_world, &q, 0.0, &holds)
            .unwrap();
        assert_eq!(found.len(), holds.len(), "{name}");
        assert!(found[0].is_empty(), "{name}: {:?}", found[0]);
        assert!(held_at_slab(&found[1]), "{name}: {:?}", found[1]);
        assert!(
            found[3].is_empty() && found[4].is_empty(),
            "{name}: lifted clear"
        );
        assert!(found[5].is_empty(), "{name}: the world lets it rest there");
        // Each as a device holding its object alone lists it.
        let n = s.robot.dof();
        for (i, hold) in holds.iter().enumerate() {
            let holder = s.holder(&device, *hold);
            let alone = holder
                .contacts(&worlds, &item_world[i..=i], &q[i * n..(i + 1) * n], 0.0)
                .unwrap();
            assert_eq!(found[i], alone[0], "{name}: item {i}");
        }
        assert_ne!(found[1], found[2], "{name}: the cube and the bar differ");
        // As many holds as items, and holds the robot can take, named by their item.
        let input = |e: Error, says: &str| {
            assert!(
                matches!(e, Error::Input(_)) && e.to_string().contains(says),
                "{name}: {e}"
            );
        };
        let short = device.contacts_holding(&worlds, &item_world, &q, 0.0, &holds[..2]);
        input(short.unwrap_err(), "2 holds for 6 items");
        let other_hand = Held {
            hand: 1,
            ..bar.clone()
        };
        let wrong = [None, Some(&other_hand)];
        let refused = device.contacts_holding(&worlds, &[0, 0], &q[..2 * n], 0.0, &wrong);
        input(refused.unwrap_err(), "item 1");
        // A configuration that is not finite is named by its item, whichever lot it is in.
        let mut broken = q.clone();
        broken[5 * n] = f32::NAN;
        let refused = device.contacts_holding(&worlds, &item_world, &broken, 0.0, &holds);
        input(refused.unwrap_err(), "item 5");
        let refused = device.contacts_holding(&worlds, &item_world, &q, f32::NAN, &holds);
        input(refused.unwrap_err(), "finite distance");
        // Holding nothing anywhere, it is `contacts`.
        let nothing = [None; 6];
        assert_eq!(
            device
                .contacts_holding(&worlds, &item_world, &q, 0.01, &nothing)
                .unwrap(),
            device.contacts(&worlds, &item_world, &q, 0.01).unwrap(),
            "{name}"
        );
        assert!(
            device
                .contacts_holding(&worlds, &[], &[], 0.0, &[])
                .unwrap()
                .is_empty()
        );
    }
}

/// Problems that hold objects in many ways are planned on a device for each, and what runs on
/// one waits for work it hands the device's threads, where a thread that waits takes up
/// another's on top of its own: four thousand ways to hold a cube, two plans holding it each
/// way, are planned in one call, the last as they are alone.
#[test]
fn as_many_holds_as_fill_a_stack_at_once_are_planned_in_one_call() {
    let s = Scene::new();
    let device = Device::cpu(&s.robot).unwrap();
    let worlds = device.upload(std::slice::from_ref(&s.world)).unwrap();
    let o = PlanOptions::default();
    const WAYS: usize = 4000;
    let holding = |way: usize| cube(0.05 + way as f32 * 1e-6);
    let across = |way: usize| {
        let there = PlanProblem {
            holding: Some(holding(way)),
            ..PlanProblem::new(0, s.up.clone(), s.aside.clone())
        };
        let back = PlanProblem {
            holding: Some(holding(way)),
            ..PlanProblem::new(0, s.aside.clone(), s.up.clone())
        };
        [there, back]
    };
    let problems: Vec<PlanProblem> = (0..WAYS).flat_map(across).collect();
    let planned = batchplan::plan(&device, &worlds, &problems, &o).unwrap();
    assert_eq!(planned.solved().count(), 2 * WAYS);
    let alone = batchplan::plan(&device, &worlds, &across(WAYS - 1), &o).unwrap();
    let among: Vec<_> = planned.solved().skip(2 * WAYS - 2).collect();
    for (a, b) in among.iter().zip(alone.solved()) {
        assert_eq!(a.solution, b.solution);
    }
}
