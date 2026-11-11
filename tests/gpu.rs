//! The GPU device against the CPU device, plus end-to-end validity of GPU plans.
//! Skipped when no adapter is available unless `BATCHPLAN_REQUIRE_GPU=1`.
#![cfg(feature = "gpu")]

#[path = "../examples/common/mod.rs"]
mod common;

use std::sync::Arc;

use batchplan::rng::Rng;
use batchplan::{
    AttachedObject, CollisionWeights, Device, Error, HandConstraint, IkOptions, IkProblem,
    Obstacle, PlanOptions, PlanProblem, Robot, RobotOptions, SdfGrid, Solved, SphereOptions, Touch,
    Trajectory, World,
};
use glam::{EulerRot, Quat, Vec2, Vec3};

fn setup() -> Option<(Robot, Device, Device)> {
    let robot = common::panda().unwrap();
    match Device::gpu(&robot) {
        Ok(gpu) => {
            eprintln!("using {}", gpu.name());
            let cpu = Device::cpu(&robot).unwrap();
            Some((robot, gpu, cpu))
        }
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => {
            eprintln!("skipping GPU test: {e}");
            None
        }
        Err(e) => panic!("no GPU: {e}"),
    }
}

#[test]
fn unknown_adapter_is_a_clear_error() {
    let robot = common::panda().unwrap();
    let err = Device::gpu_named(&robot, "no-such-gpu").expect_err("should fail");
    assert!(
        matches!(&err, Error::Gpu(m) if m.contains("no-such-gpu")),
        "{err:?}"
    );
}

fn worlds(n: usize, seed: u64) -> Vec<World> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| common::tabletop(&mut rng)).collect()
}

fn ik_problems(worlds: &[World], seed: u64) -> Vec<IkProblem> {
    let mut rng = Rng::new(seed);
    worlds
        .iter()
        .enumerate()
        .map(|(i, s)| IkProblem::new(i as u32, common::grasp_target(s, &mut rng)))
        .collect()
}

#[test]
fn evaluate_matches_cpu() {
    let Some((robot, gpu, cpu)) = setup() else {
        return;
    };
    let n = robot.dof();
    let scene = worlds(16, 3);
    let (on_cpu, on_gpu) = (cpu.upload(&scene).unwrap(), gpu.upload(&scene).unwrap());
    let mut rng = Rng::new(9);
    let items = 20_000;
    let q: Vec<f32> = (0..items)
        .flat_map(|_| {
            (0..n)
                .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
                .collect::<Vec<_>>()
        })
        .collect();
    let item_world: Vec<u32> = (0..items as u32).map(|i| i % 16).collect();
    let w = CollisionWeights {
        world: 1000.0,
        self_collision: 1000.0,
        margin: 0.02,
        self_margin: 0.01,
    };
    let a = cpu.evaluate(&on_cpu, &item_world, &q, &w).unwrap();
    let b = gpu.evaluate(&on_gpu, &item_world, &q, &w).unwrap();
    let mut worst = [0.0f32; 4];
    for i in 0..items {
        worst[0] = worst[0].max((a.world_clearance[i] - b.world_clearance[i]).abs());
        worst[1] = worst[1].max((a.self_clearance[i] - b.self_clearance[i]).abs());
        worst[2] = worst[2].max((a.cost[i] - b.cost[i]).abs() / a.cost[i].abs().max(1.0));
        // Gradient error relative to the gradient's norm (components cancel in f32).
        let row = |g: &[f32]| g[i * n..(i + 1) * n].to_vec();
        let (ga, gb) = (row(&a.gradient), row(&b.gradient));
        let norm = ga.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
        let diff = ga
            .iter()
            .zip(&gb)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
        worst[3] = worst[3].max(diff / norm);
    }
    eprintln!(
        "max |cpu-gpu|: world {:.2e} self {:.2e} cost(rel) {:.2e} grad(rel) {:.2e}",
        worst[0], worst[1], worst[2], worst[3]
    );
    let colliding = a.cost.iter().filter(|&&c| c > 0.0).count();
    assert!(
        colliding > items / 10,
        "only {colliding} of {items} configurations carry collision cost"
    );
    assert!(worst[0] < 1e-4 && worst[1] < 1e-4 && worst[2] < 1e-3 && worst[3] < 1e-3);
}

#[test]
fn evaluate_matches_cpu_for_every_obstacle_kind() {
    let Some((robot, gpu, cpu)) = setup() else {
        return;
    };
    let n = robot.dof();
    let rotation = Quat::from_euler(EulerRot::XYZ, 0.4, -0.7, 1.1);
    let center = Vec3::new(0.45, 0.0, 0.4);
    // Distance grids: a torus with an odd number of points (its last word holds one value) and a
    // box after it in the grid buffer; the last world places both, one grid shared with world 4.
    let sampled = |dims: [u32; 3], origin: Vec3, f: &dyn Fn(Vec3) -> f32| {
        let mut values = vec![];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    values.push(f(origin + Vec3::new(i as f32, j as f32, k as f32) * 0.01));
                }
            }
        }
        Arc::new(SdfGrid::try_new(dims, 0.01, origin, &values).unwrap())
    };
    let torus = sampled([41, 41, 21], Vec3::new(-0.2, -0.2, -0.1), &|p| {
        Vec2::new(p.truncate().length() - 0.15, p.z).length() - 0.05
    });
    let block = Obstacle::Cuboid {
        center: Vec3::ZERO,
        half_extents: Vec3::new(0.1, 0.05, 0.04),
        rotation: Quat::IDENTITY,
    };
    let boxed = sampled([30, 20, 16], Vec3::new(-0.15, -0.1, -0.08), &|p| {
        block.distance(p).value
    });
    let grid = |grid: &Arc<SdfGrid>, center: Vec3| Obstacle::Sdf {
        grid: grid.clone(),
        center,
        rotation,
    };
    let worlds: Vec<World> = [
        vec![Obstacle::Cuboid {
            center,
            half_extents: Vec3::new(0.1, 0.2, 0.08),
            rotation,
        }],
        vec![Obstacle::sphere(center, 0.15)],
        vec![Obstacle::Cylinder {
            center,
            rotation,
            radius: 0.1,
            half_height: 0.2,
        }],
        vec![Obstacle::Capsule {
            center,
            rotation,
            radius: 0.08,
            half_length: 0.15,
        }],
        vec![grid(&torus, center)],
        vec![
            grid(&boxed, center),
            grid(&torus, center + Vec3::new(0.0, 0.3, 0.1)),
        ],
    ]
    .into_iter()
    .map(World::new)
    .collect();
    let kinds = [
        "cuboid",
        "sphere",
        "cylinder",
        "capsule",
        "grid",
        "two grids",
    ];
    let mut rng = Rng::new(19);
    let items = 12_000;
    let q: Vec<f32> = (0..items * n)
        .map(|i| rng.range(robot.lower()[i % n], robot.upper()[i % n]))
        .collect();
    let item_world: Vec<u32> = (0..items as u32).map(|i| i % 6).collect();
    let w = CollisionWeights {
        world: 1000.0,
        self_collision: 0.0,
        margin: 0.02,
        self_margin: 0.0,
    };
    let a = cpu
        .evaluate(&cpu.upload(&worlds).unwrap(), &item_world, &q, &w)
        .unwrap();
    let b = gpu
        .evaluate(&gpu.upload(&worlds).unwrap(), &item_world, &q, &w)
        .unwrap();
    for (kind, name) in kinds.iter().enumerate() {
        let (mut colliding, mut worst_clearance, mut worst_grad) = (0, 0.0f32, 0.0f32);
        for i in (kind..items).step_by(6) {
            colliding += usize::from(a.cost[i] > 0.0);
            worst_clearance =
                worst_clearance.max((a.world_clearance[i] - b.world_clearance[i]).abs());
            let (ga, gb) = (
                &a.gradient[i * n..(i + 1) * n],
                &b.gradient[i * n..(i + 1) * n],
            );
            let norm = ga.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
            let diff = ga
                .iter()
                .zip(gb)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt();
            worst_grad = worst_grad.max(diff / norm);
        }
        eprintln!(
            "{name}: {colliding} colliding, clearance {worst_clearance:.2e}, grad {worst_grad:.2e}"
        );
        assert!(
            colliding > items / 6 / 20,
            "only {colliding} configurations touch the {name}"
        );
        assert!(
            worst_clearance < 1e-4 && worst_grad < 1e-3,
            "the {name} differs between devices"
        );
    }
}

#[test]
fn touches_past_the_first_word_match_cpu() {
    let Some((robot, gpu, cpu)) = setup() else {
        return;
    };
    // A ring of small balls held around the hand: the Panda's 12 links and 60 balls, so the touch
    // sets span two 64-bit words on the CPU (ball 52 on) and three 32-bit words on the GPU.
    let mut held = robot;
    for k in 0..60 {
        let angle = k as f32 * std::f32::consts::TAU / 60.0;
        let at = Vec3::new(0.12 * angle.cos(), 0.12 * angle.sin(), 0.05);
        held = held
            .attach(&AttachedObject {
                name: format!("ball {k}"),
                link: "panda_hand".into(),
                shapes: vec![Obstacle::sphere(at, 0.01)],
                touch_links: vec![],
                spheres: SphereOptions::default(),
            })
            .unwrap();
    }
    let (bare_gpu, bare_cpu) = (gpu, cpu);
    let (gpu, cpu) = (
        bare_gpu.with_robot(&held).unwrap(),
        bare_cpu.with_robot(&held).unwrap(),
    );
    let block = Obstacle::cuboid(Vec3::new(0.4, 0.0, 0.4), Vec3::new(0.25, 0.3, 0.2));
    // A ball as a distance grid beside it, sharing the world data with the touch sets.
    let (dims, origin) = ([21u32, 21, 21], Vec3::new(0.25, 0.25, 0.45));
    let mut values = vec![];
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let p = origin + Vec3::new(i as f32, j as f32, k as f32) * 0.01;
                values.push(p.distance(Vec3::new(0.35, 0.35, 0.55)) - 0.08);
            }
        }
    }
    let grid = Obstacle::from_grid(SdfGrid::try_new(dims, 0.01, origin, &values).unwrap());
    let mut links: Vec<String> = (40..60).map(|k| format!("ball {k}")).collect();
    links.push("panda_leftfinger".into());
    let worlds = [
        World::new(vec![block.clone(), grid.clone()]),
        World {
            touch: vec![
                Touch::new(0, links),
                Touch::new(1, ["ball 3", "ball 58", "panda_link7"]),
            ],
            ..World::new(vec![block.clone(), grid.clone()])
        },
    ];
    let n = held.dof();
    let mut rng = Rng::new(23);
    let items = 2000;
    let q: Vec<f32> = (0..items * n)
        .map(|i| rng.range(held.lower()[i % n], held.upper()[i % n]))
        .collect();
    let item_world: Vec<u32> = (0..items as u32).map(|i| i % 2).collect();
    let w = CollisionWeights {
        world: 1000.0,
        self_collision: 0.0,
        margin: 0.02,
        self_margin: 0.0,
    };
    let a = cpu
        .evaluate(&cpu.upload(&worlds).unwrap(), &item_world, &q, &w)
        .unwrap();
    let b = gpu
        .evaluate(&gpu.upload(&worlds).unwrap(), &item_world, &q, &w)
        .unwrap();
    for i in 0..items {
        let (ga, gb) = (
            &a.gradient[i * n..(i + 1) * n],
            &b.gradient[i * n..(i + 1) * n],
        );
        let norm = ga.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
        let diff = ga
            .iter()
            .zip(gb)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            (a.world_clearance[i] - b.world_clearance[i]).abs() < 1e-4 && diff / norm < 1e-3,
            "item {i}: clearance {} vs {}",
            a.world_clearance[i],
            b.world_clearance[i]
        );
    }
    // The touches matter: the same configurations clear by more in the second world.
    let same_q = vec![1; items];
    let touching = cpu
        .evaluate(&cpu.upload(&worlds).unwrap(), &same_q, &q, &w)
        .unwrap();
    let plain = cpu
        .evaluate(&cpu.upload(&worlds).unwrap(), &vec![0; items], &q, &w)
        .unwrap();
    let freed = (0..items)
        .filter(|&i| touching.world_clearance[i] > plain.world_clearance[i])
        .count();
    assert!(freed > 20, "the touches change only {freed} clearances");
    // So do the grid's alone, on links in either word of the second obstacle's set, with the
    // block out of reach.
    let far = Obstacle::cuboid(Vec3::new(-3.0, 0.0, 0.0), Vec3::splat(0.1));
    let clearances = |touch: Vec<Touch>| {
        let world = World {
            touch,
            ..World::new(vec![far.clone(), grid.clone()])
        };
        let uploaded = cpu.upload(std::slice::from_ref(&world)).unwrap();
        cpu.evaluate(&uploaded, &vec![0; items], &q, &w)
            .unwrap()
            .world_clearance
    };
    let (alone, touched) = (
        clearances(vec![]),
        clearances(vec![Touch::new(1, ["ball 3", "ball 58", "panda_link7"])]),
    );
    let freed = (0..items).filter(|&i| touched[i] > alone[i]).count();
    assert!(
        freed > 5,
        "the grid's touches change only {freed} clearances"
    );
    // Worlds uploaded for the robot without the balls stay valid holding them, and the balls,
    // past the links their sets were built for, touch nothing.
    let hand = World {
        touch: vec![Touch::new(0, ["panda_leftfinger", "panda_hand"])],
        ..World::new(vec![block, grid])
    };
    let world = std::slice::from_ref(&hand);
    let each = vec![0; items];
    let for_held = cpu
        .evaluate(&cpu.upload(world).unwrap(), &each, &q, &w)
        .unwrap();
    for (device, bare) in [(&cpu, &bare_cpu), (&gpu, &bare_gpu)] {
        let from_bare = bare.upload(world).unwrap();
        let e = device.evaluate(&from_bare, &each, &q, &w).unwrap();
        for i in 0..items {
            assert!(
                (e.world_clearance[i] - for_held.world_clearance[i]).abs() < 1e-4,
                "{}: item {i}: {} vs {}",
                device.name(),
                e.world_clearance[i],
                for_held.world_clearance[i]
            );
        }
    }
    // A robot holding another object where the balls were cannot use the bare robot's held
    // worlds.
    let other = bare_cpu
        .robot()
        .attach(&AttachedObject {
            name: "cup".into(),
            link: "panda_hand".into(),
            shapes: vec![Obstacle::sphere(Vec3::Z * 0.1, 0.03)],
            touch_links: vec![],
            spheres: SphereOptions::default(),
        })
        .unwrap();
    let cup = bare_cpu.with_robot(&other).unwrap();
    let held_worlds = cpu.upload(&worlds).unwrap();
    assert!(matches!(
        cup.evaluate(&held_worlds, &[1], &q[..n], &w),
        Err(Error::Input(_))
    ));
}

/// A small batch gives each configuration several lanes, which split its collision gradient and
/// step as one: its seeds end where they do in a batch large enough for one lane each, up to the
/// lanes' rounding.
#[test]
fn ik_lanes_step_as_one_configuration() {
    let Some((robot, gpu, _)) = setup() else {
        return;
    };
    let n = robot.dof();
    let scene = worlds(8, 4);
    let few = ik_problems(&scene, 5);
    // The same problems again and again: enough that neither the screening nor the seeds that go
    // on are few enough for lanes.
    let many: Vec<IkProblem> = few.iter().cycle().take(400).cloned().collect();
    let o = IkOptions::default();
    let on_gpu = gpu.upload(&scene).unwrap();
    let laned = batchplan::solve_ik(&gpu, &on_gpu, &few, &o).unwrap();
    let single = batchplan::solve_ik(&gpu, &on_gpu, &many, &o).unwrap();
    let seeds = few.len() * o.seeds;
    let apart = |i: usize| {
        let (a, b) = (laned.solution(i), single.solution(i));
        (0..n).map(|j| (a[j] - b[j]).abs()).fold(0.0, f32::max)
    };
    // The push away from obstacles moved some of them: without it the lanes would have nothing to
    // split.
    let unpushed = IkOptions {
        collision_step: 0.0,
        ..o
    };
    let plain = batchplan::solve_ik(&gpu, &on_gpu, &few, &unpushed).unwrap();
    let pushed = (0..seeds)
        .filter(|&i| laned.solution(i) != plain.solution(i))
        .count();
    let together = (0..seeds).filter(|&i| apart(i) < 1e-2).count();
    eprintln!("{pushed} of {seeds} seeds pushed, {together} end within 0.01 rad");
    assert!(
        pushed > seeds / 4,
        "only {pushed} of {seeds} seeds are pushed"
    );
    assert!(
        together * 100 >= seeds * 95,
        "only {together} of {seeds} seeds end where one lane each puts them"
    );
}

/// A problem's seeds share its targets, and a batch goes to the GPU in chunks of 65,536 items that
/// begin partway through a problem: every seed that succeeds reaches its own problem's target.
#[test]
fn ik_seeds_past_a_chunk_reach_their_own_targets() {
    let Some((robot, gpu, _)) = setup() else {
        return;
    };
    // 34 problems with targets of their own: the second chunk begins 1,536 seeds into the 17th
    // and runs on through the 33rd.
    let scene = worlds(34, 4);
    let problems = ik_problems(&scene, 5);
    // Every seed takes every step: chunks of seeds being screened hold whole problems.
    let o = IkOptions {
        seeds: 4000,
        iterations: 10,
        screening: 0,
        ..Default::default()
    };
    assert!(
        problems.len() * o.seeds > 2 * 65_536 && 65_536 % o.seeds != 0,
        "no chunk begins partway through a problem and goes on into the next"
    );
    let on_gpu = gpu.upload(&scene).unwrap();
    let result = batchplan::solve_ik(&gpu, &on_gpu, &problems, &o).unwrap();
    let mut past = 0;
    let succeeded = (0..result.success.len()).filter(|&i| result.success[i]);
    for i in succeeded {
        let target = problems[i / o.seeds].targets[0].expect("one hand");
        let reached = robot.ee_pose(result.solution(i), 0).position;
        assert!(
            (reached - target.position).length() < o.position_tolerance * 1.01,
            "seed {i} of problem {} ends {reached} from its target {}",
            i / o.seeds,
            target.position
        );
        past += usize::from(i >= 65_536);
    }
    assert!(past > 100, "only {past} seeds past the first chunk succeed");
}

/// Seeds are chosen among a problem's, so a batch being screened goes to the GPU in chunks of
/// whole problems: past the first chunk too, no problem sends on more seeds than are kept.
#[test]
fn ik_screens_whole_problems_past_a_chunk() {
    let Some((robot, gpu, _)) = setup() else {
        return;
    };
    let n = robot.dof();
    let scene = worlds(34, 4);
    let problems = ik_problems(&scene, 5);
    let o = IkOptions {
        seeds: 4000,
        iterations: 10,
        ..Default::default()
    };
    assert!(
        o.screening > 0 && o.screening < o.iterations && o.kept < o.seeds,
        "nothing is screened"
    );
    assert!(
        problems.len() * o.seeds > 2 * 65_536 && 65_536 % o.seeds != 0,
        "chunks of 65,536 seeds would not split a problem"
    );
    let on_gpu = gpu.upload(&scene).unwrap();
    let solve = |o: &IkOptions| batchplan::solve_ik(&gpu, &on_gpu, &problems, o).unwrap();
    let screened = solve(&IkOptions {
        iterations: o.screening,
        screening: 0,
        ..o
    });
    let staged = solve(&o);
    let mut went_on = 0;
    for p in 0..problems.len() {
        let seeds = p * o.seeds..(p + 1) * o.seeds;
        let moved = seeds
            .filter(|&i| staged.q[i * n..(i + 1) * n] != screened.q[i * n..(i + 1) * n])
            .count();
        assert!(moved <= o.kept, "problem {p} sent {moved} seeds on");
        went_on += moved;
    }
    assert!(went_on > problems.len(), "only {went_on} seeds went on");
}

#[test]
fn ik_matches_cpu() {
    let Some((robot, gpu, cpu)) = setup() else {
        return;
    };
    let scene = worlds(64, 4);
    let problems = ik_problems(&scene, 5);
    let o = IkOptions::default();
    let (on_cpu, on_gpu) = (cpu.upload(&scene).unwrap(), gpu.upload(&scene).unwrap());
    let a = batchplan::solve_ik(&cpu, &on_cpu, &problems, &o).unwrap();
    let b = batchplan::solve_ik(&gpu, &on_gpu, &problems, &o).unwrap();
    let agree = a
        .success
        .iter()
        .zip(&b.success)
        .filter(|(x, y)| x == y)
        .count();
    let (sa, sb) = (
        a.success.iter().filter(|&&s| s).count(),
        b.success.iter().filter(|&&s| s).count(),
    );
    eprintln!(
        "seed successes cpu {sa} gpu {sb}, agreement {agree}/{}",
        a.success.len()
    );
    assert!(
        sa as f32 > 0.2 * a.success.len() as f32,
        "too few IK successes ({sa}) to compare"
    );
    assert!(agree as f32 >= 0.97 * a.success.len() as f32);
    // Every GPU success must be a real solution according to the CPU model.
    let eval = cpu
        .evaluate(
            &on_cpu,
            &(0..b.success.len())
                .map(|i| (i / o.seeds) as u32)
                .collect::<Vec<_>>(),
            &b.q,
            &CollisionWeights::NONE,
        )
        .unwrap();
    let succeeded = (0..b.success.len()).filter(|&i| b.success[i]);
    for i in succeeded {
        let target = problems[i / o.seeds].targets[0].expect("one hand");
        assert!(
            (robot.ee_pose(b.solution(i), 0).position - target.position).length()
                < o.position_tolerance * 1.01
        );
        assert!(eval.collision_free(i));
    }
}

/// One IK step of hands that share joints, which IK solves as one stacked system, matches the
/// CPU's element-wise: the Panda's hand and a second hand at its fourth link, on tabletops whose
/// collision push moves the arm too.
#[test]
fn ik_steps_of_hands_sharing_joints_match_cpu() {
    if setup().is_none() {
        return;
    }
    let options = RobotOptions {
        ee_links: vec!["panda_hand".into(), "panda_link4".into()],
        ..common::panda_options()
    };
    let robot = Robot::load(common::asset("franka/franka_panda.urdf"), &options).unwrap();
    let (gpu, cpu) = (Device::gpu(&robot).unwrap(), Device::cpu(&robot).unwrap());
    let scene = worlds(32, 6);
    let mut rng = Rng::new(7);
    let random = |rng: &mut Rng| -> Vec<f32> {
        (0..robot.dof())
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect()
    };
    let problems: Vec<IkProblem> = (0..scene.len() as u32)
        .map(|world| {
            let aim = random(&mut rng);
            IkProblem {
                world,
                targets: vec![Some(robot.ee_pose(&aim, 0)), Some(robot.ee_pose(&aim, 1))],
                seed: Some(random(&mut rng)),
                hold: vec![],
                holding: None,
            }
        })
        .collect();
    // The GPU's rounding grows with the step (about 1.7e-4 of it): a 0.3 rad step keeps the bound.
    let o = IkOptions {
        seeds: 1,
        iterations: 1,
        max_step: 0.3,
        ..IkOptions::default()
    };
    let (on_cpu, on_gpu) = (cpu.upload(&scene).unwrap(), gpu.upload(&scene).unwrap());
    let a = batchplan::solve_ik(&cpu, &on_cpu, &problems, &o).unwrap();
    let b = batchplan::solve_ik(&gpu, &on_gpu, &problems, &o).unwrap();
    let seeds = problems.iter().flat_map(|p| p.seed.clone().expect("given"));
    let moved =
        a.q.iter()
            .zip(seeds)
            .fold(0.0f32, |m, (x, s)| m.max((x - s).abs()));
    let apart =
        a.q.iter()
            .zip(&b.q)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    eprintln!("one step moves joints up to {moved:.3} rad; |cpu-gpu| at most {apart:.2e}");
    assert!(moved > 0.05, "the step moves the arm ({moved})");
    assert!(apart < 1e-4, "one IK step differs by {apart} rad");
}

#[test]
fn gpu_plans_are_collision_free_under_cpu_check() {
    let Some((robot, gpu, cpu)) = setup() else {
        return;
    };
    let n = robot.dof();
    let scene = worlds(128, 6);
    let problems = ik_problems(&scene, 7);
    let (on_cpu, on_gpu) = (cpu.upload(&scene).unwrap(), gpu.upload(&scene).unwrap());
    let ik = batchplan::solve_ik(&gpu, &on_gpu, &problems, &IkOptions::default()).unwrap();
    let plan_problems: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution))
        .collect();
    let o = PlanOptions::default();
    let result = batchplan::plan(&gpu, &on_gpu, &plan_problems, &o).unwrap();
    let solved = result.solved().count();
    eprintln!(
        "ik solved {}/{}; planned {solved}/{}",
        plan_problems.len(),
        problems.len(),
        plan_problems.len()
    );
    assert!(
        plan_problems.len() >= problems.len() * 3 / 4,
        "IK solved only {} of {}",
        plan_problems.len(),
        problems.len()
    );
    assert!(solved as f32 >= 0.8 * plan_problems.len() as f32);

    // Independent check: the CPU model on the timed trajectory, sampled 4x denser per span than
    // the planner's validation.
    let per_span = (o.validate_substeps * 4) as f32;
    let (mut dense, mut dense_world) = (vec![], vec![]);
    for Solved {
        problem: prob,
        solution: cp,
        ..
    } in result.solved()
    {
        let trajectory = Trajectory::try_new(&robot, cp, 1.0).unwrap();
        trajectory.check(&robot).unwrap();
        let samples = trajectory
            .sample(per_span / trajectory.knot_interval)
            .unwrap();
        let (first, last) = (
            &samples.positions[..n],
            &samples.positions[samples.positions.len() - n..],
        );
        assert!(
            first == &prob.start[..] && last == prob.goal.joints().unwrap(),
            "the trajectory does not run start to goal"
        );
        dense.extend(&samples.positions);
        dense_world.extend(std::iter::repeat_n(prob.world, samples.len()));
    }
    let eval = cpu
        .evaluate(&on_cpu, &dense_world, &dense, &CollisionWeights::NONE)
        .unwrap();
    let worst = eval
        .world_clearance
        .iter()
        .chain(&eval.self_clearance)
        .fold(f32::INFINITY, |m, &v| m.min(v));
    eprintln!(
        "worst clearance along {} dense samples: {worst:.4} m",
        dense_world.len()
    );
    assert!(worst > -2e-3, "trajectory penetrates by {worst}");
}

/// The direction of each path's `step`-th L-BFGS step, scaled to its largest entry (the line
/// search only scales it). The first step has no history, so it is the negative gradient; later
/// ones come from the two-loop recursion. Also returns a mask of values a joint limit clamped
/// (those carry no direction information).
/// A ball where each problem's hand passes halfway along its straight line, so every search
/// detours and the problem is optimized; problems whose ends the ball touches are dropped. Each
/// problem has a world of its own.
fn block_straight_lines(
    robot: &Robot,
    scene: &mut [World],
    problems: Vec<PlanProblem>,
) -> Vec<PlanProblem> {
    for p in &problems {
        let middle: Vec<f32> = p
            .start
            .iter()
            .zip(p.goal.joints().unwrap())
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let center = robot.ee_pose(&middle, 0).position;
        scene[p.world as usize]
            .obstacles
            .push(Obstacle::sphere(center, 0.1));
    }
    let cpu = Device::cpu(robot).unwrap();
    let worlds = cpu.upload(scene).unwrap();
    let free = |p: &PlanProblem| {
        let ends = [&p.start[..], p.goal.joints().unwrap()].concat();
        let e = cpu
            .evaluate(&worlds, &[p.world, p.world], &ends, &CollisionWeights::NONE)
            .unwrap();
        e.collision_free(0) && e.collision_free(1)
    };
    problems.into_iter().filter(free).collect()
}

/// Every seed's control points back to back, `per_path` values each, a seed without a path as
/// zeros: the layout the step and cost comparisons index. No searched path here needs more control
/// points than the options give.
fn back_to_back(paths: Vec<Vec<f32>>, per_path: usize) -> Vec<f32> {
    let mut all = Vec::with_capacity(paths.len() * per_path);
    for path in paths {
        if path.is_empty() {
            all.resize(all.len() + per_path, 0.0);
        } else {
            assert_eq!(path.len(), per_path, "a searched path grew");
            all.extend(path);
        }
    }
    all
}

fn step_directions(
    d: &Device,
    worlds: &[World],
    problems: &[PlanProblem],
    o: &PlanOptions,
    step: u32,
) -> (Vec<f32>, Vec<bool>) {
    let worlds = d.upload(worlds).unwrap();
    let robot = d.robot();
    let n = robot.dof();
    let after = |iterations: u32| {
        let result =
            batchplan::plan(d, &worlds, problems, &PlanOptions { iterations, ..*o }).unwrap();
        back_to_back(result.paths, o.control_points * n)
    };
    let (seed, stepped) = (after(step - 1), after(step));
    let clamped = stepped
        .iter()
        .enumerate()
        .map(|(i, &q)| q <= robot.lower()[i % n] || q >= robot.upper()[i % n])
        .collect();
    let mut gradients: Vec<f32> = seed.iter().zip(&stepped).map(|(a, b)| a - b).collect();
    for path in gradients.chunks_mut(o.control_points * n) {
        let largest = path.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if largest > 0.0 {
            for v in path.iter_mut() {
                *v /= largest;
            }
        }
    }
    (gradients, clamped)
}

#[test]
fn trajopt_directions_match_cpu_element_wise() {
    let Some((robot, gpu, cpu)) = setup() else {
        return;
    };
    let mut worlds = worlds(64, 8);
    let ik = batchplan::solve_ik(
        &cpu,
        &cpu.upload(&worlds).unwrap(),
        &ik_problems(&worlds, 9),
        &common::fixture_ik(),
    )
    .unwrap();
    let reaches: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution))
        .collect();
    let problems = block_straight_lines(&robot, &mut worlds, reaches);
    let o = PlanOptions {
        seeds: 8,
        ..Default::default()
    };
    // Step 1 follows the gradient; step 4 the two-loop recursion over three remembered steps.
    for step in [1, 4] {
        let (worst, compared, optimized) =
            directions_apart(&cpu, &gpu, &worlds, &problems, &o, step);
        assert!(
            optimized >= 48,
            "step {step}: only {optimized} paths optimized"
        );
        assert!(
            compared > optimized / 2,
            "step {step}: only {compared} of {optimized} paths moved without clamping"
        );
        assert!(
            worst < 1e-2,
            "step {step}: directions differ by {worst} of their largest entry"
        );
    }
}

/// How far apart the CPU's and the GPU's `step`-th directions are (`step_directions`): the largest
/// difference of their entries over the paths no joint limit clamped, how many paths those are,
/// and the paths optimized.
fn directions_apart(
    cpu: &Device,
    gpu: &Device,
    worlds: &[World],
    problems: &[PlanProblem],
    o: &PlanOptions,
    step: u32,
) -> (f32, usize, usize) {
    let per_path = o.control_points * cpu.robot().dof();
    let (a, clamped_a) = step_directions(cpu, worlds, problems, o, step);
    let (b, clamped_b) = step_directions(gpu, worlds, problems, o, step);
    let (mut worst, mut compared) = (0.0f32, 0);
    let moved = |g: &[f32]| g.iter().any(|&v| v != 0.0);
    for (p, (ga, gb)) in a.chunks(per_path).zip(b.chunks(per_path)).enumerate() {
        if (0..per_path).any(|i| clamped_a[p * per_path + i] || clamped_b[p * per_path + i])
            || !moved(ga)
        {
            continue;
        }
        worst = ga
            .iter()
            .zip(gb)
            .fold(worst, |m, (x, y)| m.max((x - y).abs()));
        compared += 1;
    }
    // Problems searched to within a few percent of the straight line are not optimized.
    let optimized = a.chunks(per_path).filter(|g| moved(g)).count();
    eprintln!(
        "step {step}: worst |cpu-gpu| of directions scaled to their largest entry, {compared} of {optimized} \
         optimized paths: {worst:.2e}"
    );
    (worst, compared, optimized)
}

/// Constrained problems' directions match the CPU's too, their constraints' cost and gradient
/// twins of each other: the gripper within 5° of straight down on reaches past a pillar, and two
/// arms carrying a tray level, the right hand pinned in the left's frame, whose gradient runs
/// through both hands' Jacobians.
#[test]
fn constrained_trajopt_directions_match_cpu_element_wise() {
    let Some((_, gpu, cpu)) = setup() else {
        return;
    };
    let o = PlanOptions {
        seeds: 4,
        ..Default::default()
    };
    let scene = vec![common::pillar_scene()];
    let reaching = cpu.upload(&scene).unwrap();
    let (start, goals) = common::upright_reaches(&cpu, &reaching, 48, &mut Rng::new(6)).unwrap();
    let upright = HandConstraint {
        tilt: 5f32.to_radians(),
        ..HandConstraint::new(0, batchplan::Pose::new(Vec3::ZERO, common::upright(0.0)))
    };
    let uprights: Vec<PlanProblem> = goals
        .into_iter()
        .map(|g| PlanProblem {
            constraints: vec![upright],
            ..PlanProblem::new(0, start.clone(), g)
        })
        .collect();
    let two = common::two_pandas().unwrap();
    let (two_cpu, two_gpu) = (Device::cpu(&two).unwrap(), Device::gpu(&two).unwrap());
    let tray = vec![common::tray_scene()];
    let carrying = two_cpu.upload(&tray).unwrap();
    let common::TrayCarries {
        start,
        place,
        goals,
    } = common::tray_carries(&two_cpu, &carrying).unwrap();
    let carried = common::carried_level(place, 5f32.to_radians()).to_vec();
    let carries: Vec<PlanProblem> = goals
        .into_iter()
        .map(|g| PlanProblem {
            constraints: carried.clone(),
            ..PlanProblem::new(0, start.clone(), g)
        })
        .collect();
    // Step 1 checks the constraints' gradient. A step's direction doesn't show how far its line
    // search went, so their cost there shows from the next step on: the first search's at step 2,
    // the third's at step 4. Each step also turns on where the last landed, and an upright reach's
    // first step can cancel a constraint gradient of thousands down to a few: the constraint's
    // stiffness then turns f32 differences in where steps land into thousandths of the next
    // gradient, compounding. By step 4 the CPU against itself, its gradients perturbed by 1e-5, is
    // 1.1e-2 apart on the M4 Pro and 8.6e-3 on the Ryzen, and RADV (Mesa 26.0.8) against the CPU
    // 1.9e-2, so the reaches stop at step 2. The carries, 2.5e-4 apart by that measure at step 5,
    // go on to step 4.
    let cases = [
        (&cpu, &gpu, &scene, &uprights, 48, 2),
        (&two_cpu, &two_gpu, &tray, &carries, 12, 4),
    ];
    for (cpu, gpu, worlds, problems, least, last) in cases {
        let steps = [(1, 1e-3), (2, 1e-2), (4, 1e-2)];
        for (step, bound) in steps.into_iter().filter(|&(step, _)| step <= last) {
            let (worst, compared, optimized) =
                directions_apart(cpu, gpu, worlds, problems, &o, step);
            assert!(
                optimized >= least && compared > optimized / 2,
                "step {step}: {compared} of {optimized} paths compared"
            );
            assert!(
                worst < bound,
                "step {step}: directions differ by {worst} of their largest entry"
            );
        }
    }
}

/// Weights of a uniform cubic B-spline span's control points at `u` (same as the planner's).
fn basis(u: f32) -> [f32; 4] {
    let v = 1.0 - u;
    [
        v * v * v / 6.0,
        (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0,
        (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0,
        u * u * u / 6.0,
    ]
}

#[test]
fn trajopt_lowers_the_cost_as_far_as_the_cpu() {
    // The optimizers' line searches and directions, checked through what they achieve: the full
    // trajectory cost after optimization, computed independently on the CPU.
    let Some((robot, gpu, cpu)) = setup() else {
        return;
    };
    let n = robot.dof();
    // Two sets of scenes, pooled: one alone leaves the bound below little room.
    let (mut start, mut a, mut b, mut paths) = (0.0, 0.0, 0.0, 0);
    for (obstacles, targets) in [(12, 13), (22, 23)] {
        let mut scene = worlds(32, obstacles);
        let ik = batchplan::solve_ik(
            &cpu,
            &cpu.upload(&scene).unwrap(),
            &ik_problems(&scene, targets),
            &IkOptions::default(),
        )
        .unwrap();
        let reaches: Vec<PlanProblem> = ik
            .solved()
            .map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution))
            .collect();
        let problems = block_straight_lines(&robot, &mut scene, reaches);
        let on_cpu = cpu.upload(&scene).unwrap();
        let o = PlanOptions {
            seeds: 8,
            ..Default::default()
        };
        let (points, k) = (o.control_points, o.samples_per_span);
        let cost =
            |paths: &[f32]| -> f64 {
                let (mut q, mut item_world, mut smooth) = (vec![], vec![], 0.0f64);
                for (item, cp) in paths.chunks(points * n).enumerate() {
                    for span in 0..points - 3 {
                        for s in 0..k {
                            let w = basis((s as f32 + 0.5) / k as f32);
                            q.extend((0..n).map(|j| {
                                (0..4).map(|i| w[i] * cp[(span + i) * n + j]).sum::<f32>()
                            }));
                            item_world.push(problems[item / (1 + o.seeds)].world);
                        }
                    }
                    let p = |t: usize, j: usize| cp[t * n + j] as f64;
                    for j in 0..n {
                        smooth += (0..points - 1)
                            .map(|t| o.velocity_weight as f64 * (p(t + 1, j) - p(t, j)).powi(2))
                            .sum::<f64>();
                        smooth += (1..points - 1)
                            .map(|t| {
                                o.acceleration_weight as f64
                                    * (p(t + 1, j) - 2.0 * p(t, j) + p(t - 1, j)).powi(2)
                            })
                            .sum::<f64>();
                    }
                }
                let e = cpu
                    .evaluate(&on_cpu, &item_world, &q, &o.collision)
                    .unwrap();
                e.cost.iter().map(|&c| c as f64).sum::<f64>() + smooth
            };
        let per_path = points * robot.dof();
        let seeds = batchplan::plan(
            &cpu,
            &on_cpu,
            &problems,
            &PlanOptions { iterations: 0, ..o },
        )
        .unwrap();
        let seeds = back_to_back(seeds.paths, per_path);
        let on_cpu_paths = batchplan::plan(&cpu, &on_cpu, &problems, &o).unwrap();
        let on_cpu_paths = back_to_back(on_cpu_paths.paths, per_path);
        let on_gpu_paths =
            batchplan::plan(&gpu, &gpu.upload(&scene).unwrap(), &problems, &o).unwrap();
        let on_gpu_paths = back_to_back(on_gpu_paths.paths, per_path);
        start += cost(&seeds);
        a += cost(&on_cpu_paths);
        b += cost(&on_gpu_paths);
        paths += seeds.len() / (points * n);
    }
    eprintln!("trajectory cost of {paths} paths: seeds {start:.0}, CPU {a:.0}, GPU {b:.0}");
    assert!(
        a < 0.2 * start && b < 0.2 * start,
        "optimization barely lowered the cost: {start} -> {a}, {b}"
    );
    // The devices round differently, and a rounding's worth of change moves a set's total by up
    // to 1.7% either way; over both sets the GPU ended at most 0.3% above the CPU through such
    // changes. A GPU that ended a path's optimization at any search that found nothing better,
    // where only a steepest-descent one ends it, ended 13% above.
    assert!(
        b - a < 0.02 * a,
        "the GPU ends at a higher cost than the CPU: {b} against {a}"
    );
}
