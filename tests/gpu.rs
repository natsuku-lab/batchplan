//! The GPU device against the CPU device, plus end-to-end validity of GPU plans.
//! Skipped when no adapter is available unless `BATCHPLAN_REQUIRE_GPU=1`.
#![cfg(feature = "gpu")]

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::*;

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
    let err = Device::gpu_named(&robot, "no-such-gpu").err().expect("should fail");
    assert!(matches!(&err, Error::Gpu(m) if m.contains("no-such-gpu")), "{err:?}");
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
        .map(|(i, s)| IkProblem { world: i as u32, targets: vec![Some(common::grasp_target(s, &mut rng))], seed: None })
        .collect()
}

#[test]
fn evaluate_matches_cpu() {
    let Some((robot, gpu, cpu)) = setup() else { return };
    let n = robot.dof();
    let scene = worlds(16, 3);
    let (on_cpu, on_gpu) = (cpu.upload(&scene).unwrap(), gpu.upload(&scene).unwrap());
    let mut rng = Rng::new(9);
    let items = 20_000;
    let q: Vec<f32> = (0..items)
        .flat_map(|_| (0..n).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect::<Vec<_>>())
        .collect();
    let item_world: Vec<u32> = (0..items as u32).map(|i| i % 16).collect();
    let w = CollisionWeights { world: 1000.0, self_collision: 1000.0, margin: 0.02, self_margin: 0.01 };
    let a = cpu.evaluate(&on_cpu, &item_world, &q, &w).unwrap();
    let b = gpu.evaluate(&on_gpu, &item_world, &q, &w).unwrap();
    let mut worst = [0.0f32; 4];
    for i in 0..items {
        worst[0] = worst[0].max((a.world_clearance[i] - b.world_clearance[i]).abs());
        worst[1] = worst[1].max((a.self_clearance[i] - b.self_clearance[i]).abs());
        worst[2] = worst[2].max((a.cost[i] - b.cost[i]).abs() / a.cost[i].abs().max(1.0));
        // Gradient error relative to the gradient's norm (components cancel in f32).
        let row = |g: &[f32]| g[i * n..(i + 1) * n].to_vec();
        let (ga, gb) = (row(&a.grad), row(&b.grad));
        let norm = ga.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
        let diff = ga.iter().zip(&gb).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
        worst[3] = worst[3].max(diff / norm);
    }
    eprintln!(
        "max |cpu-gpu|: world {:.2e} self {:.2e} cost(rel) {:.2e} grad(rel) {:.2e}",
        worst[0], worst[1], worst[2], worst[3]
    );
    let colliding = a.cost.iter().filter(|&&c| c > 0.0).count();
    assert!(colliding > items / 10, "only {colliding} of {items} configurations carry collision cost");
    assert!(worst[0] < 1e-4 && worst[1] < 1e-4 && worst[2] < 1e-3 && worst[3] < 1e-3);
}

#[test]
fn evaluate_matches_cpu_for_every_obstacle_kind() {
    let Some((robot, gpu, cpu)) = setup() else { return };
    let n = robot.dof();
    let rotation = glam::Quat::from_euler(glam::EulerRot::XYZ, 0.4, -0.7, 1.1);
    let center = glam::Vec3::new(0.45, 0.0, 0.4);
    // Distance grids: a torus with an odd number of points (its last word holds one value) and a
    // box after it in the grid buffer; the last world places both, one grid shared with world 4.
    let sampled = |dims: [u32; 3], origin: glam::Vec3, f: &dyn Fn(glam::Vec3) -> f32| {
        let mut values = vec![];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    values.push(f(origin + glam::Vec3::new(i as f32, j as f32, k as f32) * 0.01));
                }
            }
        }
        std::sync::Arc::new(SdfGrid::new(dims, 0.01, origin, &values).unwrap())
    };
    let torus = sampled([41, 41, 21], glam::Vec3::new(-0.2, -0.2, -0.1), &|p| {
        glam::Vec2::new(p.truncate().length() - 0.15, p.z).length() - 0.05
    });
    let block = Obstacle::Cuboid {
        center: glam::Vec3::ZERO,
        half_extents: glam::Vec3::new(0.1, 0.05, 0.04),
        rotation: glam::Quat::IDENTITY,
    };
    let boxed = sampled([30, 20, 16], glam::Vec3::new(-0.15, -0.1, -0.08), &|p| block.distance(p).0);
    let grid =
        |grid: &std::sync::Arc<SdfGrid>, center: glam::Vec3| Obstacle::Sdf { grid: grid.clone(), center, rotation };
    let worlds: Vec<World> = [
        vec![Obstacle::Cuboid { center, half_extents: glam::Vec3::new(0.1, 0.2, 0.08), rotation }],
        vec![Obstacle::Sphere { center, radius: 0.15 }],
        vec![Obstacle::Cylinder { center, rotation, radius: 0.1, half_height: 0.2 }],
        vec![Obstacle::Capsule { center, rotation, radius: 0.08, half_length: 0.15 }],
        vec![grid(&torus, center)],
        vec![grid(&boxed, center), grid(&torus, center + glam::Vec3::new(0.0, 0.3, 0.1))],
    ]
    .into_iter()
    .map(|obstacles| World { obstacles })
    .collect();
    let kinds = ["cuboid", "sphere", "cylinder", "capsule", "grid", "two grids"];
    let mut rng = Rng::new(19);
    let items = 12_000;
    let q: Vec<f32> = (0..items * n).map(|i| rng.range(robot.lower()[i % n], robot.upper()[i % n])).collect();
    let item_world: Vec<u32> = (0..items as u32).map(|i| i % 6).collect();
    let w = CollisionWeights { world: 1000.0, self_collision: 0.0, margin: 0.02, self_margin: 0.0 };
    let a = cpu.evaluate(&cpu.upload(&worlds).unwrap(), &item_world, &q, &w).unwrap();
    let b = gpu.evaluate(&gpu.upload(&worlds).unwrap(), &item_world, &q, &w).unwrap();
    for (kind, name) in kinds.iter().enumerate() {
        let (mut colliding, mut worst_clearance, mut worst_grad) = (0, 0.0f32, 0.0f32);
        for i in (kind..items).step_by(6) {
            colliding += usize::from(a.cost[i] > 0.0);
            worst_clearance = worst_clearance.max((a.world_clearance[i] - b.world_clearance[i]).abs());
            let (ga, gb) = (&a.grad[i * n..(i + 1) * n], &b.grad[i * n..(i + 1) * n]);
            let norm = ga.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
            let diff = ga.iter().zip(gb).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
            worst_grad = worst_grad.max(diff / norm);
        }
        eprintln!("{name}: {colliding} colliding, clearance {worst_clearance:.2e}, grad {worst_grad:.2e}");
        assert!(colliding > items / 6 / 20, "only {colliding} configurations touch the {name}");
        assert!(worst_clearance < 1e-4 && worst_grad < 1e-3, "the {name} differs between devices");
    }
}

#[test]
fn ik_matches_cpu() {
    let Some((robot, gpu, cpu)) = setup() else { return };
    let scene = worlds(64, 4);
    let problems = ik_problems(&scene, 5);
    let o = IkOptions::default();
    let (on_cpu, on_gpu) = (cpu.upload(&scene).unwrap(), gpu.upload(&scene).unwrap());
    let a = solve_ik(&cpu, &on_cpu, &problems, &o).unwrap();
    let b = solve_ik(&gpu, &on_gpu, &problems, &o).unwrap();
    let agree = a.success.iter().zip(&b.success).filter(|(x, y)| x == y).count();
    let (sa, sb) = (a.success.iter().filter(|&&s| s).count(), b.success.iter().filter(|&&s| s).count());
    eprintln!("seed successes cpu {sa} gpu {sb}, agreement {agree}/{}", a.success.len());
    assert!(sa as f32 > 0.2 * a.success.len() as f32, "too few IK successes ({sa}) to compare");
    assert!(agree as f32 >= 0.97 * a.success.len() as f32);
    // Every GPU success must be a real solution according to the CPU model.
    let eval = cpu
        .evaluate(
            &on_cpu,
            &(0..b.success.len()).map(|i| (i / o.seeds) as u32).collect::<Vec<_>>(),
            &b.q,
            &CollisionWeights::NONE,
        )
        .unwrap();
    for i in (0..b.success.len()).filter(|&i| b.success[i]) {
        let target = problems[i / o.seeds].targets[0].expect("one hand");
        assert!((robot.ee_pose(b.solution(i), 0).position - target.position).length() < o.position_tolerance * 1.01);
        assert!(eval.collision_free(i));
    }
}

#[test]
fn gpu_plans_are_collision_free_under_cpu_check() {
    let Some((robot, gpu, cpu)) = setup() else { return };
    let n = robot.dof();
    let scene = worlds(128, 6);
    let problems = ik_problems(&scene, 7);
    let (on_cpu, on_gpu) = (cpu.upload(&scene).unwrap(), gpu.upload(&scene).unwrap());
    let ik = solve_ik(&gpu, &on_gpu, &problems, &IkOptions::default()).unwrap();
    let plan_problems: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem {
            world: s.problem.world,
            start: robot.default_q().to_vec(),
            goal: s.solution.to_vec(),
            start_motion: None,
            hold: vec![],
        })
        .collect();
    let o = PlanOptions::default();
    let result = plan(&gpu, &on_gpu, &plan_problems, &o).unwrap();
    let solved = result.solved().count();
    eprintln!("ik solved {}/{}; planned {solved}/{}", plan_problems.len(), problems.len(), plan_problems.len());
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
    for Solved { problem: prob, solution: cp, .. } in result.solved() {
        let trajectory = Trajectory::new(&robot, cp, 1.0).unwrap();
        trajectory.check(&robot).unwrap();
        let samples = trajectory.sample(per_span / trajectory.knot_interval).unwrap();
        let (first, last) = (&samples.positions[..n], &samples.positions[samples.positions.len() - n..]);
        assert!(first == &prob.start[..] && last == &prob.goal[..], "the trajectory does not run start to goal");
        dense.extend(&samples.positions);
        dense_world.extend(std::iter::repeat_n(prob.world, samples.len()));
    }
    let eval = cpu.evaluate(&on_cpu, &dense_world, &dense, &CollisionWeights::NONE).unwrap();
    let worst = eval.world_clearance.iter().chain(&eval.self_clearance).fold(f32::INFINITY, |m, &v| m.min(v));
    eprintln!("worst clearance along {} dense samples: {worst:.4} m", dense_world.len());
    assert!(worst > -2e-3, "trajectory penetrates by {worst}");
}

/// The direction of each path's `step`-th L-BFGS step, scaled to its largest entry (the line
/// search only scales it). The first step has no history, so it is the negative gradient; later
/// ones come from the two-loop recursion. Also returns a mask of values a joint limit clamped
/// (those carry no direction information).
/// A ball where each problem's hand passes halfway along its straight line, so every search
/// detours and the problem is optimized; problems whose ends the ball touches are dropped. Each
/// problem has a world of its own.
fn block_straight_lines(robot: &Robot, scene: &mut [World], problems: Vec<PlanProblem>) -> Vec<PlanProblem> {
    for p in &problems {
        let middle: Vec<f32> = p.start.iter().zip(&p.goal).map(|(a, b)| 0.5 * (a + b)).collect();
        let center = robot.ee_pose(&middle, 0).position;
        scene[p.world as usize].obstacles.push(Obstacle::Sphere { center, radius: 0.1 });
    }
    let cpu = Device::cpu(robot).unwrap();
    let worlds = cpu.upload(scene).unwrap();
    let free = |p: &PlanProblem| {
        let ends = [&p.start[..], &p.goal[..]].concat();
        let e = cpu.evaluate(&worlds, &[p.world, p.world], &ends, &CollisionWeights::NONE).unwrap();
        e.collision_free(0) && e.collision_free(1)
    };
    problems.into_iter().filter(free).collect()
}

fn step_directions(
    d: &Device,
    worlds: &[World],
    problems: &[PlanProblem],
    o: &PlanOptions,
    step: u32,
) -> (Vec<f32>, Vec<bool>) {
    let worlds = d.upload(worlds).unwrap();
    let seed = plan(d, &worlds, problems, &PlanOptions { iterations: step - 1, ..*o }).unwrap().paths.positions;
    let stepped = plan(d, &worlds, problems, &PlanOptions { iterations: step, ..*o }).unwrap().paths.positions;
    let robot = d.robot();
    let n = robot.dof();
    let clamped =
        stepped.iter().enumerate().map(|(i, &q)| q <= robot.lower()[i % n] || q >= robot.upper()[i % n]).collect();
    let mut gradients: Vec<f32> = seed.iter().zip(&stepped).map(|(a, b)| a - b).collect();
    for path in gradients.chunks_mut(o.control_points * n) {
        let largest = path.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if largest > 0.0 {
            path.iter_mut().for_each(|v| *v /= largest);
        }
    }
    (gradients, clamped)
}

#[test]
fn trajopt_directions_match_cpu_element_wise() {
    let Some((robot, gpu, cpu)) = setup() else { return };
    let n = robot.dof();
    let mut worlds = worlds(64, 8);
    let ik = solve_ik(&cpu, &cpu.upload(&worlds).unwrap(), &ik_problems(&worlds, 9), &IkOptions::default()).unwrap();
    let reaches: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem {
            world: s.problem.world,
            start: robot.default_q().to_vec(),
            goal: s.solution.to_vec(),
            start_motion: None,
            hold: vec![],
        })
        .collect();
    let problems = block_straight_lines(&robot, &mut worlds, reaches);
    let o = PlanOptions { seeds: 8, ..Default::default() };
    let per_path = o.control_points * n;
    // Step 1 follows the gradient; step 4 the two-loop recursion over three remembered steps.
    for step in [1, 4] {
        let (a, clamped_a) = step_directions(&cpu, &worlds, &problems, &o, step);
        let (b, clamped_b) = step_directions(&gpu, &worlds, &problems, &o, step);
        let (mut worst, mut compared) = (0.0f32, 0);
        let moved = |g: &[f32]| g.iter().any(|&v| v != 0.0);
        for (p, (ga, gb)) in a.chunks(per_path).zip(b.chunks(per_path)).enumerate() {
            if (0..per_path).any(|i| clamped_a[p * per_path + i] || clamped_b[p * per_path + i]) || !moved(ga) {
                continue;
            }
            worst = ga.iter().zip(gb).fold(worst, |m, (x, y)| m.max((x - y).abs()));
            compared += 1;
        }
        // Problems searched to within a few percent of the straight line are not optimized.
        let optimized = a.chunks(per_path).filter(|g| moved(g)).count();
        eprintln!(
            "step {step}: worst |cpu-gpu| of directions scaled to their largest entry, {compared} of {optimized} \
             optimized paths: {worst:.2e}"
        );
        assert!(optimized >= 48, "step {step}: only {optimized} paths optimized");
        assert!(compared > optimized / 2, "step {step}: only {compared} of {optimized} paths moved without clamping");
        assert!(worst < 1e-2, "step {step}: directions differ by {worst} of their largest entry");
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
    let Some((robot, gpu, cpu)) = setup() else { return };
    let n = robot.dof();
    let mut scene = worlds(32, 12);
    let ik = solve_ik(&cpu, &cpu.upload(&scene).unwrap(), &ik_problems(&scene, 13), &IkOptions::default()).unwrap();
    let reaches: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem {
            world: s.problem.world,
            start: robot.default_q().to_vec(),
            goal: s.solution.to_vec(),
            start_motion: None,
            hold: vec![],
        })
        .collect();
    let problems = block_straight_lines(&robot, &mut scene, reaches);
    let on_cpu = cpu.upload(&scene).unwrap();
    let o = PlanOptions { seeds: 8, ..Default::default() };
    let (points, k) = (o.control_points, o.samples_per_span);
    let cost = |paths: &[f32]| -> f64 {
        let (mut q, mut item_world, mut smooth) = (vec![], vec![], 0.0f64);
        for (item, cp) in paths.chunks(points * n).enumerate() {
            for span in 0..points - 3 {
                for s in 0..k {
                    let w = basis((s as f32 + 0.5) / k as f32);
                    q.extend((0..n).map(|j| (0..4).map(|i| w[i] * cp[(span + i) * n + j]).sum::<f32>()));
                    item_world.push(problems[item / (1 + o.seeds)].world);
                }
            }
            let p = |t: usize, j: usize| cp[t * n + j] as f64;
            for j in 0..n {
                smooth += (0..points - 1).map(|t| o.w_vel as f64 * (p(t + 1, j) - p(t, j)).powi(2)).sum::<f64>();
                smooth += (1..points - 1)
                    .map(|t| o.w_acc as f64 * (p(t + 1, j) - 2.0 * p(t, j) + p(t - 1, j)).powi(2))
                    .sum::<f64>();
            }
        }
        let e = cpu.evaluate(&on_cpu, &item_world, &q, &o.collision).unwrap();
        e.cost.iter().map(|&c| c as f64).sum::<f64>() + smooth
    };
    let seeds = plan(&cpu, &on_cpu, &problems, &PlanOptions { iterations: 0, ..o }).unwrap().paths.positions;
    let on_cpu_paths = plan(&cpu, &on_cpu, &problems, &o).unwrap().paths.positions;
    let on_gpu_paths = plan(&gpu, &gpu.upload(&scene).unwrap(), &problems, &o).unwrap().paths.positions;
    let (start, a, b) = (cost(&seeds), cost(&on_cpu_paths), cost(&on_gpu_paths));
    eprintln!("trajectory cost of {} paths: seeds {start:.0}, CPU {a:.0}, GPU {b:.0}", seeds.len() / (points * n));
    assert!(a < 0.2 * start && b < 0.2 * start, "optimization barely lowered the cost: {start} -> {a}, {b}");
    assert!((a - b).abs() < 0.05 * a, "the devices end at different costs: {a} and {b}");
}
