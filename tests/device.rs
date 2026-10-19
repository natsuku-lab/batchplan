//! Behaviour every `Device` must share, checked on each available device (the GPU is skipped when
//! absent unless `BATCHPLAN_REQUIRE_GPU=1`).

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::*;
use glam::Vec3;

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

#[test]
fn malformed_batches_are_errors_on_every_device() {
    let robot = panda();
    let q = robot.default_q().to_vec();
    let scene = vec![World { obstacles: vec![Obstacle::Sphere { center: Vec3::new(2.0, 0.0, 0.0), radius: 0.1 }] }];
    let none = CollisionWeights::NONE;
    for d in devices(&robot) {
        let name = d.name();
        let worlds = d.upload(&scene).unwrap();
        let elsewhere = Device::cpu(&robot).unwrap().upload(&scene).unwrap();
        let err = d.evaluate(&elsewhere, &[0], &q, &none).unwrap_err();
        assert!(matches!(err, Error::Input(_)), "{name}: worlds uploaded to another device give {err:?}");
        assert!(d.evaluate(&worlds, &[0, 0], &q, &none).is_err(), "{name}: two items, one configuration");
        assert!(d.evaluate(&worlds, &[0], &q[..6], &none).is_err(), "{name}: configuration missing a joint");
        assert!(d.evaluate(&worlds, &[1], &q, &none).is_err(), "{name}: world index past the end");
        let target = robot.ee_pose(&q, 0);
        assert!(
            solve_ik(
                &d,
                &worlds,
                &[IkProblem { world: 1, targets: vec![Some(target)], seed: None }],
                &IkOptions::default()
            )
            .is_err(),
            "{name}: IK"
        );
        let problem = PlanProblem { world: 1, start: q.clone(), goal: q.clone(), start_motion: None, hold: vec![] };
        assert!(plan(&d, &worlds, &[problem], &PlanOptions::default()).is_err(), "{name}: plan");
        // Inputs no device could check consistently are refused before any work.
        let input = |r: Result<_, Error>| matches!(r, Err(Error::Input(_)));
        let mut beyond = q.clone();
        beyond[3] = robot.upper()[3] + 0.5;
        let past_limits = PlanProblem { world: 0, start: q.clone(), goal: beyond, start_motion: None, hold: vec![] };
        assert!(
            input(plan(&d, &worlds, &[past_limits], &PlanOptions::default()).map(|_| ())),
            "{name}: goal past limits"
        );
        let skewed = Pose { rotation: glam::Quat::from_xyzw(0.0, 0.0, 0.0, 2.0), ..target };
        let ik = solve_ik(
            &d,
            &worlds,
            &[IkProblem { world: 0, targets: vec![Some(skewed)], seed: None }],
            &IkOptions::default(),
        );
        assert!(input(ik.map(|_| ())), "{name}: a non-unit target rotation");
        let rotated = |rotation| Obstacle::Cuboid { center: Vec3::ZERO, half_extents: Vec3::ONE, rotation };
        for bad in [
            Obstacle::Sphere { center: Vec3::new(f32::NAN, 0.0, 0.0), radius: 0.1 },
            Obstacle::Sphere { center: Vec3::ZERO, radius: -0.1 },
            rotated(glam::Quat::from_xyzw(0.0, 0.0, 0.0, 0.5)),
        ] {
            let err = d.upload(&[World { obstacles: vec![bad.clone()] }]).map(|_| ());
            assert!(input(err), "{name}: {bad:?} uploaded");
        }
    }
}

#[test]
fn clearances_are_exact_up_to_the_margins() {
    let robot = panda();
    let (n, items) = (robot.dof(), 4000);
    let mut rng = batchplan::rng::Rng::new(5);
    let q: Vec<f32> = (0..items * n).map(|i| rng.range(robot.lower()[i % n], robot.upper()[i % n])).collect();
    let item_world = vec![0; items];
    let scene = [common::tabletop(&mut batchplan::rng::Rng::new(4))];
    let margin = 0.05;
    let near = CollisionWeights { world: 1.0, self_collision: 1.0, margin, self_margin: margin };
    // Margins wider than the robot leave nothing far enough apart to skip.
    let everything = CollisionWeights { margin: 10.0, self_margin: 10.0, ..CollisionWeights::NONE };
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
                    assert!(g > margin && g <= e + 1e-5, "{}: {kind} {g} does not bound {e}", d.name());
                    bounded += usize::from(g < e);
                }
            }
            assert!(close > 100 && bounded > 20, "{}: {kind}: {close} close, {bounded} bounded", d.name());
        }
    }
}

#[test]
fn ik_reports_the_clearances_evaluate_finds() {
    let robot = panda();
    let scene = [common::tabletop(&mut batchplan::rng::Rng::new(4))];
    let mut rng = batchplan::rng::Rng::new(9);
    let problems: Vec<IkProblem> = (0..16)
        .map(|_| IkProblem { world: 0, targets: vec![Some(common::grasp_target(&scene[0], &mut rng))], seed: None })
        .collect();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let ik = solve_ik(&d, &worlds, &problems, &IkOptions::default()).unwrap();
        let items = ik.q.len() / robot.dof();
        let e = d.evaluate(&worlds, &vec![0; items], &ik.q, &CollisionWeights::NONE).unwrap();
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-6);
        assert!(close(&ik.world_clearance, &e.world_clearance), "{}: world clearance", d.name());
        assert!(close(&ik.self_clearance, &e.self_clearance), "{}: self clearance", d.name());
        assert!(!close(&ik.world_clearance, &ik.self_clearance), "world and self clearance should differ");
    }
}

#[test]
fn evaluation_does_not_depend_on_the_batch() {
    // Small batches split each configuration's collision work across several GPU invocations,
    // large ones give it one; the results must agree.
    let robot = panda();
    let n = robot.dof();
    let mut rng = batchplan::rng::Rng::new(13);
    let q: Vec<f32> = (0..40_000 * n).map(|i| rng.range(robot.lower()[i % n], robot.upper()[i % n])).collect();
    let scene = [common::tabletop(&mut batchplan::rng::Rng::new(4))];
    let w = CollisionWeights { world: 1.0, self_collision: 1.0, margin: 0.05, self_margin: 0.02 };
    let few = 64;
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let alone = d.evaluate(&worlds, &vec![0; few], &q[..few * n], &w).unwrap();
        let batched = d.evaluate(&worlds, &vec![0; q.len() / n], &q, &w).unwrap();
        assert!(alone.cost.iter().filter(|&&c| c > 0.0).count() > 10, "too few configurations carry cost");
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4 * x.abs().max(1.0));
        assert!(close(&alone.world_clearance, &batched.world_clearance[..few]), "{}: world clearance", d.name());
        assert!(close(&alone.self_clearance, &batched.self_clearance[..few]), "{}: self clearance", d.name());
        assert!(close(&alone.cost, &batched.cost[..few]), "{}: cost", d.name());
        assert!(close(&alone.grad, &batched.grad[..few * n]), "{}: gradient", d.name());
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
        let e = d.evaluate(&worlds, &[0], &start, &CollisionWeights::NONE).unwrap();
        assert!(e.world_clearance[0] > 1e29, "{name}: an empty world has nothing to hit");
        let ik = solve_ik(
            &d,
            &worlds,
            &[IkProblem { world: 0, targets: vec![Some(target)], seed: None }],
            &IkOptions::default(),
        )
        .unwrap();
        let goal = ik.best(0).unwrap_or_else(|| panic!("{name}: reachable target unsolved")).to_vec();
        let result = plan(
            &d,
            &worlds,
            &[PlanProblem { world: 0, start: start.clone(), goal, start_motion: None, hold: vec![] }],
            &PlanOptions::default(),
        )
        .unwrap();
        assert!(result.best(0).is_some(), "{name}: no valid plan in an empty world");
    }
}

#[test]
fn results_keep_the_worlds_of_their_problems() {
    // Two targets in world 1 and one in world 0: problem index != world index.
    let robot = panda();
    let mut rng = batchplan::rng::Rng::new(21);
    let scene: Vec<World> = (0..2).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = [1u32, 1, 0]
        .iter()
        .map(|&w| IkProblem {
            world: w,
            targets: vec![Some(common::grasp_target(&scene[w as usize], &mut rng))],
            seed: None,
        })
        .collect();
    let cpu = Device::cpu(&robot).unwrap();
    let on_cpu = cpu.upload(&scene).unwrap();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let ik = solve_ik(&d, &worlds, &goals, &IkOptions::default()).unwrap();
        assert_eq!(ik.problems.len(), goals.len());
        let problems: Vec<PlanProblem> = ik
            .solved()
            .map(|s| PlanProblem {
                world: s.problem.world,
                start: robot.default_q().to_vec(),
                goal: s.solution.to_vec(),
                start_motion: None,
                hold: vec![],
            })
            .collect();
        assert!(problems.len() >= 2, "{}: too few IK solutions to test", d.name());
        let result = plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        assert!(
            result.solved().any(|s| s.index as u32 != s.problem.world),
            "{}: no solved problem whose index differs from its world",
            d.name()
        );
        for s in result.solved() {
            assert_eq!(s.problem.world, problems[s.index].world);
            // Every reported path is collision-free in its own world along its whole length.
            let n = robot.dof();
            let trajectory = Trajectory::new(&robot, s.solution, 1.0).unwrap();
            let q = trajectory.sample(32.0 / trajectory.knot_interval).unwrap().positions;
            let item_world = vec![s.problem.world; q.len() / n];
            let e = cpu.evaluate(&on_cpu, &item_world, &q, &CollisionWeights::NONE).unwrap();
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
    //   J = w_acc * sum |q[t+1] - 2 q[t] + q[t-1]|^2 + w_vel * sum |q[t+1] - q[t]|^2,
    // whose gradient sets the direction of the first L-BFGS step (see tests/gpu.rs).
    let robot = panda();
    let n = robot.dof();
    let mut goal = robot.default_q().to_vec();
    goal[0] += 1.0;
    goal[2] -= 0.7;
    // A ball on the straight line, so the searched path detours and the problem is optimized.
    let middle: Vec<f32> = robot.default_q().iter().zip(&goal).map(|(a, b)| 0.5 * (a + b)).collect();
    let ball = Obstacle::Sphere { center: robot.ee_pose(&middle, 0).position, radius: 0.05 };
    let problems =
        vec![PlanProblem { world: 0, start: robot.default_q().to_vec(), goal, start_motion: None, hold: vec![] }];
    let scene = vec![World { obstacles: vec![ball] }];
    let o = PlanOptions { collision: CollisionWeights::NONE, seeds: 8, ..Default::default() };
    let cost = |path: &[f64]| {
        let t_count = path.len() / n;
        let q = |t: usize, j: usize| path[t * n + j];
        let mut c = 0.0;
        for j in 0..n {
            for t in 0..t_count - 1 {
                c += o.w_vel as f64 * (q(t + 1, j) - q(t, j)).powi(2);
            }
            for t in 1..t_count - 1 {
                c += o.w_acc as f64 * (q(t + 1, j) - 2.0 * q(t, j) + q(t - 1, j)).powi(2);
            }
        }
        c
    };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let seed = plan(&d, &worlds, &problems, &PlanOptions { iterations: 0, ..o }).unwrap().paths.positions;
        let stepped = plan(&d, &worlds, &problems, &PlanOptions { iterations: 1, ..o }).unwrap().paths.positions;
        // Optimized seed 2's path, bent through a via point (seed 0 is the searched path, seed 1
        // the straight line), after the searched path. Only control points 3..points - 3 move.
        let points = o.control_points;
        let offset = 3 * points * n;
        let path: Vec<f64> = seed[offset..offset + points * n].iter().map(|&v| v as f64).collect();
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
        let step: Vec<f64> = free.map(|i| (seed[offset + i] - stepped[offset + i]) as f64).collect();
        let largest = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let (fd_max, step_max) = (largest(&fd), largest(&step));
        assert!(fd_max > 0.1, "{}: test path is too straight to exercise the gradient ({fd_max})", d.name());
        assert!(step_max > 0.0, "{}: the step did not move the path", d.name());
        let worst = fd.iter().zip(&step).map(|(f, s)| (f / fd_max - s / step_max).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-2, "{}: the step's direction is off the smoothness gradient by {worst:.2e}", d.name());
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
    let middle: Vec<f32> = start.iter().zip(&goal).map(|(a, b)| 0.5 * (a + b)).collect();
    let center = robot.ee_pose(&middle, 0).position;
    let post = Obstacle::Cylinder { center, rotation: glam::Quat::IDENTITY, radius: 0.05, half_height: 0.3 };
    let scene = vec![World { obstacles: vec![post] }];
    let problems = vec![PlanProblem { world: 0, start, goal, start_motion: None, hold: vec![] }];
    // No search beyond the straight line, which the post blocks, so optimized seed 0 is that line.
    let o = PlanOptions { seeds: 8, rrt: RrtOptions { max_iterations: 0, ..Default::default() }, ..Default::default() };
    let (points, k) = (o.control_points, o.samples_per_span);
    let samples_of = |cp: &[f64]| -> Vec<f32> {
        let mut out = vec![];
        for span in 0..points - 3 {
            for s in 0..k {
                let w = basis((s as f64 + 0.5) / k as f64);
                out.extend((0..n).map(|j| (0..4).map(|i| w[i] * cp[(span + i) * n + j]).sum::<f64>() as f32));
            }
        }
        out
    };
    let smoothness = |cp: &[f64]| {
        let q = |t: usize, j: usize| cp[t * n + j];
        let mut c = 0.0;
        for j in 0..n {
            for t in 0..points - 1 {
                c += o.w_vel as f64 * (q(t + 1, j) - q(t, j)).powi(2);
            }
            for t in 1..points - 1 {
                c += o.w_acc as f64 * (q(t + 1, j) - 2.0 * q(t, j) + q(t - 1, j)).powi(2);
            }
        }
        c
    };
    let cpu = Device::cpu(&robot).unwrap();
    let on_cpu = cpu.upload(&scene).unwrap();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let seed = plan(&d, &worlds, &problems, &PlanOptions { iterations: 0, ..o }).unwrap().paths.positions;
        let stepped = plan(&d, &worlds, &problems, &PlanOptions { iterations: 1, ..o }).unwrap().paths.positions;
        // Optimized seed 0, the straight line through the post, after the (unsolved) searched path.
        let path: Vec<f64> = seed[points * n..2 * points * n].iter().map(|&v| v as f64).collect();
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
            let costs = cpu.evaluate(&on_cpu, &vec![0; items], &perturbed, &o.collision).unwrap().cost;
            let collision = |p: usize| costs[p * per_path..(p + 1) * per_path].iter().map(|&c| c as f64).sum::<f64>();
            free.iter()
                .enumerate()
                .map(|(f, &i)| {
                    let (mut up, mut down) = (path.clone(), path.clone());
                    up[i] += h;
                    down[i] -= h;
                    (collision(2 * f) - collision(2 * f + 1) + smoothness(&up) - smoothness(&down)) / (2.0 * h)
                })
                .collect()
        };
        let (fine, coarse) = (fd_at(2e-4), fd_at(1e-3));
        // Without history the step is steepest descent: -gradient times a positive scale, so
        // gradients are compared scaled to their largest entries.
        let step: Vec<f64> = free.iter().map(|&i| (seed[points * n + i] - stepped[points * n + i]) as f64).collect();
        let largest = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let (fine_max, coarse_max, step_max) = (largest(&fine), largest(&coarse), largest(&step));
        assert!(step_max > 0.0, "{}: the step did not move the path", d.name());
        let mut worst = 0.0f64;
        for f in 0..free.len() {
            let s = step[f] / step_max;
            let err = (fine[f] / fine_max - s).abs().min((coarse[f] / coarse_max - s).abs());
            worst = worst.max(err / (fine[f].abs() / fine_max).max(0.05));
        }
        eprintln!("{}: largest gradient {fine_max:.1}, worst relative error {worst:.2e}", d.name());
        let seed_samples = samples_of(&path);
        let items = seed_samples.len() / n;
        let touching = cpu.evaluate(&on_cpu, &vec![0; items], &seed_samples, &o.collision).unwrap().cost;
        assert!(touching.iter().filter(|&&c| c > 1.0).count() > 3, "{}: the straight seed misses the post", d.name());
        assert!(worst < 2e-2, "{}: trajectory gradient off by {worst:.2e} relative", d.name());
    }
}

/// 64 tabletop reaches from the default pose, the worlds they live in, and their IK goals.
fn reaches(robot: &Robot) -> (Vec<World>, Vec<PlanProblem>) {
    let mut rng = batchplan::rng::Rng::new(41);
    let scene: Vec<World> = (0..64).map(|_| common::tabletop(&mut rng)).collect();
    let cpu = Device::cpu(robot).unwrap();
    let goals: Vec<IkProblem> = scene
        .iter()
        .enumerate()
        .map(|(i, w)| IkProblem { world: i as u32, targets: vec![Some(common::grasp_target(w, &mut rng))], seed: None })
        .collect();
    let ik = solve_ik(&cpu, &cpu.upload(&scene).unwrap(), &goals, &IkOptions::default()).unwrap();
    let problems = ik
        .solved()
        .map(|s| PlanProblem {
            world: s.problem.world,
            start: robot.default_q().to_vec(),
            goal: s.solution.to_vec(),
            start_motion: None,
            hold: vec![],
        })
        .collect();
    (scene, problems)
}

/// `reaches` with a ball where each hand passes halfway along its straight line, so every search
/// detours and optimization has work to do; problems whose ends the ball touches are dropped.
fn blocked_reaches(robot: &Robot) -> (Vec<World>, Vec<PlanProblem>) {
    let (mut scene, problems) = reaches(robot);
    for p in &problems {
        let middle: Vec<f32> = p.start.iter().zip(&p.goal).map(|(a, b)| 0.5 * (a + b)).collect();
        let center = robot.ee_pose(&middle, 0).position;
        scene[p.world as usize].obstacles.push(Obstacle::Sphere { center, radius: 0.1 });
    }
    let cpu = Device::cpu(robot).unwrap();
    let worlds = cpu.upload(&scene).unwrap();
    let free = |p: &PlanProblem| {
        let ends = [&p.start[..], &p.goal[..]].concat();
        let e = cpu.evaluate(&worlds, &[p.world, p.world], &ends, &CollisionWeights::NONE).unwrap();
        e.collision_free(0) && e.collision_free(1)
    };
    let problems = problems.into_iter().filter(free).collect();
    (scene, problems)
}

#[test]
fn a_tiny_time_budget_returns_after_at_most_one_more_chunk() {
    let robot = panda();
    let (scene, problems) = blocked_reaches(&robot);
    // Enough rounds (20 chunks) that a fast GPU's plan still lasts well beyond one chunk.
    let o = PlanOptions { seeds: 8, iterations: 160, ..Default::default() };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let timed = |o: &PlanOptions| {
            let t = std::time::Instant::now();
            let result = plan(&d, &worlds, &problems, o).unwrap();
            (t.elapsed(), result)
        };
        timed(&o); // warm up
        let (full, _) = timed(&o);
        // A GPU checks the budget between submissions of 8 rounds; time one such chunk, with
        // seeding and validation.
        let (chunk, _) = timed(&PlanOptions { iterations: 8, ..o });
        let budget = std::time::Duration::from_millis(1);
        let (spent, result) = timed(&PlanOptions { time_budget: Some(budget), ..o });
        eprintln!("{}: full {full:?}, one chunk {chunk:?}, budget {budget:?} -> {spent:?}", d.name());
        assert!(spent < budget + 2 * chunk + std::time::Duration::from_millis(20), "{}: took {spent:?}", d.name());
        assert!(spent < full / 2, "{}: the budget saved little ({spent:?} of {full:?})", d.name());
        assert_eq!(result.valid.len(), problems.len() * (1 + o.seeds));
    }
}

#[test]
fn planning_without_a_budget_gives_the_same_result_every_run() {
    let robot = panda();
    let (scene, problems) = reaches(&robot);
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let a = plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        let b = plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        assert_eq!(a.paths, b.paths, "{}: paths differ between runs", d.name());
        assert_eq!((a.valid, a.length), (b.valid, b.length), "{}", d.name());
    }
}

#[test]
fn plans_from_a_moving_start_continue_its_motion() {
    let robot = panda();
    let n = robot.dof();
    let (scene, problems) = reaches(&robot);
    let motion = StartMotion {
        velocity: vec![0.5, -0.3, 0.2, 0.4, 0.0, 0.3, 0.0],
        acceleration: vec![1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.5],
    };
    let moving: Vec<PlanProblem> =
        problems.iter().take(8).map(|p| PlanProblem { start_motion: Some(motion.clone()), ..p.clone() }).collect();
    let start = JointState {
        position: robot.default_q().to_vec(),
        velocity: motion.velocity.clone(),
        acceleration: motion.acceleration.clone(),
    };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let result = plan(&d, &worlds, &moving, &PlanOptions::default()).unwrap();
        let solved: Vec<_> = result.solved().collect();
        assert!(solved.len() >= 6, "{}: {} of 8 planned", d.name(), solved.len());
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
            assert_eq!(end.position, s.problem.goal, "{}", d.name());
        }
    }
}

#[test]
fn reaches_replan_mid_motion_at_full_speed() {
    // Partway through full-speed reaches, replanned to the same goals: the old motions' remainders
    // show that paths exist.
    let robot = panda();
    let n = robot.dof();
    let (scene, problems) = reaches(&robot);
    let cpu = Device::cpu(&robot).unwrap();
    let rest = plan(&cpu, &cpu.upload(&scene).unwrap(), &problems[..24], &PlanOptions::default()).unwrap();
    let mut state = JointState::new(n);
    let (mut moving, mut states) = (vec![], vec![]);
    for s in rest.solved() {
        let trajectory = Trajectory::new(&robot, s.solution, 1.0).unwrap();
        trajectory.at(0.4 * trajectory.duration(), &mut state);
        if state.velocity.iter().fold(0.0f32, |m, v| m.max(v.abs())) < 1.5 {
            continue;
        }
        let motion = StartMotion { velocity: state.velocity.clone(), acceleration: state.acceleration.clone() };
        moving.push(PlanProblem { start: state.position.clone(), start_motion: Some(motion), ..s.problem.clone() });
        states.push(state.clone());
    }
    assert!(moving.len() >= 8, "only {} reaches move fast", moving.len());
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let result = plan(&d, &worlds, &moving, &PlanOptions::default()).unwrap();
        let solved: Vec<_> = result.solved().collect();
        eprintln!("{}: {} of {} replanned at full speed", d.name(), solved.len(), moving.len());
        // 18 of 24 on an M4 Pro, 17 on CI's x86-64; the start knot interval sized for rest gives 0.
        assert!(5 * solved.len() >= 3 * moving.len(), "{}: {} of {} replanned", d.name(), solved.len(), moving.len());
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
        &RobotOptions { collision_model: None, ..common::panda_options() },
    )
    .unwrap();
    let base = robot.link_pose(robot.default_q(), "panda_link0").unwrap();
    let lowest = robot.collision_model().spheres["panda_link0"]
        .iter()
        .map(|s| (base.position + base.rotation * Vec3::new(s[0], s[1], s[2])).z - s[3])
        .fold(f32::INFINITY, f32::min);
    assert!(lowest < 0.0, "the base's spheres should reach below the table: {lowest}");
    let table = Obstacle::Cuboid {
        center: Vec3::new(0.4, 0.0, -0.02),
        half_extents: Vec3::new(0.7, 0.8, 0.02),
        rotation: glam::Quat::IDENTITY,
    };
    let mounted = World { obstacles: vec![table.clone()] };
    // A pin through the elbow still collides.
    let elbow = robot.link_pose(robot.default_q(), "panda_link4").unwrap().position;
    let pin =
        Obstacle::Cuboid { center: elbow, half_extents: Vec3::new(0.3, 0.005, 0.005), rotation: glam::Quat::IDENTITY };
    let pinned = World { obstacles: vec![table, pin] };
    let goal = vec![0.5, -0.8, 0.0, -2.2, 0.0, 1.6, 0.8];
    for d in devices(&robot) {
        let worlds = d.upload(&[mounted.clone(), pinned.clone()]).unwrap();
        let q: Vec<f32> = robot.default_q().iter().chain(robot.default_q()).copied().collect();
        let e = d.evaluate(&worlds, &[0, 1], &q, &CollisionWeights::NONE).unwrap();
        assert!(e.collision_free(0), "{}: on its table, world clearance {}", d.name(), e.world_clearance[0]);
        assert!(!e.collision_free(1), "{}: the pin should collide", d.name());
        let problem = PlanProblem {
            world: 0,
            start: robot.default_q().to_vec(),
            goal: goal.clone(),
            start_motion: None,
            hold: vec![],
        };
        let result = plan(&d, &worlds, &[problem], &PlanOptions::default()).unwrap();
        assert!(result.best(0).is_some(), "{}: no plan on the table", d.name());
    }
}

#[test]
fn seeded_ik_stays_on_the_branch_it_starts_from() {
    let robot = panda();
    // A configuration far from the default one, and a target 2 cm above its IK frame.
    let here = vec![1.2, 0.3, -0.8, -1.6, 0.9, 2.0, -0.4];
    let far: f32 = here.iter().zip(robot.default_q()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
    assert!(far > 1.5, "the start should be far from the default configuration: {far}");
    let mut target = robot.ee_pose(&here, 0);
    target.position.z += 0.02;
    let one_seed = IkOptions { seeds: 1, ..IkOptions::default() };
    for d in devices(&robot) {
        let worlds = d.upload(&[World::default()]).unwrap();
        let seeded = IkProblem { world: 0, targets: vec![Some(target)], seed: Some(here.clone()) };
        let ik = solve_ik(&d, &worlds, &[seeded], &one_seed).unwrap();
        let q = ik.nearest(0, &here).unwrap_or_else(|| panic!("{}: no solution from the seed", d.name()));
        let moved: f32 = q.iter().zip(&here).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        assert!(moved < 0.2, "{}: moved {moved} rad from the seed", d.name());
        // Unseeded, the one seed starts from the default configuration and ends elsewhere.
        let ik = solve_ik(&d, &worlds, &[IkProblem { world: 0, targets: vec![Some(target)], seed: None }], &one_seed)
            .unwrap();
        let elsewhere = ik.solution(0).iter().zip(&here).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        assert!(elsewhere > 0.5, "{}: from the default it lands {elsewhere} rad away", d.name());
    }
}

/// Optimization starts from the searched path's spline, then the straight line to the goal: with no
/// steps taken, seeds 0 and 1 are those.
#[test]
fn optimization_seeds_start_with_the_searched_path_and_the_straight_line() {
    let robot = panda();
    let n = robot.dof();
    let (scene, problems) = blocked_reaches(&robot);
    let o = PlanOptions { seeds: 2, iterations: 0, ..Default::default() };
    let (points, per_path) = (o.control_points, o.control_points * n);
    for d in devices(&robot) {
        let result = plan(&d, &d.upload(&scene).unwrap(), &problems, &o).unwrap();
        let mut seeded = 0;
        for (pi, p) in problems.iter().enumerate() {
            let path = |k: usize| result.paths.path(pi * 3 + k);
            if path(1).iter().all(|&v| v == 0.0) {
                continue;
            }
            seeded += 1;
            assert_eq!(path(1), path(0), "{}: problem {pi}'s first seed is not its searched spline", d.name());
            for (t, q) in path(2).chunks(n).enumerate() {
                let u = (t.saturating_sub(2) as f32 / (points - 5) as f32).min(1.0);
                let off =
                    (0..n).map(|j| (q[j] - (p.start[j] + u * (p.goal[j] - p.start[j]))).abs()).fold(0.0, f32::max);
                assert!(off < 1e-5, "{}: problem {pi}'s second seed leaves the straight line by {off}", d.name());
            }
        }
        assert!(seeded * 3 >= problems.len(), "{}: only {seeded} of {} problems seeded", d.name(), problems.len());
        assert_eq!(result.paths.positions.len(), problems.len() * 3 * per_path);
    }
}

/// A problem searched to its straight line is not optimized: no seed could be shorter.
#[test]
fn problems_searched_straight_are_not_optimized() {
    let robot = panda();
    let (_, reaches) = reaches(&robot);
    let scene = vec![World::default()];
    let problems: Vec<PlanProblem> = reaches.into_iter().take(8).map(|p| PlanProblem { world: 0, ..p }).collect();
    let o = PlanOptions { seeds: 4, ..Default::default() };
    for d in devices(&robot) {
        let result = plan(&d, &d.upload(&scene).unwrap(), &problems, &o).unwrap();
        for pi in 0..problems.len() {
            assert!(result.valid[pi * 5], "{}: problem {pi} has no searched path", d.name());
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
    let (_, reaches) = reaches(&robot);
    let scene = vec![World::default()];
    let problems: Vec<PlanProblem> = reaches.into_iter().take(8).map(|p| PlanProblem { world: 0, ..p }).collect();
    let o = PlanOptions { seeds: 4, time_budget: Some(std::time::Duration::ZERO), ..Default::default() };
    for d in devices(&robot) {
        let result = plan(&d, &d.upload(&scene).unwrap(), &problems, &o).unwrap();
        assert!(result.valid.iter().all(|&v| !v), "{}: a seed was planned or optimized", d.name());
    }
}

/// Repairs keep their own weights: without optimization seeds, plans do not depend on the
/// optimization's.
#[test]
fn plans_without_seeds_ignore_the_optimization_weights() {
    let robot = panda();
    let n = robot.dof();
    let (scene, problems) = reaches(&robot);
    let cpu = Device::cpu(&robot).unwrap();
    let worlds = cpu.upload(&scene).unwrap();
    let rest = plan(&cpu, &worlds, &problems, &PlanOptions::default()).unwrap();
    let mut state = JointState::new(n);
    let moving: Vec<PlanProblem> = rest
        .solved()
        .map(|s| {
            let trajectory = Trajectory::new(&robot, s.solution, 1.0).unwrap();
            trajectory.at(0.4 * trajectory.duration(), &mut state);
            let motion = StartMotion { velocity: state.velocity.clone(), acceleration: state.acceleration.clone() };
            PlanProblem { start: state.position.clone(), start_motion: Some(motion), ..s.problem.clone() }
        })
        .collect();
    let weighed = PlanOptions {
        w_acc: 0.5,
        w_vel: 40.0,
        collision: CollisionWeights { margin: 0.003, ..PlanOptions::default().collision },
        ..Default::default()
    };
    let a = plan(&cpu, &worlds, &moving, &PlanOptions::default()).unwrap();
    let b = plan(&cpu, &worlds, &moving, &weighed).unwrap();
    assert_eq!(a.paths.positions, b.paths.positions, "the optimization's weights changed plans without seeds");
}
