//! Executable trajectories: limits hold everywhere along planned paths, sampling a trajectory does
//! not allocate, and `check` refuses unsafe trajectories.

#[path = "../examples/common/mod.rs"]
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use batchplan::rng::Rng;
use batchplan::{
    Device, Error, HandSpeed, IkOptions, IkProblem, JointState, PlanOptions, PlanProblem, Robot,
    RobotOptions, StartMotion, Trajectory, World,
};

/// Counts allocations made by the current thread, so other test threads cannot interfere.
struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: every call goes to the system allocator unchanged; only a thread-local count is added.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        // SAFETY: forwarded unchanged; the caller meets the system allocator's requirements.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged; the caller meets the system allocator's requirements.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Collision-free plans from the default pose to IK solutions in tabletop worlds.
fn planned(robot: &Robot, count: usize) -> Vec<Vec<f32>> {
    let cpu = Device::cpu(robot).unwrap();
    let mut rng = Rng::new(31);
    let worlds: Vec<World> = (0..count).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = worlds
        .iter()
        .enumerate()
        .map(|(i, w)| IkProblem::new(i as u32, common::grasp_target(w, &mut rng)))
        .collect();
    let worlds = cpu.upload(&worlds).unwrap();
    let ik = batchplan::solve_ik(&cpu, &worlds, &goals, &IkOptions::default()).unwrap();
    let problems: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution))
        .collect();
    let result = batchplan::plan(&cpu, &worlds, &problems, &PlanOptions::default()).unwrap();
    let paths: Vec<Vec<f32>> = result.solved().map(|s| s.solution.to_vec()).collect();
    assert!(
        paths.len() >= count * 3 / 4,
        "only {} of {count} problems planned",
        paths.len()
    );
    paths
}

#[test]
fn trajectories_stay_within_limits_everywhere() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let limits = [
        robot.max_velocity(),
        robot.max_acceleration(),
        robot.max_jerk(),
    ];
    let mut peak = [0.0f32; 3];
    for path in planned(&robot, 16) {
        let trajectory = Trajectory::try_new(&robot, &path, 1.0).unwrap();
        trajectory.check(&robot).unwrap();
        let dt = trajectory.knot_interval / 64.0;
        let samples = trajectory.sample(1.0 / dt).unwrap();
        let (p, v, a) = (
            &samples.positions,
            &samples.velocities,
            &samples.accelerations,
        );
        assert_eq!(&p[..n], &path[..n], "starts exactly at the start");
        assert_eq!(
            &p[p.len() - n..],
            &path[path.len() - n..],
            "ends exactly at the goal"
        );
        assert!(
            v[..n].iter().chain(&v[v.len() - n..]).all(|&x| x == 0.0),
            "at rest at both ends"
        );
        for i in 0..p.len() {
            let j = i % n;
            // The curve stays in the control points' hull, up to f32 rounding of the blend.
            let excess = (robot.lower()[j] - p[i]).max(p[i] - robot.upper()[j]);
            assert!(excess < 1e-5, "joint {j} leaves its range by {excess}");
            peak[0] = peak[0].max(v[i].abs() / limits[0][j]);
            peak[1] = peak[1].max(a[i].abs() / limits[1][j]);
        }
        // Jerk is constant along each span, so finite differences of acceleration never exceed it.
        // A coarser step keeps f32 cancellation in the accelerations from dominating.
        let coarse_dt = trajectory.knot_interval / 4.0;
        let coarse = trajectory.sample(1.0 / coarse_dt).unwrap().accelerations;
        for i in n..coarse.len() - n {
            peak[2] =
                peak[2].max(((coarse[i] - coarse[i - n]) / coarse_dt).abs() / limits[2][i % n]);
        }
        // The analytic derivatives match finite differences of the samples (away from the ends,
        // where sampling clamps).
        for i in 2 * n..p.len() - 2 * n {
            let fd_v = (p[i + n] - p[i - n]) / (2.0 * dt);
            let fd_a = (v[i + n] - v[i - n]) / (2.0 * dt);
            assert!(
                (fd_v - v[i]).abs() < 2e-3 * limits[0][i % n],
                "velocity off: {} vs {fd_v}",
                v[i]
            );
            assert!(
                (fd_a - a[i]).abs() < 2e-2 * limits[1][i % n],
                "acceleration off: {} vs {fd_a}",
                a[i]
            );
        }
    }
    eprintln!("peak fraction of the velocity, acceleration and jerk limits: {peak:?}");
    assert!(
        peak.iter().all(|&p| p <= 1.0 + 1e-3),
        "a limit is exceeded: {peak:?}"
    );
    // The binding limit is reached: the timing is as fast as the limits allow.
    assert!(
        peak.iter().any(|&p| p > 0.9),
        "no limit is close to binding: {peak:?}"
    );
}

/// Samples of paths that run along each joint's upper, then lower, limit for a stretch, starting
/// and ending at rest: blends of control points at the limit round past it unless each is held to
/// the points it blends.
#[test]
fn samples_of_a_path_along_a_limit_stay_within_it() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let (lower, upper) = (robot.lower(), robot.upper());
    let mut rng = Rng::new(7);
    let mut past = 0.0f32;
    for trial in 0..4 * n {
        let held = trial % n;
        let mut points: Vec<Vec<f32>> = (0..4)
            .map(|_| {
                let q = robot.default_q().iter().enumerate();
                q.map(|(j, &v)| (v + rng.range(-0.3, 0.3)).clamp(lower[j], upper[j]))
                    .collect()
            })
            .collect();
        let limit = if trial < 2 * n {
            upper[held]
        } else {
            lower[held]
        };
        for p in &mut points {
            p[held] = limit;
        }
        let (first, last) = (points[0].clone(), points[3].clone());
        let cp = [vec![first; 3], points, vec![last; 3]].concat().concat();
        let samples = Trajectory::try_new(&robot, &cp, 1.0)
            .unwrap()
            .sample(2000.0)
            .unwrap();
        for q in samples.positions.chunks(n) {
            for j in 0..n {
                past = past.max(q[j] - upper[j]).max(lower[j] - q[j]);
            }
        }
    }
    assert!(past <= 0.0, "a sample lies {past} rad past a limit");
}

#[test]
fn slower_speed_scales_stretch_time() {
    let robot = common::panda().unwrap();
    let path = &planned(&robot, 4)[0];
    let fast = Trajectory::try_new(&robot, path, 1.0).unwrap();
    let slow = Trajectory::try_new(&robot, path, 0.5).unwrap();
    assert!((slow.duration() - 2.0 * fast.duration()).abs() < 1e-5 * fast.duration());
    slow.check(&robot).unwrap();
}

#[test]
fn sampling_does_not_allocate() {
    let robot = common::panda().unwrap();
    let trajectory = Trajectory::try_new(&robot, &planned(&robot, 4)[0], 1.0).unwrap();
    let mut state = JointState::new(robot.dof());
    let before = ALLOCATIONS.with(Cell::get);
    let mut checksum = 0.0;
    for k in 0..=1000 {
        trajectory.at(trajectory.duration() * k as f32 / 1000.0, &mut state);
        checksum += state.position[0] + state.velocity[0] + state.acceleration[0];
    }
    let allocations = ALLOCATIONS.with(Cell::get) - before;
    assert!(checksum.is_finite());
    assert_eq!(
        allocations, 0,
        "Trajectory::at allocated {allocations} times"
    );
    // The counter does see allocations on this thread.
    let probe = std::hint::black_box(Box::new(1u8));
    assert_eq!(ALLOCATIONS.with(Cell::get) - before, 1);
    drop(probe);
}

#[test]
fn check_rejects_unsafe_trajectories() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let path = &planned(&robot, 4)[0];
    let good = Trajectory::try_new(&robot, path, 1.0).unwrap();
    good.check(&robot).unwrap();
    // Timing refuses what it cannot honour instead of producing an unsafe or empty trajectory.
    for scale in [0.0, -1.0, 1.5, f32::NAN] {
        assert!(
            matches!(
                Trajectory::try_new(&robot, path, scale),
                Err(Error::Input(_))
            ),
            "speed scale {scale}"
        );
    }
    assert!(
        matches!(
            Trajectory::try_new(&robot, &path[..6 * n], 1.0),
            Err(Error::Input(_))
        ),
        "six control points"
    );
    assert!(
        matches!(good.sample(0.0), Err(Error::Input(_))),
        "sampling at 0 Hz"
    );
    let points = good.control_points.len() / n;
    let rejects = |t: Trajectory, why: &str| {
        let err = t.check(&robot).expect_err(why);
        eprintln!("{why}: {err}");
        assert!(
            matches!(err, Error::Unsafe(_)),
            "{why} is an unsafe trajectory, not {err:?}"
        );
    };

    let mut nan = good.clone();
    nan.control_points[5 * n + 2] = f32::NAN;
    rejects(nan, "a NaN");

    let mut fast = good.clone();
    fast.knot_interval *= 0.8;
    rejects(fast, "running too fast");

    // A long move over few control points, timed to its limits, sped up 10%.
    let mut cp = vec![];
    for q0 in [-2.5, -2.5, -2.5, 0.0, 2.5, 2.5, 2.5] {
        let mut q = robot.default_q().to_vec();
        q[0] = q0;
        cp.extend(q);
    }
    let sweep = Trajectory::try_new(&robot, &cp, 1.0).unwrap();
    sweep.check(&robot).unwrap();
    let mut faster = sweep;
    faster.knot_interval *= 0.9;
    let err = faster.check(&robot).unwrap_err().to_string();
    assert!(
        err.contains("velocity") || err.contains("acceleration"),
        "{err}"
    );

    let mut spike = good.clone();
    spike.control_points[(points / 2) * n] += 0.05;
    let err = spike.check(&robot).expect_err("a jerk spike").to_string();
    assert!(
        err.contains("jerk") || err.contains("acceleration"),
        "{err}"
    );

    let mut moving_start = good.clone();
    moving_start.control_points[n] += 0.01;
    rejects(moving_start, "not at rest at the start");

    let mut out_of_range = good.clone();
    out_of_range.control_points[(points / 2) * n + 3] = robot.upper()[3] + 0.1;
    rejects(out_of_range, "leaving a joint range");

    let mut fewer = good;
    fewer.dof = n - 1;
    let err = fewer.check(&robot).unwrap_err();
    assert!(
        matches!(&err, Error::Input(m) if m.contains("joints")),
        "{err:?}"
    );
}

#[test]
fn trajectories_round_trip_through_json() {
    let robot = common::panda().unwrap();
    let trajectory = Trajectory::try_new(&robot, &planned(&robot, 4)[0], 0.7).unwrap();
    let text = serde_json::to_string(&trajectory).unwrap();
    let back: Trajectory = serde_json::from_str(&text).unwrap();
    assert_eq!(back, trajectory);
    back.check(&robot).unwrap();
}

#[test]
fn time_optimal_timing_beats_uniform_timing_within_the_limits() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let limits = [
        robot.max_velocity(),
        robot.max_acceleration(),
        robot.max_jerk(),
    ];
    // Uniform timing's knot interval: control-point differences over their limits.
    let uniform = |cp: &[f32]| {
        let diff = |k: usize, j: usize, w: &[f32]| {
            w.iter()
                .enumerate()
                .map(|(i, w)| w * cp[(k + i) * n + j])
                .sum::<f32>()
        };
        let weights: [&[f32]; 3] = [&[-1.0, 1.0], &[1.0, -2.0, 1.0], &[-1.0, 3.0, -3.0, 1.0]];
        let points = cp.len() / n;
        let mut h = 0.0f32;
        for (order, w) in weights.iter().enumerate() {
            for (j, limit) in limits[order].iter().enumerate() {
                let largest = (0..=points - w.len())
                    .map(|k| diff(k, j, w).abs())
                    .fold(0.0, f32::max);
                h = h.max((largest / limit).powf(1.0 / (order + 1) as f32));
            }
        }
        h * (points - 3) as f32
    };
    let (mut ratios, mut peak) = (vec![], [0.0f32; 3]);
    for path in planned(&robot, 16) {
        let trajectory = Trajectory::try_new(&robot, &path, 1.0).unwrap();
        trajectory.check(&robot).unwrap();
        ratios.push(trajectory.duration() / uniform(&path));
        // Four times denser than the check samples.
        let dt = trajectory.knot_interval / 256.0;
        let s = trajectory.sample(1.0 / dt).unwrap();
        for i in 0..s.positions.len() {
            let j = i % n;
            peak[0] = peak[0].max(s.velocities[i].abs() / limits[0][j]);
            peak[1] = peak[1].max(s.accelerations[i].abs() / limits[1][j]);
        }
    }
    let mean = ratios.iter().sum::<f32>() / ratios.len() as f32;
    eprintln!(
        "duration against uniform timing: mean {mean:.3}, worst {:.3}; peaks {peak:?}",
        ratios.iter().fold(0.0f32, |m, &r| m.max(r))
    );
    assert!(
        ratios.iter().all(|&r| r <= 1.0 + 1e-5),
        "slower than uniform timing: {ratios:?}"
    );
    assert!(mean < 0.95, "time-optimal timing gains little: {mean}");
    assert!(
        peak.iter().take(2).all(|&p| p <= 1.0),
        "between the checked samples a limit is exceeded: {peak:?}"
    );
}

#[test]
fn moving_starts_are_timed_from_their_motion_or_refused() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let q0 = robot.default_q().to_vec();
    // Joint 0 at 2 rad/s, on a path that stops 2 mrad later: no deceleration within the limits
    // brakes in time.
    let mut velocity = vec![0.0; n];
    velocity[0] = 2.0;
    let motion = StartMotion {
        velocity,
        acceleration: vec![0.0; n],
    };
    let h0 = 1e-3;
    let mut cp = vec![];
    for k in 0..8 {
        let mut q = q0.clone();
        q[0] += match k {
            0 => -2.0 * h0,
            1 => 0.0,
            _ => 2.0 * h0,
        };
        cp.extend(q);
    }
    let err = Trajectory::moving(&robot, &cp, &motion).unwrap_err();
    assert!(matches!(err, Error::Unsafe(_)), "{err:?}");
    // Control points that do not continue the motion are an input error.
    let still = StartMotion {
        velocity: vec![0.5; n],
        acceleration: vec![0.0; n],
    };
    assert!(matches!(
        Trajectory::moving(&robot, &cp, &still),
        Err(Error::Input(_))
    ));
    // A path that starts moving cannot be timed as if from rest.
    assert!(matches!(
        Trajectory::try_new(&robot, &cp, 1.0),
        Err(Error::Input(_))
    ));
}

#[test]
fn replanning_while_moving_is_timed_about_as_fast_as_from_rest() {
    // Halfway along planned motions, at about 3 rad/s, replanned to new goals: the motion must
    // brake and turn first, but is timed as fast as the limits allow, not at the pace the
    // replanned path's first span was fitted to (which took up to 15 times as long).
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let cpu = Device::cpu(&robot).unwrap();
    let worlds = cpu.upload(&[World::default()]).unwrap();
    let mut rng = Rng::new(7);
    let mut random = || -> Vec<f32> {
        (0..n)
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect()
    };
    let problems: Vec<PlanProblem> = (0..16)
        .map(|_| PlanProblem::new(0, random(), random()))
        .collect();
    let plans = batchplan::plan(&cpu, &worlds, &problems, &PlanOptions::default()).unwrap();
    let mut state = JointState::new(n);
    let mut replanned = 0;
    for s in plans.solved().take(12) {
        let trajectory = Trajectory::try_new(&robot, s.solution, 1.0).unwrap();
        trajectory.at(0.5 * trajectory.duration(), &mut state);
        let motion = StartMotion {
            velocity: state.velocity.clone(),
            acceleration: state.acceleration.clone(),
        };
        let rest = PlanProblem::new(0, state.position.clone(), random());
        let moving = PlanProblem {
            start_motion: Some(motion.clone()),
            ..rest.clone()
        };
        let both =
            batchplan::plan(&cpu, &worlds, &[rest, moving], &PlanOptions::default()).unwrap();
        let (Some(from_rest), Some(from_moving)) = (both.best(0), both.best(1)) else {
            continue;
        };
        let from_rest = Trajectory::try_new(&robot, from_rest, 1.0)
            .unwrap()
            .duration();
        let moving = Trajectory::moving(&robot, from_moving, &motion).unwrap();
        moving.check_from(&robot, &state).unwrap();
        assert!(
            moving.duration() < 3.0 * from_rest,
            "motion {}: {} s from its moving start, {from_rest} s from rest",
            s.index,
            moving.duration()
        );
        replanned += 1;
    }
    assert!(replanned >= 10, "only {replanned} replanned");
}

#[test]
fn paths_planned_from_a_moving_start_run_from_it() {
    // Replanned from 30%, 50% and 70% of the way along planned motions to new goals: every path
    // `plan` returns is timed from exactly that state and keeps every limit and joint range, even
    // where momentum would carry a joint out of its range in the first span.
    let robots = [
        common::panda().unwrap(),
        Robot::load(
            common::asset("so101/so101_new_calib.urdf"),
            &RobotOptions::default(),
        )
        .unwrap(),
    ];
    for robot in robots {
        let n = robot.dof();
        let cpu = Device::cpu(&robot).unwrap();
        let worlds = cpu.upload(&[World::default()]).unwrap();
        let mut rng = Rng::new(17);
        let mut random = || -> Vec<f32> {
            (0..n)
                .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
                .collect()
        };
        let problems: Vec<PlanProblem> = (0..160)
            .map(|_| PlanProblem::new(0, random(), random()))
            .collect();
        let plans = batchplan::plan(&cpu, &worlds, &problems, &PlanOptions::default()).unwrap();
        let mut state = JointState::new(n);
        let (mut moving, mut states) = (vec![], vec![]);
        for (k, s) in plans.solved().take(96).enumerate() {
            let trajectory = Trajectory::try_new(&robot, s.solution, 1.0).unwrap();
            trajectory.at([0.3, 0.5, 0.7][k % 3] * trajectory.duration(), &mut state);
            let motion = StartMotion {
                velocity: state.velocity.clone(),
                acceleration: state.acceleration.clone(),
            };
            moving.push(PlanProblem {
                start_motion: Some(motion),
                ..PlanProblem::new(0, state.position.clone(), random())
            });
            states.push(state.clone());
        }
        let replans = batchplan::plan(&cpu, &worlds, &moving, &PlanOptions::default()).unwrap();
        let solved: Vec<_> = replans.solved().collect();
        assert!(
            solved.len() >= moving.len() * 3 / 4,
            "only {} of {} replanned",
            solved.len(),
            moving.len()
        );
        for s in solved {
            let motion = s.problem.start_motion.as_ref().unwrap();
            let trajectory = Trajectory::moving(&robot, s.solution, motion).unwrap();
            if let Err(e) = trajectory.check_from(&robot, &states[s.index]) {
                panic!("replan {}: {e}", s.index);
            }
        }
        // So does every seed trajectory optimization keeps.
        let o = PlanOptions {
            seeds: 2,
            ..Default::default()
        };
        let optimized = batchplan::plan(&cpu, &worlds, &moving[..32], &o).unwrap();
        for item in (0..optimized.valid.len()).filter(|&i| optimized.valid[i]) {
            let index = item / optimized.seeds;
            let motion = moving[index].start_motion.as_ref().unwrap();
            let path = &optimized.paths[item];
            let trajectory = Trajectory::moving(&robot, path, motion).unwrap();
            if let Err(e) = trajectory.check_from(&robot, &states[index]) {
                panic!("replan {index}, seed {}: {e}", item % optimized.seeds);
            }
        }
    }
}

#[test]
fn scaled_limits_slow_every_motion() {
    let robot = common::panda().unwrap();
    // Half speed: velocity by a half, acceleration by its square, jerk by its cube.
    let scaled = |velocity, acceleration, jerk| RobotOptions {
        velocity_scale: velocity,
        acceleration_scale: acceleration,
        jerk_scale: jerk,
        ..common::panda_options()
    };
    let slow = Robot::load(
        common::asset("franka/franka_panda.urdf"),
        &scaled(0.5, 0.25, 0.125),
    )
    .unwrap();
    for j in 0..robot.dof() {
        assert_eq!(slow.max_velocity()[j], 0.5 * robot.max_velocity()[j]);
        assert_eq!(
            slow.max_acceleration()[j],
            0.25 * robot.max_acceleration()[j]
        );
        assert_eq!(slow.max_jerk()[j], 0.125 * robot.max_jerk()[j]);
    }
    for path in planned(&robot, 8) {
        let (fast, slow_trajectory) = (
            Trajectory::try_new(&robot, &path, 1.0).unwrap(),
            Trajectory::try_new(&slow, &path, 1.0).unwrap(),
        );
        slow_trajectory.check(&slow).unwrap();
        let ratio = slow_trajectory.duration() / fast.duration();
        assert!(
            (ratio - 2.0).abs() < 0.02,
            "half the limits took {ratio} times as long"
        );
    }
    for bad in [
        scaled(0.0, 1.0, 1.0),
        scaled(1.0, 1.5, 1.0),
        scaled(1.0, 1.0, f32::NAN),
    ] {
        assert!(matches!(
            Robot::load(common::asset("franka/franka_panda.urdf"), &bad),
            Err(Error::Load { .. })
        ));
    }
}

/// How the hand moves along `trajectory`: the fastest it moves and turns, and how far, from
/// central differences of its pose sampled 64 times per knot interval.
fn hand_motion(robot: &Robot, trajectory: &Trajectory) -> ([f32; 2], [f32; 2]) {
    let dt = trajectory.knot_interval / 64.0;
    let samples = trajectory.sample(1.0 / dt).unwrap();
    let poses: Vec<_> = samples
        .positions
        .chunks(robot.dof())
        .map(|q| robot.ee_pose(q, 0))
        .collect();
    let (mut peaks, mut totals) = ([0.0f32; 2], [0.0f32; 2]);
    for w in poses.windows(3) {
        let moved = w[2].position.distance(w[0].position);
        // From the difference's vector part: acos loses small angles to f32 rounding.
        let turned = 2.0
            * (w[0].rotation.inverse() * w[2].rotation)
                .xyz()
                .length()
                .asin();
        for (i, change) in [moved, turned].into_iter().enumerate() {
            peaks[i] = peaks[i].max(change / (2.0 * dt));
            totals[i] += change / 2.0;
        }
    }
    (peaks, totals)
}

#[test]
fn hand_speed_caps_hold_and_bind() {
    let robot = common::panda().unwrap();
    let paths = planned(&robot, 8);
    let mut ratios = vec![];
    for path in &paths {
        // Each cap alone, then both, at half how fast the joints' limits alone move the hand.
        let fastest = Trajectory::try_new(&robot, path, 1.0).unwrap();
        let [linear, angular] = hand_motion(&robot, &fastest).0.map(|peak| peak / 2.0);
        assert!(linear > 0.1 && angular > 0.1, "the hand barely moves");
        let caps = [
            [linear, f32::INFINITY],
            [f32::INFINITY, angular],
            [linear, angular],
        ];
        for cap in caps {
            let speed = HandSpeed {
                hand: 0,
                linear: cap[0],
                angular: cap[1],
            };
            let trajectory = Trajectory::with_hand_speed(&robot, path, &speed).unwrap();
            trajectory.check(&robot).unwrap();
            let (peaks, totals) = hand_motion(&robot, &trajectory);
            // The least time the hand's travel and turn take at its caps.
            let at_caps = (totals[0] / cap[0]).max(totals[1] / cap[1]);
            ratios.push(trajectory.duration() / at_caps);
            assert!(
                peaks.iter().zip(cap).all(|(p, c)| *p <= c * (1.0 + 1e-3)),
                "under caps {cap:?} the hand reaches {peaks:?}"
            );
            assert!(
                peaks.iter().zip(cap).any(|(p, c)| *p > 0.9 * c),
                "no cap of {cap:?} is close to binding: {peaks:?}"
            );
        }
    }
    let mean = ratios.iter().sum::<f32>() / ratios.len() as f32;
    eprintln!("duration against the caps' whole way: mean {mean:.3}");
    // Stretching the joints' timing until the hand keeps its caps averages about 1.5.
    assert!(
        mean < 1.35,
        "the caps slow the motion more than they need: {mean}"
    );
    // A hand with neither cap is timed as the joints alone allow.
    let uncapped = HandSpeed {
        hand: 0,
        linear: f32::INFINITY,
        angular: f32::INFINITY,
    };
    for path in &paths {
        assert_eq!(
            Trajectory::with_hand_speed(&robot, path, &uncapped).unwrap(),
            Trajectory::try_new(&robot, path, 1.0).unwrap()
        );
    }
    let bad = [(1, 0.2, 0.5), (0, 0.0, 0.5), (0, 0.2, f32::NAN)];
    for (hand, linear, angular) in bad {
        let speed = HandSpeed {
            hand,
            linear,
            angular,
        };
        assert!(matches!(
            Trajectory::with_hand_speed(&robot, &paths[0], &speed),
            Err(Error::Input(_))
        ));
    }
}

#[test]
fn stops_brake_to_rest_along_the_path() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let (mut state, mut end) = (JointState::new(n), JointState::new(n));
    let mut stops = 0;
    for path in planned(&robot, 12) {
        let trajectory = Trajectory::try_new(&robot, &path, 1.0).unwrap();
        for fraction in [0.2f32, 0.5] {
            let t = fraction * trajectory.duration();
            trajectory.at(t, &mut state);
            // Late in a short, fast motion, the path may end before any stop would.
            let Ok(stop) = trajectory.stop(&robot, t) else {
                continue;
            };
            // From exactly this state, within every limit, at rest at its end, on the same path.
            stop.check_from(&robot, &state).unwrap();
            // Not from one a hundredth of a joint's acceleration limit off.
            let mut off = state.clone();
            off.acceleration[1] += 0.01 * robot.max_acceleration()[1];
            assert!(matches!(
                stop.check_from(&robot, &off),
                Err(Error::Unsafe(_))
            ));
            assert!(
                trajectory.control_points.ends_with(&stop.control_points),
                "the stop leaves the path"
            );
            stop.at(stop.duration(), &mut end);
            assert!(
                end.velocity
                    .iter()
                    .chain(&end.acceleration)
                    .all(|v| v.abs() < 1e-3),
                "{end:?}"
            );
            assert!(
                stop.duration() < trajectory.duration() - t,
                "stopping took longer than going on"
            );
            stops += 1;
        }
    }
    assert!(stops >= 20, "{stops} of 24 stops");
    // A trajectory at rest stops where it is.
    let trajectory = Trajectory::try_new(&robot, &planned(&robot, 1)[0], 1.0).unwrap();
    trajectory.at(trajectory.duration(), &mut state);
    let stop = trajectory.stop(&robot, trajectory.duration()).unwrap();
    stop.check_from(&robot, &state).unwrap();
}

#[test]
fn a_trajectory_starts_and_ends_where_its_motion_does() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let path = planned(&robot, 1).remove(0);
    let trajectory = Trajectory::try_new(&robot, &path, 1.0).unwrap();
    // At rest at both ends of its path: exactly the first and last control points.
    assert_eq!(trajectory.start(), path[..n]);
    assert_eq!(trajectory.end(), path[path.len() - n..]);
    // A stop starts where it was called and ends short of the path's end.
    let t = 0.4 * trajectory.duration();
    let mut state = JointState::new(n);
    trajectory.at(t, &mut state);
    let stop = trajectory.stop(&robot, t).unwrap();
    let mut end = JointState::new(n);
    stop.at(stop.duration(), &mut end);
    let distance = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    };
    assert!(distance(&stop.start(), &state.position) < 1e-5);
    assert_eq!(stop.end(), end.position);
    assert!(
        distance(&stop.end(), &path[path.len() - n..]) > 1e-3,
        "the stop runs to the path's end"
    );
}
