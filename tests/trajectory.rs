//! Executable trajectories: limits hold everywhere along planned paths, sampling a trajectory does
//! not allocate, and `check` refuses unsafe trajectories.

#[path = "../examples/common/mod.rs"]
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use batchplan::rng::Rng;
use batchplan::{
    Device, Error, HandSpeed, IkProblem, JointState, Near, PlanOptions, PlanProblem, Robot,
    RobotOptions, StartMotion, Trajectory, World,
};
use glam::Vec3;

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
    let ik = batchplan::solve_ik(&cpu, &worlds, &goals, &common::fixture_ik()).unwrap();
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

/// A call's paths are timed in one call as each is alone, in their order, a path that cannot be
/// timed giving its own error among them.
#[test]
fn many_paths_are_timed_in_one_call_as_each_alone() {
    let robot = common::panda().unwrap();
    let mut paths = planned(&robot, 16);
    paths.insert(3, vec![0.0; 5]);
    let paths: Vec<&[f32]> = paths.iter().map(Vec::as_slice).collect();
    let devices = [
        Device::cpu(&robot).unwrap(),
        Device::cpu_threads(&robot, 3).unwrap(),
    ];
    for device in devices {
        let name = device.name();
        for count in [paths.len(), 1, 0] {
            let timed = batchplan::time(&device, &paths[..count], 0.5, &[]).unwrap();
            assert_eq!(timed.len(), count, "{name}");
            for (i, (timed, path)) in timed.iter().zip(&paths).enumerate() {
                match (timed, Trajectory::try_new(&robot, path, 0.5)) {
                    (Ok(timed), Ok(alone)) => assert_eq!(*timed, alone, "{name}: path {i}"),
                    (Err(Error::Input(_)), Err(Error::Input(_))) => assert_eq!(i, 3, "{name}"),
                    (timed, alone) => panic!("{name}: path {i}: {timed:?} against {alone:?}"),
                }
            }
        }
        // Each path under its own caps: none, one throughout, one near its end.
        let caps: Vec<Vec<HandSpeed>> = (0..paths.len())
            .map(|i| {
                if paths[i].len() < 7 {
                    return vec![];
                }
                let near = Near {
                    point: robot.ee_pose(&paths[i][paths[i].len() - 7..], 0).position,
                    radius: 0.1,
                };
                let cap = |near: Option<Near>| HandSpeed {
                    hand: 0,
                    linear: 0.1,
                    angular: f32::INFINITY,
                    near,
                };
                match i % 3 {
                    0 => vec![],
                    1 => vec![cap(None)],
                    _ => vec![cap(Some(near))],
                }
            })
            .collect();
        let lists: Vec<&[HandSpeed]> = caps.iter().map(Vec::as_slice).collect();
        let (full, half) = (
            batchplan::time(&device, &paths, 1.0, &lists).unwrap(),
            batchplan::time(&device, &paths, 0.5, &lists).unwrap(),
        );
        let mut slowed = 0;
        for (i, path) in paths.iter().enumerate() {
            let Ok(alone) = Trajectory::with_hand_speed(&robot, path, &caps[i]) else {
                assert_eq!(i, 3, "{name}");
                assert!(matches!(full[i], Err(Error::Input(_))), "{name}");
                continue;
            };
            let (full, half) = (full[i].as_ref().unwrap(), half[i].as_ref().unwrap());
            assert_eq!(*full, alone, "{name}: path {i}");
            assert_eq!(half.time_points, alone.time_points, "{name}: path {i}");
            assert_eq!(
                half.knot_interval,
                2.0 * alone.knot_interval,
                "{name}: path {i}"
            );
            let free = Trajectory::try_new(&robot, path, 1.0).unwrap();
            slowed += usize::from(alone.duration() > free.duration());
            assert_eq!(caps[i].is_empty(), alone == free, "{name}: path {i}");
        }
        assert!(slowed >= 8, "{name}: {slowed} paths slowed by their caps");
        // A list for some paths only is refused.
        assert!(matches!(
            batchplan::time(&device, &paths, 1.0, &lists[..2]),
            Err(Error::Input(_))
        ));
        // The paths differ, so one's trajectory in another's place would show.
        let timed = batchplan::time(&device, &paths, 0.5, &[]).unwrap();
        let durations: Vec<f32> = timed.iter().flatten().map(Trajectory::duration).collect();
        assert!(
            durations.windows(2).any(|w| w[0] != w[1]),
            "{name}: {durations:?}"
        );
    }
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

/// A straight line is timed close to the soonest its slowest joint could cover its distance: the
/// time map is not stretched as a whole for what its smoothing overshoots braking into the end.
#[test]
fn straight_lines_take_little_longer_than_their_slowest_joint_needs() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let (velocity, acceleration) = (robot.max_velocity(), robot.max_acceleration());
    let mut rng = Rng::new(7);
    let (mut ratios, mut cruising) = (vec![], 0);
    for _ in 0..16 {
        // Most of a sweep across the front, long enough to cruise at a velocity limit.
        let start = [-1.2, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8];
        let across = [1.2, -0.4, 0.3, -2.0, 0.3, 1.9, 0.5];
        let part = rng.range(0.7, 1.0);
        let end: Vec<f32> = (0..n)
            .map(|j| start[j] + part * (across[j] - start[j]))
            .collect();
        // Ends tripled, so it rests there; the rest evenly spaced.
        let points: usize = 24;
        let mut path = Vec::with_capacity(points * n);
        for i in 0..points {
            let u = i.saturating_sub(2).min(points - 5) as f32 / (points - 5) as f32;
            path.extend((0..n).map(|j| start[j] + u * (end[j] - start[j])));
        }
        // Rest to rest at the acceleration limit, cruising at the velocity limit if it gets there:
        // no joint can cover its distance sooner.
        let soonest = (0..n)
            .map(|j| {
                let (d, v, a) = ((end[j] - start[j]).abs(), velocity[j], acceleration[j]);
                cruising += usize::from(d > v * v / a);
                if d > v * v / a {
                    d / v + v / a
                } else {
                    2.0 * (d / a).sqrt()
                }
            })
            .fold(0.0, f32::max);
        let trajectory = Trajectory::try_new(&robot, &path, 1.0).unwrap();
        trajectory.check(&robot).unwrap();
        ratios.push(trajectory.duration() / soonest);
    }
    assert!(cruising > 0, "no joint reaches its velocity limit");
    assert!(
        ratios.iter().all(|&r| r >= 1.0),
        "sooner than a joint can move: {ratios:?}"
    );
    // The rest is the jerk limit's: the acceleration cannot change at once.
    assert!(
        ratios.iter().all(|&r| r < 1.19),
        "longer than the limits need: {ratios:?}"
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
    let (mut replanned, mut against_rest) = (0, 0.0);
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
        against_rest += moving.duration() / from_rest;
        replanned += 1;
    }
    assert!(replanned >= 10, "only {replanned} replanned");
    // 1.813 on average. A time map kept on the quarter-density measure alone, never measured at
    // the full density, takes 1.851: the bound sits between.
    let mean = against_rest / replanned as f32;
    assert!(mean < 1.845, "{mean} times as long as from rest on average");
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
fn a_moving_start_is_timed_where_a_coarse_measure_finds_no_map() {
    // A path `plan` returned for the SO-101 from halfway along a motion (seed 17's 23rd replan in
    // `paths_planned_from_a_moving_start_run_from_it`). Turning candidates down at a quarter of
    // the checking density alone steers the timing away from every map that fits.
    let robot = Robot::load(
        common::asset("so101/so101_new_calib.urdf"),
        &RobotOptions::default(),
    )
    .unwrap();
    let state = JointState {
        position: vec![
            0.3351964,
            -0.82430524,
            0.24536689,
            -0.71683973,
            1.3519034,
            0.6695086,
        ],
        velocity: vec![
            1.0468822, 1.4234978, 1.343335, -1.3973533, 2.6722677, -1.9513334,
        ],
        acceleration: vec![
            -0.031902675,
            -0.043385915,
            -0.040934008,
            0.042589433,
            -0.08144881,
            0.05946624,
        ],
    };
    #[rustfmt::skip] // one control point per row
    let path = [
        -0.22735098, -1.5892296, -0.4764807, 0.03403575, -0.084054455, 1.7180675,
        0.3367152, -0.82223976, 0.24731562, -0.7188673, 1.355781, 0.6666776,
        0.8916687, -0.06764277, 0.9594195, -1.459605, 2.7723513, -0.36772642,
        0.33694562, -0.71867406, 0.19489802, -0.7650671, 1.285549, 0.6531232,
        0.2518697, -0.72754353, 0.03119972, -0.69898456, 0.9951236, 0.79812276,
        0.22435135, -0.659929, -0.057733312, -0.70893735, 0.8528402, 0.8360644,
        0.187403, -0.6051369, -0.15876673, -0.70630324, 0.6864857, 0.891583,
        0.15045464, -0.5503448, -0.25980014, -0.70366913, 0.52013135, 0.9471016,
        0.11350629, -0.49555272, -0.36083353, -0.701035, 0.35377687, 1.0026202,
        0.076557934, -0.4407606, -0.46186697, -0.6984009, 0.1874224, 1.0581388,
        0.039609578, -0.3859685, -0.56290036, -0.69576675, 0.021067973, 1.1136575,
        0.0026612019, -0.3311764, -0.6639338, -0.69313264, -0.14528643, 1.1691761,
        -0.03428713, -0.27638435, -0.76496714, -0.69049853, -0.31164074, 1.2246946,
        -0.07123551, -0.22159225, -0.8660006, -0.6878644, -0.47799528, 1.2802132,
        -0.10818383, -0.1668002, -0.9670339, -0.68523026, -0.6443497, 1.3357319,
        -0.14513218, -0.112008095, -1.0680672, -0.68259615, -0.810704, 1.3912504,
        -0.18208054, -0.05721599, -1.1691008, -0.67996204, -0.9770586, 1.446769,
        -0.21902892, -0.0024238816, -1.2701341, -0.67732793, -1.1434128, 1.5022876,
        -0.25597724, 0.05236822, -1.3711675, -0.6746938, -1.3097675, 1.5578063,
        -0.29292563, 0.10716028, -1.4722009, -0.67205966, -1.4761219, 1.6133249,
        -0.33050013, 0.16288093, -1.5749465, -0.6693809, -1.6452954, 1.6697843,
        -0.36682236, 0.21674442, -1.6742678, -0.66679144, -1.8088307, 1.7243621,
        -0.36682236, 0.21674442, -1.6742678, -0.66679144, -1.8088307, 1.7243621,
        -0.36682236, 0.21674442, -1.6742678, -0.66679144, -1.8088307, 1.7243621,
    ];
    let motion = StartMotion {
        velocity: state.velocity.clone(),
        acceleration: state.acceleration.clone(),
    };
    let trajectory = Trajectory::moving(&robot, &path, &motion).unwrap();
    trajectory.check_from(&robot, &state).unwrap();
}

#[test]
fn a_moving_start_keeps_its_limits_between_the_samples_checked() {
    // A path `plan` returned for the SO-101 from 15% of the way along a motion. Its timing keeps
    // the limits at every sample `check_from` reads. Where it crosses the path's knots,
    // acceleration has a corner and jerk jumps: held only at evenly spaced samples, or with one
    // sample at each crossing rather than one either side, its jerk reaches 1.038 times a limit.
    let robot = Robot::load(
        common::asset("so101/so101_new_calib.urdf"),
        &RobotOptions::default(),
    )
    .unwrap();
    let motion = StartMotion {
        velocity: vec![
            0.24729584,
            0.25957254,
            -0.6956112,
            0.7756721,
            -0.19901524,
            -0.19785903,
        ],
        acceleration: vec![
            1.1482126,
            1.2052137,
            -3.2297754,
            3.6015053,
            -0.9240415,
            -0.91867447,
        ],
    };
    #[rustfmt::skip] // one control point per row
    let path = [
        0.25124088, 0.3646535, 0.85718244, -1.0552261, -0.61234355, 1.0622548,
        0.2607916, 0.37467834, 0.83031774, -1.0252695, -0.6200297, 1.0546134,
        0.44307217, 0.566008, 0.31758595, -0.45352513, -0.7667228, 0.9087724,
        0.24289702, 0.43632713, 0.654251, -0.8623696, -0.59685236, 0.9772505,
        0.19621414, 0.46775845, 0.55916214, -0.7897677, -0.5505072, 0.922921,
        0.14953128, 0.49918973, 0.46407327, -0.7171658, -0.504162, 0.8685915,
        0.10284838, 0.53062105, 0.36898434, -0.6445639, -0.45781678, 0.8142619,
        0.0561655, 0.56205237, 0.27389544, -0.571962, -0.4114716, 0.75993234,
        0.009482652, 0.5934837, 0.17880666, -0.4993601, -0.36512646, 0.7056029,
        -0.037200212, 0.624915, 0.08371776, -0.42675823, -0.3187813, 0.6512733,
        -0.08388314, 0.6563463, -0.011371195, -0.35415626, -0.27243605, 0.59694374,
        -0.13056603, 0.68777764, -0.106460094, -0.28155434, -0.22609088, 0.54261416,
        -0.1772489, 0.71920896, -0.201549, -0.20895243, -0.17974567, 0.4882846,
        -0.22393176, 0.7506403, -0.29663783, -0.13635051, -0.1334005, 0.43395507,
        -0.2706146, 0.7820716, -0.39172655, -0.06374872, -0.087055385, 0.37962562,
        -0.3172975, 0.8135029, -0.4868155, 0.008853257, -0.04071021, 0.32529604,
        -0.36398032, 0.8449342, -0.58190435, 0.08145505, 0.0056349635, 0.27096653,
        -0.41066328, 0.8763656, -0.6769934, 0.1540572, 0.051980257, 0.2166369,
        -0.45734617, 0.90779686, -0.77208227, 0.226659, 0.09832543, 0.16230732,
        -0.50402904, 0.9392282, -0.8671712, 0.29926103, 0.1446706, 0.10797775,
        -0.550712, 0.9706595, -0.96226007, 0.37186283, 0.19101578, 0.053648233,
        -0.5973948, 1.0020908, -1.0573488, 0.4444648, 0.23736095, -0.0006812662,
        -0.5973948, 1.0020908, -1.0573488, 0.4444648, 0.23736095, -0.0006812662,
        -0.5973948, 1.0020908, -1.0573488, 0.4444648, 0.23736095, -0.0006812662,
    ];
    let trajectory = Trajectory::moving(&robot, &path, &motion).unwrap();
    // Jerk by differences of acceleration: finer steps than these read f32 rounding.
    let (samples, mut state) = (4_000, JointState::new(robot.dof()));
    let step = trajectory.duration() / samples as f32;
    let (mut acceleration, mut jerk) = (0.0f32, 0.0f32);
    let mut last: Option<Vec<f32>> = None;
    for sample in 0..=samples {
        trajectory.at(step * sample as f32, &mut state);
        for (j, a) in state.acceleration.iter().enumerate() {
            acceleration = acceleration.max(a.abs() / robot.max_acceleration()[j]);
            if let Some(last) = &last {
                jerk = jerk.max(((a - last[j]) / step).abs() / robot.max_jerk()[j]);
            }
        }
        last = Some(state.acceleration.clone());
    }
    assert!(
        jerk > 0.9,
        "the path does not come near a jerk limit: {jerk}"
    );
    assert!(
        acceleration < 1.001,
        "{acceleration} times an acceleration limit"
    );
    assert!(jerk < 1.001, "{jerk} times a jerk limit");
}

#[test]
fn a_replan_keeps_the_start_only_the_timings_second_pass_fits() {
    // Two UR5e replans from fast starts among posts whose spline follows the start motion only
    // once it stops at every waypoint, and then only by the timing's second pass at the first
    // knot interval tried: left out there, neither is planned.
    let ur5e = Robot::load(
        common::asset("ur5e/ur_description/urdf/ur5e.urdf"),
        &RobotOptions::default(),
    )
    .unwrap();
    let cpu = Device::cpu(&ur5e).unwrap();
    let (scene, _) = common::far_reaches(&ur5e, &cpu, 4, /* above_table */ true);
    let worlds = cpu.upload(&scene).unwrap();
    let cases = [
        (
            1,
            [
                -6.0853887, -1.1014895, 0.9422271, -6.1463857, -4.681736, -3.8040044,
            ],
            [
                0.057526644,
                0.38948536,
                -0.08450991,
                -0.8712345,
                -0.8916578,
                -0.41459402,
            ],
            [
                -3.190817, 5.4199834, -0.5299742, 5.873759, 1.9869857, 1.662416,
            ],
        ),
        (
            3,
            [
                6.1318765, -2.096115, 1.3915989, -3.4039252, -4.8458357, -1.5567546,
            ],
            [
                0.5194375,
                -0.8530892,
                -0.8761967,
                -0.09256578,
                -2.7561107,
                -1.8844876,
            ],
            [
                3.9170928, -3.049145, 2.2163622, -3.614392, 5.3634467, 0.3522582,
            ],
        ),
    ];
    for (world, start, velocity, goal) in cases {
        let state = JointState {
            position: start.to_vec(),
            velocity: velocity.to_vec(),
            acceleration: vec![0.0; 6],
        };
        let motion = StartMotion {
            velocity: state.velocity.clone(),
            acceleration: state.acceleration.clone(),
        };
        let problem = PlanProblem {
            start_motion: Some(motion.clone()),
            ..PlanProblem::new(world, start.to_vec(), goal.to_vec())
        };
        let plans = batchplan::plan(&cpu, &worlds, &[problem], &PlanOptions::default()).unwrap();
        let path = plans
            .best(0)
            .unwrap_or_else(|| panic!("world {world}: {:?}", plans.failures[0]));
        let trajectory = Trajectory::moving(&ur5e, path, &motion).unwrap();
        trajectory.check_from(&ur5e, &state).unwrap();
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
                near: None,
            };
            let trajectory = Trajectory::with_hand_speed(&robot, path, &[speed]).unwrap();
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
        near: None,
    };
    for path in &paths {
        assert_eq!(
            Trajectory::with_hand_speed(&robot, path, &[uncapped]).unwrap(),
            Trajectory::try_new(&robot, path, 1.0).unwrap()
        );
    }
    let bad = [(1, 0.2, 0.5), (0, 0.0, 0.5), (0, 0.2, f32::NAN)];
    for (hand, linear, angular) in bad {
        let speed = HandSpeed {
            hand,
            linear,
            angular,
            near: None,
        };
        assert!(matches!(
            Trajectory::with_hand_speed(&robot, &paths[0], &[speed]),
            Err(Error::Input(_))
        ));
    }
}

/// How fast hand `hand` moves along `trajectory`, 256 times per knot interval, and whether it is
/// within `near` there, so the ball's very edge is measured. A sample's speed is read from its
/// joint velocities: the hand's positions 5 ms of that motion either side of the sample, far
/// enough apart for single precision.
fn hand_speeds(
    robot: &Robot,
    trajectory: &Trajectory,
    hand: usize,
    near: &Near,
) -> Vec<(bool, f32)> {
    let n = robot.dof();
    let samples = trajectory.sample(256.0 / trajectory.knot_interval).unwrap();
    let states = samples
        .positions
        .chunks(n)
        .zip(samples.velocities.chunks(n));
    let h = 0.005;
    let each = states.map(|(q, dq)| {
        let moved = |by: f32| -> Vec3 {
            let q: Vec<f32> = q.iter().zip(dq).map(|(q, dq)| q + by * dq).collect();
            robot.ee_pose(&q, hand).position
        };
        let inside = moved(0.0).distance(near.point) <= near.radius;
        (inside, moved(h).distance(moved(-h)) / (2.0 * h))
    });
    each.collect()
}

/// The fastest the hand moves within the ball (`true`) or outside it.
fn peak(speeds: &[(bool, f32)], inside: bool) -> f32 {
    let of = speeds.iter().filter(|(i, _)| *i == inside);
    of.map(|(_, v)| *v).fold(0.0, f32::max)
}

/// A ball the hand only clips, or one a few millimeters across, slows the hand while it is
/// inside and the motion by little: the cap is neither missed where the hand crosses between two
/// of the timing's samples nor answered by slowing the whole motion.
#[test]
fn a_ball_the_hand_only_clips_slows_it_there_alone() {
    let robot = common::panda().unwrap();
    let cap = 0.05;
    let (mut clipped, mut fast) = (0, 0);
    for path in &planned(&robot, 8)[..4] {
        let free = Trajectory::try_new(&robot, path, 1.0).unwrap();
        let whole = HandSpeed {
            hand: 0,
            linear: cap,
            angular: f32::INFINITY,
            near: None,
        };
        let throughout = Trajectory::with_hand_speed(&robot, path, &[whole]).unwrap();
        let dt = free.knot_interval / 64.0;
        let at: Vec<Vec3> = free
            .sample(1.0 / dt)
            .unwrap()
            .positions
            .chunks(robot.dof())
            .map(|q| robot.ee_pose(q, 0).position)
            .collect();
        for tenth in 2..9 {
            // A ball beside the path where the hand is after this share of the motion, reaching
            // across it by a hundredth or a thousandth of its radius.
            let k = at.len() * tenth / 10;
            let along = (at[k + 1] - at[k - 1]).normalize();
            let aside = along.any_orthonormal_vector();
            for (radius, depth) in [(0.005, 0.999), (0.01, 0.99), (0.05, 0.99)] {
                let near = Near {
                    point: at[k] + aside * depth * radius,
                    radius,
                };
                let speed = HandSpeed {
                    near: Some(near),
                    ..whole
                };
                let zoned = Trajectory::with_hand_speed(&robot, path, &[speed]).unwrap();
                zoned.check(&robot).unwrap();
                let speeds = hand_speeds(&robot, &zoned, 0, &near);
                if !speeds.iter().any(|(inside, _)| *inside) {
                    continue;
                }
                clipped += 1;
                fast += usize::from(peak(&speeds, false) > 4.0 * cap);
                let within = peak(&speeds, true);
                assert!(
                    within <= cap * (1.0 + 1e-3),
                    "{within} m/s inside a ball of {radius} m clipped by {depth}"
                );
                let (zoned, free, throughout) =
                    (zoned.duration(), free.duration(), throughout.duration());
                assert!(
                    zoned > free && zoned < free + 2.0 && zoned < 0.5 * throughout,
                    "{zoned} s past a ball of {radius} m, {free} s without, {throughout} s capped throughout"
                );
            }
        }
    }
    assert!(
        clipped >= 60 && fast >= 40,
        "{clipped} balls clipped, {fast} at speed"
    );
}

/// A cap near a point holds for the hand it names: on two arms, the second hand arrives slowly
/// at its goal while the first stands still elsewhere.
#[test]
fn a_hand_speed_near_a_point_follows_its_own_hand() {
    let robot = common::two_pandas().unwrap();
    let n = robot.dof();
    let cpu = Device::cpu(&robot).unwrap();
    let worlds = cpu.upload(&[World::default()]).unwrap();
    let start = robot.default_q().to_vec();
    let mut goal = start.clone();
    let joints = robot.hand_joints(1);
    goal[joints[0]] += 0.8;
    goal[joints[3]] += 0.5;
    let problem = PlanProblem::new(0, &start[..], &goal[..]);
    let planned = batchplan::plan(&cpu, &worlds, &[problem], &PlanOptions::default()).unwrap();
    let path = planned
        .solved()
        .next()
        .expect("a free motion plans")
        .solution;
    let still = [&start, &goal].map(|q| robot.ee_pose(q, 0).position);
    assert!(still[0].distance(still[1]) < 1e-6, "the first hand moves");
    let near = Near {
        point: robot.ee_pose(&path[path.len() - n..], 1).position,
        radius: 0.08,
    };
    let speed = HandSpeed {
        hand: 1,
        linear: 0.03,
        angular: f32::INFINITY,
        near: Some(near),
    };
    let trajectory = Trajectory::with_hand_speed(&robot, path, &[speed]).unwrap();
    let speeds = hand_speeds(&robot, &trajectory, 1, &near);
    let (within, outside) = (peak(&speeds, true), peak(&speeds, false));
    assert!(
        within <= speed.linear * (1.0 + 1e-3) && within > 0.9 * speed.linear,
        "the second hand arrives at {within} m/s under a cap of {}",
        speed.linear
    );
    assert!(outside > 4.0 * speed.linear, "{outside} m/s on the way");
}

/// A cap that holds near a point slows the hand there and nowhere else: a planned motion arrives
/// at its goal under the cap, runs as fast as the joints allow before that, and takes little more
/// than the capped stretch adds.
#[test]
fn a_hand_speed_near_a_point_holds_there_alone() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let cap = 0.05;
    let mut zoned_paths = 0;
    for path in &planned(&robot, 8) {
        let goal = robot.ee_pose(&path[path.len() - n..], 0).position;
        let near = Near {
            point: goal,
            radius: 0.1,
        };
        let speed = |near: Option<Near>| HandSpeed {
            hand: 0,
            linear: cap,
            angular: f32::INFINITY,
            near,
        };
        let free = Trajectory::try_new(&robot, path, 1.0).unwrap();
        let whole = Trajectory::with_hand_speed(&robot, path, &[speed(None)]).unwrap();
        let zoned = Trajectory::with_hand_speed(&robot, path, &[speed(Some(near))]).unwrap();
        zoned.check(&robot).unwrap();
        let speeds = hand_speeds(&robot, &zoned, 0, &near);
        let (within, outside) = (peak(&speeds, true), peak(&speeds, false));
        assert!(
            within <= cap * (1.0 + 1e-3),
            "{within} m/s near the goal under a cap of {cap}"
        );
        assert!(within > 0.9 * cap, "the cap does not bind: {within} m/s");
        // The hand's travel inside the zone at the cap: what the cap must add, at least.
        let dt = zoned.knot_interval / 256.0;
        let inside: f32 = speeds.iter().filter(|(i, _)| *i).map(|(_, v)| v * dt).sum();
        assert!(inside > 0.08, "{inside} m of the path within the zone");
        if outside < 4.0 * cap {
            // A motion too short to pick up speed before the zone shows nothing.
            continue;
        }
        zoned_paths += 1;
        let (free, whole, zoned) = (free.duration(), whole.duration(), zoned.duration());
        assert!(
            zoned > free && zoned < free + 1.25 * inside / cap && zoned < 0.7 * whole,
            "{zoned} s with the cap near the goal, {free} s without, {whole} s under it throughout"
        );
    }
    assert!(
        zoned_paths >= 4,
        "{zoned_paths} paths reach speed before the zone"
    );
}

/// Several caps hold together: one at each end of a motion slows both ends, the tighter of two
/// in one place binds, and a cap whose point the hand never comes near, alone, changes nothing.
#[test]
fn hand_speeds_hold_together_each_where_it_is() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let path = &planned(&robot, 8)[0];
    let ends = [&path[..n], &path[path.len() - n..]].map(|q| robot.ee_pose(q, 0).position);
    assert!(
        ends[0].distance(ends[1]) > 0.3,
        "the motion is short: {ends:?}"
    );
    let zone = |point: Vec3, linear: f32| HandSpeed {
        hand: 0,
        linear,
        angular: f32::INFINITY,
        near: Some(Near {
            point,
            radius: 0.08,
        }),
    };
    let both = [zone(ends[0], 0.04), zone(ends[1], 0.08)];
    let trajectory = Trajectory::with_hand_speed(&robot, path, &both).unwrap();
    for speed in both {
        let near = speed.near.unwrap();
        let speeds = hand_speeds(&robot, &trajectory, 0, &near);
        let within = speeds.iter().filter(|(i, _)| *i).map(|(_, v)| *v);
        let within = within.fold(0.0, f32::max);
        assert!(
            within <= speed.linear * (1.0 + 1e-3) && within > 0.9 * speed.linear,
            "{within} m/s under a cap of {}",
            speed.linear
        );
    }
    let alone = |speed: HandSpeed| Trajectory::with_hand_speed(&robot, path, &[speed]).unwrap();
    assert!(trajectory.duration() > alone(both[0]).duration().max(alone(both[1]).duration()));

    // Two caps in one place, given either way round: the tighter one's timing.
    let (tight, loose) = (zone(ends[1], 0.04), zone(ends[1], 0.08));
    for pair in [[tight, loose], [loose, tight]] {
        assert_eq!(
            Trajectory::with_hand_speed(&robot, path, &pair).unwrap(),
            alone(tight)
        );
    }
    // A cap on the hand's speed near the goal with one on its turning throughout: each binds,
    // though each leaves the other's kind uncapped.
    let turning = HandSpeed {
        hand: 0,
        linear: f32::INFINITY,
        angular: hand_motion(&robot, &alone(tight)).0[1] / 2.0,
        near: None,
    };
    let mixed = Trajectory::with_hand_speed(&robot, path, &[tight, turning]).unwrap();
    let turned = hand_motion(&robot, &mixed).0[1];
    assert!(
        turned <= turning.angular * (1.0 + 1e-3) && turned > 0.9 * turning.angular,
        "the hand turns at {turned} rad/s under a cap of {}",
        turning.angular
    );
    let speeds = hand_speeds(&robot, &mixed, 0, &tight.near.unwrap());
    let within = speeds.iter().filter(|(i, _)| *i).map(|(_, v)| *v);
    let within = within.fold(0.0, f32::max);
    assert!(
        within <= tight.linear * (1.0 + 1e-3) && mixed.duration() > alone(tight).duration(),
        "{within} m/s near the goal under a cap of {}",
        tight.linear
    );
    // A cap out of the hand's way, and no caps at all.
    let free = Trajectory::try_new(&robot, path, 1.0).unwrap();
    let away = zone(ends[1] + Vec3::new(0.0, 0.0, 2.0), 0.04);
    assert_eq!(alone(away), free);
    assert_eq!(
        Trajectory::with_hand_speed(&robot, path, &[]).unwrap(),
        free
    );

    let bad = [
        (Vec3::ZERO, 0.0),
        (Vec3::ZERO, -0.1),
        (Vec3::NAN, 0.1),
        (Vec3::ZERO, f32::INFINITY),
    ];
    for (point, radius) in bad {
        let speed = HandSpeed {
            near: Some(Near { point, radius }),
            ..tight
        };
        assert!(
            matches!(
                Trajectory::with_hand_speed(&robot, path, &[speed]),
                Err(Error::Input(_))
            ),
            "{point} within {radius}"
        );
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

/// A trajectory can be stopped from the states it passes through before it brakes for its own
/// end: at a joint's velocity limit, since a stop's velocity limit is the full one where it
/// starts; and while it still gathers speed, since a stop may carry on with the jerk it has before
/// braking, which also stops sooner than a map that brakes at once and creeps to the path's end.
#[test]
fn motions_can_be_stopped_before_they_brake_for_their_end() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let (mut early, mut early_refused, mut early_slow) = (0, 0, 0);
    let (mut fast, mut fast_stopped) = (0, 0);
    let mut state = JointState::new(n);
    for path in planned(&robot, 48) {
        let trajectory = Trajectory::try_new(&robot, &path, 1.0).unwrap();
        // Every hundredth of the first 60%: later, more and more are braking already.
        for k in 1..=60 {
            let t = 0.01 * k as f32 * trajectory.duration();
            trajectory.at(t, &mut state);
            let ratios = state.velocity.iter().zip(robot.max_velocity());
            let at_limit = ratios.map(|(v, l)| v.abs() / l).fold(0.0, f32::max) > 0.995;
            let stop = trajectory.stop(&robot, t);
            early += usize::from(k <= 20);
            early_refused += usize::from(k <= 20 && stop.is_err());
            fast += usize::from(at_limit);
            fast_stopped += usize::from(at_limit && stop.is_ok());
            let Ok(stop) = stop else { continue };
            stop.check_from(&robot, &state).unwrap();
            early_slow += usize::from(k <= 20 && stop.duration() > trajectory.duration() - t);
        }
    }
    // With a stop's velocity held to the margin under the limit at its start too, none of those
    // at a limit stops (19 on an M4 Pro, 9 of them stopped; 3 other sets of worlds: 9 of 12, 20 of
    // 25, 13 of 17).
    assert!(fast >= 8, "only {fast} states at a velocity limit");
    assert!(
        4 * fast_stopped >= fast,
        "{fast_stopped} of {fast} at a velocity limit stop"
    );
    // Braking at once only, 153 of 960 are refused; with the starting jerks, 52 (32-52 over three
    // other sets of worlds).
    assert!(early >= 600, "only {early} stops asked for early on");
    assert!(
        10 * early_refused <= early,
        "{early_refused} of {early} stops in a motion's first fifth refused"
    );
    // Carrying on only where braking at once finds no stop at all, 66 take longer than going on;
    // one does.
    assert!(
        50 * early_slow <= early,
        "{early_slow} of {early} stops in a motion's first fifth take longer than going on"
    );
}
