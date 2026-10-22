//! Executable trajectories: limits hold everywhere along planned paths, sampling a trajectory does
//! not allocate, and `check` refuses unsafe trajectories.

#[path = "../examples/common/mod.rs"]
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use batchplan::rng::Rng;
use batchplan::{
    Device, Error, IkOptions, IkProblem, JointState, PlanOptions, PlanProblem, Robot, RobotOptions,
    StartMotion, Trajectory, World,
};

/// Counts allocations made by the current thread, so other test threads cannot interfere.
struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
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
        let trajectory = Trajectory::new(&robot, &path, 1.0).unwrap();
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

#[test]
fn slower_speed_scales_stretch_time() {
    let robot = common::panda().unwrap();
    let path = &planned(&robot, 4)[0];
    let fast = Trajectory::new(&robot, path, 1.0).unwrap();
    let slow = Trajectory::new(&robot, path, 0.5).unwrap();
    assert!((slow.duration() - 2.0 * fast.duration()).abs() < 1e-5 * fast.duration());
    slow.check(&robot).unwrap();
}

#[test]
fn sampling_does_not_allocate() {
    let robot = common::panda().unwrap();
    let trajectory = Trajectory::new(&robot, &planned(&robot, 4)[0], 1.0).unwrap();
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
    let good = Trajectory::new(&robot, path, 1.0).unwrap();
    good.check(&robot).unwrap();
    // Timing refuses what it cannot honour instead of producing an unsafe or empty trajectory.
    for scale in [0.0, -1.0, 1.5, f32::NAN] {
        assert!(
            matches!(Trajectory::new(&robot, path, scale), Err(Error::Input(_))),
            "speed scale {scale}"
        );
    }
    assert!(
        matches!(
            Trajectory::new(&robot, &path[..6 * n], 1.0),
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

    rejects(
        Trajectory {
            knot_interval: good.knot_interval * 0.8,
            ..good.clone()
        },
        "running too fast",
    );

    // A long move over few control points, timed to its limits, sped up 10%.
    let mut cp = vec![];
    for q0 in [-2.5, -2.5, -2.5, 0.0, 2.5, 2.5, 2.5] {
        let mut q = robot.default_q().to_vec();
        q[0] = q0;
        cp.extend(q);
    }
    let sweep = Trajectory::new(&robot, &cp, 1.0).unwrap();
    sweep.check(&robot).unwrap();
    let err = Trajectory {
        knot_interval: sweep.knot_interval * 0.9,
        ..sweep
    }
    .check(&robot)
    .unwrap_err()
    .to_string();
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

    let err = Trajectory {
        dof: n - 1,
        ..good.clone()
    }
    .check(&robot)
    .unwrap_err();
    assert!(
        matches!(&err, Error::Input(m) if m.contains("joints")),
        "{err:?}"
    );
}

#[test]
fn trajectories_round_trip_through_json() {
    let robot = common::panda().unwrap();
    let trajectory = Trajectory::new(&robot, &planned(&robot, 4)[0], 0.7).unwrap();
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
        let trajectory = Trajectory::new(&robot, &path, 1.0).unwrap();
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
        Trajectory::new(&robot, &cp, 1.0),
        Err(Error::Input(_))
    ));
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
            Trajectory::new(&robot, &path, 1.0).unwrap(),
            Trajectory::new(&slow, &path, 1.0).unwrap(),
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

#[test]
fn stops_brake_to_rest_along_the_path() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let (mut state, mut end) = (JointState::new(n), JointState::new(n));
    let mut stops = 0;
    for path in planned(&robot, 12) {
        let trajectory = Trajectory::new(&robot, &path, 1.0).unwrap();
        for fraction in [0.2f32, 0.5] {
            let t = fraction * trajectory.duration();
            trajectory.at(t, &mut state);
            // Late in a short, fast motion, the path may end before any stop would.
            let Ok(stop) = trajectory.stop(&robot, t) else {
                continue;
            };
            // From exactly this state, within every limit, at rest at its end, on the same path.
            stop.check_from(&robot, &state).unwrap();
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
    let trajectory = Trajectory::new(&robot, &planned(&robot, 1)[0], 1.0).unwrap();
    trajectory.at(trajectory.duration(), &mut state);
    let stop = trajectory.stop(&robot, trajectory.duration()).unwrap();
    stop.check_from(&robot, &state).unwrap();
}
