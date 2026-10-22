//! Executable trajectories: planned B-spline paths timed so that position, velocity, acceleration
//! and jerk stay within the robot's limits.
//!
//! A trajectory is the planned path `P(s)`, a uniform cubic B-spline over `s` in spans, and a time
//! map `s = σ(t)`, a uniform cubic B-spline over time. Two timings are computed and the faster
//! kept:
//! - Uniform: `σ(t) = t / h`. On the path, velocity is a quadratic B-spline over the control-point
//!   differences, acceleration a linear one over the second differences, and jerk is constant per
//!   span, so `h` chosen from those differences bounds the whole curve exactly.
//! - Time-optimal (TOPP-RA, Pham & Pham 2018): the fastest velocity- and acceleration-limited
//!   timing of the same path, smoothed and stretched until velocity, acceleration and jerk, sampled
//!   densely along it, keep their limits.

use crate::error::{Error, Result, ensure_input};
use serde::{Deserialize, Serialize};

use crate::robot::Robot;
use crate::spline::{self, BASIS_D3};
use crate::topp::{self, CHECK_SAMPLES};
use crate::types::{JointTrajectory, StartMotion};

/// Joint position, velocity and acceleration at one instant.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JointState {
    pub position: Vec<f32>,
    pub velocity: Vec<f32>,
    pub acceleration: Vec<f32>,
}

impl JointState {
    pub fn new(dof: usize) -> Self {
        Self {
            position: vec![0.0; dof],
            velocity: vec![0.0; dof],
            acceleration: vec![0.0; dof],
        }
    }
}

/// A rest-to-rest joint trajectory: a path whose first three and last three control points are
/// equal, and the time map that runs along it. A trajectory from a moving start
/// ([`Trajectory::moving`]) starts with the motion its first control points encode; a controlled
/// stop ([`Trajectory::stop`]) starts partway along its path and comes to rest before its end, its
/// time map's last three points equal. Plain data, so it can be sent to another process;
/// [`Trajectory::check`] verifies one before it runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    pub dof: usize,
    /// The path, `[points, dof]`: a uniform cubic B-spline over `s` from 0 to `points - 3` spans.
    pub control_points: Vec<f32>,
    /// Seconds per span of the time map.
    pub knot_interval: f32,
    /// The time map's control points: a uniform cubic B-spline in `s`, from 0 at the start to the
    /// path's span count at the end (or from and to where a stop starts and ends).
    pub time_points: Vec<f32>,
}

/// The values of `d`-th differences of control points: (width, weights).
const DIFFERENCES: [(usize, &[f32]); 3] =
    [(2, &[-1.0, 1.0]), (3, &[1.0, -2.0, 1.0]), (4, &BASIS_D3)];

impl Trajectory {
    /// Times a planned path (`[points, dof]` control points, at least 7) as fast as the robot's
    /// velocity, acceleration and jerk limits allow, then slows it by `speed_scale` in (0, 1].
    pub fn new(robot: &Robot, control_points: &[f32], speed_scale: f32) -> Result<Self> {
        let n = robot.dof();
        ensure_input!(
            speed_scale > 0.0 && speed_scale <= 1.0,
            "speed_scale must be in (0, 1], got {speed_scale}"
        );
        check_path(control_points, n)?;
        let h = uniform_knot(robot, control_points);
        let spans = control_points.len() / n - 3;
        let first = &control_points[..3 * n];
        ensure_input!(
            first.chunks(n).all(|p| p == &first[..n]),
            "the path starts moving: time it with Trajectory::moving and its start motion"
        );
        let (mut knot, mut points) = (h, linear(spans));
        if h > 0.0
            && let Some((tau, map)) = topp::time_map(robot, control_points, None)
            && tau * ((map.len() - 3) as f32) < h * spans as f32
        {
            (knot, points) = (tau, map);
        }
        Ok(Self {
            dof: n,
            control_points: control_points.to_vec(),
            knot_interval: knot / speed_scale,
            time_points: points,
        })
    }

    /// Times a path planned from a moving start ([`crate::PlanProblem::start_motion`]) so that it
    /// starts with exactly that velocity and acceleration, which its first three control points
    /// encode. Errors with [`Error::Unsafe`] if no timing of the path from that motion keeps the
    /// limits, as when it cannot brake in time.
    pub fn moving(robot: &Robot, control_points: &[f32], motion: &StartMotion) -> Result<Self> {
        let n = robot.dof();
        ensure_input!(
            motion.velocity.len() == n && motion.acceleration.len() == n,
            "a start motion needs {n} velocities and accelerations"
        );
        check_path(control_points, n)?;
        let Some(h0) = start_knot(control_points, n, motion)? else {
            return Self::new(robot, control_points, 1.0);
        };
        let spans = control_points.len() / n - 3;
        // Uniform timing at exactly h0, when that is no faster than the differences allow.
        let at = |knot_interval: f32, time_points: Vec<f32>| Self {
            dof: n,
            control_points: control_points.to_vec(),
            knot_interval,
            time_points,
        };
        let uniform = (uniform_knot(robot, control_points) <= h0).then(|| at(h0, linear(spans)));
        let optimal = topp::time_map(robot, control_points, Some(1.0 / h0 as f64))
            .map(|(knot, map)| at(knot, map));
        [uniform, optimal]
            .into_iter()
            .flatten()
            .min_by(|a, b| a.duration().total_cmp(&b.duration()))
            .ok_or_else(|| {
                Error::Unsafe(
                    "no timing of this path from its start motion keeps the limits".into(),
                )
            })
    }

    fn spans(&self) -> usize {
        self.control_points.len() / self.dof - 3
    }

    pub fn duration(&self) -> f32 {
        (self.time_points.len() - 3) as f32 * self.knot_interval
    }

    /// The state at time `t` (clamped to the trajectory), written into `out` without allocating.
    pub fn at(&self, t: f32, out: &mut JointState) {
        let n = self.dof;
        for v in [&mut out.position, &mut out.velocity, &mut out.acceleration] {
            v.resize(n, 0.0);
        }
        let (s, sd, sdd) = self.map_at(t);
        let (h, maps) = (self.knot_interval, self.time_points.len() - 3);
        let x = if h <= 0.0 || t <= 0.0 {
            0.0
        } else if t >= self.duration() {
            maps as f32
        } else {
            t / h
        };
        let spans = self.spans();
        let s = s.clamp(0.0, spans as f32);
        let span = (s as usize).min(spans - 1);
        let v = s - span as f32;
        let cp = &self.control_points;
        spline::blend(cp, n, span, spline::basis(v), &mut out.position);
        spline::blend(cp, n, span, spline::basis_d1(v), &mut out.velocity);
        spline::blend(cp, n, span, spline::basis_d2(v), &mut out.acceleration);
        for j in 0..n {
            let (p1, p2) = (out.velocity[j], out.acceleration[j]);
            out.velocity[j] = p1 * sd;
            out.acceleration[j] = p2 * sd * sd + p1 * sdd;
        }
        // At rest at an end of the path, the position is exactly that end's control point (the
        // blend of three equal points can round off by one unit in the last place). A stop's map
        // ends before the path does, and starts partway along it.
        let end = if x <= 0.0 && s <= 1e-3 {
            Some(0)
        } else if x >= maps as f32 && s >= spans as f32 - 1e-3 {
            Some(cp.len() / n - 3)
        } else {
            None
        };
        if let Some(e) = end
            && (e..e + 3).all(|i| cp[i * n..(i + 1) * n] == cp[e * n..(e + 1) * n])
        {
            out.position.copy_from_slice(&cp[e * n..(e + 1) * n]);
        }
    }

    /// Where the time map is at time `t` (clamped to the trajectory): the path parameter `s`, and
    /// its first and second derivatives per second.
    fn map_at(&self, t: f32) -> (f32, f32, f32) {
        let (h, maps) = (self.knot_interval, self.time_points.len() - 3);
        // t / h at t = duration() can round to just under the last span's end.
        let x = if h <= 0.0 || t <= 0.0 {
            0.0
        } else if t >= self.duration() {
            maps as f32
        } else {
            (t / h).min(maps as f32)
        };
        let m = (x as usize).min(maps - 1);
        let u = x - m as f32;
        let tp = &self.time_points;
        let at = |w: [f32; 4]| (0..4).map(|i| w[i] * tp[m + i]).sum::<f32>();
        if h > 0.0 {
            (
                at(spline::basis(u)),
                at(spline::basis_d1(u)) / h,
                at(spline::basis_d2(u)) / (h * h),
            )
        } else {
            (at(spline::basis(u)), 0.0, 0.0)
        }
    }

    /// This trajectory from time `t` on, braking to rest along its path: a controlled stop, on a
    /// path already checked, for when the motion must not go on (an obstacle appeared, a replan
    /// failed). It starts in exactly the state at `t` and stops as soon along the path as a time
    /// map braking at an even rate within the limits allows; verify it with
    /// [`Trajectory::check_from`]. Errors with [`Error::Unsafe`] if no such stop ends before
    /// the path does, where this trajectory comes to rest anyway.
    pub fn stop(&self, robot: &Robot, t: f32) -> Result<Self> {
        ensure_input!(
            self.dof == robot.dof(),
            "trajectory has {} joints, the robot has {}",
            self.dof,
            robot.dof()
        );
        let (s0, v0, a0) = self.map_at(t);
        // At rest already (as where a path ends at rest under a uniform map that does not slow):
        // nothing to brake, so the map stands still.
        let mut state = JointState::new(self.dof);
        self.at(t, &mut state);
        let (v0, a0) = if state
            .velocity
            .iter()
            .chain(&state.acceleration)
            .all(|x| x.abs() < 1e-6)
        {
            (0.0, 0.0)
        } else {
            (v0, a0)
        };
        // The path from the span the stop starts in: dropping k control points shifts s by exactly
        // k, and a map near zero keeps its start's derivatives precise in f32.
        let n = self.dof;
        let k = (s0.max(0.0) as usize).min(self.control_points.len() / n - 7);
        let cp = &self.control_points[k * n..];
        let (s0, spans) = (s0 - k as f32, (self.spans() - k) as f32);
        // The map of `m` spans braking hardest within the limits (see `stop_map`): knot intervals
        // double until one keeps the limits, then bisect between it and the last that did not.
        // The `m` that stops soonest along the path wins.
        let mut points = [0.0f32; STOP_MAX_SPANS + 3];
        let mut best: Option<(f32, f32, usize)> = None;
        let fits = |tau: f32, m: usize, map: &mut [f32]| {
            let end = stop_map(s0, v0, a0, m, tau, map);
            (
                end,
                end <= spans + 1e-4 * spans.max(1.0)
                    && runs_forward(map)
                    && topp::within_limits(robot, cp, tau, map),
            )
        };
        for m in STOP_SPANS {
            let map = &mut points[..m + 3];
            let (mut fail, mut tau) = (0.0f32, STOP_FIRST_KNOT);
            let found = loop {
                let (end, ok) = fits(tau, m, map);
                if ok {
                    break true;
                }
                if end > spans || tau > STOP_LAST_KNOT || best.is_some_and(|(e, ..)| end >= e) {
                    break false;
                }
                (fail, tau) = (tau, 2.0 * tau);
            };
            if !found {
                continue;
            }
            for _ in 0..STOP_BISECTIONS {
                let middle = if fail > 0.0 {
                    (fail * tau).sqrt()
                } else {
                    0.5 * tau
                };
                if fits(middle, m, map).1 {
                    tau = middle
                } else {
                    fail = middle
                }
            }
            let end = stop_map(s0, v0, a0, m, tau, map);
            if best.is_none_or(|(e, ..)| end < e) {
                best = Some((end, tau, m));
            }
        }
        let Some((_, tau, m)) = best else {
            return Err(Error::Unsafe(format!(
                "no stop from {t} s ends before the path does; the trajectory comes to rest there"
            )));
        };
        let mut map = vec![0.0f32; m + 3];
        stop_map(s0, v0, a0, m, tau, &mut map);
        Ok(Self {
            dof: n,
            control_points: cp.to_vec(),
            knot_interval: tau,
            time_points: map,
        })
    }

    /// The trajectory sampled `hz` times per second, from its start to its exact end.
    pub fn sample(&self, hz: f32) -> Result<JointTrajectory> {
        let duration = self.duration();
        ensure_input!(
            hz.is_finite() && hz > 0.0 && (duration * hz) < 1e8,
            "cannot sample a {duration} s trajectory at {hz} Hz"
        );
        let dt = 1.0 / hz;
        let samples = (duration * hz).ceil() as usize + 1;
        let mut state = JointState::new(self.dof);
        let mut out = JointTrajectory {
            dof: self.dof,
            dt,
            duration,
            positions: Vec::with_capacity(samples * self.dof),
            velocities: Vec::with_capacity(samples * self.dof),
            accelerations: Vec::with_capacity(samples * self.dof),
        };
        for k in 0..samples {
            // The last sample lands exactly on the end, whatever the rounding of k * dt.
            self.at(
                if k + 1 == samples {
                    duration
                } else {
                    (k as f32 * dt).min(duration)
                },
                &mut state,
            );
            out.positions.extend_from_slice(&state.position);
            out.velocities.extend_from_slice(&state.velocity);
            out.accelerations.extend_from_slice(&state.acceleration);
        }
        Ok(out)
    }

    /// Verifies the trajectory is safe to run on `robot`: finite, at rest at both ends, within
    /// every position limit, and within every velocity, acceleration and jerk limit: exactly
    /// for uniform timing, at 64 points per time-map span otherwise.
    pub fn check(&self, robot: &Robot) -> Result<()> {
        self.verify(robot, None)
    }

    /// [`Trajectory::check`] for a trajectory that starts in motion ([`Trajectory::moving`]): it
    /// must start exactly at `start`, position, velocity and acceleration, instead of at rest.
    pub fn check_from(&self, robot: &Robot, start: &JointState) -> Result<()> {
        self.verify(robot, Some(start))
    }

    fn verify(&self, robot: &Robot, start: Option<&JointState>) -> Result<()> {
        let n = robot.dof();
        let cp = &self.control_points;
        ensure_input!(
            self.dof == n,
            "trajectory has {} joints, the robot has {n}",
            self.dof
        );
        ensure_input!(
            cp.len().is_multiple_of(n) && cp.len() / n >= 7,
            "need a whole number of at least 7 control points"
        );
        macro_rules! ensure_safe {
            ($cond:expr, $($message:tt)+) => {
                if !$cond {
                    return Err(Error::Unsafe(format!($($message)+)));
                }
            };
        }
        ensure_safe!(
            cp.iter().all(|v| v.is_finite()),
            "control points must be finite"
        );
        let h = self.knot_interval;
        ensure_safe!(
            h.is_finite() && h >= 0.0,
            "knot interval must be finite and non-negative, got {h}"
        );
        let points = cp.len() / n;
        if h == 0.0 {
            ensure_safe!(
                cp.chunks(n).all(|p| p == &cp[..n]),
                "a trajectory that takes no time must not move"
            );
        }
        let tp = &self.time_points;
        ensure_safe!(
            tp.len() >= 4 && tp.iter().all(|v| v.is_finite()),
            "the time map needs at least 4 finite control points"
        );
        ensure_safe!(runs_forward(tp), "the time map must not run backward");
        let ends = [
            (tp[0] + 4.0 * tp[1] + tp[2]) / 6.0,
            (tp[tp.len() - 3] + 4.0 * tp[tp.len() - 2] + tp[tp.len() - 1]) / 6.0,
        ];
        let spans = (points - 3) as f32;
        let slack = 1e-4 * spans.max(1.0);
        ensure_safe!(
            ends[0] >= -slack && ends[0] <= ends[1] && ends[1] <= spans + slack,
            "the time map must run forward along the path, from {} to {} of its {spans} spans",
            ends[0],
            ends[1]
        );
        let rests = |end: usize| {
            (end..end + 3).all(|i| cp[i * n..(i + 1) * n] == cp[end * n..(end + 1) * n])
        };
        // At rest at the end: the path ends there, or the time map stops (a controlled stop).
        let stops = tp[tp.len() - 3..].iter().all(|&p| p == tp[tp.len() - 1]);
        let at_path_end = (ends[1] - spans).abs() < slack;
        ensure_safe!(
            stops || (at_path_end && rests(points - 3)),
            "trajectory must end at rest: three equal control points at the path's end, or a time map that stops"
        );
        match start {
            None => ensure_safe!(
                ends[0].abs() < slack && rests(0),
                "trajectory must start at rest (three equal control points at the start)"
            ),
            Some(expected) => {
                ensure_input!(
                    [
                        &expected.position,
                        &expected.velocity,
                        &expected.acceleration
                    ]
                    .iter()
                    .all(|v| v.len() == n),
                    "a start state needs {n} positions, velocities and accelerations"
                );
                let mut first = JointState::new(n);
                self.at(0.0, &mut first);
                let close = |a: &[f32], b: &[f32]| {
                    a.iter()
                        .zip(b)
                        .all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + y.abs()))
                };
                ensure_safe!(
                    close(&first.position, &expected.position)
                        && close(&first.velocity, &expected.velocity)
                        && close(&first.acceleration, &expected.acceleration),
                    "trajectory does not start at the given state: it starts at {first:?}"
                );
            }
        }
        let limits = [
            robot.max_velocity(),
            robot.max_acceleration(),
            robot.max_jerk(),
        ];
        let names = ["velocity", "acceleration", "jerk"];
        let uniform = *tp == linear(points - 3);
        if !uniform && h > 0.0 {
            let mut out = vec![[0.0f32; 3]; n];
            for m in 0..tp.len() - 3 {
                for sample in 0..=CHECK_SAMPLES {
                    topp::derivatives(
                        cp,
                        n,
                        h,
                        tp,
                        m,
                        sample as f32 / CHECK_SAMPLES as f32,
                        &mut out,
                    );
                    for (j, d) in out.iter().enumerate() {
                        for order in 0..3 {
                            let (peak, limit) = (d[order].abs(), limits[order][j]);
                            // False for NaN too.
                            let within = peak <= limit * (1.0 + 1e-3);
                            ensure_safe!(
                                within,
                                "joint {j} reaches {} {peak}, over its limit {limit}",
                                names[order]
                            );
                        }
                    }
                }
            }
        }
        for j in 0..n {
            let (lo, hi) = robot.bounds(j);
            // The curve's own range, not its control points': a moving start's first control point
            // lies behind the start, where the curve never goes.
            if let Some((span, (low, high))) = (0..points - 3)
                .map(|k| (k, span_range(cp, n, j, k)))
                .find(|&(_, (low, high))| !(low >= lo - RANGE_SLACK && high <= hi + RANGE_SLACK))
            {
                return Err(Error::Unsafe(format!(
                    "joint {j} leaves its range [{lo}, {hi}] in span {span}, reaching [{low}, {high}]"
                )));
            }
            let joint_limits = limits.map(|l| l[j]);
            let orders = DIFFERENCES
                .iter()
                .enumerate()
                .filter(|_| h > 0.0 && uniform);
            for (order, &(width, weights)) in orders {
                let peak = largest_difference(cp, n, j, width, weights) / h.powi(order as i32 + 1);
                let limit = joint_limits[order];
                if peak > limit * (1.0 + 1e-4) {
                    return Err(Error::Unsafe(format!(
                        "joint {j} reaches {} {peak}, over its limit {limit}",
                        names[order]
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Errors unless `cp` is a whole, finite path of at least 7 control points of `n` joints.
fn check_path(cp: &[f32], n: usize) -> Result<()> {
    ensure_input!(
        cp.len().is_multiple_of(n) && cp.len() / n >= 7,
        "a path needs a whole number of at least 7 control points of {n} joints"
    );
    ensure_input!(
        cp.iter().all(|v| v.is_finite()),
        "control points must be finite"
    );
    Ok(())
}

/// The smallest knot interval for which uniform timing keeps every joint's velocity, acceleration
/// and jerk limits along the whole of `cp`.
/// Whether the time map `tp` never runs backward: along each span, its rate is a quadratic in u
/// whose least value on [0, 1] is at an end or at its vertex. Its control points may still dip, as
/// a start that is accelerating places them.
fn runs_forward(tp: &[f32]) -> bool {
    tp.windows(4).all(|w| {
        let d = [w[1] - w[0], w[2] - w[1], w[3] - w[2]];
        // Twice the rate: a u² + b u + c.
        let (a, b, c) = (d[0] - 2.0 * d[1] + d[2], 2.0 * (d[1] - d[0]), d[0] + d[1]);
        let tolerance = -1e-6 * d.iter().fold(1.0f32, |m, x| m.max(x.abs()));
        let vertex = (a > 0.0)
            .then(|| -b / (2.0 * a))
            .filter(|u| *u > 0.0 && *u < 1.0);
        c >= tolerance
            && a + b + c >= tolerance
            && vertex.is_none_or(|u| a * u * u + b * u + c >= tolerance)
    })
}

/// A stop's time map of `m` spans at `tau` seconds each into `map` (`m + 3` points): the first
/// three continue (s, ṡ, s̈); the steps between points, its rate, then fall evenly to zero, and
/// the last three are equal, so it ends at rest. Returns where it stops.
fn stop_map(s0: f32, v0: f32, a0: f32, m: usize, tau: f32, map: &mut [f32]) -> f32 {
    map[0] = s0 - v0 * tau + a0 * tau * tau / 3.0;
    map[1] = s0 - a0 * tau * tau / 6.0;
    map[2] = s0 + v0 * tau + a0 * tau * tau / 3.0;
    let second = map[2] - map[1];
    for k in 2..m {
        map[k + 1] = map[k] + second * (m - k) as f32 / (m - 1) as f32;
    }
    let end = map[m];
    map[m + 1..].fill(end);
    end
}

/// Spans a stop's time map tries: more brake more evenly, closer to a constant deceleration.
const STOP_SPANS: [usize; 5] = [2, 3, 4, 6, 8];
const STOP_MAX_SPANS: usize = 8;
/// Knot intervals a stop tries, from the hardest braking up, in seconds.
const STOP_FIRST_KNOT: f32 = 0.01;
const STOP_LAST_KNOT: f32 = 5.0;
/// Bisections of the knot interval: within 2^-8 of a doubling, about 0.3%.
const STOP_BISECTIONS: usize = 8;

/// How far past a joint's range rounding may carry a curve whose control points lie on it.
const RANGE_SLACK: f32 = 1e-5;

/// The lowest and highest value joint `j` takes along span `k`: at its ends, or where its
/// derivative, a quadratic in u, is zero inside it. NaN anywhere makes both NaN.
fn span_range(cp: &[f32], n: usize, j: usize, k: usize) -> (f32, f32) {
    let p = [0, 1, 2, 3].map(|w| cp[(k + w) * n + j]);
    let at = |u: f32| {
        spline::basis(u)
            .iter()
            .zip(&p)
            .map(|(w, x)| w * x)
            .sum::<f32>()
    };
    // Six times the derivative: a u² + b u + c.
    let (a, b, c) = (
        -3.0 * p[0] + 9.0 * p[1] - 9.0 * p[2] + 3.0 * p[3],
        6.0 * (p[0] - 2.0 * p[1] + p[2]),
        3.0 * (p[2] - p[0]),
    );
    // The ends, and up to two roots inside: a fixed array, since `check` may run in a control loop.
    let (mut candidates, mut count) = ([0.0, 1.0, 0.0, 0.0], 2);
    let mut root = |u: f32| {
        if u > 0.0 && u < 1.0 {
            candidates[count] = u;
            count += 1;
        }
    };
    if a.abs() > 1e-12 {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            root((-b + disc.sqrt()) / (2.0 * a));
            root((-b - disc.sqrt()) / (2.0 * a));
        }
    } else if b.abs() > 1e-12 {
        root(-c / b);
    }
    let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
    for &u in &candidates[..count] {
        let x = at(u);
        if x.is_nan() {
            return (f32::NAN, f32::NAN);
        }
        (low, high) = (low.min(x), high.max(x));
    }
    (low, high)
}

pub(crate) fn uniform_knot(robot: &Robot, cp: &[f32]) -> f32 {
    let n = robot.dof();
    let limits = [
        robot.max_velocity(),
        robot.max_acceleration(),
        robot.max_jerk(),
    ];
    let mut h = 0.0f32;
    for (order, &(width, weights)) in DIFFERENCES.iter().enumerate() {
        for (j, &limit) in limits[order].iter().enumerate() {
            let largest = largest_difference(cp, n, j, width, weights);
            h = h.max((largest / limit).powf(1.0 / (order + 1) as f32));
        }
    }
    h
}

/// The knot interval `h0` a path's first three control points continue `motion` at: they are
/// `q0 - v0 h0 + a0 h0²/3`, `q0 - a0 h0²/6`, `q0 + v0 h0 + a0 h0²/3`. `None` for a start at rest.
fn start_knot(cp: &[f32], n: usize, motion: &StartMotion) -> Result<Option<f32>> {
    let (v, a) = (&motion.velocity, &motion.acceleration);
    ensure_input!(
        v.iter().chain(a).all(|x| x.is_finite()),
        "the start motion must be finite"
    );
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
    let (p0, p1, p2) = (&cp[..n], &cp[n..2 * n], &cp[2 * n..3 * n]);
    let first: Vec<f32> = (0..n).map(|j| (p2[j] - p0[j]) / 2.0).collect();
    let second: Vec<f32> = (0..n).map(|j| p0[j] - 2.0 * p1[j] + p2[j]).collect();
    let h0 = if dot(v, v) > 1e-12 {
        dot(&first, v) / dot(v, v)
    } else if dot(a, a) > 1e-12 {
        (dot(&second, a) / dot(a, a)).max(0.0).sqrt()
    } else {
        return Ok(None);
    };
    let matches = (0..n).all(|j| {
        (first[j] - v[j] * h0).abs() <= 1e-4 * (1.0 + first[j].abs())
            && (second[j] - a[j] * h0 * h0).abs() <= 1e-4 * (1.0 + second[j].abs())
    });
    ensure_input!(
        h0 > 0.0 && matches,
        "the path's first control points do not continue this start motion"
    );
    Ok(Some(h0))
}

/// The time map of uniform timing over `spans` path spans: `σ(t) = t / h`.
fn linear(spans: usize) -> Vec<f32> {
    (0..spans + 3).map(|k| k as f32 - 1.0).collect()
}

/// Largest `|Σ weights[i] * cp[k + i]|` over every window of joint `j`.
fn largest_difference(cp: &[f32], n: usize, j: usize, width: usize, weights: &[f32]) -> f32 {
    let points = cp.len() / n;
    (0..points + 1 - width)
        .map(|k| {
            weights
                .iter()
                .enumerate()
                .map(|(i, w)| w * cp[(k + i) * n + j])
                .sum::<f32>()
                .abs()
        })
        .fold(0.0, f32::max)
}
