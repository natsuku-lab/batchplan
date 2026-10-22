//! Time-optimal timing of a planned path (TOPP-RA, Pham & Pham 2018) under per-joint velocity and
//! acceleration limits, then smoothed so jerk stays bounded too.
//!
//! The path is a uniform cubic B-spline `P(s)` with `s` in spans. On a grid of stages along `s`,
//! the state is `x = ṡ²` and the control `u = s̈`, constant between stages, so
//! `x[i + 1] = x[i] + 2 Δ u[i]`. Joint velocity `P′ ṡ` and acceleration `P′ s̈ + P″ ṡ²` are linear
//! in `(u, x)`: velocity limits cap `x`, acceleration limits are two half-planes per joint. A
//! backward pass finds each stage's controllable interval of `x` (what the later stages can still
//! brake from); a forward pass then takes the largest `u` that stays in them. Each stage's interval
//! is exact: for fixed `x` the feasible `u` lie between the largest of some lines in `x` and the
//! smallest of others, so the feasible `x` are where every such pair is ordered.
//!
//! The resulting `s(t)` has steps in `s̈`, so unbounded jerk. The time map that replaces it is a
//! uniform cubic B-spline `σ(t)` whose control points are `s` at their Greville points, which
//! smooths it while keeping it monotone; the map is then stretched in time just enough that
//! velocity, acceleration and jerk, sampled densely, keep their limits.

use crate::robot::Robot;
use crate::spline::{BASIS_D3, basis, basis_d1, basis_d2, blend};

/// TOPP grid stages per path span.
const STAGES_PER_SPAN: usize = 12;
/// Fraction of each velocity and acceleration limit the time-optimal profile may use, leaving room
/// for the smoothing.
const HEADROOM: f64 = 0.95;
/// Samples per time-map span when stretching and checking a time map.
pub(crate) const CHECK_SAMPLES: usize = 64;
/// A smoothing candidate: its stretch, where it overshoots (path position, factor on x) and its
/// number of map spans.
type Candidate = (f32, Vec<(f32, f32)>, usize);

/// Extra stretch beyond what the samples need, for the curve between them (sampling 4x denser
/// finds acceleration 0.13% over without it).
const STRETCH_MARGIN: f32 = 1.003;
/// Rounds of lowering the caps where the smoothed map overshoots.
const REFINEMENTS: usize = 3;

/// A time map for `cp`: seconds per span and control points, or `None` when the path does not
/// move or cannot be timed. With `start_speed`, the map starts at that ṡ (spans per second) with
/// no s̈, to continue a motion already under way; it is then never stretched (that would change the
/// start), so a path that cannot be timed from that speed within the limits has no map.
pub(crate) fn time_map(
    robot: &Robot,
    cp: &[f32],
    start_speed: Option<f64>,
) -> Option<(f32, Vec<f32>)> {
    let n = robot.dof();
    let spans = cp.len() / n - 3;
    let (stages, step) = (spans * STAGES_PER_SPAN, 1.0 / STAGES_PER_SPAN as f64);
    let derivative = |s: f64, weights: fn(f32) -> [f32; 4]| -> Vec<f64> {
        let span = (s as usize).min(spans - 1);
        let mut out = vec![0.0f32; n];
        blend(cp, n, span, weights((s - span as f64) as f32), &mut out);
        out.into_iter().map(f64::from).collect()
    };
    let first: Vec<Vec<f64>> = (0..=stages)
        .map(|i| derivative(i as f64 * step, basis_d1))
        .collect();
    let second: Vec<Vec<f64>> = (0..=stages)
        .map(|i| derivative(i as f64 * step, basis_d2))
        .collect();
    // The path's third derivative is constant along each span; the stage takes the larger of the
    // spans it touches.
    let third: Vec<Vec<f64>> = (0..=stages)
        .map(|i| {
            let s = i as f64 * step;
            let spans_here = [
                (s as usize).min(spans - 1),
                (s.ceil() as usize).saturating_sub(1).min(spans - 1),
            ];
            (0..n)
                .map(|j| {
                    spans_here
                        .iter()
                        .map(|&k| {
                            (0..4)
                                .map(|w| BASIS_D3[w] as f64 * cp[(k + w) * n + j] as f64)
                                .sum::<f64>()
                                .abs()
                        })
                        .fold(0.0, f64::max)
                })
                .collect()
        })
        .collect();
    let vmax: Vec<f64> = robot
        .max_velocity()
        .iter()
        .map(|&v| v as f64 * HEADROOM)
        .collect();
    let amax: Vec<f64> = robot
        .max_acceleration()
        .iter()
        .map(|&a| a as f64 * HEADROOM)
        .collect();
    let jmax: Vec<f64> = robot
        .max_jerk()
        .iter()
        .map(|&j| j as f64 * HEADROOM)
        .collect();

    // Bounds on u at stage i for state x, as lines u = slope x + offset, plus the largest x the
    // velocity limits and the joints that do not move allow, scaled by `scale`. A given start is
    // held to the full limits rather than the headroom: it is not ours to choose.
    let bounds = |i: usize, next: (f64, f64), scale: f64| {
        let (mut lower, mut upper, mut cap) = (vec![], vec![], f64::INFINITY);
        let full = if i == 0 && start_speed.is_some() {
            1.0 / HEADROOM
        } else {
            1.0
        };
        let (vmax, amax, jmax): (Vec<f64>, Vec<f64>, Vec<f64>) = (
            vmax.iter().map(|v| v * full).collect(),
            amax.iter().map(|a| a * full).collect(),
            jmax.iter().map(|j| j * full).collect(),
        );
        for j in 0..n {
            // Jerk's speed term, the path's third derivative times ṡ³, caps ṡ like velocity does.
            if third[i][j] > 1e-9 {
                cap = cap.min((jmax[j] / third[i][j]).powf(2.0 / 3.0));
            }
            let (a, b, c) = (first[i][j], second[i][j], amax[j]);
            if a.abs() > 1e-9 {
                cap = cap.min((vmax[j] / a).powi(2));
                // -c <= a u + b x <= c
                let (lo, hi) = (((-b / a), (-c / a)), ((-b / a), (c / a)));
                if a > 0.0 {
                    lower.push(lo);
                    upper.push(hi);
                } else {
                    lower.push(hi);
                    upper.push(lo);
                }
            } else if b.abs() > 1e-9 {
                cap = cap.min(c / b.abs());
            }
        }
        // The next stage's interval: next.0 <= x + 2 step u <= next.1.
        lower.push((-1.0 / (2.0 * step), next.0 / (2.0 * step)));
        upper.push((-1.0 / (2.0 * step), next.1 / (2.0 * step)));
        (lower, upper, cap * scale)
    };
    // The x in [0, cap] for which some u lies between every lower and upper bound.
    let feasible = |lower: &[(f64, f64)], upper: &[(f64, f64)], cap: f64| -> Option<(f64, f64)> {
        let (mut lo, mut hi) = (0.0f64, cap);
        for &(ls, lo_off) in lower {
            for &(us, up_off) in upper {
                // ls x + lo_off <= us x + up_off
                let (k, r) = (ls - us, up_off - lo_off);
                if k > 1e-12 {
                    hi = hi.min(r / k);
                } else if k < -1e-12 {
                    lo = lo.max(r / k);
                } else if r < -1e-12 {
                    return None;
                }
            }
        }
        (lo <= hi + 1e-12).then_some((lo, hi.max(lo)))
    };

    // The time-optimal profile under the caps scaled by `scale`: each stage's time and x. A planned
    // path's three equal control points at each end make it rest there whatever ṡ is, so the
    // profile may start and end at any speed its caps allow.
    let profile = |scale: &[f64]| -> Option<(Vec<f64>, Vec<f64>)> {
        // Backward: controllable intervals, from anything within the caps at the end.
        let mut controllable = vec![(0.0f64, 0.0f64); stages + 1];
        let (lower, upper, cap) = bounds(stages, (0.0, f64::MAX), scale[stages]);
        controllable[stages] = feasible(&lower[..lower.len() - 1], &upper[..upper.len() - 1], cap)?;
        for i in (0..stages).rev() {
            let (lower, upper, cap) = bounds(i, controllable[i + 1], scale[i]);
            controllable[i] = feasible(&lower, &upper, cap)?;
        }
        // Forward: from the given start or the fastest one, the largest u each stage allows, except
        // that a given start keeps s̈ as near 0 as it can.
        let mut x = vec![0.0f64; stages + 1];
        x[0] = match start_speed {
            Some(speed) => {
                let x0 = speed * speed;
                let (lo, hi) = controllable[0];
                (x0 >= lo * (1.0 - 1e-9) && x0 <= hi * (1.0 + 1e-9)).then_some(x0)?
            }
            None => controllable[0].1,
        };
        for i in 0..stages {
            let (lower, upper, _) = bounds(i, controllable[i + 1], scale[i]);
            let at = |lines: &[(f64, f64)], pick: fn(f64, f64) -> f64, from: f64| {
                lines
                    .iter()
                    .map(|&(slope, offset)| slope * x[i] + offset)
                    .fold(from, pick)
            };
            let (u_lo, u_hi) = (
                at(&lower, f64::max, f64::NEG_INFINITY),
                at(&upper, f64::min, f64::INFINITY),
            );
            let u = if i == 0 && start_speed.is_some() {
                0.0f64.clamp(u_lo.min(u_hi), u_hi)
            } else {
                u_hi
            };
            x[i + 1] = (x[i] + 2.0 * step * u)
                .clamp(controllable[i + 1].0, controllable[i + 1].1)
                .max(0.0);
        }
        // Time at each stage; s̈ is constant between stages.
        let mut t = vec![0.0f64; stages + 1];
        for i in 0..stages {
            let speeds = x[i].sqrt() + x[i + 1].sqrt();
            if speeds <= 0.0 {
                return None;
            }
            t[i + 1] = t[i] + 2.0 * step / speeds;
        }
        (t[stages].is_finite() && t[stages] > 0.0).then_some((t, x))
    };

    // Where the smoothed map overshoots a limit, the caps of the stages around it come down by what
    // that limit needs (x is speed squared: velocity needs 1/r², acceleration 1/r, jerk r^(-2/3)),
    // and the profile is solved again; whatever overshoot remains is taken out by stretching.
    let mut scale = vec![1.0f64; stages + 1];
    let mut best: Option<(f32, Vec<f32>)> = None;
    let rounds = if start_speed.is_some() {
        2 * REFINEMENTS
    } else {
        REFINEMENTS
    };
    for _ in 0..rounds {
        let (t, x) = profile(&scale)?;
        let total = t[stages];
        // s at time `time`, between stages at constant s̈.
        let s_at = |time: f64| {
            let time = time.clamp(0.0, total);
            let i = t
                .partition_point(|&ti| ti <= time)
                .saturating_sub(1)
                .min(stages - 1);
            let (dt, sd) = (time - t[i], x[i].sqrt());
            let u = (x[i + 1] - x[i]) / (2.0 * step);
            (i as f64 * step + sd * dt + 0.5 * u * dt * dt).min((i + 1) as f64 * step)
        };
        // The candidate needing the least stretch: that stretch, its overshoots and its span count.
        let mut round: Option<Candidate> = None;
        for per_span in [0.5, 1.0, 2.0] {
            let m = ((spans as f64 * per_span) as usize).max(4);
            let tau = total / m as f64;
            // Schoenberg's approximation of s(t): s at the control points' Greville points, the
            // first and last reflected so the map starts at 0 and ends at the path's end exactly.
            let mut points: Vec<f32> = (0..m + 3)
                .map(|k| s_at((k as f64 - 1.0) * tau) as f32)
                .collect();
            points[0] = -4.0 * points[1] - points[2];
            points[m + 2] = 6.0 * spans as f32 - 4.0 * points[m + 1] - points[m];
            // A given start: σ(0) = 0, σ̇(0) = that speed, σ̈(0) = 0.
            if let Some(speed) = start_speed {
                let lead = (tau * speed) as f32;
                points[..3].copy_from_slice(&[-lead, 0.0, lead]);
            }
            // Candidates are compared and located at a quarter of the checking density; the winner
            // is stretched at the full density below. A given start cannot be stretched, so its
            // candidates are measured at the full density against limits that leave the margin.
            let (samples, target) = match start_speed {
                None => (CHECK_SAMPLES / 4, 1.0),
                Some(_) => (CHECK_SAMPLES, 1.0 / (STRETCH_MARGIN * STRETCH_MARGIN)),
            };
            let (stretch, overshoots) = assess(robot, cp, tau as f32, &points, samples, target);
            let knot = tau as f32 * stretch;
            let duration = |knot: f32, points: &[f32]| knot * (points.len() - 3) as f32;
            let usable = start_speed.is_none() || stretch <= 1.0;
            if usable
                && best
                    .as_ref()
                    .is_none_or(|(b, p)| duration(knot, &points) < duration(*b, p))
            {
                best = Some((knot, points));
            }
            if round.as_ref().is_none_or(|(r, ..)| stretch < *r) {
                round = Some((stretch, overshoots, m));
            }
        }
        let (stretch, overshoots, m) = round?;
        if stretch <= 1.0 {
            break;
        }
        // Two map spans either side, in stages at the profile's average pace.
        let reach = (2 * STAGES_PER_SPAN * spans).div_ceil(m) as isize;
        let mut factor = vec![1.0f64; stages + 1];
        for (s, f) in overshoots {
            let center = (s as f64 / step).round() as isize;
            for i in (center - reach).max(0)..=(center + reach).min(stages as isize) {
                factor[i as usize] = factor[i as usize].min(f as f64);
            }
        }
        // A given start speed stays.
        if start_speed.is_some() {
            factor[0] = 1.0;
        }
        for (sc, f) in scale.iter_mut().zip(factor) {
            *sc *= f;
        }
    }
    let (knot, points) = best?;
    match start_speed {
        None => {
            let (stretch, _) = assess(robot, cp, knot, &points, CHECK_SAMPLES, 1.0);
            Some((knot * stretch * STRETCH_MARGIN, points))
        }
        // Only candidates that needed no stretch at full density are kept.
        Some(_) => Some((knot, points)),
    }
}

/// The path's joint velocity, acceleration and jerk where the time map is at `u` of span `m`.
pub(crate) fn derivatives(
    cp: &[f32],
    n: usize,
    knot: f32,
    points: &[f32],
    m: usize,
    u: f32,
    out: &mut [[f32; 3]],
) {
    let spans = cp.len() / n - 3;
    let at = |w: [f32; 4]| (0..4).map(|i| w[i] * points[m + i]).sum::<f32>();
    let (s, sd, sdd, sddd) = (
        at(basis(u)),
        at(basis_d1(u)) / knot,
        at(basis_d2(u)) / (knot * knot),
        at(BASIS_D3) / (knot * knot * knot),
    );
    let span = (s.max(0.0) as usize).min(spans - 1);
    let (w1, w2) = (basis_d1(s - span as f32), basis_d2(s - span as f32));
    for (j, o) in out.iter_mut().enumerate().take(n) {
        let path = |w: [f32; 4]| (0..4).map(|i| w[i] * cp[(span + i) * n + j]).sum::<f32>();
        let (p1, p2, p3) = (path(w1), path(w2), path(BASIS_D3));
        *o = [
            p1 * sd,
            p2 * sd * sd + p1 * sdd,
            p3 * sd * sd * sd + 3.0 * p2 * sd * sdd + p1 * sddd,
        ];
    }
}

/// Whether the time map `points` at `knot` seconds per span keeps every joint's velocity,
/// acceleration and jerk within its limits, less the margin a map that cannot stretch leaves.
pub(crate) fn within_limits(robot: &Robot, cp: &[f32], knot: f32, points: &[f32]) -> bool {
    assess(
        robot,
        cp,
        knot,
        points,
        CHECK_SAMPLES,
        1.0 / (STRETCH_MARGIN * STRETCH_MARGIN),
    )
    .0 <= 1.0
}

/// How much longer the time map must take for every joint's sampled velocity, acceleration and jerk
/// to keep its limits (velocity scales with 1/k, acceleration 1/k², jerk 1/k³; at least 1), and
/// where it overshoots: each such sample's path position and the factor on x that limit needs
/// there.
fn assess(
    robot: &Robot,
    cp: &[f32],
    knot: f32,
    points: &[f32],
    samples: usize,
    target: f32,
) -> (f32, Vec<(f32, f32)>) {
    let n = robot.dof();
    let limits = [
        robot.max_velocity(),
        robot.max_acceleration(),
        robot.max_jerk(),
    ];
    let mut out = vec![[0.0f32; 3]; n];
    let (mut k, mut overshoots) = (1.0f32, vec![]);
    for m in 0..points.len() - 3 {
        for sample in 0..=samples {
            let u = sample as f32 / samples as f32;
            derivatives(cp, n, knot, points, m, u, &mut out);
            let mut factor = 1.0f32;
            for (j, d) in out.iter().enumerate() {
                for order in 0..3 {
                    let ratio = d[order].abs() / (limits[order][j] * target);
                    k = k.max(ratio.powf(1.0 / (order + 1) as f32));
                    if ratio > 1.0 {
                        factor = factor.min(ratio.powf(-2.0 / (order + 1) as f32));
                    }
                }
            }
            if factor < 1.0 {
                let s = (0..4).map(|i| basis(u)[i] * points[m + i]).sum::<f32>();
                overshoots.push((s, factor * 0.98));
            }
        }
    }
    (k, overshoots)
}
