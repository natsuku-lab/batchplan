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
use crate::spline::{self, BASIS_D3};

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
    // The path's first and second derivatives at each stage, `[stages + 1, n]`.
    let derivatives = |weights: fn(f32) -> [f32; 4]| -> Vec<f64> {
        let mut table = Vec::with_capacity((stages + 1) * n);
        let mut out = vec![0.0f32; n];
        for i in 0..=stages {
            let s = i as f64 * step;
            let span = (s as usize).min(spans - 1);
            spline::blend(cp, n, span, weights((s - span as f64) as f32), &mut out);
            table.extend(out.iter().map(|&v| f64::from(v)));
        }
        table
    };
    let first = derivatives(spline::basis_d1);
    let second = derivatives(spline::basis_d2);
    // The path's third derivative is constant along each span; the stage takes the larger of the
    // spans it touches.
    let mut third = Vec::with_capacity((stages + 1) * n);
    for i in 0..=stages {
        let s = i as f64 * step;
        let spans_here = [
            (s as usize).min(spans - 1),
            (s.ceil() as usize).saturating_sub(1).min(spans - 1),
        ];
        for j in 0..n {
            let largest = spans_here
                .iter()
                .map(|&k| {
                    (0..4)
                        .map(|w| BASIS_D3[w] as f64 * cp[(k + w) * n + j] as f64)
                        .sum::<f64>()
                        .abs()
                })
                .fold(0.0, f64::max);
            third.push(largest);
        }
    }
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

    // Each stage's bounds on u for state x, as lines u = slope x + offset, and the largest x the
    // velocity and jerk limits and the joints that do not move allow, before scaling. A given start
    // is held to the full limits rather than the headroom: it is not ours to choose.
    let stage_bounds: Vec<Bounds> = (0..=stages)
        .map(|i| {
            let mut b = Bounds {
                lower: Vec::with_capacity(n),
                upper: Vec::with_capacity(n),
                cap: f64::INFINITY,
                own: None,
            };
            let full = if i == 0 && start_speed.is_some() {
                1.0 / HEADROOM
            } else {
                1.0
            };
            for j in 0..n {
                let (vmax, amax, jmax) = (vmax[j] * full, amax[j] * full, jmax[j] * full);
                let at = i * n + j;
                // Jerk's speed term, the path's third derivative times ṡ³, caps ṡ like velocity
                // does.
                if third[at] > 1e-9 {
                    b.cap = b.cap.min((jmax / third[at]).powf(2.0 / 3.0));
                }
                let (a, d, c) = (first[at], second[at], amax);
                if a.abs() > 1e-9 {
                    b.cap = b.cap.min((vmax / a).powi(2));
                    // -c <= a u + d x <= c
                    let (lo, hi) = (((-d / a), (-c / a)), ((-d / a), (c / a)));
                    if a > 0.0 {
                        b.lower.push(lo);
                        b.upper.push(hi);
                    } else {
                        b.lower.push(hi);
                        b.upper.push(lo);
                    }
                } else if d.abs() > 1e-9 {
                    b.cap = b.cap.min(c / d.abs());
                }
            }
            let mut own = (f64::NEG_INFINITY, f64::INFINITY);
            let ordered = b
                .lower
                .iter()
                .all(|&l| b.upper.iter().all(|&u| narrow(l, u, &mut own)));
            b.own = ordered.then_some(own);
            b
        })
        .collect();
    // The next stage's interval as a lower and an upper line: next.0 <= x + 2 step u <= next.1.
    let next_lines = |next: (f64, f64)| {
        (
            (-1.0 / (2.0 * step), next.0 / (2.0 * step)),
            (-1.0 / (2.0 * step), next.1 / (2.0 * step)),
        )
    };
    // The x in [0, cap] for which some u lies between every lower and upper bound: the stage's own,
    // and the next stage's interval's when there is one.
    let feasible = |b: &Bounds, next: Option<(f64, f64)>, cap: f64| -> Option<(f64, f64)> {
        let (own_lo, own_hi) = b.own?;
        let mut x = (own_lo.max(0.0), own_hi.min(cap));
        if let Some(next) = next {
            let (lower, upper) = next_lines(next);
            let ordered = b.lower.iter().all(|&l| narrow(l, upper, &mut x))
                && b.upper.iter().all(|&u| narrow(lower, u, &mut x))
                && narrow(lower, upper, &mut x);
            if !ordered {
                return None;
            }
        }
        let (lo, hi) = x;
        (lo <= hi + 1e-12).then_some((lo, hi.max(lo)))
    };

    // The time-optimal profile under the caps scaled by `scale`: each stage's time and x. A planned
    // path's three equal control points at each end make it rest there whatever ṡ is, so the
    // profile may start and end at any speed its caps allow.
    let profile = |scale: &[f64]| -> Option<(Vec<f64>, Vec<f64>)> {
        // Backward: controllable intervals, from anything within the caps at the end.
        let mut controllable = vec![(0.0f64, 0.0f64); stages + 1];
        let last = &stage_bounds[stages];
        controllable[stages] = feasible(last, None, last.cap * scale[stages])?;
        for i in (0..stages).rev() {
            let b = &stage_bounds[i];
            controllable[i] = feasible(b, Some(controllable[i + 1]), b.cap * scale[i])?;
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
            let b = &stage_bounds[i];
            let (next_lower, next_upper) = next_lines(controllable[i + 1]);
            let line = |&(slope, offset): &(f64, f64)| slope * x[i] + offset;
            let u_lo = b.lower.iter().map(line).fold(line(&next_lower), f64::max);
            let u_hi = b.upper.iter().map(line).fold(line(&next_upper), f64::min);
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
            // Candidates share the profile, so one needing more stretch than the round's best takes
            // longer, and its assessment can stop there (with room for the duration's rounding).
            let beaten = round
                .as_ref()
                .map_or(f32::INFINITY, |(r, ..)| r * (1.0 + 1e-4));
            let (stretch, overshoots) =
                assess(robot, cp, tau as f32, &points, samples, target, beaten);
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
            let (stretch, _) = assess(robot, cp, knot, &points, CHECK_SAMPLES, 1.0, f32::INFINITY);
            Some((knot * stretch * STRETCH_MARGIN, points))
        }
        // Only candidates that needed no stretch at full density are kept.
        Some(_) => Some((knot, points)),
    }
}

/// A TOPP stage's acceleration bounds on u as lines u = slope x + offset, its cap on x, and the x
/// for which its own lines are ordered (`None` if none).
struct Bounds {
    lower: Vec<(f64, f64)>,
    upper: Vec<(f64, f64)>,
    cap: f64,
    own: Option<(f64, f64)>,
}

/// Narrows the interval `x` to where `lower` (slope, offset) lies at or below `upper`; false if
/// they are parallel and `lower` is above.
fn narrow(lower: (f64, f64), upper: (f64, f64), x: &mut (f64, f64)) -> bool {
    let (k, r) = (lower.0 - upper.0, upper.1 - lower.1);
    if k > 1e-12 {
        x.1 = x.1.min(r / k);
    } else if k < -1e-12 {
        x.0 = x.0.max(r / k);
    } else if r < -1e-12 {
        return false;
    }
    true
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
        at(spline::basis(u)),
        at(spline::basis_d1(u)) / knot,
        at(spline::basis_d2(u)) / (knot * knot),
        at(BASIS_D3) / (knot * knot * knot),
    );
    let span = (s.max(0.0) as usize).min(spans - 1);
    let (w1, w2) = (
        spline::basis_d1(s - span as f32),
        spline::basis_d2(s - span as f32),
    );
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
        f32::INFINITY,
    )
    .0 <= 1.0
}

/// How much longer the time map must take for every joint's sampled velocity, acceleration and jerk
/// to keep its limits (velocity scales with 1/k, acceleration 1/k², jerk 1/k³; at least 1), and
/// where it overshoots: each such sample's path position and the factor on x that limit needs
/// there. Once the stretch passes `beaten`, it returns what it has found.
fn assess(
    robot: &Robot,
    cp: &[f32],
    knot: f32,
    points: &[f32],
    samples: usize,
    target: f32,
    beaten: f32,
) -> (f32, Vec<(f32, f32)>) {
    let n = robot.dof();
    let limits = [
        robot.max_velocity(),
        robot.max_acceleration(),
        robot.max_jerk(),
    ];
    let targets: Vec<[f32; 3]> = (0..n)
        .map(|j| std::array::from_fn(|order| limits[order][j] * target))
        .collect();
    let mut out = vec![[0.0f32; 3]; n];
    let (mut k, mut overshoots) = (1.0f32, vec![]);
    // A ratio's root raises k only if the ratio exceeds k to the root's power. Below these powers,
    // shaded down past rounding, the root is not taken.
    let floors = |k: f32| [k, k * k, k * k * k].map(|power| power * (1.0 - 1e-4));
    let mut floor = floors(k);
    for m in 0..points.len() - 3 {
        for sample in 0..=samples {
            let u = sample as f32 / samples as f32;
            derivatives(cp, n, knot, points, m, u, &mut out);
            let mut factor = 1.0f32;
            for (d, limit) in out.iter().zip(&targets) {
                for order in 0..3 {
                    let ratio = d[order].abs() / limit[order];
                    if ratio > floor[order] {
                        let root = ratio.powf(1.0 / (order + 1) as f32);
                        if root > k {
                            k = root;
                            floor = floors(k);
                        }
                    }
                    if ratio > 1.0 {
                        factor = factor.min(ratio.powf(-2.0 / (order + 1) as f32));
                    }
                }
            }
            if factor < 1.0 {
                let s = (0..4)
                    .map(|i| spline::basis(u)[i] * points[m + i])
                    .sum::<f32>();
                overshoots.push((s, factor * 0.98));
            }
        }
        if k > beaten {
            break;
        }
    }
    (k, overshoots)
}
