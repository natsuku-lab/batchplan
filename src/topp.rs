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
//! smooths it while keeping it monotone. Where the smoothed map overshoots a limit, the profile is
//! solved again going slower there; the map is then stretched in time just enough that velocity,
//! acceleration and jerk, sampled densely, keep their limits.

use glam::Vec3;

use crate::robot::{Fk, Robot};
use crate::spline::{self, BASIS_D3};

/// TOPP grid stages per path span.
const STAGES_PER_SPAN: usize = 12;
/// Samples of a hand's rates per path span: a multiple of the stages, so every stage has one.
const HAND_SAMPLES_PER_SPAN: usize = 4 * STAGES_PER_SPAN;
/// Samples per time-map span when stretching and checking a time map.
pub(crate) const CHECK_SAMPLES: usize = 64;
/// A smoothing candidate: its stretch, where it overshoots (path position, factor on x) and its
/// number of map spans.
type Candidate = (f32, Vec<(f32, f32)>, usize);

/// Extra stretch beyond what the samples need, for the curve between them (sampling 4x denser
/// finds acceleration 0.13% over without it).
const STRETCH_MARGIN: f32 = 1.003;
/// Rounds of slowing the profile where the smoothed map overshoots.
const REFINEMENTS: usize = 3;
/// Rounds of stretching a given start's map at most, each by what the last one still needed.
const RESTRETCHES: usize = 4;
/// Path spans over which a given start's limits ease from the full ones to those that leave the
/// stretch's margin, and its speeds come down as far as elsewhere: its motion is not ours to
/// choose, may use the full limits, and cannot change at once.
const START_RAMP: f32 = 2.0;
/// Time-map span lengths tried, in multiples of the longest time a joint's jerk limit takes to
/// change its acceleration by its acceleration limit.
const SWING: [f64; 5] = [0.5, 0.7, 1.0, 1.4, 2.0];
/// The stretch every map sized by the path must need before maps sized by the limits are tried,
/// and before they also choose where the profile slows. Below the second they only compete to be
/// the result; past it, the rounds slow different stages. Either way the result is sooner on
/// average, but not for every path: candidates are compared at a quarter of the checking density.
const SWUNG_AFTER: f32 = 1.1;
const STEERED_AFTER: f32 = 1.3;

/// How fast a hand moves along a path per unit of `s`, sampled densely, and the speeds it must
/// keep under: meters and radians, per unit of `s` and per second. A hand's speed is its rate
/// times ṡ, so its caps are velocity limits on `s` alone.
pub(crate) struct HandRates {
    rates: Vec<[f32; 2]>,
    limits: [f32; 2],
}

impl HandRates {
    pub(crate) fn new(robot: &Robot, cp: &[f32], hand: usize, limits: [f32; 2]) -> Self {
        let n = robot.dof();
        let spans = cp.len() / n - 3;
        let mut fk = Fk::new(robot.links.len());
        let (mut q, mut dq) = (vec![0.0f32; n], vec![0.0f32; n]);
        let mut rates = Vec::with_capacity(spans * HAND_SAMPLES_PER_SPAN + 1);
        for k in 0..=spans * HAND_SAMPLES_PER_SPAN {
            let s = k as f32 / HAND_SAMPLES_PER_SPAN as f32;
            let span = (s as usize).min(spans - 1);
            let u = s - span as f32;
            spline::blend(cp, n, span, spline::basis(u), &mut q);
            spline::blend(cp, n, span, spline::basis_d1(u), &mut dq);
            robot.fk_into(&q, &mut fk);
            rates.push(robot.hand_velocity(&fk, hand, &dq).map(Vec3::length));
        }
        Self { rates, limits }
    }

    /// The larger of the rates sampled either side of `s`, for position and for rotation.
    fn at(&self, s: f32) -> [f32; 2] {
        let k = s.max(0.0) * HAND_SAMPLES_PER_SPAN as f32;
        let last = self.rates.len() - 1;
        let (below, above) = (
            (k.floor() as usize).min(last),
            (k.ceil() as usize).min(last),
        );
        std::array::from_fn(|i| self.rates[below][i].max(self.rates[above][i]))
    }

    /// The smallest knot interval at which uniform timing keeps the hand under its caps.
    pub(crate) fn uniform_knot(&self) -> f32 {
        let mut h = 0.0f32;
        for rates in &self.rates {
            for (rate, limit) in rates.iter().zip(self.limits) {
                h = h.max(rate / limit);
            }
        }
        h * STRETCH_MARGIN
    }
}

/// A time map for `cp`: seconds per span and control points, or `None` when the path does not
/// move or cannot be timed. With `start_speed`, the map starts at that ṡ (spans per second) with
/// no s̈, to continue a motion already under way, and is not stretched as a whole (that would
/// change the start). Where no map fits so, it is timed again with the start held to the full
/// limits, easing over `START_RAMP` path spans to those that leave the stretch's margin, the map
/// stretched with its start set again (`restretched`), and the refinement ending, rather than
/// failing, where lowered speeds leave the start's out of reach. Both tries turn candidates down
/// at a quarter of the checking density first; where neither finds a map, both run again with
/// every candidate measured at the full density, which steers the refinement to maps the coarse
/// measure misses. With `hand`, the hand's speed keeps under its caps as the joints keep their
/// limits.
pub(crate) fn time_map(
    robot: &Robot,
    cp: &[f32],
    start_speed: Option<f64>,
    hand: Option<&HandRates>,
) -> Option<(f32, Vec<f32>)> {
    let Some(_) = start_speed else {
        return timed_map(
            robot, cp, None, hand, /* eased */ false, /* full */ false,
        );
    };
    [(false, false), (true, false), (false, true), (true, true)]
        .into_iter()
        .find_map(|(eased, full)| timed_map(robot, cp, start_speed, hand, eased, full))
}

/// [`time_map`], with `eased` its second try at a given start, and `full` measuring each of a
/// given start's candidates at the full checking density rather than at a quarter of it first.
fn timed_map(
    robot: &Robot,
    cp: &[f32],
    start_speed: Option<f64>,
    hand: Option<&HandRates>,
    eased: bool,
    full: bool,
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
    let vmax: Vec<f64> = robot.max_velocity().iter().map(|&v| f64::from(v)).collect();
    let amax: Vec<f64> = robot
        .max_acceleration()
        .iter()
        .map(|&a| f64::from(a))
        .collect();
    let jmax: Vec<f64> = robot.max_jerk().iter().map(|&j| f64::from(j)).collect();

    // Each stage's bounds on u for state x, as lines u = slope x + offset, and the largest x the
    // velocity and jerk limits and the joints that do not move allow.
    let stage_bounds: Vec<Bounds> = (0..=stages)
        .map(|i| {
            let mut b = Bounds {
                lower: Vec::with_capacity(n),
                upper: Vec::with_capacity(n),
                cap: f64::INFINITY,
                own: None,
            };
            for j in 0..n {
                let (vmax, amax, jmax) = (vmax[j], amax[j], jmax[j]);
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
            if let Some(hand) = hand {
                let rates = hand.at((i as f64 * step) as f32);
                for (rate, limit) in rates.into_iter().zip(hand.limits) {
                    if rate > 1e-9 {
                        b.cap = b.cap.min((f64::from(limit) / f64::from(rate)).powi(2));
                    }
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

    // The time-optimal profile with x at most `lowered` at each stage: each stage's time and x. A
    // planned path's three equal control points at each end make it rest there whatever ṡ is, so
    // the profile may start and end at any speed its caps allow.
    let profile = |lowered: &[f64]| -> Option<(Vec<f64>, Vec<f64>)> {
        // Backward: controllable intervals, from anything within the caps at the end.
        let mut controllable = vec![(0.0f64, 0.0f64); stages + 1];
        let last = &stage_bounds[stages];
        controllable[stages] = feasible(last, None, last.cap.min(lowered[stages]))?;
        for i in (0..stages).rev() {
            let b = &stage_bounds[i];
            controllable[i] = feasible(b, Some(controllable[i + 1]), b.cap.min(lowered[i]))?;
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

    // Where the smoothed map overshoots a limit, the stages around it may go no faster than the
    // profile did there, less what that limit needs (x is speed squared: velocity needs 1/r²,
    // acceleration 1/r, jerk r^(-2/3)), and the profile is solved again; whatever overshoot remains
    // is taken out by stretching. Lowering the stages' caps instead would change nothing where the
    // profile runs under them, as it does braking into a path's end.
    let mut lowered = vec![f64::INFINITY; stages + 1];
    let mut best: Option<(f32, Vec<f32>)> = None;
    // A given start's candidate needing the least stretch, for `restretched`: that stretch (as
    // measured at the density that turned it down, a quarter of the full one unless `full`), its
    // knot and its points. `restretched` measures again at the full density.
    let mut nearest: Option<(f32, f32, Vec<f32>)> = None;
    let swing = (0..n)
        .map(|j| f64::from(robot.max_acceleration()[j] / robot.max_jerk()[j]))
        .fold(0.0, f64::max);
    let rounds = if start_speed.is_some() {
        2 * REFINEMENTS
    } else {
        REFINEMENTS
    };
    for _ in 0..rounds {
        // Lowered speeds can leave a given start's no longer controllable: eased, the rounds
        // then stop, keeping what the earlier ones found.
        let Some((t, x)) = profile(&lowered) else {
            if eased {
                break;
            }
            return None;
        };
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
        // Map spans to try: half, one and two per path span; then, where all of those need
        // stretching by more than `SWUNG_AFTER`, spans lasting from half to twice the time the
        // jerk limit takes to change acceleration by its limit, coarser than any of them, steering
        // the rounds only past `STEERED_AFTER`. A map much finer than that follows the profile's
        // steps in s̈ too closely to keep jerk, so without them a path split into more spans would
        // time slower.
        let coarsest = ((spans as f64 * 0.5) as usize).max(4);
        let mut counts: Vec<usize> = [0.5, 1.0, 2.0]
            .map(|per_span| ((spans as f64 * per_span) as usize).max(4))
            .into();
        counts.dedup();
        let swung: Vec<usize> = SWING
            .map(|c| ((total / (c * swing)).ceil() as usize).max(4))
            .into_iter()
            .filter(|&m| m < coarsest)
            .collect();
        let (per_path, mut tried, mut stretched, mut steers) =
            (counts.len(), 0, f32::INFINITY, true);
        while tried < counts.len() {
            let m = counts[tried];
            tried += 1;
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
            // is stretched at the full density below. A given start cannot be stretched: its
            // candidates are held to limits that leave the margin, and one that keeps them at the
            // quarter density is measured again at the full one, whose samples include those, so
            // none is kept unchecked and most are turned down at a quarter of the cost.
            let target = match start_speed {
                None => 1.0,
                Some(_) => 1.0 / (STRETCH_MARGIN * STRETCH_MARGIN),
            };
            // Candidates share the profile, so one needing more stretch than the round's best takes
            // longer, and its assessment can stop there (with room for the duration's rounding).
            let beaten = round
                .as_ref()
                .map_or(f32::INFINITY, |(r, ..)| r * (1.0 + 1e-4));
            let measured = |samples: usize| {
                assess(
                    robot,
                    cp,
                    hand,
                    tau as f32,
                    &points,
                    samples,
                    target,
                    eased.then_some((0.0, 3)),
                    beaten,
                )
            };
            let coarse = if full {
                CHECK_SAMPLES
            } else {
                CHECK_SAMPLES / 4
            };
            let mut assessed = measured(coarse);
            if start_speed.is_some() && coarse < CHECK_SAMPLES && assessed.0 <= 1.0 {
                assessed = measured(CHECK_SAMPLES);
            }
            let (stretch, overshoots) = assessed;
            let knot = tau as f32 * stretch;
            let duration = |knot: f32, points: &[f32]| knot * (points.len() - 3) as f32;
            let usable = start_speed.is_none() || stretch <= 1.0;
            if usable
                && best
                    .as_ref()
                    .is_none_or(|(b, p)| duration(knot, &points) < duration(*b, p))
            {
                best = Some((knot, points));
            } else if !usable && nearest.as_ref().is_none_or(|(r, ..)| stretch < *r) {
                nearest = Some((stretch, tau as f32, points));
            }
            if steers && round.as_ref().is_none_or(|(r, ..)| stretch < *r) {
                round = Some((stretch, overshoots, m));
            }
            stretched = stretched.min(stretch);
            if tried == per_path && stretched > SWUNG_AFTER {
                counts.extend(&swung);
                steers = stretched > STEERED_AFTER;
            }
        }
        let (stretch, overshoots, m) = round?;
        if stretch <= 1.0 {
            break;
        }
        // Half a map span either side, in stages at the profile's average pace: wider, and the
        // slowing costs more time than the stretch it saves.
        let reach = (STAGES_PER_SPAN * spans).div_ceil(2 * m) as isize;
        let mut factor = vec![1.0f64; stages + 1];
        for (s, f) in overshoots {
            let center = (s as f64 / step).round() as isize;
            for i in (center - reach).max(0)..=(center + reach).min(stages as isize) {
                factor[i as usize] = factor[i as usize].min(f as f64);
            }
        }
        // A given start's speed stays, and the speeds after it come down only as it can slow to
        // them: by nothing at the start, easing to all of it over `START_RAMP` path spans.
        if start_speed.is_some() {
            for (i, f) in factor.iter_mut().enumerate() {
                let along = (i as f64 * step / f64::from(START_RAMP)).min(1.0);
                *f = 1.0 - (1.0 - *f) * along;
            }
        }
        for ((cap, f), x) in lowered.iter_mut().zip(factor).zip(&x) {
            if f < 1.0 {
                *cap = cap.min(x * f);
            }
        }
    }
    let (knot, points) = match (best, start_speed) {
        (Some(best), _) => best,
        (None, Some(speed)) if eased => return restretched(robot, cp, hand, speed, nearest?),
        (None, _) => return None,
    };
    match start_speed {
        None => {
            let (stretch, _) = assess(
                robot,
                cp,
                hand,
                knot,
                &points,
                CHECK_SAMPLES,
                1.0,
                /* eased from */ None,
                f32::INFINITY,
            );
            Some((knot * stretch * STRETCH_MARGIN, points))
        }
        // Only candidates that needed no stretch at full density are kept.
        Some(_) => Some((knot, points)),
    }
}

/// A given start's time map stretched as far as its limits need, which stretching it as it is
/// cannot do: that would slow its start too. Its first three points are set again for the start
/// speed at the longer knot interval, so it starts with that speed and no change of pace, and the
/// rest runs slower. `nearest` is the candidate needing the least stretch: that stretch, its knot
/// interval and its points. `None` if the start's points would overtake the map's next.
fn restretched(
    robot: &Robot,
    cp: &[f32],
    hand: Option<&HandRates>,
    speed: f64,
    nearest: (f32, f32, Vec<f32>),
) -> Option<(f32, Vec<f32>)> {
    let (mut stretch, tau, mut points) = nearest;
    let target = 1.0 / (STRETCH_MARGIN * STRETCH_MARGIN);
    for _ in 0..RESTRETCHES {
        let knot = tau * stretch * STRETCH_MARGIN;
        let lead = (f64::from(knot) * speed) as f32;
        if lead >= points[3] {
            return None;
        }
        points[..3].copy_from_slice(&[-lead, 0.0, lead]);
        let (needed, _) = assess(
            robot,
            cp,
            hand,
            knot,
            &points,
            CHECK_SAMPLES,
            target,
            /* eased from */ Some((0.0, 3)),
            f32::INFINITY,
        );
        if needed <= 1.0 {
            return Some((knot, points));
        }
        stretch *= needed;
    }
    None
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

/// Whether the time map `points` at `knot` seconds per span, which starts in a given motion at
/// path position `from`, keeps every joint's velocity, acceleration and jerk within its limits,
/// less the margin a map that cannot stretch leaves. The velocity it starts with is not the map's
/// to choose: its limit is the full one there, easing to the margin over `START_RAMP` path spans.
pub(crate) fn within_limits(
    robot: &Robot,
    cp: &[f32],
    knot: f32,
    points: &[f32],
    from: f32,
) -> bool {
    assess(
        robot,
        cp,
        None,
        knot,
        points,
        CHECK_SAMPLES,
        1.0 / (STRETCH_MARGIN * STRETCH_MARGIN),
        // The velocity alone: the map sets the stop's acceleration and jerk from its start on.
        Some((from, 1)),
        // Past the limits at all, it does not fit: no need to find by how much.
        1.0,
    )
    .0 <= 1.0
}

/// How much longer the time map must take for every joint's sampled velocity, acceleration and jerk
/// to keep its limits, and the hand's speed its caps (velocity scales with 1/k, acceleration 1/k²,
/// jerk 1/k³; at least 1), and where it overshoots: each such sample's path position and the factor
/// on x that limit needs there. The limits are `target` of the robot's; with `eased`, the path
/// position of a given start and how many of velocity, acceleration and jerk it eases, those ease
/// from the full limits there to `target` over `START_RAMP` path spans, and a hand's caps with
/// them.
/// Once the stretch passes `beaten`, it returns what it has found.
#[expect(
    clippy::too_many_arguments,
    reason = "private, with four callers that each set the sampling apart"
)]
fn assess(
    robot: &Robot,
    cp: &[f32],
    hand: Option<&HandRates>,
    knot: f32,
    points: &[f32],
    samples: usize,
    target: f32,
    eased: Option<(f32, usize)>,
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
            let map = |w: [f32; 4]| (0..4).map(|i| w[i] * points[m + i]).sum::<f32>();
            // How much more than `target` the limits are here: from the full ones at a given
            // start to none after the ramp.
            let (relief, eased_orders) = eased.map_or((1.0, 0), |(from, orders)| {
                let along = ((map(spline::basis(u)) - from) / START_RAMP).clamp(0.0, 1.0);
                ((target + (1.0 - target) * (1.0 - along)) / target, orders)
            });
            let mut factor = 1.0f32;
            if let Some(hand) = hand {
                let speed = map(spline::basis_d1(u)).abs() / knot;
                let rates = hand.at(map(spline::basis(u)));
                for (rate, limit) in rates.into_iter().zip(hand.limits) {
                    let ratio = rate * speed / (limit * target * relief);
                    if ratio > k {
                        k = ratio;
                        floor = floors(k);
                    }
                    if ratio > 1.0 {
                        factor = factor.min(ratio.powi(-2));
                    }
                }
            }
            for (d, limit) in out.iter().zip(&targets) {
                for order in 0..3 {
                    let relief = if order < eased_orders { relief } else { 1.0 };
                    let ratio = d[order].abs() / (limit[order] * relief);
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
                overshoots.push((map(spline::basis(u)), factor * 0.98));
            }
        }
        if k > beaten {
            break;
        }
    }
    (k, overshoots)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::device::Device;
    use crate::robot::{CollisionModel, Robot, RobotOptions};
    use crate::spline;
    use crate::timing::Trajectory;
    use crate::trajopt::{PlanOptions, PlanProblem};
    use crate::world::World;

    /// A path split into more spans is the same curve, so it takes about as long: the time map is
    /// sized by the limits as well as by the path's spans.
    #[test]
    fn a_split_path_takes_as_long() {
        let fingers = [
            ("panda_finger_joint1".to_string(), 0.04),
            ("panda_finger_joint2".to_string(), 0.04),
        ];
        let options = RobotOptions {
            lock_joints: HashMap::from(fingers),
            collision_model: Some(
                CollisionModel::load("assets/franka/panda_collision.json").unwrap(),
            ),
            ..Default::default()
        };
        let robot = Robot::load("assets/franka/franka_panda.urdf", &options).unwrap();
        let n = robot.dof();
        let device = Device::cpu(&robot).unwrap();
        let worlds = device.upload(&[World::default()]).unwrap();
        let goals = [
            [0.6, -0.8, 0.2, -2.2, 0.1, 1.7, 0.5],
            [-0.5, -0.5, -0.3, -1.8, 0.4, 1.9, 1.2],
            [0.2, -1.0, 0.5, -2.6, -0.3, 1.4, 0.3],
        ];
        let problems: Vec<PlanProblem> = goals
            .iter()
            .map(|goal| PlanProblem::new(0, robot.default_q().to_vec(), goal.to_vec()))
            .collect();
        let plans = crate::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap();
        assert_eq!(plans.solved().count(), goals.len(), "every motion plans");
        for s in plans.solved() {
            let whole = Trajectory::try_new(&robot, s.solution, 1.0)
                .unwrap()
                .duration();
            let mut path = s.solution.to_vec();
            for splits in 1..=2 {
                path = spline::split(&path, n);
                let split = Trajectory::try_new(&robot, &path, 1.0).unwrap().duration();
                assert!(
                    split < 1.03 * whole,
                    "motion {}, split {splits} times: {split} s, {whole} s whole",
                    s.index
                );
            }
        }
    }
}
