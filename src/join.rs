//! Joining motions without stopping, as MoveIt's Pilz planner blends a motion sequence. A path
//! ends at rest because its last three control points coincide, and the next starts with three on
//! the same configuration, so one spline through both stops there. A blend drops the control
//! points around the join while every hand there stays within the blend radius of its place at
//! the join, and bridges the gap along a quintic that sets out and arrives along both paths
//! without curving there: its curvature rises from zero, so jerk-limited timing can follow it at
//! speed, as clothoid blends do. The joined spline leaves the first path at the last knot it keeps
//! and rejoins the second at the first, and is the paths themselves elsewhere.
//!
//! Timing follows a path best where its control points are spaced alike along it, so paths are
//! first split at their spans' middles (the same curve on twice the spans) until each is spaced
//! within a factor of two of the finest, and until the narrowest blend at each of its joins turns
//! within half the radius. The bridge is spaced as the paths either side are.
//!
//! Each join weighs three blends: the widest within the radius, one dropping half as many control
//! points, and the narrowest. Those that make the stretch so far and the next motion finish sooner
//! than with a stop between them (turning back rarely does, nor a tight turn at speed) are checked
//! for collision, soonest first, and the first whose new spans are free is kept. A join none passes
//! keeps its stop, so no stretch takes longer than its motions one after another. Timing the
//! stretch so far at each join makes a long sequence cost more per join than a short one. A blend
//! takes at most the half of each path nearest its join, so blends never meet.

use crate::device::{Device, Worlds};
use crate::edges::{self, Deferred, EdgeQuery, Search};
use crate::error::{Result, ensure_input};
use crate::robot::Robot;
use crate::rrt;
use crate::spline::{self, basis};
use crate::timing::Trajectory;

/// Motions to run one after another in one world, each path starting where the one before ends.
#[derive(Clone, Debug)]
pub struct Sequence {
    /// Index of its world in the uploaded batch.
    pub world: u32,
    /// The motions' paths in order: rest-to-rest B-spline control points (`[points, dof]`, three
    /// equal at each end), as [`plan`](crate::plan) and [`follow`](crate::follow) return them.
    pub paths: Vec<Vec<f32>>,
}

/// How [`join`] passes from one motion to the next.
#[derive(Clone, Debug)]
pub struct JoinOptions {
    /// The blend radius (meters): near each join, the joined path leaves the paths only while
    /// every hand is within this distance of where it is at the join, as Pilz's and UR's blend
    /// radii: 5 cm.
    pub blend: f32,
    /// Joint-space spacing of collision checks, as [`RrtOptions::resolution`](crate::RrtOptions).
    pub resolution: f32,
}

impl Default for JoinOptions {
    fn default() -> Self {
        Self {
            blend: 0.05,
            resolution: 1.0 / 32.0,
        }
    }
}

/// What [`join`] did with each sequence.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct JoinResult {
    /// The sequences, in order.
    pub problems: Vec<Sequence>,
    /// Per sequence: its motions as joined, one rest-to-rest path (B-spline control points, for
    /// [`Trajectory::try_new`](crate::Trajectory::try_new)) for each stretch the robot moves
    /// without stopping, in order. Every join blended gives one path; every stop starts another.
    pub paths: Vec<Vec<Vec<f32>>>,
    /// Per sequence, per join: the radius its blend turns within (meters), the farthest any hand
    /// is from its place at the join where the path leaves or rejoins the motions: no more than
    /// [`JoinOptions::blend`], less where a narrower blend is sooner or a wider one collides.
    /// `None` where the motion stops.
    pub blends: Vec<Vec<Option<f32>>>,
}

/// Splits of a path at most, each doubling its spans. Each makes the third differences timing reads
/// eight times smaller; past three, single-precision rounding in them caps the speed.
const MAX_SPLITS: usize = 3;
/// Spline samples per span for its collision check, at least, as tool paths check theirs.
const CHECK_SAMPLES: usize = 4;
/// How far apart, per joint, one path's end and the next one's start may lie: rounding.
const MEET: f32 = 1e-5;

/// Joins each sequence's motions into one path that passes each join without stopping, within the
/// blend radius of it, where that is sooner and collision-free in the sequence's world
/// ([`JoinResult`]). Sequences run in parallel on the device's threads.
pub fn join(
    device: &Device,
    worlds: &Worlds,
    sequences: &[Sequence],
    o: &JoinOptions,
) -> Result<JoinResult> {
    let robot = device.robot();
    let n = robot.dof();
    ensure_input!(
        o.blend >= 0.0 && o.blend.is_finite(),
        "the blend radius must be finite and not negative, got {}",
        o.blend
    );
    ensure_input!(
        o.resolution > 0.0,
        "the resolution must be positive, got {}",
        o.resolution
    );
    for (i, sequence) in sequences.iter().enumerate() {
        ensure_input!(
            (sequence.world as usize) < worlds.len(),
            "sequence {i}: world {} of {}",
            sequence.world,
            worlds.len()
        );
        ensure_input!(!sequence.paths.is_empty(), "sequence {i} has no paths");
        for (k, path) in sequence.paths.iter().enumerate() {
            ensure_input!(
                path.len() % n == 0 && path.len() / n >= 7,
                "sequence {i}, path {k}: {} values are not at least 7 configurations of {n} joints",
                path.len()
            );
            ensure_input!(
                path.iter().all(|v| v.is_finite()),
                "sequence {i}, path {k} is not finite"
            );
            let rests = |ends: &[f32]| ends.chunks(n).all(|q| q == &ends[..n]);
            ensure_input!(
                rests(&path[..3 * n]) && rests(&path[path.len() - 3 * n..]),
                "sequence {i}, path {k} does not start and end at rest"
            );
            if k > 0 {
                let before = &sequence.paths[k - 1];
                let (end, start) = (&before[before.len() - n..], &path[..n]);
                ensure_input!(
                    end.iter().zip(start).all(|(a, b)| (a - b).abs() <= MEET),
                    "sequence {i}: path {k} starts at {start:?}, not where path {} ends, {end:?}",
                    k - 1
                );
            }
        }
    }
    let mut searches: Vec<_> = sequences
        .iter()
        .map(|sequence| Deferred::new(None, move || Joining::new(robot, sequence, o)))
        .collect();
    let mut running: Vec<&mut dyn Search> =
        searches.iter_mut().map(|s| s as &mut dyn Search).collect();
    device.search(worlds, &mut running, None)?;
    let mut result = JoinResult {
        problems: sequences.to_vec(),
        paths: Vec::with_capacity(sequences.len()),
        blends: Vec::with_capacity(sequences.len()),
    };
    for search in searches {
        let done = search.into_inner();
        result.paths.push(done.stretches());
        result.blends.push(done.blends);
    }
    Ok(result)
}

/// One join of the path under construction: its stop's three control points start at `at`, and a
/// blend may change the curve from knot `lower` to knot `upper`, the middles of the paths on
/// either side.
#[derive(Clone, Copy, Debug)]
struct Seam {
    at: usize,
    lower: usize,
    upper: usize,
}

/// A blend at a seam: the path keeps its control points up to `enter`, bridges the gap, and keeps
/// them again from `leave`, `back` and `ahead` control points out from the stop's. Within `radius`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Blend {
    back: usize,
    ahead: usize,
    radius: f32,
}

/// One sequence being joined, a search whose edge queries check each blend's new spans.
struct Joining<'a> {
    robot: &'a Robot,
    world: u32,
    n: usize,
    blend: f32,
    resolution: f32,
    /// The motions as given, how long each takes so, and the path so far: them, split, one after
    /// another, the joins decided so far blended.
    given: &'a [Vec<f32>],
    apart: Vec<f32>,
    path: Vec<f32>,
    seams: Vec<Seam>,
    /// Where the stretch the next seam ends begins in `path`, and how long that stretch takes as
    /// it would be returned.
    start: usize,
    so_far: f32,
    /// The seam being decided, its blends still to try (soonest first) with how long the stretch
    /// takes with each, and the one under check: the path with it and the number of its samples
    /// checked.
    next: usize,
    candidates: Vec<(Blend, f32)>,
    candidate: Vec<f32>,
    samples: usize,
    starts: Vec<usize>,
    blends: Vec<Option<f32>>,
}

impl<'a> Joining<'a> {
    fn new(robot: &'a Robot, sequence: &'a Sequence, o: &JoinOptions) -> Self {
        let n = robot.dof();
        let last = sequence.paths.len() - 1;
        let mut motions = sequence.paths.clone();
        if o.blend > 0.0 {
            let mut splits = vec![0; motions.len()];
            loop {
                // Motions of fewer than eight control points have no room for a blend, so never
                // share a stretch with another.
                let finest = motions
                    .iter()
                    .filter(|p| p.len() / n >= 8)
                    .map(|p| spacing(p, n))
                    .filter(|&d| d > 1e-6)
                    .fold(f32::INFINITY, f32::min);
                let mut split = false;
                for (k, p) in motions.iter_mut().enumerate() {
                    let points = p.len() / n;
                    let coarse = |knot: usize, at: usize| {
                        reach(robot, &knot_point(p, n, knot), &p[at * n..(at + 1) * n])
                            > 0.5 * o.blend
                    };
                    let wanted = spacing(p, n) > 2.0 * finest
                        || (k < last && coarse(points - 5, points - 1))
                        || (k > 0 && coarse(4, 0));
                    if wanted && splits[k] < MAX_SPLITS {
                        *p = spline::split(p, n);
                        splits[k] += 1;
                        split = true;
                    }
                }
                if !split {
                    break;
                }
            }
        }
        let mut path: Vec<f32> = vec![];
        let mut seams: Vec<Seam> = vec![];
        for (k, p) in motions.iter().enumerate() {
            let points = p.len() / n;
            if let Some(seam) = seams.last_mut() {
                // This path's first half bounds the join before it.
                seam.upper = seam.at + points / 2;
                path.extend_from_slice(&p[3 * n..]);
            } else {
                path.clone_from(p);
            }
            if k < last {
                let at = path.len() / n - 3;
                seams.push(Seam {
                    at,
                    lower: at + 3 - points / 2,
                    upper: 0,
                });
            }
        }
        let apart = if o.blend > 0.0 {
            let duration = |p: &Vec<f32>| {
                Trajectory::try_new(robot, p, 1.0).map_or(f32::INFINITY, |t| t.duration())
            };
            sequence.paths.iter().map(duration).collect()
        } else {
            vec![]
        };
        let mut joining = Self {
            robot,
            world: sequence.world,
            n,
            blend: o.blend,
            resolution: o.resolution,
            given: &sequence.paths,
            so_far: apart.first().copied().unwrap_or(0.0),
            apart,
            path,
            seams,
            start: 0,
            next: 0,
            candidates: vec![],
            candidate: vec![],
            samples: 0,
            starts: vec![],
            blends: vec![],
        };
        joining.candidates = joining.blends_at(0);
        joining
    }

    /// Seam `s`'s blends within the radius: the widest, one dropping half as many control points,
    /// and the narrowest, which drops only the stop's. None if the narrowest already turns wider
    /// than the radius or reaches past either path's middle.
    fn shapes_at(&self, s: usize) -> Vec<Blend> {
        let Some(seam) = self.seams.get(s) else {
            return vec![];
        };
        if self.blend == 0.0 {
            return vec![];
        }
        let (n, at) = (self.n, seam.at);
        let join = &self.path[at * n..(at + 1) * n];
        let reach_at = |knot: usize| reach(self.robot, &knot_point(&self.path, n, knot), join);
        // The narrowest blend leaves the path at knot `at - 2` and rejoins it at `at + 4`; wider
        // ones a knot further out at a time, while every hand there stays within the radius.
        if at < seam.lower + 2 || at + 4 > seam.upper {
            return vec![];
        }
        let (mut before, mut after) = (vec![reach_at(at - 2)], vec![reach_at(at + 4)]);
        if before[0].max(after[0]) > self.blend {
            return vec![];
        }
        while at >= seam.lower + 2 + before.len() {
            let r = reach_at(at - 2 - before.len());
            if r > self.blend {
                break;
            }
            before.push(r);
        }
        while at + 4 + after.len() <= seam.upper {
            let r = reach_at(at + 4 + after.len());
            if r > self.blend {
                break;
            }
            after.push(r);
        }
        let blend = |back: usize, ahead: usize| Blend {
            back,
            ahead,
            radius: before[..=back]
                .iter()
                .chain(&after[..=ahead])
                .fold(0.0, |widest, &r| r.max(widest)),
        };
        let (back, ahead) = (before.len() - 1, after.len() - 1);
        let mut blends = vec![blend(back, ahead), blend(back / 2, ahead / 2), blend(0, 0)];
        blends.dedup();
        blends
    }

    /// The blends to try at seam `s`, the next to decide: those of `shapes_at` that make the
    /// stretch so far and the motion after the seam finish sooner than one after the other, with
    /// how long they take, soonest first.
    fn blends_at(&self, s: usize) -> Vec<(Blend, f32)> {
        let blends = self.shapes_at(s);
        let Some(seam) = self.seams.get(s).filter(|_| !blends.is_empty()) else {
            return vec![];
        };
        let (n, at) = (self.n, seam.at);
        // The stretch so far and the next motion, which ends at the next seam's stop.
        let end = self
            .seams
            .get(s + 1)
            .map_or(self.path.len() / n, |next| next.at + 3);
        let stretch = &self.path[self.start * n..end * n];
        let apart = self.so_far + self.apart[s + 1];
        let mut joined = vec![];
        let mut timed: Vec<(Blend, f32)> = blends
            .into_iter()
            .map(|b| {
                bridge(stretch, n, at - self.start, &b, &mut joined);
                (b, self.duration(&joined))
            })
            .filter(|&(_, duration)| duration < apart)
            .collect();
        timed.sort_by(|a, b| a.1.total_cmp(&b.1));
        timed
    }

    /// Seconds a rest-to-rest path takes, timed as [`Trajectory::try_new`] times it.
    fn duration(&self, path: &[f32]) -> f32 {
        Trajectory::try_new(self.robot, path, 1.0).map_or(f32::INFINITY, |t| t.duration())
    }

    /// Decides the seam under way: blended within `radius`, its stretch then taking `duration`, and
    /// `shift` control points added before the seams after it; or stopping there.
    fn advance(&mut self, blended: Option<(f32, f32)>, shift: isize) {
        match blended {
            Some((_, duration)) => self.so_far = duration,
            None => {
                self.start = self.seams[self.next].at;
                self.so_far = self.apart.get(self.next + 1).copied().unwrap_or(0.0);
            }
        }
        self.blends.push(blended.map(|(radius, _)| radius));
        self.next += 1;
        for seam in &mut self.seams[self.next..] {
            for index in [&mut seam.at, &mut seam.lower, &mut seam.upper] {
                *index = index
                    .checked_add_signed(shift)
                    .expect("a blend changes control points only before the seams after it");
            }
        }
        self.candidates = self.blends_at(self.next);
    }
}

impl Joining<'_> {
    /// The joined path cut at the stops it keeps, each piece ending and the next starting there;
    /// a piece of one motion is that motion as given.
    fn stretches(&self) -> Vec<Vec<f32>> {
        let n = self.n;
        let mut out = vec![];
        let (mut start, mut first) = (0, 0);
        for (k, seam) in self.seams.iter().enumerate() {
            if self.blends[k].is_none() {
                if first == k {
                    out.push(self.given[k].clone());
                } else {
                    out.push(self.path[start * n..(seam.at + 3) * n].to_vec());
                }
                (start, first) = (seam.at, k + 1);
            }
        }
        if first == self.seams.len() {
            out.push(self.given[first].clone());
        } else {
            out.push(self.path[start * n..].to_vec());
        }
        out
    }
}

/// `path` with blend `b` at the stop whose three control points start at `at`, into `out`; returns
/// the index of the first span that differs and the one after the last.
fn bridge(path: &[f32], n: usize, at: usize, b: &Blend, out: &mut Vec<f32>) -> (usize, usize) {
    let (enter, leave) = (at - 1 - b.back, at + 3 + b.ahead);
    let point = |i: usize| &path[i * n..(i + 1) * n];
    let (from, join, to) = (point(enter), point(at), point(leave));
    // The quintic from where the blend enters to where it leaves that sets out and arrives along
    // the paths there without curving yet: its curvature rises from zero, so jerk can follow it at
    // speed. Its speed at either end is the distance through the join, so along a straight line it
    // is the line at an even pace. It is spaced as the paths either side are, so the pace along the
    // path stays even, or, where both neighbours coincide (an interior stop), as the control
    // points it replaces are; never more than twice as many.
    let heading = |a: &[f32], b: &[f32], or: &[f32]| -> Vec<f32> {
        let length = rrt::distance(a, b);
        if length > 1e-9 {
            a.iter().zip(b).map(|(x, y)| (y - x) / length).collect()
        } else {
            let length = rrt::distance(a, or).max(1e-9);
            a.iter().zip(or).map(|(x, y)| (y - x) / length).collect()
        }
    };
    let length = rrt::distance(from, join) + rrt::distance(join, to);
    let set_out = heading(point(enter - 1), from, join);
    let arrive = heading(to, point(leave + 1), to);
    let neighbours =
        0.5 * (rrt::distance(point(enter - 1), from) + rrt::distance(to, point(leave + 1)));
    let spacing = if neighbours > 1e-9 {
        neighbours
    } else {
        let replaced: Vec<f32> = (enter..leave)
            .map(|i| rrt::distance(point(i), point(i + 1)))
            .filter(|&d| d > 1e-9)
            .collect();
        replaced.iter().sum::<f32>() / replaced.len().max(1) as f32
    };
    let gaps = if spacing > 1e-9 {
        ((length / spacing).round() as usize).clamp(1, 2 * (leave - enter))
    } else {
        1
    };
    out.clear();
    out.extend_from_slice(&path[..(enter + 1) * n]);
    for g in 1..gaps {
        let t = g as f32 / gaps as f32;
        let (t3, t4, t5) = (t * t * t, t * t * t * t, t * t * t * t * t);
        let (h0, h1) = (
            1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5,
            t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5,
        );
        let (h4, h5) = (
            -4.0 * t3 + 7.0 * t4 - 3.0 * t5,
            10.0 * t3 - 15.0 * t4 + 6.0 * t5,
        );
        out.extend(
            (0..n).map(|j| h0 * from[j] + h5 * to[j] + length * (h1 * set_out[j] + h4 * arrive[j])),
        );
    }
    out.extend_from_slice(&path[leave * n..]);
    // The spans from two before the gap to the one after it are new; the rest are the paths'.
    (enter - 2, enter + gaps)
}

/// The farthest any hand at `q` is from where it is at `join` (meters).
fn reach(robot: &Robot, q: &[f32], join: &[f32]) -> f32 {
    (0..robot.hands())
        .map(|hand| {
            let (a, b) = (robot.ee_pose(q, hand), robot.ee_pose(join, hand));
            a.position.distance(b.position)
        })
        .fold(0.0, f32::max)
}

/// A path's mean joint-space distance between neighbouring control points: how far it moves per
/// span. Timing follows a path best where this changes little along it.
fn spacing(cp: &[f32], n: usize) -> f32 {
    let points: Vec<&[f32]> = cp.chunks(n).collect();
    let length: f32 = points.windows(2).map(|w| rrt::distance(w[0], w[1])).sum();
    length / (points.len() - 3) as f32
}

/// The curve at knot `k`, the start of span `k - 1`: `(P[k-1] + 4 P[k] + P[k+1]) / 6`.
fn knot_point(cp: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut q = vec![0.0; n];
    spline::blend(cp, n, k - 1, basis(0.0), &mut q);
    q
}

impl Search for Joining<'_> {
    fn request(&mut self, buffer: &mut Vec<f32>) -> Option<EdgeQuery> {
        // The next blend to check, at this seam or a later one; a seam with none left keeps its
        // stop.
        let b = loop {
            if self.next == self.seams.len() {
                return None;
            }
            match self.candidates.first() {
                Some(&(b, _)) => break b,
                None => self.advance(None, 0),
            }
        };
        let (n, at) = (self.n, self.seams[self.next].at);
        let mut candidate = std::mem::take(&mut self.candidate);
        let (first, end) = bridge(&self.path, n, at, &b, &mut candidate);
        self.candidate = candidate;
        let mut samples = edges::reuse(buffer);
        self.starts.clear();
        spline::checked_spans(
            &self.candidate,
            n,
            first..end,
            CHECK_SAMPLES,
            self.resolution,
            &mut samples,
            &mut self.starts,
        );
        let row = samples.len();
        samples.resize(row + n, 0.0);
        spline::blend(&self.candidate, n, end, basis(0.0), &mut samples[row..]);
        self.samples = samples.len() / n;
        Some(EdgeQuery {
            world: self.world,
            edges: samples,
            wanted: None,
            spacing: 0.0,
            keep: None,
        })
    }

    fn answer(&mut self, answer: usize) {
        let (b, duration) = self.candidates.remove(0);
        if answer >= self.samples {
            let shift =
                (self.candidate.len() as isize - self.path.len() as isize) / self.n as isize;
            std::mem::swap(&mut self.path, &mut self.candidate);
            self.advance(Some((b.radius, duration)), shift);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::robot::{CollisionModel, RobotOptions};
    use crate::trajopt::{PlanOptions, PlanProblem};
    use crate::world::World;

    fn panda() -> Robot {
        let fingers = [
            ("panda_finger_joint1".to_string(), 0.04),
            ("panda_finger_joint2".to_string(), 0.04),
        ];
        let options = RobotOptions {
            ee_links: vec!["ee_link".into()],
            lock_joints: HashMap::from(fingers),
            collision_model: Some(
                CollisionModel::load("assets/franka/panda_collision.json").unwrap(),
            ),
            ..Default::default()
        };
        Robot::load("assets/franka/franka_panda.urdf", &options).unwrap()
    }

    /// Two planned motions sweeping across the front, the second carrying on where the first was
    /// going.
    fn sweep(robot: &Robot) -> Sequence {
        let stops = [
            vec![-1.2, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8],
            vec![-0.4, -0.3, 0.1, -1.9, 0.1, 1.7, 0.7],
            vec![0.4, -0.2, 0.2, -1.8, 0.2, 1.8, 0.6],
        ];
        let problems: Vec<PlanProblem> = stops
            .windows(2)
            .map(|w| PlanProblem::new(0, w[0].clone(), w[1].clone()))
            .collect();
        let device = Device::cpu(robot).unwrap();
        let worlds = device.upload(&[World::default()]).unwrap();
        let plans = crate::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap();
        let paths: Vec<Vec<f32>> = plans.solved().map(|s| s.solution.to_vec()).collect();
        assert_eq!(paths.len(), 2);
        Sequence { world: 0, paths }
    }

    /// A blend changes the curve only on the spans `bridge` reports, so checking them checks it.
    #[test]
    fn a_blend_changes_only_the_spans_it_reports() {
        let robot = panda();
        let n = robot.dof();
        let sequence = sweep(&robot);
        let o = JoinOptions {
            blend: 0.2,
            ..Default::default()
        };
        let joining = Joining::new(&robot, &sequence, &o);
        let at = joining.seams[0].at;
        assert!(!joining.candidates.is_empty(), "the sweep blends");
        let curve = |cp: &[f32], span: usize, u: f32| {
            let mut q = vec![0.0; n];
            spline::blend(cp, n, span, basis(u), &mut q);
            q
        };
        for (b, _) in &joining.candidates {
            let mut out = vec![];
            let (first, end) = bridge(&joining.path, n, at, b, &mut out);
            let (spans, out_spans) = (joining.path.len() / n - 3, out.len() / n - 3);
            let shift = out_spans as isize - spans as isize;
            for k in (0..first).chain(end..out_spans) {
                let given = if k < first {
                    k
                } else {
                    (k as isize - shift) as usize
                };
                for u in [0.0, 0.3, 0.7, 1.0] {
                    assert_eq!(
                        curve(&out, k, u),
                        curve(&joining.path, given, u),
                        "{b:?}: span {k} changed outside {first}..{end}"
                    );
                }
            }
            assert_ne!(
                curve(&out, first, 1.0),
                curve(&joining.path, first, 1.0),
                "{b:?}"
            );
            assert_ne!(
                curve(&out, end - 1, 0.0),
                curve(&joining.path, (end as isize - 1 - shift) as usize, 0.0),
                "{b:?}"
            );
        }
    }

    /// However wide the radius, the blends weighed at neighbouring joins never change the same
    /// knots, nor the sequence's ends.
    #[test]
    fn blends_at_neighbouring_joins_never_meet() {
        let robot = panda();
        let n = robot.dof();
        let mut sequence = sweep(&robot);
        let device = Device::cpu(&robot).unwrap();
        let worlds = device.upload(&[World::default()]).unwrap();
        let last = &sequence.paths[1];
        let onward = PlanProblem::new(
            0,
            last[last.len() - n..].to_vec(),
            vec![1.2, -0.4, 0.3, -2.0, 0.3, 1.9, 0.5],
        );
        let plans = crate::plan(&device, &worlds, &[onward], &PlanOptions::default()).unwrap();
        sequence.paths.push(plans.best(0).unwrap().to_vec());
        let o = JoinOptions {
            blend: 1.0,
            ..Default::default()
        };
        let joining = Joining::new(&robot, &sequence, &o);
        // The knots bounding the spans each seam's blends change: from `enter - 1` to `leave + 1`,
        // both still the paths' own; neighbours may share one.
        let changed: Vec<(usize, usize)> = (0..joining.seams.len())
            .map(|s| {
                let at = joining.seams[s].at;
                let shapes = joining.shapes_at(s);
                assert!(!shapes.is_empty(), "seam {s} has blends to weigh");
                let first = shapes.iter().map(|b| at - 2 - b.back).min().unwrap();
                let last = shapes.iter().map(|b| at + 4 + b.ahead).max().unwrap();
                (first, last)
            })
            .collect();
        assert!(
            changed[0].0 >= 1,
            "the first blend reaches the start: {changed:?}"
        );
        let end = joining.path.len() / n - 2;
        assert!(
            changed.last().unwrap().1 <= end,
            "the last reaches the end: {changed:?}"
        );
        for pair in changed.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "neighbouring blends meet: {changed:?}"
            );
        }
    }

    /// Where the control points either side of a blend coincide, as at a stop inside a path, the
    /// bridge is spaced as what it replaces, and no longer than twice as many points.
    #[test]
    fn a_bridge_between_stops_stays_short() {
        let n = 2;
        let configurations: [[f32; 2]; 5] =
            [[0.0, 0.0], [0.3, 0.1], [0.5, 0.4], [0.6, 0.8], [0.9, 1.0]];
        let path: Vec<f32> = configurations.iter().flat_map(|q| q.repeat(3)).collect();
        // The stop at index 6; the blend keeps index 4 and 10, each beside the same point (3 and
        // 11).
        let b = Blend {
            back: 1,
            ahead: 1,
            radius: 0.0,
        };
        let (enter, leave) = (4, 10);
        // Neighbours that coincide, and ones a hair apart, which leave the spacing as fine.
        let mut apart = path.clone();
        apart[3 * n] += 1e-7;
        apart[11 * n] += 1e-7;
        for path in [path, apart] {
            let mut out = vec![];
            let (first, end) = bridge(&path, n, 6, &b, &mut out);
            assert!(
                end - first <= 2 * (leave - enter) + 2,
                "{} new spans",
                end - first
            );
            assert!(out.len() / n <= path.len() / n + 2 * (leave - enter));
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }

    /// A blend whose new spans collide gives way to the next, soonest first, and the last to the
    /// stop, which leaves the motions as they were.
    #[test]
    fn colliding_blends_give_way_to_the_next_then_to_the_stop() {
        let robot = panda();
        let sequence = sweep(&robot);
        let o = JoinOptions {
            blend: 0.2,
            ..Default::default()
        };
        let mut buffer = vec![];
        let mut every_one_collides = Joining::new(&robot, &sequence, &o);
        let tries = every_one_collides.candidates.len();
        assert!(tries >= 2, "the sweep has blends to fall back between");
        let mut asked = 0;
        while every_one_collides.request(&mut buffer).is_some() {
            every_one_collides.answer(0);
            asked += 1;
        }
        assert_eq!(asked, tries, "each blend is checked once");
        assert_eq!(every_one_collides.blends, vec![None]);
        assert_eq!(every_one_collides.stretches(), sequence.paths);

        let mut second_clears = Joining::new(&robot, &sequence, &o);
        let (second, _) = second_clears.candidates[1];
        second_clears.request(&mut buffer).unwrap();
        second_clears.answer(0);
        let query = second_clears.request(&mut buffer).unwrap();
        second_clears.answer(query.edges.len() / robot.dof());
        assert!(second_clears.request(&mut buffer).is_none());
        assert_eq!(second_clears.blends, vec![Some(second.radius)]);
        assert_eq!(second_clears.stretches().len(), 1);
    }
}
