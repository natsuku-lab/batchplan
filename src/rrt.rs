//! RRT-Connect (Kuffner and LaValle, 2000) as VAMP runs it (Thomason, Kingston and Kavraki,
//! 2024). The straight line to each goal is tried first. If it collides, it is bent: a little off
//! the line and then farther, its colliding stretch is moved aside (via points beside where it
//! first and last collides), which keeps the straight route when an obstacle only grazes it. Then
//! one tree grows from the start and
//! one from the goal set, in turn (the smaller one alone while the other is at least twice its
//! size), toward Halton samples: first from a tube around the straight line that widens until it
//! spans the joint ranges, as cuRobo's roadmap samples an ellipsoid around it, so the route found
//! stays near the line where one does; then from every joint's range. An extension steps at most `range` from the nearest node and is
//! checked as an edge. The other tree then connects toward the new node in equal steps of at most
//! `range`, each checked in order, inserting every valid one. Dynamic domains skip samples farther
//! from their nearest node than its radius, which shrinks when extending from the node fails.
//! Continuous joints step and measure the short way round.
//!
//! Each problem is its own search: device threads run searches whole, checking their edges with
//! the CPU's SIMD collision checks (see `edges`), so results do not depend on the batch or device.
//! Threads a batch has no more searches for check the edges a long search will ask about next
//! (`ahead`), which speeds it without changing what it finds.

use std::sync::Arc;
use std::time::Instant;

use fearless_simd::{Level, Simd, dispatch, f32x8, prelude::*};

use crate::device::{Device, Worlds};
use crate::edges::{self, Deferred, EdgeQuery, Keep, Search, Wanted};
use crate::error::{Result, ensure_input};
use crate::lanes::LANES;
use crate::robot::Robot;

mod ahead;
pub(crate) use ahead::{Ahead, HELPERS};

#[derive(Clone, Copy, Debug)]
pub struct RrtOptions {
    /// Longest edge a tree grows by (joint-space distance).
    pub range: f32,
    /// Spacing of collision checks along edges (joint-space distance).
    pub resolution: f32,
    /// Samples a problem draws from the joint ranges before it gives up; the tube's (see the
    /// module docs) come on top. 0 tries only the straight lines and their bends.
    pub max_iterations: usize,
    /// Where in the Halton sequence samples start.
    pub rng_seed: u64,
    /// Clearance edges keep from the world (meters), shrinking to zero within `clearance_ramp`
    /// (joint-space distance) of the start and the goals, which may touch the world.
    pub clearance: f32,
    pub clearance_ramp: f32,
}

impl Default for RrtOptions {
    fn default() -> Self {
        Self {
            range: 1.0,
            resolution: 1.0 / 32.0,
            max_iterations: 10_000,
            rng_seed: 0,
            clearance: 0.0,
            clearance_ramp: 0.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RrtProblem {
    pub world: u32,
    pub start: Vec<f32>,
    /// The path may end at any of these.
    pub goals: Vec<Vec<f32>>,
    /// Joints that stay where `start` has them, as an arm that waits while another moves; every
    /// goal has the same values there.
    pub hold: Vec<usize>,
}

/// Whether `hold` names joints of the configurations whose values `goal` keeps from `start`.
pub(crate) fn holds(start: &[f32], goal: &[f32], hold: &[usize]) -> bool {
    hold.iter().all(|&j| j < start.len() && goal[j] == start[j])
}

#[derive(Clone, Debug)]
pub struct RrtResult {
    pub problems: Vec<RrtProblem>,
    /// Each problem's path as waypoints (`[k, dof]`, start first), if one was found. The edges
    /// between waypoints are collision-free at `resolution`.
    pub paths: Vec<Option<Vec<f32>>>,
}

pub(crate) fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

/// How far (joint-space distance) a bent straight line's via points move off the line, nearest
/// first, along each joint's axis both ways (across the line).
const BEND_RADII: [f32; 4] = [0.05, 0.1, 0.2, 0.4];

/// How far across the straight line (joint-space distance, along each axis across it) and past
/// its ends the first samples lie, and how many samples the tube takes before it doubles.
const TUBE_RADIUS: f32 = 0.4;
const TUBE_DOUBLING: usize = 64;

/// A dynamic domain's radius after a failed extension from a node without one.
const DOMAIN_RADIUS: f32 = 4.0;
/// How much a radius grows after a successful extension, and shrinks after a failed one.
const DOMAIN_ALPHA: f32 = 1e-4;
/// The smallest radius.
const DOMAIN_MIN: f32 = 1.0;

/// Where a search's samples come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sampling {
    /// First from the tube around the straight line, then from the joint ranges.
    NearLine,
    /// From the joint ranges alone.
    JointRanges,
}

/// Finds a collision-free waypoint path from each problem's start to one of its goals. Starts and
/// goals in collision are skipped; a problem without a free start or goal has no path.
pub fn connect(device: &Device, worlds: &Worlds, problems: &[RrtProblem], o: &RrtOptions) -> Result<RrtResult> {
    connect_until(device, worlds, problems, o, None)
}

/// [`connect`] that stops searching once `deadline` passes.
pub(crate) fn connect_until(
    device: &Device,
    worlds: &Worlds,
    problems: &[RrtProblem],
    o: &RrtOptions,
    deadline: Option<Instant>,
) -> Result<RrtResult> {
    let robot = device.robot();
    let n = robot.dof();
    ensure_input!(o.range > 0.0 && o.resolution > 0.0, "the range and resolution must be positive");
    ensure_input!(
        o.clearance >= 0.0 && o.clearance_ramp > 0.0,
        "the clearance must not be negative, and its ramp positive"
    );
    for (i, p) in problems.iter().enumerate() {
        ensure_input!(
            (p.world as usize) < worlds.len(),
            "problem {i} references world {} of {}",
            p.world,
            worlds.len()
        );
        ensure_input!(p.start.len() == n, "problem {i}: the start must have {n} values");
        ensure_input!(p.goals.iter().all(|g| g.len() == n), "problem {i}: goals must have {n} values");
        ensure_input!(
            robot.within(&p.start) && p.goals.iter().all(|g| robot.within(g)),
            "problem {i}: the start and goals must be within the joint limits"
        );
        ensure_input!(
            p.goals.iter().all(|g| holds(&p.start, g, &p.hold)),
            "problem {i}: held joints must be joints of the robot, with the start's values in every goal"
        );
    }
    let mut searches: Vec<_> = problems
        .iter()
        .map(|p| {
            let ahead = Arc::<Ahead>::default();
            Deferred::new(Some(ahead.clone()), move || Connect::new(robot, p, o, Sampling::NearLine, ahead))
        })
        .collect();
    let mut running: Vec<&mut dyn Search> = searches.iter_mut().map(|s| s as &mut dyn Search).collect();
    device.search(worlds, &mut running, deadline)?;
    Ok(RrtResult { problems: problems.to_vec(), paths: searches.into_iter().map(|s| s.into_inner().path).collect() })
}

/// Nodes with parents, and each node's dynamic-domain radius. Coordinates are stored per joint
/// (`coords[j][node]`), padded with NaN to whole groups of eight. Nearest neighbours come from a
/// k-d tree, or, for robots with continuous joints, whose distances wrap, from a scan of eight
/// nodes per instruction, both at the CPU's SIMD level.
#[derive(Clone)]
struct Tree {
    coords: Vec<Vec<f32>>,
    parent: Vec<Option<usize>>,
    radius: Vec<f32>,
    kd: Option<KdTree>,
    level: Level,
}

impl Tree {
    fn new(wrap: &[bool]) -> Self {
        let kd = (!wrap.contains(&true)).then(KdTree::default);
        Self { coords: vec![vec![]; wrap.len()], parent: vec![], radius: vec![], kd, level: Level::new() }
    }

    fn len(&self) -> usize {
        self.parent.len()
    }

    /// Appends node `i` to `out`.
    fn push_node(&self, i: usize, out: &mut Vec<f32>) {
        out.extend(self.coords.iter().map(|c| c[i]));
    }

    fn add(&mut self, q: &[f32], parent: Option<usize>) -> usize {
        let i = self.len();
        for (c, &v) in self.coords.iter_mut().zip(q) {
            if i.is_multiple_of(8) {
                c.extend([f32::NAN; 8]);
            }
            c[i] = v;
        }
        self.parent.push(parent);
        self.radius.push(f32::INFINITY);
        if let Some(kd) = &mut self.kd {
            kd.insert(q, i);
        }
        i
    }

    /// The node nearest `q` and its distance, continuous joints measured the short way around.
    fn nearest(&mut self, q: &[f32], wrap: &[bool]) -> (usize, f32) {
        if let Some(kd) = &mut self.kd {
            let (i, squared) = dispatch!(self.level, simd => kd.nearest(simd, q));
            return (i, squared.sqrt());
        }
        self.scan(q, wrap)
    }

    /// [`Tree::nearest`] by scanning every node.
    fn scan(&self, q: &[f32], wrap: &[bool]) -> (usize, f32) {
        dispatch!(self.level, simd => self.scan_at(simd, q, wrap))
    }

    #[inline(always)]
    fn scan_at<S: Simd>(&self, simd: S, q: &[f32], wrap: &[bool]) -> (usize, f32) {
        use std::f32::consts::TAU;
        // Per lane, the smallest squared distance and the group it was in.
        let (mut least, mut group) = (f32x8::splat(simd, f32::INFINITY), f32x8::splat(simd, 0.0));
        for g in 0..self.len().div_ceil(8) {
            let mut sum = f32x8::splat(simd, 0.0);
            for (j, coords) in self.coords.iter().enumerate() {
                let mut d = f32x8::from_slice(simd, &coords[8 * g..8 * g + 8]) - q[j];
                if wrap[j] {
                    d = d - (d * (1.0 / TAU)).round_ties_even() * TAU;
                }
                sum += d * d;
            }
            // NaN padding compares false, so it never wins.
            let closer = sum.simd_lt(least);
            least = closer.select(sum, least);
            group = closer.select(f32x8::splat(simd, g as f32), group);
        }
        let (least, group): ([f32; 8], [f32; 8]) = (least.into(), group.into());
        let lane = (0..8)
            .min_by(|&a, &b| least[a].total_cmp(&least[b]).then(group[a].total_cmp(&group[b])))
            .expect("eight lanes");
        (8 * group[lane] as usize + lane, least[lane].sqrt())
    }

    /// Node `i` and its ancestors up to the root, `i` first.
    fn branch(&self, mut i: usize) -> Vec<f32> {
        let mut out = vec![];
        self.push_node(i, &mut out);
        while let Some(p) = self.parent[i] {
            self.push_node(p, &mut out);
            i = p;
        }
        out
    }
}

/// Nodes a k-d tree leaf holds before it splits, as in nigh's `KDTreeBatch<128>`.
const LEAF: usize = 128;

/// An incremental k-d tree over a [`Tree`]'s nodes, as nigh's batched k-d tree that VAMP uses: a
/// leaf that outgrows `LEAF` nodes splits at the median of its widest axis. Leaves keep their own
/// copy of their nodes' coordinates, eight nodes to a group, so a leaf is measured eight nodes per
/// instruction from memory of its own.
#[derive(Clone, Default)]
struct KdTree {
    cells: Vec<Cell>,
    /// Cells still to search and their gap to the query, squared.
    pending: Vec<(usize, f32)>,
}

#[derive(Clone)]
enum Cell {
    Leaf(Leaf),
    /// Nodes below `value` on `axis` are under `low`, the rest under `high`.
    Split {
        axis: usize,
        value: f32,
        low: usize,
        high: usize,
    },
}

/// A leaf's nodes, and their coordinates in groups of eight nodes per joint
/// (`coords[(group * dof + joint) * 8 + lane]`), padded with NaN.
#[derive(Clone, Default)]
struct Leaf {
    nodes: Vec<usize>,
    coords: Vec<f32>,
}

impl Leaf {
    fn push(&mut self, node: usize, q: &[f32]) {
        let (n, k) = (q.len(), self.nodes.len());
        if k.is_multiple_of(8) {
            self.coords.extend(std::iter::repeat_n(f32::NAN, 8 * n));
        }
        for (j, &v) in q.iter().enumerate() {
            self.coords[((k / 8) * n + j) * 8 + k % 8] = v;
        }
        self.nodes.push(node);
    }

    /// Coordinate `j` of the leaf's `k`th node.
    fn coord(&self, k: usize, j: usize, n: usize) -> f32 {
        self.coords[((k / 8) * n + j) * 8 + k % 8]
    }

    /// Updates `best` (node, squared distance) with the leaf's nodes nearer `q`, the lower node
    /// first among equals.
    #[inline(always)]
    fn search<S: Simd>(&self, simd: S, q: &[f32], best: &mut (usize, f32)) {
        let n = q.len();
        for (g, group) in self.coords.chunks(8 * n).enumerate() {
            let mut sum = f32x8::splat(simd, 0.0);
            for (j, &v) in q.iter().enumerate() {
                let d = f32x8::from_slice(simd, &group[8 * j..8 * j + 8]) - v;
                sum += d * d;
            }
            // NaN padding compares false, so it never wins.
            if !sum.simd_le(f32x8::splat(simd, best.1)).any_true() {
                continue;
            }
            let sums: [f32; 8] = sum.into();
            for (k, &squared) in sums.iter().enumerate().take(self.nodes.len() - 8 * g) {
                let node = self.nodes[8 * g + k];
                if squared < best.1 || (squared == best.1 && node < best.0) {
                    *best = (node, squared);
                }
            }
        }
    }
}

impl KdTree {
    fn insert(&mut self, q: &[f32], node: usize) {
        if self.cells.is_empty() {
            self.cells.push(Cell::Leaf(Leaf::default()));
        }
        let mut cell = 0;
        while let Cell::Split { axis, value, low, high } = self.cells[cell] {
            cell = if q[axis] < value { low } else { high };
        }
        let Cell::Leaf(leaf) = &mut self.cells[cell] else { unreachable!() };
        leaf.push(node, q);
        if leaf.nodes.len() <= LEAF {
            return;
        }
        let n = q.len();
        let count = leaf.nodes.len();
        let spread = |axis: usize| {
            let (lo, hi) = (0..count).fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), k| {
                let v = leaf.coord(k, axis, n);
                (lo.min(v), hi.max(v))
            });
            hi - lo
        };
        let axis = (0..n).max_by(|&a, &b| spread(a).total_cmp(&spread(b))).expect("a joint");
        let mut values: Vec<f32> = (0..count).map(|k| leaf.coord(k, axis, n)).collect();
        let middle = values.len() / 2;
        let value = *values.select_nth_unstable_by(middle, f32::total_cmp).1;
        let (mut low, mut high) = (Leaf::default(), Leaf::default());
        let mut point = vec![0.0; n];
        for k in 0..count {
            for (j, p) in point.iter_mut().enumerate() {
                *p = leaf.coord(k, j, n);
            }
            if point[axis] < value { &mut low } else { &mut high }.push(leaf.nodes[k], &point);
        }
        // Nodes all equal on the widest axis cannot be split; the leaf grows instead.
        if low.nodes.is_empty() || high.nodes.is_empty() {
            return;
        }
        let (l, h) = (self.cells.len(), self.cells.len() + 1);
        self.cells[cell] = Cell::Split { axis, value, low: l, high: h };
        self.cells.push(Cell::Leaf(low));
        self.cells.push(Cell::Leaf(high));
    }

    /// The node nearest `q` and its squared distance, the lowest node among equals. The search
    /// descends to the leaf holding `q`, then visits the far sides it passed, nearest last pushed
    /// first, skipping those farther than the best node found.
    #[inline(always)]
    fn nearest<S: Simd>(&mut self, simd: S, q: &[f32]) -> (usize, f32) {
        let Self { cells, pending } = self;
        let mut best = (0, f32::INFINITY);
        pending.clear();
        pending.push((0, 0.0));
        while let Some((mut cell, gap)) = pending.pop() {
            if gap > best.1 {
                continue;
            }
            loop {
                match &cells[cell] {
                    &Cell::Split { axis, value, low, high } => {
                        let d = q[axis] - value;
                        let (near, far) = if d < 0.0 { (low, high) } else { (high, low) };
                        pending.push((far, d * d));
                        cell = near;
                    }
                    Cell::Leaf(leaf) => {
                        leaf.search(simd, q, &mut best);
                        break;
                    }
                }
            }
        }
        best
    }
}

/// Halton points in the unit cube, in prime bases from 3.
#[derive(Clone)]
struct Halton {
    bases: Vec<u32>,
    index: u64,
}

impl Halton {
    fn new(dims: usize, start: u64) -> Self {
        let mut bases = vec![];
        let mut k = 3;
        while bases.len() < dims {
            if (3..k).step_by(2).take_while(|d| d * d <= k).all(|d| k % d != 0) {
                bases.push(k);
            }
            k += 2;
        }
        Self { bases, index: start }
    }

    fn next(&mut self, out: &mut [f32]) {
        self.index += 1;
        self.point(self.index, out);
    }

    /// Point `index` of the sequence.
    fn point(&self, index: u64, out: &mut [f32]) {
        for (o, &b) in out.iter_mut().zip(&self.bases) {
            let (mut i, mut f, mut r) = (index, 1.0f64, 0.0f64);
            while i > 0 {
                f /= f64::from(b);
                r += f * (i % u64::from(b)) as f64;
                i /= u64::from(b);
            }
            *o = r as f32;
        }
    }
}

enum Stage {
    /// Checking the start.
    Start,
    /// Checking goal `i`.
    Goal(usize),
    /// Checking the straight line to each free goal.
    Straight,
    /// Finding where the straight line to the nearest free goal (`Connect::line_goal`) first
    /// collides, from the start (`true`), then from the goal.
    Locate(bool),
    /// Checking the line bent the `k`th way (`Connect::vias`).
    Bend(usize),
    Sample,
    /// Checking the step from node `near` of the extending tree to `Connect::new`.
    Extend {
        near: usize,
    },
    /// Checking `Connect::steps`, from node `from` of the extending tree to node `other` of the other.
    Connect {
        from: usize,
        other: usize,
    },
    Done,
}

/// One problem's RRT-Connect.
pub(crate) struct Connect<'r> {
    robot: &'r Robot,
    world: u32,
    range: f32,
    resolution: f32,
    max_iterations: usize,
    start: Vec<f32>,
    goals: Vec<Vec<f32>>,
    /// Per joint, whether it stays where `start` has it.
    held: Vec<bool>,
    free_goals: Vec<usize>,
    /// The extending tree, then the other; `start_first` while the first is the start's.
    trees: [Tree; 2],
    start_first: bool,
    halton: Halton,
    sampling: Sampling,
    /// Samples drawn from the joint ranges, and from the tube.
    iterations: usize,
    tube_samples: usize,
    stage: Stage,
    keep: Option<Arc<Keep>>,
    /// Working configurations, reused from one iteration to the next: the sample (then the node a
    /// connection heads for), the configuration an extension reaches, and a connection's steps.
    sample: Vec<f32>,
    new: Vec<f32>,
    steps: Vec<f32>,
    /// Bending: the goal turned toward the start, where along the line it first and last collides
    /// (0 to 1), the directions via points move in (`[k, dof]`, unit, across the line), and the
    /// via points of the bend under check.
    line_goal: Vec<f32>,
    collides: [f32; 2],
    directions: Vec<f32>,
    vias: Vec<f32>,
    /// An orthonormal basis across the line the tube follows (`[k, dof]`), set when the straight
    /// line first collides.
    across: Vec<f32>,
    /// The path found, once the search is over.
    pub(crate) path: Option<Vec<f32>>,
    /// What the search shares with threads that help it (see `ahead`).
    sharing: ahead::Sharing,
}

impl<'r> Connect<'r> {
    pub(crate) fn new(robot: &'r Robot, p: &RrtProblem, o: &RrtOptions, sampling: Sampling, ahead: Arc<Ahead>) -> Self {
        let n = robot.dof();
        Self {
            robot,
            world: p.world,
            range: o.range,
            resolution: o.resolution,
            max_iterations: o.max_iterations,
            start: p.start.clone(),
            goals: p.goals.clone(),
            held: (0..n).map(|j| p.hold.contains(&j)).collect(),
            free_goals: vec![],
            trees: [Tree::new(&robot.continuous), Tree::new(&robot.continuous)],
            start_first: true,
            halton: Halton::new(n, o.rng_seed),
            sampling,
            iterations: 0,
            tube_samples: 0,
            stage: if p.goals.is_empty() { Stage::Done } else { Stage::Start },
            keep: (o.clearance > 0.0).then(|| {
                let ends = std::iter::once(&p.start).chain(&p.goals).cloned().collect();
                Arc::new(Keep { clearance: o.clearance, ramp: o.clearance_ramp, ends })
            }),
            sample: vec![0.0; n],
            new: Vec::with_capacity(n),
            steps: vec![],
            line_goal: vec![0.0; n],
            collides: [0.5; 2],
            directions: vec![],
            vias: Vec::with_capacity(2 * n),
            across: vec![],
            path: None,
            sharing: ahead::Sharing::new(ahead),
        }
    }

    /// Continues a search that kept a clearance and found no path, without it, for `samples` more
    /// samples on the trees it has grown: what keeps a clearance is free without one. The straight
    /// lines are checked again first, and the dynamic domains start over, since extensions that
    /// failed keeping the clearance may not fail without it. A search that never started (start
    /// or every goal in collision) stays over.
    pub(crate) fn relax(&mut self, samples: usize) {
        self.keep = None;
        self.sharing.restart();
        if self.path.is_some() || self.trees.iter().any(|t| t.len() == 0) {
            return;
        }
        if !self.start_first {
            self.trees.swap(0, 1);
            self.start_first = true;
        }
        for tree in &mut self.trees {
            tree.radius.fill(f32::INFINITY);
        }
        self.max_iterations = self.iterations + samples;
        self.stage = Stage::Straight;
    }

    /// The free goal nearest the start, turned toward it, into `line_goal`, and the directions
    /// across the line from the start to it that via points move in: each joint's axis, both ways,
    /// with the part along the line removed.
    fn line_to(&mut self, goal: usize) {
        let n = self.robot.dof();
        self.line_goal.copy_from_slice(&self.goals[goal]);
        self.robot.turn_toward(&self.start, &mut self.line_goal);
        let along: Vec<f32> = self.line_goal.iter().zip(&self.start).map(|(g, s)| g - s).collect();
        let length = distance(&self.line_goal, &self.start);
        let mut d = vec![0.0; n];
        self.directions.clear();
        for k in (0..2 * n).filter(|k| !self.held[k / 2]) {
            d.fill(0.0);
            d[k / 2] = if k % 2 == 0 { 1.0 } else { -1.0 };
            if length > 0.0 {
                let dot: f32 = d.iter().zip(&along).map(|(a, b)| a * b).sum::<f32>() / (length * length);
                d.iter_mut().zip(&along).for_each(|(v, a)| *v -= dot * a);
            }
            let norm = d.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-3 {
                self.directions.extend(d.iter().map(|v| v / norm));
            }
        }
        // The tube follows the first line that collides: Gram-Schmidt on the free joints' axes.
        if self.sampling == Sampling::JointRanges || !self.across.is_empty() || length == 0.0 {
            return;
        }
        for j in (0..n).filter(|&j| !self.held[j]) {
            d.fill(0.0);
            d[j] = 1.0;
            let dot = along[j] / (length * length);
            d.iter_mut().zip(&along).for_each(|(v, a)| *v -= dot * a);
            for b in self.across.chunks(n) {
                let dot: f32 = d.iter().zip(b).map(|(x, y)| x * y).sum();
                d.iter_mut().zip(b).for_each(|(v, y)| *v -= dot * y);
            }
            let norm = d.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-3 {
                self.across.extend(d.iter().map(|v| v / norm));
            }
        }
    }

    /// The tube's radius for the next sample, while it is narrower than the widest joint range.
    fn tube(&self) -> Option<f32> {
        tube_radius(self.robot, &self.across, self.max_iterations, self.tube_samples)
    }

    /// Whether any sample came from the tube.
    pub(crate) fn sampled_tube(&self) -> bool {
        self.tube_samples > 0
    }

    /// The free goal nearest the start.
    fn nearest_goal(&self) -> usize {
        let mut goal = vec![0.0; self.robot.dof()];
        let mut best = (self.free_goals[0], f32::INFINITY);
        for &g in &self.free_goals {
            goal.copy_from_slice(&self.goals[g]);
            self.robot.turn_toward(&self.start, &mut goal);
            let d = distance(&self.start, &goal);
            if d < best.1 {
                best = (g, d);
            }
        }
        best.0
    }

    fn query(&self, edges: Vec<f32>, wanted: Wanted) -> Option<EdgeQuery> {
        Some(EdgeQuery {
            world: self.world,
            edges,
            wanted: Some(wanted),
            spacing: self.resolution,
            keep: self.keep.clone(),
        })
    }

    /// The path from the start through the extending tree's node `a` and the other's node `b`.
    fn finish(&mut self, a: usize, b: usize) {
        let n = self.robot.dof();
        let mut path: Vec<f32> = self.trees[0].branch(a).chunks(n).rev().flatten().copied().collect();
        path.extend(self.trees[1].branch(b));
        if !self.start_first {
            path = path.chunks(n).rev().flatten().copied().collect();
        }
        // The trees may hold a continuous joint a turn apart where they meet.
        for k in 1..path.len() / n {
            let (before, after) = path.split_at_mut(k * n);
            self.robot.turn_toward(&before[(k - 1) * n..], &mut after[..n]);
        }
        self.path = Some(path);
        self.stage = Stage::Done;
    }
}

/// Whether the trees swap before a sample, given the extending tree's size `a` and the other's
/// `b`: they take turns, unless the other is at least twice the extending one's size.
fn swaps(a: f32, b: f32) -> bool {
    (a - b).abs() / a < 1.0
}

/// The tube's radius for a sample after `tube_samples` from it, while it is narrower than the
/// widest joint range (see `Connect::tube`).
fn tube_radius(robot: &Robot, across: &[f32], max_iterations: usize, tube_samples: usize) -> Option<f32> {
    if across.is_empty() || max_iterations == 0 {
        return None;
    }
    let doublings = (tube_samples / TUBE_DOUBLING).min(31) as i32;
    let r = TUBE_RADIUS * 2f32.powi(doublings);
    let widest = robot.lower.iter().zip(&robot.upper).map(|(l, u)| u - l).fold(0.0, f32::max);
    (r < widest).then_some(r)
}

/// Turns `target`, a Halton point in the unit cube, into a sample: in the tube of radius `tube`
/// around the line from `start` to `line_goal` (`across` spanning the directions across it), or in
/// the joint ranges, with held joints at the start. `unit` is working memory.
#[allow(clippy::too_many_arguments)]
fn place(
    robot: &Robot,
    start: &[f32],
    line_goal: &[f32],
    across: &[f32],
    held: &[bool],
    tube: Option<f32>,
    target: &mut [f32],
    unit: &mut Vec<f32>,
) {
    let n = robot.dof();
    if let Some(r) = tube {
        // Along the line, past either end by the radius, then across it in a box.
        unit.clear();
        unit.extend_from_slice(target);
        let past = r / distance(start, line_goal);
        let t = -past + (1.0 + 2.0 * past) * unit[0];
        for (j, q) in target.iter_mut().enumerate() {
            *q = start[j] + (line_goal[j] - start[j]) * t;
        }
        for (b, u) in across.chunks(n).zip(&unit[1..]) {
            let c = r * (2.0 * u - 1.0);
            target.iter_mut().zip(b).for_each(|(q, y)| *q += c * y);
        }
    }
    for (j, t) in target.iter_mut().enumerate() {
        *t = if held[j] {
            start[j]
        } else if tube.is_none() {
            robot.lower[j] + *t * (robot.upper[j] - robot.lower[j])
        } else if robot.continuous[j] {
            *t
        } else {
            t.clamp(robot.lower[j], robot.upper[j])
        };
    }
}

/// Into `new`, where an extension from node `near` of `tree` toward `target`, `d` away, reaches: at
/// most `range` along the way, `target` turned toward the node first.
fn reach(robot: &Robot, range: f32, tree: &Tree, near: usize, d: f32, target: &mut [f32], new: &mut Vec<f32>) {
    new.clear();
    tree.push_node(near, new);
    robot.turn_toward(new, target);
    if d > range {
        for (a, b) in new.iter_mut().zip(target.iter()) {
            *a += (b - *a) * (range / d);
        }
    } else {
        new.copy_from_slice(target);
    }
}

/// Into `steps`, a connection's steps from `new` to node `other` of `tree`, `d` away: equal, at most
/// `range` long, the last at the node turned toward `new` (into `target`).
#[allow(clippy::too_many_arguments)]
fn steps_to(
    robot: &Robot,
    range: f32,
    new: &[f32],
    tree: &Tree,
    other: usize,
    d: f32,
    target: &mut Vec<f32>,
    steps: &mut Vec<f32>,
) {
    target.clear();
    tree.push_node(other, target);
    robot.turn_toward(new, target);
    let count = (d / range).ceil().max(1.0) as usize;
    steps.clear();
    for k in 1..count {
        let t = k as f32 / count as f32;
        steps.extend(new.iter().zip(target.iter()).map(|(a, b)| a + (b - a) * t));
    }
    steps.extend_from_slice(target);
}

/// Appends the edges between a connection's steps (`[k, dof]`) to `edges`, which holds the node
/// it starts from.
fn connect_edges(steps: &[f32], n: usize, edges: &mut Vec<f32>) {
    for (k, step) in steps.chunks(n).enumerate() {
        if k > 0 {
            edges.extend_from_slice(&steps[(k - 1) * n..k * n]);
        }
        edges.extend_from_slice(step);
    }
}

impl Search for Connect<'_> {
    fn help(&self) -> Option<Arc<Ahead>> {
        Some(self.sharing.ahead())
    }

    fn request(&mut self, buffer: &mut Vec<f32>) -> Option<EdgeQuery> {
        let n = self.robot.dof();
        loop {
            match &self.stage {
                Stage::Start | Stage::Goal(_) => {
                    let q = match self.stage {
                        Stage::Goal(i) => &self.goals[i],
                        _ => &self.start,
                    };
                    let mut edges = edges::reuse(buffer);
                    edges.extend_from_slice(q);
                    edges.extend_from_slice(q);
                    return self.query(edges, Wanted::Prefix);
                }
                Stage::Straight => {
                    if self.free_goals.is_empty() {
                        self.stage = Stage::Done;
                        continue;
                    }
                    let mut edges = edges::reuse(buffer);
                    for &g in &self.free_goals {
                        let mut goal = self.goals[g].clone();
                        self.robot.turn_toward(&self.start, &mut goal);
                        edges.extend_from_slice(&self.start);
                        edges.extend_from_slice(&goal);
                    }
                    return self.query(edges, Wanted::FirstValid);
                }
                &Stage::Locate(forward) => {
                    // The line's states `resolution` apart, ends excluded, from the start or the goal.
                    let count = ((distance(&self.start, &self.line_goal) / self.resolution).ceil() as usize).max(2);
                    let mut edges = edges::reuse(buffer);
                    for m in 1..count {
                        let t = if forward { m } else { count - m } as f32 / count as f32;
                        edges.extend(self.start.iter().zip(&self.line_goal).map(|(s, g)| s + (g - s) * t));
                    }
                    if edges.is_empty() {
                        self.stage = Stage::Sample;
                        continue;
                    }
                    return Some(EdgeQuery {
                        world: self.world,
                        edges,
                        wanted: None,
                        spacing: self.resolution,
                        keep: self.keep.clone(),
                    });
                }
                &Stage::Bend(k) => {
                    // Per radius and direction, the colliding stretch moved aside: two via points, or
                    // one where the line collides at a point.
                    let directions = self.directions.len() / n;
                    if directions == 0 || k >= BEND_RADII.len() * directions {
                        self.stage = Stage::Sample;
                        continue;
                    }
                    let (r, d) = (BEND_RADII[k / directions], &self.directions[(k % directions) * n..][..n]);
                    let [first, last] = self.collides;
                    let at: &[f32] = if last > first { &[first, last] } else { &[first] };
                    self.vias.clear();
                    for &t in at {
                        let start = self.vias.len();
                        self.vias.extend(
                            self.start.iter().zip(&self.line_goal).zip(d).map(|((s, g), d)| s + (g - s) * t + r * d),
                        );
                        if !self.robot.within(&self.vias[start..]) {
                            self.vias.clear();
                            break;
                        }
                    }
                    if self.vias.is_empty() {
                        self.stage = Stage::Bend(k + 1);
                        continue;
                    }
                    // The moved stretch first, where a bend most often still collides; the answer
                    // counts the edges free in this order, so all of them are free when it is their count.
                    let mut edges = edges::reuse(buffer);
                    let last = self.vias.len() - n;
                    if last > 0 {
                        edges.extend_from_slice(&self.vias);
                    }
                    edges.extend_from_slice(&self.start);
                    edges.extend_from_slice(&self.vias[..n]);
                    edges.extend_from_slice(&self.vias[last..]);
                    edges.extend_from_slice(&self.line_goal);
                    return self.query(edges, Wanted::Prefix);
                }
                Stage::Sample => {
                    let sample = self.tube_samples + self.iterations;
                    let tube = self.tube();
                    if tube.is_some() {
                        self.tube_samples += 1;
                    } else if self.iterations >= self.max_iterations {
                        self.stage = Stage::Done;
                        continue;
                    } else {
                        self.iterations += 1;
                    }
                    let (a, b) = (self.trees[0].len() as f32, self.trees[1].len() as f32);
                    if swaps(a, b) {
                        self.trees.swap(0, 1);
                        self.start_first = !self.start_first;
                    }
                    let target = &mut self.sample;
                    self.halton.next(target);
                    place(
                        self.robot,
                        &self.start,
                        &self.line_goal,
                        &self.across,
                        &self.held,
                        tube,
                        target,
                        &mut self.new,
                    );
                    let (near, d) = self.trees[0].nearest(target, &self.robot.continuous);
                    if self.trees[0].radius[near] < d {
                        continue;
                    }
                    self.report_sample(sample);
                    reach(self.robot, self.range, &self.trees[0], near, d, &mut self.sample, &mut self.new);
                    self.stage = Stage::Extend { near };
                }
                &Stage::Extend { near } => {
                    let mut edges = edges::reuse(buffer);
                    self.trees[0].push_node(near, &mut edges);
                    edges.extend_from_slice(&self.new);
                    if let Some(answer) = self.forecast(&edges, false) {
                        *buffer = edges;
                        self.answer(answer);
                        continue;
                    }
                    return self.query(edges, Wanted::Prefix);
                }
                &Stage::Connect { from, .. } => {
                    let mut edges = edges::reuse(buffer);
                    self.trees[0].push_node(from, &mut edges);
                    connect_edges(&self.steps, n, &mut edges);
                    if let Some(answer) = self.forecast(&edges, true) {
                        *buffer = edges;
                        self.answer(answer);
                        continue;
                    }
                    return self.query(edges, Wanted::Prefix);
                }
                Stage::Done => return None,
            }
        }
    }

    fn answer(&mut self, answer: usize) {
        let n = self.robot.dof();
        match std::mem::replace(&mut self.stage, Stage::Sample) {
            Stage::Start => {
                self.stage = if answer == 1 {
                    self.trees[0].add(&self.start, None);
                    Stage::Goal(0)
                } else {
                    Stage::Done
                };
            }
            Stage::Goal(i) => {
                if answer == 1 {
                    self.trees[1].add(&self.goals[i], None);
                    self.free_goals.push(i);
                }
                self.stage = if i + 1 < self.goals.len() { Stage::Goal(i + 1) } else { Stage::Straight };
            }
            Stage::Straight => {
                if let Some(&g) = self.free_goals.get(answer) {
                    let goal = self.free_goals.iter().position(|&f| f == g).expect("a free goal");
                    self.finish(0, goal);
                } else if !self.free_goals.is_empty() {
                    self.line_to(self.nearest_goal());
                    self.stage = Stage::Locate(true);
                }
            }
            Stage::Locate(forward) => {
                // Free states come in groups of eight; the first collision is in the next group.
                let count = ((distance(&self.start, &self.line_goal) / self.resolution).ceil() as usize).max(2);
                let m = (answer + LANES / 2 + 1).min(count - 1);
                if forward {
                    self.collides[0] = m as f32 / count as f32;
                    self.stage = Stage::Locate(false);
                } else {
                    self.collides[1] = ((count - m) as f32 / count as f32).max(self.collides[0]);
                    self.stage = Stage::Bend(0);
                }
            }
            Stage::Bend(k) => {
                if answer == self.vias.len() / n + 1 {
                    let path = [&self.start[..], &self.vias, &self.line_goal].concat();
                    self.path = Some(path);
                    self.stage = Stage::Done;
                } else {
                    self.stage = Stage::Bend(k + 1);
                }
            }
            Stage::Extend { near } => {
                let tree = &mut self.trees[0];
                if answer == 0 {
                    let r = &mut tree.radius[near];
                    *r = if r.is_infinite() { DOMAIN_RADIUS } else { (*r * (1.0 - DOMAIN_ALPHA)).max(DOMAIN_MIN) };
                    return;
                }
                if tree.radius[near].is_finite() {
                    tree.radius[near] *= 1.0 + DOMAIN_ALPHA;
                }
                let from = tree.add(&self.new, Some(near));
                let (other, d) = self.trees[1].nearest(&self.new, &self.robot.continuous);
                steps_to(
                    self.robot,
                    self.range,
                    &self.new,
                    &self.trees[1],
                    other,
                    d,
                    &mut self.sample,
                    &mut self.steps,
                );
                self.stage = Stage::Connect { from, other };
            }
            Stage::Connect { from, other } => {
                let count = self.steps.len() / n;
                let mut last = from;
                for step in self.steps.chunks(n).take(answer.min(count - 1)) {
                    last = self.trees[0].add(step, Some(last));
                }
                if answer == count {
                    self.finish(last, other);
                }
            }
            Stage::Sample | Stage::Done => unreachable!("answers follow requests"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Bends move via points only along joints that may move: none of the directions a bend can
    /// take moves a held joint.
    #[test]
    fn bends_leave_held_joints_alone() {
        use crate::robot::RobotOptions;
        let ur5e = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/ur5e/ur_description/urdf/ur5e.urdf");
        let robot = Robot::load(ur5e, &RobotOptions::default()).unwrap();
        let start = vec![0.0; 6];
        let goal = vec![0.5, 0.0, -0.4, 0.0, 0.3, 0.0];
        let hold = vec![1, 3, 5];
        let problem = RrtProblem { world: 0, start, goals: vec![goal], hold: hold.clone() };
        let mut connect = Connect::new(&robot, &problem, &RrtOptions::default(), Sampling::NearLine, Arc::default());
        connect.line_to(0);
        let directions: Vec<&[f32]> = connect.directions.chunks(6).collect();
        assert!(directions.len() >= 4, "{} directions", directions.len());
        for d in directions {
            assert!(hold.iter().all(|&j| d[j] == 0.0), "a bend moves a held joint: {d:?}");
        }
    }

    /// The k-d tree finds exactly the node the scan finds, as the tree grows: the same distance,
    /// and the lowest node among equals.
    #[test]
    fn the_kd_tree_finds_the_nearest_node() {
        let mut rng = Rng::new(3);
        let mut tree = Tree::new(&[false; 7]);
        assert!(tree.kd.is_some());
        for added in 0..3000 {
            let q: Vec<f32> = (0..7).map(|_| rng.range(-3.0, 3.0)).collect();
            tree.add(&q, None);
            if added % 50 == 0 {
                for _ in 0..20 {
                    let target: Vec<f32> = (0..7).map(|_| rng.range(-3.5, 3.5)).collect();
                    let (a, da) = tree.nearest(&target, &[false; 7]);
                    let (b, db) = tree.scan(&target, &[false; 7]);
                    assert_eq!((a, da), (b, db), "the k-d tree and the scan disagree");
                }
            }
        }
    }
}
