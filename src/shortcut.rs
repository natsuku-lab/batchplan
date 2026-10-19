//! Shortening and smoothing waypoint paths, as VAMP simplifies them. A path whose ends see each
//! other becomes that edge. Otherwise each pass shortcuts (from every waypoint in turn, to the
//! farthest later one it sees directly, dropping the waypoints between) and then smooths (each
//! edge split at its middle, and every original waypoint moved a quarter of the way toward its
//! neighbours' midpoints where both new edges are free), until a pass changes nothing. With
//! `straighten`, between the first pass's shortcuts and its smoothing, each joint in turn is
//! straightened where that shortens the path: it changes at one rate along the whole path while the
//! others keep their values, as HPP's partial shortcut makes it, which removes a joint's swing out
//! and back that shortcuts between whole configurations keep. None lengthens a path or moves its
//! ends, and every new edge is checked.
//!
//! Each path is its own search (see `edges`), so results do not depend on the batch or device.

use std::sync::Arc;
use std::time::Instant;

use crate::device::{Device, Worlds};
use crate::edges::{self, EdgeQuery, Keep, Search, Wanted};
use crate::error::{Result, ensure_input};
use crate::rrt::distance;

#[derive(Clone, Copy, Debug)]
pub struct ShortcutOptions {
    /// Spacing of collision checks along new edges (joint-space distance).
    pub resolution: f32,
    /// Passes of shortcutting and smoothing at most; they stop early once a pass changes nothing.
    pub passes: usize,
    /// Smoothing moves a waypoint, and straightening a joint shortens the path, only by at least
    /// this much (joint-space distance); infinity turns both off.
    pub min_change: f32,
    /// Straighten each joint in turn after the first pass's shortcuts (see the module docs): about
    /// 1% shorter paths on the Panda benchmark, and 13-18% on pick-and-place carries.
    pub straighten: bool,
    /// Clearance new edges keep from the world (meters), so the path does not pull taut against
    /// obstacles. It shrinks to zero within `clearance_ramp` (joint-space distance) of the path's
    /// ends, which may touch the world.
    pub clearance: f32,
    pub clearance_ramp: f32,
}

impl Default for ShortcutOptions {
    fn default() -> Self {
        Self {
            resolution: 1.0 / 32.0,
            passes: 3,
            min_change: 0.1,
            clearance: 0.01,
            clearance_ramp: 0.5,
            straighten: true,
        }
    }
}

/// Shortens and smooths each path (`[k, dof]` waypoints in world `world[i]`) in place. Endpoints
/// stay; the length never grows.
pub fn shortcut(
    device: &Device,
    worlds: &Worlds,
    world: &[u32],
    paths: &mut [Vec<f32>],
    o: &ShortcutOptions,
) -> Result<()> {
    shortcut_until(device, worlds, world, paths, o, None).map(|_| ())
}

/// [`shortcut`] that stops once `deadline` passes. Returns each path as the first pass's
/// shortcuts and straightened joints left it, before smoothing: few waypoints, which a spline can
/// trace.
pub(crate) fn shortcut_until(
    device: &Device,
    worlds: &Worlds,
    world: &[u32],
    paths: &mut [Vec<f32>],
    o: &ShortcutOptions,
    deadline: Option<Instant>,
) -> Result<Vec<Vec<f32>>> {
    let n = device.robot().dof();
    ensure_input!(o.resolution > 0.0 && o.min_change > 0.0, "the resolution and smallest change must be positive");
    ensure_input!(
        o.clearance >= 0.0 && o.clearance_ramp > 0.0,
        "the clearance must not be negative, and its ramp positive"
    );
    ensure_input!(world.len() == paths.len(), "{} worlds for {} paths", world.len(), paths.len());
    ensure_input!(paths.iter().all(|p| !p.is_empty() && p.len() % n == 0), "paths must hold whole waypoints");
    ensure_input!(world.iter().all(|&w| (w as usize) < worlds.len()), "a path references a world that was not given");
    let mut searches: Vec<Simplify> =
        paths.iter_mut().zip(world).map(|(path, &world)| Simplify::new(world, n, std::mem::take(path), o)).collect();
    let mut running: Vec<&mut dyn Search> = searches.iter_mut().map(|s| s as &mut dyn Search).collect();
    device.search(worlds, &mut running, deadline)?;
    let mut shortcut = Vec::with_capacity(paths.len());
    for (path, s) in paths.iter_mut().zip(searches) {
        let (simplified, short) = s.into_paths();
        *path = simplified;
        shortcut.push(short);
    }
    Ok(shortcut)
}

enum Step {
    /// Checking the edge between the ends.
    Straight,
    /// Checking shortcuts from waypoint `i`, the farthest first.
    Shortcut(usize),
    /// Checking joint `j` straightened (`Simplify::line`), from joint `j` on: first at the waypoints
    /// it moves and the middle of every edge, a batch of configurations that rules out most joints
    /// before their edges are checked.
    Joint(usize),
    /// Checking the edges of the path with joint `j` straightened.
    JointEdges(usize),
    /// Checking waypoint `v` (of the split path) moved to `Simplify::to`.
    Smooth(usize),
    Done,
}

/// One path's simplification.
pub(crate) struct Simplify {
    world: u32,
    n: usize,
    path: Vec<f32>,
    o: ShortcutOptions,
    pass: usize,
    changed: bool,
    step: Step,
    /// The path after the first pass's shortcuts and straightened joints.
    shortcut: Option<Vec<f32>>,
    keep: Option<Arc<Keep>>,
    /// Where smoothing would move the waypoint under check.
    to: Vec<f32>,
    /// The joint under check at each waypoint, straightened.
    line: Vec<f32>,
}

impl Simplify {
    pub(crate) fn new(world: u32, n: usize, path: Vec<f32>, o: &ShortcutOptions) -> Self {
        let ends = vec![path[..n].to_vec(), path[path.len() - n..].to_vec()];
        let keep = (o.clearance > 0.0).then(|| Arc::new(Keep { clearance: o.clearance, ramp: o.clearance_ramp, ends }));
        let (to, line) = (vec![0.0; n], Vec::with_capacity(path.len() / n));
        Self { world, n, path, o: *o, pass: 0, changed: false, step: Step::Straight, shortcut: None, keep, to, line }
    }

    /// The simplified path, and the path as the first pass's shortcuts left it.
    pub(crate) fn into_paths(self) -> (Vec<f32>, Vec<f32>) {
        let shortcut = self.shortcut.unwrap_or_else(|| self.path.clone());
        (self.path, shortcut)
    }

    fn waypoint(&self, i: usize) -> &[f32] {
        &self.path[i * self.n..(i + 1) * self.n]
    }

    fn count(&self) -> usize {
        self.path.len() / self.n
    }

    fn query(&self, edges: Vec<f32>, wanted: Wanted) -> Option<EdgeQuery> {
        Some(EdgeQuery {
            world: self.world,
            edges,
            wanted: Some(wanted),
            spacing: self.o.resolution,
            keep: self.keep.clone(),
        })
    }

    /// Splits every edge at its middle; the original waypoints land at even indices.
    fn split(&mut self) {
        let n = self.n;
        let mut out = Vec::with_capacity(2 * self.path.len());
        for (a, b) in self.path.chunks(n).zip(self.path.chunks(n).skip(1)) {
            out.extend_from_slice(a);
            out.extend(a.iter().zip(b).map(|(x, y)| 0.5 * (x + y)));
        }
        out.extend_from_slice(&self.path[self.path.len() - n..]);
        self.path = out;
    }

    /// Fills `line` with joint `j` changing at one rate along the path's length, from its value at
    /// the start to its value at the goal; whether that shortens the path by at least `min_change`.
    fn straighten(&mut self, j: usize) -> bool {
        let (n, last) = (self.n, self.count() - 1);
        let path = &self.path;
        let total = length(path, n);
        let (from, to) = (path[j], path[last * n + j]);
        self.line.clear();
        self.line.push(from);
        let (mut along, mut after) = (0.0, 0.0);
        for (k, (a, b)) in path.chunks(n).zip(path.chunks(n).skip(1)).enumerate() {
            along += distance(a, b);
            let x = if k + 1 == last { to } else { from + (to - from) * along / total };
            let y = self.line[k];
            let moved: f32 =
                (0..n).map(|i| if i == j { (x - y) * (x - y) } else { (b[i] - a[i]) * (b[i] - a[i]) }).sum();
            after += moved.sqrt();
            self.line.push(x);
        }
        total - after >= self.o.min_change
    }

    /// Ends a pass's shortcuts and straightened joints: smoothing next, if it is on, else the next
    /// pass.
    fn smooth(&mut self) {
        if self.shortcut.is_none() {
            self.shortcut = Some(self.path.clone());
        }
        if self.o.min_change.is_finite() {
            self.split();
            self.step = Step::Smooth(0);
        } else {
            self.next_pass();
        }
    }

    /// Ends a pass: another one if this one changed the path and passes remain.
    fn next_pass(&mut self) {
        self.pass += 1;
        self.step = if self.changed && self.pass < self.o.passes { Step::Shortcut(0) } else { Step::Done };
        self.changed = false;
    }
}

impl Search for Simplify {
    fn request(&mut self, buffer: &mut Vec<f32>) -> Option<EdgeQuery> {
        let n = self.n;
        loop {
            match &self.step {
                Step::Straight => {
                    if self.count() <= 2 {
                        self.step = Step::Done;
                        continue;
                    }
                    let mut edges = edges::reuse(buffer);
                    edges.extend_from_slice(self.waypoint(0));
                    edges.extend_from_slice(self.waypoint(self.count() - 1));
                    return self.query(edges, Wanted::Prefix);
                }
                &Step::Shortcut(i) => {
                    let last = self.count() - 1;
                    if i + 2 > last {
                        if self.pass == 0 && self.o.straighten {
                            self.step = Step::Joint(0);
                        } else {
                            self.smooth();
                        }
                        continue;
                    }
                    let mut edges = edges::reuse(buffer);
                    for j in (i + 2..=last).rev() {
                        edges.extend_from_slice(self.waypoint(i));
                        edges.extend_from_slice(self.waypoint(j));
                    }
                    return self.query(edges, Wanted::FirstValid);
                }
                &Step::Joint(j) => {
                    let Some(j) = (j..n).find(|&j| self.straighten(j)) else {
                        self.smooth();
                        continue;
                    };
                    // The moved waypoints, then the middle of every edge.
                    let mut moved = edges::reuse(buffer);
                    let last = self.count() - 1;
                    for w in 1..last {
                        let at = moved.len();
                        moved.extend_from_slice(self.waypoint(w));
                        moved[at + j] = self.line[w];
                    }
                    for w in 0..last {
                        let at = moved.len();
                        moved.extend(self.waypoint(w).iter().zip(self.waypoint(w + 1)).map(|(x, y)| 0.5 * (x + y)));
                        moved[at + j] = 0.5 * (self.line[w] + self.line[w + 1]);
                    }
                    self.step = Step::Joint(j);
                    let keep = self.keep.clone();
                    return Some(EdgeQuery { world: self.world, edges: moved, wanted: None, spacing: 0.0, keep });
                }
                &Step::JointEdges(j) => {
                    let mut edges = edges::reuse(buffer);
                    for k in 0..self.count() - 1 {
                        for w in [k, k + 1] {
                            let at = edges.len();
                            edges.extend_from_slice(self.waypoint(w));
                            edges[at + j] = self.line[w];
                        }
                    }
                    return self.query(edges, Wanted::Prefix);
                }
                &Step::Smooth(v) => {
                    // The next original interior waypoint that would move far enough.
                    let mut v = v + 2;
                    let found = loop {
                        if v + 1 >= self.count() {
                            break false;
                        }
                        let p = &self.path;
                        let (before, at, after) =
                            (&p[(v - 1) * n..v * n], &p[v * n..(v + 1) * n], &p[(v + 1) * n..(v + 2) * n]);
                        for (j, to) in self.to.iter_mut().enumerate() {
                            *to = 0.25 * before[j] + 0.5 * at[j] + 0.25 * after[j];
                        }
                        if distance(&self.to, at) >= self.o.min_change {
                            break true;
                        }
                        v += 2;
                    };
                    if !found {
                        self.next_pass();
                        continue;
                    }
                    let mut edges = edges::reuse(buffer);
                    edges.extend_from_slice(self.waypoint(v - 1));
                    edges.extend_from_slice(&self.to);
                    edges.extend_from_slice(&self.to);
                    edges.extend_from_slice(self.waypoint(v + 1));
                    self.step = Step::Smooth(v);
                    return self.query(edges, Wanted::Prefix);
                }
                Step::Done => return None,
            }
        }
    }

    fn answer(&mut self, answer: usize) {
        let n = self.n;
        match std::mem::replace(&mut self.step, Step::Done) {
            Step::Straight => {
                if answer == 1 {
                    let end = self.path.len() - n;
                    self.path.drain(n..end);
                } else {
                    self.step = Step::Shortcut(0);
                }
            }
            Step::Shortcut(i) => {
                let last = self.count() - 1;
                // Answer k is the edge to waypoint last - k; the first valid one is the farthest.
                if answer <= last - i - 2 {
                    let j = last - answer;
                    self.path.drain((i + 1) * n..j * n);
                    self.changed = true;
                }
                self.step = Step::Shortcut(i + 1);
            }
            Step::Joint(j) => {
                let checked = 2 * self.count() - 3;
                self.step = if answer == checked { Step::JointEdges(j) } else { Step::Joint(j + 1) };
            }
            Step::JointEdges(j) => {
                let last = self.count() - 1;
                if answer == last {
                    for w in 1..last {
                        self.path[w * n + j] = self.line[w];
                    }
                    self.changed = true;
                }
                self.step = Step::Joint(j + 1);
            }
            Step::Smooth(v) => {
                if answer == 2 {
                    self.path[v * n..(v + 1) * n].copy_from_slice(&self.to);
                    self.changed = true;
                }
                self.step = Step::Smooth(v);
            }
            Step::Done => unreachable!("answers follow requests"),
        }
    }
}

pub(crate) fn length(path: &[f32], n: usize) -> f32 {
    path.chunks(n).zip(path.chunks(n).skip(1)).map(|(a, b)| distance(a, b)).sum()
}

/// The point `along` the path (by joint-space distance), into `out`.
pub(crate) fn locate(path: &[f32], n: usize, along: f32, out: &mut [f32]) {
    let mut walked = 0.0;
    let edges = path.len() / n - 1;
    for e in 0..edges {
        let (a, b) = (&path[e * n..(e + 1) * n], &path[(e + 1) * n..(e + 2) * n]);
        let d = distance(a, b);
        if walked + d >= along || e + 1 == edges {
            let t = if d > 0.0 { ((along - walked) / d).clamp(0.0, 1.0) } else { 0.0 };
            for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
                *o = x + (y - x) * t;
            }
            return;
        }
        walked += d;
    }
    unreachable!("paths have at least one edge")
}
