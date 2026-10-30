//! Moving a hand along a set path: straight lines between poses and the shortest rotation between
//! their rotations, followed from a start configuration by inverse kinematics a few millimeters at
//! a time, as MoveIt's Cartesian paths are. Each path is a search (`edges`): it steps along on one
//! of the device's threads, fits a spline through its steps, and asks for the spline to be checked
//! for collision, cutting it back to where it collides.

use glam::{Mat3, Quat, Vec3};

use crate::cpu::Reach;
use crate::device::{Device, Worlds};
use crate::edges::{self, Deferred, EdgeQuery, Search};
use crate::error::{Result, ensure_input};
use crate::ik::IkOptions;
use crate::robot::Robot;
use crate::spline;
use crate::types::Pose;

/// A path for one hand to follow from a start configuration.
#[derive(Clone, Debug)]
pub struct ToolPath {
    /// Index of its world in the uploaded batch.
    pub world: u32,
    /// Where the robot starts; the path starts at the hand's pose there.
    pub start: Vec<f32>,
    /// The hand that follows the path. Only the joints that move it move.
    pub hand: usize,
    /// Poses the hand passes through after its start, in the robot's frame.
    pub waypoints: Vec<Pose>,
}

/// How [`follow`] tracks a path.
#[derive(Clone, Debug)]
pub struct ToolPathOptions {
    /// Largest step of the hand between configurations (meters): 5 mm.
    pub step: f32,
    /// Largest turn of the hand between configurations (radians): 0.05.
    pub rotation_step: f32,
    /// How far the hand may stray from the path (meters): 1 mm.
    pub precision: f32,
    /// How far its rotation may stray from the path's (radians): 0.01.
    pub rotation_precision: f32,
    /// Joint-space spacing of collision checks, as [`RrtOptions::resolution`](crate::RrtOptions).
    pub resolution: f32,
    /// The IK that takes each step. Its tolerances hold each configuration to its pose.
    pub ik: IkOptions,
}

impl Default for ToolPathOptions {
    fn default() -> Self {
        Self {
            step: 0.005,
            rotation_step: 0.05,
            precision: 0.001,
            rotation_precision: 0.01,
            resolution: 1.0 / 32.0,
            ik: IkOptions {
                iterations: 30,
                position_tolerance: 1e-4,
                rotation_tolerance: 1e-3,
                ..IkOptions::default()
            },
        }
    }
}

/// What [`follow`] did with each path.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ToolPathResult {
    /// The paths, in order.
    pub problems: Vec<ToolPath>,
    /// Per path: how much of it the motion covers, from 0 to 1, by its steps: its length, or its
    /// turning where the hand only turns. A path that does not move the hand is covered.
    pub fraction: Vec<f32>,
    /// Per path: why the motion stops short, where it does.
    pub stopped: Vec<Option<ToolPathStop>>,
    /// Per path: the motion from the start to the last pose it reaches, as B-spline control
    /// points (`[points, dof]`, three equal at each end) for
    /// [`Trajectory::try_new`](crate::Trajectory::try_new). It stops only where it must to keep
    /// the precision, as at a sharp turn. Empty where it reaches nothing past the start.
    pub paths: Vec<Vec<f32>>,
}

/// Why a tool path's motion stops short.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ToolPathStop {
    /// No configuration near the last one reaches this pose: it lies beyond the arm's reach or
    /// its joint limits, or at a singularity the arm cannot step through.
    Unreachable {
        /// The pose it stops before.
        at: Pose,
    },
    /// The hand's next step collides.
    Collision {
        /// The pose it stops before.
        at: Pose,
    },
    /// Configurations reach the poses on either side of this one, but the joints would jump
    /// between them, as a wrist spins half a turn at a singularity: no motion between them keeps
    /// the hand within the precision of the path.
    Jump {
        /// The pose it stops before.
        at: Pose,
    },
}

/// Follows each path from its start as far as it goes ([`ToolPathResult`]), its steps checked for
/// collision in its world. Paths run in parallel on the device's threads.
pub fn follow(
    device: &Device,
    worlds: &Worlds,
    paths: &[ToolPath],
    o: &ToolPathOptions,
) -> Result<ToolPathResult> {
    let robot = device.robot();
    let n = robot.dof();
    ensure_input!(
        o.step > 0.0 && o.rotation_step > 0.0 && o.precision > 0.0 && o.rotation_precision > 0.0,
        "steps and precisions must be positive"
    );
    ensure_input!(o.resolution > 0.0, "the resolution must be positive");
    ensure_input!(
        o.ik.rotation_weight > 0.0
            && o.ik.position_tolerance > 0.0
            && o.ik.rotation_tolerance > 0.0,
        "the IK's rotation weight and tolerances must be positive"
    );
    for (i, p) in paths.iter().enumerate() {
        ensure_input!(
            p.start.len() == n,
            "path {i}: a start of {} joints, not {n}",
            p.start.len()
        );
        ensure_input!(
            p.hand < robot.hands(),
            "path {i}: hand {} of {}",
            p.hand,
            robot.hands()
        );
        ensure_input!(
            (p.world as usize) < worlds.len(),
            "path {i}: world {} of {}",
            p.world,
            worlds.len()
        );
        ensure_input!(!p.waypoints.is_empty(), "path {i} has no waypoints");
        let finite = |w: &Pose| w.position.is_finite() && w.rotation.is_finite();
        ensure_input!(
            p.waypoints.iter().all(finite) && p.start.iter().all(|v| v.is_finite()),
            "path {i} is not finite"
        );
    }
    let mut searches: Vec<_> = paths
        .iter()
        .map(|path| Deferred::new(None, move || Follow::new(robot, path, o)))
        .collect();
    let mut running: Vec<&mut dyn Search> =
        searches.iter_mut().map(|s| s as &mut dyn Search).collect();
    device.search(worlds, &mut running, None)?;
    let mut result = ToolPathResult {
        problems: paths.to_vec(),
        fraction: Vec::with_capacity(paths.len()),
        stopped: Vec::with_capacity(paths.len()),
        paths: Vec::with_capacity(paths.len()),
    };
    for search in searches {
        let done = search.into_inner();
        result.fraction.push(done.fraction());
        result.stopped.push(done.stop);
        result.paths.push(done.control);
    }
    Ok(result)
}

/// Halvings of a step, at most, before the joints' motion between configurations is called a jump:
/// the hand then strays from the path over less than a 256th of a step.
const MAX_SPLITS: u32 = 8;

/// Spline samples per span checked against the path's precision, ends excluded.
const PRECISION_SAMPLES: usize = 4;

/// Spline samples per span for its collision check, at least.
const CHECK_SAMPLES: usize = 4;

/// A pose along the path and how far along it lies, from 0 to 1.
#[derive(Clone, Copy, Debug)]
struct Point {
    position: Vec3,
    rotation: Quat,
    along: f32,
}

impl Point {
    fn at(pose: Pose, along: f32) -> Point {
        Point {
            position: pose.position,
            rotation: pose.rotation,
            along,
        }
    }

    fn between(a: &Point, b: &Point, t: f32) -> Point {
        Point {
            position: a.position.lerp(b.position, t),
            rotation: a.rotation.slerp(b.rotation, t),
            along: a.along + (b.along - a.along) * t,
        }
    }

    fn target(&self) -> (Vec3, Mat3) {
        (self.position, Mat3::from_quat(self.rotation))
    }

    fn pose(&self) -> Pose {
        Pose::new(self.position, self.rotation)
    }

    /// How far `pose` strays from the path between `self` and `to`: from the nearest point of the
    /// straight line between them, and from the path's rotation there (meters, radians).
    fn deviation(&self, to: &Point, pose: &Pose) -> [f32; 2] {
        let span = to.position - self.position;
        let turn = self.rotation.angle_between(to.rotation);
        let t = if span.length_squared() > 1e-12 {
            (pose.position - self.position).dot(span) / span.length_squared()
        } else if turn > 1e-6 {
            self.rotation.angle_between(pose.rotation) / turn
        } else {
            0.0
        };
        let on = Point::between(self, to, t.clamp(0.0, 1.0));
        [
            on.position.distance(pose.position),
            on.rotation.angle_between(pose.rotation),
        ]
    }
}

/// What a path's search does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Step along the path.
    Steps,
    /// Fit the spline through the steps and ask for it to be checked.
    Spline,
    /// Wait for the spline's answer.
    Checking,
    Done,
}

/// One path's search.
struct Follow<'a> {
    robot: &'a Robot,
    o: &'a ToolPathOptions,
    world: u32,
    hand: usize,
    n: usize,
    /// The path's poses a step apart, the start's first.
    points: Vec<Point>,
    reach: Reach,
    /// Configurations along the path (`[k, dof]`), the start's first, the point each reaches, and
    /// whether the spline stops there.
    q: Vec<f32>,
    reached: Vec<Point>,
    pinned: Vec<bool>,
    /// The spline's control points, and for each of its spans the configuration it starts from:
    /// it runs from there to the next.
    control: Vec<f32>,
    spans: Vec<usize>,
    /// Where each span's samples start in the last spline check, then the end's.
    starts: Vec<usize>,
    stop: Option<ToolPathStop>,
    stage: Stage,
}

impl<'a> Follow<'a> {
    fn new(robot: &'a Robot, path: &ToolPath, o: &'a ToolPathOptions) -> Self {
        let points = points(robot.ee_pose(&path.start, path.hand), &path.waypoints, o);
        Self {
            robot,
            o,
            world: path.world,
            hand: path.hand,
            n: robot.dof(),
            points,
            reach: Reach::new(robot),
            q: path.start.clone(),
            reached: vec![],
            pinned: vec![],
            control: vec![],
            spans: vec![],
            starts: vec![],
            stop: None,
            stage: Stage::Steps,
        }
    }

    fn fraction(&self) -> f32 {
        if self.points.len() < 2 {
            return 1.0;
        }
        self.reached.last().map_or(0.0, |p| p.along)
    }

    fn config(&self, k: usize) -> &[f32] {
        &self.q[k * self.n..(k + 1) * self.n]
    }

    /// Steps from the start along every point, halving a step where the joints' motion would take
    /// the hand off the path, until the path ends or a pose cannot be reached.
    fn walk(&mut self) {
        self.reached.push(self.points[0]);
        self.pinned.push(true);
        let mut next = self.q.clone();
        let mut middle = vec![0.0; self.n];
        for p in 1..self.points.len() {
            // Poses still to reach before this point is, the next one last, with how often their
            // step was halved.
            let mut pending = vec![(self.points[p], 0)];
            while let Some(&(to, splits)) = pending.last() {
                let k = self.reached.len() - 1;
                next.copy_from_slice(self.config(k));
                let reached =
                    self.reach
                        .reach(self.robot, self.hand, &to.target(), &mut next, &self.o.ik);
                if !reached {
                    self.stop = Some(ToolPathStop::Unreachable { at: to.pose() });
                    return;
                }
                let from = self.reached[k];
                for ((m, a), b) in middle.iter_mut().zip(self.config(k)).zip(&next) {
                    *m = 0.5 * (a + b);
                }
                let halfway = Point::between(&from, &to, 0.5);
                let [position, rotation] =
                    self.reach
                        .error(self.robot, self.hand, &halfway.target(), &middle);
                if position <= self.o.precision && rotation <= self.o.rotation_precision {
                    pending.pop();
                    self.q.extend_from_slice(&next);
                    self.reached.push(to);
                    self.pinned.push(false);
                } else if splits == MAX_SPLITS {
                    self.stop = Some(ToolPathStop::Jump { at: to.pose() });
                    return;
                } else {
                    let top = pending.len() - 1;
                    pending[top].1 = splits + 1;
                    pending.push((halfway, splits + 1));
                }
            }
        }
    }

    /// Keeps the first `k` configurations, the next of which was found to collide.
    fn truncate(&mut self, k: usize) {
        self.stop = Some(ToolPathStop::Collision {
            at: self.reached[k].pose(),
        });
        self.q.truncate(k * self.n);
        self.reached.truncate(k);
        self.pinned.truncate(k);
        if let Some(last) = self.pinned.last_mut() {
            *last = true;
        }
    }

    /// The spline through every configuration (`control`), resting at the pinned ones, held to the
    /// path's precision: a span that strays pins the nearer free configuration it runs between,
    /// and the fit is redone.
    fn fit(&mut self) {
        if self.reached.len() == 2 {
            // One step: its middle as a configuration of its own, for a spline of seven points.
            let middle: Vec<f32> = self
                .config(0)
                .iter()
                .zip(self.config(1))
                .map(|(a, b)| 0.5 * (a + b))
                .collect();
            let point = Point::between(&self.reached[0], &self.reached[1], 0.5);
            self.q.splice(self.n..self.n, middle);
            self.reached.insert(1, point);
            self.pinned.insert(1, false);
        }
        loop {
            self.interpolate();
            match self.stray() {
                Some(k) => self.pinned[k] = true,
                None => return,
            }
        }
    }

    /// Control points through every configuration: between consecutive pinned ones `a` and `b`,
    /// `[a, a, a, P.., b, b, b]` with `P` solving `P[i-1] + 4 P[i] + P[i+1] = 6 q[i]` over the
    /// configurations between them (`P[-1] = a`, `P[r] = b`), so the spline passes through each.
    fn interpolate(&mut self) {
        let n = self.n;
        self.control.clear();
        self.spans.clear();
        for _ in 0..3 {
            self.control.extend_from_slice(&self.q[..n]);
        }
        let pins: Vec<usize> = (0..self.reached.len())
            .filter(|&k| self.pinned[k])
            .collect();
        let (mut rhs, mut scratch, mut solved) = (vec![], vec![], vec![]);
        for pair in pins.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let r = b - a - 1;
            solved.clear();
            solved.resize(r * n, 0.0);
            for j in 0..n {
                rhs.clear();
                rhs.extend((a + 1..b).map(|k| 6.0 * self.q[k * n + j]));
                if r > 0 {
                    rhs[0] -= self.q[a * n + j];
                    rhs[r - 1] -= self.q[b * n + j];
                }
                solve_one_four_one(&mut rhs, &mut scratch);
                for (i, x) in rhs.iter().enumerate() {
                    solved[i * n + j] = *x;
                }
            }
            self.control.extend_from_slice(&solved);
            for _ in 0..3 {
                self.control.extend_from_slice(&self.q[b * n..(b + 1) * n]);
            }
            // The stretch's `r + 3` spans: two on the way to the first configuration after `a`,
            // one from each configuration after that to the next, two on the way into `b`.
            if r == 0 {
                self.spans.extend([a; 3]);
            } else {
                self.spans.extend([a, a]);
                self.spans.extend(a + 1..b - 1);
                self.spans.extend([b - 1, b - 1]);
            }
        }
    }

    /// The configuration to pin, if some span of the spline strays from the path: the nearer of
    /// the two it runs between that is not pinned yet.
    fn stray(&mut self) -> Option<usize> {
        let n = self.n;
        let mut q = vec![0.0; n];
        for (span, &k) in self.spans.iter().enumerate() {
            for s in 1..PRECISION_SAMPLES {
                let u = s as f32 / PRECISION_SAMPLES as f32;
                spline::blend(&self.control, n, span, spline::basis(u), &mut q);
                let pose = self.reach.pose(self.robot, self.hand, &q);
                let [position, rotation] = self.reached[k].deviation(&self.reached[k + 1], &pose);
                if position <= self.o.precision && rotation <= self.o.rotation_precision {
                    continue;
                }
                let ends = if u <= 0.5 { [k, k + 1] } else { [k + 1, k] };
                if let Some(pin) = ends.into_iter().find(|&e| !self.pinned[e]) {
                    return Some(pin);
                }
            }
        }
        None
    }
}

/// Solves `x[i-1] + 4 x[i] + x[i+1] = d[i]` (`x[-1] = x[r] = 0`) for `x` in place of `d`, by
/// Thomas's algorithm; `scratch` is working memory. The system is diagonally dominant, so this is
/// stable.
fn solve_one_four_one(d: &mut [f32], scratch: &mut Vec<f32>) {
    let r = d.len();
    scratch.clear();
    scratch.resize(r, 0.0);
    // Forward: each row's multiplier on the next unknown, and its right-hand side, after
    // elimination.
    let mut c = 0.0;
    for i in 0..r {
        let m = 4.0 - c;
        c = 1.0 / m;
        scratch[i] = c;
        d[i] = (d[i] - if i > 0 { d[i - 1] } else { 0.0 }) / m;
    }
    for i in (0..r.saturating_sub(1)).rev() {
        d[i] -= scratch[i] * d[i + 1];
    }
}

/// The path's poses a step apart, from `start` through each waypoint. A path that does not move the
/// hand is its start alone.
fn points(start: Pose, waypoints: &[Pose], o: &ToolPathOptions) -> Vec<Point> {
    let poses: Vec<Pose> = std::iter::once(start)
        .chain(waypoints.iter().copied())
        .collect();
    let steps = |a: &Pose, b: &Pose| {
        let turn = a.rotation.angle_between(b.rotation) / o.rotation_step;
        (a.position.distance(b.position) / o.step).max(turn)
    };
    let total: f32 = poses.windows(2).map(|w| steps(&w[0], &w[1])).sum();
    let mut points = vec![Point::at(start, 0.0)];
    if total == 0.0 {
        return points;
    }
    let mut along = 0.0;
    for pair in poses.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let length = steps(&a, &b);
        if length == 0.0 {
            continue;
        }
        let from = Point::at(a, along / total);
        along += length;
        let to = Point::at(b, along / total);
        let count = length.ceil() as usize;
        points.extend((1..=count).map(|i| Point::between(&from, &to, i as f32 / count as f32)));
    }
    points
}

impl Search for Follow<'_> {
    fn request(&mut self, buffer: &mut Vec<f32>) -> Option<EdgeQuery> {
        if self.stage == Stage::Steps {
            if self.points.len() > 1 {
                self.walk();
            }
            // The motion rests wherever it ends, short of the path's end too.
            if let Some(last) = self.pinned.last_mut() {
                *last = true;
            }
            self.stage = Stage::Spline;
        }
        if self.stage != Stage::Spline {
            return None;
        }
        if self.reached.len() < 2 {
            self.control.clear();
            self.stage = Stage::Done;
            return None;
        }
        self.fit();
        let mut samples = edges::reuse(buffer);
        spline::checked(
            &self.control,
            self.n,
            CHECK_SAMPLES,
            self.o.resolution,
            &mut samples,
            &mut self.starts,
        );
        self.stage = Stage::Checking;
        Some(EdgeQuery {
            world: self.world,
            edges: samples,
            wanted: None,
            spacing: 0.0,
            keep: None,
        })
    }

    fn answer(&mut self, answer: usize) {
        let samples = *self.starts.last().expect("a spline has an end sample") + 1;
        if answer >= samples {
            self.stage = Stage::Done;
            return;
        }
        // The configurations before the span of the first sample that may collide.
        let span = self
            .starts
            .partition_point(|&s| s <= answer)
            .saturating_sub(1);
        self.truncate(self.spans[span.min(self.spans.len() - 1)] + 1);
        self.stage = Stage::Spline;
    }
}
