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
    /// Damped-least-squares iterations per step at most: 30. Each configuration is held to a
    /// tenth of the precisions.
    pub iterations: u32,
}

impl Default for ToolPathOptions {
    fn default() -> Self {
        Self {
            step: 0.005,
            rotation_step: 0.05,
            precision: 0.001,
            rotation_precision: 0.01,
            resolution: 1.0 / 32.0,
            iterations: 30,
        }
    }
}

/// What [`follow`] did with each path.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ToolPathResult {
    /// The paths, in order.
    pub problems: Vec<ToolPath>,
    /// Per path: how much of it the motion covers, from 0 to 1, in steps: each stretch between
    /// poses counts its length over `step` or its turn over `rotation_step`, whichever is more. A
    /// path that does not move the hand is covered.
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
        "steps and precisions must be positive, got {} m, {} rad, {} m and {} rad",
        o.step,
        o.rotation_step,
        o.precision,
        o.rotation_precision
    );
    ensure_input!(
        o.resolution > 0.0,
        "the resolution must be positive, got {}",
        o.resolution
    );
    ensure_input!(o.iterations > 0, "the IK needs at least one iteration");
    for (i, p) in paths.iter().enumerate() {
        ensure_input!(
            p.start.len() == n,
            "path {i}: a start of {} joints, not {n}",
            p.start.len()
        );
        ensure_input!(
            robot.within(&p.start),
            "path {i}: the start {:?} leaves the joint limits",
            p.start
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
        result.fraction.push(done.covered);
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

/// The share of a step's motion, in precisions, below which its turn does not place a sample
/// along it, which saves an arccosine per sample of a step that only moves.
const TURN_SHARE: f32 = 1e-3;

/// The share of the precisions each configuration is held to by its IK.
const TOLERANCE: f32 = 0.1;

/// How far past a joint limit a control point may lie, for rounding: well inside the slack
/// `Trajectory::check` allows.
const LIMIT_SLACK: f32 = 1e-6;

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

    /// How far `pose` strays from the path between `self` and `to`, from the point of it `pose`
    /// is nearest and from the path's rotation there (meters, radians). Nearest weighs position
    /// and rotation each by its precision, so a step that mostly turns is placed by its turn.
    fn deviation(&self, to: &Point, pose: &Pose, o: &ToolPathOptions) -> [f32; 2] {
        let span = to.position - self.position;
        let turn = self.rotation.angle_between(to.rotation);
        // How far along the step `pose` lies by its position and by its turn, each weighed by
        // how many precisions the step moves it; a turn that weighs next to nothing is not read.
        let (moved, turned) = (
            span.length_squared() / (o.precision * o.precision),
            (turn / o.rotation_precision).powi(2),
        );
        let share = if moved + turned > 0.0 {
            turned / (moved + turned)
        } else {
            0.0
        };
        let along_span = if span.length_squared() > 1e-12 {
            (pose.position - self.position).dot(span) / span.length_squared()
        } else {
            0.0
        };
        let along_turn = if share > TURN_SHARE && turn > 1e-6 {
            self.rotation.angle_between(pose.rotation) / turn
        } else {
            along_span
        };
        let t = along_span + share * (along_turn - along_span);
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
    /// Each step's IK, holding configurations to a tenth of the precisions.
    ik: IkOptions,
    /// Each joint's range ([`Robot::bounds`]), which the spline's control points keep.
    bounds: Vec<(f32, f32)>,
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
    /// How much of the path the motion covers, once the search is done (`fraction`).
    covered: f32,
    /// Working memory, reused by every query: a configuration being stepped to and one between
    /// two, whether each control point lies past a joint limit, the poses still to reach before
    /// the next point, the pinned configurations, and the fit's solves.
    next: Vec<f32>,
    between: Vec<f32>,
    outside: Vec<bool>,
    pending: Vec<(Point, u32)>,
    pins: Vec<usize>,
    rhs: Vec<f32>,
    scratch: Vec<f32>,
    solved: Vec<f32>,
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
            ik: IkOptions {
                iterations: o.iterations,
                position_tolerance: o.precision * TOLERANCE,
                rotation_tolerance: o.rotation_precision * TOLERANCE,
                ..IkOptions::default()
            },
            bounds: (0..robot.dof()).map(|j| robot.bounds(j)).collect(),
            q: path.start.clone(),
            reached: vec![],
            pinned: vec![],
            control: vec![],
            spans: vec![],
            starts: vec![],
            stop: None,
            stage: Stage::Steps,
            covered: 0.0,
            next: path.start.clone(),
            between: vec![0.0; robot.dof()],
            outside: vec![],
            pending: vec![],
            pins: vec![],
            rhs: vec![],
            scratch: vec![],
            solved: vec![],
        }
    }

    /// Ends the search. It keeps what `follow` reads and frees the rest of its memory here, on the
    /// thread that allocated it: freed later by the caller, a batch's buffers all return to one
    /// thread, and glibc frees memory from another thread's arena slowly.
    fn finish(&mut self) {
        self.covered = self.fraction();
        self.stage = Stage::Done;
        (self.points, self.reached, self.pending) = Default::default();
        (
            self.q,
            self.next,
            self.between,
            self.rhs,
            self.scratch,
            self.solved,
        ) = Default::default();
        (
            self.pinned,
            self.outside,
            self.pins,
            self.spans,
            self.starts,
        ) = Default::default();
        self.bounds = vec![];
    }

    fn fraction(&self) -> f32 {
        if self.points.len() < 2 {
            return 1.0;
        }
        self.reached.last().map_or(0.0, |p| p.along)
    }

    /// Steps from the start along every point, halving a step where the joints' motion would take
    /// the hand off the path, until the path ends or a pose cannot be reached.
    fn walk(&mut self) {
        let n = self.n;
        self.reached.push(self.points[0]);
        self.pinned.push(true);
        for p in 1..self.points.len() {
            // Poses still to reach before this point is, the next one last, with how often their
            // step was halved.
            self.pending.clear();
            self.pending.push((self.points[p], 0));
            while let Some(&(to, splits)) = self.pending.last() {
                let k = self.reached.len() - 1;
                self.next.copy_from_slice(&self.q[k * n..(k + 1) * n]);
                let reached = self.reach.reach(
                    self.robot,
                    self.hand,
                    &to.target(),
                    &mut self.next,
                    &self.ik,
                );
                if !reached {
                    self.stop = Some(ToolPathStop::Unreachable { at: to.pose() });
                    return;
                }
                let from = self.reached[k];
                let at = &self.q[k * n..(k + 1) * n];
                for ((b, a), z) in self.between.iter_mut().zip(at).zip(&self.next) {
                    *b = 0.5 * (a + z);
                }
                let halfway = Point::between(&from, &to, 0.5);
                let [position, rotation] =
                    self.reach
                        .error(self.robot, self.hand, &halfway.target(), &self.between);
                let on_path = position <= self.o.precision && rotation <= self.o.rotation_precision;
                if on_path {
                    self.pending.pop();
                    self.q.extend_from_slice(&self.next);
                    self.reached.push(to);
                    self.pinned.push(false);
                } else if splits == MAX_SPLITS {
                    self.stop = Some(ToolPathStop::Jump { at: to.pose() });
                    return;
                } else {
                    let top = self.pending.len() - 1;
                    self.pending[top].1 = splits + 1;
                    self.pending.push((halfway, splits + 1));
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
    /// path's precision and the joint limits: a span that strays or leaves them pins the nearer free
    /// configuration it runs between, and the fit is redone.
    fn fit(&mut self) {
        let n = self.n;
        if self.reached.len() == 2 {
            // One step: its middle as a configuration of its own, for a spline of seven points.
            for ((b, x), y) in self
                .between
                .iter_mut()
                .zip(&self.q[..n])
                .zip(&self.q[n..2 * n])
            {
                *b = 0.5 * (x + y);
            }
            let point = Point::between(&self.reached[0], &self.reached[1], 0.5);
            self.q.splice(n..n, self.between.iter().copied());
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
        self.pins.clear();
        let pinned = (0..self.reached.len()).filter(|&k| self.pinned[k]);
        self.pins.extend(pinned);
        for pair in self.pins.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let r = b - a - 1;
            self.solved.clear();
            self.solved.resize(r * n, 0.0);
            for j in 0..n {
                self.rhs.clear();
                let between = (a + 1..b).map(|k| 6.0 * self.q[k * n + j]);
                self.rhs.extend(between);
                if r > 0 {
                    self.rhs[0] -= self.q[a * n + j];
                    self.rhs[r - 1] -= self.q[b * n + j];
                }
                solve_one_four_one(&mut self.rhs, &mut self.scratch);
                for (i, x) in self.rhs.iter().enumerate() {
                    self.solved[i * n + j] = *x;
                }
            }
            self.control.extend_from_slice(&self.solved);
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

    /// The configuration to pin, if some span of the spline strays from the path or has a control
    /// point past a joint limit: the nearer of the two configurations it runs between that is not
    /// pinned yet. A span's curve stays within its control points, so one whose points keep the
    /// limits keeps them along its whole length.
    fn stray(&mut self) -> Option<usize> {
        let n = self.n;
        // Each control point checked once, then each span by its four.
        self.outside.clear();
        let past = |c: &[f32]| {
            c.iter()
                .zip(&self.bounds)
                .any(|(&v, &(lo, hi))| v < lo - LIMIT_SLACK || v > hi + LIMIT_SLACK)
        };
        self.outside.extend(self.control.chunks(n).map(past));
        for (span, &k) in self.spans.iter().enumerate() {
            let outside = self.outside[span..span + 4].contains(&true);
            if outside && let Some(pin) = [k + 1, k].into_iter().find(|&e| !self.pinned[e]) {
                return Some(pin);
            }
            for s in 1..PRECISION_SAMPLES {
                let u = s as f32 / PRECISION_SAMPLES as f32;
                spline::blend(&self.control, n, span, spline::basis(u), &mut self.between);
                let pose = self.reach.pose(self.robot, self.hand, &self.between);
                let [position, rotation] =
                    self.reached[k].deviation(&self.reached[k + 1], &pose, self.o);
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
            self.finish();
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
            self.finish();
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
