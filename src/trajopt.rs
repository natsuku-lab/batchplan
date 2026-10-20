//! Batched motion planning. Each problem is searched first: the straight line to its goal, else
//! RRT-Connect toward every equivalent goal, then shortcut and smoothed (see `rrt` and
//! `shortcut`). The path becomes a uniform cubic B-spline whose first three and last three control
//! points are pinned to the start and goal, so it starts and ends at rest: the control points
//! spread along the path by arc length (smooth, cutting its corners) or, where that is not
//! collision-free, traced through the corners nearest the collisions, then eased off each where a
//! blend around it stays collision-free, so it stops only at corners no blend clears. Optionally,
//! trajectory optimization then polishes it and
//! further seeds: L-BFGS on smoothness plus collision cost sampled along the curve, each step
//! pricing four fractions of its direction at once and taking the cheapest. Every candidate is
//! validated by sampling its curve densely; the shortest valid one is the result.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{Result, ensure_input};

use crate::device::{CollisionWeights, Device, Worlds};
use crate::edges::{self, EdgeQuery, Keep, Optimize, Search};
use crate::rng::Rng;
use crate::robot::Robot;
use crate::rrt::{Ahead, Connect, RrtOptions, RrtProblem, Sampling, distance};
use crate::shortcut::{ShortcutOptions, Simplify, length, locate};
use crate::spline;
use crate::timing::Trajectory;
use crate::types::{JointPaths, Solved, StartMotion};

#[derive(Clone, Copy, Debug)]
pub struct PlanOptions {
    /// Seeds trajectory optimization runs per problem, besides the searched path: its spline, the
    /// straight lines to the goal and its equivalents, then bends through random via points.
    /// Problems searched to within 2% of the straight line are not optimized (their seeds stay
    /// invalid). 0 keeps the searched path as it is.
    pub seeds: usize,
    /// B-spline control points per path, including the three pinned at each end (at least 7).
    pub control_points: usize,
    /// Collision samples per span during optimization.
    pub samples_per_span: usize,
    /// L-BFGS steps of trajectory optimization.
    pub iterations: u32,
    /// Steps L-BFGS remembers (1 to [`MAX_HISTORY`]).
    pub history: usize,
    /// Largest joint change of a step without history: the first, and any after a reset.
    pub initial_step: f32,
    /// Weight of squared second differences of control points.
    pub w_acc: f32,
    /// Weight of squared differences of control points (path length).
    pub w_vel: f32,
    /// Collision cost of the optimization. Repairs of stuck splines keep their own weights and a
    /// 2 cm margin.
    pub collision: CollisionWeights,
    /// Samples per span used to validate the result.
    pub validate_substeps: usize,
    pub rng_seed: u64,
    /// Searching for each path: RRT-Connect after the straight line.
    pub rrt: RrtOptions,
    /// Shortening and smoothing the path found.
    pub shortcut: ShortcutOptions,
    /// Stop optimizing and searching once this much time has passed, and return the best valid
    /// paths so far. The budget is checked between rounds (on a GPU, between submissions of
    /// several rounds), and validation still runs after it; optimization does not start once it
    /// has passed. Results then depend on machine speed; without a budget they are the same on
    /// every run.
    pub time_budget: Option<Duration>,
}

/// Fractions of the L-BFGS direction each line search tries, in order of preference on ties.
pub(crate) const LINE_SEARCH: [f32; 4] = [1.0, 0.5, 0.25, 0.1];
/// The most steps L-BFGS can remember.
pub const MAX_HISTORY: usize = 8;

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            seeds: 0,
            control_points: 24,
            samples_per_span: 3,
            iterations: 100,
            history: 6,
            initial_step: 0.1,
            w_acc: 5.0,
            w_vel: 10.0,
            collision: CollisionWeights { world: 1000.0, self_collision: 1000.0, margin: 0.012, self_margin: 0.01 },
            validate_substeps: 8,
            rng_seed: 2,
            rrt: RrtOptions { clearance: 0.015, ..RrtOptions::default() },
            shortcut: ShortcutOptions::default(),
            time_budget: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlanProblem {
    pub world: u32,
    pub start: Vec<f32>,
    /// Where the path ends, up to whole turns: continuous joints turn the short way (or the long
    /// way, if that is the one that is free), and a joint whose range spans more than a turn may
    /// end a turn away from this value when that is nearer or free. Every such end is the same pose.
    pub goal: Vec<f32>,
    /// How the robot is already moving at `start`, to replan mid-motion; `None` starts at rest.
    /// The path's first three control points then continue that motion, and
    /// [`crate::Trajectory::moving`] times it.
    pub start_motion: Option<StartMotion>,
    /// Joints that stay where `start` has them, as an arm that waits while another moves
    /// (`Robot::hand_joints` lists an arm's); `goal` has the same values there. Searching,
    /// shortening and optimizing all leave them still.
    pub hold: Vec<usize>,
}

impl PlanProblem {
    /// From `start` at rest to `goal` in world `world`, every joint free to move.
    pub fn new(world: u32, start: impl Into<Vec<f32>>, goal: impl Into<Vec<f32>>) -> Self {
        Self { world, start: start.into(), goal: goal.into(), start_motion: None, hold: Vec::new() }
    }
}

/// Per-seed results for `problems`; paths are problem-major (`item = problem * seeds + seed`).
/// Seed 0 is the searched path; optimized seeds follow, invalid where a problem was not optimized.
#[derive(Clone, Debug)]
pub struct PlanResult {
    pub problems: Vec<PlanProblem>,
    pub seeds: usize,
    /// B-spline control points of every seed's path.
    pub paths: JointPaths,
    pub valid: Vec<bool>,
    /// Joint-space path length.
    pub length: Vec<f32>,
}

impl PlanResult {
    /// The shortest valid seed.
    pub fn best(&self, problem: usize) -> Option<&[f32]> {
        (problem * self.seeds..(problem + 1) * self.seeds)
            .filter(|&i| self.valid[i])
            .min_by(|&a, &b| self.length[a].total_cmp(&self.length[b]))
            .map(|i| self.paths.path(i))
    }

    /// Every problem with a valid seed, with its best path's control points (`[points, dof]`);
    /// [`crate::timing::Trajectory::new`] times one for execution.
    pub fn solved(&self) -> impl Iterator<Item = Solved<'_, PlanProblem>> {
        self.problems
            .iter()
            .enumerate()
            .filter_map(|(index, problem)| Some(Solved { index, problem, solution: self.best(index)? }))
    }
}

pub fn plan(device: &Device, worlds: &Worlds, problems: &[PlanProblem], o: &PlanOptions) -> Result<PlanResult> {
    let deadline = o.time_budget.map(|budget| Instant::now() + budget);
    let robot = device.robot();
    let n = robot.dof();
    let points = o.control_points;
    ensure_input!(points >= 7, "need at least 7 control points");
    for (pi, p) in problems.iter().enumerate() {
        ensure_input!(
            (p.world as usize) < worlds.len(),
            "problem {pi} references world {} of {}",
            p.world,
            worlds.len()
        );
        ensure_input!(p.start.len() == n && p.goal.len() == n, "problem {pi}: start/goal must have {n} values");
        ensure_input!(
            robot.within(&p.start) && robot.within(&p.goal),
            "problem {pi}: start and goal must be within the joint limits"
        );
        ensure_input!(
            crate::rrt::holds(&p.start, &p.goal, &p.hold),
            "problem {pi}: held joints must be joints of the robot, with the start's values in the goal"
        );
        if let Some(m) = &p.start_motion {
            ensure_input!(
                m.velocity.len() == n && m.acceleration.len() == n,
                "problem {pi}: the start motion needs {n} velocities and accelerations"
            );
            let within = |v: &[f32], limits: &[f32]| v.iter().zip(limits).all(|(x, l)| x.is_finite() && x.abs() <= *l);
            ensure_input!(
                within(&m.velocity, robot.max_velocity()) && within(&m.acceleration, robot.max_acceleration()),
                "problem {pi}: the start motion exceeds the velocity or acceleration limits"
            );
        }
    }
    let seeds = 1 + o.seeds;
    let mut result = PlanResult {
        problems: problems.to_vec(),
        seeds,
        paths: JointPaths::zeros(problems.len() * seeds, points, n),
        valid: vec![false; problems.len() * seeds],
        length: vec![f32::INFINITY; problems.len() * seeds],
    };
    let mut plannings: Vec<Planning> = problems.iter().map(|p| Planning::new(robot, p, o)).collect();
    let mut running: Vec<&mut dyn Search> = plannings.iter_mut().map(|p| p as &mut dyn Search).collect();
    device.search(worlds, &mut running, deadline)?;
    let mut searched = Vec::with_capacity(problems.len());
    for (pi, planning) in plannings.into_iter().enumerate() {
        if let Some((spline, length)) = &planning.chosen {
            let item = pi * seeds;
            result.paths.path_mut(item).copy_from_slice(spline);
            result.valid[item] = true;
            result.length[item] = *length;
        }
        searched.push(planning.start);
    }
    if o.seeds > 0 && deadline.is_none_or(|d| Instant::now() < d) {
        optimize(device, worlds, &mut result, &searched, o, deadline)?;
    }
    Ok(result)
}

/// Iterations RRT-Connect gets to find a path keeping the clearance before it searches without.
const KEEP_ITERATIONS: usize = 2000;

/// One problem's planning, run whole on a device thread: RRT-Connect toward every equivalent goal,
/// simplification, then splines along the path checked shortest first. If none is valid (one cuts
/// a corner into an obstacle, or cannot follow the start motion), the first is optimized for
/// `REPAIR_ITERATIONS` steps on the same thread and checked again.
struct Planning<'r> {
    robot: &'r Robot,
    problem: &'r PlanProblem,
    o: &'r PlanOptions,
    phase: Phase<'r>,
    /// The shortest valid spline and its length.
    chosen: Option<(Vec<f32>, f32)>,
    /// The spline optimization starts from: the chosen one, or the shortest candidate.
    start: Option<Vec<f32>>,
    /// Whether the search found its path after sampling the tube around the straight line; if no
    /// spline follows that path, the problem is searched again without the tube.
    near_line: bool,
    /// What its searches share with threads that help them along.
    ahead: Arc<Ahead>,
}

enum Phase<'r> {
    /// RRT-Connect keeping the clearance first (if any), then without it.
    Search(Box<Connect<'r>>, bool),
    Simplify(Simplify),
    Fitting(Box<Fitting>),
    /// The spline to repair, and whether it has been optimized.
    Repair(Vec<f32>, bool),
    /// The repaired spline and its dense samples, under check.
    Check(Vec<f32>, Vec<f32>),
    Done,
}

impl<'r> Planning<'r> {
    fn new(robot: &'r Robot, problem: &'r PlanProblem, o: &'r PlanOptions) -> Self {
        let ahead = Arc::default();
        let phase = searching(robot, problem, o, Sampling::NearLine, &ahead);
        Self { robot, problem, o, phase, chosen: None, start: None, near_line: false, ahead }
    }
}

/// RRT-Connect for `problem`. Keeping the clearance can make narrow passages impassable, so that
/// search is short; it continues without the clearance if it fails.
fn searching<'r>(
    robot: &'r Robot,
    problem: &PlanProblem,
    o: &'r PlanOptions,
    sampling: Sampling,
    ahead: &Arc<Ahead>,
) -> Phase<'r> {
    let goals = search_goals(robot, problem);
    let search = RrtProblem { world: problem.world, start: problem.start.clone(), goals, hold: problem.hold.clone() };
    let keeping = o.rrt.clearance > 0.0;
    let first = RrtOptions { max_iterations: o.rrt.max_iterations.min(KEEP_ITERATIONS), ..o.rrt };
    let options = if keeping { &first } else { &o.rrt };
    Phase::Search(Box::new(Connect::new(robot, &search, options, sampling, ahead.clone())), keeping)
}

/// Every equivalent of `problem`'s goal a whole turn away (`Robot::goal_variants`) that leaves its
/// held joints where they start.
fn search_goals(robot: &Robot, problem: &PlanProblem) -> Vec<Vec<f32>> {
    let mut goals = robot.goal_variants(&problem.start, &problem.goal);
    goals.retain(|g| crate::rrt::holds(&problem.start, g, &problem.hold));
    goals
}

/// The next spline `fit` checks for `problem`, or `None` once every path and round is tried.
fn next_fit(
    robot: &Robot,
    problem: &PlanProblem,
    o: &PlanOptions,
    fit: &mut Fitting,
    buffer: &mut Vec<f32>,
) -> Option<EdgeQuery> {
    let (n, k) = (robot.dof(), o.validate_substeps.max(1));
    loop {
        if fit.path == fit.paths.len() {
            (fit.round, fit.path) = (fit.round + 1, 0);
            fit.restart();
        }
        if fit.round == fit.keeps.len() {
            return None;
        }
        let path = &fit.paths[fit.path];
        if !fit_spline(path, n, &fit.pins, &mut fit.spline, &mut fit.at) {
            fit.path += 1;
            fit.restart();
            continue;
        }
        if let Some(m) = &problem.start_motion {
            // A spline that cannot follow the start motion may at a longer knot interval, or once
            // it stops at every waypoint.
            let goal = &path[path.len() - n..];
            let follows = START_KNOTS.iter().any(|&longer| {
                continue_motion(robot, &problem.start, goal, m, longer, &mut fit.spline);
                Trajectory::moving(robot, &fit.spline, m).is_ok()
            });
            if !follows {
                if fit.pins.iter().all(|&p| p == EXACT) {
                    fit.path += 1;
                    fit.restart();
                } else {
                    fit.pins.fill(EXACT);
                }
                continue;
            }
        }
        spline::checked(&fit.spline, n, k, o.shortcut.resolution, &mut fit.dense, &mut fit.starts);
        // A span depends only on its four control points, so the samples before the first span a
        // changed control point reaches are as the last check found them.
        let changed = if fit.checked.len() == fit.spline.len() {
            fit.spline.chunks(n).zip(fit.checked.chunks(n)).position(|(a, b)| a != b).unwrap_or(fit.at.len())
        } else {
            0
        };
        fit.from = fit.clear.min(fit.starts[changed.saturating_sub(3).min(fit.starts.len() - 1)]);
        fit.checked.clone_from(&fit.spline);
        let keep = Some(fit.keeps[fit.round].clone());
        let mut edges = edges::reuse(buffer);
        edges.extend_from_slice(&fit.dense[fit.from * n..]);
        return Some(EdgeQuery { world: problem.world, edges, wanted: None, spacing: 0.0, keep });
    }
}

/// Clearance every spline keeps from the world away from its ends, so it cannot graze an obstacle
/// between the samples it is checked at; at most the shortcuts' clearance.
const SPLINE_MARGIN: f32 = 0.002;

/// Maximum waypoints pinned per path before the next path is tried.
const MAX_PINS: usize = 6;

/// Blends a pinned waypoint is passed within before the spline stops at it (see `fit_spline`).
const BLENDS: u8 = 3;

/// The pin of a waypoint the spline passes through, stopping there.
const EXACT: u8 = BLENDS + 1;

/// Knot intervals a moving start's spline is tried at, as multiples of `continue_motion`'s.
const START_KNOTS: [f32; 3] = [1.0, 1.5, 2.25];

/// Splines along the simplified path, then along the shortcut one. Each starts spread along its
/// path by arc length (smooth, cutting corners); where a check fails, the waypoint nearest the
/// failure is pinned, so the spline passes through it, stopping there, and the spline checked
/// again. Both paths are tried keeping the shortcuts' clearance, then again keeping
/// `SPLINE_MARGIN`. A valid spline's stops are then replaced by blends where the spline stays
/// valid, each stop's widest first (`BLENDS` of them), as OpenRAVE's smoother takes a path that
/// stops at its waypoints and removes the stops, and as TOTG passes corners within a deviation,
/// which it does not check. Each check after the first starts at the first span a change reaches.
struct Fitting {
    paths: [Vec<f32>; 2],
    /// What the splines keep from the world, round by round: the shortcuts' clearance, then
    /// `SPLINE_MARGIN`.
    keeps: Vec<Arc<Keep>>,
    round: usize,
    path: usize,
    /// Each waypoint's pin: 0 free, `1..=BLENDS` passed within a blend, `EXACT` passed through.
    pins: Vec<u8>,
    /// The spline under check, its dense samples and where each span's begin, and each control
    /// point's distance along the path.
    spline: Vec<f32>,
    dense: Vec<f32>,
    starts: Vec<usize>,
    at: Vec<f32>,
    /// The spline last checked, how many of its samples from the first were collision-free, and the
    /// first sample the check under way began at.
    checked: Vec<f32>,
    clear: usize,
    from: usize,
    /// The stop of a valid spline that blends are being tried at.
    relaxing: Option<usize>,
}

impl Fitting {
    /// No waypoint of the current path pinned.
    fn restart(&mut self) {
        let n = self.spline.len() / self.at.len();
        let waypoints = self.paths.get(self.path).map_or(0, |p| p.len() / n);
        self.pins.clear();
        self.pins.resize(waypoints, 0);
        self.clear = 0;
        self.relaxing = None;
    }

    /// Pins the unpinned interior waypoint nearest the spline's dense sample `sample`, so the
    /// spline stops there; false if `MAX_PINS` are pinned or none is left.
    fn pin_near(&mut self, sample: usize) -> bool {
        if self.pins.iter().filter(|&&p| p > 0).count() >= MAX_PINS {
            return false;
        }
        let n = self.spline.len() / self.at.len();
        let span = self.starts.partition_point(|&s| s <= sample).saturating_sub(1).min(self.at.len() - 4);
        let along = 0.5 * (self.at[span + 1] + self.at[span + 2]);
        let path = &self.paths[self.path];
        let mut walked = 0.0;
        let mut nearest = None;
        for (w, pair) in path.chunks(n).zip(path.chunks(n).skip(1)).enumerate() {
            let w = w + 1;
            walked += distance(pair.0, pair.1);
            if w + 1 < self.pins.len() && self.pins[w] == 0 {
                let gap = (walked - along).abs();
                if nearest.is_none_or(|(_, g)| gap < g) {
                    nearest = Some((w, gap));
                }
            }
        }
        let Some((w, _)) = nearest else { return false };
        self.pins[w] = EXACT;
        true
    }

    /// The next blend to try in place of a valid spline's stops, after a check that `passed` of the
    /// last one (or of the valid spline, which starts this): the next level at the stop under way,
    /// else the widest at the next stop, restoring the stop where no blend passed. False once every
    /// stop has been tried.
    fn relax(&mut self, passed: bool) -> bool {
        let next = match self.relaxing {
            Some(w) if !passed && self.pins[w] < BLENDS => {
                self.pins[w] += 1;
                return true;
            }
            Some(w) => {
                if !passed {
                    self.pins[w] = EXACT;
                }
                w + 1
            }
            None => 1,
        };
        self.relaxing = (next..self.pins.len().saturating_sub(1)).find(|&w| self.pins[w] == EXACT);
        if let Some(w) = self.relaxing {
            self.pins[w] = 1;
        }
        self.relaxing.is_some()
    }
}

impl Search for Planning<'_> {
    fn help(&self) -> Option<Arc<Ahead>> {
        Some(self.ahead.clone())
    }

    fn request(&mut self, buffer: &mut Vec<f32>) -> Option<EdgeQuery> {
        loop {
            match &mut self.phase {
                Phase::Search(connect, keeping) => {
                    if let Some(query) = connect.request(buffer) {
                        return Some(query);
                    }
                    if connect.path.is_none() && *keeping {
                        connect.relax(self.o.rrt.max_iterations);
                        *keeping = false;
                        continue;
                    }
                    self.near_line = connect.path.is_some() && connect.sampled_tube();
                    self.ahead.finish();
                    self.phase = match connect.path.take() {
                        Some(path) => {
                            Phase::Simplify(Simplify::new(self.problem.world, self.robot.dof(), path, &self.o.shortcut))
                        }
                        None => Phase::Done,
                    };
                }
                Phase::Simplify(simplify) => {
                    if let Some(query) = simplify.request(buffer) {
                        return Some(query);
                    }
                    let Phase::Simplify(simplify) = std::mem::replace(&mut self.phase, Phase::Done) else {
                        unreachable!()
                    };
                    let (smooth, short) = simplify.into_paths();
                    let (n, points) = (self.robot.dof(), self.o.control_points);
                    let ends = vec![short[..n].to_vec(), short[short.len() - n..].to_vec()];
                    let ramp = self.o.shortcut.clearance_ramp;
                    // The margin never exceeds what the shortcuts kept, or no spline could follow them.
                    let clearance = self.o.shortcut.clearance;
                    let rounds: &[f32] =
                        if clearance > SPLINE_MARGIN { &[clearance, SPLINE_MARGIN] } else { &[clearance] };
                    let keeps = rounds
                        .iter()
                        .map(|&clearance| Arc::new(Keep { clearance, ramp, ends: ends.clone() }))
                        .collect();
                    let mut fit = Fitting {
                        paths: [smooth, short],
                        keeps,
                        round: 0,
                        path: 0,
                        pins: vec![],
                        spline: vec![0.0; points * n],
                        dense: vec![],
                        starts: vec![],
                        at: vec![0.0; points],
                        checked: vec![],
                        clear: 0,
                        from: 0,
                        relaxing: None,
                    };
                    fit.restart();
                    // Optimization, if any, starts from the smoothed path spread out.
                    fit_spline(&fit.paths[0], n, &fit.pins, &mut fit.spline, &mut fit.at);
                    if let Some(m) = &self.problem.start_motion {
                        let goal = &fit.paths[0][fit.paths[0].len() - n..];
                        continue_motion(self.robot, &self.problem.start, goal, m, 1.0, &mut fit.spline);
                    }
                    self.start = Some(fit.spline.clone());
                    self.phase = Phase::Fitting(Box::new(fit));
                }
                Phase::Fitting(fit) => {
                    if let Some(query) = next_fit(self.robot, self.problem, self.o, fit, buffer) {
                        return Some(query);
                    }
                    self.phase = Phase::Repair(self.start.clone().expect("fitting follows simplification"), false);
                }
                // Not yet optimized: `optimize` hands the spline over first.
                Phase::Repair(_, false) => return None,
                Phase::Repair(spline, true) => {
                    let (n, k) = (self.robot.dof(), self.o.validate_substeps.max(1));
                    hold_still(spline, &self.problem.start, &self.problem.hold);
                    let (mut dense, mut starts) = (vec![], vec![]);
                    spline::checked(spline, n, k, self.o.shortcut.resolution, &mut dense, &mut starts);
                    let mut edges = edges::reuse(buffer);
                    edges.extend_from_slice(&dense);
                    let query = EdgeQuery { world: self.problem.world, edges, wanted: None, spacing: 0.0, keep: None };
                    self.phase = Phase::Check(std::mem::take(spline), dense);
                    return Some(query);
                }
                // A path found near the straight line that no spline follows: the search a planner
                // without the tube would run.
                Phase::Done if self.chosen.is_none() && self.near_line => {
                    self.near_line = false;
                    self.phase = searching(self.robot, self.problem, self.o, Sampling::JointRanges, &self.ahead);
                }
                Phase::Check(..) | Phase::Done => return None,
            }
        }
    }

    fn answer(&mut self, answer: usize) {
        match &mut self.phase {
            Phase::Search(connect, _) => connect.answer(answer),
            Phase::Simplify(simplify) => simplify.answer(answer),
            Phase::Fitting(fit) => {
                let n = self.robot.dof();
                let answer = fit.from + answer;
                let passed = answer == fit.dense.len() / n;
                fit.clear = answer;
                if passed {
                    self.start = Some(fit.spline.clone());
                    self.chosen = Some((fit.spline.clone(), length(&fit.dense, n)));
                }
                if passed || fit.relaxing.is_some() {
                    // A moving start keeps its stops: its first spans are fitted to the motion.
                    if self.problem.start_motion.is_some() || !fit.relax(passed) {
                        self.phase = Phase::Done;
                    }
                } else if !fit.pin_near(answer) {
                    fit.path += 1;
                    fit.restart();
                }
            }
            Phase::Check(spline, dense) => {
                let n = self.robot.dof();
                let follows = |m: &StartMotion| Trajectory::moving(self.robot, spline, m).is_ok();
                if answer == dense.len() / n && self.problem.start_motion.as_ref().is_none_or(follows) {
                    self.chosen = Some((std::mem::take(spline), length(dense, n)));
                }
                self.phase = Phase::Done;
            }
            Phase::Repair(..) | Phase::Done => unreachable!("answers follow requests"),
        }
    }

    fn optimize(&mut self) -> Option<Optimize<'_>> {
        let Phase::Repair(spline, optimized @ false) = &mut self.phase else { return None };
        *optimized = true;
        Some(Optimize { world: self.problem.world, path: spline, options: repairing(self.o) })
    }
}

/// L-BFGS steps that repair a stuck spline: a few push it off the obstacles, and many more trade
/// that clearance back for length, leaving fewer valid.
const REPAIR_ITERATIONS: u32 = 10;

/// Repairs weigh smoothness over length and keep a 2 cm margin whatever the options say: their
/// job is a spline clear of the obstacles, which `PlanOptions`' weights for shortening seeds solve
/// fewer of.
fn repairing(o: &PlanOptions) -> PlanOptions {
    let collision = CollisionWeights { margin: 0.02, ..o.collision };
    PlanOptions { iterations: REPAIR_ITERATIONS, w_acc: 50.0, w_vel: 2.0, collision, ..*o }
}

/// Searched paths within this fraction of the straight line between their ends are not optimized:
/// no seed could be much shorter.
const OPTIMIZE_SLACK: f32 = 0.02;

/// Trajectory optimization of `o.seeds` seeds per problem into seeds 1 onward: the searched
/// spline where there is one, straight lines to the goal and its equivalents, and bends through
/// random via points. Problems whose searched path is within `OPTIMIZE_SLACK` of straight keep it.
fn optimize(
    device: &Device,
    worlds: &Worlds,
    result: &mut PlanResult,
    searched: &[Option<Vec<f32>>],
    o: &PlanOptions,
    deadline: Option<Instant>,
) -> Result<()> {
    let robot = device.robot();
    let (n, points) = (robot.dof(), o.control_points);
    ensure_input!(o.samples_per_span > 0, "need at least one collision sample per span");
    ensure_input!((1..=MAX_HISTORY).contains(&o.history), "L-BFGS history must be 1 to {MAX_HISTORY} steps");
    ensure_input!(o.initial_step > 0.0, "the initial step must be positive");
    let mut rng = Rng::new(o.rng_seed);
    let near_straight = |pi: usize| {
        let item = pi * result.seeds;
        let searched = result.paths.path(item);
        let straight = distance(&searched[..n], &searched[searched.len() - n..]);
        result.valid[item] && result.length[item] <= straight * (1.0 + OPTIMIZE_SLACK)
    };
    let wanted: Vec<usize> = (0..result.problems.len()).filter(|&pi| !near_straight(pi)).collect();
    let mut paths = JointPaths::zeros(wanted.len() * o.seeds, points, n);
    let mut item_world = Vec::with_capacity(paths.len());
    for (w, &pi) in wanted.iter().enumerate() {
        let p = &result.problems[pi];
        let goals = robot.goal_variants(&p.start, &p.goal);
        let lines_from = usize::from(searched[pi].is_some());
        for s in 0..o.seeds {
            let path = paths.path_mut(w * o.seeds + s);
            match &searched[pi] {
                Some(spline) if s == 0 => path.copy_from_slice(spline),
                _ => {
                    let line = s - lines_from;
                    let (goal, straight) = match goals.get(line) {
                        Some(goal) => (goal, true),
                        None => (&goals[0], false),
                    };
                    seed_path(robot, &p.start, goal, straight, &mut rng, path);
                    if let Some(m) = &p.start_motion {
                        continue_motion(robot, &p.start, goal, m, 1.0, path);
                    }
                }
            }
            item_world.push(p.world);
        }
    }
    device.trajopt(worlds, &item_world, &mut paths, o, deadline)?;
    for (w, &pi) in wanted.iter().enumerate() {
        for s in 0..o.seeds {
            hold_still(paths.path_mut(w * o.seeds + s), &result.problems[pi].start, &result.problems[pi].hold);
        }
    }
    let (clearance, lengths) = validate(device, worlds, &paths, &item_world, o)?;
    for (w, &pi) in wanted.iter().enumerate() {
        for s in 0..o.seeds {
            let (from, to) = (w * o.seeds + s, pi * result.seeds + 1 + s);
            result.paths.path_mut(to).copy_from_slice(paths.path(from));
            result.valid[to] = clearance[from] >= 0.0;
            result.length[to] = lengths[from];
        }
    }
    Ok(())
}

/// The smallest clearance along each path, sampled densely along the curve, and its length.
fn validate(
    device: &Device,
    worlds: &Worlds,
    paths: &JointPaths,
    item_world: &[u32],
    o: &PlanOptions,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (n, items) = (paths.dof, paths.len());
    let k = o.validate_substeps.max(1);
    let samples = (paths.points - 3) * k + 1;
    let mut dense = vec![0.0; items * samples * n];
    for (item, out) in dense.chunks_mut(samples * n).enumerate() {
        spline::dense(paths.path(item), n, k, out);
    }
    let dense_world: Vec<u32> = item_world.iter().flat_map(|&w| std::iter::repeat_n(w, samples)).collect();
    let clear = device.clearance(worlds, &dense_world, &dense)?;
    let min_clearance = (0..items)
        .map(|item| {
            (item * samples..(item + 1) * samples)
                .map(|i| clear[i][0].min(clear[i][1]))
                // NaN marks the path invalid rather than vanishing in a minimum.
                .fold(f32::INFINITY, |m, c| if m.is_nan() || c.is_nan() { f32::NAN } else { m.min(c) })
        })
        .collect();
    let lengths = (0..items).map(|item| length(&dense[item * samples * n..(item + 1) * samples * n], n)).collect();
    Ok((min_clearance, lengths))
}

/// Sets every control point's `hold` joints to `start`'s values: optimization moves every joint,
/// and the check that follows decides whether the path still holds.
fn hold_still(path: &mut [f32], start: &[f32], hold: &[usize]) {
    for point in path.chunks_mut(start.len()) {
        for &j in hold {
            point[j] = start[j];
        }
    }
}

/// B-spline control points along a waypoint path, three at each of its ends and of its `pinned`
/// waypoints, and the rest spread along it by arc length between them (the curve cuts the corners
/// there). A waypoint pinned at level `1..=BLENDS` gets its three on the path a blend before it, at
/// it and a blend after it, so the curve turns within the blend without stopping: the first blend
/// is the free control points' spacing, or half the shorter edge at the waypoint if less (as TOTG's
/// blends use at most half of either edge), and each level a quarter of the last. At the ends and
/// at `EXACT` the three coincide: the curve passes through, stopping there. `at` gets each control
/// point's distance along the path. With nothing pinned the control points spread along the whole
/// path; with every waypoint pinned `EXACT` the curve runs exactly along it. False if the pins need
/// more control points than `out` holds.
fn fit_spline(path: &[f32], n: usize, pinned: &[u8], out: &mut [f32], at: &mut [f32]) -> bool {
    let (waypoints, count) = (path.len() / n, out.len() / n);
    let point = |w: usize| &path[w * n..(w + 1) * n];
    let mut along = vec![0.0; waypoints];
    for w in 1..waypoints {
        along[w] = along[w - 1] + distance(point(w - 1), point(w));
    }
    let last = waypoints - 1;
    let spacing = along[last] / count.saturating_sub(5).max(1) as f32;
    let blend = |w: usize| {
        if w == 0 || w == last || pinned[w] >= EXACT {
            return 0.0;
        }
        let room = 0.5 * (along[w] - along[w - 1]).min(along[w + 1] - along[w]);
        spacing.min(room) * 0.25f32.powi(pinned[w] as i32 - 1)
    };
    let pins: Vec<(usize, f32)> =
        (0..waypoints).filter(|&w| w == 0 || w == last || pinned[w] > 0).map(|w| (w, blend(w))).collect();
    let Some(free) = count.checked_sub(3 * pins.len()) else { return false };
    // Free control points per stretch between pins' blends, in proportion to its length (largest
    // remainders, so the counts add up).
    let stretch: Vec<f32> = pins.windows(2).map(|p| (along[p[1].0] - p[1].1) - (along[p[0].0] + p[0].1)).collect();
    let total = stretch.iter().sum::<f32>().max(1e-9);
    let share = |i: usize| free as f32 * stretch[i] / total;
    let mut counts: Vec<usize> = (0..stretch.len()).map(|i| share(i) as usize).collect();
    while counts.iter().sum::<usize>() < free {
        let i =
            (0..stretch.len()).max_by(|&a, &b| (share(a) - counts[a] as f32).total_cmp(&(share(b) - counts[b] as f32)));
        counts[i.expect("a path has two ends")] += 1;
    }
    let mut t = 0;
    let mut q = vec![0.0; n];
    let mut put = |q: &[f32], s: f32, t: &mut usize| {
        out[*t * n..(*t + 1) * n].copy_from_slice(q);
        at[*t] = s;
        *t += 1;
    };
    for (i, &(pin, blend)) in pins.iter().enumerate() {
        // Without a blend, the three are the waypoint itself, copied bit-exact.
        for offset in [-blend, 0.0, blend] {
            if offset == 0.0 {
                put(point(pin), along[pin], &mut t);
            } else {
                locate(path, n, along[pin] + offset, &mut q);
                put(&q, along[pin] + offset, &mut t);
            }
        }
        for c in 1..=counts.get(i).copied().unwrap_or(0) {
            let s = along[pin] + blend + stretch[i] * c as f32 / (counts[i] + 1) as f32;
            locate(path, n, s, &mut q);
            put(&q, s, &mut t);
        }
    }
    true
}

/// Pins `path`'s first three control points so the spline leaves `start` with `motion`'s velocity
/// and acceleration at the knot interval uniform timing gives a straight path to `goal` (see
/// [`crate::Trajectory::moving`]), at `longer` times the knot interval chosen for it.
fn continue_motion(robot: &Robot, start: &[f32], goal: &[f32], motion: &StartMotion, longer: f32, path: &mut [f32]) {
    let n = start.len();
    let mut straight = vec![0.0; path.len()];
    for (t, q) in straight.chunks_mut(n).enumerate() {
        let u = (t.saturating_sub(2) as f32 / (path.len() / n - 5) as f32).min(1.0);
        q.iter_mut().enumerate().for_each(|(j, v)| *v = start[j] + (goal[j] - start[j]) * u);
    }
    // A quarter slower than the straight path allows, leaving room for the bends optimization adds;
    // a straight path that does not move still needs a time scale to leave its start with. A fast
    // start needs longer: changing its velocity takes about |v| / a, and the spline's first spans
    // must last that long to turn within the jerk limit (jerk grows with the cube of the speed
    // through them).
    let (v, a) = (&motion.velocity, &motion.acceleration);
    let turn = (0..n).map(|j| v[j].abs() / robot.max_acceleration[j]).fold(0.0, f32::max);
    let h0 = longer * (1.25 * crate::timing::uniform_knot(robot, &straight)).max(turn).max(1e-2);
    for j in 0..n {
        path[j] = start[j] - v[j] * h0 + a[j] * h0 * h0 / 3.0;
        path[n + j] = start[j] - a[j] * h0 * h0 / 6.0;
        path[2 * n + j] = start[j] + v[j] * h0 + a[j] * h0 * h0 / 3.0;
    }
}

/// Control points on the straight line from `start` to `goal`, or bent through a random via point.
/// The first three and last three are the start and goal.
fn seed_path(robot: &Robot, start: &[f32], goal: &[f32], straight: bool, rng: &mut Rng, out: &mut [f32]) {
    let n = start.len();
    let t_count = out.len() / n;
    let (via_u, via): (f32, Vec<f32>) = if straight {
        (0.5, (0..n).map(|j| 0.5 * (start[j] + goal[j])).collect())
    } else {
        let u = rng.range(0.3, 0.7);
        let alpha = rng.range(0.2, 0.6);
        let via = (0..n)
            .map(|j| {
                let mid = start[j] + u * (goal[j] - start[j]);
                mid + alpha * (rng.range(robot.lower[j], robot.upper[j]) - mid)
            })
            .collect();
        (u, via)
    };
    // Free control points 3..t_count - 3 spread evenly along the seed between the pinned ends.
    for t in 3..t_count - 3 {
        let u = (t - 2) as f32 / (t_count - 5) as f32;
        for j in 0..n {
            out[t * n + j] = if u <= via_u {
                start[j] + (via[j] - start[j]) * (u / via_u)
            } else {
                via[j] + (goal[j] - via[j]) * ((u - via_u) / (1.0 - via_u))
            };
        }
    }
    // Copied, not interpolated, so they stay bit-exact.
    for t in 0..3 {
        out[t * n..(t + 1) * n].copy_from_slice(start);
        out[(t_count - 1 - t) * n..(t_count - t) * n].copy_from_slice(goal);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The control points pass through the ends and every pinned waypoint, and with every
    /// waypoint pinned the curve runs along the path; without pins they spread evenly along it.
    #[test]
    fn splines_pass_through_their_pinned_waypoints() {
        let n = 2;
        let path = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0];
        let (mut out, mut at) = (vec![0.0; 24 * n], vec![0.0; 24]);
        let k = 16;
        let mut dense = vec![0.0; (21 * k + 1) * n];
        let near = |dense: &[f32], q: &[f32]| dense.chunks(n).any(|d| distance(d, q) < 1e-4);
        for pinned in [[0, EXACT, 0, EXACT, 0], [0, 0, EXACT, 0, 0]] {
            assert!(fit_spline(&path, n, &pinned, &mut out, &mut at));
            spline::dense(&out, n, k, &mut dense);
            for (w, q) in path.chunks(n).enumerate() {
                let must = w == 0 || w == 4 || pinned[w] == EXACT;
                assert!(!must || near(&dense, q), "the spline misses pinned waypoint {w} ({pinned:?})");
            }
            assert!(at.windows(2).all(|a| a[0] <= a[1]), "control points out of order along the path");
        }
        // Every waypoint pinned: each dense sample lies on the path.
        assert!(fit_spline(&path, n, &[EXACT; 5], &mut out, &mut at));
        spline::dense(&out, n, k, &mut dense);
        let on_path = |d: &[f32]| {
            (0..4).any(|e| {
                let (a, b) = (&path[e * n..(e + 1) * n], &path[(e + 1) * n..(e + 2) * n]);
                ((d[0] - a[0]) * (b[1] - a[1]) - (d[1] - a[1]) * (b[0] - a[0])).abs() < 1e-5
                    && (0..n).all(|j| d[j] >= a[j].min(b[j]) - 1e-5 && d[j] <= a[j].max(b[j]) + 1e-5)
            })
        };
        assert!(dense.chunks(n).all(on_path), "a fully pinned spline leaves its path");
        // No pins: the free control points are evenly spaced along the path, cutting its corners.
        assert!(fit_spline(&path, n, &[0; 5], &mut out, &mut at));
        let gaps: Vec<f32> = at[3..21].windows(2).map(|a| a[1] - a[0]).collect();
        assert!(gaps.iter().all(|g| (g - gaps[0]).abs() < 1e-5), "free control points not even: {gaps:?}");
        // Too many pins for the control points.
        assert!(!fit_spline(&path, n, &[EXACT; 5], &mut vec![0.0; 12 * n], &mut [0.0; 12]));
    }

    /// A joint whose range spans more than a turn may end a turn away, unless it is held.
    #[test]
    fn search_goals_never_turn_held_joints() {
        let ur5e = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/ur5e/ur_description/urdf/ur5e.urdf");
        let robot = Robot::load(ur5e, &crate::robot::RobotOptions::default()).unwrap();
        let start = vec![0.2; 6];
        let goal = vec![1.0, 0.2, 0.5, 0.2, -1.0, 0.2];
        let hold = vec![1, 3, 5];
        let problem = |hold: &[usize]| PlanProblem {
            world: 0,
            start: start.clone(),
            goal: goal.clone(),
            start_motion: None,
            hold: hold.to_vec(),
        };
        let (held, free) = (search_goals(&robot, &problem(&hold)), search_goals(&robot, &problem(&[])));
        assert!(held.len() > 1, "the moving joints should have variants a turn away");
        assert!(free.len() > held.len(), "turning held joints should add variants");
        for g in &held {
            assert!(hold.iter().all(|&j| g[j] == start[j]), "a goal turns a held joint: {g:?}");
        }
    }

    /// After a change, a check starts no later than the first dense sample that changed, and runs
    /// to the end: the samples before it are ones the last check passed.
    #[test]
    fn rechecks_start_where_the_spline_changed() {
        use crate::robot::{CollisionModel, RobotOptions};
        let fingers = [("panda_finger_joint1".to_string(), 0.04), ("panda_finger_joint2".to_string(), 0.04)];
        let options = RobotOptions {
            lock_joints: std::collections::HashMap::from(fingers),
            collision_model: Some(CollisionModel::load("assets/franka/panda_collision.json").unwrap()),
            ..Default::default()
        };
        let robot = Robot::load("assets/franka/franka_panda.urdf", &options).unwrap();
        let n = robot.dof();
        let path: Vec<f32> =
            (0..6).flat_map(|w| (0..n).map(move |j| 0.5 * ((j + 1) as f32 * w as f32 * 0.5).sin())).collect();
        let (start, goal) = (path[..n].to_vec(), path[path.len() - n..].to_vec());
        let problem = PlanProblem::new(0, start, goal);
        let o = PlanOptions::default();
        let keep = Arc::new(Keep { clearance: 0.0, ramp: 1.0, ends: vec![] });
        let points = o.control_points;
        let mut fit = Fitting {
            paths: [path.clone(), path],
            keeps: vec![keep],
            round: 0,
            path: 0,
            pins: vec![],
            spline: vec![0.0; points * n],
            dense: vec![],
            starts: vec![],
            at: vec![0.0; points],
            checked: vec![],
            clear: 0,
            from: 0,
            relaxing: None,
        };
        fit.restart();
        let mut buffer = vec![];
        let query = next_fit(&robot, &problem, &o, &mut fit, &mut buffer).unwrap();
        assert_eq!((fit.from, query.edges.len()), (0, fit.dense.len()), "the first check covers every sample");
        // A stop, then blends in its place, then another stop, each after a check that passed.
        for (w, pin) in [(3, EXACT), (3, 1), (3, 2), (2, EXACT), (4, EXACT), (4, 1)] {
            let before = fit.dense.clone();
            fit.clear = before.len() / n;
            fit.pins[w] = pin;
            let query = next_fit(&robot, &problem, &o, &mut fit, &mut buffer).unwrap();
            let changed = before.chunks(n).zip(fit.dense.chunks(n)).position(|(a, b)| a != b);
            let changed = changed.unwrap_or(before.len().min(fit.dense.len()) / n);
            assert!(
                fit.from <= changed,
                "pin {pin} at {w}: the check starts at {} past a change at {changed}",
                fit.from
            );
            assert_eq!(query.edges, fit.dense[fit.from * n..], "pin {pin} at {w}: the check stops short");
        }
    }

    /// A blended pin draws the curve in around its corner without stopping it there, closer at each
    /// level; only `EXACT` passes through, stopping.
    #[test]
    fn blended_pins_turn_without_stopping() {
        let n = 2;
        let path = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let (mut out, mut at) = (vec![0.0; 24 * n], vec![0.0; 24]);
        let k = 64;
        let mut dense = vec![0.0; (21 * k + 1) * n];
        let mut gaps = vec![];
        for level in 0..=EXACT {
            assert!(fit_spline(&path, n, &[0, level, 0], &mut out, &mut at));
            assert!(at.windows(2).all(|a| a[0] <= a[1]), "control points out of order at level {level}");
            spline::dense(&out, n, k, &mut dense);
            gaps.push(dense.chunks(n).map(|d| distance(d, &path[2..4])).fold(f32::INFINITY, f32::min));
            // The curve stops where three control points in a row coincide: at both ends, and at
            // an `EXACT` pin.
            let points: Vec<&[f32]> = out.chunks(n).collect();
            let stops =
                (1..points.len() - 1).filter(|&t| points[t - 1] == points[t] && points[t] == points[t + 1]).count();
            assert_eq!(stops, if level == EXACT { 3 } else { 2 }, "stops at level {level}");
        }
        assert!(gaps.windows(2).all(|g| g[1] < g[0]), "each level should pass the corner closer: {gaps:?}");
        assert!(gaps[EXACT as usize] < 1e-6, "an exact pin misses its waypoint by {}", gaps[EXACT as usize]);
    }
}
