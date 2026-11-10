//! Planning paths.

use std::time::Duration;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::device::{Device, Worlds};
use crate::error::OrRaise;
use crate::field::{self, Field, tuple_field, value_class};
use crate::solved::{Cached, Solved};
use crate::world::Pose;

value_class! {
    /// How the search for each path runs: RRT-Connect after the straight line.
    RrtOptions(batchplan::RrtOptions) {
        /// Longest edge a tree grows by (joint-space distance).
        range: f32,
        /// Spacing of collision checks along edges (joint-space distance), at least 1e-4.
        resolution: f32,
        /// Samples a problem draws from the joint ranges before it gives up. 0 tries only the
        /// straight lines and their bends.
        max_iterations: usize,
        /// Where in the Halton sequence samples start.
        rng_seed: u64,
        /// Clearance edges keep from the world (meters), shrinking to zero within
        /// `clearance_ramp` (joint-space distance) of the start and the goals, which may touch the
        /// world.
        clearance: f32,
        /// Joint-space distance from the path's ends over which the clearance ramps up.
        clearance_ramp: f32,
    }
    methods {
        #[new]
        #[pyo3(signature = (**changes))]
        fn new(changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            Ok(Self(Self::changed(Default::default(), changes)?))
        }
    }
}

value_class! {
    /// How the path found is shortened and smoothed.
    ShortcutOptions(batchplan::ShortcutOptions) {
        /// Spacing of collision checks along new edges (joint-space distance), at least 1e-4.
        resolution: f32,
        /// Passes of shortcutting and smoothing at most; they stop early once a pass changes
        /// nothing.
        passes: usize,
        /// Smoothing moves a waypoint, and straightening a joint shortens the path, only by at
        /// least this much (joint-space distance); infinity turns both off.
        min_change: f32,
        /// Straighten each joint in turn after the first pass's shortcuts: about 1% shorter paths
        /// on the Panda benchmark, and 13-18% on pick-and-place carries.
        straighten: bool,
        /// Clearance new edges keep from the world (meters), so the path does not pull taut
        /// against obstacles. It shrinks to zero within `clearance_ramp` (joint-space distance) of
        /// the path's ends, which may touch the world.
        clearance: f32,
        /// Joint-space distance from the path's ends over which the clearance ramps up.
        clearance_ramp: f32,
    }
    methods {
        #[new]
        #[pyo3(signature = (**changes))]
        fn new(changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            Ok(Self(Self::changed(Default::default(), changes)?))
        }
    }
}

value_class! {
    /// How `plan` searches, shortens, fits and optimizes paths.
    PlanOptions(batchplan::PlanOptions) {
        /// Seeds trajectory optimization runs per problem, besides the searched path: its spline,
        /// the straight lines to the goal and its equivalents, then bends through random via
        /// points. 0 keeps the searched path as it is.
        seeds: usize,
        /// B-spline control points per path, including the three pinned at each end (at least 7).
        /// A searched path that no spline this size follows is fitted again with twice as many.
        control_points: usize,
        /// Collision samples per span during optimization.
        samples_per_span: usize,
        /// L-BFGS steps of trajectory optimization.
        iterations: u32,
        /// Steps L-BFGS remembers (1 to 8).
        history: usize,
        /// Largest joint change of a step without history: the first, and any after a reset.
        initial_step: f32,
        /// Weight of squared second differences of control points.
        acceleration_weight: f32,
        /// Weight of squared differences of control points (path length).
        velocity_weight: f32,
        /// Collision cost of the optimization.
        collision: batchplan::CollisionWeights,
        /// Samples per span splines are checked at, at least: more where samples would lie
        /// farther apart than `shortcut.resolution`.
        validate_substeps: usize,
        /// Seed of the optimization seeds' random via points. Each problem draws its own from it
        /// and its start and goal, so a problem's plan does not depend on the others in its batch.
        rng_seed: u64,
        /// Searching for each path: RRT-Connect after the straight line.
        rrt: batchplan::RrtOptions,
        /// Shortening and smoothing the path found.
        shortcut: batchplan::ShortcutOptions,
        /// IK for pose goals: its seeds, iterations and tolerances. Problems it finds no
        /// collision-free solution for are solved again with four times the seeds, up to sixteen
        /// times.
        goal_ik: batchplan::IkOptions,
        /// Seconds after which optimizing and searching stop and the best valid paths so far are
        /// returned; validation still runs after it. Pose goals' first round of IK counts toward
        /// it and always runs; their later rounds do not start once it has passed. Results then
        /// depend on machine speed; without a budget they are the same on every run.
        time_budget: Option<Duration>,
    }
    methods {
        #[new]
        #[pyo3(signature = (**changes))]
        fn new(changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            Ok(Self(Self::changed(Default::default(), changes)?))
        }
    }
}

value_class! {
    /// How the robot is already moving where a plan starts: joint velocity and acceleration.
    StartMotion(batchplan::StartMotion) {
        /// Per joint, rad/s (m/s for prismatic joints).
        velocity: Vec<f32>,
        /// Per joint, rad/s² (m/s²).
        acceleration: Vec<f32>,
    }
    methods {
        #[new]
        fn new(velocity: &Bound<'_, PyAny>, acceleration: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(Self(batchplan::StartMotion {
                velocity: Vec::from_py(velocity)?,
                acceleration: Vec::from_py(acceleration)?,
            }))
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
}

value_class! {
    /// A path from a start to a goal, a configuration or a pose per hand, in one world.
    PlanProblem(batchplan::PlanProblem) {
        /// The world's index in the batch.
        world: u32,
        /// Where the path starts.
        start: Vec<f32>,
        /// Where the path ends: a configuration, or a pose per hand.
        ///
        /// A configuration is reached up to whole turns: continuous joints turn the short way (or
        /// the long way, if that is the one that is free), and a joint whose range spans more than
        /// a turn may end a turn away from this value when that is nearer or free.
        ///
        /// Poses are a `Pose` for a robot with one hand, or a `Pose` or `None` per hand (for
        /// `None`, the joints only that hand moves stay where `start` has them), and come back as
        /// that tuple. `plan`
        /// solves IK for them from `start` (`PlanOptions.goal_ik`, with more seeds where those
        /// find nothing collision-free), and the path ends at the collision-free solution nearest
        /// `start`, up to whole turns, that keeps the problem's constraints.
        goal: batchplan::Goal,
        /// How the robot is already moving at `start`, to replan mid-motion; `None` starts at
        /// rest.
        start_motion: Option<batchplan::StartMotion>,
        /// Joints that stay where `start` has them, as an arm that waits while another moves;
        /// `goal` has the same values there.
        hold: Vec<usize>,
        /// Where the hands must stay along the whole path, as a tray carried upright; `start` and
        /// `goal` must keep them. Every state the path is checked at keeps them.
        constraints: Vec<batchplan::HandConstraint>,
        /// An object a hand holds along the path, as a pick leaves it: checked as part of the
        /// robot, as attaching it (`Held.attachment`) would check it. `None` holds nothing.
        holding: Option<batchplan::Held>,
    }
    methods {
        /// From `start` at rest to `goal` (a configuration or poses) in world `world`, every joint
        /// free to move.
        #[new]
        #[pyo3(signature = (world, start, goal, **changes))]
        fn new(
            world: u32,
            start: &Bound<'_, PyAny>,
            goal: &Bound<'_, PyAny>,
            changes: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Self> {
            let problem = batchplan::PlanProblem::new(
                world,
                Vec::<f32>::from_py(start)?,
                batchplan::Goal::from_py(goal)?,
            );
            Ok(Self(Self::changed(problem, changes)?))
        }
    }
}

tuple_field!(batchplan::PlanProblem);

/// A configuration from any array-like, or poses: a `Pose`, or a sequence holding a `Pose` or
/// `None` per hand, which is how poses go out.
impl Field for batchplan::Goal {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            batchplan::Goal::Joints(q) => q.to_py(py),
            batchplan::Goal::Poses(targets) => targets.to_py(py),
        }
    }

    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(pose) = value.cast::<Pose>() {
            return Ok(batchplan::Goal::Poses(vec![Some(pose.get().0)]));
        }
        // A scalar, a configuration of one joint, has no items.
        let Ok(items) = field::items(value) else {
            return Ok(batchplan::Goal::Joints(field::floats(value)?));
        };
        if items
            .iter()
            .any(|v| v.is_none() || v.is_instance_of::<Pose>())
        {
            let targets = items.iter().map(Option::<batchplan::Pose>::from_py);
            return Ok(batchplan::Goal::Poses(targets.collect::<PyResult<_>>()?));
        }
        // From the items, since an iterator has given them up.
        let q = field::floats(PyList::new(value.py(), items)?.as_any())?;
        Ok(batchplan::Goal::Joints(q))
    }
}

/// Why a problem of a `PlanResult` has no valid path, the first thing that stopped it.
#[pyclass(module = "batchplan", frozen)]
pub(crate) enum PlanFailure {
    /// The start collides, with `contacts`, nearest first: plan from a free state.
    StartCollides { contacts: Py<PyAny> },
    /// The goal collides, and so does every equivalent of it a whole turn away; for a pose
    /// goal, IK reaches the poses only in collision, at the solution with the most clearance.
    /// `contacts` says with what, nearest first.
    GoalCollides { contacts: Py<PyAny> },
    /// No configuration IK finds reaches the goal's poses within its tolerances, keeping the
    /// problem's held joints and constraints: they lie beyond the arm's reach or its joint limits,
    /// or need more seeds or iterations than `PlanOptions.goal_ik` gives.
    GoalUnreachable(),
    /// The search drew every sample it may (`RrtOptions.max_iterations`) without joining the
    /// start to the goal: the way is blocked, or too narrow for the samples given.
    NoPath(),
    /// A path was found, but no spline along it (fitted, repaired, or with more control points)
    /// checked collision-free, or followed the start's motion.
    NoSpline(),
    /// The time budget ran out first.
    OutOfTime(),
    /// A reason these bindings don't know yet, as batchplan describes it.
    Other { description: String },
}

#[pymethods]
impl PlanFailure {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(match self {
            Self::StartCollides { contacts } => {
                format!(
                    "PlanFailure.StartCollides(contacts={})",
                    contacts.bind(py).repr()?
                )
            }
            Self::GoalCollides { contacts } => {
                format!(
                    "PlanFailure.GoalCollides(contacts={})",
                    contacts.bind(py).repr()?
                )
            }
            Self::GoalUnreachable() => "PlanFailure.GoalUnreachable()".to_owned(),
            Self::NoPath() => "PlanFailure.NoPath()".to_owned(),
            Self::NoSpline() => "PlanFailure.NoSpline()".to_owned(),
            Self::OutOfTime() => "PlanFailure.OutOfTime()".to_owned(),
            Self::Other { description } => format!("PlanFailure.Other({description:?})"),
        })
    }
}

impl PlanFailure {
    fn new(py: Python<'_>, f: &batchplan::PlanFailure) -> PyResult<Self> {
        Ok(match f {
            batchplan::PlanFailure::StartCollides { contacts } => Self::StartCollides {
                contacts: contacts.to_py(py)?.unbind(),
            },
            batchplan::PlanFailure::GoalCollides { contacts } => Self::GoalCollides {
                contacts: contacts.to_py(py)?.unbind(),
            },
            batchplan::PlanFailure::GoalUnreachable => Self::GoalUnreachable(),
            batchplan::PlanFailure::NoPath => Self::NoPath(),
            batchplan::PlanFailure::NoSpline => Self::NoSpline(),
            batchplan::PlanFailure::OutOfTime => Self::OutOfTime(),
            other => Self::Other {
                description: format!("{other:?}"),
            },
        })
    }
}

/// Per-seed results for `problems`, `(problems, seeds)`: seed 0 is the searched path; optimized
/// seeds follow, invalid where a problem was not optimized.
#[pyclass(module = "batchplan", frozen)]
pub(crate) struct PlanResult {
    pub(crate) result: batchplan::PlanResult,
    dof: usize,
    problems: Cached,
    paths: Cached,
    valid: Cached,
    length: Cached,
    failures: Cached,
}

impl PlanResult {
    /// Control points `cp` as `(points, dof)`.
    fn points<'py>(&self, py: Python<'py>, cp: &[f32]) -> PyResult<Bound<'py, PyAny>> {
        field::array(py, cp, &[cp.len() / self.dof, self.dof])
    }

    fn per_seed(&self) -> [usize; 2] {
        [self.result.problems.len(), self.result.seeds]
    }
}

#[pymethods]
impl PlanResult {
    /// The problems, in order.
    #[getter]
    fn problems<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.problems.get(py, || self.result.problems.to_py(py))
    }

    /// Seeds per problem: the searched path, then `PlanOptions.seeds` optimized ones.
    #[getter]
    fn seeds(&self) -> usize {
        self.result.seeds
    }

    /// Every seed's path as B-spline control points `(points, dof)`, problem by problem, or
    /// `None` where a seed has none.
    #[getter]
    fn paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.paths.get(py, || {
            let paths = self.result.paths.iter().map(|cp| {
                if cp.is_empty() {
                    Ok(py.None().into_bound(py))
                } else {
                    self.points(py, cp)
                }
            });
            let paths: Vec<Bound<'py, PyAny>> = paths.collect::<PyResult<_>>()?;
            Ok(PyTuple::new(py, paths)?.into_any())
        })
    }

    /// Per seed: whether its path passed validation, collision-free along it.
    #[getter]
    fn valid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = self.per_seed();
        self.valid
            .get(py, || field::array(py, &self.result.valid, &shape))
    }

    /// Per seed: joint-space path length.
    #[getter]
    fn length<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = self.per_seed();
        self.length
            .get(py, || field::array(py, &self.result.length, &shape))
    }

    /// Per problem: why it has no valid seed, or `None` where it has one.
    #[getter]
    fn failures<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.failures.get(py, || {
            let failures = self.result.failures.iter().map(|f| match f {
                Some(f) => PlanFailure::new(py, f)?.into_bound_py_any(py),
                None => Ok(py.None().into_bound(py)),
            });
            let failures: Vec<Bound<'py, PyAny>> = failures.collect::<PyResult<_>>()?;
            Ok(PyTuple::new(py, failures)?.into_any())
        })
    }

    /// The shortest valid seed's control points `(points, dof)`, or `None` if none is valid.
    fn best<'py>(&self, py: Python<'py>, problem: usize) -> PyResult<Option<Bound<'py, PyAny>>> {
        let n = self.result.problems.len();
        if problem >= n {
            return Err(PyIndexError::new_err(format!("problem {problem} of {n}")));
        }
        let best = self.result.best(problem);
        best.map(|cp| self.points(py, cp)).transpose()
    }

    /// Every problem with a valid seed, with its best path's control points `(points, dof)`;
    /// `Trajectory` times one for execution.
    fn solved<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let solved = self.result.solved().map(|s| {
            Ok(Solved {
                index: s.index,
                problem: s.problem.to_py(py)?.unbind(),
                solution: self.points(py, s.solution)?.unbind(),
            })
        });
        PyTuple::new(py, solved.collect::<PyResult<Vec<_>>>()?)
    }

    fn __len__(&self) -> usize {
        self.result.problems.len()
    }

    fn __repr__(&self) -> String {
        let solved = self.result.solved().count();
        format!(
            "PlanResult({} problems, {solved} solved, {} seeds each)",
            self.result.problems.len(),
            self.result.seeds
        )
    }
}

/// Plans a collision-free path for each problem, all in one batch: a search, shortcuts and
/// smoothing, then a B-spline fitted along it and checked; with `PlanOptions.seeds`, trajectory
/// optimization too. `PlanResult.solved` yields each solved problem's best path.
#[pyfunction]
#[pyo3(signature = (device, worlds, problems, options = None))]
pub(crate) fn plan(
    py: Python<'_>,
    device: &Device,
    worlds: &Worlds,
    problems: &Bound<'_, PyAny>,
    options: Option<&PlanOptions>,
) -> PyResult<PlanResult> {
    let problems = Vec::<batchplan::PlanProblem>::from_py(problems)?;
    let options = options.map(|o| o.0).unwrap_or_default();
    let result = py
        .detach(|| batchplan::plan(&device.device, &worlds.0, &problems, &options))
        .or_raise()?;
    Ok(PlanResult {
        result,
        dof: device.device.robot().dof(),
        problems: Cached::default(),
        paths: Cached::default(),
        valid: Cached::default(),
        length: Cached::default(),
        failures: Cached::default(),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RrtOptions>()?;
    m.add_class::<ShortcutOptions>()?;
    m.add_class::<PlanOptions>()?;
    m.add_class::<StartMotion>()?;
    m.add_class::<PlanProblem>()?;
    m.add_class::<PlanFailure>()?;
    m.add_class::<PlanResult>()?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    Ok(())
}
