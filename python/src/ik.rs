//! Inverse kinematics.

use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::device::{CollisionWeights, Device, Worlds};
use crate::error::OrRaise;
use crate::field::{self, Field, tuple_field, value_class};
use crate::solved::{Cached, Solved};
use crate::world::Pose;

value_class! {
    /// How `solve_ik` searches.
    IkOptions(batchplan::IkOptions) {
        /// Starting configurations per problem: the problem's seed or the robot's default
        /// configuration, then random ones within the joint limits.
        seeds: usize,
        /// Damped-least-squares steps per seed, at most: one that has come to rest takes no more.
        iterations: u32,
        /// Of `iterations`, the steps every seed takes before only `kept` of each problem's go
        /// on; 0 (or `iterations` and more) for every seed to take them all.
        screening: u32,
        /// Seeds per problem that go on past `screening`: of those not yet at their targets
        /// within the tolerances and clear of collision, the nearest, the collision-free first.
        /// The others are returned as the screening left them.
        kept: usize,
        /// Damped-least-squares damping (lambda).
        damping: f32,
        /// Meters of position error equivalent to one radian of rotation error.
        rotation_weight: f32,
        /// Largest joint change per iteration (radians).
        max_step: f32,
        /// Step along the null-space-projected collision gradient.
        collision_step: f32,
        /// The collision cost that gradient comes from.
        collision: batchplan::CollisionWeights,
        /// Largest position error of a successful seed (meters).
        position_tolerance: f32,
        /// Largest rotation error of a successful seed (radians).
        rotation_tolerance: f32,
        /// Seed of the random starting configurations. Each problem draws its own from it and its
        /// targets and seed, so a problem's starting configurations do not depend on the others in
        /// its batch.
        rng_seed: u64,
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
    /// Target poses for a robot's hands, in one world.
    IkProblem(batchplan::IkProblem) {
        /// The world's index in the batch.
        world: u32,
        /// Each hand's target pose (see `RobotOptions.ee_links`), or `None` to hold the joints
        /// only that hand moves where the seed has them, as an arm that waits.
        targets: Vec<Option<batchplan::Pose>>,
        /// Where seed 0 starts instead of the robot's default configuration: the current state,
        /// for a target close to it (see `IkResult.nearest`). Every seed starts held joints here.
        seed: Option<Vec<f32>>,
        /// Joints that stay where the seed has them, in every seed, even where a targeted hand
        /// moves on them: a mobile base placed beside a table while its arms reach.
        hold: Vec<usize>,
        /// An object a hand holds, as a pick leaves it: checked as part of the robot, as
        /// attaching it (`Held.attachment`) would check it. `None` holds nothing.
        holding: Option<batchplan::Held>,
    }
    methods {
        /// A one-handed robot's hand at `target` in world `world`; for several hands, give
        /// `targets` instead.
        #[new]
        #[pyo3(signature = (world, target = None, **changes))]
        fn new(
            world: u32,
            target: Option<&Pose>,
            changes: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Self> {
            let targets_too = changes.is_some_and(|c| c.contains("targets").unwrap_or(false));
            let both = target.is_some() && targets_too;
            if both {
                return Err(PyTypeError::new_err(
                    "give `target` for one hand or `targets` for several, not both",
                ));
            }
            let mut problem = batchplan::IkProblem::new(world, batchplan::Pose::IDENTITY);
            problem.targets = target.map(|t| Some(t.0)).into_iter().collect();
            Ok(Self(Self::changed(problem, changes)?))
        }
    }
}

tuple_field!(batchplan::IkProblem);

/// Per-seed results for `problems`: arrays are `(problems, seeds, ...)`.
#[pyclass(module = "batchplan", frozen)]
pub(crate) struct IkResult {
    pub(crate) result: batchplan::IkResult,
    problems: Cached,
    q: Cached,
    position_error: Cached,
    rotation_error: Cached,
    world_clearance: Cached,
    self_clearance: Cached,
    success: Cached,
}

impl IkResult {
    fn new(result: batchplan::IkResult) -> Self {
        Self {
            result,
            problems: Cached::default(),
            q: Cached::default(),
            position_error: Cached::default(),
            rotation_error: Cached::default(),
            world_clearance: Cached::default(),
            self_clearance: Cached::default(),
            success: Cached::default(),
        }
    }

    fn per_item(&self) -> [usize; 2] {
        [self.result.problems.len(), self.result.seeds]
    }

    fn check_problem(&self, problem: usize) -> PyResult<()> {
        let n = self.result.problems.len();
        if problem < n {
            Ok(())
        } else {
            Err(PyIndexError::new_err(format!("problem {problem} of {n}")))
        }
    }
}

#[pymethods]
impl IkResult {
    /// The problems, in order.
    #[getter]
    fn problems<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.problems.get(py, || self.result.problems.to_py(py))
    }

    /// Joints per configuration.
    #[getter]
    fn dof(&self) -> usize {
        self.result.dof
    }

    /// Seeds per problem.
    #[getter]
    fn seeds(&self) -> usize {
        self.result.seeds
    }

    /// Each seed's final configuration, `(problems, seeds, dof)`.
    #[getter]
    fn q<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let [p, s] = self.per_item();
        self.q.get(py, || {
            field::array(py, &self.result.q, &[p, s, self.result.dof])
        })
    }

    /// Per seed: meters from the target, the worst over the targeted hands.
    #[getter]
    fn position_error<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = self.per_item();
        self.position_error
            .get(py, || field::array(py, &self.result.position_error, &shape))
    }

    /// Per seed: radians from the target, the worst over the targeted hands.
    #[getter]
    fn rotation_error<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = self.per_item();
        self.rotation_error
            .get(py, || field::array(py, &self.result.rotation_error, &shape))
    }

    /// Per seed: meters from the nearest obstacle.
    #[getter]
    fn world_clearance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = self.per_item();
        self.world_clearance.get(py, || {
            field::array(py, &self.result.world_clearance, &shape)
        })
    }

    /// Per seed: meters between the nearest checked pair of the robot's links.
    #[getter]
    fn self_clearance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = self.per_item();
        self.self_clearance
            .get(py, || field::array(py, &self.result.self_clearance, &shape))
    }

    /// Per seed: within both tolerances, and collision-free.
    #[getter]
    fn success<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = self.per_item();
        self.success
            .get(py, || field::array(py, &self.result.success, &shape))
    }

    /// The successful seed with the most obstacle clearance, or `None` if none succeeded.
    fn best<'py>(&self, py: Python<'py>, problem: usize) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.check_problem(problem)?;
        let best = self.result.best(problem);
        best.map(|q| field::array(py, q, &[q.len()])).transpose()
    }

    /// The successful seed of `problem` nearest `q` (Euclidean in joint space), which keeps a
    /// small move on the same IK branch; `None` if none succeeded.
    fn nearest<'py>(
        &self,
        py: Python<'py>,
        problem: usize,
        q: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.check_problem(problem)?;
        let q = field::floats_of(q, self.result.dof, "a configuration")?;
        let nearest = self.result.nearest(problem, &q);
        nearest.map(|q| field::array(py, q, &[q.len()])).transpose()
    }

    /// Every problem with a successful seed, with its best configuration.
    fn solved<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let solved = self.result.solved().map(|s| {
            Ok(Solved {
                index: s.index,
                problem: s.problem.to_py(py)?.unbind(),
                solution: field::array(py, s.solution, &[s.solution.len()])?.unbind(),
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
            "IkResult({} problems, {solved} solved, {} seeds each)",
            self.result.problems.len(),
            self.result.seeds
        )
    }
}

/// Why a problem of an `IkResult` has no solution.
#[pyclass(module = "batchplan", frozen)]
pub(crate) enum IkFailure {
    /// No seed reaches the targets within the tolerances, even without the push away from
    /// obstacles: they lie beyond the arm's reach or its joint limits, or need more seeds or
    /// iterations than were given.
    Unreachable(),
    /// Seeds reach the targets, but only in collision: what the one with the most clearance
    /// collides with, nearest first.
    Collides { contacts: Py<PyAny> },
    /// A reason these bindings don't know yet, as batchplan describes it.
    Other { description: String },
}

#[pymethods]
impl IkFailure {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(match self {
            Self::Unreachable() => "IkFailure.Unreachable()".to_owned(),
            Self::Collides { contacts } => {
                format!("IkFailure.Collides(contacts={})", contacts.bind(py).repr()?)
            }
            Self::Other { description } => format!("IkFailure.Other({description:?})"),
        })
    }
}

/// Collision-free IK from many seeds per problem, all in one batch.
#[pyfunction]
#[pyo3(signature = (device, worlds, problems, options = None))]
pub(crate) fn solve_ik(
    py: Python<'_>,
    device: &Device,
    worlds: &Worlds,
    problems: &Bound<'_, PyAny>,
    options: Option<&IkOptions>,
) -> PyResult<IkResult> {
    let problems = Vec::<batchplan::IkProblem>::from_py(problems)?;
    let options = options.map(|o| o.0).unwrap_or_default();
    let result = py
        .detach(|| batchplan::solve_ik(&device.device, &worlds.0, &problems, &options))
        .or_raise()?;
    Ok(IkResult::new(result))
}

/// Why each problem of `result`, solved with `options`, has no solution: `None` where it has
/// one. Problems no seed reached are solved once more without the push away from obstacles: a
/// pose reached then collides, one still not reached is out of reach. That solve costs what theirs
/// did, so it runs only when asked for, here.
#[pyfunction]
#[pyo3(name = "ik_failures", signature = (device, worlds, result, options = None))]
pub(crate) fn failures<'py>(
    py: Python<'py>,
    device: &Device,
    worlds: &Worlds,
    result: &IkResult,
    options: Option<&IkOptions>,
) -> PyResult<Bound<'py, PyTuple>> {
    let options = options.map(|o| o.0).unwrap_or_default();
    let failures = py
        .detach(|| batchplan::ik::failures(&device.device, &worlds.0, &result.result, &options))
        .or_raise()?;
    let mirrored = failures.into_iter().map(|f| {
        f.map(|f| {
            Ok(match f {
                batchplan::IkFailure::Unreachable => IkFailure::Unreachable(),
                batchplan::IkFailure::Collides { contacts } => IkFailure::Collides {
                    contacts: contacts.to_py(py)?.unbind(),
                },
                other => IkFailure::Other {
                    description: format!("{other:?}"),
                },
            })
        })
        .transpose()
    });
    PyTuple::new(py, mirrored.collect::<PyResult<Vec<_>>>()?)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IkOptions>()?;
    m.add_class::<IkProblem>()?;
    m.add_class::<IkResult>()?;
    m.add_class::<IkFailure>()?;
    m.add_class::<CollisionWeights>()?;
    m.add_function(wrap_pyfunction!(solve_ik, m)?)?;
    m.add_function(wrap_pyfunction!(failures, m)?)?;
    Ok(())
}
