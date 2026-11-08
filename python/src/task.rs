//! Picking objects up and setting them down, each a whole task in one call.

use batchplan::glam::Vec3;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::device::{Device, Worlds};
use crate::error::OrRaise;
use crate::field::{self, Field, tuple_field, value_class};
use crate::held::Held;
use crate::timing::Trajectory;
use crate::world::{Obstacle, Pose};

value_class! {
    /// Grasps `grasps` proposes for a parallel gripper.
    GraspOptions(batchplan::GraspOptions) {
        /// The widest the fingers open (meters): 8 cm.
        opening: f32,
        /// How far in from the object's surface the IK frame holds it (meters): 2 cm. Above a
        /// `floor`, a grasp from above may hold it less deep, its frame even outside the object.
        depth: f32,
        /// Grasps spread around a round object's axis: 8.
        turns: usize,
        /// How far the fingers reach past the IK frame along its z axis (meters), as `Robot.tip`
        /// measures it: 0.
        tip: f32,
        /// A height the hand's frame and its open fingers stay above, in the frame the object is
        /// placed in (meters): the top of the table it stands on, and room to spare. A grasp from
        /// above that would take them lower holds the object less deep, and one that leaves less
        /// than `overlap` of the fingers in it is left out, as are grasps from below and from the
        /// side whose lower finger would be under the floor. `None` for no floor.
        floor: Option<f32>,
        /// The least the fingers reach into the object, from their tips, in a grasp from above
        /// over a `floor` (meters): 1 cm. An object whose top is less than this over the floor has
        /// no grasp from above.
        overlap: f32,
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
    /// How `pick` and `place` solve, follow, plan and time their steps.
    TaskOptions(batchplan::TaskOptions) {
        /// IK at each grasp or release.
        ik: batchplan::IkOptions,
        /// The planned paths from the start.
        plan: batchplan::PlanOptions,
        /// The straight moves.
        tool_path: batchplan::ToolPathOptions,
        /// The hand's speed cap on the straight moves (m/s): 10 cm/s. `math.inf` leaves it
        /// uncapped, the moves as fast as the joints' limits allow.
        tool_speed: f32,
        /// How much of each straight move a candidate must make, in (0, 1]: 1, all of it. Less
        /// lets an obstacle shorten the move in or away rather than stop the candidate; a move
        /// stopped short for any other reason still stops it.
        min_fraction: f32,
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
    /// A hand picking up one obstacle of its world.
    Pick(batchplan::Pick) {
        /// The world's index in the batch.
        world: u32,
        /// Where the robot starts, at rest.
        start: Vec<f32>,
        /// The hand that picks it up.
        hand: usize,
        /// The object: an obstacle of the world, by index. A distance grid cannot be picked up.
        object: usize,
        /// Poses of the hand's IK frame that grasp the object, in the object's frame (its center
        /// and rotation); any one will do. `grasps` proposes some for a box, ball, cylinder or
        /// capsule.
        grasps: Vec<batchplan::Pose>,
        /// How far back along its own z axis the hand begins to move in to a grasp (meters).
        approach: f32,
        /// Where the hand takes the object once it holds it, in the world's frame (meters).
        lift: Vec3,
        /// Obstacles the object rests on, by index: it may touch them as it leaves.
        support: Vec<usize>,
        /// Where the hands must stay along the transit. The start must keep them; a grasp whose
        /// move in begins where they break stops at the transit.
        constraints: Vec<batchplan::HandConstraint>,
    }
    methods {
        /// Hand 0 picks up `object` of world `world` from `start` by any of `grasps`, moving in
        /// 10 cm along the hand's z axis and lifting the object 10 cm, off nothing it rests on.
        #[new]
        #[pyo3(signature = (world, start, object, grasps, **changes))]
        fn new(
            world: u32,
            start: &Bound<'_, PyAny>,
            object: usize,
            grasps: &Bound<'_, PyAny>,
            changes: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Self> {
            let pick = batchplan::Pick::new(
                world,
                Vec::<f32>::from_py(start)?,
                object,
                Vec::from_py(grasps)?,
            );
            Ok(Self(Self::changed(pick, changes)?))
        }
    }
}

value_class! {
    /// A hand setting down the object it holds.
    Place(batchplan::Place) {
        /// The world's index in the batch. The held object is not one of its obstacles.
        world: u32,
        /// Where the robot starts, at rest, holding the object.
        start: Vec<f32>,
        /// Where the joints only other hands move go on the transit, as a configuration of which
        /// only those joints count: another arm going home while this one carries the object.
        /// `None` keeps them where `start` has them.
        idle: Option<Vec<f32>>,
        /// What the hand holds, and how: a pick's `TaskPlan.held`.
        held: batchplan::Held,
        /// Poses of the object's frame to set it down at; any one will do. `place_poses` gives an
        /// object's symmetric turns at one spot.
        poses: Vec<batchplan::Pose>,
        /// How the hand moves in to where it lets go, in the world's frame (meters).
        lower: Vec3,
        /// How far back along its own z axis the hand moves once it lets go (meters).
        retreat: f32,
        /// Obstacles the object is set on, by index: it may touch them as it arrives.
        support: Vec<usize>,
        /// Where the hands must stay along the transit, as a cup carried upright. The start must
        /// keep them; a pose whose move in begins where they break stops at the transit.
        constraints: Vec<batchplan::HandConstraint>,
    }
    methods {
        /// Sets `held` down from `start` at any of `poses` in world `world`, lowering it the last
        /// 10 cm and backing off 10 cm along the hand's z axis, onto nothing.
        #[new]
        #[pyo3(signature = (world, start, held, poses, **changes))]
        fn new(
            world: u32,
            start: &Bound<'_, PyAny>,
            held: &Held,
            poses: &Bound<'_, PyAny>,
            changes: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Self> {
            let place = batchplan::Place::new(
                world,
                Vec::<f32>::from_py(start)?,
                held.0.clone(),
                Vec::from_py(poses)?,
            );
            Ok(Self(Self::changed(place, changes)?))
        }
    }
}

tuple_field!(batchplan::Pick, batchplan::Place);

/// A task's steps, in order.
#[pyclass(module = "batchplan", frozen, eq, eq_int, ord)]
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub(crate) enum TaskStep {
    /// The start, checked for collision.
    Start,
    /// IK at the grasp, or where the hand lets go.
    Reach,
    /// The straight move in, to there.
    Approach,
    /// The straight move away: lifting the object, or backing off from it.
    Depart,
    /// The planned path from the start to where the move in begins.
    Transit,
}

impl TaskStep {
    fn new(step: batchplan::TaskStep) -> PyResult<Self> {
        Ok(match step {
            batchplan::TaskStep::Start => Self::Start,
            batchplan::TaskStep::Reach => Self::Reach,
            batchplan::TaskStep::Approach => Self::Approach,
            batchplan::TaskStep::Depart => Self::Depart,
            batchplan::TaskStep::Transit => Self::Transit,
            other => {
                return Err(crate::error::Error::new_err(format!(
                    "batchplan reported a step these bindings don't know: {other:?}"
                )));
            }
        })
    }

    fn rust(self) -> batchplan::TaskStep {
        match self {
            Self::Start => batchplan::TaskStep::Start,
            Self::Reach => batchplan::TaskStep::Reach,
            Self::Approach => batchplan::TaskStep::Approach,
            Self::Depart => batchplan::TaskStep::Depart,
            Self::Transit => batchplan::TaskStep::Transit,
        }
    }
}

/// A task's motions by one candidate: run the transit and the approach, close (pick) or open
/// (place) the gripper, then run the departure.
#[pyclass(module = "batchplan", frozen)]
pub(crate) struct TaskPlan(batchplan::TaskPlan);

#[pymethods]
impl TaskPlan {
    /// The candidate: its index among the task's grasps or poses.
    #[getter]
    fn candidate(&self) -> usize {
        self.0.candidate
    }

    /// The planned path from the start to where the move in begins.
    #[getter]
    fn transit(&self) -> Trajectory {
        Trajectory(self.0.transit.clone())
    }

    /// The straight move in. The gripper closes (pick) or opens (place) at its end.
    #[getter]
    fn approach(&self) -> Trajectory {
        Trajectory(self.0.approach.clone())
    }

    /// The straight move away: holding the object after a pick.
    #[getter]
    fn depart(&self) -> Trajectory {
        Trajectory(self.0.depart.clone())
    }

    /// After a pick, what the hand holds.
    #[getter]
    fn held(&self) -> Option<Held> {
        self.0.held.clone().map(Held)
    }

    /// After a place, the object where it was set down, to add to the world.
    #[getter]
    fn placed(&self) -> Option<Obstacle> {
        self.0.placed.clone().map(Obstacle)
    }

    /// Seconds the three motions take, one after another.
    #[getter]
    fn duration(&self) -> f64 {
        field::shortest(self.0.duration())
    }

    /// Where the departure comes to rest, its last control point: the start of the next task.
    #[getter]
    fn end<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.end().to_vec().to_py(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "TaskPlan(candidate={}, duration={} s)",
            self.0.candidate,
            self.0.duration()
        )
    }
}

/// Where one candidate stopped, and why.
#[pyclass(module = "batchplan", frozen)]
pub(crate) struct CandidateStop(batchplan::CandidateStop);

#[pymethods]
impl CandidateStop {
    /// The step it did not get past.
    #[getter]
    fn step(&self) -> PyResult<TaskStep> {
        TaskStep::new(self.0.step)
    }

    /// What stopped it there: a pose out of reach, a collision or a jump at the start, at IK and
    /// at the straight moves; `None` for a transit no plan was found for.
    #[getter]
    fn why<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.why.to_py(py)
    }

    /// For a collision, what collides, nearest first. Empty for the rest. Obstacles are
    /// numbered as in the task's world; in a place, `len(world.obstacles)` is the object where it
    /// is set down.
    #[getter]
    fn contacts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.contacts.to_py(py)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "CandidateStop(step={:?}, why={})",
            self.0.step,
            self.0.why.to_py(py)?.repr()?
        ))
    }
}

/// Why none of a task's candidates got through: each one's step and reason, in the task's order.
/// `str()` of it is a sentence saying where they stopped.
#[pyclass(module = "batchplan", frozen)]
pub(crate) struct TaskFailure(batchplan::TaskFailure);

#[pymethods]
impl TaskFailure {
    /// Per candidate: where it stopped.
    #[getter]
    fn candidates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let stops = self.0.candidates.iter().cloned().map(CandidateStop);
        PyTuple::new(py, stops)
    }

    /// The furthest step a candidate got to.
    #[getter]
    fn step(&self) -> PyResult<TaskStep> {
        TaskStep::new(self.0.step())
    }

    /// How many candidates stopped at `step`.
    fn at(&self, step: &TaskStep) -> usize {
        self.0.at(step.rust())
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("TaskFailure({:?})", self.0.to_string())
    }
}

/// What `pick` or `place` did with each task: per task, the motions of every candidate that got
/// through, quickest first (a tuple of `TaskPlan`), or where each stopped (a `TaskFailure`).
#[pyclass(module = "batchplan", frozen)]
pub(crate) struct TaskResult {
    problems: Py<PyAny>,
    outcomes: Py<PyTuple>,
}

impl TaskResult {
    fn new<T: Field>(py: Python<'_>, result: batchplan::TaskResult<T>) -> PyResult<Self>
    where
        Vec<T>: Field,
    {
        let batchplan::TaskResult {
            problems, outcomes, ..
        } = result;
        let outcomes = outcomes.into_iter().map(|o| match o {
            Ok(plans) => {
                let plans: Vec<TaskPlan> = plans.into_iter().map(TaskPlan).collect();
                PyTuple::new(py, plans)?.into_bound_py_any(py)
            }
            Err(failure) => TaskFailure(failure).into_bound_py_any(py),
        });
        let outcomes: Vec<Bound<'_, PyAny>> = outcomes.collect::<PyResult<_>>()?;
        Ok(Self {
            problems: problems.to_py(py)?.unbind(),
            outcomes: PyTuple::new(py, outcomes)?.unbind(),
        })
    }
}

#[pymethods]
impl TaskResult {
    /// The tasks, in order.
    #[getter]
    fn problems(&self, py: Python<'_>) -> Py<PyAny> {
        self.problems.clone_ref(py)
    }

    /// Per task: a tuple of `TaskPlan`, quickest first, or a `TaskFailure`.
    #[getter]
    fn outcomes(&self, py: Python<'_>) -> Py<PyTuple> {
        self.outcomes.clone_ref(py)
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.outcomes.bind(py).len()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let outcomes = self.outcomes.bind(py);
        let failed = outcomes
            .iter()
            .filter(|o| o.is_instance_of::<TaskFailure>())
            .count();
        Ok(format!(
            "TaskResult({} tasks, {} through)",
            outcomes.len(),
            outcomes.len() - failed
        ))
    }
}

/// Picks up each task's object, trying each of its grasps. IK solves at the grasp, where the
/// fingers may touch the object, and the move in runs backward from there. The move away holds the
/// object, which may touch what it rests on. Run the transit and the approach, close the gripper,
/// then run the departure holding the object (`Held.attachment`).
#[pyfunction]
#[pyo3(signature = (device, worlds, picks, options = None))]
pub(crate) fn pick(
    py: Python<'_>,
    device: &Device,
    worlds: &Worlds,
    picks: &Bound<'_, PyAny>,
    options: Option<&TaskOptions>,
) -> PyResult<TaskResult> {
    let picks = Vec::<batchplan::Pick>::from_py(picks)?;
    let options = options.map(|o| o.0.clone()).unwrap_or_default();
    let result = py
        .detach(|| batchplan::pick(&device.device, &worlds.0, &picks, &options))
        .or_raise()?;
    TaskResult::new(py, result)
}

/// Sets down each task's held object, trying each of its poses. IK solves where the hand lets go,
/// holding the object, which may touch what it is set on there and on the move in, run backward
/// from there. The move away lets the fingers touch the object where it now stands. Run the
/// transit and the approach holding the object, open the gripper, then run the departure. `device`
/// is the robot's without the object; each task's `held` is attached to it here.
#[pyfunction]
#[pyo3(signature = (device, worlds, places, options = None))]
pub(crate) fn place(
    py: Python<'_>,
    device: &Device,
    worlds: &Worlds,
    places: &Bound<'_, PyAny>,
    options: Option<&TaskOptions>,
) -> PyResult<TaskResult> {
    let places = Vec::<batchplan::Place>::from_py(places)?;
    let options = options.map(|o| o.0.clone()).unwrap_or_default();
    let result = py
        .detach(|| batchplan::place(&device.device, &worlds.0, &places, &options))
        .or_raise()?;
    TaskResult::new(py, result)
}

/// Grasps of a box, ball, cylinder or capsule in its frame, for `Pick.grasps`: the hand facing
/// each side and closing across what fits between its fingers. With a `floor`, the object's pose
/// places it: pass a world's obstacle as it stands. Raises `InputError` saying why where no grasp
/// fits.
#[pyfunction]
#[pyo3(signature = (object, options = None))]
pub(crate) fn grasps(object: &Obstacle, options: Option<&GraspOptions>) -> PyResult<Vec<Pose>> {
    let options = options.map(|o| o.0).unwrap_or_default();
    let found = batchplan::grasps(&object.0, &options).or_raise()?;
    Ok(found.into_iter().map(Pose).collect())
}

/// Poses that set `object` down at `pose` alike, for `Place.poses`: `pose` turned about the
/// object's own z axis by its symmetries. A box gives 2, or 4 where its x and y sides match; a
/// ball, cylinder or capsule `turns` (at least 1); a distance grid `pose` alone.
#[pyfunction]
pub(crate) fn place_poses(object: &Obstacle, pose: &Pose, turns: usize) -> Vec<Pose> {
    let poses = batchplan::place_poses(&object.0, pose.0, turns);
    poses.into_iter().map(Pose).collect()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GraspOptions>()?;
    m.add_class::<TaskOptions>()?;
    m.add_class::<Pick>()?;
    m.add_class::<Place>()?;
    m.add_class::<TaskStep>()?;
    m.add_class::<TaskPlan>()?;
    m.add_class::<CandidateStop>()?;
    m.add_class::<TaskFailure>()?;
    m.add_class::<TaskResult>()?;
    m.add_function(wrap_pyfunction!(pick, m)?)?;
    m.add_function(wrap_pyfunction!(place, m)?)?;
    m.add_function(wrap_pyfunction!(grasps, m)?)?;
    m.add_function(wrap_pyfunction!(place_poses, m)?)?;
    Ok(())
}
