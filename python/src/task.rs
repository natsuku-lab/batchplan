//! Picking objects up and setting them down, each a whole task in one call.

use batchplan::glam::Vec3;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::device::{Device, Worlds};
use crate::error::OrRaise;
use crate::field::{self, Field, tuple_field, value_class};
use crate::held::Held;
use crate::robot::Robot;
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
        /// How far toward the ends a grasp holds the object, along the side the fingers close on
        /// rather than across, as a fraction of half that side, in [0, 1): 0, the middle. Above
        /// 0, each grasp comes once toward each end; a hand that will pass the object to another
        /// (`handover`) holds it toward one end, as 0.75.
        along: f32,
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
        /// Where the cap holds (m): within this distance of the hand's position at the grasp or
        /// release, which the move in ends at and the move away starts from, the rest of each
        /// move as fast as the joints' limits allow (`HandSpeed.near`). `None`: along all of both.
        tool_speed_near: Option<f32>,
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
        /// The world's index in the batch: the one the hand grasps and moves in in, and every
        /// other step's unless that names its own.
        world: u32,
        /// The world the start is checked and the transit planned in, where it is another: as
        /// many obstacles as `world` has, in its order, shaped or touched otherwise.
        transit_world: Option<u32>,
        /// The world the object is lifted in, where it is another, likewise.
        depart_world: Option<u32>,
        /// Where the robot starts, at rest.
        start: Vec<f32>,
        /// Whether the path from the start to where the move in begins is planned: `True`.
        /// Without it the plans hold the two straight moves alone (`TaskPlan.begin` is where a
        /// way to them ends), and the start, which is not checked for collision, only says which
        /// solution of a grasp is nearest and where the joints other hands move wait.
        transit: bool,
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
        /// move in begins where they break stops at the transit. A task without its transit has
        /// no use for them.
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
        /// The world's index in the batch: the one the hand moves in in to where it lets go, and
        /// every other step's unless that names its own. The held object is not one of its
        /// obstacles.
        world: u32,
        /// The world the start is checked and the transit planned in, where it is another: as
        /// many obstacles as `world` has, in its order, shaped or touched otherwise.
        transit_world: Option<u32>,
        /// The world the hand backs off in, where it is another, likewise.
        depart_world: Option<u32>,
        /// Where the robot starts, at rest, holding the object.
        start: Vec<f32>,
        /// Whether the path from the start to where the move in begins is planned: `True`.
        /// Without it the plans hold the two straight moves alone (`TaskPlan.begin` is where a
        /// way to them ends), and the start, which is not checked for collision, only says which
        /// solution of a release is nearest and where the joints other hands move wait, unless
        /// `idle` says so.
        transit: bool,
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
        /// keep them; a pose whose move in begins where they break stops at the transit. A task
        /// without its transit has no use for them.
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

value_class! {
    /// One hand passing the object it holds to another hand of the same robot.
    Handover(batchplan::Handover) {
        /// The world's index in the batch. The held object is not one of its obstacles.
        world: u32,
        /// Where the robot starts, at rest, the giving hand holding the object.
        start: Vec<f32>,
        /// What the giving hand holds, and how: a pick's `TaskPlan.held`.
        held: batchplan::Held,
        /// The hand that takes it.
        to: usize,
        /// Poses of the object's frame where it changes hands, in the robot's frame; any one will
        /// do. `handover_poses` proposes some where both arms reach.
        meetings: Vec<batchplan::Pose>,
        /// Poses of the taking hand's IK frame that grasp the object, in the object's frame.
        /// `handover_grasps` turns the giver's grasp to the object's other end.
        grasps: Vec<batchplan::Pose>,
        /// How far back along its own z axis the taking hand begins to move in (meters).
        approach: f32,
        /// How far back along its own z axis the giving hand moves once it lets go (meters).
        retreat: f32,
        /// Where the hands must stay on the way to the meeting (`HandConstraint`).
        constraints: Vec<batchplan::HandConstraint>,
    }
    methods {
        /// `held` passed from its hand to hand `to` from `start` in world `world`: at the meetings
        /// `handover_poses` proposes, by the grasps `handover_grasps` gives, the taker moving in
        /// 10 cm and the giver backing off 10 cm, each along its z axis.
        #[new]
        #[pyo3(signature = (robot, world, start, held, to, **changes))]
        fn new(
            robot: &Robot,
            world: u32,
            start: &Bound<'_, PyAny>,
            held: &Held,
            to: usize,
            changes: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Self> {
            let handover = batchplan::Handover::new(
                &robot.0,
                world,
                Vec::<f32>::from_py(start)?,
                held.0.clone(),
                to,
            );
            Ok(Self(Self::changed(handover, changes)?))
        }
    }
}

tuple_field!(batchplan::Pick, batchplan::Place, batchplan::Handover);

/// A task's steps, in order.
#[pyclass(module = "batchplan", frozen, eq, eq_int, ord, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
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

    /// The planned path from the start to where the move in begins; `None` for a task that
    /// leaves its transit out.
    #[getter]
    fn transit(&self) -> Option<Trajectory> {
        self.0.transit.clone().map(Trajectory)
    }

    /// Its motions in the order they run: the transit if it has one, the move in, the move away.
    #[getter]
    fn motions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let motions: Vec<Trajectory> = self.0.motions().cloned().map(Trajectory).collect();
        PyTuple::new(py, motions)
    }

    /// Where the move in begins, its first control point: where the transit ends, or a way
    /// planned apart must.
    #[getter]
    fn begin<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.begin().to_vec().to_py(py)
    }

    /// The straight move in. The gripper closes (pick) or opens (place) at its end; in a
    /// handover the taker's closes, then the giver's opens.
    #[getter]
    fn approach(&self) -> Trajectory {
        Trajectory(self.0.approach.clone())
    }

    /// The straight move away: holding the object after a pick; the giver's after a handover.
    #[getter]
    fn depart(&self) -> Trajectory {
        Trajectory(self.0.depart.clone())
    }

    /// After a pick, what the hand holds; after a handover, what the taker holds.
    #[getter]
    fn held(&self) -> Option<Held> {
        self.0.held.clone().map(Held)
    }

    /// After a place, the object where it was set down, to add to the world.
    #[getter]
    fn placed(&self) -> Option<Obstacle> {
        self.0.placed.clone().map(Obstacle)
    }

    /// After a handover, the object's pose where it changed hands, in the robot's frame.
    #[getter]
    fn meeting(&self) -> Option<Pose> {
        self.0.meeting.map(Pose)
    }

    /// Seconds its motions take, one after another.
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
    /// is set down, and in a handover, the object where it changes hands.
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

/// Passes each task's held object to the hand that takes it, trying each meeting with each of the
/// taker's grasps: candidate `meeting * len(grasps) + grasp`. The object stands where it changes
/// hands, the giver's hand and the taker's fingers free to touch it; each hand's straight move
/// holds the other's joints still. Run the transit and the approach, close the taker's gripper,
/// open the giver's, then run the departure. `device` is the robot's without the object.
#[pyfunction]
#[pyo3(signature = (device, worlds, handovers, options = None))]
pub(crate) fn handover(
    py: Python<'_>,
    device: &Device,
    worlds: &Worlds,
    handovers: &Bound<'_, PyAny>,
    options: Option<&TaskOptions>,
) -> PyResult<TaskResult> {
    let handovers = Vec::<batchplan::Handover>::from_py(handovers)?;
    let options = options.map(|o| o.0.clone()).unwrap_or_default();
    let result = py
        .detach(|| batchplan::handover(&device.device, &worlds.0, &handovers, &options))
        .or_raise()?;
    TaskResult::new(py, result)
}

/// Grasps for the hand that takes `held`'s object, for `Handover.grasps`, at the object's other
/// end: the giver's grasp turned half about the axis nearest its approach other than the one
/// toward the end it holds (from the same side), and about the axis across both (from the other
/// side).
#[pyfunction]
pub(crate) fn handover_grasps<'py>(py: Python<'py>, held: &Held) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        batchplan::handover_grasps(&held.0).into_iter().map(Pose),
    )
}

/// Poses of `held`'s object where hand `to` may take it, for `Handover.meetings`: where both arms
/// reach, in front of and halfway between their shoulders as `start` has them, the object at its
/// tilt in the giver's hand, its held end facing the giver.
#[pyfunction]
pub(crate) fn handover_poses<'py>(
    py: Python<'py>,
    robot: &Robot,
    start: &Bound<'_, PyAny>,
    held: &Held,
    to: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let start = Vec::<f32>::from_py(start)?;
    let poses = batchplan::handover_poses(&robot.0, &start, &held.0, to);
    PyTuple::new(py, poses.into_iter().map(Pose))
}

/// Grasps of a box, ball, cylinder or capsule in its frame, for `Pick.grasps`: the hand facing
/// each side and closing across what fits between its fingers. With a `floor`, the object's pose
/// places it: pass a world's obstacle as it stands. Raises `InputError` saying why where no grasp
/// fits.
#[pyfunction]
#[pyo3(signature = (object, options = None))]
pub(crate) fn grasps<'py>(
    py: Python<'py>,
    object: &Obstacle,
    options: Option<&GraspOptions>,
) -> PyResult<Bound<'py, PyTuple>> {
    let options = options.map(|o| o.0).unwrap_or_default();
    let found = batchplan::grasps(&object.0, &options).or_raise()?;
    PyTuple::new(py, found.into_iter().map(Pose))
}

/// Poses that set `object` down at `pose` alike, for `Place.poses`: `pose` turned about the
/// object's own z axis by its symmetries. A box gives 2, or 4 where its x and y sides match; a
/// ball, cylinder or capsule `turns` (at least 1); a distance grid `pose` alone.
#[pyfunction]
pub(crate) fn place_poses<'py>(
    py: Python<'py>,
    object: &Obstacle,
    pose: &Pose,
    turns: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let poses = batchplan::place_poses(&object.0, pose.0, turns);
    PyTuple::new(py, poses.into_iter().map(Pose))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GraspOptions>()?;
    m.add_class::<TaskOptions>()?;
    m.add_class::<Pick>()?;
    m.add_class::<Place>()?;
    m.add_class::<Handover>()?;
    m.add_class::<TaskStep>()?;
    m.add_class::<TaskPlan>()?;
    m.add_class::<CandidateStop>()?;
    m.add_class::<TaskFailure>()?;
    m.add_class::<TaskResult>()?;
    m.add_function(wrap_pyfunction!(pick, m)?)?;
    m.add_function(wrap_pyfunction!(place, m)?)?;
    m.add_function(wrap_pyfunction!(handover, m)?)?;
    m.add_function(wrap_pyfunction!(handover_grasps, m)?)?;
    m.add_function(wrap_pyfunction!(handover_poses, m)?)?;
    m.add_function(wrap_pyfunction!(grasps, m)?)?;
    m.add_function(wrap_pyfunction!(place_poses, m)?)?;
    Ok(())
}
