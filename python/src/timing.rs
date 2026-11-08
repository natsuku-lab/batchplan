//! Timing paths into trajectories.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::OrRaise;
use crate::field::{self, Field, value_class};
use crate::plan::StartMotion;
use crate::robot::Robot;

value_class! {
    /// Joint position, velocity and acceleration at one instant.
    JointState(batchplan::JointState) {
        /// Per joint, radians (meters for prismatic joints).
        position: Vec<f32>,
        /// Per joint, per second.
        velocity: Vec<f32>,
        /// Per joint, per second squared.
        acceleration: Vec<f32>,
    }
    methods {
        /// The state at `position`, with velocity and acceleration zero where not given.
        #[new]
        #[pyo3(signature = (position, velocity = None, acceleration = None))]
        fn new(
            position: &Bound<'_, PyAny>,
            velocity: Option<&Bound<'_, PyAny>>,
            acceleration: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            let position = Vec::from_py(position)?;
            let zeros = || vec![0.0; position.len()];
            Ok(Self(batchplan::JointState {
                velocity: velocity.map(Vec::from_py).transpose()?.unwrap_or_else(zeros),
                acceleration: acceleration.map(Vec::from_py).transpose()?.unwrap_or_else(zeros),
                position,
            }))
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
}

value_class! {
    /// Caps on how fast a hand moves, on top of the joints' limits, for
    /// `Trajectory.with_hand_speed`. `math.inf` leaves one uncapped.
    HandSpeed(batchplan::HandSpeed) {
        /// The hand, as `Robot.ee_pose` numbers them.
        hand: usize,
        /// The largest speed of the hand's frame, meters per second.
        linear: f32,
        /// The largest rate at which it turns, radians per second.
        angular: f32,
    }
    methods {
        #[new]
        #[pyo3(signature = (hand, linear = f32::INFINITY, angular = f32::INFINITY))]
        fn new(hand: usize, linear: f32, angular: f32) -> Self {
            Self(batchplan::HandSpeed {
                hand,
                linear,
                angular,
            })
        }
    }
}

/// One trajectory sampled every `dt` seconds: arrays are `(samples, dof)`.
#[pyclass(module = "batchplan", frozen, get_all)]
pub(crate) struct JointTrajectory {
    /// Joints per sample.
    dof: usize,
    /// Seconds between samples.
    dt: f64,
    /// Seconds from the first sample to the last, which lands exactly on the motion's end (so the
    /// last interval may be shorter than `dt`).
    duration: f64,
    /// `(samples, dof)`, radians (meters for prismatic joints).
    positions: Py<PyAny>,
    /// `(samples, dof)`, per second.
    velocities: Py<PyAny>,
    /// `(samples, dof)`, per second squared.
    accelerations: Py<PyAny>,
}

impl JointTrajectory {
    /// Moves the samples into arrays, without copying them.
    fn new(py: Python<'_>, j: batchplan::JointTrajectory) -> PyResult<Self> {
        let rows = j.len();
        let batchplan::JointTrajectory {
            dof,
            dt,
            duration,
            positions,
            velocities,
            accelerations,
        } = j;
        let shaped = |v: Vec<f32>| -> PyResult<Py<PyAny>> {
            let flat = PyArray1::from_vec(py, v);
            Ok(field::read_only(flat.reshape([rows, dof])?.into_any())?.unbind())
        };
        Ok(Self {
            dof,
            dt: field::shortest(dt),
            duration: field::shortest(duration),
            positions: shaped(positions)?,
            velocities: shaped(velocities)?,
            accelerations: shaped(accelerations)?,
        })
    }
}

#[pymethods]
impl JointTrajectory {
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.positions.bind(py).len()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "JointTrajectory({} samples every {} s, {} s)",
            self.positions.bind(py).len()?,
            self.dt,
            self.duration
        ))
    }
}

/// A rest-to-rest joint trajectory: a path whose first three and last three control points are
/// equal, and the time map that runs along it. A trajectory from a moving start
/// (`Trajectory.moving`) starts with the motion its first control points encode; a controlled stop
/// (`Trajectory.stop`) starts partway along its path and comes to rest before its end.
/// `Trajectory.check` verifies one before it runs.
#[pyclass(module = "batchplan", frozen)]
pub(crate) struct Trajectory(pub(crate) batchplan::Trajectory);

#[pymethods]
impl Trajectory {
    /// Times a planned path (control points `(points, dof)`, at least 7) as fast as the robot's
    /// velocity, acceleration and jerk limits allow, then slows it by `speed_scale` in (0, 1].
    #[new]
    #[pyo3(signature = (robot, control_points, speed_scale = 1.0))]
    fn new(
        py: Python<'_>,
        robot: &Robot,
        control_points: &Bound<'_, PyAny>,
        speed_scale: f32,
    ) -> PyResult<Self> {
        let cp = field::floats(control_points)?;
        py.detach(|| batchplan::Trajectory::try_new(&robot.0, &cp, speed_scale))
            .or_raise()
            .map(Self)
    }

    /// Times a planned path as fast as `Trajectory(...)` does while hand `speed.hand` keeps under
    /// `speed`'s caps too, as a tool moved along a path may need. `check` checks the joints'
    /// limits only.
    #[staticmethod]
    fn with_hand_speed(
        py: Python<'_>,
        robot: &Robot,
        control_points: &Bound<'_, PyAny>,
        speed: &HandSpeed,
    ) -> PyResult<Self> {
        let cp = field::floats(control_points)?;
        py.detach(|| batchplan::Trajectory::with_hand_speed(&robot.0, &cp, &speed.0))
            .or_raise()
            .map(Self)
    }

    /// Times a path planned from a moving start (`PlanProblem.start_motion`) so that it starts
    /// with exactly that velocity and acceleration. Raises `UnsafeError` if no timing of the path
    /// from that motion keeps the limits, as when it cannot brake in time.
    #[staticmethod]
    fn moving(
        py: Python<'_>,
        robot: &Robot,
        control_points: &Bound<'_, PyAny>,
        motion: &StartMotion,
    ) -> PyResult<Self> {
        let cp = field::floats(control_points)?;
        py.detach(|| batchplan::Trajectory::moving(&robot.0, &cp, &motion.0))
            .or_raise()
            .map(Self)
    }

    /// Joints per configuration.
    #[getter]
    fn dof(&self) -> usize {
        self.0.dof
    }

    /// The path, `(points, dof)`: a uniform cubic B-spline over `s` from 0 to `points - 3` spans.
    #[getter]
    fn control_points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cp = &self.0.control_points;
        field::array(py, cp, &[cp.len() / self.0.dof, self.0.dof])
    }

    /// Seconds per span of the time map.
    #[getter]
    fn knot_interval(&self) -> f64 {
        field::shortest(self.0.knot_interval)
    }

    /// The time map's control points: a uniform cubic B-spline in `s`, from 0 at the start to the
    /// path's span count at the end (or from and to where a stop starts and ends).
    #[getter]
    fn time_points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.time_points.to_py(py)
    }

    /// Seconds from its start to where it comes to rest.
    #[getter]
    fn duration(&self) -> f64 {
        field::shortest(self.0.duration())
    }

    /// The configuration it starts at: its first control point, or where a stop begins along its
    /// path.
    #[getter]
    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.start().to_py(py)
    }

    /// The configuration it comes to rest at: its last control point, or where a stop ends,
    /// short of its path's end. The start of the motion that follows.
    #[getter]
    fn end<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.end().to_py(py)
    }

    /// The state at time `t` (clamped to the trajectory).
    fn at(&self, t: f32) -> JointState {
        let mut out = batchplan::JointState::new(self.0.dof);
        self.0.at(t, &mut out);
        JointState(out)
    }

    /// This trajectory from time `t` on, braking to rest along its path: a controlled stop, for
    /// when the motion must not go on. It starts in exactly the state at `t`; verify it with
    /// `check_from`. Raises `UnsafeError` if no such stop ends before the path does.
    fn stop(&self, py: Python<'_>, robot: &Robot, t: f32) -> PyResult<Self> {
        py.detach(|| self.0.stop(&robot.0, t)).or_raise().map(Self)
    }

    /// The trajectory sampled `hz` times per second, from its start to its exact end.
    fn sample(&self, py: Python<'_>, hz: f32) -> PyResult<JointTrajectory> {
        let sampled = py.detach(|| self.0.sample(hz)).or_raise()?;
        JointTrajectory::new(py, sampled)
    }

    /// Verifies the trajectory is safe to run on `robot`: finite, at rest at both ends, within
    /// every position limit, and within every velocity, acceleration and jerk limit. Raises
    /// `UnsafeError` if not.
    fn check(&self, robot: &Robot) -> PyResult<()> {
        self.0.check(&robot.0).or_raise()
    }

    /// `check` for a trajectory that starts in motion (`Trajectory.moving`): it must start at
    /// `start` instead of at rest.
    fn check_from(&self, robot: &Robot, start: &JointState) -> PyResult<()> {
        self.0.check_from(&robot.0, &start.0).or_raise()
    }

    fn __repr__(&self) -> String {
        format!(
            "Trajectory({} s, {} control points)",
            self.0.duration(),
            self.0.control_points.len() / self.0.dof
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<JointState>()?;
    m.add_class::<HandSpeed>()?;
    m.add_class::<JointTrajectory>()?;
    m.add_class::<Trajectory>()?;
    Ok(())
}
