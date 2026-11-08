//! Moving a hand along a path of poses.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::device::{Device, Worlds};
use crate::error::OrRaise;
use crate::field::{self, Field, tuple_field, value_class};
use crate::solved::Cached;

value_class! {
    /// How `follow` tracks a path.
    ToolPathOptions(batchplan::ToolPathOptions) {
        /// Largest step of the hand between configurations (meters): 5 mm, at least 10 µm.
        step: f32,
        /// Largest turn of the hand between configurations (radians): 0.05, at least 1e-5.
        rotation_step: f32,
        /// How far the hand may stray from the path (meters): 1 mm.
        precision: f32,
        /// How far its rotation may stray from the path's (radians): 0.01.
        rotation_precision: f32,
        /// Joint-space spacing of collision checks, as `RrtOptions.resolution`.
        resolution: f32,
        /// Damped-least-squares iterations per step at most: 30. Each configuration is held to a
        /// tenth of the precisions.
        iterations: u32,
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
    /// A path for one hand to follow from a start configuration.
    ToolPath(batchplan::ToolPath) {
        /// Index of its world in the uploaded batch.
        world: u32,
        /// Where the robot starts; the path starts at the hand's pose there.
        start: Vec<f32>,
        /// The hand that follows the path. Only the joints that move it move.
        hand: usize,
        /// Poses the hand passes through after its start, in the robot's frame.
        waypoints: Vec<batchplan::Pose>,
        /// An object a hand holds, as a pick leaves it: checked as part of the robot, as
        /// attaching it (`Held.attachment`) would check it. `None` holds nothing.
        holding: Option<batchplan::Held>,
    }
    methods {
        /// Hand 0 from `start` through `waypoints` in world `world`, holding nothing.
        #[new]
        #[pyo3(signature = (world, start, waypoints, **changes))]
        fn new(
            world: u32,
            start: &Bound<'_, PyAny>,
            waypoints: &Bound<'_, PyAny>,
            changes: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Self> {
            let path = batchplan::ToolPath {
                world,
                start: Vec::from_py(start)?,
                hand: 0,
                waypoints: Vec::from_py(waypoints)?,
                holding: None,
            };
            Ok(Self(Self::changed(path, changes)?))
        }
    }
}

tuple_field!(batchplan::ToolPath);

/// Why a tool path's motion stops short, and the pose it stops before (`at`).
#[pyclass(module = "batchplan", frozen)]
pub(crate) enum ToolPathStop {
    /// No configuration near the last one reaches this pose: it lies beyond the arm's reach or
    /// its joint limits, or at a singularity the arm cannot step through.
    Unreachable { at: Py<PyAny> },
    /// The hand's next step collides.
    Collision { at: Py<PyAny> },
    /// Configurations reach the poses on either side of this one, but the joints would jump
    /// between them, as a wrist spins half a turn at a singularity.
    Jump { at: Py<PyAny> },
    /// A reason these bindings don't know yet, as batchplan describes it.
    Other { description: String },
}

#[pymethods]
impl ToolPathStop {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(match self {
            Self::Unreachable { at } => {
                format!("ToolPathStop.Unreachable(at={})", at.bind(py).repr()?)
            }
            Self::Collision { at } => {
                format!("ToolPathStop.Collision(at={})", at.bind(py).repr()?)
            }
            Self::Jump { at } => format!("ToolPathStop.Jump(at={})", at.bind(py).repr()?),
            Self::Other { description } => format!("ToolPathStop.Other({description:?})"),
        })
    }
}

impl Field for batchplan::ToolPathStop {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mirror = match self {
            batchplan::ToolPathStop::Unreachable { at } => ToolPathStop::Unreachable {
                at: at.to_py(py)?.unbind(),
            },
            batchplan::ToolPathStop::Collision { at } => ToolPathStop::Collision {
                at: at.to_py(py)?.unbind(),
            },
            batchplan::ToolPathStop::Jump { at } => ToolPathStop::Jump {
                at: at.to_py(py)?.unbind(),
            },
            other => ToolPathStop::Other {
                description: format!("{other:?}"),
            },
        };
        mirror.into_bound_py_any(py)
    }
    fn from_py(_: &Bound<'_, PyAny>) -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "stops are what batchplan reports, not something to pass it",
        ))
    }
}

tuple_field!(Option<batchplan::ToolPathStop>, Vec<batchplan::Contact>);

/// What `follow` did with each path.
#[pyclass(module = "batchplan", frozen)]
pub(crate) struct ToolPathResult {
    result: batchplan::ToolPathResult,
    dof: usize,
    problems: Cached,
    fraction: Cached,
    stopped: Cached,
    paths: Cached,
    contacts: Cached,
}

#[pymethods]
impl ToolPathResult {
    /// The paths, in order.
    #[getter]
    fn problems<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.problems.get(py, || self.result.problems.to_py(py))
    }

    /// Per path: how much of it the motion covers, from 0 to 1, in steps. A path that does not
    /// move the hand is covered.
    #[getter]
    fn fraction<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.fraction.get(py, || self.result.fraction.to_py(py))
    }

    /// Per path: why the motion stops short, where it does.
    #[getter]
    fn stopped<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.stopped.get(py, || self.result.stopped.to_py(py))
    }

    /// Per path: the motion from the start to the last pose it reaches, as B-spline control
    /// points `(points, dof)` for `Trajectory`, or `None` where it reaches nothing past the start.
    #[getter]
    fn paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.paths.get(py, || {
            let paths = self.result.paths.iter().map(|cp| {
                if cp.is_empty() {
                    Ok(py.None().into_bound(py))
                } else {
                    field::array(py, cp, &[cp.len() / self.dof, self.dof])
                }
            });
            let paths: Vec<Bound<'py, PyAny>> = paths.collect::<PyResult<_>>()?;
            Ok(PyTuple::new(py, paths)?.into_any())
        })
    }

    /// Per path stopped by a collision: what the hand's next step collides with, nearest first.
    /// Empty for the rest.
    #[getter]
    fn contacts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.contacts.get(py, || self.result.contacts.to_py(py))
    }

    fn __len__(&self) -> usize {
        self.result.problems.len()
    }

    fn __repr__(&self) -> String {
        let covered = self.result.stopped.iter().filter(|s| s.is_none()).count();
        format!(
            "ToolPathResult({} paths, {covered} covered)",
            self.result.problems.len()
        )
    }
}

/// Follows each path from its start as far as it goes, its steps checked for collision in its
/// world. Paths run in parallel on the device's threads. Paths that hold an object run as a batch
/// for each object and grasp; `device` must not hold one in that hand already.
#[pyfunction]
#[pyo3(signature = (device, worlds, paths, options = None))]
pub(crate) fn follow(
    py: Python<'_>,
    device: &Device,
    worlds: &Worlds,
    paths: &Bound<'_, PyAny>,
    options: Option<&ToolPathOptions>,
) -> PyResult<ToolPathResult> {
    let paths = Vec::<batchplan::ToolPath>::from_py(paths)?;
    let options = options.map(|o| o.0.clone()).unwrap_or_default();
    let result = py
        .detach(|| batchplan::follow(&device.device, &worlds.0, &paths, &options))
        .or_raise()?;
    Ok(ToolPathResult {
        result,
        dof: device.device.robot().dof(),
        problems: Cached::default(),
        fraction: Cached::default(),
        stopped: Cached::default(),
        paths: Cached::default(),
        contacts: Cached::default(),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ToolPathOptions>()?;
    m.add_class::<ToolPath>()?;
    m.add_class::<ToolPathStop>()?;
    m.add_class::<ToolPathResult>()?;
    m.add_function(wrap_pyfunction!(follow, m)?)?;
    Ok(())
}
