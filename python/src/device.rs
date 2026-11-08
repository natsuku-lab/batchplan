//! Devices, the worlds uploaded to them, and the contacts they report.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::OrRaise;
use crate::field::{self, Field, tuple_field, value_class};
use crate::robot::Robot;
use crate::world::World;

value_class! {
    /// The collision cost IK and trajectory optimization lower: each clearance's shortfall below
    /// its margin, weighted. The default weighs nothing.
    CollisionWeights(batchplan::CollisionWeights) {
        /// Weight of the world cost.
        world: f32,
        /// Weight of the self-collision cost.
        self_collision: f32,
        /// Obstacle clearance below which the world cost activates (meters).
        margin: f32,
        /// Self clearance below which the self-collision cost activates (meters).
        self_margin: f32,
    }
    methods {
        #[new]
        #[pyo3(signature = (**changes))]
        fn new(changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            Ok(Self(Self::changed(Default::default(), changes)?))
        }
    }
}

/// A link of the robot near an obstacle, or two of its links near each other.
#[pyclass(module = "batchplan", frozen)]
#[derive(Clone)]
pub(crate) enum Contact {
    /// A link and an obstacle of the world: the link's name, the obstacle's index in
    /// `World.obstacles`, and the signed distance from the link's spheres to it (meters),
    /// negative where they overlap.
    World {
        link: String,
        obstacle: usize,
        distance: f64,
    },
    /// Two links of the robot that are checked against each other, and the signed distance
    /// between their spheres (meters), self-collision buffers included.
    SelfCollision {
        links: (String, String),
        distance: f64,
    },
    /// A kind of contact these bindings don't know yet, as batchplan describes it.
    Other { description: String },
}

impl Field for batchplan::Contact {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mirror = match self {
            batchplan::Contact::World {
                link,
                obstacle,
                distance,
            } => Contact::World {
                link: link.clone(),
                obstacle: *obstacle,
                distance: field::shortest(*distance),
            },
            batchplan::Contact::SelfCollision {
                links: [a, b],
                distance,
            } => Contact::SelfCollision {
                links: (a.clone(), b.clone()),
                distance: field::shortest(*distance),
            },
            other => Contact::Other {
                description: format!("{other:?}"),
            },
        };
        mirror.into_bound_py_any(py)
    }
    fn from_py(_: &Bound<'_, PyAny>) -> PyResult<Self> {
        Err(PyTypeError::new_err(
            "contacts are what batchplan reports, not something to pass it",
        ))
    }
}

#[pymethods]
impl Contact {
    fn __repr__(&self) -> String {
        match self {
            Self::World {
                link,
                obstacle,
                distance,
            } => format!("Contact.World(link={link:?}, obstacle={obstacle}, distance={distance})"),
            Self::SelfCollision { links, distance } => format!(
                "Contact.SelfCollision(links=({:?}, {:?}), distance={distance})",
                links.0, links.1
            ),
            Self::Other { description } => format!("Contact.Other({description:?})"),
        }
    }
}

tuple_field!(batchplan::Contact);

/// Runs batched IK, planning and optimization on a GPU or on CPU threads: the only way to run
/// batched work. Each device owns its threads.
#[pyclass(module = "batchplan", frozen)]
pub(crate) struct Device {
    pub(crate) device: batchplan::Device,
    robot: Py<Robot>,
}

#[pymethods]
impl Device {
    /// CPU threads: `threads` worker threads, or one per core if not given, started now and owned
    /// by the device.
    #[staticmethod]
    #[pyo3(signature = (robot, threads = None))]
    fn cpu(py: Python<'_>, robot: Py<Robot>, threads: Option<usize>) -> PyResult<Self> {
        let device = {
            let r = &robot.get().0;
            py.detach(|| match threads {
                Some(n) => batchplan::Device::cpu_threads(r, n),
                None => batchplan::Device::cpu(r),
            })
            .or_raise()?
        };
        Ok(Self { device, robot })
    }

    /// A GPU through wgpu: the fastest one found (Vulkan on Linux and Windows, Metal on macOS,
    /// DX12 as fallback), or the first adapter whose name contains `name` (case-insensitive), e.g.
    /// `"radv"` or `"llvmpipe"`.
    #[staticmethod]
    #[pyo3(signature = (robot, name = None))]
    fn gpu(py: Python<'_>, robot: Py<Robot>, name: Option<&str>) -> PyResult<Self> {
        let device = {
            let r = &robot.get().0;
            py.detach(|| match name {
                Some(name) => batchplan::Device::gpu_named(r, name),
                None => batchplan::Device::gpu(r),
            })
            .or_raise()?
        };
        Ok(Self { device, robot })
    }

    /// A device for `robot` on the same hardware: it shares this device's GPU or threads, and
    /// worlds uploaded to either work on both, except that worlds with touches need the links both
    /// robots have to match by name. Use it when the robot changes, as when it picks up an object
    /// or a locked joint moves, without starting a device or uploading worlds again.
    fn with_robot(&self, py: Python<'_>, robot: Py<Robot>) -> PyResult<Self> {
        let device = {
            let r = &robot.get().0;
            py.detach(|| self.device.with_robot(r)).or_raise()?
        };
        Ok(Self { device, robot })
    }

    /// Prepares `worlds` for this device: every batched call takes the result. Errors if an
    /// obstacle has non-finite or negative sizes, a rotation that is not a unit quaternion, or a
    /// malformed distance grid, or if a touch names an obstacle the world lacks or a link the
    /// robot lacks.
    fn upload(&self, py: Python<'_>, worlds: &Bound<'_, PyAny>) -> PyResult<Worlds> {
        let worlds = Vec::<batchplan::World>::from_py(worlds)?;
        let uploaded = py.detach(|| self.device.upload(&worlds)).or_raise()?;
        Ok(Worlds(uploaded))
    }

    /// Pays now what the first calls would: builds the kernels a GPU builds when first needed,
    /// then runs one small call of each kind. Call it once after making the device and before
    /// timing anything.
    fn warmup(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.device.warmup()).or_raise()
    }

    /// What runs the batches: the GPU adapter and its API, or the CPU's thread count.
    #[getter]
    fn name(&self) -> String {
        self.device.name()
    }

    /// The robot it plans for.
    #[getter]
    fn robot(&self, py: Python<'_>) -> Py<Robot> {
        self.robot.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!("Device({:?})", self.device.name())
    }
}

/// Worlds prepared for one `Device` by `Device.upload`. On a GPU their obstacles and distance
/// grids stay in device memory, so every call that takes them reuses one upload. They read as the
/// list of worlds uploaded.
#[pyclass(module = "batchplan", frozen, sequence)]
pub(crate) struct Worlds(pub(crate) batchplan::Worlds);

#[pymethods]
impl Worlds {
    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<World> {
        let n = self.0.len();
        let i = if index < 0 {
            n.checked_sub(index.unsigned_abs())
        } else {
            Some(index.unsigned_abs()).filter(|&i| i < n)
        };
        let i = i.ok_or_else(|| PyIndexError::new_err(format!("world {index} of {n}")))?;
        Ok(World(self.0[i].clone()))
    }

    fn __repr__(&self) -> String {
        format!("Worlds({} uploaded)", self.0.len())
    }
}
