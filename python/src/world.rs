//! Poses, obstacles and worlds.

use batchplan::glam::{Quat, Vec3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::OrRaise;
use crate::field::{Field, tuple_field, value_class};

value_class! {
    /// A position and a rotation: a frame relative to another. Rotations are unit quaternions,
    /// scalar last, `(x, y, z, w)`, as ROS, SciPy and glam order them.
    Pose(batchplan::Pose) {
        /// `(x, y, z)`, in meters.
        position: Vec3,
        /// `(x, y, z, w)`.
        rotation: Quat,
    }
    methods {
        #[new]
        #[pyo3(signature = (position = None, rotation = None))]
        fn new(
            position: Option<&Bound<'_, PyAny>>,
            rotation: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            let position = position.map(Vec3::from_py).transpose()?.unwrap_or(Vec3::ZERO);
            let rotation = rotation.map(Quat::from_py).transpose()?.unwrap_or(Quat::IDENTITY);
            Ok(Self(batchplan::Pose::new(position, rotation)))
        }

        /// No translation, no rotation.
        #[classattr]
        #[pyo3(name = "IDENTITY")]
        fn identity() -> Self {
            Self(batchplan::Pose::IDENTITY)
        }

        /// This pose followed by `other`, in this pose's frame:
        /// `world_from_hand.mul_pose(hand_from_tool)`.
        fn mul_pose(&self, other: &Self) -> Self {
            Self(self.0.mul_pose(other.0))
        }

        /// The pose that undoes this one.
        fn inverse(&self) -> Self {
            Self(self.0.inverse())
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
}

value_class! {
    /// A signed distance to a surface, and its gradient.
    Distance(batchplan::Distance) {
        /// Meters: negative inside.
        value: f32,
        /// How the distance grows with the point: the unit outward direction, except inside
        /// distance grids, where it is the interpolation's gradient.
        gradient: Vec3,
    }
}

/// Something in a world: a box, ball, cylinder or capsule, made axis-aligned at a center. `placed`
/// sets its frame (center and rotation) to a pose.
#[pyclass(module = "batchplan", frozen)]
#[derive(Clone)]
pub(crate) struct Obstacle(pub(crate) batchplan::Obstacle);

#[pymethods]
impl Obstacle {
    /// A box at `center` reaching `half_extents` along each axis (meters).
    #[staticmethod]
    fn cuboid(center: &Bound<'_, PyAny>, half_extents: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(batchplan::Obstacle::cuboid(
            Vec3::from_py(center)?,
            Vec3::from_py(half_extents)?,
        )))
    }

    /// A ball.
    #[staticmethod]
    fn sphere(center: &Bound<'_, PyAny>, radius: f32) -> PyResult<Self> {
        Ok(Self(batchplan::Obstacle::sphere(
            Vec3::from_py(center)?,
            radius,
        )))
    }

    /// A solid cylinder along z, `half_height` above and below its center.
    #[staticmethod]
    fn cylinder(center: &Bound<'_, PyAny>, radius: f32, half_height: f32) -> PyResult<Self> {
        Ok(Self(batchplan::Obstacle::cylinder(
            Vec3::from_py(center)?,
            radius,
            half_height,
        )))
    }

    /// A capsule around the z segment from `-half_length` to `half_length` about its center.
    #[staticmethod]
    fn capsule(center: &Bound<'_, PyAny>, radius: f32, half_length: f32) -> PyResult<Self> {
        Ok(Self(batchplan::Obstacle::capsule(
            Vec3::from_py(center)?,
            radius,
            half_length,
        )))
    }

    /// The obstacle moved so that its frame (center and rotation) is `pose`: the center it was
    /// made at is replaced. Spheres keep no rotation.
    fn placed(&self, pose: &Pose) -> Self {
        Self(self.0.placed(pose.0))
    }

    /// The obstacle's frame: its center and rotation.
    #[getter]
    fn pose(&self) -> Pose {
        Pose(self.0.pose())
    }

    /// The signed distance from point `p` to its surface, and the distance's gradient.
    fn distance(&self, p: &Bound<'_, PyAny>) -> PyResult<Distance> {
        Ok(Distance(self.0.distance(Vec3::from_py(p)?)))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

impl Field for batchplan::Obstacle {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(py, Obstacle(self.clone()))?.into_any())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(value.cast::<Obstacle>()?.get().0.clone())
    }
}

value_class! {
    /// Links that may touch one obstacle of a `World`, as fingers touch what they grasp: contact
    /// between them is not a collision. Names are resolved when the world is uploaded.
    Touch(batchplan::Touch) {
        /// The obstacle's index in `World.obstacles`.
        obstacle: usize,
        /// Links of the robot, such as a hand's (`Robot.hand_links`).
        links: Vec<String>,
    }
    methods {
        #[new]
        fn new(obstacle: usize, links: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(Self(batchplan::Touch::new(obstacle, Vec::<String>::from_py(links)?)))
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
}

value_class! {
    /// One scene to plan in. Problems name their world by its index in the list uploaded with
    /// `Device.upload`.
    World(batchplan::World) {
        /// Everything in the scene; uploading checks each is well formed.
        obstacles: Vec<batchplan::Obstacle>,
        /// Links allowed to touch particular obstacles.
        touch: Vec<batchplan::Touch>,
    }
    methods {
        #[new]
        #[pyo3(signature = (obstacles = None, **changes))]
        fn new(
            obstacles: Option<&Bound<'_, PyAny>>,
            changes: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Self> {
            let obstacles = obstacles.map(Vec::from_py).transpose()?.unwrap_or_default();
            Ok(Self(Self::changed(batchplan::World::new(obstacles), changes)?))
        }

        /// A copy without obstacle `obstacle`, as a hand picking it up takes it (Rust's
        /// `World::remove`): its touches go with it, and the touches of the obstacles after it
        /// follow them down.
        fn without(&self, obstacle: usize) -> PyResult<Self> {
            let mut world = self.0.clone();
            world.remove(obstacle).or_raise()?;
            Ok(Self(world))
        }

        /// The links that touch an obstacle in every configuration, as a link turning just
        /// above the base touches its table.
        fn always_touching<'py>(
            &self,
            py: Python<'py>,
            robot: &crate::robot::Robot,
        ) -> PyResult<Bound<'py, PyAny>> {
            self.0.always_touching(&robot.0).to_py(py)
        }
    }
}

tuple_field!(
    batchplan::Obstacle,
    batchplan::Touch,
    batchplan::World,
    Option<batchplan::Pose>
);
