//! Robots and how they load.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{InputError, OrRaise};
use crate::field::{self, Field, value_class};
use crate::world::Pose;

/// Which link geometry spheres are fitted to.
#[pyclass(module = "batchplan", frozen, eq, eq_int, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SphereGeometry {
    /// The links' collision geometry.
    Collision,
    /// Their visual geometry, for descriptions whose collision geometry is coarse or missing.
    Visual,
}

impl Field for batchplan::SphereGeometry {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mirror = match self {
            batchplan::SphereGeometry::Collision => SphereGeometry::Collision,
            batchplan::SphereGeometry::Visual => SphereGeometry::Visual,
        };
        Ok(Bound::new(py, mirror)?.into_any())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(match *value.cast::<SphereGeometry>()?.get() {
            SphereGeometry::Collision => batchplan::SphereGeometry::Collision,
            SphereGeometry::Visual => batchplan::SphereGeometry::Visual,
        })
    }
}

value_class! {
    /// Limits replacing a joint's own: its description's, or the defaults in `RobotOptions` where
    /// the description gives none. Each is optional, as in MoveIt's `joint_limits.yaml`.
    JointLimits(batchplan::JointLimits) {
        /// `(lower, upper)`; giving them makes a continuous joint an ordinary one.
        position: Option<[f32; 2]>,
        /// rad/s or m/s.
        velocity: Option<f32>,
        /// rad/s² or m/s².
        acceleration: Option<f32>,
        /// rad/s³ or m/s³.
        jerk: Option<f32>,
    }
    methods {
        #[new]
        #[pyo3(signature = (**changes))]
        fn new(changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            Ok(Self(Self::changed(Default::default(), changes)?))
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
}

value_class! {
    /// How spheres are fitted to a robot's links.
    SphereOptions(batchplan::SphereOptions) {
        /// Spheres for the whole robot.
        budget: usize,
        /// How far spheres may reach beyond the geometry (meters) before the fit spends more of
        /// the budget. When the budget cannot reach it, the fit uses the smallest overhang it can.
        tolerance: f32,
        /// Largest spacing of the interior grid that proposes sphere centers (meters). Small links
        /// use a finer grid.
        voxel_size: f32,
        /// The geometry the spheres cover.
        geometry: batchplan::SphereGeometry,
        /// Random configurations used to find link pairs that never collide.
        self_collision_samples: usize,
        /// Seed of the fit's random choices and those configurations.
        rng_seed: u64,
    }
    methods {
        #[new]
        #[pyo3(signature = (**changes))]
        fn new(changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            Ok(Self(Self::changed(Default::default(), changes)?))
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
}

value_class! {
    /// Collision spheres and self-collision settings by link, as a file stores them: fitted when a
    /// robot loads without one; saved to commit and hand-tune.
    CollisionModel(batchplan::CollisionModel) {
        /// Where the spheres came from (provenance, licence).
        source: Option<String>,
        /// Per link: spheres, `(n, 4)` as `x, y, z, radius` in the link frame.
        spheres: BTreeMap<String, Vec<[f32; 4]>>,
        /// Per link: extra radius used only for self-collision.
        self_collision_buffer: BTreeMap<String, f32>,
        /// Link pairs never checked against each other.
        self_collision_ignore: BTreeMap<String, Vec<String>>,
        /// Joints locked while fitting, at their values.
        locked_joints: BTreeMap<String, f32>,
    }
    methods {
        #[new]
        #[pyo3(signature = (**changes))]
        fn new(changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            Ok(Self(Self::changed(Default::default(), changes)?))
        }

        /// Reads a model written by `save`.
        #[staticmethod]
        fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
            py.detach(|| batchplan::CollisionModel::load(path)).or_raise().map(Self)
        }

        /// The model of a robot made with `Robot.assemble`, from each part's own model and the
        /// prefix its names take, as `(prefix, model)` pairs: two arms fitted once share one
        /// model. Links of different parts are always checked against each other.
        #[staticmethod]
        fn assemble(parts: &Bound<'_, PyAny>) -> PyResult<Self> {
            let items = field::items(parts)?;
            let parts = items.iter().map(|part| {
                let (prefix, model): (String, Bound<'_, PyAny>) = part.extract()?;
                Ok((prefix, batchplan::CollisionModel::from_py(&model)?))
            });
            let parts: Vec<(String, batchplan::CollisionModel)> = parts.collect::<PyResult<_>>()?;
            let named: Vec<(&str, &batchplan::CollisionModel)> =
                parts.iter().map(|(p, m)| (p.as_str(), m)).collect();
            Ok(Self(batchplan::CollisionModel::assemble(&named)))
        }

        /// Writes the model as JSON.
        fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
            py.detach(|| self.0.save(path)).or_raise()
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
}

value_class! {
    /// A robot description placed in a robot made of several (see `Robot.assemble`).
    Part(batchplan::Part) {
        /// The description file: URDF or MJCF.
        path: PathBuf,
        /// Prefixed to every link and joint name of the description, such as `left_`.
        prefix: String,
        /// Where the description's root link sits in the robot's frame.
        base: batchplan::Pose,
    }
    methods {
        /// The description at `path`, its names prefixed with `prefix`, its root at `base` (the
        /// robot's origin if not given).
        #[new]
        #[pyo3(signature = (path, prefix = String::new(), base = None))]
        fn new(path: PathBuf, prefix: String, base: Option<&Pose>) -> Self {
            Self(batchplan::Part {
                path,
                prefix,
                base: base.map_or(batchplan::Pose::IDENTITY, |b| b.0),
            })
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
}

crate::field::tuple_field!(batchplan::Part);

value_class! {
    /// How `Robot.load` builds a robot from its description.
    RobotOptions(batchplan::RobotOptions) {
        /// Root of the planned tree; defaults to the description's root link.
        base_link: Option<String>,
        /// IK target frames, one per hand (end effector); defaults to the link with the most
        /// actuated joints above it. Set it for robots with grippers, and name each hand of a robot
        /// with several.
        ee_links: Vec<String>,
        /// Per hand, by its link in `ee_links`: the links it grips with, for a hand whose fingers
        /// are not the links below its palm, as a gripper whose fixed jaw is part of the palm's
        /// link (then name the palm too). Hands left out keep the links below their palm.
        finger_links: HashMap<String, Vec<String>>,
        /// Defaults to zero for every joint whose limits contain zero, else the middle of its
        /// range.
        default_q: Option<Vec<f32>>,
        /// Joints held at a value and removed from planning (mimics of a locked joint follow it).
        lock_joints: HashMap<String, f32>,
        /// Limits by joint name, replacing the description's; mimic joints may be named too. The
        /// scales below apply on top. Planar, ball and floating joints become joints named
        /// `<joint>_x`, `_y`, `_z` (translations, which need position limits from here) and
        /// `_theta` or `_rx`, `_ry`, `_rz` (rotations).
        joint_limits: HashMap<String, batchplan::JointLimits>,
        /// Directories searched for `package://` packages: each is a package or holds packages.
        /// Directories above the description file are searched after these.
        package_dirs: Vec<PathBuf>,
        /// Collision spheres; fitted to the description's geometry with `spheres` when `None`.
        collision_model: Option<batchplan::CollisionModel>,
        /// How spheres are fitted when there is no `collision_model`.
        spheres: batchplan::SphereOptions,
        /// Per link, by name: how far its spheres may reach beyond its geometry (meters), in place
        /// of `spheres.tolerance`, as fingers fitted tighter than the arm so they can grasp small
        /// objects.
        sphere_tolerances: HashMap<String, f32>,
        /// A MoveIt SRDF's text, whose disabled collision pairs are added to the collision
        /// model's.
        srdf: Option<String>,
        /// Velocity limit (rad/s or m/s) for joints whose description gives none (MJCF has none).
        max_velocity: f32,
        /// Acceleration limit (rad/s² or m/s²) for joints whose description gives none.
        max_acceleration: f32,
        /// Jerk limit (rad/s³ or m/s³) for joints whose description gives none.
        max_jerk: f32,
        /// Scales every joint's velocity limit, its description's or `max_velocity`, in (0, 1].
        /// Plans, timings and checks all keep within the scaled limits.
        velocity_scale: f32,
        /// Scales every joint's acceleration limit, in (0, 1].
        acceleration_scale: f32,
        /// Scales every joint's jerk limit, in (0, 1].
        jerk_scale: f32,
    }
    hidden { variants }
    methods {
        #[new]
        #[pyo3(signature = (**changes))]
        fn new(changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            Ok(Self(Self::changed(Default::default(), changes)?))
        }
    }
}

/// A robot's kinematic tree and collision spheres, from URDF or MJCF. Immutable once loaded.
#[pyclass(module = "batchplan", frozen)]
pub(crate) struct Robot(pub(crate) batchplan::Robot);

#[pymethods]
impl Robot {
    /// Loads a URDF or MJCF file. Collision spheres are fitted unless `options.collision_model`
    /// gives them.
    #[staticmethod]
    #[pyo3(signature = (path, options = None))]
    fn load(py: Python<'_>, path: PathBuf, options: Option<&RobotOptions>) -> PyResult<Self> {
        let options = options.map_or_else(batchplan::RobotOptions::default, |o| o.0.clone());
        py.detach(|| batchplan::Robot::load(path, &options))
            .or_raise()
            .map(Self)
    }

    /// A robot made of several descriptions, such as two arms at a table: each part's link and
    /// joint names get its prefix, its root link is fixed at its base, and the parts hang from one
    /// root link, `world`. The options name links and joints with their prefixes; give each arm's
    /// hand in `ee_links`.
    #[staticmethod]
    #[pyo3(signature = (parts, options = None))]
    fn assemble(
        py: Python<'_>,
        parts: &Bound<'_, PyAny>,
        options: Option<&RobotOptions>,
    ) -> PyResult<Self> {
        let parts = Vec::<batchplan::Part>::from_py(parts)?;
        let options = options.map_or_else(batchplan::RobotOptions::default, |o| o.0.clone());
        py.detach(|| batchplan::Robot::assemble(&parts, &options))
            .or_raise()
            .map(Self)
    }

    /// The descriptions the robot was loaded from: one for `Robot.load`, its parts for
    /// `Robot.assemble`.
    #[getter]
    fn parts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.parts().to_vec().to_py(py)
    }

    /// A robot from a URDF's text, as ROS publishes it on `/robot_description`.
    #[staticmethod]
    #[pyo3(signature = (urdf, options = None))]
    fn from_urdf(py: Python<'_>, urdf: &str, options: Option<&RobotOptions>) -> PyResult<Self> {
        let options = options.map_or_else(batchplan::RobotOptions::default, |o| o.0.clone());
        py.detach(|| batchplan::Robot::from_urdf(urdf, &options))
            .or_raise()
            .map(Self)
    }

    /// The robot name from its description.
    #[getter]
    fn name(&self) -> &str {
        self.0.name()
    }

    /// The number of actuated joints: the length of every configuration.
    #[getter]
    fn dof(&self) -> usize {
        self.0.dof()
    }

    /// The actuated joints' names, in the order configurations hold them.
    #[getter]
    fn joint_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.joint_names().to_vec().to_py(py)
    }

    /// Each joint's lower position limit (radians, or meters for prismatic joints).
    #[getter]
    fn lower<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.lower().to_vec().to_py(py)
    }

    /// Each joint's upper position limit.
    #[getter]
    fn upper<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.upper().to_vec().to_py(py)
    }

    /// Which joints are continuous: they turn without limit and their values wrap every 2π. For
    /// them, `lower` and `upper` give the turn (−π to π) that seeds and samples come from, not
    /// limits.
    #[getter]
    fn continuous<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        field::array(py, self.0.continuous(), &[self.0.dof()])
    }

    /// The configuration plans start from by default and IK's first seed starts at.
    #[getter]
    fn default_q<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.default_q().to_vec().to_py(py)
    }

    /// Each joint's velocity limit (per second), with `RobotOptions.velocity_scale` applied.
    #[getter]
    fn max_velocity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.max_velocity().to_vec().to_py(py)
    }

    /// Each joint's acceleration limit, with `RobotOptions.acceleration_scale` applied.
    #[getter]
    fn max_acceleration<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.max_acceleration().to_vec().to_py(py)
    }

    /// Each joint's jerk limit, with `RobotOptions.jerk_scale` applied.
    #[getter]
    fn max_jerk<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.max_jerk().to_vec().to_py(py)
    }

    /// The collision spheres in use, generated or as given.
    #[getter]
    fn collision_model(&self) -> CollisionModel {
        CollisionModel(self.0.collision_model().clone())
    }

    /// The number of hands: IK target frames (`RobotOptions.ee_links`).
    #[getter]
    fn hands(&self) -> usize {
        self.0.hands()
    }

    /// How far from the origin of its frame the robot reaches (meters): a bound on its collision
    /// spheres' farthest points over every configuration within the joint limits.
    #[getter]
    fn reach(&self) -> f64 {
        field::shortest(self.0.reach())
    }

    /// World pose of hand `hand`'s IK target frame at configuration `q`.
    #[pyo3(signature = (q, hand = 0))]
    fn ee_pose(&self, q: &Bound<'_, PyAny>, hand: usize) -> PyResult<Pose> {
        let q = self.configuration(q)?;
        self.check_hand(hand)?;
        Ok(Pose(self.0.ee_pose(&q, hand)))
    }

    /// World pose of the named link at configuration `q`, or `None` if the robot has no such
    /// link.
    fn link_pose(&self, q: &Bound<'_, PyAny>, link: &str) -> PyResult<Option<Pose>> {
        let q = self.configuration(q)?;
        Ok(self.0.link_pose(&q, link).map(Pose))
    }

    /// The joints that move hand `hand`, root first: the actuated joints of its chain (a mimic
    /// joint counts as its leader).
    fn hand_joints<'py>(&self, py: Python<'py>, hand: usize) -> PyResult<Bound<'py, PyAny>> {
        self.check_hand(hand)?;
        self.0.hand_joints(hand).to_py(py)
    }

    /// Names of the links of hand `hand`, for `Touch`: the first link at or above its frame that
    /// has collision spheres and every link below that one, its fingers and any object it holds.
    fn hand_links<'py>(&self, py: Python<'py>, hand: usize) -> PyResult<Bound<'py, PyAny>> {
        self.check_hand(hand)?;
        let links: Vec<String> = self
            .0
            .hand_links(hand)
            .into_iter()
            .map(String::from)
            .collect();
        links.to_py(py)
    }

    /// Names of the links hand `hand` grips with: those `RobotOptions.finger_links` names for it,
    /// or else its links below the palm that have collision spheres. Empty for a hand with none,
    /// such as a suction cup or a bare flange.
    fn finger_links<'py>(&self, py: Python<'py>, hand: usize) -> PyResult<Bound<'py, PyAny>> {
        self.check_hand(hand)?;
        let links: Vec<String> = self
            .0
            .finger_links(hand)
            .into_iter()
            .map(String::from)
            .collect();
        links.to_py(py)
    }

    /// How far hand `hand`'s spheres reach past its IK frame along the frame's z axis at the
    /// default configuration (meters): how much deeper than a grasp the fingertips go, for
    /// `GraspOptions.tip`. Negative where every sphere stays short of the frame.
    fn tip(&self, hand: usize) -> PyResult<f64> {
        self.check_hand(hand)?;
        Ok(field::shortest(self.0.tip(hand)))
    }

    /// Every collision sphere at configuration `q`, in the robot's frame, `(n, 4)` as
    /// `x, y, z, radius`, link by link: what the world is checked against.
    fn spheres<'py>(&self, py: Python<'py>, q: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let q = self.configuration(q)?;
        self.0.spheres(&q).to_py(py)
    }

    /// The robot holding `object`, fixed to one of its links: its spheres fitted to the object's
    /// shapes and checked as part of the robot. Make a device for it with `Device.with_robot`, so
    /// the uploaded worlds keep working.
    fn attach(&self, py: Python<'_>, object: &crate::held::AttachedObject) -> PyResult<Self> {
        py.detach(|| self.0.attach(&object.0)).or_raise().map(Self)
    }

    /// The robot without the attached object named `name`.
    fn detach(&self, name: &str) -> PyResult<Self> {
        self.0.detach(name).or_raise().map(Self)
    }

    /// The robot with joints it locked at load held at new values, as a torso that moved: by
    /// name, each one it locked.
    fn relock(&self, joints: &Bound<'_, PyAny>) -> PyResult<Self> {
        let joints = HashMap::<String, f32>::from_py(joints)?;
        self.0.relock(&joints).or_raise().map(Self)
    }

    fn __repr__(&self) -> String {
        format!("Robot(name={:?}, dof={})", self.0.name(), self.0.dof())
    }
}

impl Robot {
    /// `q` as floats, checked to have one per joint.
    fn configuration(&self, q: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
        field::floats_of(q, self.0.dof(), "a configuration")
    }

    fn check_hand(&self, hand: usize) -> PyResult<()> {
        if hand < self.0.hands() {
            Ok(())
        } else {
            Err(InputError::new_err(format!(
                "hand {hand} out of range: the robot has {}",
                self.0.hands()
            )))
        }
    }
}
