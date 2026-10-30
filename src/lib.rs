//! Batched, collision-aware motion generation on any GPU wgpu supports (Vulkan on AMD, NVIDIA and
//! Intel; Metal; DX12), with a CPU implementation that produces the same results.
//!
//! - **Batch-first.** Every query covers many seeds, problems and worlds at once; a [`Device`]
//!   checks each batch and runs it in parallel on a GPU or on all CPU cores.
//! - **Separate algorithms over shared types.** [`ik`], [`trajopt`], [`timing`], [`datagen`] and
//!   the exporters ([`npy`], `lerobot`) are independent modules that take a `Device` and exchange
//!   the [`types`]; there are no planner plugins or runtime configuration. Results keep the
//!   problems they answer (`solved()`).
//! - **Standalone.** No middleware. Robots load from URDF, MJCF or OpenUSD; collision spheres are
//!   fitted to their geometry or loaded from a committed collision-model file. Datasets export to
//!   LeRobot v3.0 with the optional `lerobot` feature.
//!
//! ```no_run
//! use batchplan::{Device, PlanOptions, PlanProblem, Robot, RobotOptions, World};
//! let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &RobotOptions::default())?;
//! let device = Device::gpu(&robot)?;
//! let worlds = device.upload(&[World::default()])?;
//! let goal = vec![0.5, -1.2, 1.0, -1.4, -1.5, 0.3];
//! let problems = vec![PlanProblem::new(0, robot.default_q(), goal)];
//! let result = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default())?;
//! println!("solved: {}", result.best(0).is_some());
//! # anyhow::Ok(())
//! ```

// The library's own rules from Google's Rust style guide: documented public items, `#[must_use]`
// pure functions, and `expect` with its reason rather than `unwrap` outside tests.
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::must_use_candidate, clippy::return_self_not_must_use)]
#![cfg_attr(not(test), warn(clippy::unwrap_used))]

mod cpu;
pub mod datagen;
mod description;
pub mod device;
mod edges;
pub mod error;
#[cfg(feature = "gpu")]
mod gpu;
pub mod ik;
mod lanes;
#[cfg(feature = "lerobot")]
pub mod lerobot;
mod loops;
pub mod meshes;
mod mjcf;
pub mod npy;
pub mod rng;
pub mod robot;
pub mod rrt;
pub mod sdf;
pub mod shortcut;
pub mod spheres;
mod spline;
pub mod timing;
mod topp;
pub mod trajopt;
pub mod types;
mod urdf;
#[cfg(feature = "usd")]
mod usd;
pub mod world;

pub use device::{CollisionWeights, Device, Evaluation, Worlds};
pub use error::Error;
pub use ik::{IkOptions, IkProblem, IkResult, solve_ik};
pub use meshes::{Clearance, MeshModel};
pub use robot::{AttachedObject, CollisionModel, JointLimits, Part, Robot, RobotOptions};
pub use rrt::{RrtOptions, RrtProblem, RrtResult};
pub use sdf::{DepthImage, Intrinsics, Occlusion, OccupancyMap, RobotMask, SdfGrid, SdfOptions};
pub use shortcut::ShortcutOptions;
pub use spheres::{SphereGeometry, SphereOptions};
pub use timing::{JointState, Trajectory};
pub use trajopt::{PlanOptions, PlanProblem, PlanResult, plan};
pub use types::{JointPaths, JointTrajectory, Pose, Solved, StartMotion};
pub use world::{Distance, Obstacle, Touch, World};

/// The math types in batchplan's interface (`Vec3`, `Quat`), so callers use the same version.
pub use glam;

/// Compiles the code in the README and the guides under `docs/`.
#[cfg(doctest)]
mod guides {
    #[cfg(feature = "lerobot")]
    #[doc = include_str!("../README.md")]
    pub struct Readme;
    #[doc = include_str!("../docs/devices.md")]
    pub struct Devices;
    #[doc = include_str!("../docs/perception.md")]
    pub struct Perception;
    #[doc = include_str!("../docs/planning.md")]
    pub struct Planning;
    #[doc = include_str!("../docs/robots.md")]
    pub struct Robots;
    #[cfg(feature = "lerobot")]
    #[doc = include_str!("../docs/training-data.md")]
    pub struct TrainingData;
}
