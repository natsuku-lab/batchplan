//! Where batched work runs. A [`Device`] owns an immutable copy of the robot (uploaded once on
//! the GPU) and evaluates independent items in parallel. Worlds are uploaded once as [`Worlds`],
//! and items reference them by index.
//!
//! The CPU and GPU implementations sit behind a crate-private `Backend` trait and are kept
//! numerically interchangeable, so code written against a `Device` runs unchanged on either.
//! Every batch is checked here before it reaches either one, so malformed input is an `Err` on
//! both rather than a panic on one and a plausible wrong answer on the other.

use std::any::Any;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::error::{Result, ensure_input, input};

use crate::cpu::CpuBackend;
use crate::edges::Search;
#[cfg(feature = "gpu")]
use crate::gpu::GpuBackend;
use crate::ik::IkOptions;
use crate::robot::Robot;
use crate::sdf::{DepthImage, Layout, Occupancy, RobotMask};
use crate::trajopt::PlanOptions;
use crate::types::{JointPaths, Pose};
use crate::world::World;

#[derive(Clone, Copy, Debug)]
pub struct CollisionWeights {
    pub world: f32,
    pub self_collision: f32,
    /// Obstacle clearance below which the world cost activates (meters).
    pub margin: f32,
    /// Self clearance below which the self-collision cost activates (meters).
    pub self_margin: f32,
}

impl CollisionWeights {
    /// Zero weights: evaluate clearances only.
    pub const NONE: Self = Self { world: 0.0, self_collision: 0.0, margin: 0.0, self_margin: 0.0 };
}

/// Per-configuration collision evaluation, indexed by item. Clearances are 1e30 when there is
/// nothing to check (an empty world, or no self-collision pairs).
#[derive(Clone, Debug, Default)]
pub struct Evaluation {
    /// Signed distance from the robot spheres to the nearest obstacle (negative = penetrating).
    /// Exact up to the margin (or zero, if larger); above it, a lower bound that is still above it,
    /// because a link whose bounding sphere is that far from an obstacle is not checked sphere by
    /// sphere.
    pub world_clearance: Vec<f32>,
    /// Signed distance between the closest checked pair of robot spheres. Exact up to the self
    /// margin (or zero, if larger); above it, a lower bound that is still above it, as for
    /// `world_clearance` with pairs of links.
    pub self_clearance: Vec<f32>,
    pub cost: Vec<f32>,
    /// d(cost)/dq, `[items, dof]`.
    pub grad: Vec<f32>,
}

impl Evaluation {
    pub fn collision_free(&self, item: usize) -> bool {
        collision_free([self.world_clearance[item], self.self_clearance[item]])
    }
}

/// Whether a `[world, self]` clearance from [`Device::clearance`] is collision-free.
pub(crate) fn collision_free(c: [f32; 2]) -> bool {
    c[0] >= 0.0 && c[1] >= 0.0
}

pub struct Device {
    id: u64,
    backend: Box<dyn Backend>,
}

/// Worlds prepared for one [`Device`] by [`Device::upload`]. On a GPU their obstacles and distance
/// grids stay in device memory, so every call that takes them reuses one upload. They deref to the
/// worlds uploaded, as `&[World]`.
pub struct Worlds {
    device: u64,
    worlds: Vec<World>,
    prepared: Box<dyn Any + Send + Sync>,
}

impl Deref for Worlds {
    type Target = [World];

    fn deref(&self) -> &[World] {
        &self.worlds
    }
}

impl Worlds {
    /// The backend's form of the worlds; `Device` checks they were prepared by that backend.
    pub(crate) fn prepared<T: 'static>(&self) -> &T {
        self.prepared.downcast_ref().expect("Device checks that worlds were uploaded to it")
    }
}

impl Device {
    /// One thread per CPU core.
    pub fn cpu(robot: &Robot) -> Result<Self> {
        Self::cpu_threads(robot, 0)
    }

    /// `threads` worker threads (0: one per core), started now and owned by the device: batched
    /// work never runs on rayon's global pool or on threads the device did not start.
    pub fn cpu_threads(robot: &Robot, threads: usize) -> Result<Self> {
        Ok(Self::new(Box::new(CpuBackend::new(robot, threads)?)))
    }

    /// The fastest GPU wgpu finds: Vulkan on Linux and Windows, Metal on macOS, DX12 as fallback.
    /// An error when built without the `gpu` feature.
    pub fn gpu(robot: &Robot) -> Result<Self> {
        Self::gpu_adapter(robot, None)
    }

    /// The first adapter whose name contains `name` (case-insensitive), e.g. `"radv"` or `"llvmpipe"`.
    pub fn gpu_named(robot: &Robot, name: &str) -> Result<Self> {
        Self::gpu_adapter(robot, Some(name))
    }

    #[cfg(feature = "gpu")]
    fn gpu_adapter(robot: &Robot, name: Option<&str>) -> Result<Self> {
        Ok(Self::new(Box::new(GpuBackend::new(robot, name)?)))
    }

    #[cfg(not(feature = "gpu"))]
    fn gpu_adapter(_: &Robot, _: Option<&str>) -> Result<Self> {
        Err(crate::error::Error::Gpu("batchplan was built without the `gpu` feature".into()))
    }

    fn new(backend: Box<dyn Backend>) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), backend }
    }

    /// A device for `robot` on the same hardware: it shares this device's GPU or threads, and
    /// worlds uploaded to either work on both. Use it when the robot changes, as when it picks up
    /// an object ([`Robot::attach`]), without starting a device or uploading worlds again.
    pub fn with_robot(&self, robot: &Robot) -> Result<Self> {
        Ok(Self { id: self.id, backend: self.backend.with_robot(robot)? })
    }

    /// Prepares `worlds` for this device. Calls on other devices refuse the result. Errors if an
    /// obstacle has non-finite or negative sizes, a rotation that is not a unit quaternion, or a
    /// malformed distance grid.
    pub fn upload(&self, worlds: &[World]) -> Result<Worlds> {
        for (i, w) in worlds.iter().enumerate() {
            w.check().map_err(|e| input!("world {i}: {e}"))?;
        }
        Ok(Worlds { device: self.id, worlds: worlds.to_vec(), prepared: self.backend.upload(worlds)? })
    }

    pub fn name(&self) -> String {
        self.backend.name()
    }

    pub fn robot(&self) -> &Robot {
        self.backend.robot()
    }

    /// Collision cost, its gradient and clearances for each configuration in `q` (`[items, dof]`)
    /// against `worlds[item_world[item]]`. Errors if `q` does not hold `dof` values per item, an
    /// item names a world outside `worlds`, or `worlds` were uploaded to another device.
    pub fn evaluate(&self, worlds: &Worlds, item_world: &[u32], q: &[f32], w: &CollisionWeights) -> Result<Evaluation> {
        self.check_batch(worlds, item_world, q.len(), self.robot().dof())?;
        self.backend.evaluate(worlds, item_world, q, w)
    }

    /// Runs every search to its end, or to `deadline` (see `edges`).
    pub(crate) fn search(
        &self,
        worlds: &Worlds,
        searches: &mut [&mut dyn Search],
        deadline: Option<Instant>,
    ) -> Result<()> {
        ensure_input!(worlds.device == self.id, "these worlds were uploaded to a different device");
        self.backend.search(worlds, searches, deadline);
        Ok(())
    }

    /// World and self clearance of each configuration (`[items, dof]`), without cost or gradient:
    /// what validity checks need. Exact up to zero (see [`Evaluation`]).
    pub(crate) fn clearance(&self, worlds: &Worlds, item_world: &[u32], q: &[f32]) -> Result<Vec<[f32; 2]>> {
        self.check_batch(worlds, item_world, q.len(), self.robot().dof())?;
        self.backend.clearance(worlds, item_world, q)
    }

    /// The stored values of the distance grid over `grid` whose occupancy is given, or seen in a
    /// checked depth image.
    pub(crate) fn grid_values(&self, grid: &Layout, occupancy: Occupancy) -> Result<Vec<u16>> {
        if let Occupancy::Given(occupied) = occupancy {
            ensure_input!(
                occupied.len() == grid.points(),
                "{} occupancies for {} points",
                occupied.len(),
                grid.points()
            );
        }
        self.backend.grid_values(grid, occupancy)
    }

    /// Fuses a checked `image` into a map's log-odds, one per point of `grid`.
    pub(crate) fn integrate(
        &self,
        grid: &Layout,
        log_odds: &mut [i8],
        image: &DepthImage,
        robot: Option<&RobotMask>,
    ) -> Result<()> {
        ensure_input!(log_odds.len() == grid.points(), "{} log-odds for {} points", log_odds.len(), grid.points());
        self.backend.integrate(grid, log_odds, image, robot)
    }

    pub(crate) fn ik(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        targets: &[Option<Pose>],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 2]>> {
        self.check_batch(worlds, item_world, q.len(), self.robot().dof())?;
        let hands = self.robot().hands();
        ensure_input!(
            targets.len() == item_world.len() * hands,
            "{} IK targets for {} items of {hands} hands",
            targets.len(),
            item_world.len()
        );
        self.backend.ik(worlds, item_world, targets, q, o)
    }

    /// Optimizes `paths` in place; stops early once `deadline` passes.
    pub(crate) fn trajopt(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        paths: &mut JointPaths,
        o: &PlanOptions,
        deadline: Option<Instant>,
    ) -> Result<()> {
        ensure_input!(
            paths.dof == self.robot().dof(),
            "paths have {} joints, the robot has {}",
            paths.dof,
            self.robot().dof()
        );
        ensure_input!(paths.points >= 7, "paths need at least 7 control points");
        self.check_batch(worlds, item_world, paths.positions.len(), paths.points * paths.dof)?;
        self.backend.trajopt(worlds, item_world, paths, o, deadline)
    }

    /// The device, shape and index invariants every batch must satisfy before it reaches a backend.
    fn check_batch(&self, worlds: &Worlds, item_world: &[u32], values: usize, per_item: usize) -> Result<()> {
        ensure_input!(worlds.device == self.id, "these worlds were uploaded to a different device");
        let items = item_world.len();
        ensure_input!(
            values == items * per_item,
            "{items} items need {} values ({per_item} each), got {values}",
            items * per_item
        );
        if let Some((item, &world)) = item_world.iter().enumerate().find(|&(_, &w)| w as usize >= worlds.len()) {
            return Err(input!("item {item} references world {world}, but only {} worlds were given", worlds.len()));
        }
        Ok(())
    }
}

/// The parallel inner loops each device implements. Algorithm modules own seeding, validation
/// and selection so both implementations see identical inputs.
pub(crate) trait Backend: Send + Sync {
    fn name(&self) -> String;
    fn robot(&self) -> &Robot;
    /// The backend's own form of `worlds`, which [`Worlds::prepared`] hands back to it.
    fn upload(&self, worlds: &[World]) -> Result<Box<dyn Any + Send + Sync>>;
    /// The same backend for another robot, sharing the hardware and the form of worlds.
    fn with_robot(&self, robot: &Robot) -> Result<Box<dyn Backend>>;
    fn evaluate(&self, worlds: &Worlds, item_world: &[u32], q: &[f32], w: &CollisionWeights) -> Result<Evaluation>;
    /// `[world, self]` clearance per configuration, computed as `evaluate` does with zero weights
    /// and margins, without the gradient.
    fn clearance(&self, worlds: &Worlds, item_world: &[u32], q: &[f32]) -> Result<Vec<[f32; 2]>>;
    /// Runs every search to its end, or to `deadline`, each on one CPU thread (see `edges`).
    fn search(&self, worlds: &Worlds, searches: &mut [&mut dyn Search], deadline: Option<Instant>);
    /// Runs IK in place on `q` (`[items, dof]`) toward `targets` (`[items, hands]`, `None` for a
    /// hand held still); returns each item's largest `[position error, rotation error]` over its
    /// targeted hands.
    fn ik(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        targets: &[Option<Pose>],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 2]>>;
    /// Optimizes each path's control points in place, holding the three at each end fixed, and
    /// stops between rounds once `deadline` passes.
    fn trajopt(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        paths: &mut JointPaths,
        o: &PlanOptions,
        deadline: Option<Instant>,
    ) -> Result<()>;
    /// The stored values of a distance grid over `grid` (`sdf::depth_occupancy` for images, then
    /// `sdf::squared_edt` and `sdf::finish`).
    fn grid_values(&self, grid: &Layout, occupancy: Occupancy) -> Result<Vec<u16>>;
    /// Fuses `image` into a map's log-odds (`sdf::integrate`).
    fn integrate(
        &self,
        grid: &Layout,
        log_odds: &mut [i8],
        image: &DepthImage,
        robot: Option<&RobotMask>,
    ) -> Result<()>;
}
