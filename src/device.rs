//! Where batched work runs. A [`Device`] owns an immutable copy of the robot (uploaded once on
//! the GPU) and evaluates independent items in parallel. Worlds are uploaded once as [`Worlds`],
//! and items reference them by index.
//!
//! The CPU and GPU implementations sit behind a crate-private `Backend` trait and are kept
//! numerically interchangeable, so code written against a `Device` runs unchanged on either.
//! Every batch is checked here before it reaches either one, so malformed input is an `Err` on
//! both rather than a panic on one and a plausible wrong answer on the other.

use std::any::Any;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::ThreadPool;
use rayon::prelude::*;

use crate::constraint::Region;
use crate::error::{Result, ensure_input, input};
use crate::held::Fits;

use crate::cpu::{CpuBackend, CpuWorlds};
use crate::edges::Search;
#[cfg(feature = "gpu")]
use crate::gpu::GpuBackend;
use crate::ik::IkOptions;
use crate::robot::Robot;
use crate::sdf::{DepthImage, Layout, Occupancy, RobotMask};
use crate::trajopt::PlanOptions;
use crate::types::{JointPaths, Pose};
use crate::world::World;

/// The collision cost IK and trajectory optimization lower: each clearance's shortfall below its
/// margin, weighted. The default, [`CollisionWeights::NONE`], weighs nothing.
#[derive(Clone, Copy, Debug)]
pub struct CollisionWeights {
    /// Weight of the world cost.
    pub world: f32,
    /// Weight of the self-collision cost.
    pub self_collision: f32,
    /// Obstacle clearance below which the world cost activates (meters).
    pub margin: f32,
    /// Self clearance below which the self-collision cost activates (meters).
    pub self_margin: f32,
}

impl CollisionWeights {
    /// Zero weights: evaluate clearances only.
    pub const NONE: Self = Self {
        world: 0.0,
        self_collision: 0.0,
        margin: 0.0,
        self_margin: 0.0,
    };
}

impl Default for CollisionWeights {
    fn default() -> Self {
        Self::NONE
    }
}

/// Per-configuration collision evaluation, indexed by item. Clearances are 1e30 when there is
/// nothing to check (an empty world, or no self-collision pairs).
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct Evaluation {
    /// Signed distance from the robot spheres to the nearest obstacle (negative = penetrating).
    /// Exact up to the margin (or zero, if larger); above it, a lower bound that is still above
    /// it, because a link whose bounding sphere is that far from an obstacle is not checked
    /// sphere by sphere.
    pub world_clearance: Vec<f32>,
    /// Signed distance between the closest checked pair of robot spheres. Exact up to the self
    /// margin (or zero, if larger); above it, a lower bound that is still above it, as for
    /// `world_clearance` with pairs of links.
    pub self_clearance: Vec<f32>,
    /// The collision cost the weights define.
    pub cost: Vec<f32>,
    /// d(cost)/dq, `[items, dof]`.
    pub gradient: Vec<f32>,
}

impl Evaluation {
    /// Whether item `item` clears the world and itself.
    #[must_use]
    pub fn collision_free(&self, item: usize) -> bool {
        collision_free([self.world_clearance[item], self.self_clearance[item]])
    }
}

/// A link of the robot near an obstacle, or two of its links near each other: one of a
/// configuration's [`Device::contacts`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Contact {
    /// A link and an obstacle of the world.
    World {
        /// The link's name.
        link: String,
        /// The obstacle's index in [`World::obstacles`].
        obstacle: usize,
        /// Signed distance from the link's spheres to the obstacle (meters): negative where they
        /// overlap.
        distance: f32,
    },
    /// Two links of the robot that are checked against each other.
    SelfCollision {
        /// The links' names.
        links: [String; 2],
        /// Signed distance between their spheres (meters), self-collision buffers included.
        distance: f32,
    },
}

impl Contact {
    /// The signed distance between the two (meters): negative where they overlap.
    #[must_use]
    pub fn distance(&self) -> f32 {
        match *self {
            Contact::World { distance, .. } | Contact::SelfCollision { distance, .. } => distance,
        }
    }
}

/// Whether a `[world, self]` clearance from [`Device::clearance`] is collision-free.
pub(crate) fn collision_free(c: [f32; 2]) -> bool {
    c[0] >= 0.0 && c[1] >= 0.0
}

/// Runs batched IK, planning, optimization and grid building on a GPU or on CPU threads: the
/// only way to run batched work. Each device owns its threads.
pub struct Device {
    id: u64,
    backend: Box<dyn Backend>,
    /// Held objects' spheres, shared with the devices made from this one.
    fits: Arc<Fits>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name())
            .finish_non_exhaustive()
    }
}

/// Worlds prepared for one [`Device`] by [`Device::upload`]. On a GPU their obstacles and distance
/// grids stay in device memory, so every call that takes them reuses one upload. They deref to the
/// worlds uploaded, as `&[World]`.
pub struct Worlds {
    device: u64,
    worlds: Vec<World>,
    prepared: Box<dyn Any + Send + Sync>,
    /// The uploading robot's links, by name, when a world lets some touch: the touch sets hold
    /// their indices.
    links: Vec<String>,
}

impl fmt::Debug for Worlds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Worlds")
            .field("worlds", &self.worlds.len())
            .finish_non_exhaustive()
    }
}

impl Worlds {
    /// The name link `link` had on the robot these worlds were uploaded for, where a world lets
    /// links touch obstacles and the robot had that link.
    pub(crate) fn link_name(&self, link: usize) -> Option<&str> {
        self.links.get(link).map(String::as_str)
    }
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
        self.prepared
            .downcast_ref()
            .expect("Device checks that worlds were uploaded to it")
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

    /// The first adapter whose name contains `name` (case-insensitive), e.g. `"radv"` or
    /// `"llvmpipe"`.
    pub fn gpu_named(robot: &Robot, name: &str) -> Result<Self> {
        Self::gpu_adapter(robot, Some(name))
    }

    #[cfg(feature = "gpu")]
    fn gpu_adapter(robot: &Robot, name: Option<&str>) -> Result<Self> {
        Ok(Self::new(Box::new(GpuBackend::new(robot, name)?)))
    }

    #[cfg(not(feature = "gpu"))]
    fn gpu_adapter(_: &Robot, _: Option<&str>) -> Result<Self> {
        Err(crate::error::Error::Gpu(
            "batchplan was built without the `gpu` feature".into(),
        ))
    }

    fn new(backend: Box<dyn Backend>) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            backend,
            fits: Arc::default(),
        }
    }

    /// A device for `robot` on the same hardware: it shares this device's GPU or threads, and
    /// worlds uploaded to either work on both, except that worlds with touches need the links
    /// both robots have to match by name ([`Device::upload`]). Use it when the robot changes, as
    /// when it picks up an object ([`Robot::attach`]) or a locked joint moves ([`Robot::relock`]),
    /// without starting a device or uploading worlds again. A GPU device builds kernels for a robot
    /// of a shape it has not seen; a relocked robot keeps its shape.
    pub fn with_robot(&self, robot: &Robot) -> Result<Self> {
        Ok(Self {
            id: self.id,
            backend: self.backend.with_robot(robot)?,
            fits: self.fits.clone(),
        })
    }

    /// Prepares `worlds` for this device. Calls on other devices refuse the result. When a world
    /// lets links touch obstacles, so do devices made with [`Device::with_robot`] whose links
    /// differ by name from this one's where both have links: attaching an object, detaching the
    /// last one attached, or holding another of the same name keeps the worlds valid; an object of
    /// another name at the same link does not. Errors if an obstacle has non-finite or negative
    /// sizes, a rotation that is not a unit quaternion, or a malformed distance grid, or if a
    /// touch names an obstacle the world lacks or a link the robot lacks.
    pub fn upload(&self, worlds: &[World]) -> Result<Worlds> {
        for (i, w) in worlds.iter().enumerate() {
            w.check().map_err(|e| input!("world {i}: {e}"))?;
        }
        let touching = worlds.iter().any(|w| !w.touch.is_empty());
        Ok(Worlds {
            device: self.id,
            worlds: worlds.to_vec(),
            prepared: self.backend.upload(worlds)?,
            links: if touching {
                self.robot().links.iter().map(|l| l.name.clone()).collect()
            } else {
                vec![]
            },
        })
    }

    /// Pays now what the first calls would: builds the kernels a GPU builds when first needed (for
    /// constrained trajectory optimization and for distance grids), then runs one small call of
    /// each kind on the device's threads and GPU: a collision check, IK and a step of trajectory
    /// optimization. Call it once after making the device and before timing anything. A device
    /// made with [`Device::with_robot`] for a robot of the same shape (the same one, or one
    /// relocked) shares what it built; one for a robot of another shape, such as one holding an
    /// object, builds kernels of its own, so warm it up too. Buffers grow to the largest batch they
    /// have carried, so warm up with batches the size of those you time too.
    pub fn warmup(&self) -> Result<()> {
        self.backend.build_all()?;
        let robot = self.robot();
        let n = robot.dof();
        let worlds = self.upload(&[World::default()])?;
        let q = robot.default_q();
        self.evaluate(&worlds, &[0], q, &CollisionWeights::NONE)?;
        let target = (0..robot.hands())
            .map(|h| Some(robot.ee_pose(q, h)))
            .collect();
        let problem = crate::IkProblem {
            targets: target,
            ..crate::IkProblem::new(0, Pose::IDENTITY)
        };
        let ik = IkOptions {
            seeds: 1,
            iterations: 1,
            ..IkOptions::default()
        };
        crate::solve_ik(self, &worlds, &[problem], &ik)?;
        let points = 7;
        let mut paths = JointPaths::zeros(1, points, n);
        for p in 0..points {
            paths.path_mut(0)[p * n..(p + 1) * n].copy_from_slice(q);
        }
        let o = PlanOptions {
            iterations: 1,
            control_points: points,
            ..PlanOptions::default()
        };
        self.trajopt(&worlds, &[0], &[None], &mut paths, &o, None)
    }

    /// What runs the batches: the GPU adapter and its API, or the CPU's thread count.
    #[must_use]
    pub fn name(&self) -> String {
        self.backend.name()
    }

    /// The robot it plans for.
    #[must_use]
    pub fn robot(&self) -> &Robot {
        self.backend.robot()
    }

    /// Collision cost, its gradient and clearances for each configuration in `q` (`[items, dof]`)
    /// against `worlds[item_world[item]]`. Errors if `q` does not hold `dof` finite values per
    /// item, an item names a world outside `worlds`, or `worlds` were uploaded to another device.
    pub fn evaluate(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        q: &[f32],
        w: &CollisionWeights,
    ) -> Result<Evaluation> {
        self.check_batch(worlds, item_world, q.len(), self.robot().dof())?;
        finite(q, self.robot().dof())?;
        self.backend.evaluate(worlds, item_world, q, w)
    }

    /// What each configuration in `q` (`[items, dof]`) comes within `within` meters of in
    /// `worlds[item_world[item]]`, nearest first: every link and obstacle, and every pair of links
    /// checked against each other, whose spheres come that close. `within` 0 lists what collides,
    /// to say why a start or a goal is in collision. The checks' rules hold: links touch what the
    /// world lets them, links no joint moves are not checked against the world, and pairs the
    /// collision model ignores are left out. Distances are exact to primitives and interpolated in
    /// distance grids, as the checks see them, computed on the device's CPU threads from the copy
    /// of the worlds searches check, so both devices list the same. Errors as [`Device::evaluate`]
    /// does, or for a `within` or a configuration value that is not finite.
    pub fn contacts(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        q: &[f32],
        within: f32,
    ) -> Result<Vec<Vec<Contact>>> {
        let robot = self.robot();
        self.check_batch(worlds, item_world, q.len(), robot.dof())?;
        ensure_input!(
            within.is_finite(),
            "contacts within a finite distance, got {within}"
        );
        finite(q, robot.dof())?;
        let checked = self.backend.checked(worlds);
        let contacts = |(q, &world): (&[f32], &u32)| {
            crate::cpu::contacts(robot, &checked[world as usize], &robot.fk(q), within)
        };
        let configurations = q.chunks(robot.dof()).zip(item_world);
        if item_world.len() <= 1 {
            return Ok(configurations.map(contacts).collect());
        }
        Ok(self.install(|| {
            q.par_chunks(robot.dof())
                .zip(item_world.par_iter())
                .map(contacts)
                .collect()
        }))
    }

    /// The spheres fitted to held objects on this device's hardware.
    pub(crate) fn fits(&self) -> &Fits {
        &self.fits
    }

    /// Runs `op` on this device's threads, so that its parallel loops run there too.
    pub(crate) fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        self.backend.pool().install(op)
    }

    /// Runs every search to its end, or to `deadline` (see `edges`).
    pub(crate) fn search(
        &self,
        worlds: &Worlds,
        searches: &mut [&mut dyn Search],
        deadline: Option<Instant>,
    ) -> Result<()> {
        self.check_worlds(worlds)?;
        self.backend.search(worlds, searches, deadline);
        Ok(())
    }

    /// World and self clearance of each configuration (`[items, dof]`), without cost or gradient:
    /// what validity checks need. Exact up to zero (see [`Evaluation`]).
    pub(crate) fn clearance(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        q: &[f32],
    ) -> Result<Vec<[f32; 2]>> {
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
        ensure_input!(
            log_odds.len() == grid.points(),
            "{} log-odds for {} points",
            log_odds.len(),
            grid.points()
        );
        self.backend.integrate(grid, log_odds, image, robot)
    }

    pub(crate) fn ik(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        targets: &[Option<Pose>],
        held: &[bool],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 4]>> {
        self.check_batch(worlds, item_world, q.len(), self.robot().dof())?;
        let (items, hands) = (item_world.len(), self.robot().hands());
        let problems = targets.len() / hands;
        ensure_input!(
            targets.len() == problems * hands
                && (items == 0 || (problems > 0 && items % problems == 0)),
            "{} IK targets for {items} items of {hands} hands",
            targets.len()
        );
        ensure_input!(
            held.is_empty() || held.len() == q.len(),
            "{} held flags for {} joint values",
            held.len(),
            q.len()
        );
        self.backend.ik(worlds, item_world, targets, held, q, o)
    }

    /// Optimizes `paths` in place, each in its world, keeping its constraints (`Region::inset`) if
    /// it has any; stops early once `deadline` passes.
    pub(crate) fn trajopt(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        item_region: &[Option<Arc<Region>>],
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
        self.check_batch(
            worlds,
            item_world,
            paths.positions.len(),
            paths.points * paths.dof,
        )?;
        debug_assert_eq!(item_region.len(), item_world.len());
        self.backend
            .trajopt(worlds, item_world, item_region, paths, o, deadline)
    }

    /// Whether `worlds` were uploaded to this device, for links named as this robot's: their
    /// touch sets hold link indices.
    fn check_worlds(&self, worlds: &Worlds) -> Result<()> {
        ensure_input!(
            worlds.device == self.id,
            "these worlds were uploaded to a different device"
        );
        let links = &self.robot().links;
        if let Some((i, (was, is))) = worlds
            .links
            .iter()
            .zip(links)
            .enumerate()
            .find(|(_, (was, is))| **was != is.name)
        {
            return Err(input!(
                "these worlds let links touch obstacles, and link {i} was '{was}' when they were \
                 uploaded but is '{}' now: upload them again for this robot",
                is.name
            ));
        }
        Ok(())
    }

    /// The device, shape and index invariants every batch must satisfy before it reaches a backend.
    fn check_batch(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        values: usize,
        per_item: usize,
    ) -> Result<()> {
        self.check_worlds(worlds)?;
        let items = item_world.len();
        ensure_input!(
            values == items * per_item,
            "{items} items need {} values ({per_item} each), got {values}",
            items * per_item
        );
        if let Some((item, &world)) = item_world
            .iter()
            .enumerate()
            .find(|&(_, &w)| w as usize >= worlds.len())
        {
            return Err(input!(
                "item {item} references world {world}, but only {} worlds were given",
                worlds.len()
            ));
        }
        Ok(())
    }
}

/// That every value of the configurations `q` (`[items, dof]`) is finite: a NaN compares as clear of
/// everything.
fn finite(q: &[f32], dof: usize) -> Result<()> {
    if let Some(i) = q.iter().position(|v| !v.is_finite()) {
        return Err(input!(
            "item {} has a configuration value that is not finite: {}",
            i / dof,
            q[i]
        ));
    }
    Ok(())
}

/// The parallel inner loops each device implements. Algorithm modules own seeding, validation
/// and selection so both implementations see identical inputs.
pub(crate) trait Backend: Send + Sync {
    fn name(&self) -> String;
    fn robot(&self) -> &Robot;
    /// The threads the backend's batched work runs on.
    fn pool(&self) -> &ThreadPool;
    /// The backend's own form of `worlds`, which [`Worlds::prepared`] hands back to it.
    fn upload(&self, worlds: &[World]) -> Result<Box<dyn Any + Send + Sync>>;
    /// The same backend for another robot, sharing the hardware and the form of worlds.
    fn with_robot(&self, robot: &Robot) -> Result<Box<dyn Backend>>;
    /// `worlds` as the CPU checks them: what searches run against on either backend.
    fn checked<'w>(&self, worlds: &'w Worlds) -> &'w CpuWorlds;
    /// Builds every kernel the backend would otherwise build the first time it is needed.
    fn build_all(&self) -> Result<()>;
    fn evaluate(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        q: &[f32],
        w: &CollisionWeights,
    ) -> Result<Evaluation>;
    /// `[world, self]` clearance per configuration, computed as `evaluate` does with zero weights
    /// and margins, without the gradient.
    fn clearance(&self, worlds: &Worlds, item_world: &[u32], q: &[f32]) -> Result<Vec<[f32; 2]>>;
    /// Runs every search to its end, or to `deadline`, each on one CPU thread (see `edges`).
    fn search(&self, worlds: &Worlds, searches: &mut [&mut dyn Search], deadline: Option<Instant>);
    /// Runs IK in place on `q` (`[items, dof]`) toward `targets` (`[problems, hands]`, `None` for a
    /// hand held still; each problem's consecutive items, as many to every problem, share its targets), the joints
    /// `held` marks (`[items, dof]`, or empty for none) staying put;
    /// returns each item's largest position and rotation errors over its targeted hands, then
    /// the `[world, self]` clearance of its final configuration (as `clearance` gives it):
    /// `[position error, rotation error, world clearance, self clearance]`.
    fn ik(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        targets: &[Option<Pose>],
        held: &[bool],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 4]>>;
    /// Optimizes each path's control points in place, holding the three at each end fixed, and
    /// stops between rounds once `deadline` passes.
    fn trajopt(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        item_region: &[Option<Arc<Region>>],
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
