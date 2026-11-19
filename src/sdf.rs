//! Signed distance grids: obstacles of any shape, built from meshes, point clouds or depth images.
//!
//! A grid holds distances at regular points as half floats, and both devices interpolate them the
//! same way (`grid_distance` here and in kernels.wgsl). The builders make grids conservative. Each
//! point holds at most the true signed distance of the geometry (for point clouds and depth
//! images, of the solid voxels as cubes), lowered by half a voxel diagonal: the most trilinear
//! interpolation can overestimate by. A built grid therefore never reads farther from an obstacle
//! than its geometry, and walls thinner than a voxel still block. The price is that obstacles grow,
//! by up to a voxel diagonal for meshes and about three voxels for points and depth images.
//! Inside a mesh of several shells the distance is to the nearest face, one buried in another
//! shell too: there a grid reads solid, and less deep than the solid is.
//!
//! Point clouds, depth images and occupancy maps become grids through an exact distance transform
//! of their solid voxels, which runs on the [`Device`] the builder is given.

use std::borrow::Cow;
use std::num::FpCategory;

use glam::{IVec3, UVec3, Vec3, Vec4};
use half::f16;
use parry3d::shape::{TriMesh as ParryMesh, TriMeshFlags};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::description::TriMesh;
use crate::device::Device;
use crate::error::{Error, Result, ensure_input, input};
use crate::held::Held;
use crate::robot::Robot;
use crate::types::Pose;
use crate::winding::Crossings;
use crate::world::{Distance, Obstacle};

/// Grids with more points are refused; 2^26 half floats take 128 MiB.
const MAX_POINTS: u64 = 1 << 26;
/// Grids whose diagonal reaches 4,096 voxels are refused.
const MAX_DIAGONAL: f64 = (1 << 24) as f64;

/// How builders lay out a grid.
#[derive(Clone, Copy, Debug)]
pub struct SdfOptions {
    /// Spacing of the grid points (meters).
    pub voxel: f32,
    /// How far the grid reaches beyond the geometry (meters). Beyond the grid, distances are
    /// extrapolated (see [`SdfGrid`]); keep this at least the collision margin plus the robot's
    /// largest sphere radius.
    pub padding: f32,
}

impl Default for SdfOptions {
    fn default() -> Self {
        Self {
            voxel: 0.01,
            padding: 0.15,
        }
    }
}

/// Pinhole intrinsics of a depth camera, in pixels.
#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    /// Focal length along the image's rows.
    pub fx: f32,
    /// Focal length along its columns.
    pub fy: f32,
    /// The principal point's column.
    pub cx: f32,
    /// The principal point's row.
    pub cy: f32,
}

/// What [`SdfGrid::from_depth`] assumes about space hidden behind observed surfaces, and
/// [`OccupancyMap::grid`] about space no image has observed.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Occlusion {
    /// Only observed surfaces are obstacles.
    Free,
    /// Everything behind an observed surface, as seen from the camera, is solid, the robot's
    /// shadow included.
    Occupied,
    /// As `Occupied`, except the space only the robot hides (with a [`RobotMask`]), which is
    /// free; in a map, space no image has hidden behind another surface either. No camera sees
    /// past the arm, so its shadow, solid, walls in the space it reaches through next. What the
    /// shadow holds is unknown: this trades that for being able to move.
    OccupiedExceptRobotShadow,
    /// Within this distance (meters) behind an observed surface, in depth as seen from the camera,
    /// is solid; past it, the robot's shadow and pixels without a reading are free. In a map, space
    /// no image has observed is solid within this distance of an occupied voxel, in any direction
    /// (in front of a surface too, where its pixels had no reading), and the robot's shadow is
    /// free.
    Band(f32),
}

impl Occlusion {
    /// How far behind a surface is solid by the band (`Band`), or 0.
    pub(crate) fn band(self) -> f32 {
        match self {
            Occlusion::Band(distance) => distance,
            _ => 0.0,
        }
    }

    /// That a band, if this is one, is finite and not negative.
    fn check(self) -> Result<()> {
        let band = self.band();
        ensure_input!(
            band.is_finite() && band >= 0.0,
            "an occlusion band must be finite and not negative, got {band} m"
        );
        Ok(())
    }
}

/// One depth image: z distances in meters, row by row, `width` pixels per row, with zero, negative
/// or non-finite values for pixels without a reading. `camera` places the camera (x right, y down,
/// z forward, as in OpenCV) in the frame grids and maps are built in.
#[derive(Clone, Copy, Debug)]
pub struct DepthImage<'depth> {
    /// Meters along the camera's z axis, `[height, width]`.
    pub depth: &'depth [f32],
    /// Pixels per row.
    pub width: usize,
    /// The camera's pinhole model.
    pub intrinsics: Intrinsics,
    /// The camera's pose in the frame grids and maps are built in.
    pub camera: Pose,
}

impl DepthImage<'_> {
    fn check(&self) -> Result<()> {
        ensure_input!(
            self.width > 0 && self.depth.len().is_multiple_of(self.width),
            "{} depth values do not form rows of {}",
            self.depth.len(),
            self.width
        );
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        ensure_input!(
            fx > 0.0 && fy > 0.0 && cx.is_finite() && cy.is_finite(),
            "invalid intrinsics {:?}",
            self.intrinsics
        );
        ensure_input!(
            self.camera.is_unit(),
            "the camera's pose must be finite with a unit rotation, got {:?}",
            self.camera
        );
        Ok(())
    }

    /// The image with its camera's rotation made unit ([`Pose::rigid`]): one a little off unit
    /// would scale every depth the grid points are compared at.
    pub(crate) fn rigid(&self) -> Self {
        Self {
            camera: self.camera.rigid(),
            ..*self
        }
    }

    fn height(&self) -> usize {
        self.depth.len() / self.width
    }

    /// Pixel `i`'s depth where it has a reading: positive and finite, which a float is when its
    /// bits lie between zero's and infinity's. One comparison, in every caller: asked as
    /// `is_finite() && > 0.0`, the compiler tests the float's class, by a longer way in some
    /// callers than in others.
    #[inline(always)]
    fn reading(&self, i: usize) -> Option<f32> {
        let z = self.depth[i];
        (z.to_bits().wrapping_sub(1) < f32::INFINITY.to_bits() - 1).then_some(z)
    }

    /// The pixel `p` (world frame) projects to, and its depth; `None` behind or beside the camera.
    fn project(&self, p: Vec3) -> Option<(usize, f32)> {
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        let c = self.camera.rotation.inverse() * (p - self.camera.position);
        if c.z <= 0.0 {
            return None;
        }
        let (u, v) = (
            (fx * c.x / c.z + cx + 0.5).floor(),
            (fy * c.y / c.z + cy + 0.5).floor(),
        );
        (u >= 0.0 && v >= 0.0 && u < self.width as f32 && v < self.height() as f32)
            .then(|| (v as usize * self.width + u as usize, c.z))
    }

    /// Pixel `i`'s reading as a world point, unless it is the robot's.
    pub(crate) fn point(&self, i: usize, robot: Option<&RobotMask>) -> Option<Vec3> {
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        let z = self
            .reading(i)
            .filter(|_| !robot.is_some_and(|r| r.on_robot[i]))?;
        let (u, v) = ((i % self.width) as f32, (i / self.width) as f32);
        Some(
            self.camera.rotation * Vec3::new((u - cx) / fx * z, (v - cy) / fy * z, z)
                + self.camera.position,
        )
    }

    /// Each pixel's reading as a point in the frame the camera is posed in, row by row: pixel `i`
    /// is column `i % width` of row `i / width`, its centre at those whole coordinates. `None`
    /// where the pixel has no reading, or its reading is the robot's (`robot`, this image's mask).
    pub fn points(&self, robot: Option<&RobotMask>) -> Result<Vec<Option<Vec3>>> {
        self.check_with(robot)?;
        let image = self.rigid();
        let pixels = 0..self.depth.len();
        Ok(pixels.map(|i| image.point(i, robot)).collect())
    }

    /// What the image says about the point `p` of a grid with `voxel` spacing, with the space the
    /// robot fills free: `inside` says whether `p` is in it ([`RobotMask::inside`]). `classify` in
    /// grids.wgsl mirrors it. `BAND` compiles in the test for points within `band` behind a
    /// reading (`Seen::Near`); `MASK` keeps the robot's checks out of images without a mask.
    fn classify<const BAND: bool, const MASK: bool>(
        &self,
        robot: Option<&RobotMask>,
        inside: bool,
        p: Vec3,
        voxel: f32,
        band: f32,
    ) -> Seen {
        let robot = if MASK { robot } else { None };
        let seen = if BAND || !MASK {
            self.sees::<BAND, MASK>(robot, p, voxel, band)
        } else {
            self.sees_without_band(robot, p, voxel)
        };
        match seen {
            Seen::Near | Seen::Hidden | Seen::Shadow | Seen::Unseen if MASK && inside => Seen::Free,
            seen => seen,
        }
    }

    /// Masked, non-band projection stays out of line; other classifiers keep their own copies.
    #[inline(never)]
    fn sees_without_band(&self, robot: Option<&RobotMask>, p: Vec3, voxel: f32) -> Seen {
        self.sees::<false, true>(robot, p, voxel, 0.0)
    }

    /// What the image shows of the point `p`. `sees` in grids.wgsl mirrors it.
    #[inline(always)]
    fn sees<const BAND: bool, const MASK: bool>(
        &self,
        robot: Option<&RobotMask>,
        p: Vec3,
        voxel: f32,
        band: f32,
    ) -> Seen {
        let half = 0.5 * voxel;
        let Some((i, z_voxel)) = self.project(p) else {
            return Seen::Unseen;
        };
        let Some(z) = self.reading(i) else {
            return Seen::Unseen;
        };
        if MASK && robot.is_some_and(|r| r.on_robot[i]) {
            // The pixel sees the robot.
            if z_voxel < z - half {
                Seen::Free
            } else {
                Seen::Shadow
            }
        } else if z_voxel < z - half {
            Seen::Free
        } else if z_voxel <= z + half {
            Seen::Surface
        } else if BAND && z_voxel <= z + half + band {
            Seen::Near
        } else {
            Seen::Hidden
        }
    }

    /// Why the image, with this robot mask, cannot be used, if it cannot.
    pub(crate) fn check_with(&self, robot: Option<&RobotMask>) -> Result<()> {
        self.check()?;
        if let Some(r) = robot {
            let n = r.on_robot.len();
            ensure_input!(
                n == self.depth.len(),
                "a robot mask of {n} pixels for an image of {}",
                self.depth.len()
            );
        }
        Ok(())
    }

    /// Which of this image's readings are of `robot` at configuration `q`, or of what it holds:
    /// those whose points lie inside its collision spheres grown by `padding`, or within `padding`
    /// of an object of `holding` where its hand holds it (the fingers as each `Held` locks them).
    /// Pass it to [`SdfGrid::from_depth`] or [`OccupancyMap::integrate`]: readings of the robot
    /// are then not obstacles, and the space it fills is free, so it is not an obstacle to itself
    /// even where it hides space that counts as solid. A reading past the robot is the world's,
    /// however near the robot its ray went: a camera the hand carries stands inside or beside the
    /// hand's grown spheres and keeps what it sees. Grids from depth images grow toward collision
    /// by about three voxels; pad by more, or the robot's surface reads as touching the space it
    /// hides.
    pub fn robot_mask(
        &self,
        robot: &Robot,
        q: &[f32],
        padding: f32,
        holding: &[Held],
    ) -> Result<RobotMask> {
        self.check()?;
        let camera = self.rigid().camera;
        let to_camera = camera.rotation.inverse();
        ensure_input!(
            q.len() == robot.dof(),
            "q has {} values, the robot {} joints",
            q.len(),
            robot.dof()
        );
        ensure_input!(
            padding.is_finite() && padding >= 0.0,
            "the padding must not be negative, got {padding}"
        );
        let mut robot = Cow::Borrowed(robot);
        for held in holding {
            held.check_on(&robot, "a robot mask")?;
            if !held.locked.is_empty() {
                robot = Cow::Owned(robot.relock(&held.locked)?);
            }
        }
        let fk = robot.fk(q);
        let spheres: Vec<Vec4> = robot
            .spheres
            .iter()
            .map(|s| (fk.rot[s.link] * s.center + fk.pos[s.link]).extend(s.radius + padding))
            .collect();
        let held = holding.iter().map(|held| {
            let pose = robot
                .ee_pose(q, held.hand)
                .mul_pose(held.grasp.rigid().inverse());
            held.object.placed(pose)
        });
        let mask = RobotMask {
            on_robot: vec![],
            spheres,
            held: held.collect(),
            padding,
        };
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        let (width, height) = (self.width, self.height());
        // Each ball in the camera's frame.
        let balls: Vec<(Vec3, f32)> = (mask.balls())
            .map(|ball| (to_camera * (ball.truncate() - camera.position), ball.w))
            .collect();
        // The nearest reading bounds the pixels of a ball that reaches to the camera or behind
        // it, as the hand's do for a camera it carries; a camera clear of every ball needs none.
        let nearest = if balls.iter().any(|(center, r)| center.z - r <= 0.0) {
            let rows = self.depth.par_chunks(width);
            rows.map(|row| {
                let readings = row.iter().filter(|z| z.is_finite() && **z > 0.0);
                readings.fold(f32::INFINITY, |least, &z| least.min(z))
            })
            .reduce(|| f32::INFINITY, f32::min)
        } else {
            0.0
        };
        // The pixels whose readings can lie in each ball: the projections of its bounding box's
        // sides at the least and the greatest depth a reading in it has. None where every
        // reading is farther than the ball.
        let balls: Vec<_> = (balls.into_iter().enumerate())
            .filter_map(|(k, (center, r))| {
                let (near, far) = ((center.z - r).max(nearest), center.z + r);
                if far <= near {
                    return None;
                }
                let span = |c: f32, f: f32, k: f32, n: usize| {
                    let ends = [(c - r) / near, (c - r) / far, (c + r) / near, (c + r) / far]
                        .map(|x| f * x + k);
                    let lo = ends
                        .iter()
                        .fold(f32::INFINITY, |m, &e| m.min(e))
                        .floor()
                        .max(0.0);
                    let hi = ends.iter().fold(f32::NEG_INFINITY, |m, &e| m.max(e)).ceil() + 1.0;
                    lo.min(n as f32) as usize..hi.clamp(0.0, n as f32) as usize
                };
                let (us, vs) = (
                    span(center.x, fx, cx, width),
                    span(center.y, fy, cy, height),
                );
                Some((k, center, r, us, vs))
            })
            .collect();
        let mut on_robot = vec![false; self.depth.len()];
        let rows = on_robot.par_chunks_mut(width).enumerate();
        rows.for_each(|(v, row)| {
            for (k, center, r, us, vs) in &balls {
                if !vs.contains(&v) {
                    continue;
                }
                for u in us.clone() {
                    let Some(z) = self.reading(v * width + u).filter(|_| !row[u]) else {
                        continue;
                    };
                    // The reading in the camera's frame.
                    let p = Vec3::new((u as f32 - cx) / fx * z, (v as f32 - cy) / fy * z, z);
                    row[u] = (p - *center).length_squared() < r * r
                        && mask.holds(*k, camera.rotation * p + camera.position);
                }
            }
        });
        Ok(RobotMask { on_robot, ..mask })
    }
}

/// Which of one depth image's readings are a robot's ([`DepthImage::robot_mask`]).
#[derive(Clone, Debug)]
pub struct RobotMask {
    /// Per pixel, whether its reading is of the robot or of what it holds: its point lies inside
    /// one of the robot's grown spheres, or within the padding of a held object. False where the
    /// pixel has no reading.
    pub on_robot: Vec<bool>,
    /// The grown spheres in the image's frame: centre and radius.
    spheres: Vec<Vec4>,
    /// What the robot holds, in the image's frame.
    held: Vec<Obstacle>,
    /// How far from a held object a point is its.
    padding: f32,
}

impl RobotMask {
    /// A ball around each sphere and each held object, in that order: the sphere itself, and for
    /// an object every point within the padding of it.
    fn balls(&self) -> impl Iterator<Item = Vec4> + '_ {
        let held = self.held.iter().map(|object| {
            let reach = match *object {
                Obstacle::Cuboid { half_extents, .. } => half_extents.length(),
                Obstacle::Sphere { radius, .. } => radius,
                Obstacle::Cylinder {
                    radius,
                    half_height,
                    ..
                } => radius.hypot(half_height),
                Obstacle::Capsule {
                    radius,
                    half_length,
                    ..
                } => radius + half_length,
                // No grid is held (`Held::check_on`).
                Obstacle::Sdf { .. } => 0.0,
            };
            object.pose().position.extend(reach + self.padding)
        });
        self.spheres.iter().copied().chain(held)
    }

    /// Whether `p`, a point of ball `k` of `balls`, is the robot's: every point of a sphere's is,
    /// and of an object's those within the padding of the object.
    fn holds(&self, k: usize, p: Vec3) -> bool {
        (k.checked_sub(self.spheres.len()))
            .is_none_or(|object| self.held[object].distance(p).value < self.padding)
    }

    /// Which points of `grid` are inside the robot, in one of the grown spheres or within the
    /// padding of what it holds: a bit per point, x fastest, the first the lowest of the first
    /// word ([`set`] reads one). Each ball marks the points it holds among those of its box, a
    /// slab of the grid at a time, so the cost goes by the balls' volumes and not by the grid's
    /// points times the spheres: a robot fitted with several hundred costs a map's update what
    /// one of a few dozen does. Both devices read these bits (`inside_robot` in grids.wgsl).
    pub(crate) fn inside(&self, grid: &Layout) -> Vec<u32> {
        let [nx, ny, nz] = grid.dims.to_array().map(|d| d as usize);
        let (origin, voxel) = (grid.origin, grid.voxel);
        // The points along one axis within `reach` of `center`, and one more to either side: a
        // point's coordinate is rounded, and so are these.
        let span = |center: f32, reach: f32, origin: f32, n: usize| {
            let lo = ((center - reach - origin) / voxel).floor() - 1.0;
            let hi = ((center + reach - origin) / voxel).ceil() + 1.0;
            lo.max(0.0) as usize..(hi + 1.0).clamp(0.0, n as f32) as usize
        };
        let balls: Vec<Vec4> = self.balls().collect();
        // Each slab's bits from its own first point on; none for a slab no ball reaches.
        let slabs: Vec<Vec<u32>> = (0..nz)
            .into_par_iter()
            .map(|z| {
                let height = origin.z + z as f32 * voxel;
                let mut bits = vec![];
                for (k, &ball) in balls.iter().enumerate() {
                    // The disc the slab cuts the ball in, and each row's chord of it.
                    let across = ball.w * ball.w - (height - ball.z).powi(2);
                    if across < 0.0 && (height - ball.z).abs() > ball.w + voxel {
                        continue;
                    }
                    bits.resize((nx * ny).div_ceil(32), 0);
                    let disc = across.max(0.0).sqrt();
                    for y in span(ball.y, disc, origin.y, ny) {
                        let along = across - (origin.y + y as f32 * voxel - ball.y).powi(2);
                        for x in span(ball.x, along.max(0.0).sqrt(), origin.x, nx) {
                            let n = x + nx * y;
                            if set(&bits, n) {
                                continue;
                            }
                            let p = grid.point(n + nx * ny * z);
                            let holds = (p - ball.truncate()).length_squared() < ball.w * ball.w
                                && self.holds(k, p);
                            bits[n / 32] |= u32::from(holds) << (n % 32);
                        }
                    }
                }
                bits
            })
            .collect();
        // Each slab's bits moved to where its points lie among the grid's.
        let mut inside = vec![0; grid.points().div_ceil(32)];
        for (z, bits) in slabs.iter().enumerate() {
            let (word, shift) = (z * nx * ny / 32, z * nx * ny % 32);
            for (k, &bits) in bits.iter().enumerate().filter(|(_, bits)| **bits != 0) {
                inside[word + k] |= bits << shift;
                if shift > 0 && bits >> (32 - shift) != 0 {
                    inside[word + k + 1] |= bits >> (32 - shift);
                }
            }
        }
        inside
    }
}

/// Whether bit `n` of `bits` is set, the first the lowest of the first word.
#[inline(always)]
pub(crate) fn set(bits: &[u32], n: usize) -> bool {
    bits[n / 32] >> (n % 32) & 1 == 1
}

/// What a depth image says about a point.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Seen {
    /// Outside the image, or along a pixel without a reading.
    Unseen,
    /// In front of a reading, in front of the robot along a pixel that sees it, or inside the
    /// robot.
    Free,
    /// At a reading, within half a voxel.
    Surface,
    /// Behind a reading, within the band ([`Occlusion::Band`]) past the half voxel.
    Near,
    /// Behind a reading, past the band.
    Hidden,
    /// At or behind the robot, along a pixel that sees it.
    Shadow,
}

/// The points of a grid: `dims` per axis, `voxel` apart, from `origin`, x fastest.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub(crate) dims: UVec3,
    pub(crate) origin: Vec3,
    pub(crate) voxel: f32,
}

impl Layout {
    pub(crate) fn points(&self) -> usize {
        self.dims.element_product() as usize
    }

    /// The squared diagonal, in voxels: the largest squared distance between two points.
    pub(crate) fn diagonal(&self) -> u32 {
        (self.dims - 1).length_squared()
    }

    fn point(&self, n: usize) -> Vec3 {
        point(self.dims, self.origin, self.voxel, n as u32)
    }

    fn nearest(&self, p: Vec3) -> usize {
        nearest(self.dims, self.origin, self.voxel, p)
    }

    /// The points from the last before `lo` to the first after `hi` along each axis, as the
    /// indices of the first and the last: those that can lie in a box with these corners. None
    /// where the box lies off the grid.
    fn among(&self, lo: Vec3, hi: Vec3) -> Option<(UVec3, UVec3)> {
        let index = |p: Vec3| (p - self.origin) / self.voxel;
        let from = index(lo).floor().max(Vec3::ZERO);
        let to = index(hi).ceil().min((self.dims - 1).as_vec3());
        (from.cmple(to).all()).then(|| (from.as_uvec3(), to.as_uvec3()))
    }

    /// The index of the point `x`, `y`, `z` points along the axes.
    fn index(&self, at: UVec3) -> usize {
        (at.x + self.dims.x * (at.y + self.dims.y * at.z)) as usize
    }

    /// Whether `p` lies within half a voxel of the points.
    fn contains(&self, p: Vec3) -> bool {
        let half = 0.5 * self.voxel;
        p.cmpge(self.origin - half).all()
            && p.cmplt(self.origin + (self.dims - 1).as_vec3() * self.voxel + half)
                .all()
    }
}

/// Where a grid's occupancy comes from.
pub(crate) enum Occupancy<'source> {
    /// One flag per point.
    Given(&'source [bool]),
    /// A depth image, as [`SdfGrid::from_depth`] reads it.
    Seen {
        image: &'source DepthImage<'source>,
        robot: Option<&'source RobotMask>,
        behind: Occlusion,
    },
}

/// The stored value of a point `n` squared voxels from the nearest point of the other kind, for
/// every `n` up to `max`: a free point's in the low half of each word, an occupied point's in the
/// high half. Free points are at least `sqrt(n) - sqrt(3) / 2` voxels from the nearest occupied
/// cube; occupied points at most `sqrt(n) - 1 / 2` deep. Both bounds are then lowered by half a
/// voxel diagonal and rounded down to half floats (see the module documentation).
pub(crate) fn finishing_table(voxel: f32, max: u32) -> Vec<u32> {
    let (half_diagonal, shift) = (3f64.sqrt() / 2.0, voxel * 3f32.sqrt() / 2.0);
    (0..=max)
        .into_par_iter()
        .map(|n| {
            let d = f64::from(n).sqrt();
            let free = at_most(voxel * (d - half_diagonal) as f32 - shift).to_bits();
            let occupied = at_most(voxel * (0.5 - d) as f32 - shift).to_bits();
            u32::from(free) | u32::from(occupied) << 16
        })
        .collect()
}

/// Why a grid whose points are all of one kind, as its first, has no distances.
pub(crate) fn one_kind(first_occupied: bool) -> Error {
    match first_occupied {
        true => input!("every voxel is occupied; increase the padding"),
        false => input!("no voxel is occupied"),
    }
}

/// Each point's stored value from its occupancy and squared distance (`finish` in grids.wgsl).
pub(crate) fn finish(grid: &Layout, occupied: &[bool], squared: &[u32]) -> Result<Vec<u16>> {
    if squared[0] == FAR {
        return Err(one_kind(occupied[0]));
    }
    let table = finishing_table(grid.voxel, grid.diagonal());
    Ok(occupied
        .par_iter()
        .zip(squared)
        .map(|(&o, &n)| (table[n as usize] >> (16 * u32::from(o))) as u16)
        .collect())
}

/// Each point's occupancy as [`SdfGrid::from_depth`] decides it. `depth_occupancy` and
/// `depth_hits` in grids.wgsl mirror it.
pub(crate) fn depth_occupancy(
    grid: &Layout,
    image: &DepthImage,
    robot: Option<&RobotMask>,
    behind: Occlusion,
) -> Vec<bool> {
    // Select the band's and robot mask's checks once, outside the loop over grid points.
    let mut occupied = match (behind, robot.is_some()) {
        (Occlusion::Band(band), true) => classified::<true, true>(grid, image, robot, behind, band),
        (Occlusion::Band(band), false) => {
            classified::<true, false>(grid, image, robot, behind, band)
        }
        (_, true) => classified::<false, true>(grid, image, robot, behind, 0.0),
        (_, false) => classified::<false, false>(grid, image, robot, behind, 0.0),
    };
    // Dense images resolve surfaces finer than voxels project; their points are solid too.
    for n in hits(grid, image, robot, |_| true) {
        occupied[n] = true;
    }
    occupied
}

/// Each point's occupancy by how the image classifies it.
fn classified<const BAND: bool, const MASK: bool>(
    grid: &Layout,
    image: &DepthImage,
    robot: Option<&RobotMask>,
    behind: Occlusion,
    band: f32,
) -> Vec<bool> {
    let inside = robot
        .filter(|_| MASK)
        .map_or_else(Vec::new, |r| r.inside(grid));
    (0..grid.points())
        .into_par_iter()
        .map(|n| {
            let inside = MASK && set(&inside, n);
            match image.classify::<BAND, MASK>(robot, inside, grid.point(n), grid.voxel, band) {
                Seen::Surface | Seen::Near => true,
                Seen::Hidden => !BAND && behind != Occlusion::Free,
                Seen::Shadow => behind == Occlusion::Occupied,
                Seen::Free | Seen::Unseen => false,
            }
        })
        .collect()
}

/// The grid point holding each reading off the robot that `keep` accepts.
fn hits(
    grid: &Layout,
    image: &DepthImage,
    robot: Option<&RobotMask>,
    keep: impl Fn(Vec3) -> bool + Sync,
) -> Vec<usize> {
    (0..image.depth.len())
        .into_par_iter()
        .filter_map(|i| {
            image
                .point(i, robot)
                .filter(|&p| keep(p))
                .map(|p| grid.nearest(p))
        })
        .collect()
}

/// Fuses one image into a map's log-odds as [`OccupancyMap::integrate`] describes. `depth_hits`
/// and `map_update` in grids.wgsl mirror it.
pub(crate) fn integrate(
    grid: &Layout,
    log_odds: &mut [i8],
    image: &DepthImage,
    robot: Option<&RobotMask>,
) {
    let mut hit = vec![false; grid.points()];
    for n in hits(grid, image, robot, |p| grid.contains(p)) {
        hit[n] = true;
    }
    if robot.is_some() {
        integrated::<true>(grid, log_odds, image, robot, hit);
    } else {
        integrated::<false>(grid, log_odds, image, robot, hit);
    }
}

/// Updates the map with one image's classification, with masking selected before the voxel loop.
fn integrated<const MASK: bool>(
    grid: &Layout,
    log_odds: &mut [i8],
    image: &DepthImage,
    robot: Option<&RobotMask>,
    hit: Vec<bool>,
) {
    let inside = robot
        .filter(|_| MASK)
        .map_or_else(Vec::new, |r| r.inside(grid));
    log_odds
        .par_iter_mut()
        .zip(hit)
        .enumerate()
        .for_each(|(n, (l, hit))| {
            let seen = if MASK {
                image.classify::<false, true>(
                    robot,
                    set(&inside, n),
                    grid.point(n),
                    grid.voxel,
                    0.0,
                )
            } else {
                image.sees::<false, false>(None, grid.point(n), grid.voxel, 0.0)
            };
            let step = match seen {
                _ if hit => HIT,
                Seen::Surface => HIT,
                Seen::Free => MISS,
                // Only the robot hides it: noted, until an image observes it or another surface
                // hides it.
                Seen::Shadow => {
                    if *l == UNOBSERVED {
                        *l = SHADOWED;
                    }
                    return;
                }
                Seen::Hidden => {
                    if *l == UNOBSERVED || *l == SHADOWED {
                        *l = OCCLUDED;
                    }
                    return;
                }
                Seen::Near | Seen::Unseen => return,
            };
            *l = (if *l == UNOBSERVED || *l == SHADOWED || *l == OCCLUDED {
                0
            } else {
                *l
            } + step)
                .clamp(CLAMP.0, CLAMP.1);
        });
}

/// Signed distances (meters, negative inside) on a regular grid in its own frame: `dims` points
/// per axis, `voxel` apart, starting at `origin`, stored as half floats with x varying fastest.
///
/// Between points, distances are interpolated trilinearly. Outside the grid's box, the distance
/// is the value at the nearest point of the box plus the distance to that point. Collision checks
/// stay exact there as long as the box reaches at least `margin + largest sphere radius` beyond
/// every surface it holds ([`SdfOptions::padding`]): both the grid and the true distance then
/// report clearance beyond the margin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "GridParts")]
pub struct SdfGrid {
    pub(crate) dims: [u32; 3],
    pub(crate) voxel: f32,
    pub(crate) origin: Vec3,
    pub(crate) values: Vec<u16>,
    /// The least value on the faces of the grid's box: a point outside the box is at least this
    /// plus its distance to the box away, which rules the grid out for links far from it.
    #[serde(skip)]
    pub(crate) least_on_faces: f32,
}

/// A grid as serialized, checked like [`SdfGrid::try_new`] before it becomes one.
#[derive(Deserialize)]
struct GridParts {
    dims: [u32; 3],
    voxel: f32,
    origin: Vec3,
    values: Vec<u16>,
}

impl TryFrom<GridParts> for SdfGrid {
    type Error = Error;

    fn try_from(p: GridParts) -> Result<Self> {
        SdfGrid::checked(p.dims, p.voxel, p.origin, p.values)
    }
}

impl SdfGrid {
    /// A grid holding `distances` at `dims` points (`dims[0] * dims[1] * dims[2]` values, x
    /// fastest), rounded to the nearest half float: no conservative offset is applied.
    pub fn try_new(dims: [u32; 3], voxel: f32, origin: Vec3, distances: &[f32]) -> Result<Self> {
        let values = distances
            .iter()
            .map(|&d| flush(f16::from_f32(d)).to_bits())
            .collect();
        Self::checked(dims, voxel, origin, values)
    }

    /// The grid of `values` (half floats, x fastest), if it can be sampled.
    fn checked(dims: [u32; 3], voxel: f32, origin: Vec3, values: Vec<u16>) -> Result<Self> {
        let mut grid = Self {
            dims,
            voxel,
            origin,
            values,
            least_on_faces: 0.0,
        };
        grid.check()?;
        let [nx, ny, nz] = dims;
        let mut least = f32::INFINITY;
        for z in 0..nz {
            for y in 0..ny {
                let row = nx * (y + ny * z);
                let whole = z == 0 || z == nz - 1 || y == 0 || y == ny - 1;
                let step = if whole { 1 } else { nx - 1 };
                for x in (0..nx).step_by(step as usize) {
                    least = least.min(grid.value(row + x));
                }
            }
        }
        grid.least_on_faces = least;
        Ok(grid)
    }

    /// What a value outside the grid's box is at least, less the distance to the box: the least
    /// on its faces where that is negative, and a hair for the rounding of the checks that rely
    /// on it.
    pub(crate) fn sunk(&self) -> f32 {
        self.least_on_faces.min(0.0) - 1e-5
    }

    /// Half the extent of the grid's box, whose centre is `origin` plus this.
    pub(crate) fn half_extent(&self) -> Vec3 {
        (UVec3::from(self.dims) - 1).as_vec3() * (0.5 * self.voxel)
    }

    /// Why this grid cannot be sampled, if it cannot: the rules of [`SdfGrid::try_new`].
    pub(crate) fn check(&self) -> Result<()> {
        let (dims, voxel) = (self.dims, self.voxel);
        ensure_input!(
            dims.iter().all(|&d| d >= 2),
            "a grid needs at least 2 points per axis, got {dims:?}"
        );
        ensure_input!(
            voxel.is_finite() && voxel > 0.0,
            "the voxel size must be positive, got {voxel}"
        );
        ensure_input!(self.origin.is_finite(), "the grid origin must be finite");
        let count = dims.iter().map(|&d| u64::from(d)).product::<u64>();
        ensure_input!(
            count <= MAX_POINTS,
            "{dims:?} is {count} points, more than {MAX_POINTS}; use larger voxels"
        );
        ensure_input!(
            self.values.len() as u64 == count,
            "{dims:?} needs {count} distances, got {}",
            self.values.len()
        );
        let bad = self
            .values
            .iter()
            .map(|&v| f16::from_bits(v))
            .find(|h| !h.is_finite() || subnormal(*h));
        ensure_input!(
            bad.is_none(),
            "distance {bad:?} is not finite or is subnormal"
        );
        Ok(())
    }

    /// The distance field of a closed triangle mesh, in the mesh's frame: of the solid its surface
    /// winds around. A mesh of several closed shells that face the same way is their union where
    /// they overlap, and a shell facing in, inside one facing out, is a cavity; a mesh is the
    /// same solid whichever way it faces as a whole, and a shell that crosses itself is solid
    /// wherever it encloses space. Where two shells that face opposite ways overlap, the overlap
    /// reads free. Distances are to the nearest face, also inside a solid of several shells,
    /// where a face buried in another shell makes the depth read less than it is. Meshes that
    /// are open or not manifold have no inside and are refused.
    pub fn from_mesh(vertices: &[Vec3], triangles: &[[u32; 3]], o: &SdfOptions) -> Result<Self> {
        use parry3d::query::PointQuery;

        ensure_input!(!triangles.is_empty(), "the mesh has no triangles");
        ensure_input!(
            triangles
                .iter()
                .flatten()
                .all(|&i| (i as usize) < vertices.len()),
            "a triangle refers to a vertex past the {} given",
            vertices.len()
        );
        ensure_input!(
            vertices.iter().all(|v| v.is_finite()),
            "mesh vertices must be finite"
        );
        // The side does not depend on which way the mesh faces; the nearest face's point does, in
        // its last bit, so the mesh is turned to face out as a whole.
        let mut mesh = TriMesh {
            vertices: vertices.to_vec(),
            triangles: triangles.to_vec(),
        };
        mesh.orient_outward();
        let flags = TriMeshFlags::HALF_EDGE_TOPOLOGY
            | TriMeshFlags::MERGE_DUPLICATE_VERTICES
            | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES;
        // `with_flags` drops topology errors, so the flags are set separately.
        let mut solid = ParryMesh::new(mesh.vertices, mesh.triangles).map_err(|e| input!("{e}"))?;
        solid
            .set_flags(flags)
            .map_err(|e| input!("the mesh is not a manifold surface: {e}"))?;
        let topology = solid.topology().expect("set_flags computed the topology");
        let open = topology
            .half_edges
            .iter()
            .filter(|h| h.twin == u32::MAX)
            .count();
        ensure_input!(
            open == 0,
            "the mesh is not closed ({open} edges border a single triangle), so it has no inside"
        );
        let (lo, hi) = bounds(vertices);
        let Layout { dims, origin, .. } = layout(lo, hi, o)?;
        // The grid's coordinates along each axis, as `point` computes them.
        let along = |axis: usize| -> Vec<f32> {
            let points = 0..dims[axis];
            points.map(|i| origin[axis] + i as f32 * o.voxel).collect()
        };
        let crossings = Crossings::new(
            vertices,
            triangles,
            &along(1),
            &along(2),
            /* parallel */ true,
        );
        let xs = along(0);
        let mut distances = vec![0.0; dims.element_product() as usize];
        let rows = distances.par_chunks_mut(dims.x as usize).enumerate();
        rows.for_each(|(row, distances)| {
            // The nearest face gives the distance and the winding number the side: the face's
            // own normal says which side only for one shell that does not cross itself.
            let points = (row as u32 * dims.x..).zip(crossings.numbers(row, &xs));
            for (d, (n, winding)) in distances.iter_mut().zip(points) {
                let p = point(dims, origin, o.voxel, n);
                let proj = solid.project_local_point(p, false);
                let nearest = (p - proj.point).length();
                *d = if winding == 0 { nearest } else { -nearest };
            }
        });
        Self::conservative(dims, o.voxel, origin, distances)
    }

    /// The distance field of surface points, such as a point cloud. Voxels holding a point are
    /// solid; the inside of a closed surface reads as free beyond its shell of voxels.
    pub fn from_points(device: &Device, points: &[Vec3], o: &SdfOptions) -> Result<Self> {
        ensure_input!(!points.is_empty(), "no points");
        ensure_input!(
            points.iter().all(|p| p.is_finite()),
            "points must be finite"
        );
        let (lo, hi) = bounds(points);
        let grid = layout(lo, hi, o)?;
        let mut occupied = vec![false; grid.points()];
        for &p in points {
            occupied[grid.nearest(p)] = true;
        }
        Self::from_occupancy(device, &grid, &occupied)
    }

    /// The distance field seen in one depth image. The grid spans the observed points plus the
    /// padding.
    ///
    /// Unobserved space follows one rule: space outside the image and along pixels without a
    /// reading is free, and space behind observed surfaces is what `behind` says. Each voxel is
    /// classified by projecting its center into the image, so coarse sensors such as 8x8
    /// time-of-flight arrays give solid surfaces, not scattered points. With a [`RobotMask`],
    /// the readings it marks as the robot's are not obstacles: space in front of them and the
    /// space the robot fills are free, and space behind them is what `behind` says.
    pub fn from_depth(
        device: &Device,
        image: &DepthImage,
        robot: Option<&RobotMask>,
        behind: Occlusion,
        o: &SdfOptions,
    ) -> Result<Self> {
        image.check_with(robot)?;
        behind.check()?;
        let image = &image.rigid();
        let (lo, hi) = (0..image.depth.len())
            .into_par_iter()
            .filter_map(|i| image.point(i, robot))
            .fold(
                || (Vec3::INFINITY, Vec3::NEG_INFINITY),
                |(lo, hi), p| (lo.min(p), hi.max(p)),
            )
            .reduce(
                || (Vec3::INFINITY, Vec3::NEG_INFINITY),
                |(a, b), (c, d)| (a.min(c), b.max(d)),
            );
        ensure_input!(
            lo.cmple(hi).all(),
            "the depth image has no valid pixels off the robot"
        );
        let grid = layout(lo, hi, o)?;
        let values = device.grid_values(
            &grid,
            Occupancy::Seen {
                image,
                robot,
                behind,
            },
        )?;
        Self::from_values(&grid, values)
    }

    /// Grid points along x, y and z.
    #[must_use]
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// Meters between neighbouring grid points.
    #[must_use]
    pub fn voxel(&self) -> f32 {
        self.voxel
    }

    /// The grid's first point, its box's lower corner, in its frame.
    #[must_use]
    pub fn lower_corner(&self) -> Vec3 {
        self.origin
    }

    /// The grid's last point, its box's upper corner, in its frame.
    #[must_use]
    pub fn upper_corner(&self) -> Vec3 {
        self.origin + (UVec3::from(self.dims) - 1).as_vec3() * self.voxel
    }

    /// Signed distance at `p` (in the grid's frame), and its gradient.
    #[must_use]
    pub fn distance(&self, p: Vec3) -> Distance {
        let (value, gradient) = grid_distance(self, p);
        Distance { value, gradient }
    }

    #[inline]
    fn value(&self, n: u32) -> f32 {
        f16::from_bits(self.values[n as usize]).to_f32()
    }

    /// Exact distances lowered by half a voxel diagonal (see the module documentation), each
    /// rounded down to a half float so the stored value never exceeds it.
    fn conservative(dims: UVec3, voxel: f32, origin: Vec3, distances: Vec<f32>) -> Result<Self> {
        let shift = voxel * 3f32.sqrt() / 2.0;
        let values = distances
            .iter()
            .map(|&d| at_most(d - shift).to_bits())
            .collect();
        Self::checked(dims.into(), voxel, origin, values)
    }

    /// Signed distances to the occupied voxels as solid cubes, from their centers' distances.
    /// A free point is at least `d - sqrt(3) / 2` voxels from the nearest cube when its center is
    /// `d` away; an occupied point is at most `d - 1 / 2` deep when the nearest free center is `d`
    /// away. Both bounds keep each value at or below the true distance.
    fn from_occupancy(device: &Device, grid: &Layout, occupied: &[bool]) -> Result<Self> {
        Self::from_values(grid, device.grid_values(grid, Occupancy::Given(occupied))?)
    }

    fn from_values(grid: &Layout, values: Vec<u16>) -> Result<Self> {
        Self::checked(grid.dims.into(), grid.voxel, grid.origin, values)
    }
}

fn subnormal(h: f16) -> bool {
    h.classify() == FpCategory::Subnormal
}

/// GPUs may flush subnormal halves to zero; storing them as zero keeps devices equal.
fn flush(h: f16) -> f16 {
    if subnormal(h) { f16::ZERO } else { h }
}

/// The largest half float that is not subnormal and not above `d` (infinite when `d` overflows).
fn at_most(d: f32) -> f16 {
    let mut h = f16::from_f32(d);
    if h.to_f32() > d {
        h = match h.to_bits() {
            0x0000 | 0x8000 => f16::from_bits(0x8001),
            bits if h.is_sign_negative() => f16::from_bits(bits + 1),
            bits => f16::from_bits(bits - 1),
        };
    }
    match subnormal(h) {
        true if h.is_sign_negative() => -f16::MIN_POSITIVE,
        true => f16::ZERO,
        false => h,
    }
}

/// Must stay in sync with `grid_distance` in kernels.wgsl.
#[inline(always)]
pub(crate) fn grid_distance(g: &SdfGrid, p: Vec3) -> (f32, Vec3) {
    let top = (UVec3::from(g.dims) - 1).as_vec3();
    let x = (p - g.origin) / g.voxel;
    let c = x.clamp(Vec3::ZERO, top);
    let i = c.floor().min(top - 1.0);
    let f = c - i;
    let (sy, sz) = (g.dims[0], g.dims[0] * g.dims[1]);
    let n = i.x as u32 + sy * i.y as u32 + sz * i.z as u32;
    let v = |offset: u32| g.value(n + offset);
    let (v000, v100, v010, v110) = (v(0), v(1), v(sy), v(sy + 1));
    let (v001, v101, v011, v111) = (v(sz), v(sz + 1), v(sz + sy), v(sz + sy + 1));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let (x00, x10) = (lerp(v000, v100, f.x), lerp(v010, v110, f.x));
    let (x01, x11) = (lerp(v001, v101, f.x), lerp(v011, v111, f.x));
    let (y0, y1) = (lerp(x00, x10, f.y), lerp(x01, x11, f.y));
    let dx = lerp(
        lerp(v100 - v000, v110 - v010, f.y),
        lerp(v101 - v001, v111 - v011, f.y),
        f.z,
    );
    let inside = Vec3::new(dx, lerp(x10 - x00, x11 - x01, f.z), y1 - y0) / g.voxel;
    // Outside the box, the distance grows with the distance to the box along the clamped axes.
    let outside = (x - c) * g.voxel;
    let away = outside.length();
    let grad = Vec3::select(x.cmpne(c), outside / away.max(1e-30), inside);
    (lerp(y0, y1, f.z) + away, grad)
}

/// `grid_distance`'s distance alone, the same to the bit: what a check that takes no gradient
/// looks up, for less work.
#[inline(always)]
pub(crate) fn grid_value(g: &SdfGrid, p: Vec3) -> f32 {
    let top = (UVec3::from(g.dims) - 1).as_vec3();
    let x = (p - g.origin) / g.voxel;
    let c = x.clamp(Vec3::ZERO, top);
    let i = c.floor().min(top - 1.0);
    let f = c - i;
    let (sy, sz) = (g.dims[0], g.dims[0] * g.dims[1]);
    let n = i.x as u32 + sy * i.y as u32 + sz * i.z as u32;
    let v = |offset: u32| g.value(n + offset);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let (x00, x10) = (lerp(v(0), v(1), f.x), lerp(v(sy), v(sy + 1), f.x));
    let (x01, x11) = (
        lerp(v(sz), v(sz + 1), f.x),
        lerp(v(sz + sy), v(sz + sy + 1), f.x),
    );
    let (y0, y1) = (lerp(x00, x10, f.y), lerp(x01, x11, f.y));
    lerp(y0, y1, f.z) + ((x - c) * g.voxel).length()
}

fn bounds(points: &[Vec3]) -> (Vec3, Vec3) {
    points
        .iter()
        .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        })
}

/// A grid covering `lo..=hi` plus the padding.
fn layout(lo: Vec3, hi: Vec3, o: &SdfOptions) -> Result<Layout> {
    ensure_input!(
        o.voxel.is_finite() && o.voxel > 0.0,
        "the voxel size must be positive, got {}",
        o.voxel
    );
    ensure_input!(
        o.padding.is_finite() && o.padding >= 0.0,
        "the padding must not be negative, got {}",
        o.padding
    );
    let origin = lo - o.padding;
    let dims = ((hi + o.padding - origin) / o.voxel).ceil().max(Vec3::ONE) + 1.0;
    let points = dims.as_dvec3().element_product();
    ensure_input!(
        points <= MAX_POINTS as f64,
        "a grid of {} m voxels over {} m is too large",
        o.voxel,
        hi - lo
    );
    // Squared distances, in voxels, index the table that finishes each value.
    let diagonal = (dims.as_dvec3() - 1.0).length_squared();
    ensure_input!(
        diagonal < MAX_DIAGONAL,
        "a grid of {} m voxels over {} m is too long",
        o.voxel,
        hi - lo
    );
    Ok(Layout {
        dims: dims.as_uvec3(),
        origin,
        voxel: o.voxel,
    })
}

fn point(dims: UVec3, origin: Vec3, voxel: f32, n: u32) -> Vec3 {
    let (x, y, z) = (n % dims.x, n / dims.x % dims.y, n / (dims.x * dims.y));
    origin + UVec3::new(x, y, z).as_vec3() * voxel
}

fn nearest(dims: UVec3, origin: Vec3, voxel: f32, p: Vec3) -> usize {
    let i = ((p - origin) / voxel)
        .round()
        .max(Vec3::ZERO)
        .as_uvec3()
        .min(dims - 1);
    (i.x + dims.x * (i.y + dims.y * i.z)) as usize
}

/// Squared distance, in voxels, from every voxel to the nearest voxel of the other kind: from
/// each occupied voxel to the nearest free one and from each free voxel to the nearest occupied
/// one. Felzenszwalb and Huttenlocher's separable transform, one axis after another, in exact
/// integers; `layout` keeps every value below 2^24. grids.wgsl mirrors it.
pub(crate) fn squared_edt(occupied: &[bool], [nx, ny, nz]: [usize; 3]) -> Vec<u32> {
    let slab = nx * ny;
    // Per kind of site, the x and y passes, one z slab at a time.
    let planes = [true, false].map(|site| {
        let mut d: Vec<u32> = occupied
            .iter()
            .map(|&o| if o == site { 0 } else { FAR })
            .collect();
        d.par_chunks_mut(slab)
            .for_each_init(Line::default, |line, s| slab_passes(line, s, nx, ny));
        d
    });
    // The z pass, column by column, keeping each voxel's distance to the other kind.
    let mut columns = vec![0; occupied.len()];
    columns
        .par_chunks_mut(nz)
        .enumerate()
        .for_each_init(Line::default, |line, (c, column)| {
            column_pass(line, column, c, slab, occupied, &planes);
        });
    let [mut out, _] = planes;
    out.par_chunks_mut(slab).enumerate().for_each(|(z, s)| {
        for (c, v) in s.iter_mut().enumerate() {
            *v = columns[c * nz + z];
        }
    });
    out
}

/// `squared_edt`'s x and y passes over the z slab `s`. Out of line, as is `column_pass`: inlined
/// into rayon's splitting of the slabs, the passes compiled into a slower shape.
#[inline(never)]
fn slab_passes(line: &mut Line, s: &mut [u32], nx: usize, ny: usize) {
    for row in s.chunks_mut(nx) {
        line.transform(row.iter().copied());
        row.copy_from_slice(&line.out);
    }
    for x in 0..nx {
        line.transform((0..ny).map(|y| s[x + nx * y]));
        for y in 0..ny {
            s[x + nx * y] = line.out[y];
        }
    }
}

/// `squared_edt`'s z pass over column `c`, keeping each voxel's distance to the other kind.
#[inline(never)]
fn column_pass(
    line: &mut Line,
    column: &mut [u32],
    c: usize,
    slab: usize,
    occupied: &[bool],
    planes: &[Vec<u32>; 2],
) {
    let nz = column.len();
    for (site, plane) in [true, false].into_iter().zip(planes) {
        line.transform((0..nz).map(|z| plane[c + slab * z]));
        let other_kind = (0..nz).filter(|&z| occupied[c + slab * z] != site);
        for z in other_kind {
            column[z] = line.out[z];
        }
    }
}

/// No site along the line.
pub(crate) const FAR: u32 = u32::MAX;

/// One-dimensional squared distance transform: the lower envelope of parabolas rooted at the
/// values below `FAR`.
#[derive(Default)]
struct Line {
    f: Vec<u32>,
    roots: Vec<usize>,
    /// The first position at which each root's parabola is at most the previous root's, within
    /// `0..=len`: outside the line, where a root starts does not matter.
    starts: Vec<u32>,
    out: Vec<u32>,
}

impl Line {
    fn transform(&mut self, f: impl Iterator<Item = u32>) {
        self.f.clear();
        self.f.extend(f);
        let (f, roots, starts) = (&self.f, &mut self.roots, &mut self.starts);
        roots.clear();
        starts.clear();
        let sites = (0..f.len()).filter(|&q| f[q] != FAR);
        for q in sites {
            let mut s = 0;
            while let Some(&p) = roots.last() {
                s = meet(p, f[p], q, f[q]).clamp(0, f.len() as i32) as u32;
                if s > *starts.last().expect("one start per root") {
                    break;
                }
                roots.pop();
                starts.pop();
                s = 0;
            }
            roots.push(q);
            starts.push(s);
        }
        self.out.clear();
        if roots.is_empty() {
            self.out.resize(f.len(), FAR);
            return;
        }
        let mut k = 0;
        for x in 0..f.len() {
            while k + 1 < roots.len() && starts[k + 1] <= x as u32 {
                k += 1;
            }
            let d = x.abs_diff(roots[k]) as u32;
            self.out.push(d * d + f[roots[k]]);
        }
    }
}

/// The first position at which the parabola rooted at `q` is at most the one rooted at `p < q`:
/// where they meet, rounded up.
fn meet(p: usize, fp: u32, q: usize, fq: u32) -> i32 {
    let num = (fq + (q * q) as u32) as i32 - (fp + (p * p) as u32) as i32;
    let den = 2 * (q - p) as i32;
    num / den + i32::from(num % den > 0)
}

/// Log-odds in twentieths: a hit adds 0.85, a miss -0.4, clamped to [-2, 3.5], as in OctoMap.
pub(crate) const HIT: i8 = 17;
pub(crate) const MISS: i8 = -8;
pub(crate) const CLAMP: (i8, i8) = (-40, 70);
/// The log-odds of a voxel no image has updated.
pub(crate) const UNOBSERVED: i8 = i8::MIN;
/// The log-odds of a voxel no image has observed, which only a robot mask has hidden.
pub(crate) const SHADOWED: i8 = i8::MIN + 1;
/// The log-odds of a voxel no image has observed, which a surface other than the robot has hidden:
/// never shadow again.
pub(crate) const OCCLUDED: i8 = i8::MIN + 2;
/// The log-odds of a voxel a part of a map leaves out, outside its region
/// ([`OccupancyMap::part`]): no solid in its grids, and not counted.
const APART: i8 = i8::MIN + 3;

/// Every point from `from` to `to` along each axis, both included.
fn points((from, to): (UVec3, UVec3)) -> impl Iterator<Item = UVec3> {
    (from.z..=to.z).flat_map(move |z| {
        (from.y..=to.y).flat_map(move |y| (from.x..=to.x).map(move |x| UVec3::new(x, y, z)))
    })
}

/// Whether log-odds are those of a voxel an image has observed.
fn observed(l: i8) -> bool {
    !matches!(l, UNOBSERVED | SHADOWED | OCCLUDED | APART)
}

/// Occupancy fused from depth images over time, on a fixed grid in the world frame: per voxel, the
/// log-odds that it is occupied, updated as OctoMap updates them (a hit adds 0.85, a miss -0.4,
/// clamped to [-2, 3.5]). Readings that come and go settle: something seen once and then seen
/// through three times is free again. Build distance grids from it with [`OccupancyMap::grid`].
///
/// What a camera cannot tell, a caller can: [`free`](OccupancyMap::free) marks space known to
/// be empty, as where an object was before it was carried away, and
/// [`clear`](OccupancyMap::clear) space that is no longer known, as where something was moved.
/// [`part`](OccupancyMap::part) is the map inside one region, for a grid of one object as the
/// cameras saw into it.
#[derive(Clone, Debug)]
pub struct OccupancyMap {
    grid: Layout,
    log_odds: Vec<i8>,
    /// Whether this is a part of a map, some of its voxels left out: it takes no images.
    part: bool,
}

/// How many of a map's voxels are occupied, free and never observed
/// ([`OccupancyMap::counts`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct MapCounts {
    /// Voxels more likely occupied than not.
    pub occupied: usize,
    /// Voxels observed and not more likely occupied than not.
    pub free: usize,
    /// Voxels no image has observed: outside every image, along pixels without a reading, or
    /// hidden behind a surface or the robot.
    pub unobserved: usize,
}

impl OccupancyMap {
    /// An unobserved map with `voxel` spacing whose points cover `lo..=hi`.
    pub fn try_new(lo: Vec3, hi: Vec3, voxel: f32) -> Result<Self> {
        ensure_input!(
            lo.is_finite() && hi.is_finite() && lo.cmplt(hi).all(),
            "a map needs lo < hi, got {lo} and {hi}"
        );
        let grid = layout(
            lo,
            hi,
            &SdfOptions {
                voxel,
                padding: 0.0,
            },
        )?;
        Ok(Self {
            grid,
            log_odds: vec![UNOBSERVED; grid.points()],
            part: false,
        })
    }

    /// Fuses one depth image on `device`. Each voxel is projected into it: voxels in front of a
    /// reading are misses, voxels at it (within half a voxel) and those holding a reading's point
    /// are hits, and voxels behind readings, outside the image or along pixels without a reading
    /// are not updated. With a [`RobotMask`], pixels that see the robot only clear the space in
    /// front of it, and the space the robot fills is a miss wherever it is.
    pub fn integrate(
        &mut self,
        device: &Device,
        image: &DepthImage,
        robot: Option<&RobotMask>,
    ) -> Result<()> {
        image.check_with(robot)?;
        ensure_input!(
            !self.part,
            "a part of a map takes no images: fuse them into the map and take its part again"
        );
        device.integrate(&self.grid, &mut self.log_odds, &image.rigid(), robot)
    }

    /// The probability that the voxel holding `p` is occupied; `None` if it was never observed or
    /// lies outside the map, or outside the region of a part of one.
    #[must_use]
    pub fn occupancy(&self, p: Vec3) -> Option<f32> {
        let l = self.log_odds[self.grid.nearest(p)];
        (self.grid.contains(p) && observed(l)).then(|| 1.0 / (1.0 + (-f32::from(l) / 20.0).exp()))
    }

    /// How many voxels are occupied, free and never observed. Of a part, those inside its
    /// region: a region with free voxels was seen into, and one that is all occupied and never
    /// observed was not.
    #[must_use]
    pub fn counts(&self) -> MapCounts {
        let mut counts = MapCounts::default();
        for &l in self.log_odds.iter().filter(|&&l| l != APART) {
            match (observed(l), l > 0) {
                (true, true) => counts.occupied += 1,
                (true, false) => counts.free += 1,
                (false, _) => counts.unobserved += 1,
            }
        }
        counts
    }

    /// The part of the map inside `region`, a shape in the map's frame: a map of its own that
    /// reaches `padding` past the region's bounds, as a grid must reach past the solid it holds
    /// ([`SdfOptions::padding`]), with every voxel that does not reach into the region left out.
    /// A voxel reaches into it where its middle is within half a voxel's diagonal of the region:
    /// a reading on the region's own face lies in such a voxel, on either side of the face. A
    /// left-out voxel is no solid in the part's grid whatever is asked of never-observed space,
    /// and is not counted. So `map.part(&object, padding)?.grid(&device, Occlusion::Occupied)` is
    /// the object as the cameras saw into it, solid where a reading landed or nothing saw past,
    /// with nothing around it. Where the region reaches past the map, it was never observed; past
    /// a part it is taken from, it is left out. A part takes no images.
    pub fn part(&self, region: &Obstacle, padding: f32) -> Result<Self> {
        region.check().map_err(|e| input!("the region {e}"))?;
        ensure_input!(
            padding.is_finite() && padding >= 0.0,
            "the padding must not be negative, got {padding}"
        );
        let (lo, hi) = region.bounds();
        let voxel = self.grid.voxel;
        // The part's points among the map's, counted from the map's first: possibly before it.
        let first = ((lo - padding - self.grid.origin) / voxel).floor();
        let grid = layout(
            self.grid.origin + first * voxel,
            hi + padding,
            &SdfOptions {
                voxel,
                padding: 0.0,
            },
        )?;
        let (first, dims) = (first.as_ivec3(), self.grid.dims.as_ivec3());
        // A voxel's cube reaches this far from its middle.
        let reach = 0.5 * 3f32.sqrt() * voxel;
        let mut log_odds = vec![APART; grid.points()];
        for at in (grid.among(lo - reach, hi + reach).into_iter()).flat_map(points) {
            // The same point of the map, at the map's own coordinates, whatever the padding.
            let of_map = first + at.as_ivec3();
            let off = region.distance(self.grid.origin + of_map.as_vec3() * voxel);
            if off.value > reach {
                continue;
            }
            let reached = of_map.cmpge(IVec3::ZERO).all() && of_map.cmplt(dims).all();
            log_odds[grid.index(at)] = match reached {
                true => self.log_odds[self.grid.index(of_map.as_uvec3())],
                // Past a map nothing was observed; past a part of one, the voxel was left out.
                false if self.part => APART,
                false => UNOBSERVED,
            };
        }
        Ok(Self {
            grid,
            log_odds,
            part: true,
        })
    }

    /// Marks the voxels within `margin` of `shape` (in the map's frame) free, as seen through
    /// once, or as they were where cameras saw through them more: space the caller knows is
    /// empty though no camera has looked, as where an object stood before the robot carried it
    /// away, or where the world already holds the shape as an obstacle of its own and the map's
    /// grid should not hold it again. One reading there afterwards makes a voxel the shape held
    /// occupied again.
    pub fn free(&mut self, shape: &Obstacle, margin: f32) -> Result<()> {
        self.mark(shape, margin, |l| match observed(l) {
            true => l.min(MISS),
            false => MISS,
        })
    }

    /// Marks the voxels within `margin` of `shape` (in the map's frame) never observed: what
    /// was seen there no longer holds, as where a drawer stood before it was pulled and where it
    /// stands now. A grid then takes them as it is told to take never-observed space.
    pub fn clear(&mut self, shape: &Obstacle, margin: f32) -> Result<()> {
        self.mark(shape, margin, |_| UNOBSERVED)
    }

    /// Sets the log-odds of each voxel within `margin` of `shape`, among those of the shape's
    /// bounds, to what `to` makes of it. A part's left-out voxels stay left out.
    fn mark(&mut self, shape: &Obstacle, margin: f32, to: impl Fn(i8) -> i8) -> Result<()> {
        shape.check().map_err(|e| input!("the shape {e}"))?;
        ensure_input!(
            margin.is_finite() && margin >= 0.0,
            "the margin must not be negative, got {margin}"
        );
        let (lo, hi) = shape.bounds();
        for at in (self.grid.among(lo - margin, hi + margin).into_iter()).flat_map(points) {
            let n = self.grid.index(at);
            let l = self.log_odds[n];
            if l != APART && shape.distance(self.grid.point(n)).value <= margin {
                self.log_odds[n] = to(l);
            }
        }
        Ok(())
    }

    /// The distance grid of the map: voxels more likely occupied than not are solid, and voxels
    /// never observed are solid or free as `unknown` says. Errors if nothing is solid.
    pub fn grid(&self, device: &Device, unknown: Occlusion) -> Result<SdfGrid> {
        unknown.check()?;
        let occupied: Vec<bool> = self
            .log_odds
            .par_iter()
            .map(|&l| match l {
                UNOBSERVED | OCCLUDED => matches!(
                    unknown,
                    Occlusion::Occupied | Occlusion::OccupiedExceptRobotShadow
                ),
                SHADOWED => unknown == Occlusion::Occupied,
                APART => false,
                _ => l > 0,
            })
            .collect();
        ensure_input!(
            occupied.iter().any(|&o| o),
            "the map holds nothing occupied"
        );
        let solid = SdfGrid::from_occupancy(device, &self.grid, &occupied)?;
        let Occlusion::Band(band) = unknown else {
            return Ok(solid);
        };
        // A free point stores at most its distance to the nearest occupied cube lowered by half a
        // voxel diagonal, so one within the band of a cube stores at most `reach`: the band comes
        // out no narrower.
        let reach = band - 0.5 * 3f32.sqrt() * self.grid.voxel;
        let banded: Vec<bool> = device.install(|| {
            self.log_odds
                .par_iter()
                .zip(&occupied)
                .zip(&solid.values)
                .map(|((&l, &o), &v)| {
                    o || (l == UNOBSERVED || l == OCCLUDED) && f16::from_bits(v).to_f32() <= reach
                })
                .collect()
        });
        SdfGrid::from_occupancy(device, &self.grid, &banded)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use glam::Quat;

    use super::*;
    use crate::{CollisionModel, RobotOptions};

    /// The Panda as the examples hold it.
    fn panda() -> Robot {
        let fingers = [
            ("panda_finger_joint1".to_string(), 0.04),
            ("panda_finger_joint2".to_string(), 0.04),
        ];
        let options = RobotOptions {
            lock_joints: HashMap::from(fingers),
            default_q: Some(vec![0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]),
            collision_model: Some(
                CollisionModel::load("assets/franka/panda_collision.json").unwrap(),
            ),
            ..Default::default()
        };
        Robot::load("assets/franka/franka_panda.urdf", &options).unwrap()
    }

    /// A 640x480 depth image of a table at z = 0 with a ball on it, from 1 m above the robot.
    fn table_image() -> (Vec<f32>, Intrinsics, Pose) {
        let camera = Pose::new(
            Vec3::new(0.0, 0.0, 1.0),
            Quat::from_rotation_x(std::f32::consts::PI),
        );
        let k = Intrinsics {
            fx: 525.0,
            fy: 525.0,
            cx: 319.5,
            cy: 239.5,
        };
        let (c, r) = (Vec3::new(0.05, 0.1, 0.1), 0.1);
        let depth = (0..640 * 480)
            .map(|i| {
                let ray = camera.rotation
                    * Vec3::new(
                        ((i % 640) as f32 - k.cx) / k.fx,
                        ((i / 640) as f32 - k.cy) / k.fy,
                        1.0,
                    );
                let (b, cc) = (
                    ray.dot(camera.position - c),
                    (camera.position - c).length_squared() - r * r,
                );
                let disc = b * b - ray.length_squared() * cc;
                let ball = if disc >= 0.0 {
                    (-b - disc.sqrt()) / ray.length_squared()
                } else {
                    f32::INFINITY
                };
                ball.min(-camera.position.z / ray.z)
            })
            .collect();
        (depth, k, camera)
    }

    /// The least value on a grid's faces is the least of its points with a coordinate at either
    /// end: every face, and no point inside. Each face of an uneven grid holds the least in turn.
    #[test]
    fn the_least_on_a_grids_faces_is_found_on_every_face() {
        let dims = [3u32, 4, 5];
        let index = |x: u32, y: u32, z: u32| (x + dims[0] * (y + dims[1] * z)) as usize;
        let faces = [
            (0, 1, 2),
            (2, 2, 1),
            (1, 0, 3),
            (1, 3, 2),
            (1, 2, 0),
            (1, 1, 4),
        ];
        for (x, y, z) in faces {
            let mut distances = vec![0.5; 60];
            // Lower inside, where it must not count, and lowest at one point of one face.
            distances[index(1, 1, 2)] = -0.75;
            distances[index(x, y, z)] = -0.25;
            let grid = SdfGrid::try_new(dims, 0.1, Vec3::ZERO, &distances).unwrap();
            assert_eq!(
                grid.least_on_faces, -0.25,
                "the face through ({x}, {y}, {z})"
            );
        }
        let even = SdfGrid::try_new(dims, 0.1, Vec3::ZERO, &[0.5; 60]).unwrap();
        assert_eq!(even.least_on_faces, 0.5);
    }

    /// The lookup that takes no gradient reads what the one that takes it reads, to the bit:
    /// inside the box, on its faces, edges and corners, and outside it.
    #[test]
    fn a_lookup_without_its_gradient_reads_the_same_distance() {
        let dims = [4u32, 3, 5];
        let mut rng = crate::rng::Rng::new(11);
        let distances: Vec<f32> = (0..60).map(|_| rng.range(-0.2, 0.4)).collect();
        let grid = SdfGrid::try_new(dims, 0.07, Vec3::new(0.1, -0.2, 0.3), &distances).unwrap();
        let (lo, hi) = (grid.origin, grid.origin + 2.0 * grid.half_extent());
        let mut outside = 0;
        for k in 0..4000 {
            let mut p = Vec3::new(
                rng.range(lo.x - 0.05, hi.x + 0.05),
                rng.range(lo.y - 0.05, hi.y + 0.05),
                rng.range(lo.z - 0.05, hi.z + 0.05),
            );
            // Every eighth point on a face, an edge or a corner of the box, at either end.
            if k % 8 == 0 {
                let end = if k & 32 != 0 { hi } else { lo };
                p = Vec3::select(glam::BVec3::new(k & 8 != 0, k & 16 != 0, true), end, p);
            }
            outside += usize::from(p.cmplt(lo).any() || p.cmpgt(hi).any());
            let (with, without) = (grid_distance(&grid, p).0, grid_value(&grid, p));
            assert_eq!(with.to_bits(), without.to_bits(), "at {p}");
        }
        assert!(
            outside > 1500 && outside < 3500,
            "{outside} of 4000 outside"
        );
    }

    /// The points a mask marks ball by ball are those of asking every ball about every point: of
    /// spheres inside the grid, across its faces and outside it, and of a held shape of every
    /// kind, on a grid whose points do not lie on whole centimeters.
    #[test]
    fn the_points_a_mask_marks_inside_are_every_balls() {
        let mut rng = crate::rng::Rng::new(5);
        let grid = Layout {
            dims: UVec3::new(41, 33, 27),
            origin: Vec3::new(-0.2013, 0.0507, 0.3001),
            voxel: 0.0125,
        };
        let top = grid.point(grid.points() - 1);
        let spheres: Vec<Vec4> = (0..60)
            .map(|_| {
                let mut at = |lo: f32, hi: f32| rng.range(lo - 0.08, hi + 0.08);
                Vec3::new(
                    at(grid.origin.x, top.x),
                    at(grid.origin.y, top.y),
                    at(grid.origin.z, top.z),
                )
                .extend(rng.range(0.004, 0.09))
            })
            .collect();
        // One held shape of each kind, turned, each reaching its farthest from its middle
        // inside the grid: a ball too small for one would leave such points unmarked.
        let rotation = Quat::from_rotation_y(0.7) * Quat::from_rotation_z(0.4);
        let held = vec![
            Obstacle::Cuboid {
                center: Vec3::new(-0.1, 0.15, 0.4),
                half_extents: Vec3::new(0.02, 0.06, 0.03),
                rotation,
            },
            Obstacle::sphere(Vec3::new(0.2, 0.15, 0.4), 0.03),
            Obstacle::Cylinder {
                center: Vec3::new(-0.1, 0.35, 0.5),
                rotation,
                radius: 0.04,
                half_height: 0.05,
            },
            Obstacle::Capsule {
                center: Vec3::new(0.2, 0.35, 0.5),
                rotation,
                radius: 0.025,
                half_length: 0.06,
            },
        ];
        let mask = RobotMask {
            on_robot: vec![],
            spheres,
            held,
            padding: 0.015,
        };
        let inside = mask.inside(&grid);
        assert_eq!(inside.len(), grid.points().div_ceil(32));
        let (mut in_spheres, mut in_held) = (0, [0; 4]);
        for n in 0..grid.points() {
            let (p, flag) = (grid.point(n), set(&inside, n));
            let sphere =
                (mask.spheres.iter()).any(|s| (p - s.truncate()).length_squared() < s.w * s.w);
            let object = (mask.held.iter()).position(|o| o.distance(p).value < mask.padding);
            assert_eq!(flag, sphere || object.is_some(), "point {n} at {p}");
            in_spheres += usize::from(sphere);
            if let Some(object) = object.filter(|_| !sphere) {
                in_held[object] += 1;
            }
        }
        assert!(
            in_held.iter().all(|&points| points > 100),
            "{in_held:?} points in each held shape alone"
        );
        let in_held: usize = in_held.iter().sum();
        let outside = grid.points() - in_spheres - in_held;
        assert!(
            in_spheres > 2_000 && outside > 2_000,
            "{in_spheres} points in spheres, {in_held} in held shapes alone, {outside} in neither"
        );
        // Spheres cross every face of the grid, and one lies wholly outside.
        let crossing =
            |axis: usize, end: f32| (mask.spheres.iter()).any(|s| (s[axis] - end).abs() < s.w);
        for axis in 0..3 {
            assert!(crossing(axis, grid.origin[axis]) && crossing(axis, top[axis]));
        }
        let apart = |s: &Vec4| (0..3).any(|a| s[a] + s.w < grid.origin[a] || s[a] - s.w > top[a]);
        assert!(mask.spheres.iter().any(apart), "a sphere outside the grid");
    }

    /// Classification projects points in floating point, so a point within rounding of a pixel's
    /// edge or of a depth may land either way on different devices; in a dense image, a handful.
    /// Maps show it directly: grids spread each such point over its neighbours' distances.
    #[test]
    fn every_device_classifies_depth_images_alike() {
        let robot = panda();
        let gpu = match Device::gpu(&robot) {
            Ok(gpu) => gpu,
            Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => {
                return eprintln!("skipping GPU: {e}");
            }
            Err(e) => panic!("no GPU: {e}"),
        };
        let cpu = Device::cpu(&robot).unwrap();
        let (table, intrinsics, camera) = table_image();
        // The arm's spheres before the table, as a camera that sees the arm reads them.
        let spheres = robot.spheres(robot.default_q());
        let depth: Vec<f32> = (0..table.len())
            .map(|i| {
                let (u, v) = ((i % 640) as f32, (i / 640) as f32);
                let ray = camera.rotation
                    * Vec3::new(
                        (u - intrinsics.cx) / intrinsics.fx,
                        (v - intrinsics.cy) / intrinsics.fy,
                        1.0,
                    );
                let hits = spheres.iter().filter_map(|&[x, y, z, r]| {
                    let from = camera.position - Vec3::new(x, y, z);
                    let (b, c) = (ray.dot(from), from.length_squared() - r * r);
                    let disc = b * b - ray.length_squared() * c;
                    (disc >= 0.0).then(|| (-b - disc.sqrt()) / ray.length_squared())
                });
                hits.fold(table[i], f32::min)
            })
            .collect();
        let image = DepthImage {
            depth: &depth,
            width: 640,
            intrinsics,
            camera,
        };
        let mask = image
            .robot_mask(&robot, robot.default_q(), 0.02, &[])
            .unwrap();
        assert!(
            mask.on_robot.iter().filter(|&&on| on).count() > 10_000,
            "the camera should see the arm"
        );
        let table = DepthImage {
            depth: &table,
            ..image
        };
        let grid = layout(
            Vec3::new(-0.6, -0.6, -0.1),
            Vec3::new(0.6, 0.6, 1.0),
            &SdfOptions {
                voxel: 0.01,
                padding: 0.0,
            },
        )
        .unwrap();
        // The image with the arm masked out, then the table without the arm before it: the
        // log-odds after each.
        let fused = |device: &Device| {
            let mut log_odds = vec![UNOBSERVED; grid.points()];
            device
                .integrate(&grid, &mut log_odds, &image, Some(&mask))
                .unwrap();
            let masked = log_odds.clone();
            device
                .integrate(&grid, &mut log_odds, &table, None)
                .unwrap();
            [masked, log_odds]
        };
        let (cpu_fused, gpu_fused) = (fused(&cpu), fused(&gpu));
        let [masked, both] = &cpu_fused;
        let count = |log_odds: &[i8], l: i8| log_odds.iter().filter(|&&x| x == l).count();
        assert!(
            count(masked, SHADOWED) > 1_000,
            "the arm should hide space no image has seen"
        );
        assert!(
            count(masked, OCCLUDED) > 1_000,
            "the table should hide space below it"
        );
        assert!(
            count(both, HIT + HIT) > 2_000,
            "the table should be hit twice"
        );
        assert!(
            count(both, HIT) > 2_000,
            "the arm should hide part of the table the first time"
        );
        assert!(
            count(both, MISS + MISS) > 100_000,
            "the space in front of the table should be cleared twice"
        );
        assert!(
            count(both, OCCLUDED) > count(masked, OCCLUDED),
            "the table should hide space below it that the arm hid"
        );
        for (a, b) in cpu_fused.iter().zip(&gpu_fused) {
            let differ = a.iter().zip(b).filter(|(a, b)| a != b).count();
            eprintln!("{}: {differ} of {} log-odds differ", gpu.name(), a.len());
            assert!(
                differ * 2000 <= a.len(),
                "{differ} of {} log-odds differ",
                a.len()
            );
        }
    }
}
