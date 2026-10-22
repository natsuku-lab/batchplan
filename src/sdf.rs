//! Signed distance grids: obstacles of any shape, built from meshes, point clouds or depth images.
//!
//! A grid holds distances at regular points as half floats, and both devices interpolate them the
//! same way (`grid_distance` here and in kernels.wgsl). The builders make grids conservative. Each
//! point holds at most the true signed distance of the geometry (for point clouds and depth
//! images, of the solid voxels as cubes), lowered by half a voxel diagonal: the most trilinear
//! interpolation can overestimate by. A built grid therefore never reads farther from an obstacle
//! than its geometry, and walls thinner than a voxel still block. The price is that obstacles grow,
//! by up to a voxel diagonal for meshes and about three voxels for points and depth images.
//!
//! Point clouds, depth images and occupancy maps become grids through an exact distance transform
//! of their solid voxels, which runs on the [`Device`] the builder is given.

use std::num::FpCategory;

use glam::{UVec3, Vec3, Vec4};
use half::f16;
use parry3d::shape::{TriMesh as ParryMesh, TriMeshFlags};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::description::TriMesh;
use crate::device::Device;
use crate::error::{Error, Result, ensure_input, input};
use crate::robot::Robot;
use crate::types::Pose;

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
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// What [`SdfGrid::from_depth`] assumes about space hidden behind observed surfaces, and
/// [`OccupancyMap::grid`] about space no image has observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Occlusion {
    /// Only observed surfaces are obstacles.
    Free,
    /// Everything behind an observed surface, as seen from the camera, is solid, the robot's
    /// shadow included.
    Occupied,
    /// As `Occupied`, except the space only the robot hides (with a [`RobotMask`]), which is
    /// free. No camera sees past the arm, so its shadow, solid, walls in the space it reaches
    /// through next. What the shadow holds is unknown: this trades that for being able to move.
    OccupiedExceptRobotShadow,
}

/// One depth image: z distances in meters, row by row, `width` pixels per row, with zero, negative
/// or non-finite values for pixels without a reading. `camera` places the camera (x right, y down,
/// z forward, as in OpenCV) in the frame grids and maps are built in.
#[derive(Clone, Copy, Debug)]
pub struct DepthImage<'a> {
    pub depth: &'a [f32],
    pub width: usize,
    pub intrinsics: Intrinsics,
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
        Ok(())
    }

    fn height(&self) -> usize {
        self.depth.len() / self.width
    }

    fn reading(&self, i: usize) -> Option<f32> {
        Some(self.depth[i]).filter(|z| z.is_finite() && *z > 0.0)
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

    /// Pixel `i`'s reading as a world point, unless it reads the robot: at or beyond the robot's
    /// depth there, within `voxel`.
    fn point(&self, i: usize, robot: Option<&RobotMask>, voxel: f32) -> Option<Vec3> {
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        let z = self
            .reading(i)
            .filter(|&z| z < robot.map_or(f32::INFINITY, |r| r.depth[i]) - voxel)?;
        let (u, v) = ((i % self.width) as f32, (i / self.width) as f32);
        Some(
            self.camera.rotation * Vec3::new((u - cx) / fx * z, (v - cy) / fy * z, z)
                + self.camera.position,
        )
    }

    /// What the image says about the point `p` of a grid with `voxel` spacing, with the space the
    /// robot fills free. `classify` in grids.wgsl mirrors it.
    fn classify(&self, robot: Option<&RobotMask>, p: Vec3, voxel: f32) -> Seen {
        match self.sees(robot, p, voxel) {
            Seen::Hidden | Seen::Shadow | Seen::Unseen if robot.is_some_and(|r| r.contains(p)) => {
                Seen::Free
            }
            seen => seen,
        }
    }

    /// What the image shows of the point `p`. `sees` in grids.wgsl mirrors it.
    fn sees(&self, robot: Option<&RobotMask>, p: Vec3, voxel: f32) -> Seen {
        let half = 0.5 * voxel;
        let Some((i, z_voxel)) = self.project(p) else {
            return Seen::Unseen;
        };
        let Some(z) = self.reading(i) else {
            return Seen::Unseen;
        };
        let robot_at = robot.map_or(f32::INFINITY, |r| r.depth[i]);
        if z >= robot_at - voxel {
            // The pixel sees the robot.
            if z_voxel < robot_at - half {
                Seen::Free
            } else {
                Seen::Shadow
            }
        } else if z_voxel < z - half {
            Seen::Free
        } else if z_voxel <= z + half {
            Seen::Surface
        } else {
            Seen::Hidden
        }
    }

    /// Why the image, with this robot mask, cannot be used, if it cannot.
    fn check_with(&self, robot: Option<&RobotMask>) -> Result<()> {
        self.check()?;
        if let Some(r) = robot {
            let n = r.depth.len();
            ensure_input!(
                n == self.depth.len(),
                "a robot mask of {n} pixels for an image of {}",
                self.depth.len()
            );
        }
        Ok(())
    }

    /// Where `robot` is in this image at configuration `q`, its collision spheres grown by
    /// `padding`. Pass it to [`SdfGrid::from_depth`] or [`OccupancyMap::integrate`]: readings of
    /// the robot are then not obstacles, and the space it fills is free, so it is not an obstacle
    /// to itself even where it hides space that counts as solid. Grids from depth images grow
    /// toward collision by about three voxels; pad by more, or the robot's surface reads as
    /// touching the space it hides.
    pub fn robot_mask(&self, robot: &Robot, q: &[f32], padding: f32) -> Result<RobotMask> {
        self.check()?;
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
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        let (width, height) = (self.width, self.height());
        let fk = robot.fk(q);
        let grown: Vec<Vec4> = robot
            .spheres
            .iter()
            .map(|s| (fk.rot[s.link] * s.center + fk.pos[s.link]).extend(s.radius + padding))
            .collect();
        let to_camera = self.camera.rotation.inverse();
        // Each sphere in the camera's frame, and the pixels it can cover: the projections of its
        // bounding box's sides at its near and far depths. The whole image when the camera is
        // inside or next to it.
        let spheres: Vec<_> = grown
            .iter()
            .filter_map(|sphere| {
                let (center, r) = (
                    to_camera * (sphere.truncate() - self.camera.position),
                    sphere.w,
                );
                let (near, far) = (center.z - r, center.z + r);
                if far <= 0.0 {
                    return None;
                }
                let span = |c: f32, f: f32, k: f32, n: usize| {
                    if near <= 1e-3 {
                        return 0..n;
                    }
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
                Some((
                    center,
                    r,
                    span(center.x, fx, cx, width),
                    span(center.y, fy, cy, height),
                ))
            })
            .collect();
        let mut out = vec![f32::INFINITY; self.depth.len()];
        out.par_chunks_mut(width).enumerate().for_each(|(v, row)| {
            for (center, r, us, vs) in &spheres {
                if !vs.contains(&v) {
                    continue;
                }
                for u in us.clone() {
                    // Points t * ray along the pixel; their depth is t.
                    let ray = Vec3::new((u as f32 - cx) / fx, (v as f32 - cy) / fy, 1.0);
                    let (a, b, c) = (
                        ray.length_squared(),
                        ray.dot(*center),
                        center.length_squared() - r * r,
                    );
                    let disc = b * b - a * c;
                    if disc < 0.0 {
                        continue;
                    }
                    let t = ((b - disc.sqrt()) / a).max(0.0);
                    row[u] = row[u].min(t);
                }
            }
        });
        Ok(RobotMask {
            depth: out,
            spheres: grown,
        })
    }
}

/// Where a robot is in one depth image ([`DepthImage::robot_mask`]).
#[derive(Clone, Debug)]
pub struct RobotMask {
    /// Each pixel's depth to the robot's grown spheres; infinite where the ray misses them.
    pub depth: Vec<f32>,
    /// The grown spheres in the image's frame: centre and radius.
    pub(crate) spheres: Vec<Vec4>,
}

impl RobotMask {
    /// Whether `p` is inside the robot (`inside_robot` in grids.wgsl).
    fn contains(&self, p: Vec3) -> bool {
        self.spheres
            .iter()
            .any(|s| (p - s.truncate()).length_squared() < s.w * s.w)
    }
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
    /// Behind a reading.
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

    /// Whether `p` lies within half a voxel of the points.
    fn contains(&self, p: Vec3) -> bool {
        let half = 0.5 * self.voxel;
        p.cmpge(self.origin - half).all()
            && p.cmplt(self.origin + (self.dims - 1).as_vec3() * self.voxel + half)
                .all()
    }
}

/// Where a grid's occupancy comes from.
pub(crate) enum Occupancy<'a> {
    /// One flag per point.
    Given(&'a [bool]),
    /// A depth image, as [`SdfGrid::from_depth`] reads it.
    Seen {
        image: &'a DepthImage<'a>,
        robot: Option<&'a RobotMask>,
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
    let mut occupied: Vec<bool> = (0..grid.points())
        .into_par_iter()
        .map(|n| match image.classify(robot, grid.point(n), grid.voxel) {
            Seen::Surface => true,
            Seen::Hidden => behind != Occlusion::Free,
            Seen::Shadow => behind == Occlusion::Occupied,
            Seen::Free | Seen::Unseen => false,
        })
        .collect();
    // Dense images resolve surfaces finer than voxels project; their points are solid too.
    for n in hits(grid, image, robot, |_| true) {
        occupied[n] = true;
    }
    occupied
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
                .point(i, robot, grid.voxel)
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
    log_odds
        .par_iter_mut()
        .zip(hit)
        .enumerate()
        .for_each(|(n, (l, hit))| {
            let step = match image.classify(robot, grid.point(n), grid.voxel) {
                _ if hit => HIT,
                Seen::Surface => HIT,
                Seen::Free => MISS,
                // Only the robot hides it: noted, until an image observes it.
                Seen::Shadow => {
                    if *l == UNOBSERVED {
                        *l = SHADOWED;
                    }
                    return;
                }
                Seen::Hidden | Seen::Unseen => return,
            };
            *l = (if *l == UNOBSERVED || *l == SHADOWED {
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
}

/// A grid as serialized, checked like [`SdfGrid::new`] before it becomes one.
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
        let grid = SdfGrid {
            dims: p.dims,
            voxel: p.voxel,
            origin: p.origin,
            values: p.values,
        };
        grid.check()?;
        Ok(grid)
    }
}

impl SdfGrid {
    /// A grid holding `distances` at `dims` points (`dims[0] * dims[1] * dims[2]` values, x
    /// fastest), rounded to the nearest half float: no conservative offset is applied.
    pub fn new(dims: [u32; 3], voxel: f32, origin: Vec3, distances: &[f32]) -> Result<Self> {
        let values = distances
            .iter()
            .map(|&d| flush(f16::from_f32(d)).to_bits())
            .collect();
        let grid = Self {
            dims,
            voxel,
            origin,
            values,
        };
        grid.check()?;
        Ok(grid)
    }

    /// Why this grid cannot be sampled, if it cannot: the rules of [`SdfGrid::new`].
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

    /// The distance field of a closed triangle mesh, in the mesh's frame. Triangles may face in
    /// or out, consistently; meshes that are open or not manifold have no inside and are refused.
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
        let mut mesh = TriMesh {
            vertices: vertices.to_vec(),
            triangles: triangles.to_vec(),
        };
        mesh.orient_outward();
        let flags = TriMeshFlags::HALF_EDGE_TOPOLOGY
            | TriMeshFlags::ORIENTED
            | TriMeshFlags::MERGE_DUPLICATE_VERTICES
            | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES;
        // `with_flags` drops topology errors, so the flags are set separately.
        let mut solid =
            ParryMesh::new(mesh.vertices.clone(), mesh.triangles).map_err(|e| input!("{e}"))?;
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
        let (lo, hi) = bounds(&mesh.vertices);
        let Layout { dims, origin, .. } = layout(lo, hi, o)?;
        let distances = (0..dims.element_product())
            .into_par_iter()
            .map(|n| {
                let p = point(dims, origin, o.voxel, n);
                let proj = solid.project_local_point(p, false);
                let d = (p - proj.point).length();
                if proj.is_inside { -d } else { d }
            })
            .collect();
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
    /// readings on or behind the robot are not obstacles: space in front of it and the space it
    /// fills are free, and space behind it is what `behind` says.
    pub fn from_depth(
        device: &Device,
        image: &DepthImage,
        robot: Option<&RobotMask>,
        behind: Occlusion,
        o: &SdfOptions,
    ) -> Result<Self> {
        image.check_with(robot)?;
        let (lo, hi) = (0..image.depth.len())
            .into_par_iter()
            .filter_map(|i| image.point(i, robot, o.voxel))
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

    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    pub fn voxel(&self) -> f32 {
        self.voxel
    }

    /// Lower and upper corners of the grid's box, in its frame.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (
            self.origin,
            self.origin + (UVec3::from(self.dims) - 1).as_vec3() * self.voxel,
        )
    }

    /// Signed distance at `p` (in the grid's frame) and its gradient.
    pub fn distance(&self, p: Vec3) -> (f32, Vec3) {
        grid_distance(self, p)
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
        let grid = Self {
            dims: dims.into(),
            voxel,
            origin,
            values,
        };
        grid.check()?;
        Ok(grid)
    }

    /// Signed distances to the occupied voxels as solid cubes, from their centers' distances.
    /// A free point is at least `d - sqrt(3) / 2` voxels from the nearest cube when its center is
    /// `d` away; an occupied point is at most `d - 1 / 2` deep when the nearest free center is `d`
    /// away. Both bounds keep each value at or below the true distance.
    fn from_occupancy(device: &Device, grid: &Layout, occupied: &[bool]) -> Result<Self> {
        Self::from_values(grid, device.grid_values(grid, Occupancy::Given(occupied))?)
    }

    fn from_values(grid: &Layout, values: Vec<u16>) -> Result<Self> {
        let grid = Self {
            dims: grid.dims.into(),
            voxel: grid.voxel,
            origin: grid.origin,
            values,
        };
        grid.check()?;
        Ok(grid)
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
#[inline]
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
            .for_each_init(Line::default, |line, s| {
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
            });
        d
    });
    // The z pass, column by column, keeping each voxel's distance to the other kind.
    let mut columns = vec![0; occupied.len()];
    columns
        .par_chunks_mut(nz)
        .enumerate()
        .for_each_init(Line::default, |line, (c, column)| {
            for (site, plane) in [true, false].into_iter().zip(&planes) {
                line.transform((0..nz).map(|z| plane[c + slab * z]));
                let other_kind = (0..nz).filter(|&z| occupied[c + slab * z] != site);
                for z in other_kind {
                    column[z] = line.out[z];
                }
            }
        });
    let [mut out, _] = planes;
    out.par_chunks_mut(slab).enumerate().for_each(|(z, s)| {
        for (c, v) in s.iter_mut().enumerate() {
            *v = columns[c * nz + z];
        }
    });
    out
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
/// The log-odds of a voxel no image has observed, which a robot mask has hidden.
pub(crate) const SHADOWED: i8 = i8::MIN + 1;

/// Occupancy fused from depth images over time, on a fixed grid in the world frame: per voxel, the
/// log-odds that it is occupied, updated as OctoMap updates them (a hit adds 0.85, a miss -0.4,
/// clamped to [-2, 3.5]). Readings that come and go settle: something seen once and then seen
/// through three times is free again. Build distance grids from it with [`OccupancyMap::grid`].
#[derive(Clone, Debug)]
pub struct OccupancyMap {
    grid: Layout,
    log_odds: Vec<i8>,
}

impl OccupancyMap {
    /// An unobserved map with `voxel` spacing whose points cover `lo..=hi`.
    pub fn new(lo: Vec3, hi: Vec3, voxel: f32) -> Result<Self> {
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
        device.integrate(&self.grid, &mut self.log_odds, image, robot)
    }

    /// The probability that the voxel holding `p` is occupied; `None` if it was never observed or
    /// lies outside the map.
    pub fn occupancy(&self, p: Vec3) -> Option<f32> {
        let l = self.log_odds[self.grid.nearest(p)];
        (self.grid.contains(p) && l != UNOBSERVED && l != SHADOWED)
            .then(|| 1.0 / (1.0 + (-f32::from(l) / 20.0).exp()))
    }

    /// The distance grid of the map: voxels more likely occupied than not are solid, and voxels
    /// never observed are solid or free as `unknown` says. Errors if nothing is solid.
    pub fn grid(&self, device: &Device, unknown: Occlusion) -> Result<SdfGrid> {
        let occupied: Vec<bool> = self
            .log_odds
            .par_iter()
            .map(|&l| match l {
                UNOBSERVED => unknown != Occlusion::Free,
                SHADOWED => unknown == Occlusion::Occupied,
                _ => l > 0,
            })
            .collect();
        ensure_input!(
            occupied.iter().any(|&o| o),
            "the map holds nothing occupied"
        );
        SdfGrid::from_occupancy(device, &self.grid, &occupied)
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
        let (depth, intrinsics, camera) = table_image();
        let image = DepthImage {
            depth: &depth,
            width: 640,
            intrinsics,
            camera,
        };
        let mask = image.robot_mask(&robot, robot.default_q(), 0.02).unwrap();
        assert!(
            mask.depth.iter().filter(|d| d.is_finite()).count() > 10_000,
            "the camera should see the arm"
        );
        let grid = layout(
            Vec3::new(-0.6, -0.6, -0.1),
            Vec3::new(0.6, 0.6, 1.0),
            &SdfOptions {
                voxel: 0.01,
                padding: 0.0,
            },
        )
        .unwrap();
        let fused = |device: &Device| {
            let mut log_odds = vec![UNOBSERVED; grid.points()];
            device
                .integrate(&grid, &mut log_odds, &image, None)
                .unwrap();
            device
                .integrate(&grid, &mut log_odds, &image, Some(&mask))
                .unwrap();
            log_odds
        };
        let (a, b) = (fused(&cpu), fused(&gpu));
        let count = |l: i8| a.iter().filter(|&&x| x == l).count();
        assert!(count(HIT + HIT) > 2_000, "the table should be hit twice");
        assert!(
            count(HIT) > 2_000,
            "the arm should hide part of the table the second time"
        );
        assert!(
            count(MISS + MISS) > 100_000,
            "the space in front of the table should be cleared twice"
        );
        assert!(
            count(SHADOWED) > 1_000,
            "the arm should hide space no image has seen"
        );
        let differ = a.iter().zip(&b).filter(|(a, b)| a != b).count();
        assert!(
            differ * 2000 <= a.len(),
            "{differ} of {} log-odds differ",
            a.len()
        );
    }
}
