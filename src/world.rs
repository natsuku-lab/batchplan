//! Collision worlds: obstacles and their signed distance functions. Batched queries reference
//! worlds by index, so one call can cover thousands of different environments.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;

use crate::error::{Error, Result};
use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::description::Geometry;
use crate::sdf::{SdfGrid, SdfOptions, grid_distance};
use crate::types::Pose;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Obstacle {
    /// Oriented box; `rotation` maps box axes to world axes.
    Cuboid {
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
    },
    Sphere {
        center: Vec3,
        radius: f32,
    },
    /// Solid cylinder along its local z axis; `rotation` maps local axes to world axes.
    Cylinder {
        center: Vec3,
        rotation: Quat,
        radius: f32,
        half_height: f32,
    },
    /// Capsule around the local z segment from `-half_length` to `half_length`.
    Capsule {
        center: Vec3,
        rotation: Quat,
        radius: f32,
        half_length: f32,
    },
    /// A signed distance grid with its frame at `center`, turned by `rotation`. Worlds can share
    /// a grid; a device uploads it once.
    Sdf {
        grid: Arc<SdfGrid>,
        center: Vec3,
        rotation: Quat,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct World {
    pub obstacles: Vec<Obstacle>,
}

/// Value returned for clearances when there is nothing to collide with.
pub(crate) const FAR: f32 = 1e30;

/// Half thickness of the slab a plane becomes, and its half extent when the file gives none.
const PLANE_SLAB: f32 = 0.05;
const PLANE_EXTENT: f32 = 50.0;

impl World {
    /// The static collision geometry of a scene file: MJCF (`.xml`, `.mjcf`) geoms in the world
    /// body and in bodies without joints, or OpenUSD (`.usd*`, feature `usd`) colliders outside
    /// rigid bodies. Planes become slabs below their surface; ellipsoids become their bounding
    /// boxes; meshes become distance grids laid out by `grids`, as convex hulls where the format
    /// collides them that way (MJCF always, USD with `physics:approximation = "convexHull"`).
    /// Robots in the same file are left out.
    pub fn load(path: impl AsRef<Path>, grids: &SdfOptions) -> Result<World> {
        let path = path.as_ref();
        Self::read(path, grids).map_err(|e| Error::Load { path: path.to_path_buf(), message: format!("{e:#}") })
    }

    fn read(path: &Path, grids: &SdfOptions) -> anyhow::Result<World> {
        let scene = crate::description::load_scene(path)?;
        let obstacles = scene
            .iter()
            .map(|s| {
                let (center, rotation) = (s.origin.trans, Quat::from_mat3(&s.origin.rot));
                Ok(match &s.geometry {
                    Geometry::Box { half } => Obstacle::Cuboid { center, half_extents: *half, rotation },
                    Geometry::Sphere { radius } => Obstacle::Sphere { center, radius: *radius },
                    Geometry::Cylinder { radius, half_length } => {
                        Obstacle::Cylinder { center, rotation, radius: *radius, half_height: *half_length }
                    }
                    Geometry::Capsule { radius, half_length } => {
                        Obstacle::Capsule { center, rotation, radius: *radius, half_length: *half_length }
                    }
                    Geometry::Ellipsoid { radii } => Obstacle::Cuboid { center, half_extents: *radii, rotation },
                    Geometry::Plane => Obstacle::Cuboid {
                        center: center - s.origin.rot * Vec3::Z * PLANE_SLAB,
                        half_extents: Vec3::new(PLANE_EXTENT, PLANE_EXTENT, PLANE_SLAB),
                        rotation,
                    },
                    Geometry::Mesh { .. } | Geometry::TriMesh(_) | Geometry::ConvexHull(_) => {
                        let mesh = s.geometry.mesh()?;
                        let grid = SdfGrid::from_mesh(&mesh.vertices, &mesh.triangles, grids)
                            .with_context(|| format!("the scene mesh at {center}"))?;
                        Obstacle::Sdf { grid: Arc::new(grid), center, rotation }
                    }
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(World { obstacles })
    }
}

/// Worlds as JSON with each distance grid written once: `{"grids": [...], "worlds": [...]}`,
/// where an `sdf` obstacle names its grid by index.
pub(crate) fn worlds_json(worlds: &[World]) -> serde_json::Result<serde_json::Value> {
    let mut grids: Vec<&Arc<SdfGrid>> = vec![];
    let worlds = worlds
        .iter()
        .map(|w| {
            let obstacles = w
                .obstacles
                .iter()
                .map(|o| match o {
                    Obstacle::Sdf { grid, center, rotation } => {
                        let index = grids.iter().position(|g| Arc::ptr_eq(g, grid)).unwrap_or_else(|| {
                            grids.push(grid);
                            grids.len() - 1
                        });
                        Ok(serde_json::json!({"type": "sdf", "grid": index, "center": center, "rotation": rotation}))
                    }
                    o => serde_json::to_value(o),
                })
                .collect::<serde_json::Result<Vec<_>>>()?;
            Ok(serde_json::json!({ "obstacles": obstacles }))
        })
        .collect::<serde_json::Result<Vec<_>>>()?;
    Ok(serde_json::json!({ "grids": grids, "worlds": worlds }))
}

impl World {
    /// Why these obstacles cannot be collision-checked, if they cannot.
    pub(crate) fn check(&self) -> Result<()> {
        for (k, o) in self.obstacles.iter().enumerate() {
            o.check().map_err(|e| Error::Input(format!("obstacle {k}: {e}")))?;
        }
        Ok(())
    }
}

impl Obstacle {
    fn check(&self) -> Result<()> {
        let size = |v: f32| v.is_finite() && v >= 0.0;
        let pose = |center: Vec3, rotation: Quat| {
            center.is_finite() && rotation.is_finite() && (rotation.length() - 1.0).abs() < 1e-3
        };
        let ok = match *self {
            Obstacle::Cuboid { center, half_extents, rotation } => {
                pose(center, rotation) && half_extents.to_array().into_iter().all(size)
            }
            Obstacle::Sphere { center, radius } => center.is_finite() && size(radius),
            Obstacle::Cylinder { center, rotation, radius, half_height: half } => {
                pose(center, rotation) && size(radius) && size(half)
            }
            Obstacle::Capsule { center, rotation, radius, half_length: half } => {
                pose(center, rotation) && size(radius) && size(half)
            }
            Obstacle::Sdf { ref grid, center, rotation } => {
                grid.check()?;
                pose(center, rotation)
            }
        };
        if ok {
            Ok(())
        } else {
            Err(Error::Input("needs finite, non-negative sizes and a unit-quaternion rotation".into()))
        }
    }

    /// The obstacle moved so that its frame (centre and rotation) is `pose`. Spheres keep no
    /// rotation.
    pub fn placed(&self, pose: Pose) -> Obstacle {
        let (center, rotation) = (pose.position, pose.rotation);
        match self.clone() {
            Obstacle::Cuboid { half_extents, .. } => Obstacle::Cuboid { center, half_extents, rotation },
            Obstacle::Sphere { radius, .. } => Obstacle::Sphere { center, radius },
            Obstacle::Cylinder { radius, half_height, .. } => {
                Obstacle::Cylinder { center, rotation, radius, half_height }
            }
            Obstacle::Capsule { radius, half_length, .. } => {
                Obstacle::Capsule { center, rotation, radius, half_length }
            }
            Obstacle::Sdf { grid, .. } => Obstacle::Sdf { grid, center, rotation },
        }
    }

    /// The obstacle's frame: its centre and rotation.
    pub fn pose(&self) -> Pose {
        let (position, rotation) = match *self {
            Obstacle::Cuboid { center, rotation, .. }
            | Obstacle::Cylinder { center, rotation, .. }
            | Obstacle::Capsule { center, rotation, .. }
            | Obstacle::Sdf { center, rotation, .. } => (center, rotation),
            Obstacle::Sphere { center, .. } => (center, Quat::IDENTITY),
        };
        Pose { position, rotation }
    }

    /// Signed distance from `p` to the obstacle surface and its gradient (the unit outward
    /// direction, except inside distance grids, where it is the interpolation's gradient).
    pub fn distance(&self, p: Vec3) -> (f32, Vec3) {
        match *self {
            Obstacle::Cuboid { center, half_extents, rotation } => {
                let r = Mat3::from_quat(rotation);
                box_distance(r, center, half_extents, p)
            }
            Obstacle::Sphere { center, radius } => sphere_distance(center, radius, p),
            Obstacle::Cylinder { center, rotation, radius, half_height } => {
                cylinder_distance(Mat3::from_quat(rotation), center, radius, half_height, p)
            }
            Obstacle::Capsule { center, rotation, radius, half_length } => {
                capsule_distance(Mat3::from_quat(rotation), center, radius, half_length, p)
            }
            Obstacle::Sdf { ref grid, center, rotation } => sdf_distance(Mat3::from_quat(rotation), center, grid, p),
        }
    }
}

#[inline]
pub(crate) fn sphere_distance(center: Vec3, radius: f32, p: Vec3) -> (f32, Vec3) {
    let v = p - center;
    let len = v.length();
    (len - radius, if len > 1e-9 { v / len } else { Vec3::Z })
}

/// Must stay in sync with `obstacle_distance` in kernels.wgsl.
#[inline]
pub(crate) fn box_distance(r: Mat3, center: Vec3, half: Vec3, p: Vec3) -> (f32, Vec3) {
    let lp = r.transpose() * (p - center);
    let sgn = Vec3::select(lp.cmpge(Vec3::ZERO), Vec3::ONE, Vec3::NEG_ONE);
    let qd = lp.abs() - half;
    let outside = qd.max(Vec3::ZERO);
    let olen = outside.length();
    let (d, gl) = if olen > 0.0 {
        (olen, sgn * outside / olen)
    } else if qd.x >= qd.y && qd.x >= qd.z {
        (qd.x, Vec3::new(sgn.x, 0.0, 0.0))
    } else if qd.y >= qd.z {
        (qd.y, Vec3::new(0.0, sgn.y, 0.0))
    } else {
        (qd.z, Vec3::new(0.0, 0.0, sgn.z))
    };
    (d, r * gl)
}

/// Must stay in sync with `obstacle_distance` in kernels.wgsl.
#[inline]
pub(crate) fn cylinder_distance(r: Mat3, center: Vec3, radius: f32, half_height: f32, p: Vec3) -> (f32, Vec3) {
    let lp = r.transpose() * (p - center);
    let rho = (lp.x * lp.x + lp.y * lp.y).sqrt();
    let radial = if rho > 1e-9 { Vec3::new(lp.x / rho, lp.y / rho, 0.0) } else { Vec3::X };
    let axial = Vec3::new(0.0, 0.0, if lp.z >= 0.0 { 1.0 } else { -1.0 });
    let (dr, dz) = (rho - radius, lp.z.abs() - half_height);
    let (d, gl) = if dr > 0.0 && dz > 0.0 {
        let d = (dr * dr + dz * dz).sqrt();
        (d, (radial * dr + axial * dz) / d)
    } else if dr >= dz {
        (dr, radial)
    } else {
        (dz, axial)
    };
    (d, r * gl)
}

/// Must stay in sync with `obstacle_distance` in kernels.wgsl.
#[inline]
pub(crate) fn sdf_distance(r: Mat3, center: Vec3, grid: &SdfGrid, p: Vec3) -> (f32, Vec3) {
    let (d, g) = grid_distance(grid, r.transpose() * (p - center));
    (d, r * g)
}

/// Must stay in sync with `obstacle_distance` in kernels.wgsl.
#[inline]
pub(crate) fn capsule_distance(r: Mat3, center: Vec3, radius: f32, half_length: f32, p: Vec3) -> (f32, Vec3) {
    let lp = r.transpose() * (p - center);
    let v = lp - Vec3::new(0.0, 0.0, lp.z.clamp(-half_length, half_length));
    let len = v.length();
    (len - radius, r * if len > 1e-9 { v / len } else { Vec3::X })
}
