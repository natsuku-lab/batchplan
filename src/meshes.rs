//! Checking configurations against the robot's own geometry rather than its collision spheres.
//!
//! Planning and validation run on spheres, which reach beyond the links' surfaces. A
//! [`MeshModel`] measures the same clearances between the links' meshes, the world's obstacles and
//! the link pairs checked for self-collision, the way MoveIt checks a planning scene with FCL. It
//! runs on the CPU and is meant for finished trajectories, not for the inner loops.

use std::f32::consts::FRAC_PI_2;
use std::path::PathBuf;
use std::sync::Arc;

use glam::{Quat, Vec3};
use parry3d::math::Pose;
use parry3d::query::distance;
use parry3d::shape::{SharedShape, TriMesh as ParryMesh, TriMeshFlags};
use rayon::prelude::*;

use crate::description::{Geometry, TriMesh};
use crate::error::{Error, Result, ensure_input, input};
use crate::robot::Robot;
use crate::sdf::{SdfGrid, grid_distance};
use crate::world::{FAR, Obstacle, World};

/// Points cover each link's surface for checks against distance grids: every surface point lies
/// within this distance of one.
const SURFACE_SPACING: f32 = 0.005;

/// A robot's links as triangle meshes, for clearances measured against its geometry.
///
/// Meshes are surfaces, as in FCL: a link entirely inside another closed mesh does not touch it.
/// Links that no joint moves (a robot's base and what is fixed to it) are checked against the
/// robot's moving links but not against the world, which they touch the same way whatever the
/// motion, as a robot mounted on a table rests on it.
/// Primitive obstacles are solid. Against distance grids, which hold no exact surface, a link's
/// clearance is the grid's distance at points covering its surface to within 5 mm, less 5 mm.
pub struct MeshModel {
    robot: Robot,
    links: Vec<Option<LinkMesh>>,
}

struct LinkMesh {
    mesh: SharedShape,
    /// The mesh's convex hull: nothing is nearer to the hull than to the mesh, so the hull's
    /// distance bounds the mesh's cheaply.
    hull: SharedShape,
    /// A sphere around the mesh: centre in the link frame, radius.
    bound: (Vec3, f32),
    /// Points covering the surface, in the link frame.
    surface: Vec<Vec3>,
}

/// An obstacle as the checks read it, with a sphere around it (world centre, radius).
enum Placed {
    Solid(Pose, SharedShape, (Vec3, f32)),
    Grid(Arc<SdfGrid>, Vec3, Quat, (Vec3, f32)),
}

impl MeshModel {
    /// Loads the meshes of `robot`'s links: their collision geometry, or the visual geometry where
    /// spheres were fitted to that. Links without geometry are not checked.
    pub fn new(robot: &Robot) -> Result<Self> {
        let links = robot
            .links
            .iter()
            .map(|link| {
                let mut all = TriMesh::default();
                for shape in &link.shapes {
                    let mesh = shape.mesh().map_err(|e| {
                        let path = match &shape.geometry {
                            Geometry::Mesh { path, .. } => path.clone(),
                            _ => PathBuf::from(&link.name),
                        };
                        Error::Load { path, message: format!("{e:#}") }
                    })?;
                    all.append(mesh);
                }
                if all.triangles.is_empty() {
                    return Ok(None);
                }
                let (lo, hi) = all
                    .vertices
                    .iter()
                    .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                let centre = (lo + hi) / 2.0;
                let radius = all.vertices.iter().map(|v| v.distance(centre)).fold(0.0, f32::max);
                let surface = cover(&all, SURFACE_SPACING);
                let hull = SharedShape::convex_hull(&all.vertices)
                    .ok_or_else(|| input!("link '{}' has a flat mesh with no convex hull", link.name))?;
                let flags = TriMeshFlags::MERGE_DUPLICATE_VERTICES | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES;
                let mesh = ParryMesh::with_flags(all.vertices, all.triangles, flags)
                    .map_err(|e| input!("link '{}' has no usable mesh: {e}", link.name))?;
                Ok(Some(LinkMesh { mesh: SharedShape::new(mesh), hull, bound: (centre, radius), surface }))
            })
            .collect::<Result<_>>()?;
        Ok(Self { robot: robot.clone(), links })
    }

    /// World and self clearance (meters) of each configuration (`[items, dof]`): from the links'
    /// meshes to the obstacles of `worlds[item_world[i]]`, and between the meshes of the link
    /// pairs checked for self-collision. Zero where they touch or overlap; `1e30` where there is
    /// nothing to measure. Runs on the current rayon pool.
    pub fn clearance(&self, worlds: &[World], item_world: &[u32], q: &[f32]) -> Result<Vec<[f32; 2]>> {
        let n = self.robot.dof();
        ensure_input!(
            q.len() == item_world.len() * n,
            "{} values for {} configurations of {n}",
            q.len(),
            item_world.len()
        );
        if let Some(w) = item_world.iter().find(|&&w| w as usize >= worlds.len()) {
            return Err(input!("world {w} of {}", worlds.len()));
        }
        for (i, w) in worlds.iter().enumerate() {
            w.check().map_err(|e| input!("world {i}: {e}"))?;
        }
        let placed: Vec<Vec<Placed>> = worlds.iter().map(|w| w.obstacles.iter().map(place).collect()).collect();
        Ok(q.par_chunks(n).zip(item_world).map(|(q, &w)| self.clearances(&placed[w as usize], q)).collect())
    }

    /// Branch and bound: pairs in order of their bounding spheres' gap, which stops the search
    /// once it reaches the nearest distance found; then each pair's hulls, then its meshes.
    fn clearances(&self, obstacles: &[Placed], q: &[f32]) -> [f32; 2] {
        let fk = self.robot.fk(q);
        let pose = |i: usize| Pose::from_parts(fk.pos[i], Quat::from_mat3(&fk.rot[i]));
        let bound = |i: usize, m: &LinkMesh| (fk.rot[i] * m.bound.0 + fk.pos[i], m.bound.1);
        let gap = |(a, ra): (Vec3, f32), (b, rb): (Vec3, f32)| a.distance(b) - ra - rb;
        let apart = |p1: &Pose, s1: &SharedShape, p2: &Pose, s2: &SharedShape| {
            distance(p1, s1.as_ref(), p2, s2.as_ref()).map_or(0.0, |d| d.distance)
        };
        let nearest = |mut pairs: Vec<(f32, usize, usize)>, exact: &dyn Fn(usize, usize, f32) -> f32| {
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut best = FAR;
            for (lower, a, b) in pairs {
                if lower >= best {
                    break;
                }
                best = best.min(exact(a, b, best));
            }
            best
        };

        // Links no joint moves touch the scene the same way whatever the motion: a robot mounted on
        // a table rests on it.
        let links: Vec<(usize, &LinkMesh)> = self
            .links
            .iter()
            .enumerate()
            .filter(|&(i, _)| !self.robot.links[i].chain.is_empty())
            .filter_map(|(i, l)| l.as_ref().map(|l| (i, l)))
            .collect();
        let world_pairs = links
            .iter()
            .enumerate()
            .flat_map(|(k, &(i, link))| {
                obstacles.iter().enumerate().map(move |(o, obstacle)| {
                    let (Placed::Solid(_, _, around) | Placed::Grid(.., around)) = obstacle;
                    (gap(bound(i, link), *around), k, o)
                })
            })
            .collect();
        let world = nearest(world_pairs, &|k, o, best| {
            let (i, link) = links[k];
            match &obstacles[o] {
                Placed::Solid(at, shape, _) => {
                    if apart(&pose(i), &link.hull, at, shape) >= best {
                        return best;
                    }
                    apart(&pose(i), &link.mesh, at, shape)
                }
                Placed::Grid(grid, centre, rotation, _) => {
                    let to_grid = rotation.inverse();
                    let nearest = link
                        .surface
                        .iter()
                        .map(|&p| grid_distance(grid, to_grid * (fk.rot[i] * p + fk.pos[i] - *centre)).0)
                        .fold(f32::INFINITY, f32::min);
                    (nearest - SURFACE_SPACING).max(0.0)
                }
            }
        });

        let self_pairs = self
            .robot
            .self_link_pairs
            .iter()
            .filter_map(|pair| {
                let (a, b) = (pair.a as usize, pair.b as usize);
                let (Some(ma), Some(mb)) = (&self.links[a], &self.links[b]) else { return None };
                Some((gap(bound(a, ma), bound(b, mb)), a, b))
            })
            .collect();
        let self_collision = nearest(self_pairs, &|a, b, best| {
            let (ma, mb) = (self.links[a].as_ref().expect("paired"), self.links[b].as_ref().expect("paired"));
            if apart(&pose(a), &ma.hull, &pose(b), &mb.hull) >= best {
                return best;
            }
            apart(&pose(a), &ma.mesh, &pose(b), &mb.mesh)
        });
        [world, self_collision]
    }
}

fn place(obstacle: &Obstacle) -> Placed {
    let solid = |centre: Vec3, rotation: Quat, shape: SharedShape, radius: f32| {
        Placed::Solid(Pose::from_parts(centre, rotation), shape, (centre, radius))
    };
    match *obstacle {
        Obstacle::Cuboid { center, half_extents: h, rotation } => {
            solid(center, rotation, SharedShape::cuboid(h.x, h.y, h.z), h.length())
        }
        Obstacle::Sphere { center, radius } => solid(center, Quat::IDENTITY, SharedShape::ball(radius), radius),
        // parry's cylinders run along their y axis, ours along z.
        Obstacle::Cylinder { center, rotation, radius, half_height } => solid(
            center,
            rotation * Quat::from_rotation_x(FRAC_PI_2),
            SharedShape::cylinder(half_height, radius),
            radius.hypot(half_height),
        ),
        Obstacle::Capsule { center, rotation, radius, half_length } => {
            solid(center, rotation, SharedShape::capsule_z(half_length, radius), radius + half_length)
        }
        Obstacle::Sdf { ref grid, center, rotation } => {
            let (lo, hi) = grid.bounds();
            let around = (center + rotation * ((lo + hi) / 2.0), (hi - lo).length() / 2.0);
            Placed::Grid(grid.clone(), center, rotation, around)
        }
    }
}

/// Points on `mesh` such that every point of its surface lies within `spacing` of one: each
/// triangle divided evenly until its pieces' edges are at most `spacing` long.
fn cover(mesh: &TriMesh, spacing: f32) -> Vec<Vec3> {
    let mut points = vec![];
    for &t in &mesh.triangles {
        let [a, b, c] = mesh.corners(t);
        let longest = a.distance(b).max(b.distance(c)).max(c.distance(a));
        let n = (longest / spacing).ceil().max(1.0) as u32;
        for i in 0..=n {
            for j in 0..=n - i {
                points.push(a + (b - a) * (i as f32 / n as f32) + (c - a) * (j as f32 / n as f32));
            }
        }
    }
    points
}
