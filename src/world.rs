//! Collision worlds: obstacles and their signed distance functions. Batched queries reference
//! worlds by index, so one call can cover thousands of different environments.

use std::path::Path;
use std::sync::Arc;

use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::description::Geometry;
use crate::error::{Error, Result};
use crate::held;
use crate::rng::Rng;
use crate::robot::Robot;
use crate::sdf::{self, SdfGrid, SdfOptions};
use crate::types::Pose;

/// A solid shape a robot must not touch, in world coordinates (meters). Each is placed by its
/// centre and, except for a sphere, a rotation that maps its own axes to the world's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
#[allow(
    missing_docs,
    reason = "the enum's doc gives every field's units and frame"
)]
pub enum Obstacle {
    /// Oriented box.
    Cuboid {
        center: Vec3,
        /// Half the box's size along each of its axes.
        half_extents: Vec3,
        rotation: Quat,
    },
    Sphere {
        center: Vec3,
        radius: f32,
    },
    /// Solid cylinder along its local z axis.
    Cylinder {
        center: Vec3,
        rotation: Quat,
        radius: f32,
        /// Half its length along its axis.
        half_height: f32,
    },
    /// Capsule around the local z segment from `-half_length` to `half_length`.
    Capsule {
        center: Vec3,
        rotation: Quat,
        radius: f32,
        half_length: f32,
    },
    /// A signed distance grid with its frame at `center`. Worlds can share a grid; a device
    /// uploads it once.
    Sdf {
        grid: Arc<SdfGrid>,
        center: Vec3,
        rotation: Quat,
    },
}

/// One scene to plan in. Problems name their world by its index in the batch uploaded with
/// [`Device::upload`](crate::Device::upload).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct World {
    /// Everything in the scene; uploading checks each is well formed.
    pub obstacles: Vec<Obstacle>,
    /// Links allowed to touch particular obstacles, as fingers touch what they grasp: contact
    /// between them is not a collision, anywhere batchplan checks one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub touch: Vec<Touch>,
}

/// Links that may touch one obstacle of a [`World`]. Their names are resolved when the world is
/// uploaded, against the robot of the device it is uploaded to; devices made from that one with
/// [`Device::with_robot`](crate::Device::with_robot) for an attached object keep them. What a
/// problem's held object may touch is named with [`Touch::held`], on the device that holds
/// nothing; a world uploaded to a device that holds an object names its link.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Touch {
    /// The obstacle's index in [`World::obstacles`].
    pub obstacle: usize,
    /// Links of the robot, such as a hand's ([`Robot::hand_links`](crate::Robot::hand_links)).
    pub links: Vec<String>,
}

impl Touch {
    /// Lets `links` touch obstacle `obstacle`: `Touch::new(0, robot.hand_links(0))`.
    #[must_use]
    pub fn new<S: Into<String>>(obstacle: usize, links: impl IntoIterator<Item = S>) -> Touch {
        Touch {
            obstacle,
            links: links.into_iter().map(Into::into).collect(),
        }
    }

    /// Lets what hand `hand` holds touch obstacle `obstacle`, as an object rests on what it is set
    /// down on: the object of a problem's `holding` ([`Held`](crate::Held)). The device that
    /// holds nothing takes it, for one hand in an upload; problems that hold nothing are checked
    /// as without it.
    #[must_use]
    pub fn held(obstacle: usize, hand: usize) -> Touch {
        Touch {
            obstacle,
            links: vec![held::held_link(hand)],
        }
    }
}

/// A signed distance to a surface, and its gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Distance {
    /// Meters: negative inside.
    pub value: f32,
    /// How the distance grows with the point: the unit outward direction, except inside
    /// distance grids, where it is the interpolation's gradient.
    pub gradient: Vec3,
}

/// Random configurations [`World::always_touching`] tests besides the default one, and their seed.
const ALWAYS_SAMPLES: usize = 1000;
const ALWAYS_SEED: u64 = 11;

/// Value returned for clearances when there is nothing to collide with.
pub(crate) const FAR: f32 = 1e30;

/// Half thickness of the slab a plane becomes, and its half extent when the file gives none.
const PLANE_SLAB: f32 = 0.05;
const PLANE_EXTENT: f32 = 50.0;

impl World {
    /// A world of `obstacles`, none of which any link may touch.
    #[must_use]
    pub fn new(obstacles: Vec<Obstacle>) -> World {
        World {
            obstacles,
            touch: vec![],
        }
    }

    /// Takes obstacle `obstacle` out of the world, as a hand picking it up does, and returns it.
    /// Its touches go with it, and the touches of the obstacles after it follow them down.
    pub fn remove(&mut self, obstacle: usize) -> Result<Obstacle> {
        let count = self.obstacles.len();
        if obstacle >= count {
            return Err(Error::Input(format!("obstacle {obstacle} of {count}")));
        }
        self.touch.retain(|t| t.obstacle != obstacle);
        for t in &mut self.touch {
            if t.obstacle > obstacle {
                t.obstacle -= 1;
            }
        }
        Ok(self.obstacles.remove(obstacle))
    }

    /// Touches for the links of `robot` that touch an obstacle in every configuration, as the
    /// links just above a robot's base rest on the table it stands on: from rest there, every
    /// plan's start would collide. Add them to [`World::touch`]. MoveIt's setup assistant finds
    /// link pairs that always collide the same way, by sampling: a link counts where its spheres
    /// overlap the obstacle at the default configuration and at each of 1,000 random ones within
    /// the joint limits. Links no joint moves are never checked against the world, so they are
    /// left out; so are links the world already lets touch the obstacle. One touch per obstacle
    /// that has any, in obstacle order.
    #[must_use]
    pub fn always_touching(&self, robot: &Robot) -> Vec<Touch> {
        let mut rng = Rng::new(ALWAYS_SEED);
        // Within `lower` and `upper`: a continuous joint's are the turn it samples from.
        let samples =
            std::iter::once(robot.default_q.clone()).chain((0..ALWAYS_SAMPLES).map(|_| {
                (0..robot.dof())
                    .map(|j| rng.range(robot.lower[j], robot.upper[j]))
                    .collect::<Vec<f32>>()
            }));
        let allowed = |obstacle: usize, link: &str| {
            self.touch
                .iter()
                .any(|t| t.obstacle == obstacle && t.links.iter().any(|l| l == link))
        };
        // Per obstacle, the moving links with spheres that touch it in every sample so far.
        let mut touching: Vec<Vec<usize>> = (0..self.obstacles.len())
            .map(|k| {
                (0..robot.links.len())
                    .filter(|&l| robot.sphere_ranges[l][1] > 0 && !robot.links[l].chain.is_empty())
                    .filter(|&l| !allowed(k, &robot.links[l].name))
                    .collect()
            })
            .collect();
        for q in samples {
            if touching.iter().all(Vec::is_empty) {
                break;
            }
            let fk = robot.fk(&q);
            for (obstacle, links) in self.obstacles.iter().zip(&mut touching) {
                links.retain(|&link| {
                    let [first, count] = robot.sphere_ranges[link];
                    robot.spheres[first as usize..(first + count) as usize]
                        .iter()
                        .any(|s| {
                            let center = fk.rot[link] * s.center + fk.pos[link];
                            obstacle.distance(center).value < s.radius
                        })
                });
            }
        }
        touching
            .into_iter()
            .enumerate()
            .filter(|(_, links)| !links.is_empty())
            .map(|(k, links)| Touch::new(k, links.into_iter().map(|l| robot.links[l].name.clone())))
            .collect()
    }

    /// The static collision geometry of a scene file: MJCF (`.xml`, `.mjcf`) geoms in the world
    /// body and in bodies without joints, or OpenUSD (`.usd*`, feature `usd`) colliders outside
    /// rigid bodies. Planes become slabs below their surface; ellipsoids become their bounding
    /// boxes; meshes become distance grids laid out by `grids`, as convex hulls where the format
    /// collides them that way (MJCF always, USD with `physics:approximation = "convexHull"`).
    /// Robots in the same file are left out.
    pub fn load(path: impl AsRef<Path>, grids: &SdfOptions) -> Result<World> {
        let path = path.as_ref();
        Self::read(path, grids).map_err(|e| Error::Load {
            path: path.to_path_buf(),
            message: format!("{e:#}"),
        })
    }

    fn read(path: &Path, grids: &SdfOptions) -> anyhow::Result<World> {
        use anyhow::Context;

        let scene = crate::description::load_scene(path)?;
        let obstacles = scene
            .iter()
            .map(|s| {
                let (center, rotation) = (s.origin.trans, Quat::from_mat3(&s.origin.rot));
                Ok(match &s.geometry {
                    Geometry::Box { half } => Obstacle::Cuboid {
                        center,
                        half_extents: *half,
                        rotation,
                    },
                    Geometry::Sphere { radius } => Obstacle::Sphere {
                        center,
                        radius: *radius,
                    },
                    Geometry::Cylinder {
                        radius,
                        half_length,
                    } => Obstacle::Cylinder {
                        center,
                        rotation,
                        radius: *radius,
                        half_height: *half_length,
                    },
                    Geometry::Capsule {
                        radius,
                        half_length,
                    } => Obstacle::Capsule {
                        center,
                        rotation,
                        radius: *radius,
                        half_length: *half_length,
                    },
                    Geometry::Ellipsoid { radii } => Obstacle::Cuboid {
                        center,
                        half_extents: *radii,
                        rotation,
                    },
                    Geometry::Plane => Obstacle::Cuboid {
                        center: center - s.origin.rot * Vec3::Z * PLANE_SLAB,
                        half_extents: Vec3::new(PLANE_EXTENT, PLANE_EXTENT, PLANE_SLAB),
                        rotation,
                    },
                    Geometry::Mesh { .. } | Geometry::TriMesh(_) | Geometry::ConvexHull(_) => {
                        let mesh = s.geometry.mesh()?;
                        let grid = SdfGrid::from_mesh(&mesh.vertices, &mesh.triangles, grids)
                            .with_context(|| format!("the scene mesh at {center}"))?;
                        Obstacle::Sdf {
                            grid: Arc::new(grid),
                            center,
                            rotation,
                        }
                    }
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(World::new(obstacles))
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
                    Obstacle::Sdf {
                        grid,
                        center,
                        rotation,
                    } => {
                        let index = grids
                            .iter()
                            .position(|g| Arc::ptr_eq(g, grid))
                            .unwrap_or_else(|| {
                                grids.push(grid);
                                grids.len() - 1
                            });
                        Ok(serde_json::json!({
                            "type": "sdf",
                            "grid": index,
                            "center": center,
                            "rotation": rotation,
                        }))
                    }
                    o => serde_json::to_value(o),
                })
                .collect::<serde_json::Result<Vec<_>>>()?;
            let mut world = serde_json::json!({ "obstacles": obstacles });
            if !w.touch.is_empty() {
                world["touch"] = serde_json::to_value(&w.touch)?;
            }
            Ok(world)
        })
        .collect::<serde_json::Result<Vec<_>>>()?;
    Ok(serde_json::json!({ "grids": grids, "worlds": worlds }))
}

impl World {
    /// Why these obstacles cannot be collision-checked, if they cannot.
    pub(crate) fn check(&self) -> Result<()> {
        for (k, o) in self.obstacles.iter().enumerate() {
            o.check()
                .map_err(|e| Error::Input(format!("obstacle {k}: {e}")))?;
        }
        for (k, t) in self.touch.iter().enumerate() {
            if t.obstacle >= self.obstacles.len() {
                return Err(Error::Input(format!(
                    "touch {k}: obstacle {} of {}",
                    t.obstacle,
                    self.obstacles.len()
                )));
            }
        }
        Ok(())
    }

    /// The links allowed to touch each obstacle, by index in `robot`'s links: `words` 64-bit
    /// words per obstacle, link `l` bit `l % 64` of word `l / 64`. Empty, with no words, when no
    /// link may touch anything.
    pub(crate) fn touch_sets(&self, robot: &Robot) -> Result<LinkSets> {
        if self.touch.is_empty() {
            return Ok(LinkSets::default());
        }
        // What a hand may come to hold is the link after the robot's own, as holding it adds it.
        let held = self.held_touch(robot)?.map(held::held_link);
        let words = (robot.links.len() + usize::from(held.is_some())).div_ceil(64);
        let mut bits = vec![0u64; self.obstacles.len() * words];
        for (k, t) in self.touch.iter().enumerate() {
            for name in &t.links {
                let own = robot.links.iter().position(|l| &l.name == name);
                let link = own
                    .or_else(|| (held.as_ref() == Some(name)).then_some(robot.links.len()))
                    .ok_or_else(|| {
                        Error::Input(format!(
                            "touch {k}: robot '{}' has no link '{name}'",
                            robot.name()
                        ))
                    })?;
                bits[t.obstacle * words + link / 64] |= 1 << (link % 64);
            }
        }
        Ok(LinkSets { bits, words })
    }

    /// The hand whose held object the world's touches name ([`Touch::held`]) though `robot` holds
    /// nothing in it: one hand's at most, since what either holds is the same link after the
    /// robot's own.
    pub(crate) fn held_touch(&self, robot: &Robot) -> Result<Option<usize>> {
        let mut held = None;
        for hand in 0..robot.hands() {
            let link = held::held_link(hand);
            let named = self.touch.iter().any(|t| t.links.contains(&link));
            if !named || robot.links.iter().any(|l| l.name == link) {
                continue;
            }
            if let Some(other) = held.replace(hand) {
                return Err(Error::Input(format!(
                    "touches name what hands {other} and {hand} hold: a world lets one hand's \
                     held object touch obstacles, on a device that holds nothing"
                )));
            }
        }
        Ok(held)
    }
}

/// The links allowed to touch each obstacle of a world ([`World::touch_sets`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct LinkSets {
    bits: Vec<u64>,
    words: usize,
}

impl LinkSets {
    /// Obstacle `obstacle`'s set: link `l` is bit `l % 64` of word `l / 64`. Empty when no link
    /// may touch anything.
    #[cfg(feature = "gpu")]
    pub(crate) fn row(&self, obstacle: usize) -> &[u64] {
        &self.bits[obstacle * self.words..(obstacle + 1) * self.words]
    }

    /// Whether `link` may touch obstacle `obstacle`. Links beyond those the sets were built for,
    /// such as an object attached since, touch nothing.
    #[inline(always)]
    pub(crate) fn contains(&self, obstacle: usize, link: usize) -> bool {
        let word = link / 64;
        word < self.words && self.bits[obstacle * self.words + word] >> (link % 64) & 1 == 1
    }
}

impl From<SdfGrid> for Obstacle {
    /// The grid in world coordinates (its frame the world's), where grids built from points and
    /// depth images belong.
    fn from(grid: SdfGrid) -> Self {
        Obstacle::Sdf {
            grid: Arc::new(grid),
            center: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

impl Obstacle {
    /// The grid in world coordinates, as `Obstacle::from` makes it.
    #[must_use]
    pub fn from_grid(grid: SdfGrid) -> Obstacle {
        Obstacle::from(grid)
    }

    /// An axis-aligned box; [`Obstacle::placed`] moves and turns it.
    #[must_use]
    pub fn cuboid(center: Vec3, half_extents: Vec3) -> Obstacle {
        Obstacle::Cuboid {
            center,
            half_extents,
            rotation: Quat::IDENTITY,
        }
    }

    /// A ball.
    #[must_use]
    pub fn sphere(center: Vec3, radius: f32) -> Obstacle {
        Obstacle::Sphere { center, radius }
    }

    /// A cylinder along the z axis; [`Obstacle::placed`] moves and turns it.
    #[must_use]
    pub fn cylinder(center: Vec3, radius: f32, half_height: f32) -> Obstacle {
        Obstacle::Cylinder {
            center,
            rotation: Quat::IDENTITY,
            radius,
            half_height,
        }
    }

    /// A capsule along the z axis; [`Obstacle::placed`] moves and turns it.
    #[must_use]
    pub fn capsule(center: Vec3, radius: f32, half_length: f32) -> Obstacle {
        Obstacle::Capsule {
            center,
            rotation: Quat::IDENTITY,
            radius,
            half_length,
        }
    }

    /// Why this obstacle cannot be collision-checked, if it cannot.
    pub(crate) fn check(&self) -> Result<()> {
        let size = |v: f32| v.is_finite() && v >= 0.0;
        let pose = |center: Vec3, rotation: Quat| {
            center.is_finite() && rotation.is_finite() && (rotation.length() - 1.0).abs() < 1e-3
        };
        let ok = match *self {
            Obstacle::Cuboid {
                center,
                half_extents,
                rotation,
            } => pose(center, rotation) && half_extents.to_array().into_iter().all(size),
            Obstacle::Sphere { center, radius } => center.is_finite() && size(radius),
            Obstacle::Cylinder {
                center,
                rotation,
                radius,
                half_height: half,
            } => pose(center, rotation) && size(radius) && size(half),
            Obstacle::Capsule {
                center,
                rotation,
                radius,
                half_length: half,
            } => pose(center, rotation) && size(radius) && size(half),
            // A grid is checked when it is made (`SdfGrid::checked`), and nothing changes one after.
            Obstacle::Sdf {
                center, rotation, ..
            } => pose(center, rotation),
        };
        if ok {
            Ok(())
        } else {
            Err(Error::Input(
                "needs finite, non-negative sizes and a unit-quaternion rotation".into(),
            ))
        }
    }

    /// The obstacle moved so that its frame (centre and rotation) is `pose`. Spheres keep no
    /// rotation.
    #[must_use]
    pub fn placed(&self, pose: Pose) -> Obstacle {
        let (center, rotation) = (pose.position, pose.rotation);
        match self.clone() {
            Obstacle::Cuboid { half_extents, .. } => Obstacle::Cuboid {
                center,
                half_extents,
                rotation,
            },
            Obstacle::Sphere { radius, .. } => Obstacle::Sphere { center, radius },
            Obstacle::Cylinder {
                radius,
                half_height,
                ..
            } => Obstacle::Cylinder {
                center,
                rotation,
                radius,
                half_height,
            },
            Obstacle::Capsule {
                radius,
                half_length,
                ..
            } => Obstacle::Capsule {
                center,
                rotation,
                radius,
                half_length,
            },
            Obstacle::Sdf { grid, .. } => Obstacle::Sdf {
                grid,
                center,
                rotation,
            },
        }
    }

    /// The obstacle's frame: its centre and rotation.
    #[must_use]
    pub fn pose(&self) -> Pose {
        let (position, rotation) = match *self {
            Obstacle::Cuboid {
                center, rotation, ..
            }
            | Obstacle::Cylinder {
                center, rotation, ..
            }
            | Obstacle::Capsule {
                center, rotation, ..
            }
            | Obstacle::Sdf {
                center, rotation, ..
            } => (center, rotation),
            Obstacle::Sphere { center, .. } => (center, Quat::IDENTITY),
        };
        Pose { position, rotation }
    }

    /// Signed distance from `p` to the obstacle surface, and its gradient.
    #[must_use]
    pub fn distance(&self, p: Vec3) -> Distance {
        let (value, gradient) = match *self {
            Obstacle::Cuboid {
                center,
                half_extents,
                rotation,
            } => {
                let r = frame(rotation);
                box_distance(r, center, half_extents, p)
            }
            Obstacle::Sphere { center, radius } => sphere_distance(center, radius, p),
            Obstacle::Cylinder {
                center,
                rotation,
                radius,
                half_height,
            } => cylinder_distance(frame(rotation), center, radius, half_height, p),
            Obstacle::Capsule {
                center,
                rotation,
                radius,
                half_length,
            } => capsule_distance(frame(rotation), center, radius, half_length, p),
            Obstacle::Sdf {
                ref grid,
                center,
                rotation,
            } => sdf_distance(frame(rotation), center, grid, p),
        };
        Distance { value, gradient }
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
pub(crate) fn cylinder_distance(
    r: Mat3,
    center: Vec3,
    radius: f32,
    half_height: f32,
    p: Vec3,
) -> (f32, Vec3) {
    let lp = r.transpose() * (p - center);
    let rho = (lp.x * lp.x + lp.y * lp.y).sqrt();
    let radial = if rho > 1e-9 {
        Vec3::new(lp.x / rho, lp.y / rho, 0.0)
    } else {
        Vec3::X
    };
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

/// A rotation the checks accept, made unit. They accept a quaternion a thousandth off unit, and
/// as it stands that one scales what it turns: by up to 0.2% applied to a vector, 0.4% as a
/// matrix. Every distance, bounding sphere and gate measures in the frame it gives an obstacle, a
/// held shape or a camera, so those take it from here. One that is unit to rounding is returned
/// as it is.
#[inline]
pub(crate) fn unit(rotation: Quat) -> Quat {
    let off = (rotation.length_squared() - 1.0).abs();
    if off > 1e-6 {
        rotation.normalize()
    } else {
        rotation
    }
}

/// An obstacle's rotation as a matrix that keeps lengths ([`unit`]).
#[inline]
pub(crate) fn frame(rotation: Quat) -> Mat3 {
    Mat3::from_quat(unit(rotation))
}

/// Must stay in sync with `obstacle_distance` in kernels.wgsl.
#[inline]
pub(crate) fn sdf_distance(r: Mat3, center: Vec3, grid: &SdfGrid, p: Vec3) -> (f32, Vec3) {
    let (d, g) = sdf::grid_distance(grid, r.transpose() * (p - center));
    (d, r * g)
}

/// Must stay in sync with `obstacle_distance` in kernels.wgsl.
#[inline]
pub(crate) fn capsule_distance(
    r: Mat3,
    center: Vec3,
    radius: f32,
    half_length: f32,
    p: Vec3,
) -> (f32, Vec3) {
    let lp = r.transpose() * (p - center);
    let v = lp - Vec3::new(0.0, 0.0, lp.z.clamp(-half_length, half_length));
    let len = v.length();
    (len - radius, r * if len > 1e-9 { v / len } else { Vec3::X })
}
