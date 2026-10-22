//! Robot model: a kinematic tree with collision spheres, built from a robot description file.

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail, ensure};
use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::description::{
    self, Curve, Geometry, JointDesc, JointType, LinkDesc, RobotDescription, Shape, TriMesh,
};
use crate::error::{self, Error, ensure_input, input};
use crate::spheres::{self, SphereGeometry, SphereOptions};
use crate::types::Pose;
use crate::world::Obstacle;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Transform {
    pub(crate) rot: Mat3,
    pub(crate) trans: Vec3,
}

impl Transform {
    pub(crate) const IDENTITY: Self = Self {
        rot: Mat3::IDENTITY,
        trans: Vec3::ZERO,
    };

    /// URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    pub(crate) fn from_xyz_rpy(xyz: [f64; 3], rpy: [f64; 3]) -> Self {
        let [r, p, y] = rpy.map(|v| v as f32);
        Self {
            rot: Mat3::from_rotation_z(y) * Mat3::from_rotation_y(p) * Mat3::from_rotation_x(r),
            trans: Vec3::from_array(xyz.map(|v| v as f32)),
        }
    }

    pub(crate) fn mul(&self, other: &Self) -> Self {
        Self {
            rot: self.rot * other.rot,
            trans: self.rot * other.trans + self.trans,
        }
    }
}

/// How a link moves relative to its parent. A moving joint is driven by actuated joint `dof`:
/// its value is `curve(q[dof])` (`q[dof]` itself unless it mimics another joint).
#[derive(Clone, Copy, Debug)]
pub(crate) enum JointKind {
    Fixed,
    Revolute {
        dof: usize,
        axis: Vec3,
        curve: Curve,
    },
    Prismatic {
        dof: usize,
        axis: Vec3,
        curve: Curve,
    },
}

impl JointKind {
    /// The actuated joint that drives this one, and how.
    #[inline]
    pub(crate) fn actuation(&self) -> Option<(usize, Curve)> {
        match *self {
            JointKind::Fixed => None,
            JointKind::Revolute { dof, curve, .. } | JointKind::Prismatic { dof, curve, .. } => {
                Some((dof, curve))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Link {
    pub(crate) name: String,
    /// Parent link index; parents always precede children. `None` for the root.
    pub(crate) parent: Option<usize>,
    /// Joint frame relative to the parent link frame.
    pub(crate) origin: Transform,
    pub(crate) joint: JointKind,
    /// The links at or above this one with a moving joint, root first.
    pub(crate) chain: Vec<usize>,
    /// The link's collision geometry in its frame (the visual geometry where spheres are fitted
    /// to that); mesh files are read only when something needs the meshes.
    pub(crate) shapes: Vec<Shape>,
}

/// A pair of links checked for self-collision: sphere pairs `first..first + count` of
/// `Robot::self_pairs`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LinkPair {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) first: u32,
    pub(crate) count: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CollisionSphere {
    pub(crate) link: usize,
    pub(crate) center: Vec3,
    pub(crate) radius: f32,
    /// Extra radius used only for self-collision.
    pub(crate) self_buffer: f32,
}

/// A robot's collision spheres and self-collision settings, keyed by link name. Generated from the
/// robot's geometry when not given; save it to commit and hand-tune it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollisionModel {
    /// Where the spheres came from (provenance, licence).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Per link: spheres as `[x, y, z, radius]` in the link frame.
    pub spheres: BTreeMap<String, Vec<[f32; 4]>>,
    /// Per link: extra radius used only for self-collision.
    #[serde(default)]
    pub self_collision_buffer: BTreeMap<String, f32>,
    /// Link pairs never checked against each other. Names of links the robot lacks are skipped.
    #[serde(default)]
    pub self_collision_ignore: BTreeMap<String, Vec<String>>,
}

impl CollisionModel {
    pub fn load(path: impl AsRef<Path>) -> error::Result<Self> {
        let path = path.as_ref();
        let load = |message: String| Error::Load {
            path: path.to_path_buf(),
            message,
        };
        let text = std::fs::read_to_string(path).map_err(|e| load(e.to_string()))?;
        serde_json::from_str(&text).map_err(|e| load(e.to_string()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> error::Result<()> {
        let path = path.as_ref();
        std::fs::write(path, serde_json::to_string_pretty(self)?)
            .map_err(|e| Error::Write(format!("writing {}: {e}", path.display())))
    }

    /// The model of a robot made with [`Robot::assemble`], from each part's own model and the
    /// prefix its names take: two arms fitted once share one model. Links of different parts are
    /// always checked against each other.
    pub fn assemble(parts: &[(&str, &CollisionModel)]) -> CollisionModel {
        let mut out = CollisionModel::default();
        let sources: Vec<String> = parts
            .iter()
            .filter_map(|(prefix, m)| m.source.as_ref().map(|s| format!("{prefix}: {s}")))
            .collect();
        out.source = (!sources.is_empty()).then(|| sources.join("; "));
        for (prefix, m) in parts {
            let named = |link: &String| format!("{prefix}{link}");
            out.spheres
                .extend(m.spheres.iter().map(|(link, s)| (named(link), s.clone())));
            out.self_collision_buffer.extend(
                m.self_collision_buffer
                    .iter()
                    .map(|(link, b)| (named(link), *b)),
            );
            out.self_collision_ignore.extend(
                m.self_collision_ignore
                    .iter()
                    .map(|(link, others)| (named(link), others.iter().map(named).collect())),
            );
        }
        out
    }
}

/// Limits replacing a joint's own: its description's, or the defaults in [`RobotOptions`] where
/// the description gives none. Each is optional, as in MoveIt's `joint_limits.yaml`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JointLimits {
    /// `[lower, upper]`; giving them makes a continuous joint an ordinary one.
    pub position: Option<[f32; 2]>,
    /// rad/s or m/s.
    pub velocity: Option<f32>,
    /// rad/s² or m/s².
    pub acceleration: Option<f32>,
    /// rad/s³ or m/s³.
    pub jerk: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct RobotOptions {
    /// Root of the planned tree; defaults to the description's root link.
    pub base_link: Option<String>,
    /// IK target frames, one per hand (end effector); defaults to the link with the most actuated
    /// joints above it. Set it for robots with grippers, and name each hand of a robot with
    /// several.
    pub ee_links: Vec<String>,
    /// Defaults to zero for every joint whose limits contain zero, else the middle of its range.
    pub default_q: Option<Vec<f32>>,
    /// Joints held at a value and removed from planning (mimics of a locked joint follow it).
    pub lock_joints: HashMap<String, f32>,
    /// Limits by joint name, replacing the description's; mimic joints may be named too. The
    /// scales below apply on top. Planar, ball and floating joints become joints named
    /// `<joint>_x`, `_y`, `_z` (translations, which need position limits from here) and
    /// `_theta` or `_rx`, `_ry`, `_rz` (rotations).
    pub joint_limits: HashMap<String, JointLimits>,
    /// Directories searched for `package://` packages: each is a package or holds packages.
    /// Directories above the description file are searched after these.
    pub package_dirs: Vec<PathBuf>,
    /// Collision spheres; fitted to the description's geometry with `spheres` when `None`.
    pub collision_model: Option<CollisionModel>,
    pub spheres: SphereOptions,
    /// A MoveIt SRDF's text, whose disabled collision pairs are added to the collision model's.
    pub srdf: Option<String>,
    /// OpenUSD variant selections, as (variant set, variant), used where the file authors none.
    pub variants: Vec<(String, String)>,
    /// Velocity limit (rad/s or m/s) for joints whose description gives none (MJCF has none).
    pub max_velocity: f32,
    /// Acceleration limit (rad/s² or m/s²) for joints whose description gives none.
    pub max_acceleration: f32,
    /// Jerk limit (rad/s³ or m/s³) for joints whose description gives none.
    pub max_jerk: f32,
    /// Scales every joint's velocity limit, its description's or `max_velocity`, in (0, 1], as
    /// MoveIt's velocity scaling factor does. Plans, timings and checks all keep within the scaled
    /// limits, so a robot slowed here replans mid-motion at its own pace. To slow every motion by
    /// `s`, as `Trajectory::new`'s speed scale does, scale velocity by `s`, acceleration by `s²`
    /// and jerk by `s³`.
    pub velocity_scale: f32,
    /// Scales every joint's acceleration limit, in (0, 1].
    pub acceleration_scale: f32,
    /// Scales every joint's jerk limit, in (0, 1].
    pub jerk_scale: f32,
}

impl Default for RobotOptions {
    fn default() -> Self {
        Self {
            base_link: None,
            ee_links: vec![],
            default_q: None,
            lock_joints: HashMap::new(),
            joint_limits: HashMap::new(),
            package_dirs: vec![],
            collision_model: None,
            spheres: SphereOptions::default(),
            srdf: None,
            variants: vec![],
            max_velocity: 2.0,
            max_acceleration: 5.0,
            max_jerk: 50.0,
            velocity_scale: 1.0,
            acceleration_scale: 1.0,
            jerk_scale: 1.0,
        }
    }
}

/// A robot's kinematic tree and collision-sphere model. Immutable once loaded.
#[derive(Clone, Debug)]
pub struct Robot {
    pub(crate) name: String,
    pub(crate) links: Vec<Link>,
    pub(crate) dof_names: Vec<String>,
    pub(crate) lower: Vec<f32>,
    pub(crate) upper: Vec<f32>,
    /// Joints that turn without limit; their values wrap every 2π.
    pub(crate) continuous: Vec<bool>,
    pub(crate) max_velocity: Vec<f32>,
    pub(crate) max_acceleration: Vec<f32>,
    pub(crate) max_jerk: Vec<f32>,
    pub(crate) spheres: Vec<CollisionSphere>,
    /// Sphere index pairs checked for self-collision, grouped by `self_link_pairs`.
    pub(crate) self_pairs: Vec<[u32; 2]>,
    pub(crate) self_link_pairs: Vec<LinkPair>,
    /// Per link: a sphere around all its collision spheres and their self buffers (link frame
    /// centre, radius).
    pub(crate) link_bounds: Vec<[f32; 4]>,
    /// Per link: its collision spheres, `first..first + count` of `spheres`.
    pub(crate) sphere_ranges: Vec<[u32; 2]>,
    /// Links whose frames are the IK target frames, one per hand.
    pub(crate) ee_links: Vec<usize>,
    pub(crate) default_q: Vec<f32>,
    collision_model: CollisionModel,
    /// Names of attached objects, each a fixed link at the end of `links`.
    attached: Vec<String>,
    /// The descriptions it was loaded from.
    parts: Vec<Part>,
}

/// A robot description placed in a robot made of several (see [`Robot::assemble`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub path: PathBuf,
    /// Prefixed to every link and joint name of the description, such as `left_`.
    pub prefix: String,
    /// Where the description's root link sits in the robot's frame.
    pub base: Pose,
}

/// An object the robot holds, attached the way MoveIt attaches one: it becomes a frame fixed to
/// `link` with spheres fitted to its shapes, collides with the world and the rest of the robot,
/// and never with `link` or `touch_links` (such as the fingers holding it).
#[derive(Clone, Debug)]
pub struct AttachedObject {
    /// The object's frame name, which must not name a link of the robot.
    pub name: String,
    pub link: String,
    /// The object's shapes, placed in `link`'s frame. Distance grids are not supported.
    pub shapes: Vec<Obstacle>,
    pub touch_links: Vec<String>,
    /// How spheres are fitted to the shapes; a held object rarely needs more than 8 to 16.
    pub spheres: SphereOptions,
}

/// World-frame kinematic state for one configuration, per link. Hot loops reuse one per thread
/// through [`Robot::fk_into`].
#[derive(Clone)]
pub(crate) struct Fk {
    pub(crate) rot: Vec<Mat3>,
    pub(crate) pos: Vec<Vec3>,
    /// World axis of each link's joint (zero for fixed joints).
    pub(crate) axis: Vec<Vec3>,
    /// Whether each link slides on a prismatic joint.
    pub(crate) slides: Vec<bool>,
    /// Each moving link's joint motion per unit of its actuated joint: its curve's slope there.
    /// Read through [`Fk::rate`].
    rate: Vec<f32>,
}

impl Fk {
    pub(crate) fn new(links: usize) -> Self {
        Self {
            rot: vec![Mat3::IDENTITY; links],
            pos: vec![Vec3::ZERO; links],
            axis: vec![Vec3::ZERO; links],
            slides: vec![false; links],
            rate: vec![0.0; links],
        }
    }

    /// Link `i`'s joint motion per unit of its actuated joint: the multiplier of a linear curve,
    /// the slope at this configuration for others (only those are recorded).
    #[inline]
    pub(crate) fn rate(&self, i: usize, curve: &Curve) -> f32 {
        if curve.is_linear() {
            curve.0[1]
        } else {
            self.rate[i]
        }
    }

    /// Derivative of a point rigidly attached downstream of link `i`'s joint, per unit of joint
    /// motion.
    #[inline]
    pub(crate) fn dpoint(&self, i: usize, p: Vec3) -> Vec3 {
        if self.slides[i] {
            self.axis[i]
        } else {
            self.axis[i].cross(p - self.pos[i])
        }
    }
}

impl Robot {
    /// Loads a robot from URDF, MJCF (`.xml`, `.mjcf`) or OpenUSD (`.usd*`, feature `usd`).
    pub fn load(path: impl AsRef<Path>, o: &RobotOptions) -> error::Result<Self> {
        let path = path.as_ref();
        let part = Part {
            path: path.to_path_buf(),
            prefix: String::new(),
            base: Pose::IDENTITY,
        };
        description::load_robot(path, &o.package_dirs, &o.variants)
            .and_then(|description| Self::from_description(&description, o))
            .map(|robot| Robot {
                parts: vec![part],
                ..robot
            })
            .map_err(|e| Error::Load {
                path: path.to_path_buf(),
                message: format!("{e:#}"),
            })
    }

    /// Loads a robot from a URDF's text, as ROS publishes it on `/robot_description`. Its mesh
    /// paths must be `package://` (found through [`RobotOptions::package_dirs`]), `file://` or
    /// absolute.
    pub fn from_urdf(urdf: &str, o: &RobotOptions) -> error::Result<Self> {
        crate::urdf::from_text(urdf, None, &o.package_dirs)
            .and_then(|description| Self::from_description(&description, o))
            .map_err(|e| Error::Input(format!("URDF: {e:#}")))
    }

    /// A robot made of several descriptions, such as two arms at a table: each part's link and
    /// joint names get its prefix, its root link is fixed at its base, and the parts hang from one
    /// root link, `world`. The options name links and joints with their prefixes; give each arm's
    /// hand in `ee_links`.
    pub fn assemble(parts: &[Part], o: &RobotOptions) -> error::Result<Self> {
        let load_error = |path: &Path, e: anyhow::Error| Error::Load {
            path: path.to_path_buf(),
            message: format!("{e:#}"),
        };
        let mut descriptions = vec![];
        for part in parts {
            let description = description::load_robot(&part.path, &o.package_dirs, &o.variants)
                .map_err(|e| load_error(&part.path, e))?;
            let base = Transform {
                rot: Mat3::from_quat(part.base.rotation),
                trans: part.base.position,
            };
            descriptions.push((description, part.prefix.clone(), base));
        }
        let first = parts.first().map_or_else(PathBuf::new, |p| p.path.clone());
        description::assemble(descriptions)
            .and_then(|description| Self::from_description(&description, o))
            .map(|robot| Robot {
                parts: parts.to_vec(),
                ..robot
            })
            .map_err(|e| load_error(&first, e))
    }

    /// The descriptions the robot was loaded from: one for [`Robot::load`], its parts for
    /// [`Robot::assemble`].
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    fn from_description(desc: &RobotDescription, o: &RobotOptions) -> Result<Self> {
        let closed;
        let desc = if desc.loops.is_empty() {
            desc
        } else {
            // Loops are solved on the tree they leave open, with every joint planned.
            let open = RobotOptions {
                lock_joints: HashMap::new(),
                default_q: None,
                ..o.clone()
            };
            closed = crate::loops::close_loops(desc, &kinematics(desc, &open)?)?;
            &closed
        };
        let mut robot = kinematics(desc, o)?;
        for link in &mut robot.links {
            if let Some(l) = desc.links.iter().find(|l| l.name == link.name) {
                link.shapes = match o.spheres.geometry {
                    SphereGeometry::Collision => l.collision.clone(),
                    SphereGeometry::Visual => l.visual.clone(),
                };
            }
        }
        let mut model = match &o.collision_model {
            Some(model) => model.clone(),
            None => robot.fit_collision_model(&o.spheres)?,
        };
        if let Some(srdf) = &o.srdf {
            for (a, b) in crate::urdf::srdf_disabled_pairs(srdf)? {
                model.self_collision_ignore.entry(a).or_default().push(b);
            }
        }
        robot.set_collision_model(model)?;
        Ok(robot)
    }

    /// Fits spheres to every link's geometry, then finds the link pairs not worth checking.
    fn fit_collision_model(&mut self, o: &SphereOptions) -> Result<CollisionModel> {
        let meshes: Vec<Vec<TriMesh>> = self
            .links
            .iter()
            .map(|link| link.shapes.iter().map(|s| s.mesh()).collect())
            .collect::<Result<_>>()?;
        o.check().map_err(anyhow::Error::msg)?;
        let with_geometry = meshes.iter().filter(|m| !m.is_empty()).count();
        ensure!(
            o.budget >= with_geometry,
            "a budget of {} spheres cannot cover {with_geometry} links with geometry",
            o.budget
        );
        let (fitted, overhang) = spheres::fit(&meshes, o, true);
        let source = format!(
            "fitted by batchplan: spheres reach up to {:.1} mm beyond the geometry",
            overhang * 1e3
        );
        let mut model = CollisionModel {
            source: Some(source),
            ..Default::default()
        };
        for (link, fitted) in self.links.iter().zip(fitted) {
            if !fitted.is_empty() {
                model.spheres.insert(link.name.clone(), fitted);
            }
        }
        self.set_collision_model(model.clone())?;
        model.self_collision_ignore = spheres::ignore_pairs(self, &meshes, o);
        Ok(model)
    }

    fn set_collision_model(&mut self, model: CollisionModel) -> Result<()> {
        let index: HashMap<&str, usize> = self
            .links
            .iter()
            .enumerate()
            .map(|(i, l)| (l.name.as_str(), i))
            .collect();
        if let Some(name) = model
            .spheres
            .keys()
            .find(|n| !index.contains_key(n.as_str()))
        {
            bail!("collision spheres given for link '{name}', which is not part of the robot");
        }
        let mut spheres = vec![];
        for (li, link) in self.links.iter().enumerate() {
            let buffer = model
                .self_collision_buffer
                .get(&link.name)
                .copied()
                .unwrap_or(0.0);
            for s in model.spheres.get(&link.name).into_iter().flatten() {
                spheres.push(CollisionSphere {
                    link: li,
                    center: Vec3::new(s[0], s[1], s[2]),
                    radius: s[3],
                    self_buffer: buffer,
                });
            }
        }
        let mut ignore = HashSet::new();
        for (a, others) in &model.self_collision_ignore {
            for b in others {
                if let (Some(&ia), Some(&ib)) = (index.get(a.as_str()), index.get(b.as_str())) {
                    ignore.insert((ia.min(ib), ia.max(ib)));
                }
            }
        }
        // Sphere pairs grouped by link pair, so a pair of links whose bounding spheres are apart
        // skips all of them.
        let held = &spheres;
        let on = |link: usize| (0..held.len()).filter(move |&s| held[s].link == link);
        let (mut self_pairs, mut self_link_pairs) = (vec![], vec![]);
        for la in 0..self.links.len() {
            for lb in la + 1..self.links.len() {
                if ignore.contains(&(la, lb)) {
                    continue;
                }
                let first = self_pairs.len();
                for a in on(la) {
                    self_pairs.extend(on(lb).map(|b| [a as u32, b as u32]));
                }
                if self_pairs.len() > first {
                    let count = (self_pairs.len() - first) as u32;
                    self_link_pairs.push(LinkPair {
                        a: la as u32,
                        b: lb as u32,
                        first: first as u32,
                        count,
                    });
                }
            }
        }
        self.link_bounds = (0..self.links.len())
            .map(|link| {
                let own: Vec<&CollisionSphere> = on(link).map(|s| &held[s]).collect();
                let center = own.iter().map(|s| s.center).sum::<Vec3>() / own.len().max(1) as f32;
                let radius = own
                    .iter()
                    .map(|s| (s.center - center).length() + s.radius + s.self_buffer)
                    .fold(0.0, f32::max);
                [center.x, center.y, center.z, radius]
            })
            .collect();
        self.sphere_ranges = (0..self.links.len())
            .map(|link| [on(link).next().unwrap_or(0) as u32, on(link).count() as u32])
            .collect();
        self.spheres = spheres;
        self.self_pairs = self_pairs;
        self.self_link_pairs = self_link_pairs;
        self.collision_model = model;
        Ok(())
    }

    /// This robot holding `object`. Errors if the link is unknown or an attached object, the name
    /// is taken, or the robot would exceed the kernel limits.
    pub fn attach(&self, object: &AttachedObject) -> error::Result<Robot> {
        let link = |name: &str| self.links.iter().position(|l| l.name == name);
        let parent = link(&object.link).ok_or_else(|| input!("unknown link '{}'", object.link))?;
        ensure_input!(
            !self.attached.contains(&object.link),
            "objects attach to robot links, not to '{}'",
            object.link
        );
        ensure_input!(
            link(&object.name).is_none(),
            "'{}' already names a link",
            object.name
        );
        if let Some(t) = object.touch_links.iter().find(|t| link(t).is_none()) {
            return Err(input!("unknown touch link '{t}'"));
        }
        object.spheres.check().map_err(Error::Input)?;
        let shapes = object
            .shapes
            .iter()
            .map(attached_shape)
            .collect::<error::Result<Vec<Shape>>>()?;
        ensure_input!(!shapes.is_empty(), "'{}' has no shapes", object.name);
        // Boxes and spheres are covered directly; other shapes are fitted. Attaching happens inside
        // control processes: fit on the caller's rayon pool if it has one, otherwise on the calling
        // thread alone, never on the global pool. (A one-thread pool on the calling thread would
        // leave it registered with rayon for good, and every parallel loop it started afterwards
        // would run on that one thread.)
        let fitted = match spheres::cover_primitives(&object.shapes, &object.spheres) {
            Some(covered) => covered,
            None => {
                let meshes = shapes
                    .iter()
                    .map(|shape| shape.mesh().map_err(|e| input!("{e:#}")))
                    .collect::<error::Result<Vec<TriMesh>>>()?;
                let (fitted, _) = spheres::fit(
                    &[meshes],
                    &object.spheres,
                    rayon::current_thread_index().is_some(),
                );
                fitted.into_iter().flatten().collect()
            }
        };
        let mut robot = self.clone();
        let chain = robot.links[parent].chain.clone();
        robot.links.push(Link {
            name: object.name.clone(),
            parent: Some(parent),
            origin: Transform::IDENTITY,
            joint: JointKind::Fixed,
            chain,
            shapes,
        });
        robot.attached.push(object.name.clone());
        let mut model = self.collision_model.clone();
        model.spheres.insert(object.name.clone(), fitted);
        let ignored = model
            .self_collision_ignore
            .entry(object.name.clone())
            .or_default();
        ignored.push(object.link.clone());
        ignored.extend(object.touch_links.iter().cloned());
        robot
            .set_collision_model(model)
            .map_err(|e| input!("{e:#}"))?;
        Ok(robot)
    }

    /// This robot without the attached object `name`.
    pub fn detach(&self, name: &str) -> error::Result<Robot> {
        let index = self
            .attached
            .iter()
            .position(|a| a == name)
            .ok_or_else(|| input!("nothing named '{name}' is attached"))?;
        let mut robot = self.clone();
        robot.attached.remove(index);
        // Attached objects are leaves after every robot link, so no other link refers to this one.
        robot.links.retain(|l| l.name != name);
        let mut model = self.collision_model.clone();
        model.spheres.remove(name);
        model.self_collision_ignore.remove(name);
        robot
            .set_collision_model(model)
            .map_err(|e| input!("{e:#}"))?;
        Ok(robot)
    }

    /// The robot name from its description.
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    pub fn dof(&self) -> usize {
        self.dof_names.len()
    }

    pub fn joint_names(&self) -> &[String] {
        &self.dof_names
    }

    pub fn lower(&self) -> &[f32] {
        &self.lower
    }

    pub fn upper(&self) -> &[f32] {
        &self.upper
    }

    /// Which joints are continuous: they turn without limit and their values wrap every 2π. For
    /// them, [`Robot::lower`] and [`Robot::upper`] give the turn (−π to π) that seeds and samples
    /// come from, not limits.
    pub fn continuous(&self) -> &[bool] {
        &self.continuous
    }

    /// Joint `j`'s position limits; unbounded for continuous joints.
    #[inline]
    pub(crate) fn bounds(&self, j: usize) -> (f32, f32) {
        if self.continuous[j] {
            (f32::NEG_INFINITY, f32::INFINITY)
        } else {
            (self.lower[j], self.upper[j])
        }
    }

    /// `q` with each continuous joint turned by whole turns to lie within half a turn of `from`.
    pub(crate) fn turn_toward(&self, from: &[f32], q: &mut [f32]) {
        use std::f32::consts::{PI, TAU};
        for (j, v) in q
            .iter_mut()
            .enumerate()
            .filter(|&(j, _)| self.continuous[j])
        {
            *v = from[j] + (*v - from[j] + PI).rem_euclid(TAU) - PI;
        }
    }

    /// The configurations equivalent to `goal` that a path from `start` may end at, nearest first:
    /// `goal` with continuous joints turned the short way (see [`Robot::turn_toward`]) and each
    /// joint whose range spans more than a turn at its value nearest `start`, then, for each such
    /// joint, the same with it a turn further either way where its limits allow. All are one pose.
    pub(crate) fn goal_variants(&self, start: &[f32], goal: &[f32]) -> Vec<Vec<f32>> {
        use std::f32::consts::TAU;
        let mut base = goal.to_vec();
        self.turn_toward(start, &mut base);
        // Each wide joint's values, nearest the start first.
        let turns: Vec<Vec<f32>> = (0..goal.len())
            .map(|j| {
                if self.continuous[j] {
                    return vec![base[j]];
                }
                let mut values: Vec<f32> = (-2..=2)
                    .map(|k| goal[j] + k as f32 * TAU)
                    .filter(|v| (self.lower[j]..=self.upper[j]).contains(v))
                    .collect();
                values.sort_by(|a, b| (a - start[j]).abs().total_cmp(&(b - start[j]).abs()));
                values
            })
            .collect();
        for (j, t) in turns.iter().enumerate() {
            base[j] = t[0];
        }
        let mut variants = vec![base.clone()];
        let distance = |q: &[f32]| {
            q.iter()
                .zip(start)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
        };
        let mut flips = vec![];
        for (j, t) in turns.iter().enumerate() {
            for &v in &t[1..] {
                let mut q = base.clone();
                q[j] = v;
                flips.push(q);
            }
        }
        flips.sort_by(|a, b| distance(a).total_cmp(&distance(b)));
        variants.extend(flips);
        variants
    }

    /// Whether `q` is within every joint's limits.
    pub(crate) fn within(&self, q: &[f32]) -> bool {
        q.iter().enumerate().all(|(j, &v)| {
            let (lo, hi) = self.bounds(j);
            v >= lo && v <= hi
        })
    }

    pub fn max_velocity(&self) -> &[f32] {
        &self.max_velocity
    }

    pub fn max_acceleration(&self) -> &[f32] {
        &self.max_acceleration
    }

    pub fn max_jerk(&self) -> &[f32] {
        &self.max_jerk
    }

    pub fn default_q(&self) -> &[f32] {
        &self.default_q
    }

    /// The collision spheres in use, generated or as given.
    pub fn collision_model(&self) -> &CollisionModel {
        &self.collision_model
    }

    /// The number of hands: IK target frames (`RobotOptions::ee_links`).
    pub fn hands(&self) -> usize {
        self.ee_links.len()
    }

    /// World pose of hand `hand`'s IK target frame at configuration `q`.
    pub fn ee_pose(&self, q: &[f32], hand: usize) -> Pose {
        self.pose_of(&self.fk(q), self.ee_links[hand])
    }

    /// The prefix of the part hand `hand`'s frame belongs to (`left_`), on a robot made of parts.
    pub(crate) fn hand_prefix(&self, hand: usize) -> Option<&str> {
        let link = &self.links[self.ee_links[hand]].name;
        self.parts
            .iter()
            .map(|p| p.prefix.as_str())
            .find(|prefix| !prefix.is_empty() && link.starts_with(prefix))
    }

    /// The joints that move hand `hand`, root first: the actuated joints of its chain (a mimic
    /// joint counts as its leader).
    pub fn hand_joints(&self, hand: usize) -> Vec<usize> {
        let mut joints = vec![];
        for (_, dof) in self.chain(self.ee_links[hand]) {
            if !joints.contains(&dof) {
                joints.push(dof);
            }
        }
        joints
    }

    /// World pose of the named link at configuration `q`.
    pub fn link_pose(&self, q: &[f32], link: &str) -> Option<Pose> {
        let i = self.links.iter().position(|l| l.name == link)?;
        Some(self.pose_of(&self.fk(q), i))
    }

    fn pose_of(&self, fk: &Fk, link: usize) -> Pose {
        Pose::new(fk.pos[link], Quat::from_mat3(&fk.rot[link]))
    }

    /// The moving joints at or above `link`, root first: (link index, actuated joint).
    #[inline]
    pub(crate) fn chain(&self, link: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.links[link].chain.iter().map(|&i| {
            let (dof, _) = self.links[i]
                .joint
                .actuation()
                .expect("chain links have moving joints");
            (i, dof)
        })
    }

    pub(crate) fn fk(&self, q: &[f32]) -> Fk {
        let mut fk = Fk::new(self.links.len());
        self.fk_into(q, &mut fk);
        fk
    }

    /// Forward kinematics into `fk`, which must have one entry per link.
    // Hot in the CPU backend's inner loops: `#[inline]` keeps it inlinable whichever codegen unit
    // the caller lands in (a 20% swing on the CPU benchmark otherwise).
    #[inline]
    pub(crate) fn fk_into(&self, q: &[f32], fk: &mut Fk) {
        let (rot, pos, axes) = (&mut fk.rot[..], &mut fk.pos[..], &mut fk.axis[..]);
        let (slides, rates) = (&mut fk.slides[..], &mut fk.rate[..]);
        for (i, link) in self.links.iter().enumerate() {
            let (prot, ppos) = link
                .parent
                .map_or((Mat3::IDENTITY, Vec3::ZERO), |p| (rot[p], pos[p]));
            let jrot = prot * link.origin.rot;
            let jpos = prot * link.origin.trans + ppos;
            (rot[i], pos[i], axes[i], slides[i], rates[i]) = match link.joint {
                JointKind::Fixed => (jrot, jpos, Vec3::ZERO, false, 0.0),
                JointKind::Revolute { dof, axis, curve } => {
                    let (angle, rate) = curve.value_and_slope(q[dof]);
                    (
                        jrot * Mat3::from_axis_angle(axis, angle),
                        jpos,
                        jrot * axis,
                        false,
                        rate,
                    )
                }
                JointKind::Prismatic { dof, axis, curve } => {
                    let a = jrot * axis;
                    let (offset, rate) = curve.value_and_slope(q[dof]);
                    (jrot, jpos + a * offset, a, true, rate)
                }
            };
        }
    }
}

/// `desc` with planar, ball and floating joints written as chains of prismatic and revolute joints
/// on massless links, so kinematics and both devices see only one-axis joints:
/// - planar: `<joint>_x` and `<joint>_y` slide in the plane (`_x` along the joint frame's x axis
///   projected into it), `<joint>_theta` turns about its normal;
/// - ball: `<joint>_rx`, `<joint>_ry`, `<joint>_rz` turn about the joint frame's axes in that order
///   (intrinsic x-y-z angles), which reach every rotation with `_ry` in [-π/2, π/2] and lose a
///   direction of motion (gimbal lock) at its ends;
/// - floating: `<joint>_x`, `<joint>_y`, `<joint>_z` slide along the joint frame's axes, then a
///   ball.
///
/// Translations get no limits here: they come from `RobotOptions::joint_limits`.
fn expand_compound_joints(desc: &RobotDescription) -> Result<RobotDescription> {
    use std::f32::consts::{FRAC_PI_2, PI};
    let mut out = RobotDescription {
        name: desc.name.clone(),
        links: desc.links.clone(),
        joints: vec![],
        loops: desc.loops.clone(),
    };
    for j in &desc.joints {
        let ball = |range: f32| {
            let r = if range.is_finite() && range > 0.0 {
                range.min(PI)
            } else {
                PI
            };
            [
                ("rx", Vec3::X, r),
                ("ry", Vec3::Y, r.min(FRAC_PI_2)),
                ("rz", Vec3::Z, r),
            ]
            .map(|(name, axis, r)| (name, JointType::Revolute, axis, -r, r))
        };
        let slide = |name, axis| (name, JointType::Prismatic, axis, f32::NAN, f32::NAN);
        let parts: Vec<(&str, JointType, Vec3, f32, f32)> = match j.kind {
            JointType::Planar => {
                // `_x` follows the joint frame's x axis, projected into the plane (y if x is
                // normal).
                let inplane = |v: Vec3| v - j.axis * j.axis.dot(v);
                let u = inplane(Vec3::X)
                    .try_normalize()
                    .unwrap_or_else(|| inplane(Vec3::Y).normalize());
                vec![
                    slide("x", u),
                    slide("y", j.axis.cross(u)),
                    ("theta", JointType::Continuous, j.axis, 0.0, 0.0),
                ]
            }
            JointType::Ball => ball(j.upper).to_vec(),
            JointType::Floating => [
                slide("x", Vec3::X),
                slide("y", Vec3::Y),
                slide("z", Vec3::Z),
            ]
            .into_iter()
            .chain(ball(PI))
            .collect(),
            _ => {
                out.joints.push(j.clone());
                continue;
            }
        };
        ensure!(
            j.mimic.is_none(),
            "{:?} joint '{}' cannot mimic another joint",
            j.kind,
            j.name
        );
        let mut parent = j.parent.clone();
        for (k, &(suffix, kind, axis, lower, upper)) in parts.iter().enumerate() {
            let name = format!("{}_{suffix}", j.name);
            let child = if k + 1 == parts.len() {
                j.child.clone()
            } else {
                format!("{}/{name}", j.child)
            };
            if child != j.child {
                out.links.push(LinkDesc {
                    name: child.clone(),
                    ..Default::default()
                });
            }
            let origin = if k == 0 {
                j.origin
            } else {
                Transform::IDENTITY
            };
            out.joints.push(JointDesc {
                name,
                kind,
                parent,
                child: child.clone(),
                origin,
                axis,
                lower,
                upper,
                ..j.clone()
            });
            parent = child;
        }
    }
    Ok(out)
}

/// The part of `range` (a leader's) over which `curve` stays within `limits` (a mimic's): exact for
/// linear curves; for others, the longest such stretch among 1,024 samples. `None` if there is
/// none.
fn within(curve: &Curve, range: (f32, f32), limits: (f32, f32)) -> Option<(f32, f32)> {
    if curve.is_linear() {
        let (m, off) = (curve.0[1], curve.0[0]);
        let (a, b) = ((limits.0 - off) / m, (limits.1 - off) / m);
        return Some((range.0.max(a.min(b)), range.1.min(a.max(b))));
    }
    const N: usize = 1024;
    let at = |k: usize| range.0 + (range.1 - range.0) * k as f32 / N as f32;
    let ok = |k: usize| (limits.0..=limits.1).contains(&curve.value(at(k)));
    let (mut best, mut start) = (None::<(usize, usize)>, None);
    for k in 0..=N {
        match (ok(k), start) {
            (true, None) => start = Some(k),
            (false, Some(s)) => {
                if best.is_none_or(|(a, b)| k - 1 - s > b - a) {
                    best = Some((s, k - 1));
                }
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start
        && best.is_none_or(|(a, b)| N - s > b - a)
    {
        best = Some((s, N));
    }
    best.map(|(a, b)| (at(a), at(b)))
}

/// The kinematic tree below the base link, with actuated joints numbered in breadth-first order.
fn kinematics(desc: &RobotDescription, o: &RobotOptions) -> Result<Robot> {
    use anyhow::Context;

    let expanded = expand_compound_joints(desc)?;
    let desc = &expanded;
    for (name, limits) in &o.joint_limits {
        ensure!(
            desc.joints.iter().any(|j| &j.name == name),
            "joint_limits names '{name}', which is not a joint of the robot"
        );
        if let Some([lo, hi]) = limits.position {
            ensure!(
                lo.is_finite() && hi.is_finite() && lo <= hi,
                "joint_limits gives '{name}' the range [{lo}, {hi}]"
            );
        }
        for (what, limit) in [
            ("velocity", limits.velocity),
            ("acceleration", limits.acceleration),
            ("jerk", limits.jerk),
        ] {
            if let Some(limit) = limit {
                ensure!(
                    limit.is_finite() && limit > 0.0,
                    "joint_limits gives '{name}' a {what} limit of {limit}"
                );
            }
        }
    }
    let given = |joint: &JointDesc| o.joint_limits.get(&joint.name).copied().unwrap_or_default();
    let joint_index: HashMap<&str, usize> = desc
        .joints
        .iter()
        .enumerate()
        .map(|(i, j)| (j.name.as_str(), i))
        .collect();
    let base = match &o.base_link {
        Some(b) => {
            ensure!(
                desc.links.iter().any(|l| &l.name == b),
                "unknown base_link '{b}'"
            );
            b.clone()
        }
        None => {
            let children: HashSet<&str> = desc.joints.iter().map(|j| j.child.as_str()).collect();
            let roots: Vec<&str> = desc
                .links
                .iter()
                .map(|l| l.name.as_str())
                .filter(|n| !children.contains(n))
                .collect();
            match roots[..] {
                [root] => root.to_string(),
                _ => bail!(
                    "the description has {} root links {roots:?}; set RobotOptions::base_link",
                    roots.len()
                ),
            }
        }
    };
    for name in o.lock_joints.keys() {
        let j = joint_index
            .get(name.as_str())
            .with_context(|| format!("cannot lock unknown joint '{name}'"))?;
        ensure!(
            desc.joints[*j].kind != JointType::Fixed,
            "cannot lock fixed joint '{name}'"
        );
    }
    // Each moving joint resolved to the joint that drives it, and the curve from that joint's value
    // to its own.
    let driver = |start: usize| -> Result<(usize, Curve)> {
        let (mut j, mut curve) = (start, Curve::IDENTITY);
        for _ in 0..desc.joints.len() {
            let Some(mimic) = &desc.joints[j].mimic else {
                return Ok((j, curve));
            };
            let leader = *joint_index.get(mimic.joint.as_str()).with_context(|| {
                format!(
                    "joint '{}' mimics unknown joint '{}'",
                    desc.joints[j].name, mimic.joint
                )
            })?;
            curve = curve.after(&mimic.curve).with_context(|| {
                format!(
                    "joint '{}' follows its driver by a polynomial of degree above four",
                    desc.joints[start].name
                )
            })?;
            j = leader;
        }
        bail!(
            "mimic joints form a cycle through '{}'",
            desc.joints[start].name
        )
    };

    // Breadth-first so parents precede children.
    let mut order = vec![];
    let mut queue = VecDeque::from([base.clone()]);
    let mut seen = HashSet::from([base.clone()]);
    while let Some(link) = queue.pop_front() {
        for (j, joint) in desc
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.parent == link)
        {
            ensure!(
                seen.insert(joint.child.clone()),
                "link '{}' is reached twice; the description is not a tree",
                joint.child
            );
            order.push(j);
            queue.push_back(joint.child.clone());
        }
    }
    let moving = |j: usize| desc.joints[j].kind != JointType::Fixed;
    let mut dof_of: HashMap<usize, usize> = HashMap::new();
    let (mut dof_names, mut lower, mut upper, mut continuous) = (vec![], vec![], vec![], vec![]);
    let (mut max_velocity, mut max_acceleration, mut max_jerk) = (vec![], vec![], vec![]);
    let or_default = |v: f32, default: f32| if v.is_finite() { v } else { default };
    for &j in order
        .iter()
        .filter(|&&j| moving(j) && desc.joints[j].mimic.is_none())
    {
        let joint = &desc.joints[j];
        if o.lock_joints.contains_key(&joint.name) {
            continue;
        }
        dof_of.insert(j, dof_names.len());
        dof_names.push(joint.name.clone());
        let limits = given(joint);
        let (lo, hi) = match (limits.position, joint.kind) {
            (Some([lo, hi]), _) => (lo, hi),
            (None, JointType::Continuous) => (-std::f32::consts::PI, std::f32::consts::PI),
            (None, _) => (joint.lower, joint.upper),
        };
        continuous.push(joint.kind == JointType::Continuous && limits.position.is_none());
        ensure!(
            lo.is_finite() && hi.is_finite(),
            "joint '{}' has no position limits; give them in RobotOptions::joint_limits",
            joint.name
        );
        lower.push(lo);
        upper.push(hi);
        max_velocity.push(
            limits
                .velocity
                .unwrap_or(or_default(joint.max_velocity, o.max_velocity)),
        );
        max_acceleration.push(
            limits
                .acceleration
                .unwrap_or(or_default(joint.max_acceleration, o.max_acceleration)),
        );
        max_jerk.push(
            limits
                .jerk
                .unwrap_or(or_default(joint.max_jerk, o.max_jerk)),
        );
    }
    ensure!(
        !dof_names.is_empty(),
        "the robot has no actuated joints to plan with"
    );

    let root = Link {
        name: base.clone(),
        parent: None,
        origin: Transform::IDENTITY,
        joint: JointKind::Fixed,
        chain: vec![],
        shapes: vec![],
    };
    let mut links = vec![root];
    let mut link_index: HashMap<String, usize> = HashMap::from([(base, 0)]);
    for &j in &order {
        let joint = &desc.joints[j];
        let parent = link_index[&joint.parent];
        let mut origin = joint.origin;
        let kind = if moving(j) {
            let (leader, curve) = driver(j)?;
            let prismatic = joint.kind == JointType::Prismatic;
            if let Some(&locked) = o.lock_joints.get(&desc.joints[leader].name) {
                let value = curve.value(locked);
                let motion = if prismatic {
                    Transform {
                        rot: Mat3::IDENTITY,
                        trans: joint.axis * value,
                    }
                } else {
                    Transform {
                        rot: Mat3::from_axis_angle(joint.axis, value),
                        trans: Vec3::ZERO,
                    }
                };
                origin = origin.mul(&motion);
                JointKind::Fixed
            } else {
                let dof = *dof_of.get(&leader).with_context(|| {
                    format!(
                        "joint '{}' mimics '{}', which is not below the base link",
                        joint.name, desc.joints[leader].name
                    )
                })?;
                if leader != j {
                    let limits = given(joint);
                    // The leader's range keeps the mimic joint within its own limits too.
                    if joint.kind != JointType::Continuous || limits.position.is_some() {
                        let own = limits
                            .position
                            .map_or((joint.lower, joint.upper), |[lo, hi]| (lo, hi));
                        (lower[dof], upper[dof]) = within(&curve, (lower[dof], upper[dof]), own)
                            .with_context(|| {
                                format!(
                                    "joint '{}' leaves its limits for every value of '{}'",
                                    joint.name, desc.joints[leader].name
                                )
                            })?;
                    }
                    // Its derivatives scale by the curve's steepest slope over the leader's range
                    // (exactly, for linear mimics; acceleration ignores the curvature term).
                    let steepest = (0..=64)
                        .map(|k| {
                            curve
                                .slope(lower[dof] + (upper[dof] - lower[dof]) * k as f32 / 64.0)
                                .abs()
                        })
                        .fold(0.0f32, f32::max);
                    max_velocity[dof] = max_velocity[dof]
                        .min(limits.velocity.unwrap_or(joint.max_velocity) / steepest);
                    max_acceleration[dof] = max_acceleration[dof]
                        .min(limits.acceleration.unwrap_or(joint.max_acceleration) / steepest);
                    max_jerk[dof] =
                        max_jerk[dof].min(limits.jerk.unwrap_or(joint.max_jerk) / steepest);
                }
                let axis = joint.axis;
                if prismatic {
                    JointKind::Prismatic { dof, axis, curve }
                } else {
                    JointKind::Revolute { dof, axis, curve }
                }
            }
        } else {
            JointKind::Fixed
        };
        let mut chain = links[parent].chain.clone();
        if kind.actuation().is_some() {
            chain.push(links.len());
        }
        link_index.insert(joint.child.clone(), links.len());
        links.push(Link {
            name: joint.child.clone(),
            parent: Some(parent),
            origin,
            joint: kind,
            chain,
            shapes: vec![],
        });
    }
    if let Some(dof) = (0..dof_names.len())
        .find(|&d| lower[d] > upper[d] || !(lower[d].is_finite() && upper[d].is_finite()))
    {
        bail!(
            "joint '{}' has an empty range once its mimic joints' limits apply",
            dof_names[dof]
        );
    }

    let ee_links = if o.ee_links.is_empty() {
        let mut depth = vec![0usize; links.len()];
        for i in 1..links.len() {
            let p = links[i].parent.expect("non-root links have parents");
            depth[i] = depth[p] + usize::from(links[i].joint.actuation().is_some());
        }
        vec![
            (0..links.len())
                .max_by_key(|&i| (depth[i], Reverse(i)))
                .expect("the base link"),
        ]
    } else {
        o.ee_links
            .iter()
            .map(|name| {
                link_index
                    .get(name)
                    .copied()
                    .with_context(|| format!("unknown ee_link '{name}'"))
            })
            .collect::<anyhow::Result<_>>()?
    };
    let default_q = match &o.default_q {
        Some(q) => {
            ensure!(
                q.len() == dof_names.len(),
                "default_q has {} values but the robot has {} actuated joints",
                q.len(),
                dof_names.len()
            );
            if let Some(j) = (0..q.len()).find(|&j| !(lower[j]..=upper[j]).contains(&q[j])) {
                bail!(
                    "default_q puts joint '{}' at {} outside [{}, {}]",
                    dof_names[j],
                    q[j],
                    lower[j],
                    upper[j]
                );
            }
            q.clone()
        }
        None => (0..dof_names.len())
            .map(|j| {
                if (lower[j]..=upper[j]).contains(&0.0) {
                    0.0
                } else {
                    0.5 * (lower[j] + upper[j])
                }
            })
            .collect(),
    };
    for (limits, scale, what) in [
        (&mut max_velocity, o.velocity_scale, "velocity_scale"),
        (
            &mut max_acceleration,
            o.acceleration_scale,
            "acceleration_scale",
        ),
        (&mut max_jerk, o.jerk_scale, "jerk_scale"),
    ] {
        ensure!(
            scale > 0.0 && scale <= 1.0,
            "RobotOptions::{what} must be in (0, 1], got {scale}"
        );
        limits.iter_mut().for_each(|l| *l *= scale);
    }
    Ok(Robot {
        name: desc.name.clone(),
        links,
        dof_names,
        lower,
        upper,
        continuous,
        max_velocity,
        max_acceleration,
        max_jerk,
        spheres: vec![],
        self_pairs: vec![],
        self_link_pairs: vec![],
        link_bounds: vec![],
        sphere_ranges: vec![],
        ee_links,
        default_q,
        collision_model: CollisionModel::default(),
        attached: vec![],
        parts: vec![],
    })
}

/// An attached object's shape, placed in its link's frame.
fn attached_shape(shape: &Obstacle) -> error::Result<Shape> {
    use crate::world::Obstacle;
    let placed = |center: Vec3, rotation: Quat, geometry: Geometry| Shape {
        origin: Transform {
            rot: Mat3::from_quat(rotation),
            trans: center,
        },
        geometry,
    };
    Ok(match *shape {
        Obstacle::Cuboid {
            center,
            half_extents,
            rotation,
        } => placed(center, rotation, Geometry::Box { half: half_extents }),
        Obstacle::Sphere { center, radius } => {
            placed(center, Quat::IDENTITY, Geometry::Sphere { radius })
        }
        Obstacle::Cylinder {
            center,
            rotation,
            radius,
            half_height,
        } => placed(
            center,
            rotation,
            Geometry::Cylinder {
                radius,
                half_length: half_height,
            },
        ),
        Obstacle::Capsule {
            center,
            rotation,
            radius,
            half_length,
        } => placed(
            center,
            rotation,
            Geometry::Capsule {
                radius,
                half_length,
            },
        ),
        Obstacle::Sdf { .. } => return Err(input!("attached objects cannot be distance grids")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panda() -> Robot {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/franka");
        let options = RobotOptions {
            ee_links: vec!["ee_link".into()],
            lock_joints: HashMap::from([
                ("panda_finger_joint1".into(), 0.04),
                ("panda_finger_joint2".into(), 0.04),
            ]),
            collision_model: Some(
                CollisionModel::load(format!("{dir}/panda_collision.json")).unwrap(),
            ),
            ..Default::default()
        };
        Robot::load(format!("{dir}/franka_panda.urdf"), &options).unwrap()
    }

    #[test]
    fn self_collision_pairs_skip_same_and_ignored_links() {
        let robot = panda();
        assert_eq!(robot.spheres.len(), 61);
        let link = |name: &str| robot.links.iter().position(|l| l.name == name).unwrap();
        let pairs_between = |a: usize, b: usize| {
            robot
                .self_pairs
                .iter()
                .filter(|&&[x, y]| {
                    let (lx, ly) = (
                        robot.spheres[x as usize].link,
                        robot.spheres[y as usize].link,
                    );
                    (lx, ly) == (a, b) || (lx, ly) == (b, a)
                })
                .count()
        };
        assert_eq!(pairs_between(link("panda_link1"), link("panda_link2")), 0);
        assert_eq!(
            pairs_between(link("panda_link0"), link("panda_link5")),
            2 * 13
        );
        assert!(
            robot
                .self_pairs
                .iter()
                .all(|&[a, b]| robot.spheres[a as usize].link != robot.spheres[b as usize].link)
        );
    }
}
