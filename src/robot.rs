//! Robot model: a kinematic tree with collision spheres, built from a robot description file.

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail, ensure};
use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::constraint::{self, HandConstraint};
use crate::description::{
    self, Curve, Geometry, JointDesc, JointType, LinkDesc, RobotDescription, Shape, TriMesh,
};
use crate::error::{self, Error, ensure_input, input};
use crate::spheres::{self, Qualification, SphereGeometry, SphereOptions};
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

/// Where a link's frame can be, for [`Robot::reach`]: the frame of the nearest moving joint at
/// or above it (or the root) lies in the box from `lo` to `hi`, grown by `carried` in every
/// direction; that joint's frame turns by `turn` in every configuration, or by different turns
/// (`None`); and the link's frame is `frame` in it.
#[derive(Clone, Copy)]
struct Reachable {
    lo: Vec3,
    hi: Vec3,
    carried: f32,
    turn: Option<Mat3>,
    frame: Transform,
}

impl Reachable {
    const ROOT: Reachable = Reachable {
        lo: Vec3::ZERO,
        hi: Vec3::ZERO,
        carried: 0.0,
        turn: Some(Mat3::IDENTITY),
        frame: Transform::IDENTITY,
    };

    /// Where the point `p` of the moving joint's frame can be, as the joint's own frame can.
    fn place(&self, p: Vec3) -> Reachable {
        match self.turn {
            Some(turn) => Reachable {
                lo: self.lo + turn * p,
                hi: self.hi + turn * p,
                frame: Transform::IDENTITY,
                ..*self
            },
            None => Reachable {
                carried: self.carried + p.length(),
                frame: Transform::IDENTITY,
                ..*self
            },
        }
    }
}

/// Points of a prismatic joint's range at which [`Robot::reach`] measures its travel: a mimic
/// joint's travel is a polynomial of its leader's value, whose largest magnitude may lie inside
/// the range.
const PRISMATIC_SAMPLES: usize = 64;

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

/// A link a locked joint moves (`RobotOptions::lock_joints`), its own joint folded into its
/// origin: what [`Robot::relock`] folds in again at another value.
#[derive(Clone, Debug)]
struct LockedLink {
    /// The locked joint: the link's joint, or the joint it mimics.
    joint: String,
    link: usize,
    /// The link's joint frame before its motion.
    origin: Transform,
    axis: Vec3,
    prismatic: bool,
    /// The link's joint's value from the locked joint's.
    curve: Curve,
}

impl LockedLink {
    /// The link's origin with the locked joint at `value`.
    fn origin(&self, value: f32) -> Transform {
        let value = self.curve.value(value);
        let motion = if self.prismatic {
            Transform {
                rot: Mat3::IDENTITY,
                trans: self.axis * value,
            }
        } else {
            Transform {
                rot: Mat3::from_axis_angle(self.axis, value),
                trans: Vec3::ZERO,
            }
        };
        self.origin.mul(&motion)
    }
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
    /// Joints locked while fitting, at their values. The fit found the pairs it ignores with them
    /// there alone, so a robot loaded with the model must lock them there, and [`Robot::relock`]
    /// leaves them. Models saved before batchplan 0.8.0 have none: refit them for joints that move.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub locked_joints: BTreeMap<String, f32>,
}

impl CollisionModel {
    /// Reads a model written by [`CollisionModel::save`].
    pub fn load(path: impl AsRef<Path>) -> error::Result<Self> {
        let path = path.as_ref();
        let load = |message: String| Error::Load {
            path: path.to_path_buf(),
            message,
        };
        let text = std::fs::read_to_string(path).map_err(|e| load(e.to_string()))?;
        serde_json::from_str(&text).map_err(|e| load(e.to_string()))
    }

    /// Writes the model as JSON, to commit and hand-tune.
    pub fn save(&self, path: impl AsRef<Path>) -> error::Result<()> {
        let path = path.as_ref();
        std::fs::write(path, serde_json::to_string_pretty(self)?)
            .map_err(|e| Error::Write(format!("writing {}: {e}", path.display())))
    }

    /// The model of a robot made with [`Robot::assemble`], from each part's own model and the
    /// prefix its names take: two arms fitted once share one model. Links of different parts are
    /// always checked against each other.
    #[must_use]
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
            out.locked_joints
                .extend(m.locked_joints.iter().map(|(j, v)| (named(j), *v)));
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

/// How [`Robot::load`] and [`Robot::assemble`] build a robot from its description.
#[derive(Clone, Debug)]
pub struct RobotOptions {
    /// Root of the planned tree; defaults to the description's root link.
    pub base_link: Option<String>,
    /// IK target frames, one per hand (end effector); defaults to the link with the most actuated
    /// joints above it. Set it for robots with grippers, and name each hand of a robot with
    /// several.
    pub ee_links: Vec<String>,
    /// Per hand, by its link in `ee_links`: the links it grips with, for a hand whose fingers are
    /// not the links below its palm, as a gripper whose fixed jaw is part of the palm's link
    /// (then name the palm too). [`Robot::finger_links`] returns them; hands left out keep the
    /// links below their palm.
    pub finger_links: HashMap<String, Vec<String>>,
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
    /// How spheres are fitted when there is no `collision_model`.
    pub spheres: SphereOptions,
    /// Per link, by name: how far its spheres may reach beyond its geometry (meters), in place of
    /// `spheres.tolerance`, as fingers fitted tighter than the arm so they can grasp small objects.
    /// Each link's spheres reach as far as its tolerance, or where the budget cannot reach every
    /// tolerance, as far as every tolerance scaled up by one factor: a tighter link costs the
    /// others overhang.
    pub sphere_tolerances: HashMap<String, f32>,
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
    /// `s`, as `Trajectory::try_new`'s speed scale does, scale velocity by `s`, acceleration by `s²`
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
            finger_links: HashMap::new(),
            default_q: None,
            lock_joints: HashMap::new(),
            joint_limits: HashMap::new(),
            package_dirs: vec![],
            collision_model: None,
            spheres: SphereOptions::default(),
            sphere_tolerances: HashMap::new(),
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
    /// Per hand, the finger links `RobotOptions::finger_links` names, if it names any.
    fingers: Vec<Option<Vec<usize>>>,
    pub(crate) default_q: Vec<f32>,
    collision_model: CollisionModel,
    /// How spheres are fitted to its links (`RobotOptions::spheres`), which `qualify` samples by.
    sphere_options: SphereOptions,
    /// Names of attached objects, each a fixed link at the end of `links`.
    attached: Vec<String>,
    /// The links locked joints move.
    locked: Vec<LockedLink>,
    /// The descriptions it was loaded from.
    parts: Vec<Part>,
}

/// A robot description placed in a robot made of several (see [`Robot::assemble`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    /// The description file: URDF, MJCF or OpenUSD.
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
    /// The link holding it.
    pub link: String,
    /// The object's shapes, placed in `link`'s frame. Distance grids are not supported.
    pub shapes: Vec<Obstacle>,
    /// Other links it may touch.
    pub touch_links: Vec<String>,
    /// How spheres are fitted to the shapes; a held object rarely needs more than 8 to 16. A single
    /// shape is fitted in its own frame, so where it sits on `link` does not change its spheres.
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
    #[must_use]
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
            Some(model) => {
                let moved = model
                    .locked_joints
                    .iter()
                    .find(|&(joint, value)| o.lock_joints.get(joint) != Some(value));
                if let Some((joint, value)) = moved {
                    bail!(
                        "the collision model was fitted with joint '{joint}' locked at {value}, so \
                         the pairs it ignores were found there alone: lock it there, or fit the \
                         model with it free"
                    );
                }
                model.clone()
            }
            None => {
                let mut model = robot.fit_collision_model(&o.spheres, &o.sphere_tolerances)?;
                model.locked_joints = o.lock_joints.iter().map(|(j, v)| (j.clone(), *v)).collect();
                model
            }
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
    fn fit_collision_model(
        &mut self,
        o: &SphereOptions,
        tolerances: &HashMap<String, f32>,
    ) -> Result<CollisionModel> {
        if let Some(name) = tolerances
            .keys()
            .find(|n| !self.links.iter().any(|l| &l.name == *n))
        {
            bail!("sphere_tolerances names '{name}', which is not a link of the robot");
        }
        if let Some((name, t)) = tolerances
            .iter()
            .find(|(_, t)| !(t.is_finite() && **t > 0.0))
        {
            bail!("the sphere tolerance of '{name}' must be positive, got {t}");
        }
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
        let tolerances: Vec<f32> = self
            .links
            .iter()
            .map(|link| tolerances.get(&link.name).copied().unwrap_or(o.tolerance))
            .collect();
        let fitted = spheres::fit(&meshes, &tolerances, o, /* parallel */ true);
        let source = format!(
            "fitted by batchplan: spheres reach up to {:.1} mm beyond the geometry and up to {:.2} \
             times their link's tolerance, with every tolerance scaled by {:.2} for the budget",
            fitted.overhang * 1e3,
            fitted.relative_overhang,
            fitted.scale
        );
        let mut model = CollisionModel {
            source: Some(source),
            ..Default::default()
        };
        for (link, fitted) in self.links.iter().zip(fitted.spheres) {
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
        let malformed = model.spheres.iter().find_map(|(name, spheres)| {
            let bad = |s: &&[f32; 4]| !(s.iter().all(|v| v.is_finite()) && s[3] >= 0.0);
            spheres.iter().find(bad).map(|s| (name, s))
        });
        if let Some((name, s)) = malformed {
            bail!(
                "collision sphere {s:?} of link '{name}' (given, or fitted to its geometry) needs a \
                 finite center and a finite radius of at least 0"
            );
        }
        // A buffer may shrink a link's spheres for self-collision, but not below nothing.
        for (name, &b) in &model.self_collision_buffer {
            let smallest = model
                .spheres
                .get(name)
                .into_iter()
                .flatten()
                .map(|s| s[3])
                .fold(f32::INFINITY, f32::min);
            ensure!(
                b.is_finite() && (smallest.is_infinite() || b >= -smallest),
                "the self-collision buffer of link '{name}' must be finite and at least minus its \
                 smallest sphere's radius ({smallest}), got {b}"
            );
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
    /// is taken, a shape is a distance grid, has a size that is negative or not finite, or a
    /// rotation that is not a unit quaternion (as [`Device::upload`](crate::Device::upload) refuses
    /// it), or the robot would exceed the kernel limits.
    pub fn attach(&self, object: &AttachedObject) -> error::Result<Robot> {
        self.attach_with(object, || object_spheres(&object.shapes, &object.spheres))
    }

    /// [`Robot::attach`], with `spheres` giving the object's spheres in `link`'s frame once the
    /// object checks out: [`Held`](crate::Held) fits its object once for every grasp of it.
    pub(crate) fn attach_with(
        &self,
        object: &AttachedObject,
        spheres: impl FnOnce() -> error::Result<Vec<[f32; 4]>>,
    ) -> error::Result<Robot> {
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
        // Distance grids are refused first, so a shape's error never prints a grid.
        let shapes = object
            .shapes
            .iter()
            .map(attached_shape)
            .collect::<error::Result<Vec<Shape>>>()?;
        for (k, shape) in object.shapes.iter().enumerate() {
            shape
                .check()
                .map_err(|e| input!("'{}' shape {k}: {e}, got {shape:?}", object.name))?;
        }
        ensure_input!(!shapes.is_empty(), "'{}' has no shapes", object.name);
        let fitted = spheres()?;
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

    /// How many of `links` are the robot's own: the objects it holds are links after them.
    #[cfg(feature = "gpu")]
    pub(crate) fn own_links(&self) -> usize {
        self.links.len() - self.attached.len()
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

    /// This robot with joints it was loaded with locked (`RobotOptions::lock_joints`) held at
    /// new values, as a torso that has moved: the links they carry move, and nothing else changes,
    /// as if loaded there with the same collision model.
    /// [`Device::with_robot`](crate::Device::with_robot) makes a device for it without building
    /// anything, so worlds uploaded stay valid. Joints not named keep their values. A model fitted
    /// with a joint locked found the pairs it ignores at that joint's value alone, so pairs that
    /// meet only at others would go unchecked: for joints that will move, fit the model with them
    /// free and load with it ([`RobotOptions::collision_model`]). Errors if a joint was not locked,
    /// is moved from where it was locked when the collision model was fitted
    /// ([`CollisionModel::locked_joints`]), or a value is not finite.
    pub fn relock(&self, joints: &HashMap<String, f32>) -> error::Result<Robot> {
        for (name, &value) in joints {
            ensure_input!(
                self.locked.iter().any(|l| l.joint == *name),
                "joint '{name}' was not locked when the robot was loaded"
            );
            if let Some(&fitted) = self.collision_model.locked_joints.get(name) {
                ensure_input!(
                    value == fitted,
                    "joint '{name}' was locked at {fitted} when the collision model was fitted, so \
                     the pairs it ignores were found there alone: fit it with the joint free, then \
                     load with that model (RobotOptions::collision_model)"
                );
            }
            ensure_input!(
                value.is_finite(),
                "joint '{name}' locked at {value}, not a finite value"
            );
        }
        let mut robot = self.clone();
        for held in &self.locked {
            if let Some(&value) = joints.get(&held.joint) {
                robot.links[held.link].origin = held.origin(value);
            }
        }
        Ok(robot)
    }

    /// The robot name from its description.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The number of actuated joints: the length of every configuration.
    #[must_use]
    #[inline]
    pub fn dof(&self) -> usize {
        self.dof_names.len()
    }

    /// The actuated joints' names, in the order configurations hold them.
    #[must_use]
    pub fn joint_names(&self) -> &[String] {
        &self.dof_names
    }

    /// Each joint's lower position limit (radians, or meters for prismatic joints).
    #[must_use]
    pub fn lower(&self) -> &[f32] {
        &self.lower
    }

    /// Each joint's upper position limit.
    #[must_use]
    pub fn upper(&self) -> &[f32] {
        &self.upper
    }

    /// Which joints are continuous: they turn without limit and their values wrap every 2π. For
    /// them, [`Robot::lower`] and [`Robot::upper`] give the turn (−π to π) that seeds and samples
    /// come from, not limits.
    #[must_use]
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
        let continuous = q
            .iter_mut()
            .enumerate()
            .filter(|&(j, _)| self.continuous[j]);
        for (j, v) in continuous {
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

    /// Each joint's velocity limit (per second), with `RobotOptions::velocity_scale` applied.
    #[must_use]
    pub fn max_velocity(&self) -> &[f32] {
        &self.max_velocity
    }

    /// Each joint's acceleration limit, with `RobotOptions::acceleration_scale` applied.
    #[must_use]
    pub fn max_acceleration(&self) -> &[f32] {
        &self.max_acceleration
    }

    /// Each joint's jerk limit, with `RobotOptions::jerk_scale` applied.
    #[must_use]
    pub fn max_jerk(&self) -> &[f32] {
        &self.max_jerk
    }

    /// The configuration plans start from by default and IK's first seed starts at.
    #[must_use]
    pub fn default_q(&self) -> &[f32] {
        &self.default_q
    }

    /// The collision spheres in use, generated or as given.
    #[must_use]
    pub fn collision_model(&self) -> &CollisionModel {
        &self.collision_model
    }

    /// The number of hands: IK target frames (`RobotOptions::ee_links`).
    #[must_use]
    pub fn hands(&self) -> usize {
        self.ee_links.len()
    }

    /// World pose of hand `hand`'s IK target frame at configuration `q`.
    #[must_use]
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
    #[must_use]
    pub fn hand_joints(&self, hand: usize) -> Vec<usize> {
        let mut joints = vec![];
        for (_, dof) in self.chain(self.ee_links[hand]) {
            if !joints.contains(&dof) {
                joints.push(dof);
            }
        }
        joints
    }

    /// Whether two hands share a joint, as the arms on a mobile base or a torso do: IK then
    /// solves their targets as one system.
    pub(crate) fn hands_share_joints(&self) -> bool {
        let joints: Vec<Vec<usize>> = (0..self.hands()).map(|h| self.hand_joints(h)).collect();
        joints.iter().enumerate().any(|(h, a)| {
            joints[h + 1..]
                .iter()
                .any(|b| a.iter().any(|j| b.contains(j)))
        })
    }

    /// Names of the links of hand `hand`, for [`Touch`](crate::Touch): the first link at or
    /// above its frame that has collision spheres (the Panda's `panda_hand`, above its tool frame)
    /// and every link below that one, its fingers and any object it holds, with the fingers
    /// [`RobotOptions::finger_links`] names.
    #[must_use]
    pub fn hand_links(&self, hand: usize) -> Vec<&str> {
        let below = self.below_hand(hand, /* with_palm */ true);
        let named = self.fingers[hand].iter().flatten();
        let links =
            (0..self.links.len()).filter(|i| below.contains(i) || named.clone().any(|f| f == i));
        links.map(|i| self.links[i].name.as_str()).collect()
    }

    /// Names of the links hand `hand` grips with: those [`RobotOptions::finger_links`] names for
    /// it, or else the links below the first of [`Robot::hand_links`] that have collision spheres,
    /// its fingers (and any object it holds) without its palm. Empty for a hand with none, such as
    /// a suction cup or a bare flange, whose frames below the palm are only frames.
    #[must_use]
    pub fn finger_links(&self, hand: usize) -> Vec<&str> {
        let links = match &self.fingers[hand] {
            Some(named) => named.clone(),
            None => self.below_hand(hand, /* with_palm */ false),
        };
        links
            .into_iter()
            .map(|i| self.links[i].name.as_str())
            .collect()
    }

    fn below_hand(&self, hand: usize, with_palm: bool) -> Vec<usize> {
        let ancestors = |link: usize| std::iter::successors(Some(link), |&l| self.links[l].parent);
        let spheres = |link: usize| self.sphere_ranges[link][1] > 0;
        let frame = self.ee_links[hand];
        let palm = ancestors(frame).find(|&l| spheres(l)).unwrap_or(frame);
        (0..self.links.len())
            .filter(|&i| ancestors(i).any(|a| a == palm))
            .filter(|&i| with_palm || (i != palm && spheres(i)))
            .collect()
    }

    /// The joints other hands move and hand `hand` does not, sorted: they wait while it works.
    pub(crate) fn idle_joints(&self, hand: usize) -> Vec<usize> {
        let mine = self.hand_joints(hand);
        let mut others: Vec<usize> = (0..self.hands())
            .filter(|&h| h != hand)
            .flat_map(|h| self.hand_joints(h))
            .filter(|j| !mine.contains(j))
            .collect();
        others.sort_unstable();
        others.dedup();
        others
    }

    /// World pose of the named link at configuration `q`.
    #[must_use]
    pub fn link_pose(&self, q: &[f32], link: &str) -> Option<Pose> {
        let i = self.links.iter().position(|l| l.name == link)?;
        Some(self.pose_of(&self.fk(q), i))
    }

    /// How far hand `hand`'s spheres reach past its IK frame along the frame's z axis at the
    /// default configuration (meters): how much deeper than a grasp the fingertips go, for
    /// [`GraspOptions::tip`](crate::GraspOptions::tip). Negative where every sphere stays short of
    /// the frame.
    #[must_use]
    pub fn tip(&self, hand: usize) -> f32 {
        let fk = self.fk(&self.default_q);
        let frame = self.ee_links[hand];
        let (rot, pos) = (fk.rot[frame], fk.pos[frame]);
        let links = self.hand_links(hand);
        let hand_spheres = self
            .spheres
            .iter()
            .filter(|s| links.contains(&self.links[s.link].name.as_str()));
        hand_spheres
            .map(|s| {
                let center = fk.rot[s.link] * s.center + fk.pos[s.link];
                rot.z_axis.dot(center - pos) + s.radius
            })
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// How well the collision spheres fit the robot's geometry, link by link, and how far each
    /// hand's reach past its frame: what to check before planning with a new robot or a
    /// hand-tuned collision model ([`Qualification`]). Measured against the geometry spheres are
    /// fitted to ([`RobotOptions::spheres`]), with its seed; reads mesh files, so it takes about
    /// as long as fitting. An `Error::Load` where a mesh cannot be read.
    pub fn qualify(&self) -> error::Result<Qualification> {
        let mut links = vec![];
        for (i, link) in self.links.iter().enumerate() {
            let [first, count] = self.sphere_ranges[i];
            if count == 0 || link.shapes.is_empty() {
                continue;
            }
            let meshes = link
                .shapes
                .iter()
                .map(Shape::mesh)
                .collect::<Result<Vec<TriMesh>>>()
                .map_err(|e| Error::Load {
                    path: self
                        .parts
                        .first()
                        .map(|p| p.path.clone())
                        .unwrap_or_default(),
                    message: format!("link '{}': {e:#}", link.name),
                })?;
            let own = &self.spheres[first as usize..(first + count) as usize];
            let spheres: Vec<[f32; 4]> = own
                .iter()
                .map(|s| [s.center.x, s.center.y, s.center.z, s.radius])
                .collect();
            links.extend(spheres::measure(
                &link.name,
                &meshes,
                &spheres,
                &self.sphere_options,
            ));
        }
        Ok(Qualification {
            links,
            tips: (0..self.hands()).map(|h| self.tip(h)).collect(),
        })
    }

    /// Every collision sphere at configuration `q`, in the robot's frame, as `[x, y, z, radius]`,
    /// link by link: what the world is checked against, to mask the robot out of a camera's view
    /// or draw it.
    #[must_use]
    pub fn spheres(&self, q: &[f32]) -> Vec<[f32; 4]> {
        let fk = self.fk(q);
        self.spheres
            .iter()
            .map(|s| {
                let c = fk.rot[s.link] * s.center + fk.pos[s.link];
                [c.x, c.y, c.z, s.radius]
            })
            .collect()
    }

    /// How far from the origin of its frame the robot reaches (meters): a bound on its collision
    /// spheres' farthest points over every configuration within the joint limits, to leave out
    /// what a camera sees beyond it. Joint by joint down each chain, the bound keeps a box that
    /// holds every place a joint's frame can be and how far a revolute joint above it can carry
    /// it: translations along fixed directions (a mobile base's) grow the box, and a turn carries
    /// what lies past it as far as it lies from the turning joint.
    #[must_use]
    pub fn reach(&self) -> f32 {
        let mut places: Vec<Reachable> = Vec::with_capacity(self.links.len());
        for link in &self.links {
            let above = link.parent.map_or(Reachable::ROOT, |p| places[p]);
            let joint = above.frame.mul(&link.origin);
            let at = above.place(joint.trans);
            places.push(match link.joint {
                JointKind::Fixed => Reachable {
                    frame: joint,
                    ..above
                },
                JointKind::Revolute { .. } => Reachable { turn: None, ..at },
                JointKind::Prismatic { dof, axis, curve } => {
                    let (lo, hi) = (self.lower[dof], self.upper[dof]);
                    let (least, most) = (0..=PRISMATIC_SAMPLES)
                        .map(|k| curve.value(lo + (hi - lo) * k as f32 / PRISMATIC_SAMPLES as f32))
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                            (a.min(v), b.max(v))
                        });
                    match at.turn {
                        Some(turn) => {
                            let turn = turn * joint.rot;
                            let ends = [turn * axis * least, turn * axis * most];
                            Reachable {
                                lo: at.lo + ends[0].min(ends[1]),
                                hi: at.hi + ends[0].max(ends[1]),
                                turn: Some(turn),
                                ..at
                            }
                        }
                        None => Reachable {
                            carried: at.carried + least.abs().max(most.abs()),
                            ..at
                        },
                    }
                }
            });
        }
        self.spheres
            .iter()
            .map(|s| {
                let place = places[s.link];
                let at = place.place(place.frame.rot * s.center + place.frame.trans);
                at.lo.abs().max(at.hi.abs()).length() + at.carried + s.radius
            })
            .fold(0.0, f32::max)
    }

    /// Why configuration `q` breaks `constraints`, if it does: the constraint, the bound and by how
    /// much, in words. Bounds below IK's precision count as [`crate::plan`] counts them, so this is
    /// the constraint test a plan's start and goals must pass. An `Error::Input` for constraints the
    /// robot can't keep, as `plan` refuses them, or a `q` without one finite value per joint.
    pub fn breach(
        &self,
        constraints: &[HandConstraint],
        q: &[f32],
    ) -> error::Result<Option<String>> {
        ensure_input!(
            q.len() == self.dof() && q.iter().all(|v| v.is_finite()),
            "a configuration needs {} finite values, got {q:?}",
            self.dof()
        );
        constraint::check(self, constraints, "the constraints")?;
        Ok(constraint::breach(self, constraints, q))
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

    /// Hand `hand`'s linear and angular velocity at the configuration `fk` holds, with the joints
    /// moving at `dq`.
    pub(crate) fn hand_velocity(&self, fk: &Fk, hand: usize, dq: &[f32]) -> [Vec3; 2] {
        let ee = self.ee_links[hand];
        let (mut linear, mut angular) = (Vec3::ZERO, Vec3::ZERO);
        for (i, dof) in self.chain(ee) {
            let curve = &self.links[i]
                .joint
                .actuation()
                .expect("chain links have moving joints")
                .1;
            let rate = fk.rate(i, curve) * dq[dof];
            linear += rate * fk.dpoint(i, fk.pos[ee]);
            if !fk.slides[i] {
                angular += rate * fk.axis[i];
            }
        }
        [linear, angular]
    }

    /// Into `out`, per joint, hand `hand`'s linear and angular velocity per unit of that joint's
    /// velocity at the configuration `fk` holds: its Jacobian's columns.
    #[inline]
    pub(crate) fn hand_jacobian(&self, fk: &Fk, hand: usize, out: &mut [[Vec3; 2]]) {
        let ee = self.ee_links[hand];
        out.fill([Vec3::ZERO; 2]);
        for (i, dof) in self.chain(ee) {
            let curve = &self.links[i]
                .joint
                .actuation()
                .expect("chain links have moving joints")
                .1;
            let rate = fk.rate(i, curve);
            out[dof][0] += rate * fk.dpoint(i, fk.pos[ee]);
            if !fk.slides[i] {
                out[dof][1] += rate * fk.axis[i];
            }
        }
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
    for (name, value) in &o.lock_joints {
        let j = joint_index
            .get(name.as_str())
            .with_context(|| format!("cannot lock unknown joint '{name}'"))?;
        ensure!(
            desc.joints[*j].kind != JointType::Fixed,
            "cannot lock fixed joint '{name}'"
        );
        ensure!(
            value.is_finite(),
            "cannot lock joint '{name}' at {value}, not a finite value"
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
        let children = desc
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.parent == link);
        for (j, joint) in children {
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
    let actuated = order
        .iter()
        .filter(|&&j| moving(j) && desc.joints[j].mimic.is_none());
    for &j in actuated {
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
                .unwrap_or_else(|| or_default(joint.max_velocity, o.max_velocity)),
        );
        max_acceleration.push(
            limits
                .acceleration
                .unwrap_or_else(|| or_default(joint.max_acceleration, o.max_acceleration)),
        );
        max_jerk.push(
            limits
                .jerk
                .unwrap_or_else(|| or_default(joint.max_jerk, o.max_jerk)),
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
    let mut locked = vec![];
    let mut link_index: HashMap<String, usize> = HashMap::from([(base, 0)]);
    for &j in &order {
        let joint = &desc.joints[j];
        let parent = link_index[&joint.parent];
        let mut origin = joint.origin;
        let kind = if moving(j) {
            let (leader, curve) = driver(j)?;
            let prismatic = joint.kind == JointType::Prismatic;
            if let Some(&value) = o.lock_joints.get(&desc.joints[leader].name) {
                let held = LockedLink {
                    joint: desc.joints[leader].name.clone(),
                    link: links.len(),
                    origin,
                    axis: joint.axis,
                    prismatic,
                    curve,
                };
                origin = held.origin(value);
                locked.push(held);
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
    let mut fingers = vec![None; ee_links.len()];
    for (hand, named) in &o.finger_links {
        let at = ee_links
            .iter()
            .position(|&e| links[e].name == *hand)
            .with_context(|| {
                format!("finger_links names '{hand}', which is not a hand's ee_link")
            })?;
        let found = named
            .iter()
            .map(|name| {
                link_index
                    .get(name)
                    .copied()
                    .with_context(|| format!("unknown finger link '{name}' of hand '{hand}'"))
            })
            .collect::<anyhow::Result<Vec<usize>>>()?;
        fingers[at] = Some(found);
    }
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
        for l in limits.iter_mut() {
            *l *= scale;
        }
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
        fingers,
        default_q,
        collision_model: CollisionModel::default(),
        sphere_options: o.spheres,
        attached: vec![],
        locked,
        parts: vec![],
    })
}

/// Spheres for an attached object's `shapes`, in their frame: boxes and spheres covered directly,
/// other shapes fitted. One shape is covered in its own frame and its spheres placed by its pose,
/// so where it sits on its link does not change them.
pub(crate) fn object_spheres(
    shapes: &[Obstacle],
    o: &SphereOptions,
) -> error::Result<Vec<[f32; 4]>> {
    o.check().map_err(Error::Input)?;
    if let [shape] = shapes {
        let own = covered(std::slice::from_ref(&shape.placed(Pose::IDENTITY)), o)?;
        return Ok(placed_spheres(&own, shape.pose()));
    }
    covered(shapes, o)
}

/// `spheres` (`[x, y, z, radius]`) moved from a frame into the one `pose` is given in.
pub(crate) fn placed_spheres(spheres: &[[f32; 4]], pose: Pose) -> Vec<[f32; 4]> {
    let place = |s: &[f32; 4]| {
        let c = pose.position + pose.rotation * Vec3::new(s[0], s[1], s[2]);
        [c.x, c.y, c.z, s[3]]
    };
    spheres.iter().map(place).collect()
}

/// [`object_spheres`] for `shapes` as they are placed. Attaching happens inside control processes:
/// fit on the caller's rayon pool if it has one, otherwise on the calling thread alone, never on
/// the global pool. (A one-thread pool on the calling thread would leave it registered with rayon
/// for good, and every parallel loop it started afterwards would run on that one thread.)
fn covered(shapes: &[Obstacle], o: &SphereOptions) -> error::Result<Vec<[f32; 4]>> {
    if let Some(covered) = spheres::cover_primitives(shapes, o) {
        return Ok(covered);
    }
    let meshes = shapes
        .iter()
        .map(|shape| attached_shape(shape)?.mesh().map_err(|e| input!("{e:#}")))
        .collect::<error::Result<Vec<TriMesh>>>()?;
    let fitted = spheres::fit(
        &[meshes],
        &[o.tolerance],
        o,
        rayon::current_thread_index().is_some(),
    );
    Ok(fitted.spheres.into_iter().flatten().collect())
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
