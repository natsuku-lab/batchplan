//! OpenUSD loading (feature `usd`): a UsdPhysics articulation becomes a [`RobotDescription`];
//! colliders outside rigid bodies become scene shapes.
//!
//! - Rigid bodies (`PhysicsRigidBodyAPI`) are links. The tree grows from the world over Revolute,
//!   Prismatic, Spherical (a ball joint, its cone limits bounding each rotation angle) and Fixed
//!   joints: a joint frame is `localPos0`/`localRot0` in body0, and body1
//!   sits at `localPos1`/`localRot1` from it (a fixed child frame when that is not identity).
//!   Joints authored from child to parent are flipped; joints that close a loop, or are excluded
//!   from the articulation, become loop closures (`loops.rs`), driven by joints that `MjcActuator`s
//!   (directly or through an `MjcTendon`) or `PhysicsDriveAPI` drive. Bodies without a joint stay where they are authored.
//! - Plain `Xform` prims without physics whose children are all `Xform`s (the ghost links URDF
//!   converters leave, such as an end-effector frame) become fixed frames of the rigid body above
//!   them. Prims holding geometry, such as visual groups, do not.
//! - Revolute limits and angular velocity limits are degrees, as UsdPhysics and PhysX author them.
//! - Mimic joints come from `NewtonMimicAPI` or `PhysxMimicJointAPI`.
//! - Lengths are scaled by `metersPerUnit` (USD's fallback is centimeters) and Y-up stages are
//!   turned Z-up.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use glam::{DAffine3, DQuat, DVec3, Mat3, Vec3};
use openusd::sdf::Value;
use openusd::usd::{Prim, PrimPredicate, SchemaBase, SchemaKind, Stage, TimeCode};
use openusd_schemas::geom::{Imageable, Xformable};

use crate::description::{
    Curve, Geometry, JointDesc, JointType, LinkDesc, LoopClosure, Mimic, Model, RobotDescription, Shape, TriMesh,
};
use crate::mjcf::WORLD;
use crate::robot::Transform;

/// Any prim, viewed through the xformable schema to evaluate its transform ops.
struct Node(Prim);

impl SchemaBase for Node {
    const KIND: SchemaKind = SchemaKind::ConcreteTyped;

    fn prim(&self) -> &Prim {
        &self.0
    }
}

impl Imageable for Node {}
impl Xformable for Node {}

fn value(prim: &Prim, name: &str) -> Result<Option<Value>> {
    Ok(prim.attribute(name).get::<Value>()?)
}

fn scalar(prim: &Prim, name: &str) -> Result<Option<f64>> {
    Ok(match value(prim, name)? {
        Some(Value::Float(v)) => Some(v as f64),
        Some(Value::Double(v)) => Some(v),
        Some(Value::Half(v)) => Some(f32::from(v) as f64),
        Some(Value::Int(v)) => Some(v as f64),
        None => None,
        Some(other) => bail!("{}.{name}: expected a number, got {other:?}", prim.path()),
    })
}

fn vec3(prim: &Prim, name: &str) -> Result<Option<DVec3>> {
    Ok(match value(prim, name)? {
        Some(Value::Vec3f(v)) => Some(DVec3::new(v.x as f64, v.y as f64, v.z as f64)),
        Some(Value::Vec3d(v)) => Some(DVec3::new(v.x, v.y, v.z)),
        None => None,
        Some(other) => bail!("{}.{name}: expected a 3-vector, got {other:?}", prim.path()),
    })
}

fn quat(prim: &Prim, name: &str) -> Result<Option<DQuat>> {
    Ok(match value(prim, name)? {
        Some(Value::Quatf(q)) => Some(DQuat::from_xyzw(q.x as f64, q.y as f64, q.z as f64, q.w as f64)),
        Some(Value::Quatd(q)) => Some(DQuat::from_xyzw(q.x, q.y, q.z, q.w)),
        None => None,
        Some(other) => bail!("{}.{name}: expected a quaternion, got {other:?}", prim.path()),
    })
}

fn token(prim: &Prim, name: &str) -> Result<Option<String>> {
    Ok(match value(prim, name)? {
        Some(Value::Token(t)) => Some(t.as_str().to_string()),
        Some(Value::String(s)) => Some(s),
        None => None,
        Some(other) => bail!("{}.{name}: expected a token, got {other:?}", prim.path()),
    })
}

fn boolean(prim: &Prim, name: &str, default: bool) -> Result<bool> {
    Ok(match value(prim, name)? {
        Some(Value::Bool(b)) => b,
        _ => default,
    })
}

fn target(prim: &Prim, name: &str) -> Result<Option<String>> {
    Ok(prim.relationship(name).targets()?.first().map(|p| p.to_string()))
}

fn only_xform_children(prim: &Prim) -> Result<bool> {
    for child in prim.children()? {
        if child.type_name()?.is_none_or(|t| t.as_str() != "Xform") {
            return Ok(false);
        }
    }
    Ok(true)
}

fn has_api(prim: &Prim, api: &str) -> Result<bool> {
    Ok(prim.api_schemas()?.iter().any(|a| a.as_str() == api))
}

fn name_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Unit axis for an axis token.
fn axis(token: &str) -> Result<DVec3> {
    Ok(match token {
        "X" => DVec3::X,
        "Y" => DVec3::Y,
        "Z" => DVec3::Z,
        other => bail!("unknown axis '{other}'"),
    })
}

fn rigid(affine: DAffine3) -> Transform {
    let (_, rot, trans) = affine.to_scale_rotation_translation();
    Transform { rot: Mat3::from_quat(rot.as_quat()), trans: trans.as_vec3() }
}

struct Loader {
    stage: Stage,
    meters: f64,
    /// Stage-to-world rotation (Y-up stages turn Z-up).
    up: DQuat,
    world: HashMap<String, DAffine3>,
}

impl Loader {
    /// World transform of a prim in stage units, Y-up corrected.
    fn world(&mut self, path: &str) -> Result<DAffine3> {
        if let Some(&w) = self.world.get(path) {
            return Ok(w);
        }
        let prim = self.stage.prim(path)?;
        let node = Node(prim);
        let m = node.local_to_parent_transform(TimeCode::EARLIEST).map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
        // gf matrices are row-vector (translation in the last row): read as columns, they are
        // the column-vector transform.
        let local = DAffine3::from_mat4(glam::DMat4::from_cols_array(&m.0));
        let parent = match path.rsplit_once('/') {
            Some(("", _)) | None => DAffine3::from_quat(self.up),
            Some((parent, _)) if !node.resets_xform_stack()? => self.world(parent)?,
            _ => DAffine3::from_quat(self.up),
        };
        let w = parent * local;
        self.world.insert(path.to_string(), w);
        Ok(w)
    }

    /// A joint frame (`localPos`, `localRot`) in its body's frame, in meters.
    fn local_frame(&self, joint: &Prim, side: u8) -> Result<DAffine3> {
        let pos = vec3(joint, &format!("physics:localPos{side}"))?.unwrap_or(DVec3::ZERO);
        let rot = quat(joint, &format!("physics:localRot{side}"))?.unwrap_or(DQuat::IDENTITY).normalize();
        Ok(DAffine3::from_rotation_translation(rot, pos * self.meters))
    }

    /// Rigid frame of a body (or world-fixed prim) in meters.
    fn body_frame(&mut self, path: &str) -> Result<DAffine3> {
        let (_, rot, trans) = self.world(path)?.to_scale_rotation_translation();
        Ok(DAffine3::from_rotation_translation(rot, trans * self.meters))
    }

    /// A collider's shape relative to `frame` (a body path, or the world when `None`).
    fn shape(&mut self, prim: &Prim, frame: Option<&str>) -> Result<Shape> {
        let path = prim.path().to_string();
        let world = self.world(&path)?;
        let relative = match frame {
            Some(body) => self.world(body)?.inverse() * world,
            None => world,
        };
        let (scale, rot, trans) = relative.to_scale_rotation_translation();
        let mut origin = DAffine3::from_rotation_translation(rot, trans * self.meters);
        let s = scale.abs() * self.meters;
        let kind = prim.type_name()?.map(|t| t.as_str().to_string()).unwrap_or_default();
        let along = |prim: &Prim| -> Result<(DQuat, usize)> {
            Ok(match token(prim, "axis")?.as_deref().unwrap_or("Z") {
                "X" => (DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2), 0),
                "Y" => (DQuat::from_rotation_x(-std::f64::consts::FRAC_PI_2), 1),
                _ => (DQuat::IDENTITY, 2),
            })
        };
        let geometry = match kind.as_str() {
            "Cube" => {
                let size = scalar(prim, "size")?.unwrap_or(2.0);
                Geometry::Box { half: (s * size * 0.5).as_vec3() }
            }
            "Sphere" => {
                let r = scalar(prim, "radius")?.unwrap_or(1.0);
                if (s.max_element() - s.min_element()).abs() < 1e-9 * s.max_element() {
                    Geometry::Sphere { radius: (r * s.x) as f32 }
                } else {
                    Geometry::Ellipsoid { radii: (s * r).as_vec3() }
                }
            }
            "Cylinder" | "Capsule" => {
                let (to_axis, a) = along(prim)?;
                let defaults = if kind == "Cylinder" { (1.0, 2.0) } else { (0.5, 1.0) };
                let r = scalar(prim, "radius")?.unwrap_or(defaults.0);
                let h = scalar(prim, "height")?.unwrap_or(defaults.1);
                let radial = (0..3).filter(|&i| i != a).map(|i| s[i]).fold(0.0, f64::max);
                origin *= DAffine3::from_quat(to_axis);
                let (radius, half_length) = ((r * radial) as f32, (0.5 * h * s[a]) as f32);
                if kind == "Cylinder" {
                    Geometry::Cylinder { radius, half_length }
                } else {
                    Geometry::Capsule { radius, half_length }
                }
            }
            "Plane" => {
                let (to_axis, _) = along(prim)?;
                origin *= DAffine3::from_quat(to_axis);
                Geometry::Plane
            }
            "Mesh" => {
                let points = match value(prim, "points")? {
                    Some(Value::Vec3fVec(p)) => {
                        p.iter().map(|v| DVec3::new(v.x as f64, v.y as f64, v.z as f64)).collect::<Vec<_>>()
                    }
                    Some(Value::Vec3dVec(p)) => p.iter().map(|v| DVec3::new(v.x, v.y, v.z)).collect(),
                    other => bail!("{path}: mesh points are {other:?}"),
                };
                let ints = |name: &str| -> Result<Vec<u32>> {
                    match value(prim, name)? {
                        Some(Value::IntVec(v)) => Ok(v.iter().map(|&i| i as u32).collect()),
                        other => bail!("{path}: {name} is {other:?}"),
                    }
                };
                let (counts, indices) = (ints("faceVertexCounts")?, ints("faceVertexIndices")?);
                let mut triangles = vec![];
                let mut start = 0usize;
                for &count in &counts {
                    let face =
                        indices.get(start..start + count as usize).context("faceVertexCounts overrun the indices")?;
                    triangles.extend((1..face.len().saturating_sub(1)).map(|k| [face[0], face[k], face[k + 1]]));
                    start += count as usize;
                }
                let vertices = points.iter().map(|&p| (p * s).as_vec3()).collect();
                let mesh = Geometry::TriMesh(TriMesh { vertices, triangles });
                // Other approximations stand in for the exact mesh, which is used instead.
                match token(prim, "physics:approximation")?.as_deref() {
                    Some("convexHull") => Geometry::ConvexHull(Box::new(mesh)),
                    _ => mesh,
                }
            }
            other => bail!("{path}: collider type '{other}' is not supported"),
        };
        Ok(Shape { origin: rigid(origin), geometry })
    }
}

#[derive(Clone)]
struct JointPrim {
    path: String,
    kind: String,
    body0: Option<String>,
    body1: Option<String>,
}

pub(crate) fn load(path: &Path, variants: &[(String, String)]) -> Result<Model> {
    let mut fallbacks = openusd::pcp::VariantFallbackMap::new();
    for (set, selection) in variants {
        fallbacks = fallbacks.add(set.clone(), [selection.clone()]);
    }
    let file = path.to_str().context("USD paths must be valid UTF-8")?;
    let stage = Stage::builder().variant_fallbacks(fallbacks).open(file).with_context(|| format!("opening {file}"))?;
    let meters = match stage.stage_metadata("metersPerUnit")? {
        Some(Value::Double(m)) => m,
        Some(Value::Float(m)) => m as f64,
        _ => 0.01,
    };
    let up = match stage.stage_metadata("upAxis")? {
        Some(Value::Token(t)) => t.as_str().to_string(),
        _ => "Y".to_string(),
    };
    let up = match up.as_str() {
        "Z" => DQuat::IDENTITY,
        "Y" => DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
        other => bail!("unsupported upAxis '{other}'"),
    };
    let robot_name = stage.default_prim().map(|t| t.as_str().to_string()).unwrap_or_else(|| "usd".into());
    let mut paths = vec![];
    // Instance proxies too, so instanced colliders count.
    stage.traverse(PrimPredicate::DEFAULT_PROXIES, |p| paths.push(p.to_string()))?;
    let mut l = Loader { stage, meters, up, world: HashMap::new() };

    let mut bodies: Vec<String> = vec![];
    let mut joints: Vec<JointPrim> = vec![];
    // Joints that close loops rather than grow the tree.
    let mut closures: Vec<JointPrim> = vec![];
    let mut colliders: Vec<String> = vec![];
    let mut frames: Vec<String> = vec![];
    // Joints a motor drives: MuJoCo actuators (directly or through a fixed tendon) and UsdPhysics
    // drives. A stage with neither drives every joint.
    let mut driven: HashSet<String> = HashSet::new();
    let mut any_drive = false;
    for p in &paths {
        let prim = l.stage.prim(p.as_str())?;
        let kind = prim.type_name()?.map(|t| t.as_str().to_string()).unwrap_or_default();
        let body = has_api(&prim, "PhysicsRigidBodyAPI")? && boolean(&prim, "physics:rigidBodyEnabled", true)?;
        let collider = has_api(&prim, "PhysicsCollisionAPI")?;
        if body {
            bodies.push(p.clone());
        }
        if collider && boolean(&prim, "physics:collisionEnabled", true)? {
            colliders.push(p.clone());
        }
        if kind == "Xform" && !body && !collider && prim.api_schemas()?.is_empty() && only_xform_children(&prim)? {
            frames.push(p.clone());
        }
        if kind == "MjcActuator" {
            any_drive = true;
            for t in prim.relationship("mjc:target").targets()? {
                let t = l.stage.prim(t.as_str())?;
                if t.type_name()?.is_some_and(|k| k.as_str() == "MjcTendon") {
                    driven.extend(t.relationship("mjc:path").targets()?.iter().map(|p| p.to_string()));
                } else {
                    driven.insert(t.path().to_string());
                }
            }
        }
        if prim.api_schemas()?.iter().any(|a| a.as_str().starts_with("PhysicsDriveAPI")) {
            any_drive = true;
            driven.insert(p.clone());
        }
        if kind.starts_with("Physics") && kind.ends_with("Joint") {
            if !boolean(&prim, "physics:jointEnabled", true)? {
                continue;
            }
            let joint = JointPrim {
                path: p.clone(),
                kind,
                body0: target(&prim, "physics:body0")?,
                body1: target(&prim, "physics:body1")?,
            };
            if boolean(&prim, "physics:excludeFromArticulation", false)? {
                closures.push(joint);
            } else {
                joints.push(joint);
            }
        }
    }
    let body_set: HashSet<&str> = bodies.iter().map(String::as_str).collect();
    let is_body = |p: &Option<String>| p.as_deref().is_some_and(|p| body_set.contains(p));
    // Unique link names: prim names, or full paths where those collide.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for b in bodies.iter().chain(&frames) {
        *counts.entry(name_of(b)).or_default() += 1;
    }
    let link_name = |p: &str| if counts.get(name_of(p)) == Some(&1) { name_of(p).to_string() } else { p.to_string() };
    let joint_name: HashMap<String, String> =
        joints.iter().map(|j| (j.path.clone(), name_of(&j.path).to_string())).collect();

    let mut links = vec![LinkDesc { name: WORLD.into(), ..Default::default() }];
    let mut out_joints = vec![];
    let mut placed: HashSet<String> = HashSet::new();
    let mut used = vec![false; joints.len()];
    loop {
        let mut progress = false;
        for (k, j) in joints.iter().enumerate() {
            if used[k] {
                continue;
            }
            let in_tree = |side: &Option<String>| match side {
                Some(b) if body_set.contains(b.as_str()) => placed.contains(b),
                _ => true,
            };
            let (parent_side, child_side, flipped) = match (in_tree(&j.body0), in_tree(&j.body1)) {
                (true, true) => {
                    // Closes a loop (or joins two world-fixed sides): not part of the tree.
                    closures.push(j.clone());
                    used[k] = true;
                    continue;
                }
                (true, false) => (&j.body0, &j.body1, false),
                (false, true) => (&j.body1, &j.body0, true),
                (false, false) => continue,
            };
            ensure!(is_body(child_side), "{}: joint has no rigid body to move", j.path);
            let prim = l.stage.prim(j.path.as_str())?;
            let (parent_local, child_local) = if flipped {
                (l.local_frame(&prim, 1)?, l.local_frame(&prim, 0)?)
            } else {
                (l.local_frame(&prim, 0)?, l.local_frame(&prim, 1)?)
            };
            let (parent, parent_frame) = match parent_side {
                Some(b) if body_set.contains(b.as_str()) => (link_name(b), DAffine3::IDENTITY),
                Some(b) => (WORLD.to_string(), l.body_frame(b)?),
                None => (WORLD.to_string(), DAffine3::from_quat(l.up)),
            };
            let child_path = child_side.as_ref().expect("checked above");
            let child = link_name(child_path);
            let joint =
                joint_from(&l, &prim, j, flipped, joint_name[&j.path].clone(), rigid(parent_frame * parent_local))?;
            let attach = if child_local.abs_diff_eq(DAffine3::IDENTITY, 1e-9) {
                JointDesc { parent, child: child.clone(), ..joint }
            } else {
                // The body sits away from the joint frame: a fixed child frame carries it.
                let frame = format!("{child}/{}", joint.name);
                links.push(LinkDesc { name: frame.clone(), ..Default::default() });
                out_joints.push(JointDesc { parent, child: frame.clone(), ..joint.clone() });
                JointDesc::fixed(&format!("{}_frame", joint.name), &frame, &child, rigid(child_local.inverse()))
            };
            out_joints.push(attach);
            links.push(LinkDesc { name: child, ..Default::default() });
            placed.insert(child_path.clone());
            used[k] = true;
            progress = true;
        }
        if progress {
            continue;
        }
        // No joint reaches the remaining bodies from the world: the highest of them stays where
        // it is authored, and the tree grows on from it.
        let Some(root) =
            bodies.iter().filter(|b| !placed.contains(*b)).min_by_key(|b| (b.matches('/').count(), (*b).clone()))
        else {
            break;
        };
        let frame = l.body_frame(root)?;
        out_joints.push(JointDesc::fixed(&format!("{}_fixed", link_name(root)), WORLD, &link_name(root), rigid(frame)));
        links.push(LinkDesc { name: link_name(root), ..Default::default() });
        placed.insert(root.clone());
    }
    // Mimic joints, actuation and springs, now that every joint has its name.
    for j in out_joints.iter_mut() {
        let Some(path) = joints.iter().find(|p| joint_name[&p.path] == j.name).map(|p| p.path.clone()) else {
            continue;
        };
        let prim = l.stage.prim(path.as_str())?;
        j.mimic = mimic(&prim, &joint_name, j.kind, l.meters)?;
        j.actuated = !any_drive || driven.contains(&path);
        // MjcJointAPI keeps MJCF's values: stiffness per radian or meter, springref in radians or meters.
        j.stiffness = scalar(&prim, "mjc:stiffness")?.unwrap_or(0.0) as f32;
        let spring_ref = scalar(&prim, "mjc:springref")?.unwrap_or(0.0);
        j.spring_ref = if j.kind == JointType::Prismatic { spring_ref * l.meters } else { spring_ref } as f32;
    }
    // Loop closures hold their joint's two anchor points together; a point is exact for spherical
    // joints and enough for planar loops.
    let mut loops = vec![];
    for c in &closures {
        let (Some(b0), Some(b1)) = (&c.body0, &c.body1) else { continue };
        if !(body_set.contains(b0.as_str()) && body_set.contains(b1.as_str())) {
            continue;
        }
        let prim = l.stage.prim(c.path.as_str())?;
        let anchor = |side: u8| -> Result<Vec3> { Ok(l.local_frame(&prim, side)?.translation.as_vec3()) };
        let anchors = (anchor(0)?, Some(anchor(1)?));
        loops.push(LoopClosure { name: c.path.clone(), links: [link_name(b0), link_name(b1)], anchors });
    }

    // Ghost-link frames, fixed to the nearest rigid body above them or to the world.
    let owner_of = |p: &str| bodies.iter().filter(|b| p.starts_with(&format!("{b}/"))).max_by_key(|b| b.len()).cloned();
    for f in &frames {
        let (parent, origin) = match owner_of(f) {
            Some(b) => (link_name(&b), l.body_frame(&b)?.inverse() * l.body_frame(f)?),
            None => (WORLD.to_string(), l.body_frame(f)?),
        };
        out_joints.push(JointDesc::fixed(&format!("{}_frame", link_name(f)), &parent, &link_name(f), rigid(origin)));
        links.push(LinkDesc { name: link_name(f), ..Default::default() });
    }

    let mut scene = vec![];
    for c in &colliders {
        let prim = l.stage.prim(c.as_str())?;
        let owner = if bodies.contains(c) { Some(c.clone()) } else { owner_of(c) };
        let shape = l.shape(&prim, owner.as_deref())?;
        match owner {
            Some(b) => links
                .iter_mut()
                .find(|link| link.name == link_name(&b))
                .expect("every body is a link")
                .collision
                .push(shape),
            None => scene.push(shape),
        }
    }
    Ok(Model { robot: RobotDescription { name: robot_name, links, joints: out_joints, loops }, scene })
}

/// The joint's kind, axis, limits and velocity limit; parent, child and mimic are filled in later.
fn joint_from(
    l: &Loader,
    prim: &Prim,
    j: &JointPrim,
    flipped: bool,
    name: String,
    origin: Transform,
) -> Result<JointDesc> {
    let sign = if flipped { -1.0 } else { 1.0 };
    let unit_axis =
        || -> Result<Vec3> { Ok((axis(token(prim, "physics:axis")?.as_deref().unwrap_or("X"))? * sign).as_vec3()) };
    let limit = |name: &str| scalar(prim, name).map(|v| v.filter(|v| v.is_finite() && v.abs() < 1e30));
    let velocity = scalar(prim, "newton:velocityLimit")?
        .or(scalar(prim, "physxJoint:maxJointVelocity")?)
        .filter(|v| v.is_finite());
    let (kind, axis, lower, upper, max_velocity) = match j.kind.as_str() {
        "PhysicsFixedJoint" => (JointType::Fixed, Vec3::Z, 0.0, 0.0, f32::INFINITY),
        "PhysicsRevoluteJoint" => {
            let v = velocity.map_or(f32::INFINITY, |v| v.to_radians() as f32);
            match (limit("physics:lowerLimit")?, limit("physics:upperLimit")?) {
                (Some(lo), Some(hi)) => {
                    (JointType::Revolute, unit_axis()?, lo.to_radians() as f32, hi.to_radians() as f32, v)
                }
                _ => (JointType::Continuous, unit_axis()?, 0.0, 0.0, v),
            }
        }
        "PhysicsPrismaticJoint" => {
            let (lo, hi) = (limit("physics:lowerLimit")?, limit("physics:upperLimit")?);
            let (Some(lo), Some(hi)) = (lo, hi) else { bail!("{}: prismatic joints need limits", j.path) };
            let v = velocity.map_or(f32::INFINITY, |v| (v * l.meters) as f32);
            (JointType::Prismatic, unit_axis()?, (lo * l.meters) as f32, (hi * l.meters) as f32, v)
        }
        "PhysicsSphericalJoint" => {
            // Cone limits are degrees from the joint axis; a missing or negative one leaves its
            // direction free.
            let upper = match [limit("physics:coneAngle0Limit")?, limit("physics:coneAngle1Limit")?] {
                [Some(a), Some(b)] if a >= 0.0 && b >= 0.0 => a.max(b).to_radians() as f32,
                _ => f32::INFINITY,
            };
            (JointType::Ball, Vec3::Z, 0.0, upper, velocity.map_or(f32::INFINITY, |v| v.to_radians() as f32))
        }
        other => bail!("{}: joint type {other} is not supported", j.path),
    };
    Ok(JointDesc {
        name,
        kind,
        parent: String::new(),
        child: String::new(),
        origin,
        axis,
        lower,
        upper,
        max_velocity,
        max_acceleration: f32::INFINITY,
        max_jerk: f32::INFINITY,
        mimic: None,
        actuated: true,
        stiffness: 0.0,
        spring_ref: 0.0,
    })
}

/// `NewtonMimicAPI` (follower = coef0 + coef1 * leader, coef0 in the follower's units, degrees for
/// revolute joints) or `PhysxMimicJointAPI:<axis>` (follower + gearing * reference + offset = 0).
fn mimic(prim: &Prim, names: &HashMap<String, String>, kind: JointType, meters: f64) -> Result<Option<Mimic>> {
    let leader = |path: Option<String>| -> Result<Option<String>> {
        Ok(match path {
            Some(p) => {
                Some(names.get(&p).with_context(|| format!("{}: mimics unknown joint {p}", prim.path()))?.clone())
            }
            None => None,
        })
    };
    if has_api(prim, "NewtonMimicAPI")? && boolean(prim, "newton:mimicEnabled", true)? {
        let Some(joint) = leader(target(prim, "newton:mimicJoint")?)? else { return Ok(None) };
        let offset = scalar(prim, "newton:mimicCoef0")?.unwrap_or(0.0);
        let offset = if kind == JointType::Prismatic { offset * meters } else { offset.to_radians() };
        let multiplier = scalar(prim, "newton:mimicCoef1")?.unwrap_or(1.0);
        return Ok(Some(Mimic { joint, curve: Curve::linear(multiplier as f32, offset as f32) }));
    }
    for api in prim.api_schemas()? {
        let Some(instance) = api.as_str().strip_prefix("PhysxMimicJointAPI:") else { continue };
        let Some(joint) = leader(target(prim, &format!("physxMimicJoint:{instance}:referenceJoint"))?)? else {
            continue;
        };
        let gearing = scalar(prim, &format!("physxMimicJoint:{instance}:gearing"))?.unwrap_or(1.0);
        let offset = scalar(prim, &format!("physxMimicJoint:{instance}:offset"))?.unwrap_or(0.0);
        return Ok(Some(Mimic { joint, curve: Curve::linear(-gearing as f32, -offset as f32) }));
    }
    Ok(None)
}
