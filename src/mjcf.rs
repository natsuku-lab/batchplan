//! MJCF (MuJoCo XML) loading. The robot becomes a [`RobotDescription`]; static geometry becomes
//! scene shapes. Covers what kinematics and collision need: `<include>`, `<default>` classes and
//! `childclass`, the compiler's angle units, mesh directory and Euler sequence, every orientation
//! form, bodies with several joints, free (`freejoint` or `type="free"`) and ball joints,
//! `fromto`, and `<equality><joint>` as mimic joints.
//!
//! Body subtrees that contain a joint are the robot; bodies with no joint anywhere below them, and
//! geoms placed directly in the world body, are the scene. `connect` equalities close kinematic
//! loops, which `loops.rs` solves; joints that actuators drive (directly or through a fixed tendon)
//! drive them. `weld` equalities are ignored.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use glam::{Mat3, Quat, Vec3};

use crate::description::{
    Curve, Geometry, JointDesc, JointType, LinkDesc, LoopClosure, Mimic, Model, RobotDescription, Shape,
};
use crate::robot::Transform;

/// The world body's link name.
pub(crate) const WORLD: &str = "world";

/// An XML element with `<include>`s spliced in.
struct Element {
    tag: String,
    attrs: Vec<(String, String)>,
    children: Vec<Element>,
}

impl Element {
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    fn children<'a>(&'a self, tag: &'a str) -> impl Iterator<Item = &'a Element> + 'a {
        self.children.iter().filter(move |c| c.tag == tag)
    }
}

/// Reads `path`, splicing the children of every `<include file>` (relative to `root_dir`, the
/// main model's directory) in its place.
fn read(path: &Path, root_dir: &Path, depth: usize) -> Result<Element> {
    ensure!(depth < 16, "includes nest too deeply at {}", path.display());
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let doc = roxmltree::Document::parse(&text).with_context(|| format!("parsing {}", path.display()))?;
    convert(doc.root_element(), root_dir, depth)
}

fn convert(node: roxmltree::Node, root_dir: &Path, depth: usize) -> Result<Element> {
    let mut children = vec![];
    for c in node.children().filter(|c| c.is_element()) {
        if c.has_tag_name("include") {
            let file = c.attribute("file").context("<include> without file")?;
            children.extend(read(&root_dir.join(file), root_dir, depth + 1)?.children);
        } else {
            children.push(convert(c, root_dir, depth)?);
        }
    }
    Ok(Element {
        tag: node.tag_name().name().to_string(),
        attrs: node.attributes().map(|a| (a.name().to_string(), a.value().to_string())).collect(),
        children,
    })
}

/// `<compiler>` settings that change how numbers are read.
struct Compiler {
    degrees: bool,
    eulerseq: Vec<char>,
    meshdir: PathBuf,
}

/// Attribute defaults per class and element tag; "main" is the root class.
struct Defaults {
    parent: HashMap<String, String>,
    attrs: HashMap<(String, String), Vec<(String, String)>>,
}

impl Defaults {
    fn collect(model: &Element) -> Result<Self> {
        let mut d = Defaults { parent: HashMap::new(), attrs: HashMap::new() };
        for top in model.children("default") {
            d.add(top, "main", None)?;
        }
        Ok(d)
    }

    fn add(&mut self, el: &Element, class: &str, parent: Option<&str>) -> Result<()> {
        if let Some(p) = parent {
            self.parent.insert(class.to_string(), p.to_string());
        }
        for c in &el.children {
            if c.tag == "default" {
                let name = c.attr("class").context("nested <default> without class")?;
                self.add(c, name, Some(class))?;
            } else {
                self.attrs.entry((class.to_string(), c.tag.clone())).or_default().extend(c.attrs.iter().cloned());
            }
        }
        Ok(())
    }

    /// The element's attributes over its class's defaults (and the class's ancestors').
    fn resolve(&self, el: &Element, inherited: &str) -> Result<Vec<(String, String)>> {
        let class = el.attr("class").unwrap_or(inherited);
        let mut chain = vec![class.to_string()];
        while let Some(p) = self.parent.get(chain.last().expect("non-empty")) {
            chain.push(p.clone());
        }
        ensure!(class == "main" || self.parent.contains_key(class), "unknown default class '{class}'");
        let mut out: Vec<(String, String)> = vec![];
        for c in chain.iter().rev() {
            for (k, v) in self.attrs.get(&(c.clone(), el.tag.clone())).into_iter().flatten() {
                set(&mut out, k, v);
            }
        }
        for (k, v) in &el.attrs {
            set(&mut out, k, v);
        }
        Ok(out)
    }
}

fn set(attrs: &mut Vec<(String, String)>, k: &str, v: &str) {
    match attrs.iter_mut().find(|(key, _)| key == k) {
        Some(entry) => entry.1 = v.to_string(),
        None => attrs.push((k.to_string(), v.to_string())),
    }
}

/// Resolved attributes of one element.
struct Attrs(Vec<(String, String)>);

impl Attrs {
    fn get(&self, name: &str) -> Option<&str> {
        self.0.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    fn floats(&self, name: &str) -> Result<Option<Vec<f32>>> {
        self.get(name)
            .map(|v| {
                v.split_whitespace()
                    .map(|x| x.parse::<f32>().with_context(|| format!("'{name}' is not a list of numbers: {v}")))
                    .collect()
            })
            .transpose()
    }

    fn vec3(&self, name: &str) -> Result<Option<Vec3>> {
        match self.floats(name)? {
            Some(v) if v.len() == 3 => Ok(Some(Vec3::new(v[0], v[1], v[2]))),
            Some(v) => bail!("'{name}' needs 3 numbers, got {}", v.len()),
            None => Ok(None),
        }
    }
}

impl Compiler {
    fn angle(&self, v: f32) -> f32 {
        if self.degrees { v.to_radians() } else { v }
    }

    /// The frame an element's `pos` and orientation attributes describe.
    fn frame(&self, a: &Attrs) -> Result<Transform> {
        let trans = a.vec3("pos")?.unwrap_or(Vec3::ZERO);
        Ok(Transform { rot: Mat3::from_quat(self.orientation(a)?), trans })
    }

    fn orientation(&self, a: &Attrs) -> Result<Quat> {
        if let Some(q) = a.floats("quat")? {
            ensure!(q.len() == 4, "quat needs 4 numbers");
            return Ok(Quat::from_xyzw(q[1], q[2], q[3], q[0]).normalize());
        }
        if let Some(v) = a.floats("axisangle")? {
            ensure!(v.len() == 4, "axisangle needs 4 numbers");
            return Ok(Quat::from_axis_angle(Vec3::new(v[0], v[1], v[2]).normalize(), self.angle(v[3])));
        }
        if let Some(v) = a.floats("euler")? {
            ensure!(v.len() == 3, "euler needs 3 numbers");
            let mut q = Quat::IDENTITY;
            for (&axis, &angle) in self.eulerseq.iter().zip(&v) {
                let r = match axis.to_ascii_lowercase() {
                    'x' => Quat::from_rotation_x(self.angle(angle)),
                    'y' => Quat::from_rotation_y(self.angle(angle)),
                    'z' => Quat::from_rotation_z(self.angle(angle)),
                    other => bail!("eulerseq has invalid axis '{other}'"),
                };
                // Lowercase axes rotate with the frame (intrinsic), uppercase ones stay fixed.
                q = if axis.is_ascii_lowercase() { q * r } else { r * q };
            }
            return Ok(q);
        }
        if let Some(v) = a.floats("xyaxes")? {
            ensure!(v.len() == 6, "xyaxes needs 6 numbers");
            let x = Vec3::new(v[0], v[1], v[2]).normalize();
            let y = (Vec3::new(v[3], v[4], v[5]) - x * x.dot(Vec3::new(v[3], v[4], v[5]))).normalize();
            return Ok(Quat::from_mat3(&Mat3::from_cols(x, y, x.cross(y))));
        }
        if let Some(z) = a.vec3("zaxis")? {
            return Ok(Quat::from_rotation_arc(Vec3::Z, z.normalize()));
        }
        Ok(Quat::IDENTITY)
    }
}

pub(crate) fn load(path: &Path) -> Result<Model> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let model = read(path, dir, 0)?;
    ensure!(model.tag == "mujoco", "{}: the root element must be <mujoco>", path.display());
    let mut compiler = Compiler { degrees: true, eulerseq: "xyz".chars().collect(), meshdir: dir.to_path_buf() };
    for c in model.children("compiler") {
        if let Some(angle) = c.attr("angle") {
            compiler.degrees = match angle {
                "degree" => true,
                "radian" => false,
                other => bail!("unknown compiler angle '{other}'"),
            };
        }
        if let Some(seq) = c.attr("eulerseq") {
            compiler.eulerseq = seq.chars().collect();
            ensure!(compiler.eulerseq.len() == 3, "eulerseq needs 3 axes");
        }
        if let Some(d) = c.attr("meshdir").or(c.attr("assetdir")) {
            compiler.meshdir = dir.join(d);
        }
    }
    let defaults = Defaults::collect(&model)?;
    let mut meshes = HashMap::new();
    for asset in model.children("asset") {
        for m in asset.children("mesh") {
            let a = Attrs(defaults.resolve(m, "main")?);
            let file = a.get("file").context("only meshes from files are supported")?;
            let name = a.get("name").map(str::to_string).unwrap_or_else(|| {
                Path::new(file).file_stem().map_or(file.to_string(), |s| s.to_string_lossy().into_owned())
            });
            let scale = a.vec3("scale")?.unwrap_or(Vec3::ONE);
            meshes.insert(name, (compiler.meshdir.join(file), scale));
        }
    }
    let mut out = Builder {
        compiler: &compiler,
        defaults: &defaults,
        meshes: &meshes,
        links: vec![LinkDesc { name: WORLD.into(), ..Default::default() }],
        joints: vec![],
        scene: vec![],
    };
    for world in model.children("worldbody") {
        for g in world.children("geom") {
            if let Some((true, shape)) = out.geom(g, "main")? {
                out.scene.push(shape);
            }
        }
        for body in world.children("body") {
            if has_joint(body) {
                out.body(body, WORLD, Transform::IDENTITY, "main")?;
            } else {
                out.static_body(body, Transform::IDENTITY, "main")?;
            }
        }
    }
    let mut loops = vec![];
    for eq in model.children("equality") {
        for j in eq.children("joint") {
            out.mimic(j)?;
        }
        for (k, c) in eq.children("connect").enumerate() {
            let a = Attrs(c.attrs.clone());
            let (Some(body1), Some(body2)) = (a.get("body1"), a.get("body2")) else {
                bail!("connect equalities between sites are not supported; name body1 and body2");
            };
            let anchor = a.vec3("anchor")?.context("connect equality without an anchor")?;
            let name = a.get("name").map_or_else(|| format!("connect {k} ({body1}, {body2})"), str::to_string);
            loops.push(LoopClosure { name, links: [body1.to_string(), body2.to_string()], anchors: (anchor, None) });
        }
    }
    // Joints that actuators drive, directly or through a fixed tendon. A model without actuators
    // drives every joint.
    let mut tendons: HashMap<&str, Vec<&str>> = HashMap::new();
    for t in model.children("tendon") {
        for fixed in t.children("fixed") {
            let joints = fixed.children("joint").filter_map(|j| j.attr("joint")).collect();
            tendons.insert(fixed.attr("name").unwrap_or_default(), joints);
        }
    }
    let mut driven: Vec<&str> = vec![];
    let mut any_actuator = false;
    for actuators in model.children("actuator") {
        for act in &actuators.children {
            any_actuator = true;
            driven.extend(act.attr("joint"));
            driven.extend(act.attr("tendon").and_then(|t| tendons.get(t)).into_iter().flatten());
        }
    }
    for j in &mut out.joints {
        j.actuated = !any_actuator || driven.contains(&j.name.as_str());
    }
    let name = model.attr("model").unwrap_or("mjcf").to_string();
    let robot = RobotDescription { name, links: out.links, joints: out.joints, loops };
    Ok(Model { robot, scene: out.scene })
}

fn has_joint(body: &Element) -> bool {
    body.children.iter().any(|c| matches!(c.tag.as_str(), "joint" | "freejoint"))
        || body.children("body").any(has_joint)
}

struct Builder<'a> {
    compiler: &'a Compiler,
    defaults: &'a Defaults,
    meshes: &'a HashMap<String, (PathBuf, Vec3)>,
    links: Vec<LinkDesc>,
    joints: Vec<JointDesc>,
    scene: Vec<Shape>,
}

impl Builder<'_> {
    /// Adds `body` below link `parent`, `offset` being the parent body's frame in that link.
    fn body(&mut self, body: &Element, parent: &str, offset: Transform, inherited: &str) -> Result<()> {
        let name = body.attr("name").context("robot bodies need names")?.to_string();
        let class = body.attr("childclass").unwrap_or(inherited).to_string();
        let a = Attrs(body.attrs.clone());
        let frame = offset.mul(&self.compiler.frame(&a)?);
        // Each joint turns about its anchor in the frame the previous joints left behind, so the
        // body becomes a chain: joint i's frame sits at anchor i, the body frame at -anchor.
        let joints: Vec<&Element> =
            body.children.iter().filter(|c| matches!(c.tag.as_str(), "joint" | "freejoint")).collect();
        let (mut link, mut origin, mut anchor) = (parent.to_string(), frame, Vec3::ZERO);
        for (i, j) in joints.iter().enumerate() {
            let ja = Attrs(self.defaults.resolve(j, &class)?);
            let joint_name = ja.get("name").map(str::to_string).unwrap_or_else(|| format!("{name}_joint{i}"));
            let pos = ja.vec3("pos")?.unwrap_or(Vec3::ZERO);
            let last = i + 1 == joints.len();
            let child = if last && pos == Vec3::ZERO { name.clone() } else { format!("{name}/{joint_name}") };
            let kind_name = if j.tag == "freejoint" { "free" } else { ja.get("type").unwrap_or("hinge") };
            let range = ja.floats("range")?.unwrap_or(vec![0.0, 0.0]);
            ensure!(range.len() == 2, "joint '{joint_name}' range needs 2 numbers");
            let limited = match ja.get("limited").unwrap_or("auto") {
                "true" => true,
                "false" => false,
                _ => ja.get("range").is_some() && range[0] < range[1],
            };
            let (kind, lower, upper) = match (kind_name, limited) {
                ("hinge", true) => (JointType::Revolute, self.compiler.angle(range[0]), self.compiler.angle(range[1])),
                ("hinge", false) => (JointType::Continuous, 0.0, 0.0),
                ("slide", true) => (JointType::Prismatic, range[0], range[1]),
                ("slide", false) => bail!("slide joint '{joint_name}' needs a range"),
                // A ball joint's range is [0, largest rotation angle].
                ("ball", true) => (JointType::Ball, 0.0, self.compiler.angle(range[1])),
                ("ball", false) => (JointType::Ball, 0.0, f32::INFINITY),
                ("free", _) => (JointType::Floating, f32::NAN, f32::NAN),
                (other, _) => bail!("joint '{joint_name}' has unsupported type '{other}'"),
            };
            let scalar =
                |name: &str| -> Result<f32> { Ok(ja.floats(name)?.and_then(|v| v.first().copied()).unwrap_or(0.0)) };
            let spring_ref = scalar("springref")?;
            self.links.push(LinkDesc { name: child.clone(), ..Default::default() });
            self.joints.push(JointDesc {
                name: joint_name,
                kind,
                parent: link.clone(),
                child: child.clone(),
                origin: origin.mul(&Transform { rot: Mat3::IDENTITY, trans: pos - anchor }),
                axis: ja.vec3("axis")?.unwrap_or(Vec3::Z).normalize(),
                lower,
                upper,
                max_velocity: f32::INFINITY,
                max_acceleration: f32::INFINITY,
                max_jerk: f32::INFINITY,
                mimic: None,
                // Set once the actuators are read.
                actuated: false,
                stiffness: scalar("stiffness")?,
                spring_ref: if kind_name == "slide" { spring_ref } else { self.compiler.angle(spring_ref) },
            });
            (link, origin, anchor) = (child, Transform::IDENTITY, pos);
        }
        if joints.is_empty() || anchor != Vec3::ZERO {
            self.links.push(LinkDesc { name: name.clone(), ..Default::default() });
            self.joints.push(JointDesc::fixed(
                &format!("{name}_frame"),
                &link,
                &name,
                origin.mul(&translation(-anchor)),
            ));
        }
        for g in body.children("geom") {
            if let Some((collision, shape)) = self.geom(g, &class)? {
                let desc = self.links.iter_mut().rev().find(|l| l.name == name).expect("added above");
                if collision { desc.collision.push(shape) } else { desc.visual.push(shape) }
            }
        }
        for child in body.children("body") {
            self.body(child, &name, Transform::IDENTITY, &class)?;
        }
        Ok(())
    }

    /// Adds the collision geometry of a jointless body subtree to the scene.
    fn static_body(&mut self, body: &Element, parent: Transform, inherited: &str) -> Result<()> {
        let class = body.attr("childclass").unwrap_or(inherited).to_string();
        let frame = parent.mul(&self.compiler.frame(&Attrs(body.attrs.clone()))?);
        for g in body.children("geom") {
            if let Some((true, shape)) = self.geom(g, &class)? {
                self.scene.push(Shape { origin: frame.mul(&shape.origin), geometry: shape.geometry });
            }
        }
        for child in body.children("body") {
            self.static_body(child, frame, &class)?;
        }
        Ok(())
    }

    /// A geom as (collides, shape in its body frame); `None` for kinds without volume.
    fn geom(&self, g: &Element, class: &str) -> Result<Option<(bool, Shape)>> {
        let a = Attrs(self.defaults.resolve(g, class)?);
        let kind = a.get("type").unwrap_or("sphere");
        let size = a.floats("size")?.unwrap_or_default();
        let size_at = |i: usize| size.get(i).copied().with_context(|| format!("{kind} geom needs size[{i}]"));
        let collides = |name: &str| a.get(name).map_or(Ok(1), |v| v.parse::<i64>()).map(|v| v != 0);
        let collision = collides("contype")? || collides("conaffinity")?;
        let mut origin = self.compiler.frame(&a)?;
        // `fromto` replaces the frame: centered on the segment, z along it.
        let mut half_length = None;
        if let Some(ft) = a.floats("fromto")? {
            ensure!(ft.len() == 6, "fromto needs 6 numbers");
            let (from, to) = (Vec3::new(ft[0], ft[1], ft[2]), Vec3::new(ft[3], ft[4], ft[5]));
            let along = to - from;
            origin = Transform {
                rot: Mat3::from_quat(Quat::from_rotation_arc(Vec3::Z, along.normalize())),
                trans: 0.5 * (from + to),
            };
            half_length = Some(0.5 * along.length());
        }
        let geometry = match kind {
            "sphere" => Geometry::Sphere { radius: size_at(0)? },
            "capsule" => {
                Geometry::Capsule { radius: size_at(0)?, half_length: half_length.map_or_else(|| size_at(1), Ok)? }
            }
            "cylinder" => {
                Geometry::Cylinder { radius: size_at(0)?, half_length: half_length.map_or_else(|| size_at(1), Ok)? }
            }
            "box" => match half_length {
                Some(h) => Geometry::Box { half: Vec3::new(size_at(0)?, size_at(1)?, h) },
                None => Geometry::Box { half: Vec3::new(size_at(0)?, size_at(1)?, size_at(2)?) },
            },
            "ellipsoid" => Geometry::Ellipsoid { radii: Vec3::new(size_at(0)?, size_at(1)?, size_at(2)?) },
            "plane" => Geometry::Plane,
            "mesh" => {
                let name = a.get("mesh").context("mesh geom without mesh")?;
                let (path, scale) = self.meshes.get(name).with_context(|| format!("unknown mesh '{name}'"))?;
                Geometry::ConvexHull(Box::new(Geometry::Mesh { path: path.clone(), scale: *scale }))
            }
            other => bail!("geom type '{other}' is not supported"),
        };
        Ok(Some((collision, Shape { origin, geometry })))
    }

    /// `<equality><joint joint1 joint2 polycoef>`: joint1 = c0 + c1 * joint2.
    fn mimic(&mut self, eq: &Element) -> Result<()> {
        let follower = eq.attr("joint1").context("joint equality without joint1")?;
        let leader = eq.attr("joint2").context("joint equalities fixing a joint are not supported; lock it instead")?;
        let c = Attrs(eq.attrs.clone()).floats("polycoef")?.unwrap_or(vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        ensure!(c.len() == 5, "joint equality polycoef needs 5 numbers");
        let joint = self
            .joints
            .iter_mut()
            .find(|j| j.name == follower)
            .with_context(|| format!("unknown joint '{follower}'"))?;
        joint.mimic = Some(Mimic { joint: leader.to_string(), curve: Curve([c[0], c[1], c[2], c[3], c[4]]) });
        Ok(())
    }
}

fn translation(t: Vec3) -> Transform {
    Transform { rot: Mat3::IDENTITY, trans: t }
}
