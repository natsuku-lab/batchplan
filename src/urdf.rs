//! URDF (and SRDF) loading into a [`RobotDescription`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::Vec3;

use crate::description::{Curve, Geometry, JointDesc, JointType, LinkDesc, Mimic, RobotDescription, Shape};
use crate::robot::Transform;

pub(crate) fn load(path: &Path, package_dirs: &[PathBuf]) -> Result<RobotDescription> {
    let urdf = urdf_rs::read_file(path).with_context(|| format!("parsing {}", path.display()))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let shapes = |elements: Vec<(&urdf_rs::Pose, &urdf_rs::Geometry)>| -> Result<Vec<Shape>> {
        elements
            .into_iter()
            .map(|(origin, geometry)| {
                Ok(Shape { origin: pose(origin), geometry: geometry_of(geometry, dir, package_dirs)? })
            })
            .collect()
    };
    let links = urdf
        .links
        .iter()
        .map(|l| {
            Ok(LinkDesc {
                name: l.name.clone(),
                collision: shapes(l.collision.iter().map(|c| (&c.origin, &c.geometry)).collect())?,
                visual: shapes(l.visual.iter().map(|v| (&v.origin, &v.geometry)).collect())?,
            })
        })
        .collect::<Result<_>>()?;
    let joints = urdf
        .joints
        .iter()
        .map(|j| {
            use urdf_rs::JointType as J;
            let kind = match j.joint_type {
                J::Fixed => JointType::Fixed,
                J::Revolute => JointType::Revolute,
                J::Continuous => JointType::Continuous,
                J::Prismatic => JointType::Prismatic,
                J::Planar => JointType::Planar,
                J::Floating => JointType::Floating,
                ref other => bail!("joint '{}' has unsupported type {other:?}", j.name),
            };
            let (lower, upper) = (j.limit.lower as f32, j.limit.upper as f32);
            if matches!(kind, JointType::Revolute | JointType::Prismatic) && !(lower.is_finite() && upper.is_finite()) {
                bail!("joint '{}' needs finite position limits", j.name);
            }
            let limit = |v: f64| if v.is_finite() && v > 0.0 { v as f32 } else { f32::INFINITY };
            Ok(JointDesc {
                name: j.name.clone(),
                kind,
                parent: j.parent.link.clone(),
                child: j.child.link.clone(),
                origin: pose(&j.origin),
                axis: Vec3::from_array(j.axis.xyz.0.map(|v| v as f32)).normalize_or_zero(),
                lower,
                upper,
                max_velocity: limit(j.limit.velocity),
                max_acceleration: limit(j.limit.acceleration),
                max_jerk: limit(j.limit.jerk),
                mimic: j.mimic.as_ref().map(|m| Mimic {
                    joint: m.joint.clone(),
                    curve: Curve::linear(m.multiplier.unwrap_or(1.0) as f32, m.offset.unwrap_or(0.0) as f32),
                }),
                actuated: true,
                stiffness: 0.0,
                spring_ref: 0.0,
            })
        })
        .collect::<Result<_>>()?;
    Ok(RobotDescription { name: urdf.name, links, joints, loops: vec![] })
}

fn pose(p: &urdf_rs::Pose) -> Transform {
    Transform::from_xyz_rpy(p.xyz.0, p.rpy.0)
}

fn geometry_of(g: &urdf_rs::Geometry, dir: &Path, package_dirs: &[PathBuf]) -> Result<Geometry> {
    use urdf_rs::Geometry as G;
    Ok(match *g {
        G::Box { size } => Geometry::Box { half: Vec3::from_array(size.0.map(|v| v as f32)) * 0.5 },
        G::Sphere { radius } => Geometry::Sphere { radius: radius as f32 },
        G::Cylinder { radius, length } => {
            Geometry::Cylinder { radius: radius as f32, half_length: length as f32 * 0.5 }
        }
        G::Capsule { radius, length } => Geometry::Capsule { radius: radius as f32, half_length: length as f32 * 0.5 },
        G::Mesh { ref filename, scale } => Geometry::Mesh {
            path: resolve(filename, dir, package_dirs)?,
            scale: scale.map_or(Vec3::ONE, |s| Vec3::from_array(s.0.map(|v| v as f32))),
        },
    })
}

/// Resolves a mesh reference: `package://name/rest` through `package_dirs` (each either the
/// package itself or a directory holding it) and then the directories above the URDF;
/// `file://` and absolute paths as given; anything else relative to the URDF.
fn resolve(filename: &str, dir: &Path, package_dirs: &[PathBuf]) -> Result<PathBuf> {
    let Some(rest) = filename.strip_prefix("package://") else {
        let path = Path::new(filename.strip_prefix("file://").unwrap_or(filename));
        return Ok(if path.is_absolute() { path.to_path_buf() } else { dir.join(path) });
    };
    let (package, relative) = rest.split_once('/').with_context(|| format!("malformed mesh path {filename}"))?;
    let candidates = package_dirs.iter().map(PathBuf::as_path).chain(dir.ancestors());
    for base in candidates {
        for root in [base.to_path_buf(), base.join(package)] {
            if package_name(&root).as_deref() == Some(package) {
                return Ok(root.join(relative));
            }
        }
    }
    bail!("package '{package}' (for {filename}) not found; add the directory holding it to RobotOptions::package_dirs")
}

/// The `<name>` declared in `dir/package.xml`, if there is one.
fn package_name(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("package.xml")).ok()?;
    let doc = roxmltree::Document::parse(&text).ok()?;
    let name = doc.root_element().children().find(|n| n.has_tag_name("name"))?;
    Some(name.text()?.trim().to_string())
}

/// Link pairs an SRDF disables for collision checking (`<disable_collisions link1 link2>`).
pub(crate) fn srdf_disabled_pairs(path: &Path) -> Result<Vec<(String, String)>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let doc = roxmltree::Document::parse(&text).with_context(|| format!("parsing {}", path.display()))?;
    doc.root_element()
        .children()
        .filter(|n| n.has_tag_name("disable_collisions"))
        .map(|n| {
            let link = |attr: &str| {
                n.attribute(attr)
                    .map(str::to_string)
                    .with_context(|| format!("{}: disable_collisions without {attr}", path.display()))
            };
            Ok((link("link1")?, link("link2")?))
        })
        .collect()
}
