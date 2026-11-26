//! Format-agnostic robot descriptions. Every loader produces a [`RobotDescription`]; kinematics,
//! collision spheres and self-collision analysis are built from it, so they never depend on the
//! file format a robot came from.

use std::f32::consts::{PI, TAU};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use glam::Vec3;
use mesh_loader::Loader;

use crate::robot::Transform;

/// A model file's robot and scene.
pub(crate) struct Model {
    pub(crate) robot: RobotDescription,
    /// Static collision geometry, with origins in the world frame.
    pub(crate) scene: Vec<Shape>,
}

#[derive(Clone, Debug)]
pub(crate) struct RobotDescription {
    pub(crate) name: String,
    pub(crate) links: Vec<LinkDesc>,
    pub(crate) joints: Vec<JointDesc>,
    /// Kinematic loops the tree of joints leaves open.
    pub(crate) loops: Vec<LoopClosure>,
}

/// A point on one link held to a point on another: a kinematic loop a tree cannot express (MJCF
/// `connect`, USD joints that close a loop).
#[derive(Clone, Debug)]
pub(crate) struct LoopClosure {
    /// Where it was declared, for messages.
    pub(crate) name: String,
    pub(crate) links: [String; 2],
    /// The point in the first link's frame, and in the second's; `None` there means wherever the
    /// first point is with every joint at zero, as MuJoCo's `connect` defines it.
    pub(crate) anchors: (Vec3, Option<Vec3>),
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LinkDesc {
    pub(crate) name: String,
    pub(crate) collision: Vec<Shape>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JointType {
    Fixed,
    Revolute,
    /// Revolute without position limits.
    Continuous,
    Prismatic,
    /// Motion in the plane perpendicular to `axis`: two translations and a rotation about `axis`.
    Planar,
    /// Any rotation about the joint origin; `upper`, when finite, bounds each rotation angle.
    Ball,
    /// Free motion of the child relative to the parent.
    Floating,
}

#[derive(Clone, Debug)]
pub(crate) struct JointDesc {
    pub(crate) name: String,
    pub(crate) kind: JointType,
    pub(crate) parent: String,
    pub(crate) child: String,
    /// Joint frame relative to the parent link frame; the child link frame at zero motion.
    pub(crate) origin: Transform,
    /// Unit motion axis in the joint frame.
    pub(crate) axis: Vec3,
    pub(crate) lower: f32,
    pub(crate) upper: f32,
    /// Velocity, acceleration and jerk limits; infinite when the description gives none.
    pub(crate) max_velocity: f32,
    pub(crate) max_acceleration: f32,
    pub(crate) max_jerk: f32,
    pub(crate) mimic: Option<Mimic>,
    /// Whether a motor drives the joint. Loops are driven by their actuated joints; elsewhere
    /// every joint is planned.
    pub(crate) actuated: bool,
    /// A spring toward `spring_ref` (MJCF `stiffness`, `springref`): what holds a loop's passive
    /// joints where closure alone does not.
    pub(crate) stiffness: f32,
    pub(crate) spring_ref: f32,
}

impl JointDesc {
    pub(crate) fn fixed(name: &str, parent: &str, child: &str, origin: Transform) -> Self {
        JointDesc {
            name: name.to_string(),
            kind: JointType::Fixed,
            parent: parent.to_string(),
            child: child.to_string(),
            origin,
            axis: Vec3::Z,
            lower: 0.0,
            upper: 0.0,
            max_velocity: f32::INFINITY,
            max_acceleration: f32::INFINITY,
            max_jerk: f32::INFINITY,
            mimic: None,
            actuated: false,
            stiffness: 0.0,
            spring_ref: 0.0,
        }
    }
}

/// `value = curve(value(joint))`.
#[derive(Clone, Debug)]
pub(crate) struct Mimic {
    pub(crate) joint: String,
    pub(crate) curve: Curve,
}

/// A polynomial of degree at most four, `c[0] + c[1] x + ... + c[4] x⁴`: how a mimic joint follows
/// its leader (linear for URDF and USD mimics, quartic for MJCF joint equalities and closed loops).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Curve(pub(crate) [f32; 5]);

impl Curve {
    pub(crate) const IDENTITY: Curve = Curve([0.0, 1.0, 0.0, 0.0, 0.0]);

    pub(crate) fn linear(multiplier: f32, offset: f32) -> Curve {
        Curve([offset, multiplier, 0.0, 0.0, 0.0])
    }

    #[inline]
    pub(crate) fn value(&self, x: f32) -> f32 {
        let c = self.0;
        c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])))
    }

    /// d value / dx.
    #[inline]
    pub(crate) fn slope(&self, x: f32) -> f32 {
        let c = self.0;
        c[1] + x * (2.0 * c[2] + x * (3.0 * c[3] + x * 4.0 * c[4]))
    }

    /// `(value(x), slope(x))`, with a shortcut for linear curves (most joints).
    #[inline]
    pub(crate) fn value_and_slope(&self, x: f32) -> (f32, f32) {
        let c = self.0;
        if c[2] == 0.0 && c[3] == 0.0 && c[4] == 0.0 {
            (c[1] * x + c[0], c[1])
        } else {
            (self.value(x), self.slope(x))
        }
    }

    pub(crate) fn is_linear(&self) -> bool {
        self.0[2..] == [0.0; 3]
    }

    /// `self(inner(x))`, or `None` when that has degree above four.
    pub(crate) fn after(&self, inner: &Curve) -> Option<Curve> {
        let mul = |a: &[f64], b: &[f64]| {
            let mut out = vec![0.0; a.len() + b.len() - 1];
            for (i, x) in a.iter().enumerate() {
                for (j, y) in b.iter().enumerate() {
                    out[i + j] += x * y;
                }
            }
            out
        };
        let inner: Vec<f64> = inner.0.iter().map(|&c| c as f64).collect();
        let (mut sum, mut power) = (vec![0.0; 17], vec![1.0]);
        for &c in &self.0 {
            for (s, p) in sum.iter_mut().zip(&power) {
                *s += c as f64 * p;
            }
            power = mul(&power, &inner);
        }
        sum[5..]
            .iter()
            .all(|&c| c == 0.0)
            .then(|| Curve(std::array::from_fn(|i| sum[i] as f32)))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Shape {
    /// Shape frame relative to its link frame.
    pub(crate) origin: Transform,
    pub(crate) geometry: Geometry,
}

/// Primitive sizes are half extents; round primitives are aligned with their local z axis.
#[derive(Clone, Debug)]
pub(crate) enum Geometry {
    Box {
        half: Vec3,
    },
    Sphere {
        radius: f32,
    },
    Cylinder {
        radius: f32,
        half_length: f32,
    },
    Capsule {
        radius: f32,
        half_length: f32,
    },
    Ellipsoid {
        radii: Vec3,
    },
    /// The local z = 0 plane, solid below; only scenes have planes.
    Plane,
    Mesh {
        path: PathBuf,
        scale: Vec3,
    },
    /// A mesh given inline (USD) rather than by file.
    #[cfg_attr(
        not(feature = "usd"),
        allow(dead_code, reason = "only the USD loader makes one")
    )]
    TriMesh(TriMesh),
    /// The convex hull of a geometry, as MuJoCo collides meshes.
    ConvexHull(Box<Geometry>),
}

/// A triangle mesh with outward-facing (counterclockwise) triangles.
#[derive(Clone, Debug, Default)]
pub(crate) struct TriMesh {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) triangles: Vec<[u32; 3]>,
}

impl TriMesh {
    pub(crate) fn append(&mut self, other: TriMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend(other.vertices);
        self.triangles
            .extend(other.triangles.iter().map(|t| t.map(|i| i + base)));
    }

    pub(crate) fn corners(&self, t: [u32; 3]) -> [Vec3; 3] {
        t.map(|i| self.vertices[i as usize])
    }

    /// Flips the triangles if they face inward.
    pub(crate) fn orient_outward(&mut self) {
        if self.signed_volume6() < 0.0 {
            for t in &mut self.triangles {
                t.swap(1, 2);
            }
        }
    }

    /// Six times the enclosed volume; negative when the triangles face inward.
    fn signed_volume6(&self) -> f32 {
        self.triangles
            .iter()
            .map(|&t| {
                let [a, b, c] = self.corners(t);
                a.dot(b.cross(c))
            })
            .sum()
    }
}

pub(crate) fn triangle_area([a, b, c]: [Vec3; 3]) -> f32 {
    0.5 * (b - a).cross(c - a).length()
}

impl Shape {
    /// The shape as an outward-facing triangle mesh in its link's frame.
    pub(crate) fn mesh(&self) -> Result<TriMesh> {
        self.placed(self.geometry.mesh()?)
    }

    /// [`Shape::mesh`], with a round shape cut into sectors fine enough that its mesh lies no
    /// farther than `error` inside it ([`Geometry::mesh_within`]).
    pub(crate) fn mesh_within(&self, error: f32) -> Result<TriMesh> {
        self.placed(self.geometry.mesh_within(error)?)
    }

    fn placed(&self, mut mesh: TriMesh) -> Result<TriMesh> {
        for v in &mut mesh.vertices {
            *v = self.origin.rot * *v + self.origin.trans;
        }
        Ok(mesh)
    }
}

impl Geometry {
    /// The geometry as an outward-facing triangle mesh in its own frame.
    pub(crate) fn mesh(&self) -> Result<TriMesh> {
        self.mesh_in(SECTORS)
    }

    /// [`Geometry::mesh`], with a cylinder, capsule or sphere cut into at least `SECTORS` sectors
    /// and as many more as keep its mesh within `error` of it: a chord across a sector of radius
    /// `r` lies `r (1 - cos(pi / sectors))` inside it, and a capsule's or a sphere's facets that
    /// much again, along their meridians.
    pub(crate) fn mesh_within(&self, error: f32) -> Result<TriMesh> {
        let (radius, directions) = match *self {
            Geometry::Cylinder { radius, .. } => (radius, 1.0),
            Geometry::Capsule { radius, .. } | Geometry::Sphere { radius } => (radius, 2.0),
            _ => return self.mesh(),
        };
        let chord = (1.0 - error / (directions * radius)).clamp(-1.0, 1.0);
        let sectors = (PI / chord.acos()).ceil();
        // Even, so a capsule's or a sphere's half profile has whole steps.
        let sectors = (sectors as u32).max(SECTORS).next_multiple_of(2);
        self.mesh_in(sectors)
    }

    fn mesh_in(&self, sectors: u32) -> Result<TriMesh> {
        let steps = sectors / 2;
        let mut mesh = match self {
            Geometry::Box { half } => box_mesh(*half),
            Geometry::Sphere { radius } => sphere_mesh(Vec3::splat(*radius), sectors),
            Geometry::Ellipsoid { radii } => sphere_mesh(*radii, sectors),
            Geometry::Plane => bail!("a plane has no finite surface to fit spheres to"),
            Geometry::Cylinder {
                radius,
                half_length,
            } => lathe(
                &[
                    (0.0, -half_length),
                    (*radius, -half_length),
                    (*radius, *half_length),
                    (0.0, *half_length),
                ],
                sectors,
            ),
            Geometry::Capsule {
                radius,
                half_length,
            } => {
                let half = steps / 2;
                let at = |k: u32, z: f32| polar(*radius, PI * k as f32 / steps as f32, z);
                let mut profile: Vec<(f32, f32)> =
                    (0..=half).map(|k| at(k, -half_length)).collect();
                profile.extend((half..=steps).map(|k| at(k, *half_length)));
                lathe(&profile, sectors)
            }
            Geometry::Mesh { path, scale } => load_mesh(path, *scale)?,
            Geometry::TriMesh(mesh) => mesh.clone(),
            Geometry::ConvexHull(inner) => {
                let (vertices, triangles) =
                    parry3d::transformation::try_convex_hull(&inner.mesh()?.vertices)
                        .map_err(|e| anyhow!("no convex hull: {e}"))?;
                TriMesh {
                    vertices,
                    triangles,
                }
            }
        };
        mesh.orient_outward();
        Ok(mesh)
    }
}

fn sphere_mesh(radii: Vec3, sectors: u32) -> TriMesh {
    let steps = sectors / 2;
    let profile: Vec<(f32, f32)> = (0..=steps)
        .map(|k| polar(1.0, PI * k as f32 / steps as f32, 0.0))
        .collect();
    let mut mesh = lathe(&profile, sectors);
    for v in &mut mesh.vertices {
        *v *= radii;
    }
    mesh
}

/// Point of a profile running from the bottom pole (`angle` 0) to the top pole (`angle` pi),
/// as (radius, z).
fn polar(radius: f32, angle: f32, z_offset: f32) -> (f32, f32) {
    let s = angle.sin();
    (
        if s.abs() < 1e-6 { 0.0 } else { radius * s },
        -radius * angle.cos() + z_offset,
    )
}

/// Sectors a round shape's mesh is cut into about its axis, and twice the steps of a capsule's or
/// a sphere's profile from pole to pole.
const SECTORS: u32 = 24;

/// Revolves a (radius, z) profile from bottom to top about the z axis in `sectors`. Profile
/// points with zero radius become single pole vertices.
fn lathe(profile: &[(f32, f32)], sectors: u32) -> TriMesh {
    let mut mesh = TriMesh::default();
    let rings: Vec<Vec<u32>> = profile
        .iter()
        .map(|&(r, z)| {
            let first = mesh.vertices.len() as u32;
            if r <= 0.0 {
                mesh.vertices.push(Vec3::new(0.0, 0.0, z));
                return vec![first; sectors as usize];
            }
            mesh.vertices.extend((0..sectors).map(|j| {
                let a = TAU * j as f32 / sectors as f32;
                Vec3::new(r * a.cos(), r * a.sin(), z)
            }));
            (first..first + sectors).collect()
        })
        .collect();
    for pair in rings.windows(2) {
        let (lo, hi) = (&pair[0], &pair[1]);
        for j in 0..sectors as usize {
            let k = (j + 1) % sectors as usize;
            let (a, b, c, d) = (lo[j], lo[k], hi[k], hi[j]);
            if a != b {
                mesh.triangles.push([a, b, c]);
            }
            if c != d {
                mesh.triangles.push([a, c, d]);
            }
        }
    }
    mesh
}

fn box_mesh(half: Vec3) -> TriMesh {
    // Vertex i has coordinate +half where bit (x, y, z) = (1, 2, 4) of i is set.
    let vertices = (0..8)
        .map(|i| {
            Vec3::new(
                sign(i & 1) * half.x,
                sign(i & 2) * half.y,
                sign(i & 4) * half.z,
            )
        })
        .collect();
    let triangles = vec![
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5], // -x, +x
        [0, 1, 5],
        [0, 5, 4],
        [2, 6, 7],
        [2, 7, 3], // -y, +y
        [0, 2, 3],
        [0, 3, 1],
        [4, 5, 7],
        [4, 7, 6], // -z, +z
    ];
    TriMesh {
        vertices,
        triangles,
    }
}

fn sign(bit: u32) -> f32 {
    if bit != 0 { 1.0 } else { -1.0 }
}

fn load_mesh(path: &Path, scale: Vec3) -> Result<TriMesh> {
    let loading = || format!("loading mesh {}", path.display());
    let scene = if binary_stl(path).with_context(loading)? {
        // The header holds no geometry: blanked, it cannot pass for ASCII.
        let mut bytes = std::fs::read(path).with_context(loading)?;
        bytes[..80].fill(0);
        Loader::default().load_stl_from_slice(&bytes, path)
    } else {
        Loader::default().load(path)
    }
    .with_context(loading)?;
    let mut mesh = TriMesh::default();
    for m in scene.meshes {
        mesh.append(TriMesh {
            vertices: m.vertices.iter().map(|&v| Vec3::from(v) * scale).collect(),
            triangles: m.faces,
        });
    }
    if mesh.triangles.is_empty() {
        bail!("mesh {} has no triangles", path.display());
    }
    Ok(mesh)
}

/// Whether `path` is an STL of exactly the binary layout's size (an 80-byte header, the triangle
/// count, 50 bytes per triangle), as trimesh tells binary from ASCII. mesh-loader reads a file
/// that starts with "solid" as ASCII, and binary ones may: CAD exporters write "solid <name>" in
/// the header, and a line break in the count then fails its parse.
fn binary_stl(path: &Path) -> std::io::Result<bool> {
    use std::io::Read;
    if !path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("stl"))
    {
        return Ok(false);
    }
    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    if size < 84 {
        return Ok(false);
    }
    let mut head = [0u8; 84];
    file.read_exact(&mut head)?;
    let triangles = u32::from_le_bytes([head[80], head[81], head[82], head[83]]);
    Ok(size == 84 + 50 * u64::from(triangles))
}

/// Loads a robot description, choosing the format by file extension.
/// One description of several: each part's link and joint names get its prefix, its root link
/// hangs from `world` at its base, and its loops come along.
pub(crate) fn assemble(
    parts: Vec<(RobotDescription, String, Transform)>,
) -> Result<RobotDescription> {
    let mut out = RobotDescription {
        name: parts
            .iter()
            .map(|(d, prefix, _)| format!("{prefix}{}", d.name))
            .collect::<Vec<_>>()
            .join("+"),
        links: vec![LinkDesc {
            name: "world".into(),
            ..Default::default()
        }],
        joints: vec![],
        loops: vec![],
    };
    for (desc, prefix, base) in parts {
        let named = |name: &str| format!("{prefix}{name}");
        let root = desc
            .links
            .iter()
            .find(|l| desc.joints.iter().all(|j| j.child != l.name))
            .with_context(|| format!("description '{}' has no root link", desc.name))?;
        out.joints.push(JointDesc::fixed(
            &named("base"),
            "world",
            &named(&root.name),
            base,
        ));
        out.links.extend(desc.links.into_iter().map(|l| LinkDesc {
            name: named(&l.name),
            ..l
        }));
        out.joints
            .extend(desc.joints.into_iter().map(|j| JointDesc {
                name: named(&j.name),
                parent: named(&j.parent),
                child: named(&j.child),
                mimic: j.mimic.map(|m| Mimic {
                    joint: named(&m.joint),
                    ..m
                }),
                ..j
            }));
        out.loops
            .extend(desc.loops.into_iter().map(|l| LoopClosure {
                name: named(&l.name),
                links: l.links.map(|link| named(&link)),
                ..l
            }));
    }
    let mut names: Vec<&str> = out.links.iter().map(|l| l.name.as_str()).collect();
    names.sort_unstable();
    if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
        bail!(
            "two parts name a link '{}': give them different prefixes",
            w[0]
        );
    }
    Ok(out)
}

pub(crate) fn load_robot(
    path: &Path,
    package_dirs: &[PathBuf],
    variants: &[(String, String)],
) -> Result<RobotDescription> {
    match extension(path).as_deref() {
        Some("urdf") => crate::urdf::load(path, package_dirs),
        Some("xml" | "mjcf") => Ok(crate::mjcf::load(path)?.robot),
        Some("usd" | "usda" | "usdc" | "usdz") => Ok(load_usd(path, variants)?.robot),
        _ => bail!(
            "{}: unsupported robot description format (expected .urdf, .xml, .mjcf or .usd*)",
            path.display()
        ),
    }
}

/// Loads a scene's static geometry, choosing the format by file extension.
pub(crate) fn load_scene(path: &Path) -> Result<Vec<Shape>> {
    match extension(path).as_deref() {
        Some("xml" | "mjcf") => Ok(crate::mjcf::load(path)?.scene),
        Some("usd" | "usda" | "usdc" | "usdz") => Ok(load_usd(path, &[])?.scene),
        _ => bail!(
            "{}: unsupported scene format (expected .xml, .mjcf or .usd*)",
            path.display()
        ),
    }
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
}

#[cfg(feature = "usd")]
fn load_usd(path: &Path, variants: &[(String, String)]) -> Result<Model> {
    crate::usd::load(path, variants)
}

#[cfg(not(feature = "usd"))]
fn load_usd(path: &Path, _: &[(String, String)]) -> Result<Model> {
    bail!(
        "{}: rebuild with `--features usd` to load OpenUSD files",
        path.display()
    )
}
