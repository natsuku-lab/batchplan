//! What the demos share: the Panda as batchplan loads it, a robot's meshes drawn once and placed in
//! every scene, tabletop scenes, and a recording to log them to.

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use batchplan::glam::{EulerRot, Quat, Vec3};
use batchplan::rng::Rng;
use batchplan::{CollisionModel, Device, Obstacle, Pose, Robot, RobotOptions, World};

pub const ASSETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets");
/// How far each Panda finger opens, metres.
pub const OPEN: f32 = 0.04;

/// Command-line options every demo takes: a count, `--gpu` or `--cpu`, and `--save <file.rrd>`
/// to write a recording instead of opening the viewer.
pub struct Args {
    pub count: Option<usize>,
    pub gpu: bool,
    pub cpu: bool,
    pub save: Option<PathBuf>,
}

impl Args {
    pub fn parse() -> Result<Self> {
        let mut parsed = Args {
            count: None,
            gpu: false,
            cpu: false,
            save: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--gpu" => parsed.gpu = true,
                "--cpu" => parsed.cpu = true,
                "--save" => {
                    parsed.save = Some(PathBuf::from(args.next().context("--save needs a path")?))
                }
                n => parsed.count = Some(n.parse().with_context(|| format!("not a count: {n}"))?),
            }
        }
        Ok(parsed)
    }

    /// The device asked for, else `default_gpu`'s choice (falling back to the CPU).
    pub fn device(&self, robot: &Robot, default_gpu: bool) -> Result<Device> {
        Ok(if self.cpu || (!self.gpu && !default_gpu) {
            Device::cpu(robot)?
        } else {
            Device::gpu(robot).or_else(|_| Device::cpu(robot))?
        })
    }

    /// The viewer, or a file with `--save`; z is up.
    pub fn recording(&self, name: &'static str) -> Result<rerun::RecordingStream> {
        let builder = rerun::RecordingStreamBuilder::new(name);
        let rec = match &self.save {
            Some(path) => builder.save(path.clone())?,
            None => builder.spawn()?,
        };
        rec.log_static("world", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP())?;
        Ok(rec)
    }
}

pub fn urdf() -> PathBuf {
    Path::new(ASSETS).join("franka/franka_panda.urdf")
}

/// The Panda with its fingers held open and the grasp point between the fingertips as its IK
/// frame, with the collision spheres that ship with batchplan.
pub fn panda() -> Result<Robot> {
    let fingers = [
        ("panda_finger_joint1".to_string(), OPEN),
        ("panda_finger_joint2".to_string(), OPEN),
    ];
    let options = RobotOptions {
        ee_links: vec!["ee_link".into()],
        lock_joints: fingers.into_iter().collect(),
        default_q: Some(vec![0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]),
        collision_model: Some(CollisionModel::load(
            Path::new(ASSETS).join("franka/panda_collision.json"),
        )?),
        ..Default::default()
    };
    Ok(Robot::load(urdf(), &options)?)
}

/// A robot's links as meshes, each logged once under `root` and drawn wherever a scene's robot
/// puts that link (Rerun's instance poses), so a hundred robots cost one robot's meshes.
pub struct RobotMeshes {
    root: String,
    links: Vec<LinkMeshes>,
}

struct LinkMeshes {
    name: String,
    meshes: Vec<Mesh>,
}

struct Mesh {
    path: PathBuf,
    /// In the link frame.
    pose: Pose,
    /// The description's colour for a visual mesh, else the mesh's own; collision meshes are grey.
    color: Option<u32>,
}

impl RobotMeshes {
    /// Each link's visual meshes, else its collision meshes, from every description the robot was
    /// loaded from (`Robot::parts`), its links named with the part's prefix. A mesh missing beside
    /// its description is looked for by file name in `fallback`.
    pub fn log(
        rec: &rerun::RecordingStream,
        root: &str,
        robot: &Robot,
        fallback: &Path,
    ) -> Result<Self> {
        let mut links = vec![];
        for part in robot.parts() {
            let description = urdf_rs::read_file(&part.path)?;
            let dir = part.path.parent().context("the URDF's directory")?;
            let resolve = |filename: &str| {
                let path = dir.join(filename);
                if path.exists() {
                    Some(path)
                } else {
                    Path::new(filename)
                        .file_name()
                        .map(|f| fallback.join(f))
                        .filter(|p| p.exists())
                }
            };
            // A visual's colour is its material's, or that of the description's material of its
            // name.
            let named: HashMap<&str, &urdf_rs::Color> = description
                .materials
                .iter()
                .filter_map(|m| Some((m.name.as_str(), m.color.as_ref()?)))
                .collect();
            let rgba = |m: &urdf_rs::Material| {
                let c = m
                    .color
                    .as_ref()
                    .or_else(|| named.get(m.name.as_str()).copied())?
                    .rgba
                    .0;
                Some(u32::from_be_bytes(
                    c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8),
                ))
            };
            let mesh = |g: &urdf_rs::Geometry, origin: &urdf_rs::Pose, color: Option<u32>| {
                let urdf_rs::Geometry::Mesh { filename, .. } = g else {
                    return None;
                };
                let (xyz, rpy) = (origin.xyz.0, origin.rpy.0);
                let rotation =
                    Quat::from_euler(EulerRot::ZYX, rpy[2] as f32, rpy[1] as f32, rpy[0] as f32);
                let position = Vec3::new(xyz[0] as f32, xyz[1] as f32, xyz[2] as f32);
                Some(Mesh {
                    path: resolve(filename)?,
                    pose: Pose { position, rotation },
                    color,
                })
            };
            for link in description.links {
                let name = format!("{}{}", part.prefix, link.name);
                if robot.link_pose(robot.default_q(), &name).is_none() {
                    continue;
                }
                let visual: Vec<Mesh> = link
                    .visual
                    .iter()
                    .filter_map(|v| {
                        mesh(&v.geometry, &v.origin, v.material.as_ref().and_then(rgba))
                    })
                    .collect();
                let meshes = if !visual.is_empty() && visual.len() == link.visual.len() {
                    visual
                } else {
                    link.collision
                        .iter()
                        .filter_map(|c| mesh(&c.geometry, &c.origin, Some(0xd8d8dcff)))
                        .collect()
                };
                links.push(LinkMeshes { name, meshes });
            }
        }
        if links.is_empty() {
            bail!("no links of {} found in its descriptions", robot.name());
        }
        for link in &links {
            for (k, m) in link.meshes.iter().enumerate() {
                let asset = rerun::Asset3D::from_file_path(&m.path)?;
                let asset = match m.color {
                    Some(rgba) => asset.with_albedo_factor(rgba),
                    None => asset,
                };
                rec.log_static(format!("{root}/{}_{k}", link.name), &asset)?;
            }
        }
        Ok(Self {
            root: root.into(),
            links,
        })
    }

    /// Every link's pose at `q`, after `adjust` (given the link's name), for joints the planning
    /// model holds still, such as fingers.
    pub fn link_poses(
        &self,
        robot: &Robot,
        q: &[f32],
        adjust: impl Fn(&str, Pose) -> Pose,
    ) -> Result<Vec<Pose>> {
        self.links
            .iter()
            .map(|link| {
                Ok(adjust(
                    &link.name,
                    robot.link_pose(q, &link.name).context("a robot link")?,
                ))
            })
            .collect()
    }

    /// Draws one robot per entry: its base at `offset`, its links at `poses` (from `link_poses`).
    pub fn log_robots(
        &self,
        rec: &rerun::RecordingStream,
        robots: &[(Vec3, Vec<Pose>)],
    ) -> Result<()> {
        for (l, link) in self.links.iter().enumerate() {
            for (k, m) in link.meshes.iter().enumerate() {
                let placed: Vec<Pose> = robots
                    .iter()
                    .map(|(offset, poses)| {
                        let p = poses[l];
                        Pose::new(
                            *offset + p.position + p.rotation * m.pose.position,
                            p.rotation * m.pose.rotation,
                        )
                    })
                    .collect();
                rec.log(
                    format!("{}/{}_{k}", self.root, link.name),
                    &rerun::InstancePoses3D::new()
                        .with_translations(placed.iter().map(|p| p.position.to_array()))
                        .with_quaternions(placed.iter().map(|p| quaternion(p.rotation))),
                )?;
            }
        }
        Ok(())
    }
}

/// The Panda's meshes. The repository ships only its collision meshes; its visual meshes
/// (franka_ros 0.7.0, Apache-2.0) come from `scripts/fetch_panda_meshes.sh`, else the collision
/// meshes stand in.
pub fn panda_meshes(
    rec: &rerun::RecordingStream,
    root: &str,
    robot: &Robot,
) -> Result<RobotMeshes> {
    RobotMeshes::log(
        rec,
        root,
        robot,
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../data/franka_visual"
        )),
    )
}

/// Moves the Panda's fingers `opening` metres from the hand's centre line at `q` (the planning
/// model holds them fully open), for `RobotMeshes::link_poses`.
pub fn panda_fingers(
    robot: &Robot,
    q: &[f32],
    opening: f32,
) -> Result<impl Fn(&str, Pose) -> Pose> {
    let hand = robot
        .link_pose(q, "panda_hand")
        .context("the Panda's hand")?;
    let across = hand.rotation * Vec3::Y;
    Ok(move |name: &str, mut pose: Pose| {
        match name {
            "panda_leftfinger" => pose.position += across * (opening - OPEN),
            "panda_rightfinger" => pose.position -= across * (opening - OPEN),
            _ => {}
        }
        pose
    })
}

/// Where scene `i` of `n` sits: a square grid, `spacing` metres apart.
pub fn grid(n: usize, spacing: f32) -> Vec<Vec3> {
    let columns = (n as f32).sqrt().ceil() as usize;
    (0..n)
        .map(|i| {
            Vec3::new(
                (i % columns) as f32 * spacing,
                (i / columns) as f32 * spacing,
                0.0,
            )
        })
        .collect()
}

pub fn wood() -> rerun::Color {
    rerun::Color::from_rgb(150, 120, 90)
}

pub fn slate() -> rerun::Color {
    rerun::Color::from_rgb(90, 110, 140)
}

/// Boxes as one solid `Boxes3D`, coloured by `color(index)`.
pub fn boxes(
    obstacles: &[(usize, &Obstacle)],
    color: impl Fn(usize) -> rerun::Color,
) -> rerun::Boxes3D {
    let (mut centers, mut halves, mut rotations, mut colors) = (vec![], vec![], vec![], vec![]);
    for &(k, o) in obstacles {
        let Obstacle::Cuboid {
            center,
            half_extents,
            rotation,
        } = *o
        else {
            continue;
        };
        centers.push(center.to_array());
        halves.push(half_extents.to_array());
        rotations.push(quaternion(rotation));
        colors.push(color(k));
    }
    rerun::Boxes3D::from_centers_and_half_sizes(centers, halves)
        .with_quaternions(rotations)
        .with_colors(colors)
        .with_fill_mode(rerun::FillMode::Solid)
}

pub fn transform(pose: &Pose) -> rerun::Transform3D {
    rerun::Transform3D::from_translation_rotation(
        pose.position.to_array(),
        quaternion(pose.rotation),
    )
}

pub fn quaternion(q: Quat) -> rerun::Quaternion {
    rerun::Quaternion::from_xyzw(q.to_array())
}

/// A table at the robot's base height with 2-6 random boxes on it, in front of the robot.
pub fn tabletop(rng: &mut Rng) -> World {
    let mut obstacles = vec![Obstacle::cuboid(
        Vec3::new(0.4, 0.0, -0.02),
        Vec3::new(0.7, 0.8, 0.02),
    )];
    for _ in 0..2 + (rng.uniform() * 5.0) as usize {
        let half = Vec3::new(
            rng.range(0.02, 0.08),
            rng.range(0.02, 0.08),
            rng.range(0.03, 0.2),
        );
        obstacles.push(Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.3, 0.75), rng.range(-0.5, 0.5), half.z),
            half_extents: half,
            rotation: Quat::from_rotation_z(rng.range(0.0, PI)),
        });
    }
    World::new(obstacles)
}

/// A top-down grasp (the gripper's z axis pointing down) at least 6 cm from every obstacle.
pub fn grasp_target(world: &World, rng: &mut Rng) -> Pose {
    loop {
        let p = Vec3::new(
            rng.range(0.3, 0.7),
            rng.range(-0.45, 0.45),
            rng.range(0.05, 0.4),
        );
        if world.obstacles.iter().all(|o| o.distance(p).value > 0.06) {
            let rotation = Quat::from_rotation_z(rng.range(-PI, PI)) * Quat::from_rotation_x(PI);
            return Pose {
                position: p,
                rotation,
            };
        }
    }
}
