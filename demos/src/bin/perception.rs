//! Planning around what a depth camera sees.
//!
//! Three simulated depth cameras look at a cluttered table, robot included (a ray caster here
//! stands in for the sensors, drawing the arm as its collision spheres). batchplan masks the robot out
//! of each image, fuses them in an occupancy map, turns that into a distance grid, and plans a run of reaches with the grid as the only
//! obstacle. The viewer shows each camera's point cloud, the surface of the grid the robot plans
//! around, the true scene (wireframe, for comparison), and the robot moving through its reaches.
//!
//!   cargo run --release --bin perception
//!   cargo run --release --bin perception -- --cpu

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use batchplan::glam::{Mat3, Quat, Vec3};
use batchplan::rng::Rng;
use batchplan::*;
use batchplan_demos::*;
use rayon::prelude::*;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const REACHES: usize = 8;
const HZ: f32 = 30.0;
/// Reaches run at this fraction of the Panda's speed.
const SPEED: f32 = 0.6;

fn main() -> Result<()> {
    let args = Args::parse()?;
    let robot = panda()?;
    let device = args.device(&robot, true)?;
    let mut rng = Rng::new(args.count.unwrap_or(4) as u64);
    let mut truth = tabletop(&mut rng);
    truth.obstacles.push(Obstacle::sphere(Vec3::new(0.55, 0.25, 0.08), 0.08));
    truth.obstacles.push(Obstacle::cylinder(Vec3::new(0.45, -0.3, 0.12), 0.05, 0.12));

    // What the cameras see, the robot in view at its start.
    let intrinsics = Intrinsics { fx: 525.0, fy: 525.0, cx: 319.5, cy: 239.5 };
    let aim = Vec3::new(0.45, 0.0, 0.1);
    let cameras: Vec<(&str, Pose)> = vec![
        ("right", look_at(Vec3::new(1.5, 0.9, 1.0), aim)),
        ("left", look_at(Vec3::new(1.5, -0.9, 1.0), aim)),
        ("back", look_at(Vec3::new(-0.4, 0.2, 1.5), aim)),
    ];
    let q0 = robot.default_q().to_vec();
    let mut seen = truth.clone();
    for (link, spheres) in &robot.collision_model().spheres {
        let pose = robot.link_pose(&q0, link).context("a robot link")?;
        seen.obstacles.extend(
            spheres
                .iter()
                .map(|&[x, y, z, radius]| Obstacle::sphere(pose.position + pose.rotation * Vec3::new(x, y, z), radius)),
        );
    }
    let views: Vec<Vec<f32>> = cameras.iter().map(|(_, pose)| depth_image(&seen, *pose, intrinsics)).collect();

    // The map: each view with the robot masked out, then a grid in which never-seen space is solid.
    let t = Instant::now();
    let mut map = OccupancyMap::new(Vec3::new(-0.3, -0.9, -0.05), Vec3::new(1.2, 0.9, 1.0), 0.01)?;
    for ((_, pose), view) in cameras.iter().zip(&views) {
        let image = DepthImage { depth: view, width: WIDTH as usize, intrinsics, camera: *pose };
        let arm = image.robot_mask(&robot, &q0, 0.04)?;
        map.integrate(&device, &image, Some(&arm))?;
    }
    // Space no camera saw is solid, except where only the robot hid it: no camera sees under the
    // arm, and that shadow, solid, would wall in the space every reach passes through.
    let grid = Arc::new(map.grid(&device, Occlusion::OccupiedExceptRobotShadow)?);
    let built = t.elapsed().as_secs_f64() * 1e3;
    let observed =
        World { obstacles: vec![Obstacle::Sdf { grid: grid.clone(), center: Vec3::ZERO, rotation: Quat::IDENTITY }] };

    // A run of reaches to targets clear of the true scene, planned against the grid alone.
    let t = Instant::now();
    let worlds = device.upload(std::slice::from_ref(&observed))?;
    let targets: Vec<IkProblem> = (0..REACHES * 3).map(|_| IkProblem::new(0, grasp_target(&truth, &mut rng))).collect();
    let ik = solve_ik(&device, &worlds, &targets, &IkOptions::default())?;
    let goals: Vec<Vec<f32>> =
        (0..targets.len()).filter_map(|k| ik.best(k).map(<[f32]>::to_vec)).take(REACHES).collect();
    let problems: Vec<PlanProblem> = goals
        .iter()
        .enumerate()
        .map(|(k, goal)| PlanProblem::new(0, if k == 0 { q0.clone() } else { goals[k - 1].clone() }, goal.clone()))
        .collect();
    let planned = plan(&device, &worlds, &problems, &PlanOptions::default())?;
    let planning = t.elapsed().as_secs_f64() * 1e3;
    // Played one after another, stopping at the first reach without a path.
    let mut samples: Vec<f32> = vec![];
    let mut reached = 0;
    for k in 0..problems.len() {
        let Some(path) = planned.best(k) else { break };
        samples.extend(Trajectory::new(&robot, path, SPEED)?.sample(HZ)?.positions);
        reached += 1;
    }
    let n = robot.dof();
    let frames = samples.len() / n;

    // Every sample against the true scene, exactly (clearances are exact up to the margin).
    let cpu = Device::cpu(&robot)?;
    let exact = cpu.upload(std::slice::from_ref(&truth))?;
    let weights = CollisionWeights { margin: 1.0, ..CollisionWeights::NONE };
    let eval = cpu.evaluate(&exact, &vec![0; frames], &samples, &weights)?;
    let closest = eval.world_clearance.iter().copied().fold(f32::INFINITY, f32::min);

    let [nx, ny, nz] = grid.dims();
    let summary = format!(
        "## Planning around what three depth cameras see\n\n\
         On {}:\n\n\
         - three simulated {WIDTH}x{HEIGHT} depth images, the robot in view;\n\
         - the robot masked out of each, the three fused in an occupancy map and turned into a \
           {nx}x{ny}x{nz} distance grid at {:.0} mm in {built:.1} ms, unseen space solid except the robot's shadow;\n\
         - IK for {} targets and {reached} reaches planned against the grid alone in {planning:.1} ms.\n\n\
         Checked against the true scene, the robot comes no closer than {:.0} mm.\n\n\
         Orange points: the grid's surface, what the robot plans around. Wireframe: the true scene.",
        device.name(),
        grid.voxel() * 1e3,
        targets.len(),
        closest * 1e3,
    );
    println!("{}", summary.replace("## ", "").replace("\n\n", "\n"));

    let rec = args.recording("batchplan_perception")?;
    rec.log_static("summary", &rerun::TextDocument::from_markdown(summary))?;
    let truth_boxes: Vec<(usize, &Obstacle)> = truth.obstacles.iter().enumerate().collect();
    rec.log_static("world/truth", &boxes(&truth_boxes, |_| slate()).with_fill_mode(rerun::FillMode::MajorWireframe))?;
    for (k, o) in truth.obstacles.iter().enumerate() {
        match *o {
            Obstacle::Sphere { center, radius } => {
                rec.log_static(
                    format!("world/truth/sphere_{k}"),
                    &rerun::Ellipsoids3D::from_centers_and_radii([center.to_array()], [radius]).with_colors([slate()]),
                )?;
            }
            Obstacle::Cylinder { center, rotation, radius, half_height } => {
                rec.log_static(
                    format!("world/truth/cylinder_{k}"),
                    &rerun::Cylinders3D::from_lengths_and_radii([2.0 * half_height], [radius])
                        .with_centers([center.to_array()])
                        .with_quaternions([quaternion(rotation)])
                        .with_colors([slate()]),
                )?;
            }
            _ => {}
        }
    }

    // The cameras, and the point cloud each one's depth makes.
    for ((name, pose), view) in cameras.iter().zip(&views) {
        let entity = format!("world/cameras/{name}");
        rec.log_static(entity.as_str(), &transform(pose))?;
        rec.log_static(
            entity.as_str(),
            &rerun::Pinhole::from_focal_length_and_resolution(
                [intrinsics.fx, intrinsics.fy],
                [WIDTH as f32, HEIGHT as f32],
            )
            .with_image_plane_distance(0.15),
        )?;
        let depth: Vec<u8> = view.iter().flat_map(|v| v.to_le_bytes()).collect();
        rec.log_static(
            format!("{entity}/depth"),
            &rerun::DepthImage::from_data_type_and_bytes(depth, [WIDTH, HEIGHT], rerun::ChannelDatatype::F32)
                .with_meter(1.0),
        )?;
    }

    // The grid's surface: points within half a voxel of distance zero.
    let (lo, hi) = grid.bounds();
    let step = grid.voxel();
    let mut surface = vec![];
    let mut p = lo;
    while p.z <= hi.z {
        p.y = lo.y;
        while p.y <= hi.y {
            p.x = lo.x;
            while p.x <= hi.x {
                if grid.distance(p).0.abs() < step * 0.5 {
                    surface.push(p.to_array());
                }
                p.x += step;
            }
            p.y += step;
        }
        p.z += step;
    }
    rec.log_static(
        "world/grid_surface",
        &rerun::Points3D::new(surface).with_radii([step * 0.4]).with_colors([rerun::Color::from_rgb(240, 150, 40)]),
    )?;

    let meshes = panda_meshes(&rec, "world/panda", &robot)?;
    let hand: Vec<[f32; 3]> = samples.chunks(n).map(|q| robot.ee_pose(q, 0).position.to_array()).collect();
    rec.log_static(
        "world/hand_path",
        &rerun::LineStrips3D::new([hand]).with_radii([0.003]).with_colors([rerun::Color::from_rgb(40, 200, 90)]),
    )?;
    for f in 0..frames.max(1) {
        rec.set_duration_secs("time", f as f32 / HZ);
        let q = if frames == 0 { &q0[..] } else { &samples[f * n..][..n] };
        meshes.log_robots(&rec, &[(Vec3::ZERO, meshes.link_poses(&robot, q, panda_fingers(&robot, q, OPEN)?)?)])?;
    }
    if args.save.is_none() {
        println!("Opened in the Rerun viewer; press play on the time panel.");
    }
    Ok(())
}

/// A camera at `eye` looking at `target`, in OpenCV axes (x right, y down, z forward).
fn look_at(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
}

/// z depth per pixel by sphere tracing `scene` from `camera`; 0 where a ray hits nothing.
fn depth_image(scene: &World, camera: Pose, k: Intrinsics) -> Vec<f32> {
    let width = WIDTH as usize;
    (0..width * HEIGHT as usize)
        .into_par_iter()
        .map(|i| {
            let (u, v) = ((i % width) as f32, (i / width) as f32);
            let ray = Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0).normalize();
            let dir = camera.rotation * ray;
            let mut t = 0.0;
            while t < 4.0 {
                let p = camera.position + dir * t;
                let d = scene.obstacles.iter().map(|o| o.distance(p).0).fold(f32::INFINITY, f32::min);
                if d < 1e-4 {
                    return t * ray.z;
                }
                t += d;
            }
            0.0
        })
        .collect()
}
