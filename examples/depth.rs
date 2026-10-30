//! Planning around what a depth camera sees. Renders a 640x480 depth image of a tabletop scene,
//! turns it into a distance grid, and plans Panda reaches with that grid as the only obstacle.
//! The plans are then checked against the true scene.
//!
//! cargo run --release --example depth -- [--cpu]
//!
//! The robot is not in the rendered scene. A real camera sees the arm too: pass
//! `DepthImage::robot_mask` to `SdfGrid::from_depth` so it is not an obstacle to itself.

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use anyhow::Result;
use batchplan::rng::Rng;
use batchplan::{
    CollisionWeights, DepthImage, Device, IkOptions, IkProblem, Intrinsics, Obstacle, Occlusion,
    PlanOptions, PlanProblem, Pose, SdfGrid, SdfOptions, Trajectory, World,
};
use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

/// A camera at `eye` looking at `target`, in OpenCV axes (x right, y down, z forward).
fn look_at(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
}

/// z depth per pixel by sphere tracing the scene's distance functions; 0 where a ray hits nothing.
fn render(scene: &World, camera: Pose, k: Intrinsics) -> Vec<f32> {
    (0..WIDTH * HEIGHT)
        .into_par_iter()
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let ray = Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
            let dir = camera.rotation * ray.normalize();
            let mut t = 0.0;
            while t < 4.0 {
                let p = camera.position + dir * t;
                let d = scene
                    .obstacles
                    .iter()
                    .map(|o| o.distance(p).value)
                    .fold(f32::INFINITY, f32::min);
                if d < 1e-4 {
                    return t * ray.normalize().z;
                }
                t += d;
            }
            0.0
        })
        .collect()
}

fn main() -> Result<()> {
    let robot = common::panda()?;
    let device = if std::env::args().any(|a| a == "--cpu") {
        Device::cpu(&robot)?
    } else {
        Device::gpu(&robot)?
    };
    println!("device: {}", device.name());
    let mut rng = Rng::new(3);
    let scene = common::tabletop(&mut rng);
    let camera = look_at(Vec3::new(1.6, 0.4, 1.0), Vec3::new(0.5, 0.0, 0.0));
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let depth = render(&scene, camera, k);
    let seen = depth.iter().filter(|&&z| z > 0.0).count();
    println!("rendered {WIDTH}x{HEIGHT} depth, {seen} pixels hit the scene");

    let t = Instant::now();
    let image = DepthImage {
        depth: &depth,
        width: WIDTH,
        intrinsics: k,
        camera,
    };
    let grid = SdfGrid::from_depth(
        &device,
        &image,
        None,
        Occlusion::Occupied,
        &SdfOptions::default(),
    )?;
    let [nx, ny, nz] = grid.dims();
    println!(
        "grid {nx}x{ny}x{nz} at {} m in {:.0} ms",
        grid.voxel(),
        t.elapsed().as_secs_f64() * 1e3
    );
    let observed = World::new(vec![Obstacle::from(grid)]);

    let t = Instant::now();
    let worlds = device.upload(&[observed])?;
    // Grasp targets clear of the true scene; some lie in space the camera cannot see.
    let goals: Vec<IkProblem> = (0..64)
        .map(|_| IkProblem::new(0, common::grasp_target(&scene, &mut rng)))
        .collect();
    let ik = batchplan::solve_ik(&device, &worlds, &goals, &IkOptions::default())?;
    let problems: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem::new(0, robot.default_q(), s.solution))
        .collect();
    let result = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default())?;
    println!(
        "IK reached {} of {} targets, planned {} in {:.0} ms",
        problems.len(),
        goals.len(),
        result.solved().count(),
        t.elapsed().as_secs_f64() * 1e3
    );

    // Every plan, timed and sampled densely, against the true scene.
    let cpu = Device::cpu(&robot)?;
    let truth = cpu.upload(std::slice::from_ref(&scene))?;
    let (mut samples, mut worst) = (vec![], f32::INFINITY);
    for s in result.solved() {
        let trajectory = Trajectory::try_new(&robot, s.solution, 1.0)?;
        samples.extend(
            trajectory
                .sample(32.0 / trajectory.knot_interval)?
                .positions,
        );
    }
    let items = samples.len() / robot.dof();
    // Clearances are exact up to the margin.
    let exact = CollisionWeights {
        margin: 1.0,
        ..CollisionWeights::NONE
    };
    let eval = cpu.evaluate(&truth, &vec![0; items], &samples, &exact)?;
    for i in 0..items {
        worst = worst.min(eval.world_clearance[i]);
    }
    println!(
        "closest approach to the true scene over {items} samples: {:.1} mm",
        worst * 1e3
    );
    Ok(())
}
