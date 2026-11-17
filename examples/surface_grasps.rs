//! Grasps of what cameras see, and what finding them costs. A bowl too wide to span, as points on
//! its surfaces, is grasped across its wall; a box on a table, as two 640x480 depth images, is
//! grasped across the sides the two cameras see between them; then 64 bowls in one call.
//!
//! cargo run --release --example surface_grasps

#[path = "common/mod.rs"]
mod common;

use std::f32::consts::TAU;
use std::time::Instant;

use anyhow::Result;
use batchplan::{DepthImage, Device, GraspOptions, Intrinsics, Obstacle, Pose, Surface, grasps_on};
use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

/// A bowl 11 cm across and 5 cm tall with a wall 8 mm thick and its bottom 3 cm under its rim:
/// points 2 mm apart on its walls, rim and bottom, with their normals.
fn bowl() -> Surface {
    let (outer, inner, top, bottom, spacing) = (0.055, 0.047, 0.05, 0.02, 0.002);
    let mut s = Surface::default();
    let mut ring = |r: f32, z: f32, normal: &dyn Fn(Vec3) -> Vec3| {
        let around = ((TAU * r / spacing) as usize).max(1);
        for k in 0..around {
            let angle = TAU * k as f32 / around as f32;
            let radial = Vec3::new(angle.cos(), angle.sin(), 0.0);
            s.points.push(radial * r + Vec3::Z * z);
            s.normals.push(normal(radial));
        }
    };
    let rows = (top / spacing) as usize;
    for j in 0..=rows {
        let z = j as f32 * spacing;
        ring(outer, z, &|radial| radial);
        if z >= bottom {
            ring(inner, z, &|radial| -radial);
        }
    }
    let across = ((outer - inner) / spacing) as usize;
    for i in 0..=across {
        ring(inner + i as f32 * spacing, top, &|_| Vec3::Z);
    }
    for i in 0..(inner / spacing) as usize {
        ring(i as f32 * spacing, bottom, &|_| Vec3::Z);
    }
    s
}

/// A camera at `eye` looking at `target`, in OpenCV axes (x right, y down, z forward).
fn look_at(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
}

/// z depth per pixel by sphere tracing the scene's distance functions; 0 where a ray hits nothing.
fn render(scene: &[Obstacle], camera: Pose, k: Intrinsics) -> Vec<f32> {
    (0..WIDTH * HEIGHT)
        .into_par_iter()
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let ray = Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
            let dir = camera.rotation * ray.normalize();
            let mut t = 0.0;
            while t < 2.0 {
                let p = camera.position + dir * t;
                let d = scene
                    .iter()
                    .map(|o| o.distance(p).value)
                    .fold(f32::INFINITY, f32::min);
                if d < 1e-5 {
                    return t * ray.normalize().z;
                }
                t += d;
            }
            0.0
        })
        .collect()
}

/// The quickest of `rounds` runs of `run`, in milliseconds, and its last result.
fn quickest<T>(rounds: usize, mut run: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..rounds {
        let start = Instant::now();
        last = Some(run());
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
    }
    (best, last.expect("at least one round"))
}

fn main() -> Result<()> {
    let robot = common::panda()?;
    let device = Device::cpu(&robot)?;
    let o = GraspOptions::default();

    let bowl = bowl();
    let (ms, found) = quickest(5, || grasps_on(&device, 0, std::slice::from_ref(&bowl), &o));
    let found = found?.remove(0);
    let wall = found.iter().filter(|g| g.width() < 0.01).count();
    println!(
        "a bowl of {} points: {} grasps, {wall} across its wall, in {ms:.1} ms",
        bowl.points.len(),
        found.len()
    );

    // A box 4 by 6 by 5 cm on a table, seen from either side.
    let table = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.01), Vec3::new(0.5, 0.5, 0.01));
    let block = Obstacle::cuboid(Vec3::new(0.0, 0.0, 0.025), Vec3::new(0.02, 0.03, 0.025));
    let scene = [table, block.clone()];
    let k = Intrinsics {
        fx: 600.0,
        fy: 600.0,
        cx: WIDTH as f32 / 2.0,
        cy: HEIGHT as f32 / 2.0,
    };
    let mut seen = Surface::default();
    let mut reading = 0.0;
    for eye in [Vec3::new(0.35, 0.0, 0.35), Vec3::new(-0.35, 0.0, 0.35)] {
        let camera = look_at(eye, Vec3::new(0.0, 0.0, 0.025));
        let depth = render(&scene, camera, k);
        let image = DepthImage {
            depth: &depth,
            width: WIDTH,
            intrinsics: k,
            camera,
        };
        // The box's pixels, as a segmentation would give them.
        let object: Vec<bool> = (0..depth.len())
            .map(|i| {
                let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
                let z = depth[i];
                let p = camera.rotation
                    * Vec3::new((u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z)
                    + camera.position;
                z > 0.0 && block.distance(p).value < 5e-4 && p.z > 1e-3
            })
            .collect();
        let (ms, surface) = quickest(5, || Surface::from_depth(&image, &object, None, 0.005));
        reading += ms;
        seen.join(surface?);
    }
    let (ms, found) = quickest(5, || grasps_on(&device, 0, std::slice::from_ref(&seen), &o));
    let found = found?.remove(0);
    let object = seen.object.iter().filter(|&&o| o).count();
    println!(
        "a box in two depth images: {} points, {object} of them the box's, read in {reading:.1} ms: {} grasps in {ms:.1} ms",
        seen.points.len(),
        found.len()
    );

    let bowls = vec![bowl; 64];
    let (ms, found) = quickest(3, || grasps_on(&device, 0, &bowls, &o));
    let grasps: usize = found?.iter().map(Vec::len).sum();
    println!(
        "64 bowls in one call: {grasps} grasps in {ms:.1} ms, {:.1} ms a bowl",
        ms / 64.0
    );
    Ok(())
}
