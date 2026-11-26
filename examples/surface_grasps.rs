//! Grasps of what cameras see, and what finding them costs. A bowl too wide to span, as points on
//! its surfaces, is grasped across its wall; a box on a table, as two 640x480 depth images, is
//! grasped across the sides the two cameras see between them; a flared bowl seen from above alone
//! across its rim, where the camera saw past it (`GraspOptions::shell`), and beside a wall the
//! open fingers would be in (`GraspOptions::narrow`); then 64 bowls in one call.
//!
//! cargo run --release --example surface_grasps

#[path = "common/mod.rs"]
mod common;

use std::f32::consts::TAU;
use std::time::Instant;

use anyhow::Result;
use batchplan::{DepthImage, Device, Grasp, GraspOptions, Intrinsics, Obstacle, Pose, Surface};
use glam::{Mat3, Quat, Vec2, Vec3};
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

/// How far `p` is from a bowl as LIBERO's, standing on z = 0: 11 cm across its rim and 5 cm tall,
/// its wall 3 mm thick, flared 26 degrees from upright and rounded at the rim.
fn flared_bowl(p: Vec3) -> f32 {
    let (half, at) = (0.0015, Vec2::new(p.truncate().length(), p.z));
    let from = |a: Vec2, b: Vec2| {
        let t = ((at - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
        at.distance(a + (b - a) * t)
    };
    let edge = Vec2::new(0.03, half);
    let wall = from(edge, Vec2::new(0.055 - half, 0.05 - half));
    wall.min(from(Vec2::new(0.0, half), edge)) - half
}

/// A camera at `eye` looking at `target`, in OpenCV axes (x right, y down, z forward).
fn look_at(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
}

/// z depth per pixel by sphere tracing the scene's distance function; 0 where a ray hits nothing.
fn render(distance: impl Fn(Vec3) -> f32 + Sync, camera: Pose, k: Intrinsics) -> Vec<f32> {
    (0..WIDTH * HEIGHT)
        .into_par_iter()
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let ray = Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
            let dir = camera.rotation * ray.normalize();
            let mut t = 0.0;
            while t < 2.0 {
                let p = camera.position + dir * t;
                let d = distance(p);
                if d < 1e-5 {
                    return t * ray.normalize().z;
                }
                t += d;
            }
            0.0
        })
        .collect()
}

/// A sum over every number of `grasps`, to tell whether two builds find the same.
fn sum(grasps: &[Grasp]) -> f64 {
    let of = |v: Vec3| f64::from(v.x) + 2.0 * f64::from(v.y) + 3.0 * f64::from(v.z);
    let each = grasps.iter().enumerate().map(|(k, g)| {
        let turn = g.pose.rotation * Vec3::new(1.0, 2.0, 3.0);
        (k % 7 + 1) as f64
            * (of(g.pose.position) + of(turn) + of(g.contacts[0]) + of(g.contacts[1]))
    });
    each.sum()
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
    let first = GraspOptions {
        first_touch: true,
        ..o
    };
    // The same surfaces' grasps, those whose fingers close onto their patches first.
    let first_touch = |surfaces: &[Surface], rounds: usize| -> Result<()> {
        let (ms, found) = quickest(rounds, || {
            batchplan::grasps_on(&device, 0, surfaces, &first)
        });
        let found: Vec<Grasp> = found?.into_iter().flatten().collect();
        println!(
            "  with first_touch: {} grasps in {ms:.1} ms (sum {:.6})",
            found.len(),
            sum(&found)
        );
        Ok(())
    };

    let bowl = bowl();
    let (ms, found) = quickest(5, || {
        batchplan::grasps_on(&device, 0, std::slice::from_ref(&bowl), &o)
    });
    let found = found?.remove(0);
    let wall = found.iter().filter(|g| g.width() < 0.01).count();
    println!(
        "a bowl of {} points: {} grasps, {wall} across its wall, in {ms:.1} ms (sum {:.6})",
        bowl.points.len(),
        found.len(),
        sum(&found)
    );
    first_touch(std::slice::from_ref(&bowl), 5)?;

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
        let depth = render(
            |p| {
                scene
                    .iter()
                    .map(|o| o.distance(p).value)
                    .fold(f32::INFINITY, f32::min)
            },
            camera,
            k,
        );
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
        let (ms, surface) = quickest(5, || {
            Surface::from_depth(&device, &image, &object, None, 0.005)
        });
        reading += ms;
        seen.join(surface?);
    }
    let (ms, found) = quickest(5, || {
        batchplan::grasps_on(&device, 0, std::slice::from_ref(&seen), &o)
    });
    let found = found?.remove(0);
    let object = seen.object.iter().filter(|&&o| o).count();
    println!(
        "a box in two depth images: {} points, {object} of them the box's, read in {reading:.1} ms: {} grasps in {ms:.1} ms (sum {:.6})",
        seen.points.len(),
        found.len(),
        sum(&found)
    );
    first_touch(std::slice::from_ref(&seen), 5)?;

    // LIBERO's bowl on the table, seen from 40 cm above by a wrist camera alone: its rim and its
    // inner wall, not its outer wall, which flares out of sight under the rim.
    let table = scene[0].clone();
    let wrist = Intrinsics {
        fx: 313.0,
        fy: 313.0,
        ..k
    };
    let camera = look_at(Vec3::new(0.02, 0.0, 0.42), Vec3::new(0.0, 0.0, 0.025));
    // The bowl among what else `around` holds, as the camera sees them.
    let from_above = |around: &(dyn Fn(Vec3) -> f32 + Sync)| {
        let depth = render(|p| flared_bowl(p).min(around(p)), camera, wrist);
        let image = DepthImage {
            depth: &depth,
            width: WIDTH,
            intrinsics: wrist,
            camera,
        };
        let object: Vec<bool> = (0..depth.len())
            .map(|i| {
                let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
                let z = depth[i];
                let p = camera.rotation
                    * Vec3::new(
                        (u - wrist.cx) / wrist.fx * z,
                        (v - wrist.cy) / wrist.fy * z,
                        z,
                    )
                    + camera.position;
                z > 0.0 && flared_bowl(p) < 5e-4 && p.z > 1e-3
            })
            .collect();
        Surface::from_depth(&device, &image, &object, None, 0.005)
    };
    let seen = from_above(&|p| table.distance(p).value)?;
    let without = batchplan::grasps_on(&device, 0, std::slice::from_ref(&seen), &o)?.remove(0);
    let shell = GraspOptions {
        shell: Some(0.015),
        ..o
    };
    let (ms, found) = quickest(5, || {
        batchplan::grasps_on(&device, 0, std::slice::from_ref(&seen), &shell)
    });
    let found = found?.remove(0);
    // How wide they are, and those that press on the inner wall within 5 mm of the rim.
    let range = |widths: &mut dyn Iterator<Item = f32>| {
        widths.fold((f32::INFINITY, 0f32), |(a, b), w| {
            (a.min(w * 1e3), b.max(w * 1e3))
        })
    };
    let (narrowest, widest) = range(&mut found.iter().map(Grasp::width));
    let at_the_rim = |g: &&Grasp| g.contacts[0].z.max(g.contacts[1].z) > 0.045;
    let (low, high) = range(&mut found.iter().filter(at_the_rim).map(Grasp::width));
    println!(
        "a flared bowl seen from above: {} points, {} of them the bowl's: {} grasps, {} with a shell of 1.5 cm in {ms:.1} ms, {narrowest:.1} to {widest:.1} mm wide, {low:.1} to {high:.1} at its rim (sum {:.6})",
        seen.points.len(),
        seen.object.iter().filter(|&&o| o).count(),
        without.len(),
        found.len(),
        sum(&found)
    );

    // The bowl beside a cabinet: a wall twice as tall, 4 cm from its rim, which the open fingers
    // of a grasp on that side would be in. Fingers that may come in narrower pass between them.
    let cabinet = Obstacle::cuboid(Vec3::new(0.105, 0.0, 0.05), Vec3::new(0.01, 0.15, 0.05));
    let seen = from_above(&|p| table.distance(p).value.min(cabinet.distance(p).value))?;
    let (open_ms, open) = quickest(5, || {
        batchplan::grasps_on(&device, 0, std::slice::from_ref(&seen), &shell)
    });
    let open = open?.remove(0);
    let narrow = GraspOptions {
        narrow: true,
        ..shell
    };
    let (ms, found) = quickest(5, || {
        batchplan::grasps_on(&device, 0, std::slice::from_ref(&seen), &narrow)
    });
    let found = found?.remove(0);
    let narrower = found.iter().filter(|g| g.opening < o.opening).count();
    let least = found
        .iter()
        .map(|g| g.opening)
        .fold(f32::INFINITY, f32::min);
    println!(
        "the bowl 4 cm from a wall: {} grasps with the fingers open in {open_ms:.1} ms, {} where they may come in narrower in {ms:.1} ms, {narrower} of them narrower, down to {:.1} cm (sum {:.6})",
        open.len(),
        found.len(),
        least * 100.0,
        sum(&found) + found.iter().map(|g| f64::from(g.opening)).sum::<f64>()
    );

    let bowls = vec![bowl; 64];
    let (ms, found) = quickest(3, || batchplan::grasps_on(&device, 0, &bowls, &o));
    let grasps: usize = found?.iter().map(Vec::len).sum();
    println!(
        "64 bowls in one call: {grasps} grasps in {ms:.1} ms, {:.1} ms a bowl",
        ms / 64.0
    );
    first_touch(&bowls, 3)?;
    Ok(())
}
