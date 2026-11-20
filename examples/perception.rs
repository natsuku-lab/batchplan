//! What the depth calls cost, a 640x480 image at a time: the image without the readings between
//! two surfaces, which readings are the robot's or a known shape's and which lie beside them, the
//! image's points, a surface with its normals, a grid of the image, the image fused into a map,
//! that map's grid, where it is occupied, one object's part of it, and what a caller marks in it.
//! For a camera over a table that sees the arm and for one the hand carries, with the Panda's 61
//! spheres and with it fitted to several hundred (a budget of 576).
//!
//! cargo run --release --example perception [-- --cpu [threads]]

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use anyhow::Result;
use batchplan::rng::Rng;
use batchplan::{
    DepthFilter, DepthImage, Device, Intrinsics, Occlusion, OccupancyMap, Pose, Robot, RobotMask,
    RobotOptions, SdfGrid, SdfOptions, SphereOptions, Surface, World,
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

/// z depth per pixel: the scene by sphere tracing its distance functions, and the arm's spheres
/// before it where they stand in front; 0 where a ray meets neither.
fn render(scene: &World, arm: &[[f32; 4]], camera: Pose, k: Intrinsics) -> Vec<f32> {
    (0..WIDTH * HEIGHT)
        .into_par_iter()
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let ray = Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
            let dir = camera.rotation * ray.normalize();
            let (mut t, mut hit) = (0.0, f32::INFINITY);
            while t < 4.0 {
                let p = camera.position + dir * t;
                let obstacles = scene.obstacles.iter().map(|o| o.distance(p).value);
                let d = obstacles.fold(f32::INFINITY, f32::min);
                if d < 1e-4 {
                    hit = t;
                    break;
                }
                t += d;
            }
            for &[x, y, z, r] in arm {
                let from = camera.position - Vec3::new(x, y, z);
                let (b, c) = (dir.dot(from), from.length_squared() - r * r);
                let disc = b * b - c;
                if disc >= 0.0 && -b - disc.sqrt() > 0.0 {
                    hit = hit.min(-b - disc.sqrt());
                }
            }
            if hit.is_finite() {
                hit * ray.normalize().z
            } else {
                0.0
            }
        })
        .collect()
}

/// The median and the quickest of `runs` calls (ms), and the last call's result.
fn timed<T>(runs: usize, mut call: impl FnMut() -> T) -> (f64, f64, T) {
    let mut times = vec![];
    let mut last = None;
    for _ in 0..runs {
        let start = Instant::now();
        last = Some(call());
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    let last = last.expect("at least one run");
    (times[runs / 2], times[0], last)
}

/// A grid's distances over a lattice of its box, summed: a change to the grid moves it.
fn sum(grid: &SdfGrid) -> f64 {
    let (lo, hi) = (grid.lower_corner(), grid.upper_corner());
    let lattice = (0..48 * 48 * 24).map(|n| {
        let t = Vec3::new((n % 48) as f32, (n / 48 % 48) as f32, (n / 2304) as f32);
        lo + (hi - lo) * t / Vec3::new(47.0, 47.0, 23.0)
    });
    lattice.map(|p| f64::from(grid.distance(p).value)).sum()
}

fn main() -> Result<()> {
    let panda = common::panda()?;
    let fitted = Robot::load(
        common::asset("franka/franka_panda.urdf"),
        &RobotOptions {
            collision_model: None,
            spheres: SphereOptions {
                budget: 576,
                ..SphereOptions::default()
            },
            ..common::panda_options()
        },
    )?;
    let threads = std::env::args().find_map(|a| a.parse::<usize>().ok());
    let device = match (std::env::args().any(|a| a == "--cpu"), threads) {
        (true, Some(threads)) => Device::cpu_threads(&panda, threads)?,
        (true, None) => Device::cpu(&panda)?,
        (false, _) => Device::gpu(&panda)?,
    };
    println!("device: {}", device.name());
    let scene = common::tabletop(&mut Rng::new(3));
    let q = panda.default_q().to_vec();
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    // A camera across the table from the robot, and one on the palm beside the fingers, looking
    // along them with a wider lens.
    let hand = panda.link_pose(&q, "panda_hand").expect("the Panda's hand");
    let wide = Intrinsics {
        fx: 320.0,
        fy: 320.0,
        ..k
    };
    let cameras = [
        (
            "a camera over the table",
            look_at(Vec3::new(1.4, 0.3, 0.9), Vec3::new(0.4, 0.0, 0.1)),
            k,
        ),
        (
            "a camera the hand carries",
            hand.mul_pose(Pose::new(Vec3::new(0.06, 0.0, 0.03), Quat::IDENTITY)),
            wide,
        ),
    ];
    for (name, camera, intrinsics) in cameras {
        let depth = render(&scene, &panda.spheres(&q), camera, intrinsics);
        let image = DepthImage {
            depth: &depth,
            width: WIDTH,
            intrinsics,
            camera,
        };
        let readings = depth.iter().filter(|&&z| z > 0.0).count();
        println!("\n{name}: {readings} of {} pixels read", depth.len());

        // The scene is rendered without noise: what goes is seen edge-on, or has a step on one
        // side and nothing read straight across.
        let (median, best, kept) = timed(21, || image.filtered(&DepthFilter::default()));
        let out = kept?.iter().zip(&depth).filter(|(k, z)| k != z).count();
        println!(
            "  the image without the readings between two surfaces: median {median:.2} ms, best {best:.2} ({out} left out)"
        );

        let mut masks: Vec<(String, Option<RobotMask>)> = vec![("no mask".into(), None)];
        for robot in [&panda, &fitted] {
            let label = format!("{} spheres", robot.spheres(&q).len());
            let (median, best, mask) = timed(21, || image.robot_mask(robot, &q, 0.04, &[]));
            let mask = mask?;
            let on = mask.on_robot.iter().filter(|&&on| on).count();
            println!(
                "  the robot's readings, {label}: median {median:.2} ms, best {best:.2} ({on} pixels)"
            );
            masks.push((label, Some(mask)));
        }
        let mask = masks[2].1.as_ref();
        let (median, best, points) = timed(21, || image.points(mask));
        let points = points?;
        let off = points.iter().flatten().count();
        println!(
            "  the image's points: median {median:.2} ms, best {best:.2} ({off} off the robot)"
        );

        // The table's pixels, as a segmentation would give them, and every pixel.
        let table: Vec<bool> = points
            .iter()
            .map(|p| p.is_some_and(|p| scene.obstacles[0].distance(p).value < 5e-4))
            .collect();
        for (label, object) in [("the table's", &table), ("every reading's", &vec![])] {
            let (median, best, surface) = timed(9, || {
                Surface::from_depth(&device, &image, object, mask, 0.005)
            });
            let surface = surface?;
            let normals = surface.normals.iter().filter(|n| **n != Vec3::ZERO).count();
            let sum: f64 = (surface.normals.iter().chain(&surface.points))
                .map(|n| f64::from(n.x) + 2.0 * f64::from(n.y) + 3.0 * f64::from(n.z))
                .sum();
            println!(
                "  a surface, {label} normals: median {median:.2} ms, best {best:.2} ({} points, {normals} normals, sum {sum:.6})",
                surface.points.len()
            );
        }

        let mut map =
            OccupancyMap::try_new(Vec3::new(-0.3, -0.8, -0.1), Vec3::new(1.1, 0.8, 0.7), 0.01)?;
        for (label, mask) in &masks {
            let mask = mask.as_ref();
            let (median, best, grid) = timed(9, || {
                let o = SdfOptions::default();
                SdfGrid::from_depth(&device, &image, mask, Occlusion::Occupied, &o)
            });
            let grid = grid?;
            let [nx, ny, nz] = grid.dims();
            println!(
                "  a grid of the image, {label}: median {median:.2} ms, best {best:.2} ({nx}x{ny}x{nz}, sum {:.6})",
                sum(&grid)
            );
            let (median, best, fused) = timed(9, || map.integrate(&device, &image, mask));
            fused?;
            let grid = map.grid(&device, Occlusion::OccupiedExceptRobotShadow)?;
            println!(
                "  the image fused into a map of 1.8M voxels, {label}: median {median:.2} ms, best {best:.2} (sum {:.6})",
                sum(&grid)
            );
        }
        // A map of the room around the robot, 2.9 m by 2.9 m by 1.55 m: on a CPU device a fuse
        // costs by what the camera sees of it, on a GPU by its size.
        let (lo, hi) = (Vec3::new(-1.45, -1.45, -0.1), Vec3::new(1.45, 1.45, 1.45));
        let mut room = OccupancyMap::try_new(lo, hi, 0.01)?;
        for (label, mask) in &masks[..2] {
            let (median, best, fused) = timed(9, || room.integrate(&device, &image, mask.as_ref()));
            fused?;
            let counts = room.counts();
            println!(
                "  the image fused into a map of 13.4M voxels, {label}: median {median:.2} ms, best {best:.2} ({} voxels occupied, {} free)",
                counts.occupied, counts.free
            );
        }
        // The scene's shapes left out with the robot, as a world that holds them as obstacles
        // of its own would have them: what is then fused is what no shape explains.
        let robot = masks[1].1.clone().expect("the Panda's mask");
        let (median, best, held) = timed(21, || {
            (robot.clone()).with_shapes(&device, &image, &scene.obstacles, [0.004, 0.0, 0.0])
        });
        let held = held?;
        let on = held.on_robot.iter().filter(|&&on| on).count();
        println!(
            "  the scene's {} shapes left out too: median {median:.2} ms, best {best:.2} ({on} pixels)",
            scene.obstacles.len()
        );
        // The pixel behind each of their outlines that a camera reads toward what lies behind,
        // at the shapes' margin. Here they already flag every reading the camera has.
        let (median, best, banded) = timed(21, || {
            (held.clone()).with_band(&device, &image, [0.004, 0.0, 0.0], 1)
        });
        let on = banded?.on_robot.iter().filter(|&&on| on).count();
        println!(
            "  a band of a pixel behind the robot and the shapes: median {median:.2} ms, best {best:.2} ({on} pixels)"
        );
        let mut rest =
            OccupancyMap::try_new(Vec3::new(-0.3, -0.8, -0.1), Vec3::new(1.1, 0.8, 0.7), 0.01)?;
        let (median, best, fused) = timed(9, || rest.integrate(&device, &image, Some(&held)));
        fused?;
        println!(
            "  the image fused into a map of 1.8M voxels without them: median {median:.2} ms, best {best:.2} ({} voxels occupied)",
            rest.counts().occupied
        );
        for unknown in [
            Occlusion::Free,
            Occlusion::OccupiedExceptRobotShadow,
            Occlusion::Band(0.05),
        ] {
            let (median, best, grid) = timed(9, || map.grid(&device, unknown));
            println!(
                "  that map's grid, {unknown:?}: median {median:.2} ms, best {best:.2} (sum {:.6})",
                sum(&grid?)
            );
        }
        // Where the map is occupied, one object's part of it, as the cameras saw into its box, and
        // what a caller marks.
        let (median, best, occupied) = timed(9, || map.occupied());
        println!(
            "  where the map is occupied: median {median:.2} ms, best {best:.2} ({} voxels)",
            occupied.len()
        );
        let object = &scene.obstacles[1];
        let (median, best, part) = timed(9, || map.part(object, 0.15));
        let part = part?;
        let counts = part.counts();
        println!(
            "  the map's part in a box of the scene: median {median:.2} ms, best {best:.2} ({} voxels occupied, {} free, {} never observed)",
            counts.occupied, counts.free, counts.unobserved
        );
        let (median, best, grid) = timed(9, || part.grid(&device, Occlusion::Occupied));
        let grid = grid?;
        let [nx, ny, nz] = grid.dims();
        println!(
            "  that part's grid: median {median:.2} ms, best {best:.2} ({nx}x{ny}x{nz}, sum {:.6})",
            sum(&grid)
        );
        let (median, best, marked) = timed(9, || {
            let mut copy = map.clone();
            copy.free(object, 0.01)?;
            copy.clear(object, 0.01)?;
            Ok::<_, batchplan::Error>(copy.counts())
        });
        println!(
            "  a copy of the map with the box marked free, then cleared: median {median:.2} ms, best {best:.2} ({} never observed)",
            marked?.unobserved
        );
    }
    Ok(())
}
