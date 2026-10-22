//! Signed distance grids: interpolation and its gradients, the conservative offset, the mesh,
//! point-cloud and depth-image builders, robot masking, fused occupancy maps, and scene meshes
//! loaded as grids.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::PI;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use batchplan::rng::Rng;
use batchplan::{
    CollisionWeights, DepthImage, Device, Intrinsics, JointTrajectory, Obstacle, Occlusion,
    OccupancyMap, Pose, Robot, SdfGrid, SdfOptions, World,
};
use glam::{Quat, Vec3};

/// A CPU device for the builders' distance transform, which needs no particular robot.
fn cpu() -> &'static Device {
    &devices()[0]
}

/// The CPU device, then the GPU's if there is one.
fn devices() -> &'static [Device] {
    static DEVICES: OnceLock<Vec<Device>> = OnceLock::new();
    DEVICES.get_or_init(|| {
        let robot = common::panda().unwrap();
        let mut devices = vec![Device::cpu(&robot).unwrap()];
        match Device::gpu(&robot) {
            Ok(gpu) => devices.push(gpu),
            Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => {
                eprintln!("skipping GPU: {e}")
            }
            Err(e) => panic!("no GPU: {e}"),
        }
        devices
    })
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("batchplan-sdf-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// An axis-aligned box as a closed, outward-facing triangle mesh, appended to `mesh`.
fn add_box(mesh: &mut (Vec<Vec3>, Vec<[u32; 3]>), lo: Vec3, hi: Vec3) {
    let base = mesh.0.len() as u32;
    // Vertex i takes the `hi` coordinate where bit (x, y, z) = (1, 2, 4) of i is set.
    mesh.0.extend(
        (0..8).map(|i| Vec3::select(glam::BVec3::new(i & 1 != 0, i & 2 != 0, i & 4 != 0), hi, lo)),
    );
    let faces = [
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5],
        [0, 1, 5],
        [0, 5, 4],
    ];
    let faces = faces.into_iter().chain([
        [2, 6, 7],
        [2, 7, 3],
        [0, 2, 3],
        [0, 3, 1],
        [4, 5, 7],
        [4, 7, 6],
    ]);
    mesh.1.extend(faces.map(|t: [u32; 3]| t.map(|i| i + base)));
}

fn box_distance(lo: Vec3, hi: Vec3, p: Vec3) -> f32 {
    let (center, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
    Obstacle::cuboid(center, half).distance(p).0
}

/// Half a voxel diagonal: how much built grids lower distances.
fn shift(voxel: f32) -> f32 {
    voxel * 3f32.sqrt() / 2.0
}

/// How far below the true distance grids from points and depth images may read: the surface is
/// known to the voxel holding it, and every step rounds toward collision.
fn occupancy_slack(voxel: f32) -> f32 {
    3.0 * voxel
}

#[test]
fn grid_gradients_match_finite_differences() {
    let (lo, hi) = (Vec3::new(-0.1, -0.05, 0.0), Vec3::new(0.1, 0.05, 0.08));
    let mut mesh = (vec![], vec![]);
    add_box(&mut mesh, lo, hi);
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.05,
    };
    let grid = Arc::new(SdfGrid::from_mesh(&mesh.0, &mesh.1, &o).unwrap());
    let rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.5, 0.9);
    let center = Vec3::new(0.4, -0.2, 0.3);
    let obstacle = Obstacle::Sdf {
        grid: grid.clone(),
        center,
        rotation,
    };
    let (glo, ghi) = grid.bounds();
    let mut rng = Rng::new(3);
    let h = 2e-4;
    let (mut inside_box, mut outside_box, mut worst) = (0, 0, 0.0f32);
    while inside_box + outside_box < 400 {
        // Points in the grid's frame, some beyond its box, kept away from cell faces where the
        // interpolation's gradient jumps.
        let local = Vec3::new(
            rng.range(-0.2, 0.2),
            rng.range(-0.15, 0.15),
            rng.range(-0.1, 0.2),
        );
        let cell = (local - glo) / o.voxel;
        let fraction = cell - cell.floor();
        if fraction.min_element() < 2.0 * h / o.voxel
            || fraction.max_element() > 1.0 - 2.0 * h / o.voxel
        {
            continue;
        }
        let beyond = local.cmplt(glo) | local.cmpgt(ghi);
        let near_box_face = ((local - glo).abs().min((local - ghi).abs())).min_element() < 2.0 * h;
        if near_box_face {
            continue;
        }
        let p = center + rotation * local;
        let (_, g) = obstacle.distance(p);
        let fd = Vec3::from_array(std::array::from_fn(|k| {
            let e = Vec3::AXES[k] * h;
            (obstacle.distance(p + e).0 - obstacle.distance(p - e).0) / (2.0 * h)
        }));
        worst = worst.max((fd - g).length() / g.length().max(0.1));
        if beyond.any() {
            outside_box += 1
        } else {
            inside_box += 1
        }
    }
    assert!(
        outside_box > 20,
        "only {outside_box} points beyond the grid's box"
    );
    assert!(worst < 5e-3, "gradient off by {worst} of its size");
}

#[test]
fn grid_collision_gradients_match_finite_differences() {
    let robot = common::panda().unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let mut mesh = (vec![], vec![]);
    add_box(
        &mut mesh,
        Vec3::new(-0.06, -0.06, 0.0),
        Vec3::new(0.06, 0.06, 0.4),
    );
    let grid = Arc::new(SdfGrid::from_mesh(&mesh.0, &mesh.1, &SdfOptions::default()).unwrap());
    let post = Obstacle::Sdf {
        grid,
        center: Vec3::new(0.45, 0.0, 0.0),
        rotation: Quat::from_rotation_z(0.4),
    };
    let worlds = cpu
        .upload(&[World {
            obstacles: vec![post],
        }])
        .unwrap();
    let w = CollisionWeights {
        world: 1000.0,
        self_collision: 0.0,
        margin: 0.05,
        self_margin: 0.0,
    };
    let mut rng = Rng::new(17);
    let mut checked = 0;
    while checked < 60 {
        let q: Vec<f32> = (0..n)
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect();
        let e = cpu.evaluate(&worlds, &[0], &q, &w).unwrap();
        if e.cost[0] < 1e-3 {
            continue;
        }
        let norm = e.grad.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
        // Spheres crossing cell faces make the cost only piecewise smooth; accept either step.
        let rel_err = |h: f32| {
            let fd = (0..n).map(|j| {
                let (mut qp, mut qm) = (q.clone(), q.clone());
                qp[j] += h;
                qm[j] -= h;
                let cost = |q: &[f32]| cpu.evaluate(&worlds, &[0], q, &w).unwrap().cost[0];
                (cost(&qp) - cost(&qm)) / (2.0 * h)
            });
            fd.zip(&e.grad)
                .map(|(f, g)| (f - g).powi(2))
                .sum::<f32>()
                .sqrt()
                / norm
        };
        let err = rel_err(2e-4).min(rel_err(5e-5));
        assert!(err < 3e-2, "gradient off by {err} of |g| = {norm} at {q:?}");
        checked += 1;
    }
}

#[test]
fn thin_walls_leak_through_raw_samples_but_not_through_built_grids() {
    // A 2 mm wall midway between grid points 1 cm apart. A second box sets where the grid starts.
    let voxel = 0.01;
    let (wall_lo, wall_hi) = (Vec3::new(0.004, -0.2, -0.2), Vec3::new(0.006, 0.2, 0.2));
    let mut mesh = (vec![], vec![]);
    add_box(
        &mut mesh,
        Vec3::new(-0.3, -0.2, -0.2),
        Vec3::new(-0.2, -0.1, -0.1),
    );
    add_box(&mut mesh, wall_lo, wall_hi);
    let o = SdfOptions {
        voxel,
        padding: 0.05,
    };
    let built = SdfGrid::from_mesh(&mesh.0, &mesh.1, &o).unwrap();
    let (origin, _) = built.bounds();
    assert!(
        (((0.005 - origin.x) / voxel).fract() - 0.5).abs() < 1e-3,
        "the wall is not midway between points"
    );
    // The same grid points holding exact distances, without the conservative offset.
    let dims = built.dims();
    let mut exact = vec![];
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let p = origin + Vec3::new(i as f32, j as f32, k as f32) * voxel;
                exact.push(box_distance(wall_lo, wall_hi, p).min(box_distance(
                    Vec3::new(-0.3, -0.2, -0.2),
                    Vec3::new(-0.2, -0.1, -0.1),
                    p,
                )));
            }
        }
    }
    let raw = SdfGrid::new(dims, voxel, origin, &exact).unwrap();
    let mut rng = Rng::new(5);
    let (mut leaked, mut worst_inside, mut worst_growth) = (0.0f32, f32::MIN, 0.0f32);
    for _ in 0..500 {
        let inside = Vec3::new(
            rng.range(0.004, 0.006),
            rng.range(-0.15, 0.15),
            rng.range(-0.15, 0.15),
        );
        leaked = leaked.max(raw.distance(inside).0);
        worst_inside = worst_inside.max(built.distance(inside).0);
        let near = Vec3::new(
            rng.range(0.03, 0.1),
            rng.range(-0.15, 0.15),
            rng.range(-0.15, 0.15),
        );
        let truth = box_distance(wall_lo, wall_hi, near);
        let (d, _) = built.distance(near);
        assert!(
            d <= truth + 1e-4,
            "built grid reads {d} where the wall is {truth} away"
        );
        worst_growth = worst_growth.max(truth - d);
    }
    // Interpolating exact samples across the wall reads several millimeters of clearance inside it.
    assert!(
        leaked > 0.003,
        "raw samples should leak, read at most {leaked}"
    );
    // Built grids block it, at the price of growing obstacles by up to a voxel diagonal.
    assert!(
        worst_inside <= 0.0,
        "the built grid reads {worst_inside} inside the wall"
    );
    assert!(
        worst_growth <= 2.0 * shift(voxel) + 1e-4,
        "obstacles grew by {worst_growth}"
    );
}

#[test]
fn mesh_grids_are_conservative_within_a_voxel_diagonal() {
    let (lo, hi) = (Vec3::new(-0.1, -0.05, 0.0), Vec3::new(0.1, 0.05, 0.08));
    let mut mesh = (vec![], vec![]);
    add_box(&mut mesh, lo, hi);
    // Inward-facing triangles describe the same solid.
    let flipped: Vec<[u32; 3]> = mesh.1.iter().map(|t| [t[0], t[2], t[1]]).collect();
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.06,
    };
    let grid = SdfGrid::from_mesh(&mesh.0, &mesh.1, &o).unwrap();
    assert_eq!(grid, SdfGrid::from_mesh(&mesh.0, &flipped, &o).unwrap());
    let mut rng = Rng::new(8);
    for _ in 0..2000 {
        let p = Vec3::new(
            rng.range(-0.16, 0.16),
            rng.range(-0.11, 0.11),
            rng.range(-0.06, 0.14),
        );
        let (truth, d) = (box_distance(lo, hi, p), grid.distance(p).0);
        assert!(
            d <= truth + 1e-4 && d >= truth - 2.0 * shift(o.voxel) - 1e-4,
            "{d} at {p}, truth {truth}"
        );
    }
    // Beyond the padded box, distances keep growing.
    let far = Vec3::new(0.5, 0.0, 0.04);
    assert!(grid.distance(far).0 > 0.35, "{}", grid.distance(far).0);
}

#[test]
fn open_and_non_manifold_meshes_are_refused() {
    let mut mesh = (vec![], vec![]);
    add_box(&mut mesh, Vec3::ZERO, Vec3::ONE * 0.1);
    let o = SdfOptions::default();
    let open: Vec<[u32; 3]> = mesh.1[2..].to_vec();
    let err = SdfGrid::from_mesh(&mesh.0, &open, &o)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not closed"), "{err}");
    // A fin sharing an edge with two box faces: three triangles on one edge.
    let mut fin = mesh.clone();
    fin.0.push(Vec3::new(-0.1, -0.1, 0.0));
    fin.1.push([0, 4, 8]);
    let err = SdfGrid::from_mesh(&fin.0, &fin.1, &o)
        .unwrap_err()
        .to_string();
    assert!(err.contains("manifold"), "{err}");
    assert!(
        SdfGrid::from_mesh(&mesh.0, &[[0, 1, 99]], &o).is_err(),
        "vertex index out of range"
    );
    let huge = SdfOptions {
        voxel: 1e-4,
        padding: 1.0,
    };
    assert!(
        SdfGrid::from_mesh(&mesh.0, &mesh.1, &huge)
            .unwrap_err()
            .to_string()
            .contains("too large")
    );
}

#[test]
fn point_clouds_become_solid_shells() {
    // Points on a 10 cm sphere.
    let (center, radius) = (Vec3::new(0.3, -0.1, 0.2), 0.1);
    let mut rng = Rng::new(2);
    let points: Vec<Vec3> = (0..20_000)
        .map(|_| {
            let (z, a) = (rng.range(-1.0, 1.0), rng.range(-PI, PI));
            let r = (1.0 - z * z).sqrt();
            center + radius * Vec3::new(r * a.cos(), r * a.sin(), z)
        })
        .collect();
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.1,
    };
    let grid = SdfGrid::from_points(cpu(), &points, &o).unwrap();
    for &p in points.iter().step_by(97) {
        assert!(
            grid.distance(p).0 <= 0.0,
            "a surface point reads {}",
            grid.distance(p).0
        );
    }
    for _ in 0..500 {
        let dir = Vec3::new(rng.normal(), rng.normal(), rng.normal()).normalize();
        let truth = rng.range(0.02, 0.08);
        let d = grid.distance(center + dir * (radius + truth)).0;
        assert!(
            d <= truth + 2e-3 && d >= truth - occupancy_slack(o.voxel),
            "{d} for {truth}"
        );
    }
    // Inside the shell is unobserved and reads as free.
    assert!(grid.distance(center).0 > 0.05);
}

#[test]
fn point_grids_never_read_beyond_their_solid_voxels() {
    // A few scattered points make sharp creases in the distance field, where interpolation
    // errs most; the grid must still read at most the distance to the voxels holding points.
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.03,
    };
    let mut rng = Rng::new(6);
    let mut worst = f32::MIN;
    for _ in 0..40 {
        let points: Vec<Vec3> = (0..3)
            .map(|_| {
                Vec3::new(
                    rng.range(0.0, 0.03),
                    rng.range(0.0, 0.03),
                    rng.range(0.0, 0.03),
                )
            })
            .collect();
        let grid = SdfGrid::from_points(cpu(), &points, &o).unwrap();
        let (origin, hi) = grid.bounds();
        let cubes: Vec<Vec3> = points
            .iter()
            .map(|&p| origin + ((p - origin) / o.voxel).round() * o.voxel)
            .collect();
        for _ in 0..2000 {
            let p = Vec3::new(
                rng.range(origin.x, hi.x),
                rng.range(origin.y, hi.y),
                rng.range(origin.z, hi.z),
            );
            let truth = cubes
                .iter()
                .map(|&c| box_distance(c - 0.5 * o.voxel, c + 0.5 * o.voxel, p))
                .fold(f32::INFINITY, f32::min);
            worst = worst.max(grid.distance(p).0 - truth);
        }
    }
    assert!(
        worst <= 1e-4,
        "the grid reads {worst} beyond the solid voxels"
    );
}

/// A pinhole camera 1 m above a table (z = 0) looking straight down, with a 10 cm ball on it.
struct TableScene {
    camera: Pose,
    ball: (Vec3, f32),
}

impl TableScene {
    fn new() -> Self {
        let camera = Pose::new(Vec3::new(0.0, 0.0, 1.0), Quat::from_rotation_x(PI));
        Self {
            camera,
            ball: (Vec3::new(0.05, 0.1, 0.1), 0.1),
        }
    }

    /// `depth` as seen by this scene's camera.
    fn image<'a>(&self, depth: &'a [f32], width: usize, intrinsics: Intrinsics) -> DepthImage<'a> {
        DepthImage {
            depth,
            width,
            intrinsics,
            camera: self.camera,
        }
    }

    /// z depth per pixel; `with_ball` adds the ball.
    fn render(&self, width: usize, height: usize, k: Intrinsics, with_ball: bool) -> Vec<f32> {
        let mut depth = vec![];
        for v in 0..height {
            for u in 0..width {
                let ray = Vec3::new((u as f32 - k.cx) / k.fx, (v as f32 - k.cy) / k.fy, 1.0);
                let dir = self.camera.rotation * ray;
                let origin = self.camera.position;
                let mut t = -origin.z / dir.z;
                let (c, r) = self.ball;
                let (b, cc) = (dir.dot(origin - c), (origin - c).length_squared() - r * r);
                let disc = b * b - dir.length_squared() * cc;
                if with_ball && disc >= 0.0 {
                    t = t.min((-b - disc.sqrt()) / dir.length_squared());
                }
                depth.push(t);
            }
        }
        depth
    }
}

/// Checks a grid of the bare table: distances above it, the occlusion rule below it, free space
/// outside the view.
fn check_table(grid: &SdfGrid, behind: Occlusion, o: &SdfOptions, rng: &mut Rng) {
    for _ in 0..300 {
        let height = rng.range(0.03, 0.25);
        let p = Vec3::new(rng.range(-0.25, 0.25), rng.range(-0.25, 0.25), height);
        let d = grid.distance(p).0;
        assert!(
            d <= height + 1e-4 && d >= height - occupancy_slack(o.voxel),
            "{d} at {p}"
        );
        let below = grid.distance(Vec3::new(p.x, p.y, -0.05)).0;
        match behind {
            // The table, not a robot, hides what is below it.
            Occlusion::Occupied | Occlusion::OccupiedExceptRobotShadow => {
                assert!(below < 0.0, "{below} below the table at {p}")
            }
            Occlusion::Free => assert!(below > 0.02, "{below} below the table at {p}"),
        }
    }
    // Outside the view nothing was seen, so it is free even under the occupied rule.
    let (_, hi) = grid.bounds();
    let aside = Vec3::new(hi.x - 0.005, 0.0, -0.1);
    assert!(
        grid.distance(aside).0 > 0.02,
        "{} outside the view",
        grid.distance(aside).0
    );
}

#[test]
fn depth_images_of_any_resolution_become_grids() {
    for d in devices() {
        depth_images_become_grids_on(d);
    }
}

fn depth_images_become_grids_on(device: &Device) {
    let scene = TableScene::new();
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.15,
    };
    let mut rng = Rng::new(4);
    // An 8x8 time-of-flight array with a 45 degree field of view: each zone covers about 10 cm
    // of the table, ten voxels across.
    let f = 4.0 / (22.5f32).to_radians().tan();
    let tof = Intrinsics {
        fx: f,
        fy: f,
        cx: 3.5,
        cy: 3.5,
    };
    // A 640x480 camera.
    let vga = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    for (width, height, k) in [(8, 8, tof), (640, 480, vga)] {
        let depth = scene.render(width, height, k, false);
        for behind in [
            Occlusion::Occupied,
            Occlusion::OccupiedExceptRobotShadow,
            Occlusion::Free,
        ] {
            let grid =
                SdfGrid::from_depth(device, &scene.image(&depth, width, k), None, behind, &o)
                    .unwrap();
            check_table(&grid, behind, &o, &mut rng);
        }
    }
    // The full frame resolves the ball: clearance above it, and inside it as the rule says.
    let depth = scene.render(640, 480, vga, true);
    let (c, r) = scene.ball;
    for (behind, inside) in [(Occlusion::Occupied, -0.05), (Occlusion::Free, 0.05)] {
        let grid =
            SdfGrid::from_depth(device, &scene.image(&depth, 640, vga), None, behind, &o).unwrap();
        let above = grid.distance(c + Vec3::new(0.0, 0.0, r + 0.05)).0;
        assert!(
            above <= 0.05 + 2e-3 && above > 0.05 - occupancy_slack(o.voxel),
            "{above} above the ball"
        );
        let center = grid.distance(c).0;
        assert!(
            center.signum() == f32::signum(inside),
            "{center} at the ball's center under {behind:?}"
        );
        // As an obstacle, the grid is where the camera saw it.
        let obstacle = Obstacle::from(grid);
        assert_eq!(
            [
                obstacle.distance(c).0,
                obstacle.distance(c + Vec3::new(0.0, 0.0, r + 0.05)).0
            ],
            [center, above]
        );
    }
    // Pixels without a reading observe nothing. Below the table, zone (4, 4) is centered at
    // x = 0.054 and zone (5, 4) at x = 0.163 (y = -0.054 for both; the camera's y axis points
    // along the world's -y).
    let mut holes = scene.render(8, 8, tof, false);
    holes.iter_mut().step_by(2).for_each(|z| *z = 0.0);
    let grid = SdfGrid::from_depth(
        device,
        &scene.image(&holes, 8, tof),
        None,
        Occlusion::Occupied,
        &o,
    )
    .unwrap();
    assert!(
        grid.distance(Vec3::new(0.054, -0.054, -0.05)).0 > 0.0,
        "below a pixel without a reading"
    );
    assert!(
        grid.distance(Vec3::new(0.163, -0.054, -0.05)).0 < 0.0,
        "below a pixel with one"
    );
    let all_holes = vec![f32::NAN; 64];
    assert!(
        SdfGrid::from_depth(
            device,
            &scene.image(&all_holes, 8, tof),
            None,
            Occlusion::Free,
            &o
        )
        .is_err()
    );
    assert!(
        SdfGrid::from_depth(
            device,
            &scene.image(&depth, 7, vga),
            None,
            Occlusion::Free,
            &o
        )
        .is_err(),
        "rows of 7"
    );
}

/// An L-shaped prism 10 cm tall whose convex hull also fills the notch at x, y in (0.1, 0.2).
fn l_prism_obj() -> String {
    let outline = [
        (0.0, 0.0),
        (0.2, 0.0),
        (0.2, 0.1),
        (0.1, 0.1),
        (0.1, 0.2),
        (0.0, 0.2),
    ];
    let mut obj = String::new();
    for z in [0.0, 0.1] {
        for (x, y) in outline {
            obj += &format!("v {x} {y} {z}\n");
        }
    }
    // Bottom faces down, top faces up (OBJ indices start at 1).
    for f in [[1, 4, 2], [2, 4, 3], [1, 6, 5], [1, 5, 4]] {
        obj += &format!("f {} {} {}\n", f[0], f[1], f[2]);
    }
    for f in [[7, 8, 10], [8, 9, 10], [7, 11, 12], [7, 10, 11]] {
        obj += &format!("f {} {} {}\n", f[0], f[1], f[2]);
    }
    for i in 1..=6 {
        let j = i % 6 + 1;
        obj += &format!("f {i} {j} {}\nf {i} {} {}\n", j + 6, j + 6, i + 6);
    }
    obj
}

#[test]
fn mjcf_scene_meshes_load_as_grids_of_their_convex_hulls() {
    let dir = scratch("mjcf");
    std::fs::write(dir.join("l.obj"), l_prism_obj()).unwrap();
    let scene = r#"<mujoco>
      <asset><mesh name="l" file="l.obj"/></asset>
      <worldbody><geom type="mesh" mesh="l" pos="0.5 0 0" euler="0 0 90"/></worldbody>
    </mujoco>"#;
    std::fs::write(dir.join("scene.xml"), scene).unwrap();
    let o = SdfOptions {
        voxel: 0.005,
        padding: 0.1,
    };
    let world = World::load(dir.join("scene.xml"), &o).unwrap();
    assert!(matches!(world.obstacles[..], [Obstacle::Sdf { .. }]));
    // Rotated a quarter turn about z: local (x, y) lies at world (0.5 - y, x).
    let at = |x: f32, y: f32, z: f32| world.obstacles[0].distance(Vec3::new(0.5 - y, x, z)).0;
    assert!(at(0.05, 0.05, 0.05) < -0.04, "inside the L");
    assert!(
        at(0.14, 0.14, 0.05) < 0.0,
        "MuJoCo collides the notch through the hull"
    );
    assert!(
        (at(0.05, 0.05, 0.15) - 0.05).abs() < 0.01,
        "5 cm above the top"
    );
}

#[cfg(feature = "usd")]
#[test]
fn usd_scene_meshes_load_as_grids_honoring_the_approximation() {
    let dir = scratch("usd");
    let outline = "(0, 0, 0), (0.2, 0, 0), (0.2, 0.1, 0), (0.1, 0.1, 0), (0.1, 0.2, 0), (0, 0.2, 0), \
        (0, 0, 0.1), (0.2, 0, 0.1), (0.2, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.2, 0.1), (0, 0.2, 0.1)";
    // Bottom and top as hexagons, then the six sides.
    let counts = "[6, 6, 4, 4, 4, 4, 4, 4]";
    let indices = "[0, 5, 4, 3, 2, 1, 6, 7, 8, 9, 10, 11, 0, 1, 7, 6, 1, 2, 8, 7, 2, 3, 9, 8, 3, 4, 10, 9, \
        4, 5, 11, 10, 5, 0, 6, 11]";
    let mesh = |name: &str, x: f32, approximation: &str| {
        format!(
            r#"    def Mesh "{name}" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"])
    {{
        point3f[] points = [{outline}]
        int[] faceVertexCounts = {counts}
        int[] faceVertexIndices = {indices}
        uniform token physics:approximation = "{approximation}"
        double3 xformOp:translate = ({x}, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
"#
        )
    };
    let stage = format!(
        "#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n\ndef Xform \"scene\"\n{{\n{}{}}}\n",
        mesh("exact", 0.0, "none"),
        mesh("hull", 1.0, "convexHull")
    );
    std::fs::write(dir.join("scene.usda"), stage).unwrap();
    let world = World::load(
        dir.join("scene.usda"),
        &SdfOptions {
            voxel: 0.005,
            padding: 0.1,
        },
    )
    .unwrap();
    assert_eq!(world.obstacles.len(), 2);
    let notch = |o: &Obstacle, x: f32| o.distance(Vec3::new(x + 0.15, 0.15, 0.05)).0;
    assert!(
        notch(&world.obstacles[0], 0.0) > 0.0,
        "the exact mesh leaves the notch free"
    );
    assert!(
        notch(&world.obstacles[1], 1.0) < 0.0,
        "the convex hull fills it"
    );
}

/// A two-frame demonstration in `world` that stays at the default pose.
fn still_demo(robot: &Robot, world: u32) -> batchplan::datagen::Demonstration {
    let q = robot.default_q();
    let trajectory = JointTrajectory {
        dof: q.len(),
        dt: 0.1,
        duration: 0.1,
        positions: [q, q].concat(),
        velocities: vec![0.0; 2 * q.len()],
        accelerations: vec![0.0; 2 * q.len()],
    };
    batchplan::datagen::Demonstration {
        origin: batchplan::datagen::Origin::Nominal,
        world,
        task: batchplan::datagen::REACH_TASK.into(),
        goal: robot.ee_pose(q, 0),
        trajectory,
        gripper: vec![1.0; 2],
        carried: None,
    }
}

#[test]
fn exports_write_each_grid_once() {
    let robot = common::panda().unwrap();
    let mut mesh = (vec![], vec![]);
    add_box(&mut mesh, Vec3::ZERO, Vec3::splat(0.1));
    let grid = Arc::new(SdfGrid::from_mesh(&mesh.0, &mesh.1, &SdfOptions::default()).unwrap());
    let shared = |x: f32| Obstacle::Sdf {
        grid: grid.clone(),
        center: Vec3::new(x, 0.5, 0.0),
        rotation: Quat::IDENTITY,
    };
    let worlds = vec![
        World {
            obstacles: vec![shared(0.3)],
        },
        World {
            obstacles: vec![shared(0.5), shared(0.6)],
        },
    ];
    let root = scratch("export");
    batchplan::npy::export(
        &root,
        &robot,
        &worlds,
        &[still_demo(&robot, 1)],
        &Default::default(),
    )
    .unwrap();
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("worlds.json")).unwrap()).unwrap();
    assert_eq!(json["grids"].as_array().unwrap().len(), 1);
    let grid_of = |w: usize, o: usize| json["worlds"][w]["obstacles"][o]["grid"].as_u64();
    assert_eq!([grid_of(0, 0), grid_of(1, 0), grid_of(1, 1)], [Some(0); 3]);
    let back: SdfGrid = serde_json::from_value(json["grids"][0].clone()).unwrap();
    assert_eq!(back, *grid);
}

#[cfg(feature = "lerobot")]
#[test]
fn lerobot_exports_describe_grids_by_their_boxes() {
    use arrow_array::{Array, FixedSizeListArray, Float32Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let robot = common::panda().unwrap();
    let mut mesh = (vec![], vec![]);
    add_box(&mut mesh, Vec3::ZERO, Vec3::new(0.2, 0.1, 0.1));
    let grid = Arc::new(SdfGrid::from_mesh(&mesh.0, &mesh.1, &SdfOptions::default()).unwrap());
    let (center, rotation) = (Vec3::new(0.5, 0.2, 0.0), Quat::from_rotation_z(0.5));
    let worlds = vec![World {
        obstacles: vec![Obstacle::Sdf {
            grid: grid.clone(),
            center,
            rotation,
        }],
    }];
    let root = scratch("lerobot");
    batchplan::lerobot::export(
        &root,
        &robot,
        &worlds,
        &[still_demo(&robot, 0)],
        &Default::default(),
    )
    .unwrap();
    let file = std::fs::File::open(root.join("data/chunk-000/file-000.parquet")).unwrap();
    let batch = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let column = batch
        .column_by_name("observation.environment_state")
        .unwrap();
    let rows = column
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    let first = rows.value(0);
    let values = first
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .values();
    let (lo, hi) = grid.bounds();
    let (mid, half) = (center + rotation * (0.5 * (lo + hi)), 0.5 * (hi - lo));
    let expected = [
        1.0, 4.0, mid.x, mid.y, mid.z, half.x, half.y, half.z, rotation.x, rotation.y, rotation.z,
        rotation.w,
    ];
    for (got, want) in values[7..19].iter().zip(expected) {
        assert!(
            (got - want).abs() < 1e-6,
            "{:?} vs {expected:?}",
            &values[7..19]
        );
    }
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("meta/batchplan.json")).unwrap()).unwrap();
    assert_eq!(meta["worlds"]["grids"].as_array().unwrap().len(), 1);
}

#[test]
fn deserialized_grids_are_checked_like_built_ones() {
    let mut mesh = (vec![], vec![]);
    add_box(&mut mesh, Vec3::ZERO, Vec3::splat(0.1));
    let grid = SdfGrid::from_mesh(&mesh.0, &mesh.1, &SdfOptions::default()).unwrap();
    let mut json = serde_json::to_value(&grid).unwrap();
    assert_eq!(
        serde_json::from_value::<SdfGrid>(json.clone()).unwrap(),
        grid
    );
    // A grid missing values would index past its data on the CPU and read the next grid on a GPU.
    json["values"].as_array_mut().unwrap().pop();
    assert!(serde_json::from_value::<SdfGrid>(json.clone()).is_err());
    json["values"].as_array_mut().unwrap().push(0x7c00.into()); // an infinite half float
    assert!(serde_json::from_value::<SdfGrid>(json).is_err());
}

#[test]
fn fused_maps_keep_what_stays_and_clear_what_leaves() {
    for d in devices() {
        fused_maps_on(d);
    }
}

fn fused_maps_on(device: &Device) {
    let scene = TableScene::new();
    let vga = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let (with, without) = (
        scene.render(640, 480, vga, true),
        scene.render(640, 480, vga, false),
    );
    let mut map =
        OccupancyMap::new(Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3), 0.01).unwrap();
    let (c, r) = scene.ball;
    let top = c + Vec3::new(0.0, 0.0, r);
    let above = top + Vec3::new(0.0, 0.0, 0.05);
    map.integrate(device, &scene.image(&with, 640, vga), None)
        .unwrap();
    assert!(map.occupancy(top).unwrap() > 0.5, "the ball's top is seen");
    assert!(
        map.grid(device, Occlusion::Free).unwrap().distance(above).0 < 0.06,
        "the ball is in the grid"
    );
    // Seen through twice, the ball's top still holds (0.85 - 2 x 0.4); a third time clears it.
    for frame in 1..=3 {
        map.integrate(device, &scene.image(&without, 640, vga), None)
            .unwrap();
        assert_eq!(
            map.occupancy(top).unwrap() > 0.5,
            frame < 3,
            "after {frame} frames without the ball"
        );
    }
    let grid = map.grid(device, Occlusion::Free).unwrap();
    assert!(
        grid.distance(above).0 > 0.2,
        "the ball is gone: {}",
        grid.distance(above).0
    );
    // The table stays; below it nothing was observed, which the grid treats as told.
    assert!(
        map.occupancy(Vec3::new(0.1, 0.1, 0.0)).unwrap() > 0.5,
        "the table"
    );
    let below = Vec3::new(0.1, 0.1, -0.05);
    assert_eq!(map.occupancy(below), None);
    assert!(
        map.grid(device, Occlusion::Occupied)
            .unwrap()
            .distance(below)
            .0
            < 0.0
    );
    assert!(map.grid(device, Occlusion::Free).unwrap().distance(below).0 > 0.0);
    // A map ending inside the ball (its last voxels at x = 0) takes no hits from the readings
    // beyond its edge.
    let mut part = OccupancyMap::new(
        Vec3::new(-0.3, -0.3, -0.1),
        Vec3::new(-0.005, 0.3, 0.3),
        0.01,
    )
    .unwrap();
    part.integrate(device, &scene.image(&with, 640, vga), None)
        .unwrap();
    assert!(
        part.occupancy(Vec3::new(0.0, 0.1, 0.2)).unwrap() < 0.5,
        "the map's edge"
    );
}

/// A camera at `eye` looking at `target`, in OpenCV axes (x right, y down, z forward).
fn look_at(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let down = forward.cross(right);
    Pose::new(
        eye,
        Quat::from_mat3(&glam::Mat3::from_cols(right, down, forward)),
    )
}

#[test]
fn robot_masking_drops_the_arm_and_keeps_what_is_around_it() {
    for d in devices() {
        robot_masking_on(d);
    }
}

fn robot_masking_on(device: &Device) {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    // The robot's spheres in the world, from its link poses.
    let spheres: Vec<(Vec3, f32)> = robot
        .collision_model()
        .spheres
        .iter()
        .flat_map(|(link, s)| {
            let pose = robot.link_pose(&q, link).unwrap();
            s.iter().map(move |s| {
                (
                    pose.rotation * Vec3::new(s[0], s[1], s[2]) + pose.position,
                    s[3],
                )
            })
        })
        .collect();
    // A box between the camera and the arm, and a wall behind the robot.
    let (box_center, box_half, wall) = (Vec3::new(0.9, 0.0, 0.5), Vec3::splat(0.05), -0.5);
    let camera = look_at(Vec3::new(1.6, 0.0, 0.6), Vec3::new(0.3, 0.0, 0.4));
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let depth: Vec<f32> = (0..640 * 480)
        .map(|i| {
            let ray = Vec3::new(
                ((i % 640) as f32 - k.cx) / k.fx,
                ((i / 640) as f32 - k.cy) / k.fy,
                1.0,
            );
            let dir = camera.rotation * ray;
            let mut t = f32::INFINITY;
            for &(c, r) in &spheres {
                let oc = camera.position - c;
                let (b, cc) = (dir.dot(oc), oc.length_squared() - r * r);
                let disc = b * b - dir.length_squared() * cc;
                if disc >= 0.0 {
                    t = t.min((-b - disc.sqrt()) / dir.length_squared());
                }
            }
            // The box by slabs.
            let inv = dir.recip();
            let (t0, t1) = (
                (box_center - box_half - camera.position) * inv,
                (box_center + box_half - camera.position) * inv,
            );
            let (near, far) = (t0.min(t1).max_element(), t0.max(t1).min_element());
            if near <= far && near > 0.0 {
                t = t.min(near);
            }
            if dir.x < 0.0 {
                t = t.min((wall - camera.position.x) / dir.x);
            }
            if t.is_finite() { t } else { 0.0 }
        })
        .collect();
    let image = DepthImage {
        depth: &depth,
        width: 640,
        intrinsics: k,
        camera,
    };
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.15,
    };
    let face = box_center + Vec3::new(box_half.x, 0.0, 0.0);
    // Unmasked, the arm is an obstacle.
    let raw = SdfGrid::from_depth(device, &image, None, Occlusion::Free, &o).unwrap();
    assert!(
        spheres.iter().any(|&(c, r)| raw.distance(c).0 < r),
        "the arm should read as an obstacle"
    );
    // Masked, its space is free while the box in front of it stays.
    let mask = image.robot_mask(&robot, &q, 0.02).unwrap();
    let hidden = (0..depth.len())
        .filter(|&i| depth[i] > 0.0 && depth[i] < mask.depth[i] - 0.1)
        .count();
    assert!(
        hidden > 1000,
        "the box should hide part of the arm: {hidden} pixels"
    );
    let masked = SdfGrid::from_depth(device, &image, Some(&mask), Occlusion::Free, &o).unwrap();
    for &(c, r) in &spheres {
        assert!(
            masked.distance(c).0 > r,
            "an obstacle at the arm: {} at {c}",
            masked.distance(c).0
        );
    }
    assert!(
        masked.distance(face).0 <= 0.0,
        "the box in front of the arm is lost: {}",
        masked.distance(face).0
    );
    // A fused map clears space in front of the arm and adds nothing at it.
    let mut map =
        OccupancyMap::new(Vec3::new(-0.2, -0.6, 0.0), Vec3::new(1.2, 0.6, 1.2), 0.02).unwrap();
    map.integrate(device, &image, Some(&mask)).unwrap();
    for &(c, _) in &spheres {
        assert!(
            map.occupancy(c).is_none_or(|p| p < 0.5),
            "the map holds the arm at {c}"
        );
    }
    assert!(map.occupancy(face).unwrap() > 0.5, "the map lost the box");

    // Where hidden space is solid, the arm still reads free, since nothing else can be where it
    // is, while what it hides stays solid. The mask is padded past the grid's growth.
    let mask = image.robot_mask(&robot, &q, 0.04).unwrap();
    let pixel = |p: Vec3| {
        let c = camera.rotation.inverse() * (p - camera.position);
        let (u, v) = (
            (k.fx * c.x / c.z + k.cx + 0.5) as usize,
            (k.fy * c.y / c.z + k.cy + 0.5) as usize,
        );
        v * 640 + u
    };
    let shadow = spheres
        .iter()
        .map(|&(c, r)| c + (c - camera.position).normalize() * (r + 0.06))
        .filter(|&p| p.x > wall + 0.1 && spheres.iter().all(|&(c, r)| p.distance(c) > r + 0.06))
        .find(|&p| depth[pixel(p)] >= mask.depth[pixel(p)] - 0.01)
        .expect("a point only the arm hides");
    let mut map = OccupancyMap::new(
        Vec3::new(wall + 0.05, -0.6, 0.0),
        Vec3::new(1.2, 0.6, 1.2),
        0.01,
    )
    .unwrap();
    map.integrate(device, &image, Some(&mask)).unwrap();
    let grids = [
        (
            "one image",
            SdfGrid::from_depth(device, &image, Some(&mask), Occlusion::Occupied, &o).unwrap(),
        ),
        ("a map", map.grid(device, Occlusion::Occupied).unwrap()),
    ];
    for (name, grid) in grids {
        for &(c, r) in &spheres {
            assert!(
                grid.distance(c).0 > r,
                "{name}: an obstacle at the arm: {} at {c}",
                grid.distance(c).0
            );
        }
        assert!(
            grid.distance(shadow).0 <= 0.0,
            "{name}: what the arm hides at {shadow} is free"
        );
    }
    // Except in the robot's shadow, which is free, while what the box hides stays solid.
    let behind_box = face - Vec3::new(2.0 * box_half.x, 0.0, 0.0);
    let grids = [
        (
            "one image",
            SdfGrid::from_depth(
                device,
                &image,
                Some(&mask),
                Occlusion::OccupiedExceptRobotShadow,
                &o,
            )
            .unwrap(),
        ),
        (
            "a map",
            map.grid(device, Occlusion::OccupiedExceptRobotShadow)
                .unwrap(),
        ),
    ];
    for (name, grid) in grids {
        assert!(
            grid.distance(shadow).0 > 0.0,
            "{name}: what only the arm hides at {shadow} is solid"
        );
        assert!(
            grid.distance(behind_box).0 <= 0.0,
            "{name}: what the box hides at {behind_box} is free"
        );
        for &(c, r) in &spheres {
            assert!(
                grid.distance(c).0 > r,
                "{name}: an obstacle at the arm: {} at {c}",
                grid.distance(c).0
            );
        }
    }
}

/// The distance transform is exact on every device. (Classifying a depth image's points is
/// checked where it happens, in `sdf.rs`, and the tests above run every builder on every device.)
#[test]
fn every_device_builds_the_same_grids_from_occupancy() {
    let robot = common::panda().unwrap();
    let scene = TableScene::new();
    let vga = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let depth = scene.render(640, 480, vga, true);
    let image = scene.image(&depth, 640, vga);
    let mask = image.robot_mask(&robot, robot.default_q(), 0.02).unwrap();
    let mut map =
        OccupancyMap::new(Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3), 0.01).unwrap();
    map.integrate(cpu(), &image, Some(&mask)).unwrap();
    let mut rng = Rng::new(5);
    let points: Vec<Vec3> = (0..2000)
        .map(|_| {
            Vec3::new(
                rng.range(-0.4, 0.4),
                rng.range(-0.1, 0.1),
                rng.range(0.0, 0.3),
            )
        })
        .collect();
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.15,
    };
    let grids = |d: &Device| {
        [
            ("points", SdfGrid::from_points(d, &points, &o).unwrap()),
            ("map, unknown free", map.grid(d, Occlusion::Free).unwrap()),
            (
                "map, unknown occupied",
                map.grid(d, Occlusion::Occupied).unwrap(),
            ),
            (
                "map, unknown occupied but the robot's shadow",
                map.grid(d, Occlusion::OccupiedExceptRobotShadow).unwrap(),
            ),
        ]
    };
    let reference = grids(cpu());
    for d in &devices()[1..] {
        for ((name, a), (_, b)) in reference.iter().zip(grids(d)) {
            assert!(
                *a == b,
                "{name}: {} and {} build different grids",
                cpu().name(),
                d.name()
            );
        }
    }
}
