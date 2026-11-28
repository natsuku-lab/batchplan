//! Signed distance grids: interpolation and its gradients, the conservative offset, the mesh,
//! point-cloud and depth-image builders, robot masking, fused occupancy maps, and scene meshes
//! loaded as grids.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use batchplan::datagen::{Demonstration, Origin};
use batchplan::rng::Rng;
use batchplan::{
    CollisionWeights, DepthFilter, DepthImage, Device, Distance, Error, Held, Intrinsics,
    JointTrajectory, Obstacle, Occlusion, OccupancyMap, Pose, Robot, RobotMask, SdfGrid,
    SdfOptions, SphereOptions, Weighing, World, sdf,
};
use glam::{BVec3, EulerRot, Mat3, Quat, Vec2, Vec3};

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
        (0..8).map(|i| Vec3::select(BVec3::new(i & 1 != 0, i & 2 != 0, i & 4 != 0), hi, lo)),
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
    Obstacle::cuboid(center, half).distance(p).value
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
    let rotation = Quat::from_euler(EulerRot::XYZ, 0.3, -0.5, 0.9);
    let center = Vec3::new(0.4, -0.2, 0.3);
    let obstacle = Obstacle::Sdf {
        grid: grid.clone(),
        center,
        rotation,
    };
    let (glo, ghi) = (grid.lower_corner(), grid.upper_corner());
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
        let Distance { gradient: g, .. } = obstacle.distance(p);
        let fd = Vec3::from_array(std::array::from_fn(|k| {
            let e = Vec3::AXES[k] * h;
            (obstacle.distance(p + e).value - obstacle.distance(p - e).value) / (2.0 * h)
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
    let worlds = cpu.upload(&[World::new(vec![post])]).unwrap();
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
        let norm = e
            .gradient
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1.0);
        // Spheres crossing cell faces make the cost only piecewise smooth; accept either step.
        let rel_err = |h: f32| {
            let fd = (0..n).map(|j| {
                let (mut qp, mut qm) = (q.clone(), q.clone());
                qp[j] += h;
                qm[j] -= h;
                let cost = |q: &[f32]| cpu.evaluate(&worlds, &[0], q, &w).unwrap().cost[0];
                (cost(&qp) - cost(&qm)) / (2.0 * h)
            });
            fd.zip(&e.gradient)
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

/// A link far from a grid's box skips the grid, by what the values on the box's faces say of
/// those outside it. The grid of a ball larger than its box has only negative values on its
/// faces: links wholly outside the box collide with it, and every device reports what each
/// sphere's own distance does, in clearance and in cost. The second grid is a solid box, zero on
/// its faces, so that the bound is tight, and its rotation is as far from a unit quaternion as a
/// world may have it, about a frame whose origin is metres from the grid: the box that rules the
/// grid out must turn as the lookups do.
#[test]
fn a_grid_reaches_past_its_box_as_far_as_its_faces_say() {
    let robot = common::panda().unwrap();
    let n = robot.dof();
    let (voxel, radius) = (0.02, 0.25);
    let half = 4.5 * voxel;
    let point = |i: usize| Vec3::new((i % 10) as f32, (i / 10 % 10) as f32, (i / 100) as f32);
    let ball: Vec<f32> = (0..1000)
        .map(|i| ((point(i) - 4.5) * voxel).length() - radius)
        .collect();
    let solid: Vec<f32> = (0..1000)
        .map(|i| ((point(i) - 4.5).abs().max_element() - 4.5) * voxel)
        .collect();
    let mut rng = Rng::new(11);
    let items = 6000;
    let q: Vec<f32> = (0..items * n)
        .map(|i| rng.range(robot.lower()[i % n], robot.upper()[i % n]))
        .collect();
    let item_world = vec![0; items];
    let weights = CollisionWeights {
        world: 1.0,
        margin: 0.05,
        ..CollisionWeights::NONE
    };
    let placements = [
        (
            &ball,
            Vec3::ZERO,
            Vec3::new(0.4, 0.25, 0.55),
            Quat::from_rotation_x(0.4),
        ),
        (
            &solid,
            Vec3::new(2.9, -0.1, -0.1),
            Vec3::new(3.4, 0.25, 0.65),
            Quat::from_xyzw(0.0, 0.0, 0.99905, 0.0),
        ),
    ];
    for (distances, origin, center, rotation) in placements {
        let grid = SdfGrid::try_new([10, 10, 10], voxel, origin, distances).unwrap();
        let obstacle = Obstacle::Sdf {
            grid: Arc::new(grid),
            center,
            rotation,
        };
        // Each configuration's clearance and cost sphere by sphere, and whether a sphere's
        // centre is in the box.
        let frame = Mat3::from_quat(rotation).transpose();
        let checked: Vec<(f32, f32, bool)> = q
            .chunks(n)
            .map(|q| {
                let mut found = (f32::INFINITY, 0.0, false);
                for s in robot.spheres(q) {
                    let p = Vec3::new(s[0], s[1], s[2]);
                    let local = frame * (p - center) - (origin + half);
                    let clearance = obstacle.distance(p).value - s[3];
                    let within = (weights.margin - clearance).max(0.0);
                    found.0 = found.0.min(clearance);
                    found.1 += weights.world * within * within;
                    found.2 |= local.abs().max_element() <= half;
                }
                found
            })
            .collect();
        let beyond = checked.iter().filter(|c| c.0 < -0.01 && !c.2).count();
        let clear = checked.iter().filter(|c| c.0 > 0.06).count();
        assert!(
            beyond > 50 && clear > 50,
            "{beyond} configurations collide from outside the box and {clear} are clear"
        );
        let world = World::new(vec![obstacle]);
        for device in devices() {
            let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
            for weights in [CollisionWeights::NONE, weights] {
                let found = device.evaluate(&worlds, &item_world, &q, &weights).unwrap();
                for (i, &(least, cost, _)) in checked.iter().enumerate() {
                    let reported = found.world_clearance[i];
                    // Past the margin a device may report a bound from below; within it, the
                    // distance.
                    assert!(
                        reported <= least + 1e-4
                            && (least > weights.margin || (reported - least).abs() < 1e-4),
                        "{}: configuration {i} is {least} m from the grid, reported {reported}",
                        device.name()
                    );
                    let cost = if weights.world > 0.0 { cost } else { 0.0 };
                    assert!(
                        (found.cost[i] - cost).abs() <= 1e-3 * cost + 1e-6,
                        "{}: configuration {i} costs {cost}, reported {}",
                        device.name(),
                        found.cost[i]
                    );
                }
            }
        }
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
    let origin = built.lower_corner();
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
    let raw = SdfGrid::try_new(dims, voxel, origin, &exact).unwrap();
    let mut rng = Rng::new(5);
    let (mut leaked, mut worst_inside, mut worst_growth) = (0.0f32, f32::MIN, 0.0f32);
    for _ in 0..500 {
        let inside = Vec3::new(
            rng.range(0.004, 0.006),
            rng.range(-0.15, 0.15),
            rng.range(-0.15, 0.15),
        );
        leaked = leaked.max(raw.distance(inside).value);
        worst_inside = worst_inside.max(built.distance(inside).value);
        let near = Vec3::new(
            rng.range(0.03, 0.1),
            rng.range(-0.15, 0.15),
            rng.range(-0.15, 0.15),
        );
        let truth = box_distance(wall_lo, wall_hi, near);
        let Distance { value: d, .. } = built.distance(near);
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
        let (truth, d) = (box_distance(lo, hi, p), grid.distance(p).value);
        assert!(
            d <= truth + 1e-4 && d >= truth - 2.0 * shift(o.voxel) - 1e-4,
            "{d} at {p}, truth {truth}"
        );
    }
    // Beyond the padded box, distances keep growing.
    let far = Vec3::new(0.5, 0.0, 0.04);
    assert!(
        grid.distance(far).value > 0.35,
        "{}",
        grid.distance(far).value
    );
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

/// A mesh of several closed shells is the solid they wind around. Where shells overlap it is
/// their union, as a table's leg that runs up into its top; a shell facing in is still solid
/// beside one facing out, and a cavity inside one; a shell with a corner pushed through its
/// opposite face is solid where it encloses space. The nearest face's normal got each but the
/// cavity wrong: free space in the top above the leg's end, a hollow leg with solid around it,
/// and a free pocket past the crossed face.
#[test]
fn mesh_grids_of_several_shells_are_the_solid_they_wind_around() {
    type Mesh = (Vec<Vec3>, Vec<[u32; 3]>);
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.06,
    };
    let inward = |mesh: &mut Mesh, from: usize| {
        for t in &mut mesh.1[from..] {
            t.swap(1, 2);
        }
    };
    // Each mesh with the boxes it is the union of, and the box that is a cavity in them.
    let top = (Vec3::new(-0.3, -0.3, 0.45), Vec3::new(0.3, 0.3, 0.55));
    let leg = (Vec3::new(-0.05, -0.05, 0.0), Vec3::new(0.05, 0.05, 0.5));
    let short = (leg.0, Vec3::new(0.05, 0.05, 0.4));
    let outer = (Vec3::splat(-0.2), Vec3::splat(0.2));
    let hole = (Vec3::splat(-0.1), Vec3::splat(0.1));
    let mut overlapping = Mesh::default();
    add_box(&mut overlapping, top.0, top.1);
    add_box(&mut overlapping, leg.0, leg.1);
    let mut disagreeing = Mesh::default();
    add_box(&mut disagreeing, top.0, top.1);
    add_box(&mut disagreeing, short.0, short.1);
    inward(&mut disagreeing, 12);
    let mut hollow = Mesh::default();
    add_box(&mut hollow, outer.0, outer.1);
    add_box(&mut hollow, hole.0, hole.1);
    inward(&mut hollow, 12);
    let mut inside_out_hollow = hollow.clone();
    inward(&mut inside_out_hollow, 0);
    type Case<'a> = (&'a str, &'a Mesh, Vec<(Vec3, Vec3)>, Option<(Vec3, Vec3)>);
    let cases: [Case; 4] = [
        ("a leg into its top", &overlapping, vec![top, leg], None),
        (
            "an inward leg under a top",
            &disagreeing,
            vec![top, short],
            None,
        ),
        ("a box with a cavity", &hollow, vec![outer], Some(hole)),
        (
            "an inward box with a cavity",
            &inside_out_hollow,
            vec![outer],
            Some(hole),
        ),
    ];
    let mut rng = Rng::new(31);
    for (name, mesh, solids, cavity) in cases {
        let grid = SdfGrid::from_mesh(&mesh.0, &mesh.1, &o).unwrap();
        let (lo, hi) = (grid.lower_corner(), grid.upper_corner());
        // Points buried in one box nearer another's face than their own: what a face's normal
        // got wrong.
        let mut buried = 0;
        for _ in 0..4000 {
            let p = Vec3::new(
                rng.range(lo.x, hi.x),
                rng.range(lo.y, hi.y),
                rng.range(lo.z, hi.z),
            );
            let each: Vec<f32> = solids.iter().map(|b| box_distance(b.0, b.1, p)).collect();
            let union = each.iter().copied().fold(f32::INFINITY, f32::min);
            let in_cavity = cavity.map_or(f32::INFINITY, |b| box_distance(b.0, b.1, p));
            let d = grid.distance(p).value;
            if in_cavity < 0.0 {
                // Free, as far from the cavity's walls as it is deep in its box.
                let truth = -in_cavity;
                assert!(
                    d <= truth + 1e-4 && d >= truth - 2.0 * shift(o.voxel) - 1e-4,
                    "{name}: {d} in the cavity at {p}, truth {truth}"
                );
            } else if union < 0.0 {
                assert!(d < 0.0, "{name}: {d} inside at {p}");
                let nearest = each.iter().map(|d| d.abs()).fold(f32::INFINITY, f32::min);
                buried +=
                    usize::from(solids.len() > 1 && nearest > shift(o.voxel) && nearest < -union);
            } else {
                let truth = union.min(in_cavity);
                assert!(
                    d <= truth + 1e-4 && d >= truth - 2.0 * shift(o.voxel) - 1e-4,
                    "{name}: {d} outside at {p}, truth {truth}"
                );
            }
        }
        if name == "a leg into its top" {
            assert!(
                buried > 0,
                "no point sampled in the top beside the leg's end"
            );
        }
    }

    // A box with its far corner pushed down through the opposite face. The faces at the corner
    // follow it: they dent the box from above and enclose a pocket below it, facing inward.
    let mut crossed = Mesh::default();
    add_box(
        &mut crossed,
        Vec3::new(-0.1, -0.1, -0.05),
        Vec3::new(0.1, 0.1, 0.05),
    );
    crossed.0[7] = Vec3::new(0.02, 0.02, -0.15);
    let grid = SdfGrid::from_mesh(&crossed.0, &crossed.1, &o).unwrap();
    let value = |p: Vec3| grid.distance(p).value;
    for body in [
        Vec3::new(-0.09, -0.09, -0.02),
        Vec3::new(0.09, -0.09, -0.04),
    ] {
        assert!(value(body) < 0.0, "{} in the body at {body}", value(body));
    }
    let pocket = Vec3::new(0.02, 0.02, -0.1);
    assert!(
        value(pocket) < 0.0,
        "{} in the pocket the pushed corner encloses past the face",
        value(pocket)
    );
    for free in [Vec3::new(0.0, 0.0, 0.03), Vec3::new(0.0, 0.0, 0.2)] {
        assert!(
            value(free) > 0.0,
            "{} in and above the dent at {free}",
            value(free)
        );
    }
}

/// Rays that run exactly through the edges and corners of a mesh count each crossing once. A
/// cube with an off-centre cubic cavity, every coordinate a multiple of the voxel: the grid's
/// rows run through the cube's edges and corners and along the diagonals that split its faces,
/// and the cavity's diagonals lie on other rows than the cube's, so that a crossing counted for
/// both triangles of an edge, or for neither, changes a row's winding.
#[test]
fn mesh_grids_count_a_ray_through_an_edge_once() {
    let step = 1.0 / 64.0;
    let o = SdfOptions {
        voxel: step,
        padding: 4.0 * step,
    };
    let outer = (Vec3::splat(-12.0 * step), Vec3::splat(12.0 * step));
    let hole = (
        Vec3::new(-6.0, -2.0, -6.0) * step,
        Vec3::new(6.0, 10.0, 6.0) * step,
    );
    let mut mesh = (vec![], vec![]);
    add_box(&mut mesh, outer.0, outer.1);
    add_box(&mut mesh, hole.0, hole.1);
    for t in &mut mesh.1[12..] {
        t.swap(1, 2);
    }
    let grid = SdfGrid::from_mesh(&mesh.0, &mesh.1, &o).unwrap();
    assert_eq!(
        (grid.lower_corner(), grid.voxel(), grid.dims()),
        (Vec3::splat(-16.0 * step), step, [33; 3]),
        "the grid's points are the multiples of its voxel the mesh's corners are"
    );
    for n in 0..33u32.pow(3) {
        let at = Vec3::new((n % 33) as f32, (n / 33 % 33) as f32, (n / 1089) as f32);
        let p = grid.lower_corner() + at * step;
        let truth = box_distance(outer.0, outer.1, p).max(-box_distance(hole.0, hole.1, p));
        let d = grid.distance(p).value;
        assert!(
            (d - (truth - shift(step))).abs() < 1e-3,
            "{d} at {p}, {truth} from the surface"
        );
    }

    // A box turned about every axis, so that each row crosses slanted faces: where along the row
    // decides every point's side.
    let turned = Obstacle::Cuboid {
        center: Vec3::new(0.013, -0.021, 0.007),
        half_extents: Vec3::new(0.12, 0.07, 0.05),
        rotation: Quat::from_euler(EulerRot::XYZ, 0.4, -0.7, 1.1),
    };
    let Obstacle::Cuboid {
        center,
        half_extents,
        rotation,
    } = turned
    else {
        unreachable!("a cuboid");
    };
    let mut slanted = (vec![], vec![]);
    add_box(&mut slanted, -half_extents, half_extents);
    for v in &mut slanted.0 {
        *v = center + rotation * *v;
    }
    let centimeter = SdfOptions {
        voxel: 0.01,
        padding: 0.06,
    };
    let grid = SdfGrid::from_mesh(&slanted.0, &slanted.1, &centimeter).unwrap();
    let (lo, dims) = (grid.lower_corner(), grid.dims());
    let mut inside = 0;
    for n in 0..dims.iter().product::<u32>() {
        let at = [n % dims[0], n / dims[0] % dims[1], n / (dims[0] * dims[1])];
        let p = lo + Vec3::from(at.map(|i| i as f32)) * centimeter.voxel;
        let truth = turned.distance(p).value;
        let d = grid.distance(p).value;
        assert!(
            (d - (truth - shift(centimeter.voxel))).abs() < 1e-3,
            "{d} at {p}, {truth} from the turned box"
        );
        inside += usize::from(truth < -centimeter.voxel);
    }
    assert!(inside > 100, "{inside} points deep in the turned box");

    // A hexagonal prism along x whose caps are fans about their centres: two of each fan's edges
    // run along y on the row through the centre, where no hair along y tells their sides apart.
    let half = 6.0 * step;
    let hexagon = [
        [8.0, 0.0],
        [4.0, 6.0],
        [-4.0, 6.0],
        [-8.0, 0.0],
        [-4.0, -6.0],
        [4.0, -6.0],
    ];
    let corner = |k: u32| {
        let [y, z] = hexagon[k as usize];
        Vec3::new(0.0, y, z) * step
    };
    let mut prism = (vec![], vec![]);
    for x in [-half, half] {
        prism.0.push(Vec3::new(x, 0.0, 0.0));
        prism.0.extend((0..6).map(|k| corner(k) + Vec3::X * x));
    }
    for k in 0..6u32 {
        let (a, b) = (1 + k, 1 + (k + 1) % 6);
        prism.1.extend([
            [0, b, a],
            [7, 7 + a, 7 + b],
            [a, b, 7 + b],
            [a, 7 + b, 7 + a],
        ]);
    }
    let grid = SdfGrid::from_mesh(&prism.0, &prism.1, &o).unwrap();
    let (lo, dims) = (grid.lower_corner(), grid.dims());
    let mut on_the_edges = 0;
    for n in 0..dims.iter().product::<u32>() {
        let at = [n % dims[0], n / dims[0] % dims[1], n / (dims[0] * dims[1])];
        let p = lo + Vec3::from(at.map(|i| i as f32)) * step;
        // How deep in the prism: under its caps, its flat sides and its slanted ones.
        let depth = (half - p.x.abs())
            .min(6.0 * step - p.z.abs())
            .min((6.0 * (8.0 * step - p.y.abs()) - 4.0 * p.z.abs()) / 52f32.sqrt());
        if depth > 2.0 * step {
            let d = grid.distance(p).value;
            assert!(d < -step, "{d} at {p}, {depth} deep in the prism");
            on_the_edges += usize::from(p.z == 0.0 && p.y != 0.0);
        }
    }
    assert!(
        on_the_edges > 0,
        "no row of the grid runs along the fans' edges"
    );
}

/// A mesh of thousands of triangles, which are crossed on several threads, reads as its solid:
/// a ball of 9,000 facets, off the grid's lattice, at every grid point.
#[test]
fn a_mesh_of_thousands_of_triangles_reads_as_its_solid() {
    let (center, radius) = (Vec3::new(0.0031, -0.0017, 0.0023), 0.1);
    let (around, down) = (96u32, 48u32);
    let mut ball = (
        vec![center + Vec3::Z * radius, center - Vec3::Z * radius],
        vec![],
    );
    for ring in 1..down {
        let polar = std::f32::consts::PI * ring as f32 / down as f32;
        for k in 0..around {
            let turn = std::f32::consts::TAU * k as f32 / around as f32;
            let unit = Vec3::new(
                polar.sin() * turn.cos(),
                polar.sin() * turn.sin(),
                polar.cos(),
            );
            ball.0.push(center + radius * unit);
        }
    }
    let at = |ring: u32, k: u32| 2 + (ring - 1) * around + k % around;
    for k in 0..around {
        ball.1.push([0, at(1, k), at(1, k + 1)]);
        ball.1.push([1, at(down - 1, k + 1), at(down - 1, k)]);
        for ring in 1..down - 1 {
            ball.1
                .push([at(ring, k), at(ring + 1, k), at(ring + 1, k + 1)]);
            ball.1
                .push([at(ring, k), at(ring + 1, k + 1), at(ring, k + 1)]);
        }
    }
    assert!(ball.1.len() > 8192, "{} triangles", ball.1.len());
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.05,
    };
    let grid = SdfGrid::from_mesh(&ball.0, &ball.1, &o).unwrap();
    let (lo, dims) = (grid.lower_corner(), grid.dims());
    for n in 0..dims.iter().product::<u32>() {
        let at = [n % dims[0], n / dims[0] % dims[1], n / (dims[0] * dims[1])];
        let p = lo + Vec3::from(at.map(|i| i as f32)) * o.voxel;
        // The facets lie within a sixth of a millimetre inside the ball.
        let truth = (p - center).length() - radius;
        let d = grid.distance(p).value;
        assert!(
            (d - (truth - shift(o.voxel))).abs() < 1e-3,
            "{d} at {p}, {truth} from the ball"
        );
    }
    // Facing in, it is the same grid to the bit: the nearest point of a face depends on the order
    // of its corners in its last bit, so the mesh is measured facing out either way.
    let fine = SdfOptions {
        voxel: 0.005,
        padding: 0.05,
    };
    let inward: Vec<[u32; 3]> = ball.1.iter().map(|t| [t[0], t[2], t[1]]).collect();
    assert_eq!(
        SdfGrid::from_mesh(&ball.0, &ball.1, &fine).unwrap(),
        SdfGrid::from_mesh(&ball.0, &inward, &fine).unwrap()
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
            grid.distance(p).value <= 0.0,
            "a surface point reads {}",
            grid.distance(p).value
        );
    }
    for _ in 0..500 {
        let dir = Vec3::new(rng.normal(), rng.normal(), rng.normal()).normalize();
        let truth = rng.range(0.02, 0.08);
        let d = grid.distance(center + dir * (radius + truth)).value;
        assert!(
            d <= truth + 2e-3 && d >= truth - occupancy_slack(o.voxel),
            "{d} for {truth}"
        );
    }
    // Inside the shell is unobserved and reads as free.
    assert!(grid.distance(center).value > 0.05);
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
        let (origin, hi) = (grid.lower_corner(), grid.upper_corner());
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
            worst = worst.max(grid.distance(p).value - truth);
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

/// Per pixel of `image`, the depth at which its ray meets `robot`'s collision spheres at `q`, each
/// grown by `padding`; infinite where it meets none.
fn arm_depth(image: &DepthImage, robot: &Robot, q: &[f32], padding: f32) -> Vec<f32> {
    let k = image.intrinsics;
    let origin = image.camera.position;
    let spheres = robot.spheres(q);
    (0..image.depth.len())
        .map(|i| {
            let (u, v) = ((i % image.width) as f32, (i / image.width) as f32);
            let dir = image.camera.rotation * Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
            let hits = spheres.iter().filter_map(|&[x, y, z, r]| {
                let (from, r) = (origin - Vec3::new(x, y, z), r + padding);
                let (b, c) = (dir.dot(from), from.length_squared() - r * r);
                let disc = b * b - dir.length_squared() * c;
                Some((-b - disc.sqrt()) / dir.length_squared()).filter(|&t| disc >= 0.0 && t > 0.0)
            });
            hits.fold(f32::INFINITY, f32::min)
        })
        .collect()
}

/// `image`'s depths with the arm at `q` in front of what it hides: what a camera that sees the
/// arm reads.
fn with_arm(image: &DepthImage, robot: &Robot, q: &[f32]) -> Vec<f32> {
    let arm = arm_depth(image, robot, q, 0.0);
    image
        .depth
        .iter()
        .zip(arm)
        .map(|(&d, a)| d.min(a))
        .collect()
}

/// A camera's rotation a thousandth off a unit quaternion turns the camera as the unit one does:
/// the grid of what it sees from 3 m, where a depth 0.2% off is 5 mm, reads the same, and so do a
/// map the image is fused into and which of the image's readings are the robot's.
#[test]
fn a_camera_turned_by_a_rotation_not_quite_unit_sees_the_same() {
    let o = SdfOptions {
        voxel: 0.02,
        padding: 0.15,
    };
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let turn = Quat::from_rotation_x(PI + 0.2) * Quat::from_rotation_z(0.3);
    let robot = common::panda().unwrap();
    for device in devices() {
        for norm in [0.9991f32, 1.0009] {
            let seen = |rotation: Quat| {
                let scene = TableScene {
                    camera: Pose::new(Vec3::new(0.0, -0.6, 3.0), rotation),
                    ..TableScene::new()
                };
                // What the camera sees does not depend on its rotation's length.
                let depth = TableScene {
                    camera: Pose::new(scene.camera.position, turn),
                    ..TableScene::new()
                }
                .render(640, 480, k, true);
                let image = scene.image(&depth, 640, k);
                let grid =
                    SdfGrid::from_depth(device, &image, None, Occlusion::Occupied, &o).unwrap();
                let mut map = OccupancyMap::try_new(
                    Vec3::new(-0.3, -0.3, -0.1),
                    Vec3::new(0.3, 0.3, 0.3),
                    0.02,
                )
                .unwrap();
                map.integrate(device, &image, None).unwrap();
                // The arm as the camera turned by the unit rotation sees it.
                let unit = DepthImage {
                    camera: Pose::new(scene.camera.position, turn),
                    ..image
                };
                let arm = with_arm(&unit, &robot, robot.default_q());
                let image = DepthImage {
                    depth: &arm,
                    ..image
                };
                let mask = image
                    .robot_mask(&robot, robot.default_q(), 0.02, &[])
                    .unwrap();
                (grid, map, mask.on_robot)
            };
            let (unit, unit_map, unit_mask) = seen(turn);
            let (scaled, scaled_map, scaled_mask) = seen(turn * norm);
            // Which readings are the robot's: the same but along the grown spheres' outline,
            // where a reading lies in a sphere's last place.
            let (mut on_robot, mut moved) = (0, 0);
            for (u, s) in unit_mask.iter().zip(&scaled_mask) {
                on_robot += usize::from(*u || *s);
                moved += usize::from(u != s);
            }
            assert!(
                on_robot > 500 && moved <= on_robot / 100,
                "norm {norm}: {moved} of the robot's {on_robot} readings changed sides"
            );
            // A grid point's pixel is the one its projection rounds to, so a camera turned in
            // its last place moves some to their neighbours (a GPU's rounding more than the
            // CPU's): a few points may read otherwise. A depth 0.2% off moves most.
            let mut rng = Rng::new(8);
            let (mut near, mut read, mut filled) = (0, 0, 0);
            for _ in 0..2000 {
                let p = Vec3::new(
                    rng.range(-0.3, 0.3),
                    rng.range(-0.3, 0.3),
                    rng.range(-0.05, 0.3),
                );
                let (u, s) = (unit.distance(p).value, scaled.distance(p).value);
                near += usize::from(u.abs() < 0.05);
                read += usize::from((u - s).abs() >= 2e-4);
                filled += usize::from(unit_map.occupancy(p) != scaled_map.occupancy(p));
            }
            eprintln!(
                "{}, norm {norm}: {read} of 2000 points read otherwise, {filled} filled otherwise",
                device.name()
            );
            assert!(near > 200, "{near} points near a surface");
            assert!(
                read <= 200 && filled <= 20,
                "norm {norm}: {read} of 2000 points read otherwise, {filled} filled otherwise"
            );
        }
    }
}

/// Checks a grid of the bare table: distances above it, the occlusion rule below it, free space
/// outside the view.
fn check_table(grid: &SdfGrid, behind: Occlusion, o: &SdfOptions, rng: &mut Rng) {
    for _ in 0..300 {
        let height = rng.range(0.03, 0.25);
        let p = Vec3::new(rng.range(-0.25, 0.25), rng.range(-0.25, 0.25), height);
        let d = grid.distance(p).value;
        assert!(
            d <= height + 1e-4 && d >= height - occupancy_slack(o.voxel),
            "{d} at {p}"
        );
        let below = grid.distance(Vec3::new(p.x, p.y, -0.05)).value;
        match behind {
            // The table, not a robot, hides what is below it.
            Occlusion::Occupied | Occlusion::OccupiedExceptRobotShadow => {
                assert!(below < 0.0, "{below} below the table at {p}")
            }
            Occlusion::Free => assert!(below > 0.02, "{below} below the table at {p}"),
            // Solid just behind the top, as seen from the camera above, and free past the band.
            Occlusion::Band(band) => {
                let near = grid.distance(Vec3::new(p.x, p.y, -band)).value;
                let past = grid.distance(Vec3::new(p.x, p.y, -band - 0.06)).value;
                assert!(near < 0.0, "{near} within the band below the table at {p}");
                assert!(past > 0.02, "{past} past the band below the table at {p}");
            }
            other => panic!("no expectation below the table under {other:?}"),
        }
    }
    // Outside the view nothing was seen, so it is free even under the occupied rule.
    let hi = grid.upper_corner();
    let aside = Vec3::new(hi.x - 0.005, 0.0, -0.1);
    assert!(
        grid.distance(aside).value > 0.02,
        "{} outside the view",
        grid.distance(aside).value
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
            Occlusion::Band(0.04),
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
        let above = grid.distance(c + Vec3::new(0.0, 0.0, r + 0.05)).value;
        assert!(
            above <= 0.05 + 2e-3 && above > 0.05 - occupancy_slack(o.voxel),
            "{above} above the ball"
        );
        let center = grid.distance(c).value;
        assert!(
            center.signum() == f32::signum(inside),
            "{center} at the ball's center under {behind:?}"
        );
        // As an obstacle, the grid is where the camera saw it.
        let obstacle = Obstacle::from(grid);
        assert_eq!(
            [
                obstacle.distance(c).value,
                obstacle.distance(c + Vec3::new(0.0, 0.0, r + 0.05)).value
            ],
            [center, above],
            "an obstacle made from the grid measures it as the grid does"
        );
    }
    // Pixels without a reading observe nothing. Below the table, zone (4, 4) is centered at
    // x = 0.054 and zone (5, 4) at x = 0.163 (y = -0.054 for both; the camera's y axis points
    // along the world's -y).
    let mut holes = scene.render(8, 8, tof, false);
    for z in holes.iter_mut().step_by(2) {
        *z = 0.0;
    }
    let grid = SdfGrid::from_depth(
        device,
        &scene.image(&holes, 8, tof),
        None,
        Occlusion::Occupied,
        &o,
    )
    .unwrap();
    assert!(
        grid.distance(Vec3::new(0.054, -0.054, -0.05)).value > 0.0,
        "below a pixel without a reading"
    );
    assert!(
        grid.distance(Vec3::new(0.163, -0.054, -0.05)).value < 0.0,
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
        .is_err(),
        "an image without a single reading is refused"
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
    for bad in [-0.01, f32::NAN] {
        let image = scene.image(&depth, 640, vga);
        assert!(
            matches!(
                SdfGrid::from_depth(device, &image, None, Occlusion::Band(bad), &o),
                Err(Error::Input(_))
            ),
            "a band of {bad} m"
        );
    }
    // A camera's pose is checked as other poses are.
    let robot = common::panda().unwrap();
    let poses = [
        Pose::new(scene.camera.position, scene.camera.rotation * 2.0),
        Pose::new(scene.camera.position, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
        Pose::new(Vec3::NAN, scene.camera.rotation),
    ];
    for camera in poses {
        let image = DepthImage {
            camera,
            ..scene.image(&depth, 640, vga)
        };
        let refused = |result: Result<(), Error>| {
            assert!(
                matches!(&result, Err(Error::Input(m)) if m.contains("camera")),
                "{camera:?}: {result:?}"
            );
        };
        let grid = SdfGrid::from_depth(device, &image, None, Occlusion::Free, &o);
        refused(grid.map(|_| ()));
        let mut map = OccupancyMap::try_new(Vec3::splat(-0.3), Vec3::splat(0.3), 0.02).unwrap();
        refused(map.integrate(device, &image, None));
        refused(
            image
                .robot_mask(&robot, robot.default_q(), 0.02, &[])
                .map(|_| ()),
        );
    }
}

/// A robot's mask refuses a configuration with a joint that is no number, naming `q`: masked so,
/// a fraction of the arm's readings would be its own and the rest the world's.
#[test]
fn a_robot_mask_refuses_a_joint_that_is_no_number() {
    let robot = common::panda().unwrap();
    let depth = vec![1.0; 64 * 48];
    let image = DepthImage {
        depth: &depth,
        width: 64,
        intrinsics: Intrinsics {
            fx: 60.0,
            fy: 60.0,
            cx: 31.5,
            cy: 23.5,
        },
        camera: Pose::new(Vec3::new(0.5, 0.0, 1.0), Quat::from_rotation_x(PI)),
    };
    assert!(
        image
            .robot_mask(&robot, robot.default_q(), 0.01, &[])
            .is_ok()
    );
    for (joint, value) in [(1, f32::NAN), (3, f32::INFINITY)] {
        let mut q = robot.default_q().to_vec();
        q[joint] = value;
        let mask = image.robot_mask(&robot, &q, 0.01, &[]);
        assert!(
            matches!(&mask, Err(Error::Input(m)) if m.contains('q') && m.contains(&value.to_string())),
            "joint {joint} at {value}: {mask:?}"
        );
    }
}

/// An L-shaped prism 10 cm tall whose convex hull also fills the notch at x, y in (0.1, 0.2).
fn l_prism_obj() -> String {
    use std::fmt::Write;

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
            writeln!(obj, "v {x} {y} {z}").unwrap();
        }
    }
    // Bottom faces down, top faces up (OBJ indices start at 1).
    for f in [[1, 4, 2], [2, 4, 3], [1, 6, 5], [1, 5, 4]] {
        writeln!(obj, "f {} {} {}", f[0], f[1], f[2]).unwrap();
    }
    for f in [[7, 8, 10], [8, 9, 10], [7, 11, 12], [7, 10, 11]] {
        writeln!(obj, "f {} {} {}", f[0], f[1], f[2]).unwrap();
    }
    for i in 1..=6 {
        let j = i % 6 + 1;
        writeln!(obj, "f {i} {j} {}\nf {i} {} {}", j + 6, j + 6, i + 6).unwrap();
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
    let at = |x: f32, y: f32, z: f32| world.obstacles[0].distance(Vec3::new(0.5 - y, x, z)).value;
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
    let notch = |o: &Obstacle, x: f32| o.distance(Vec3::new(x + 0.15, 0.15, 0.05)).value;
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
fn still_demo(robot: &Robot, world: u32) -> Demonstration {
    let q = robot.default_q();
    let trajectory = JointTrajectory {
        dof: q.len(),
        dt: 0.1,
        duration: 0.1,
        positions: [q, q].concat(),
        velocities: vec![0.0; 2 * q.len()],
        accelerations: vec![0.0; 2 * q.len()],
    };
    Demonstration {
        origin: Origin::Nominal,
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
        World::new(vec![shared(0.3)]),
        World::new(vec![shared(0.5), shared(0.6)]),
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
    use std::fs::File;

    use arrow_array::{Array, FixedSizeListArray, Float32Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let robot = common::panda().unwrap();
    let mut mesh = (vec![], vec![]);
    add_box(&mut mesh, Vec3::ZERO, Vec3::new(0.2, 0.1, 0.1));
    let grid = Arc::new(SdfGrid::from_mesh(&mesh.0, &mesh.1, &SdfOptions::default()).unwrap());
    let (center, rotation) = (Vec3::new(0.5, 0.2, 0.0), Quat::from_rotation_z(0.5));
    let worlds = vec![World::new(vec![Obstacle::Sdf {
        grid: grid.clone(),
        center,
        rotation,
    }])];
    let root = scratch("lerobot");
    batchplan::lerobot::export(
        &root,
        &robot,
        &worlds,
        &[still_demo(&robot, 0)],
        &Default::default(),
    )
    .unwrap();
    let file = File::open(root.join("data/chunk-000/file-000.parquet")).unwrap();
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
    let (lo, hi) = (grid.lower_corner(), grid.upper_corner());
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
        OccupancyMap::try_new(Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3), 0.01).unwrap();
    let (c, r) = scene.ball;
    let top = c + Vec3::new(0.0, 0.0, r);
    let above = top + Vec3::new(0.0, 0.0, 0.05);
    map.integrate(device, &scene.image(&with, 640, vga), None)
        .unwrap();
    assert!(map.occupancy(top).unwrap() > 0.5, "the ball's top is seen");
    assert!(
        map.grid(device, Occlusion::Free)
            .unwrap()
            .distance(above)
            .value
            < 0.06,
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
        grid.distance(above).value > 0.2,
        "the ball is gone: {}",
        grid.distance(above).value
    );
    // The table stays; below it nothing was observed, which the grid treats as told.
    assert!(
        map.occupancy(Vec3::new(0.1, 0.1, 0.0)).unwrap() > 0.5,
        "the table"
    );
    let below = Vec3::new(0.1, 0.1, -0.05);
    assert_eq!(
        map.occupancy(below),
        None,
        "nothing was observed below the table"
    );
    assert!(
        map.grid(device, Occlusion::Occupied)
            .unwrap()
            .distance(below)
            .value
            < 0.0,
        "unobserved space is solid under Occlusion::Occupied"
    );
    assert!(
        map.grid(device, Occlusion::Free)
            .unwrap()
            .distance(below)
            .value
            > 0.0,
        "unobserved space is free under Occlusion::Free"
    );
    // Solid to the band's distance under the table, and free three voxels past it.
    let banded = map.grid(device, Occlusion::Band(0.04)).unwrap();
    let (at, past) = (Vec3::new(0.1, 0.1, -0.04), Vec3::new(0.1, 0.1, -0.07));
    assert!(
        banded.distance(at).value < 0.0,
        "{} at the band under the table",
        banded.distance(at).value
    );
    assert!(
        banded.distance(past).value > 0.0,
        "{} past the band under the table",
        banded.distance(past).value
    );
    assert!(
        map.grid(device, Occlusion::Band(0.0)).unwrap()
            == map.grid(device, Occlusion::Free).unwrap(),
        "a band of nothing walls in more than the free rule"
    );
    for bad in [-0.01, f32::NAN] {
        assert!(matches!(
            map.grid(device, Occlusion::Band(bad)),
            Err(batchplan::Error::Input(_))
        ));
    }
    // Where the arm hides the table, the band leaves what only the robot hides free, as images
    // do, and with it the space just above the table, observed or not.
    let robot = common::panda().unwrap();
    let table = scene.image(&without, 640, vga);
    let arm = with_arm(&table, &robot, robot.default_q());
    let image = scene.image(&arm, 640, vga);
    let mask = image
        .robot_mask(&robot, robot.default_q(), 0.02, &[])
        .unwrap();
    let mut masked =
        OccupancyMap::try_new(Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3), 0.01).unwrap();
    masked.integrate(device, &image, Some(&mask)).unwrap();
    let banded = masked.grid(device, Occlusion::Band(0.04)).unwrap();
    let above: Vec<Vec3> = (-25..=25)
        .flat_map(|i| (-25..=25).map(move |j| Vec3::new(i as f32, j as f32, 2.0) * 0.01))
        .collect();
    let hidden = above
        .iter()
        .filter(|&&p| masked.occupancy(p).is_none())
        .count();
    assert!(
        hidden > 100,
        "the arm should hide space above the table: {hidden} points"
    );
    for &p in &above {
        let d = banded.distance(p).value;
        assert!(d > 0.0, "{d} at {p}, 2 cm above the table");
    }
    // Below the table where the arm hid it, the table hides it too, whichever image comes first:
    // solid as the rules say of what other surfaces hide.
    let under: Vec<Vec3> = above
        .iter()
        .filter(|&&p| masked.occupancy(p).is_none())
        .map(|&p| p - Vec3::new(0.0, 0.0, 0.04))
        .collect();
    let images = [(image, Some(&mask)), (table, None)];
    for order in [[0, 1], [1, 0]] {
        let mut map =
            OccupancyMap::try_new(Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3), 0.01)
                .unwrap();
        for i in order {
            map.integrate(device, &images[i].0, images[i].1).unwrap();
        }
        for unknown in [Occlusion::OccupiedExceptRobotShadow, Occlusion::Band(0.04)] {
            let grid = map.grid(device, unknown).unwrap();
            for &p in &under {
                let d = grid.distance(p).value;
                assert!(
                    d < 0.0,
                    "{d} at {p} under {unknown:?}, images in order {order:?}"
                );
            }
        }
    }
    // A map ending inside the ball (its last voxels at x = 0) takes no hits from the readings
    // beyond its edge.
    let mut part = OccupancyMap::try_new(
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

/// The points of a map made with `try_new(lo, hi, voxel)`.
fn lattice(lo: Vec3, hi: Vec3, voxel: f32) -> Vec<Vec3> {
    let [nx, ny, nz] = ((hi - lo) / voxel).ceil().to_array().map(|n| n as u32 + 1);
    let all = (0..nz).flat_map(|z| (0..ny).flat_map(move |y| (0..nx).map(move |x| (x, y, z))));
    all.map(|(x, y, z)| lo + Vec3::new(x as f32, y as f32, z as f32) * voxel)
        .collect()
}

/// A caller marks space it knows is empty, and space it no longer knows: the voxels within the
/// margin of a shape of any kind, turned any way, and no other. Marked free, a voxel is free
/// until a reading lands there; cleared, it was never observed.
#[test]
fn a_caller_marks_space_free_or_no_longer_known() {
    let (lo, hi, voxel) = (Vec3::new(-0.2, -0.15, 0.0), Vec3::new(0.2, 0.15, 0.3), 0.01);
    let points = lattice(lo, hi, voxel);
    let turn = Quat::from_rotation_y(0.7) * Quat::from_rotation_z(0.4);
    let cloud: Vec<Vec3> = (0..400)
        .map(|i| Vec3::new((i % 20) as f32, (i / 20) as f32, 0.0) * 0.004)
        .collect();
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.03,
    };
    let plate = SdfGrid::from_points(cpu(), &cloud, &o).unwrap();
    let shapes = [
        Obstacle::Cuboid {
            center: Vec3::new(-0.05, 0.0, 0.15),
            half_extents: Vec3::new(0.02, 0.06, 0.03),
            rotation: turn,
        },
        Obstacle::sphere(Vec3::new(0.1, 0.05, 0.1), 0.04),
        Obstacle::Cylinder {
            center: Vec3::new(0.0, 0.0, 0.2),
            rotation: turn,
            radius: 0.05,
            half_height: 0.02,
        },
        // Half of it past the map's edge.
        Obstacle::Capsule {
            center: Vec3::new(0.2, 0.0, 0.15),
            rotation: turn,
            radius: 0.02,
            half_length: 0.07,
        },
        Obstacle::Sdf {
            grid: Arc::new(plate),
            center: Vec3::new(-0.1, -0.05, 0.1),
            rotation: turn,
        },
        // A grid solid on its faces, which reads solid past its box as deep as they are.
        Obstacle::Sdf {
            grid: Arc::new(
                SdfGrid::try_new([6; 3], 0.02, Vec3::splat(-0.05), &[-0.03; 216]).unwrap(),
            ),
            center: Vec3::new(0.05, 0.0, 0.15),
            rotation: turn,
        },
    ];
    let margin = 0.015;
    for shape in &shapes {
        let mut map = OccupancyMap::try_new(lo, hi, voxel).unwrap();
        assert_eq!(map.counts().unobserved, points.len());
        map.free(shape, margin).unwrap();
        let mut within = 0;
        for &p in &points {
            let marked = shape.distance(p).value <= margin;
            let seen = map.occupancy(p);
            assert_eq!(seen.is_some(), marked, "{shape:?} at {p}");
            assert!(seen.is_none_or(|o| o < 0.5), "{shape:?} at {p}");
            within += usize::from(marked);
        }
        assert!(
            within > 100 && 2 * within < points.len(),
            "{within} points within the margin of {shape:?}"
        );
        let counts = map.counts();
        assert_eq!(
            (counts.occupied, counts.free, counts.unobserved),
            (0, within, points.len() - within)
        );
        map.clear(shape, margin).unwrap();
        assert_eq!(map.counts().unobserved, points.len(), "{shape:?}");
    }
    // A box whose margin ends on the map's points, in lengths a float holds exactly: the points
    // a voxel out from its faces are within the margin, and are the first and last looked at.
    let (corner, eighth) = (Vec3::splat(2.0), 0.125);
    let mut map = OccupancyMap::try_new(Vec3::ZERO, corner, eighth).unwrap();
    let cube = Obstacle::cuboid(Vec3::ONE, Vec3::splat(0.5));
    map.free(&cube, eighth).unwrap();
    let exact = lattice(Vec3::ZERO, corner, eighth);
    assert_eq!(cube.distance(Vec3::new(0.375, 1.0, 1.0)).value, eighth);
    for &p in &exact {
        let marked = cube.distance(p).value <= eighth;
        assert_eq!(map.occupancy(p).is_some(), marked, "at {p}");
    }
    assert_eq!(
        map.counts().free,
        9 * 9 * 9 + 6 * 9 * 9,
        "the cube and a voxel off each face"
    );
    // A margin that is no length and a shape that is no shape are refused.
    let mut map = OccupancyMap::try_new(lo, hi, voxel).unwrap();
    let broken = Obstacle::sphere(Vec3::ZERO, f32::NAN);
    for (shape, margin) in [(&shapes[0], -0.01), (&shapes[0], f32::NAN), (&broken, 0.0)] {
        assert!(matches!(map.free(shape, margin), Err(Error::Input(_))));
        assert!(matches!(map.clear(shape, margin), Err(Error::Input(_))));
        assert!(matches!(map.part(shape, margin), Err(Error::Input(_))));
    }
    // So is a region whose part would pass a grid's limits, however little of it the map covers:
    // more than 2^26 points, or a diagonal of 4,096 voxels.
    let long = Obstacle::Capsule {
        center: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        radius: 0.01,
        half_length: 21.0,
    };
    for region in [Obstacle::cuboid(Vec3::ZERO, Vec3::splat(2.1)), long] {
        assert!(
            matches!(map.part(&region, 0.0), Err(Error::Input(_))),
            "{region:?}"
        );
    }

    // In a fused scene: a ball on a table, carried away with no camera looking.
    let scene = TableScene::new();
    let vga = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let depth = scene.render(640, 480, vga, true);
    let image = scene.image(&depth, 640, vga);
    let (c, r) = scene.ball;
    let ball = Obstacle::sphere(c, r);
    let (top, table) = (c + Vec3::Z * r, Vec3::new(-0.2, -0.2, 0.0));
    for device in devices() {
        let mut map =
            OccupancyMap::try_new(Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3), 0.01)
                .unwrap();
        map.integrate(device, &image, None).unwrap();
        let seen = map.counts();
        assert!(seen.occupied > 1_000 && seen.free > 10_000 && seen.unobserved > 10_000);
        let before = (map.occupancy(top).unwrap(), map.occupancy(table).unwrap());
        assert!(
            before.0 > 0.5 && before.1 > 0.5,
            "the ball and the table are seen"
        );
        map.free(&ball, 0.02).unwrap();
        assert!(
            map.occupancy(top).unwrap() < 0.5,
            "the ball's place is free"
        );
        assert!(
            map.occupancy(c).unwrap() < 0.5,
            "and its middle, never seen"
        );
        assert_eq!(map.occupancy(table).unwrap(), before.1, "the table stays");
        assert!(map.counts().occupied < seen.occupied);
        let gone = map.grid(device, Occlusion::Occupied).unwrap();
        assert!(
            gone.distance(c).value > 0.03,
            "{}: {} where the ball was",
            device.name(),
            gone.distance(c).value
        );
        // One more look at the ball where it stands, and it is there again.
        map.integrate(device, &image, None).unwrap();
        assert!(map.occupancy(top).unwrap() > 0.5, "the ball is seen again");
        // Space seen through three times is more surely free than marking it makes it: it
        // stays as it was, and one reading there does not fill it.
        let mut empty =
            OccupancyMap::try_new(Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3), 0.01)
                .unwrap();
        let bare = scene.render(640, 480, vga, false);
        for _ in 0..3 {
            empty
                .integrate(device, &scene.image(&bare, 640, vga), None)
                .unwrap();
        }
        let seen_through = empty.occupancy(top).unwrap();
        empty.free(&ball, 0.02).unwrap();
        assert_eq!(empty.occupancy(top).unwrap(), seen_through);
        empty.integrate(device, &image, None).unwrap();
        assert!(
            empty.occupancy(top).unwrap() < 0.5,
            "one reading against three"
        );
        // What was seen there no longer holds: never observed, and taken as the grid is told.
        map.clear(&ball, 0.02).unwrap();
        assert_eq!(map.occupancy(top), None);
        assert!(map.occupancy(table).unwrap() > 0.5, "the table stays");
        let (solid, free) = (
            map.grid(device, Occlusion::Occupied).unwrap(),
            map.grid(device, Occlusion::Free).unwrap(),
        );
        assert!(solid.distance(c).value <= 0.0 && free.distance(c).value > 0.03);
    }
}

/// A part of a map is one object as the cameras saw into it: a tray with a block in it, seen
/// from above. With never-observed space solid, its walls and the block's hidden inside are
/// solid, the space seen into is free, and nothing outside the tray's box is in the grid, which
/// reaches the padding past it and never reads farther from the tray than the tray is.
#[test]
fn a_part_of_a_map_is_an_object_as_seen_into() {
    for device in devices() {
        a_part_of_a_map_on(device);
    }
}

fn a_part_of_a_map_on(device: &Device) {
    // A tray 20 by 16 cm and 10 cm tall, its floor and walls 2 cm thick, turned on a table, and
    // a block 4 cm on a side on its floor.
    let pose = Pose::new(Vec3::new(0.02, -0.01, 0.05), Quat::from_rotation_z(0.4));
    let piece = |at: Vec3, half: Vec3| {
        Obstacle::cuboid(Vec3::ZERO, half).placed(pose.mul_pose(Pose::new(at, Quat::IDENTITY)))
    };
    let (half, wall) = (Vec3::new(0.10, 0.08, 0.05), 0.01);
    let tray = [
        piece(
            Vec3::new(0.0, 0.0, wall - half.z),
            Vec3::new(half.x, half.y, wall),
        ),
        piece(
            Vec3::new(half.x - wall, 0.0, 0.0),
            Vec3::new(wall, half.y, half.z),
        ),
        piece(
            Vec3::new(wall - half.x, 0.0, 0.0),
            Vec3::new(wall, half.y, half.z),
        ),
        piece(
            Vec3::new(0.0, half.y - wall, 0.0),
            Vec3::new(half.x, wall, half.z),
        ),
        piece(
            Vec3::new(0.0, wall - half.y, 0.0),
            Vec3::new(half.x, wall, half.z),
        ),
    ];
    let block = piece(Vec3::new(0.03, 0.0, -0.01), Vec3::splat(0.02));
    let table = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.05), Vec3::new(0.5, 0.5, 0.05));
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let far = vec![0.0; 640 * 480];
    let image = DepthImage {
        depth: &far,
        width: 640,
        intrinsics: k,
        camera: Pose::new(Vec3::new(0.0, 0.0, 0.7), Quat::from_rotation_x(PI)),
    };
    let scene: Vec<&Obstacle> = tray.iter().chain([&block, &table]).collect();
    let depths: Vec<Vec<f32>> = scene.iter().map(|o| shape_depth(&image, o)).collect();
    let depth: Vec<f32> = (0..far.len())
        .map(|i| depths.iter().map(|d| d[i]).fold(f32::INFINITY, f32::min))
        .map(|z| if z.is_finite() { z } else { 0.0 })
        .collect();
    let image = DepthImage {
        depth: &depth,
        ..image
    };
    let (lo, hi) = (Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3));
    let mut map = OccupancyMap::try_new(lo, hi, 0.01).unwrap();
    map.integrate(device, &image, None).unwrap();
    let region = Obstacle::cuboid(Vec3::ZERO, half).placed(pose);
    let padding = 0.1;
    let part = map.part(&region, padding).unwrap();

    // Where a voxel reaches into the tray's box the part is the map, and elsewhere nothing.
    let points = lattice(lo, hi, 0.01);
    let reach = 0.5 * 3f32.sqrt() * 0.01;
    let mut inside = 0;
    for &p in &points {
        let within = region.distance(p).value <= reach;
        inside += usize::from(within);
        let expected = map.occupancy(p).filter(|_| within);
        assert_eq!(part.occupancy(p), expected, "at {p}");
    }
    let counts = part.counts();
    assert_eq!(counts.occupied + counts.free + counts.unobserved, inside);
    assert!(
        counts.occupied > 200 && counts.free > 500 && counts.unobserved > 200,
        "{counts:?}: walls seen, space seen into, and space hidden"
    );
    assert!(matches!(
        part.clone().integrate(device, &image, None),
        Err(Error::Input(_))
    ));
    // What a caller marks in a part stays in its region: the voxels left out stay out.
    let everything = Obstacle::cuboid(Vec3::ZERO, Vec3::splat(1.0));
    let (mut freed, mut cleared) = (part.clone(), part.clone());
    freed.free(&everything, 0.0).unwrap();
    cleared.clear(&everything, 0.0).unwrap();
    assert_eq!(freed.counts().free, inside);
    assert_eq!(cleared.counts().unobserved, inside);
    let beside = pose.position + pose.rotation * Vec3::new(half.x + 0.05, 0.0, 0.0);
    assert_eq!(
        (freed.occupancy(beside), cleared.occupancy(beside)),
        (None, None)
    );
    // A part of a part is the part of the map in both regions: what the first left out stays
    // out, also where the second region reaches past the first part's own points.
    let ball = Obstacle::sphere(pose.position, 0.06);
    let wider = Obstacle::sphere(pose.position, 0.3);
    let small = map.part(&ball, 0.0).unwrap();
    for padding in [0.0, 0.05] {
        let nested = small.part(&wider, padding).unwrap();
        assert_eq!(nested.counts(), small.counts(), "padded {padding}");
    }
    assert!(small.counts().free > 100 && small.counts().occupied > 100);

    let (whole, grid) = (
        map.grid(device, Occlusion::Occupied).unwrap(),
        part.grid(device, Occlusion::Occupied).unwrap(),
    );
    let at = |local: Vec3| pose.position + pose.rotation * local;
    // Over the tray's floor beside the block, seen into: free. In a wall, in the floor and in
    // the block under its top, where nothing saw past: solid.
    let seen_into = at(Vec3::new(-0.04, 0.0, 0.0));
    assert!(grid.distance(seen_into).value > 0.0, "the tray's inside");
    for hidden in [
        at(Vec3::new(half.x - wall, 0.0, 0.0)),
        at(Vec3::new(0.0, 0.0, wall - half.z)),
        at(Vec3::new(0.03, 0.0, -0.015)),
    ] {
        let d = grid.distance(hidden).value;
        assert!(d <= 0.0, "{}: {d} at {hidden}, hidden", device.name());
    }
    // Under the table, beside the tray: solid in the whole map's grid, and not in the part's.
    let under = Vec3::new(-0.2, 0.2, -0.05);
    assert!(whole.distance(under).value <= 0.0 && grid.distance(under).value > 0.05);
    // The grid reaches the padding past the tray's box, to within a voxel.
    let (corner, bounds) = (
        half.x.hypot(half.y),
        (grid.lower_corner(), grid.upper_corner()),
    );
    assert!(
        bounds.0.x < pose.position.x - 0.9 * corner - padding + 0.02,
        "{bounds:?}"
    );
    assert!(
        bounds.1.x > pose.position.x + 0.9 * corner + padding - 0.02,
        "{bounds:?}"
    );
    assert!(
        (bounds.0.z - (-padding)).abs() < 0.011 && (bounds.1.z - (0.1 + padding)).abs() < 0.011
    );
    // Around the tray, in the grid's box, it never reads farther than the tray's walls are;
    // past the box, where the tray is the padding away or more, it reads clear by most of that.
    let mut rng = Rng::new(9);
    let (mut near, mut past) = (0, 0);
    for _ in 0..2000 {
        let p = at(Vec3::new(
            rng.range(-0.3, 0.3),
            rng.range(-0.3, 0.3),
            rng.range(-0.04, 0.2),
        ));
        let truly = tray
            .iter()
            .map(|o| o.distance(p).value)
            .fold(f32::INFINITY, f32::min);
        if region.distance(p).value <= 0.0 {
            continue;
        }
        let d = grid.distance(p).value;
        if p.cmpge(bounds.0).all() && p.cmple(bounds.1).all() {
            near += usize::from(truly < 0.05);
            assert!(
                d <= truly + 1e-4,
                "{}: {d} at {p}, {truly} from the tray",
                device.name()
            );
        } else {
            past += 1;
            assert!(truly > padding - 0.01 && d > 0.5 * padding, "{d} at {p}");
        }
    }
    assert!(
        near > 100 && past > 100,
        "{near} points near the tray, {past} past the grid"
    );

    // A region half off the map: what the map does not reach was never observed.
    let edge = Obstacle::cuboid(Vec3::new(0.3, 0.0, 0.1), Vec3::splat(0.05));
    let off = map.part(&edge, 0.05).unwrap();
    let counts = off.counts();
    assert!(
        counts.unobserved > 300 && counts.free > 300 && counts.occupied == 0,
        "{counts:?}: the half the map reaches was seen through, the other never observed"
    );
    assert_eq!(off.occupancy(Vec3::new(0.33, 0.0, 0.1)), None);
    let solid = off.grid(device, Occlusion::Occupied).unwrap();
    assert!(solid.distance(Vec3::new(0.33, 0.0, 0.1)).value <= 0.0);
}

/// A reading on a region's own face lies in a voxel whose middle may be just outside it: a block
/// seen from above, its top between two layers of the map's points, and the block itself as the
/// region. The part holds the top, so its grid has the block whatever never-observed space is.
#[test]
fn a_part_keeps_the_readings_on_its_regions_faces() {
    let top = 0.1054;
    let block = Obstacle::cuboid(
        Vec3::new(0.0, 0.0, 0.5 * top),
        Vec3::new(0.05, 0.05, 0.5 * top),
    );
    let table = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.05), Vec3::new(0.5, 0.5, 0.05));
    let far = vec![0.0; 640 * 480];
    let image = DepthImage {
        depth: &far,
        width: 640,
        intrinsics: Intrinsics {
            fx: 525.0,
            fy: 525.0,
            cx: 319.5,
            cy: 239.5,
        },
        camera: Pose::new(Vec3::new(0.0, 0.0, 0.7), Quat::from_rotation_x(PI)),
    };
    let (on_block, on_table) = (shape_depth(&image, &block), shape_depth(&image, &table));
    let depth: Vec<f32> = (on_block.iter().zip(&on_table))
        .map(|(a, b)| a.min(*b))
        .collect();
    let image = DepthImage {
        depth: &depth,
        ..image
    };
    let above = Vec3::new(0.0, 0.0, top + 0.01);
    for device in devices() {
        let mut map =
            OccupancyMap::try_new(Vec3::new(-0.2, -0.2, -0.05), Vec3::new(0.2, 0.2, 0.3), 0.01)
                .unwrap();
        map.integrate(device, &image, None).unwrap();
        // The top's readings lie in the layer of points over the block: outside it.
        let layer = Vec3::new(0.0, 0.0, 0.11);
        assert!(map.occupancy(layer).unwrap() > 0.5 && block.distance(layer).value > 0.0);
        let part = map.part(&block, 0.05).unwrap();
        assert!(part.counts().occupied >= 100, "{:?}", part.counts());
        for unknown in [Occlusion::Free, Occlusion::Band(0.03), Occlusion::Occupied] {
            let (whole, grid) = (
                map.grid(device, unknown).unwrap(),
                part.grid(device, unknown).unwrap(),
            );
            let d = grid.distance(above).value;
            assert!(
                d <= whole.distance(above).value + 1e-6 && d < 0.01,
                "{}, {unknown:?}: {d} a centimeter over the block",
                device.name()
            );
        }
    }
}

/// A map says where it is occupied: the middles of the voxels more likely occupied than not, on
/// its points and in their order, as many as it counts, on what the camera saw; of a part, those
/// in its region alone. A block on a table seen from above, and a voxel at even odds, which is
/// not occupied.
#[test]
fn a_map_says_where_it_is_occupied() {
    let block = Obstacle::cuboid(Vec3::new(0.06, -0.03, 0.031), Vec3::new(0.04, 0.025, 0.031));
    let table = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.05), Vec3::new(0.5, 0.5, 0.05));
    let far = vec![0.0; 640 * 480];
    let image = DepthImage {
        depth: &far,
        width: 640,
        intrinsics: Intrinsics {
            fx: 525.0,
            fy: 525.0,
            cx: 319.5,
            cy: 239.5,
        },
        camera: Pose::new(Vec3::new(0.0, 0.0, 0.7), Quat::from_rotation_x(PI)),
    };
    let bare = shape_depth(&image, &table);
    let both: Vec<f32> = (shape_depth(&image, &block).iter().zip(&bare))
        .map(|(a, b)| a.min(*b))
        .collect();
    let (with, without) = (
        DepthImage {
            depth: &both,
            ..image
        },
        DepthImage {
            depth: &bare,
            ..image
        },
    );
    // Longer along x than along y, and the block off both axes: x and y are not alike.
    let (lo, hi, voxel) = (
        Vec3::new(-0.2, -0.15, -0.05),
        Vec3::new(0.25, 0.15, 0.2),
        0.01,
    );
    let reach = 0.5 * 3f32.sqrt() * voxel;
    for device in devices() {
        let name = device.name();
        let fresh = OccupancyMap::try_new(lo, hi, voxel).unwrap();
        assert!(fresh.occupied().is_empty(), "a new map");
        let mut map = fresh.clone();
        map.integrate(device, &with, None).unwrap();
        let points = map.occupied();
        assert_eq!(points.len(), map.counts().occupied, "{name}");
        let (mut on_block, mut last) = (0, None);
        for &p in &points {
            assert!(map.occupancy(p).unwrap() > 0.5, "{name}: {p}");
            let at = (p - lo) / voxel;
            assert!(
                (at - at.round()).abs().max_element() < 1e-3,
                "{name}: {p} is off the map's points"
            );
            let [x, y, z] = at.round().to_array().map(|a| a as u32);
            assert!(last < Some((z, y, x)), "{name}: {p} out of order");
            last = Some((z, y, x));
            let (from_block, from_table) = (block.distance(p).value, table.distance(p).value);
            assert!(
                from_block.abs().min(from_table.abs()) <= 2.0 * reach,
                "{name}: {p} is {from_block} from the block and {from_table} from the table"
            );
            on_block += usize::from(from_block.abs() < from_table.abs());
        }
        assert!(
            on_block > 30 && points.len() > on_block + 1_000,
            "{name}: {on_block} of {} points on the block",
            points.len()
        );

        // A part's are the map's within its reach of the region, and no voxel it leaves out.
        let part = map.part(&block, 0.05).unwrap();
        let kept = part.occupied();
        assert_eq!(kept.len(), part.counts().occupied, "{name}");
        let near: Vec<Vec3> = (points.iter())
            .copied()
            .filter(|&p| block.distance(p).value <= reach)
            .collect();
        assert!(
            near.len() > 30,
            "{name}: {} points near the block",
            near.len()
        );
        assert_eq!(kept.len(), near.len(), "{name}");
        for (&p, &q) in kept.iter().zip(&near) {
            assert!(
                p.distance(q) < 1e-5,
                "{name}: {p} in the part, {q} in the map"
            );
            assert!(block.distance(p).value <= reach + 1e-5, "{name}: {p}");
        }

        // The block's top seen five times, seen through thirteen and seen twice more: even odds,
        // which are not more likely occupied than not.
        let mut even = fresh.clone();
        let looks = [&with; 5]
            .into_iter()
            .chain([&without; 13])
            .chain([&with; 2]);
        for image in looks {
            even.integrate(device, image, None).unwrap();
        }
        let top = Vec3::new(0.06, -0.03, 0.06);
        assert_eq!(even.occupancy(top), Some(0.5), "{name}");
        let occupied = even.occupied();
        assert_eq!(occupied.len(), even.counts().occupied, "{name}");
        assert!(
            (occupied.iter()).all(|&p| even.occupancy(p).unwrap() > 0.5),
            "{name}"
        );
    }
}

/// A 640x480 camera 70 cm over the middle of a table, looking straight down, reading `depth`.
fn from_above(depth: &[f32]) -> DepthImage<'_> {
    DepthImage {
        depth,
        width: 640,
        intrinsics: Intrinsics {
            fx: 525.0,
            fy: 525.0,
            cx: 319.5,
            cy: 239.5,
        },
        camera: Pose::new(Vec3::new(0.0, 0.0, 0.7), Quat::from_rotation_x(PI)),
    }
}

/// Per pixel of `image`, the depth of the nearest of `shapes`; 0 where its ray meets none.
fn scene_depth(image: &DepthImage, shapes: &[&Obstacle]) -> Vec<f32> {
    let depths: Vec<Vec<f32>> = shapes.iter().map(|s| shape_depth(image, s)).collect();
    (0..image.depth.len())
        .map(|i| depths.iter().map(|d| d[i]).fold(f32::INFINITY, f32::min))
        .map(|z| if z.is_finite() { z } else { 0.0 })
        .collect()
}

/// A table whose top is at `z`, a meter on a side.
fn table_at(z: f32) -> Obstacle {
    Obstacle::cuboid(Vec3::new(0.0, 0.0, z - 0.05), Vec3::new(0.5, 0.5, 0.05))
}

/// When one image decides (`Weighing::Latest`), a voxel five looks saw empty is occupied after one
/// look that reads a surface in it, and one five looks filled is free after one look that sees
/// through it, each as surely as five looks weighed as evidence make it. Weighed as evidence, the
/// first takes three looks at the surface, and the second is still occupied after one look
/// through.
#[test]
fn one_image_decides_what_it_sees() {
    let block = Obstacle::cuboid(Vec3::new(0.06, -0.03, 0.031), Vec3::new(0.04, 0.025, 0.031));
    let blank = vec![0.0; 640 * 480];
    let (with, without) = (
        scene_depth(&from_above(&blank), &[&block, &table_at(0.0)]),
        scene_depth(&from_above(&blank), &[&table_at(0.0)]),
    );
    let (with, without) = (from_above(&with), from_above(&without));
    // The middle of the block's top.
    let top = Vec3::new(0.06, -0.03, 0.06);
    for device in devices() {
        let name = device.name();
        let looked = |weighing: Weighing, images: &[&DepthImage]| {
            let (lo, hi) = (Vec3::new(-0.2, -0.15, -0.05), Vec3::new(0.25, 0.15, 0.2));
            let mut map = OccupancyMap::try_new(lo, hi, 0.01)
                .unwrap()
                .with_weighing(weighing);
            for image in images {
                map.integrate(device, image, None).unwrap();
            }
            map.occupancy(top).unwrap()
        };
        // Five looks weighed as evidence reach the clamps: as sure as a map holds anything.
        let (full, empty) = (
            looked(Weighing::Evidence, &[&with; 5]),
            looked(Weighing::Evidence, &[&without; 5]),
        );
        assert!(full > 0.95 && empty < 0.15, "{name}: {full} and {empty}");
        let mut images = vec![&without; 5];
        for looks in 1..=3 {
            images.push(&with);
            let weighed = looked(Weighing::Evidence, &images);
            assert_eq!(
                weighed > 0.5,
                looks == 3,
                "{name}: {weighed} after {looks} looks at the block"
            );
        }
        assert_eq!(
            looked(Weighing::Latest, &images[..6]),
            full,
            "{name}: seen empty five times, then read"
        );
        let images = [&with, &with, &with, &with, &with, &without];
        let weighed = looked(Weighing::Evidence, &images);
        assert!(
            weighed > 0.5,
            "{name}: {weighed}, filled, then seen through"
        );
        assert_eq!(
            looked(Weighing::Latest, &images),
            empty,
            "{name}: filled five times, then seen through"
        );
    }
}

/// What a deciding image does not see keeps what earlier images said: a block outside the image,
/// and a block hidden behind a board held over it. What it sees, it decides: another block that
/// is gone, and the board.
#[test]
fn what_one_image_does_not_see_keeps_what_it_was() {
    let right = Obstacle::cuboid(Vec3::new(0.06, -0.03, 0.031), Vec3::new(0.04, 0.025, 0.031));
    let left = Obstacle::cuboid(Vec3::new(-0.08, 0.04, 0.031), Vec3::new(0.03, 0.03, 0.031));
    let board = Obstacle::cuboid(Vec3::new(0.06, -0.03, 0.203), Vec3::new(0.07, 0.05, 0.005));
    let table = table_at(0.0);
    let blank = vec![0.0; 640 * 480];
    let seen = |shapes: &[&Obstacle]| scene_depth(&from_above(&blank), shapes);
    let (both, bare, covered) = (
        seen(&[&right, &left, &table]),
        seen(&[&table]),
        seen(&[&right, &left, &board, &table]),
    );
    // The left half of the image without the blocks, which shows no point past x = 0.
    let half: Vec<f32> = bare
        .chunks(640)
        .flat_map(|row| &row[..320])
        .copied()
        .collect();
    let half = DepthImage {
        depth: &half,
        width: 320,
        ..from_above(&blank)
    };
    let (right_top, left_top) = (Vec3::new(0.06, -0.03, 0.06), Vec3::new(-0.08, 0.04, 0.06));
    // The layer of points holding the board's top.
    let on_board = Vec3::new(0.06, -0.03, 0.21);
    for device in devices() {
        let name = device.name();
        let (lo, hi) = (Vec3::new(-0.2, -0.15, -0.05), Vec3::new(0.25, 0.15, 0.25));
        let mut map = OccupancyMap::try_new(lo, hi, 0.01)
            .unwrap()
            .with_weighing(Weighing::Latest);
        map.integrate(device, &from_above(&both), None).unwrap();
        let before = map.occupancy(right_top);
        assert!(
            before.unwrap() > 0.5 && map.occupancy(left_top).unwrap() > 0.5,
            "{name}: both blocks are seen"
        );
        map.integrate(device, &half, None).unwrap();
        let gone = map.occupancy(left_top).unwrap();
        assert!(gone < 0.5, "{name}: {gone} where the left block was");
        assert_eq!(
            map.occupancy(right_top),
            before,
            "{name}: the right block, outside the image"
        );
        map.integrate(device, &from_above(&covered), None).unwrap();
        let board = map.occupancy(on_board).unwrap();
        assert!(board > 0.5, "{name}: {board} on the board");
        assert_eq!(
            map.occupancy(right_top),
            before,
            "{name}: the right block, hidden behind the board"
        );
    }
}

/// A voxel that holds a reading is occupied whatever rays pass through it, also when one image
/// decides: a table seen at a grazing angle, its top 4 mm under a layer of the map's points, so
/// that the voxels holding its readings lie in front of the readings their middles project to.
#[test]
fn a_voxel_holding_a_reading_is_occupied_whatever_passes_through_it() {
    let blank = vec![0.0; 640 * 480];
    let image = DepthImage {
        camera: look_at(Vec3::new(0.0, -0.6, 0.2), Vec3::ZERO),
        ..from_above(&blank)
    };
    let depth = scene_depth(&image, &[&table_at(-0.004)]);
    let image = DepthImage {
        depth: &depth,
        ..image
    };
    let (lo, hi, voxel) = (
        Vec3::new(-0.15, -0.15, -0.05),
        Vec3::new(0.15, 0.15, 0.1),
        0.01,
    );
    // The voxels holding a reading, of readings not within rounding of a voxel's face.
    let mut voxels: Vec<[i32; 3]> = vec![];
    for p in image.points(None).unwrap().into_iter().flatten() {
        let at = (p - lo) / voxel;
        let inside = at.cmpge(Vec3::splat(-0.5)).all() && at.cmplt((hi - lo) / voxel + 0.5).all();
        let off_faces = ((at - at.floor() - 0.5).abs()).min_element() > 1e-3;
        if inside && off_faces {
            voxels.push(at.round().to_array().map(|a| a as i32));
        }
    }
    voxels.sort_unstable();
    voxels.dedup();
    let holding: Vec<Vec3> = (voxels.iter())
        .map(|&[x, y, z]| lo + Vec3::new(x as f32, y as f32, z as f32) * voxel)
        .collect();
    // Those whose middle the ray of the pixel it projects to passes, in front of its reading by
    // more than half a voxel.
    let k = image.intrinsics;
    let to_camera = image.camera.inverse();
    let passed = |p: Vec3| {
        let c = to_camera.position + to_camera.rotation * p;
        let u = (k.fx * c.x / c.z + k.cx + 0.5).floor();
        let v = (k.fy * c.y / c.z + k.cy + 0.5).floor();
        let within = (0.0..640.0).contains(&u) && (0.0..480.0).contains(&v);
        within && c.z < depth[v as usize * 640 + u as usize] - 0.5 * voxel - 1e-3
    };
    let passed_through = holding.iter().filter(|&&p| passed(p)).count();
    assert!(
        passed_through > 200,
        "{passed_through} of {} voxels holding a reading are passed through",
        holding.len()
    );
    for device in devices() {
        for weighing in [Weighing::Evidence, Weighing::Latest] {
            let mut map = OccupancyMap::try_new(lo, hi, voxel)
                .unwrap()
                .with_weighing(weighing);
            map.integrate(device, &image, None).unwrap();
            for &p in &holding {
                let seen = map.occupancy(p).unwrap();
                assert!(
                    seen > 0.5,
                    "{}, {weighing:?}: {seen} at {p}, passed through: {}",
                    device.name(),
                    passed(p)
                );
            }
        }
    }
}

/// What no image has observed is marked alike whatever the weighing: after one look at a table
/// with the arm before it, masked out, the voxels with no occupancy are the same under either,
/// and so are the grids that take never-observed space as solid, and solid but for the arm's
/// shadow; after a second look, at the table alone, the voxels with no occupancy still are.
#[test]
fn either_weighing_marks_what_no_image_observed_alike() {
    let scene = TableScene::new();
    let vga = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let robot = common::panda().unwrap();
    let bare = scene.render(640, 480, vga, false);
    let table = scene.image(&bare, 640, vga);
    let arm = with_arm(&table, &robot, robot.default_q());
    let image = scene.image(&arm, 640, vga);
    let mask = image
        .robot_mask(&robot, robot.default_q(), 0.02, &[])
        .unwrap();
    let (lo, hi) = (Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3));
    let points = lattice(lo, hi, 0.01);
    let unobserved = |map: &OccupancyMap| -> Vec<bool> {
        (points.iter())
            .map(|&p| map.occupancy(p).is_none())
            .collect()
    };
    for device in devices() {
        let name = device.name();
        let mut maps = [Weighing::Evidence, Weighing::Latest].map(|weighing| {
            let mut map = OccupancyMap::try_new(lo, hi, 0.01)
                .unwrap()
                .with_weighing(weighing);
            map.integrate(device, &image, Some(&mask)).unwrap();
            map
        });
        let [evidence, latest] = &maps;
        let never = unobserved(evidence);
        let hidden = never.iter().filter(|&&n| n).count();
        assert!(hidden > 10_000, "{name}: {hidden} voxels never observed");
        assert!(never == unobserved(latest), "{name}");
        let differ = (points.iter())
            .filter(|&&p| evidence.occupancy(p) != latest.occupancy(p))
            .count();
        assert!(
            differ > 10_000,
            "{name}: the weighings agree on all but {differ}"
        );
        let grids = |map: &OccupancyMap| {
            [Occlusion::Occupied, Occlusion::OccupiedExceptRobotShadow]
                .map(|unknown| map.grid(device, unknown).unwrap())
        };
        let (weighed, decided) = (grids(evidence), grids(latest));
        assert!(weighed[0] != weighed[1], "{name}: the arm hides space");
        assert!(weighed == decided, "{name}");
        for map in &mut maps {
            map.integrate(device, &table, None).unwrap();
        }
        assert!(
            unobserved(&maps[0]) == unobserved(&maps[1]),
            "{name}: after a second look"
        );
    }
}

/// The GPU fuses images into a map that one image decides as the CPU does: after three looks (a
/// ball on a table, the arm before the table, masked out, and the table alone), the maps differ at
/// no more voxels than images are classified differently on the devices, one in 2,000 at most.
#[test]
fn every_device_fuses_alike_when_one_image_decides() {
    let robot = common::panda().unwrap();
    let scene = TableScene::new();
    let vga = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let (ball, bare) = (
        scene.render(640, 480, vga, true),
        scene.render(640, 480, vga, false),
    );
    let table = scene.image(&bare, 640, vga);
    let arm = with_arm(&table, &robot, robot.default_q());
    let arm = scene.image(&arm, 640, vga);
    let mask = arm
        .robot_mask(&robot, robot.default_q(), 0.02, &[])
        .unwrap();
    let looks = [
        (scene.image(&ball, 640, vga), None),
        (arm, Some(&mask)),
        (table, None),
    ];
    let (lo, hi) = (Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3));
    let points = lattice(lo, hi, 0.01);
    let fused = |device: &Device, weighing: Weighing| -> Vec<Option<f32>> {
        let mut map = OccupancyMap::try_new(lo, hi, 0.01)
            .unwrap()
            .with_weighing(weighing);
        for (image, mask) in &looks {
            map.integrate(device, image, *mask).unwrap();
        }
        points.iter().map(|&p| map.occupancy(p)).collect()
    };
    let (decided, weighed) = (
        fused(cpu(), Weighing::Latest),
        fused(cpu(), Weighing::Evidence),
    );
    let differ =
        |a: &[Option<f32>], b: &[Option<f32>]| a.iter().zip(b).filter(|(a, b)| a != b).count();
    let between = differ(&decided, &weighed);
    assert!(between > 10_000, "the weighings agree on all but {between}");
    for device in &devices()[1..] {
        let count = differ(&decided, &fused(device, Weighing::Latest));
        eprintln!(
            "{}: {count} of {} voxels differ",
            device.name(),
            points.len()
        );
        assert!(
            count * 2000 <= points.len(),
            "{}: {count} of {} voxels differ",
            device.name(),
            points.len()
        );
    }
}

/// A fuse changes what the image sees, the voxels that hold its readings and, with a mask, the
/// space the robot and the mask's shapes fill, and nothing else. The CPU visits only those voxels
/// and the GPU every one, so the GPU's map is the reference. On a map 3 m across and 1.8 m high,
/// after a look from high over it, a camera inside it looking out past its side, the top and
/// bottom of its image running along the map's rows, and a zoomed one outside it looking in, turned
/// to them, each see a table with a block on it, the floor and the arm: with no mask, with the arm
/// masked out, and with the table, the block and a box neither camera sees given to the mask as
/// well, under both weighings. The devices' maps differ at no more voxels than their
/// classification does (one in 2,000, as `every_device_fuses_alike_when_one_image_decides`
/// allows: a voxel within rounding of a pixel's edge or a reading's depth lands either way). On
/// both, every voxel more than a voxel outside the view keeps what the look from above made it
/// unless the mask frees it, as it frees the whole box; and the voxels holding a reading whose
/// middles project past the image's edge take their hits, some of them in rows the view misses.
#[test]
fn a_fuse_changes_only_what_the_image_sees_and_the_mask_frees() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let table = Obstacle::cuboid(Vec3::new(0.5, 0.0, 0.1), Vec3::new(0.3, 0.5, 0.05));
    let block = Obstacle::cuboid(Vec3::new(0.5, 0.2, 0.2), Vec3::splat(0.05));
    let floor = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.25), Vec3::new(3.0, 3.0, 0.05));
    let aside = Obstacle::cuboid(Vec3::new(-1.1, -1.1, 0.4), Vec3::splat(0.1));
    let (lo, hi, voxel) = (Vec3::new(-1.5, -1.5, -0.3), Vec3::new(1.5, 1.5, 1.5), 0.03);
    let points = lattice(lo, hi, voxel);
    // What `camera` reads of the scene, and of the arm before it.
    let look = |camera: Pose, intrinsics: Intrinsics| {
        let blank = vec![0.0; 640 * 480];
        let image = DepthImage {
            depth: &blank,
            width: 640,
            intrinsics,
            camera,
        };
        let scene: Vec<f32> = scene_depth(&image, &[&table, &block, &floor, &aside])
            .into_iter()
            .map(|z| if z > 0.0 { z } else { f32::INFINITY })
            .collect();
        with_arm(
            &DepthImage {
                depth: &scene,
                ..image
            },
            &robot,
            &q,
        )
    };
    // Whether `p` lies outside one of the half-spaces a point projects into `image` in, by more
    // than `by` and, for a side slanted to the map's rows (x), `along` times the slant (meters),
    // in doubles. A row's span reaches past the view by its margin and a voxel along the row at
    // most (`sdf::span`): past `by` 1 mm and `along` two voxels, a voxel lies outside its span.
    let outside = |image: &DepthImage, p: Vec3, by: f64, along: f64| {
        let rotation = image.camera.rotation.as_dquat();
        let c = rotation.inverse() * (p.as_dvec3() - image.camera.position.as_dvec3());
        let Intrinsics { fx, fy, cx, cy } = image.intrinsics;
        let sides = [
            Vec3::Z,
            Vec3::new(fx, 0.0, cx + 0.5),
            Vec3::new(-fx, 0.0, 640.0 - cx - 0.5),
            Vec3::new(0.0, fy, cy + 0.5),
            Vec3::new(0.0, -fy, 480.0 - cy - 0.5),
        ];
        sides.iter().any(|n| {
            let n = n.as_dvec3().normalize();
            n.dot(c) < -(by + along * (rotation * n).x.abs())
        })
    };
    // The log-odds at the clamps, as `occupancy` gives them.
    let (bottom, top) = (1.0 / (1.0 + 2f32.exp()), 1.0 / (1.0 + (-3.5f32).exp()));
    let raised = |was: Option<f32>, is: Option<f32>| {
        is.is_some_and(|is| is > was.unwrap_or(0.5) || is == top)
    };
    let lowered = |was: Option<f32>, is: Option<f32>| {
        is.is_some_and(|is| is < was.unwrap_or(0.5) || is == bottom)
    };
    let wide = Intrinsics {
        fx: 200.0,
        fy: 200.0,
        ..k
    };
    let above = look_at(Vec3::new(0.1, 0.1, 2.4), Vec3::ZERO);
    let from_above = look(above, wide);
    let from_above = DepthImage {
        depth: &from_above,
        width: 640,
        intrinsics: wide,
        camera: above,
    };
    let spheres = robot.spheres(&q);
    let reach = 0.5 * 3f32.sqrt() * voxel;
    let zoomed = Intrinsics {
        fx: 1000.0,
        fy: 1000.0,
        ..k
    };
    let cameras = [
        (
            "inside, looking out",
            look_at(Vec3::new(0.5, -1.2, 0.7), Vec3::new(0.5, 1.0, 0.0)),
            k,
        ),
        (
            "outside, looking in",
            look_at(Vec3::new(2.4, 0.9, 1.3), Vec3::new(0.4, 0.0, 0.1)),
            zoomed,
        ),
    ];
    let mut past_spans = 0;
    for (name, camera, intrinsics) in cameras {
        let depth = look(camera, intrinsics);
        let image = DepthImage {
            depth: &depth,
            width: 640,
            intrinsics,
            camera,
        };
        let arm = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
        let on = arm.on_robot.iter().filter(|&&on| on).count();
        assert!(on > 1_000, "{name}: {on} readings of the arm");
        let given = [table.clone(), block.clone(), aside.clone()];
        let held = (arm.clone())
            .with_shapes(cpu(), &image, &given, [0.004, 0.0, 0.0])
            .unwrap();
        let away: Vec<bool> = (points.iter())
            .map(|&p| outside(&image, p, voxel.into(), 0.0))
            .collect();
        let unseen = away.iter().filter(|&&a| a).count();
        assert!(
            unseen * 3 > points.len() * 2,
            "{name}: {unseen} of {} voxels outside the view",
            points.len()
        );
        let in_aside: Vec<usize> = (0..points.len())
            .filter(|&n| aside.distance(points[n]).value <= 0.0)
            .collect();
        let past_view = |n: &usize| outside(&image, points[*n], 1e-3, 2.0 * f64::from(voxel));
        assert!(
            in_aside.len() > 200 && in_aside.iter().all(past_view),
            "{name}: the box's {} voxels lie outside the view and their rows' spans",
            in_aside.len()
        );
        let masks = [
            ("no mask", None),
            ("the arm", Some(&arm)),
            ("shapes", Some(&held)),
        ];
        for (label, mask) in masks {
            // The middles of the voxels holding a reading whose middles project past the image's
            // edge, of readings not within rounding of a voxel's face.
            let mut edge: Vec<Vec3> = vec![];
            for p in image.points(mask).unwrap().into_iter().flatten() {
                let at = (p - lo) / voxel;
                let on_map =
                    at.cmpge(Vec3::splat(-0.5)).all() && at.cmplt((hi - lo) / voxel + 0.5).all();
                let off_faces = ((at - at.floor() - 0.5).abs()).min_element() > 1e-3;
                let middle = lo + at.round() * voxel;
                if on_map && off_faces && outside(&image, middle, 1e-6, 0.0) {
                    edge.push(middle);
                }
            }
            edge.sort_unstable_by(|a, b| a.to_array().partial_cmp(&b.to_array()).unwrap());
            edge.dedup();
            assert!(
                edge.len() > 30,
                "{name}, {label}: {} voxels at the edge",
                edge.len()
            );
            past_spans += (edge.iter())
                .filter(|&&p| outside(&image, p, 1e-3, 2.0 * f64::from(voxel)))
                .count();
            // Which voxels the mask may free: inside a grown sphere or reaching into a shape,
            // with a voxel to spare.
            let freed = |p: Vec3| {
                let in_arm = (spheres.iter())
                    .any(|&[x, y, z, r]| (p - Vec3::new(x, y, z)).length() < r + 0.04 + voxel);
                let in_shape = given.iter().any(|s| s.distance(p).value <= reach + voxel);
                mask.is_some() && in_arm || label == "shapes" && in_shape
            };
            for weighing in [Weighing::Evidence, Weighing::Latest] {
                let mut before = OccupancyMap::try_new(lo, hi, voxel)
                    .unwrap()
                    .with_weighing(weighing);
                before.integrate(cpu(), &from_above, None).unwrap();
                let was: Vec<Option<f32>> = points.iter().map(|&p| before.occupancy(p)).collect();
                let fused: Vec<Vec<Option<f32>>> = (devices().iter())
                    .map(|device| {
                        let mut map = before.clone();
                        map.integrate(device, &image, mask).unwrap();
                        let name = device.name();
                        let is: Vec<Option<f32>> =
                            points.iter().map(|&p| map.occupancy(p)).collect();
                        let kept = (0..points.len()).filter(|&n| away[n] && !freed(points[n]));
                        for n in kept {
                            assert_eq!(
                                is[n], was[n],
                                "{name}, {label}, {weighing:?}: {} outside the view",
                                points[n]
                            );
                        }
                        let box_freed = label == "shapes";
                        for &n in in_aside.iter().filter(|_| box_freed) {
                            assert!(
                                lowered(was[n], is[n]),
                                "{name}, {weighing:?}: {} in the box, {:?} after {:?}",
                                points[n],
                                is[n],
                                was[n]
                            );
                        }
                        for &p in &edge {
                            let (was, is) = (before.occupancy(p), map.occupancy(p));
                            assert!(
                                raised(was, is),
                                "{name}, {label}, {weighing:?}: {p} holds a reading, {is:?} after \
                                 {was:?}"
                            );
                        }
                        is
                    })
                    .collect();
                let changed = (fused[0].iter().zip(&was)).filter(|(a, b)| a != b).count();
                assert!(
                    changed > 2_000,
                    "{name}, {label}, {weighing:?}: {changed} voxels change"
                );
                for (device, is) in devices().iter().zip(&fused).skip(1) {
                    let differ = (is.iter().zip(&fused[0])).filter(|(a, b)| a != b).count();
                    eprintln!(
                        "{name}, {label}, {weighing:?}, {}: {differ} of {} voxels differ",
                        device.name(),
                        points.len()
                    );
                    assert!(
                        differ * 2000 <= points.len(),
                        "{name}, {label}, {weighing:?}, {}: {differ} voxels differ",
                        device.name()
                    );
                }
            }
        }
    }
    assert!(
        past_spans > 60,
        "{past_spans} voxels holding a reading lie outside their rows' spans"
    );
}

/// A depth camera reads a pixel that mixes a near surface with what lies behind it somewhere
/// between the two. Filtered, a reading between a nearer and a farther neighbour, or between one
/// of them and a pixel without a reading, is left out; the surfaces on both sides keep every
/// reading up to the edge, and pixels without a reading beside one surface take nothing from it.
/// A reading of either surface with a step on one side and a pixel without a reading straight
/// across goes too: the pixel without one may be the other surface.
#[test]
fn a_reading_between_two_surfaces_is_left_out() {
    let (width, height) = (64, 48);
    let k = Intrinsics {
        fx: 60.0,
        fy: 60.0,
        cx: 31.5,
        cy: 23.5,
    };
    // A near surface over the left, a far one over the right, and between them two columns
    // that read in between, as mixed pixels do. A slit one pixel wide in the near surface shows
    // the far one, and a wire one pixel wide hangs before the far one: neither lies between.
    let (near, far) = (0.4, 0.7);
    let column = |u: usize| match u {
        15 => far,
        0..=29 | 50 => near,
        30 => 0.5,
        31 => 0.6,
        _ => far,
    };
    let mut depth: Vec<f32> = (0..width * height).map(|i| column(i % width)).collect();
    let image = |depth: &[f32]| {
        let image = DepthImage {
            depth,
            width,
            intrinsics: k,
            camera: Pose::IDENTITY,
        };
        image.filtered(&DepthFilter::default()).unwrap()
    };
    let kept = image(&depth);
    for (i, (&was, &is)) in depth.iter().zip(&kept).enumerate() {
        let between = (30..=31).contains(&(i % width));
        assert_eq!(is, if between { 0.0 } else { was }, "pixel {i}");
    }
    // Where the far surface beside them has no reading, as in a stereo camera's shadow, where
    // the near one has none in its top rows, at one pixel on its edge and in a ring around one
    // reading in its middle: the two columns go all the same, and neither surface loses a
    // reading to the pixels without one beside it, but for the one reading of the near surface
    // that has such a pixel on one side and the step straight across from it.
    for (i, depth) in depth.iter_mut().enumerate() {
        let (u, v) = (i % width, i / width);
        let shadow = (32..=34).contains(&u);
        let unread = v < 8 && (27..=29).contains(&u);
        let ring = (5..=7).contains(&u) && (20..=22).contains(&v) && (u, v) != (6, 21);
        if shadow || unread || ring || (u, v) == (29, 30) {
            *depth = 0.0;
        }
    }
    let kept = image(&depth);
    for (i, (&was, &is)) in depth.iter().zip(&kept).enumerate() {
        let (u, v) = (i % width, i / width);
        let between = (30..=31).contains(&u) || (u, v) == (29, 8);
        assert_eq!(
            is,
            if between { 0.0 } else { was },
            "pixel {u}, {v} beside a hole"
        );
    }
    // A depth that is no reading comes out as none.
    for none in [-0.3, f32::NAN, f32::INFINITY] {
        depth[5] = none;
        assert_eq!(image(&depth)[5], 0.0);
    }
    // Options that say nothing are refused.
    let image = DepthImage {
        depth: &depth,
        width,
        intrinsics: k,
        camera: Pose::IDENTITY,
    };
    let bad = [
        DepthFilter {
            noise: [0.0, -0.001, 0.0],
            ..Default::default()
        },
        DepthFilter {
            noise: [f32::NAN, 0.0, 0.0],
            ..Default::default()
        },
        DepthFilter {
            steepest: 0.0,
            ..Default::default()
        },
        DepthFilter {
            steepest: PI / 2.0,
            ..Default::default()
        },
        DepthFilter {
            deviations: -1.0,
            ..Default::default()
        },
        // Each of these would make every step one surface's, and the filter leave all in.
        DepthFilter {
            noise: [f32::INFINITY, 0.0, 0.0],
            ..Default::default()
        },
        DepthFilter {
            steepest: f32::NAN,
            ..Default::default()
        },
        DepthFilter {
            deviations: f32::INFINITY,
            ..Default::default()
        },
    ];
    for options in bad {
        assert!(
            matches!(image.filtered(&options), Err(Error::Input(_))),
            "{options:?}"
        );
    }
    // And an image whose depths do not form rows.
    for (depth, width) in [(&depth[..10], 3), (&depth[..], 0)] {
        let image = DepthImage {
            depth,
            width,
            ..image
        };
        let ragged = image.filtered(&DepthFilter::default());
        assert!(matches!(ragged, Err(Error::Input(_))), "rows of {width}");
    }
}

/// What one surface can step between two pixels is the width they are apart times the tangent
/// of the steepest turn, and so many deviations of the noise in two readings' difference. Along
/// a row, a column and both diagonals, a wall turned 78 degrees keeps every reading and one
/// turned 82 loses those with a neighbour on both sides; a step just inside what the noise
/// reaches stays and one just past it goes, whichever coefficient the noise comes from.
#[test]
fn a_surface_is_one_as_far_as_it_is_turned_and_the_noise_reaches() {
    let (width, height) = (64, 48);
    // A long lens, so a wall stands to every ray as it is turned, and pixels taller than wide.
    let k = Intrinsics {
        fx: 6000.0,
        fy: 4000.0,
        cx: 31.5,
        cy: 23.5,
    };
    let filtered = |depth: &[f32], options: &DepthFilter| {
        let image = DepthImage {
            depth,
            width,
            intrinsics: k,
            camera: Pose::IDENTITY,
        };
        image.filtered(options).unwrap()
    };
    // A wall through a point 0.5 m ahead, turned so its depth grows along (du, dv) of the image.
    let wall = |turn: f32, (du, dv): (isize, isize)| -> Vec<f32> {
        let along = Vec2::new(du as f32, dv as f32).normalize();
        (0..width * height)
            .map(|i| {
                let (u, v) = ((i % width) as f32, (i / width) as f32);
                let ray = Vec2::new((u - k.cx) / k.fx, (v - k.cy) / k.fy);
                0.5 / (1.0 - turn.to_radians().tan() * along.dot(ray))
            })
            .collect()
    };
    for (du, dv) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
        let (shallow, steep) = (wall(78.0, (du, dv)), wall(82.0, (du, dv)));
        assert!(
            filtered(&shallow, &DepthFilter::default()) == shallow,
            "along {du}, {dv}"
        );
        let kept = filtered(&steep, &DepthFilter::default());
        for (i, &z) in kept.iter().enumerate() {
            let (u, v) = ((i % width) as isize, (i / width) as isize);
            let inside = |u: isize, v: isize| {
                (0..width as isize).contains(&u) && (0..height as isize).contains(&v)
            };
            let both = inside(u + du, v + dv) && inside(u - du, v - dv);
            assert_eq!(
                z == 0.0,
                both,
                "pixel {u}, {v} of a wall edge on along {du}, {dv}"
            );
        }
        // Turned as far as the caller allows, it stays.
        let allowed = DepthFilter {
            steepest: 85f32.to_radians(),
            ..Default::default()
        };
        assert!(filtered(&steep, &allowed) == steep, "along {du}, {dv}");
    }

    // Three levels 2 m off, a column of the middle one between the other two. A reading's
    // noise there is 1 cm from each coefficient alone, so two deviations of the difference of
    // two readings is 2.83 cm, and the slope's part 0.19 cm: a step of 2.9 cm is one surface
    // and one of 3.15 cm is not.
    let steps = |step: f32| -> Vec<f32> {
        let level = |u: usize| 2.0 + step * (u.cmp(&32) as i8 as f32);
        (0..width * height).map(|i| level(i % width)).collect()
    };
    for noise in [[0.01, 0.0, 0.0], [0.0, 0.005, 0.0], [0.0, 0.0, 0.0025]] {
        let options = DepthFilter {
            noise,
            deviations: 2.0,
            ..Default::default()
        };
        let (within, past) = (steps(0.029), steps(0.0315));
        assert!(filtered(&within, &options) == within, "{noise:?}");
        let kept = filtered(&past, &options);
        for (i, (&was, &is)) in past.iter().zip(&kept).enumerate() {
            let middle = i % width == 32;
            assert_eq!(is, if middle { 0.0 } else { was }, "pixel {i}, {noise:?}");
        }
    }

    // A wall square on at 1 m, each reading 2 cm off as a stereo camera's are across a room,
    // many times what the wall's own slope could step between two pixels: two thirds of its
    // readings go where the filter is not told the noise, and none where it is.
    let mut rng = Rng::new(4);
    let noisy: Vec<f32> = (0..width * height)
        .map(|_| 1.0 + 0.02 * rng.normal())
        .collect();
    let lost = |options: &DepthFilter| {
        let kept = filtered(&noisy, options);
        kept.iter().filter(|&&z| z == 0.0).count()
    };
    let told = DepthFilter {
        noise: [0.0, 0.0, 0.02],
        ..Default::default()
    };
    let (unaware, aware) = (lost(&DepthFilter::default()), lost(&told));
    assert!(
        unaware * 2 > width * height && aware == 0,
        "{unaware} readings lost without the noise, {aware} with it"
    );
}

/// In a scene: a block on a table seen from above without noise loses no reading, its outline
/// included. With the pixels around its outline reading between its top and the table, a map
/// fused from the image holds points in mid-air beside the block; filtered first, it holds
/// none, and the block and the table are there as before.
#[test]
fn a_filtered_image_puts_nothing_in_mid_air() {
    let top = 0.1;
    let block = Obstacle::cuboid(
        Vec3::new(0.0, 0.0, 0.5 * top),
        Vec3::new(0.05, 0.05, 0.5 * top),
    );
    let table = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.05), Vec3::new(0.5, 0.5, 0.05));
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let far = vec![0.0; 640 * 480];
    let image = DepthImage {
        depth: &far,
        width: 640,
        intrinsics: k,
        camera: Pose::new(Vec3::new(0.0, 0.0, 0.7), Quat::from_rotation_x(PI)),
    };
    let (on_block, on_table) = (shape_depth(&image, &block), shape_depth(&image, &table));
    let exact: Vec<f32> = (on_block.iter().zip(&on_table))
        .map(|(a, b)| a.min(*b))
        .collect();
    let image = DepthImage {
        depth: &exact,
        ..image
    };
    assert!(image.filtered(&DepthFilter::default()).unwrap() == exact);
    // The table's pixels that touch the block's read halfway up its side.
    let mixed: Vec<f32> = (0..exact.len())
        .map(|i| {
            let (u, v) = (i % 640, i / 640);
            // The block is in the image's middle: its pixels have every neighbour.
            let inner = (1..639).contains(&u) && (1..479).contains(&v);
            let beside = inner
                && [i - 1, i + 1, i - 640, i + 640]
                    .iter()
                    .any(|&n| on_block[n].is_finite());
            match on_block[i].is_infinite() && beside {
                true => 0.7 - 0.5 * top,
                false => exact[i],
            }
        })
        .collect();
    let flying = mixed.iter().zip(&exact).filter(|(a, b)| a != b).count();
    assert!(flying > 200, "{flying} pixels around the block");
    let image = DepthImage {
        depth: &mixed,
        ..image
    };
    let kept = image.filtered(&DepthFilter::default()).unwrap();
    for (i, (&was, &is)) in exact.iter().zip(&kept).enumerate() {
        let lost = mixed[i] != was;
        assert_eq!(is, if lost { 0.0 } else { was }, "pixel {i}");
    }
    // Halfway up the block's side, a voxel out from it.
    let beside = Vec3::new(0.06, 0.0, 0.5 * top);
    for device in devices() {
        let fused = |depth: &[f32]| {
            let mut map =
                OccupancyMap::try_new(Vec3::new(-0.2, -0.2, -0.05), Vec3::new(0.2, 0.2, 0.3), 0.01)
                    .unwrap();
            let image = DepthImage { depth, ..image };
            map.integrate(device, &image, None).unwrap();
            map
        };
        let (raw, clean) = (fused(&mixed), fused(&kept));
        assert!(
            raw.occupancy(beside).is_some_and(|o| o > 0.5),
            "the mixed pixels' points"
        );
        assert!(
            clean.occupancy(beside).is_none_or(|o| o < 0.5),
            "{}",
            device.name()
        );
        for seen in [Vec3::new(0.0, 0.0, top), Vec3::new(0.15, 0.1, 0.0)] {
            assert!(
                clean.occupancy(seen).is_some_and(|o| o > 0.5),
                "{seen} is seen"
            );
        }
    }
}

/// The table, two blocks and the camera of the tests of shapes a caller holds: a table two meters
/// from the arm, a block on it, and a smaller block beside the first.
fn held_shapes() -> ([Obstacle; 3], Pose, Intrinsics) {
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let shapes = [
        Obstacle::cuboid(Vec3::new(2.5, 0.0, -0.05), Vec3::new(0.4, 0.4, 0.05)),
        Obstacle::cuboid(Vec3::new(2.5, 0.0, 0.05), Vec3::splat(0.05)),
        Obstacle::cuboid(Vec3::new(2.3, 0.2, 0.03), Vec3::splat(0.03)),
    ];
    let camera = look_at(Vec3::new(3.3, 0.0, 0.7), Vec3::new(2.5, 0.0, 0.0));
    (shapes, camera, k)
}

/// The depths of a 640 by 480 look at `shapes` from `camera`, 0 where a ray meets none.
fn look_at_shapes(shapes: &[&Obstacle], camera: Pose, k: Intrinsics) -> Vec<f32> {
    let none = vec![0.0; 640 * 480];
    let image = DepthImage {
        depth: &none,
        width: 640,
        intrinsics: k,
        camera,
    };
    let depths: Vec<Vec<f32>> = shapes.iter().map(|s| shape_depth(&image, s)).collect();
    let nearest = |i: usize| depths.iter().map(|d| d[i]).fold(f32::INFINITY, f32::min);
    let depth = (0..none.len()).map(nearest);
    depth.map(|z| if z.is_finite() { z } else { 0.0 }).collect()
}

/// A mask given shapes flags exactly the readings within the margin of one, the margin growing
/// with the reading's depth as its three coefficients say: for a box, a ball, a cylinder and a
/// capsule, turned, and for a floor that reaches behind the camera.
#[test]
fn a_mask_flags_the_readings_within_the_margin_of_a_shape() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let (shapes, camera, k) = held_shapes();
    let turn = Quat::from_euler(EulerRot::XYZ, 0.3, -0.4, 0.9);
    let more = [
        Obstacle::Cuboid {
            center: Vec3::new(2.6, -0.25, 0.08),
            half_extents: Vec3::new(0.06, 0.02, 0.04),
            rotation: turn,
        },
        Obstacle::Sphere {
            center: Vec3::new(2.3, -0.2, 0.05),
            radius: 0.05,
        },
        Obstacle::Cylinder {
            center: Vec3::new(2.7, 0.25, 0.06),
            rotation: turn,
            radius: 0.03,
            half_height: 0.05,
        },
        Obstacle::Capsule {
            center: Vec3::new(2.45, 0.3, 0.05),
            rotation: turn,
            radius: 0.02,
            half_length: 0.04,
        },
    ];
    // A floor under it all, out past the camera's own place.
    let floor = Obstacle::cuboid(Vec3::new(2.5, 0.0, -0.3), Vec3::new(3.0, 3.0, 0.1));
    let scene: Vec<&Obstacle> = shapes.iter().chain(&more).chain([&floor]).collect();
    let depth = look_at_shapes(&scene, camera, k);
    let image = DepthImage {
        depth: &depth,
        width: 640,
        intrinsics: k,
        camera,
    };
    let robot_mask = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
    assert!(!robot_mask.on_robot.contains(&true), "the arm is elsewhere");
    let points = image.points(None).unwrap();
    // Held: every shape but the second block, each alone and all at once.
    let all: Vec<Obstacle> = (scene.iter().enumerate())
        .filter(|&(i, _)| i != 2)
        .map(|(_, s)| (*s).clone())
        .collect();
    let alone = all.iter().map(std::slice::from_ref);
    for held in alone.chain([&all[..]]) {
        // The last two reach farther the deeper a reading lies, past the box's own far end.
        let margins = [
            [0.004, 0.0, 0.0],
            [0.001, 0.002, 0.003],
            [0.0, 0.0, 0.02],
            [0.0, 0.0, 0.05],
            [0.05, 0.05, 0.05],
        ];
        for margin in margins {
            let mask = (robot_mask.clone())
                .with_shapes(&devices()[0], &image, held, margin)
                .unwrap();
            let within = |(i, p): (usize, &Option<Vec3>)| {
                let reach = margin[0] + (margin[1] + margin[2] * depth[i]) * depth[i];
                p.is_some_and(|p| held.iter().any(|s| s.distance(p).value <= reach))
            };
            let expected: Vec<bool> = points.iter().enumerate().map(within).collect();
            let flags = expected.iter().filter(|&&f| f).count();
            assert!(flags > 500, "{flags} readings of {} shapes", held.len());
            let wrong = (mask.on_robot.iter().zip(&expected))
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                wrong,
                0,
                "of {flags} readings, {} shapes, {margin:?}",
                held.len()
            );
        }
    }
    // A box off to one side and a wall behind it, at margins that reach much farther for a
    // reading on the wall than for one at the box's far end: the pixels a box may be read in
    // are those of its reach at the deepest such reading (30 cm past the box at the first
    // margin), and every pixel where the reach outruns the depth (the second). The reach
    // outruns the depth again past 3.5 m at the first margin: a wall 4 m off is read within
    // 3.2 m of the box, far past the box grown by its reach at 1.5 m. A second box nearer the
    // camera is outrun again only past 4.6 m: the wall 4 m off is the first box's at any pixel
    // and the second's in none but those of its own reach.
    let aside = Pose::new(Vec3::splat(5.0), Quat::IDENTITY);
    let boxed = [
        Obstacle::cuboid(
            aside.position + Vec3::new(-0.5, 0.0, 1.0),
            Vec3::splat(0.05),
        ),
        Obstacle::cuboid(aside.position + Vec3::new(0.3, 0.0, 0.3), Vec3::splat(0.05)),
    ];
    let walls = [
        ([0.0, 0.0, 0.2], 1.3),
        ([0.0, 0.0, 0.3], 3.0),
        ([0.0, 0.0, 0.2], 4.0),
    ];
    for (margin, wall) in walls {
        let depth = vec![wall; 640 * 480];
        let image = DepthImage {
            depth: &depth,
            camera: aside,
            ..image
        };
        let mask = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
        let mask = mask
            .with_shapes(&devices()[0], &image, &boxed, margin)
            .unwrap();
        let reach = margin[2] * wall * wall;
        let within = |p: &Option<Vec3>| {
            p.is_some_and(|p| boxed.iter().any(|s| s.distance(p).value <= reach))
        };
        let expected: Vec<bool> = image.points(None).unwrap().iter().map(within).collect();
        let flags = expected.iter().filter(|&&f| f).count();
        assert!(
            flags > 10_000,
            "{flags} readings of the wall within {reach} m of a box"
        );
        assert!(
            mask.on_robot == expected,
            "a wall {wall} m off at {margin:?}"
        );
    }
    // The second block, held by none, is in view: what of it lies farther than the margin from
    // every shape is unflagged, as the check above holds every pixel to.
    let second = shape_depth(&image, &shapes[2]);
    let its = (0..depth.len()).filter(|&i| second[i] == depth[i]).count();
    assert!(its > 1_000, "{its} readings of the second block");
    // A mask of another image, a margin below zero or no number, and a shape that is none.
    let small = DepthImage {
        depth: &depth[..640 * 400],
        ..image
    };
    let flat = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.1, -0.1, 0.1));
    let mask = || robot_mask.clone();
    for refused in [
        mask().with_shapes(&devices()[0], &small, &all, [0.004, 0.0, 0.0]),
        mask().with_shapes(&devices()[0], &image, &all, [0.004, -0.1, 0.0]),
        mask().with_shapes(&devices()[0], &image, &all, [f32::NAN, 0.0, 0.0]),
        mask().with_shapes(&devices()[0], &image, &[flat], [0.004, 0.0, 0.0]),
    ] {
        assert!(matches!(refused, Err(Error::Input(_))));
    }
    // A cell's fixtures: a floor far larger than the view, a wall beside the camera whose box
    // reaches behind it, a wall wholly behind it, which no reading lies within the margin of, and
    // a thin wall turned 45 degrees about the vertical; each alone, and all with the table and
    // the blocks.
    let boxes = [
        (Vec3::new(3.0, 0.0, -0.25), Vec3::new(8.0, 8.0, 0.05)),
        (Vec3::new(2.8, 0.9, 0.4), Vec3::new(2.0, 0.02, 0.6)),
        (Vec3::new(4.5, 0.0, 0.5), Vec3::new(0.02, 2.0, 1.0)),
    ];
    // The least and the greatest depth of each box's corners.
    let depths = boxes.map(|(center, half)| {
        let corners = (0..8).map(|c| {
            let pick = BVec3::new(c & 1 > 0, c & 2 > 0, c & 4 > 0);
            let corner = Vec3::select(pick, center + half, center - half);
            (camera.rotation.inverse() * (corner - camera.position)).z
        });
        corners.fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), z| {
            (a.min(z), b.max(z))
        })
    });
    let [floor, beside, behind] = boxes.map(|(center, half)| Obstacle::cuboid(center, half));
    assert!(
        depths[..2]
            .iter()
            .all(|&(near, far)| near < 0.0 && far > 0.0)
    );
    assert!(depths[2].1 < 0.0, "the third wall is behind the camera");
    let turned = Obstacle::Cuboid {
        center: Vec3::new(1.9, -0.5, 0.3),
        half_extents: Vec3::new(0.6, 0.01, 0.5),
        rotation: Quat::from_rotation_z(PI / 4.0),
    };
    let cell = [floor, beside, behind, turned];
    let scene: Vec<&Obstacle> = shapes.iter().chain(&cell).collect();
    let depth = look_at_shapes(&scene, camera, k);
    let image = DepthImage {
        depth: &depth,
        ..image
    };
    let points = image.points(None).unwrap();
    let robot_mask = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
    let all: Vec<Obstacle> = scene.iter().map(|s| (*s).clone()).collect();
    let margins = [
        [0.004, 0.0, 0.0],
        [0.001, 0.002, 0.003],
        [0.0, 0.0, 0.02],
        [0.0, 0.0, 0.05],
        [0.05, 0.05, 0.05],
    ];
    let alone = cell.iter().map(std::slice::from_ref);
    for (which, held) in alone.chain([&all[..]]).enumerate() {
        for margin in margins {
            let mask = (robot_mask.clone())
                .with_shapes(&devices()[0], &image, held, margin)
                .unwrap();
            let within = |(i, p): (usize, &Option<Vec3>)| {
                let reach = margin[0] + (margin[1] + margin[2] * depth[i]) * depth[i];
                p.is_some_and(|p| held.iter().any(|s| s.distance(p).value <= reach))
            };
            let expected: Vec<bool> = points.iter().enumerate().map(within).collect();
            let flags = expected.iter().filter(|&&f| f).count();
            // The wall behind the camera is read nowhere, the others in many pixels.
            let least = [100_000, 10_000, 0, 20_000, 250_000][which];
            assert!(
                if which == 2 {
                    flags == 0
                } else {
                    flags > least
                },
                "{flags} readings of shapes {which} at {margin:?}"
            );
            assert!(
                mask.on_robot == expected,
                "shapes {which} at {margin:?}: {} flagged where {flags}",
                mask.on_robot.iter().filter(|&&f| f).count()
            );
        }
    }
}

/// A mask given a distance grid as a shape flags the readings within the margin of the grid as
/// it is turned: a bar measured as a grid and turned over a table, and a grid solid on its faces.
#[test]
fn a_mask_flags_the_readings_within_the_margin_of_a_turned_grid() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let (shapes, camera, k) = held_shapes();
    let turn = Quat::from_euler(EulerRot::XYZ, 0.3, -0.4, 0.9);
    // A bar 16 by 6 by 4 cm, the distances to it at a grid's points.
    let (dims, voxel, origin) = ([14u32, 9, 8], 0.02, Vec3::new(-0.13, -0.08, -0.07));
    let bar = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.08, 0.03, 0.02));
    let mut distances = vec![];
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let p = origin + Vec3::new(i as f32, j as f32, k as f32) * voxel;
                distances.push(bar.distance(p).value);
            }
        }
    }
    let at = Vec3::new(2.5, 0.05, 0.15);
    let grids = [
        Obstacle::Sdf {
            grid: Arc::new(SdfGrid::try_new(dims, voxel, origin, &distances).unwrap()),
            center: at,
            rotation: turn,
        },
        Obstacle::Sdf {
            grid: Arc::new(
                SdfGrid::try_new([6; 3], 0.02, Vec3::splat(-0.05), &[-0.03; 216]).unwrap(),
            ),
            center: at,
            rotation: turn,
        },
    ];
    // What the camera reads: the table, and the bar as a box where the grid stands.
    let stand_in = bar.placed(Pose::new(at, turn));
    let depth = look_at_shapes(&[&shapes[0], &stand_in], camera, k);
    let image = DepthImage {
        depth: &depth,
        width: 640,
        intrinsics: k,
        camera,
    };
    let robot_mask = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
    let points = image.points(None).unwrap();
    for (which, grid) in grids.iter().enumerate() {
        // Turned as it stands and not, the grid's readings are not the same ones.
        let straight = match grid {
            Obstacle::Sdf { grid, center, .. } => Obstacle::Sdf {
                grid: Arc::clone(grid),
                center: *center,
                rotation: Quat::IDENTITY,
            },
            _ => unreachable!("both are grids"),
        };
        for margin in [[0.004, 0.0, 0.0], [0.001, 0.002, 0.003]] {
            let mask = (robot_mask.clone())
                .with_shapes(&devices()[0], &image, std::slice::from_ref(grid), margin)
                .unwrap();
            let reach = |i: usize| margin[0] + (margin[1] + margin[2] * depth[i]) * depth[i];
            let within = |shape: &Obstacle| -> Vec<bool> {
                let its = |(i, p): (usize, &Option<Vec3>)| {
                    p.is_some_and(|p| shape.distance(p).value <= reach(i))
                };
                points.iter().enumerate().map(its).collect()
            };
            let expected = within(grid);
            let flags = expected.iter().filter(|&&f| f).count();
            assert!(flags > 500, "{flags} readings of grid {which}");
            // The bar's readings tell how it is turned; the solid cube's need not.
            let unturned = within(&straight);
            let moved = (expected.iter().zip(&unturned))
                .filter(|(a, b)| a != b)
                .count();
            assert!(
                which == 1 || moved > 100,
                "{moved} readings tell the bar's turn"
            );
            assert!(
                mask.on_robot == expected,
                "grid {which} at {margin:?}: {} flagged where {flags}",
                mask.on_robot.iter().filter(|&&f| f).count()
            );
        }
    }
}

/// A mask given shapes keeps what it held: the robot's readings stay flagged and its space free
/// in the slabs a shape reaches too, and a second call keeps the first call's shapes.
#[test]
fn a_mask_given_shapes_keeps_the_robot_and_earlier_shapes() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    // A table before the arm, as high as its base, and a block on it, seen from across the
    // table with the arm.
    let table = Obstacle::cuboid(Vec3::new(0.6, 0.0, 0.1), Vec3::new(0.3, 0.6, 0.05));
    let block = Obstacle::cuboid(Vec3::new(0.6, 0.3, 0.18), Vec3::splat(0.03));
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let camera = look_at(Vec3::new(1.3, 0.2, 0.8), Vec3::new(0.45, 0.0, 0.0));
    // Where a ray meets neither shape it may still meet the arm: no reading is as far as can be.
    let shapes: Vec<f32> = look_at_shapes(&[&table, &block], camera, k)
        .into_iter()
        .map(|z| if z > 0.0 { z } else { f32::INFINITY })
        .collect();
    let image = DepthImage {
        depth: &shapes,
        width: 640,
        intrinsics: k,
        camera,
    };
    let depth = with_arm(&image, &robot, &q);
    let image = DepthImage {
        depth: &depth,
        ..image
    };
    let margin = [0.004, 0.0, 0.0];
    // A point of the map's lattice inside one of the arm's spheres as the mask grows them, on
    // a layer of voxels the table reaches, and clear of the table itself: at the height of its
    // middle, at the arm's base.
    let spheres = robot.spheres(&q);
    let lattice = (-20..=20).flat_map(|x| (-20..=20).map(move |y| (x as f32, y as f32)));
    let behind = lattice
        .map(|(x, y)| Vec3::new(x, y, 10.0) / 100.0)
        .find(|&p| {
            let in_arm = |&[x, y, z, r]: &[f32; 4]| (p - Vec3::new(x, y, z)).length() < r + 0.03;
            table.distance(p).value > 0.02 && spheres.iter().any(in_arm)
        });
    let behind = behind.expect("a sphere of the arm's base reaches the table's height");
    for device in devices() {
        let arm = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
        let its = arm.on_robot.iter().filter(|&&own| own).count();
        assert!(its > 1_000, "{its} readings of the arm");
        let held = (arm.clone())
            .with_shapes(device, &image, std::slice::from_ref(&table), margin)
            .unwrap()
            .with_shapes(device, &image, std::slice::from_ref(&block), margin)
            .unwrap();
        let kept = arm.on_robot.iter().zip(&held.on_robot);
        assert!(
            kept.clone().all(|(arm, held)| !arm || *held),
            "the arm's readings stay"
        );
        let read = depth.iter().filter(|z| z.is_finite()).count();
        let flagged = held.on_robot.iter().filter(|&&own| own).count();
        assert_eq!(
            flagged, read,
            "the arm, the table and the block are all that is read"
        );
        let mut map =
            OccupancyMap::try_new(Vec3::new(-0.3, -0.8, -0.1), Vec3::new(1.1, 0.8, 0.9), 0.01)
                .unwrap();
        map.integrate(device, &image, Some(&held)).unwrap();
        assert_eq!(map.counts().occupied, 0);
        let free = [
            ("in the arm, at the table's height", behind),
            (
                "in the table, which the first call gave",
                Vec3::new(0.6, 0.0, 0.1),
            ),
            (
                "in the block, which the second gave",
                Vec3::new(0.6, 0.3, 0.18),
            ),
        ];
        for (what, p) in free {
            assert!(
                map.occupancy(p).is_some_and(|o| o < 0.5),
                "{p}, {what}, is free on {}",
                device.name()
            );
        }
    }
}

/// What a caller holds as a shape is left out of a map. With a table and a block on it given to
/// the mask, one look fuses neither: the space each fills is free, with the voxels that reach
/// into it, and what only the block hides is shadow. So nothing is said to the map when the
/// block is carried away. What a shape does not cover stays the map's, as does a block no shape
/// explains, and a reading the caller flags itself is left out too.
#[test]
fn what_a_caller_holds_as_a_shape_is_left_out_of_a_map() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let ([table, block, other], camera, k) = held_shapes();
    let margin = [0.004, 0.0, 0.0];
    // Inside the block, at its middle and a centimeter in from a corner, and at the middle of
    // its top; before it toward the camera; on the table behind it, which the camera sees only
    // once the block is gone; and inside the table, at its middle and near an edge.
    let inside = Vec3::new(2.5, 0.0, 0.05);
    let corner = Vec3::new(2.54, 0.04, 0.05);
    let top = Vec3::new(2.5, 0.0, 0.1);
    let before = Vec3::new(2.62, 0.0, 0.2);
    let hidden = Vec3::new(2.4, 0.0, 0.02);
    let under = Vec3::new(2.5, 0.0, -0.05);
    let edge = Vec3::new(2.88, 0.38, -0.03);
    for device in devices() {
        let new = || {
            OccupancyMap::try_new(Vec3::new(2.0, -0.5, -0.1), Vec3::new(3.0, 0.5, 0.4), 0.01)
                .unwrap()
        };
        // A look at `shapes` fused into `map`, with `held` given to the mask and the readings
        // `flag` picks flagged by the caller.
        let fuse = |map: &mut OccupancyMap,
                    shapes: &[&Obstacle],
                    held: &[Obstacle],
                    flag: &dyn Fn(Vec3) -> bool| {
            let depth = look_at_shapes(shapes, camera, k);
            let image = DepthImage {
                depth: &depth,
                width: 640,
                intrinsics: k,
                camera,
            };
            let mask = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
            let mut mask = mask
                .with_shapes(&devices()[0], &image, held, margin)
                .unwrap();
            let points = image.points(None).unwrap();
            for (own, p) in mask.on_robot.iter_mut().zip(&points) {
                *own |= p.is_some_and(flag);
            }
            map.integrate(device, &image, Some(&mask)).unwrap();
            let left = image.points(Some(&mask)).unwrap();
            let flags = mask.on_robot.iter().filter(|&&f| f).count();
            assert_eq!(
                left.iter().flatten().count() + flags,
                points.iter().flatten().count(),
                "a flagged reading is no point either"
            );
        };
        let both = [table.clone(), block.clone()];
        let no = |_: Vec3| false;

        // One look, and nothing said to the map besides.
        let mut map = new();
        fuse(&mut map, &[&table, &block], &both, &no);
        assert_eq!(map.counts().occupied, 0, "nothing of a held shape is fused");
        for free in [inside, corner, top, before, under, edge] {
            assert!(
                map.occupancy(free).is_some_and(|o| o < 0.5),
                "{free} is free on {}",
                device.name()
            );
        }
        // What only the block hides is shadow: solid where never-observed space is, and free
        // where the caller takes shadow for free.
        assert!(map.occupancy(hidden).is_none(), "nothing saw {hidden}");
        let solid = map.grid(device, Occlusion::Occupied).unwrap();
        let shadow = map
            .grid(device, Occlusion::OccupiedExceptRobotShadow)
            .unwrap();
        assert!(solid.distance(hidden).value <= 0.0 && shadow.distance(hidden).value > 0.0);

        // The block is carried away: the looks after say so, and nobody else has to. Five of
        // them, so that the space over the table is as surely free as a map holds it.
        for _ in 0..5 {
            fuse(&mut map, &[&table], &both[..1], &no);
        }
        assert_eq!(map.counts().occupied, 0);
        for free in [inside, before, hidden] {
            assert!(
                map.occupancy(free).is_some_and(|o| o < 0.5),
                "{free} is free once the block is gone"
            );
        }
        // A block no shape explains is the map's: filled by its third look, as a voxel seen
        // through that often is, and not before.
        let read = Vec3::new(2.3, 0.2, 0.06);
        for look in 1..=3 {
            fuse(&mut map, &[&table, &other], &both[..1], &no);
            let filled = map.occupancy(read).is_some_and(|o| o > 0.5);
            assert_eq!(filled, look == 3, "after {look} looks at the other block");
        }

        // A shape measured small, a centimeter in on each side and two under the top: the
        // block's top, which it does not cover, is the map's.
        let small = Obstacle::cuboid(Vec3::new(2.5, 0.0, 0.04), Vec3::splat(0.04));
        let mut map = new();
        fuse(&mut map, &[&table, &block], &[table.clone(), small], &no);
        assert!(map.occupancy(top).is_some_and(|o| o > 0.5), "{top} is read");
        // A block whose top lies 2 mm under a layer of voxels, nearer than the camera tells a
        // voxel before a surface from one on it: the layer reaches into the block, and is free
        // with it.
        let low = Obstacle::cuboid(Vec3::new(2.5, 0.0, 0.049), Vec3::new(0.05, 0.05, 0.049));
        let mut map = new();
        fuse(
            &mut map,
            &[&table, &low],
            &[table.clone(), low.clone()],
            &no,
        );
        assert_eq!(map.counts().occupied, 0);
        assert!(map.occupancy(top).is_some_and(|o| o < 0.5), "{top} is free");
        // The block's readings flagged by the caller, as of an object it is about to grasp, and
        // no shape given for it: they are left out too.
        let mut map = new();
        let its = |p: Vec3| block.distance(p).value < 0.004;
        fuse(&mut map, &[&table, &block], &both[..1], &its);
        assert_eq!(map.counts().occupied, 0, "a caller's flags are honoured");
    }
}

/// How many pixels apart `i` and `j` of an image `width` pixels wide lie along rows, columns and
/// diagonals: the larger of their offsets.
fn pixels_apart(i: usize, j: usize, width: usize) -> usize {
    (i % width)
        .abs_diff(j % width)
        .max((i / width).abs_diff(j / width))
}

/// The point of the box `shape` nearest `p`, found in the box's frame: `p` itself inside it.
fn box_nearest(shape: &Obstacle, p: Vec3) -> Vec3 {
    let Obstacle::Cuboid {
        center,
        half_extents,
        rotation,
    } = *shape
    else {
        unreachable!("the band's tests hold boxes")
    };
    let local = rotation.inverse() * (p - center);
    rotation * local.clamp(-half_extents, half_extents) + center
}

/// Whether the ray from `p` along `dir`, from `p` on, meets the box `shape`: its slabs crossed in
/// the box's frame.
fn box_ahead(shape: &Obstacle, p: Vec3, dir: Vec3) -> bool {
    let Obstacle::Cuboid {
        center,
        half_extents,
        rotation,
    } = *shape
    else {
        unreachable!("the band's tests hold boxes")
    };
    let (from, along) = (rotation.inverse() * (p - center), rotation.inverse() * dir);
    let (mut first, mut last) = (0.0f32, f32::INFINITY);
    for axis in 0..3 {
        let (from, along, half) = (from[axis], along[axis], half_extents[axis]);
        if along == 0.0 {
            if from.abs() > half {
                return false;
            }
            continue;
        }
        let (a, b) = ((-half - from) / along, (half - from) / along);
        first = first.max(a.min(b));
        last = last.min(a.max(b));
    }
    first <= last
}

/// Whether the ray from `p` along `dir`, from `p` on, meets the ball of `radius` about `center`:
/// the ray's point nearest the centre lies in it.
fn ball_ahead(center: Vec3, radius: f32, p: Vec3, dir: Vec3) -> bool {
    let t = (center - p).dot(dir).max(0.0);
    (p + dir * t - center).length() <= radius
}

/// A band behind a mask's flags flags exactly the readings it should, as a check of every
/// unflagged reading against every flagged reading within the band and every thing the mask
/// holds says. A reading at depth z is taken where a flagged reading within the band is of a
/// thing (within the padding of one of the robot's spheres or of a held object, within the margin
/// at its depth of a shape) whose point nearest it lies nearer than z less the margin's reach at
/// z, and the reading's ray, carried on past it, does not meet that thing. A view of the Panda's
/// hand holding a box over a table with a turned block on it, two thirds of its readings moved
/// nearer or deeper at random: among them readings behind a sphere, the held box and a shape on
/// rays that pass through it first, taken, and readings in front of each, kept. At margins of
/// each coefficient alone and all three, and paddings of 1 and 3 cm; at bands of 0, 1, 2 and 5
/// pixels, one wider than the image and the widest there is; with flags a caller sets on pixels
/// without a reading (at 0, below it, no number and infinity) and on readings; on every device,
/// and on one thread, which runs its rows one after another with the same buffers. A margin
/// below zero or no number, and a mask of another image, are input errors.
#[test]
fn a_band_flags_exactly_the_readings_behind_a_thing_of_a_flag() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let hand = robot.ee_pose(&q, 0).position;
    let padding = 0.01;
    // A box 6 by 6 by 16 cm in the hand, a table 30 cm under the hand and a block on it.
    let held = Held {
        touch_links: vec![],
        also: vec![],
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.08)),
        grasp: Pose::new(Vec3::new(0.0, 0.0, -0.04), Quat::IDENTITY),
        locked: Default::default(),
        spheres: SphereOptions::default(),
    };
    let object = held
        .object
        .placed(robot.ee_pose(&q, 0).mul_pose(held.grasp.inverse()));
    let shapes = [
        Obstacle::cuboid(hand + Vec3::new(0.0, 0.0, -0.32), Vec3::new(0.4, 0.4, 0.02)),
        Obstacle::Cuboid {
            center: hand + Vec3::new(0.1, -0.12, -0.25),
            half_extents: Vec3::new(0.04, 0.03, 0.05),
            rotation: Quat::from_rotation_z(0.5),
        },
    ];
    let (width, height) = (48, 36);
    let k = Intrinsics {
        fx: 36.0,
        fy: 36.0,
        cx: 23.5,
        cy: 17.5,
    };
    let camera = look_at(
        hand + Vec3::new(0.55, 0.3, 0.2),
        hand + Vec3::new(0.0, 0.0, -0.15),
    );
    let none = vec![0.0; width * height];
    let blank = DepthImage {
        depth: &none,
        width,
        intrinsics: k,
        camera,
    };
    let seen: Vec<Vec<f32>> = (shapes.iter().chain([&object]))
        .map(|shape| shape_depth(&blank, shape))
        .collect();
    let scene: Vec<f32> = (0..none.len())
        .map(|i| seen.iter().map(|d| d[i]).fold(f32::INFINITY, f32::min))
        .collect();
    let rendered = with_arm(
        &DepthImage {
            depth: &scene,
            ..blank
        },
        &robot,
        &q,
    );
    // Of the readings, one in five moved 1 to 10 cm nearer, one in six 1 to 15 cm deeper and
    // three in ten up to 2.5 cm either way, which sets many near a thing's margin; and one pixel
    // in 25 without a reading.
    let mut rng = Rng::new(6);
    let unread = [0.0, -1.0, f32::NAN, f32::INFINITY];
    let depth: Vec<f32> = (rendered.iter().enumerate())
        .map(|(i, &z)| match rng.uniform() {
            _ if !z.is_finite() => 0.0,
            x if x < 0.04 => unread[i % 4],
            x if x < 0.24 => z - rng.range(0.01, 0.1),
            x if x < 0.4 => z + rng.range(0.01, 0.15),
            x if x < 0.7 => z + rng.range(-0.025, 0.025),
            _ => z,
        })
        .collect();
    let image = DepthImage {
        depth: &depth,
        ..blank
    };
    let points = image.points(None).unwrap();
    let read = |i: usize| points[i].is_some();
    for value in unread {
        let pixels = (0..depth.len()).filter(|&i| depth[i].to_bits() == value.to_bits());
        assert!(pixels.count() > 2, "pixels at {value}");
    }
    let forward = camera.rotation * Vec3::Z;
    let depth_of = |p: Vec3| forward.dot(p - camera.position);
    let spheres: Vec<(Vec3, f32)> = (robot.spheres(&q).iter())
        .map(|&[x, y, z, r]| (Vec3::new(x, y, z), r))
        .collect();
    // The things, by kind: the spheres, the held box, the shapes.
    let kind =
        |thing: usize| usize::from(thing >= spheres.len()) + usize::from(thing > spheres.len());
    let rays: Vec<Vec3> = (points.iter())
        .map(|p| p.map_or(Vec3::Z, |p| (p - camera.position).normalize()))
        .collect();
    // Whether the ray through reading `i`, carried on from `from`, meets thing `thing`.
    let meets = |thing: usize, from: Vec3, i: usize| match thing.checked_sub(spheres.len()) {
        None => ball_ahead(spheres[thing].0, spheres[thing].1, from, rays[i]),
        Some(0) => box_ahead(&object, from, rays[i]),
        Some(s) => box_ahead(&shapes[s - 1], from, rays[i]),
    };
    let things = spheres.len() + 1 + shapes.len();
    let ahead: Vec<Vec<bool>> = (0..depth.len())
        .map(|i| match points[i] {
            Some(p) => (0..things).map(|thing| meets(thing, p, i)).collect(),
            None => vec![],
        })
        .collect();
    let one = Device::cpu_threads(&robot, 1).unwrap();
    let every: Vec<&Device> = devices().iter().chain([&one]).collect();
    // Margins of each coefficient alone and all three, two reaching farther than the padding
    // and two less far, and one that grows so steeply with depth that it reaches a centimeter
    // farther at a reading 5 cm deeper; and two paddings, the second a shell in which a flagged
    // reading lies farther from its sphere than most margins reach.
    let margins = [
        [0.02, 0.0, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.0, 0.005],
        [0.001, 0.002, 0.003],
        [0.0, 0.0, 0.15],
    ];
    let cases = [padding, 0.03]
        .into_iter()
        .flat_map(|p| margins.map(|m| (p, m)));
    // Readings taken through a thing of each kind that their rays passed through first, and
    // readings in front of one, kept, at a band of five pixels.
    let (mut through, mut before) = ([0; 3], [0; 3]);
    for (padding, margin) in cases {
        let reach = |z: f32| margin[0] + (margin[1] + margin[2] * z) * z;
        let mut mask = image
            .robot_mask(&robot, &q, padding, std::slice::from_ref(&held))
            .unwrap()
            .with_shapes(cpu(), &image, &shapes, margin)
            .unwrap();
        // A caller's flags on every pixel without a reading, and on one reading in fifty.
        for i in 0..depth.len() {
            mask.on_robot[i] |= !read(i) || rng.uniform() < 0.02;
        }
        let flags = mask.on_robot.clone();
        // What each flagged reading is of, and the depth of that thing's point nearest it.
        let owners: Vec<Vec<(usize, f32)>> = (0..depth.len())
            .map(|j| {
                let Some(p) = points[j].filter(|_| flags[j]) else {
                    return vec![];
                };
                let balls = spheres.iter().enumerate().filter_map(|(s, &(c, r))| {
                    let off = p - c;
                    let near = if off.length() > r {
                        c + off * (r / off.length())
                    } else {
                        p
                    };
                    let grown = r + padding;
                    (off.length_squared() < grown * grown).then(|| (s, depth_of(near)))
                });
                let box_of = (object.distance(p).value < padding)
                    .then(|| (spheres.len(), depth_of(box_nearest(&object, p))));
                let shapes_of = (shapes.iter().enumerate())
                    .filter(|(_, shape)| shape.distance(p).value <= reach(depth[j]))
                    .map(|(s, shape)| (spheres.len() + 1 + s, depth_of(box_nearest(shape, p))));
                balls.chain(box_of).chain(shapes_of).collect()
            })
            .collect();
        let kinds = (0..3).map(|n| owners.iter().flatten().filter(|o| kind(o.0) == n).count());
        assert!(
            kinds.clone().all(|n| n > 0),
            "flags of each kind: {:?}",
            kinds.collect::<Vec<_>>()
        );
        for pixels in [0, 1, 2, 5, 2 * width, usize::MAX] {
            let window = |i: usize| {
                let (u, v) = (i % width, i / width);
                let end = |at: usize, n: usize| at.saturating_add(pixels).saturating_add(1).min(n);
                let columns = u.saturating_sub(pixels)..end(u, width);
                let rows = v.saturating_sub(pixels)..end(v, height);
                rows.flat_map(move |w| columns.clone().map(move |x| w * width + x))
            };
            let taken = |i: usize| {
                let nearer = depth[i] - reach(depth[i]);
                let by = |j: usize| owners[j].iter().any(|&(t, d)| d < nearer && !ahead[i][t]);
                read(i) && !flags[i] && window(i).any(by)
            };
            let expected: Vec<bool> = (0..depth.len()).map(|i| flags[i] || taken(i)).collect();
            let (count, after) = (
                flags.iter().filter(|&&f| f).count(),
                expected.iter().filter(|&&f| f).count(),
            );
            if pixels == 0 {
                assert_eq!(after, count, "a band of 0 at {margin:?}");
            } else {
                assert!(after > count, "a band of {pixels} at {margin:?}");
            }
            if pixels == 5 {
                for i in (0..depth.len()).filter(|&i| read(i) && !flags[i]) {
                    let nearer = depth[i] - reach(depth[i]);
                    let pairs = window(i).flat_map(|j| owners[j].iter().copied());
                    for (thing, _) in pairs.filter(|&(_, d)| d < nearer) {
                        let passed = meets(thing, camera.position, i);
                        if ahead[i][thing] {
                            before[kind(thing)] += 1;
                        } else if passed {
                            through[kind(thing)] += 1;
                        }
                    }
                }
            }
            for device in &every {
                let banded = (mask.clone())
                    .with_band(device, &image, margin, pixels)
                    .unwrap();
                assert!(
                    banded.on_robot == expected,
                    "a band of {pixels} at {margin:?} on {}: {} flagged where {after}, of {count}",
                    device.name(),
                    banded.on_robot.iter().filter(|&&f| f).count()
                );
            }
        }
    }
    assert!(
        through.iter().chain(&before).all(|&n| n > 0),
        "taken through a sphere, the held box and a shape: {through:?}; kept before: {before:?}"
    );
    // A margin below zero or no number, and a mask of another image.
    let other = DepthImage {
        depth: &depth[..width * (height - 1)],
        ..image
    };
    let mask = || image.robot_mask(&robot, &q, padding, &[]).unwrap();
    for refused in [
        mask().with_band(cpu(), &image, [0.004, -0.1, 0.0], 1),
        mask().with_band(cpu(), &image, [0.004, 0.0, f32::NAN], 1),
        mask().with_band(cpu(), &other, [0.004, 0.0, 0.0], 1),
    ] {
        assert!(matches!(refused, Err(Error::Input(_))));
    }
}

/// What stands on a table given as a shape keeps every reading whose ray goes on into the table,
/// whatever the band and the view. The table's own flagged readings a few rows below a box on it
/// lie nearer than the box's lower rows by more than the margin, but the table lies behind the
/// box along each of the box's rays. A box 1 cm high and 16 cm long that no shape explains, seen
/// from across the table as `examples/perception.rs` sees it, at bands of 1, 2, 5 and 8 pixels,
/// and from 30 cm and from 8.5 cm over the table 1.5 m out, at bands of 3 and 5: the band takes
/// none of its readings, and the map holds the voxels it holds without the band.
#[test]
fn a_band_keeps_what_stands_on_a_held_table_at_any_band() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let table = Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let block = Obstacle::cuboid(Vec3::new(0.55, -0.1, 0.005), Vec3::new(0.02, 0.08, 0.005));
    let margin = [0.004, 0.0, 0.0];
    // The last view meets the table at 3 degrees: a ray carried on from the box comes down to
    // the table too slowly to arrive in the steps a ray is given, and is taken to enter it.
    let views = [
        (
            look_at(Vec3::new(1.4, 0.3, 0.9), Vec3::new(0.4, 0.0, 0.1)),
            &[1, 2, 5, 8][..],
            250,
        ),
        (
            look_at(Vec3::new(2.05, -0.1, 0.3), Vec3::new(0.55, -0.1, 0.005)),
            &[3, 5][..],
            250,
        ),
        (
            look_at(Vec3::new(2.05, -0.1, 0.085), Vec3::new(0.55, -0.1, 0.005)),
            &[3, 5][..],
            50,
        ),
    ];
    let reaches = |p: Vec3| block.distance(p).value <= 0.5 * 3f32.sqrt() * 0.01;
    for device in devices() {
        let on = device.name();
        for (view, &(camera, bands, least)) in views.iter().enumerate() {
            let depth = look_at_shapes(&[&table, &block], camera, k);
            let image = DepthImage {
                depth: &depth,
                width: 640,
                intrinsics: k,
                camera,
            };
            let its = shape_depth(&image, &block);
            let mask = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
            let mask = mask
                .with_shapes(device, &image, std::slice::from_ref(&table), margin)
                .unwrap();
            // The box's readings but those within the margin of the table, which are its.
            let readings: Vec<usize> = (0..depth.len())
                .filter(|&i| its[i] == depth[i] && !mask.on_robot[i])
                .collect();
            assert!(
                readings.len() > least,
                "{} readings of the box, which no shape explains, from view {view}",
                readings.len()
            );
            let voxels = |mask: &RobotMask| {
                let mut map = OccupancyMap::try_new(
                    Vec3::new(-0.3, -0.8, -0.1),
                    Vec3::new(1.1, 0.8, 0.3),
                    0.01,
                )
                .unwrap();
                map.integrate(device, &image, Some(mask)).unwrap();
                map.occupied().into_iter().filter(|&p| reaches(p)).count()
            };
            let without = voxels(&mask);
            for &pixels in bands {
                // The box's readings with a flagged reading within the band nearer by more than
                // the margin: those a comparison of depths alone would take.
                let nearer = (readings.iter())
                    .filter(|&&i| {
                        let (u, v) = (i % 640, i / 640);
                        let rows = v.saturating_sub(pixels)..(v + pixels + 1).min(480);
                        let mut window = rows.flat_map(|w| {
                            (u.saturating_sub(pixels)..(u + pixels + 1).min(640))
                                .map(move |x| w * 640 + x)
                        });
                        window.any(|j| mask.on_robot[j] && depth[j] < depth[i] - margin[0])
                    })
                    .count();
                let banded = (mask.clone())
                    .with_band(device, &image, margin, pixels)
                    .unwrap();
                let taken = readings.iter().filter(|&&i| banded.on_robot[i]).count();
                let with = voxels(&banded);
                assert!(
                    (pixels < 3 || nearer > 0) && taken == 0 && with == without && without > 10,
                    "view {view}, a band of {pixels} on {on}: {nearer} of the box's {} readings \
                     have a flagged one nearer within the band; {taken} taken; its voxels \
                     {without} without the band and {with} with it",
                    readings.len()
                );
            }
        }
    }
}

/// A surface level with a flagged one keeps its readings: they are measured against the thing's
/// surface, not against the flagged readings beside them and their noise. A wall square on to
/// the camera, the half of it on one side a shape and the other half not. Read exactly at one
/// depth, at a margin of zero and a band of one pixel, the level half keeps every reading. Read
/// with noise of a third of the margin a deviation, at bands of 1, 2 and 5, it keeps every
/// reading within the margin of the wall's plane, though for many of them a flagged reading
/// within the band lies nearer by more than the margin.
#[test]
fn a_band_keeps_a_surface_level_with_a_flagged_one() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let (width, height) = (64, 48);
    let k = Intrinsics {
        fx: 64.0,
        fy: 64.0,
        cx: 31.5,
        cy: 23.5,
    };
    // A camera with the world's axes, away from the robot, and a wall a meter before it whose
    // half to the left of the image's middle is a shape.
    let camera = Pose::new(Vec3::new(5.0, 5.0, 0.0), Quat::IDENTITY);
    let half = Obstacle::cuboid(Vec3::new(4.5, 5.0, 1.25), Vec3::new(0.5, 1.0, 0.25));
    let mut rng = Rng::new(9);
    // Exact depths with a margin of zero and the mask padded, so that the depths alone pass
    // every level reading to the band's second look; then noise, with the mask unpadded.
    let cases = [
        (0.0, [0.0; 3], 0.04, &[1][..]),
        (0.004 / 3.0, [0.004, 0.0, 0.0], 0.0, &[1, 2, 5][..]),
    ];
    for (deviation, margin, padding, bands) in cases {
        let depth: Vec<f32> = (0..width * height)
            .map(|_| 1.0 + deviation * rng.normal())
            .collect();
        let image = DepthImage {
            depth: &depth,
            width,
            intrinsics: k,
            camera,
        };
        let points = image.points(None).unwrap();
        assert!(
            deviation > 0.0 || points.iter().flatten().all(|p| p.z == 1.0),
            "the wall is read exactly"
        );
        for device in devices() {
            let on = device.name();
            let mask = image.robot_mask(&robot, &q, padding, &[]).unwrap();
            let mask = mask
                .with_shapes(device, &image, std::slice::from_ref(&half), margin)
                .unwrap();
            let flagged = mask.on_robot.iter().filter(|&&f| f).count();
            assert!(
                flagged > 1_000 && flagged < 2_000,
                "{flagged} flagged on {on}"
            );
            // The unflagged half's readings within the margin of the wall's plane.
            let level: Vec<usize> = (0..depth.len())
                .filter(|&i| !mask.on_robot[i])
                .filter(|&i| points[i].is_some_and(|p| (p.z - 1.0).abs() <= margin[0]))
                .collect();
            for &pixels in bands {
                let nearer = (level.iter())
                    .filter(|&&i| {
                        let (u, v) = (i % width, i / width);
                        let rows = v.saturating_sub(pixels)..(v + pixels + 1).min(height);
                        let mut window = rows.flat_map(|w| {
                            (u.saturating_sub(pixels)..(u + pixels + 1).min(width))
                                .map(move |x| w * width + x)
                        });
                        let slack = margin[0].max(padding);
                        window.any(|j| mask.on_robot[j] && depth[j] - slack < depth[i] - margin[0])
                    })
                    .count();
                let banded = (mask.clone())
                    .with_band(device, &image, margin, pixels)
                    .unwrap();
                let taken = level.iter().filter(|&&i| banded.on_robot[i]).count();
                assert!(
                    nearer > 10 && taken == 0,
                    "a band of {pixels} at {margin:?} on {on}: of {} level readings, {nearer} \
                     pass the depths, {taken} taken",
                    level.len()
                );
            }
        }
    }
}

/// A band about what a mask flags leaves out what a camera reads behind a held shape's outline,
/// and keeps what stands in front of one. A block on a table, both given as shapes, its outline
/// read away from its true depth by more than the margin: on one side halfway to what lies behind
/// it, on the other 2 cm nearer. Fused without a band, the readings behind fill voxels in mid-air
/// beside the block; with a band of one pixel or five none do, and the readings in front stay,
/// which is what the band gives up. A second block and a slat one pixel wide, which no shape explains and
/// which stand on the table, keep every reading and every voxel: they are in front of the table.
#[test]
fn a_band_leaves_out_what_is_read_behind_a_held_shape() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let ([table, block, _], camera, k) = held_shapes();
    let margin = [0.004, 0.0, 0.0];
    let second = Obstacle::cuboid(Vec3::new(2.5, 0.09, 0.03), Vec3::splat(0.03));
    // A slat 1.6 mm thick on its edge, 16 mm high and 5 cm long, before the block and along the
    // camera's view, under the middle of a column of pixels.
    let slat = Obstacle::cuboid(
        Vec3::new(2.7, 0.0009, 0.008),
        Vec3::new(0.025, 0.0008, 0.008),
    );
    let width = 640;
    let mut depth = look_at_shapes(&[&table, &block, &second, &slat], camera, k);
    let behind = look_at_shapes(&[&table, &second, &slat], camera, k);
    let image = DepthImage {
        depth: &depth,
        width,
        intrinsics: k,
        camera,
    };
    let [ours, seconds, slats] = [&block, &second, &slat].map(|shape| shape_depth(&image, shape));
    let its =
        |of: &[f32]| -> Vec<usize> { (0..depth.len()).filter(|&i| of[i] == depth[i]).collect() };
    let (block_pixels, second_pixels, slat_pixels) = (its(&ours), its(&seconds), its(&slats));
    // The block's readings with a pixel beside them that is not the block's.
    let outline: Vec<usize> = (block_pixels.iter().copied())
        .filter(|&i| {
            let (u, v) = (i % width, i / width);
            let mut around =
                (v - 1..=v + 1).flat_map(|v| (u - 1..=u + 1).map(move |u| v * width + u));
            around.any(|j| ours[j] != depth[j])
        })
        .collect();
    let apart = (second_pixels.iter())
        .flat_map(|&i| outline.iter().map(move |&j| pixels_apart(i, j, width)))
        .min();
    assert_eq!(
        apart,
        Some(3),
        "the second block stands three pixels from the outline"
    );
    assert!(block_pixels.len() > 3_000 && outline.len() > 200 && second_pixels.len() > 1_000);
    // The slat is one pixel wide: no row holds two of its pixels.
    let mut rows = vec![0; depth.len() / width];
    for &i in &slat_pixels {
        rows[i / width] += 1;
    }
    let long = rows.iter().filter(|&&n| n > 0).count();
    assert!(
        rows.iter().all(|&n| n <= 1) && long > 20,
        "the slat is read one pixel wide in {long} rows"
    );
    // The outline read away from the block: on the side toward the second block halfway to what
    // lies behind it, on the other 2 cm nearer the camera.
    let middle = block_pixels.iter().map(|&i| i % width).sum::<usize>() / block_pixels.len();
    let (deeper, nearer): (Vec<usize>, Vec<usize>) =
        outline.iter().partition(|&&i| i % width >= middle);
    for &i in &deeper {
        assert!(behind[i] > depth[i], "pixel {i} has something behind it");
        depth[i] = 0.5 * (depth[i] + behind[i]);
    }
    for &i in &nearer {
        depth[i] -= 0.02;
    }
    let image = DepthImage {
        depth: &depth,
        width,
        intrinsics: k,
        camera,
    };
    let points = image.points(None).unwrap();
    let held = [table.clone(), block.clone()];
    let beyond = |moved: &[usize]| -> Vec<usize> {
        let far = |&i: &usize| {
            let p = points[i].expect("an outline pixel has a reading");
            held.iter().all(|s| s.distance(p).value > margin[0])
        };
        moved.iter().copied().filter(far).collect()
    };
    let (deeper, nearer) = (beyond(&deeper), beyond(&nearer));
    assert!(
        deeper.len() > 40 && nearer.len() > 100,
        "{} readings moved deeper and {} nearer lie farther than the margin from both shapes, of {}",
        deeper.len(),
        nearer.len(),
        outline.len()
    );
    // A voxel reaches into a shape where its middle lies within half its diagonal of it.
    let reach = 0.5 * 3f32.sqrt() * 0.01;
    let reaches = |p: Vec3, shape: &Obstacle| shape.distance(p).value <= reach;
    for device in devices() {
        let on = device.name();
        let mask = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
        assert!(!mask.on_robot.contains(&true), "the arm is elsewhere");
        let mask = mask.with_shapes(device, &image, &held, margin).unwrap();
        let moved = deeper.iter().chain(&nearer);
        assert!(
            moved.clone().all(|&i| !mask.on_robot[i]),
            "unflagged before the band"
        );
        // A mask's map: its voxels in mid-air, the second block's and the slat's, and whether
        // every reading moved nearer fills its voxel.
        let fused = |mask: &RobotMask| {
            let mut map =
                OccupancyMap::try_new(Vec3::new(2.0, -0.5, -0.1), Vec3::new(3.0, 0.5, 0.4), 0.01)
                    .unwrap();
            map.integrate(device, &image, Some(mask)).unwrap();
            let occupied = map.occupied();
            let count = |shape: &Obstacle| occupied.iter().filter(|&&p| reaches(p, shape)).count();
            // The voxel a reading moved nearer falls in.
            let holds_nearer = |v: Vec3| {
                let mut moved = nearer.iter().filter_map(|&i| points[i]);
                moved.any(|p| (v - p).abs().max_element() <= 0.005)
            };
            let shapes = [&table, &block, &second, &slat];
            let mid_air = (occupied.iter())
                .filter(|&&p| !shapes.iter().any(|s| reaches(p, s)) && !holds_nearer(p))
                .count();
            let filled =
                |&i: &usize| points[i].is_some_and(|p| map.occupancy(p).is_some_and(|o| o > 0.5));
            (
                mid_air,
                count(&second),
                count(&slat),
                nearer.iter().all(filled),
            )
        };
        let (mid_air, second_without, slat_without, filled) = fused(&mask);
        assert!(
            mid_air > 8 && second_without > 80 && slat_without > 5 && filled,
            "without a band on {on}: {mid_air} voxels in mid-air, {second_without} of the \
             second block's, {slat_without} of the slat's"
        );
        for pixels in [1, 5] {
            let banded = (mask.clone())
                .with_band(device, &image, margin, pixels)
                .unwrap();
            let taken = |pixels: &[usize]| {
                let by_band = |&&i: &&usize| banded.on_robot[i] && !mask.on_robot[i];
                pixels.iter().filter(by_band).count()
            };
            let (mid_air, second_with, slat_with, filled) = fused(&banded);
            assert!(
                deeper.iter().all(|&i| banded.on_robot[i]) && mid_air == 0,
                "a band of {pixels} on {on} takes the readings behind the block and leaves \
                 {mid_air} voxels in mid-air"
            );
            // The second block and the slat stand in front of the table, which is a shape: the
            // ray through each of their readings goes on into it, and the table takes none of
            // them. The second block stands three pixels from the first's outline and behind
            // its near face, on rays that miss it: within a band wider than that, its readings
            // are the ring behind that outline the band gives up, and no others.
            let ring = |i: &&usize| {
                let mut first = block_pixels.iter().filter(|&&j| mask.on_robot[j]);
                first.any(|&j| pixels_apart(**i, j, width) <= pixels)
            };
            let in_ring = second_pixels.iter().filter(ring).count();
            let (second_taken, slat_taken) = (taken(&second_pixels), taken(&slat_pixels));
            let outside = (second_pixels.iter())
                .filter(|&&i| banded.on_robot[i] && !mask.on_robot[i])
                .filter(|i| !ring(i))
                .count();
            let kept = second_taken == 0 && second_with == second_without;
            assert!(
                outside == 0
                    && (pixels > 3 || (in_ring == 0 && kept))
                    && slat_taken == 0
                    && slat_with == slat_without,
                "a band of {pixels} on {on} takes {second_taken} of the second block's readings, \
                 {outside} of them outside the {in_ring} behind the first's outline, and \
                 {slat_taken} of the slat's; their voxels without and with it: \
                 {second_without} and {second_with}, {slat_without} and {slat_with}"
            );
            // What the band gives up: an outline read in front of the block stays a reading.
            assert!(
                !nearer.iter().any(|&i| banded.on_robot[i]) && filled,
                "the readings before the block stay and fill their voxels, a band of {pixels} on \
                 {on}"
            );
        }
    }
}

/// A camera at `eye` looking at `target`, in OpenCV axes (x right, y down, z forward).
fn look_at(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
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
        spheres.iter().any(|&(c, r)| raw.distance(c).value < r),
        "the arm should read as an obstacle"
    );
    // Masked, its space is free while the box in front of it stays.
    let mask = image.robot_mask(&robot, &q, 0.02, &[]).unwrap();
    let arm = arm_depth(&image, &robot, &q, 0.0);
    let hidden = (0..depth.len())
        .filter(|&i| depth[i] > 0.0 && depth[i] < arm[i] - 0.1)
        .count();
    assert!(
        hidden > 1000,
        "the box should hide part of the arm: {hidden} pixels"
    );
    let masked = SdfGrid::from_depth(device, &image, Some(&mask), Occlusion::Free, &o).unwrap();
    for &(c, r) in &spheres {
        assert!(
            masked.distance(c).value > r,
            "an obstacle at the arm: {} at {c}",
            masked.distance(c).value
        );
    }
    assert!(
        masked.distance(face).value <= 0.0,
        "the box in front of the arm is lost: {}",
        masked.distance(face).value
    );
    // A fused map clears space in front of the arm and adds nothing at it.
    let mut map =
        OccupancyMap::try_new(Vec3::new(-0.2, -0.6, 0.0), Vec3::new(1.2, 0.6, 1.2), 0.02).unwrap();
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
    let mask = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
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
        .find(|&p| mask.on_robot[pixel(p)])
        .expect("a point only the arm hides");
    // And a point as far before the arm, along a pixel that reads the arm: seen past, so free.
    let before = spheres
        .iter()
        .map(|&(c, r)| c - (c - camera.position).normalize() * (r + 0.06))
        .filter(|&p| spheres.iter().all(|&(c, r)| p.distance(c) > r + 0.06))
        .find(|&p| mask.on_robot[pixel(p)])
        .expect("a point before the arm");
    let mut map = OccupancyMap::try_new(
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
                grid.distance(c).value > r,
                "{name}: an obstacle at the arm: {} at {c}",
                grid.distance(c).value
            );
        }
        assert!(
            grid.distance(shadow).value <= 0.0,
            "{name}: what the arm hides at {shadow} is free"
        );
        assert!(
            grid.distance(before).value > 0.0,
            "{name}: the space before the arm at {before} is solid"
        );
    }
    assert!(
        map.occupancy(before).is_some_and(|o| o < 0.5),
        "the map has not seen past {before}, before the arm"
    );
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
            grid.distance(shadow).value > 0.0,
            "{name}: what only the arm hides at {shadow} is solid"
        );
        assert!(
            grid.distance(behind_box).value <= 0.0,
            "{name}: what the box hides at {behind_box} is free"
        );
        for &(c, r) in &spheres {
            assert!(
                grid.distance(c).value > r,
                "{name}: an obstacle at the arm: {} at {c}",
                grid.distance(c).value
            );
        }
    }
    // A surface 1.5 cm in front of the grown arm, no reading of it the robot's: its band reaches
    // into the arm, which still reads free.
    let grown = arm_depth(&image, &robot, &q, 0.04);
    let front: Vec<f32> = depth
        .iter()
        .zip(&grown)
        .map(|(&d, &m)| if m.is_finite() { d.min(m - 0.015) } else { d })
        .collect();
    let image = DepthImage {
        depth: &front,
        ..image
    };
    let mask = image.robot_mask(&robot, &q, 0.04, &[]).unwrap();
    assert!(
        !mask.on_robot.iter().any(|&on| on),
        "a surface before the grown arm is not the robot's"
    );
    let band = Occlusion::Band(0.04);
    let unmasked = SdfGrid::from_depth(device, &image, None, band, &o).unwrap();
    assert!(
        spheres.iter().any(|&(c, r)| unmasked.distance(c).value < r),
        "the band should reach into the arm"
    );
    let grid = SdfGrid::from_depth(device, &image, Some(&mask), band, &o).unwrap();
    for &(c, r) in &spheres {
        assert!(
            grid.distance(c).value > r,
            "under a band, an obstacle at the arm: {} at {c}",
            grid.distance(c).value
        );
    }
}

/// A pixel has no reading where its depth is zero, negative or not finite: its point is none, and
/// grids and maps are those of the image with zeros there, on every device.
#[test]
fn a_depth_that_is_zero_negative_or_not_finite_is_no_reading() {
    let scene = TableScene::new();
    let vga = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let rendered = scene.render(640, 480, vga, true);
    let none = [0.0, -0.0, -0.3, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    // Every seventh pixel without a reading, each way in turn.
    let (mut zeroed, mut odd) = (rendered.clone(), rendered.clone());
    for (k, i) in (0..rendered.len()).step_by(7).enumerate() {
        (zeroed[i], odd[i]) = (0.0, none[k % none.len()]);
    }
    let (zeroed, odd) = (scene.image(&zeroed, 640, vga), scene.image(&odd, 640, vga));
    let points = odd.points(None).unwrap();
    for (i, point) in points.iter().enumerate() {
        assert_eq!(
            point.is_none(),
            i % 7 == 0,
            "pixel {i} reads {}",
            odd.depth[i]
        );
    }
    assert!(points == zeroed.points(None).unwrap());
    let robot = common::panda().unwrap();
    let mask = |image: &DepthImage| image.robot_mask(&robot, robot.default_q(), 0.02, &[]);
    assert_eq!(
        mask(&odd).unwrap().on_robot,
        mask(&zeroed).unwrap().on_robot
    );
    let o = SdfOptions {
        voxel: 0.02,
        padding: 0.15,
    };
    for device in devices() {
        let seen = |image: &DepthImage| {
            let grid = SdfGrid::from_depth(device, image, None, Occlusion::Occupied, &o).unwrap();
            let mut map =
                OccupancyMap::try_new(Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3), 0.02)
                    .unwrap();
            map.integrate(device, image, None).unwrap();
            (grid, map.grid(device, Occlusion::Free).unwrap())
        };
        assert!(seen(&odd) == seen(&zeroed), "{}", device.name());
    }
}

/// Per pixel of `image`, the depth at which its ray meets `shape`, by sphere tracing; infinite
/// where it meets none within 3 m.
fn shape_depth(image: &DepthImage, shape: &Obstacle) -> Vec<f32> {
    let k = image.intrinsics;
    (0..image.depth.len())
        .map(|i| {
            let (u, v) = ((i % image.width) as f32, (i / image.width) as f32);
            let ray = Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
            let dir = image.camera.rotation * ray.normalize();
            let mut t = 0.0;
            while t < 3.0 {
                let d = shape.distance(image.camera.position + dir * t).value;
                if d < 1e-4 {
                    return t * ray.normalize().z;
                }
                t += d;
            }
            f32::INFINITY
        })
        .collect()
}

/// A camera on the Panda's palm, 6 cm aside of the fingers and looking along them, stands inside
/// the hand's spheres once they are grown by the 4 cm a mask is padded with. A reading is the
/// robot's where its point lies in a grown sphere, not where its ray enters one: the camera keeps
/// the floor it sees past the hand, and maps and grids hold that floor.
#[test]
fn a_camera_the_hand_carries_keeps_what_it_sees_past_the_hand() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let hand = robot.link_pose(&q, "panda_hand").unwrap();
    let camera = hand.mul_pose(Pose::new(Vec3::new(0.06, 0.0, 0.03), Quat::IDENTITY));
    let padding = 0.04;
    let spheres = robot.spheres(&q);
    let around = |p: Vec3, pad: f32| {
        (spheres.iter()).any(|&[x, y, z, r]| p.distance(Vec3::new(x, y, z)) < r + pad)
    };
    assert!(
        around(camera.position, padding),
        "the camera should stand inside a grown sphere"
    );
    // A floor square on to the camera, 30 cm ahead, with the arm before it.
    let k = Intrinsics {
        fx: 320.0,
        fy: 320.0,
        cx: 319.5,
        cy: 239.5,
    };
    let floor = vec![0.3; 640 * 480];
    let image = DepthImage {
        depth: &floor,
        width: 640,
        intrinsics: k,
        camera,
    };
    let arm = arm_depth(&image, &robot, &q, 0.0);
    let depth = with_arm(&image, &robot, &q);
    let image = DepthImage {
        depth: &depth,
        ..image
    };
    let mask = image.robot_mask(&robot, &q, padding, &[]).unwrap();
    let points = image.points(None).unwrap();
    let (mut of_arm, mut of_floor, mut lost) = (0, 0, 0);
    for i in 0..depth.len() {
        let p = points[i].expect("every pixel has a reading");
        if arm[i] < 0.3 {
            of_arm += 1;
            assert!(mask.on_robot[i], "pixel {i} reads the arm at {p}");
        } else {
            of_floor += 1;
            lost += usize::from(mask.on_robot[i]);
            assert_eq!(mask.on_robot[i], around(p, padding), "pixel {i} at {p}");
        }
    }
    assert!(
        of_arm > 5_000,
        "the camera sees its fingers: {of_arm} pixels"
    );
    assert!(
        lost * 20 < of_floor,
        "{lost} of the floor's {of_floor} readings taken for the robot"
    );
    // Floor points the camera sees, farther from the robot than a grid grows.
    let seen: Vec<Vec3> = (0..depth.len())
        .step_by(997)
        .filter(|&i| arm[i] >= 0.3 && !around(points[i].unwrap(), 0.1))
        .map(|i| points[i].unwrap())
        .collect();
    assert!(seen.len() > 100, "{} floor points", seen.len());
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.15,
    };
    for device in devices() {
        let (lo, hi) = (camera.position - 0.5, camera.position + 0.5);
        let mut map = OccupancyMap::try_new(lo, hi, 0.01).unwrap();
        map.integrate(device, &image, Some(&mask)).unwrap();
        let grid = SdfGrid::from_depth(device, &image, Some(&mask), Occlusion::Free, &o).unwrap();
        for &p in &seen {
            assert!(
                map.occupancy(p).is_some_and(|o| o > 0.5),
                "{}: the map lost the floor at {p}",
                device.name()
            );
            let d = grid.distance(p).value;
            assert!(d <= 0.0, "{}: {d} on the floor at {p}", device.name());
        }
        for &[x, y, z, r] in &spheres {
            let c = Vec3::new(x, y, z);
            assert!(map.occupancy(c).is_none_or(|o| o < 0.5), "the hand at {c}");
            let d = grid.distance(c).value;
            assert!(d > r, "{}: {d} at the robot's sphere at {c}", device.name());
        }
    }
}

/// What a hand holds is left out as the robot is: its readings are no obstacle and the space it
/// fills is free, where without it named the box is an obstacle in the hand.
#[test]
fn a_mask_leaves_out_what_a_hand_holds() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    // A box 6 by 6 by 16 cm, the hand's frame 4 cm over its middle.
    let held = Held {
        touch_links: vec![],
        also: vec![],
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.08)),
        grasp: Pose::new(Vec3::new(0.0, 0.0, -0.04), Quat::IDENTITY),
        locked: Default::default(),
        spheres: SphereOptions::default(),
    };
    let pose = robot.ee_pose(&q, 0).mul_pose(held.grasp.inverse());
    let object = held.object.placed(pose);
    let camera = look_at(pose.position + Vec3::new(0.7, 0.5, 0.3), pose.position);
    // An image whose pixels do not fill whole words of a mask's bits.
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 314.5,
        cy: 234.5,
    };
    let far = vec![0.0; 630 * 470];
    let image = DepthImage {
        depth: &far,
        width: 630,
        intrinsics: k,
        camera,
    };
    let (arm, boxed) = (
        arm_depth(&image, &robot, &q, 0.0),
        shape_depth(&image, &object),
    );
    // A wall 2 m from the camera behind both.
    let depth: Vec<f32> = (arm.iter().zip(&boxed))
        .map(|(a, b)| a.min(*b).min(2.0))
        .collect();
    let image = DepthImage {
        depth: &depth,
        ..image
    };
    let padding = 0.02;
    let bare = image.robot_mask(&robot, &q, padding, &[]).unwrap();
    let holding = image
        .robot_mask(&robot, &q, padding, std::slice::from_ref(&held))
        .unwrap();
    let points = image.points(Some(&holding)).unwrap();
    let spheres = robot.spheres(&q);
    let by_robot = |p: Vec3| {
        (spheres.iter()).any(|&[x, y, z, r]| p.distance(Vec3::new(x, y, z)) < r + padding)
    };
    let all = image.points(None).unwrap();
    let (mut of_box, mut apart) = (0, vec![]);
    for i in 0..depth.len() {
        let p = all[i].unwrap();
        assert_eq!(points[i].is_none(), holding.on_robot[i], "pixel {i}");
        if boxed[i] < arm[i] {
            of_box += 1;
            assert!(holding.on_robot[i], "pixel {i} reads the held box at {p}");
            assert_eq!(bare.on_robot[i], by_robot(p), "pixel {i} at {p}");
            // The box's readings well clear of the hand's spheres.
            if !(spheres.iter()).any(|&[x, y, z, r]| p.distance(Vec3::new(x, y, z)) < r + 0.06) {
                apart.push(p);
            }
        } else if arm[i] > 2.0 {
            assert!(!holding.on_robot[i], "pixel {i} reads the wall at {p}");
        }
    }
    assert!(of_box > 2_000, "the camera sees the box: {of_box} pixels");
    assert!(apart.len() > 500, "{} readings of the box", apart.len());
    let o = SdfOptions {
        voxel: 0.01,
        padding: 0.15,
    };
    for device in devices() {
        let grids = [(&bare, true), (&holding, false)].map(|(mask, solid)| {
            let mut map =
                OccupancyMap::try_new(pose.position - 0.4, pose.position + 0.4, 0.01).unwrap();
            map.integrate(device, &image, Some(mask)).unwrap();
            let grid =
                SdfGrid::from_depth(device, &image, Some(mask), Occlusion::Occupied, &o).unwrap();
            (map, grid, solid)
        });
        for (map, grid, solid) in grids {
            for &p in apart.iter().step_by(7) {
                let (seen, d) = (map.occupancy(p), grid.distance(p).value);
                assert_eq!(
                    seen.is_some_and(|o| o > 0.5),
                    solid,
                    "{}: the map at {p} on the box, the box named: {}",
                    device.name(),
                    !solid
                );
                assert_eq!(d <= 0.0, solid, "{}: {d} at {p} on the box", device.name());
            }
            // Hidden space counts as solid, and the box's middle is hidden by its own faces.
            let middle = grid.distance(pose.position - pose.rotation * Vec3::Z * 0.05);
            assert_eq!(middle.value <= 0.0, solid, "{} in the box", middle.value);
        }
    }
    // A robot that holds the object already, and a hand the robot lacks, are refused.
    let attached = robot.attach(&held.attachment(&robot).unwrap()).unwrap();
    assert!(matches!(
        image.robot_mask(&attached, &q, padding, std::slice::from_ref(&held)),
        Err(Error::Input(_))
    ));
    let other = Held { hand: 1, ..held };
    assert!(matches!(
        image.robot_mask(&robot, &q, padding, &[other]),
        Err(Error::Input(_))
    ));
}

/// A hold says where the fingers closed (`Held::locked`), and the mask takes the robot so: the
/// readings of fingers shut on a small object are the robot's, where the fingers as the robot
/// was loaded, open, leave most of them in as obstacles.
#[test]
fn a_mask_holds_the_fingers_where_a_hold_locks_them() {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let shut: HashMap<String, f32> = ["panda_finger_joint1", "panda_finger_joint2"]
        .map(|joint| (joint.to_string(), 0.0))
        .into();
    let closed = robot.relock(&shut).unwrap();
    let open = Held {
        touch_links: vec![],
        also: vec![],
        hand: 0,
        object: Obstacle::sphere(Vec3::ZERO, 0.005),
        grasp: Pose::IDENTITY,
        locked: Default::default(),
        spheres: SphereOptions::default(),
    };
    let held = Held {
        locked: shut,
        ..open.clone()
    };
    let hand = robot.ee_pose(&q, 0).position;
    let k = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let far = vec![2.0; 640 * 480];
    let image = DepthImage {
        depth: &far,
        width: 640,
        intrinsics: k,
        camera: look_at(hand + Vec3::new(0.5, 0.4, 0.1), hand),
    };
    // The arm with its fingers shut, before a wall; and where it would read with them open.
    let (seen, as_loaded) = (
        arm_depth(&image, &closed, &q, 0.0),
        arm_depth(&image, &robot, &q, 0.0),
    );
    let depth = with_arm(&image, &closed, &q);
    let image = DepthImage {
        depth: &depth,
        ..image
    };
    // Padded by less than the fingers move, so that where they stand shows.
    let padding = 0.002;
    let locked = image.robot_mask(&robot, &q, padding, &[held]).unwrap();
    let loaded = image.robot_mask(&robot, &q, padding, &[open]).unwrap();
    let (mut moved, mut left_in) = (0, 0);
    for i in 0..depth.len() {
        if seen[i].is_finite() {
            assert!(locked.on_robot[i], "pixel {i} reads the arm");
        }
        // A reading of a shut finger where an open one would read elsewhere, or nothing.
        if seen[i].is_finite() && (seen[i] - as_loaded[i]).abs() > 0.01 {
            moved += 1;
            left_in += usize::from(!loaded.on_robot[i]);
        }
    }
    assert!(moved > 300, "the camera sees the fingers: {moved} pixels");
    assert!(
        left_in * 2 > moved,
        "{left_in} of {moved} readings of the shut fingers left in with the fingers open"
    );
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
    let mask = image
        .robot_mask(&robot, robot.default_q(), 0.02, &[])
        .unwrap();
    let mut map =
        OccupancyMap::try_new(Vec3::new(-0.3, -0.3, -0.1), Vec3::new(0.3, 0.3, 0.3), 0.01).unwrap();
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
            (
                "map, unknown solid in a band",
                map.grid(d, Occlusion::Band(0.04)).unwrap(),
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

/// The shapes [`points_lie_on_the_nearest_shape_within_their_reach`] asks of: a table, on it a
/// turned box, a cylinder, a sphere, a capsule and a bar as a distance grid, the turned box once
/// more, and a wall far from all of them.
fn shapes_to_lie_on() -> Vec<Obstacle> {
    let turn = Quat::from_euler(EulerRot::XYZ, 0.3, -0.4, 0.9);
    let turned = Obstacle::Cuboid {
        center: Vec3::new(0.3, -0.2, 0.05),
        half_extents: Vec3::new(0.05, 0.08, 0.03),
        rotation: turn,
    };
    let (dims, voxel, origin) = ([14u32, 9, 8], 0.02, Vec3::new(-0.13, -0.08, -0.07));
    let bar = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.08, 0.03, 0.02));
    let mut distances = vec![];
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let p = origin + Vec3::new(i as f32, j as f32, k as f32) * voxel;
                distances.push(bar.distance(p).value);
            }
        }
    }
    vec![
        Obstacle::cuboid(Vec3::new(0.5, 0.0, -0.02), Vec3::new(0.6, 0.6, 0.02)),
        turned.clone(),
        Obstacle::cylinder(Vec3::new(0.6, 0.2, 0.06), 0.04, 0.06),
        Obstacle::sphere(Vec3::new(0.8, -0.3, 0.05), 0.05),
        Obstacle::Capsule {
            center: Vec3::new(0.2, 0.3, 0.1),
            rotation: turn,
            radius: 0.02,
            half_length: 0.07,
        },
        Obstacle::Sdf {
            grid: Arc::new(SdfGrid::try_new(dims, voxel, origin, &distances).unwrap()),
            center: Vec3::new(0.7, 0.0, 0.04),
            rotation: turn,
        },
        turned,
        Obstacle::cuboid(Vec3::new(3.0, 0.0, 1.0), Vec3::new(0.01, 2.0, 1.0)),
    ]
}

/// Each point lies on the nearest of the shapes it is within its reach of, as the shapes' own
/// distances say, and of two as near on the first: points in rows over the shapes, as an image
/// reads them, and scattered about, on and inside them; reaches of nothing, of millimetres, of
/// meters and without bound, and points marked as having none, some of them inside a shape; and
/// points with a coordinate that is no number, which lie on none. The same on one thread, on
/// many, and whatever the points' order.
#[test]
fn points_lie_on_the_nearest_shape_within_their_reach() {
    let shapes = shapes_to_lie_on();
    let mut rng = Rng::new(11);
    let mut points = vec![];
    // Rows 4 mm apart at heights that cut through the shapes, each row a run of its own and more.
    for z in [-0.021, 0.0, 0.001, 0.03, 0.058, 0.12] {
        for row in 0..120 {
            for column in 0..300 {
                points.push(Vec3::new(
                    -0.1 + 0.004 * column as f32,
                    -0.24 + 0.004 * row as f32,
                    z,
                ));
            }
        }
    }
    // Scattered through the room, the far wall's too, so that a run's points lie far apart.
    for _ in 0..20_000 {
        points.push(Vec3::new(
            rng.range(-0.3, 3.2),
            rng.range(-1.0, 1.0),
            rng.range(-0.1, 2.1),
        ));
    }
    let reaches = [0.0, 0.002, 0.01, 0.3, f32::INFINITY, -0.001, f32::NAN];
    let mut reach: Vec<f32> = (0..points.len())
        .map(|i| reaches[(i + i / 300) % reaches.len()])
        .collect();
    // Points among the rows with a coordinate that is no number and a reach: the first of a run
    // (runs are 1,024 points), the middle of another and the last of a third.
    let unnumbered = [3 * 1024, 40 * 1024 + 512, 101 * 1024 - 1];
    for (axis, &i) in unnumbered.iter().enumerate() {
        points[i][axis] = f32::NAN;
        reach[i] = 0.01;
    }
    // What the shapes' own distances say.
    let expected = |points: &[Vec3], reach: &[f32]| -> Vec<Option<(usize, f32)>> {
        let each = points.iter().zip(reach).map(|(&p, &reach)| {
            let distances = shapes.iter().map(|shape| shape.distance(p).value);
            let nearest = distances.enumerate().min_by(|a, b| a.1.total_cmp(&b.1));
            nearest.filter(|&(_, distance)| reach >= 0.0 && distance <= reach)
        });
        each.collect()
    };
    let found = |device: &Device, points: &[Vec3], reach: &[f32]| -> Vec<Option<(usize, f32)>> {
        let on = sdf::on_shapes(device, &shapes, points, reach).unwrap();
        on.iter()
            .map(|on| on.map(|on| (on.shape, on.distance)))
            .collect()
    };
    let wanted = expected(&points, &reach);
    assert_eq!(found(cpu(), &points, &reach), wanted);
    // With no reach longer than a centimetre, a run of points comes within reach of few shapes'
    // boxes, and its other shapes are left out for all of it.
    let short: Vec<f32> = reach.iter().map(|reach| reach.min(0.01)).collect();
    let wanted_short = expected(&points, &short);
    assert_eq!(found(cpu(), &points, &short), wanted_short);
    assert!(wanted_short.iter().flatten().count() > 10_000);
    let alone = Device::cpu_threads(&common::panda().unwrap(), 1).unwrap();
    assert_eq!(found(&alone, &points, &reach), wanted, "on one thread");
    // The points in another order: each lies on what it did.
    let mut order: Vec<usize> = (0..points.len()).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
    let shuffled: Vec<Vec3> = order.iter().map(|&i| points[i]).collect();
    let shuffled_reach: Vec<f32> = order.iter().map(|&i| reach[i]).collect();
    let again = found(cpu(), &shuffled, &shuffled_reach);
    for (k, &i) in order.iter().enumerate() {
        assert_eq!(again[k], wanted[i], "point {i} as number {k}");
    }

    // The points ask all of it: every shape but the turned box given again has points on it,
    // which a point never lies on, the first of the two being as near; some lie inside a shape
    // and some on none; a point marked as having no reach lies inside a shape; and a point lies
    // on a shape that it reaches only by its reach, outside the shape's box.
    let on = |shape: usize| wanted.iter().flatten().filter(|on| on.0 == shape).count();
    for shape in [0, 1, 2, 3, 4, 5, 7] {
        assert!(on(shape) > 20, "{} points on shape {shape}", on(shape));
    }
    assert_eq!(on(6), 0);
    assert!(wanted.iter().flatten().filter(|on| on.1 < -0.005).count() > 100);
    assert!(wanted.iter().filter(|on| on.is_none()).count() > 1000);
    let inside = |p: Vec3| shapes.iter().any(|shape| shape.distance(p).value < -0.002);
    let marked = points
        .iter()
        .zip(&reach)
        .filter(|&(&p, &reach)| (reach.is_nan() || reach < 0.0) && inside(p));
    assert!(marked.count() > 100);
    let table_top = |p: Vec3| p.z > 0.002 && p.z < 0.2 && (p.x - 0.5).abs() < 0.5;
    let by_reach = points
        .iter()
        .zip(&wanted)
        .filter(|&(&p, on)| table_top(p) && on.is_some_and(|on| on.0 == 0 && on.1 > 0.002));
    assert!(by_reach.count() > 100);
    // A point with a coordinate that is no number lies on none, and most of its run on shapes:
    // a run's box taken over it would leave them on none.
    for i in unnumbered {
        assert!(points[i].is_nan() && reach[i] >= 0.0 && wanted[i].is_none());
        let run = &wanted[i / 1024 * 1024..][..1024];
        let on = run.iter().flatten().count();
        assert!(on > 500, "{on} points on shapes in the run of point {i}");
    }
}

#[test]
fn points_with_no_reach_each_or_a_shape_that_is_none_are_refused() {
    let shapes = shapes_to_lie_on();
    let points = [Vec3::ZERO; 3];
    let refused =
        |shapes: &[Obstacle], reach: &[f32]| match sdf::on_shapes(cpu(), shapes, &points, reach) {
            Err(Error::Input(message)) => message,
            other => panic!("not refused as input: {other:?}"),
        };
    assert!(refused(&shapes, &[0.01; 2]).contains("3 points with 2 reaches"));
    let none = Obstacle::sphere(Vec3::ZERO, -1.0);
    assert!(refused(&[shapes[0].clone(), none], &[0.01; 3]).contains("shape 1"));
    // No shapes: every point lies on none.
    let on = sdf::on_shapes(cpu(), &[], &points, &[0.01; 3]).unwrap();
    assert!(on.iter().all(Option::is_none));
}

#[test]
fn separating_grasp_readings_keeps_mixed_voxels_and_unknown_space() {
    let voxel = 0.01;
    let mut map = OccupancyMap::try_new(-Vec3::splat(0.06), Vec3::splat(0.06), voxel).unwrap();
    let own = -2.0 * voxel * Vec3::X;
    let mixed = 2.0 * voxel * Vec3::X;
    let foreign = mixed + 0.1 * voxel * Vec3::Y;
    for point in [own, mixed] {
        map.occupy(&Obstacle::sphere(point, 0.1 * voxel), 0.0)
            .unwrap();
    }
    assert_eq!(map.counts().occupied, 2);
    let unknown = map.counts().unobserved;
    let before = map.occupied();
    let detached = map
        .take_readings(&[own, mixed], &[foreign], 0.02, |_, _| true)
        .unwrap()
        .expect("one occupied voxel contains only a grasp reading");
    assert_eq!(detached.counts().occupied, 1);
    assert_eq!(map.counts().occupied, 1);
    assert_eq!(map.counts().unobserved, unknown);
    assert!(detached.occupancy(own).is_some_and(|p| p > 0.5));
    assert!(map.occupancy(own).is_some_and(|p| p < 0.5));
    assert!(map.occupancy(mixed).is_some_and(|p| p > 0.5));
    assert_eq!(map.occupancy(Vec3::ZERO), None);
    let mut together = map.occupied();
    together.extend(detached.occupied());
    for point in before {
        assert!(together.iter().any(|p| point.distance(*p) < 1e-6));
    }
    // Without fresh rays, refinement cannot open a hole in what remains.
    let refined = map.refined(cpu(), &[], 2).unwrap();
    assert_eq!(refined.counts().occupied, 8);
    assert_eq!(refined.counts().unobserved, 8 * unknown);
    assert!(refined.occupancy(mixed).is_some_and(|p| p > 0.5));
    assert_eq!(refined.occupancy(Vec3::ZERO), None);
}

#[test]
fn refining_a_map_remembers_what_hid_each_cell() {
    let robot = common::panda().unwrap();
    let ([table, block, _], camera, intrinsics) = held_shapes();
    let depth = look_at_shapes(&[&table, &block], camera, intrinsics);
    let image = DepthImage {
        depth: &depth,
        width: 640,
        intrinsics,
        camera,
    };
    let hidden = Vec3::new(2.4, 0.0, 0.02);
    let new = || {
        OccupancyMap::try_new(hidden - Vec3::splat(0.1), hidden + Vec3::splat(0.1), 0.01).unwrap()
    };
    let shapes = [table, block];
    for device in devices() {
        let mask = image
            .robot_mask(&robot, robot.default_q(), 0.04, &[])
            .unwrap()
            .with_shapes(device, &image, &shapes, [0.004, 0.0, 0.0])
            .unwrap();
        let distance = |map: &OccupancyMap| {
            map.grid(device, Occlusion::OccupiedExceptRobotShadow)
                .unwrap()
                .distance(hidden)
                .value
        };
        let mut behind_surface = new();
        behind_surface.integrate(device, &image, None).unwrap();
        assert!(behind_surface.occupancy(hidden).is_none() && distance(&behind_surface) <= 0.0);
        let shadowed_again = behind_surface
            .refined(device, &[(&image, Some(&mask))], 2)
            .unwrap();
        assert!(
            distance(&shadowed_again) <= 0.0,
            "a surface previously hid this space"
        );

        let mut behind_mask = new();
        behind_mask.integrate(device, &image, Some(&mask)).unwrap();
        // Keep a solid away from the shadow so a grid can be built even if every unknown is shadow.
        behind_mask
            .occupy(&Obstacle::sphere(hidden + 0.07 * Vec3::Y, 0.01), 0.0)
            .unwrap();
        assert!(behind_mask.occupancy(hidden).is_none() && distance(&behind_mask) > 0.0);
        let without_new_views = behind_mask.refined(device, &[], 2).unwrap();
        assert!(
            distance(&without_new_views) > 0.0,
            "only the mask ever hid this space"
        );
    }
}

#[test]
fn a_grid_with_a_thin_negative_pocket_cannot_be_voxelized_as_a_primitive() {
    let mut distances = [1.0; 27];
    distances[13] = -0.001;
    let grid = SdfGrid::try_new([3; 3], 0.01, Vec3::ZERO, &distances).unwrap();
    let shape = Obstacle::from_grid(grid);
    assert!(shape.distance(Vec3::splat(0.01)).value < 0.0);
    let options = SdfOptions {
        voxel: 0.02,
        padding: 0.0,
    };
    assert!(matches!(
        OccupancyMap::from_shape(&shape, &options),
        Err(Error::Input(_))
    ));
}

#[test]
fn a_pixel_owns_its_surface_cells_unless_another_view_also_hits_them() {
    let map = OccupancyMap::try_new(
        Vec3::new(-0.04, -0.02, 0.98),
        Vec3::new(0.04, 0.02, 1.02),
        0.01,
    )
    .unwrap();
    let front = DepthImage {
        depth: &[1.0],
        width: 1,
        intrinsics: Intrinsics {
            fx: 10.0,
            fy: 10.0,
            cx: 0.0,
            cy: 0.0,
        },
        camera: Pose::IDENTITY,
    };
    let beside = DepthImage {
        intrinsics: Intrinsics {
            fx: 45.0,
            ..front.intrinsics
        },
        camera: Pose::new(0.01 * Vec3::X, Quat::IDENTITY),
        ..front
    };
    let views = [(&front, None), (&beside, None)];
    let mut occupied = map.refined(cpu(), &views, 1).unwrap();
    let unknown = occupied.counts().unobserved;
    let all = occupied.counts().occupied;
    let selected = occupied
        .take_surfaces(cpu(), &views, |view, _| view == 0, 0.02, |_, _| true)
        .unwrap()
        .expect("the wide pixel sees either side of the other view");
    let alone = Vec3::new(-0.02, 0.0, 1.0);
    let shared = Vec3::Z;
    assert!(
        selected.occupancy(alone).is_some_and(|v| v > 0.5),
        "the selected pixel sees the surface here although its endpoint lies elsewhere"
    );
    assert!(
        occupied.occupancy(shared).is_some_and(|v| v > 0.5),
        "the second camera projects a foreign hit where the first camera's endpoint lands"
    );
    assert_eq!(selected.counts().occupied + occupied.counts().occupied, all);
    assert_eq!(occupied.counts().unobserved, unknown);
}

#[test]
fn a_foreign_depth_hit_at_a_pixel_boundary_stays_when_planning_on_cpu() {
    let a = 0.0213_f32;
    let rotation = Quat::from_euler(EulerRot::XYZ, a, a * 0.33, -0.45 * a);
    let p = rotation * Vec3::new(0.0625, 0.0, 1.0);
    let map = OccupancyMap::try_new(p, p + Vec3::splat(0.008), 0.002).unwrap();
    let foreign = DepthImage {
        depth: &[0.0, 1.0],
        width: 2,
        intrinsics: Intrinsics {
            fx: 8.0,
            fy: 8.0,
            cx: 0.0,
            cy: 0.0,
        },
        camera: Pose::new(Vec3::ZERO, rotation),
    };
    let both = DepthImage {
        depth: &[1.0, 1.0],
        ..foreign
    };
    for device in devices() {
        let mut reference = map.clone();
        reference.integrate(device, &foreign, None).unwrap();
        let hits = reference.occupied();
        assert!(
            !hits.is_empty(),
            "the foreign pixel supplied occupied cells"
        );
        let mut full = map.clone();
        full.integrate(device, &both, None).unwrap();
        full.take_surfaces(cpu(), &[(&both, None)], |_, px| px == 0, 0.002, |_, _| true)
            .unwrap();
        for hit in hits {
            assert!(
                full.occupancy(hit).is_some_and(|v| v > 0.5),
                "foreign hit {hit}"
            );
        }
    }
}

#[test]
fn a_selected_surface_includes_the_far_depth_boundary_used_by_fusion() {
    let edge = Vec3::new(0.02, 0.0, 1.001);
    let mut map = OccupancyMap::try_new(edge, edge + Vec3::splat(0.004), 0.002).unwrap();
    let image = DepthImage {
        depth: &[1.0],
        width: 1,
        intrinsics: Intrinsics {
            fx: 10.0,
            fy: 10.0,
            cx: 0.0,
            cy: 0.0,
        },
        camera: Pose::IDENTITY,
    };
    map.integrate(cpu(), &image, None).unwrap();
    assert!(map.occupancy(edge).is_some_and(|v| v > 0.5));
    let selected = map
        .take_surfaces(cpu(), &[(&image, None)], |_, _| true, 0.002, |_, _| true)
        .unwrap()
        .expect("the cell is hit away from the pixel's central endpoint");
    assert!(selected.occupancy(edge).is_some_and(|v| v > 0.5));
}

#[test]
fn a_fine_surface_hit_keeps_unresolved_occupancy_inherited_from_its_parent() {
    let p = Vec3::Z;
    let lower = Vec3::new(-0.02, -0.02, 0.98);
    let mut map = OccupancyMap::try_new(lower, Vec3::new(0.02, 0.02, 1.02), 0.01).unwrap();
    map.free(&Obstacle::cuboid(p, Vec3::splat(0.1)), 0.0)
        .unwrap();
    let old = DepthImage {
        depth: &[1.0],
        width: 1,
        intrinsics: Intrinsics {
            fx: 1200.0,
            fy: 1200.0,
            cx: 0.0,
            cy: 0.0,
        },
        camera: Pose::new(Vec3::new(0.004, 0.0, 0.0), Quat::IDENTITY),
    };
    map.integrate(cpu(), &old, None).unwrap();
    assert!(map.occupancy(p).unwrap() > 0.5);
    let own = DepthImage {
        depth: &[1.0],
        width: 1,
        intrinsics: Intrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 0.0,
            cy: 0.0,
        },
        camera: Pose::new(Vec3::new(0.007, 0.0, 0.0), Quat::IDENTITY),
    };
    let mut fresh = OccupancyMap::try_new(lower, Vec3::new(0.02, 0.02, 1.02), 0.01).unwrap();
    fresh.integrate(cpu(), &own, None).unwrap();
    assert!(
        !fresh.occupancy(p).is_some_and(|a| a > 0.5),
        "the target view cannot locate the old cell more finely"
    );
    let mut fine = map.refined(cpu(), &[(&own, None)], 5).unwrap();
    let q = p + 0.004 * Vec3::X;
    assert!(fine.occupancy(q).unwrap() > 0.5);
    let mut cropped = fine
        .part(&Obstacle::cuboid(q, Vec3::splat(0.012)), 0.002)
        .unwrap();
    for region in [&mut fine, &mut cropped] {
        region
            .take_surfaces(cpu(), &[(&own, None)], |_, _| true, 0.002, |_, _| true)
            .unwrap()
            .expect("the selected pixel also hits cells outside the old parent");
        assert!(
            region.occupancy(q).is_some_and(|v| v > 0.5),
            "the current pixel cannot identify the material inherited from the old parent"
        );
    }
    let replacement = Obstacle::sphere(q, 0.0005);
    cropped.free(&replacement, 0.0).unwrap();
    assert!(cropped.occupancy(q).is_some_and(|v| v < 0.5));
    cropped.occupy(&replacement, 0.0).unwrap();
    let part = cropped
        .take_surfaces(cpu(), &[(&own, None)], |_, _| true, 0.002, |_, _| true)
        .unwrap()
        .expect("the former cell now contains newly occupied material");
    assert!(part.occupancy(q).is_some_and(|v| v > 0.5));
}
