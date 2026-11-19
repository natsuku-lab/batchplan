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
    CollisionWeights, DepthImage, Device, Distance, Error, Held, Intrinsics, JointTrajectory,
    Obstacle, Occlusion, OccupancyMap, Pose, Robot, SdfGrid, SdfOptions, SphereOptions, World,
};
use glam::{BVec3, EulerRot, Mat3, Quat, Vec3};

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
