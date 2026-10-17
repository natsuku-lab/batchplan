//! Clearances measured on the robot's meshes: exact distances to every obstacle kind, following
//! the joints, against distance grids, between links, for held objects, and never farther than
//! the spheres planning uses.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::FRAC_PI_2;
use std::sync::{Arc, OnceLock};

use batchplan::rng::Rng;
use batchplan::*;
use glam::{Quat, Vec3};

/// Three box links in a horizontal plane at z = 0.5, each turning about z: `a` (x from 0 to 0.4,
/// 10 cm square), `b` (0.3 long, 8 cm square) and `c` (0.25 long, 8 cm square).
const CHAIN: &str = r#"<robot name="chain">
  <link name="base"/>
  <link name="a"><collision><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.1 0.1"/></geometry></collision></link>
  <link name="b"><collision><origin xyz="0.15 0 0"/><geometry><box size="0.3 0.08 0.08"/></geometry></collision></link>
  <link name="c"><collision><origin xyz="0.125 0 0"/><geometry><box size="0.25 0.08 0.08"/></geometry></collision></link>
  <joint name="ja" type="revolute"><parent link="base"/><child link="a"/><origin xyz="0 0 0.5"/>
    <axis xyz="0 0 1"/><limit lower="-3" upper="3" velocity="2" effort="1"/></joint>
  <joint name="jb" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0.4 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3" upper="3" velocity="2" effort="1"/></joint>
  <joint name="jc" type="revolute"><parent link="b"/><child link="c"/><origin xyz="0.3 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3" upper="3" velocity="2" effort="1"/></joint>
</robot>"#;

fn chain() -> Robot {
    static CHAIN_ROBOT: OnceLock<Robot> = OnceLock::new();
    CHAIN_ROBOT
        .get_or_init(|| {
            let dir = std::env::temp_dir().join(format!("batchplan-chain-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("chain.urdf"), CHAIN).unwrap();
            Robot::load(dir.join("chain.urdf"), &RobotOptions::default()).unwrap()
        })
        .clone()
}

fn world(obstacle: Obstacle) -> World {
    World { obstacles: vec![obstacle] }
}

/// A closed cube mesh of side `2 * half`, centred at the origin.
fn cube(half: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let vertices =
        (0..8).map(|i| Vec3::new([-half, half][i & 1], [-half, half][i >> 1 & 1], [-half, half][i >> 2 & 1])).collect();
    let faces = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
    let triangles = faces.iter().flat_map(|&[a, b, c, d]| [[a, b, c], [a, c, d]]).collect();
    (vertices, triangles)
}

#[test]
fn clearances_to_every_obstacle_kind_are_exact() {
    let robot = chain();
    let meshes = MeshModel::new(&robot).unwrap();
    // Each obstacle sits beside link `a`, centred 0.3 m along y, so 0.2 m from its side; links `b`
    // and `c` are farther. The cylinder and the capsule stand upright: lying along y, they would
    // reach into `a`.
    let at = Vec3::new(0.2, 0.3, 0.5);
    let upright = Quat::IDENTITY;
    let worlds = [
        world(Obstacle::Sphere { center: at, radius: 0.05 }),
        world(Obstacle::Cuboid { center: at, half_extents: Vec3::splat(0.05), rotation: upright }),
        world(Obstacle::Cylinder { center: at, rotation: upright, radius: 0.05, half_height: 0.3 }),
        world(Obstacle::Capsule { center: at, rotation: upright, radius: 0.05, half_length: 0.3 }),
    ];
    let straight = [0.0; 3];
    for (w, obstacle) in worlds.iter().enumerate() {
        let [world, _] = meshes.clearance(&worlds, &[w as u32], &straight).unwrap()[0];
        assert!((world - 0.2).abs() < 1e-4, "{:?}: {world}", obstacle.obstacles[0]);
    }
    // Turned a quarter turn, `a` runs along +y and passes the sphere 0.1 m away.
    let [turned, _] = meshes.clearance(&worlds, &[0], &[FRAC_PI_2, 0.0, 0.0]).unwrap()[0];
    assert!((turned - 0.1).abs() < 1e-4, "after a quarter turn: {turned}");
    // A box across `a`'s surface leaves nothing. One entirely inside `a` is 3 cm from its
    // surface: meshes are surfaces, as in FCL.
    let across = world(Obstacle::Cuboid {
        center: Vec3::new(0.2, 0.0, 0.5),
        half_extents: Vec3::splat(0.08),
        rotation: upright,
    });
    assert_eq!(meshes.clearance(&[across], &[0], &straight).unwrap()[0][0], 0.0);
    let inside = world(Obstacle::Cuboid {
        center: Vec3::new(0.2, 0.0, 0.5),
        half_extents: Vec3::splat(0.02),
        rotation: upright,
    });
    assert!((meshes.clearance(&[inside], &[0], &straight).unwrap()[0][0] - 0.03).abs() < 1e-4);
    // Obstacles listed far to near: the search must not stop at the first one it measures.
    let crowd = World {
        obstacles: [0.6, 0.45, 0.3].map(|y| Obstacle::Sphere { center: Vec3::new(0.2, y, 0.5), radius: 0.05 }).to_vec(),
    };
    let [nearest, _] = meshes.clearance(&[crowd], &[0], &straight).unwrap()[0];
    assert!((nearest - 0.2).abs() < 1e-4, "among several: {nearest}");
    // A ball 6 cm from link `c` (0.36 m from `a`), then one far away: measured link by link in
    // list order, `a` against the far ball would end the search before `c` is reached.
    let scattered = World {
        obstacles: vec![
            Obstacle::Sphere { center: Vec3::new(0.8, 0.15, 0.5), radius: 0.05 },
            Obstacle::Sphere { center: Vec3::new(-2.0, 0.0, 0.5), radius: 0.05 },
        ],
    };
    let [nearest, _] = meshes.clearance(&[scattered], &[0], &straight).unwrap()[0];
    assert!((nearest - 0.06).abs() < 1e-4, "beside `c`: {nearest}");
    // Nothing to measure against.
    assert_eq!(meshes.clearance(&[World::default()], &[0], &straight).unwrap()[0][0], 1e30);
}

#[test]
fn grids_are_checked_at_points_covering_the_surface() {
    let robot = chain();
    let meshes = MeshModel::new(&robot).unwrap();
    let (vertices, triangles) = cube(0.05);
    let grid = Arc::new(SdfGrid::from_mesh(&vertices, &triangles, &SdfOptions::default()).unwrap());
    let worlds =
        [world(Obstacle::Sdf { grid, center: Vec3::new(0.2, 0.3, 0.5), rotation: Quat::from_rotation_z(0.3) })];
    // The cube turned by 0.3 rad is at most 0.2 m from `a`, a little less at its corners. The grid
    // reads up to a voxel diagonal less, and the surface points 5 mm less again.
    let corner = 0.05 * (0.3f32.cos() + 0.3f32.sin()) - 0.05;
    let [world, _] = meshes.clearance(&worlds, &[0], &[0.0; 3]).unwrap()[0];
    let truth = 0.2 - corner;
    assert!(world <= truth && world >= truth - 0.0174 - 0.005, "{world} against a true {truth}");
}

#[test]
fn self_clearance_is_measured_between_link_meshes() {
    let robot = chain();
    let meshes = MeshModel::new(&robot).unwrap();
    let none = [World::default()];
    // Folded back twice by a quarter turn, `c` runs parallel to `a`, 0.3 m over: their sides are
    // 0.21 m apart.
    let [_, folded] = meshes.clearance(&none, &[0], &[0.0, FRAC_PI_2, FRAC_PI_2]).unwrap()[0];
    assert!((folded - 0.21).abs() < 1e-4, "folded: {folded}");
    // Folded further, `c` runs from (0.25, 0.26) down through `a` near (0.2, 0.015).
    let [_, crossing] = meshes.clearance(&none, &[0], &[0.0, 2.094, 2.43]).unwrap()[0];
    assert_eq!(crossing, 0.0);
}

#[test]
fn spheres_never_read_farther_than_the_meshes() {
    let robot = chain();
    let meshes = MeshModel::new(&robot).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let scene = vec![World {
        obstacles: vec![
            Obstacle::Cuboid {
                center: Vec3::new(0.3, 0.35, 0.5),
                half_extents: Vec3::splat(0.06),
                rotation: Quat::IDENTITY,
            },
            Obstacle::Cylinder {
                center: Vec3::new(-0.2, -0.3, 0.5),
                rotation: Quat::IDENTITY,
                radius: 0.05,
                half_height: 0.2,
            },
        ],
    }];
    let worlds = cpu.upload(&scene).unwrap();
    let mut rng = Rng::new(3);
    let q: Vec<f32> = (0..300 * 3).map(|_| rng.range(-3.0, 3.0)).collect();
    let items = vec![0; 300];
    // Sphere clearances are exact up to the margins.
    let exact = CollisionWeights { margin: 1.0, self_margin: 1.0, ..CollisionWeights::NONE };
    let spheres = cpu.evaluate(&worlds, &items, &q, &exact).unwrap();
    let on_meshes = meshes.clearance(&scene, &items, &q).unwrap();
    let (mut touching, mut apart) = (0, 0);
    for (i, [world, self_collision]) in on_meshes.into_iter().enumerate() {
        // Fitted spheres hold 20,000 surface samples per link, not every surface point.
        assert!(
            spheres.world_clearance[i] <= world + 1e-3,
            "{}: spheres {} meshes {world}",
            i,
            spheres.world_clearance[i]
        );
        assert!(
            spheres.self_clearance[i] <= self_collision + 1e-3,
            "{i}: spheres {} meshes {self_collision}",
            spheres.self_clearance[i]
        );
        touching += usize::from(world == 0.0 || self_collision == 0.0);
        apart += usize::from(world > 0.0 && self_collision > 0.0);
    }
    assert!(touching > 20 && apart > 20, "the configurations should span both: {touching} touching, {apart} apart");
}

#[test]
fn held_objects_and_real_robots_are_checked_as_meshes() {
    // A box held at the end of `c`, which reaches 0.15 m past it.
    let robot = chain();
    let held = robot
        .attach(&AttachedObject {
            name: "tool".into(),
            link: "c".into(),
            shapes: vec![Obstacle::Cuboid {
                center: Vec3::new(0.325, 0.0, 0.0),
                half_extents: Vec3::new(0.075, 0.02, 0.02),
                rotation: Quat::IDENTITY,
            }],
            touch_links: vec![],
            spheres: SphereOptions::default(),
        })
        .unwrap();
    // Straight, the tool's tip is at x = 1.1; a sphere 0.1 m beyond it.
    let tip = [world(Obstacle::Sphere { center: Vec3::new(1.25, 0.0, 0.5), radius: 0.05 })];
    let [bare, _] = MeshModel::new(&robot).unwrap().clearance(&tip, &[0], &[0.0; 3]).unwrap()[0];
    let [holding, _] = MeshModel::new(&held).unwrap().clearance(&tip, &[0], &[0.0; 3]).unwrap()[0];
    assert!((bare - 0.25).abs() < 1e-4 && (holding - 0.1).abs() < 1e-4, "bare {bare}, holding {holding}");

    // The Panda's collision meshes: clear of itself at its default pose, and a pin through its
    // forearm leaves nothing (a small ball inside the forearm would not touch its surface).
    let panda = common::panda().unwrap();
    let meshes = MeshModel::new(&panda).unwrap();
    let q = panda.default_q().to_vec();
    let [free, self_collision] = meshes.clearance(&[World::default()], &[0], &q).unwrap()[0];
    assert!(free == 1e30 && self_collision > 0.0, "{free} {self_collision}");
    let elbow = panda.link_pose(&q, "panda_link4").unwrap().position;
    let pin = [world(Obstacle::Cuboid {
        center: elbow,
        half_extents: Vec3::new(0.3, 0.005, 0.005),
        rotation: Quat::IDENTITY,
    })];
    assert_eq!(meshes.clearance(&pin, &[0], &q).unwrap()[0][0], 0.0);

    // Mounted on a table, the base rests on it; that is not a collision.
    let mut rng = Rng::new(1);
    let table = common::tabletop(&mut rng);
    let [on_table, _] = meshes.clearance(std::slice::from_ref(&table), &[0], &q).unwrap()[0];
    assert!(on_table > 0.0, "{on_table} on the table");

    // Malformed batches are errors.
    assert!(meshes.clearance(&pin, &[0], &q[1..]).is_err(), "a short configuration");
    assert!(meshes.clearance(&pin, &[1], &q).is_err(), "a world past the end");
}

/// An L-shaped `hook` (a bar along x and an arm up y at its far end) and a small `peg` on its own
/// joint, sitting in the hook's notch: inside its convex hull, 8 cm from its surface.
const HOOK: &str = r#"<robot name="hook">
  <link name="base"/>
  <link name="hook">
    <collision><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.1 0.1"/></geometry></collision>
    <collision><origin xyz="0.35 0.15 0"/><geometry><box size="0.1 0.2 0.1"/></geometry></collision>
  </link>
  <link name="peg"><collision><origin xyz="0.15 0.15 0"/><geometry><box size="0.04 0.04 0.04"/></geometry></collision></link>
  <joint name="jh" type="revolute"><parent link="base"/><child link="hook"/><origin xyz="0 0 0.5"/>
    <axis xyz="0 0 1"/><limit lower="-3.2" upper="3.2" velocity="2" effort="1"/></joint>
  <joint name="jp" type="revolute"><parent link="base"/><child link="peg"/><origin xyz="0 0 0.5"/>
    <axis xyz="0 0 1"/><limit lower="-3.2" upper="3.2" velocity="2" effort="1"/></joint>
</robot>"#;

#[test]
fn concave_links_are_measured_on_their_surface_not_their_hull() {
    let dir = std::env::temp_dir().join(format!("batchplan-hook-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("hook.urdf"), HOOK).unwrap();
    let robot = Robot::load(dir.join("hook.urdf"), &RobotOptions::default()).unwrap();
    let meshes = MeshModel::new(&robot).unwrap();
    // The peg's near corner is 8 cm above the bar, inside the hull's slope from (0, 0.05) to
    // (0.3, 0.25).
    let [_, peg] = meshes.clearance(&[World::default()], &[0], &[0.0, 0.0]).unwrap()[0];
    assert!((peg - 0.08).abs() < 1e-4, "the peg: {peg}");
    // With the peg turned away, a ball in the notch is 9 cm from the bar.
    let ball = world(Obstacle::Sphere { center: Vec3::new(0.15, 0.17, 0.5), radius: 0.03 });
    let [notch, _] = meshes.clearance(&[ball], &[0], &[0.0, std::f32::consts::PI]).unwrap()[0];
    assert!((notch - 0.09).abs() < 1e-4, "the ball: {notch}");
}
