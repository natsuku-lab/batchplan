//! MJCF loading: MuJoCo Menagerie models against our URDFs, MuJoCo's conventions on a handwritten
//! model, and scenes as worlds.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;
use std::path::PathBuf;

use batchplan::rng::Rng;
use batchplan::*;
use glam::{DAffine3, DMat3, DQuat, DVec3, Vec3};

/// Kinematics only: these Menagerie models reference meshes that are not bundled.
fn no_spheres() -> RobotOptions {
    RobotOptions { collision_model: Some(CollisionModel::default()), ..Default::default() }
}

fn affine(p: Pose) -> DAffine3 {
    DAffine3::from_rotation_translation(p.rotation.as_dquat(), p.position.as_dvec3())
}

/// Largest difference between two transforms (rotation entries and translation in meters).
fn difference(a: DAffine3, b: DAffine3) -> f64 {
    let rot = (a.matrix3 - b.matrix3).to_cols_array().iter().fold(0.0f64, |m, v| m.max(v.abs()));
    rot.max((a.translation - b.translation).length())
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("batchplan-mjcf-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn menagerie_panda_matches_the_urdf_panda() {
    // Menagerie couples the fingers with a joint equality; locking one locks both.
    let options = RobotOptions { lock_joints: HashMap::from([("finger_joint2".into(), 0.04)]), ..no_spheres() };
    let mjcf = Robot::load(common::asset("menagerie/franka_emika_panda/panda.xml"), &options).unwrap();
    let urdf = common::panda().unwrap();
    assert_eq!(mjcf.joint_names(), ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]);
    assert_eq!(mjcf.lower(), urdf.lower());
    assert_eq!(mjcf.upper(), urdf.upper());
    let mut rng = Rng::new(5);
    let mut worst = 0.0f64;
    for _ in 0..100 {
        let q: Vec<f32> = (0..7).map(|j| rng.range(urdf.lower()[j], urdf.upper()[j])).collect();
        let pairs = (0..8)
            .map(|k| (format!("link{k}"), format!("panda_link{k}")))
            .chain([("hand".into(), "panda_hand".into())]);
        for (m, u) in pairs {
            worst =
                worst.max(difference(affine(mjcf.link_pose(&q, &m).unwrap()), affine(urdf.link_pose(&q, &u).unwrap())));
        }
        let (left, right) = (mjcf.link_pose(&q, "left_finger").unwrap(), mjcf.link_pose(&q, "right_finger").unwrap());
        assert!(((left.position - right.position).length() - 0.08).abs() < 1e-5, "fingers are not 4 cm open each");
    }
    assert!(worst < 1e-5, "link poses differ by {worst}");
}

#[test]
fn menagerie_ur5e_matches_the_urdf_ur5e_up_to_frame_offsets() {
    // Menagerie's world frame is the UR controller's base frame, which the ROS description keeps
    // as its `base` link (rotated half a turn about z). Body frames sit at different points of
    // each link than the URDF's, a constant offset per link, and Menagerie rounds a few
    // dimensions to the millimeter (0.163 m shoulder height against 0.1625 m).
    let mjcf =
        Robot::load(common::asset("menagerie/universal_robots_ur5e/ur5e.xml"), &RobotOptions::default()).unwrap();
    let urdf = Robot::load(common::asset("ur5e/ur_description/urdf/ur5e.urdf"), &RobotOptions::default()).unwrap();
    assert_eq!(mjcf.joint_names(), urdf.joint_names());
    let zero = vec![0.0; 6];
    let world = affine(urdf.link_pose(&zero, "base").unwrap()).inverse();
    let mut rng = Rng::new(6);
    let mut worst = 0.0f64;
    for link in ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"] {
        let offset = (world * affine(urdf.link_pose(&zero, link).unwrap())).inverse()
            * affine(mjcf.link_pose(&zero, link).unwrap());
        for _ in 0..50 {
            let q: Vec<f32> = (0..6).map(|_| rng.range(-3.0, 3.0)).collect();
            let expected = world * affine(urdf.link_pose(&q, link).unwrap()) * offset;
            worst = worst.max(difference(expected, affine(mjcf.link_pose(&q, link).unwrap())));
        }
    }
    assert!(worst < 3e-3, "links differ by {worst} beyond constant offsets");
    // Its collision geometry is primitives, so spheres fit without any mesh files.
    assert!(!mjcf.collision_model().spheres.is_empty());
}

#[test]
fn menagerie_2f85_closes_its_four_bars_as_mujoco_settles_them() {
    let robot = Robot::load(common::asset("menagerie/robotiq_2f85/2f85.xml"), &no_spheres()).unwrap();
    // right_driver_joint follows left_driver_joint through a joint equality, and each finger's
    // coupler, spring link and follower follow it around the four-bar `connect` closes.
    assert_eq!(robot.joint_names(), ["left_driver_joint"]);
    assert_eq!((robot.lower()[0], robot.upper()[0]), (0.0, 0.8));
    // MuJoCo 3.15 with gravity and contact off, settled at actuator targets 0, 31.875, ..., 255:
    // (left_driver_joint, left_pad x, left_pad z).
    let mujoco = [
        (0.00260, -0.04918, 0.12282),
        (0.10251, -0.04459, 0.12626),
        (0.20241, -0.03967, 0.12924),
        (0.30232, -0.03447, 0.13171),
        (0.40222, -0.02905, 0.13367),
        (0.50212, -0.02345, 0.13507),
        (0.60202, -0.01773, 0.13591),
        (0.70193, -0.01195, 0.13618),
        (0.80002, -0.00626, 0.13588),
    ];
    for (driver, x, z) in mujoco {
        let left = robot.link_pose(&[driver], "left_pad").unwrap().position;
        let off = (left - Vec3::new(x, 0.0, z)).length();
        assert!(off < 2e-4, "left pad {left} at {driver}, MuJoCo ({x}, 0, {z}): {off} m off");
        // The right finger mirrors the left.
        let right = robot.link_pose(&[driver], "right_pad").unwrap().position;
        assert!((right - Vec3::new(-left.x, left.y, left.z)).length() < 1e-4, "right pad {right}, left {left}");
    }
}

const CONVENTIONS: &str = r#"<mujoco model="conventions">
  <include file="defaults.xml"/>
  <worldbody>
    <body name="base" pos="0 0 0.1" euler="10 20 30" childclass="arm">
      <joint name="j1" axis="0 0 1"/>
      <body name="upper" pos="0.2 0 0" axisangle="0 0 1 30">
        <joint name="j2"/>
        <joint name="j3" type="slide" axis="1 0 0" range="0 0.1" pos="0.05 0 0"/>
        <geom fromto="0 0 0 0.3 0 0"/>
        <body name="tip" pos="0.3 0 0" xyaxes="0 1 0 -1 0 0">
          <joint name="j4" pos="0 0 0.05" range="-45 45"/>
          <body name="tool" pos="0 0 0.1" zaxis="1 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"#;

const DEFAULTS: &str = r#"<mujoco>
  <default>
    <default class="arm">
      <joint axis="0 1 0" range="-90 90"/>
      <geom type="capsule" size="0.02"/>
    </default>
  </default>
</mujoco>"#;

#[test]
fn mjcf_follows_mujoco_conventions() {
    let dir = scratch("conventions");
    std::fs::write(dir.join("model.xml"), CONVENTIONS).unwrap();
    std::fs::write(dir.join("defaults.xml"), DEFAULTS).unwrap();
    let robot = Robot::load(dir.join("model.xml"), &RobotOptions::default()).unwrap();
    assert_eq!(robot.joint_names(), ["j1", "j2", "j3", "j4"]);
    // Angles are degrees unless the compiler says otherwise; slide ranges are meters.
    let (q, h) = (FRAC_PI_2 as f32, std::f32::consts::FRAC_PI_4);
    assert_eq!(robot.lower(), [-q, -q, 0.0, -h]);
    assert_eq!(robot.upper(), [q, q, 0.1, h]);
    let deg = |d: f64| d.to_radians();
    let t = |x: f64, y: f64, z: f64| DAffine3::from_translation(DVec3::new(x, y, z));
    let r = |m: DMat3| DAffine3::from_mat3(m);
    let mut rng = Rng::new(2);
    for _ in 0..50 {
        let q: Vec<f32> = (0..4).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect();
        let [q1, q2, q3, q4] = [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64];
        // Euler angles default to the intrinsic x-y-z sequence.
        let base_rot =
            DMat3::from_rotation_x(deg(10.0)) * DMat3::from_rotation_y(deg(20.0)) * DMat3::from_rotation_z(deg(30.0));
        let expected = t(0.0, 0.0, 0.1)
            * r(base_rot)
            * r(DMat3::from_rotation_z(q1))
            * t(0.2, 0.0, 0.0)
            * r(DMat3::from_rotation_z(deg(30.0)))
            * r(DMat3::from_rotation_y(q2))
            * t(q3, 0.0, 0.0)
            * t(0.3, 0.0, 0.0)
            * r(DMat3::from_rotation_z(FRAC_PI_2)) // xyaxes "0 1 0 -1 0 0"
            * t(0.0, 0.0, 0.05)
            * r(DMat3::from_rotation_y(q4))
            * t(0.0, 0.0, -0.05)
            * t(0.0, 0.0, 0.1)
            * r(DMat3::from_quat(DQuat::from_rotation_arc(DVec3::Z, DVec3::X))); // zaxis "1 0 0"
        let got = affine(robot.link_pose(&q, "tool").unwrap());
        assert!(difference(got, expected) < 1e-5, "tool pose off by {} at {q:?}", difference(got, expected));
    }
    // The fromto capsule spans the upper body: spheres were fitted along it.
    let spheres = &robot.collision_model().spheres["upper"];
    let reach = spheres.iter().map(|s| s[0] + s[3]).fold(f32::MIN, f32::max);
    assert!((reach - 0.32).abs() < 0.01, "capsule should reach x = 0.32, spheres reach {reach}");
}

#[test]
fn scenes_load_as_worlds_without_their_robots() {
    let floor = World::load(common::asset("menagerie/franka_emika_panda/scene.xml"), &SdfOptions::default()).unwrap();
    assert_eq!(floor.obstacles.len(), 1, "the floor only; the Panda is a robot, not scene");
    let (top, _) = floor.obstacles[0].distance(Vec3::new(0.3, 0.2, 0.0));
    assert!(top.abs() < 1e-6, "the floor's surface is z = 0");

    let dir = scratch("scene");
    let scene = r#"<mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <geom name="post" type="cylinder" pos="0.5 0 0.3" size="0.05 0.3"/>
        <body name="table" pos="0.6 0 0">
          <geom type="box" size="0.3 0.4 0.02" pos="0 0 -0.02"/>
          <geom type="sphere" size="0.1" contype="0" conaffinity="0"/>
        </body>
        <body name="arm" pos="0 0 0">
          <joint name="j" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        </body>
      </worldbody>
    </mujoco>"#;
    std::fs::write(dir.join("scene.xml"), scene).unwrap();
    let world = World::load(dir.join("scene.xml"), &SdfOptions::default()).unwrap();
    assert_eq!(world.obstacles.len(), 2, "the post and the table top; not the visual sphere or the arm");
    let robot = Robot::load(dir.join("scene.xml"), &RobotOptions::default()).unwrap();
    assert_eq!(robot.joint_names(), ["j"]);
    assert_eq!(robot.collision_model().spheres.len(), 1, "only the arm's geometry belongs to the robot");
}

#[test]
fn free_and_ball_joints_become_one_axis_joints() {
    let dir = scratch("mjcf-compound");
    let path = dir.join("free.xml");
    std::fs::write(
        &path,
        r#"<mujoco model="free">
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint name="free"/>
      <geom type="box" size="0.1 0.1 0.05"/>
      <body name="arm" pos="0 0 0.05">
        <joint name="shoulder" type="ball" range="0 60"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>"#,
    )
    .unwrap();
    let limits = ["free_x", "free_y", "free_z"].map(|n| (n.to_string(), [-1.0f32, 1.0]));
    let options =
        RobotOptions { joint_limits: limits.into_iter().collect(), ee_links: vec!["arm".into()], ..no_spheres() };
    let robot = Robot::load(&path, &options).unwrap();
    let names =
        ["free_x", "free_y", "free_z", "free_rx", "free_ry", "free_rz", "shoulder_rx", "shoulder_ry", "shoulder_rz"];
    assert_eq!(robot.joint_names(), names);
    // A ball joint's range bounds each of its angles (60 degrees, under the 90 the middle one allows).
    let sixty = 60f32.to_radians();
    assert!(robot.upper()[6..].iter().all(|&u| (u - sixty).abs() < 1e-6), "{:?}", robot.upper());
    // The free joint starts at the body's authored pose and moves it in the world frame.
    let mut q = vec![0.0; 9];
    assert!((robot.link_pose(&q, "base").unwrap().position - Vec3::new(0.0, 0.0, 0.5)).length() < 1e-6);
    q[2] = 0.2;
    q[5] = 0.4;
    let base = robot.link_pose(&q, "base").unwrap();
    let expected = DAffine3::from_translation(DVec3::new(0.0, 0.0, 0.7)) * DAffine3::from_rotation_z(0.4);
    assert!(difference(affine(base), expected) < 1e-6, "{base:?}");
}

#[test]
fn nonlinear_joint_equalities_follow_their_polynomial_on_every_device() {
    let dir = scratch("mjcf-polycoef");
    let path = dir.join("rocker.xml");
    std::fs::write(
        &path,
        r#"<mujoco model="rocker">
  <compiler angle="radian"/>
  <worldbody>
    <body name="crank" pos="0 0 0.3">
      <joint name="crank" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      <body name="rocker" pos="0.3 0 0">
        <joint name="rocker" axis="0 1 0" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="rocker" joint2="crank" polycoef="0.1 0.5 0.3 -0.2 0.1"/>
  </equality>
</mujoco>"#,
    )
    .unwrap();
    let robot = Robot::load(&path, &RobotOptions { ee_links: vec!["rocker".into()], ..Default::default() }).unwrap();
    assert_eq!(robot.joint_names(), ["crank"]);
    let curve = |x: f64| 0.1 + x * (0.5 + x * (0.3 + x * (-0.2 + x * 0.1)));
    for k in 0..=20 {
        let x = -1.0 + k as f64 / 10.0;
        let rocker = affine(robot.link_pose(&[x as f32], "rocker").unwrap());
        // Both hinges turn about y, so the rocker's angle is the crank's plus its curve.
        let expected = DMat3::from_rotation_y(x + curve(x));
        let off = (rocker.matrix3 - expected).to_cols_array().iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(off < 1e-5, "rocker off by {off} at {x}");
    }
    // The collision gradient carries the curve's slope: it matches finite differences on the CPU,
    // and the GPU matches the CPU.
    let scene = [World {
        obstacles: vec![Obstacle::Cuboid {
            center: Vec3::new(0.55, 0.0, 0.1),
            half_extents: Vec3::new(0.2, 0.2, 0.05),
            rotation: glam::Quat::IDENTITY,
        }],
    }];
    let w = CollisionWeights { world: 1000.0, self_collision: 0.0, margin: 0.05, self_margin: 0.0 };
    let q: Vec<f32> = (0..400).map(|k| -1.0 + 2.0 * k as f32 / 399.0).collect();
    let cpu = Device::cpu(&robot).unwrap();
    let reference = cpu.evaluate(&cpu.upload(&scene).unwrap(), &vec![0; 400], &q, &w).unwrap();
    let cost = |x: f32| cpu.evaluate(&cpu.upload(&scene).unwrap(), &[0], &[x], &w).unwrap().cost[0];
    let mut checked = 0;
    for (i, &x) in q.iter().enumerate().filter(|&(i, _)| reference.cost[i] > 1e-3) {
        let fd = |h: f32| (cost(x + h) - cost(x - h)) / (2.0 * h);
        let g = reference.grad[i];
        let err = (fd(2e-4) - g).abs().min((fd(5e-5) - g).abs()) / g.abs().max(1.0);
        assert!(err < 2e-2, "gradient {g} against finite differences at {x}");
        checked += 1;
    }
    assert!(checked > 20, "only {checked} configurations touch the box");
    match Device::gpu(&robot) {
        Ok(gpu) => {
            let e = gpu.evaluate(&gpu.upload(&scene).unwrap(), &vec![0; 400], &q, &w).unwrap();
            for (i, x) in q.iter().enumerate() {
                let scale = reference.grad[i].abs().max(1.0);
                assert!((e.grad[i] - reference.grad[i]).abs() / scale < 1e-3, "gradient at {x}");
                assert!((e.cost[i] - reference.cost[i]).abs() / reference.cost[i].max(1.0) < 1e-3, "cost at {x}");
            }
        }
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
}
