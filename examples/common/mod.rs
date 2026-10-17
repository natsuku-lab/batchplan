//! Random tabletop worlds shared by the examples and tests.
#![allow(dead_code)]

use std::f32::consts::PI;

use std::collections::HashMap;

use batchplan::rng::Rng;
use batchplan::{CollisionModel, CollisionWeights, Device, Obstacle, Pose, Robot, RobotOptions, World};
use glam::{Quat, Vec3};

pub fn asset(path: &str) -> String {
    format!("{}/assets/{path}", env!("CARGO_MANIFEST_DIR"))
}

/// The Franka Panda with its fingers held open, a 10 cm tool frame below the hand as the IK
/// frame, and hand-tuned collision spheres converted from cuRobo.
pub fn panda_options() -> RobotOptions {
    let fingers = [("panda_finger_joint1".to_string(), 0.04), ("panda_finger_joint2".to_string(), 0.04)];
    RobotOptions {
        ee_links: vec!["ee_link".into()],
        lock_joints: HashMap::from(fingers),
        default_q: Some(vec![0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]),
        collision_model: Some(CollisionModel::load(asset("franka/panda_collision.json")).expect("Panda spheres")),
        ..Default::default()
    }
}

pub fn panda() -> Result<Robot, batchplan::Error> {
    Robot::load(asset("franka/franka_panda.urdf"), &panda_options())
}

/// Two Pandas 70 cm apart along y, facing +x, as `left_` and `right_` parts: hands 0 and 1, each
/// at its ready pose with its fingers held open.
pub fn two_pandas() -> Result<Robot, batchplan::Error> {
    let one = panda_options();
    let arm = |prefix: &str, y: f32| batchplan::Part {
        path: asset("franka/franka_panda.urdf").into(),
        prefix: prefix.into(),
        base: Pose { position: Vec3::new(0.0, y, 0.0), rotation: Quat::IDENTITY },
    };
    let parts = [arm("left_", 0.35), arm("right_", -0.35)];
    let spheres = one.collision_model.as_ref().expect("Panda spheres");
    let mut options = RobotOptions {
        ee_links: vec!["left_ee_link".into(), "right_ee_link".into()],
        lock_joints: ["left_", "right_"]
            .iter()
            .flat_map(|p| one.lock_joints.iter().map(move |(j, v)| (format!("{p}{j}"), *v)))
            .collect(),
        collision_model: Some(CollisionModel::assemble(&[("left_", spheres), ("right_", spheres)])),
        ..Default::default()
    };
    // Joints come breadth first, the arms interleaved: each takes its ready pose by name.
    let names = Robot::assemble(&parts, &options)?.joint_names().to_vec();
    let single = Robot::load(asset("franka/franka_panda.urdf"), &one)?;
    let ready = |name: &str| {
        let joint = name.trim_start_matches("left_").trim_start_matches("right_");
        single.joint_names().iter().position(|j| j == joint).map(|k| single.default_q()[k])
    };
    options.default_q = Some(names.iter().map(|n| ready(n).expect("a Panda joint")).collect());
    Robot::assemble(&parts, &options)
}

/// A table at z = 0 (the robot base height) with 2-6 random boxes in front of the robot.
pub fn tabletop(rng: &mut Rng) -> World {
    let mut obstacles = vec![Obstacle::Cuboid {
        center: Vec3::new(0.4, 0.0, -0.02),
        half_extents: Vec3::new(0.7, 0.8, 0.02),
        rotation: Quat::IDENTITY,
    }];
    let boxes = 2 + (rng.uniform() * 5.0) as usize;
    for _ in 0..boxes {
        let half = Vec3::new(rng.range(0.02, 0.08), rng.range(0.02, 0.08), rng.range(0.03, 0.2));
        obstacles.push(Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.3, 0.75), rng.range(-0.5, 0.5), half.z),
            half_extents: half,
            rotation: Quat::from_rotation_z(rng.range(0.0, PI)),
        });
    }
    World { obstacles }
}

/// A top-down grasp pose (gripper z axis pointing down) in free space above the table.
pub fn grasp_target(world: &World, rng: &mut Rng) -> Pose {
    loop {
        let p = Vec3::new(rng.range(0.3, 0.7), rng.range(-0.45, 0.45), rng.range(0.05, 0.4));
        if world.obstacles.iter().all(|o| o.distance(p).0 > 0.06) {
            let rotation = Quat::from_rotation_z(rng.range(-PI, PI)) * Quat::from_rotation_x(PI);
            return Pose { position: p, rotation };
        }
    }
}

/// A table under any robot and three posts within its reach; none of them touch the default pose.
pub fn tabletop_for(robot: &Robot, cpu: &Device, rng: &mut Rng) -> World {
    let mut reach = 0.0f32;
    for _ in 0..500 {
        let q: Vec<f32> = (0..robot.dof()).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect();
        reach = reach.max(robot.ee_pose(&q, 0).position.truncate().length());
    }
    let model = robot.collision_model();
    let floor = model
        .spheres
        .iter()
        .flat_map(|(link, spheres)| {
            let pose = robot.link_pose(robot.default_q(), link).unwrap();
            spheres.iter().map(move |s| (pose.position + pose.rotation * Vec3::new(s[0], s[1], s[2])).z - s[3])
        })
        .fold(f32::INFINITY, f32::min);
    let table = Obstacle::Cuboid {
        center: Vec3::new(0.0, 0.0, floor - 0.01 - reach * 0.05),
        half_extents: Vec3::new(2.0 * reach, 2.0 * reach, reach * 0.05),
        rotation: Quat::IDENTITY,
    };
    let mut world = World { obstacles: vec![table] };
    while world.obstacles.len() < 4 {
        let (angle, radius) = (rng.range(-3.1, 3.1), rng.range(0.4, 0.8) * reach);
        let half = Vec3::new(0.06, 0.06, 0.3) * reach;
        let candidate = Obstacle::Cuboid {
            center: Vec3::new(radius * angle.cos(), radius * angle.sin(), floor - 0.01 + half.z),
            half_extents: half,
            rotation: Quat::from_rotation_z(rng.range(0.0, 3.1)),
        };
        let mut trial = world.clone();
        trial.obstacles.push(candidate);
        let uploaded = cpu.upload(std::slice::from_ref(&trial)).unwrap();
        if cpu.evaluate(&uploaded, &[0], robot.default_q(), &CollisionWeights::NONE).unwrap().collision_free(0) {
            world = trial;
        }
    }
    world
}

/// A table with its top at z = 0, two tall boxes, and a small box to move across the table: the
/// world and the task. The small box is the world's last obstacle.
pub fn pick_place_scene(world: u32, rng: &mut Rng) -> (World, batchplan::datagen::PickPlace) {
    let table = Obstacle::Cuboid {
        center: Vec3::new(0.4, 0.0, -0.02),
        half_extents: Vec3::new(0.7, 0.8, 0.02),
        rotation: Quat::IDENTITY,
    };
    let mut obstacles = vec![table];
    for side in [-1.0, 1.0] {
        obstacles.push(Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.6, 0.7), side * rng.range(0.3, 0.45), 0.15),
            half_extents: Vec3::new(0.04, 0.04, 0.15),
            rotation: Quat::from_rotation_z(rng.range(0.0, 3.0)),
        });
    }
    let half = Vec3::new(rng.range(0.015, 0.03), rng.range(0.025, 0.04), rng.range(0.02, 0.04));
    let spot = |rng: &mut Rng, side: f32| Vec3::new(rng.range(0.35, 0.55), side * rng.range(0.1, 0.3), half.z);
    let side = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
    obstacles.push(Obstacle::Cuboid {
        center: spot(rng, side),
        half_extents: half,
        rotation: Quat::from_rotation_z(rng.range(-1.2, 1.2)),
    });
    let place = Pose { position: spot(rng, -side), rotation: Quat::from_rotation_z(rng.range(-1.2, 1.2)) };
    let object = obstacles.len() - 1;
    (World { obstacles }, batchplan::datagen::PickPlace { world, object, place })
}
