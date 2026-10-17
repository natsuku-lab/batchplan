//! Two reBot B601-DM arms at a table, and the handover scenes `examples/handover` plans: shared
//! with the Rerun demo (`demos/src/bin/handover.rs`), which includes this file.

use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;
use std::path::Path;

use anyhow::{Context, Result};
use batchplan::datagen::{Handover, PickPlaceOptions};
use batchplan::glam::{Quat, Vec3};
use batchplan::rng::Rng;
use batchplan::*;

/// The arms' bases, either side of the table's centre line.
const BASE_Y: f32 = 0.22;
/// Each arm's ready pose, its gripper pointing down 27 cm in front of its base.
const READY: [f32; 6] = [0.0, -1.5, -1.5, 1.45, 0.0, 0.0];
/// The fingers held open: each 3.5 cm from the middle.
pub const FINGERS_OPEN: f32 = 0.035;
/// The blocks' half sizes: long enough for both hands, tall enough that the fingertips, 2 cm below
/// the grasp point, stay clear of the table. A few fixed blocks, like a kit of objects, plan in a
/// few batches; each size carried needs its own spheres, so every size is a batch of its own.
const BLOCKS: [Vec3; 4] = [
    Vec3::new(0.12, 0.0125, 0.025),
    Vec3::new(0.125, 0.016, 0.028),
    Vec3::new(0.13, 0.015, 0.027),
    Vec3::new(0.14, 0.0175, 0.03),
];

/// The two arms as one robot, `left_` and `right_`, each with its grasp point (`tcp`) as a hand.
/// Spheres are fitted to one arm once and kept in `spheres.json` beside it.
pub fn two_rebots(dir: &Path) -> Result<Robot> {
    let urdf = dir.join("DM/urdf/ReBot_Arm_DM_tcp.urdf");
    anyhow::ensure!(urdf.exists(), "{} is missing: run scripts/fetch_rebot.sh", urdf.display());
    let arm = |prefix: &str| -> RobotOptions {
        RobotOptions {
            ee_links: vec![format!("{prefix}tcp")],
            lock_joints: HashMap::from([(format!("{prefix}finger_left"), FINGERS_OPEN)]),
            // The description's 50-200 rad/s are motor limits, not the arm's.
            velocity_scale: 0.06,
            ..Default::default()
        }
    };
    let cached = dir.join("spheres.json");
    let spheres = match CollisionModel::load(&cached) {
        Ok(spheres) => spheres,
        Err(_) => {
            let fit = RobotOptions {
                spheres: SphereOptions { budget: 128, voxel_size: 0.005, ..Default::default() },
                ..arm("")
            };
            let spheres = Robot::load(&urdf, &fit)?.collision_model().clone();
            spheres.save(&cached)?;
            spheres
        }
    };
    let part = |prefix: &str, y: f32| Part {
        path: urdf.clone(),
        prefix: prefix.into(),
        base: Pose { position: Vec3::new(0.0, y, 0.0), rotation: Quat::IDENTITY },
    };
    let parts = [part("left_", BASE_Y), part("right_", -BASE_Y)];
    let (left, right) = (arm("left_"), arm("right_"));
    let mut options = RobotOptions {
        ee_links: [left.ee_links, right.ee_links].concat(),
        lock_joints: left.lock_joints.into_iter().chain(right.lock_joints).collect(),
        collision_model: Some(CollisionModel::assemble(&[("left_", &spheres), ("right_", &spheres)])),
        ..arm("")
    };
    // Joints come breadth first, the arms interleaved: each takes its ready pose by name.
    let names = Robot::assemble(&parts, &options)?.joint_names().to_vec();
    let ready = |name: &str| -> Option<f32> {
        let joint = name.trim_start_matches("left_").trim_start_matches("right_");
        let k: usize = joint.strip_prefix("joint")?.parse().ok()?;
        READY.get(k.checked_sub(1)?).copied()
    };
    options.default_q = Some(names.iter().map(|n| ready(n).context("a reBot joint")).collect::<Result<_>>()?);
    Ok(Robot::assemble(&parts, &options)?)
}

/// A table, a block lying across the arms in front of one, where the hands meet between them, the
/// place in front of the other arm, and a few small boxes elsewhere. Even tasks go left to right.
pub fn scene(world: u32, rng: &mut Rng) -> (World, Handover) {
    let (from, to, side) = if world.is_multiple_of(2) { (0, 1, 1.0) } else { (1, 0, -1.0) };
    let across = |rng: &mut Rng, spread: f32| Quat::from_rotation_z(FRAC_PI_2 + rng.range(-spread, spread));
    let table = Obstacle::Cuboid {
        center: Vec3::new(0.25, 0.0, -0.02),
        half_extents: Vec3::new(0.45, 0.6, 0.02),
        rotation: Quat::IDENTITY,
    };
    let half = BLOCKS[(rng.uniform() * BLOCKS.len() as f32) as usize % BLOCKS.len()];
    let block = Obstacle::Cuboid {
        center: Vec3::new(rng.range(0.2, 0.3), side * rng.range(0.18, 0.28), half.z),
        half_extents: half,
        rotation: across(rng, 0.4),
    };
    let meet = Pose {
        position: Vec3::new(rng.range(0.24, 0.28), rng.range(-0.03, 0.03), rng.range(0.09, 0.12)),
        rotation: across(rng, 0.2),
    };
    let place = Pose {
        position: Vec3::new(rng.range(0.2, 0.3), -side * rng.range(0.18, 0.28), half.z),
        rotation: across(rng, 0.4),
    };
    let mut obstacles = vec![table, block];
    // Clutter beyond the blocks' reach, clear of where the hands work.
    for _ in 0..(rng.uniform() * 3.0) as usize {
        let h = Vec3::new(rng.range(0.015, 0.03), rng.range(0.015, 0.03), rng.range(0.02, 0.06));
        let center = Vec3::new(rng.range(0.4, 0.5), rng.range(-0.35, 0.35), h.z);
        obstacles.push(Obstacle::Cuboid {
            center,
            half_extents: h,
            rotation: Quat::from_rotation_z(rng.range(0.0, 3.0)),
        });
    }
    (World { obstacles }, Handover { world, object: 1, from, to, meet, place })
}

/// The options both arms plan with: they approach from 5 cm above, hold blocks 1 cm below their
/// top, open 7 cm, and are sampled `hz` times a second.
pub fn options(hz: f32) -> PickPlaceOptions {
    PickPlaceOptions {
        approach: 0.05,
        grasp_depth: 0.01,
        max_width: 2.0 * FINGERS_OPEN,
        dt: 1.0 / hz,
        ..Default::default()
    }
}
