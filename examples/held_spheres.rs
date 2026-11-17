//! How spheres cover what a hand holds, and what covering or fitting it costs: a plate, a can
//! and a capsule held by the Panda, with the default 64 spheres and with 256 to spend. For each,
//! how many spheres it takes and how far they reach past its side and past its ends.
//!
//! cargo run --release --example held_spheres

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use anyhow::Result;
use batchplan::{Held, Obstacle, Pose, SphereOptions};
use glam::{Quat, Vec3};

fn main() -> Result<()> {
    let robot = common::panda()?;
    let q = robot.default_q().to_vec();
    let hand = robot.ee_pose(&q, 0);
    let own = robot.spheres(&q).len();
    let round = |radius: f32, half: f32, capsule: bool| match capsule {
        true => Obstacle::Capsule {
            center: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            radius,
            half_length: half,
        },
        false => Obstacle::Cylinder {
            center: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            radius,
            half_height: half,
        },
    };
    let shapes = [
        ("a plate 13.9 cm across, 18 mm thick", 0.0695, 0.009, false),
        ("a can 6.6 cm across, 12 cm tall", 0.033, 0.06, false),
        ("a capsule 4 cm across, 14 cm long", 0.02, 0.05, true),
    ];
    for (name, radius, half, capsule) in shapes {
        for budget in [64, 256] {
            let held = Held {
                hand: 0,
                object: round(radius, half, capsule),
                grasp: Pose::IDENTITY,
                locked: Default::default(),
                spheres: SphereOptions {
                    budget,
                    ..Default::default()
                },
            };
            let start = Instant::now();
            let holding = robot.attach(&held.attachment(&robot)?)?;
            let took = start.elapsed().as_secs_f64() * 1e3;
            let spheres = &holding.spheres(&q)[own..];
            // In the object's frame, which the grasp puts at the hand's.
            let (mut side, mut ends) = (f32::MIN, f32::MIN);
            for s in spheres {
                let c = hand.rotation.inverse() * (Vec3::new(s[0], s[1], s[2]) - hand.position);
                side = side.max(c.truncate().length() + s[3] - radius);
                let end = if capsule { half + radius } else { half };
                ends = ends.max(c.z.abs() + s[3] - end);
            }
            println!(
                "{name}, {budget} to spend: {} spheres in {took:.2} ms, {:.1} mm past its side, {:.1} mm past its ends",
                spheres.len(),
                side * 1e3,
                ends * 1e3
            );
        }
    }
    Ok(())
}
