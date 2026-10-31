//! Trajectories for `scripts/actuators.py` to play back on actuator models in MuJoCo: random
//! motions of one arm, each timed under several limits, or pick-like sequences, each timed with
//! its motions one after another and joined (`join`) at several blend radii.
//!
//!   scripts/fetch_rebot.sh
//!   cargo run --release --example actuators -- <so101|rebot> <out_dir> [motions=256]
//!   cargo run --release --example actuators -- <so101|rebot> <out_dir> [targets=128] --sequences
//!
//! It writes `<out_dir>/manifest.json` and `<out_dir>/samples.bin`: per variant and motion, `len`
//! samples at 500 Hz of position, velocity and acceleration, `[len, 3, dof]` little-endian f32.

#[path = "common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use batchplan::glam::Vec3;
use batchplan::rng::Rng;
use batchplan::{
    CollisionModel, Device, JoinOptions, PlanOptions, PlanProblem, Pose, Robot, RobotOptions,
    Sequence, ToolPath, ToolPathOptions, Trajectory, World,
};
use serde_json::json;

/// Samples per second: MuJoCo's step, so every step gets its own target.
const HZ: f32 = 500.0;

/// The limits each motion is timed under: scales of the robot's own, then a speed scale.
const VARIANTS: [(&str, f32, f32, f32); 6] = [
    ("limits", 1.0, 1.0, 1.0),
    ("speed 0.75", 1.0, 1.0, 0.75),
    ("speed 0.5", 1.0, 1.0, 0.5),
    ("jerk 0.5", 1.0, 0.5, 1.0),
    ("jerk 0.25", 1.0, 0.25, 1.0),
    ("acceleration 0.5", 0.5, 1.0, 1.0),
];

/// Blend radii the sequences are joined at (meters).
const BLENDS: [f32; 3] = [0.02, 0.05, 0.1];

/// One motion's interleaved samples and its duration (s).
type Timed = (Vec<f32>, f32);
/// A variant's name and its motions.
type Variant = (String, Vec<Timed>);

/// The arm's description, its options, and how far its hand moves in and out (meters).
fn arm(name: &str) -> Result<(PathBuf, RobotOptions, f32)> {
    Ok(match name {
        "so101" => (
            common::asset("so101/so101_new_calib.urdf").into(),
            RobotOptions::default(),
            0.03,
        ),
        "rebot" => {
            let urdf = PathBuf::from("data/rebot/DM/urdf/ReBot_Arm_DM_tcp.urdf");
            ensure!(
                urdf.exists(),
                "{} is missing: run scripts/fetch_rebot.sh",
                urdf.display()
            );
            let options = RobotOptions {
                ee_links: vec!["tcp".into()],
                lock_joints: HashMap::from([("finger_left".to_string(), 0.035)]),
                // As examples/handover: the description's 50-200 rad/s are motor limits.
                velocity_scale: 0.06,
                collision_model: CollisionModel::load("data/rebot/spheres.json").ok(),
                ..Default::default()
            };
            (urdf, options, 0.05)
        }
        _ => anyhow::bail!("unknown arm {name}: so101 or rebot"),
    })
}

/// One trajectory's samples, interleaved per sample: position, velocity, acceleration.
fn rows(trajectory: &Trajectory, n: usize) -> Result<Vec<f32>> {
    let s = trajectory.sample(HZ)?;
    let mut out = Vec::with_capacity(3 * s.positions.len());
    for k in 0..s.positions.len() / n {
        for row in [&s.positions, &s.velocities, &s.accelerations] {
            out.extend_from_slice(&row[k * n..(k + 1) * n]);
        }
    }
    Ok(out)
}

/// Writes the variants' trajectories and the manifest that indexes them.
fn write(out: &Path, arm: &str, urdf: &Path, robot: &Robot, variants: &[Variant]) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let mut samples = BufWriter::new(File::create(out.join("samples.bin"))?);
    let (mut offset, mut listed) = (0, vec![]);
    for (name, motions) in variants {
        let mut entries = vec![];
        for (rows, duration) in motions {
            for x in rows {
                samples.write_all(&x.to_le_bytes())?;
            }
            let len = rows.len() / (3 * robot.dof());
            entries.push(json!({"offset": offset, "len": len, "duration": duration}));
            offset += rows.len();
        }
        listed.push(json!({"name": name, "motions": entries}));
    }
    samples.flush()?;
    let manifest = json!({
        "arm": arm,
        "urdf": urdf.canonicalize().context("the description's path")?,
        "joints": robot.joint_names(),
        "hz": HZ,
        "variants": listed,
    });
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 2,
        "usage: actuators <so101|rebot> <out_dir> [count] [--sequences]"
    );
    let sequences = args.iter().any(|a| a == "--sequences");
    let count: usize = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .map_or(Ok(if sequences { 128 } else { 256 }), |a| a.parse())?;
    let (urdf, options, reach) = arm(&args[0])?;
    let robot = Robot::load(&urdf, &options)?;
    let n = robot.dof();
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;

    // Random configurations within the joints' limits, in free space: the planner turns down any
    // in self-collision.
    let mut rng = Rng::new(1);
    let mut random = || -> Vec<f32> {
        (0..n)
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect()
    };
    let mut variants = vec![];
    if !sequences {
        let problems: Vec<PlanProblem> = (0..count)
            .map(|_| PlanProblem::new(0, random(), random()))
            .collect();
        let plans = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default())?;
        let paths: Vec<&[f32]> = plans.solved().map(|s| s.solution).collect();
        println!("{} of {count} motions planned", paths.len());
        for (name, acceleration, jerk, speed) in VARIANTS {
            let limited = Robot::load(
                &urdf,
                &RobotOptions {
                    collision_model: Some(robot.collision_model().clone()),
                    acceleration_scale: acceleration,
                    jerk_scale: jerk,
                    ..options.clone()
                },
            )?;
            let mut motions = vec![];
            for path in &paths {
                let trajectory = Trajectory::try_new(&limited, path, speed)?;
                motions.push((rows(&trajectory, n)?, trajectory.duration()));
            }
            variants.push((name.to_string(), motions));
        }
    } else {
        // A transit to a pre-grasp, then the hand in along its z; or back out, then the transit
        // to the next pre-grasp.
        let targets: Vec<Vec<f32>> = (0..=count).map(|_| random()).collect();
        let transits: Vec<PlanProblem> = std::iter::once(robot.default_q().to_vec())
            .chain(targets.iter().cloned())
            .zip(&targets)
            .map(|(a, b)| PlanProblem::new(0, a, b.clone()))
            .collect();
        let plans = batchplan::plan(&device, &worlds, &transits, &PlanOptions::default())?;
        let o = ToolPathOptions::default();
        let straight = |start: &[f32], to: Pose| -> Result<Option<Vec<f32>>> {
            let path = ToolPath {
                world: 0,
                start: start.to_vec(),
                hand: 0,
                waypoints: vec![to],
            };
            let followed = batchplan::follow(&device, &worlds, &[path], &o)?;
            Ok((followed.fraction[0] == 1.0).then(|| followed.paths[0].clone()))
        };
        let mut found = vec![];
        for k in 0..count {
            let (Some(transit), Some(_)) = (plans.best(k), plans.best(k + 1)) else {
                continue;
            };
            let pose = robot.ee_pose(&targets[k], 0);
            let inside = Pose::new(
                pose.position + pose.rotation * Vec3::Z * reach,
                pose.rotation,
            );
            let Some(inward) = straight(&targets[k], inside)? else {
                continue;
            };
            let Some(outward) = straight(&inward[inward.len() - n..], pose)? else {
                continue;
            };
            let end = outward[outward.len() - n..].to_vec();
            let onward = PlanProblem::new(0, end, targets[k + 1].clone());
            let onward = batchplan::plan(&device, &worlds, &[onward], &PlanOptions::default())?;
            let Some(onward) = onward.best(0) else {
                continue;
            };
            found.push(Sequence {
                world: 0,
                paths: vec![transit.to_vec(), inward],
            });
            found.push(Sequence {
                world: 0,
                paths: vec![outward, onward.to_vec()],
            });
        }
        println!("{} sequences of two motions", found.len());
        let timed = |stretches: &[Vec<f32>]| -> Result<Timed> {
            let (mut all, mut duration) = (vec![], 0.0);
            for path in stretches {
                let trajectory = Trajectory::try_new(&robot, path, 1.0)?;
                duration += trajectory.duration();
                all.extend(rows(&trajectory, n)?);
            }
            Ok((all, duration))
        };
        let apart = found
            .iter()
            .map(|s| timed(&s.paths))
            .collect::<Result<_>>()?;
        variants.push(("apart".to_string(), apart));
        for blend in BLENDS {
            let o = JoinOptions {
                blend,
                ..Default::default()
            };
            let joined = batchplan::join(&device, &worlds, &found, &o)?;
            let motions = joined
                .paths
                .iter()
                .map(|p| timed(p))
                .collect::<Result<_>>()?;
            variants.push((format!("joined {} cm", blend * 100.0), motions));
        }
    }
    let out = PathBuf::from(&args[1]);
    write(&out, &args[0], &urdf, &robot, &variants)?;
    println!("wrote {}", out.display());
    Ok(())
}
