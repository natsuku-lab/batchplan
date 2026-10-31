//! Joining pick-like motions without stopping (`join`), on one arm: how much sooner they finish
//! than one after another, at several blend radii, and how long joining takes. Each sequence is a
//! planned transit to a random pre-grasp configuration and the hand's straight move in along its z,
//! or the straight move back out and the transit to the next pre-grasp.
//!
//!   scripts/fetch_rebot.sh
//!   cargo run --release --example join                  # a reBot B601-DM, 64 pre-grasps
//!   cargo run --release --example join -- so101 128     # the SO-101, more

#[path = "common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, ensure};
use batchplan::glam::Vec3;
use batchplan::rng::Rng;
use batchplan::{
    CollisionModel, Device, JoinOptions, PlanOptions, PlanProblem, Pose, Robot, RobotOptions,
    Sequence, ToolPath, ToolPathOptions, Trajectory, World,
};

/// Blend radii tried (meters).
const BLENDS: [f32; 3] = [0.02, 0.05, 0.1];

/// The arm, and how far its hand moves in and out (meters).
fn arm(name: &str) -> Result<(Robot, f32)> {
    match name {
        "rebot" => {
            let urdf = Path::new("data/rebot/DM/urdf/ReBot_Arm_DM_tcp.urdf");
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
            Ok((Robot::load(urdf, &options)?, 0.05))
        }
        "so101" => {
            let urdf = common::asset("so101/so101_new_calib.urdf");
            Ok((Robot::load(urdf, &RobotOptions::default())?, 0.03))
        }
        _ => anyhow::bail!("unknown arm {name}: rebot or so101"),
    }
}

/// Seconds the paths take, each timed alone.
fn duration(robot: &Robot, paths: &[Vec<f32>]) -> Result<f32> {
    let mut total = 0.0;
    for path in paths {
        total += Trajectory::try_new(robot, path, 1.0)?.duration();
    }
    Ok(total)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().map_or("rebot", String::as_str);
    let count: usize = args.get(1).map_or(Ok(64), |a| a.parse())?;
    let (robot, reach) = arm(name)?;
    let n = robot.dof();
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;

    // Pre-grasps at random configurations, in free space: the planner turns down any in
    // self-collision. Transits run from home to the first and from each to the next.
    let mut rng = Rng::new(1);
    let mut random = || -> Vec<f32> {
        (0..n)
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect()
    };
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
    let mut sequences = vec![];
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
        sequences.push(Sequence {
            world: 0,
            paths: vec![transit.to_vec(), inward],
        });
        sequences.push(Sequence {
            world: 0,
            paths: vec![outward, onward.to_vec()],
        });
    }
    println!("{name}: {} sequences of two motions", sequences.len());

    let apart: Vec<f32> = sequences
        .iter()
        .map(|s| duration(&robot, &s.paths))
        .collect::<Result<_>>()?;
    for blend in BLENDS {
        let o = JoinOptions {
            blend,
            ..Default::default()
        };
        let start = Instant::now();
        let joined = batchplan::join(&device, &worlds, &sequences, &o)?;
        let took = start.elapsed().as_secs_f64() * 1e3;
        let mut saved: Vec<f32> = joined
            .paths
            .iter()
            .zip(&apart)
            .map(|(paths, apart)| Ok(1.0 - duration(&robot, paths)? / apart))
            .collect::<Result<_>>()?;
        saved.sort_by(f32::total_cmp);
        let blended = joined
            .blends
            .iter()
            .flatten()
            .filter(|b| b.is_some())
            .count();
        println!(
            "blend {:>2.0} cm: {blended} of {} joined without stopping; sooner by {:.1}% on average, \
             {:.1}% at the median, {:.1}% at most, {:.1}% at least; {took:.1} ms",
            blend * 100.0,
            sequences.len(),
            100.0 * saved.iter().sum::<f32>() / saved.len() as f32,
            100.0 * saved[saved.len() / 2],
            100.0 * saved[saved.len() - 1],
            100.0 * saved[0],
        );
    }
    Ok(())
}
