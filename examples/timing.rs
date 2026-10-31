//! How long timing takes and how long the trajectories it gives last, on the Panda, by kind of
//! path: paths planned from rest, paths replanned from halfway along them while moving, and the
//! hand's straight moves. Timing runs on the calling thread; each kind is timed five times and
//! the fastest pass kept.
//!
//!   cargo run --release --example timing

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use anyhow::Result;
use batchplan::glam::Vec3;
use batchplan::rng::Rng;
use batchplan::{
    Device, JointState, PlanOptions, PlanProblem, Pose, Robot, StartMotion, ToolPath,
    ToolPathOptions, Trajectory, World,
};

/// Passes over each kind, the fastest kept.
const PASSES: usize = 5;

/// The trajectories' mean duration (s) and the fastest pass's time per trajectory (µs).
fn timed(
    paths: &[Vec<f32>],
    mut time: impl FnMut(usize, &[f32]) -> Result<Trajectory>,
) -> Result<(f64, f64)> {
    let (mut fastest, mut total) = (f64::INFINITY, 0.0);
    for _ in 0..PASSES {
        let start = Instant::now();
        total = 0.0;
        for (i, path) in paths.iter().enumerate() {
            total += f64::from(time(i, path)?.duration());
        }
        fastest = fastest.min(start.elapsed().as_secs_f64());
    }
    let count = paths.len() as f64;
    Ok((total / count, 1e6 * fastest / count))
}

fn main() -> Result<()> {
    let robot: Robot = common::panda()?;
    let n = robot.dof();
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;
    let mut rng = Rng::new(7);
    let mut random = || -> Vec<f32> {
        (0..n)
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect()
    };
    let problems: Vec<PlanProblem> = (0..512)
        .map(|_| PlanProblem::new(0, random(), random()))
        .collect();
    let plans = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default())?;
    let planned: Vec<Vec<f32>> = plans.solved().map(|s| s.solution.to_vec()).collect();

    // Halfway along the first 64, moving, replanned to a new goal.
    let mut state = JointState::new(n);
    let mut moving = vec![];
    let mut motions = vec![];
    for path in planned.iter().take(64) {
        let trajectory = Trajectory::try_new(&robot, path, 1.0)?;
        trajectory.at(0.5 * trajectory.duration(), &mut state);
        let motion = StartMotion {
            velocity: state.velocity.clone(),
            acceleration: state.acceleration.clone(),
        };
        let problem = PlanProblem {
            start_motion: Some(motion.clone()),
            ..PlanProblem::new(0, state.position.clone(), random())
        };
        let replanned = batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default())?;
        if let Some(path) = replanned.best(0) {
            moving.push(path.to_vec());
            motions.push(motion);
        }
    }

    // Straight moves of the hand from the planned paths' ends: 5 to 20 cm in random directions.
    let mut rng = Rng::new(8);
    let moves: Vec<ToolPath> = planned
        .iter()
        .take(128)
        .map(|p| {
            let start = p[p.len() - n..].to_vec();
            let pose = robot.ee_pose(&start, 0);
            let direction = Vec3::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            );
            let to = pose.position + direction.normalize() * rng.range(0.05, 0.2);
            ToolPath {
                world: 0,
                start,
                hand: 0,
                waypoints: vec![Pose::new(to, pose.rotation)],
            }
        })
        .collect();
    let followed = batchplan::follow(&device, &worlds, &moves, &ToolPathOptions::default())?;
    let straight: Vec<Vec<f32>> = followed
        .paths
        .into_iter()
        .filter(|p| p.len() / n >= 7)
        .collect();

    let rest = |_: usize, p: &[f32]| Ok(Trajectory::try_new(&robot, p, 1.0)?);
    println!(
        "{:<34}{:>8}{:>14}{:>12}",
        "", "paths", "duration s", "µs each"
    );
    for (kind, paths) in [
        ("planned from rest", &planned),
        ("straight moves of the hand", &straight),
    ] {
        let (duration, cost) = timed(paths, rest)?;
        println!("{kind:<34}{:>8}{duration:>14.4}{cost:>12.1}", paths.len());
    }
    let (duration, cost) = timed(&moving, |i, p| {
        Ok(Trajectory::moving(&robot, p, &motions[i])?)
    })?;
    println!(
        "{:<34}{:>8}{duration:>14.4}{cost:>12.1}",
        "replanned while moving",
        moving.len()
    );
    Ok(())
}
