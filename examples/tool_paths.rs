//! How fast `follow` moves the Panda's hand along straight lines and pours, from its ready pose:
//! one 10 cm move at a time on one thread, 64 moves in random directions on every thread, and 64
//! pours (half a radian about a random axis while the hand moves 3 mm).
//!
//!   cargo run --release --example tool_paths

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use anyhow::Result;
use batchplan::glam::{Quat, Vec3};
use batchplan::rng::Rng;
use batchplan::{Device, Pose, ToolPath, ToolPathOptions, World};

/// The median of `runs` timings of `f`, in milliseconds, after `warm` more.
fn median(warm: usize, runs: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..warm {
        f()?;
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        f()?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    Ok(times[runs / 2])
}

fn main() -> Result<()> {
    let robot = common::panda()?;
    let mut rng = Rng::new(5);
    let start = robot.default_q().to_vec();
    let hand = robot.ee_pose(&start, 0);
    let mut direction = || Vec3::new(rng.normal(), rng.normal(), rng.normal()).normalize();
    let path = |to: Pose| ToolPath {
        world: 0,
        start: start.clone(),
        hand: 0,
        waypoints: vec![to],
    };
    let moves: Vec<ToolPath> = (0..64)
        .map(|_| path(Pose::new(hand.position + direction() * 0.1, hand.rotation)))
        .collect();
    let pours: Vec<ToolPath> = (0..64)
        .map(|_| {
            let (axis, shift) = (direction(), direction() * 0.003);
            let turn = Quat::from_axis_angle(axis, 0.5);
            path(Pose::new(hand.position + shift, turn * hand.rotation))
        })
        .collect();
    let o = ToolPathOptions::default();
    let one = Device::cpu_threads(&robot, 1)?;
    let all = Device::cpu(&robot)?;
    let (on_one, on_all) = (
        one.upload(&[World::default()])?,
        all.upload(&[World::default()])?,
    );
    let follow = |device: &Device, worlds, paths: &[ToolPath]| -> Result<()> {
        batchplan::follow(device, worlds, paths, &o)?;
        Ok(())
    };
    // A move at a time: the median over the 64 moves of each one's median of five.
    let mut single: Vec<f64> = moves
        .iter()
        .map(|m| median(1, 5, || follow(&one, &on_one, std::slice::from_ref(m))))
        .collect::<Result<_>>()?;
    single.sort_by(f64::total_cmp);
    let batch = median(20, 200, || follow(&all, &on_all, &moves))?;
    let poured = median(20, 200, || follow(&all, &on_all, &pours))?;
    let covered = batchplan::follow(&all, &on_all, &pours, &o)?
        .fraction
        .iter()
        .filter(|&&f| f == 1.0)
        .count();
    println!(
        "{}: one move {:.4} ms on one thread; 64 moves {batch:.3} ms; 64 pours {poured:.3} ms \
         ({covered} covered)",
        all.name(),
        single[single.len() / 2],
    );
    Ok(())
}
