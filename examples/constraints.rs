//! Plans that keep the Panda's gripper upright, as a cup or a tray must be carried: above a floor,
//! past a pillar, to upright goals at random turns about the vertical. How many solve, how long
//! they take, alone and in one batch, and the most the gripper tilts along each path, sampled
//! densely once timed; then the same problems without the constraint, for comparison, and
//! optimized with 4 seeds. `--zone` also keeps the gripper inside the zone the goals are drawn
//! from. Last, two Pandas carry a tray level between them over a block, the right hand pinned in
//! the left hand's frame.
//!
//!   cargo run --release --example constraints               # 64 problems, within 5°
//!   cargo run --release --example constraints -- 256 10     # problems, the bound in degrees
//!   cargo run --release --example constraints -- 64 5 --zone

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use anyhow::Result;
use batchplan::glam::Vec3;
use batchplan::rng::Rng;
use batchplan::{
    Device, HandConstraint, PlanOptions, PlanProblem, PlanResult, Pose, Robot, Trajectory, Worlds,
};

/// Each solved path's largest tilt of hand 0 from straight down, in degrees, sampled at 1 kHz
/// once timed.
fn tilts(robot: &Robot, result: &PlanResult) -> Result<Vec<f32>> {
    let mut out = vec![];
    for s in result.solved() {
        let samples = Trajectory::try_new(robot, s.solution, 1.0)?.sample(1000.0)?;
        let most = samples
            .positions
            .chunks(samples.dof)
            .map(|q| (robot.ee_pose(q, 0).rotation * Vec3::Z).angle_between(Vec3::NEG_Z))
            .fold(0.0, f32::max);
        out.push(most.to_degrees());
    }
    Ok(out)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f32::NAN)
}

/// Plans `problems` in one batch and each alone: the batch's result, its time and the median time
/// alone (ms).
fn timed(
    device: &Device,
    worlds: &Worlds,
    problems: &[PlanProblem],
    o: &PlanOptions,
) -> Result<(PlanResult, f32, f32)> {
    let began = Instant::now();
    let batch = batchplan::plan(device, worlds, problems, o)?;
    let batched = began.elapsed().as_secs_f32() * 1e3;
    let mut alone = vec![];
    for p in problems {
        let began = Instant::now();
        batchplan::plan(device, worlds, std::slice::from_ref(p), o)?;
        alone.push(began.elapsed().as_secs_f32() * 1e3);
    }
    Ok((batch, batched, median(alone)))
}

/// The mean length of the shortest valid path of each solved problem.
fn mean_length(result: &PlanResult) -> f32 {
    let lengths: Vec<f32> = (0..result.problems.len())
        .filter_map(|p| {
            let seeds = p * result.seeds..(p + 1) * result.seeds;
            seeds
                .filter(|&k| result.valid[k])
                .map(|k| result.length[k])
                .min_by(f32::total_cmp)
        })
        .collect();
    lengths.iter().sum::<f32>() / lengths.len() as f32
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let numbers: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let count: usize = numbers.first().map_or(Ok(64), |a| a.parse())?;
    let tilt: f32 = numbers.get(1).map_or(Ok(5.0), |a| a.parse())?;
    let zone = args.iter().any(|a| a == "--zone");
    let robot = common::panda()?;
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[common::pillar_scene()])?;
    let (start, goals) = common::upright_reaches(&device, &worlds, count, &mut Rng::new(1234))?;
    let middle = Pose::new(
        0.5 * (common::ZONE_MIN + common::ZONE_MAX),
        common::upright(0.0),
    );
    let upright = HandConstraint {
        position: if zone {
            0.5 * (common::ZONE_MAX - common::ZONE_MIN)
        } else {
            Vec3::INFINITY
        },
        tilt: tilt.to_radians(),
        ..HandConstraint::new(0, middle)
    };
    println!(
        "{} upright goals of {count} in reach; the gripper within {tilt}° of straight down{}",
        goals.len(),
        if zone { ", inside the zone" } else { "" }
    );
    let o = PlanOptions::default();
    for constraints in [vec![upright], vec![]] {
        let problems: Vec<PlanProblem> = goals
            .iter()
            .map(|goal| PlanProblem {
                constraints: constraints.clone(),
                ..PlanProblem::new(0, start.clone(), goal.clone())
            })
            .collect();
        let (batch, batched, alone) = timed(&device, &worlds, &problems, &o)?;
        let tilts = tilts(&robot, &batch)?;
        println!(
            "{:>13}: {} of {} solved, {alone:.2} ms each alone (median), {batched:.1} ms in one batch; tilt at most {:.2}°, {:.2}° on the median path; {:.3} rad long on average",
            if constraints.is_empty() {
                "unconstrained"
            } else {
                "upright"
            },
            batch.solved().count(),
            problems.len(),
            tilts.iter().copied().fold(0.0, f32::max),
            median(tilts),
            mean_length(&batch),
        );
        if !constraints.is_empty() {
            let optimizing = PlanOptions { seeds: 4, ..o };
            let began = Instant::now();
            let optimized = batchplan::plan(&device, &worlds, &problems, &optimizing)?;
            let took = began.elapsed().as_secs_f32() * 1e3;
            let improved = (0..problems.len())
                .filter(|&p| {
                    let base = p * optimized.seeds;
                    (1..optimized.seeds).any(|k| {
                        optimized.valid[base + k]
                            && optimized.length[base + k] < optimized.length[base]
                    })
                })
                .count();
            println!(
                "{:>13}: {improved} of {} shorter, {:.3} rad long on average, {took:.1} ms in one batch; tilt at most {:.2}°",
                "optimized",
                problems.len(),
                mean_length(&optimized),
                tilts_of(&robot, &optimized)?,
            );
        }
    }

    // Two arms carrying a tray level over a block, the right hand pinned in the left's frame.
    let two = common::two_pandas()?;
    let device = Device::cpu(&two)?;
    let worlds = device.upload(&[common::tray_scene()])?;
    let carries = common::tray_carries(&device, &worlds)?;
    let carried = common::carried_level(carries.place, tilt.to_radians()).to_vec();
    let problems: Vec<PlanProblem> = carries
        .goals
        .iter()
        .map(|goal| PlanProblem {
            constraints: carried.clone(),
            ..PlanProblem::new(0, carries.start.clone(), goal.clone())
        })
        .collect();
    let (batch, batched, alone) = timed(&device, &worlds, &problems, &o)?;
    let optimized = batchplan::plan(&device, &worlds, &problems, &PlanOptions { seeds: 4, ..o })?;
    println!(
        "two arms carrying a tray level: {} of {} solved, {alone:.2} ms each alone (median), {batched:.1} ms in one batch; {:.3} rad long on average, {:.3} optimized",
        batch.solved().count(),
        problems.len(),
        mean_length(&batch),
        mean_length(&optimized),
    );
    Ok(())
}

/// The largest tilt over every solved path (`tilts`).
fn tilts_of(robot: &Robot, result: &PlanResult) -> Result<f32> {
    Ok(tilts(robot, result)?.into_iter().fold(0.0, f32::max))
}
