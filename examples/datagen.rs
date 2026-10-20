//! Generates policy-training demonstrations: reach-to-grasp trajectories in random cluttered
//! worlds plus recoveries from perturbed mid-path states, timed within the robot's limits at
//! randomized speed; or, with `--pick-place`, boxes picked up and set down across a table.
//!
//! cargo run --release --example datagen -- [--pick-place] <out_dir> [episodes=256] [fps=20]
//!     writes plain NumPy `.npy` arrays
//! cargo run --release --features lerobot --example datagen -- --lerobot [--file-mb N] <out_dir> [episodes] [fps]
//!     writes a LeRobot v3.0 dataset, starting new files past N MB (default 100)

#[path = "common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use batchplan::datagen::{DemoOptions, Demonstration, Origin, PickPlaceOptions, demonstrations, pick_and_place};
use batchplan::rng::Rng;
use batchplan::*;

fn main() -> Result<()> {
    let (mut lerobot, mut pick_place, mut file_mb, mut positional) = (false, false, 100, vec![]);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--lerobot" => lerobot = true,
            "--pick-place" => pick_place = true,
            "--file-mb" => file_mb = args.next().context("--file-mb N")?.parse()?,
            _ => positional.push(arg),
        }
    }
    let mut args = positional.into_iter();
    let out = PathBuf::from(
        args.next().context("usage: datagen [--pick-place] [--lerobot [--file-mb N]] <out_dir> [episodes] [fps]")?,
    );
    let episodes: usize = args.next().map_or(Ok(256), |a| a.parse())?;
    let fps: u32 = args.next().map_or(Ok(20), |a| a.parse())?;

    let robot = common::panda()?;
    let device = match Device::gpu(&robot) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU ({e}); using the CPU");
            Device::cpu(&robot)?
        }
    };
    println!("device: {}", device.name());
    let mut rng = Rng::new(7);
    let started = Instant::now();
    let (worlds, demos) = if pick_place {
        let (worlds, tasks): (Vec<World>, Vec<_>) =
            (0..episodes).map(|i| common::pick_place_scene(i as u32, &mut rng)).unzip();
        let options = PickPlaceOptions { dt: 1.0 / fps as f32, ..Default::default() };
        let demos = pick_and_place(&device, &worlds, &tasks, &options)?;
        println!(
            "{episodes} tasks -> {} pick-and-place demonstrations in {:.2} s",
            demos.len(),
            started.elapsed().as_secs_f64()
        );
        (worlds, demos)
    } else {
        let worlds: Vec<World> = (0..episodes).map(|_| common::tabletop(&mut rng)).collect();
        let goals: Vec<IkProblem> = worlds
            .iter()
            .enumerate()
            .map(|(i, w)| IkProblem::new(i as u32, common::grasp_target(w, &mut rng)))
            .collect();
        let options = DemoOptions { dt: 1.0 / fps as f32, ..Default::default() };
        let demos = demonstrations(&device, &device.upload(&worlds)?, &goals, &options)?;
        let recoveries = demos.iter().filter(|d| matches!(d.origin, Origin::Recovery { .. })).count();
        println!(
            "{episodes} worlds -> {} nominal + {recoveries} recovery demonstrations in {:.2} s",
            demos.len() - recoveries,
            started.elapsed().as_secs_f64()
        );
        (worlds, demos)
    };

    if lerobot {
        export_lerobot(&out, &robot, &worlds, &demos, file_mb)?;
    } else {
        let metadata = serde_json::json!({
            "max_acceleration": robot.max_acceleration(),
            "max_jerk": robot.max_jerk(),
            "device": device.name(),
        });
        let metadata = metadata.as_object().expect("an object").clone();
        batchplan::npy::export(&out, &robot, &worlds, &demos, &batchplan::npy::ExportOptions { metadata })?;
    }
    println!("wrote {} episodes to {}", demos.len(), out.display());
    Ok(())
}

#[cfg(feature = "lerobot")]
fn export_lerobot(out: &Path, robot: &Robot, worlds: &[World], demos: &[Demonstration], file_mb: usize) -> Result<()> {
    let options = batchplan::lerobot::ExportOptions { data_files_size_in_mb: file_mb, ..Default::default() };
    Ok(batchplan::lerobot::export(out, robot, worlds, demos, &options)?)
}

#[cfg(not(feature = "lerobot"))]
fn export_lerobot(_: &Path, _: &Robot, _: &[World], _: &[Demonstration], _: usize) -> Result<()> {
    anyhow::bail!("rebuild with `--features lerobot` to write LeRobot datasets")
}
