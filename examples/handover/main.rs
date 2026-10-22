//! Two reBot B601 arms hand blocks to each other: the bimanual benchmark and data generator.
//!
//! One arm picks a block up, both arms move at once to where they meet, the other takes it and
//! sets it down, every step batched over all the tasks (`datagen::handover`). Half the tasks go
//! left to right and half right to left. It prints how many succeed and how fast, and can write a
//! LeRobot dataset laid out arm by arm, which `scripts/replay_mujoco.py` renders camera videos for,
//! or the episodes and states perturbed from them as `.npy` arrays, a reset bank for reinforcement
//! learning (`datagen::reset_states`).
//!
//! scripts/fetch_rebot.sh
//! cargo run --release --example handover -- [tasks=512] [--cpu]
//! cargo run --release --features lerobot --example handover -- [tasks] --lerobot <out_dir>
//! cargo run --release --example handover -- [tasks] --resets <out_dir>

mod rebot;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use batchplan::datagen::{self, Demonstration, Handover, RecoveryOptions};
use batchplan::rng::Rng;
use batchplan::{Device, Robot, World, npy};

/// Where `scripts/fetch_rebot.sh` puts the arm.
const REBOT: &str = "data/rebot";

fn main() -> Result<()> {
    use anyhow::Context;

    let (mut cpu, mut lerobot, mut resets, mut positional) = (false, None, None, vec![]);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--cpu" => cpu = true,
            "--lerobot" => {
                lerobot = Some(PathBuf::from(args.next().context("--lerobot <out_dir>")?))
            }
            "--resets" => resets = Some(PathBuf::from(args.next().context("--resets <out_dir>")?)),
            _ => positional.push(arg),
        }
    }
    let count: usize = positional.first().map_or(Ok(512), |a| a.parse())?;

    let robot = rebot::two_rebots(Path::new(REBOT))?;
    let device = if cpu {
        Device::cpu(&robot)?
    } else {
        Device::gpu(&robot).or_else(|_| Device::cpu(&robot))?
    };
    println!(
        "two reBot B601-DM arms, {} joints; {}",
        robot.dof(),
        device.name()
    );

    let mut rng = Rng::new(5);
    let (worlds, tasks): (Vec<World>, Vec<Handover>) =
        (0..count).map(|i| rebot::scene(i as u32, &mut rng)).unzip();
    let options = rebot::options(30.0);
    // The first call on a GPU compiles its kernels; time the second.
    datagen::handover(
        &device,
        &worlds[..8.min(count)],
        &tasks[..8.min(count)],
        &options,
    )?;
    let started = Instant::now();
    let demos = datagen::handover(&device, &worlds, &tasks, &options)?;
    let seconds = started.elapsed().as_secs_f64();
    let durations: Vec<f32> = demos.iter().map(|d| d.trajectory.duration).collect();
    let motion: f32 = durations.iter().sum();
    let mut sorted = durations;
    sorted.sort_by(f32::total_cmp);
    println!(
        "{} of {count} handovers in {seconds:.2} s ({:.0} episodes/s, {:.1} ms each): {:.0} minutes of motion, \
         episodes {:.1} s median, {:.1} s longest",
        demos.len(),
        demos.len() as f64 / seconds,
        seconds * 1e3 / demos.len().max(1) as f64,
        motion / 60.0,
        sorted.get(sorted.len() / 2).copied().unwrap_or(0.0),
        sorted.last().copied().unwrap_or(0.0),
    );
    if let Some(out) = resets {
        let options = RecoveryOptions {
            per_trajectory: 8,
            phase: (0.0, 1.0),
            ..Default::default()
        };
        let started = Instant::now();
        let states = datagen::reset_states(&device, &worlds, &demos, &options)?;
        println!(
            "{} of {} perturbed states collision-free in {:.2} s",
            states.len(),
            demos.len() * options.per_trajectory,
            started.elapsed().as_secs_f64()
        );
        let episodes = out.join("episodes");
        npy::export(&episodes, &robot, &worlds, &demos, &Default::default())?;
        let metadata = serde_json::json!({"parents": "../episodes, which parent.npy indexes"});
        let metadata = metadata.as_object().expect("an object").clone();
        npy::export(
            out.join("resets"),
            &robot,
            &worlds,
            &states,
            &npy::ExportOptions { metadata },
        )?;
        println!(
            "wrote {} and {}",
            episodes.display(),
            out.join("resets").display()
        );
    }
    if let Some(out) = lerobot {
        export(&out, &robot, &worlds, &demos)?;
        println!(
            "wrote {}; render camera videos with: .venv/bin/python scripts/replay_mujoco.py {} {}_video",
            out.display(),
            out.display(),
            out.display()
        );
    }
    Ok(())
}

/// The episodes as a LeRobot dataset, with `replay.json` for `scripts/replay_mujoco.py`: each
/// gripper's fingers, and where the cameras are (one over the table, one on each wrist).
#[cfg(feature = "lerobot")]
fn export(out: &Path, robot: &Robot, worlds: &[World], demos: &[Demonstration]) -> Result<()> {
    use serde_json::json;
    let options = batchplan::lerobot::ExportOptions {
        robot_type: Some("rebot_b601_dm_bimanual".into()),
        ..Default::default()
    };
    batchplan::lerobot::export(out, robot, worlds, demos, &options)?;
    // The right finger mirrors the left one.
    let fingers = |prefix: &str| {
        json!([
            {"joint": format!("{prefix}finger_left"), "open": rebot::FINGERS_OPEN},
            {"joint": format!("{prefix}finger_right"), "open": -rebot::FINGERS_OPEN},
        ])
    };
    // Above the gripper on the side toward the arm's base, looking past the fingertips; the grasp
    // point's x axis points toward the base.
    let wrist = |prefix: &str| {
        json!({
            "frame": format!("{prefix}tcp"),
            "pos": [0.08, 0.0, -0.04],
            "target": [0.0, 0.0, 0.08],
            "up": [-1.0, 0.0, 0.0],
            "fovy": 75,
        })
    };
    let replay = json!({
        "grippers": {"left_gripper": fingers("left_"), "right_gripper": fingers("right_")},
        "cameras": {
            "cam_high": {"pos": [-0.05, 0.0, 0.75], "target": [0.3, 0.0, 0.0], "fovy": 60},
            "cam_left_wrist": wrist("left_"),
            "cam_right_wrist": wrist("right_"),
        },
    });
    std::fs::write(
        out.join("meta/replay.json"),
        serde_json::to_string_pretty(&replay)?,
    )?;
    Ok(())
}

#[cfg(not(feature = "lerobot"))]
fn export(_: &Path, _: &Robot, _: &[World], _: &[Demonstration]) -> Result<()> {
    anyhow::bail!("rebuild with `--features lerobot` to write LeRobot datasets")
}
