//! How long one call takes on a device, by kind of call, on the Panda: IK of 1 to 1,000 targets
//! without collision spheres, with them and no push away from obstacles, and with the push;
//! `evaluate` of 512 and 16,384 configurations; plans of 16 and 256 problems with 4 optimized
//! seeds. Each kind is called many times and its median and best kept: a GPU's pace over calls
//! this short varies, so compare builds by running them in turn.
//!
//!   cargo run --release --example latency [-- --cpu]

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use anyhow::Result;
use batchplan::rng::Rng;
use batchplan::{
    CollisionModel, CollisionWeights, Device, IkOptions, IkProblem, PlanOptions, PlanProblem,
    Robot, RobotOptions, World,
};

/// The median and the best of `calls` runs of `call`, in milliseconds.
fn timed(calls: usize, mut call: impl FnMut() -> Result<()>) -> Result<(f64, f64)> {
    let mut times = Vec::with_capacity(calls);
    for _ in 0..calls {
        let started = Instant::now();
        call()?;
        times.push(started.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    Ok((times[calls / 2], times[0]))
}

fn main() -> Result<()> {
    let cpu = std::env::args().any(|a| a == "--cpu");
    let device = |robot: &Robot| {
        if cpu {
            Device::cpu(robot)
        } else {
            Device::gpu(robot)
        }
    };
    let bare = Robot::load(
        common::asset("franka/franka_panda.urdf"),
        &RobotOptions {
            collision_model: Some(CollisionModel::default()),
            ..common::panda_options()
        },
    )?;
    let panda = common::panda()?;
    let n = panda.dof();
    let mut rng = Rng::new(3);
    let mut random = |robot: &Robot| -> Vec<f32> {
        let joints = robot.lower().iter().zip(robot.upper());
        joints.map(|(&lo, &hi)| rng.range(lo, hi)).collect()
    };
    for (kind, robot, push) in [
        ("no spheres", &bare, true),
        ("spheres, no push", &panda, false),
        ("spheres and the push", &panda, true),
    ] {
        let device = device(robot)?;
        let worlds = device.upload(&[World::default()])?;
        let options = IkOptions {
            collision_step: if push {
                IkOptions::default().collision_step
            } else {
                0.0
            },
            ..IkOptions::default()
        };
        for count in [1, 10, 100, 1000] {
            let problems: Vec<IkProblem> = (0..count)
                .map(|_| IkProblem::new(0, robot.ee_pose(&random(robot), 0)))
                .collect();
            let mut solved = 0;
            let (median, best) = timed(60, || {
                solved = batchplan::solve_ik(&device, &worlds, &problems, &options)?
                    .solved()
                    .count();
                Ok(())
            })?;
            println!(
                "IK, {kind:<21} {count:>5} targets: median {median:8.3} ms, best {best:8.3}, {solved} solved"
            );
        }
    }
    let device = device(&panda)?;
    let mut scenes = Rng::new(7);
    let tabletops: Vec<World> = (0..64).map(|_| common::tabletop(&mut scenes)).collect();
    let worlds = device.upload(&tabletops)?;
    for count in [512, 16_384] {
        let q: Vec<f32> = (0..count).flat_map(|_| random(&panda)).collect();
        let item_world: Vec<u32> = (0..count).map(|i| (i % 64) as u32).collect();
        let weights = CollisionWeights::default();
        let (median, best) = timed(30, || {
            device.evaluate(&worlds, &item_world, &q, &weights)?;
            Ok(())
        })?;
        println!("evaluate {count:>6} configurations: median {median:8.3} ms, best {best:8.3}");
    }
    for count in [16, 256] {
        let problems: Vec<PlanProblem> = (0..count)
            .map(|i| PlanProblem::new((i % 64) as u32, panda.default_q(), random(&panda)))
            .collect();
        let options = PlanOptions {
            seeds: 4,
            ..PlanOptions::default()
        };
        let mut solved = 0;
        let (median, best) = timed(8, || {
            solved = batchplan::plan(&device, &worlds, &problems, &options)?
                .solved()
                .count();
            Ok(())
        })?;
        println!(
            "plans, {count:>3} problems of 4 optimized seeds: median {median:8.2} ms, best {best:8.2}, {solved} solved ({n} joints)"
        );
    }
    Ok(())
}
