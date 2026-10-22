//! Batched IK + planning throughput on every available backend, on identical problems.
//!
//! cargo run --release --example bench -- [problems=512]
//! Set BENCH_LLVMPIPE=1 to also run the WGSL kernels on the CPU through Mesa's llvmpipe.

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use anyhow::Result;
use batchplan::rng::Rng;
use batchplan::{Device, IkOptions, IkProblem, PlanOptions, PlanProblem, World};

fn main() -> Result<()> {
    let count: usize = std::env::args().nth(1).map_or(Ok(512), |a| a.parse())?;
    let robot = common::panda()?;
    let mut rng = Rng::new(42);
    let worlds: Vec<World> = (0..count).map(|_| common::tabletop(&mut rng)).collect();
    let ik_problems: Vec<IkProblem> = worlds
        .iter()
        .enumerate()
        .map(|(i, w)| IkProblem::new(i as u32, common::grasp_target(w, &mut rng)))
        .collect();

    let mut devices = vec![];
    match Device::gpu(&robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) => eprintln!("no GPU device: {e}"),
    }
    devices.push(Device::cpu(&robot)?);
    if std::env::var("BENCH_LLVMPIPE").is_ok() {
        devices.push(Device::gpu_named(&robot, "llvmpipe")?);
    }

    let ik_opts = IkOptions::default();
    let plan_opts = PlanOptions::default();
    println!(
        "{count} worlds (table + 2-6 boxes), IK {} seeds x {} iters, plans searched, shortcut and smoothed ({} passes) into {} control points\n",
        ik_opts.seeds, ik_opts.iterations, plan_opts.shortcut.passes, plan_opts.control_points
    );
    for device in &devices {
        let worlds = device.upload(&worlds)?;
        // Warm up (pipeline compilation, thread pool).
        batchplan::solve_ik(
            device,
            &worlds,
            &ik_problems[..1],
            &IkOptions {
                iterations: 1,
                ..ik_opts
            },
        )?;

        let t = Instant::now();
        let ik = batchplan::solve_ik(device, &worlds, &ik_problems, &ik_opts)?;
        let ik_time = t.elapsed().as_secs_f64();
        let problems: Vec<PlanProblem> = ik
            .solved()
            .map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution))
            .collect();

        let t = Instant::now();
        let result = batchplan::plan(device, &worlds, &problems, &plan_opts)?;
        let plan_time = t.elapsed().as_secs_f64();
        let solved = result.solved().count();

        println!("{}", device.name());
        println!(
            "  IK    {:7.3} s  {:9.0} seeds/s   targets solved {}/{count}",
            ik_time,
            (count * ik_opts.seeds) as f64 / ik_time,
            problems.len()
        );
        println!(
            "  plan  {:7.3} s  {:9.0} problems/s   problems solved {solved}/{}",
            plan_time,
            problems.len() as f64 / plan_time,
            problems.len()
        );
        println!(
            "  end-to-end {:.1} solved problems/s\n",
            solved as f64 / (ik_time + plan_time)
        );
    }
    Ok(())
}
