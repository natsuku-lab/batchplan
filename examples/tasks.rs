//! Picks and places on random tabletops, on every device and at several batch sizes: how many
//! tasks get through, where the others stop, and how long a call takes.
//!
//!   cargo run --release --example tasks             # 64 scenes
//!   cargo run --release --example tasks -- 256      # more

#[path = "common/mod.rs"]
mod common;

use std::f32::consts::PI;
use std::time::Instant;

use anyhow::Result;
use batchplan::glam::{Quat, Vec3};
use batchplan::rng::Rng;
use batchplan::{
    Device, GraspOptions, Obstacle, Pick, Place, Pose, TaskFailure, TaskOptions, TaskPlan,
    TaskResult, TaskStep, World,
};

/// Calls of each size timed, and the best of so many runs kept.
const BATCHES: [usize; 3] = [1, 16, 64];
const RUNS: usize = 5;

/// A table with 2-5 boxes on it, a box to move among them (obstacle 1), and a free spot for it.
fn scene(rng: &mut Rng) -> (World, Pose) {
    let table = Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let half = Vec3::new(
        rng.range(0.015, 0.035),
        rng.range(0.015, 0.035),
        rng.range(0.02, 0.05),
    );
    let at = |rng: &mut Rng, half: Vec3| {
        Pose::new(
            Vec3::new(rng.range(0.3, 0.6), rng.range(-0.4, 0.4), half.z),
            Quat::from_rotation_z(rng.range(-PI, PI)),
        )
    };
    let target = at(rng, half);
    let mut obstacles = vec![table, Obstacle::cuboid(Vec3::ZERO, half).placed(target)];
    let clear =
        |p: Vec3, obstacles: &[Obstacle]| obstacles[1..].iter().all(|o| o.distance(p).value > 0.12);
    while obstacles.len() < 2 + 2 + (rng.uniform() * 4.0) as usize {
        let size = Vec3::new(
            rng.range(0.02, 0.06),
            rng.range(0.02, 0.06),
            rng.range(0.03, 0.15),
        );
        let pose = at(rng, size);
        if clear(pose.position, &obstacles) {
            obstacles.push(Obstacle::cuboid(Vec3::ZERO, size).placed(pose));
        }
    }
    loop {
        let spot = at(rng, half);
        if clear(spot.position, &obstacles) {
            return (World::new(obstacles), spot);
        }
    }
}

/// How many tasks got through, and where the others' candidates stopped, by step.
fn tally(outcomes: &[std::result::Result<Vec<TaskPlan>, TaskFailure>]) -> String {
    let done = outcomes.iter().filter(|o| o.is_ok()).count();
    let failures: Vec<&TaskFailure> = outcomes.iter().filter_map(|o| o.as_ref().err()).collect();
    let steps = [
        (TaskStep::Start, "start"),
        (TaskStep::Reach, "reach"),
        (TaskStep::Approach, "in"),
        (TaskStep::Depart, "away"),
        (TaskStep::Transit, "transit"),
    ];
    let furthest: Vec<String> = steps
        .iter()
        .map(|&(step, name)| {
            let count = failures.iter().filter(|f| f.step() == step).count();
            format!("{name} {count}")
        })
        .collect();
    format!(
        "{done}/{} through; failed tasks by furthest step: {}",
        outcomes.len(),
        furthest.join(", ")
    )
}

/// The best of `RUNS` timings of `f`, in milliseconds, and its last result.
fn timed<T>(mut f: impl FnMut() -> Result<T>) -> Result<(f64, T)> {
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..RUNS {
        let t = Instant::now();
        last = Some(f()?);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    Ok((best, last.expect("at least one run")))
}

fn main() -> Result<()> {
    let count: usize = std::env::args()
        .nth(1)
        .map_or(Ok(64), |a| a.parse())?
        .max(BATCHES[BATCHES.len() - 1]);
    let robot = common::panda()?;
    let mut rng = Rng::new(7);
    let (worlds, spots): (Vec<World>, Vec<Pose>) = (0..count).map(|_| scene(&mut rng)).unzip();
    let o = TaskOptions::default();
    let mut devices = vec![Device::cpu(&robot)?];
    match Device::gpu(&robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) => println!("no GPU: {e}"),
    }
    for device in devices {
        println!("{}", device.name());
        let uploaded = device.upload(&worlds)?;
        let picks: Vec<Pick> = worlds
            .iter()
            .enumerate()
            .map(|(w, world)| {
                let grasps = batchplan::grasps(
                    &world.obstacles[1].placed(Pose::IDENTITY),
                    &GraspOptions::default(),
                )?;
                Ok(Pick {
                    support: vec![0],
                    ..Pick::new(w as u32, robot.default_q(), 1, grasps)
                })
            })
            .collect::<Result<_>>()?;
        // A first call compiles the GPU's kernels; time the calls after it.
        batchplan::pick(&device, &uploaded, &picks[..1], &o)?;
        for size in BATCHES {
            let (ms, result) =
                timed(|| Ok(batchplan::pick(&device, &uploaded, &picks[..size], &o)?))?;
            println!(
                "  pick {size:>3}: {ms:8.1} ms, {:6.2} ms a task; {}",
                ms / size as f64,
                tally(&result.outcomes)
            );
        }
        let picked = batchplan::pick(&device, &uploaded, &picks, &o)?;
        println!("  pick {count:>3}: {}", tally(&picked.outcomes));
        if let Some(f) = picked.outcomes.iter().find_map(|o| o.as_ref().err()) {
            println!("    e.g. {f}");
        }
        // Set each picked box down at its spot, the world without it.
        let (mut after, mut places) = (vec![], vec![]);
        for (w, (outcome, spot)) in picked.outcomes.iter().zip(&spots).enumerate() {
            let Ok(plans) = outcome else { continue };
            let mut world = worlds[w].clone();
            let object = world.remove(1)?;
            after.push(world);
            let held = plans[0]
                .held
                .clone()
                .expect("a pick leaves its object held");
            places.push(Place {
                support: vec![0],
                ..Place::new(
                    places.len() as u32,
                    plans[0].end(),
                    held,
                    batchplan::place_poses(&object, *spot, 8),
                )
            });
        }
        let after = device.upload(&after)?;
        for size in BATCHES.into_iter().filter(|&s| s <= places.len()) {
            let (ms, result): (f64, TaskResult<Place>) =
                timed(|| Ok(batchplan::place(&device, &after, &places[..size], &o)?))?;
            println!(
                "  place {size:>2}: {ms:8.1} ms, {:6.2} ms a task; {}",
                ms / size as f64,
                tally(&result.outcomes)
            );
        }
        let placed = batchplan::place(&device, &after, &places, &o)?;
        println!("  place {:>2}: {}", places.len(), tally(&placed.outcomes));
        if let Some(f) = placed.outcomes.iter().find_map(|o| o.as_ref().err()) {
            println!("    e.g. {f}");
        }
    }
    Ok(())
}
