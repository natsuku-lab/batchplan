//! Robot loading and sphere fitting, with collision-model checksums and qualification outside the
//! timed calls. Set self-collision samples to zero to isolate fitting from self-collision analysis.
//!
//!   cargo run --release --example fitting -- [threads] [model] [budget] [samples] [passes]
//!
//! Models: stand, panda, so101, ur5e. Without a model or budget, runs each at several budgets.
//! Defaults: one thread, 10,000 self-collision samples, six passes per fit.
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use rayon::ThreadPoolBuilder;

use batchplan::{Robot, RobotOptions, SphereOptions};

fn hash(robot: &Robot) -> Result<u64> {
    let mut value = 0xcbf29ce484222325u64;
    for byte in serde_json::to_vec(&robot.collision_model())? {
        value = (value ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    Ok(value)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let threads = args.first().map_or(Ok(1usize), |s| s.parse())?;
    let only = args.get(1).map(String::as_str);
    anyhow::ensure!(threads > 0, "threads must be positive");
    anyhow::ensure!(
        only.is_none_or(|name| matches!(name, "stand" | "panda" | "so101" | "ur5e")),
        "unknown model: {}",
        only.unwrap_or_default()
    );
    let budget = args.get(2).map(|s| s.parse::<usize>()).transpose()?;
    let samples = args.get(3).map_or(Ok(10_000usize), |s| s.parse())?;
    let repeats = args.get(4).map_or(Ok(6usize), |s| s.parse())?;
    anyhow::ensure!(repeats > 0, "repeats must be positive");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
    pool.install(|| -> Result<()> {
        for (name, path, strict) in [
            ("stand", root.join("assets/fitting/stand.urdf"), 512),
            ("panda", root.join("assets/franka/franka_panda.urdf"), 1200),
            (
                "so101",
                root.join("assets/so101/so101_new_calib.urdf"),
                700,
            ),
            (
                "ur5e",
                root.join("assets/ur5e/ur_description/urdf/ur5e.urdf"),
                1200,
            ),
        ] {
            if only.is_some_and(|s| s != name) {
                continue;
            }
            let budgets = budget.map_or_else(
                || {
                    if name == "stand" {
                        vec![384, 512, 1024]
                    } else {
                        vec![64, 256, strict, 4096]
                    }
                },
                |b| vec![b],
            );
            for budget in budgets {
                let mut options = RobotOptions {
                    spheres: SphereOptions {
                        budget,
                        self_collision_samples: samples,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                if name == "stand" {
                    options.sphere_tolerances =
                        [("feet".to_string(), 0.02), ("box".to_string(), 0.02)].into();
                }
                let reference = Robot::load(&path, &options)?;
                let expected = hash(&reference)?;
                let count: usize = reference.collision_model().spheres.values().map(Vec::len).sum();
                anyhow::ensure!(count <= budget, "the fit exceeded its budget");
                let qualification = reference.qualify()?;
                let overhang = qualification
                    .links
                    .iter()
                    .map(|l| l.overhang)
                    .fold(0.0f32, f32::max);
                let mut times = Vec::new();
                for _ in 0..repeats {
                    let started = Instant::now();
                    let robot = Robot::load(&path, &options)?;
                    times.push(started.elapsed().as_secs_f64() * 1e3);
                    anyhow::ensure!(hash(&robot)? == expected, "repeated fitting changed the model");
                }
                times.sort_by(f64::total_cmp);
                println!(
                    "fit {name} threads={threads} budget={budget} self_collision_samples={samples} spheres={count} overhang={overhang:.9} best_ms={:.6} q1_ms={:.6} median_ms={:.6} checksum={expected:016x}",
                    times[0],
                    times[times.len() / 4],
                    times[times.len() / 2]
                );
            }
        }
        Ok(())
    })
}
