//! Time budgets, and the devices' threads they depend on. Wall-clock times measured beside other
//! tests' threads swing by more than the slack the tests allow, so this file's tests run one at a
//! time, with no other test file beside them. GPU parts are skipped when no adapter is available
//! unless `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use std::sync::Mutex;

use batchplan::*;

const UR5E: &str = "ur5e/ur_description/urdf/ur5e.urdf";

/// Held by each test, so they run one at a time.
static ALONE: Mutex<()> = Mutex::new(());

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot).unwrap()];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

#[test]
fn the_search_keeps_the_time_budget() {
    let _alone = ALONE.lock().unwrap_or_else(|e| e.into_inner());
    // Fine steps make the search long enough to cut.
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let (scene, problems) = common::far_reaches(&robot, &cpu, 48, false);
    let slow = PlanOptions { rrt: RrtOptions { range: 0.02, ..Default::default() }, ..Default::default() };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let timed = |o: &PlanOptions| {
            let t = std::time::Instant::now();
            plan(&d, &worlds, &problems, o).unwrap();
            t.elapsed()
        };
        let full = timed(&slow);
        let budget = full / 4;
        let spent = timed(&PlanOptions { time_budget: Some(budget), ..slow });
        eprintln!("{}: full {full:?}, budget {budget:?} -> {spent:?}", d.name());
        assert!(
            full > std::time::Duration::from_millis(100),
            "{}: the search is too short to cut ({full:?})",
            d.name()
        );
        assert!(spent < budget + full / 8, "{}: took {spent:?} of a {budget:?} budget", d.name());
    }
}

#[test]
fn a_tiny_time_budget_returns_after_at_most_one_more_chunk() {
    let _alone = ALONE.lock().unwrap_or_else(|e| e.into_inner());
    let robot = common::panda().unwrap();
    let (scene, problems) = common::blocked_reaches(&robot);
    // Enough rounds (20 chunks) that a fast GPU's plan still lasts well beyond one chunk.
    let o = PlanOptions { seeds: 8, iterations: 160, ..Default::default() };
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let timed = |o: &PlanOptions| {
            let t = std::time::Instant::now();
            let result = plan(&d, &worlds, &problems, o).unwrap();
            (t.elapsed(), result)
        };
        timed(&o); // warm up
        let (full, _) = timed(&o);
        // A GPU checks the budget between submissions of 8 rounds; time one such chunk, with
        // seeding and validation.
        let (chunk, _) = timed(&PlanOptions { iterations: 8, ..o });
        let budget = std::time::Duration::from_millis(1);
        let (spent, result) = timed(&PlanOptions { time_budget: Some(budget), ..o });
        eprintln!("{}: full {full:?}, one chunk {chunk:?}, budget {budget:?} -> {spent:?}", d.name());
        assert!(spent < budget + 2 * chunk + std::time::Duration::from_millis(20), "{}: took {spent:?}", d.name());
        assert!(spent < full / 2, "{}: the budget saved little ({spent:?} of {full:?})", d.name());
        assert_eq!(result.valid.len(), problems.len() * (1 + o.seeds));
    }
}

/// The GPU device searches on threads of its own: a host keeping rayon's global pool busy does not
/// hold up its plans.
#[test]
fn a_busy_global_pool_does_not_hold_up_gpu_plans() {
    let _alone = ALONE.lock().unwrap_or_else(|e| e.into_inner());
    let robot = common::panda().unwrap();
    let gpu = match Device::gpu(&robot) {
        Ok(gpu) => gpu,
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => return eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    };
    let (scene, problems) = common::reaches(&robot);
    let worlds = gpu.upload(&scene).unwrap();
    plan(&gpu, &worlds, &problems, &PlanOptions::default()).unwrap();
    let busy = std::time::Duration::from_secs(1);
    let started = std::time::Instant::now();
    for _ in 0..rayon::current_num_threads() {
        rayon::spawn(move || std::thread::sleep(busy));
    }
    let result = plan(&gpu, &worlds, &problems, &PlanOptions::default()).unwrap();
    let spent = started.elapsed();
    assert!(spent < busy / 2, "planning waited for the global pool: {spent:?}");
    assert!(result.solved().count() * 2 > problems.len(), "only {} solved", result.solved().count());
}
