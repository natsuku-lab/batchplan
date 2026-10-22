//! A planner feeding a fixed-rate control loop, shaped like a ros2_control or robotd controller.
//! A worker thread plans reaches and hands each checked trajectory over through a latest-value
//! slot. The 50 Hz loop never waits for it: each tick it takes a new trajectory if one is ready,
//! samples the current one (`Trajectory::at` does not allocate), and holds its pose when it has
//! nothing to run. The planner uses a CPU device with two threads of its own.
//!
//! cargo run --release --example control_loop -- [seconds=6]

#[path = "common/mod.rs"]
mod common;

use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use batchplan::rng::Rng;
use batchplan::{
    Device, IkOptions, IkProblem, JointState, PlanOptions, PlanProblem, Robot, Trajectory, World,
};

const PERIOD: Duration = Duration::from_millis(20);

/// Plans a reach from each requested start to a random grasp target, until the requests stop.
fn planner(
    robot: Robot,
    scene: World,
    starts: mpsc::Receiver<Vec<f32>>,
    latest: Arc<Mutex<Option<Trajectory>>>,
) -> Result<()> {
    let device = Device::cpu_threads(&robot, 2)?;
    let worlds = device.upload(std::slice::from_ref(&scene))?;
    let mut rng = Rng::new(11);
    for start in starts {
        let t = Instant::now();
        let target = common::grasp_target(&scene, &mut rng);
        let ik = batchplan::solve_ik(
            &device,
            &worlds,
            &[IkProblem::new(0, target)],
            &IkOptions::default(),
        )?;
        let Some(goal) = ik.best(0) else { continue };
        let problem = PlanProblem::new(0, start, goal);
        let options = PlanOptions {
            time_budget: Some(Duration::from_millis(500)),
            ..Default::default()
        };
        let result = batchplan::plan(&device, &worlds, &[problem], &options)?;
        let Some(path) = result.best(0) else { continue };
        let trajectory = Trajectory::new(&robot, path, 0.8)?;
        // The controller only ever receives trajectories that pass the safety check.
        trajectory.check(&robot)?;
        println!(
            "planner: {:.2} s reach planned in {:.0} ms",
            trajectory.duration(),
            t.elapsed().as_secs_f64() * 1e3
        );
        *latest
            .lock()
            .expect("the controller never panics holding the slot") = Some(trajectory);
    }
    Ok(())
}

fn main() -> Result<()> {
    let seconds: f64 = std::env::args().nth(1).map_or(Ok(6.0), |a| a.parse())?;
    let robot = common::panda()?;
    let scene = common::tabletop(&mut Rng::new(4));
    let latest: Arc<Mutex<Option<Trajectory>>> = Arc::new(Mutex::new(None));
    let (request, starts) = mpsc::channel();
    let worker = {
        let (robot, latest) = (robot.clone(), latest.clone());
        thread::spawn(move || planner(robot, scene, starts, latest))
    };

    let mut state = JointState::new(robot.dof());
    let mut command = robot.default_q().to_vec();
    let mut running: Option<(Trajectory, Instant)> = None;
    request.send(command.clone())?;
    let (mut ticks, mut holding, mut executed, mut worst_late) = (0, 0, 0, Duration::ZERO);
    let start = Instant::now();
    let mut next = start;
    while start.elapsed().as_secs_f64() < seconds {
        let now = Instant::now();
        worst_late = worst_late.max(now.saturating_duration_since(next));
        // Take a new trajectory if the planner has finished one; never wait for the lock.
        if let Ok(mut slot) = latest.try_lock()
            && let Some(trajectory) = slot.take()
        {
            running = Some((trajectory, now));
        }
        match &running {
            Some((trajectory, began)) => {
                let t = now.duration_since(*began).as_secs_f32();
                trajectory.at(t, &mut state);
                command.copy_from_slice(&state.position);
                if t >= trajectory.duration() {
                    running = None;
                    executed += 1;
                    request.send(command.clone())?;
                }
            }
            None => holding += 1,
        }
        // Here a real controller writes `command` (and `state.velocity`) to the hardware.
        ticks += 1;
        next += PERIOD;
        thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    drop(request);
    worker.join().expect("the planner thread panicked")?;
    println!(
        "{ticks} ticks at {} Hz: {executed} reaches executed, {holding} ticks holding, worst tick {:.2} ms late",
        1000 / PERIOD.as_millis(),
        worst_late.as_secs_f64() * 1e3
    );
    Ok(())
}
