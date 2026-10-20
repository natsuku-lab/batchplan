//! Live replanning, streamed to Rerun at 50 Hz.
//!
//! Each robot reaches from one grasp target to the next on its own table. About once a second a
//! box (red) appears in the path ahead of one to three moving robots. Every robot whose remaining
//! motion now collides replans from where it is, at its current velocity and acceleration, in one
//! batched `plan` call; its old path turns grey and the new one green. When it is moving too fast
//! to get around, it stops on its path (yellow, `Trajectory::stop`) and replans from rest. The event log lists
//! each replan and how long it took.
//!
//!   cargo run --release --bin replan             # 4 robots, 40 s
//!   cargo run --release --bin replan -- 16

use std::time::{Duration, Instant};

use anyhow::Result;
use batchplan::glam::Vec3;
use batchplan::rng::Rng;
use batchplan::*;
use batchplan_demos::*;

const SPACING: f32 = 2.4;
/// The control period, seconds.
const TICK: f32 = 0.02;
const SECONDS: f32 = 40.0;
/// Reaches run at this fraction of the Panda's speed.
const SPEED: f32 = 1.0;
/// How far ahead of the robot a box appears, as fractions of its remaining motion.
const AHEAD: (f32, f32) = (0.4, 0.65);
const BOX: Vec3 = Vec3::new(0.05, 0.05, 0.05);
/// Where a box waits until it is pushed in: far below the table.
const PARKED: Vec3 = Vec3::new(0.0, 0.0, -5.0);

/// One robot's current motion.
struct Motion {
    trajectory: Option<Trajectory>,
    /// When the trajectory started, in seconds.
    began: f32,
    /// The configuration it is heading to.
    goal: Vec<f32>,
    /// Where the robot is when it has no trajectory.
    hold: Vec<f32>,
    /// A goal to resume from rest, after braking short of a box it could not get around.
    resume: Option<Vec<f32>>,
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let count = args.count.unwrap_or(4);
    let robot = panda()?;
    // Replanning is a few problems at a time, where the CPU answers soonest.
    let device = args.device(&robot, false)?;
    let mut rng = Rng::new(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(21));
    let mut worlds: Vec<World> = (0..count)
        .map(|_| {
            let mut w = tabletop(&mut rng);
            w.obstacles.push(Obstacle::cuboid(PARKED, BOX));
            w
        })
        .collect();
    let mut uploaded = device.upload(&worlds)?;
    let n = robot.dof();

    let rec = args.recording("batchplan_replanning")?;
    rec.log_static(
        "summary",
        &rerun::TextDocument::from_markdown(format!(
            "## Replanning mid-motion\n\n{count} robots on {}, at full speed. Every 2-3 s a red box appears in the path ahead of \
             one to three of them; each robot whose motion now collides replans from its current position, velocity and \
             acceleration, all of them in one `plan` call. Red dot: where its old path (grey) would have hit the box. Green: the new path. Yellow: when no way around exists at its speed, a controlled stop on the path it is on, before it replans from rest.",
            device.name()
        )),
    )?;
    let meshes = panda_meshes(&rec, "world/panda", &robot)?;
    let offsets = grid(count, SPACING);
    for (i, w) in worlds.iter().enumerate() {
        let scene = format!("world/scene_{i}");
        rec.log_static(scene.as_str(), &rerun::Transform3D::from_translation(offsets[i].to_array()))?;
        let fixed: Vec<(usize, &Obstacle)> = w.obstacles.iter().enumerate().take(w.obstacles.len() - 1).collect();
        rec.log_static(format!("{scene}/obstacles"), &boxes(&fixed, |k| if k == 0 { wood() } else { slate() }))?;
    }

    let mut motions: Vec<Motion> = (0..count)
        .map(|_| Motion {
            trajectory: None,
            began: 0.0,
            goal: robot.default_q().to_vec(),
            hold: robot.default_q().to_vec(),
            resume: None,
        })
        .collect();
    let mut state = JointState::new(n);
    let mut next_push = 1.5;
    let mut label_until = vec![0.0f32; count];
    let clock = Instant::now();
    let ticks = (SECONDS / TICK) as usize;
    for tick in 0..ticks {
        let t = tick as f32 * TICK;
        rec.set_duration_secs("time", t);

        // Where every robot is now, and which have arrived.
        let mut now: Vec<JointState> = vec![];
        let mut arrived = vec![];
        for (i, m) in motions.iter_mut().enumerate() {
            match &m.trajectory {
                Some(tr) if t - m.began < tr.duration() => {
                    tr.at(t - m.began, &mut state);
                    now.push(state.clone());
                }
                _ => {
                    if let Some(tr) = m.trajectory.take() {
                        tr.at(tr.duration(), &mut state);
                        m.hold = state.position.clone();
                    }
                    now.push(JointState { position: m.hold.clone(), ..JointState::new(n) });
                    arrived.push(i);
                }
            }
        }

        // Robots at rest reach for a new target, all in one IK call and one plan call.
        if !arrived.is_empty() {
            let goals: Vec<IkProblem> =
                arrived.iter().map(|&i| IkProblem::new(i as u32, grasp_target(&worlds[i], &mut rng))).collect();
            let ik = solve_ik(&device, &uploaded, &goals, &IkOptions::default())?;
            let reaches: Vec<(usize, PlanProblem)> = arrived
                .iter()
                .enumerate()
                .filter_map(|(k, &i)| {
                    let goal = match motions[i].resume.take() {
                        Some(goal) => goal,
                        None => ik.best(k)?.to_vec(),
                    };
                    Some((i, PlanProblem::new(i as u32, now[i].position.clone(), goal)))
                })
                .collect();
            let problems: Vec<PlanProblem> = reaches.iter().map(|(_, p)| p.clone()).collect();
            let planned = plan(&device, &uploaded, &problems, &PlanOptions::default())?;
            for (k, (i, p)) in reaches.iter().enumerate() {
                let Some(path) = planned.best(k) else {
                    continue;
                };
                motions[*i] = Motion {
                    trajectory: Some(Trajectory::new(&robot, path, SPEED)?),
                    began: t,
                    goal: p.goal.clone(),
                    hold: p.start.clone(),
                    resume: None,
                };
                log_path(&rec, &robot, *i, "path", motions[*i].trajectory.as_ref(), 0.0, [40, 200, 90])?;
                log_path(&rec, &robot, *i, "old_path", None, 0.0, [0, 0, 0])?;
            }
        }

        // Every 2-3 s, a box appears ahead of a moving robot: the first one, in random order,
        // whose path it blocks without touching where the robot is or where it is going.
        if t >= next_push {
            let mut moving: Vec<usize> = (0..count)
                .filter(|&i| {
                    motions[i].trajectory.as_ref().is_some_and(|tr| tr.duration() - (t - motions[i].began) > 1.0)
                })
                .collect();
            let mut blocked = vec![];
            while !moving.is_empty() && blocked.is_empty() {
                let i = moving.swap_remove((rng.uniform() * moving.len() as f32) as usize % moving.len());
                let m = &motions[i];
                let tr = m.trajectory.as_ref().expect("a moving robot");
                let left = tr.duration() - (t - m.began);
                tr.at(t - m.began + left * rng.range(AHEAD.0, AHEAD.1), &mut state);
                let center = robot.ee_pose(&state.position, 0).position;
                let last = worlds[i].obstacles.len() - 1;
                let previous = worlds[i].obstacles[last].clone();
                worlds[i].obstacles[last] = Obstacle::cuboid(center, BOX);
                uploaded = device.upload(&worlds)?;
                let ends = device.evaluate(
                    &uploaded,
                    &[i as u32, i as u32],
                    &[now[i].position.clone(), m.goal.clone()].concat(),
                    &CollisionWeights::NONE,
                )?;
                let hit = if ends.collision_free(0) && ends.collision_free(1) {
                    collides_ahead(&device, &uploaded, &robot, m, i, t)
                } else {
                    None
                };
                match hit {
                    Some(hit) => {
                        rec.log(
                            format!("world/scene_{i}/pushed"),
                            &rerun::Boxes3D::from_centers_and_half_sizes([center.to_array()], [BOX.to_array()])
                                .with_colors([rerun::Color::from_rgb(230, 50, 50)])
                                .with_fill_mode(rerun::FillMode::Solid),
                        )?;
                        // Where the old path would have hit the box.
                        rec.log(
                            format!("world/scene_{i}/would_hit"),
                            &rerun::Points3D::new([hit.to_array()])
                                .with_radii([0.025])
                                .with_colors([rerun::Color::from_rgb(255, 0, 0)]),
                        )?;
                        blocked.push(i);
                    }
                    None => {
                        worlds[i].obstacles[last] = previous;
                        uploaded = device.upload(&worlds)?;
                    }
                }
            }
            if !blocked.is_empty() {
                next_push = t + rng.range(2.0, 3.0);
                let problems: Vec<PlanProblem> = blocked
                    .iter()
                    .map(|&i| PlanProblem {
                        world: i as u32,
                        start: now[i].position.clone(),
                        goal: motions[i].goal.clone(),
                        start_motion: Some(StartMotion {
                            velocity: now[i].velocity.clone(),
                            acceleration: now[i].acceleration.clone(),
                        }),
                        hold: vec![],
                    })
                    .collect();
                let options = PlanOptions { time_budget: Some(Duration::from_millis(100)), ..Default::default() };
                let started = Instant::now();
                let replanned = plan(&device, &uploaded, &problems, &options)?;
                let mut done = 0;
                let mut stuck = vec![];
                let mut labels: Vec<(usize, String)> = vec![];
                for (k, &i) in blocked.iter().enumerate() {
                    let old_began = motions[i].began;
                    log_path(
                        &rec,
                        &robot,
                        i,
                        "old_path",
                        motions[i].trajectory.as_ref(),
                        t - old_began,
                        [150, 150, 150],
                    )?;
                    let motion =
                        StartMotion { velocity: now[i].velocity.clone(), acceleration: now[i].acceleration.clone() };
                    match replanned.best(k).map(|path| Trajectory::moving(&robot, path, &motion)) {
                        Some(Ok(trajectory)) => {
                            motions[i].trajectory = Some(trajectory);
                            motions[i].began = t;
                            log_path(&rec, &robot, i, "path", motions[i].trajectory.as_ref(), 0.0, [40, 200, 90])?;
                            labels.push((i, "replanned around it".to_string()));
                            done += 1;
                        }
                        _ => stuck.push(i),
                    }
                }
                // No way around at this speed: a controlled stop on the path it is on
                // (`Trajectory::stop`), then the goal again from rest.
                let mut braking = 0;
                for &i in &stuck {
                    let stop = motions[i].trajectory.as_ref().and_then(|tr| tr.stop(&robot, t - motions[i].began).ok());
                    motions[i].resume = Some(motions[i].goal.clone());
                    match stop {
                        Some(trajectory) => {
                            motions[i].trajectory = Some(trajectory);
                            motions[i].began = t;
                            log_path(&rec, &robot, i, "path", motions[i].trajectory.as_ref(), 0.0, [240, 180, 40])?;
                            labels.push((i, "too fast to get around: stopping on its path".to_string()));
                            braking += 1;
                        }
                        None => {
                            // The path ends sooner than any stop would: it finishes its motion.
                            labels.push((i, "finishing its motion".to_string()));
                        }
                    }
                }
                let took = started.elapsed().as_secs_f64() * 1e3;
                let scenes: Vec<String> = blocked.iter().map(|i| i.to_string()).collect();
                let braked =
                    if stuck.is_empty() { String::new() } else { format!("; {braking} braking to replan from rest") };
                let event = format!(
                    "{t:.2} s: box in the path of robot {}: {done} of {} replanned mid-motion{braked}, in {took:.2} ms",
                    scenes.join(", "),
                    blocked.len()
                );
                println!("{event}");
                // A label over each robot, for two seconds.
                for (i, text) in labels {
                    rec.log(
                        format!("world/scene_{i}/label"),
                        &rerun::Points3D::new([[0.3, 0.0, 1.0]])
                            .with_radii([0.001])
                            .with_labels([format!("{text} in {took:.1} ms")])
                            .with_show_labels(true),
                    )?;
                    label_until[i] = t + 2.0;
                }
                rec.log("events", &rerun::TextLog::new(event))?;
            }
        }

        for (i, until) in label_until.iter_mut().enumerate() {
            if *until > 0.0 && t >= *until {
                *until = 0.0;
                rec.log(format!("world/scene_{i}/label"), &rerun::Points3D::new(Vec::<[f32; 3]>::new()))?;
                rec.log(format!("world/scene_{i}/would_hit"), &rerun::Points3D::new(Vec::<[f32; 3]>::new()))?;
                rec.log(format!("world/scene_{i}/old_path"), &rerun::LineStrips3D::new(Vec::<Vec<[f32; 3]>>::new()))?;
            }
        }
        let robots: Vec<(Vec3, Vec<Pose>)> = (0..count)
            .map(|i| {
                Ok((
                    offsets[i],
                    meshes.link_poses(&robot, &now[i].position, panda_fingers(&robot, &now[i].position, OPEN)?)?,
                ))
            })
            .collect::<Result<_>>()?;
        meshes.log_robots(&rec, &robots)?;
        if args.save.is_none() {
            // Live: keep to the wall clock.
            let due = Duration::from_secs_f32(t + TICK);
            std::thread::sleep(due.saturating_sub(clock.elapsed()));
        }
    }
    Ok(())
}

/// Where robot `i`'s hand is when its motion first collides ahead of time `t` (checked every
/// 10 ms), if it does.
fn collides_ahead(device: &Device, worlds: &Worlds, robot: &Robot, m: &Motion, i: usize, t: f32) -> Option<Vec3> {
    let tr = m.trajectory.as_ref()?;
    let mut state = JointState::new(robot.dof());
    let mut q = vec![];
    let mut s = t - m.began;
    while s < tr.duration() {
        tr.at(s, &mut state);
        q.extend_from_slice(&state.position);
        s += 0.01;
    }
    let items = q.len() / robot.dof();
    let eval = device.evaluate(worlds, &vec![i as u32; items], &q, &CollisionWeights::NONE).expect("a valid batch");
    let hit = (0..items).find(|&k| !eval.collision_free(k))?;
    Some(robot.ee_pose(&q[hit * robot.dof()..][..robot.dof()], 0).position)
}

/// The hand's path along `trajectory` from time `from`, or nothing.
fn log_path(
    rec: &rerun::RecordingStream,
    robot: &Robot,
    i: usize,
    name: &str,
    trajectory: Option<&Trajectory>,
    from: f32,
    color: [u8; 3],
) -> Result<()> {
    let mut points = vec![];
    if let Some(tr) = trajectory {
        let mut state = JointState::new(robot.dof());
        let mut s = from;
        while s <= tr.duration() {
            tr.at(s, &mut state);
            points.push(robot.ee_pose(&state.position, 0).position.to_array());
            s += 1.0 / 30.0;
        }
    }
    let strips = if points.is_empty() { vec![] } else { vec![points] };
    rec.log(
        format!("world/scene_{i}/{name}"),
        &rerun::LineStrips3D::new(strips)
            .with_radii([0.006])
            .with_colors([rerun::Color::from_rgb(color[0], color[1], color[2])]),
    )?;
    Ok(())
}
