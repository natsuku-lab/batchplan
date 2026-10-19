//! The MotionBenchMaker and MπNets Panda problem sets (2,600 problems, as packaged by robometrics).
//! Fetch them first with `scripts/fetch_benchmark.sh`.
//!
//! cargo run --release --example benchmark -- [data_dir=data/robometrics] [--latency N=20] [--device gpu|cpu|all]
//!     [--points N]                 (B-spline control points)
//!     [--optimize S]               (also run trajectory optimization on S seeds per problem)
//!     [--clearance M] [--ramp R]   (clearance paths keep, and how far from their ends it ramps up)
//!     [--passes N] [--min-change D] (shortcut and smoothing passes, smallest smoothing move)
//!     [--rrt-seed N]               (where RRT-Connect's Halton samples start)
//!     [--no-straighten]            (shortcuts without straightening each joint)
//!     [--search-clearance M] [--shortcut-clearance M] (either clearance alone)
//!     [--meshes]                   (also check successful plans against the Panda's meshes)
//!     [--spheres curobo|vamp|fitted] (the Panda's collision spheres: cuRobo's hand-tuned ones, VAMP's,
//!                                    or fitted to the meshes)
//!     [--save paths.json]          (write the planned paths, as `scripts/compare_planners.py` does;
//!                                    with one device, and --latency covering every problem)
//! cargo run --release --example benchmark -- [data_dir] [--spheres S] --check <paths.json> [--check ...]
//!     judges planners' paths (`scripts/compare_planners.py`, `--save`) by one check
//!
//! For every set and device it reports two modes:
//! - **plan**: plan from the start to the set's first IK solution (planning only);
//! - **ik+plan**: solve IK for the goal pose, then plan to the best solution.
//!
//! A problem succeeds when the final `panda_hand` position is within 1 cm of the goal, every joint
//! is within its limits, and an independent CPU check at 4x the planner's validation density finds
//! no collision. Batch-1 latency is measured on the first N problems of each set; everything else
//! runs each set as one batch. With `--meshes`, "meshes ok" counts the successes whose dense samples
//! also clear the obstacles and each other on the Panda's collision meshes (`MeshModel`).

#[path = "common/mod.rs"]
mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use batchplan::*;
use glam::{Quat, Vec3};
use serde::Deserialize;

#[derive(Deserialize)]
struct Problem {
    start: Vec<f32>,
    goal_pose: GoalPose,
    goal_ik: Vec<Vec<f32>>,
    obstacles: Obstacles,
    world_frame: String,
}

#[derive(Deserialize)]
struct GoalPose {
    frame: String,
    position_xyz: [f32; 3],
    quaternion_wxyz: [f32; 4],
}

#[derive(Deserialize)]
struct Obstacles {
    cuboid: Option<BTreeMap<String, Cuboid>>,
    cylinder: Option<BTreeMap<String, Cylinder>>,
}

#[derive(Deserialize)]
struct Cuboid {
    dims: [f32; 3],
    pose: [f32; 7],
}

#[derive(Deserialize)]
struct Cylinder {
    height: f32,
    radius: f32,
    pose: [f32; 7],
}

/// `[x, y, z, qw, qx, qy, qz]`.
fn pose(p: &[f32; 7]) -> (Vec3, Quat) {
    (Vec3::new(p[0], p[1], p[2]), Quat::from_xyzw(p[4], p[5], p[6], p[3]).normalize())
}

impl Problem {
    fn world(&self) -> World {
        let mut obstacles = vec![];
        for c in self.obstacles.cuboid.iter().flat_map(|m| m.values()) {
            let (center, rotation) = pose(&c.pose);
            obstacles.push(Obstacle::Cuboid { center, half_extents: Vec3::from(c.dims) * 0.5, rotation });
        }
        for c in self.obstacles.cylinder.iter().flat_map(|m| m.values()) {
            let (center, rotation) = pose(&c.pose);
            obstacles.push(Obstacle::Cylinder { center, rotation, radius: c.radius, half_height: c.height * 0.5 });
        }
        World { obstacles }
    }

    fn hand_goal(&self) -> Pose {
        let [w, x, y, z] = self.goal_pose.quaternion_wxyz;
        Pose { position: Vec3::from(self.goal_pose.position_xyz), rotation: Quat::from_xyzw(x, y, z, w).normalize() }
    }
}

fn load(path: &PathBuf) -> Result<BTreeMap<String, Vec<Problem>>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {} (run scripts/fetch_benchmark.sh first)", path.display()))?;
    let sets: BTreeMap<String, Vec<Problem>> =
        // The files are large but checksummed by the fetch script, so the parser's size budget is off.
        serde_saphyr::from_str_with_options(&text, serde_saphyr::options! { budget: None })
            .with_context(|| format!("parsing {}", path.display()))?;
    for p in sets.values().flatten() {
        ensure!(p.goal_pose.frame == "panda_hand" && p.world_frame == "panda_link0", "unexpected frames");
    }
    Ok(sets)
}

/// VAMP's Panda spheres (`panda_spherized.urdf` from VAMP 0.6.4, fetched by
/// `scripts/fetch_benchmark.sh`), on this URDF's links. VAMP fixes the fingers 6.5 cm from the hand's
/// axis, where this URDF holds them 4 cm out, so the spheres of every link VAMP fixes are placed where
/// VAMP places them relative to the link's parent.
fn vamp_spheres(robot: &Robot, data: &std::path::Path) -> Result<CollisionModel> {
    let urdf = urdf_rs::read_file(data.join("vamp_panda_spherized.urdf"))?;
    let pose = |o: &urdf_rs::Pose| {
        let [r, p, y] = o.rpy.0.map(|v| v as f32);
        Pose {
            position: Vec3::from(o.xyz.0.map(|v| v as f32)),
            rotation: Quat::from_euler(glam::EulerRot::ZYX, y, p, r),
        }
    };
    let at = |link: &str| robot.link_pose(robot.default_q(), link).with_context(|| format!("no link {link}"));
    let source = "Collision spheres of KavrakiLab/vamp resources/panda/panda_spherized.urdf (Apache-2.0).";
    let mut model = CollisionModel { source: Some(source.into()), ..Default::default() };
    for link in &urdf.links {
        let spheres: Vec<(Vec3, f32)> = link
            .collision
            .iter()
            .filter_map(|c| match c.geometry {
                urdf_rs::Geometry::Sphere { radius } => Some((pose(&c.origin).position, radius as f32)),
                _ => None,
            })
            .collect();
        if spheres.is_empty() {
            continue;
        }
        let mut to_ours = Pose { position: Vec3::ZERO, rotation: Quat::IDENTITY };
        if let Some(j) = urdf.joints.iter().find(|j| j.child.link == link.name)
            && matches!(j.joint_type, urdf_rs::JointType::Fixed)
        {
            to_ours = at(&link.name)?.inverse().mul_pose(at(&j.parent.link)?).mul_pose(pose(&j.origin));
        }
        let placed = spheres.iter().map(|&(c, r)| (to_ours.position + to_ours.rotation * c).extend(r).to_array());
        model.spheres.insert(link.name.clone(), placed.collect());
    }
    Ok(model)
}

/// Samples per spline span the judge checks: 4x the planner's default validation density.
const JUDGE_PER_SPAN: usize = 32;

struct Args {
    data: PathBuf,
    latency: usize,
    device: String,
    plan: PlanOptions,
    meshes: bool,
    spheres: String,
    save: Option<PathBuf>,
    check: Vec<PathBuf>,
}

fn args() -> Result<Args> {
    let mut a = Args {
        data: PathBuf::from("data/robometrics"),
        latency: 20,
        device: "all".into(),
        plan: PlanOptions::default(),
        meshes: false,
        spheres: "curobo".into(),
        save: None,
        check: vec![],
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--latency" => a.latency = it.next().context("--latency N")?.parse()?,
            "--device" => a.device = it.next().context("--device gpu|cpu|all")?,
            "--points" => a.plan.control_points = it.next().context("--points N")?.parse()?,
            "--optimize" => a.plan.seeds = it.next().context("--optimize S")?.parse()?,
            "--clearance" => {
                let m = it.next().context("--clearance M")?.parse()?;
                (a.plan.rrt.clearance, a.plan.shortcut.clearance) = (m, m);
            }
            "--passes" => a.plan.shortcut.passes = it.next().context("--passes N")?.parse()?,
            "--no-straighten" => a.plan.shortcut.straighten = false,
            "--rrt-seed" => a.plan.rrt.rng_seed = it.next().context("--rrt-seed N")?.parse()?,
            "--substeps" => a.plan.validate_substeps = it.next().context("--substeps K")?.parse()?,
            "--search-clearance" => a.plan.rrt.clearance = it.next().context("--search-clearance M")?.parse()?,
            "--shortcut-clearance" => {
                a.plan.shortcut.clearance = it.next().context("--shortcut-clearance M")?.parse()?
            }
            "--min-change" => a.plan.shortcut.min_change = it.next().context("--min-change D")?.parse()?,
            "--ramp" => {
                let r = it.next().context("--ramp R")?.parse()?;
                (a.plan.rrt.clearance_ramp, a.plan.shortcut.clearance_ramp) = (r, r);
            }
            "--meshes" => a.meshes = true,
            "--spheres" => a.spheres = it.next().context("--spheres curobo|vamp|fitted")?,
            "--save" => a.save = Some(PathBuf::from(it.next().context("--save paths.json")?)),
            "--check" => a.check.push(PathBuf::from(it.next().context("--check paths.json")?)),
            s if s.starts_with("--") => bail!("unknown flag {s}"),
            s => a.data = PathBuf::from(s),
        }
    }
    Ok(a)
}

#[derive(Default)]
struct Outcome {
    success: usize,
    /// Successes that also clear everything on the meshes.
    on_meshes: usize,
    seconds: f64,
    path_length: Vec<f32>,
    motion_time: Vec<f32>,
    peak_acceleration: Vec<f32>,
    peak_jerk: Vec<f32>,
}

/// Successful problems under the independent check, plus path metrics of the successful paths.
fn score(
    robot: &Robot,
    cpu: &Device,
    worlds: &Worlds,
    meshes: Option<&MeshModel>,
    problems: &[Problem],
    result: &PlanResult,
) -> Result<Outcome> {
    let n = robot.dof();
    let per_span = JUDGE_PER_SPAN as f32;
    let mut out = Outcome::default();
    for s in result.solved() {
        // Check the timed trajectory a controller would run, sampled denser per span than the
        // planner validates by default.
        let trajectory = Trajectory::new(robot, s.solution, 1.0)?;
        let dense = trajectory.sample(per_span / trajectory.knot_interval)?.positions;
        let items = dense.len() / n;
        let eval = cpu.evaluate(worlds, &vec![s.problem.world; items], &dense, &CollisionWeights::NONE)?;
        let collision_free = (0..items).all(|i| eval.collision_free(i));
        let within_limits = trajectory.check(robot).is_ok();
        let goal = problems[s.problem.world as usize].hand_goal();
        let hand = robot.link_pose(&dense[(items - 1) * n..], "panda_hand").expect("panda_hand");
        let reached = (hand.position - goal.position).length() < 0.01;
        if !(collision_free && within_limits && reached) {
            continue;
        }
        out.success += 1;
        if let Some(meshes) = meshes {
            let clear = meshes.clearance(worlds.as_slice(), &vec![s.problem.world; items], &dense)?;
            out.on_meshes +=
                usize::from(clear.iter().all(|&[world, self_collision]| world > 0.0 && self_collision > 0.0));
        }
        let q = |i: usize| &dense[i * n..(i + 1) * n];
        out.path_length.push(
            (0..items - 1).map(|i| q(i).iter().zip(q(i + 1)).map(|(a, b)| (b - a).powi(2)).sum::<f32>().sqrt()).sum(),
        );
        let dt = 0.01;
        let traj = trajectory.sample(1.0 / dt)?;
        out.motion_time.push(traj.duration);
        let v = &traj.velocities;
        let acc: Vec<f32> = (0..v.len().saturating_sub(n)).map(|i| (v[i + n] - v[i]) / dt).collect();
        let jerk: Vec<f32> = (0..acc.len().saturating_sub(n)).map(|i| (acc[i + n] - acc[i]) / dt).collect();
        out.peak_acceleration.push(acc.iter().fold(0.0, |m: f32, a| m.max(a.abs())));
        out.peak_jerk.push(jerk.iter().fold(0.0, |m: f32, a| m.max(a.abs())));
    }
    Ok(out)
}

fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mut v = v.to_vec();
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// NaN for an empty set (no latency runs).
fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() { f64::NAN } else { v[((v.len() - 1) as f64 * p).round() as usize] }
}

fn ik_problems(problems: &[Problem]) -> Vec<IkProblem> {
    problems
        .iter()
        .enumerate()
        .map(|(i, p)| IkProblem { world: i as u32, targets: vec![Some(p.hand_goal())], seed: None })
        .collect()
}

/// IK for every goal, then a plan from each start to its best IK solution.
fn ik_and_plan(device: &Device, worlds: &Worlds, problems: &[Problem], o: &PlanOptions) -> Result<PlanResult> {
    let ik = solve_ik(device, worlds, &ik_problems(problems), &IkOptions::default())?;
    let plans: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem {
            world: s.problem.world,
            start: problems[s.problem.world as usize].start.clone(),
            goal: s.solution.to_vec(),
            start_motion: None,
            hold: vec![],
        })
        .collect();
    Ok(plan(device, worlds, &plans, o)?)
}

fn main() -> Result<()> {
    let args = args()?;
    // Goals are poses of the hand frame.
    let mut options = RobotOptions { ee_links: vec!["panda_hand".into()], ..common::panda_options() };
    let urdf = common::asset("franka/franka_panda.urdf");
    match args.spheres.as_str() {
        "curobo" => {}
        "vamp" => {
            options.collision_model = Some(vamp_spheres(&Robot::load(&urdf, &options)?, &args.data)?);
            let srdf = args.data.join("vamp_panda.srdf");
            options.srdf = Some(std::fs::read_to_string(&srdf).with_context(|| format!("reading {}", srdf.display()))?);
        }
        "fitted" => options.collision_model = None,
        s => bail!("unknown spheres {s}"),
    }
    let robot = Robot::load(&urdf, &options)?;
    let cpu = Device::cpu(&robot)?;
    if !args.check.is_empty() {
        let mut sets = vec![];
        for f in ["mb_set.yaml", "mpinets_set.yaml"] {
            sets.extend(load(&args.data.join(f))?);
        }
        return check(&args.check, &robot, &cpu, &MeshModel::new(&robot)?, &sets);
    }
    let meshes = args.meshes.then(|| MeshModel::new(&robot)).transpose()?;
    let mut devices = vec![];
    if args.device != "cpu" {
        devices.push(Device::gpu(&robot)?);
    }
    if args.device != "gpu" {
        devices.push(Device::cpu(&robot)?);
    }
    ensure!(args.save.is_none() || devices.len() == 1, "--save needs --device cpu or gpu");
    let mut saved = serde_json::Map::new();
    let files = ["mb_set.yaml", "mpinets_set.yaml"];
    let mut sets = vec![];
    for f in files {
        sets.extend(load(&args.data.join(f))?);
    }
    println!(
        "{} sets, {} problems; batch-1 latency on the first {} problems of each set\n",
        sets.len(),
        sets.iter().map(|(_, p)| p.len()).sum::<usize>(),
        args.latency
    );

    for device in &devices {
        println!("{}", device.name());
        println!(
            "{:<28} {:>5} {:>10} {:>11} {:>11} {:>9} {:>29} {:>7} {:>7} {:>8} {:>9}",
            "set",
            "n",
            "free ends",
            "plan ok",
            "ik+plan ok",
            "prob/s",
            "ik+plan batch-1 ms mean/75/98",
            "length",
            "time s",
            "acc",
            "jerk"
        );
        let mut total_meshes = 0;
        // Planning only, start to the set's first IK goal: what other planners are compared with.
        let (mut plan_latency, mut plan_seconds, mut plan_meshes, mut plan_lengths) = (vec![], 0.0, 0, vec![]);
        let (mut total, mut total_free, mut total_plan, mut total_full, mut total_time) = (0, 0, 0, 0, 0.0);
        let mut all_latency = vec![];
        solve_ik(
            device,
            &device.upload(&[World::default()])?,
            &[IkProblem { world: 0, targets: vec![Some(robot.ee_pose(robot.default_q(), 0))], seed: None }],
            &IkOptions::default(),
        )?;
        for (name, problems) in &sets {
            // Timings include uploading the worlds.
            let scene: Vec<World> = problems.iter().map(Problem::world).collect();
            let on_cpu = cpu.upload(&scene)?;
            let to_ik: Vec<PlanProblem> = problems
                .iter()
                .enumerate()
                .map(|(i, p)| PlanProblem {
                    world: i as u32,
                    start: p.start.clone(),
                    goal: p.goal_ik[0].clone(),
                    start_motion: None,
                    hold: vec![],
                })
                .collect();
            let t = Instant::now();
            let planned = plan(device, &device.upload(&scene)?, &to_ik, &args.plan)?;
            let plan_only = Outcome {
                seconds: t.elapsed().as_secs_f64(),
                ..score(&robot, &cpu, &on_cpu, meshes.as_ref(), problems, &planned)?
            };

            let t = Instant::now();
            let full_result = ik_and_plan(device, &device.upload(&scene)?, problems, &args.plan)?;
            let full = Outcome {
                seconds: t.elapsed().as_secs_f64(),
                ..score(&robot, &cpu, &on_cpu, meshes.as_ref(), problems, &full_result)?
            };

            // Problems whose start and given IK goal are collision-free under this robot model.
            let ends: Vec<f32> = problems.iter().flat_map(|p| p.start.iter().chain(&p.goal_ik[0]).copied()).collect();
            let end_world: Vec<u32> = (0..problems.len() as u32).flat_map(|i| [i, i]).collect();
            let eval = cpu.evaluate(&on_cpu, &end_world, &ends, &CollisionWeights::NONE)?;
            let free_ends =
                (0..problems.len()).filter(|&i| eval.collision_free(2 * i) && eval.collision_free(2 * i + 1)).count();
            total_free += free_ends;

            let mut latency = vec![];
            let timed_alone = plan_latency.len();
            for i in 0..args.latency.min(problems.len()) {
                let t = Instant::now();
                ik_and_plan(device, &device.upload(&scene[i..i + 1])?, &problems[i..i + 1], &args.plan)?;
                latency.push(t.elapsed().as_secs_f64() * 1e3);
                let one = [PlanProblem { world: 0, ..to_ik[i].clone() }];
                let t = Instant::now();
                plan(device, &device.upload(&scene[i..i + 1])?, &one, &args.plan)?;
                plan_latency.push(t.elapsed().as_secs_f64() * 1e3);
            }
            if args.save.is_some() {
                ensure!(
                    args.latency >= problems.len(),
                    "--save times every problem alone: pass --latency {}",
                    problems.len()
                );
                let attempts = (0..problems.len())
                    .map(|i| {
                        let path = planned.best(i).map(|solution| dense_path(&robot, solution)).transpose()?;
                        let seconds = plan_latency[timed_alone + i] * 1e-3;
                        Ok(serde_json::json!({ "solved": path.is_some(), "seconds": seconds, "path": path }))
                    })
                    .collect::<Result<Vec<_>>>()?;
                saved.insert(
                    name.clone(),
                    serde_json::json!({ "wall_seconds": plan_only.seconds, "problems": attempts }),
                );
            }
            plan_seconds += plan_only.seconds;
            plan_meshes += plan_only.on_meshes;
            plan_lengths.extend(plan_only.path_length.iter().copied());
            all_latency.extend(&latency);
            let mean = latency.iter().sum::<f64>() / latency.len() as f64;
            let n = problems.len();
            println!(
                "{:<28} {:>5} {:>9.1}% {:>10.1}% {:>10.1}% {:>9.1} {:>13.0} {:>7.0} {:>7.0} {:>7.2} {:>7.2} {:>8.1} {:>9.0}",
                name,
                n,
                100.0 * free_ends as f64 / n as f64,
                100.0 * plan_only.success as f64 / n as f64,
                100.0 * full.success as f64 / n as f64,
                n as f64 / full.seconds,
                mean,
                percentile(&mut latency, 0.75),
                percentile(&mut latency, 0.98),
                median(&full.path_length),
                median(&full.motion_time),
                median(&full.peak_acceleration),
                median(&full.peak_jerk),
            );
            total += n;
            total_plan += plan_only.success;
            total_full += full.success;
            total_time += full.seconds;
            if meshes.is_some() {
                println!(
                    "{:<28} meshes ok {:>5.1}% ({} of {} successes)",
                    "",
                    100.0 * full.on_meshes as f64 / n as f64,
                    full.on_meshes,
                    full.success
                );
                total_meshes += full.on_meshes;
            }
        }
        let mean = all_latency.iter().sum::<f64>() / all_latency.len() as f64;
        println!(
            "{:<28} {:>5} {:>9.1}% {:>10.1}% {:>10.1}% {:>9.1} {:>13.0} {:>7.0} {:>7.0}\n",
            "all",
            total,
            100.0 * total_free as f64 / total as f64,
            100.0 * total_plan as f64 / total as f64,
            100.0 * total_full as f64 / total as f64,
            total as f64 / total_time,
            mean,
            percentile(&mut all_latency, 0.75),
            percentile(&mut all_latency, 0.98),
        );
        let on_meshes = if meshes.is_some() {
            format!(", {:.1}% clear on meshes", 100.0 * plan_meshes as f64 / total as f64)
        } else {
            String::new()
        };
        println!(
            "plan only: {:.1}% solved{on_meshes}, {:.1} problems/s batched, batch-1 {:.1} / {:.1} / {:.1} ms (median / p75 / p98), median length {:.2} rad\n",
            100.0 * total_plan as f64 / total as f64,
            total as f64 / plan_seconds,
            percentile(&mut plan_latency, 0.5),
            percentile(&mut plan_latency, 0.75),
            percentile(&mut plan_latency, 0.98),
            median(&plan_lengths),
        );
        if meshes.is_some() {
            println!("{:<28} meshes ok {:>5.1}%\n", "all", 100.0 * total_meshes as f64 / total as f64);
        }
    }
    if let Some(path) = &args.save {
        let machine = devices[0].name();
        let planners = serde_json::json!({ "batchplan": saved });
        std::fs::write(path, serde_json::json!({ "machine": machine, "planners": planners }).to_string())?;
    }
    Ok(())
}

/// A planned path as waypoints: its timed trajectory sampled as densely as `score` checks it.
fn dense_path(robot: &Robot, solution: &[f32]) -> Result<Vec<Vec<f32>>> {
    let trajectory = Trajectory::new(robot, solution, 1.0)?;
    let dense = trajectory.sample(JUDGE_PER_SPAN as f32 / trajectory.knot_interval)?.positions;
    Ok(dense.chunks(robot.dof()).map(<[f32]>::to_vec).collect())
}

#[derive(Deserialize)]
struct Comparison {
    machine: String,
    planners: BTreeMap<String, BTreeMap<String, SetAttempts>>,
}

#[derive(Deserialize)]
struct SetAttempts {
    wall_seconds: f64,
    problems: Vec<Attempt>,
}

#[derive(Deserialize)]
struct Attempt {
    solved: bool,
    seconds: f64,
    path: Option<Vec<Vec<f32>>>,
}

/// One path as the judge found it.
struct Verdict {
    length: f32,
    /// It solves its problem on the loaded spheres.
    on_spheres: bool,
    /// It solves its problem clear of the meshes.
    on_meshes: bool,
}

/// Judges planners' paths (`scripts/compare_planners.py` or `--save`, start to each set's first IK
/// goal) by one check: the hand within 1 cm of the goal, the joints within their limits, and no
/// collision along the path, sampled at most 0.005 rad apart, on the loaded spheres and on the Panda's
/// meshes. Path lengths are compared again over the problems every planner solves on the spheres, and
/// over those every planner solves clear of the meshes.
fn check(
    paths: &[PathBuf],
    robot: &Robot,
    cpu: &Device,
    meshes: &MeshModel,
    sets: &[(String, Vec<Problem>)],
) -> Result<()> {
    let mut planners = BTreeMap::new();
    for path in paths {
        let comparison: Comparison = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        println!("paths from {} ({})", path.display(), comparison.machine);
        for (planner, by_set) in comparison.planners {
            // A planner already loaded (batchplan on another device, say) is named after its file.
            let name = match path.file_stem() {
                Some(stem) if planners.contains_key(&planner) => format!("{planner} ({})", stem.to_string_lossy()),
                _ => planner,
            };
            ensure!(!planners.contains_key(&name), "{name} appears twice");
            planners.insert(name, by_set);
        }
    }
    println!(
        "{:<10} {:>5} {:>9} {:>11} {:>10} {:>14} {:>11} {:>8}",
        "planner", "n", "claimed", "spheres ok", "meshes ok", "median ms", "problems/s", "length"
    );
    let n = robot.dof();
    // Per planner and problem, how its path was judged.
    let mut judged = BTreeMap::new();
    for (planner, by_set) in &planners {
        let (mut total, mut claimed, mut on_spheres, mut on_meshes, mut wall) = (0, 0, 0, 0, 0.0);
        let mut per_set = vec![];
        let (mut seconds, mut lengths, mut verdicts) = (vec![], vec![], vec![]);
        for (name, problems) in sets {
            let attempts = by_set.get(name).with_context(|| format!("{planner} has no results for {name}"))?;
            ensure!(attempts.problems.len() == problems.len(), "{planner} {name}: {} results", attempts.problems.len());
            let scene: Vec<World> = problems.iter().map(Problem::world).collect();
            let worlds = cpu.upload(&scene)?;
            wall += attempts.wall_seconds;
            total += problems.len();
            let before = (claimed, on_spheres, on_meshes);
            for (i, (problem, attempt)) in problems.iter().zip(&attempts.problems).enumerate() {
                seconds.push(attempt.seconds * 1e3);
                let Some(waypoints) = attempt.path.as_ref().filter(|_| attempt.solved) else {
                    verdicts.push(None);
                    continue;
                };
                claimed += 1;
                let mut dense: Vec<f32> = waypoints[0].clone();
                for pair in waypoints.windows(2) {
                    let step = pair[0].iter().zip(&pair[1]).map(|(a, b)| (b - a).abs()).fold(0.0, f32::max);
                    let k = (step / 0.005).ceil().max(1.0) as usize;
                    for s in 1..=k {
                        let t = s as f32 / k as f32;
                        dense.extend(pair[0].iter().zip(&pair[1]).map(|(a, b)| a + (b - a) * t));
                    }
                }
                let items = dense.len() / n;
                let goal = problem.hand_goal();
                let hand = robot.link_pose(&dense[(items - 1) * n..], "panda_hand").expect("panda_hand");
                let reached = (hand.position - goal.position).length() < 0.01;
                let within = dense.chunks(n).all(|q| {
                    q.iter().enumerate().all(|(j, &v)| v >= robot.lower()[j] - 1e-4 && v <= robot.upper()[j] + 1e-4)
                });
                let eval = cpu.evaluate(&worlds, &vec![i as u32; items], &dense, &CollisionWeights::NONE)?;
                let solved = reached && within && (0..items).all(|k| eval.collision_free(k));
                on_spheres += usize::from(solved);
                let length: f32 = waypoints
                    .windows(2)
                    .map(|w| w[0].iter().zip(&w[1]).map(|(a, b)| (b - a).powi(2)).sum::<f32>().sqrt())
                    .sum();
                lengths.push(length);
                let clear = meshes.clearance(&scene, &vec![i as u32; items], &dense)?;
                let clear = reached && within && clear.iter().all(|&[w, s]| w > 0.0 && s > 0.0);
                on_meshes += usize::from(clear);
                verdicts.push(Some(Verdict { length, on_spheres: solved, on_meshes: clear }));
            }
            per_set.push((name, problems.len(), claimed - before.0, on_spheres - before.1, on_meshes - before.2));
        }
        for (name, n, c, s, m) in per_set {
            println!("  {name:<28} {n:>5} {c:>9} {s:>11} {m:>10}");
        }
        let pct = |k: usize| format!("{:.1}%", 100.0 * k as f64 / total as f64);
        println!(
            "{:<10} {:>5} {:>9} {:>11} {:>10} {:>14.2} {:>11.1} {:>8.2}",
            planner,
            total,
            pct(claimed),
            pct(on_spheres),
            pct(on_meshes),
            percentile(&mut seconds, 0.5),
            total as f64 / wall,
            median(&lengths)
        );
        judged.insert(planner, verdicts);
    }
    let problems = sets.iter().map(|(_, p)| p.len()).sum::<usize>();
    for meshes in [false, true] {
        let passes = |v: &Verdict| if meshes { v.on_meshes } else { v.on_spheres };
        let common: Vec<usize> =
            (0..problems).filter(|&i| judged.values().all(|v| v[i].as_ref().is_some_and(passes))).collect();
        let judge = if meshes { "clear of the meshes" } else { "on the spheres" };
        println!("median length over the {} problems every planner solves {judge}:", common.len());
        for (planner, verdicts) in &judged {
            let lengths: Vec<f32> = common.iter().map(|&i| verdicts[i].as_ref().expect("common").length).collect();
            println!("  {planner:<10} {:.2}", median(&lengths));
        }
    }
    Ok(())
}
