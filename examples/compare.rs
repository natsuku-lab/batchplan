//! batchplan in the planner comparison (`scripts/compare.py`): the canonical spherized Panda and
//! the problems' scenes exported by `scripts/bench_model.py export`, read as every other planner
//! reads them.
//!
//! cargo run --release --example compare -- parity <inputs_dir> <fixtures.json> <out.json>
//!     forward kinematics of every link and clearances, for the parity check
//!     (`scripts/parity.py`)
//! cargo run --release --example compare -- run <job.json> <out_dir>
//!     a run of the comparison, as `scripts/compare.py run --planner batchplan` writes its job:
//!     for every repetition and seed, each problem planned alone ("single": latency), then every
//!     problem in one `plan` call, each in its own world ("batch": throughput)
//! cargo run --release --example compare -- mesh <inputs_dir> <run_dir>
//!     any planner's run, checked on the Panda's collision meshes in the exact scenes, apart from
//!     the judge's check on the canonical spheres: `mesh_verdicts.jsonl`

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use batchplan::{
    Clearance, CollisionModel, CollisionWeights, Device, JointLimits, MeshModel, Obstacle,
    PlanOptions, PlanProblem, Pose, Robot, RobotOptions, Trajectory, World,
};
use glam::{EulerRot, Quat, Vec3};
use serde::Deserialize;
use serde_json::json;

/// A shape of a problem's scene, in the robot's base frame.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Shape {
    Box {
        centre: [f32; 3],
        wxyz: [f32; 4],
        half: [f32; 3],
    },
    Cylinder {
        centre: [f32; 3],
        wxyz: [f32; 4],
        radius: f32,
        half_height: f32,
    },
    Capsule {
        a: [f32; 3],
        b: [f32; 3],
        radius: f32,
    },
}

impl Shape {
    fn obstacle(&self) -> Obstacle {
        let quat = |[w, x, y, z]: [f32; 4]| Quat::from_xyzw(x, y, z, w).normalize();
        match *self {
            Shape::Box { centre, wxyz, half } => Obstacle::Cuboid {
                center: centre.into(),
                half_extents: half.into(),
                rotation: quat(wxyz),
            },
            Shape::Cylinder {
                centre,
                wxyz,
                radius,
                half_height,
            } => Obstacle::Cylinder {
                center: centre.into(),
                rotation: quat(wxyz),
                radius,
                half_height,
            },
            Shape::Capsule { a, b, radius } => {
                let (a, b) = (Vec3::from(a), Vec3::from(b));
                Obstacle::Capsule {
                    center: (a + b) / 2.0,
                    rotation: Quat::from_rotation_arc(Vec3::Z, (b - a).normalize()),
                    radius,
                    half_length: (b - a).length() / 2.0,
                }
            }
        }
    }
}

fn world(shapes: &[Shape]) -> World {
    World::new(shapes.iter().map(Shape::obstacle).collect())
}

/// The canonical Panda: the spherized URDF's kinematics, its spheres exactly as written (not
/// fitted), and the SRDF's disabled pairs. Also its links' names.
fn panda(inputs: &Path, limits: Option<&Limits>) -> Result<(Robot, Vec<String>)> {
    let urdf_path = inputs.join("panda.urdf");
    let urdf = urdf_rs::read_file(&urdf_path)?;
    let pose = |o: &urdf_rs::Pose| {
        let [r, p, y] = o.rpy.0.map(|v| v as f32);
        Pose::new(
            Vec3::from(o.xyz.0.map(|v| v as f32)),
            Quat::from_euler(EulerRot::ZYX, y, p, r),
        )
    };
    let mut model = CollisionModel {
        source: Some("Collision spheres of KavrakiLab/vamp resources/panda/panda_spherized.urdf (Apache-2.0).".into()),
        ..Default::default()
    };
    for link in &urdf.links {
        let mut spheres = vec![];
        for c in &link.collision {
            let urdf_rs::Geometry::Sphere { radius } = c.geometry else {
                bail!("{}: the canonical model has only spheres", link.name);
            };
            spheres.push(pose(&c.origin).position.extend(radius as f32).to_array());
        }
        if !spheres.is_empty() {
            model.spheres.insert(link.name.clone(), spheres);
        }
    }
    let srdf = inputs.join("panda.srdf");
    let joint_limits = limits.map_or_else(HashMap::new, |l| {
        (0..7)
            .map(|j| {
                let limits = JointLimits {
                    velocity: Some(l.velocity[j]),
                    acceleration: Some(l.acceleration[j]),
                    jerk: Some(l.jerk[j]),
                    ..Default::default()
                };
                (format!("panda_joint{}", j + 1), limits)
            })
            .collect()
    });
    let options = RobotOptions {
        ee_links: vec!["panda_hand".into()],
        joint_limits,
        default_q: Some(vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
        collision_model: Some(model),
        srdf: Some(
            std::fs::read_to_string(&srdf)
                .with_context(|| format!("reading {}", srdf.display()))?,
        ),
        ..Default::default()
    };
    let links = urdf.links.iter().map(|l| l.name.clone()).collect();
    Ok((Robot::load(&urdf_path, &options)?, links))
}

#[derive(Deserialize)]
struct Problem {
    id: String,
    start: Vec<f32>,
    goal: Vec<f32>,
    scene: Vec<Shape>,
    scene_capsules: Vec<Shape>,
}

#[derive(Deserialize)]
struct Fixtures {
    /// Problem ids whose scenes the configurations are checked in.
    scenes: Vec<String>,
    /// Joint positions and an index into `scenes`.
    configurations: Vec<(Vec<f32>, usize)>,
}

/// Every link's pose and the world and self clearances of each fixture configuration. The margins
/// make clearances exact up to 10 cm (beyond that they are lower bounds).
fn parity(inputs: &Path, fixtures: &Path, out: &Path) -> Result<()> {
    let (robot, links) = panda(inputs, None)?;
    let problems: Vec<Problem> =
        serde_json::from_str(&std::fs::read_to_string(inputs.join("problems.json"))?)?;
    let fixtures: Fixtures = serde_json::from_str(&std::fs::read_to_string(fixtures)?)?;
    let worlds: Vec<World> = fixtures
        .scenes
        .iter()
        .map(|id| {
            problems
                .iter()
                .find(|p| &p.id == id)
                .map(|p| world(&p.scene))
                .with_context(|| format!("no problem {id}"))
        })
        .collect::<Result<_>>()?;
    let cpu = Device::cpu(&robot)?;
    let uploaded = cpu.upload(&worlds)?;
    let q: Vec<f32> = fixtures
        .configurations
        .iter()
        .flat_map(|(q, _)| q.clone())
        .collect();
    let item_world: Vec<u32> = fixtures
        .configurations
        .iter()
        .map(|&(_, w)| w as u32)
        .collect();
    let weights = CollisionWeights {
        margin: 0.1,
        self_margin: 0.1,
        ..CollisionWeights::NONE
    };
    let eval = cpu.evaluate(&uploaded, &item_world, &q, &weights)?;
    let rows: Vec<_> = fixtures
        .configurations
        .iter()
        .enumerate()
        .map(|(i, (q, _))| {
            let poses: serde_json::Map<_, _> = links
                .iter()
                .filter_map(|l| {
                    let p = robot.link_pose(q, l)?;
                    Some((
                        l.clone(),
                        json!([
                            p.position.x,
                            p.position.y,
                            p.position.z,
                            p.rotation.w,
                            p.rotation.x,
                            p.rotation.y,
                            p.rotation.z
                        ]),
                    ))
                })
                .collect();
            json!({
                "links": poses,
                "world": eval.world_clearance[i],
                "self": eval.self_clearance[i],
            })
        })
        .collect();
    let report = json!({
        "joint_names": robot.joint_names(),
        "lower": robot.lower(),
        "upper": robot.upper(),
        "rows": rows,
    });
    std::fs::write(out, report.to_string())?;
    Ok(())
}

/// A query of the one-scene set: its scene is `scene_ref`'s.
#[derive(Deserialize)]
struct Query {
    id: String,
    start: Vec<f32>,
    goal: Vec<f32>,
    scene_ref: String,
}

/// A query to plan: a problem, or a one-scene query, with the problem whose scene it is in.
struct Item<'a> {
    id: &'a str,
    start: &'a [f32],
    goal: &'a [f32],
    scene: &'a Problem,
}

/// The trajectory limits of workload B, every planner's.
#[derive(Deserialize)]
struct Limits {
    velocity: [f32; 7],
    acceleration: [f32; 7],
    jerk: [f32; 7],
}

/// What `scripts/compare.py` asks of a run.
#[derive(Deserialize)]
struct Job {
    inputs: String,
    workload: String,
    family: String,
    /// "exact" or "capsules".
    scene: String,
    /// Problem ids, with their positions in `problems.json`.
    problems: Vec<(String, usize)>,
    warmup: String,
    seeds: Vec<u64>,
    reps: usize,
    threads: usize,
    limits: Limits,
    /// Overrides of `PlanOptions` (search and shortcut clearances).
    #[serde(default)]
    settings: HashMap<String, f32>,
}

/// Arrays of the first repetition's outputs, written as `scripts/compare.py` writes them: one
/// float32 file and an index of key: [offset, rows, cols].
#[derive(Default)]
struct Outputs {
    data: Vec<f32>,
    index: serde_json::Map<String, serde_json::Value>,
}

impl Outputs {
    fn add(&mut self, key: String, values: &[f32], cols: usize) {
        self.index
            .insert(key, json!([self.data.len(), values.len() / cols, cols]));
        self.data.extend_from_slice(values);
    }

    fn write(&self, stem: &Path) -> Result<()> {
        let bytes: Vec<u8> = self.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(stem.with_extension("bin"), bytes)?;
        let index = serde_json::Value::Object(self.index.clone());
        std::fs::write(stem.with_extension("json"), index.to_string())?;
        Ok(())
    }
}

/// Seed `seed`'s options: RRT-Connect's Halton samples start at a point of their own.
fn options(job: &Job, seed: u64) -> PlanOptions {
    let mut o = PlanOptions::default();
    o.rrt.rng_seed = seed * 1_000_003;
    if let Some(&c) = job.settings.get("search_clearance") {
        o.rrt.clearance = c;
    }
    if let Some(&c) = job.settings.get("shortcut_clearance") {
        o.shortcut.clearance = c;
    }
    o
}

/// A solved problem's timed trajectory, sampled at `hz`.
fn sampled(robot: &Robot, solution: &[f32], hz: f32) -> Result<batchplan::JointTrajectory> {
    Ok(Trajectory::try_new(robot, solution, 1.0)?.sample(hz)?)
}

fn record(out: &mut impl Write, job: &Job, mut value: serde_json::Value) -> Result<()> {
    let o = value.as_object_mut().expect("an object");
    o.insert("planner".into(), json!("batchplan"));
    o.insert("workload".into(), json!(job.workload));
    o.insert("family".into(), json!(job.family));
    writeln!(out, "{value}")?;
    Ok(())
}

/// Each sample's time: `dt` apart, the last at the trajectory's end.
fn sample_times(t: &batchplan::JointTrajectory) -> Vec<f32> {
    (0..t.len())
        .map(|i| (i as f32 * t.dt).min(t.duration))
        .collect()
}

fn run(job_path: &Path, out_dir: &Path) -> Result<()> {
    use rayon::prelude::*;

    let job: Job = serde_json::from_str(&std::fs::read_to_string(job_path)?)?;
    let inputs = Path::new(&job.inputs);
    let (robot, _) = panda(inputs, Some(&job.limits))?;
    let problems: Vec<Problem> =
        serde_json::from_str(&std::fs::read_to_string(inputs.join("problems.json"))?)?;
    let queries: Vec<Query> = serde_json::from_str(&std::fs::read_to_string(
        inputs.join("queries_onescene.json"),
    )?)?;
    let by_id: HashMap<&str, &Problem> = problems.iter().map(|p| (p.id.as_str(), p)).collect();
    let query_by_id: HashMap<&str, &Query> = queries.iter().map(|q| (q.id.as_str(), q)).collect();
    let scene = |p: &Problem| {
        world(if job.scene == "capsules" {
            &p.scene_capsules
        } else {
            &p.scene
        })
    };
    let chosen: Vec<Item> = job
        .problems
        .iter()
        .map(|(id, _)| {
            if let Some(p) = by_id.get(id.as_str()) {
                return Ok(Item {
                    id,
                    start: &p.start,
                    goal: &p.goal,
                    scene: p,
                });
            }
            let q = query_by_id
                .get(id.as_str())
                .with_context(|| format!("no problem or query {id}"))?;
            let p = by_id
                .get(q.scene_ref.as_str())
                .with_context(|| format!("no scene {}", q.scene_ref))?;
            Ok(Item {
                id,
                start: &q.start,
                goal: &q.goal,
                scene: p,
            })
        })
        .collect::<Result<_>>()?;
    // Each distinct scene once; `world_of[k]` is item k's.
    let mut scene_index: HashMap<&str, u32> = HashMap::new();
    let mut worlds: Vec<World> = vec![];
    let world_of: Vec<u32> = chosen
        .iter()
        .map(|item| {
            *scene_index
                .entry(item.scene.id.as_str())
                .or_insert_with(|| {
                    worlds.push(scene(item.scene));
                    worlds.len() as u32 - 1
                })
        })
        .collect();
    let device = Device::cpu_threads(&robot, job.threads)?;
    let timing_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(job.threads)
        .build()?;
    std::fs::create_dir_all(out_dir)?;
    let mut records =
        std::io::BufWriter::new(std::fs::File::create(out_dir.join("records.jsonl"))?);
    let mut passes = std::io::BufWriter::new(std::fs::File::create(out_dir.join("passes.jsonl"))?);
    // Warm up, untimed: one problem planned and timed.
    let warm = by_id
        .get(job.warmup.as_str())
        .context("no warmup problem")?;
    let w = device.upload(&[scene(warm)])?;
    let one = [PlanProblem::new(0, warm.start.clone(), warm.goal.clone())];
    if let Some(s) = batchplan::plan(&device, &w, &one, &options(&job, 0))?.best(0) {
        sampled(&robot, s, 100.0)?;
    }
    let timed = job.workload == "B";
    for rep in 1..=job.reps {
        for &seed in &job.seeds {
            let o = options(&job, seed);
            let mut outputs = Outputs::default();
            // Singles: each query alone, its scene uploaded unless the last query's was the same.
            let mut single_solutions = vec![];
            let started = Instant::now();
            let mut loaded: Option<(u32, batchplan::Worlds)> = None;
            for (k, p) in chosen.iter().enumerate() {
                let t0 = Instant::now();
                if loaded.as_ref().is_none_or(|(w, _)| *w != world_of[k]) {
                    let w = &worlds[world_of[k] as usize];
                    loaded = Some((world_of[k], device.upload(std::slice::from_ref(w))?));
                }
                let w = &loaded.as_ref().expect("a scene is loaded").1;
                let t1 = Instant::now();
                let one = [PlanProblem::new(0, p.start.to_vec(), p.goal.to_vec())];
                let result = batchplan::plan(&device, w, &one, &o)?;
                let t2 = Instant::now();
                let solution = result.best(0).map(<[f32]>::to_vec);
                let mut phases =
                    json!({ "setup": (t1 - t0).as_secs_f64(), "plan": (t2 - t1).as_secs_f64() });
                let mut trajectory = None;
                if timed && let Some(s) = &solution {
                    let t = sampled(&robot, s, 100.0)?;
                    let t3 = Instant::now();
                    phases["timing"] = json!((t3 - t2).as_secs_f64());
                    phases["total"] = json!((t3 - t0).as_secs_f64());
                    trajectory = Some(t);
                } else {
                    phases["total"] = json!((t2 - t0).as_secs_f64());
                }
                let status = if solution.is_some() {
                    "solved"
                } else {
                    "failed"
                };
                let mut r = json!({
                    "problem": p.id, "seed": seed, "rep": rep, "api": "single",
                    "status": status, "phases": phases,
                });
                if let Some(t) = &trajectory {
                    r["duration"] = json!(t.duration);
                }
                record(&mut records, &job, r)?;
                if rep == 1
                    && let Some(s) = &solution
                {
                    // Outside the timed phases for workload A: the path sampled along its timing.
                    let t = match trajectory {
                        Some(t) => t,
                        None => sampled(&robot, s, 250.0)?,
                    };
                    outputs.add(format!("{}|path", p.id), &t.positions, t.dof);
                    if timed {
                        outputs.add(format!("{}|t", p.id), &sample_times(&t), 1);
                        outputs.add(format!("{}|q", p.id), &t.positions, t.dof);
                        outputs.add(format!("{}|v", p.id), &t.velocities, t.dof);
                        outputs.add(format!("{}|a", p.id), &t.accelerations, t.dof);
                    }
                }
                single_solutions.push(solution);
            }
            let single_wall = started.elapsed().as_secs_f64();
            // Batch: every problem in one call, each in its own world, then each one's timing on
            // as many threads.
            let t0 = Instant::now();
            let uploaded = device.upload(&worlds)?;
            let t1 = Instant::now();
            let problems: Vec<PlanProblem> = chosen
                .iter()
                .enumerate()
                .map(|(k, p)| PlanProblem::new(world_of[k], p.start.to_vec(), p.goal.to_vec()))
                .collect();
            let result = batchplan::plan(&device, &uploaded, &problems, &o)?;
            let t2 = Instant::now();
            let solved: Vec<Option<&[f32]>> = (0..chosen.len()).map(|k| result.best(k)).collect();
            if timed {
                timing_pool.install(|| {
                    solved
                        .par_iter()
                        .flatten()
                        .map(|s| sampled(&robot, s, 100.0).map(|_| ()))
                        .collect::<Result<Vec<()>>>()
                })?;
            }
            let t3 = Instant::now();
            let mut matches = 0;
            for (k, p) in chosen.iter().enumerate() {
                let same = solved[k].map(<[f32]>::to_vec) == single_solutions[k];
                matches += usize::from(same);
                let status = if solved[k].is_some() {
                    "solved"
                } else {
                    "failed"
                };
                let r = json!({
                    "problem": p.id, "seed": seed, "rep": rep, "api": "batch",
                    "status": status, "matches_single": same,
                });
                record(&mut records, &job, r)?;
            }
            let pass = json!({
                "rep": rep, "seed": seed, "problems": chosen.len(), "single_wall": single_wall,
                "wall": (t3 - t0).as_secs_f64(), "setup": (t1 - t0).as_secs_f64(),
                "plan": (t2 - t1).as_secs_f64(), "timing": (t3 - t2).as_secs_f64(),
                "batch_matches_single": matches,
            });
            writeln!(passes, "{pass}")?;
            records.flush()?;
            passes.flush()?;
            if rep == 1 {
                outputs.write(&out_dir.join(format!("outputs_seed{seed}")))?;
            }
            println!(
                "batchplan {} {} rep {rep} seed {seed}: singles {single_wall:.2} s, batch {:.3} s, \
                 {matches}/{} identical",
                job.workload,
                job.family,
                (t3 - t0).as_secs_f64(),
                chosen.len()
            );
        }
    }
    Ok(())
}

/// The Panda's collision meshes: batchplan's Franka description, posed as the canonical model is
/// (its fingers 6.5 cm out, as VAMP's spherized URDF fixes them), with the canonical SRDF's pairs.
fn mesh_model(inputs: &Path) -> Result<MeshModel> {
    let asset = |path: &str| format!("{}/assets/franka/{path}", env!("CARGO_MANIFEST_DIR"));
    let fingers = ["panda_finger_joint1", "panda_finger_joint2"].map(|j| (j.to_string(), 0.065));
    let options = RobotOptions {
        lock_joints: fingers.into_iter().collect(),
        collision_model: Some(CollisionModel::load(asset("panda_collision.json"))?),
        srdf: Some(std::fs::read_to_string(inputs.join("panda.srdf"))?),
        ..Default::default()
    };
    Ok(MeshModel::try_new(&Robot::load(
        asset("franka_panda.urdf"),
        &options,
    )?)?)
}

/// `waypoints` (`[k, dof]`) with states added along each edge, at most `step` apart in every joint.
fn densified(waypoints: &[f32], dof: usize, step: f32) -> Vec<f32> {
    let mut dense = waypoints[..dof].to_vec();
    for (a, b) in waypoints.chunks(dof).zip(waypoints.chunks(dof).skip(1)) {
        let widest = a
            .iter()
            .zip(b)
            .map(|(x, y)| (y - x).abs())
            .fold(0.0, f32::max);
        let k = ((widest / step).ceil() as usize).max(1);
        for s in 1..=k {
            let t = s as f32 / k as f32;
            dense.extend(a.iter().zip(b).map(|(x, y)| x + (y - x) * t));
        }
    }
    dense
}

/// Checks every answer of a run's first repetition on the meshes (`mesh_model`), in the exact
/// scene, sampled as the judge samples: the final output (the timed trajectory in workload B),
/// densified to 0.005 rad. Writes, per problem and seed, whether the output is clear, whether its
/// problem's start and goal are, and its least world and self clearance.
fn mesh(inputs: &Path, run_dir: &Path) -> Result<()> {
    use rayon::prelude::*;

    let model = mesh_model(inputs)?;
    let n = 7;
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json"))?)?;
    let trajectories = manifest["workload"] == "B";
    let problems: Vec<Problem> =
        serde_json::from_str(&std::fs::read_to_string(inputs.join("problems.json"))?)?;
    let queries: Vec<Query> = serde_json::from_str(&std::fs::read_to_string(
        inputs.join("queries_onescene.json"),
    )?)?;
    let by_id: HashMap<&str, &Problem> = problems.iter().map(|p| (p.id.as_str(), p)).collect();
    // Each answered id's ends and its scene's problem.
    let mut ends: HashMap<&str, (&[f32], &[f32], &Problem)> = problems
        .iter()
        .map(|p| (p.id.as_str(), (&p.start[..], &p.goal[..], p)))
        .collect();
    for q in &queries {
        let p = by_id
            .get(q.scene_ref.as_str())
            .with_context(|| format!("no scene {}", q.scene_ref))?;
        ends.insert(&q.id, (&q.start, &q.goal, p));
    }
    let records: Vec<serde_json::Value> = std::fs::read_to_string(run_dir.join("records.jsonl"))?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let answered: Vec<(String, u64)> = records
        .iter()
        .filter(|r| {
            r["rep"] == 1 && r.get("api").is_none_or(|a| a == "single") && r["status"] == "solved"
        })
        .map(|r| {
            let problem = r["problem"].as_str().context("a problem id")?.to_string();
            Ok((problem, r["seed"].as_u64().context("a seed")?))
        })
        .collect::<Result<_>>()?;
    // The outputs of each seed: values and index.
    let mut outputs: HashMap<u64, (Vec<f32>, serde_json::Value)> = HashMap::new();
    for &(_, seed) in &answered {
        if outputs.contains_key(&seed) {
            continue;
        }
        let stem = run_dir.join(format!("outputs_seed{seed}"));
        let bytes = std::fs::read(stem.with_extension("bin"))?;
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        let index = serde_json::from_str(&std::fs::read_to_string(stem.with_extension("json"))?)?;
        outputs.insert(seed, (values, index));
    }
    let output = |problem: &str, seed: u64| -> Result<&[f32]> {
        let (values, index) = &outputs[&seed];
        let key = [format!("{problem}|q"), format!("{problem}|path")]
            .into_iter()
            .find(|k| (trajectories || k.ends_with("|path")) && index.get(k).is_some())
            .with_context(|| format!("no output for {problem} seed {seed}"))?;
        let at = |i: usize| index[&key][i].as_u64().map(|v| v as usize);
        let (offset, rows, cols) = (at(0), at(1), at(2));
        let (Some(offset), Some(rows), Some(7)) = (offset, rows, cols) else {
            bail!("{key}: not [offset, rows, 7]");
        };
        Ok(&values[offset..offset + rows * n])
    };
    // Each distinct scene once, and per answer its densified output, scene and ends.
    let mut scene_index: HashMap<&str, u32> = HashMap::new();
    let (mut worlds, mut jobs) = (vec![], vec![]);
    for (problem, seed) in &answered {
        let &(start, goal, p) = ends.get(problem.as_str()).context("an answered id")?;
        let w = *scene_index.entry(p.id.as_str()).or_insert_with(|| {
            worlds.push(world(&p.scene));
            worlds.len() as u32 - 1
        });
        let path = densified(output(problem, *seed)?, n, 0.005);
        jobs.push((path, w, [start, goal].concat()));
    }
    let least = |c: &[Clearance]| {
        c.iter().fold((f32::INFINITY, f32::INFINITY), |(w, s), c| {
            (w.min(c.world), s.min(c.self_collision))
        })
    };
    let verdicts: Vec<serde_json::Value> = jobs
        .par_iter()
        .zip(&answered)
        .map(|((path, w, ends), (problem, seed))| {
            let (world, self_collision) =
                least(&model.clearance(&worlds, &vec![*w; path.len() / n], path)?);
            let (end_world, end_self) = least(&model.clearance(&worlds, &[*w; 2], ends)?);
            Ok(json!({
                "problem": problem, "seed": seed,
                "clear": world > 0.0 && self_collision > 0.0,
                "ends_clear": end_world > 0.0 && end_self > 0.0,
                "world": world, "self": self_collision,
            }))
        })
        .collect::<Result<_>>()?;
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(run_dir.join("mesh_verdicts.jsonl"))?);
    for v in &verdicts {
        writeln!(out, "{v}")?;
    }
    let clear = verdicts.iter().filter(|v| v["clear"] == true).count();
    println!(
        "{}: {clear} of {} answers clear on the meshes",
        run_dir.display(),
        verdicts.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["parity", inputs, fixtures, out] => {
            parity(Path::new(inputs), Path::new(fixtures), Path::new(out))
        }
        ["run", job, out] => run(Path::new(job), Path::new(out)),
        ["mesh", inputs, run_dir] => mesh(Path::new(inputs), Path::new(run_dir)),
        _ => bail!(
            "usage: compare parity <inputs_dir> <fixtures.json> <out.json> | run <job.json> \
             <out_dir> | mesh <inputs_dir> <run_dir>"
        ),
    }
}
