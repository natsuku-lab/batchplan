//! Optional bridge to LeRobot: writes [`Demonstration`]s as a LeRobot v3.0 dataset that
//! `lerobot.datasets.LeRobotDataset` loads directly. Enabled by the `lerobot` cargo feature.
//!
//! Each demonstration becomes one episode with one frame per trajectory sample, and its task one
//! of the dataset's tasks:
//! - `observation.state`: joint positions and each hand's gripper opening (1 open, 0 closed), in
//!   per-arm groups as ALOHA lays out two arms: the joints no hand moves, then for each hand the
//!   joints that move it and its gripper. One arm's state is its joints, then its gripper. Grippers
//!   are named `gripper`, or after their part's prefix (`left_gripper`) on a robot with several hands.
//! - `action`: the state at the next frame (the last frame repeats its own).
//! - `observation.environment_state`: the goal pose (position xyz, quaternion xyzw with w >= 0),
//!   then every obstacle of the episode's world, where it is at that frame, as `[present, kind,
//!   center xyz, size xyz, quaternion xyzw]`, zero-padded to the largest world. `kind` is 0 cuboid,
//!   1 sphere, 2 cylinder, 3 capsule, 4 distance grid; `size` is the half extents of a cuboid or of
//!   a grid's box, else (radius, radius, half height or half length).
//! - `is_recovery`, `parent_episode_index` (-1 for nominal episodes) and `world_index`: extra
//!   columns that LeRobot policies ignore.
//!
//! `meta/batchplan.json` additionally holds the worlds (`grids` and `worlds`, as in the `.npy`
//! export's `worlds.json`), the environment-state layout, and the robot: the descriptions it was
//! assembled from (`Robot::parts`), its joints, and each hand's frame, joints and gripper, what a
//! simulator needs to replay the episodes and render them.

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{Error, Result, ensure_input};
use arrow_array::builder::{Float64Builder, Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde_json::{Map, Value, json};

use crate::datagen::{Demonstration, Origin, check_demos};
use crate::robot::Robot;
use crate::types::Pose;
use crate::world::{Obstacle, World, worlds_json};

const CODEBASE_VERSION: &str = "v3.0";
const CHUNKS_SIZE: usize = 1000;
const VIDEO_FILE_SIZE_IN_MB: usize = 200;
/// Episode-metadata rows written at a time.
const EPISODES_PER_WRITE: usize = 1000;
const DATA_PATH: &str = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet";
const QUANTILES: [(f64, &str); 5] = [(0.01, "q01"), (0.10, "q10"), (0.50, "q50"), (0.90, "q90"), (0.99, "q99")];
const GOAL_WIDTH: usize = 7;
const OBSTACLE_WIDTH: usize = 12;

#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// `robot_type` in `meta/info.json`; the URDF robot name when `None`.
    pub robot_type: Option<String>,
    /// Frame data and episode metadata start a new file past this size, as LeRobot's own
    /// `data_files_size_in_mb`.
    pub data_files_size_in_mb: usize,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self { robot_type: None, data_files_size_in_mb: 100 }
    }
}

/// Writes `demos` of `robot` in `worlds` as a LeRobot v3.0 dataset under `root`, which must not
/// already hold one. All demonstrations must share a sample period that is a whole number of
/// frames per second. Frames and episodes are written as they are converted, so the dataset can
/// exceed memory. It holds no camera images: for those, replay the episodes in a simulator.
pub fn export(
    root: impl AsRef<Path>,
    robot: &Robot,
    worlds: &[World],
    demos: &[Demonstration],
    o: &ExportOptions,
) -> Result<()> {
    let root = root.as_ref();
    let dt = check_demos(robot, worlds, demos)?;
    ensure_input!(!root.join("meta/info.json").exists(), "{} already holds a LeRobot dataset", root.display());
    ensure_input!(o.data_files_size_in_mb > 0, "data files need a positive size limit");
    let fps = (1.0 / dt).round();
    ensure_input!(
        fps >= 1.0 && (fps * dt - 1.0).abs() < 1e-4,
        "LeRobot needs a whole number of frames per second; dt = {dt} s is {} fps",
        1.0 / dt
    );
    let fps = fps as u32;
    let max_obstacles = demos.iter().map(|d| worlds[d.world as usize].obstacles.len()).max().unwrap_or(0);
    // Tasks in order of first appearance.
    let mut tasks: Vec<&str> = vec![];
    let mut task_index: HashMap<&str, usize> = HashMap::new();
    for d in demos {
        task_index.entry(&d.task).or_insert_with(|| {
            tasks.push(&d.task);
            tasks.len() - 1
        });
    }

    let limit = o.data_files_size_in_mb << 20;
    let mut data = Files::new(root.join("data"), limit);
    let mut episode_files = Files::new(root.join("meta/episodes"), limit);
    // Features in column order (`Map` sorts its keys) and their dataset statistics.
    let (mut rows, mut names, mut features, mut totals) = (vec![], vec![], Map::new(), vec![]);
    let mut index = 0usize;
    for (e, demo) in demos.iter().enumerate() {
        let world = &worlds[demo.world as usize];
        let columns = episode_columns(robot, world, demo, e, index, fps, max_obstacles, task_index[demo.task.as_str()]);
        let batch = record_batch(&columns)?;
        let (chunk, file) = data.place(batch.get_array_memory_size())?;
        data.write(&batch)?;
        let stats: Vec<Stats> = columns.iter().map(Column::stats).collect();
        if e == 0 {
            names = columns.iter().map(|c| c.name.clone()).collect();
            features = columns.iter().map(|c| (c.name.clone(), c.feature())).collect();
            totals = stats.iter().map(|_| Aggregate::default()).collect();
        }
        totals.iter_mut().zip(&stats).for_each(|(t, s)| t.add(s));
        let length = demo.trajectory.len();
        rows.push(EpisodeRow { length, chunk, file, from: index, task: demo.task.clone(), stats });
        if rows.len() == EPISODES_PER_WRITE {
            write_episodes(&mut episode_files, e + 1 - rows.len(), &rows, &names)?;
            rows.clear();
        }
        index += length;
    }
    if !rows.is_empty() {
        write_episodes(&mut episode_files, demos.len() - rows.len(), &rows, &names)?;
    }
    data.finish()?;
    episode_files.finish()?;

    let meta = root.join("meta");
    write_parquet(&meta.join("tasks.parquet"), &tasks_batch(&tasks)?)?;
    let info = json!({
        "codebase_version": CODEBASE_VERSION,
        "robot_type": o.robot_type.clone().unwrap_or_else(|| robot.name().to_string()),
        "total_episodes": demos.len(),
        "total_frames": index,
        "total_tasks": tasks.len(),
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": o.data_files_size_in_mb,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_IN_MB,
        "fps": fps,
        "splits": {"train": format!("0:{}", demos.len())},
        "data_path": DATA_PATH,
        "video_path": null,
        "features": features,
    });
    let stats: Map<String, Value> =
        names.iter().zip(&totals).map(|(name, t)| (name.clone(), t.stats().to_json())).collect();
    let extension = json!({
        "generator": "batchplan",
        "environment_state": {
            "goal": "[0:7] position xyz, quaternion xyzw (w >= 0)",
            "obstacles": format!(
                "[7 + {OBSTACLE_WIDTH}k : 7 + {OBSTACLE_WIDTH}(k+1)] obstacle k where it is at the frame: present, kind (0 cuboid, 1 sphere, 2 cylinder, 3 capsule, 4 distance grid), center xyz, size xyz (half extents of a cuboid or a grid's box, else radius, radius, half height or half length), quaternion xyzw"
            ),
            "max_obstacles": max_obstacles,
        },
        "episodes": demos.iter().map(|d| {
            let mut e = match d.origin {
                Origin::Nominal => json!({"origin": "nominal"}),
                Origin::Recovery { parent, phase } => json!({"origin": "recovery", "parent": parent, "phase": phase}),
            };
            if let Some(c) = &d.carried {
                e["carried_obstacle"] = json!(c.obstacle);
            }
            e
        }).collect::<Vec<_>>(),
        "worlds": worlds_json(worlds)?,
        "robot": robot_json(robot),
    });
    fs::write(meta.join("info.json"), serde_json::to_string_pretty(&info)?)?;
    fs::write(meta.join("stats.json"), serde_json::to_string_pretty(&stats)?)?;
    fs::write(meta.join("batchplan.json"), serde_json::to_string(&extension)?)?;
    Ok(())
}

enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

/// One per-frame feature: `width` values per frame (1 = scalar).
struct Column {
    name: String,
    width: usize,
    names: Option<Vec<String>>,
    data: Data,
}

impl Column {
    fn f32(name: &str, width: usize, names: Option<Vec<String>>, values: Vec<f32>) -> Self {
        Self { name: name.into(), width, names, data: Data::F32(values) }
    }

    fn i64(name: &str, values: Vec<i64>) -> Self {
        Self { name: name.into(), width: 1, names: None, data: Data::I64(values) }
    }

    fn feature(&self) -> Value {
        let dtype = match self.data {
            Data::F32(_) => "float32",
            Data::I64(_) => "int64",
        };
        json!({"dtype": dtype, "shape": [self.width], "names": self.names})
    }

    fn values(&self) -> Vec<f64> {
        match &self.data {
            Data::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Data::I64(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }

    fn arrow(&self) -> (Field, ArrayRef) {
        match (&self.data, self.width) {
            (Data::F32(v), 1) => {
                (Field::new(&self.name, DataType::Float32, true), Arc::new(Float32Array::from(v.clone())))
            }
            (Data::I64(v), _) => (Field::new(&self.name, DataType::Int64, true), Arc::new(Int64Array::from(v.clone()))),
            (Data::F32(v), w) => {
                let item = Arc::new(Field::new_list_field(DataType::Float32, true));
                let array =
                    FixedSizeListArray::new(item.clone(), w as i32, Arc::new(Float32Array::from(v.clone())), None);
                (Field::new(&self.name, DataType::FixedSizeList(item, w as i32), true), Arc::new(array))
            }
        }
    }

    /// Per-dimension statistics over all frames, as LeRobot computes them.
    fn stats(&self) -> Stats {
        let values = self.values();
        let frames = values.len() / self.width;
        let mut s = Stats { count: frames, ..Default::default() };
        for d in 0..self.width {
            let mut x: Vec<f64> = (0..frames).map(|f| values[f * self.width + d]).collect();
            let mean = x.iter().sum::<f64>() / frames as f64;
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / frames as f64;
            x.sort_by(f64::total_cmp);
            s.min.push(x[0]);
            s.max.push(x[frames - 1]);
            s.mean.push(mean);
            s.std.push(var.sqrt());
            for (q, (level, _)) in s.quantiles.iter_mut().zip(QUANTILES) {
                // Linear interpolation between order statistics (numpy's default).
                let pos = level * (frames - 1) as f64;
                let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
                q.push(x[lo] + (pos - lo as f64) * (x[hi] - x[lo]));
            }
        }
        s
    }
}

/// Dataset statistics from per-episode ones, combined as LeRobot's `aggregate_feature_stats`
/// combines them: count-weighted means and quantiles, and the variance of the union.
#[derive(Default)]
struct Aggregate {
    count: f64,
    min: Vec<f64>,
    max: Vec<f64>,
    /// Count-weighted sums of the means, of the second moments and of each quantile.
    mean: Vec<f64>,
    second: Vec<f64>,
    quantiles: [Vec<f64>; 5],
}

impl Aggregate {
    fn add(&mut self, s: &Stats) {
        let c = s.count as f64;
        if self.min.is_empty() {
            let zeros = vec![0.0; s.mean.len()];
            (self.min, self.max) = (s.min.clone(), s.max.clone());
            (self.mean, self.second) = (zeros.clone(), zeros.clone());
            self.quantiles = std::array::from_fn(|_| zeros.clone());
        }
        self.count += c;
        for d in 0..s.mean.len() {
            self.min[d] = self.min[d].min(s.min[d]);
            self.max[d] = self.max[d].max(s.max[d]);
            self.mean[d] += c * s.mean[d];
            self.second[d] += c * (s.std[d].powi(2) + s.mean[d].powi(2));
            for (q, sq) in self.quantiles.iter_mut().zip(&s.quantiles) {
                q[d] += c * sq[d];
            }
        }
    }

    fn stats(&self) -> Stats {
        let mean: Vec<f64> = self.mean.iter().map(|m| m / self.count).collect();
        let std = self.second.iter().zip(&mean).map(|(s, m)| (s / self.count - m * m).max(0.0).sqrt()).collect();
        Stats {
            min: self.min.clone(),
            max: self.max.clone(),
            mean,
            std,
            count: self.count as usize,
            quantiles: self.quantiles.clone().map(|q| q.iter().map(|v| v / self.count).collect()),
        }
    }
}

#[derive(Clone, Default)]
struct Stats {
    min: Vec<f64>,
    max: Vec<f64>,
    mean: Vec<f64>,
    std: Vec<f64>,
    count: usize,
    quantiles: [Vec<f64>; 5],
}

impl Stats {
    /// `(name, values)` in LeRobot's order; `count` is reported separately.
    fn vectors(&self) -> Vec<(&'static str, &[f64])> {
        let mut v =
            vec![("min", &self.min[..]), ("max", &self.max[..]), ("mean", &self.mean[..]), ("std", &self.std[..])];
        v.extend(QUANTILES.iter().zip(&self.quantiles).map(|((_, name), q)| (*name, &q[..])));
        v
    }

    fn to_json(&self) -> Value {
        let mut m: Map<String, Value> = self.vectors().into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        m.insert("count".into(), json!([self.count]));
        Value::Object(m)
    }
}

/// One entry of the state: a joint, or a hand's gripper.
enum Slot {
    Joint(usize),
    Gripper(usize),
}

/// The state's entries: the joints no hand moves, then each hand's joints and its gripper.
fn state_layout(robot: &Robot) -> Vec<Slot> {
    let hands: Vec<Vec<usize>> = (0..robot.hands()).map(|h| robot.hand_joints(h)).collect();
    let mut layout: Vec<Slot> =
        (0..robot.dof()).filter(|j| hands.iter().all(|joints| !joints.contains(j))).map(Slot::Joint).collect();
    let mut placed = vec![false; robot.dof()];
    for (h, joints) in hands.iter().enumerate() {
        // A joint two hands share (a torso) goes with the first.
        for &j in joints {
            if !std::mem::replace(&mut placed[j], true) {
                layout.push(Slot::Joint(j));
            }
        }
        layout.push(Slot::Gripper(h));
    }
    layout
}

/// Hand `h`'s gripper: `gripper` on a one-handed robot, else after the prefix of the part its
/// frame belongs to (`left_gripper`), or its frame's name.
fn gripper_name(robot: &Robot, h: usize) -> String {
    if robot.hands() == 1 {
        return "gripper".into();
    }
    match robot.hand_prefix(h) {
        Some(prefix) => format!("{prefix}gripper"),
        None => format!("{}_gripper", robot.links[robot.ee_links[h]].name),
    }
}

fn state_names(robot: &Robot) -> Vec<String> {
    state_layout(robot)
        .iter()
        .map(|slot| match *slot {
            Slot::Joint(j) => robot.joint_names()[j].clone(),
            Slot::Gripper(h) => gripper_name(robot, h),
        })
        .collect()
}

/// The robot in `meta/batchplan.json`.
fn robot_json(robot: &Robot) -> Value {
    let pose = |p: Pose| {
        let (t, r) = (p.position, p.rotation);
        json!([t.x, t.y, t.z, r.x, r.y, r.z, r.w])
    };
    json!({
        "name": robot.name(),
        "parts": robot.parts().iter().map(|p| json!({
            "path": p.path.to_string_lossy(),
            "prefix": p.prefix,
            "base": pose(p.base),
        })).collect::<Vec<_>>(),
        "joints": robot.joint_names(),
        "hands": (0..robot.hands()).map(|h| json!({
            "frame": robot.links[robot.ee_links[h]].name,
            "joints": robot.hand_joints(h).iter().map(|&j| robot.joint_names()[j].clone()).collect::<Vec<_>>(),
            "gripper": gripper_name(robot, h),
        })).collect::<Vec<_>>(),
    })
}

#[allow(clippy::too_many_arguments)]
fn episode_columns(
    robot: &Robot,
    world: &World,
    demo: &Demonstration,
    episode: usize,
    first_index: usize,
    fps: u32,
    max_obstacles: usize,
    task: usize,
) -> Vec<Column> {
    let (n, hands) = (robot.dof(), robot.hands());
    let traj = &demo.trajectory;
    let frames = traj.len();
    let layout = state_layout(robot);
    let width = layout.len();
    let state: Vec<f32> = (0..frames)
        .flat_map(|f| {
            layout.iter().map(move |slot| match *slot {
                Slot::Joint(j) => traj.positions[f * n + j],
                Slot::Gripper(h) => demo.gripper[f * hands + h],
            })
        })
        .collect();
    let mut action = state[width..].to_vec();
    action.extend_from_slice(&state[(frames - 1) * width..]);
    let env: Vec<f32> = (0..frames)
        .flat_map(|f| {
            let carried = demo.carried.as_ref().map(|c| (c.obstacle, c.poses[f]));
            environment_state(world, demo.goal, carried, max_obstacles)
        })
        .collect();
    let (is_recovery, parent) =
        (i64::from(demo.origin.parent().is_some()), demo.origin.parent().map_or(-1, |p| p as i64));
    let names = state_names(robot);
    let per_frame = |v: i64| vec![v; frames];
    vec![
        Column::f32("observation.state", width, Some(names.clone()), state),
        Column::f32("observation.environment_state", env.len() / frames, Some(environment_names(max_obstacles)), env),
        Column::f32("action", width, Some(names), action),
        Column::i64("is_recovery", per_frame(is_recovery)),
        Column::i64("parent_episode_index", per_frame(parent)),
        Column::i64("world_index", per_frame(demo.world as i64)),
        Column::f32("timestamp", 1, None, (0..frames).map(|f| f as f32 / fps as f32).collect()),
        Column::i64("frame_index", (0..frames as i64).collect()),
        Column::i64("episode_index", per_frame(episode as i64)),
        Column::i64("index", (first_index as i64..(first_index + frames) as i64).collect()),
        Column::i64("task_index", per_frame(task as i64)),
    ]
}

/// The goal, then every obstacle of `world`, `carried` (an obstacle's index and pose) where it
/// is now.
fn environment_state(world: &World, goal: Pose, carried: Option<(usize, Pose)>, max_obstacles: usize) -> Vec<f32> {
    let canonical = |q: glam::Quat| if q.w < 0.0 { -q } else { q };
    let (p, q) = (goal.position, canonical(goal.rotation));
    let mut env = vec![p.x, p.y, p.z, q.x, q.y, q.z, q.w];
    for (k, o) in world.obstacles.iter().enumerate() {
        let o = match carried {
            Some((c, pose)) if c == k => o.placed(pose),
            _ => o.clone(),
        };
        let round = |radius: f32, half: f32| glam::Vec3::new(radius, radius, half);
        let (kind, center, size, rot) = match o {
            Obstacle::Cuboid { center, half_extents, rotation } => (0.0, center, half_extents, rotation),
            Obstacle::Sphere { center, radius } => (1.0, center, glam::Vec3::splat(radius), glam::Quat::IDENTITY),
            Obstacle::Cylinder { center, rotation, radius, half_height } => {
                (2.0, center, round(radius, half_height), rotation)
            }
            Obstacle::Capsule { center, rotation, radius, half_length } => {
                (3.0, center, round(radius, half_length), rotation)
            }
            Obstacle::Sdf { ref grid, center, rotation } => {
                let (lo, hi) = grid.bounds();
                (4.0, center + rotation * (0.5 * (lo + hi)), 0.5 * (hi - lo), rotation)
            }
        };
        let rot = canonical(rot);
        env.extend([1.0, kind, center.x, center.y, center.z, size.x, size.y, size.z, rot.x, rot.y, rot.z, rot.w]);
    }
    env.resize(GOAL_WIDTH + OBSTACLE_WIDTH * max_obstacles, 0.0);
    env
}

fn environment_names(max_obstacles: usize) -> Vec<String> {
    let mut names: Vec<String> = ["x", "y", "z", "qx", "qy", "qz", "qw"].iter().map(|s| format!("goal.{s}")).collect();
    for k in 0..max_obstacles {
        let fields = ["present", "kind", "x", "y", "z", "sx", "sy", "sz", "qx", "qy", "qz", "qw"];
        names.extend(fields.iter().map(|f| format!("obstacle{k}.{f}")));
    }
    names
}

fn record_batch(columns: &[Column]) -> Result<RecordBatch> {
    let (fields, arrays): (Vec<Field>, Vec<ArrayRef>) = columns.iter().map(Column::arrow).unzip();
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?)
}

struct EpisodeRow {
    length: usize,
    chunk: usize,
    file: usize,
    from: usize,
    task: String,
    stats: Vec<Stats>,
}

/// Appends `rows`, episodes `first..`, to the episode-metadata files, each row recording the file
/// it lands in.
fn write_episodes(files: &mut Files, first: usize, rows: &[EpisodeRow], features: &[String]) -> Result<()> {
    let size = episodes_batch(first, rows, features, (0, 0))?.get_array_memory_size();
    let at = files.place(size)?;
    files.write(&episodes_batch(first, rows, features, at)?)
}

/// `meta/episodes` rows: boundaries into the data files plus flattened per-episode stats of each
/// of `features` (in column order), all in the episode-metadata file `at` (chunk, file).
fn episodes_batch(
    first: usize,
    episodes: &[EpisodeRow],
    features: &[String],
    at: (usize, usize),
) -> Result<RecordBatch> {
    let int_column = |f: &dyn Fn(usize, &EpisodeRow) -> usize| -> ArrayRef {
        Arc::new(Int64Array::from(episodes.iter().enumerate().map(|(i, e)| f(i, e) as i64).collect::<Vec<_>>()))
    };
    let mut tasks = ListBuilder::new(StringBuilder::new());
    for e in episodes {
        tasks.values().append_value(&e.task);
        tasks.append(true);
    }
    let mut columns: Vec<(String, ArrayRef)> = vec![
        ("episode_index".into(), int_column(&|i, _| first + i)),
        ("tasks".into(), Arc::new(tasks.finish())),
        ("length".into(), int_column(&|_, e| e.length)),
        ("data/chunk_index".into(), int_column(&|_, e| e.chunk)),
        ("data/file_index".into(), int_column(&|_, e| e.file)),
        ("dataset_from_index".into(), int_column(&|_, e| e.from)),
        ("dataset_to_index".into(), int_column(&|_, e| e.from + e.length)),
        ("meta/episodes/chunk_index".into(), int_column(&|_, _| at.0)),
        ("meta/episodes/file_index".into(), int_column(&|_, _| at.1)),
    ];
    for (f, feature) in features.iter().enumerate() {
        let names: Vec<&str> = episodes[0].stats[f].vectors().iter().map(|(n, _)| *n).collect();
        for (s, stat) in names.iter().enumerate() {
            let mut b = ListBuilder::new(Float64Builder::new());
            for e in episodes {
                b.values().append_slice(e.stats[f].vectors()[s].1);
                b.append(true);
            }
            let array: ArrayRef = Arc::new(b.finish());
            columns.push((format!("stats/{feature}/{stat}"), array));
        }
        let mut count = ListBuilder::new(Int64Builder::new());
        for e in episodes {
            count.values().append_value(e.stats[f].count as i64);
            count.append(true);
        }
        columns.push((format!("stats/{feature}/count"), Arc::new(count.finish())));
    }
    let fields: Vec<Field> = columns.iter().map(|(n, a)| Field::new(n, a.data_type().clone(), true)).collect();
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns.into_iter().map(|(_, a)| a).collect())?)
}

/// `meta/tasks.parquet` as pandas writes it: a `task_index` column indexed by the task string.
fn tasks_batch(tasks: &[&str]) -> Result<RecordBatch> {
    let pandas = json!({
        "index_columns": ["task"],
        "column_indexes": [{"name": null, "field_name": null, "pandas_type": "unicode", "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}],
        "columns": [
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64", "numpy_type": "int64", "metadata": null},
            {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": null},
        ],
        "creator": {"library": "batchplan", "version": env!("CARGO_PKG_VERSION")},
        "pandas_version": "2.2.3",
    });
    let schema =
        Schema::new(vec![Field::new("task_index", DataType::Int64, true), Field::new("task", DataType::Utf8, true)])
            .with_metadata(HashMap::from([("pandas".to_string(), pandas.to_string())]));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from((0..tasks.len() as i64).collect::<Vec<_>>())),
        Arc::new(StringArray::from(tasks.to_vec())),
    ];
    Ok(RecordBatch::try_new(Arc::new(schema), arrays)?)
}

fn properties() -> WriterProperties {
    WriterProperties::builder().set_compression(Compression::SNAPPY).build()
}

fn write_parquet(path: &Path, batch: &RecordBatch) -> Result<()> {
    let mut w = ArrowWriter::try_new(File::create(path)?, batch.schema(), Some(properties()))?;
    w.write(batch)?;
    w.close()?;
    Ok(())
}

/// Parquet files under `dir` as `chunk-XXX/file-YYY.parquet`, one row group per write, starting a
/// new file once the current one would pass `limit` bytes and a new chunk every `CHUNKS_SIZE`
/// files.
struct Files {
    dir: PathBuf,
    limit: usize,
    chunk: usize,
    file: usize,
    bytes: usize,
    writer: Option<ArrowWriter<File>>,
    /// Bytes the next write takes, set by `place`.
    pending: usize,
}

impl Files {
    fn new(dir: PathBuf, limit: usize) -> Self {
        Self { dir, limit, chunk: 0, file: 0, bytes: 0, writer: None, pending: 0 }
    }

    /// The (chunk, file) the next write of `size` bytes lands in, starting a new file if needed.
    fn place(&mut self, size: usize) -> Result<(usize, usize)> {
        if self.writer.is_some() && self.bytes + size > self.limit {
            self.finish()?;
            self.file += 1;
            if self.file == CHUNKS_SIZE {
                (self.chunk, self.file) = (self.chunk + 1, 0);
            }
        }
        self.pending = size;
        Ok((self.chunk, self.file))
    }

    /// Writes `batch` where `place` put it.
    fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.writer.is_none() {
            let dir = self.dir.join(format!("chunk-{:03}", self.chunk));
            fs::create_dir_all(&dir)?;
            let path = dir.join(format!("file-{:03}.parquet", self.file));
            let file = File::create(&path).map_err(|e| Error::Write(format!("creating {}: {e}", path.display())))?;
            self.writer = Some(ArrowWriter::try_new(file, batch.schema(), Some(properties()))?);
            self.bytes = 0;
        }
        let w = self.writer.as_mut().expect("writer opened above");
        w.write(batch)?;
        w.flush()?;
        self.bytes += self.pending;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(w) = self.writer.take() {
            w.close()?;
        }
        Ok(())
    }
}

impl From<parquet::errors::ParquetError> for Error {
    fn from(e: parquet::errors::ParquetError) -> Self {
        Error::Write(e.to_string())
    }
}

impl From<arrow_schema::ArrowError> for Error {
    fn from(e: arrow_schema::ArrowError) -> Self {
        Error::Write(e.to_string())
    }
}
