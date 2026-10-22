//! Demonstrations as plain NumPy arrays: one `.npy` file per field, loadable with `numpy.load`
//! and nothing else. The dependency-free counterpart of [`crate::lerobot`].

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use bytemuck::Pod;
use serde_json::{Map, Value, json};

use crate::datagen::{self, Demonstration};
use crate::error::{Result, ensure_input};
use crate::robot::Robot;
use crate::world::{self, World};

#[derive(Clone, Debug, Default)]
pub struct ExportOptions {
    /// Extra entries merged into `meta.json`, e.g. generation parameters.
    pub metadata: Map<String, Value>,
}

/// Writes `demos` under `root`, which must not already hold a dataset:
/// - `positions.npy`, `velocities.npy`: `[episodes, steps, dof]` float32, padded past each
///   episode's `length` with its final position and zero velocity.
/// - `gripper.npy`: `[episodes, steps, hands]` float32, each hand's gripper opening (1 open, 0
///   closed), padded with its final value.
/// - `carried.npy` (int32: the obstacle of the episode's world it moves, -1 if none) and
///   `carried_pose.npy`: `[episodes, steps, 7]` float32, that obstacle's position xyz + quaternion
///   xyzw, padded with its final pose (zero where nothing is carried).
/// - `task.npy` (int32, `[episodes]`): each episode's index into `meta.json`'s `tasks`.
/// - `length.npy` (int32), `kind.npy` (uint8: 0 nominal, 1 recovery), `parent.npy` (int32: the
///   episode a recovery branches from, -1 otherwise), `world.npy` (int32): `[episodes]`.
/// - `goal_pose.npy`: `[episodes, 7]` float32, target position xyz + quaternion xyzw.
/// - `worlds.json` (the obstacles of every world, each distance grid written once) and `meta.json`.
///
/// All demonstrations must share one sample period.
pub fn export(
    root: impl AsRef<Path>,
    robot: &Robot,
    worlds: &[World],
    demos: &[Demonstration],
    o: &ExportOptions,
) -> Result<()> {
    let root = root.as_ref();
    let dt = datagen::check_demos(robot, worlds, demos)?;
    ensure_input!(
        !root.join("meta.json").exists(),
        "{} already holds a dataset",
        root.display()
    );
    std::fs::create_dir_all(root)?;

    let (n, m) = (robot.dof(), demos.len());
    let horizon = demos.iter().map(|d| d.trajectory.len()).max().unwrap_or(0);
    let mut positions = vec![0.0f32; m * horizon * n];
    let mut velocities = vec![0.0f32; m * horizon * n];
    for (i, d) in demos.iter().enumerate() {
        let t = &d.trajectory;
        for h in 0..horizon {
            let (src, dst) = (h.min(t.len() - 1) * n, (i * horizon + h) * n);
            positions[dst..dst + n].copy_from_slice(&t.positions[src..src + n]);
            if h < t.len() {
                velocities[dst..dst + n].copy_from_slice(&t.velocities[src..src + n]);
            }
        }
    }
    let padded = |i: usize, h: usize| h.min(demos[i].trajectory.len() - 1);
    let hands = robot.hands();
    let gripper: Vec<f32> = (0..m)
        .flat_map(|i| {
            (0..horizon).flat_map(move |h| {
                let row = padded(i, h) * hands;
                demos[i].gripper[row..row + hands].iter().copied()
            })
        })
        .collect();
    let carried_pose: Vec<f32> = (0..m)
        .flat_map(|i| {
            (0..horizon).flat_map(move |h| {
                demos[i].carried.as_ref().map_or([0.0; 7], |c| {
                    let (p, q) = (
                        c.poses[padded(i, h)].position,
                        c.poses[padded(i, h)].rotation,
                    );
                    [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
                })
            })
        })
        .collect();
    let mut tasks: Vec<&str> = vec![];
    let task: Vec<i32> = demos
        .iter()
        .map(|d| {
            let k = tasks.iter().position(|&t| t == d.task).unwrap_or_else(|| {
                tasks.push(&d.task);
                tasks.len() - 1
            });
            k as i32
        })
        .collect();
    let column = |f: &dyn Fn(&Demonstration) -> i32| demos.iter().map(f).collect::<Vec<i32>>();
    write(root.join("positions.npy"), &[m, horizon, n], &positions)?;
    write(root.join("gripper.npy"), &[m, horizon, hands], &gripper)?;
    write(
        root.join("carried.npy"),
        &[m],
        &column(&|d| d.carried.as_ref().map_or(-1, |c| c.obstacle as i32)),
    )?;
    write(
        root.join("carried_pose.npy"),
        &[m, horizon, 7],
        &carried_pose,
    )?;
    write(root.join("task.npy"), &[m], &task)?;
    write(root.join("velocities.npy"), &[m, horizon, n], &velocities)?;
    write(
        root.join("length.npy"),
        &[m],
        &column(&|d| d.trajectory.len() as i32),
    )?;
    write(
        root.join("parent.npy"),
        &[m],
        &column(&|d| d.origin.parent().map_or(-1, |p| p as i32)),
    )?;
    write(root.join("world.npy"), &[m], &column(&|d| d.world as i32))?;
    let kinds: Vec<u8> = demos
        .iter()
        .map(|d| u8::from(d.origin.parent().is_some()))
        .collect();
    write(root.join("kind.npy"), &[m], &kinds)?;
    let poses: Vec<f32> = demos
        .iter()
        .flat_map(|d| {
            let (p, q) = (d.goal.position, d.goal.rotation);
            [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        })
        .collect();
    write(root.join("goal_pose.npy"), &[m, 7], &poses)?;
    std::fs::write(
        root.join("worlds.json"),
        serde_json::to_string(&world::worlds_json(worlds)?)?,
    )?;

    let recoveries = kinds.iter().filter(|&&k| k == 1).count();
    let mut meta = json!({
        "dt": dt,
        "joint_names": robot.joint_names(),
        "max_velocity": robot.max_velocity(),
        "episodes": m,
        "nominal": m - recoveries,
        "recovery": recoveries,
        "kind": {"0": "nominal", "1": "recovery from a perturbed state of episode `parent`"},
        "goal_pose": "the pose the episode drives toward (the IK frame's for a reach, the object's for pick-and-place): position xyz + quaternion xyzw",
        "tasks": tasks,
    });
    meta.as_object_mut()
        .expect("meta is an object")
        .extend(o.metadata.clone());
    std::fs::write(root.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

trait Element: Pod {
    const DESCR: &'static str;
}

impl Element for f32 {
    const DESCR: &'static str = "<f4";
}

impl Element for i32 {
    const DESCR: &'static str = "<i4";
}

impl Element for u8 {
    const DESCR: &'static str = "|u1";
}

/// One `.npy` file (format version 1.0, little-endian, C order).
fn write<T: Element>(path: impl AsRef<Path>, shape: &[usize], data: &[T]) -> Result<()> {
    use std::io::Write;

    ensure_input!(
        shape.iter().product::<usize>() == data.len(),
        "shape {shape:?} does not match {} elements",
        data.len()
    );
    let dims = match shape {
        [n] => format!("{n},"),
        _ => shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({dims}), }}",
        T::DESCR
    );
    // Magic (6) + version (2) + header length (2) + header must be a multiple of 64.
    let pad = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    f.write_all(bytemuck::cast_slice(data))?;
    f.flush()?;
    Ok(())
}
