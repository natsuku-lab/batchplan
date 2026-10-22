//! Dataset exporters. The LeRobot half needs `--features lerobot`; `scripts/validate_lerobot.py`
//! additionally checks that output with the real `lerobot` package.

#[path = "../examples/common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};

use batchplan::datagen::{self, Carried, DemoOptions, Demonstration};
use batchplan::rng::Rng;
use batchplan::{Device, Error, IkProblem, Pose, Robot, World};
use glam::Vec3;

fn demos(dt: f32) -> (Robot, Vec<World>, Vec<Demonstration>) {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let mut rng = Rng::new(3);
    let worlds: Vec<World> = (0..6).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = worlds
        .iter()
        .enumerate()
        .map(|(i, w)| IkProblem::new(i as u32, common::grasp_target(w, &mut rng)))
        .collect();
    let uploaded = device.upload(&worlds).unwrap();
    let demos = datagen::demonstrations(
        &device,
        &uploaded,
        &goals,
        &DemoOptions {
            dt,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        demos.iter().any(|d| d.origin.parent().is_some()),
        "fixture should include recoveries"
    );
    (robot, worlds, demos)
}

#[test]
fn demonstration_options_are_checked_before_any_work() {
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let goals = [IkProblem::new(0, robot.ee_pose(robot.default_q(), 0))];
    for o in [
        DemoOptions {
            speed_scale: (0.0, 1.0),
            ..Default::default()
        },
        DemoOptions {
            speed_scale: (0.5, 1.5),
            ..Default::default()
        },
        DemoOptions {
            dt: 0.0,
            ..Default::default()
        },
    ] {
        let err = datagen::demonstrations(&device, &worlds, &goals, &o).unwrap_err();
        assert!(matches!(err, Error::Input(_)), "{err:?}");
    }
}

fn scratch(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("batchplan-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    root
}

/// Reads a `.npy` file written by the exporter: (shape, raw little-endian data).
fn read_npy(path: &Path) -> (Vec<usize>, Vec<u8>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    assert_eq!((10 + header_len) % 64, 0, "header must be 64-byte aligned");
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    let shape = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let shape = shape
        .split(',')
        .filter_map(|d| d.trim().parse().ok())
        .collect();
    (shape, bytes[10 + header_len..].to_vec())
}

fn as_f32(data: &[u8]) -> Vec<f32> {
    data.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn as_i32(data: &[u8]) -> Vec<i32> {
    data.chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn npy_export_round_trips_the_demonstrations() {
    let (robot, worlds, mut demos) = demos(0.05);
    // The first episode closes the gripper while carrying obstacle 1 upward, under its own task.
    let frames = demos[0].trajectory.len();
    let start = worlds[demos[0].world as usize].obstacles[1].pose();
    demos[0].task = "Lift the box.".into();
    demos[0].gripper = (0..frames)
        .map(|f| 1.0 - f as f32 / (frames - 1) as f32)
        .collect();
    demos[0].carried = Some(Carried {
        obstacle: 1,
        poses: (0..frames)
            .map(|f| Pose {
                position: start.position + Vec3::Z * 0.01 * f as f32,
                ..start
            })
            .collect(),
    });
    let root = scratch("npy");
    batchplan::npy::export(&root, &robot, &worlds, &demos, &Default::default()).unwrap();

    let n = robot.dof();
    let horizon = demos.iter().map(|d| d.trajectory.len()).max().unwrap();
    let (shape, positions) = read_npy(&root.join("positions.npy"));
    assert_eq!(shape, [demos.len(), horizon, n]);
    let positions = as_f32(&positions);
    let lengths = as_i32(&read_npy(&root.join("length.npy")).1);
    let parents = as_i32(&read_npy(&root.join("parent.npy")).1);
    let kinds = read_npy(&root.join("kind.npy")).1;
    let worlds_column = as_i32(&read_npy(&root.join("world.npy")).1);
    let (gripper_shape, gripper) = read_npy(&root.join("gripper.npy"));
    assert_eq!(gripper_shape, [demos.len(), horizon, 1]);
    let gripper = as_f32(&gripper);
    let carried = as_i32(&read_npy(&root.join("carried.npy")).1);
    let carried_pose = as_f32(&read_npy(&root.join("carried_pose.npy")).1);
    let task = as_i32(&read_npy(&root.join("task.npy")).1);
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("meta.json")).unwrap()).unwrap();
    assert_eq!(
        meta["tasks"],
        serde_json::json!(["Lift the box.", batchplan::datagen::REACH_TASK])
    );
    for (i, d) in demos.iter().enumerate() {
        let t = &d.trajectory;
        assert_eq!(lengths[i] as usize, t.len());
        assert_eq!(parents[i], d.origin.parent().map_or(-1, |p| p as i32));
        assert_eq!(kinds[i], u8::from(d.origin.parent().is_some()));
        assert_eq!(worlds_column[i], d.world as i32);
        let row = &positions[i * horizon * n..(i + 1) * horizon * n];
        assert_eq!(
            &row[..t.positions.len()],
            &t.positions[..],
            "episode {i} positions"
        );
        // Padding repeats the final position.
        assert!(
            row[t.positions.len()..]
                .chunks(n)
                .all(|q| q == &t.positions[t.positions.len() - n..])
        );
        assert_eq!(
            &gripper[i * horizon..i * horizon + t.len()],
            &d.gripper[..],
            "episode {i} gripper"
        );
        assert_eq!(task[i], i32::from(i > 0), "episode {i} task");
        assert_eq!(carried[i], if i == 0 { 1 } else { -1 });
        let z = carried_pose[(i * horizon + t.len() - 1) * 7 + 2];
        let expected = d
            .carried
            .as_ref()
            .map_or(0.0, |c| c.poses[t.len() - 1].position.z);
        assert_eq!(z, expected, "episode {i}'s last carried pose");
    }
    // Exporting over an existing dataset is refused.
    assert!(batchplan::npy::export(&root, &robot, &worlds, &demos, &Default::default()).is_err());
    // Demonstrations must belong to the worlds given and match the robot.
    let elsewhere = scratch("npy-elsewhere");
    let lost = vec![Demonstration {
        world: worlds.len() as u32,
        ..demos[0].clone()
    }];
    let err = batchplan::npy::export(&elsewhere, &robot, &worlds, &lost, &Default::default())
        .unwrap_err();
    assert!(matches!(err, Error::Input(_)), "{err:?}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[cfg(feature = "lerobot")]
mod lerobot {
    use std::collections::BTreeSet;
    use std::fs::File;

    use arrow_array::{Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch};
    use batchplan::JointTrajectory;
    use batchplan::datagen::{Carried, Origin};
    use batchplan::lerobot::{self, ExportOptions};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    use super::*;

    fn write(
        root: &Path,
        robot: &Robot,
        worlds: &[World],
        demos: &[Demonstration],
        o: &ExportOptions,
    ) -> batchplan::error::Result<()> {
        lerobot::export(root, robot, worlds, demos, o)
    }

    /// Every row of every parquet file under `dir`, in file order, with the file's (chunk, file).
    fn read_all(dir: &Path) -> Vec<(usize, usize, RecordBatch)> {
        let mut files = vec![];
        for chunk in std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()) {
            for file in std::fs::read_dir(&chunk)
                .unwrap()
                .map(|e| e.unwrap().path())
            {
                files.push(file);
            }
        }
        files.sort();
        files
            .into_iter()
            .flat_map(|path| {
                let index = |s: &str| {
                    s.split('-')
                        .nth(1)
                        .unwrap()
                        .trim_end_matches(".parquet")
                        .parse::<usize>()
                        .unwrap()
                };
                let (chunk, file) = (
                    index(
                        path.parent()
                            .unwrap()
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap(),
                    ),
                    index(path.file_name().unwrap().to_str().unwrap()),
                );
                let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
                    .unwrap()
                    .build()
                    .unwrap();
                reader
                    .map(move |b| (chunk, file, b.unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn ints(batch: &RecordBatch, name: &str) -> Vec<i64> {
        batch
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    /// Each frame's values of a vector feature.
    fn vectors(batch: &RecordBatch, name: &str) -> Vec<Vec<f32>> {
        let list = batch
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        (0..list.len())
            .map(|i| {
                list.value(i)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    #[test]
    fn large_exports_spread_frames_and_episode_metadata_over_files() {
        let (robot, worlds, fixture) = demos(0.05);
        // Enough episodes to pass 1 MB of frames and of episode metadata several times over.
        let demos: Vec<Demonstration> = fixture.iter().cycle().take(1500).cloned().collect();
        let root = scratch("lerobot-large");
        write(
            &root,
            &robot,
            &worlds,
            &demos,
            &ExportOptions {
                data_files_size_in_mb: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let episodes = read_all(&root.join("meta/episodes"));
        let data = read_all(&root.join("data"));
        let files = |rows: &[(usize, usize, RecordBatch)]| {
            rows.iter()
                .map(|r| (r.0, r.1))
                .collect::<BTreeSet<_>>()
                .len()
        };
        assert!(
            files(&episodes) > 1 && files(&data) > 1,
            "{} metadata and {} data files",
            files(&episodes),
            files(&data)
        );
        let mut next = 0;
        let frames: Vec<(usize, usize, i64)> = data
            .iter()
            .flat_map(|(chunk, file, b)| {
                ints(b, "episode_index")
                    .into_iter()
                    .map(move |e| (*chunk, *file, e))
            })
            .collect();
        for (chunk, file, batch) in &episodes {
            for (row, &e) in ints(batch, "episode_index").iter().enumerate() {
                assert_eq!(e, next, "episodes in order");
                next += 1;
                // Each row names the metadata file it is in and the data file holding its frames.
                assert_eq!(
                    (
                        ints(batch, "meta/episodes/chunk_index")[row],
                        ints(batch, "meta/episodes/file_index")[row]
                    ),
                    (*chunk as i64, *file as i64)
                );
                let (from, to) = (
                    ints(batch, "dataset_from_index")[row] as usize,
                    ints(batch, "dataset_to_index")[row] as usize,
                );
                let at = (
                    ints(batch, "data/chunk_index")[row] as usize,
                    ints(batch, "data/file_index")[row] as usize,
                );
                assert!(
                    frames[from..to]
                        .iter()
                        .all(|&(c, f, fe)| (c, f) == at && fe == e),
                    "episode {e}'s frames"
                );
            }
        }
        assert_eq!(next as usize, demos.len());
        // Dataset statistics combine the episodes' as LeRobot does, weighted by length.
        let stats: serde_json::Value =
            serde_json::from_reader(File::open(root.join("meta/stats.json")).unwrap()).unwrap();
        let n = robot.dof();
        let total: usize = demos.iter().map(|d| d.trajectory.len()).sum();
        for j in 0..n {
            let values = demos
                .iter()
                .flat_map(|d| d.trajectory.positions.chunks(n).map(move |q| q[j] as f64));
            let mean = values.clone().sum::<f64>() / total as f64;
            let std =
                (values.clone().map(|v| (v - mean).powi(2)).sum::<f64>() / total as f64).sqrt();
            let min = values.fold(f64::INFINITY, f64::min);
            let s = &stats["observation.state"];
            assert!(
                (s["mean"][j].as_f64().unwrap() - mean).abs() < 1e-5,
                "joint {j} mean"
            );
            assert!(
                (s["std"][j].as_f64().unwrap() - std).abs() < 1e-5,
                "joint {j} std"
            );
            assert!(
                (s["min"][j].as_f64().unwrap() - min).abs() < 1e-6,
                "joint {j} min"
            );
        }
        assert_eq!(stats["observation.state"]["count"][0], total);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn tasks_grippers_and_carried_objects_are_exported_per_frame() {
        let (robot, worlds, fixture) = demos(0.05);
        let n = robot.dof();
        let mut demos: Vec<Demonstration> = fixture
            .into_iter()
            .filter(|d| d.origin == Origin::Nominal)
            .take(3)
            .collect();
        assert_eq!(demos.len(), 3);
        // The second episode closes the gripper and carries its world's obstacle 1 upward.
        let frames = demos[1].trajectory.len();
        let start = worlds[demos[1].world as usize].obstacles[1].pose();
        demos[1].task = "Lift the box.".into();
        demos[1].gripper = (0..frames)
            .map(|f| 1.0 - f as f32 / (frames - 1) as f32)
            .collect();
        demos[1].carried = Some(Carried {
            obstacle: 1,
            poses: (0..frames)
                .map(|f| Pose {
                    position: start.position + Vec3::Z * 0.01 * f as f32,
                    ..start
                })
                .collect(),
        });
        let start_of = |e: usize| {
            worlds[demos[e].world as usize].obstacles[1]
                .pose()
                .position
                .z
        };
        let root = scratch("lerobot-tasks");
        write(&root, &robot, &worlds, &demos, &ExportOptions::default()).unwrap();
        let tasks = read_parquet_rows(&root.join("meta/tasks.parquet"));
        assert_eq!(tasks, 2, "two distinct tasks");
        // Frames in order, whatever batches the reader groups them in.
        let (mut f, mut seen) = (0, vec![0usize; demos.len()]);
        for (_, _, batch) in read_all(&root.join("data")) {
            let (state, env, task) = (
                vectors(&batch, "observation.state"),
                vectors(&batch, "observation.environment_state"),
                ints(&batch, "task_index"),
            );
            let (episode, frame_index) =
                (ints(&batch, "episode_index"), ints(&batch, "frame_index"));
            for row in 0..batch.num_rows() {
                let (e, frame) = (episode[row] as usize, frame_index[row] as usize);
                seen[e] += 1;
                assert_eq!(
                    state[row][n], demos[e].gripper[frame],
                    "episode {e} frame {frame}: the gripper last"
                );
                assert_eq!(task[row], i64::from(e == 1), "episode {e}'s task");
                // Obstacle 1's centre in the environment state: 7 goal values, 12 per obstacle,
                // then present and kind.
                let z = env[row][7 + 12 + 4];
                let expected = demos[e]
                    .carried
                    .as_ref()
                    .map_or(start_of(e), |c| c.poses[frame].position.z);
                assert!(
                    (z - expected).abs() < 1e-6,
                    "episode {e} frame {frame}: obstacle 1 at {z}, not {expected}"
                );
                f += 1;
            }
        }
        assert!(
            seen.iter()
                .zip(&demos)
                .all(|(&s, d)| s == d.trajectory.len()),
            "every frame once"
        );
        assert!(f > 0);
        std::fs::remove_dir_all(&root).unwrap();
    }

    fn read_parquet_rows(path: &Path) -> usize {
        SerializedFileReader::new(File::open(path).unwrap())
            .unwrap()
            .metadata()
            .file_metadata()
            .num_rows() as usize
    }

    #[test]
    fn export_writes_a_consistent_v3_dataset() {
        let (robot, worlds, demos) = demos(0.05);
        let root = scratch("lerobot");
        write(&root, &robot, &worlds, &demos, &ExportOptions::default()).unwrap();

        let info: serde_json::Value =
            serde_json::from_reader(File::open(root.join("meta/info.json")).unwrap()).unwrap();
        let frames: usize = demos.iter().map(|d| d.trajectory.len()).sum();
        assert_eq!(info["codebase_version"], "v3.0");
        assert_eq!(info["fps"], 20);
        assert_eq!(info["total_episodes"], demos.len());
        assert_eq!(info["total_frames"], frames);
        assert_eq!(
            info["features"]["observation.state"]["shape"][0],
            robot.dof() + 1,
            "joints, then the gripper"
        );
        assert_eq!(
            info["features"]["observation.state"]["names"][robot.dof()],
            "gripper"
        );

        let reader =
            |path: &str| SerializedFileReader::new(File::open(root.join(path)).unwrap()).unwrap();
        let rows = |path: &str| reader(path).metadata().file_metadata().num_rows() as usize;
        assert_eq!(rows("data/chunk-000/file-000.parquet"), frames);
        assert_eq!(
            rows("meta/episodes/chunk-000/file-000.parquet"),
            demos.len()
        );
        assert_eq!(rows("meta/tasks.parquet"), 1);
        let data = reader("data/chunk-000/file-000.parquet");
        assert_eq!(
            data.metadata().num_row_groups(),
            demos.len(),
            "one row group per episode"
        );

        // Exporting over an existing dataset is refused.
        assert!(write(&root, &robot, &worlds, &demos, &ExportOptions::default()).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Two hands lay the state out arm by arm, as ALOHA does: each arm's joints, then its
    /// gripper, named after the arm; `meta/batchplan.json` describes the robot's parts and hands.
    #[test]
    fn two_hands_export_arm_by_arm() {
        let robot = common::two_pandas().unwrap();
        let n = robot.dof();
        let frames = 4;
        // Each joint's value names it (joint j at frame f is j + f / 10), each gripper its hand.
        let positions: Vec<f32> = (0..frames)
            .flat_map(|f| (0..n).map(move |j| j as f32 + f as f32 / 10.0))
            .collect();
        let trajectory = JointTrajectory {
            dof: n,
            dt: 0.05,
            duration: 0.15,
            velocities: vec![0.0; positions.len()],
            accelerations: vec![0.0; positions.len()],
            positions,
        };
        let demo = Demonstration {
            origin: Origin::Nominal,
            world: 0,
            task: "Pass the box.".into(),
            goal: Pose::IDENTITY,
            trajectory,
            gripper: (0..frames).flat_map(|_| [0.25, 0.75]).collect(),
            carried: None,
        };
        let worlds = [World::default()];
        let root = scratch("two-hands");
        write(
            &root,
            &robot,
            &worlds,
            std::slice::from_ref(&demo),
            &ExportOptions::default(),
        )
        .unwrap();

        let (left, right) = (robot.hand_joints(0), robot.hand_joints(1));
        let names = robot.joint_names();
        let expected: Vec<String> = left
            .iter()
            .map(|&j| names[j].clone())
            .chain(["left_gripper".into()])
            .chain(right.iter().map(|&j| names[j].clone()))
            .chain(["right_gripper".into()])
            .collect();
        assert!(
            expected[..7].iter().all(|n| n.starts_with("left_"))
                && expected[8..15].iter().all(|n| n.starts_with("right_"))
        );
        let info: serde_json::Value =
            serde_json::from_reader(File::open(root.join("meta/info.json")).unwrap()).unwrap();
        let state = &info["features"]["observation.state"];
        assert_eq!(state["shape"][0], n + 2);
        assert_eq!(
            state["names"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect::<Vec<_>>(),
            expected
        );
        let rows = vectors(&read_all(&root.join("data"))[0].2, "observation.state");
        for (f, row) in rows.iter().enumerate() {
            let values: Vec<f32> = left
                .iter()
                .map(|&j| j as f32 + f as f32 / 10.0)
                .chain([0.25])
                .chain(right.iter().map(|&j| j as f32 + f as f32 / 10.0))
                .chain([0.75])
                .collect();
            assert_eq!(row, &values, "frame {f}");
        }
        let extension: serde_json::Value =
            serde_json::from_reader(File::open(root.join("meta/batchplan.json")).unwrap()).unwrap();
        let robot_json = &extension["robot"];
        assert_eq!(robot_json["parts"].as_array().unwrap().len(), 2);
        assert_eq!(robot_json["parts"][1]["prefix"], "right_");
        assert_eq!(robot_json["hands"][0]["frame"], "left_ee_link");
        assert_eq!(robot_json["hands"][1]["gripper"], "right_gripper");
        std::fs::remove_dir_all(&root).unwrap();

        // The arrays keep the robot's joint order and every hand's gripper.
        let root = scratch("two-hands-npy");
        batchplan::npy::export(&root, &robot, &worlds, &[demo], &Default::default()).unwrap();
        let (shape, gripper) = read_npy(&root.join("gripper.npy"));
        assert_eq!(shape, [1, frames, 2]);
        assert_eq!(as_f32(&gripper)[..2], [0.25, 0.75]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn export_rejects_fractional_frame_rates() {
        let (robot, worlds, demos) = demos(0.03);
        let err = write(
            &scratch("lerobot-fps"),
            &robot,
            &worlds,
            &demos,
            &ExportOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("frames per second"), "{err}");
    }
}
