# Training data

batchplan turns planning into datasets for robot learning: reaches with recoveries from mistakes, pick-and-place episodes with a gripper, carried objects and task strings, and handovers between two arms. Datasets export as plain NumPy arrays or as LeRobot v3.0 datasets that `lerobot` loads and trains on directly.

Both kinds plan as [`plan`](planning.md) does, which straightens joints (`ShortcutOptions::straighten`), so a joint does not swing one way and back on its way, as a wrist can on a searched path. Across 512 pick-and-place tasks with obstacles across the carry, carries move their joints 13% less in the median and 18% less at the 90th percentile.

## Reaches and recoveries

`datagen::demonstrations(&device, &worlds, &goals, &DemoOptions)` turns goal poses into demonstrations, batched on the device:

1. **IK** for each goal with many seeds, keeping the collision-free solution with the most clearance.
2. **Nominal reaches** from the default pose plus joint noise (a collision-free sample) to the IK goal.
3. **Recoveries**: each solved path is perturbed partway along it (20–80% by default), the collision-free perturbed states are kept, and they are replanned to the same goal. These are the examples of getting back on track that raw planner data lacks.
4. **Timing** within the robot's limits at a random fraction of full speed (60–100% by default), sampled every `dt`.

`datagen::recovery_problems` exposes step 3 on its own for your own plans.

## Pick-and-place

`datagen::pick_and_place(&device, &worlds, &tasks, &PickPlaceOptions)` turns tasks (a box in a world and the pose to set it down at) into episodes. Each chains segments planned in batches, all from rest to rest:

1. **Approach** a pose 10 cm above the box from the default configuration, with the box an obstacle.
2. **Descend** to a top-down grasp across the box's narrower side, the box no longer an obstacle as the fingers close around it; **close** the gripper over half a second.
3. **Lift** back up and **transfer** to above the place pose holding the box (`Robot::attach`, touching only the links fixed to the wrist).
4. **Lower** it, **open** the gripper and **retreat**.

IK for the configuration above the box takes the solution nearest the start, and IK for the grasp and the place pose starts from the configuration above them and takes the solution nearest it, so the arm stays on one configuration and short moves stay short. The robot's IK frame must be the grasp point, its z axis pointing out of the gripper and its y axis along the fingers' closing direction (as the Panda's `ee_link`). Each episode records the gripper's opening, where the box is at every sample, and a task such as "Pick up the box and put it 20 cm to the left." `examples/datagen.rs --pick-place` completes 492 of 512 random tabletop tasks, boxes of 512 different sizes, in 0.5 s on an M4 Pro's GPU. The IK for every task runs in three batches whatever the sizes; only the transfers, which hold the box, plan per box size, on CPU threads.

```rust,no_run
use batchplan::datagen::{PickPlace, PickPlaceOptions, pick_and_place};
use batchplan::glam::{Quat, Vec3};
use batchplan::*;

fn main() -> Result<(), Error> {
    // IK targets the grasp point between the fingers. The fingers stay locked: episodes record the gripper apart.
    let fingers = [("panda_finger_joint1".to_string(), 0.04), ("panda_finger_joint2".to_string(), 0.04)];
    let options = RobotOptions {
        ee_links: vec!["ee_link".into()],
        lock_joints: fingers.into_iter().collect(),
        collision_model: Some(CollisionModel::load("assets/franka/panda_collision.json")?),
        ..Default::default()
    };
    let robot = Robot::load("assets/franka/franka_panda.urdf", &options)?;
    let device = Device::gpu(&robot).or_else(|_| Device::cpu(&robot))?;
    let table = Obstacle::Cuboid { center: Vec3::new(0.4, 0.0, -0.02), half_extents: Vec3::new(0.7, 0.8, 0.02), rotation: Quat::IDENTITY };
    let block = Obstacle::Cuboid { center: Vec3::new(0.45, 0.2, 0.03), half_extents: Vec3::new(0.02, 0.03, 0.03), rotation: Quat::IDENTITY };
    let worlds = vec![World { obstacles: vec![table, block] }];
    let place = Pose { position: Vec3::new(0.45, -0.2, 0.03), rotation: Quat::from_rotation_z(0.5) };
    let tasks = [PickPlace { world: 0, object: 1, place }];
    let demos = pick_and_place(&device, &worlds, &tasks, &PickPlaceOptions::default())?;
    for d in &demos {
        println!("{}: {} samples", d.task, d.trajectory.len());
    }
    Ok(())
}
```

On a robot with several hands, `pick_and_place` works with the first and holds the joints only the others move.

## Handovers

`datagen::handover(&device, &worlds, &tasks, &PickPlaceOptions)` passes a box between the two hands of a robot made of two arms (see [Robots](robots.md#several-arms-several-hands)). A `Handover` names the box, the hand that picks it up (`from`), the hand that sets it down (`to`), the pose where the hands meet, and the place pose. Each hand holds one end of the box's longer side from above, the end toward its own arm where they meet. Every segment is planned in one batch over the tasks:

1. **The giver picks the box up**: approach, descend, close, lift. The taker waits: its joints are held (`PlanProblem::hold`).
2. **Both arms move at once**: the giver carries the box to the meeting pose while the taker comes above its end. One plan moves every joint, so they arrive together.
3. **The taker takes it**: it descends onto its end and closes while the giver waits; the giver opens and lifts off while the taker waits.
4. **Both arms move at once again**: the giver returns to the default configuration while the taker carries the box above the place pose.
5. **The taker sets it down**: lower, open, retreat, while the giver waits.

`examples/handover/` runs this on two [reBot B601](https://github.com/Seeed-Projects/reBot-DevArm) arms facing a table, half the tasks passing left to right and half right to left; see [Benchmarks](benchmarks.md#bimanual-handovers) for its speed.

These are DexMimicGen's subtask kinds: sequential (one arm waits for the other), parallel (both move, each on its own errand) and coordinated (both must arrive together). IK solves both hands at once where they meet, and every plan holds the box in the hand that has it, with both wrists allowed to touch it. The episode records both grippers, where the box is at every sample, and a task such as "Pass the box from the left hand to the right hand and put it down."

```rust,no_run
use batchplan::datagen::{Handover, PickPlaceOptions, handover};
use batchplan::glam::{Quat, Vec3};
use batchplan::*;

/// Hands a 22 cm block lying in front of the left arm (obstacle 1 of world 0) to the right arm.
fn pass_it(device: &Device, worlds: &[World]) -> Result<Vec<datagen::Demonstration>, Error> {
    let across = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let meet = Pose { position: Vec3::new(0.5, 0.0, 0.3), rotation: across };
    let place = Pose { position: Vec3::new(0.45, -0.3, 0.025), rotation: across };
    let task = Handover { world: 0, object: 1, from: 0, to: 1, meet, place };
    handover(device, worlds, &[task], &PickPlaceOptions::default())
}
```

## Reset states

Reinforcement learning that starts episodes from many states along a task, as [OmniReset](https://arxiv.org/abs/2603.15789) and [AlphaNIST](https://alpha-nist.github.io/) do, needs a bank of valid states: reaching, the object in hand, the object nearly placed. `datagen::reset_states(&device, &worlds, &demos, &RecoveryOptions)` makes one from any demonstrations:

1. **Samples** of each demonstration, `per_trajectory` of them at random fractions of the way along it (within `phase`), with joint noise (`sigma`), at rest.
2. **The carried object** moves with a hand whose gripper is closed on it, in the grip that hand last holds it in alone. Samples where both hands hold it are skipped, since noise would pull the hands apart.
3. **A check** on the device keeps the collision-free states: an object no hand holds is an obstacle where it is, and a held one is attached to its hand, one batch per hand and grip. The check is conservative. Fingers about to close around an object collide with it, and a held object must clear the table, so states right at a grasp, a handover or a set-down are rare.

Each state is a one-sample `Demonstration` whose origin is a recovery from its sample, `parent` indexing `demos`, so `npy::export` writes them as it writes episodes (`positions.npy` is `[states, 1, dof]`). The episodes' own samples are valid states too. `cargo run --release --example handover -- 512 --resets <dir>` writes both, to `episodes/` and `resets/`. From 504 handovers, 8 states each at the default 0.15 rad of noise, 2,305 of 4,032 are collision-free: 691 with the block in the left hand, 746 in the right and 868 with it on the table. The check takes under 0.1 s on an M4 Pro, GPU or CPU.

## Demonstrations

Each `Demonstration` holds:
- its origin: nominal, or a recovery with the demonstration it branches from and where;
- its world, task string and goal pose (the IK frame's for a reach, the object's for pick-and-place);
- the timed trajectory, sampled every `dt`;
- each hand's gripper opening at every sample (`[samples, hands]`: 1 open, 0 closed; reaches hold it open);
- an obstacle it carries, if any, with its pose at every sample.

## Camera images

batchplan makes the motions and renders no images: policies that learn from cameras need realistic ones, which simulators render. Both exports carry what a simulator needs to replay an episode and render it: every world's obstacles (`worlds.json`, or `meta/batchplan.json`), and each episode's joint positions, gripper opening and carried object's pose at every frame.

`scripts/replay_mujoco.py` renders them with [MuJoCo](https://mujoco.org). It replays each LeRobot frame kinematically and writes a new dataset, the same frames plus a video per camera:
- the robot's parts (`meta/batchplan.json`), from their URDF or MJCF, at their bases;
- each obstacle where `observation.environment_state` has it, the carried one moving with the hand;
- the fingers following each gripper's opening, closing until they touch the carried object.

What batchplan cannot know comes from `meta/replay.json`, which the exporting program writes: which finger joints each gripper drives, and where the cameras are, in the world or on a hand's frame. `examples/handover/` writes one for two reBot arms, with an overhead camera and one on each wrist, as ALOHA has:

```json
{
  "grippers": {"left_gripper": [{"joint": "left_finger_left", "open": 0.035}, {"joint": "left_finger_right", "open": -0.035}], "...": []},
  "cameras": {
    "cam_high": {"pos": [-0.05, 0, 0.75], "target": [0.3, 0, 0], "fovy": 60},
    "cam_left_wrist": {"frame": "left_tcp", "pos": [0.08, 0, -0.04], "target": [0, 0, 0.08], "up": [-1, 0, 0], "fovy": 75}
  }
}
```

Visual meshes are simplified to 2,000 triangles each as they load: the reBot's CAD exports carry 860,000, which made rendering three times slower without changing what a 640×480 image shows. On an M4 Pro it renders three 640×480 cameras at about 15 ms a frame, 16 handover episodes (5,779 frames) in 85 s.

## Exporting

### Plain arrays: `npy::export`

`npy::export(root, &robot, &worlds, &demos, &ExportOptions)` writes NumPy arrays that `numpy.load` reads with nothing else:

| File | Contents |
|---|---|
| `positions.npy`, `velocities.npy` | `[episodes, steps, dof]` float32, padded past each episode's length with its final position and zero velocity |
| `length.npy` | `[episodes]` int32, valid steps per episode |
| `kind.npy`, `parent.npy` | `[episodes]`: 0 nominal or 1 recovery (uint8), and the episode a recovery branches from (int32, −1 otherwise) |
| `world.npy`, `worlds.json` | each episode's world index, and every world's obstacles (each distance grid written once) |
| `goal_pose.npy` | `[episodes, 7]` float32: position xyz, quaternion xyzw |
| `gripper.npy` | `[episodes, steps, hands]` float32, padded with its final value |
| `carried.npy`, `carried_pose.npy` | the obstacle each episode moves (−1 if none), and its pose at each step |
| `task.npy`, `meta.json` | each episode's task index, and `dt`, joint names, limits, counts and the task strings |

### LeRobot v3.0: `lerobot::export` (feature `lerobot`)

`lerobot::export(root, &robot, &worlds, &demos, &ExportOptions)` writes a dataset that `LeRobotDataset(repo_id, root=path)` loads directly: frames, episode metadata and tasks as parquet, plus `info.json` and normalization `stats.json`. Frames and episode metadata are written as they are converted, each starting a new file past `data_files_size_in_mb` (100 MB, LeRobot's default), so datasets can exceed memory. Statistics combine per-episode statistics as LeRobot does. Each demonstration is one episode:

| Feature | Contents |
|---|---|
| `observation.state` | joint positions and each hand's gripper opening, arm by arm as ALOHA lays out two arms: the joints no hand moves, then each hand's joints and its gripper (`left_gripper`, named after the arm's prefix; `gripper` on one arm) |
| `action` | the next frame's state (absolute targets); the last frame repeats its own |
| `observation.environment_state` | the goal pose, then each obstacle where it is at that frame as `[present, kind, center xyz, size xyz, quaternion xyzw]`, zero-padded to the largest world (`meta/batchplan.json` describes the layout) |
| `task` | the episode's task string, indexed in `meta/tasks.parquet` |
| `is_recovery`, `parent_episode_index`, `world_index` | extensions that LeRobot's policies ignore |

```rust,no_run
use batchplan::datagen::{DemoOptions, demonstrations};
use batchplan::glam::{Quat, Vec3};
use batchplan::lerobot::{ExportOptions, export};
use batchplan::*;

fn main() -> Result<(), Error> {
    let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &RobotOptions::default())?;
    let device = Device::gpu(&robot).or_else(|_| Device::cpu(&robot))?;
    let worlds = device.upload(&[World::default()])?;
    let down = Quat::from_rotation_x(std::f32::consts::PI);
    let goals = vec![IkProblem { world: 0, targets: vec![Some(Pose { position: Vec3::new(0.4, 0.2, 0.3), rotation: down })], seed: None }];
    let demos = demonstrations(&device, &worlds, &goals, &DemoOptions::default())?;
    let options = ExportOptions { data_files_size_in_mb: 100, ..Default::default() };
    export(std::path::Path::new("data/my_dataset"), &robot, worlds.as_slice(), &demos, &options)
}
```

### Checking an export with LeRobot

`scripts/validate_lerobot.py` checks an export with the real `lerobot` package (0.6.1): episodes, frames and tasks load as written from every file; episode boundaries are right and `action` is the next frame's state; normalization statistics match the frames; and action chunks are padded correctly at episode ends.

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python "lerobot[dataset,training]==0.6.1"
.venv/bin/python scripts/validate_lerobot.py data/my_dataset
.venv/bin/lerobot-train --dataset.repo_id=local/batchplan --dataset.root=data/my_dataset --policy.type=act \
  --policy.device=cpu --policy.push_to_hub=false --steps=50 --batch_size=16 --num_workers=0
```

It passes on reach exports (5,693 episodes across 6 episode-metadata files with 1 MB files), pick-and-place exports (18 tasks), and two-arm handovers before and after `replay_mujoco.py` adds their videos. LeRobot's ACT policy trains on each: a 50-step CPU run cut the loss from 44 to 5.8 on reaches and from 45 to 5.8 on pick-and-place, and a 40-step run on 16 rendered handovers, from the three cameras and both arms' states, from 58 to 6.5.

## `examples/datagen.rs`

```bash
cargo run --release --example datagen -- data/demo 512 20                     # reaches as .npy, 512 worlds at 20 fps
cargo run --release --features lerobot --example datagen -- --lerobot data/lerobot 512 20
cargo run --release --features lerobot --example datagen -- --pick-place --lerobot data/pick 512 20
```

`--file-mb N` starts new files sooner.
