# Training data

batchplan turns plans into datasets for robot learning: reaches with recoveries from mistakes, pick-and-place episodes, and handovers between two arms, with grippers, carried objects and task strings. They export as NumPy arrays or as LeRobot v3.0 datasets that `lerobot` loads and trains on directly.

Every episode plans as [`plan`](planning.md) does, straightening each joint (`ShortcutOptions::straighten`) so it does not swing one way and back, as a wrist can on a searched path. On 512 pick-and-place tasks with obstacles across the carry, that cuts the carries' joint travel by 13% at the median and 18% at the 90th percentile.

## Reaches and recoveries

`datagen::demonstrations(&device, &worlds, &goals, &DemoOptions)` turns goal poses into demonstrations, batched on the device:

1. **IK** for each goal with many seeds, keeping the collision-free solution with the most clearance.
2. **Nominal reaches** from the default pose plus joint noise (a collision-free sample) to the IK goal.
3. **Recoveries**: each solved path is perturbed partway along (20–80% by default), and the collision-free perturbed states are replanned to the same goal. Raw planner data has no such examples of getting back on track.
4. **Timing** within the robot's limits at a random fraction of full speed (60–100% by default), sampled every `dt`.

`datagen::recovery_problems` runs step 3 alone, on your own plans.

## Pick-and-place

`datagen::pick_and_place(&device, &worlds, &tasks, &PickPlaceOptions)` turns tasks (a box in a world, the pose to set it down at, and what it rests on) into episodes. Each is a [`pick`](planning.md#picking-up-and-setting-down) from the default configuration and a `place` from where the pick ends, each batched over the tasks, every motion from rest to rest:

1. **Approach**: a planned path to 10 cm above a top-down grasp across the box's narrower side, 2 cm below its top, then straight down to the grasp, where only the fingers may touch the box. **Close** the gripper over half a second.
2. **Lift** the box straight up, where it may touch what it rests on (`PickPlace::support`), and **carry** it on a planned path to above the place pose, holding it (`Robot::attach`).
3. **Lower** it straight down onto its supports there, **open** the gripper and **back off** straight up.

A task the grasp cannot pick up is tried again with the hand turned half way round, and one that cannot be set down at its pose with the box turned onto itself there (half a turn, or a quarter for a square footprint), each in a batch only as large as the failures.

IK at the grasp and where the hand lets go takes the solution nearest the configuration before, so the arm keeps to one IK branch and short moves stay short. The straight moves go as fast as the joints allow, as the planned paths do (`PickPlaceOptions::task` leaves the hand's speed uncapped), and every motion is slowed by its own random speed scale. Where an obstacle blocks part of a move in or away, the move may be shortened to 30% of its length (`TaskOptions::min_fraction`): a move in then starts nearer, the planned path going round the obstacle to it, and a move away ends sooner, the next planned path starting there. A move stopped by anything else, such as a joint limit, still rules the grasp or pose out. The robot's IK frame must be the grasp point, its z axis pointing out of the gripper and its y axis along the fingers' closing direction (as the Panda's `ee_link`). Each episode records the gripper's opening, where the box is at every sample, and a task such as "Pick up the box and put it 20 cm to the left."

`examples/datagen.rs --pick-place` completes 492 of 512 random tabletop tasks, with boxes of 512 different sizes, in 0.24–0.25 s on the Radeon 8060S, as many as and no slower than before it moved onto `pick` and `place` (0.25–0.26 s). IK runs in two batches however many sizes there are, plus small ones for tasks tried again; every step that holds a box runs per box size, the sizes in parallel. The tasks tried again matter: by one grasp and pose alone, 469 get through.

```rust,no_run
use batchplan::datagen::{PickPlace, PickPlaceOptions};
use batchplan::glam::{Quat, Vec3};
use batchplan::{
    CollisionModel, Device, Error, Obstacle, Pose, Robot, RobotOptions, World, datagen,
};

fn main() -> Result<(), Error> {
    // IK targets the grasp point between the fingers. The fingers stay locked: episodes record the gripper apart.
    let fingers = [
        ("panda_finger_joint1".to_string(), 0.04),
        ("panda_finger_joint2".to_string(), 0.04),
    ];
    let options = RobotOptions {
        ee_links: vec!["ee_link".into()],
        lock_joints: fingers.into_iter().collect(),
        collision_model: Some(CollisionModel::load("assets/franka/panda_collision.json")?),
        ..Default::default()
    };
    let robot = Robot::load("assets/franka/franka_panda.urdf", &options)?;
    let device = Device::gpu(&robot).or_else(|_| Device::cpu(&robot))?;
    let table = Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let block = Obstacle::cuboid(Vec3::new(0.45, 0.2, 0.03), Vec3::new(0.02, 0.03, 0.03));
    let worlds = vec![World::new(vec![table, block])];
    let place = Pose::new(Vec3::new(0.45, -0.2, 0.03), Quat::from_rotation_z(0.5));
    // The block rests on the table (obstacle 0) where it lies and where it goes.
    let tasks = [PickPlace {
        world: 0,
        object: 1,
        place,
        support: vec![0],
    }];
    let demos = datagen::pick_and_place(&device, &worlds, &tasks, &PickPlaceOptions::default())?;
    for d in &demos {
        println!("{}: {} samples", d.task, d.trajectory.len());
    }
    Ok(())
}
```

On a robot with several hands, `pick_and_place` uses the first and holds the joints that only the others move.

## Handovers

`datagen::handover(&device, &worlds, &tasks, &PickPlaceOptions)` passes a box between the two hands of a two-arm robot (see [Robots](robots.md#several-arms-several-hands)). A `Handover` names the box, the hand that picks it up (`from`), the hand that sets it down (`to`), the pose where the hands meet, the place pose, and what the box rests on. Each hand holds the box from above by one end of its longer side, the end nearer its own arm at the meeting pose. Every step is batched over the tasks:

1. **The giver picks the box up** (`pick`): approach, close, lift, while the taker waits.
2. **Both arms move at once**: the giver carries the box to the meeting pose while the taker comes above its end. One plan moves every joint, holding the box in the giver's hand, so they arrive together.
3. **The taker takes it**: it moves straight down onto its end and closes while the giver waits, then the giver opens and backs straight off while the taker waits. The box stands where the giver held it; the giver's hand and the taker's fingers may touch it.
4. **The taker sets it down** (`place`): it carries the box to above the place pose while the giver returns to the default configuration, both at once (`Place::idle`), then lowers it, opens and backs off while the giver waits.

`examples/handover/` runs this on two [reBot B601](https://github.com/Seeed-Projects/reBot-DevArm) arms facing a table, half the tasks passing left to right and half right to left. Three tables in four are divided down the middle, by a low wall, a screen with an opening or a row of posts, which the arms plan over or around. [Benchmarks](benchmarks.md#bimanual-handovers) gives its speed.

The steps cover DexMimicGen's subtask kinds: sequential (one arm waits for the other), parallel (both move, each on its own errand) and coordinated (both must arrive together). IK solves both hands at once at the meeting pose. The box is checked against the scene and the other arm the whole way: lifted, carried, handed over and set down. The episode records both grippers, where the box is at every sample, and a task such as "Pass the box from the left hand to the right hand and put it down."

```rust,no_run
use batchplan::datagen::{Handover, PickPlaceOptions};
use batchplan::glam::{Quat, Vec3};
use batchplan::{Device, Error, Pose, World, datagen};

/// Hands a 22 cm block (obstacle 1 of world 0), lying on the table (obstacle 0) in front of the
/// left arm, to the right arm.
fn pass_it(device: &Device, worlds: &[World]) -> Result<Vec<datagen::Demonstration>, Error> {
    let across = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let meet = Pose::new(Vec3::new(0.5, 0.0, 0.3), across);
    let place = Pose::new(Vec3::new(0.45, -0.3, 0.025), across);
    let task = Handover {
        world: 0,
        object: 1,
        from: 0,
        to: 1,
        meet,
        place,
        support: vec![0],
    };
    datagen::handover(device, worlds, &[task], &PickPlaceOptions::default())
}
```

## Reset states

Reinforcement learning that starts episodes from many states along a task, as [OmniReset](https://arxiv.org/abs/2603.15789) and [AlphaNIST](https://alpha-nist.github.io/) do, needs a bank of valid states: reaching, the object in hand, the object nearly placed. `datagen::reset_states(&device, &worlds, &demos, &RecoveryOptions)` makes one from any demonstrations:

1. **Samples**: `per_trajectory` per demonstration, at random fractions of the way along it (within `phase`), with joint noise (`sigma`), at rest.
2. **The carried object** moves with the hand whose gripper is closed on it, in the grip that hand had when it last held the object alone. Samples where both hands hold it are skipped, since noise would pull the hands apart.
3. **A check** on the device keeps the collision-free states. An object no hand holds is an obstacle where it is; a held one is attached to its hand. The check is conservative: fingers about to close around an object collide with it, and a held object must clear the table, so states right at a grasp, a handover or a set-down are rare.

Each state is a one-sample `Demonstration` whose origin is a recovery from its sample, with `parent` indexing `demos`, so `npy::export` writes states as it writes episodes (`positions.npy` is `[states, 1, dof]`). The episodes' own samples are valid states too; `cargo run --release --example handover -- 512 --resets <dir>` writes both, to `episodes/` and `resets/`. From 510 handovers, 8 states each at the default 0.15 rad of noise, 2,015 of 4,080 are collision-free on the Ryzen's CPU device: 595 with the block in the left hand, 603 in the right and 817 on the table (2,020 on the Radeon, whose clearances differ from the CPU's in the last bits at the margin). The check takes 0.04 s on the Ryzen's CPU device and 0.07 s on the Radeon.

## Demonstrations

Each `Demonstration` holds:
- its origin: nominal, or a recovery with the demonstration it branches from and where;
- its world, task string and goal pose (the IK frame's for a reach, the object's for pick-and-place);
- the timed trajectory, sampled every `dt`;
- each hand's gripper opening at every sample (`[samples, hands]`: 1 open, 0 closed; reaches hold it open);
- an obstacle it carries, if any, with its pose at every sample.

## Camera images

batchplan renders no images: policies that learn from cameras need realistic ones, which simulators render. Both exports carry what a simulator needs to replay and render an episode: every world's obstacles (`worlds.json`, or `meta/batchplan.json`), and each episode's joint positions, gripper opening and carried object's pose at every frame.

`scripts/replay_mujoco.py` renders them with [MuJoCo](https://mujoco.org), writing a new dataset: the same frames plus a video per camera. It replays each LeRobot frame kinematically, with:
- the robot's parts (`meta/batchplan.json`), from their URDF or MJCF, at their bases;
- each obstacle where `observation.environment_state` has it, the carried one moving with the hand;
- the fingers following each gripper's opening, closing until they touch the carried object.

What batchplan cannot know comes from `meta/replay.json`, which the exporting program writes: which finger joints each gripper drives, and where the cameras are, in the world or on a hand's frame. `examples/handover/` writes one for two reBot arms with an overhead camera and one on each wrist, as ALOHA has:

```json
{
  "grippers": {"left_gripper": [{"joint": "left_finger_left", "open": 0.035}, {"joint": "left_finger_right", "open": -0.035}], "...": []},
  "cameras": {
    "cam_high": {"pos": [-0.05, 0, 0.75], "target": [0.3, 0, 0], "fovy": 60},
    "cam_left_wrist": {"frame": "left_tcp", "pos": [0.08, 0, -0.04], "target": [0, 0, 0.08], "up": [-1, 0, 0], "fovy": 75}
  }
}
```

Visual meshes are simplified to 2,000 triangles each as they load. The reBot's CAD exports carry 860,000, which made rendering three times slower with no visible difference at 640×480. On an M4 Pro, three 640×480 cameras render at about 14 ms a frame: 16 handover episodes (6,075 frames) in 85 s.

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

`lerobot::export(root, &robot, &worlds, &demos, &ExportOptions)` writes a dataset that `LeRobotDataset(repo_id, root=path)` loads directly: frames, episode metadata and tasks as parquet, plus `info.json` and normalization `stats.json`. Frames and episode metadata are written as they are converted, starting a new file past `data_files_size_in_mb` (100 MB, LeRobot's default), so datasets can exceed memory. Statistics combine per-episode statistics as LeRobot does. Each demonstration is one episode:

| Feature | Contents |
|---|---|
| `observation.state` | joint positions and each hand's gripper opening, arm by arm as ALOHA lays out two arms: the joints no hand moves, then each hand's joints and its gripper (`left_gripper`, named after the arm's prefix; `gripper` on one arm) |
| `action` | the next frame's state (absolute targets); the last frame repeats its own |
| `observation.environment_state` | the goal pose, then each obstacle where it is at that frame as `[present, kind, center xyz, size xyz, quaternion xyzw]`, zero-padded to the largest world (`meta/batchplan.json` describes the layout) |
| `task` | the episode's task string, indexed in `meta/tasks.parquet` |
| `is_recovery`, `parent_episode_index`, `world_index` | extensions that LeRobot's policies ignore |

```rust,no_run
use batchplan::datagen::DemoOptions;
use batchplan::glam::{Quat, Vec3};
use batchplan::lerobot::ExportOptions;
use batchplan::{Device, Error, IkProblem, Pose, Robot, RobotOptions, World, datagen, lerobot};

fn main() -> Result<(), Error> {
    let robot = Robot::load(
        "assets/ur5e/ur_description/urdf/ur5e.urdf",
        &RobotOptions::default(),
    )?;
    let device = Device::gpu(&robot).or_else(|_| Device::cpu(&robot))?;
    let worlds = device.upload(&[World::default()])?;
    let down = Quat::from_rotation_x(std::f32::consts::PI);
    let goals = vec![IkProblem::new(0, Pose::new(Vec3::new(0.4, 0.2, 0.3), down))];
    let demos = datagen::demonstrations(&device, &worlds, &goals, &DemoOptions::default())?;
    let options = ExportOptions {
        data_files_size_in_mb: 100,
        ..Default::default()
    };
    lerobot::export("data/my_dataset", &robot, &worlds, &demos, &options)
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

It passes on reach exports (5,704 episodes from 2,048 worlds, split over 6 episode-metadata files at a 1 MB limit), pick-and-place exports (492 episodes, 18 tasks), and 16 two-arm handovers before and after `replay_mujoco.py` adds their videos. LeRobot's ACT policy trains on each. A 50-step CPU run (batch 16, the loss logged every 10 steps) cut the loss from 4.5 at step 10 to 0.52 at step 50 on reaches, and from 4.2 to 0.49 on pick-and-place; a 40-step run on the rendered handovers, from the three cameras and both arms' states, from 4.0 to 0.61.

## `examples/datagen.rs`

```bash
cargo run --release --example datagen -- data/demo 512 20                     # reaches as .npy, 512 worlds at 20 fps
cargo run --release --features lerobot --example datagen -- --lerobot data/lerobot 512 20
cargo run --release --features lerobot --example datagen -- --pick-place --lerobot data/pick 512 20
```

`--file-mb N` starts new files past N MB instead of 100.
