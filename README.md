<h1 align="center">batchplan</h1>

<p align="center">
  <em>Thousands of robot motions at once, a fraction of a millisecond each.</em>
</p>

<p align="center">
  <a href="#quick-start"><b>Quick start</b></a> ·
  <a href="docs/README.md">Docs</a> ·
  <a href="docs/benchmarks.md">Benchmarks</a> ·
  <a href="examples/">Examples</a> ·
  <a href="docs/architecture.md">How it works</a>
</p>

<p align="center">
  <a href="https://github.com/natsuku-lab/batchplan/actions/workflows/ci.yml"><img src="https://github.com/natsuku-lab/batchplan/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

---

**batchplan is a Rust library for collision-aware motion planning, in batches.** One call plans thousands of motions, each in its own scene: IK, a collision-free path, and a trajectory timed within the robot's limits. A typical problem plans in 0.17 ms on one CPU core, faster than [VAMP](#how-it-compares), with paths that keep clear of the real geometry more often than any planner we compared. IK, trajectory optimization and distance grids run on any GPU that [wgpu](https://wgpu.rs) supports (Vulkan, Metal or DX12), with no CUDA, no ROCm and no ROS, or on the CPU with the same results.

It is built for robot learning at scale. It turns planning into training data: reaches with recoveries from mistakes, pick-and-place episodes, and handovers between two arms, with grippers and task strings. These are written as [LeRobot](https://github.com/huggingface/lerobot) datasets that train as they are, and a MuJoCo script renders camera videos into them. Two reBot arms hand blocks to each other in 512 scenes in 1.6 s on a laptop's GPU, or 1.9 s on its CPU.

- **Plan:** batched IK; RRT-Connect, shortcutting and smoothing on SIMD collision checks, a fraction of a millisecond per problem; smooth B-spline paths, optionally optimized with L-BFGS; time-optimal timing within velocity, acceleration and jerk limits; replanning from a moving state; and controlled stops along the path.
- **Robots:** load from URDF, MJCF or OpenUSD, including mimic joints, closed loops, continuous joints and ball or floating joints, or assemble several, such as two arms. Collision spheres are fitted automatically. IK takes a target for each hand at once, and plans can hold one arm still while another moves.
- **Worlds:** boxes, spheres, cylinders, capsules and signed distance grids. Grids are built from meshes, point clouds and depth images, or from frames fused over time, on the GPU.
- **Data:** demonstrations with recoveries, pick-and-place episodes and handovers between two arms, and states perturbed from them to start reinforcement learning from, exported as LeRobot v3.0 datasets (arm by arm, as ALOHA's) or NumPy arrays. `scripts/replay_mujoco.py` replays them in MuJoCo and adds camera videos.
- **Embeddable:** CPU-only builds, threads you own, time budgets, allocation-free sampling and typed errors, for use inside a robot's control process.

## Install

```toml
[dependencies]
batchplan = { git = "https://github.com/natsuku-lab/batchplan" }
```

| Cargo feature | Default | Adds |
|---|---|---|
| `gpu` | on | GPU devices through wgpu. Without it, batchplan is CPU-only and has no GPU dependencies. |
| `lerobot` | off | LeRobot v3.0 export (Arrow, Parquet). |
| `usd` | off | Robots and scenes from OpenUSD. |

batchplan builds on stable Rust (edition 2024), with no build flags to set: its collision checks use the widest SIMD instructions the CPU has, chosen when the program runs (SSE2 up to AVX-512 on x86-64, NEON on ARM).

On Linux, a faster allocator helps: glibc's slows down when many threads allocate at once, and [mimalloc](https://crates.io/crates/mimalloc) as your program's global allocator plans about 10% more problems per second on 32 threads.

## Quick start

The robot paths below point at the test robots in this repository's `assets/`. Use your own robot's file in their place.

### Plan a batch of motions

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::*;

fn main() -> Result<(), Error> {
    // Any URDF, MJCF or USD robot. Collision spheres are fitted when it loads.
    let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &RobotOptions::default())?;
    let device = Device::gpu(&robot).or_else(|_| Device::cpu(&robot))?;

    // Scenes are uploaded once. Problems name their scene by index.
    let table = Obstacle::Cuboid { center: Vec3::new(0.0, 0.0, -0.02), half_extents: Vec3::new(1.0, 1.0, 0.02), rotation: Quat::IDENTITY };
    let worlds = device.upload(&[World { obstacles: vec![table] }])?;

    // IK for a batch of gripper poses...
    let down = Quat::from_rotation_x(std::f32::consts::PI);
    let goals: Vec<IkProblem> = [Vec3::new(0.4, 0.2, 0.3), Vec3::new(0.3, -0.3, 0.4)]
        .into_iter()
        .map(|position| IkProblem { world: 0, targets: vec![Some(Pose { position, rotation: down })], seed: None })
        .collect();
    let ik = solve_ik(&device, &worlds, &goals, &IkOptions::default())?;

    // ...then a collision-free motion to every solution, in one call.
    let problems: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem { world: s.problem.world, start: robot.default_q().to_vec(), goal: s.solution.to_vec(), start_motion: None, hold: vec![] })
        .collect();
    let plans = plan(&device, &worlds, &problems, &PlanOptions::default())?;

    // Time each path within the robot's velocity, acceleration and jerk limits.
    for s in plans.solved() {
        let trajectory = Trajectory::new(&robot, s.solution, 1.0)?;
        let samples = trajectory.sample(100.0)?; // 100 Hz, for a controller
        println!("motion {}: {:.2} s, {} samples", s.index, trajectory.duration(), samples.len());
    }
    Ok(())
}
```

### Make a training dataset

```rust,no_run
use batchplan::datagen::{DemoOptions, demonstrations};
use batchplan::glam::{Quat, Vec3};
use batchplan::lerobot::{ExportOptions, export};
use batchplan::*;

fn main() -> Result<(), Error> {
    let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &RobotOptions::default())?;
    let device = Device::gpu(&robot).or_else(|_| Device::cpu(&robot))?;
    let worlds = device.upload(&[World::default()])?;

    // Reaches to 64 goals, plus recoveries from states perturbed along them.
    let down = Quat::from_rotation_x(std::f32::consts::PI);
    let goals: Vec<IkProblem> = (0..64)
        .map(|i| Vec3::new(0.3 + 0.03 * (i % 8) as f32, -0.25 + 0.06 * (i / 8) as f32, 0.3))
        .map(|position| IkProblem { world: 0, targets: vec![Some(Pose { position, rotation: down })], seed: None })
        .collect();
    let demos = demonstrations(&device, &worlds, &goals, &DemoOptions::default())?;

    // A LeRobot v3.0 dataset (cargo feature `lerobot`), ready for `lerobot-train`.
    export("data/reaches".as_ref(), &robot, worlds.as_slice(), &demos, &ExportOptions::default())
}
```

### Plan around what a depth camera sees

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::*;
use std::sync::Arc;

/// Turns one depth frame into a world to plan in, leaving the arm itself out.
fn world_from_camera(device: &Device, robot: &Robot, depth: &[f32], q: &[f32]) -> Result<Worlds, Error> {
    let image = DepthImage {
        depth, // metres, row by row
        width: 640,
        intrinsics: Intrinsics { fx: 525.0, fy: 525.0, cx: 319.5, cy: 239.5 },
        // 1.5 m above the table, looking down (OpenCV axes: z forward, y down).
        camera: Pose { position: Vec3::new(0.5, 0.0, 1.5), rotation: Quat::from_rotation_x(std::f32::consts::PI) },
    };
    let arm = image.robot_mask(robot, q, 0.04)?; // the arm is not an obstacle to itself
    // What objects hide is solid; what only the arm hides is free, or it would wall the arm in.
    let behind = Occlusion::OccupiedExceptRobotShadow;
    let grid = SdfGrid::from_depth(device, &image, Some(&arm), behind, &SdfOptions::default())?;
    device.upload(&[World { obstacles: vec![Obstacle::Sdf { grid: Arc::new(grid), center: Vec3::ZERO, rotation: Quat::IDENTITY }] }])
}
```

To fuse frames over time, use an `OccupancyMap` ([Worlds and perception](docs/perception.md)).

## How it compares

Each library planned the same 2,600 Panda problems from MotionBenchMaker and MπNets, and one checker judged every path, batchplan's included. *Solved* means the path reaches the goal within the joint limits without touching the model it was planned on; *clear on the meshes* checks it against the Panda's meshes, the real robot. [Setup and per-set results](docs/benchmarks.md#against-other-planners) · [Run them yourself](docs/benchmarks.md#reproducing)

- **The fastest of these planners:** 0.17 ms per problem and 20,287 problems/s on an M4 Pro CPU, ahead of VAMP on the same spheres.
- **Clear of the real robot most often:** 85–93% of its paths clear the Panda's meshes; no other planner's reach 82%.
- **Solves the most** in every comparison, with shorter paths than every other sampling-based planner, and, with two optimization seeds, than cuRobo.

### Against VAMP and OMPL, on VAMP's spheres

Apple M4 Pro CPU, 14 cores.

| Planner | Solved | Clear on the meshes | Path length¹ | Time per problem (median) | All 2,600, one core | Batched |
|---|---|---|---|---|---|---|
| **batchplan** | **90.4%** | **85.2%** | **3.41 rad** | **0.17 ms** | **0.88 s** | **20,287 /s** |
| [VAMP](https://github.com/KavrakiLab/vamp) 0.6.4 | 86.1% | 71.6% | 3.56 rad | 0.26 ms | 2.43 s | 2,704 /s |
| VAMP, stopped at 10,000 samples² | 86.1% | 71.5% | 3.56 rad | 0.26 ms | 1.05 s | 15,396 /s |
| VAMP, 2 rad steps³ | 86.4% | 75.3% | 3.56 rad | 0.27 ms | 2.71 s | 2,705 /s |
| [OMPL](https://ompl.kavrakilab.org) 2.0, RRT-Connect⁴ | 63.9% | 63.5% | 4.02 rad | 2.75 ms | 631 s | 30 /s |

¹ Median over the 1,557 problems all five solve. ² As batchplan's searches do; VAMP's default is 100,000. ³ VAMP's library default; the other rows step 1 rad, its Panda setting. ⁴ With VAMP's collision checker and a 5 s limit.

### Against cuRobo, on cuRobo's spheres

cuRobo on an NVIDIA T4 GPU with 8 Xeon threads; batchplan on the M4 Pro². The 1,300 problems whose scenes are only boxes, which both libraries load exactly.

| Planner | Solved | Clear on the meshes | Path length¹ | Time per problem (median) | Batched |
|---|---|---|---|---|---|
| **batchplan** | **98.9%** | 92.8% | 3.243 rad | **0.22 ms** | **13,384 /s** |
| batchplan, optimizing 2 seeds³ | 98.8% | **92.9%** | **3.137 rad** | 16.8 ms | 138 /s, 769 /s on its GPU |
| [cuRobo](https://github.com/NVlabs/curobo) 0.8 | 85.2% | 81.6% | 3.144 rad | 197 ms | 3.0 /s |
| cuRobo, 32 problems per call | 85.0% | 81.9% | 3.154 rad | 660 ms⁴ | 1.4 /s |

¹ Median over the 1,092 problems all four solve. ² The T4 machine has not been available since batchplan's previous version, which planned there in 0.64 ms and 2,152 problems/s. ³ `PlanOptions::seeds`, off by default: the searched path and the straight line, optimized. ⁴ Each call's time over its problems.

### Against MoveIt 2 and RoboPlan, which plan on the meshes

AMD Ryzen AI Max+ 395 CPU, 32 threads. batchplan plans on the spheres it ships with; every path is judged on the meshes.

| Planner | Solved, on the meshes | Path length¹ | Time per problem (median) | Batched |
|---|---|---|---|---|
| **batchplan** | **93.1%** | **3.23 rad** | **0.12 ms** | **22,685 /s** |
| [RoboPlan](https://github.com/open-planning/roboplan) 0.7 | 73.8% | 3.78 rad | 10.3 ms | 174 /s |
| [MoveIt 2](https://moveit.ai) Jazzy (OMPL, FCL) | 63.1% | 3.80 rad | 59.4 ms | 82 /s |

¹ Median over the 1,334 problems all three solve.

### Batches and the GPU

Planning runs on CPU threads on either device; IK and trajectory optimization also run on the GPU. Each call here solves IK and plans a batch of tabletop grasps, end to end, in solved problems per second:

| Problems per call | M4 Pro GPU | M4 Pro CPU (14 cores) | Radeon 8060S GPU | Ryzen AI Max+ 395 CPU (16 cores) |
|---|---|---|---|---|
| 1 | 129 | **1,658** | 125 | **1,345** |
| 8 | 949 | **3,850** | 930 | **4,879** |
| 64 | 2,771 | **4,236** | 3,614 | **5,864** |
| 512 | **8,883** | 4,705 | **15,119** | 5,784 |
| 4,096 | **10,815** | 4,713 | **18,886** | 6,041 |

A GPU call costs about 10 ms, so one problem at a time runs 11–13 times faster on the CPU; from a few hundred problems per call the GPU overtakes it, and at 4,096 it solves 2.3–3.1 times as many.

### When another library fits better

- **VAMP**, from Python, or for its asymptotically optimal planners (AORRTC, FCIT*) and roadmaps (PRM) on the same fast checks.
- **cuRobo**, for model-predictive control and NVIDIA's perception stack (nvblox), if you have an NVIDIA GPU.
- **MoveIt 2**, for ROS 2: perception, controllers, grasping and the rest of a robot's stack around the planner.
- **OMPL**, for its range of planners, such as asymptotically optimal and kinodynamic ones.

## Where to find things

### You want to use it

| | |
|---|---|
| [Planning](docs/planning.md) | IK, searching and smoothing paths, keeping clear of obstacles, optional trajectory optimization, goals a turn away, timing, replanning mid-motion, and the options worth knowing. |
| [Robots](docs/robots.md) | URDF, MJCF and OpenUSD; joint types, mimic joints and closed loops; collision spheres; holding objects; checking against meshes. |
| [Worlds and perception](docs/perception.md) | Obstacles, scenes, distance grids, depth images, masking the robot, and fused occupancy maps. |
| [Training data](docs/training-data.md) | Reaches with recoveries, pick-and-place, and exporting to NumPy or LeRobot. |
| [Devices](docs/devices.md) | GPU or CPU, which GPUs work, and running inside a robot's control process. |
| [Benchmarks](docs/benchmarks.md) | Against VAMP, OMPL, RoboPlan and MoveIt 2, GPU against CPU, spheres against meshes, and how to reproduce them. |

### You are working on it

| | |
|---|---|
| [How it works](docs/architecture.md) | The design rules, the CPU/GPU twin, collision, per-robot kernels, and how the optimizer runs on the device. |
| [Verification](docs/verification.md) | Running the tests, and what each one pins down. |
| [CLAUDE.md](CLAUDE.md) | Working rules and the architecture in full detail, for people and coding agents. |

## Examples

| Command | What it does |
|---|---|
| `scripts/fetch_rebot.sh && cargo run --release --example handover` | Two reBot B601 arms hand blocks to each other in 512 scenes: the bimanual benchmark. `--features lerobot` and `-- 16 --lerobot data/handover` write a LeRobot dataset, and `scripts/replay_mujoco.py data/handover data/handover_video` adds three camera videos. |
| `cargo run --release --example bench -- 512` | IK and planning for 512 cluttered tabletop scenes, GPU against CPU. |
| `cargo run --release --example datagen -- data/demo 512 20` | Reaches and recoveries in 512 scenes, as `.npy` arrays at 20 fps. |
| `cargo run --release --features lerobot --example datagen -- --pick-place --lerobot data/pick 512 20` | Pick-and-place episodes as a LeRobot dataset. |
| `cargo run --release --example depth` | Builds a distance grid from a rendered depth image and plans around it. |
| `cargo run --release --example control_loop -- 10` | A planner thread feeding a 50 Hz control loop. |
| `scripts/fetch_benchmark.sh && cargo run --release --example benchmark` | The 2,600 MotionBenchMaker and MπNets problems, on every device. |

[`demos/`](demos/) shows four of them live in [Rerun](https://rerun.io): two reBot arms handing blocks over, Panda pick-and-place around obstacles, planning around what depth cameras see, and replanning mid-motion.

## Limitations

- **Spheres.** Planning sees the robot as spheres. `MeshModel` checks finished plans against its meshes on the CPU, as a filter.
- **Joints.** Each closed loop must have exactly one actuated joint. Ball and floating joints are expanded into Euler angles, which lose a direction of motion where the middle angle reaches ±90°.
- **Timing.** Trajectories end at rest. Time-optimal timing respects jerk limits but does not minimize jerk.
- **Small batches on a GPU.** A GPU call costs about as much for one problem as for a small batch: IK for one target takes about 10 ms on an M4 Pro's GPU and 1 ms on its CPU. Planning itself runs on CPU threads on either device.
- **Pick-and-place and handovers.** Episodes grasp upright boxes from above and have no recoveries. Camera videos come from replaying them in MuJoCo, kinematically: no physics checks the grasps.
- **Maturity.** batchplan is at version 0.1 and not on crates.io. Its interface may change. It is verified on AMD (Vulkan) and Apple (Metal) GPUs. On NVIDIA, batchplan was timed against cuRobo on a T4 at the previous version, and the tests ran at an early one.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

The test robots under `assets/` keep their own licences, listed with their sources and changes in [assets/README.md](assets/README.md):

- The Franka Panda is from franka_ros and NVlabs/curobo (Apache-2.0). Its collision spheres in `assets/franka/panda_collision.json` are converted from cuRobo's `franka.yml` (Apache-2.0, © NVIDIA).
- The UR5e is from Universal Robots' ROS 2 description (BSD-3-Clause).
- The Robotiq 2F-85 is from PickNik's ros2_robotiq_gripper (BSD-3-Clause).
- The SO-101 is from TheRobotStudio's SO-ARM100 (Apache-2.0).

The reBot B601 arms that `examples/handover/` uses are not in the repository: `scripts/fetch_rebot.sh` downloads them from Seeed Studio's [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) at a pinned commit (CERN-OHL-W-2.0).
