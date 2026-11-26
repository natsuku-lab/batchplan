# Docs

The [README](../README.md) is the front door: what batchplan is, how to install it, and three snippets to start from. These pages go deeper. Each page owns its topic, and the others link to it rather than explain it again.

## Using batchplan

| | |
|---|---|
| [`../AGENTS.md`](../AGENTS.md) | For coding agents using batchplan: conventions, the lifecycle, which call fits a task, qualifying a robot, reading results, diagnosing a failed request in order, and footguns. |
| [`api.md`](api.md) | Every type and function a program uses, by task, with the guide for each. |
| [`planning.md`](planning.md) | IK, searching and smoothing paths with SIMD collision checks, keeping clear of obstacles, optional trajectory optimization, goals a turn away, constraints on the hand along the path, moving the hand along a path, picking up and setting down, handing over, joining motions, time-optimal timing, replanning from a moving state, live motion and servoing, and the options worth knowing. |
| [`grasps.md`](grasps.md) | The gripper and its finger openings; grasps of any shape from what cameras see: walls seen from one side, fingers that come in narrower, room for a planner, closing on a point, and the order grasps come in. |
| [`placing.md`](placing.md) | Releases; where a held object rests on what cameras see, and why it has no rest; where to let it go to fall into a container; the place nearest a spot. |
| [`turning.md`](turning.md) | Turning a knob, a cap, a lever or a door about an axis, and drawing a drawer by its handle along a line. |
| [`robots.md`](robots.md) | Loading URDF, MJCF and OpenUSD; joint types, mimic joints, closed loops and a bound on what a hand can reach; several arms and hands; fitted collision spheres, a limit on their overhang, collision-model files and qualifying a robot; holding objects; checking plans against meshes. |
| [`perception.md`](perception.md) | Obstacles, allowed contacts and scenes; distance grids from meshes, point clouds and depth images; leaving out readings between two surfaces; masking the robot, the shapes a caller holds and the band beside them out of a camera's view; fusing frames into an occupancy map, and taking parts and things out of it; meeting a perception library; building grids on the GPU. |
| [`../miru/README.md`](../miru/README.md) | `batchplan-miru`, an optional crate built on the library: what depth cameras show, as objects. |
| [`training-data.md`](training-data.md) | Reaches with recoveries, pick-and-place episodes, handovers between two arms, reset states, and exporting to NumPy or LeRobot v3.0. |
| [`devices.md`](devices.md) | GPU and CPU devices, which GPUs work, and running inside a robot's control process. |
| [`python.md`](python.md) | The `batchplan` Python package: installing it, how its calls map to Rust's (frozen values, read-only arrays, quaternions scalar last, exceptions per error kind), threads, and what it binds so far. |
| [`benchmarks.md`](benchmarks.md) | Planner comparisons, CPU/GPU scaling, tool paths and constraints, tasks and turns, grasps and rests of what cameras see, perception, held objects' and robots' spheres, joins and timing, servo tracking, handovers, and the commands that reproduce them. |

## Working on batchplan

| | |
|---|---|
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Building and testing, the style guide the code follows, and what a change needs. |
| [`architecture.md`](architecture.md) | How the code fits together: a map of the modules, the principles they follow, planning a path, searches, collision, the GPU, tool paths, joins and tasks, grasps of what cameras see, timing, live motion, perception and training data, the robot model and its collision spheres. |
| [`verification.md`](verification.md) | Running the tests, and what each test pins down. |
| [`../CLAUDE.md`](../CLAUDE.md) | The design rules, invariants and working rules, for people and coding agents alike. |
