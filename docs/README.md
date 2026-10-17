# Docs

The [README](../README.md) is the front door: what batchplan is, how to install it, and three snippets to start from. These pages go deeper. Each page owns its topic, and the others link to it rather than explain it again.

## Using batchplan

| | |
|---|---|
| [`planning.md`](planning.md) | IK, searching and smoothing paths with SIMD collision checks, keeping clear of obstacles, optional trajectory optimization, goals a turn away, time-optimal timing, replanning from a moving state, and the options worth knowing. |
| [`robots.md`](robots.md) | Loading URDF, MJCF and OpenUSD; joint types, mimic joints and closed loops; fitted collision spheres and collision-model files; holding objects; checking plans against meshes. |
| [`perception.md`](perception.md) | Obstacles and scenes; distance grids from meshes, point clouds and depth images; masking the robot out of a camera's view; fusing frames into an occupancy map; building grids on the GPU. |
| [`training-data.md`](training-data.md) | Reaches with recoveries, pick-and-place episodes, and exporting to NumPy or LeRobot v3.0. |
| [`devices.md`](devices.md) | GPU and CPU devices, which GPUs work, and running inside a robot's control process. |
| [`benchmarks.md`](benchmarks.md) | batchplan against VAMP, OMPL, cuRobo, RoboPlan and MoveIt 2, GPU against CPU, spheres against meshes, what made it fast, and how to reproduce each number. |

## Working on batchplan

| | |
|---|---|
| [`architecture.md`](architecture.md) | The design rules, the CPU/GPU twin, collision, the kernels written per robot, and trajectory optimization on the device. |
| [`verification.md`](verification.md) | Running the tests, and what each test pins down. |
| [`../CLAUDE.md`](../CLAUDE.md) | The working rules and the architecture in full detail, for people and coding agents alike. |
