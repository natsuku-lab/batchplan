# Docs

The [README](../README.md) is the front door: what batchplan is, how to install it, and three snippets to start from. These pages go deeper. Each page owns its topic, and the others link to it rather than explain it again.

## Using batchplan

| | |
|---|---|
| [`api.md`](api.md) | Every type and function a program uses, by task, with the guide for each. |
| [`planning.md`](planning.md) | IK, searching and smoothing paths with SIMD collision checks, keeping clear of obstacles, optional trajectory optimization, goals a turn away, moving the hand along a path, picking up and setting down, time-optimal timing, replanning from a moving state, and the options worth knowing. |
| [`robots.md`](robots.md) | Loading URDF, MJCF and OpenUSD; joint types, mimic joints and closed loops; fitted collision spheres and collision-model files; holding objects; checking plans against meshes. |
| [`perception.md`](perception.md) | Obstacles and scenes; distance grids from meshes, point clouds and depth images; masking the robot out of a camera's view; fusing frames into an occupancy map; building grids on the GPU. |
| [`training-data.md`](training-data.md) | Reaches with recoveries, pick-and-place episodes, handovers between two arms, reset states, and exporting to NumPy or LeRobot v3.0. |
| [`devices.md`](devices.md) | GPU and CPU devices, which GPUs work, and running inside a robot's control process. |
| [`benchmarks.md`](benchmarks.md) | batchplan against VAMP, OMPL, cuRobo, RoboPlan and MoveIt 2, GPU against CPU, spheres against meshes, what made it fast, and how to reproduce each number. |

## Working on batchplan

| | |
|---|---|
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Building and testing, the style guide the code follows, and what a change needs. |
| [`architecture.md`](architecture.md) | How the code fits together: a map of the modules, the principles they follow, planning a path, searches, collision, the GPU, tool paths, joins and tasks, timing and training data. |
| [`verification.md`](verification.md) | Running the tests, and what each test pins down. |
| [`../CLAUDE.md`](../CLAUDE.md) | The design rules, invariants and working rules, for people and coding agents alike. |
