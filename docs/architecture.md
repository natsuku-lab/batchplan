# Architecture

How batchplan's code is organised, and why. [`CLAUDE.md`](../CLAUDE.md) holds the rules for changing it; each module's doc comment holds its details, and its constants say why they have the values they do.

A robot and its worlds go onto a device once. Every algorithm then takes the device, the worlds and a batch of problems, and returns each problem with its answer.

## The code

```
src/
├── lib.rs           # the public API: re-exports, and the guides' code compiled as tests
│
│   loading a robot
├── robot.rs         # Robot: the kinematic tree, limits, hands, held objects
├── description.rs   # the one description every format loads into
├── urdf.rs          # ┐
├── mjcf.rs          # ├ loaders: a file into the description, nothing more
├── usd.rs           # ┘
├── loops.rs         # closed loops become mimic joints of their one motor
├── spheres.rs       # collision spheres fitted to the meshes (cuRobo's voxel method)
├── meshes.rs        # exact clearance to the meshes, to check finished plans
│
│   worlds
├── world.rs         # obstacles, allowed contacts, scenes
├── sdf.rs           # distance grids from meshes, points and depth images; occupancy maps
│
│   running a batch
├── device.rs        # Device: checks every batch, then hands it to a backend
├── cpu.rs           # the CPU backend, the scalar twin of the GPU's math
├── lanes.rs         # SIMD collision checks, eight configurations at a time
├── gpu.rs           # the GPU backend (wgpu), kernels written per robot
├── *.wgsl           # the GPU's math, and grid building
├── edges.rs         # how searches ask for edge checks, and the threads that answer
│
│   motion
├── ik.rs            # solve_ik: damped least squares from many seeds
├── rrt/             # RRT-Connect, and idle threads that help long searches
├── shortcut.rs      # simplifying a found path
├── spline.rs        # uniform cubic B-splines, the shape of every path
├── trajopt.rs       # plan: search, simplify, fit a spline, optionally optimize it
├── tool_path.rs     # follow: the hand in straight lines through poses
├── join.rs          # join: consecutive motions without stopping between them
├── task.rs          # pick and place: a whole task in one call
├── timing.rs        # Trajectory: a path timed within the limits; stops; moving starts
├── topp.rs          # time-optimal timing (TOPP-RA), smoothed for jerk
│
│   training data
├── datagen.rs       # reaches with recoveries, pick-and-place, handovers, reset states
├── npy.rs           # export as NumPy arrays
├── lerobot.rs       # export as LeRobot v3.0 datasets
│
│   shared
├── types.rs         # Pose, JointTrajectory, StartMotion, Solved
├── error.rs         # Error: the kinds a host matches on
└── rng.rs           # the generator every algorithm seeds from
```

Modules are grouped by what they do, in the order a call meets them. `demos/` is a crate of its own (Rerun), so the library never depends on a viewer.

## How it fits together

- **`Device` is the only way to run a batch.** It holds the robot and a CPU or GPU backend behind the crate-private `Backend` trait, and checks every batch (array shapes, world indices, worlds uploaded to it) before either backend sees it. Worlds go onto it once (`Device::upload`) and every algorithm takes them as `&Worlds`.
- **Both backends compute the same numbers.** The GPU kernels and `cpu.rs` implement the same math function by function, held together by parity tests. Searches run on CPU threads on either device, so results depend on neither the batch nor the device.
- **Algorithms are plain functions.** `solve_ik`, `plan`, `follow`, `join`, `pick`, `place` and `datagen` take a device, its worlds and problems, and exchange the shared types. Each owns its seeding, validation and choice of answer. There are no planner plugins and no runtime configuration.
- **Batch-first.** Every call covers many problems, and each problem names its world by index, so one call spans many scenes. Results carry their problems (`solved()`).
- **Seeds come from the problem.** Each problem's generator mixes the options' seed with the problem's own numbers, so a problem draws alike in any batch and on either device.

## Principles

The library's engineering principles, and where the code shows them:

- **The simplest implementation that fully works.** One path shape (uniform cubic B-splines) serves planning, tool paths, joins, timing and checks. Loaders only translate; everything after them sees one description.
- **Layers on a working product.** Each capability was added on top of ones that already worked end to end: tool paths on IK and splines, tasks on tool paths and planning, joins on paths from both, datagen on tasks.
- **Separate concerns.** Collision lives in the backends, searching in `rrt/` and `edges.rs`, timing in `timing.rs` and `topp.rs`. A change to what collides touches the CPU, the SIMD lanes, the GPU and `MeshModel` together, and nothing else.
- **Proven designs, studied first.** RRT-Connect and its edge checks as VAMP runs them; spline stops eased as OpenRAVE's smoother does; spheres fitted as cuRobo fits them; TOPP-RA for timing; allowed contacts as MoveIt's matrix; blends at a radius as Pilz and UR blend. Each is adapted to batches, and documented where it differs.
- **Decisions for the long term, without compatibility layers.** Obsolete paths are removed rather than kept beside their replacement; minor versions before 1.0 may change the interface.

## Planning a path

`plan` runs each problem as one search on one CPU thread. Only optional trajectory optimization runs on the device, batched over every problem.

1. **Search** (`rrt::Connect`). RRT-Connect as VAMP runs it: Halton samples, dynamic domains, balanced trees, a k-d tree for nearest neighbours. Samples come first from a tube around the straight line, widening until it spans the joint ranges; the search keeps a clearance from obstacles for its first samples, then continues the same trees without it. A straight line that is free, or one bent around the collision, skips the search.
2. **Simplify** (`shortcut::Simplify`). The straight line, then passes of greedy shortcuts and smoothing.
3. **Fit** (`trajopt::Fitting`). A B-spline through the simplified path, its control points spread by arc length. Where it collides, the nearest waypoint is pinned, so the curve passes through it; each pin is then eased into the widest blend that keeps the spline valid. Splines keep the shortcuts' clearance, then a 2 mm margin.
4. **Repair.** A path searched but left without a valid spline has its spline optimized a few steps on its own thread, and takes the first step that checks valid.
5. **Optimize** (optional, `PlanOptions::seeds`). The spline and seeds straight to the goal or bent through random points are optimized together on the device (L-BFGS) and validated densely; the shortest valid path wins.

Goals on joints whose range spans more than a turn are reached at whichever equivalent is nearest or free. Held joints never move. A start in motion pins the spline's first span to the current velocity and acceleration.

## Searches

A search (RRT-Connect, simplification, a spline's check, a tool path, a join) is a small state machine (`edges::Search`): it asks for an edge check, takes the answer and decides what to ask next. The SIMD lanes answer at once; a GPU device keeps a CPU copy of its worlds for them.

- **Edge checks** follow VAMP: eight states spread over the edge, each pass stepping back, stopping at the first collision. A link's spheres are placed only when its bounding sphere is near something, a sphere around each obstacle rules it out before the exact test, and distances are compared squared.
- **Scheduling.** Each thread takes the next search until none is left. Threads with nothing left help searches that have checked many samples (`rrt/ahead.rs`): they replay the trees from a log and check the edges the next samples will ask about. A search takes an answer only for a query equal to its own bit for bit, so it grows the same trees as alone.
- **Memory.** A search allocates nothing per query, and is built on the thread that runs it (`edges::Deferred`), so its memory belongs to that thread's allocator arena.

## Collision

The robot is a set of spheres; obstacles are boxes, spheres, cylinders, capsules or distance grids. A configuration's cost sums the squared penetration of every sphere into every obstacle, and of every self-collision pair worth checking, within a margin.

- **Gating.** Each link has a bounding sphere; an obstacle farther than the margin skips its spheres. Clearances stay exact up to the margin.
- **Gradients** accumulate a wrench per link; one reverse pass gives each joint its share.
- **Links no joint moves** are not checked against the world.
- **Allowed contacts** become one link bit set per obstacle at upload, which every check skips.
- **SIMD** (`lanes.rs`): forward kinematics and clearances for eight configurations in `f32x8` lanes, at the widest level the CPU has (SSE2 to AVX-512, NEON), chosen once per search and inlined all the way down. `cpu::collision` is the scalar twin the lanes are tested against.

## The GPU

- **The twin.** `kernels.wgsl` and `cpu.rs` implement the same functions: kinematics, the collision cost and gradient, the IK step, the optimizer's passes.
- **Layouts declared once.** Every struct the GPU reads is declared in Rust with `shader_struct!` and its WGSL generated from that, with shared constants written in from their Rust values.
- **Kernels written per robot.** `robot_wgsl` writes the kinematics, Jacobians and collision cost link by link, each in its own variable: run-time-indexed arrays spill to scratch memory on Mesa's RADV, and writing them out made GPU planning 4.8× faster on a Radeon and 2.7× on an M4 Pro. Small batches give each configuration 8 invocations that split its sphere loops.
- **Trajectory optimization** is L-BFGS with a parallel line search, five passes per round, split across submissions to stay under driver watchdogs.

## Tool paths, joins and tasks

- **`follow`** splits a path into poses a step apart and solves each by damped least squares from the configuration before, halving a step whose joint motion strays from the path; eight halvings is a jump. A spline then passes through the configurations, checked densely; a collision cuts the motion back.
- **`join`** blends consecutive rest-to-rest paths: it drops the control points around each join within the blend radius at the hand, bridges the gap with a curve tangent to both paths whose curvature rises from zero, and checks only the spans that changed. A blend is kept only where it makes the stretch so far and the next motion finish sooner than stopping between them, so no stretch takes longer than its motions apart. Paths are first split at their spans' middles (the same curve) until they are spaced alike, because timing follows a path best where its pace is even.
- **`pick` and `place`** run every task's grasps or place poses as candidates through the same steps, each one batch across every task: start, IK at the grasp or release, the straight move in (followed backward), the straight move away, the planned transit. Each step allows only its own contacts (the fingers on the object, the held object on its support). A held object is a link attached to the robot, so lifts and carries run on a device for the holding robot.

## Timing

A `Trajectory` is the path plus a time map `s = σ(t)`, itself a uniform cubic B-spline. `Trajectory::try_new` keeps the faster of uniform timing (exact along the whole curve) and time-optimal timing: TOPP-RA on the path parameter, with the jerk limit as a cap on speed, smoothed into the time map and stretched where the map overshoots a limit. Map spans are sized by the path and, for finely split paths, by the time the jerk limit takes to swing acceleration. A start in motion is timed from its speed along the path; a controlled stop brakes to rest along the path.

## Perception and training data

- **Distance grids** are a CPU/GPU twin too (`sdf.rs` / `grids.wgsl`): voxels classified from depth images or points, then an exact distance transform in integers, so both devices build identical grids. A robot mask keeps a camera that sees the arm from making it an obstacle to itself.
- **Training data** (`datagen.rs`) runs the planners in batches: reaches with recoveries from perturbed states, pick-and-place as a `pick` then a `place`, handovers that move both arms at once, and reset states. `npy` and `lerobot` export them.

## The robot model

`Robot::load` reads the description, closes kinematic loops (each loop's passive joints become polynomial mimics of its one actuated joint, fitted to the loop's static equilibrium), expands ball, planar and floating joints into one-axis joints, numbers the actuated joints breadth first, and fits collision spheres unless a model is given. `Robot::assemble` does the same for several descriptions, their names prefixed.

## Performance

Performance claims come from alternating runs of two builds on one machine; [Benchmarks](benchmarks.md#how-the-speed-came) lists what those runs found.
