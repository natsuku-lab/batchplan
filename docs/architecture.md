# Architecture

This page is for contributors: how the code is organised and why. `CLAUDE.md` holds the same rules in more detail, for people and coding agents alike.

## Design rules

- **Batch-first.** Every call covers many items, and items reference worlds by index (`IkProblem::world`, `PlanProblem::world`, `item_world`), so one call spans many scenes.
- **Worlds live on the device.** `Device::upload(&[World]) -> Worlds` prepares worlds once (GPU buffers, CPU rotation matrices); every algorithm takes `&Worlds`. A device checks that worlds were uploaded to it.
- **One public handle, two hidden implementations.** `Device` is the only way to run batched work. The CPU and GPU implementations sit behind a crate-private `Backend` trait and are tested against each other through `Device`. `Device` validates every batch before either sees it.
- **Algorithms are plain functions over shared types.** `ik`, `trajopt`, `rrt` and `datagen` take a `Device` and its worlds, `timing` and the exporters take the `Robot`, and all of them exchange the `types`. There are no planner plugins and no runtime configuration. Algorithm modules own seeding (one shared random generator), validation and selection, so both devices see bit-identical inputs.
- **Results carry their problems.** `solved()` yields each problem with its solution; the world comes from the problem, never from its position in a list.
- **Loaders stay thin.** URDF, MJCF and USD loaders only translate a file into one internal description. Kinematics, sphere fitting and self-collision analysis never see a format.
- **Errors are typed at the surface.** Public functions return `batchplan::Error`; its kind is what a host matches on.

## The CPU/GPU twin

The GPU kernels (`src/kernels.wgsl`, plus the per-robot code `gpu.rs` generates) and `src/cpu.rs` implement the same math function by function: forward kinematics, the collision cost and its gradient, the IK step, and the trajectory optimizer's passes. Grid building (`grids.wgsl` / `sdf.rs`) is a twin too. Any change to the math lands on both sides in the same change, and parity tests catch drift.

Every struct the GPU reads (parameters, links, spheres, obstacles) and every shared constant is declared once in Rust with `shader_struct!`; its WGSL declaration is generated from that, so host and shader layouts cannot drift.

The CPU also checks collisions eight configurations at a time in SIMD lanes (`lanes.rs`, `fearless_simd::f32x8` on stable Rust): forward kinematics, then world and self clearance. The lanes answer the CPU device's clearances and every search's edge checks. `cpu.rs` stays the line-by-line twin of the kernels, and a test holds the lanes to it on four robots and every obstacle kind. The lanes run at the CPU's widest SIMD level, found at run time (SSE2 up to AVX-512 on x86-64, NEON on ARM), so a default build needs no target flags. Code enters that level once per search and once per run of up to eight clearances that share a world; the code between that choice and the vector operations is inlined and has no closures.

## Searches

Sampling-based planning is sequential and branchy, so on either device it runs on CPU threads, one problem per thread, each search to its end (`edges.rs`). A search (RRT-Connect, simplification, choosing a spline) is a small state machine: it asks for an edge check, takes the answer and decides what to ask next. The SIMD lanes answer each check at once; a GPU device keeps the CPU form of its worlds for them. Results therefore never depend on the batch or the device. If no spline is valid, a search asks once for its first spline to be optimized; the thread running it does so, with idle threads helping, and the search checks the result. A search allocates nothing per query, since the allocator would serialize small allocations across a batch's threads: it writes each query's edges into a buffer its thread lends it and keeps its working configurations.

Edges are checked as VAMP checks them: eight states spread over the edge, each pass stepping back, stopping at the first collision. A link's spheres are placed only when its bounding sphere is near something, and tests compare squared distances. Cheap tests go first. A sphere around each box rules it out before the exact test (only 1.6% of the benchmark's link-against-obstacle tests find the link near), and in self-collision each sphere of one link is first tested against the other link's bounding sphere. A required clearance (`Keep`) adds a margin per lane that ramps down toward a path's ends; an edge entirely beyond the ramp takes the full clearance without measuring each state's distance to the ends.

A batch of searches runs on the device's threads as VAMP's batches do: each thread takes the next search from a shared counter until none is left, so no thread idles while another still holds a fixed share. A batch then lasts as long as its longest searches, often a few unsolvable problems that draw every sample. Threads left without a search therefore help searches that have checked 2,000 samples (`rrt/ahead.rs`), and sleep while none has and others still sample. A search planned alone summons helpers into the pool once it has checked 4,000.

RRT-Connect is sequential, but most samples extend from the node they would have extended from a few dozen samples earlier. A helper replays the search's trees from a log, forecasts the extension and connection each coming sample makes, and checks them; when the trees' sizes are near the ratio that decides which one a sample extends, it forecasts both. The search takes a forecast's answer only for the exact query it would have asked, bit for bit, so it grows the same trees as it would alone. Forecasts match where the trees are large and grow slowly, as an unsolvable search's do. On shorter searches, helpers answer few queries and slow those searches by keeping every core busy. A batch's helpers run the pool's other work while they wait. They also run it before each sample while another search's repair is spreading its samples over the pool, since the repair needs the threads.

## Collision

The robot is a set of spheres; obstacles are primitives or distance grids. One configuration's collision cost sums squared penetrations of every sphere into every obstacle, and of every sphere pair the self-collision model checks, within a margin.

- **Gating.** Each link has a bounding sphere around its spheres. An obstacle farther from it than the margin skips the link's spheres; a link pair whose bounding spheres are farther apart than the self margin skips its sphere pairs. The gap stands in for their clearance, so clearances stay exact up to the margin.
- **Gradients.** Collision gradients accumulate a wrench per link (force, and moment about the origin); one reverse pass from the leaves gives each joint its share.
- **Links no joint moves** are not checked against the world: their contact with the scene does not depend on the configuration.

## Kernels written per robot

`robot_wgsl` writes the robot-specific part of the GPU code: forward kinematics, the end effector's Jacobian and the collision cost, link by link, with each link's frame and wrench in its own variable. Mesa's RADV driver moves any private array indexed at run time and larger than 256 bytes into scratch memory. Per-link arrays cost 3–5 KB of it per invocation; writing the code per robot made GPU planning 4.8× faster on a Radeon and 2.7× on an M4 Pro. Kernels compile per robot shape and are cached, so attaching different objects to one link reuses them.

For small batches, the collision kernels give each configuration 8 invocations ("lanes") that split its inner sphere loops. Each is built for 1 lane and for 8, through a WGSL override constant.

## Trajectory optimization on the device

Optimization is optional in `plan` (`PlanOptions::seeds`) and runs batched on the device.

Each L-BFGS round dispatches five passes: the collision cost at every sample of every path moved by each of four step sizes; a per-path pass that takes the cheapest step; the collision gradient at every sample; the gradient per control point; and a per-path pass for the next L-BFGS direction. The per-path passes run one workgroup per path, with workgroup reductions. Work is split across queue submissions to stay under driver watchdogs. Paths are optimized in chunks whose state fits one storage binding.

## Data pipeline

`datagen` plans in batches and times trajectories with `Trajectory::new(..).sample(1 / dt)`. Exporters write `Demonstration`s: `npy` as arrays, `lerobot` natively in LeRobot's v3.0 layout (parquet through Arrow, streamed to rolling files, with per-episode statistics aggregated as LeRobot does).

## Performance work

Performance claims come from alternating runs of two builds on one machine, after checking the machine is otherwise idle; a single run proves nothing. [Benchmarks](benchmarks.md#how-the-speed-came) lists what those runs found.
