# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

batchplan is a Rust library for batched, collision-aware robot motion generation: IK and trajectory optimization over thousands of seeds, problems and worlds at once. It runs on any GPU wgpu supports (Vulkan, Metal, DX12) without CUDA or ROCm, and its CPU implementation produces the same numbers. A primary purpose is generating robot-learning training data: nominal demonstrations plus *recoveries* from perturbed states, exported as `.npy` arrays or LeRobot v3.0 datasets.

## Commands

```bash
cargo build --release --all-targets [--features lerobot] [--no-default-features]   # without `gpu`: CPU only, no wgpu
BATCHPLAN_REQUIRE_GPU=1 cargo test --release                    # 142 tests, guides' code included; without the env var, GPU tests skip silently when no adapter exists
BATCHPLAN_REQUIRE_GPU=1 cargo test --release --features lerobot # + 6 export tests and the README's and training-data guide's code
BATCHPLAN_REQUIRE_GPU=1 cargo test --release --features usd     # + 7 OpenUSD tests
cargo test --release --test gpu trajopt_directions_match_cpu_element_wise   # one test (test files: cpu, gpu, device, budget, export, pick_place, robot, hands, trajectory, mjcf, usd, sdf, meshes, rrt, attach, threads, allocations)
cargo fmt --check && cargo +nightly fmt --check -- --config imports_granularity=Module,hex_literal_case=Lower,format_macro_matchers=true   # 100 columns; the second checks the style guide's nightly-only options (CI runs both)
cargo clippy --release --all-targets [--features lerobot,usd | --no-default-features]   # zero warnings everywhere; CI denies them (lints in Cargo.toml)
cargo build --release --lib --target aarch64-unknown-linux-gnu --no-default-features     # the CPU-only robot build CI checks
cargo doc --no-deps --features lerobot,usd                      # keep at zero warnings
git tag -a v0.1.1 -m "batchplan 0.1.1" && git push origin v0.1.1   # a release: bump Cargo.toml's version, wait for CI on main, tag; release.yml publishes to crates.io
cargo run --release --example bench -- 512                      # GPU vs CPU throughput; BENCH_LLVMPIPE=1 adds the WGSL kernels on Mesa's CPU Vulkan driver
scripts/fetch_benchmark.sh && cargo run --release --example benchmark   # MotionBenchMaker + MπNets (2,600 Panda problems), GPU and CPU
cargo run --release --example benchmark -- --spheres vamp --check <paths.json>...   # one judge for every planner's paths (`--save` writes batchplan's; scripts/compare_planners.py and scripts/cpp_planners.cc the others')
scripts/fetch_rebot.sh && cargo run --release --example handover -- 512 [--cpu]   # two reBot arms handing blocks over: the bimanual benchmark
cargo run --release --features lerobot --example handover -- 16 --lerobot data/handover && .venv/bin/python scripts/replay_mujoco.py data/handover data/handover_video   # + MuJoCo camera videos
cargo run --release --example handover -- 512 --resets data/resets   # a reset bank: the episodes and states perturbed from them, as .npy
cargo run --release --example datagen -- data/demo 512 20       # .npy dataset: <out_dir> [worlds] [fps]
cargo run --release --example depth [-- --cpu]                  # a distance grid from a rendered depth image, then plans around it
cargo run --release --example control_loop -- [seconds]         # a planner thread feeding a 50 Hz loop
cargo run --release --features lerobot --example datagen -- --lerobot data/lerobot_demo 512 20   # --pick-place for pick-and-place, --file-mb N to split files sooner
REMOTE=user@host SSH_OPTS='...' scripts/sync.sh '<command>'     # rsync to ~/batchplan on a GPU box and run there
(cd demos && cargo clippy --release --all-targets && cargo run --release --bin handover -- 16 --save handover.rrd)   # Rerun demos: handover, pick, perception, replan (demos/README.md)
```

Checking a LeRobot export with the real package needs a git-ignored `.venv`:

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python "lerobot[dataset,training]==0.6.1"
.venv/bin/python scripts/validate_lerobot.py data/lerobot_demo
.venv/bin/lerobot-train --dataset.repo_id=local/batchplan --dataset.root=data/lerobot_demo --policy.type=act \
  --policy.device=cpu --policy.push_to_hub=false --steps=50 --batch_size=16 --num_workers=0 --save_checkpoint=false --wandb.enable=false --log_freq=10
```

In the same `.venv`:
- Rendering camera videos into an export with MuJoCo: `uv pip install --python .venv/bin/python mujoco fast-simplification`, then `scripts/replay_mujoco.py <dataset> <out_dir>`. It reads `meta/replay.json` (finger joints, cameras), which the exporting example writes.
- Checking USD loading against Pixar's `usd-core`: `uv pip install --python .venv/bin/python usd-core && .venv/bin/python scripts/validate_usd.py`, which runs `examples/usd_fk.rs` (feature `usd`) for batchplan's side.

## Design rules

These decisions are settled. Keep to them unless the user decides otherwise.

- **Batch-first.** Every query covers many items. Items reference worlds by index (`item_world`, `IkProblem.world`, `PlanProblem.world`), so one call spans many scenes.
- **Worlds live on the device.** `Device::upload(&[World]) -> Worlds` prepares worlds once (GPU buffers, CPU rotation matrices); every algorithm takes `&Worlds`. Each backend's `upload` returns its own form, which `Worlds::prepared` hands back. `npy::export` and `lerobot::export` take the `Robot` and `&[World]` (pass `&worlds`: `Worlds` derefs to the worlds uploaded). `Device::with_robot` shares the device id (and the GPU context or thread pool), so worlds stay valid when the robot changes, as with `Robot::attach`. Attached objects are fixed links appended after every robot link.
- **One public handle, two hidden implementations.** `Device` (`device.rs`) is the only way to run batched work. `CpuBackend` and `GpuBackend` sit behind the crate-private `Backend` trait. Never make `Backend` public, and never add a way to call a backend that bypasses `Device`.
- **`Device` validates every batch** (array shapes, world indices, the device the worlds were uploaded to) before either backend sees it, so malformed input is an `Err` on both devices. New `Device` entry points must go through `check_batch`.
- **Algorithms are separate modules over shared types.** `ik`, `trajopt`, `timing`, `datagen`, `npy` and `lerobot` are plain functions taking `&Device`, `&Worlds` and the `types` (`Pose`, `JointPaths`, `JointTrajectory`, `Solved`). There are no planner plugins and no runtime configuration.
  - Algorithm modules own seeding (the shared `rng::Rng`), validation and selection, so both devices see bit-identical inputs.
- **Results carry their problems.** Use `IkResult::solved()` / `PlanResult::solved()`, which yield each problem with its best solution. Take the world from the problem, never from its position in the list.
- **Small interfaces.** `Robot` exposes accessors and pose queries only; its internals are `pub(crate)`. Prefer deepening an existing module to adding a new public one.
- **Loaders stay thin.** A loader (`urdf.rs`, `mjcf.rs`, `usd.rs` behind feature `usd`) only translates a file into the crate-private `Model` (`description.rs`): a `RobotDescription` and static scene shapes. Kinematics, sphere fitting, self-collision analysis and `World::load` work on those, never on a file format. Format semantics are translated, not approximated away: MJCF mesh geoms become `Geometry::ConvexHull` because MuJoCo collides their hulls.
- **Distance grids err toward collision.** `sdf.rs` builders store at most the true signed distance per point, minus half a voxel diagonal (trilinear interpolation's largest overestimate). Keep that invariant when touching a builder; `tests/sdf.rs` checks it.
- **Typed errors at the surface.** Public functions return `batchplan::Error` (`error.rs`). A host matches on its kinds, so choose the kind deliberately (`ensure_input!`/`input!` for bad input). The format loaders and robot building keep `anyhow` internally for context chains; `Robot::load`, `World::load` and `CollisionModel::load` turn them into `Error::Load` with the path.
- **Embeddable.** Each device owns its rayon pool (`Device::cpu_threads`; the GPU device's has one thread per core).
  - Every parallel loop in `cpu.rs` and `gpu.rs` runs inside `self.pool.install` or on `cpu::run_searches`' broadcast, never on the global pool. `tests/threads.rs` counts OS threads; `tests/budget.rs` checks that a busy global pool does not hold up GPU plans.
  - Batches too small to be worth handing over (`SMALL_BATCH` clearances, a single search) run on the calling thread.
  - `Robot::attach` covers boxes and spheres directly (`spheres::cover_primitives`, microseconds) and fits other shapes on the caller's pool, or on the calling thread with plain loops (`spheres::fit(.., parallel: false)`), not on a local pool: one built with `use_current_thread()` leaves the thread registered with rayon for good, and every parallel loop it starts afterwards runs on that one thread.
  - `PlanOptions::time_budget` becomes a deadline passed to `Backend::search` (checked before every edge check) and `Backend::trajopt`. Without one, nothing waits or checks the clock, and results are deterministic.
- **USD stays optional.** The `openusd` crates are pinned exactly (`=0.7.0`, pre-1.0) and only built with feature `usd`.
- **Standalone, with optional bridges.**
  - The core has no middleware and reads no environment variables; only tests and examples do (`BATCHPLAN_REQUIRE_GPU`, `BENCH_LLVMPIPE`).
  - LeRobot export is behind the `lerobot` cargo feature, so the Arrow/Parquet dependencies stay optional.

## Architecture that spans files

**CPU/GPU twin.**
- The GPU kernels (`src/kernels.wgsl` and `src/grid.wgsl`, plus the per-robot code `robot_wgsl` writes in `gpu.rs`) and `src/cpu.rs` implement the same math function by function: `fk`, `collision`, `rot_log`, `chol6`, `ik_step`, the trajectory passes (`traj_costs`/`traj_cost`, `traj_search`, `traj_samples`/`traj_sample_grad`, `traj_grad`, `lbfgs_direction`). `spline.rs` holds the B-spline basis that `basis` in WGSL mirrors; `world.rs` and `sdf.rs` (`grid_distance`) hold the obstacle distances that `obstacle_distance` and `grid_distance` mirror.
- `cpu.rs` keeps index loops on purpose so the two read side by side.
- Any change to the math lands in both files in the same change. The parity tests in `tests/gpu.rs` and `tests/device.rs` catch drift.
- Grid building has its own twin: `src/grids.wgsl` mirrors `sdf.rs`'s `DepthImage::classify`/`sees`, `RobotMask::contains`, `depth_occupancy`, `integrate`, `squared_edt`/`Line`/`meet` and `finish`. Builders (`SdfGrid::from_points`/`from_depth`, `OccupancyMap::integrate`/`grid`) take a `Device` and go through `Backend::grid_values` and `Backend::integrate`.
  - A `RobotMask` (`DepthImage::robot_mask`) carries each pixel's depth to the robot and its grown spheres. The space they fill is free, so a robot is never an obstacle to itself, even where hidden space is solid. On the GPU the spheres follow the depths in binding 2.
  - Space behind the robot along a pixel that sees it is `Seen::Shadow` (`SHADOW` in grids.wgsl), solid only under `Occlusion::Occupied`. Maps mark such voxels `SHADOWED` until an image observes them, and `OccupancyMap::grid` treats them as free under `OccupiedExceptRobotShadow`.
  - The distance transform is integer arithmetic, so both devices build identical grids from the same occupancy (`every_device_builds_the_same_grids_from_occupancy`).
  - Each value is finished by a table the host builds (`finishing_table`, sized by the grid's squared diagonal, hence `MAX_DIAGONAL`), so the GPU never needs `sqrt` to match the CPU.
  - Classification projects voxel centres in floating point and may differ at rounding ties. The internal test `every_device_classifies_depth_images_alike` bounds that at 0.05% of voxels; the behavioural tests in `tests/sdf.rs` run on every device.
  - The GPU keeps its grid buffers (`GridBuffers`) between builds, because wgpu zero-fills new buffers.

**Host/shader structs.**
- Each struct the GPU reads is declared once with `shader_struct!` in `gpu.rs`: `GpuParams`, `GpuLink`, `GpuSphere`, `GpuObstacle` and the grid builder's `GpuGridParams`. The constants (`MAX_HISTORY` and the line-search steps `LINE_SEARCH` from `trajopt.rs`, obstacle kinds, `WORKGROUP`) are written into the shader prelude from their Rust values.
- Their WGSL declarations are generated from the same field list. `gpu.rs` builds the shader source as: generated prelude (`alias Vec4`, the constants, the structs) + `robot_wgsl(robot)` + `kernels.wgsl` + `grid.wgsl` (`grid_distance`, mirroring `sdf.rs`). Grid building (`grids.wgsl`) is a separate module that needs no robot.
- To add a kernel parameter:
  1. Add a field to the right `shader_struct!`.
  2. Fill it in `GpuBackend`.
  3. Read it in WGSL as `P.<field>`.
  4. Mirror it in `cpu.rs`.
- There are no fixed size limits on robots.
  - The CPU kernels keep per-configuration state in `Scratch` (`cpu.rs`: FK frames, sphere centres, wrenches, IK buffers). It is sized to the robot and reused per thread (`for_each_init`, or one per path), so hot loops neither allocate nor clear more than the robot needs. Inline small vectors were tried first and cost 5–27% of CPU throughput.
  - The GPU kernels are written per robot. Self-collision is unrolled per link pair up to `UNROLLED_LINK_PAIRS` (64); beyond that, a run-time loop over frames copied into arrays keeps the generated code growing with links, not pairs.
- Links no joint moves (`Link.chain` empty: the base and what is fixed to it) are not checked against the world on either device, nor by `MeshModel`: their contact with the scene does not depend on the configuration, and a mounted robot's base rests on its table. They still self-collide.
- Collision is gated by each link's bounding sphere (`link_bounds`, `GpuLink.bound`); the gap stands in for the clearance of whatever it skips.
  - World: an obstacle farther from a link's bound than `max(margin, 0)` skips the link's spheres (`Robot::sphere_ranges`). Distance grids are never gated: their interpolated distance is not 1-Lipschitz.
  - Self: sphere pairs are grouped by link pair (`Robot::self_link_pairs`). A link pair whose bounds are farther apart than `max(self_margin, 0)` skips its sphere pairs. On the GPU, the `pairs` buffer starts with one `(first, count)` entry per link pair, followed by the sphere pairs.

**Per-robot GPU kernels.**
- `robot_wgsl` writes the robot-specific code: `MAX_DOF`, `HANDS`, forward kinematics, each hand's frame and Jacobian (`ee_pos(h)`, `ee_rot(h)`, `ee_jacobian(h)`, `mark_moving(h)`, switches with hand 0 as the default) and the collision cost.
  - IK (`ik_step`, twinned in `cpu.rs`) runs two passes: each targeted hand's pose step, then the collision gradient pushed out of each targeted hand's Jacobian row space (`grad − Σ J^T A^-1 J grad` on the joints the targeted hands move; held joints never move). The hand that went last keeps its system in `jac`/`chol`, so a single hand computes it once. For `HANDS == 1` the kernel takes a branch without the loop over hands: the loop cost 4–10% of GPU IK throughput on the M4 Pro.
  - Each link's frame (`rot_i`, `pos_i`) and collision wrench (`force_i`, `moment_i`) is its own private variable, indexed only by constants.
  - Sphere and pair ranges come from the buffers (`GpuLink.first_sphere`, the `pairs` header), so robots that differ only in their spheres share code.
- Why: Mesa's RADV moves any run-time-indexed private array larger than 256 bytes into scratch memory. With per-link and per-sphere arrays, every collision kernel spilled 3–5 KB per invocation. Writing the code per robot removed that and made GPU planning 4.8× faster on the Radeon and 2.7× on the M4 Pro. Never add such an array to a collision kernel. To check on RADV, `MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaderstats cargo run --release --example bench -- 8` prints each pipeline's VGPRs and scratch size.
- Collision gradients accumulate a wrench per link (force, and moment about the world origin). One reverse pass, leaves first, adds each link's wrench into its parent and gives each joint `a·(M − o×F)` (revolute) or `a·F` (prismatic). `cpu.rs` does the same.
- `GpuBackend` compiles kernels per robot shape and caches them by generated source, shared by every backend `with_robot` derives.
- **Lanes.** The collision kernels (`evaluate_main`, `clearance_main`, `ik_main`, `traj_costs`, `traj_samples`) give each configuration `LANES` consecutive invocations. `LANES` is a WGSL `override`, and each kernel is built twice:
  - with 1 lane for large batches, where the lane code folds away;
  - with `SHARED_LANES` (8) for batches whose configurations times 8 fit within `LANE_TARGET` threads.

  Lanes split the inner sphere loops of `collision()`. Every lane runs every block, so a wave stays converged; splitting whole blocks across lanes diverged and was slower. `group_collision` and `group_grad` combine the lanes. These kernels have no early returns, because lanes meet at barriers. `evaluation_does_not_depend_on_the_batch` checks that both builds agree. Pipelines skip workgroup zero-initialization: every kernel writes its workgroup memory before reading it.

**GPU details.**
- Bind group 0 has 12 storage buffers. `unmet_limits` skips adapters that can't provide them, and the error names each rejected adapter and what it lacks.
- `GpuWorlds` holds every world's obstacles, per-world ranges and `grid_data`: each distinct grid (by `Arc` pointer) once, two half floats per `u32`, read with `unpack2x16float`. Grids store subnormal halves as zero because GPUs may flush them.
- Storage buffers are created by `storage::<T>()`, padded to at least one shader element: empty obstacle lists used to crash.
- Work is split per queue submission (`EVAL_CHUNK`, `IK_ITERS_PER_SUBMIT`, `TRAJ_ITERS_PER_SUBMIT`) to stay under driver watchdogs.
- Trajectory optimization is L-BFGS with a parallel line search. Each round dispatches five passes:
  1. `traj_costs`: the collision cost at every sample of every path moved by each line-search step;
  2. `traj_search` (one workgroup per path): the cheapest step moves the path if it lowers the cost, otherwise the history resets;
  3. `traj_samples`: the collision gradient at every sample;
  4. `traj_grad`: per free control point, basis-weighted sample gradients plus smoothness;
  5. `lbfgs_direction` (one workgroup per path): record the last step and gradient change, then the two-loop recursion.

  The first round only prices the seeds (their direction is still zero). In the per-path passes, each invocation owns every `WORKGROUP`-th element (coalesced loads), and sums go through `workgroup_sum`/`workgroup_max`, whose results come back through `workgroupUniformLoad` so later barriers stay in uniform control flow. `aux` holds each path's L-BFGS state (`lbfgs_stride`, mirrored by the `Lbfgs` struct in `cpu.rs`); paths are optimized in chunks whose state fits one storage binding.

**Held joints.** `PlanProblem::hold` / `RrtProblem::hold` keep joints at their start values: `Connect` samples them at the start and never bends along them (`line_to`), `search_goals` drops goal variants that turn them, and `hold_still` puts optimized splines (repair and seeds) back on them before their check. Shortcuts, straightening and spline fitting keep them because every waypoint has the same value; timed samples keep them up to the blend's rounding.

**Planning flow.** In `trajopt::plan`, each problem is a `Planning` search (see **Searches** below), run whole on one CPU thread on either device:
1. `rrt::Connect`: the straight line to each goal variant (`Robot::goal_variants`), else that line bent (`Stage::Locate`, `Stage::Bend`: the stretch where it collides moved aside along each joint's axis, `BEND_RADII` off it), else VAMP-style RRT-Connect:
   - Halton samples, range 1 rad, dynamic domains, balanced trees, connect in equal steps, a k-d tree for nearest neighbours unless a joint is continuous (leaves of up to 128 nodes keep their coordinates in groups of eight per joint, as nigh's batched k-d tree).
   - The first samples come from a tube around the straight line to the nearest free goal: `TUBE_RADIUS` across it and past its ends, doubling every `TUBE_DOUBLING` samples until it spans the joint ranges. Its orthonormal basis across the line (`Connect::across`) is set once, when the straight line first collides. The tube's samples come on top of `max_iterations`; none are drawn when it is 0.
   - It first searches keeping `rrt.clearance` for up to `KEEP_ITERATIONS` samples, then `Connect::relax` continues the same trees without it for `max_iterations` more.
   - If no spline follows a path found after sampling the tube (`Planning::near_line`), the problem is searched again with `Sampling::JointRanges`, which draws the samples a search without the tube would.
   - Continuous joints step and measure the short way (`Robot::turn_toward`), and the found path is unwrapped.
2. `shortcut::Simplify`: VAMP's simplification (straight line, then passes of greedy shortcuts and smoothing), keeping `shortcut.clearance`. It records the path after the first pass's shortcuts.
3. `Fitting`: `fit_spline` spreads control points by arc length along the smoothed path, then the shortcut path. It pins (triples) the waypoint nearest each failed sample, up to `MAX_PINS`; with every waypoint pinned, the spline traces the path exactly. A valid spline's stops are then eased: each pin passes within the widest of `BLENDS` blends that keeps the spline valid, and stays `EXACT` (a stop) where none does, as OpenRAVE's smoother removes a path's stops. Splines are checked keeping `shortcut.clearance`, then `SPLINE_MARGIN` (2 mm, so a curve cannot graze an obstacle between its samples; never more than `shortcut.clearance`). Configuration queries locate the failure. The first spline that passes is seed 0. A path simplified to its straight line takes its spline unchecked (`Planning::line_checked`): simplification only cuts a path to that line once it checks free keeping the shortcuts' clearance, and a straight line the search found covers that when `covers_splines` and there is one goal.
4. Moving starts (`start_motion`): splines must also pass `Trajectory::moving` (a failure retries with every waypoint pinned). Problems searched but with no valid spline, moving or not, are repaired inside their search (`Phase::Repair`). `Search::optimize` hands the searched spline to the thread running the search, which optimizes it one step at a time, up to `REPAIR_ITERATIONS` (10) (`cpu::PathOptimizer`, which resumes where it stopped, so the steps match one uninterrupted run; its samples spread over idle pool threads). The search checks each step's spline keeping the splines' last clearance (`SPLINE_MARGIN`), and the last step's without; the first that passes is taken. Most pass after one to three steps, and more steps trade the clearance back for length and leave fewer valid. Both devices repair on CPU threads, so they repair alike.
5. With `seeds > 0`, `optimize` runs batched trajectory optimization on the device: the searched spline, then `seed_path` seeds straight to the goal and its variants, then bent through random via points, validated densely (`validate`, `Device::clearance`). Problems searched to within `OPTIMIZE_SLACK` of the straight line are not optimized (their seeds stay invalid), and optimization does not start once the time budget is spent. `best()` picks the shortest valid path of all. The options' optimization weights favour length; repairs take `repairing`'s own (10 steps, `acceleration_weight` 50, `velocity_weight` 2, a 2 cm margin), which repair more splines.

**Searches.** `edges.rs`: a `Search` (RRT-Connect, simplification, planning) asks for `EdgeQuery`s and takes answers.
- `request(buffer)` writes a query's edges into the thread's buffer (`edges::reuse`), which the driver hands back after the answer, and searches keep their working configurations as fields. A search allocates nothing per query, since glibc's allocator serializes small allocations across a batch's threads.
- Batched calls build each search on the thread that runs it (`edges::Deferred`, in `plan`, `rrt::connect` and `shortcut::shortcut`). Built on the calling thread, the memory a search grows belongs to the caller's malloc arena, and every thread of a batch waits for that arena's lock: 32 threads planned easy reaches slower than 4. `tests/allocations.rs` counts the calling thread's allocations per problem.
- Answers are how many edges from the first are free (`Wanted::Prefix`), the first free one (`FirstValid`), or how many configurations from the first are free, in groups of `LANES` (`wanted: None`).
- `Device::search` → `Backend::search` runs every search to its end on CPU threads (`edges::run`, which also runs the optimizations a search asks for), answering at once with `lanes.rs`: each edge's `LANES * n` states `spacing` apart, in rake order (spread over the edge, each pass stepping back), stopping at the first collision. `Keep` adds a clearance per lane that ramps to zero toward a path's ends (an edge beyond the ramp, `Keep::beyond_ramp`, takes the full clearance without a distance per state). The GPU backend keeps the CPU form of its worlds (`GpuWorlds::cpu`) and a `Lanes` for this, so searches give the same answers on both devices.
- On either device (the GPU backend also plans through `cpu::run_searches`), each pool thread takes the next search from a shared counter (`ThreadPool::broadcast`), then helps the searches still running that have checked `ahead::JOIN_AFTER` samples. While none has and some still sample, it sleeps (`cpu::WAIT`, after the pool's other work). `Search::help` hands `cpu::run_searches` an `rrt::Ahead`, and `rrt/ahead.rs`'s helpers check the extensions and connections RRT-Connect's next samples will ask about, on trees replayed from its log. A search takes an answer only for a query equal to its own bit for bit, so results stay the same.
- A search alone stops after checking `ahead::ALONE_AFTER` samples (`edges::run`'s `pause`) and goes on in a scope with helpers it summons into the pool.
- `Connect` deals with its helpers only through `ahead::Sharing`, `report_sample` and `forecast`, all in `ahead.rs`. It reports each sample it checks, except those its dynamic domains turn down. A search planned again on the same `Ahead` is helped as a new one. Batch helpers run pool work before each sample while `edges::run`'s `spread` counter shows a repair spreading its samples.

**SIMD collision checks.** `lanes.rs` runs eight configurations per call in `fearless_simd::f32x8` lanes, generic over the SIMD level (`S: Simd`): forward kinematics (a revolute joint's rotation folded into `fixed + cos * C + sin * S`), link-bound gating, and world and self clearance (`Lanes::clearance`, behind `CpuBackend::clearance`) or validity with early exit (`Lanes::keeps`).
- `Lanes::keeps` shortcuts: squared distances; sphere centres placed per link only when its bound is near something; a box first ruled out by its surrounding sphere (`Prepared::Cuboid::radius`) and turned in the plane alone when `upright`; each link's bound centre computed once per check; in self-collision, each sphere first held against the tighter link's bound (`SphereGroups`). They never change an answer: their spheres are a hair wide.
- `cpu::collision` stays the line-by-line twin of the GPU kernels; `lanes_match_the_scalar_twin_on_every_robot` holds the lanes to it. The parity test also runs `Level::baseline()` (SSE2 on x86-64).
- Callers `dispatch!` on `Lanes::level()` (the CPU's widest level, found at run time) once: `edges::run` per search, `CpuBackend::clearance` per run of eight. Everything between the dispatch and the vector operations is `#[inline(always)]` code without closures. A closure is a function of its own without the dispatched level, so its SIMD operations become calls (seven times slower on x86); dispatching per `keeps` call costs 11%.

`timing::Trajectory` is the path plus a time map `σ(t)` (a uniform cubic B-spline in time). `Trajectory::try_new` keeps the faster of uniform timing (`σ = t / h`, `h` from control-point differences, exact along the whole curve) and `topp::time_map`: TOPP-RA on the path parameter, smoothed (Schoenberg), with caps lowered locally where the smoothed map overshoots and a final stretch measured at `CHECK_SAMPLES` per map span. `check` is exact for uniform maps and samples `CHECK_SAMPLES` per span otherwise. Moving starts (`PlanProblem::start_motion`):
- `trajopt::continue_motion` pins the first three control points for a knot interval `h0`, at least the time the robot needs to change its velocity (`max |v_j| / a_j`): shorter first spans put the jerk at a fast start over its limit. `next_fit` tries `START_KNOTS` multiples of it before pinning waypoints;
- `Trajectory::moving` recovers `h0` from them and times the path from ṡ = 1/h0. Stage 0 is held to the full limits, candidates are refined at full density against limits less the margin, and nothing is stretched;
- `check_from` checks the start state. A robot slows everywhere through `RobotOptions::velocity_scale`, `acceleration_scale` and `jerk_scale`, applied to its limits at load; `Trajectory::moving` has no speed scale of its own. `check` measures joint ranges on the curve (`span_range`, each span's exact extremes), not the control points: a moving start's first control point lies behind the start. Likewise `runs_forward` checks the time map's rate along each span exactly (a quadratic), since an accelerating start's map points may dip.
- `Trajectory::stop(robot, t)`: a controlled stop. It keeps the path from the span it starts in (dropping whole spans shifts `s` exactly, and a map near zero keeps f32 precision) and a new time map (`stop_map`): three points continuing (s, ṡ, s̈), steps falling evenly to zero over `m` spans, three equal points at the end. For each of `STOP_SPANS`, knot intervals double then bisect to the shortest within `topp::within_limits`; the soonest stop along the path wins. A trajectory's time map may therefore start partway along its path (from a moving start) and end before the path's end, where three equal map points bring it to rest; `at` snaps to a path end only where the map reaches it.

**Data pipeline.**
- `datagen::demonstrations` runs:
  1. IK with many seeds.
  2. Nominal plans from noisy, collision-free starts.
  3. `recovery_problems`: perturb solved paths and keep collision-free starts.
  4. Replanning from those starts.
  5. `Trajectory::try_new(..).sample(1 / dt)`: timing within the robot's limits at a random speed scale.
- `datagen::pick_and_place` chains segments per task: approach (the object an obstacle), descend (worlds without the object, at index `tasks + i`), close, lift (the descent reversed), transfer (on a `with_robot` device holding the object, one per object size), lower, open, retreat (the lowering reversed).
  - IK above the object takes the solution nearest `default_q` (not `best`, the most clearance, which can sit on a far branch and make every later move swing the arm round). IK for the grasp and the place pose is seeded from the configuration above them and takes `IkResult::nearest`.
  - The grasp and the pose above the place pose are one batch for the bare robot (the held object hangs below the hand there, and the transfer's plan, which holds it, checks that pose), so object sizes cost no IK call each; only the transfer plans per size, on CPU threads.
  - On a robot with several hands it uses the first and holds the joints only the others move. `Episode` concatenates rest-to-rest segments, dropping each one's duplicate first sample, with one gripper opening per hand per sample (`Demonstration::gripper` is `[samples, hands]`).
- `datagen::handover` passes a cuboid between two hands: six IK batches (above the giver's end; its grasp; both hands at the meeting pose; the taker's grasp there; the giver lifted off and the taker above the place pose with the giver home; the place pose), then seven planned segments from a table (`segments`) of (from, to, who holds the object, which arm waits). Batches seeded with the configuration before them use `NEAR_SEEDS`. Waiting arms are held (`PlanProblem::hold`). Plans that hold the object run on `with_robot` devices per (size, hand), with both wrists as touch links; the lowering plans bare, as in `pick_and_place`. Grasps are `Grasp::at` the ends of the longer side (`HANDOVER_GRIP`), each hand facing as its arm's does at the default configuration (`Arm`).
- `datagen::reset_states` turns demonstrations into a reset bank: samples with joint noise, at rest, as one-sample `Demonstration`s whose `Origin::Recovery` parent indexes the input, so `npy::export` writes them unchanged. A carried object follows the one hand closed on it, in the grip from that hand's last sample holding it alone (where a gripper has just closed, the object still sits where it was). Samples with both hands closed on it are skipped. States are checked in one world each (the object placed, or left out when held), batched per hand and grip on `with_robot` devices holding it, that hand's wrist links touching it. `same_hold` compares rotations by `dot`, since `Quat::angle_between` turns quaternion-norm rounding into milliradians and split one grip into hundreds of batches.
- `Demonstration` carries `task`, the gripper's opening per sample and an optional `Carried` obstacle with its pose per sample; `check_demos` checks their lengths.
- `Origin::parent()` encodes recovery links for both exporters.
- `lerobot.rs` writes the v3.0 layout natively:
  - data and episode metadata through `Files`, which rolls over past `data_files_size_in_mb`; each episode row records its own file. Global stats are `Aggregate`d from per-episode stats as LeRobot's `aggregate_feature_stats` does, so nothing holds every frame. Keep features in column order (`serde_json::Map` sorts its keys);
  - `tasks.parquet` lists the distinct tasks with pandas index metadata; `info.json`, `stats.json`, and a `meta/batchplan.json` extension.
  - The state is laid out arm by arm (`state_layout`): the joints no hand moves, then each hand's `hand_joints` and its gripper (`left_gripper` from the part prefix, `gripper` on one arm); `action` is the next frame's state. `meta/batchplan.json` also describes the robot (`robot_json`: parts, joints, hands) for simulators that replay the episodes. `observation.environment_state` holds the goal plus padded obstacles, each where it is at that frame.
  - Extension columns use names outside `observation.*` and `action*`, so LeRobot policies ignore them.

**Robot model.**
- `Robot::load` runs:
  1. `description::load_robot` (by extension; `Robot::from_urdf` takes URDF text to `urdf::from_text` instead);
  2. if the description has loops, `loops::close_loops` on the open tree: each loop's passive joints become quartic mimics of its one actuated joint (the least spring energy that closes the loop within their limits, by projected Levenberg-Marquardt over the driver's range; `tests/mjcf.rs` holds MuJoCo's settled 2F-85 as the reference);
  3. `kinematics`: planar, ball and floating joints expanded into one-axis joints on massless links by `expand_compound_joints`, so nothing downstream sees them; a breadth-first tree, actuated joints numbered in that order; mimic joints resolved to their driving joint with a `Curve` (a polynomial of degree ≤ 4, composed along mimic chains; `Fk` records each nonlinear joint's slope, which gradients and Jacobians use in place of the multiplier); locked joints baked into fixed origins;
  4. the collision model (given, or fitted by `spheres.rs`);
  5. SRDF pairs.
- `Robot::assemble(&[Part])` loads each part's description, prefixes its link and joint names (mimics and loops included) and fixes its root to `world` at the part's base (`description::assemble`); the rest is `Robot::load`'s path. `Robot::parts` records where a robot came from, for exports and simulators. A robot's hands are `ee_links` (`RobotOptions::ee_links`, else the deepest link); `hand_joints(h)` are the actuated joints of a hand's chain.
- Continuous joints (`Robot::continuous`) are unbounded reals whose values wrap every 2π; `lower`/`upper` give them only the turn that seeds and samples come from. Clamp and check with `Robot::bounds`/`Robot::within`, never with `lower`/`upper` directly; the GPU limits buffer holds ±∞ for them.
- IK Jacobians walk `Link.chain`, the indices of the moving links at or above a link, root first, so mimic joints add into their leader. Both devices iterate it in the same order (`robot_wgsl` unrolls it).
- Each `Link` keeps its collision shapes (`Link.shapes`, from the description or an attached object). `MeshModel` (`meshes.rs`) loads them as parry3d meshes and convex hulls and measures exact mesh clearances by branch and bound; it is a CPU post-check, not part of either device.
- `assets/franka/panda_collision.json` holds cuRobo's hand-tuned Panda spheres (Apache-2.0, credited in the README and `assets/README.md`); keep that credit when touching assets. `examples/common/mod.rs` has the Panda's options (locked fingers, `ee_links`, default pose).

**Demos.** `demos/` is a crate of its own (Rerun, urdf-rs), so the library never depends on the viewer; CI lints it in its own job. `RobotMeshes` draws any robot's URDF meshes, the parts of an assembled one included. The handover demo includes `examples/handover/rebot.rs`, so the two reBots and their scenes are written once.

## Docs

- `README.md` is for people who use the library: the pitch, install, three quick-start snippets, the comparison, links and limitations. Everything else lives in `docs/`, one page per topic (`docs/README.md` indexes them). A page owns its topic; others link to it rather than repeat it.
- Every Rust block in the README and the guides compiles as a doctest (`lib.rs`, `mod guides`; the README and `training-data.md` need `--features lerobot`). Mark them `rust,no_run` and run each once by hand when it changes.
- Numbers in the docs come from runs you can name: `docs/benchmarks.md` says how each was measured and how to reproduce it. When a change moves a number, rerun it and update every page that quotes it.

## Style

The code follows [Google's Rust style guide](https://google.github.io/styleguide/rustguide.html). Lints check what they can: `[lints]` in `Cargo.toml` and `demos/Cargo.toml` for every target, and `lib.rs`'s attributes for the library alone (`missing_docs`, `missing_debug_implementations`, `must_use_candidate`, `unwrap_used` outside tests). Decisions the guide leaves to the codebase:
- **Formatting.** rustfmt's defaults at 100 columns. Comments wrap at 100 too; the examples' hand-aligned usage blocks stay aligned. Long string literals are left as rustfmt leaves them.
- **Imports.** No globs, except `use super::*` in test modules and the rayon and fearless_simd preludes. Free functions are called through their module (`world::box_distance`, `datagen::handover`, `batchplan::plan`). Types are imported by name, except wgpu's and rerun's and names that would clash (`urdf_rs::Pose` beside `Pose`, `npy::ExportOptions` beside `lerobot::ExportOptions`, `Result`, `Error`, `Value`). A trait whose methods one function calls is imported in that function. Groups: std, external crates, then the crate.
- **Modules.** A module with children only declares and re-exports them: `rrt.rs` holds `rrt/connect.rs` (the search, its trees and samples) and `rrt/ahead.rs`, and what the helpers read of `Connect` is `pub(super)`.
- **API.** Constructors that return a `Result` are `try_new`. Public returns use named fields, not tuples (`Distance`, `Clearance`, `lower_corner`/`upper_corner`). Names are spelled out (`rotation_weight`, `acceleration_weight`, `gradient`), except terms of the field (`dof`, `ee_links`, `q`, `fx`, `dt`). `#[non_exhaustive]` marks what callers read but never build (results, `Trajectory`, `Recovery`) and the enums that may grow (`Obstacle`, `Occlusion`, `Origin`, `Error`); options, problems and value types (`Pose`, `JointState`, `PickPlace`, `Demonstration`, which exporters take from callers too) keep struct literals, under the guide's exceptions for struct update syntax and fixed structures. Options structs implement `Default`.
- **Implementation.** `expect` says why it cannot fail. Iterator chains are named before a `for` loop rather than built in its header, and sequential `for_each` is a `for` loop (rayon's parallel `for_each` stays). Assertions say why they hold. Tests never read through a poisoned mutex. Bare `bool` arguments carry a comment (`/* parallel */ true`).

## Working rules

- **A new test must fail on the unfixed code.** Prove it by planting the bug. For example, swapping `acceleration_weight`/`velocity_weight` in the GPU packing must fail `trajopt_directions_match_cpu_element_wise` and `trajopt_smoothness_gradient_matches_its_cost`.
- **Assert that test inputs exercise the code.** The smoothness test asserts its path isn't straight, because a straight path has zero smoothness gradient.
- **Test through the interface** (`Device`, `solve_ik`, `plan`, the exporters). Internal unit tests are only for internal seams, such as sphere pairs and `unmet_limits`.
- **Compare optimizer steps by direction, not position.** The line search scales each L-BFGS step, so `step_directions` in `tests/gpu.rs` scales each path's step to its largest entry: step 1 is the negative gradient, later steps the two-loop recursion. Outcomes are compared through the full trajectory cost (`trajopt_lowers_the_cost_as_far_as_the_cpu`).
- **Performance claims need an A/B comparison on one machine.**
  - Build the previous commit in a `git worktree` with its own `CARGO_TARGET_DIR`, and alternate runs with the current code.
  - Check the load first; the dev machines often run other heavy work.
  - A single run proves nothing.
- **Keep `#[inline]` on hot functions the CPU backend calls across modules** (`Robot::fk_into`, `Fk::dpoint`, `box_distance`, `sphere_distance`). Without it, how the compiler splits the crate into chunks swings CPU throughput by 15–20%. Mark new hot cross-module helpers the same way. `cpu::collision` is `#[inline(always)]`: with a second cost-only caller, LLVM stopped inlining it into the line search (−5%).
- **`unwrap`/`expect` only for invariants you can prove.** Fallible paths return errors.
- **The repo is public.** Keep hostnames, usernames and account details out of committed files.
- **Comments only for what the next reader can't quickly recover from the code.** No comments referencing conversation context. If a workaround needs a paragraph of justification, fix the code instead.
