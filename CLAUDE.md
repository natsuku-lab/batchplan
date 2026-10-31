# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

batchplan is a Rust library for batched, collision-aware robot motion generation: IK and trajectory optimization over thousands of seeds, problems and worlds at once. It runs on any GPU wgpu supports (Vulkan, Metal, DX12) without CUDA or ROCm, and its CPU implementation produces the same numbers. A primary purpose is generating robot-learning training data: nominal demonstrations plus *recoveries* from perturbed states, exported as `.npy` arrays or LeRobot v3.0 datasets.

## Commands

```bash
cargo build --release --all-targets [--features lerobot] [--no-default-features]   # without `gpu`: CPU only, no wgpu
BATCHPLAN_REQUIRE_GPU=1 cargo test --release                    # every test, the guides' code included; without the env var, GPU tests skip silently when no adapter exists
BATCHPLAN_REQUIRE_GPU=1 cargo test --release --features lerobot # + the export tests and the README's and training-data guide's code
BATCHPLAN_REQUIRE_GPU=1 cargo test --release --features usd     # + the OpenUSD tests
cargo test --release --test gpu trajopt_directions_match_cpu_element_wise   # one test of one file in tests/
cargo fmt --check && cargo +nightly fmt --check -- --config imports_granularity=Module,hex_literal_case=Lower,format_macro_matchers=true   # 100 columns; the second checks the style guide's nightly-only options (CI runs both)
cargo clippy --release --all-targets [--features lerobot,usd | --no-default-features]   # zero warnings everywhere; CI denies them (lints in Cargo.toml)
cargo build --release --lib --target aarch64-unknown-linux-gnu --no-default-features     # the CPU-only robot build CI checks
cargo doc --no-deps --features lerobot,usd                      # keep at zero warnings
scripts/check.sh                                                # every lint CI runs (rustfmt stable and nightly, clippy in each feature set, docs), on the library and demos/, warnings denied; run before every push
git tag -a v0.1.1 -m "batchplan 0.1.1" && git push origin v0.1.1   # a release: bump Cargo.toml's version, wait for CI on main, tag; release.yml publishes to crates.io
cargo run --release --example bench -- 512                      # GPU vs CPU throughput; BENCH_LLVMPIPE=1 adds the WGSL kernels on Mesa's CPU Vulkan driver
scripts/fetch_benchmark.sh && cargo run --release --example benchmark   # MotionBenchMaker + MπNets (2,600 Panda problems), GPU and CPU
cargo run --release --example benchmark -- --spheres vamp --check <paths.json>...   # the earlier comparison's judge (`--save` writes batchplan's; scripts/compare_planners.py and scripts/cpp_planners.cc the others')
scripts/run_suite.sh data/bench/final final && scripts/report_suite.sh data/bench/final   # the comparison with other planners on one spherized Panda and one judge (docs/benchmarks.md#reproducing; `compare mesh` adds the mesh check)
scripts/fetch_rebot.sh && cargo run --release --example handover -- 512 [--cpu]   # two reBot arms handing blocks over: the bimanual benchmark
cargo run --release --features lerobot --example handover -- 16 --lerobot data/handover && .venv/bin/python scripts/replay_mujoco.py data/handover data/handover_video   # + MuJoCo camera videos
cargo run --release --example handover -- 512 --resets data/resets   # a reset bank: the episodes and states perturbed from them, as .npy
cargo run --release --example datagen -- data/demo 512 20       # .npy dataset: <out_dir> [worlds] [fps]
cargo run --release --example tool_paths                        # straight moves and pours: one at a time on one thread, and 64 at once
cargo run --release --example tasks -- 64                       # picks and places on random tabletops: how many get through, where the rest stop, per batch size
cargo run --release --example join [-- so101]                  # pick-like motions joined without stopping (one reBot, or the SO-101): how much sooner, per blend radius
cargo run --release --example timing                            # timing's cost and the trajectories' durations, by kind of path, on one thread
cargo run --release --example actuators -- so101 data/actuators/so101 && .venv-actuators/bin/python scripts/actuators.py servo data/actuators/so101   # tracking on actuator models (docs/benchmarks.md#tracking-on-actuator-models)
cargo run --release --example depth [-- --cpu]                  # a distance grid from a rendered depth image, then plans around it
cargo run --release --example control_loop -- [seconds]         # a planner thread feeding a 50 Hz loop
cargo run --release --features lerobot --example datagen -- --lerobot data/lerobot_demo 512 20   # --pick-place for pick-and-place, --file-mb N to split files sooner
REMOTE=user@host SSH_OPTS='...' scripts/sync.sh '<command>'     # rsync to ~/batchplan on a GPU box and run there
(cd demos && cargo clippy --release --all-targets && cargo run --release --bin handover -- 16 --save handover.rrd)   # Rerun demos: handover, pick, tasks, perception, replan (demos/README.md)
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
- **Typed errors at the surface.** Public functions return `batchplan::Error` (`error.rs`). A host matches on its kinds, so choose the kind deliberately (`ensure_input!`/`input!` for bad input), and name what was rejected and the value given. The format loaders and robot building keep `anyhow` internally for context chains; `Robot::load`, `World::load` and `CollisionModel::load` turn them into `Error::Load` with the path.
- **Embeddable.** Each device owns its rayon pool (`Device::cpu_threads`; the GPU device's has one thread per core).
  - Every parallel loop in `cpu.rs` and `gpu.rs` runs inside `self.pool.install` or on `cpu::run_searches`' broadcast, never on the global pool. `tests/threads.rs` counts OS threads; `tests/budget.rs` checks that a busy global pool does not hold up GPU plans.
  - Batches too small to be worth handing over (`SMALL_BATCH` clearances, a single search) run on the calling thread.
  - `Robot::attach` covers boxes and spheres directly (`spheres::cover_primitives`, microseconds) and fits other shapes on the caller's pool, or on the calling thread with plain loops (`spheres::fit(.., parallel: false)`), not on a local pool: one built with `use_current_thread()` leaves the thread registered with rayon for good, and every parallel loop it starts afterwards runs on that one thread.
  - `PlanOptions::time_budget` becomes a deadline passed to `Backend::search` (checked before every edge check) and `Backend::trajopt`. Without one, nothing waits or checks the clock, and results are deterministic.
- **USD stays optional.** The `openusd` crates are pinned exactly (`=0.7.0`, pre-1.0) and only built with feature `usd`.
- **Standalone, with optional bridges.**
  - The core has no middleware and reads no environment variables; only tests and examples do (`BATCHPLAN_REQUIRE_GPU`, `BENCH_LLVMPIPE`).
  - LeRobot export is behind the `lerobot` cargo feature, so the Arrow/Parquet dependencies stay optional.

## Invariants when editing

How the pieces work is in [`docs/architecture.md`](docs/architecture.md) and each module's doc comment. These rules guard what is easy to break:

- **The twin stays a twin.** The GPU kernels (`kernels.wgsl`, `grid.wgsl`, the per-robot code `robot_wgsl` writes) and `cpu.rs` implement the same math function by function, and `cpu.rs` keeps index loops so the two read side by side. Any change to the math, or to what collides, lands on both sides in the same change, with parity tests (`tests/gpu.rs`, `tests/device.rs`). `cpu::collision` stays the scalar twin the SIMD lanes are held to (`lanes_match_the_scalar_twin_on_every_robot`). Grid building is a twin too (`grids.wgsl` / `sdf.rs`).
- **Shared structs are declared once** with `shader_struct!` in `gpu.rs`, and their WGSL is generated from it; WGSL structs pad to 16 bytes, so pad the Rust side to match. A kernel parameter goes in the right struct, is filled in `GpuBackend`, read as `P.<field>`, and mirrored in `cpu.rs`.
- **No run-time-indexed private arrays in collision kernels.** Mesa's RADV spills them to scratch (writing the code per robot made GPU planning 4.8× faster). Check with `MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaderstats cargo run --release --example bench -- 8`. Lane kernels have no early returns, since lanes meet at barriers. Pipelines skip workgroup zero-initialization, so every kernel writes its workgroup memory before reading it.
- **No fixed size limits on robots.** Per-configuration state lives in buffers sized to the robot and reused per thread; hot loops neither allocate nor clear.
- **Searches allocate nothing per query** (they reuse the buffer their thread lends them) and are built on the thread that runs them (`edges::Deferred`): glibc's allocator serializes a batch's threads otherwise.
- **SIMD dispatch happens once** per search or run of eight configurations, and everything between the dispatch and the vector operations is `#[inline(always)]` code without closures: a closure loses the dispatched level and runs about seven times slower.
- **Results do not depend on the batch or the device.** Searches run on CPU threads on both devices with the same checks, and without a time budget nothing reads the clock.
- **Continuous joints** are checked and clamped with `Robot::bounds` / `Robot::within`, never `lower` / `upper`.
- **Timing does not depend on how finely a path is split.** A path split at its spans' middles (`spline::split`) is the same curve and must time within 3% of it (`a_split_path_takes_as_long`): `join` splits paths to space them alike, and tool paths are dense.

## Docs

- `README.md` is for people who use the library: the pitch, install, three quick-start snippets, the comparison, links and limitations. Everything else lives in `docs/`, one page per topic (`docs/README.md` indexes them). A page owns its topic; others link to it rather than repeat it.
- Every Rust block in the README and the guides compiles as a doctest (`lib.rs`, `mod guides`; the README and `training-data.md` need `--features lerobot`). Mark them `rust,no_run` and run each once by hand when it changes.
- Numbers in the docs come from runs you can name: `docs/benchmarks.md` says how each was measured and how to reproduce it. When a change moves a number, rerun it and update every page that quotes it.
- This file holds lasting rules, not a log of changes: how the code works belongs in `docs/architecture.md` and the modules' doc comments. No page quotes test counts or lists the test files, which every new test would change.

## Style

The code follows [Google's Rust style guide](https://google.github.io/styleguide/rustguide.html). Lints check what they can: `[lints]` in `Cargo.toml` and `demos/Cargo.toml` for every target, and `lib.rs`'s attributes for the library alone (`missing_docs`, `missing_debug_implementations`, `unreachable_pub`, `unsafe_code`, `must_use_candidate`, `unwrap_used` and `clippy.toml`'s ban on reading environment variables outside tests). Bun's rules that suit a library are lints too: no `todo!`, `unimplemented!` or `dbg!`, lazy defaults (`or_fun_call`), `clone_from` over assigning a clone, and a reason on every `allow`, as `expect` where the lint must keep firing. Decisions the guide leaves to the codebase:
- **Formatting.** rustfmt's defaults at 100 columns. Comments wrap at 100 too; the examples' hand-aligned usage blocks stay aligned. Long string literals are left as rustfmt leaves them.
- **Imports.** No globs, except `use super::*` in test modules and the rayon and fearless_simd preludes. Free functions are called through their module (`world::box_distance`, `datagen::handover`, `batchplan::plan`). Types are imported by name, except wgpu's and rerun's and names that would clash (`urdf_rs::Pose` beside `Pose`, `npy::ExportOptions` beside `lerobot::ExportOptions`, `Result`, `Error`, `Value`). A trait whose methods one function calls is imported in that function. Groups: std, external crates, then the crate.
- **Modules.** A module with children only declares and re-exports them: `rrt.rs` holds `rrt/connect.rs` (the search, its trees and samples) and `rrt/ahead.rs`, and what the helpers read of `Connect` is `pub(super)`.
- **API.** Constructors that return a `Result` are `try_new`. Public returns use named fields, not tuples (`Distance`, `Clearance`, `lower_corner`/`upper_corner`). Names are spelled out (`rotation_weight`, `acceleration_weight`, `gradient`), except terms of the field (`dof`, `ee_links`, `q`, `fx`, `dt`). `#[non_exhaustive]` marks what callers read but never build (results, `Trajectory`, `Recovery`) and the enums that may grow (`Obstacle`, `Occlusion`, `Origin`, `Error`); options, problems and value types (`Pose`, `JointState`, `PickPlace`, `Demonstration`, which exporters take from callers too) keep struct literals, under the guide's exceptions for struct update syntax and fixed structures. Options structs implement `Default`.
- **Implementation.** `expect` says why it cannot fail. Iterator chains are named before a `for` loop rather than built in its header, and sequential `for_each` is a `for` loop (rayon's parallel `for_each` stays). Assertions say why they hold. Tests never read through a poisoned mutex. Bare `bool` arguments carry a comment (`/* parallel */ true`).

## Working rules

- **Run `scripts/check.sh` and the reviewer before every push.** The script runs every lint CI runs, on the library and on `demos/`, which `cargo fmt` at the root does not reach. The reviewer agent (`.claude/agents/reviewer.md`) checks the change against these rules and reports verified defects.
- **A new test must fail on the unfixed code.** Prove it by planting the bug. A bound loosened later names the run that needed it. For example, swapping `acceleration_weight`/`velocity_weight` in the GPU packing must fail `trajopt_directions_match_cpu_element_wise` and `trajopt_smoothness_gradient_matches_its_cost`.
- **Assert that test inputs exercise the code.** The smoothness test asserts its path isn't straight, because a straight path has zero smoothness gradient.
- **Test through the interface** (`Device`, `solve_ik`, `plan`, the exporters). Internal unit tests are only for internal seams, such as sphere pairs and `unmet_limits`.
- **Compare optimizer steps by direction, not position.** The line search scales each L-BFGS step, so `step_directions` in `tests/gpu.rs` scales each path's step to its largest entry: step 1 is the negative gradient, later steps the two-loop recursion. Outcomes are compared through the full trajectory cost (`trajopt_lowers_the_cost_as_far_as_the_cpu`).
- **Performance claims need an A/B comparison on one machine.**
  - Build the previous commit in a `git worktree` with its own `CARGO_TARGET_DIR`, and alternate runs with the current code.
  - Check the load first; the dev machines often run other heavy work.
  - A single run proves nothing.
  - Cover every class the change touches (CPU and GPU, each robot, batch sizes); none may regress.
- **Keep `#[inline]` on hot functions the CPU backend calls across modules** (`Robot::fk_into`, `Fk::dpoint`, `box_distance`, `sphere_distance`). Without it, how the compiler splits the crate into chunks swings CPU throughput by 15–20%. Mark new hot cross-module helpers the same way. `cpu::collision` is `#[inline(always)]`: with a second cost-only caller, LLVM stopped inlining it into the line search (−5%).
- **`unwrap`/`expect` only for invariants you can prove.** Fallible paths return errors.
- **The repo is public.** Keep hostnames, usernames and account details out of committed files.
- **Comments only for what the next reader can't quickly recover from the code.** No comments referencing conversation context. If a workaround needs a paragraph of justification, fix the code instead.
