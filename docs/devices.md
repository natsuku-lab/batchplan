# Devices

A `Device` runs batched work: forward kinematics, collision costs and gradients, IK, trajectory optimization and grid building. There are two kinds, with one interface:

- `Device::gpu(&robot)` runs WGSL compute kernels through [wgpu](https://wgpu.rs) on Vulkan, Metal or DX12.
- `Device::cpu(&robot)` runs the same math on all CPU cores; `Device::cpu_threads(&robot, n)` on `n` threads it owns (0: one per core).

The two produce the same results (the test suite compares them throughout), so code written for one runs on the other. To prefer the GPU and fall back to the CPU:

```rust,no_run
use batchplan::*;

fn main() -> Result<(), Error> {
    let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &RobotOptions::default())?;
    let device = Device::gpu(&robot).or_else(|_| Device::cpu(&robot))?;
    println!("planning on {}", device.name());
    let worlds = device.upload(&[World::default()])?;   // uploaded once, used by every call
    let _ = worlds;
    Ok(())
}
```

A device is bound to its robot. `device.with_robot(&other)` gives a device for another robot (the same robot holding an object, say) that shares the GPU context or threads and every uploaded `Worlds`. Every batch is checked before either implementation sees it (array shapes, world indices, worlds uploaded to this device), so malformed input is an `Err` on both.

## Which GPUs

The kernels use only core features: 32-bit floats, and no subgroups, float atomics or extensions. Any adapter wgpu exposes through Vulkan, Metal or DX12 should work if it allows 12 storage buffers per shader stage and 64-wide compute workgroups. Desktop drivers allow far more; browser WebGPU and some embedded GPUs do not. `Device::gpu` skips adapters that fall short; if none qualifies, its error names each adapter and what it lacks. OpenGL adapters are not used. `Device::gpu_named(&robot, "radeon")` picks an adapter by name.

| Adapter | Status |
|---|---|
| AMD Radeon 8060S, Mesa RADV (Vulkan), Ubuntu 26.04 | Verified: all tests pass |
| Apple M4 Pro (Metal), macOS | Verified: all tests pass |
| NVIDIA Tesla T4, driver 595.91 (Vulkan), Ubuntu 24.04 | Verified at an early version (14 tests); not re-run since |
| Mesa llvmpipe (Vulkan on the CPU) | Verified in CI |
| Other NVIDIA cards and Jetson, Intel, DX12 on Windows | Untested |

What runs where:
- **Planning** searches each problem on CPU threads, with SIMD collision checks, on either device: branchy, early-exit searches suit CPUs, and both devices then plan identically.
- **The GPU** wins on large, dense batches: IK seeds, trajectory optimization and distance grids.
- **For one problem at a time,** the CPU device is often as fast, because each GPU call pays a fixed setup and readback cost.

See [Benchmarks](benchmarks.md).

## Embedding in a control process

batchplan can run inside a robot's own controller process, next to a fixed-rate control loop.

- **CPU-only builds.** `default-features = false` drops the `gpu` feature and wgpu with it (CI builds that for `aarch64-unknown-linux-gnu`). `Device::gpu` then returns an error.
- **Threads you control.** `Device::cpu_threads(&robot, n)` starts `n` workers and runs every batch on them, never on rayon's global pool. Some work runs on the current rayon pool instead: loading a robot that needs sphere fitting, building mesh grids, the host's share of grid building, and timing `datagen`'s episodes. Call these inside your own pool's `install` to keep them there. `Robot::attach` covers boxes and spheres directly and fits other shapes on your pool if called inside one, otherwise on the calling thread alone.
- **Time budgets.** `PlanOptions::time_budget` returns the valid paths found within the budget ([Planning](planning.md#time-budgets-and-determinism)). Searches check it before every edge check; optimization checks it between rounds (on a GPU, between submissions of several rounds).
- **Allocation-free sampling.** `trajectory.at(t, &mut state)` writes position, velocity and acceleration without allocating.
- **Typed errors.** Every public function returns `batchplan::Error`, whose kind a host can act on: `Input` (shapes, indices, options, geometry), `Load` (a robot or scene file, with its path), `Gpu`, `Threads`, `Unsafe` (a trajectory that breaks a limit) and `Write`.

`examples/control_loop.rs` puts these together, shaped like a ros2_control controller. A planner thread with a two-thread CPU device plans reaches under a 500 ms budget, checks each trajectory, and hands it over through a latest-value slot. A 50 Hz loop takes a new trajectory when one is ready, without waiting on a lock, and samples it with `at(t)`; otherwise it holds its pose.

```bash
cargo run --release --example control_loop -- 10    # run for 10 seconds
```

## Running on another machine

`REMOTE=user@host scripts/sync.sh '<command>'` rsyncs the checkout to `~/batchplan` on another machine (a GPU box, say) and runs the command there, keeping its build cache and data. Set `SSH_OPTS` for extra ssh flags.
