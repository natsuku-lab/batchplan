---
name: max-speed
description: Make batchplan faster, or prove a change costs no speed. Use for IK, planning, trajectory optimization or kernel work on the GPU or CPU - latency, throughput, where the time goes, A/B runs, what a check costs.
---

# Max speed

Speed here is found by measuring and kept by proving no class got slower. CLAUDE.md's working rules say how an A/B is run; this adds what they leave out.

## The loop

1. **Split the time.** For the batch size in question, name the largest term: host work around the call (seeding, uploads, copies, assembling results), fixed cost per GPU call, serial steps, or total GPU work. `cargo run --release --example latency [-- --cpu]` times one call by kind. Done when you can say which term the change moves and what its floor is.
2. **Look at what else is running** before timing, your own stray processes included (`ps -Ao pcpu,comm -r | head` on macOS, `ps -eo pcpu,comm --sort=-pcpu | head` on Linux). A difference counts once it exceeds the spread between rounds. On a machine that stays busy, compare the best and the lower quartile of many alternated rounds, and the instructions a one-thread probe retires, which load moves far less than time (`/usr/bin/time -l` on macOS, which counts the kernel's work too; `perf stat -e instructions:u` on Linux, with `kernel.perf_event_paranoid` at 2 or below); where the outputs are bit-identical and the path's source is unchanged, a moved count is how the crate compiled, not work added, and the times and `nm -n` say whether it costs.
3. **Attribute by switching one thing.** To say what a check or a transformation costs, build one scratch binary that toggles only that. A comparison against an older commit blames the wrong change. After putting a toggled or planted file back, `touch` it: cargo compares modification times, so a file restored with `mv` from a copy older than the build leaves the toggled build in place, and the next run measures the toggle again.
4. **Every class.** 1, 10, 100 and 1,000+ items; with collision spheres and without; IK with the push and without; hands that share joints; a robot of several hundred spheres; another robot; plans past obstacles. Metal, RADV and NVIDIA disagree, so a kernel change is measured on each. A class that comes out slower goes to the user with its number before anything merges.
5. **Outputs.** Checksum them before and after, and say what moves. A change to the kernels alone leaves the CPU's bit-identical.
6. **A name for every number.** A fixture goes into an example before the docs quote it.

## What decides speed

- **Small batches are serial.** A batch that does not fill the GPU takes as long as its steps one after another: fewer items save nothing; fewer steps, cheaper steps and fewer passes do. One submission and kept buffers for a small call.
- **Large batches are total work.** Leave out items that are done: seeds past the screening, paths whose optimization is over. Where the CPU twin stops an item early, the kernels skip it and the results stay the same.
- **Host work is the floor of a large GPU batch.** Read the mapped buffer once, into where the data goes; build large buffers around their data. A contended host doubles large-batch times.
- **naga counts every loop round**, and only `unsafe` turns that off. Loops of a known count are written out when the kernels are built (`loops_written_out`); fixed algebra is generated per robot (`least_squares_wgsl`). Loops of run-time counts stay and pay: on an M4 Pro, 13-20% for hands that share joints and 4-7% for optimized plans.
- **A piece made faster alone can slow the kernel it lives in.** Writing IK's algebra out made IK without spheres 2-3x faster and a 399-sphere robot's 4-5% slower. Time the whole call.
- **Private arrays indexed at run time inside a loop** do not compile under DX12's FXC (CI's Windows wheels). Index by constants.
- **Code shape moves a kernel a few percent either way**, without pattern. Thresholds and orderings are chosen by measuring each robot.
- **Code a call does not need stays out of its code.** A branch added for grids moved calls without grids by up to 10% either way on a GPU and 3% on the CPU; kernels of its own for worlds that hold a grid (`kernels_in`), and a second compiled copy of the CPU's checks (`collision_with`), left the rest as it was. Check with a dump of the generated kernel source and with the hot functions' sizes (`nm -n`), before and after: a copy inlined beside another can push its caller past what is inlined.
- **A second set of kernels is built with the first**, side by side, when a robot's kernels are built. Built lazily it made first calls wait; built on the device's pool it held up searches; looked up per derived device it slowed every call that holds an object.
- **Pipelines compile on the thread that asks.** Ask for each on a thread of its own (`together`): making a device went from 2.6 to 0.8 s cold. Measure compile times with the driver's shader cache defeated (change a constant in the source per run).
- **The generated code must not depend on which object is held**, or picking up another compiles shaders inside a call.
- **Lanes of one configuration share serial work.** They split the collision gradient; one takes the step and the others copy the configuration.

## Dead ends

- Constant loop bounds with the loops kept: faster on Metal, several times slower on an RTX 4090.
- A link's spheres four to a round, or their first rounds peeled: nothing. Obstacles in the outer loop: slower. Two to a round helps only a robot's own links of 24 spheres or more, or 128 sphere pairs.
- Clearances at the end of the IK pass: slower.
- Breaking out of single-lane kernels for seeds at rest: nothing.
- Small GPU batches sent to the CPU: breaks "results do not depend on the batch".

The changes that were kept, with their numbers: `docs/benchmarks.md#how-the-speed-came`.
