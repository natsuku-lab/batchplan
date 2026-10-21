# Benchmarks

The numbers on this page come from three machines:

- an **Apple M4 Pro**: 14 CPU cores and a 20-core GPU, on Metal;
- an **AMD Ryzen AI Max+ 395** CPU (16 cores, 32 threads) with a **Radeon 8060S** GPU, on Vulkan through Mesa's RADV driver, under Ubuntu 26.04;
- an **NVIDIA T4** GPU with 8 threads of a 2.5 GHz Xeon (an AWS g4dn.2xlarge), for cuRobo, which needs CUDA, under Ubuntu 24.04.

The load was checked before each run. Every command to reproduce them is [at the end](#reproducing).

## Against other planners

### The problems

MotionBenchMaker's 800 problems (8 sets) and MπNets' 1,800 (12 sets), all for the Franka Panda, as [robometrics](https://github.com/fishbotics/robometrics) packages them. Each planner plans from the start to the set's first IK solution: a joint-space goal, so only planning is compared.

| Planner | Robot model | How it ran |
|---|---|---|
| batchplan | cuRobo's Panda spheres, which ship with it; and VAMP's | Default `PlanOptions`. `benchmark --spheres vamp` loads VAMP's spheres onto batchplan's Panda; all 59 sphere centres match `vamp.panda.fk` within 0.4 µm, and self-collision verdicts agree, on 300 random configurations. Each set is one `plan` call on the CPU device. Time per problem is one problem per call, for every problem. |
| [VAMP](https://github.com/KavrakiLab/vamp) 0.6.4 | VAMP's Panda spheres | RRT-Connect, then VAMP's simplification, with the Halton sampler and its SIMD collision checker, called from C++ as its Python bindings call them (`scripts/cpp_planners.cc`, built with VAMP's release flags). RRT-Connect steps 1 rad, VAMP's Panda range (`ROBOT_RRT_RANGES`, which its benchmark script uses), or 2 rad, `RRTCSettings`' default, and stops at VAMP's 100,000 samples or at batchplan's limit, 10,000. Each problem runs on one core, as many in parallel as there are cores. |
| [OMPL](https://ompl.kavrakilab.org) 2.0.2 | VAMP's Panda spheres | `RRTConnect` with a 5 s limit, then `simplifySolution` for 1 s, from C++. VAMP's check of one configuration is the state validity checker; edges are checked by OMPL's own motion validator. Each problem runs on one core. |
| [cuRobo](https://github.com/NVlabs/curobo) 0.8.0 | cuRobo's Panda spheres | `MotionPlanner.plan_cspace` with its defaults (5 attempts, graph seeding from the second, a 1 cm collision activation distance) and CUDA graphs, one problem at a time; and `BatchMotionPlanner`, 32 problems per call, each in its own world. On the T4, compared on the 1,300 problems whose scenes are only boxes (below). |
| [RoboPlan](https://github.com/open-planning/roboplan) 0.7.0 | The Panda's meshes | RRT-Connect with its default settings and a 5 s limit, then its path shortcutter. It checks the meshes against exact boxes and cylinders with Coal. Groups and allowed collisions come from MoveIt's Panda SRDF. Each problem runs on one core. |
| [MoveIt 2](https://moveit.ai) (Jazzy) | The Panda's meshes | `moveit_py` with the default OMPL pipeline (RRT-Connect, simplification, time parameterization and validation) and a 5 s limit. It checks the meshes with FCL, configured by `moveit_resources_panda_moveit_config`. Each problem runs on one core, in the official `moveit/moveit2:jazzy-release` image. |

Every planner gets the same scenes, as close as it can take them:
- VAMP and OMPL get cylinders as capsules, as VAMP's own MotionBenchMaker loader does, and boxes in the `box` sets. The others get exact boxes and cylinders.
- Every scene's floor or table top is level with the Panda's base. batchplan, RoboPlan, MoveIt 2 and the judge never check the base against the scene. VAMP does, and its base sphere reaches 3 cm below the base, so the boxes under it get a hole around it, a centimetre wider than the sphere, where no other link can reach. Everywhere else, VAMP plans in the scene the judge checks. Cutting a box into pieces costs VAMP about 9% of its planning time: every problem planned with the same geometry as one box and as pieces took 0.91 and 1.00 ms on average.

### The checks

One checker, `benchmark --check`, judges every planner's paths, batchplan's included (`benchmark --save` writes batchplan's in the other planners' format, its splines sampled at 32 points per span). A path solves its problem when:

- the hand ends within 1 cm of the goal;
- every joint stays within its limits;
- the path is collision-free, sampled at most 0.005 rad apart.

Collision is checked twice:

- on the spheres the planners plan on: "solved" below. That is VAMP's spheres for batchplan, VAMP and OMPL, and cuRobo's for batchplan and cuRobo;
- on the Panda's collision meshes, with [`MeshModel`](robots.md#checking-against-meshes), for every planner: "clear on the meshes". Self-collision counts between the link pairs MoveIt's Panda SRDF leaves enabled. (batchplan's own `--meshes` check, in the device table below, uses its model's pairs, so its numbers differ slightly.)

The meshes are the neutral judge: three of the planners plan on spheres, and RoboPlan and MoveIt 2 check their meshes only at intervals along each edge. Path lengths are medians over the problems every planner in a table solves.

### Results

**On the same robot model.** batchplan, VAMP and OMPL plan on VAMP's Panda spheres and are judged on them, on the M4 Pro. *One core* plans every problem in turn. *In batches* counts each set's problems over the time the whole set took on all 14 cores: batchplan's set in one `plan` call, VAMP's and OMPL's one problem per thread, each thread taking the next. batchplan is built without flags and picks the CPU's widest SIMD instructions at run time; VAMP is built with its release flags for the machine (`-mcpu=native`). Times are medians of two or three alternating rounds.

| Planner | Solved | Clear on the meshes | Path length (median, 1,526 problems all seven solve) | Time per problem (median / p98) | All 2,600, one core | In batches |
|---|---|---|---|---|---|---|
| batchplan | **90.4%** | **85.2%** | 3.40 rad | 0.17 / 1.65 ms | 0.88 s | 20,287 /s |
| batchplan without clearance (`--clearance 0`) | 88.2% | 70.8% | **3.37 rad** | **0.15** / **0.88** ms | **0.65 s** | **26,874 /s** |
| VAMP, 1 rad steps | 86.1% | 71.6% | 3.53 rad | 0.26 / 2.12 ms | 2.43 s | 2,704 /s |
| VAMP, 1 rad steps, stopping at 10,000 samples | 86.1% | 71.5% | 3.53 rad | 0.26 / 1.76 ms | 1.05 s | 15,396 /s |
| VAMP, 2 rad steps | 86.4% | 75.3% | 3.53 rad | 0.27 / 3.23 ms | 2.71 s | 2,705 /s |
| VAMP, 2 rad steps, stopping at 10,000 samples | 86.1% | 75.0% | 3.53 rad | 0.27 / 2.22 ms | 1.20 s | 14,078 /s |
| OMPL | 63.9% | 63.5% | 3.99 rad | 2.75 / 5,049 ms | 631 s | 30 /s |

batchplan's and VAMP's rows are three alternating rounds of one session; OMPL's are from an earlier one. From Python, with 2 rad steps, VAMP's bindings plan the same paths in 0.31 ms (median) and 2,409 problems/s in worker processes. OMPL, calling VAMP's checker back from Python, takes 3.34 ms and plans 30 problems/s. OMPL's totals are set by its 5 s limit, which about 120 problems run out.

**Against MoveIt 2 and RoboPlan**, which plan on the Panda's meshes, on the Ryzen. batchplan plans on cuRobo's spheres, which ship with it; every path is judged on the meshes.

| Planner | Solved, on the meshes | Path length (median, 1,334 problems all three solve) | Time per problem (median) | In batches |
|---|---|---|---|---|
| batchplan | **93.1%** | **3.23 rad** | **0.13 ms** | **21,796 /s** |
| RoboPlan | 73.8% | 3.78 rad | 10.3 ms | 174 /s |
| MoveIt 2 | 63.1% | 3.80 rad | 59.4 ms | 82 /s |

On the M4 Pro, batchplan plans in 0.16 ms and 19,451 problems/s, and RoboPlan in 6.28 ms and 176/s, with as many of its paths clear as on the Ryzen. MoveIt 2 ran only on the Ryzen, in Docker. VAMP built with GCC 15 crashed on the Ryzen, so VAMP and OMPL ran only on the M4 Pro.

**Against cuRobo**, both on cuRobo's spheres, cuRobo on the T4 machine. That run loaded every obstacle as an oriented box, as cuRobo's own benchmark does. A cylinder loaded as a box closes the gap between the fingers at the goals that grasp it, so cuRobo is compared on the 1,300 problems whose scenes have no cylinders: `cage_panda` and the `cubby`, `dresser` and `merged_cubby` sets. (`scripts/compare_planners.py` now loads cylinders as meshes, with `get_collision_check_world`, so a run can cover all 2,600; boxes load as before.) *In batches* is each set over its wall time: batchplan's set in one call, cuRobo's in calls of 32.

| Planner | Solved | Clear on the meshes | Path length (median, 1,092 problems all six solve) | Time per problem (median) | In batches |
|---|---|---|---|---|---|
| batchplan, CPU device | **98.9%** | 92.8% | 3.243 rad | **0.22 ms**¹ | **13,384 /s**¹ |
| batchplan, GPU device | **98.9%** | 92.8% | 3.243 rad | 0.24 ms¹ | 13,038 /s¹ |
| batchplan, optimizing 2 seeds (`--optimize 2`) | 98.8% | **92.9%** | **3.137 rad** | 16.8 ms¹ | 138 /s¹, 769 /s on the GPU |
| batchplan, optimizing 4 seeds (`--optimize 4`) | 98.8% | 92.8% | **3.137 rad** | 29.1 ms¹ | 69 /s¹, 463 /s on the GPU |
| cuRobo, one at a time | 85.2% | 81.6% | 3.144 rad | 197 ms | 3.0 /s |
| cuRobo, 32 problems per call | 85.0% | 81.9% | 3.154 rad | 660 ms | 1.4 /s |

¹ On the M4 Pro, its 14 CPU threads or its GPU: the T4 machine has not been available since batchplan's previous version, which planned there in 0.64 ms and 2,152 problems/s on 8 threads. Over the problems every planner solves clear of the meshes (1,026), the optimized paths are 3.124 rad, cuRobo's 3.129 and batchplan's without optimization 3.219.

batchplan plans the same paths on either device: planning runs on CPU threads, and the GPU serves IK and trajectory optimization. Every path cuRobo returned was valid on its spheres; its 192 failures are problems it did not solve in 5 attempts, 54 of them in `cage_panda`. cuRobo's own benchmark allows 100 attempts with a 2.5 mm activation distance; on the T4, each problem that fails 100 attempts takes about 45 s, so that configuration was not run. A batch call retries until every problem in it succeeds or the attempts run out, so one hard problem slows its whole batch: that is why batches ran slower than single problems here.

By their own checks, VAMP solved 98.6% (98.5% with 2 rad steps), OMPL 90.5%, RoboPlan 86.3% (86.2% on the Ryzen) and MoveIt 2 78.6%.

<details>
<summary>Per set, solved on VAMP's spheres</summary>

| Set | Problems | batchplan | VAMP, 1 rad steps | VAMP, 2 rad steps | OMPL |
|---|---|---|---|---|---|
| `bookshelf_small_panda` | 100 | 97 | 93 | 92 | 66 |
| `bookshelf_tall_panda` | 100 | 100 | 94 | 93 | 71 |
| `bookshelf_thin_panda` | 100 | 100 | 93 | 93 | 65 |
| `box_panda` | 100 | 100 | 94 | 97 | 64 |
| `box_panda_flipped` | 100 | 100 | 95 | 97 | 70 |
| `cage_panda` | 100 | 99 | 94 | 87 | 46 |
| `table_pick_panda` | 100 | 94 | 91 | 92 | 79 |
| `table_under_pick_panda` | 100 | 94 | 87 | 85 | 36 |
| `cubby_neutral_goal` | 75 | 72 | 71 | 72 | 66 |
| `cubby_neutral_start` | 75 | 74 | 71 | 72 | 64 |
| `cubby_task_oriented` | 150 | 137 | 124 | 131 | 51 |
| `dresser_neutral_goal` | 150 | 141 | 136 | 135 | 106 |
| `dresser_neutral_start` | 150 | 142 | 141 | 138 | 109 |
| `dresser_task_oriented` | 300 | 291 | 261 | 266 | 143 |
| `merged_cubby_neutral_goal` | 75 | 74 | 73 | 71 | 65 |
| `merged_cubby_neutral_start` | 75 | 71 | 70 | 71 | 64 |
| `merged_cubby_task_oriented` | 150 | 136 | 127 | 131 | 94 |
| `tabletop_neutral_goal` | 150 | 119 | 118 | 116 | 113 |
| `tabletop_neutral_start` | 150 | 114 | 114 | 114 | 109 |
| `tabletop_task_oriented` | 300 | 196 | 192 | 193 | 180 |
</details>

<details>
<summary>Per set, solved on the meshes: batchplan, RoboPlan and MoveIt 2 on the Ryzen</summary>

| Set | Problems | batchplan | RoboPlan | MoveIt 2 |
|---|---|---|---|---|
| `bookshelf_small_panda` | 100 | 92 | 68 | 48 |
| `bookshelf_tall_panda` | 100 | 95 | 62 | 59 |
| `bookshelf_thin_panda` | 100 | 100 | 82 | 71 |
| `box_panda` | 100 | 100 | 98 | 71 |
| `box_panda_flipped` | 100 | 100 | 89 | 48 |
| `cage_panda` | 100 | 68 | 25 | 16 |
| `table_pick_panda` | 100 | 92 | 64 | 80 |
| `table_under_pick_panda` | 100 | 94 | 59 | 26 |
| `cubby_neutral_goal` | 75 | 75 | 62 | 52 |
| `cubby_neutral_start` | 75 | 73 | 51 | 55 |
| `cubby_task_oriented` | 150 | 147 | 101 | 49 |
| `dresser_neutral_goal` | 150 | 141 | 108 | 89 |
| `dresser_neutral_start` | 150 | 137 | 105 | 104 |
| `dresser_task_oriented` | 300 | 264 | 206 | 127 |
| `merged_cubby_neutral_goal` | 75 | 75 | 63 | 58 |
| `merged_cubby_neutral_start` | 75 | 74 | 58 | 65 |
| `merged_cubby_task_oriented` | 150 | 145 | 113 | 105 |
| `tabletop_neutral_goal` | 150 | 141 | 132 | 132 |
| `tabletop_neutral_start` | 150 | 138 | 125 | 133 |
| `tabletop_task_oriented` | 300 | 263 | 247 | 252 |
</details>

### Reading the results

- **On the same spheres, batchplan solves the most, with the shortest paths.** On VAMP's spheres, 9.3% of the problems start or end in collision, so no planner can solve more than 90.7% on them; batchplan solves 2,351 of the 2,358 others. VAMP counts 98.6% solved because it checks neither the start nor the goal, and checks edges 1/32 rad apart, where the judge's denser samples find grazes. OMPL's paths graze most often: its own motion validator checks edges at 1% of the joint space's extent, about 0.13 rad. batchplan's paths are the shortest because its searches start near the straight line ([below](#closing-the-path-length-gap)).
- **batchplan plans faster than VAMP.** Both run RRT-Connect on SIMD collision checks. VAMP searches its few unsolvable problems up to 100,000 samples, so all 2,600 take it 2.43 s on one core. Stopped at 10,000, as batchplan's searches are, VAMP is still slower on one core and in batches. A batch lasts as long as its few unsolvable problems, and batchplan's idle threads help those along ([below](#how-the-speed-came)). The slowest problems take about as long: VAMP's 98th percentile is 1.76 ms against batchplan's 1.65, which moves by ±0.3 ms with where batchplan's Halton samples start (`--rrt-seed`). In batches, batchplan plans about 125 times as many problems per second as RoboPlan and 270 times as many as MoveIt 2 (on the Ryzen), and 680 times as many as OMPL (on the M4 Pro).
- **The clearance costs a quarter of the speed, and buys the clear paths.** Without it (`--clearance 0`), batchplan plans 32% more problems per second on VAMP's spheres, but only 70.8% of its paths clear the meshes, against 85.2% with it and 71.5–75.3% for VAMP.
- **Judged on the real robot, batchplan's paths clear it most often.** On the spheres that ship with it, 93.1% of its paths clear the Panda's meshes, against 73.8% for RoboPlan and 63.1% for MoveIt 2, which plan on the meshes. Over the 1,334 problems all three solve, its paths are also the shortest: 3.23 rad against 3.78–3.80. On the scenes of boxes, it clears 92.8% against cuRobo's 81.6%. cuRobo's hand-tuned spheres are smaller around the hand than VAMP's: 98.1% of the problems have a free start and goal on them, and batchplan solves 98.0%.
- **With two optimization seeds, batchplan's paths are the shortest.** Optimizing the searched path and the straight line (`PlanOptions::seeds = 2`) gives 3.137 rad where all solve, against cuRobo's 3.144, and the shorter path on 78% of the problems. It costs 16.8 ms per problem on 14 CPU threads, or 769 problems/s in batches on the M4 Pro's GPU. cuRobo's mean stays 1% shorter (0.2% against four seeds), from a few long problems. Without optimization, batchplan's paths are 3% longer than cuRobo's (3.243 rad), from searches that stay near the straight line ([below](#closing-the-path-length-gap)), at a seventieth of the optimized time.
- **The outputs differ.** VAMP, OMPL and RoboPlan return waypoints joined by straight lines, so a controller must stop at every corner, or smooth the path and check it again. MoveIt 2 and cuRobo return timed trajectories. batchplan returns a smooth cubic B-spline that starts and ends at rest, which `Trajectory::new` times within velocity, acceleration and jerk limits in about 0.5 ms.
- **The time per problem is measured as each library is used.** batchplan's covers a whole `plan` call. VAMP's and OMPL's cover building the scene, planning and simplifying, from C++. cuRobo's covers loading the world and `plan_cspace`, with the GPU synchronized. RoboPlan's and MoveIt 2's cover their planning calls from Python.

## The 2,600 problems on every device

`examples/benchmark.rs` runs every set in two modes:

- **plan:** plan from the start to the set's first IK solution, as in the comparison above;
- **IK + plan:** solve IK for the goal pose, then plan to the solution with the most clearance.

It checks success as above, on cuRobo's spheres, which ship with batchplan. Timings are medians of three runs; the M4 Pro's GPU also drives its display. "Free ends" counts the problems whose start and given IK solution are collision-free on these spheres, which caps what planning can reach. Throughput runs each set as one batch; time per problem covers the first 20 problems of each set, one at a time.

| Device | Free ends | Plan | IK + plan | IK + plan, clear on meshes | Plan, problems/s | Plan, time per problem (median / p98) | IK + plan, problems/s | IK + plan, time per problem (mean / p98) |
|---|---|---|---|---|---|---|---|---|
| M4 Pro CPU, 14 threads | 98.1% | 98.0% | 97.0% | 89.8% | 20,766 | 0.2 / 1.9 ms | 2,504 | **1 / 3 ms** |
| M4 Pro GPU | 98.1% | 98.0% | 97.0% | 90.0% | 19,313 | 0.2 / 2.0 ms | 2,152 | 10 / 24 ms |
| Ryzen AI Max+ 395, 32 threads | 98.1% | 98.0% | 97.0% | 89.8% | **23,305** | **0.1 / 1.7 ms** | **2,894** | **1 / 3 ms** |
| Radeon 8060S | 98.1% | 98.0% | 97.0% | 89.7% | 22,134 | 0.2 / 1.7 ms | 1,885 | 11 / 37 ms |

Searching near the straight line and straightening cost 7% of batched throughput on the Ryzen and 14% on the M4 Pro. Idle threads now help the longest searches along ([below](#how-the-speed-came)), which more than pays it back. In three alternating rounds against the version before helpers, the Ryzen's batches plan 21,796 against 18,116 problems/s (22,806 against 17,869 on VAMP's spheres), with every path the same. On the M4 Pro they plan 19,898 against 15,779 in three rotating rounds; in six rotating rounds of 15 batches of each set, all the sets' batches take 122.1 ms against 154.4. Single problems plan as fast as before at the median (0.126 against 0.126 ms on the Ryzen, 0.172 against 0.173 on the M4 Pro) and within 3% at the 98th percentile (1.39 against 1.39 ms on the Ryzen, 1.48 against 1.46 on VAMP's spheres, 1.69 against 1.74 on the M4 Pro). The slowest plans in 21.2 ms against 30.9 on the Ryzen, and 26 against 46 on the M4 Pro.

Planning runs on each device's own CPU threads, so the devices plan alike. IK runs on the device, where the GPU pays a fixed cost per call (about 10 ms for one problem) that the CPU does not.

The motions stay within the Panda's limits: over the sets, median motions take 2.0–3.3 s, peak accelerations reach 4.9 rad/s² against a 5 rad/s² limit, and peak jerks 40–46 rad/s³ against 50.

Planning alone clears the meshes in 93.1% of the problems. IK + plan clears them less often: its goals are chosen on the spheres too, so some of them touch the meshes.

## GPU and CPU throughput

`examples/bench.rs` solves IK and plans for 512 random scenes, each a table with 2–6 boxes and one top-down grasp target:

- **IK:** 32 seeds × 60 iterations per target;
- **planning:** default `PlanOptions`.

Every device solves the same 484 of 512 targets and plans all 484.

| Machine | IK seeds/s, GPU / CPU | Planning, problems/s | End to end, GPU | End to end, CPU |
|---|---|---|---|---|
| Ryzen AI Max+ 395 and Radeon 8060S | 581,000–616,000 / 181,000–212,000 | 90,000–135,000 | **14,539–15,280 problems/s** | 5,098–5,973 problems/s (32 threads) |
| M4 Pro | 313,000–319,000 / 151,000–168,000 | 152,000–185,000 | **8,735–8,872 problems/s** | 4,359–4,825 problems/s (14 threads) |

Ranges cover runs after a warm-up run. The first run after the GPU idles is up to 30% slower, and the M4 Pro's GPU, which also drives its display, varies with what is on screen.

By batch size, end to end, in solved problems per second (medians of three runs):

| Problems per call | M4 Pro GPU | M4 Pro CPU | Radeon 8060S | Ryzen CPU |
|---|---|---|---|---|
| 1 | 129 | **1,658** | 125 | **1,345** |
| 8 | 949 | **3,850** | 930 | **4,879** |
| 64 | 2,771 | **4,236** | 3,614 | **5,864** |
| 512 | **8,883** | 4,705 | **15,119** | 5,784 |
| 4,096 | **10,815** | 4,713 | **18,886** | 6,041 |

A GPU call costs about 10 ms, so one problem at a time runs 11–13 times faster on the CPU. The CPU fills its cores by 64 problems. The GPU keeps gaining, overtakes the CPU between 64 and 512 problems, and at 4,096 solves 2.3 (M4 Pro) to 3.1 (Radeon) times as many problems per second. These tabletop reaches are easy, mostly straight lines: planning all 484 takes a few milliseconds, so IK sets the pace, and the GPU speeds IK up 2 to 3 times.

## Bimanual handovers

`examples/handover/` generates the [handover episodes](training-data.md#handovers) for two reBot B601-DM arms 44 cm apart at a table. In each task one arm picks up a block lying in front of it and hands it over between the arms, and the other arm sets it down in front of itself. The blocks are 24–28 cm long, in four sizes; the block, meeting and place poses vary in position and yaw, and 0–2 small boxes stand elsewhere on the table. Half the tasks go left to right. Every step is batched over all the tasks: six IK batches (the meeting pose solves both hands at once), seven planned segments, two of them moving both arms, then timing and the episodes.

| 512 tasks, M4 Pro | GPU | CPU (14 threads) |
|---|---|---|
| Handovers | 504 of 512 | 504 of 512 |
| Time | 1.35–1.37 s | 1.43–1.46 s |
| Episodes/s | 368–373 | 344–353 |

The 504 episodes hold 104 minutes of motion at 30 fps, 12.4 s each at the median. Times are three runs each, with other applications open; the previous version, alternated with these, took 1.35–1.39 s on the GPU and 1.42–1.49 s on the CPU. Four of the IK batches are seeded with the configuration before them (each grasp with the one just above it), so they keep 4 of the 32 seeds: that seed nearly always converges, and fewer seeds took the GPU run from 2.17 to 1.56 s with the same 504 handovers. The blocks come in four sizes because each size needs its own carried spheres and so plans in batches of its own: sizes drawn from a continuous range took 3.67 s.

Rendering the episodes' camera videos in MuJoCo (`scripts/replay_mujoco.py`, three 640×480 cameras) takes about 15 ms a frame on the same machine, 50 times longer than generating them.

## Spheres and meshes

Planning runs on spheres. `benchmark --meshes` checks each successful plan against the meshes with [`MeshModel`](robots.md#checking-against-meshes), at 4× the planner's validation density.

cuRobo's hand-tuned Panda spheres leave parts of its meshes uncovered, by up to 18 mm on `panda_link5`, so a path that only just clears the spheres can graze an obstacle on the meshes. That is why plans keep a [clearance](planning.md#planning-a-path) away from their ends, wider while searching than while shortcutting. On the Ryzen, one run each:

| Clearance, searching / shortcutting | Plan | Clear on the meshes | Path length (median) | Planning, problems/s |
|---|---|---|---|---|
| none | 95.5% | 56.8% | 3.43 rad | 26,991 |
| 1 cm / 1 cm | 98.0% | 92.7% | 3.46 rad | 23,405 |
| 1.5 cm / 1 cm (the default) | 98.0% | 93.1% | 3.46 rad | 22,822 |
| 2 cm / 0.75 cm | 98.0% | 92.2% | 3.46 rad | 22,852 |
| 2 cm / 1.25 cm | 98.0% | 93.5% | 3.46 rad | 23,587 |
| 2 cm / 2 cm | 98.0% | 93.9% | 3.50 rad | 23,189 |

Without clearance, paths pull taut against the spheres: they are the shortest, but the judge's denser samples find some of them touching, and 4 in 10 problems end with a path that touches the meshes. Below 1 cm in shortcuts, paths get no shorter but graze the meshes more often. On these spheres, a clearance costs 13–15% of planning throughput and its width little: the runs that keep one are within 4% of each other, all set by the longest searches, which helpers speed.

Spheres fitted to the meshes (`--spheres fitted`) cover them completely, but their 3 cm overhang puts most of these tight scenes' starts or goals in collision: 40% of the problems have free ends on them.

## Closing the path-length gap

batchplan's paths were 12% longer than cuRobo's at the median where both solved (3.53 against 3.14 rad over the box-only problems). The gap was concentrated: half the problems were within 5% of cuRobo's, and the 266 more than 30% longer held 85% of it. Those were detours: sampling the whole joint space, RRT-Connect found routes far from the straight line (1.4–2.8 rad off it at the median, against 0.4 for cuRobo), with joints swinging out and back that shortcuts between whole configurations cannot undo. cuRobo's `plan_cspace` first optimizes the straight line and falls back to a roadmap sampled in an ellipsoid around it, so its paths averaged 5.9% longer than the straight line, against batchplan's 44%. What was tried, on the box-only problems on the M4 Pro unless noted:

- **Searching near the straight line first.** RRT-Connect's first samples come from a tube around the straight line that doubles until it spans the joint ranges. A tube 0.2 rad wide doubling every 128 samples cut the mean excess over the straight line from 44% to 17% and the gap to cuRobo from 28% to 9% (in total length), for about 12% of batched throughput on VAMP's spheres. Starting at 0.4 rad, doubling every 64 samples and drawing the tube once per search (not again when the search drops its clearance) costs little: on VAMP's spheres, batches are within noise of the previous version, and the 98th percentile is lower.
- **The tube's samples on top of the budget.** Counted against `max_iterations`, the tube cost one solved problem on average over four Halton starts on VAMP's spheres; on top of it, the search over the joint ranges keeps its 10,000 samples and solves as many.
- **Searching again when no spline follows.** Two of the 2,600 problems found a path near the line that no spline or repair followed, where the previous version's path fitted. Such a problem is now searched again over the joint ranges alone, as without the tube. Over four Halton starts (`--rrt-seed 0` to 3), the current version solves, on average, every problem the previous one does and more: 2,547 against 2,546 on cuRobo's spheres, 2,352 against 2,349.75 on VAMP's.
- **Straightening by default.** On paths near the straight line, straightening each joint costs about 3% of batched throughput rather than a sixth, and shortens them about 1% (8.7% to 6.8% total gap on the box problems).
- **Optimizing the straight line.** Optimization's second seed is now the straight line, as in cuRobo's first attempt, and problems searched to within 2% of it are not optimized. The optimizer's defaults favour length (100 steps rather than 40, length weighted 10 and smoothness 5 rather than 2 and 50, a 1.2 cm margin rather than 2 cm). With the previous defaults and searches, two seeds gave 3.18 rad; with these, 3.137. Repairs keep the previous weights, which repair more stuck splines: the new ones cost one solved problem on average over four Halton starts on cuRobo's spheres.
- **Not adopted.** Tightening the spline (pulling each control point toward its neighbours' midpoint where its spans stay clear): 0.5% shorter paths on all problems for 2% of throughput; four sweeps took the box problems' gap from 6.8% to 4.9% for 17% of throughput on VAMP's spheres. Optimizing the straight line inside each search: 40 steps cost 12 ms per problem on one thread. Optimizing it for problems the search failed: two or three more solved on VAMP's spheres, for 44% of batched throughput. Ending the tube after 32 failed extensions: longer paths and fewer solved. A Halton sequence of the tube's own, so the search over the joint ranges draws the previous version's samples: a 98th percentile of 2.0 ms rather than 1.7. Without clearance (`--clearance 0`), a quarter of the gap closed, for 3% of the problems solved.

## How the speed came

**Planning, from trajectory optimization to search.** Earlier versions of `plan` optimized eight seeds per problem with L-BFGS and fell back to RRT-Connect. That took 12.8 ms per problem on the M4 Pro's GPU and 19 ms on its CPU, and 372 problems/s. Now each problem is searched first, as in VAMP, and optimization is optional: 0.2 ms and 17,986 problems/s on the same machine and problems. Measured along the way, one problem on one M4 Pro thread:

- **SIMD collision checks.** Eight configurations per call, with forward kinematics in the lanes, cut RRT-Connect alone from 3.05 to 0.68 ms (median).
- **Each problem's search runs whole on one thread.** Running RRT-Connect, simplification and the spline's check as one search left no serial phase between problems and no thread hand-offs: the median went from 0.26 ms to 0.18 ms.
- **Fewer smoothing passes.** VAMP's five took 0.36 ms against 0.26 ms for three, for the same path length.
- **Placing a link's spheres only when its bounding sphere is near an obstacle,** and comparing squared distances: 0.22 to 0.19 ms median.
- **Giving up sooner on unsolvable problems.** 10,000 samples instead of 100,000 lose one of 2,600 problems, and the mean fell from 2.2 to 0.5 ms: hopeless searches no longer run on, which matters for batches.
- **Repairing in few steps.** A problem whose every spline collides is repaired by trajectory optimization. Ten L-BFGS steps rather than forty took batched throughput on all 14 threads from about 9,200 to 10,400 problems/s, and repaired all 13 such problems rather than 12.
- **Repairing inside the search.** Each set's one to three stuck problems used to be optimized after all of its searches had ended, which cost 3.5% (Ryzen) to 5% (M4 Pro) of throughput. Repairing on the thread that searched the problem, with idle threads helping, made repair free: on the Ryzen, 9,793 problems/s became 10,128, against 10,138 with no repair at all, which solves 10 fewer problems. Both devices now repair on CPU threads, so they repair alike.
- **Reusing the searches' buffers.** RRT-Connect allocated its sample, the nearest node, the step and the edges to check anew on every iteration: about 2,100 small allocations per problem, which glibc's allocator serializes across the Ryzen's 32 threads. The driver now lends each search its thread's edge buffer (`edges::reuse`), and the searches keep their working configurations: 368 allocations per problem, the same paths, and 14,237 → 15,700 problems/s on the Ryzen (alternating runs), 4.6% more on the M4 Pro.
- **Bun's build practices, measured.** Bun, now mostly Rust, builds with one codegen unit, `panic = "abort"`, ThinLTO, profile-guided optimization and mimalloc. In CPU planning on the Ryzen, one codegen unit and abort changed nothing, fat LTO cost 5%, and PGO, trained on other scenes, gained 1%. mimalloc as the allocator gained 18% before the buffer reuse and 10% after it (3.5% on the M4 Pro), so the README suggests it for Linux programs; a library leaves the allocator to them.
- **The CPU's SIMD instructions, chosen when it runs.** The collision checks used `wide`, which picks its instructions at compile time, and Rust's x86-64 builds target SSE2 unless told otherwise. So the Ryzen planned about 10,500 problems/s by default, and 15,900 only when built for x86-64-v3 (AVX2, FMA), which every user would have had to ask for. They now use [`fearless_simd`](https://github.com/linebender/fearless_simd), which picks the CPU's widest level at run time, as Bun does for its hot loops, without `unsafe` in batchplan. A default build plans 16,200 problems/s in alternating runs, 2% more than the x86-64-v3 build of `wide`, with the same paths. Choosing the level on every check rather than once per search cost 11%. SIMD code left inside closures, which do not take on the chosen level, ran seven times slower. CPU IK, which is scalar code, is 4% slower than an x86-64-v3 build.

**Overtaking VAMP.** Against VAMP stopped at 10,000 samples, batchplan planned fewer problems per second in batches. Each set's batch lasts as long as its few unsolvable problems, which run all 10,000 samples, and each sample cost batchplan about twice what it costs VAMP. The changes below, each measured in three alternating rounds on the Ryzen with VAMP's spheres, took all 2,600 problems on one core from 1.07 s to 0.71 s, and batches from about 13,600 to 21,700 problems/s. Unless a step says otherwise, every path stayed the same.

- **Nearest neighbours eight at a time.** A k-d tree that measured one node at a time had made no difference: it touched seven cache lines per node. In the layout of nigh's batched tree, which VAMP uses, each leaf keeps its own copy of its nodes' coordinates, eight nodes to a SIMD group, and splits at 128 nodes. The unsolvable dresser problems that set each set's batch time went from 35.4 / 32.0 / 19.3 ms to 25.0 / 22.1 / 15.0, and batches from 13,265–14,007 to 15,499–16,211 problems/s.
- **Upright boxes rotated in the plane.** Most boxes turn only about the world's z axis; the lanes rotate points into them in the plane alone, as VAMP's z-aligned cuboids do: 1.01–1.02 s on one core against 1.03 s.
- **Keeping the trees when the clearance is dropped.** A search that fails to connect while keeping the clearance used to start over without it. It now goes on from the trees it grew: 0.89–0.90 s against 1.01–1.02 s, batches 16,731–17,418 against 15,494–16,429 problems/s, and 85.2% of the paths clear the meshes against 84.8%. The 128 paths that needed the search without the clearance changed.
- **Full clearance at once, away from the ends.** An edge that no end of the path comes near takes the full clearance for all its states, rather than each state working out its distance to the ends: 0.88 s against 0.89–0.90 s.
- **Bending grazed straight lines.** When the straight line to the goal collides, its colliding stretch is moved aside, 0.05 to 0.4 rad along each joint's axis, before any search. A line an obstacle only grazes keeps its route instead of becoming a search's detour: the median path went from 3.81 to 3.66 rad, in the same time (0.83–0.84 s on one core, against 0.83–0.85 s without bending).
- **Ruling boxes out by the spheres around them.** Of the 265 million link-against-box tests the benchmark makes, 1.6% find the link near the box. A test against a sphere around the box, about half the work of the exact test, now rules out 78% of them first, and each link's bounding sphere is placed once per check rather than once per pair. The hard dresser problems take 20.2–20.4 / 18.4–18.7 / 11.6–11.8 ms against 25.1–25.8 / 22.2–23.0 / 14.7–15.4, and batches plan 17,927–18,354 against 15,988–16,600 problems/s.
- **Each thread takes the next search.** Batches handed their searches to rayon's work-stealing iterator. Now each pool thread takes the next search from a shared counter, as VAMP's batches run: 18,900–20,151 problems/s against 17,771–18,741.
- **Self-collision rules spheres out against the other link first.** Two links whose bounding spheres overlap had every pair of their spheres tested, up to 72. Each sphere of one link is now first held against the other link's bounding sphere: 0.118–0.119 ms per problem against 0.152–0.156, 0.71–0.72 s on one core against 0.83–0.85 s, batches 21,382–21,949 against 19,174–20,020 problems/s.
- **Idle threads checking the longest searches ahead of them.** On the Ryzen, `dresser_neutral_goal`'s batch took 35.9 ms while its other problems were done in about 5, as one unsolvable search drew its 12,000 samples alone. A search is sequential, but on the longest ones 98.6% of the samples extend from the node they would have 32 samples earlier. So once a batch has no search left to start, its threads help the searches that have checked 2,000 samples, up to eight to a search. Each helper replays the search's trees from a log, forecasts the extension and connection of each of its next 64 samples, and checks them ([how](architecture.md#searches)); the search takes an answer only for a query equal to its own, bit for bit. Every path stays the same, and `dresser_neutral_goal`'s batch takes 17.8 ms. A search planned alone stops after checking 4,000 samples and goes on beside helpers it summons into the pool, which cuts the slowest problem's time by a third on the Ryzen and by four tenths on the M4 Pro. The comparisons below are the Ryzen's, where two builds of one version differ by up to 2% on a short set. On the M4 Pro, two builds of the same code, or two launches of one build, can differ by 5–35% on a short set, so differences of a few percent there are within noise. Four choices made it pay, and three others did not:
  - Forecasts cover both trees when one is near twice the other's size, the ratio that decides whose turn it is: one tree's forecasts matched 54% of the queries, both trees' 89%.
  - Helpers run the pool's other work while they wait, and before each sample while a repair spreads its samples over the pool, since repairs need the idle threads. Helpers that ran that work only every 16 samples left `cage_panda` 9% slower than without helpers on the Ryzen (15.3 against 14.0 ms on VAMP's spheres); it now plans as fast as without them (13.8 ms).
  - Helpers join only searches that have checked 2,000 samples, and threads with none to help sleep until one has or every search is done. Forecasts match where trees are large and grow slowly: helpers answered most queries of the unsolvable searches, which check 8,000 samples and more, and mostly fewer than one in thirty of the queries of searches that check 500 to 5,000. Beside those shorter searches they only kept every core busy as the batch drained, and on the Ryzen a search takes 1.96 times as long beside 23 others as alone, its core's second thread busy. Joining from 500 checked samples, `bookshelf_small_panda` planned 2–7% slower than without helpers; from 2,000, no set plans more than 1% slower, at a cost of 1.1 ms to `dresser_neutral_goal` (17.8 against 16.7 ms helped from 500 drawn samples).
  - A search tells its helpers only of the samples it checks. One of `bookshelf_small_panda`'s searches turned down 87% of its 12,138 samples before checking anything, at a tenth of a microsecond each. Helped, it took 1.32 times as long as alone while it reported every sample, and 1.17 times reporting only those it checks.
  - Not adopted: handing the search each forecast's nearest node as well, so it measures only the nodes added since (within noise); helpers summoned as rayon jobs by searches that run long, which took the threads repairs need (`cage_panda`'s last search repaired in 7.1 ms rather than 2.2); and sending helpers away from a search when they answer fewer than one in eight of its queries, letting them back once it has checked twice as many samples, which sped `bookshelf_small_panda` up as much but left `dresser_task_oriented` 6% slower (24.4 against 23.1 ms on VAMP's spheres), as its unsolvable searches were answered little at first.
- **Not adopted.** Random and partial shortcuts, as some planners use, shortened paths 1% for 17% more time. Straightening one joint at a time along the whole path, as HPP does, shortened them from 3.60 to 3.56 rad for a sixth less batched throughput, so at first only training data used it. On paths near the straight line it costs about 3%, and it is now on by default (`ShortcutOptions::straighten`, [above](#closing-the-path-length-gap)). Wider bends, up to 1.6 rad, shortened them to 3.59 rad but cost 77% more time. Checking each link's last-hit obstacle first cost more than it saved. Four-wide NEON lanes gained nothing over the full set on the M4 Pro. VAMP's 2 rad steps planned sooner but gave longer paths and solved fewer problems.

**GPU kernels.** GPU trajectory optimization on the Radeon went from about 1,550 to 22,800 seeds/s (15×), and on the M4 Pro from 1,220 to 7,660 (6.3×), in six steps, each measured with alternating runs of the old and new builds:

| Change | Radeon (seeds/s) | M4 Pro (seeds/s) |
|---|---|---|
| Skip link pairs whose bounding spheres are apart (self-collision) | 1,550 → 1,975 | 1,220 → 1,440 |
| Write the kernels per robot, so no private array is indexed at run time | 1,980 → 9,450 (4.8×) | 1,610 → 4,450 (2.7×) |
| Skip a link's spheres when an obstacle is beyond its bounding sphere | 9,550 → 15,000 | 4,480 → 7,025 |
| Check validity without computing gradients | +7% | +1.5% |
| Run per-path passes as cooperative workgroups | 15,900 → 22,800 | 7,130 → 7,660 |
| Give small batches several invocations per configuration | batch-1 latency 21 → 16 ms | 44 → 29 ms |

- **Per-robot kernels.** On Mesa's RADV, per-link arrays cost 3–5 KB of scratch memory per invocation; writing the kernels per robot removed it ([why](architecture.md#kernels-written-per-robot)).
- **Grids.** [Building distance grids on the GPU](perception.md#building-grids-on-the-gpu), with an integer distance transform, made a 640×480 depth image's 1 cm grid 11× faster on the Radeon (21.1–22.7 → 1.9–2.0 ms) and 3.3× faster on the M4 Pro.
- **Mesh checks.** A branch-and-bound search took mesh clearance from 620 to 110 µs per Panda configuration.
- **Not adopted.** Shader modules without bounds checks would add 6–12%, but need `unsafe`. Reusing each call's buffers would save under 1.5%: creating, filling and binding them takes 0.04–0.07 ms of a 4.5–18 ms IK call for one target on the M4 Pro, and 0.36 ms of 53 ms for 16,384 seeds. Grid buffers, which are megabytes, are reused.

## Reproducing

```bash
# GPU against CPU
cargo run --release --example bench -- 512

# Two reBot arms handing blocks over (downloads the arm at a pinned commit, with checksums), and their camera videos
scripts/fetch_rebot.sh
cargo run --release --example handover -- 512 [--cpu]
cargo run --release --features lerobot --example handover -- 16 --lerobot data/handover
.venv/bin/python scripts/replay_mujoco.py data/handover data/handover_video

# The 2,600 problems on every device (downloads robometrics' YAML and VAMP's Panda spheres at pinned commits, with checksums)
scripts/fetch_benchmark.sh
cargo run --release --example benchmark -- --meshes    # --clearance M (or --search-clearance, --shortcut-clearance), --passes N, --no-straighten, --optimize S, --rrt-seed N, --spheres curobo|vamp|fitted, --device gpu|cpu

# batchplan on its own spheres, on VAMP's, and on VAMP's without clearance, every path saved for the judge
cargo run --release --example benchmark -- --device cpu --latency 300 --save data/batchplan.json
cargo run --release --example benchmark -- --device cpu --spheres vamp --latency 300 --save data/batchplan_vamp.json
cargo run --release --example benchmark -- --device cpu --spheres vamp --clearance 0 --latency 300 --save data/batchplan_vamp_noclear.json

# VAMP and OMPL from Python, and the scenes VAMP's Python loader builds, for C++
uv venv .venv-bench --python 3.12
uv pip install --python .venv-bench/bin/python vamp-planner==0.6.4 ompl==2.0.1 pyyaml numpy
.venv-bench/bin/python scripts/compare_planners.py data/robometrics data/compare.json vamp ompl
.venv-bench/bin/python scripts/compare_planners.py data/robometrics data/vamp_problems.txt vamp-export

# VAMP and OMPL from C++: VAMP 0.6.4's source with its dependencies (Eigen 3.4, nigh, pdqsort), and OMPL 2.0
# (`brew install ompl` on macOS); the build line is at the top of the file
./cpp_planners data/vamp_problems.txt data/compare_vamp_cpp.json vamp
./cpp_planners data/vamp_problems.txt data/compare_vamp_cpp10k.json vamp 10000
./cpp_planners data/vamp_problems.txt data/compare_vamp_cpp_range2.json vamp 0 2    # 2 rad steps, 100,000 samples
./cpp_planners data/vamp_problems.txt data/compare_vamp_cpp10k_range2.json vamp 10000 2
./cpp_planners data/vamp_problems.txt data/compare_ompl_cpp.json ompl

# RoboPlan
uv venv .venv-roboplan --python 3.12
uv pip install --python .venv-roboplan/bin/python roboplan==0.7.0 pyyaml numpy
.venv-roboplan/bin/python scripts/compare_planners.py data/robometrics data/compare_roboplan.json roboplan

# MoveIt 2, in the official image
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/batchplan" -w /batchplan moveit/moveit2:jazzy-release \
  bash -c 'source /opt/ros/jazzy/setup.bash && python3 scripts/compare_planners.py data/robometrics data/compare_moveit.json moveit'

# cuRobo, on an NVIDIA GPU with CUDA 12, and batchplan on the same machine
git clone --depth 1 --branch v0.8.0 https://github.com/NVlabs/curobo ../curobo
python3 -m venv .venv-curobo && .venv-curobo/bin/pip install -e "../curobo[cu12-torch]" pyyaml numpy
.venv-curobo/bin/python scripts/compare_planners.py data/robometrics data/compare_curobo.json curobo
.venv-curobo/bin/python scripts/compare_planners.py data/robometrics data/compare_curobo_batch.json curobo-batch
cargo run --release --example benchmark -- --device gpu --latency 300 --save data/batchplan_gpu.json
cargo run --release --example benchmark -- --device cpu --latency 300 --save data/batchplan_cpu.json

# The 1,300 problems whose scenes are only boxes, and batchplan optimizing two and four seeds on them
.venv-bench/bin/python - <<'EOF'
import pathlib, re, yaml
out = pathlib.Path("data/robometrics_boxes")
out.mkdir(exist_ok=True)
for name in ["mb_set.yaml", "mpinets_set.yaml"]:
    sets = yaml.safe_load(open(f"data/robometrics/{name}"))
    yaml.safe_dump({k: v for k, v in sets.items() if re.match("cage|cubby|dresser|merged_cubby", k)}, open(out / name, "w"))
EOF
cargo run --release --example benchmark -- data/robometrics_boxes --device cpu --optimize 2 --latency 300 --save data/batchplan_seeds2.json
cargo run --release --example benchmark -- data/robometrics_boxes --device cpu --optimize 4 --latency 300 --save data/batchplan_seeds4.json

# One judge for every planner's paths: on the same spheres, against the planners on the meshes (paths from one machine), and against cuRobo
cargo run --release --example benchmark -- --spheres vamp --check data/batchplan_vamp.json --check data/batchplan_vamp_noclear.json \
  --check data/compare_vamp_cpp.json --check data/compare_vamp_cpp10k.json --check data/compare_vamp_cpp_range2.json \
  --check data/compare_vamp_cpp10k_range2.json --check data/compare_ompl_cpp.json
cargo run --release --example benchmark -- --check data/batchplan.json --check data/compare_roboplan.json --check data/compare_moveit.json
cargo run --release --example benchmark -- data/robometrics_boxes --check data/batchplan_cpu.json --check data/batchplan_gpu.json \
  --check data/batchplan_seeds2.json --check data/batchplan_seeds4.json --check data/compare_curobo.json --check data/compare_curobo_batch.json
```
