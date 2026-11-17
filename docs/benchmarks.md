# Benchmarks

Measured results for planner comparisons, CPU/GPU scaling, motion timing and manipulation.
These are snapshots at the revisions named below, rather than measurements of the latest release.
No new benchmark runs accompany this documentation cleanup.

The main machines are an Apple M4 Pro (14 CPU cores, 20-core GPU, Metal) and an AMD Ryzen AI
Max+ 395 (16 cores, 32 threads, Radeon 8060S, Vulkan/RADV, Ubuntu 26.04). Load, warm-up and
thread counts are recorded per experiment. The commands in [Reproducing](#reproducing) rerun
the fixtures; use each result's revision and options when comparing with its recorded numbers.

## Against other planners

One robot model for every planner, one checker for every output, one machine, planners in rotating order. The tables cover the 262 final problems; `scripts/report.py` regenerates each of them from the saved records ([commands](#reproducing)).

### Setup

| | |
|---|---|
| Robot | The Franka Panda of VAMP's spherized URDF: 59 spheres, fingers fixed 6.5 cm out, and the 34 disabled pairs of MoveIt's Panda SRDF. Every planner loads these files (`scripts/bench_model.py export`). On 2,250 fixture configurations (random, within half a millimetre of contact, and colliding), every planner's forward kinematics agrees with the judge's within 0.3 µm, and every collision verdict agrees (`scripts/parity.py`). |
| Problems | MotionBenchMaker's 800 and MπNets' 1,800 Panda problems from [robometrics](https://github.com/fishbotics/robometrics), with joint-space starts and goals: 520 tuning, 1,818 held-out baseline, 262 final problems. The final set was rerun during development and informed fixes; it is not an untouched test set. |
| Scenes | Exact boxes and cylinders for batchplan, RoboPlan and MoveIt 2. VAMP and OMPL take no cylinders, so they get each cylinder as the capsule around it, and batchplan runs those scenes too. Boxes the base sphere touches are cut around it for every planner; the base touches nothing in any problem. |
| Judge | `scripts/validate.py`, independent of every planner. It checks the start and goal, the joint bounds, and world and self collision on the spheres, sampled at most 0.005 rad apart. For trajectories it also checks velocity, acceleration (the reported velocities' rate of change) and jerk against the Panda's datasheet limits, with 1% tolerance. |
| On the meshes | A separate practical check (`compare mesh`, with batchplan's `MeshModel`): each output against the exact scene on the Panda's collision meshes, posed as the canonical model, with the SRDF's pairs. It covers the first round of each configuration. |
| Machine | AMD Ryzen AI Max+ 395, Ubuntu 26.04, otherwise idle. *16 cores* is CPUs 0–15, one thread per core; *one core* is CPU 2. |
| Runs | 5 seeds × 3 rounds per configuration (one core: 1 seed, 1 round), 5 s per query at most. A pass is every problem once with one seed. |

| Planner | Version | Timed per query | Defaults | Matched |
|---|---|---|---|---|
| batchplan | ef619b1 | `plan` on its scene: straight line or RRT-Connect, simplification, a B-spline; for trajectories, also `Trajectory::try_new` and sampling at 100 Hz | `PlanOptions::default()` | No clearance |
| [VAMP](https://github.com/KavrakiLab/vamp) | 0.6.4, C++, `-O3 -march=native` | Building the scene, RRT-Connect (1 rad steps), simplification | 100,000 samples | 10,000 samples |
| [OMPL](https://ompl.kavrakilab.org) | 1.7.0, C++, with VAMP's state checks | `RRTConnect`, `simplifySolution` | Edges checked every 1% of the space's extent | Every 1/32 rad |
| [RoboPlan](https://github.com/open-planning/roboplan) | 0.7.0, Python, Coal | Its scene, RRT-Connect, shortcutting; for trajectories, TOPP-RA in each of its four modes | Edge steps of 0.05 rad, 100,000 nodes | 1/32 rad |
| [MoveIt 2](https://moveit.ai) | Jazzy image, `moveit_py`, FCL | The planning pipeline (OMPL's RRTConnect, simplification); for trajectories, with TOTG or Ruckig and its validation | Edges every 1% of the space's extent | Every 1/32 rad |

On 16 cores, VAMP, OMPL, RoboPlan and MoveIt 2 run 16 single-threaded workers, one pinned to each core, each taking the next query. batchplan plans one query at a time on 16 threads (*time per query*), then each pass in one `plan` call (*valid/s*). *Valid/s* counts valid outputs per second of a pass's wall time, failures included. The *relative* columns compare each row with batchplan's in the same family, on the problems both solve validly, paired by problem: time as the geometric mean of the ratios, length and duration as the mean. The reports give 95% bootstrap intervals.

### Paths

On the exact scenes, 16 cores:

| Planner | Valid | On the meshes | Time per query (median / p99) | Valid/s | Time, relative | Length, relative |
|---|---|---|---|---|---|---|
| **batchplan** | **91.2%** | **86.6%** | **0.11 / 1.5 ms** | **42,842** | 1 | 1 |
| RoboPlan | 84.7% | 68.9% | 75 / 644 ms | 72 | 757× | 1.44× |
| MoveIt 2 | 62.0% | 55.1% | 40 / 1,660 ms | 34 | 444× | 1.40× |
| batchplan, matched | 88.4% | 67.6% | **0.10 / 0.7 ms** | **51,576** | 1 | 1 |
| RoboPlan, matched | **89.1%** | **71.4%** | 85 / 676 ms | 74 | 920× | 1.48× |
| MoveIt 2, matched | 87.3% | 67.5% | 271 / 5,004 ms | 20 | 3,119× | 1.42× |

On the capsule scenes, 16 cores:

| Planner | Valid | On the meshes | Time per query (median / p99) | Valid/s | Time, relative | Length, relative |
|---|---|---|---|---|---|---|
| **batchplan** | **91.2%** | **86.7%** | **0.11 / 1.5** ms | **42,773** | 1 | 1 |
| VAMP | 87.6% | 67.3% | 0.16 / **1.5** ms | 31,517 | 1.20× | 1.11× |
| OMPL | 64.9% | 56.9% | 1.1 / 5,056 ms | 17 | 10× | 1.33× |
| **batchplan, matched** | **88.5%** | 68.0% | **0.10 / 0.7 ms** | 51,759 | 1 | 1 |
| VAMP, matched | 87.5% | 67.3% | 0.14 / 1.1 ms | **61,501** | 1.07× | 1.14× |
| OMPL, matched | 85.8% | **68.8%** | 3.3 / 5,056 ms | 22 | 35× | 1.39× |

### Trajectories

On the exact scenes, 16 cores: from the start and goal to a timed trajectory within the Panda's velocity, acceleration and jerk limits, RoboPlan in each of its TOPP-RA modes.

| Planner | Valid (within 10%) | On the meshes | Time per query (median / p99) | Valid/s | Time, relative | Duration, relative |
|---|---|---|---|---|---|---|
| **batchplan** | **91.2%** (91.2%) | **86.6%** | **0.52 / 1.8 ms** | **17,677** | 1 | 1 |
| RoboPlan, Hermite | 56.2% (64.1%) | 48.7% | 78 / 652 ms | 47 | 147× | 1.13× |
| RoboPlan, Cubic | 75.9% (79.6%) | 69.5% | 84 / 665 ms | 62 | 165× | 1.11× |
| RoboPlan, Adaptive | 81.1% (85.9%) | 73.9% | 86 / 700 ms | 65 | 167× | 1.11× |
| RoboPlan, Linear Blend | 84.0% (84.0%) | 68.3% | 82 / 670 ms | 69 | 162× | 1.17× |
| MoveIt 2, TOTG | 2.5% (63.0%) | 2.4% | 49 / 1,794 ms | 1 | 79× | 1.07×¹ |
| MoveIt 2, Ruckig | 2.5% (57.7%) | 2.2% | 52 / 1,721 ms | 1 | 76× | 1.03×¹ |

¹ Over the 26 and 22 problems MoveIt 2 solves validly. Over the problems it solves within 10% of the limits (235 and 232), its trajectories take 14% and 15% longer than batchplan's.

### One core

| Planner | Scenes | Valid | Time per query (median / p99) | Valid/s | Time, relative | Length or duration, relative |
|---|---|---|---|---|---|---|
| **batchplan** | exact | **91.2%** | **0.11 / 1.3 ms** | **4,895** | 1 | 1 |
| RoboPlan | exact | 83.6% | 37 / 536 ms | 9 | 400× | 1.47× |
| MoveIt 2 | exact | 59.9% | 39 / 1,749 ms | 6 | 468× | 1.34× |
| **batchplan** | capsules | **91.2%** | **0.11 / 1.2 ms** | **4,922** | 1 | 1 |
| VAMP | capsules | 88.2% | 0.13 / 1.4 ms | 4,062 | 1.01× | 1.13× |
| OMPL | capsules | 64.1% | 0.88 / 5,046 ms | 2 | 10× | 1.30× |
| **batchplan, matched** | capsules | **88.9%** | **0.11 / 0.6 ms** | **6,313** | 1 | 1 |
| VAMP, matched | capsules | 88.2% | 0.12 / 1.3 ms | 4,465 | 0.96× | 1.16× |
| **batchplan, trajectories** | exact | **91.2%** | **0.53 / 1.6 ms** | **1,641** | 1 | 1 |
| RoboPlan, Adaptive | exact | 82.4% | 46 / 590 ms | 8 | 90× | 1.13× duration |
| MoveIt 2, TOTG | exact | 1.9% | 44 / 1,131 ms | 0 | 50× | 0.86× duration, 5 problems |

### Repeated queries in one scene

25 queries in each of 20 scenes, 16 cores; each worker loads a scene once:

| Planner | Valid | On the meshes | Time per query (median / p99) | Valid/s | Time, relative | Length, relative |
|---|---|---|---|---|---|---|
| **batchplan** | **100%** | **99.4%** | **0.02 / 1.0 ms** | **92,666** | 1 | 1 |
| RoboPlan | 98.2% | 94.2% | 39 / 381 ms | 161 | 1,076× | 1.06× |
| MoveIt 2 | 61.6% | 60.5% | 25 / 75 ms | 352 | 724× | 1.05× |

### Reading the results

- **Endpoints.** 23 of the 262 problems start or end in collision on the model, so no planner can solve them, and none can pass 91.2% valid. batchplan solves all 3,585 queries of the other 239, and every path and trajectory it returns passes the judge. VAMP checks neither end, so it returns paths for those 23 too.
- **Speed.** batchplan answers a query in 0.11 ms, or 0.52 ms with its trajectory: 400 to 3,100 times sooner than RoboPlan and MoveIt 2 on the same spheres, and 50 to 167 times sooner for trajectories. Against VAMP, on 16 cores, its queries take 17% less time with each library's defaults and 7% less with matched settings. On one core the two tie (VAMP 1.01×, interval 0.95–1.08; matched 0.96×, interval 0.89–1.04).
- **Throughput.** One `plan` call returns 42,000–93,000 valid paths per second on 16 cores, and 4,895–6,313 on one, more than any other planner except VAMP's 16 workers with matched settings (61,501 against 51,759). A batch lasts as long as its slowest searches: batchplan's matched passes take 4.5 ms, where its queries one at a time add up to 2.4 ms per core (VAMP's take 3.7 ms for 3.4). On the 1,818 held-out problems batchplan's batches return 38,814 valid paths per second against VAMP's 4,346 (VAMP's 100,000-sample searches of unsolvable problems set its pace there), and 47,503 against 38,382 matched.
- **Quality.** batchplan's paths are the shortest: VAMP's are 11–16% longer, RoboPlan's and MoveIt 2's 34–48% longer. Its trajectories reach the goal sooner than RoboPlan's in every mode (11–17%), within the limits.
- **The clearance.** batchplan keeps 1.5 cm while searching and 1 cm while shortcutting, and 86.6% of its paths clear the meshes. Without it (matched), 67.6% do, and RoboPlan's longer paths, checked at the same resolution, are valid slightly more often (89.1% against 88.4%) and clear the meshes more often (71.4% against 67.6%).
- **Tails.** p99 counts every query, failures included. The 5 s figures are queries that ran into the limit.

### Before and after optimization

The suite first ran on the 1,818 held-out problems at a237326, and that baseline was frozen. The changes under [How the speed came](#how-the-speed-came) were then made against it on the tuning problems, and batchplan was measured again on the held-out ones. Paths are 0.2% shorter and durations the same. Six fewer of the 27,270 answers are valid (24,780 against 24,786): searches of a few problems at the edge of solvability ran out of samples in 15 queries that found a path before, and found one in 9 that did not:

| batchplan, held-out problems, 16 cores | a237326 | ef619b1 |
|---|---|---|
| Paths: time per query (median / p99), paired | 0.12 / 3.6 ms | 0.11 / 1.9 ms, 0.78× |
| Paths: valid/s | 30,867 | 37,560 |
| Paths, one core: p99 | 6.8 ms | 1.9 ms |
| Trajectories: time per query (median / p99), paired | 1.06 / 4.4 ms | 0.52 / 2.3 ms, 0.50× |
| Trajectories: valid/s | 9,018 | 16,147 |

## The 2,600 problems on every device

`examples/benchmark.rs` runs every set in two modes:

- **plan:** plan from the start to the set's first IK solution, as in the comparison above;
- **IK + plan:** plan to the goal pose: `plan` solves IK from the start and ends at the nearest collision-free solution ([pose goals](planning.md#planning-a-path)).

Success requires the final `panda_hand` position within 1 cm of the goal, a timed trajectory passing `Trajectory::check`, and independent CPU collision checks at 4× the planner's validation density. These runs use cuRobo's spheres, which ship with batchplan. The timings are medians of three runs, each started with the machine's load under 6: the M4 Pro's with the code of bf2f5b9 (its GPU also drives its display), the Ryzen's and the Radeon's with the code of 9d49051. All were taken without `--meshes`, which only the mesh column was run with: in runs with it, in another session, the Ryzen's and the Radeon's IK + plan read 3,048 and 2,510 problems/s. "Free ends" counts the problems whose start and given IK solution are collision-free on these spheres, which caps what planning can reach. Throughput runs each set as one batch; time per problem covers the first 20 problems of each set, one at a time.

| Device | Free ends | Plan | IK + plan | IK + plan, clear on meshes | Plan, problems/s | Plan, time per problem (median / p98) | IK + plan, problems/s | IK + plan, time per problem (mean / p98) |
|---|---|---|---|---|---|---|---|---|
| M4 Pro CPU, 14 threads | 98.1% | 97.9% | 98.3% | 92.1% | 19,278 | 0.2 / 2.7 ms | 2,565 | 1 / 6 ms |
| M4 Pro GPU | 98.1% | 97.9% | 98.3% | 92.2% | 22,734 | 0.2 / 1.4 ms | 3,300 | 5 / 14 ms |
| Ryzen AI Max+ 395, 32 threads | 98.1% | 97.9% | 98.3% | 92.2% | 38,158 | 0.1 / 1.1 ms | 3,344 | 1 / 2 ms |
| Radeon 8060S | 98.1% | 97.9% | 98.3% | 92.1% | 34,121 | 0.1 / 1.1 ms | 6,225 | 5 / 8 ms |

Planning runs on CPU threads on both devices; the GPU accelerates goal IK. The motions take 1.3–3.2 s at the median across sets and keep the Panda's checked limits. Mesh clearance is lower for pose goals because those goals are selected using spheres too.

## GPU and CPU throughput

`examples/bench.rs` solves IK and plans for 512 random scenes, each a table with 2–6 boxes and one top-down grasp target:

- **IK:** 64 seeds per target and 30 iterations at most, 4 of the seeds past the first 8 (the defaults);
- **planning:** default `PlanOptions`.

Every device solves the same 484 of 512 targets and plans all 484.

| Machine | IK seeds/s, GPU / CPU (64 per target, 4 of them past the first 8 steps) | Planning, problems/s | End to end, GPU | End to end, CPU |
|---|---|---|---|---|
| Ryzen AI Max+ 395 and Radeon 8060S | 2,240,000–2,630,000 / 1,027,000–1,061,000 | 263,000–362,000 | **29,857–35,078 problems/s** | 14,341–14,981 problems/s (32 threads) |
| M4 Pro | 939,000–1,564,000 / 609,000–833,000 | 148,000–156,000 | **12,679–20,032 problems/s** | 8,485–11,408 problems/s (14 threads) |

Each range covers three runs started with load under 6, at [#34](https://github.com/natsuku-lab/batchplan/pull/34). GPU warm-up and display activity affect these times; the first call after idling can be up to 30% slower.

Uploading the 512 worlds, which the example times first, takes 0.1 ms on the M4 Pro's CPU and 0.2 ms on its GPU: preparing each obstacle for the CPU's checks and, on the GPU, one buffer write. A distance grid adds its values, uploaded once however many worlds share it.

By batch size, end to end, in solved problems per second (medians of three runs in the same sessions):

| Problems per call | M4 Pro GPU | M4 Pro CPU | Radeon 8060S | Ryzen CPU |
|---|---|---|---|---|
| 1 | 83 | **2,583** | 292 | **3,145** |
| 8 | 1,263 | **4,602** | 1,959 | **6,617** |
| 64 | 3,796 | **7,140** | 8,289 | **11,755** |
| 512 | **14,676** | 10,135 | **33,196** | 14,588 |
| 4,096 | **20,622** | 10,441 | **37,740** | 14,422 |

For this workload the CPU is faster at small batches; the GPU overtakes it between 64 and 512 problems. These measurements predate the later kernel improvements, so they describe scaling at that revision, not current single-call latency.

## Tool paths

`examples/tool_paths.rs` times `follow` on the Panda from its ready pose: 64 straight 10 cm moves in random directions, and 64 pours (half a radian about a random axis while the hand moves 3 mm). A move at a time on a one-thread device is the median over the 64 of each one's median of five; a batch of 64 on every thread, the median of 200 calls after 20.

| Machine | One move, one thread | 64 moves | 64 pours |
|---|---|---|---|
| Ryzen AI Max+ 395, 32 threads (at 2c98826) | 0.042 ms | 0.27 ms | 0.20 ms |
| M4 Pro, 14 threads (at bf2f5b9, the median of three runs) | 0.035 ms | 0.37 ms | 0.24 ms |

These are end-to-end `follow` measurements, including path construction and checking.

## Constraints along the path

`examples/constraints.rs` keeps the Panda's gripper upright on reaches past a pillar: a floor, a pillar 6 cm in radius and 0.6 m tall, and a zone 0.30–0.75 m out, ±0.45 m across and 0.25–0.70 m up. The start is in a corner of the zone, by IK nearest the ready pose; each of the 53 goals is a random place in the zone at a random turn about the vertical, solved by IK from the start. The tool frame between the fingertips stays within the bound of straight down and turns freely. Each problem is planned alone on one thread (the median), then all of them in one `plan` call on every thread; the largest tilt is over every path sampled at 1 kHz once timed. Times are the median of three runs, each started with the machine's load under 6 but one of the 5° row's, at 11. On the Ryzen AI Max+ 395 (32 threads):

| Bound | Solved | Alone | One batch | Largest tilt | Length | Optimized, 4 seeds |
|---|---|---|---|---|---|---|
| 1° | 53 of 53 | 0.14 ms | 3.5 ms | 0.56° | 2.211 rad | 26 shorter, 2.149 rad |
| 2° | 53 of 53 | 0.11 ms | 3.6 ms | 1.24° | 2.194 rad | 25 shorter, 2.143 rad |
| 5° | 53 of 53 | 0.09 ms | 2.7 ms | 4.99° | 2.189 rad | 25 shorter, 2.134 rad |
| 10° | 53 of 53 | 0.09 ms | 2.1 ms | 9.89° | 2.183 rad | 27 shorter, 2.111 rad |
| 5°, inside the zone | 53 of 53 | 0.09 ms | 3.4 ms | 4.93° | 2.209 rad | 25 shorter, 2.133 rad |
| none | 53 of 53 | 0.06 ms | 0.6 ms | 48.52° | 2.129 rad | |

- **The constraint costs**, at 5°, 0.03 ms more per problem alone and 3% more length. A batch takes longer because constrained searches take no help from idle threads, and its longest search sets its time.
- **Optimization** with 4 seeds takes 129–155 ms for the batch and keeps every bound; the optimized lengths are of the shortest valid seed.
- **Two arms carrying a tray level** over a block 25 cm tall (two Pandas, the right hand pinned in the left hand's frame, the left within 5° of straight down), from below the block's top to 5 places beyond it: 5 of 5 solved, 1.40 ms per problem alone, 1.9 ms in one batch. Optimization shortens the paths by 4% (2.460 to 2.369 rad on average): the shortcuts' projected walks leave it less to take out.

## Picking and placing

`examples/tasks.rs` picks a box out of clutter on 64 random tabletops and sets it down at a free spot: a table, a box 3–7 cm wide and 4–10 cm tall, 2–5 more boxes up to 30 cm tall, and a spot 12 cm clear of them. Each `pick` tries every grasp `grasps` proposes for the box (16 or 24), each `place` both of its turns at the spot (`place_poses`). Times are the best of five calls of each size, the median of three runs, after a first call that compiles the GPU's kernels, each run started with the machine's load under 6. The M4 Pro rows are bf2f5b9's.

| Device | Pick, 1 task | 16 | 64 | Place, 1 task | 16 | Through |
|---|---|---|---|---|---|---|
| Ryzen AI Max+ 395, 32 threads | 6.8 ms | 82 ms | 323 ms | 2.4 ms | 12.1 ms | 63 of 64 picks, every place |
| Radeon 8060S | 9.6 ms | 31 ms | 107 ms | 6.7 ms | 17.2 ms | the same |
| M4 Pro, 14 threads | 7.4 ms | 92 ms | 374 ms | 2.2 ms | 12.5 ms | the same |
| M4 Pro's GPU | 10.3 ms | 48 ms | 175 ms | 5.8 ms | 19.1 ms | the same |

The failed pick has every grasp blocked or out of reach. Grasp IK dominates picking; the GPU gains from batching tasks. Place candidates are checked with their held objects, grouped by hold.

## Grasps of a surface

`examples/surface_grasps.rs` finds the grasps of what cameras see (`grasps_on`) for the Panda's hand of 22 spheres, open 8 cm, on the M4 Pro's 14 threads, the quickest of five calls (of three for the 64 bowls):

| | Points | Grasps | Time |
|---|---|---|---|
| A bowl 11 cm across with a wall 8 mm thick, as points 2 mm apart with their normals | 9,201 | 1,980, all across its wall | 5.5 ms |
| A box on a table in two 640×480 depth images | 593,920, 12,492 of them the box's | 686 | 9.7 ms to read both images, 10.1 ms to grasp |
| LIBERO's bowl, its wall 3 mm thick and flared, seen from 40 cm above in one 640×480 depth image (focal length 313 pixels), its outer wall nowhere | 307,200, 6,728 of them the bowl's | none; with a `shell` of 1.5 cm 1,132, 5.5 to 14.5 mm wide (5.5 to 11.5 mm at its rim) | 9.0 ms with the shell |
| 64 such bowls in one call | 588,864 | 126,720 | 232 ms, 3.6 ms a bowl |

Nearly all of it is the hand's check: each of a bowl's 673 pairs of patches from 16 directions, 22 spheres swept 10 cm against the points near them. Checked against every point of the scene through one coarse grid, before the points near the object were kept in cells of their own and a check stopped at its first point in the way, the bowl took 211 ms and the box 314 ms on one thread, with the same grasps to the last bit.

## Spheres for what a hand holds

`examples/held_spheres.rs` holds three round shapes with the default 64 spheres to spend and with 256 (`Held::spheres`), within the default 5 mm, on the M4 Pro: how many spheres each takes, how far they reach past it, and how long covering or fitting it took.

| | To spend | Spheres | Past its side | Past its ends | Time |
|---|---|---|---|---|---|
| A plate 13.9 cm across, 18 mm thick | 64 | 64, fitted | 8.6 mm | 10.6 mm | 27 ms |
| | 256 | 145, covered | 4.7 mm | 4.7 mm | 0.14 ms |
| A can 6.6 cm across, 12 cm tall | 64 | 64, fitted | 7.1 mm | 6.7 mm | 8.4 ms |
| | 256 | 90, fitted | 4.7 mm | 5.0 mm | 7.2 ms |
| A capsule 4 cm across, 14 cm long | 64 or 256 | 5, covered | 3.6 mm | 3.6 mm | 0.02 ms |

Fitted, as stout shapes are, a thin wide one keeps reaching past its tolerance until it has a few hundred spheres; covered in one layer, the plate is within it at 145.

## Joining motions

`examples/join.rs` [joins](planning.md#joining-motions) pick-like motions on one arm in free space: a transit planned from home or the last pre-grasp to a random pre-grasp configuration, then the hand's straight move in along its z (5 cm on the reBot, 3 cm on the SO-101); or the straight move back out, then the transit to the next pre-grasp. Each sequence is timed with its motions one after another, and joined; the Ryzen AI Max+ 395, 32 threads, three runs each started with the machine's load under 6.

| Blend radius | reBot B601-DM, 60 sequences: joined | Sooner, mean (most) | SO-101, 34 sequences: joined | Sooner, mean (most) |
|---|---|---|---|---|
| 2 cm | 29 | 1.9% (8.8%) | 29 | 4.7% (15.2%) |
| 5 cm | 52 | 5.8% (15.0%) | 33 | 8.4% (17.0%) |
| 10 cm | 45 | 5.6% (19.9%) | 29 | 7.1% (18.3%) |

No sequence finished later than running its motions separately. Joining took 15–23 ms for the
reBot batch and 7–9 ms for the SO-101 batch. Tight turns may retain their stops when a blend
cannot improve duration under the jerk limit.

## Replanning while moving

`examples/timing.rs` replans Panda motions from halfway along their paths, with nonzero joint
velocities, using `Trajectory::moving`. In three alternating Ryzen runs, the timing change in
[008bc10](https://github.com/natsuku-lab/batchplan/commit/008bc10) reduced mean duration from 12.93 s to
3.76 s on 61 replans. Timing each path on one thread cost 1,160 µs instead of 735 µs.

In a wider set of 96 replans, 85 returned runnable paths. Before the change, 75 returned paths,
some of which could not run from their starts. The batch took
59 ms on either 14 or 32 threads, against 68 and 48 ms respectively: the improvement in
runnable motions came with a regression for that 32-thread batch. Later timing improvements
are measured below.

## Timing up to the limits

[#20](https://github.com/natsuku-lab/batchplan/pull/20) improved the time map to use the full
velocity and acceleration limits while retaining jerk checks. `examples/timing.rs` and the
2,600-problem benchmark compared it with its parent on the M4 Pro in three alternating runs:

| | Before | Now |
|---|---|---|
| Planned from rest (438), mean duration | 2.323 s | 2.171 s (6.6% sooner) |
| The hand's straight moves (121) | 1.057 s | 1.025 s (3.0% sooner) |
| Replanned while moving (61) | 3.763 s | 3.548 s (5.7% sooner) |
| Timing each, one thread: from rest, straight moves | 302, 201 µs | 273, 189 µs |
| The 2,600 problems' motions, per set | | 2.2–7.1% sooner, 5.0% on average |
| The 2,600 problems, `ik+plan` per second, 14 threads | 1,733, 2,441, 2,421 | 2,437, 2,465, 2,426 |

Paths, lengths and solved problems were unchanged by that change. Subsequent work in
[#27](https://github.com/natsuku-lab/batchplan/pull/27) and
[#29](https://github.com/natsuku-lab/batchplan/pull/29) reduced moving-start timing from
1,172 to 475 µs on one Ryzen thread (three rotating runs, load under 6). On 1,393 Panda and
SO-101 replans, mean duration changed by −0.2%; three took over 10% longer, at most 11.5%.
Stricter checks also caught between-sample acceleration and jerk overshoots and can refuse
stops accepted by earlier versions. See the [changelog](../CHANGELOG.md) for behavior changes
and [replanning](planning.md#starting-from-a-moving-state) for the contract.

## Tracking on the SO-101's servos

`examples/so101_tracking.rs` times random motions of the SO-101 in free space under several limits, and `scripts/so101_tracking.py` plays them in MuJoCo through the model of its Feetech STS3215 (7.4 V) servos that [BAM](https://github.com/Rhoban/bam) identified from recorded trajectories (model m6: the firmware's rate-limited P controller, PWM and supply limits, load-dependent friction). It measures the largest gap between the arm and the trajectory while it runs, the median over the motions, any joint. Each motion starts settled under gravity, and every 2 ms step sends that step's target. The stock arm powers these servos from 5 V; BAM identified them at 7.4 V. Gain 16 is what LeRobot sets; 32 is the firmware's default.

| 165 motions | Duration | 5 V, gain 16 | 5 V, gain 32 | 7.4 V, gain 16 | 7.4 V, gain 32 |
|---|---|---|---|---|---|
| Limits as given | 1.70 s | 16.5° (7.4%) | 8.3° (7.3%) | 11.3° | 5.6° |
| Speed × 0.75 | 2.26 s | 13.0° (0.8%) | 6.5° (0.7%) | 8.7° | 4.4° |
| Acceleration × 0.5 | 2.15 s | 13.5° (0.8%) | 6.8° (0.7%) | 9.2° | 4.6° |
| Jerk × 0.25 | 2.35 s | 14.2° (1.5%) | 7.0° (1.4%) | 9.5° | 4.6° |

The share of steps with a servo at its PWM limit is in brackets; at 7.4 V none were. On the Ryzen, at 9d49051.

- **Speed, not jerk, sets how far the arm lags.** At about the same duration, a quarter of the jerk limit tracks no better than a slower speed or acceleration.
- **At 5 V the SO-101 cannot keep up with its limits at full speed.** Its servos sit at the PWM limit on 7.4% of steps, and the worst motion lags 58°; at 0.75 of the speed, 0.8% and 29°. `Trajectory::try_new`'s speed scale, or `RobotOptions::velocity_scale`, slows it.
- **[Joined](#joining-motions) pick-like sequences track as their motions do one after another:** 16.2° against 16.6° at 5 V, joined within 5 cm, finishing 8% sooner.

## Bimanual handovers

`examples/handover/` generates [training episodes](training-data.md#handovers) on two reBot
B601-DM arms 44 cm apart. One arm picks up a 24–28 cm block, passes it to the other, and the
other places it. Half go in each direction. Three quarters of the tables have a wall, a screen
with an opening, or posts; both arms and the held block must clear the divider. Each stage is
batched across tasks.

| 512 tasks | M4 Pro GPU | M4 Pro CPU, 14 threads | Radeon 8060S | Ryzen CPU (32 threads) |
|---|---|---|---|---|
| Handovers | 510 of 512 | 510 of 512 | 510 of 512 | 510 of 512 |
| Time | 0.86–0.87 s | 1.06–1.07 s | 0.81–0.90 s | 0.97–1.19 s |
| Episodes/s | 584–596 | 478–483 | 566–628 | 428–525 |

Three runs per device, each started with load under 6; the M4 Pro runs are at bf2f5b9.
The 510 episodes contain about 110–111 minutes of motion at 30 fps, with a 12.7 s median
episode. The two failures occur at grasp IK.

This fixture uses up to 30,000 search samples rather than the default 10,000 for its difficult
two-arm carries, four block sizes to share held-object models, and four seeds kept for place IK.
These settings are part of the measured workload.

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

## How the speed came

Selected A/B measurements explain the main performance changes. The
[architecture guide](architecture.md) owns implementation details; the
[changelog](../CHANGELOG.md) records release history. Improvements from separate experiments
are not additive.

### CPU planning and collision checks

The frozen-baseline comparison [above](#before-and-after-optimization) measures the combined
search and timing improvements. Searches use SIMD checks, reuse per-thread buffers and let idle
threads help long searches; their results are independent of helper scheduling.

The CPU changes in [#55](https://github.com/natsuku-lab/batchplan/pull/55) avoid heavy grid code
in primitive checks and keep hot call boundaries stable. `examples/latency.rs` and
`examples/tasks.rs` compared `5d1fffb` with `64a3868` in six alternating rounds on the Ryzen's
32 threads. Times below are best / lower quartile in milliseconds. The baseline used sixteen
codegen units; the changed build used one. Background load varied; one-thread controls also
covered smaller calls. Results were bit-identical.

| | Before, best / lower quartile | After, best / lower quartile |
|---|---|---|
| IK, 1,000 targets among clutter | 69.83 / 77.54 | 63.44 / 63.51 |
| A UR5e, 1,000 targets among clutter | 97.34 / 111.63 | 88.43 / 88.84 |
| 1,000 targets, a grid among the clutter | 81.68 / 92.29 | 77.81 / 78.36 |
| `evaluate`, 16,384 configurations | 2.007 / 2.579 | 1.865 / 1.935 |
| 256 plans, 4 optimized seeds | 389.17 / 469.62 | 303.38 / 307.77 |
| 256 plans past a ball | 557.03 / 625.79 | 522.55 / 527.50 |
| 64 picks | 160.3 / 183.5 | 156.9 / 162.2 |

The M4 Pro's best/lower-quartile reductions were 4.1/4.0% for IK among clutter,
5.7/6.5% for `evaluate`, 6.1/6.3% for optimized plans and 3.1/3.3% for 64 picks.
Plans past a ball took 0.9/2.0% longer on all 14 threads; one-thread controls were faster.
Dependent builds using sixteen codegen units were checked too; their Ryzen optimized-plan
times were 253.86/313.21 ms, so build profiles must be matched when comparing results.

Earlier [#45](https://github.com/natsuku-lab/batchplan/pull/45) skipped self-collision sphere
pairs that cannot contribute to IK or optimization cost. On an M4 Pro, a place with four holds
and eight release poses on a 512-sphere Panda took 10.8–11.1 ms instead of 17.7–18.7 ms
(medians of fifteen calls, three alternating rounds, load 10), with identical costs and gradients.

### GPU latency and grids

[#36](https://github.com/natsuku-lab/batchplan/pull/36) writes out short kernel loops and
shares per-configuration IK work while retaining naga's safety checks. `examples/latency.rs`
on the M4 Pro, four alternating rounds started with load under 5, median milliseconds:

| Targets per call | No spheres | Spheres, no push | Spheres and the push | Two hands that share joints, the push |
|---|---|---|---|---|
| 1 | 1.14 → 0.34 | 1.23 → 0.39 | 2.03 → 0.92 | 4.31 → 3.59 |
| 10 | 1.13 → 0.34 | 1.24 → 0.40 | 2.89 → 1.81 | 5.73 → 4.74 |
| 100 | 1.41 → 0.61 | 1.70 → 0.85 | 5.59 → 4.38 | 8.91 → 7.65 |
| 1,000 | 3.83 → 1.72 | 5.93 → 2.98 | 30.0 → 22.1 | 52.0 → 42.2 |

In that experiment, optimized plans improved from 93.8 to 83.1 ms for a batch of 256.
Regressions were also measured: collision-heavy IK for a 399-sphere Panda took 4–5% longer,
plans past a ball about 5% longer, and `evaluate` 1.5–4% longer. GPU numerical results can
change in their last places. These are workload-specific gains, not a uniform speedup.

[#38](https://github.com/natsuku-lab/batchplan/pull/38) skips grid lookups for distant links.
For 1,000 Panda IK targets among clutter on the M4 Pro, a grid 2 m behind the arm changed GPU
latency from 35.7 to 28.8 ms and CPU latency from 107 to 67 ms (four GPU and twelve CPU
alternating rounds, `examples/latency.rs`). A grid filling the scene gains little from that test.

Compiling pipelines concurrently reduced cold GPU device creation from 2.6 to 1.4 s on the M4
Pro (three alternating runs with the driver cache emptied). Separate grid kernels increased
resident device memory from 35–37 to 66–88 MB, and memory per held shape from about 9 to 20 MB.

### Fitting collision spheres

Dense surface samples check the selected spheres four at a time, using SIMD with the scalar
test's addition order and inclusive boundary. The fitted models are unchanged: the sampling,
greedy selection, refinement order and final inflation are the same. This speeds up fitting
without loosening its coverage or overhang bounds.

Four alternating rounds against `30ca805` on 2026-11-14, with one caller thread and one codegen
unit, using the fixtures in `examples/fitting.rs`. Each invocation warms up with one load,
qualifies that model outside timing, and times three more loads. The table gives the best and
lower quartile of the rounds' best times, in milliseconds. Both machines had other work running;
the comparison uses builds run beside each other, not earlier idle measurements.

| Model / sphere budget | M4 Pro before | M4 Pro after | Ryzen AI Max+ 395 before | Ryzen after |
|---|---:|---:|---:|---:|
| Stand / 512 | 430.33 / 440.53 | 346.96 / 349.90 | 481.38 / 484.20 | 338.90 / 339.00 |
| Panda / 1,200 | 338.16 / 341.92 | 325.08 / 332.63 | 378.39 / 379.23 | 338.14 / 341.19 |
| SO-101 / 700 | 141.43 / 142.48 | 131.43 / 132.78 | 152.17 / 153.99 | 127.74 / 127.80 |
| UR5e / 1,200 | 254.71 / 258.37 | 225.78 / 233.74 | 272.70 / 273.48 | 221.81 / 221.81 |

These runs set `self_collision_samples` to zero to isolate fitting from self-collision pair
analysis. Runs with its default 10,000 samples, with smaller and larger sphere budgets, with
fourteen or thirty-two caller threads, and with sixteen codegen units also compare the complete
collision-model checksums before and after. The example prints those checksums, sphere counts
and measured overhangs alongside times.

```bash
cargo run --release --example fitting -- 1 stand 512 0 3
cargo run --release --example fitting -- 1 panda 1200 0 3
# Default self-collision analysis, on fourteen threads:
cargo run --release --example fitting -- 14 panda 1200 10000 3
# A dependent crate's default codegen-unit count:
cargo run --release --config 'profile.release.package.batchplan.codegen-units=16' --example fitting -- 1 so101 700 0 3
```

## Reproducing

This workspace builds batchplan as one codegen unit (`Cargo.toml`); measurements before [#55](https://github.com/natsuku-lab/batchplan/pull/55) used sixteen unless stated otherwise. A crate that depends on batchplan builds it with its own profile; to build it as here, add `[profile.release.package.batchplan]` with `codegen-units = 1` to that crate's `Cargo.toml`.

```bash
# GPU against CPU
cargo run --release --example bench -- 512

# Tool paths, and picks and places on random tabletops
cargo run --release --example tool_paths
cargo run --release --example tasks -- 64

# Joining pick-like motions on one reBot (after scripts/fetch_rebot.sh), or the SO-101; timing by kind of path
cargo run --release --example join
cargo run --release --example join -- so101
cargo run --release --example latency             # one call's time by kind, on the GPU; -- --cpu on the CPU
cargo run --release --example timing

# Tracking on the SO-101's servos, on BAM's model of them
uv venv .venv-so101 --python 3.12
uv pip install --python .venv-so101/bin/python "better-actuator-models[mujoco]==1.0.3" mujoco==3.17.0 numpy==2.5.4
cargo run --release --example so101_tracking -- data/so101_tracking      # --sequences for joined pick-like motions
.venv-so101/bin/python scripts/so101_tracking.py data/so101_tracking      # --kp 32, --vin 7.4

# Two reBot arms handing blocks over (downloads the arm at a pinned commit, with checksums), and their camera videos
scripts/fetch_rebot.sh
cargo run --release --example handover -- 512  # add --cpu for the CPU device
cargo run --release --features lerobot --example handover -- 16 --lerobot data/handover
.venv/bin/python scripts/replay_mujoco.py data/handover data/handover_video

# The 2,600 problems on every device (downloads robometrics' YAML and VAMP's Panda spheres at pinned commits, with checksums)
scripts/fetch_benchmark.sh
cargo run --release --example benchmark                # the table's timings; with --meshes, its mesh column
cargo run --release --example benchmark -- --meshes    # --clearance M (or --search-clearance, --shortcut-clearance), --passes N, --no-straighten, --optimize S, --rrt-seed N, --spheres curobo|vamp|fitted, --device gpu|cpu

# The gripper kept upright past a pillar, and kept inside the zone too
cargo run --release --example constraints -- 64 5
cargo run --release --example constraints -- 64 5 --zone

# The comparison with other planners: one spherized Panda for every planner, one judge
uv venv .venv-roboplan --python 3.12
uv pip install --python .venv-roboplan/bin/python roboplan==0.7.0 pyyaml numpy pytest
.venv-roboplan/bin/python scripts/bench_model.py export data/robometrics data/bench/inputs   # the canonical model and scenes, with checksums
.venv-roboplan/bin/pytest scripts/test_bench.py      # faults the judge and the parity check must catch
scripts/build_compare_cpp.sh                         # VAMP 0.6.4 and OMPL, in the moveit/moveit2:jazzy-release image
# The parity check: one command per planner, as scripts/parity.py's header lists them, then `parity.py compare`
scripts/run_suite.sh data/bench/final final          # every configuration in rotating rounds, each run judged
for d in data/bench/final/*/*/round1; do cargo run --release --example compare -- mesh data/bench/inputs "$d"; done
scripts/report_suite.sh data/bench/final             # the planner comparison tables
```
