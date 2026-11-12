# Benchmarks

The numbers on this page come from five machines:

- an **Apple M4 Pro**: 14 CPU cores and a 20-core GPU, on Metal;
- an **AMD Ryzen AI Max+ 395** CPU (16 cores, 32 threads) with a **Radeon 8060S** GPU, on Vulkan through Mesa's RADV driver, under Ubuntu 26.04;
- an **NVIDIA T4** GPU with 8 threads of a 2.5 GHz Xeon (an AWS g4dn.2xlarge), for cuRobo, which needs CUDA, under Ubuntu 24.04;
- an **NVIDIA A10G** GPU with 4 threads of an AMD EPYC 7R32 (an AWS g5.xlarge), for RoboKit, under Ubuntu 24.04;
- an **NVIDIA RTX 4090** GPU with 13 threads of an AMD EPYC 75F3 (a rented container, Ubuntu 22.04), for RoboKit and batchplan on one machine, a second such with 48 threads of an AMD EPYC 9254 and a third on a shared EPYC 75F3 where a table says so.

The load was checked before each run on the first two; the rented machines ran nothing else of ours, and what their other tenants ran is not known. Every command to reproduce them is [at the end](#reproducing).

## Against other planners

One robot model for every planner, one checker for every output, one machine, planners in rotating order. The tables are the 262 final problems, held back until optimization had ended; `scripts/report.py` regenerates each of them from the saved records ([commands](#reproducing)).

### Setup

| | |
|---|---|
| Robot | The Franka Panda of VAMP's spherized URDF: 59 spheres, fingers fixed 6.5 cm out, and the 34 disabled pairs of MoveIt's Panda SRDF. Every planner loads these files (`scripts/bench_model.py export`). On 2,250 fixture configurations (random, within half a millimetre of contact, and colliding), every planner's forward kinematics agrees with the judge's within 0.3 µm, and every collision verdict agrees (`scripts/parity.py`). |
| Problems | MotionBenchMaker's 800 and MπNets' 1,800 Panda problems (robometrics), from a joint-space start to a joint-space goal. The first 20% of each set (520) were for tuning, the next 70% (1,818) for the frozen baseline, and the last 10% (262) are the final set below. The final set ran four times: the first two, at 17266d6 and be046f4, came before fixes to regressions found on the held-out problems, and the second was read; the third, at 44b086b, was published, and the changes since were measured on the tuning problems alone. |
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

<details>
<summary>Per set: the share of queries solved validly</summary>

| Set | Problems | batchplan | RoboPlan | MoveIt 2 | batchplan, capsules | VAMP | OMPL |
|---|---|---|---|---|---|---|---|
| `bookshelf_small_panda` | 10 | 100% | 90% | 59% | 100% | 96% | 62% |
| `bookshelf_tall_panda` | 10 | 100% | 94% | 59% | 100% | 98% | 60% |
| `bookshelf_thin_panda` | 10 | 100% | 90% | 64% | 100% | 94% | 75% |
| `box_panda` | 10 | 100% | 98% | 71% | 100% | 98% | 69% |
| `box_panda_flipped` | 10 | 100% | 96% | 59% | 100% | 98% | 57% |
| `cage_panda` | 10 | 100% | 90% | 44% | 100% | 96% | 55% |
| `cubby_neutral_goal` | 8 | 88% | 85% | 72% | 88% | 85% | 78% |
| `cubby_neutral_start` | 8 | 100% | 90% | 59% | 100% | 98% | 62% |
| `cubby_task_oriented` | 15 | 80% | 63% | 34% | 80% | 72% | 33% |
| `dresser_neutral_goal` | 15 | 93% | 87% | 53% | 93% | 89% | 64% |
| `dresser_neutral_start` | 15 | 100% | 89% | 76% | 100% | 99% | 81% |
| `dresser_task_oriented` | 30 | 100% | 90% | 46% | 100% | 89% | 50% |
| `merged_cubby_neutral_goal` | 8 | 100% | 100% | 94% | 100% | 100% | 93% |
| `merged_cubby_neutral_start` | 8 | 88% | 88% | 76% | 88% | 85% | 82% |
| `merged_cubby_task_oriented` | 15 | 93% | 89% | 78% | 93% | 91% | 69% |
| `table_pick_panda` | 10 | 100% | 92% | 84% | 100% | 98% | 88% |
| `table_under_pick_panda` | 10 | 100% | 88% | 38% | 100% | 90% | 40% |
| `tabletop_neutral_goal` | 15 | 93% | 92% | 84% | 93% | 92% | 85% |
| `tabletop_neutral_start` | 15 | 53% | 53% | 53% | 53% | 53% | 53% |
| `tabletop_task_oriented` | 30 | 73% | 70% | 68% | 73% | 73% | 71% |
</details>

### Before and after optimization

The suite first ran on the 1,818 held-out problems at a237326, and that baseline was frozen. The changes under [How the speed came](#how-the-speed-came) were then made against it on the tuning problems, and batchplan was measured again on the held-out ones. Paths are 0.2% shorter and durations the same. Six fewer of the 27,270 answers are valid (24,780 against 24,786): searches of a few problems at the edge of solvability ran out of samples in 15 queries that found a path before, and found one in 9 that did not:

| batchplan, held-out problems, 16 cores | a237326 | ef619b1 |
|---|---|---|
| Paths: time per query (median / p99), paired | 0.12 / 3.6 ms | 0.11 / 1.9 ms, 0.78× |
| Paths: valid/s | 30,867 | 37,560 |
| Paths, one core: p99 | 6.8 ms | 1.9 ms |
| Trajectories: time per query (median / p99), paired | 1.06 / 4.4 ms | 0.52 / 2.3 ms, 0.50× |
| Trajectories: valid/s | 9,018 | 16,147 |

### The earlier comparison

Before this harness, batchplan was compared with each planner on that planner's own robot model: on VAMP's spheres against VAMP and OMPL, on its own spheres against RoboPlan and MoveIt 2 (which planned on the meshes), and against cuRobo on a T4. The comparison above replaces it. Its numbers are as recorded then, at batchplan 0.1.1–0.2.0.

<details>
<summary>The earlier comparison, as published</summary>

These results describe the configurations and hardware recorded below. They do not isolate planner implementation speed where collision geometry, pipeline stages, settings or hardware differ. The same output checker makes validation consistent, but does not equalize the work done during planning. No new benchmark runs accompany this clarification.

**The problems.**

MotionBenchMaker's 800 problems (8 sets) and MπNets' 1,800 (12 sets), all for the Franka Panda, as [robometrics](https://github.com/fishbotics/robometrics) packages them. Each planner plans from the start to the set's first IK solution: a joint-space goal, so only planning is compared.

| Planner | Robot model | How it ran |
|---|---|---|
| batchplan | cuRobo's Panda spheres, which ship with it; and VAMP's | Default `PlanOptions`. `benchmark --spheres vamp` loads VAMP's spheres onto batchplan's Panda; all 59 sphere centres match `vamp.panda.fk` within 0.4 µm, and self-collision verdicts agree, on 300 random configurations. Each set is one `plan` call on the CPU device. Time per problem is one problem per call, for every problem. |
| [VAMP](https://github.com/KavrakiLab/vamp) 0.6.4 | VAMP's Panda spheres | RRT-Connect, then VAMP's simplification, with the Halton sampler and its SIMD collision checker, called from C++ as its Python bindings call them (`scripts/cpp_planners.cc`, built with VAMP's release flags). RRT-Connect steps 1 rad, VAMP's Panda range (`ROBOT_RRT_RANGES`, which its benchmark script uses), or 2 rad, `RRTCSettings`' default, and stops at VAMP's 100,000 samples or at batchplan's limit, 10,000. Each problem runs on one core, as many in parallel as there are cores. |
| [OMPL](https://ompl.kavrakilab.org) 2.0.2 | VAMP's Panda spheres | `RRTConnect` with a 5 s limit, then `simplifySolution` for 1 s, from C++. VAMP's check of one configuration is the state validity checker; edges are checked by OMPL's own motion validator. Each problem runs on one core. |
| [cuRobo](https://github.com/NVlabs/curobo) 0.8.0 | cuRobo's Panda spheres | `MotionPlanner.plan_cspace` with its defaults (5 attempts, graph seeding from the second, a 1 cm collision activation distance) and CUDA graphs, one problem at a time; and `BatchMotionPlanner`, 32 problems per call, each in its own world. On the T4, compared on the 1,300 problems whose scenes are only boxes (below). |
| [RoboPlan](https://github.com/open-planning/roboplan) 0.7.0 | The Panda's meshes | RRT-Connect with a 100,000-node cap and a 5 s limit, then its path shortcutter. It checks the meshes against exact boxes and cylinders with Coal. Groups and allowed collisions come from MoveIt's Panda SRDF. Each problem runs on one core. The TOPP-RA trajectory generator is not called. |
| [MoveIt 2](https://moveit.ai) (Jazzy) | The Panda's meshes | `moveit_py` with the default OMPL pipeline (RRT-Connect, simplification, time parameterization and validation) and a 5 s limit. It checks the meshes with FCL, configured by `moveit_resources_panda_moveit_config`. Each problem runs on one core, in the official `moveit/moveit2:jazzy-release` image. |

Every planner gets the same scenes, as close as it can take them:
- VAMP and OMPL get cylinders as capsules, as VAMP's own MotionBenchMaker loader does, and boxes in the `box` sets. The others get exact boxes and cylinders.
- Every scene's floor or table top is level with the Panda's base. batchplan, RoboPlan, MoveIt 2 and the judge never check the base against the scene. VAMP does, and its base sphere reaches 3 cm below the base, so the boxes under it get a hole around it, a centimetre wider than the sphere, where no other link can reach. Everywhere else, VAMP plans in the scene the judge checks. Cutting a box into pieces costs VAMP about 9% of its planning time: every problem planned with the same geometry as one box and as pieces took 0.91 and 1.00 ms on average.

**The checks.**

One checker, `benchmark --check`, judges every planner's paths, batchplan's included (`benchmark --save` writes batchplan's in the other planners' format, its splines sampled at 32 points per span). A path solves its problem when:

- the hand ends within 1 cm of the goal;
- every joint stays within its limits;
- the path is collision-free, sampled at most 0.005 rad apart.

Collision is checked twice:

- on the spheres the planners plan on: "solved" below. That is VAMP's spheres for batchplan, VAMP and OMPL, and cuRobo's for batchplan and cuRobo;
- on the Panda's collision meshes, with [`MeshModel`](robots.md#checking-against-meshes), for every planner: "clear on the meshes". Self-collision counts between the link pairs MoveIt's Panda SRDF leaves enabled. (batchplan's own `--meshes` check, in the device table below, uses its model's pairs, so its numbers differ slightly.)

The Panda's collision meshes provide a shared model for the output check. Three of the planners plan on spheres, while RoboPlan and MoveIt 2 check meshes at intervals along each edge. The judge also samples paths; passing it is not a continuous collision guarantee or a physical-robot validation. Path lengths are medians over the problems every planner in a table solves.

**Results.**

**On the same robot model.** batchplan, VAMP and OMPL plan on VAMP's Panda spheres and are judged on them, on the M4 Pro. *One core* plans every problem in turn. *In batches* counts each set's problems over the time the whole set took on all 14 cores: batchplan's set in one `plan` call, VAMP's and OMPL's one problem per thread, each thread taking the next. batchplan is built without flags and picks the CPU's widest SIMD instructions at run time; VAMP is built with its release flags for the machine (`-mcpu=native`). Times are medians of two or three alternating rounds.

| Planner | Solved | Clear on the meshes | Path length (median, 1,526 problems all seven solve) | Time per problem (median / p98) | All 2,600, one core | In batches |
|---|---|---|---|---|---|---|
| batchplan | **90.4%** | **85.2%** | 3.40 rad | 0.17 / 1.68 ms | 0.91 s | 19,902 /s |
| batchplan without clearance (`--clearance 0`) | 88.2% | 70.8% | **3.37 rad** | **0.15** / **0.88** ms | **0.66 s** | **26,011 /s** |
| VAMP, 1 rad steps | 86.1% | 71.6% | 3.53 rad | 0.26 / 2.13 ms | 2.43 s | 2,689 /s |
| VAMP, 1 rad steps, stopping at 10,000 samples | 86.1% | 71.5% | 3.53 rad | 0.26 / 1.75 ms | 1.06 s | 16,111 /s |
| VAMP, 2 rad steps | 86.4% | 75.3% | 3.53 rad | 0.27 / 3.23 ms | 2.72 s | 2,697 /s |
| VAMP, 2 rad steps, stopping at 10,000 samples | 86.1% | 75.0% | 3.53 rad | 0.27 / 2.22 ms | 1.20 s | 13,494 /s |
| OMPL | 63.9% | 63.5% | 3.99 rad | 2.75 / 5,049 ms | 631 s | 30 /s |

batchplan's and VAMP's rows are three alternating rounds of one session; OMPL's are from an earlier one. From Python, with 2 rad steps, VAMP's bindings plan the same paths in 0.31 ms (median) and 2,409 problems/s in worker processes. OMPL, calling VAMP's checker back from Python, takes 3.34 ms and plans 30 problems/s. OMPL's totals are set by its 5 s limit, which about 120 problems run out.

**Against MoveIt 2 and RoboPlan**, which plan on the Panda's meshes, on the Ryzen. batchplan plans on cuRobo's spheres, which ship with it; every path is judged on the meshes. This compares different collision representations and pipeline stages, not equivalent planning workloads. A shared-sphere comparison with RoboPlan and MoveIt has not been run.

| Planner | Solved, on the meshes | Path length (median, 1,334 problems all three solve) | Time per problem (median) | In batches |
|---|---|---|---|---|
| batchplan | **93.1%** | **3.23 rad** | **0.13 ms** | **31,196 /s** |
| RoboPlan | 73.8% | 3.78 rad | 10.3 ms | 174 /s |
| MoveIt 2 | 63.1% | 3.80 rad | 59.4 ms | 82 /s |

On the M4 Pro, batchplan plans in 0.18 ms and 19,194 problems/s, and RoboPlan in 6.28 ms and 176/s, with as many of its paths clear as on the Ryzen. MoveIt 2 ran only on the Ryzen, in Docker. VAMP built with GCC 15 crashed on the Ryzen, so VAMP and OMPL ran only on the M4 Pro.

**Against cuRobo**, both on cuRobo's spheres, cuRobo on the T4 machine. That run loaded every obstacle as an oriented box, as cuRobo's own benchmark does. A cylinder loaded as a box closes the gap between the fingers at the goals that grasp it, so cuRobo is compared on the 1,300 problems whose scenes have no cylinders: `cage_panda` and the `cubby`, `dresser` and `merged_cubby` sets. (`scripts/compare_planners.py` now loads cylinders as meshes, with `get_collision_check_world`, so a run can cover all 2,600; boxes load as before.) *In batches* is each set over its wall time: batchplan's set in one call, cuRobo's in calls of 32.

| Planner | Solved | Clear on the meshes | Path length (median, 1,092 problems all six solve) | Time per problem (median) | In batches |
|---|---|---|---|---|---|
| batchplan, CPU device | **98.9%** | 92.8% | 3.243 rad | **0.24 ms**¹ | 13,167 /s¹ |
| batchplan, GPU device | **98.9%** | 92.8% | 3.243 rad | 0.25 ms¹ | **13,613 /s**¹ |
| batchplan, optimizing 2 seeds (`--optimize 2`) | 98.8% | 93.0% | **3.137 rad** | 14.8 ms¹ | 142 /s¹, 745 /s on the GPU |
| batchplan, optimizing 4 seeds (`--optimize 4`) | 98.8% | **93.1%** | **3.137 rad** | 25.6 ms¹ | 72 /s¹, 441 /s on the GPU |
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

**Reading the results.**

- **Results for the tested shared-sphere configurations.** On VAMP's spheres, 9.3% of the problems start or end in collision, so no planner can solve more than 90.7% on them; batchplan solves 2,351 of the 2,358 others. VAMP counts 98.6% solved because it checks neither the start nor the goal, and checks edges 1/32 rad apart, where the judge's denser samples find grazes. OMPL's paths graze most often: its own motion validator checks edges at 1% of the joint space's extent, about 0.13 rad. batchplan's paths are the shortest because its searches start near the straight line ([below](#closing-the-path-length-gap)).
- **Planning times in the tested VAMP comparison.** Both run RRT-Connect on SIMD collision checks. VAMP searches its few unsolvable problems up to 100,000 samples, so all 2,600 take it 2.43 s on one core. Stopped at 10,000, as batchplan's searches are, VAMP is still slower on one core and in batches. A batch lasts as long as its few unsolvable problems, and batchplan's idle threads help those along ([below](#how-the-speed-came)). The slowest problems take about as long: VAMP's 98th percentile is 1.75 ms against batchplan's 1.68, which moves by ±0.3 ms with where batchplan's Halton samples start (`--rrt-seed`). The throughput tables describe the tested pipelines; the RoboPlan and MoveIt rows use different planning geometry and should not be read as isolated algorithm speedups.
- **The clearance costs a quarter of the speed, and buys the clear paths.** Without it (`--clearance 0`), batchplan plans 31% more problems per second on VAMP's spheres, but only 70.8% of its paths clear the meshes, against 85.2% with it and 71.5–75.3% for VAMP.
- **Sampled checks against the Panda collision meshes.** On the spheres that ship with it, 93.1% of its paths clear the Panda's meshes, against 73.8% for RoboPlan and 63.1% for MoveIt 2, which plan on the meshes. Over the 1,334 problems all three solve, its paths are also the shortest: 3.23 rad against 3.78–3.80. On the scenes of boxes, it clears 92.8% against cuRobo's 81.6%. cuRobo's hand-tuned spheres are smaller around the hand than VAMP's: 98.1% of the problems have a free start and goal on them, and batchplan solves 98.0%.
- **Path lengths with two optimization seeds in the tested cuRobo comparison.** Optimizing the searched path and the straight line (`PlanOptions::seeds = 2`) gives 3.137 rad where all solve, against cuRobo's 3.144, and the shorter path on 78% of the problems. It costs 14.8 ms per problem on 14 CPU threads, or 745 problems/s in batches on the M4 Pro's GPU. cuRobo's mean stays 1% shorter (0.2% against four seeds), from a few long problems. Without optimization, batchplan's paths are 3% longer than cuRobo's (3.243 rad), from searches that stay near the straight line ([below](#closing-the-path-length-gap)), at a sixtieth of the optimized time.
- **The measured pipelines produce different outputs.** Our VAMP, OMPL and RoboPlan adapters return geometric waypoint paths. We did not call RoboPlan's TOPP-RA trajectory generator. Its default Hermite mode stops at waypoints, but its Cubic, Adaptive and Linear Blend modes can avoid intermediate stops, with collision checks and fallback behavior ([RoboPlan trajectory generation](https://roboplan.readthedocs.io/en/latest/concepts/trajectory_generation.html)). Those modes were not evaluated here. The measured MoveIt 2 and cuRobo pipelines return timed trajectories. batchplan's `plan` returns a cubic B-spline path; the separate `Trajectory::try_new` step times it within velocity, acceleration and jerk limits in about 0.3 ms. These results do not establish a ranking of executable trajectory quality or duration.
- **The timing boundaries differ.** batchplan's covers `plan`, excluding `Trajectory::try_new`. VAMP's and OMPL's cover building the scene, planning and simplifying, from C++. cuRobo's covers loading the world and `plan_cspace`, with the GPU synchronized. RoboPlan's covers RRT planning and shortcutting from Python, excluding scene setup and trajectory generation. MoveIt 2's covers its planning call from Python, including its configured simplification, time parameterization and validation. An equivalent-output comparison needs these stages measured separately.
</details>

## Against RoboKit

[RoboKit](https://github.com/openrobokit/robokit) (736f112) ships a benchmark of inverse kinematics and one of motion planning to pose goals, both judged by its own scorer. Its harnesses ran unchanged, and batchplan (9d49051) ran inside them: the same targets and problems, warm-ups and judge. `scripts/robokit/` holds the adapters ([commands](#reproducing)). The first tables ran on one AWS g5.xlarge, five runs each, but the two rows timed under 150 rad/s³, which ran on a second, and the four sampled 31 times per plan, on a third: an NVIDIA A10G (driver 595.112.07; RoboKit on CUDA, batchplan on Vulkan) and 4 threads of an AMD EPYC 7R32.

**Motion planning.** The 2,590 problems of MotionBenchMaker (799) and MπNets (1,791), each a start configuration and a goal pose among obstacles, planned one at a time. The judge is the benchmark's, with cuRobo's checks: the goal reached, no collision, and velocity, acceleration (15 rad/s²) and jerk (500 rad/s³) within limits, by 5-point stencils on each plan's own samples (batchplan's 25 ms apart or less, or 31 per plan in the last four rows; RoboKit's 31 per plan) interpolated onto 25 ms. Success allows joint positions 0.2 rad past their limits, as cuRobo's benchmark does, and "strict" allows none. For batchplan one `plan` call takes each pose goal, and the path is timed 1% under the judge's limits. Each planner's timer covers planning and timing the trajectory; building the scene is outside it for both. As run, the adapter counted one sample too many on about one batchplan motion in eight, which lengthens that motion by a step (25 ms at most) in the table's first seven rows; the adapter in `scripts/robokit/` no longer does, nor did it for the rows sampled 31 times per plan.

| Planner | Success (strict) | Time per problem, mean / median / 98th | Motion (median) | Path (median) | Jerk (median) |
|---|---|---|---|---|---|
| RoboKit, A10G | 99.59% (99.59%) | 23.5 / 18.9 / 67.6 ms | 1.220 s | 3.293 rad | 126 rad/s³ |
| batchplan, 4 CPU threads | 99.61% (99.61%) | 5.8 / 5.4 / 10.3 ms | 1.260 s | 3.293 rad | 245 rad/s³ |
| batchplan, A10G | 99.58% (99.58%) | 5.3 / 4.9 / 8.6 ms | 1.257 s | 3.293 rad | 245 rad/s³ |
| batchplan, 4 optimized seeds, 4 CPU threads | 99.61% (99.61%) | 139.6 / 137.4 / 515 ms | 1.168 s | 3.179 rad | 239 rad/s³ |
| batchplan, 4 optimized seeds, A10G | 99.58% (99.58%) | 17.7 / 19.2 / 47.7 ms | 1.164 s | 3.180 rad | 239 rad/s³ |
| batchplan timed under 150 rad/s³, A10G | 99.61% (99.61%) | 5.1 / 4.7 / 8.6 ms | 1.468 s | 3.293 rad | 125 rad/s³ |
| batchplan timed under 150 rad/s³, 4 optimized seeds, A10G | 99.61% (99.61%) | 17.5 / 19.3 / 47.8 ms | 1.344 s | 3.180 rad | 116 rad/s³ |
| batchplan, 31 samples per plan, A10G | 99.46% (99.46%) | 5.2 / 4.8 / 8.5 ms | 1.255 s | 3.292 rad | 186 rad/s³ |
| batchplan, 31 samples per plan, 4 optimized seeds, A10G | 99.50% (99.50%) | 17.6 / 19.1 / 47.5 ms | 1.159 s | 3.178 rad | 178 rad/s³ |
| batchplan timed under 300 rad/s³, 31 samples per plan, A10G | 99.38% (99.38%) | 5.2 / 4.8 / 8.7 ms | 1.311 s | 3.290 rad | 147 rad/s³ |
| batchplan timed under 300 rad/s³, 31 samples per plan, 4 optimized seeds, A10G | 99.46% (99.46%) | 17.4 / 18.9 / 47.4 ms | 1.201 s | 3.178 rad | 152 rad/s³ |

- **Success** at 25 ms samples (the first seven rows) is the same to within one problem. Every planner solves all of MotionBenchMaker; on MπNets RoboKit solves 99.41%, batchplan 99.44% on the CPU and 99.39% on the A10G.
- **Time.** Without optimization batchplan answers in a quarter of the time on either device. With four optimized seeds (`PlanOptions::seeds`) on the A10G it takes 17.7 ms on average against RoboKit's 23.5, and 19.2 against 18.9 at the median; on four CPU threads the optimization adds about 134 ms.
- **Motions.** Searched and shortcut, batchplan's motions take 3% longer than RoboKit's at the median, over paths of the same median length; with four optimized seeds, 4–5% less, over paths 3.5% shorter.
- **Jerk.** The first five rows are timed under different limits: batchplan's motions under the judge's 500 rad/s³, which they keep along the whole curve and whose samples read 245; RoboKit plans under 150 rad/s³, and its read 126. Timed under 150 too (`RobotOptions::max_jerk`, a second run on the same kind of instance), batchplan's read 125 and take 20% longer than RoboKit's, or 10% with four optimized seeds: at the same jerk RoboKit's motions are the shorter.
- **Sampling.** The judge's jerk is a reading of the samples it is given, and sparser samples read lower. Sampled 31 times per plan as RoboKit's are (`BATCHPLAN_FRAMES=31`), the same motions read 186 rad/s³ where 25 ms samples read 245, and motions timed under 300 rad/s³ read 147–152: with four optimized seeds those take 1.201 s at the median against RoboKit's 1.220 s, for 17.4 ms of planning on average against 23.5 and 18.9 at the median against 18.9, and read 152 against 126. At 31 samples batchplan's success reads 2 to 5 problems under RoboKit's: in the first of these rows' runs, four motions of 2.7–4.6 s, whose samples are 90–150 ms apart, read past a limit, and all else that failed did not reach its goal.

On one machine for both, a rented NVIDIA RTX 4090 (driver 595.91.07; 13 threads of an AMD EPYC 75F3; RoboKit 736f112 on CUDA; batchplan 75ec9fb on Vulkan with [#32](https://github.com/natsuku-lab/batchplan/pull/32)'s first change, which moves no motion), five runs each, batchplan sampled 31 times per plan and timed under a share of the judge's acceleration limit (`BATCHPLAN_ACCELERATION`):

| Planner | Success (strict) | Time per problem, mean / median / 98th | Motion (median) | Path (median) | Jerk (median) |
|---|---|---|---|---|---|
| RoboKit | 99.61% (99.61%) | 15.3 / 12.4 / 45.9 ms | 1.219 s | 3.291 rad | 126 rad/s³ |
| batchplan, 4 optimized seeds | 99.50% (99.50%) | 12.4 / 13.2 / 32.2 ms | 1.159 s | 3.178 rad | 178 rad/s³ |
| batchplan, 4 optimized seeds, acceleration × 0.8 | 99.46% (99.46%) | 12.6 / 13.3 / 32.8 ms | 1.208 s | 3.178 rad | 145 rad/s³ |
| batchplan, 4 optimized seeds, acceleration × 0.7 | 99.46% (99.42%) | 12.5 / 13.2 / 32.5 ms | 1.242 s | 3.178 rad | 125 rad/s³ |
| batchplan, acceleration × 0.7 | 99.54% (99.54%) | 3.8 / 3.4 / 6.4 ms | 1.338 s | 3.292 rad | 143 rad/s³ |

- **At an equal jerk reading RoboKit's motions are the shorter, by 1.9%** (1.219 s against 1.242 s at 126 and 125 rad/s³), for about the same planning time (12.4 ms against 13.2 at the median, 15.3 against 12.5 on average). At an equal duration batchplan's read rougher (145 at 1.208 s). With goal IK that screens its seeds (the second RTX 4090 of the IK table below, RoboKit not run again), the acceleration × 0.7 run solves as many as RoboKit, 99.61% (99.58% strict), in 11.7 / 12.3 / 31.0 ms, with motions of 1.253 s over 3.200 rad that read 124: the goals it plans to differ, and the motions are 2.8% longer than RoboKit's.
- **With four optimized seeds, a lower acceleration limit smooths the reading for less time than a lower jerk limit.** Timed under jerk limits of 250 and 220 rad/s³ instead, the optimized motions take 1.218 s and 1.241 s and read 145 and 133 (those two runs also had #32's second change and kernels built without shader runtime checks, which change the last bits of a GPU's results).
- **Without optimization** batchplan answers in a quarter of the time, with motions 10% longer than RoboKit's that read 14% rougher. There a jerk limit of 250 rad/s³ does better than the acceleration share: the same 1.338 s, reading 137 (built as the two runs above).

**Inverse kinematics.** The benchmark's Panda targets without obstacles, one batched solve, in milliseconds (success): a target is solved within 5 mm and 0.05 rad. batchplan runs its defaults, 64 seeds of 30 steps.

| Targets per call | RoboKit, A10G | batchplan, A10G | batchplan, 4 CPU threads | cuRobo, A10G | PyRoKi, A10G |
|---|---|---|---|---|---|
| 1 | 0.77 (100%) | 1.64 (100%) | 0.80 (100%) | 7.7 (100%) | 10.7 (100%) |
| 10 | 0.74 (100%) | 1.64 (100%) | 6.2 (100%) | 8.0 (100%) | 15.6 (100%) |
| 100 | 1.38 (100%) | 2.79 (100%) | 59.7 (100%) | 15.6 (99.8%) | 23.1 (100%) |
| 1,000 | 9.4 (99.94%) | 14.6 (100%) | 587 (100%) | 94.3 (99.94%) | 156 (99.88%) |
| 2,000 | 18.5 (99.91%) | 29.3 (99.98%) | 1,171 (99.98%) | 174 (99.92%) | 312 (99.89%) |

RoboKit's IK is the quickest on the A10G at every batch size: about twice as quick as batchplan's up to 100 targets per call, and 1.6 times at 1,000 and 2,000. One target takes batchplan's CPU threads 0.80 ms against RoboKit's 0.77. batchplan solves every target up to 1,000 per call. With 16 seeds it takes 4.7 ms for 1,000 targets and 8.1 ms for 2,000 on the A10G, and solves 99.6% of them.

Since then a problem's seeds share its targets, large batches are seeded on the device's threads, buffers are uploaded without a copy, and a robot without collision spheres, as this benchmark's is, skips the collision push. The same harness on a rented NVIDIA RTX 4090 (driver 595.91.07, Vulkan; 13 threads of an AMD EPYC 75F3; RoboKit 736f112; five runs, and batchplan's GPU columns the mean of two such), in milliseconds:

| Targets per call | RoboKit, RTX 4090 | batchplan at 75ec9fb, RTX 4090 | batchplan with the changes, RTX 4090 | batchplan, 13 CPU threads |
|---|---|---|---|---|
| 1 | 0.57 | 1.26 | 0.89 | 0.33–0.40 |
| 10 | 0.55 | 1.20 | 0.94 | 1.4–1.5 |
| 100 | 0.79 | 1.58 | 1.36 | 11.8–13.1 |
| 1,000 | 3.18 | 10.8 | 4.46 | 108–111 |
| 2,000 | 7.08 | 23.2 | 9.12 | 215–218 |

RoboKit's IK is still the quicker on the GPU, by 1.3 to 1.7 times where it was 2.0 to 3.4; for one target batchplan's CPU device is the quickest of all.

IK now [screens its seeds](planning.md#inverse-kinematics): all 64 take 8 steps, then 4 per target take the rest, in one round trip on a GPU. On another rented RTX 4090 (driver 595.91.07, Vulkan; 48 threads of an AMD EPYC 9254; RoboKit 736f112 and batchplan in one session; five runs, each batchplan column the range of two to six such), in milliseconds (success):

| Targets per call | RoboKit | batchplan before (ac861c5) | batchplan, 30 steps (the default) | batchplan, 20 steps |
|---|---|---|---|---|
| 1 | 0.56 (100%) | 0.69–0.71 (100%) | 0.75–0.95 (100%) | 0.60–0.82 (100%) |
| 10 | 0.53 (100%) | 0.71 (100%) | 0.77–0.80 (100%) | 0.62–0.63 (100%) |
| 100 | 0.76 (100%) | 0.97–1.00 (100%) | 1.06–1.09 (100%) | 0.88–0.92 (100%) |
| 1,000 | 3.14 (99.94%) | 3.58–3.72 (100%) | 3.05–3.28 (99.98%) | 2.99–3.10 (99.98%) |
| 2,000 | 6.81 (99.91%) | 7.21–10.24 (99.98%) | 6.13–6.32 (99.96%) | 5.65–5.82 (99.96%) |

- **From 1,000 targets per call batchplan is level with RoboKit or quicker,** by 10% at 2,000 with its defaults and 16% with 20 steps (`IkOptions { iterations: 20, .. }`), and solves a few more of them.
- **Up to 100 targets RoboKit was the quicker,** by 7–46% against batchplan's 20 steps (most runs 13–19%) and by a third and more against its defaults. So few seeds do not fill the GPU, so a call takes as long as its steps take one after another, and leaving most seeds behind saves nothing. RoboKit takes 16 steps where batchplan's default is 30. With every seed taking 20 steps (`screening: 0`), batchplan takes 0.58, 0.57 and 0.84 ms for 1, 10 and 100 targets.
- **Small calls since.** In that session the screening cost these sizes 0.05–0.1 ms: it ran as six submissions with a pass for the seeds' clearances halfway. Since then a small call is one submission; its buffers are kept from the call before; and where the steps do not push away from collisions, as for this robot without spheres, the seeds are chosen by their errors alone and their clearances found once, at the end. No result of the default IK changes with any of it. This is measured on the M4 Pro and not yet on an NVIDIA GPU. On the M4 Pro, with ac861c5's build and this change's in turn (random reachable targets in an empty world for the Panda with its 61 spheres and without any, 60 calls of `solve_ik` each; the medians of five rounds, less one round each build lost to other load), one target without spheres takes 1.02–1.03 ms where it took 1.02–1.07, and ten 1.04–1.06 where 1.05–1.09; with spheres and no push, 1.16–1.18 where 1.16–1.19, and 1.17–1.19 where 1.17–1.19; with the push (three rounds), 2.71–2.74 where 2.89, and 2.53–2.56 where 3.27–3.31.
- **With collision spheres, on the M4 Pro** (the Panda's 61, random reachable targets in an empty world, the default and `screening: 0` in turn in one process, the machine under other load): 1,000 targets take 36–39 ms where they took 68–82, and the CPU's 14 threads 95 ms where 212, with 945 solved where 944.
- **Where the time goes.** For this robot with collision spheres, a call of one target with no steps takes 0.17 ms on this machine and each step about 35 µs (about 12 µs without spheres). At 1,000 targets, of 5.7 ms with 20 steps, the GPU works for 2.6 ms; seeding takes 0.5 ms, reading the results back 0.7 ms, and copying and assembling them 1.1 ms, which has since been cut to one copy.

With the kernels' short loops written out ([how](#how-the-speed-came)) and the kernels built with every check naga makes, on a third rented RTX 4090 (driver 595.91.07, Vulkan; an AMD EPYC 75F3 shared with other tenants, its load between 25 and 43 through the session; RoboKit 736f112 and batchplan in turn, four rounds of five runs each), in milliseconds:

| Targets per call | RoboKit | batchplan, 30 steps (the default) | batchplan, 20 steps |
|---|---|---|---|
| 1 | 0.57–0.59 | 0.33–0.42 | 0.32–0.40 |
| 10 | 0.55–0.56 | 0.35–0.43 | 0.34–0.38 |
| 100 | 0.78–0.79 | 0.70–1.00 | 0.67–1.22 |
| 1,000 | 3.14–3.16 | 2.80 in three rounds, 7.6 in one | 2.8–3.0 in two rounds, 6.1 in two |
| 2,000 | 6.81–7.09 | 5.5–8.0 in three rounds, 10.7 in one | 5.6–6.5 in two rounds, 11 in two |

- **One target and ten take batchplan's defaults about 40% less time than RoboKit,** solving all of them.
- **At 1,000 and 2,000 targets batchplan is the quicker while the host's processors are free,** and takes twice as long in the rounds where they were not: seeding 64,000 seeds, copying and assembling their results is work on the host, where RoboKit's seeds never leave the GPU. At 100 targets the two are level.
- **The same session's other measures** (the Panda with its 61 spheres; the build before and this one): one target with the push away from obstacles takes 0.72 ms where it took 1.11, 1,000 take 4.3 ms where 5.3, and without spheres one takes 0.19 ms where 0.57; 16 plans with 4 optimized seeds take 6.8 ms where 10.3, and 256 take 33 ms as before. Built without naga's loop counters, the same kernels gain 3% at most.
- **Motion plans** in RoboKit's harness with these kernels (4 optimized seeds, 31 frames, five runs): timed under 0.7 of the acceleration limit, 99.58% (99.54% strict) in 12.2 / 12.1 / 33.8 ms with motions of 1.253 s that read 124.5; under the full limit, 99.58% (99.58%), 1.172 s, reading 174.

## The 2,600 problems on every device

`examples/benchmark.rs` runs every set in two modes:

- **plan:** plan from the start to the set's first IK solution, as in the comparison above;
- **IK + plan:** plan to the goal pose: `plan` solves IK from the start and ends at the nearest collision-free solution ([pose goals](planning.md#planning-a-path)).

It checks success as above, on cuRobo's spheres, which ship with batchplan. The timings are medians of three runs, each started with the machine's load under 6: the M4 Pro's with the code of bf2f5b9 (its GPU also drives its display), the Ryzen's and the Radeon's with the code of 9d49051. All were taken without `--meshes`, which only the mesh column was run with: in runs with it, in another session, the Ryzen's and the Radeon's IK + plan read 3,048 and 2,510 problems/s. "Free ends" counts the problems whose start and given IK solution are collision-free on these spheres, which caps what planning can reach. Throughput runs each set as one batch; time per problem covers the first 20 problems of each set, one at a time.

| Device | Free ends | Plan | IK + plan | IK + plan, clear on meshes | Plan, problems/s | Plan, time per problem (median / p98) | IK + plan, problems/s | IK + plan, time per problem (mean / p98) |
|---|---|---|---|---|---|---|---|---|
| M4 Pro CPU, 14 threads | 98.1% | 97.9% | 98.3% | 92.1% | 19,278 | 0.2 / 2.7 ms | 2,565 | 1 / 6 ms |
| M4 Pro GPU | 98.1% | 97.9% | 98.3% | 92.2% | 22,734 | 0.2 / 1.4 ms | 3,300 | 5 / 14 ms |
| Ryzen AI Max+ 395, 32 threads | 98.1% | 97.9% | 98.3% | 92.2% | 38,158 | 0.1 / 1.1 ms | 3,344 | 1 / 2 ms |
| Radeon 8060S | 98.1% | 97.9% | 98.3% | 92.1% | 34,121 | 0.1 / 1.1 ms | 6,225 | 5 / 8 ms |

Searching near the straight line and straightening cost 7% of batched throughput on the Ryzen and 14% on the M4 Pro. Idle threads now help the longest searches along ([below](#how-the-speed-came)), which more than pays it back. In three alternating rounds against the version before helpers, the Ryzen's batches plan 21,796 against 18,116 problems/s (22,806 against 17,869 on VAMP's spheres), with every path the same. On the M4 Pro they plan 19,898 against 15,779 in three rotating rounds; in six rotating rounds of 15 batches of each set, all the sets' batches take 122.1 ms against 154.4. Single problems plan as fast as before at the median (0.126 against 0.126 ms on the Ryzen, 0.172 against 0.173 on the M4 Pro) and within 3% at the 98th percentile (1.39 against 1.39 ms on the Ryzen, 1.48 against 1.46 on VAMP's spheres, 1.69 against 1.74 on the M4 Pro). The slowest plans in 21.2 ms against 30.9 on the Ryzen, and 26 against 46 on the M4 Pro.

Planning runs on each device's own CPU threads, so the devices plan alike. IK runs on the device, where the GPU pays a fixed cost per call (milliseconds for one problem) that the CPU does not.

The motions stay within the Panda's limits: over the sets, median motions take 1.3–3.2 s, peak accelerations reach 4.9 rad/s² against a 5 rad/s² limit, and peak jerks 44–46 rad/s³ against 50.

Planning alone clears the meshes in 92.9% of the problems. IK + plan clears them less often, in 92.1–92.2%: its goals are chosen on the spheres too, so some of them touch the meshes.

## GPU and CPU throughput

`examples/bench.rs` solves IK and plans for 512 random scenes, each a table with 2–6 boxes and one top-down grasp target:

- **IK:** 64 seeds per target and 30 iterations at most, 4 of the seeds past the first 8 (the defaults);
- **planning:** default `PlanOptions`.

Every device solves the same 484 of 512 targets and plans all 484.

| Machine | IK seeds/s, GPU / CPU (64 per target, 4 of them past the first 8 steps) | Planning, problems/s | End to end, GPU | End to end, CPU |
|---|---|---|---|---|
| Ryzen AI Max+ 395 and Radeon 8060S | 2,240,000–2,630,000 / 1,027,000–1,061,000 | 263,000–362,000 | **29,857–35,078 problems/s** | 14,341–14,981 problems/s (32 threads) |
| M4 Pro | 939,000–1,564,000 / 609,000–833,000 | 148,000–156,000 | **12,679–20,032 problems/s** | 8,485–11,408 problems/s (14 threads) |

Each range covers three runs, each started with the machine's load under 6, with IK that screens its seeds ([#34](https://github.com/natsuku-lab/batchplan/pull/34); before it, with every seed taking every step, the Radeon solved 17,171–19,602 problems per second and the M4 Pro's GPU 7,771–8,254). The M4 Pro's three ran one after another as its load fell to 6, and the first is its slowest. The first run after the GPU idles is up to 30% slower, and the M4 Pro's GPU, which also drives its display, varies with what is on screen.

Uploading the 512 worlds, which the example times first, takes 0.1 ms on the M4 Pro's CPU and 0.2 ms on its GPU: preparing each obstacle for the CPU's checks and, on the GPU, one buffer write. A distance grid adds its values, uploaded once however many worlds share it.

By batch size, end to end, in solved problems per second (medians of three runs, in the sessions of the table above; both tables were measured before the kernels' loops were [written out](#how-the-speed-came), which makes a GPU's IK quicker again):

| Problems per call | M4 Pro GPU | M4 Pro CPU | Radeon 8060S | Ryzen CPU |
|---|---|---|---|---|
| 1 | 83 | **2,583** | 292 | **3,145** |
| 8 | 1,263 | **4,602** | 1,959 | **6,617** |
| 64 | 3,796 | **7,140** | 8,289 | **11,755** |
| 512 | **14,676** | 10,135 | **33,196** | 14,588 |
| 4,096 | **20,622** | 10,441 | **37,740** | 14,422 |

A GPU call costs 3.4 ms on the Radeon (12 ms on the M4 Pro), so one problem at a time runs 11–31 times faster on the CPU. On the M4 Pro that cost varies from run to run and with how recently the GPU worked: one problem's IK took 8 to 20 ms within one session on batchplan 0.1.0 and 11 to 13 ms in these runs, one call after a warm-up, while the 2,600 problems' calls, one after another, take 5 ms each end to end. The CPU fills its cores by 512 problems. The GPU keeps gaining, overtakes the CPU between 64 and 512 problems, and at 4,096 solves 2.0 (M4 Pro) to 2.6 (Radeon) times as many problems per second. These tabletop reaches are easy, mostly straight lines: planning all 484 takes a few milliseconds, so IK sets the pace, and the GPU speeds IK up 1.5 (M4 Pro: 30 against 44 ms for 512 targets, the middle runs) to 2.5 times (Radeon: 13 against 32 ms).

## Tool paths

`examples/tool_paths.rs` times `follow` on the Panda from its ready pose: 64 straight 10 cm moves in random directions, and 64 pours (half a radian about a random axis while the hand moves 3 mm). A move at a time on a one-thread device is the median over the 64 of each one's median of five; a batch of 64 on every thread, the median of 200 calls after 20.

| Machine | One move, one thread | 64 moves | 64 pours |
|---|---|---|---|
| Ryzen AI Max+ 395, 32 threads (at 2c98826) | 0.042 ms | 0.27 ms | 0.20 ms |
| M4 Pro, 14 threads (at bf2f5b9, the median of three runs) | 0.035 ms | 0.37 ms | 0.24 ms |

Against the version before the audit that pinned the spline to the joint limits, weighed rotation in placing a sample along the path and kept the search's buffers as fields, in five alternating runs on the Ryzen: a move on one thread takes 2% longer (0.0416 against 0.0407 ms), batches of 64 moves 7% less time (0.265 against 0.284 ms), and pours 36% less (0.197 against 0.308 ms), since they no longer stop on the way. Counted with `perf stat`, a call of the 64 moves on one thread runs 0.3% more instructions (53.1 million against 52.9 million): the rest of the 2% is fewer instructions per cycle. Along the way, `perf` showed batches 7% slower while each search's buffers lived until the caller dropped it: 32 threads' allocations all freed on one thread, which glibc does slowly. A search now frees them on its own thread when it finishes.

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

The pick that fails has every grasp blocked by clutter or out of reach. A pick's time is mostly IK at the grasps (24 problems of 64 seeds each), which the GPU speeds up from 16 tasks on. A place solves IK for the robot alone in one batch: where the hand lets go fixes where the object is, so each task's solutions are then checked holding its object, on the CPU's lanes. Holding each object in its own IK call took 198 ms for 16 places on the M4 Pro's GPU, against 25 ms at 2c98826. Each held object's candidates are a batch of their own at every step, and the batches run in parallel: run one after another, 16 places took 15.9 ms on the Ryzen and 21.5 ms on the Radeon, in alternating runs with the same plans.

## Joining motions

`examples/join.rs` [joins](planning.md#joining-motions) pick-like motions on one arm in free space: a transit planned from home or the last pre-grasp to a random pre-grasp configuration, then the hand's straight move in along its z (5 cm on the reBot, 3 cm on the SO-101); or the straight move back out, then the transit to the next pre-grasp. Each sequence is timed with its motions one after another, and joined; the Ryzen AI Max+ 395, 32 threads, three runs each started with the machine's load under 6.

| Blend radius | reBot B601-DM, 60 sequences: joined | Sooner, mean (most) | SO-101, 34 sequences: joined | Sooner, mean (most) |
|---|---|---|---|---|
| 2 cm | 29 | 1.9% (8.8%) | 29 | 4.7% (15.2%) |
| 5 cm | 52 | 5.8% (15.0%) | 33 | 8.4% (17.0%) |
| 10 cm | 45 | 5.6% (19.9%) | 29 | 7.1% (18.3%) |

No sequence finished later than its motions one after another. Joining a batch took 15–23 ms for the reBot's 60 and 7–9 ms for the SO-101's 34. The joins that keep their stops turn back on themselves or turn tightly at speed, where a blend is no sooner. Wide blends through sharp turns lose to the timing: TOPP-RA's profile through them is sooner than stopping, but keeping jerk within its limit, by smoothing the profile, slowing it where the smoothed map overshoots and stretching what remains, gives the gain back, so such corners keep narrow blends.

`join` splits paths so their control points are spaced alike, and timing used to slow finely split paths: in `a_split_path_takes_as_long`'s three planned Panda paths, split twice, by 21–41%. Where every time map sized by the path needs more than 10% stretch, timing now also tries maps whose spans last about as long as the jerk limit takes to swing acceleration, and past 30% lets them choose where the profile slows; those paths then timed within 3% of the whole. Against the previous commit, on one Ryzen thread (`examples/timing.rs`, three alternating runs):

| | Paths | Duration | Timing each |
|---|---|---|---|
| Planned from rest | 438 | the same, 2.323 s on average | the same, about 350 µs |
| Replanned while moving | 61 | the same | the same, about 745 µs |
| The hand's straight moves | 121 | 1.058 s, against 1.080 (2.1% sooner) | 250 µs, against 220 (14% more) |

Some straight moves finish later than before, while more finish sooner. In the [handovers](#bimanual-handovers), 32 of the 510 episodes changed length on the Ryzen's CPU device: 14 longer, by at most 2 frames, and 18 shorter, 9 frames shorter in all (`--resets` writes `episodes/length.npy` at both commits). In four alternating runs, with the machine's load at 5–13, the change ran 420 against 418 episodes/s on the Radeon and 369 against 354 on the Ryzen, the same 510 of 512 with a 12.8 s median episode.

## Replanning while moving

`examples/timing.rs` replans 61 planned Panda motions to new goals from halfway along them, where joints move at up to their limits (about 3 rad/s together), and times them with `Trajectory::moving`. Timing used to give up on such a start when its smoothed time map overshot a limit: the map could not be stretched without slowing the start, and lowered caps near the start left its speed out of reach. It then fell back to uniform timing at the pace the path's first span was fitted to, 18–23 s for most. Since then, where that first try finds no timing, a second one gives the start the full limits, easing over two path spans to those the rest keeps, stretches the map with its first points set again for the start's speed, and ends the refinement with what the earlier rounds found where a slowed profile cannot start at that speed. Starts the first try times keep their timing exactly; the cost is the second try, on the starts the first gave up on. Against the previous commit, on the Ryzen, three alternating runs:

| | Before | Now |
|---|---|---|
| Replanned while moving, mean duration | 12.93 s | 3.76 s |
| Timing each, one thread | 735 µs | 1,160 µs |
| Full-speed reaches replanned to their goals (`tests/device.rs`) | 18 of 24 | 24 of 24 |
| 96 Panda motions replanned from 30–70% along them (`tests/trajectory.rs`'s construction), planned | 75, some unable to run from their starts | 85, every one runnable |
| The same, as one batch on 14 threads | 68 ms | 59 ms |
| The same, as one batch on 32 threads | 48 ms | 59 ms |
| The same, one at a time | 8.0 ms | 6.6 ms |

Paths planned from rest (2.3 s on average then) and the hand's straight moves time exactly as before, and the handover episodes are the same bit for bit on both devices, at the same pace (466 against 464 episodes/s on the Ryzen, 476 on the Radeon, four alternating runs). `plan` also no longer returns a path from a moving start that cannot run from it: where momentum carried a joint out of its range in the first span, it tries longer first spans, and otherwise leaves the problem unsolved. On 32 threads the batch waits on its longest search, which that lengthens, so it no longer gains from the threads beyond 14.

## Timing up to the limits

Time-optimal timing used to keep 5% under every velocity and acceleration limit, to leave room for smoothing its profile into the time map, and then stretched the whole map by what the smoothing still overshot. On a straight line that overshoot is where the motion brakes into its end, and the rounds meant to cure it lowered speed caps the profile was already under there, so the whole motion ran about 7% slower for it. Now the profile uses the full limits; where the smoothed map overshoots one, it is solved again going slower than it did there, half a map span either side; and the stretch takes out only what remains. Every trajectory still keeps its limits along the whole curve (`Trajectory::check`).

Straight lines most of the way across the Panda's front took 1.20–1.22 times as long as their slowest joint needs at its velocity and acceleration limits alone; they now take 1.13–1.17 times (1.15 on average), the rest being the jerk limit's (`tests/trajectory.rs`). `examples/timing.rs` and the 2,600 problems against the previous commit, on the M4 Pro, three alternating runs:

| | Before | Now |
|---|---|---|
| Planned from rest (438), mean duration | 2.323 s | 2.171 s (6.6% sooner) |
| The hand's straight moves (121) | 1.057 s | 1.025 s (3.0% sooner) |
| Replanned while moving (61) | 3.763 s | 3.548 s (5.7% sooner) |
| Timing each, one thread: from rest, straight moves | 302, 201 µs | 273, 189 µs |
| The 2,600 problems' motions, per set | | 2.2–7.1% sooner, 5.0% on average |
| The 2,600 problems, `ik+plan` per second, 14 threads | 1,733, 2,441, 2,421 | 2,437, 2,465, 2,426 |

Paths, their lengths and what is solved are unchanged, and more [joins](#joining-motions) are worth blending: 29, 52 and 45 of the reBot's 60 at 2, 5 and 10 cm, where 20, 47 and 40 were. The rounds that slow the profile now change it where they used to leave it, which made a moving start, whose rounds are twice as many, cost 32% more to time. Three changes took that back and more, measured on 1,393 replanned paths `plan` returned for the Panda and the SO-101 (the replans of `examples/timing.rs` and of `paths_planned_from_a_moving_start_run_from_it`, and two wider sets built as the latter's: 300 replans from 5–95% of the way along). A moving start is timed once, with its start given the full limits, where a first try without that came before. Its candidate time maps are turned down at a quarter of the checking density, and only one that keeps the limits there is measured at the full one, whose samples include those; a path no map fits that way is timed again with every candidate measured at the full density, which 46 of the 1,393 need. And a map being stretched with its start set again is given up once a stretch leaves more to take out than the one before. A moving start's map, and a stop's, is also measured either side of where it crosses the path's knots, where acceleration has corners and jerk jumps that even samples miss. Sampled 40,000 times each for velocity and acceleration and 4,000 for jerk (by differences), none of them passes 1.001 times a limit, at their start speeds or at 0.8, 0.9 and 1.2 times them; without those samples 57–63 do (5 of the 90 that can be timed at 1.2), jerk by up to 12% and acceleration by 0.35%. Stops of the 438 planned Panda motions at each hundredth of their first 60%, sampled 2,048 times per span of their maps: 191 of 24,534 passed 1.001 times a limit, acceleration by up to 20% and jerk by 19%, and none does. The 1,393 take 0.2% less time on average than they did; three take over 10% longer (11.5% at most). Last, a map is measured over the joints in contiguous rows, and each order's largest ratio alone takes the roots: the same numbers, bit for bit, in less time. On one of the Ryzen's threads, three rotating runs each started with the machine's load under 6 (medians), timing took 344, 249 and 1,172 µs before; it takes 314, 234 and 475 µs, for the durations above (1.026 s for the hand's moves, and 3.548 s from a moving start). `plan` from a moving start times every candidate path, and most of a Panda's have no timing: 96 Panda replans (the test's construction) take 4.48 ms each one at a time, where they took 5.01 ms before, and 65 ms as one batch (61 ms before: the three runs read 60, 67 and 65 ms against 61, 67 and 61). A path with no timing costs 2.0 ms to turn down where it cost 1.6 ms, and a batch lasts as long as its slowest problem; one that is timed costs 0.6 ms where it cost 1.6. The SO-101's take 1.25 ms each (4.22 ms before) and 26 ms as a batch (74 ms before). Since then a spline that follows the start motion at none of its three start intervals is tried again, stopping at every waypoint, with the timing's second pass at the first interval only, and not at all where that would be the same spline: the same paths over 1,400 replans, and on the M4 Pro (three alternating runs, medians) the Panda's replans take 2.10 ms each where they took 2.89 and 42 ms as a batch where 43, the SO-101's 0.63 ms where 0.75 and 10.5 ms where 16.6. A stop a quarter of the way along a planned motion (`Trajectory::stop`) lasts 0.633 s and takes 0.96 ms to find; it took 1.04 ms before its map was measured at the knots.

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

`examples/handover/` generates the [handover episodes](training-data.md#handovers) for two reBot B601-DM arms 44 cm apart at a table. In each task one arm picks up a block lying in front of it and hands it over between the arms, and the other arm sets it down in front of itself. The blocks are 24–28 cm long, in four sizes; the block, meeting and place poses vary in position and yaw, and 0–2 small boxes stand elsewhere on the table. Three tables in four are divided down the middle, a quarter each by a low wall (6–8 cm) that the block is lifted over, a screen 16–18 cm tall with an opening 8.5–10.5 cm wide, and a row of posts as tall. The block cannot be lifted over the tall ones, so the hands meet at one end of their reach and the block lies, and is set down, toward the other: it is carried along the divider to the gap and back. Planned without its divider, the block alone would pass through it on 267 of the 372 divided tables (79 of 120 walls, 82 of 118 screens, 106 of 134 rows of posts). Half the tasks go left to right. Every step is batched over all the tasks: the giver's `pick`, IK for both hands at the meeting pose, one plan moving both arms there, the straight moves of the handover, and the taker's `place`, whose transit moves both arms (the giver going home); then the episodes' sampling.

| 512 tasks | M4 Pro GPU | M4 Pro CPU, 14 threads | Radeon 8060S | Ryzen CPU (32 threads) |
|---|---|---|---|---|
| Handovers | 510 of 512 | 510 of 512 | 510 of 512 | 510 of 512 |
| Time | 0.86–0.87 s | 1.06–1.07 s | 0.81–0.90 s | 0.97–1.19 s |
| Episodes/s | 584–596 | 478–483 | 566–628 | 428–525 |

The 510 episodes hold 111 minutes of motion at 30 fps (110 on the CPU devices and on the M4 Pro's GPU), 12.7 s each at the median. The two tasks lost fail IK at the grasp. The times are three runs on each device, each started with the machine's load under 6, the M4 Pro's at bf2f5b9. Against 2c98826, which planned the moves in and away and lifted and lowered the block unchecked, alternating runs took 1.14–1.16 s on the Radeon and 1.12–1.15 s on the Ryzen, with the same 510 handovers and a 12.8 s median episode then.

Searches draw up to 30,000 samples, against `plan`'s default 10,000: with the block clear of the other hand, the hardest moves of both arms at once, past a divider, need more, and five tasks were lost without them. IK where the taker lets go keeps 4 of its seeds, since it starts from where the carry starts: with 32, the run was 0.18 s slower on the Radeon and 0.21 s on the Ryzen. The blocks come in four sizes because each size needs its own carried spheres and so plans in batches of its own: at 0.2.0, sizes drawn from a continuous range took 3.65–3.67 s on the M4 Pro's GPU, against 1.52 s.

Rendering the episodes' camera videos in MuJoCo (`scripts/replay_mujoco.py`, three 640×480 cameras) takes about 14 ms a frame on the M4 Pro: 85 s for 16 episodes (6,075 frames, on an earlier version); 16 episodes take 0.13 s to generate.

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
- **Searches built on the threads that run them.** `plan`, `rrt::connect` and `shortcut::shortcut` built every problem's search on the calling thread. The memory a search grew as it ran then belonged to the caller's allocator arena, and every thread of a batch waited for that arena's lock: on the Ryzen, adding threads beyond four made a batch of easy reaches slower. Each search is now built by the thread that runs it (`edges::Deferred`). In alternating runs on the Ryzen, with the same paths, `bench.rs`'s 484 reaches plan at 226,000–310,000 problems/s rather than about 67,000, and the 2,600 problems at 31,196 rather than 23,014 (28,362 rather than 21,997 on the Radeon's device). On the M4 Pro, in three alternating rounds with every path the same, the 2,600 plan at 20,244 against 19,619 problems/s on VAMP's spheres and 19,194 against 19,603 on cuRobo's, within the runs' spread.
- **Bun's build practices, measured.** Bun, now mostly Rust, builds with one codegen unit, `panic = "abort"`, ThinLTO, profile-guided optimization and mimalloc. In CPU planning on the Ryzen, one codegen unit and abort changed nothing, fat LTO cost 5%, and PGO, trained on other scenes, gained 1%. mimalloc as the allocator gained 18% before the buffer reuse and 10% after it (3.5% on the M4 Pro). With searches built on the threads that run them, it no longer does: 34,939 against 33,895 problems/s at the median of three alternating runs, within their spread, so the README no longer suggests it.
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

**Fewer and cheaper checks, sooner trajectories.** Measured against the comparison's frozen baseline (a237326), each change in interleaved runs on the tuning problems, then the whole on the 1,818 held-out ones ([above](#before-and-after-optimization)): paths 22% sooner per query, trajectories 50% sooner, six fewer of 27,270 answers valid.

- **A straight line's spline is not checked again.** Simplification leaves only the straight line once it is free keeping the shortcuts' clearance, and a straight line the search found was checked keeping a wider one, so the spline along it, which runs on it, needs no check of its own. Outputs are the same, and single queries take 8% less time (M4 Pro).
- **Repairs a step at a time.** A spline no fit validates was optimized for ten L-BFGS steps (about 5 ms on one thread) before its check. The optimizer now resumes where it stopped, and the repair checks each step's spline, keeping the 2 mm margin; most pass after one to three steps. Of the 1,560 tuning answers, the 9 repaired ones change, as valid. One-core p99 on the held-out problems fell from 6.8 to 3.1 ms.
- **TOPP-RA: each stage's own constraints once.** A stage's acceleration lines are ordered over an interval of x that does not depend on the next stage, so it is found once per time map, and the passes check only the lines the next stage adds (15 pairs instead of 64). Smoothing candidates share their round's profile, so one needing more stretch than the round's best takes longer, and its assessment stops there. Each stage's bounds are built once, not once per pass. Trajectories are byte-identical, and timing one takes half as long.
- **Where the time went.** On the 21 hardest tuning problems on one core, `perf` put 59–63% of the time in the SIMD edge checks and 13% in drawing samples, most of it computing Halton points. With every core busy, each query takes longer than alone because the Ryzen runs at 4.0–4.4 GHz rather than 5.0; its instructions per cycle barely change (4.25 against 4.37).
- **Joints about z turned in the plane.** Every Panda joint turns about its own z axis. Forward kinematics now multiplies the parent's frame by the joint's fixed origin and turns two of the result's columns in their plane, 12 operations, rather than building the joint's rotation from three fixed matrices first: eight configurations' frames take 139 ns rather than 178, and queries take 4–6% less time for paths and 1–2% less for trajectories, with the same answers.
- **Halton points counted up.** Each sample's Halton point divided its index by each prime base, digit by digit. The sampler now keeps each base's digits and adds one, carrying as an odometer does: the same points bit for bit, and the 21 hardest problems on one core in 1,322 ms rather than 1,412.
- **Obstacles ruled out from a compact array.** Ruling an obstacle out by the sphere around it read the obstacle's whole prepared form. The spheres now sit in an array of their own beside the obstacles, four to a cache line: the same answers, and queries take 5% less time for paths and 1% less for trajectories.
- **A shorter search keeping the clearance.** RRT-Connect keeps its clearance for 500 samples rather than 2,000 before it goes on without it: few searches that find a path keeping it need more, and every search that relaxes spent all 2,000 first. On 16 cores, batches of the tuning problems' paths take 4–6% less time, and final-sized sets holding an unsolvable problem 7% less; on one core, three seeds' batches take 2.8% less. 80 of 1,385 paths change, 0.2% shorter in all, one more problem is solved, and 4–5 fewer answers clear the meshes. Searches that find no path give up after 10,500 samples rather than 12,000.
- **Not adopted.** Checking splines at the edges' spacing alone, without `validate_substeps` per span: 6% faster on the M4 Pro, but on the held-out problems one spline dipped 1 mm into an obstacle between its samples. Searches taking turns, a slice of samples each, so that the slowest start sooner: final-sized batches 19–37% slower. Obstacles sorted by distance with an early exit, as VAMP checks them: checks 3–8% slower, since the table beside the base never lets them stop early. Obstacles grouped under shared bounding spheres: no faster than the compact array.

**GPU kernels.** GPU trajectory optimization on the Radeon went from about 1,550 to 22,800 seeds/s (15×), and on the M4 Pro from 1,220 to 7,660 (6.3×), in six steps, each measured with alternating runs of the old and new builds:

| Change | Radeon (seeds/s) | M4 Pro (seeds/s) |
|---|---|---|
| Skip link pairs whose bounding spheres are apart (self-collision) | 1,550 → 1,975 | 1,220 → 1,440 |
| Write the kernels per robot, so no private array is indexed at run time | 1,980 → 9,450 (4.8×) | 1,610 → 4,450 (2.7×) |
| Skip a link's spheres when an obstacle is beyond its bounding sphere | 9,550 → 15,000 | 4,480 → 7,025 |
| Check validity without computing gradients | +7% | +1.5% |
| Run per-path passes as cooperative workgroups | 15,900 → 22,800 | 7,130 → 7,660 |
| Give small batches several invocations per configuration | batch-1 latency 21 → 16 ms | 44 → 29 ms |

- **Lanes for IK.** Up to 4,096 configurations, every kernel gave each configuration 8 invocations. For IK, how far that pays depends on the robot. IK of 32 to 512 problems (32 seeds each) on a tabletop, with 1 and 8 invocations per configuration, crossed over at:

  | Panda's spheres | Radeon 8060S | M4 Pro |
  |---|---|---|
  | 61 (cuRobo's) | about 1,500 configurations | about 2,100 |
  | 128 (fitted) | about 3,300 | about 5,500 |
  | 507 (fitted) | about 16,000–20,000 | past 16,384, where 8 were still 35% faster |

  The SO-101 (64 spheres) crosses over near 3,500 on the M4 Pro: switching light robots sooner made it 44% slower at 2,080 configurations, so the 4,096 bound stays, and IK keeps 8 invocations past it up to 25 configurations per sphere. Heavy robots gain there: 8,192 configurations of the 507-sphere Panda took 1,009 ms with 1 invocation and 498 with 8 on the M4 Pro. IK without the push away from obstacles checks no collisions, so it never splits: 8 invocations only repeat the kinematics. That IK is what tells why picks' failed grasps failed, which 0.7.0 runs on fewer problems than 0.6.1 did: 114 for 16 Panda picks (3,648 configurations), which took 13–16 ms with 8 invocations on the M4 Pro and about 5 with 1. 16 picks (`examples/tasks`, best of five calls, median of three alternating runs) took 62.5 ms on the M4 Pro's GPU in 0.6.1, 73.5 ms in 0.7.0 and 61.4 ms now; 38.1, 44.9 and 39.1 ms on the Radeon.
- **Per-robot kernels.** On Mesa's RADV, per-link arrays cost 3–5 KB of scratch memory per invocation; writing the kernels per robot removed it ([why](architecture.md#the-gpu)).
- **Grids.** [Building distance grids on the GPU](perception.md#building-grids-on-the-gpu), with an integer distance transform, made a 640×480 depth image's 1 cm grid 11× faster on the Radeon (21.1–22.7 → 1.9–2.0 ms) and 3.3× faster on the M4 Pro.
- **Mesh checks.** A branch-and-bound search took mesh clearance from 620 to 110 µs per Panda configuration.
- **Loops written out.** naga counts the rounds of every shader loop, so that none can run forever. For IK's linear algebra, loops of six or seven rounds around a multiplication, the counting cost more than the work. One hand's damped least squares is written out term by term for each robot (`least_squares_wgsl`), and the kernels' loops of a known count, over the joints, the hands and the line search's steps, are written out when the kernels are built (`loops_written_out`); the source keeps its loops. `examples/latency.rs` on the M4 Pro, the build before the four changes listed here and the one after them in turn (four rounds, each started with the load under 5; medians), in milliseconds:

  | Targets per call | No spheres | Spheres, no push | Spheres and the push | Two hands that share joints, the push |
  |---|---|---|---|---|
  | 1 | 1.14 → 0.34 | 1.23 → 0.39 | 2.03 → 0.92 | 4.31 → 3.59 |
  | 10 | 1.13 → 0.34 | 1.24 → 0.40 | 2.89 → 1.81 | 5.73 → 4.74 |
  | 100 | 1.41 → 0.61 | 1.70 → 0.85 | 5.59 → 4.38 | 8.91 → 7.65 |
  | 1,000 | 3.83 → 1.72 | 5.93 → 2.98 | 30.0 → 22.1 | 52.0 → 42.2 |

  Its 16 plans with 4 optimized seeds take 15.9 ms where they took 23.7, its 256 take 83.1 ms where 93.8, and its UR5e's 1,000 targets among clutter 23.7 ms where 30.0 (that row in three rounds of its own). A round now and then reads 0.8 ms or more for one target without spheres, in either build: the M4's GPU varies over calls this short. On the Radeon 8060S, measured before the two changes that follow and on other random targets (three rounds, each started with its load under 6), 1, 10, 100 and 1,000 targets without spheres take 0.18, 0.20, 0.37 and 1.22 ms where they took 1.12, 1.14, 1.34 and 3.85, and with the push 1.10, 0.97, 1.84 and 10.4 ms where 2.70, 2.51, 3.51 and 17.3. A GPU's results move in their last places, IK's and trajectory optimization's alike.

  The kernels keep every check naga makes. For a while they were built without the loop counters, through wgpu's `unsafe` way to vouch for a shader, which made IK 10–28% faster then. What the counters still cost, from this build and one that differs only in vouching for its shaders, in turn (three rounds of the same example on the M4 Pro): nothing for IK without spheres, for `evaluate` and for a Panda of 399 spheres; up to 4% for IK with spheres (22.1 against 21.3 ms for 1,000 targets with the push); 4–7% for optimized plans (83.3 against 79.2 ms for the 256, 128.7 against 120.3 for 256 past a ball in the way); and 13–20% for two hands that share joints (42.2 against 37.2 ms for 1,000 targets), whose stacked system keeps loops of run-time counts.

  Three of the example's kinds are slower than the build before, in the four rounds of the table. IK with the push for a Panda fitted with 399 spheres, among tabletops' clutter, takes 191 ms for 300 targets where it took 184, and 469 ms for 1,000 where 446: that came with the written-out algebra, the counters still off, and its cause is not found. 256 plans to goals past a ball, with 4 optimized seeds, take 129.3 ms where 123.5, which is the counters. `evaluate` of 16,384 configurations reads 1.23 ms where 1.19 (1.5–4% over three sets of rounds).
- **A link of many spheres takes them two to a round.** Where a link of the robot's own has 24 spheres or more, or two of them 128 sphere pairs, the loop over them takes two rounds at a time, the second where there is still one to take: half the rounds for naga to count, and the same spheres in the same order. It brought the 399-sphere Panda above from 204 ms to 191 for 300 targets, and from 491 to 471 for 1,000 (three rounds in turn). The two counts are where no robot of the example is slower than with one to a round (IK with the push, 1,000 targets, three rounds in turn): its UR5e among clutter takes 23.7 ms where it took 24.5, and the Panda, none of whose links reach them, is unchanged. From 16 spheres and 64 pairs the Panda's hand, 18 spheres and 72 pairs with each of four links, pairs up and its IK takes 22.0 ms where 21.8; from 24 and 64 the UR5e takes 24.7 where 24.5. The objects a robot holds keep one to a round whatever their size, so that holding another builds no kernels.
- **Sphere pairs out of reach.** A link pair whose bounding spheres are within reach had every sphere of the one checked against every sphere of the other, so the collision push of an IK step, and each sample of trajectory optimization, cost a robot of many spheres as the square of their number: one `place` of four holds and eight release poses took 4.2 ms on a Panda fitted with 128 spheres, 7.4 ms with 256 and 19.9 ms with 512, against 3.3, 4.3 and 7.8 ms without the push. In the CPU's scalar code, where the clearances are not read (the IK step and the optimizer, on either device; `evaluate` reads them and checks every pair as it did), a sphere past the margin from the other link's bounding sphere now skips its pairs with that link's spheres. Such pairs add nothing, so the cost and its gradient are the same to the bit (`sphere_pairs_out_of_reach_add_nothing_to_the_cost_or_its_gradient`). That `place` on the 512-sphere Panda takes 10.8 to 11.1 ms where it took 17.7 to 18.7, and 5.2 to 6.3 ms where 6.5 to 7.0 with 256 spheres (medians of fifteen calls, three rounds in turn on an M4 Pro at a load of 10). Instructions retired settle what a busy machine's times cannot (`/usr/bin/time -l`, twice each, within 0.3%): five of that call with the robot's fitting, 19.7 billion where 33.8 at 512 spheres and 12.4 where 15.1 at 256; five `solve_ik` calls for 200 targets on one thread for the repository's Panda of 60 spheres, 9.12 billion where 10.83 with the push, 19.63 where 21.06 for two hands that share joints, and 4.09 both without the push, which checks no pairs; twenty `evaluate` calls of 4,096 configurations among tabletops on one thread, 1.51 billion both, and 29.1 where 29.4 for a Panda fitted with 400 spheres. The kernels still visit every pair of a link pair within reach.
- **A grid a link is far from.** A link whose bounding sphere keeps away from an obstacle skips its spheres' checks against it; against a distance grid, whose values are interpolated, it never did, and every sphere was looked up in every grid. A point outside a grid's box reads the value at the nearest point of the box plus its distance to that point, so the least value on the box's faces bounds what a link outside the box can read, and the box rules the grid out as an obstacle's own distance rules it out (`cpu::grid_gap`, in the SIMD checks and the kernels alike). The rule is compiled into copies that only worlds uploaded with a grid run: kernels of their own on a GPU (`GpuBackend::kernels_in`, for an upload that holds a grid), built beside the others, and on the CPU a second copy of each check, chosen where a call or a search is dispatched (`cpu::collision_with`). Written into the code every world runs, it moved calls that have no grid. On the M4 Pro's GPU, IK with the push became 9% quicker at 100 and 1,000 targets, 16 plans 4% slower and a UR5e's IK 5% slower; on its CPU, IK for two hands 7% slower once a second copy inlined beside the first kept the step out of its loop, and 16 plans 7% slower. So worlds without grids run what they ran: the kernels' generated source is the same to the byte, and the CPU's hot loops compile to within a twentieth of the size they had (the IK loop 13.6 KB where 13.5, a search's 35.1 KB where 36.6, one helper now out of line). `examples/latency.rs`'s IK for 1,000 targets among cluttered tabletops, the build before and this one in turn on the M4 Pro (four rounds on its GPU, twelve on its CPU), in milliseconds:

  | The tabletops' clutter and | GPU | CPU |
  |---|---|---|
  | no grid | 29.9 → 30.0 | 62 → 63 |
  | the grid of a ball 2 m behind the arm | 35.7 → 28.8 | 107 → 67 |
  | the same ball beside the table, where the arm reaches | 35.7 → 31.7 | 108 → 72 |

  The example's other rows run the same kernels as before on the GPU, and over the ten rounds none of their medians is more than 1.3% slower than the build before, but for IK without spheres at 10 targets, 0.347 ms where 0.328. The example times that row 20 to 40 ms after its device is made; the same call repeated from there settles at 0.317 ms on this build and 0.316 on the one before (eight rounds in turn), and what the first sixty calls after a device is made take varies more from one session to the next than between the builds (0.49 against 0.34 ms in one, level in another). On the CPU the rows without a grid read a few percent either way in each of two sessions on a busy machine (twelve rounds in turn, then five), four of them slower in both by more rounds than not, which their times cannot settle. What the processor does can: five `solve_ik` calls for 200 random reachable targets in an empty world, with the options' defaults, on a CPU device of one thread (`Device::cpu_threads(&robot, 1)`), each of the example's robots in a process of its own under `/usr/bin/time -l`, twice each with the counts within 0.3% of each other: this build retires 4.11 billion instructions where the one before retired 4.10 without the push, 10.69 where 10.81 with it, and 20.96 where 21.08 for two hands that share joints. A grid that fills the scene, which every link is inside, costs what it did (`examples/depth.rs` plans in 13 to 25 ms where it took 12 to 20).
- **Kernels compiled side by side.** A driver compiles a pipeline on the thread that asks for it, and the kernels' fourteen were asked for one or two at a time. Each now has a thread of its own, and the kernels of uploads with grids, a second set, are built beside the first. With the driver's shader cache emptied (three runs in turn on the M4 Pro), making a GPU device takes 1.4 s where it took 2.6, `Device::warmup` 0.7 s where 2.0, and a device for the robot holding an object 1.7 s where 3.0; with the cache warm, 35 ms where 40, 10 where 16, and 18 where 24. A device derived again for a shape already built takes the 0.13 ms it took (six rounds in turn). The second set costs memory: a GPU device keeps 66 to 88 MB after it is made where it kept 35 to 37, and about 20 MB for each held shape where 9.
- **One lane takes the step.** A small batch gives each configuration eight invocations, which split its collision gradient. Each then worked out the same step from it: the Jacobian, the least squares, the limits. Now the first does, and the others take the configuration from it; the collision gradient is found before the step either way. The Panda's results are the same bit for bit, and two UR5e arms' move in their last places, the reordered step being rounded otherwise. With the push, 1,000 targets take 22.2 ms where they took 24.8 on the M4 Pro, and 42.4 ms where 45.9 for two hands that share joints (three rounds in turn).
- **Paths that are done.** The CPU's trajectory optimization stops a path at a steepest-descent search that finds nothing better: no later round can change it. A GPU ran every round for every path. Its kernels now leave such a path out, which changes no path: of the 180 optimizations the CPU runs to their end for `examples/latency.rs`'s two batches of plans, 151 end before the last round, at round 60 of 100 on average. The plans' times are with the table above; the Radeon is not measured yet.
- **Not adopted.** Shader modules without naga's checks, which wgpu builds only through `unsafe`: what the loop counters cost is above; without bounds checks as well, nothing more on the M4 Pro and under a tenth on an RTX 4090, with every index to prove in range. Taking a workgroup's strided loops several rounds at a time, or their first rounds out of the loop; one loop over a world's obstacles with every link inside it (4% slower without the counters). Reusing every call's buffers: creating, filling and binding them takes 0.36 ms of 53 ms for 16,384 seeds on the M4 Pro, and large ones are made around their data, which saves a copy. Small IK calls do reuse theirs (up to 64 KiB each), which saves them 0.04–0.07 ms of about a millisecond. Grid buffers, which are megabytes, are reused.

## Reproducing

```bash
# GPU against CPU
cargo run --release --example bench -- 512

# RoboKit's own benchmarks with batchplan as a solver, on an NVIDIA GPU with CUDA 12: RoboKit 736f112 checked out
# beside batchplan (../robokit), with a Rust toolchain and a Vulkan loader (libvulkan1) for batchplan's package
cp scripts/robokit/*.py ../robokit/ && cd ../robokit
uv run scripts/benchmark/ik/run_benchmark.py                                   # RoboKit, cuRobo, PyRoKi
uv run scripts/benchmark/motion_plan/benchmark.py --config robokit-primitive --runs 5 --output-dir out/robokit
uv run batchplan_ik.py --device gpu                                            # or --device cpu, --seeds 16, --iterations 20, --screening 0
uv run batchplan_mp.py --config batchplan-gpu --runs 5 --output-dir out/batchplan-gpu
BATCHPLAN_SEEDS=4 uv run batchplan_mp.py --config batchplan-gpu --runs 5 --output-dir out/batchplan-gpu-4   # batchplan-cpu likewise
BATCHPLAN_JERK=150 uv run batchplan_mp.py --config batchplan-gpu --runs 5 --output-dir out/batchplan-gpu-j150   # timed under RoboKit's jerk limit
BATCHPLAN_FRAMES=31 BATCHPLAN_JERK=300 BATCHPLAN_SEEDS=4 uv run batchplan_mp.py --config batchplan-gpu --runs 5 --output-dir out/batchplan-gpu-4-j300-f31   # sampled as RoboKit's plans are
BATCHPLAN_FRAMES=31 BATCHPLAN_ACCELERATION=0.7 BATCHPLAN_SEEDS=4 uv run batchplan_mp.py --config batchplan-gpu --runs 5 --output-dir out/batchplan-gpu-4-a07-f31   # under 0.7 of the acceleration limit

# Tool paths, and picks and places on random tabletops
cargo run --release --example tool_paths
cargo run --release --example tasks -- 64

# Joining pick-like motions on one reBot (after scripts/fetch_rebot.sh), or the SO-101; timing by kind of path
cargo run --release --example join [-- so101]
cargo run --release --example latency             # one call's time by kind, on the GPU; -- --cpu on the CPU
cargo run --release --example timing

# Tracking on the SO-101's servos, on BAM's model of them
uv venv .venv-so101 --python 3.12
uv pip install --python .venv-so101/bin/python "better-actuator-models[mujoco]==1.0.3" mujoco==3.17.0 numpy==2.5.4
cargo run --release --example so101_tracking -- data/so101_tracking      # --sequences for joined pick-like motions
.venv-so101/bin/python scripts/so101_tracking.py data/so101_tracking      # --kp 32, --vin 7.4

# Two reBot arms handing blocks over (downloads the arm at a pinned commit, with checksums), and their camera videos
scripts/fetch_rebot.sh
cargo run --release --example handover -- 512 [--cpu]
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
scripts/run_suite.sh data/bench/baseline heldout     # the frozen baseline: the held-out problems, at a237326
for d in data/bench/final/*/*/round1; do cargo run --release --example compare -- mesh data/bench/inputs "$d"; done
scripts/report_suite.sh data/bench/final             # every table above
```

<details>
<summary>The earlier comparison's commands</summary>

```bash
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
</details>
