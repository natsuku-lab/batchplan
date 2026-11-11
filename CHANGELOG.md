# Changelog

What changed in each release, for people upgrading, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Cargo's semver](https://doc.rust-lang.org/cargo/reference/semver.html): before 1.0, a new minor
version (0.9 → 0.10) may break the interface, and a patch (0.9.0 → 0.9.1) breaks nothing.
[Contributing](CONTRIBUTING.md#releases) says how releases are made.

## [Unreleased]

### Added

- **Breaking:** `plan` takes the hands' poses as a goal: `PlanProblem::goal` is a `Goal`, a
  configuration (`Goal::Joints`) or a pose or `None` per hand (`Goal::Poses`), and
  `PlanProblem::new` takes either, a `Pose` for a one-handed robot. `plan` solves IK for every
  pose goal of the call in one batch, seeded from each start and keeping its held joints, what it
  holds and its constraints, and plans to the collision-free solution nearest the start up to
  whole turns. Poses no seed reaches collision-free are solved again with four times the seeds, up
  to sixteen times, while a `time_budget` lasts; those still unreached fail as
  `PlanFailure::GoalUnreachable`, or `GoalCollides` where IK reaches them only in collision.
  `PlanOptions::goal_ik` sets that IK. `PlanProblem::new` takes a configuration as a `Vec<f32>`,
  `&[f32]`, `&Vec<f32>`, `[f32; N]` or `&[f32; N]`; convert any other form with `to_vec()`. Code
  reading a goal as a slice reads `goal.joints()`, a struct literal wraps a configuration in
  `Goal::Joints`, and code that writes out every field of `PlanOptions` needs the new one. The benchmark's ik+plan mode plans to the
  goal pose this way. In Python, `PlanProblem` takes a `Pose`, or a `Pose` or `None` per hand, as
  its goal, beside `PlanOptions.goal_ik` and `PlanFailure.GoalUnreachable`.
- Python bindings, in `python/`: the `batchplan` package, built with maturin
  ([Python](docs/python.md)). Robots, assembled ones among them, obstacles and worlds, devices, IK
  (`solve_ik`, `ik_failures`), planning with hand constraints (`plan`), timing (`Trajectory`),
  tool paths (`follow`), whole tasks (`pick`, `place`, `grasps`, `place_poses`) and held objects,
  with NumPy arrays in and out, `batchplan::Error`'s kinds as exception classes, and the
  interpreter released while each call works. Values are frozen and changed with
  `replace(**changes)`, as Rust's struct update syntax does. Install it from the repository with
  `pip install "git+https://github.com/natsuku-lab/batchplan#subdirectory=python"`, which needs a
  Rust toolchain.
- **Breaking:** `PlanProblem::holding`, `IkProblem::holding` and `ToolPath::holding`: an object a
  hand holds (a `Held`, as `TaskPlan::held` gives it), per problem. One `plan`, `solve_ik` or
  `follow` call takes problems holding nothing, different objects, or one object by different
  grasps, and gives each what a device holding its object alone gives: it groups them by what
  they hold and runs a batch per group, in parallel, on the device's GPU or threads. Holding per
  problem took a device and a call per hold. Pass the device without the object; one already
  holding a `Held` in that hand is an `Error::Input`. Code that writes out every field of
  `IkProblem`, `PlanProblem` or `ToolPath` needs the new one (`holding: None`).
- `SphereOptions` implements `PartialEq`.
- **Breaking:** `GraspOptions::overlap`, 1 cm by default: the least the fingers reach into the
  object in a grasp from above over a `floor`. An object flatter than `tip` and the floor's
  clearance is now held from above with the fingertips at the floor and the frame over its top,
  as long as its top is at least `overlap` over the floor. `grasps` left out every grasp from
  above of such an object: an 8.8 x 4.8 x 1.85 cm package got only grasps from the sides, which
  put the hand into the table. Every grasp from above over a floor now keeps at least `overlap`
  of the fingers in the object, where it kept the frame inside it, so with `tip` under 1 cm (the
  default 0 among them) some it returned are left out. Code that writes out every field of
  `GraspOptions` needs the new one.
- **Breaking:** `FollowerOptions::replan`, on by default. Off, a `Follower` keeps its
  trajectory's path, as a motion along a tool path must: where the rest of the motion collides,
  it stops on the path, and where the robot is measured past `tolerance`, it holds it there,
  `Blocked`. In either case it used to plan a free-space motion to the goal, off the path. Code
  that writes out every field of `FollowerOptions` needs the new one.

### Changed

- **Breaking:** IK screens its seeds. Every seed takes `IkOptions::screening` steps (8 by
  default), then only `IkOptions::kept` of each problem's (4) take the rest of `iterations`: those
  not yet at their targets, the nearest and the collision-free first, then seeds that have
  arrived. The others are returned as the screening left them. Code that writes out every field of `IkOptions` needs the two new
  ones, and `screening: 0` is the IK of before, as are problems of more than 32,768 seeds. On a
  GPU the screening, the choice and the remaining steps are one round trip. On an M4 Pro, 1,000
  targets for a Panda with collision spheres take 36–39 ms where they took 68–82, and its CPU
  threads 95 ms where 212, with 945 solved where 944. In
  RoboKit's IK benchmark on an RTX 4090 (a Panda without spheres), 1,000 targets take 3.1–3.3 ms
  where they took 3.6–3.7 and 2,000 take 6.1–6.3 ms where 7.2, while 10 take 0.77–0.80 ms where
  0.71: so few seeds are not sped up by leaving most behind
  ([Benchmarks](docs/benchmarks.md#against-robokit)). The 2,600 pose-goal problems solve as
  before (98.3%). In Python, `IkOptions.screening` and `IkOptions.kept`.
- Batched IK spends less time around the solve: a problem's seeds share its targets instead
  of each carrying a copy, large batches are seeded on the device's threads, buffers go to the
  GPU without an extra copy, and a robot without collision spheres skips the collision push,
  which is zero for it. The same results, bit for bit. In RoboKit's IK benchmark on an RTX 4090
  (a Panda, collision off, 64 seeds of 30 steps), 1,000 targets take 4.5 ms where they took
  10.8, 2,000 take 9.1 ms where 23.2, and one takes 0.9 ms where 1.26
  ([Benchmarks](docs/benchmarks.md#against-robokit)).
- **Breaking:** `IkOptions::collision_step` must be finite: an infinite or NaN one is an input
  error. NaN and negative infinity used to mean no push, and positive infinity solved nothing
  (the CPU left every seed where it started).
- IK on a GPU takes 10-28% less time: the kernels are built without the counter naga adds to
  every loop to rule out infinite ones, which theirs cannot be (the crate's one `unsafe`). One
  target's 64 seeds of 30 steps take 1.01 ms where they took 1.27 on an M4 Pro for a Panda
  without collision spheres (2.83 ms where 3.33 with them), and 0.53 ms where 0.74 on an RTX
  4090 (about 1.2 ms where 1.33). A GPU's results round differently for it, as they do from
  driver to driver ([Benchmarks](docs/benchmarks.md#how-the-speed-came)).
- Trajectories are sooner: time-optimal timing uses the full velocity and acceleration limits,
  and where its smoothed time map overshoots a limit, slows the profile there instead of
  stretching the whole motion for it. Planned Panda motions take 6.6% less time on average, the
  hand's straight moves 3.0%, replans from a moving start 5.7%, and the 2,600 benchmark
  problems' motions 5.0%, within the same limits (`Trajectory::check`); more joins are worth
  blending. Timing costs less than it did (a Ryzen thread): a path from rest 9% less, and one
  from a moving start 59% less (475 µs where 1,172), timed once with its start at the full
  limits, its candidate time maps turned down at a quarter of the checking density, and each
  map measured over the joints in contiguous rows. A moving start's timing and a stop's are
  also held to their limits either side of where they cross the path's knots, between the
  samples checked, where jerk could pass its limit by up to 12%, and in a stop acceleration
  by up to 20% and jerk by 19%: of 960 stops asked early in planned Panda motions 52 are
  refused where 40 were. Replans from a moving start take 11% less time one at a time on a
  Panda and 70% less on an SO-101; a batch of 96 Panda replans takes 65 ms where it took 61
  (within the runs' spread), since a moving start no timing fits costs more to turn down
  (2.0 ms where 1.6). A spline that cannot follow the start motion is then tried again,
  stopping at every waypoint, with less of that timing, which takes a further quarter off a
  Panda's replans one at a time and a sixth off the SO-101's (an M4 Pro)
  ([Benchmarks](docs/benchmarks.md#timing-up-to-the-limits)).
- **Breaking:** `IkOptions::default()` is 64 seeds of 30 steps of at most 1 rad, where it was 32
  seeds of 60 steps of at most 0.3 rad: at 0.3 rad, about half the seeds ran out of steps before
  converging. The same problems solve. Results carry 64 seeds per problem, and `TaskOptions`,
  `DemoOptions`, `PickPlaceOptions` and `PlanOptions::goal_ik` inherit the new IK; pass the old
  values to keep the old results.
- **Breaking:** An IK step that would move no joint by more than 1e-6 rad is not taken, on the
  CPU and the GPU alike: a seed that has come to rest stays exactly there, and CPU threads skip
  its remaining steps, which makes their IK about a fifth faster. Seeds can differ from before
  by a few microradians. Together with the new defaults, the 2,600 benchmark problems plan to
  pose goals 1.8 times as fast in batches on a Radeon 8060S (6,225 against 3,489 per second, and
  in 5 ms instead of 9 for one) and 1.18 times as fast on 32 CPU threads (3,344 against 2,836);
  IK for 512 tabletop targets takes 71 ms instead of 84 on those threads.
- `solve_ik` on a GPU checks the solved configurations' clearances in the same submission, from
  the configurations already on the GPU, and reads everything back at once: one wait per 65,536
  seeds instead of three, about 0.4 ms less per call.
- **Breaking:** `Robot::load`, `Robot::from_urdf` and `Robot::assemble` refuse a collision
  model whose spheres have a non-finite center or radius or a negative one, or whose
  self-collision buffer for a link is not finite or would shrink one of its spheres below nothing,
  naming the link and the value (`Error::Load`; `Error::Input` from `from_urdf`). Spheres fitted to
  geometry that is not finite, or so large it overflows, are refused alike. Such models loaded
  before, and a sphere smaller than nothing checked as farther from what it touched than it was.
- **Breaking:** `Held` is `batchplan::Held`, no longer also `batchplan::task::Held`.
- A device keeps the spheres it fits to held objects, the latest 64, and shares them with the
  devices `with_robot` makes from it. `pick`, `place` and every call holding a cylinder or a
  capsule fitted it again, which takes milliseconds; a box or a ball is covered in microseconds.
- **Breaking:** `FollowStatus::Stopping { contacts }` and `FollowStatus::Collides { at, contacts }`
  name what is in the way, as `Device::contacts` lists it: what the rest of the motion collided
  with where it was first found to, numbered as in the worlds of the tick the stop began at and
  repeated on every tick of the stop while it is clear, and what the robot collides with at
  `at`. A graze a GPU device's check finds just outside what its CPU threads list names the
  nearest within 0.1 mm. They said only that something collided. `Stopping` was a unit variant.
- **Breaking:** `grasps` keeps the open fingers above `GraspOptions::floor` too, `opening` apart
  and reaching `tip` past the frame, where it kept only the frame and the point between the
  fingertips. A grasp tilted about a bar over a support, as a drawer's handle, swung one open
  finger into it: of a bar whose top is 3.7 cm over a plate, every one of the 12 grasps tilted
  from above at 16 turns put a Panda's finger into the plate. They are now held as far out as
  keeps the finger over the floor, and left out where that holds less than `overlap`: the 4 left
  all pick it. Grasps from the side that close upright on an object standing on the table, their
  lower finger below the table's top, are left out too, so an object only those fit, such as a
  box too wide across its top to grasp from above, gets an `Error::Input` where it got them. An
  infinite `opening` is an `Error::Input`.

### Fixed

- A plan from a moving start is optimized (`PlanOptions::seeds`) unless its path is within 2% of
  the straight line from the start: the line was measured from the path's first control point,
  which lies behind a moving start, so paths that bent by half as much again were left
  unoptimized.
- `Trajectory::stop` stops a motion from more of the states it passes through, and sooner. It
  refused where a joint ran within 0.6% of its velocity limit, since a stop's velocity was held
  to the margin under the limit even where it starts, and often while a motion still gathered
  speed, since braking at once breaks the jerk limit there, or stopped only by creeping to the
  path's end. It now also tries carrying on for one span with the motion's jerk, half of it or
  none before braking, and keeps the stop that ends soonest along the path. Trajectories timed
  up to the limits pass through such states more often. Through the first fifth of 48 planned
  Panda motions, 52 of 960 stops are refused where 153 were and one takes longer than going on
  where 66 did; of 19 states at a velocity limit, 9 stop where none did. A quarter of the way
  along 438 motions, stops last 0.633 s where they lasted 0.646, and take 0.62 ms to find where
  they took 0.73 (`examples/timing.rs`, which now times them, on an M4 Pro). A `Follower` that
  must stop where none is found keeps to its trajectory, which comes to rest at the path's end.
- Optimized seeds (`PlanOptions::seeds`) are validated as searched splines are checked: at samples no
  farther apart than `shortcut.resolution`, keeping the margin from the world searched splines keep
  away from their ends (2 mm, or `shortcut.clearance` if less). Validation kept no margin and
  sampled each span `validate_substeps` times however far it moved, so optimization, which pulls
  paths taut against obstacles, returned valid paths that collided between samples, as on picks
  under a table. Their lengths, which `best` compares, are measured at those samples too.
- `plan` refuses a start or a joint goal that is not finite. A continuous joint's range has no
  limits, so a start at infinity there passed the limit check and planned a path of infinite
  control points marked valid.
- `Trajectory::at` and `Trajectory::sample` keep each position between the control points it
  blends, so a path that runs along a joint limit never samples past it. Rounding left samples a
  unit or two in the last place past (up to 4.8e-7 rad), which limit checks with no tolerance
  refuse.
- `pick` builds each holding device where it moves with it again, so building one overlaps the
  other grasps' moves. 0.10.0 built them all first, which cost picks on a GPU device about 1.5%:
  64 picks of the tasks example (`cargo run --release --example tasks -- 64`) took 185.5 ms on
  an M4 Pro and take 183.2 ms now, against 181.5 ms in 0.9.0 (medians of seven alternated runs).
  The rest of that gap is in grasping and moving in, whose code did not change.
- `grasps` refuses an object whose size is negative or not finite, or whose rotation is not a
  unit quaternion, naming it, as uploading a world does.

## [0.10.0] - 2026-11-07

### Changed

- **Breaking:** `Robot::attach` refuses a shape whose size is negative or not finite, or whose
  rotation is not a unit quaternion to within 1e-3, as uploading a world does, naming the shape.
  0.9.0 attached such shapes with spheres that did not cover them. Normalize a rotation read
  with few digits, such as (0, 0, 0.71, 0.71), before attaching it.
- The model's `source` also gives the largest of each link's reach over its own tolerance.
- `Robot::attach` fits a single shape in its own frame and places its spheres by the shape's
  pose, so where it sits on its link no longer changes them, and `pick`, `place` and attaching
  `Held::attachment` give a held object the same spheres.
- `pick` and `place` fit each held object once and place its spheres by each grasp, where they
  attached it at each grasp and fitted it again: picks of cylinders and capsules spend less of
  their time attaching.

### Fixed

- `place` refuses a held object whose size is not finite, as an input error naming its task.
- A pick's contacts where it lifts the object name obstacles as the task's world numbers them:
  every obstacle after the picked one came back one index too low.
- Where the budget reaches every tolerance but cannot hold each link within its own, what it has
  left holds the spheres tighter. It used to go unspent, with spheres reaching up to twice their
  link's tolerance. A stand of two boxes held to 2 cm (`tests/robot.rs`) reached 36.6 mm with 379
  of 512 spheres, and now 24.1 mm with 502; the Panda at 1,200 spheres reached 9.5 mm against
  5 mm, and now 5.8 mm. The search for the least growth makes such fits two to three times as
  long: the Panda's at 1,200 spheres takes 0.20 s, where it took 0.10 s, and the stand's at 512
  0.13 s, where it took 0.05 s.
- Covers are refined until they reach every surface sample, where they stopped after 8 rounds,
  so large boxes keep to their tolerance where the budget allows: the stand reached 25.7 mm at
  1,024 spheres, and now 20.0 mm.

## [0.9.0] - 2026-11-06

### Changed

- Fitted spheres reach less far past thin parts. Each link's spheres reach at most twice its
  tolerance, scaled by what the budget allows, and less as the budget grows. Where the budget
  affords it, each closed link is held within its own tolerance. Parts thinner than 0.1 mm and
  vertices whose averaged normal leads out of the geometry may reach a little farther: at 2,048
  spheres the Panda's base reaches 5.18 mm against 5 mm. The model's `source` names the scale.
  Fitted models change, and saved models load as before.
- Spheres held that tight leave slivers of surface about a millimeter deep outside them:
  `LinkQualification::uncovered` reaches about 2% of a link, where it was a fraction of a
  percent.
- `Robot::qualify` measures on samples the fit did not draw.

## [0.8.0] - 2026-11-06

### Added

- `Robot::relock` holds joints locked at load at new values, and `Device::with_robot` takes the
  relocked robot without compiling kernels or uploading worlds again.
- `FollowerOptions::lead`: time after the planning budget for a replan to reach the robot.

### Changed

- **Breaking:** `FollowStatus::Replanned` carries the new trajectory: `Replanned { at, trajectory }`.
- **Breaking:** `FollowerOptions` gains `lead`, and `CollisionModel` gains `locked_joints`, the
  joints locked while fitting. A model fitted with locked joints must be loaded with them locked
  at the same values, and `relock` refuses to move them. Models saved before 0.8.0 load as before.

### Fixed

- `Device::evaluate` refuses configurations that are not finite, and `Robot::load` lock values
  that are not finite.
- GPU picks at small batches are as fast as in 0.6.1 again: IK that does not push away from
  obstacles no longer splits each configuration's work across GPU invocations, and robots with
  many spheres keep the split past 4,096 configurations.

Earlier releases have no notes beyond their commits.

[Unreleased]: https://github.com/natsuku-lab/batchplan/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/natsuku-lab/batchplan/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/natsuku-lab/batchplan/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/natsuku-lab/batchplan/compare/v0.7.0...v0.8.0
