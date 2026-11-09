# Changelog

What changed in each release, for people upgrading, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Cargo's semver](https://doc.rust-lang.org/cargo/reference/semver.html): before 1.0, a new minor
version (0.9 → 0.10) may break the interface, and a patch (0.9.0 → 0.9.1) breaks nothing.
[Contributing](CONTRIBUTING.md#releases) says how releases are made.

## [Unreleased]

### Added

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
