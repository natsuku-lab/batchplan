# Changelog

What changed in each release, for people upgrading, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Cargo's semver](https://doc.rust-lang.org/cargo/reference/semver.html): before 1.0, a new minor
version (0.9 → 0.10) may break the interface, and a patch (0.9.0 → 0.9.1) breaks nothing.
[Contributing](CONTRIBUTING.md#releases) says how releases are made.

## [Unreleased]

### Added

- **Breaking:** `FollowerOptions::replan`, on by default. Off, a `Follower` keeps its
  trajectory's path, as a motion along a tool path must: where the rest of the motion collides,
  it stops on the path, and where the robot is measured past `tolerance`, it holds it there,
  `Blocked`. In either case it used to plan a free-space motion to the goal, off the path. Code
  that writes out every field of `FollowerOptions` needs the new one.

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
