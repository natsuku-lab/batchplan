# Changelog

What changed in each release, for people upgrading, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Cargo's semver](https://doc.rust-lang.org/cargo/reference/semver.html): before 1.0, a new minor
version (0.9 → 0.10) may break the interface, and a patch (0.9.0 → 0.9.1) breaks nothing.
[Contributing](CONTRIBUTING.md#releases) says how releases are made.

## [Unreleased]

## [0.12.0] - 2026-11-27

### Added

- **Breaking:** one object held by two hands. `Held` gains `also`, the other hands that hold it
  too, each by its grasp (`Grip { hand, grasp }`); a `Held` literal sets `also: vec![]` for one
  hand's hold. The object may touch the links of every hand that holds it, and every call that
  holds it keeps each at its grasp: `solve_ik` places the other hands from one hand's target,
  `plan` takes one hand's goal pose and keeps the hands pinned along the path (a start or a
  configuration goal with a hand off its grasp is an input error), `join`'s blends and a
  `Follower`'s replans keep them too, and `follow` takes the other hands along a path of one of
  them. **Breaking:** `Pick` gains `also`, a second hand that picks the object up with the first
  (`HandGrasps { hand, grasps }`, one grasp per grasp of the pick's, entry by entry, each pair a
  candidate costing both holds' sum); a `Pick` literal sets `also: None`. The plan's `held` names
  the second hand, and `place` sets down what two hands hold: both move in and back off one after
  the other, and the transit, the lift and the lower keep them at their grasps. `handover`
  refuses what two hands hold. In Python, `Held(hand, object, grasp, also=[Grip(hand, grasp)])`
  and `Pick(..., also=HandGrasps(hand, grasps))`.
- **Breaking:** `Held` gains `touch_links`, other links of the robot that may touch what is held,
  by name, beside those of the hands that hold it; a `Held` literal sets `touch_links: vec![]`.
  A hand that has let go of what another still holds backs off with its links there
  (`robot.hand_links(hand)`, as `String`s), and a hand that works on a held object touches it
  the same way. A name that is no link of the robot is an input error. In Python,
  `Held(..., touch_links=[...])`.
- miru's `Memory` takes a hold by two hands (`Held::also`) as each hand's: `Memory::held` gives it
  for either, a pick into either is refused while they hold it, and a let-go or a loss empties
  both. To say a held thing is held otherwise, let it go where it is and pick it as it is held
  then, in one `Memory::after`.

- **Breaking:** a task's candidates carry their own data and a cost. A pick's, a turn's and a
  pull's grasps are `Hold`s: the hand's `pose` in the object's frame, the fingers' `locked` values
  and a `cost`. `Hold::locked` replaces `Pick::locked`, so one pick takes grasps at different
  openings, and `Hold::new(pose, opening, &gripper)` holds a grasp of a surface at its opening. A
  place's poses are `Release`s in `Place::releases`, which replaces `Place::poses`: the object's
  `pose`, how far it `drop`s once let go (the move away is checked through the fall,
  `TaskPlan::falling`, and `TaskPlan::placed` is where it lands) and a `cost`. A task's plans come
  least cost first (`TaskPlan::cost`); the candidates are still tried in their order, and all
  costs 0 give the plans as before. `Grasp::unseen_span()` as a hold's cost puts grasps closing on
  two seen patches first. A `Pose` converts into either (cost 0, the fingers as loaded, no drop),
  and `Pick::new`, `Turn::new`, `Pull::new` and `Place::new` take any of them, so a list of poses
  passes as before, but for one whose type was inferred from the call (a `.collect()` into it, or
  an empty `vec![]`): name it `Vec<Pose>`. A `Pick` literal sets `grasps:
  poses.into_iter().map(Hold::from).collect()` and gives each hold the `locked` map it set; a
  `Place` literal's `poses` become `releases`. In Python, `Hold(pose, locked=..., cost=...)`,
  `Hold.at_opening(pose, opening, gripper)` and `Release(pose, drop=..., cost=...)`;
  `Pick.grasps` gives `Hold`s, `Pick.locked` is gone, and `Place.poses` is `Place.releases`.
- Grasps of what cameras see, whatever its shape: `grasps_on(&device, &gripper, &[Surface],
  &SurfaceGraspOptions)`. A `Surface` holds the points cameras saw, their normals, which are the
  object's and where each was seen from: `Surface::from_depth(&device, &image, &object,
  robot_mask, radius)` reads a depth image, `Surface::from_points` a cloud seen from one place,
  and `Surface::join` adds a view. A grasp closes on two seen patches that face away from each
  other within the gripper's capacity, the line between them inside both friction cones, with the
  hand and its straight way in clear of every point. Each `Grasp` gives the hand's `pose`, its two
  `contacts` and its `opening`. `examples/surface_grasps.rs` shows it; Python binds neither
  surfaces nor `grasps_on` yet. `SurfaceGraspOptions`:
  - `turns` (8), `approach` (10 cm), `floor`, `overlap` (1 cm) and `contact` (5 mm): the directions
    each pair of patches is grasped from, the straight way in, a height the hand stays over, how
    much of the fingers stays on the patches, a patch's radius.
  - `shell` (by default `Some(f32::INFINITY)`, a wall as thick as the gripper grasps; `None` turns
    it off): a wall seen from one side only, as a bowl's rim from above, is closed on across it to
    where the cameras saw past it, and the grasp is `one_sided`. A cloud fused from several views
    and given one place is grasped with `None`.
  - `first_touch` (on by default): grasps whose fingers close onto their patches before meeting
    any other part of the object come first. It adds and removes no grasp.
  - `narrow` (off by default): for a gripper whose jaws slide (`Gripper::sliding`), fingers come
    in narrower than they open where a wall or a neighbor stands near, at one of
    `Gripper::openings`, and no nearer their patches than 2.5 mm each.
  - `clearance` (0 by default): how far the hand's spheres stay from what they do not close on.
  - `near`: a point to grip near, the grasps nearest it first.
  - `close_on`: more approach angles and shallower insertions whose fingers close on a point,
    after the ordinary grasps.
  - `SurfaceGraspOptions::order_for_turn` and `Grasp::moment(&screw, &gripper, &options)`: grips
    for a turn, ordered by the moment their fingertips can give about its line.
  - `Grasp::closes_on(&gripper, at, radius)`: whether a ball lies between the fingers as they
    close.
  - `Robot::hand_spheres(hand)` (`HandSphere`): the hand's spheres in its IK frame, what
    `grasps_on` keeps clear. In Python, `Robot.hand_spheres`.
- Where a held object rests on what cameras see: `rests_on(&device, &[RestProblem], &[Surface],
  &RestOptions)`. A problem names the object, how it is turned as it is set down (`rotation`) and
  a surface whose `object` points are the support; each `Rest` gives the object's `pose`, the
  `support` it touches and how far it can be `tipping`. The object comes straight down, not tipped
  or slid into place; a can laid on its side rolls into a cradle or down to a lip.
  `Release::from(&rest)` carries a rest to a place. `examples/surface_rests.rs` shows it, and
  `scripts/validate_rests.py` lets each rest go in MuJoCo. Also:
  - `RestOptions`: `spacing` (1 cm), `turns` (8), `friction` (0.5), `tipping` (5°), `clearance`
    (1 cm) and `touching` (2 mm), and two that find more, both on by default: `look_closer`, for a
    rest narrower than the spots, and `seen_from_above_holds`, for a top whose normals were read at
    a slant.
  - `RestProblem::near`: a point to look closer around, the rests and falls nearest it first
    (`Rest::cost`, `Fall::cost`).
  - `falls_into(&device, &[RestProblem], &[Surface], &RestOptions)`: where to let an object go for
    it to fall into a container and stay. Each `Fall` gives the `pose` to let go at, how far it
    drops to where it lands (`drop`), how deep under the rim (`sunk`) and a `cost`;
    `Release::from(&fall)` carries it to a place. A tray, a plate or a table has none.
  - `rest_failures(&device, &problems, &surfaces, &rests, &options)`: why a problem has no rest
    (`RestFailure`: `Blocked`, `Tips`, `Unheld`, `Unsupported`), and what is in its way
    (`Blocking`).
- A hand as a parallel-jaw gripper, built once: `Robot::gripper(hand, &GripperCalibration)`, from
  the loaded and closed pad gaps and the finger joints at the closed command
  (`GripperCalibration::new(loaded_gap, closed_gap, closed)`), the widest it grasps
  (`capacity`, the loaded gap by default) and the pads' `friction` (0.5), checked against the
  model's stroke and joint locks. `grasps_on`, `grasps` and miru's grip calls take it.
  `Gripper::locked_at` and `opening_of` turn a gap into finger joint values and back, `openings`
  lists the openings `narrow` tries, and `tip` is how far its spheres reach past the hand's frame.
  In Python, `Robot.gripper`, `GripperCalibration` and `Gripper`.
- Turning a thing about a line in one call: `turn(&device, &worlds, &[Turn], &TaskOptions)`, a
  knob, a cap, a lever or a valve's wheel, composed of IK, `follow` and `plan` as `pick` is.
  `Turn::new(world, start, object, grasps, angle)` turns the object by `angle` about its `screw`
  (none: about the hand's own z axis). The hand holds the object through the turn, and what it
  sweeps is checked against everything but `Turn::support`. With `Turn::stays`, for a world that
  holds the part and what it turns in as one obstacle, the obstacle stays where it is and only the
  fingers may touch it. A plan has its `turn` between `approach` and
  `depart`, and a candidate that makes no turn stops at `TaskStep::Turn` with how far it turned
  (`CandidateStop::turned`). `examples/turns.rs` turns a knob, a cap, a lever and a wheel. In
  Python, `Turn` and `turn`.
- `Screw { point, direction, pitch }`, a line and how far a thing carried about it moves along it
  per radian, with `Screw::about(point, direction)`: `carry(pose, angle)` carries a pose about it,
  and `path(pose, angle, precision)` gives a tool path's waypoints along the arc. In Python,
  `Screw`.
- A straight draw by a handle: `pull(&device, &worlds, &[Pull], &TaskOptions)`. The hand keeps its
  orientation and moves a signed `distance` along `direction`; a plan has its `pull` between
  `approach` and `depart`, and a candidate stopped partway says how far it drew
  (`CandidateStop::pulled`, at `TaskStep::Pull`). A distance grid that stays where it is may be
  what a pull or a turn grips. In Python, `Pull` and `pull`.
- A pick can grasp one obstacle and lift the body it belongs to: `Pick::body`, a `PickBody` with
  the body's shape and the obstacles that leave with it, and `Pick::lift_world` builds the world
  to execute the lift in, the hold attached. In Python, `PickBody` and `Pick.lift_world`.
- A task stops at as many plans as will do: `Pick::enough`, `Place::enough`, `Turn::enough` and
  `Pull::enough` (`None` by default: every candidate is tried). With a count, the candidates are
  tried in their order a few at a time until that many get through: the plans are those trying
  every one gives, without the IK and motions of the rest, so a pick given the hundreds of grasps
  `grasps_on` finds stops after the first few. In Python, `Pick(..., enough=1)` and the same for
  the others.
- A place can let go as wide as it has room to: `Place::open`, finger values in the order
  preferred (empty by default: the hand opens to where it was loaded). Each release lets go at the
  first opening with which the hand is clear there and on its way out, and the plan names it
  (`TaskPlan::opening`). `Gripper::openings` with `Gripper::locked_at` offers the openings `narrow`
  grasps come in at. In Python, `Place(..., open=[...])` and `TaskPlan.opening`.
- A `Follower` can hold an object: `FollowerOptions::holding`, the `Held` the trajectory was
  planned holding. Its checks, stops, replans and re-anchors carry it, and contacts name it
  (`"held by hand 0"`). Its robot and the devices its ticks are given are those that do not hold
  it, as for a problem's `holding`.
- What configurations touch while they hold objects:
  `device.contacts_holding(&worlds, &item_world, &q, within, &holding)`, a `Held` or `None` per
  item, the held object's contacts listed under its link.
- Depth images and fused maps:
  - `image.filtered(&DepthFilter)`: the depths without the readings between two surfaces, which a
    grid or a map would hold as points in mid-air.
  - `image.points(robot_mask)`: each pixel's reading as a point.
  - `mask.with_shapes(&device, &image, &shapes, margin)`: shapes the world holds as obstacles of
    their own, left out of grids and maps as the robot is.
  - `mask.with_band(&device, &image, margin, pixels)`: also the readings a camera gives just behind
    an outline the mask flags, within `pixels` pixels of it.
  - `sdf::on_shapes(&device, &shapes, &points, &reach)` (`OnShape`): which of a world's shapes
    each point lies on, within its reach.
  - `map.with_weighing(Weighing::Latest)`: each image decides what it sees, for a caller that
    looks once every few seconds. `Weighing::Evidence`, the default, weighs as before.
  - `map.free`, `clear` and `occupy`: a shape's voxels marked free, never observed or occupied.
    `map.part(&region, padding)` is the map inside one region, as a map of its own;
    `map.counts()` (`MapCounts`) and `map.occupied()` say what a map holds.
  - `OccupancyMap::from_shape`, `refined`, `take_readings` and `take_surfaces`: a map of a
    primitive, a map refined into smaller voxels from the same views, and the voxels of chosen
    readings or pixels in a part of their own.
  - `examples/perception.rs` times each depth call for a 640×480 image.
- `batchplan-miru`, an optional crate published with batchplan's version and in its release: what
  depth cameras show, as objects. Its library is `miru`, so a dependent writes
  `miru = { package = "batchplan-miru", version = "0.12" }`. `miru::look` says which pixels are
  one thing, the shape of each and what it stands on, with the same id from one look to the next;
  `miru::observe` is a look through the library in one call, with a `Memory` told what the robot
  did (`Memory::after`) that gives the world each step plans against (`Memory::world`); and
  `Kept::surface`, `Observed::grip_worlds`, `Kept::turn_grips` and `Kept::extractions` give what
  grasps, turns and lifts out of containers need, and `Kept::surface_about` every reading about
  a point, for a place into a thing the look gave in pieces. The memory's map reaches as far as the robot
  does, within `MemoryOptions::lower_corner` and `upper_corner`, which a robot whose base travels
  needs. batchplan does not depend on it. `miru/docs/status.md` says where it stands.
- A limit on how far collision spheres reach past the geometry, whatever the budget:
  `SphereOptions::max_overhang` (`Some(0.0)` keeps every sphere inside; `None`, the default, fits
  as before). `LinkQualification::gap`, new in `Robot::qualify`, is how far the farthest surface
  outside the spheres lies from them: how deep an obstacle reaches into the link unseen. Under a
  limit the geometry must be closed: a link with open edges or a shell facing inward is refused
  with their count, and a collision model given with the options is held to the limit too. In
  Python, `SphereOptions(max_overhang=0.002)`.
- `Robot::hand_reach(hand)` (`HandReach`): an outer bound on the poses a hand reaches;
  `HandReach::excludes(target, &options)` rules a pose out before any IK. In Python,
  `Robot.hand_reach` and `HandReach.excludes`.
- `Pose::is_unit()`: whether a pose is finite with a unit rotation, as every call that takes one
  accepts. In Python, `Pose.is_unit`.
- `Device::install(op)` runs `op` on the device's threads, so a crate built on the library keeps
  its parallel loops there, as miru does.
- Options a host can check where they are set: `IkOptions::check()`, `PlanOptions::check()`,
  `RrtOptions::check()`, `ShortcutOptions::check()`, `ToolPathOptions::check()`,
  `JoinOptions::check()` and `TaskOptions::check()` return the `Error::Input` their call returns
  for them. In Python, `options.check()` raises it.

### Changed

- **Breaking:** a literal that lists every field needs the new ones: `Pick` `body: None` and
  `enough: None`; `Place` `open: vec![]` and `enough: None`; `Held`
  `spheres: SphereOptions::default()`; `TaskOptions` `held_spheres: SphereOptions::default()`;
  `FollowerOptions` `holding: None`; `SphereOptions` `max_overhang: None`. Options literals that
  end in `..Default::default()` are unaffected.
- **Breaking:** `GraspOptions` is two structs, each field with the default it had:
  `ShapeGraspOptions` (`depth`, `turns`, `tip`, `floor`, `overlap`, `along`) for
  `grasps(&object, &gripper, &ShapeGraspOptions)`, which takes the gripper, and
  `SurfaceGraspOptions` for `grasps_on`. `opening` and `friction` are gone: they are the
  gripper's, so a shape's grasps close across at most `Gripper::capacity` and over a floor keep
  their open fingers `Gripper::opening` apart. `ShapeGraspOptions::tip` is `None` by default, the
  gripper's `tip`, where it was 0: a literal that sets `floor` and relied on a tip of 0 sets
  `tip: Some(0.0)`. In Python, `grasps(object, gripper, options=None)` takes a
  `ShapeGraspOptions`, and `GraspOptions` is gone.
- **Breaking:** what a robot holds (`Robot::attach`, `Held`) is covered within its `tolerance` (5 mm
  by default) whatever the budget, which caps the robot's own spheres only, and its spheres hold all
  of it. So more of a tight passage reads as a contact, and clearances and contacts measured to a
  held object differ. A box, a cylinder, a capsule or a ball is covered directly where its tiles
  reach the tolerance; any other shape is fitted, once per object and device. The caller sets the
  cover: `Held::spheres`, and `TaskOptions::held_spheres` for what a pick or a moving turn carries,
  which the pick's `TaskPlan::held` hands to the place; a smaller tolerance fits closer, with more
  spheres. In Python, `Held(..., spheres=SphereOptions(tolerance=0.002))`.
  `examples/held_spheres.rs` gives the counts
  ([runs](docs/benchmarks.md#spheres-for-what-a-hand-holds)).
- **Breaking:** a robot mask judges each reading by where its point lies:
  `image.robot_mask(&robot, &q, padding, &holding)` flags a reading whose point is inside a
  collision sphere grown by the padding, and `RobotMask::on_robot`, a flag per pixel, replaces
  `RobotMask::depth`. A camera the hand carries keeps every reading past the hand, where it lost
  its whole image, and a camera across the table keeps the surfaces just past the arm's outline;
  a sensor's stray readings there, farther than the padding behind the arm, are obstacles, which
  `with_band` takes out. `holding` names what the hands hold (`&[]` for nothing): its readings are
  left out and the space it fills is free, without attaching it.
- **Breaking:** options a call cannot use are input errors where the call ran before: IK
  tolerances that are zero, negative or not a number; collision weights that are not finite,
  margins that are not numbers, and infinite margins with a weight on them; a plan's acceleration
  and velocity weights that are not finite; an infinite `initial_step`; and seeds, control points
  or samples per span past what 32 bits count, alone or as a batch's problems times their seeds.
  `Device::evaluate` refuses such weights and margins too, `ik::failures` the options `solve_ik`
  refuses, and `servo` a `slow_within` or `halt_within` that is not a number. A robot's
  `SphereOptions` are checked whether its spheres are fitted or given
  (`RobotOptions::collision_model`): a budget of zero, or a tolerance or voxel size that is not
  positive, is an `Error::Input` from `Robot::load`, `from_urdf` and `assemble`, where a given
  model ignored them and a fit gave an `Error::Load`.
- **Breaking:** `Clearance` (what `MeshModel::clearance` returns) is `#[non_exhaustive]`, as the
  other results are: a pattern that names both its fields ends in `..`, and callers no longer
  build one.
- **Breaking:** `TaskStep` has `Turn` and then `Pull` between `Reach` and `Approach`, the order
  the steps are checked in. The steps that were there keep their order among themselves, and as
  numbers (`as isize`, and in Python) `Approach`, `Depart` and `Transit` are 4, 5 and 6 where they
  were 2, 3 and 4.
- Faster:
  - Fitting a robot's spheres takes 7% to 23% less time, with the same fitted models
    ([runs](docs/benchmarks.md#fitting-collision-spheres)).
  - Fusing a frame or building a grid with a robot mask costs by the robot's volume, not by its
    spheres times the grid's points.
  - A frame fused into a map on the CPU device costs by what its camera sees, not by the map's
    size, with the same map to the byte.
  - `datagen::handover` fits a carried object once for all its grasps and `datagen::reset_states`
    once per shape, and a device for a robot holding an object of many spheres is made in time
    linear in them.
  - Two groups of one call that hold the same object fit its spheres once, where both fitted it.

### Removed

- **Breaking:** `SphereGeometry` and `SphereOptions::geometry`, in Python too: spheres are fitted
  to the links' collision geometry, the default before. A `SphereOptions` literal that names
  `geometry` drops it. The loaders no longer read visual geometry, so a URDF whose visual meshes
  cannot be found (in a package that neither `RobotOptions::package_dirs` nor the directories
  above the file hold, or at a relative path in a URDF given as text) now loads.
- The `benchmark` example's `--save` and `--check` modes, `scripts/compare_planners.py` and
  `scripts/cpp_planners.cc`: the earlier comparison with other planners, which
  `scripts/run_suite.sh` replaced and the docs stopped running in 0.11.0.

### Fixed

- A planned path's spline keeps 2 mm between the robot's links away from its ends, ramped as its
  margin from the world is, where it kept none: a curve cutting a corner past what the other hand
  held passed into it between the samples it was checked at.
- `place` refuses worlds uploaded to another device with no tasks too, as `pick` does, rather
  than returning no outcomes.
- A plan with a `time_budget` of `Duration::MAX` has no deadline, where it panicked, and a search
  with `RrtOptions::max_iterations` of `usize::MAX` that searches again without its clearance
  finds what it finds, where its count wrapped and it found nothing.
- A call that held objects in several hundred ways at once, as a `place` after each of a pick's
  plans with more than one candidate for each, could overflow the stack of one of the device's
  threads and abort the process. `place`, `pick`, `handover` and problems with a `holding` now
  keep no more holds in hand than the device has threads.
- **Breaking:** a mesh of more than one closed shell is inside where its surface winds around a
  point, for distance grids (`SdfGrid::from_mesh`, and scene meshes `World::load` reads), sphere
  fitting and `Robot::qualify`: a table's top no longer reads free above the end of a leg that runs
  up into it, a shell facing the other way no longer reads hollow, and a fit no longer puts spheres
  outside the link around a shell facing inward. Shells that overlap are their union, and a shell
  apart from the others is solid whichever way it faces. Grids of a single shell are bit for bit
  what they were, and every robot in this repository is fitted as before; a robot with such a mesh
  is fitted differently, and clearances and contacts measured to it differ.

## [0.11.0] - 2026-11-14

### Added

- **Breaking:** `PlanOptions::start_tolerance` and `ToolPathOptions::start_tolerance` optionally
  clamp measured starts just past a joint limit before the normal collision and constraint checks.
  Both default to zero, preserving strict validation. Larger or nonfinite violations are input
  errors; continuous joints, goals and start velocities are unchanged. `pick`, `place` and
  `handover` use `TaskOptions::plan.start_tolerance` for their initial states. Results retain the
  corrected starts. Exact options literals need the new field; Python options expose it too.
- **Breaking:** picks and holds say where the fingers are: `Pick::locked` and `Held::locked` hold
  the robot's locked joints (as `Robot::relock` takes them) at those values for that task, or
  while the object is held, so a rim between walls can be picked with the fingers part open and
  a hold checked where they closed. One call takes tasks and holds at many openings, a relocked
  robot for each group; empty, the default, is as loaded, to the bit. A `Pick` or `Held` literal
  needs `locked: HashMap::new()`. In Python, `Pick(..., locked={...})`, `Held(..., locked={...})`.
- **Breaking:** `Occlusion::Band(distance)`, for `SdfGrid::from_depth` and `OccupancyMap::grid`:
  solid within `distance` behind an observed surface, as seen from the camera, and free past it,
  in the robot's shadow and along pixels without a reading. In a map, unobserved space within
  `distance` of an occupied voxel is solid, in any direction, and the robot's shadow is free. A
  band that is not finite or is negative is an input error. `Occlusion` is no longer `Eq` (the
  band is an `f32`); from depth images, the other rules build the same grids as before.
- One hand can pass what it holds to another: `handover(&device, &worlds, &[Handover], &TaskOptions)`,
  with `Handover::new(&robot, world, start, held, to)`. Candidates are each meeting with each of the
  taker's grasps; both hands' IK solves at the meeting with the object standing there, each hand's
  straight move holds the other arm still, and the way to the meeting holds the object in the
  giver's hand. Every candidate that gets through comes back quickest first, naming its `meeting`
  (`TaskPlan::meeting`) and the taker's `held`. `handover_poses` proposes meetings where both arms
  reach and `handover_grasps` the taker's grasps at the object's other end. In Python, the same
  names.
- **Breaking:** `grasps` can hold an object toward its ends: `GraspOptions::along`, a fraction of
  half the side the fingers close on (0, the middle, as before), for a pick whose object one hand
  passes to another. A `GraspOptions` literal needs the new field (`along: 0.0`); literals with
  `..GraspOptions::default()` are unaffected. In Python, `GraspOptions(along=0.75)`.
- **Breaking:** a tool path can hold joints still: `ToolPath::hold` lists joints that stay where
  `start` has them along the whole path, as `PlanProblem::hold` and `IkProblem::hold` do, for a
  hand whose arm shares joints (a torso) with one that holds what it reaches for. The hand follows
  with the joints left, and stops where they cannot keep it on the path. A `ToolPath` literal
  needs the new field (`hold: vec![]`). In Python, `ToolPath(..., hold=[0, 1])`.
- **Breaking:** the straight moves of `pick` and `place` can be capped near the grasp or release
  alone: `TaskOptions::tool_speed_near` (meters) holds `tool_speed` within that distance of where
  the hand grasps or lets go, which the move in ends at and the move away starts from, and the
  rest of each move runs as fast as the joints allow. The handovers of `datagen::handover` take
  it too. `None`, the default, caps both moves all along, as before, to the bit. A `TaskOptions`
  literal needs the new field (`tool_speed_near: None`). In Python,
  `TaskOptions(tool_speed_near=0.02)`.
- **Breaking:** a `Follower` can follow a motion that joins a way to a move in, replanning only
  the way: `FollowerOptions::detour` takes a `Detour { from, until, then }`. The follower keeps
  the path before `from` and after `until`. Where the rest of the motion collides within that
  window, it plans to `then`'s first state (the move in as planned apart), from no sooner than
  `from`, then takes up `then` from rest there and reports `FollowStatus::Resumed { at,
  trajectory }`; along `then` it keeps the path. `None`, the default, changes nothing. A
  `FollowerOptions` literal needs the new field (`detour: None`).

- `CollisionModel::to_json` and `CollisionModel::from_json` give and take the JSON text `save`
  and `load` write and read, for a model sent where no file goes. In Python, `model.to_json()`
  and `CollisionModel.from_json(text)`.

- **Breaking:** a `Pick` or `Place` can leave its transit out and run each step in a world of its
  own. With `transit: false` a task solves its grasps or releases and its two straight moves
  alone, without checking its start, and keeps every candidate that passes them, so a program of
  tasks solves all its moves in two calls and the ways between them in one `plan`
  (`TaskPlan::begin` and `end` are where a way ends and the next begins), then joins them.
  `transit_world` and `depart_world` name the worlds of the start and the planned path and of the
  move away where they are not the task's `world`: the same obstacles, shaped or touched
  otherwise, as a transit that touches nothing the grasp may. `TaskPlan::transit` is an `Option`,
  `None` for a task without one, and `TaskPlan::motions` gives a plan's motions in order. Code
  reading `plan.transit` unwraps it or reads `plan.motions()`, and a struct literal that writes
  out every field of `Pick` or `Place` needs the three new ones. In Python, `Pick` and `Place`
  take `transit`, `transit_world` and `depart_world`, and `TaskPlan` has `motions` and `begin`.
- **Breaking:** `join` checks a sequence's blends with what a hand holds: `Sequence::holding`, a
  pick's `Held`, as `PlanProblem::holding`, so a carry joins into the move in of its place on the
  device without the object. A `Sequence` literal needs the new field (`holding: None`).
- A world can let what a hand comes to hold touch an obstacle: `Touch::held(obstacle, hand)`, for
  one hand in an upload, on the device that holds nothing. `solve_ik`, `follow` and `plan`
  problems with `holding` then rest their objects on what they are set down on in one call
  against one upload, as a device holding the object does with the world uploaded to it;
  problems that hold nothing are checked as without it. In Python, `Touch.held`.
- `time(&device, &paths, speed_scale, &hand_speeds)` times many planned paths in one call on the
  device's threads, each as `Trajectory::try_new` times one, or under its own list of hand speed
  caps (`&[]` caps none), and gives each path's trajectory or error in order. In Python,
  `time(device, paths, speed_scale=1.0, hand_speeds=None)`.
- **Breaking:** a hand speed cap can hold on part of a path: `HandSpeed::near`
  (`Near { point, radius }`) caps the hand only while its frame is within `radius` of `point`,
  so a way joined into the move to a grasp comes as fast as the joints allow and arrives slowly,
  and `Trajectory::with_hand_speed` takes several caps, each holding where it is. A `HandSpeed`
  literal needs `near: None`, and `with_hand_speed` takes a slice (`&[speed]`). A 15 cm way
  joined into a 6 cm move capped at 5 cm/s takes 2.21 s where the two take 2.26 s one after the
  other and the joined path 4.41 s capped throughout (a Panda, `tests/join.rs`). In Python,
  `HandSpeed(hand, linear, angular, near=Near(point, radius))`, and `with_hand_speed` takes one
  cap or a list.

- Publish the Python bindings as `batches` on PyPI (`import batchplan`). Version tags publish
  the Rust crate and Python wheels and source together, using trusted publishing.

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

- Faster sphere fitting, with the same fitted collision models. Dense surface checks test four
  spheres at once: on one thread, fitting the stand with a 512-sphere budget takes 19% less time
  on an M4 Pro and 30% less on a Ryzen AI Max+ 395; the Panda with a 1,200-sphere budget takes
  4% and 11% less, respectively.
  These isolate fitting from self-collision analysis. The
  [benchmark guide](https://github.com/natsuku-lab/batchplan/blob/v0.11.0/docs/benchmarks.md#fitting-collision-spheres) gives the runs and
  `examples/fitting.rs` reproduces the fixtures.
- Faster CPU collision checks, contact queries, IK, planning, tasks, depth-grid construction and
  occupancy updates, with results the same to the bit. In six alternating runs, the best CPU
  times for 1,000 IK targets among clutter fell 9% on a Ryzen AI Max+ 395 and 4% on an M4 Pro;
  256 plans with four optimized seeds took 22% and 6% less time, respectively. The
  [benchmark guide](https://github.com/natsuku-lab/batchplan/blob/v0.11.0/docs/benchmarks.md#how-the-speed-came) gives the times, lower quartiles and
  build configurations. This workspace builds batchplan as one
  codegen unit (`[profile.release.package.batchplan] codegen-units = 1`), so a change measured here
  no longer moves with how rustc splits the crate; a crate that depends on batchplan can do the same.
- Published crates exclude development agent instructions and skills; the library-user
  `AGENTS.md` guide remains included.
- A world with a distance grid uploads without checking the grid's every value again: a grid is
  checked when it is made, and nothing changes one after. `pick` and `place` upload their worlds
  by the dozen, so a grid in the world cost each call that many passes over its values. The
  searches' and tool paths' collision checks and the CPU device's `clearance` also look a grid
  up without its gradient, which they never read.
- IK on the CPU device, and trajectory optimization's repairs on either, skip the self-collision
  sphere pairs that cannot add to the cost: a sphere past the margin from another link's
  bounding sphere is not checked against that link's spheres. Results are the same to the bit.
  A robot fitted with many spheres gains most: a `place` of four holds on a Panda of 512 spheres
  takes 11 ms on an M4 Pro where it took 18 ([Benchmarks](https://github.com/natsuku-lab/batchplan/blob/v0.11.0/docs/benchmarks.md#how-the-speed-came)).
- **Breaking:** IK screens its seeds. Every seed takes `IkOptions::screening` steps (8 by
  default), then only `IkOptions::kept` of each problem's (4) take the rest of `iterations`: those
  not yet at their targets, the nearest first, then seeds that have arrived. Where the steps push
  away from collisions (`collision_step` above 0, for a robot with collision spheres),
  collision-free seeds go before the others; elsewhere whether a seed collides after the
  screening says little about where it ends, and they go by their errors alone. The seeds left behind are returned as the screening left
  them. Code that writes out every field of `IkOptions` needs the two new ones, and
  `screening: 0` is the IK of before, as are problems of more than 32,768 seeds. On a GPU the
  screening, the choice and the remaining steps are one round trip, a small call is one
  submission, and its buffers are kept for the next. On an M4 Pro, 1,000 targets for a Panda
  with collision spheres take 36–39 ms where they took 68–82, and its CPU threads 95 ms where
  212, with 945 solved where 944; one target and ten take no longer than before, with the push
  or without. In RoboKit's IK benchmark on an RTX 4090 (a Panda without spheres), 1,000 targets
  take 3.1–3.3 ms where they took 3.6–3.7 and 2,000 take 6.1–6.3 ms where 7.2; 10 took
  0.77–0.80 ms where 0.71 before the changes for small calls, which are not measured there
  ([Historical benchmark][historical-robokit]). The 2,600 pose-goal problems solve as
  before (98.3%). In Python, `IkOptions.screening` and `IkOptions.kept`.
- Batched IK spends less time around the solve: a problem's seeds share its targets instead
  of each carrying a copy, large batches are seeded on the device's threads, buffers go to the
  GPU without an extra copy, and a robot without collision spheres skips the collision push,
  which is zero for it. The same results, bit for bit. In RoboKit's IK benchmark on an RTX 4090
  (a Panda, collision off, 64 seeds of 30 steps), 1,000 targets take 4.5 ms where they took
  10.8, 2,000 take 9.1 ms where 23.2, and one takes 0.9 ms where 1.26
  ([Historical benchmark][historical-robokit]).
- **Breaking:** `IkOptions::collision_step` must be finite: an infinite or NaN one is an input
  error. NaN and negative infinity used to mean no push, and positive infinity solved nothing
  (the CPU left every seed where it started).
- **Breaking:** `Device::gpu` no longer chooses DX12's software renderer (WARP, "Microsoft Basic
  Render Driver"), which every Windows machine has: it fails as it does where there is no
  adapter, naming the renderer it left out, so `Device::gpu(..).or_else(|_| Device::cpu(..))`
  falls to the CPU device. On a four-core machine that renderer had not finished one pick after
  82 minutes, where the CPU device takes 16 ms. `Device::gpu_named` still takes it by name.
- A distance grid costs a link nothing while the link keeps away from the grid's box: the
  collision checks rule the grid out by its box and the least value on the box's faces, as they
  rule other obstacles out by their distance, where every sphere used to be looked up in every
  grid. Among cluttered tabletops, IK for 1,000 targets with a grid 2 m from the arm in every
  world takes 28.8 ms on an M4 Pro's GPU where it took 35.7, and 67 ms on its CPU where 107;
  with the grid beside the table, 31.7 ms where 35.7, and 72 ms where 108
  (`examples/latency.rs`). `world_clearance` for a link past the margin from a grid is a bound
  from below, as it is for other obstacles. Uploads without a grid run the code they ran: the
  rule is in kernels, and in copies of the CPU's checks, that only worlds uploaded with a grid
  run ([Benchmarks](https://github.com/natsuku-lab/batchplan/blob/v0.11.0/docs/benchmarks.md#how-the-speed-came)).
- A GPU device is made sooner: its kernels' pipelines compile each on a thread of its own. With
  the driver's shader cache empty, `Device::gpu` takes 1.4 s on an M4 Pro where it took 2.6,
  though it now builds the kernels of uploads with grids as well, `Device::warmup` 0.7 s where
  2.0, and `Device::with_robot` for a robot holding an object 1.7 s where 3.0. The second set
  costs memory: a GPU device keeps 66 to 88 MB after it is made where it kept 35 to 37, and
  about 20 MB for each held shape where 9.
- GPU kernels leave naga fewer loops to count, and trajectory optimization on a GPU leaves out
  paths that are done. naga counts the rounds of every shader loop to rule out endless ones,
  which cost IK's short loops more than their work. One hand's damped least squares is now
  written out term by term for each robot, as its kinematics are, and the kernels' loops of a
  known count (over the joints, the hands, the line search's steps) are written out when the
  kernels are built. A path whose optimization is over, a steepest-descent search having found
  nothing better, takes no part in later rounds, as the CPU's already stopped. Of the lanes a
  small batch gives each configuration, one works out the IK step and the others take the
  configuration from it, and a robot's own link of 24 spheres or more takes them two to a round.
  On an M4 Pro, one target's default IK takes 0.34 ms where it took 1.14 for a Panda without
  collision spheres, and 0.92 ms where 2.03 with them; 1,000 targets take 1.7 and 22.1 ms where
  3.8 and 30.0; 256 plans with 4 optimized seeds take 83 ms where 94, and 16 take 15.9 ms where
  23.7 (`examples/latency.rs`). On a Radeon 8060S, before the last two of these changes, one
  target took 0.18 ms where 1.12, and 1.10 ms where 2.70. In RoboKit's IK benchmark on an RTX
  4090 one target takes 0.33–0.42 ms and ten 0.35–0.43, where RoboKit takes 0.57–0.59 and
  0.55–0.56 ([Historical benchmark][historical-robokit]). The crate has no `unsafe`: the
  kernels are built with every check naga makes, the loop counters among them, which on an M4
  Pro make optimized plans 4–7% slower and IK for hands that share joints 13–20%. Three kinds of
  call are slower than before: IK with the push for a robot of several hundred spheres (4–5%),
  plans to goals past an obstacle (5%) and `evaluate` (1.5–4%)
  ([Benchmarks](https://github.com/natsuku-lab/batchplan/blob/v0.11.0/docs/benchmarks.md#how-the-speed-came)). A GPU's results move in their last
  places, IK's and trajectory optimization's alike: a seed in thousands arrives or does not.
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

- A map no longer frees what another surface hides because the arm once hid it too: a voxel no
  image has observed is the robot's shadow only while nothing else has hidden it, whichever image
  came first. Under `Occlusion::OccupiedExceptRobotShadow` such a voxel is solid, where it read
  free.
- An obstacle whose rotation is a little off a unit quaternion (up to a thousandth is accepted)
  is turned, not stretched. Its frame was built from the quaternion as given, which scaled it
  by up to 0.4%: a 1.2 m box reached 2 mm past its ends, and a search, which first rules
  obstacles out by bounding spheres of their stated size, could pass a configuration that
  `evaluate` found 2 mm inside one. Frames are now built from the rotation made unit, on both
  devices and in `Obstacle::distance`; a rotation that is unit to rounding is used as it is, so
  results for those do not move. The same holds for a held object's shapes (`Robot::attach`
  and a problem's `holding`), whose spheres sat up to 0.2% nearer or farther along the object,
  for the obstacles `MeshModel::clearance` checks, and for a depth image's camera, whose
  readings were placed 0.2% nearer or farther (5 mm at 3 m) in `SdfGrid::from_depth`,
  `OccupancyMap::integrate` and `DepthImage::robot_mask`. A hold's grasp places its object by
  the grasp made unit, and `Robot::assemble` a part by its base made unit. A camera's pose and
  a part's base are now checked as other poses are: one that is not finite, or whose rotation
  is more than a thousandth off unit, is an `Error::Input`, where it built a wrong grid or
  robot without a word.

- GPU devices on Windows using the FXC DX12 shader compiler can build the hand-frame kernels.

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

[Unreleased]: https://github.com/natsuku-lab/batchplan/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/natsuku-lab/batchplan/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/natsuku-lab/batchplan/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/natsuku-lab/batchplan/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/natsuku-lab/batchplan/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/natsuku-lab/batchplan/compare/v0.7.0...v0.8.0

[historical-robokit]: https://github.com/natsuku-lab/batchplan/blob/9d7823a53148a3f571fd5615d0ebbd5818d8f1d4/docs/benchmarks.md#against-robokit
