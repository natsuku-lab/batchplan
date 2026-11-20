# Changelog

What changed in each release, for people upgrading, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Cargo's semver](https://doc.rust-lang.org/cargo/reference/semver.html): before 1.0, a new minor
version (0.9 → 0.10) may break the interface, and a patch (0.9.0 → 0.9.1) breaks nothing.
[Contributing](CONTRIBUTING.md#releases) says how releases are made.

## [Unreleased]

### Added

- A band behind what a mask flags: `mask.with_band(&device, &image, margin, pixels)` flags a
  reading where a flagged reading within `pixels` pixels of it, along rows, columns and
  diagonals, is nearer the camera than it by more than `margin` at its depth (`margin` as
  `with_shapes` takes it). A depth camera reads the pixels beside a depth jump toward the other
  surface, one for a sensor that samples each pixel's disparity a fraction of a pixel off and up
  to half the window for a stereo block matcher; behind the arm's or a shape's outline, those
  readings lie farther than the padding or the margin from both surfaces and the filter keeps
  the ones near either, so they were obstacles in mid-air where fingers go. `pixels` is that
  fact of the camera. What stands in front of a flagged thing, as an object no shape explains on
  a table given as one, is nearer than the flagged readings around it or level with them, and
  keeps every reading. Given up: an outline read too near stays a reading, and the honest
  readings just behind each flagged outline are taken with the rest. A band of one pixel behind
  the Panda's mask and the scene's three shapes (`examples/perception.rs`, the camera across the
  table) takes 0.14 ms for a 640×480 image on the M4 Pro's 14 threads and 0.22 on the Ryzen's 32.

- How much one image counts in a fused map: `map.with_weighing(Weighing)`. `Weighing::Evidence`,
  the default and the weighing until now, suits a stream of frames: a reading adds 0.85 to a
  voxel's log-odds and a ray through it takes 0.4, so it takes three images to fill a voxel seen
  empty often and nine to empty one seen full often. Under `Weighing::Latest` each image decides
  what it sees: a voxel it reads a surface in, or that holds one of its readings, is as surely
  occupied as a map holds anything, and one its rays pass through as surely free; what it does
  not see keeps what earlier images said, but for the space a mask's robot and shapes fill,
  which is as surely free wherever it is. For a caller that looks once every few seconds; one
  stray reading is an obstacle until a look sees through it.

- Where a fused map is occupied: `map.occupied()`, the middles of the voxels `counts` counts as
  occupied, in the map's frame and in the order of its voxels, and of a part none it leaves out.
  Where never-observed space is taken for free, none means there is no grid to make, and
  otherwise a part around them is a grid of what is there.
  `part`'s documentation now says that a part's points cover its region and padding whatever the
  map covers, and that one past a grid's size limits is an input error.

- `mask.with_shapes(&device, &image, &shapes, margin)`: the mask with shapes the caller holds left out
  as the robot is, as a cell's fixtures and objects measured as shapes, which the world holds
  as obstacles of their own. A reading within the margin of a shape is not fused and what only
  it hides is shadow; the space the shape fills is free in the map; what a shape does not
  cover stays a reading. So each thing has one memory, a shape or the map's voxels, and a plan
  is checked against both. The perception page says where a perception library meets
  batchplan, with an example of one look through both. A shape costs by what the camera may see
  of it and by the rows of voxels it spans: it is tested only in the pixels its box, grown by
  the margin and cut to the camera's view, projects to, so a floor or a wall around the cell
  costs by what is seen of it and one behind the camera nothing, and its space in a map is
  marked a run of voxels a row. With the scene's three shapes (`examples/perception.rs`, the
  camera across the table) a 640×480 image takes 0.26 ms on the M4 Pro's 14 threads and 0.23 on
  the Ryzen's 32, and the frame fused into a map of 1.8M voxels with them and the Panda's mask
  1.3 ms on the M4 Pro's CPU, 1.5 on its GPU, 2.8 on the Ryzen and 1.0 on the Radeon 8060S.

- `image.filtered(&DepthFilter)`: a depth image's depths without the readings that lie between
  two surfaces, which a depth camera gives where a pixel covers both a near surface and what is
  behind it, and which a grid or a map would hold as points in mid-air. A reading goes where,
  along a row, a column or a diagonal through its pixel, one side is nearer by more than one
  surface can step, or has no reading, and the other side farther, or has none. What one surface
  can step comes from how far it may be turned to the rays (`steepest`, 80 degrees) and from the
  camera's `noise` (`deviations`, 4 of them). Surfaces keep their outlines, but for a reading
  with a step on one side and no reading straight across; depth without noise loses those and
  surfaces seen that edge-on, and nothing else.

- What a caller knows of a fused map, said in the map's frame with any `Obstacle` as the shape:
  `map.free(&shape, margin)` marks the voxels within the margin free, as seen through once (where
  an object stood before the robot carried it away, or a shape the world holds as an obstacle
  of its own, on a copy of the map), and `map.clear(&shape, margin)` marks them never observed
  (where something was moved). `map.part(&region, padding)` is the map inside one region, as a
  map of its own that reaches the padding past the region, every voxel outside left out:
  `part(&object, padding)?.grid(&device, Occlusion::Occupied)` is one object as the cameras saw
  into it, solid where a reading landed or nothing saw past, with nothing around it. A part
  takes no images. `map.counts()` gives the voxels occupied, free and never observed
  (`MapCounts`), of a part those inside its region.

- Each pixel's reading as a point: `DepthImage::points(robot_mask)`, row by row, `None` where a
  pixel has no reading or reads the robot. Pixel `i` is column `i % width` of row `i / width`, its
  centre at those whole coordinates, as grids, maps and surfaces read it.

- `examples/perception.rs` times each depth call for a 640×480 image: the robot's mask, the
  image's points, a surface, a grid, a frame fused into a map and the map's grid, for a camera
  over a table and one the hand carries.

- **Breaking:** a limit on how far collision spheres reach past the geometry:
  `SphereOptions::max_overhang`, `Some(0.002)` for 2 mm and `Some(0.0)` for spheres inside it,
  whatever the budget. Spheres are chosen as before and each radius cut back to its center's depth
  plus the limit, so the budget is kept and the surface it cannot cover within the limit is left
  outside the spheres: `LinkQualification::gap`, new in `Robot::qualify`, is how far outside the
  farthest of it lies, which is how deep an obstacle reaches into the link unseen. The geometry
  must be closed, with no shell facing inward: a link whose mesh has open edges, or such a shell
  (a cavity's among them), is refused with their count, and one too thin for a center inside it
  under a limit of zero by name. Shells that overlap or share a face are their union. Round
  shapes are measured as the fit sees them, on their meshes of 24 sides. A collision model given
  with the options is held to the limit too, a sphere past it refused by link and index with how
  far it reaches. Held objects take their own (`AttachedObject::spheres`, `Held::spheres`): a
  box, capsule, disc or rod is covered whole where the budget has a cover within the limit, and
  fitted and held to it where not; a shape without thickness is refused at a limit of zero. `None`, the default, fits as before. A `SphereOptions` literal
  that lists every field needs the new one (`max_overhang: None`); in Python,
  `SphereOptions(max_overhang=0.002)`.

- Where to let a held object go for it to fall into a container and stay:
  `falls_into(&device, &[RestProblem], &[Surface], &RestOptions)`, with the same problems,
  surfaces and options as `rests_on`. Each `Fall` gives a `pose` to let go at, for
  `Place::poses`, its bottom at least `clearance` over the rim of a hollow of what was seen that
  the object lands in deeper than its middle is over its bottom, and how deep (`sunk`). Whether
  it could leave is asked of it worn down by half the spots' diagonal, so a gap it fits through
  holds nothing in. A basket, a bin or an open drawer has falls; a tray, a plate or a table, a
  container whose floor (or a wall sloping in under the object) was not seen that deep, and an
  object thinner than that wearing leaves a cell of, have none.

- Where a held object rests on what cameras see: `rests_on(&device, &[RestProblem], &[Surface],
  &RestOptions)`. A problem names the object, how it is turned as it is set down (`rotation`:
  upright, on its side, flat on a slope) and a surface whose `object` points are the support. The
  object comes straight down over spots `spacing` (1 cm) apart at `turns` (8) turns about the
  vertical, and a `Rest` comes back where what it then touches holds it, with `friction` (0.5) at
  the contacts and its weight tipped `tipping` (5°) any way, clear of every other point by
  `clearance` (1 cm): the object's `pose`, for `Place::poses`, the `support` it touches and how far
  it can be `tipping`, at each of the turns, so a box comes at its half turn and a bottle laid down
  comes end for end too. It rests as it is set down, not tipped or slid into place: a slope needs
  the rotation that lies flat on it. `RestProblem::near` names a point to look closer around, a cell
  of the support apart, for a rest the spots are too far apart to find, as the middle of a dished
  plate under a wide flat bottom. A can or a capsule laid on its side rolls, within half the spacing
  and `touching`, into the cradle between two bars or a rack's groove or down to a lip, and has no
  rest on a flat support. Many problems run in one call, each surface read once.
  `examples/surface_rests.rs` times it, and `scripts/validate_rests.py` lets go of each rest in
  MuJoCo.

- What a configuration touches while it holds an object:
  `device.contacts_holding(&worlds, &item_world, &q, within, &holding)`, with a `Held` or `None`
  per item, on the device that holds nothing, as a problem's `holding` takes it. What the held
  object comes near is listed under its link (`"held by hand 0"`), what the worlds let it touch
  (`Touch::held`) is not a contact, and one call takes items holding different objects.

- **Breaking:** how spheres cover what a hand holds is the caller's to set: `Held::spheres`, a
  `SphereOptions` (64 spheres within 5 mm by default, as before). A thin wide object, as a plate
  14 cm across, is fitted to 8.6 mm past its side at the default, so that a neighbor a few
  millimeters away reads as touching it; with `budget: 256` it is covered within 4.7 mm. One
  call takes one object held with different options, each fitted on its own. A `Held` literal
  needs the new field (`spheres: SphereOptions::default()`); a pick's `TaskPlan::held` carries the
  default. In Python, `Held(..., spheres=SphereOptions(budget=256))`.

- **Breaking:** grasps of what cameras see, whatever its shape:
  `grasps_on(&device, hand, &[Surface], &GraspOptions)`. A grasp closes on two patches of the
  object's surface that were both seen, face away from each other within the opening, and have
  the line between them inside both friction cones (`GraspOptions::friction`, 0.5, and `contact`,
  the patch's radius, 5 mm), so a bowl's wall, a handle and a box's sides are found alike. The
  robot's own hand, with its fingers as locked, and its straight way in (`GraspOptions::approach`,
  10 cm) are kept clear of every point, the object's and the rest; `floor` holds the frame and
  the open fingertips over a height, and a hand that is not clear is drawn back while `overlap`
  of the fingers stays on the patches. Each `Grasp` gives the hand's `pose` and its two `contacts`, the first few from
  above and spread over the object. `Surface::from_depth` reads a depth image (the object's
  pixels marked, the robot masked out, normals toward the camera), `Surface::from_points` a
  cloud seen from one place, and `Surface::join` adds a view. Surfaces of one call run in parallel
  on the device's threads. A `GraspOptions` literal needs the new fields (`friction: 0.5`,
  `contact: 0.005`, `approach: 0.1`); in Python, `GraspOptions(friction=..., contact=...,
  approach=...)` (the surfaces and `grasps_on` are not bound there yet).

- **Breaking:** grasps of a wall seen from one side only, as a bowl's rim seen from above, its outer
  wall hidden under the rim: `GraspOptions::shell`, the thickest such wall a grasp may close across
  (`None` by default, which changes nothing). From a patch whose other side no camera saw, the line
  in along its normal meets, within that thickness, space every ray around it went past on its way
  to something farther: the second contact goes there, so `width()` is the most the wall can be,
  never past the `opening`, and the `Grasp` says so (`one_sided`). Inside a solid no ray goes, so a
  box seen from one side is closed on only across all of it, and a patch whose other side was seen
  keeps to the seen pair. A wall that leans is closed on level too, where the level line through the
  patch is inside its friction cone, so a hand comes straight down across a flared rim where one
  closing along the wall's normal leans with it. These grasps follow every grasp of two patches
  seen (1.5 cm suits a bowl's
  rim: LIBERO's, its wall 3 mm thick, seen from 40 cm above, is grasped 5.5 to 11.5 mm wide at its
  rim). The evidence is where each point was seen from, `Surface::seen_from`, which `from_depth` and
  `from_points` record and `join` carries (infinite where a joined surface did not say). The points
  seen from exactly one place make one view, so looks joined from one place are one view, and a
  cloud fused from several views must not be given one place. A `GraspOptions` literal needs
  `shell: None`, and a `Surface` literal `seen_from: vec![]` (or `..Default::default()`); in Python,
  `GraspOptions(shell=0.015)`.

- **Breaking:** a place says how far its hand may open where it lets go: `Place::open`, finger
  values as `Held::locked` names them, in the order preferred (empty by default, which changes
  nothing: the hand opens to where the robot was loaded with them, and a spot the open fingers do
  not fit is refused at its release). With openings named, where the hand lets go is solved with
  the fingers where they hold the object, each pose lets go at the first opening with which the
  hand is clear there and on its move away, and the plan says which (`TaskPlan::opening`): a hand
  can set an object down where it has no room to open all the way. The openings share the
  place's IK, move in and transit; each one tried costs a move away.
  `GraspOptions::openings()` gives the openings a grasp with `narrow` may come in at, widest
  first, to offer a place the same. A `Place` literal needs `open: vec![]`; in Python,
  `Place(..., open=[{...}, ...])`, `TaskPlan.opening` and `GraspOptions.openings`.

- **Breaking:** a place says how many plans will do: `Place::enough` (`None` by default: every
  pose is tried, as before). With a count, the poses are tried in their order, 32 and then twice
  as many each time, until that many have got through, and the plans are the first that many in
  the poses' order, as trying every pose gives them: a place given every spot of a table plans
  the first that will do without the IK and the motions of the rest. A place none gets through
  has tried every pose and says where each stopped. A `Place` literal needs `enough: None`; in
  Python, `Place(..., enough=1)`.

- **Breaking:** a pick says how many plans will do: `Pick::enough` (`None` by default: every
  grasp is tried, as before). With a count, the grasps are tried in their order, 32 and then
  twice as many each time, until that many have got through, and the plans are the first that
  many in the grasps' order, as trying every grasp gives them. A pick's time is the IK of its
  grasps, so one given the hundreds `grasps_on` finds and asked for the first that gets through
  stops after the first few. A pick none gets through has tried every one and says where each
  stopped. A `Pick` literal needs `enough: None`; in Python, `Pick(..., enough=1)`.

- **Breaking:** grasps whose fingers come in narrower than they open, where something stands
  near: `GraspOptions::narrow` (false by default, which changes nothing). Open fingers beside a
  wall or a neighbor are in it, though the grasp needs a fraction of the opening. With `narrow`,
  `grasps_on` takes each finger to slide along the hand's y axis and checks every grasp at
  sixteenths of `opening`; each `Grasp` comes back with the `opening` that keeps its fingers
  farthest from what is not the object's, the floor included, which with something within the
  fingers' travel is the narrowest the hand is clear at: a planner lets the fingers touch what
  they grasp and nothing else. Of openings as far from all that it takes the one with the most
  room from the object's other sides, then the widest, so a box with nothing near keeps
  `GraspOptions::opening`. A grasp the open hand has stays where it is; where the open hand is clear at no
  depth, the hand comes in as deep as it is clear narrower. LIBERO's bowl 4 cm
  from a wall twice as tall is grasped all round so (698 grasps where the open fingers find
  654), in 10.7 ms where the open hand's check takes 8.9: a grasp that is found is checked
  across each finger's whole slide, so a hand of many spheres pays more. Without `narrow` every grasp's `opening` is
  the options'. A `GraspOptions` literal needs `narrow: false`; in Python,
  `GraspOptions(narrow=True)`.

- **Breaking:** a `Follower` can hold an object: `FollowerOptions::holding` takes the `Held` the
  trajectory was planned holding. The rest of the motion, and any stop, is checked with the object
  in the hand, contacts naming it (`"held by hand 0"`), and replans and re-anchors carry it. The
  follower's robot and the devices its ticks are given are those that do not hold it, as for a
  problem's `holding`, so one device plans and follows a carry; the first tick on a device builds
  the one that holds it, and later ticks on that device reuse it. `None`, the default, is as before. A `FollowerOptions` literal needs
  the new field (`holding: None`).

- Options a host can check where they are set, before a request uses them: `IkOptions::check()`,
  `PlanOptions::check()` (its search, shortcuts and goal IK included), `RrtOptions::check()`,
  `ShortcutOptions::check()`, `ToolPathOptions::check()` and `JoinOptions::check()` return the
  `Error::Input` their call returns for those options. In Python, `options.check()` raises it.

### Changed

- **Breaking:** a robot mask judges each reading by where its point lies:
  `image.robot_mask(&robot, &q, padding, &holding)` marks a reading the robot's where its point
  is inside a collision sphere grown by the padding, and `RobotMask::on_robot`, a flag per pixel,
  replaces `RobotMask::depth`. Before, every reading at or behind where its ray entered a grown
  sphere was the robot's, so a camera the hand carries, which stands inside or beside the hand's
  grown spheres, lost all it saw: a camera on the Panda's palm, inside a sphere grown by the 4 cm
  a mask is padded with, lost its whole image. Now it keeps every reading past the hand, and a
  camera across the table keeps the surfaces seen just past the arm's outline. A sensor's stray readings there, farther than the padding behind the arm, are obstacles
  where they were dropped. `holding` names what the hands hold (`&[]` for nothing), a `Held` each
  as a problem's `holding` takes one: its readings are left out and the space it fills is free, as
  the robot's, without attaching it. In Python the mask was not bound.

- **Breaking:** `Surface::from_depth(&device, &image, &object, robot_mask, radius)` takes the
  device and reads the image's rows on its threads; the surface is the same to the bit. On the M4
  Pro's 14 threads a 640×480 image of a tabletop with the table's 148,461 normals takes 2.7 ms
  where 28, and on the Ryzen's 32 threads 3.4 ms where 34 (`examples/perception.rs`).

- Fusing a frame or building a grid with a robot mask costs by the robot's volume, not by its
  spheres times the grid's points: which points the robot fills is found once, sphere by sphere
  over the points near each, for both devices, and which points those are is the same to the bit
  on the CPU. A
  640×480 frame fused into a map of 1.8M voxels with the mask of a Panda fitted with 573 spheres
  (`examples/perception.rs`, runs alternated with the release before): 2.2 ms where 49 on the M4
  Pro's CPU and 1.6 where 11.2 on its GPU, 5.4 ms where 41 on the Ryzen's 32 threads and 1.2 where
  3.0 on the Radeon 8060S. With the Panda's 61 spheres: 1.9 ms where 6.3, 1.4 where 2.5, 5.2 where
  8.8 and 0.9 where 1.4. Calls without a mask take what they took, with the same results to the
  bit.

- A frame fused into a map on the CPU device costs by what its camera sees, not by the map's
  size: a row of the map at a time, the fuse visits the voxels the image may see, and past them
  only those holding one of its readings and those a mask frees, with the same map to the byte.
  A 640×480 frame of a tabletop fused into a map of 13.4M voxels (2.9 m by 2.9 m by 1.55 m at
  1 cm) takes 2.4 ms on the M4 Pro's 14 threads from a camera across the table, 1.4 times what a
  map of 1.8M voxels it sees nearly all of takes, and 1.6 ms from one on the Panda's hand, which
  sees less of it; with the Panda's mask, 3.1 and 1.9 ms. On the Ryzen AI Max+ 395's 32 threads
  it takes 5.2 and 3.1 ms, and 6.5 and 3.5 with the mask (`examples/perception.rs`, the quickest
  median of eight runs, at a load of 3.5 to 7.5 on the M4 Pro and 2 to 10 on the Ryzen). The
  GPU's fuse is as it was: it sends the map to the device and back, so a frame fused into that
  map takes 8.3 to 9.7 ms on the Radeon 8060S and 4.7 to 6.5 on the M4 Pro's GPU, and a map that
  large now fuses quicker on a CPU device; any device fuses a map.

- **Breaking:** a robot's `SphereOptions` are checked whether its spheres are fitted or given
  (`RobotOptions::collision_model`), before anything is read: a budget of zero, or a tolerance or
  voxel size that is not positive, is an `Error::Input` from `Robot::load`, `from_urdf` and
  `assemble`. With a collision model they were not read and the robot loaded; without one, `load`
  and `assemble` returned an `Error::Load`.

- **Breaking:** options a call cannot use are input errors where the call ran before: IK
  tolerances that are zero, negative or not a number, which no seed meets, so every problem came
  back unsolved; collision weights that are not finite, margins that are not numbers, and
  infinite margins with a weight on them, each of which makes the cost not a number (an infinite
  margin without a weight still asks for exact clearances at any distance); a plan's acceleration
  and velocity weights that are not finite; an infinite `initial_step`; and seeds, control points
  or samples per span past what 32 bits count, alone or as a batch's problems times their seeds
  (a plan's pose goals at the sixteen times their `goal_ik` seeds its IK may come to try): a
  trillion IK seeds ended the process when it allocated for them. `Device::evaluate` refuses
  such weights and margins too, where it returned costs that were not numbers, `ik::failures`
  the options `solve_ik` refuses, and `servo` a `slow_within` or `halt_within` that is not a
  number, with which it never slowed.

- **Breaking:** a pick's grasps each say where the fingers are: `Pick::locked` is a
  `Vec<HashMap<String, f32>>`, the locked joints' values per grasp (empty: every grasp as loaded,
  as before), where it was one map for the task. One pick then takes grasps that come in at
  different openings, as `grasps_on` gives them with `GraspOptions::narrow`, and plans by any:
  `pick` groups a call's grasps by those values, a relocked robot for each group, and each plan's
  `candidate` numbers the pick's grasps and its `held` starts with its grasp's values. A pick
  that set `locked: narrow` sets `locked: vec![narrow; grasps.len()]`; in Python,
  `Pick(..., locked=[narrow] * len(grasps))`.

- **Breaking:** a held capsule is covered by spheres along its axis, and a held cylinder that is thin or
  slender (a disc, a rod) by one layer of rings of spheres or one line of them, where that reaches
  the tolerance within the budget: without fitting, in microseconds where a fit took 4 to 25 ms,
  and with fewer spheres than their fit took (a 14 cm plate: 145 within 4.7 mm of it). Their spheres, and so clearances and contacts measured to them, differ from the fit's.
  Stout cylinders, and discs the budget cannot bring within the tolerance (a plate at the default
  64), are fitted as before.

### Fixed

- A plan with a `time_budget` longer than the clock counts (`Duration::MAX`) panicked on adding
  it to the time; it now has no deadline. A plan whose search had `RrtOptions::max_iterations`
  of `usize::MAX` found nothing where it had to search again without its clearance: the samples
  its first search drew, added to the limit, wrapped.

- A call that held objects in several hundred ways at once, as a `place` after each of a pick's
  plans, with more than one candidate for each, could overflow the stack of one of the device's
  threads and abort the process: on an M4 Pro, 609 holds of a box with two poses each did so in
  most runs, and 1,500 in every one. What runs for each hold waits for work it hands the device's
  threads, and a thread that waits took up another hold's on top of its own, as deep as there
  were holds. `place`, `pick` (its lift, and its grasps' finger values), `handover` and problems
  with a `holding` now keep no more in hand than the device has threads. On the M4 Pro's CPU a
  `place` of up to 4 holds takes what it did, and one of 16, 64 or 609 is 5 to 11% quicker.

- **Breaking:** spheres fitted to a link mesh of more than one closed shell, and how
  `Robot::qualify` measures spheres against one. The side of the nearest face decided inside
  from outside: in a shell
  beside a face of another that is buried in it, a center read as outside, so the fit had no
  candidates there and `qualify` gave a sphere there its distance to that face as overhang; and
  around a shell that faces inward, the fit put spheres outside the link, 13 cm past a box with
  an inward cube beside it. Inside a closed mesh is now where its surface winds around the
  point, as for distance grids: shells that overlap are their union, and a shell the grid's
  points fall inside is solid whichever way it faces (one that faces inward and is thinner than
  the grid still gets spheres on its outside). Meshes with open edges, and every robot in this repository, are fitted
  as before, sphere for sphere. A robot with such a mesh is fitted differently, and clearances
  and contacts measured to its spheres differ: the reBot's arm, where 41 of a link's grid points
  read the other way, keeps its sphere count and how far they reach, and the handover and join
  benchmarks on it give the results they did.

- Distance grids of meshes with more than one closed shell (`SdfGrid::from_mesh`, and scene
  meshes `World::load` reads). The side of the nearest face decided inside from outside, which is
  right for one shell only: where a table's leg ran up into its top, the top read free above the
  leg's end; a shell facing the other way than the rest read hollow, with solid around it. A
  point is now inside where the surface winds around it: shells that face the same way are their
  union where they overlap, a shell apart from the others is solid whichever way it faces, a
  shell facing in inside one facing out is a cavity as before, and a shell that crosses itself is
  solid wherever it encloses space. Where two shells that face opposite ways overlap, the overlap
  reads free. Grids of a single shell that does not cross itself are bit for bit what they were.

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
  [benchmark guide](docs/benchmarks.md#fitting-collision-spheres) gives the runs and
  `examples/fitting.rs` reproduces the fixtures.
- Faster CPU collision checks, contact queries, IK, planning, tasks, depth-grid construction and
  occupancy updates, with results the same to the bit. In six alternating runs, the best CPU
  times for 1,000 IK targets among clutter fell 9% on a Ryzen AI Max+ 395 and 4% on an M4 Pro;
  256 plans with four optimized seeds took 22% and 6% less time, respectively. The
  [benchmark guide](docs/benchmarks.md#how-the-speed-came) gives the times, lower quartiles and
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
  takes 11 ms on an M4 Pro where it took 18 ([Benchmarks](docs/benchmarks.md#how-the-speed-came)).
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
  run ([Benchmarks](docs/benchmarks.md#how-the-speed-came)).
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
  ([Benchmarks](docs/benchmarks.md#how-the-speed-came)). A GPU's results move in their last
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

[Unreleased]: https://github.com/natsuku-lab/batchplan/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/natsuku-lab/batchplan/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/natsuku-lab/batchplan/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/natsuku-lab/batchplan/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/natsuku-lab/batchplan/compare/v0.7.0...v0.8.0

[historical-robokit]: https://github.com/natsuku-lab/batchplan/blob/9d7823a53148a3f571fd5615d0ebbd5818d8f1d4/docs/benchmarks.md#against-robokit
