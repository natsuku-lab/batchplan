# API

Everything a program uses, by task. Names are in `batchplan::*` unless a module is shown (`datagen::`, `npy::`, `lerobot::`, `rrt::`, `shortcut::`, `rng::`). [docs.rs](https://docs.rs/batchplan) has every signature and field; the guide named under each heading explains the behaviour.

Every batched call has the same shape: a `Device`, the `Worlds` uploaded to it, a slice of problems that each name their world by index, and an options struct whose `Default` is a good start. Results keep the problems they answer: `best(problem)` gives one problem's answer, and `solved()` yields each solved problem with its index and answer.

Options and problems are built with struct update syntax (`..Default::default()`, `..PlanProblem::new(..)`). Results, `Trajectory` and the enums that may grow (`Obstacle`, `Occlusion`, `Weighing`, `Origin`, `Error`) are `#[non_exhaustive]`: read their fields, and give a `match` on them a wildcard arm.

## Robots

Guide: [robots.md](robots.md).

| | |
|---|---|
| `Robot::load(path, &RobotOptions)` | A robot from URDF, MJCF or OpenUSD (feature `usd`). |
| `Robot::from_urdf(text, &RobotOptions)` | A robot from a URDF's text, as ROS publishes it on `/robot_description`. |
| `Robot::assemble(&[Part], &RobotOptions)` | One robot from several descriptions, such as two arms at a table; each `Part` has a name prefix and a base pose. |
| `RobotOptions` | The base link, the hands (`ee_links`) and their fingers where they are part of the palm (`finger_links`), the default configuration, locked joints, joint limits and their scales, `package://` directories, an SRDF, OpenUSD variants, and the collision spheres, with tighter tolerances for some links (`sphere_tolerances`). |
| `JointLimits` | One joint's position, velocity, acceleration and jerk limits, replacing its description's. |
| `CollisionModel` | Collision spheres and self-collision settings by link: `load` and `save` a file, `to_json` and `from_json` the same text, `assemble` the model of an assembled robot; `robot.collision_model()` is a loaded robot's. |
| `SphereOptions` | How spheres are fitted to the links' collision geometry when no collision model is given: the budget, the tolerance, and `max_overhang`, a limit on how far any sphere reaches past the geometry, which holds a given model too ([guide](robots.md#collision-spheres)). |
| `Robot::attach(&AttachedObject)`, `Robot::detach(name)` | The robot holding an object fixed to one of its links, or without it ([guide](robots.md#holding-objects)). |
| `Robot::relock(&joints)` | The robot with joints it locked at load held at new values, as a torso that moved ([guide](robots.md#kinematics)). |
| `dof`, `joint_names`, `lower`, `upper`, `continuous`, `default_q`, `max_velocity`, `max_acceleration`, `max_jerk` | The robot's joints and their limits. |
| `hands`, `hand_joints(hand)`, `hand_links(hand)`, `finger_links(hand)`, `tip(hand)`, `hand_spheres(hand)`, `ee_pose(q, hand)`, `link_pose(q, link)` | Hands and forward kinematics: a hand's joints, its links with its fingers (for allowed contacts), its fingers alone, how far its spheres reach past its frame, and those spheres in its frame (`HandSphere`). |
| `spheres(q)`, `reach()` | Every collision sphere at a configuration, and a bound on how far they reach from the base in any. |
| `hand_reach(hand)`, `HandReach::excludes(pose, &IkOptions)` | An outer bound on a hand's IK targets (`None` for a chain with a prismatic or relockable joint, or fewer than two moving joints); `excludes` is true only for a pose beyond it, IK's tolerances allowed ([guide](robots.md#kinematics)). |
| `qualify()` | How well the spheres fit the geometry (`Qualification`): per link (`LinkQualification`) the `overhang`, the `uncovered` share of the surface, how deep the `gap` and the `thickness`, and each hand's tip ([guide](robots.md#qualifying-a-robot)). |
| `breach(constraints, q)` | Why a configuration breaks `HandConstraint`s, if it does, tested as `plan` tests its start and goals: to keep only goals that keep them before planning. |
| `MeshModel::try_new(&robot)`, `MeshModel::clearance` | Clearances on the links' meshes instead of the spheres, to check plans against the real geometry: a `Clearance` (`world`, `self_collision`) per configuration ([guide](robots.md#checking-against-meshes)). |

## Worlds and perception

Guide: [perception.md](perception.md).

| | |
|---|---|
| `World::new(obstacles)` | One scene: a list of obstacles. |
| `World::remove(obstacle)` | Takes an obstacle out, as a hand picking it up does, and renumbers the touches of those after it. |
| `World::touch`, `Touch::new(obstacle, links)`, `Touch::held(obstacle, hand)` | Links allowed to touch an obstacle, such as the hand that grasps it ([allowed contacts](perception.md#allowed-contacts)), or what a hand comes to hold, as an object set down on a table ([holding objects](robots.md#holding-objects)). |
| `World::always_touching(&robot)` | The touches of links that touch an obstacle in every configuration, as a link turning just above the base touches its table. |
| `World::load(path, &SdfOptions)` | A scene's static geometry from MJCF or OpenUSD (feature `usd`). |
| `Obstacle::cuboid`, `sphere`, `cylinder`, `capsule` | Axis-aligned shapes. `placed(pose)` moves and turns one, `pose()` returns its frame, `distance(p)` its signed distance and gradient (`Distance`). |
| `Obstacle::from_grid(grid)`, `Obstacle::from(grid)` | A distance grid as an obstacle, in the world's frame. |
| `SdfGrid::from_mesh`, `from_points`, `from_depth`, `try_new` | Distance grids from a closed mesh, surface points, one depth image, or your own values. `lower_corner()`, `upper_corner()` and `distance(p)` read one. |
| `SdfOptions` | A grid's voxel size and padding. |
| `DepthImage`, `Intrinsics` | A depth image with its pinhole intrinsics and camera pose. `points(robot_mask)` gives each pixel's reading as a point, `None` without one or on the robot. |
| `DepthImage::filtered(&DepthFilter)` | The image's depths without the readings between two surfaces that a depth camera gives along every edge. `DepthFilter` holds the camera's `noise`, the `steepest` a surface is turned to the rays and the `deviations` noise is taken to reach ([guide](perception.md#readings-between-two-surfaces)). |
| `DepthImage::robot_mask(&robot, q, padding, &holding)` | Which of the image's readings are of the robot or of what it holds (`RobotMask::on_robot`, a flag per pixel), so grids, maps and surfaces leave them out ([guide](perception.md#masking-the-robot)). |
| `RobotMask::with_shapes(&device, &image, &shapes, margin)` | The mask with shapes the caller holds left out as the robot is: a reading within the margin of one is not an obstacle, and the space the shape fills is free in a map ([guide](perception.md#shapes-the-caller-holds)). |
| `RobotMask::with_band(&device, &image, margin, pixels)` | The mask with the readings flagged that a camera gives just behind the outline of a flagged thing (a sphere, a held object, a shape), within `pixels` of it; after `with_shapes`, the shapes' too ([guide](perception.md#shapes-the-caller-holds)). |
| `sdf::on_shapes(&device, &shapes, &points, &reach)` | Which of `shapes` each of `points` lies on (`OnShape`: the `shape` and the point's signed `distance` from its surface), where the point is within its `reach`; `None` otherwise ([guide](perception.md#shapes-the-caller-holds)). |
| `Occlusion` | What a grid assumes behind observed surfaces: free, solid, solid except the robot's shadow, or solid in a band behind surfaces. |
| `OccupancyMap::try_new`, `integrate`, `occupancy`, `grid` | Depth images fused over time into occupancy, then into a distance grid ([guide](perception.md#fused-maps)). |
| `OccupancyMap::with_weighing(Weighing)` | How much one image counts: as evidence (`Weighing::Evidence`, the default, for a stream of frames), or deciding what it sees (`Weighing::Latest`, for a caller that looks once every few seconds). |
| `OccupancyMap::free(&shape, margin)`, `clear(&shape, margin)`, `part(&region, padding)`, `counts()`, `occupied()` | What a caller knows of a map: space that is empty, space that is no longer known, the map inside one region, how many voxels are occupied, free and never observed (`MapCounts`), and where the occupied ones are ([guide](perception.md#what-a-caller-knows-and-one-objects-part)). |
| `OccupancyMap::occupy(&shape, margin)` | Marks the voxels within `margin` of a shape occupied, as one hit or the voxel's evidence where that is stronger ([docs.rs](https://docs.rs/batchplan/latest/batchplan/sdf/struct.OccupancyMap.html#method.occupy)). |
| `OccupancyMap::from_shape(&shape, &SdfOptions)` | A map of a primitive as solid voxels, free out to the padding ([docs.rs](https://docs.rs/batchplan/latest/batchplan/sdf/struct.OccupancyMap.html#method.from_shape)). |
| `OccupancyMap::refined(&device, &views, divisions)` | The map with each voxel split into `divisions` cubes along each edge, the same views fused in them ([docs.rs](https://docs.rs/batchplan/latest/batchplan/sdf/struct.OccupancyMap.html#method.refined)). |
| `OccupancyMap::take_readings(&take, &keep, padding, inside)` | The occupied voxels that hold a reading of `take`, none of `keep`, and pass `inside`, separated into a part of their own; `None` where there are none ([docs.rs](https://docs.rs/batchplan/latest/batchplan/sdf/struct.OccupancyMap.html#method.take_readings)). |
| `OccupancyMap::take_surfaces(&device, &views, selected, padding, inside)` | The occupied voxels hit by selected depth pixels (`selected(view, pixel)`) and meeting no other reading, separated into a part of their own; `None` where there are none ([guide](perception.md#what-a-caller-knows-and-one-objects-part)). |

## Devices

Guide: [devices.md](devices.md).

| | |
|---|---|
| `Device::gpu(&robot)`, `Device::gpu_named(&robot, name)` | A GPU through wgpu: the fastest one found, or the first whose name contains `name`. An error without the default `gpu` feature. |
| `Device::cpu(&robot)`, `Device::cpu_threads(&robot, n)` | CPU threads: one per core, or exactly `n` that the device owns. |
| `Device::upload(&[World])` | The `Worlds` every batched call takes, prepared for this device. `Worlds` derefs to `[World]`. |
| `Device::with_robot(&robot)` | A device for a changed robot, such as one holding an object, on the same GPU or threads; uploaded worlds work on both. |
| `Device::evaluate` | Collision cost, gradient and clearances for a batch of configurations (`Evaluation`), weighted by `CollisionWeights`. |
| `Device::contacts`, `Device::contacts_holding` | What each configuration of a batch comes within a distance of, nearest first (`Contact`): a link and an obstacle, or two links, with the distance between them. `contacts_holding` takes a `Held` or `None` per configuration, on the device that holds nothing, and names the held object's link `"held by hand N"`. |
| `Device::warmup` | Builds the kernels a GPU builds on first use and runs one small call of each kind, before anything is timed. |
| `Device::install(op)` | Runs `op` on the device's threads, so rayon's parallel loops in it run there and not on the global pool; the calling thread waits ([docs.rs](https://docs.rs/batchplan/latest/batchplan/device/struct.Device.html#method.install)). |

## Inverse kinematics

Guide: [planning.md](planning.md#inverse-kinematics).

| | |
|---|---|
| `solve_ik(&device, &worlds, &[IkProblem], &IkOptions)` | Collision-free IK from many seeds per target (`IkResult`). |
| `IkProblem::new(world, target)` | A one-handed robot's hand at `target`. The fields `targets` (a pose or `None` per hand), `seed` (where seed 0 starts), `hold` (joints that stay where the seed has them) and `holding` (an object a hand holds, a `Held`) cover the rest. |
| `IkOptions` | Seeds per problem, iterations, tolerances and collision weights. `options.check()` returns the input error `solve_ik` would, for a host to refuse options when they are set; `PlanOptions`, `RrtOptions`, `ShortcutOptions`, `ToolPathOptions`, `JoinOptions` and `TaskOptions` have the same. |
| `IkResult::best(problem)`, `nearest(problem, q)`, `solved()` | The seed with the most clearance, the seed nearest `q`, and every solved problem. |
| `ik::failures(&device, &worlds, &result, &IkOptions)` | Why each unsolved problem has no solution (`IkFailure`): out of reach, or reached only in collision, with its contacts. |

## Planning

Guide: [planning.md](planning.md#planning-a-path).

| | |
|---|---|
| `plan(&device, &worlds, &[PlanProblem], &PlanOptions)` | Collision-free B-spline paths, one search per problem (`PlanResult`). |
| `PlanProblem::new(world, start, goal)` | From `start` at rest to `goal` (a `Goal`): a configuration, or a `Pose` per hand (`Goal::Poses`), which `plan` solves IK for from `start`. The fields `hold` (joints that stay still), `start_motion` (a `StartMotion`, to replan from a moving state), `constraints` and `holding` (a `Held`; [holding objects](robots.md#holding-objects)) cover the rest. |
| `HandConstraint::new(hand, pose)` | Where a hand may be along the whole path, measured from `pose` ([guide](planning.md#constraints-along-the-path)): `tilt` (its z axis within a cone), `twist` (its turn about that axis) and `position` (a box), all free by default, and `relative_to` another hand. `RrtProblem`, `Pick`, `Place`, `Sequence` and `FollowerOptions` take them too. |
| `PlanOptions` | Control points, optimization seeds and iterations, the search (`rrt`: `RrtOptions`), shortcutting (`shortcut`: `ShortcutOptions`), IK for pose goals (`goal_ik`: `IkOptions`) and a `time_budget`. |
| `PlanResult::best(problem)`, `solved()` | The shortest valid path's control points, and every solved problem. Every seed's control points are in `paths`, empty where a seed has none. |
| `PlanResult::failures` | Per problem, why it has no valid path (`PlanFailure`): the start or the goal collides (with its contacts), the goal's poses are out of reach, no path, no valid spline, or out of time ([guide](planning.md#why-a-problem-failed)). |
| `rrt::connect`, `shortcut::shortcut` | The search (`RrtResult`) and the shortcutting that `plan` runs, on their own, for waypoint paths. |
| `follow(&device, &worlds, &[ToolPath], &ToolPathOptions)` | A hand moved in straight lines through poses, the joints a path holds (`hold`) still, and other hands that hold its object with it (`holding`) taken along ([guide](planning.md#moving-the-hand-along-a-path)): per path (`ToolPathResult`), the `fraction` it covers, why it `stopped` short (`ToolPathStop`), what a collision there is with (`contacts`), and the motion's control points. |
| `join(&device, &worlds, &[Sequence], &JoinOptions)` | Consecutive motions run without stopping between them ([guide](planning.md#joining-motions)): per sequence (`JoinResult`), one path for each stretch the robot moves without stopping, and each join's blend radius, `None` where it stops. A `Sequence` gives each join's radius (`blends`, 0 to stop), what every blend keeps (`constraints`) and what a hand carries (`holding`). |

## Tasks

Guide: [planning.md](planning.md#picking-up-and-setting-down).

| | |
|---|---|
| `pick(&device, &worlds, &[Pick], &TaskOptions)`, `place(.., &[Place], ..)` | A whole task in one call: per task (`TaskResult`), a `TaskPlan` for every candidate that gets through, least `cost` first and the quickest among equals, or a `TaskFailure`. |
| `Pick::new(world, start, object, grasps)`, `Place::new(world, start, held, releases)` | Hand 0 picking up `object` by any of its grasps, or the holding hand setting `held` down at any of its releases, with 10 cm straight moves in and away. The fields `support`, `constraints` (on the transit), `enough` (how many plans will do), `transit` (`false`: the straight moves alone), `transit_world` and `depart_world` cover the rest, and a place's `idle` (the other arms on its transit), `open` ([finger openings](grasps.md#finger-openings)) and `onto` (another hand's `Held` that the object is set on or into, [which stays where it is](planning.md#onto-what-another-hand-holds)). |
| `Hold { pose, locked, cost }`, `Hold::new(pose, opening, &gripper)` | One way to hold what a pick, a turn or a pull grips: the hand's pose in the object's frame, the fingers' values (empty: as loaded; [finger openings](grasps.md#finger-openings)) and its cost. A `Pose` is a hold at cost 0 with the fingers as loaded. Costs order what comes back; candidates are tried in their order ([guide](planning.md#costs-and-how-many-will-do)). |
| `Release { pose, drop, cost }` | One place to let go: the object's pose, how far it drops from there and its cost. A `Pose` is a release at drop and cost 0; `Release::from(&rest)` and `Release::from(&fall)` take a rest's or a fall's ([guide](placing.md#releases)). |
| `Pick::also` (a `HandGrasps`) | A second hand that picks the object up with the first: its `hand` and its `grasps`, one per grasp of the pick's, entry by entry; each pair is a candidate, costing both holds' sum. The plan's `held` names it in `Held::also`, and `place` sets down what two hands hold ([guide](planning.md#two-hands)). |
| `Pick::body` (a `PickBody`), `Pick::lift_world(&departure)` | The body lifted when the hand grasps one part of a larger object: its `shape` and the `other_parts` that leave with the grasped one (`None` lifts `object` alone); `lift_world` is the world to run the lift in with the returned hold attached ([guide](planning.md#worlds-bodies-and-other-arms)). |
| `TaskPlan` | One plan: its `candidate` and `cost`; the `transit` (`None` without one), `approach`, `turn`, `pull` and `depart` trajectories (`motions()` those it has, in order; `duration()`); what is `held` after it (after a place onto what another hand holds, that hold carrying the object), where the object is `placed` and what rode on it is `carried`, the body `falling` after a release with a drop, the `opening` of a place's `open` it let go at, a handover's `meeting`; `begin()` and `end()`, the configurations it starts and ends at. |
| `TaskFailure`, `CandidateStop` | Why no candidate got through, which it prints as a sentence: per candidate, its `step` (`TaskStep`), the reason (`why`), the `contacts`, and how far a turn `turned` or a pull got (`pulled`). `step()` is the furthest step any reached, `at(step)` how many stopped there. |
| `Held` | What a hand holds, as a pick's plan gives it: the `hand`, the `object` in its own frame, the `grasp`, the fingers' `locked` values and the `spheres` that cover it. `also` names other hands that hold it too (`Grip { hand, grasp }`): calls that hold it keep each at its grasp ([two hands](robots.md#holding-objects)). `touch_links` names other links that may touch it, as a hand that has let go and backs off. `carries` lists what rides on it, shapes in its frame that every call holding it checks with it ([what rides on it](robots.md#holding-objects)). `Held::attachment(&robot)` gives the `AttachedObject` for a device that holds it ([guide](robots.md#holding-objects)). |
| `TaskOptions` | IK, planning and tool path options, the straight moves' speed cap (`tool_speed`, `f32::INFINITY` for none), where it holds (`tool_speed_near`: within that distance of the grasp or release; `None`, all along), how much of each they must make (`min_fraction`), and the spheres that cover what a pick or a moving turn carries (`held_spheres`). |
| `handover(&device, &worlds, &[Handover], &TaskOptions)`, `Handover::new(&robot, world, start, held, to)` | One hand passing what it holds (`held`) to hand `to` ([guide](planning.md#handing-over)): per task, a `TaskPlan` for every meeting and taker's grasp that gets through, quickest first, with its `meeting` and the taker's `held`. `handover_poses(&robot, start, &held, to)` proposes meetings and `handover_grasps(&held)` the taker's grasps. |
| `turn(&device, &worlds, &[Turn], &TaskOptions)`, `Turn::new(world, start, object, grasps, angle)` | A hand turning an obstacle by `angle` about a `Screw` (`screw`; `None`: the hand's own z axis), as a knob, a cap, a lever or a wheel ([guide](turning.md#turning-a-thing-about-an-axis)): per task, a `TaskPlan` for every grasp that gets through, with the turn in `turn` and the object turned as `placed`, or each grasp's stop (`TaskStep::Turn`, how far it `turned`). `support` names what the object may touch as it turns, and `stays` keeps the obstacle in place. |
| `pull(&device, &worlds, &[Pull], &TaskOptions)`, `Pull::new(world, start, object, grasps, direction, distance)` | Hand 0 gripping `object` by any of `grasps` and drawing it `distance` meters along the world `direction` ([guide](turning.md#drawing-a-handle-along-a-line)): per task, a `TaskPlan` with the draw in `pull`, or a stop at `TaskStep::Pull` with the distance planned (`pulled`). |

## Grasps

Guide: [grasps.md](grasps.md).

| | |
|---|---|
| `robot.gripper(hand, &GripperCalibration)` | A hand as a parallel-jaw `Gripper`, from what was measured of it (`GripperCalibration::new(loaded_gap, closed_gap, closed)`, with `capacity` and the pads' `friction`), for every call that grasps with it; an `Error::Input` where the calibration contradicts itself or the model ([guide](grasps.md#the-gripper)). |
| `Gripper` | `hand()`, `opening()` (the loaded gap), `closed_gap()`, `capacity()`, `friction()`, `spheres()`, `tip()`, `sliding()` (whether its jaws slide, as `narrow` needs), and `openings()`, `locked_at(opening)` and `opening_of(&locked)` for the fingers' openings ([finger openings](grasps.md#finger-openings)). |
| `grasps(&obstacle, &gripper, &ShapeGraspOptions)` | Grasps of a box, ball, cylinder or capsule in its frame, closing across at most the gripper's capacity, above a `floor` if given; or why there are none ([guide](planning.md#picking-up-and-setting-down)). |
| `Surface::from_depth(&device, &image, &object, robot_mask, radius)`, `Surface::from_points(&points, seen_from, radius)`, `surface.join(other)` | A surface from one depth image, the `object` flags saying which pixels are the object's, or from a cloud seen from one place; and two views as one. A `Surface` is plain data: `points`, `normals`, `object` and `seen_from` ([guide](grasps.md#grasps-of-what-cameras-see)). |
| `grasps_on(&device, &gripper, &[Surface], &SurfaceGraspOptions)` | Grasps of any shape from what cameras see of it: per surface, each `Grasp` (the hand's `pose`, the two `contacts`, `width()`, whether it is `one_sided`, and the `opening` the fingers come in at) ([guide](grasps.md#grasps-of-what-cameras-see)). |
| `SurfaceGraspOptions` | `turns`, `floor`, `overlap`, `contact` and `approach`; `shell` ([walls seen from one side](grasps.md#walls-seen-from-one-side)), `narrow` ([fingers that come in narrower](grasps.md#fingers-that-come-in-narrower)), `clearance` ([room a planner needs](grasps.md#room-a-planner-needs)), `close_on` ([closing on a point](grasps.md#closing-on-a-point)), `first_touch` and `near` ([the order](grasps.md#what-comes-back)). |
| `Grasp::closes_on(&gripper, at, radius)` | Whether a ball in the surface's frame lies wholly in the gap the grasp closes ([guide](grasps.md#closing-on-a-point)). |
| `Grasp::unseen_span()` | How far a grasp closes across what no camera saw: its width where it is `one_sided`, else 0. As a hold's cost, it puts the grasps seen on both sides first ([guide](grasps.md#using-them-in-a-pick)). |
| `options.order_for_turn(&gripper, &mut grasps, screw)`, `grasp.moment(&screw, &gripper, &options)` | Grasps ordered by the moment their fingers can give about a turn's axis, greatest first (`None`: each hand's own z axis), and one grasp's moment ([guide](turning.md#turning-a-thing-about-an-axis)). |

## Placing

Guide: [placing.md](placing.md).

| | |
|---|---|
| `place_poses(&obstacle, pose, turns)` | An object's symmetric turns at one spot ([guide](planning.md#picking-up-and-setting-down)). |
| `Release::drop`, `TaskPlan::falling` | How far the object drops after release (meters along world −z; 0: set down there), and the body as the move away is checked with it, for a follower to check against ([guide](placing.md#releases)). |
| `rests_on(&device, &[RestProblem], &[Surface], &RestOptions)` | Where a held object rests on a measured support: per problem, each `Rest` (the object's `pose`, the `support` it touches, how far it can be `tipping`, and its `cost`), at every turn ([guide](placing.md#rests-on-what-cameras-see)). |
| `RestProblem::new(object, surface)` | `object` set down upright on surface `surface`; `rotation` turns it as it is set down, and `near` is a point to set it down near ([guide](placing.md#rests-on-what-cameras-see)). |
| `RestOptions` | `turns`, `spacing`, `touching`, `clearance`, `tipping`, `friction`, and `look_closer` and `seen_from_above_holds` (both on by default), for `rests_on`, `rest_failures` and `falls_into`. |
| `rest_failures(&device, &[RestProblem], &[Surface], &[Vec<Rest>], &RestOptions)` | Why each problem without a rest has none: a `RestFailure` (`Blocked`, `Tips`, `Unheld`, `Unsupported`), with the point in its way (`Blocking`) where it is not held or tips; `None` where it has rests. Pass the options the rests were found with ([guide](placing.md#why-no-rest)). |
| `falls_into(&device, &[RestProblem], &[Surface], &RestOptions)` | Where to let a held object go over a measured container for it to fall in and stay: per problem, each `Fall` (the `pose` to let go at, how far it `drop`s, how deep under the rim it lands, `sunk`, and its `cost`), at every turn ([guide](placing.md#letting-go-into-a-container)). |

## Timing

Guide: [planning.md](planning.md#executable-trajectories).

| | |
|---|---|
| `Trajectory::try_new(&robot, control_points, speed_scale)` | A rest-to-rest trajectory within the robot's velocity, acceleration and jerk limits. |
| `time(&device, &paths, speed_scale, &hand_speeds)` | Many paths timed in one call on the device's threads, each as `Trajectory::try_new` times it, or under its own caps (`&[]` for none): its trajectory or its error, in order. |
| `Trajectory::with_hand_speed(&robot, control_points, &[HandSpeed])` | The same, with hands under caps on their speed and on how fast they turn, along the whole path or only near a point (`HandSpeed::near`, a `Near`). |
| `Trajectory::moving(&robot, control_points, &StartMotion)` | A trajectory that starts in motion, for a path planned with `start_motion`. |
| `duration()`, `at(t, &mut JointState)`, `sample(hz)`, `start()`, `end()` | Its length in seconds, the state at a time without allocating, samples at a fixed rate (`JointTrajectory`), and the configurations it starts at and comes to rest at. |
| `stop(&robot, t)` | A controlled stop along the path from time `t`. |
| `check(&robot)`, `check_from(&robot, &start)` | Verifying a trajectory before running it. |

## Live motion

Guide: [planning.md](planning.md#live-motion).

| | |
|---|---|
| `Follower::try_new(&robot, world, trajectory, now, FollowerOptions)` | Follows a trajectory from time `now`, on the caller's clock. |
| `follower.tick(&device, &worlds, now, measured)` | The state to command at `now`, given the measured state if known, and what the follower did (`Tick`, `FollowStatus`): following, replanned, re-anchored on a measured state, resumed on a detour, stopping, colliding, arrived, or blocked at rest. |
| `follower.trajectory()`, `began()` | The trajectory it follows now, and when it began on the caller's clock. |
| `FollowerOptions` | A replan's time `budget` and the `lead` it has to reach the robot, the spacing of the checks along the rest of the motion (`check_every`), the `plan` options, the `constraints` replans keep, what a hand holds (`holding`), how far a measured state may be from the trajectory (`tolerance`), whether it replans at all (`replan`), and a window to replan a way joined into a move in (`detour`, a `Detour { from, until, then }`). |
| `servo(&device, &worlds, world, &state, target, &ServoOptions)` | One control step of a hand toward a pose (`ServoStep`): the next state, the `ServoStatus` (normal, slowing, halted with its clearance while braking, or arrived) and the error. |
| `ServoOptions` | The hand, the period `dt`, the `gain`, the `linear` and `angular` caps, the `slow_within` and `halt_within` clearances, and the `damping`. |

## Training data

Guide: [training-data.md](training-data.md).

| | |
|---|---|
| `datagen::demonstrations(&device, &worlds, &[IkProblem], &DemoOptions)` | Reaches to each goal, and recoveries from perturbed states along them. |
| `datagen::pick_and_place(&device, &[World], &[PickPlace], &PickPlaceOptions)` | Top-down pick-and-place episodes. |
| `datagen::handover(&device, &[World], &[datagen::Handover], &PickPlaceOptions)` | One hand picks an object up and passes it to another, which puts it down. |
| `datagen::recovery_problems` | Planning problems that start from perturbed states along planned paths. |
| `datagen::reset_states` | Collision-free states sampled from demonstrations, for a reinforcement learner's reset bank. |
| `Demonstration` | One episode: its trajectory, task, goal, grippers, carried object, and `Origin` (nominal, or a recovery from another episode). |
| `npy::export(root, &robot, &worlds, &demos, &npy::ExportOptions)` | Episodes as NumPy arrays. |
| `lerobot::export(root, &robot, &worlds, &demos, &lerobot::ExportOptions)` | Episodes as a LeRobot v3.0 dataset (feature `lerobot`). |

## Shared types

| | |
|---|---|
| `Pose` | A position and rotation: `Pose::new`, `Pose::IDENTITY`, `mul_pose`, `inverse`, and `is_unit`: whether it is finite with a unit rotation, a pose every call that takes one accepts. |
| `Screw` | A line (`point`, unit `direction`) and how far a thing carried about it moves along it a radian (`pitch`); `Screw::about(point, direction)` without one. `carry(pose, angle)` carries a pose about it, and `path(pose, angle, precision)` gives the waypoints a `ToolPath` follows along the arc, as few as keep the straight moves between them within `precision` of it. |
| `Solved` | What `solved()` yields: the problem's index in the batch, the problem, and its answer. |
| `Error`, `error::Result` | Every fallible call's error: bad input, a file that cannot load, the GPU, threads, an unsafe trajectory, or a failed write. |
| `glam` | The math crate of `Pose` (`Vec3`, `Quat`), re-exported so versions match. |
| `rng::Rng` | The deterministic generator behind every `rng_seed`. |
