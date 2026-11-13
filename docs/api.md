# API

Everything a program uses, by task. Names are in `batchplan::*` unless a module is shown (`datagen::`, `npy::`, `lerobot::`, `rrt::`, `shortcut::`, `rng::`). [docs.rs](https://docs.rs/batchplan) has every signature and field; the guide named under each heading explains the behaviour.

Every batched call has the same shape: a `Device`, the `Worlds` uploaded to it, a slice of problems that each name their world by index, and an options struct whose `Default` is a good start. Results keep the problems they answer: `best(problem)` gives one problem's answer, and `solved()` yields each solved problem with its index and answer.

Options and problems are built with struct update syntax (`..Default::default()`, `..PlanProblem::new(..)`). Results, `Trajectory` and the enums that may grow (`Obstacle`, `Occlusion`, `Origin`, `Error`) are `#[non_exhaustive]`: read their fields, and give a `match` on them a wildcard arm.

## Robots

Guide: [robots.md](robots.md).

| | |
|---|---|
| `Robot::load(path, &RobotOptions)` | A robot from URDF, MJCF or OpenUSD (feature `usd`). |
| `Robot::from_urdf(text, &RobotOptions)` | A robot from a URDF's text, as ROS publishes it on `/robot_description`. |
| `Robot::assemble(&[Part], &RobotOptions)` | One robot from several descriptions, such as two arms at a table; each `Part` has a name prefix and a base pose. |
| `RobotOptions` | The base link, the hands (`ee_links`) and their fingers where they are part of the palm (`finger_links`), the default configuration, locked joints, joint limits and their scales, `package://` directories, an SRDF, OpenUSD variants, and the collision spheres, with tighter tolerances for some links (`sphere_tolerances`). |
| `JointLimits` | One joint's position, velocity, acceleration and jerk limits, replacing its description's. |
| `CollisionModel` | Collision spheres and self-collision settings by link: `load` and `save` a file, `to_json` and `from_json` the same text, `assemble` the model of an assembled robot. |
| `SphereOptions`, `SphereGeometry` | How spheres are fitted when no collision model is given: the budget, the tolerance, and collision or visual geometry. |
| `Robot::attach(&AttachedObject)`, `Robot::detach(name)` | The robot holding an object fixed to one of its links, or without it. |
| `Robot::relock(&joints)` | The robot with joints it locked at load held at new values, as a torso that moved ([guide](robots.md#kinematics)). |
| `dof`, `joint_names`, `lower`, `upper`, `continuous`, `default_q`, `max_velocity`, `max_acceleration`, `max_jerk` | The robot's joints and their limits. |
| `hands`, `hand_joints(hand)`, `hand_links(hand)`, `finger_links(hand)`, `tip(hand)`, `ee_pose(q, hand)`, `link_pose(q, link)` | Hands and forward kinematics; `hand_links` names a hand and its fingers, for allowed contacts, `finger_links` its fingers alone, and `tip` how far its spheres reach past its frame. |
| `spheres(q)`, `reach()` | Every collision sphere at a configuration, and a bound on how far they reach from the base in any. |
| `qualify()` | How well the spheres fit the geometry (`Qualification`): per link (`LinkQualification`) the `overhang`, the `uncovered` share of the surface and the `thickness`, and each hand's tip ([guide](robots.md#qualifying-a-robot)). |
| `breach(constraints, q)` | Why a configuration breaks `HandConstraint`s, if it does, tested as `plan` tests its start and goals: to keep only goals that keep them before planning. |
| `MeshModel::try_new(&robot)`, `MeshModel::clearance` | Clearances on the links' meshes instead of the spheres, to check plans against the real geometry: a `Clearance` (`world`, `self_collision`) per configuration. |

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
| `DepthImage`, `Intrinsics` | A depth image with its pinhole intrinsics and camera pose. |
| `DepthImage::robot_mask(&robot, q, padding)` | Where the robot is in the image (`RobotMask`), so grids leave it out. |
| `Occlusion` | What a grid assumes behind observed surfaces: free, solid, or solid except the robot's shadow. |
| `OccupancyMap::try_new`, `integrate`, `occupancy`, `grid` | Depth images fused over time into occupancy, then into a distance grid. |

## Devices

Guide: [devices.md](devices.md).

| | |
|---|---|
| `Device::gpu(&robot)`, `Device::gpu_named(&robot, name)` | A GPU through wgpu: the fastest one found, or the first whose name contains `name`. An error without the default `gpu` feature. |
| `Device::cpu(&robot)`, `Device::cpu_threads(&robot, n)` | CPU threads: one per core, or exactly `n` that the device owns. |
| `Device::upload(&[World])` | The `Worlds` every batched call takes, prepared for this device. `Worlds` derefs to `[World]`. |
| `Device::with_robot(&robot)` | A device for a changed robot, such as one holding an object, on the same GPU or threads; uploaded worlds work on both. |
| `Device::evaluate` | Collision cost, gradient and clearances for a batch of configurations (`Evaluation`), weighted by `CollisionWeights`. |
| `Device::contacts` | What each configuration of a batch comes within a distance of, nearest first (`Contact`): a link and an obstacle, or two links, with the distance between them. |
| `Device::warmup` | Builds the kernels a GPU builds on first use and runs one small call of each kind, before anything is timed. |

## Inverse kinematics

Guide: [planning.md](planning.md#inverse-kinematics).

| | |
|---|---|
| `solve_ik(&device, &worlds, &[IkProblem], &IkOptions)` | Collision-free IK from many seeds per target (`IkResult`). |
| `IkProblem::new(world, target)` | A one-handed robot's hand at `target`. The fields `targets` (a pose or `None` per hand), `seed` (where seed 0 starts), `hold` (joints that stay where the seed has them) and `holding` (an object a hand holds, a `Held`) cover the rest. |
| `IkOptions` | Seeds per problem, iterations, tolerances and collision weights. |
| `IkResult::best(problem)`, `nearest(problem, q)`, `solved()` | The seed with the most clearance, the seed nearest `q`, and every solved problem. |
| `ik::failures(&device, &worlds, &result, &IkOptions)` | Why each unsolved problem has no solution (`IkFailure`): out of reach, or reached only in collision, with its contacts. |

## Planning

Guide: [planning.md](planning.md#planning-a-path).

| | |
|---|---|
| `plan(&device, &worlds, &[PlanProblem], &PlanOptions)` | Collision-free B-spline paths, one search per problem (`PlanResult`). |
| `PlanProblem::new(world, start, goal)` | From `start` at rest to `goal` (a `Goal`): a configuration, or a `Pose` (`Goal::Poses`, one per hand), which `plan` solves IK for from `start`, ending at the nearest collision-free solution ([guide](planning.md#planning-a-path)). The fields `hold` (joints that stay still), `start_motion` (a `StartMotion`, to replan from a moving state), `constraints` (below) and `holding` (an object a hand holds, a `Held`) cover the rest. Problems of one call may each hold their own ([holding objects](robots.md#holding-objects)), as may IK problems, tool paths and sequences. |
| `HandConstraint::new(hand, pose)` | Where a hand may be along the whole path, measured from `pose` ([guide](planning.md#constraints-along-the-path)): set `tilt` (its z axis within a cone), `twist` (its turn about that axis) and `position` (a box), all free by default, and `relative_to` another hand to measure them in its frame. `RrtProblem`, `Pick`, `Place`, `Sequence` and `FollowerOptions` take them too. |
| `PlanOptions` | Control points, optimization seeds and iterations, the search (`rrt`: `RrtOptions`), shortcutting (`shortcut`: `ShortcutOptions`), IK for pose goals (`goal_ik`: `IkOptions`) and a `time_budget`. |
| `PlanResult::best(problem)`, `solved()` | The shortest valid path's control points, and every solved problem. Every seed's control points are in `paths`, empty where a seed has none. |
| `PlanResult::failures` | Per problem, why it has no valid path (`PlanFailure`): the start or the goal collides (with its contacts), the goal's poses are out of reach, no path, no valid spline, or out of time ([guide](planning.md#why-a-problem-failed)). |
| `rrt::connect`, `shortcut::shortcut` | The search and the shortcutting that `plan` runs, on their own, for waypoint paths. |
| `follow(&device, &worlds, &[ToolPath], &ToolPathOptions)` | A hand moved in straight lines through poses, the joints a path holds (`hold`) still ([guide](planning.md#moving-the-hand-along-a-path)): per path, the `fraction` it covers, why it `stopped` short (`ToolPathStop`) and what a collision there is with (`contacts`), and the motion's control points (`ToolPathResult`). |
| `pick(&device, &worlds, &[Pick], &TaskOptions)`, `place(.., &[Place], ..)` | A whole task in one call ([guide](planning.md#picking-up-and-setting-down)): per task, a `TaskPlan` for every candidate that gets through, quickest first (`transit`, `None` for a task without one, `approach` and `depart` trajectories, `motions` for those it has in order, the `candidate`, what is `held` or `placed`, `begin` and `end`), or a `TaskFailure` with each candidate's step, reason and contacts (`CandidateStop`). |
| `Pick::new(world, start, object, grasps)`, `Place::new(world, start, held, poses)` | Hand 0 (a pick) or the holding hand (a place), 10 cm straight moves in and away, no `support` and no `constraints` on the transit; a place's `idle` takes the other arms elsewhere on its transit. `transit: false` leaves the transit out and gives the two straight moves alone, and `transit_world` and `depart_world` run the start and the transit, and the move away, in worlds of their own. |
| `TaskOptions` | IK, planning and tool path options, the straight moves' speed cap (`tool_speed`, `f32::INFINITY` for none), where it holds (`tool_speed_near`: within that distance of the grasp or release; `None`, all along) and how much of each they must make (`min_fraction`). |
| `grasps(&obstacle, &GraspOptions)`, `place_poses(&obstacle, pose, turns)`, `Held::attachment(&robot)` | Grasps of a box, ball, cylinder or capsule for a parallel gripper, keeping the frame and the open fingers (`opening` apart, reaching `tip`) above a `floor` if given, and from above at least `overlap` of the fingers in the object, or why there are none; an object's symmetric turns at one spot; a held object as an `AttachedObject`. |
| `join(&device, &worlds, &[Sequence], &JoinOptions)` | Consecutive motions run without stopping between them ([guide](planning.md#joining-motions)): per sequence, one path for each stretch the robot moves without stopping, and each join's blend radius, `None` where it stops (`JoinResult`). `Sequence::blends` gives each join's radius, 0 to stop, `Sequence::constraints` what every blend keeps, and `Sequence::holding` what a hand carries through it. |

## Timing

Guide: [planning.md](planning.md#executable-trajectories).

| | |
|---|---|
| `Trajectory::try_new(&robot, control_points, speed_scale)` | A rest-to-rest trajectory within the robot's velocity, acceleration and jerk limits. |
| `time(&device, &paths, speed_scale, &hand_speeds)` | Many paths timed in one call on the device's threads, each as `Trajectory::try_new` times it, or under its own caps (`&[]` for none): its trajectory or its error, in order. |
| `Trajectory::with_hand_speed(&robot, control_points, &[HandSpeed])` | The same, with hands under caps on their speed and on how fast they turn, along the whole path or only near a point (`HandSpeed::near`). |
| `Trajectory::moving(&robot, control_points, &StartMotion)` | A trajectory that starts in motion, for a path planned with `start_motion`. |
| `duration()`, `at(t, &mut JointState)`, `sample(hz)`, `start()`, `end()` | Its length in seconds, the state at a time without allocating, samples at a fixed rate (`JointTrajectory`), and the configurations it starts at and comes to rest at. |
| `stop(&robot, t)` | A controlled stop along the path from time `t`. |
| `check(&robot)`, `check_from(&robot, &start)` | Verifying a trajectory before running it. |

## Live motion

Guide: [planning.md](planning.md#live-motion).

| | |
|---|---|
| `Follower::try_new(&robot, world, trajectory, now, FollowerOptions)` | Follows a trajectory from time `now`, on the caller's clock. |
| `follower.tick(&device, &worlds, now, measured)` | The state to command at `now`, given the state the robot was measured in if known, and what the follower did (`Tick`, `FollowStatus`): following, replanned to a `trajectory` that takes over `at` a time, re-anchored on a measured state farther than its tolerance, resumed on a detour's `then` `at` a time, stopping, colliding `at` a time with no stop short of it (both naming what is in the way, `contacts`), arrived, or blocked at rest. |
| `follower.trajectory()`, `began()` | The trajectory it follows now, and when it began on the caller's clock. |
| `FollowerOptions` | A replan's time `budget` and the `lead` it then has to reach the robot before it takes over, the spacing of the checks along the rest of the motion (`check_every`), the `plan` options, the `constraints` replans keep, how far a measured state may be from the trajectory (`tolerance`), whether it replans at all (`replan`; off, it stops on a tool path's path and holds the robot where it lags), and, for a way joined into a move in, where it may (`detour`: a `Detour { from, until, then }`, replanning within that window to the start of `then` and taking `then` up from rest there). |
| `servo(&device, &worlds, world, &state, target, &ServoOptions)` | One control step of a hand toward a pose (`ServoStep`): the next state, the `ServoStatus` (normal, slowing, halted with its clearance while braking, or arrived) and the error. |
| `ServoOptions` | The hand, the period `dt`, the `gain`, the `linear` and `angular` caps, the `slow_within` and `halt_within` clearances, and the `damping`. |

## Training data

Guide: [training-data.md](training-data.md).

| | |
|---|---|
| `datagen::demonstrations(&device, &worlds, &[IkProblem], &DemoOptions)` | Reaches to each goal, and recoveries from perturbed states along them. |
| `datagen::pick_and_place(&device, &[World], &[PickPlace], &PickPlaceOptions)` | Top-down pick-and-place episodes. |
| `datagen::handover(&device, &[World], &[Handover], &PickPlaceOptions)` | One hand picks an object up and passes it to another, which puts it down. |
| `datagen::recovery_problems` | Planning problems that start from perturbed states along planned paths. |
| `datagen::reset_states` | Collision-free states sampled from demonstrations, for a reinforcement learner's reset bank. |
| `Demonstration` | One episode: its trajectory, task, goal, grippers, carried object, and `Origin` (nominal, or a recovery from another episode). |
| `npy::export(root, &robot, &worlds, &demos, &npy::ExportOptions)` | Episodes as NumPy arrays. |
| `lerobot::export(root, &robot, &worlds, &demos, &lerobot::ExportOptions)` | Episodes as a LeRobot v3.0 dataset (feature `lerobot`). |

## Shared types

| | |
|---|---|
| `Pose` | A position and rotation: `Pose::new`, `Pose::IDENTITY`, `mul_pose`, `inverse`. |
| `Solved` | What `solved()` yields: the problem's index in the batch, the problem, and its answer. |
| `Error`, `error::Result` | Every fallible call's error: bad input, a file that cannot load, the GPU, threads, an unsafe trajectory, or a failed write. |
| `glam` | The math crate of `Pose` (`Vec3`, `Quat`), re-exported so versions match. |
| `rng::Rng` | The deterministic generator behind every `rng_seed`. |
