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
| `RobotOptions` | The base link, the hands (`ee_links`), the default configuration, locked joints, joint limits and their scales, `package://` directories, an SRDF, OpenUSD variants, and the collision spheres. |
| `JointLimits` | One joint's position, velocity, acceleration and jerk limits, replacing its description's. |
| `CollisionModel` | Collision spheres and self-collision settings by link: `load` and `save` a file, `assemble` the model of an assembled robot. |
| `SphereOptions`, `SphereGeometry` | How spheres are fitted when no collision model is given: the budget, the tolerance, and collision or visual geometry. |
| `Robot::attach(&AttachedObject)`, `Robot::detach(name)` | The robot holding an object fixed to one of its links, or without it. |
| `dof`, `joint_names`, `lower`, `upper`, `continuous`, `default_q`, `max_velocity`, `max_acceleration`, `max_jerk` | The robot's joints and their limits. |
| `hands`, `hand_joints(hand)`, `hand_links(hand)`, `finger_links(hand)`, `ee_pose(q, hand)`, `link_pose(q, link)` | Hands and forward kinematics; `hand_links` names a hand and its fingers, for allowed contacts, and `finger_links` its fingers alone. |
| `MeshModel::try_new(&robot)`, `MeshModel::clearance` | Clearances on the links' meshes instead of the spheres, to check plans against the real geometry: a `Clearance` (`world`, `self_collision`) per configuration. |

## Worlds and perception

Guide: [perception.md](perception.md).

| | |
|---|---|
| `World::new(obstacles)` | One scene: a list of obstacles. |
| `World::remove(obstacle)` | Takes an obstacle out, as a hand picking it up does, and renumbers the touches of those after it. |
| `World::touch`, `Touch::new(obstacle, links)` | Links allowed to touch an obstacle, such as the hand that grasps it ([allowed contacts](perception.md#allowed-contacts)). |
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

## Inverse kinematics

Guide: [planning.md](planning.md#inverse-kinematics).

| | |
|---|---|
| `solve_ik(&device, &worlds, &[IkProblem], &IkOptions)` | Collision-free IK from many seeds per target (`IkResult`). |
| `IkProblem::new(world, target)` | A one-handed robot's hand at `target`. The fields `targets` (a pose or `None` per hand) and `seed` (where seed 0 starts) cover the rest. |
| `IkOptions` | Seeds per problem, iterations, tolerances and collision weights. |
| `IkResult::best(problem)`, `nearest(problem, q)`, `solved()` | The seed with the most clearance, the seed nearest `q`, and every solved problem. |

## Planning

Guide: [planning.md](planning.md#planning-a-path).

| | |
|---|---|
| `plan(&device, &worlds, &[PlanProblem], &PlanOptions)` | Collision-free B-spline paths, one search per problem (`PlanResult`). |
| `PlanProblem::new(world, start, goal)` | From `start` at rest to `goal`. The fields `hold` (joints that stay still) and `start_motion` (a `StartMotion`, to replan from a moving state) cover the rest. |
| `PlanOptions` | Control points, optimization seeds and iterations, the search (`rrt`: `RrtOptions`), shortcutting (`shortcut`: `ShortcutOptions`) and a `time_budget`. |
| `PlanResult::best(problem)`, `solved()` | The shortest valid path's control points, and every solved problem. Every seed's path is in `paths` (`JointPaths`). |
| `rrt::connect`, `shortcut::shortcut` | The search and the shortcutting that `plan` runs, on their own, for waypoint paths. |
| `follow(&device, &worlds, &[ToolPath], &ToolPathOptions)` | A hand moved in straight lines through poses ([guide](planning.md#moving-the-hand-along-a-path)): per path, the `fraction` it covers, why it `stopped` short (`ToolPathStop`), and the motion's control points (`ToolPathResult`). |
| `pick(&device, &worlds, &[Pick], &TaskOptions)`, `place(.., &[Place], ..)` | A whole task in one call ([guide](planning.md#picking-up-and-setting-down)): per task, a `TaskPlan` for every candidate that gets through, quickest first (`transit`, `approach` and `depart` trajectories, the `candidate`, what is `held` or `placed`, `end`), or a `TaskFailure` with each candidate's step and reason (`CandidateStop`). |
| `Pick::new(world, start, object, grasps)`, `Place::new(world, start, held, poses)` | Hand 0 (a pick) or the holding hand (a place), 10 cm straight moves in and away, no `support`. |
| `TaskOptions` | IK, planning and tool path options, the straight moves' speed cap (`tool_speed`) and how much of each they must make (`min_fraction`). |
| `grasps(&obstacle, &GraspOptions)`, `place_poses(&obstacle, pose, turns)`, `Held::attachment(&robot)` | Grasps of a box, ball, cylinder or capsule for a parallel gripper; an object's symmetric turns at one spot; a held object as an `AttachedObject`. |

## Timing

Guide: [planning.md](planning.md#executable-trajectories).

| | |
|---|---|
| `Trajectory::try_new(&robot, control_points, speed_scale)` | A rest-to-rest trajectory within the robot's velocity, acceleration and jerk limits. |
| `Trajectory::with_hand_speed(&robot, control_points, &HandSpeed)` | The same, with a hand under caps on its speed and on how fast it turns. |
| `Trajectory::moving(&robot, control_points, &StartMotion)` | A trajectory that starts in motion, for a path planned with `start_motion`. |
| `duration()`, `at(t, &mut JointState)`, `sample(hz)` | Its length in seconds, the state at a time without allocating, and samples at a fixed rate (`JointTrajectory`). |
| `stop(&robot, t)` | A controlled stop along the path from time `t`. |
| `check(&robot)`, `check_from(&robot, &start)` | Verifying a trajectory before running it. |

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
