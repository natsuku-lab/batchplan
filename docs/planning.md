# Planning

Planning in batchplan is three steps. IK and planning are batched calls on a [`Device`](devices.md); timing runs on each path:

1. **IK** (`solve_ik`) turns target poses into joint configurations.
2. **Planning** (`plan`) finds a collision-free path from a start to a goal configuration (the straight line, bent if an obstacle only grazes it, else RRT-Connect), then shortens and smooths it into a B-spline. Trajectory optimization can polish it further.
3. **Timing** (`Trajectory::try_new`) turns the path into a trajectory a controller can run, within the robot's velocity, acceleration and jerk limits.

Every call takes many problems, each naming its world, so one call can cover thousands of scenes.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::{
    Device, Error, IkOptions, IkProblem, PlanOptions, PlanProblem, Pose, Robot, RobotOptions,
    Trajectory, World,
};

fn main() -> Result<(), Error> {
    let robot = Robot::load(
        "assets/ur5e/ur_description/urdf/ur5e.urdf",
        &RobotOptions::default(),
    )?;
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;

    // Several targets in one call; each problem names its world.
    let down = Quat::from_rotation_x(std::f32::consts::PI);
    let targets = [Vec3::new(0.4, 0.2, 0.3), Vec3::new(0.3, -0.3, 0.4)];
    let goals: Vec<IkProblem> = targets
        .iter()
        .map(|&p| IkProblem::new(0, Pose::new(p, down)))
        .collect();
    let ik = batchplan::solve_ik(&device, &worlds, &goals, &IkOptions::default())?;

    // `solved()` yields each problem that has a solution, with its world, so they cannot be mixed up.
    let problems: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution))
        .collect();
    let result = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default())?;
    for s in result.solved() {
        let trajectory = Trajectory::try_new(&robot, s.solution, 1.0)?;
        println!("problem {}: {:.2} s", s.index, trajectory.duration());
    }
    Ok(())
}
```

## Inverse kinematics

`solve_ik` runs damped least squares from many seeds per target (32 by default), with the collision gradient projected into the Jacobian's null space so seeds slide away from obstacles without leaving the target. A seed succeeds when it is within `position_tolerance` (1 mm) and `rotation_tolerance` (0.01 rad) of the target and collision-free.

- `IkProblem::new(world, target)` targets a one-handed robot's hand from the default pose; the fields below cover the rest.
- `ik.best(problem)` is the successful seed with the most obstacle clearance; `ik.solved()` yields every solved problem with that configuration.
- `IkProblem::seed` starts seed 0 from a configuration you give (the robot's current state, say) instead of the default pose, and `ik.nearest(problem, &q)` returns the successful seed nearest `q`. Together they keep short moves on the same IK branch, as `pick` and `place` do at the grasp and where the hand lets go.
- `IkProblem::targets` gives each of the robot's hands (`RobotOptions::ee_links`) a target pose, or `None`. Each step solves every targeted hand's damped least squares and pushes the collision gradient out of their Jacobians' row spaces, which is exact for hands on separate arms. A hand without a target holds its arm: in every seed, the joints only that hand moves stay where the seed has them. So one arm can reach while the other waits, or both can reach at once and keep clear of each other.

## Planning a path

`PlanProblem::new(world, start, goal)` plans from rest with every joint free to move; set the other fields on top of it (`PlanProblem { hold, ..PlanProblem::new(world, start, goal) }`).

A plan into a grasp ends with the fingers around the object: let the hand touch it ([allowed contacts](perception.md#allowed-contacts)), and the rest of the arm still keeps clear of it.

`PlanProblem::hold` lists joints that stay where `start` has them, such as the other arm of a two-armed robot while one arm moves (`robot.hand_joints(hand)` lists an arm's); `goal` must have the same values there. The search samples and bends only the other joints, no goal equivalent turns a held joint, and optimized paths are put back on the held values before they are checked, so every control point keeps them.

`plan` runs every problem as its own search, whole on one CPU thread, with the searches in parallel on either device, as VAMP does (Thomason, Kingston and Kavraki, 2024):

1. **The straight line** to each of the goal's equivalents (below) is tried first.
2. **A bent line.** If every straight line collides, the one to the nearest goal is bent: the stretch from where it first collides to where it last does moves 0.05 rad aside, across the line along each joint's axis in turn, both ways, then 0.1, 0.2 and 0.4 rad. The first bend whose edges are free is the path. So a line that an obstacle only grazes keeps its route instead of taking the detour a search would find. Bending costs no measurable time.
3. **RRT-Connect** grows a tree from the start and one from the goals, in turn, toward Halton samples. The first samples lie in a tube around the straight line to the nearest goal: 0.4 rad across it (along each axis across it) and past its ends, doubling every 64 samples until it spans the joint ranges. So the route found stays near the straight line where such a route exists, as cuRobo's roadmap does by sampling an ellipsoid around it; samples from the whole space find detours that shortcuts cannot undo. Extensions step at most `rrt.range` (1 rad), and dynamic domains skip samples far from any node that recently failed. A search gives up after up to 500 samples from the joint ranges keeping its clearance and `rrt.max_iterations` (10,000) more without it (see Clearance, below); the tube's samples come on top.
4. **Shortcutting and smoothing** run for up to `shortcut.passes` passes (3). Each pass replaces waypoints with the longest free edge, splits every edge, and moves each waypoint a quarter of the way toward its neighbours where both new edges stay free. With `shortcut.straighten` (on), after the first pass's shortcuts each joint in turn is straightened where that shortens the path by at least `shortcut.min_change`: it changes at one rate along the whole path while the other joints keep theirs, as in HPP's partial shortcut. That removes a joint's swing out and back (a wrist's, say), which shortcuts between whole configurations keep.
5. **A B-spline** follows the path, with `control_points` control points (24). The first and last three are pinned to the start and goal, so it starts and ends at rest. The others first spread along the smoothed path by arc length, so the curve is smooth and cuts the path's corners. It is checked at samples no farther apart than `shortcut.resolution`, as edges are, except along a straight line, which the search or the shortcuts already checked keeping at least the spline's clearance. Where it fails, the waypoint nearest the failure is pinned: three control points sit on it, so the curve passes through it, stopping there. After up to six pins, the fit starts over on the shortcut path; with every waypoint pinned, the curve runs exactly along the checked path. The first spline that validates is the result, once its stops are eased (as in OpenRAVE's smoother): each stop's three control points spread into the widest blend around the corner that keeps the curve valid. Blends are the smaller of the free control points' spacing and half the shorter edge at the corner, then a quarter and a sixteenth of that; a corner no blend clears keeps its stop. If no spline validates, trajectory optimization pushes the first one off the obstacles a step at a time, for up to ten steps. The first step whose spline keeps the 2 mm margin is the result; the tenth's needs only to be collision-free.

**Checking edges.** An edge is collision-checked at evenly spaced states, `rrt.resolution` (1/32 rad) apart, eight at once in SIMD lanes: spread over the edge, then stepping back, so a collision anywhere is found early. [Architecture](architecture.md#searches) describes the cheaper tests that skip most of the work.

**Clearance.** A path that only just clears the spheres can graze the meshes where the spheres leave them uncovered. So RRT-Connect keeps `rrt.clearance` (1.5 cm) from obstacles, and shortcuts keep `shortcut.clearance` (1 cm). Both shrink to zero within `clearance_ramp` (0.5 rad) of the start and the goals, since paths often start or end in contact, as at a grasp. The search's wider clearance finds routes through open space, which clear the meshes more often; the shortcuts' narrower one lets the path pull tight. If RRT-Connect finds no path within 500 samples, it continues from the same trees without its clearance for up to `rrt.max_iterations` samples more, so narrow passages stay passable. Splines keep the shortcuts' clearance first, then a 2 mm margin (or the shortcuts' clearance, if smaller), so the curve cannot graze an obstacle between the samples it is checked at.

**Validation.** Each spline candidate is sampled densely (`validate_substeps` per span) and checked for world and self-collision and joint limits. `result.best(problem)` is the shortest valid path, and `result.solved()` yields every solved problem with it.

### Trajectory optimization (optional)

`PlanOptions::seeds` (0 by default) adds trajectory optimization of that many seeds per problem, batched on the device:
- **Seeds.** The first starts from the searched spline, the second from the straight line to the goal, as cuRobo's first attempt does. Further seeds run straight to the goal's equivalents, or bend through random via points. Problems searched to within 2% of the straight line are not optimized, since no seed could be much shorter; their seeds stay invalid.
- **Cost.** Length and smoothness, as squared differences of the control points (weighted `velocity_weight`, 10) and squared second differences (`acceleration_weight`, 5), plus a collision cost for the world and for self-collision, sampled along the curve (`samples_per_span` points per span) and active within `collision.margin` (1.2 cm) of an obstacle and `collision.self_margin` (1 cm) between links.
- **Optimizer.** L-BFGS (`iterations`, 100, remembering `history` steps). Each iteration prices four step sizes along the L-BFGS direction in one batched pass and keeps the cheapest.

Repairing a stuck spline (above) uses the same optimizer with its own settings, whatever these options say: ten steps, smoothness weighted over length (`acceleration_weight` 50, `velocity_weight` 2) and a 2 cm margin. These repair more splines than the length-favouring weights do.

Optimized seeds are validated like the searched path, and `best` picks the shortest valid one of all. On the benchmark's box-only problems, two seeds give 3.137 rad paths at the median, against 3.243 without and cuRobo's 3.144 ([benchmarks](benchmarks.md#results)). Optimization costs milliseconds per problem (15.1 ms on 14 CPU threads; 746 problems/s in batches on an M4 Pro's GPU), against a fraction of a millisecond for the search.

### Goals a turn away

Continuous joints are unbounded, and joints whose range spans more than a turn (the UR5e's ±2π) reach the same pose at angles a turn apart. Goals are reached at whichever equivalent is nearest or free: the search takes them all as its goal set, and continuous joints step and measure the short way round. On the UR5e, a goal whose shoulder has turned past the table below is reached the long way round when the short way is blocked.

### Time budgets and determinism

`PlanOptions::time_budget` stops every search and the optimization once it passes, and returns the valid paths found so far. Without a budget nothing checks the clock, and two runs give bit-identical results, on either device.

## Moving the hand along a path

`follow(&device, &worlds, &[ToolPath], &ToolPathOptions)` moves a hand through poses: in a straight line from each to the next, turning by the shortest rotation, as MoveIt's Cartesian paths do. Each `ToolPath` names its world, its start configuration, the hand, and the `waypoints` after the start; only the joints that move that hand move. It is what an approach, a lift or a pour asks for, where a planned path would take any route.

```rust,no_run
use batchplan::glam::Vec3;
use batchplan::{
    Device, Error, HandSpeed, Pose, Robot, RobotOptions, ToolPath, ToolPathOptions, Trajectory,
    World,
};

fn main() -> Result<(), Error> {
    let robot = Robot::load("assets/franka/franka_panda.urdf", &RobotOptions::default())?;
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;
    let start = robot.default_q().to_vec();
    let hand = robot.ee_pose(&start, 0);
    // Lower the hand 10 cm straight down, keeping its rotation.
    let down = Pose::new(hand.position - Vec3::new(0.0, 0.0, 0.1), hand.rotation);
    let path = ToolPath {
        world: 0,
        start,
        hand: 0,
        waypoints: vec![down],
    };
    let result = batchplan::follow(&device, &worlds, &[path], &ToolPathOptions::default())?;
    if let Some(why) = result.stopped[0] {
        println!("{:.0}% of the way: {why:?}", 100.0 * result.fraction[0]);
    }
    if !result.paths[0].is_empty() {
        // At 5 cm/s, turning as fast as the joints allow.
        let speed = HandSpeed {
            hand: 0,
            linear: 0.05,
            angular: f32::INFINITY,
        };
        let descent = Trajectory::with_hand_speed(&robot, &result.paths[0], &speed)?;
        println!("lowered in {:.2} s", descent.duration());
    }
    Ok(())
}
```

- **Steps.** The hand moves at most `step` (5 mm) or `rotation_step` (0.05 rad) between configurations, each solved by damped least squares from the one before, in at most `iterations` (30) and to a tenth of the precision. Where the joints' straight motion between two configurations would take the hand more than `precision` (1 mm) or `rotation_precision` (0.01 rad) off the path at its middle, the step is halved, up to 8 times.
- **The motion.** A cubic B-spline passes through every configuration and rests at both ends. Where it would stray from the path by more than the precision, as at a sharp turn, or pass a joint limit, as where a joint arrives at its limit and stays, it stops at the configuration there instead. It is checked for collision at the planner's resolution, honouring [allowed contacts](perception.md#allowed-contacts), so an approach can end with the fingers around what they grasp.
- **How far it got.** A motion that cannot finish goes as far as it can. `fraction` says how much of the path it covers, by its steps, and `stopped` why it ended there: `Unreachable` (beyond reach or joint limits, or at a singularity IK cannot step through), `Collision`, or `Jump` (configurations either side, but no motion between them keeps the hand on the path, as when IK leaps to another branch). Near a singularity, IK given more `iterations` may instead carry the arm through, swinging the wrist on the way while the hand keeps to the path. Each names the pose it stopped before. `paths` holds the motion that does run, ready for timing.
- **Timing.** `Trajectory::with_hand_speed` times the motion as fast as the joints' limits allow with the hand under caps on its speed and on how fast it turns ([below](#executable-trajectories)); `Trajectory::try_new` times it by the joints alone.
- **Speed.** Paths are searches, run in parallel on the device's threads like plans. One 10 cm move takes 0.042 ms on one of the Ryzen's threads and 0.048 ms on the M4 Pro's; 64 take 0.27 ms on 32 threads and 0.42 ms on 14 ([Benchmarks](benchmarks.md#tool-paths)).

## Picking up and setting down

`pick(&device, &worlds, &[Pick], &TaskOptions)` and `place(&device, &worlds, &[Place], &TaskOptions)` each do a whole task in one call: three motions from rest to rest. The `transit` is a planned path from the start to where the hand begins to move in, the `approach` a straight move in (to the grasp, or down to where the object goes), and the `depart` a straight move away (lifting the object, or backing off from it). Close or open the gripper between the approach and the departure. Every motion is checked with the fingers where the start has them, so open them first.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::{
    CollisionModel, Device, Error, GraspOptions, Obstacle, Pick, Place, Pose, Robot, RobotOptions,
    TaskOptions, World,
};

fn main() -> Result<(), Error> {
    // IK targets the grasp point between the open fingers, as `grasps` expects.
    let fingers = [
        ("panda_finger_joint1".to_string(), 0.04),
        ("panda_finger_joint2".to_string(), 0.04),
    ];
    let options = RobotOptions {
        ee_links: vec!["ee_link".into()],
        lock_joints: fingers.into_iter().collect(),
        collision_model: Some(CollisionModel::load("assets/franka/panda_collision.json")?),
        ..Default::default()
    };
    let robot = Robot::load("assets/franka/franka_panda.urdf", &options)?;
    let device = Device::cpu(&robot)?;
    let table = Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let block = Obstacle::cuboid(Vec3::new(0.45, 0.2, 0.03), Vec3::new(0.02, 0.03, 0.03));
    let mut world = World::new(vec![table, block]);
    let o = TaskOptions::default();
    // By any grasp that fits the fingers, lifting the block off the table (obstacle 0).
    let grasps = batchplan::grasps(&world.obstacles[1].placed(Pose::IDENTITY), &GraspOptions::default())?;
    let pick = Pick {
        support: vec![0],
        ..Pick::new(0, robot.default_q(), 1, grasps)
    };
    let worlds = device.upload(std::slice::from_ref(&world))?;
    let picked = match batchplan::pick(&device, &worlds, &[pick], &o)?.outcomes.remove(0) {
        Ok(plans) => plans[0].clone(),
        Err(why) => {
            println!("no pick: {why}");
            return Ok(());
        }
    };
    // Run the transit and the approach, close the gripper, run the departure. Then set the block
    // down 40 cm to the right, at any of its turns there, on the table it has left.
    let block = world.remove(1)?;
    let worlds = device.upload(&[world])?;
    let to = Pose::new(Vec3::new(0.45, -0.2, 0.03), Quat::from_rotation_z(0.5));
    let held = picked.held.clone().expect("a pick holds what it picked up");
    let place = Place {
        support: vec![0],
        ..Place::new(0, picked.end(), held, batchplan::place_poses(&block, to, 8))
    };
    match batchplan::place(&device, &worlds, &[place], &o)?.outcomes.remove(0) {
        Ok(plans) => println!("{:.1} s up, {:.1} s down", picked.duration(), plans[0].duration()),
        Err(why) => println!("no place: {why}"),
    }
    Ok(())
}
```

- **Candidates.** Each of a task's grasps or poses is a candidate, and every task's candidates run together, step by step, one batch per step:
  1. The start, checked for collision.
  2. IK at the grasp, or where the hand lets go, taking the solution nearest the start (for a place with `idle`, the start with the other arms where they wait).
  3. The straight move in, followed backward from there so it ends exactly there: from `approach` back along the hand's z axis to a grasp, or along `lower` to where a place lets go.
  4. The straight move away: `lift` in the world's frame, or `retreat` back along the hand's z axis.
  5. A planned path from the start to where the move in begins.

  Every candidate that passes every step comes back, quickest first (`TaskPlan::candidate` says which). So a place batched over each surviving grasp's `end` and `held` finds the quickest pick and place together, in two calls. The straight moves keep the hand under `TaskOptions::tool_speed` (10 cm/s), and must make `min_fraction` of their length (all of it): less lets an obstacle shorten a move in or away rather than rule the candidate out, while a move stopped short by anything else (out of reach, a jump) still rules it out.
- **What may touch.** At the grasp and moving in, the fingers (`Robot::finger_links`, or the whole hand where it has no links with collision spheres below its palm) may touch the object; the palm may not, so a grasp set too deep fails. The held object may touch what it rests on as it leaves (`Pick::support`) and what it is set on as it arrives (`Place::support`). Backing off, the fingers may touch the object where it now stands. Everything else collides.
- **What the hand holds.** A pick's plan holds a `Held` (`TaskPlan::held`): the object in its own frame and the grasp. `place` takes it and attaches the object itself, to the device without it; `Held::attachment(&robot)` gives the `AttachedObject` to plan with it otherwise ([`Robot::attach`](robots.md)). A place's plan gives the object where it now stands (`TaskPlan::placed`), to add to the world, and `TaskPlan::end` is where the next task starts. `World::remove` takes the object out of its world for the place, renumbering the touches.
- **Other arms.** A pick or place moves its hand's joints alone; the joints only other hands move wait where the start has them. A place's transit can take them elsewhere at the same time (`Place::idle`), as another arm going home while this one carries the object, and they wait there through the moves in and away: IK at the release starts from there.
- **Proposals.** `grasps(&obstacle, &GraspOptions)` proposes grasps for a box, ball, cylinder or capsule, in its frame: the hand facing each side, closing across what fits between its fingers (`opening`, 8 cm), `depth` (2 cm) in from the surface. They suit a parallel gripper whose IK frame lies between its fingertips, its z axis pointing out of the hand and its y axis along the closing direction, as the Panda's `ee_link`. `place_poses(&obstacle, pose, turns)` gives an object's symmetric turns at one spot: 2 for a box (4 with a square footprint), `turns` for a round object.
- **Why it failed.** When no candidate gets through, the task's `TaskFailure` holds each candidate's step and reason (`CandidateStop`: a pose out of reach, a collision or a jump, or no path), and says it in a sentence: "none of 24 candidates got through: 20 at the grasp or release (1 out of reach, 19 colliding), 4 moving away (4 colliding)". IK that finds nothing is solved once more without its push away from obstacles, which tells a pose out of reach from one that collides.
- **Speed.** On the Ryzen AI Max+ 395 (32 threads), picking a box by its 24 grasps takes 6.5 ms on the CPU device, and 64 such picks at once 333 ms; on the Radeon 8060S, 15 ms and 114 ms. Setting a box down at either of its turns takes 2.4 ms on the CPU, and 16 at once 11.6 ms. [Benchmarks](benchmarks.md#picking-and-placing) has both machines and every batch size.

## Executable trajectories

`Trajectory::try_new(&robot, path, speed_scale)` keeps the path and adds a time map `s = σ(t)` (itself a uniform cubic B-spline), choosing the faster of two timings:

- **Time-optimal** (TOPP-RA, Pham and Pham 2018): the fastest timing of the path under joint velocity and acceleration limits, plus a speed cap from the jerk limit, smoothed into a B-spline time map. Where the smoothed map overshoots a limit, the caps are lowered there and the profile solved again. On planned Panda paths it is 12% faster than uniform timing on average and never slower, at about 0.5 ms per trajectory.
- **Uniform**: `σ(t) = t / h`, with the smallest `h` that keeps every joint within its limits. Velocity, acceleration and jerk are then B-splines over the control points' differences, so the bound holds exactly along the whole curve.

Either is slowed by `speed_scale` (1.0 is as fast as the limits allow). Limits come from `RobotOptions::joint_limits` for joints it names, then from the robot description when it has them (URDF 1.2 `acceleration` and `jerk`), otherwise from `RobotOptions` (5 rad/s² and 50 rad/s³ by default). `Trajectory::with_hand_speed(&robot, path, &HandSpeed)` keeps a hand under caps too: `linear` on its speed (m/s) and `angular` on how fast it turns (rad/s), where `f32::INFINITY` leaves one uncapped. Both timings keep them, and `check` checks the joints' limits only.

- **For control loops**: `trajectory.at(t, &mut state)` writes position, velocity and acceleration without allocating.
- **For datasets**: `trajectory.sample(hz)` returns fixed-rate samples.
- **For safety layers**: a `Trajectory` is plain serializable data. `trajectory.check(&robot)` refuses one that is non-finite, does not start and end at rest, leaves a joint range, runs backward, or exceeds a velocity, acceleration or jerk limit.
- **For stopping**: `trajectory.stop(&robot, t)` is a controlled stop: the same path from time `t`, braking to rest as soon as an even deceleration within the limits allows. It stays on the checked path, so it needs no new collision check; verify it with `check_from` and the state at `t`. On planned Panda reaches it takes under 1.2 ms to compute, rests in about half the time left, and errors only where the path ends sooner than any stop would (late in short, fast motions).

### Starting from a moving state

To replan mid-motion, give `PlanProblem::start_motion` the robot's current velocity and acceleration. The spline's first control points then continue that motion, and `Trajectory::moving(&robot, path, &start_motion)` times it from exactly that state, as fast as the limits allow.
- `plan` only returns splines that can be timed this way.
- The spline's first spans last as long as the robot needs to change its velocity (about |v| / a). Where it still cannot follow the motion, longer spans are tried, and then trajectory optimization smooths the spline until it can.
- When reaches are replanned to the same goal from partway through, plans succeed every time with every joint below 1.5 rad/s, and three times in four with one faster, near the Panda's full speed. In the rest, momentum carries a joint out of its range or into the scene, or no timing keeps the limits.
- A path that cannot brake in time is refused with `Error::Unsafe`.
- `trajectory.check_from(&robot, &state)` verifies that a trajectory starts at a given state: each position to 1e-4 of (1 + |q|) rad, each velocity and acceleration to 1e-4 of its value plus the joint's limit.

```rust,no_run
use batchplan::{
    Device, Error, JointState, PlanOptions, PlanProblem, Robot, RobotOptions, StartMotion,
    Trajectory, World,
};

fn main() -> Result<(), Error> {
    let robot = Robot::load(
        "assets/ur5e/ur_description/urdf/ur5e.urdf",
        &RobotOptions::default(),
    )?;
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;
    let now = JointState {
        position: robot.default_q().to_vec(),
        velocity: vec![0.2; robot.dof()],
        acceleration: vec![0.0; robot.dof()],
    };
    let motion = StartMotion {
        velocity: now.velocity.clone(),
        acceleration: now.acceleration.clone(),
    };
    let goal = vec![0.5, -1.2, 1.0, -1.4, -1.5, 0.3];
    let problem = PlanProblem {
        start_motion: Some(motion.clone()),
        ..PlanProblem::new(0, now.position.clone(), goal)
    };
    let result = batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default())?;
    if let Some(path) = result.best(0) {
        let trajectory = Trajectory::moving(&robot, path, &motion)?;
        trajectory.check_from(&robot, &now)?;
        println!("{:.2} s from the moving start", trajectory.duration());
    }
    Ok(())
}
```

## Options worth knowing

| Option | Default | What it trades |
|---|---|---|
| `IkOptions::seeds` | 32 | More seeds find more and better-cleared solutions, at proportional cost. |
| `PlanOptions::rrt.max_iterations` | 10,000 | Samples a search draws without its clearance before it gives up, after up to 500 with it: more solve rarer problems, and unsolvable ones take longer to give up. |
| `PlanOptions::rrt.clearance`, `shortcut.clearance` | 1.5 cm, 1 cm | Distance searches and shortcuts keep from obstacles, away from the path's ends. More in shortcuts keeps paths clear of geometry the spheres miss, but makes them longer. |
| `PlanOptions::shortcut.passes` | 3 | Shortcut and smoothing passes: more give shorter paths, at some time. |
| `PlanOptions::shortcut.straighten` | on | Straighten each joint along the path: about 1% shorter paths on the Panda benchmark and 13–18% less joint travel on pick-and-place carries, for about 3% of batched throughput. |
| `PlanOptions::control_points` | 24 | Control points of the spline: more follow longer paths more closely. |
| `PlanOptions::seeds` | 0 | Trajectory optimization seeds per problem: with 2, paths are about 3% shorter (shorter than cuRobo's on its benchmark), at milliseconds per problem. |
| `RobotOptions::velocity_scale`, `acceleration_scale`, `jerk_scale` | 1 | A slower robot everywhere: plans, timings, checks and replanning from motion all keep to the scaled limits. `Trajectory::try_new`'s speed scale slows one trajectory instead. |
| `PlanOptions::time_budget` | none | Bounded latency, at the cost of determinism. |
