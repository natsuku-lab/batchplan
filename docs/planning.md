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
- `ik::failures(&device, &worlds, &ik, &options)` says why each unsolved problem has no solution (`IkFailure`): `Unreachable`, or `Collides` with what the reaching seed with the most clearance collides with. The push away from obstacles can hold every seed off a pose that collides, so problems no seed reached are solved once more without it; that costs what their solve did, so it runs only when called.
- `IkProblem::targets` gives each of the robot's hands (`RobotOptions::ee_links`) a target pose, or `None`. Each step solves every targeted hand's damped least squares and pushes the collision gradient out of their Jacobians' row spaces. Hands that share joints, such as two arms on a mobile base or a torso, solve as one system; solved alone, each would move the shared joints for itself in full. A hand without a target holds its arm: in every seed, the joints only that hand moves stay where the seed has them. So one arm can reach while the other waits, or both can reach at once and keep clear of each other. `IkProblem::hold` holds named joints the same way, even where a targeted hand moves on them: a mobile base placed beside a table stays there while its arms reach, as `PlanProblem::hold` keeps the arms still while the base drives.

## Planning a path

`PlanProblem::new(world, start, goal)` plans from rest with every joint free to move; set the other fields on top of it (`PlanProblem { hold, ..PlanProblem::new(world, start, goal) }`).

A plan into a grasp ends with the fingers around the object: let the hand touch it ([allowed contacts](perception.md#allowed-contacts)), and the rest of the arm still keeps clear of it.

`PlanProblem::hold` lists joints that stay where `start` has them, such as the other arm of a two-armed robot while one arm moves (`robot.hand_joints(hand)` lists an arm's); `goal` must have the same values there. The search samples and bends only the other joints, no goal equivalent turns a held joint, and optimized paths are put back on the held values before they are checked, so every control point keeps them.

`plan` runs every problem as its own search, whole on one CPU thread, with the searches in parallel on either device, as VAMP does (Thomason, Kingston and Kavraki, 2024):

1. **The straight line** to each of the goal's equivalents (below) is tried first.
2. **A bent line.** If every straight line collides, the one to the nearest goal is bent: the stretch from where it first collides to where it last does moves 0.05 rad aside, across the line along each joint's axis in turn, both ways, then 0.1, 0.2 and 0.4 rad. The first bend whose edges are free is the path. So a line that an obstacle only grazes keeps its route instead of taking the detour a search would find. Bending costs no measurable time.
3. **RRT-Connect** grows a tree from the start and one from the goals, in turn, toward Halton samples. The first samples lie in a tube around the straight line to the nearest goal: 0.4 rad across it (along each axis across it) and past its ends, doubling every 64 samples until it spans the joint ranges. So the route found stays near the straight line where such a route exists, as cuRobo's roadmap does by sampling an ellipsoid around it; samples from the whole space find detours that shortcuts cannot undo. Extensions step at most `rrt.range` (1 rad), and dynamic domains skip samples far from any node that recently failed. A search gives up after up to 500 samples from the joint ranges keeping its clearance and `rrt.max_iterations` (10,000) more without it (see Clearance, below); the tube's samples come on top.
4. **Shortcutting and smoothing** run for up to `shortcut.passes` passes (3). Each pass replaces waypoints with the longest free edge, splits every edge, and moves each waypoint a quarter of the way toward its neighbours where both new edges stay free. With `shortcut.straighten` (on), after the first pass's shortcuts each joint in turn is straightened where that shortens the path by at least `shortcut.min_change`: it changes at one rate along the whole path while the other joints keep theirs, as in HPP's partial shortcut. That removes a joint's swing out and back (a wrist's, say), which shortcuts between whole configurations keep.
5. **A B-spline** follows the path, with `control_points` control points (24). The first and last three are pinned to the start and goal, so it starts and ends at rest. The others first spread along the smoothed path by arc length, so the curve is smooth and cuts the path's corners. It is checked at samples no farther apart than `shortcut.resolution`, as edges are, except along a straight line, which the search or the shortcuts already checked keeping at least the spline's clearance. Where it fails, the waypoint nearest the failure is pinned: three control points sit on it, so the curve passes through it, stopping there. After up to six pins, the fit starts over on the shortcut path; with every waypoint pinned, the curve runs exactly along the checked path. The first spline that validates is the result, once its stops are eased (as in OpenRAVE's smoother): each stop's three control points spread into the widest blend around the corner that keeps the curve valid. Blends are the smaller of the free control points' spacing and half the shorter edge at the corner, then a quarter and a sixteenth of that; a corner no blend clears keeps its stop. If no spline validates, trajectory optimization pushes the first one off the obstacles a step at a time, for up to ten steps. The first step whose spline keeps the 2 mm margin is the result; the tenth's needs only to be collision-free. If the repair fails too, a path found near the straight line is searched again over the joint ranges alone. A path that still has no valid spline, too long or winding for its control points, is fitted and repaired again with twice as many, while fewer are free than the path has waypoints. So a searched path in `PlanResult::paths` may have more than `control_points`. Optimized seeds keep `control_points`, and a problem whose spline grew is optimized even near the straight line.

**Checking edges.** An edge is collision-checked at evenly spaced states, `rrt.resolution` (1/32 rad; at least 1e-4) apart, eight at once in SIMD lanes: spread over the edge, then stepping back, so a collision anywhere is found early. [Architecture](architecture.md#searches) describes the cheaper tests that skip most of the work.

**Clearance.** A path that only just clears the spheres can graze the meshes where the spheres leave them uncovered. So RRT-Connect keeps `rrt.clearance` (1.5 cm) from obstacles, and shortcuts keep `shortcut.clearance` (1 cm). Both shrink to zero within `clearance_ramp` (0.5 rad) of the start and the goals, since paths often start or end in contact, as at a grasp. The search's wider clearance finds routes through open space, which clear the meshes more often; the shortcuts' narrower one lets the path pull tight. If RRT-Connect finds no path within 500 samples, it continues from the same trees without its clearance for up to `rrt.max_iterations` samples more, so narrow passages stay passable. Splines keep the shortcuts' clearance first, then a 2 mm margin (or the shortcuts' clearance, if smaller), so the curve cannot graze an obstacle between the samples it is checked at.

**Validation.** Each spline candidate is sampled densely (`validate_substeps` per span) and checked for world and self-collision and joint limits. `result.best(problem)` is the shortest valid path, and `result.solved()` yields every solved problem with it.

### Trajectory optimization (optional)

`PlanOptions::seeds` (0 by default) adds trajectory optimization of that many seeds per problem, batched on the device:
- **Seeds.** The first starts from the searched spline, the second from the straight line to the goal, as cuRobo's first attempt does. Further seeds run straight to the goal's equivalents, or bend through random via points. Problems searched to within 2% of the straight line are not optimized, since no seed could be much shorter; their seeds stay invalid.
- **Cost.** Length and smoothness, as squared differences of the control points (weighted `velocity_weight`, 10) and squared second differences (`acceleration_weight`, 5), plus a collision cost for the world and for self-collision, sampled along the curve (`samples_per_span` points per span) and active within `collision.margin` (1.2 cm) of an obstacle and `collision.self_margin` (1 cm) between links.
- **Optimizer.** L-BFGS (`iterations`, 100, remembering `history` steps). Each iteration prices four step sizes along the L-BFGS direction in one batched pass and keeps the cheapest.

Repairing a stuck spline (above) uses the same optimizer with its own settings, whatever these options say: ten steps, smoothness weighted over length (`acceleration_weight` 50, `velocity_weight` 2) and a 2 cm margin. These repair more splines than the length-favouring weights do.

Optimized seeds are validated like the searched path, and `best` picks the shortest valid one of all. On the benchmark's box-only problems, two seeds give 3.137 rad paths at the median, against 3.243 without and cuRobo's 3.144 ([benchmarks](benchmarks.md#the-earlier-comparison)). Optimization costs milliseconds per problem (15.1 ms on 14 CPU threads; 746 problems/s in batches on an M4 Pro's GPU), against a fraction of a millisecond for the search.

### Goals a turn away

Continuous joints are unbounded, and joints whose range spans more than a turn (the UR5e's ±2π) reach the same pose at angles a turn apart. Goals are reached at whichever equivalent is nearest or free: the search takes them all as its goal set, and continuous joints step and measure the short way round. On the UR5e, a goal whose shoulder has turned past the table below is reached the long way round when the short way is blocked.

### Time budgets and determinism

`PlanOptions::time_budget` stops every search and the optimization once it passes, and returns the valid paths found so far. Without a budget nothing checks the clock, and two runs give bit-identical results, on either device.

### Why a problem failed

`PlanResult::failures` holds, per problem, why it has no valid path (`PlanFailure`), the first thing that stopped it, or `None` where it has one:

| `PlanFailure` | What stopped it | What to try |
|---|---|---|
| `StartCollides { contacts }` | the start collides, with what `contacts` lists | plan from a free state; a link that always touches the table needs a [touch](perception.md#allowed-contacts) |
| `GoalCollides { contacts }` | the goal collides, and so does every equivalent a turn away | another IK solution or goal |
| `NoPath` | the search drew `rrt.max_iterations` samples without joining start and goal | more samples, or the way is blocked |
| `NoSpline` | a path was found, but no spline along it checked valid or followed the start's motion | more `control_points`, or a start moving slower |
| `OutOfTime` | the `time_budget` ran out first | a longer budget |

Each contact (`Contact`) names a link and an obstacle, or two links, with the signed distance between them. `device.contacts(&worlds, &item_world, &q, within)` lists them for any batch of configurations, nearest first, as the checks see them: touches, links no joint moves and ignored pairs are left out. Both devices compute them exactly on the CPU, so they list the same. `pick` and `place` report a candidate's contacts too (below), and `follow` the contacts of a straight move's step that collides (`ToolPathResult::contacts`).

## Constraints along the path

A `HandConstraint` bounds where a hand may be at every point of a plan, as a cup or a tray must be carried: its z axis within a cone (its tilt), its turn about that axis (its twist), and its origin within a box, in the world or in another hand's frame. `HandConstraint::new(hand, pose)` leaves all three free; set the ones that matter. `PlanProblem::constraints` takes any number of them, on any hands.

```rust,no_run
use std::f32::consts::PI;

use batchplan::glam::{Quat, Vec3};
use batchplan::{
    Device, Error, HandConstraint, IkOptions, IkProblem, PlanOptions, PlanProblem, Pose, Robot,
    RobotOptions, World,
};

fn main() -> Result<(), Error> {
    let robot = Robot::load("assets/franka/franka_panda.urdf", &RobotOptions::default())?;
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;
    let o = IkOptions::default();
    // The gripper pointing straight down, then 40 cm across, by IK from where it starts.
    let down = Quat::from_rotation_x(PI);
    let here = IkProblem::new(0, Pose::new(Vec3::new(0.4, -0.2, 0.3), down));
    let ik = batchplan::solve_ik(&device, &worlds, &[here], &o)?;
    let Some(start) = ik.best(0).map(<[f32]>::to_vec) else {
        return Ok(());
    };
    let there = IkProblem {
        seed: Some(start.clone()),
        ..IkProblem::new(0, Pose::new(Vec3::new(0.4, 0.2, 0.3), down))
    };
    let ik = batchplan::solve_ik(&device, &worlds, &[there], &o)?;
    let Some(goal) = ik.nearest(0, &start).map(<[f32]>::to_vec) else {
        return Ok(());
    };
    // Within 5° of straight down all the way, turning freely about the vertical.
    let level = HandConstraint {
        tilt: 5f32.to_radians(),
        ..HandConstraint::new(0, Pose::new(Vec3::ZERO, down))
    };
    let problem = PlanProblem {
        constraints: vec![level],
        ..PlanProblem::new(0, start, goal)
    };
    let result = batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default())?;
    println!("carried level: {}", result.best(0).is_some());
    Ok(())
}
```

- **What it bounds.** Each bound is measured from `pose`, in the world. `tilt` is the angle between the hand's z axis and `pose`'s: a cone, which "within 5° of upright" means, rather than a box on roll and pitch, which allows 7.07° at its corners and is singular at a pitch of ±90°. `twist` is how far the hand turns about its z axis once its tilt is taken out (the swing-twist decomposition); a hand tilted half a turn has no such turn, so a bounded twist needs a tilt bound of at most π/2. `position` is how far the hand's origin may be from `pose.position` along each world axis. Infinity leaves an axis of the box free, and π or more an angle. Bounds below what IK solves to (1 mm, 0.01 rad) count as those, so a goal IK solved for the pose keeps a constraint pinned to it.
- **Relative to another hand.** With `relative_to: Some(other)`, `pose` and the bounds are in that hand's frame and move with it, the box along its axes, as MoveIt's constraints take a link's frame. Two hands holding one tray pin the second hand where it holds it in the first's frame, `HandConstraint { relative_to: Some(0), position: Vec3::ZERO, tilt: 0.0, twist: 0.0, ..HandConstraint::new(1, place) }` with `place` the first hand's pose inverted times the second's at the start, and keep the first upright to carry the tray level. The floors double (2 mm, 0.02 rad): both hands' IK errors add.
- **The ends.** The start and the goal must keep their constraints; an input error names the bound broken and by how much. `Robot::breach(constraints, q)` runs the same test, to drop IK goals that break them before planning. Take them from one IK branch: on the example's reaches, a start IK placed with the Panda's third joint at its limit, the arm wound the other way round its base, reaches only 5 of 54 goals inside the zone even with 100,000 samples, since unwinding the arm with the gripper upright leaves the zone; from the solution nearest the ready pose, 54 of 54.
- **How it plans.** As CBiRRT does, by projection. RRT-Connect's extensions and connections walk toward their targets in hops of eight `rrt.resolution` steps (0.25 rad), one SIMD pass each. Each hop is projected back inside the bounds by damped least squares on the bounds it breaks, through the Jacobians of the hand and of the hand it is relative to. Projection lands a little inside (half the bound, up to 1 cm and 2°), so the edges between hops have room to bulge. The straight line comes first, then its projected walk, before any sampling. Shortcuts and smoothing keep the constraints too, and a smoothed waypoint is projected back inside before it is checked. Under a bound below 2 cm or 4°, twice that room, as for a hand pinned in another's frame, straight shortcuts mostly leave it: from a waypoint whose straight shortcuts stop short of the end, shortcutting walks projected toward farther waypoints, the end first, halving a hop whose edge bulges out at its middle, and keeps a walk that arrives shorter.
- **The guarantee.** Every state a search, a shortcut or a spline is checked at is tested against the bounds as given, in the SIMD lanes beside the collision test, so the returned path keeps them wherever it is checked: at `shortcut.resolution` along searched paths and their splines (and at least `validate_substeps` per span), and at `validate_substeps` per span along optimized seeds. Between those states a path may stray past a bound by a hair.
- **The spline.** The spline's control points are fitted so that the curve passes through points along the path rather than near them, clamped to the joint limits: its gap from the path falls with the fourth power of their spacing rather than the square. Under a bound below 2 cm or 4°, each point is projected inside the bounds first, since one on a straight edge between waypoints can leave them. So the usual 24 control points follow a long, curved constrained path within 1°, and a long carry with one hand pinned in the other's frame. A path too long for them gets more (step 5 above).
- **Optimization.** With `seeds`, trajectory optimization keeps the constraints as a cost, as cuRobo and TrajOpt do: the squared distance past each bound projection lands within, at every optimizer sample, on the CPU and the GPU alike. An optimized seed is kept only where every validation sample keeps the bounds as given. On the upright reaches below, 4 seeds shorten 25 of 54 paths and their mean length by 3.2%; on two arms carrying a tray, the mean length by 4%. Splines too short to follow a path are repaired the same way. On a GPU, the first constrained optimization a device runs builds its kernels (1.8 s on the M4 Pro the first time, 10 ms once the driver has cached them), which a time budget does not cover.
- **Everywhere a path is planned.** `join` keeps a `Sequence`'s constraints in every blend, and a join whose blends all break them stops. `pick` and `place` keep theirs along the transit, as MTC's Connect stage takes path constraints; a candidate whose move in begins where they break stops at the transit. `Follower` keeps `FollowerOptions::constraints` in its replans, and stops where a replan would start outside them. `follow` and `servo` set the hand's path themselves and take none.
- **What it costs.** On upright reaches past a pillar (`cargo run --release --example constraints`), the gripper within 5° of straight down: 54 of 54 solved, 0.11 ms per problem alone at the median and 2.6 ms for all 54 in one batch on the M4 Pro, against 0.09 and 1.0 ms without the constraint. Two arms carry a tray level over a block in 1.11 ms per problem ([Benchmarks](benchmarks.md#constraints-along-the-path)).

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
- **Proposals.** `grasps(&obstacle, &GraspOptions)` proposes grasps for a box, ball, cylinder or capsule, in its frame: the hand facing each side, closing across what fits between its fingers (`opening`, 8 cm), `depth` (2 cm) in from the surface. They suit a parallel gripper whose IK frame lies between its fingertips, its z axis pointing out of the hand and its y axis along the closing direction, as the Panda's `ee_link`. Where no grasp fits, `grasps` says why: the object is wider than the fingers open every way, or a floor rules out every grasp. `place_poses(&obstacle, pose, turns)` gives an object's symmetric turns at one spot: 2 for a box (4 with a square footprint), `turns` for a round object.
- **Near a support.** A small object on a table puts the fingertips in the table at the full depth. Give `GraspOptions::tip` the fingers' reach past the frame (`robot.tip(hand)`) and `floor` the height they keep above, such as the table's top and a few millimeters, and pass the obstacle as it stands in the world: grasps from above then hold the object as deep as keeps the fingertips over the floor, and grasps that cannot, or come from below, are left out.
- **Why it failed.** When no candidate gets through, the task's `TaskFailure` holds each candidate's step and reason (`CandidateStop`: a pose out of reach, a collision or a jump, or no path), and says it in a sentence: "none of 24 candidates got through: 20 at the grasp or release (1 out of reach, 19 colliding), 4 moving away (4 colliding)". IK that finds nothing is classified by `ik::failures`. A candidate stopped by a collision carries what collides (`CandidateStop::contacts`): at the start, at the IK solution with the most clearance, holding the object where a place lets go, or at the straight move's next step.
- **Speed.** On the Ryzen AI Max+ 395 (32 threads), picking a box by its 24 grasps takes 6.5 ms on the CPU device, and 64 such picks at once 333 ms; on the Radeon 8060S, 15 ms and 114 ms. Setting a box down at either of its turns takes 2.4 ms on the CPU, and 16 at once 11.6 ms. [Benchmarks](benchmarks.md#picking-and-placing) has both machines and every batch size.

## Joining motions

`join(&device, &worlds, &sequences, &JoinOptions)` runs consecutive motions without stopping between them, as MoveIt's Pilz planner blends a motion sequence. A `Sequence` names its world, its paths in order, each rest-to-rest as `plan` and `follow` return them and starting where the one before ends, and a blend radius for each join, as each item of a Pilz sequence gives its own.

```rust,no_run
# use batchplan::{Device, Error, JoinOptions, Sequence, Trajectory, World};
# fn run(device: &Device, transit: Vec<f32>, approach: Vec<f32>) -> Result<(), Error> {
let worlds = device.upload(&[World::default()])?;
let sequence = Sequence {
    world: 0,
    paths: vec![transit, approach],
    blends: vec![0.05],
    constraints: vec![],
};
let joined = batchplan::join(device, &worlds, &[sequence], &JoinOptions::default())?;
// One path per stretch the robot moves without stopping: here one, unless the join stopped.
for path in &joined.paths[0] {
    let trajectory = Trajectory::try_new(device.robot(), path, 1.0)?;
    println!("{:.2} s", trajectory.duration());
}
# Ok(())
# }
```

- **The blend.** Near a join, the joined path leaves the motions only while every hand is within the join's radius of where it is at the join (0 stops there), and is the motions themselves elsewhere: a tool path stays on its line outside the blend. Inside it, the path follows a curve that sets out and arrives along both motions without curving there, so jerk-limited timing can follow it at speed.
- **Kept only where it is sooner.** Each join weighs three blends: the widest within the radius, one dropping half as many control points, and the narrowest. Those that make the stretch so far and the next motion finish sooner than with a stop between them are checked for collision, soonest first, and the first that is free is kept. A turn back, or a tight turn at speed, rarely gains, and keeps its stop. `JoinResult::blends` gives each join's radius, `None` where it stops. Joining never makes a sequence take longer.
- **The result.** Per sequence, one path for each stretch the robot moves without stopping, to time with `Trajectory::try_new`. A stretch of one motion is that motion as given.
- **What it gains.** On pick-like sequences, a transit into a straight move in and a straight move out into the next transit, a 5 cm radius finished 4.9% sooner on average on a reBot B601 (14% at most) and 7.5% on an SO-101 (18%), and joining 60 of them took 16 ms on 32 threads ([Benchmarks](benchmarks.md#joining-motions)). Motions that carry on in about the same direction gain more; a sharp turn needs most of a stop's braking anyway.
- **Limits.** Each stretch has one timing, so a hand speed cap (`Trajectory::with_hand_speed`) caps all of it, and `pick` and `place` still stop between their steps. Each join times the stretch so far, so joining a long sequence costs more per join than a short one.

## Executable trajectories

`Trajectory::try_new(&robot, path, speed_scale)` keeps the path and adds a time map `s = σ(t)` (itself a uniform cubic B-spline), choosing the faster of two timings:

- **Time-optimal** (TOPP-RA, Pham and Pham 2018): the fastest timing of the path under joint velocity and acceleration limits, plus a speed cap from the jerk limit, smoothed into a B-spline time map. Where the smoothed map overshoots a limit, the caps are lowered there and the profile solved again. On planned Panda paths it is 12% faster than uniform timing on average and never slower, at about 0.5 ms per trajectory.
- **Uniform**: `σ(t) = t / h`, with the smallest `h` that keeps every joint within its limits. Velocity, acceleration and jerk are then B-splines over the control points' differences, so the bound holds exactly along the whole curve.

Either is slowed by `speed_scale` (1.0 is as fast as the limits allow). Limits come from `RobotOptions::joint_limits` for joints it names, then from the robot description when it has them (URDF 1.2 `acceleration` and `jerk`), otherwise from `RobotOptions` (5 rad/s² and 50 rad/s³ by default). `Trajectory::with_hand_speed(&robot, path, &HandSpeed)` keeps a hand under caps too: `linear` on its speed (m/s) and `angular` on how fast it turns (rad/s), where `f32::INFINITY` leaves one uncapped. Both timings keep them, and `check` checks the joints' limits only.

- **For control loops**: `trajectory.at(t, &mut state)` writes position, velocity and acceleration without allocating.
- **For datasets**: `trajectory.sample(hz)` returns fixed-rate samples.
- **For safety layers**: a `Trajectory` is plain serializable data. `trajectory.check(&robot)` refuses one that is non-finite, does not start and end at rest, leaves a joint range, runs backward, or exceeds a velocity, acceleration or jerk limit.
- **Its ends**: `trajectory.start()` and `trajectory.end()` are where it starts and comes to rest: its first and last control points, or for a stop, where the stop begins and ends along the path.
- **Payloads**: timing knows the joints' limits, not the load. A held object collides as part of the robot ([holding objects](robots.md#holding-objects)), but batchplan has no payload mass or inertia, so a heavy carry is timed as an empty one: slow it with `speed_scale` or the robot's limit scales.
- **For stopping**: `trajectory.stop(&robot, t)` is a controlled stop: the same path from time `t`, braking to rest as soon as an even deceleration within the limits allows. It stays on the checked path, so it needs no new collision check; verify it with `check_from` and the state at `t`. On planned Panda reaches it takes under 1.2 ms to compute, rests in about half the time left, and errors only where the path ends sooner than any stop would (late in short, fast motions).

### Starting from a moving state

To replan mid-motion, give `PlanProblem::start_motion` the robot's current velocity and acceleration. The spline's first control points then continue that motion, and `Trajectory::moving(&robot, path, &start_motion)` times it from exactly that state, as fast as the limits allow.
- `plan` only returns splines that can run from that state: `Trajectory::moving` times them from it, keeping every limit and joint range, as `check_from` checks. Where momentum would carry a joint out of its range in the first span, longer first spans are tried; a path that still cannot run is not returned.
- The spline's first spans last as long as the robot needs to change its velocity (about |v| / a). Where it still cannot follow the motion, longer spans are tried, and then trajectory optimization smooths the spline until it can.
- Where no timing fits from the start as it is, timing tries again with the full limits at the start, since its motion is not ours to choose, easing to the usual headroom over the path's first two spans, and stretches the rest of the motion where its smoothed map needs it while keeping the start's speed. Replanned to new goals from halfway along planned Panda motions, at about 3 rad/s, motions took 3.8 s on average, against 12.9 s when timing gave up on such starts, and none took longer ([Benchmarks](benchmarks.md#replanning-while-moving)).
- When reaches are replanned to the same goal from partway through, plans succeeded every time, with every joint below 1.5 rad/s and with one faster, near the Panda's full speed (24 of 24). Where one fails, momentum carries a joint out of its range or into the scene.
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

## Live motion

A `Follower` runs a trajectory while the world changes. Each `tick(&device, &worlds, now, measured)` returns the state to command (`Tick`) and checks the rest of the motion against the worlds it is given: samples `check_every` apart (20 ms), in one `evaluate` batch.
- **Measured state:** given the state the robot was measured in, a tick compares it with where the trajectory has it. Within `tolerance` (0.2 rad per joint), as a servo settles short under load, the follower goes on from its own states, so commands never step. Farther, as an arm held back or pushed, it plans from the measured state, within `budget`, and reports `Reanchored { off }`; where it cannot, it holds the robot where it is and reports `Blocked`. `None` trusts the trajectory, as in simulation.
- **Replanned:** when the rest collides, it plans from the state the trajectory reaches `budget + lead` later (50 ms and 0), starting in motion, within `budget`, and switches to the new trajectory then. A tick that replans takes as long as the plan, up to `budget`; a controller that keeps running the current trajectory meanwhile (`trajectory()`, `began()`) never waits for it. `Replanned { at, trajectory }` hands the new trajectory over when it is planned, so a controller in another process or on the robot can take it up at `at`; `lead` is the time it has to get there.
- **Before a takeover:** the rest of the motion is the current trajectory up to the takeover, then the replan. If either collides, the replan is dropped and the robot stops.
- **Stopping, then blocked:** when no plan arrives in time, or the collision comes sooner than the budget, it takes the controlled stop along its path and reports `Blocked` at rest. Plan again from there, or give up.
- **Collides:** the stop is checked too. When no stop within the limits ends short of the collision, the status is `Collides { at }`, with the time it comes: stop the robot some other way.

`servo(&device, &worlds, world, &state, target, &ServoOptions)` is one control step of the hand toward a pose that may move, as MoveIt Servo does. The step works like this:
- **Velocity:** the hand moves at the pose's error times `gain`, within the `linear` and `angular` caps, by damped least squares on its Jacobian. It turns the shorter way round.
- **Limits:** the joints keep their velocity, acceleration and jerk limits, and reach the velocity they are asked for without overshooting it. A joint brakes in time to rest within its range.
- **Near obstacles:** it checks where the robot would come to rest if it braked at the limits from the state it commands, and the path there, with clearances exact out to `slow_within`. Within `slow_within` of an obstacle (2 cm) it slows, so that an even deceleration brings it to `halt_within` (2 mm); there it brakes to rest instead.
- **Too close to stop:** `Halted { clearance }` carries the least clearance while it brakes. At zero or below, no stop within the limits ends short of contact, as when an obstacle appears closer than the robot can stop.

`ServoStep::status` says whether it moved normally, slowed, halted or arrived.

```rust,no_run
use batchplan::{
    Device, Error, FollowStatus, Follower, FollowerOptions, PlanOptions, PlanProblem, Robot,
    RobotOptions, Trajectory, World,
};

fn main() -> Result<(), Error> {
    let robot = Robot::load(
        "assets/ur5e/ur_description/urdf/ur5e.urdf",
        &RobotOptions::default(),
    )?;
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;
    let goal = vec![0.5, -1.2, 1.0, -1.4, -1.5, 0.3];
    let problem = PlanProblem::new(0, robot.default_q(), goal);
    let result = batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default())?;
    let Some(path) = result.best(0) else {
        return Ok(());
    };
    let trajectory = Trajectory::try_new(&robot, path, 1.0)?;
    let mut follower = Follower::try_new(&robot, 0, trajectory, 0.0, FollowerOptions::default())?;
    // A 100 Hz loop; upload the worlds again whenever perception updates them. A tick that
    // replans can take up to the 50 ms budget.
    for step in 0..10_000 {
        let tick = follower.tick(&device, &worlds, step as f32 * 0.01, None)?;
        // Command tick.state here.
        if matches!(tick.status, FollowStatus::Arrived | FollowStatus::Blocked) {
            break;
        }
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
| `PlanOptions::control_points` | 24 | Control points of the spline and of optimized seeds: more follow longer paths more closely. A searched path that no spline this size follows gets more. |
| `PlanOptions::seeds` | 0 | Trajectory optimization seeds per problem: with 2, paths are about 3% shorter (shorter than cuRobo's on its benchmark), at milliseconds per problem. |
| `RobotOptions::velocity_scale`, `acceleration_scale`, `jerk_scale` | 1 | A slower robot everywhere: plans, timings, checks and replanning from motion all keep to the scaled limits. `Trajectory::try_new`'s speed scale slows one trajectory instead. |
| `PlanOptions::time_budget` | none | Bounded latency, at the cost of determinism. |
