# Planning

Planning in batchplan is three steps. IK and planning are batched calls on a [`Device`](devices.md); timing runs on each path:

1. **IK** (`solve_ik`) turns target poses into joint configurations. `plan` runs it itself for a goal given as a pose.
2. **Planning** (`plan`) finds a collision-free path from a start to a goal, a configuration or the hand's pose (the straight line, bent if an obstacle only grazes it, else RRT-Connect), then shortens and smooths it into a B-spline. Trajectory optimization can polish it further.
3. **Timing** (`Trajectory::try_new`) turns the path into a trajectory a controller can run, within the robot's velocity, acceleration and jerk limits.

Every call takes many problems, each naming its world, so one call can cover thousands of scenes.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::{
    Device, Error, PlanOptions, PlanProblem, Pose, Robot, RobotOptions, Trajectory, World,
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
    let problems: Vec<PlanProblem> = targets
        .iter()
        .map(|&p| PlanProblem::new(0, robot.default_q(), Pose::new(p, down)))
        .collect();
    let result = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default())?;

    // `solved()` yields each problem that has a path, with its world, so they cannot be mixed up.
    for s in result.solved() {
        let trajectory = Trajectory::try_new(&robot, s.solution, 1.0)?;
        println!("problem {}: {:.2} s", s.index, trajectory.duration());
    }
    Ok(())
}
```

## Inverse kinematics

`solve_ik` runs damped least squares from many seeds per target (64 by default, 30 steps at most), with the collision gradient projected into the Jacobian's null space so seeds slide away from obstacles without leaving the target. Every seed takes the first `screening` steps (8); then only `kept` of each problem's (4) take the rest: the seeds not yet at their targets, the nearest first, then seeds that have arrived, most of which stay at rest. Collision-free seeds go before the others where the steps push away from collisions; with `collision_step: 0`, or a robot without collision spheres, whether a seed collides after the screening says little about where it ends, and the seeds go by their errors alone. Most seeds that will arrive have by then, so the remaining steps go where they still matter, and the others are returned as the screening left them. On a GPU all of it is one round trip, and a small call of up to 64 steps one submission. `screening: 0` gives every seed every step, as more than 32,768 seeds per problem get. A step that would move no joint by more than 1e-6 rad is not taken, so a seed that has come to rest stays exactly there, and CPU threads skip the rest of its steps. A seed succeeds when it is within `position_tolerance` (1 mm) and `rotation_tolerance` (0.01 rad) of the target and collision-free.

- `IkProblem::new(world, target)` targets a one-handed robot's hand from the default pose; the fields below cover the rest.
- `ik.best(problem)` is the successful seed with the most obstacle clearance; `ik.solved()` yields every solved problem with that configuration.
- `IkProblem::seed` starts seed 0 from a configuration you give (the robot's current state, say) instead of the default pose, and `ik.nearest(problem, &q)` returns the successful seed nearest `q`. Together they keep short moves on the same IK branch, as `pick` and `place` do at the grasp and where the hand lets go.
- `ik::failures(&device, &worlds, &ik, &options)` says why each unsolved problem has no solution (`IkFailure`): `Unreachable`, or `Collides` with what the reaching seed with the most clearance collides with. The push away from obstacles can hold every seed off a pose that collides, so problems no seed reached are solved once more without it; that costs what their solve did, so it runs only when called.
- `IkProblem::targets` gives each of the robot's hands (`RobotOptions::ee_links`) a target pose, or `None`. Each step solves every targeted hand's damped least squares and pushes the collision gradient out of their Jacobians' row spaces. Hands that share joints, such as two arms on a mobile base or a torso, solve as one system; solved alone, each would move the shared joints for itself in full. A hand without a target holds its arm: in every seed, the joints only that hand moves stay where the seed has them. So one arm can reach while the other waits, or both can reach at once and keep clear of each other. `IkProblem::hold` holds named joints the same way, even where a targeted hand moves on them: a mobile base placed beside a table stays there while its arms reach, as `PlanProblem::hold` keeps the arms still while the base drives.

## Planning a path

`PlanProblem::new(world, start, goal)` plans from rest with every joint free to move; set the other fields on top of it (`PlanProblem { hold, ..PlanProblem::new(world, start, goal) }`). `holding` plans with an object in a hand, each problem its own ([holding objects](robots.md#holding-objects)); `IkProblem` and `ToolPath` take it too.

A goal (`Goal`) is a configuration or the hands' poses. `PlanProblem::new` takes a `Pose` for a one-handed robot, and `Goal::Poses` takes a pose or `None` per hand (`RobotOptions::ee_links`); for `None`, the joints only that hand moves stay where `start` has them. `plan` solves IK for every pose goal of the call in one batch, seeded from each start, keeping its `hold`, its `holding` and its `constraints` (`PlanOptions::goal_ik` sets the seeds, iterations and tolerances). The path ends at the collision-free solution nearest the start, counting each solution at its nearest whole turn, as the path reaches it ([goals a turn away](#goals-a-turn-away)). Poses no seed reaches collision-free are solved again with four times the seeds, up to sixteen times: in clutter, more seeds reach what the first miss. The first round counts toward a `time_budget` and always runs; later rounds do not start once the budget has passed, and their problems fail as `OutOfTime`. `PlanResult::problems` keeps the goals as given. To choose among the solutions yourself, by clearance or by branch, solve IK ([above](#inverse-kinematics)) and plan to the configuration.

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

**Validation.** Each spline candidate is sampled at least `validate_substeps` times per span, no farther apart than `shortcut.resolution`, and checked for world and self-collision and joint limits. `result.best(problem)` is the shortest valid path, and `result.solved()` yields every solved problem with it.

### Trajectory optimization (optional)

`PlanOptions::seeds` (0 by default) adds trajectory optimization of that many seeds per problem, batched on the device:
- **Seeds.** The first starts from the searched spline, the second from the straight line to the goal, as cuRobo's first attempt does. Further seeds run straight to the goal's equivalents, or bend through random via points. Problems searched to within 2% of the straight line are not optimized, since no seed could be much shorter; their seeds stay invalid.
- **Cost.** Length and smoothness, as squared differences of the control points (weighted `velocity_weight`, 10) and squared second differences (`acceleration_weight`, 5), plus a collision cost for the world and for self-collision, sampled along the curve (`samples_per_span` points per span) and active within `collision.margin` (1.2 cm) of an obstacle and `collision.self_margin` (1 cm) between links.
- **Optimizer.** L-BFGS (`iterations`, 100, remembering `history` steps). Each iteration prices four step sizes along the L-BFGS direction in one batched pass and keeps the cheapest.

Repairing a stuck spline (above) uses the same optimizer with its own settings, whatever these options say: ten steps, smoothness weighted over length (`acceleration_weight` 50, `velocity_weight` 2) and a 2 cm margin. These repair more splines than the length-favouring weights do.

Optimized seeds are validated as the searched path's spline is checked: at the same samples, keeping the margin from the world it keeps away from its ends (2 mm, or `shortcut.clearance` if less), and their lengths are measured at those samples too. `best` picks the shortest valid one of all. Optimization trades additional solve time for shorter paths; measure it on the intended workload with `PlanOptions::seeds` and the [benchmark commands](benchmarks.md#reproducing).

### Goals a turn away

Continuous joints are unbounded, and joints whose range spans more than a turn (the UR5e's ±2π) reach the same pose at angles a turn apart. Goals are reached at whichever equivalent is nearest or free: the search takes them all as its goal set, and continuous joints step and measure the short way round. On the UR5e, a goal whose shoulder has turned past the table below is reached the long way round when the short way is blocked.

### Measured starts at joint limits

Starts outside the joint limits are input errors by default. A measured state just past a stop can
opt into correction with `PlanOptions { start_tolerance: 0.001, ..Default::default() }`: each bounded
joint at most that far outside its limits is clamped to the nearest limit, in radians or meters for
prismatic joints. A larger violation is an `Error::Input` naming the joint and value. The tolerance
must be finite and nonnegative; zero keeps strict validation. Continuous joints keep their values,
including whole turns, and a nonfinite start is always refused.

Correction happens before IK, collision and constraint checks. Goals, held-joint rules and start
velocities and accelerations remain unchanged: a joint goal still needs to be within its limits and
to agree with the corrected start on held joints, and a moving start still needs a feasible path.
The caller's inputs stay intact; `PlanResult::problems` holds the starts actually planned.
`ToolPathOptions::start_tolerance` does the same for `follow`, whose hand path starts at the corrected
configuration. `pick`, `place` and `handover` use `TaskOptions::plan.start_tolerance` for their initial
states, even when a task omits its transit, and retain the corrected starts in their results. Python
uses the same options, for example `bp.PlanOptions(start_tolerance=0.001)`.

### Time budgets and determinism

`PlanOptions::time_budget` stops every search and the optimization once it passes, and returns the valid paths found so far. Without a budget nothing checks the clock, and two runs give bit-identical results, on either device. A plan to a pose ends where IK solves it, and a GPU rounds IK unlike the CPU, and by the batch's size, so its plan there can end slightly away from the CPU's and from the same problem's in another batch.

### Why a problem failed

`PlanResult::failures` holds, per problem, why it has no valid path (`PlanFailure`), the first thing that stopped it, or `None` where it has one:

| `PlanFailure` | What stopped it | What to try |
|---|---|---|
| `StartCollides { contacts }` | the start collides, with what `contacts` lists | plan from a free state; a link that always touches the table needs a [touch](perception.md#allowed-contacts) |
| `GoalCollides { contacts }` | the goal collides, and so does every equivalent a turn away; for a pose goal, IK reaches the poses only in collision | another IK solution or goal |
| `GoalUnreachable` | no IK seed reaches the goal's poses within the tolerances, keeping the held joints and constraints | a pose within reach, or more `goal_ik` iterations |
| `NoPath` | the search drew `rrt.max_iterations` samples without joining start and goal | more samples, or the way is blocked |
| `NoSpline` | a path was found, but no spline along it checked valid or followed the start's motion | more `control_points`, or a start moving slower |
| `OutOfTime` | the `time_budget` ran out first | a longer budget |

Each contact (`Contact`) names a link and an obstacle, or two links, with the signed distance between them. `device.contacts(&worlds, &item_world, &q, within)` lists them for any batch of configurations, nearest first, as the checks see them: touches, links no joint moves and ignored pairs are left out. Both devices compute them exactly on the CPU, so they list the same. `pick` and `place` report a candidate's contacts too (below), and `follow` the contacts of a straight move's step that collides (`ToolPathResult::contacts`). `device.contacts_holding(.., &holding)` lists them for configurations that hold objects (a `Held` or `None` per item), on the device that holds nothing: what a held object comes near is under its link, `"held by hand 0"`.

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
- **The ends.** The start and the goal must keep their constraints; an input error names the bound broken and by how much. `Robot::breach(constraints, q)` runs the same test, to drop IK goals that break them before planning. Take them from one IK branch: on the example's reaches, a start IK placed with the Panda's third joint at its limit, the arm wound the other way round its base, reaches only a few of the goals inside the zone even with 100,000 samples, since unwinding the arm with the gripper upright leaves the zone; from the solution nearest the ready pose, all 53.
- **How it plans.** As CBiRRT does, by projection. RRT-Connect's extensions and connections walk toward their targets in hops of eight `rrt.resolution` steps (0.25 rad), one SIMD pass each. Each hop is projected back inside the bounds by damped least squares on the bounds it breaks, through the Jacobians of the hand and of the hand it is relative to. Projection lands a little inside (half the bound, up to 1 cm and 2°), so the edges between hops have room to bulge. The straight line comes first, then its projected walk, before any sampling. Shortcuts and smoothing keep the constraints too, and a smoothed waypoint is projected back inside before it is checked. Under a bound below 2 cm or 4°, twice that room, as for a hand pinned in another's frame, straight shortcuts mostly leave it: from a waypoint whose straight shortcuts stop short of the end, shortcutting walks projected toward farther waypoints, the end first, halving a hop whose edge bulges out at its middle, and keeps a walk that arrives shorter.
- **The guarantee.** Every state a search, a shortcut or a spline is checked at is tested against the bounds as given, in the SIMD lanes beside the collision test, so the returned path keeps them wherever it is checked: at `shortcut.resolution` along searched paths, their splines and optimized seeds (and at least `validate_substeps` per span along a spline). Between those states a path may stray past a bound by a hair.
- **The spline.** The spline's control points are fitted so that the curve passes through points along the path rather than near them, clamped to the joint limits: its gap from the path falls with the fourth power of their spacing rather than the square. Under a bound below 2 cm or 4°, each point is projected inside the bounds first, since one on a straight edge between waypoints can leave them. So the usual 24 control points follow a long, curved constrained path within 1°, and a long carry with one hand pinned in the other's frame. A path too long for them gets more (step 5 above).
- **Optimization.** With `seeds`, trajectory optimization keeps the constraints as a cost, as cuRobo and TrajOpt do: the squared distance past each bound projection lands within, at every optimizer sample, on the CPU and the GPU alike. An optimized seed is kept only where every validation sample keeps the bounds as given. On the upright reaches below, 4 seeds shorten 25 of 53 paths and their mean length by 2.5%; on two arms carrying a tray, the mean length by 4%. Splines too short to follow a path are repaired the same way. On a GPU, the first constrained optimization a device runs builds its kernels (0.7 s on the M4 Pro the first time, 5 ms once the driver has cached them), which a time budget does not cover.
- **Everywhere a path is planned.** `join` keeps a `Sequence`'s constraints in every blend, and a join whose blends all break them stops. `pick` and `place` keep theirs along the transit, as MTC's Connect stage takes path constraints; a candidate whose move in begins where they break stops at the transit. `Follower` keeps `FollowerOptions::constraints` in its replans, and stops where a replan would start outside them. `follow` and `servo` set the hand's path themselves and take none.
- **What it costs.** On upright reaches past a pillar (`cargo run --release --example constraints`), the gripper within 5° of straight down: 53 of 53 solved, 0.09 ms per problem alone at the median and 2.7 ms for all 53 in one batch on the Ryzen's 32 threads, against 0.06 and 0.6 ms without the constraint. Two arms carry a tray level over a block in 1.40 ms per problem ([Benchmarks](benchmarks.md#constraints-along-the-path)).

## Moving the hand along a path

`follow(&device, &worlds, &[ToolPath], &ToolPathOptions)` moves a hand through poses: in a straight line from each to the next, turning by the shortest rotation, as MoveIt's Cartesian paths do. Each `ToolPath` names its world, its start configuration, the hand, and the `waypoints` after the start; only the joints that move that hand move, and of those, `hold` keeps where the start has them: on a robot whose arms share joints (a torso), the other arm's, as it holds what this hand reaches for. The hand follows with the joints left, and stops where they cannot keep it on the path. It is what an approach, a lift or a pour asks for, where a planned path would take any route.

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
        hold: vec![],
        holding: None,
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
            near: None,
        };
        let descent = Trajectory::with_hand_speed(&robot, &result.paths[0], &[speed])?;
        println!("lowered in {:.2} s", descent.duration());
    }
    Ok(())
}
```

- **Steps.** The hand moves at most `step` (5 mm) or `rotation_step` (0.05 rad) between configurations, each solved by damped least squares from the one before, in at most `iterations` (30) and to a tenth of the precision. Where the joints' straight motion between two configurations would take the hand more than `precision` (1 mm) or `rotation_precision` (0.01 rad) off the path at its middle, the step is halved, up to 8 times.
- **The motion.** A cubic B-spline passes through every configuration and rests at both ends. Where it would stray from the path by more than the precision, as at a sharp turn, or pass a joint limit, as where a joint arrives at its limit and stays, it stops at the configuration there instead. It is checked for collision at the planner's resolution, honouring [allowed contacts](perception.md#allowed-contacts), so an approach can end with the fingers around what they grasp.
- **How far it got.** A motion that cannot finish goes as far as it can. `fraction` says how much of the path it covers, by its steps, and `stopped` why it ended there: `Unreachable` (beyond reach or joint limits, at a singularity IK cannot step through, or where the joints left by `hold` cannot keep the hand on the path), `Collision`, or `Jump` (configurations either side, but no motion between them keeps the hand on the path, as when IK leaps to another branch). Near a singularity, IK given more `iterations` may instead carry the arm through, swinging the wrist on the way while the hand keeps to the path. Each names the pose it stopped before. `paths` holds the motion that does run, ready for timing.
- **Timing.** `Trajectory::with_hand_speed` times the motion as fast as the joints' limits allow with the hand under caps on its speed and on how fast it turns ([below](#executable-trajectories)); `Trajectory::try_new` times it by the joints alone.
- **Speed.** Paths are searches, run in parallel on the device's threads like plans. One 10 cm move takes 0.042 ms on one of the Ryzen's threads and 0.035 ms on the M4 Pro's; 64 take 0.27 ms on 32 threads and 0.37 ms on 14 ([Benchmarks](benchmarks.md#tool-paths)).

## Picking up and setting down

`pick(&device, &worlds, &[Pick], &TaskOptions)` and `place(&device, &worlds, &[Place], &TaskOptions)` each do a whole task in one call: three motions from rest to rest, or the last two alone for a task that leaves its transit out. The `transit` is a planned path from the start to where the hand begins to move in, the `approach` a straight move in (to the grasp, or down to where the object goes), and the `depart` a straight move away (lifting the object, or backing off from it). Close or open the gripper between the approach and the departure. Every motion is checked with the fingers where the start has them, so open them first.

When the world represents a movable object in pieces, give `Pick::object` the index of the part the fingers may touch. Express each grasp in that part's frame. In `Pick::body`, give the shape of the entire body in world coordinates and the indices of the other pieces that move with it. They still block the hand on approach. Once the gripper closes, the returned `Held` uses the body's frame and the departure checks that shape against the remaining world. An enclosing box or cylinder for a hollow body also fills its interior during the lift. Call `pick.lift_world(&departure_world)` to reproduce the departure checks with that hold attached. A failure's contacts name obstacles by their indices before the pieces were removed.

For measured grasps, `Grasp::closes_on` checks that an entire ball fits between the opposing
pads at the selected opening and clears the hand. Pass the hand spheres at the opening used by
`grasps_on`. The [miru guide](../miru/README.md) shows how to project a candidate into its own
world and retain the remaining obstacles after a successful lift.

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

  Every candidate that passes every step comes back, quickest first (`TaskPlan::candidate` says which). A pick with hundreds of grasps, as `grasps_on` gives, spends nearly all its time on the IK of grasps it will not use: `Pick::enough` says how many plans will do, and the grasps are then tried in their order, 32 and then twice as many each time, until that many have got through. Its plans are the first that many in the grasps' order to get through, as trying every grasp gives them, so put the grasps worth having first; a pick none gets through has tried every one and says where each stopped. So a place batched over each surviving grasp's `end` and `held` finds the quickest pick and place together, in two calls. A place with many poses, as every spot of a table, does likewise with `Place::enough`: its poses are tried in their order, 32 and then twice as many each time, and its plans are the first that many to get through; where none does, every pose has been tried, a few at a time, which takes about a fifth longer than trying them at once. The straight moves keep the hand under `TaskOptions::tool_speed` (10 cm/s), all along or, with `tool_speed_near`, only within that distance of the grasp or release, the rest of each move as fast as the joints allow; and they must make `min_fraction` of their length (all of it): less lets an obstacle shorten a move in or away rather than rule the candidate out, while a move stopped short by anything else (out of reach, a jump) still rules it out.
- **What may touch.** At the grasp and moving in, the fingers (`Robot::finger_links`, or the whole hand where it has no links with collision spheres below its palm) may touch the object; the palm may not, so a grasp set too deep fails. The held object may touch what it rests on as it leaves (`Pick::support`) and what it is set on as it arrives (`Place::support`). Backing off, the fingers may touch the object where it now stands. Everything else collides.
- **The carried object's sphere fit.** `TaskOptions::held_spheres` controls the cover used during lift and during a turn that carries the object. For a narrow gap, increase the budget and decrease the tolerance. Keeping `max_overhang: None` leaves the spheres untrimmed. Pass the pick's returned `Held` to the follower and to `place` so they use the same cover.
- **A world per step.** A task's `world` is the one the hand grasps or lets go in, and each other step's unless that names its own: `transit_world` for the start and the planned path, `depart_world` for the move away. Each holds the task's obstacles, in the same order, shaped or touched (`World::touch`) otherwise, so a hand that may lean on the bin an object stands in at the grasp plans its way there touching nothing, or around the bin as measured where the grasp sees it as a box. Several ways of allowing a task are several tasks of one call, and the first of them with a plan is the one to run.
- **Moves without the way to them.** A task with `transit: false` runs steps 2 to 4 alone and keeps every candidate that passes them: its plans hold the two straight moves (`TaskPlan::transit` is `None`, `TaskPlan::motions` gives what there is in order), `TaskPlan::begin` is where a way to them must end and `TaskPlan::end` where the next begins. Its start is not checked for collision, and its constraints, which are the transit's, are not applied: the start only says which solution of a grasp or release is nearest and where the joints other hands move wait (a place's `idle` still says the latter where it is given). So a program of many tasks need not plan them one after another: every pick's moves solve in one call and every place's in a second (each holding what its pick's grasp holds, in the world its step of the program leaves), and every way between them is one `plan`, the carries `holding` their objects. `join` then runs each way on into the move in after it, a carry with its `holding`:

  ```rust,no_run
  # use batchplan::{Device, Error, JoinOptions, Pick, PlanOptions, PlanProblem, Sequence, TaskOptions, Worlds};
  # fn run(device: &Device, worlds: &Worlds, picks: Vec<Pick>) -> Result<(), Error> {
  // Every pick's moves in one call, none planning its way from its start.
  let picks: Vec<Pick> = picks.into_iter().map(|p| Pick { transit: false, ..p }).collect();
  let moves = batchplan::pick(device, worlds, &picks, &TaskOptions::default())?;
  // The way to each task's quickest moves, in one more.
  let mut quickest = vec![];
  let mut ways = vec![];
  for (task, outcome) in picks.iter().zip(&moves.outcomes) {
      if let Ok(plans) = outcome {
          let world = task.transit_world.unwrap_or(task.world);
          ways.push(PlanProblem::new(world, task.start.clone(), plans[0].begin()));
          quickest.push(&plans[0]);
      }
  }
  let planned = batchplan::plan(device, worlds, &ways, &PlanOptions::default())?;
  // Each way found runs on into its move in without stopping, where that is sooner.
  let sequences: Vec<Sequence> = planned
      .solved()
      .map(|way| Sequence {
          world: way.problem.world,
          paths: vec![way.solution.to_vec(), quickest[way.index].approach.control_points.clone()],
          blends: vec![0.05],
          constraints: vec![],
          holding: None,
      })
      .collect();
  let joined = batchplan::join(device, worlds, &sequences, &JoinOptions::default())?;
  # let _ = joined;
  # Ok(())
  # }
  ```
- **What the hand holds.** A pick's plan holds a `Held` (`TaskPlan::held`): the object in its own frame and the grasp. `place` takes it and attaches the object itself, to the device without it, and so do `plan`, `solve_ik` and `follow` for a problem's `holding` ([holding objects](robots.md#holding-objects)); `Held::attachment(&robot)` gives the `AttachedObject` for a device that holds it, on the robot relocked at `held.locked` first where the hold sets the fingers ([`Robot::attach`](robots.md)). A place's plan gives the object where it now stands (`TaskPlan::placed`), to add to the world, and `TaskPlan::end` is where the next task starts. `World::remove` takes the object out of its world for the place, renumbering the touches.
- **Where the fingers are.** A robot loaded with its fingers locked models them where it locked them. A pick's grasps may each keep them narrower (`Pick::locked`, per grasp the fingers' joint values, as `Robot::relock` takes them; empty for every grasp as loaded), as for a rim between walls that the open fingers would hit: one pick takes grasps at several openings and plans by any of them, a relocked robot for each opening, which builds nothing. Each plan's `held` starts with its grasp's values, and its transit and move in are run with the fingers there: follow them on that robot too (`device.with_robot(&robot.relock(&held.locked)?)?` for a `Follower`'s ticks), since the device for the robot as loaded checks the open fingers and stops where they are in what the narrow ones pass. A hold (`Held::locked`) says where the fingers closed, and every call that holds the object (`place`, `plan`, `follow`, `join`, IK, `handover`) checks the robot so. The collision model must be fitted with the fingers free, as `relock` requires.
- **How far the hand opens.** A place opens the hand to where the robot was loaded with its fingers, and solves where it lets go with them there, so a spot without room for the open fingers is refused at its release. `Place::open` names the values the fingers may open to instead, as `Held::locked` names them, in the order preferred (widest first, say: `GraspOptions::openings()` gives those a grasp with `narrow` may come in at, to map to the fingers' values). Where the hand lets go is then solved with the fingers where they hold the object, and each pose lets go at the first opening with which the hand is clear there and on its move away, the object standing where it was set and the fingers free to touch it; the plan says which (`TaskPlan::opening`). Command that opening, and follow the move away on the robot relocked at it. The openings share the place's IK, its move in and its transit: each one tried costs a move away. Joints an opening does not name are as loaded, whatever the hold has them at. The next opening is tried only where the fingers themselves stopped the one before (out of reach, or the arm or palm in something, is the same at each), and a pose none clears stops at its departure, with what the last one tried met. After a grasp that missed, likewise: open only as far as the fingers came in, and plan or follow the way back on the robot relocked there.
- **Other arms.** A pick or place moves its hand's joints alone; the joints only other hands move wait where the start has them. A place's transit can take them elsewhere at the same time (`Place::idle`), as another arm going home while this one carries the object, and they wait there through the moves in and away: IK at the release starts from there.
- **Proposals.** `grasps(&obstacle, &GraspOptions)` proposes grasps for a box, ball, cylinder or capsule, in its frame: the hand facing each side, closing across what fits between its fingers (`opening`, 8 cm), `depth` (2 cm) in from the surface. They suit a parallel gripper whose IK frame lies between its fingertips, its z axis pointing out of the hand and its y axis along the closing direction, as the Panda's `ee_link`. Where no grasp fits, `grasps` says why: the object is wider than the fingers open every way, or a floor rules out every grasp. `place_poses(&obstacle, pose, turns)` gives an object's symmetric turns at one spot: 2 for a box (4 with a square footprint), `turns` for a round object.
- **Grasps of what was seen.** `grasps_on(&device, hand, &[Surface], &GraspOptions)` proposes grasps for any shape from points on its surfaces and the normals out of them, as cameras see it. A grasp closes on two patches of surface (each the mean of the points within `contact`, 5 mm) that face away from each other, no farther apart than `opening`, with the line between them within `atan(friction)` of each patch's normal (`friction` 0.5: 27°): the condition under which two fingers that press with friction hold, which Dex-Net tests on meshes. A bowl's wall, a handle and a box's sides are found alike, with no shape assumed.
  For a hand that opens wider than its graspable width, use
  `grasps_on_with_capacity(&device, hand, &surfaces, &options, capacity)`. Set
  `options.opening` to the gap from `robot.parallel_jaw_opening(hand, loaded_gap, capacity,
  closed_gap, &closed_locks)?`. Pass the actual open and closed pad measurements and the joint
  values commanded closed. A supported hand returns `Some(gap)`. Both measured and inferred
  contact pairs must fit within `capacity`. With `narrow` enabled, a 70 mm loaded gap and
  60 mm capacity yield a 70 mm full command and partial widths of 56.25, 52.5, 48.75,
  45 mm and so on. Pass the loaded gap to `Observed::grip_worlds` in miru. Lock the joints
  at the returned physical opening for each candidate. Narrowing assumes the finger spheres
  translate equally and oppositely along the closing line. On `None`, a full-open preshape
  can still be checked against observed points with `narrow: false`. A measured closing region
  for this motion requires a model of those jaws’ closure.
  - **Both patches must have been seen.** A surface with nothing seen behind it gives no grasp, since where its other side lies is a guess: one camera's view of a box on a table gives none, and two views give those across what the cameras see between them. Patches nearer than 2.5 mm are one surface's noise.
  - **Closing on a measured point.** Set `close_on` to a point in the surface's frame when the
    fingers must close at a particular reading, such as a pixel on one bar among several.
    The ordinary grasps keep their order. Additional approaches rotate about nearby contact
    pairs in smaller steps until the point moves at most 2 mm per angular step. Finally, for
    each grasp that encloses the point, the hand backs out in 2 mm steps while the fingers
    retain the requested overlap on the patches. Each insertion checks the swept approach,
    floor and opening again and must still enclose the point. These alternatives can clear
    the palm when the deepest insertion collides. A failed search can visit more candidates. Apply `Grasp::closes_on` to the ordinary grasps too when selecting a motion.
    `near` controls the order of the ordinary contact pairs independently.
  - **Walls seen from one side** (`shell`, off by default). A bowl seen only from above shows its rim and its inner wall; its outer wall flares out of sight under the rim. `GraspOptions { shell: Some(0.015), .. }` grasps such a wall on evidence rather than a guess: a ray that reached something farther crossed empty space, and no ray goes into a solid. From a patch whose other side no camera saw, the line in along its normal is followed, up to `shell` and never past `opening`, to the first point every ray around it went past by more than 2.5 mm. The wall has ended there: the second contact goes there, the other side taken parallel to the side seen, so `width()` is the most the wall can be, not what it is, and the `Grasp` says so (`one_sided`): judge whether such a grasp holds by something other than its width. Where the wall leans, as a flared bowl's does, it is closed on level too if the level line through the patch is inside the patch's friction cone (`friction`: a wall leaning 27° needs 0.51, and rubber pads at 2 have it): a hand closing along the normal leans as the wall does and puts a finger down beside it, where one closing level comes straight down across the rim. A patch whose line first meets a surface of the object seen facing along it keeps to the seen pair, and inside a solid no ray goes: a box seen from one side is closed on only across all of it, where the cameras saw past its far side. Behind a flared wall the space the rays went past starts beyond the wall's other side: LIBERO's bowl, its wall 3 mm thick, seen from 40 cm above, is grasped [5.5 to 11.5 mm wide at its rim](benchmarks.md#grasps-of-a-surface), so 1.5 cm suits it and 4 mm finds nothing. The evidence is each point's `seen_from`, which `from_depth` and `from_points` record: the points seen from exactly one place make one view, of the nearest reading in each direction, so looks joined from one place are one view, and a cloud from a moving sensor, each point seen from a place of its own, shows nothing. `from_points` takes its `seen_from` as one camera's place, the space between it and every point empty: give a cloud fused from several views no shell. Grasps of two seen patches come first within the ordinary grasps and again within the intermediate angles requested by `close_on`. Shallower insertions follow in parent grasp order. Dex-Net's depth-image sampler grasps from one overhead image by pairs of depth edges whose gradients point away from each other, as across a rim, with the gripper coming down the camera's axis; GPD's candidates need points between the fingers and none in them, and it computes what no camera saw only for its classifier. This takes from Dex-Net that a depth edge bounds a wall, and from GPD the camera each point was seen from.
  - **Surfaces.** `Surface::from_depth(&device, &image, &object, robot_mask, radius)` reads one depth image, on the device's threads: a point for every reading off the robot, `object` saying per pixel which show the object to grasp (a segmentation; empty for all), each of those with the normal of the plane through the readings within `radius` of it (5 mm suits), toward the camera, or none where they lie on one line or near no plane. `surface.join(other)` adds another view, and `Surface::from_points(&points, seen_from, radius)` takes a cloud seen from one place. Each point records where it was seen from (`seen_from`, the camera's position; infinite where a joined surface did not say). A `Surface` is plain data (`points`, `normals`, `object`, `seen_from`), so points with normals from elsewhere, as sampled from a mesh, go in directly, with `seen_from` empty where no camera saw them. Points that are not the object's stay in: nothing else may lie between a grasp's patches, and the hand is kept clear of them.
  - **The hand** is the robot's own, with its fingers as the locked joints hold them. Its collision spheres at the grasp, each swept back `approach` (10 cm, a pick's) along the hand's z axis, lie clear of every point: the way in is checked because a sheet of points can pass between two spheres of a finger, and the one that came through it then finds it. Only the fingers may be in what they close on, the object's points between the two patches. What no camera saw, the check cannot know: behind a wall seen from one side a finger is checked against the points seen alone, so it is known clear only where the rays went past, not in the shadow under a flared rim. A hand that is not clear is drawn back while `overlap` of the fingers stays on the patches, and `floor` holds the frame and the open fingertips above a height. To come in with the fingers part open, as past a neighboring wall, set `narrow` (below), or pass `device.with_robot(&robot.relock(&fingers)?)?` and that `opening` for one opening of your choosing. `depth`, `tip` and `along` are for `grasps` and not read here: the reach is the hand's own.
  - **Fingers that come in narrower** (`narrow`, off by default). Open fingers beside a drawer's wall or a neighbor are in it, though the grasp needs a fraction of the opening. `GraspOptions { narrow: true, .. }` takes each finger to slide along the hand's y axis, as a parallel gripper's do, checks every grasp at sixteenths of `opening`, from open down to the fingers on the patches, and gives each `Grasp` the `opening` it comes in with:
    - **Farthest from what is not the object's**, the floor included. A planner lets the fingers touch what they grasp and nothing else, and knows what stands around less finely than the points do (a distance grid errs toward collision), so room on the object's side buys nothing there. With something within the fingers' own travel of the open fingers, in their way or past them, that is the narrowest opening the hand is clear at.
    - **Of openings as far from all that, the one with the most room** from the object's other sides: as many steps from the first at which the hand would be in anything farther apart as from the first nearer, the patches at the last. A finger inside a bowl stands midway between its walls.
    - **Of those, the widest.** A box with nothing near is grasped at the whole `opening`, as without the option.

    A grasp the open hand has stays where it is, drawn back as the open hand is, so `narrow` moves no grasp found without it and adds those the open hand has not: there the hand comes in as deep as it is clear narrower. The floor is kept at each opening, the lower finger standing higher as they near. Lock the fingers there for the pick: `Pick::locked` takes, per grasp, the finger joints' values at `grasp.opening`, which is the robot's to say (the Panda's two fingers each stand at half of it), and a call's grasps share at most sixteen openings. A grasp that is refused costs about what the open hand's check does, and one that is found is checked across each finger's whole slide: a fifth more for the Panda's hand of 22 spheres on a bowl beside a wall, and more for a hand of many ([numbers](benchmarks.md#grasps-of-a-surface)).
  - **What comes back.** Each pair of patches from `turns` directions around its line, each both ways round, as a `Grasp`: the hand's `pose`, the two `contacts` (`width()` apart) and the `opening` the fingers come in with, in the surface's frame. For `Pick::grasps`, which are in the picked obstacle's frame, bring the pose there: `obstacle.pose().inverse().mul_pose(g.pose)`. The order suits trying the first few: every pair from above (where its line is not upright), then every pair a turn to either side, and so on round; within a turn, the pairs within half a friction cone of square before the rest, each the farthest from those before it, so they spread over the object.
  - **Cost.** The surfaces of a call run in parallel on the device's threads, and a surface's grasps are the same alone, among others and on either device ([numbers](benchmarks.md#grasps-of-a-surface)).
- **Rests on what was seen.** `rests_on(&device, &[RestProblem], &[Surface], &RestOptions)` finds where a held object can be let go on a measured support and stay: on a plate, a cabinet's top, the top of a rack. Each `RestProblem` gives the object in its own frame (a box, ball, cylinder or capsule), how it is turned as it is set down (`rotation`: upright, on its side, flat on a slope) and which surface it is set on.
  - **The support** is the surface's `object` points (every point, where it flags none); the rest are kept `clearance` (1 cm) away, and what is under them is taken as solid too: nothing rests under a point that is not the support's, however high it is, so a shelf with another over it has rests only where the one above was not seen or is flagged with it. It is read from above, as the highest point in each cell `touching` (2 mm) wide (a support of more than four million cells is an input error that asks for wider ones): what is under a cell's highest point is taken as solid, a point whose normal faces down holds nothing up, and a cell no camera saw is empty space. Flag everything the object may lie on: a fixture measured in pieces has rests across its pieces only where all of them are flagged. `seen_from_above_holds` (off by default) lets a point whose normal faces down hold where the camera that saw it was above it (`Surface::seen_from`): ask for it for a top read by a camera above it with noise, whose normals fitted over a few millimeters can face down in places, where they block rests on that top. What it gives up is a face of the support that leans out over empty space, seen from above with what is over it out of view: it is taken as a top at its own height. `falls_into` reads the support so too.
  - **How it is set down.** Over each spot `spacing` (1 cm) apart across the support, at each of `turns` (8) turns of `rotation` about the vertical, the object comes straight down until it touches. It rests there if what it touches holds it: the contacts carry its weight, pushing and never pulling, with `friction` (0.5) where the weight presses, and still do with the weight tipped `tipping` (5°) any way, so a tall object needs its middle farther in from an edge than a flat one, and a slope holds one only that far short of `atan(friction)` unless a stop below it does.
  - **Looking closer** (`look_closer`, off by default). A rest narrower than the spots are apart, as an upright bottle's against the lip at the foot of a sloping rack, which stands in a strip a few millimeters deep, lies within a spacing of a spot where the object nearly rests: held there as set down, but tipping too easily. With `look_closer`, of those spots at a turn, the one nearest `near`, or without it the middle of the support, has the spots a cell (`touching`) apart within `spacing` of it tried too, and what rests there comes in the order the others do: with `near` at every turn, and without it only where no spot rests at any turn, so that a search that finds rests costs what it does without the option and one that finds none at most as many spots more as `near` adds. Ask for it where such a rest is wanted near a point, or where the spots find nothing else; without it such a strip is found only where a row of the spots happens to lie over it, or within a spacing of `near`. One place is looked at, not every place: such a strip is found near that spot and not along its length, so a caller that wants it elsewhere points there.
  - **As it is set down.** The object is not tipped or slid to where it would come to rest. Set down level on a slope it touches along an edge and has no rest: give the rotation that lies flat on the slope. One thing moves: a can or a capsule laid on its side that is not held where it is set down rolls, across its axis and no farther than half the spots' spacing and `touching` more, to where it lies lowest, and rests there if that holds it. So it is found in the cradle between two bars, in a rack's groove or against a lip at the foot of a slope from spots a centimeter apart, each place once a turn. A can stood on its end is not rolled, however it leans. On a flat support it rolls and has no rest, however noisily the support was read, as a ball has none; a ball rests on a rim or on posts that hold it from every side, where a spot is over them. Sides shallower than `touching` tells from flat are no cradle. Give the object its true shape: a round bottle given as a box has rests on a ridge the bottle would roll off. The support under a patch of contacts is taken as a plane, so a flat bottom over the top of one round bar, within `touching` of it, is read as standing on a band of the bar: where a spot lands there, as spots 2 mm apart do, a tray is found balanced over one bar, which a round bar does not hold. A test of rocking, the bar's radius against how high the object's middle is, would remedy it.
  - **Where you would have it.** A rest can be narrower than the spots are apart with nothing beside it that nearly holds: a bowl given as a flat cylinder wider than a dished plate's floor stands evenly on the plate's raised ring only within a few millimeters of the middle, and set down a centimeter off it touches the ring on one side and is not held at all, so spots a centimeter apart find or miss the middle by where the plate was measured, and looking closer does not replace a point to look near there. `RestProblem::near` is a point you would have the object set down at, as the middle of the plate or where an agent pointed: within `spacing` of it, seen from above, and over the support, the object is also set down a cell of the support apart (`touching`, 2 mm): as many more spots a turn as cells lie that near the point, 77 at the defaults and more by the square of `spacing` over `touching` (more than a million is an input error). With `look_closer`, it also says which place where the object nearly rests is looked at closer: the one nearest it. Leave it out for a support with no place in mind, as a whole table, where the spots find every place already. It looks closer and judges no differently: near a point on a plate's rim it finds the rests a flat cylinder has there, where a bowl on a small foot would not stay, so give the object the shape it stands on.
  - **What comes back.** Per problem, each `Rest`: the object's `pose` in the surface's frame, the `support` it touches (the corners of its hull seen from above) and how far the weight can be `tipping` before the contacts let go. The poses go in `Place::poses`, which checks the hand and the way there. A rest comes at each of the `turns`: where the shape is set down the same at two of them, as a box at a half turn, an upright can at any, or a bottle laid down end for end, the two rest in the same places and are two poses for the hand. `place_poses` adds the copies about a lying object's own axis, where a grasp needs them. They come every spot at the first turn, then the next, and within a turn the steadiest first, then each the farthest from those before it, so the first few spread over the support; past the first 1,024 of a turn, in the order of their spots.
  - **Why none.** `rest_failures(&device, &problems, &surfaces, &rests, &o)` says, for each problem `rests_on` found no rest for, how near the search came (`RestFailure`), the furthest stage any spot and turn reached: `Blocked`, a rest but for a point in its way (the steadiest such); `Tips`, held but tipping by less than `tipping` (the steadiest, with how far it can tip); `Unheld`, touching the support and not held (the spot nearest `near`, or the support's middle, with the unit mean of the directions the support pushes it along: up on a level support, leaning on a slope); `Unsupported`, nothing of the support under it anywhere. Where it is not held or tips, what would be in its way there even if it held comes too (`blocked`): a `Blocking` point that is not the support's says the flags leave out part of what the object would lie against, one that is says a part of the support seen only from below. Pass it the options the rests were found with: it searches the problems without rests again as `rests_on` did with them, looking closer or not and reading the support as they say, and where the furthest stage is `Tips` or `Blocked` judges each place of that stage once more for the steadiest, so it costs one to two more searches of those (on the example's table a box asked to tip 40 degrees, which tips everywhere, takes 63 ms to search and 126 ms to say why), and runs only when asked:

    ```rust,no_run
    # use batchplan::{Device, Error, RestFailure, RestOptions, RestProblem, Surface};
    # fn run(device: &Device, problems: &[RestProblem], surfaces: &[Surface]) -> Result<(), Error> {
    let o = RestOptions::default();
    let rests = batchplan::rests_on(device, problems, surfaces, &o)?;
    let why = batchplan::rest_failures(device, problems, surfaces, &rests, &o)?;
    for (k, failure) in why.iter().enumerate() {
        match failure {
            None => {}
            Some(RestFailure::Unheld { blocked: Some(by), .. }) if !by.support => {
                println!("problem {k}: flag what it would lie against, as {}", by.point);
            }
            Some(RestFailure::Tips { tipping, .. }) => println!("problem {k}: tips at {tipping} rad"),
            Some(other) => println!("problem {k}: {other:?}"),
        }
    }
    # Ok(())
    # }
    ```

  - **Cost.** The problems of a call run in parallel on the device's threads, each surface read once however many problems name it, and a problem's rests are the same alone, among others and on either device ([numbers](benchmarks.md#rests-on-a-surface)).
- **Letting go into what was seen.** A can into a basket, a bowl into an open drawer: the object does not rest where it is let go, it falls and comes to lie somewhere inside, and what matters is that it cannot leave. `falls_into(&device, &[RestProblem], &[Surface], &RestOptions)` takes the same problems, surfaces and options as `rests_on` and gives, per problem, each `Fall`: a `pose` to let go at, for `Place::poses`, with the object's lowest point at least `clearance` over the hollow's rim (that much for an object standing level, unless something nearer stands higher) and the object that far clear of all around it, and how deep under the rim it lands (`sunk`). `Place::lower` as for any place: the move in ends at that pose, above the rim.
  - **What holds it in.** Over each spot, at each turn, the object lands on the support or on anything else there, as what is already in the basket. It lands as it is: the clearance is kept where it is let go, not where it lands, so a box that goes through a bowl's opening, and grown by the clearance would not, falls in. A hollow is where it would have to rise to get out, found as the hollows of a terrain are, by flooding the heights from the edge inward, lowest first (Barnes et al.'s Priority-Flood); under gravity that is what cages it (Mahler et al.'s energy-bounded caging). It is held in where the rim is higher over where it lands than its own middle is over its bottom. It keeps its turn as it moves: one that could turn through a gap it cannot pass as it is let go is taken as held.
  - **Gaps.** Whether it can leave is asked of the object worn down by a level disc of half the diagonal between spots (7 mm at the defaults), so no way out between spots is missed: a gap it fits through holds nothing in, and a gap up to that much narrower than it may be taken as one it passes too. Worn down so, an object must keep at least a cell of the support (`touching`) of itself every way, or it could slip between the points seen of a solid wall: one thinner, as a ball under about 1.6 cm across or a standing pencil, has no falls; space the spots nearer for it.
  - **What it needs.** A floor, or a wall sloping in under the object, seen as deep as its middle is over its bottom somewhere in the hollow: a floor no camera saw is no floor, and an upright wall is read as its top. Where some of the floor was seen that deep, the object is let go over the patches no camera saw too. A tray, a plate or a table holds nothing in, and gives nothing.
  - **Looking closer** (`look_closer`, off by default). A container the object barely fits holds it in only nearer one place than the spots lie apart: a box in a bowl a little wider than its diagonal lands as deep under the rim as its middle is over its bottom only within millimeters of the bowl's axis, and spots a centimeter apart may all lie 7 mm from it. With `look_closer`, falls look closer as rests do: the spots a cell (`touching`) apart within `spacing` of `near` are tried at every turn, or without it those of the spot of a hollow where the object lands the deepest under the rim, only where no spot holds it in at any turn, so that a search that finds falls tries no spot more than without the option (it keeps only each spot's rim and whether it holds the object in, at each kind of turn, for the look it does not take) and one that finds none at most as many spots more a turn as `near` adds to rests. Each is held in as a spot is, against the rim of the spot whose cell it lies in (the object there holds that spot's object worn down by half the cell's diagonal, so it cannot leave lower), on something seen. Falls found there come after the spots' of their turn, the nearest the middle of where they are held in first.
  - **Into or onto.** A basket has rests on its rim and falls inside it; which is wanted is the task's (LIBERO's goals say `in` or `on`), as the target is, so call the one the task asks for. They come at every turn, those with the most room around them first and of those the nearest the middle of where it is held in. Its problems are checked as `rests_on` checks them, though falls read only `object`, `surface`, `rotation` and, to look closer, `near`, and of the options `turns`, `spacing`, `touching`, `clearance`, `look_closer` and `seen_from_above_holds`.
- **Near a support.** A small object on a table puts the fingertips in the table at the full depth. Give `GraspOptions::tip` the fingers' reach past the frame (`robot.tip(hand)`) and `floor` the height they keep above, such as the table's top and a few millimeters, and pass the obstacle as it stands in the world: grasps from above then hold the object as deep as keeps the fingertips over the floor, the frame above it where the object is flatter than the fingertips' reach and the floor's clearance, and grasps that come from below are left out. The floor holds the open fingers too, `opening` apart: tilted about a bar lying over a support, as a drawer's handle, one finger swings down, so the grasp holds the bar as far out as keeps it over the floor, and a grasp from the side that closes upright on an object standing on the table, its lower finger below the table's top, is left out. Leave room in the floor for the fingers' thickness. Every grasp from above keeps at least `overlap` of the fingers in the object (1 cm, as MoveIt Grasps' minimum depth for the Panda), so one whose top is less than that over the floor has no grasp from above, and with `tip` under 1 cm, the default 0 among them, grasps the floor alone would allow can be left out too.
- **Why it failed.** When no candidate gets through, the task's `TaskFailure` holds each candidate's step and reason (`CandidateStop`: a pose out of reach, a collision or a jump, or no path), and says it in a sentence: "none of 24 candidates got through: 20 at the grasp or release (1 out of reach, 19 colliding), 4 moving away (4 colliding)". IK that finds nothing is classified by `ik::failures`. A candidate stopped by a collision carries what collides (`CandidateStop::contacts`): at the start, at the IK solution with the most clearance, holding the object where a place lets go, or at the straight move's next step. Its obstacles are numbered as in the task's world, the picked object included; in a place, `world.obstacles.len()` is the object where it is set down, and in a handover, the object where it changes hands.
- **Speed.** On the Ryzen AI Max+ 395 (32 threads), picking a box by its 24 grasps takes 6.8 ms on the CPU device, and 64 such picks at once 323 ms; on the Radeon 8060S, 10 ms and 107 ms. Setting a box down at either of its turns takes 2.4 ms on the CPU, and 16 at once 12.1 ms. [Benchmarks](benchmarks.md#picking-and-placing) has both machines and every batch size.
- **Handing over.** `handover(&device, &worlds, &[Handover], &TaskOptions)` passes what one hand holds to another hand of the same robot, as a pick leaves it: `Handover::new(&robot, world, start, held, to)` with the giver's `Held`. Its candidates are each meeting (the object's pose as it changes hands) with each of the taker's grasps, and run through the same steps:
  1. The start, clear holding the object.
  2. IK for both hands at once at the meeting, the object standing there as an obstacle that the giver's hand and the taker's fingers may touch and no other link.
  3. The taker's move in, its last `approach` run backward from its grasp, with the giver's joints held still (`ToolPath::hold`): on a robot whose arms share a torso, the other arm does not move.
  4. The giver's move away, `retreat` back along its z axis, with the taker's held still.
  5. The way from the start to the meeting, the object in the giver's hand.

  Run the transit and the approach, close the taker's gripper, open the giver's, then run the departure. Each plan names its `meeting` and the taker's `held`. `handover_poses(&robot, start, &held, to)` proposes meetings where both arms reach, in front of and halfway between their shoulders, the object at its tilt in the giver's hand with its held end toward the giver; `handover_grasps(&held)` turns the giver's grasp to the object's other end, from the same side and from the other. Each hand needs its own end: pick the object toward one with `GraspOptions::along` (0.75 holds it three quarters of the way to an end). A bar too short for both hands leaves them no room, and the failure names the links of both arms that collide.

## Turning a thing about an axis

`turn(&device, &worlds, &[Turn], &TaskOptions)` turns something about a line in one call: a knob, a cap, a lever, a valve's wheel, a door's handle. It runs as a pick does, a candidate per grasp, with a turn between the move in and the move away: the hand follows its grasp carried about a `Screw` by the task's angle, the fingers closed on the object, then lets go and backs off.

```rust,no_run
use batchplan::glam::Vec3;
use batchplan::{
    CollisionModel, Device, Error, GraspOptions, Obstacle, Pose, Robot, RobotOptions, Screw,
    TaskOptions, Turn, World,
};

fn main() -> Result<(), Error> {
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
    // A lever on a post, its hinge upright through the post, swung 60 degrees.
    let table = Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let hinge = Vec3::new(0.45, -0.2, 0.15);
    let post = Obstacle::cylinder(Vec3::new(hinge.x, hinge.y, 0.07), 0.015, 0.07);
    let arm = Obstacle::cuboid(
        hinge + Vec3::new(0.06, 0.0, 0.0075),
        Vec3::new(0.075, 0.01, 0.0075),
    );
    let o = GraspOptions {
        tip: robot.tip(0),
        ..Default::default()
    };
    let grasps = batchplan::grasps(&arm.placed(Pose::IDENTITY), &o)?;
    // The arm rests on the post as it swings: what it may touch.
    let turn = Turn {
        screw: Some(Screw::about(hinge, Vec3::Z)),
        support: vec![1],
        ..Turn::new(0, robot.default_q(), 2, grasps, 60f32.to_radians())
    };
    let worlds = device.upload(&[World::new(vec![table, post, arm])])?;
    match batchplan::turn(&device, &worlds, &[turn], &TaskOptions::default())?.outcomes.remove(0) {
        // Run the transit and the approach, close the gripper, run the turn, open it and run the
        // departure; the lever then stands where `placed` has it.
        Ok(plans) => println!("{:.1} s, the lever at {:?}", plans[0].duration(), plans[0].placed),
        Err(why) => println!("no turn: {why}"),
    }
    Ok(())
}
```

- **The line.** `Screw { point, direction, pitch }` is a line in the world's frame and how far what turns about it moves along it for each radian (meters): 0 for a knob, a wheel or a lever; for a cap on a thread of 5 mm a turn, 0.005 / 2π, which turned right-handed about a `direction` up (anticlockwise seen from above) rises as it unscrews. `Screw::about(point, direction)` has no pitch. A task without one (`Turn::screw` `None`) turns about the hand's own z axis through its IK frame at the grasp: a turn in place, as a knob gripped on its axis turns. `screw.carry(pose, angle)` carries a pose about it, and `screw.path(pose, angle, precision)` gives the waypoints a `ToolPath` follows along the arc or helix, as few as keep the straight moves between them within `precision` of it: a frame `r` from the line turns by at most `4 asin(√(precision / 2r))` between two (the chord's sag), and never by more than a quarter turn, so that the shortest rotation between them is the screw's. A frame on the line needs only the quarter turns.
- **Steps.** Each step is a batch for every candidate of the call:
  1. The start, checked for collision.
  2. IK at the grasp, the fingers free to touch the object: every configuration it finds, nearest the start first, not only the nearest.
  3. The turn, followed from each of those configurations, the hand holding the object as the grasp does, the hand kept within the tool path's precision of its true arc: half of it for how far the straight moves between the screw's waypoints sag, half for how far the hand strays from them. A grasp none of whose configurations makes the whole turn stops here.
  4. The straight move in, run backward from the grasp as a pick's, from each configuration that made the turn.
  5. The straight move away, `approach` back along the hand's z axis from where the turn ends, the object where the turn leaves it.
  6. The planned path from the start to where the move in begins, for the configuration nearest the start that got through steps 3 to 5.

  Run the transit and the approach, close the gripper, run the turn (`TaskPlan::turn`), open the gripper and run the departure, then put `TaskPlan::placed` (the object carried through the angle) in the world in its place. `TaskOptions::min_fraction` applies to the moves in and away; the turn is made whole or not at all. The hand keeps under `TaskOptions::tool_speed` all along the turn, which it holds the object through. `enough`, `locked`, `transit_world`, `transit` and `constraints` are a pick's.
- **Which configuration.** A grasp is one pose of the hand and many configurations of the arm, and the one nearest the start, which a pick takes, often runs into a joint limit partway through a long turn where another at the same grasp, the wrist over the other way, has the room. So every configuration IK finds at each grasp is followed through the turn, all of a call's in one batch of tool paths, and of those that make it the nearest the start is the plan's. Of the example's knob's 972 grips (`examples/turns.rs`), 799 make its quarter turn from the configuration nearest the start and 834 from some configuration IK finds at the grip on the M4 Pro (797 and 836 on the Ryzen); turned 170 degrees by a grasp across its middle (`tests/turn.rs`), the nearest stops a quarter of the way round with the last joint at its limit, and two of the eleven others IK finds there make it.
- **What may touch.** Through the reach and the move in the fingers (`Robot::finger_links`, or the hand's links where it has none) may touch the object, and nothing else of the robot may. Through the turn the hand holds it, as a pick's lift holds what it picked up: the object is part of the robot, in the task's world without it, so what it sweeps is checked against everything else of the world and against the arm, and it may touch only what `Turn::support` names, what it turns in or on (a lever's mount, a cap's bottle, the panel a knob sits in). Name everything the object touches as it stands: a cap sits on its bottle all the way round, and a turn that does not name the bottle stops before it starts. Where the world holds the part that turns and what it turns in as one obstacle, as a knob a camera could not tell from its panel, set `Turn::stays`: then the obstacle stays where it is, the fingers alone may touch it, and nothing checks the part's sweep. The move away sees the object where the turn leaves it, or where it stood where it stays (`TaskPlan::placed`), the fingers free to touch it; everything else of the world is an obstacle throughout.
- **Why it failed.** `TaskStep::Turn` comes after the reach and before the straight moves, and a candidate stopped there says how far the hand turned before it stopped (`CandidateStop::turned`, radians, signed as the task's angle) from the configuration that turned it farthest, and why: out of reach or at a joint limit (`Unreachable`), colliding (`Collision`, with what collides) or jumping. "none of 2 candidates got through: 2 turning (2 colliding)".
- **Grips for a turn from what cameras saw.** Read each camera with `Surface::from_depth`, the thing's readings flagged as `object` and everything around it kept in the surface, so the hand is kept clear of it; join the views. `grasps_on` with `narrow` on (the fingers come in as open as keeps them off what stands around) and `floor` at the top of what the thing stands on (the dial under a knob's bar, the table under a lever) gives grasps of any shape. Order them with `options.order_for_turn(&mut grasps, Some(screw))`, the greatest moment about the measured axis first (pass `None` for each grasp's own z axis), bring each `pose` into the object's frame (`obstacle.pose().inverse().mul_pose(g.pose)`), lock the fingers at each grasp's `opening` (`Turn::locked`, the robot's finger joints at that opening: the Panda's two each at half of it), and pass them to `turn` with what the thing turns on as its `support` and `enough` for how many plans will do. `GraspOptions::near` puts the grasps nearest a point first within each of the tiers `grasps_on` orders them by, for a caller told where to grip.
  - **The moment** is what the two fingertips can give the thing about the line for each newton each presses with (meters): what turning it asks of the grip, as a grasp is judged by the wrench its task needs. A force `F` at a contact `c` gives `F · (ŝ × (c − q))` about the line through `q` along `ŝ`. Friction lets a fingertip push across the closing line `n` by up to `friction` times its press (Murray, Li and Sastry's point contact with friction, *A Mathematical Introduction to Robotic Manipulation*, 1994, §5.2.1), which gives `friction · |ŝ × (c − q)|` less its part along `n`; the press may sit anywhere on the fingertip's patch, `contact` in radius, which gives `contact · |n × ŝ|`; and a soft fingertip twists about `n` by up to `γ = 2/3 · friction · contact` times its press (their soft-finger contact, the press even over the patch), which gives `γ |n · ŝ|`. The two presses cancel while they are equal, but a parallel gripper's fingers close evenly about its frame, so the hand can put all of the closing force on either finger and push the thing along `n`: up to `2 |n · (ŝ × (m − q))|` more, `m` the grip's middle, less what friction loses on the finger it unloads. The moment is the most these give together, the way round that gives less. A bar 2.5 cm wide gripped across it on its axis (friction 0.5, patches 5 mm) gives 2.25 cm a newton, half of it friction; 3 cm along the bar from its axis, 8.25 cm, the push adding twice the 3 cm. A grip closing along the axis has friction alone, its slide and its twist.
  - **Precedent.** MoveIt Task Constructor's `MoveRelative` takes a twist (`setDirection` with a `TwistStamped`) and its `GenerateGraspPose` samples grasps about a settable axis of the object (`setRotationAxis`, `setAngleDelta`); PickNik's `affordance_primitives` moves along a `ScrewAxis` of an origin, an axis and a pitch (`getTF(delta_theta)`); GPD filters its grasp candidates by approach direction (`filter_approach_direction`); MoveIt's perception tutorial fits a cylinder to a cloud by RANSAC and takes its axis. None of them orders grips by the turn they are to make, or checks many grips' turns, from every configuration of the arm, in one batched call: `turn` does both.
- **Speed.** On the M4 Pro, other work loading it to 11–15, the example's knob takes 457 ms on the CPU device by all 972 grips and 18.5 ms to its first plan (`enough: Some(1)`); its cap, lever and wheel 6 to 14 ms; and 64 knobs in one call, to the first plan of each, 1.0 s. On the Ryzen AI Max+ 395's 32 threads, 390 ms, 16 ms, 6 to 12 ms and 0.83 s. Each machine's GPU takes within an eighth of what its CPU device does: the turns, which are most of it, are followed on the CPU's threads on either device ([Benchmarks](benchmarks.md#turning-things-about-a-line)).

## Drawing a handle along a line

Use `Pull::new(world, start, object, grasps, direction, distance)` for a drawer. The grasps are
in the contact obstacle's frame. The unit direction is in the world frame, and the signed distance
is in meters. Each grip keeps its orientation as it travels that distance. For a door, use
[`turn`](#turning-a-thing-about-an-axis) with the hinge's point and direction.

Give `batchplan::pull` the device, uploaded worlds, a slice of pulls and the task options. Each
pull tries every IK solution at each grasp through the draw, then plans the approach, release
and transit as a turn does. The hand's fingers may touch the indexed obstacle through the
approach, draw and departure. A distance grid can be the contact obstacle for a pull or for a
turn whose obstacle stays. A task that names a grid takes its grasps in that grid obstacle's
frame, including its placement. Give the approach and transit worlds matching obstacle indices.

Run the transit and approach, close the gripper, run `TaskPlan::pull`, open the gripper and run
the departure. The fixture stays at its measured pose throughout planning. The arm is checked
along its path. The moving drawer's swept volume is outside this model. A candidate stopped
during the draw reports the signed meters planned in `CandidateStop::pulled` at
`TaskStep::Pull`, with its reason and contacts. At an execution force stop the controller
measures how far the handle actually travelled before updating the world.

Run `cargo run --release --example turns -- --fixtures` for a drawer and a hinged door grasped
from two rendered depth images.

## Joining motions

`join(&device, &worlds, &sequences, &JoinOptions)` runs consecutive motions without stopping between them, as MoveIt's Pilz planner blends a motion sequence. A `Sequence` names its world, its paths in order, each rest-to-rest as `plan` and `follow` return them and starting where the one before ends, and a blend radius for each join, as each item of a Pilz sequence gives its own.

```rust,no_run
# use batchplan::{Device, Error, HandSpeed, JoinOptions, Near, Sequence, Trajectory, World};
# fn run(device: &Device, transit: Vec<f32>, approach: Vec<f32>) -> Result<(), Error> {
let robot = device.robot();
let grasp = robot.ee_pose(&approach[approach.len() - robot.dof()..], 0).position;
let worlds = device.upload(&[World::default()])?;
let sequence = Sequence {
    world: 0,
    paths: vec![transit, approach],
    blends: vec![0.05],
    constraints: vec![],
    holding: None,
};
let joined = batchplan::join(device, &worlds, &[sequence], &JoinOptions::default())?;
// One path per stretch the robot moves without stopping: here one, unless the join stopped.
// The hand comes as fast as the joints allow and covers the last 6 cm to the grasp at 5 cm/s.
let arriving = HandSpeed {
    hand: 0,
    linear: 0.05,
    angular: f32::INFINITY,
    near: Some(Near { point: grasp, radius: 0.06 }),
};
for path in &joined.paths[0] {
    let trajectory = Trajectory::with_hand_speed(robot, path, &[arriving])?;
    println!("{:.2} s", trajectory.duration());
}
# Ok(())
# }
```

- **The blend.** Near a join, the joined path leaves the motions only while every hand is within the join's radius of where it is at the join (0 stops there), and is the motions themselves elsewhere: a tool path stays on its line outside the blend. Inside it, the path follows a curve that sets out and arrives along both motions without curving there, so jerk-limited timing can follow it at speed.
- **Kept only where it is sooner.** Each join weighs three blends: the widest within the radius, one dropping half as many control points, and the narrowest. Those that make the stretch so far and the next motion finish sooner than with a stop between them are checked for collision, soonest first, and the first that is free is kept. A turn back, or a tight turn at speed, rarely gains, and keeps its stop. `JoinResult::blends` gives each join's radius, `None` where it stops. Joining never makes a sequence take longer.
- **The result.** Per sequence, one path for each stretch the robot moves without stopping, to time with `Trajectory::try_new` or `with_hand_speed`. A stretch of one motion is that motion as given.
- **Holding an object.** A sequence's `holding` (a pick's `Held`) is carried through it: every blend is checked with the object in the hand, as `plan` and `follow` check a problem's, on the device without it. A carry joins into the move in of its place this way.
- **What it gains.** On pick-like sequences, a transit into a straight move in and a straight move out into the next transit, a 5 cm radius finished 5.8% sooner on average on a reBot B601 (15% at most) and 8.4% on an SO-101 (17%), and joining 60 of them took 15 ms on 32 threads ([Benchmarks](benchmarks.md#joining-motions)). Motions that carry on in about the same direction gain more; a sharp turn needs most of a stop's braking anyway.
- **Arriving slowly.** Each stretch has one timing. For the hand to come fast and reach a grasp slowly, time the stretch with a cap that holds near the grasp alone (`HandSpeed::near`, [below](#executable-trajectories)): a cap without it slows the whole stretch.
- **Limits.** `pick` and `place` still stop between their steps; a task without its transit leaves the way to its moves to join with them. Each join times the stretch so far, so joining a long sequence costs more per join than a short one.

## Executable trajectories

`Trajectory::try_new(&robot, path, speed_scale)` keeps the path and adds a time map `s = σ(t)` (itself a uniform cubic B-spline), choosing the faster of two timings:

- **Time-optimal** (TOPP-RA, Pham and Pham 2018): the fastest timing of the path under joint velocity and acceleration limits, plus a speed cap from the jerk limit, smoothed into a B-spline time map. Where the smoothed map overshoots a limit, the profile is solved again going slower there, and the map stretched by what overshoot remains. On planned Panda paths it takes 0.53 times as long as uniform timing on average and never longer (`tests/trajectory.rs`), at about 0.3 ms per trajectory on one of the Ryzen's threads (`examples/timing.rs`).
- **Uniform**: `σ(t) = t / h`, with the smallest `h` that keeps every joint within its limits. Velocity, acceleration and jerk are then B-splines over the control points' differences, so the bound holds exactly along the whole curve.

Either is slowed by `speed_scale` (1.0 is as fast as the limits allow). Limits come from `RobotOptions::joint_limits` for joints it names, then from the robot description when it has them (URDF 1.2 `acceleration` and `jerk`), otherwise from `RobotOptions` (5 rad/s² and 50 rad/s³ by default). `Trajectory::with_hand_speed(&robot, path, &[HandSpeed])` keeps hands under caps too: `linear` on a hand's speed (m/s) and `angular` on how fast it turns (rad/s), where `f32::INFINITY` leaves one uncapped. Both timings keep them, and `check` checks the joints' limits only.

A cap holds along the whole path, or, with `near: Some(Near { point, radius })`, only while the hand's frame is within `radius` of `point` (in the robot's frame): the motion slows as the hand enters that ball and runs as fast as the joints allow outside it. A ball as wide as the move to a grasp is long, around the grasp, covers that move and the first of the move away. Several caps hold together, each where it is, so one stretch from a grasp to a place can be slow at both ends.

- **For control loops**: `trajectory.at(t, &mut state)` writes position, velocity and acceleration without allocating.
- **For datasets**: `trajectory.sample(hz)` returns fixed-rate samples.
- **Many at once**: `time(&device, &paths, speed_scale, &hand_speeds)` times every path of a call in one call on the device's threads, each as `Trajectory::try_new` times it, or under its own list of caps where `hand_speeds` gives one per path (`&[]` caps none), and gives each its trajectory or its error in order.
- **For safety layers**: a `Trajectory` is plain serializable data. `trajectory.check(&robot)` refuses one that is non-finite, does not start and end at rest, leaves a joint range, runs backward, or exceeds a velocity, acceleration or jerk limit.
- **Its ends**: `trajectory.start()` and `trajectory.end()` are where it starts and comes to rest: its first and last control points, or for a stop, where the stop begins and ends along the path.
- **Payloads**: timing knows the joints' limits, not the load. A held object collides as part of the robot ([holding objects](robots.md#holding-objects)), but batchplan has no payload mass or inertia, so a heavy carry is timed as an empty one: slow it with `speed_scale` or the robot's limit scales.
- **For stopping**: `trajectory.stop(&robot, t)` is a controlled stop: the same path from time `t`, braking to rest as soon as an even deceleration within the limits allows, at once or after carrying on as it was going for one span of the stop's time map, whichever stops sooner along the path: while the motion still gathers speed, braking at once would break a jerk limit. It stays on the checked path, so it needs no new collision check; verify it with `check_from` and the state at `t`. On planned Panda reaches it takes 0.6 ms to compute a quarter of the way along (`examples/timing.rs`, an M4 Pro), rests in about two fifths of the time left, and errors where none of its braking maps fits before the path ends: late in short, fast motions, and for some states early in a motion at speed (52 of 960 through the first fifth of 48 planned motions).

### Starting from a moving state

To replan mid-motion, give `PlanProblem::start_motion` the robot's current velocity and acceleration. The spline's first control points then continue that motion, and `Trajectory::moving(&robot, path, &start_motion)` times it from exactly that state, as fast as the limits allow.
- `plan` only returns splines that can run from that state: `Trajectory::moving` times them from it, keeping every limit and joint range, as `check_from` checks. Where momentum would carry a joint out of its range in the first span, longer first spans are tried; a path that still cannot run is not returned.
- The spline's first spans last as long as the robot needs to change its velocity (about |v| / a). Where it still cannot follow the motion, longer spans are tried, and then trajectory optimization smooths the spline until it can.
- Timing gives the start the full limits, since its motion is not ours to choose, easing over the path's first two spans to those the rest keeps, and stretches the rest of the motion where its smoothed map needs it while keeping the start's speed. Replanned to new goals from halfway along planned Panda motions, at about 3 rad/s, motions take 3.5 s on average, against 12.9 s when timing gave up on such starts, and none took longer ([Benchmarks](benchmarks.md#replanning-while-moving)).
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
- **Replanned:** when the rest collides, it plans from the state the trajectory reaches `budget + lead` later (50 ms and 0), starting in motion, within `budget`, and switches to the new trajectory then. A tick that replans takes as long as the plan, up to `budget`; a controller that keeps running the current trajectory meanwhile (`trajectory()`, `began()`) never waits for it. `Replanned { at, trajectory }` hands the new trajectory over when it is planned, so a controller in another process or on the robot can take it up at `at`; `lead` is the time it has to get there. Only replans and the start of a detour's `then` (`Resumed`, below) wait for the lead: stops, re-anchors and collisions take effect at the tick that reports them, and each drops a replan sent before. On `Stopping`, `Collides` or `Blocked` such a controller stops the robot itself; on `Reanchored` it takes up `trajectory()` from `began()`, the tick's time, as soon as it can.
- **Before a takeover:** the rest of the motion is the current trajectory up to the takeover, then the replan. If either collides, the replan is dropped and the robot stops.
- **Stopping, then blocked:** when no plan arrives in time, or the collision comes sooner than `budget + lead`, it takes the controlled stop along its path and reports `Blocked` at rest. Every tick of the stop while it is clear reports `Stopping { contacts }`: what the rest of the motion collided with where it was first found to, as `Device::contacts` lists it, its obstacles numbered as in the worlds given to the tick the stop began at, however the worlds change after. A graze the GPU's check finds just outside what the CPU threads list names the nearest within 0.1 mm. Plan again from there, or give up.
- **Collides:** the stop is checked too. When no stop within the limits ends short of the collision, the status is `Collides { at, contacts }`, with the time it comes and what the robot collides with then, numbered as in that tick's worlds, and at a graze the nearest within 0.1 mm, as for `Stopping`: stop the robot some other way.
- **Holding an object:** `FollowerOptions::holding` takes the `Held` the trajectory was planned holding ([holding objects](robots.md#holding-objects)), with the robot and the devices that do not hold it, as a problem's `holding` does. The rest of the motion and any stop are checked with the object in the hand, contacts name it (`"held by hand 0"`), and replans and re-anchors carry it; what it may touch is the worlds' `Touch::held`. The first tick on a device builds the device that holds it, and later ticks on that device reuse it (a follower keeps one, so alternating ticks between two devices builds each time): on a GPU that builds kernels unless a call there has already held an object in that hand, as the plan the trajectory came from has.
- **Keeping the path:** with `replan` off, it never plans, for a motion that must stay on its path, as one along a tool path (`follow`, timed with `HandSpeed`) must: a replan would leave it for a free-space motion at full speed. Where the rest collides, it stops on its path and reports `Blocked` at rest; measured past `tolerance`, it holds the robot where it is, `Blocked`. Follow the rest of the path again from there, or give up.
- **A detour, for a joined motion:** a way joined into a move in (`join`) must keep the move in's path, and may change the way's. `FollowerOptions::detour` takes a `Detour { from, until, then }`: times on the trajectory's own clock, and the part whose path must not change as planned apart (a task's move in, without its transit). The follower keeps the path before `from` and after `until`, as with `replan` off. Where the rest of the motion collides within the window, it plans to `then`'s first state, from no sooner than `from`, at rest there. At the end of that detour it takes up `then` and reports `Resumed { at, trajectory }`, the tick a controller in another process takes `then` up at, from rest. Along a detour it replans to the same state; along `then` it keeps the path. Measured past `tolerance` within the window or along a detour, it plans from the measured state to `then`'s first state; elsewhere it holds the robot. Once a stop or a hold has left the trajectory, the window is read where it left: where the stop came to rest on the path, or where the hold began. `then` must pass `Trajectory::check`. Set `until` where the joined path rejoins the move in, past the blend, and `from` after a move away joined into the way, where a motion begins with one. It needs `replan` on; without it, `None`, nothing changes.

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
| `IkOptions::seeds` | 64 | More seeds find more and better-cleared solutions, at proportional cost. |
| `PlanOptions::rrt.max_iterations` | 10,000 | Samples a search draws without its clearance before it gives up, after up to 500 with it: more solve rarer problems, and unsolvable ones take longer to give up. |
| `PlanOptions::rrt.clearance`, `shortcut.clearance` | 1.5 cm, 1 cm | Distance searches and shortcuts keep from obstacles, away from the path's ends. More in shortcuts keeps paths clear of geometry the spheres miss, but makes them longer. |
| `PlanOptions::shortcut.passes` | 3 | Shortcut and smoothing passes: more give shorter paths, at some time. |
| `PlanOptions::shortcut.straighten` | on | Straighten each joint along the path: about 1% shorter paths on the Panda benchmark and 13–18% less joint travel on pick-and-place carries, for about 3% of batched throughput. |
| `PlanOptions::control_points` | 24 | Control points of the spline and of optimized seeds: more follow longer paths more closely. A searched path that no spline this size follows gets more. |
| `PlanOptions::seeds` | 0 | Trajectory optimization seeds per problem: with 2, paths are about 3% shorter (shorter than cuRobo's on its benchmark), at milliseconds per problem. |
| `RobotOptions::velocity_scale`, `acceleration_scale`, `jerk_scale` | 1 | A slower robot everywhere: plans, timings, checks and replanning from motion all keep to the scaled limits. `Trajectory::try_new`'s speed scale slows one trajectory instead. |
| `PlanOptions::time_budget` | none | Bounded latency, at the cost of determinism. |
