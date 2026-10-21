# Planning

Planning in batchplan is three steps, each a batched call on a [`Device`](devices.md):

1. **IK** (`solve_ik`) turns target poses into joint configurations.
2. **Planning** (`plan`) finds a collision-free path from a start to a goal configuration (the straight line, bent if an obstacle only grazes it, else RRT-Connect), then shortens and smooths it into a B-spline. Trajectory optimization can polish it further.
3. **Timing** (`Trajectory::new`) turns the path into a trajectory a controller can run, within the robot's velocity, acceleration and jerk limits.

Every call takes many problems, each naming its world, so one call can cover thousands of scenes.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::*;

fn main() -> Result<(), Error> {
    let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &RobotOptions::default())?;
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;

    // Several targets in one call; each problem names its world.
    let down = Quat::from_rotation_x(std::f32::consts::PI);
    let targets = [Vec3::new(0.4, 0.2, 0.3), Vec3::new(0.3, -0.3, 0.4)];
    let goals: Vec<IkProblem> = targets.iter().map(|&p| IkProblem::new(0, Pose::new(p, down))).collect();
    let ik = solve_ik(&device, &worlds, &goals, &IkOptions::default())?;

    // `solved()` yields each problem that has a solution, with its world, so they cannot be mixed up.
    let problems: Vec<PlanProblem> =
        ik.solved().map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution)).collect();
    let result = plan(&device, &worlds, &problems, &PlanOptions::default())?;
    for s in result.solved() {
        let trajectory = Trajectory::new(&robot, s.solution, 1.0)?;
        println!("problem {}: {:.2} s", s.index, trajectory.duration());
    }
    Ok(())
}
```

## Inverse kinematics

`solve_ik` runs damped least squares from many seeds per target (32 by default), with the collision gradient projected into the Jacobian's null space so seeds slide away from obstacles without leaving the target. A seed succeeds when it is within `position_tolerance` (1 mm) and `rotation_tolerance` (0.01 rad) of the target and collision-free.

- `IkProblem::new(world, target)` targets a one-handed robot's hand from the default pose; the fields below cover the rest.
- `ik.best(problem)` is the successful seed with the most obstacle clearance; `ik.solved()` yields every solved problem with that configuration.
- `IkProblem::seed` starts seed 0 from a configuration you give (the robot's current state, say) instead of the default pose, and `ik.nearest(problem, &q)` returns the successful seed nearest `q`. Together they keep short moves on the same IK branch, as pick-and-place does for its descents.
- `IkProblem::targets` gives each of the robot's hands (`RobotOptions::ee_links`) a target pose, or `None`. Each step solves every targeted hand's damped least squares and pushes the collision gradient out of their Jacobians' row spaces, which is exact for hands on separate arms. A hand without a target holds its arm: in every seed, the joints only that hand moves stay where the seed has them. So one arm can reach while the other waits, or both can reach at once and keep clear of each other.

## Planning a path

`PlanProblem::new(world, start, goal)` plans from rest with every joint free to move; set the other fields on top of it (`PlanProblem { hold, ..PlanProblem::new(world, start, goal) }`).

`PlanProblem::hold` lists joints that stay where `start` has them, such as the other arm of a two-armed robot while one arm moves (`robot.hand_joints(hand)` lists an arm's); `goal` must have the same values there. The search samples and bends only the other joints, no goal equivalent turns a held joint, and optimized paths are put back on the held values before they are checked, so every control point keeps them.

`plan` runs every problem as its own search, whole on one CPU thread, with the searches in parallel on either device, as VAMP does (Thomason, Kingston and Kavraki, 2024):

1. **The straight line** to each of the goal's equivalents (below) is tried first.
2. **A bent line.** If every straight line collides, the one to the nearest goal is bent: the stretch from where it first collides to where it last does moves 0.05 rad aside, across the line along each joint's axis in turn, both ways, then 0.1, 0.2 and 0.4 rad. The first bend whose edges are free is the path. So a line that an obstacle only grazes keeps its route instead of taking the detour a search would find. Bending costs no measurable time.
3. **RRT-Connect** grows a tree from the start and one from the goals, in turn, toward Halton samples. The first samples lie in a tube around the straight line to the nearest goal: 0.4 rad across it (along each axis across it) and past its ends, doubling every 64 samples until it spans the joint ranges. So the route found stays near the straight line where such a route exists, as cuRobo's roadmap does by sampling an ellipsoid around it; samples from the whole space find detours that shortcuts cannot undo. Extensions step at most `rrt.range` (1 rad), and dynamic domains skip samples far from any node that recently failed. A search gives up after `rrt.max_iterations` (10,000) samples from the joint ranges; the tube's samples come on top.
4. **Shortcutting and smoothing** run for up to `shortcut.passes` passes (3). Each pass replaces waypoints with the longest free edge, splits every edge, and moves each waypoint a quarter of the way toward its neighbours where both new edges stay free. With `shortcut.straighten` (on), after the first pass's shortcuts each joint in turn is straightened where that shortens the path by at least `shortcut.min_change`: it changes at one rate along the whole path while the other joints keep theirs, as in HPP's partial shortcut. That removes a joint's swing out and back (a wrist's, say), which shortcuts between whole configurations keep.
5. **A B-spline** follows the path, with `control_points` control points (24). The first and last three are pinned to the start and goal, so it starts and ends at rest. The others first spread along the smoothed path by arc length, so the curve is smooth and cuts the path's corners. It is checked at samples no farther apart than `shortcut.resolution`, as edges are. Where it fails, the waypoint nearest the failure is pinned: three control points sit on it, so the curve passes through it, stopping there. After up to six pins, the fit starts over on the shortcut path; with every waypoint pinned, the curve runs exactly along the checked path. The first spline that validates is the result, once its stops are eased (as in OpenRAVE's smoother): each stop's three control points spread into the widest blend around the corner that keeps the curve valid. Blends are the smaller of the free control points' spacing and half the shorter edge at the corner, then a quarter and a sixteenth of that; a corner no blend clears keeps its stop. If no spline validates, ten steps of trajectory optimization push the first one off the obstacles, and the result is checked like any other spline.

**Checking edges.** An edge is collision-checked at evenly spaced states, `rrt.resolution` (1/32 rad) apart, eight at once in SIMD lanes: spread over the edge, then stepping back, so a collision anywhere is found early. [Architecture](architecture.md#searches) describes the cheaper tests that skip most of the work.

**Clearance.** A path that only just clears the spheres can graze the meshes where the spheres leave them uncovered. So RRT-Connect keeps `rrt.clearance` (1.5 cm) from obstacles, and shortcuts keep `shortcut.clearance` (1 cm). Both shrink to zero within `clearance_ramp` (0.5 rad) of the start and the goals, since paths often start or end in contact, as at a grasp. The search's wider clearance finds routes through open space, which clear the meshes more often; the shortcuts' narrower one lets the path pull tight. If RRT-Connect finds no path within 2,000 samples, it continues from the same trees without its clearance for up to `rrt.max_iterations` samples more, so narrow passages stay passable. Splines keep the shortcuts' clearance first, then a 2 mm margin (or the shortcuts' clearance, if smaller), so the curve cannot graze an obstacle between the samples it is checked at.

**Validation.** Each spline candidate is sampled densely (`validate_substeps` per span) and checked for world and self-collision and joint limits. `result.best(problem)` is the shortest valid path, and `result.solved()` yields every solved problem with it.

### Trajectory optimization (optional)

`PlanOptions::seeds` (0 by default) adds trajectory optimization of that many seeds per problem, batched on the device:
- **Seeds.** The first starts from the searched spline, the second from the straight line to the goal, as cuRobo's first attempt does. Further seeds run straight to the goal's equivalents, or bend through random via points. Problems searched to within 2% of the straight line are not optimized, since no seed could be much shorter; their seeds stay invalid.
- **Cost.** Length and smoothness, as squared differences of the control points (weighted `w_vel`, 10) and squared second differences (`w_acc`, 5), plus a collision cost for the world and for self-collision, sampled along the curve (`samples_per_span` points per span) and active within `collision.margin` (1.2 cm) of an obstacle.
- **Optimizer.** L-BFGS (`iterations`, 100, remembering `history` steps). Each iteration prices four step sizes along the L-BFGS direction in one batched pass and keeps the cheapest.

Repairing a stuck spline (above) uses the same optimizer with its own settings, whatever these options say: ten steps, smoothness weighted over length (`w_acc` 50, `w_vel` 2) and a 2 cm margin. These repair more splines than the length-favouring weights do.

Optimized seeds are validated like the searched path, and `best` picks the shortest valid one of all. On the benchmark's box-only problems, two seeds give 3.137 rad paths at the median, against 3.243 without and cuRobo's 3.144 ([benchmarks](benchmarks.md#results)). Optimization costs milliseconds per problem (16 ms on 14 CPU threads; 734 problems/s in batches on an M4 Pro's GPU), against a fraction of a millisecond for the search.

### Goals a turn away

Continuous joints are unbounded, and joints whose range spans more than a turn (the UR5e's ±2π) reach the same pose at angles a turn apart. Goals are reached at whichever equivalent is nearest or free: the search takes them all as its goal set, and continuous joints step and measure the short way round. On the UR5e, a goal whose shoulder has turned past the table below is reached the long way round when the short way is blocked.

### Time budgets and determinism

`PlanOptions::time_budget` stops every search and the optimization once it passes, and returns the valid paths found so far. Without a budget nothing checks the clock, and two runs give bit-identical results, on either device.

## Executable trajectories

`Trajectory::new(&robot, path, speed_scale)` keeps the path and adds a time map `s = σ(t)` (itself a uniform cubic B-spline), choosing the faster of two timings:

- **Time-optimal** (TOPP-RA, Pham and Pham 2018): the fastest timing of the path under joint velocity and acceleration limits, plus a speed cap from the jerk limit, smoothed into a B-spline time map. Where the smoothed map overshoots a limit, the caps are lowered there and the profile solved again. On planned Panda paths it is 12% faster than uniform timing on average and never slower, at about 0.5 ms per trajectory.
- **Uniform**: `σ(t) = t / h`, with the smallest `h` that keeps every joint within its limits. Velocity, acceleration and jerk are then B-splines over the control points' differences, so the bound holds exactly along the whole curve.

Either is slowed by `speed_scale` (1.0 is as fast as the limits allow). Limits come from `RobotOptions::joint_limits` for joints it names, then from the robot description when it has them (URDF 1.2 `acceleration` and `jerk`), otherwise from `RobotOptions` (5 rad/s² and 50 rad/s³ by default).

- **For control loops**: `trajectory.at(t, &mut state)` writes position, velocity and acceleration without allocating.
- **For datasets**: `trajectory.sample(hz)` returns fixed-rate samples.
- **For safety layers**: a `Trajectory` is plain serializable data. `trajectory.check(&robot)` refuses one that is non-finite, does not start and end at rest, leaves a joint range, runs backward, or exceeds a velocity, acceleration or jerk limit.
- **For stopping**: `trajectory.stop(&robot, t)` is a controlled stop: the same path from time `t`, braking to rest as soon as an even deceleration within the limits allows. It stays on the checked path, so it needs no new collision check; verify it with `check_from` and the state at `t`. On planned Panda reaches it takes under 1.2 ms to compute, rests in about half the time left, and errors only where the path ends sooner than any stop would (late in short, fast motions).

### Starting from a moving state

To replan mid-motion, give `PlanProblem::start_motion` the robot's current velocity and acceleration. The spline's first control points then continue that motion, and `Trajectory::moving(&robot, path, &start_motion)` times it from exactly that state, as fast as the limits allow.
- `plan` only returns splines that can be timed this way.
- The spline's first spans last as long as the robot needs to change its velocity (about |v| / a). Where it still cannot follow the motion, longer spans are tried, and then trajectory optimization smooths the spline until it can.
- When reaches are replanned to the same goal from partway through, plans succeed every time with joints moving at up to 1.5 rad/s, and four times in five at the Panda's full speed (2 rad/s). In the rest, momentum carries a joint out of its range or into the scene, or no timing keeps the limits.
- A path that cannot brake in time is refused with `Error::Unsafe`.
- `trajectory.check_from(&robot, &state)` verifies that a trajectory starts at a given state.

```rust,no_run
use batchplan::*;

fn main() -> Result<(), Error> {
    let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &RobotOptions::default())?;
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;
    let now = JointState { position: robot.default_q().to_vec(), velocity: vec![0.2; robot.dof()], acceleration: vec![0.0; robot.dof()] };
    let motion = StartMotion { velocity: now.velocity.clone(), acceleration: now.acceleration.clone() };
    let goal = vec![0.5, -1.2, 1.0, -1.4, -1.5, 0.3];
    let problem = PlanProblem { start_motion: Some(motion.clone()), ..PlanProblem::new(0, now.position.clone(), goal) };
    let result = plan(&device, &worlds, &[problem], &PlanOptions::default())?;
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
| `PlanOptions::rrt.max_iterations` | 10,000 | Samples before a search gives up: more solve rarer problems, and unsolvable ones take longer to give up. |
| `PlanOptions::rrt.clearance`, `shortcut.clearance` | 1.5 cm, 1 cm | Distance searches and shortcuts keep from obstacles, away from the path's ends. More in shortcuts keeps paths clear of geometry the spheres miss, but makes them longer. |
| `PlanOptions::shortcut.passes` | 3 | Shortcut and smoothing passes: more give shorter paths, at some time. |
| `PlanOptions::shortcut.straighten` | on | Straighten each joint along the path: about 1% shorter paths on the Panda benchmark and 13–18% less joint travel on pick-and-place carries, for about 3% of batched throughput. |
| `PlanOptions::control_points` | 24 | Control points of the spline: more follow longer paths more closely. |
| `PlanOptions::seeds` | 0 | Trajectory optimization seeds per problem: with 2, paths are about 3% shorter (shorter than cuRobo's on its benchmark), at milliseconds per problem. |
| `RobotOptions::velocity_scale`, `acceleration_scale`, `jerk_scale` | 1 | A slower robot everywhere: plans, timings, checks and replanning from motion all keep to the scaled limits. `Trajectory::new`'s speed scale slows one trajectory instead. |
| `PlanOptions::time_budget` | none | Bounded latency, at the cost of determinism. |
