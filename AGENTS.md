# Agent instructions

This file is for coding agents that use batchplan in a program. To change batchplan itself, read [`CLAUDE.md`](CLAUDE.md): its design rules, invariants and working rules. The guides under [`docs/`](docs/README.md) explain each topic in full; this page gathers what an agent needs first and links to them.

## Use batchplan: make once, warm up, reuse

Build the robot and its device once, warm the device, upload each scene once, and reuse all three for every request. Results are per problem: read each one's answer or its reason, never assume a batch succeeded together.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::{
    Device, Error, IkOptions, IkProblem, Obstacle, PlanFailure, PlanOptions, PlanProblem, Pose,
    Robot, RobotOptions, Trajectory, World,
};

fn main() -> Result<(), Error> {
    // Once per robot: load it, check its spheres, make its device and warm it up.
    let options = RobotOptions {
        ee_links: vec!["tool0".into()],
        ..Default::default()
    };
    let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &options)?;
    for link in robot.qualify()?.links {
        println!("{}: spheres reach {:.1} mm past it", link.link, link.overhang * 1e3);
    }
    let device = Device::gpu(&robot).or_else(|_| Device::cpu(&robot))?;
    device.warmup()?;

    // Once per scene, and again whenever it changes: obstacles in the robot's frame, with the
    // links that always touch the table it stands on allowed to.
    let table = Obstacle::cuboid(Vec3::new(0.3, 0.0, -0.02), Vec3::new(0.6, 0.6, 0.02));
    let mut world = World::new(vec![table]);
    let mounted = world.always_touching(&robot);
    world.touch.extend(mounted);
    let worlds = device.upload(&[world])?;

    // Per request: a pose in the robot's frame (quaternions x, y, z, w), from the measured state
    // in `robot.joint_names()` order.
    let target = Pose::new(Vec3::new(0.4, 0.2, 0.3), Quat::from_xyzw(1.0, 0.0, 0.0, 0.0));
    let start = robot.default_q();
    let ik_options = IkOptions::default();
    let ik = batchplan::solve_ik(&device, &worlds, &[IkProblem::new(0, target)], &ik_options)?;
    let Some(goal) = ik.best(0) else {
        let why = batchplan::ik::failures(&device, &worlds, &ik, &ik_options)?;
        println!("no IK solution: {:?}", why[0]);
        return Ok(());
    };
    let problem = PlanProblem::new(0, start, goal);
    let plans = batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default())?;
    match (plans.best(0), &plans.failures[0]) {
        (Some(path), _) => {
            let trajectory = Trajectory::try_new(&robot, path, 1.0)?;
            trajectory.check(&robot)?;
            let samples = trajectory.sample(100.0)?; // positions, velocities, accelerations
            println!("{} samples over {:.2} s", samples.len(), trajectory.duration());
        }
        (None, Some(PlanFailure::StartCollides { contacts })) => {
            println!("the start collides: {contacts:?}");
        }
        (None, why) => println!("no plan: {why:?}"),
    }
    Ok(())
}
```

### Conventions

- **Units.** Meters, radians (meters for prismatic joints), seconds, `f32` throughout.
- **Poses** are `Pose { position, rotation }` with [glam](https://docs.rs/glam) types, re-exported as `batchplan::glam`. glam's quaternions are x, y, z, w (`Quat::from_xyzw`), as ROS messages are; MuJoCo and Eigen's constructor are w, x, y, z. Rotations must be unit quaternions.
- **Frames.** Targets, obstacles and depth-camera poses are all in the robot's frame: its description's root link, or `world` for a robot made with `Robot::assemble`. Transform camera and map data into it first.
- **Joint order** is `robot.joint_names()`: the actuated joints after locking, numbered breadth first, so two arms' joints interleave. Map a controller's names to it; never assume the description's order. Planar, ball and floating joints become one-axis joints named `<joint>_x`, `_theta` and so on ([robots](docs/robots.md#kinematics)).
- **Hands** are the IK frames of `RobotOptions::ee_links`, numbered in that order. For a gripper, use a frame between the fingertips, z out of the hand and y along the closing direction: what `grasps` assumes.

### Lifecycle

- **Robot.** `Robot::load` fits collision spheres unless given a model, in a fraction of a second; save `robot.collision_model()` and pass it back as `RobotOptions::collision_model` to load the same spheres every time.
- **Device.** One per robot (`Device::gpu`, falling back to `Device::cpu`), made once: never inside a control loop or per request. `device.warmup()` before timing anything ([devices](docs/devices.md#make-once-warm-up-reuse)).
- **Worlds.** `device.upload(&worlds)` once per scene. Uploads are cheap (0.1–0.2 ms for 512 tabletop worlds), so a scene that changes is uploaded again; there is no partial update ([changing a scene](docs/perception.md#changing-a-scene)).
- **Locked joints that move**, as a torso or head the robot plans without: `robot.relock(&values)` then `device.with_robot(&relocked)`, which compiles nothing and keeps the uploaded worlds. Fit the collision model with those joints free and load with it: a model records the joints it was fitted with locked, whose pairs that meet only at other values would go unchecked, and `relock` refuses to move them ([robots](docs/robots.md#kinematics)).
- **Held objects.** `robot.attach(&AttachedObject)` then `device.with_robot(&held)`: the same GPU or threads, and the uploaded worlds still work. The held robot's GPU kernels are its own: warm its device up too before timing it. `robot.detach(name)` reverses it. Attachment is collision only: there is no payload mass or inertia, so timing a heavy carry ignores the load; slow it with a speed scale ([trajectories](docs/planning.md#executable-trajectories)).
- **Live loops** call `Follower::tick` or `servo` each period and nothing that builds: pass the measured state to `tick` so a robot held back or pushed is followed from where it is. A controller in another process takes each replan from `FollowStatus::Replanned { at, trajectory }`, with `FollowerOptions::lead` for the time sending it takes ([live motion](docs/planning.md#live-motion)).

### Choose the call

| Task | Call | Guide |
|---|---|---|
| Forward kinematics | `robot.ee_pose(q, hand)`, `robot.link_pose(q, link)`, `robot.spheres(q)` | [robots](docs/robots.md) |
| Configurations for hand poses | `solve_ik`; `ik::failures` for why one has none | [IK](docs/planning.md#inverse-kinematics) |
| A collision-free path between configurations | `plan`, then `Trajectory::try_new` | [planning](docs/planning.md#planning-a-path) |
| The hand along straight lines | `follow` | [tool paths](docs/planning.md#moving-the-hand-along-a-path) |
| A hand kept upright, in a box or relative to another hand | `HandConstraint` on the problem | [constraints](docs/planning.md#constraints-along-the-path) |
| Picking up and setting down | `grasps`, `pick`, `place` | [tasks](docs/planning.md#picking-up-and-setting-down) |
| Motions run without stopping between them | `join` | [joins](docs/planning.md#joining-motions) |
| Running a motion while the world changes | `Follower`, `servo` | [live motion](docs/planning.md#live-motion) |
| Obstacles from depth images or point clouds | `SdfGrid`, `OccupancyMap`, `DepthImage::robot_mask` | [perception](docs/perception.md) |
| Training data | `datagen`, `npy::export`, `lerobot::export` | [training data](docs/training-data.md) |

IK finds configurations; a plan finds the path to one. Every call takes many problems, each naming its world: batch them rather than calling once per problem, since each GPU call has a fixed cost.

### Qualify the robot

A robot that loads may still have spheres unfit to plan with. Before planning with a new robot or a hand-tuned collision model ([qualifying a robot](docs/robots.md#qualifying-a-robot)):

1. `robot.qualify()`: each link's `overhang` past its geometry, the `uncovered` share of its surface and its `thickness`, and each hand's `tips`. Fingers whose overhang or tip is a large part of the smallest object you will grasp will collide with what it stands on; give them a tighter tolerance (`RobotOptions::sphere_tolerances`) or a finer `SphereOptions::voxel_size`.
2. `world.always_touching(&robot)` with the table the robot stands on: links that touch it in every configuration need a touch, or every plan's start collides.
3. Fingers that are part of the palm's link, as the SO-101's fixed jaw: name them in `RobotOptions::finger_links`, or grasps where they meet the object fail.
4. `device.contacts(&worlds, &item_world, &q, 0.0)` at the default configuration, the SRDF's named states and a few others: each should collide with nothing.

### Read results

- `ik.best(i)`, `plans.best(i)` and `solved()` return only valid answers; `IkResult::success` and `PlanResult::valid` are per seed, problem-major (`problem * seeds + seed`).
- A plan is B-spline control points, not a trajectory: time it with `Trajectory::try_new` (`Trajectory::moving` from a moving start), sample it with `at` or `sample`, and `check` it before it runs on hardware. Never send control points to a controller.
- Plans do not depend on the batch or the device; with a `time_budget`, they depend on the machine's speed. IK starts every problem from the same seeds in any batch, but a GPU rounds differently from the CPU, and differently on either side of where it switches how many invocations share a configuration (4,096 configurations, more for robots with many spheres), so solutions there can differ slightly.

### Diagnose a failed request, in order

Check these before raising budgets:

1. **The input.** An `Error::Input` names what was refused and the value given: shapes, joint order, unit quaternions, joint limits (starts and goals must be within them), frames.
2. **The start.** `PlanFailure::StartCollides { contacts }` names each link and obstacle, or pair of links. A link touching the table in every configuration needs a touch (`always_touching`); a start inside a held object's spheres needs the object attached.
3. **The goal.** `PlanFailure::GoalCollides`; for IK, `ik::failures` tells `Unreachable` from `Collides { contacts }`.
4. **The search.** `NoPath`: more `rrt.max_iterations`, or the way is blocked. `NoSpline`: more `control_points`, or a slower start motion. `OutOfTime`: a longer `time_budget`.
5. **Tasks.** A `TaskFailure` prints where and why every candidate stopped; each `CandidateStop` has the step, the reason and the contacts. `grasps` says why it found none. Grasps near a support need `GraspOptions::tip` and `floor`.

### Footguns

- glam's quaternion order is x, y, z, w; a w-first source passed straight through turns the hand.
- `PlanOptions::seeds` is 0: no trajectory optimization unless asked for.
- A world's touches name links: worlds where a held object may touch something are uploaded to the device of the robot holding it.
- `grasps` returns poses in the object's frame; with a `floor`, pass the obstacle as it stands in the world.
- `Robot::lower` and `upper` are ranges; continuous joints turn without limit and wrap.
- `Follower::tick` with `None` trusts the trajectory; on hardware, pass the measured state.
