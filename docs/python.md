# Python

The `batchplan` package binds the library for Python: the same calls, batches and results, with NumPy arrays in and out. It is the Rust code underneath, so a plan made from Python is the plan Rust makes, on the same GPU or threads.

## Installing

Install [batches from PyPI](https://pypi.org/project/batches/):

```bash
pip install --pre batches
```

The distribution is named `batches`; the import is `batchplan`. Python 3.10 or newer is required.
The initial `0.11.0b1` release contains source only, so installing it needs a Rust toolchain
([rustup](https://rustup.rs)). `--pre` includes beta releases; omit it when you want only stable
releases once those are available.

For changes on `main` that have not reached PyPI, build from the repository:

```bash
pip install "git+https://github.com/natsuku-lab/batchplan#subdirectory=python"
```

To work on the bindings, `cd python && uv run pytest` builds them and runs their tests.

## A whole program

```python
import batchplan as bp

robot = bp.Robot.load("assets/ur5e/ur_description/urdf/ur5e.urdf")
device = bp.Device.cpu(robot)  # bp.Device.gpu(robot) pays off from a few hundred problems

table = bp.Obstacle.cuboid((0.0, 0.0, -0.02), (1.0, 1.0, 0.02))
worlds = device.upload([bp.World([table])])

down = (1.0, 0.0, 0.0, 0.0)  # half a turn about x, scalar last
targets = [(0.4, 0.2, 0.3), (0.3, -0.3, 0.4)]
problems = [bp.PlanProblem(0, robot.default_q, bp.Pose(p, down)) for p in targets]
plans = bp.plan(device, worlds, problems)  # IK from the start finds where each path ends
for s in plans.solved():
    trajectory = bp.Trajectory(robot, s.solution)
    samples = trajectory.sample(100.0)  # positions, velocities, accelerations: (samples, dof)
    print(f"motion {s.index}: {trajectory.duration:.2f} s, {len(samples)} samples")
```

## How it maps to Rust

The names are Rust's: `Robot.load`, `Device.upload`, `solve_ik`, `IkResult.best`, `plan`, `PlanResult.solved`, `pick`, `follow`. [The API page](api.md) lists them by task, and each guide explains what they do. What differs:

- **Values are frozen.** Poses, obstacles, worlds, problems and options can't change in place: `options.replace(seeds=4)` returns a copy with a field changed, as Rust's struct update syntax does, and `world.without(i)` is Rust's `World::remove` on a copy. Constructors take the fields Rust's `new` does, and any others by keyword: `bp.PlanProblem(0, start, goal, hold=[0])`, `bp.PlanOptions(seeds=2)`. Defaults are Rust's.
- **Arrays come out read-only.** Configurations, control points `(points, dof)` and per-seed results `(problems, seeds, ...)` are NumPy arrays, `float32` (`bool` for flags such as `success` and `valid`); copy one to change it. Anything array-like goes in. A result builds each array once, so reading `result.q[i]` in a loop costs nothing more.
- **Rotations are unit quaternions, scalar last:** `(x, y, z, w)`, as ROS, SciPy and glam order them.
- **A plan's goal is a configuration or poses:** an array, a `Pose` for a one-handed robot, or a `Pose` or `None` per hand, as Rust's `Goal` is. `problem.goal` gives poses back as that tuple.
- **No-argument accessors are properties:** `robot.dof`, `robot.default_q`, `trajectory.duration`, `device.name`.
- **Sequences come out as tuples, and maps as read-only views:** `world.obstacles`, `result.failures`, `result.solved()`, `bp.grasps(...)`, a task's plans, `options.lock_joints`.
- **Failures and contacts are classes per kind,** for `isinstance` or `match`: `bp.PlanFailure.StartCollides(contacts)`, `bp.Contact.World(link, obstacle, distance)`, `bp.ToolPathStop.Collision(at)`.

## Picking up and setting down

A whole task in one call, as [planning](planning.md#picking-up-and-setting-down) explains:

```python
import batchplan as bp

options = bp.RobotOptions(
    ee_links=["ee_link"],
    lock_joints={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
    default_q=[0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8],
    collision_model=bp.CollisionModel.load("assets/franka/panda_collision.json"),
)
robot = bp.Robot.load("assets/franka/franka_panda.urdf", options)
device = bp.Device.cpu(robot)

table = bp.Obstacle.cuboid((0.4, 0.0, -0.02), (0.7, 0.8, 0.02))
box = bp.Obstacle.cuboid((0.45, 0.2, 0.03), (0.02, 0.03, 0.03))
worlds = device.upload([bp.World([table, box])])

grasps = bp.grasps(box, bp.GraspOptions(tip=robot.tip(0), floor=0.005))
outcome = bp.pick(device, worlds, [bp.Pick(0, robot.default_q, 1, grasps, support=[0])]).outcomes[0]
if isinstance(outcome, bp.TaskFailure):
    print(outcome)  # where each grasp stopped, and why
else:
    plan = outcome[0]  # the quickest of the grasps that got through
    print(f"{plan.duration:.2f} s, holding by grasp {plan.candidate}")
    # Run plan.transit and plan.approach, close the gripper, then run plan.depart. A task with
    # transit=False has no plan.transit: plan.begin is where a way planned apart must end.
```

## Errors

Every error batchplan reports is a `bp.Error`; each kind of `batchplan::Error` is its own subclass, to catch one or all: `InputError` (what the call was given), `LoadError` (a file), `GpuError`, `ThreadsError`, `UnsafeError` (a trajectory past a limit) and `WriteError`. An argument of the wrong type, an index out of range or a number past what its field holds raises Python's own `TypeError`, `IndexError` or `OverflowError` before batchplan sees it.

```python
import batchplan as bp

robot = bp.Robot.load("assets/ur5e/ur_description/urdf/ur5e.urdf")
device = bp.Device.cpu(robot)
worlds = device.upload([bp.World()])
try:
    bp.solve_ik(device, worlds, [bp.IkProblem(3, bp.Pose((0.4, 0.0, 0.3)))])
except bp.InputError as e:
    print(e)  # names the problem and the world it asked for
```

## Threads

Every call lets go of the interpreter while it works, so other Python threads run meanwhile, and calls from several threads run at once. The work itself runs on the device's own threads ([devices](devices.md)).

## What it binds

Robots (`Robot.load`, `from_urdf`, `assemble` from `Part`s, `attach`, `detach`, `relock`, forward kinematics and spheres; `CollisionModel` with `load`, `save`, `to_json`, `from_json` and `assemble`), obstacles and worlds (`World.without`, `always_touching`), devices, IK, planning with hand constraints, timing, tool paths (`follow`), whole tasks (`pick`, `place`, `grasps`, `place_poses`) and held objects (`Held`, per problem as `holding`). Not yet: training data (`datagen`, the NumPy and LeRobot exports), live motion (`Follower`, `servo`), joining motions (`join`), distance grids and perception, `World.load`, `MeshModel`, `Robot.qualify`, `Robot.breach`, `Device.evaluate` and `Device.contacts`. OpenUSD files need the library's `usd` feature, which these builds leave out.
