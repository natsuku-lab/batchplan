# Agent instructions

This file is for coding agents that use batchplan in a program. To change batchplan itself, read [`CLAUDE.md`](CLAUDE.md): its design rules, invariants and working rules. The guides under [`docs/`](docs/README.md) explain each topic in full; this page gathers what an agent needs first and links to them. From Python, the `batchplan` package makes the same calls by the same names; [Python](docs/python.md) says how they map and what it binds so far.

## Use batchplan: make once, warm up, reuse

Build the robot and its device once, warm the device, upload each scene once, and reuse all three for every request. Results are per problem: read each one's answer or its reason, never assume a batch succeeded together.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::{
    Device, Error, Obstacle, PlanFailure, PlanOptions, PlanProblem, Pose, Robot, RobotOptions,
    Trajectory, World,
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
    // IK from the start finds where the path ends: the nearest collision-free solution.
    let problem = PlanProblem::new(0, start, target);
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
        (None, Some(PlanFailure::GoalUnreachable)) => println!("no IK seed reaches the pose"),
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
- **Hands** are the IK frames of `RobotOptions::ee_links`, numbered in that order: links by name, or an MJCF's named `<site>`s. For a gripper, use a frame between the fingertips, z out of the hand and y along the closing direction: what `grasps` assumes.

### Lifecycle

- **Robot.** `Robot::load` fits collision spheres unless given a model: a fraction of a second for an arm at the default budget, seconds for thousands of spheres; save `robot.collision_model()` and pass it back as `RobotOptions::collision_model` to load the same spheres every time.
- **Device.** One per robot (`Device::gpu`, falling back to `Device::cpu`), made once: never inside a control loop or per request. `device.warmup()` before timing anything ([devices](docs/devices.md#make-once-warm-up-reuse)).
- **Worlds.** `device.upload(&worlds)` once per scene. Uploads are cheap (0.1–0.2 ms for 512 tabletop worlds, [benchmarks](docs/benchmarks.md#gpu-and-cpu-throughput)), so a scene that changes is uploaded again; there is no partial update ([changing a scene](docs/perception.md#changing-a-scene)).
- **Locked joints that move**, as a torso or head the robot plans without: `robot.relock(&values)` then `device.with_robot(&relocked)`, which compiles nothing and keeps the uploaded worlds. Fit the collision model with those joints free and load with it: a model records the joints it was fitted with locked, whose pairs that meet only at other values would go unchecked, and `relock` refuses to move them ([robots](docs/robots.md#kinematics)).
- **Held objects.** Pass a pick's `Held` on as `holding: Some(held)` on a `PlanProblem`, `IkProblem`, `ToolPath`, `Sequence` or `FollowerOptions`, or to `device.contacts_holding`, on the device without the object: one call takes problems holding different objects, a batch per hold. On a GPU, warm up holding an object in each hand that will hold one, and in each set of hands that will hold one together (`Held::also`) or with other links let touch it (`Held::touch_links`): the first call holding so builds its kernels. A device that holds one itself (`robot.attach`, then `device.with_robot`) has kernels of its own to warm up. Holding is collision only, without mass: slow a heavy carry with a speed scale ([holding objects](docs/robots.md#holding-objects)).
- **Live loops** call `Follower::tick` or `servo` each period and nothing that builds: pass the measured state to `tick` so a robot held back or pushed is followed from where it is (held there, `Blocked`, with `FollowerOptions::replan` off). A controller in another process takes each replan from `FollowStatus::Replanned { at, trajectory }`, with `FollowerOptions::lead` for the time sending it takes. `Stopping { contacts }` and `Collides { at, contacts }` name what is in the way, links and obstacles as `PlanFailure::StartCollides` names them ([live motion](docs/planning.md#live-motion)).

### Choose the call

| Task | Call | Guide |
|---|---|---|
| Forward kinematics | `robot.ee_pose(q, hand)`, `robot.link_pose(q, link)`, `robot.spheres(q)` | [robots](docs/robots.md) |
| Configurations for hand poses | `solve_ik`; `ik::failures` for why one has none | [IK](docs/planning.md#inverse-kinematics) |
| A collision-free path to a configuration or a hand pose | `plan`, then `time` (one path: `Trajectory::try_new`) | [planning](docs/planning.md#planning-a-path) |
| The hand along straight lines | `follow` | [tool paths](docs/planning.md#moving-the-hand-along-a-path) |
| A hand kept upright, in a box or relative to another hand | `HandConstraint` on the problem | [constraints](docs/planning.md#constraints-along-the-path) |
| A hand backing off, or working on, what another hand holds | `Held::touch_links` on the hold, as `robot.hand_links(hand).into_iter().map(String::from).collect()` | [holding objects](docs/robots.md#holding-objects) |
| Setting what one hand holds on or into what another holds | `Place::onto`, the other hand's `Held`, on the device without either; the plan's `held` is that hold carrying the object (`Held::carries`): hold it from then on, and `place` of it sets both down (`TaskPlan::carried`) | [onto what another hand holds](docs/planning.md#onto-what-another-hand-holds) |
| Two hands carrying one object | `Held::also` (a `Grip` per other hand), then `plan` with `holding` and one hand's goal pose, or `follow` with `holding` for a straight move | [holding objects](docs/robots.md#holding-objects) |
| Two hands picking up and setting down one object | `Pick::also` (a `HandGrasps`, one grasp per grasp of the pick's), then `place` with the plan's `held` | [two hands](docs/planning.md#two-hands) |
| Picking up and setting down | `robot.gripper` once per hand, then `grasps`, `pick`, `place` | [tasks](docs/planning.md#picking-up-and-setting-down) |
| Preferring some candidates, and stopping at enough plans | `Hold::cost`, `Release::cost`, `Pick::enough`, `Place::enough`; among tasks, the least `TaskPlan::cost` | [costs](docs/planning.md#costs-and-how-many-will-do) |
| Picking up a body by one part | `Pick::object` (the part gripped), `Pick::body`, then `Pick::lift_world` for the lift | [bodies](docs/planning.md#worlds-bodies-and-other-arms) |
| Fingers held at an opening other than as loaded | `Hold::new(pose, opening, &gripper)` per grasp, `Held::locked`, `Place::open`: many openings in one call | [finger openings](docs/grasps.md#finger-openings) |
| Grasps of a measured object of any shape (a rim, a handle) | `Surface::from_depth` per camera, `join`, then `grasps_on`; in a pick, `Hold::new(pose, grasp.opening, &gripper)` | [grasps](docs/grasps.md#grasps-of-what-cameras-see) |
| Grasps whose fingers close on one reading | `SurfaceGraspOptions::close_on`, then `Grasp::closes_on` | [closing on a point](docs/grasps.md#closing-on-a-point) |
| Finger contact with a measured part (a rim, a handle) | `miru::Observed::grip_worlds`, then `GripWorlds::plan`, in the optional crate `batchplan-miru` | [miru: grips](miru/docs/grips.md#a-grip-on-part-of-an-object) |
| Where a held object rests on a measured support (a plate, a rack) | `rests_on`, then `place` with every rest as a release (`Release::from(&rest)`); `rest_failures` for why one has none | [rests](docs/placing.md#rests-on-what-cameras-see) |
| Letting go just above a measured rest | `Release::drop`; the follower's move away checked against `TaskPlan::falling` | [releases](docs/placing.md#releases) |
| Where to let a held object fall into a measured container (a basket, a drawer) | `falls_into`, then `place` with every fall as a release (`Release::from(&fall)`); over every reading about the container, none flagged, where a segmentation gives it in pieces | [containers](docs/placing.md#letting-go-into-a-container) |
| Setting a held object down nearest a spot, from whichever of several picks gets it there | one `place` call, a `Place` per pick with releases costing their distance and `enough: Some(1)`; the least `TaskPlan::cost` | [placing](docs/placing.md#the-place-nearest-a-spot) |
| Turning a knob, a cap, a lever or a wheel | `turn` about a `Screw`, with what it turns on as `support`; measured grips from `grasps_on`, ordered by `SurfaceGraspOptions::order_for_turn` (with `shell: None`) | [turning](docs/turning.md#turning-a-thing-about-an-axis) |
| Grips to turn a control the cameras measured | `miru::Kept::turn_grips`, its groups planned in order with `turn`, in the optional crate `batchplan-miru` | [miru: grips](miru/docs/grips.md#turning-a-control) |
| Drawing a drawer by its handle | `pull` with the handle's obstacle, grasps in its frame, a world direction and signed meters | [drawing a handle](docs/turning.md#drawing-a-handle-along-a-line) |
| Passing an object from one hand to another | `pick` with `ShapeGraspOptions::along` (toward an end), then `handover` | [tasks](docs/planning.md#handing-over) |
| A program of picks and places planned together | `pick` and `place` with `transit: false`, then one `plan` and `join` | [tasks](docs/planning.md#a-program-of-tasks) |
| Motions run without stopping between them | `join` | [joins](docs/planning.md#joining-motions) |
| A motion that comes fast and arrives slowly | `Trajectory::with_hand_speed` with a cap near the grasp (`HandSpeed::near`); many paths: `time`; in `pick`/`place`: `TaskOptions::tool_speed_near` | [timing](docs/planning.md#executable-trajectories) |
| Running a motion while the world changes | `Follower`, `servo` | [live motion](docs/planning.md#live-motion) |
| Obstacles from depth images or point clouds | `SdfGrid`, `OccupancyMap`, `DepthImage::filtered`, `DepthImage::robot_mask` | [perception](docs/perception.md) |
| Objects in depth images: which pixels are one thing, its shape, what it stands on, the same id from look to look | `miru::look`, in the optional crate `batchplan-miru` | [miru](miru/README.md) |
| Which shape of a world each of a camera's readings lies on | `sdf::on_shapes` | [perception](docs/perception.md#shapes-the-caller-holds) |
| A perception library's labels and shapes, and what it needs back | `RobotMask::with_shapes`, `RobotMask::with_band`, `Surface::from_depth` with the labels as flags | [perception](docs/perception.md#with-a-perception-library) |
| One object as the cameras saw into it, space known empty or stale, and where a map is occupied | `OccupancyMap::part`, `free`, `clear`, `counts`, `occupied`, `take_surfaces` | [perception](docs/perception.md#what-a-caller-knows-and-one-objects-part) |
| Training data | `datagen`, `npy::export`, `lerobot::export` | [training data](docs/training-data.md) |

IK finds configurations; a plan finds the path to one, or to a pose, solving IK from the start itself. Solve IK first only to choose among its solutions. Every call takes many problems, each naming its world: batch them rather than calling once per problem, since each GPU call has a fixed cost. [API](docs/api.md) lists every call.

### Qualify the robot

A robot that loads may still have spheres unfit to plan with. Before planning with a new robot or a hand-tuned collision model ([qualifying a robot](docs/robots.md#qualifying-a-robot)):

1. `robot.qualify()`: each link's `overhang` past its geometry, the `uncovered` share of its surface, how deep the `gap` it leaves and its `thickness`, and each hand's `tips`. Fingers whose overhang or tip is a large part of the smallest object you will grasp will collide with what it stands on; give them a tighter tolerance (`RobotOptions::sphere_tolerances`) or a finer `SphereOptions::voxel_size`. `SphereOptions::max_overhang` holds every sphere to a distance past the geometry instead, at the cost of coverage: read each link's `gap` after setting it, since an obstacle reaches that deep into the link unseen ([a limit on overhang](docs/robots.md#collision-spheres)).
2. `world.always_touching(&robot)` with the table the robot stands on: links that touch it in every configuration need a touch, or every plan's start collides.
3. Fingers that are part of the palm's link, as the SO-101's fixed jaw: name them in `RobotOptions::finger_links`, or grasps where they meet the object fail.
4. `device.contacts(&worlds, &item_world, &q, 0.0)` at the default configuration, the SRDF's named states and a few others: each should collide with nothing.

### Read results

- `ik.best(i)`, `plans.best(i)` and `solved()` return only valid answers; `IkResult::success` and `PlanResult::valid` are per seed, problem-major (`problem * seeds + seed`).
- A plan is B-spline control points, not a trajectory: time a call's paths together with `time`, or one with `Trajectory::try_new` (`Trajectory::moving` from a moving start), sample it with `at` or `sample`, and `check` it before it runs on hardware. Never send control points to a controller.
- With `start_tolerance`, use the corrected start retained in the result's problem; goals and start motion remain as given ([measured starts](docs/planning.md#measured-starts-at-joint-limits)).
- Plans do not depend on the batch or the device; with a `time_budget`, they depend on the machine's speed. IK starts every problem from the same seeds in any batch, but a GPU rounds differently from the CPU, and differently on either side of where it switches how many invocations share a configuration (4,096 configurations, more for robots with many spheres), so solutions there can differ slightly, and so can plans to poses, which end where IK solves them.

### Diagnose a failed request, in order

Check these before raising budgets:

1. **The input.** An `Error::Input` names what was refused and the value given: shapes, joint order, unit quaternions, joint limits (strict by default; [measured starts can opt into correction](docs/planning.md#measured-starts-at-joint-limits)), frames. Check options where they are set: `check()` on `IkOptions`, `PlanOptions`, `RrtOptions`, `ShortcutOptions`, `ToolPathOptions`, `JoinOptions` and `TaskOptions` returns the error their call would. It refuses what a call cannot run with, not what is slow.
2. **The start.** `PlanFailure::StartCollides { contacts }` names each link and obstacle, or pair of links. A link touching the table in every configuration needs a touch (`always_touching`); a start inside a held object's spheres needs the object attached.
3. **The goal.** `PlanFailure::GoalCollides`, and for a pose goal `GoalUnreachable`, which more `PlanOptions::goal_ik` iterations may fix, or more seeds `kept` past its screening (`screening: 0` gives every seed every iteration); for IK, `ik::failures` tells `Unreachable` from `Collides { contacts }`.
4. **The search.** `NoPath`: more `rrt.max_iterations`, or the way is blocked. `NoSpline`: more `control_points`, or a slower start motion. `OutOfTime`: a longer `time_budget`.
5. **Tasks.** A `TaskFailure` prints where and why every candidate stopped; each `CandidateStop` has the step, the reason and the contacts, whose obstacles are numbered as in the task's world; in a place, `world.obstacles.len()` is the object where it is set down, and in a handover, the object where it changes hands. A turn stopped at `TaskStep::Turn` says how far it `turned`; `Unreachable` there means every configuration IK found ran out of joint range, so give grips turned the other way about the axis ([turning](docs/turning.md#turning-a-thing-about-an-axis)). `grasps` says why it found none. Grasps near a support need `ShapeGraspOptions::floor`, which keeps the open fingers over it too; from above, they keep at least `overlap` of the fingers (1 cm) in the object, so an object whose top is less than that over the floor has none.
6. **Rests.** For a problem `rests_on` found no rest for, `rest_failures`, given the options the rests were found with, says how near the search came ([why no rest](docs/placing.md#why-no-rest)):

   | `RestFailure` | What to try |
   |---|---|
   | `Blocked` | A point is in the way: set the object down elsewhere, or flag the point if the object may lie on it. |
   | `Tips` | Ask less `tipping`, or set it down elsewhere. Against a lip or a rim, a rest may lie in a strip narrower than the spacing: pass `RestProblem::near` there. |
   | `Unheld` | On a slope, give the rotation that lies flat on it, or flag the stop at its foot. |
   | `Unsupported` | Nothing flagged lies under it: flag the support's points. |

   On `Unheld` or `Tips`, a `blocked` point that is not the support's is part of what the object would lie against, left unflagged.

### Footguns

- glam's quaternion order is x, y, z, w; a w-first source passed straight through turns the hand ([conventions](#conventions)).
- `PlanOptions::seeds` is 0: no trajectory optimization unless asked for ([trajectory optimization](docs/planning.md#trajectory-optimization-optional)).
- A world's touches name links; what a problem's `holding` may touch is named with `Touch::held(obstacle, hand)`, one hand's per upload, on the device that holds nothing ([holding objects](docs/robots.md#holding-objects)).
- `holding` takes the device without the object: one already holding a `Held` in that hand is an `Error::Input` ([holding objects](docs/robots.md#holding-objects)).
- A held object's spheres reach past it by up to their tolerance (5 mm by default), and anything nearer reads as a contact: for a narrower gap, give a smaller tolerance in `Held::spheres` or `TaskOptions::held_spheres`, and leave room for where IK puts the hand (1 mm and 0.01 rad by default) ([holding objects](docs/robots.md#holding-objects)).
- A candidate's cost orders the plans that come back, not the order candidates are tried in: pass every candidate, those worth having first, with `enough` for how many plans will do ([costs](docs/planning.md#costs-and-how-many-will-do)).
- `grasps` returns poses in the object's frame; with a `floor`, pass the obstacle as it stands in the world ([tasks](docs/planning.md#picking-up-and-setting-down)).
- Open the gripper to a plan's opening before its transit, and follow its transit and move in on the robot relocked at `TaskPlan::held.locked`, and a place's move away at the opening it let go at: the device for the robot as loaded sees the open fingers ([finger openings](docs/grasps.md#finger-openings)).
- `grasps_on` also closes on walls seen from one side (`SurfaceGraspOptions::shell`, on by default): those grasps come after the rest, `one_sided`, their `width()` the most the wall can be, and a cloud fused from several views needs `shell: None` ([walls seen from one side](docs/grasps.md#walls-seen-from-one-side)). Within each turn, grasps whose fingers meet the object before their patches come after the rest (`SurfaceGraspOptions::first_touch`, on by default; [what comes back](docs/grasps.md#what-comes-back)).
- `grasps_on` checks the hand against the points with nothing to spare, and a planner's distance grids err toward collision: for worlds of grids, give `SurfaceGraspOptions::clearance` (2.5 mm suits grids a few millimeters fine), or a hand the points find clear is refused at its reach ([room a planner needs](docs/grasps.md#room-a-planner-needs)).
- A real depth camera reads points in mid-air along every edge: pass its depths through `image.filtered(&DepthFilter { noise, .. })`, with the camera's noise, before anything else reads them; a simulator's exact depth needs no filter ([readings between two surfaces](docs/perception.md#readings-between-two-surfaces)).
- `robot_mask` leaves out what a hand holds only where it is named (`holding`); an object in the hand that is not named reads as an obstacle ([masking the robot](docs/perception.md#masking-the-robot)).
- Beside the arm's or a shape's outline a camera reads points in mid-air, where fingers go. `mask.with_band(&device, &image, margin, pixels)` after `with_shapes` takes them. `pixels` is a fact of the camera (1 for sub-pixel disparity, half the window for block matching), not a knob ([shapes the caller holds](docs/perception.md#shapes-the-caller-holds)).
- A fused map keeps what it saw until a camera sees through it: tell it what the robot moved (`free` where it took something away, `clear` where something moved), and take `part` of what must be seen into, since under `Occlusion::Occupied` its grid walls in all no camera has seen ([what a caller knows](docs/perception.md#what-a-caller-knows-and-one-objects-part)).
- A map weighs each image as evidence, so an object put where cameras saw empty space takes up to three looks to fill it; a caller that acts on each look wants `map.with_weighing(Weighing::Latest)` ([fused maps](docs/perception.md#fused-maps)).
- `rests_on` sets the object straight down at the `rotation` given; nothing is tipped or slid into place. Flag every point it may lie on, give its true shape, and pass every rest to `place`. A rest narrower than the spot spacing is found about one place: pass `RestProblem::near` where you would have it ([rests](docs/placing.md#rests-on-what-cameras-see)).
- "Into" a container is `falls_into`, "on" is `rests_on`. Call the one the task asks for, not one and then the other. `falls_into` gives nothing for a tray, a plate or a table, nor over a support that stands beside the hollow ([letting go into a container](docs/placing.md#letting-go-into-a-container)).
- A task's world holds for all its steps unless `transit_world` or `depart_world` name others: what the hand may touch at the grasp, it may touch all the way from the start ([worlds](docs/planning.md#worlds-bodies-and-other-arms)).
- `TaskPlan::transit` is `None` for a task with `transit: false`, whose start is a seed and is not checked ([a program of tasks](docs/planning.md#a-program-of-tasks)).
- A turn checks what its object sweeps: name what it turns in or on in `Turn::support` (a cap's bottle, a lever's mount), or set `Turn::stays` where the world holds the part and its housing as one obstacle ([turning](docs/turning.md#turning-a-thing-about-an-axis)).
- `Robot::lower` and `upper` are ranges; continuous joints turn without limit and wrap ([kinematics](docs/robots.md#kinematics)).
- `Follower::tick` with `None` trusts the trajectory; on hardware, pass the measured state ([live motion](docs/planning.md#live-motion)).
- `FollowStatus::Stopping { contacts }` numbers obstacles as in the worlds of the tick the stop began at, on every tick of the stop: map them through those worlds, not the current ones ([live motion](docs/planning.md#live-motion)).
- A `Follower` on a tool path (`follow`) needs `FollowerOptions::replan` off, since a replan leaves the path; on a way joined into a move in, give it a `Detour` instead ([live motion](docs/planning.md#live-motion)).
