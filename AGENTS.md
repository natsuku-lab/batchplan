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
- **Hands** are the IK frames of `RobotOptions::ee_links`, numbered in that order. For a gripper, use a frame between the fingertips, z out of the hand and y along the closing direction: what `grasps` assumes.

### Lifecycle

- **Robot.** `Robot::load` fits collision spheres unless given a model: a fraction of a second for an arm at the default budget, seconds for thousands of spheres; save `robot.collision_model()` and pass it back as `RobotOptions::collision_model` to load the same spheres every time.
- **Device.** One per robot (`Device::gpu`, falling back to `Device::cpu`), made once: never inside a control loop or per request. `device.warmup()` before timing anything ([devices](docs/devices.md#make-once-warm-up-reuse)).
- **Worlds.** `device.upload(&worlds)` once per scene. Uploads are cheap (0.1–0.2 ms for 512 tabletop worlds), so a scene that changes is uploaded again; there is no partial update ([changing a scene](docs/perception.md#changing-a-scene)).
- **Locked joints that move**, as a torso or head the robot plans without: `robot.relock(&values)` then `device.with_robot(&relocked)`, which compiles nothing and keeps the uploaded worlds. Fit the collision model with those joints free and load with it: a model records the joints it was fitted with locked, whose pairs that meet only at other values would go unchecked, and `relock` refuses to move them ([robots](docs/robots.md#kinematics)).
- **Held objects.** `TaskOptions::held_spheres` sets the fit for a new hold. Pass the returned `Held` through lift, execution and placement. `holding: Some(held)` on a `PlanProblem`, `IkProblem`, `ToolPath`, `Sequence` or a follower's `FollowerOptions`, and `device.contacts_holding` for what a configuration touches holding one, with the device without the object: one call takes problems holding different objects or grasps, a batch per hold ([holding objects](docs/robots.md#holding-objects)). On a GPU, warm up a device holding any object in each hand you will hold in: the first call holding there builds its kernels. For a device that holds one, `robot.attach(&AttachedObject)` then `device.with_robot(&held)`: the same GPU or threads, and the uploaded worlds still work, unless they name what a hand comes to hold (`Touch::held`). The held robot's GPU kernels are its own: warm its device up too before timing it. `robot.detach(name)` reverses it. Attachment is collision only: there is no payload mass or inertia, so timing a heavy carry ignores the load; slow it with a speed scale ([trajectories](docs/planning.md#executable-trajectories)).
- **Choosing a hold.** For an object-wide surface pick, `prefer_measured_contacts` can reorder the completed plans before pairing with place. Pass the original grasp list. Keep a requested point’s priority when choosing a local grip ([tasks](docs/planning.md#picking-up-and-setting-down)).
- **Many grasps.** `grasps_on` gives hundreds, and a pick's time is their IK: set `Pick::enough` to the plans you will use (1 to run the first that gets through) and pass every grasp in the order they came, rather than cutting the list yourself, which drops the ones in reach ([tasks](docs/planning.md#picking-up-and-setting-down)). `Place::enough` does the same for a place given many poses, as every spot of a support: pass them all, those worth having first.
- **Picking up a body by one part.** `Pick::object` is the finger-contact index. `Pick::body` supplies a whole shape and the other pieces that depart. Keep the split world for the approach and use `Pick::lift_world` and the returned hold for the lift ([tasks](docs/planning.md#picking-up-and-setting-down)).
- **Gripping a specific reading.** Pass its measured 3D point in `GraspOptions::close_on` for additional approach angles and shallower insertions. The ordinary candidates still come first. Check `Grasp::closes_on` during selection, or pass the point to Miru's measured-grip search. [Surface grasps](docs/planning.md#picking-up-and-setting-down)
- **Finger openings.** `Pick::locked` (per grasp) and `Held::locked` hold the robot's locked joints (the fingers) at given values for that grasp or hold, many openings in one call; empty is as loaded ([tasks](docs/planning.md#picking-up-and-setting-down)). `grasps_on` with `GraspOptions::narrow` gives each grasp the `opening` to lock them at. If the loaded gap exceeds the graspable width, pass the loaded gap as `GraspOptions::opening` and the graspable width to `grasps_on_with_capacity`. For measured closing regions, `Robot::parallel_jaw_opening` checks whether the modeled jaws slide as assumed and gives the gap to use for both generation and projection. Open the gripper to the chosen plan's opening before its transit, and follow the transit and the move in on the robot relocked at `TaskPlan::held.locked`: on the device for the robot as loaded a live check sees the open fingers. Don't build a relocked device per opening for calls that take them. A place opens the hand as loaded and refuses a spot the open fingers do not fit; `Place::open` lists the openings it may let go at instead, widest first (`GraspOptions::openings()` gives the widths `narrow` chooses from): open the gripper to `open[plan.opening]` at the release and follow `depart` on the robot relocked there.
- **Live loops** call `Follower::tick` or `servo` each period and nothing that builds: pass the measured state to `tick` so a robot held back or pushed is followed from where it is (held there, `Blocked`, with `FollowerOptions::replan` off). A controller in another process takes each replan from `FollowStatus::Replanned { at, trajectory }`, with `FollowerOptions::lead` for the time sending it takes. `Stopping { contacts }` and `Collides { at, contacts }` name what is in the way, links and obstacles as `PlanFailure::StartCollides` names them ([live motion](docs/planning.md#live-motion)).

### Choose the call

| Task | Call | Guide |
|---|---|---|
| Forward kinematics | `robot.ee_pose(q, hand)`, `robot.link_pose(q, link)`, `robot.spheres(q)` | [robots](docs/robots.md) |
| Configurations for hand poses | `solve_ik`; `ik::failures` for why one has none | [IK](docs/planning.md#inverse-kinematics) |
| A collision-free path to a configuration or a hand pose | `plan`, then `time` (one path: `Trajectory::try_new`) | [planning](docs/planning.md#planning-a-path) |
| The hand along straight lines | `follow` | [tool paths](docs/planning.md#moving-the-hand-along-a-path) |
| A hand kept upright, in a box or relative to another hand | `HandConstraint` on the problem | [constraints](docs/planning.md#constraints-along-the-path) |
| Picking up and setting down | `grasps`, `pick`, `place` | [tasks](docs/planning.md#picking-up-and-setting-down) |
| Grasps of a measured object of any shape (a rim, a handle) | `Surface::from_depth` per camera, `join`, then `grasps_on` | [tasks](docs/planning.md#picking-up-and-setting-down) |
| Finger contact with a measured part, including a rim or handle | Prepare the kept look with `miru::Observed::grip_worlds`, then plan the ordered candidates. Use `with_reach_bound()` to reject poses outside the arm’s reach before building their worlds. Keep each winner’s world, opening and original candidate index together through planning and execution. One support can occupy several collision slots. | [miru](miru/README.md) |
| Where to set a held object down on a measured support (a plate, a cabinet's top, a rack's top) | `rests_on` with the support's points flagged, then `place` with the rests' `pose`s, every one, and `Place::enough` for how many plans will do; `rest_failures` for why one has none | [tasks](docs/planning.md#picking-up-and-setting-down) |
| Releasing a held object just above a measured rest | `Place::falls`, then use `TaskPlan::falling` in the departure follower world | [release above a measured rest](docs/planning.md#picking-up-and-setting-down) |
| Where to let a held object go into a measured container (a basket, an open drawer) | `falls_into` with the container's points flagged, then `place` with the falls' `pose`s | [tasks](docs/planning.md#picking-up-and-setting-down) |
| Turning a knob, a cap, a lever or a wheel | `turn` with the `Screw` it turns about (none: in place, about the hand's own axis) and what it turns on as `support`; grips from `grasps_on` ordered by `GraspOptions::order_for_turn` about that line, the fingers locked at each grip's `opening`, `GraspOptions::near` for where to grip | [turning](docs/planning.md#turning-a-thing-about-an-axis) |
| Drawing a drawer by its handle | `pull` with the contact obstacle's index, grasps in its frame, world direction and signed meters. The drawer stays in the planning scene. A force stop needs measured travel from the controller. | [drawing a handle](docs/planning.md#drawing-a-handle-along-a-line) |
| Passing an object from one hand to another | `pick` with `GraspOptions::along` (toward an end), then `handover` | [tasks](docs/planning.md#picking-up-and-setting-down) |
| A program of picks and places planned together | `pick` and `place` with `transit: false`, then one `plan` and `join` | [tasks](docs/planning.md#picking-up-and-setting-down) |
| Motions run without stopping between them | `join` | [joins](docs/planning.md#joining-motions) |
| A motion that comes fast and arrives slowly | `Trajectory::with_hand_speed` with a cap near the grasp (`HandSpeed::near`); many paths: `time`; in `pick`/`place`: `TaskOptions::tool_speed_near` | [timing](docs/planning.md#executable-trajectories) |
| Running a motion while the world changes | `Follower`, `servo` | [live motion](docs/planning.md#live-motion) |
| Obstacles from depth images or point clouds | `SdfGrid`, `OccupancyMap`, `DepthImage::filtered`, `DepthImage::robot_mask` | [perception](docs/perception.md) |
| What a camera's depth shows as objects: which pixels are one thing, a shape for each, what it stands on, the same id from look to look | `miru::look` with each camera's `DepthImage` and the fixtures as obstacles (`miru`, a crate of this repository, not published) | [miru](miru/README.md) |
| Which shape of a world each of a camera's readings lies on, within the camera's noise | `sdf::on_shapes` | [perception](docs/perception.md#shapes-the-caller-holds) |
| A perception library's labels and shapes, and what it needs back; keeping one memory of each thing between its objects and a fused map | `RobotMask::with_shapes` for what is held as a shape, `RobotMask::with_band` for what the camera reads behind its outline, `Surface::from_depth` with the labels as flags | [perception](docs/perception.md#with-a-perception-library) |
| One object as the cameras saw into it (an open drawer, a basket), space known empty or stale, and where a map is occupied | `OccupancyMap::part`, `free`, `clear`, `counts`, `occupied`, `take_surfaces` | [perception](docs/perception.md#what-a-caller-knows-and-one-objects-part) |
| Training data | `datagen`, `npy::export`, `lerobot::export` | [training data](docs/training-data.md) |

IK finds configurations; a plan finds the path to one, or to a pose, solving IK from the start itself. Solve IK first only to choose among its solutions. Every call takes many problems, each naming its world: batch them rather than calling once per problem, since each GPU call has a fixed cost.

### Qualify the robot

A robot that loads may still have spheres unfit to plan with. Before planning with a new robot or a hand-tuned collision model ([qualifying a robot](docs/robots.md#qualifying-a-robot)):

1. `robot.qualify()`: each link's `overhang` past its geometry, the `uncovered` share of its surface, how deep the `gap` it leaves and its `thickness`, and each hand's `tips`. Fingers whose overhang or tip is a large part of the smallest object you will grasp will collide with what it stands on; give them a tighter tolerance (`RobotOptions::sphere_tolerances`) or a finer `SphereOptions::voxel_size`. `SphereOptions::max_overhang` holds every sphere to a distance past the geometry instead, at the cost of coverage: read each link's `gap` after setting it, since an obstacle reaches that deep into the link unseen ([a limit on overhang](docs/robots.md#collision-spheres)).
2. `world.always_touching(&robot)` with the table the robot stands on: links that touch it in every configuration need a touch, or every plan's start collides.
3. Fingers that are part of the palm's link, as the SO-101's fixed jaw: name them in `RobotOptions::finger_links`, or grasps where they meet the object fail.
4. `device.contacts(&worlds, &item_world, &q, 0.0)` at the default configuration, the SRDF's named states and a few others: each should collide with nothing.

### Read results

- `ik.best(i)`, `plans.best(i)` and `solved()` return only valid answers; `IkResult::success` and `PlanResult::valid` are per seed, problem-major (`problem * seeds + seed`).
- A plan is B-spline control points, not a trajectory: time a call's paths together with `time`, or one with `Trajectory::try_new` (`Trajectory::moving` from a moving start), sample it with `at` or `sample`, and `check` it before it runs on hardware. Never send control points to a controller.
- With `start_tolerance`, use the corrected start retained in the result’s problem; goals and start motion remain as given ([measured starts](docs/planning.md#measured-starts-at-joint-limits)).
- Plans do not depend on the batch or the device; with a `time_budget`, they depend on the machine's speed. IK starts every problem from the same seeds in any batch, but a GPU rounds differently from the CPU, and differently on either side of where it switches how many invocations share a configuration (4,096 configurations, more for robots with many spheres), so solutions there can differ slightly, and so can plans to poses, which end where IK solves them.

### Diagnose a failed request, in order

Check these before raising budgets:

1. **The input.** An `Error::Input` names what was refused and the value given: shapes, joint order, unit quaternions, joint limits (strict by default; [measured starts can opt into correction](docs/planning.md#measured-starts-at-joint-limits)), frames. Options can be checked where they are set, before any request uses them: `IkOptions::check()`, and the same on `PlanOptions`, `RrtOptions`, `ShortcutOptions`, `ToolPathOptions` and `JoinOptions`, return the error their call would. They refuse what a call cannot run with, not what would take it long: how many seeds or iterations are too many for a cell is the host's to decide.
2. **The start.** `PlanFailure::StartCollides { contacts }` names each link and obstacle, or pair of links. A link touching the table in every configuration needs a touch (`always_touching`); a start inside a held object's spheres needs the object attached.
3. **The goal.** `PlanFailure::GoalCollides`, and for a pose goal `GoalUnreachable`, which more `PlanOptions::goal_ik` iterations may fix, or more seeds `kept` past its screening (`screening: 0` gives every seed every iteration); for IK, `ik::failures` tells `Unreachable` from `Collides { contacts }`.
4. **The search.** `NoPath`: more `rrt.max_iterations`, or the way is blocked. `NoSpline`: more `control_points`, or a slower start motion. `OutOfTime`: a longer `time_budget`.
5. **Tasks.** A `TaskFailure` prints where and why every candidate stopped; each `CandidateStop` has the step, the reason and the contacts, whose obstacles are numbered as in the task's world; in a place, `world.obstacles.len()` is the object where it is set down, and in a handover, the object where it changes hands. A turn's candidate stopped at `TaskStep::Turn` says how far it turned (`turned`) from the configuration at its grasp that turned farthest, and why: a joint limit there (`Unreachable`) is every configuration IK found running out of room, so give grips turned the other way about the axis. `grasps` says why it found none. Grasps near a support need `GraspOptions::tip` and `floor`, which keeps the open fingers over it too; from above, they keep at least `overlap` of the fingers (1 cm) in the object, so an object whose top is less than that over the floor has none.
6. **Rests.** No rest: ask `rest_failures`, which says how near the search came. `Blocked` names the point in the way; `Tips`, how far it can tip (ask less `tipping`, or set it down elsewhere); `Unheld`, where it touched and how the support pushes it (on a slope, give the rotation that lies flat on it, or flag the stop at its foot); `Unsupported`, that nothing flagged lies under it. A `blocked` point that is not the support's on `Unheld` or `Tips` says the flags leave out part of what the object would lie against, as a rack's lip measured as another object. `Tips` against a lip or a rim may hide a rest in a strip narrower than the spacing: ask `RestOptions::look_closer`. Few rests or none on a top a camera above it read with noise may be its fitted normals facing down there: ask `RestOptions::seen_from_above_holds`. Pass `rest_failures` the options the rests were found with.

### Footguns

- glam's quaternion order is x, y, z, w; a w-first source passed straight through turns the hand.
- `PlanOptions::seeds` is 0: no trajectory optimization unless asked for.
- A world's touches name links; what a problem's `holding` may touch is named with `Touch::held(obstacle, hand)`, one hand's per upload, on the device that holds nothing.
- `holding` takes the device without the object: one already holding a `Held` in that hand is an `Error::Input`.
- A held object's spheres reach past it by up to their tolerance, 5 mm by default, however many spheres that takes (the budget does not cap them): a neighbor, or what it lies on, nearer than that reads as a contact. A smaller tolerance in `Held::spheres` or `TaskOptions::held_spheres` fits closer with more spheres, each checked wherever the object is ([numbers](docs/benchmarks.md#spheres-for-what-a-hand-holds)); leave room for where IK puts the hand (1 mm and 0.01 rad by default).
- `grasps` returns poses in the object's frame; with a `floor`, pass the obstacle as it stands in the world.
- A fused map keeps what it saw until a camera sees through it, three times for what one image saw and up to nine for what many did: what the robot itself moved is the caller's to tell it (`free` where it took something away, `clear` where something moved), and under `Occlusion::Occupied` the whole map's grid walls in all no camera has seen; take `part` of what must be seen into.
- A map weighs each image as evidence by default, so an object put where cameras saw empty space takes up to three looks to fill it. A caller that looks once every few seconds and acts on each look wants `map.with_weighing(Weighing::Latest)`, under which each look decides what it sees and one stray reading is an obstacle until a look sees through it ([fused maps](docs/perception.md#fused-maps)).
- A real depth camera reads points in mid-air along every edge, between a surface and what is behind it: pass its depths through `image.filtered(&DepthFilter { noise, .. })` before anything else reads them, with the camera's noise. Without the noise, readings of flat surfaces that noise sets apart go too; depth a simulator renders exactly needs no filter.
- `robot_mask` leaves out what a hand holds only where it is named (`holding`); an object in the hand that is not named reads as an obstacle there. It judges a reading by its point, so a camera on the hand works, and a sensor's stray readings farther than the padding behind the arm's outline stay obstacles.
- The filter keeps a reading between two surfaces that lies near one of them, and a camera reads the pixels beside a depth jump toward the other surface: beside the arm or a shape given to `with_shapes`, those lie in mid-air past the padding or the margin, where fingers go. `mask.with_band(&device, &image, margin, pixels)`, after `with_shapes` and with its margin, takes a reading where a flagged reading within `pixels` is of a thing (a sphere, a held object, a shape) whose surface lies nearer than the reading by more than the margin and the reading's ray, carried on past it, does not enter that thing. So a table given as a shape takes nothing of what stands on it, at any band, and a surface level with a flagged one keeps its readings within the margin. `pixels` is how many the camera reads beside a jump (one for a sensor that samples disparity a fraction of a pixel off, half the window for a block matcher), not a number to tune. Given up: an outline read too near stays an obstacle, and the honest readings just behind each flagged outline go with the rest ([shapes the caller holds](docs/perception.md#shapes-the-caller-holds)).
- `grasps_on` closes only on patches seen from both sides by default: a bowl seen only from above has none. `GraspOptions::shell` (1.5 cm suits a bowl's rim) also closes on a wall seen from one side, where the cameras saw past it; those grasps come after the rest and are marked `one_sided`: their `width()` is the most the wall can be, not what the fingers will close to. A leaning wall is closed on level too, the hand straight down across a rim, only where `friction` puts the level line inside the cone: pass the pads' friction, not the default 0.5. They need where each point was seen from (`Surface::seen_from`, which `from_depth` and `from_points` record), so a `Surface` built by hand gets none without it, nor a cloud from a moving sensor, each point seen from a place of its own. `from_points` reads its `seen_from` as one camera's place, the space to every point empty: a cloud fused from several views must not be given a shell ([grasps of what was seen](docs/planning.md#picking-up-and-setting-down)).
- `rests_on` finds a rest as the object is set down, straight down at the `rotation` given: nothing is tipped or slid into place. On a slope, give the rotation that lies flat on it. Where you have a place in mind (the middle of a plate, a point an agent gave), pass it as `RestProblem::near`: a rest narrower than the spacing, as a wide flat bottom on a dished plate, is found only where the spots are set closer. `RestOptions::look_closer` (off by default) also looks closer where the object nearly rests, for a strip narrower than the spacing (an upright bottle against a rack's lip): about the place nearest `near`, or without it, only where the search finds nothing else, the middle of the support; such a strip is found near one place, not along its length. `RestOptions::seen_from_above_holds` (off by default) lets a support point whose fitted normal faces down hold where a camera above it saw it, for a top read with noise at a slant. Only a can or a capsule laid on its side moves: it rolls up to half the spacing and `touching` more into a cradle (between bars, in a groove, against a lip) and has no rest on a flat support. Flag every point the object may lie on (a rack measured as six pieces has rests across them only with all six flagged), and expect none under anything unflagged, however high over the spot it is; give the object its true shape (a round bottle given as a box rests where a round one rolls off), and pass every rest to `place`: they come at each turn, a lying bottle end for end too ([rests on what was seen](docs/planning.md#picking-up-and-setting-down)).
- "Into" a container is `falls_into`, not `rests_on`: a basket's rests are on its rim, and a can set down there is where it was put but not in the basket. Which one is the task's to say (LIBERO's goals say `in` or `on`); don't try one and fall back to the other. `falls_into` gives nothing for a tray, a plate or a table, nothing where neither the container's floor nor a wall sloping in under the object was seen that deep, and nothing for an object under about 1.6 cm across at the default spacing. A container the object barely fits (a box in a bowl a little wider than its diagonal) holds it in only within millimeters of its middle, which spots a centimeter apart can miss: ask `RestOptions::look_closer`, which looks closer about `near` (the container's middle, if you know it) or, where the spots find nothing, about the hollow's lowest point ([letting go into what was seen](docs/planning.md#picking-up-and-setting-down)).
- A task's world holds for all its steps unless `transit_world` or `depart_world` name others: what the hand may touch at the grasp, it may touch all the way from the start.
- `TaskPlan::transit` is `None` for a task with `transit: false`, whose start is a seed and is not checked.
- A turn holds its object through the turn and checks what it sweeps: name what it turns in or on in `Turn::support` (a cap's bottle, a lever's mount), or a turn of a thing that touches it stops at `TaskStep::Turn` before it starts. Where the world holds the turning part and its housing as one obstacle (a knob a camera could not tell from its panel), set `Turn::stays`: the obstacle stays where it is, the fingers alone may touch it, and nothing checks the part's sweep ([turning](docs/planning.md#turning-a-thing-about-an-axis)).
- `Robot::lower` and `upper` are ranges; continuous joints turn without limit and wrap.
- `Follower::tick` with `None` trusts the trajectory; on hardware, pass the measured state.
- `FollowStatus::Stopping { contacts }` numbers obstacles as in the worlds of the tick the stop began at, on every tick of the stop: map them through those worlds, not the current ones.
- A `Follower` on a tool path (`follow`) needs `FollowerOptions::replan` off: a replan leaves the path for a free-space motion to its end. Off, a collision ahead stops it on the path and a robot past `tolerance` is held where it is, both `Blocked` at rest ([live motion](docs/planning.md#live-motion)). On a way joined into a move in, give it a `Detour` instead: it replans only the way, to where the move in starts, and takes the move in up from rest there (`Resumed`).
