# Robots

`Robot::load(path, &RobotOptions)` loads a robot from URDF, MJCF or OpenUSD, builds its kinematic tree, and gives it a collision model of spheres, fitted to the links' geometry or loaded from a file you commit.

```rust,no_run
use batchplan::{CollisionModel, Error, Robot, RobotOptions};

fn main() -> Result<(), Error> {
    let options = RobotOptions {
        ee_links: vec!["tool0".into()], // the frame IK targets: one per hand
        ..Default::default()
    };
    let robot = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &options)?;
    println!("{} joints: {:?}", robot.dof(), robot.joint_names());
    // Fitting spheres takes a few seconds; save the model and load it next time.
    robot.collision_model().save("ur5e_spheres.json")?;
    let fast = RobotOptions {
        collision_model: Some(CollisionModel::load("ur5e_spheres.json")?),
        ..options
    };
    let _same = Robot::load("assets/ur5e/ur_description/urdf/ur5e.urdf", &fast)?;
    Ok(())
}
```

## File formats

`Robot::load` picks a loader by extension.

- **URDF.** `package://` paths resolve through `RobotOptions::package_dirs`, then through the directories above the file. URDF 1.2's `acceleration` and `jerk` limits are read when present. `Robot::from_urdf(text, &options)` loads a URDF from its text, as ROS publishes it on `/robot_description`. With no file to start from, its `package://` paths resolve through `package_dirs` alone, and relative mesh paths are an error.
- **MJCF (`.xml`, `.mjcf`).** A native parser covers what kinematics and collision need: `<include>`, `<default>` classes and `childclass`, the compiler's angle units (degrees unless told otherwise), mesh directory and Euler sequence, every orientation form, bodies with several joints, `fromto`, and `<equality><joint>` as mimic joints with their full polynomial. A named `<site>` on a robot body is a frame `RobotOptions::ee_links` may name for a hand, as a tool center point: the site's pose in its body, and a link of the robot only when it is named (a body of the same name keeps the name). `<contact><exclude>` pairs of robot bodies are pairs of links that never collide, added to the collision model's as an SRDF's are. Body subtrees with a joint are the robot; jointless bodies and world geoms are the scene (see [`World::load`](perception.md#scenes)). MJCF has no velocity limits, so `RobotOptions::max_velocity` (2 rad/s) applies.
- **OpenUSD (`.usd`, `.usda`, `.usdc`, `.usdz`; cargo feature `usd`).** Built on the pure-Rust `openusd` crates (0.7.0, pinned).
  - UsdPhysics rigid bodies are links. The tree grows from the world over revolute, prismatic, spherical and fixed joints, using `localPos`/`localRot` on both sides; joints authored from child to parent are flipped.
  - Stage units (`metersPerUnit`, centimetres when unset) and Y-up stages are converted. Revolute limits and angular velocity limits are degrees.
  - Mimic joints come from `NewtonMimicAPI` or `PhysxMimicJointAPI`; collision geometry from `PhysicsCollisionAPI` prims, instanced ones included.
  - Ghost-link `Xform`s (such as an end-effector frame) become fixed frames. `RobotOptions::variants` selects variants the file leaves unselected.

## Kinematics

- **Joints.** Revolute, continuous, prismatic and fixed joints load directly. Planar, ball and floating joints (URDF `planar`/`floating`, MJCF `free`/`ball`, USD spherical joints) become chains of one-axis joints. Translations become `<joint>_x`, `_y` and `_z`, whose limits come from `RobotOptions::joint_limits`. Rotations become `_theta`, or `_rx`, `_ry` and `_rz` (intrinsic x-y-z angles, singular where `_ry` reaches ±90°).
- **Continuous joints** turn without limit: planning takes the short way round, and interpolation and distances wrap (RRT-Connect included). For any joint whose range spans more than a turn (the UR5e's ±2π), a goal is reached at whichever equivalent (a whole turn away) is nearest or free; see [Planning](planning.md#goals-a-turn-away).
- **Mimic joints** follow their leader by a polynomial of degree up to four (URDF and USD mimics are linear, MJCF joint equalities quartic). The leader's range and velocity limit shrink so that every mimic joint stays within its own limits.
- **Closed loops** (MJCF `connect`, USD joints that close a loop) are solved when loading. Each loop's passive joints follow its one actuated joint by quartics fitted to their static equilibrium: the least spring energy that closes the loop within their limits, which is what MuJoCo settles to without contact. MuJoCo Menagerie's Robotiq 2F-85 loads as one joint whose pads track MuJoCo's settled positions within 0.2 mm. A loop must have exactly one actuated joint and close within 0.5 mm.
- **Locked joints.** `RobotOptions::lock_joints` fixes joints at a value, such as a gripper's fingers, or a torso and head that a mobile manipulator plans without. A joint that follows another (a mimic) is locked through the one that leads it: lock the leader, and the follower may be named with it at the value the leader gives it, as both fingers of a coupled gripper at one opening; named alone, or at another value, it is an error that names the leader. `robot.relock(&joints)` holds them at new values, as when the torso has moved, taking the same names, and `device.with_robot(&relocked)` gives a device for it that compiles nothing and keeps the uploaded worlds. A fitted model ignores the pairs that never touch across sampled configurations of the joints that were free when it was fitted, so the model records the joints locked then and their values (`CollisionModel::locked_joints`): a robot loaded with it must lock them there, and `relock` leaves them. For joints that will move, fit the model with them free and load with it locked (`RobotOptions::collision_model`). Models saved before 0.8.0 have no record, so refit those.
- **Per-joint limits.** `RobotOptions::joint_limits` replaces any joint's position, velocity, acceleration or jerk limit by name (`JointLimits`, each optional, as in MoveIt's `joint_limits.yaml`), mimic joints included.
- **Slower limits.** `RobotOptions::velocity_scale`, `acceleration_scale` and `jerk_scale` (each in (0, 1], like MoveIt's scaling factors) scale every joint's limits when the robot loads. Planning, timing and checks keep to the scaled limits, so a robot slowed this way also replans mid-motion at its own pace. Scaling velocity by `s`, acceleration by `s²` and jerk by `s³` makes every motion `1/s` times as long.
- **What a hand can reach.** `robot.hand_reach(hand)` is an outer bound on the poses the hand can reach (`HandReach`), allowing every revolute joint a full turn; `None` for a chain with a prismatic or relockable joint, or fewer than two moving joints. `excludes(pose, &ik_options)` is true only where the pose lies beyond the bound by more than IK's position and rotation tolerances, so IK need not try it. A pose it leaves open still needs IK.
- **Size.** Robots have no fixed size limit; a 40-joint, 41-link test robot with more than 128 spheres runs on both devices.

## Several arms, several hands

`Robot::assemble(&[Part], &RobotOptions)` makes one robot of several descriptions, such as two arms at a table. Each `Part` gives a description's path, a prefix for its link and joint names (`left_`, say) and the pose of its root link; the parts hang from one root link, `world`. Options name links and joints with their prefixes. `robot.parts()` lists the descriptions a robot came from, for a simulator to rebuild it.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::{Error, Part, Pose, Robot, RobotOptions};

fn main() -> Result<(), Error> {
    let arm = |prefix: &str, y: f32| Part {
        path: "assets/ur5e/ur_description/urdf/ur5e.urdf".into(),
        prefix: prefix.into(),
        base: Pose::new(Vec3::new(0.0, y, 0.0), Quat::IDENTITY),
    };
    let options = RobotOptions {
        ee_links: vec!["left_tool0".into(), "right_tool0".into()],
        ..Default::default()
    };
    let robot = Robot::assemble(&[arm("left_", 0.5), arm("right_", -0.5)], &options)?;
    assert_eq!((robot.dof(), robot.hands()), (12, 2));
    Ok(())
}
```

A robot has one hand per entry of `ee_links`; with none given, its one hand is the link with the most actuated joints above it. `robot.ee_pose(&q, hand)` is a hand's pose, and `robot.hand_joints(hand)` the joints that move it. Joints are numbered breadth first, so two arms' joints interleave; `hand_joints` tells them apart. IK gives each hand a target or none (see [Planning](planning.md#inverse-kinematics)), and plans move every joint at once, so two arms arrive together.

A hand's fingers, `robot.finger_links(hand)`, are the links with spheres below its palm (the first link at or above its frame with spheres); they may touch what the hand grasps. Where a finger is part of the palm's link, as the SO-101's fixed jaw is part of `gripper_link`, name the hand's fingers, the palm among them: `RobotOptions::finger_links` maps a hand's `ee_links` entry to them. `robot.tip(hand)` is how far the hand's spheres reach past its frame along its z axis, the depth the fingertips go beyond a grasp.

## Collision spheres

Planning checks the robot as a few dozen spheres against obstacles, which are exact primitives or distance grids. Spheres are what make thousands of configurations per millisecond possible on a GPU, as in cuRobo and VAMP.

- **Fitted spheres.** Without a collision model, spheres are fitted to each link's collision geometry by cuRobo's voxel method:
  1. Interior grid points become candidate spheres that touch the surface.
  2. A greedy cover picks the candidates that reach the most surface samples, each within the link's overhang beyond its radius. Samples no candidate reaches, on parts too thin for the grid, get candidates of their own: the largest sphere inside the geometry along the sample's inward normal that reaches it.
  3. A search finds one scale for the whole robot: the smallest at which the cover fits the budget (64 spheres by default) with every link's overhang its tolerance times that scale. `SphereOptions::tolerance` is every link's, unless `RobotOptions::sphere_tolerances` gives a link its own: on the SO-101 at 256 spheres, its links reach 8.9–12.0 mm past their geometry, and with its jaws held to 2 mm they reach 5.4–5.5 mm and the rest of the arm 11.7–13.6 mm (`links_given_a_tighter_tolerance_get_spheres_that_reach_less_far` fits both).
  4. Spheres grow until denser samples, one per 3 × 3 mm of surface (20,000 to 80,000 per link), lie inside them, so the model is conservative. A sphere would grow farther than twice its link's overhang only where the cover's samples lie far apart, so those dense samples join them and the search runs again. Where the budget reaches every tolerance, what it has left holds the spheres tighter: the least growth past each link's tolerance it affords, down to none. A stand's two boxes, 77 × 70 × 31 cm and 22 × 40 × 53 cm, held to 2 cm, reach 35.5, 24.1 and 20.0 mm at 384, 512 and 1,024 spheres (`a_budget_past_the_tolerances_holds_spheres_tighter` and `fitted_spheres_keep_each_links_tolerance_and_reach_less_as_the_budget_grows` fit them). Finding that growth takes a few more covers, so a fit at such a budget takes up to about three times as long: 0.13 s for the stand at 512 spheres, where the covers alone take 0.05 s, and 0.20 s for the Panda at 1,200, against 0.10 s.

  So no closed link's spheres reach farther than twice its tolerance times the scale, which falls as the budget grows: the SO-101's jaws held to 1 mm reach 8.6, 4.3 and 3.1 mm at 64, 256 and 512 spheres. Parts thinner than 0.1 mm, and vertices whose averaged normal leads out of the geometry, get no candidates of their own and may reach a little farther: at 2,048 spheres the Panda's base reaches 5.18 mm against a 5 mm tolerance. The model's `source` records how far its spheres reach beyond the geometry, the largest of each link's reach over its own tolerance, and the scale: with 64 spheres, 3.0 cm on the UR5e and the Panda, 1.9 cm on the SO-101 and 1.2 cm on the 2F-85. Fresh surface samples land at most about a millimeter outside an arm's spheres at the default budget: 1.05 mm on the UR5e's forearm and 0.24 mm on the SO-101, as `qualify` reports each link's `gap`. Parts thinner than the grid (`SphereOptions::voxel_size`, 1 cm) get candidates of their own, but a finer grid gives them more.
- **A limit on overhang.** `SphereOptions::max_overhang` holds every sphere to that far past the geometry, whatever the budget: `Some(0.002)` to 2 mm, `Some(0.0)` inside it. Spheres are chosen as without it, and each radius is then cut back to its center's depth plus the limit. The budget is kept, so what it cannot cover within the limit is left outside the spheres, and an obstacle can reach that deep into a link unseen: `qualify` reports it as the link's `gap`, about what the spheres reach past the link without the limit, less the limit. Lower `tolerance` toward the limit and raise the budget until the gap is one you accept; a limit the fit already keeps changes nothing. A sphere's overhang is measured from how deep its center lies inside, so the geometry must be closed: a link whose mesh has open edges is refused with their count, one with a shell that faces inward (a cavity's among them) with theirs, and so is a link too thin for a center inside it under a limit of zero ([how depth is measured](architecture.md#collision-spheres)). A collision model given with the options is held to the limit as a fitted one is: a sphere that reaches farther is refused by its link and index, with how far it reaches. What a robot holds takes its own limit (`AttachedObject::spheres`, `Held::spheres`).
- **Collision-model files.** `robot.collision_model().save(path)` writes the spheres, self-collision buffers and ignored pairs as JSON, with the joints locked while fitting (`locked_joints`), which a robot loaded with it must lock at the same values. Commit it, tune it by hand if needed, and load it through `RobotOptions::collision_model`. `to_json` and `from_json` give and take the same text where no file goes. The repository's Panda uses cuRobo's hand-tuned spheres this way (`assets/franka/panda_collision.json`).
- **Self-collision pairs** follow MoveIt's setup assistant, which skips adjacent links, pairs whose spheres already overlap at the default configuration, and pairs whose meshes never touch across 10,000 random configurations. `RobotOptions::srdf` adds the disabled pairs of a MoveIt SRDF, given as text, and an MJCF's `<contact><exclude>` pairs are added the same way.
- **Mounted robots.** Links that no joint moves (the base and anything fixed to it) are not checked against the world. They touch it the same way in every configuration, so a robot standing on its table is not in collision with it. They still collide with the robot's moving links. A moving link can touch the table in every configuration too, as the SO-101's shoulder turns just above its base: `world.always_touching(&robot)` finds such links by sampling, as MoveIt's setup assistant finds pairs that always collide, and returns the touches to add to the world.

How tight the spheres are is a trade-off you can measure: see [Spheres and meshes](benchmarks.md#spheres-and-meshes).

### Qualifying a robot

A robot that loads may still have spheres unfit to plan with. `robot.qualify()?` measures them against the geometry they cover. Per link with geometry and spheres (`LinkQualification`):

- `overhang`: how far its spheres reach beyond the geometry, at most;
- `uncovered`: the share of its surface outside every sphere, from fresh samples: up to about 2% for fitted spheres, in slivers about a millimeter deep where links are held tight, and more where a hand-tuned model leaves gaps;
- `gap`: how far the farthest of that surface lies outside the spheres: how deep an obstacle can reach into the link unseen. Up to about a millimeter for spheres fitted to an arm's links and two or three on a stand's wide boxes (2.5 mm at the default budget), and what a limit on overhang costs where one is set (`SphereOptions::max_overhang`), which at zero leaves the whole surface outside;
- `thickness`: the geometry's thickest point, across; a link thinner than its overhang is mostly sphere.

`tips` holds each hand's `tip`. Check that the fingers' overhang and tip leave room to grasp the smallest object you will pick and to keep off what it stands on (see [Picking up and setting down](planning.md#picking-up-and-setting-down)); that no link the camera sees is uncovered; and, with `device.contacts`, that the default configuration and the SRDF's named states collide with nothing but what the world lets touch. `robot.spheres(&q)` places every sphere at a configuration, to draw them or mask the robot from a camera, and `robot.reach()` bounds how far they reach from the base in any configuration.

## Holding objects

`robot.attach(&AttachedObject)` follows MoveIt: the object becomes a frame fixed to a link that collides with the world and the rest of the robot, never with its link or `touch_links`; `detach` takes it off. Its spheres cover all of it within its tolerance (`AttachedObject::spheres`, `Held::spheres`; 5 mm by default), however many that takes: the budget caps only the robot's own spheres. Spheres, boxes, capsules and cylinders are covered from their geometry in microseconds to milliseconds. Anything else, as a cylinder no rings of spheres cover, is fitted, in tens of milliseconds, once per object and device. A smaller tolerance fits closer with more spheres, each checked wherever the object is ([numbers](benchmarks.md#spheres-for-what-a-hand-holds), [how](architecture.md#collision-spheres)). Its own limit on overhang (`max_overhang` in those options) holds its spheres back: a box, a cylinder or a capsule is still covered whole within the smaller of the limit and the tolerance unless the limit is zero, and anything else may be left partly outside them, as `qualify` measures; a shape without thickness is refused under a limit of zero. `device.with_robot(&held)` gives a device for the new robot that shares the old one's GPU or threads and uploaded worlds, so picking something up needs no new device or upload. Setting it down touches the surface it rests on: a world whose [`touch`](perception.md#allowed-contacts) lets the object's link touch the table, uploaded to the holding device.

```rust,no_run
use batchplan::glam::Vec3;
use batchplan::{
    AttachedObject, Device, Error, Obstacle, Robot, RobotOptions, SphereOptions, World,
};

fn main() -> Result<(), Error> {
    let robot = Robot::load("assets/franka/franka_panda.urdf", &RobotOptions::default())?;
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&[World::default()])?;
    let held = robot.attach(&AttachedObject {
        name: "box".into(),
        link: "panda_hand".into(),
        shapes: vec![Obstacle::cuboid(
            Vec3::new(0.0, 0.0, 0.1),
            Vec3::splat(0.03),
        )],
        touch_links: vec!["panda_leftfinger".into(), "panda_rightfinger".into()],
        spheres: SphereOptions::default(),
    })?;
    let holding = device.with_robot(&held)?; // plans with the box, in the same `worlds`
    let _ = (holding, worlds);
    Ok(())
}
```

**A held object per problem.** `PlanProblem`, `IkProblem`, `ToolPath` and `Sequence` take `holding: Some(held)`, a `Held` as a pick leaves it (`TaskPlan::held`): the hand, the object in its own frame and the grasp. One call then takes problems holding nothing, different objects, or one object by different grasps, as the candidates a pick leaves each hold it by the grasp that got them through. It groups them by what they hold, runs each group as a batch of its own on a device holding it, in parallel, as `Held::attachment` attaches it (the link `"held by hand {hand}"`), and gives the results in the problems' order, as each group's device alone would. Pass the device without the object: one already holding a `Held` in that hand is an input error. Worlds uploaded to it serve every group, with the held object touching nothing unless a world says what it may touch: `Touch::held(obstacle, hand)` lets what that hand comes to hold touch an obstacle, as the table it is set down on, for one hand in an upload, and problems holding nothing are checked as without it. So objects held by many grasps are set down in one call against one upload, each problem naming the world whose touches fit it. A world uploaded to a device that already holds one in that hand names the held link itself, and serves problems holding in that hand or holding nothing. Each group is a batch of its own, so a GPU call pays its fixed cost once per group. A GPU device builds kernels for the robot holding something in a hand the first time a call holds one there, inside that call and its time budget; kernels are shared by robots of one shape, so warm up a device holding any object in each hand you will hold in, `device.with_robot(&robot.attach(&held.attachment(&robot)?)?)?.warmup()` (a hold by several hands, or one with touch links, has kernels of its own: below). A device keeps the spheres it fits to held objects (the latest 64), shared with the devices `with_robot` makes from it: fitting a cylinder or capsule takes milliseconds, and a call holding it would otherwise fit it again. A hold's `locked` (the fingers where they closed) relocks the robot that holds it; to attach a hold yourself, relock at `held.locked` first. A `Follower` carries a hold through a live motion the same way (`FollowerOptions::holding`, [live motion](planning.md#live-motion)). `device.contacts_holding` lists what a configuration touches holding one ([contacts](planning.md#why-a-problem-failed)).

**Held by two hands.** `Held::also` names the other hands that hold the object too, each by its grasp (`Grip { hand, grasp }`, the hand's IK frame in the object's frame), as two arms carry a tray or a pot by its handles. The object stays fixed to `hand`, may touch the links of every hand that holds it, and is rigid between them, which every call that holds it keeps:

- **`solve_ik`:** a target for one of the hands places the others at their grasps, and the result's problem has their targets. A target of its own for another of them must keep its pin (below), and is then taken as its grasp.
- **`plan`:** every state of the path keeps each hand at its grasp, as under a `HandConstraint` that pins it in `hand`'s frame ([constraints](planning.md#constraints-along-the-path), with a pin's floors: 2 mm along each axis of that frame, 0.02 rad of tilt and of twist). A pose goal for one hand is enough. A start, or a configuration given as the goal, with a hand off its grasp is an input error. The result's problems are as given.
- **`join` and a `Follower`:** blends and replans keep the hands at their grasps the same way, a blend at the states it is checked at ([the guarantee](planning.md#constraints-along-the-path)): between them it may stray past the pin by a fraction of a millimeter.
- **`follow`:** a path of one of the hands takes the others along, each kept where the start has it in that hand's frame, at every step and within the path's precisions between ([tool paths](planning.md#moving-the-hand-along-a-path)). The start must have them at their grasps, to what a pin allows, and the motion stops where one cannot keep up.
- **`pick` and `place`:** two hands pick an object up together by pairs of grasps (`Pick::also`), and set down what they hold together ([two hands](planning.md#two-hands)).
- **`handover`** passes what one hand holds, and refuses one held by two.

On a GPU, each set of hands that hold together has kernels of its own, built by the first call that holds so: warm up with that hold's own attachment, `device.with_robot(&robot.attach(&held.attachment(&robot)?)?)?.warmup()`.

Carrying it level is one more constraint, on `hand` (`tilt`). A hand that lets go leaves `also`; a hand that closes on what another holds joins it with the grasp it has then: with `object = robot.ee_pose(&q, held.hand).mul_pose(held.grasp.inverse())`, the grasp of hand `h` is `object.inverse().mul_pose(robot.ee_pose(&q, h))`.

**Other links that may touch it.** A hand that has let go still stands at the object, and a hold lets only the hands that hold it touch what is held. `Held::touch_links` names other links of the robot that may, by name, as MoveIt's touch links do: that hand backs off with its links there, `touch_links: robot.hand_links(hand).into_iter().map(String::from).collect()`, on the hold (a `follow` or a `plan` holding it so), and a hand that works on what another holds is let touch it the same way. An object set on what another hand holds is a place's ([`Place::onto`](planning.md#onto-what-another-hand-holds)). It allows the contact and nothing more: those links are kept nowhere. A name that is no link of the robot is an input error. On a GPU a hold's touch links give it kernels of its own, as hands that hold together do: warm up with that hold's own attachment.

```rust,no_run
use batchplan::glam::Vec3;
use batchplan::{
    Device, Error, Goal, Grip, Held, Obstacle, PlanProblem, PlanResult, Pose, Worlds,
};

/// Carries a bar that hand 0 holds by `left` and hand 1 by `right` to where hand 0 is at `goal`.
fn carry(
    device: &Device,
    worlds: &Worlds,
    start: &[f32],
    (left, right): (Pose, Pose),
    goal: Pose,
) -> Result<PlanResult, Error> {
    let bar = Held {
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.015, 0.16, 0.015)),
        grasp: left,
        also: vec![Grip { hand: 1, grasp: right }],
        touch_links: vec![],
        locked: Default::default(),
        spheres: Default::default(),
    };
    // Hand 0's goal pose alone: hand 1 follows from its grasp, and the path keeps it there.
    let carry = PlanProblem {
        holding: Some(bar),
        ..PlanProblem::new(0, start, Goal::Poses(vec![Some(goal), None]))
    };
    batchplan::plan(device, worlds, &[carry], &Default::default())
}
```

## Checking against meshes

Spheres only approximate the links. `MeshModel::try_new(&robot)` loads each link's collision meshes, and `clearance(&worlds, &item_world, &q)` measures a batch of configurations' world and self clearance on the meshes, as MoveIt checks a planning scene with FCL. It runs on the CPU, to check finished trajectories rather than inside planning:

- **Distances are exact** (parry3d's, found by branch and bound): about 110 µs per Panda configuration on an M4 Pro's 14 threads.
- **Meshes are surfaces**, as in FCL: a link entirely inside another closed mesh does not touch it.
- **Distance grids** hold no exact surface, so a link's clearance against one is the grid's distance at points covering its surface to within 5 mm, less 5 mm.
- Held objects are checked as part of the robot, and links no joint moves are skipped against the world, as in planning.

```rust,no_run
use batchplan::glam::Vec3;
use batchplan::{Error, MeshModel, Obstacle, Robot, RobotOptions, World};

fn main() -> Result<(), Error> {
    let robot = Robot::load(
        "assets/ur5e/ur_description/urdf/ur5e.urdf",
        &RobotOptions::default(),
    )?;
    let meshes = MeshModel::try_new(&robot)?;
    let wall = Obstacle::cuboid(Vec3::new(1.0, 0.0, 0.4), Vec3::new(0.02, 0.5, 0.4));
    let worlds = vec![World::new(vec![wall])];
    let q = robot.default_q().to_vec(); // one configuration; pass a trajectory's samples to check it all
    let clear = meshes.clearance(&worlds, &[0], &q)?[0];
    println!(
        "{:.3} m from the wall, {:.3} m between links",
        clear.world, clear.self_collision
    );
    Ok(())
}
```

## Robots in this repository

`assets/` holds test robots with their licences ([assets/README.md](../assets/README.md)):
- URDF: the Franka Panda (with cuRobo's hand-tuned spheres), the UR5e, the SO-101 and the Robotiq 2F-85 (one actuated joint driving five mimic joints);
- MJCF: MuJoCo Menagerie's Panda, UR5e and 2F-85;
- USD: newton-assets' UR5e and 2F-85, a Franka converted from our URDF with NVIDIA's urdf-usd-converter, and handwritten fixtures for each UsdPhysics convention.
