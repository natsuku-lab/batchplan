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
- **MJCF (`.xml`, `.mjcf`).** A native parser covers what kinematics and collision need: `<include>`, `<default>` classes and `childclass`, the compiler's angle units (degrees unless told otherwise), mesh directory and Euler sequence, every orientation form, bodies with several joints, `fromto`, and `<equality><joint>` as mimic joints with their full polynomial. Body subtrees with a joint are the robot; jointless bodies and world geoms are the scene (see [`World::load`](perception.md#scenes)). MJCF has no velocity limits, so `RobotOptions::max_velocity` (2 rad/s) applies.
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
- **Locked joints.** `RobotOptions::lock_joints` fixes joints at a value, such as a gripper's fingers.
- **Per-joint limits.** `RobotOptions::joint_limits` replaces any joint's position, velocity, acceleration or jerk limit by name (`JointLimits`, each optional, as in MoveIt's `joint_limits.yaml`), mimic joints included.
- **Slower limits.** `RobotOptions::velocity_scale`, `acceleration_scale` and `jerk_scale` (each in (0, 1], like MoveIt's scaling factors) scale every joint's limits when the robot loads. Planning, timing and checks keep to the scaled limits, so a robot slowed this way also replans mid-motion at its own pace. Scaling velocity by `s`, acceleration by `s²` and jerk by `s³` makes every motion `1/s` times as long.
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

## Collision spheres

Planning checks the robot as a few dozen spheres against obstacles, which are exact primitives or distance grids. Spheres are what make thousands of configurations per millisecond possible on a GPU, as in cuRobo and VAMP.

- **Fitted spheres.** Without a collision model, spheres are fitted to each link's collision geometry (or its visual geometry, with `SphereOptions::geometry`) by cuRobo's voxel method:
  1. Interior grid points become candidate spheres that touch the surface.
  2. A greedy cover picks the candidates that reach the most surface samples.
  3. A search finds one overhang for the whole robot: the smallest at which the cover fits the budget (64 spheres by default).
  4. Spheres grow until 20,000 surface samples per link lie inside them, so the model is conservative.

  The model's `source` records how far its spheres reach beyond the geometry: with 64 spheres, 3.0 cm on the UR5e and the Panda, 1.9 cm on the SO-101 and 1.2 cm on the 2F-85. Fresh surface samples land at most 0.8 mm outside the UR5e's spheres.
- **Collision-model files.** `robot.collision_model().save(path)` writes the spheres, self-collision buffers and ignored pairs as JSON. Commit it, tune it by hand if needed, and load it through `RobotOptions::collision_model`. The repository's Panda uses cuRobo's hand-tuned spheres this way (`assets/franka/panda_collision.json`).
- **Self-collision pairs** follow MoveIt's setup assistant, which skips adjacent links, pairs whose spheres already overlap at the default configuration, and pairs whose meshes never touch across 10,000 random configurations. `RobotOptions::srdf` adds the disabled pairs of a MoveIt SRDF, given as text.
- **Mounted robots.** Links that no joint moves (the base and anything fixed to it) are not checked against the world. They touch it the same way in every configuration, so a robot standing on its table is not in collision with it. They still collide with the robot's moving links.

How tight the spheres are is a trade-off you can measure: see [Spheres and meshes](benchmarks.md#spheres-and-meshes).

## Holding objects

`robot.attach(&AttachedObject)` follows MoveIt: the object becomes a frame fixed to a link. It collides with the world and the rest of the robot, but never with its link or the `touch_links` that hold it; `detach` takes it off. Spheres are fitted to its shapes, except that boxes and spheres are covered directly, in microseconds. A sphere is itself. A box splits into a grid of cells, each covered by the sphere through its corners, so that the spheres overhang the box by at most the tolerance (or the least the budget allows). `device.with_robot(&held)` gives a device for the new robot that shares the old one's GPU or threads and uploaded worlds, so picking something up needs no new device or upload.

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

## Checking against meshes

Spheres only approximate the links. `MeshModel::new(&robot)` loads each link's collision meshes, and `clearance(&worlds, &item_world, &q)` measures a batch of configurations' world and self clearance on the meshes, as MoveIt checks a planning scene with FCL. It runs on the CPU, to check finished trajectories rather than inside planning:

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
    let meshes = MeshModel::new(&robot)?;
    let wall = Obstacle::cuboid(Vec3::new(1.0, 0.0, 0.4), Vec3::new(0.02, 0.5, 0.4));
    let worlds = vec![World {
        obstacles: vec![wall],
    }];
    let q = robot.default_q().to_vec(); // one configuration; pass a trajectory's samples to check it all
    let [world, self_collision] = meshes.clearance(&worlds, &[0], &q)?[0];
    println!("{world:.3} m from the wall, {self_collision:.3} m between links");
    Ok(())
}
```

## Robots in this repository

`assets/` holds test robots with their licences ([assets/README.md](../assets/README.md)):
- URDF: the Franka Panda (with cuRobo's hand-tuned spheres), the UR5e, the SO-101 and the Robotiq 2F-85 (one actuated joint driving five mimic joints);
- MJCF: MuJoCo Menagerie's Panda, UR5e and 2F-85;
- USD: newton-assets' UR5e and 2F-85, a Franka converted from our URDF with NVIDIA's urdf-usd-converter, and handwritten fixtures for each UsdPhysics convention.
