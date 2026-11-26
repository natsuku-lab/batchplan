# Turning and pulling

Two tasks move what the hand grips without lifting it: `turn` turns it about a line, and `pull` draws it along one. Both build on `pick`'s steps ([picking up and setting down](planning.md#picking-up-and-setting-down)) and take grips from `grasps`, or from `grasps_on` for what cameras see ([grasps](grasps.md)).

## Turning a thing about an axis

`turn(&device, &worlds, &[Turn], &TaskOptions)` turns something about a line in one call: a knob, a cap, a lever, a valve's wheel, a door's handle. It runs as a pick does, a candidate per grasp, with a turn between the move in and the move away: the hand follows its grasp carried about a `Screw` by the task's angle, the fingers closed on the object, then lets go and backs off.

```rust,no_run
use std::collections::HashMap;

use batchplan::glam::Vec3;
use batchplan::{
    CollisionModel, Device, Error, GripperCalibration, Obstacle, Pose, Robot, RobotOptions, Screw,
    ShapeGraspOptions, TaskOptions, Turn, World,
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
    let hand = robot.gripper(0, &GripperCalibration::new(0.08, 0.0, HashMap::new()))?;
    let o = ShapeGraspOptions::default();
    let grasps = batchplan::grasps(&arm.placed(Pose::IDENTITY), &hand, &o)?;
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

- **The line.** `Screw { point, direction, pitch }` is a line in the world's frame and how far what turns about it moves along it for each radian (meters): 0 for a knob, a wheel or a lever; for a cap on a thread of 5 mm a turn, 0.005 / 2π, which turned right-handed about a `direction` up (anticlockwise seen from above) rises as it unscrews. `Screw::about(point, direction)` has no pitch. A task without one (`Turn::screw` `None`) turns about the hand's own z axis through its IK frame at the grasp: a turn in place, as a knob gripped on its axis turns. `screw.carry(pose, angle)` carries a pose about it, and `screw.path(pose, angle, precision)` gives the waypoints a `ToolPath` follows along the arc or helix, as few as keep the straight moves between them within `precision` of it, and no more than a quarter turn apart.
- **Steps.** Each step is a batch for every candidate of the call:
  1. The start, checked for collision.
  2. IK at the grasp, the fingers free to touch the object: every configuration it finds, nearest the start first, not only the nearest.
  3. The turn, followed from each of those configurations, the hand holding the object as the grasp does, the hand kept within the tool path's precision of its true arc: half of it for how far the straight moves between the screw's waypoints sag, half for how far the hand strays from them. A grasp none of whose configurations makes the whole turn stops here.
  4. The straight move in, run backward from the grasp as a pick's, from each configuration that made the turn.
  5. The straight move away, `approach` back along the hand's z axis from where the turn ends, the object where the turn leaves it.
  6. The planned path from the start to where the move in begins, for the configuration nearest the start that got through steps 3 to 5.

  Run the transit and the approach, close the gripper, run the turn (`TaskPlan::turn`), open the gripper and run the departure, then put `TaskPlan::placed` (the object carried through the angle) in the world in its place. `TaskOptions::min_fraction` applies to the moves in and away; the turn is made whole or not at all. The hand keeps under `TaskOptions::tool_speed` all along the turn, which it holds the object through. `enough`, `transit_world`, `transit` and `constraints` are a pick's, and the fingers are each hold's (`Hold::locked`).
- **Which configuration.** A grasp is one pose of the hand and many configurations of the arm, and the one nearest the start, which a pick takes, often runs into a joint limit partway through a long turn where another at the same grasp, the wrist over the other way, has the room. So every configuration IK finds at each grasp is followed through the turn, all of a call's in one batch of tool paths, and of those that make it the nearest the start is the plan's. [Benchmarks](benchmarks.md#turning-things-about-a-line) counts the grips it saves on the example's knob.
- **What may touch.** Through the reach and the move in the fingers (`Robot::finger_links`, or the hand's links where it has none) may touch the object, and nothing else of the robot may. Through the turn the hand holds it, as a pick's lift holds what it picked up: the object is part of the robot, in the task's world without it, so what it sweeps is checked against everything else of the world and against the arm, and it may touch only what `Turn::support` names, what it turns in or on (a lever's mount, a cap's bottle, the panel a knob sits in). Name everything the object touches as it stands: a cap sits on its bottle all the way round, and a turn that does not name the bottle stops before it starts. Where the world holds the part that turns and what it turns in as one obstacle, as a knob a camera could not tell from its panel, set `Turn::stays`: then the obstacle stays where it is, the fingers alone may touch it, and nothing checks the part's sweep. The move away sees the object where the turn leaves it, or where it stood where it stays (`TaskPlan::placed`), the fingers free to touch it; everything else of the world is an obstacle throughout.
- **Why it failed.** `TaskStep::Turn` comes after the reach and before the straight moves, and a candidate stopped there says how far the hand turned before it stopped (`CandidateStop::turned`, radians, signed as the task's angle) from the configuration that turned it farthest, and why: out of reach or at a joint limit (`Unreachable`), colliding (`Collision`, with what collides) or jumping. "none of 2 candidates got through: 2 turning (2 colliding)".
- **Grips for a turn from what cameras saw.** Read each camera with `Surface::from_depth`, the thing's readings flagged as `object` and everything around it kept in the surface, so the hand is kept clear of it; join the views. `grasps_on` with `narrow` on (the fingers come in as open as keeps them off what stands around) and `floor` at the top of what the thing stands on (the dial under a knob's bar, the table under a lever) gives grasps of any shape. Order them with `options.order_for_turn(&gripper, &mut grasps, Some(screw))`, the greatest moment about the measured axis first at the gripper's friction (pass `None` for each grasp's own z axis); it reads a one-sided grip's far contact, which no camera saw, so pass `shell: None` to turn by grips whose two sides were both seen. Make each a `Hold` as [for a pick](grasps.md#using-them-in-a-pick), and pass them to `turn` with what the thing turns on as its `support` and `enough` for how many plans will do. `SurfaceGraspOptions::near` puts the grasps nearest a point first within each of the tiers `grasps_on` orders them by, for a caller told where to grip.
  - **The moment** is what the two fingertips can give the thing about the line for each newton each presses with (meters): what turning it asks of the grip, as a grasp is judged by the wrench its task needs. `Grasp::moment` gives one grasp's, and its documentation says how it is found.
- **Cost.** Following every configuration through the turn is most of a call that tries every grip; `enough` stops it at the first plans. A GPU device takes about what the CPU device does: the turns are followed on the CPU's threads on either device ([Benchmarks](benchmarks.md#turning-things-about-a-line)).

## Drawing a handle along a line

Use `Pull::new(world, start, object, grasps, direction, distance)` for a drawer. The grasps are in the contact obstacle's frame. The unit direction is in the world frame, and the signed distance is in meters. Each grip keeps its orientation as it travels that distance. For a door, use [`turn`](#turning-a-thing-about-an-axis) with the hinge's point and direction.

Give `batchplan::pull` the device, uploaded worlds, a slice of pulls and the task options. Each pull tries every IK solution at each grasp through the draw, then plans the approach, release and transit as a turn does. The hand's fingers may touch the indexed obstacle through the approach, draw and departure. A distance grid can be the contact obstacle for a pull or for a turn whose obstacle stays. A task that names a grid takes its grasps in that grid obstacle's frame, including its placement. Give the approach and transit worlds matching obstacle indices.

Run the transit and approach, close the gripper, run `TaskPlan::pull`, open the gripper and run the departure. The fixture stays at its measured pose throughout planning. The arm is checked along its path. The moving drawer's swept volume is outside this model. A candidate stopped during the draw reports the signed meters planned in `CandidateStop::pulled` at `TaskStep::Pull`, with its reason and contacts. At an execution force stop the controller measures how far the handle actually travelled before updating the world.

Run `cargo run --release --example turns -- --fixtures` for a drawer and a hinged door grasped from two rendered depth images.
