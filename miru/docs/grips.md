# Grips of what was seen

How to plan a grip on part of a measured thing, a turn of a control, a lift out of a container,
and a place onto what a hand holds. Each call's rustdoc has its full contract.

## Which object a point is on

`kept.labels_at(point)?` gives the label each camera reads at a point in the look's frame. A sole
`Label::Object(i)` names `kept.scene().objects[i].id`. No label means no reading supports the
point; two that differ mean the views disagree.

## A grip on part of an object

`observed.grip_worlds(&device, &step, &gripper, id)?` prepares the look once for grasps of object
`id`. Pass the gripper `grasps_on` used. `GripWorlds::plan` then plans an ordered list of grasps,
each in a world of its own where the fingers may touch only the measured part between them, and
gives the first `GripSearch::enough` that succeed, least cost first. The grasps are batchplan's
`grasps_on` of the object's readings, `kept.surface(&device, id)?`. A straight draw whose fingers
close on a point picked in a depth image:

```rust,no_run
use batchplan::glam::Vec3;
use batchplan::{Device, Grasp, Gripper, TaskOptions, Touch};
use miru::{GripOutcome, GripSearch, GripTarget, Observed, Step};

fn draw(
    seen: &Observed, device: &Device, gripper: &Gripper, id: u32,
    q: &[f32], grasps: &[Grasp], point: Vec3, outward: Vec3, distance: f32,
    mounted: &[Touch], should_stop: impl FnMut() -> bool,
) -> Result<GripOutcome, batchplan::Error> {
    let step = Step { at: &[q], ..Step::default() };
    let prepared = seen.grip_worlds(device, &step, gripper, id)?;
    let search = GripSearch { target: Some(GripTarget::Point(point)), ..GripSearch::default() };
    prepared.plan(grasps, &search, &TaskOptions::default(), should_stop, |c| {
        c.touches.extend_from_slice(mounted);
        c.pull(q, outward, distance)
    })
}
```

- `prepared.with_reach_bound()` rejects poses past the arm's reach before their worlds are built
  (`GripStop::OutsideReach`).
- Each `GripPlan` keeps its grasp, opening, locked finger joints and worlds. Execute against
  `GripPlan::world(stage)` by its obstacle indices: one support may fill several obstacles.
- A pick of a thing another object stands on is refused before any world is built
  (`GripPlans::blocked`).
- After a pick, `observed.picked(&plan.grip, &held)?` is the memory with the object in the hand.
- For a motion of your own, `prepared.project(&grasp)?` gives one grasp's world. Its grids are
  dense: drop each one you do not use before projecting the next.
- The worlds need the memory of the same look. After an act, look again.
- `Gripped::owner` names the object whose readings the fingers close on. A handle can have an id
  of its own even where it is fixed to a drawer: name the assembly yourself for an act on the
  drawer, and give the axis of a turn or the direction of a pull yourself.

## Turning a control

`kept.turn_grips(&device, id, &gripper, up, &shape_options, &surface_options)?` proposes grips to
turn a control, `up` being its outward normal. It gives groups to plan in order with batchplan's
`turn`, each with its `enough`: a raised bar over a round face where the readings show one,
then grips from above the fitted shape, then grasps of the measured surface. It also gives the
screw to turn about.

Plan against `seen.memory.world(&device, &step)?`, with `Turn::object` the obstacle `i` that stands
for the control (`planned.of[i] == Of::Object(id)`). A grip's pose is in the look's frame, and
`Turn::grasps` takes it in that obstacle's. With
`to_object = planned.world.obstacles[i].pose().inverse()`, each grip is
`Hold::new(to_object.mul_pose(grip.pose), grip.opening, &gripper)`.

Each grip carries what a hand closing there meets in the readings (`TurnGrip::evidence`).

## Out of a container

`kept.extractions(id, margin)?` gives each opening the look measured that the object's middle
stands over, with the straight lift that takes its lowest point `margin` over the lip. Where the
look measured none, `kept.enclosing(id, margin)?` names the boxes that hold the object in: look
closer before lifting. `kept.is_current(&memory)` says whether a memory still matches the look.

## Onto what a hand holds

`kept.surface_held(&device, id, &held, &q)?` is object `id`'s readings at the look as the surface
of what `held` holds with the robot at `q`, in the world's frame: what `rests_on` and
`falls_into` take for a place onto it (batchplan's `Place::onto`). Pass the configuration the
place starts from. Only where `held.hand` is there is read, so one configuration serves every
start whose holding arm is the same.

- **Planned ahead.** For a thing the look saw standing, the readings `kept.surface(&device, id)?`
  flags as its, carried from where it stood to where the hold has it: a pick of one thing, a pick
  of another and a place of the second on the first are planned from the one look taken before
  either pick. The frame is the room the thing's shape takes up, its box or cylinder, the frame a
  pick of it holds it in, whether the thing is solid, kept as its room, or a box open at the top.
  A thing the look gives in pieces carries the one piece's readings
  ([status](status.md#a-held-things-surface)).
- **Measured.** For a thing the look saw in that hand, the memory it was made with holding it
  there: the readings the look's mask gave the hold rather than the robot's links, moved with the
  hand from where it was at the look to where it is at `q`. They are where the thing was in the
  hand, slip included, as far as the stance's padding reaches past where the hold had it, and
  what rides on it (`Held::carries`) is part of them. A reading two holds' paddings both reach is
  neither's, and one near a known shape or a look's object and not near the hold's shapes, as
  what the thing was lifted off, is left out. The id of what rides on a held thing is refused,
  naming the thing whose surface its readings are part of.
- **None** where the look read nothing of the thing: look again.

```rust,no_run
use batchplan::{Device, Held, Place, RestOptions, RestProblem, TaskOptions, TaskPlan, Worlds};
use miru::Kept;

/// The plans that set what `cube` holds on what `plank` holds, object `id` of the look, from
/// `start`, where both hands hold their things.
fn onto_the_plank(
    kept: &Kept, device: &Device, worlds: &Worlds, id: u32,
    plank: &Held, cube: &Held, start: &[f32],
) -> Result<Vec<TaskPlan>, batchplan::Error> {
    let Some(surface) = kept.surface_held(device, id, plank, start)? else {
        return Ok(vec![]); // the look read nothing of it
    };
    let problem = RestProblem::new(cube.object.clone(), 0);
    let rests = batchplan::rests_on(device, &[problem], &[surface], &RestOptions::default())?;
    let place = Place {
        onto: Some(plank.clone()),
        ..Place::new(0, start, cube.clone(), &rests[0])
    };
    let mut placed = batchplan::place(device, worlds, &[place], &TaskOptions::default())?;
    Ok(placed.outcomes.remove(0).unwrap_or_default())
}
```

## What the evidence covers

A grip is checked against the latest readings only, and says nothing of what no camera saw. Plan
each motion against the current world and guard it in execution.
