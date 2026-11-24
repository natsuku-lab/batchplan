# miru

What depth cameras show, as objects: which pixels are one thing, what shape it has, what it
stands on, and which object is which from one look to the next. It is the perception a motion
library leaves out: [batchplan](https://github.com/natsuku-lab/batchplan) turns depth into a
collision world and plans on what the cameras saw; miru says what the depth shows.

miru is a crate of batchplan's repository, built on batchplan, which does not depend on it. A
view is batchplan's `DepthImage` with what a look needs besides, the shapes the world is known to
hold are its `Obstacle`s, an object's solids come back as obstacles, and a look runs on a
`Device`'s threads. Nothing in it knows a robot, a hand, a camera model or a table: the robot is
a mask per pixel, and a look reads nothing of its device's robot (`observe`, the look through
batchplan, reads it to mask it). It is not published to crates.io; a program takes it from the
repository with batchplan, at one revision.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::{DepthImage, Device, Intrinsics, Obstacle, Pose, Robot, RobotOptions};
use miru::{Look, LookOptions, Noise, Scene, View};

fn main() -> Result<(), batchplan::Error> {
    // the device the robot's motions are planned on: a look runs on its threads
    let robot = Robot::load("assets/franka/franka_panda.urdf", &RobotOptions::default())?;
    let device = Device::cpu(&robot)?;
    // meters along the camera's z axis; 0 where there is no reading
    let depth = vec![1.0; 640 * 480];
    let view = View {
        image: DepthImage {
            depth: &depth,
            width: 640,
            intrinsics: Intrinsics { fx: 600.0, fy: 600.0, cx: 319.5, cy: 239.5 },
            camera: Pose::new(Vec3::new(0.5, 0.0, 1.0), Quat::from_rotation_x(std::f32::consts::PI)),
        },
        color: &[], // rgb, three bytes a pixel, or none
        // the camera's depth noise from its datasheet: here a stereo camera's, of 640 pixels'
        // focal length, 5 cm between its imagers and matching to 0.08 of a pixel: 2.5 mm at 1 m
        noise: Noise::stereo(640.0, 0.05, 0.08),
        robot: &[], // per pixel, whether the reading is the robot's: `on_robot` of its mask
        masks: &[], // per pixel, the number of the thing a mask model says it shows, or none
    };
    // a known shape: the table, as the world the robot plans in has it
    let table = Obstacle::cuboid(Vec3::new(0.5, 0.0, -0.02), Vec3::new(0.6, 0.6, 0.02));
    let look = Look { views: &[view], known: &[table] };
    let scene = miru::look(&device, &look, &Scene::default(), &LookOptions::default())?;
    // scene.objects: id, shape (a box, an upright cylinder, or a box open at the top with the
    // room a look saw in it), whether that shape stands for the thing or only names the room it
    // takes up (solid), colour and what it stands on
    // shape.solids(): the boxes and cylinders a shape is made of, as obstacles
    // scene.labels[view][pixel]: Nothing, Robot, Known(shape) or Object(index): an object this
    // look saw is named by some pixel, one kept unseen by none; the flags Surface::from_depth
    // takes are scene.labels[view].iter().map(|l| *l == Label::Object(index)).collect()
    println!("{} objects", scene.objects.len());
    Ok(())
}
```

The next look passes this scene as `previous`. An object that stayed where it was keeps its id,
and what the look cannot see stays as that scene had it. A view cannot see an object where its
pixels are the robot's or have no reading, where they read something that stands before the
object, and where the object lies outside its image. An object that none of the look's continues
stays, in the scene with no label naming it, unless some view shows more of where it stood than
is hidden from it: reads there what lies on it or past it. One seen in part keeps the shape it had,
unless a view shows more of the rest than is hidden from it, or its readings lie outside that shape
by a window's worth, as when it moved or grew: then it is measured afresh. A thing whose shape stood
for it (`solid`) and that comes back in pieces with the robot's arm over it is one object still,
with its id: each piece lies in the solid of the shape it had and, in some view, at the robot's
pixels with the object that continues the thing, and no view reads more of where that shape
would show as something else than as them. A look with no views
gives the scene before with nothing seen. `Scene::next_id` is the id the next new object takes: a
caller that gives ids of its own raises it past them, and never lowers it. A look with more new
objects than a `u32` has ids left is refused.

With batchplan, one look is one call, `miru::observe`, and each thing is held in one memory: as
a shape or in batchplan's map of what the cameras read, never both. A `Memory` carries the
objects, what each hand holds and the map from look to look; what the robot did is said to it
once (`Memory::after`), and it gives the world to plan a step against, every obstacle named
(`Memory::world`):

```rust,no_run
use std::collections::HashMap;

use batchplan::glam::{Quat, Vec3};
use batchplan::{DepthImage, Device, Intrinsics, Obstacle, Pose, Robot, RobotOptions};
use miru::{Act, Camera, Memory, MemoryOptions, Noise, Of, Stance, Step};

fn main() -> Result<(), batchplan::Error> {
    let robot = Robot::load("assets/franka/franka_panda.urdf", &RobotOptions::default())?;
    let device = Device::cpu(&robot)?;
    // the cell's fixtures, known without looking: a table
    let table = Obstacle::cuboid(Vec3::new(0.5, 0.0, -0.02), Vec3::new(0.6, 0.6, 0.02));
    let memory = Memory::try_new(&device, vec![table], &MemoryOptions::default())?;
    // a frame as the camera read it, and what is known of the camera: facts, with no default
    let depth = vec![1.0; 640 * 480];
    let camera = Camera {
        image: DepthImage {
            depth: &depth,
            width: 640,
            intrinsics: Intrinsics { fx: 600.0, fy: 600.0, cx: 319.5, cy: 239.5 },
            camera: Pose::new(Vec3::new(0.5, 0.0, 1.0), Quat::from_rotation_x(std::f32::consts::PI)),
        },
        color: &[],
        noise: Noise::stereo(600.0, 0.05, 0.08), // its datasheet's
        band: 1,       // pixels it reads beside a jump in depth toward the other surface
        off_by: 0.005, // what its pose and the fixtures are off by
        masks: &[],
    };
    // where the robot stood when the frame was taken, and how far past its spheres it reads
    let q = robot.default_q().to_vec();
    let locked = HashMap::new();
    let stance = Stance { q: &q, locked: &locked, padding: 0.01 };
    let seen = miru::observe(&device, &[camera], &stance, &memory)?;
    // the world to plan a step against: the fixtures first, then each object
    let planned = seen.memory.world(&device, &Step::default())?;
    assert_eq!(planned.of.first(), Some(&Of::Known(0)));
    if let Some(object) = seen.memory.objects().first() {
        // its readings for grasps and rests, from the look kept
        let _surface = seen.kept.surface(&device, object.id)?;
        // what the robot did, said once
        let by = Vec3::new(0.05, 0.0, 0.0);
        let _pushed = seen.memory.after(&[Act::Pushed { id: object.id, by }])?;
    }
    Ok(())
}
```

These are the steps it takes, as batchplan's `docs/perception.md` ("With a perception library")
has them as a program; per camera:

1. **Filter the depth** (`image.filtered(&filter)?`, at the camera's noise): it drops most of what
   a stereo camera reads between a thing's edge and what lies behind it, which miru would take
   for the thing.
2. **Mask the robot** (`image.robot_mask(&arm, &q, padding, &holding)?`, the arm relocked with
   the stance's locked joints, at its padding, holding what the memory's hands hold): `View::robot`
   is its `on_robot`, one flag a pixel.
3. **Give the fixtures** as `Look::known`: the obstacles themselves.
4. **Look** (`miru::look`), with the memory's objects as `previous` and the corners as far as the
   robot reaches (`Robot::reach`). A camera's noise is raised there by what it is off by, a third
   for each of the three deviations a reading may lie from a shape and be on it: stated exact, a
   camera 5 mm off its pose makes the table an object of its own.
5. **Leave what is held as a shape out of the map.** The shapes are the fixtures and the solids
   (`Shape::solids`) of each object that is `solid`, and of no other, which come as obstacles: a
   box or a cylinder is itself, and a box open at the top, as a tray or a drawer pulled out, its
   floor and four walls, so that a hand reaches into its room between walls held as thin boxes.
   An object that is not `solid` is never passed as its solids, a box open at the top no more
   than a box (below): `robot.clone().with_shapes(&device, &image, &shapes, margin)?`, then
   `.with_band(&device, &image, margin, band)?` and `map.integrate(&device, &image, Some(&held))?`.
   `margin` is three times the camera's `Noise` (`[3 * constant, 3 * linear, 3 * quadratic]`) and
   what the shapes and the camera's pose are off by. The map's mask is padded past what a grid of
   it grows toward collision, 2√3 of its voxels (3.5 cm for batchplan's 1 cm), where the stance's
   padding is less; what an act told the map since the last look is fused first, and what becomes a
   shape this look, a solid object the memory had not as one, is freed in it first as far as a
   reading of it may lie, so that what earlier looks read of it is not held twice.
6. **Take grasps and rests on an object's own pixels** (`Kept::surface`): the view's labels that
   name it, as flags (`scene.labels[view].iter().map(|label| *label == Label::Object(object)).collect()`),
   go to `Surface::from_depth(&device, &image, &flags, Some(&robot), radius)`, the views joined.

For a pick out of an open container, `kept.extractions(id, margin)` gives each visible opening
that contains the body's measured footprint. Its lift takes the body's lowest point above the
lip by the requested margin, along the opening axis. When the body stands on the container,
floor contact is resolved to one single-precision step across the container height. An absent or
unseen target gives an empty list. Use `kept.is_current(&memory)` when planning from a memory that may
have advanced, and check the requested lift through the ordinary pick planner.

What the robot did is said once (`Act`): a pick puts the thing into a hand, which every mask then
holds, and frees the map where a thing that was not solid stood; what stood on it leaves the
objects with it and is in no memory until a look finds it ("Where it stands"). A let-go puts it
back with its id where it was set down, unseen, standing on what the caller set it down on, a
known shape or an object (`Under`, as its place plan's support), so that a push or a pick of that
carries it along until the next look measures it afresh; and that look says one of three of it:
seen there (among the objects, a label naming it), its place not seen (among them, no label), or
its place seen and empty (gone). A loss leaves it in no hand and no scene; a push moves the thing and
what stands on it; and a draw, as of a drawer, leaves the object as it was. Where a thing let go,
pushed or drawn is not solid, the map is never seen where it lies, and where a pushed or drawn one
stood. A pick of an id the memory never gave, above its next id, is a hold a restart brings back:
it is held, and a let-go makes an object of its box or cylinder. Acts tell the map nothing at
once: memories after different acts share it, and what each act says of it is fused when it is
next read.

Two things follow from step 5. What stands within the margin of a shape, or within about a
voxel of it, is taken for the shape, so a shape larger than its thing takes in what stands
beside it: on depth as a stereo camera reads it miru's shapes are larger, by the noise's reach
on each side (see "Where it stands"). And a box or a cylinder is an obstacle all through, so an
object that is not `solid`, as a bowl, a basket, or a cabinet with a drawer open that no look has
seen into, is not passed as a shape where a hand reaches into it or the thing acted on stands in
it (`on` says so): its shape would fill that room, and the map holds what was read of it. Kept
away from, its shape still covers it and is the cheaper obstacle; which of the two a step needs
is the caller's to say (`Step`): `Memory::world` gives its part of the map, never-seen space
solid in it, where the step names it in `into`, or where its room holds the robot at a
configuration of `at` or the hand at a pose of `reach`, and the room it takes up otherwise; and
last one grid of what the map holds that no shape and no room explains. The flag does not come and go with the view: an object that was not
solid stays so while it continues, unless a look sees into it and it is a box open at the top.
That one is solid, its room what the looks saw of the hollow: a wall whose inner face no view
shows is as thick as what hides the floor before it, toward collision, until a look sees in. It
keeps that room while it continues; where a look shows only pieces of it whose box cannot hold
that room, it is a box that covers them, and not solid, for it was open at the top. A box open
at the top that something rests in, its lowest reading inside the box's walls or floor, is no
shape that stands for the thing either: it is a box that covers it, not solid.

`examples/score.rs` computes no robot mask: a recorded look carries each view's as its executor
computed it, and its fixtures are the obstacles its run's `cell.json` names.

## Planning a grip on part of an object

To select an object at a point measured in the robot frame, call `kept.labels_at(point)?`.
Each camera contributes its label only when its filtered reading agrees with the point's
depth within the camera's noise and calibration allowance. A sole `Label::Object(i)` names
`kept.scene().objects[i].id`. An empty result has no supporting reading, and distinct labels
identify an ambiguous point. A pixel supplied with its originating view already has a label in
`kept.scene().labels`.

For a complete pick, pull or turn, prepare the current observation once and pass the ordered
grasps to `GripWorlds::plan`. The cell supplies its own finger-joint calibration, mounted
touches and motion options. For example, a straight draw can require that the fingers close
on the point selected in a depth image:

Call `with_reach_bound()` on the prepared source to check the arm's geometric reach before
projecting candidate worlds. A certified rejection reports `GripStop::OutsideReach` with the
original candidate index. Other candidates proceed in their supplied order. Arms with prismatic
or relockable joints along the hand's chain use the ordinary search.

```rust,no_run
use std::collections::HashMap;
use batchplan::{Device, Grasp, Touch, TaskOptions};
use glam::Vec3;
use miru::{GripOutcome, GripSearch, GripTarget, Observed, Step};

fn draw(
    seen: &Observed, device: &Device, hand: usize, id: u32, loaded_opening: f32,
    q: &[f32], grasps: &[Grasp], point: Vec3, outward: Vec3, distance: f32,
    mounted: &[Touch], finger_locks: impl Fn(f32) -> HashMap<String, f32>,
    should_stop: impl FnMut() -> bool,
) -> Result<GripOutcome, batchplan::Error> {
    let step = Step { at: &[q], ..Step::default() };
    let prepared = seen.grip_worlds(device, &step, hand, id, loaded_opening)?;
    let search = GripSearch { target: Some(GripTarget::Point(point)), ..GripSearch::default() };
    prepared.plan(grasps, &search, &TaskOptions::default(), should_stop, |c| {
        c.touches.extend_from_slice(mounted);
        let mut task = c.pull(q, outward, distance);
        task.locked = vec![finger_locks(c.candidate.opening)];
        task
    })
}
```

The search keeps `batch` candidate worlds in flight plus the requested winners. The first
`enough` feasible candidates in input order are returned quickest first, with their original
indices, finger openings and exact locked joint settings. `GripPlan::world(stage)` gives the collision world to follow each
motion against, with its touch permissions and obstacle names. Failed motions carry their own
obstacle names. When another object stands on a pick target, `GripPlans::blocked` names the
target and the objects it supports once, before any candidate world is projected. The result counts actual batched solver calls. A cancellation yields `GripOutcome::Interrupted`, including any work completed.
An approximate target can use `GripTarget::Near` with an explicit radius about the measured
contact midpoint, or `None` allows the whole surface.

When a solid target stands on a hollow object, the kept depth views also resolve that support
near the measured contacts. The stationary map excludes fresh returns labelled as the target
and covered by its fitted solid. Returns outside the fit, other objects and unseen material
still take part in collision checks. The rest of the support uses the coarse occupancy map.
Several obstacles can therefore name the same support. Keep all returned slots when following
a motion. The first projection in a new window builds the fine grid, and consecutive grasps in
the same window reuse it during the search.

For a custom motion, `prepared.project(&candidate)?` yields a single collision world or
`None` when there is no attributable contact part. These worlds contain dense distance grids:
drop each rejected projection before proceeding. Projection requires the memory of the same
look. After a simulated act, another split requires a new observation. The regular
`Memory::world` and unsplit picks remain available for planning from predicted memory. A very
large solid or local refinement can exceed batchplan's distance-grid size limit and return an
input error.

`Gripped.scene` starts with every collision enabled. Its `part` is the sole obstacle that the
fingers may touch while grasping; `grasp` is the hand pose in that obstacle's frame. A grasp on
the rim of a bowl leaves the opposite rim, the fingers' approach to it, other objects and any
unattributed cells subject to collision checks. The local room uses finer voxels near the hand:
current rays may resolve the contents of a coarse cell, while old material lacking a fresh
reading and space no camera saw through stay in the residual collision world. A hollow
object carries cells hit by its labelled pixels, including surface cells beside the central
ray of a pixel. A cell whose volume also meets a differently labelled reading stays in the
collision world throughout the lift.

For a pick, give `Pick` the result's `part`, `grasp` and `body`. A body is available only
when no other perceived object stands on the target. Its shape covers the whole object,
including any hollow interior while carried. The pick checks contact and computes the departure
world with the other indices the body vacates. After picking, call `observed.picked(&grip, &held)?`
with the returned hold. It carries the object's ID into the hand, preserves residual occupancy
in the old room and retains the identity of additional measured fixture cells. For a known
support, collect every obstacle with the fixture's `Of::Known(index)` for held-only contact.

A turn or pull uses the fixed world's `part` and leaves the rest in place. `Gripped.owner` names
the current object that supplied the contact readings. A handle can have its own perceived ID
even when mechanically attached to a drawer. The caller identifies the assembly for an act on
the drawer and supplies the axis of a turn or pull independently. An overlapping object or
unresolved wall can still prevent a grip.

## How it tells things apart

By geometry, at the scale the camera's stated noise allows; `src/segment.rs` has the
steps and their sources. In short: depth is compared as its inverse, where a flat surface is a
plane over the image and a stereo camera's noise is the same at every depth. Each reading gets
the plane of the window around it, where that window is one plane. Readings are cut apart where
the depth jumps, where two surfaces step, and where they meet in a concave crease, as things
that touch or stand on one another do (LCCP's convexity rule); they are kept together across
convex edges and curved surfaces. A stretch of surface that hides nothing and lies inside what
rises around it is that thing's floor, as a tray's or a ramekin's.

What lies on no plane, as a box's side seen edge on, a rim or a handle's post, is told apart by
jumps alone, and is a part of its own where it holds a window: where some reading's whole window
is its readings, so that it is as wide as a window that could have had a plane. One narrower,
however long, that lies along the one part beside it, more than half of its readings within a
window's width of it in the image, and does not stand in front of it, across a jump or a step,
along a window's width of their border, as a box's side seen edge on runs along its top and does
not stand in front of it, is that part's edge, and its readings go to it. One that only reaches
what it touches, as a pencil whose tip meets a box's corner, one that stands in front of it, as a
slab a box stands on or a rod leaning on the box, one with nothing beside it but a known shape,
as a rod alone on a table, and one between two things, whose readings the depth does not say are
whose edge, are parts of their own where they fill a window, as is a stretch of readings on a
known shape, which is the shape's or a thing's as a whole.

What lies on a shape the look knows is decided for a whole stretch of surface, not reading by
reading. Readings within three deviations of a known shape and readings on none are told apart
alike but never into one part. A part is the shape's where more than half of its readings lie on
known shapes and it stands in front of none of the shape: a surface cannot be in front of itself.
Of two readings on a shape's surface the noise does not say which is in front, so what such a
part stands in front of is what reads inside the shape: a board whose top reads on a mount known
to reach a centimetre over the table, where only the table is, stands in front of the table
around it, which reads inside the mount. A part that is the shape's goes to an object only where
more than half of it lies inside that object's fitted shape, so a plate keeps its floor a few
millimetres over the table, and a thing takes the readings at its foot that lie nearer its own
plane than they lie over the shape. The tolerances are derived from `View::noise`, the camera's
own account of its depth error by depth; the one number taken from the literature is LCCP's 10
degrees.

What nothing under it holds up is held up by what it touches. An object rests on what its
lowest reading lies on or in, the nearest straight under it of the known shapes and of the
shapes fitted to lower objects, but in a shape that is not solid (`Object::solid`) only on its
surface: the shape of a tray, a bowl or a cabinet with a handle joined to it names the room it
takes up, and says nothing of what it holds. Otherwise an object rests on what lies under its
middle, seen from above. It hangs where a view sees through the space under its middle, from
the height of its lowest reading down to what lies there, past what a shape that is not solid
holds: where the view sees past more of the points along it, a
pixel's width apart, than are hidden from it. A point is seen past where its pixel reads
farther than the point by more than three deviations of that reading, and hidden where the
pixel is the robot's, has no reading or reads something nearer, but for what the object itself
hides, which says nothing of what holds it up. So a bowl whose foot overhangs the box it stands
on rests on the box under its middle, which it hides, a box whose foot is hidden stands for all
the look can tell, and a handle that points at the camera hangs, though it hides most of what
lies under its own middle. One that hangs is one object with what it touches along the most
border with no jump in it, a window's width or more, unless a view's masks say the two are two
things: a handle with the front it stands out of, a thick-walled bowl's rim and a shelf's top
with what they rise from, and each of these, where it hangs in its turn, with what holds it. One
that does not hang, where no view sees under its middle, stands on the lower thing it touches
along the most border, a window's width or more, whose top is not above its lowest reading by
more than that reading's tolerance, unlike a taller box beside it: a bowl on a box narrower
than its foot stands on the box, and its shape reaches down to the box's top, not to the table.
Where a thing stands on another and the two hold a few of each other's readings, the lower ends
where the upper begins ("Where it stands").

What holds a hollow's floor among the walls that rise around it, and is square from above, is a
box open at the top (`src/fit.rs`, `room`). The box is turned as the least rectangle around the
floor is, for the walls stand along the floor's sides where a handle or a wall that is a thing of
its own turns the rectangle around the whole, and reaches around all its readings. A floor is a
part the hollow rule joined, or one a view sees into (below), more than half of whose planes face
up: of it, the readings no higher than its level by more than their slack, the level being the
height more than half of the part's readings lie at or under, so that the strips of a wall read
as the floor's edges are no floor. Its top is as high as a window's worth of those reach. Each
wall's inner face is where a window's worth of readings whose planes face into the room show it,
above the floor by their slack: at their innermost window's worth. Where none do, the room
reaches as far as a window's worth of the floor does, so a wall that hides the floor behind it is
as thick as what it hides. A hollow whose floor is round from above, or with a face more than
half of whose readings lie farther from it than their slack, is round, and stays a box that
covers it. A shape kept from the look before takes in the room a look shows in its frame: a wall
seen thinner is thinner, and one no look sees into stays. One measured afresh keeps the room it
had as well, its box turned as it was where the look does not open it; the room ends where the
look shows an inner face inside it, and is not kept where it does not lie in the box the look
measures, by more than the readings' slack: the look sees a piece of the thing then, or another,
or one turned apart from it.

A box a view sees into is open at the top as well, whether or not its floor is a thing of its
own that hides nothing (`src/look.rs`, `sees_into`). A solid box shows a camera above it nothing
under its top, so a part of a square object that faces up under walls of its own is a floor: the
object's readings higher than the part's level by more than their slack add up to a window or
more (fewer cannot be told from noise, as those that stray over a box's top on a camera's
depth); none of them, but for less than a window's worth, stands over the floor itself, inside
the least rectangle around it from above by more than its slack, which is what a room over it
would span (what rises out of the middle of a top, as a burner out of a stove's, or beside a
floor cut aslant, as the higher half of a box that rests tilted, leaves no room a box can hold);
and the floor takes up more than half of the object from above, by the least outlines around
the two (a notch in a thing, as between a cabinet's handles, leaves it the box that covers
it). So a drawer's floor seen from a hand held over it, with what little of its
walls the view takes in, is a floor with room over it and not a slab as tall as those walls;
and a step is open over its lower part, its riser the one wall.

What stands on the floor shows that the floor runs under it, and no wall stands where it does
(`src/look.rs`, `hold_what_stands`). A thing whose lowest reading lies in the box, on or over the
floor's top and nearer it than the box's top, and that stands in front of the floor in a view
along a window's width or more, widens the room over its outline from above, as far as the box:
a bowl the near wall hid the floor beyond, and so stood in, is in the room. A thing that rests on
the box with its lowest reading inside its walls or floor by more than its slack, after that,
stands where the box says nothing can: the box covers it instead, and is not solid.

A round hollow thing is put together again (`src/segment/round.rs`). A bowl's outside stands in
front of its inside along the near rim, with a jump in depth between them and a rim too thin to
show: the two are one part where they are round about one axis and end in one rim, unless the
view's masks say they are two things. The normals of a surface of revolution all meet its axis,
so one axis is fitted to the normals of both parts' planes, in closed form, as a linear line
complex (Pottmann, Hofer, Odehnal and Wallner, "Line geometry for 3D shape understanding and
reconstruction", ECCV 2004), each normal taken at the middle of the readings its window held,
where the two parts' planes fill more than five windows, one for each number the axis and its
pitch are fitted by. Three things are then asked. Only the third asks the planes for evidence:
a part with no plane faces neither way and stays apart. The first is a bound, which planes told
loosely pass easily and a part with no plane passes; the second is asked of the edges' readings,
which need no plane.

- **Each part's normals miss the axis by no more than its planes' own errors make of them**, at
  three deviations of the sum of their squares. Each part alone, not the two together: one with
  many planes, as a shelf's top, would hide what the other misses by.
- **The edge of the one over the other and the other's own edges lie on one circle about the
  axis**: more than half of each edge, a window's width at least, within three deviations of a
  reading's depth and half a window's width of the circle at the middle radius and height of
  their readings. Not the depth's deviations alone: a wall has a thickness and a pixel a
  footprint on a wall that leans, and the readings of the 4 mm test bowl's two edges spread over
  4 mm in radius and 2.5 mm in height where three deviations are 0.8 to 0.9 mm (measured in
  `Shell::one_circle` for the run); a rim wider than a window shows as a surface with planes of
  its own, so half of one is what the edges of one that does not may differ by. And not every
  reading: a part carries a few readings of what lies behind its rim.
- **The part in front faces away from the axis and the part behind toward it**: more than half
  of the planes each has, at three deviations of each. However few they are: an outside seen
  from above holds a window's worth of them or less, and asked for a window's worth that face
  its way as well (in `Shell::faces`, which no flag does), the rule matches a bowl over both
  cameras in 21 of 28 looks on `rows4` and 9 on `rows4-stereo`, where it matches 23 and 12.

## Masks from a model

Geometry tells readings apart where the depth says so and nowhere else. It cannot join the parts of
one thing that the depth shows no connection between, as a rack's slabs or a stove's base and its
knobs, and at a real camera's noise it loses some splits. An instance mask does both, and
`View::masks` takes one: per pixel, the number of the thing a mask model says the pixel shows, 0
where it says none, and empty for a view without a model. The numbers are plain data, from whatever
the caller ran: miru runs no model, links none and asks none for coordinates, and a number means a
thing in its own view only. `src/masks.rs` has the three rules. They act on the readings geometry
tells apart: the robot's, the known shapes' and those outside the look's corners keep their labels
whatever a mask says.

- **Split.** A part that geometry made whose readings lie in more than one mask, or in a mask and
  in none, is divided along them: the readings in no mask are a side as those in a mask are. A
  side other than the part's largest, the one that fills the most windows, is a part of its own
  where it fills a window, the least geometry makes a part of, and either is as wide as a window,
  some reading of it having its whole window in it (the sliver rule's test), or lies in a mask
  and either is more than half of all that mask covers in the view or the part's largest side
  lies in another mask. Every other side stays with the part's largest side. A model's masks are
  never exact to the pixel: a narrow side against readings in no mask is the fringe of a mask
  whose body lies elsewhere, the rim of a thing just outside its mask or of a neighbour or the
  table just inside it, unless it is all there is of a thin thing's mask; a narrow side beside
  another mask is a thing the model names beside another. A side cut off takes what its own
  readings hide and the creases they meet others in. A reading on a known shape in no mask, in a
  part most of which lies in one, is the shape's again.
- **Join.** The parts of a view more than half of whose readings lie in one mask are one object.
  Readings in no mask join nothing.
- **Apart.** Two parts of a view that lie so in two different masks are two objects, whatever
  else would make them one. A part in a mask and a part in none are not kept apart: where the
  depth joins them, by another view, as a hollow's floor or as what hangs and what holds it up,
  they are one. With one mask over a bowl on the cabinet in `libero-spatial-3-0` on
  `rows4-stereo`, the bowl hangs on stray readings and is one object with the cabinet, as without
  masks; with the truth as masks it is not. Keeping such parts apart as well breaks what a mask
  that misses part of a thing still joins (a tray's floor its mask leaves out, a handle outside
  the mask of the box it stands out of) and leaves a mask a few pixels short in pieces.

A set of parts lies in the masks of every part joined to it so far, one a view at most: a part in
no mask joined to one in a mask carries that mask into the joins after.

The rest of a look is as without masks, as far as the three allow: parts of two views that show
one surface are one object, taken from the two that share the most surface down, so that a part
which would join two things of another view is one with the thing it shares most with; a
hollow's floor is its walls'; and shapes, supports and ids are made of the result. Where two
views' masks differ, each holds in its own view. With every view's masks empty, or every number
0, a look is geometry's alone.

A model's masks are never exact to the pixel. `cargo run --release --example score -- <set>
--masks truth|one --border <pixels>` moves every mask's border in or out by that many pixels (with
one mask, `--masks one`, each true object in view gets a look of its own whose masks are its
pixels alone, as a model prompted for one thing gives; a mask grows over its neighbours too, and
with the truth as masks over what shows none). Matched / merged / in pieces / missed of the 140
over both cameras:

| Set | Masks | 3 px short | 1 px short | exact | 1 px over | 3 px over |
|---|---|---|---|---|---|---|
| `rows4` | the truth | 126 / 0 / 14 / 0 | 133 / 0 / 7 / 0 | 135 / 0 / 5 / 0 | 135 / 0 / 5 / 0 | 135 / 0 / 5 / 0 |
| | one | 126 / 0 / 14 / 0 | 133 / 0 / 7 / 0 | 135 / 0 / 5 / 0 | 135 / 0 / 5 / 0 | 134 / 4 / 2 / 0 |
| `rows4-stereo` | the truth | 116 / 0 / 23 / 1 | 135 / 0 / 5 / 0 | 134 / 0 / 6 / 0 | 134 / 0 / 6 / 0 | 134 / 0 / 6 / 0 |
| | one | 107 / 14 / 18 / 1 | 124 / 9 / 6 / 1 | 129 / 7 / 4 / 0 | 129 / 7 / 4 / 0 | 127 / 12 / 1 / 0 |
| `rows4-stereo-filtered` | the truth | 121 / 0 / 19 / 0 | 134 / 0 / 6 / 0 | 134 / 0 / 6 / 0 | 134 / 0 / 6 / 0 | 134 / 0 / 6 / 0 |
| | one | 116 / 9 / 15 / 0 | 130 / 3 / 6 / 1 | 132 / 2 / 5 / 1 | 133 / 2 / 4 / 1 | 131 / 6 / 3 / 0 |

How that compares with Split before it asked a side's width, and what it leaves, is in "Where it
stands".

A model that needs prompts, points or boxes, takes two looks of one moment: look without masks,
take a prompt from each object's pixels (those labelled `Label::Object` with its index), run the
model, and look again with its masks, after the same `previous`.

## Where it stands

First looks of 20 LIBERO scenes, both cameras, scored against the simulator's body under each
pixel by the rules of batchplan-orchestrator's perception scorecard, so that miru and the executor
that recorded the looks are quoted under one definition. The sets are batchplan-orchestrator's
recordings `rows4` and `rows4-stereo`, 2026-11-19: `rows4` is exact depth, `rows4-stereo` the
same tasks and seeds with depth as two stereo cameras read it (holes, flying pixels), and both
state each camera's noise as that of the camera it stands for: the table camera a D435i, 2.8 mm
at 1 m, and the hand camera a D405, 14.2 mm. miru's rows are
`cargo run --release --example score -- <set>`; the executor's are the same command with
`--recorded`, which scores the objects that executor named under each pixel and gives the counts
of the scorecards recorded beside the sets. The rows with the truth as masks are the command with
`--masks truth`, which gives every view the simulator's truth as its masks, each pixel numbered
by the truth object it shows: no mask model can say more, so those rows are what a model can
bring miru to, and what they leave is the look's own to mend. `--rooms` holds the shapes against
what the views read, by the truth's thing under each reading, over a look's cameras together: of
each solid box open at the top, how many of its own thing's readings lie in its room, which the
room says is free, and of each solid object, how many of another thing's lie inside its solids,
which the shape says are its own; each by more than three deviations of the reading's camera's
noise, per kind, with the ten objects that hold the most of each (`examples/score.rs` has the
rules).

The numbers say that something is wrong; a picture says what. `--images <dir>` writes what each
scored look made of each view as `<dir>/<run>-<look>-<camera>.png`: every pixel in the colour of
what the look says it shows, each object a colour of its own, known shapes dark gray, the robot
light gray, and black where it shows nothing. The recording keeps what the camera saw beside
it, `<set>/<run>/<look>/<camera>.png`. Open the two side by side before reasoning about a shape
or an object that came out wrong:

```bash
cargo run --release --example score -- <set> --images /tmp/labels r1   # only run r1
open /tmp/labels/r1-7-robot0_eye_in_hand.png <set>/r1/7/robot0_eye_in_hand.png
```

In `hand-in-drawer` that pair shows at a glance what the object list could not: with the hand
in the drawer the hand's camera sees the drawer's floor around the bowl and one strip of wall,
and the 14 by 21 by 5 cm "box" the bowl was listed on is that floor.

A true object is matched when one measured object is mostly it and holds most of it; merged when
most of it is in a measured object that is not mostly it, or one that is mostly it holds most of
another as well; in pieces when several measured objects are mostly it, however small
(`examples/score.rs` has the rules in full). Each row is the 140 true objects in view over a
look's two cameras together; none has most of its pixels under the robot's mask.

| Set | Measured by | Matched | Merged | In pieces | Missed |
|---|---|---|---|---|---|
| `rows4` (exact depth) | miru | 91 | 0 | 49 | 0 |
| | miru, the truth as masks | 135 | 0 | 5 | 0 |
| | the executor | 92 | 22 | 26 | 0 |
| `rows4-stereo` | miru | 69 | 14 | 56 | 1 |
| | miru, the truth as masks | 134 | 0 | 6 | 0 |
| | the executor | 58 | 34 | 48 | 0 |

The command also scores each camera by its own labels, and some figures below are those. A few
are of what no flag does, and say so: one camera's view given as a look of its own (the other
left out of `examples/score.rs`), a count of labels added to `examples/score.rs` for the run, a
rule taken out of the library, or sums the library keeps printed for the run. Those of rendered
scenes are of scenes rendered as `tests/look.rs` renders them, looked at by a test written for
the run.

- **One view merges nothing on exact depth, nor do two.** Each camera's view as a look of its
  own, nothing merges on `rows4`: the table camera matches 91 of its 140 with 49 in pieces, the
  hand camera 94 of its 133 with 37 in pieces and 2 missed; and nothing merges over both
  cameras, nor in either camera's own labels. Parts of two views are one object where one reads
  on the other's surface within three deviations of the difference of the two readings. The
  cameras' errors are independent, so that is the root of the sum of the squares of their
  deviations, not the sum, which is up to 41% wider: within three times the sum, the top of a
  box read by the table camera lies on the wall of the bowl that stands on it as the hand
  camera reads the wall, flaring out over the box, and a bowl and the cookie box are one object
  in 1 look over both cameras and in 2 more in the table camera's own labels
  (`a_bowl_standing_over_a_boxs_edge_is_not_the_boxs_in_two_views` in `tests/look.rs` renders
  it). The executor merges 22 of the 140 there: a bowl with the cookie box in 4 looks, with the
  ramekin in 4 and with the cabinet in 2, and the rack with the cabinet in 1.
- **Pieces are what is left there, and this scorecard counts every one**: a measured object that is
  mostly on one true object is a piece of it, whatever its size. On `rows4` the stove (2 pieces),
  the rack (3 to 6) and the basket (2 to 4) are in pieces in every look that shows them, the
  cabinet in 10 of 18 (2 to 8), a bowl in 5 of 28, the cookie box in 4 of 10, the ramekin in 1 of
  10 and the tomato sauce in 1 of 2. The table camera has the cabinet whole in 10 looks of 18, and
  the largest of its pieces holds all but a few of its pixels at the median; the hand camera has
  the stove whole in 17 looks of 18 and a bowl in 21 of 22. Joining a rack's slabs
  or a stove's parts is not something depth alone decides: a mask does (`View::masks`), and these
  rows are of looks without one.
- **A strip with no plane, narrower than a window, is the edge of the part beside it.** Kept as
  a part wherever it filled a window's worth of readings, a box's side seen edge on, a seam or a
  rim was a piece of its own as soon as the noise stated asked for windows smaller than its
  readings and larger than its width. As the edge of the part beside it, on `rows4` the milk is
  whole in 2 looks of 2 and the ramekin in 9 of 10, from none and 8, and the hand camera has the
  stove whole in 17 of 18, from 15; on `rows4-stereo` 69 are matched, from 64, with the cream
  cheese whole in 10 of 10, and nothing more merges on either set or on `rows4-stereo-filtered`
  (70 / 14 / 55 / 1, from 65 / 14 / 60 / 1). With the truth as masks no row moves. What it leaves:
  - **What stands in front of what it touches stays a thing of its own**, for the depth cannot
    tell a stove's knob, which is the stove's, from a slab a box stands on, which is not: each is
    a strip with no plane in front of the part beside it. Taken as edges as well, `rows4-stereo`
    gives 92 / 14 / 33 / 1 and `rows4-stereo-filtered` 96 / 14 / 29 / 1, the stove whole in 17
    looks of 18 on each; but the slab under a box on a real camera's noise
    (`a_box_across_a_narrower_slab_stands_on_it_where_the_slab_reads_over_its_foot`) is one
    object with the box in 37 of 40 draws, where it is apart in all 40. A strip stands in front
    of the part wherever it hides it, across a jump or a step, along a window's width of their
    border: the unit borders are measured in, not a measured threshold. At two widths a pencil
    lying along a box's front would be taken for its edge
    (`a_pencil_lying_along_a_boxs_front_is_not_the_boxs_on_noisy_depth`), but no test tells one
    width from any length at all.
  - **What only reaches a thing is no edge of it.** A pencil lying on the table up to a box's
    front corner, 6, 8 or 10 mm through on exact depth, stands in front of the box only at its
    tip, and was taken for the box's edge and made one object with it, where it is apart at
    4e9f9c1. An edge runs along its part: a strip is one only where more than half of its
    readings lie within a window's width of the part in the image, the reach of one plane, and
    the pencil has a tip's worth there (`a_pencil_lying_up_to_a_boxs_corner_is_not_the_boxs`).
    No row of the three sets moves, in either camera's labels or over both, with or without the
    truth as masks, and `one1` and `work-miru2` look as they did; at 0.25 and 0.1 of the stated
    noise the table camera's own labels have the cabinet in pieces in 17 and 18 looks of 18,
    from 11 and 16, where over both cameras nothing moves. Two other statements put the pencil
    right and are not in. Taking a strip for an edge only where it touches the part along a
    window's width leaves the ramekin under a bowl in two pieces, on `rows4` in
    `libero-spatial-5-1` (90 / 0 / 50 / 0) and in `one1`'s `libero-spatial-5-0`, where the bowl
    on it then stands on the table, 9.30 cm tall, from 8.18 cm on the ramekin; and the hand
    camera's own labels have the cabinet in two pieces in `libero-spatial-4-1` on
    `rows4-stereo`. Counting a strip in front of the part however short the border leaves a
    cream cheese in two on `rows4-stereo-filtered` (69 / 14 / 56 / 1).
  - **What lies between two things stays a part of its own.** At a bowl's foot on stereo depth a
    band of stray readings holds some of the bowl and some of what it stands on; given to the parts
    beside them by the planes they lie nearest, its readings go crosswise, and another view's part
    joins the two. Such a band taken for the edge of the first part found beside it, under the same
    test of standing in front (`!two` taken out of `segment::parts`, which no flag does), nothing
    more merges, the stereo sets do not move, and on `rows4` a ramekin is whole in one look more and
    the cabinet in one less (91 / 0 / 49 / 0, as with it); but four rendered tests fail, each of
    things that must stay apart. Taken for an edge whatever stands in front, with the in-front
    clause out as well, `rows4` gives 96 / 0 / 44 / 0, every bowl matched, but a bowl merges with
    the ramekin in `libero-spatial-5-1` on `rows4-stereo` and in `libero-spatial-5-0` on
    `rows4-stereo-filtered`, and with the cookie box in `libero-spatial-5-1` there (16 and 18
    merge), and eight rendered tests fail. Each band holds both things here too, a piece of its own
    or in one thing's object.
- **A bowl is one object in one view where its outside shows a plane.** Its outside stands in
  front of its inside along the near rim, and that the two are round about one axis and end in
  one rim makes them one (`src/segment/round.rs`). On `rows4`, with the table camera's view as a
  look of its own, a bowl is whole in 23 of 28 views and the ramekin in 9 of 10, where with the
  rule taken out (of `segment::parts`, which no flag does) a bowl is in 9 and the ramekin in 1;
  with the hand camera's view as one, a bowl in 17 of 22, 14 without the rule. On
  `rows4-stereo` a bowl is whole in 9 of 28 of the table camera's views, 4 without it, and in 5
  of 22 of the hand camera's, 4 without it. Over both cameras a bowl is matched in 23 of 28 looks
  on `rows4`, 21 without it, and in 12 on `rows4-stereo`, 9 without it; no other kind's row
  changes, and nothing more merges, over both cameras, in either's own labels or in either's
  view as a look of its own. What it leaves:
  - **An outside with no plane says nothing of which way it faces.** Seen from above, a bowl's
    outside is a strip under a window wide. Of the 5 bowls still in pieces on `rows4` and the 10
    on `rows4-stereo`, taking the facing out of the rule (`Shell::faces` in `shells`, which no
    flag does) leaves 0 and 3, with no merge more on `rows4` and on `rows4-stereo` a bowl merged
    with the ramekin in 1 look more. It is asked all the same: a flat thing whose edge lies along
    a bowl's rim turns about the bowl's axis as any flat thing does, and is told from the bowl's
    own wall by its facing alone (`a_bowl_under_the_edge_of_a_shelf_is_not_the_shelfs`).
    The axis and the circle have each a rendered scene that they alone keep apart (a bowl askew
    in another; a can or a ball in a bowl); taken out, neither changes a row of these two sets,
    and only the axis changes a camera's own labels: without it a bowl and the cookie box are one
    object in 1 look more of the table camera's on `rows4-stereo`.
  - **Where an outside has planes they are few**, and three deviations of so few bound loosely:
    the noisy test bowl's add up to 0.8 to 0.9 of a window in each of three draws (summed in
    `shells` for the run), from which its facing and whether it turns about the axis are told.
  - **On rendered depth the axis is accepted more easily than on a camera's.** A plane's stated
    error counts a pixel's footprint on a surface that leans, which a renderer that reads at
    each pixel's middle does not have: the squares of the misses of the test bowl's normals add
    up to 0.17 of what their planes' covariances make of them for its outside and 0.25 for its
    inside on exact depth, and for its inside to 0.51 to 0.56 at 0.5 mm of noise at a meter,
    0.70 to 0.84 at 1 mm and 0.67 to 1.2 at 2.8 mm (the two sums `Shell::turns` holds against
    each other, printed for the run, over three draws each).
  - **A thick wall's rim is a part of its own.** To a camera of 0.5 mm at a meter the round top
    of a wall 6 mm or more through lies clear of the wall's plane: a strip of readings between
    outside and inside that holds no window, so has no plane, and this rule leaves the bowl in
    two or three. The rim and the inside hang over the hollow, which the camera sees past them,
    and touch the outside: by that the 8 mm bowl is one object
    (`a_thick_walled_bowl_is_one_object`). To a camera of 2.8 mm at a meter the same rim reads on
    the wall's plane, and a wall 8 mm through is one object by this rule, on exact depth and on
    20 of 20 draws of noisy depth.
  - **A bowl that is part of a ball's shell has no one axis.** Its normals all pass the ball's
    middle, and every line through that is as good an axis. On exact depth the fit finds the
    upright one, and three such bowls, of 6, 7 and 8 cm to the shell, are one object each; with
    2.8 mm at a meter of noise the first is in 8 of 20 draws, the second in 16 and the third in
    none by this rule. The fit takes the least eigenvalue's axis without asking that it stand
    apart from the next. What it leaves apart hangs over the hollow and touches the rest: so the
    first is one object in 11 of 20 draws, the one that
    `a_bowl_round_as_a_ball_is_one_object_on_noisy_depth` renders among them, the second in 18 and
    the third in 14.
  - **Rendered, the 4 mm bowl is one object from 12 to 34 degrees above the table, from 39 to
    49 and from 51 to 53**, and its wall with its inside from 35 to 38, where its floor stays a
    second object: the floor reads as hiding the wall behind it there, as it does to a camera of
    1.5 or 2 mm at a meter, which is the floor's rule, not this one. At 50 the near rim hides the
    floor, and the far wall along under half a window's width; from 54 to 60 the outside shows no
    plane that tells its facing, but at 54 to 57, 59 and 60 the bowl is one object all the same:
    the outside's lowest reading lies inside the shape fitted to the inside, which is not solid,
    and the camera sees the hollow past the space under the outside's middle.
  - **On noisy depth** of 2.8 mm at a meter, 1.7 mm where the bowl stands, the bowl is one object in
    20 of 20 draws, holding 93 to 95% of its pixels: 5 to 6% of them, of its lowest readings, read
    within three deviations of the table and are the table's.
  - **Two rims within half a window of one another are one circle to it.** A bowl lying in another,
    its rim 6 mm narrower and 2 mm higher, is one object with it by this rule when tipped 6 or 10
    degrees toward the tests' camera; it is apart upright, where its own outside shows between the
    two, tipped 2, 4 or 12 that way, and tipped 6 or more to a side, up to the 16 degrees asked. (At
    8, 14 and 16 toward the camera and at 4 and 8 to a side the two are one object without the
    rule.) A tilt toward the camera is told least: the outside's normals lie near the plane through
    the axis and the camera. For the same reason the fitted axis is loose that way, 4.8 degrees off
    an upright ramekin's on exact depth. Masks that tell the two bowls apart keep them two
    (`two_bowls_one_askew_in_the_other_are_two_where_their_masks_say_so`).
- **With perfect masks 135 of the 140 are matched on `rows4` and 134 on `rows4-stereo`, and none
  merges**, over both cameras or in either's own labels: the table camera matches all 140 it has
  in view on both sets, the hand camera 131 and 132 of its 133. No model has been run yet: this
  is the ceiling, and the truth as masks does not mend what it leaves:
  - The 5 and the 6 in pieces are two cameras' parts of one thing that nothing joins. The cookie
    box, in 4 of its 10 looks on each set, has a bowl standing on it, and what each camera sees
    of the box is behind the bowl from the other. Of the tomato sauce, in 1 of its 2, the
    cameras see opposite sides. On `rows4-stereo` the ramekin is two in 1 of its 10, where the
    hand camera shows 165 pixels of it.
  - What the hand camera misses is small and near: a bowl that 96 pixels show, on both sets,
    and on `rows4` a ramekin that 179 show, each 32 cm from it. A window at its noise is 121
    readings or more there, no stretch of them fills one, and they are in no object.
  - A reading within three deviations of a known shape whose part stands in front of none of
    the shape stays the shape's whatever its mask, as a thing's foot does where it is not sharp.
    Counted from the labels (`--shares`), per kind over its looks: in the table camera on
    `rows4` the butter loses 22% of its pixels to the table, the cream cheese 7%, the cookie box
    5% and the ramekin 5%, the plate none; on `rows4-stereo` the butter 20%, the cream cheese 11%,
    the cookie box 7%, the ramekin 5%, the stove and the wine bottle 3%, and the milk, the plate
    and the tomato sauce 2%. In the hand camera every kind loses 3% or less but the tomato sauce,
    9% and 8% in its one view there.
  - Another thing's readings in a part stay in it while they fill no window, and a window is
    169 readings at a meter from the table camera and 225 at half a meter from the hand camera.
    Counted from the labels again, 523 pixels of true objects are in an object that is mostly
    another's on `rows4`, and 1089 on `rows4-stereo`. The most in one object are a bowl's 136 in
    that of the ramekin it stands on, 9% of it, and on `rows4-stereo` the cabinet's 118 in a
    bowl's, 3% of it.
  - No object shows no true object, on either set, by the same count.
- **A mask a few pixels off its thing's border does about as well as an exact one**, by the table
  in "Masks from a model" (`--masks truth|one --border <pixels>`). At 65ef83e, where Split cut every
  side that filled a window, a mask 1 px short gave 39 / 0 / 101 / 0 on `rows4` with one mask,
  12 / 3 / 125 / 0 and 23 / 0 / 117 / 0 on the stereo sets, and 39, 11 and 21 matched with the
  truth; 3 px short 46, 30 and 27 matched with the truth; and 3 px over with one mask 106 / 33 /
  1 / 0, 129 / 10 / 1 / 0 and 126 / 13 / 1 / 0: the rim just outside each mask, a pixel wide and
  several windows long, was a part of its own that nothing joined back, and the fringe of a mask
  grown over a neighbour was cut off the neighbour. What it leaves:
  - **With one exact mask the stereo sets match fewer than at 65ef83e**: 129 / 7 / 4 / 0 and
    132 / 2 / 5 / 1, from 132 / 2 / 6 / 0 and 134 / 0 / 6 / 0. Eight true objects over both
    cameras, each a thing on or in another: on `rows4-stereo` the cabinet merges with the rack on
    it (`libero-goal-1-1`), the bowl with the cookie box it stands on and the box with the bowl
    (`libero-spatial-3-0`), the bowl with the ramekin under it and the ramekin with the bowl
    (`libero-spatial-5-0`); on `rows4-stereo-filtered` no object holds most of the ramekin
    (`libero-spatial-5-0`), and the bowl and the ramekin merge, each under a mask of its own
    (`libero-spatial-5-1`). In each a strip narrower than a window lies in one part with the
    masked thing and its unmasked neighbour: the neighbour's in the thing's part, which is what a
    mask a pixel or two short leaves of the thing itself, or the thing's own in the neighbour's
    part where it is not most of its mask, which is what a mask a pixel or two over takes of a
    neighbour. Depth and mask together say the same of both, and no rule of Split tells them
    apart.
  - **A mask 1 px short leaves the ramekin in two pieces** in `libero-spatial-5-1` on `rows4`, with
    one mask and with the truth (133 / 0 / 7 / 0, where asking the width alone between a mask and
    none gave 134 / 0 / 6 / 0). Why is not traced.
  - **A rim goes with any wide unmasked neighbour in its part**: a part's unmasked readings are one
    side, whatever thing they are of. A box on a wider box, their fronts flush, under a mask 1 or
    2 px short: 146 and 284 of its 1344 pixels are in the lower box's object, as at 65ef83e
    (`a_box_on_a_wider_box_keeps_its_rim_where_its_mask_stops_short`, ignored).
  - **At 3 px short what is left in pieces is what geometry already leaves apart**: a piece of the
    thing with less than half its readings in the short mask, which no mask joins
    (`libero-spatial-5-1` on `rows4`, the bowl on the ramekin under its mask alone: 163 of the
    piece's 174 readings the bowl's, 50 of them in the mask; without masks the bowl is in two
    pieces in that camera too).
  - **With the truth as exact masks no row moves**, but a flat thing's readings on the table,
    narrower than a window inside the table's part, stay the table's: in the table camera's
    labels the stove's object holds 87 to 93% of its pixels on `rows4`, 85 to 94% and 88 to 95% on
    the stereo sets, where it held 87 to 100%, 91 to 100% and 88 to 100%. At 1 and 3 px off the
    truth's rows move as the table shows.
  - **Asking a side's width costs nothing five alternated runs on one thread tell apart**, asked
    only of the sides it may cut: 2614 against 2607 ms at the median for the 20 looks of
    `rows4-stereo` with the truth as masks, and 21.9 against 22.3 s for its 140 looks under one
    mask each, 65ef83e's the second, at a load from other work that spreads those runs from 21 to
    25 s. Asked of every side of a view where any could be cut, it cost 1.3%, and of every side
    of every view with masks, 2.7%.
- **Two cameras' parts of one thing are not always joined.** On `rows4` the tomato sauce is
  matched in each camera and is two objects over both in 1 of its 2 looks, as the cookie box is
  in 4 of its 10: the 4 it is two in with the truth as masks, where what each camera sees of the
  box is behind the bowl from the other.
- **Stereo depth loses some splits again**: 14 of the 140 merge on `rows4-stereo`, where the
  executor merges 34. Over both cameras a bowl merges with the cookie box in 4 looks and with the
  ramekin in 1, the rack with the cabinet in 1, and the bowl on the cabinet's top with the cabinet
  in 1 (`libero-spatial-3-0`), by what hangs on readings a stereo camera strays (that rule's entry
  below); in 1 more look the ramekin is in no object that is mostly it. One view alone loses fewer:
  as a look of its own the table camera's merges 2 of 140, a bowl and the cookie box in 1 look, and
  the hand camera's 6 of 133, a bowl with the ramekin in 2 looks and with the cookie box in 1. The
  other 6 are two views' parts joined, each through a part of one view that holds two things that
  touch: a sliver, of readings with no plane, told apart by jumps alone, so that no crease divides
  it. The table camera's sliver of the rack, 16000 pixels in every look that shows it, holds 5000 of
  the cabinet's; the hand camera's at a bowl's foot holds about as much of the cookie box's top
  under it as of the bowl. The other view's parts of each thing read on the sliver where it shows
  that thing, at half the slack or less on the mean, and the two are one object. In the hand
  camera's labels of the look of both, the rack is one with the cabinet in all 8 looks that show it.
  No sliver holds two things in those looks on exact depth. More is in pieces there as well: a bowl
  in 10 of 28.
  - **Dividing such a sliver along the other view's parts was tried and is not in** (branch
    `divide-tried`, on a3c2a24): where a sliver's readings read on two of the other view's parts,
    each a window's worth and with a plane there, it was cut between them. On `rows4-stereo` that
    took the rack from the cabinet in 6 looks of the hand camera's own labels and in 1 over both
    cameras (64 / 14 / 61 to 64 / 12 / 63), every one of them into pieces and none into a match.
    And it cut what is one thing: the other view's two parts are as often two pieces of one thing
    as two things. A ramekin with a wall of 10 mm, its rim and wall one sliver to a camera 70
    degrees above it on noisy depth, was cut in two along the side camera's inside and outside in
    61 of 160 rendered cases (walls of 8 to 12 mm, 60 to 80 degrees, four draws) where it is one
    object without the rule
    (`a_thick_walled_ramekin_is_one_object_in_two_views_one_of_them_noisy`, which any rule that
    divides by another view must keep). Asking that the two parts meet in a crease in the other
    view kept the ramekins whole and gave all of the gain back. At a bowl's foot on a box neither
    view has a plane, and nothing was divided there either way.
- **What lies on a known shape is the shape's when nothing stands in front of it.** A thing too
  thin to show a step over the table stays the table's. And the table seen inside a closed ring
  reads as a plate's thin floor does: lying on the table and inside the ring's shape, it is the
  ring's. `a_ring_leaves_the_table_inside_it_the_tables` in `tests/look.rs` reproduces it and is
  ignored until it passes. On `rows4-stereo` a few readings of the table become an object's
  (0.1% of the table camera's and 0.3% of the hand camera's, with the truth as masks).
- **Thin things split where their planes are interrupted**: two stretches of one narrow bar,
  either side of something that crosses it, have planes too loosely told to be called one.
- **One reading nearer than about a micrometre spoils the planes below and right of it**: the
  sums kept over the whole image lose their digits to it. A depth whose inverse or noise is no
  number is left out; smaller offenders are not yet.
- **Depth stated as more exact does not yet give a better scene.** The recorded sets' exact
  depth is exact to a few micrometres: the table's readings lie within 0.5 um of its top in the
  table camera (the floor's within 7.3 um, out to 2.8 m) and within 2.5 um in the hand camera,
  whose pose is given to 1 um. `--noise` states it
  more exactly than the cameras the sets stand for. On `rows4`, at 1, 0.5, 0.25, 0.1 and 0.03 of
  the stated noise, 91, 80, 83, 80 and 80 are matched (the cream cheese whole in 10 looks of 10
  down to 0.1), and at 0.001, 36. What makes it fall:
  - **A convex edge is told from planes carried along it.** A reading with no plane of its own
    takes the plane growth brought it, which on a flat face of exact depth can come from 70 to 110
    pixels along the edge, and the two sides' planes are held against each other there. Where the
    noise is small, a pixel's footprint on a face that leans, not the noise, sets a plane's error,
    and the window the noise asks for tells the plane's lean less exactly than the windows were
    sized for: in `libero-spatial-1-1` the cabinet's top and front meet in a convex edge told so
    on 59 of 235 crossings at the stated noise, 19 at 0.5 and none at 0.25. The cabinet is whole
    in 8 looks of 18 at the stated noise, 1 at 0.5 and none at 0.25 and 0.1. Sizing the window by
    the pixel's footprint as well, tried on its own at f2fa048, scored lower at every factor.
  - **A touch that a smaller noise makes a jump.** In `one1`, `libero-spatial-5-0` look 0, the
    bowl on the cookie box touches the box only where its own outline lies over the box's top,
    which is no jump at the stated noise (twelve deviations there are 2.7 cm in depth) and is one
    at 0.5: it touches nothing, and stands on the table, 6.98 cm tall, from 5.04 on the box. And
    the bowl in the ramekin's mouth beside it has its lowest reading 2.6 to 3.5 mm under the
    ramekin's rim (at f2fa048), more than three deviations of that reading from 0.25 of the stated
    noise on, so that what it touches is too high to hold it up.
  - **A mesh's relief is surface below about 0.05.** As the deviation a white noise would need
    for the third differences of their inverse depth, the bowls' rendered surfaces carry 0.04 of
    the stated noise at the median and 0.18 at the 90th percentile, the ramekin's 0.02 and 0.13 to
    0.17, a box's 0.004 or less and the table's 0.00003. At 0.001 of the stated noise a bowl is
    one object in 2 looks of 28.
  - **A concave crease too small for the stated noise's windows.** The cabinet's plinth, a ledge
    2 mm high at its foot, meets its side in a crease that a window of seven pixels tells and one
    of thirteen does not.
- **Twenty-two tests need the jump in depth**: a slab hides the table beside it by a jump alone, and
  `a_tray_keeps_its_floor_where_a_mask_cuts_from_it_a_slab_that_hides_the_table` fails with every
  two neighboring readings linked whatever the step between them (that test taken out of `links`,
  which no flag does); so do `the_table_between_and_around_things_is_the_tables`,
  `a_board_across_a_gap_between_two_boxes_is_its_own_on_noisy_depth` and nineteen of those of round
  things, a wall's outside being in front of its inside by that jump. Every other rendered scene is
  told apart without it. On the recorded sets, linked so, 0 merge over both cameras on `rows4`, as
  otherwise, and 12 on `rows4-stereo`, from 14: the bowl on the cabinet's top in
  `libero-spatial-3-0` is not the cabinet's. In one camera's own labels more does: on `rows4` a bowl
  with the cookie box in 3 looks of the table camera and with the ramekin in 1 of the hand camera;
  on `rows4-stereo` the alphabet soup with the milk in 1 look of the hand camera, and a bowl with
  the cookie box in 1 look more of the table camera. Fourteen tests are on noisy depth, seven of
  bowls and seven of boxes and what stands on, against or before them; none is on holed depth, which
  the recorded stereo set and the truth table's two stereo readings (next) are the checks of.
- **The truth table holds the look to scenes built from known things, read through three
  cameras** (`cargo test --release -p miru --test truth -- --nocapture`): the simulator's exact
  depth, an Intel RealSense D435 by its datasheet, and the D435 with a D405 near, the two stereo
  ones through batchplan's filter (`DepthImage::filtered`), as a caller's depth comes to a look.
  Fourteen scenes, a row per scene, camera and thing: how many objects the thing came out as, and
  of the one that holds most of it, its shape's kind, size and middle, what it stands on, whether
  it is solid, the share of the thing's pixels it holds and, on a second look, its id. On exact
  depth 17 of 24 rows are wholly right, 154 properties holding and 18 not; through the D435 none of
  24, with 93 and 79; through the D435 with the D405 near none, with 101 and 71. The gaps on exact
  depth are three scenes'. A box 200 mm square and 60 mm tall with a round dial 90 mm across and
  10 mm tall on it, and on the dial a bar 25 by 90 mm and 38 mm tall: the box is not solid, and the
  dial and the bar are one object, a box of 57 by 90 by 48 mm whose middle is 25 mm off the dial's
  and 17 mm off the bar's (6 allowed), not solid, standing on the box where the bar stands on the
  dial. And a plate 60 mm square and 5 mm thick on a box 140 mm square and 60 mm tall: the two are
  one object through every camera, on exact depth the box's shape 5 mm taller than the box and not
  solid. So a plate set down on a box (`Act::LetGo`) reads gone at the next look while it stands
  there, through every camera that made an object of it on the table
  (`a_plate_let_go_on_a_box_is_seen_there` in `tests/memory.rs`). And two boxes 80 by 120 mm and
  60 mm tall, side by side and touching: they are one object through every camera, on exact depth
  a box 80 by 240 mm. So a box set down flush against another reads gone at the next look, or the
  other one does, which did not move: on exact depth and through the D435 the other, through the
  D435 with the D405 near the one set down (`a_thing_let_go_against_another_keeps_both_ids`).
  Every gap is listed row by row in `tests/truth.rs`'s doc, with what its object shows.
- **What stood on a picked thing is in no memory until a look finds it.** `Act::Picked` takes the
  thing and what stands on it, in turn, out of the objects, but the hand and the masks hold the
  picked thing alone, and the map never held a solid thing that stood on it. A can standing on a
  box whose box is picked is either riding in the hand, unmasked and missing from the `Held` that
  plans carry, or where it stood, and no obstacle holds it: the nearest to its middle is 9.0 cm off
  on exact depth and through the D435 with the D405, 3.4 cm through the D435
  (`what_stood_on_a_picked_thing_is_in_a_memory` in `tests/memory.rs`).
- **What a look through `observe` costs.** On `rows4`, at the median of its 20 looks, 237 ms
  against the 72 ms of the look inside it on one of an M4 Pro's threads and 56 against 34 ms on its
  14; on `hand-in-drawer`, 280 against 116 ms and 82 against 60 ms; a `world` after it 6.0 to
  6.1 ms on either (`cargo run --release -p miru --example score -- <set> --observe --threads <n>`,
  which prints it per look and at the median: the median of three runs, at a load of 5 to 9). Over
  `look` alone it does batchplan's part: the filter, the two masks, the shapes and the band, and
  fusing the map, for the LIBERO Panda 23.9M voxels on the CPU device; and it clones that map, so
  that the memory it was given stays as it was. A `world` after it reads the map's occupied voxels
  once for that look, and a world made again of the same memory, or of one after acts that differ
  only in what is held, builds nothing again.
- **Through a stereo camera a shape leaves readings in the map that nothing explains.** A reading
  of a shape, known or an object's, that lands past three deviations of the camera's stated noise
  from it is no shape's, and the map holds it: on the scene of `tests/batchplan.rs` the grid of
  what nothing explains is solid beside the table through the D435 and through the D435 with the
  D405 near, as much with the cameras stated 5 mm off their poses and `off_by` 5 mm as with them
  stated where they stand (`a_world_is_the_known_shapes_then_each_object`,
  `a_camera_off_its_pose_is_held_to_what_it_is_off_by`); and a pebble 1 cm across seen from 20 cm
  as a solid object is held there by that look's own readings, as by a first look of that camera
  alone, whatever earlier looks filled around it is freed (`what_becomes_a_shape_leaves_the_map`).
- **What stays unseen is tested on rendered scenes alone**, and exact depth: the sets are first
  looks. An unseen object stays until a view shows its place, so a thing moved to stand before
  where it stood leaves the object it was behind it. And the robot's pixels say nothing either
  way: what a hand took stays where it stood while the hand is before the place.
- **An object stands on what this look reads under it**, also one that keeps the shape it had.
  Where the look reads nothing lower than the object, it stands on nothing: a tray seen from a
  camera inside its shape keeps its shape, and its `on`, the table in the look before, is none
  (`a_tray_keeps_its_shape_seen_from_inside_it` has the scene). Where its lowest reading
  overhangs and no view sees under its middle, it stands on the lower thing it touches most:
  on `rows4` a bowl on the cookie box does in 4 looks, 5.05 cm tall from 6.98 cm on the table
  (the bowl is 4.7 cm, the box's top hidden under it). What that leaves:
  - **What it touches nowhere does not hold it up.** A touch of any length says which lower
    thing holds a thing up: in batchplan-orchestrator's recording `one1`, `libero-spatial-5-0`,
    the bowl on the cookie box touches the box's two pieces along 0.64 and 0.18 of a window and
    stands on the first, 5.04 cm tall, where it stood on the table, 6.98 cm; a bowl on a turned-up
    ramekin narrower than its foot stands on it
    (`a_bowl_on_a_turned_up_ramekin_narrower_than_its_foot_stands_on_it`). But a bowl whose foot
    overhangs a box on every side hides the box's top and touches it nowhere a camera sees, and
    stands on the table.
  - **On noisy depth the holder's top reads over the thing's foot.** A box across a narrower
    slab stands on the slab in 2 of 8 draws of a real camera's noise
    (`a_box_across_a_narrower_slab_stands_on_it_where_the_slab_reads_over_its_foot`) and on the
    table in the other 6, where the slab's highest reading lies over the box's lowest by more
    than that reading's tolerance.
  - **A thing in pieces stands on its own piece.** In `libero-goal-9-1` the cabinet's largest
    piece stands on a piece 1.4 cm tall at its foot, and reaches down to 1.0 cm over the table.
  - **Where one thing stands on another, the lower ends where the upper begins.** Where the depth
    does not cut two things apart at their contact, each object can hold a few of the other's
    readings, under a window's worth, which no view's masks divide off (a side of a part with
    fewer readings than a window stays with the part). In `one1`, `libero-spatial-5-0` look 0,
    nine of the ramekin's foot lie in the bowl's object and a strip of 136 of the bowl's in the
    ramekin's, so the bowl reached down to -0.0007 m, 8.18 cm tall, and the ramekin, a box, up to
    the bowl's top at 0.0805 m: two things in one place. At the noise the camera states the labels
    cannot be otherwise (down the image the bowl's outside ends at 1.077 m and the ramekin under
    its overhang begins at 1.126 m, a step of 0.040 per meter in inverse depth, where a jump there
    must pass the steps beside it by 0.034, twelve deviations of 2.76 mm at a meter), and the
    truth as masks leaves them so. Now an object A that stands
    on an object B, its shape reaching under B's top by more than the tolerance of A's lowest
    reading and B's highest, is parted from it (`part` in `src/look.rs`). A begins at b_hi, where
    B's readings summed from the top by their shares of a window reach a window's worth, or at
    its own lowest reading where that is higher; B ends where the highest of what stands on it
    begins, so that nothing on it is left over it. Not where b_hi lies over a_lo, where A's
    summed from below do, by more than their tolerance: the two interleave, as a bowl in a tray
    or a block in a bowl. Not where every reading of A lies over the room of a B open at the top,
    within its slack, as a can whose top alone shows over the walls. Not where B's top window is
    not over its bottom one: a sliver of about a window, not a thing under A. And not where
    nothing of A lies over b_hi. B is fitted again from its readings up to where it ends, and A
    from its readings from where it begins, but only where B so fitted is solid or open at the
    top, for what lies inside a shape that is not solid says nothing of whose it is. B keeps its
    top where those of its readings over where it would end that no other object's shape holds
    reach up to it, and each cut object reaches past its cut as far as those of its readings lie
    that no shape, as cut, holds: parting leaves none of an object's readings out of every shape.
    Every pair is decided once, against the shapes the look's readings alone give, and the same
    cuts make the sizes ids are kept by and the shapes at the end; so neither the order the views
    name the objects in nor a second look at the same frames moves a cut. A shape kept from the
    scene before is not cut. The bowl is now 0.0302 to 0.0810 m (its exact depth 0.0293 to
    0.0810) and stands on the ramekin, which is -0.0120 to 0.0302 (to 0.0306), a box 8.7 by 7.4
    cm from 9.3 by 7.9; the same on `rows4` with the truth as masks, there and in
    `libero-spatial-5-1`. Over `rows4`, `rows4-stereo`, `rows4-stereo-filtered`, `one1`,
    `work-miru2` and `repro4`, with and without the truth as masks, 60 objects change: 19 come
    right against the exact depth of their own pixels, 4 move within the tolerance and none
    wrongly; 37 are pieces of one thing, as the stove's body on its foot piece or a rack's and a
    cabinet's pieces, whose readings the pieces still hold between them. No reading that a shape
    held is left out of every shape, nothing stands over what holds it by more than the
    tolerance, and the 400 label images, every object's id and `on` and whether it was seen in
    each first look, the scorecards and `--offsets` are as they were, but for 4 objects that become
    solid: the ramekin under the bowl in three looks with the truth as masks on stereo depth, and a
    can's top piece on `rows4-stereo-filtered`. Parting takes 0.25 ms of a look at the median on
    `rows4` and 0.67 ms on `rows4-stereo`, 4.7 ms at most, on one thread, under 1% of a look;
    alternated with the code before, a look took 0.3% longer on `rows4` and 1.8% on
    `rows4-stereo` over five rounds at a load of 4, and as long and 0.3% longer over four at a
    load of 10 to 13
    (`a_bowl_and_the_ramekin_under_it_that_hold_a_few_of_each_others_readings_part_at_its_top`
    renders the case from two cameras;
    `a_can_seen_only_over_the_walls_of_a_lidless_box_it_stands_in_reaches_its_floor`,
    `a_bowl_and_a_ramekin_seen_steeply_part_the_same_on_the_same_frames_again`,
    `a_bowl_on_a_ramekin_on_a_box_parts_the_same_whichever_view_comes_first` and
    `a_post_on_a_box_a_can_stands_on_stays_the_boxs` each a clause; and
    `what_is_in_a_thing_or_does_not_reach_into_it_is_not_parted_from_it` what must stay). What it
    leaves:
    - **The ramekin under the bowl stays a box.** Its own readings up to the parting fit a box:
      the bowl hides its rim from the table camera, and the hand camera's 157 pixels of it are in
      no object. Fitted only from its readings more than their slack under the parting it is
      round, 8.1 cm (the truth, fitted alike, 8.8), but that band cuts a stove's and a rack's own
      readings elsewhere.
    - **A strip of a window or more is not parted.** Rendered from two cameras of `rows4`'s noise,
      one behind the bowl, the ramekin's object holds 1060 of the bowl's readings: its top window
      lies in the strip, over the bowl's bottom window, and the two read as interleaved.
    - **What lies in no other shape stays.** On `rows4-stereo` the stove's base reaches 14 to 24
      cm up by flying pixels, which no other object's shape holds, and keeps them. Three bowls on
      stereo depth, on the ramekin or the cookie box, reach down past where they part to low
      readings of theirs that no shape holds; cut there, they would come right, and 43 of the
      camera's readings, none of the exact depth's, would lie in no shape.
    - **A second look stands a piece of the cabinet on another piece.** Given each first look's
      frames again (`--again`), every object keeps its id, its shape and its `on`, but one `on`:
      on `rows4-stereo`, in `libero-spatial-4-1`, a piece of the cabinet whose lowest reading
      lies in two other pieces stands on the smaller of them, which in the first look is one and
      in the second, where the shapes kept from the first stand it, the other, cut smaller.
      Standing every first look on the cut shapes moves 5 first looks' `on`, the case's among
      them; telling the two apart by their sizes before the cut moves one in `one1`. The second
      looks' other differences from the first are as they were.
    - **Clauses only the sets plant.** Taken out alone, none fails a rendered test: A beginning at
      its own lowest reading where that is over b_hi moves 2 shapes within the tolerance; B ending
      where what stands on it unparted begins leaves, on `rows4`'s `libero-goal-4-1`, the rack
      3.8 cm under what stands on it; B reaching past its cut leaves 2 readings of `work-miru2`'s
      rack of the exact depth out of every shape, 2.2 cm, and 11 of the stereo sets'; B keeping
      its top where those readings reach it moves 3 pieces by 0.4 mm at most, and saves fitting
      the lower again. Without the last two together, the post on a box fails.
    - **Measured before and not in.** The remedies on standing measured before, each beside the
      rule in use over every test and recorded set, missed those readings or cut true ones with
      them: extremes past a window's worth (a thing's own sparse foot cut, 29 of 93 changes
      wrong), resting by a window's worth on or in what is under (7 rendered tests fail), a low
      group apart in height (a steep face's foot rows cut), a group apart by a jump (none is), and
      touching from above (a taller thing beside a lower one, and flush meetings, cut). And
      parting at b_hi as first put, with kept shapes and over a lower that is not solid: a kept
      stove cut 2.85 cm under a bowl in `one1`'s `libero-spatial-7-0`, a basket's sparse wall on
      `rows4-stereo-filtered` left out of both shapes (60 readings, up to 2.5 cm), and the inner
      bowl's piece in `a_bowl_askew_in_another_is_not_the_others` cut 1.2 cm into the other's
      room.
  - **One view cannot tell standing on a low box's far edge from standing against it.** A thing
    that stands behind a low box and against it looks, from in front, like one that stands on
    the box's far edge, and a taller thing beside a low one shows the same border along the low
    one's top.
- **Where nothing changed, every object keeps its id**: each first look's frames given again as
  the next look (`--again`), all 226 objects of `rows4` and all 227 of `rows4-stereo` keep
  theirs, each held against the object in its place, and none stays unseen. An object lies
  where an object of the scene before did whose shape holds more than half of its readings, and
  of several such, as a cabinet's or a rack's overlapping pieces are, the one nearest its own
  size; where several lie where one did, the one nearest that one's size continues it, and the
  others the next nearest they lie where, the nearest pairs first. On `open4`, every look of its
  eight runs in order, the drawer whose floor was a thing of its own and whose walls were the
  cabinet's keeps the floor's id once the bowl is lifted out (r5), and after the look from the
  hand a piece of the cabinet keeps its id and the drawer with the bowl in it the bowl's (r6).
- **What the robot divides of one thing is one thing still, where its shape stood for it.**
  With the arm down at the drawer, the drawer and one of its walls came out of the cabinet's
  object as new objects, each lying in the box open at the top the cabinet had been (`open4`,
  every look of its eight runs in order: 2, 3, 5 and 4 new ids at that look in r1, r3, r4 and
  r7). A piece is an object that continues none and has more than half of its readings in the
  solid of the earlier object that holds the most of them, in its walls and floor and not in
  its room. It is one object with the object that continues that one where the robot may hide
  what joins them: in some view both lie at one stretch of the robot's pixels and those
  without a reading that hang together with them. And not where a view reads more of where
  the earlier shape would show as something else than as them: the measure is against what
  reads as the thing, not against what is hidden, because a shape may claim more than its
  thing, as the cabinet's box claims the room under the pulled-out drawer, where the table
  reads in every look by more than the arm hides. In those four looks the drawer and its wall
  are the cabinet's again, its id and its room kept, and the bowl in the drawer stands on it;
  what is new is 1, 1, 3 and 2. Scored over the looks that hold the truth, each after the one
  before: `work-miru2` 45 / 2 / 18 / 1 to 46 / 2 / 17 / 1; `open4` (130 / 4 / 39), `one1`, and
  `repro4`'s three runs as they were; `rows4`, `rows4-stereo` and `rows4-stereo-filtered` to the
  bit as before, first looks and repeated. A block the robot crosses in a second look is one
  object with the shape it had (`a_block_the_robot_crosses_is_one_object`). What it leaves:
  - **A piece that is not at the robot's pixels takes a new id.** The cabinet's side as the
    table camera alone sees it leaves the cabinet in the look with the arm down (r4, r7: 11
    thousand pixels, which is why `open4` scores as before); why it does is not traced. A tray
    seen again from low over the table comes back as its near wall and the walls behind it
    (`a_tray_seen_again_in_pieces_is_no_solid_box`). To join these as well takes telling them
    from a thing newly set down in what a shape claims beyond its thing, which the look shows
    apart in the same way (`a_thing_set_down_in_what_a_trays_box_claims_is_not_the_trays`).
  - **A new thing set down in what a shape claims beyond its thing, beside the robot in a
    view, is taken for a piece of it**: a block behind a tray's near wall, which the tray's
    box takes to be as thick as what it hides, with the arm beside both
    (`what_comes_back_as_one_thing_stands_on_nothing_of_its_own`). Masks keep it apart only
    where both it and the thing lie in masks.
  - **One of two things taken for one that is pushed a little away, with the robot's pixels at
    both**, stays one with the other while the look reads more of their place as them than as
    anything else. With nothing of the robot's at them they are two
    (`two_boxes_that_were_one_object_and_are_pushed_apart_are_two`).
  - **Pieces of a thing that is not solid take new ids**: a bowl's shape only covers it, and
    what lies in a cover may have been put there
    (`a_thing_put_into_a_tray_or_a_bowl_is_not_the_trays_or_the_bowls`). A sliver of the bowl
    on the cabinet's top is new in each of the four looks, and a stove knob seen from the hand
    in r4.
  - **A thing the robot crosses in a FIRST look comes out in two**: there is no scene before to
    say it was one; a mask over it does (the test after the block's).
  - **What a look shows for the first time is new**: the drawer's floor and a wall seen first
    from the hand (r5, r6, r8 at look 2) are new objects beside the cabinet, which was a cover
    then.
  - **Looks taken while the arm moves are not recorded**, and the ids an executor saw change
    in them are not traced.
- **What hangs is one object with what it touches.** On `rows4` the cabinet is whole in 8 looks
  of 18 over both cameras, from none with the rule taken out (of `attach`, which no flag does), in
  10 in the table camera's labels, from none, and in 12 in the hand camera's, from 4; no other
  kind's row moves. On `rows4-stereo` 69 are matched, as without it, and 14 merge, from 12: the
  tomato sauce is whole in the table camera's labels in 2 looks, the basket and the cabinet are
  in fewer pieces in a dozen looks, and the bowl on the cabinet's top in `libero-spatial-3-0` is
  one with the cabinet (below). With the truth as masks no row moves; on `rows4` nothing more
  merges, over both cameras, in either camera's labels or in either's view as a look of its own.
  In `work-miru2`, batchplan-orchestrator's recording of the drawer the robot was refused, the
  cabinet is whole in look 3 of `libero-goal-0-0`, where it was in 5 pieces, and in 3 from 5 in
  look 0, where the robot was refused; in 3 from 5 in look 0 of `libero-goal-3-0` and in 2 from 5
  in look 3. What it leaves:
  - **What lies inside a shape that is not solid does not rest there**, and what lies on its
    surface, within the reading's tolerance, does. In look 0 of `libero-goal-0-0` two of the
    handles are the cabinet's by what the hand camera shows of them, and the cabinet's box takes
    them in: the third handle's lowest reading lies inside that box, which is not solid, and the
    handle hangs and is the cabinet's. Left of it are a post of 144 pixels in the table camera
    and a strip of 202 along its lower edge in the hand camera. What lies in a hollow thing
    stands on it all the same, under whose middle no view sees through
    (`what_lies_in_a_hollow_thing_stands_on_it`), and what lies inside a solid shape rests
    there. What lies under a thing is each object as its own readings are fitted, joined to
    others or not, so a block on a ledge that is the cabinet's rests on the ledge
    (`a_block_on_a_ledge_out_of_a_cabinet_is_not_the_cabinets`); and a thing that hangs is held
    by what reaches lower than it, where it touches any, so the ledge is the cabinet's and not
    the block's on it.
  - **What a joined thing's own shape holds up stays held**, where the view cannot see through
    the space between. In look 0 of `libero-goal-3-0` a handle 6 cm over another that is the
    cabinet's stands on the lower one's shape, and the cabinet is in 3 pieces, where the shape
    fitted to the cabinet and its handles together, not solid, let it hang (2 pieces); in look 3
    a piece of the rack whose lowest reading lies inside the cabinet's own box rests there, and
    the rack is in 6 pieces, from 5. On `rows4` the cabinet is whole in 8 looks, from 9.
  - **A thing flush with what it stands on is one part with it.** A block whose front is flush
    with the front of the ledge it stands on, or within the noise of it, is one surface with the
    ledge to the depth, as two boxes standing flush are, and is the cabinet's with the ledge.
  - **A part that reaches lower than where a thing rests makes it hang.** A board laid across a
    gap between two boxes, with a tag under its front over the gap, rests by the tag's lowest
    reading: under the board's middle the camera sees the gap, and the board and the tag are one
    with a box. A part that joins it by hanging does not do so: the two rest by the one that
    holds the other up.
  - **What hangs on a known shape is held by nothing the rule names**: a plate 1 cm thick that
    stands out of a known wall and touches a box's side along a window's width of border is the
    box's; at 8 mm it touches the box along less and stays apart
    (`a_plate_on_a_known_wall_that_touches_a_box_at_an_edge_is_not_the_boxs`).
  - **A stray reading can make a shape not solid, and a thing hang.** On `rows4-stereo`, in
    `libero-spatial-3-0`, the cabinet's object holds a reading 33 cm under the table and more than
    a meter from it, so its box takes in the bowl on its top and is not solid; the bowl's object
    holds one 1.6 cm over the table and half a meter from the bowl, its lowest. The space under
    its middle, placed so, shows, and the bowl joins the cabinet: 2 more merge than where what
    lies inside a shape that is not solid rests there. Such readings are the caller's depth
    filter's to remove (below).
  - **On stereo depth the cabinet is whole in no look**, as without the rule; why is not yet
    traced.
  - **What rests by its lowest reading is not asked about its middle**, so a thing that stands
    on one end and hangs out over free space at the other stays apart, and a thing with creases
    of its own whose pieces each stand on what is under them stays in pieces, as a step of two
    blocks does (`a_step_is_one_object_by_the_mask_of_a_model_prompted_from_a_first_look`).
- **Which objects are solid** (`Object::solid`), counted for the object that holds most of each
  true one over both cameras (`--offsets`, of the matched ones). Of those matched on `rows4`,
  every cream cheese (10), cookie box (6), wine bottle (8), butter (2), milk carton (2) and tomato
  sauce (1) is solid, no bowl (23) or plate (18), and 1 ramekin of 9, the one a bowl stands on in
  `libero-spatial-5-1`, whose mouth the bowl covers; with the truth as masks, no cabinet (18),
  basket (2), rack (8) or stove (18) is. Some that should be are not:
  - **A can whose lips are wider than its body**: neither alphabet soup is (2), the readings of
    its body lying inside the cylinder its lips span by more than their noise.
  - **A round thing measured as a box**, inside whose corners its body lies: 1 salad dressing of
    the 2.
  - **On a stereo camera's depth.** A shape's faces lie at its outermost readings, which the noise
    puts farther out than three deviations of most of the rest, and stray readings at its edges
    farther still: on `rows4-stereo` 1 of 10 matched cream cheeses is solid, no cookie box (6) or
    butter (2), and no wine bottle (8). Moving each reading that a plane accounts for along its ray
    onto that plane before the fit, tried at 3b7349f (a step added to the library for the run),
    leaves the faces where the readings no plane accounts for lie, along a box's edges: eight in
    ten of a butter's moved readings lie 4.3 to 5.4 mm inside its shape, where three deviations are
    4.5 mm, and 3 of 9 cream cheeses come out solid, 1 of 6 cookie boxes, no butter and no wine
    bottle, but 2 of 18 plates. On `rows4` it moves a round thing's readings by the planes'
    curvature across a window, 1 mm at the median on the wine bottle and up to 6 mm, up to 11 on a
    bowl, and the bottle's middle from 0.02 cm to 0.41 cm off, solid in 1 look of 8.
- **What a stereo camera reads between a thing's edge and what lies behind it stretches the thing's
  shape**: such readings are the thing's, and a shape is fitted around every reading of it.
  Unfiltered, the median distance from above of a matched object's middle from the truth's
  (`--offsets`, the scorecard's median offset) is under 0.1 cm for every kind matched on `rows4` but
  the cabinet, 0.36 cm, whose shape takes in its handles; on `rows4-stereo` it is 7.39 cm for the
  salad dressing, 7.15 for the wine bottle, 5.01 for the milk, 4.48 for the alphabet soup, 3.96 for
  the tomato sauce, 1.85 for the butter, 1.83 for a bowl, 1.67 for the ramekin, 0.48 for the plate,
  0.31 for the cookie box and 0.20 for the cream cheese, and with the truth as masks 63.72 for the
  rack, 17.91 for the cabinet, 10.12 for the basket and 6.47 for the salad dressing. They are the
  caller's depth filter's to remove before the look (batchplan's `DepthImage::filtered`). Leaving
  out the readings that lie on no plane, tried at 3b7349f, cannot tell them from a rim or a surface
  that turns away from the camera: on `rows4` it moved the wine bottle's middle from 0.02 to 0.26
  cm off and a bowl's from 0.04 to 1.8.
- **A box a view sees into is open at the top; a floor seen from over it is no solid slab.** In
  batchplan-orchestrator's recording `hand-in-drawer` (libero-spatial-4 with the hand left in the
  drawer over the bowl, two runs of 13 looks, exact depth) the hand's camera sees the drawer's
  floor and little of its walls, and at 65ef83e the floor was an object of its own, a box 14 by
  21 by 5 cm, solid because its readings lay on its bottom face, with the bowl inside it: every
  grasp of the bowl's rim met it. `--rooms` counts that: readings of another thing inside a
  solid shape, 273,061 over 12 objects at 65ef83e and 60 in one now; the drawer is a box open at
  the top from the look with the hand over it on, in both runs. In `open4` r5 the drawer after
  the bowl is lifted out, a solid box 20.8 by 11.7 by 5.2 cm at 65ef83e, is open, its room 20.1
  by 11.6 cm. `--rooms`, 65ef83e to now, as boxes open at the top / those with their own thing's
  readings in the room (pixels) / solid objects with another thing's readings inside (pixels):

  | set | 65ef83e | now |
  |---|---|---|
  | `rows4` | 0 / 0 (0) / 1 (1,979) | 15 / 7 (4,728) / 1 (1,979) |
  | `rows4-stereo` | 0 / 0 (0) / 11 (94,830) | the same |
  | `rows4-stereo-filtered` | 1 / 1 (3,109) / 6 (3,368) | 2 / 2 (3,123) / 6 (3,368) |
  | `open4` | 10 / 10 (86,712) / 3 (52) | 13 / 10 (86,712) / 3 (52) |
  | `hand-in-drawer` | 1 / 1 (7,408) / 12 (273,061) | 22 / 5 (7,575) / 1 (60) |
  | `work-miru2` | 0 / 0 (0) / 0 (0) | 10 / 6 (3,960) / 0 (0) |
  | `one1` | 0 / 0 (0) / 0 (0) | 1 / 1 (627) / 0 (0) |
  | `repro4-fix`, `repro4-fix2`, `repro4-nolook` | 6, 8, 4 open | the same |

  The scorecards of all ten sets are as they were, every id is kept (`--again`), and
  `rows4-stereo`'s digest is the same; `rows4`'s and `rows4-stereo-filtered`'s change by the
  shapes. On one machine, runs alternated, a whole set took 1.3% and 1.0% longer on `rows4` and
  `rows4-stereo` on one thread and 3% and 2.5% on fourteen, `open4` 1% less, and
  `hand-in-drawer`, where the rule opens the drawer in most looks, 9% and 18% more: what a look
  with boxes open at the top costs, here and before. What it leaves:
  - **A wine rack's pieces open** (26 over the sets), and their rooms hold readings of the rack
    that are other pieces' (the 4,728 pixels on `rows4`): the rack is still in pieces.
  - **Slivers of a bowl open.** In `hand-in-drawer` ten pieces of the bowl on the cabinet's
    top, each about 1.4 by 0.8 by 0.8 cm and an object of its own, are boxes open at the top,
    and four hold 40 to 43 of the bowl's readings in their room; one in `open4`. On
    `rows4-stereo-filtered` one more box than at 65ef83e is open, a wine rack's piece.
  - **A sliver of a thing's own can still close its box.** In `hand-in-drawer` r1, with the
    hand over the drawer, 180 pixels of the cabinet's body read as an object of their own lie
    8 mm inside a wall of its box; the box covers the cabinet until the drawer is seen as its
    own.
  - **A room can still hold its own thing's readings**: the cabinet's top over the drawer's
    opening (`open4`, `repro4`: up to 16,355 pixels in one box, at most 12 mm inside a face).
    Drawing the room in to hold none of them was tried and set aside: under the top the floor
    runs on, and what lies on it there then rested inside a wall, which closed the box
    (`plans/perception/open-box/clear/`).
  - **A step, a ledge and a post are held by rendered scenes with one mask over the thing**:
    from above their parts meet in no crease a camera reads, and without the mask each is a
    thing that stands on the next.
- **A drawer pulled out is a box open at the top where a look sees into it.** In
  batchplan-orchestrator's recordings `repro4-fix` and `repro4-fix2` (libero-spatial-4, the bowl in
  the cabinet's open top drawer, exact depth, three runs each), look 3 has the hand's camera 20 cm
  over the bowl, and in all six the drawer is open at the top, with the cabinet in three and of its
  own in three; at 9d44ee4, before boxes open at the top, it was a box over its room that is not
  solid. Held against the room exact depth shows from there (an example written for the run):
  each wall's inner face lies 0.5 mm inside it to 0.2 mm outside, the floor's top 0 to 0.3 mm
  over it, and the free room between the bowl's rim and each wall, 3.6 to 61.1 mm, comes out
  within 0.5 mm; a point halfway between the rim and each wall lies in none of the solids. No
  thing but the cabinet has a window's worth of readings more than three deviations inside an
  open box's solids in any look of `repro4`, `one1`, `rows4`, `rows4-stereo` and
  `rows4-stereo-filtered`, with or without the truth as masks, where at ca7d64f, before
  `hold_what_stands`, the bowl, the stove, a ramekin or the cookie box had in 26 (an example
  written for the run, by the truth's body of each reading). Against 9d44ee4 the scorecards of
  the three sets are the same, with the truth as masks too, the 240 label images (`--images`) the
  same to the byte, every id kept (`--again`), and the digests with the truth as masks; without
  masks the digests change by the open boxes. Every object of every `repro4` look keeps the id it
  has at 9d44ee4. On one thread ca7d64f's best of three alternated rounds was 108.7 and 116.6 ms
  on `rows4` and `rows4-stereo`, where 12f9fc0 took 107.3 and 116.7, at a load of 6.5 to 8.3.
  What it leaves:
  - **Where the hollow rule does not make the drawer, it is pieces or a cover.** From the table
    camera it makes it in 9 of 15 looks on exact depth, 1 of 2 on `rows4-stereo` and none on
    `rows4-stereo-filtered`; elsewhere the floor is a slab of its own, which meets the walls'
    inner faces, slabs of their own too, but lies inside none of them. In `one1` look 2, from 34 cm
    over it, the drawer is one object, but its floor ends where the view looks down past it into the
    cabinet, hides 74.5 windows' width there and is no floor: a box that covers the room. Once
    open, the drawer keeps its room: in `repro4-fix` r0 look 5 the look alone makes it a box that
    covers it, and it stays open, its walls within 0.5 mm. With the truth as masks no drawer is
    open: the masks number the cabinet and its drawer as one thing, one object before any floor
    could be joined to it.
  - **From the table camera alone the near walls are thick.** In `repro4-nolook`, in look 0 of
    `repro4-fix2` r0 and on `rows4` the far walls' faces show and lie within 0.3 mm, and the walls
    toward the camera hide the floor behind them: 42 to 44 mm too thick where the bowl does not
    stand, and 8.5 to 10.8 where it does and the room reaches over it; no reading of the bowl lies
    in them. In `repro4-nolook` r0 and look 0 of `repro4-fix2` r1 the floor joined is 9 windows,
    and a piece of the cabinet rests inside the walls: the cabinet is a box that covers it, not
    solid, as at 9d44ee4. That errs toward collision, as it is meant to, until a look sees in.
  - **A piece of a wall can widen the room through the wall.** In look 4 of `repro4-fix2` r2,
    which has no truth, a piece of the drawer that the hand's camera sees as an object of its own
    stands in the room's corner and in front of the floor along 7 windows' width, and the room
    reaches over its outline through the front wall; look 5 keeps that room, and its wall there
    is 6.8 mm thinner than it is (9d44ee4 and ca7d64f: no open box, and within 0.5 mm). Telling a
    piece of the walls from a thing on the floor by where most of its readings lie, or by what it
    rests on, put the bowl back into the walls in rendered scenes from above.
  - **The bowl stands on the drawer at the floor's height in 2 of the six looks.** In 3 it stands,
    as at 9d44ee4, on a piece of the drawer or of the cabinet whose box is not solid and reaches
    over the room, down to its own lowest reading 7 mm over the floor; in 1 on the floor's slab of
    look 0, which no look since has shown gone, where at 9d44ee4 it stood on the cabinet's box.
  - **The drawer's open side can be a piece of its own**, as at 9d44ee4: a box 16.3 by 2.8 by 6.1
    cm in `repro4-fix` r0, and in r2 of both sets one of 27 by 14 by 6 cm, not solid, that reaches
    over the room. The open box then has a thin wall there or none.
  - **A round hollow can come out open at the top.** One whose floor is round from above, or whose
    inner face turns away from where it is put by more than its readings' slack, stays a box that
    covers it (`a_round_hollow_in_a_square_block_is_no_box_open_at_the_top`). But seen in part,
    with the turn within the noise, a square block's round hollow from 50 degrees stating 2.8 mm
    at 1 m is open, the corners of its room 10 to 19 mm into the block
    (`a_round_hollow_in_a_square_block_seen_from_50_degrees_is_no_box_open_at_the_top`, ignored
    until it passes); and on `rows4-stereo-filtered` the ramekin of `libero-spatial-4-0`, whose
    outline the noise made square, is open. So is the 8 mm bowl from 50 degrees on depth as noisy
    as a real camera's, seeds 1 and 3, some 600 of its 1432 readings inside its solids
    (`a_thick_walled_bowl_seen_on_noisy_depth_is_no_box_open_at_the_top`, ignored): how many of
    its own readings an open box holds does not tell, for the drawer's walls hold 41 to 1259
    windows' worth of its own over the floor, the cabinet's beside and behind it. Round hollow things whose outline is round, as a bowl,
    stay cylinders that are not solid, and the basket's floor is joined to its walls on no set
    without masks.
  - **Some of the rules hold by the recorded looks alone.** Six rules no rendered scene was
    found to need, in 480 of lidless boxes, trays and long boxes, a bowl in them or not, turned,
    from five cameras, on exact and noisy depth: that an open box is solid whatever its readings
    (the cabinet's face under the drawer lies in its floor), that a face lies over the floor's top
    and within its reach across, that a room carried gives way to an inner face seen in it, that
    it is carried only where it lies in the box, and that a box covers what rests in its walls.
    Without each, planted out one at a time, `repro4` or `rows4-stereo` looks lose the drawer or
    its walls, or a stove or a bowl lies in them.
  - **From straight above on exact depth stating 0.5 mm at 1 m** a rendered tray is one object by
    what hangs (`attach`), its floor meeting the walls along no crease the planes tell, and stays a
    box that is not solid; stating a real camera's noise it is open at the top
    (`a_lidless_box_seen_from_above_is_open_at_the_top_on_every_side`).
- **Time**: a look of two 640 by 480 views among a cell's 17 fixtures takes 41 ms at the median
  on `rows4` and 42 ms on `rows4-stereo-filtered` on an M4 Pro's 14 threads, 48 and 56 ms on two,
  and 77 and 93 ms on one; a later look with open boxes in it (`hand-in-drawer`) 58, 71 and
  115 ms (`cargo run --release -p miru --example score -- <set> --again --digest --threads <n>`:
  the median look of the best of three rounds, alternated with the same looks through miru's
  own repository's code at dd15ef3, which took 50 and 55, 76 and 78, and 119 and 127 ms, and 90,
  116 and 168 on the later looks, at a load of 8 to 14 from other work). The scenes are the same
  to the bit on 1, 2 and 14 threads, and the same as that code's (`--digest`: ten sets, with no
  masks, with the truth as masks and with one mask at a time).
  Where the crate is ahead of that code: the known shapes are held against every reading by
  batchplan's `sdf::on_shapes`, which turns each shape's rotation into its matrix once and
  measures a shape only against the runs of readings that come within reach of its box, where
  asking each shape for its distance at each reading took 40 ms of a one-thread look; a look
  gathers its objects' readings in bands on its threads; an object is measured against no
  earlier shape it lies far from, and against the others pair by pair on the threads; and a box
  open at the top makes its parts once for all the points asked of it. What is left on 14
  threads is mostly one thread's: telling the views apart takes about 18 ms of a first look, per
  view the floods that find the seeds 2.7 ms, the growth of the seeds over the readings beside
  them 3.3 (7.2 on a stereo camera's depth), the sums the planes are fitted from 2.1; those sums
  made on several threads were no faster, the step being bound by memory. No GPU is used. What
  is done for each pixel by itself, and could be a kernel, is about 3 ms a view on 14 threads;
  the sums are in double precision, which a kernel on Metal does not have; and what each pixel's
  step gives is held against a threshold, so a GPU's rounding would make a scene depend on the
  device.

## Working on it

`AGENTS.md` has miru's design rules, and the repository's `CLAUDE.md` the style and the working
rules. From the repository's root, `scripts/check.sh` runs every lint and `cargo test -p miru` the
tests, which render their own scenes.
