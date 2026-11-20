# Worlds and perception

A `World` is a list of obstacles, and the links allowed to touch some of them. Worlds are uploaded to a [`Device`](devices.md) once (`device.upload(&worlds)`) and stay there. Every IK and planning call takes the uploaded `Worlds`, and each problem names its world by index.

## Obstacles

| `Obstacle` | Shape |
|---|---|
| `Cuboid { center, half_extents, rotation }` | an oriented box |
| `Sphere { center, radius }` | a sphere |
| `Cylinder { center, rotation, radius, half_height }` | a solid cylinder along its local z axis |
| `Capsule { center, rotation, radius, half_length }` | a capsule around its local z segment |
| `Sdf { grid, center, rotation }` | a signed distance grid, for any other shape |

`Obstacle::cuboid(center, half_extents)`, `sphere`, `cylinder` and `capsule` build the primitives axis-aligned, and `.placed(pose)` moves and turns one. `Obstacle::from(grid)` puts a grid in world coordinates, where grids built from points and depth images belong. Primitives are exact.

## Changing a scene

Uploaded `Worlds` never change: a scene that changes is uploaded again. To move one obstacle, change it in your `World` (`world.obstacles[k] = moved`) and upload the worlds; to take one out, `World::remove`, which keeps every touch on its obstacle. Every call after that takes the new `Worlds`, and the old ones stay valid for calls already holding them. An upload prepares each obstacle for the CPU's checks and, on a GPU, writes one buffer: 512 tabletop worlds took 0.1–0.2 ms on the M4 Pro ([Benchmarks](benchmarks.md#gpu-and-cpu-throughput)). A distance grid adds its values, uploaded once however many worlds share its `Arc`. So uploading per motion or per camera frame is cheap; what costs is building a grid from a depth image, not uploading it. Worlds work on every device made from the one they were uploaded to with `Device::with_robot`, as long as their touches name links both robots have.

## Allowed contacts

A grasp touches what it grasps. `World::touch` lists links that may touch particular obstacles, as MoveIt's allowed collision matrix does for a scene: contact between them is not a collision, while every other link still keeps clear of the obstacle, and the touching links of every other obstacle. IK, planning, trajectory optimization, `Device::evaluate` and `MeshModel` all skip those pairs. `robot.hand_links(hand)` names a hand's links: the first link at or above its frame with collision spheres, and everything below it, its fingers included.

```rust,no_run
use batchplan::glam::Vec3;
use batchplan::{
    Device, Error, Obstacle, PlanOptions, PlanProblem, Robot, RobotOptions, Touch, World,
};

fn main() -> Result<(), Error> {
    let robot = Robot::load("assets/franka/franka_panda.urdf", &RobotOptions::default())?;
    let device = Device::cpu(&robot)?;
    let can = Obstacle::cylinder(Vec3::new(0.5, 0.0, 0.1), 0.03, 0.06);
    let table = Obstacle::cuboid(Vec3::new(0.5, 0.0, -0.02), Vec3::new(0.4, 0.4, 0.02));
    // The hand may touch the can it closes on, not the table; the arm may touch neither.
    let grasp = World {
        touch: vec![Touch::new(0, robot.hand_links(0))],
        ..World::new(vec![can, table])
    };
    let worlds = device.upload(&[grasp])?;
    let (start, goal) = (robot.default_q().to_vec(), robot.default_q().to_vec());
    let problem = PlanProblem::new(0, start, goal); // goal: the grasp, solved by IK in this world
    let _ = batchplan::plan(&device, &worlds, &[problem], &PlanOptions::default())?;
    Ok(())
}
```

Link names are resolved when the worlds are uploaded, against the device's robot; an unknown name, or an obstacle index past the end, is an input error. A held object is a link of the robot holding it ([holding objects](robots.md#holding-objects)): a world where it may rest on a table names it with `Touch::held(obstacle, hand)` on the device that holds nothing, or by its link on the device for a robot that holds it. Links that touch the table a robot stands on in every configuration need a touch too: `world.always_touching(&robot)` finds them ([mounted robots](robots.md#collision-spheres)).

## Scenes

`World::load(path, &SdfOptions)` reads the static geometry of an MJCF or OpenUSD scene as obstacles. Planes become slabs and ellipsoids their bounding boxes. Meshes become distance grids of their convex hulls where the format collides them that way: MJCF always (as MuJoCo does), USD when `physics:approximation` is `convexHull`. Other USD approximations use the exact mesh.

## Distance grids

An `SdfGrid` holds signed distances on a regular grid, stored as half floats and interpolated trilinearly on both devices. Worlds can share a grid through its `Arc`, and devices and exporters then store it once. `SdfOptions` lays grids out (1 cm voxels and 15 cm padding by default).

- **`SdfGrid::from_mesh(&vertices, &triangles, &options)`**: signed distances of a closed triangle mesh: to its nearest face, negative where the surface winds around the point. A mesh may hold several closed shells, as a CAD table's top and legs: where they overlap it is their union, a shell facing in is solid as one facing out is, and a shell facing in inside one facing out is a cavity. A shell that crosses itself is solid wherever it encloses space. Two limits: where two shells that face opposite ways overlap, the overlap reads free; and inside a solid, a face buried in another shell still counts as the nearest, so the depth read near it is less than the true one. Open or non-manifold meshes have no inside and are refused with the reason.
- **`SdfGrid::from_points(&device, &points, &options)`**: voxels holding a point are solid. A closed surface's inside reads as free beyond its shell.
- **`SdfGrid::from_depth(&device, &image, robot_mask, behind, &options)`**: one depth image (below).
- **`OccupancyMap`**: depth images fused over time (below).

**Grids err toward collision.** Each grid point holds at most the true signed distance (for points and depth images, the distance to the solid voxels as cubes), lowered by half a voxel diagonal: the most trilinear interpolation can overestimate by. A built grid therefore never reads farther from an obstacle than its geometry, and a wall thinner than a voxel still blocks. In return, obstacles grow: by up to a voxel diagonal for meshes, and about three voxels for points and depth images.

**Padding.** Outside its box, a grid reads the value at the nearest box point plus the distance to it. That is exact for collision checking when the box reaches beyond every surface by at least the collision margin plus the robot's largest sphere radius, which `SdfOptions::padding` sets. A grid's diagonal is limited to 4,096 voxels.

## Depth images

A `DepthImage` is one frame: z distances in meters row by row (zero, negative or non-finite where a pixel has no reading), its width, pinhole `Intrinsics`, and the camera's pose (OpenCV axes: x right, y down, z forward). `image.points(robot_mask)` gives each pixel's reading as a point in the frame the camera is posed in, pixel `i` being column `i % width` of row `i / width` with its centre at those whole coordinates, and `None` where the pixel has no reading or reads the robot (below). `from_depth` classifies each voxel by projecting its centre into the image, so an 8×8 time-of-flight array gives solid surfaces rather than 64 points. Dense images also mark the voxel of every pixel. Unobserved space is:
- free outside the image and along pixels without a reading;
- free or solid behind observed surfaces, as `Occlusion` says (`Occlusion::Occupied` is the safe choice for a single view that does not see the robot). `Occlusion::Band(distance)` makes solid only what lies within `distance` behind an observed surface in depth (along the camera's z axis), and space past it free: a drawer pulled out over a table walls in `distance` below what the camera sees of it, not the space under it down to the table. The band grows toward collision as surfaces do, and a band of 0 is `Occlusion::Free`.

### Readings between two surfaces

A depth camera reads a pixel that covers both a near surface and what lies behind it somewhere between the two. Those readings are points in mid-air along every edge: obstacles where nothing is, in the space beside an object that a hand reaches into. `image.filtered(&DepthFilter { noise, ..Default::default() })` gives the image's depths with them left out (0, as a pixel without a reading), for an image that the mask, the points, a surface, a grid and a map then read, as the example under [Fused maps](#fused-maps) does.

A reading is left out where, along a row, a column or a diagonal through its pixel, the pixel on one side is nearer by more than one surface can step, or has no reading, and the pixel on the other side is farther by more than that, or has no reading, at least one of the two a step. What one surface can step between two pixels is the width they are apart at that depth times the tangent of `steepest` (80 degrees from square on to the rays that meet it, at the image's middle; a few degrees steeper toward the corners of a wide lens), and `deviations` (4) of the noise in the difference of two readings. `noise` gives a reading's standard deviation at depth z as `noise[0] + noise[1] z + noise[2] z²`: for a stereo camera `[0, 0, e / (f b)]`, with a disparity error of `e` pixels, a focal length of `f` pixels and `b` meters between its two cameras. So:
- the surface in front keeps its readings up to its outline, and so does what lies behind it. A wire one pixel wide stays, and so does what is seen through a slit;
- a reading with a step on one side and no reading straight across from it goes: what the camera did not read there may be the other surface, as in the strip beside a near object that only one of a stereo camera's two cameras sees;
- depth without noise loses those readings and surfaces seen more edge-on than `steepest`, and nothing else. Depth with noise the filter is not told of loses readings of flat surfaces too, wherever noise sets neighbours that far apart;
- mixed readings that agree with one another stay, a few pixels along an edge. Fusing more than one image clears those: a view that sees through them counts against them.

It runs on the calling thread and takes as long whatever the image holds: 0.7 ms for a 640×480 image on the M4 Pro and 1.0 ms on the Ryzen ([below](#building-grids-on-the-gpu)).

### Masking the robot

A camera that sees the arm would make it an obstacle to itself. `image.robot_mask(&robot, &q, padding, &holding)` says which of the image's readings are the robot's at configuration `q` (`RobotMask::on_robot`, a flag per pixel): those whose points lie inside its collision spheres grown by `padding`, or within `padding` of an object a hand holds. `holding` names those objects, a `Held` each as a problem's `holding` takes one, placed where its hand holds it and with the fingers where it locks them; `&[]` for a robot that holds nothing. With the mask passed to `from_depth`, `OccupancyMap::integrate` or `Surface::from_depth`:
- readings of the robot and of what it holds are not obstacles, and every other reading stays one, between the camera and the arm or past it. A reading is judged by where its point lies, not by what its ray passed: a camera the hand carries stands inside or beside the hand's grown spheres, and keeps all it sees past the hand;
- the space in front of the arm and the space it and what it holds fill are free, even with `Occlusion::Occupied`: nothing else can be where the robot is;
- the space only the arm hides (its shadow) is solid under `Occlusion::Occupied` and free under `Occlusion::OccupiedExceptRobotShadow`, which keeps what other surfaces hide solid. No camera sees behind the arm, so under `Occupied` its shadow walls in the space the arm reaches through next: in the perception demo (`demos/`), with three cameras fused into a map, IK found none of 24 targets. What the shadow holds is unknown either way; `OccupiedExceptRobotShadow` accepts that so the arm can move, and so does `Occlusion::Band`. In maps, a voxel is shadow while only the robot has hidden it: once an image observes it, or hides it behind another surface, it is not shadow again, whatever the order of the images.

Grids from depth images grow toward collision by about three voxels, so set the mask's `padding` above that (4 cm for 1 cm voxels); otherwise the robot's surface touches the space it hides. The padding is also what covers a sensor's errors at the arm's outline: a pixel that mixes the arm with what lies behind it reads somewhere between the two, and such a reading farther than the padding behind the arm is the world's to the mask.

### Shapes the caller holds

A cell's fixtures and the objects a caller has measured as shapes are obstacles of their own in the `World`. Fused into a map as well, each would be held twice, and what reads of it would grow toward collision around the shape itself. `mask.with_shapes(&device, &image, &shapes, margin)` leaves them out as the robot is, for the image the mask was made of:
- a reading within the margin of a shape is the shape's: no obstacle, the space before it free, and what only it hides shadow;
- the space a shape fills is free in a grid or a map of the image, and with it every voxel that reaches into the shape, its middle within half a voxel's diagonal of it (those `map.part` keeps for it). That holds in every look, seen or not, as for the robot's space, but where the look reads something farther than the margin from every shape: such a reading fills its voxel as anywhere, and in a voxel that reaches into a shape it is kept as long as looks read it;
- a reading farther than the margin from every shape stays a reading like any other. What a shape does not cover, as the handle of a mug measured as a cylinder, is the map's.

`margin` is how far from a shape a reading at depth z may lie and be the shape's, `margin[0] + margin[1] z + margin[2] z²`: what the camera's noise reaches there, as three times a `DepthFilter`'s `noise`, and what the shapes and the camera's pose are off by. What only a shape hides is shadow as what only the robot hides is: solid under `Occlusion::Occupied`, and free under `OccupiedExceptRobotShadow` and `Band`, where a caller that gives shapes takes the space behind them for free.

The readings are told apart on the device's threads, each pixel's point once for every shape. A shape is tested only in the pixels its box, grown by the margin, projects to once cut to the camera's view, and only where a point lies in that box: a floor or a wall around the cell costs by what of it the camera may see, and one behind the camera or beside its view costs nothing. With the scene's three shapes a 640×480 image takes 0.2 to 0.3 ms. A fuse with the mask frees the shapes' space as it frees the robot's ([below](#building-grids-on-the-gpu)). It finds that space a row of voxels at a time: every shape but a distance grid is convex, so a row of its box holds one run of voxels that reach into it, found by a few distance tests. A shape then costs a frame by the rows its box spans in the map rather than by their voxels: a wall turned across a corner of the map no longer tests every voxel of that corner, and a shape outside the map costs nothing.

A depth camera also reads the pixels beside a depth jump toward the other surface: one for a sensor that samples each pixel's disparity a fraction of a pixel off, up to half its window for a stereo block matcher. Beside the outline of a shape or the arm, such a reading lies between it and what is behind it, farther than the margin or the padding from either, so it stays a reading: an obstacle in mid-air beside the object, where fingers go. The filter ([above](#readings-between-two-surfaces)) leaves out those far from both surfaces and keeps those near one. `mask.with_band(&device, &image, margin, pixels)` takes such readings from the mask's own things, the robot's spheres, what it holds and the shapes. A reading at depth z that the mask does not flag is taken where a flagged reading within `pixels` pixels of it, along rows, columns and diagonals, is of a thing whose surface where it reads it lies nearer the camera than z by more than `margin` at z, and the reading's own ray, carried on past it, does not enter that thing. A flagged reading is of a sphere or a held object within the padding, as `robot_mask` flags it, and of a shape within `margin` at its depth, as `with_shapes` flags it (`margin` as `with_shapes` takes it); a flag the caller sets on a reading of none of them takes nothing. The reading taken is then that thing's, with all that follows for a flagged reading:
- the readings a camera gives wrong beside an outline lie behind the thing in front: its own outline read too deep, on a ray that missed the thing or has passed through it, and the pixel behind it read toward it. Their rays go on away from the thing, and the band takes them;
- a reading whose ray goes on into a thing lies in front of it, and that thing never takes it: a table given as a shape takes no reading of what stands on it, at any band and from any view, however far the table's own readings below an object lie nearer than the object's lower rows;
- a reading level with a flagged surface is measured against the thing's surface, not against the flagged readings and their noise: within the margin of that surface it is kept;
- given up: an outline read too near, in front of the thing by more than the margin, stays a reading; and the honest readings just behind each flagged outline, `pixels` wide, are taken with the rest, as the table behind an object on it, or an object no shape explains that stands within the band behind another's outline;
- `pixels` is how many pixels beside a jump the camera reads toward the other surface, which its depth method sets, not a number to tune to a scene. A band of 0 changes nothing;
- called after `with_shapes`, it grows the shapes' flags with the robot's; called before, the robot's alone. The space the mask frees in a grid or a map is what it was.

The band runs on the device's threads. A pass along the columns and one along each row find, for each reading, the least depth of a flagged reading within the band less the padding or the margin, which is as much nearer as that reading's thing may lie; a reading no thing can lie nearer than by the margin is kept there, at a cost that goes by the image's pixels times `pixels`. Only the readings that pass are looked into, against the flagged readings of the band and their things, the spheres first and a shape only where the reading lies in its box. A band of one pixel behind the robot alone takes 0.17 to 0.24 ms for a 640×480 image; behind the robot and the scene's three shapes, which flag every reading of that scene already and leave none to look into, 0.17 to 0.23 ms ([below](#building-grids-on-the-gpu)).

`RobotMask::on_robot` holds the flags, one per pixel, and a caller may set more of them: the readings of an object it is about to grasp, as whatever told them apart says, where a fitted shape does not cover the object and its readings would stand in the fingers' way.

## Fused maps

`OccupancyMap::try_new(lo, hi, voxel)` is a fixed grid in the world frame that fuses frames as OctoMap does. Each voxel holds the log-odds that it is occupied: a hit adds 0.85 and a miss subtracts 0.4, clamped to [−2, 3.5]. Something seen once and then seen through three times is free again; something seen in five images or more, nine times. Voxels behind readings, outside the image or beyond the map are not updated. On the CPU device a fuse visits only the voxels it can change, those the image may see, those holding one of its readings and those a mask frees, so its cost follows the view, not the map's size. On the GPU device a fuse sends the map to the device and reads it back, so its cost follows the map's size: a map of 1.8M voxels fuses several times quicker there, and one of 13M quicker on the CPU device (two 640×480 frames in 10 ms on the Ryzen's 32 threads against 19 on its Radeon). Any device fuses a map, whichever the plans run on, so a caller with a large map fuses it on a CPU device.

Those weights are the map's `Weighing`. `Weighing::Evidence`, the default, suits a stream of frames: what comes and goes settles, and one stray reading (noise, or a mixed pixel the filter kept) is outweighed by the looks that see through it. A caller that looks once every few seconds and acts on each look wants `map.with_weighing(Weighing::Latest)`: each image decides what it sees. A voxel it reads a surface in, or that holds one of its readings, is as surely occupied as a map holds anything (the upper clamp, 3.5), and a voxel its rays pass through is as surely free (the lower, −2). What the image does not see, outside it, behind its surfaces or along pixels without a reading, keeps what earlier images said, and space no image has observed, the robot's shadow and hidden space are marked as under `Evidence`. With a mask, the space the robot, what it holds and the mask's shapes fill is as surely free wherever it is, seen or not. One stray reading is then an obstacle until a look sees through it.

`map.grid(&device, unknown)` makes voxels more likely occupied than not solid, and voxels never observed as `unknown` says. A map keeps no rays, so under `Occlusion::Band(distance)` the band is every unobserved voxel within `distance` of an occupied one, in any direction: in front of a surface too where its pixels had no reading, and beside the view. The robot's shadow stays free. The band builds the grid twice.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::{
    DepthFilter, DepthImage, Device, Error, Intrinsics, Obstacle, Occlusion, OccupancyMap, Pose,
    Robot, RobotOptions, World,
};

fn main() -> Result<(), Error> {
    let robot = Robot::load("assets/franka/franka_panda.urdf", &RobotOptions::default())?;
    let device = Device::gpu(&robot).or_else(|_| Device::cpu(&robot))?;
    let mut map =
        OccupancyMap::try_new(Vec3::new(-0.2, -0.8, -0.1), Vec3::new(1.2, 0.8, 1.0), 0.01)?;
    // Your camera's pose in OpenCV axes (z forward, y down): here 1.5 m up, looking down.
    let camera = Pose::new(
        Vec3::new(0.5, 0.0, 1.5),
        Quat::from_rotation_x(std::f32::consts::PI),
    );
    let intrinsics = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let q = robot.default_q().to_vec(); // the arm's current state
    // A stereo camera's noise: 0.08 pixels of disparity, its two cameras 5 cm apart.
    let filter = DepthFilter {
        noise: [0.0, 0.0, 0.08 / (intrinsics.fx * 0.05)],
        ..Default::default()
    };
    for depth in frames() {
        let image = DepthImage {
            depth: &depth,
            width: 640,
            intrinsics,
            camera,
        };
        let depth = image.filtered(&filter)?; // no readings between two surfaces
        let image = DepthImage {
            depth: &depth,
            ..image
        };
        let mask = image.robot_mask(&robot, &q, 0.04, &[])?; // the arm is not an obstacle
        map.integrate(&device, &image, Some(&mask))?;
    }
    let grid = map.grid(&device, Occlusion::Occupied)?; // never-seen space is solid
    let world = World::new(vec![Obstacle::from(grid)]);
    let _worlds = device.upload(&[world])?;
    Ok(())
}

/// Depth frames from your camera driver.
fn frames() -> Vec<Vec<f32>> {
    vec![vec![1.0; 640 * 480]]
}
```

### What a caller knows, and one object's part

A map holds what the cameras saw. A caller knows more, and says so in the map's frame:

- **`map.free(&shape, margin)`** marks the voxels within `margin` of a shape free, as one image weighed as evidence sees through them, or as they were where cameras saw through them more: where an object stood before the robot carried it away, which no camera has looked at since. One reading there afterwards makes a voxel the shape held occupied again. A shape the world holds as an obstacle of its own is kept out of the map where images are fused, by the mask ([above](#shapes-the-caller-holds)).
- **`map.clear(&shape, margin)`** marks them never observed: what was seen there no longer holds, as where a drawer stood before it was pulled and where it stands now. A grid takes them as it is told to take never-observed space.
- **`map.part(&region, padding)`** is the map inside one region, as a map of its own that reaches `padding` past the region's bounds (as a grid must reach past the solid it holds, [above](#distance-grids)), every voxel that does not reach into the region left out: no solid in its grid whatever `unknown` says. A voxel reaches into it where its middle lies within half a voxel's diagonal of it, so that a reading on the region's own face is kept whichever side of the face its voxel's middle lies. `map.part(&object, padding)?.grid(&device, Occlusion::Occupied)` is one object as the cameras saw into it: solid where a reading landed or nothing saw past, free where a camera saw in, and nothing around it, where the whole map's grid under `Occupied` would wall in everything no camera has seen. Where the region reaches past the map it was never observed, and past a part it is taken from, it is left out. A part takes no images: fuse into the map and take the part again. Its points cover the region and the padding whatever the map covers, so a region that reaches past the map costs by its own size, and one whose points would number more than 2^26 (67 million), or whose diagonal would reach 4,096 voxels, is an input error, as a grid that large is.
- **`map.counts()`** says how many voxels are occupied, free and never observed, of a part those inside its region: a region with free voxels was seen into, and one that is all occupied and never observed was not.
- **`map.occupied()`** says where: the middles of the voxels `counts` counts as occupied, in the map's frame, and of a part none it leaves out. A caller that keeps the shapes it holds out of the map ([`with_shapes`](#shapes-the-caller-holds)) asks it after each look. Where never-observed space is taken for free (`Occlusion::Free`), none means there is no grid to make, and otherwise `map.part(&bounds, padding)` around them is a grid of what is there and no more.

The shape is any `Obstacle`, turned as it is: a measured object's box, a ball about a grasp, a grid. These run on the calling thread: on the M4 Pro, doing other work beside (a load of 15 on its 14 cores), the part of a 1.8M-voxel map in a box of the scene takes 0.1 ms and its grid of 94,000 points 1.3 to 2.0 ms on either device, and a copy of the map with a shape marked free, then cleared, 1.7 ms. Where that map is occupied, 11,000 to 18,000 voxels, takes 0.5 ms, at a load of 24 (`examples/perception.rs`).

`examples/depth.rs` renders a 640×480 depth image of a tabletop scene, builds a grid of 172×192×72 points from it, and plans Panda reaches with the grid as the only obstacle. IK reaches 48 of 64 grasp targets (the others lie in the camera's shadow, which `Occlusion::Occupied` treats as solid); all 48 plan, and their closest approach to the true scene is 27.0 mm.

## With a perception library

batchplan reads depth; it does not say what things are. It cleans a camera's depths, says which readings are the robot's, keeps what the cameras saw of space and gives its distance field. A library that says what things are (which pixels are one thing, its shape, what it stands on, what is still there while hidden) meets batchplan at plain data, so neither depends on the other:

| | batchplan | The perception library |
|---|---|---|
| A camera's depths | Leaves out the readings between two surfaces (`filtered`) and says which are the robot's (`robot_mask`), for both to read | Reads them, with the robot's flags (`RobotMask::on_robot`) |
| Things | Takes each as a shape (an `Obstacle` in the `World`, a problem's `holding` once picked, and left out of the map by `with_shapes`) and as flags per pixel (`Surface::from_depth`, for `grasps_on`, `rests_on` and `falls_into`); gives the distance to a shape known beforehand (`obstacle.distance(point)`) | Tells them apart, measures each as a shape, says what each stands on, and keeps the ones no camera can see |
| Space | Holds what the cameras saw free, filled or never (`OccupancyMap`), and gives its distance field (`grid`, `part`) | Nothing |

### One memory for each thing

A scene remembered twice, as objects and as voxels, has to be kept in step by whoever uses both, and a fact said to one and not the other leaves a ghost. So each thing has one memory:

- **What the caller holds as a shape is left out of the map** ([above](#shapes-the-caller-holds)): the robot, what a hand holds, the cell's fixtures, and the objects the perception library measured, seen or kept unseen. The shape itself is the obstacle.
- **Everything else the cameras read is the map's**: what no object was made of, what a shape does not cover, and a thing whose shape is a poor solid, as a cabinet with a drawer open, for which the perception library's box only names a region (`map.part`).
- **A plan is checked against both together**, the shapes and the map's grid. Nothing is taken out of that but readings within the margin of a shape, and what a voxel that reaches into one held from looks that read it no longer, so where the two differ the plan is the more careful one. Three choices give some of that up, and are the caller's to make: an occlusion rule under which shadow is free, readings it flags itself, and a band behind the mask's flags ([above](#shapes-the-caller-holds)). The band's readings belong to no memory: they are left out as the shape's are, the ones a camera reads behind its outline and with them what lies just behind it. What stands in front of a shape, its rays going on into the shape, keeps its readings in the map.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::{
    DepthFilter, DepthImage, Device, Error, Intrinsics, Obstacle, Occlusion, OccupancyMap, Pose,
    Robot, RobotOptions, Surface, Weighing, World,
};

/// Stands in for a perception library: what it says one look shows.
mod perception {
    use batchplan::Obstacle;
    use batchplan::glam::Vec3;

    /// The objects a look found, each measured as a shape, and per pixel which object its
    /// reading is of.
    #[derive(Default)]
    pub struct Scene {
        pub shapes: Vec<Obstacle>,
        pub object: Vec<Option<usize>>,
    }

    /// One camera's depths, which of them are the robot's and the distance to each shape known
    /// beforehand, after the scene before.
    pub fn look(
        depth: &[f32],
        robot: &[bool],
        known: &[&(dyn Fn(Vec3) -> f32 + Sync)],
        previous: Scene,
    ) -> Scene {
        let _ = (robot, known);
        Scene {
            shapes: previous.shapes,
            object: vec![None; depth.len()],
        }
    }
}

fn main() -> Result<(), Error> {
    let arm = Robot::load("assets/franka/franka_panda.urdf", &RobotOptions::default())?;
    let device = Device::gpu(&arm).or_else(|_| Device::cpu(&arm))?;
    let q = arm.default_q().to_vec();
    // The cell, known without looking: a table.
    let fixtures = [Obstacle::cuboid(
        Vec3::new(0.5, 0.0, -0.05),
        Vec3::new(0.5, 0.6, 0.05),
    )];
    // A caller that looks now and then lets each look decide what it sees.
    let mut map =
        OccupancyMap::try_new(Vec3::new(-0.2, -0.8, -0.1), Vec3::new(1.2, 0.8, 0.9), 0.01)?
            .with_weighing(Weighing::Latest);

    // One look. The depths come from your camera driver.
    let depth = vec![1.0; 640 * 480];
    let intrinsics = Intrinsics {
        fx: 525.0,
        fy: 525.0,
        cx: 319.5,
        cy: 239.5,
    };
    let image = DepthImage {
        depth: &depth,
        width: 640,
        intrinsics,
        camera: Pose::new(
            Vec3::new(0.5, 0.0, 1.5),
            Quat::from_rotation_x(std::f32::consts::PI),
        ),
    };
    // batchplan: the depths without readings between two surfaces, and the robot's readings.
    let filter = DepthFilter {
        noise: [0.0, 0.0, 0.08 / (intrinsics.fx * 0.05)],
        ..Default::default()
    };
    let depth = image.filtered(&filter)?;
    let image = DepthImage {
        depth: &depth,
        ..image
    };
    let robot = image.robot_mask(&arm, &q, 0.04, &[])?;

    // The perception library: what things there are. After a pick, a place or a pull, the scene
    // passed in says so: that is the one place such a fact is said.
    let distances: Vec<Box<dyn Fn(Vec3) -> f32 + Sync>> = fixtures
        .iter()
        .map(|fixture| Box::new(move |p| fixture.distance(p).value) as _)
        .collect();
    let known: Vec<&(dyn Fn(Vec3) -> f32 + Sync)> = distances.iter().map(|d| &**d).collect();
    let scene = perception::look(&depth, &robot.on_robot, &known, perception::Scene::default());

    // What is held as a shape is left out of the map: a reading is a shape's within three
    // deviations of the camera's noise and the 5 mm this cell's shapes and camera are known
    // to, and the space each shape fills is free there. Noise alone would leave a camera
    // stated without any a margin of zero. This camera reads the pixel beside each depth jump
    // toward the other surface: behind a shape's outline, on a ray that goes on past the shape,
    // that reading is the shape's too.
    let shapes: Vec<Obstacle> = fixtures.iter().chain(&scene.shapes).cloned().collect();
    let margin = [0.005, 0.0, 3.0 * filter.noise[2]];
    let held = (robot.clone())
        .with_shapes(&device, &image, &shapes, margin)?
        .with_band(&device, &image, margin, 1)?;
    map.integrate(&device, &image, Some(&held))?;

    // The world a plan is checked against: the shapes, and whatever was read that none
    // explains. Where they explain all of it, there is no grid to make.
    let mut obstacles = shapes;
    if map.counts().occupied > 0 {
        obstacles.push(Obstacle::from(map.grid(&device, Occlusion::Free)?));
    }
    let _worlds = device.upload(&[World::new(obstacles)])?;

    // Grasps and rests of one object are taken on its pixels: the library's labels are the
    // surface's flags, and the robot's mask alone stays the surface's mask.
    let object: Vec<bool> = scene.object.iter().map(|of| *of == Some(0)).collect();
    let _surface = Surface::from_depth(&device, &image, &object, Some(&robot), 0.005)?;
    Ok(())
}
```

The `perception` module stands in for the library; every call on batchplan's side is as a program run against one makes it, with the library's shapes turned into `Obstacle`s field for field.

Said once, each to the memory that holds the thing:
- **picked, placed, or gone**: to the scene. The map never held the object, so there is nothing to mark, and an object someone took away leaves the world in the look that sees its place empty;
- **an object kept unseen**, as one the arm hides: it stays a shape, and the map reads free where it stands because the shape explains that space. If someone took it meanwhile, plans stay clear of it until a camera sees through;
- **a pull or a push of something the map holds**, as a drawer: to the map, `map.clear(&shape, margin)` where it stood and where it stands.

A thing passes from one memory to the other as the map weighs images ([`Weighing`](#fused-maps)). One that becomes a shape is no longer read from that look on, but what earlier looks filled around it stays until seen through: `map.free(&shape, reach)` before the look is fused frees it at once, with a reach of the mask's margin and half a voxel's diagonal, so that it takes the voxels its readings fell in; after the fuse it would take what that look read beside the shape as well. One that stops being a shape is read from that look on, and solid once its readings outweigh the looks that saw its place empty, three at most; `map.clear(&shape, reach)` where it stood makes its first reading count. In a map where one image decides, as a caller that looks now and then wants, a thing that stops being a shape is solid at its first reading, and one that becomes a shape is free in that look as far as its voxels reach into it; `free` is then only for readings that lay farther from the shape than that.

Never-observed space is taken two ways, by where it lies:
- Around the scene it is free, as it is for a world of shapes alone: `map.grid(&device, Occlusion::Free)`, and none where `map.occupied()` gives no point. Where it gives a few, the grid of a `map.part` around them is all there is to make, however large the map.
- Inside a region the map holds for one large object it is solid: `map.part(&region, padding)?.grid(&device, Occlusion::Occupied)`. The copy the first grid is made of has that region marked free as far as the part reaches, `free(&region, 0.5 * 3f32.sqrt() * voxel)`, so the object's matter is in one grid only.

## Building grids on the GPU

Point clouds, depth images and maps become grids on the device they are given. On a GPU, everything after the host computes a depth image's bounds, and with a mask the grid points the robot fills, runs in compute kernels; the CPU device runs the same steps on its threads. Which points the robot fills is found sphere by sphere over the points near each, once for both devices, so a robot fitted with several hundred spheres costs a frame what one of a few dozen does.

- **The distance transform is exact on both devices**: Felzenszwalb and Huttenlocher's separable transform in integers, so GPUs and CPUs build identical grids from the same occupancy.
- **Classification projects voxel centres in floating point**, so a voxel within rounding of a pixel's edge may land either way: 70 of 1.67 million voxels on Metal, 51 on Mesa's RADV, for a 640×480 image of a table with the arm before it, masked out.

Median build times for a 640×480 image of a table seen from 1 m:

| Machine | 1 cm grid (0.96M points) | 5 mm grid (7.5M) | Fusing one frame (1.8M-point map) | That map's grid |
|---|---|---|---|---|
| Radeon 8060S (GPU) | 1.9 ms | 10.6 ms | 0.6 ms | 2.5 ms |
| Ryzen AI Max+ 395 (CPU, 32 threads) | 9.1 ms | 48 ms | 4.8 ms | 6.2 ms |
| M4 Pro (GPU) | 3.6 ms | 14.8 ms | 1.5 ms | 4.4 ms |
| M4 Pro (CPU, 14 threads) | 6.4 ms | 36 ms | 2.2 ms | 6.9 ms |

`examples/perception.rs` times each call for a 640×480 image that shows the arm: a tabletop seen from across the table, and from a camera on the Panda's palm that looks along its fingers. The mask is padded 4 cm, for the Panda's 61 spheres and for a Panda fitted with 573. Each figure is the quickest median of several runs (twenty on the M4 Pro, ten for the four rows of its CPU fusing, eight on the Ryzen and its Radeon, eight on each device for the two rows of the 13.4M-voxel map and for the fuse with the scene's three shapes left out, eight for those shapes' readings told apart and eight for the two rows of the band, on the M4 Pro each after a pause, as its first ten seconds of work run quicker than the rest) on machines doing other work beside them (a load of 3 to 10 on the M4 Pro's 14 cores and 13 to 14 for its CPU fusing; 10 to 32 on the Ryzen's 32, 7 to 14 for its CPU fusing; 3.5 to 7.5 on the M4 Pro and 2 to 10 on the Ryzen for the 13.4M-voxel map; 4.5 to 7.6 on the M4 Pro and 8 to 15.5 on the Ryzen for the fuse with the three shapes; 7 to 81 on the M4 Pro and 4.7 to 14 on the Ryzen for their readings told apart; 4.7 to 9.0 on the M4 Pro and 3.5 to 12.2 on the Ryzen for the band); each cell is the camera over the table, then the one on the hand.

The filter, the masks, the points and a surface are read on the host:

| | M4 Pro (14 threads) | Ryzen AI Max+ 395 (32 threads) |
|---|---|---|
| The image without the readings between two surfaces (`filtered`) | 0.73 ms, 0.73 ms | 0.98 ms, 0.99 ms |
| Which readings are the robot's (`robot_mask`), 61 spheres | 0.07 ms, 0.42 ms | 0.06 ms, 0.41 ms |
| The same, 573 spheres | 0.20 ms, 1.05 ms | 0.21 ms, 1.12 ms |
| The scene's three shapes left out too (`with_shapes`) | 0.26 ms, 0.31 ms | 0.23 ms, 0.27 ms |
| A band of a pixel behind the robot's mask alone, 61 spheres (`with_band`) | 0.17 ms, 0.20 ms | 0.23 ms, 0.24 ms |
| The same behind the robot and the three shapes, which flag every reading of this scene already | 0.17 ms, 0.18 ms | 0.22 ms, 0.23 ms |
| The image's points (`points`) | 0.66 ms, 0.71 ms | 1.77 ms, 1.97 ms |
| A surface with the table's normals (`Surface::from_depth`; 148,461 and 208,100 normals) | 3.0 ms, 3.8 ms | 3.4 ms, 3.4 ms |

Fusing and grids run on the device:

| | M4 Pro (CPU, 14 threads) | M4 Pro (GPU) | Ryzen AI Max+ 395 (CPU, 32 threads) | Radeon 8060S (GPU) |
|---|---|---|---|---|
| Fusing the image into a 1.8M-voxel map, no mask | 1.6 ms, 1.5 ms | 1.2 ms, 1.2 ms | 3.8 ms, 3.0 ms | 0.8 ms, 0.9 ms |
| With a mask of 61 spheres | 2.0 ms, 1.7 ms | 1.6 ms, 1.6 ms | 4.5 ms, 3.0 ms | 1.1 ms, 1.4 ms |
| With a mask of 573 spheres | 2.3 ms, 1.8 ms | 1.7 ms, 1.9 ms | 4.5 ms, 3.3 ms | 1.6 ms, 1.6 ms |
| With the robot's mask and the scene's three shapes left out | 1.3 ms, 0.9 ms | 1.5 ms, 1.6 ms | 2.8 ms, 1.4 ms | 1.0 ms, 1.2 ms |
| Fusing the image into a 13.4M-voxel map (2.9 m by 2.9 m by 1.55 m), no mask | 2.4 ms, 1.6 ms | 4.7 ms, 6.5 ms | 5.2 ms, 3.1 ms | 8.4 ms, 8.3 ms |
| With a mask of 61 spheres | 3.1 ms, 1.9 ms | 5.6 ms, 5.7 ms | 6.5 ms, 3.5 ms | 9.5 ms, 9.7 ms |
| A 1 cm grid of the image, no mask (3.1M and 2.1M points) | 14.1 ms, 10.1 ms | 6.4 ms, 5.4 ms | 21 ms, 14.0 ms | 7.4 ms, 5.7 ms |
| With a mask of 61 spheres (2.3M and 1.6M points) | 11.7 ms, 8.7 ms | 6.0 ms, 5.1 ms | 16.6 ms, 12.6 ms | 6.1 ms, 5.6 ms |
| With a mask of 573 spheres | 11.5 ms, 8.8 ms | 6.0 ms, 5.4 ms | 16.4 ms, 12.7 ms | 6.4 ms, 5.1 ms |

A mask's cost goes by the pixels the robot covers and, where it is used, by the volume it fills, not by how many spheres cover it.
