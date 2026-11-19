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

### Masking the robot

A camera that sees the arm would make it an obstacle to itself. `image.robot_mask(&robot, &q, padding, &holding)` says which of the image's readings are the robot's at configuration `q` (`RobotMask::on_robot`, a flag per pixel): those whose points lie inside its collision spheres grown by `padding`, or within `padding` of an object a hand holds. `holding` names those objects, a `Held` each as a problem's `holding` takes one, placed where its hand holds it and with the fingers where it locks them; `&[]` for a robot that holds nothing. With the mask passed to `from_depth`, `OccupancyMap::integrate` or `Surface::from_depth`:
- readings of the robot and of what it holds are not obstacles, and every other reading stays one, between the camera and the arm or past it. A reading is judged by where its point lies, not by what its ray passed: a camera the hand carries stands inside or beside the hand's grown spheres, and keeps all it sees past the hand;
- the space in front of the arm and the space it and what it holds fill are free, even with `Occlusion::Occupied`: nothing else can be where the robot is;
- the space only the arm hides (its shadow) is solid under `Occlusion::Occupied` and free under `Occlusion::OccupiedExceptRobotShadow`, which keeps what other surfaces hide solid. No camera sees behind the arm, so under `Occupied` its shadow walls in the space the arm reaches through next: in the perception demo (`demos/`), with three cameras fused into a map, IK found none of 24 targets. What the shadow holds is unknown either way; `OccupiedExceptRobotShadow` accepts that so the arm can move, and so does `Occlusion::Band`. In maps, a voxel is shadow while only the robot has hidden it: once an image observes it, or hides it behind another surface, it is not shadow again, whatever the order of the images.

Grids from depth images grow toward collision by about three voxels, so set the mask's `padding` above that (4 cm for 1 cm voxels); otherwise the robot's surface touches the space it hides. The padding is also what covers a sensor's errors at the arm's outline: a pixel that mixes the arm with what lies behind it reads somewhere between the two, and such a reading farther than the padding behind the arm is the world's to the mask.

## Fused maps

`OccupancyMap::try_new(lo, hi, voxel)` is a fixed grid in the world frame that fuses frames as OctoMap does. Each voxel holds the log-odds that it is occupied: a hit adds 0.85 and a miss subtracts 0.4, clamped to [−2, 3.5]. Something seen once and then seen through three times is free again; something seen in five images or more, nine times. Voxels behind readings, outside the image or beyond the map are not updated.

`map.grid(&device, unknown)` makes voxels more likely occupied than not solid, and voxels never observed as `unknown` says. A map keeps no rays, so under `Occlusion::Band(distance)` the band is every unobserved voxel within `distance` of an occupied one, in any direction: in front of a surface too where its pixels had no reading, and beside the view. The robot's shadow stays free. The band builds the grid twice.

```rust,no_run
use batchplan::glam::{Quat, Vec3};
use batchplan::{
    DepthImage, Device, Error, Intrinsics, Obstacle, Occlusion, OccupancyMap, Pose, Robot,
    RobotOptions, World,
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
    for depth in frames() {
        let image = DepthImage {
            depth: &depth,
            width: 640,
            intrinsics,
            camera,
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

- **`map.free(&shape, margin)`** marks the voxels within `margin` of a shape free, as seen through once, or as they were where cameras saw through them more: where an object stood before the robot carried it away, which no camera has looked at since. It is also how a shape the world already holds as an obstacle of its own is kept out of a grid, on a copy of the map (`map.clone()`; a map is a byte a voxel). One reading there afterwards makes a voxel the shape held occupied again.
- **`map.clear(&shape, margin)`** marks them never observed: what was seen there no longer holds, as where a drawer stood before it was pulled and where it stands now. A grid takes them as it is told to take never-observed space.
- **`map.part(&region, padding)`** is the map inside one region, as a map of its own that reaches `padding` past the region's bounds (as a grid must reach past the solid it holds, [above](#distance-grids)), every voxel that does not reach into the region left out: no solid in its grid whatever `unknown` says. A voxel reaches into it where its middle lies within half a voxel's diagonal of it, so that a reading on the region's own face is kept whichever side of the face its voxel's middle lies. `map.part(&object, padding)?.grid(&device, Occlusion::Occupied)` is one object as the cameras saw into it: solid where a reading landed or nothing saw past, free where a camera saw in, and nothing around it, where the whole map's grid under `Occupied` would wall in everything no camera has seen. Where the region reaches past the map it was never observed, and past a part it is taken from, it is left out. A part takes no images: fuse into the map and take the part again.
- **`map.counts()`** says how many voxels are occupied, free and never observed, of a part those inside its region: a region with free voxels was seen into, and one that is all occupied and never observed was not.

The shape is any `Obstacle`, turned as it is: a measured object's box, a ball about a grasp, a grid. These run on the calling thread: on the M4 Pro, doing other work beside (a load of 15 on its 14 cores), the part of a 1.8M-voxel map in a box of the scene takes 0.1 ms and its grid of 94,000 points 1.3 to 2.0 ms on either device, and a copy of the map with a shape marked free, then cleared, 1.7 ms (`examples/perception.rs`).

`examples/depth.rs` renders a 640×480 depth image of a tabletop scene, builds a grid of 172×192×72 points from it, and plans Panda reaches with the grid as the only obstacle. IK reaches 48 of 64 grasp targets (the others lie in the camera's shadow, which `Occlusion::Occupied` treats as solid); all 48 plan, and their closest approach to the true scene is 27.0 mm.

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

`examples/perception.rs` times each call for a 640×480 image that shows the arm: a tabletop seen from across the table, and from a camera on the Panda's palm that looks along its fingers. The mask is padded 4 cm, for the Panda's 61 spheres and for a Panda fitted with 573. Each figure is the quickest median of several runs (twenty on the M4 Pro, eight on the Ryzen and its Radeon) on machines doing other work beside them (a load of 6 to 8 on the M4 Pro's 14 cores, 10 to 32 on the Ryzen's 32); each cell is the camera over the table, then the one on the hand.

The mask, the points and a surface are read on the host:

| | M4 Pro (14 threads) | Ryzen AI Max+ 395 (32 threads) |
|---|---|---|
| Which readings are the robot's (`robot_mask`), 61 spheres | 0.07 ms, 0.42 ms | 0.06 ms, 0.41 ms |
| The same, 573 spheres | 0.20 ms, 1.05 ms | 0.21 ms, 1.12 ms |
| The image's points (`points`) | 0.66 ms, 0.71 ms | 1.77 ms, 1.97 ms |
| A surface with the table's normals (`Surface::from_depth`; 148,461 and 208,100 normals) | 3.0 ms, 3.8 ms | 3.4 ms, 3.4 ms |

Fusing and grids run on the device:

| | M4 Pro (CPU, 14 threads) | M4 Pro (GPU) | Ryzen AI Max+ 395 (CPU, 32 threads) | Radeon 8060S (GPU) |
|---|---|---|---|---|
| Fusing the image into a 1.8M-voxel map, no mask | 1.8 ms, 1.8 ms | 1.2 ms, 1.2 ms | 4.9 ms, 4.6 ms | 0.8 ms, 0.9 ms |
| With a mask of 61 spheres | 2.1 ms, 2.1 ms | 1.6 ms, 1.6 ms | 5.4 ms, 5.0 ms | 1.1 ms, 1.4 ms |
| With a mask of 573 spheres | 2.4 ms, 2.3 ms | 1.7 ms, 1.9 ms | 5.7 ms, 5.5 ms | 1.6 ms, 1.6 ms |
| A 1 cm grid of the image, no mask (3.1M and 2.1M points) | 14.1 ms, 10.1 ms | 6.4 ms, 5.4 ms | 21 ms, 14.0 ms | 7.4 ms, 5.7 ms |
| With a mask of 61 spheres (2.3M and 1.6M points) | 11.7 ms, 8.7 ms | 6.0 ms, 5.1 ms | 16.6 ms, 12.6 ms | 6.1 ms, 5.6 ms |
| With a mask of 573 spheres | 11.5 ms, 8.8 ms | 6.0 ms, 5.4 ms | 16.4 ms, 12.7 ms | 6.4 ms, 5.1 ms |

A mask's cost goes by the pixels the robot covers and, where it is used, by the volume it fills, not by how many spheres cover it.
