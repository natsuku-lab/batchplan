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

Link names are resolved when the worlds are uploaded, against the device's robot; an unknown name, or an obstacle index past the end, is an input error. A held object is a link of the robot holding it ([holding objects](robots.md#holding-objects)), so a world where it may rest on a table is uploaded to the device for that robot.

## Scenes

`World::load(path, &SdfOptions)` reads the static geometry of an MJCF or OpenUSD scene as obstacles. Planes become slabs and ellipsoids their bounding boxes. Meshes become distance grids of their convex hulls where the format collides them that way: MJCF always (as MuJoCo does), USD when `physics:approximation` is `convexHull`. Other USD approximations use the exact mesh.

## Distance grids

An `SdfGrid` holds signed distances on a regular grid, stored as half floats and interpolated trilinearly on both devices. Worlds can share a grid through its `Arc`, and devices and exporters then store it once. `SdfOptions` lays grids out (1 cm voxels and 15 cm padding by default).

- **`SdfGrid::from_mesh(&vertices, &triangles, &options)`**: exact signed distances of a closed triangle mesh, inside and outside decided by parry3d's pseudo-normals. Open or non-manifold meshes have no inside and are refused with the reason.
- **`SdfGrid::from_points(&device, &points, &options)`**: voxels holding a point are solid. A closed surface's inside reads as free beyond its shell.
- **`SdfGrid::from_depth(&device, &image, robot_mask, behind, &options)`**: one depth image (below).
- **`OccupancyMap`**: depth images fused over time (below).

**Grids err toward collision.** Each grid point holds at most the true signed distance (for points and depth images, the distance to the solid voxels as cubes), lowered by half a voxel diagonal: the most trilinear interpolation can overestimate by. A built grid therefore never reads farther from an obstacle than its geometry, and a wall thinner than a voxel still blocks. In return, obstacles grow: by up to a voxel diagonal for meshes, and about three voxels for points and depth images.

**Padding.** Outside its box, a grid reads the value at the nearest box point plus the distance to it. That is exact for collision checking when the box reaches beyond every surface by at least the collision margin plus the robot's largest sphere radius, which `SdfOptions::padding` sets. A grid's diagonal is limited to 4,096 voxels.

## Depth images

A `DepthImage` is one frame: z distances in meters row by row (zero, negative or non-finite where a pixel has no reading), its width, pinhole `Intrinsics`, and the camera's pose (OpenCV axes: x right, y down, z forward). `from_depth` classifies each voxel by projecting its centre into the image, so an 8×8 time-of-flight array gives solid surfaces rather than 64 points. Dense images also mark the voxel of every pixel. Unobserved space is:
- free outside the image and along pixels without a reading;
- free or solid behind observed surfaces, as `Occlusion` says (`Occlusion::Occupied` is the safe choice for a single view that does not see the robot).

### Masking the robot

A camera that sees the arm would make it an obstacle to itself. `image.robot_mask(&robot, &q, padding)` says where the robot is in the image at configuration `q`, its collision spheres grown by `padding`. With the mask passed to `from_depth` or `OccupancyMap::integrate`:
- readings on the robot are not obstacles, and anything between the camera and the arm stays one;
- the space in front of the arm and the space it fills are free, even with `Occlusion::Occupied`: nothing else can be where the robot is;
- the space only the arm hides (its shadow) is solid under `Occlusion::Occupied` and free under `Occlusion::OccupiedExceptRobotShadow`, which keeps what other surfaces hide solid. No camera sees behind the arm, so under `Occupied` its shadow walls in the space the arm reaches through next: in the perception demo (`demos/`), with three cameras fused into a map, IK found none of 24 targets. What the shadow holds is unknown either way; `OccupiedExceptRobotShadow` accepts that so the arm can move. In maps, a voxel only the robot has hidden stays shadow until an image observes it.

Grids from depth images grow toward collision by about three voxels, so set the mask's `padding` above that (4 cm for 1 cm voxels); otherwise the robot's surface touches the space it hides.

## Fused maps

`OccupancyMap::try_new(lo, hi, voxel)` is a fixed grid in the world frame that fuses frames as OctoMap does. Each voxel holds the log-odds that it is occupied: a hit adds 0.85 and a miss subtracts 0.4, clamped to [−2, 3.5]. Something seen once and then seen through three times is free again. Voxels behind readings, outside the image or beyond the map are not updated.

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
        let mask = image.robot_mask(&robot, &q, 0.04)?; // the arm is not an obstacle
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

`examples/depth.rs` renders a 640×480 depth image of a tabletop scene, builds a grid of 172×192×72 points from it, and plans Panda reaches with the grid as the only obstacle. IK reaches 48 of 64 grasp targets (the others lie in the camera's shadow, which `Occlusion::Occupied` treats as solid); all 48 plan, and their closest approach to the true scene is 29 mm.

## Building grids on the GPU

Point clouds, depth images and maps become grids on the device they are given. On a GPU, everything after the host computes a depth image's bounds runs in compute kernels; the CPU device runs the same steps on its threads.

- **The distance transform is exact on both devices**: Felzenszwalb and Huttenlocher's separable transform in integers, so GPUs and CPUs build identical grids from the same occupancy.
- **Classification projects voxel centres in floating point**, so a voxel within rounding of a pixel's edge may land either way: 2 of 600,000 voxels on Metal, 51 on Mesa's RADV, in a 640×480 image with the arm masked out.

Median build times for a 640×480 image of a table seen from 1 m:

| Machine | 1 cm grid (0.96M points) | 5 mm grid (7.5M) | Fusing one frame (1.8M-point map) | That map's grid |
|---|---|---|---|---|
| Radeon 8060S (GPU) | 1.9 ms | 10.6 ms | 0.6 ms | 2.5 ms |
| Ryzen AI Max+ 395 (CPU, 32 threads) | 9.1 ms | 48 ms | 4.8 ms | 6.2 ms |
| M4 Pro (GPU) | 3.6 ms | 14.8 ms | 1.5 ms | 4.4 ms |
| M4 Pro (CPU, 14 threads) | 6.4 ms | 36 ms | 2.2 ms | 6.9 ms |

Masking the robot (`robot_mask`) takes 0.5 ms on the M4 Pro.
