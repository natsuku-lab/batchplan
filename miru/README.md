# miru

miru says what depth cameras show as objects: which pixels are one thing, the shape of each,
what it stands on, and which object is which from one look to the next. It is built on
[batchplan](https://github.com/natsuku-lab/batchplan)'s types and runs on a batchplan `Device`'s
threads.

miru is optional: batchplan plans without it and does not depend on it.

## What it adds

- Objects from depth: `look` says which pixels are one thing, the shape of each, what it is on.
- One memory across looks: `observe` carries a `Memory`; `Memory::after` tells it each act.
- The world to plan each step against: `Memory::world`, every obstacle named.
- Grips of what was seen: `Observed::grip_worlds`, `Kept::turn_grips`, `Kept::extractions`.
- A mask model's masks, through one seam and after geometry: `View::masks`.

## What stays batchplan's

- Cleaning a camera's depth (`DepthImage::filtered`) and masking the robot (`robot_mask`).
- Leaving what is held as a shape out of the map (`RobotMask::with_shapes`, `with_band`).
- The map of what the cameras saw and its distance grids (`OccupancyMap`).
- Every plan: IK, paths, picks, places, turns and pulls.

`observe` runs the depth, mask and map steps for you; the planning calls are yours, or
`GripWorlds::plan`'s for a grip on a measured part.

## Adding it

```toml
[dependencies]
batchplan = "0.13"
miru = { package = "batchplan-miru", version = "0.13" }
```

The package is `batchplan-miru` because the name `miru` is taken on crates.io; the library is
`miru`, so code writes `use miru::..`.

One look, from a camera's depth to the objects and the world the next motion is planned against:

```rust,no_run
use std::collections::HashMap;
use std::f32::consts::PI;

use batchplan::glam::{Quat, Vec3};
use batchplan::{DepthImage, Device, Intrinsics, Obstacle, Pose, Robot, RobotOptions};
use miru::{Camera, Memory, MemoryOptions, Noise, Stance, Step};

fn main() -> Result<(), batchplan::Error> {
    let robot = Robot::load("assets/franka/franka_panda.urdf", &RobotOptions::default())?;
    let device = Device::cpu(&robot)?;
    // What the cell holds without looking: a table.
    let table = Obstacle::cuboid(Vec3::new(0.5, 0.0, -0.02), Vec3::new(0.6, 0.6, 0.02));
    let memory = Memory::try_new(&device, vec![table], &MemoryOptions::default())?;
    // One frame of a camera 1 m over the table, looking down: meters along its z axis.
    let depth = vec![1.0; 640 * 480];
    let camera = Camera {
        image: DepthImage {
            depth: &depth,
            width: 640,
            intrinsics: Intrinsics { fx: 600.0, fy: 600.0, cx: 319.5, cy: 239.5 },
            camera: Pose::new(Vec3::new(0.5, 0.0, 1.0), Quat::from_rotation_x(PI)),
        },
        color: &[],
        noise: Noise::stereo(600.0, 0.05, 0.08), // from the camera's datasheet
        band: 1,       // pixels it reads beside a jump in depth, toward the other surface
        off_by: 0.005, // how far its pose and the fixtures may be off, in meters
        masks: &[],    // no mask model
    };
    // Where the robot stood when the frame was taken.
    let (q, locked) = (robot.default_q().to_vec(), HashMap::new());
    let stance = Stance { q: &q, locked: &locked, padding: 0.01 };
    let seen = miru::observe(&device, &[camera], &stance, &memory)?;
    for object in seen.memory.objects() {
        println!("object {} on {:?}: {:?}", object.id, object.on, object.shape);
    }
    // The table, then each object, then what the map holds that no shape explains.
    let planned = seen.memory.world(&device, &Step::default())?;
    let _worlds = device.upload(&[planned.world])?; // plan against these as any batchplan worlds
    Ok(())
}
```

Pass `seen.memory` to the next look, or `seen.memory.after(&acts)?` once the robot has acted.

## More

- The API: [docs.rs/batchplan-miru][api].
- Grips on part of a thing, turns of a control, lifts out of a container: [grips].
- A mask model's masks: [masks].
- What it does not do: [limits].
- What is measured, and where it stands: [status].
- Working on it: [AGENTS.md][agents].

[api]: https://docs.rs/batchplan-miru
[grips]: https://github.com/natsuku-lab/batchplan/blob/main/miru/docs/grips.md
[masks]: https://github.com/natsuku-lab/batchplan/blob/main/miru/docs/masks.md
[limits]: https://github.com/natsuku-lab/batchplan/blob/main/miru/docs/limits.md
[status]: https://github.com/natsuku-lab/batchplan/blob/main/miru/docs/status.md
[agents]: https://github.com/natsuku-lab/batchplan/blob/main/miru/AGENTS.md
