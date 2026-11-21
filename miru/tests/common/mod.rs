//! Scenes of boxes, cans and bowls, and what a camera reads of them: depth, colour, and which
//! thing each pixel shows, by sphere tracing the things' distance functions.

use std::sync::LazyLock;

use batchplan::{
    CollisionModel, DepthImage, Device, Intrinsics, Obstacle, Pose, Robot, RobotOptions, SdfGrid,
};
use glam::{Mat3, Quat, Vec2, Vec3};
use miru::{Noise, View};

pub const WIDTH: usize = 320;
pub const HEIGHT: usize = 240;
pub const K: Intrinsics = Intrinsics {
    fx: 300.0,
    fy: 300.0,
    cx: 159.5,
    cy: 119.5,
};
/// The noise the tests' cameras state: planes over windows of five pixels at their distances.
pub const NOISE: Noise = Noise {
    constant: 0.0,
    linear: 0.0,
    quadratic: 0.0005,
};

/// A device of `threads` threads, or of as many as the machine has, for looks to run on. A look
/// reads nothing of a device's robot, so this one's is a single joint, given an empty collision
/// model: fitting one would start rayon's global pool, which no look may run on.
pub fn device_of(threads: Option<usize>) -> Device {
    let urdf = r#"<robot name="none"><link name="base"/><link name="tip"/>
        <joint name="joint" type="revolute"><parent link="base"/><child link="tip"/>
        <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint></robot>"#;
    let options = RobotOptions {
        collision_model: Some(CollisionModel::default()),
        ..Default::default()
    };
    let robot = Robot::from_urdf(urdf, &options).expect("a robot of one joint");
    let device = match threads {
        Some(threads) => Device::cpu_threads(&robot, threads),
        None => Device::cpu(&robot),
    };
    device.expect("a CPU device")
}

/// The device the tests look on.
pub fn device() -> &'static Device {
    static DEVICE: LazyLock<Device> = LazyLock::new(|| device_of(None));
    &DEVICE
}

/// One thing of a scene: the signed distance to its surface, its colour, and the obstacle it is
/// to a look that knows it, where it is a box or a can.
pub struct Thing {
    pub distance: Box<dyn Fn(Vec3) -> f32 + Sync>,
    pub color: [u8; 3],
    pub solid: Option<Obstacle>,
}

/// `thing` as a shape a look knows.
pub fn known(thing: &Thing) -> Obstacle {
    thing.solid.clone().expect("a box or a can")
}

/// `thing` as a shape a look knows that is neither a box nor a can: its distances at the points
/// of a grid `voxel` apart between `lower` and `upper`, which must hold it and what reads on it.
pub fn known_as_grid(thing: &Thing, lower: Vec3, upper: Vec3, voxel: f32) -> Obstacle {
    let dims = ((upper - lower) / voxel).ceil().as_uvec3() + 1;
    let mut distances = Vec::with_capacity((dims.x * dims.y * dims.z) as usize);
    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let p = lower + Vec3::new(x as f32, y as f32, z as f32) * voxel;
                distances.push((thing.distance)(p));
            }
        }
    }
    let grid = SdfGrid::try_new(dims.to_array(), voxel, lower, &distances).expect("a sound grid");
    Obstacle::Sdf {
        grid: grid.into(),
        center: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    }
}

/// A box with its sides along the axes.
pub fn block(center: Vec3, half: Vec3, color: [u8; 3]) -> Thing {
    let distance = move |p: Vec3| {
        let past = (p - center).abs() - half;
        past.max(Vec3::ZERO).length() + past.max_element().min(0.0)
    };
    Thing {
        distance: Box::new(distance),
        color,
        solid: Some(Obstacle::cuboid(center, half)),
    }
}

/// An upright can.
pub fn can(center: Vec3, radius: f32, half_height: f32, color: [u8; 3]) -> Thing {
    let distance = move |p: Vec3| {
        let own = p - center;
        let past = Vec2::new(own.truncate().length() - radius, own.z.abs() - half_height);
        past.max(Vec2::ZERO).length() + past.max_element().min(0.0)
    };
    Thing {
        distance: Box::new(distance),
        color,
        solid: Some(Obstacle::cylinder(center, radius, half_height)),
    }
}

/// What is left of `thing` with `hollow` cut out of it.
pub fn without(thing: Thing, hollow: Thing) -> Thing {
    Thing {
        distance: Box::new(move |p: Vec3| (thing.distance)(p).max(-(hollow.distance)(p))),
        color: thing.color,
        solid: None,
    }
}

/// An upright ring about `center`: a wall between the radii `inner` and `outer`, `half_height`
/// up and down.
pub fn ring(center: Vec3, inner: f32, outer: f32, half_height: f32, color: [u8; 3]) -> Thing {
    let (middle, half) = (0.5 * (inner + outer), 0.5 * (outer - inner));
    let distance = move |p: Vec3| {
        let own = p - center;
        let across = (own.truncate().length() - middle).abs() - half;
        let past = Vec2::new(across, own.z.abs() - half_height);
        past.max(Vec2::ZERO).length() + past.max_element().min(0.0)
    };
    Thing {
        distance: Box::new(distance),
        color,
        solid: None,
    }
}

/// A bowl standing at `foot`, the middle of its floor's underside: a floor `floor` in radius and
/// a wall that widens to `rim` at `height`, both `thick` through, their edges round.
pub fn bowl(foot: Vec3, floor: f32, rim: f32, height: f32, thick: f32, color: [u8; 3]) -> Thing {
    // Its section from the axis outward, down the middle of its floor and of its wall.
    let half = thick / 2.0;
    let (corner, lip) = (Vec2::new(floor, half), Vec2::new(rim, height - half));
    let to_segment = |q: Vec2, a: Vec2, b: Vec2| {
        let along = ((q - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
        q.distance(a + (b - a) * along)
    };
    let distance = move |p: Vec3| {
        let own = p - foot;
        let q = Vec2::new(own.truncate().length(), own.z);
        to_segment(q, Vec2::new(0.0, half), corner).min(to_segment(q, corner, lip)) - half
    };
    Thing {
        distance: Box::new(distance),
        color,
        solid: None,
    }
}

/// A bowl that is part of a ball's shell, standing at `foot`: the shell is `radius` to its
/// middle and `thick` through, and the bowl is what of it lies under `height`.
pub fn cup(foot: Vec3, radius: f32, height: f32, thick: f32, color: [u8; 3]) -> Thing {
    let center = foot + Vec3::Z * (radius + 0.5 * thick);
    let distance = move |p: Vec3| {
        let shell = (p.distance(center) - radius).abs() - 0.5 * thick;
        shell.max(p.z - foot.z - height)
    };
    Thing {
        distance: Box::new(distance),
        color,
        solid: None,
    }
}

/// A ball.
pub fn ball(center: Vec3, radius: f32, color: [u8; 3]) -> Thing {
    Thing {
        distance: Box::new(move |p: Vec3| p.distance(center) - radius),
        color,
        solid: None,
    }
}

/// `thing` turned by `rotation` about the point `about`.
pub fn turned(thing: Thing, about: Vec3, rotation: Quat) -> Thing {
    let Thing {
        distance, color, ..
    } = thing;
    Thing {
        distance: Box::new(move |p: Vec3| distance(about + rotation.inverse() * (p - about))),
        color,
        solid: None,
    }
}

/// A table whose top is the plane z = 0, 1.2 m square around (0.5, 0).
pub fn table() -> Thing {
    block(
        Vec3::new(0.5, 0.0, -0.02),
        Vec3::new(0.6, 0.6, 0.02),
        [120, 100, 80],
    )
}

/// A camera at `eye` looking at `target`, its image's x axis level.
pub fn looking(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
}

/// What a camera reads of a scene.
pub struct Frame {
    pub camera: Pose,
    /// Pixels to a row.
    pub width: usize,
    pub intrinsics: Intrinsics,
    pub depth: Vec<f32>,
    pub color: Vec<u8>,
    /// Per pixel, the thing it shows, by its index in the scene.
    pub shows: Vec<Option<usize>>,
    /// Per pixel, whether its reading is the robot's.
    pub robot: Vec<bool>,
    /// Per pixel, the number of the thing a mask model says it shows; empty without a model.
    pub masks: Vec<u32>,
    /// The noise the camera states.
    pub noise: Noise,
}

impl Frame {
    /// The frame the tests' camera reads of `things` from `camera`.
    pub fn of(things: &[Thing], camera: Pose) -> Self {
        Self::through(things, camera, K, (WIDTH, HEIGHT))
    }

    /// The frame a camera at `camera` reads of `things`, through the pinhole `k` onto an image
    /// of `size` columns and rows.
    pub fn through(things: &[Thing], camera: Pose, k: Intrinsics, size: (usize, usize)) -> Self {
        let (width, pixels) = (size.0, size.0 * size.1);
        let mut frame = Frame {
            camera,
            width,
            intrinsics: k,
            depth: vec![0.0; pixels],
            color: vec![0; 3 * pixels],
            shows: vec![None; pixels],
            robot: vec![],
            masks: vec![],
            noise: NOISE,
        };
        for i in 0..pixels {
            let (u, v) = ((i % width) as f32, (i / width) as f32);
            let ray = Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
            let direction = camera.rotation * ray.normalize();
            let mut along = 0.0;
            while along < 3.0 {
                let p = camera.position + direction * along;
                let nearest = things.iter().map(|thing| (thing.distance)(p)).enumerate();
                let (thing, distance) = nearest
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("a scene has a thing");
                if distance < 1e-5 {
                    frame.depth[i] = along * ray.normalize().z;
                    frame.color[3 * i..3 * i + 3].copy_from_slice(&things[thing].color);
                    frame.shows[i] = Some(thing);
                    break;
                }
                along += distance;
            }
        }
        frame
    }

    /// The frame with the masks of a model that sees what the renderer does: each pixel that
    /// shows a thing has the number `number` gives that thing, and the rest 0.
    pub fn masked(mut self, number: impl Fn(usize) -> u32) -> Self {
        let shown = self.shows.iter();
        self.masks = shown.map(|shows| shows.map_or(0, &number)).collect();
        self
    }

    /// The frame as a camera whose depth is off by `noise` reads it, and states as much: each
    /// reading moved by a draw from a normal distribution of that deviation at its depth, the
    /// draws from a generator seeded with `seed`, so the same every time.
    pub fn noisy(mut self, noise: Noise, seed: u64) -> Self {
        // SplitMix64, and two of its draws to a normal one (Box and Muller).
        let mut state = seed;
        let mut draw = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mixed = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            let mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((mixed ^ (mixed >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        for depth in self.depth.iter_mut().filter(|depth| **depth > 0.0) {
            let normal =
                (-2.0 * (1.0 - draw()).ln()).sqrt() * (std::f64::consts::TAU * draw()).cos();
            let deviation = noise.constant + (noise.linear + noise.quadratic * *depth) * *depth;
            *depth += deviation * normal as f32;
        }
        self.noise = noise;
        self
    }

    /// The frame as a view.
    pub fn view(&self) -> View<'_> {
        View {
            image: DepthImage {
                depth: &self.depth,
                width: self.width,
                intrinsics: self.intrinsics,
                camera: self.camera,
            },
            color: &self.color,
            noise: self.noise,
            robot: &self.robot,
            masks: &self.masks,
        }
    }
}
