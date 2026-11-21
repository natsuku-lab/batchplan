//! Scenes of boxes, cans and bowls, and what a camera reads of them: depth, colour, and which
//! thing each pixel shows, by sphere tracing the things' distance functions. The depth is exact,
//! or as a camera whose depth is off by a stated noise reads it ([`Frame::noisy`]), or as a
//! stereo camera reads it by its datasheet ([`Frame::sensed`], [`D435`], [`D405`]).

use std::sync::LazyLock;

use batchplan::{
    CollisionModel, DepthImage, Device, Intrinsics, Obstacle, Pose, Robot, RobotOptions, SdfGrid,
};
use glam::{Mat3, Quat, Vec2, Vec3};
use miru::{Noise, Shape, View};

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

/// Where the tests' camera stands: before the table, looking down on it at about 40 degrees.
pub const EYE: Vec3 = Vec3::new(1.1, 0.0, 0.5);

/// The middle of the table, where the tests' bowls stand.
pub const MIDDLE: Vec3 = Vec3::new(0.5, 0.0, 0.0);

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

/// The box of the tests' first scene.
pub fn a_box() -> Thing {
    block(
        Vec3::new(0.5, 0.0, 0.03),
        Vec3::new(0.04, 0.06, 0.03),
        [200, 30, 30],
    )
}

/// A bowl with a wall 4 mm through: round about one axis, its rim too thin to show.
pub fn standing_bowl(foot: Vec3) -> Thing {
    bowl(foot, 0.03, 0.056, 0.05, 0.004, [40, 40, 40])
}

/// A tray standing in the middle of the table, 24 cm square and 6 cm tall, with walls and a
/// floor 1 cm through.
pub fn thin_tray() -> Vec<Thing> {
    let board = |at: Vec3, half: Vec3| block(at, half, [150, 110, 60]);
    let (wall, up) = (0.115, 0.03);
    vec![
        board(Vec3::new(0.5, 0.0, 0.005), Vec3::new(0.11, 0.11, 0.005)),
        board(Vec3::new(0.5 - wall, 0.0, up), Vec3::new(0.005, 0.12, up)),
        board(Vec3::new(0.5 + wall, 0.0, up), Vec3::new(0.005, 0.12, up)),
        board(Vec3::new(0.5, -wall, up), Vec3::new(0.12, 0.005, up)),
        board(Vec3::new(0.5, wall, up), Vec3::new(0.12, 0.005, up)),
    ]
}

/// The thin tray with a block set down behind its near wall at `x`, on its floor.
pub fn tray_with_block(x: f32) -> Vec<Thing> {
    let mut things = thin_tray();
    things.push(block(
        Vec3::new(x, 0.0, 0.025),
        Vec3::new(0.02, 0.02, 0.015),
        [30, 30, 200],
    ));
    things
}

/// The half sizes of a box seen from above (the lesser first) or twice a cylinder's radius, and
/// the shape's middle and half height.
pub fn sizes(shape: &Shape) -> ([f32; 2], Vec3, f32) {
    match *shape {
        Shape::Cuboid {
            center,
            half_extents: half,
            ..
        }
        | Shape::OpenBox {
            center,
            half_extents: half,
            ..
        } => ([half.x.min(half.y), half.x.max(half.y)], center, half.z),
        Shape::Cylinder {
            center,
            radius,
            half_height,
            ..
        } => ([radius, radius], center, half_height),
        _ => panic!("a shape these tests do not know: {shape:?}"),
    }
}

/// A camera at `eye` looking at `target`, its image's x axis level.
pub fn looking(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
}

/// A stereo depth camera, by its datasheet: what [`Frame::sensed`] reads a scene as. Its values
/// are doubles, as a datasheet's decimals are, and each step of the reading takes them at the
/// precision it computes in.
#[derive(Clone, Copy, Debug)]
pub struct Stereo {
    /// Meters between its imagers, the second to the right of the one depth is given in.
    pub baseline: f64,
    /// Meters a count of its 16-bit depth stands for.
    pub unit: f64,
    /// The RMS error of the disparity it matches (pixels): Intel puts a well-calibrated D400's
    /// under 0.1, and 0.08 in its own examples.
    pub subpixel: f64,
    /// How many pixels of disparity it searches: 126 for the D400 series.
    pub disparities: f64,
    /// The standard deviation of where in the image a pixel reads (pixels): 0.5 in Handa et al.
    pub jitter: f64,
}

/// An Intel RealSense D435 (or D435i), by its datasheet.
pub const D435: Stereo = Stereo {
    baseline: 0.050,
    unit: 0.001,
    subpixel: 0.08,
    disparities: 126.0,
    jitter: 0.5,
};

/// An Intel RealSense D405: the D435's matcher, 18 mm between its imagers and 0.1 mm units.
pub const D405: Stereo = Stereo {
    baseline: 0.018,
    unit: 0.0001,
    ..D435
};

impl Stereo {
    /// The standard deviation of its depth with the focal length `fx` (pixels), as a view states
    /// it: growing with the square of the distance alone.
    pub fn noise(&self, fx: f32) -> Noise {
        Noise::stereo(fx, self.baseline as f32, self.subpixel as f32)
    }

    /// How near it reads with the focal length `fx` (meters): nearer than this a point's
    /// disparity is more than it searches.
    pub fn least(&self, fx: f32) -> f32 {
        (f64::from(fx) * self.baseline / self.disparities) as f32
    }
}

/// Draws from the standard normal distribution, the same for the same seed every time: SplitMix64,
/// and two of its draws to a normal one (Box and Muller).
struct Normals(u64);

impl Normals {
    /// A draw from the uniform distribution on [0, 1).
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mixed = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((mixed ^ (mixed >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A draw from the standard normal distribution.
    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * (1.0 - self.uniform()).ln()).sqrt();
        radius * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

/// What a camera reads of a scene.
#[derive(Clone)]
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
        let mut normals = Normals(seed);
        for depth in self.depth.iter_mut().filter(|depth| **depth > 0.0) {
            let normal = normals.normal();
            let deviation = noise.constant + (noise.linear + noise.quadratic * *depth) * *depth;
            *depth += deviation * normal as f32;
        }
        self.noise = noise;
        self
    }

    /// The frame as the stereo camera `camera` reads it where the renderer gives the truth, and
    /// states as much ([`Stereo::noise`]), the draws from a generator seeded with `seed`, so the
    /// same every time. Only the depth is the camera's: the colour, the pinhole and which thing
    /// each pixel shows stay the renderer's. Each part of the model is the camera maker's or the
    /// depth simulation literature's, with the datasheet's values and nothing fitted to a scene:
    ///
    /// - **Noise in disparity.** A point's disparity is `fx * baseline / z` pixels, read with a
    ///   normal error of `subpixel` pixels, so depth's error grows with the square of the
    ///   distance: `z^2 * subpixel / (fx * baseline)`, the depth RMS error of Intel's tuning
    ///   guide [1].
    /// - **Shadows.** A point reads only where both imagers see it. The second sits `baseline` to
    ///   the right of the one depth is given in: a point shows there `disparity` columns to the
    ///   left, over the column it lands in and the next, and has no reading where another point
    ///   lands on its column a pixel of disparity nearer or more, nor past the image's left edge.
    ///   So the background has none in a band left of every near edge, as wide as the near
    ///   thing or as its disparity is more, whichever is less, nor has a surface turned from that
    ///   imager.
    /// - **The least distance.** The camera searches `disparities` pixels, so nearer than
    ///   `fx * baseline / disparities` it reads nothing [1] ([`Stereo::least`]).
    /// - **Jitter.** Each pixel reads the scene `jitter` pixels off, a normal draw along each
    ///   axis, its disparity interpolated between the four pixels around where it reads [2]:
    ///   across an edge, a reading between the near and the far surface.
    /// - **Depth units.** A reading is a whole count of `unit`, and none past 65,535 counts.
    ///
    /// Left out: no reading on dark, shiny or clear surfaces.
    ///
    /// [1] Intel, "Best-Known-Methods for Tuning Intel RealSense D400 Depth Cameras for Best
    ///     Performance" (depth RMS error; minimum Z).
    /// [2] A. Handa, T. Whelan, J. McDonald and A. J. Davison, "A Benchmark for RGB-D Visual
    ///     Odometry, 3D Reconstruction and SLAM", ICRA 2014.
    pub fn sensed(mut self, camera: Stereo, seed: u64) -> Self {
        let (width, pixels) = (self.width, self.depth.len());
        let height = pixels / width;
        let scale = (f64::from(self.intrinsics.fx) * camera.baseline) as f32;
        let disparity: Vec<f32> = (self.depth.iter())
            .map(|&z| if z > 0.0 { scale / z } else { 0.0 })
            .collect();

        // Where the second imager sees: per row, the most disparity that lands on each of its
        // columns, and one column past the image.
        let in_second = |i: usize| (i % width) as f32 - disparity[i];
        let row = |i: usize| i / width * (width + 1);
        let mut nearest = vec![0.0_f32; height * (width + 1)];
        for (i, &point) in disparity.iter().enumerate() {
            let column = in_second(i).floor();
            if self.depth[i] > 0.0 && column >= 0.0 {
                let at = row(i) + column as usize;
                for most in &mut nearest[at..at + 2] {
                    *most = most.max(point);
                }
            }
        }
        let both: Vec<bool> = (0..pixels)
            .map(|i| {
                let lands = in_second(i).round_ties_even().clamp(0.0, width as f32) as usize;
                self.depth[i] > 0.0
                    && in_second(i) > -0.5
                    && disparity[i] > nearest[row(i) + lands] - 1.0
            })
            .collect();

        // Each pixel's disparity read `jitter` off, between its four neighbours there: per
        // pixel, the index along an axis it reads from, and how far toward the next.
        let mut normals = Normals(seed);
        let mut off = |size: usize, along: &dyn Fn(usize) -> usize| -> Vec<(usize, f64)> {
            let last = ((size - 1) as f32).next_down().max(0.0);
            (0..pixels)
                .map(|i| {
                    let at = along(i) as f32 + camera.jitter as f32 * normals.normal() as f32;
                    let at = at.clamp(0.0, last);
                    let whole = at as usize;
                    (whole, f64::from(at) - whole as f64)
                })
                .collect()
        };
        let columns = off(width, &|i| i % width);
        let rows = off(height, &|i| i / width);
        let disparity = |i: usize| f64::from(disparity[i]);
        let read: Vec<f64> = (0..pixels)
            .map(|i| {
                let ((left, across), (top, down)) = (columns[i], rows[i]);
                let corner = top * width + left;
                let upper = disparity(corner) * (1.0 - across) + disparity(corner + 1) * across;
                let lower = disparity(corner + width) * (1.0 - across)
                    + disparity(corner + width + 1) * across;
                let read = upper * (1.0 - down) + lower * down;
                read + f64::from(camera.subpixel as f32 * normals.normal() as f32)
            })
            .collect();

        // A whole count of the unit, where it reads at all.
        let (unit, counts) = (camera.unit as f32, scale / camera.unit as f32);
        for (i, depth) in self.depth.iter_mut().enumerate() {
            let reads = both[i] && read[i] > 0.0 && read[i] <= camera.disparities;
            let count = if reads {
                ((f64::from(counts) / read[i]) as f32).round_ties_even()
            } else {
                0.0
            };
            *depth = if count > f32::from(u16::MAX) {
                0.0
            } else {
                count * unit
            };
        }
        self.noise = camera.noise(self.intrinsics.fx);
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
