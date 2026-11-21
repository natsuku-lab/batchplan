//! One view's readings as points, sorted into what is already known (nothing, the robot) and what
//! is left to tell apart: the readings on a known shape, and those on none.

use batchplan::{Intrinsics, Obstacle, Pose};
use glam::Vec3;

use crate::look::{LookOptions, Noise, View};
use crate::threads;

/// How many standard deviations of the cameras' noise a reading may lie from a surface it is on.
pub(crate) const SIGMAS: f32 = 3.0;

/// What a reading is before objects are told apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// No reading, or one outside the look's corners.
    Nothing,
    /// The robot or what it holds.
    Robot,
    /// On a known shape, by its index, or in it: no farther over its surface than a reading on it
    /// may read. The shape's, or a thing's that lies on it: telling the readings apart says.
    Known(usize),
    /// On no known shape.
    Open,
}

/// One view's readings.
pub(crate) struct Seen {
    pub(crate) width: usize,
    pub(crate) height: usize,
    intrinsics: Intrinsics,
    noise: Noise,
    /// The camera, its rotation unit.
    camera: Pose,
    /// Per pixel, where it reads in the look's frame; read only where `inverse` is not 0.
    pub(crate) points: Vec<Vec3>,
    /// Per pixel, one over its depth along the camera's z axis, in which a flat surface is a
    /// straight line from pixel to pixel however it leans; 0 without a reading.
    pub(crate) inverse: Vec<f32>,
    pub(crate) kinds: Vec<Kind>,
    /// Per pixel whose reading is on a known shape, how far from the shape's surface it lies
    /// (meters): negative inside.
    pub(crate) off: Vec<f32>,
}

impl Seen {
    /// What `view` reads: its sizes, intrinsics and pose have been checked. Each reading is
    /// sorted by itself, so the image is read in bands on up to `threads` threads.
    pub(crate) fn of(view: &View<'_>, known: &[Obstacle], o: &LookOptions, threads: usize) -> Self {
        let image = &view.image;
        let Intrinsics { fx, fy, cx, cy } = image.intrinsics;
        let camera = Pose::new(image.camera.position, image.camera.rotation.normalize());
        let width = image.width;
        let pixels = image.depth.len();
        // Where pixel `i` reads, one over its depth, what it is, and how far off a known shape.
        let read = |i: usize| {
            let z = image.depth[i];
            // A depth too small or too large for its inverse and its noise to be numbers is no
            // reading either.
            let sigma = view.noise.at(z);
            let spread = sigma / (z * z);
            if !(z.is_finite() && z > 0.0 && (1.0 / z).is_finite() && spread.is_finite()) {
                return (Vec3::ZERO, 0.0, Kind::Nothing, 0.0);
            }
            // Integer pixel coordinates are pixel centres, as in OpenCV and batchplan.
            let (u, v) = ((i % width) as f32, (i / width) as f32);
            let point = camera.rotation * Vec3::new((u - cx) / fx * z, (v - cy) / fy * z, z)
                + camera.position;
            let (kind, off) = if view.robot.get(i).copied().unwrap_or(false) {
                (Kind::Robot, 0.0)
            } else if !(point.cmpge(o.lower_corner).all() && point.cmple(o.upper_corner).all()) {
                (Kind::Nothing, 0.0)
            } else {
                let distances = known.iter().map(|shape| shape.distance(point).value);
                let nearest = distances.enumerate().min_by(|a, b| a.1.total_cmp(&b.1));
                match nearest {
                    Some((index, distance)) if distance <= SIGMAS * sigma => {
                        (Kind::Known(index), distance)
                    }
                    _ => (Kind::Open, 0.0),
                }
            };
            (point, 1.0 / z, kind, off)
        };
        let band = pixels.div_ceil(threads).max(1);
        let bands = threads::each(threads, pixels.div_ceil(band), |number| {
            let readings = (number * band..pixels.min((number + 1) * band)).map(read);
            readings.collect::<Vec<_>>()
        });
        let mut seen = Seen {
            width,
            height: pixels / width,
            intrinsics: image.intrinsics,
            noise: view.noise,
            camera,
            points: Vec::with_capacity(pixels),
            inverse: Vec::with_capacity(pixels),
            kinds: Vec::with_capacity(pixels),
            off: Vec::with_capacity(pixels),
        };
        for (point, inverse, kind, off) in bands.into_iter().flatten() {
            seen.points.push(point);
            seen.inverse.push(inverse);
            seen.kinds.push(kind);
            seen.off.push(off);
        }
        seen
    }

    /// Whether pixel `i` has a reading to tell apart: one inside the look's corners that is not
    /// the robot's.
    pub(crate) fn reads(&self, i: usize) -> bool {
        matches!(self.kinds[i], Kind::Open | Kind::Known(_))
    }

    /// The standard deviation of pixel `i`'s reading (meters), as its camera's noise has it at
    /// the depth it reads.
    pub(crate) fn sigma(&self, i: usize) -> f32 {
        self.noise.at(1.0 / self.inverse[i])
    }

    /// That deviation in inverse depth, the scale readings are compared at: the same at every
    /// depth for a camera whose noise grows with the square of the depth.
    pub(crate) fn spread(&self, i: usize) -> f32 {
        self.sigma(i) * self.inverse[i] * self.inverse[i]
    }

    /// How far apart neighboring pixels' readings lie at pixel `i`'s depth, on a surface that
    /// faces the camera (meters).
    pub(crate) fn spacing(&self, i: usize) -> f32 {
        1.0 / (self.inverse[i] * self.intrinsics.fx.max(self.intrinsics.fy))
    }

    /// Where the camera stands.
    pub(crate) fn eye(&self) -> Vec3 {
        self.camera.position
    }

    /// The angle between the rays of neighboring pixels at the image's middle (radians).
    pub(crate) fn pixel(&self) -> f32 {
        1.0 / self.intrinsics.fx.max(self.intrinsics.fy)
    }

    /// The unit normal, toward the camera, of the surface whose inverse depth is `c` at pixel
    /// (`x`, `y`) and grows by `a` a column and `b` a row.
    pub(crate) fn facing(&self, pixel: (usize, usize), plane: [f32; 3]) -> Vec3 {
        -self.away(pixel, plane).normalize_or_zero()
    }

    /// That surface as the plane `n . (p - eye) = 1` in the look's frame: its `n`, which points
    /// away from the camera and is linear in `a`, `b` and `c`.
    pub(crate) fn away(&self, (x, y): (usize, usize), [a, b, c]: [f32; 3]) -> Vec3 {
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        self.camera.rotation
            * Vec3::new(
                a * fx,
                b * fy,
                c + a * (cx - x as f32) + b * (cy - y as f32),
            )
    }

    /// Where the camera looks through the image at column `u` and row `v`, in the look's frame:
    /// the vector from the eye to the point there at a depth of 1.
    pub(crate) fn ray(&self, (u, v): (f32, f32)) -> Vec3 {
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        self.camera.rotation * Vec3::new((u - cx) / fx, (v - cy) / fy, 1.0)
    }

    /// The pixel next to `i` by `by` columns and rows, if the image has it.
    pub(crate) fn step(&self, i: usize, by: (isize, isize)) -> Option<usize> {
        let (x, y) = (
            (i % self.width) as isize + by.0,
            (i / self.width) as isize + by.1,
        );
        (x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height)
            .then(|| y as usize * self.width + x as usize)
    }

    /// The pixel `p` projects to; `None` behind or beside the camera.
    pub(crate) fn project(&self, p: Vec3) -> Option<usize> {
        let Intrinsics { fx, fy, cx, cy } = self.intrinsics;
        let c = self.camera.rotation.inverse() * (p - self.camera.position);
        if c.z <= 0.0 {
            return None;
        }
        let (u, v) = (
            (fx * c.x / c.z + cx + 0.5).floor(),
            (fy * c.y / c.z + cy + 0.5).floor(),
        );
        (u >= 0.0 && v >= 0.0 && u < self.width as f32 && v < self.height as f32)
            .then(|| v as usize * self.width + u as usize)
    }
}
