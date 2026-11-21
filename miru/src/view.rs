//! One view's readings as points, sorted into what is already known (nothing, the robot) and what
//! is left to tell apart: the readings on a known shape, and those on none.

use batchplan::{Device, Intrinsics, Obstacle, Pose, sdf};
use glam::Vec3;

use crate::error::Result;
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
    /// sorted by itself, so the image is read in bands on up to `threads` threads, and the
    /// readings are held against the known shapes on `device`'s ([`sdf::on_shapes`]).
    pub(crate) fn of(
        device: &Device,
        view: &View<'_>,
        known: &[Obstacle],
        o: &LookOptions,
        threads: usize,
    ) -> Result<Self> {
        let image = &view.image;
        let Intrinsics { fx, fy, cx, cy } = image.intrinsics;
        let camera = Pose::new(image.camera.position, image.camera.rotation.normalize());
        let width = image.width;
        let pixels = image.depth.len();
        // Where pixel `i` reads, one over its depth, what it is before the known shapes are
        // asked, and how far from one a reading there may lie and be on it: less than nothing
        // for a pixel that is asked of none.
        let read = |i: usize| {
            let z = image.depth[i];
            // A depth too small or too large for its inverse and its noise to be numbers is no
            // reading either.
            let sigma = view.noise.at(z);
            let spread = sigma / (z * z);
            if !(z.is_finite() && z > 0.0 && (1.0 / z).is_finite() && spread.is_finite()) {
                return (Vec3::ZERO, 0.0, Kind::Nothing, -1.0);
            }
            // Integer pixel coordinates are pixel centres, as in OpenCV and batchplan.
            let (u, v) = ((i % width) as f32, (i / width) as f32);
            let point = camera.rotation * Vec3::new((u - cx) / fx * z, (v - cy) / fy * z, z)
                + camera.position;
            let (kind, reach) = if view.robot.get(i).copied().unwrap_or(false) {
                (Kind::Robot, -1.0)
            } else if !(point.cmpge(o.lower_corner).all() && point.cmple(o.upper_corner).all()) {
                (Kind::Nothing, -1.0)
            } else {
                (Kind::Open, SIGMAS * sigma)
            };
            (point, 1.0 / z, kind, reach)
        };
        let mut seen = Seen {
            width,
            height: pixels / width,
            intrinsics: image.intrinsics,
            noise: view.noise,
            camera,
            points: vec![Vec3::ZERO; pixels],
            inverse: vec![0.0; pixels],
            kinds: vec![Kind::Nothing; pixels],
            off: vec![0.0; pixels],
        };
        let mut reach = vec![-1.0; pixels];
        let band = pixels.div_ceil(threads).max(1);
        let bands = (seen.points.chunks_mut(band))
            .zip(seen.inverse.chunks_mut(band))
            .zip(seen.kinds.chunks_mut(band).zip(reach.chunks_mut(band)));
        threads::each_piece(
            threads,
            bands.collect(),
            |number, ((points, inverse), (kinds, reach))| {
                for (k, i) in (number * band..).take(points.len()).enumerate() {
                    (points[k], inverse[k], kinds[k], reach[k]) = read(i);
                }
            },
        );
        // A reading no farther from the known shape nearest it than a reading on it may read is
        // on that shape.
        let on = sdf::on_shapes(device, known, &seen.points, &reach)?;
        let bands = (seen.kinds.chunks_mut(band))
            .zip(seen.off.chunks_mut(band))
            .zip(on.chunks(band));
        threads::each_piece(threads, bands.collect(), |_, ((kinds, off), on)| {
            for (k, on) in on.iter().enumerate() {
                if let Some(on) = on {
                    (kinds[k], off[k]) = (Kind::Known(on.shape), on.distance);
                }
            }
        });
        Ok(seen)
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
