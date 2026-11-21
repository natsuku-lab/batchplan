//! Round hollow things: a wall's outside and the inside it stands in front of, as one part.
//!
//! A bowl seen from before and above shows its outside and, over the near rim, its inside, with
//! a jump in depth between the two all along that rim: the rim the wall turns over is too thin
//! to show as a surface, so no convex edge says they are one thing. What does is that the two
//! are round about one axis and end in one rim:
//!
//! 1. **Candidates** ([`shells`]). A part that stands in front of another, where the other
//!    stands in front of something too: an outside's edge over an inside, and the inside's own
//!    edges, its far rim among them. Both are of readings on no known shape: what reads on one
//!    is never made one with what reads on none, and a known shape's own parts are the shape's.
//! 2. **One axis** ([`Lines::axis`], [`Shell::turns`]). The normals of a surface of revolution
//!    all meet its axis, and lines that meet one line lie in a linear line complex: one axis is
//!    fitted to the normals of both parts' planes in closed form (Pottmann, Hofer, Odehnal and
//!    Wallner, "Line geometry for 3D shape understanding and reconstruction", ECCV 2004), and
//!    kept where each part's normals miss it by no more than its planes' own errors make of
//!    them.
//! 3. **One circle** ([`Shell::one_circle`]). About that axis the two edges lie on one circle,
//!    at one radius and one height: the rim.
//! 4. **A shell** ([`Shell::faces`]). The part in front faces away from the axis and the part
//!    behind toward it, as a wall's outside and its inside do.
//!
//! The facing alone asks the planes for evidence: a part whose planes are too few or too loosely
//! told to show which way it faces, or that has none, faces neither way and stays apart. The
//! axis needs more than five windows of planes over the two parts, but whether each part turns
//! about it is a bound on how far its normals miss it, which planes told loosely pass easily
//! and a part with no plane passes; the circle is asked of the edges' readings, which need no
//! plane. The axis is held as fitted: what it is off by is not carried into the circle or the
//! facing.

use std::collections::BTreeMap;

use glam::{DMat3, DVec3};

use super::{NONE, Planes};
use crate::view::{SIGMAS, Seen};

/// The pairs of parts that are the outside and the inside of one round hollow thing. `of` says
/// each pixel's part, of `count` parts; `on_shape` whether a part is of readings on a known
/// shape, which no part of readings on none is made one with, so neither of a pair is; `width`
/// a window's width in pixels at a pixel; and `hidden` every place along a border where one
/// reading hides the other, the one in front first.
pub(super) fn shells(
    seen: &Seen,
    planes: &Planes,
    of: &[u32],
    count: usize,
    on_shape: &dyn Fn(u32) -> bool,
    width: &dyn Fn(usize) -> f32,
    hidden: &[(usize, usize)],
) -> Vec<(u32, u32)> {
    // Per part in front, of readings on no shape, and part behind (`NONE` for a reading in no
    // part), the readings in front along their border.
    let mut edges: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
    for &(front, back) in hidden {
        if of[front] != of[back] && of[front] != NONE && !on_shape(of[front]) {
            edges.entry((of[front], of[back])).or_default().push(front);
        }
    }
    let mut candidates = vec![];
    for (&(outside, inside), near) in &edges {
        if inside == NONE || on_shape(inside) {
            continue;
        }
        let behind = edges.range((inside, 0)..=(inside, u32::MAX));
        let far: Vec<usize> = behind.flat_map(|(_, edge)| edge.iter().copied()).collect();
        if !far.is_empty() {
            candidates.push((outside, inside, near.as_slice(), far));
        }
    }
    if candidates.is_empty() {
        return vec![];
    }

    // The candidates' readings that have planes, and their normals summed as lines: a pair
    // then costs one sum of two and its axis.
    let mut surfaces: Vec<Option<Surface>> = vec![None; count];
    for &(outside, inside, ..) in &candidates {
        for part in [outside, inside] {
            surfaces[part as usize].get_or_insert_with(Surface::default);
        }
    }
    for i in (0..of.len()).filter(|&i| planes.has[i] && of[i] != NONE) {
        if let Some(surface) = &mut surfaces[of[i] as usize] {
            let (at, toward) = tangent(seen, planes, i);
            surface.lines.add(at, toward, share(width(i)));
            surface.readings.push(i);
        }
    }

    let surface = |part: u32| surfaces[part as usize].as_ref().expect("a candidate's");
    let is_shell = |&(outside, inside, near, ref far): &(u32, u32, &[usize], Vec<usize>)| {
        let (outside, inside) = (surface(outside), surface(inside));
        let Some(axis) = outside.lines.with(&inside.lines).axis() else {
            return false;
        };
        let shell = Shell {
            seen,
            planes,
            width,
            axis,
        };
        // The circle first: it is asked of the edges alone, and few pairs have it.
        shell.one_circle(near, far)
            && shell.turns(&outside.readings)
            && shell.turns(&inside.readings)
            && shell.faces(&outside.readings, /* away */ true)
            && shell.faces(&inside.readings, /* away */ false)
    };
    let shells = candidates.iter().filter(|candidate| is_shell(candidate));
    shells
        .map(|&(outside, inside, ..)| (outside, inside))
        .collect()
}

/// The share one reading is of a window `width` pixels wide: the readings of one window share
/// its plane's error, so sums over planes count each window once.
fn share(width: f32) -> f64 {
    f64::from(1.0 / (width * width))
}

/// A part's readings that have planes.
#[derive(Clone, Default)]
struct Surface {
    readings: Vec<usize>,
    /// Their normals, each weighed by its share of a window.
    lines: Lines,
}

/// Two parts held against the axis fitted to them.
struct Shell<'a> {
    seen: &'a Seen,
    planes: &'a Planes,
    width: &'a dyn Fn(usize) -> f32,
    axis: Axis,
}

impl Shell<'_> {
    /// Whether the two edges `near` and `far` lie on one circle about the axis: more than half
    /// of each edge, a window's width at least.
    ///
    /// The circle is the one at the middle radius and the middle height of all their readings,
    /// so that of the longer edge where the two differ. A reading lies on it within three
    /// deviations of its depth and half a window's width: the two edges of a rim may lie that
    /// far apart, half of what a rim can be wide before it shows as a surface with planes of
    /// its own.
    fn one_circle(&self, near: &[usize], far: &[usize]) -> bool {
        let (seen, eye) = (self.seen, self.seen.eye());
        let place = |&i: &usize| {
            let (out, up) = self.axis.around((seen.points[i] - eye).as_dvec3());
            (out.length(), up)
        };
        let middle = |mut values: Vec<f64>| {
            values.sort_unstable_by(f64::total_cmp);
            values[values.len() / 2]
        };
        let places = || near.iter().chain(far).map(place);
        let (radius, height) = (
            middle(places().map(|(radius, _)| radius).collect()),
            middle(places().map(|(_, height)| height).collect()),
        );
        [near, far].into_iter().all(|edge| {
            let (mut on, mut all) = (0.0, 0.0);
            for &i in edge {
                let (around, up) = place(&i);
                let window = (self.width)(i);
                let reach = f64::from(0.5 * window * seen.spacing(i) + SIGMAS * seen.sigma(i));
                all += 1.0 / window;
                if (around - radius).abs() <= reach && (up - height).abs() <= reach {
                    on += 1.0 / window;
                }
            }
            on >= 1.0 && 2.0 * on > all
        })
    }

    /// Whether the planes of `readings`, one part's, are of a surface that turns about the
    /// axis.
    ///
    /// A normal misses the axis by its part along the circle its point turns on, times that
    /// circle's radius. The misses' squares, each window's counted once, are held against what
    /// the planes' own covariances make of them, as a plane's readings are against their noise:
    /// at three deviations of their sum. Each part is asked alone, or one with many planes
    /// would hide what the other misses by. A part without a plane misses it by nothing.
    fn turns(&self, readings: &[usize]) -> bool {
        let (mut off, mut expected, mut windows) = (0.0, 0.0, 0.0);
        for &i in readings {
            let (at, toward) = tangent(self.seen, self.planes, i);
            let (d_at, d_toward) = leans(self.seen, self.planes, i, at, toward);
            // How fast the point turns about the axis, a unit turn a unit time.
            let turn = self.axis.along.cross(self.axis.around(at).0);
            let slopes = std::array::from_fn(|k| {
                d_toward[k].dot(turn) + toward.dot(self.axis.along.cross(d_at[k]))
            });
            let share = share((self.width)(i));
            off += share * toward.dot(turn).powi(2);
            expected += share * spread(self.planes.covariances[i], slopes);
            windows += share;
        }
        // Asked this way round, so that sums that are not numbers give no shell.
        off * windows <= expected * (windows + f64::from(SIGMAS) * (2.0 * windows).sqrt())
    }

    /// Whether the surface with the planes of `readings` faces away from the axis, or toward
    /// it: whether more than half of them do, each where its normal points out from the axis,
    /// or in to it, by more than three deviations of what its plane's covariance makes of
    /// that. A part without a plane faces neither way.
    fn faces(&self, readings: &[usize], away: bool) -> bool {
        let along = self.axis.along;
        let (mut facing, mut all) = (0.0, 0.0);
        for &i in readings {
            let (at, toward) = tangent(self.seen, self.planes, i);
            let (d_at, d_toward) = leans(self.seen, self.planes, i, at, toward);
            let (out, _) = self.axis.around(at);
            let (out, radius) = (out.normalize(), out.length());
            let slopes = std::array::from_fn(|k| {
                // The way out turns as the point moves across it.
                let across = d_at[k] - along * along.dot(d_at[k]);
                d_toward[k].dot(out) + toward.dot(across - out * out.dot(across)) / radius
            });
            let outward = if away {
                toward.dot(out)
            } else {
                -toward.dot(out)
            };
            let share = share((self.width)(i));
            all += share;
            if outward > f64::from(SIGMAS) * spread(self.planes.covariances[i], slopes).sqrt() {
                facing += share;
            }
        }
        2.0 * facing > all
    }
}

/// `g' S g` for `S` the covariance of a plane's `[c, a, b]`, its upper triangle row by row: the
/// variance of what changes by `g` with them.
fn spread(covariance: [f32; 6], [c, a, b]: [f64; 3]) -> f64 {
    let [cc, ca, cb, aa, ab, bb] = covariance.map(f64::from);
    cc * c * c + aa * a * a + bb * b * b + 2.0 * (ca * c * a + cb * c * b + ab * a * b)
}

/// Reading `i`'s plane where it is told best, the middle of the readings its window held: the
/// point of it there, from the eye, and its unit normal toward the camera.
///
/// On a curved surface a window's plane leans as the surface does at that middle, which is not
/// the reading's own pixel where part of the window had no readings, as beside an outline.
fn tangent(seen: &Seen, planes: &Planes, i: usize) -> (DVec3, DVec3) {
    let [c, a, b] = planes.fits[i];
    let pixel = (i % planes.width, i / planes.width);
    let away = seen.away(pixel, [a, b, c]).as_dvec3();
    // The middle is where the plane's height and its lean are told apart from one another.
    let [_, ca, cb, aa, ab, bb] = planes.covariances[i].map(f64::from);
    let determinant = aa * bb - ab * ab;
    let (du, dv) = (
        (ab * cb - bb * ca) / determinant,
        (ab * ca - aa * cb) / determinant,
    );
    let middle = (pixel.0 as f32 + du as f32, pixel.1 as f32 + dv as f32);
    let ray = seen.ray(middle).as_dvec3();
    (ray / away.dot(ray), -away.normalize())
}

/// How that point and that normal change with the plane's `c`, `a` and `b`, in that order.
fn leans(
    seen: &Seen,
    planes: &Planes,
    i: usize,
    at: DVec3,
    toward: DVec3,
) -> ([DVec3; 3], [DVec3; 3]) {
    let [c, a, b] = planes.fits[i];
    let pixel = (i % planes.width, i / planes.width);
    let size = f64::from(seen.away(pixel, [a, b, c]).length());
    // The plane is `n . p = 1` from the eye, and `n` is linear in the three.
    let units = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let leans = units.map(|unit| seen.away(pixel, unit).as_dvec3());
    (
        leans.map(|lean| -at * lean.dot(at)),
        leans.map(|lean| (toward * toward.dot(lean) - lean) / size),
    )
}

/// A line that surfaces turn about.
#[derive(Clone, Copy, Debug)]
struct Axis {
    /// Its point nearest the eye, from the eye.
    point: DVec3,
    /// Its direction, a unit vector.
    along: DVec3,
}

impl Axis {
    /// Where `p`, from the eye, lies about the axis: the vector to it straight out from the
    /// axis, and its height along the axis.
    fn around(&self, p: DVec3) -> (DVec3, f64) {
        let from = p - self.point;
        let up = from.dot(self.along);
        (from - self.along * up, up)
    }
}

/// Sums over lines, each of a unit direction `d` through a point `p`: with its moment
/// `m = p x d`, the line's Plücker coordinates.
#[derive(Clone, Copy)]
struct Lines {
    /// How many lines, by their weights.
    weight: f64,
    /// The sum of `m m'`.
    moments: DMat3,
    /// The sum of `m d'`.
    mixed: DMat3,
    /// The sum of `d d'`.
    directions: DMat3,
}

impl Default for Lines {
    /// No lines: zeros, where a matrix's own default is the identity.
    fn default() -> Self {
        Self {
            weight: 0.0,
            moments: DMat3::ZERO,
            mixed: DMat3::ZERO,
            directions: DMat3::ZERO,
        }
    }
}

/// `u v'`.
fn outer(u: DVec3, v: DVec3) -> DMat3 {
    DMat3::from_cols(u * v.x, u * v.y, u * v.z)
}

impl Lines {
    /// Adds the line through `at` along `direction`, weighed by `weight`.
    fn add(&mut self, at: DVec3, direction: DVec3, weight: f64) {
        let moment = at.cross(direction);
        self.weight += weight;
        self.moments += outer(moment, moment) * weight;
        self.mixed += outer(moment, direction) * weight;
        self.directions += outer(direction, direction) * weight;
    }

    /// The sums over these lines and those of `other`.
    fn with(&self, other: &Self) -> Self {
        Self {
            weight: self.weight + other.weight,
            moments: self.moments + other.moments,
            mixed: self.mixed + other.mixed,
            directions: self.directions + other.directions,
        }
    }

    /// The axis the lines meet most nearly; none where they weigh no more than the five
    /// numbers fitted, an axis's four and the pitch, of which each line tells one.
    ///
    /// A motion that turns about a line and slides along it moves the point `p` by
    /// `c x p + s`, and a surface it moves along itself has every normal at right angles to
    /// that: `c . m + s . d = 0` for the normal's line, a linear line complex. Pottmann and
    /// his coauthors fit the motion to the lines by least squares over the unit turns `c`: it
    /// is the eigenvector of the least eigenvalue of `M - B D+ B'` for the three sums, with
    /// the slide `s = -D+ B' c`. The axis is that motion's, through `c x s`, without its pitch,
    /// the slide's part along the turn: lines that belong to a screw miss its axis, and are
    /// held against it as any others are ([`Shell::turns`]).
    fn axis(&self) -> Option<Axis> {
        if self.weight <= 5.0 {
            return None;
        }
        // A direction the normals do not span, as along an upright cylinder's axis, takes no
        // part in the slide: its sum of squares is what rounding leaves, under half of a
        // double's digits of the largest.
        let spans = eigen(self.directions);
        let most = spans.iter().map(|span| span.0).fold(0.0, f64::max);
        let mut inverse = DMat3::ZERO;
        for (value, vector) in spans {
            if value > f64::EPSILON.sqrt() * most {
                inverse += outer(vector, vector) / value;
            }
        }
        let slides = inverse * self.mixed.transpose();
        let turns = eigen(self.moments - self.mixed * slides);
        let least = turns.iter().min_by(|x, y| x.0.total_cmp(&y.0));
        let along = least.expect("three eigenvalues").1;
        Some(Axis {
            point: along.cross(-(slides * along)),
            along,
        })
    }
}

/// The eigenvalues of a symmetric matrix, each with its unit eigenvector, by Jacobi's
/// rotations: each turns two axes so that the entry between them goes, until none is left.
fn eigen(matrix: DMat3) -> [(f64, DVec3); 3] {
    let mut a = matrix.to_cols_array_2d();
    // The eigenvectors, as columns: `v[row][column]`.
    let mut v = DMat3::IDENTITY.to_cols_array_2d();
    // Asked this way round, so that entries that are not numbers end it.
    while a[0][1].abs() + a[0][2].abs() + a[1][2].abs() > 0.0 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let between = a[p][q];
            // An entry too small to change either of its diagonal entries is gone as it is.
            let shows = |diagonal: f64| diagonal.abs() + between.abs() != diagonal.abs();
            if between != 0.0 && (shows(a[p][p]) || shows(a[q][q])) {
                let theta = (a[q][q] - a[p][p]) / (2.0 * between);
                let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
                let (c, s) = (1.0 / t.hypot(1.0), t / t.hypot(1.0));
                for row in a.iter_mut().chain(&mut v) {
                    (row[p], row[q]) = (c * row[p] - s * row[q], s * row[p] + c * row[q]);
                }
                let (upper, lower) = (a[p], a[q]);
                a[p] = std::array::from_fn(|k| c * upper[k] - s * lower[k]);
                a[q] = std::array::from_fn(|k| s * upper[k] + c * lower[k]);
            }
            a[p][q] = 0.0;
            a[q][p] = 0.0;
        }
    }
    std::array::from_fn(|k| (a[k][k], DVec3::new(v[0][k], v[1][k], v[2][k])))
}

#[cfg(test)]
mod tests {
    use batchplan::{DepthImage, Intrinsics, Pose};
    use glam::{Quat, Vec3};

    use super::*;
    use crate::look::{LookOptions, Noise, View};

    #[test]
    fn a_symmetric_matrix_times_each_eigenvector_is_its_eigenvalue_times_it() {
        let full = DMat3::from_cols(
            DVec3::new(4.0, 1.0, -2.0),
            DVec3::new(1.0, 3.0, 0.5),
            DVec3::new(-2.0, 0.5, 6.0),
        );
        // And one of rank 1, two of whose eigenvalues are 0.
        let flat = outer(DVec3::new(1.0, 2.0, -2.0), DVec3::new(1.0, 2.0, -2.0));
        for (matrix, trace) in [(full, 13.0), (flat, 9.0)] {
            let found = eigen(matrix);
            let sum: f64 = found.iter().map(|(value, _)| value).sum();
            assert!(
                (sum - trace).abs() < 1e-12,
                "the eigenvalues add up to {sum}"
            );
            for (value, vector) in found {
                assert!((vector.length() - 1.0).abs() < 1e-12, "{vector}");
                let off = (matrix * vector - vector * value).length();
                assert!(off < 1e-12, "{value} with {vector} is off by {off}");
            }
        }
    }

    #[test]
    fn a_planes_point_and_normal_change_with_it_as_its_leans_say() {
        // A camera that looks down at a slant, and one reading's plane, which leans both ways
        // and was told from a window with more readings to one side.
        let depth = [1.0; 81];
        let view = View {
            image: DepthImage {
                depth: &depth,
                width: 9,
                intrinsics: Intrinsics {
                    fx: 300.0,
                    fy: 300.0,
                    cx: 4.0,
                    cy: 4.0,
                },
                camera: Pose::new(Vec3::new(0.2, -0.1, 1.0), Quat::from_rotation_x(2.6)),
            },
            color: &[],
            noise: Noise {
                constant: 0.0,
                linear: 0.0,
                quadratic: 0.001,
            },
            robot: &[],
            masks: &[],
        };
        let seen = Seen::of(&view, &[], &LookOptions::default(), 1);
        let i = 4 * 9 + 6;
        let with = |fit: [f32; 3]| {
            let mut planes = Planes {
                fits: vec![[1.0, 0.0, 0.0]; 81],
                covariances: vec![[0.0; 6]; 81],
                has: vec![false; 81],
                noise: vec![0.0; 81],
                width: 9,
            };
            planes.fits[i] = fit;
            planes.covariances[i] = [3e-8, -1e-9, 2e-9, 4e-9, 1e-9, 5e-9];
            planes
        };
        let fit = [1.02, 0.004, -0.007];
        let (at, toward) = tangent(&seen, &with(fit), i);
        let (d_at, d_toward) = leans(&seen, &with(fit), i, at, toward);
        // Against the difference a small step either way makes, to a hundredth: the steps are
        // as small as single precision tells within that.
        for (k, step) in [1e-3, 1e-4, 1e-4].into_iter().enumerate() {
            let moved = |by: f32| {
                let mut fit = fit;
                fit[k] += by;
                tangent(&seen, &with(fit), i)
            };
            let ((at_up, toward_up), (at_down, toward_down)) = (moved(step), moved(-step));
            let by = 2.0 * f64::from(step);
            for (name, slope, said) in [
                ("point", (at_up - at_down) / by, d_at[k]),
                ("normal", (toward_up - toward_down) / by, d_toward[k]),
            ] {
                assert!(
                    (slope - said).length() < 1e-2 * said.length(),
                    "the {name} changes by {slope} with number {k} of the plane, not {said}"
                );
            }
        }
    }

    #[test]
    fn the_normals_of_a_cone_and_of_a_cylinder_give_its_axis() {
        let (point, along) = (
            DVec3::new(0.3, -0.2, 0.9),
            DVec3::new(0.1, 0.2, 1.0).normalize(),
        );
        let side = along.any_orthonormal_vector();
        // A cone that widens by half its height, and a cylinder, whose normals span no
        // direction along its axis.
        for widens in [0.5, 0.0] {
            let mut lines = Lines::default();
            for k in 0..40 {
                let (angle, up) = (0.15 * f64::from(k), 0.01 * f64::from(k % 7));
                let out = side * angle.cos() + along.cross(side) * angle.sin();
                let at = point + along * up + out * (0.05 + widens * up);
                lines.add(at, (out - along * widens).normalize(), 1.0);
            }
            let axis = lines.axis().expect("forty lines");
            let (off, _) = axis.around(point);
            assert!(
                axis.along.cross(along).length() < 1e-9 && off.length() < 1e-9,
                "widening by {widens}: {axis:?}"
            );
        }
    }
}
