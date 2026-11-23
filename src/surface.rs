//! Grasps of what was seen ([`grasps_on`]): a parallel gripper's grasps from points on an object's
//! surfaces and the normals out of them ([`Surface`]), whatever its shape.
//!
//! A grasp closes on two patches of surface that face away from each other, no farther apart than
//! the fingers open, with the line between them inside both friction cones: the condition under
//! which two fingers that press with friction hold an object against any push (Nguyen's
//! force closure for two soft fingers). Both patches must have been seen: a surface with nothing
//! seen behind it gives no grasp, since where its other side lies is a guess. A thin wall, a
//! handle and a box's faces are found alike, with no shape assumed.
//!
//! A wall seen from one side only is closed on where the cameras saw past it
//! ([`GraspOptions::shell`]): a ray that reached something farther crossed empty space, so where
//! the line in from a patch meets space every ray around it went past ([`Free`]), the wall has
//! ended, and its other side lies between. Inside a solid no ray goes, so none is assumed there.
//!
//! From each seed point the search goes in along the normal, through every surface that faces the
//! other way within the opening: a cup seen from outside gives its wall, grasped with one finger
//! inside, and its whole width, grasped around it. Patches are a fingertip's size
//! ([`GraspOptions::contact`]), each the mean of the points in it, so noise in single points and
//! their normals averages out. Where two patches lean toward each other, as a wedge's faces, the
//! line is the one that splits the angle between their normals.
//!
//! The hand is the robot's own: its collision spheres at each grasp and along the straight way
//! in to it lie clear of every point, the object's and the rest, except that its fingers may be
//! in what they close on. Seeds run in parallel on the device's threads, then pairs; the points
//! the hand is checked against are those near the object, kept cell by cell ([`Near`]), and a
//! check stops at the first point in the way.
//!
//! Fingers that may come in narrower ([`GraspOptions::narrow`]) are taken to slide along the
//! closing line, as a parallel gripper's do. A finger's sphere is in a point over one stretch of
//! that slide, so a pass over the points near the slide marks every opening the sphere is in
//! something at, and stops once none is left: a grasp that is refused costs about what the open
//! hand's check does, and one that is found a pass over each finger's whole slide. It comes in
//! at the one farthest from what is not the object's, which a planner lets no finger touch.

use std::collections::{HashMap, HashSet};
use std::f32::consts::TAU;

use glam::{Mat3, Quat, Vec2, Vec3};
use parry3d::utils::SymmetricEigen3;
use rayon::prelude::*;

use crate::device::Device;
use crate::error::{Result, ensure_input, input};
use crate::robot::HandSphere;
use crate::sdf::{DepthImage, RobotMask};
use crate::task::GraspOptions;
use crate::types::{Pose, Screw};

/// Points on the surfaces seen and the normals out of them, in one frame: what [`grasps_on`]
/// finds grasps on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Surface {
    /// Points on surfaces (meters).
    pub points: Vec<Vec3>,
    /// Per point, the unit normal out of its surface, toward where it was seen from, or zero
    /// where how the surface faces is not known: no finger presses there, and the point is kept
    /// clear of or closed on like the rest. Each must be unit or zero; only the object's are
    /// read.
    pub normals: Vec<Vec3>,
    /// Per point, whether it lies on the object to grasp: grasps close on the object's points
    /// alone, and nothing else may lie between the fingers. Empty: every point does.
    pub object: Vec<bool>,
    /// Per point, where it was seen from (the camera's position): the ray from there to the point
    /// crossed empty space, which tells [`GraspOptions::shell`] where a wall seen from one side
    /// has ended. The points seen from exactly one place make one view, of the nearest reading in
    /// each direction; a view needs at least nine readings over the object and what lies around
    /// it, so a cloud from a moving sensor, each point with a place of its own, shows nothing and
    /// gives no grasp of one side. Not finite where it is not known (as `join` leaves it,
    /// infinite); empty: not known for any point.
    pub seen_from: Vec<Vec3>,
}

impl Surface {
    /// What one depth image sees: a point for each reading off the robot, seen from the camera,
    /// and for each that shows the object (`object`, a flag per pixel; empty: every pixel does),
    /// the normal of the plane through the readings within `radius` of it (meters; about a
    /// fingertip's patch, 5 mm), toward the camera. A reading whose neighbors within `radius` are
    /// too few or lie on one line, as along an edge seen end on, has no normal. Two views of one
    /// scene make one surface with [`Surface::join`]. The image's rows are read on the device's
    /// threads, and the points come row by row whatever their number.
    pub fn from_depth(
        device: &Device,
        image: &DepthImage<'_>,
        object: &[bool],
        robot: Option<&RobotMask>,
        radius: f32,
    ) -> Result<Self> {
        image.check_with(robot)?;
        ensure_input!(
            object.is_empty() || object.len() == image.depth.len(),
            "{} object flags for {} pixels",
            object.len(),
            image.depth.len()
        );
        ensure_input!(
            radius > 0.0 && radius.is_finite(),
            "a finite positive radius, got {radius} m"
        );
        let image = &image.rigid();
        let (width, height) = (image.width, image.depth.len() / image.width);
        // One row's readings off the robot, those that show the object with their normals.
        let read = |seen: &[Option<Vec3>], v: usize| {
            let mut row = Surface::default();
            for u in 0..width {
                let i = v * width + u;
                let Some(at) = seen[i] else {
                    continue;
                };
                let shows = object.is_empty() || object[i];
                let mut normal = Vec3::ZERO;
                if shows {
                    // The pixels a patch of `radius` covers at this depth, seen square on.
                    let depth = image.depth[i];
                    let reach =
                        (radius * image.intrinsics.fx.max(image.intrinsics.fy) / depth).ceil();
                    let reach = (reach as usize).clamp(1, WINDOW);
                    let rows = v.saturating_sub(reach)..=(v + reach).min(height - 1);
                    let near = rows.flat_map(|row| {
                        let columns = u.saturating_sub(reach)..=(u + reach).min(width - 1);
                        columns.filter_map(move |column| seen[row * width + column])
                    });
                    let near = near.filter(|p| p.distance_squared(at) <= radius * radius);
                    normal = plane(near, at, image.camera.position).unwrap_or(Vec3::ZERO);
                }
                row.points.push(at);
                row.normals.push(normal);
                if !object.is_empty() {
                    row.object.push(shows);
                }
            }
            row
        };
        let rows: Vec<Surface> = device.install(|| {
            let pixels = (0..image.depth.len()).into_par_iter();
            let seen: Vec<Option<Vec3>> = pixels.map(|i| image.point(i, robot)).collect();
            let rows = (0..height).into_par_iter();
            rows.map(|v| read(&seen, v)).collect()
        });
        let mut surface = Surface::default();
        for row in rows {
            surface.points.extend(row.points);
            surface.normals.extend(row.normals);
            surface.object.extend(row.object);
        }
        surface.seen_from = vec![image.camera.position; surface.points.len()];
        Ok(surface)
    }

    /// The surface `points` lie on, all of them the object's, as seen from `seen_from`: each
    /// with the normal of the plane through the points within `radius` of it (meters), toward
    /// `seen_from`, or none where they are too few or lie on one line. `seen_from` is taken as
    /// where one camera saw every point from, the space between it and each point empty, which
    /// [`GraspOptions::shell`] closes on: a cloud fused from several views and given one place
    /// would read as one camera's view, and must not be grasped with a shell.
    pub fn from_points(points: &[Vec3], seen_from: Vec3, radius: f32) -> Result<Self> {
        if let Some(k) = points.iter().position(|p| !p.is_finite()) {
            return Err(input!("point {k} is not finite, got {}", points[k]));
        }
        ensure_input!(
            seen_from.is_finite(),
            "where the points were seen from must be finite, got {seen_from}"
        );
        ensure_input!(
            radius > 0.0 && radius.is_finite(),
            "a finite positive radius, got {radius} m"
        );
        let cells = Cells::new(points, radius, |_| true);
        let mut near = vec![];
        let normals = points
            .iter()
            .map(|&at| {
                cells.within(points, at, at, radius, &mut near);
                let near = near.iter().map(|&(_, i)| points[i]);
                plane(near, at, seen_from).unwrap_or(Vec3::ZERO)
            })
            .collect();
        Ok(Surface {
            points: points.to_vec(),
            normals,
            object: vec![],
            seen_from: vec![seen_from; points.len()],
        })
    }

    /// Adds `other`'s points after this surface's: two views of one scene as one surface. Where
    /// one of the two says which of its points are the object's and the other does not, every
    /// point of the other is; where one says where its points were seen from and the other does
    /// not, the other's are not known (infinite).
    pub fn join(&mut self, other: Surface) {
        if !(self.object.is_empty() && other.object.is_empty()) {
            if self.object.is_empty() {
                self.object = vec![true; self.points.len()];
            }
            if other.object.is_empty() {
                self.object
                    .extend(std::iter::repeat_n(true, other.points.len()));
            } else {
                self.object.extend(&other.object);
            }
        }
        if !(self.seen_from.is_empty() && other.seen_from.is_empty()) {
            if self.seen_from.is_empty() {
                self.seen_from = vec![Vec3::INFINITY; self.points.len()];
            }
            if other.seen_from.is_empty() {
                self.seen_from
                    .extend(std::iter::repeat_n(Vec3::INFINITY, other.points.len()));
            } else {
                self.seen_from.extend(&other.seen_from);
            }
        }
        self.points.extend(other.points);
        self.normals.extend(other.normals);
    }
}

/// The most pixels to either side of a reading its plane is sought among: enough for the
/// readings within 5 mm of one 10 cm from a camera of 600 pixels' focal length.
const WINDOW: usize = 32;

/// The unit normal of the plane nearest the points `near`, toward `toward` from `at`: the
/// direction they spread least along. `None` for fewer than three, for points on one line, whose
/// plane is anyone's guess, and for points that lie near no plane.
fn plane(near: impl Iterator<Item = Vec3>, at: Vec3, toward: Vec3) -> Option<Vec3> {
    let (mut count, mut sum, mut squares) = (0, Vec3::ZERO, Mat3::ZERO);
    for p in near {
        let d = p - at;
        count += 1;
        sum += d;
        squares += Mat3::from_cols(d * d.x, d * d.y, d * d.z);
    }
    if count < 3 {
        return None;
    }
    let mean = sum / count as f32;
    let spread = squares * (1.0 / count as f32)
        - Mat3::from_cols(mean * mean.x, mean * mean.y, mean * mean.z);
    let size = spread.x_axis.x + spread.y_axis.y + spread.z_axis.z;
    if size <= 0.0 {
        return None;
    }
    // Ascending, the plane's normal first, of the spread scaled to a sum of one.
    let eigen = SymmetricEigen3::new(spread * (1.0 / size));
    let [least, middle, _] = eigen.eigenvalues.to_array();
    // On one line, with no spread across it to speak of, or as far off a plane as across it.
    if middle <= 1e-2 || least > 0.5 * middle {
        return None;
    }
    let normal = eigen.eigenvectors.x_axis.try_normalize()?;
    Some(if normal.dot(toward - at) < 0.0 {
        -normal
    } else {
        normal
    })
}

/// A grasp [`grasps_on`] found.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Grasp {
    /// The hand's IK frame in the surface's frame, as [`GraspOptions`] has it: between the
    /// fingertips, its z axis out of the hand and its y axis along the closing direction.
    pub pose: Pose,
    /// Where the fingers press, in the surface's frame: the first on the side the frame's y axis
    /// points from. On the frame's y axis, or ahead of it along its z axis where the hand is
    /// drawn back to stay clear.
    pub contacts: [Vec3; 2],
    /// Whether one of the two contacts was not seen ([`GraspOptions::shell`]): it is where the
    /// cameras saw past the wall, so the wall ends at or before it.
    pub one_sided: bool,
    /// How far apart the fingers come in to it (meters): [`GraspOptions::opening`], or less where
    /// they may come in narrower ([`GraspOptions::narrow`]) and that keeps them farther from
    /// what stands around.
    pub opening: f32,
}

impl Grasp {
    /// How far apart the two points it closes on are (meters): what the fingers close to, or
    /// for a [`one_sided`](Self::one_sided) grasp the most they may, the wall being no thicker.
    #[must_use]
    pub fn width(&self) -> f32 {
        (self.contacts[1] - self.contacts[0]).length()
    }

    /// Whether a ball in the surface's frame lies wholly in the gap this grasp closes, along
    /// its y axis. The ball's projection must fit inside the silhouette of a finger sphere on
    /// either side, and clear every sphere of the hand at the opening the grasp chose. Pass
    /// the same hand spheres and `GraspOptions::opening` that `grasps_on` used.
    #[must_use]
    pub fn closes_on(
        &self,
        hand: &[HandSphere],
        loaded_opening: f32,
        at: Vec3,
        radius: f32,
    ) -> bool {
        if !(at.is_finite()
            && radius.is_finite()
            && radius >= 0.0
            && self.pose.is_unit()
            && self.opening > 0.0
            && loaded_opening.is_finite()
            && self.opening <= loaded_opening)
        {
            return false;
        }
        let pose = self.pose.rigid();
        let p = pose.rotation.conjugate() * (at - pose.position);
        if p.y.abs() + radius > self.opening * 0.5 {
            return false;
        }
        let inward = (loaded_opening - self.opening) * 0.5;
        let (mut left, mut right) = (false, false);
        for sphere in hand {
            let center = sphere.center - Vec3::Y * sphere.side * inward;
            let delta = p - center;
            if delta.length_squared() < (sphere.radius + radius).powi(2) {
                return false;
            }
            let across = Vec2::new(delta.x, delta.z);
            if sphere.radius >= radius
                && across.length_squared() <= (sphere.radius - radius).powi(2)
            {
                left |= sphere.side < 0.0 && delta.y > 0.0;
                right |= sphere.side > 0.0 && delta.y < 0.0;
            }
        }
        left && right
    }
}

/// The fewest points a patch a fingertip presses on holds: fewer say too little of where the
/// surface is and how it faces.
const SUPPORT: usize = 3;

/// The nearest two patches may be (meters): closer, they are one surface's noise, not two sides.
const THINNEST: f32 = 0.0025;

/// How far a hand that is not clear is drawn back at a time (meters).
const BACK: f32 = 0.002;

/// How many openings fingers that may come in narrower ([`GraspOptions::narrow`]) are tried at,
/// evenly from as far as they open to closed: few enough that a call's grasps share them, and
/// 5 mm apart for a hand that opens 8 cm, a quarter of a fingertip's patch to each finger.
const OPENINGS: i32 = 16;

/// The smallest contact or preshape whose width a measured hand must be able to reach.
pub(crate) fn smallest_gap(capacity: f32) -> f32 {
    let step = capacity / (2 * OPENINGS) as f32;
    THINNEST.min(capacity - 2.0 * last_opening(capacity, THINNEST, step) as f32 * step)
}

fn last_opening(capacity: f32, width: f32, step: f32) -> i32 {
    (((capacity - width) / 2.0 / step + 1e-3) as i32).clamp(0, OPENINGS - 1)
}

impl GraspOptions {
    /// Orders measured grasps by the moment their finger contacts can exert about this turn's
    /// axis, greatest first. Equal moments retain the input order. Give the measured screw in
    /// the grasps' frame, or `None` to turn each hand about its own z axis.
    pub fn order_for_turn(&self, grasps: &mut [Grasp], screw: Option<Screw>) {
        let moment = |g: &Grasp| {
            let axis = screw.unwrap_or_else(|| {
                Screw::about(g.pose.position, g.pose.rigid().rotation * Vec3::Z)
            });
            g.moment(&axis, self)
        };
        grasps.sort_by(|a, b| moment(b).total_cmp(&moment(a)));
    }

    /// The openings a grasp may come in at where [`narrow`](GraspOptions::narrow) lets it
    /// (meters), widest first: `opening` and each sixteenth of it less. A place may let go at
    /// the same ([`Place::open`](crate::Place::open)), each as the fingers' joint values.
    #[must_use]
    pub fn openings(&self) -> Vec<f32> {
        let step = self.opening / (2 * OPENINGS) as f32;
        let openings = (0..OPENINGS).map(|k| self.opening - 2.0 * k as f32 * step);
        openings.collect()
    }
}

/// The cells the points near the hand are kept in (meters): a few to a sphere.
const HAND_CELLS: f32 = 0.01;

/// The most pairs of patches that are put in an order that spreads them over the object.
const SPREAD: usize = 1024;

/// The most cells they are kept in: past it the cells are larger.
const MOST_CELLS: usize = 1 << 22;

/// How finely the line behind a patch seen from one side is searched for where the cameras saw
/// past the wall (meters).
const BEHIND: f32 = 0.0005;

/// How much wider than the spacing of a view's readings its cells are: wide enough that every
/// cell over a surface seen whole holds one.
const WIDER: f32 = 1.5;

/// How many of a view's readings its spacing is measured at.
const SAMPLES: usize = 4096;

/// The most rings of cells around a ray its nearest is sought over, cells as wide as the rays
/// would be apart spread evenly: past it, a ray lies too far from the rest to tell their spacing.
const RINGS: usize = 8;

/// Grasps on each of `surfaces` for hand `hand` of `device`'s robot, as its locked joints hold
/// its fingers: each pair of patches the hand can close on is grasped from `turns` directions
/// around its closing line, each both ways round, wherever the hand and its straight way in
/// (`approach`) are clear of every point and over the `floor`, drawn back by up to what leaves
/// `overlap` of the fingers on the patches. Where the fingers may come in narrower
/// ([`GraspOptions::narrow`]), a grasp is one the hand is clear at with them at some opening, and
/// says which ([`Grasp::opening`]).
///
/// They come in an order whose first few are worth trying first: every pair from above (where
/// its line is not upright), then every pair a turn to either side, and so on round; and within
/// a turn, the pairs that face each other within half a friction cone of square before the rest,
/// spread over the object, each the farthest from those before it, or with a point to grip near
/// ([`GraspOptions::near`]) the nearest it first. With a
/// [`shell`](GraspOptions::shell), grasps supported by views past a wall follow those with
/// both sides seen. [`GraspOptions::close_on`] appends intermediate approaches that enclose the
/// point after all ordinary grasps, again with two seen sides first. Finally it tries shallower
/// insertions of those grasps that enclose the point, in parent grasp order. The surfaces run on
/// the device's threads, and a surface's grasps are the same alone and among others.
pub fn grasps_on(
    device: &Device,
    hand: usize,
    surfaces: &[Surface],
    o: &GraspOptions,
) -> Result<Vec<Vec<Grasp>>> {
    grasps_on_with_capacity(device, hand, surfaces, o, o.opening)
}

/// As [`grasps_on`], for a hand whose maximum graspable width (`capacity`, meters) may be
/// smaller than the gap between its pads as loaded ([`GraspOptions::opening`]). Contact pairs
/// fit within the capacity. With [`GraspOptions::narrow`], the preshapes are the loaded gap
/// and 15/16 through 1/16 of the capacity. Each moving finger slides half the change in gap
/// along the closing line. Use the chosen [`Grasp::opening`] to obtain the commanded locks.
/// [`Robot::parallel_jaw_opening`](crate::Robot::parallel_jaw_opening) checks the model and
/// gives the calibrated loaded gap to use here and in closing-region projection. With narrowing
/// disabled, the loaded hand geometry is used directly.
pub fn grasps_on_with_capacity(
    device: &Device,
    hand: usize,
    surfaces: &[Surface],
    o: &GraspOptions,
    capacity: f32,
) -> Result<Vec<Vec<Grasp>>> {
    let hands = device.robot().hands();
    ensure_input!(hand < hands, "hand {hand} of {hands}");
    ensure_input!(
        o.opening > 0.0
            && o.opening.is_finite()
            && o.turns > 0
            && o.friction > 0.0
            && o.friction.is_finite()
            && o.contact > 0.0
            && o.contact.is_finite()
            && o.approach >= 0.0
            && o.approach.is_finite()
            && o.overlap > 0.0
            && o.overlap.is_finite(),
        "grasps need a finite positive opening, a turn, a finite positive friction, contact and overlap, and a finite approach of at least 0, got {} m, {}, {}, {} m, {} m and {} m",
        o.opening,
        o.turns,
        o.friction,
        o.contact,
        o.overlap,
        o.approach
    );
    ensure_input!(
        capacity > 0.0 && capacity <= o.opening,
        "grasp capacity {capacity} m must be positive and at most the loaded gap {} m",
        o.opening
    );
    if let Some(floor) = o.floor {
        ensure_input!(floor.is_finite(), "the floor must be finite, got {floor}");
    }
    if let Some(shell) = o.shell {
        ensure_input!(
            shell > 0.0 && shell.is_finite(),
            "a shell must be a finite positive thickness, got {shell} m"
        );
    }
    if let Some(near) = o.near {
        ensure_input!(
            near.is_finite(),
            "the point to grip near must be finite, got {near}"
        );
    }
    if let Some(at) = o.close_on {
        ensure_input!(
            at.is_finite(),
            "the point to close on must be finite, got {at}"
        );
    }
    for (i, s) in surfaces.iter().enumerate() {
        ensure_input!(
            s.normals.len() == s.points.len()
                && (s.object.is_empty() || s.object.len() == s.points.len())
                && (s.seen_from.is_empty() || s.seen_from.len() == s.points.len()),
            "surface {i}: {} points with {} normals, {} object flags and {} places seen from",
            s.points.len(),
            s.normals.len(),
            s.object.len(),
            s.seen_from.len()
        );
        if let Some(k) = s.points.iter().position(|p| !p.is_finite()) {
            return Err(input!(
                "surface {i}: point {k} is not finite, got {}",
                s.points[k]
            ));
        }
        let unit = |n: &Vec3| *n == Vec3::ZERO || (n.length_squared() - 1.0).abs() < 1e-3;
        if let Some(k) = s.normals.iter().position(|n| !unit(n)) {
            return Err(input!(
                "surface {i}: normal {k} must be unit or zero, got {}",
                s.normals[k]
            ));
        }
    }
    // The fingers first: they are the likeliest to be in the way.
    let mut hand = device.robot().hand_spheres(hand);
    hand.sort_by_key(|sphere| !sphere.finger);
    let partial = (capacity < o.opening && o.narrow).then(|| {
        let inset = (o.opening - capacity) * 0.5;
        hand.iter()
            .map(|sphere| HandSphere {
                center: sphere.center - Vec3::Y * (sphere.side * inset),
                ..*sphere
            })
            .collect::<Vec<_>>()
    });
    let open = OpenHand {
        spheres: &hand,
        at_capacity: partial.as_deref(),
        gap: o.opening,
    };
    let options = GraspOptions {
        opening: capacity,
        ..*o
    };
    let each = || surfaces.par_iter().map(|s| on(s, open, &options)).collect();
    Ok(device.install(each))
}

/// Two patches a hand can close on.
struct Pair {
    /// The middle of the two patches.
    middle: Vec3,
    /// The unit closing line, from the first patch to the second.
    line: Vec3,
    width: f32,
    /// The cosine of the larger angle between the line and a patch's normal.
    square: f32,
    /// Where it comes among the pairs found, the squarest first.
    order: usize,
}

/// The geometry and pad gap at the loaded joint positions.
#[derive(Clone, Copy)]
struct OpenHand<'a> {
    spheres: &'a [HandSphere],
    at_capacity: Option<&'a [HandSphere]>,
    gap: f32,
}

/// One surface's grasps for a hand of spheres `hand`.
fn on(s: &Surface, open: OpenHand<'_>, o: &GraspOptions) -> Vec<Grasp> {
    let hand = open.spheres;
    let (pairs, one_sided) = pairs(s, o);
    if pairs.is_empty() && one_sided.is_empty() {
        return vec![];
    }
    // How far the hand reaches past its frame, and how far from it any of it comes, its way in
    // included.
    let reach = hand
        .iter()
        .map(|sphere| sphere.center.z + sphere.radius)
        .fold(0.0, f32::max);
    let mut extent = hand
        .iter()
        .map(|sphere| sphere.center.length() + sphere.radius)
        .fold(0.0, f32::max)
        + o.approach
        + reach;
    if o.narrow {
        // What stands around is sought as far out as the fingers could come in.
        extent += open.gap / 2.0;
    }
    let middles = pairs.iter().chain(&one_sided).map(|pair| pair.middle);
    let lower = middles.clone().fold(Vec3::INFINITY, Vec3::min) - extent;
    let upper = middles.fold(Vec3::NEG_INFINITY, Vec3::max) + extent;
    let near = Near::new(s, lower, upper);
    // Whether the hand at `pose`, open, and the straight way in to it along its z axis, are
    // clear of what was seen and over the floor: each sphere swept back the approach, as GPD
    // takes a hand's fingers to reach back without end, so that a sheet of points between two
    // of a finger's spheres is in the way of the one that passed through it. Its fingers may
    // be in what they close on: the object's points between the two patches, `width` apart
    // along the frame's y axis.
    let clear = |pose: &Pose, width: f32| {
        let (closing, toward) = (pose.rotation * Vec3::Y, pose.rotation * Vec3::Z);
        if let Some(floor) = o.floor {
            let tips = (pose.position + toward * reach).z - closing.z.abs() * open.gap / 2.0;
            if pose.position.z.min(tips) < floor {
                return false;
            }
        }
        hand.iter().all(|sphere| {
            let center = pose.position + pose.rotation * sphere.center;
            let from = center - toward * o.approach;
            !near.any(from, center, sphere.radius, |point, object| {
                let across = (point - pose.position).dot(closing).abs();
                !(sphere.finger && object && across <= width / 2.0 + THINNEST)
            })
        })
    };
    // A pair's grasps, by turn: from above first, where the line is not upright, then a turn to
    // either side of that, and so on round to from below.
    let full = |pair: &Pair, one_sided: bool, turns: usize| {
        let down = Vec3::NEG_Z.reject_from_normalized(pair.line);
        let first = down
            .try_normalize()
            .unwrap_or_else(|| pair.line.any_orthonormal_vector());
        let by_turn = (0..turns).map(|turn| {
            let mut grasps = Vec::with_capacity(2);
            let side = if turn % 2 == 1 { 1.0 } else { -1.0 };
            let angle = side * turn.div_ceil(2) as f32 * TAU / turns as f32;
            let toward = Quat::from_axis_angle(pair.line, angle) * first;
            for closing in [pair.line, -pair.line] {
                let axes = Mat3::from_cols(closing.cross(toward), closing, toward);
                let rotation = Quat::from_mat3(&axes);
                // As deep as the hand is clear: its frame on the line between the patches, or
                // drawn back while its fingers still overlap them by `overlap`.
                let mut back = 0.0;
                while back == 0.0 || reach - back >= o.overlap {
                    let pose = Pose::new(pair.middle - toward * back, rotation);
                    if clear(&pose, pair.width) {
                        let half = closing * pair.width / 2.0;
                        grasps.push(Grasp {
                            pose,
                            contacts: [pair.middle - half, pair.middle + half],
                            one_sided,
                            opening: open.gap,
                        });
                        break;
                    }
                    back += BACK;
                }
            }
            grasps
        });
        by_turn.collect::<Vec<_>>()
    };
    // Every pair from above, then every pair a turn on: the pairs being spread over the object,
    // so are the first few grasps. Those of walls seen from one side after the rest, alike.
    let proposals = |pair: &Pair, one_sided: bool, turns: usize| match o.narrow {
        true => narrowly(pair, one_sided, &near, open, reach, o, turns),
        false => full(pair, one_sided, turns),
    };
    let mut grasps = vec![];
    for (lot, one_sided) in [(&pairs, false), (&one_sided, true)] {
        let each: Vec<_> = lot
            .par_iter()
            .map(|p| proposals(p, one_sided, o.turns))
            .collect();
        for turn in 0..o.turns {
            let at_turn = each
                .iter()
                .flat_map(|by_turn| by_turn[turn].iter().copied());
            grasps.extend(at_turn);
        }
    }
    if let Some(point) = o.close_on {
        let back = (reach - o.overlap).max(0.0);
        let radial_limit = hand
            .iter()
            .filter(|s| s.side != 0.0)
            .map(|s| {
                s.center
                    .x
                    .hypot(s.center.z)
                    .max(s.center.x.hypot(s.center.z - back))
                    + s.radius
            })
            .fold(0.0, f32::max);
        for (lot, one_sided) in [(&pairs, false), (&one_sided, true)] {
            let extras = |pair: &Pair| {
                let delta = point - pair.middle;
                let radial = delta.reject_from_normalized(pair.line).length();
                if delta.dot(pair.line).abs() > open.gap * 0.5 || radial > radial_limit {
                    return Vec::new();
                }
                // Resolve the point's orbit about the contact line to the same spatial
                // increment used for insertion. Dyadic subdivisions retain the coarse angles.
                let mut turns = o.turns;
                while TAU * radial / turns as f32 > BACK {
                    let Some(next) = turns.checked_mul(2) else {
                        break;
                    };
                    turns = next;
                }
                if turns == o.turns {
                    return Vec::new();
                }
                let every = turns / o.turns;
                let by_turn = proposals(pair, one_sided, turns);
                let mut extra = Vec::new();
                for (turn, gs) in by_turn.into_iter().enumerate() {
                    let offset = turn.div_ceil(2);
                    if offset % every != 0 {
                        for g in gs {
                            if g.closes_on(hand, open.gap, point, 0.0) {
                                extra.push((offset as f64 / turns as f64, turn % 2 == 0, g));
                            }
                        }
                    }
                }
                extra
            };
            let each: Vec<_> = lot.par_iter().map(extras).collect();
            let mut extra: Vec<_> = each.into_iter().flatten().collect();
            extra.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            grasps.extend(extra.into_iter().map(|(_, _, g)| g));
        }
        // The deepest point-clear insertion may put the palm into the collision world.
        // Keep each ranked grasp first, then try shallower insertions on the same patches.
        let fingertip = |side| {
            hand.iter()
                .filter(|s| s.side == side)
                .map(|s| s.center.z + s.radius)
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let finger_reach = fingertip(-1.0).min(fingertip(1.0));
        let shallower = |g: &Grasp| {
            let mut extra = Vec::new();
            if !g.closes_on(hand, open.gap, point, 0.0) {
                return extra;
            }
            let toward = g.pose.rotation * Vec3::Z;
            let middle = g.contacts[0].midpoint(g.contacts[1]);
            let depth = finger_reach - (middle - g.pose.position).dot(toward);
            let mut back = BACK;
            while depth - back >= o.overlap {
                let pose = Pose::new(g.pose.position - toward * back, g.pose.rotation);
                let opening = if o.narrow {
                    narrowed(&near, open, &pose, g.width(), reach, o, true).map(|(gap, _)| gap)
                } else {
                    clear(&pose, g.width()).then_some(open.gap)
                };
                if let Some(opening) = opening {
                    let next = Grasp {
                        pose,
                        opening,
                        ..*g
                    };
                    if next.closes_on(hand, open.gap, point, 0.0) {
                        extra.push(next);
                    }
                }
                back += BACK;
            }
            extra
        };
        let depths: Vec<_> = grasps.par_iter().flat_map_iter(shallower).collect();
        grasps.extend(depths);
    }
    grasps
}

/// A pair's grasps by turn, as [`on`] takes them, where the fingers may come in narrower
/// ([`GraspOptions::narrow`]): where the open hand is clear, as deep as it is, so that a grasp
/// it has stays where it is; where it is clear at no depth, as deep as the hand is clear at some
/// opening. Each with the opening it comes in with there.
#[inline(never)]
fn narrowly(
    pair: &Pair,
    one_sided: bool,
    near: &Near,
    open: OpenHand<'_>,
    reach: f32,
    o: &GraspOptions,
    turns: usize,
) -> Vec<Vec<Grasp>> {
    let down = Vec3::NEG_Z.reject_from_normalized(pair.line);
    let first = down
        .try_normalize()
        .unwrap_or_else(|| pair.line.any_orthonormal_vector());
    let by_turn = (0..turns).map(|turn| {
        let mut grasps = Vec::with_capacity(2);
        let side = if turn % 2 == 1 { 1.0 } else { -1.0 };
        let angle = side * turn.div_ceil(2) as f32 * TAU / turns as f32;
        let toward = Quat::from_axis_angle(pair.line, angle) * first;
        for closing in [pair.line, -pair.line] {
            let axes = Mat3::from_cols(closing.cross(toward), closing, toward);
            let rotation = Quat::from_mat3(&axes);
            // The deepest the hand is clear at narrower, kept for where the open hand is
            // clear nowhere; past it only the open hand is looked for.
            let mut narrower = None;
            let mut back = 0.0;
            while back == 0.0 || reach - back >= o.overlap {
                let pose = Pose::new(pair.middle - toward * back, rotation);
                let whole = narrower.is_none();
                match narrowed(near, open, &pose, pair.width, reach, o, whole) {
                    Some((opening, /* open */ true)) => {
                        let opening = match whole {
                            true => opening,
                            false => narrowed(near, open, &pose, pair.width, reach, o, true)
                                .map_or(opening, |(opening, _)| opening),
                        };
                        narrower = Some((pose, opening));
                        break;
                    }
                    Some(found) if whole => narrower = Some((pose, found.0)),
                    _ => {}
                }
                back += BACK;
            }
            if let Some((pose, opening)) = narrower {
                let half = closing * pair.width / 2.0;
                grasps.push(Grasp {
                    pose,
                    contacts: [pair.middle - half, pair.middle + half],
                    one_sided,
                    opening,
                });
            }
        }
        grasps
    });
    by_turn.collect()
}

/// The opening the hand comes in with at `pose`, closing on patches `width` apart, where its
/// fingers may come in narrower ([`GraspOptions::narrow`]), if it is clear at one. Each finger's
/// spheres are taken to slide along the closing line, and the opening is the one farthest from
/// any at which they would be in what is not the object's, were they to open as far past open as
/// they close: a planner lets the fingers touch what they grasp and nothing else, and knows what
/// stands around less finely than the points do. Of the openings as far, the one with the most
/// room, as far from the nearest the hand is in anything at farther apart as from the nearest
/// nearer, the patches at the last; and of those, the widest, so the whole opening with nothing
/// near. With it, whether the open hand is clear there; and unless `whole`, only that is looked
/// for, the opening the whole one. `reach` is how far the hand reaches past its frame.
fn narrowed(
    near: &Near,
    open: OpenHand<'_>,
    pose: &Pose,
    width: f32,
    reach: f32,
    o: &GraspOptions,
    whole: bool,
) -> Option<(f32, bool)> {
    let hand = open.spheres;
    let (closing, toward) = (pose.rotation * Vec3::Y, pose.rotation * Vec3::Z);
    // A finger's way from one opening to the next, and how many take it onto its patch: a
    // rounding short of one more is one more.
    let step = o.opening / (2 * OPENINGS) as f32;
    let gap = |k: i32| {
        let origin = if k > 0 { o.opening } else { open.gap };
        origin - 2.0 * k as f32 * step
    };
    let most = match whole {
        true => last_opening(o.opening, width, step),
        false => 0,
    };
    // The openings from `first` to `last`, as bits: one for each count of steps the fingers
    // are moved toward each other, below zero for apart.
    let span = |first: i32, last: i32| {
        if first > last {
            return 0u64;
        }
        ((1u64 << (last - first + 1)) - 1) << (first + OPENINGS)
    };
    // Those the hand is in something at, and in something that is not the object's.
    let (mut taken, mut others) = (0, 0);
    if let Some(floor) = o.floor {
        if pose.position.z < floor {
            return None;
        }
        let tip = (pose.position + toward * reach).z;
        for k in -most..=most {
            if tip - closing.z.abs() * (gap(k) / 2.0) < floor {
                others |= span(k, k);
            }
        }
        taken = others;
    }
    let closed_on = |sphere: &HandSphere, point: Vec3, object: bool| {
        let across = (point - pose.position).dot(closing).abs();
        sphere.finger && object && across <= width / 2.0 + THINNEST
    };
    let swept = |sphere: &HandSphere| {
        let center = pose.position + pose.rotation * sphere.center;
        (center - toward * o.approach, center)
    };
    // What does not move with the fingers, as the hand's own opening is checked.
    for sphere in hand.iter().filter(|sphere| sphere.side == 0.0) {
        let (from, center) = swept(sphere);
        let other = |point, object| !closed_on(sphere, point, object);
        if near.any(from, center, sphere.radius, other) {
            return None;
        }
    }
    // Marks the openings from `first` to `last` at which a finger's sphere is in something,
    // until none is left to mark, and says whether the hand is in something at every one. Of
    // the openings it has, none is left once it is in something at each; of those past open,
    // which only place the nearest thing that is not the object's, once that is at each.
    let mut fingers = |first: i32, last: i32, spheres: &[HandSphere]| {
        let all = span(first, last);
        let marked = |taken: u64, others: u64| match first < 0 {
            true => others & all == all,
            false => taken & all == all,
        };
        for sphere in spheres.iter().filter(|sphere| sphere.side != 0.0) {
            if marked(taken, others) {
                break;
            }
            let (from, center) = swept(sphere);
            let reached = sphere.radius * sphere.radius;
            // A finger nears the other against the side it stands on.
            let (least, far) = (first as f32 * step, last as f32 * step);
            let moves = if sphere.side > 0.0 {
                (-far, -least)
            } else {
                (least, far)
            };
            let slides = (closing, moves);
            near.until(from, center, sphere.radius, slides, |off, point, object| {
                if closed_on(sphere, point, object) {
                    return false;
                }
                // As the hand's own opening is checked.
                let mut at = 0;
                if first <= 0 && last >= 0 && off.length_squared() <= reached {
                    at = span(0, 0);
                }
                // With the fingers `k` steps nearer the point is `along + side * k * step`
                // along the closing line from the sphere's path, which reaches it within
                // `half` of there.
                let along = off.dot(closing);
                let half = (reached - (off.length_squared() - along * along)).sqrt();
                if !half.is_nan() {
                    let (nearer, farther) = if sphere.side > 0.0 {
                        (-along - half, -along + half)
                    } else {
                        (along - half, along + half)
                    };
                    let nearer = ((nearer / step).ceil() as i32).max(first);
                    let farther = ((farther / step).floor() as i32).min(last);
                    at |= span(nearer, farther);
                }
                taken |= at;
                if !object {
                    others |= at;
                }
                marked(taken, others)
            });
        }
        taken & all == all
    };
    // The openings it has, then what stands around past the widest of them.
    let blocked = match open.at_capacity {
        Some(partial) if most > 0 => {
            // Only the full command goes to the loaded joints.
            let full = fingers(0, 0, hand);
            fingers(1, most, partial) && full
        }
        _ => fingers(0, most, hand),
    };
    if blocked {
        return None;
    }
    fingers(-most, -1, hand);
    let free = |k: i32| taken & span(k, k) == 0;
    // How many openings lie between this one and the nearest at which the hand is in what is
    // not the object's, and in anything.
    let from_others = |k: i32| {
        let theirs = (-most..=most).filter(|&j| others & span(j, j) != 0);
        theirs.map(|j| (j - k).abs()).min().unwrap_or(i32::MAX)
    };
    let room = |k: i32| {
        let apart = (-most..k).rev().find(|&j| !free(j));
        let together = (k + 1..=most).find(|&j| !free(j)).unwrap_or(most + 1);
        apart.map_or(i32::MAX, |j| k - j).min(together - k)
    };
    let clear = (0..=most).filter(|&k| free(k));
    let best = clear.max_by_key(|&k| (from_others(k), room(k), -k))?;
    Some((gap(best), free(0)))
}

/// `pairs`, those that face each other most squarely first, in an order that spreads them:
/// first those within half a friction cone of square, then the rest, each lot starting with its
/// squarest pair and going on with the pair farthest from every one before it. Past
/// [`SPREAD`] pairs of a lot, the rest follow as they came. With a point `near`, each lot's
/// pairs come nearest it first instead.
fn spread(pairs: Vec<Pair>, cone: f32, near: Option<Vec3>) -> Vec<Pair> {
    let well = ((1.0 + cone) / 2.0).sqrt();
    let (square, askew): (Vec<Pair>, Vec<Pair>) = pairs.into_iter().partition(|p| p.square >= well);
    let mut out = Vec::with_capacity(square.len() + askew.len());
    for mut lot in [square, askew] {
        if let Some(point) = near {
            let from = |p: &Pair| p.middle.distance_squared(point);
            lot.sort_by(|a, b| from(a).total_cmp(&from(b)).then(a.order.cmp(&b.order)));
            out.extend(lot);
            continue;
        }
        // How far each pair left is from the nearest one taken, squared.
        let mut from: Vec<f32> = vec![f32::INFINITY; lot.len()];
        let mut next = 0;
        for _ in 0..lot.len().min(SPREAD) {
            let taken = lot.swap_remove(next);
            from.swap_remove(next);
            next = 0;
            for (k, pair) in lot.iter().enumerate() {
                from[k] = from[k].min(pair.middle.distance_squared(taken.middle));
                // The farthest, and of those as far, the squarest and then the first found.
                let better = (from[k], pair.square, lot[next].order).partial_cmp(&(
                    from[next],
                    lot[next].square,
                    pair.order,
                ));
                if better == Some(std::cmp::Ordering::Greater) {
                    next = k;
                }
            }
            out.push(taken);
        }
        lot.sort_by_key(|pair| pair.order);
        out.extend(lot);
    }
    out
}

/// The pairs of patches of the object's surfaces a hand can close on, those that face each other
/// most squarely first; and with a [`shell`](GraspOptions::shell), those of walls seen from one
/// side, apart.
fn pairs(s: &Surface, o: &GraspOptions) -> (Vec<Pair>, Vec<Pair>) {
    let object = |i: usize| s.object.is_empty() || s.object[i];
    // The object's points, and what else lies among them: only that can be between two of its
    // patches.
    let its = (0..s.points.len())
        .filter(|&i| object(i))
        .map(|i| s.points[i]);
    let lower = its.clone().fold(Vec3::INFINITY, Vec3::min) - o.contact;
    let upper = its.fold(Vec3::NEG_INFINITY, Vec3::max) + o.contact;
    let among = |i: usize| s.points[i].cmpge(lower).all() && s.points[i].cmple(upper).all();
    let cells = Cells::new(&s.points, o.contact, among);
    // What the cameras saw past, where a wall seen from one side may end within the shell.
    let thickest = o.shell.map_or(0.0, |shell| shell.min(o.opening));
    let free = o.shell.and_then(|_| {
        let behind = Vec3::splat(thickest);
        Some(Free::new(s, lower - behind, upper + behind)).filter(|free| !free.views.is_empty())
    });
    // The cosine of the friction cone's half angle.
    let cone = 1.0 / (1.0 + o.friction * o.friction).sqrt();
    // Where a line from `from` along `toward` crosses the object's surfaces going out of it
    // within the opening, nearest first, and the normal out of each: each run of points that
    // face along the line (`facing`, the least cosine between their normals and it), within a
    // patch's size of it and of each other, is one patch, crossed where the line meets its plane.
    // A surface the line runs along, as a box's side, faces across it and is not crossed.
    let crossed = |from: Vec3, toward: Vec3, facing: f32, near: &mut Vec<(f32, usize)>| {
        cells.within(&s.points, from, from + toward * o.opening, o.contact, near);
        near.retain(|&(t, i)| object(i) && t > 0.0 && s.normals[i].dot(toward) >= facing);
        near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut crossings = vec![];
        let mut start = 0;
        for end in 1..=near.len() {
            if end == near.len() || near[end].0 - near[end - 1].0 > o.contact {
                if let Some((middle, normal)) = patch(s, &near[start..end]) {
                    let along = (middle - from).dot(normal) / toward.dot(normal);
                    crossings.push((from + toward * along, normal));
                }
                start = end;
            }
        }
        crossings
    };
    // Two patches whose normals are more than two cones apart share no line inside both. Past a
    // sixth of a turn apart they are not sought: nearer a quarter, a surface a line runs along
    // would count as one it crosses.
    let apart = (2.0 * cone * cone - 1.0).max(0.5);
    // Whether anything else lies between patches `width` apart along the line from `first`: it
    // would be between the fingers.
    let between = |first: Vec3, second: Vec3, width: f32, near: &mut Vec<(f32, usize)>| {
        cells.within(&s.points, first, second, o.contact, near);
        let other = |&(t, i): &(f32, usize)| !object(i) && t > o.contact && t < width - o.contact;
        near.iter().any(other)
    };
    // The pair a patch seen from one side makes where a line in from it (`line`, `square` the
    // cosine between it and the patch's normal) meets space the cameras saw past, within the
    // shell's thickness, with nothing of the object seen facing along it on the way: the wall
    // has ended there, its other side taken parallel to this one.
    let unseen_side = |free: &Free, first: Vec3, line: Vec3, square: f32, near: &mut Vec<_>| {
        let steps = ((thickest - THINNEST) / BEHIND).floor().max(-1.0) as i32;
        let width = (0..=steps)
            .map(|k| THINNEST + k as f32 * BEHIND)
            .find(|&behind| free.seen_past(first + line * behind))?;
        let second = first + line * width;
        let seen = crossed(first, line, f32::MIN_POSITIVE, near);
        let before = |&(at, _): &(Vec3, Vec3)| (at - first).dot(line) <= width;
        if seen.iter().any(before) || between(first, second, width, near) {
            return None;
        }
        Some(Pair {
            middle: (first + second) / 2.0,
            line,
            width,
            square,
            order: 0,
        })
    };
    // The pairs a seed's patch makes with those across from it, each with the cells its two
    // patches fall in, and the pair it makes as a wall seen from one side.
    let across = |near: &mut Vec<(f32, usize)>, seed: usize| {
        let mut pairs = vec![];
        let (p, n) = (s.points[seed], s.normals[seed]);
        // The patch around the seed: the object's points there that face as it does.
        cells.within(&s.points, p, p, o.contact, near);
        near.retain(|&(_, i)| object(i) && s.normals[i].dot(n) >= cone);
        let Some((first, out)) = patch(s, near) else {
            return (pairs, [None, None]);
        };
        let inward = -out;
        for (second, facing) in crossed(first, inward, apart, near) {
            // Two patches hold along the line that splits the angle between their normals,
            // which is inside the first's cone, the two being no more than two cones apart.
            // Where it is not the line searched along, as between a wedge's faces, or the line
            // searched along is outside the second's cone, the patch it crosses is the one to
            // close on.
            let split = (inward + facing).normalize();
            let (second, facing) = if split.dot(inward) < 1.0 - 1e-4 || facing.dot(inward) < cone {
                let nearest = |a: &(Vec3, Vec3), b: &(Vec3, Vec3)| {
                    let to = |x: &(Vec3, Vec3)| (x.0 - second).length_squared();
                    to(a).total_cmp(&to(b))
                };
                match crossed(first, split, cone, near)
                    .into_iter()
                    .min_by(nearest)
                {
                    Some(there) => there,
                    None => continue,
                }
            } else {
                (second, facing)
            };
            let width = (second - first).length();
            if !(THINNEST..=o.opening).contains(&width) {
                continue;
            }
            // Inside both cones: the line splits the normals, and every point of the patch it
            // crosses faces within a cone of it.
            let line = (second - first) / width;
            let square = inward.dot(line).min(facing.dot(line));
            if between(first, second, width, near) {
                continue;
            }
            let pair = Pair {
                middle: (first + second) / 2.0,
                line,
                width,
                square,
                order: 0,
            };
            pairs.push((pair, cells.key(first), cells.key(second)));
        }
        // A wall seen from one side is closed on along its normal and, where it leans, level
        // too if that is inside its friction cone: a hand closing level comes in from straight
        // above, as across a flared rim, where one closing along the normal leans as the wall
        // does and puts a finger down beside it.
        let level = Vec3::new(inward.x, inward.y, 0.0).try_normalize();
        let level = level.filter(|level| (cone..1.0 - 1e-4).contains(&level.dot(inward)));
        let alone = free.as_ref().map_or([None, None], |free| {
            let leveled =
                level.and_then(|line| unseen_side(free, first, line, line.dot(inward), near));
            [unseen_side(free, first, inward, 1.0, near), leveled]
        });
        (pairs, alone)
    };
    // One seed per cell: its first point of the object.
    let seeds: Vec<usize> = cells.firsts().filter(|&i| object(i)).collect();
    let each: Vec<_> = seeds
        .par_iter()
        .map_init(Vec::new, |near, &seed| across(near, seed))
        .collect();
    // The same two patches come from seeds on either side: the first to find them keeps them.
    let mut found = HashSet::new();
    let mut pairs = vec![];
    let mut one_sided = vec![];
    for (seen, alone) in each {
        for (pair, a, b) in seen {
            if found.insert((a.min(b), a.max(b))) {
                pairs.push(pair);
            }
        }
        one_sided.extend(alone.into_iter().flatten());
    }
    let in_order = |mut pairs: Vec<Pair>| {
        pairs.sort_by(|a, b| b.square.total_cmp(&a.square));
        for (order, pair) in pairs.iter_mut().enumerate() {
            pair.order = order;
        }
        spread(pairs, cone, o.near)
    };
    (in_order(pairs), in_order(one_sided))
}

/// The patch `near` makes (each a point's index, after how far along a line it lies): its middle
/// and the unit normal out of it, the means of its points'. `None` for fewer points than a patch
/// needs, or normals that cancel.
fn patch(s: &Surface, near: &[(f32, usize)]) -> Option<(Vec3, Vec3)> {
    if near.len() < SUPPORT {
        return None;
    }
    let (mut middle, mut normal) = (Vec3::ZERO, Vec3::ZERO);
    for &(_, i) in near {
        middle += s.points[i];
        normal += s.normals[i];
    }
    Some((middle / near.len() as f32, normal.try_normalize()?))
}

/// Points by the cell of a grid they fall in, to find those near a line.
struct Cells {
    size: f32,
    of: HashMap<[i32; 3], Vec<usize>>,
    /// The cells in the order their first points come.
    order: Vec<[i32; 3]>,
}

impl Cells {
    /// Those of `points` that `keep` accepts, by index.
    fn new(points: &[Vec3], size: f32, keep: impl Fn(usize) -> bool) -> Self {
        let mut cells = Self {
            size,
            of: HashMap::new(),
            order: vec![],
        };
        for (i, &p) in points.iter().enumerate() {
            if !keep(i) {
                continue;
            }
            let key = cells.key(p);
            let cell = cells.of.entry(key).or_default();
            if cell.is_empty() {
                cells.order.push(key);
            }
            cell.push(i);
        }
        cells
    }

    fn key(&self, p: Vec3) -> [i32; 3] {
        (p / self.size).floor().as_ivec3().to_array()
    }

    /// The first point of each cell, in the points' order.
    fn firsts(&self) -> impl Iterator<Item = usize> + '_ {
        self.order.iter().map(|key| self.of[key][0])
    }

    /// Into `out`, the points within `radius` of the segment from `a` to `b`: each one's index,
    /// after how far along the segment from `a` it lies (meters).
    fn within(&self, points: &[Vec3], a: Vec3, b: Vec3, radius: f32, out: &mut Vec<(f32, usize)>) {
        out.clear();
        let length = (b - a).length();
        let along = if length > 0.0 {
            (b - a) / length
        } else {
            Vec3::ZERO
        };
        let lower = self.key(a.min(b) - Vec3::splat(radius));
        let upper = self.key(a.max(b) + Vec3::splat(radius));
        // A cell whose middle lies farther from the segment than the radius and its own half
        // diagonal holds none of them.
        let reach = radius + self.size * 3f32.sqrt() / 2.0;
        for x in lower[0]..=upper[0] {
            for y in lower[1]..=upper[1] {
                for z in lower[2]..=upper[2] {
                    let middle = (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * self.size;
                    let t = (middle - a).dot(along).clamp(0.0, length);
                    if (middle - (a + along * t)).length() > reach {
                        continue;
                    }
                    let Some(cell) = self.of.get(&[x, y, z]) else {
                        continue;
                    };
                    for &i in cell {
                        let t = (points[i] - a).dot(along).clamp(0.0, length);
                        if (points[i] - (a + along * t)).length() <= radius {
                            out.push((t, i));
                        }
                    }
                }
            }
        }
    }
}

/// The points near an object, each cell's of a grid together: to ask whether any lies near a
/// segment, as in the way of a sphere swept along it.
struct Near {
    lower: Vec3,
    size: f32,
    cells: [usize; 3],
    /// Where each cell's points start among `points`, and after the last, where they end.
    starts: Vec<u32>,
    points: Vec<Vec3>,
    /// Per point, whether it is the object's.
    object: Vec<bool>,
}

impl Near {
    /// The points of `s` from `lower` to `upper`.
    fn new(s: &Surface, lower: Vec3, upper: Vec3) -> Self {
        let mut size = HAND_CELLS;
        let cells = loop {
            let cells = ((upper - lower) / size).ceil().max(Vec3::ONE);
            if cells.x * cells.y * cells.z <= MOST_CELLS as f32 {
                break cells.as_usizevec3().to_array();
            }
            size *= 2.0;
        };
        let mut near = Self {
            lower,
            size,
            cells,
            starts: vec![0; cells[0] * cells[1] * cells[2] + 1],
            points: vec![],
            object: vec![],
        };
        // Counted per cell, then placed: each cell's points end up together.
        let within = |p: Vec3| p.cmpge(lower).all() && p.cmplt(upper).all();
        let inside: Vec<usize> = (0..s.points.len())
            .filter(|&i| within(s.points[i]))
            .collect();
        for &i in &inside {
            let cell = near.cell(s.points[i]);
            near.starts[cell + 1] += 1;
        }
        for cell in 1..near.starts.len() {
            near.starts[cell] += near.starts[cell - 1];
        }
        let mut next = near.starts.clone();
        near.points = vec![Vec3::ZERO; inside.len()];
        near.object = vec![true; inside.len()];
        for &i in &inside {
            let cell = near.cell(s.points[i]);
            let at = next[cell] as usize;
            next[cell] += 1;
            near.points[at] = s.points[i];
            near.object[at] = s.object.is_empty() || s.object[i];
        }
        near
    }

    /// The cell `p` falls in along each axis, at the nearest where it lies outside them.
    fn along(&self, p: Vec3) -> [usize; 3] {
        let cell = ((p - self.lower) / self.size).floor().max(Vec3::ZERO);
        let last = Vec3::new(
            (self.cells[0] - 1) as f32,
            (self.cells[1] - 1) as f32,
            (self.cells[2] - 1) as f32,
        );
        cell.min(last).as_usizevec3().to_array()
    }

    fn cell(&self, p: Vec3) -> usize {
        let [x, y, z] = self.along(p);
        (z * self.cells[1] + y) * self.cells[0] + x
    }

    /// Whether `hit` holds for a point within `radius` of the segment from `a` to `b`, given the
    /// point and whether it is the object's.
    fn any(&self, a: Vec3, b: Vec3, radius: f32, hit: impl Fn(Vec3, bool) -> bool) -> bool {
        let length = (b - a).length();
        let along = if length > 0.0 {
            (b - a) / length
        } else {
            Vec3::ZERO
        };
        let off = |p: Vec3| {
            let t = (p - a).dot(along).clamp(0.0, length);
            (p - (a + along * t)).length_squared()
        };
        let lower = self.along(a.min(b) - Vec3::splat(radius));
        let upper = self.along(a.max(b) + Vec3::splat(radius));
        // A cell whose middle lies farther from the segment than the radius and its own half
        // diagonal holds none of them.
        let reach = radius + self.size * 3f32.sqrt() / 2.0;
        for z in lower[2]..=upper[2] {
            for y in lower[1]..=upper[1] {
                for x in lower[0]..=upper[0] {
                    let cell = (z * self.cells[1] + y) * self.cells[0] + x;
                    let (start, end) = (self.starts[cell] as usize, self.starts[cell + 1] as usize);
                    if start == end {
                        continue;
                    }
                    let middle =
                        self.lower + (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * self.size;
                    if off(middle) > reach * reach {
                        continue;
                    }
                    for at in start..end {
                        let point = self.points[at];
                        if off(point) <= radius * radius && hit(point, self.object[at]) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Calls `visit` for each point within `radius` of the segment from `a` to `b` slid along
    /// a unit vector square to it (`slides.0`) by anything between two distances (`slides.1`),
    /// until it says to stop: with where the point lies from the segment where it is, the
    /// point, and whether it is the object's.
    fn until(
        &self,
        a: Vec3,
        b: Vec3,
        radius: f32,
        slides: (Vec3, (f32, f32)),
        mut visit: impl FnMut(Vec3, Vec3, bool) -> bool,
    ) {
        let (across, moves) = slides;
        let length = (b - a).length();
        let along = if length > 0.0 {
            (b - a) / length
        } else {
            Vec3::ZERO
        };
        let from = |p: Vec3| {
            let t = (p - a).dot(along).clamp(0.0, length);
            p - (a + along * t)
        };
        // How far `p` lies from the segment moved to where it comes nearest, squared.
        let off = |p: Vec3| {
            let unmoved = from(p);
            let moved = unmoved.dot(across).clamp(moves.0, moves.1);
            (unmoved - across * moved).length_squared()
        };
        let ends = [moves.0, moves.1].map(|moved| across * moved);
        let ends = [a + ends[0], a + ends[1], b + ends[0], b + ends[1]];
        let lowest = ends.into_iter().fold(Vec3::INFINITY, Vec3::min);
        let highest = ends.into_iter().fold(Vec3::NEG_INFINITY, Vec3::max);
        let lower = self.along(lowest - Vec3::splat(radius));
        let upper = self.along(highest + Vec3::splat(radius));
        let reach = radius + self.size * 3f32.sqrt() / 2.0;
        for z in lower[2]..=upper[2] {
            for y in lower[1]..=upper[1] {
                for x in lower[0]..=upper[0] {
                    let cell = (z * self.cells[1] + y) * self.cells[0] + x;
                    let (start, end) = (self.starts[cell] as usize, self.starts[cell + 1] as usize);
                    if start == end {
                        continue;
                    }
                    let middle =
                        self.lower + (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * self.size;
                    if off(middle) > reach * reach {
                        continue;
                    }
                    for at in start..end {
                        let point = self.points[at];
                        // Unmoved too, as `any` finds it, whatever rounding makes of the move.
                        let unmoved = from(point);
                        let within = off(point) <= radius * radius
                            || unmoved.length_squared() <= radius * radius;
                        if within && visit(unmoved, point, self.object[at]) {
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// The space the cameras saw past near an object: for each place points were seen from, an image
/// of the nearest reading in each direction. A ray that reached a reading crossed empty space on
/// its way, and no ray goes into a solid, so a point in front of every reading around it lies
/// outside whatever was seen: as GPD's shadow of what no camera saw, the other way round.
struct Free {
    views: Vec<View>,
}

/// One place's image: cells over the directions to a box, each holding its nearest reading.
struct View {
    origin: Vec3,
    /// From the surface's frame into the image's: x and y across, z toward the box.
    to_view: Mat3,
    /// The directions the cells start at, as x and y over z.
    lower: Vec2,
    /// A cell's width, as x and y over z.
    cell: f32,
    cells: [usize; 2],
    /// Per cell, the distance from the origin of its nearest reading; infinite without one.
    nearest: Vec<f32>,
}

impl Free {
    /// What the points of `s` with a place they were seen from show of the box from `lower` to
    /// `upper`.
    fn new(s: &Surface, lower: Vec3, upper: Vec3) -> Self {
        let mut free = Self { views: vec![] };
        // No place known, or no object.
        if s.seen_from.len() != s.points.len() || !(lower.is_finite() && upper.is_finite()) {
            return free;
        }
        // The points by where they were seen from, in the order those places first come.
        let mut index = HashMap::new();
        let mut places: Vec<(Vec3, Vec<usize>)> = vec![];
        for (i, &from) in s.seen_from.iter().enumerate() {
            if !from.is_finite() {
                continue;
            }
            let place = *index
                .entry(from.to_array().map(f32::to_bits))
                .or_insert_with(|| {
                    places.push((from, vec![]));
                    places.len() - 1
                });
            places[place].1.push(i);
        }
        let views = places.into_iter().filter_map(|(origin, seen)| {
            View::new(origin, seen.iter().map(|&i| s.points[i]), lower, upper)
        });
        free.views.extend(views);
        free
    }

    /// Whether some view shows `p` in front of every reading around it.
    fn seen_past(&self, p: Vec3) -> bool {
        self.views.iter().any(|view| view.seen_past(p))
    }
}

impl View {
    /// The image `seen` gives of the box from `lower` to `upper`, seen from `origin`; `None` where
    /// the box is not all ahead of it or too few readings fall over it.
    fn new(
        origin: Vec3,
        seen: impl Iterator<Item = Vec3>,
        lower: Vec3,
        upper: Vec3,
    ) -> Option<Self> {
        let toward = ((lower + upper) / 2.0 - origin).try_normalize()?;
        let (across, down) = toward.any_orthonormal_pair();
        let to_view = Mat3::from_cols(across, down, toward).transpose();
        let direction = |p: Vec3| {
            let c = to_view * (p - origin);
            (c.z > 0.0).then(|| c.truncate() / c.z)
        };
        let (mut low, mut high) = (Vec2::INFINITY, Vec2::NEG_INFINITY);
        for corner in 0..8 {
            let pick = |bit: usize, a: f32, b: f32| if corner & bit == 0 { a } else { b };
            let at = Vec3::new(
                pick(1, lower.x, upper.x),
                pick(2, lower.y, upper.y),
                pick(4, lower.z, upper.z),
            );
            let at = direction(at)?;
            (low, high) = (low.min(at), high.max(at));
        }
        let over = |d: &Vec2| d.cmpge(low).all() && d.cmple(high).all();
        let readings: Vec<(Vec2, f32)> = seen
            .filter_map(|p| Some((direction(p).filter(over)?, p.distance(origin))))
            .collect();
        let span = high - low;
        let mut cell = WIDER * spacing(&rays(&readings, low, span), low, span)?;
        let cells = loop {
            let cells = (span / cell).ceil().max(Vec2::ONE);
            if cells.x * cells.y <= MOST_CELLS as f32 {
                break [cells.x as usize, cells.y as usize];
            }
            cell *= 2.0;
        };
        let mut view = Self {
            origin,
            to_view,
            lower: low,
            cell,
            cells,
            nearest: vec![f32::INFINITY; cells[0] * cells[1]],
        };
        for (d, range) in readings {
            let at = ((d - low) / cell).floor();
            let x = (at.x as usize).min(cells[0] - 1);
            let y = (at.y as usize).min(cells[1] - 1);
            let nearest = &mut view.nearest[y * cells[0] + x];
            *nearest = nearest.min(range);
        }
        Some(view)
    }

    /// Whether every cell around the one `p` falls in has a reading, each farther than `p` by
    /// more than one surface's noise: the rays around it went on past it.
    fn seen_past(&self, p: Vec3) -> bool {
        let c = self.to_view * (p - self.origin);
        if c.z <= 0.0 {
            return false;
        }
        let at = ((c.truncate() / c.z - self.lower) / self.cell).floor();
        let [width, height] = self.cells.map(|n| n as f32);
        if !(at.x >= 1.0 && at.y >= 1.0 && at.x + 2.0 <= width && at.y + 2.0 <= height) {
            return false;
        }
        let (x, y) = (at.x as usize, at.y as usize);
        let beyond = c.length() + THINNEST;
        (y - 1..=y + 1).all(|row| {
            let cells = &self.nearest[row * self.cells[0] + x - 1..=row * self.cells[0] + x + 1];
            cells
                .iter()
                .all(|&nearest| nearest.is_finite() && nearest > beyond)
        })
    }
}

/// The directions of `readings` (x and y over z, from `low` over `span`), one per ray, in the
/// order they first come: what several looks from one place read along a ray is one direction,
/// to within a cell a 65,536th of the span wide, far finer than any camera's pixels over it and
/// far coarser than rounding.
fn rays(readings: &[(Vec2, f32)], low: Vec2, span: Vec2) -> Vec<Vec2> {
    let fine = span.max_element() / 65536.0;
    if fine.is_nan() || fine <= 0.0 {
        return readings.iter().map(|&(d, _)| d).collect();
    }
    let keys: Vec<u64> = readings
        .iter()
        .map(|&(d, _)| {
            let at = ((d - low) / fine).floor().max(Vec2::ZERO).as_uvec2();
            (u64::from(at.y) << 32) | u64::from(at.x)
        })
        .collect();
    let mut by_key: Vec<u32> = (0..readings.len() as u32).collect();
    by_key.sort_unstable_by_key(|&k| (keys[k as usize], k));
    let mut first = vec![false; readings.len()];
    for (at, &k) in by_key.iter().enumerate() {
        first[k as usize] = at == 0 || keys[by_key[at - 1] as usize] != keys[k as usize];
    }
    let firsts = readings.iter().zip(first).filter(|&(_, first)| first);
    firsts.map(|(&(d, _), _)| d).collect()
}

/// How far apart the directions `rays` (x and y over z, from `low` over `span`) lie from their
/// nearest other than one along the same ray: the median over a sample, sought among cells as
/// wide as they would be apart spread evenly over the span. `None` for fewer than nine.
fn spacing(rays: &[Vec2], low: Vec2, span: Vec2) -> Option<f32> {
    let count = rays.len();
    let even = (span.x * span.y / count as f32).sqrt();
    if count < 9 || even.is_nan() || even <= 0.0 {
        return None;
    }
    let [width, height] = (span / even)
        .ceil()
        .max(Vec2::ONE)
        .as_uvec2()
        .to_array()
        .map(|n| n as usize);
    let cell = |d: Vec2| {
        let at = ((d - low) / even).floor().max(Vec2::ZERO).as_uvec2();
        [
            (at.x as usize).min(width - 1),
            (at.y as usize).min(height - 1),
        ]
    };
    // Counted per cell, then placed: each cell's rays together.
    let mut starts = vec![0u32; width * height + 1];
    for &d in rays {
        let [x, y] = cell(d);
        starts[y * width + x + 1] += 1;
    }
    for k in 1..starts.len() {
        starts[k] += starts[k - 1];
    }
    let mut next = starts.clone();
    let mut order = vec![0u32; count];
    for (k, &d) in rays.iter().enumerate() {
        let [x, y] = cell(d);
        order[next[y * width + x] as usize] = k as u32;
        next[y * width + x] += 1;
    }
    // A ray's own rays from several looks that fell across the edge of a cell of `rays`
    // remain: nearer than a hundredth of the even spacing, a direction is the same ray's, and
    // the nearest is sought past those, ring by ring of cells, out to where none nearer can lie.
    let twin = (even * 1e-2).powi(2);
    let nearest = |k: usize| {
        let (d, [x, y]) = (rays[k], cell(rays[k]));
        let along = |row: usize, from: usize, to: usize| {
            let cells = starts[row * width + from] as usize..starts[row * width + to + 1] as usize;
            let apart = order[cells]
                .iter()
                .map(|&j| rays[j as usize].distance_squared(d));
            apart
                .filter(|&apart| apart > twin)
                .fold(f32::INFINITY, f32::min)
        };
        let mut best = f32::INFINITY;
        for ring in 0..=RINGS {
            let (left, right) = (x.saturating_sub(ring), (x + ring).min(width - 1));
            for row in y.saturating_sub(ring)..=(y + ring).min(height - 1) {
                if row.abs_diff(y) == ring {
                    best = best.min(along(row, left, right));
                } else {
                    if x >= ring {
                        best = best.min(along(row, x - ring, x - ring));
                    }
                    if x + ring < width {
                        best = best.min(along(row, x + ring, x + ring));
                    }
                }
            }
            // Past this ring every reading lies at least `ring` cells away.
            if best <= (ring as f32 * even).powi(2) {
                break;
            }
        }
        best.is_finite().then(|| best.sqrt())
    };
    let sample = (0..count).step_by((count / SAMPLES).max(1));
    let mut apart: Vec<f32> = sample.filter_map(nearest).collect();
    if apart.is_empty() {
        return Some(even);
    }
    let middle = apart.len() / 2;
    Some(*apart.select_nth_unstable_by(middle, f32::total_cmp).1)
}

impl Grasp {
    /// The moment about `screw`'s line that the grasp's two fingertips can give what they hold,
    /// for each newton each presses with (meters): what turning it about the line asks of the
    /// grip, to order grips for a [`turn`](crate::turn()) by, the most first. It reads the
    /// options' `friction` and `contact`; the screw's direction must be unit, as
    /// [`Screw::path`] checks, or the moment is no number.
    ///
    /// A force `F` at a contact `c` gives `F · (ŝ × (c − q))` about the line through `q` along
    /// `ŝ`, and each fingertip presses along the closing line `n`. Friction lets it push across
    /// `n` by up to `friction` times its press, which gives `friction · |ŝ × (c − q)|` less its
    /// part along `n`; the press may sit anywhere on its patch, `contact` in radius, which gives
    /// `contact · |n × ŝ|`; and a soft fingertip twists about `n` by up to `2/3 · friction ·
    /// contact` times its press (Murray, Li and Sastry's soft-finger contact, *A Mathematical
    /// Introduction to Robotic Manipulation*, 1994, §5.2.1, with the press even over the patch),
    /// which gives that times `|n · ŝ|`. The two presses cancel while they are equal, but a
    /// parallel gripper's fingers close evenly about its frame, so the hand can put all the
    /// closing force on either finger and push the thing along `n`: up to
    /// `2 |n · (ŝ × (m − q))|` more, `m` the middle between the contacts, and friction's part
    /// changes by what the loaded finger's lever gains over the unloaded one's, more where the
    /// loaded one is the farther. The moment is the most these give together, the way round the
    /// line that gives less. The patch's term puts the press at its edge and the twist's spreads
    /// it evenly: they are separate bounds, exact together where the line is square to `n`, where
    /// the twist gives nothing. A bar 2.5 cm wide gripped across it on its axis gives 2.25 cm at
    /// the default options, half of it friction; 3 cm along the bar from its axis, 8.25 cm.
    #[must_use]
    pub fn moment(&self, screw: &Screw, o: &GraspOptions) -> f32 {
        let along = screw.direction.normalize();
        let [first, second] = self.contacts;
        let closing = (second - first)
            .try_normalize()
            .unwrap_or_else(|| self.pose.rotation * Vec3::Y);
        // How far a push across the closing line at `c` acts from the screw's line.
        let lever = |c: Vec3| {
            let arm = along.cross(c - screw.point);
            (arm - closing * arm.dot(closing)).length()
        };
        let (near, far) = (lever(first), lever(second));
        let twist = 2.0 / 3.0 * o.friction * o.contact;
        let patches = o.contact * closing.cross(along).length() + twist * closing.dot(along).abs();
        let held = o.friction * (near + far) + 2.0 * patches;
        let middle = (first + second) / 2.0;
        let pushed = 2.0 * closing.dot(along.cross(middle - screw.point)).abs();
        held + (pushed - o.friction * (near - far).abs()).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::{Robot, RobotOptions};
    use crate::sdf::Intrinsics;

    #[test]
    fn a_view_shows_nothing_inside_a_solid_as_seen_past() {
        let robot = Robot::load("assets/franka/franka_panda.urdf", &RobotOptions::default());
        let device = Device::cpu(&robot.unwrap()).unwrap();
        // A slab 10 cm thick over x < edge seen from 50 cm straight above, the floor past its
        // edge: across the edge, at a twentieth of a pixel's steps and edges set across a pixel.
        let k = Intrinsics {
            fx: 300.0,
            fy: 300.0,
            cx: 160.0,
            cy: 120.0,
        };
        let down = Mat3::from_cols(Vec3::X, Vec3::NEG_Y, Vec3::NEG_Z);
        let camera = Pose::new(Vec3::Z * 0.5, Quat::from_mat3(&down));
        let (mut inside, mut past) = (0, 0);
        for shift in 0..12 {
            let edge = 0.0007 + shift as f32 * 0.0001;
            let depth: Vec<f32> = (0..320 * 240)
                .map(|i| {
                    let x = 0.4 * ((i % 320) as f32 - k.cx) / k.fx;
                    if x < edge { 0.4 } else { 0.5 }
                })
                .collect();
            let image = DepthImage {
                depth: &depth,
                width: 320,
                intrinsics: k,
                camera,
            };
            let seen = Surface::from_depth(&device, &image, &[], None, 0.005).unwrap();
            let free = Free::new(
                &seen,
                Vec3::new(-0.03, -0.03, 0.0),
                Vec3::new(0.03, 0.03, 0.12),
            );
            assert_eq!(free.views.len(), 1);
            for step in 0..120 {
                let x = edge - 0.006 + step as f32 * 0.00005;
                for y in [-0.0101, 0.0, 0.0037] {
                    // Inside the slab, under its top, nothing is seen past.
                    if x < edge {
                        inside += 1;
                        let under = Vec3::new(x, y, 0.099);
                        assert!(!free.seen_past(under), "{under} in the slab, edge {edge}");
                    }
                    // Nor within one surface's noise in front of its top.
                    let over = Vec3::new(x.min(edge - 0.004), y, 0.101);
                    assert!(!free.seen_past(over), "{over} on the slab");
                }
            }
            // In front of it and past its edge, the rays went on.
            for p in [
                Vec3::new(-0.01, 0.0, 0.11),
                Vec3::new(edge + 0.006, 0.0, 0.05),
            ] {
                assert!(free.seen_past(p), "{p}");
                past += 1;
            }
        }
        assert!(inside > 1000 && past == 24);
    }
}
