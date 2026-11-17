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
//! From each seed point the search goes in along the normal, through every surface that faces the
//! other way within the opening: a cup seen from outside gives its wall, grasped with one finger
//! inside, and its whole width, grasped around it. Patches are a fingertip's size
//! ([`GraspOptions::contact`]), each the mean of the points in it, so noise in single points and
//! their normals averages out.

use std::collections::{HashMap, HashSet};
use std::f32::consts::TAU;

use glam::{Mat3, Quat, Vec3};
use parry3d::utils::SymmetricEigen3;
use rayon::prelude::*;

use crate::device::Device;
use crate::error::{Result, ensure_input};
use crate::robot::HandSphere;
use crate::sdf::{DepthImage, RobotMask};
use crate::task::GraspOptions;
use crate::types::Pose;

/// Points on the surfaces seen and the normals out of them, in one frame: what [`grasps_on`]
/// finds grasps on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Surface {
    /// Points on surfaces (meters).
    pub points: Vec<Vec3>,
    /// Per point, the unit normal out of its surface, toward where it was seen from, or zero
    /// where how the surface faces is not known: no finger presses there, and the point is kept
    /// clear of or closed on like the rest. Only the object's are read.
    pub normals: Vec<Vec3>,
    /// Per point, whether it lies on the object to grasp: grasps close on the object's points
    /// alone, and nothing else may lie between the fingers. Empty: every point does.
    pub object: Vec<bool>,
}

impl Surface {
    /// What one depth image sees: a point for each reading off the robot, and for each that
    /// shows the object (`object`, a flag per pixel; empty: every pixel does), the normal of
    /// the plane through the readings within `radius` of it (meters; about a fingertip's patch,
    /// 5 mm), toward the camera. A reading whose neighbors within `radius` are too few or lie on
    /// one line, as along an edge seen end on, has no normal. Two views of one scene make one
    /// surface with [`Surface::join`].
    pub fn from_depth(
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
        let seen: Vec<Option<Vec3>> = (0..image.depth.len())
            .map(|i| image.point(i, robot, 0.0))
            .collect();
        let seen = &seen;
        let mut surface = Surface::default();
        for (i, &at) in seen.iter().enumerate() {
            let Some(at) = at else {
                continue;
            };
            let shows = object.is_empty() || object[i];
            let mut normal = Vec3::ZERO;
            if shows {
                // The pixels a patch of `radius` covers at this depth, seen square on.
                let depth = image.depth[i];
                let reach = (radius * image.intrinsics.fx.max(image.intrinsics.fy) / depth).ceil();
                let reach = (reach as usize).clamp(1, WINDOW);
                let (u, v) = (i % width, i / width);
                let rows = v.saturating_sub(reach)..=(v + reach).min(height - 1);
                let near = rows.flat_map(|row| {
                    let columns = u.saturating_sub(reach)..=(u + reach).min(width - 1);
                    columns.filter_map(move |column| seen[row * width + column])
                });
                let near = near.filter(|p| p.distance_squared(at) <= radius * radius);
                normal = plane(near, at, image.camera.position).unwrap_or(Vec3::ZERO);
            }
            surface.points.push(at);
            surface.normals.push(normal);
            if !object.is_empty() {
                surface.object.push(shows);
            }
        }
        Ok(surface)
    }

    /// The surface `points` lie on, all of them the object's, as seen from `seen_from`: each
    /// with the normal of the plane through the points within `radius` of it (meters), toward
    /// `seen_from`, or none where they are too few or lie on one line.
    pub fn from_points(points: &[Vec3], seen_from: Vec3, radius: f32) -> Result<Self> {
        ensure_input!(
            points.iter().all(|p| p.is_finite()) && seen_from.is_finite(),
            "points and where they were seen from must be finite"
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
        })
    }

    /// Adds `other`'s points after this surface's: two views of one scene as one surface.
    pub fn join(&mut self, other: Surface) {
        if !(self.object.is_empty() && other.object.is_empty()) {
            self.object.resize(self.points.len(), true);
            self.object.extend(&other.object);
        }
        self.points.extend(other.points);
        self.normals.extend(other.normals);
        if !self.object.is_empty() {
            self.object.resize(self.points.len(), true);
        }
    }
}

/// The most pixels to either side of a reading its plane is sought among.
const WINDOW: usize = 6;

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
}

impl Grasp {
    /// How far apart the two points it closes on are (meters): what the fingers close to.
    #[must_use]
    pub fn width(&self) -> f32 {
        (self.contacts[1] - self.contacts[0]).length()
    }
}

/// The fewest points a patch a fingertip presses on holds: fewer say too little of where the
/// surface is and how it faces.
const SUPPORT: usize = 3;

/// The nearest two patches may be (meters): closer, they are one surface's noise, not two sides.
const THINNEST: f32 = 0.0025;

/// How far a hand that is not clear is drawn back at a time (meters).
const BACK: f32 = 0.002;

/// The cells the points near the hand are kept in (meters): a few to a sphere.
const HAND_CELLS: f32 = 0.01;

/// The most cells they are kept in: past it the cells are larger.
const MOST_CELLS: usize = 1 << 22;

/// Grasps on each of `surfaces` for hand `hand` of `device`'s robot, the best first: those whose
/// patches face each other most squarely. Each pair of patches is grasped from `turns` directions
/// around its closing line, the first from above where the line is not upright, each both ways
/// round.
pub fn grasps_on(
    device: &Device,
    hand: usize,
    surfaces: &[Surface],
    o: &GraspOptions,
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
            && o.approach.is_finite(),
        "grasps need a finite positive opening, a turn, a finite positive friction and contact, and a finite approach of at least 0, got {} m, {}, {}, {} m and {} m",
        o.opening,
        o.turns,
        o.friction,
        o.contact,
        o.approach
    );
    for (i, s) in surfaces.iter().enumerate() {
        ensure_input!(
            s.normals.len() == s.points.len()
                && (s.object.is_empty() || s.object.len() == s.points.len()),
            "surface {i}: {} points with {} normals and {} object flags",
            s.points.len(),
            s.normals.len(),
            s.object.len()
        );
        let unit = |n: &Vec3| *n == Vec3::ZERO || (n.length_squared() - 1.0).abs() < 1e-3;
        ensure_input!(
            s.points.iter().all(|p| p.is_finite()) && s.normals.iter().all(unit),
            "surface {i}: points must be finite and normals unit or zero"
        );
    }
    // The fingers first: they are the likeliest to be in the way.
    let mut hand = device.robot().hand_spheres(hand);
    hand.sort_by_key(|sphere| !sphere.finger);
    let each = || surfaces.par_iter().map(|s| on(s, &hand, o)).collect();
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
}

/// One surface's grasps for a hand of spheres `hand`.
fn on(s: &Surface, hand: &[HandSphere], o: &GraspOptions) -> Vec<Grasp> {
    let pairs = pairs(s, o);
    if pairs.is_empty() {
        return vec![];
    }
    // How far the hand reaches past its frame, and how far from it any of it comes, its way in
    // included.
    let reach = hand
        .iter()
        .map(|sphere| sphere.center.z + sphere.radius)
        .fold(0.0, f32::max);
    let extent = hand
        .iter()
        .map(|sphere| sphere.center.length() + sphere.radius)
        .fold(0.0, f32::max)
        + o.approach
        + reach;
    let middles = pairs.iter().map(|pair| pair.middle);
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
            let tips = (pose.position + toward * reach).z - closing.z.abs() * o.opening / 2.0;
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
    let of = |pair: &Pair| {
        let mut grasps = Vec::with_capacity(2 * o.turns);
        // From above first, where the line is not upright.
        let down = Vec3::NEG_Z.reject_from_normalized(pair.line);
        let first = down
            .try_normalize()
            .unwrap_or_else(|| pair.line.any_orthonormal_vector());
        for turn in 0..o.turns {
            let angle = TAU * turn as f32 / o.turns as f32;
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
                        });
                        break;
                    }
                    back += BACK;
                }
            }
        }
        grasps
    };
    let each: Vec<Vec<Grasp>> = pairs.par_iter().map(of).collect();
    each.into_iter().flatten().collect()
}

/// The pairs of patches of the object's surfaces a hand can close on, those that face each other
/// most squarely first.
fn pairs(s: &Surface, o: &GraspOptions) -> Vec<Pair> {
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
    // The pairs a seed's patch makes with those across from it, each with the cells its two
    // patches fall in.
    let across = |near: &mut Vec<(f32, usize)>, seed: usize| {
        let mut pairs = vec![];
        let (p, n) = (s.points[seed], s.normals[seed]);
        // The patch around the seed: the object's points there that face as it does.
        cells.within(&s.points, p, p, o.contact, near);
        near.retain(|&(_, i)| object(i) && s.normals[i].dot(n) >= cone);
        let Some((first, out)) = patch(s, near) else {
            return pairs;
        };
        let inward = -out;
        for (second, facing) in crossed(first, inward, apart, near) {
            // Two patches hold along the line that splits the angle between their normals,
            // which is inside the first's cone, the two being no more than two cones apart.
            // Where it is not the line searched along, as between a wedge's faces, the patch it
            // crosses is the one to close on.
            let split = (inward + facing).normalize();
            let (second, facing) = if split.dot(inward) < 1.0 - 1e-4 {
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
            // Anything else between the two patches would be between the fingers.
            cells.within(&s.points, first, second, o.contact, near);
            let between =
                |&(t, i): &(f32, usize)| !object(i) && t > o.contact && t < width - o.contact;
            if near.iter().any(between) {
                continue;
            }
            let pair = Pair {
                middle: (first + second) / 2.0,
                line,
                width,
                square,
            };
            pairs.push((pair, cells.key(first), cells.key(second)));
        }
        pairs
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
    for (pair, a, b) in each.into_iter().flatten() {
        if found.insert((a.min(b), a.max(b))) {
            pairs.push(pair);
        }
    }
    pairs.sort_by(|a, b| b.square.total_cmp(&a.square));
    pairs
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
}
