//! The object [`rests_on`](super::rests_on) sets down: a convex shape about its middle, and the
//! same shape turned, read from above as the set-down pass reads it.

use glam::{Mat3, Quat, Vec2, Vec3, Vec3A};

use crate::world::{self, Obstacle};

/// A convex shape whose middle is its frame's origin.
#[derive(Clone, Copy, Debug)]
pub(super) enum Shape {
    /// Half its size along each axis.
    Cuboid(Vec3),
    Sphere(f32),
    /// Along its z axis: the radius, and half its length.
    Cylinder(f32, f32),
    /// Around the segment along its z axis: the radius, and half the segment's length.
    Capsule(f32, f32),
}

impl Shape {
    pub(super) fn of(object: &Obstacle) -> Self {
        match *object {
            Obstacle::Cuboid { half_extents, .. } => Shape::Cuboid(half_extents),
            Obstacle::Sphere { radius, .. } => Shape::Sphere(radius),
            Obstacle::Cylinder {
                radius,
                half_height,
                ..
            } => Shape::Cylinder(radius, half_height),
            Obstacle::Capsule {
                radius,
                half_length,
                ..
            } => Shape::Capsule(radius, half_length),
            Obstacle::Sdf { .. } => unreachable!("distance grids are refused before"),
        }
    }

    /// The shape grown by `by` all round; a box and a cylinder to the box and the cylinder that
    /// hold that.
    pub(super) fn grown(self, by: f32) -> Self {
        match self {
            Shape::Cuboid(half) => Shape::Cuboid(half + by),
            Shape::Sphere(r) => Shape::Sphere(r + by),
            Shape::Cylinder(r, h) => Shape::Cylinder(r + by, h + by),
            Shape::Capsule(r, h) => Shape::Capsule(r + by, h),
        }
    }

    /// The radius of its round side.
    pub(super) fn round(self) -> f32 {
        match self {
            Shape::Cuboid(_) => 0.0,
            Shape::Sphere(r) | Shape::Cylinder(r, _) | Shape::Capsule(r, _) => r,
        }
    }

    /// The shape shrunk by `by` all round, or by half its least size where that is less, so
    /// that it stays a solid: a box and a cylinder to the box and the cylinder that it holds.
    pub(super) fn shrunk(self, by: f32) -> Self {
        let least = match self {
            Shape::Cuboid(half) => half.min_element(),
            Shape::Sphere(r) | Shape::Capsule(r, _) => r,
            Shape::Cylinder(r, h) => r.min(h),
        };
        let by = by.min(0.5 * least);
        match self {
            Shape::Cuboid(half) => Shape::Cuboid(half - by),
            Shape::Sphere(r) => Shape::Sphere(r - by),
            Shape::Cylinder(r, h) => Shape::Cylinder(r - by, h - by),
            Shape::Capsule(r, h) => Shape::Capsule(r - by, h),
        }
    }

    /// How far from its middle any of it lies.
    pub(super) fn radius(self) -> f32 {
        match self {
            Shape::Cuboid(half) => half.length(),
            Shape::Sphere(r) => r,
            Shape::Cylinder(r, h) => r.hypot(h),
            Shape::Capsule(r, h) => r + h,
        }
    }

    /// Whether turning it by `turn`, in its own frame, leaves it where it was, to the rounding
    /// of a turn put together from others.
    pub(super) fn same_turned(self, turn: Quat) -> bool {
        let axes = Mat3::from_quat(turn);
        let along = |a: Vec3, b: Vec3| a.dot(b).abs() > 1.0 - 1e-6;
        match self {
            Shape::Sphere(_) => true,
            Shape::Cylinder(..) | Shape::Capsule(..) => along(axes.z_axis, Vec3::Z),
            // Each axis onto an axis of the same length.
            Shape::Cuboid(half) => (0..3).all(|i| {
                (0..3).any(|j| {
                    along(axes.col(i), Vec3::AXES[j])
                        && (half[i] - half[j]).abs() <= 1e-6 * half.max_element()
                })
            }),
        }
    }

    /// Along the line through `from` that runs along the unit `along`, both in the shape's
    /// frame, from how far to how far it lies inside the shape; none where it misses it.
    #[inline]
    fn span(self, from: Vec3, along: Vec3) -> Option<(f32, f32)> {
        match self {
            Shape::Cuboid(half) => {
                let x = within(from.x, along.x, half.x);
                let y = within(from.y, along.y, half.y);
                meet(meet(x, y), within(from.z, along.z, half.z))
            }
            Shape::Sphere(r) => ball(from, along, r),
            Shape::Cylinder(r, h) => meet(round(from, along, r), within(from.z, along.z, h)),
            Shape::Capsule(r, h) => {
                let side = meet(round(from, along, r), within(from.z, along.z, h));
                let ends = [
                    ball(from - Vec3::Z * h, along, r),
                    ball(from + Vec3::Z * h, along, r),
                ];
                let spans = ends.into_iter().chain([side]).flatten();
                spans.reduce(|(a0, a1), (b0, b1)| (a0.min(b0), a1.max(b1)))
            }
        }
    }
}

/// The `t` for which `from + t along` lies within `half` of 0: every `t` where `along` is 0
/// and `from` within, none where it is outside.
#[inline]
fn within(from: f32, along: f32, half: f32) -> Option<(f32, f32)> {
    if along.abs() < 1e-9 {
        return (from.abs() <= half).then_some((f32::NEG_INFINITY, f32::INFINITY));
    }
    let (a, b) = ((-half - from) / along, (half - from) / along);
    Some((a.min(b), a.max(b)))
}

/// Where two spans overlap.
#[inline]
fn meet(a: Option<(f32, f32)>, b: Option<(f32, f32)>) -> Option<(f32, f32)> {
    let ((a0, a1), (b0, b1)) = (a?, b?);
    let (lo, hi) = (a0.max(b0), a1.min(b1));
    (lo <= hi).then_some((lo, hi))
}

/// The span of a line, from `from` along the unit `along`, inside a ball of radius `r` at the
/// origin.
#[inline]
fn ball(from: Vec3, along: Vec3, r: f32) -> Option<(f32, f32)> {
    let b = from.dot(along);
    let reach = b * b - (from.length_squared() - r * r);
    (reach >= 0.0).then(|| (-b - reach.sqrt(), -b + reach.sqrt()))
}

/// The span of a line inside the endless cylinder of radius `r` around the z axis.
#[inline]
fn round(from: Vec3, along: Vec3, r: f32) -> Option<(f32, f32)> {
    let (from, along) = (from.truncate(), along.truncate());
    let a = along.length_squared();
    let c = from.length_squared() - r * r;
    if a < 1e-12 {
        return (c <= 0.0).then_some((f32::NEG_INFINITY, f32::INFINITY));
    }
    let b = from.dot(along);
    let reach = b * b - a * c;
    (reach >= 0.0).then(|| ((-b - reach.sqrt()) / a, (-b + reach.sqrt()) / a))
}

/// A flat piece of a shape's surface, by number, or [`ROUND`]: what lies on one flat piece pushes
/// along one direction.
pub(super) type Piece = u8;

/// A piece of a shape's surface that is not flat.
pub(super) const ROUND: Piece = 0;

/// A shape turned, as the set-down pass reads it from above: its underside is a height over each
/// point of its footprint.
#[derive(Clone, Copy, Debug)]
pub(super) struct Turned {
    pub(super) shape: Shape,
    /// The shape's axes in the surface's frame.
    pub(super) frame: Mat3,
    /// The surface's axes in the shape's frame.
    x: Vec3,
    y: Vec3,
    up: Vec3,
    /// A box read from above without a division: half its size, and one over how far the
    /// surface's upward axis runs along each of its own, an axis it does not run along taken as
    /// one it barely does.
    half: Vec3A,
    over_up: Vec3A,
    /// How far its footprint reaches from its middle along the surface's x and y axes, and how
    /// far under its middle its underside reaches.
    pub(super) reach: Vec2,
    pub(super) depth: f32,
}

impl Turned {
    pub(super) fn new(shape: Shape, rotation: Quat) -> Self {
        let frame = world::frame(rotation);
        let to_shape = frame.transpose();
        let reach = |axis: usize| match shape {
            Shape::Cuboid(half) => (0..3).map(|j| frame.col(j)[axis].abs() * half[j]).sum(),
            Shape::Sphere(r) => r,
            Shape::Cylinder(r, h) => {
                let along = frame.z_axis[axis];
                h * along.abs() + r * (1.0 - along * along).max(0.0).sqrt()
            }
            Shape::Capsule(r, h) => h * frame.z_axis[axis].abs() + r,
        };
        let up = to_shape.z_axis;
        let over = |v: f32| {
            1.0 / if v.abs() < 1e-9 {
                1e-9f32.copysign(v)
            } else {
                v
            }
        };
        let half = match shape {
            Shape::Cuboid(half) => half,
            _ => Vec3::ZERO,
        };
        Self {
            shape,
            frame,
            x: to_shape.x_axis,
            y: to_shape.y_axis,
            up,
            half: half.into(),
            over_up: Vec3A::new(over(up.x), over(up.y), over(up.z)),
            reach: Vec2::new(reach(0), reach(1)),
            depth: reach(2),
        }
    }

    /// Over the point `offset` from its middle, seen from above: the heights of its underside and
    /// of its top, from its middle; none where it does not reach over that point.
    #[inline]
    pub(super) fn span(&self, offset: Vec2) -> Option<(f32, f32)> {
        let from = self.x * offset.x + self.y * offset.y;
        if let Shape::Cuboid(_) = self.shape {
            // Between each pair of faces, from where to where; inside all three at once.
            let from = Vec3A::from(from);
            let (a, b) = (
                (-self.half - from) * self.over_up,
                (self.half - from) * self.over_up,
            );
            let (enters, leaves) = (a.min(b).max_element(), a.max(b).min_element());
            return (enters <= leaves).then_some((enters, leaves));
        }
        self.shape.span(from, self.up)
    }

    /// The point of its underside `lower` from its middle over the point `offset` from it, in its
    /// own frame.
    #[inline]
    pub(super) fn local(&self, offset: Vec2, lower: f32) -> Vec3 {
        self.x * offset.x + self.y * offset.y + self.up * lower
    }

    /// The piece of its surface the point `at` of it (in its own frame) lies on, and the unit
    /// direction out of it there, in the surface's frame.
    pub(super) fn piece_at(&self, at: Vec3) -> (Piece, Vec3) {
        match self.shape {
            Shape::Cuboid(half) => {
                // The face it is on; on an edge, to a hundredth of a millimeter, the face of
                // the two that faces down the most.
                let past = at.abs() - half;
                let nearest = past.max_element();
                let mut found = (ROUND, Vec3::ZERO);
                for axis in 0..3 {
                    let positive = at[axis] >= 0.0;
                    let out = self.frame.col(axis) * if positive { 1.0 } else { -1.0 };
                    if past[axis] >= nearest - 1e-5 && (found.0 == ROUND || out.z < found.1.z) {
                        found = (1 + 2 * axis as Piece + Piece::from(positive), out);
                    }
                }
                found
            }
            Shape::Sphere(_) => (ROUND, self.frame * at.normalize_or(Vec3::NEG_Z)),
            Shape::Cylinder(r, h) => {
                let across = at.truncate().length();
                let positive = at.z >= 0.0;
                let end = self.frame.z_axis * if positive { 1.0 } else { -1.0 };
                let side = self.frame * (at.truncate() / across.max(1e-9)).extend(0.0);
                // On the rim, to a hundredth of a millimeter, whichever faces down the most.
                let (on_end, on_side) = (at.z.abs() - h, across - r);
                let nearest = on_end.max(on_side);
                if on_end >= nearest - 1e-5 && (on_side < nearest - 1e-5 || end.z <= side.z) {
                    (1 + Piece::from(positive), end)
                } else {
                    (ROUND, side)
                }
            }
            Shape::Capsule(_, h) => {
                let from = at - Vec3::Z * at.z.clamp(-h, h);
                (ROUND, self.frame * from.normalize_or(Vec3::NEG_Z))
            }
        }
    }

    /// The point of it nearest `from`, both from its middle in the surface's frame, where `from`
    /// is outside it.
    pub(super) fn nearest(&self, from: Vec3) -> Vec3 {
        let at = self.frame.transpose() * from;
        let nearest = match self.shape {
            Shape::Cuboid(half) => at.clamp(-half, half),
            Shape::Sphere(r) => at.normalize_or(Vec3::Z) * r,
            Shape::Cylinder(r, h) => {
                let across = at.truncate().clamp_length_max(r);
                across.extend(at.z.clamp(-h, h))
            }
            Shape::Capsule(r, h) => {
                let on = Vec3::Z * at.z.clamp(-h, h);
                on + (at - on).normalize_or(Vec3::Z) * r
            }
        };
        self.frame * nearest
    }

    /// Where `at`, a point of it in its own frame, lies on a round side that runs straight along
    /// its axis, as a can's: the side's radius and half its length. None on a ball, or on the
    /// round end of a capsule.
    pub(super) fn straight(&self, at: Vec3) -> Option<(f32, f32)> {
        match self.shape {
            Shape::Cylinder(r, h) => Some((r, h)),
            Shape::Capsule(r, h) if at.z.abs() < h => Some((r, h)),
            _ => None,
        }
    }

    /// The piece of it that faces the unit direction `toward` (in the shape's frame) the most,
    /// and how squarely the flat `piece` faces it less than that one does: nought for that piece
    /// itself.
    pub(super) fn facing(&self, piece: Piece, toward: Vec3) -> (Piece, f32) {
        match self.shape {
            Shape::Cuboid(_) => {
                let axis = (0..3)
                    .max_by(|&a, &b| toward[a].abs().total_cmp(&toward[b].abs()))
                    .expect("three axes");
                let most = 1 + 2 * axis as Piece + Piece::from(toward[axis] >= 0.0);
                let own = usize::from((piece - 1) / 2);
                let out = if (piece - 1) % 2 == 1 { 1.0 } else { -1.0 };
                (most, toward[axis].abs() - toward[own] * out)
            }
            Shape::Cylinder(..) => {
                let (end, side) = (toward.z.abs(), toward.truncate().length());
                let most = if end >= side {
                    1 + Piece::from(toward.z >= 0.0)
                } else {
                    ROUND
                };
                let own = toward.z * if piece == 2 { 1.0 } else { -1.0 };
                (most, end.max(side) - own)
            }
            Shape::Sphere(_) | Shape::Capsule(..) => unreachable!("neither has a flat piece"),
        }
    }

    /// What of the flat `piece` meets a plane it is set on, where the plane lies the unit `toward` (in the
    /// shape's frame) from it: the whole piece where it stays within `near` of the plane from
    /// `lower` to `upper`, the corners of the part of it that is near the plane; else the edge,
    /// the line or the point of it that reaches farthest toward the plane. A piece that stands
    /// `beside` the one that faces the plane meets it along an edge at most.
    pub(super) fn meets(
        &self,
        piece: Piece,
        toward: Vec3,
        [lower, upper]: [Vec3; 2],
        near: f32,
        beside: bool,
    ) -> Meets {
        let extent = upper - lower;
        // Whether the piece stays near the plane all along axis `j` of it.
        let level = |j: usize| toward[j].abs() * extent[j] < near;
        let side = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
        match self.shape {
            Shape::Cuboid(half) => {
                let axis = usize::from((piece - 1) / 2);
                let mut at = Vec3::ZERO;
                at[axis] = if (piece - 1) % 2 == 1 {
                    half[axis]
                } else {
                    -half[axis]
                };
                let (j, k) = ((axis + 1) % 3, (axis + 2) % 3);
                // Beside the facing piece it meets the plane along the edge they share: the one
                // it rises from.
                let along_j = if beside {
                    toward[j].abs() < toward[k].abs()
                } else {
                    match (level(j), level(k)) {
                        (true, true) => return Meets::All,
                        (false, false) => {
                            at[j] = half[j] * side(toward[j]);
                            at[k] = half[k] * side(toward[k]);
                            return Meets::Point(at);
                        }
                        (along_j, _) => along_j,
                    }
                };
                let (free, fixed) = if along_j { (j, k) } else { (k, j) };
                at[fixed] = half[fixed] * side(toward[fixed]);
                Meets::Line {
                    at,
                    axis: free,
                    half: half[free],
                }
            }
            Shape::Cylinder(r, h) => {
                // An end: all of it, or the point of its rim nearest the plane.
                let across = toward.truncate();
                let lean = across.length();
                let over = extent.truncate().length();
                if lean * over < near && !beside {
                    return Meets::All;
                }
                let end = if piece == 2 { h } else { -h };
                Meets::Point((across / lean * r).extend(end))
            }
            Shape::Sphere(_) | Shape::Capsule(..) => unreachable!("neither has a flat piece"),
        }
    }
}

/// What of a piece of a shape meets a plane the shape is set on ([`Turned::meets`]), in the
/// shape's frame.
#[derive(Clone, Copy, Debug)]
pub(super) enum Meets {
    /// The piece lies on the plane.
    All,
    /// A line of it does, through `at` along the shape's axis `axis`, `half` to either side.
    Line { at: Vec3, axis: usize, half: f32 },
    /// One point of it does.
    Point(Vec3),
}
