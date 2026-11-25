//! An object's shape from its readings: seen from above, the smaller of the least rectangle and
//! the least circle around them, over the heights they span, and a box open at the top where they
//! hold a hollow's floor; and what lies straight under a point.

use std::f32::consts::PI;

use glam::{Quat, Vec2, Vec3};

use crate::scene::Shape;

/// One of an object's readings, as the room of a box open at the top is measured from it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reading {
    pub(crate) point: Vec3,
    /// The unit normal of its plane, toward the camera; zero where it has none.
    pub(crate) normal: Vec3,
    /// How far off a surface it may read and be on it (meters).
    pub(crate) slack: f32,
    /// The share it is of a window at its depth.
    pub(crate) share: f32,
    /// Whether it is of the floor of the object's hollow.
    pub(crate) floor: bool,
    /// How far apart the readings of two neighbouring pixels lie across at its depth (meters).
    pub(crate) spacing: f32,
}

/// An object's outline seen from above.
#[derive(Clone, Copy, Debug)]
enum Outline {
    /// A circle.
    Round { center: Vec2, radius: f32 },
    /// A rectangle turned by `yaw` about z, `half` along its own axes.
    Square { center: Vec2, half: Vec2, yaw: f32 },
}

/// The shape fitted to an object's readings: its outline from above, between two heights, and
/// the room of its hollow where it has one ([`Fit::open`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Fit {
    outline: Outline,
    top: f32,
    /// The height of its lowest reading, or of what it stands on ([`Fit::stand_on`]).
    bottom: f32,
    /// Its lowest reading.
    pub(crate) lowest: Vec3,
    /// The room of its hollow, in the frame of its square outline (along x and y from its middle)
    /// but for the floor's top, which is a height.
    room: Option<(Vec3, Vec2)>,
}

impl Fit {
    /// The fit of `points`, which must hold one at least: round where the least circle around
    /// them, seen from above, is smaller than the least rectangle, as a disc's is and a square's
    /// is not.
    pub(crate) fn of(points: &[Vec3]) -> Self {
        let lowest = points
            .iter()
            .copied()
            .min_by(|a, b| a.z.total_cmp(&b.z))
            .expect("an object has a reading");
        let top = points.iter().map(|p| p.z).fold(f32::NEG_INFINITY, f32::max);
        let hull = hull(points.iter().map(|p| p.truncate()).collect());
        let (center, radius) = circle(&hull);
        let (middle, half, yaw) = rectangle(&hull);
        let outline = if PI * radius * radius < 4.0 * half.x * half.y {
            Outline::Round { center, radius }
        } else {
            Outline::Square {
                center: middle,
                half,
                yaw,
            }
        };
        Self {
            outline,
            top,
            bottom: lowest.z,
            lowest,
            room: None,
        }
    }

    /// Opens a square fit at the top where `readings`, the object's own, hold its hollow's floor,
    /// turned as the least rectangle around the floor is turned: the walls stand along the floor's
    /// sides, where a handle, or a wall that is a thing of its own, can turn the least rectangle
    /// around the whole. The room is as [`room`] measures it; a round hollow's room is no box,
    /// as a bowl's is not on depth so noisy that its outline is square. A round fit stays round.
    pub(crate) fn open(&mut self, readings: &[Reading]) {
        let Outline::Square { .. } = self.outline else {
            return;
        };
        let floor = readings.iter().filter(|r| r.floor);
        let (_, _, yaw) = rectangle(&hull(floor.map(|r| r.point.truncate()).collect()));
        let all = hull(readings.iter().map(|r| r.point.truncate()).collect());
        let (center, half, yaw) = around(&all, Vec2::from_angle(yaw));
        // In the frame of the outline, at the height 0: the floor's top comes out a height.
        let frame = Quat::from_rotation_z(yaw);
        if let Some(room) = room(center.extend(0.0), half, frame, readings) {
            self.outline = Outline::Square { center, half, yaw };
            self.room = Some(room);
        }
    }

    /// Widens its room, where it has one, so that it holds what stands in it ([`widened`]), and
    /// draws it in from under what its own `readings` show overhangs it, but where `rests_inside`
    /// says that a thing resting on it would then rest inside the box's solids
    /// ([`clear_overhangs`]).
    pub(crate) fn widen(
        &mut self,
        standing: &[Shape],
        readings: &[Reading],
        rests_inside: &dyn Fn(&Shape) -> bool,
    ) {
        if let (Some((lower, upper)), Outline::Square { center, half, yaw }) =
            (self.room, self.outline)
        {
            let frame = Quat::from_rotation_z(yaw);
            let room = widen(center, half, frame, (lower.truncate(), upper), standing);
            let into = |point: Vec3| frame.inverse() * (point - center.extend(0.0));
            let fit = *self;
            let closes = |(low, high): (Vec2, Vec2)| {
                let room = Some((low.extend(lower.z), high));
                rests_inside(&Fit { room, ..fit }.shape())
            };
            let (low, high) = clear_overhangs(room, lower.z, self.top, readings, into, closes);
            self.room = Some((low.extend(lower.z), high));
        }
    }

    /// This fit, as another of the same readings stands, with the room of `widened`.
    pub(crate) fn with_room_of(mut self, widened: &Fit) -> Fit {
        self.room = widened.room;
        self
    }

    /// Closes its room: a box that covers the thing.
    pub(crate) fn close(&mut self) {
        self.room = None;
    }

    /// Takes in the room of `had`, the box open at the top that the object was in the scene
    /// before, where the fit is square: a wall a look before saw thinner stays thinner, as what
    /// a look cannot see stays as the scene before had it. A fit this look did not open is turned
    /// as the box was, around the same readings: the frame its walls stand in.
    ///
    /// Where `readings`, the object's own in this look, show an inner face inside the room, the
    /// room ends there, as [`room`] puts a face: at the innermost window's worth of the readings
    /// inside it by more than their slack, above the floor's top by their slack and under the
    /// box's top, whose planes face into the room from that side, more than half of their normal.
    /// The room is taken into the fit's frame as the box along its axes between the second least
    /// and second most of its corners along each, which lies inside it.
    ///
    /// Not where the room does not lie in the box this look measures, a corner past a side by more
    /// than the most slack of the readings, which bound the box: the thing this look sees is then
    /// not the one that had the room, but a piece of it or another, or is turned apart from it. The
    /// fit stays as this look has it.
    pub(crate) fn carry(&mut self, had: &Shape, readings: &[Reading]) {
        let (
            Outline::Square { .. },
            Shape::OpenBox {
                center: was,
                half_extents,
                rotation,
                ..
            },
        ) = (self.outline, *had)
        else {
            return;
        };
        let outline = match self.room {
            Some(_) => self.outline,
            None => {
                let all = hull(readings.iter().map(|r| r.point.truncate()).collect());
                let (center, half, yaw) = around(&all, (rotation * Vec3::X).truncate().normalize());
                Outline::Square { center, half, yaw }
            }
        };
        let Outline::Square { center, half, yaw } = outline else {
            return;
        };
        let (mut lower, mut upper) = had.room();
        let own: Vec<(Vec3, Vec3, &Reading)> = (readings.iter())
            .map(|r| {
                (
                    rotation.inverse() * (r.point - was),
                    rotation.inverse() * r.normal,
                    r,
                )
            })
            .filter(|&(point, _, r)| {
                let inside = (point.truncate() - lower.truncate()).min(upper - point.truncate());
                inside.min_element() > r.slack
                    && point.z > lower.z + r.slack
                    && point.z <= half_extents.z
            })
            .collect();
        for axis in 0..2 {
            for sign in [-1.0, 1.0] {
                let inward = (own.iter())
                    .filter(|&&(_, normal, _)| -sign * normal[axis] > 0.5)
                    .map(|&(point, _, r)| (-sign * point[axis], r.share));
                if let Some(near) = window(inward) {
                    if sign < 0.0 {
                        lower[axis] = lower[axis].max(near);
                    } else {
                        upper[axis] = upper[axis].min(-near);
                    }
                }
            }
        }
        let frame = Quat::from_rotation_z(yaw);
        let corners = [
            lower.truncate(),
            Vec2::new(upper.x, lower.y),
            upper,
            Vec2::new(lower.x, upper.y),
        ]
        .map(|corner| {
            let world = was + rotation * corner.extend(lower.z);
            (frame.inverse() * (world - center.extend(0.0))).truncate()
        });
        let slack = readings.iter().map(|r| r.slack).fold(0.0, f32::max);
        if corners
            .iter()
            .any(|corner| (corner.abs() - half).max_element() > slack)
        {
            return;
        }
        let middle_two = |axis: usize| {
            let mut along = corners.map(|corner| corner[axis]);
            along.sort_by(f32::total_cmp);
            (along[1], along[2])
        };
        let ((x0, x1), (y0, y1)) = (middle_two(0), middle_two(1));
        let floor = was.z + (rotation * lower).z;
        let (low, high) = (Vec2::new(x0, y0).max(-half), Vec2::new(x1, y1).min(half));
        if !low.cmplt(high).all() {
            return;
        }
        self.outline = outline;
        self.room = Some(match self.room {
            Some((now_lower, now_upper)) => (
                now_lower.truncate().min(low).extend(now_lower.z.min(floor)),
                now_upper.max(high),
            ),
            None => (low.extend(floor), high),
        });
    }

    /// Lowers its bottom to `height`, the top of what it stands on there: what a camera sees of
    /// an object ends where something hides the rest, and the object does not.
    pub(crate) fn stand_on(&mut self, height: f32) {
        self.bottom = self.bottom.min(height);
    }

    /// Checks the height at which the support search stopped, before center/half-height
    /// encoding. The lowest reading must be over the inside of that floor.
    pub(crate) fn stood_on_floor(&self, container: &Shape, body: &Shape) -> bool {
        let Shape::OpenBox {
            center,
            rotation,
            half_extents,
            ..
        } = *container
        else {
            return false;
        };
        if rotation.x != 0.0 || rotation.y != 0.0 {
            return false;
        }
        let (lower, upper) = container.room();
        let plane = f64::from(center.z) + f64::from(lower.z);
        let height = plane as f32;
        if lower.z <= -half_extents.z
            || f64::from(self.lowest.z) < plane
            || (f64::from(self.bottom) - plane).abs() > f64::from(REACHED)
            || self.bottom < height.next_down()
            || self.bottom > height.next_up()
        {
            return false;
        }
        let own = rotation.as_dquat().normalize().inverse()
            * (self.lowest.as_dvec3() - center.as_dvec3());
        own.truncate().cmpgt(lower.truncate().as_dvec2()).all()
            && own.truncate().cmplt(upper.as_dvec2()).all()
            && self.shape() == *body
    }

    /// The fit as a shape: an upright cylinder or a box turned about z.
    pub(crate) fn shape(&self) -> Shape {
        let (middle, half_height) = (
            0.5 * (self.top + self.bottom),
            0.5 * (self.top - self.bottom),
        );
        match self.outline {
            Outline::Round { center, radius } => Shape::Cylinder {
                center: center.extend(middle),
                rotation: Quat::IDENTITY,
                radius,
                half_height,
            },
            Outline::Square { center, half, yaw } => match self.room {
                Some((lower, upper)) => Shape::OpenBox {
                    center: center.extend(middle),
                    half_extents: half.extend(half_height),
                    rotation: Quat::from_rotation_z(yaw),
                    room_lower: lower
                        .truncate()
                        .extend((lower.z - middle).clamp(-half_height, half_height)),
                    room_upper: upper,
                },
                None => Shape::Cuboid {
                    center: center.extend(middle),
                    half_extents: half.extend(half_height),
                    rotation: Quat::from_rotation_z(yaw),
                },
            },
        }
    }
}

/// `shape`, a box or one open at the top, opened at the top where `readings`, its object's own in
/// this look, hold its hollow's floor: with the room they show in its frame, and of one open
/// already, with the room as well that it had. A wall a look sees thinner is thinner, and one no
/// look sees into stays as it was. `None` where they show no room, or `shape` is round.
pub(crate) fn opened(shape: &Shape, readings: &[Reading]) -> Option<Shape> {
    let (center, half_extents, rotation, had) = match *shape {
        Shape::Cuboid {
            center,
            half_extents,
            rotation,
        } => (center, half_extents, rotation, None),
        Shape::OpenBox {
            center,
            half_extents,
            rotation,
            room_lower,
            room_upper,
        } => (
            center,
            half_extents,
            rotation,
            Some((room_lower, room_upper)),
        ),
        _ => return None,
    };
    let (mut lower, mut upper) = room(center, half_extents.truncate(), rotation, readings)?;
    lower.z = lower.z.clamp(-half_extents.z, half_extents.z);
    if let Some((had_lower, had_upper)) = had {
        (lower, upper) = (lower.min(had_lower), upper.max(had_upper));
    }
    Some(Shape::OpenBox {
        center,
        half_extents,
        rotation,
        room_lower: lower,
        room_upper: upper,
    })
}

/// `shape`, a box open at the top, its room widened so that it holds what stands in it ([`widen`]):
/// `standing`, the shapes of the things that stand on its floor ([`stands_in`]); and drawn in
/// from under what its own `readings` show overhangs it, as [`Fit::widen`] draws it, but where
/// that puts in a wall the lowest reading of a thing of `resting`. Any other shape as it is.
pub(crate) fn widened(
    shape: &Shape,
    standing: &[Shape],
    readings: &[Reading],
    rests_inside: &dyn Fn(&Shape) -> bool,
) -> Shape {
    let Shape::OpenBox {
        center,
        half_extents,
        rotation,
        ..
    } = *shape
    else {
        return *shape;
    };
    let (lower, upper) = shape.room();
    let room = (lower.truncate(), upper);
    let room = widen(
        center.truncate(),
        half_extents.truncate(),
        rotation,
        room,
        standing,
    );
    let into = |point: Vec3| rotation.inverse() * (point - center);
    let open = |(low, high): (Vec2, Vec2)| Shape::OpenBox {
        center,
        half_extents,
        rotation,
        room_lower: low.extend(lower.z),
        room_upper: high,
    };
    let closes = |room: (Vec2, Vec2)| rests_inside(&open(room));
    open(clear_overhangs(
        room,
        lower.z,
        half_extents.z,
        readings,
        into,
        closes,
    ))
}

/// Whether a thing whose lowest reading is `lowest`, a reading that may lie `slack` off a surface,
/// stands in the room of `shape`, a box open at the top: whether it lies inside the box seen from
/// above, on or over the floor's top, and nearer the floor's top than the box's top, so that it
/// stands on the floor or on what stands there and not on a wall.
pub(crate) fn stands_in(shape: &Shape, lowest: Vec3, slack: f32) -> bool {
    let Shape::OpenBox {
        center,
        half_extents: half,
        rotation,
        ..
    } = *shape
    else {
        return false;
    };
    let own = rotation.inverse() * (lowest - center);
    let floor = shape.room().0.z;
    own.truncate().abs().cmple(half.truncate()).all()
        && own.z >= floor - slack
        && own.z - floor < half.z - own.z
}

/// The room `(lower, upper)` along x and y in the frame of a box of `half` sizes from above about
/// `center`, turned by `rotation`, widened to hold `standing`, the shapes of things that stand on
/// its floor: a thing that stands there shows that the floor runs under it, and no wall stands
/// where it does. Its outline from above counts as the floor's reach on every side it lies
/// toward, as far as the box.
fn widen(
    center: Vec2,
    half: Vec2,
    rotation: Quat,
    (mut lower, mut upper): (Vec2, Vec2),
    standing: &[Shape],
) -> (Vec2, Vec2) {
    let own = |point: Vec3| (rotation.inverse() * (point - center.extend(0.0))).truncate();
    for shape in standing {
        let (least, most) = match *shape {
            Shape::Cylinder { center, radius, .. } => {
                let middle = own(center.with_z(0.0));
                (middle - Vec2::splat(radius), middle + Vec2::splat(radius))
            }
            Shape::Cuboid {
                center,
                half_extents,
                rotation,
            }
            | Shape::OpenBox {
                center,
                half_extents,
                rotation,
                ..
            } => {
                let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(x, y)| {
                    own(center.with_z(0.0) + rotation * (half_extents * Vec3::new(x, y, 0.0)))
                });
                let least = corners.iter().copied().reduce(Vec2::min);
                let most = corners.iter().copied().reduce(Vec2::max);
                (
                    least.expect("a box has corners"),
                    most.expect("a box has corners"),
                )
            }
        };
        (lower, upper) = (lower.min(least), upper.max(most));
    }
    (lower.max(-half), upper.min(half))
}

/// The room of the hollow whose floor `readings` hold (those [`Reading::floor`] says are), in the
/// frame of a box of `half` sizes from above about `center`, turned by `rotation`: its lower
/// corner, the floor's top its height, and its upper corner along x and y. `None` without a floor,
/// where the hollow is round, or where the faces close the room.
///
/// The floor's top is as high as a window's worth of its readings whose planes face up (more than
/// half of the normal up) reach: fewer cannot be told from noise, and a reading along its foot
/// without a plane can lie on a wall. A wall's inner face is shown by the readings of planes that
/// face into the room, more than half of their normal pointing in, above the floor's top by their
/// slack, beyond the floor's middle toward the wall and within its reach across. Where they add up
/// to a window, the face lies at their innermost window's worth, and is flat where more than half
/// of them lie within their slack of it. Where none add up to a window, as for a wall that faces
/// away from every camera, the room reaches as far as the floor is seen: to the farthest a
/// window's worth of the floor reaches. Such a wall is as thick as what hides the floor, which
/// errs toward the wall. A face is never outside the box: where the room reaches a side, that side
/// has no wall.
///
/// A hollow is round, and no box holds its room, where its floor is round from above (its least
/// circle is smaller than its least rectangle, as an outline's is told round) or a face it shows
/// is not flat: a round hollow's faces turn away from any one face, and its floor seen in part
/// can be square.
fn room(center: Vec3, half: Vec2, rotation: Quat, readings: &[Reading]) -> Option<(Vec3, Vec2)> {
    let own: Vec<(Vec3, Vec3, &Reading)> = (readings.iter())
        .map(|r| {
            let point = rotation.inverse() * (r.point - center);
            (point, rotation.inverse() * r.normal, r)
        })
        .collect();
    let floor: Vec<(Vec3, f32)> = (own.iter())
        .filter(|(_, _, r)| r.floor)
        .map(|&(point, _, r)| (point, r.share))
        .collect();
    let outline = hull(floor.iter().map(|(point, _)| point.truncate()).collect());
    let ((_, radius), (_, square, _)) = (circle(&outline), rectangle(&outline));
    if PI * radius * radius < 4.0 * square.x * square.y {
        return None;
    }
    // How far toward `along` the floor reaches: the farthest a window's worth of it lies at or
    // beyond.
    let reach = |along: Vec3| {
        window(
            floor
                .iter()
                .map(|&(point, share)| (point.dot(along), share)),
        )
    };
    // Its top by the readings whose planes face up: those along its foot that have none can lie
    // on a wall.
    let up = (own.iter()).filter(|(_, normal, r)| r.floor && normal.z > 0.5);
    let top = window(up.map(|&(point, _, r)| (point.z, r.share)))?;
    // Per axis, how far the floor reaches toward it and how far back: each asked once.
    let mut spans = [(0.0, 0.0); 2];
    for (axis, span) in spans.iter_mut().enumerate() {
        let out = Vec3::AXES[axis];
        *span = (reach(out)?, -reach(-out)?);
    }
    let (mut lower, mut upper) = (Vec2::ZERO, Vec2::ZERO);
    for axis in 0..2 {
        let ((high, low), (wide, narrow)) = (spans[axis], spans[1 - axis]);
        for sign in [-1.0, 1.0] {
            let faces = own.iter().filter(|&&(point, normal, r)| {
                !r.floor
                    && -sign * normal[axis] > 0.5
                    && point.z > top + r.slack
                    && sign * (point[axis] - 0.5 * (high + low)) > 0.0
                    && (narrow..=wide).contains(&point[1 - axis])
            });
            // How far toward the room's middle each lies, and how far off the face it may read.
            let inward: Vec<(f32, f32, f32)> =
                (faces.map(|&(point, _, r)| (-sign * point[axis], r.share, r.slack))).collect();
            // The innermost window's worth, as the farthest toward the room's middle.
            let face = match window(inward.iter().map(|&(at, share, _)| (at, share))) {
                Some(near) => {
                    // A flat wall's readings lie on its face, within their slack; a round
                    // hollow's turn away from it.
                    let (on, all) =
                        (inward.iter()).fold((0.0, 0.0), |(on, all), &(at, share, slack)| {
                            (
                                on + if at >= near - slack { share } else { 0.0 },
                                all + share,
                            )
                        });
                    if 2.0 * on <= all {
                        return None;
                    }
                    -sign * near
                }
                None if sign < 0.0 => low,
                None => high,
            };
            let face = face.clamp(-half[axis], half[axis]);
            if sign < 0.0 {
                lower[axis] = face;
            } else {
                upper[axis] = face;
            }
        }
    }
    lower.cmplt(upper).all().then(|| (lower.extend(top), upper))
}

/// The room of a box open at the top between two corners, along x and y in the box's frame,
/// drawn in from under the box's own readings that overhang it: of `readings`, those that `into`
/// takes into that frame over the floor's top `floor` and inside the room by more than their
/// slack, no higher than the box's top `top`, and of no floor. A top over the back of a drawer
/// pulled out of a cabinet is wall, not room, though the floor reads on under it.
///
/// A wall moves in only to readings that reach it and span it. They reach it where each lies
/// within its slack and a window's width ([`Reading::spacing`] over the square root of its share)
/// of the wall or of one nearer it: a bar across the middle of the room reaches neither wall it
/// runs along. They span it where, along it from one side of the room to the other,
/// no gap wider than that lies between them, or between them and a side, with the floor read in
/// it within the strip the wall would take: a handle arched over the room reaches the walls it
/// stands on but leaves room seen on both sides; a gap no view sees into, as behind an arm, is
/// none. The wall moves to the innermost window's worth of them ([`window`]), as [`room`] puts a
/// face; fewer readings than add up to a window cannot be told from noise, and stay. Of the walls
/// that can move, the one that takes the most of them per area of room it gives up moves first,
/// until fewer than a window's worth are left in the room; a wall stops at a reading inside the
/// room, so the room never closes. No wall moves where `rests_in_wall` says a thing that rests
/// on the box would then rest in its solids: the room stays as it is on that side.
fn clear_overhangs(
    (mut lower, mut upper): (Vec2, Vec2),
    floor: f32,
    top: f32,
    readings: &[Reading],
    into: impl Fn(Vec3) -> Vec3,
    rests_in_wall: impl Fn((Vec2, Vec2)) -> bool,
) -> (Vec2, Vec2) {
    let inside = |at: Vec2, slack: f32, (lower, upper): (Vec2, Vec2)| {
        (at - lower).min(upper - at).min_element() > slack
    };
    // Each reading over the room: where it lies from above, its slack, its share and its window's
    // width.
    let mut over: Vec<(Vec2, f32, f32, f32)> = (readings.iter())
        .filter(|r| !r.floor)
        .filter_map(|r| {
            let point = into(r.point);
            let over = point.z > floor + r.slack
                && point.z <= top
                && inside(point.truncate(), r.slack, (lower, upper));
            over.then(|| {
                (
                    point.truncate(),
                    r.slack,
                    r.share,
                    r.spacing / r.share.sqrt(),
                )
            })
        })
        .collect();
    // Where the floor reads, from above: what shows a gap beside an overhang to be room, and not
    // hidden from the views.
    let floor_at: Vec<Vec2> = match over.iter().map(|&(_, _, share, _)| share).sum::<f32>() < 1.0 {
        true => vec![],
        false => (readings.iter())
            .filter(|r| r.floor)
            .map(|r| into(r.point).truncate())
            .collect(),
    };
    // The walls, each an axis and the side it stands on.
    let walls = [(0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0)];
    loop {
        over.retain(|&(at, slack, ..)| inside(at, slack, (lower, upper)));
        if over.iter().map(|&(_, _, share, _)| share).sum::<f32>() < 1.0 {
            return (lower, upper);
        }
        let span = upper - lower;
        // The room the best wall leaves, and how much of the overhang it takes per area given up.
        let mut best: Option<(f32, (Vec2, Vec2))> = None;
        for &(axis, sign) in &walls {
            // How far in from its side the wall stands; a reading lies `-sign * at[axis]` in.
            let stands = if sign < 0.0 {
                lower[axis]
            } else {
                -upper[axis]
            };
            let mut nearest: Vec<(f32, f32, f32, f32)> = (over.iter())
                .map(|&(at, slack, share, width)| {
                    (-sign * at[axis], at[1 - axis], slack + width, share)
                })
                .collect();
            nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
            // Those that reach the wall: each within a window's width of the wall or of one
            // nearer it.
            let mut reach = stands;
            let reached: Vec<(f32, f32, f32, f32)> = (nearest.into_iter())
                .take_while(|&(inward, _, width, _)| {
                    let near = inward - reach <= width;
                    reach = reach.max(inward);
                    near
                })
                .collect();
            if reached.iter().map(|&(.., share)| share).sum::<f32>() < 1.0 {
                continue;
            }
            let Some(to) = window(reached.iter().map(|&(inward, .., share)| (inward, share)))
            else {
                continue;
            };
            // That span it, along it from side to side: a gap between them, or between them and a
            // side, where the floor reads within the strip the wall would take is room seen
            // beside them; one the views do not see into, as behind an arm, is none.
            {
                let mut along: Vec<(f32, f32)> = reached
                    .iter()
                    .map(|&(_, along, width, _)| (along, width))
                    .collect();
                along.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut gaps = vec![];
                let mut from = (lower[1 - axis], 0.0);
                for &(at, width) in &along {
                    if at - from.0 > width.max(from.1) {
                        gaps.push((from.0, at));
                    }
                    from = (from.0.max(at), width);
                }
                if upper[1 - axis] - from.0 > from.1 {
                    gaps.push((from.0, upper[1 - axis]));
                }
                let open = |&(low, high): &(f32, f32)| {
                    (floor_at.iter()).any(|at| {
                        let inward = -sign * at[axis] - stands;
                        at[1 - axis] > low
                            && at[1 - axis] < high
                            && inward > 0.0
                            && inward < to - stands
                    })
                };
                if gaps.iter().any(open) {
                    continue;
                }
            }
            let mut room = (lower, upper);
            match sign < 0.0 {
                true => room.0[axis] = to,
                false => room.1[axis] = -to,
            }
            if rests_in_wall(room) {
                continue;
            }
            let given = (to - stands).min(span[axis]) * span[1 - axis];
            let taken: f32 = (over.iter())
                .filter(|&&(at, slack, ..)| !inside(at, slack, room))
                .map(|&(_, _, share, _)| share)
                .sum();
            let rate = taken / given;
            if best.is_none_or(|(most, _)| rate > most) {
                best = Some((rate, room));
            }
        }
        let Some((_, room)) = best else {
            return (lower, upper);
        };
        (lower, upper) = room;
    }
}

/// Whether a current inner wall separates every reading of another part from the room.
/// Correspondence is in the plane of the wall, so a crop, a hole or a conflict below the
/// floor cannot be filled in by extending a face that was only seen elsewhere.
pub(crate) fn behind_wall(
    room: &Shape,
    readings: &[Reading],
    points: &[Vec3],
    slack: &[f32],
) -> bool {
    let Shape::OpenBox {
        center,
        rotation,
        half_extents,
        ..
    } = *room
    else {
        return false;
    };
    let (floor, upper) = room.room();
    let local = |p: Vec3| rotation.inverse() * (p - center);
    for axis in 0..2 {
        let tangent = 1 - axis;
        for (side, edge) in [(-1.0, floor[axis]), (1.0, upper[axis])] {
            if !points
                .iter()
                .zip(slack)
                .all(|(&p, &s)| side * (local(p)[axis] - edge) > s)
            {
                continue;
            }
            let mut wall: Vec<_> = readings
                .iter()
                .filter_map(|r| {
                    let p = local(r.point);
                    let normal = rotation.inverse() * r.normal;
                    (!r.floor
                        && -side * normal[axis] > 0.5
                        && (p[axis] - edge).abs() <= r.slack
                        && p.z > floor.z + r.slack
                        && p.z <= half_extents.z
                        && p[tangent] >= floor[tangent]
                        && p[tangent] <= upper[tangent])
                        .then_some((p[tangent], p.z, r.slack))
                })
                .collect();
            wall.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let behind = points.iter().zip(slack).all(|(&p, &s)| {
                let p = local(p);
                let first = wall.partition_point(|q| q.0 < p[tangent] - s);
                wall[first..]
                    .iter()
                    .take_while(|q| q.0 <= p[tangent] + s)
                    .any(|&(t, z, w)| {
                        let distance = Vec2::new(p[tangent] - t, p.z - z);
                        distance.length_squared() <= s.min(w).powi(2)
                            && side * (p[axis] - edge) > s + w
                    })
            });
            if behind {
                return true;
            }
        }
    }
    false
}

/// Of `values`, each a position and the share of a window its reading is, the farthest position
/// that readings adding up to a window lie at or beyond; `None` where all add up to less.
fn window(values: impl Iterator<Item = (f32, f32)>) -> Option<f32> {
    let mut values: Vec<(f32, f32)> = values.collect();
    values.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    reached(values.into_iter())
}

/// The least of `values` that more than half of them lie at or before: where most of a floor
/// lies, whatever of its walls is read with it. `None` where there are none.
pub(crate) fn level(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut values: Vec<f32> = values.collect();
    let middle = values.len() / 2;
    (!values.is_empty()).then(|| *values.select_nth_unstable_by(middle, f32::total_cmp).1)
}

/// As [`window`] from both ends: the nearest position that readings adding up to a window lie at
/// or before, and the farthest that they lie at or beyond; `None` where all add up to less. Only
/// the readings at either end are put in order, as many as it takes.
pub(crate) fn windows(values: impl Iterator<Item = (f32, f32)>) -> Option<(f32, f32)> {
    let mut values: Vec<(f32, f32)> = values.collect();
    let low = from_end(&mut values, |a, b| a.0.total_cmp(&b.0))?;
    let high = from_end(&mut values, |a, b| b.0.total_cmp(&a.0))?;
    Some((low, high))
}

/// Where `values`, in the order `order` puts them, add up to a window: found among the first of
/// them in that order, more of them each time they add up to less.
fn from_end(
    values: &mut [(f32, f32)],
    order: impl Fn(&(f32, f32), &(f32, f32)) -> std::cmp::Ordering + Copy,
) -> Option<f32> {
    let mut first = values.len().min(256);
    loop {
        if first < values.len() {
            values.select_nth_unstable_by(first, order);
        }
        let mut head = values[..first].to_vec();
        head.sort_unstable_by(order);
        if let Some(position) = reached(head.into_iter()) {
            return Some(position);
        }
        if first == values.len() {
            return None;
        }
        first = values.len().min(4 * first);
    }
}

/// The position where `values`, in their order, add up to a window.
fn reached(mut values: impl Iterator<Item = (f32, f32)>) -> Option<f32> {
    let mut sum = 0.0;
    values.find_map(|(position, share)| {
        sum += share;
        (sum >= 1.0).then_some(position)
    })
}

/// How far straight under `from` the surface lies whose signed distance `distance` gives
/// (meters): 0 where `from` is inside it, and `None` where nothing of it lies under `from` down
/// to the height `floor`. Found by stepping down by the distance, which never passes a surface.
pub(crate) fn under(distance: impl Fn(Vec3) -> f32, from: Vec3, floor: f32) -> Option<f32> {
    let mut at = from;
    while at.z >= floor {
        let to = distance(at);
        if to <= REACHED {
            return Some(from.z - at.z);
        }
        // A step too small to lower it, or none at all, would never end.
        let lowers = at.z - to < at.z;
        if !lowers {
            return None;
        }
        at.z -= to;
    }
    None
}

/// How near a surface the steps down to it stop (meters): a numerical bound, a hundred of the
/// steps a single-precision height has at a meter, and under any camera's noise.
pub(crate) const REACHED: f32 = 1e-5;

/// The convex hull of `points`, counterclockwise from its lowest leftmost corner (Andrew's
/// monotone chain). Points on its sides are left out.
pub(crate) fn hull(mut points: Vec<Vec2>) -> Vec<Vec2> {
    // A point inside the eight-sided outline through the farthest points toward eight sides
    // is no corner of the hull, and most points are: only the rest are put in order (Akl and
    // Toussaint). Inside by more than rounding can undo: sixteen times what a float holds of
    // the largest coordinate.
    let toward = [
        Vec2::X,
        Vec2::ONE,
        Vec2::Y,
        Vec2::new(-1.0, 1.0),
        Vec2::NEG_X,
        Vec2::NEG_ONE,
        Vec2::NEG_Y,
        Vec2::new(1.0, -1.0),
    ];
    if let Some(&first) = points.first() {
        let mut ends = [first; 8];
        for &point in &points {
            for (end, toward) in ends.iter_mut().zip(toward) {
                if point.dot(toward) > end.dot(toward) {
                    *end = point;
                }
            }
        }
        let largest = ends.iter().map(|end| end.abs().max_element());
        let margin = 16.0 * f32::EPSILON * largest.fold(0.0, f32::max);
        let mut ends = ends.to_vec();
        ends.dedup();
        // Each side's start, its run, and how far inside it a point must lie, as `holds` asks.
        let sides: Vec<(Vec2, Vec2, f32)> = (ends.iter().zip(ends.iter().cycle().skip(1)))
            .map(|(&a, &b)| (a, b - a, margin * (b - a).length()))
            .collect();
        if sides.len() > 2 {
            let inside = |point: Vec2| {
                (sides.iter()).all(|&(start, run, least)| run.perp_dot(point - start) > least)
            };
            points.retain(|&point| !inside(point));
        }
    }
    // Two points this order holds equal are the same to the bit, so no sort orders them apart.
    points.sort_unstable_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    points.dedup();
    if points.len() < 3 {
        return points;
    }
    let mut hull: Vec<Vec2> = Vec::with_capacity(2 * points.len());
    let turns_left = |hull: &[Vec2], p: Vec2| {
        let (a, b) = (hull[hull.len() - 2], hull[hull.len() - 1]);
        (b - a).perp_dot(p - b) > 0.0
    };
    for &p in &points {
        while hull.len() >= 2 && !turns_left(&hull, p) {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in points.iter().rev().skip(1) {
        while hull.len() >= lower && !turns_left(&hull, p) {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// The area a convex hull takes up, its corners in order.
pub(crate) fn area(hull: &[Vec2]) -> f32 {
    let around = hull.iter().zip(hull.iter().cycle().skip(1));
    0.5 * around.map(|(a, b)| a.perp_dot(*b)).sum::<f32>().abs()
}

/// Whether `point` lies inside a convex hull, its corners counterclockwise, farther from each of
/// its sides than `margin`.
pub(crate) fn holds(hull: &[Vec2], point: Vec2, margin: f32) -> bool {
    let mut sides = hull.iter().zip(hull.iter().cycle().skip(1));
    hull.len() > 2
        && sides.all(|(a, b)| (*b - *a).perp_dot(point - *a) > margin * (*b - *a).length())
}

/// The rectangle of least area around the corners of a convex hull: its middle, its half sizes
/// along its own axes and its turn about z. One of its sides lies along one of the hull's, so
/// each of those is tried.
pub(crate) fn rectangle(hull: &[Vec2]) -> (Vec2, Vec2, f32) {
    let sides = (0..hull.len()).map(|k| hull[(k + 1) % hull.len()] - hull[k]);
    let axes = sides.filter_map(Vec2::try_normalize);
    let least = axes
        .map(|axis| around(hull, axis))
        .min_by(|a, b| (a.1.x * a.1.y).total_cmp(&(b.1.x * b.1.y)));
    // One point, or several at one place, has no side: a rectangle of no size along x and y.
    least.unwrap_or_else(|| around(hull, Vec2::X))
}

/// The rectangle around `points` with a side along `axis`, a unit vector: its middle, its half
/// sizes along `axis` and across it, and its turn about z.
fn around(points: &[Vec2], axis: Vec2) -> (Vec2, Vec2, f32) {
    let across = axis.perp();
    let mut span = (Vec2::INFINITY, Vec2::NEG_INFINITY);
    for p in points {
        let own = Vec2::new(p.dot(axis), p.dot(across));
        span = (span.0.min(own), span.1.max(own));
    }
    let (middle, half) = (0.5 * (span.0 + span.1), 0.5 * (span.1 - span.0));
    (
        middle.x * axis + middle.y * across,
        half,
        axis.y.atan2(axis.x),
    )
}

/// The least circle around `points`: its middle and radius (Welzl's algorithm, the points taken
/// in an order drawn from a fixed seed, which keeps it linear and the same every time).
fn circle(points: &[Vec2]) -> (Vec2, f32) {
    let mut points = points.to_vec();
    // SplitMix64 from a fixed seed.
    let mut state = 0u64;
    let mut draw = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mixed = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^ (mixed >> 31)
    };
    for k in (1..points.len()).rev() {
        points.swap(k, (draw() % (k as u64 + 1)) as usize);
    }
    // A little past the radius counts as inside, or rounding sends a point on the circle round
    // again.
    let outside =
        |(center, radius): (Vec2, f32), p: Vec2| p.distance(center) > radius * (1.0 + 1e-6);
    let through_two = |a: Vec2, b: Vec2| (0.5 * (a + b), 0.5 * a.distance(b));
    let mut least = (points.first().copied().unwrap_or(Vec2::ZERO), 0.0);
    for i in 0..points.len() {
        if !outside(least, points[i]) {
            continue;
        }
        least = (points[i], 0.0);
        for j in 0..i {
            if !outside(least, points[j]) {
                continue;
            }
            least = through_two(points[i], points[j]);
            for k in 0..j {
                if outside(least, points[k]) {
                    least = through_three(points[i], points[j], points[k]);
                }
            }
        }
    }
    least
}

/// The circle through three points, or around the two farthest apart where they lie on a line.
fn through_three(a: Vec2, b: Vec2, c: Vec2) -> (Vec2, f32) {
    let (ab, ac) = (b - a, c - a);
    let twice_area = ab.perp_dot(ac);
    if twice_area.abs() <= f32::EPSILON * ab.length() * ac.length() {
        let pairs = [(a, b), (a, c), (b, c)];
        let (p, q) = pairs
            .into_iter()
            .max_by(|x, y| x.0.distance(x.1).total_cmp(&y.0.distance(y.1)))
            .expect("three pairs");
        return (0.5 * (p + q), 0.5 * p.distance(q));
    }
    let offset = (ab * ac.length_squared() - ac * ab.length_squared()).perp() / (2.0 * twice_area);
    (a + offset, offset.length())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_floor_return_is_still_a_witness_after_center_encoding() {
        let container = Shape::OpenBox {
            center: Vec3::new(0.5, 0.0, 0.78000504),
            half_extents: Vec3::new(0.12, 0.12, 0.030005038),
            rotation: Quat::from_rotation_z(0.65),
            room_lower: Vec3::new(-0.1099, -0.1099, -0.019995153),
            room_upper: Vec2::splat(0.1099),
        };
        let floor = 0.7600099;
        let fit = Fit::of(&[
            Vec3::new(0.48, -0.02, floor),
            Vec3::new(0.48, 0.02, floor),
            Vec3::new(0.52, -0.02, 0.80400985),
            Vec3::new(0.52, 0.02, 0.80400985),
        ]);
        let Shape::Cuboid {
            center,
            half_extents,
            ..
        } = fit.shape()
        else {
            panic!("these four readings fit a cuboid");
        };
        assert!(f64::from(center.z) - f64::from(half_extents.z) < f64::from(floor) - 2e-8);
        assert!(fit.stood_on_floor(&container, &fit.shape()));
    }

    #[test]
    fn a_floor_witness_requires_readings_over_the_standing_plane() {
        for (scale, height) in [(0.25, -0.5), (1.0, 0.15), (4.0, 2.0)] {
            let center = Vec3::new(scale, -scale, height);
            let half = scale * Vec3::new(0.12, 0.08, 0.04);
            let level = 0.01 * scale;
            let container = Shape::OpenBox {
                center,
                half_extents: half,
                rotation: Quat::from_rotation_z(-0.6),
                room_lower: (-half.truncate()).extend(level),
                room_upper: half.truncate(),
            };
            let plane = height + level;
            let lowest = center.with_z(plane + scale * 0.002);
            let mut fit = Fit::of(&[lowest, lowest + scale * Vec3::new(0.02, 0.01, 0.04)]);
            fit.stand_on(plane);
            let body = fit.shape();
            assert!(fit.stood_on_floor(&container, &body));
            // A reading below the floor and a support search that ended below it are independent
            // reasons to withhold a witness, even if its shape is exactly the one published.
            let penetrating = Fit {
                lowest: lowest.with_z(plane - scale * 0.001),
                ..fit
            };
            assert!(!penetrating.stood_on_floor(&container, &penetrating.shape()));
            let deep = Fit {
                bottom: plane - scale * 1e-5,
                ..fit
            };
            assert!(!deep.stood_on_floor(&container, &deep.shape()));
            let outside = Fit {
                lowest: lowest + Vec3::X * scale,
                ..fit
            };
            assert!(!outside.stood_on_floor(&container, &outside.shape()));
            assert!(!fit.stood_on_floor(&container, &deep.shape()));
        }
    }

    /// The circle through three points passes each of them, and three on a line are spanned by
    /// the two ends.
    #[test]
    fn a_circle_through_three_points_passes_each() {
        let (a, b, c) = (
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        );
        let (center, radius) = through_three(a, b, c);
        for p in [a, b, c] {
            assert!(
                (p.distance(center) - radius).abs() < 1e-6,
                "{p} off the circle"
            );
        }
        assert!(
            (center - Vec2::new(1.0, 1.0)).length() < 1e-6,
            "its middle is {center}"
        );
        let (center, radius) = through_three(a, Vec2::new(1.0, 0.0), b);
        assert_eq!((center, radius), (Vec2::new(1.0, 0.0), 1.0));
    }
}
