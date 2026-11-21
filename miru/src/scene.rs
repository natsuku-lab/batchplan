//! What a look found: the objects, and what each pixel of each view shows.

use batchplan::Obstacle;
use glam::{Quat, Vec2, Vec3};

use crate::fit;

/// What one pixel of a view shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Label {
    /// No reading, a reading outside the look's corners, or one no object was made of: too few
    /// readings stand together there to tell them from noise.
    Nothing,
    /// The robot, or what one of its hands holds, as the view's mask says.
    Robot,
    /// One of the look's known shapes, by its index in [`Look::known`](crate::Look::known).
    Known(usize),
    /// One of the scene's objects, by its index in [`Scene::objects`].
    Object(usize),
}

/// What holds an object up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Support {
    /// One of the look's known shapes, by its index in [`Look::known`](crate::Look::known).
    Known(usize),
    /// Another object, by its index in [`Scene::objects`].
    Object(usize),
}

/// A solid an object is measured as, in the look's frame (meters). A box and a cylinder are
/// [`Obstacle`]'s, field for field; a box open at the top is the boxes it is made of. A world
/// takes any of them as obstacles ([`Shape::solids`]).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[allow(
    missing_docs,
    reason = "the enum's doc gives every field's units and frame"
)]
pub enum Shape {
    /// An oriented box.
    Cuboid {
        center: Vec3,
        /// Half the box's size along each of its axes.
        half_extents: Vec3,
        rotation: Quat,
    },
    /// A solid cylinder along its local z axis.
    Cylinder {
        center: Vec3,
        rotation: Quat,
        radius: f32,
        /// Half its length along its axis.
        half_height: f32,
    },
    /// A box open at the top, as a tray, a lidless box or a drawer pulled out: the box of a
    /// [`Shape::Cuboid`] with a room hollowed out of it that reaches up through its top. The room
    /// lies inside the box, between `room_lower` and `room_upper` in the box's own frame (meters
    /// from `center`, along the box's axes): along x and y between the walls' inner faces, and up
    /// from the floor's top, `room_lower.z`. So each wall is as thick as it is from the room to
    /// the box's side, and the floor from the box's bottom to the room's; where the room reaches a
    /// side of the box, that side has no wall. A room given past the box is taken as far as the
    /// box, in every use of the shape.
    OpenBox {
        center: Vec3,
        /// Half the box's size along each of its axes.
        half_extents: Vec3,
        rotation: Quat,
        room_lower: Vec3,
        room_upper: Vec2,
    },
}

impl Shape {
    /// The solids the shape is made of, as the obstacles a world keeps: a box or a cylinder
    /// itself, and a box open at the top five boxes, its floor, the whole of its bottom up to
    /// the room's, and its walls standing on the floor, those at either end along x as long as the
    /// box and those along y between them. A wall or a floor of no thickness is left out.
    #[must_use]
    pub fn solids(&self) -> Vec<Obstacle> {
        match *self {
            Shape::OpenBox { .. } => (self.parts().iter().flatten())
                .map(Shape::obstacle)
                .collect(),
            shape => vec![shape.obstacle()],
        }
    }

    /// A box or a cylinder as the obstacle it is, field for field, and a box open at the top as
    /// its box.
    fn obstacle(&self) -> Obstacle {
        match *self {
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
            } => Obstacle::Cuboid {
                center,
                half_extents,
                rotation,
            },
            Shape::Cylinder {
                center,
                rotation,
                radius,
                half_height,
            } => Obstacle::Cylinder {
                center,
                rotation,
                radius,
                half_height,
            },
        }
    }

    /// A box open at the top's floor and four walls, which [`Shape::solids`] gives as
    /// obstacles, `None` for one of no thickness; nothing for any other shape.
    fn parts(&self) -> [Option<Shape>; 5] {
        let Shape::OpenBox {
            center,
            half_extents: half,
            rotation,
            ..
        } = *self
        else {
            return [None; 5];
        };
        let (lower, upper) = self.room();
        // A box between two corners of the box's own frame, where it has a size along every axis.
        let between = |low: Vec3, high: Vec3| {
            (low.cmplt(high).all()).then(|| Shape::Cuboid {
                center: center + rotation * (0.5 * (low + high)),
                half_extents: 0.5 * (high - low),
                rotation,
            })
        };
        [
            between(-half, half.truncate().extend(lower.z)),
            between(
                Vec3::new(-half.x, -half.y, lower.z),
                Vec3::new(lower.x, half.y, half.z),
            ),
            between(Vec3::new(upper.x, -half.y, lower.z), half),
            between(
                Vec3::new(lower.x, -half.y, lower.z),
                Vec3::new(upper.x, lower.y, half.z),
            ),
            between(
                Vec3::new(lower.x, upper.y, lower.z),
                Vec3::new(upper.x, half.y, half.z),
            ),
        ]
    }

    /// A box open at the top's room, taken as far as its box: its lower corner and its upper
    /// corner along x and y, in the box's own frame. Nothing for any other shape.
    pub(crate) fn room(&self) -> (Vec3, Vec2) {
        match *self {
            Shape::OpenBox {
                half_extents: half,
                room_lower,
                room_upper,
                ..
            } => (
                room_lower.clamp(-half, half),
                room_upper.clamp(-half.truncate(), half.truncate()),
            ),
            _ => (Vec3::ZERO, Vec2::ZERO),
        }
    }

    /// The room it takes up: a box open at the top's box, as a cover of it would be, and any
    /// other shape itself. Where a thing lies and how large it is are held against this, so that
    /// a piece of a thing in its room lies where the thing does.
    pub(crate) fn outer(&self) -> Shape {
        match *self {
            Shape::OpenBox {
                center,
                half_extents,
                rotation,
                ..
            } => Shape::Cuboid {
                center,
                half_extents,
                rotation,
            },
            shape => shape,
        }
    }

    /// [`Shape::faces`] of a box open at the top: each part's faces, but where they lie inside
    /// another part, as a wall's foot on the floor does, which no eye sees. Not generic, so that
    /// the parts' own calls are not of another closure each.
    fn parts_faces(&self, eye: Vec3, pixel: f32, most: usize, visit: &mut dyn FnMut(Vec3, f32)) {
        let distance = self.distances();
        for part in self.parts().iter().flatten() {
            part.faces(eye, pixel, most, |point, angle| {
                if distance(point) >= -fit::REACHED {
                    visit(point, angle);
                }
            });
        }
    }

    /// The signed distance from `p` to the shape's surface (meters): negative inside.
    pub(crate) fn distance(&self, p: Vec3) -> f32 {
        match *self {
            Shape::Cuboid {
                center,
                half_extents,
                rotation,
            } => signed((rotation.inverse() * (p - center)).abs() - half_extents),
            Shape::Cylinder {
                center,
                rotation,
                radius,
                half_height,
            } => {
                let own = rotation.inverse() * (p - center);
                let past = Vec2::new(own.truncate().length() - radius, own.z.abs() - half_height);
                signed(past.extend(f32::NEG_INFINITY))
            }
            Shape::OpenBox { .. } => self.open_distance(p, || self.parts()),
        }
    }

    /// [`Shape::distance`] from each point it is asked of, a box open at the top's parts made
    /// once for all of them.
    pub(crate) fn distances(&self) -> impl Fn(Vec3) -> f32 + '_ {
        let parts = self.parts();
        move |p| match self {
            Shape::OpenBox { .. } => self.open_distance(p, || parts),
            _ => self.distance(p),
        }
    }

    /// [`Shape::distance`] of a box open at the top, `parts` making its parts where they are
    /// needed ([`Shape::parts`]).
    fn open_distance(&self, p: Vec3, parts: impl FnOnce() -> [Option<Shape>; 5]) -> f32 {
        let Shape::OpenBox {
            center,
            half_extents,
            rotation,
            ..
        } = *self
        else {
            return self.distance(p);
        };
        // Inside, how deep it lies in the box and out of the room, the room taken up without
        // end, as nothing over it is solid. Outside, the nearest part's distance, which is no
        // part's where two parts meet inside it.
        let (room_lower, room_upper) = self.room();
        let own = rotation.inverse() * (p - center);
        let middle = 0.5 * (room_lower.truncate() + room_upper);
        let half = 0.5 * (room_upper - room_lower.truncate());
        let room = ((own.truncate() - middle).abs() - half).extend(room_lower.z - own.z);
        let inside = signed(own.abs() - half_extents).max(-signed(room));
        if inside < 0.0 {
            return inside;
        }
        (parts().iter().flatten())
            .map(|part| part.distance(p))
            .fold(f32::INFINITY, f32::min)
    }

    /// How much space it takes up (cubic meters).
    pub(crate) fn volume(&self) -> f32 {
        match *self {
            Shape::Cuboid { half_extents, .. } => 8.0 * half_extents.element_product(),
            Shape::Cylinder {
                radius,
                half_height,
                ..
            } => std::f32::consts::TAU * radius * radius * half_height,
            Shape::OpenBox { half_extents, .. } => {
                let (lower, upper) = self.room();
                let room = (upper - lower.truncate()).extend(half_extents.z - lower.z);
                8.0 * half_extents.element_product() - room.element_product()
            }
        }
    }

    /// Its middle: halfway between its sides, and between its bottom and top (a box open at the
    /// top's box's).
    pub(crate) fn middle(&self) -> Vec3 {
        match *self {
            Shape::Cuboid { center, .. }
            | Shape::Cylinder { center, .. }
            | Shape::OpenBox { center, .. } => center,
        }
    }

    /// The height of its lowest point.
    pub(crate) fn lowest(&self) -> f32 {
        // The look's z axis in the shape's own frame: how far the shape reaches along it.
        match *self {
            Shape::Cuboid {
                center,
                half_extents,
                rotation,
            } => center.z - (rotation.inverse() * Vec3::Z).abs().dot(half_extents),
            Shape::Cylinder {
                center,
                rotation,
                radius,
                half_height,
            } => {
                let up = rotation.inverse() * Vec3::Z;
                center.z - up.truncate().length() * radius - up.z.abs() * half_height
            }
            Shape::OpenBox { .. } => (self.parts().iter().flatten())
                .map(Shape::lowest)
                .fold(self.floor(), f32::min),
        }
    }

    /// The height of its highest point.
    pub(crate) fn highest(&self) -> f32 {
        match *self {
            // Its solids', which a wall of no thickness does not raise.
            Shape::OpenBox { .. } => (self.parts().iter().flatten())
                .map(Shape::highest)
                .fold(self.floor(), f32::max),
            _ => 2.0 * self.middle().z - self.lowest(),
        }
    }

    /// The height of a box open at the top's floor's top, which is its lowest and its highest
    /// where it has no solid; its middle's for any other shape.
    fn floor(&self) -> f32 {
        let (lower, _) = self.room();
        self.middle().z + lower.z
    }

    /// Calls `visit` with points of its surface that face `eye`, the part a camera there would
    /// show, and the angle each takes up seen from there (steradians: in proportion to the pixels
    /// that would show it). The points are a grid on each face, as far apart as two rays from
    /// `eye` that are `pixel` radians apart lie at the shape's middle, and no more than `most`
    /// along a side.
    pub(crate) fn faces(
        &self,
        eye: Vec3,
        pixel: f32,
        most: usize,
        mut visit: impl FnMut(Vec3, f32),
    ) {
        let (center, rotation) = match *self {
            Shape::Cuboid {
                center, rotation, ..
            }
            | Shape::Cylinder {
                center, rotation, ..
            } => (center, rotation),
            Shape::OpenBox { .. } => {
                self.parts_faces(eye, pixel, most, &mut visit);
                return;
            }
        };
        // In the shape's own frame, where a flat face faces the eye when the eye lies beyond it.
        let eye = rotation.inverse() * (eye - center);
        let spacing = pixel * eye.length();
        // How many cells a side `length` long is cut into, and how long each is.
        let cells = |length: f32| {
            let count = (length / spacing).ceil().max(1.0).min(most as f32);
            (count as usize, length / count)
        };
        let middle = |cell: usize, size: f32, half: f32| (cell as f32 + 0.5) * size - half;
        // A patch of `area` whose plane the eye stands `above` takes up its area, times the
        // cosine of the angle it is seen at, over the square of its distance.
        let mut patch = |own: Vec3, area: f32, above: f32| {
            visit(
                center + rotation * own,
                area * above / eye.distance(own).powi(3),
            );
        };
        match *self {
            Shape::Cuboid { half_extents, .. } => {
                for axis in 0..3 {
                    let above = eye[axis].abs() - half_extents[axis];
                    if above <= 0.0 {
                        continue;
                    }
                    let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
                    let (across, width) = cells(2.0 * half_extents[a]);
                    let (along, length) = cells(2.0 * half_extents[b]);
                    for (i, j) in (0..across).flat_map(|i| (0..along).map(move |j| (i, j))) {
                        let mut own = Vec3::ZERO;
                        own[axis] = eye[axis].signum() * half_extents[axis];
                        own[a] = middle(i, width, half_extents[a]);
                        own[b] = middle(j, length, half_extents[b]);
                        patch(own, width * length, above);
                    }
                }
            }
            Shape::Cylinder {
                radius,
                half_height,
                ..
            } => {
                // The cap on the eye's side: the cells of the square around it that lie in it.
                let above = eye.z.abs() - half_height;
                if above > 0.0 {
                    let (across, width) = cells(2.0 * radius);
                    for (i, j) in (0..across).flat_map(|i| (0..across).map(move |j| (i, j))) {
                        let at = Vec2::new(middle(i, width, radius), middle(j, width, radius));
                        if at.length() <= radius {
                            let own = at.extend(eye.z.signum() * half_height);
                            patch(own, width * width, above);
                        }
                    }
                }
                // The wall, where it turns toward the eye.
                let (around, arc) = cells(std::f32::consts::TAU * radius);
                let (along, length) = cells(2.0 * half_height);
                for i in 0..around {
                    let turn = (i as f32 + 0.5) * std::f32::consts::TAU / around as f32;
                    let out = Vec2::from_angle(turn);
                    let above = out.dot(eye.truncate()) - radius;
                    for j in (0..along).filter(|_| above > 0.0) {
                        let own = (radius * out).extend(middle(j, length, half_height));
                        patch(own, arc * length, above);
                    }
                }
            }
            // Its parts' faces are visited above.
            Shape::OpenBox { .. } => {}
        }
    }
}

/// The signed distance to a box's surface from a point that lies `past` each pair of its faces by
/// that much (negative: between them): how far outside a corner region, plus how deep inside the
/// nearest face.
fn signed(past: Vec3) -> f32 {
    past.max(Vec3::ZERO).length() + past.max_element().min(0.0)
}

/// One thing the cameras saw apart from the rest.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Object {
    /// A number that stays with the object from look to look while it stays where it was. Never
    /// given twice among the scenes that follow from one first look.
    pub id: u32,
    /// Its shape in the look's frame, whose z axis is up: an upright cylinder where it is round
    /// seen from above, else a box turned about z, and a box open at the top where some look
    /// showed its hollow's floor among the walls that rise around it, as a tray's or a drawer's
    /// pulled out ([`Shape::OpenBox`]). It reaches down to what the object stands on.
    pub shape: Shape,
    /// Whether the shape stands for the thing: true where more of the readings the look measured
    /// it by lie on the shape's surface than inside it, as a box's and a can's do, and for a box
    /// open at the top, whose room is what the looks saw of the hollow; false where the shape only
    /// names the room the thing takes up, as a bowl's or a basket's does, much of what shows of it
    /// lying inside. On depth as noisy as a stereo camera's most boxes read not solid, their faces
    /// lying at their outermost readings (README). An object that was not solid in the scene
    /// before stays so, unless it is measured open at the top: a look that cannot see into a
    /// thing does not make it solid. The shape of one that is not solid still covers it, and is
    /// the least to keep away from; what reaches into it, or stands in it, needs what the depth
    /// itself shows there.
    pub solid: bool,
    /// The mean colour of its readings, as red, green and blue. None when no view that shows it
    /// has colour.
    pub color: Option<[u8; 3]>,
    /// What it stands on or in: the nearest thing under its lowest reading, or, where that lies
    /// lower than the reading and the look cannot see under the object's middle, the lower
    /// thing it touches most, as a bowl whose foot overhangs a box stands on the box. None with
    /// nothing under it.
    pub on: Option<Support>,
    /// Whether this look saw it. False for an object kept from the scene before, which no view
    /// could see ([`look`](crate::look)): nothing of it was measured and no label names it. Its
    /// id, shape, whether it is solid and its colour are that scene's, as is what it stands on: an
    /// object while that is still in the scene, a known shape by the index it had in that look.
    pub seen: bool,
}

/// What one look saw, and what the next look needs of it.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Scene {
    /// The objects, in the order the views show them: by view, then row by row. After them,
    /// those of the scene before that no view could see ([`Object::seen`]), in its order.
    pub objects: Vec<Object>,
    /// Per view of the look, per pixel in the depth's order, what the pixel shows.
    pub labels: Vec<Vec<Label>>,
    /// The id the next new object takes: 0 before the first look, and one more with each id
    /// given. A caller that gives ids of its own takes them from here and raises this past them
    /// before the next look. It must never be lowered: an id is given once only as long as this
    /// only grows. A look that has more new objects than there are ids left below `u32::MAX` is
    /// refused ([`Error::Input`](batchplan::Error::Input)).
    pub next_id: u32,
}

impl Scene {
    /// Per pixel of view `view`, whether it shows object `object` (its index in
    /// [`Scene::objects`]): the flags batchplan's `Surface::from_depth` takes for the object to
    /// grasp or the support to rest on. As long as the view's depth, false everywhere for an
    /// object the view does not show.
    ///
    /// # Panics
    ///
    /// If the look had no view `view`.
    #[must_use]
    pub fn mask(&self, view: usize, object: usize) -> Vec<bool> {
        let shows = Label::Object(object);
        self.labels[view].iter().map(|l| *l == shows).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box's and a cylinder's distances at points whose distance is known: outside a face, a
    /// corner and the rim, on the surface, and inside.
    #[test]
    fn a_shape_measures_the_distance_to_its_surface() {
        let quarter = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let block = Shape::Cuboid {
            center: Vec3::new(1.0, 0.0, 0.0),
            half_extents: Vec3::new(0.3, 0.1, 0.2),
            rotation: quarter,
        };
        // Turned a quarter, it is 0.1 wide along x and 0.3 along y.
        let cases = [
            (Vec3::new(1.5, 0.0, 0.0), 0.4),
            (Vec3::new(1.0, 0.5, 0.0), 0.2),
            (Vec3::new(1.1, 0.0, 0.0), 0.0),
            (Vec3::new(1.0, 0.0, 0.0), -0.1),
            (Vec3::new(1.4, 0.7, 0.0), 0.5),
        ];
        for (p, distance) in cases {
            assert!(
                (block.distance(p) - distance).abs() < 1e-6,
                "box at {p}: {}",
                block.distance(p)
            );
        }
        let can = Shape::Cylinder {
            center: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            radius: 0.3,
            half_height: 0.5,
        };
        let cases = [
            (Vec3::new(0.5, 0.0, 0.0), 0.2),
            (Vec3::new(0.0, 0.0, 0.9), 0.4),
            (Vec3::new(0.6, 0.0, 0.9), 0.5),
            (Vec3::new(0.0, 0.2, 0.0), -0.1),
        ];
        for (p, distance) in cases {
            assert!(
                (can.distance(p) - distance).abs() < 1e-6,
                "can at {p}: {}",
                can.distance(p)
            );
        }
    }

    /// A box's and a can's lowest points upright, lying down and leaning at 45 degrees, where
    /// each is known.
    #[test]
    fn a_shape_reaches_down_to_its_lowest_point() {
        let turns = [
            Quat::from_rotation_z(1.0),
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_4),
        ];
        // How far under their middles a box of half sizes 0.3, 0.1 and 0.2 and a can of radius
        // 0.3 and half height 0.5 reach, turned so.
        let reaches = [
            (0.2, 0.5),
            (0.1, 0.3),
            (0.3 * 0.5f32.sqrt(), 0.8 * 0.5f32.sqrt()),
        ];
        for (rotation, (block, can)) in turns.into_iter().zip(reaches) {
            let center = Vec3::new(1.0, 2.0, 3.0);
            let shapes = [
                Shape::Cuboid {
                    center,
                    half_extents: Vec3::new(0.3, 0.1, 0.2),
                    rotation,
                },
                Shape::Cylinder {
                    center,
                    rotation,
                    radius: 0.3,
                    half_height: 0.5,
                },
            ];
            for (shape, reach) in shapes.iter().zip([block, can]) {
                assert!(
                    (shape.lowest() - (3.0 - reach)).abs() < 1e-6,
                    "{shape:?} reaches down to {}",
                    shape.lowest()
                );
            }
        }
    }

    /// What of a box and of a can faces an eye, where the angle it takes up is known: a
    /// rectangle and a disc seen from straight above their middles, and a can's wall from far
    /// off, where it is as wide and as high as the can.
    #[test]
    fn the_surface_that_faces_an_eye_takes_up_the_angle_it_shows_under() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        // Turned, so that the shapes' own axes are not the frame's.
        let rotation = Quat::from_rotation_x(0.7);
        let (pixel, most) = (0.002, 1000);
        // The angle the points take up, after checking that each lies on the shape, on the
        // eye's side of its middle.
        let angle = |shape: &Shape, eye: Vec3, pixel: f32, most: usize| {
            let (mut sum, mut count) = (0.0, 0);
            shape.faces(eye, pixel, most, |point, angle| {
                assert!(
                    shape.distance(point).abs() < 1e-5,
                    "{point} is off {shape:?}"
                );
                assert!(
                    (point - center).dot(eye - center) > 0.0,
                    "{point} faces away"
                );
                sum += angle;
                count += 1;
            });
            (sum, count)
        };
        let block = Shape::Cuboid {
            center,
            half_extents: Vec3::new(0.3, 0.1, 0.2),
            rotation,
        };
        // Its top, 0.6 by 0.2, from 0.5 above it: 4 atan(a b / (h sqrt(a^2 + b^2 + h^2))).
        let above = center + rotation * Vec3::new(0.0, 0.0, 0.7);
        let top = 4.0 * (0.03f32 / (0.5 * 0.35f32.sqrt())).atan();
        let (sum, count) = angle(&block, above, pixel, most);
        assert!((sum - top).abs() < 0.01 * top, "{sum} for {top}");
        // A point a pixel: 0.6 by 0.2 at 0.7 from its middle, 0.0014 apart.
        assert_eq!(count, 429 * 143);
        // And no more along a side than asked for.
        assert_eq!(angle(&block, above, pixel, 5).1, 25);
        // Seen from beyond a corner, three of its faces show.
        let corner = center + rotation * Vec3::new(3.0, 2.0, 1.0);
        let mut faces = [false; 6];
        block.faces(corner, pixel, most, |point, _| {
            let own = rotation.inverse() * (point - center) / Vec3::new(0.3, 0.1, 0.2);
            let axis = own.abs().max_position();
            faces[2 * axis + usize::from(own[axis] > 0.0)] = true;
        });
        assert_eq!(faces, [false, true, false, true, false, true]);

        let can = Shape::Cylinder {
            center,
            rotation,
            radius: 0.3,
            half_height: 0.5,
        };
        // Its top from 0.4 above it, a disc: 2 pi (1 - h / sqrt(h^2 + r^2)).
        let above = center + rotation * Vec3::new(0.0, 0.0, 0.9);
        let disc = std::f32::consts::TAU * (1.0 - 0.4 / 0.5);
        let (sum, _) = angle(&can, above, pixel, most);
        assert!((sum - disc).abs() < 0.01 * disc, "{sum} for {disc}");
        // Its wall from 100 away, 0.6 wide and 1 high, with pixels fine enough to show it.
        let beside = center + rotation * Vec3::new(100.0, 0.0, 0.0);
        let wall = 0.6 / 100.0f32.powi(2);
        let (sum, _) = angle(&can, beside, 0.05 * pixel, most);
        assert!((sum - wall).abs() < 0.01 * wall, "{sum} for {wall}");
    }

    /// A box open at the top 0.6 by 0.4 by 0.2, its walls 0.1, 0.05, 0.05 and 0.1 through at
    /// -x, +x, -y and +y and its floor 0.05, about `center`, turned by `rotation`.
    fn open_box(center: Vec3, rotation: Quat) -> Shape {
        Shape::OpenBox {
            center,
            half_extents: Vec3::new(0.3, 0.2, 0.1),
            rotation,
            room_lower: Vec3::new(-0.2, -0.15, -0.05),
            room_upper: Vec2::new(0.25, 0.1),
        }
    }

    /// A box open at the top's distances at points whose distance is known: in its room, over
    /// it, in a wall, in its floor, beside it and on its floor's top.
    #[test]
    fn a_box_open_at_the_top_measures_the_distance_to_its_solid() {
        let (center, rotation) = (Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_z(0.7));
        let shape = open_box(center, rotation);
        let cases = [
            // Over the floor's middle, nearer the floor than any wall.
            (Vec3::ZERO, 0.05),
            // Over the box's top: to the top of the +y wall, the nearest.
            (Vec3::new(0.0, 0.0, 0.3), 0.2f32.hypot(0.1)),
            // In the -x wall, as near its side as the room.
            (Vec3::new(-0.25, 0.0, 0.0), -0.05),
            // In the floor under the room.
            (Vec3::new(0.0, 0.0, -0.08), -0.02),
            (Vec3::new(0.5, 0.0, 0.0), 0.2),
            (Vec3::new(0.0, 0.0, -0.05), 0.0),
        ];
        for (own, distance) in cases {
            let p = center + rotation * own;
            assert!(
                (shape.distance(p) - distance).abs() < 1e-6,
                "at {own}: {}",
                shape.distance(p)
            );
        }
    }

    /// A box open at the top is its floor and four walls, which take up what it does, and a wall
    /// of no thickness is none.
    #[test]
    fn a_box_open_at_the_top_is_the_boxes_it_is_made_of() {
        let shape = open_box(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_z(0.7));
        let solids = shape.solids();
        assert_eq!(solids.len(), 5);
        let sizes = solids.iter().map(|solid| match solid {
            Obstacle::Cuboid { half_extents, .. } => 8.0 * half_extents.element_product(),
            other => unreachable!("its solids are boxes, not {other:?}"),
        });
        let volume: f32 = sizes.sum();
        // The box less the room, from the floor's top up: 0.6 * 0.4 * 0.2 - 0.45 * 0.25 * 0.15.
        assert!((volume - 0.031_125).abs() < 1e-6, "{volume}");
        assert!(
            (shape.volume() - 0.031_125).abs() < 1e-6,
            "{}",
            shape.volume()
        );
        let Shape::OpenBox {
            center,
            half_extents,
            rotation,
            room_lower,
            ..
        } = shape
        else {
            unreachable!("made open");
        };
        let open_side = Shape::OpenBox {
            center,
            half_extents,
            rotation,
            room_lower,
            room_upper: Vec2::new(0.3, 0.1),
        };
        assert_eq!(open_side.solids().len(), 4);
        // Each solid's middle lies in it.
        for solid in &solids {
            let Obstacle::Cuboid { center, .. } = *solid else {
                unreachable!("its solids are boxes");
            };
            assert!(shape.distance(center) < 0.0, "{center} of {solid:?}");
        }
        assert!(shape.lowest() < shape.highest() && (shape.highest() - 3.1).abs() < 1e-6);
        let block = Shape::Cuboid {
            center,
            half_extents,
            rotation,
        };
        let as_obstacle = Obstacle::Cuboid {
            center,
            half_extents,
            rotation,
        };
        assert_eq!(block.solids(), [as_obstacle]);
    }

    /// A box open at the top whose room reaches every side has no wall: it reaches up to its
    /// floor's top, not to the box's. A room a caller gives past the box's sides is taken as far
    /// as the box: its solids lie in the box, the walls on those sides none, and its distances and
    /// heights as of that room.
    #[test]
    fn a_box_open_at_the_top_is_as_high_as_its_solids_and_its_room_within_its_box() {
        let (center, rotation) = (Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_z(0.7));
        let half = Vec3::new(0.3, 0.2, 0.1);
        let made = |room_lower: Vec3, room_upper: Vec2| Shape::OpenBox {
            center,
            half_extents: half,
            rotation,
            room_lower,
            room_upper,
        };
        let floor_only = made(Vec3::new(-0.3, -0.2, -0.05), Vec2::new(0.3, 0.2));
        assert_eq!(floor_only.solids().len(), 1);
        assert!(
            (floor_only.highest() - 2.95).abs() < 1e-6,
            "{}",
            floor_only.highest()
        );
        assert!(
            (floor_only.lowest() - 2.9).abs() < 1e-6,
            "{}",
            floor_only.lowest()
        );
        let no_floor = made(Vec3::new(-0.3, -0.2, -0.1), Vec2::new(0.3, 0.2));
        assert!(no_floor.solids().is_empty());
        assert!((no_floor.highest() - 2.9).abs() < 1e-6 && (no_floor.lowest() - 2.9).abs() < 1e-6);
        // Past the -x and +y sides, and under the box's bottom.
        let past = made(Vec3::new(-0.5, -0.15, -0.3), Vec2::new(0.25, 0.4));
        let within = made(Vec3::new(-0.3, -0.15, -0.1), Vec2::new(0.25, 0.2));
        assert_eq!(past.solids(), within.solids());
        assert_eq!(past.solids().len(), 2);
        let box_of = Shape::Cuboid {
            center,
            half_extents: half,
            rotation,
        };
        for solid in past.solids() {
            let Obstacle::Cuboid {
                center: middle,
                half_extents,
                ..
            } = solid
            else {
                unreachable!("its solids are boxes");
            };
            let corners = [-1.0, 1.0].map(|x| {
                [-1.0, 1.0].map(|y| {
                    [-1.0, 1.0].map(|z| middle + rotation * (half_extents * Vec3::new(x, y, z)))
                })
            });
            for corner in corners.iter().flatten().flatten() {
                assert!(box_of.distance(*corner) < 1e-6, "{corner} of {solid:?}");
            }
        }
        for own in [
            Vec3::new(-0.28, 0.0, 0.0),
            Vec3::new(0.0, 0.19, 0.05),
            Vec3::ZERO,
        ] {
            let p = center + rotation * own;
            assert!(
                (past.distance(p) - within.distance(p)).abs() < 1e-6,
                "at {own}"
            );
        }
        assert_eq!(
            (past.lowest(), past.highest(), past.volume()),
            (within.lowest(), within.highest(), within.volume())
        );
    }

    /// Seen from straight above, a box open at the top's wall tops, inner faces and floor take up
    /// the angle its top would, the room's faces showing within its opening; and no point is of a
    /// face that lies inside another part, as a wall's foot does.
    #[test]
    fn a_box_open_at_the_top_shows_its_room_where_its_top_would_show() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let shape = open_box(center, Quat::IDENTITY);
        let eye = center + Vec3::new(0.0, 0.0, 0.6);
        let (pixel, most) = (0.002, 1000);
        let (mut sum, mut floor) = (0.0, false);
        shape.faces(eye, pixel, most, |point, angle| {
            assert!(shape.distance(point).abs() < 1e-5, "{point} is off it");
            floor |= (point.z - (center.z - 0.05)).abs() < 1e-6;
            sum += angle;
        });
        // The top, 0.6 by 0.4, from 0.5 above it: 4 atan(a b / (h sqrt(a^2 + b^2 + h^2))).
        let top = 4.0 * (0.06f32 / (0.5 * 0.38f32.sqrt())).atan();
        assert!((sum - top).abs() < 0.01 * top, "{sum} for {top}");
        assert!(floor, "the floor shows in its room");
    }
}
