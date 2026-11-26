//! Straight lifts out of openings measured in the same look as the object.

use glam::{DVec3, Quat, Vec3};

use crate::error::{Result, ensure_input};
use crate::memory::Kept;
use crate::scene::{Label, Shape, Support};

/// How far to lift a rigid body along a measured container's opening axis before moving across it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Extraction {
    /// The enclosing open box's object id in the kept look.
    pub container: u32,
    /// Displacement in the robot's frame for [`batchplan::Pick::lift`].
    pub lift: Vec3,
}

impl Kept {
    /// Open boxes over whose measured opening the middle of object `id` stands, the object
    /// reaching below the lip, and the straight displacement that would put its lowest point
    /// `margin` meters above each lip. A body's measured footprint and an opening's measured
    /// edge lie where they do to a few millimeters, and a bowl set down against a drawer's back
    /// reads a hair past it: it is in the drawer all the same, and must rise over the lip.
    /// Whether the straight way up is free is the lift's own collision world's to say. An
    /// absent target, or one with no readings in this look, returns an empty list. The
    /// container must also have readings, and the object must reach into the box above its
    /// floor. Several openings hold a body in a box that stands in another: take the longest of
    /// their lifts. Results follow the look's object order. A lift is a geometric request
    /// at the object's measured attitude. For a body measured standing on this container, a floor
    /// discrepancy of one single-precision step across the box height counts as contact. A fresh
    /// fit can also establish contact by its standing height on an upright floor before encoding.
    /// Plan it with the step's collision world.
    pub fn extractions(&self, id: u32, margin: f32) -> Result<Vec<Extraction>> {
        ensure_input!(
            margin.is_finite() && margin >= 0.0,
            "invalid extraction margin {margin}"
        );
        let scene = self.scene();
        let Some(target) = scene.objects.iter().position(|o| o.id == id) else {
            return Ok(Vec::new());
        };
        let visible = |index| {
            scene
                .labels
                .iter()
                .any(|v| v.contains(&Label::Object(index)))
        };
        if !visible(target) {
            return Ok(Vec::new());
        }
        let body = &scene.objects[target].shape;
        let witnessed = self.floors.contains(&target);
        let mut found = Vec::new();
        for (index, container) in scene.objects.iter().enumerate() {
            let support = match scene.objects[target].on {
                Some(Support::Object(i)) if i == index && witnessed => Floor::Witnessed,
                Some(Support::Object(i)) if i == index => Floor::Support,
                _ => Floor::Other,
            };
            if index != target
                && let Some(lift) = lift_out(&container.shape, body, margin, support)
                && visible(index)
            {
                found.push(Extraction {
                    container: container.id,
                    lift,
                });
            }
        }
        Ok(found)
    }

    /// Objects that hold object `id` in with no way out measured: what it stands on, and what
    /// that stands on, where it is a box not solid or open at the top, the object's middle lies
    /// over the box, its lowest point more than `margin` under the box's top, and
    /// [`Kept::extractions`] gives no opening of that box. A bowl in a drawer seen from across
    /// the room: the cabinet reads as a box that only names the room it takes up, the drawer and
    /// the bowl in it, and no look has shown it open. A lift that does not clear such a box ends
    /// inside it. Look closer, from where the drawer shows as open at the top, and its way out
    /// is measured. Empty for an absent target, or one with no readings in this look. Results
    /// follow the way down from the object.
    pub fn enclosing(&self, id: u32, margin: f32) -> Result<Vec<u32>> {
        let openings = self.extractions(id, margin)?;
        let scene = self.scene();
        let Some(target) = scene.objects.iter().position(|o| o.id == id) else {
            return Ok(Vec::new());
        };
        let seen = scene.labels.iter();
        if !seen.into_iter().any(|v| v.contains(&Label::Object(target))) {
            return Ok(Vec::new());
        }
        let body = &scene.objects[target].shape;
        let mut found = Vec::new();
        let mut under = scene.objects[target].on;
        // Each object stands on one thing: a way down longer than the scene has objects is none.
        for _ in 0..scene.objects.len() {
            let Some(Support::Object(index)) = under else {
                break;
            };
            let container = &scene.objects[index];
            let hollow = match container.shape {
                Shape::OpenBox { .. } => true,
                Shape::Cuboid { .. } => !container.solid,
                Shape::Cylinder { .. } => false,
            };
            let measured = openings.iter().any(|o| o.container == container.id);
            if hollow && !measured && holds_in(&container.shape, body, margin) {
                found.push(container.id);
            }
            under = container.on;
        }
        Ok(found)
    }
}

/// Whether `body`'s middle lies over the box of `container` with its lowest point more than
/// `margin` under the box's top and its highest over the box's bottom.
fn holds_in(container: &Shape, body: &Shape, margin: f32) -> bool {
    let (center, half_extents, rotation) = match *container {
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
        } => (center, half_extents, rotation),
        Shape::Cylinder { .. } => return false,
    };
    let (position, half) = within(center, rotation, body);
    let reach = half_extents.as_dvec3();
    position.truncate().abs().cmplt(reach.truncate()).all()
        && position.z - half.z < reach.z - f64::from(margin)
        && position.z + half.z > -reach.z
}

/// `body`'s middle in the frame at `center` turned by `rotation`, and half its extent along each
/// of that frame's axes.
fn within(center: Vec3, rotation: Quat, body: &Shape) -> (DVec3, DVec3) {
    let pose = body.pose();
    let axes = rotation.as_dquat().normalize();
    let position = axes.inverse() * (pose.position.as_dvec3() - center.as_dvec3());
    let into_body = pose.rotation.as_dquat().normalize().inverse() * axes;
    let extent = |axis: DVec3| {
        let direction = into_body * axis;
        match *body {
            Shape::Cuboid { half_extents, .. } | Shape::OpenBox { half_extents, .. } => {
                direction.abs().dot(half_extents.as_dvec3())
            }
            Shape::Cylinder {
                radius,
                half_height,
                ..
            } => {
                f64::from(radius) * direction.truncate().length()
                    + f64::from(half_height) * direction.z.abs()
            }
        }
    };
    let half = DVec3::new(extent(DVec3::X), extent(DVec3::Y), extent(DVec3::Z));
    (position, half)
}

#[derive(Clone, Copy, PartialEq)]
enum Floor {
    Other,
    Support,
    Witnessed,
}

fn lift_out(container: &Shape, body: &Shape, margin: f32, support: Floor) -> Option<Vec3> {
    let Shape::OpenBox {
        center,
        half_extents,
        rotation,
        ..
    } = *container
    else {
        return None;
    };
    let axes = rotation.as_dquat().normalize();
    let (position, half) = within(center, rotation, body);
    let (floor, upper) = container.room();
    let bottom = position - half;
    let top = position + half;
    // A body flush with a face can land on either side by the roundoff of the frame change.
    let roundoff = 32.0 * f64::EPSILON * (position.abs() + half).max_element();
    let lip = f64::from(half_extents.z);
    // A measured support and its body store their heights separately in f32. At floor contact
    // allow one representable step across the box height, independent of its distance from the
    // origin.
    let floor_step = if support != Floor::Other && top.z > f64::from(floor.z) {
        2.0 * f64::from(half_extents.z - half_extents.z.next_down())
    } else {
        0.0
    };
    let middle = position.truncate();
    let inside = middle.cmpgt(floor.as_dvec3().truncate() + roundoff).all()
        && middle.cmplt(upper.as_dvec2() - roundoff).all()
        && bottom.z < lip - roundoff
        && (bottom.z >= f64::from(floor.z) - floor_step - roundoff
            || (support == Floor::Witnessed && top.z > f64::from(floor.z)));
    inside.then(|| (axes * DVec3::Z * (lip - bottom.z + f64::from(margin))).as_vec3())
}

#[cfg(test)]
mod tests {
    use batchplan::Pose;
    use glam::Vec2;

    use super::*;

    #[test]
    fn a_floor_reconstructed_from_single_precision_heights_can_be_one_step_below() {
        for scale in [0.25, 1.0, 4.0] {
            for yaw in [-0.6, 0.6] {
                let rotation = Quat::from_rotation_z(yaw);
                let center = scale * Vec3::new(0.7, -0.2, 0.104021534);
                let half = scale * Vec3::new(0.18, 0.12, 0.11602153);
                let floor = scale * 0.047374427;
                let container = Shape::OpenBox {
                    center,
                    rotation,
                    half_extents: half,
                    room_lower: Vec3::new(-half.x, -half.y, floor),
                    room_upper: half.truncate(),
                };
                let middle = scale * 0.17728335;
                let height = scale * 0.0258874;
                let gap = f64::from(middle)
                    - f64::from(height)
                    - (f64::from(center.z) + f64::from(floor));
                assert!(
                    gap < 0.0,
                    "the stored bottom must lie below the stored floor"
                );
                let body = Shape::Cylinder {
                    center: center.with_z(middle),
                    rotation,
                    radius: scale * 0.056,
                    half_height: height,
                };
                assert!(lift_out(&container, &body, 0.0, Floor::Other).is_none());
                let lift = lift_out(&container, &body, 0.0, Floor::Support)
                    .expect("flush at fitted precision");
                let expected = center.z + half.z - (middle - height);
                assert!((lift.z - expected).abs() <= f32::EPSILON * expected);
                let overhang = body.placed(Pose::new(
                    body.pose().position + rotation * Vec3::Y * (1.01 * half.y),
                    rotation,
                ));
                assert!(lift_out(&container, &overhang, 0.0, Floor::Witnessed).is_none());
                let past = rotation * Vec3::Y * (1.01 * half.y);
                for offset in [Vec3::NEG_Z * (scale * 1e-6), past] {
                    let pose = body.pose();
                    let outside = body.placed(Pose::new(pose.position + offset, pose.rotation));
                    assert!(lift_out(&container, &outside, 0.0, Floor::Support).is_none());
                }
            }
        }
    }

    #[test]
    fn a_box_holds_in_what_stands_under_its_top_over_its_middle() {
        let rotation = Quat::from_rotation_z(0.4);
        let cabinet = Shape::Cuboid {
            center: Vec3::new(0.7, -0.2, 0.1),
            half_extents: Vec3::new(0.12, 0.18, 0.11),
            rotation,
        };
        let bowl = |at: Vec3| Shape::Cylinder {
            center: cabinet.pose().position + rotation * at,
            rotation: Quat::IDENTITY,
            radius: 0.056,
            half_height: 0.025,
        };
        // In a drawer of it, 7 cm under its top; and against the drawer's side, its measured
        // footprint past the box's.
        assert!(holds_in(&cabinet, &bowl(Vec3::new(0.0, 0.05, 0.065)), 0.01));
        assert!(holds_in(&cabinet, &bowl(Vec3::new(0.1, 0.05, 0.065)), 0.01));
        // On top of it, to within the margin; beside it; and under it.
        assert!(!holds_in(&cabinet, &bowl(Vec3::new(0.0, 0.05, 0.13)), 0.01));
        assert!(!holds_in(
            &cabinet,
            &bowl(Vec3::new(0.13, 0.05, 0.065)),
            0.01
        ));
        assert!(!holds_in(
            &cabinet,
            &bowl(Vec3::new(0.0, 0.05, -0.14)),
            0.01
        ));
        // A cylinder holds nothing in: it is no box.
        let can = bowl(Vec3::ZERO);
        assert!(!holds_in(&can, &bowl(Vec3::ZERO), 0.01));
    }

    #[test]
    fn a_body_stands_over_the_opening_and_its_lowest_point_clears_the_lip() {
        let rotation = Quat::from_rotation_x(0.27);
        let container = Shape::OpenBox {
            center: Vec3::new(0.3, 0.2, 0.4),
            rotation,
            half_extents: Vec3::new(0.10, 0.09, 0.04),
            room_lower: Vec3::new(-0.09, -0.08, -0.035),
            room_upper: Vec2::new(0.09, 0.08),
        };
        let tilt = 0.38_f32;
        let body = Shape::Cylinder {
            center: container.pose().position,
            rotation: rotation * Quat::from_rotation_x(tilt),
            radius: 0.03,
            half_height: 0.018,
        };
        let expected = 0.04 + 0.03 * tilt.sin() + 0.018 * tilt.cos() + 0.005;
        let lift =
            lift_out(&container, &body, 0.005, Floor::Other).expect("the cylinder lies inside");
        assert!(lift.distance(rotation * Vec3::Z * expected) < 1e-6);
        let moved = |offset: Vec3| {
            let pose = body.pose();
            body.placed(Pose::new(pose.position + rotation * offset, pose.rotation))
        };
        // Its footprint 1.5 cm past the opening's edge, its middle still over the opening: in
        // the box, and lifted as far.
        let against = lift_out(&container, &moved(Vec3::X * 0.075), 0.005, Floor::Other)
            .expect("its middle is over the opening");
        assert!(against.distance(lift) < 1e-6);
        // Its middle past the edge, under the floor, or over the lip: not in it.
        let just_below_floor = -0.035 + 0.03 * tilt.sin() + 0.018 * tilt.cos() - 1e-6;
        for offset in [Vec3::X * 0.091, Vec3::Z * just_below_floor, Vec3::Z * 0.1] {
            assert!(
                lift_out(&container, &moved(offset), 0.005, Floor::Support).is_none(),
                "{offset}"
            );
        }
    }
}
