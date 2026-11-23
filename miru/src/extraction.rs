//! Straight lifts out of openings measured in the same look as the object.

use glam::{DVec3, Vec3};

use crate::error::{Result, ensure_input};
use crate::memory::Kept;
use crate::scene::{Label, Shape};

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
    /// Open boxes whose measured opening contains the entire footprint of object `id`, and the
    /// straight displacement that would put its lowest point `margin` meters above each lip.
    /// An absent target, or one with no readings in this look, returns an empty list.
    /// The container must also have readings, and the object must reach into the
    /// box above its floor. Results follow the look's object order. A lift is a geometric request
    /// at the object's measured attitude. Plan it with the step's collision world.
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
        let mut found = Vec::new();
        for (index, container) in scene.objects.iter().enumerate() {
            if index != target
                && let Some(lift) = lift_out(&container.shape, body, margin)
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
}

fn lift_out(container: &Shape, body: &Shape, margin: f32) -> Option<Vec3> {
    let Shape::OpenBox {
        center,
        half_extents,
        rotation,
        ..
    } = *container
    else {
        return None;
    };
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
    let (floor, upper) = container.room();
    let bottom = position - half;
    let top = position + half;
    // A body flush with a face can land on either side by the roundoff of the frame change.
    let roundoff = 32.0 * f64::EPSILON * (position.abs() + half).max_element();
    let lip = f64::from(half_extents.z);
    let inside = bottom.cmpge(floor.as_dvec3() - roundoff).all()
        && top.truncate().cmple(upper.as_dvec2() + roundoff).all()
        && bottom.z < lip - roundoff;
    inside.then(|| (axes * DVec3::Z * (lip - bottom.z + f64::from(margin))).as_vec3())
}

#[cfg(test)]
mod tests {
    use batchplan::Pose;
    use glam::{Quat, Vec2};

    use super::*;

    #[test]
    fn a_body_fits_the_opening_and_its_lowest_point_clears_the_lip() {
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
        let lift = lift_out(&container, &body, 0.005).expect("the cylinder lies inside");
        assert!(lift.distance(rotation * Vec3::Z * expected) < 1e-6);
        let just_below_floor = -0.035 + 0.03 * tilt.sin() + 0.018 * tilt.cos() - 1e-6;
        for offset in [Vec3::X * 0.075, Vec3::Z * just_below_floor, Vec3::Z * 0.1] {
            let pose = body.pose();
            let outside = body.placed(Pose::new(pose.position + rotation * offset, pose.rotation));
            assert!(lift_out(&container, &outside, 0.005).is_none(), "{offset}");
        }
    }
}
