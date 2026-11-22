//! An outer bound on a serial chain's reachable hand poses, including the tool orientation.

use glam::{DMat3, DVec3};

use crate::robot::{JointKind, Link, Transform};
use crate::{IkOptions, Pose};

/// A necessary condition for a hand to reach a pose. Built from the loaded chain by
/// [`crate::Robot::hand_reach`]. A pose inside the bound still needs ordinary IK.
#[derive(Clone, Debug)]
pub struct HandReach {
    shoulder: DVec3,
    arm: f64,
    /// The last joint's axis and the centre of the preceding joint's circle, in tool coordinates.
    axis: DVec3,
    center: DVec3,
    radius: f64,
    roundoff: f64,
}

impl HandReach {
    pub(crate) fn of<'a>(links: impl Iterator<Item = &'a Link>) -> Option<Self> {
        let mut joints = Vec::new();
        let mut frame = Transform::IDENTITY;
        let mut scale = 0.0;
        let mut count = 0;
        for link in links {
            let rotation = link.origin.rot.as_dmat3();
            if !link.origin.trans.is_finite()
                || !rotation.is_finite()
                || rotation.determinant() <= 0.0
                || !(rotation.transpose() * rotation)
                    .abs_diff_eq(DMat3::IDENTITY, 8.0 * f64::from(f32::EPSILON))
            {
                return None;
            }
            frame = frame.mul(&link.origin);
            count += 1;
            scale += link.origin.trans.as_dvec3().length();
            match link.joint {
                JointKind::Fixed => {}
                JointKind::Revolute { axis, .. } => {
                    // Some descriptions give a zero axis. Their FK is not a rotation about
                    // a line and cannot justify this certificate.
                    if !axis.is_finite() || (axis.length_squared() - 1.0).abs() > 8.0 * f32::EPSILON
                    {
                        return None;
                    }
                    joints.push((frame, axis.as_dvec3().normalize()));
                    frame = Transform::IDENTITY;
                }
                JointKind::Prismatic { .. } => return None,
            }
        }
        let (&(last, axis), arm_joints) = joints.split_last()?;
        let shoulder = arm_joints.first()?.0.trans.as_dvec3();

        // Bound the distance from the first joint to the penultimate joint. For each prefix,
        // either add the last link's length, or bound the last two links together over the
        // intervening joint's full circle, whichever gives the smaller upper bound.
        let mut arm = 0.0;
        let mut previous = 0.0;
        for i in 1..arm_joints.len() {
            let offset = arm_joints[i].0.trans.as_dvec3();
            let mut next = arm + offset.length();
            if i >= 2 {
                let (joint, axis) = arm_joints[i - 1];
                let incoming = joint.rot.as_dmat3().transpose() * joint.trans.as_dvec3();
                let along = (incoming + offset).dot(axis);
                let across = incoming.cross(axis).length() + offset.cross(axis).length();
                next = next.min(previous + along.hypot(across));
            }
            previous = arm;
            arm = next;
        }
        // Looking back from the tool, the penultimate joint lies on a circle as the last
        // revolute joint turns. This includes every point allowed by its actual joint limits.
        let offset = last.rot.as_dmat3().transpose() * last.trans.as_dvec3();
        let back = frame.rot.as_dmat3().transpose();
        let center = back * (-frame.trans.as_dvec3() - axis * offset.dot(axis));
        let radius = offset.cross(axis).length();
        // Bound compounded transform error even on long chains. When the product-error
        // estimate itself ceases to be finite, leave the ordinary solver in charge.
        let error = 64.0 * f64::from(f32::EPSILON) * f64::from(count);
        if error >= 1.0 {
            return None;
        }
        Some(Self {
            shoulder,
            arm,
            axis: (back * axis).normalize(),
            center,
            radius,
            // The loaded transforms and the solver's FK use f32. Grow by the accumulated
            // relative error of their matrix products at the scale of this particular chain.
            roundoff: (error / (1.0 - error)) * (1.0 + scale),
        })
    }

    /// True only when even the closest point of the wrist circle lies beyond the arm's
    /// maximum reach, allowing both of IK's pose tolerances. An unusable pose or tolerance
    /// supplies no bound.
    #[must_use]
    pub fn excludes(&self, target: Pose, options: &IkOptions) -> bool {
        // IK accepts slightly nonunit rotations and uses their unnormalized matrix. Only
        // apply the circle argument to rigid targets up to ordinary quaternion roundoff.
        if !target.is_unit()
            || (target.rotation.length_squared() - 1.0).abs() > 8.0 * f32::EPSILON
            || !(options.position_tolerance > 0.0 && options.rotation_tolerance > 0.0)
        {
            return false;
        }
        let turn = target.rotation.as_dquat().normalize();
        let to = (target.position.as_dvec3() - self.shoulder) + turn * self.center;
        let axis = turn * self.axis;
        let nearest = to.dot(axis).hypot(to.cross(axis).length() - self.radius);
        // A rotation through angle a moves a point by at most a times its distance from
        // the rotation centre, and never more than twice that distance.
        let rotation = f64::from(options.rotation_tolerance).min(2.0);
        let slack = f64::from(options.position_tolerance)
            + rotation * (self.center.length() + self.radius)
            + self.roundoff;
        nearest > self.arm + slack
    }
}
