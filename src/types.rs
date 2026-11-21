//! Standard data types. Every algorithm takes and returns these (or plain row-major arrays with
//! the documented shapes) instead of hiding behind plugin interfaces.

use std::f32::consts::FRAC_PI_2;

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::error::{Result, ensure_input};
use crate::world;

/// A rigid placement: a position, then a rotation about it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    /// Meters, in the frame the pose is given in.
    pub position: Vec3,
    /// A unit quaternion, from this pose's axes to that frame's.
    pub rotation: Quat,
}

impl Pose {
    /// No translation, no rotation.
    pub const IDENTITY: Pose = Pose {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// At `position`, turned by `rotation`.
    #[must_use]
    pub const fn new(position: Vec3, rotation: Quat) -> Pose {
        Pose { position, rotation }
    }

    /// `other`, given in this pose's frame, in the frame this pose is given in.
    #[must_use]
    pub fn mul_pose(self, other: Pose) -> Pose {
        Pose {
            position: self.position + self.rotation * other.position,
            rotation: self.rotation * other.rotation,
        }
    }

    /// Whether it is finite with a unit rotation (to 1e-3).
    pub(crate) fn is_unit(&self) -> bool {
        self.position.is_finite()
            && self.rotation.is_finite()
            && (self.rotation.length() - 1.0).abs() < 1e-3
    }

    /// The pose with its rotation made unit ([`world::unit`]), for where it places geometry.
    pub(crate) fn rigid(self) -> Pose {
        Pose::new(self.position, world::unit(self.rotation))
    }

    /// The frame this pose is given in, in this pose's frame.
    #[must_use]
    pub fn inverse(self) -> Pose {
        let rotation = self.rotation.inverse();
        Pose {
            position: rotation * -self.position,
            rotation,
        }
    }
}

/// A problem that has a solution: its position in the batch, the problem itself and its best
/// solution (a configuration for IK, a path's `[points, dof]` B-spline control points for
/// planning).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Solved<'result, P> {
    /// The problem's position in the batch.
    pub index: usize,
    /// The problem as given.
    pub problem: &'result P,
    /// Its best solution.
    pub solution: &'result [f32],
}

/// A batch of joint-space paths with a shared control-point count, row-major `[len, points, dof]`,
/// as trajectory optimization runs them. Each path is a uniform cubic B-spline over its control
/// points (see [`crate::timing`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct JointPaths {
    /// Joints per configuration.
    pub(crate) dof: usize,
    /// Control points per path.
    pub(crate) points: usize,
    /// Every path's control points, `[len, points, dof]`.
    pub(crate) positions: Vec<f32>,
}

impl JointPaths {
    pub(crate) fn zeros(len: usize, points: usize, dof: usize) -> Self {
        Self {
            dof,
            points,
            positions: vec![0.0; len * points * dof],
        }
    }

    /// The number of paths.
    pub(crate) fn len(&self) -> usize {
        self.positions.len() / (self.points * self.dof)
    }

    /// Path `i` as `[points, dof]`.
    pub(crate) fn path(&self, i: usize) -> &[f32] {
        let n = self.points * self.dof;
        &self.positions[i * n..(i + 1) * n]
    }

    pub(crate) fn path_mut(&mut self, i: usize) -> &mut [f32] {
        let n = self.points * self.dof;
        &mut self.positions[i * n..(i + 1) * n]
    }
}

/// How the robot is already moving where a plan starts: joint velocity and acceleration.
#[derive(Clone, Debug, PartialEq)]
pub struct StartMotion {
    /// Per joint, rad/s (m/s for prismatic joints).
    pub velocity: Vec<f32>,
    /// Per joint, rad/s² (m/s²).
    pub acceleration: Vec<f32>,
}

/// One trajectory sampled every `dt` seconds, row-major `[len, dof]`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointTrajectory {
    /// Joints per sample.
    pub dof: usize,
    /// Seconds between samples.
    pub dt: f32,
    /// Seconds from the first sample to the last, which lands exactly on the motion's end (so
    /// the last interval may be shorter than `dt`).
    pub duration: f32,
    /// `[len, dof]`, radians (meters for prismatic joints).
    pub positions: Vec<f32>,
    /// `[len, dof]`, per second.
    pub velocities: Vec<f32>,
    /// `[len, dof]`, per second squared.
    pub accelerations: Vec<f32>,
}

impl JointTrajectory {
    /// The number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len() / self.dof
    }

    /// Whether there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// A line a thing turns about, and how far it moves along the line as it turns: a knob or a valve's
/// wheel turns about its axis, a lever about its hinge, and a cap about its axis as its thread
/// carries it along.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Screw {
    /// A point on the line (meters), in the frame the screw is given in.
    pub point: Vec3,
    /// The line's direction, a unit vector: a positive angle turns right-handed about it.
    pub direction: Vec3,
    /// How far a thing carried about the line moves along `direction` for each radian it turns
    /// (meters): 0 for a turn about an axis. A cap of a right-handed thread 2 mm a turn, turned
    /// open (anticlockwise seen from above, about a `direction` up) rises 2 mm / 2π a radian.
    pub pitch: f32,
}

impl Screw {
    /// About the line through `point` along `direction`, moving nothing along it.
    #[must_use]
    pub const fn about(point: Vec3, direction: Vec3) -> Self {
        Self {
            point,
            direction,
            pitch: 0.0,
        }
    }

    /// `pose` carried `angle` radians about the line, right-handed about `direction` (taken as
    /// unit), and `pitch * angle` along it.
    #[must_use]
    pub fn carry(&self, pose: Pose, angle: f32) -> Pose {
        let axis = self.direction.normalize();
        let turn = Quat::from_axis_angle(axis, angle);
        let from = pose.position - self.point;
        Pose::new(
            self.point + turn * from + axis * (self.pitch * angle),
            turn * pose.rotation,
        )
    }

    /// The poses a frame at `pose` passes through carried `angle` radians about the line, after
    /// `pose` and up to it carried all the way ([`carry`](Self::carry)), as
    /// [`ToolPath::waypoints`](crate::ToolPath::waypoints) takes them: near enough together that
    /// the straight moves [`follow`](crate::follow) makes between them stray at most `precision`
    /// (meters) from the frame's true path.
    ///
    /// The frame lies `r` from the line, and the waypoints turn it by equal angles `Δ`. A straight
    /// move between two of them strays from the circle they lie on by its sag, `r (1 − cos(Δ/2))`,
    /// at its middle, and no more from a helix (along the line both move alike), so `Δ` is at most
    /// `4 asin(√(precision / 2r))`. It turns by the shortest rotation between its ends, the screw's
    /// own only under half a turn, so `Δ` is also at most a quarter turn, that far from where
    /// rounding could make it the other way round. The count is the fewest that keep both: a frame
    /// on the line needs only what its rotation needs. An `Error::Input` for a screw that is not
    /// finite or whose direction is not unit (to 1e-3), a pose not finite with a unit rotation, an
    /// angle that is not finite, or a precision not above 0, each with the value given.
    pub fn path(&self, pose: Pose, angle: f32, precision: f32) -> Result<Vec<Pose>> {
        self.check()?;
        ensure_input!(
            pose.is_unit(),
            "a pose carried by a screw must be finite with a unit rotation, got {pose:?}"
        );
        ensure_input!(
            angle.is_finite(),
            "a screw turns by a finite angle, got {angle}"
        );
        ensure_input!(
            precision > 0.0,
            "a screw's path needs a precision above 0, got {precision} m"
        );
        let off = (pose.position - self.point).reject_from(self.direction);
        let sagging = 4.0 * (precision / (2.0 * off.length())).sqrt().min(1.0).asin();
        let pieces = (angle.abs() / sagging).max(angle.abs() / FRAC_PI_2);
        let count = (pieces.ceil() as usize).max(1);
        let each = (1..=count).map(|k| self.carry(pose, angle * k as f32 / count as f32));
        Ok(each.collect())
    }

    /// That the screw is finite with a unit direction (to 1e-3).
    pub(crate) fn check(&self) -> Result<()> {
        ensure_input!(
            self.point.is_finite() && self.direction.is_finite() && self.pitch.is_finite(),
            "a screw must be finite, got {self:?}"
        );
        ensure_input!(
            (self.direction.length() - 1.0).abs() < 1e-3,
            "a screw's direction must be unit, got {} of length {}",
            self.direction,
            self.direction.length()
        );
        Ok(())
    }
}
