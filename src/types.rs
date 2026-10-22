//! Standard data types. Every algorithm takes and returns these (or plain row-major arrays with
//! the documented shapes) instead of hiding behind plugin interfaces.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

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

/// A batch of joint-space paths with a shared control-point count, row-major `[len, points, dof]`.
/// Each path is a uniform cubic B-spline over its control points (see [`crate::timing`]).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct JointPaths {
    /// Joints per configuration.
    pub dof: usize,
    /// Control points per path.
    pub points: usize,
    /// Every path's control points, `[len, points, dof]`.
    pub positions: Vec<f32>,
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
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len() / (self.points * self.dof)
    }

    /// Whether there are no paths.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Path `i` as `[points, dof]`.
    #[must_use]
    pub fn path(&self, i: usize) -> &[f32] {
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
