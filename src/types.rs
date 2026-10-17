//! Standard data types. Every algorithm takes and returns these (or plain row-major arrays with
//! the documented shapes) instead of hiding behind plugin interfaces.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Pose {
    /// No translation, no rotation.
    pub const IDENTITY: Pose = Pose { position: Vec3::ZERO, rotation: Quat::IDENTITY };

    /// `other`, given in this pose's frame, in the frame this pose is given in.
    pub fn mul_pose(self, other: Pose) -> Pose {
        Pose { position: self.position + self.rotation * other.position, rotation: self.rotation * other.rotation }
    }

    /// The frame this pose is given in, in this pose's frame.
    pub fn inverse(self) -> Pose {
        let rotation = self.rotation.inverse();
        Pose { position: rotation * -self.position, rotation }
    }
}

/// A problem that has a solution: its position in the batch, the problem itself and its best
/// solution (a configuration for IK, a path's `[points, dof]` B-spline control points for planning).
#[derive(Clone, Copy, Debug)]
pub struct Solved<'a, P> {
    pub index: usize,
    pub problem: &'a P,
    pub solution: &'a [f32],
}

/// A batch of joint-space paths with a shared control-point count, row-major `[len, points, dof]`.
/// Each path is a uniform cubic B-spline over its control points (see [`crate::timing`]).
#[derive(Clone, Debug, PartialEq)]
pub struct JointPaths {
    pub dof: usize,
    pub points: usize,
    pub positions: Vec<f32>,
}

impl JointPaths {
    pub(crate) fn zeros(len: usize, points: usize, dof: usize) -> Self {
        Self { dof, points, positions: vec![0.0; len * points * dof] }
    }

    pub fn len(&self) -> usize {
        self.positions.len() / (self.points * self.dof)
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Path `i` as `[points, dof]`.
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
    pub velocity: Vec<f32>,
    pub acceleration: Vec<f32>,
}

/// One trajectory sampled every `dt` seconds, row-major `[len, dof]`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointTrajectory {
    pub dof: usize,
    pub dt: f32,
    pub duration: f32,
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub accelerations: Vec<f32>,
}

impl JointTrajectory {
    pub fn len(&self) -> usize {
        self.positions.len() / self.dof
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}
