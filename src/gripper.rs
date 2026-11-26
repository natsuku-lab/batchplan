//! One hand of a robot as a parallel-jaw gripper ([`Gripper`], from
//! [`Robot::gripper`](crate::Robot::gripper)): the widths a caller measures of it once
//! ([`GripperCalibration`]), its spheres, and how its finger joints move between closed and
//! loaded, for every call that grasps with it.

use std::collections::HashMap;

use crate::robot::HandSphere;

/// What a caller measures of a parallel-jaw hand, once, for
/// [`Robot::gripper`](crate::Robot::gripper).
#[derive(Clone, Debug, PartialEq)]
pub struct GripperCalibration {
    /// How far apart the pads stand as the robot's locked joints hold the fingers (meters).
    pub loaded_gap: f32,
    /// How far apart the pads stand with the finger joints at `closed` (meters).
    pub closed_gap: f32,
    /// The widest thing the hand grasps (meters), at most `loaded_gap`: `None`, the loaded gap.
    pub capacity: Option<f32>,
    /// The finger joints at the closed command, by name, each a joint the robot was loaded with
    /// locked (`RobotOptions::lock_joints`); none where no finger moves.
    pub closed: HashMap<String, f32>,
    /// The friction between a pad and what it presses on: 0.5, as Dex-Net samples grasps with. A
    /// grasp of a surface ([`grasps_on`](crate::grasps_on)) closes along a line within
    /// `atan(friction)` of the normal of each patch it presses on; less asks for patches that face
    /// each other more squarely.
    pub friction: f32,
}

impl GripperCalibration {
    /// A hand whose pads stand `loaded_gap` apart as loaded and `closed_gap` apart with its finger
    /// joints at `closed`, grasping all it opens to (`capacity: None`), its pads' friction 0.5.
    #[must_use]
    pub fn new(loaded_gap: f32, closed_gap: f32, closed: HashMap<String, f32>) -> Self {
        Self {
            loaded_gap,
            closed_gap,
            capacity: None,
            closed,
            friction: 0.5,
        }
    }
}

/// One hand of a robot as a parallel-jaw gripper, from [`Robot::gripper`](crate::Robot::gripper):
/// built once, taken by every call that grasps with it ([`grasps`](crate::grasps),
/// [`grasps_on`](crate::grasps_on), [`Grasp::closes_on`](crate::Grasp::closes_on),
/// [`SurfaceGraspOptions::order_for_turn`](crate::SurfaceGraspOptions::order_for_turn),
/// [`Grasp::moment`](crate::Grasp::moment)).
#[derive(Clone, Debug)]
pub struct Gripper {
    pub(crate) hand: usize,
    /// The loaded gap.
    pub(crate) opening: f32,
    pub(crate) closed_gap: f32,
    pub(crate) capacity: f32,
    pub(crate) friction: f32,
    pub(crate) sliding: bool,
    pub(crate) spheres: Vec<HandSphere>,
    pub(crate) tip: f32,
    /// Each joint the calibration closes, by name: its value closed and as loaded. Sorted, so
    /// that what is read of several joints does not depend on a map's order.
    pub(crate) joints: Vec<(String, [f32; 2])>,
}

impl Gripper {
    /// The robot's hand it is.
    #[must_use]
    pub fn hand(&self) -> usize {
        self.hand
    }

    /// How far apart the pads stand as the robot's locked joints hold the fingers (meters): the
    /// calibration's loaded gap, or its capacity where the two are one floating-point step apart,
    /// so that grasps, their closing regions and commands share one value.
    #[must_use]
    pub fn opening(&self) -> f32 {
        self.opening
    }

    /// How far apart the pads stand at the closed command (meters).
    #[must_use]
    pub fn closed_gap(&self) -> f32 {
        self.closed_gap
    }

    /// The widest thing the hand grasps (meters), at most [`opening`](Self::opening).
    #[must_use]
    pub fn capacity(&self) -> f32 {
        self.capacity
    }

    /// The openings a grasp of a surface may come in at where
    /// [`SurfaceGraspOptions::narrow`](crate::SurfaceGraspOptions::narrow) lets it (meters), widest
    /// first: the loaded gap, then 15/16 of the capacity and each sixteenth of it less, to 1/16. A
    /// place may let go at the same ([`Place::open`](crate::Place::open)), each as the finger
    /// joints' values ([`locked_at`](Self::locked_at)).
    #[must_use]
    pub fn openings(&self) -> Vec<f32> {
        crate::surface::openings(self.opening, self.capacity)
    }

    /// The friction between a pad and what it presses on.
    #[must_use]
    pub fn friction(&self) -> f32 {
        self.friction
    }

    /// Whether the hand's joints, jaws and spheres move as opposed jaws sliding along its y axis
    /// between the closed and loaded gaps, as
    /// [`SurfaceGraspOptions::narrow`](crate::SurfaceGraspOptions::narrow)
    /// takes them to ([`Robot::gripper`](crate::Robot::gripper) says what is checked).
    #[must_use]
    pub fn sliding(&self) -> bool {
        self.sliding
    }

    /// The hand's collision spheres in its IK frame, as
    /// [`Robot::hand_spheres`](crate::Robot::hand_spheres) gave them when the gripper was built:
    /// what its grasps are checked with.
    #[must_use]
    pub fn spheres(&self) -> &[HandSphere] {
        &self.spheres
    }

    /// How far the hand's spheres reach past its IK frame along the frame's z axis (meters), as
    /// [`Robot::tip`](crate::Robot::tip) measured it when the gripper was built: how far a shape's
    /// grasps take the fingertips past the frame where their options name no `tip`
    /// ([`ShapeGraspOptions::tip`](crate::ShapeGraspOptions::tip)).
    #[must_use]
    pub fn tip(&self) -> f32 {
        self.tip
    }

    /// The finger joints' values at which the pads stand `opening` apart (meters), as
    /// [`Pick::locked`](crate::Pick::locked), [`Place::open`](crate::Place::open) and
    /// [`Robot::relock`](crate::Robot::relock) take them: each joint the calibration closes,
    /// between its closed value at the closed gap and its loaded value at the loaded gap, in
    /// proportion, whether or not the jaws slide. An opening outside the two gaps is taken at the
    /// nearer. Joints the calibration does not name, or names at their loaded values, are left
    /// out: they stay as loaded.
    #[must_use]
    pub fn locked_at(&self, opening: f32) -> HashMap<String, f32> {
        let share =
            ((opening - self.closed_gap) / (self.opening - self.closed_gap)).clamp(0.0, 1.0);
        let each = self.joints.iter().map(|(name, [closed, loaded])| {
            (name.clone(), closed * (1.0 - share) + loaded * share)
        });
        each.collect()
    }

    /// How far apart the pads stand (meters) with the finger joints at `locked`, as
    /// [`locked_at`](Self::locked_at) relates them: where each joint the calibration closes stands
    /// between its closed and loaded values, the mean of those, between the closed and the loaded
    /// gap. A joint `locked` does not name stands as loaded, and joints the calibration does not
    /// close are not read; an opening outside the two gaps is taken at the nearer.
    #[must_use]
    pub fn opening_of(&self, locked: &HashMap<String, f32>) -> f32 {
        let shares = self.joints.iter().map(|(name, [closed, loaded])| {
            locked
                .get(name)
                .map_or(1.0, |value| (value - closed) / (loaded - closed))
        });
        let (sum, count) = shares.fold((0.0, 0), |(sum, count), share| (sum + share, count + 1));
        let share = match count {
            0 => 1.0,
            _ => (sum / count as f32).clamp(0.0, 1.0),
        };
        self.closed_gap * (1.0 - share) + self.opening * share
    }
}
