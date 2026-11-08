//! Objects a hand holds, objects attached to a robot, and where hands must stay along a path.

use batchplan::glam::Vec3;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::OrRaise;
use crate::field::{Field, tuple_field, value_class};
use crate::robot::Robot;
use crate::world::{Obstacle, Pose};

value_class! {
    /// An object a hand holds: `TaskPlan.held` after a pick, and what `PlanProblem.holding`,
    /// `IkProblem.holding` and `ToolPath.holding` take.
    Held(batchplan::Held) {
        /// The hand holding it.
        hand: usize,
        /// Its shape in its own frame: centered at the origin, unrotated.
        object: batchplan::Obstacle,
        /// The hand's IK frame in the object's frame: the grasp that holds it.
        grasp: batchplan::Pose,
    }
    methods {
        #[new]
        fn new(hand: usize, object: &Obstacle, grasp: &Pose) -> Self {
            Self(batchplan::Held {
                hand,
                object: object.0.clone(),
                grasp: grasp.0,
            })
        }

        /// The object as `robot` holds it, for `Robot.attach`: fixed to the hand's IK frame as
        /// the link `"held by hand {hand}"`, and free to touch the hand's links
        /// (`Robot.hand_links`).
        fn attachment(&self, robot: &Robot) -> PyResult<AttachedObject> {
            self.0.attachment(&robot.0).or_raise().map(AttachedObject)
        }
    }
}

value_class! {
    /// An object fixed to one of a robot's links, for `Robot.attach`: checked as part of the
    /// robot, and never with `link` or `touch_links` (such as the fingers holding it).
    AttachedObject(batchplan::AttachedObject) {
        /// The object's frame name, which must not name a link of the robot.
        name: String,
        /// The link holding it.
        link: String,
        /// The object's shapes, placed in `link`'s frame. Distance grids are not supported.
        shapes: Vec<batchplan::Obstacle>,
        /// Other links it may touch.
        touch_links: Vec<String>,
        /// How spheres are fitted to the shapes; a held object rarely needs more than 8 to 16. A
        /// single shape is fitted in its own frame, so where it sits on `link` does not change its
        /// spheres.
        spheres: batchplan::SphereOptions,
    }
    methods {
        #[new]
        #[pyo3(signature = (name, link, shapes, **changes))]
        fn new(
            name: String,
            link: String,
            shapes: &Bound<'_, PyAny>,
            changes: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Self> {
            let object = batchplan::AttachedObject {
                name,
                link,
                shapes: Vec::from_py(shapes)?,
                touch_links: Vec::new(),
                spheres: batchplan::SphereOptions::default(),
            };
            Ok(Self(Self::changed(object, changes)?))
        }
    }
}

value_class! {
    /// Where hand `hand` may be at every point of a path: its origin within `position` of
    /// `pose.position` along each axis, its z axis within `tilt` of `pose`'s z axis, and turned
    /// about that axis within `twist` of `pose`'s turn, all in the world or, with `relative_to`, in
    /// another hand's frame. A new one leaves all three free; set the ones that matter: a tray
    /// carried upright is `HandConstraint(0, pose, tilt=0.17)`.
    HandConstraint(batchplan::HandConstraint) {
        /// The hand (`RobotOptions.ee_links`).
        hand: usize,
        /// The hand whose frame `pose` and the bounds are in, moving with it; `None` for the
        /// world.
        relative_to: Option<usize>,
        /// The pose the bounds are measured from.
        pose: batchplan::Pose,
        /// How far the hand's origin may be from `pose.position` along each axis of the world, or
        /// of the hand it is relative to (meters): a box. Infinity leaves an axis free.
        position: Vec3,
        /// How far the hand's z axis may tilt from `pose`'s z axis (radians): a cone. π or more
        /// leaves it free.
        tilt: f32,
        /// How far the hand may turn about its z axis from `pose`'s turn, once its tilt is taken
        /// out (radians). π or more leaves it free. A bounded twist needs a tilt bound of at most
        /// π/2.
        twist: f32,
    }
    methods {
        /// Hand `hand` measured from `pose` in the world, with every bound free but those given.
        #[new]
        #[pyo3(signature = (hand, pose, **changes))]
        fn new(hand: usize, pose: &Pose, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            Ok(Self(Self::changed(
                batchplan::HandConstraint::new(hand, pose.0),
                changes,
            )?))
        }
    }
}

tuple_field!(batchplan::HandConstraint, batchplan::Pose);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Held>()?;
    m.add_class::<AttachedObject>()?;
    m.add_class::<HandConstraint>()?;
    Ok(())
}
