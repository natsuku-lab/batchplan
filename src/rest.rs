//! Where what a hand holds can be set down on what was seen ([`rests_on`]), the mirror of
//! [`grasps_on`](crate::grasps_on): rests of a box, ball, cylinder or capsule on the points seen of
//! the place it goes ([`Surface`](crate::Surface)), whatever that place's shape.

mod contact;
mod field;
mod search;
mod shape;
mod statics;

pub use search::{Rest, RestOptions, RestProblem, rests_on};
