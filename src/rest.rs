//! Where what a hand holds can be set down on what was seen ([`rests_on`]), the mirror of
//! [`grasps_on`](crate::grasps_on): rests of a box, ball, cylinder or capsule on the points seen of
//! the place it goes ([`Surface`](crate::Surface)), whatever that place's shape. And where it
//! can be let go to fall into a hollow of what was seen that holds it in ([`falls_into`]).

// Each hot function of the search says whether it is inlined, as one codegen unit has it, the
// quicker build, or where x86 and ARM want it otherwise, each its own: in sixteen units, as a
// crate that depends on this one builds it, the choice turned on where code fell among the units,
// and code added elsewhere made rests up to 9% slower.
mod contact;
mod fall;
mod field;
mod search;
mod shape;
mod statics;

pub use fall::{Fall, falls_into};
pub use search::{Blocking, Rest, RestFailure, RestOptions, RestProblem, rest_failures, rests_on};
