//! RRT-Connect (Kuffner and LaValle, 2000) as VAMP runs it (Thomason, Kingston and Kavraki,
//! 2024). The straight line to each goal is tried first. If it collides, it is bent: a little off
//! the line and then farther, its colliding stretch is moved aside (via points beside where it
//! first and last collides), which keeps the straight route when an obstacle only grazes it. Then
//! one tree grows from the start and
//! one from the goal set, in turn (the smaller one alone while the other is at least twice its
//! size), toward Halton samples: first from a tube around the straight line that widens until it
//! spans the joint ranges, as cuRobo's roadmap samples an ellipsoid around it, so the route found
//! stays near the line where one does; then from every joint's range. An extension steps at most
//! `range` from the nearest node and is checked as an edge. The other tree then connects toward the
//! new node in equal steps of at most `range`, each checked in order, inserting every valid one.
//! Dynamic domains skip samples farther from their nearest node than its radius, which shrinks when
//! extending from the node fails. Continuous joints step and measure the short way round.
//!
//! Each problem is its own search: device threads run searches whole, checking their edges with
//! the CPU's SIMD collision checks (see `edges`), so results do not depend on the batch or device.
//! Threads a batch has no more searches for check the edges a long search will ask about next
//! (`ahead`), which speeds it without changing what it finds.

mod ahead;
mod connect;

pub use connect::{RrtOptions, RrtProblem, RrtResult, connect};

pub(crate) use ahead::{Ahead, HELPERS};
pub(crate) use connect::{Connect, Sampling, distance, holds};
