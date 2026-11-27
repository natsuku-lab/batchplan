//! Where to let go of an object over a hollow of the support that will hold it in
//! ([`falls_into`]): a can over a basket, a bowl over an open drawer.
//!
//! Let go there, the object does not rest where it was held: it falls, and comes to lie somewhere
//! inside. Where exactly takes following the fall. What needs none of that is that it stays
//! inside: under gravity an object is caged where it would have to be lifted to get out, by
//! more than it can be (Mahler et al., Energy-Bounded Caging, 2016). Here the object keeps the
//! turn it is let go at and moves over what was seen as a height: at each spot, how high it
//! lands let go straight down there, on the support or on anything else in the way. A hollow of
//! those heights is found as the hollows of a terrain are, by flooding it from its edge inward,
//! lowest first, which gives every spot the level it would have to rise to before it could
//! leave (Barnes et al., Priority-Flood, 2014). The object is held in where that level is more
//! than the height of its own middle above where it lands: its weight is then under the rim it
//! would have to cross.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::f32::consts::FRAC_1_SQRT_2;

use glam::{Quat, Vec2, Vec3};
use rayon::prelude::*;

use super::field::Field;
use super::search::{self, Read, RestOptions, RestProblem, Seen};
use super::shape::{Shape, Turned};
use crate::device::Device;
use crate::error::Result;
use crate::surface::Surface;
use crate::types::Pose;

/// A place [`falls_into`] found to let go at.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Fall {
    /// The object's frame where it is let go, in the surface's frame: where a place lets go of it
    /// ([`Release::from`]). Its lowest point is at least [`RestOptions::clearance`] over the
    /// lowest of the hollow's rim, and the object that far clear of all around it there: that far
    /// over the rim for a shape that stands level, unless something nearer stands higher, and
    /// more for one turned askew.
    ///
    /// [`Release::from`]: crate::Release
    pub pose: Pose,
    /// How far it drops from `pose`, straight down, to where it lands as it is (meters): onto what
    /// was seen under it, or where nothing was, as deep under the rim as anything was seen in the
    /// hollow (`sunk`). A place's [`Release::drop`](crate::Release::drop): more than the
    /// clearance and `sunk` together for a shape turned askew or let go over something that
    /// stands higher than the rim.
    pub drop: f32,
    /// How far under the lowest of the hollow's rim the object's bottom lands (meters): what it
    /// would have to be lifted by to leave, to within how it fits between what it lands on.
    /// Where nothing was seen under it there, as over a floor no camera saw, the most that was
    /// seen anywhere in the hollow.
    pub sunk: f32,
    /// How far it is let go from the problem's [`near`](RestProblem::near), in x and y (meters),
    /// by which the falls come nearest it first: 0 where the problem has none. A place's release
    /// takes it as its cost.
    pub cost: f32,
}

/// Letting the object go there to drop as it does, at the fall's cost.
impl From<&Fall> for crate::Release {
    fn from(fall: &Fall) -> Self {
        Self {
            pose: fall.pose,
            drop: fall.drop,
            cost: fall.cost,
        }
    }
}

/// For each of `problems`, in their order, poses to let go of its object at from which it falls
/// into a hollow of its surface's support and is held in: over a basket, a bin, an open drawer.
///
/// The object is not followed as it falls, and does not rest where it is let go; what is found
/// is that it cannot leave. Let go straight down over each spot [`RestOptions::spacing`] apart,
/// at each turn, the object lands at some height, on the support or on anything else there, as
/// what is already in a basket; a hollow is where it would have to rise from that height to get
/// out of the support any way, sliding over it as it is turned. It is held in where the
/// hollow's rim is higher over where it lands than its own middle is, so that its weight lies
/// under the rim. A hollow needs a floor, or a wall sloping in under the object, seen that far
/// down somewhere in it: what no camera saw is no floor, and an upright wall is read as its
/// top. Walls with a gap the object could pass through hold nothing in: whether it can leave is
/// asked of it worn down by a level disc of half the diagonal between spots, which also takes a
/// gap up to that much narrower than it for one it passes. An object that leaves less than
/// [`RestOptions::touching`] of itself so worn, any way, has no falls: it could slip between the
/// points seen of a solid wall. Space the spots nearer for it.
///
/// The spots lie within the rectangle, seen from above, that bounds the support's points, and a
/// hollow is found only where its rim lies inside it: a support that stands beside the hollow has
/// no fall into it, as the body of a cabinet where the drawer pulled out of it is another
/// object's. For a place into what stands at a point whatever objects were made of it, give every
/// reading about it as the support, none flagged, and the point as the problem's `near`.
///
/// It lands as it is, and [`RestOptions::clearance`] is kept where it is let go: over the rim
/// and over all it would land on there, so that the hand lets go of it clear of what is around.
/// A box that goes through a bowl's opening, and grown by the clearance would not, falls in.
///
/// With [`RestOptions::look_closer`], a hollow that holds the object in only nearer one place
/// than the spots lie apart, as a bowl a box barely fits, is looked at closer as rests are: the
/// spots a cell (`touching`) apart within the spacing of the problem's `near` are tried at
/// every turn, or without it those of the spot of a hollow where the object lands the deepest
/// under the rim, its lowest point, only where no spot holds it in at any turn. Falls found
/// there come after the spots' of their turn.
///
/// A problem's `object`, `surface` and `rotation` are read as [`rests_on`](super::rests_on) reads
/// them, its `near` to look closer and to put the falls nearest it first, and of the options
/// `turns`, `spacing`, `touching`, `clearance`, `look_closer` and `seen_from_above_holds`; the
/// problems and options are checked as `rests_on` checks them, `tipping` and `friction` too, though
/// falls read neither. The falls come at every turn, and within a turn those with the most room
/// around them first, and of those as roomy the nearest the middle of where it is held in, as the
/// middle of a basket before its sides; where the problem has a `near`, nearest it first instead,
/// in x and y, those as near in that order, each costing its distance ([`Fall::cost`]). The
/// problems run in parallel on the device's threads, each surface read once, and a problem's falls
/// are the same alone and among others.
pub fn falls_into(
    device: &Device,
    problems: &[RestProblem],
    surfaces: &[Surface],
    o: &RestOptions,
) -> Result<Vec<Vec<Fall>>> {
    let read = Read::checked(problems, surfaces, o)?;
    let each = || {
        let seen = read.seen(surfaces, o);
        let each = problems.par_iter().map(|p| match &seen[p.surface] {
            Some(seen) => into(p, seen, o),
            None => vec![],
        });
        each.collect()
    };
    Ok(device.install(each))
}

/// `p`'s falls into the surface it names, as `seen`.
fn into(p: &RestProblem, seen: &Seen, o: &RestOptions) -> Vec<Fall> {
    let shape = Shape::of(&p.object);
    let grown = shape.grown(o.clearance);
    let (turns, first) = search::turns(shape, p.rotation, o);
    let each = (0..turns.len()).into_par_iter().map(|k| {
        if first[k] != k {
            return (vec![], None);
        }
        at_turn(shape, grown, turns[k], seen, o)
    });
    let (mut found, hollows): (Vec<Vec<Fall>>, Vec<Option<Hollows>>) = each.unzip();
    // Looking closer: about `near` at every kind of turn, and without it only where no spot of
    // any turn holds the object in, so that a search that finds falls tries no spot more.
    let near = p.near.map(Vec3::truncate);
    if o.look_closer && (near.is_some() || found.iter().all(Vec::is_empty)) {
        let each = found.par_iter_mut().zip(&hollows).zip(&turns);
        each.for_each(|((found, hollows), &turn)| {
            if let Some(hollows) = hollows {
                found.extend(hollows.closer(shape, grown, turn, near, seen, o));
            }
        });
    }
    let turned = |fall: &Fall, turn: Quat| Fall {
        pose: Pose::new(fall.pose.position, turn),
        drop: fall.drop,
        sunk: fall.sunk,
        cost: fall.cost,
    };
    let mut falls = search::at_every_turn(found, &turns, &first, turned);
    search::nearest_first(&mut falls, p.near, |f| f.pose.position, |f| &mut f.cost);
    falls
}

/// The falls of `shape`, `grown` by the clearance it keeps, turned by `turn`; and with
/// [`RestOptions::look_closer`], its hollows, for a closer look.
fn at_turn(
    shape: Shape,
    grown: Shape,
    turn: Quat,
    seen: &Seen,
    o: &RestOptions,
) -> (Vec<Fall>, Option<Hollows>) {
    let (spots, field) = (seen.spots, &seen.field);
    let (across, rows) = (spots.across, spots.rows);
    // Whether it can leave is asked of it worn down by half the diagonal between spots: a way
    // out it fits through lies within that of a spot, where so worn it fits too. Worn thinner
    // than a cell of the support, it could slip between the points seen of a solid wall.
    let Some(worn) = shape.worn(FRAC_1_SQRT_2 * o.spacing, turn, o.touching) else {
        return (vec![], None);
    };
    let leaves = Turned::new(worn, turn);
    let lands = |object: &Turned| {
        (0..spots.count())
            .into_par_iter()
            .map(|k| field.lands(object, spots.at(k, o.spacing)))
            .collect::<Vec<f32>>()
    };
    // How high under its middle the shrunk object lands over each spot, and how high it would
    // have to rise to leave from there: the lowest of the rim around it, in its bottom's height.
    let low = lands(&leaves);
    let level = filled(&low, across, rows);
    let hollow = |k: usize| level[k].is_finite() && level[k] > low[k];
    let rim: Vec<f32> = level.iter().map(|level| level - leaves.depth).collect();
    // Where the object lands over each spot of a hollow, in its bottom's height, and how far that
    // is under the rim; negative infinity where nothing was seen under it. Nothing reads it
    // outside a hollow, where it is not looked for. It lands as it is: the clearance it keeps is
    // in where it is let go.
    let object = Turned::new(shape, turn);
    let bottom: Vec<f32> = (0..low.len())
        .into_par_iter()
        .map(|k| {
            if hollow(k) {
                field.lands(&object, spots.at(k, o.spacing)) - object.depth
            } else {
                f32::NAN
            }
        })
        .collect();
    let sunk = |k: usize| rim[k] - bottom[k];
    // The hollows, each the spots that adjoin one another, and how far down it lands in each.
    let around = |k: usize| {
        let (i, j) = ((k % across) as i32, (k / across) as i32);
        let beside = (0..9).map(move |n| (i + n % 3 - 1, j + n / 3 - 1));
        let inside =
            move |&(i, j): &(i32, i32)| i >= 0 && j >= 0 && i < across as i32 && j < rows as i32;
        let own = move |n: &usize| *n != k;
        beside
            .filter(inside)
            .map(move |(i, j)| j as usize * across + i as usize)
            .filter(own)
    };
    let mut of: Vec<u32> = vec![u32::MAX; low.len()];
    let mut deepest: Vec<f32> = vec![];
    let mut next = VecDeque::new();
    for start in 0..low.len() {
        if !hollow(start) || of[start] != u32::MAX {
            continue;
        }
        let own = deepest.len() as u32;
        deepest.push(0.0);
        of[start] = own;
        next.push_back(start);
        while let Some(k) = next.pop_front() {
            if bottom[k].is_finite() {
                deepest[own as usize] = deepest[own as usize].max(sunk(k));
            }
            for n in around(k) {
                if hollow(n) && of[n] == u32::MAX {
                    of[n] = own;
                    next.push_back(n);
                }
            }
        }
    }
    // Held in: in a hollow it lands as deep in as its middle is over its bottom somewhere, where
    // it lands that deep itself, or on nothing that was seen (infinitely deep).
    let middle = object.depth;
    let held = |k: usize| hollow(k) && deepest[of[k] as usize] >= middle && sunk(k) >= middle;
    // How many spots lie between each and the nearest that is not held in: its room.
    let mut room: Vec<u32> = (0..low.len())
        .map(|k| u32::from(held(k)) * u32::MAX)
        .collect();
    for (k, room) in room.iter_mut().enumerate() {
        let edge = k % across == 0 || k % across == across - 1 || k / across == 0;
        if held(k) && (edge || k / across == rows - 1 || around(k).any(|n| !held(n))) {
            *room = 1;
            next.push_back(k);
        }
    }
    while let Some(k) = next.pop_front() {
        for n in around(k) {
            if room[n] == u32::MAX {
                room[n] = room[k] + 1;
                next.push_back(n);
            }
        }
    }
    // Of those with as much room, those nearest the middle of where it is held in that hollow
    // first.
    let mut middles = vec![(Vec2::ZERO, 0u32); deepest.len()];
    for k in (0..low.len()).filter(|&k| held(k)) {
        let middle = &mut middles[of[k] as usize];
        *middle = (middle.0 + spots.at(k, o.spacing), middle.1 + 1);
    }
    let from_middle = |k: usize| {
        let (sum, count) = middles[of[k] as usize];
        spots.at(k, o.spacing).distance(sum / count as f32)
    };
    let mut order: Vec<usize> = (0..low.len()).filter(|&k| held(k)).collect();
    order.sort_by(|&a, &b| {
        let key = |k: usize| (Reverse(room[k]), from_middle(k));
        let (ka, kb) = (key(a), key(b));
        (ka.0.cmp(&kb.0))
            .then(ka.1.total_cmp(&kb.1))
            .then(a.cmp(&b))
    });
    let grown = Turned::new(grown, turn);
    let fall = |k: usize| {
        let pose = let_go(&grown, spots.at(k, o.spacing), rim[k], field, turn);
        let sunk = if bottom[k].is_finite() {
            sunk(k)
        } else {
            deepest[of[k] as usize]
        };
        // Where its bottom lands: on what was seen under it, or as deep as the hollow was seen.
        let landed = if bottom[k].is_finite() {
            bottom[k]
        } else {
            rim[k] - sunk
        };
        Fall {
            pose,
            drop: pose.position.z - (landed + object.depth),
            sunk,
            cost: 0.0,
        }
    };
    // What a closer look reads of the turn: the rim where a hollow is, the spots that hold the
    // object in, and the spot of a hollow where it lands the deepest under the rim on something
    // seen, its lowest point.
    let hollows = o.look_closer.then(|| {
        let rim = (0..low.len()).map(|k| if hollow(k) { rim[k] } else { f32::NEG_INFINITY });
        let seen_under = (0..low.len()).filter(|&k| hollow(k) && bottom[k].is_finite());
        let lowest = seen_under.reduce(|a, b| if sunk(b) > sunk(a) { b } else { a });
        Hollows {
            rim: rim.collect(),
            held: (0..low.len()).map(held).collect(),
            lowest,
        }
    });
    (order.into_iter().map(fall).collect(), hollows)
}

/// A height with the spot it is at, ordered by the height and then the spot.
struct At(f32, u32);

impl PartialEq for At {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for At {}

impl Ord for At {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

impl PartialOrd for At {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// For each spot of a grid `across` by `rows` of `heights`, the level it would have to rise to
/// before it could leave over the grid's edge, passing from spot to adjoining spot, cornerwise
/// too: its own height where it need not rise. Flooded from the edge inward, the lowest spot
/// reached so far first.
fn filled(heights: &[f32], across: usize, rows: usize) -> Vec<f32> {
    let mut level = heights.to_vec();
    let mut reached = vec![false; heights.len()];
    let mut next = BinaryHeap::new();
    for k in 0..heights.len() {
        let (i, j) = (k % across, k / across);
        if i == 0 || j == 0 || i == across - 1 || j == rows - 1 {
            reached[k] = true;
            next.push(Reverse(At(heights[k], k as u32)));
        }
    }
    while let Some(Reverse(At(from, k))) = next.pop() {
        let (i, j) = ((k as usize % across) as i32, (k as usize / across) as i32);
        for n in 0..9 {
            let (ni, nj) = (i + n % 3 - 1, j + n / 3 - 1);
            if ni < 0 || nj < 0 || ni >= across as i32 || nj >= rows as i32 {
                continue;
            }
            let n = nj as usize * across + ni as usize;
            if !reached[n] {
                reached[n] = true;
                level[n] = heights[n].max(from);
                next.push(Reverse(At(level[n], n as u32)));
            }
        }
    }
    level
}

/// Where to let go of the object turned by `turn` over `at`, given as `grown` by the clearance it
/// keeps: the grown object's bottom at `rim`, the hollow's, or over all it would land on there
/// where that is higher, so that the object is the clearance over the rim and clear of all
/// around by it.
fn let_go(grown: &Turned, at: Vec2, rim: f32, field: &Field, turn: Quat) -> Pose {
    // Where it lands is looked for from the highest middle whose bottom, rounded, is not over the
    // rim: what lies under that leaves it at the rim, and is passed over at a comparison a point.
    let mut from = rim + grown.depth;
    while from - grown.depth > rim {
        from = from.next_down();
    }
    let bottom = rim.max(field.lands_from(grown, at, from) - grown.depth);
    Pose::new(at.extend(bottom + grown.depth), turn)
}

/// What a turn's spots say of its hollows, kept for a closer look ([`RestOptions::look_closer`]).
struct Hollows {
    /// Per spot, the rim of the hollow it is in, in the object's bottom's height: what it would
    /// have to rise to before it could leave from anywhere in the spot's cell, as from the spot
    /// itself, which worn down by half the cell's diagonal is inside it there. Negative infinity
    /// where the spot is in none.
    rim: Vec<f32>,
    /// Per spot, whether it holds the object in.
    held: Vec<bool>,
    /// The spot of a hollow where the object lands the deepest under the rim on something seen.
    lowest: Option<usize>,
}

impl Hollows {
    /// The falls at `turn` over the spots a cell (`touching`) apart within `spacing` of `near`,
    /// or without it of the hollow's lowest point, as rests look closer: a container the object
    /// barely fits holds it in where the spots may lie too far apart to meet. Each is held in
    /// where the object lands under the rim of its cell's spot by as much as its middle is over
    /// its bottom, on something seen. Where that spot holds the object in, one within half a
    /// cell of it, or of a quarter of the spacing where that is less, is the same place, as for
    /// rests. They come the nearest the middle of where they are held in first.
    fn closer(
        &self,
        shape: Shape,
        grown: Shape,
        turn: Quat,
        near: Option<Vec2>,
        seen: &Seen,
        o: &RestOptions,
    ) -> Vec<Fall> {
        let (spots, field) = (&seen.spots, &seen.field);
        let about = match (near, self.lowest) {
            (Some(near), _) => near,
            (None, Some(lowest)) => spots.at(lowest, o.spacing),
            (None, None) => return vec![],
        };
        let (object, grown) = (Turned::new(shape, turn), Turned::new(grown, turn));
        let one = o.touching.min(0.5 * o.spacing);
        let held = |at: Vec2| {
            let k = spots.holding(at, o.spacing)?;
            if self.held[k] && at.distance(spots.at(k, o.spacing)) < 0.5 * one {
                return None;
            }
            let bottom = field.lands(&object, at) - object.depth;
            let sunk = self.rim[k] - bottom;
            (bottom.is_finite() && sunk >= object.depth).then_some((at, k, sunk, bottom))
        };
        let nearer = search::nearer(about, spots, None, o);
        let found: Vec<(Vec2, usize, f32, f32)> = nearer.into_par_iter().filter_map(held).collect();
        let middle = found.iter().map(|f| f.0).sum::<Vec2>() / found.len() as f32;
        let mut order: Vec<usize> = (0..found.len()).collect();
        let from_middle = |i: usize| found[i].0.distance(middle);
        order.sort_by(|&a, &b| from_middle(a).total_cmp(&from_middle(b)).then(a.cmp(&b)));
        let fall = |i: usize| {
            let (at, k, sunk, bottom) = found[i];
            let pose = let_go(&grown, at, self.rim[k], field, turn);
            Fall {
                pose,
                drop: pose.position.z - (bottom + object.depth),
                sunk,
                cost: 0.0,
            }
        };
        order.into_iter().map(fall).collect()
    }
}
