//! What a mask model says of a view, taken into the look: [`View::masks`](crate::View::masks)
//! gives per pixel the number of the thing the model says it shows. miru runs no model and asks
//! none for anything: the numbers come in as data, from whatever the caller ran.
//!
//! Geometry tells readings apart where the depth says so and nowhere else. It cannot join the
//! parts of one thing that the depth shows no connection between, as a stove's base and its
//! knobs in one view, nor divide two things that read as one surface, as two boxes that
//! stand flush with their tops level. A mask does both, in its own view and of the readings
//! that are in parts alone: the robot's readings and those outside the look's corners are in
//! none, and stay what they are whatever a mask says. A reading on a known shape is in a part
//! as any other, and its part is a mask's thing's only where it lies inside that thing (`look`
//! says).
//!
//! 1. **Split** ([`split`]). A part whose readings lie in more than one mask, or in a mask and
//!    in none, is divided along them. Each side that fills a window, other than the part's
//!    largest, is cut off where it is as wide as a window too, or where it lies in a mask and
//!    either is more than half of all that mask covers in the view or the part's largest side
//!    lies in another mask; any other stays with the part's largest side. A model's masks are
//!    never exact to the pixel: a narrow side against readings in no mask is the fringe of a
//!    mask whose body lies elsewhere, as the rim of a thing just outside its mask or of a
//!    neighbour just inside it, unless it is all there is of a thin thing's mask; one between
//!    two masks is a thing the model names beside another.
//! 2. **Join** ([`joins`]). The parts most of whose readings lie in one mask are one object.
//! 3. **Apart** ([`lies`]). Parts that lie in two masks of one view are two objects, whatever
//!    else would make them one: a look joins none of them, by another view, as a hollow's floor
//!    or as a round thing's outside and its inside.

use std::collections::{BTreeMap, HashMap};

use crate::segment::{self, NONE};
use crate::view::{Kind, Seen};

/// Per part and mask (0 for the readings in none), the readings of the part that lie in the
/// mask: how many, and how many windows they fill. `of` is each pixel's part and `share` the
/// share it is of a window ([`Parts::of`], [`Parts::share`]).
fn sides(of: &[u32], share: &[f32], masks: &[u32]) -> BTreeMap<(u32, u32), (usize, f32)> {
    let mut sides: BTreeMap<(u32, u32), (usize, f32)> = BTreeMap::new();
    for (i, &part) in of.iter().enumerate() {
        if part != NONE {
            let side = sides.entry((part, masks[i])).or_default();
            *side = (side.0 + 1, side.1 + share[i]);
        }
    }
    sides
}

/// Divides each of the `count` parts, which `of` gives per pixel, whose readings lie in more
/// than one of `masks` along them. The readings in no mask are a side as those in one are. A
/// side other than the part's largest is a part of its own where it fills a window, a window
/// being what geometry makes no part of less than (`share` is each pixel's share of one), and
/// either is as wide as one, some reading of it having its whole window in it
/// ([`segment::holding`], of the radius `wanted` gives each reading), or lies in a mask and
/// either holds more than half of the readings in parts that lie in that mask in the view or
/// the part's largest side lies in another mask. Any other side's readings stay with the
/// part's largest side. `masks` is one view's, a number per pixel or empty, and empty leaves
/// the parts as they are.
///
/// The largest side keeps the part's number; the others are numbered after the parts there
/// were, and `count` raised by them. Returned is which were cut off: per part and mask, the
/// part that side now is. [`segment::parts`](crate::segment::parts), which measured the borders
/// by side, gives each what it hides and the creases it meets in by that.
///
/// A reading on a known shape that lies in no mask, in a part most of which lies in one, leaves
/// the part: the model says it is not the thing's, and it is the shape's again. `seen` is the
/// view's readings.
pub(crate) fn split(
    of: &mut [u32],
    count: &mut usize,
    share: &[f32],
    wanted: &[usize],
    masks: &[u32],
    seen: &Seen,
) -> HashMap<(u32, u32), u32> {
    let mut cut: HashMap<(u32, u32), u32> = HashMap::new();
    if masks.is_empty() {
        return cut;
    }
    let mut sides = sides(of, share, masks);
    // Per part, the mask of its largest side and the windows that fills: of two as large, the
    // lower numbered.
    let mut largest = vec![(0, 0.0); *count];
    for (&(part, mask), &(_, windows)) in &sides {
        if windows > largest[part as usize].1 {
            largest[part as usize] = (mask, windows);
        }
    }
    for (i, part) in of.iter_mut().enumerate() {
        let left = masks[i] == 0 && matches!(seen.kinds[i], Kind::Known(_));
        if *part != NONE && left && largest[*part as usize].0 != 0 {
            let side = sides.get_mut(&(*part, 0)).expect("the reading was counted");
            *side = (side.0 - 1, side.1 - share[i]);
            *part = NONE;
        }
    }
    // Per mask, the readings in parts that lie in it.
    let mut covers: HashMap<u32, usize> = HashMap::new();
    for (&(_, mask), &(readings, _)) in &sides {
        *covers.entry(mask).or_default() += readings;
    }
    // The sides that may be cut, in the order of `sides`, each with whether its mask says it is
    // a thing however narrow: all there is of a thin thing's mask, or a thing the model names
    // beside another.
    let candidates: Vec<((u32, u32), bool)> = (sides.iter())
        .filter(|&(&(part, mask), &(_, windows))| {
            windows >= 1.0 && mask != largest[part as usize].0
        })
        .map(|(&(part, mask), &(readings, _))| {
            let beside = largest[part as usize].0 != 0;
            let named = mask != 0 && (2 * readings > covers[&mask] || beside);
            ((part, mask), named)
        })
        .collect();
    // Whether each side that only its width may cut holds a window, those sides numbered in
    // their order and every other pixel in none.
    let asked: HashMap<(u32, u32), u32> = (candidates.iter())
        .filter(|&&(_, named)| !named)
        .enumerate()
        .map(|(number, &(side, _))| (side, number as u32))
        .collect();
    let holds = if asked.is_empty() {
        vec![]
    } else {
        let side_of: Vec<u32> = (of.iter().zip(masks))
            .map(|(&part, &mask)| asked.get(&(part, mask)).copied().unwrap_or(NONE))
            .collect();
        segment::holding(seen, &side_of, wanted, 0..asked.len() as u32)
    };
    for (side, named) in candidates {
        let wide = || {
            asked
                .get(&side)
                .is_some_and(|&n| holds.get(n as usize) == Some(&true))
        };
        if named || wide() {
            cut.insert(side, *count as u32);
            *count += 1;
        }
    }
    for (part, &mask) in of.iter_mut().zip(masks) {
        if let Some(&side) = cut.get(&(*part, mask)) {
            *part = side;
        }
    }
    cut
}

/// Per part of one view, of `count` that `of` gives per pixel, the mask it lies in: the one
/// that holds more than half of its readings, or 0 where none does and the masks do not say
/// what the part is. The readings in no mask say nothing. `masks` is that view's, a number per
/// pixel or empty, and `share` each pixel's share of a window.
pub(crate) fn lies(of: &[u32], count: usize, share: &[f32], masks: &[u32]) -> Vec<u32> {
    let mut lies = vec![0; count];
    if masks.is_empty() {
        return lies;
    }
    let sides = sides(of, share, masks);
    let mut readings = vec![0; count];
    for (&(part, _), &(within, _)) in &sides {
        readings[part as usize] += within;
    }
    for (&(part, mask), &(within, _)) in &sides {
        if mask != 0 && 2 * within > readings[part as usize] {
            lies[part as usize] = mask;
        }
    }
    lies
}

/// The parts of one view that its masks say are one thing, per mask in the masks' order: the
/// parts that lie in it, in their order. `lies` is each part's mask ([`lies`]).
pub(crate) fn joins(lies: &[u32]) -> Vec<Vec<usize>> {
    let mut things: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (part, &mask) in lies.iter().enumerate() {
        if mask != 0 {
            things.entry(mask).or_default().push(part);
        }
    }
    things.into_values().collect()
}
