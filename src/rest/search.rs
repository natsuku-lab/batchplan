//! The poses [`rests_on`] tries and what it keeps of them: each spot over the support at each
//! turn, set down, judged, and put in an order worth trying.

use std::cmp::Ordering;
use std::f32::consts::{FRAC_PI_2, TAU};

use glam::{Quat, Vec2, Vec3};
use rayon::prelude::*;

use super::contact::Touch;
use super::field::Field;
use super::shape::{Shape, Turned};
use super::statics::{Contact, Statics};
use crate::device::Device;
use crate::error::{Result, ensure_input, input};
use crate::surface::Surface;
use crate::types::Pose;
use crate::world::{self, Obstacle};

/// One object [`rests_on`] sets down, on one of the call's surfaces.
#[derive(Clone, Debug, PartialEq)]
pub struct RestProblem {
    /// The object: its shape in its own frame, as [`Held::object`] has it (its pose is not read),
    /// its centre of mass at its middle. A box, ball, cylinder or capsule; not a distance grid.
    ///
    /// [`Held::object`]: crate::Held::object
    pub object: Obstacle,
    /// The surface it is set down on, by its index among the call's: several problems may name
    /// one surface, which is read once.
    pub surface: usize,
    /// How it is turned at its first rests, in the surface's frame: identity, upright as it stands
    /// in its own frame. A can held level to lie on its side is
    /// `Quat::from_rotation_x(FRAC_PI_2)`; a place that keeps how the hand holds the object now
    /// passes the object's rotation as it is.
    pub rotation: Quat,
}

impl RestProblem {
    /// `object` set down upright, as it stands in its own frame, on surface `surface`.
    #[must_use]
    pub fn new(object: Obstacle, surface: usize) -> Self {
        Self {
            object,
            surface,
            rotation: Quat::IDENTITY,
        }
    }
}

/// How [`rests_on`] sets objects down. The surfaces' frame has its z axis up, against gravity.
#[derive(Clone, Copy, Debug)]
pub struct RestOptions {
    /// The turns about the vertical each spot is tried at, from a problem's `rotation` round a
    /// whole turn: 8, as [`place_poses`](crate::place_poses) is commonly given. Turns the
    /// object's shape maps onto an earlier one, as a box's half turn or any turn of an upright
    /// can, are left out: `place_poses` gives a rest's copies about the object's own axis, and
    /// one held by an end and laid down may be set down end for end too.
    pub turns: usize,
    /// How far apart the spots its middle is set down over lie, seen from above (meters): 1 cm,
    /// the step of Wang and Hauser's search.
    pub spacing: f32,
    /// How near under the object a support point counts as touching it (meters): 2 mm, a depth
    /// camera's noise under a meter, so a flat bottom on a measured table touches it across. The
    /// support is read in cells this wide, each with its highest point.
    pub touching: f32,
    /// How far every point that is not the support's stays from the object (meters): 1 cm, twice
    /// the 5 mm a held object's spheres reach past it by default ([`Held::spheres`]), so that
    /// [`place`](crate::place) checks it clear where it was found clear.
    ///
    /// [`Held::spheres`]: crate::Held::spheres
    pub clearance: f32,
    /// How far the object's weight must be able to tip, any way, with its contacts still holding
    /// it (radians), in [0, π/2): 5°, chosen so that a nudge as the hand opens neither topples it
    /// nor starts it sliding. A tall object needs its centre of mass farther in than a flat one,
    /// and a slope holds an object only this far short of the friction's angle.
    pub tipping: f32,
    /// The friction between the object and what it rests on: 0.5, as
    /// [`GraspOptions::friction`](crate::GraspOptions::friction). A slope steeper than
    /// `atan(friction)` (27°) less `tipping` holds the object only against a stop below it.
    pub friction: f32,
}

impl Default for RestOptions {
    fn default() -> Self {
        Self {
            turns: 8,
            spacing: 0.01,
            touching: 0.002,
            clearance: 0.01,
            tipping: 5f32.to_radians(),
            friction: 0.5,
        }
    }
}

/// A rest [`rests_on`] found.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Rest {
    /// The object's frame where it rests, in the surface's frame: one of [`Place::poses`].
    ///
    /// [`Place::poses`]: crate::Place::poses
    pub pose: Pose,
    /// What it touches: the corners of the convex hull, seen from above, of the support's points
    /// that touch it, anticlockwise from above, each at its height.
    pub support: Vec<Vec3>,
    /// How far its weight can be tipped, any way, before its contacts no longer hold it
    /// (radians): at least [`RestOptions::tipping`]. It is tried toward eight directions; where
    /// the object lies on one flat face it is exact toward each, the lesser of the angle to the
    /// edge of what that face touches and the friction's, and elsewhere it is found to half a
    /// degree, from below.
    pub tipping: f32,
}

/// The most spots one surface's support may spread over: past it, space them wider.
const MOST_SPOTS: usize = 1 << 20;

/// The most rests of a turn that are put in an order that spreads them over the support.
const SPREAD: usize = 1024;

/// Rests of each of `problems`, in their order: poses of the problem's object where it rests on its
/// surface's `object` points, the support it may touch (empty: every point is), clear of every
/// other point by [`RestOptions::clearance`] and under none: what is under a point that is not
/// the support's is taken as solid too, so nothing rests under a shelf that is not flagged, however
/// high it is.
///
/// The support is read from above, as a height in each cell [`RestOptions::touching`] wide: what
/// is under a cell's highest point is taken as solid, so a support point whose normal faces down,
/// as a shelf seen from below, holds nothing up, and a cell no point lies in is empty space: a
/// floor no camera saw holds nothing. Each spot [`RestOptions::spacing`] apart over the support is
/// tried at each turn `o` gives, the object set straight down until it touches. It rests where
/// what it then touches holds it: its contacts, as stiff springs that push and do not pull, carry
/// its weight, with friction only where the weight presses, and still do with the weight tipped
/// by `tipping` any way. It rests as it is set down: it is not tipped or slid to where it would
/// come to rest, so on a slope it needs the `rotation` that lies flat on it. A can or a capsule
/// laid on its side is the one thing that moves: where it is not held it rolls, up to half the
/// spots' spacing, to where it lies lowest, as into the cradle between two bars or down to a
/// lip, and rests there if that holds it. Only the object's rest is
/// checked: the hand that sets it down and its way there are [`place`](crate::place)'s to check,
/// with what the object may touch as its `support`.
///
/// They come in an order whose first few are worth trying first: every spot at the first turn,
/// then every spot at the next, and within a turn, the steadiest first and then each the farthest
/// from those before it, so the first few spread over the support; past the first 1,024 of a
/// turn, as their spots come. The problems run in parallel
/// on the device's threads, each surface read once however many problems name it, and a
/// problem's rests are the same alone and among others.
pub fn rests_on(
    device: &Device,
    problems: &[RestProblem],
    surfaces: &[Surface],
    o: &RestOptions,
) -> Result<Vec<Vec<Rest>>> {
    ensure_input!(
        o.turns > 0
            && o.spacing > 0.0
            && o.spacing.is_finite()
            && o.touching > 0.0
            && o.touching.is_finite()
            && o.clearance >= 0.0
            && o.clearance.is_finite()
            && (0.0..FRAC_PI_2).contains(&o.tipping)
            && o.friction >= 0.0
            && o.friction.is_finite(),
        "rests need a turn, a finite positive spacing and touching, a finite clearance and friction of at least 0 and a tipping angle in [0, pi/2), got {}, {} m, {} m, {} m, {} and {} rad",
        o.turns,
        o.spacing,
        o.touching,
        o.clearance,
        o.friction,
        o.tipping
    );
    let mut spots = Vec::with_capacity(surfaces.len());
    for (i, s) in surfaces.iter().enumerate() {
        ensure_input!(
            s.normals.len() == s.points.len()
                && (s.object.is_empty() || s.object.len() == s.points.len())
                && (s.seen_from.is_empty() || s.seen_from.len() == s.points.len()),
            "surface {i}: {} points with {} normals, {} object flags and {} places seen from",
            s.points.len(),
            s.normals.len(),
            s.object.len(),
            s.seen_from.len()
        );
        if let Some(k) = s.points.iter().position(|p| !p.is_finite()) {
            return Err(input!(
                "surface {i}: point {k} is not finite, got {}",
                s.points[k]
            ));
        }
        let unit = |n: &Vec3| *n == Vec3::ZERO || (n.length_squared() - 1.0).abs() < 1e-3;
        if let Some(k) = s.normals.iter().position(|n| !unit(n)) {
            return Err(input!(
                "surface {i}: normal {k} must be unit or zero, got {}",
                s.normals[k]
            ));
        }
        spots.push(Spots::over(i, s, o.spacing)?);
    }
    for (i, p) in problems.iter().enumerate() {
        ensure_input!(
            p.surface < surfaces.len(),
            "problem {i}: surface {} of {}",
            p.surface,
            surfaces.len()
        );
        ensure_input!(
            !matches!(p.object, Obstacle::Sdf { .. }),
            "problem {i}: the object is a distance grid, which has no rests"
        );
        p.object
            .placed(Pose::IDENTITY)
            .check()
            .map_err(|e| input!("problem {i}: the object: {e}, got {:?}", p.object))?;
        ensure_input!(
            p.rotation.is_finite() && (p.rotation.length() - 1.0).abs() < 1e-3,
            "problem {i}: the rotation must be a finite unit quaternion, got {}",
            p.rotation
        );
    }
    let each = || {
        // Each surface a problem names, read once, with what lies as far from its support as
        // the largest object set down on it reaches, grown by the clearance it keeps.
        let seen: Vec<Option<Seen>> = (0..surfaces.len())
            .into_par_iter()
            .map(|i| {
                let on = problems.iter().filter(|p| p.surface == i);
                let grown = on.map(|p| Shape::of(&p.object).grown(o.clearance).radius());
                let reach = grown.reduce(f32::max)?;
                let spots = spots[i]?;
                let margin = reach + o.touching;
                let field = Field::new(&surfaces[i], spots.lower, spots.upper, o.touching, margin);
                Some(Seen { field, spots })
            })
            .collect();
        let each = problems.par_iter().map(|p| match &seen[p.surface] {
            Some(seen) => on(p, seen, o),
            None => vec![],
        });
        each.collect()
    };
    Ok(device.install(each))
}

/// The spots over one surface's support, seen from above.
#[derive(Clone, Copy)]
struct Spots {
    /// The corners of the support, seen from above.
    lower: Vec2,
    upper: Vec2,
    /// The first spot's cell, in spacings.
    first: Vec2,
    across: usize,
    rows: usize,
}

impl Spots {
    /// The spots over surface `i`'s support, `spacing` apart; none where it has no support. An
    /// `Error::Input` where they are more than a call takes.
    fn over(i: usize, s: &Surface, spacing: f32) -> Result<Option<Self>> {
        let support = (0..s.points.len()).filter(|&k| s.object.is_empty() || s.object[k]);
        let flat = support.map(|k| s.points[k].truncate());
        let lower = flat.clone().fold(Vec2::INFINITY, Vec2::min);
        let upper = flat.fold(Vec2::NEG_INFINITY, Vec2::max);
        if !lower.cmple(upper).all() {
            return Ok(None);
        }
        let first = (lower / spacing).floor();
        let count = (upper / spacing).floor() - first + 1.0;
        ensure_input!(
            count.x * count.y <= MOST_SPOTS as f32,
            "surface {i}: the support spans {:.2} by {:.2} m, {} spots {spacing} m apart, more than {MOST_SPOTS}; space them wider",
            upper.x - lower.x,
            upper.y - lower.y,
            count.x * count.y
        );
        Ok(Some(Self {
            lower,
            upper,
            first,
            across: count.x as usize,
            rows: count.y as usize,
        }))
    }

    fn count(&self) -> usize {
        self.across * self.rows
    }

    /// Spot `k`'s middle, seen from above.
    fn at(&self, k: usize, spacing: f32) -> Vec2 {
        let cell = self.first + Vec2::new((k % self.across) as f32, (k / self.across) as f32);
        (cell + 0.5) * spacing
    }
}

/// One surface as every problem that names it reads it.
struct Seen {
    field: Field,
    spots: Spots,
}

/// What a thread keeps between the poses it tries.
#[derive(Default)]
struct Scratch {
    touch: Touch,
    contacts: Vec<Contact>,
    statics: Statics,
}

/// One problem's object on its surface.
struct Setting<'a> {
    field: &'a Field,
    shape: Shape,
    /// The shape grown by the clearance it keeps.
    grown: Shape,
    o: &'a RestOptions,
}

impl Setting<'_> {
    /// The rest of the object set down turned by `rotation` over `middle`, if what it touches
    /// there holds it and it is clear, and whether it rolled there: one lying on its round side
    /// that is not held where it is set down is tried again where it rolls to.
    fn rest(&self, rotation: Quat, middle: Vec2, s: &mut Scratch) -> Option<(Rest, bool)> {
        let object = Turned::new(self.shape, rotation);
        let (held, rolled) = match self.held(&object, middle, s) {
            Some(held) => (held, false),
            None => {
                let to = self.rolled(&object, middle, &s.contacts)?;
                (self.held(&object, to, s)?, true)
            }
        };
        let (center, tipping) = held;
        if !self.field.clear(&Turned::new(self.grown, rotation), center) {
            return None;
        }
        let rest = Rest {
            pose: Pose::new(center, rotation),
            support: s.statics.support(&s.contacts),
            tipping,
        };
        Some((rest, rolled))
    }

    /// Where the middle of `object` comes set down over `middle`, and how far its weight can be
    /// tipped there, if what it touches holds it; its contacts are left in `s`.
    fn held(&self, object: &Turned, middle: Vec2, s: &mut Scratch) -> Option<(Vec3, f32)> {
        let o = self.o;
        s.contacts.clear();
        let height = self.field.set_down(object, middle);
        if height == f32::NEG_INFINITY {
            return None;
        }
        let center = middle.extend(height);
        s.touch
            .contacts(self.field, object, center, o.touching, &mut s.contacts);
        let size = self.shape.radius();
        let tipping = s
            .statics
            .holds(&s.contacts, center, size, o.tipping, o.friction)?;
        Some((center, tipping))
    }

    /// Where a can or a capsule that lies on its side over `middle`, pushed sideways by
    /// `contacts`, rolls to: across its axis, the way they push it, to where it lies lowest, as
    /// between the two sides of a cradle. A round side is the same turned about its axis, so
    /// rolling is moving across. None where it has no such side, is not pushed across, or is
    /// still going down half the spots' spacing on, in the next spot's reach.
    fn rolled(&self, object: &Turned, middle: Vec2, contacts: &[Contact]) -> Option<Vec2> {
        if !matches!(self.shape, Shape::Cylinder(..) | Shape::Capsule(..)) {
            return None;
        }
        let axis = object.frame.z_axis.truncate();
        let across = Vec2::new(-axis.y, axis.x).try_normalize()?;
        let pushed: f32 = contacts.iter().map(|c| c.push.truncate().dot(across)).sum();
        if pushed == 0.0 {
            return None;
        }
        let across = across * pushed.signum();
        let o = self.o;
        let height = |by: f32| self.field.set_down(object, middle + across * by);
        // The lowest, by golden section, to a sixteenth of `touching`: so near it both sides of
        // a cradle lie within `touching` under it unless they stand steeper than 83 degrees.
        let (far, fine) = (0.5 * o.spacing + o.touching, o.touching / 16.0);
        let ratio = 0.618_034;
        let (mut from, mut to) = (0.0, far);
        let (mut a, mut b) = (to - ratio * (to - from), from + ratio * (to - from));
        let (mut at_a, mut at_b) = (height(a), height(b));
        while to - from > fine {
            if at_a < at_b {
                (to, b, at_b) = (b, a, at_a);
                a = to - ratio * (to - from);
                at_a = height(a);
            } else {
                (from, a, at_a) = (a, b, at_b);
                b = from + ratio * (to - from);
                at_b = height(b);
            }
        }
        let by = 0.5 * (from + to);
        (by > fine && by < far - fine).then(|| middle + across * by)
    }
}

/// `p`'s rests on the surface it names, as `seen`.
fn on(p: &RestProblem, seen: &Seen, o: &RestOptions) -> Vec<Rest> {
    let shape = Shape::of(&p.object);
    let setting = Setting {
        field: &seen.field,
        shape,
        grown: shape.grown(o.clearance),
        o,
    };
    // The turns, each once.
    let rotation = world::unit(p.rotation);
    let mut turns: Vec<Quat> = vec![];
    for k in 0..o.turns {
        let turned = Quat::from_rotation_z(TAU * k as f32 / o.turns as f32) * rotation;
        if !turns
            .iter()
            .any(|&t| shape.same_turned(t.inverse() * turned))
        {
            turns.push(turned);
        }
    }
    let spots = seen.spots;
    // Each turn's rests in their order, the turns side by side.
    let each = turns.par_iter().map(|&turned| {
        let found: Vec<Option<(Rest, bool)>> = (0..spots.count())
            .into_par_iter()
            .map_init(Scratch::default, |scratch, k| {
                setting.rest(turned, spots.at(k, o.spacing), scratch)
            })
            .collect();
        // Spots to either side of a cradle roll to one place, which comes once: the first, of
        // those within `touching` of each other.
        let mut kept: Vec<Rest> = Vec::with_capacity(found.len());
        for (rest, rolled) in found.into_iter().flatten() {
            let at = rest.pose.position;
            let near = |r: &Rest| r.pose.position.distance(at) < o.touching;
            if !(rolled && kept.iter().any(near)) {
                kept.push(rest);
            }
        }
        spread_out(kept)
    });
    let each: Vec<Vec<Rest>> = each.collect();
    each.into_iter().flatten().collect()
}

/// `found`, in the order of their spots, put in the order they are given back in: the steadiest
/// first, then each the farthest from every one before it. Past [`SPREAD`], the rest follow as
/// they came.
fn spread_out(found: Vec<Rest>) -> Vec<Rest> {
    let steadiest = (0..found.len()).reduce(|a, b| {
        if found[b].tipping > found[a].tipping {
            b
        } else {
            a
        }
    });
    let Some(mut next) = steadiest else {
        return found;
    };
    // Those not yet taken, by where they came: where each is, and how far it is from the
    // nearest one taken, squared.
    let mut left: Vec<u32> = (0..found.len() as u32).collect();
    let mut at: Vec<Vec3> = found.iter().map(|r| r.pose.position).collect();
    let mut from: Vec<f32> = vec![f32::INFINITY; found.len()];
    let mut order = Vec::with_capacity(found.len());
    for _ in 0..found.len().min(SPREAD) {
        let taken = at.swap_remove(next);
        order.push(left.swap_remove(next));
        from.swap_remove(next);
        let mut farthest = f32::NEG_INFINITY;
        for (from, at) in from.iter_mut().zip(&at) {
            *from = from.min(at.distance_squared(taken));
            farthest = farthest.max(*from);
        }
        // Of those as far, the steadiest and then the first found.
        let as_far = (0..left.len()).filter(|&k| from[k] == farthest);
        let key = |k: usize| (found[left[k] as usize].tipping, std::cmp::Reverse(left[k]));
        let best = as_far.reduce(|a, b| {
            if key(b).partial_cmp(&key(a)) == Some(Ordering::Greater) {
                b
            } else {
                a
            }
        });
        next = best.unwrap_or(0);
    }
    left.sort_unstable();
    order.extend(left);
    let mut found: Vec<Option<Rest>> = found.into_iter().map(Some).collect();
    let taken = order.into_iter().map(|k| found[k as usize].take());
    taken.map(|r| r.expect("each rest is taken once")).collect()
}
