//! The poses [`rests_on`] tries and what it keeps of them: each spot over the support at each
//! turn, set down, judged, and put in an order worth trying.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, TAU};

use glam::{IVec2, IVec3, Quat, Vec2, Vec3};
use rayon::prelude::*;

use super::contact::Touch;
use super::field::{self, Field};
use super::shape::{Shape, Turned};
use super::statics::{Contact, Holding, Statics};
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
    /// Where the caller would have it set down, if it rests near there: a point in the surface's
    /// frame, as the middle of a plate or where an agent pointed; its height is not read. Within
    /// [`RestOptions::spacing`] of it, seen from above, and over the support, the object is also
    /// set down a cell of the support apart, [`RestOptions::touching`] wide, for a rest that the
    /// spots lie too far apart to find: a flat bottom wider than a dished plate's floor stands
    /// evenly on its raised ring only within millimeters of the middle. That is as many more
    /// spots as cells lie within the spacing of the point, 77 at the defaults. None looks no
    /// closer anywhere.
    pub near: Option<Vec3>,
}

impl RestProblem {
    /// `object` set down upright, as it stands in its own frame, on surface `surface`.
    #[must_use]
    pub fn new(object: Obstacle, surface: usize) -> Self {
        Self {
            object,
            surface,
            rotation: Quat::IDENTITY,
            near: None,
        }
    }
}

/// How [`rests_on`] sets objects down. The surfaces' frame has its z axis up, against gravity.
#[derive(Clone, Copy, Debug)]
pub struct RestOptions {
    /// The turns about the vertical each spot is tried at, from a problem's `rotation` round a
    /// whole turn: 8, as [`place_poses`](crate::place_poses) is commonly given. A rest comes at
    /// each of them: a turn that sets the shape down as an earlier one does, as a box's half turn
    /// or any turn of an upright can, rests where that one does and is found with it, and is
    /// another pose for the hand, as a bottle laid down end for end.
    pub turns: usize,
    /// How far apart the spots its middle is set down over lie, seen from above (meters): 1 cm,
    /// the step of Wang and Hauser's search.
    pub spacing: f32,
    /// How near under the object a support point counts as touching it (meters): 2 mm, a depth
    /// camera's noise under a meter, so a flat bottom on a measured table touches it across. The
    /// support is read in cells this wide, each with its highest point; a support of more than
    /// four million of them is an input error that asks for a wider one.
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
/// is under a cell's highest point is taken as solid, so a support point whose normal faces down
/// and that was not seen from above ([`Surface::seen_from`] no higher than it, or not known), as a
/// shelf seen from below, holds nothing up, and a cell no point lies in is empty space: a floor no
/// camera saw holds nothing. Each spot [`RestOptions::spacing`] apart over the support is
/// tried at each turn `o` gives, the object set straight down until it touches. It rests where
/// what it then touches holds it: its contacts, as stiff springs that push and do not pull, carry
/// its weight, with friction only where the weight presses, and still do with the weight tipped
/// by `tipping` any way. It rests as it is set down: it is not tipped or slid to where it would
/// come to rest, so on a slope it needs the `rotation` that lies flat on it. A can or a capsule
/// laid on its side is the one thing that moves: where it is not held it rolls, up to half the
/// spots' spacing and `touching` more, to where it lies lowest, as into the cradle between two
/// bars or down to a lip, and rests there if that holds it.
///
/// A rest narrower than the spots are apart is looked for closer. Around each spot where the
/// object held but tipped too easily (its contacts carry its weight upright and not with the
/// weight tipped by `tipping`), with no rest at any of the eight spots around it at that turn and
/// at least one of the eight where it was not held at all, or that lies off the grid, the spots
/// a cell ([`RestOptions::touching`]) apart within `spacing` of it are tried too: a rest such as
/// an upright bottle's against the lip at the foot of a sloping rack lies between a spot that
/// nearly holds and one that does not. Rests found there come in the order the others do. Where
/// the problem says where it would have the object ([`RestProblem::near`]), the spots about that
/// point are tried as well. Only the object's rest is checked: the hand that sets it down and
/// its way there are [`place`](crate::place)'s to check, with what the object may touch as its
/// `support`.
///
/// They come at every turn, in an order whose first few are worth trying first: every spot at
/// the first turn, then every spot at the next, and within a turn, the steadiest first and then
/// each the farthest from those before it, so the first few spread over the support; past the
/// first 1,024 of a turn, as their spots come. The problems run in parallel on the device's
/// threads, each surface read once however many problems name it, and a
/// problem's rests are the same alone and among others.
pub fn rests_on(
    device: &Device,
    problems: &[RestProblem],
    surfaces: &[Surface],
    o: &RestOptions,
) -> Result<Vec<Vec<Rest>>> {
    let read = Read::checked(problems, surfaces, o)?;
    let each = || {
        let seen = read.seen(surfaces, o);
        let each = problems.par_iter().map(|p| match &seen[p.surface] {
            Some(seen) => on(p, seen, o),
            None => vec![],
        });
        each.collect()
    };
    Ok(device.install(each))
}

/// What a call reads of its surfaces, once its problems, surfaces and options are checked.
pub(super) struct Read {
    /// Per surface, the spots over its support; none where it has no support.
    spots: Vec<Option<Spots>>,
    /// Per surface, how far around its support what else lies there is read with it; none
    /// where no problem names it.
    margins: Vec<Option<f32>>,
}

impl Read {
    /// `problems`, `surfaces` and `o` checked: an `Error::Input` that names what is wrong.
    pub(super) fn checked(
        problems: &[RestProblem],
        surfaces: &[Surface],
        o: &RestOptions,
    ) -> Result<Self> {
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
            if let Some(near) = p.near {
                ensure_input!(
                    near.is_finite(),
                    "problem {i}: the point to rest near must be finite, got {near}"
                );
                let count = spots[p.surface].map_or(0.0, |spots| {
                    let [across, rows] = Closer::about(near.truncate(), &spots, o);
                    across.count as f32 * rows.count as f32
                });
                ensure_input!(
                    count <= MOST_SPOTS as f32,
                    "problem {i}: {count} spots {} m apart within {} m of the point to rest near, more than {MOST_SPOTS}; space the spots nearer",
                    o.touching,
                    o.spacing
                );
            }
        }
        // What lies as far from each surface's support as the largest object set down on it
        // reaches, grown by the clearance it keeps, is read with it.
        let mut margins = Vec::with_capacity(surfaces.len());
        for (i, spots) in spots.iter().enumerate() {
            let on = problems.iter().filter(|p| p.surface == i);
            let grown = on.map(|p| Shape::of(&p.object).grown(o.clearance).radius());
            let margin = grown.reduce(f32::max).map(|reach| reach + o.touching);
            if let (Some(spots), Some(margin)) = (spots, margin) {
                let cells = Field::cells(spots.lower, spots.upper, o.touching, margin);
                ensure_input!(
                    cells <= field::MOST_CELLS,
                    "surface {i}: {cells} cells {} m wide over its support and the {margin} m around it that its objects reach, more than {}; read it in wider cells",
                    o.touching,
                    field::MOST_CELLS
                );
            }
            margins.push(margin);
        }
        Ok(Self { spots, margins })
    }

    /// Each surface a problem names, read once, on the device's threads.
    pub(super) fn seen(&self, surfaces: &[Surface], o: &RestOptions) -> Vec<Option<Seen>> {
        (0..surfaces.len())
            .into_par_iter()
            .map(|i| {
                let (spots, margin) = (self.spots[i]?, self.margins[i]?);
                let field = Field::new(&surfaces[i], spots.lower, spots.upper, o.touching, margin);
                Some(Seen { field, spots })
            })
            .collect()
    }
}

/// The spots over one surface's support, seen from above.
#[derive(Clone, Copy)]
pub(super) struct Spots {
    /// The corners of the support, seen from above.
    lower: Vec2,
    upper: Vec2,
    /// The first spot's cell, in spacings.
    first: Vec2,
    pub(super) across: usize,
    pub(super) rows: usize,
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

    pub(super) fn count(&self) -> usize {
        self.across * self.rows
    }

    /// Spot `k`'s middle, seen from above.
    pub(super) fn at(&self, k: usize, spacing: f32) -> Vec2 {
        let cell = self.first + Vec2::new((k % self.across) as f32, (k / self.across) as f32);
        (cell + 0.5) * spacing
    }
}

/// Along one axis, the spots a cell apart about the point a problem would rest near: the first,
/// in cells from the point, and how many.
#[derive(Clone, Copy)]
struct Closer {
    first: f32,
    count: usize,
}

impl Closer {
    /// Along x and along y, the spots about `near` that lie within the spacing of it and over
    /// the support of `spots`, where alone a rest's middle can be.
    fn about(near: Vec2, spots: &Spots, o: &RestOptions) -> [Self; 2] {
        let cell = o.touching;
        let first = ((spots.lower - near).max(Vec2::splat(-o.spacing)) / cell).ceil();
        let last = ((spots.upper - near).min(Vec2::splat(o.spacing)) / cell).floor();
        let along = |axis: usize| Self {
            first: first[axis],
            count: (last[axis] - first[axis] + 1.0).max(0.0) as usize,
        };
        [along(0), along(1)]
    }
}

/// One surface as every problem that names it reads it.
pub(super) struct Seen {
    pub(super) field: Field,
    pub(super) spots: Spots,
}

/// What a thread keeps between the poses it tries.
#[derive(Default)]
struct Scratch {
    touch: Touch,
    contacts: Vec<Contact>,
    statics: Statics,
    /// What the last pose judged with `WHY` left: where its middle was set down, how far its
    /// weight can be tipped there, and the first point in its way, if any.
    center: Vec3,
    tipping: f32,
    blocking: Option<Vec3>,
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
    /// What becomes of the object set down turned by `rotation` over `middle`: it rests where
    /// what it touches there holds it and it is clear, and with the rest comes whether it rolled
    /// there: one lying on its round side that is not held where it is set down is tried again
    /// where it rolls to, and what becomes of it is what became of it there.
    ///
    /// With `WHY`, `s` keeps what [`Scratch`] says of the last pose judged, and where it is not
    /// held or tips, what would be in its way there even if it held.
    fn rest<const WHY: bool>(&self, rotation: Quat, middle: Vec2, s: &mut Scratch) -> Outcome {
        let object = Turned::new(self.shape, rotation);
        if WHY {
            s.blocking = None;
        }
        let (held, rolled) = match self.held::<WHY>(&object, middle, s) {
            Ok(held) => (held, false),
            Err(missed) => {
                let Some(to) = self.rolled(&object, middle, &s.contacts) else {
                    return self.missed::<WHY>(missed, rotation, s);
                };
                match self.held::<WHY>(&object, to, s) {
                    Ok(held) => (held, true),
                    Err(missed) => return self.missed::<WHY>(missed, rotation, s),
                }
            }
        };
        let (center, tipping) = held;
        let grown = Turned::new(self.grown, rotation);
        if !self.field.clear::<WHY>(&grown, center, &mut s.blocking) {
            return Outcome::Blocked;
        }
        let rest = Rest {
            pose: Pose::new(center, rotation),
            support: s.statics.support(&s.contacts),
            tipping,
        };
        Outcome::Rests(rest, rolled)
    }

    /// Where the middle of `object` comes set down over `middle`, and how far its weight can be
    /// tipped there, if what it touches holds it; else what became of it. Its contacts are left
    /// in `s`.
    fn held<const WHY: bool>(
        &self,
        object: &Turned,
        middle: Vec2,
        s: &mut Scratch,
    ) -> Result<(Vec3, f32), Outcome> {
        let o = self.o;
        s.contacts.clear();
        let height = self.field.set_down(object, middle);
        if height == f32::NEG_INFINITY {
            return Err(Outcome::Unsupported);
        }
        let center = middle.extend(height);
        if WHY {
            s.center = center;
        }
        s.touch
            .contacts(self.field, object, center, o.touching, &mut s.contacts);
        let size = self.shape.radius();
        match s
            .statics
            .holds::<WHY>(&s.contacts, center, size, o.tipping, o.friction)
        {
            Holding::By(tipping) => {
                if WHY {
                    s.tipping = tipping;
                }
                Ok((center, tipping))
            }
            Holding::Tips => {
                if WHY {
                    s.tipping = s.statics.tips_by;
                }
                Err(Outcome::Tips)
            }
            Holding::Not => Err(Outcome::Unheld),
        }
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
        // Lying: its side faces down more than an end does.
        let (level, upright) = (object.frame.z_axis.truncate(), object.frame.z_axis.z);
        if level.length() < upright.abs() {
            return None;
        }
        let across = Vec2::new(-level.y, level.x).try_normalize()?;
        let pushed: f32 = contacts.iter().map(|c| c.push.truncate().dot(across)).sum();
        if pushed == 0.0 {
            return None;
        }
        let across = across * pushed.signum();
        let o = self.o;
        let height = |by: f32| self.field.set_down(object, middle + across * by);
        // The lowest, by golden section, to a sixteenth of `touching`: so near it both sides of
        // a cradle lie within `touching` under it unless they stand steeper than 83 degrees.
        // Half the spacing on and `touching` more, so that a place midway between two spots is
        // in reach of both.
        let (far, fine) = (0.5 * o.spacing + o.touching, o.touching / 16.0);
        let ratio = 0.618_034f32;
        let rounds = ((far / fine).ln() / -ratio.ln()).ceil().max(0.0) as usize;
        let (mut from, mut to) = (0.0, far);
        let (mut a, mut b) = (to - ratio * (to - from), from + ratio * (to - from));
        let (mut at_a, mut at_b) = (height(a), height(b));
        for _ in 0..rounds {
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
    let problem = Problem::new(p, seen, o);
    let spots = seen.spots;
    // Each turn's rests in their order, the turns side by side.
    let each = (0..problem.turns.len()).into_par_iter().map(|k| {
        if problem.first[k] != k {
            return vec![];
        }
        let tried = problem.tried(k);
        let rolled = |f: &Outcome| matches!(f, Outcome::Rests(_, true));
        let any_alike = !problem.closer.is_empty()
            || !tried.nearer.is_empty()
            || tried.outcomes.iter().any(rolled);
        let found = tried.outcomes;
        let mut kept: Vec<Rest> = Vec::with_capacity(found.len());
        // Spots to either side of a cradle roll to one place, and a spot set closer may be one
        // of the grid's: a place comes once, the first of two. Of two of which either rolled,
        // within `touching` of each other, or half the spots' spacing where that is less; of a
        // grid's spot and a closer one, within half of that, so that a closer spot a cell from
        // a grid's is a place of its own. Those kept, by the cell that wide they are in, each
        // with whether it rolled and whether it is of the closer spots.
        let one = o.touching.min(0.5 * o.spacing);
        let mut cells: HashMap<[i32; 3], Vec<(Vec3, bool, bool)>> = HashMap::new();
        for (k, outcome) in found.into_iter().enumerate() {
            let Outcome::Rests(rest, rolled) = outcome else {
                continue;
            };
            let set_closer = k >= spots.count();
            if any_alike {
                let at = rest.pose.position;
                let cell = (at / one).floor().as_ivec3();
                let around =
                    (0..27).map(|n| cell + IVec3::new(n % 3 - 1, n / 3 % 3 - 1, n / 9 - 1));
                let same = |&(there, other_rolled, other_closer): &(Vec3, bool, bool)| {
                    let apart = there.distance(at);
                    ((rolled || other_rolled) && apart < one)
                        || (set_closer != other_closer && apart < 0.5 * one)
                };
                let mut beside = around.filter_map(|c| cells.get(&c.to_array()));
                if beside.any(|there| there.iter().any(same)) {
                    continue;
                }
                let here = cells.entry(cell.to_array()).or_default();
                here.push((at, rolled, set_closer));
            }
            kept.push(rest);
        }
        spread_out(kept)
    });
    let turned = |r: &Rest, turn: Quat| Rest {
        pose: Pose::new(r.pose.position, turn),
        ..r.clone()
    };
    at_every_turn(each.collect(), &problem.turns, &problem.first, turned)
}

/// The turns `o` gives of `rotation`, and for each the first that sets `shape` down as it does:
/// what is found at a turn is found once for its kind.
pub(super) fn turns(shape: Shape, rotation: Quat, o: &RestOptions) -> (Vec<Quat>, Vec<usize>) {
    let rotation = world::unit(rotation);
    let turns: Vec<Quat> = (0..o.turns)
        .map(|k| Quat::from_rotation_z(TAU * k as f32 / o.turns as f32) * rotation)
        .collect();
    let mut first: Vec<usize> = Vec::with_capacity(turns.len());
    for k in 0..turns.len() {
        // Of the earlier turns that are the first of their kind: being alike to within a
        // rounding does not carry from one turn through another to a third.
        let same = |&j: &usize| first[j] == j && shape.same_turned(turns[j].inverse() * turns[k]);
        first.push((0..k).find(same).unwrap_or(k));
    }
    (turns, first)
}

/// What was `found` at the first turn of each kind, at every turn: a turn that sets the shape
/// down as an earlier one does has what that one has, each `turned` its own way, for a hand that
/// holds the object by one end another pose. From the last turn back, so that the first of each
/// kind gives its own away after the others copied it.
pub(super) fn at_every_turn<T: Clone>(
    mut found: Vec<Vec<T>>,
    turns: &[Quat],
    first: &[usize],
    turned: impl Fn(&T, Quat) -> T,
) -> Vec<T> {
    let mut each: Vec<Vec<T>> = vec![vec![]; turns.len()];
    for k in (0..turns.len()).rev() {
        each[k] = if first[k] == k {
            std::mem::take(&mut found[k])
        } else {
            found[first[k]]
                .iter()
                .map(|t| turned(t, turns[k]))
                .collect()
        };
    }
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

/// What became of the object set down over one spot at one turn.
#[derive(Clone, Debug)]
enum Outcome {
    /// It rests there, and whether it rolled there.
    Rests(Rest, bool),
    /// What it touches there holds it upright, and not with its weight tipped by `tipping`.
    Tips,
    /// It would rest there but for something in its way.
    Blocked,
    /// What it touches there does not hold it.
    Unheld,
    /// Nothing of the support lies under it.
    Unsupported,
}

/// One problem's object on its surface, as each of its turns is tried.
struct Problem<'a> {
    setting: Setting<'a>,
    spots: Spots,
    turns: Vec<Quat>,
    /// Per turn, the first that sets the object down as it does.
    first: Vec<usize>,
    /// The spots a cell apart within a spacing of where the caller would have it.
    closer: Vec<Vec2>,
    near: Option<Vec2>,
}

/// What became of the object over each spot a turn tried, in order: the grid's, those about the
/// problem's `near`, then `nearer`, those set closer about the grid's near misses.
struct Tried {
    outcomes: Vec<Outcome>,
    nearer: Vec<Vec2>,
}

impl<'a> Problem<'a> {
    fn new(p: &RestProblem, seen: &'a Seen, o: &'a RestOptions) -> Self {
        let shape = Shape::of(&p.object);
        let setting = Setting {
            field: &seen.field,
            shape,
            grown: shape.grown(o.clearance),
            o,
        };
        let (turns, first) = turns(shape, p.rotation, o);
        let spots = seen.spots;
        let mut closer: Vec<Vec2> = vec![];
        if let Some(near) = p.near {
            let [across, rows] = Closer::about(near.truncate(), &spots, o);
            for j in 0..rows.count {
                for i in 0..across.count {
                    let cells = Vec2::new(across.first + i as f32, rows.first + j as f32);
                    let aside = cells * o.touching;
                    if aside.length() <= o.spacing {
                        closer.push(near.truncate() + aside);
                    }
                }
            }
        }
        Self {
            setting,
            spots,
            turns,
            first,
            closer,
            near: p.near.map(Vec3::truncate),
        }
    }

    /// Turn `k` tried over the grid's spots and those about `near`, then over the spots set
    /// closer about the grid's near misses ([`seeds`], [`nearer`]), where there are any.
    fn tried(&self, k: usize) -> Tried {
        let (setting, spots, o) = (&self.setting, &self.spots, self.setting.o);
        let turned = self.turns[k];
        let mut outcomes: Vec<Outcome> = (0..spots.count() + self.closer.len())
            .into_par_iter()
            .map_init(Scratch::default, |scratch, k| {
                let over = match k.checked_sub(spots.count()) {
                    None => spots.at(k, o.spacing),
                    Some(k) => self.closer[k],
                };
                setting.rest::<false>(turned, over, scratch)
            })
            .collect();
        let seeds = seeds(&outcomes[..spots.count()], spots.across, spots.rows);
        let at = seeds.iter().map(|&k| spots.at(k, o.spacing));
        let nearer = nearer(at, spots, self.near, o);
        if !nearer.is_empty() {
            let more = nearer
                .par_iter()
                .map_init(Scratch::default, |scratch, &over| {
                    setting.rest::<false>(turned, over, scratch)
                });
            outcomes.par_extend(more);
        }
        Tried { outcomes, nearer }
    }
}

/// The grid's spots, `across` by `rows` with `outcomes` row by row, that the spots set closer
/// start from: those where the object held but tipped too easily, with no rest at any of the
/// eight spots around and at least one around where it was not held at all, or off the grid. A
/// rest beside it means the place was found: on a table's rim a spot that overhangs sits beside
/// one that rests, and nothing more is looked for there. One beside that does not hold means a
/// rest may lie between: where every spot tips, as a pen stood on a table, only the edge of the
/// support is looked at closer.
fn seeds(outcomes: &[Outcome], across: usize, rows: usize) -> Vec<usize> {
    let mut seeds = vec![];
    let tipping = (0..outcomes.len()).filter(|&k| matches!(outcomes[k], Outcome::Tips));
    for k in tipping {
        let (column, row) = ((k % across) as isize, (k / across) as isize);
        let (mut rests, mut unheld) = (false, false);
        let around = (-1..=1).flat_map(|dr| (-1..=1).map(move |dc| (dc, dr)));
        for (dc, dr) in around.filter(|&d| d != (0, 0)) {
            let (c, r) = (column + dc, row + dr);
            if c < 0 || r < 0 || c >= across as isize || r >= rows as isize {
                unheld = true;
                continue;
            }
            match outcomes[r as usize * across + c as usize] {
                Outcome::Rests(..) => rests = true,
                Outcome::Unheld | Outcome::Unsupported => unheld = true,
                Outcome::Tips | Outcome::Blocked => {}
            }
        }
        if unheld && !rests {
            seeds.push(k);
        }
    }
    seeds
}

/// The spots set closer about `seeds`: the points a cell (`touching`) apart on the lattice the
/// field's cells are centred on, within `spacing` of a seed seen from above and over the
/// support's box, once each, row by row, so that their order does not depend on the threads.
/// Where `near` lies on that lattice its spots are some of these, and are left to it.
fn nearer(
    seeds: impl Iterator<Item = Vec2>,
    spots: &Spots,
    near: Option<Vec2>,
    o: &RestOptions,
) -> Vec<Vec2> {
    let cell = o.touching;
    let lower = (spots.lower / cell).ceil().as_ivec2();
    let upper = (spots.upper / cell).floor().as_ivec2();
    let size = (upper - lower + 1).max(IVec2::ZERO);
    let mut marked: Vec<bool> = vec![];
    for seed in seeds {
        if marked.is_empty() {
            marked = vec![false; size.x as usize * size.y as usize];
        }
        let from = ((seed - o.spacing).max(spots.lower) / cell)
            .ceil()
            .as_ivec2();
        let to = ((seed + o.spacing).min(spots.upper) / cell)
            .floor()
            .as_ivec2();
        for y in from.y..=to.y {
            for x in from.x..=to.x {
                let at = IVec2::new(x, y);
                if (at.as_vec2() * cell).distance(seed) <= o.spacing {
                    let k = at - lower;
                    marked[k.y as usize * size.x as usize + k.x as usize] = true;
                }
            }
        }
    }
    // `near`'s spots lie on the lattice where it does, to a rounding.
    let on = near
        .map(|near| near / cell)
        .filter(|n| (*n - n.round()).abs().max_element() < 1e-3);
    let taken =
        |at: IVec2| on.is_some_and(|n| ((at.as_vec2() - n.round()) * cell).length() <= o.spacing);
    let mut nearer = vec![];
    for k in (0..marked.len()).filter(|&k| marked[k]) {
        let at = lower + IVec2::new((k % size.x as usize) as i32, (k / size.x as usize) as i32);
        if !taken(at) {
            nearer.push(at.as_vec2() * cell);
        }
    }
    nearer
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid of outcomes from one letter a spot, the first row first: `r` rests, `t` tips, `b`
    /// is blocked, `u` is not held, `-` has nothing under it.
    fn grid(rows: &[&str]) -> (Vec<Outcome>, usize, usize) {
        let rest = Rest {
            pose: Pose::IDENTITY,
            support: vec![],
            tipping: 0.1,
        };
        let each = rows.iter().flat_map(|row| row.chars());
        let outcomes = each
            .map(|c| match c {
                'r' => Outcome::Rests(rest.clone(), false),
                't' => Outcome::Tips,
                'b' => Outcome::Blocked,
                'u' => Outcome::Unheld,
                _ => Outcome::Unsupported,
            })
            .collect();
        (outcomes, rows[0].len(), rows.len())
    }

    /// Where every spot tips, the seeds are its edge, whose neighbors off the grid hold nothing;
    /// a spot that tips beside a rest is none, and one beside a spot that holds nothing is one.
    #[test]
    fn looking_closer_starts_where_a_spot_tips_between_none_and_nothing() {
        let (all, across, rows) = grid(&["tttt", "tttt", "tttt", "tttt"]);
        let edge: Vec<usize> = (0..16)
            .filter(|&k| k % 4 == 0 || k % 4 == 3 || k / 4 == 0 || k / 4 == 3)
            .collect();
        assert_eq!(seeds(&all, across, rows), edge);
        let (beside_a_rest, across, rows) = grid(&["uuu", "utr", "uuu"]);
        assert!(seeds(&beside_a_rest, across, rows).is_empty());
        let (beside_nothing, across, rows) = grid(&["ttt", "tt-", "ttt"]);
        assert_eq!(
            seeds(&beside_nothing, across, rows),
            [0, 1, 2, 3, 4, 6, 7, 8]
        );
        // Inside, among spots that tip or are blocked, none is; beside one not held, those
        // around it are, and those beside a rest are not.
        let (inside, across, rows) = grid(&["bbbbb", "btttb", "btttb", "btttb", "bbbbb"]);
        assert!(seeds(&inside, across, rows).is_empty());
        let (unheld, across, rows) = grid(&["bbbbb", "btttb", "bttub", "btttb", "bbbbb"]);
        assert_eq!(seeds(&unheld, across, rows), [7, 8, 12, 17, 18]);
        let (rested, across, rows) = grid(&["bbbbb", "btttb", "bttub", "bttrb", "bbbbb"]);
        assert_eq!(seeds(&rested, across, rows), [7, 8]);
    }
}

impl Setting<'_> {
    /// `missed`, what became of the object set down turned by `rotation`; with `WHY`, where it
    /// touched the support, what would be in its way where it was set down left in `s`.
    #[inline(always)]
    fn missed<const WHY: bool>(&self, missed: Outcome, rotation: Quat, s: &mut Scratch) -> Outcome {
        if WHY && !matches!(missed, Outcome::Unsupported) {
            let grown = Turned::new(self.grown, rotation);
            self.field.clear::<true>(&grown, s.center, &mut s.blocking);
        }
        missed
    }
}

impl Problem<'_> {
    /// Where spot `k` of a turn `tried` lies, seen from above.
    fn spot(&self, k: usize, tried: &Tried) -> Vec2 {
        let count = self.spots.count();
        match k.checked_sub(count) {
            None => self.spots.at(k, self.setting.o.spacing),
            Some(k) if k < self.closer.len() => self.closer[k],
            Some(k) => tried.nearer[k - self.closer.len()],
        }
    }
}

/// Why a problem of [`rests_on`] has no rest ([`rest_failures`]): how near the search came.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RestFailure {
    /// Nothing of the support lies under the object anywhere it was set down.
    Unsupported,
    /// Set down, it touches the support and that does not hold it.
    Unheld {
        /// The object's frame set down there.
        pose: Pose,
        /// The corners of what it touches there, as [`Rest::support`].
        support: Vec<Vec3>,
        /// The unit mean of the directions the support pushes it along there: up on a level
        /// support, leaning where it slopes, as the rotation that lies flat on a slope takes
        /// the vertical to; zero where it touches nothing.
        push: Vec3,
        /// What would be in its way there even if it were held.
        blocked: Option<Blocking>,
    },
    /// It is held, but its weight can be tipped by less than [`RestOptions::tipping`].
    Tips {
        /// The object's frame set down there.
        pose: Pose,
        /// The corners of what it touches there, as [`Rest::support`].
        support: Vec<Vec3>,
        /// How far its weight can be tipped there before what it touches no longer holds it
        /// (radians), found as a rest's is.
        tipping: f32,
        /// What would be in its way there even if it held.
        blocked: Option<Blocking>,
    },
    /// It rests, but for a point in its way.
    Blocked {
        /// The rest, as [`rests_on`] would give it.
        rest: Rest,
        /// What is in its way.
        by: Blocking,
    },
}

/// A point in the way of an object set down: within [`RestOptions::clearance`] of it, or over it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Blocking {
    /// The point, the first the check meets.
    pub point: Vec3,
    /// Whether it is the support's own, an underside of it that faces down and was not seen
    /// from above, and not a reading left unflagged. Flags that leave out part of what the
    /// object would lie against give points that are not; a support seen from below gives
    /// points that are.
    pub support: bool,
}

/// Why each of `problems` has no rest, given `rests`, what [`rests_on`] found for them with `o`:
/// `None` where it has one. The answer is how far the search came: in this order, a place where
/// it rests but for a point in its way ([`RestFailure::Blocked`], the steadiest such place), one
/// where it is held but tips too easily (`Tips`, the steadiest), one where it touches the
/// support and is not held (`Unheld`, the one nearest the problem's `near`, or without one the
/// middle of the support seen from above), and none where nothing of the support lies under it
/// anywhere (`Unsupported`); of two alike, that at the lower turn, then at the lower spot.
///
/// Where it is not held or tips, what would be in its way there even if it held comes with it:
/// flags that leave out part of what the object must lie against both leave it unheld and put
/// that part in its way. The problems with no rest are searched again, every spot's outcome
/// kept, and the places of the furthest stage judged once more for their details, so this costs
/// about one more search of those problems; it runs only when asked for, here.
pub fn rest_failures(
    device: &Device,
    problems: &[RestProblem],
    surfaces: &[Surface],
    rests: &[Vec<Rest>],
    o: &RestOptions,
) -> Result<Vec<Option<RestFailure>>> {
    Read::checked(problems, surfaces, o)?;
    ensure_input!(
        rests.len() == problems.len(),
        "{} lists of rests for {} problems",
        rests.len(),
        problems.len()
    );
    let failed: Vec<usize> = (0..problems.len())
        .filter(|&k| rests[k].is_empty())
        .collect();
    let mut out: Vec<Option<RestFailure>> = vec![None; problems.len()];
    if failed.is_empty() {
        return Ok(out);
    }
    let again: Vec<RestProblem> = failed.iter().map(|&k| problems[k].clone()).collect();
    let read = Read::checked(&again, surfaces, o)?;
    let each = || {
        let seen = read.seen(surfaces, o);
        let each = again.par_iter().map(|p| match &seen[p.surface] {
            Some(seen) => failure(p, seen, &surfaces[p.surface], o),
            None => Some(RestFailure::Unsupported),
        });
        each.collect::<Vec<_>>()
    };
    for (k, failure) in failed.into_iter().zip(device.install(each)) {
        out[k] = failure;
    }
    Ok(out)
}

/// How far the search came for `p` on `surface`, read as `seen`; `None` where it rests.
fn failure(
    p: &RestProblem,
    seen: &Seen,
    surface: &Surface,
    o: &RestOptions,
) -> Option<RestFailure> {
    let problem = Problem::new(p, seen, o);
    let kinds: Vec<usize> = (0..problem.turns.len())
        .filter(|&k| problem.first[k] == k)
        .collect();
    let tried: Vec<Tried> = kinds.par_iter().map(|&k| problem.tried(k)).collect();
    // How far each outcome came.
    let stage = |f: &Outcome| match f {
        Outcome::Unsupported => 0,
        Outcome::Unheld => 1,
        Outcome::Tips => 2,
        Outcome::Blocked => 3,
        Outcome::Rests(..) => 4,
    };
    let all = tried.iter().flat_map(|t| &t.outcomes);
    let furthest = all.map(stage).max().unwrap_or(0);
    match furthest {
        0 => return Some(RestFailure::Unsupported),
        4 => return None,
        _ => {}
    }
    // The places of that stage, by turn and spot, in order.
    let mut candidates: Vec<(usize, usize)> = vec![];
    for (t, tried) in tried.iter().enumerate() {
        let of_it = (0..tried.outcomes.len()).filter(|&k| stage(&tried.outcomes[k]) == furthest);
        candidates.extend(of_it.map(|k| (t, k)));
    }
    let judge = |t: usize, k: usize, s: &mut Scratch| {
        let over = problem.spot(k, &tried[t]);
        problem
            .setting
            .rest::<true>(problem.turns[kinds[t]], over, s)
    };
    let (t, k) = if furthest == 1 {
        // Not held: nearest where the caller would have it, or the support's middle.
        let target = problem
            .near
            .unwrap_or(0.5 * (problem.spots.lower + problem.spots.upper));
        let distance = |&(t, k): &(usize, usize)| problem.spot(k, &tried[t]).distance(target);
        let nearest = candidates
            .iter()
            .copied()
            .reduce(|a, b| if distance(&b) < distance(&a) { b } else { a });
        nearest.expect("a place of the furthest stage")
    } else {
        // Held: the steadiest.
        let steadiness: Vec<f32> = candidates
            .par_iter()
            .map_init(Scratch::default, |s, &(t, k)| {
                judge(t, k, s);
                s.tipping
            })
            .collect();
        let steadiest =
            (0..candidates.len()).reduce(|a, b| if steadiness[b] > steadiness[a] { b } else { a });
        candidates[steadiest.expect("a place of the furthest stage")]
    };
    let mut s = Scratch::default();
    let outcome = judge(t, k, &mut s);
    let pose = Pose::new(s.center, problem.turns[kinds[t]]);
    let support = s.statics.support(&s.contacts);
    let blocking = s.blocking.map(|point| {
        let at = surface.points.iter().position(|q| *q == point);
        Blocking {
            point,
            support: at.is_some_and(|at| surface.object.is_empty() || surface.object[at]),
        }
    });
    Some(match outcome {
        Outcome::Blocked => RestFailure::Blocked {
            rest: Rest {
                pose,
                support,
                tipping: s.tipping,
            },
            by: blocking.expect("a point in the way of what is blocked"),
        },
        Outcome::Tips => RestFailure::Tips {
            pose,
            support,
            tipping: s.tipping,
            blocked: blocking,
        },
        _ => {
            let push: Vec3 = s.contacts.iter().map(|c| c.push).sum();
            RestFailure::Unheld {
                pose,
                support,
                push: push.normalize_or_zero(),
                blocked: blocking,
            }
        }
    })
}
