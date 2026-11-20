//! Whether an object stays where it was set down, and how far its weight can be tipped before
//! it does not: the contacts as stiff springs that push and do not pull, which the object settles
//! onto by a small rigid motion until they carry its weight.
//!
//! The forces are what the settle leaves, not any set that would balance: a contact the weight
//! does not press carries nothing, so it has no friction to offer. Springs alone are tried first;
//! Pang and Trinkle ("Stability characterizations of fixtured rigid bodies with Coulomb
//! friction", 2000, Theorem 1) show that what holds without friction holds with any. Otherwise
//! each spring resists across its direction too, and slips where that would take its force out
//! of its friction cone. Where every contact lies on one flat face of the object the settle has
//! a closed form: the weight's line meets the face inside the hull of the contacts, within the
//! friction cone about the face's normal.

use std::f32::consts::{FRAC_PI_2, TAU};

use glam::{DVec3, Vec2, Vec3};

use super::shape::{Piece, ROUND};

/// Where the support pushes on the object.
#[derive(Clone, Copy, Debug)]
pub(super) struct Contact {
    /// The support's point.
    pub(super) point: Vec3,
    /// Where on the object it pushes.
    pub(super) at: Vec3,
    /// The unit direction it pushes the object along.
    pub(super) push: Vec3,
    /// The flat piece of the object it lies on, where the support's point pushes where it is:
    /// those of one piece lie in one plane.
    pub(super) piece: Piece,
}

/// The directions around the vertical the weight is tipped toward.
const TILTS: usize = 8;

/// How much of the weight may be left unbalanced by a settle that holds.
const BALANCE: f64 = 1e-6;

/// The most rounds a settle takes: one whose contacts go on changing between holding, slipping
/// and letting go past it does not hold.
const ROUNDS: usize = 64;

/// How finely the angle the weight can be tipped by is found where it has no closed form
/// (radians): half a degree.
const FINE: f32 = 0.0087;

/// A wrench: a force and its torque about the object's middle.
type Wrench = [f64; 6];

/// The wrench of force `f` at `r` from the middle.
fn wrench(r: DVec3, f: DVec3) -> Wrench {
    let t = r.cross(f);
    [f.x, f.y, f.z, t.x, t.y, t.z]
}

/// The weight, tipped by `tilt` toward the direction `around` the vertical.
fn weight(tilt: f32, around: f32) -> Wrench {
    let (tilt, around) = (f64::from(tilt), f64::from(around));
    let across = tilt.sin();
    [
        across * around.cos(),
        across * around.sin(),
        -tilt.cos(),
        0.0,
        0.0,
        0.0,
    ]
}

/// The corners of the convex hull of `points`, anticlockwise, by Andrew's monotone chain, as
/// indices into `points`, into `corners`, sorting them in `order`: one or two where they are all
/// at one point or on one line. Points that come row by row, as a field's cells give them, are
/// in order already, or nearly.
fn hull(points: &[Vec2], order: &mut Vec<(u64, u32)>, corners: &mut Vec<u32>) {
    // A coordinate as a whole number that sorts as it does.
    let key = |v: f32| {
        let bits = (v + 0.0).to_bits();
        u64::from(if bits >> 31 == 0 {
            bits | 1 << 31
        } else {
            !bits
        })
    };
    order.clear();
    let keyed = points
        .iter()
        .enumerate()
        .map(|(k, p)| (key(p.y) << 32 | key(p.x), k as u32));
    order.extend(keyed);
    order.sort_unstable();
    order.dedup_by_key(|k| k.0);
    corners.clear();
    if order.len() < 3 {
        corners.extend(order.iter().map(|k| k.1));
        return;
    }
    let at = |k: u32| points[k as usize];
    // Whether `c` lies to the right of the line from `a` to `b`: sorted by rows, the chain
    // runs clockwise, and is turned round at the end.
    let right = |a: u32, b: u32, c: u32| (at(b) - at(a)).perp_dot(at(c) - at(a)) < 0.0;
    for half in 0..2 {
        let start = corners.len();
        let push = |k: u32, corners: &mut Vec<u32>| {
            while corners.len() >= start + 2
                && !right(corners[corners.len() - 2], corners[corners.len() - 1], k)
            {
                corners.pop();
            }
            corners.push(k);
        };
        if half == 0 {
            for k in order.iter() {
                push(k.1, corners);
            }
        } else {
            for k in order.iter().rev() {
                push(k.1, corners);
            }
        }
        corners.pop();
    }
    corners.reverse();
}

/// How far along the unit `along` from `from` the hull with `corners` (anticlockwise) is left;
/// none where `from` is outside it.
fn exit(corners: &[Vec2], from: Vec2, along: Vec2) -> Option<f32> {
    let mut nearest = f32::INFINITY;
    for (k, &a) in corners.iter().enumerate() {
        let edge = corners[(k + 1) % corners.len()] - a;
        let inside = edge.perp_dot(from - a);
        if inside < -1e-7 {
            return None;
        }
        let out = -edge.perp_dot(along);
        if out > 0.0 {
            nearest = nearest.min(inside.max(0.0) / out);
        }
    }
    Some(nearest)
}

/// What a thread keeps between the poses it judges.
#[derive(Default)]
pub(super) struct Statics {
    flat: Vec<Vec2>,
    order: Vec<(u64, u32)>,
    corners: Vec<u32>,
    /// Whether `corners` are those of every contact last judged, seen from above.
    from_above: bool,
    outline: Vec<Vec2>,
    kept: Vec<u32>,
    pieces: Vec<(Piece, Vec3)>,
    own: Vec<u32>,
    springs: Springs,
    base: Vec<DVec3>,
    tipped: Vec<DVec3>,
    /// How far the weight could be tipped where [`holds`](Self::holds) with `WHY` last found it
    /// tips (radians).
    pub(super) tips_by: f32,
}

impl Statics {
    /// Whether `contacts` hold the weight of an object whose middle is `center`, and how far it can
    /// be tipped any way before they no longer do (radians), where that is at least `tipping`.
    /// `size` is the object's, the unit its torques are balanced in.
    /// With `WHY`, where it tips, how far it can be tipped is left in `tips_by`.
    pub(super) fn holds<const WHY: bool>(
        &mut self,
        contacts: &[Contact],
        center: Vec3,
        size: f32,
        tipping: f32,
        friction: f32,
    ) -> Holding {
        self.from_above = false;
        let Some(first) = contacts.first() else {
            return Holding::Not;
        };
        // One flat piece flush with the support is one face, pushed one way: along the mean,
        // where the support is measured in patches whose planes differ by its noise, by less
        // than the angle is found to elsewhere.
        let near = FINE.cos();
        let agrees = |c: &Contact| c.piece == first.piece && c.push.dot(first.push) >= near;
        if first.piece != ROUND && contacts.iter().all(agrees) {
            let mut push = first.push;
            if contacts.iter().any(|c| c.push != first.push) {
                push = contacts.iter().map(|c| c.push).sum::<Vec3>().normalize();
            }
            return self.on_one_face::<WHY>(contacts, push, center, tipping, friction);
        }
        self.settled::<WHY>(contacts, center, size, tipping, friction)
    }

    /// What `contacts`, those last judged, touch: the corners of the hull of the support's
    /// points, seen from above, anticlockwise.
    pub(super) fn support(&mut self, contacts: &[Contact]) -> Vec<Vec3> {
        if !self.from_above {
            self.flat.clear();
            self.flat
                .extend(contacts.iter().map(|c| c.point.truncate()));
            hull(&self.flat, &mut self.order, &mut self.corners);
        }
        let corners = self.corners.iter().map(|&k| contacts[k as usize].point);
        corners.collect()
    }

    /// The hull of `contacts`, all on one flat piece of the object whose push is `push`, as their
    /// places across it from the first: its corners into `self.corners`, and where they are into
    /// `self.outline`, anticlockwise seen from the object.
    fn across(&mut self, contacts: impl Iterator<Item = Contact>, origin: Vec3, axes: [Vec3; 2]) {
        self.flat.clear();
        let flat = contacts.map(|c| {
            let from = c.at - origin;
            Vec2::new(from.dot(axes[0]), from.dot(axes[1]))
        });
        self.flat.extend(flat);
        hull(&self.flat, &mut self.order, &mut self.corners);
        self.outline.clear();
        let outline = self.corners.iter().map(|&k| self.flat[k as usize]);
        self.outline.extend(outline);
    }

    /// [`holds`](Self::holds) where every contact lies on one flat piece of the object and is
    /// pushed along `push`: the weight's line, tipped, must meet the piece inside the hull of the
    /// contacts, within the friction cone about `push`.
    fn on_one_face<const WHY: bool>(
        &mut self,
        contacts: &[Contact],
        push: Vec3,
        center: Vec3,
        tipping: f32,
        friction: f32,
    ) -> Holding {
        let origin = contacts[0].at;
        let above = (center - origin).dot(push);
        if push.z <= 1e-6 || above <= 0.0 {
            return Holding::Not;
        }
        let (u, v) = push.any_orthonormal_pair();
        let axes = [u, v];
        let across = |p: Vec3| Vec2::new((p - origin).dot(axes[0]), (p - origin).dot(axes[1]));
        // The piece is not upright, so what is a corner seen from above is one across it; from
        // above the contacts come in rows. The object's side of the piece is its under side.
        self.flat.clear();
        self.flat.extend(contacts.iter().map(|c| c.at.truncate()));
        hull(&self.flat, &mut self.order, &mut self.corners);
        self.from_above = true;
        self.outline.clear();
        let outline = self
            .corners
            .iter()
            .map(|&k| across(contacts[k as usize].at));
        self.outline.extend(outline);
        if (u.cross(v).dot(push) > 0.0) != (push.z > 0.0) {
            self.outline.reverse();
        }
        if self.outline.len() < 3 {
            return Holding::Not;
        }
        let under = across(center - Vec3::Z * (above / push.z));
        let cone = friction.atan();
        let mut least = FRAC_PI_2;
        for k in 0..TILTS {
            let around = TAU * k as f32 / TILTS as f32;
            let toward = Vec3::new(around.cos(), around.sin(), 0.0);
            // Tipped by `t` toward there, the weight presses along the push by
            // `push.z cos t - (push . toward) sin t` of itself: within the cone while that is at
            // least the cone's cosine.
            let (level, leaning) = (push.z, push.dot(toward));
            let most = level.hypot(leaning);
            if most < cone.cos() {
                return Holding::Not;
            }
            let within = (cone.cos() / most).min(1.0).acos();
            let lean = leaning.atan2(level);
            if lean.abs() > within + 1e-6 {
                return Holding::Not;
            }
            let by_friction = within - lean;
            // Where the weight's line meets the piece moves along the piece as it tips.
            let along = -push.cross(Vec3::Z.cross(toward));
            let along = Vec2::new(along.dot(axes[0]), along.dot(axes[1])).normalize_or_zero();
            let Some(reach) = exit(&self.outline, under, along) else {
                return Holding::Not;
            };
            let edge = under + along * reach;
            let to = origin + axes[0] * edge.x + axes[1] * edge.y - center;
            let by_hull = (-to.z / to.length()).clamp(-1.0, 1.0).acos();
            least = least.min(by_friction).min(by_hull);
        }
        if least >= tipping {
            Holding::By(least)
        } else {
            if WHY {
                self.tips_by = least;
            }
            Holding::Tips(false)
        }
    }

    /// [`holds`](Self::holds) by the settle: the contacts of each flat piece first cut down to
    /// the corners of their hull, which carry any forces the piece's contacts can.
    fn settled<const WHY: bool>(
        &mut self,
        contacts: &[Contact],
        center: Vec3,
        size: f32,
        tipping: f32,
        friction: f32,
    ) -> Holding {
        // The contacts that lie on one flat piece and push one way, together.
        self.kept.clear();
        self.pieces.clear();
        for (k, c) in contacts.iter().enumerate() {
            if c.piece == ROUND {
                self.kept.push(k as u32);
            } else if !self.pieces.contains(&(c.piece, c.push)) {
                self.pieces.push((c.piece, c.push));
            }
        }
        for at in 0..self.pieces.len() {
            let (piece, push) = self.pieces[at];
            let on = |c: &&Contact| c.piece == piece && c.push == push;
            let first = contacts.iter().find(on).expect("a contact of each piece");
            let (u, v) = first.push.any_orthonormal_pair();
            self.across(contacts.iter().filter(on).copied(), first.at, [u, v]);
            self.own.clear();
            let own = (0..contacts.len() as u32).filter(|&k| on(&&contacts[k as usize]));
            self.own.extend(own);
            let corners = self.corners.iter().map(|&k| self.own[k as usize]);
            self.kept.extend(corners);
        }
        self.kept.sort_unstable();
        let springs = &mut self.springs;
        springs.clear();
        for &k in &self.kept {
            let c = &contacts[k as usize];
            springs.r.push(((c.at - center) / size).as_dvec3());
            springs.n.push(c.push.as_dvec3());
        }
        let mu = f64::from(friction);
        // Upright: on the springs alone, or with their friction.
        let upright = weight(0.0, 0.0);
        if !springs.settle(upright, None, None, &mut self.base)
            && !(springs.settle(upright, Some(mu), None, &mut self.base)
                && springs.within_cones(&self.base, mu))
        {
            return Holding::Not;
        }
        // Tipped every way, from the forces it settled on upright.
        let mut tipped_by = |tilt: f32, tipped: &mut Vec<DVec3>| {
            (0..TILTS).all(|k| {
                let load = weight(tilt, TAU * k as f32 / TILTS as f32);
                springs.settle(load, Some(mu), Some(&self.base), tipped)
                    && springs.within_cones(tipped, mu)
            })
        };
        if tipping > 0.0 && !tipped_by(tipping, &mut self.tipped) {
            if WHY {
                // How far it does hold, narrowed below `tipping` as a rest's is above it.
                let (mut holds, mut gives) = (0.0, tipping);
                while gives - holds > FINE {
                    let middle = 0.5 * (holds + gives);
                    if tipped_by(middle, &mut self.tipped) {
                        holds = middle;
                    } else {
                        gives = middle;
                    }
                }
                self.tips_by = holds;
            }
            // Two ways as the one-face test tells one: some push differs from the first's.
            let (first, near) = (contacts[0].push, FINE.cos());
            return Holding::Tips(contacts.iter().any(|c| c.push.dot(first) < near));
        }
        // How much farther: doubled until it gives, then narrowed.
        let (mut holds, mut gives) = (tipping, FRAC_PI_2);
        let mut next = (2.0 * tipping).max(4.0 * FINE);
        while next < gives {
            if tipped_by(next, &mut self.tipped) {
                holds = next;
                next *= 2.0;
            } else {
                gives = next;
            }
        }
        while gives - holds > FINE {
            let middle = 0.5 * (holds + gives);
            if tipped_by(middle, &mut self.tipped) {
                holds = middle;
            } else {
                gives = middle;
            }
        }
        Holding::By(holds)
    }
}

/// The contacts as springs, as stiff across as along, and what a settle keeps of each.
#[derive(Default)]
struct Springs {
    /// Where each is from the object's middle, in the object's size.
    r: Vec<DVec3>,
    /// The unit direction each pushes along.
    n: Vec<DVec3>,
    active: Vec<bool>,
    slips: Vec<bool>,
    /// The unit direction a slipping contact's friction acts in.
    along: Vec<DVec3>,
    next: Vec<(bool, bool, DVec3)>,
}

impl Springs {
    fn clear(&mut self) {
        self.r.clear();
        self.n.clear();
    }

    /// Lets the object settle onto the springs under `load`, the whole wrench they must carry,
    /// from the forces `before` already at them (none: unloaded): with friction `mu` each spring
    /// resists across its direction too, and where its force would leave its friction cone it
    /// slips, carrying `mu` times its push against the slip. The forces come back in `forces`.
    /// Whether the settle came to rest with the load carried.
    fn settle(
        &mut self,
        load: Wrench,
        mu: Option<f64>,
        before: Option<&[DVec3]>,
        forces: &mut Vec<DVec3>,
    ) -> bool {
        let count = self.r.len();
        let before = |i: usize| before.map_or(DVec3::ZERO, |f| f[i]);
        self.active.clear();
        self.active.resize(count, true);
        self.slips.clear();
        self.slips.resize(count, false);
        self.along.clear();
        self.along.resize(count, DVec3::ZERO);
        forces.clear();
        forces.resize(count, DVec3::ZERO);
        let mut settled = false;
        for _ in 0..ROUNDS {
            let mut stiffness = [[0.0f64; 6]; 6];
            let mut carried = load;
            // Adds the spring whose force changes by `-a (b . motion)` with the object's motion.
            let add = |stiffness: &mut [[f64; 6]; 6], a: Wrench, b: Wrench| {
                for (row, &a) in stiffness.iter_mut().zip(&a) {
                    for (entry, &b) in row.iter_mut().zip(&b) {
                        *entry += a * b;
                    }
                }
            };
            for i in (0..count).filter(|&i| self.active[i]) {
                let (r, n) = (self.r[i], self.n[i]);
                let pressed = before(i).dot(n);
                let along = wrench(r, n);
                let held = match mu {
                    None => {
                        add(&mut stiffness, along, along);
                        wrench(r, n * pressed)
                    }
                    Some(mu) if self.slips[i] => {
                        let d = n + self.along[i] * mu;
                        add(&mut stiffness, wrench(r, d), along);
                        wrench(r, d * pressed)
                    }
                    Some(_) => {
                        for axis in DVec3::AXES {
                            add(&mut stiffness, wrench(r, axis), wrench(r, axis));
                        }
                        wrench(r, before(i))
                    }
                };
                for (sum, held) in carried.iter_mut().zip(held) {
                    *sum += held;
                }
            }
            let trace: f64 = (0..6).map(|k| stiffness[k][k]).sum();
            for (k, row) in stiffness.iter_mut().enumerate() {
                row[k] += 1e-9 * (trace / 6.0 + 1e-30);
            }
            let Some(motion) = solve(stiffness, carried) else {
                return false;
            };
            let (shift, turn) = (
                DVec3::new(motion[0], motion[1], motion[2]),
                DVec3::new(motion[3], motion[4], motion[5]),
            );
            self.next.clear();
            let mut changed = false;
            for (i, force) in forces.iter_mut().enumerate().take(count) {
                let (r, n) = (self.r[i], self.n[i]);
                let moved = shift + turn.cross(r);
                let pressed = before(i).dot(n) - moved.dot(n);
                let across = moved - n * moved.dot(n);
                let mut slips = self.slips[i];
                let mut along = self.along[i];
                *force = DVec3::ZERO;
                if self.active[i] {
                    match mu {
                        None => *force = n * pressed,
                        Some(mu) if slips => {
                            *force = (n + along * mu) * pressed;
                            // Slipping the way its friction pushes, it holds again.
                            if across.dot(along) > 1e-15 {
                                slips = false;
                            }
                        }
                        Some(mu) => {
                            let held = before(i) - n * before(i).dot(n) - across;
                            *force = n * pressed + held;
                            let size = held.length();
                            if size > mu * pressed.max(0.0) + 1e-12 {
                                slips = true;
                                along = held / size;
                            }
                        }
                    }
                }
                let active = pressed > 1e-12;
                changed |= active != self.active[i] || slips != self.slips[i];
                self.next.push((active, slips, along));
            }
            if !changed {
                settled = true;
                break;
            }
            for (i, &(active, slips, along)) in self.next.iter().enumerate() {
                self.active[i] = active;
                self.slips[i] = slips;
                self.along[i] = along;
            }
        }
        if !settled {
            return false;
        }
        let mut left = load;
        for (i, force) in forces.iter_mut().enumerate().take(count) {
            if !self.active[i] {
                *force = DVec3::ZERO;
            }
            for (sum, held) in left.iter_mut().zip(wrench(self.r[i], *force)) {
                *sum += held;
            }
        }
        left.iter().map(|v| v * v).sum::<f64>().sqrt() <= BALANCE
    }

    /// Whether each of `forces` pushes and lies within its contact's friction cone.
    fn within_cones(&self, forces: &[DVec3], mu: f64) -> bool {
        forces.iter().zip(&self.n).all(|(f, n)| {
            let along = f.dot(*n);
            along >= -1e-9 && (*f - *n * along).length() <= mu * along + 1e-9
        })
    }
}

/// The solution of six linear equations, by elimination with the largest pivot of each column;
/// none where they do not determine one.
fn solve(mut a: [[f64; 6]; 6], mut b: Wrench) -> Option<Wrench> {
    for column in 0..6 {
        let pivot = (column..6)
            .max_by(|&p, &q| a[p][column].abs().total_cmp(&a[q][column].abs()))
            .expect("a row at or below the diagonal");
        if a[pivot][column].abs() < 1e-300 {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        for row in column + 1..6 {
            let factor = a[row][column] / a[column][column];
            if factor == 0.0 {
                continue;
            }
            let above = a[column];
            for (below, above) in a[row].iter_mut().zip(above).skip(column) {
                *below -= factor * above;
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = [0.0; 6];
    for row in (0..6).rev() {
        let known: f64 = (row + 1..6).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - known) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// What an object's contacts do with its weight ([`Statics::holds`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Holding {
    /// They hold it with its weight tipped this far any way (radians), at least as far as asked.
    By(f32),
    /// They hold it upright, but not with its weight tipped as far as asked; and whether they
    /// push it two ways, some push differing from the first's by more than `FINE`. Those on one
    /// flat face push one way.
    Tips(bool),
    /// They do not hold it.
    Not,
}
