//! Collision checking for eight configurations at a time on the CPU, in SIMD lanes
//! (`fearless_simd::f32x8`, on the widest instructions the CPU has, chosen when the program runs):
//! forward kinematics, link-bound gating, and world and self clearance. The math is
//! `cpu::collision`'s without cost or gradient. The CPU device's clearances and edge checks run
//! here. `cpu::collision` stays the line-by-line twin of the GPU kernels, and tests hold the two
//! together.

use fearless_simd::prelude::*;
use fearless_simd::{Level, Simd, f32x8};
use glam::{Mat3, Vec3};

use crate::cpu::Prepared;
use crate::robot::{JointKind, Robot};
use crate::world::{self, FAR};

/// Configurations per call.
pub(crate) const LANES: usize = 8;

type F<S> = f32x8<S>;

/// Eight lanes kept in memory between calls, which may run at different SIMD levels.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Stored([f32; LANES]);

const STORED_ZERO: Stored = Stored([0.0; LANES]);

#[inline(always)]
fn load<S: Simd>(simd: S, v: &Stored) -> F<S> {
    F::simd_from(simd, v.0)
}

#[inline(always)]
fn store<S: Simd>(v: F<S>) -> Stored {
    Stored(v.into())
}

#[derive(Clone, Copy)]
struct V3<S: Simd> {
    x: F<S>,
    y: F<S>,
    z: F<S>,
}

impl<S: Simd> V3<S> {
    #[inline(always)]
    fn splat(simd: S, v: Vec3) -> Self {
        Self {
            x: F::splat(simd, v.x),
            y: F::splat(simd, v.y),
            z: F::splat(simd, v.z),
        }
    }

    #[inline(always)]
    fn load(simd: S, v: &[Stored; 3]) -> Self {
        Self {
            x: load(simd, &v[0]),
            y: load(simd, &v[1]),
            z: load(simd, &v[2]),
        }
    }

    #[inline(always)]
    fn store(self) -> [Stored; 3] {
        [store(self.x), store(self.y), store(self.z)]
    }

    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }

    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }

    #[inline(always)]
    fn scale(self, s: F<S>) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    #[inline(always)]
    fn length_squared(self) -> F<S> {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline(always)]
    fn length(self) -> F<S> {
        self.length_squared().sqrt()
    }

    /// `rᵀ * (self - c)` for a constant rotation `r`: the point in a frame placed at `c`.
    #[inline(always)]
    fn into_frame(self, r: &Mat3, c: Vec3) -> Self {
        let d = self.sub(V3::splat(self.x.simd, c));
        Self {
            x: d.dot(r.x_axis),
            y: d.dot(r.y_axis),
            z: d.dot(r.z_axis),
        }
    }

    /// `into_frame` for a frame turned about z only (`Prepared::Cuboid::upright`): the terms with
    /// its zeros dropped, which leaves every coordinate as `into_frame` computes it.
    #[inline(always)]
    fn into_upright_frame(self, r: &Mat3, c: Vec3) -> Self {
        let d = self.sub(V3::splat(self.x.simd, c));
        Self {
            x: d.x * r.x_axis.x + d.y * r.x_axis.y,
            y: d.x * r.y_axis.x + d.y * r.y_axis.y,
            z: d.z,
        }
    }

    /// `self · a` for a constant `a`.
    #[inline(always)]
    fn dot(self, a: Vec3) -> F<S> {
        self.x * a.x + self.y * a.y + self.z * a.z
    }
}

/// A rotation per lane, by columns.
#[derive(Clone, Copy)]
struct M3<S: Simd>([V3<S>; 3]);

impl<S: Simd> M3<S> {
    #[inline(always)]
    fn splat(simd: S, m: Mat3) -> Self {
        Self([
            V3::splat(simd, m.x_axis),
            V3::splat(simd, m.y_axis),
            V3::splat(simd, m.z_axis),
        ])
    }

    #[inline(always)]
    fn load(simd: S, m: &[[Stored; 3]; 3]) -> Self {
        Self([
            V3::load(simd, &m[0]),
            V3::load(simd, &m[1]),
            V3::load(simd, &m[2]),
        ])
    }

    #[inline(always)]
    fn store(self) -> [[Stored; 3]; 3] {
        self.0.map(V3::store)
    }

    /// `self * v` for a constant `v`.
    #[inline(always)]
    fn times(&self, v: Vec3) -> V3<S> {
        let [a, b, c] = self.0;
        let simd = a.x.simd;
        a.scale(F::splat(simd, v.x))
            .add(b.scale(F::splat(simd, v.y)))
            .add(c.scale(F::splat(simd, v.z)))
    }

    #[inline(always)]
    fn times_lanes(&self, v: V3<S>) -> V3<S> {
        let [a, b, c] = self.0;
        a.scale(v.x).add(b.scale(v.y)).add(c.scale(v.z))
    }

    #[inline(always)]
    fn times_matrix(&self, m: Mat3) -> Self {
        Self([
            self.times(m.x_axis),
            self.times(m.y_axis),
            self.times(m.z_axis),
        ])
    }

    #[inline(always)]
    fn times_lanes_matrix(&self, m: &M3<S>) -> Self {
        Self([
            self.times_lanes(m.0[0]),
            self.times_lanes(m.0[1]),
            self.times_lanes(m.0[2]),
        ])
    }
}

/// A link pair's sphere pairs for `keeps`: the link whose bounding sphere is the tighter, and each
/// sphere of the other link with the spheres it pairs with on that one.
type SphereGroups = (usize, Vec<(usize, Vec<usize>)>);

/// A link's joint, prepared for lanes.
#[derive(Clone, Debug)]
enum Joint {
    Fixed {
        rot: Mat3,
        trans: Vec3,
    },
    /// The rotation relative to the parent is `fixed + cos(angle) * cos + sin(angle) * sin`:
    /// Rodrigues' formula with the joint origin folded in.
    Revolute {
        dof: usize,
        curve: [f32; 5],
        trans: Vec3,
        fixed: Mat3,
        cos: Mat3,
        sin: Mat3,
    },
    /// `axis` is in the parent's frame (the joint origin applied).
    Prismatic {
        dof: usize,
        curve: [f32; 5],
        rot: Mat3,
        trans: Vec3,
        axis: Vec3,
    },
}

#[derive(Clone, Debug)]
struct LaneLink {
    parent: Option<usize>,
    joint: Joint,
}

/// A robot prepared for lanes.
#[derive(Clone, Debug)]
pub(crate) struct Lanes {
    dof: usize,
    links: Vec<LaneLink>,
    /// Links checked against the world: each with its spheres `first..first + count` and its
    /// bounding sphere in its frame.
    world_links: Vec<(usize, usize, usize, [f32; 4])>,
    /// Sphere centres in their link's frame, and their radii with and without the self buffer.
    centers: Vec<Vec3>,
    radius: Vec<f32>,
    self_radius: Vec<f32>,
    /// Each link's spheres, `first..end`.
    link_spheres: Vec<(usize, usize)>,
    /// Link pairs checked for self-collision, with their sphere pairs.
    pairs: Vec<(usize, usize, Vec<[usize; 2]>)>,
    /// The same sphere pairs grouped for `keeps`, per link pair.
    pair_groups: Vec<SphereGroups>,
    bounds: Vec<[f32; 4]>,
    /// The widest SIMD instructions the CPU has.
    level: Level,
}

/// Working memory for one call.
pub(crate) struct LaneScratch {
    q: Vec<Stored>,
    rot: Vec<[[Stored; 3]; 3]>,
    pos: Vec<[Stored; 3]>,
    centers: Vec<[Stored; 3]>,
    /// Whether each link's sphere centres are in `centers` for the configurations loaded.
    placed: Vec<bool>,
    /// Each link's bounding sphere's centre, and whether it is there for the configurations
    /// loaded.
    bound: Vec<[Stored; 3]>,
    bounded: Vec<bool>,
}

impl Lanes {
    pub(crate) fn new(robot: &Robot) -> Self {
        let links = robot
            .links
            .iter()
            .map(|l| {
                let (o, trans) = (l.origin.rot, l.origin.trans);
                let joint = match l.joint {
                    JointKind::Fixed => Joint::Fixed { rot: o, trans },
                    JointKind::Revolute {
                        dof,
                        axis: a,
                        curve,
                    } => {
                        let outer = Mat3::from_cols(a * a.x, a * a.y, a * a.z);
                        let cross = Mat3::from_cols(
                            Vec3::new(0.0, a.z, -a.y),
                            Vec3::new(-a.z, 0.0, a.x),
                            Vec3::new(a.y, -a.x, 0.0),
                        );
                        Joint::Revolute {
                            dof,
                            curve: curve.0,
                            trans,
                            fixed: o * outer,
                            cos: o * (Mat3::IDENTITY - outer),
                            sin: o * cross,
                        }
                    }
                    JointKind::Prismatic { dof, axis, curve } => Joint::Prismatic {
                        dof,
                        curve: curve.0,
                        rot: o,
                        trans,
                        axis: o * axis,
                    },
                };
                LaneLink {
                    parent: l.parent,
                    joint,
                }
            })
            .collect();
        let world_links = robot
            .sphere_ranges
            .iter()
            .enumerate()
            .filter(|&(link, &[_, count])| count > 0 && !robot.links[link].chain.is_empty())
            .map(|(link, &[first, count])| {
                (
                    link,
                    first as usize,
                    (first + count) as usize,
                    robot.link_bounds[link],
                )
            })
            .collect();
        let pairs: Vec<(usize, usize, Vec<[usize; 2]>)> = robot
            .self_link_pairs
            .iter()
            .map(|lp| {
                let spheres = &robot.self_pairs[lp.first as usize..(lp.first + lp.count) as usize];
                (
                    lp.a as usize,
                    lp.b as usize,
                    spheres
                        .iter()
                        .map(|&[a, b]| [a as usize, b as usize])
                        .collect(),
                )
            })
            .collect();
        let pair_groups = pairs
            .iter()
            .map(|(a, b, spheres)| {
                let (a, b) = (*a, *b);
                let tight = if robot.link_bounds[a][3] < robot.link_bounds[b][3] {
                    a
                } else {
                    b
                };
                let mut groups: Vec<(usize, Vec<usize>)> = vec![];
                for &[i, j] in spheres {
                    // `i` on `a`, `j` on `b`; grouped by the sphere not on the tighter link.
                    let (other, on_tight) = if tight == b { (i, j) } else { (j, i) };
                    match groups.iter_mut().find(|(k, _)| *k == other) {
                        Some((_, js)) => js.push(on_tight),
                        None => groups.push((other, vec![on_tight])),
                    }
                }
                (tight, groups)
            })
            .collect();
        Self {
            dof: robot.dof(),
            links,
            world_links,
            centers: robot.spheres.iter().map(|s| s.center).collect(),
            link_spheres: robot
                .sphere_ranges
                .iter()
                .map(|&[first, count]| (first as usize, (first + count) as usize))
                .collect(),
            radius: robot.spheres.iter().map(|s| s.radius).collect(),
            self_radius: robot
                .spheres
                .iter()
                .map(|s| s.radius + s.self_buffer)
                .collect(),
            pairs,
            pair_groups,
            bounds: robot.link_bounds.clone(),
            level: Level::new(),
        }
    }

    pub(crate) fn dof(&self) -> usize {
        self.dof
    }

    pub(crate) fn scratch(&self) -> LaneScratch {
        let (links, spheres) = (self.links.len(), self.centers.len());
        let zero = [STORED_ZERO; 3];
        LaneScratch {
            q: vec![STORED_ZERO; self.dof],
            rot: vec![[zero; 3]; links],
            pos: vec![zero; links],
            centers: vec![zero; spheres],
            placed: vec![false; links],
            bound: vec![zero; links],
            bounded: vec![false; links],
        }
    }

    /// The widest SIMD instructions the CPU has: callers `dispatch!` on it once, around all their
    /// calls to `clearance` and `keeps`, which must inline into the dispatched code to run fast.
    pub(crate) fn level(&self) -> Level {
        self.level
    }

    /// Loads up to eight configurations (`[k, dof]`) into lanes, repeating the last into unused
    /// lanes, and runs forward kinematics and the sphere centres.
    #[inline(always)]
    fn load<S: Simd>(&self, simd: S, q: &[f32], s: &mut LaneScratch) {
        self.pose(simd, q, s);
        for link in 0..self.links.len() {
            self.place(simd, link, s);
        }
    }

    /// Link `link`'s bounding sphere's centre, once per load.
    #[inline(always)]
    fn bound_center<S: Simd>(&self, simd: S, link: usize, s: &mut LaneScratch) -> V3<S> {
        if !s.bounded[link] {
            s.bounded[link] = true;
            let b = self.bounds[link];
            let c = M3::load(simd, &s.rot[link])
                .times(Vec3::new(b[0], b[1], b[2]))
                .add(V3::load(simd, &s.pos[link]));
            s.bound[link] = c.store();
        }
        V3::load(simd, &s.bound[link])
    }

    /// Link `link`'s sphere centres, once per load.
    #[inline(always)]
    fn place<S: Simd>(&self, simd: S, link: usize, s: &mut LaneScratch) {
        if s.placed[link] {
            return;
        }
        s.placed[link] = true;
        let (rot, pos) = (M3::load(simd, &s.rot[link]), V3::load(simd, &s.pos[link]));
        let (first, end) = self.link_spheres[link];
        for i in first..end {
            s.centers[i] = rot.times(self.centers[i]).add(pos).store();
        }
    }

    /// Loads up to eight configurations (`[k, dof]`) into lanes, repeating the last into unused
    /// lanes, and runs forward kinematics; sphere centres follow per link (`place`).
    #[inline(always)]
    fn pose<S: Simd>(&self, simd: S, q: &[f32], s: &mut LaneScratch) {
        s.placed.fill(false);
        s.bounded.fill(false);
        let n = self.dof;
        let k = q.len() / n;
        debug_assert!(
            (1..=LANES).contains(&k) && q.len() == k * n,
            "lanes load 1 to {LANES} whole configurations, not {} values for {n} joints",
            q.len()
        );
        for j in 0..n {
            s.q[j] = Stored(std::array::from_fn(|l| q[l.min(k - 1) * n + j]));
        }
        for (i, link) in self.links.iter().enumerate() {
            let (rot, pos) = match link.joint {
                Joint::Fixed { rot, trans } => match link.parent {
                    None => (M3::splat(simd, rot), V3::splat(simd, trans)),
                    Some(p) => {
                        let (pr, pp) = (M3::load(simd, &s.rot[p]), V3::load(simd, &s.pos[p]));
                        (pr.times_matrix(rot), pr.times(trans).add(pp))
                    }
                },
                Joint::Revolute {
                    dof,
                    curve,
                    trans,
                    fixed,
                    cos,
                    sin,
                } => {
                    let (sn, cs) = sin_cos(horner(&curve, load(simd, &s.q[dof])));
                    let m = M3([
                        rotation_column(fixed.x_axis, cos.x_axis, sin.x_axis, cs, sn),
                        rotation_column(fixed.y_axis, cos.y_axis, sin.y_axis, cs, sn),
                        rotation_column(fixed.z_axis, cos.z_axis, sin.z_axis, cs, sn),
                    ]);
                    match link.parent {
                        None => (m, V3::splat(simd, trans)),
                        Some(p) => {
                            let (pr, pp) = (M3::load(simd, &s.rot[p]), V3::load(simd, &s.pos[p]));
                            (pr.times_lanes_matrix(&m), pr.times(trans).add(pp))
                        }
                    }
                }
                Joint::Prismatic {
                    dof,
                    curve,
                    rot,
                    trans,
                    axis,
                } => {
                    let local = V3::splat(simd, trans)
                        .add(V3::splat(simd, axis).scale(horner(&curve, load(simd, &s.q[dof]))));
                    match link.parent {
                        None => (M3::splat(simd, rot), local),
                        Some(p) => {
                            let (pr, pp) = (M3::load(simd, &s.rot[p]), V3::load(simd, &s.pos[p]));
                            (pr.times_matrix(rot), pr.times_lanes(local).add(pp))
                        }
                    }
                }
            };
            s.rot[i] = rot.store();
            s.pos[i] = pos.store();
        }
    }

    /// World and self clearance of up to eight configurations (`[k, dof]`, `k = out.len()`), as
    /// `cpu::collision` computes them with no margins.
    #[inline(always)]
    pub(crate) fn clearance<S: Simd>(
        &self,
        simd: S,
        world: &[Prepared],
        q: &[f32],
        out: &mut [[f32; 2]],
        s: &mut LaneScratch,
    ) {
        self.load(simd, q, s);
        let zero = F::splat(simd, 0.0);
        let far = F::splat(simd, FAR);
        let mut wmin = far;
        for &(link, first, end, b) in &self.world_links {
            let bc = M3::load(simd, &s.rot[link])
                .times(Vec3::new(b[0], b[1], b[2]))
                .add(V3::load(simd, &s.pos[link]));
            for o in world {
                if matches!(o, Prepared::Sdf { .. }) {
                    wmin = wmin.min(self.spheres_clearance(simd, o, first, end, s));
                    continue;
                }
                let gap = distance(o, bc) - b[3];
                let near = gap.simd_le(zero);
                if near.any_true() {
                    wmin =
                        wmin.min(near.select(self.spheres_clearance(simd, o, first, end, s), gap));
                } else {
                    wmin = wmin.min(gap);
                }
            }
        }
        let mut smin = far;
        for (a, b, spheres) in &self.pairs {
            let (ba, bb) = (self.bounds[*a], self.bounds[*b]);
            let ca = M3::load(simd, &s.rot[*a])
                .times(Vec3::new(ba[0], ba[1], ba[2]))
                .add(V3::load(simd, &s.pos[*a]));
            let cb = M3::load(simd, &s.rot[*b])
                .times(Vec3::new(bb[0], bb[1], bb[2]))
                .add(V3::load(simd, &s.pos[*b]));
            let gap = ca.sub(cb).length() - (ba[3] + bb[3]);
            let near = gap.simd_le(zero);
            if !near.any_true() {
                smin = smin.min(gap);
                continue;
            }
            let mut pairs = far;
            for &[i, j] in spheres {
                let d = V3::load(simd, &s.centers[i]).sub(V3::load(simd, &s.centers[j]));
                pairs = pairs.min(d.length() - (self.self_radius[i] + self.self_radius[j]));
            }
            smin = smin.min(near.select(pairs, gap));
        }
        let (w, sc): ([f32; LANES], [f32; LANES]) = (wmin.into(), smin.into());
        for (l, c) in out.iter_mut().enumerate() {
            *c = [w[l], sc[l]];
        }
    }

    /// The least clearance from `o` of spheres `first..end`.
    #[inline(always)]
    fn spheres_clearance<S: Simd>(
        &self,
        simd: S,
        o: &Prepared,
        first: usize,
        end: usize,
        s: &LaneScratch,
    ) -> F<S> {
        let mut least = F::splat(simd, FAR);
        for i in first..end {
            least = least.min(distance(o, V3::load(simd, &s.centers[i])) - self.radius[i]);
        }
        least
    }

    /// Whether all of up to eight configurations (`[k, dof]`) keep `margin[lane]` from the world
    /// and are free of self-collision, as `collision_free` judges their clearance with a zero
    /// margin; stops at the first collision. Links are checked from the tip, where collisions are
    /// likeliest, each first through its bounding sphere.
    #[inline(always)]
    pub(crate) fn keeps<S: Simd>(
        &self,
        simd: S,
        world: &[Prepared],
        q: &[f32],
        margin: [f32; LANES],
        s: &mut LaneScratch,
    ) -> bool {
        self.pose(simd, q, s);
        let margin = F::simd_from(simd, margin);
        for &(link, first, end, b) in self.world_links.iter().rev() {
            let bc = self.bound_center(simd, link, s);
            let reach = margin + b[3];
            for o in world {
                // A box's surrounding sphere out of reach first, which most are: half the work of
                // the exact test, which follows.
                if let Prepared::Cuboid { center, radius, .. } = *o {
                    let r = reach + radius;
                    if !bc
                        .sub(V3::splat(simd, center))
                        .length_squared()
                        .simd_lt(r * r)
                        .any_true()
                    {
                        continue;
                    }
                }
                // Within the link's bounding sphere's reach of an obstacle, check its spheres.
                if !matches!(o, Prepared::Sdf { .. }) && !within(o, bc, reach).any_true() {
                    continue;
                }
                self.place(simd, link, s);
                for i in first..end {
                    if within(o, V3::load(simd, &s.centers[i]), margin + self.radius[i]).any_true()
                    {
                        return false;
                    }
                }
            }
        }
        for ((a, b, _), (tight, groups)) in self.pairs.iter().zip(&self.pair_groups) {
            let (ba, bb) = (self.bounds[*a], self.bounds[*b]);
            let (ca, cb) = (
                self.bound_center(simd, *a, s),
                self.bound_center(simd, *b, s),
            );
            let reach = ba[3] + bb[3];
            if !ca
                .sub(cb)
                .length_squared()
                .simd_lt(F::splat(simd, reach * reach))
                .any_true()
            {
                continue;
            }
            self.place(simd, *a, s);
            self.place(simd, *b, s);
            let (bound, radius) = if tight == b { (cb, bb[3]) } else { (ca, ba[3]) };
            for (i, js) in groups {
                let ci = V3::load(simd, &s.centers[*i]);
                // Out of reach of the tighter link's bounding sphere, out of reach of its spheres;
                // a hair wide, so it never rules out what the exact test would not.
                let near = (self.self_radius[*i] + radius) * 1.0001 + 1e-6;
                if !ci
                    .sub(bound)
                    .length_squared()
                    .simd_lt(F::splat(simd, near * near))
                    .any_true()
                {
                    continue;
                }
                for &j in js {
                    let r = self.self_radius[*i] + self.self_radius[j];
                    let d = ci.sub(V3::load(simd, &s.centers[j]));
                    if d.length_squared().simd_lt(F::splat(simd, r * r)).any_true() {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// The lanes where `p` is nearer `o` than `r` (positive): its distance (`distance`) below `r`,
/// compared squared where that is exact.
#[inline(always)]
fn within<S: Simd>(o: &Prepared, p: V3<S>, r: F<S>) -> fearless_simd::mask32x8<S> {
    let simd = p.x.simd;
    let zero = F::splat(simd, 0.0);
    match *o {
        Prepared::Cuboid {
            rot,
            center,
            half,
            upright,
            ..
        } => {
            let l = if upright {
                p.into_upright_frame(&rot, center)
            } else {
                p.into_frame(&rot, center)
            };
            let outside = V3 {
                x: (l.x.abs() - half.x).max(zero),
                y: (l.y.abs() - half.y).max(zero),
                z: (l.z.abs() - half.z).max(zero),
            };
            outside.length_squared().simd_lt(r * r)
        }
        Prepared::Sphere { center, radius } => {
            let reach = r + radius;
            p.sub(V3::splat(simd, center))
                .length_squared()
                .simd_lt(reach * reach)
        }
        Prepared::Capsule {
            rot,
            center,
            radius,
            half_length,
        } => {
            let l = p.into_frame(&rot, center);
            let h = F::splat(simd, half_length);
            let z = l.z - l.z.max(-h).min(h);
            let reach = r + radius;
            (l.x * l.x + l.y * l.y + z * z).simd_lt(reach * reach)
        }
        Prepared::Cylinder { .. } | Prepared::Sdf { .. } => distance(o, p).simd_lt(r),
    }
}

/// `curve`'s value at `x` in every lane (`Curve::value`).
#[inline(always)]
fn horner<S: Simd>(c: &[f32; 5], x: F<S>) -> F<S> {
    let simd = x.simd;
    let (c0, c1, c2, c3, c4) = (
        F::splat(simd, c[0]),
        F::splat(simd, c[1]),
        F::splat(simd, c[2]),
        F::splat(simd, c[3]),
        F::splat(simd, c[4]),
    );
    c0 + x * (c1 + x * (c2 + x * (c3 + x * c4)))
}

/// Column of a revolute joint's rotation (`Joint::Revolute`): `fixed + cos(angle) * cos +
/// sin(angle) * sin`.
#[inline(always)]
fn rotation_column<S: Simd>(fixed: Vec3, cos: Vec3, sin: Vec3, cs: F<S>, sn: F<S>) -> V3<S> {
    let simd = cs.simd;
    V3 {
        x: F::splat(simd, fixed.x) + cs * cos.x + sn * sin.x,
        y: F::splat(simd, fixed.y) + cs * cos.y + sn * sin.y,
        z: F::splat(simd, fixed.z) + cs * cos.z + sn * sin.z,
    }
}

/// Sine and cosine in every lane, for finite angles: Agner Fog's vector class library's
/// polynomials (vectormath_trig.h), as `wide` evaluates them, the quadrant found in floats.
#[inline(always)]
fn sin_cos<S: Simd>(x: F<S>) -> (F<S>, F<S>) {
    const DP1: f32 = 0.78515625 * 2.0;
    const DP2: f32 = 2.418_756_5e-4 * 2.0;
    const DP3: f32 = 3.774_895e-8 * 2.0;
    const SIN: [f32; 3] = [-1.666_665_5e-1, 8.332_161e-3, -1.951_529_6e-4];
    const COS: [f32; 3] = [4.166_664_6e-2, -1.388_731_6e-3, 2.443_315_7e-5];
    let xa = x.abs();
    // The quadrant, and the angle within it.
    let y = (xa * std::f32::consts::FRAC_2_PI).round_ties_even();
    let r = xa - y * DP1;
    let r = r - y * DP2;
    let r = r - y * DP3;
    let r2 = r * r;
    let s = (r2 * SIN[2] + SIN[1]) * r2 + SIN[0];
    let s = s * (r * r2) + r;
    let c = (r2 * COS[2] + COS[1]) * r2 + COS[0];
    let c = c * (r2 * r2) + (r2 * -0.5 + 1.0);
    let quadrant = y - (y * 0.25).floor() * 4.0;
    let odd = quadrant.simd_eq(1.0) | quadrant.simd_eq(3.0);
    let (sin, cos) = (odd.select(c, s), odd.select(s, c));
    let sin_negative = quadrant.simd_ge(2.0) ^ x.simd_lt(0.0);
    let cos_negative = quadrant.simd_eq(1.0) | quadrant.simd_eq(2.0);
    (
        sin_negative.select(-sin, sin),
        cos_negative.select(-cos, cos),
    )
}

/// Signed distance from `o` to `p` in every lane (the distances of `cpu::obstacle_distance`).
#[inline(always)]
fn distance<S: Simd>(o: &Prepared, p: V3<S>) -> F<S> {
    let simd = p.x.simd;
    let zero = F::splat(simd, 0.0);
    match *o {
        Prepared::Cuboid {
            rot,
            center,
            half,
            upright,
            ..
        } => {
            let l = if upright {
                p.into_upright_frame(&rot, center)
            } else {
                p.into_frame(&rot, center)
            };
            let (qx, qy, qz) = (l.x.abs() - half.x, l.y.abs() - half.y, l.z.abs() - half.z);
            let outside = V3 {
                x: qx.max(zero),
                y: qy.max(zero),
                z: qz.max(zero),
            }
            .length();
            outside.simd_gt(zero).select(outside, qx.max(qy).max(qz))
        }
        Prepared::Sphere { center, radius } => p.sub(V3::splat(simd, center)).length() - radius,
        Prepared::Cylinder {
            rot,
            center,
            radius,
            half_height,
        } => {
            let l = p.into_frame(&rot, center);
            let dr = (l.x * l.x + l.y * l.y).sqrt() - radius;
            let dz = l.z.abs() - half_height;
            let corner = (dr * dr + dz * dz).sqrt();
            (dr.simd_gt(zero) & dz.simd_gt(zero)).select(corner, dr.max(dz))
        }
        Prepared::Capsule {
            rot,
            center,
            radius,
            half_length,
        } => {
            let l = p.into_frame(&rot, center);
            let h = F::splat(simd, half_length);
            let z = l.z - l.z.max(-h).min(h);
            (l.x * l.x + l.y * l.y + z * z).sqrt() - radius
        }
        Prepared::Sdf {
            rot,
            center,
            ref grid,
        } => {
            let (x, y, z): ([f32; LANES], [f32; LANES], [f32; LANES]) =
                (p.x.into(), p.y.into(), p.z.into());
            F::simd_from(
                simd,
                std::array::from_fn::<f32, LANES, _>(|l| {
                    world::sdf_distance(rot, center, grid, Vec3::new(x[l], y[l], z[l])).0
                }),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use glam::Quat;

    use super::*;
    use crate::device::{CollisionWeights, Device};
    use crate::rng::Rng;
    use crate::robot::{AttachedObject, CollisionModel, RobotOptions};
    use crate::sdf::SdfGrid;
    use crate::spheres::SphereOptions;
    use crate::world::{Obstacle, World};

    fn panda() -> Robot {
        let fingers = [
            ("panda_finger_joint1".to_string(), 0.04),
            ("panda_finger_joint2".to_string(), 0.04),
        ];
        let options = RobotOptions {
            lock_joints: HashMap::from(fingers),
            collision_model: Some(
                CollisionModel::load("assets/franka/panda_collision.json").unwrap(),
            ),
            ..Default::default()
        };
        Robot::load("assets/franka/franka_panda.urdf", &options).unwrap()
    }

    /// Every obstacle kind around an arm's workspace scaled by `s`, a distance grid of a ball
    /// among them.
    fn scene(s: f32) -> World {
        let tilt = Quat::from_rotation_x(0.4) * Quat::from_rotation_z(0.3);
        let (dims, voxel, origin) = ([20, 20, 20], 0.02 * s, Vec3::new(0.1, -0.5, 0.1) * s);
        let ball = Vec3::new(0.3, -0.3, 0.3) * s;
        let distances: Vec<f32> = (0..8000)
            .map(|i| {
                let p = origin
                    + Vec3::new((i % 20) as f32, (i / 20 % 20) as f32, (i / 400) as f32) * voxel;
                p.distance(ball) - 0.08 * s
            })
            .collect();
        let grid = Arc::new(SdfGrid::new(dims, voxel, origin, &distances).unwrap());
        let v = |x: f32, y: f32, z: f32| Vec3::new(x, y, z) * s;
        World {
            obstacles: vec![
                Obstacle::Cuboid {
                    center: v(0.5, 0.0, 0.3),
                    half_extents: v(0.05, 0.2, 0.1),
                    rotation: tilt,
                },
                // Upright: turned about z only.
                Obstacle::Cuboid {
                    center: v(0.1, -0.5, 0.2),
                    half_extents: v(0.15, 0.05, 0.2),
                    rotation: Quat::from_rotation_z(0.7),
                },
                Obstacle::sphere(v(0.0, 0.5, 0.5), 0.1 * s),
                Obstacle::Cylinder {
                    center: v(-0.4, 0.0, 0.4),
                    rotation: tilt,
                    radius: 0.06 * s,
                    half_height: 0.2 * s,
                },
                Obstacle::Capsule {
                    center: v(0.2, 0.4, 0.1),
                    rotation: tilt,
                    radius: 0.05 * s,
                    half_length: 0.15 * s,
                },
                Obstacle::Sdf {
                    grid,
                    center: Vec3::ZERO,
                    rotation: Quat::IDENTITY,
                },
            ],
        }
    }

    /// The device's clearances (eight at a time, in runs that share a world) against `evaluate`,
    /// the scalar twin of the GPU kernels.
    /// With `arm`, the inputs must collide and miss both ways; otherwise the clearances must
    /// still follow the configuration.
    fn lanes_match_the_scalar_twin(robot: &Robot, arm: bool) {
        let device = Device::cpu(robot).unwrap();
        let worlds = device
            .upload(&[scene(1.0), World::default(), scene(1.0)])
            .unwrap();
        let n = robot.dof();
        let mut rng = Rng::new(7);
        let items = 3000;
        let q: Vec<f32> = (0..items)
            .flat_map(|_| {
                (0..n)
                    .map(|j| rng.range(robot.lower[j].max(-3.2), robot.upper[j].min(3.2)))
                    .collect::<Vec<_>>()
            })
            .collect();
        // Runs of three, so runs and chunks of eight cross.
        let item_world: Vec<u32> = (0..items as u32).map(|i| i / 3 % 3).collect();
        let lanes = device.clearance(&worlds, &item_world, &q).unwrap();
        let scalar = device
            .evaluate(&worlds, &item_world, &q, &CollisionWeights::NONE)
            .unwrap();
        // The baseline SIMD level too (SSE2 on x86-64), one configuration at a time.
        let mut baseline = Lanes::new(robot);
        baseline.level = Level::baseline();
        let prepared = crate::cpu::prepare(&[scene(1.0), World::default(), scene(1.0)]);
        let mut scratch = baseline.scratch();
        let mut out = [[0.0; 2]];
        let lanes_baseline: Vec<[f32; 2]> = (0..items)
            .map(|i| {
                let (world, q) = (&prepared[item_world[i] as usize], &q[i * n..(i + 1) * n]);
                fearless_simd::dispatch!(baseline.level, simd => {
                    baseline.clearance(simd, world, q, &mut out, &mut scratch)
                });
                out[0]
            })
            .collect();
        let (mut hits, mut misses, mut self_hits) = (0, 0, 0);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in 0..items {
            let expected = [scalar.world_clearance[i], scalar.self_clearance[i]];
            for k in 0..2 {
                for (level, lanes) in [("best", &lanes), ("baseline", &lanes_baseline)] {
                    let (a, b) = (lanes[i][k], expected[k]);
                    assert!(
                        (a - b).abs() <= 2e-5,
                        "{} ({level}): item {i} clearance {k}: {a} vs {b}",
                        robot.name()
                    );
                }
            }
            hits += usize::from(expected[0] < 0.0);
            misses += usize::from(item_world[i] != 1 && expected[0] > 0.0);
            self_hits += usize::from(expected[1] < 0.0);
            if item_world[i] != 1 {
                (lo, hi) = (lo.min(expected[0]), hi.max(expected[0]));
            }
        }
        // `keeps`, at the best level and the baseline, agrees with the clearances away from the
        // boundary: the world kept at a margin, no self-collision. A wide margin keeps the twin
        // from letting a link's bounding sphere stand in for its spheres, which `keeps`
        // always checks.
        let margin = 0.01;
        let exact = CollisionWeights {
            margin: 10.0,
            self_margin: 10.0,
            ..CollisionWeights::NONE
        };
        let scalar = device.evaluate(&worlds, &item_world, &q, &exact).unwrap();
        for level in [Level::new(), Level::baseline()] {
            let mut lanes = Lanes::new(robot);
            lanes.level = level;
            let mut scratch = lanes.scratch();
            let (mut kept, mut not) = (0, 0);
            for i in 0..items {
                let (w, sc) = (scalar.world_clearance[i], scalar.self_clearance[i]);
                if (w - margin).abs() < 1e-4 || sc.abs() < 1e-4 {
                    continue;
                }
                let (world, q) = (&prepared[item_world[i] as usize], &q[i * n..(i + 1) * n]);
                let keeps = fearless_simd::dispatch!(level, simd => {
                    lanes.keeps(simd, world, q, [margin; LANES], &mut scratch)
                });
                assert_eq!(
                    keeps,
                    w > margin && sc > 0.0,
                    "{}: item {i} keeps {keeps}: {w}, self {sc}",
                    robot.name()
                );
                if keeps { kept += 1 } else { not += 1 }
            }
            assert!(
                !arm || (kept > 50 && not > 50),
                "{}: {kept} kept, {not} not",
                robot.name()
            );
        }
        let name = robot.name();
        assert!(
            !arm || (hits > 50 && misses > 50 && self_hits > 10),
            "{name}: {hits} hits, {misses} misses, {self_hits} self"
        );
        assert!(
            hi - lo > 1e-3,
            "{name}: world clearances should follow the configuration"
        );
    }

    #[test]
    fn lanes_match_the_scalar_twin_on_every_robot() {
        let panda = panda();
        lanes_match_the_scalar_twin(&panda, true);
        // Its fingers slide on prismatic joints.
        let model = CollisionModel::load("assets/franka/panda_collision.json").unwrap();
        let options = RobotOptions {
            collision_model: Some(model),
            ..Default::default()
        };
        let fingers = Robot::load("assets/franka/franka_panda.urdf", &options).unwrap();
        lanes_match_the_scalar_twin(&fingers, true);
        let held = panda
            .attach(&AttachedObject {
                name: "box".into(),
                link: "panda_hand".into(),
                shapes: vec![Obstacle::cuboid(
                    Vec3::new(0.0, 0.0, 0.15),
                    Vec3::new(0.04, 0.04, 0.06),
                )],
                touch_links: vec!["panda_leftfinger".into(), "panda_rightfinger".into()],
                spheres: SphereOptions::default(),
            })
            .unwrap();
        lanes_match_the_scalar_twin(&held, true);
        // Quartic mimic joints closing a loop, and a prismatic-free arm with joints wider than a
        // turn. Its meshes are not in the repository; a sphere at each link's origin and
        // one off it.
        let path = "assets/menagerie/robotiq_2f85/2f85.xml";
        let bare = Robot::load(
            path,
            &RobotOptions {
                collision_model: Some(CollisionModel::default()),
                ..Default::default()
            },
        );
        let spheres = bare
            .unwrap()
            .links
            .iter()
            .map(|l| {
                (
                    l.name.clone(),
                    vec![[0.0, 0.0, 0.0, 0.01], [0.0, 0.02, 0.03, 0.01]],
                )
            })
            .collect();
        let model = CollisionModel {
            spheres,
            ..Default::default()
        };
        let gripper = Robot::load(
            path,
            &RobotOptions {
                collision_model: Some(model),
                ..Default::default()
            },
        )
        .unwrap();
        lanes_match_the_scalar_twin(&gripper, false);
        let ur5e = Robot::load(
            "assets/ur5e/ur_description/urdf/ur5e.urdf",
            &RobotOptions::default(),
        )
        .unwrap();
        lanes_match_the_scalar_twin(&ur5e, true);
    }
}
