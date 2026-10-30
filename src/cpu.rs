//! CPU reference backend (rayon over items, on a pool it owns). Every function here has a WGSL twin
//! in `kernels.wgsl`; the GPU parity tests compare the two.

// Index loops here mirror kernels.wgsl line for line, which keeps the two easy to compare.
#![allow(clippy::needless_range_loop)]

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fearless_simd::dispatch;
use glam::{Mat3, Vec2, Vec3};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder, Yield};

use crate::device::{Backend, CollisionWeights, Evaluation, Worlds};
use crate::edges::{self, EdgeScratch, Search};
use crate::error::{Error, Result};
use crate::ik::IkOptions;
use crate::lanes::{LANES, LaneScratch, Lanes};
use crate::robot::{Fk, JointKind, Robot};
use crate::rrt::HELPERS;
use crate::sdf::{self, DepthImage, Layout, Occupancy, RobotMask, SdfGrid};
use crate::spline;
use crate::trajopt::{LINE_SEARCH, MAX_HISTORY, PlanOptions};
use crate::types::{JointPaths, Pose};
use crate::world::{self, FAR, LinkSets, Obstacle, World};

/// Up to eight clearances out, their configurations and their worlds.
type Lanes8<'a> = (&'a mut [[f32; 2]], (&'a [f32], &'a [u32]));

/// Batches of clearances checked on the calling thread rather than the pool's.
const SMALL_BATCH: usize = 1024;

pub(crate) struct CpuBackend {
    robot: Robot,
    /// The robot prepared for checking eight configurations at a time.
    lanes: Arc<Lanes>,
    /// All batched work runs here, never on rayon's global pool.
    pool: Arc<ThreadPool>,
}

impl CpuBackend {
    /// `threads` workers; 0 means one per core.
    pub(crate) fn new(robot: &Robot, threads: usize) -> Result<Self> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("batchplan-cpu-{i}"))
            .build()
            .map_err(|e| Error::Threads(format!("starting the CPU device's threads: {e}")))?;
        Ok(Self {
            robot: robot.clone(),
            lanes: Arc::new(Lanes::new(robot)),
            pool: Arc::new(pool),
        })
    }
}

/// Obstacle with its rotation matrix precomputed. An `upright` box is turned about the world's z
/// axis only, so its z axis is the world's and the SIMD lanes rotate points in the plane alone.
pub(crate) enum Prepared {
    Cuboid {
        rot: Mat3,
        center: Vec3,
        half: Vec3,
        upright: bool,
    },
    Sphere {
        center: Vec3,
        radius: f32,
    },
    Cylinder {
        rot: Mat3,
        center: Vec3,
        radius: f32,
        half_height: f32,
    },
    Capsule {
        rot: Mat3,
        center: Vec3,
        radius: f32,
        half_length: f32,
    },
    Sdf {
        rot: Mat3,
        center: Vec3,
        grid: Arc<SdfGrid>,
    },
}

impl Prepared {
    /// A sphere around the obstacle, a hair wide, so a test against it never rules out what the
    /// exact test would not; infinite for a distance grid.
    fn bounding_sphere(&self) -> BoundingSphere {
        let (center, radius) = match *self {
            Prepared::Cuboid { center, half, .. } => (center, half.length()),
            Prepared::Sphere { center, radius } => (center, radius),
            Prepared::Cylinder {
                center,
                radius,
                half_height,
                ..
            } => (center, Vec2::new(radius, half_height).length()),
            Prepared::Capsule {
                center,
                radius,
                half_length,
                ..
            } => (center, radius + half_length),
            Prepared::Sdf { center, .. } => (center, f32::INFINITY),
        };
        BoundingSphere {
            center,
            radius: radius * 1.0001 + 1e-6,
        }
    }
}

/// A sphere around an obstacle (`Prepared::bounding_sphere`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct BoundingSphere {
    pub(crate) center: Vec3,
    pub(crate) radius: f32,
}

/// A world's obstacles, prepared, and the sphere around each, which checks (`Lanes::keeps`) test
/// first: most obstacles are out of reach of a link, and the spheres, kept apart from the
/// obstacles, scan faster than the obstacles' whole records. With them, the links that may touch
/// each obstacle, which every check skips.
pub(crate) struct CpuWorld {
    pub(crate) obstacles: Vec<Prepared>,
    pub(crate) bounds: Vec<BoundingSphere>,
    pub(crate) touch: LinkSets,
}

/// The obstacles of each world, prepared.
pub(crate) type CpuWorlds = Vec<CpuWorld>;

/// `worlds` for checks of `robot`, whose links their touches name.
pub(crate) fn prepare(robot: &Robot, worlds: &[World]) -> Result<CpuWorlds> {
    worlds
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let obstacles: Vec<Prepared> = s
                .obstacles
                .iter()
                .map(|o| match *o {
                    Obstacle::Cuboid {
                        center,
                        half_extents,
                        rotation,
                    } => {
                        let rot = Mat3::from_quat(rotation);
                        let upright =
                            rot.z_axis == Vec3::Z && rot.x_axis.z == 0.0 && rot.y_axis.z == 0.0;
                        Prepared::Cuboid {
                            rot,
                            center,
                            half: half_extents,
                            upright,
                        }
                    }
                    Obstacle::Sphere { center, radius } => Prepared::Sphere { center, radius },
                    Obstacle::Cylinder {
                        center,
                        rotation,
                        radius,
                        half_height,
                    } => Prepared::Cylinder {
                        rot: Mat3::from_quat(rotation),
                        center,
                        radius,
                        half_height,
                    },
                    Obstacle::Capsule {
                        center,
                        rotation,
                        radius,
                        half_length,
                    } => Prepared::Capsule {
                        rot: Mat3::from_quat(rotation),
                        center,
                        radius,
                        half_length,
                    },
                    Obstacle::Sdf {
                        ref grid,
                        center,
                        rotation,
                    } => Prepared::Sdf {
                        rot: Mat3::from_quat(rotation),
                        center,
                        grid: grid.clone(),
                    },
                })
                .collect();
            let bounds = obstacles.iter().map(Prepared::bounding_sphere).collect();
            let touch = s
                .touch_sets(robot)
                .map_err(|e| Error::Input(format!("world {i}: {e}")))?;
            Ok(CpuWorld {
                obstacles,
                bounds,
                touch,
            })
        })
        .collect()
}

struct CollisionOut {
    cost: f32,
    world_clearance: f32,
    self_clearance: f32,
}

/// A configuration's sphere centres and per-link collision wrenches (force, and moment about the
/// world origin).
struct CollisionBuffers {
    centers: Vec<Vec3>,
    force: Vec<Vec3>,
    moment: Vec<Vec3>,
}

/// Working memory for one configuration at a time, sized to the robot. The hot loops keep one per
/// thread and reuse it, so they neither allocate nor clear more than the robot needs.
struct Scratch {
    fk: Fk,
    q: Vec<f32>,
    collision: CollisionBuffers,
    /// IK: a hand's Jacobian (6 x dof, row-major), the step, the collision gradient, its part in
    /// the targeted hands' row spaces, and which joints the targeted hands move.
    jac: Vec<f32>,
    dq: Vec<f32>,
    grad: Vec<f32>,
    push: Vec<f32>,
    moving: Vec<bool>,
}

impl Scratch {
    fn new(robot: &Robot) -> Self {
        let (n, links) = (robot.dof(), robot.links.len());
        Self {
            fk: Fk::new(links),
            q: vec![0.0; n],
            collision: CollisionBuffers {
                centers: vec![Vec3::ZERO; robot.spheres.len()],
                force: vec![Vec3::ZERO; links],
                moment: vec![Vec3::ZERO; links],
            },
            jac: vec![0.0; 6 * n],
            dq: vec![0.0; n],
            grad: vec![0.0; n],
            push: vec![0.0; n],
            moving: vec![false; n],
        }
    }
}

/// Signed distance from `o` to `p` and its gradient. Mirrors `obstacle_distance` in kernels.wgsl.
#[inline]
fn obstacle_distance(o: &Prepared, p: Vec3) -> (f32, Vec3) {
    match *o {
        Prepared::Cuboid {
            rot, center, half, ..
        } => world::box_distance(rot, center, half, p),
        Prepared::Sphere { center, radius } => world::sphere_distance(center, radius, p),
        Prepared::Cylinder {
            rot,
            center,
            radius,
            half_height,
        } => world::cylinder_distance(rot, center, radius, half_height, p),
        Prepared::Capsule {
            rot,
            center,
            radius,
            half_length,
        } => world::capsule_distance(rot, center, radius, half_length, p),
        Prepared::Sdf {
            rot,
            center,
            ref grid,
        } => world::sdf_distance(rot, center, grid, p),
    }
}

/// Collision cost of the configuration behind `fk`; adds d(cost)/dq into `grad`.
/// Without `GRADIENT` only the cost and clearances are computed, as the WGSL `gradient` flag does.
// Inlined into every caller: once the cost-only form had two callers, LLVM stopped inlining it into
// the line search, which cost 5% of CPU planning.
#[inline(always)]
fn collision<const GRADIENT: bool>(
    robot: &Robot,
    world: &CpuWorld,
    fk: &Fk,
    w: &CollisionWeights,
    grad: &mut [f32],
    buffers: &mut CollisionBuffers,
) -> CollisionOut {
    let (rot, pos) = (&fk.rot[..], &fk.pos[..]);
    let CollisionBuffers {
        centers: sc,
        force,
        moment,
    } = buffers;
    for (c, sp) in sc.iter_mut().zip(&robot.spheres) {
        *c = rot[sp.link] * sp.center + pos[sp.link];
    }
    let sc = &sc[..];
    if GRADIENT {
        force.fill(Vec3::ZERO);
        moment.fill(Vec3::ZERO);
    }
    let mut touched = false;
    let (mut cost, mut wmin, mut smin) = (0.0f32, FAR, FAR);
    // An obstacle farther from a link's bounding sphere than the margin cannot add cost through the
    // link's spheres; the gap bounds their clearance from below. Distance grids are interpolated,
    // not exact, so their spheres are always checked. Links no joint moves are not checked against
    // the world: they touch it the same way in every configuration, as a mounted robot's base
    // rests on its table. Links may touch the obstacles the world lets them.
    let world_gate = w.margin.max(0.0);
    for (link, &[first, count]) in robot.sphere_ranges.iter().enumerate() {
        if count == 0 || robot.links[link].chain.is_empty() {
            continue;
        }
        let b = robot.link_bounds[link];
        let bc = rot[link] * Vec3::new(b[0], b[1], b[2]) + pos[link];
        for (k, o) in world.obstacles.iter().enumerate() {
            if world.touch.contains(k, link) {
                continue;
            }
            if !matches!(o, Prepared::Sdf { .. }) {
                let gap = obstacle_distance(o, bc).0 - b[3];
                if gap > world_gate {
                    wmin = wmin.min(gap);
                    continue;
                }
            }
            for s in first as usize..(first + count) as usize {
                let (dist, g) = obstacle_distance(o, sc[s]);
                let d = dist - robot.spheres[s].radius;
                wmin = wmin.min(d);
                let pen = w.margin - d;
                if pen > 0.0 {
                    cost += w.world * pen * pen;
                    if GRADIENT {
                        let f = -2.0 * w.world * pen * g;
                        force[link] += f;
                        moment[link] += sc[s].cross(f);
                        touched = true;
                    }
                }
            }
        }
    }
    // Likewise for link pairs whose bounding spheres are farther apart than the self margin.
    let gate = w.self_margin.max(0.0);
    for lp in &robot.self_link_pairs {
        let (ba, bb) = (
            robot.link_bounds[lp.a as usize],
            robot.link_bounds[lp.b as usize],
        );
        let at = |link: u32, b: [f32; 4]| {
            rot[link as usize] * Vec3::new(b[0], b[1], b[2]) + pos[link as usize]
        };
        let gap = (at(lp.a, ba) - at(lp.b, bb)).length() - ba[3] - bb[3];
        if gap > gate {
            smin = smin.min(gap);
            continue;
        }
        for &[a, b] in &robot.self_pairs[lp.first as usize..(lp.first + lp.count) as usize] {
            let (a, b) = (a as usize, b as usize);
            let (sa, sb) = (&robot.spheres[a], &robot.spheres[b]);
            let diff = sc[a] - sc[b];
            let dist = diff.length();
            let d = dist - (sa.radius + sa.self_buffer) - (sb.radius + sb.self_buffer);
            smin = smin.min(d);
            let pen = w.self_margin - d;
            if pen > 0.0 && GRADIENT {
                let u = if dist > 1e-9 { diff / dist } else { Vec3::X };
                let g = 2.0 * w.self_collision * pen * u;
                let (la, lb) = (lp.a as usize, lp.b as usize);
                force[la] -= g;
                moment[la] -= sc[a].cross(g);
                force[lb] += g;
                moment[lb] += sc[b].cross(g);
                touched = true;
            }
            if pen > 0.0 {
                cost += w.self_collision * pen * pen;
            }
        }
    }
    if !GRADIENT || !touched {
        return CollisionOut {
            cost,
            world_clearance: wmin,
            self_clearance: smin,
        };
    }
    // Each link's wrench reaches the joints at and above it, leaves first.
    for i in (0..robot.links.len()).rev() {
        let link = &robot.links[i];
        match link.joint {
            JointKind::Revolute { dof, curve, .. } => {
                grad[dof] +=
                    fk.rate(i, &curve) * fk.axis[i].dot(moment[i] - pos[i].cross(force[i]));
            }
            JointKind::Prismatic { dof, curve, .. } => {
                grad[dof] += fk.rate(i, &curve) * fk.axis[i].dot(force[i])
            }
            JointKind::Fixed => {}
        }
        if let Some(p) = link.parent {
            let (f, m) = (force[i], moment[i]);
            force[p] += f;
            moment[p] += m;
        }
    }
    CollisionOut {
        cost,
        world_clearance: wmin,
        self_clearance: smin,
    }
}

/// Rotation vector (axis * angle) of `r`. Must stay in sync with `rot_log` in kernels.wgsl.
fn rot_log(r: Mat3) -> (Vec3, f32) {
    let m = |row: usize, col: usize| r.col(col)[row];
    let cos_t = ((m(0, 0) + m(1, 1) + m(2, 2) - 1.0) * 0.5).clamp(-1.0, 1.0);
    let v = 0.5 * Vec3::new(m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1));
    let sv = v.length();
    let theta = sv.atan2(cos_t);
    if sv > 1e-6 {
        return (v * (theta / sv), theta);
    }
    if cos_t > 0.0 {
        return (v, theta);
    }
    // Near pi: the axis is the dominant column of (R + I).
    let k = if m(1, 1) > m(0, 0) {
        if m(2, 2) > m(1, 1) { 2 } else { 1 }
    } else if m(2, 2) > m(0, 0) {
        2
    } else {
        0
    };
    ((r.col(k) + Vec3::AXES[k]).normalize() * theta, theta)
}

/// In-place Cholesky factorization of a 6x6 SPD matrix (row-major, lower triangle used).
fn chol6(a: &mut [f32; 36]) {
    for i in 0..6 {
        for j in 0..=i {
            let mut s = a[i * 6 + j];
            for k in 0..j {
                s -= a[i * 6 + k] * a[j * 6 + k];
            }
            a[i * 6 + j] = if i == j {
                s.max(1e-12).sqrt()
            } else {
                s / a[j * 6 + j]
            };
        }
    }
}

fn chol6_solve(l: &[f32; 36], b: [f32; 6]) -> [f32; 6] {
    let mut y = [0.0; 6];
    for i in 0..6 {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * 6 + k] * y[k];
        }
        y[i] = s / l[i * 6 + i];
    }
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let mut s = y[i];
        for k in i + 1..6 {
            s -= l[k * 6 + i] * x[k];
        }
        x[i] = s / l[i * 6 + i];
    }
    x
}

/// Pose error of link `ee`: (6-vector [position; rotation_weight * rotation], |position|, angle).
#[inline]
fn pose_error(
    fk: &Fk,
    ee: usize,
    target: &(Vec3, Mat3),
    rotation_weight: f32,
) -> ([f32; 6], f32, f32) {
    let ep = target.0 - fk.pos[ee];
    let (eo, angle) = rot_log(target.1 * fk.rot[ee].transpose());
    let eo = eo * rotation_weight;
    ([ep.x, ep.y, ep.z, eo.x, eo.y, eo.z], ep.length(), angle)
}

/// Hand `h`'s Jacobian into `jac` (6 x dof, row-major) and its damped normal matrix, factored, into
/// `a`. Mirrors `hand_system` in kernels.wgsl.
#[inline]
fn hand_system(
    robot: &Robot,
    fk: &Fk,
    h: usize,
    o: &IkOptions,
    jac: &mut [f32],
    a: &mut [f32; 36],
) {
    let (n, ee) = (robot.dof(), robot.ee_links[h]);
    jac.fill(0.0);
    for (i, dof) in robot.chain(ee) {
        let m = fk.rate(
            i,
            &robot.links[i]
                .joint
                .actuation()
                .expect("chain links have moving joints")
                .1,
        );
        let jp = m * fk.dpoint(i, fk.pos[ee]);
        let jo = if fk.slides[i] {
            Vec3::ZERO
        } else {
            m * fk.axis[i] * o.rotation_weight
        };
        for (r, v) in [jp.x, jp.y, jp.z, jo.x, jo.y, jo.z].into_iter().enumerate() {
            jac[r * n + dof] += v;
        }
    }
    for r in 0..6 {
        for c in 0..=r {
            let mut s = 0.0;
            for j in 0..n {
                s += jac[r * n + j] * jac[c * n + j];
            }
            if r == c {
                s += o.damping * o.damping;
            }
            a[r * 6 + c] = s;
            a[c * 6 + r] = s;
        }
    }
    chol6(a);
}

/// `out += jac^T x`.
#[inline]
fn add_jt(jac: &[f32], x: [f32; 6], out: &mut [f32]) {
    let n = out.len();
    for j in 0..n {
        for r in 0..6 {
            out[j] += jac[r * n + j] * x[r];
        }
    }
}

/// One damped-least-squares IK step toward each targeted hand's pose (`targets`, one per hand),
/// then a collision push in the null space of every targeted hand's Jacobian: with `J` a hand's
/// Jacobian and `A = J J^T + damping^2 I`, each hand steps `J^T A^-1 e` toward its target, and
/// the joints `moving` marks take `-collision_step (g - sum over hands of J^T A^-1 J g)`. Hands on
/// separate arms make that the step of the stacked system; joints no targeted hand moves stay
/// where they are. Mirrors `ik_step` in kernels.wgsl, pass for pass.
fn ik_step(
    robot: &Robot,
    world: &CpuWorld,
    targets: &[Option<(Vec3, Mat3)>],
    q: &mut [f32],
    o: &IkOptions,
    s: &mut Scratch,
) {
    let n = q.len();
    robot.fk_into(q, &mut s.fk);
    let Scratch {
        fk,
        collision: buffers,
        jac,
        dq,
        grad,
        push,
        moving,
        ..
    } = s;
    let (jac, dq, grad, push, moving) = (
        &mut jac[..],
        &mut dq[..n],
        &mut grad[..n],
        &mut push[..n],
        &moving[..n],
    );
    let mut a = [0.0f32; 36];
    dq.fill(0.0);
    let mut last = None;
    for (h, target) in targets.iter().enumerate() {
        let Some(target) = target else { continue };
        let (e, _, _) = pose_error(fk, robot.ee_links[h], target, o.rotation_weight);
        hand_system(robot, fk, h, o, jac, &mut a);
        add_jt(jac, chol6_solve(&a, e), dq);
        last = Some(h);
    }
    if o.collision_step > 0.0 {
        grad.fill(0.0);
        collision::<true>(robot, world, fk, &o.collision, grad, buffers);
        push.fill(0.0);
        let targeted = targets.iter().enumerate().filter(|(_, t)| t.is_some());
        for (h, _) in targeted {
            if Some(h) != last {
                hand_system(robot, fk, h, o, jac, &mut a);
            }
            let mut jg = [0.0f32; 6];
            for r in 0..6 {
                for j in 0..n {
                    jg[r] += jac[r * n + j] * grad[j];
                }
            }
            add_jt(jac, chol6_solve(&a, jg), push);
        }
        for j in 0..n {
            if moving[j] {
                dq[j] -= o.collision_step * (grad[j] - push[j]);
            }
        }
    }
    let largest = dq.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if largest > o.max_step {
        o.max_step / largest
    } else {
        1.0
    };
    for j in 0..n {
        let (lo, hi) = robot.bounds(j);
        q[j] = (q[j] + dq[j] * scale).clamp(lo, hi);
    }
}

/// Collision gradient of the trajectory cost at sample `s` of span `span`, with respect to the
/// configuration there. Must match `traj_samples` in kernels.wgsl.
#[allow(clippy::too_many_arguments)]
fn traj_sample_grad(
    robot: &Robot,
    world: &CpuWorld,
    cp: &[f32],
    span: usize,
    s: usize,
    o: &PlanOptions,
    out: &mut [f32],
    scratch: &mut Scratch,
) {
    let n = robot.dof();
    spline::blend(
        cp,
        n,
        span,
        spline::basis(spline::sample_u(s, o.samples_per_span)),
        &mut scratch.q,
    );
    robot.fk_into(&scratch.q, &mut scratch.fk);
    out.fill(0.0);
    collision::<true>(
        robot,
        world,
        &scratch.fk,
        &o.collision,
        out,
        &mut scratch.collision,
    );
}

/// Gradient of the trajectory cost w.r.t. free control point `t`: the collision gradients of the
/// samples it shapes, each weighted by its basis value there, plus smoothness on the control
/// points. Must match `traj_grad` in kernels.wgsl.
fn traj_grad(
    cp: &[f32],
    sample_grad: &[f32],
    n: usize,
    t: usize,
    o: &PlanOptions,
    out: &mut [f32],
) {
    let t_count = cp.len() / n;
    let (spans, k) = (t_count - 3, o.samples_per_span);
    let q = |i: usize, j: usize| cp[i * n + j];
    out.fill(0.0);
    // Control point t is point i of span t - i.
    for i in 0..4 {
        if t < i || t - i >= spans {
            continue;
        }
        for s in 0..k {
            let w = spline::basis(spline::sample_u(s, k))[i];
            let g = &sample_grad[((t - i) * k + s) * n..((t - i) * k + s + 1) * n];
            for j in 0..n {
                out[j] += w * g[j];
            }
        }
    }
    for j in 0..n {
        let (qm, q0, qp) = (q(t - 1, j), q(t, j), q(t + 1, j));
        let mut g = 2.0 * o.velocity_weight * (2.0 * q0 - qm - qp);
        g += 2.0 * o.acceleration_weight * -2.0 * (qp - 2.0 * q0 + qm);
        if t >= 2 {
            g += 2.0 * o.acceleration_weight * (q0 - 2.0 * qm + q(t - 2, j));
        }
        if t + 2 < t_count {
            g += 2.0 * o.acceleration_weight * (q(t + 2, j) - 2.0 * qp + q0);
        }
        out[j] += g;
    }
}

/// Control point `t`, joint `j` of the candidate path `x + alpha d`: free points move and are
/// clamped to the joint range, the three pinned at each end stay. Must match `candidate` in
/// kernels.wgsl.
fn candidate(robot: &Robot, x: &[f32], d: &[f32], alpha: f32, t: usize, j: usize) -> f32 {
    let n = robot.dof();
    let i = t * n + j;
    if t < 3 || t >= x.len() / n - 3 {
        return x[i];
    }
    let (lo, hi) = robot.bounds(j);
    (x[i] + alpha * d[i]).clamp(lo, hi)
}

/// Collision cost at sample `s` of span `span` of the candidate path `x + alpha d`. Must match
/// `traj_costs` in kernels.wgsl.
#[allow(clippy::too_many_arguments)]
fn traj_cost(
    robot: &Robot,
    world: &CpuWorld,
    x: &[f32],
    d: &[f32],
    alpha: f32,
    span: usize,
    s: usize,
    o: &PlanOptions,
    scratch: &mut Scratch,
) -> f32 {
    let n = robot.dof();
    let w = spline::basis(spline::sample_u(s, o.samples_per_span));
    for j in 0..n {
        let c = |i: usize| candidate(robot, x, d, alpha, span + i, j);
        scratch.q[j] = w[0] * c(0) + w[1] * c(1) + w[2] * c(2) + w[3] * c(3);
    }
    robot.fk_into(&scratch.q, &mut scratch.fk);
    collision::<false>(
        robot,
        world,
        &scratch.fk,
        &o.collision,
        &mut [],
        &mut scratch.collision,
    )
    .cost
}

/// One path's L-BFGS state, as kernels.wgsl keeps it in `aux`.
struct Lbfgs {
    /// Collision cost per line-search step and sample.
    costs: Vec<f32>,
    grad: Vec<f32>,
    prev_grad: Vec<f32>,
    dir: Vec<f32>,
    /// The last accepted step, waiting for the gradient change it caused.
    pending_step: Vec<f32>,
    /// Ring buffers of `history` steps and gradient changes; `newest` is the latest slot.
    steps: Vec<f32>,
    changes: Vec<f32>,
    cost: f32,
    count: usize,
    newest: usize,
    started: bool,
    pending: bool,
}

impl Lbfgs {
    fn new(points: usize, n: usize, samples: usize, o: &PlanOptions) -> Self {
        let tn = points * n;
        Self {
            costs: vec![0.0; LINE_SEARCH.len() * samples],
            grad: vec![0.0; tn],
            prev_grad: vec![0.0; tn],
            dir: vec![0.0; tn],
            pending_step: vec![0.0; tn],
            steps: vec![0.0; o.history * tn],
            changes: vec![0.0; o.history * tn],
            cost: 0.0,
            count: 0,
            newest: 0,
            started: false,
            pending: false,
        }
    }
}

/// Line search: the trajectory cost of each step along the direction; the cheapest moves the path
/// if it lowers the cost, otherwise the history resets. Must match `traj_search` in kernels.wgsl.
fn traj_search(robot: &Robot, x: &mut [f32], st: &mut Lbfgs, o: &PlanOptions) {
    let n = robot.dof();
    let t_count = x.len() / n;
    let samples = st.costs.len() / LINE_SEARCH.len();
    let mut best = None;
    for (c, &alpha) in LINE_SEARCH.iter().enumerate() {
        let mut total = 0.0;
        for s in 0..samples {
            total += st.costs[c * samples + s];
        }
        for j in 0..n {
            // Each candidate point once, sliding along the path: (before, previous, current).
            let (mut before, mut previous) = (0.0, candidate(robot, x, &st.dir, alpha, 0, j));
            for t in 1..t_count {
                let current = candidate(robot, x, &st.dir, alpha, t, j);
                let v = current - previous;
                total += o.velocity_weight * v * v;
                if t >= 2 {
                    let a = current - 2.0 * previous + before;
                    total += o.acceleration_weight * a * a;
                }
                (before, previous) = (previous, current);
            }
        }
        if (best.is_none() && !st.started) || total < st.cost {
            st.cost = total;
            best = Some(alpha);
        }
    }
    let Some(alpha) = best else {
        st.pending = false;
        st.count = 0;
        return;
    };
    for t in 3..t_count - 3 {
        for j in 0..n {
            let i = t * n + j;
            let moved = candidate(robot, x, &st.dir, alpha, t, j);
            st.pending_step[i] = moved - x[i];
            x[i] = moved;
        }
    }
    st.started = true;
    st.pending = true;
}

/// Records the last step and the gradient change it caused, then the L-BFGS two-loop recursion
/// for the next direction; steepest descent scaled to `initial_step` without history or when
/// the recursion fails to descend. Must match `lbfgs_direction` in kernels.wgsl.
fn lbfgs_direction(st: &mut Lbfgs, n: usize, o: &PlanOptions) {
    let tn = st.grad.len();
    let free = 3 * n..tn - 3 * n;
    let m = o.history;
    if st.pending {
        let mut sy = 0.0;
        for i in free.clone() {
            sy += st.pending_step[i] * (st.grad[i] - st.prev_grad[i]);
        }
        if sy > 1e-10 {
            let slot = if st.count == 0 {
                0
            } else {
                (st.newest + 1) % m
            };
            for i in free.clone() {
                st.steps[slot * tn + i] = st.pending_step[i];
                st.changes[slot * tn + i] = st.grad[i] - st.prev_grad[i];
            }
            st.newest = slot;
            st.count = (st.count + 1).min(m);
        }
        st.pending = false;
    }
    let mut largest = 0.0f32;
    for i in free.clone() {
        st.prev_grad[i] = st.grad[i];
        st.dir[i] = -st.grad[i];
        largest = largest.max(st.grad[i].abs());
    }
    let dot = |a: &[f32], b: &[f32]| free.clone().map(|i| a[i] * b[i]).sum::<f32>();
    let mut alpha = [0.0f32; MAX_HISTORY];
    for a in 0..st.count {
        let slot = (st.newest + m - a) % m;
        let (s, y) = (
            &st.steps[slot * tn..(slot + 1) * tn],
            &st.changes[slot * tn..(slot + 1) * tn],
        );
        alpha[a] = dot(s, &st.dir) / dot(y, s);
        for i in free.clone() {
            st.dir[i] -= alpha[a] * y[i];
        }
    }
    let gamma = if st.count > 0 {
        let slot = st.newest;
        let (s, y) = (
            &st.steps[slot * tn..(slot + 1) * tn],
            &st.changes[slot * tn..(slot + 1) * tn],
        );
        dot(s, y) / dot(y, y)
    } else if largest > 0.0 {
        o.initial_step / largest
    } else {
        0.0
    };
    for i in free.clone() {
        st.dir[i] *= gamma;
    }
    for a in (0..st.count).rev() {
        let slot = (st.newest + m - a) % m;
        let (s, y) = (
            &st.steps[slot * tn..(slot + 1) * tn],
            &st.changes[slot * tn..(slot + 1) * tn],
        );
        let beta = dot(y, &st.dir) / dot(y, s);
        for i in free.clone() {
            st.dir[i] += (alpha[a] - beta) * s[i];
        }
    }
    if largest > 0.0 && dot(&st.dir, &st.grad) >= 0.0 {
        st.count = 0;
        for i in free.clone() {
            st.dir[i] = -st.grad[i] * (o.initial_step / largest);
        }
    }
}

/// `f(scratch, index, chunk)` over `size`-long chunks of `data`: on the pool's threads, each with
/// its own scratch, when `parallel`; otherwise in order with `scratch`.
fn each_chunk<T: Send>(
    parallel: bool,
    robot: &Robot,
    scratch: &mut Scratch,
    data: &mut [T],
    size: usize,
    f: impl Fn(&mut Scratch, usize, &mut [T]) + Sync,
) {
    if parallel {
        data.par_chunks_mut(size)
            .enumerate()
            .for_each_init(|| Scratch::new(robot), |s, (i, chunk)| f(s, i, chunk));
    } else {
        for (i, chunk) in data.chunks_mut(size).enumerate() {
            f(scratch, i, chunk);
        }
    }
}

/// Optimizes one path's control points in place in `world`, holding the three at each end fixed,
/// and stops between rounds once `deadline` passes. With `parallel`, its samples are spread over
/// the current pool's threads.
pub(crate) fn optimize_path(
    robot: &Robot,
    world: &CpuWorld,
    tr: &mut [f32],
    o: &PlanOptions,
    deadline: Option<Instant>,
    parallel: bool,
) {
    let mut optimizer = PathOptimizer::new(robot, tr.len() / robot.dof(), o);
    optimizer.run(robot, world, tr, o.iterations, deadline, parallel);
}

/// One path's trajectory optimization (`optimize_path`), run a few rounds at a time: each `run`
/// goes on where the last stopped, so the rounds take the same steps however they are split.
pub(crate) struct PathOptimizer {
    o: PlanOptions,
    st: Lbfgs,
    sample_grad: Vec<f32>,
    scratch: Scratch,
    /// The next round; the first only prices the seed, whose direction is still zero.
    round: u32,
    /// Whether the last round moved the path and its next direction is still to find.
    direction_due: bool,
    over: bool,
}

impl PathOptimizer {
    pub(crate) fn new(robot: &Robot, points: usize, o: &PlanOptions) -> Self {
        let n = robot.dof();
        let samples = (points - 3) * o.samples_per_span;
        Self {
            o: *o,
            st: Lbfgs::new(points, n, samples, o),
            sample_grad: vec![0.0; samples * n],
            scratch: Scratch::new(robot),
            round: 0,
            direction_due: false,
            over: false,
        }
    }

    /// Whether every round has run, or the optimization stopped early.
    pub(crate) fn finished(&self) -> bool {
        self.over
    }

    /// Runs the rounds up to and including `through` (at most the options' iterations) on `tr`,
    /// the path the earlier rounds left, or until `deadline` passes. With `parallel`, samples are
    /// spread over the current pool's threads.
    pub(crate) fn run(
        &mut self,
        robot: &Robot,
        world: &CpuWorld,
        tr: &mut [f32],
        through: u32,
        deadline: Option<Instant>,
        parallel: bool,
    ) {
        let Self {
            o,
            st,
            sample_grad,
            scratch,
            ..
        } = self;
        let (n, k) = (robot.dof(), o.samples_per_span);
        let (t_count, samples) = (tr.len() / n, (tr.len() / n - 3) * k);
        while !self.over && self.round <= through.min(o.iterations) {
            if self.direction_due {
                let tr_ref = &*tr;
                each_chunk(parallel, robot, scratch, sample_grad, n, |s, i, g| {
                    traj_sample_grad(robot, world, tr_ref, i / k, i % k, o, g, s);
                });
                for t in 3..t_count - 3 {
                    traj_grad(tr, sample_grad, n, t, o, &mut st.grad[t * n..(t + 1) * n]);
                }
                lbfgs_direction(st, n, o);
                self.direction_due = false;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                self.over = true;
                break;
            }
            let tr_ref = &*tr;
            let dir = &st.dir;
            each_chunk(parallel, robot, scratch, &mut st.costs, 1, |s, i, cost| {
                let (c, i) = (i / samples, i % samples);
                cost[0] = traj_cost(
                    robot,
                    world,
                    tr_ref,
                    dir,
                    LINE_SEARCH[c],
                    i / k,
                    i % k,
                    o,
                    s,
                );
            });
            let steepest = st.count == 0;
            traj_search(robot, tr, st, o);
            // A failed steepest-descent search leaves nothing to change in later rounds.
            if self.round == o.iterations || (steepest && st.started && !st.pending) {
                self.over = true;
            } else {
                self.direction_due = true;
            }
            self.round += 1;
        }
    }
}

/// How long a thread of a batch with no search to help sleeps before it looks again.
const WAIT: Duration = Duration::from_micros(100);

/// Runs every search to its end, or to `deadline`, on `pool`'s threads: each takes the next search
/// until none is left, as VAMP's batches run, so no thread waits on another's share and each keeps
/// one scratch. Then threads help the searches still running long, checking their edges ahead of
/// them (see `rrt::Ahead`). A search alone runs on the calling thread, outside the pool, so its
/// optimizations do too; once it has run long, it summons helpers into the pool.
pub(crate) fn run_searches(
    pool: &ThreadPool,
    searches: &mut [&mut dyn Search],
    robot: &Robot,
    lanes: &Lanes,
    worlds: &[CpuWorld],
    deadline: Option<Instant>,
) {
    if let [search] = searches {
        let scratch = &mut EdgeScratch::new(lanes);
        let Some(ahead) = search.help() else {
            edges::run(
                &mut **search,
                robot,
                lanes,
                worlds,
                deadline,
                None,
                None,
                scratch,
            );
            return;
        };
        // The search stops once it should summon helpers, and goes on beside them.
        if edges::run(
            &mut **search,
            robot,
            lanes,
            worlds,
            deadline,
            None,
            Some(&|| ahead.summon()),
            scratch,
        ) {
            pool.in_place_scope(|scope| {
                for _ in 0..HELPERS {
                    let ahead = ahead.clone();
                    scope.spawn(move |_| {
                        if let Some(mut helper) = ahead.helper(robot, None) {
                            let scratch = &mut EdgeScratch::new(lanes);
                            edges::run(
                                &mut helper,
                                robot,
                                lanes,
                                worlds,
                                deadline,
                                None,
                                None,
                                scratch,
                            );
                        }
                    });
                }
                edges::run(
                    &mut **search,
                    robot,
                    lanes,
                    worlds,
                    deadline,
                    None,
                    None,
                    scratch,
                );
                ahead.finish();
            });
        }
        return ahead.finish();
    }
    // Optimizations under way, spread over the pool's threads.
    let (next, optimizing) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let helped: Vec<_> = searches.iter().map(|s| s.help()).collect();
    let searches: Vec<Mutex<&mut &mut dyn Search>> = searches.iter_mut().map(Mutex::new).collect();
    pool.broadcast(|context| {
        let mut scratch = EdgeScratch::new(lanes);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(search) = searches.get(i) else { break };
            let mut search = search.lock().expect("a search panicked");
            edges::run(
                &mut ***search,
                robot,
                lanes,
                worlds,
                deadline,
                Some(&optimizing),
                None,
                &mut scratch,
            );
            if let Some(ahead) = &helped[i] {
                ahead.finish();
            }
        }
        // Threads spread over the searches running long, each helping one until it is done. While
        // others are still sampling, they may yet run long: threads wait, running the pool's other
        // work first.
        let count = helped.len();
        loop {
            let mut helping = false;
            for k in 0..count {
                let Some(ahead) = &helped[(context.index() + k) % count] else {
                    continue;
                };
                if let Some(mut helper) = ahead.helper(robot, Some(&optimizing)) {
                    edges::run(
                        &mut helper,
                        robot,
                        lanes,
                        worlds,
                        deadline,
                        None,
                        None,
                        &mut scratch,
                    );
                    helping = true;
                }
            }
            if !helping {
                if !helped.iter().flatten().any(|ahead| ahead.sampling()) {
                    break;
                }
                while rayon::yield_now() == Some(Yield::Executed) {}
                std::thread::sleep(WAIT);
            }
        }
    });
}

impl Backend for CpuBackend {
    fn name(&self) -> String {
        format!("cpu ({} threads)", self.pool.current_num_threads())
    }

    fn robot(&self) -> &Robot {
        &self.robot
    }

    fn upload(&self, worlds: &[World]) -> Result<Box<dyn Any + Send + Sync>> {
        Ok(Box::new(prepare(&self.robot, worlds)?))
    }

    fn with_robot(&self, robot: &Robot) -> Result<Box<dyn Backend>> {
        Ok(Box::new(Self {
            robot: robot.clone(),
            lanes: Arc::new(Lanes::new(robot)),
            pool: self.pool.clone(),
        }))
    }

    fn evaluate(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        q: &[f32],
        w: &CollisionWeights,
    ) -> Result<Evaluation> {
        let n = self.robot.dof();
        let prepared: &CpuWorlds = worlds.prepared();
        let items = item_world.len();
        let mut out = Evaluation {
            world_clearance: vec![0.0; items],
            self_clearance: vec![0.0; items],
            cost: vec![0.0; items],
            gradient: vec![0.0; items * n],
        };
        let rows = out
            .world_clearance
            .par_iter_mut()
            .zip(out.self_clearance.par_iter_mut())
            .zip(out.cost.par_iter_mut());
        self.pool.install(|| {
            rows.zip(out.gradient.par_chunks_mut(n))
                .zip(q.par_chunks(n).zip(item_world.par_iter()))
                .for_each_init(
                    || Scratch::new(&self.robot),
                    |scratch, ((((wc, sc), cost), g), (qi, &s))| {
                        self.robot.fk_into(qi, &mut scratch.fk);
                        let world = &prepared[s as usize];
                        let c = collision::<true>(
                            &self.robot,
                            world,
                            &scratch.fk,
                            w,
                            g,
                            &mut scratch.collision,
                        );
                        (*wc, *sc, *cost) = (c.world_clearance, c.self_clearance, c.cost);
                    },
                )
        });
        Ok(out)
    }

    fn clearance(&self, worlds: &Worlds, item_world: &[u32], q: &[f32]) -> Result<Vec<[f32; 2]>> {
        let n = self.robot.dof();
        let prepared: &CpuWorlds = worlds.prepared();
        let level = self.lanes.level();
        let mut out = vec![[0.0; 2]; item_world.len()];
        // Eight configurations at a time, in runs that share a world, each dispatched to the CPU's
        // SIMD level inside the closure: a closure does not take on a dispatched caller's level.
        let chunk = |scratch: &mut LaneScratch, (c, (q, worlds)): Lanes8<'_>| {
            let mut start = 0;
            while start < worlds.len() {
                let w = worlds[start];
                let end = start + worlds[start..].iter().take_while(|&&x| x == w).count();
                let (world, q, c) = (
                    &prepared[w as usize],
                    &q[start * n..end * n],
                    &mut c[start..end],
                );
                dispatch!(level, simd => self.lanes.clearance(simd, world, q, c, scratch));
                start = end;
            }
        };
        if item_world.len() <= SMALL_BATCH {
            // Handing a batch this small to the pool's threads costs more than checking it here.
            let mut scratch = self.lanes.scratch();
            let chunks = out
                .chunks_mut(LANES)
                .zip(q.chunks(n * LANES).zip(item_world.chunks(LANES)));
            for c in chunks {
                chunk(&mut scratch, c);
            }
        } else {
            let chunks = out
                .par_chunks_mut(LANES)
                .zip(q.par_chunks(n * LANES).zip(item_world.par_chunks(LANES)));
            self.pool
                .install(|| chunks.for_each_init(|| self.lanes.scratch(), chunk));
        }
        Ok(out)
    }

    fn search(&self, worlds: &Worlds, searches: &mut [&mut dyn Search], deadline: Option<Instant>) {
        let prepared: &CpuWorlds = worlds.prepared();
        run_searches(
            &self.pool,
            searches,
            &self.robot,
            &self.lanes,
            prepared,
            deadline,
        );
    }

    fn ik(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        targets: &[Option<Pose>],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 2]>> {
        let (n, hands) = (self.robot.dof(), self.robot.hands());
        let prepared: &CpuWorlds = worlds.prepared();
        let hand_joints: Vec<Vec<usize>> = (0..hands).map(|h| self.robot.hand_joints(h)).collect();
        Ok(self.pool.install(|| {
            q.par_chunks_mut(n)
                .zip(item_world.par_iter().zip(targets.par_chunks(hands)))
                .map_init(
                    || (Scratch::new(&self.robot), vec![None; hands]),
                    |(scratch, goals), (qi, (&s, item_targets))| {
                        let world = &prepared[s as usize];
                        scratch.moving.fill(false);
                        for (h, (goal, target)) in goals.iter_mut().zip(item_targets).enumerate() {
                            *goal = target.map(|t| (t.position, Mat3::from_quat(t.rotation)));
                            if goal.is_some() {
                                for &j in &hand_joints[h] {
                                    scratch.moving[j] = true;
                                }
                            }
                        }
                        for _ in 0..o.iterations {
                            ik_step(&self.robot, world, goals, qi, o, scratch);
                        }
                        self.robot.fk_into(qi, &mut scratch.fk);
                        let mut worst = [0.0f32; 2];
                        for (h, goal) in goals.iter().enumerate() {
                            if let Some(goal) = goal {
                                let (_, pos_err, rot_err) = pose_error(
                                    &scratch.fk,
                                    self.robot.ee_links[h],
                                    goal,
                                    o.rotation_weight,
                                );
                                worst = [worst[0].max(pos_err), worst[1].max(rot_err)];
                            }
                        }
                        worst
                    },
                )
                .collect()
        }))
    }

    fn trajopt(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        paths: &mut JointPaths,
        o: &PlanOptions,
        deadline: Option<Instant>,
    ) -> Result<()> {
        let prepared: &CpuWorlds = worlds.prepared();
        if o.iterations == 0 {
            return Ok(());
        }
        // Fewer paths than threads (batch-1 planning): spread each path's samples over threads too.
        let per_sample = item_world.len() < self.pool.current_num_threads();
        let points = paths.points * paths.dof;
        self.pool.install(|| {
            paths
                .positions
                .par_chunks_mut(points)
                .zip(item_world.par_iter())
                .for_each(|(tr, &s)| {
                    optimize_path(
                        &self.robot,
                        &prepared[s as usize],
                        tr,
                        o,
                        deadline,
                        per_sample,
                    )
                })
        });
        Ok(())
    }

    fn grid_values(&self, grid: &Layout, occupancy: Occupancy) -> Result<Vec<u16>> {
        self.pool.install(|| {
            let seen;
            let occupied = match occupancy {
                Occupancy::Given(occupied) => occupied,
                Occupancy::Seen {
                    image,
                    robot,
                    behind,
                } => {
                    seen = sdf::depth_occupancy(grid, image, robot, behind);
                    &seen
                }
            };
            let squared = sdf::squared_edt(occupied, grid.dims.to_array().map(|d| d as usize));
            sdf::finish(grid, occupied, &squared)
        })
    }

    fn integrate(
        &self,
        grid: &Layout,
        log_odds: &mut [i8],
        image: &DepthImage,
        robot: Option<&RobotMask>,
    ) -> Result<()> {
        self.pool
            .install(|| sdf::integrate(grid, log_odds, image, robot));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use super::*;
    use crate::rng::Rng;
    use crate::robot::{CollisionModel, RobotOptions};
    use crate::world::{Obstacle, World};

    #[test]
    fn lbfgs_direction_applies_the_bfgs_inverse_hessian() {
        // Three remembered steps in a ring of four, newest in slot 0 after wrapping around; the
        // two-loop recursion must equal -H g with H built by explicit BFGS updates.
        let (n, points) = (2, 10);
        let o = PlanOptions {
            history: 4,
            ..Default::default()
        };
        let mut st = Lbfgs::new(points, n, 1, &o);
        let free: Vec<usize> = (3 * n..(points - 3) * n).collect();
        let mut rng = Rng::new(1);
        let tn = points * n;
        let mut pairs = vec![];
        for (age, slot) in [(0, 2), (1, 3), (2, 0)] {
            let s: Vec<f32> = (0..tn)
                .map(|i| {
                    if free.contains(&i) {
                        rng.range(-1.0, 1.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            // y = A s for a fixed positive diagonal A, so s.y > 0.
            let y: Vec<f32> = s
                .iter()
                .enumerate()
                .map(|(i, v)| v * (1.0 + i as f32 * 0.1))
                .collect();
            st.steps[slot * tn..(slot + 1) * tn].copy_from_slice(&s);
            st.changes[slot * tn..(slot + 1) * tn].copy_from_slice(&y);
            pairs.push((age, s, y));
        }
        (st.count, st.newest) = (3, 0);
        for &i in &free {
            st.grad[i] = rng.range(-1.0, 1.0);
        }
        lbfgs_direction(&mut st, n, &o);
        // H = gamma I, then each pair oldest first: H <- (I - r s y') H (I - r y s') + r s s'.
        let k = free.len();
        let (_, s_new, y_new) = &pairs[2];
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let pick = |v: &[f32]| free.iter().map(|&i| v[i] as f64).collect::<Vec<f64>>();
        let gamma = dot(&pick(s_new), &pick(y_new)) / dot(&pick(y_new), &pick(y_new));
        let mut h: Vec<f64> = (0..k * k)
            .map(|i| if i % (k + 1) == 0 { gamma } else { 0.0 })
            .collect();
        for (_, s, y) in &pairs {
            let (s, y) = (pick(s), pick(y));
            let r = 1.0 / dot(&s, &y);
            let left: Vec<f64> = (0..k * k)
                .map(|i| f64::from(i / k == i % k) - r * s[i / k] * y[i % k])
                .collect();
            let mul = |a: &[f64], b: &[f64]| -> Vec<f64> {
                (0..k * k)
                    .map(|i| (0..k).map(|m| a[i / k * k + m] * b[m * k + i % k]).sum())
                    .collect()
            };
            let right: Vec<f64> = (0..k * k).map(|i| left[i % k * k + i / k]).collect();
            h = mul(&mul(&left, &h), &right);
            for i in 0..k * k {
                h[i] += r * s[i / k] * s[i % k];
            }
        }
        let g = pick(&st.grad);
        for (row, &i) in free.iter().enumerate() {
            let expected = -(0..k).map(|c| h[row * k + c] * g[c]).sum::<f64>();
            assert!(
                (st.dir[i] as f64 - expected).abs() < 1e-4 * expected.abs().max(1.0),
                "{} vs {expected}",
                st.dir[i]
            );
        }
    }

    #[test]
    fn a_failed_line_search_resets_the_history() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/franka");
        let spheres = CollisionModel::load(format!("{dir}/panda_collision.json")).unwrap();
        let options = RobotOptions {
            collision_model: Some(spheres),
            ..Default::default()
        };
        let robot = Robot::load(format!("{dir}/franka_panda.urdf"), &options).unwrap();
        let (n, points) = (robot.dof(), 10);
        let o = PlanOptions {
            control_points: points,
            ..Default::default()
        };
        let mut st = Lbfgs::new(points, n, 1, &o);
        let mut x = vec![0.0; points * n];
        (st.started, st.cost, st.count) = (true, 1.0, 3);
        st.costs.fill(2.0);
        traj_search(&robot, &mut x, &mut st, &o);
        assert_eq!(
            (st.count, st.pending),
            (0, false),
            "a step that raises the cost must not be taken"
        );
        assert!(x.iter().all(|&v| v == 0.0));
        st.costs.fill(0.0);
        st.count = 3;
        traj_search(&robot, &mut x, &mut st, &o);
        assert!(
            st.pending && st.count == 3 && st.cost < 1.0,
            "a cheaper step must be taken"
        );
    }

    /// Repairs run the optimizer a round at a time and check each round's path, so its rounds
    /// split over calls must take the steps of one uninterrupted run.
    #[test]
    fn split_optimization_rounds_take_the_same_steps() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/franka");
        let spheres = CollisionModel::load(format!("{dir}/panda_collision.json")).unwrap();
        let options = RobotOptions {
            collision_model: Some(spheres),
            ..Default::default()
        };
        let fingers = ["panda_finger_joint1", "panda_finger_joint2"].map(|j| (j.to_string(), 0.04));
        let options = RobotOptions {
            lock_joints: fingers.into_iter().collect(),
            ..options
        };
        let robot = Robot::load(format!("{dir}/franka_panda.urdf"), &options).unwrap();
        let world = World::new(vec![Obstacle::Cuboid {
            center: Vec3::new(0.45, 0.0, 0.45),
            half_extents: Vec3::new(0.05, 0.05, 0.3),
            rotation: Quat::IDENTITY,
        }]);
        let prepared = prepare(&robot, &[world]).expect("the box is well formed");
        let n = robot.dof();
        // A repair's ten rounds.
        let o = PlanOptions {
            iterations: 10,
            ..Default::default()
        };
        let (start, goal) = (
            [-1.2, 0.3, 0.0, -1.8, 0.0, 2.0, 0.8],
            [1.2, 0.3, 0.0, -1.8, 0.0, 2.0, 0.8],
        );
        // The straight line from start to goal swings the hand through the box.
        let path: Vec<f32> = (0..o.control_points)
            .flat_map(|t| {
                let a = ((t.max(2) - 2) as f32 / (o.control_points - 5) as f32).min(1.0);
                (0..n).map(move |j| start[j] + a * (goal[j] - start[j]))
            })
            .collect();
        let mut whole = path.clone();
        optimize_path(&robot, &prepared[0], &mut whole, &o, None, false);
        assert_ne!(
            whole, path,
            "the path must cross the box, or nothing moves it"
        );
        let mut split = path;
        let mut optimizer = PathOptimizer::new(&robot, o.control_points, &o);
        // The optimization may stop early (a failed steepest-descent step), but not before a few
        // rounds have resumed where the last call stopped.
        let mut calls = 0;
        for through in 1..=o.iterations {
            if optimizer.finished() {
                break;
            }
            optimizer.run(&robot, &prepared[0], &mut split, through, None, false);
            calls += 1;
        }
        assert!(
            calls >= 3,
            "only {calls} calls ran before the optimization stopped"
        );
        assert!(optimizer.finished());
        assert_eq!(split, whole);
    }
}
