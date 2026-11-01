//! Constraints on where a hand may be along a path ([`HandConstraint`]): its origin inside a box,
//! its z axis within a cone around a reference axis (its tilt), and its turn about that axis (its
//! twist), all measured from a pose in the world or in another hand's frame, as two hands holding
//! one object keep their places on it. Every state a search, a shortcut or a spline is
//! checked at is tested against them, in the SIMD lanes beside the collision test
//! (`Lanes::keeps`). Searches walk toward their samples in short hops, each projected back inside
//! by damped least squares on the bounds it breaks (as CBiRRT does), landing a
//! little inside so that the edges between hops, and the spline that cuts their corners, have room
//! to bulge.
//!
//! The hand's rotation relative to the pose is `M = Sᵀ R`, with `S` the pose's rotation in the
//! world: `R0`, or `Rf R0` in the frame of a hand at `Rf`. Its tilt is the angle between the two
//! z axes, `acos(m22)`, and its twist the turn about z left once the tilt is taken out (the
//! swing-twist decomposition), `atan2(m10 − m01, m00 + m11)`. Both are defined for every rotation
//! but a tilt of half a turn, where roll-pitch-yaw angles are singular at a pitch of ±90°, and the
//! tilt bound is a cone, as "within 10° of upright" means, rather than a box on two angles.

use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use glam::{Mat3, Vec3};

use crate::error::{Result, ensure_input};
use crate::robot::{Fk, Robot};
use crate::types::Pose;

/// Where hand `hand` may be at every point of a path: its origin within `position` of
/// `pose.position` along each axis, its z axis within `tilt` of `pose`'s z axis, and turned about
/// that axis within `twist` of `pose`'s turn, all in the world or, with `relative_to`, in another
/// hand's frame. `HandConstraint::new` leaves all three free; set the ones that matter on top of
/// it: a tray carried upright is `HandConstraint { tilt: 0.17, ..HandConstraint::new(0, pose) }`,
/// and a second hand holding it with the first is pinned to its place in the first's frame,
/// `HandConstraint { relative_to: Some(0), position: Vec3::ZERO, tilt: 0.0, twist: 0.0,
/// ..HandConstraint::new(1, place) }`.
///
/// Bounds below what IK solves to (1 mm, 0.01 rad, or twice that relative to another hand, whose
/// error adds) count as those, so a goal that IK solved for the pose keeps a constraint pinned to
/// it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandConstraint {
    /// The hand (`RobotOptions::ee_links`).
    pub hand: usize,
    /// The hand whose frame `pose` and the bounds are in, moving with it; `None` for the world.
    pub relative_to: Option<usize>,
    /// The pose the bounds are measured from.
    pub pose: Pose,
    /// How far the hand's origin may be from `pose.position` along each axis of the world, or of
    /// the hand it is relative to (meters): a box. Infinity leaves an axis free.
    pub position: Vec3,
    /// How far the hand's z axis may tilt from `pose`'s z axis (radians): a cone. π or more leaves
    /// it free.
    pub tilt: f32,
    /// How far the hand may turn about its z axis from `pose`'s turn, once its tilt is taken out
    /// (radians). π or more leaves it free. A tilt of half a turn leaves no turn to measure, so a
    /// bounded twist needs a tilt bound of at most π/2.
    pub twist: f32,
}

impl HandConstraint {
    /// Hand `hand` measured from `pose` in the world, with every bound free.
    #[must_use]
    pub fn new(hand: usize, pose: Pose) -> Self {
        Self {
            hand,
            relative_to: None,
            pose,
            position: Vec3::INFINITY,
            tilt: PI,
            twist: PI,
        }
    }
}

/// A bound below what IK solves to counts as IK's default tolerance (`IkOptions`): meters. Relative
/// to another hand, twice it.
const POSITION_FLOOR: f32 = 1e-3;
/// The same for angles: radians.
const ROTATION_FLOOR: f32 = 1e-2;
/// How far inside each bound projection lands: half the bound, up to these (meters, radians). The
/// edges between projected states bulge out by a few milliradians, and the splines that cut their
/// corners by more, which is the room they need; a wider bound needs no more of it.
const POSITION_INSET: f32 = 0.01;
const ROTATION_INSET: f32 = 0.035;
/// Steps a projection takes at most, and the residual (meters or radians) past its bounds it
/// stops at: far within the room the insets leave.
const PROJECTION_STEPS: usize = 20;
const SETTLED: f32 = 1e-4;
/// Damping of the projection's least squares, and the largest change of a joint in one step.
const DAMPING: f32 = 1e-2;
const MAX_STEP: f32 = 0.5;
/// Trajectory optimization's weight on the squared distance (meters, radians) past each bound
/// projection lands within, at each of its samples.
pub(crate) const CONSTRAINT_WEIGHT: f32 = 1e4;

/// An `Error::Input` unless every constraint names one of `robot`'s hands and has bounds it can
/// keep: finite where given, and none negative.
pub(crate) fn check(robot: &Robot, constraints: &[HandConstraint], what: &str) -> Result<()> {
    for (k, c) in constraints.iter().enumerate() {
        ensure_input!(
            c.hand < robot.hands(),
            "{what}: constraint {k} names hand {} of {}",
            c.hand,
            robot.hands()
        );
        if let Some(other) = c.relative_to {
            ensure_input!(
                other < robot.hands() && other != c.hand,
                "{what}: constraint {k} on hand {} is relative to hand {other}, which must be \
                 another of the robot's {}",
                c.hand,
                robot.hands()
            );
        }
        let p = c.pose;
        ensure_input!(
            p.position.is_finite() && p.rotation.is_finite() && p.rotation.length() > 0.5,
            "{what}: constraint {k}'s pose must be finite, with a unit rotation, got {p:?}"
        );
        ensure_input!(
            c.position.cmpge(Vec3::ZERO).all() && c.tilt >= 0.0 && c.twist >= 0.0,
            "{what}: constraint {k}'s bounds must not be negative or NaN, got {} m, {} and {} rad",
            c.position,
            c.tilt,
            c.twist
        );
        ensure_input!(
            c.twist >= PI || c.tilt <= FRAC_PI_2,
            "{what}: constraint {k} bounds its twist, so its tilt must be bounded to at most π/2, \
             got {} rad: a hand tilted half a turn has no turn about the z axis to measure",
            c.tilt
        );
    }
    Ok(())
}

/// Why `q` breaks the constraints, if it does: which one, what, and by how much.
pub(crate) fn breach(robot: &Robot, constraints: &[HandConstraint], q: &[f32]) -> Option<String> {
    if constraints.is_empty() {
        return None;
    }
    let fk = robot.fk(q);
    constraints.iter().enumerate().find_map(|(k, c)| {
        let limit = Limit::new(robot, c, /* inset */ false);
        let placed = limit.placed(&fk);
        let m = limit.relative(&fk, &placed);
        let away = limit.away(&fk, &placed).abs() - limit.half;
        let tilt = m.m22.clamp(-1.0, 1.0).acos();
        let twist = m.n.atan2(m.d).abs();
        if away.max_element() > 0.0 {
            Some(format!(
                "hand {} is {:.1} mm past constraint {k}'s box",
                c.hand,
                1e3 * away.max_element()
            ))
        } else if tilt > limit.tilt {
            Some(format!(
                "hand {} tilts {:.2}° from constraint {k}'s z axis, past {:.2}°",
                c.hand,
                tilt.to_degrees(),
                limit.tilt.to_degrees()
            ))
        } else if twist > limit.twist {
            Some(format!(
                "hand {} turns {:.2}° from constraint {k}'s turn, past {:.2}°",
                c.hand,
                twist.to_degrees(),
                limit.twist.to_degrees()
            ))
        } else {
            None
        }
    })
}

/// A problem's constraints prepared for checks: the bounds as given (with their floors), or the
/// bounds projection lands within.
#[derive(Clone, Debug)]
pub(crate) struct Region {
    pub(crate) limits: Vec<Limit>,
}

/// One constraint, prepared. A free bound is infinite, with a cosine of -2, which every rotation
/// keeps.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limit {
    pub(crate) hand: usize,
    /// The hand's frame (`Robot::ee_links`).
    pub(crate) link: usize,
    /// The hand it is measured relative to, if any.
    pub(crate) frame: Option<Frame>,
    /// The pose's rotation, by columns, in the world or the frame.
    pub(crate) reference: Mat3,
    pub(crate) center: Vec3,
    pub(crate) half: Vec3,
    pub(crate) tilt: f32,
    pub(crate) twist: f32,
    pub(crate) cos_tilt: f32,
    pub(crate) cos_twist: f32,
    /// The sine and cosine of `tilt` and of `twist`, for `past`: NaN when free.
    sin_cos_tilt: (f32, f32),
    sin_cos_twist: (f32, f32),
}

/// A hand whose frame a limit is measured in: its index and its frame's link.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frame {
    pub(crate) hand: usize,
    pub(crate) link: usize,
}

/// A limit's frame where a state has it: the pose's rotation in the world (`S`), and the box's
/// centre and axes, and the origin of the hand it is relative to (zero in the world).
struct Placed {
    rotation: Mat3,
    center: Vec3,
    axes: Mat3,
    origin: Vec3,
}

/// The entries of `M = Sᵀ R` the bounds read: `m22`, the twist's `n = m10 - m01` and
/// `d = m00 + m11`, and `m20` and `m21`, which its rate needs.
struct Relative {
    m22: f32,
    n: f32,
    d: f32,
    m20: f32,
    m21: f32,
}

impl Region {
    pub(crate) fn new(robot: &Robot, constraints: &[HandConstraint]) -> Self {
        Self::at(robot, constraints, /* inset */ false)
    }

    /// The bounds projection lands within, which trajectory optimization keeps its samples in.
    pub(crate) fn inset(robot: &Robot, constraints: &[HandConstraint]) -> Self {
        Self::at(robot, constraints, /* inset */ true)
    }

    /// The bounds, or with `inset` the bounds projection lands within.
    fn at(robot: &Robot, constraints: &[HandConstraint], inset: bool) -> Self {
        Self {
            limits: constraints
                .iter()
                .map(|c| Limit::new(robot, c, inset))
                .collect(),
        }
    }

    /// Whether the configuration `fk` holds keeps every bound, as `Lanes::keeps` tests it.
    pub(crate) fn contains(&self, fk: &Fk) -> bool {
        self.limits.iter().all(|l| l.contains(fk))
    }

    /// Trajectory optimization's cost at the configuration `fk` holds: `CONSTRAINT_WEIGHT` times
    /// the squared distance past each bound; with `grad`, its gradient per joint added there,
    /// through the Jacobians of each constraint's hand and of the hand it is relative to. Must
    /// match `constraint_cost` in kernels.wgsl.
    #[inline]
    pub(crate) fn cost(
        &self,
        robot: &Robot,
        fk: &Fk,
        grad: Option<&mut [f32]>,
        s: &mut CostScratch,
    ) -> f32 {
        s.rows.clear();
        self.rows(fk, &mut s.rows);
        let w = CONSTRAINT_WEIGHT;
        let cost = s.rows.iter().map(|r| w * r.residual * r.residual).sum();
        let Some(grad) = grad else {
            return cost;
        };
        for (k, l) in self.limits.iter().enumerate() {
            // The cost's rate per unit of each hand's linear and angular velocity.
            let (mut hand, mut frame) = ([Vec3::ZERO; 2], [Vec3::ZERO; 2]);
            for r in s.rows.iter().filter(|r| r.limit == k) {
                let g = 2.0 * w * r.residual;
                hand = [hand[0] + g * r.linear, hand[1] + g * r.angular];
                frame = [frame[0] + g * r.frame[0], frame[1] + g * r.frame[1]];
            }
            let mut add = |which: usize, rate: [Vec3; 2], columns: &mut Vec<[Vec3; 2]>| {
                if rate != [Vec3::ZERO; 2] {
                    robot.hand_jacobian(fk, which, columns);
                    for (g, c) in grad.iter_mut().zip(columns.iter()) {
                        *g += rate[0].dot(c[0]) + rate[1].dot(c[1]);
                    }
                }
            };
            add(l.hand, hand, &mut s.columns);
            if let Some(f) = l.frame {
                add(f.hand, frame, &mut s.columns);
            }
        }
        cost
    }

    /// Each bound the configuration `fk` holds breaks: how far past it, and that distance's rate
    /// per unit of the hand's linear and angular velocity (world axes), and of the hand it is
    /// relative to.
    fn rows(&self, fk: &Fk, out: &mut Vec<Row>) {
        for (k, l) in self.limits.iter().enumerate() {
            let placed = l.placed(fk);
            let lever = fk.pos[l.link] - placed.origin;
            // The frame's part of a rate: none in the world.
            let frame = |linear: Vec3, angular: Vec3| {
                if l.frame.is_some() {
                    [linear, angular]
                } else {
                    [Vec3::ZERO; 2]
                }
            };
            let d = l.away(fk, &placed);
            for axis in 0..3 {
                let past = d[axis] - d[axis].clamp(-l.half[axis], l.half[axis]);
                if past != 0.0 {
                    // Along the box's axis `a`, which turns with the frame: `a · (v - vf)` plus
                    // the frame's turn, `ωf · (a × (p - pf))`.
                    let a = placed.axes.col(axis);
                    out.push(Row {
                        limit: k,
                        residual: past,
                        linear: a,
                        angular: Vec3::ZERO,
                        frame: frame(-a, a.cross(lever)),
                    });
                }
            }
            let r = fk.rot[l.link];
            let m = l.relative(fk, &placed);
            let (a, u) = (r.z_axis, placed.rotation.z_axis);
            let away = u.cross(a);
            let length = away.length();
            let residual = past(length, m.m22, l.sin_cos_tilt);
            if residual > 0.0 {
                // The tilt falls at the hand's turn about `u × a`, at one radian per radian, and
                // rises at the frame's.
                let axis = if length > 1e-6 {
                    away / length
                } else {
                    u.any_orthonormal_vector()
                };
                out.push(Row {
                    limit: k,
                    residual,
                    linear: Vec3::ZERO,
                    angular: axis,
                    frame: frame(Vec3::ZERO, -axis),
                });
            }
            let h2 = m.n * m.n + m.d * m.d;
            let over = past(m.n.abs(), m.d, l.sin_cos_twist);
            if h2 > 1e-12 && over > 0.0 {
                // In the pose's frame the twist turns at Ωz plus the parts of Ωx and Ωy the tilt
                // couples in, Ω the hand's angular velocity less the frame's; `S` takes the rates
                // back to world axes.
                let rate = Vec3::new(
                    (m.n * m.m21 - m.d * m.m20) / h2,
                    -(m.d * m.m21 + m.n * m.m20) / h2,
                    1.0,
                );
                let angular = placed.rotation * rate;
                out.push(Row {
                    limit: k,
                    residual: if m.n >= 0.0 { over } else { -over },
                    linear: Vec3::ZERO,
                    angular,
                    frame: frame(Vec3::ZERO, -angular),
                });
            }
        }
    }
}

/// How far the angle of (`x`, `y`) is past the bound whose sine and cosine are `(s, c)`, negative
/// within it and NaN when it is free: one small angle rather than the difference of two near ones,
/// which GPU drivers' acos and atan2 (good to 1e-4 rad and 2e-5 of the angle on RADV) swamp. As
/// angle_past in kernels.wgsl.
fn past(y: f32, x: f32, (s, c): (f32, f32)) -> f32 {
    (y * c - x * s).atan2(x * c + y * s)
}

impl Limit {
    fn new(robot: &Robot, c: &HandConstraint, inset: bool) -> Self {
        let within = |bound: f32, most: f32| {
            if inset {
                bound - (0.5 * bound).min(most)
            } else {
                bound
            }
        };
        let floor = if c.relative_to.is_some() { 2.0 } else { 1.0 };
        let angle = |bound: f32| {
            if bound >= PI {
                f32::INFINITY
            } else {
                within(bound.max(floor * ROTATION_FLOOR), ROTATION_INSET)
            }
        };
        let half = c.position.max(Vec3::splat(floor * POSITION_FLOOR));
        let cos = |angle: f32| if angle.is_finite() { angle.cos() } else { -2.0 };
        let sin_cos = |angle: f32| {
            if angle.is_finite() {
                angle.sin_cos()
            } else {
                (f32::NAN, f32::NAN)
            }
        };
        let (tilt, twist) = (angle(c.tilt), angle(c.twist));
        Self {
            hand: c.hand,
            link: robot.ee_links[c.hand],
            frame: c.relative_to.map(|hand| Frame {
                hand,
                link: robot.ee_links[hand],
            }),
            reference: Mat3::from_quat(c.pose.rotation.normalize()),
            center: c.pose.position,
            half: Vec3::from_array(half.to_array().map(|h| within(h, POSITION_INSET))),
            tilt,
            twist,
            cos_tilt: cos(tilt),
            cos_twist: cos(twist),
            sin_cos_tilt: sin_cos(tilt),
            sin_cos_twist: sin_cos(twist),
        }
    }

    /// Where `fk` has the frame: the world's, or the hand's it is relative to. Written as
    /// `Lanes::keeps` computes it.
    fn placed(&self, fk: &Fk) -> Placed {
        match self.frame {
            None => Placed {
                rotation: self.reference,
                center: self.center,
                axes: Mat3::IDENTITY,
                origin: Vec3::ZERO,
            },
            Some(f) => {
                let (rf, pf) = (fk.rot[f.link], fk.pos[f.link]);
                Placed {
                    rotation: rf * self.reference,
                    center: pf + rf * self.center,
                    axes: rf,
                    origin: pf,
                }
            }
        }
    }

    /// The hand's origin from the box's centre, along the box's axes.
    fn away(&self, fk: &Fk, placed: &Placed) -> Vec3 {
        let d = fk.pos[self.link] - placed.center;
        match self.frame {
            None => d,
            Some(_) => {
                let a = placed.axes;
                Vec3::new(a.x_axis.dot(d), a.y_axis.dot(d), a.z_axis.dot(d))
            }
        }
    }

    fn relative(&self, fk: &Fk, placed: &Placed) -> Relative {
        let (r, s) = (fk.rot[self.link], placed.rotation);
        Relative {
            m22: r.z_axis.dot(s.z_axis),
            n: r.x_axis.dot(s.y_axis) - r.y_axis.dot(s.x_axis),
            d: r.x_axis.dot(s.x_axis) + r.y_axis.dot(s.y_axis),
            m20: r.x_axis.dot(s.z_axis),
            m21: r.y_axis.dot(s.z_axis),
        }
    }

    /// Written as `Lanes::keeps` tests it, so the two agree; NaN breaks every bound.
    fn contains(&self, fk: &Fk) -> bool {
        let placed = self.placed(fk);
        let away = self.away(fk, &placed).abs();
        let m = self.relative(fk, &placed);
        away.cmple(self.half).all()
            && m.m22 >= self.cos_tilt
            && m.d >= self.cos_twist * (m.n * m.n + m.d * m.d).sqrt()
    }
}

/// Working memory for `Region::cost`: the bounds broken, and a hand's Jacobian.
#[derive(Clone, Debug)]
pub(crate) struct CostScratch {
    rows: Vec<Row>,
    columns: Vec<[Vec3; 2]>,
}

impl CostScratch {
    pub(crate) fn new(robot: &Robot) -> Self {
        Self {
            rows: Vec::new(),
            columns: vec![[Vec3::ZERO; 2]; robot.dof()],
        }
    }
}

/// A bound broken by `residual`, which changes at `linear · v + angular · ω` as the hand moves,
/// plus `frame[0] · vf + frame[1] · ωf` as the hand it is relative to moves (zero in the world).
#[derive(Clone, Copy, Debug)]
struct Row {
    limit: usize,
    residual: f32,
    linear: Vec3,
    angular: Vec3,
    frame: [Vec3; 2],
}

/// Projects configurations into a problem's constraints, a little inside their bounds, moving
/// only the joints not held.
pub(crate) struct Projector<'robot> {
    robot: &'robot Robot,
    held: Vec<bool>,
    region: Region,
    /// The bounds as given, which checks keep, and which a configuration that keeps them already
    /// is taken as it is.
    bounds: Arc<Region>,
    fk: Fk,
    rows: Vec<Row>,
    /// Each constraint's hand Jacobian, per joint (linear, angular), and that of the hand it is
    /// relative to (empty in the world).
    columns: Vec<[Vec<[Vec3; 2]>; 2]>,
    jac: Vec<f32>,
    normal: Vec<f32>,
    solved: Vec<f32>,
    dq: Vec<f32>,
}

impl<'robot> Projector<'robot> {
    /// For `constraints` on `robot`, the joints `hold` lists staying.
    pub(crate) fn new(
        robot: &'robot Robot,
        constraints: &[HandConstraint],
        hold: &[usize],
    ) -> Self {
        let (n, k) = (robot.dof(), constraints.len());
        let rows = 5 * k;
        Self {
            robot,
            held: (0..n).map(|j| hold.contains(&j)).collect(),
            region: Region::at(robot, constraints, /* inset */ true),
            bounds: Arc::new(Region::new(robot, constraints)),
            fk: Fk::new(robot.links.len()),
            rows: Vec::with_capacity(rows),
            columns: constraints
                .iter()
                .map(|c| {
                    let frame = if c.relative_to.is_some() { n } else { 0 };
                    [vec![[Vec3::ZERO; 2]; n], vec![[Vec3::ZERO; 2]; frame]]
                })
                .collect(),
            jac: Vec::with_capacity(rows * n),
            normal: Vec::with_capacity(rows * rows),
            solved: Vec::with_capacity(rows),
            dq: vec![0.0; n],
        }
    }

    /// The bounds as given, for checks to keep.
    pub(crate) fn bounds(&self) -> Arc<Region> {
        self.bounds.clone()
    }

    /// Whether `q` keeps the bounds as given.
    pub(crate) fn keeps(&mut self, q: &[f32]) -> bool {
        self.robot.fk_into(q, &mut self.fk);
        self.bounds.contains(&self.fk)
    }

    /// Moves `q` inside every bound (see `POSITION_INSET`) by damped least squares on the bounds
    /// it breaks, held joints staying, each joint within `Robot::bounds`. False if it does not get
    /// there.
    pub(crate) fn project(&mut self, q: &mut [f32]) -> bool {
        let robot = self.robot;
        let n = robot.dof();
        for step in 0..=PROJECTION_STEPS {
            robot.fk_into(q, &mut self.fk);
            self.rows.clear();
            self.region.rows(&self.fk, &mut self.rows);
            if self.rows.iter().all(|r| r.residual.abs() <= SETTLED) {
                return true;
            }
            if step == PROJECTION_STEPS {
                break;
            }
            for ([hand, frame], limit) in self.columns.iter_mut().zip(&self.region.limits) {
                robot.hand_jacobian(&self.fk, limit.hand, hand);
                if let Some(f) = limit.frame {
                    robot.hand_jacobian(&self.fk, f.hand, frame);
                }
            }
            let m = self.rows.len();
            self.jac.clear();
            for row in &self.rows {
                let [hand, frame] = &self.columns[row.limit];
                self.jac.extend((0..n).map(|j| {
                    if self.held[j] {
                        return 0.0;
                    }
                    let mut rate = row.linear.dot(hand[j][0]) + row.angular.dot(hand[j][1]);
                    if let Some(f) = frame.get(j) {
                        rate += row.frame[0].dot(f[0]) + row.frame[1].dot(f[1]);
                    }
                    rate
                }));
            }
            // dq = -Jᵀ (J Jᵀ + λ² I)⁻¹ r.
            self.normal.clear();
            for a in 0..m {
                for b in 0..m {
                    let (ra, rb) = (&self.jac[a * n..(a + 1) * n], &self.jac[b * n..(b + 1) * n]);
                    let dot: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
                    self.normal
                        .push(dot + if a == b { DAMPING * DAMPING } else { 0.0 });
                }
            }
            self.solved.clear();
            self.solved.extend(self.rows.iter().map(|r| r.residual));
            if !solve(&mut self.normal, m, &mut self.solved) {
                return false;
            }
            self.dq.fill(0.0);
            for (r, y) in self.solved.iter().enumerate() {
                for (d, j) in self.dq.iter_mut().zip(&self.jac[r * n..(r + 1) * n]) {
                    *d -= j * y;
                }
            }
            let largest = self.dq.iter().fold(0.0f32, |m, d| m.max(d.abs()));
            let scale = if largest > MAX_STEP {
                MAX_STEP / largest
            } else {
                1.0
            };
            for (j, (v, d)) in q.iter_mut().zip(&self.dq).enumerate() {
                let (lo, hi) = robot.bounds(j);
                *v = (*v + d * scale).clamp(lo, hi);
            }
        }
        false
    }
}

/// Solves `a x = b` in place for a symmetric positive definite `a` (`m × m`, row-major) by
/// Cholesky factorization, leaving `x` in `b`; false if `a` is not positive definite.
fn solve(a: &mut [f32], m: usize, b: &mut [f32]) -> bool {
    for j in 0..m {
        let mut diagonal = a[j * m + j];
        for k in 0..j {
            diagonal -= a[j * m + k] * a[j * m + k];
        }
        if diagonal.is_nan() || diagonal <= 0.0 {
            return false;
        }
        let diagonal = diagonal.sqrt();
        a[j * m + j] = diagonal;
        for i in j + 1..m {
            let mut v = a[i * m + j];
            for k in 0..j {
                v -= a[i * m + k] * a[j * m + k];
            }
            a[i * m + j] = v / diagonal;
        }
    }
    for i in 0..m {
        let mut v = b[i];
        for k in 0..i {
            v -= a[i * m + k] * b[k];
        }
        b[i] = v / a[i * m + i];
    }
    for i in (0..m).rev() {
        let mut v = b[i];
        for k in i + 1..m {
            v -= a[k * m + i] * b[k];
        }
        b[i] = v / a[i * m + i];
    }
    true
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::rng::Rng;
    use crate::robot::RobotOptions;

    /// The Panda with a second hand at its fourth link, which shares the first's joints up to
    /// there: a constraint relative to it moves with the arm.
    fn two_frames() -> Robot {
        let fingers = [
            ("panda_finger_joint1".to_string(), 0.04),
            ("panda_finger_joint2".to_string(), 0.04),
        ];
        let options = RobotOptions {
            ee_links: vec!["panda_hand".into(), "panda_link4".into()],
            lock_joints: HashMap::from(fingers),
            ..Default::default()
        };
        Robot::load("assets/franka/franka_panda.urdf", &options).unwrap()
    }

    fn random(robot: &Robot, rng: &mut Rng) -> Vec<f32> {
        (0..robot.dof())
            .map(|j| rng.range(robot.lower[j], robot.upper[j]))
            .collect()
    }

    /// Each kind of row changes at its rate, through the Jacobians of the hand and of the hand it
    /// is relative to, along a small joint step: the projection steps in the direction that
    /// lowers what it breaks.
    #[test]
    fn rows_change_at_their_rates() {
        let robot = two_frames();
        let n = robot.dof();
        let mut rng = Rng::new(4);
        let kinds = [
            (Vec3::splat(0.05), PI, PI),
            (Vec3::INFINITY, 0.2, PI),
            (Vec3::INFINITY, PI, 0.3),
        ];
        for relative in [false, true] {
            for (kind, (position, tilt, twist)) in kinds.into_iter().enumerate() {
                let mut checked = 0;
                for _ in 0..300 {
                    let q = random(&robot, &mut rng);
                    let near = random(&robot, &mut rng);
                    let pose = if relative {
                        robot
                            .ee_pose(&near, 1)
                            .inverse()
                            .mul_pose(robot.ee_pose(&near, 0))
                    } else {
                        robot.ee_pose(&near, 0)
                    };
                    let c = HandConstraint {
                        relative_to: relative.then_some(1),
                        position,
                        tilt,
                        twist,
                        ..HandConstraint::new(0, pose)
                    };
                    let region = Region::new(&robot, &[c]);
                    let fk = robot.fk(&q);
                    // A bounded twist comes with a tilt bound of at most π/2: past a tilt of 2π/3,
                    // where its rate grows without bound, a finite step does not measure it.
                    let placed = region.limits[0].placed(&fk);
                    if twist < PI && region.limits[0].relative(&fk, &placed).m22 < -0.5 {
                        continue;
                    }
                    let mut rows = vec![];
                    region.rows(&fk, &mut rows);
                    let mut columns = [vec![[Vec3::ZERO; 2]; n], vec![[Vec3::ZERO; 2]; n]];
                    robot.hand_jacobian(&fk, 0, &mut columns[0]);
                    robot.hand_jacobian(&fk, 1, &mut columns[1]);
                    let dq: Vec<f32> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
                    let h = 1e-3;
                    let stepped: Vec<f32> = q.iter().zip(&dq).map(|(q, d)| q + h * d).collect();
                    let mut after = vec![];
                    region.rows(&robot.fk(&stepped), &mut after);
                    for row in &rows {
                        // The same bound, broken on the same side, after the step.
                        let same = |a: &&Row| {
                            let axis = a.linear.dot(row.linear) > 0.99
                                || a.linear == Vec3::ZERO && row.linear == Vec3::ZERO;
                            axis && a.residual.signum() == row.residual.signum()
                        };
                        let Some(next) = after.iter().find(same) else {
                            continue;
                        };
                        let rate: f32 = (0..n)
                            .map(|j| {
                                let [hand, frame] = &columns;
                                let along = row.linear.dot(hand[j][0])
                                    + row.angular.dot(hand[j][1])
                                    + row.frame[0].dot(frame[j][0])
                                    + row.frame[1].dot(frame[j][1]);
                                along * dq[j]
                            })
                            .sum();
                        let measured = (next.residual - row.residual) / h;
                        assert!(
                            (measured - rate).abs() <= 0.02 * (1.0 + rate.abs()),
                            "kind {kind}, relative {relative}: a row changes at {measured}, not \
                             its rate {rate}: {row:?}"
                        );
                        checked += 1;
                    }
                }
                assert!(
                    checked > 100,
                    "kind {kind}, relative {relative}: {checked} rows checked"
                );
            }
        }
    }

    /// The optimizer's cost changes at the rate its gradient gives, in the world and relative to
    /// a frame that moves with the arm: the gradient adds each broken bound's part through both
    /// hands' Jacobians.
    #[test]
    fn cost_gradient_matches_its_finite_differences() {
        let robot = two_frames();
        let n = robot.dof();
        let mut rng = Rng::new(12);
        let mut scratch = CostScratch::new(&robot);
        let mut checked = [0; 2];
        for relative in [false, true] {
            for _ in 0..200 {
                let q = random(&robot, &mut rng);
                let near: Vec<f32> = q.iter().map(|v| v + rng.range(-0.3, 0.3)).collect();
                let pose = if relative {
                    robot
                        .ee_pose(&near, 1)
                        .inverse()
                        .mul_pose(robot.ee_pose(&near, 0))
                } else {
                    robot.ee_pose(&near, 0)
                };
                let c = HandConstraint {
                    relative_to: relative.then_some(1),
                    position: Vec3::new(0.02, 0.03, 0.01),
                    tilt: 0.1,
                    twist: 0.2,
                    ..HandConstraint::new(0, pose)
                };
                let region = Region::new(&robot, &[c]);
                let mut grad = vec![0.0; n];
                let cost = region.cost(&robot, &robot.fk(&q), Some(&mut grad), &mut scratch);
                if cost == 0.0 {
                    continue;
                }
                let dq: Vec<f32> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
                // Rows' rates can cancel: the scale is the gradient's length times the step's.
                let h = 1e-3;
                let mut at = |t: f32| {
                    let moved: Vec<f32> = q.iter().zip(&dq).map(|(q, d)| q + t * d).collect();
                    let cost = region.cost(&robot, &robot.fk(&moved), None, &mut scratch);
                    (cost, scratch.rows.len())
                };
                let ((ahead, broken), (behind, before)) = (at(h), at(-h));
                // Where a bound turns on or off within the step, the cost's curvature jumps.
                if broken != before {
                    continue;
                }
                let measured = (ahead - behind) / (2.0 * h);
                let rate: f32 = grad.iter().zip(&dq).map(|(g, d)| g * d).sum();
                let length = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (measured - rate).abs() <= 0.01 * length(&grad) * length(&dq),
                    "relative {relative}: the cost changes at {measured}, its gradient says {rate}"
                );
                checked[usize::from(relative)] += 1;
            }
        }
        assert!(checked.iter().all(|&c| c > 100), "{checked:?} checked");
    }

    /// Projection lands inside the inset bounds, leaves held joints where they were, and a
    /// configuration it reports projected keeps the bounds as given: in the world, and relative to
    /// a frame that moves with the arm.
    #[test]
    fn projection_lands_inside_and_leaves_held_joints() {
        let robot = two_frames();
        let mut rng = Rng::new(7);
        let start = robot.default_q().to_vec();
        let world = HandConstraint {
            tilt: 0.1,
            twist: 0.4,
            ..HandConstraint::new(0, robot.ee_pose(&start, 0))
        };
        let relative = HandConstraint {
            relative_to: Some(1),
            position: Vec3::new(0.05, 0.02, 0.03),
            tilt: 0.1,
            twist: 0.2,
            ..HandConstraint::new(
                0,
                robot
                    .ee_pose(&start, 1)
                    .inverse()
                    .mul_pose(robot.ee_pose(&start, 0)),
            )
        };
        for constraints in [[world], [relative]] {
            let mut projector = Projector::new(&robot, &constraints, &[0]);
            let mut inside = Region::at(&robot, &constraints, /* inset */ true);
            for l in &mut inside.limits {
                l.half += Vec3::splat(2.0 * SETTLED);
                l.cos_tilt = (l.tilt + 2.0 * SETTLED).cos();
                l.cos_twist = (l.twist + 2.0 * SETTLED).cos();
            }
            let (mut projected, mut tried) = (0, 0);
            for _ in 0..200 {
                let mut q: Vec<f32> = start.iter().map(|v| v + rng.range(-0.4, 0.4)).collect();
                let fk = robot.fk(&q);
                if Region::new(&robot, &constraints).contains(&fk) {
                    continue;
                }
                tried += 1;
                let before = q[0];
                if projector.project(&mut q) {
                    projected += 1;
                    assert!(
                        inside.contains(&robot.fk(&q)),
                        "projected outside the inset bounds"
                    );
                    assert!(projector.keeps(&q));
                }
                assert_eq!(q[0], before, "a held joint moved");
            }
            assert!(
                tried > 100 && projected > tried * 9 / 10,
                "{projected} of {tried} projected"
            );
        }
    }
}
