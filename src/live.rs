//! Live motion: following a trajectory while the world changes ([`Follower`]), and servoing a
//! hand toward a target that moves ([`servo`]).
//!
//! A `Follower` checks the rest of its motion against the worlds it is given at every tick. When
//! the rest collides, it plans again from the state the robot will be in once the planning time
//! (and a lead, for a controller elsewhere) has passed, starting in motion, and switches to the new
//! trajectory then; when no plan arrives in time, or the collision comes sooner than that, it stops
//! on the path it is on ([`Trajectory::stop`]), naming what it stops for, checks the stop too, and
//! reports itself blocked at rest. One that keeps its path, as along a tool path, stops without
//! planning. `servo` takes one control step toward a hand pose: damped least squares on the hand's
//! Jacobian, within the joints' velocity, acceleration and jerk limits, slowed and then braked to
//! rest where braking from the state it commands would bring the robot too close to an obstacle,
//! and braking each joint in time to rest within its range.

use std::f32::consts::{PI, TAU};
use std::time::Duration;

use glam::Vec3;

use crate::constraint::{self, HandConstraint};
use crate::cpu::Reach;
use crate::device::{CollisionWeights, Contact, Device, Worlds};
use crate::error::{Result, ensure_input};
use crate::robot::Robot;
use crate::timing::{JointState, Trajectory};
use crate::trajopt::{PlanOptions, PlanProblem};
use crate::types::{Pose, StartMotion};

/// How a [`Follower`] checks and replans.
#[derive(Clone, Debug)]
pub struct FollowerOptions {
    /// The time a replan may take (seconds): 50 ms. It starts from where the robot will be once
    /// this and `lead` have passed.
    pub budget: f32,
    /// The time a replan has after its budget to reach the robot before it takes over (seconds): 0,
    /// for a controller in the same process. A controller elsewhere, sent each replan as
    /// [`FollowStatus::Replanned`] hands it over, needs about as long as sending takes. Only
    /// replans wait for it: a stop, a re-anchor and a collision take effect at the tick that
    /// reports them, and each drops a replan sent before. On `Stopping`, `Collides` or `Blocked`
    /// such a controller stops the robot itself; on `Reanchored` it takes up
    /// [`Follower::trajectory`] from [`Follower::began`] as soon as it can.
    pub lead: f32,
    /// Spacing in time of the checks along the rest of the motion (seconds), at least 1 ms: 20 ms.
    pub check_every: f32,
    /// How replans are planned; their time budget is `budget`.
    pub plan: PlanOptions,
    /// Where the hands must stay along replans ([`HandConstraint`]), as the trajectory was planned
    /// to keep them. A replan that would start where they break stops instead.
    pub constraints: Vec<HandConstraint>,
    /// How far a joint of the measured state a tick is given may be from where the trajectory has
    /// it at that tick (radians, or meters for a prismatic joint) before the follower plans again
    /// from the measured state, or holds the robot there without `replan`: 0.2. Set it above how
    /// far the robot lags its commands at speed. Within it, as a servo settles short under load,
    /// plans continue from the trajectory's own states, so commands do not step.
    pub tolerance: f32,
    /// Whether it plans a new motion to the goal where the rest of its motion collides or the
    /// robot is measured past `tolerance`: on. Off, it keeps the trajectory's path, as a motion
    /// along a tool path ([`crate::follow`]) must, which a replan would leave: where the rest
    /// collides, it stops on the path ([`FollowStatus::Stopping`]), and past `tolerance` it holds
    /// the robot where it is measured ([`FollowStatus::Blocked`]). `budget`, `lead`, `plan` and
    /// `constraints` shape replans only.
    pub replan: bool,
}

impl Default for FollowerOptions {
    fn default() -> Self {
        Self {
            budget: 0.05,
            lead: 0.0,
            check_every: 0.02,
            plan: PlanOptions::default(),
            constraints: vec![],
            tolerance: 0.2,
            replan: true,
        }
    }
}

/// What a [`Follower`] did at a tick.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum FollowStatus {
    /// On its trajectory, the rest of it clear.
    Following,
    /// The rest of the motion collided; `trajectory` takes over at `at` (seconds, on the caller's
    /// clock), from the state the old one reaches then. Until then the follower keeps commanding
    /// the old one, and [`Follower::trajectory`] is still it.
    Replanned {
        /// When the new trajectory takes over.
        at: f32,
        /// The new trajectory, which begins at `at`.
        trajectory: Trajectory,
    },
    /// No way around in time, or none sought without [`FollowerOptions::replan`]: stopping on its
    /// path, the stop clear.
    Stopping {
        /// What the rest of the motion collided with where it was first found to, which the stop
        /// is for, as [`Device::contacts`] lists it: nearest first, its obstacles numbered as in
        /// the worlds given to the tick the stop began at. The same at every tick of the stop
        /// while the stop is clear. Where a GPU device's check finds a graze its CPU threads put
        /// a hair outside, the nearest contact, less than 0.1 mm away.
        contacts: Vec<Contact>,
    },
    /// The motion it is on collides at `at` (seconds, on the caller's clock) and no stop within
    /// the limits ends short of it: stop the robot some other way.
    Collides {
        /// When the robot collides.
        at: f32,
        /// What it collides with then, as [`Device::contacts`] lists it, its obstacles numbered as
        /// in the worlds given to this tick; at a graze, as for [`FollowStatus::Stopping`].
        contacts: Vec<Contact>,
    },
    /// At the goal, at rest.
    Arrived,
    /// At rest short of the goal after a stop: plan again from here, or give up. A follower that
    /// cannot or does not plan from a measured state past its tolerance holds the robot there,
    /// blocked.
    Blocked,
    /// The measured state was farther from the trajectory than `FollowerOptions::tolerance`, by
    /// `off` at its farthest joint, as an arm held back or pushed: a new trajectory from the
    /// measured state takes over now.
    Reanchored {
        /// How far the farthest joint was from the trajectory (radians or meters).
        off: f32,
    },
}

/// The state to command at a tick, and what the follower did.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Tick {
    /// Position, velocity and acceleration to command.
    pub state: JointState,
    /// What happened at this tick.
    pub status: FollowStatus,
}

/// Follows a trajectory and replans it as the world changes.
#[derive(Clone, Debug)]
pub struct Follower {
    robot: Robot,
    world: u32,
    goal: Vec<f32>,
    trajectory: Trajectory,
    began: f32,
    /// A replan and the time it takes over.
    next: Option<(Trajectory, f32)>,
    stopping: bool,
    /// What the stop it is on is for.
    stopped_for: Vec<Contact>,
    o: FollowerOptions,
}

/// How far apart a GPU device's clearances and its CPU threads' may be (meters), as the parity
/// tests hold them.
const GRAZE: f32 = 1e-4;

/// Where the robot first collides along the rest of its motion: when, on the caller's clock, and
/// what with.
struct Hit {
    at: f32,
    contacts: Vec<Contact>,
}

/// A stretch of a trajectory to check: from `from` to `to` on its own clock, which began at
/// `began` on the caller's.
struct Stretch<'a> {
    trajectory: &'a Trajectory,
    began: f32,
    from: f32,
    to: f32,
}

impl Follower {
    /// Follows `trajectory` in world `world` of the worlds each tick is given, from time `now`,
    /// toward where it ends. Plan options that [`crate::plan`] refuses are refused here, before a
    /// replan would fail with the robot moving, and so, where it replans, are constraints the
    /// trajectory's goal breaks.
    pub fn try_new(
        robot: &Robot,
        world: u32,
        trajectory: Trajectory,
        now: f32,
        o: FollowerOptions,
    ) -> Result<Self> {
        ensure_input!(
            trajectory.dof == robot.dof(),
            "a trajectory of {} joints for a robot of {}",
            trajectory.dof,
            robot.dof()
        );
        ensure_input!(now.is_finite(), "a finite start time, got {now}");
        ensure_input!(
            o.budget.is_finite()
                && o.budget >= 0.0
                && o.lead.is_finite()
                && o.lead >= 0.0
                && o.check_every >= 1e-3
                && o.check_every.is_finite(),
            "a follower needs a finite budget and lead of at least 0 and finite checks at least 1 ms apart, got {} s, {} s and {} s",
            o.budget,
            o.lead,
            o.check_every
        );
        ensure_input!(
            o.tolerance > 0.0,
            "a follower needs a positive tolerance, got {}",
            o.tolerance
        );
        o.plan.check()?;
        constraint::check(robot, &o.constraints, "the follower")?;
        let points = &trajectory.control_points;
        let goal = points[points.len() - trajectory.dof..].to_vec();
        // A replan's goal must keep them, or planning refuses it with the robot moving.
        if o.replan
            && let Some(why) = constraint::breach(robot, &o.constraints, &goal)
        {
            return Err(crate::error::input!(
                "the trajectory's goal breaks the follower's constraints: {why}"
            ));
        }
        Ok(Self {
            robot: robot.clone(),
            world,
            goal,
            trajectory,
            began: now,
            next: None,
            stopping: false,
            stopped_for: vec![],
            o,
        })
    }

    /// The trajectory it follows now.
    #[must_use]
    pub fn trajectory(&self) -> &Trajectory {
        &self.trajectory
    }

    /// When the trajectory it follows now began (seconds, on the caller's clock).
    #[must_use]
    pub fn began(&self) -> f32 {
        self.began
    }

    /// The state to command at `now`, given the state the robot was `measured` in, if known.
    /// First, when the measured state is farther from the trajectory than the tolerance, a new
    /// trajectory from it ([`FollowStatus::Reanchored`]), or a hold there where it does not
    /// replan; then, when the rest of the motion collides in `worlds`, a replan from the state the
    /// robot will be in at `now + budget + lead`, or a controlled stop when none is found in time,
    /// the collision comes sooner or it does not replan. A tick that replans takes as long as the
    /// plan does, up to `budget`.
    pub fn tick(
        &mut self,
        device: &Device,
        worlds: &Worlds,
        now: f32,
        measured: Option<&JointState>,
    ) -> Result<Tick> {
        ensure_input!(now.is_finite(), "a tick at a finite time, got {now}");
        let n = self.robot.dof();
        if let Some(m) = measured {
            let finite = |v: &[f32]| v.len() == n && v.iter().all(|x| x.is_finite());
            ensure_input!(
                finite(&m.position) && finite(&m.velocity),
                "a measured state needs {n} finite positions and velocities"
            );
        }
        if self.next.as_ref().is_some_and(|(_, at)| now >= *at) {
            let (trajectory, at) = self.next.take().expect("a replan to take over");
            self.trajectory = trajectory;
            self.began = at;
        }
        let duration = self.trajectory.duration();
        let t = (now - self.began).max(0.0);
        let mut state = JointState::new(n);
        self.trajectory.at(t.min(duration), &mut state);
        if let Some(m) = measured {
            let off = (0..n)
                .map(|j| {
                    let d = m.position[j] - state.position[j];
                    if self.robot.continuous()[j] {
                        (d + PI).rem_euclid(TAU) - PI
                    } else {
                        d
                    }
                    .abs()
                })
                .fold(0.0, f32::max);
            if off > self.o.tolerance {
                return self.reanchor(device, worlds, now, m, off);
            }
        }
        if t >= duration {
            let status = if self.stopping {
                FollowStatus::Blocked
            } else {
                FollowStatus::Arrived
            };
            return Ok(Tick { state, status });
        }
        // The rest of the motion: this trajectory up to a pending replan's takeover, then the
        // replan.
        let mut rest = vec![Stretch {
            trajectory: &self.trajectory,
            began: self.began,
            from: t,
            to: duration,
        }];
        if let Some((next, at)) = &self.next {
            rest[0].to = (at - self.began).min(duration);
            rest.push(Stretch {
                trajectory: next,
                began: *at,
                from: 0.0,
                to: next.duration(),
            });
        }
        let hit = self.first_collision(device, worlds, &rest)?;
        let Some(hit) = hit else {
            let status = if self.stopping {
                FollowStatus::Stopping {
                    contacts: self.stopped_for.clone(),
                }
            } else {
                FollowStatus::Following
            };
            return Ok(Tick { state, status });
        };
        if self.stopping {
            return Ok(Tick {
                state,
                status: hit.collides(),
            });
        }
        // A pending replan that collides is dropped for a stop: another would start after the
        // takeover, on the stretch of this trajectory that was found to collide.
        let pending = self.next.take().is_some();
        let start = t + self.o.budget + self.o.lead;
        let mut from = JointState::new(n);
        self.trajectory.at(start.min(duration), &mut from);
        // Between the states its plan was checked at, the trajectory may stray past a bound.
        let keeps =
            || constraint::breach(&self.robot, &self.o.constraints, &from.position).is_none();
        if self.o.replan && !pending && self.began + start < hit.at && start < duration && keeps() {
            let motion = StartMotion {
                velocity: from.velocity.clone(),
                acceleration: from.acceleration.clone(),
            };
            let problem = PlanProblem {
                world: self.world,
                start: from.position.clone(),
                goal: self.goal.clone(),
                start_motion: Some(motion.clone()),
                hold: vec![],
                constraints: self.o.constraints.clone(),
                holding: None,
            };
            let options = PlanOptions {
                time_budget: Some(Duration::from_secs_f32(self.o.budget.max(1e-3))),
                ..self.o.plan
            };
            let planned = crate::plan(device, worlds, &[problem], &options)?;
            if let Some(path) = planned.best(0)
                && let Ok(next) = Trajectory::moving(&self.robot, path, &motion)
            {
                let at = self.began + start;
                self.next = Some((next.clone(), at));
                return Ok(Tick {
                    state,
                    status: FollowStatus::Replanned {
                        at,
                        trajectory: next,
                    },
                });
            }
        }
        // No way around in time, or none wanted: stop on the path. A path that ends sooner than any
        // stop would runs to its end, the quickest way to rest.
        let Ok(stop) = self.trajectory.stop(&self.robot, t) else {
            return Ok(Tick {
                state,
                status: hit.collides(),
            });
        };
        self.trajectory = stop;
        self.began = now;
        self.stopping = true;
        self.stopped_for = hit.contacts;
        self.trajectory.at(0.0, &mut state);
        let rest = [Stretch {
            trajectory: &self.trajectory,
            began: now,
            from: 0.0,
            to: self.trajectory.duration(),
        }];
        let status = match self.first_collision(device, worlds, &rest)? {
            Some(hit) => hit.collides(),
            None => FollowStatus::Stopping {
                contacts: self.stopped_for.clone(),
            },
        };
        Ok(Tick { state, status })
    }

    /// A new trajectory to the goal from the measured state `m`, taking over now: from its
    /// position (within the joint ranges) and its velocity (within the limits), planned within the
    /// budget. Where none is found, `m` breaks the constraints or the follower does not replan, the
    /// robot is held at rest where it is, blocked.
    fn reanchor(
        &mut self,
        device: &Device,
        worlds: &Worlds,
        now: f32,
        m: &JointState,
        off: f32,
    ) -> Result<Tick> {
        let n = self.robot.dof();
        let position: Vec<f32> = (0..n)
            .map(|j| {
                let (lo, hi) = self.robot.bounds(j);
                m.position[j].clamp(lo, hi)
            })
            .collect();
        let velocity: Vec<f32> = (0..n)
            .map(|j| {
                let most = self.robot.max_velocity()[j];
                m.velocity[j].clamp(-most, most)
            })
            .collect();
        let moving = velocity.iter().any(|&v| v != 0.0);
        let motion = StartMotion {
            velocity,
            acceleration: vec![0.0; n],
        };
        self.next = None;
        self.began = now;
        let keeps = constraint::breach(&self.robot, &self.o.constraints, &position).is_none();
        let planned = if self.o.replan && keeps {
            let problem = PlanProblem {
                world: self.world,
                start: position.clone(),
                goal: self.goal.clone(),
                start_motion: moving.then(|| motion.clone()),
                hold: vec![],
                constraints: self.o.constraints.clone(),
                holding: None,
            };
            let options = PlanOptions {
                time_budget: Some(Duration::from_secs_f32(self.o.budget.max(1e-3))),
                ..self.o.plan
            };
            let result = crate::plan(device, worlds, &[problem], &options)?;
            result.best(0).and_then(|path| {
                let timed = if moving {
                    Trajectory::moving(&self.robot, path, &motion)
                } else {
                    Trajectory::try_new(&self.robot, path, 1.0)
                };
                timed.ok()
            })
        } else {
            None
        };
        let mut state = JointState::new(n);
        if let Some(trajectory) = planned {
            self.trajectory = trajectory;
            self.stopping = false;
            self.trajectory.at(0.0, &mut state);
            return Ok(Tick {
                state,
                status: FollowStatus::Reanchored { off },
            });
        }
        // Held where it is: a motion that does not move, at the measured position.
        let still: Vec<f32> = position.iter().copied().cycle().take(7 * n).collect();
        self.trajectory = Trajectory::try_new(&self.robot, &still, 1.0)?;
        self.stopping = true;
        state.position = position;
        Ok(Tick {
            state,
            status: FollowStatus::Blocked,
        })
    }

    /// Where the robot first collides along `stretches`, each sampled `check_every` apart and at
    /// its end, in one batch.
    fn first_collision(
        &self,
        device: &Device,
        worlds: &Worlds,
        stretches: &[Stretch<'_>],
    ) -> Result<Option<Hit>> {
        let mut times = vec![];
        let mut q = vec![];
        let mut sample = JointState::new(self.robot.dof());
        for stretch in stretches {
            for k in 0.. {
                let at = (stretch.from + k as f32 * self.o.check_every).min(stretch.to);
                stretch.trajectory.at(at, &mut sample);
                q.extend_from_slice(&sample.position);
                times.push(stretch.began + at);
                if at >= stretch.to {
                    break;
                }
            }
        }
        let items = vec![self.world; times.len()];
        let e = device.evaluate(worlds, &items, &q, &CollisionWeights::NONE)?;
        let Some(i) = (0..times.len()).find(|&i| !e.collision_free(i)) else {
            return Ok(None);
        };
        // A GPU device's check may find a graze its CPU threads' contacts put a hair outside: the
        // nearest then, from as far out as the two devices' clearances may differ.
        let n = self.robot.dof();
        let mut near = device.contacts(worlds, &[self.world], &q[i * n..(i + 1) * n], GRAZE)?;
        let mut contacts = near.remove(0);
        let colliding = contacts.iter().take_while(|c| c.distance() <= 0.0).count();
        contacts.truncate(colliding.max(1));
        Ok(Some(Hit {
            at: times[i],
            contacts,
        }))
    }
}

impl Hit {
    fn collides(self) -> FollowStatus {
        FollowStatus::Collides {
            at: self.at,
            contacts: self.contacts,
        }
    }
}

/// How [`servo`] moves the hand.
#[derive(Clone, Debug)]
pub struct ServoOptions {
    /// The hand that moves.
    pub hand: usize,
    /// The control period (seconds): 10 ms.
    pub dt: f32,
    /// Gain from the pose's error to the hand's velocity (1/s): 4.
    pub gain: f32,
    /// Caps on the hand's speed (m/s) and on how fast it turns (rad/s): 0.1 and 0.5.
    pub linear: f32,
    /// See `linear`.
    pub angular: f32,
    /// Clearance where it would come to rest, braking from the state it commands, below which it
    /// slows (meters), and at which it brakes to rest instead: 2 cm and 2 mm, as MoveIt Servo's
    /// scene thresholds.
    pub slow_within: f32,
    /// See `slow_within`.
    pub halt_within: f32,
    /// Damping of the least-squares solve near singularities: 0.05.
    pub damping: f32,
}

impl Default for ServoOptions {
    fn default() -> Self {
        Self {
            hand: 0,
            dt: 0.01,
            gain: 4.0,
            linear: 0.1,
            angular: 0.5,
            slow_within: 0.02,
            halt_within: 0.002,
            damping: 0.05,
        }
    }
}

/// What a servo step did.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ServoStatus {
    /// Moving toward the target at the speed the gain and caps allow.
    Normal,
    /// Slowed because braking would bring it near an obstacle (`clearance` where it would rest,
    /// meters).
    Slowing {
        /// The clearance it slowed for.
        clearance: f32,
    },
    /// Braking to rest short of an obstacle (`clearance` along the way, meters). At zero or below,
    /// no stop within the limits ends short of contact.
    Halted {
        /// The least clearance while it brakes.
        clearance: f32,
    },
    /// At the target (within a millimeter and a hundredth of a radian) and nearly at rest.
    Arrived,
}

/// One servo step: the state to command next, what the step did, and how far the hand is from
/// the target (meters).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ServoStep {
    /// Position, velocity and acceleration to command for the next `dt`.
    pub state: JointState,
    /// What the step did.
    pub status: ServoStatus,
    /// The hand's distance from the target's position before the step (meters).
    pub error: f32,
}

/// Configurations checked along each braking path: from the state commanded to where it rests.
const BRAKE_SAMPLES: usize = 4;

/// One servo step from `state` toward the hand at `target`, in world `world` of `worlds`: the
/// joint state that best follows the hand velocity the pose's error asks for, within the joints'
/// velocity, acceleration and jerk limits. It is slowed where braking from that state would bring
/// the robot near an obstacle and braked to rest where it would bring it within `halt_within`,
/// and each joint brakes in time to rest within its range.
pub fn servo(
    device: &Device,
    worlds: &Worlds,
    world: u32,
    state: &JointState,
    target: Pose,
    o: &ServoOptions,
) -> Result<ServoStep> {
    let robot = device.robot();
    let n = robot.dof();
    ensure_input!(
        state.position.len() == n && state.velocity.len() == n && state.acceleration.len() == n,
        "a state of {n} joints, got {} positions, {} velocities and {} accelerations",
        state.position.len(),
        state.velocity.len(),
        state.acceleration.len()
    );
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
    ensure_input!(
        finite(&state.position) && finite(&state.velocity) && finite(&state.acceleration),
        "a finite state, got {state:?}"
    );
    ensure_input!(
        target.position.is_finite() && target.rotation.is_finite(),
        "a finite target, got {target:?}"
    );
    ensure_input!(
        o.hand < robot.hands(),
        "hand {} of {}",
        o.hand,
        robot.hands()
    );
    ensure_input!(
        o.dt > 0.0 && o.dt.is_finite() && o.gain > 0.0 && o.gain.is_finite(),
        "servo needs a finite positive period and gain, got {} s and {}",
        o.dt,
        o.gain
    );
    ensure_input!(
        o.linear > 0.0 && o.angular > 0.0 && o.damping >= 0.0 && o.damping.is_finite(),
        "servo needs positive speed caps and a finite damping of at least 0, got {} m/s, {} rad/s \
         and {}",
        o.linear,
        o.angular,
        o.damping
    );
    let q = &state.position;
    let pose = robot.ee_pose(q, o.hand);
    let dp = target.position - pose.position;
    let turn = target.rotation * pose.rotation.inverse();
    // The shorter way round: q and -q are the same rotation.
    let dr = if turn.w < 0.0 { -turn } else { turn }.to_scaled_axis();
    let error = dp.length();
    let speed: f32 = state.velocity.iter().map(|v| v.abs()).fold(0.0, f32::max);
    // The hand's velocity: proportional to the error, within the caps.
    let cap = |v: Vec3, most: f32| {
        if v.length() > most {
            v * (most / v.length())
        } else {
            v
        }
    };
    let (linear, angular) = (cap(dp * o.gain, o.linear), cap(dr * o.gain, o.angular));
    let qdot = if error < 1e-3 && dr.length() < 1e-2 {
        vec![0.0; n]
    } else {
        let twist = [
            linear.x, linear.y, linear.z, angular.x, angular.y, angular.z,
        ];
        Reach::new(robot)
            .velocity(robot, o.hand, q, twist, o.damping)
            .to_vec()
    };
    // The state it commands at full speed and the one braking now, each checked along its braking
    // path, with the state it starts in, in one batch.
    let full = command(robot, state, &qdot, o.dt);
    let brake = command(robot, state, &vec![0.0; n], o.dt);
    let mut configs = q.clone();
    for s in [&full, &brake] {
        configs.extend(braking_path(robot, s, o.dt));
    }
    let items = vec![world; configs.len() / n];
    // Clearances are exact up to the margin: as far out as it slows for.
    let exact = CollisionWeights {
        margin: o.slow_within,
        ..CollisionWeights::NONE
    };
    let e = device.evaluate(worlds, &items, &configs, &exact)?;
    let least = |c: &[f32]| c.iter().copied().fold(f32::INFINITY, f32::min);
    let (now, self_now) = (e.world_clearance[0], e.self_clearance[0]);
    let paths = [
        1..1 + BRAKE_SAMPLES,
        1 + BRAKE_SAMPLES..1 + 2 * BRAKE_SAMPLES,
    ];
    let [ahead, braking] = paths.clone().map(|p| least(&e.world_clearance[p]));
    let [self_ahead, self_braking] = paths.map(|p| least(&e.self_clearance[p]));
    // The robot's own links only halt it short of touching, as they sit close by design.
    let self_contact = self_ahead < self_now && self_ahead <= 0.0;
    let (out, mut status) = if (ahead < now && ahead <= o.halt_within) || self_contact {
        let clearance = braking.min(self_braking);
        (brake, ServoStatus::Halted { clearance })
    } else if ahead < now && ahead < o.slow_within {
        // As fast as an even deceleration to rest at the halt clearance allows, so it gets there.
        let scale = ((ahead - o.halt_within) / (o.slow_within - o.halt_within)).sqrt();
        let slowed: Vec<f32> = qdot.iter().map(|v| v * scale).collect();
        let out = command(robot, state, &slowed, o.dt);
        (out, ServoStatus::Slowing { clearance: ahead })
    } else {
        (full, ServoStatus::Normal)
    };
    if error < 1e-3 && dr.length() < 1e-2 && speed < 1e-3 {
        status = ServoStatus::Arrived;
    }
    Ok(ServoStep {
        state: out,
        status,
        error,
    })
}

/// The state one period `dt` on from `state` that tracks the joint velocities `qdot`: within the
/// velocity, acceleration and jerk limits, reaching each velocity without overshooting it, and
/// braking a joint instead where braking any later would carry it past the end of its range.
fn command(robot: &Robot, state: &JointState, qdot: &[f32], dt: f32) -> JointState {
    let (vmax, amax, jmax) = (
        robot.max_velocity(),
        robot.max_acceleration(),
        robot.max_jerk(),
    );
    let mut out = JointState::new(robot.dof());
    for j in 0..robot.dof() {
        let (v0, a0) = (state.velocity[j], state.acceleration[j]);
        let step = |wanted: f32| {
            // The acceleration that reaches the wanted velocity without overshooting it: easing
            // it to zero at the jerk limit gains about a²/2j + a dt/2 more.
            let dv = wanted - v0;
            let settle = jmax[j] * ((dt * dt / 4.0 + 2.0 * dv.abs() / jmax[j]).sqrt() - dt / 2.0);
            let a = (dv.signum() * settle.min(dv.abs() / dt))
                .clamp(-amax[j], amax[j])
                .clamp(a0 - jmax[j] * dt, a0 + jmax[j] * dt);
            let v = v0 + a * dt;
            (state.position[j] + v * dt, v, a)
        };
        let (lo, hi) = robot.bounds(j);
        let (mut p, mut v, mut a) = step(qdot[j].clamp(-vmax[j], vmax[j]));
        let rest = p + v.signum() * braking_distance(v, a, amax[j], jmax[j], dt);
        if !(lo..=hi).contains(&rest) {
            (p, v, a) = step(0.0);
        }
        out.position[j] = p.clamp(lo, hi);
        out.velocity[j] = if p < lo || p > hi { 0.0 } else { v };
        out.acceleration[j] = a;
    }
    out
}

/// The configurations from `state` to where it rests braking at the limits, `BRAKE_SAMPLES` of
/// them evenly spaced, flattened.
fn braking_path(robot: &Robot, state: &JointState, dt: f32) -> Vec<f32> {
    let (amax, jmax) = (robot.max_acceleration(), robot.max_jerk());
    let travel: Vec<f32> = (0..robot.dof())
        .map(|j| {
            let v = state.velocity[j];
            v.signum() * braking_distance(v, state.acceleration[j], amax[j], jmax[j], dt)
        })
        .collect();
    let mut path = Vec::with_capacity(BRAKE_SAMPLES * robot.dof());
    for k in 0..BRAKE_SAMPLES {
        let f = k as f32 / (BRAKE_SAMPLES - 1) as f32;
        for (j, travel) in travel.iter().enumerate() {
            let (lo, hi) = robot.bounds(j);
            path.push((state.position[j] + f * travel).clamp(lo, hi));
        }
    }
    path
}

/// At most how far a joint moving at `v` with acceleration `a` travels before it rests, braking
/// from one period `dt` on within acceleration `amax` and jerk `jmax`.
fn braking_distance(v: f32, a: f32, amax: f32, jmax: f32, dt: f32) -> f32 {
    let (v, a) = if v < 0.0 { (-v, -a) } else { (v, a) };
    // An acceleration that still speeds it up eases to zero first.
    let push = a.max(0.0);
    let v1 = v + push * push / (2.0 * jmax);
    // Then the deceleration, its jerk limited, is symmetric about its middle: it covers v1 T / 2.
    let brake = if v1 * jmax >= amax * amax {
        v1 / amax + amax / jmax
    } else {
        2.0 * (v1 / jmax).sqrt()
    };
    v1 * dt + v1 * push / jmax + v1 * brake / 2.0
}
