//! Whole tasks in one call: a hand picking up an object ([`pick`]), setting down the one it holds
//! ([`place`]), turning one about a line ([`turn`]) or drawing one along a line ([`pull`]). Each is
//! three motions from rest to rest: a planned path from the start to where the hand begins to move
//! in, a straight move in (to the grasp, or down to where the object goes), and a straight move
//! away (lifting the object, or backing off from it). The gripper closes or opens between the
//! second and the third; a turn's hand closes, turns the object, and opens between them, and a
//! pull's draws it along its line. Every motion is checked with the fingers where the robot was
//! loaded with them, so open them first, unless the task says otherwise: a pick's, a turn's or a
//! pull's grasps may keep them narrower ([`Pick::locked`], [`Turn::locked`], [`Pull::locked`]), a
//! hold says where they closed ([`Held::locked`]), and a place may name how far they open where it
//! lets go ([`Place::open`]).
//!
//! A task's candidates, its grasps or the poses it may set the object at, run with every other
//! task's, step by step, each step a batch for those whose fingers and held object are alike: IK
//! at each grasp or release, the moves in (run backward from there, so they end exactly there),
//! the moves away, then planned paths for those still going. Every candidate that passes every step
//! comes back, quickest first. When none does, each candidate's step and reason come back instead.
//!
//! A task may leave its transit out (`transit: false`) and give the two straight moves alone: a
//! program of many tasks solves all their moves in one call, plans the ways between them in one
//! [`plan`](crate::plan) and joins them ([`join`](crate::join())). Each step may run in a world of
//! its own (`transit_world`, `depart_world`), so that what the hand may touch at the grasp is not
//! what it may touch on the way there.

use std::borrow::Cow;
use std::collections::HashMap;
use std::f32::consts::TAU;
use std::fmt;
use std::ops::Range;

use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

use crate::constraint::{self, HandConstraint};
use crate::device::{Contact, Device, Worlds};
use crate::edges::{self, Deferred, EdgeQuery, Search, Wanted};
use crate::error::{Result, ensure_input, input};
use crate::held::{self, Fitted, Held};
use crate::ik::{self, IkFailure, IkOptions, IkProblem, IkResult};
use crate::robot::Robot;
use crate::spheres::SphereOptions;
use crate::timing::{HandSpeed, Near, Trajectory};
use crate::tool_path::{self, ToolPath, ToolPathOptions, ToolPathResult, ToolPathStop};
use crate::trajopt::{self, PlanOptions, PlanProblem};
use crate::types::{Pose, Screw};
use crate::world::{Obstacle, Touch, World};

/// The body lifted when the hand grasps one part of a larger object.
#[derive(Clone, Debug, PartialEq)]
pub struct PickBody {
    /// Collision shape of the entire body at its initial pose in the world. A distance grid
    /// cannot be attached. The returned hold uses this shape's frame.
    pub shape: Obstacle,
    /// World obstacles that leave with the grasped part. They remain collidable on approach.
    /// Give each index once, excluding the grasped part and supports.
    pub other_parts: Vec<usize>,
}

/// A hand picking up one obstacle of its world.
#[derive(Clone, Debug)]
pub struct Pick {
    /// The world's index in the batch: the one the hand grasps and moves in in, and every other
    /// step's unless that names its own.
    pub world: u32,
    /// The world the start is checked and the transit planned in, where it is another: as many
    /// obstacles as `world` has, in its order, shaped or touched ([`World::touch`]) otherwise.
    pub transit_world: Option<u32>,
    /// The world the object is lifted in, where it is another, likewise.
    pub depart_world: Option<u32>,
    /// Where the robot starts, at rest.
    pub start: Vec<f32>,
    /// Whether the path from the start to where the move in begins is planned: true. Without it
    /// the plans hold the two straight moves alone ([`TaskPlan::begin`] is where a way to them
    /// ends), and the start, which is not checked for collision, only says which solution of a
    /// grasp is nearest and where the joints other hands move wait.
    pub transit: bool,
    /// The hand that picks it up.
    pub hand: usize,
    /// The obstacle the fingers may touch on approach. Grasps are in this obstacle's frame.
    /// A grid can be gripped when `body` supplies the shape carried on departure.
    pub object: usize,
    /// What moves when `object` is only the contact part. None lifts that obstacle alone.
    pub body: Option<PickBody>,
    /// Poses of the hand's IK frame that grasp the object, in the object's frame (its center and
    /// rotation); any one will do. [`grasps`] proposes some for a box, ball, cylinder or capsule.
    pub grasps: Vec<Pose>,
    /// How far back along its own z axis the hand begins to move in to a grasp (meters).
    pub approach: f32,
    /// Where the hand takes the object once it holds it, in the world's frame (meters).
    pub lift: Vec3,
    /// Obstacles the object rests on, by index: it may touch them as it leaves.
    pub support: Vec<usize>,
    /// Where the hands must stay along the transit ([`HandConstraint`]), as MTC's Connect stage
    /// takes path constraints. The start must keep them; a grasp whose move in begins where they
    /// break stops at the transit. A task without its transit has no use for them.
    pub constraints: Vec<HandConstraint>,
    /// Per grasp, joints the robot was loaded with locked, at the values they keep through the
    /// task by that grasp, as [`Robot::relock`] takes them: the fingers opened only as far as the
    /// grasp has room for, as a rim between walls lets them
    /// ([`Grasp::opening`](crate::Grasp::opening)). A joint that carries a hand (a torso) is
    /// refused. A call's grasps are grouped by these, a relocked robot for each group, and each
    /// plan's [`TaskPlan::held`] starts with its grasp's (overwrite them with where the fingers
    /// closed). Empty: every grasp as loaded, as is a grasp whose own are empty.
    pub locked: Vec<HashMap<String, f32>>,
    /// How many plans will do: `None`, every grasp is tried. With a count, the grasps are tried
    /// in their order, 32 of them and then twice as many each time, until that many have got
    /// through, and the task's plans are the first that many in the grasps' order to get
    /// through, quickest first: those that trying every grasp gives, without the IK and the
    /// motions of the grasps after them (on a GPU, whose IK rounds by the size of its batch,
    /// to that rounding). Put the grasps worth having first, as
    /// [`grasps_on`](crate::grasps_on) gives them. A task none gets through has tried every
    /// one, and says where each stopped.
    pub enough: Option<usize>,
}

impl Pick {
    /// The world for executing the lift with the returned hold attached. Pass the original
    /// departure world of this pick. Obstacles keep their order after the moving parts leave,
    /// and the held link may touch the listed supports. Invalid obstacle indices give
    /// an input error.
    pub fn lift_world(&self, departure: &World) -> Result<World> {
        self.check_parts(departure, format_args!("pick"))?;
        for &support in &self.support {
            ensure_input!(
                support < departure.obstacles.len() && support != self.object,
                "pick support {support} must be another of the world's {} obstacles",
                departure.obstacles.len()
            );
        }
        self.lift_in(departure, &self.parts(), self.hand)
    }

    fn lift_in(&self, departure: &World, parts: &[usize], hand: usize) -> Result<World> {
        let mut world = departure.clone();
        for &part in parts.iter().rev() {
            world.remove(part)?;
        }
        if self.hand == hand {
            let shifted = |s: usize| s - parts.partition_point(|&part| part < s);
            world
                .touch
                .extend(self.support.iter().map(|&s| Touch::held(shifted(s), hand)));
        }
        Ok(world)
    }

    fn check_parts(&self, world: &World, what: fmt::Arguments<'_>) -> Result<()> {
        ensure_input!(
            self.object < world.obstacles.len(),
            "{what}: obstacle {} of {}",
            self.object,
            world.obstacles.len()
        );
        if let Some(body) = &self.body {
            for (k, &part) in body.other_parts.iter().enumerate() {
                ensure_input!(
                    part < world.obstacles.len()
                        && part != self.object
                        && !body.other_parts[..k].contains(&part)
                        && !self.support.contains(&part),
                    "{what}: body part {part} must be a distinct obstacle among {}, separate from the contact part {} and supports {:?}",
                    world.obstacles.len(),
                    self.object,
                    self.support
                );
            }
        }
        Ok(())
    }

    /// The indices removed on lift, in increasing order. An ordinary pick borrows its one index.
    fn parts(&self) -> Cow<'_, [usize]> {
        let Some(body) = self
            .body
            .as_ref()
            .filter(|body| !body.other_parts.is_empty())
        else {
            return Cow::Borrowed(std::slice::from_ref(&self.object));
        };
        let mut parts = body.other_parts.clone();
        parts.push(self.object);
        parts.sort_unstable();
        Cow::Owned(parts)
    }

    /// Hand 0 picks up `object` of world `world` from `start` by any of `grasps`, moving in 10 cm
    /// along the hand's z axis and lifting the object 10 cm, off nothing it rests on.
    #[must_use]
    pub fn new(world: u32, start: impl Into<Vec<f32>>, object: usize, grasps: Vec<Pose>) -> Self {
        Self {
            world,
            transit_world: None,
            depart_world: None,
            start: start.into(),
            transit: true,
            hand: 0,
            object,
            body: None,
            grasps,
            approach: 0.1,
            lift: Vec3::Z * 0.1,
            support: vec![],
            constraints: vec![],
            locked: vec![],
            enough: None,
        }
    }
}

/// A hand setting down the object it holds.
#[derive(Clone, Debug)]
pub struct Place {
    /// The world's index in the batch: the one the hand moves in in to where it lets go, and
    /// every other step's unless that names its own. The held object is not one of its obstacles.
    pub world: u32,
    /// The world the start is checked and the transit planned in, where it is another: as many
    /// obstacles as `world` has, in its order, shaped or touched ([`World::touch`]) otherwise.
    pub transit_world: Option<u32>,
    /// The world the hand backs off in, where it is another, likewise.
    pub depart_world: Option<u32>,
    /// Where the robot starts, at rest, holding the object.
    pub start: Vec<f32>,
    /// Whether the path from the start to where the move in begins is planned: true. Without it
    /// the plans hold the two straight moves alone ([`TaskPlan::begin`] is where a way to them
    /// ends), and the start, which is not checked for collision, only says which solution of a
    /// release is nearest and where the joints other hands move wait, unless `idle` says so.
    pub transit: bool,
    /// Where the joints only other hands move go on the transit, as a configuration of which only
    /// those joints count: another arm going home while this one carries the object to where it
    /// goes. They wait there through the moves in and away. None keeps them where `start` has
    /// them.
    pub idle: Option<Vec<f32>>,
    /// What the hand holds, and how: a pick's [`TaskPlan::held`].
    pub held: Held,
    /// Poses of the object's frame to set it down at; any one will do. [`place_poses`] gives an
    /// object's symmetric turns at one spot, and [`rests_on`](crate::rests_on) the poses it
    /// rests at on a support cameras see.
    pub poses: Vec<Pose>,
    /// Expected downward travel after release, in meters along world −z, one value per pose.
    /// Empty sets down at the release itself. Supply a measured displacement to a resting
    /// pose at the same attitude. Departure checks an enclosure of
    /// the object throughout that translation. The returned placed pose is the expected rest;
    /// the caller verifies where the object actually settled.
    pub falls: Vec<f32>,
    /// How the hand moves in to where it lets go, in the world's frame (meters).
    pub lower: Vec3,
    /// How far back along its own z axis the hand moves once it lets go (meters).
    pub retreat: f32,
    /// Values the fingers may open to where the hand lets go, each as [`Held::locked`] names
    /// them, in the order preferred: a pose lets go at the first with which the hand, where it
    /// lets go and on its move away, is clear, and its plan says which ([`TaskPlan::opening`]).
    /// The release is then solved with the fingers where they hold the object, not where the
    /// robot was loaded with them, so a hand can set an object down where it has no room to open
    /// all the way. Joints an opening does not name are where the robot was loaded with them,
    /// whatever the hold has them at. An opening is tried only where the fingers stopped the one
    /// before; a pose none clears stops at its departure, with what the last one tried met.
    /// Empty: the hand opens to where the robot was loaded with them.
    pub open: Vec<HashMap<String, f32>>,
    /// How many plans will do: `None`, every pose is tried. With a count, the poses are tried in
    /// their order, 32 of them and then twice as many each time, until that many have got
    /// through, and the task's plans are the first that many in the poses' order to get through,
    /// quickest first: those that trying every pose gives, without the IK and the motions of
    /// the poses after them (on a GPU, whose IK rounds by the size of its batch, to that
    /// rounding). Put the poses worth having first. A task none gets through has tried every
    /// one, and says where each stopped.
    pub enough: Option<usize>,
    /// Obstacles the object is set on, by index: it may touch them as it arrives.
    pub support: Vec<usize>,
    /// Where the hands must stay along the transit ([`HandConstraint`]), as a cup carried upright
    /// to where it goes. The start must keep them; a pose whose move in begins where they break
    /// stops at the transit. A task without its transit has no use for them.
    pub constraints: Vec<HandConstraint>,
}

impl Place {
    /// Sets `held` down from `start` at any of `poses` in world `world`, lowering it the last
    /// 10 cm and backing off 10 cm along the hand's z axis, onto nothing.
    #[must_use]
    pub fn new(world: u32, start: impl Into<Vec<f32>>, held: Held, poses: Vec<Pose>) -> Self {
        Self {
            world,
            transit_world: None,
            depart_world: None,
            start: start.into(),
            transit: true,
            idle: None,
            held,
            poses,
            falls: vec![],
            lower: Vec3::Z * -0.1,
            retreat: 0.1,
            open: vec![],
            enough: None,
            support: vec![],
            constraints: vec![],
        }
    }
}

/// How [`pick`], [`place`], [`handover`](crate::handover()) and [`turn`] solve, follow, plan
/// and time their steps.
#[derive(Clone, Debug)]
pub struct TaskOptions {
    /// Collision spheres for the carried object created by a pick or a moving turn: holding all of
    /// it, within the tolerance of it, however many spheres that takes ([`Held::spheres`]). A
    /// smaller tolerance fits closer to narrow walls; the budget does not cap them. The returned
    /// hold keeps the same options when passed to a place. Defaults to [`SphereOptions::default`].
    pub held_spheres: SphereOptions,
    /// IK at each grasp or release.
    pub ik: IkOptions,
    /// The planned paths from the start. `start_tolerance` applies to the task's initial
    /// configuration too, even when its transit is disabled; the result retains a corrected start.
    pub plan: PlanOptions,
    /// The straight moves, and a turn's, which follows it at half its precision: the other half
    /// is for how far the straight moves between the screw's waypoints sag from the arc.
    pub tool_path: ToolPathOptions,
    /// The hand's speed cap on the straight moves and on a turn (m/s): 10 cm/s. `f32::INFINITY`
    /// leaves it uncapped, the moves as fast as the joints' limits allow.
    pub tool_speed: f32,
    /// Where the cap holds on the straight moves (m): within this distance of the hand's position
    /// at the grasp or release, which the move in ends at and the move away starts from, the rest
    /// of each move as fast as the joints' limits allow ([`HandSpeed::near`]). `None`: along all
    /// of both. A turn, which the hand holds the object through, is capped all along whatever
    /// this says.
    pub tool_speed_near: Option<f32>,
    /// How much of each straight move a candidate must make, in (0, 1]: 1, all of it. Less lets
    /// an obstacle shorten the move in or away rather than stop the candidate; a move stopped
    /// short for any other reason (out of reach, a jump) still stops it.
    pub min_fraction: f32,
}

impl TaskOptions {
    /// Validates the common task options and the options of each underlying solver. This also
    /// checks stages a particular candidate may never reach.
    pub fn check(&self) -> Result<()> {
        check_options(self)?;
        self.ik.check()?;
        self.plan.check()?;
        self.tool_path.check()
    }
}

impl Default for TaskOptions {
    fn default() -> Self {
        Self {
            held_spheres: SphereOptions::default(),
            ik: IkOptions::default(),
            plan: PlanOptions::default(),
            tool_path: ToolPathOptions::default(),
            tool_speed: 0.1,
            tool_speed_near: None,
            min_fraction: 1.0,
        }
    }
}

/// A task's steps, in the order they are checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum TaskStep {
    /// The start, checked for collision. A task without its transit skips it.
    Start,
    /// IK at the grasp, or where the hand lets go.
    Reach,
    /// The turn of a [`turn`], the hand following the grasp about the screw from a configuration
    /// at the grasp: checked before the straight moves, which then run from the configurations
    /// that make it. A pick, a place or a handover skips it.
    Turn,
    /// The straight draw of a [`pull`], checked from each configuration at the grasp.
    Pull,
    /// The straight move in, to there.
    Approach,
    /// The straight move away: lifting the object, or backing off from it.
    Depart,
    /// The planned path from the start to where the move in begins. A task without its transit
    /// skips it.
    Transit,
}

const STEPS: [TaskStep; 6] = [
    TaskStep::Start,
    TaskStep::Reach,
    TaskStep::Turn,
    TaskStep::Approach,
    TaskStep::Depart,
    TaskStep::Transit,
];

/// A task's motions by one candidate.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TaskPlan {
    /// The candidate: its index among the task's grasps or poses.
    pub candidate: usize,
    /// The planned path from the start to where the move in begins; none for a task that leaves
    /// its transit out.
    pub transit: Option<Trajectory>,
    /// The straight move in. The gripper closes (pick, turn, pull) or opens (place) at its end; in
    /// a handover the taker's closes, then the giver's opens.
    pub approach: Trajectory,
    /// After a [`turn`], the hand turning the object, the fingers closed on it, from where the
    /// move in ends to where the move away starts. Open the gripper after it. None for the rest.
    pub turn: Option<Trajectory>,
    /// A [`pull`]'s straight draw from the grasp, with the fingers closed. Open them at its end.
    pub pull: Option<Box<Trajectory>>,
    /// The straight move away: holding the object after a pick; the giver's after a handover.
    pub depart: Trajectory,
    /// After a pick, what the hand holds; after a handover, what the taker holds.
    pub held: Option<Held>,
    /// After a place, the object where it was set down, to add to the world; after a turn, the
    /// object where the turn leaves it, to put in the world in its place, or where it stood where
    /// it stays ([`Turn::stays`]).
    pub placed: Option<Obstacle>,
    /// For a place with a positive fall, the released body as checked during departure.
    /// It encloses the release, the expected landing and the translation between them.
    /// Check the follower against this obstacle with the releasing fingers allowed to touch it.
    pub falling: Option<Obstacle>,
    /// After a place that says how far its hand may open ([`Place::open`]), which of those the
    /// hand lets go at and moves away with, by index.
    pub opening: Option<usize>,
    /// After a handover, the object's pose where it changed hands, in the robot's frame.
    pub meeting: Option<Pose>,
}

impl TaskPlan {
    /// Its motions in the order they run: the transit if it has one, the move in, the turn if it
    /// has one, or the draw for a pull, then the move away.
    pub fn motions(&self) -> impl Iterator<Item = &Trajectory> {
        let transit = self.transit.iter().chain([&self.approach]);
        transit
            .chain(self.turn.iter())
            .chain(self.pull.as_deref())
            .chain([&self.depart])
    }

    /// Seconds its motions take, one after another.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.motions().map(Trajectory::duration).sum()
    }

    /// Where the move in begins, its first control point: where the transit ends, or a way planned
    /// apart must.
    #[must_use]
    pub fn begin(&self) -> &[f32] {
        &self.approach.control_points[..self.approach.dof]
    }

    /// Where the departure comes to rest, its last control point: the start of the next task.
    #[must_use]
    pub fn end(&self) -> &[f32] {
        let points = &self.depart.control_points;
        &points[points.len() - self.depart.dof..]
    }
}

/// Where one candidate stopped, and why.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CandidateStop {
    /// The step it did not get past.
    pub step: TaskStep,
    /// What stopped it there: a pose out of reach, a collision or a jump at the start, at IK and
    /// at the straight moves; nothing for a transit no plan was found for.
    pub why: Option<ToolPathStop>,
    /// For a collision, what collides ([`Device::contacts`]), nearest first: at the start, at the
    /// IK solution with the most clearance that reaches the grasp or release, holding the object
    /// at the solution nearest the rest where every one collides holding it (a place), or at the
    /// straight move's next step. Empty for the rest. Obstacles are numbered as in the task's
    /// world; in a place, `world.obstacles.len()` is the object where it is set down, and in a
    /// handover, the object where it changes hands.
    pub contacts: Vec<Contact>,
    /// At [`TaskStep::Turn`]: how far the hand turned before it stopped (radians, signed as the
    /// task's angle), from the configuration at the grasp that turned it farthest, whose stop
    /// `why` and `contacts` are. None at every other step.
    pub turned: Option<f32>,
    /// At [`TaskStep::Pull`], the greatest distance planned along the draw before it stopped
    /// (meters, signed as the task's distance). Its reason and contacts come from that try.
    pub pulled: Option<f32>,
}

/// Why none of a task's candidates got through: each one's step and reason, in the task's order.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct TaskFailure {
    /// Per candidate: where it stopped.
    pub candidates: Vec<CandidateStop>,
}

impl TaskFailure {
    /// The furthest step a candidate got to.
    #[must_use]
    pub fn step(&self) -> TaskStep {
        self.candidates
            .iter()
            .map(|c| c.step)
            .max()
            .unwrap_or(TaskStep::Start)
    }

    /// How many candidates stopped at `step`.
    #[must_use]
    pub fn at(&self, step: TaskStep) -> usize {
        self.candidates.iter().filter(|c| c.step == step).count()
    }
}

impl fmt::Display for TaskFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.candidates.len();
        if self.at(TaskStep::Start) == n {
            return write!(f, "the start collides");
        }
        write!(f, "none of {n} candidates got through:")?;
        let mut first = true;
        for step in [
            TaskStep::Start,
            TaskStep::Reach,
            TaskStep::Turn,
            TaskStep::Pull,
            TaskStep::Approach,
            TaskStep::Depart,
            TaskStep::Transit,
        ] {
            let here: Vec<&CandidateStop> =
                self.candidates.iter().filter(|c| c.step == step).collect();
            if here.is_empty() {
                continue;
            }
            let doing = match step {
                TaskStep::Start => "starting in collision",
                TaskStep::Reach => "at the grasp or release",
                TaskStep::Turn => "turning",
                TaskStep::Pull => "pulling",
                TaskStep::Approach => "moving in",
                TaskStep::Depart => "moving away",
                TaskStep::Transit => "with no path from the start",
            };
            write!(f, "{} {} {doing}", if first { "" } else { "," }, here.len())?;
            first = false;
            let count = |kind: fn(&ToolPathStop) -> bool| {
                here.iter()
                    .filter(|c| c.why.as_ref().is_some_and(kind))
                    .count()
            };
            let reasons: Vec<String> = [
                (
                    count(|s| matches!(s, ToolPathStop::Unreachable { .. })),
                    "out of reach",
                ),
                (
                    count(|s| matches!(s, ToolPathStop::Collision { .. })),
                    "colliding",
                ),
                (count(|s| matches!(s, ToolPathStop::Jump { .. })), "jumping"),
            ]
            .into_iter()
            .filter(|&(k, _)| k > 0)
            .map(|(k, what)| format!("{k} {what}"))
            .collect();
            if !reasons.is_empty() {
                write!(f, " ({})", reasons.join(", "))?;
            }
        }
        Ok(())
    }
}

/// What [`pick`], [`place`], [`handover`](crate::handover()), [`turn`] or [`pull`] did with each
/// task.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TaskResult<T> {
    /// The tasks, in order.
    pub problems: Vec<T>,
    /// Per task: the motions of every candidate that got through, quickest first, or where each
    /// stopped.
    pub outcomes: Vec<std::result::Result<Vec<TaskPlan>, TaskFailure>>,
}

/// Picks up each task's object, trying each of its grasps. IK solves at the grasp, where the
/// fingers ([`Robot::finger_links`], or the hand's links if it has none) may touch the object, and
/// the move in runs backward from there. The move away holds the object, which may touch what it
/// rests on. Run the transit and the approach, close the gripper, then run the departure holding
/// the object ([`Held::attachment`], on the robot relocked at [`Held::locked`]).
pub fn pick(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Pick],
    o: &TaskOptions,
) -> Result<TaskResult<Pick>> {
    let tasks = checked_picks(device, worlds, tasks, o)?;
    by_grasps(device, worlds, &tasks, o, picked)
}

/// Tries the first `preferred[i]` grasps of each pick as one ordinary batch. A task with any
/// complete pick returns those plans. Only tasks with no complete preferred grasp try the
/// remaining grasps, together in a second batch. Zero preferred grasps means to try the entire
/// task in the ordinary batch.
///
/// Candidate indices and finger locks refer to the original task. When both groups stop, every
/// grasp has a stop in its original order. [`Pick::enough`] applies within each group. A later
/// placement has no bearing on which group a pick returns.
pub fn pick_with_preferred_grasps(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Pick],
    preferred: &[usize],
    o: &TaskOptions,
) -> Result<TaskResult<Pick>> {
    ensure_input!(
        preferred.len() == tasks.len(),
        "{} preferred grasp counts for {} picks",
        preferred.len(),
        tasks.len()
    );
    let tasks = checked_picks(device, worlds, tasks, o)?;
    for (i, (&count, task)) in preferred.iter().zip(tasks.iter()).enumerate() {
        ensure_input!(
            count <= task.grasps.len(),
            "task {i} prefers {count} of {} grasps",
            task.grasps.len()
        );
    }
    let has_fallback = |i: usize| preferred[i] > 0 && preferred[i] < tasks[i].grasps.len();
    if !(0..tasks.len()).any(has_fallback) {
        return by_grasps(device, worlds, &tasks, o, picked);
    }
    let mut first = tasks.to_vec();
    for (i, task) in first.iter_mut().enumerate() {
        if has_fallback(i) {
            task.grasps.truncate(preferred[i]);
            task.locked.truncate(preferred[i]);
        }
    }
    let mut result = by_grasps(device, worlds, &first, o, picked)?;
    let mut indices = vec![];
    let mut remaining = vec![];
    for (i, outcome) in result.outcomes.iter().enumerate() {
        if outcome.is_err() && has_fallback(i) {
            let mut task = tasks[i].clone();
            task.grasps.drain(..preferred[i]);
            if !task.locked.is_empty() {
                task.locked.drain(..preferred[i]);
            }
            indices.push(i);
            remaining.push(task);
        }
    }
    if !remaining.is_empty() {
        let tried = by_grasps(device, worlds, &remaining, o, picked)?;
        for (i, outcome) in indices.into_iter().zip(tried.outcomes) {
            match outcome {
                Ok(mut plans) => {
                    for plan in &mut plans {
                        plan.candidate += preferred[i];
                    }
                    result.outcomes[i] = Ok(plans);
                }
                Err(stopped) => {
                    let first = result.outcomes[i]
                        .as_mut()
                        .expect_err("only failed picks try more grasps");
                    first.candidates.extend(stopped.candidates);
                }
            }
        }
    }
    result.problems = tasks.into_owned();
    Ok(result)
}

/// Corrects measured starts and checks the full pick, including every candidate and finger lock.
fn checked_picks<'a>(
    device: &Device,
    worlds: &Worlds,
    tasks: &'a [Pick],
    o: &TaskOptions,
) -> Result<Cow<'a, [Pick]>> {
    let robot = device.robot();
    let corrected = robot.correct_starts(
        tasks,
        o.plan.start_tolerance,
        "task",
        |t| &t.start,
        |t| &mut t.start,
    )?;
    check_picks(robot, worlds, &corrected, o)?;
    Ok(corrected)
}

/// Each task's plans by its grasps, as [`Pick::enough`] and [`Pick::locked`] say how a pick's are
/// tried, `each` ([`picked`] or [`articulated`]) running checked tasks with some of their grasps
/// that keep the robot's locked joints as its device's robot has them.
fn by_grasps<T: Gripping>(
    device: &Device,
    worlds: &Worlds,
    tasks: &[T],
    o: &TaskOptions,
    each: impl Fn(&Device, &Worlds, &[T], &TaskOptions) -> Result<TaskResult<T>> + Sync,
) -> Result<TaskResult<T>> {
    device.check_worlds(worlds)?;
    let whole = |t: &T| t.plans_wanted().is_none() && t.locks().iter().all(HashMap::is_empty);
    if tasks.iter().all(whole) {
        return each(device, worlds, tasks, o);
    }
    let as_loaded = HashMap::new();
    let counts: Vec<usize> = tasks.iter().map(|t| t.gripped_by().len()).collect();
    let enough: Vec<Option<usize>> = tasks.iter().map(Gripping::plans_wanted).collect();
    let outcomes = in_waves(&counts, &enough, |next| {
        // A relocked robot for each set of locked joints' values, and for it each task with
        // the grasps that keep them, a batch of their own, the groups in parallel; a device
        // for it builds nothing, and the worlds stay valid.
        let mut groups: Vec<Group<'_>> = vec![];
        for (i, t) in tasks.iter().enumerate() {
            for grasp in next[i].clone() {
                let locked = t.locks().get(grasp).unwrap_or(&as_loaded);
                let k = groups.iter().position(|g| g.locked == locked);
                let k = k.unwrap_or_else(|| {
                    groups.push(Group {
                        locked,
                        tasks: vec![],
                    });
                    groups.len() - 1
                });
                match groups[k].tasks.last_mut() {
                    Some((task, grasps)) if *task == i => grasps.push(grasp),
                    _ => groups[k].tasks.push((i, vec![grasp])),
                }
            }
        }
        let results: Vec<TaskResult<T>> = device.in_turns(&mut groups, |_, group| {
            let mine: Vec<T> = group
                .tasks
                .iter()
                .map(|(i, grasps)| tasks[*i].by(grasps, group.locked))
                .collect();
            if group.locked.is_empty() {
                each(device, worlds, &mine, o)
            } else {
                let relocked = device.with_robot(&device.robot().relock(group.locked)?)?;
                each(&relocked, worlds, &mine, o)
            }
        })?;
        let mut wave: Vec<Wave> = tasks.iter().map(|_| Wave::default()).collect();
        for (group, result) in groups.iter().zip(results) {
            for ((i, grasps), outcome) in group.tasks.iter().zip(result.outcomes) {
                wave[*i].take(outcome, |k| grasps[k]);
            }
        }
        Ok(wave)
    })?;
    Ok(TaskResult {
        problems: tasks.to_vec(),
        outcomes,
    })
}

/// What some of a task's candidates came to: the plans of those that got through, and where the
/// others stopped where none did, each numbered as the task's candidates are.
#[derive(Default)]
struct Wave {
    plans: Vec<TaskPlan>,
    stops: Vec<(usize, CandidateStop)>,
}

impl Wave {
    /// Adds the `outcome` of a call given some of the task's candidates, the call's `k`th being
    /// the task's `numbered(k)`th.
    fn take(
        &mut self,
        outcome: std::result::Result<Vec<TaskPlan>, TaskFailure>,
        numbered: impl Fn(usize) -> usize,
    ) {
        match outcome {
            Ok(found) => self.plans.extend(found.into_iter().map(|plan| TaskPlan {
                candidate: numbered(plan.candidate),
                ..plan
            })),
            Err(failure) => {
                let stops = failure.candidates.into_iter().enumerate();
                self.stops
                    .extend(stops.map(|(k, stop)| (numbered(k), stop)));
            }
        }
    }
}

/// Each task's plans by its candidates tried in their order: all of them (`counts`, per task),
/// or, where only some plans will do (`enough`), [`FEW`] of them and then twice as many each
/// time until that many have got through. `tried` plans the candidates it is given per task
/// (none for a task that has enough) and says what they came to. A task's plans are the first
/// that many in its candidates' order, quickest first; one without a plan has tried every
/// candidate, and says where each stopped.
fn in_waves(
    counts: &[usize],
    enough: &[Option<usize>],
    mut tried: impl FnMut(&[Range<usize>]) -> Result<Vec<Wave>>,
) -> Result<Vec<std::result::Result<Vec<TaskPlan>, TaskFailure>>> {
    let mut plans: Vec<Vec<TaskPlan>> = counts.iter().map(|_| vec![]).collect();
    let mut stops: Vec<Vec<Option<CandidateStop>>> =
        counts.iter().map(|&count| vec![None; count]).collect();
    // How many of each task's candidates have been tried, and how many more a task that wants
    // only some plans tries at a time.
    let mut from = vec![0; counts.len()];
    let mut few = FEW;
    loop {
        let next = (0..counts.len()).map(|i| {
            let more = match enough[i] {
                Some(enough) if plans[i].len() >= enough => 0,
                Some(enough) => few.max(enough),
                None => counts[i],
            };
            from[i]..counts[i].min(from[i] + more)
        });
        let next: Vec<Range<usize>> = next.collect();
        if next.iter().all(Range::is_empty) {
            break;
        }
        for (i, wave) in tried(&next)?.into_iter().enumerate() {
            plans[i].extend(wave.plans);
            for (candidate, stop) in wave.stops {
                stops[i][candidate] = Some(stop);
            }
            from[i] = next[i].end;
        }
        few *= 2;
    }
    let outcomes = plans.into_iter().zip(stops).zip(enough);
    let outcomes = outcomes.map(|((mut plans, stops), enough)| {
        if plans.is_empty() {
            let candidates = stops.into_iter();
            return Err(TaskFailure {
                candidates: candidates
                    .map(|stop| stop.expect("a task without a plan has tried every candidate"))
                    .collect(),
            });
        }
        plans.sort_by_key(|plan| plan.candidate);
        plans.truncate(enough.unwrap_or(usize::MAX));
        plans.sort_by(|a, b| a.duration().total_cmp(&b.duration()));
        Ok(plans)
    });
    Ok(outcomes.collect())
}

/// How many grasps a pick that wants only some plans ([`Pick::enough`]) tries first, and how
/// many poses a place that does ([`Place::enough`]): enough that their IK is one batch worth
/// sending, and twice as many each time after.
const FEW: usize = 32;

/// The candidates of a call's places that open their fingers alike where they let go, each with
/// its world among those the hands back off in.
struct Opening<'a, 'c> {
    open: &'a HashMap<String, f32>,
    worlds: Vec<u32>,
    candidates: Vec<&'c mut Candidate>,
}

/// The grasps of a call's picks or turns that keep the robot's locked joints at one set of values.
struct Group<'a> {
    locked: &'a HashMap<String, f32>,
    /// Each task with one, and its grasps that do, in order.
    tasks: Vec<(usize, Vec<usize>)>,
}

/// That `tasks` and `o` can be picked by `robot` in `worlds`, each task named by its index.
fn check_picks(robot: &Robot, worlds: &Worlds, tasks: &[Pick], o: &TaskOptions) -> Result<()> {
    check_options(o)?;
    // The locked joints' values found good, each once however many grasps keep them.
    let mut checked: Vec<&HashMap<String, f32>> = vec![];
    for (i, t) in tasks.iter().enumerate() {
        let world = check_task(robot, worlds, i, &Steps::of_pick(t), &t.start, t.hand)?;
        if let Some(body) = &t.body {
            t.check_parts(world, format_args!("task {i}"))?;
            ensure_input!(
                !matches!(body.shape, Obstacle::Sdf { .. }),
                "task {i}: a distance grid cannot be carried as the pick body"
            );
            body.shape
                .check()
                .map_err(|e| input!("task {i}: pick body {:?} {e}", body.shape))?;
        } else {
            check_object(world, i, t.object, "picked up")?;
        }
        check_poses(i, "grasp", &t.grasps)?;
        ensure_input!(
            t.approach.is_finite()
                && t.approach > 0.0
                && t.lift.is_finite()
                && t.lift != Vec3::ZERO,
            "task {i} needs a positive approach and a finite lift that moves, got {} m and {}",
            t.approach,
            t.lift
        );
        check_support(i, world, &t.support, Some(t.object))?;
        check_tries(robot, i, t, &mut checked)?;
    }
    Ok(())
}

/// That task `i`'s `object` is one of `world`'s obstacles and can be gripped to be `done` so: not
/// a distance grid.
fn check_object(world: &World, i: usize, object: usize, done: &str) -> Result<()> {
    let obstacle = world
        .obstacles
        .get(object)
        .ok_or_else(|| input!("task {i}: obstacle {object} of {}", world.obstacles.len()))?;
    ensure_input!(
        !matches!(obstacle, Obstacle::Sdf { .. }),
        "task {i}: obstacle {object} is a distance grid, which cannot be {done}"
    );
    Ok(())
}

/// That task `i` wants at least one plan and keeps locked joints the robot has at values it may
/// take, per grasp: each set of values checked once among those in `checked`.
fn check_tries<'a>(
    robot: &Robot,
    i: usize,
    t: &'a impl Gripping,
    checked: &mut Vec<&'a HashMap<String, f32>>,
) -> Result<()> {
    ensure_input!(
        t.plans_wanted() != Some(0),
        "task {i}: at least one plan must be enough, got 0"
    );
    ensure_input!(
        t.locks().is_empty() || t.locks().len() == t.gripped_by().len(),
        "task {i}: locked joints' values for {} grasps of {}",
        t.locks().len(),
        t.gripped_by().len()
    );
    for (grasp, locked) in t.locks().iter().enumerate() {
        if !checked.contains(&locked) {
            held::check_locked(robot, locked, &format!("task {i}, grasp {grasp}"))?;
            checked.push(locked);
        }
    }
    Ok(())
}

/// [`pick`] for checked tasks that keep the robot's locked joints as `device`'s robot has them.
fn picked(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Pick],
    o: &TaskOptions,
) -> Result<TaskResult<Pick>> {
    let robot = device.robot();
    let mut candidates = vec![];
    for (t, task) in tasks.iter().enumerate() {
        let object = worlds[task.world as usize].obstacles[task.object]
            .pose()
            .rigid();
        for (index, &grasp) in task.grasps.iter().enumerate() {
            let at = object.mul_pose(grasp.rigid());
            let poses = [
                behind(at, task.approach),
                at,
                Pose::new(at.position + task.lift, at.rotation),
            ];
            let steps = Steps::of_pick(task);
            candidates.push(Candidate::new(
                t,
                index,
                &steps,
                task.hand,
                &task.start,
                poses,
            ));
        }
    }

    // Starts clear of everything, the object included.
    let worlds_of: Vec<u32> = tasks
        .iter()
        .map(|t| Steps::of_pick(t).transit_world)
        .collect();
    start(
        device,
        worlds,
        &mut live(&mut candidates, TaskStep::Start),
        &worlds_of,
    )?;

    // The grasp and the move in to it, the fingers touching the object.
    if !live(&mut candidates, TaskStep::Reach).is_empty() {
        let touching: Vec<World> = tasks
            .iter()
            .map(|t| {
                let mut w = worlds[t.world as usize].clone();
                let links = match t.body {
                    Some(_) => robot.finger_links(t.hand),
                    None => fingers(robot, t.hand),
                };
                w.touch.push(Touch::new(t.object, links));
                w
            })
            .collect();
        let touching = device.upload(&touching)?;
        reach(
            device,
            &touching,
            &mut live(&mut candidates, TaskStep::Reach),
            &by_task,
            &o.ik,
        )?;
        let mut moving = live(&mut candidates, TaskStep::Approach);
        straight(device, &touching, &mut moving, &by_task, Leg::In, o)?;
    }

    // Away with the object, as each grasp holds it, touching what it rests on: on a device per
    // way of holding it (candidates that hold theirs alike share one), each built and run in
    // parallel. Holders for one hand name the held object's link alike, so one upload per hand
    // serves them all.
    let departing = live(&mut candidates, TaskStep::Depart);
    if !departing.is_empty() {
        let mut alike: Vec<(Held, Vec<&mut Candidate>)> = vec![];
        for c in departing {
            let holding = held(&tasks[c.task], worlds, c.index, o.held_spheres);
            match alike.iter_mut().find(|(h, _)| *h == holding) {
                Some((_, group)) => group.push(c),
                None => alike.push((holding, vec![c])),
            }
        }
        let holding: Vec<&Held> = alike.iter().map(|(h, _)| h).collect();
        let fitted = Fitted::new(device, robot, &holding)?;
        let holder = |k: usize, held: &Held| fitted.holder(device, robot, k, held);
        // The worlds a hand lifts in: each without its object, the held link of that hand's
        // tasks touching their supports.
        let removed: Vec<_> = tasks.iter().map(Pick::parts).collect();
        let lifting = |hand: usize| {
            tasks
                .iter()
                .zip(&removed)
                .map(|(t, parts)| {
                    t.lift_in(
                        &worlds[Steps::of_pick(t).depart_world as usize],
                        parts,
                        hand,
                    )
                })
                .collect::<Result<Vec<World>>>()
        };
        let mut lifted: Vec<(usize, Worlds)> = vec![];
        for (k, (holding, _)) in alike.iter().enumerate() {
            if lifted.iter().all(|(hand, _)| *hand != holding.hand) {
                lifted.push((
                    holding.hand,
                    holder(k, holding)?.upload(&lifting(holding.hand)?)?,
                ));
            }
        }
        device.in_turns(&mut alike, |k, (holding, group)| {
            let (_, uploaded) = lifted
                .iter()
                .find(|(hand, _)| *hand == holding.hand)
                .expect("every hand that lifts has its worlds");
            // Built where it is used, so that building one overlaps the other groups' moves.
            straight(
                &holder(k, holding)?,
                uploaded,
                group,
                &by_task,
                Leg::Away,
                o,
            )?;
            // Contacts name obstacles as the task's world does, which still holds the object.
            for c in group.iter_mut() {
                for contact in &mut c.contacts {
                    if let Contact::World { obstacle, .. } = contact {
                        for &part in removed[c.task].iter() {
                            if *obstacle < part {
                                break;
                            }
                            *obstacle += 1;
                        }
                    }
                }
            }
            Ok(())
        })?;
    }

    // From the start, around the object.
    let constraints: Vec<&[HandConstraint]> = tasks.iter().map(|t| &t.constraints[..]).collect();
    transit(
        device,
        worlds,
        &mut live(&mut candidates, TaskStep::Transit),
        &constraints,
        &o.plan,
    )?;

    let outcomes = choose(device, tasks.len(), &candidates, o, |c| {
        (
            Some(held(&tasks[c.task], worlds, c.index, o.held_spheres)),
            None,
        )
    })?;
    Ok(TaskResult {
        problems: tasks.to_vec(),
        outcomes,
    })
}

/// Sets down each task's held object, trying each of its poses. IK solves where the hand lets go,
/// holding the object, which may touch what it is set on there and on the move in, run backward
/// from there. Run transit and approach holding the object, open the gripper, then follow the
/// departure. The releasing fingers may touch the object. When a fall is specified, check the
/// follower against the returned `falling` obstacle through departure. Verify the expected
/// landing in `placed` after release. If the task specifies [`Place::open`], the solver keeps
/// the fingers at the holding aperture until release, then backs off at the first clear opening
/// ([`TaskPlan::opening`]): open the gripper that far, and follow the departure on the robot
/// relocked there. A task that says how many plans will do ([`Place::enough`]) tries its poses in
/// their order until that many have got through. `device` is the robot's without the object; each
/// task's `held` is attached to it here.
pub fn place(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Place],
    o: &TaskOptions,
) -> Result<TaskResult<Place>> {
    device.check_worlds(worlds)?;
    let robot = device.robot();
    let corrected = robot.correct_starts(
        tasks,
        o.plan.start_tolerance,
        "task",
        |t| &t.start,
        |t| &mut t.start,
    )?;
    let tasks = corrected.as_ref();
    check_options(o)?;
    // The openings checked so far: tasks that offer the same ones check them once.
    let mut opened: Vec<&HashMap<String, f32>> = vec![];
    for (i, t) in tasks.iter().enumerate() {
        let world = check_task(robot, worlds, i, &Steps::of_place(t), &t.start, t.held.hand)?;
        if let Some(idle) = &t.idle {
            ensure_input!(
                idle.len() == robot.dof(),
                "task {i}: idle joints need a configuration of {} values, got {}",
                robot.dof(),
                idle.len()
            );
            let within = |j: usize| {
                let (lo, hi) = robot.bounds(j);
                idle[j].is_finite() && idle[j] >= lo && idle[j] <= hi
            };
            if let Some(j) = robot
                .idle_joints(t.held.hand)
                .into_iter()
                .find(|&j| !within(j))
            {
                return Err(input!(
                    "task {i}: idle joint {j} at {}, outside its limits",
                    idle[j]
                ));
            }
        }
        t.held.check(robot, worlds, &format!("task {i}"))?;
        check_poses(i, "pose", &t.poses)?;
        ensure_input!(
            t.falls.is_empty()
                || (t.falls.len() == t.poses.len()
                    && t.falls.iter().all(|z| z.is_finite() && *z >= 0.0)),
            "task {i} needs one finite nonnegative fall per release pose, or none"
        );
        ensure_input!(
            t.lower.is_finite()
                && t.lower != Vec3::ZERO
                && t.retreat.is_finite()
                && t.retreat > 0.0,
            "task {i} needs a finite lowering that moves and a positive retreat, got {} and {} m",
            t.lower,
            t.retreat
        );
        check_support(i, world, &t.support, None)?;
        ensure_input!(
            t.enough != Some(0),
            "task {i}: at least one plan must be enough, got 0"
        );
        for (k, open) in t.open.iter().enumerate() {
            if !opened.contains(&open) {
                held::check_locked(robot, open, &format!("task {i}: opening {k}"))?;
                opened.push(open);
            }
        }
    }
    if tasks.iter().all(|t| t.enough.is_none()) {
        return placed(device, worlds, tasks, o);
    }
    let counts: Vec<usize> = tasks.iter().map(|t| t.poses.len()).collect();
    let enough: Vec<Option<usize>> = tasks.iter().map(|t| t.enough).collect();
    let outcomes = in_waves(&counts, &enough, |next| {
        // The tasks with poses to try, each with those alone.
        let going: Vec<usize> = (0..tasks.len()).filter(|&i| !next[i].is_empty()).collect();
        let some = going.iter().map(|&i| Place {
            poses: tasks[i].poses[next[i].clone()].to_vec(),
            falls: tasks[i]
                .falls
                .get(next[i].clone())
                .unwrap_or_default()
                .to_vec(),
            ..tasks[i].clone()
        });
        let some: Vec<Place> = some.collect();
        let result = placed(device, worlds, &some, o)?;
        let mut wave: Vec<Wave> = tasks.iter().map(|_| Wave::default()).collect();
        for (&i, outcome) in going.iter().zip(result.outcomes) {
            wave[i].take(outcome, |k| next[i].start + k);
        }
        Ok(wave)
    })?;
    Ok(TaskResult {
        problems: tasks.to_vec(),
        outcomes,
    })
}

/// [`place`] for checked tasks, every pose of each.
fn placed(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Place],
    o: &TaskOptions,
) -> Result<TaskResult<Place>> {
    let robot = device.robot();
    // A device holding each task's object, built in parallel; tasks holding theirs alike share one.
    // Every step below runs each holder's candidates as a batch of their own, as many holders
    // at a time as the device has threads (`Device::in_turns`).
    let mut helds: Vec<&Held> = vec![];
    let mut holder_of = vec![];
    for t in tasks {
        let h = match helds.iter().position(|held| **held == t.held) {
            Some(h) => h,
            None => {
                helds.push(&t.held);
                helds.len() - 1
            }
        };
        holder_of.push(h);
    }
    let holders = held::holders(device, robot, &helds)?;
    let mut candidates = vec![];
    for (t, task) in tasks.iter().enumerate() {
        for (index, &pose) in task.poses.iter().enumerate() {
            let at = pose.mul_pose(task.held.grasp);
            let poses = [
                Pose::new(at.position - task.lower, at.rotation),
                at,
                behind(at, task.retreat),
            ];
            let hand = task.held.hand;
            let steps = Steps::of_place(task);
            let mut c = Candidate::new(t, index, &steps, hand, &task.start, poses);
            if let Some(idle) = &task.idle {
                for j in robot.idle_joints(hand) {
                    c.rest[j] = idle[j];
                }
            }
            candidates.push(c);
        }
    }

    // Starts clear of everything, holding the object.
    let worlds_of: Vec<u32> = tasks
        .iter()
        .map(|t| Steps::of_place(t).transit_world)
        .collect();
    let mut starting = by_holder(live(&mut candidates, TaskStep::Start), &holder_of, &holders);
    device.in_turns(&mut starting, |h, group| {
        start(&holders[h], worlds, group, &worlds_of)
    })?;

    // Where the hand lets go, and the move in to there, the object touching what it is set on.
    // Where the hand lets go fixes where the object is, so IK for the robot alone solves every
    // task in one batch; each task's solutions are then checked holding its object, nearest its
    // rest first, as searches check them. Holders for one hand name the held object's link
    // alike, so one upload per hand serves them all.
    let arriving = |hand: usize| -> Vec<World> {
        tasks
            .iter()
            .map(|t| {
                let mut w = worlds[t.world as usize].clone();
                if t.held.hand == hand {
                    let link = held::held_link(hand);
                    w.touch
                        .extend(t.support.iter().map(|&s| Touch::new(s, [&link])));
                }
                w
            })
            .collect()
    };
    let mut arrived: Vec<(usize, Worlds)> = vec![];
    let as_loaded = HashMap::new();
    let mut reaching = live(&mut candidates, TaskStep::Reach);
    if !reaching.is_empty() {
        for c in &reaching {
            if arrived.iter().all(|(hand, _)| *hand != c.hand) {
                let holder = &holders[holder_of[c.task]];
                arrived.push((c.hand, holder.upload(&arriving(c.hand))?));
            }
        }
        // The fingers each release is solved with: where the robot was loaded with them, to
        // which the hand opens there; or, for a task that says how far its hand opens, where
        // they hold the object. Those solved alike are side by side, a batch on a robot relocked
        // so; where all are, the candidates keep their order.
        let solved_with = |c: &Candidate| match &tasks[c.task] {
            t if t.open.is_empty() => &as_loaded,
            t => &t.held.locked,
        };
        let mut alike: Vec<&HashMap<String, f32>> = vec![];
        for c in &reaching {
            if !alike.contains(&solved_with(c)) {
                alike.push(solved_with(c));
            }
        }
        reaching.sort_by_key(|c| alike.iter().position(|with| *with == solved_with(c)));
        let mut lots: Vec<(Range<usize>, Option<Device>)> = vec![];
        for with in alike {
            let from = lots.last().map_or(0, |(lot, _)| lot.end);
            let count = reaching[from..]
                .iter()
                .take_while(|c| solved_with(c) == with)
                .count();
            let relocked = match with.is_empty() {
                true => None,
                false => Some(device.with_robot(&robot.relock(with)?)?),
            };
            lots.push((from..from + count, relocked));
        }
        let solve_lot = |(lot, relocked): &(Range<usize>, Option<Device>)| {
            let given = |_: usize, c: &Candidate| c.world;
            let on = relocked.as_ref().unwrap_or(device);
            solve(on, worlds, &reaching[lot.clone()], &given, &o.ik)
        };
        let solved: Vec<IkResult> = match &mut lots[..] {
            [one] => vec![solve_lot(one)?],
            lots => device.in_turns(lots, |_, lot| solve_lot(lot))?,
        };
        let mut solutions: Vec<Vec<f32>> = vec![];
        for ((lot, _), solved) in lots.iter().zip(&solved) {
            let lot = reaching[lot.clone()].iter().enumerate();
            solutions.extend(lot.map(|(k, c)| nearest_first(solved, k, &c.rest)));
        }
        let mut mine: Vec<Vec<usize>> = holders.iter().map(|_| vec![]).collect();
        for (i, c) in reaching.iter().enumerate() {
            mine[holder_of[c.task]].push(i);
        }
        let uploaded_for = |hand: usize| {
            let (_, uploaded) = arrived
                .iter()
                .find(|(h, _)| *h == hand)
                .expect("every hand that lets go has its worlds");
            uploaded
        };
        let found: Vec<Vec<Option<usize>>> = device.in_turns(&mut mine, |h, mine| {
            let Some(&first) = mine.first() else {
                return Ok(vec![]);
            };
            let lists = mine
                .iter()
                .map(|&i| (reaching[i].task as u32, solutions[i].clone()))
                .collect();
            first_free(&holders[h], uploaded_for(reaching[first].hand), lists)
        })?;
        let n = robot.dof();
        // Per holder, the candidates whose every solution collides holding the object.
        let mut holding: Vec<Vec<usize>> = holders.iter().map(|_| vec![]).collect();
        for (&i, found) in mine.iter().flatten().zip(found.into_iter().flatten()) {
            let c = &mut reaching[i];
            match found {
                Some(k) => {
                    c.grasped = solutions[i][k * n..(k + 1) * n].to_vec();
                    c.past_reach();
                }
                None if solutions[i].is_empty() => {}
                None => {
                    c.stop = Some(ToolPathStop::Collision { at: c.poses[1] });
                    holding[holder_of[c.task]].push(i);
                }
            }
        }
        // What the nearest solution collides with, holding the object.
        for (holder, colliding) in holders.iter().zip(&holding) {
            let Some(&first) = colliding.first() else {
                continue;
            };
            let item_world: Vec<u32> = colliding.iter().map(|&i| reaching[i].task as u32).collect();
            let q: Vec<f32> = colliding
                .iter()
                .flat_map(|&i| solutions[i][..n].iter().copied())
                .collect();
            let uploaded = uploaded_for(reaching[first].hand);
            let contacts = holder.contacts(uploaded, &item_world, &q, 0.0)?;
            for (&i, contacts) in colliding.iter().zip(contacts) {
                reaching[i].contacts = contacts;
            }
        }
        for ((lot, relocked), solved) in lots.iter().zip(&solved) {
            let on = relocked.as_ref().unwrap_or(device);
            classify(on, worlds, &mut reaching[lot.clone()], solved, &o.ik)?;
        }
    }
    let mut moving = by_holder(
        live(&mut candidates, TaskStep::Approach),
        &holder_of,
        &holders,
    );
    device.in_turns(&mut moving, |h, group| {
        let Some(hand) = group.first().map(|c| c.hand) else {
            return Ok(());
        };
        let (_, uploaded) = arrived
            .iter()
            .find(|(h, _)| *h == hand)
            .expect("every hand that lets go has its worlds");
        straight(&holders[h], uploaded, group, &by_task, Leg::In, o)
    })?;

    // Back from the object where it stands, the fingers touching it: a world for each.
    let leaving = live(&mut candidates, TaskStep::Depart);
    if !leaving.is_empty() {
        let left: Vec<World> = leaving
            .iter()
            .map(|c| {
                let t = &tasks[c.task];
                let mut w = worlds[Steps::of_place(t).depart_world as usize].clone();
                let released = t.held.object.placed(t.poses[c.index]);
                let fall = t.falls.get(c.index).copied().unwrap_or_default();
                w.obstacles.push(released.falling(fall));
                let object = w.obstacles.len() - 1;
                w.touch
                    .push(Touch::new(object, fingers(robot, t.held.hand)));
                w
            })
            .collect();
        let left = device.upload(&left)?;
        // The hand opens to the first of its task's openings with which it is clear there and
        // on its way back: those that open alike are a batch on a robot relocked so, and those
        // the fingers stop try their next. A task that names none opens to where the robot was
        // loaded.
        let unnamed = [as_loaded];
        let openings = |c: &Candidate| match &tasks[c.task].open[..] {
            [] => &unnamed[..],
            open => open,
        };
        // Only a stop that is the fingers' own can end otherwise at another opening: out of
        // reach, a jump and what the arm or the palm meets are the same at each.
        let of_fingers = |c: &Candidate| {
            let fingers = fingers(robot, c.hand);
            let theirs = |link: &String| fingers.contains(&link.as_str());
            let collided = matches!(c.stop, Some(ToolPathStop::Collision { .. }));
            collided
                && c.contacts.iter().all(|contact| match contact {
                    Contact::World { link, .. } => theirs(link),
                    Contact::SelfCollision { links, .. } => links.iter().any(theirs),
                })
        };
        let departing = leaving.first().map_or(0, |c| c.passed);
        let mut waiting: Vec<(u32, &mut Candidate)> = (0..).zip(leaving).collect();
        let mut round = 0;
        while !waiting.is_empty() {
            let mut lots: Vec<Opening<'_, '_>> = vec![];
            for (world, c) in waiting.drain(..) {
                let open = &openings(c)[round];
                let k = lots.iter().position(|lot| lot.open == open);
                let k = k.unwrap_or_else(|| {
                    lots.push(Opening {
                        open,
                        worlds: vec![],
                        candidates: vec![],
                    });
                    lots.len() - 1
                });
                lots[k].worlds.push(world);
                lots[k].candidates.push(c);
            }
            let away = |lot: &mut Opening<'_, '_>| {
                let each = |k: usize, _: &Candidate| lot.worlds[k];
                if lot.open.is_empty() {
                    return straight(device, &left, &mut lot.candidates, &each, Leg::Away, o);
                }
                let relocked = device.with_robot(&robot.relock(lot.open)?)?;
                straight(&relocked, &left, &mut lot.candidates, &each, Leg::Away, o)
            };
            match &mut lots[..] {
                [one] => away(one)?,
                lots => drop(device.in_turns(lots, |_, lot| away(lot))?),
            }
            for lot in lots {
                for (world, c) in lot.worlds.into_iter().zip(lot.candidates) {
                    let named = !tasks[c.task].open.is_empty();
                    if c.passed > departing {
                        c.opening = named.then_some(round);
                    } else if round + 1 < openings(c).len() && of_fingers(c) {
                        c.stop = None;
                        c.contacts.clear();
                        waiting.push((world, c));
                    }
                }
            }
            round += 1;
        }
    }

    // From the start, holding the object.
    let mut going = by_holder(
        live(&mut candidates, TaskStep::Transit),
        &holder_of,
        &holders,
    );
    let constraints: Vec<&[HandConstraint]> = tasks.iter().map(|t| &t.constraints[..]).collect();
    device.in_turns(&mut going, |h, group| {
        transit(&holders[h], worlds, group, &constraints, &o.plan)
    })?;

    let mut outcomes = choose(device, tasks.len(), &candidates, o, |c| {
        let t = &tasks[c.task];
        let mut at = t.poses[c.index];
        at.position.z -= t.falls.get(c.index).copied().unwrap_or_default();
        (None, Some(t.held.object.placed(at)))
    })?;
    for (task, outcome) in tasks.iter().zip(&mut outcomes) {
        if let Ok(plans) = outcome {
            for plan in plans {
                if let Some(&fall) = task.falls.get(plan.candidate).filter(|&&z| z > 0.0) {
                    plan.falling = Some(
                        task.held
                            .object
                            .placed(task.poses[plan.candidate])
                            .falling(fall),
                    );
                }
            }
        }
    }
    Ok(TaskResult {
        problems: tasks.to_vec(),
        outcomes,
    })
}

/// Poses that set `object` down at `pose` alike, for [`Place::poses`]: `pose` turned about the
/// object's own z axis by its symmetries. A box gives 2, or 4 where its x and y sides match; a
/// ball, cylinder or capsule `turns` (at least 1); a distance grid `pose` alone.
#[must_use]
pub fn place_poses(object: &Obstacle, pose: Pose, turns: usize) -> Vec<Pose> {
    let count = match *object {
        Obstacle::Cuboid { half_extents, .. } => {
            if (half_extents.x - half_extents.y).abs() < 1e-6 {
                4
            } else {
                2
            }
        }
        Obstacle::Sphere { .. } | Obstacle::Cylinder { .. } | Obstacle::Capsule { .. } => {
            turns.max(1)
        }
        Obstacle::Sdf { .. } => 1,
    };
    (0..count)
        .map(|k| {
            let turn = Quat::from_rotation_z(TAU * k as f32 / count as f32);
            Pose::new(pose.position, pose.rotation * turn)
        })
        .collect()
}

/// How grasps are proposed for a parallel gripper whose IK frame lies between its fingertips, its
/// z axis pointing out of the hand and its y axis along the fingers' closing direction (as the
/// examples' Panda `ee_link`): by [`grasps`], for a box, ball, cylinder or capsule, and by
/// [`grasps_on`](crate::grasps_on), for what cameras see of any shape. Both read `opening`,
/// `turns`, `floor` and `overlap`; `depth`, `tip` and `along` are for `grasps` alone, whose
/// hand is the options' own, and `friction`, `contact`, `approach` and `near` for `grasps_on`
/// alone, whose hand is the robot's. [`Grasp::moment`](crate::Grasp::moment) reads `friction`
/// and `contact`.
#[derive(Clone, Copy, Debug)]
pub struct GraspOptions {
    /// The widest the fingers open (meters): 8 cm.
    pub opening: f32,
    /// How far in from the object's surface the IK frame holds it (meters): 2 cm. Above a `floor`,
    /// a grasp from above may hold it less deep, its frame even outside the object.
    pub depth: f32,
    /// Grasps spread around a round object's axis, and for `grasps_on`, the directions each pair
    /// of patches is grasped from around the line between them: 8.
    pub turns: usize,
    /// How far the fingers reach past the IK frame along its z axis (meters), as
    /// [`Robot::tip`] measures it: 0.
    pub tip: f32,
    /// A height the hand's frame and its open fingers stay above, in the frame the object is
    /// placed in (meters): the top of the table it stands on, and room to spare, for the fingers'
    /// thickness too. The fingers are taken `opening` apart and reaching `tip` past the frame, so
    /// one sits lower where the closing direction is not level. A grasp from above that would take
    /// them lower holds the object less deep, its frame above the object if need be, and one that
    /// leaves less than `overlap` of the fingers in it is left out, as are grasps from below and
    /// from the side whose lower finger would be under the floor. `None` for no floor.
    pub floor: Option<f32>,
    /// The least the fingers reach into the object, from their tips, in a grasp from above over a
    /// `floor` (meters): 1 cm, as MoveIt Grasps' minimum depth for the Panda. An object whose top
    /// is less than this over the floor has no grasp from above. For `grasps_on`, the least of
    /// the fingers that stays on the patches they close on where the hand is drawn back to stay
    /// clear.
    pub overlap: f32,
    /// How far toward the ends a grasp holds the object, along the side the fingers close on
    /// rather than across, as a fraction of half that side, in [0, 1): 0, the middle. Above 0,
    /// each grasp comes once toward each end; a hand that will pass the object to another
    /// ([`handover`](crate::handover())) holds it toward one end, as 0.75, leaving the other to
    /// the hand that takes it. A ball's grasps, and a round object's from its ends or closing along
    /// its axis, stay in the middle; those closing around its side slide along the axis.
    pub along: f32,
    /// The friction between a fingertip and what it presses on, for grasps of a surface
    /// ([`grasps_on`](crate::grasps_on)): 0.5, as Dex-Net samples grasps with. The line a grasp
    /// closes along lies within the friction cone of each patch it presses on, `atan(friction)`
    /// from the patch's normal; less asks for patches that face each other more squarely.
    pub friction: f32,
    /// The radius of the patch a fingertip presses on, for grasps of a surface (meters): 5 mm.
    /// A patch is the mean of the points within it, and surfaces nearer each other than this
    /// along a closing line count as one.
    pub contact: f32,
    /// How far the hand comes in straight along its z axis to a grasp of a surface, a way that
    /// must be clear of what was seen (meters): 10 cm, a pick's `approach`.
    pub approach: f32,
    /// The thickest wall seen from one side only that a grasp of a surface may close across
    /// (meters): `None`, every grasp closes on two patches seen. With a thickness, a patch whose
    /// other side no camera saw is closed on where the cameras saw past the wall: where the line in
    /// along its normal first meets space every ray around it crossed on its way to something
    /// farther ([`Surface::seen_from`](crate::Surface::seen_from)), within this thickness, with
    /// nothing of the object seen facing along the line before it. The wall has ended there: the
    /// second contact is put there, the other side taken parallel to the side seen, so the width is
    /// the most the wall can be, and a wall is never taken thicker than the fingers open: a shell
    /// past the `opening` is one at it. A wall that leans is closed on level as well, where the
    /// level line through the patch is inside its friction cone (`friction`): the hand then comes
    /// straight down across a flared rim, where one closing along the wall's normal leans as the
    /// wall does; deeper down such a wall, the finger on the side it leans over meets it above the
    /// patch ([`first_touch`](Self::first_touch) puts those grasps last). A bowl's rim seen from
    /// above is grasped so, its outer wall
    /// hidden under it. Inside a solid no ray goes, so none of these grasps ends in one: a box seen
    /// from one side is closed on only across all of it, where the cameras saw past its far side.
    /// The space found lies past where the wall ends by what the rays leave unseen: under a bowl's
    /// flared wall seen from above, 5.5 to 11.5 mm behind the inner side of a 3 mm wall near its
    /// rim, so 1.5 cm suits. Grasps of two patches seen come first among the ordinary grasps
    /// and again among the intermediate angles. Shallower insertions follow their parent order.
    pub shell: Option<f32>,
    /// Whether the fingers may come in to a grasp of a surface less open than `opening`: false,
    /// every grasp is checked with the fingers as the robot's locked joints hold them. With it,
    /// each finger is taken to slide along the hand's y axis, and a grasp comes in at the
    /// opening ([`Grasp::opening`](crate::Grasp::opening)), of sixteenths of `opening` that
    /// leave each finger 2.5 mm, a surface's noise, clear of the patch it closes on, that
    /// keeps its fingers farthest from what is not the object's, the floor included: with
    /// something within their own travel of the open fingers, the narrowest the hand is clear
    /// at, since a planner lets the fingers touch what they grasp and nothing else, and knows
    /// what stands around less finely than the points do. Of openings as far from all that, it
    /// takes the one with the most room from the object's other sides, then the widest: a
    /// finger inside a bowl stands midway between its walls, and a box with nothing near is
    /// grasped at the whole `opening`. A rim beside a wall the open fingers would be in is
    /// grasped so. A grasp the open hand has stays where it is, drawn back as the open hand
    /// is; where the open hand is clear at no depth, the hand comes in as deep as it is clear
    /// narrower. Lock the fingers there for the pick ([`Pick::locked`]); a call's grasps share at
    /// most sixteen openings.
    pub narrow: bool,
    /// A point to grip a surface near, in its frame, for grasps of a surface: `None`, the pairs
    /// of patches spread over the object. With a point, within each of the tiers
    /// [`grasps_on`](crate::grasps_on) orders its grasps by (walls seen from one side after
    /// those seen from both, and within each, from above before a turn to either side, with
    /// [`first_touch`](Self::first_touch) fingers that close onto their patches first before the
    /// rest, and pairs that face each other within half a friction cone of square before the
    /// rest), the pairs whose middles lie nearest the point come first, the squarest first of two
    /// as near: where a caller was told to grip, as on a knob's axis or at a handle's middle. It
    /// orders the grasps and finds no others.
    pub near: Option<Vec3>,
    /// A point to enclose between the pads in surface grasps. After the ordinary grasps come
    /// intermediate approaches sampled around nearby contact pairs, only those whose fingers close
    /// on this point, and last shallower insertions of point-enclosing grasps, which keep the
    /// contact patches and the required overlap. Ordinary grasps come first, including ones that
    /// miss the point.
    pub close_on: Option<Vec3>,
    /// How far the hand's spheres stay from what they do not close on, on their way in to a grasp
    /// of a surface and at it (meters): 0, a hand is clear that touches nothing. The points say
    /// where a surface is to within their noise, and a planner knows what stands around less
    /// finely still, its distance grids erring toward collision: a hand the points alone find
    /// clear, a millimeter from a drawer's front behind the handle it grips, is one the planner
    /// refuses. With a clearance the hand is drawn back to where it has that much room, while
    /// `overlap` of its fingers stays on the patches, or its grasp is left out; 2.5 mm, a
    /// surface's noise, suits a world of grids a few millimeters fine. What the fingers close on
    /// is exempt, as it is at 0, and fingers that may come in narrower
    /// ([`narrow`](Self::narrow)) keep it too. MoveIt Grasps pads the hand on its approach
    /// likewise.
    pub clearance: f32,
    /// Whether grasps of a surface ([`grasps_on`](crate::grasps_on)) whose fingers close onto
    /// their patches before they meet any other part of what was seen of the object come first,
    /// within each turn of the lot of two seen sides and of the lot of one, so from above still
    /// comes before a turn to either side: false, the order is as without it. A finger closing
    /// level deep across a wall that leans over its patch, as a flared bowl's, meets the wall above
    /// the patch first, a finger's length from the other contact, and the object turns in the hand
    /// as it is lifted: with this, such grasps come after the rest, none left out. What a finger
    /// meets is the object's patches: the means of its readings in cells a patch wide
    /// ([`contact`](Self::contact)) that hold three or more, so that a reading or two off the
    /// surface by a stereo camera's noise is none. So a part of the object no camera saw, or one
    /// seen with fewer readings than that in every such cell, as a wall read sparser than every
    /// 3 mm or a lip that cells' sides split, is met unnoticed. It costs a sweep of each finger of
    /// each grasp over the patches.
    pub first_touch: bool,
}

impl Default for GraspOptions {
    fn default() -> Self {
        Self {
            opening: 0.08,
            depth: 0.02,
            turns: 8,
            tip: 0.0,
            floor: None,
            overlap: 0.01,
            along: 0.0,
            friction: 0.5,
            contact: 0.005,
            approach: 0.1,
            shell: None,
            narrow: false,
            near: None,
            close_on: None,
            clearance: 0.0,
            first_touch: false,
        }
    }
}

/// Grasps of a box, ball, cylinder or capsule in its frame, for [`Pick::grasps`]: the hand
/// facing each side and closing across what fits between its fingers. A box is grasped on each
/// face, closing along either axis of the face it fits across, each both ways round; a ball like
/// a box its size; a cylinder or capsule from `turns` directions around its side, closing around
/// it or along it, and from `turns` turns at each end. With a `floor`, the object's pose places
/// it: pass a world's obstacle as it stands. An `Error::Input` saying why where no grasp fits: the
/// object is wider than the fingers open every way, or every grasp would take the open fingers
/// below the floor or hold less than `overlap` of them.
pub fn grasps(object: &Obstacle, o: &GraspOptions) -> Result<Vec<Pose>> {
    ensure_input!(
        o.opening > 0.0
            && o.opening.is_finite()
            && o.depth >= 0.0
            && o.turns > 0
            && o.tip.is_finite()
            && o.overlap > 0.0
            && o.overlap.is_finite(),
        "grasps need a finite positive opening, a depth of at least 0, a turn, a finite tip and a finite positive overlap, got {} m, {} m, {}, {} m and {} m",
        o.opening,
        o.depth,
        o.turns,
        o.tip,
        o.overlap
    );
    if let Some(floor) = o.floor {
        ensure_input!(floor.is_finite(), "the floor must be finite, got {floor}");
    }
    ensure_input!(
        (0.0..1.0).contains(&o.along),
        "grasps hold the object toward its ends by a fraction in [0, 1), got {}",
        o.along
    );
    ensure_input!(
        !matches!(object, Obstacle::Sdf { .. }),
        "a distance grid has no grasps"
    );
    object
        .check()
        .map_err(|e| input!("the object to grasp: {e}, got {object:?}"))?;
    let fits = |width: f32| width <= o.opening;
    let placed = object.pose();
    // The hand's z axis along `toward`, its fingers closing along `closing`, `depth` in from a
    // surface `surface` from the frame's origin and `shift` along it, or less deep where that takes
    // the frame or an open finger below the floor; none where that leaves less than `overlap` of
    // the fingers in the object.
    let mut fitting = 0;
    let mut grasp = |toward: Vec3, closing: Vec3, surface: f32, shift: Vec3| {
        fitting += 1;
        let rotation = Quat::from_mat3(&Mat3::from_cols(closing.cross(toward), closing, toward));
        let mut depth = o.depth.min(surface);
        if let Some(floor) = o.floor {
            // The frame, and the lower open fingertip: `tip` along the hand's z axis and
            // `opening / 2` across, lower where the closing direction is not level. `height` is
            // that of a point `reach` along the z axis and `below` under it, at a depth `d`; the
            // lowest sets how deep a grasp from above goes.
            let down = (placed.rotation * toward).z;
            let spread = (placed.rotation * closing).z.abs() * o.opening / 2.0;
            let points = [(0.0, 0.0), (o.tip, spread)];
            let base = placed.position.z + (placed.rotation * shift).z;
            let height =
                |d: f32, (reach, below): (f32, f32)| base + down * (d - surface + reach) - below;
            if down < -1e-6 {
                for (reach, below) in points {
                    if height(depth, (reach, below)) < floor {
                        depth = surface - reach + (floor + below - base) / down;
                    }
                }
            }
            let shallow = down < -1e-6 && depth + o.tip < o.overlap;
            let under = points.iter().any(|&p| height(depth, p) < floor - 1e-6);
            // From below, the rest of the hand would be lower still.
            if down > 1e-6 || shallow || under {
                return None;
            }
        }
        Some(Pose::new(shift - toward * (surface - depth), rotation))
    };
    // Where along a side of half length `half`, running along `side`, the grasps hold: the middle,
    // or toward each end.
    let shifts = |side: Vec3, half: f32| -> Vec<Vec3> {
        if o.along == 0.0 {
            vec![Vec3::ZERO]
        } else {
            [1.0, -1.0].map(|end| side * end * o.along * half).to_vec()
        }
    };
    let mut out = vec![];
    let around = |turn: usize| {
        let angle = TAU * turn as f32 / o.turns as f32;
        Vec3::new(angle.cos(), angle.sin(), 0.0)
    };
    let narrowest = match *object {
        Obstacle::Cuboid { half_extents, .. } => {
            sides(half_extents, &fits, &shifts, &mut grasp, &mut out);
            2.0 * half_extents.min_element()
        }
        Obstacle::Sphere { radius, .. } => {
            let middle = |_: Vec3, _: f32| vec![Vec3::ZERO];
            sides(Vec3::splat(radius), &fits, &middle, &mut grasp, &mut out);
            2.0 * radius
        }
        Obstacle::Cylinder {
            radius,
            half_height: length,
            ..
        }
        | Obstacle::Capsule {
            radius,
            half_length: length,
            ..
        } => {
            let end = match object {
                Obstacle::Capsule { .. } => length + radius,
                _ => length,
            };
            for turn in 0..o.turns {
                let outward = around(turn);
                for sign in [1.0, -1.0] {
                    if fits(2.0 * radius) {
                        for shift in shifts(Vec3::Z, length) {
                            let closing = sign * Vec3::Z.cross(outward);
                            out.extend(grasp(-outward, closing, radius, shift));
                        }
                    }
                    if fits(2.0 * end) {
                        out.extend(grasp(-outward, sign * Vec3::Z, radius, Vec3::ZERO));
                    }
                }
            }
            if fits(2.0 * radius) {
                for sign in [1.0, -1.0] {
                    for turn in 0..o.turns {
                        out.extend(grasp(-sign * Vec3::Z, around(turn), end, Vec3::ZERO));
                    }
                }
            }
            2.0 * radius.min(end)
        }
        Obstacle::Sdf { .. } => unreachable!("distance grids are refused above"),
    };
    if out.is_empty() {
        return Err(if fitting == 0 {
            input!(
                "no grasp fits: the object is {:.1} cm across where narrowest, and the fingers open {:.1} cm",
                narrowest * 100.0,
                o.opening * 100.0
            )
        } else {
            input!(
                "no grasp keeps the hand above the floor at {:.3} m: each of the {fitting} that fit between the fingers would take the frame or the open fingers, {:.1} cm apart and reaching {:.1} cm past it, lower, or hold less than {:.1} cm of the fingers",
                o.floor.unwrap_or_default(),
                o.opening * 100.0,
                o.tip * 100.0,
                o.overlap * 100.0
            )
        });
    }
    Ok(out)
}

/// A box's grasps: from each face, closing along each axis of the face it fits across, both ways,
/// at each of `shifts` along the face's other axis.
fn sides(
    half: Vec3,
    fits: &impl Fn(f32) -> bool,
    shifts: &impl Fn(Vec3, f32) -> Vec<Vec3>,
    grasp: &mut impl FnMut(Vec3, Vec3, f32, Vec3) -> Option<Pose>,
    out: &mut Vec<Pose>,
) {
    for face in 0..3 {
        for sign in [1.0, -1.0] {
            let outward = Vec3::AXES[face] * sign;
            let spanned = (0..3).filter(|&a| a != face && fits(2.0 * half[a]));
            for across in spanned {
                let along = 3 - face - across;
                for turn in [1.0, -1.0] {
                    for shift in shifts(Vec3::AXES[along], half[along]) {
                        let closing = Vec3::AXES[across] * turn;
                        out.extend(grasp(-outward, closing, half[face], shift));
                    }
                }
            }
        }
    }
}

/// One candidate of a task on its way through the steps.
pub(crate) struct Candidate {
    pub(crate) task: usize,
    /// Its index among the task's grasps or poses.
    pub(crate) index: usize,
    /// Its task's world in the batch given.
    world: u32,
    /// The world its start is checked and its transit planned in.
    transit_world: u32,
    /// Whether its task plans the transit, and so checks the start.
    transits: bool,
    hand: usize,
    start: Vec<f32>,
    /// Where the joints only other hands move wait after the transit: the start, but for a
    /// place's idle joints.
    rest: Vec<f32>,
    /// The hand's IK frame where the move in begins, where it ends (the grasp or release) and
    /// where the move away ends.
    poses: [Pose; 3],
    /// How many steps it has passed.
    passed: usize,
    /// What stopped it, if something did, and for a collision, what collides.
    stop: Option<ToolPathStop>,
    contacts: Vec<Contact>,
    /// The configuration at the grasp or release, where both straight moves start.
    grasped: Vec<f32>,
    /// Which of its place's openings it lets go at, where the place names some.
    opening: Option<usize>,
    /// Its motions: in, away and from the start, as control points.
    approach: Vec<f32>,
    depart: Vec<f32>,
    transit: Vec<f32>,
    /// In a handover: the giver, which holds the object at the grasp too, and its IK frame's pose
    /// there; it moves away, the joints the other hand moves held on each straight move, and the
    /// object's pose where it changes hands.
    pub(crate) giver: Option<Giver>,
}

/// What a handover's candidate asks of the hand that gives the object.
pub(crate) struct Giver {
    pub(crate) hand: usize,
    /// Its IK frame's pose at the meeting.
    pub(crate) at: Pose,
    /// The joints held still on the taker's move in (the giver's) and on the giver's move away
    /// (the taker's).
    pub(crate) hold_in: Vec<usize>,
    pub(crate) hold_away: Vec<usize>,
    /// The joints neither hand moves, which wait where the start has them through the transit.
    pub(crate) idle: Vec<usize>,
    /// The object's pose where it changes hands.
    pub(crate) meeting: Pose,
}

impl Candidate {
    pub(crate) fn new(
        task: usize,
        index: usize,
        steps: &Steps<'_>,
        hand: usize,
        start: &[f32],
        poses: [Pose; 3],
    ) -> Self {
        Self {
            task,
            index,
            world: steps.world,
            transit_world: steps.transit_world,
            transits: steps.transit,
            hand,
            start: start.to_vec(),
            rest: start.to_vec(),
            poses,
            passed: 0,
            stop: None,
            contacts: vec![],
            grasped: vec![],
            opening: None,
            approach: vec![],
            depart: vec![],
            transit: vec![],
            giver: None,
        }
    }

    /// Passes the reach, and the turn that a pick, a place or a handover does not make.
    fn past_reach(&mut self) {
        self.passed += 2;
    }

    /// A try of this candidate of a turn from configuration `q` at its grasp, which IK found: past
    /// its reach, at its turn.
    fn tried_from(&self, q: &[f32]) -> Self {
        Self {
            task: self.task,
            index: self.index,
            world: self.world,
            transit_world: self.transit_world,
            transits: self.transits,
            hand: self.hand,
            start: self.start.clone(),
            rest: self.rest.clone(),
            poses: self.poses,
            passed: self.passed + 1,
            stop: None,
            contacts: vec![],
            grasped: q.to_vec(),
            opening: None,
            approach: vec![],
            depart: vec![],
            transit: vec![],
            giver: None,
        }
    }

    /// The hand that moves away: its own, or a handover's giver.
    fn away_hand(&self) -> usize {
        self.giver.as_ref().map_or(self.hand, |g| g.hand)
    }
}

/// The candidates that have got to `step`.
pub(crate) fn live(candidates: &mut [Candidate], step: TaskStep) -> Vec<&mut Candidate> {
    let passed = STEPS
        .iter()
        .position(|&s| s == step)
        .expect("every step is listed");
    candidates
        .iter_mut()
        .filter(|c| c.passed == passed)
        .collect()
}

/// `live` in groups by the holder of their task (`holder_of`), a group per holder.
pub(crate) fn by_holder<'a>(
    live: Vec<&'a mut Candidate>,
    holder_of: &[usize],
    holders: &[Device],
) -> Vec<Vec<&'a mut Candidate>> {
    let mut groups: Vec<Vec<&mut Candidate>> = holders.iter().map(|_| vec![]).collect();
    for c in live {
        groups[holder_of[c.task]].push(c);
    }
    groups
}

/// Whether each of `starting` starts clear of its task's transit world (`worlds_of`). One whose
/// task plans no transit passes unchecked.
pub(crate) fn start(
    device: &Device,
    worlds: &Worlds,
    starting: &mut [&mut Candidate],
    worlds_of: &[u32],
) -> Result<()> {
    let mut checked = vec![];
    for c in starting.iter_mut() {
        if c.transits {
            checked.push(c);
        } else {
            c.passed += 1;
        }
    }
    if checked.is_empty() {
        return Ok(());
    }
    let lists = checked
        .iter()
        .map(|c| (worlds_of[c.task], c.start.clone()))
        .collect();
    let free = first_free(device, worlds, lists)?;
    let mut colliding = vec![];
    for (c, free) in checked.into_iter().zip(free) {
        if free.is_some() {
            c.passed += 1;
        } else {
            let at = device.robot().ee_pose(&c.start, c.hand);
            c.stop = Some(ToolPathStop::Collision { at });
            colliding.push(c);
        }
    }
    let item_world: Vec<u32> = colliding.iter().map(|c| worlds_of[c.task]).collect();
    let q: Vec<f32> = colliding
        .iter()
        .flat_map(|c| c.start.iter().copied())
        .collect();
    let contacts = device.contacts(worlds, &item_world, &q, 0.0)?;
    for (c, contacts) in colliding.into_iter().zip(contacts) {
        c.contacts = contacts;
    }
    Ok(())
}

/// IK at each candidate's grasp or release in the world of `worlds` that `worlds_of` gives it,
/// nearest its rest.
pub(crate) fn reach(
    device: &Device,
    worlds: &Worlds,
    live: &mut [&mut Candidate],
    worlds_of: &dyn Fn(usize, &Candidate) -> u32,
    o: &IkOptions,
) -> Result<()> {
    if live.is_empty() {
        return Ok(());
    }
    let solved = solve(device, worlds, live, worlds_of, o)?;
    for (i, c) in live.iter_mut().enumerate() {
        if let Some(q) = solved.nearest(i, &c.rest) {
            c.grasped = q.to_vec();
            c.past_reach();
        }
    }
    classify(device, worlds, live, &solved, o)
}

/// IK at each candidate's grasp or release, in the world of `worlds` that `worlds_of` gives it,
/// seeded from its rest.
fn solve(
    device: &Device,
    worlds: &Worlds,
    live: &[&mut Candidate],
    worlds_of: &dyn Fn(usize, &Candidate) -> u32,
    o: &IkOptions,
) -> Result<IkResult> {
    let hands = device.robot().hands();
    let problems: Vec<IkProblem> = live
        .iter()
        .enumerate()
        .map(|(k, c)| IkProblem {
            world: worlds_of(k, c),
            targets: (0..hands)
                .map(|h| match &c.giver {
                    Some(giver) if giver.hand == h => Some(giver.at),
                    _ => (h == c.hand).then_some(c.poses[1]),
                })
                .collect(),
            seed: Some(c.rest.clone()),
            hold: vec![],
            holding: None,
        })
        .collect();
    ik::solve_ik(device, worlds, &problems, o)
}

/// Why IK found nothing for the candidates `solved` (their IK, problem by candidate) has no
/// solution for ([`ik::failures`]): a pose out of reach, or one reached only in collision, with
/// what collides.
fn classify(
    device: &Device,
    worlds: &Worlds,
    live: &mut [&mut Candidate],
    solved: &IkResult,
    o: &IkOptions,
) -> Result<()> {
    let failures = ik::failures(device, worlds, solved, o)?;
    for (c, failure) in live.iter_mut().zip(failures) {
        let at = c.poses[1];
        match failure {
            None => {}
            Some(IkFailure::Collides { contacts }) => {
                c.stop = Some(ToolPathStop::Collision { at });
                c.contacts = contacts;
            }
            Some(_) => c.stop = Some(ToolPathStop::Unreachable { at }),
        }
    }
    Ok(())
}

/// Problem `problem`'s collision-free solutions, nearest `q` first, as `[k, dof]`.
fn nearest_first(result: &IkResult, problem: usize, q: &[f32]) -> Vec<f32> {
    let distance = |item: usize| {
        let s = result.solution(item);
        s.iter().zip(q).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
    };
    let mut items: Vec<usize> = (problem * result.seeds..(problem + 1) * result.seeds)
        .filter(|&i| result.success[i])
        .collect();
    items.sort_by(|&a, &b| distance(a).total_cmp(&distance(b)));
    items
        .into_iter()
        .flat_map(|i| result.solution(i).iter().copied())
        .collect()
}

/// A search for the first collision-free configuration of a list: each as an edge that does not
/// move.
struct FirstFree {
    world: u32,
    configurations: Vec<f32>,
    n: usize,
    asked: bool,
    found: Option<usize>,
}

impl Search for FirstFree {
    fn request(&mut self, buffer: &mut Vec<f32>) -> Option<EdgeQuery> {
        if self.asked || self.configurations.is_empty() {
            return None;
        }
        self.asked = true;
        let mut edges = edges::reuse(buffer);
        for q in self.configurations.chunks(self.n) {
            edges.extend_from_slice(q);
            edges.extend_from_slice(q);
        }
        Some(EdgeQuery {
            world: self.world,
            edges,
            wanted: Some(Wanted::FirstValid),
            spacing: 1.0,
            keep: None,
            region: None,
        })
    }

    fn answer(&mut self, answer: usize) {
        self.found = (answer < self.configurations.len() / self.n).then_some(answer);
    }
}

/// The first collision-free configuration of each list (`[k, dof]` in its world of `worlds`),
/// checked on `device`'s threads as searches are: on the CPU, with no GPU call.
fn first_free(
    device: &Device,
    worlds: &Worlds,
    lists: Vec<(u32, Vec<f32>)>,
) -> Result<Vec<Option<usize>>> {
    let n = device.robot().dof();
    let mut searches: Vec<_> = lists
        .into_iter()
        .map(|(world, configurations)| {
            Deferred::new(None, move || FirstFree {
                world,
                configurations,
                n,
                asked: false,
                found: None,
            })
        })
        .collect();
    let mut running: Vec<&mut dyn Search> =
        searches.iter_mut().map(|s| s as &mut dyn Search).collect();
    device.search(worlds, &mut running, None)?;
    Ok(searches.into_iter().map(|s| s.into_inner().found).collect())
}

/// The straight moves.
#[derive(Clone, Copy)]
pub(crate) enum Leg {
    /// In to the grasp or release, followed backward from there and reversed.
    In,
    /// Away from it.
    Away,
}

/// Each candidate's straight move from its grasp or release, in the world of `worlds` that
/// `worlds_of` gives it: kept if it covers all of it, or at least `o.min_fraction` where an
/// obstacle stops it.
pub(crate) fn straight(
    device: &Device,
    worlds: &Worlds,
    live: &mut [&mut Candidate],
    worlds_of: &dyn Fn(usize, &Candidate) -> u32,
    leg: Leg,
    o: &TaskOptions,
) -> Result<()> {
    if live.is_empty() {
        return Ok(());
    }
    let n = device.robot().dof();
    let paths: Vec<ToolPath> = live
        .iter()
        .enumerate()
        .map(|(k, c)| ToolPath {
            world: worlds_of(k, c),
            start: c.grasped.clone(),
            hand: match leg {
                Leg::In => c.hand,
                Leg::Away => c.away_hand(),
            },
            waypoints: vec![match leg {
                Leg::In => c.poses[0],
                Leg::Away => c.poses[2],
            }],
            hold: match (&c.giver, leg) {
                (None, _) => vec![],
                (Some(giver), Leg::In) => giver.hold_in.clone(),
                (Some(giver), Leg::Away) => giver.hold_away.clone(),
            },
            holding: None,
        })
        .collect();
    let result = tool_path::follow(device, worlds, &paths, &o.tool_path)?;
    for (i, c) in live.iter_mut().enumerate() {
        let path = &result.paths[i];
        let fraction = result.fraction[i];
        let blocked = matches!(result.stopped[i], Some(ToolPathStop::Collision { .. }));
        if (fraction == 1.0 || blocked && fraction >= o.min_fraction) && !path.is_empty() {
            match leg {
                Leg::In => c.approach = path.chunks(n).rev().flatten().copied().collect(),
                Leg::Away => c.depart.clone_from(path),
            }
            c.passed += 1;
        } else {
            c.stop = result.stopped[i];
            c.contacts.clone_from(&result.contacts[i]);
        }
    }
    Ok(())
}

/// A planned path from each candidate's start to where its move in begins, the joints only other
/// hands move held where they rest there already, keeping its task's `constraints`. A candidate
/// whose move in begins where they break gets none. One whose task plans no transit passes
/// without one.
pub(crate) fn transit(
    device: &Device,
    worlds: &Worlds,
    live: &mut [&mut Candidate],
    constraints: &[&[HandConstraint]],
    o: &PlanOptions,
) -> Result<()> {
    let robot = device.robot();
    let n = robot.dof();
    let passing = live.iter_mut().filter(|c| !c.transits);
    for c in passing {
        c.passed += 1;
    }
    let keeping: Vec<usize> = (0..live.len())
        .filter(|&k| {
            let c = &live[k];
            c.transits && constraint::breach(robot, constraints[c.task], &c.approach[..n]).is_none()
        })
        .collect();
    if keeping.is_empty() {
        return Ok(());
    }
    let problems: Vec<PlanProblem> = keeping
        .iter()
        .map(|&k| {
            let c = &live[k];
            let idle = match &c.giver {
                Some(giver) => giver.idle.clone(),
                None => robot.idle_joints(c.hand),
            };
            PlanProblem {
                hold: idle
                    .into_iter()
                    .filter(|&j| c.rest[j] == c.start[j])
                    .collect(),
                constraints: constraints[c.task].to_vec(),
                ..PlanProblem::new(c.transit_world, c.start.clone(), &c.approach[..n])
            }
        })
        .collect();
    let result = trajopt::plan(device, worlds, &problems, o)?;
    for solved in result.solved() {
        let c = &mut live[keeping[solved.index]];
        c.transit = solved.solution.to_vec();
        c.passed += 1;
    }
    Ok(())
}

/// Each task's candidates that passed every step, timed, quickest first, with what they leave
/// held and placed; or where each candidate stopped.
pub(crate) fn choose(
    device: &Device,
    tasks: usize,
    candidates: &[Candidate],
    o: &TaskOptions,
    leaves: impl Fn(&Candidate) -> (Option<Held>, Option<Obstacle>) + Sync,
) -> Result<Vec<std::result::Result<Vec<TaskPlan>, TaskFailure>>> {
    let robot = device.robot();
    let complete: Vec<&Candidate> = candidates
        .iter()
        .filter(|c| c.passed == STEPS.len())
        .collect();
    // A robot holding an object keeps its joints and their limits, so the bare robot times every
    // motion. Timing is most of what is left to do, so the device's threads share it.
    let time = |c: &&Candidate| -> Result<TaskPlan> {
        // The cap on the straight moves; with `tool_speed_near`, it holds near the hand's
        // position at `q`.
        let speed = |hand: usize, q: &[f32]| HandSpeed {
            hand,
            linear: o.tool_speed,
            angular: f32::INFINITY,
            near: o.tool_speed_near.map(|radius| Near {
                point: robot.ee_pose(q, hand).position,
                radius,
            }),
        };
        let n = robot.dof();
        // The move in ends at the grasp or release, and the move away starts there.
        let (in_end, away_start) = (&c.approach[c.approach.len() - n..], &c.depart[..n]);
        let (held, placed) = leaves(c);
        Ok(TaskPlan {
            candidate: c.index,
            transit: match c.transits {
                true => Some(Trajectory::try_new(robot, &c.transit, 1.0)?),
                false => None,
            },
            approach: Trajectory::with_hand_speed(robot, &c.approach, &[speed(c.hand, in_end)])?,
            turn: None,
            pull: None,
            depart: Trajectory::with_hand_speed(
                robot,
                &c.depart,
                &[speed(c.away_hand(), away_start)],
            )?,
            held,
            placed,
            falling: None,
            opening: c.opening,
            meeting: c.giver.as_ref().map(|g| g.meeting),
        })
    };
    let timed: Vec<TaskPlan> =
        device.install(|| complete.par_iter().map(time).collect::<Result<_>>())?;
    let mut plans: Vec<Vec<TaskPlan>> = vec![vec![]; tasks];
    for (c, plan) in complete.iter().zip(timed) {
        plans[c.task].push(plan);
    }
    Ok(plans
        .into_iter()
        .enumerate()
        .map(|(t, mut plans)| {
            if plans.is_empty() {
                let mine = candidates.iter().filter(|c| c.task == t);
                return Err(TaskFailure {
                    candidates: mine
                        .map(|c| CandidateStop {
                            step: STEPS[c.passed],
                            why: c.stop,
                            contacts: c.contacts.clone(),
                            turned: None,
                            pulled: None,
                        })
                        .collect(),
                });
            }
            plans.sort_by(|a, b| a.duration().total_cmp(&b.duration()));
            Ok(plans)
        })
        .collect())
}

/// A candidate's world in a batch of one world per task.
fn by_task(_: usize, c: &Candidate) -> u32 {
    c.task as u32
}

/// The links of hand `hand` that may touch what it grasps: its fingers, or the whole hand if it
/// has none.
pub(crate) fn fingers(robot: &Robot, hand: usize) -> Vec<&str> {
    let fingers = robot.finger_links(hand);
    if fingers.is_empty() {
        robot.hand_links(hand)
    } else {
        fingers
    }
}

/// What `task`'s grasp `index` leaves the hand holding.
fn held(task: &Pick, worlds: &Worlds, index: usize, spheres: SphereOptions) -> Held {
    let part = &worlds[task.world as usize].obstacles[task.object];
    let (shape, grasp) = match &task.body {
        Some(body) => (
            &body.shape,
            body.shape
                .pose()
                .rigid()
                .inverse()
                .mul_pose(part.pose().rigid().mul_pose(task.grasps[index].rigid())),
        ),
        None => (part, task.grasps[index]),
    };
    Held {
        locked: task.locked.get(index).cloned().unwrap_or_default(),
        hand: task.hand,
        object: shape.placed(Pose::IDENTITY),
        grasp,
        spheres,
    }
}

/// `pose` moved `distance` back along its own z axis.
pub(crate) fn behind(pose: Pose, distance: f32) -> Pose {
    Pose::new(
        pose.position - pose.rotation * Vec3::Z * distance,
        pose.rotation,
    )
}

/// That task `i` has `poses` (its grasps or poses, called `what`), each finite with a unit
/// rotation.
pub(crate) fn check_poses(i: usize, what: &str, poses: &[Pose]) -> Result<()> {
    ensure_input!(!poses.is_empty(), "task {i} has no {what}s");
    if let Some((k, p)) = poses.iter().enumerate().find(|(_, p)| !p.is_unit()) {
        return Err(input!(
            "task {i}: {what} {k} must be finite with a unit rotation, got {p:?}"
        ));
    }
    Ok(())
}

pub(crate) fn check_options(o: &TaskOptions) -> Result<()> {
    o.held_spheres.check().map_err(crate::Error::Input)?;
    ensure_input!(
        o.tool_speed > 0.0,
        "the tool speed must be positive, got {}",
        o.tool_speed
    );
    if let Some(radius) = o.tool_speed_near {
        ensure_input!(
            radius.is_finite() && radius > 0.0,
            "tool_speed_near must be a positive distance, got {radius} m"
        );
    }
    ensure_input!(
        o.min_fraction > 0.0 && o.min_fraction <= 1.0,
        "min_fraction must be in (0, 1], got {}",
        o.min_fraction
    );
    Ok(())
}

/// The world each step of a task runs in, and whether it plans its transit.
pub(crate) struct Steps<'a> {
    pub(crate) world: u32,
    pub(crate) transit_world: u32,
    pub(crate) depart_world: u32,
    pub(crate) transit: bool,
    pub(crate) constraints: &'a [HandConstraint],
}

impl<'a> Steps<'a> {
    fn of_pick(t: &'a Pick) -> Self {
        Self {
            world: t.world,
            transit_world: t.transit_world.unwrap_or(t.world),
            depart_world: t.depart_world.unwrap_or(t.world),
            transit: t.transit,
            constraints: &t.constraints,
        }
    }

    fn of_place(t: &'a Place) -> Self {
        Self {
            world: t.world,
            transit_world: t.transit_world.unwrap_or(t.world),
            depart_world: t.depart_world.unwrap_or(t.world),
            transit: t.transit,
            constraints: &t.constraints,
        }
    }

    fn of_turn(t: &'a Turn) -> Self {
        Self {
            world: t.world,
            transit_world: t.transit_world.unwrap_or(t.world),
            depart_world: t.world,
            transit: t.transit,
            constraints: &t.constraints,
        }
    }
}

/// Task `i`'s world, after checking its steps' worlds, its start and its hand.
pub(crate) fn check_task<'a>(
    robot: &Robot,
    worlds: &'a Worlds,
    i: usize,
    steps: &Steps<'_>,
    start: &[f32],
    hand: usize,
) -> Result<&'a World> {
    let world = steps.world;
    let w = worlds
        .get(world as usize)
        .ok_or_else(|| input!("task {i}: world {world} of {}", worlds.len()))?;
    for (step, other) in [
        ("transit", steps.transit_world),
        ("depart", steps.depart_world),
    ] {
        let o = worlds
            .get(other as usize)
            .ok_or_else(|| input!("task {i}: {step} world {other} of {}", worlds.len()))?;
        ensure_input!(
            o.obstacles.len() == w.obstacles.len(),
            "task {i}: {step} world {other} has {} obstacles and world {world} {}; a step's \
             world holds the same obstacles, in the same order",
            o.obstacles.len(),
            w.obstacles.len()
        );
    }
    ensure_input!(
        start.len() == robot.dof(),
        "task {i}: a start of {} joints, not {}",
        start.len(),
        robot.dof()
    );
    ensure_input!(
        start.iter().all(|v| v.is_finite()),
        "task {i}: the start {start:?} is not finite"
    );
    ensure_input!(
        robot.within(start),
        "task {i}: the start {start:?} leaves the joint limits"
    );
    ensure_input!(
        hand < robot.hands(),
        "task {i}: hand {hand} of {}",
        robot.hands()
    );
    constraint::check(robot, steps.constraints, &format!("task {i}"))?;
    if steps.transit
        && let Some(why) = constraint::breach(robot, steps.constraints, start)
    {
        return Err(input!("task {i}: the start breaks its constraints: {why}"));
    }
    Ok(w)
}

fn check_support(i: usize, world: &World, support: &[usize], object: Option<usize>) -> Result<()> {
    if let Some(&s) = support
        .iter()
        .find(|&&s| s >= world.obstacles.len() || Some(s) == object)
    {
        return Err(input!(
            "task {i}: support {s} must be another of the world's {} obstacles",
            world.obstacles.len()
        ));
    }
    Ok(())
}

/// A hand drawing a handle along a line. The contact obstacle stays in the planning world. A
/// pull checks the arm's motion around the fixture, including the move away after release. The
/// drawer's own sweep and stops from measured force are outside the kinematic plan.
#[derive(Clone, Debug)]
pub struct Pull {
    /// World in which the hand grips, draws and lets go.
    pub world: u32,
    /// Optional world for the start and transit, with the same obstacles in the same order.
    pub transit_world: Option<u32>,
    /// Where the robot starts, at rest.
    pub start: Vec<f32>,
    /// Whether to check the start and plan the transit to the approach.
    pub transit: bool,
    /// The hand that pulls.
    pub hand: usize,
    /// Contact obstacle by index. Only fingers may touch it during the approach, draw and
    /// departure. It can be a distance grid, as the fixture stays in place.
    pub object: usize,
    /// Candidate hand IK poses in the contact obstacle's frame ([`Obstacle::pose`]).
    pub grasps: Vec<Pose>,
    /// Distance back along the hand's z axis before the grasp and after release (meters).
    pub approach: f32,
    /// The line's unit direction in the world's frame, independent of the chosen grasp.
    pub direction: Vec3,
    /// Signed nonzero distance along the line (meters).
    pub distance: f32,
    /// Constraints to keep along the transit, as a pick's.
    pub constraints: Vec<HandConstraint>,
    /// Locked joints for each grasp, as [`Pick::locked`].
    pub locked: Vec<HashMap<String, f32>>,
    /// How many plans will do, as [`Pick::enough`].
    pub enough: Option<usize>,
}

impl Pull {
    /// Hand 0 grips `object` by any of `grasps`, draws it along the world direction by
    /// `distance` meters, then lets go. The approach and departure each back off 10 cm.
    #[must_use]
    pub fn new(
        world: u32,
        start: impl Into<Vec<f32>>,
        object: usize,
        grasps: Vec<Pose>,
        direction: Vec3,
        distance: f32,
    ) -> Self {
        Self {
            world,
            transit_world: None,
            start: start.into(),
            transit: true,
            hand: 0,
            object,
            grasps,
            approach: 0.1,
            direction,
            distance,
            constraints: vec![],
            locked: vec![],
            enough: None,
        }
    }
}

/// A hand turning one obstacle of its world about a line, as a knob, a cap, a lever or a valve's
/// wheel turns.
#[derive(Clone, Debug)]
pub struct Turn {
    /// The world's index in the batch: the one the hand grips, turns and lets go in, and the
    /// transit's unless `transit_world` names another.
    pub world: u32,
    /// The world the start is checked and the transit planned in, where it is another: as many
    /// obstacles as `world` has, in its order, shaped or touched ([`World::touch`]) otherwise.
    pub transit_world: Option<u32>,
    /// Where the robot starts, at rest.
    pub start: Vec<f32>,
    /// Whether the path from the start to where the move in begins is planned: true. Without it
    /// the plans hold the moves in and away and the turn alone ([`TaskPlan::begin`] is where a way
    /// to them ends), and the start, which is not checked for collision, only says which
    /// configuration at a grasp is nearest and where the joints other hands move wait.
    pub transit: bool,
    /// The hand that turns it.
    pub hand: usize,
    /// The object: an obstacle of the world, by index. A distance grid may only be gripped where it
    /// `stays`.
    pub object: usize,
    /// Poses of the hand's IK frame that grip the object, in the object's frame (its center and
    /// rotation), as [`Pick::grasps`]; any one will do.
    pub grasps: Vec<Pose>,
    /// How far back along its own z axis the hand begins to move in to a grasp, and moves away to
    /// once it lets go (meters).
    pub approach: f32,
    /// The line the object turns about and how far it moves along it as it turns, in the world's
    /// frame: a knob's or a wheel's axis, a lever's hinge, a cap's axis and its thread's pitch.
    /// None is the hand's own z axis through its IK frame at the grasp, without a pitch: a turn in
    /// place, as a knob gripped on its axis turns.
    pub screw: Option<Screw>,
    /// How far it turns (radians), right-handed about the screw's direction, or about the hand's
    /// z axis where there is no screw: not 0.
    pub angle: f32,
    /// Obstacles the object may touch as it turns, by index: what it turns in or on, as a
    /// lever's mount, a cap's bottle or the panel a knob sits in, as [`Pick::support`] is what a
    /// picked object may touch as it leaves. Not read where the object `stays`.
    pub support: Vec<usize>,
    /// Whether the obstacle stays where it is while the hand turns: false, the whole obstacle
    /// turns, held by the hand through the turn and checked where it sweeps against everything
    /// else of the world but `support`. True where the world holds the part that turns and what
    /// it turns in as one obstacle, as a knob a camera could not tell from its panel: then the
    /// obstacle stays where it is, the fingers alone may touch it, and nothing checks the part's
    /// sweep.
    pub stays: bool,
    /// Where the hands must stay along the transit ([`HandConstraint`]), as a pick's. The start
    /// must keep them; a grasp whose move in begins where they break stops at the transit.
    pub constraints: Vec<HandConstraint>,
    /// Per grasp, joints the robot was loaded with locked, at the values they keep through the
    /// task by that grasp, as [`Pick::locked`]: the fingers opened only as far as the grasp has
    /// room for ([`Grasp::opening`](crate::Grasp::opening)). A call's grasps are grouped by these,
    /// a relocked robot for each group. Empty: every grasp as loaded.
    pub locked: Vec<HashMap<String, f32>>,
    /// How many plans will do, as [`Pick::enough`]: `None`, every grasp is tried; with a count,
    /// the grasps are tried in their order, 32 and then twice as many each time, until that many
    /// have got through.
    pub enough: Option<usize>,
}

impl Turn {
    /// Hand 0 turns `object` of world `world` by `angle` radians about its own z axis, from `start`
    /// by any of `grasps`, moving in to the grasp and away from it 10 cm along that axis, the
    /// object turning and touching nothing as it turns.
    #[must_use]
    pub fn new(
        world: u32,
        start: impl Into<Vec<f32>>,
        object: usize,
        grasps: Vec<Pose>,
        angle: f32,
    ) -> Self {
        Self {
            world,
            transit_world: None,
            start: start.into(),
            transit: true,
            hand: 0,
            object,
            grasps,
            approach: 0.1,
            screw: None,
            angle,
            support: vec![],
            stays: false,
            constraints: vec![],
            locked: vec![],
            enough: None,
        }
    }
}

/// Turns each task's object by its angle about its screw, trying each of its grasps. IK solves at
/// the grasp, where the fingers ([`Robot::finger_links`], or the hand's links if it has none
/// and the target is a primitive) may touch the object, and the turn is followed from every
/// configuration it finds there, nearest the start first. The hand's IK frame is carried about the
/// screw ([`Screw::path`]) and kept within the tool path's precision of its true arc: half of it
/// for how far the straight moves between the screw's waypoints sag from the arc, half for how far
/// the hand strays from them. A grasp gets through where some configuration at it makes the whole
/// turn within the joints' ranges and clear of the world, then the moves in and away: the nearest
/// such is its plan's. So a grasp is not lost where the configuration nearest the start runs into a
/// joint limit halfway and another at the same grasp makes the turn; where none does, its stop says
/// how far the one that turned farthest got ([`CandidateStop::turned`]).
/// [`TaskOptions::min_fraction`] holds for the moves in and away; the turn is made whole or not at
/// all.
///
/// The hand holds the object through the turn, as a pick's lift holds what it picked up: the
/// object is part of the robot there, and its world is the task's without it, so what the
/// object sweeps is checked against everything else of the world but its `support`, and against
/// the arm. A task whose obstacle `stays` turns with the obstacle where it is instead, the
/// fingers alone free to touch it, and nothing checks the part's sweep. Through the reach and
/// the move in the fingers may touch the object, and nothing else of the robot may; the move
/// away goes back along the hand's z axis from where the turn ends, the fingers free to touch
/// the object where the turn leaves it ([`TaskPlan::placed`]). The turn is timed with the hand
/// under [`TaskOptions::tool_speed`] all along. Run the transit and the approach, close the
/// gripper, run the turn, open the gripper, then run the departure, and put `placed` in the world
/// in the object's place.
pub fn turn(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Turn],
    o: &TaskOptions,
) -> Result<TaskResult<Turn>> {
    let corrected = device.robot().correct_starts(
        tasks,
        o.plan.start_tolerance,
        "task",
        |t| &t.start,
        |t| &mut t.start,
    )?;
    articulate(device, worlds, corrected.as_ref(), o)
}

/// Plans each pull from the start through the approach, draw and departure. As in [`turn`], the
/// planner tries every grasp configuration before choosing one. The draw follows a single
/// straight segment with constant hand orientation, to the task's signed distance along the
/// world direction. The whole draw must succeed. A stop during it reports signed meters in
/// [`CandidateStop::pulled`]. The task's contact obstacle stays in the world for each stage.
/// Close the gripper after the approach and open it after [`TaskPlan::pull`].
pub fn pull(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Pull],
    o: &TaskOptions,
) -> Result<TaskResult<Pull>> {
    let corrected = device.robot().correct_starts(
        tasks,
        o.plan.start_tolerance,
        "task",
        |t| &t.start,
        |t| &mut t.start,
    )?;
    articulate(device, worlds, corrected.as_ref(), o)
}

/// One constrained motion of a gripped part, in the world frame.
#[derive(Clone, Copy)]
enum Motion {
    Turn { screw: Option<Screw>, angle: f32 },
    Pull { direction: Vec3, distance: f32 },
}

impl Motion {
    fn amount(self) -> f32 {
        match self {
            Self::Turn { angle, .. } => angle,
            Self::Pull { distance, .. } => distance,
        }
    }

    fn check(self, i: usize) -> Result<()> {
        match self {
            Self::Turn { screw, angle } => {
                if let Some(screw) = screw {
                    screw.check().map_err(|e| input!("task {i}: {e}"))?;
                }
                ensure_input!(
                    angle.is_finite() && angle != 0.0,
                    "task {i} turns by a finite angle other than 0, got {angle} rad"
                );
            }
            Self::Pull {
                direction,
                distance,
            } => {
                ensure_input!(
                    direction.is_finite() && (direction.length() - 1.0).abs() < 1e-3,
                    "task {i} needs a unit pull direction, got {direction}"
                );
                ensure_input!(
                    distance.is_finite() && distance != 0.0,
                    "task {i} needs a finite nonzero pull distance, got {distance} m"
                );
            }
        }
        Ok(())
    }

    /// A pull uses the measured finger links alone. A turn keeps its grasp convention for a
    /// robot whose hand has no finger links when it grips a primitive.
    fn contact(self, robot: &Robot, hand: usize, object: usize, shape: &Obstacle) -> Touch {
        let links = match (self, shape) {
            (Self::Pull { .. }, _) | (_, Obstacle::Sdf { .. }) => robot.finger_links(hand),
            (Self::Turn { .. }, _) => fingers(robot, hand),
        };
        Touch::new(object, links)
    }

    /// At this grasp's IK pose, resolve a turn about the tool's own axis.
    fn screw(screw: Option<Screw>, at: Pose) -> Screw {
        screw.unwrap_or_else(|| Screw::about(at.position, (at.rotation * Vec3::Z).normalize()))
    }
}

/// Inputs shared by turning and pulling once a group of grasps has its fingers relocked.
struct Articulation<'a> {
    steps: Steps<'a>,
    start: &'a [f32],
    hand: usize,
    object: usize,
    grasps: &'a [Pose],
    locked: &'a [HashMap<String, f32>],
    support: &'a [usize],
    approach: f32,
    stays: bool,
    motion: Motion,
}

trait Articulated: Gripping {
    fn articulation(&self) -> Articulation<'_>;
}

impl Articulated for Turn {
    fn articulation(&self) -> Articulation<'_> {
        Articulation {
            steps: Steps::of_turn(self),
            start: &self.start,
            hand: self.hand,
            object: self.object,
            grasps: &self.grasps,
            locked: &self.locked,
            support: &self.support,
            approach: self.approach,
            stays: self.stays,
            motion: Motion::Turn {
                screw: self.screw,
                angle: self.angle,
            },
        }
    }
}

impl Articulated for Pull {
    fn articulation(&self) -> Articulation<'_> {
        Articulation {
            steps: Steps {
                world: self.world,
                transit_world: self.transit_world.unwrap_or(self.world),
                depart_world: self.world,
                transit: self.transit,
                constraints: &self.constraints,
            },
            start: &self.start,
            hand: self.hand,
            object: self.object,
            grasps: &self.grasps,
            locked: &self.locked,
            support: &[],
            approach: self.approach,
            stays: true,
            motion: Motion::Pull {
                direction: self.direction,
                distance: self.distance,
            },
        }
    }
}

fn articulate<T: Articulated>(
    device: &Device,
    worlds: &Worlds,
    tasks: &[T],
    o: &TaskOptions,
) -> Result<TaskResult<T>> {
    check_options(o)?;
    let robot = device.robot();
    let mut checked = vec![];
    for (i, task) in tasks.iter().enumerate() {
        let t = task.articulation();
        let world = check_task(robot, worlds, i, &t.steps, t.start, t.hand)?;
        if t.stays {
            ensure_input!(
                t.object < world.obstacles.len(),
                "task {i}: obstacle {} of {}",
                t.object,
                world.obstacles.len()
            );
        } else {
            check_object(world, i, t.object, "turned")?;
        }
        check_poses(i, "grasp", t.grasps)?;
        ensure_input!(
            t.approach.is_finite() && t.approach > 0.0,
            "task {i} needs a positive approach, got {} m",
            t.approach
        );
        t.motion.check(i)?;
        if let Motion::Pull { distance, .. } = t.motion {
            o.tool_path.check()?;
            let steps = (distance.abs() / o.tool_path.step).ceil() as usize;
            Vec::<(Pose, f32)>::new()
                .try_reserve_exact(steps.saturating_add(1))
                .map_err(|_| input!("task {i}: a pull of {distance} m at steps of {} m needs more poses than can be held", o.tool_path.step))?;
        }
        check_support(i, world, t.support, Some(t.object))?;
        check_tries(robot, i, task, &mut checked)?;
        let object = world.obstacles[t.object].pose().rigid();
        for (g, &grasp) in t.grasps.iter().enumerate() {
            let at = object.mul_pose(grasp.rigid());
            match t.motion {
                Motion::Turn { screw, angle } => {
                    Motion::screw(screw, at)
                        .count(at, angle, o.tool_path.precision / 2.0)
                        .map_err(|e| input!("task {i}, grasp {g}: {e}"))?;
                }
                Motion::Pull {
                    direction,
                    distance,
                } => {
                    let end = at.position + direction.normalize() * distance;
                    ensure_input!(
                        at.is_unit() && end.distance(at.position).is_finite(),
                        "task {i}, grasp {g}: drawing from {at:?} by {distance} m along {direction} ends at {end}"
                    );
                }
            }
        }
    }
    by_grasps(device, worlds, tasks, o, |device, worlds, group, o| {
        let motions: Vec<_> = group.iter().map(Articulated::articulation).collect();
        let outcomes = articulated(device, worlds, &motions, o)?;
        Ok(TaskResult {
            problems: group.to_vec(),
            outcomes,
        })
    })
}

/// How a turn is followed: at half the tool path's precision, the other half being for how far
/// the straight moves between the screw's waypoints sag from the arc. The two add, and neither is
/// held tighter than the other.
fn turning_options(o: &TaskOptions) -> ToolPathOptions {
    ToolPathOptions {
        precision: o.tool_path.precision / 2.0,
        ..o.tool_path.clone()
    }
}

/// Checked turn or pull tasks whose grasps keep the device robot's locked joints.
fn articulated(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Articulation<'_>],
    o: &TaskOptions,
) -> Result<Vec<std::result::Result<Vec<TaskPlan>, TaskFailure>>> {
    let robot = device.robot();
    let n = robot.dof();
    let pull = tasks
        .first()
        .is_some_and(|t| matches!(t.motion, Motion::Pull { .. }));
    let along = if pull {
        o.tool_path.clone()
    } else {
        turning_options(o)
    };
    // Per grasp of each task, in order: a candidate, the waypoints its turn follows and the
    // object as the turn leaves it.
    let mut candidates = vec![];
    let mut ways: Vec<Vec<Pose>> = vec![];
    let mut placed: Vec<Obstacle> = vec![];
    let mut first = vec![];
    for (t, task) in tasks.iter().enumerate() {
        first.push(candidates.len());
        let object = &worlds[task.steps.world as usize].obstacles[task.object];
        for (index, &grasp) in task.grasps.iter().enumerate() {
            let at = object.pose().rigid().mul_pose(grasp.rigid());
            let (end, path, left) = match task.motion {
                Motion::Turn { screw, angle } => {
                    let screw = Motion::screw(screw, at);
                    let left = if task.stays {
                        object.clone()
                    } else {
                        object.placed(screw.carry(object.pose().rigid(), angle))
                    };
                    (
                        screw.carry(at, angle),
                        screw.path(at, angle, along.precision)?,
                        left,
                    )
                }
                Motion::Pull {
                    direction,
                    distance,
                } => {
                    let end =
                        Pose::new(at.position + direction.normalize() * distance, at.rotation);
                    (end, vec![end], object.clone())
                }
            };
            let poses = [behind(at, task.approach), at, behind(end, task.approach)];
            candidates.push(Candidate::new(
                t,
                index,
                &task.steps,
                task.hand,
                task.start,
                poses,
            ));
            ways.push(path);
            placed.push(left);
        }
    }
    let id = |c: &Candidate| first[c.task] + c.index;

    // Starts clear of everything, the object included.
    let worlds_of: Vec<u32> = tasks.iter().map(|t| t.steps.transit_world).collect();
    start(
        device,
        worlds,
        &mut live(&mut candidates, TaskStep::Start),
        &worlds_of,
    )?;

    // Each configuration IK finds at a grasp, the fingers touching the object, is a try of its
    // own, nearest the start first.
    let mut tries: Vec<Candidate> = vec![];
    let mut reaching = live(&mut candidates, TaskStep::Reach);
    let touching: Vec<World> = tasks
        .iter()
        .map(|t| {
            let mut w = worlds[t.steps.world as usize].clone();
            w.touch.push(
                t.motion
                    .contact(robot, t.hand, t.object, &w.obstacles[t.object]),
            );
            w
        })
        .collect();
    let touching = device.upload(&touching)?;
    if !reaching.is_empty() {
        let solved = solve(device, &touching, &reaching, &by_task, &o.ik)?;
        for (k, c) in reaching.iter().enumerate() {
            let found = nearest_first(&solved, k, &c.rest);
            tries.extend(found.chunks(n).map(|q| c.tried_from(q)));
        }
        classify(device, &touching, &mut reaching, &solved, &o.ik)?;
    }

    // Every try's turn: where it is whole, its motion (`made`); where not, how far it got.
    let path_of = |c: &Candidate| ToolPath {
        world: c.task as u32,
        start: c.grasped.clone(),
        hand: c.hand,
        waypoints: ways[id(c)].clone(),
        hold: vec![],
        holding: None,
    };
    let mut made: Vec<Vec<f32>> = vec![vec![]; tries.len()];
    let mut progress: Vec<Option<f32>> = vec![None; tries.len()];
    let mut take = |i: usize, c: &mut Candidate, r: &mut ToolPathResult, k: usize| {
        if r.fraction[k] == 1.0 && !r.paths[k].is_empty() {
            made[i] = std::mem::take(&mut r.paths[k]);
            c.passed += 1;
        } else {
            c.stop = r.stopped[k];
            c.contacts = std::mem::take(&mut r.contacts[k]);
            progress[i] = Some(r.fraction[k] * tasks[c.task].motion.amount());
        }
    };
    // Where the obstacle stays, the fingers turn in it, every such try one batch.
    let (staying, holding): (Vec<usize>, Vec<usize>) =
        (0..tries.len()).partition(|&i| tasks[tries[i].task].stays);
    if !staying.is_empty() {
        let paths: Vec<ToolPath> = staying.iter().map(|&i| path_of(&tries[i])).collect();
        let mut followed = tool_path::follow(device, &touching, &paths, &along)?;
        for (k, &i) in staying.iter().enumerate() {
            take(i, &mut tries[i], &mut followed, k);
        }
    }
    // Where it turns, the hand holds it as the grasp does, in its task's world without it,
    // touching the task's support: the tries of each grasp a batch on a device holding it, built
    // and run as a pick's lifts are, the object's spheres fitted once.
    let mut grips: Vec<(Held, Vec<usize>)> = vec![];
    for i in holding {
        let c = &tries[i];
        match grips.last_mut() {
            Some((_, group)) if id(&tries[group[0]]) == id(c) => group.push(i),
            _ => {
                let t = &tasks[c.task];
                let held = Held {
                    locked: t.locked.get(c.index).cloned().unwrap_or_default(),
                    hand: t.hand,
                    object: worlds[t.steps.world as usize].obstacles[t.object]
                        .placed(Pose::IDENTITY),
                    grasp: t.grasps[c.index],
                    spheres: o.held_spheres,
                };
                grips.push((held, vec![i]));
            }
        }
    }
    if !grips.is_empty() {
        let helds: Vec<&Held> = grips.iter().map(|(held, _)| held).collect();
        let fitted = Fitted::new(device, robot, &helds)?;
        // One upload per hand: what one hand holds is the link its worlds name.
        let without = |hand: usize| {
            tasks
                .iter()
                .map(|t| {
                    let mut w = worlds[t.steps.world as usize].clone();
                    w.remove(t.object)?;
                    if t.hand == hand {
                        let shifted = |s: usize| if s > t.object { s - 1 } else { s };
                        let touches = t.support.iter().map(|&s| Touch::held(shifted(s), hand));
                        w.touch.extend(touches);
                    }
                    Ok(w)
                })
                .collect::<Result<Vec<World>>>()
        };
        let mut uploads: Vec<(usize, Worlds)> = vec![];
        for (held, _) in &grips {
            if uploads.iter().all(|(hand, _)| *hand != held.hand) {
                uploads.push((held.hand, device.upload(&without(held.hand)?)?));
            }
        }
        let each = device.in_turns(&mut grips, |k, (held, group)| {
            let (_, uploaded) = uploads
                .iter()
                .find(|(hand, _)| *hand == held.hand)
                .expect("every hand that turns an object has its worlds");
            // Built where it is used, so that building one overlaps the other grasps' turns.
            let holder = fitted.holder(device, robot, k, held)?;
            let paths: Vec<ToolPath> = group.iter().map(|&i| path_of(&tries[i])).collect();
            tool_path::follow(&holder, uploaded, &paths, &along)
        })?;
        for ((_, group), mut followed) in grips.iter().zip(each) {
            for (k, &i) in group.iter().enumerate() {
                take(i, &mut tries[i], &mut followed, k);
                // Contacts name obstacles as the task's world does, which still holds the object.
                let object = tasks[tries[i].task].object;
                for contact in &mut tries[i].contacts {
                    if let Contact::World { obstacle, .. } = contact
                        && *obstacle >= object
                    {
                        *obstacle += 1;
                    }
                }
            }
        }
    }

    // In to the grasp from each configuration that makes the turn, then away from where it ends,
    // in a world per grasp with the object where the turn leaves it.
    let mut moving = live(&mut tries, TaskStep::Approach);
    straight(device, &touching, &mut moving, &by_task, Leg::In, o)?;
    let departing = STEPS.iter().position(|&s| s == TaskStep::Depart);
    let departing = departing.expect("every step is listed");
    for (c, made) in tries.iter_mut().zip(&made) {
        if c.passed == departing {
            c.grasped = made[made.len() - n..].to_vec();
        }
    }
    let mut leaving = live(&mut tries, TaskStep::Depart);
    if !leaving.is_empty() {
        let mut grasps: Vec<usize> = leaving.iter().map(|c| id(c)).collect();
        grasps.dedup();
        let left: Vec<World> = grasps
            .iter()
            .map(|&g| {
                let c = &candidates[g];
                let t = &tasks[c.task];
                let mut w = worlds[t.steps.world as usize].clone();
                w.obstacles[t.object] = placed[g].clone();
                w.touch.push(
                    t.motion
                        .contact(robot, t.hand, t.object, &w.obstacles[t.object]),
                );
                w
            })
            .collect();
        let left = device.upload(&left)?;
        let in_its = |_: usize, c: &Candidate| {
            let g = grasps.binary_search(&id(c));
            g.expect("every grasp moving away has its world") as u32
        };
        straight(device, &left, &mut leaving, &in_its, Leg::Away, o)?;
    }

    // Each grasp's try: the first, nearest the start, that is through the move away; or where
    // none is, the one that got farthest, nearest first, and at the turn the one that turned
    // farthest.
    let mut chosen: Vec<Option<usize>> = vec![None; candidates.len()];
    let far = |i: usize| progress[i].unwrap_or(0.0).abs();
    let mut tries = tries.into_iter().enumerate().peekable();
    while let Some((mut b, mut best)) = tries.next() {
        let g = id(&best);
        while let Some((i, next)) = tries.next_if(|(_, c)| id(c) == g) {
            if next.passed > best.passed || next.passed == best.passed && far(i) > far(b) {
                (b, best) = (i, next);
            }
        }
        candidates[g] = best;
        chosen[g] = Some(b);
    }

    // From the start, around the object as it stands.
    let constraints: Vec<&[HandConstraint]> = tasks.iter().map(|t| t.steps.constraints).collect();
    transit(
        device,
        worlds,
        &mut live(&mut candidates, TaskStep::Transit),
        &constraints,
        &o.plan,
    )?;

    let mut outcomes = choose(device, tasks.len(), &candidates, o, |c| {
        (None, (!pull).then(|| placed[id(c)].clone()))
    })?;
    // Each plan's turn, timed with the hand under the tool speed all along it, which the hand
    // holds the object through, the plans then quickest first again; and how far each stop at the
    // turn got.
    let mut plans: Vec<(usize, &mut TaskPlan)> = vec![];
    for (t, outcome) in outcomes.iter_mut().enumerate() {
        match outcome {
            Ok(found) => plans.extend(found.iter_mut().map(|plan| (t, plan))),
            Err(failure) => {
                for (k, stop) in failure.candidates.iter_mut().enumerate() {
                    let tried = chosen[first[t] + k];
                    if stop.step == TaskStep::Turn {
                        let progress = tried.and_then(|i| progress[i]);
                        if pull {
                            stop.step = TaskStep::Pull;
                            stop.pulled = progress;
                        } else {
                            stop.turned = progress;
                        }
                    }
                }
            }
        }
    }
    let timed = |(t, plan): &mut (usize, &mut TaskPlan)| -> Result<()> {
        let tried = chosen[first[*t] + plan.candidate].expect("a plan's grasp has its try");
        let cap = HandSpeed {
            hand: tasks[*t].hand,
            linear: o.tool_speed,
            angular: f32::INFINITY,
            near: None,
        };
        let trajectory = Trajectory::with_hand_speed(robot, &made[tried], &[cap])?;
        if pull {
            plan.pull = Some(Box::new(trajectory));
        } else {
            plan.turn = Some(trajectory);
        }
        Ok(())
    };
    device.install(|| plans.par_iter_mut().try_for_each(timed))?;
    let succeeded = outcomes.iter_mut().filter_map(|o| o.as_mut().ok());
    for found in succeeded {
        found.sort_by(|a, b| a.duration().total_cmp(&b.duration()));
    }
    Ok(outcomes)
}

/// A task that grips its object by any of its grasps, each keeping the robot's locked joints at
/// values of its own, tried as [`Pick::enough`] says: a pick or a turn.
trait Gripping: Clone + Send + Sync {
    fn gripped_by(&self) -> &[Pose];
    fn locks(&self) -> &[HashMap<String, f32>];
    fn plans_wanted(&self) -> Option<usize>;
    /// The task by its grasps `grasps` alone, each keeping the locked joints at `locked`.
    fn by(&self, grasps: &[usize], locked: &HashMap<String, f32>) -> Self;
}

impl Gripping for Pick {
    fn gripped_by(&self) -> &[Pose] {
        &self.grasps
    }

    fn locks(&self) -> &[HashMap<String, f32>] {
        &self.locked
    }

    fn plans_wanted(&self) -> Option<usize> {
        self.enough
    }

    fn by(&self, grasps: &[usize], locked: &HashMap<String, f32>) -> Self {
        Pick {
            grasps: grasps.iter().map(|&g| self.grasps[g]).collect(),
            locked: vec![locked.clone(); grasps.len()],
            ..self.clone()
        }
    }
}

impl Gripping for Turn {
    fn gripped_by(&self) -> &[Pose] {
        &self.grasps
    }

    fn locks(&self) -> &[HashMap<String, f32>] {
        &self.locked
    }

    fn plans_wanted(&self) -> Option<usize> {
        self.enough
    }

    fn by(&self, grasps: &[usize], locked: &HashMap<String, f32>) -> Self {
        Turn {
            grasps: grasps.iter().map(|&g| self.grasps[g]).collect(),
            locked: vec![locked.clone(); grasps.len()],
            ..self.clone()
        }
    }
}
impl Gripping for Pull {
    fn gripped_by(&self) -> &[Pose] {
        &self.grasps
    }

    fn locks(&self) -> &[HashMap<String, f32>] {
        &self.locked
    }

    fn plans_wanted(&self) -> Option<usize> {
        self.enough
    }

    fn by(&self, grasps: &[usize], locked: &HashMap<String, f32>) -> Self {
        Pull {
            grasps: grasps.iter().map(|&g| self.grasps[g]).collect(),
            locked: vec![locked.clone(); grasps.len()],
            ..self.clone()
        }
    }
}

/// What [`place_nearest`] chose: the alternative that sets the object down nearest the spot, and
/// its plan that does.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct NearestPlace {
    /// The task's index among the alternatives.
    pub task: usize,
    /// The task as given, its start corrected within [`PlanOptions::start_tolerance`] as
    /// [`place`] corrects it.
    pub problem: Place,
    /// Its plan that sets the object down nearest the spot, its candidate numbered as the task's
    /// poses are.
    pub plan: TaskPlan,
}

/// Alternatives for one placement: the same object set down at the same poses from different
/// starts or holds, as each of several picks leaves it. Plans them in one batch, each as [`place`]
/// does, and returns the alternative whose plan sets the object down nearest `near` in x and y
/// (the support sets z): the earliest task among equally near ones. Distance is from each plan's
/// [`TaskPlan::placed`], the expected landing. A task that says how many plans will do
/// ([`Place::enough`]) tries its poses nearest `near` first, equally near ones in their order, as
/// [`place`] tries them in their order: its plans are the nearest that get through. One that does
/// not tries every pose. Of a task's several plans, the nearest counts for it, the quickest among
/// equally near ones. Candidates are numbered as the caller's poses are, in the plan and in each
/// failure. Where none gets through, each task's failure comes back, in task order (none for no
/// tasks). A `near` that is not finite is an input error.
// Compiled where it is called: this crate's own code stays as it was without it (`nm -n`).
#[inline]
pub fn place_nearest(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Place],
    near: Vec3,
    o: &TaskOptions,
) -> Result<std::result::Result<NearestPlace, Vec<TaskFailure>>> {
    ensure_input!(
        near.is_finite(),
        "the point to set the object down near must be finite, got {near}"
    );
    // The squared distance in x and y from `near` of an object set down at `pose`.
    let off = |pose: Pose| pose.position.truncate().distance_squared(near.truncate());
    // Each task's poses as planned, by their indices among its own: nearest first where only some
    // plans will do, else as given.
    let orders = tasks.iter().map(|t| {
        let mut order: Vec<usize> = (0..t.poses.len()).collect();
        if t.enough.is_some() {
            order.sort_by(|&a, &b| off(t.poses[a]).total_cmp(&off(t.poses[b])));
        }
        order
    });
    let orders: Vec<Vec<usize>> = orders.collect();
    let nearest_first = tasks.iter().zip(&orders).map(|(t, order)| Place {
        poses: order.iter().map(|&k| t.poses[k]).collect(),
        // One per pose, or as given for `place` to refuse.
        falls: match t.falls.len() == t.poses.len() {
            true => order.iter().map(|&k| t.falls[k]).collect(),
            false => t.falls.clone(),
        },
        ..t.clone()
    });
    let nearest_first: Vec<Place> = nearest_first.collect();
    let TaskResult {
        mut problems,
        outcomes,
    } = place(device, worlds, &nearest_first, o)?;
    // The nearest plan so far, its squared distance and its task; and the tasks' failures.
    let mut nearest: Option<(f32, usize, TaskPlan)> = None;
    let mut failures = vec![];
    let each = outcomes.into_iter().enumerate().zip(&orders);
    for ((task, outcome), order) in each {
        let plans = match outcome {
            Ok(plans) => plans,
            Err(failure) => {
                let mut stops: Vec<(usize, CandidateStop)> =
                    order.iter().copied().zip(failure.candidates).collect();
                stops.sort_by_key(|&(k, _)| k);
                failures.push(TaskFailure {
                    candidates: stops.into_iter().map(|(_, stop)| stop).collect(),
                });
                continue;
            }
        };
        for plan in plans {
            let placed = plan
                .placed
                .as_ref()
                .expect("a place's plan says where it sets the object down");
            let distance = off(placed.pose());
            if nearest
                .as_ref()
                .is_none_or(|(least, _, _)| distance < *least)
            {
                let candidate = order[plan.candidate];
                nearest = Some((distance, task, TaskPlan { candidate, ..plan }));
            }
        }
    }
    Ok(match nearest {
        Some((_, task, plan)) => Ok(NearestPlace {
            task,
            problem: Place {
                poses: tasks[task].poses.clone(),
                falls: tasks[task].falls.clone(),
                ..problems.swap_remove(task)
            },
            plan,
        }),
        None => Err(failures),
    })
}
