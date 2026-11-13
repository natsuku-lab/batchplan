//! Whole tasks in one call: a hand picking up an object ([`pick`]) or setting down the one it holds
//! ([`place`]). Each is three motions from rest to rest: a planned path from the start to where the
//! hand begins to move in, a straight move in (to the grasp, or down to where the object goes), and
//! a straight move away (lifting the object, or backing off from it). The gripper closes or opens
//! between the second and the third; every motion is checked with the fingers where the start has
//! them, so open them first.
//!
//! A task's candidates, its grasps or the poses it may set the object at, run with every other
//! task's, step by step, each step one batch: IK at each grasp or release, the moves in (run
//! backward from there, so they end exactly there), the moves away, then planned paths for those
//! still going. Every candidate that passes every step comes back, quickest first. When none does,
//! each candidate's step and reason come back instead.
//!
//! A task may leave its transit out (`transit: false`) and give the two straight moves alone: a
//! program of many tasks solves all their moves in one call, plans the ways between them in one
//! [`plan`](crate::plan) and joins them ([`join`](crate::join())). Each step may run in a world of
//! its own (`transit_world`, `depart_world`), so that what the hand may touch at the grasp is not
//! what it may touch on the way there.

use std::collections::HashMap;
use std::f32::consts::TAU;
use std::fmt;

use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

use crate::constraint::{self, HandConstraint};
use crate::device::{Contact, Device, Worlds};
use crate::edges::{self, Deferred, EdgeQuery, Search, Wanted};
use crate::error::{Result, ensure_input, input};
use crate::held::{self, Fitted, Held};
use crate::ik::{self, IkFailure, IkOptions, IkProblem, IkResult};
use crate::robot::Robot;
use crate::timing::{HandSpeed, Near, Trajectory};
use crate::tool_path::{self, ToolPath, ToolPathOptions, ToolPathStop};
use crate::trajopt::{self, PlanOptions, PlanProblem};
use crate::types::Pose;
use crate::world::{Obstacle, Touch, World};

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
    /// The object: an obstacle of the world, by index. A distance grid cannot be picked up.
    pub object: usize,
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
    /// Joints the robot was loaded with locked, at the values they keep through the task, as
    /// [`Robot::relock`] takes them: the fingers opened only as far as a rim between walls lets
    /// them. A joint that carries a hand (a torso) is refused. Tasks are grouped by these, a relocked robot for each group in one call, and each
    /// plan's [`TaskPlan::held`] starts with them (overwrite them with where the fingers closed).
    /// Empty: as loaded.
    pub locked: HashMap<String, f32>,
}

impl Pick {
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
            grasps,
            approach: 0.1,
            lift: Vec3::Z * 0.1,
            support: vec![],
            constraints: vec![],
            locked: HashMap::new(),
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
    /// object's symmetric turns at one spot.
    pub poses: Vec<Pose>,
    /// How the hand moves in to where it lets go, in the world's frame (meters).
    pub lower: Vec3,
    /// How far back along its own z axis the hand moves once it lets go (meters).
    pub retreat: f32,
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
            lower: Vec3::Z * -0.1,
            retreat: 0.1,
            support: vec![],
            constraints: vec![],
        }
    }
}

/// How [`pick`], [`place`] and [`handover`](crate::handover()) solve, follow, plan and time
/// their steps.
#[derive(Clone, Debug)]
pub struct TaskOptions {
    /// IK at each grasp or release.
    pub ik: IkOptions,
    /// The planned paths from the start.
    pub plan: PlanOptions,
    /// The straight moves.
    pub tool_path: ToolPathOptions,
    /// The hand's speed cap on the straight moves (m/s): 10 cm/s. `f32::INFINITY` leaves it
    /// uncapped, the moves as fast as the joints' limits allow.
    pub tool_speed: f32,
    /// Where the cap holds (m): within this distance of the hand's position at the grasp or
    /// release, which the move in ends at and the move away starts from, the rest of each move
    /// as fast as the joints' limits allow ([`HandSpeed::near`]). `None`: along all of both.
    pub tool_speed_near: Option<f32>,
    /// How much of each straight move a candidate must make, in (0, 1]: 1, all of it. Less lets
    /// an obstacle shorten the move in or away rather than stop the candidate; a move stopped
    /// short for any other reason (out of reach, a jump) still stops it.
    pub min_fraction: f32,
}

impl Default for TaskOptions {
    fn default() -> Self {
        Self {
            ik: IkOptions::default(),
            plan: PlanOptions::default(),
            tool_path: ToolPathOptions::default(),
            tool_speed: 0.1,
            tool_speed_near: None,
            min_fraction: 1.0,
        }
    }
}

/// A task's steps, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum TaskStep {
    /// The start, checked for collision. A task without its transit skips it.
    Start,
    /// IK at the grasp, or where the hand lets go.
    Reach,
    /// The straight move in, to there.
    Approach,
    /// The straight move away: lifting the object, or backing off from it.
    Depart,
    /// The planned path from the start to where the move in begins. A task without its transit
    /// skips it.
    Transit,
}

const STEPS: [TaskStep; 5] = [
    TaskStep::Start,
    TaskStep::Reach,
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
    /// The straight move in. The gripper closes (pick) or opens (place) at its end; in a
    /// handover the taker's closes, then the giver's opens.
    pub approach: Trajectory,
    /// The straight move away: holding the object after a pick; the giver's after a handover.
    pub depart: Trajectory,
    /// After a pick, what the hand holds; after a handover, what the taker holds.
    pub held: Option<Held>,
    /// After a place, the object where it was set down, to add to the world.
    pub placed: Option<Obstacle>,
    /// After a handover, the object's pose where it changed hands, in the robot's frame.
    pub meeting: Option<Pose>,
}

impl TaskPlan {
    /// Its motions in the order they run: the transit if it has one, the move in, the move away.
    pub fn motions(&self) -> impl Iterator<Item = &Trajectory> {
        self.transit.iter().chain([&self.approach, &self.depart])
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
        for step in STEPS {
            let here: Vec<&CandidateStop> =
                self.candidates.iter().filter(|c| c.step == step).collect();
            if here.is_empty() {
                continue;
            }
            let doing = match step {
                TaskStep::Start => "starting in collision",
                TaskStep::Reach => "at the grasp or release",
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

/// What [`pick`], [`place`] or [`handover`](crate::handover()) did with each task.
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
    check_picks(device.robot(), worlds, tasks, o)?;
    if tasks.iter().all(|t| t.locked.is_empty()) {
        return picked(device, worlds, tasks, o);
    }
    // A relocked robot for each set of locked joints' values, its tasks a batch of their own, the
    // groups in parallel; a device for it builds nothing, and the worlds stay valid.
    let mut groups: Vec<(&HashMap<String, f32>, Vec<usize>)> = vec![];
    for (i, t) in tasks.iter().enumerate() {
        match groups.iter_mut().find(|(locked, _)| **locked == t.locked) {
            Some((_, items)) => items.push(i),
            None => groups.push((&t.locked, vec![i])),
        }
    }
    let results: Vec<TaskResult<Pick>> = device.install(|| {
        let each = groups.par_iter().map(|(locked, items)| {
            let mine = held::subset(tasks, items);
            if locked.is_empty() {
                picked(device, worlds, &mine, o)
            } else {
                let relocked = device.with_robot(&device.robot().relock(locked)?)?;
                picked(&relocked, worlds, &mine, o)
            }
        });
        each.collect::<Result<_>>()
    })?;
    let mut outcomes: Vec<_> = tasks.iter().map(|_| None).collect();
    for ((_, items), result) in groups.iter().zip(results) {
        for (&i, outcome) in items.iter().zip(result.outcomes) {
            outcomes[i] = Some(outcome);
        }
    }
    let outcomes = outcomes.into_iter();
    Ok(TaskResult {
        problems: tasks.to_vec(),
        outcomes: outcomes
            .map(|outcome| outcome.expect("every task is in a group"))
            .collect(),
    })
}

/// That `tasks` and `o` can be picked by `robot` in `worlds`, each task named by its index.
fn check_picks(robot: &Robot, worlds: &Worlds, tasks: &[Pick], o: &TaskOptions) -> Result<()> {
    check_options(o)?;
    for (i, t) in tasks.iter().enumerate() {
        let world = check_task(robot, worlds, i, &Steps::of_pick(t), &t.start, t.hand)?;
        let object = world.obstacles.get(t.object).ok_or_else(|| {
            input!(
                "task {i}: obstacle {} of {}",
                t.object,
                world.obstacles.len()
            )
        })?;
        ensure_input!(
            !matches!(object, Obstacle::Sdf { .. }),
            "task {i}: obstacle {} is a distance grid, which cannot be picked up",
            t.object
        );
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
        held::check_locked(robot, &t.locked, &format!("task {i}"))?;
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
        let object = worlds[task.world as usize].obstacles[task.object].pose();
        for (index, &grasp) in task.grasps.iter().enumerate() {
            let at = object.mul_pose(grasp);
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
                w.touch.push(Touch::new(t.object, fingers(robot, t.hand)));
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
            let holding = held(&tasks[c.task], worlds, c.index);
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
        let lifting = |hand: usize| {
            tasks
                .iter()
                .map(|t| {
                    let mut w = worlds[Steps::of_pick(t).depart_world as usize].clone();
                    w.remove(t.object)?;
                    if t.hand == hand {
                        let shifted = |s: usize| if s > t.object { s - 1 } else { s };
                        let link = held::held_link(hand);
                        let touches = t.support.iter().map(|&s| Touch::new(shifted(s), [&link]));
                        w.touch.extend(touches);
                    }
                    Ok(w)
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
        device.install(|| {
            let groups = alike.par_iter_mut().enumerate();
            groups.try_for_each(|(k, (holding, group))| {
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
                    let object = tasks[c.task].object;
                    for contact in &mut c.contacts {
                        if let Contact::World { obstacle, .. } = contact
                            && *obstacle >= object
                        {
                            *obstacle += 1;
                        }
                    }
                }
                Ok::<_, crate::Error>(())
            })
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
        (Some(held(&tasks[c.task], worlds, c.index)), None)
    })?;
    Ok(TaskResult {
        problems: tasks.to_vec(),
        outcomes,
    })
}

/// Sets down each task's held object, trying each of its poses. IK solves where the hand lets go,
/// holding the object, which may touch what it is set on there and on the move in, run backward
/// from there. The move away lets the fingers touch the object where it now stands. Run the
/// transit and the approach holding the object, open the gripper, then run the departure. `device`
/// is the robot's without the object; each task's `held` is attached to it here.
pub fn place(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Place],
    o: &TaskOptions,
) -> Result<TaskResult<Place>> {
    let robot = device.robot();
    check_options(o)?;
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
            t.lower.is_finite()
                && t.lower != Vec3::ZERO
                && t.retreat.is_finite()
                && t.retreat > 0.0,
            "task {i} needs a finite lowering that moves and a positive retreat, got {} and {} m",
            t.lower,
            t.retreat
        );
        check_support(i, world, &t.support, None)?;
    }
    // A device holding each task's object, built in parallel; tasks holding theirs alike share one.
    // Every step below runs each holder's candidates as a batch of their own, all in parallel.
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
    device.install(|| {
        starting
            .par_iter_mut()
            .zip(&holders)
            .try_for_each(|(group, holder)| start(holder, worlds, group, &worlds_of))
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
    let mut reaching = live(&mut candidates, TaskStep::Reach);
    if !reaching.is_empty() {
        for c in &reaching {
            if arrived.iter().all(|(hand, _)| *hand != c.hand) {
                let holder = &holders[holder_of[c.task]];
                arrived.push((c.hand, holder.upload(&arriving(c.hand))?));
            }
        }
        let given = |_: usize, c: &Candidate| c.world;
        let solved = solve(device, worlds, &reaching, &given, &o.ik)?;
        let solutions: Vec<Vec<f32>> = reaching
            .iter()
            .enumerate()
            .map(|(i, c)| nearest_first(&solved, i, &c.rest))
            .collect();
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
        let found: Vec<Vec<Option<usize>>> = device.install(|| {
            mine.par_iter()
                .zip(&holders)
                .map(|(mine, holder)| {
                    let Some(&first) = mine.first() else {
                        return Ok(vec![]);
                    };
                    let lists = mine
                        .iter()
                        .map(|&i| (reaching[i].task as u32, solutions[i].clone()))
                        .collect();
                    first_free(holder, uploaded_for(reaching[first].hand), lists)
                })
                .collect::<Result<_>>()
        })?;
        let n = robot.dof();
        // Per holder, the candidates whose every solution collides holding the object.
        let mut holding: Vec<Vec<usize>> = holders.iter().map(|_| vec![]).collect();
        for (&i, found) in mine.iter().flatten().zip(found.into_iter().flatten()) {
            let c = &mut reaching[i];
            match found {
                Some(k) => {
                    c.grasped = solutions[i][k * n..(k + 1) * n].to_vec();
                    c.passed += 1;
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
        classify(device, worlds, &mut reaching, &solved, &o.ik)?;
    }
    let mut moving = by_holder(
        live(&mut candidates, TaskStep::Approach),
        &holder_of,
        &holders,
    );
    device.install(|| {
        moving
            .par_iter_mut()
            .zip(&holders)
            .try_for_each(|(group, holder)| {
                let Some(hand) = group.first().map(|c| c.hand) else {
                    return Ok(());
                };
                let (_, uploaded) = arrived
                    .iter()
                    .find(|(h, _)| *h == hand)
                    .expect("every hand that lets go has its worlds");
                straight(holder, uploaded, group, &by_task, Leg::In, o)
            })
    })?;

    // Back from the object where it stands, the fingers touching it: a world for each.
    let mut leaving = live(&mut candidates, TaskStep::Depart);
    if !leaving.is_empty() {
        let left: Vec<World> = leaving
            .iter()
            .map(|c| {
                let t = &tasks[c.task];
                let mut w = worlds[Steps::of_place(t).depart_world as usize].clone();
                w.obstacles.push(t.held.object.placed(t.poses[c.index]));
                let object = w.obstacles.len() - 1;
                w.touch
                    .push(Touch::new(object, fingers(robot, t.held.hand)));
                w
            })
            .collect();
        let left = device.upload(&left)?;
        let each = |k: usize, _: &Candidate| k as u32;
        straight(device, &left, &mut leaving, &each, Leg::Away, o)?;
    }

    // From the start, holding the object.
    let mut going = by_holder(
        live(&mut candidates, TaskStep::Transit),
        &holder_of,
        &holders,
    );
    let constraints: Vec<&[HandConstraint]> = tasks.iter().map(|t| &t.constraints[..]).collect();
    device.install(|| {
        going
            .par_iter_mut()
            .zip(&holders)
            .try_for_each(|(group, holder)| transit(holder, worlds, group, &constraints, &o.plan))
    })?;

    let outcomes = choose(device, tasks.len(), &candidates, o, |c| {
        let t = &tasks[c.task];
        (None, Some(t.held.object.placed(t.poses[c.index])))
    })?;
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

/// How [`grasps`] proposes grasps for a parallel gripper whose IK frame lies between its
/// fingertips, its z axis pointing out of the hand and its y axis along the fingers' closing
/// direction (as the examples' Panda `ee_link`).
#[derive(Clone, Copy, Debug)]
pub struct GraspOptions {
    /// The widest the fingers open (meters): 8 cm.
    pub opening: f32,
    /// How far in from the object's surface the IK frame holds it (meters): 2 cm. Above a `floor`,
    /// a grasp from above may hold it less deep, its frame even outside the object.
    pub depth: f32,
    /// Grasps spread around a round object's axis: 8.
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
    /// is less than this over the floor has no grasp from above.
    pub overlap: f32,
    /// How far toward the ends a grasp holds the object, along the side the fingers close on
    /// rather than across, as a fraction of half that side, in [0, 1): 0, the middle. Above 0,
    /// each grasp comes once toward each end; a hand that will pass the object to another
    /// ([`handover`](crate::handover())) holds it toward one end, as 0.75, leaving the other to
    /// the hand that takes it. A ball's grasps, and a round object's from its ends or closing along
    /// its axis, stay in the middle; those closing around its side slide along the axis.
    pub along: f32,
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
            for across in (0..3).filter(|&a| a != face && fits(2.0 * half[a])) {
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
            c.passed += 1;
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
/// `worlds_of` gives it: kept if it covers all of it, or at least `o.min_fraction` where an obstacle
/// stops it.
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
    for c in live.iter_mut().filter(|c| !c.transits) {
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
            depart: Trajectory::with_hand_speed(
                robot,
                &c.depart,
                &[speed(c.away_hand(), away_start)],
            )?,
            held,
            placed,
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
fn held(task: &Pick, worlds: &Worlds, index: usize) -> Held {
    Held {
        locked: task.locked.clone(),
        hand: task.hand,
        object: worlds[task.world as usize].obstacles[task.object].placed(Pose::IDENTITY),
        grasp: task.grasps[index],
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
