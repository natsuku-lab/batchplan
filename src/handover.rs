//! One hand passing what it holds to another hand of the same robot ([`handover`]), a task on
//! the same steps as [`pick`](crate::pick()) and [`place`](crate::place()): its candidates (each
//! meeting with each of the taker's grasps) run with every other task's, step by step, each step
//! one batch, and every one that gets through comes back quickest first. [`handover_poses`] and
//! [`handover_grasps`] propose the meetings and the grasps.

use std::f32::consts::PI;

use glam::{Quat, Vec2, Vec3};
use rayon::prelude::*;

use crate::constraint::HandConstraint;
use crate::device::{Device, Worlds};
use crate::error::{Result, ensure_input};
use crate::held::{self, Held};
use crate::rng::Rng;
use crate::robot::Robot;
use crate::task::{self, Candidate, Giver, Leg, Steps, TaskOptions, TaskResult, TaskStep};
use crate::types::Pose;
use crate::world::Touch;

/// One hand passing the object it holds to another hand of the same robot: both hands meet with
/// it, the taker moves in to its grasp, and once the taker's gripper closes and the giver's
/// opens, the giver moves off.
#[derive(Clone, Debug)]
pub struct Handover {
    /// The world's index in the batch. The held object is not one of its obstacles.
    pub world: u32,
    /// Where the robot starts, at rest, the giving hand holding the object.
    pub start: Vec<f32>,
    /// What the giving hand holds, and how: a pick's [`TaskPlan::held`](crate::TaskPlan::held).
    pub held: Held,
    /// The hand that takes it.
    pub to: usize,
    /// Poses of the object's frame where it changes hands, in the robot's frame; any one will do.
    /// [`handover_poses`] proposes some where both arms reach.
    pub meetings: Vec<Pose>,
    /// Poses of the taking hand's IK frame that grasp the object, in the object's frame.
    /// [`handover_grasps`] turns the giver's grasp to the object's other end.
    pub grasps: Vec<Pose>,
    /// How far back along its own z axis the taking hand begins to move in (meters).
    pub approach: f32,
    /// How far back along its own z axis the giving hand moves once it lets go (meters).
    pub retreat: f32,
    /// Where the hands must stay on the way to the meeting ([`HandConstraint`]), as an object
    /// carried level. The start must keep them; a meeting whose move in begins where they break
    /// stops at the transit.
    pub constraints: Vec<HandConstraint>,
}

impl Handover {
    /// `held` passed from its hand to hand `to` from `start` in world `world`: at the meetings
    /// [`handover_poses`] proposes, by the grasps [`handover_grasps`] gives, the taker moving in
    /// 10 cm and the giver backing off 10 cm, each along its z axis.
    #[must_use]
    pub fn new(
        robot: &Robot,
        world: u32,
        start: impl Into<Vec<f32>>,
        held: Held,
        to: usize,
    ) -> Self {
        let start = start.into();
        Self {
            world,
            meetings: handover_poses(robot, &start, &held, to),
            grasps: handover_grasps(&held),
            start,
            held,
            to,
            approach: 0.1,
            retreat: 0.1,
            constraints: vec![],
        }
    }
}

/// Passes each task's held object to the hand that takes it, trying each meeting with each of the
/// taker's grasps: candidate `meeting * grasps.len() + grasp`. At a meeting the object stands in
/// the world where it changes hands, the giver's hand and the taker's fingers free to touch it:
/// IK solves both hands there at once, the giver holding the object as it does and the taker at
/// its grasp; the taker's move in runs backward from there with the giver's joints held, and the
/// giver's move away runs from there with the taker's held, as on a robot whose arms share a
/// torso the other arm must not move. The way to the meeting is planned with the object in the
/// giver's hand. Run the transit and the approach, close the taker's gripper, open the giver's,
/// then run the departure. `device` is the robot's without the object.
pub fn handover(
    device: &Device,
    worlds: &Worlds,
    tasks: &[Handover],
    o: &TaskOptions,
) -> Result<TaskResult<Handover>> {
    let robot = device.robot();
    task::check_options(o)?;
    for (i, t) in tasks.iter().enumerate() {
        task::check_task(robot, worlds, i, &steps(t), &t.start, t.to)?;
        t.held.check(robot, worlds, &format!("task {i}"))?;
        ensure_input!(
            t.to != t.held.hand,
            "task {i}: hand {} takes what it holds already",
            t.to
        );
        task::check_poses(i, "meeting", &t.meetings)?;
        task::check_poses(i, "grasp", &t.grasps)?;
        ensure_input!(
            t.approach.is_finite() && t.approach > 0.0 && t.retreat.is_finite() && t.retreat > 0.0,
            "task {i} needs a positive approach and retreat, got {} and {} m",
            t.approach,
            t.retreat
        );
    }
    // A device holding each task's object, as `place` builds them.
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
        let from = task.held.hand;
        let steps = steps(task);
        let (giving, taking) = (robot.hand_joints(from), robot.hand_joints(task.to));
        let idle: Vec<usize> = robot
            .idle_joints(task.to)
            .into_iter()
            .filter(|j| !giving.contains(j))
            .collect();
        for (m, &meeting) in task.meetings.iter().enumerate() {
            let gives = meeting.mul_pose(task.held.grasp);
            for (g, &grasp) in task.grasps.iter().enumerate() {
                let at = meeting.mul_pose(grasp);
                let poses = [
                    task::behind(at, task.approach),
                    at,
                    task::behind(gives, task.retreat),
                ];
                let index = m * task.grasps.len() + g;
                let mut c = Candidate::new(t, index, &steps, task.to, &task.start, poses);
                c.giver = Some(Giver {
                    hand: from,
                    at: gives,
                    hold_in: giving.clone(),
                    hold_away: taking.clone(),
                    idle: idle.clone(),
                    meeting,
                });
                candidates.push(c);
            }
        }
    }

    // Starts clear of everything, holding the object.
    let worlds_of: Vec<u32> = tasks.iter().map(|t| t.world).collect();
    let mut starting = task::by_holder(
        task::live(&mut candidates, TaskStep::Start),
        &holder_of,
        &holders,
    );
    device.install(|| {
        starting
            .par_iter_mut()
            .zip(&holders)
            .try_for_each(|(group, holder)| task::start(holder, worlds, group, &worlds_of))
    })?;

    // Both hands at the meeting and the straight moves there, the object standing where it changes
    // hands: a world for each meeting, where the giver's hand and the taker's fingers may touch it.
    if !task::live(&mut candidates, TaskStep::Reach).is_empty() {
        let mut first_meeting = vec![];
        let mut meetings = vec![];
        for t in tasks {
            first_meeting.push(meetings.len() as u32);
            let mut touching = robot.hand_links(t.held.hand);
            touching.extend(task::fingers(robot, t.to));
            for &meeting in &t.meetings {
                let mut w = worlds[t.world as usize].clone();
                w.obstacles.push(t.held.object.placed(meeting));
                let object = w.obstacles.len() - 1;
                w.touch.push(Touch::new(object, touching.iter().copied()));
                meetings.push(w);
            }
        }
        let meetings = device.upload(&meetings)?;
        let at_its_meeting = |_: usize, c: &Candidate| {
            first_meeting[c.task] + (c.index / tasks[c.task].grasps.len()) as u32
        };
        task::reach(
            device,
            &meetings,
            &mut task::live(&mut candidates, TaskStep::Reach),
            &at_its_meeting,
            &o.ik,
        )?;
        let mut taking = task::live(&mut candidates, TaskStep::Approach);
        task::straight(device, &meetings, &mut taking, &at_its_meeting, Leg::In, o)?;
        let mut leaving = task::live(&mut candidates, TaskStep::Depart);
        task::straight(
            device,
            &meetings,
            &mut leaving,
            &at_its_meeting,
            Leg::Away,
            o,
        )?;
    }

    // From the start to the meeting, the object in the giver's hand.
    let mut going = task::by_holder(
        task::live(&mut candidates, TaskStep::Transit),
        &holder_of,
        &holders,
    );
    let constraints: Vec<&[HandConstraint]> = tasks.iter().map(|t| &t.constraints[..]).collect();
    device.install(|| {
        going
            .par_iter_mut()
            .zip(&holders)
            .try_for_each(|(group, holder)| {
                task::transit(holder, worlds, group, &constraints, &o.plan)
            })
    })?;

    let outcomes = task::choose(device, tasks.len(), &candidates, o, |c| {
        let t = &tasks[c.task];
        let held = Held {
            hand: t.to,
            object: t.held.object.clone(),
            grasp: t.grasps[c.index % t.grasps.len()],
        };
        (Some(held), None)
    })?;
    Ok(TaskResult {
        problems: tasks.to_vec(),
        outcomes,
    })
}

/// Grasps for the hand that takes `held`'s object from the hand that holds it, for
/// [`Handover::grasps`], in the object's frame. The object's own axis nearest the giver's grasp
/// points to the end it holds; the giver's grasp turned half about the other axis nearest its
/// approach puts the taker at the other end from the same side, and turned half about the axis
/// across both, at the other end from the other side; for a grasp from an end face, both come
/// from the far end face, half a turn apart. Every object a hand can hold (a box, ball, cylinder
/// or capsule) is the same object turned so. A grasp
/// at the object's center gives grasps where the giver's hand is, which the hands cannot share.
#[must_use]
pub fn handover_grasps(held: &Held) -> Vec<Pose> {
    let grasp = held.grasp;
    let approach = grasp.rotation * Vec3::Z;
    let nearest = |v: Vec3, not: Option<Vec3>| {
        let axes = Vec3::AXES.into_iter().filter(|&a| Some(a) != not);
        axes.max_by(|a, b| a.dot(v).abs().total_cmp(&b.dot(v).abs()))
            .expect("three axes")
    };
    let toward_end = nearest(grasp.position, None);
    let along = nearest(approach, Some(toward_end));
    let across = along.cross(toward_end).abs();
    [along, across]
        .into_iter()
        .map(|axis| {
            let turn = Quat::from_axis_angle(axis, PI);
            Pose::new(turn * grasp.position, (turn * grasp.rotation).normalize())
        })
        .collect()
}

/// Poses of `held`'s object where hand `to` may take it from the hand that holds it, for
/// [`Handover::meetings`]. The two arms' shoulders, the first links on the way to each hand that
/// the other hand's joints do not move, are taken where `start` has them, and the object goes where
/// both reach, in front of them halfway between: 30, 45 and 60% of the way out to where either
/// reaches no further, level with the shoulders, 0.3 rad above, or 0.3 or 0.6 rad below. At each it
/// keeps its tilt in the giver's hand at `start`, turned about the
/// vertical so that the end the giver holds faces the giver, and 0.4 rad either way. None where
/// the shoulders stand above one another, or `start` or the hands do not fit the robot.
#[must_use]
pub fn handover_poses(robot: &Robot, start: &[f32], held: &Held, to: usize) -> Vec<Pose> {
    let from = held.hand;
    let hands = robot.hands();
    let finite = start.iter().all(|v| v.is_finite());
    if start.len() != robot.dof() || !finite || from >= hands || to >= hands || from == to {
        return vec![];
    }
    let (giving, taking) = (robot.hand_joints(from), robot.hand_joints(to));
    let fk = robot.fk(start);
    let shoulder = |hand: usize, other: &[usize]| {
        let mut own = robot.chain(robot.ee_links[hand]);
        own.find(|(_, dof)| !other.contains(dof))
            .map(|(link, _)| fk.pos[link])
    };
    let (Some(giver), Some(taker)) = (shoulder(from, &taking), shoulder(to, &giving)) else {
        return vec![];
    };
    // The farthest a hand reaches from its shoulder, over configurations of its own joints.
    let reach = |hand: usize, shoulder: Vec3, other: &[usize]| {
        let own: Vec<usize> = robot
            .hand_joints(hand)
            .into_iter()
            .filter(|j| !other.contains(j))
            .collect();
        let mut rng = Rng::new(REACH_SEED);
        let mut q = start.to_vec();
        let mut farthest = 0.0f32;
        for _ in 0..REACH_SAMPLES {
            for &j in &own {
                q[j] = rng.range(robot.lower()[j], robot.upper()[j]);
            }
            farthest = farthest.max(robot.ee_pose(&q, hand).position.distance(shoulder));
        }
        farthest
    };
    let reach = reach(from, giver, &taking).min(reach(to, taker, &giving));
    let across = (taker - giver).normalize_or_zero();
    let level = Vec3::Z.cross(across).normalize_or_zero();
    if level == Vec3::ZERO {
        return vec![];
    }
    // Forward, toward where the hands are.
    let middle = 0.5 * (giver + taker);
    let ee = |hand: usize| fk.pos[robot.ee_links[hand]];
    let ahead = 0.5 * (ee(from) + ee(to)) - middle;
    let forward = if ahead.dot(level) < 0.0 {
        -level
    } else {
        level
    };
    let half = 0.5 * giver.distance(taker);
    if reach <= half {
        // Shoulders farther apart than either arm reaches.
        return vec![];
    }
    let radius = (reach * reach - half * half).sqrt();
    // The object as the giver holds it, turned about the vertical so its held end faces the giver.
    let object = robot.ee_pose(start, from).mul_pose(held.grasp.inverse());
    let end = object.rotation * held.grasp.position;
    let face = -across;
    let yaw = if Vec2::new(end.x, end.y).length() < 1e-3 {
        0.0
    } else {
        face.y.atan2(face.x) - end.y.atan2(end.x)
    };
    let mut poses = vec![];
    for out in MEETING_OUT {
        for rise in MEETING_RISE {
            let center = middle + radius * out * (forward * rise.cos() + Vec3::Z * rise.sin());
            for turn in MEETING_TURNS {
                let rotation = Quat::from_rotation_z(yaw + turn) * object.rotation;
                poses.push(Pose::new(center, rotation.normalize()));
            }
        }
    }
    poses
}

/// [`handover_poses`]' meetings: how far out toward where the arms reach no further, how far
/// above the shoulders' level (radians), and the turns about the vertical.
const MEETING_OUT: [f32; 3] = [0.3, 0.45, 0.6];
const MEETING_RISE: [f32; 4] = [0.0, 0.3, -0.3, -0.6];
const MEETING_TURNS: [f32; 3] = [0.0, 0.4, -0.4];
/// How [`handover_poses`] samples each arm's reach.
const REACH_SAMPLES: usize = 64;
const REACH_SEED: u64 = 0x6a6f_696e;

/// A handover's steps: all in its one world, the transit planned.
fn steps(t: &Handover) -> Steps<'_> {
    Steps {
        world: t.world,
        transit_world: t.world,
        depart_world: t.world,
        transit: true,
        constraints: &t.constraints,
    }
}
