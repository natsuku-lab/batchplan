//! Planning on measured closing regions, one bounded batch of worlds at a time.
use std::collections::HashMap;

use batchplan::{
    CandidateStop, Device, Grasp, Pick, Pose, Pull, Screw, TaskOptions, TaskPlan, TaskResult,
    TaskStep, Touch, Turn, Worlds,
};
use glam::Vec3;

use crate::error::{Result, ensure_input};
use crate::{GripWorlds, Gripped, Of, Planned};

/// A measured point the fingers should close on, or a caller-specified neighborhood.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum GripTarget {
    /// The point lies inside the opposing pads' closing region at the chosen opening.
    Point(Vec3),
    /// The measured contact midpoint is within the caller's radius.
    Near {
        /// Point in the observation's frame (meters).
        point: Vec3,
        /// Maximum distance from the point (meters).
        radius: f32,
    },
}

impl GripTarget {
    fn accepts(self, source: &GripWorlds<'_>, candidate: &Grasp) -> bool {
        match self {
            Self::Point(point) => source.closes_on(candidate, point),
            Self::Near { point, radius } => {
                candidate.contacts[0]
                    .midpoint(candidate.contacts[1])
                    .distance(point)
                    <= radius
            }
        }
    }
}

/// Which grasps to plan on, how many complete plans will do, and how much to keep in flight.
#[derive(Clone, Copy, Debug)]
pub struct GripSearch {
    /// Select by measured position, preserving the input order.
    pub target: Option<GripTarget>,
    /// How many complete plans will do; `None` tries every eligible candidate.
    pub enough: Option<usize>,
    /// Maximum projected worlds in an in-flight batch. Winners requested with `enough` are
    /// additionally retained. Choose this for available memory; fine grids can be tens of MB.
    pub batch: usize,
}

impl Default for GripSearch {
    fn default() -> Self {
        Self {
            target: None,
            enough: Some(1),
            batch: 1,
        }
    }
}

/// The original measured candidate and its projected world, provided while configuring a task.
#[derive(Debug)]
#[non_exhaustive]
pub struct GripCandidate<'a> {
    /// Original index in the supplied ordered grasps.
    pub index: usize,
    /// Original grasp, including its finger opening in meters.
    pub candidate: &'a Grasp,
    /// The owner of each obstacle in the projection, in world order.
    pub of: &'a [Of],
    /// Existing cell permissions, such as for a mounted robot, go here.
    pub touches: &'a mut Vec<Touch>,
    split: &'a Gripped,
    world: u32,
    hand: usize,
}

impl GripCandidate<'_> {
    /// A split pick in the candidate's world, with the whole movable body selected.
    #[must_use]
    pub fn pick(&self, start: &[f32]) -> Pick {
        Pick {
            hand: self.hand,
            body: self.split.body.clone(),
            ..Pick::new(self.world, start, self.split.part, vec![self.split.grasp])
        }
    }

    /// A turn about the supplied screw with the measured part fixed in the world. Choose
    /// `None` explicitly for rotation about the hand's z axis through the grasp.
    #[must_use]
    pub fn turn(&self, start: &[f32], screw: Option<Screw>, angle: f32) -> Turn {
        Turn {
            hand: self.hand,
            stays: true,
            screw,
            ..Turn::new(
                self.world,
                start,
                self.split.part,
                vec![self.split.grasp],
                angle,
            )
        }
    }

    /// A straight draw with the measured part fixed in the world.
    #[must_use]
    pub fn pull(&self, start: &[f32], direction: Vec3, distance: f32) -> Pull {
        Pull {
            hand: self.hand,
            ..Pull::new(
                self.world,
                start,
                self.split.part,
                vec![self.split.grasp],
                direction,
                distance,
            )
        }
    }
}

/// A complete candidate and its collision worlds. Keep this result to execute its trajectories.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GripPlan {
    /// The measured grasp in the look's frame, including the opening used for the plan.
    pub grasp: Grasp,
    /// The trajectories. The candidate index is the original index in the supplied slice.
    pub plan: TaskPlan,
    /// Exact locked joint values configured for this candidate, including asymmetric fingers.
    pub locked: HashMap<String, f32>,
    /// Source for `Observed::picked` and the original collision world.
    pub grip: Gripped,
    contact: Planned,
    lift: Option<Planned>,
}

impl GripPlan {
    /// The world checked for this motion, with its original obstacle attribution and touches.
    /// For a pick's departure, use the returned hold and the closed finger values as well.
    #[must_use]
    pub fn world(&self, stage: TaskStep) -> &Planned {
        match stage {
            TaskStep::Start | TaskStep::Transit => &self.grip.scene,
            TaskStep::Depart => self.lift.as_ref().unwrap_or(&self.contact),
            _ => &self.contact,
        }
    }
}

/// Why a measured candidate yielded no complete motions.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum GripStop {
    /// The requested hand pose is provably outside the loaded chain's reach, even allowing
    /// the IK tolerances. No world was projected for this candidate.
    OutsideReach {
        /// Requested hand pose in the look's frame.
        at: Pose,
    },
    /// The look could not isolate a contact part.
    NoPart,
    /// A batchplan motion stopped. Its contact indices name obstacles in `of`; the dense
    /// collision grids have already been dropped.
    Motion {
        /// The failed stage, reason, contacts and any partial turn/draw.
        stop: CandidateStop,
        /// Source attribution for the obstacle numbers in the stop.
        of: Vec<Of>,
    },
}

/// A stopped candidate in the original input's numbering.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GripFailure {
    /// Original candidate index.
    pub candidate: usize,
    /// The projection or motion failure.
    pub reason: GripStop,
}

/// The perceived objects that would be displaced by picking their support.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SupportedObjects {
    /// The proposed pick target's ID.
    pub object: u32,
    /// Other perceived IDs standing on the target, directly or transitively, in memory order.
    pub supports: Vec<u32>,
}

/// The requested prefix of feasible candidates and diagnostics for attempted failures.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct GripPlans {
    /// A pick was refused before projection because another perceived object stands on its target.
    /// Set once for the whole search, when at least one supplied grasp meets its target selector.
    pub blocked: Option<SupportedObjects>,
    /// Actual calls to a batched motion solver, excluding projection-only batches.
    pub batches: usize,
    /// In trajectory-duration order, ties in original input order.
    pub plans: Vec<GripPlan>,
    /// Eligible candidates that failed, in original order.
    pub failures: Vec<GripFailure>,
}

/// An interruption stays distinct from a completed search, even when a plan was already found.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum GripOutcome {
    /// The requested number of plans or the end of the input was reached.
    Complete(GripPlans),
    /// The caller stopped the search; holds the completed work.
    Interrupted(GripPlans),
}

/// An ordinary [`Pick`], [`Turn`] or [`Pull`] for a measured grasp.
pub trait GripTask: sealed::Task {}
impl GripTask for Pick {}
impl GripTask for Turn {}
impl GripTask for Pull {}

mod sealed {
    use super::{
        Device, GripCandidate, Gripped, HashMap, Pick, Planned, Pull, Result, TaskOptions,
        TaskResult, Turn, Worlds,
    };

    pub trait Task: Clone {
        const PICK: bool = false;
        fn run(
            device: &Device,
            worlds: &Worlds,
            tasks: &[Self],
            o: &TaskOptions,
        ) -> Result<TaskResult<Self>>;
        fn matches(&self, c: &GripCandidate<'_>) -> bool;
        fn locks(&self) -> &[HashMap<String, f32>];
        fn lift(&self, _split: &Gripped) -> Result<Option<Planned>> {
            Ok(None)
        }
    }

    impl Task for Pick {
        const PICK: bool = true;
        fn run(d: &Device, w: &Worlds, t: &[Self], o: &TaskOptions) -> Result<TaskResult<Self>> {
            batchplan::pick(d, w, t, o)
        }
        fn locks(&self) -> &[HashMap<String, f32>] {
            &self.locked
        }
        fn matches(&self, c: &GripCandidate<'_>) -> bool {
            self.world == c.world
                && self.hand == c.hand
                && self.object == c.split.part
                && self.grasps == [c.split.grasp]
                && self.transit_world.is_none()
                && self.depart_world.is_none()
                && self.body == c.split.body
        }
        fn lift(&self, split: &Gripped) -> Result<Option<Planned>> {
            let body = split.body.as_ref().expect("split picks have a body");
            let of = split
                .scene
                .of
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != split.part && !body.other_parts.contains(k))
                .map(|(_, of)| *of)
                .collect();
            Ok(Some(Planned {
                world: self.lift_world(&split.scene.world)?,
                of,
            }))
        }
    }
    impl Task for Pull {
        fn run(d: &Device, w: &Worlds, t: &[Self], o: &TaskOptions) -> Result<TaskResult<Self>> {
            batchplan::pull(d, w, t, o)
        }
        fn locks(&self) -> &[HashMap<String, f32>] {
            &self.locked
        }
        fn matches(&self, c: &GripCandidate<'_>) -> bool {
            self.world == c.world
                && self.hand == c.hand
                && self.object == c.split.part
                && self.grasps == [c.split.grasp]
                && self.transit_world.is_none()
        }
    }
    impl Task for Turn {
        fn run(d: &Device, w: &Worlds, t: &[Self], o: &TaskOptions) -> Result<TaskResult<Self>> {
            batchplan::turn(d, w, t, o)
        }
        fn locks(&self) -> &[HashMap<String, f32>] {
            &self.locked
        }
        fn matches(&self, c: &GripCandidate<'_>) -> bool {
            self.world == c.world
                && self.hand == c.hand
                && self.object == c.split.part
                && self.grasps == [c.split.grasp]
                && self.transit_world.is_none()
                && self.stays
        }
    }
}

impl GripWorlds<'_> {
    /// Tries an ordered slice on its per-grasp worlds. Configure one ordinary task with the
    /// candidate's constructor, setting the cell's static touches, locked finger positions and
    /// motion options. The constructor owns the single grasp, world, hand and contact geometry.
    /// All stages start from the projected world. A pick may additionally name measured support
    /// indices. Its returned departure world has these permissions confined to the held body.
    ///
    /// The first `enough` candidates that succeed in input order are returned in increasing
    /// duration order. Each batch is checked by the corresponding batched batchplan primitive;
    /// IK on the GPU can round differently at different batch sizes. `should_stop` is polled
    /// before projection, before planning a batch, and after batch completion. A pick that
    /// would take another object with it checks candidate inputs and selection, records the
    /// supported IDs once in [`GripPlans::blocked`], and projects no candidate worlds.
    pub fn plan<T: GripTask>(
        &self,
        grasps: &[Grasp],
        search: &GripSearch,
        options: &TaskOptions,
        mut should_stop: impl FnMut() -> bool,
        mut configure: impl FnMut(&mut GripCandidate<'_>) -> T,
    ) -> Result<GripOutcome> {
        ensure_input!(
            search.batch > 0 && search.enough != Some(0),
            "grip search needs a positive batch and plan count, got {search:?}"
        );
        if let Some(target) = search.target {
            let valid = match target {
                GripTarget::Point(point) => point.is_finite(),
                GripTarget::Near { point, radius } => {
                    point.is_finite() && radius.is_finite() && radius >= 0.0
                }
            };
            ensure_input!(valid, "invalid grip target {target:?}");
        }
        if self.reach.is_some() {
            options.check()?;
        }
        let mut result = GripPlans::default();
        let mut candidates = grasps.iter().enumerate();
        let blocked = if T::PICK { self.blocked.as_ref() } else { None };
        let enough = search.enough.unwrap_or(usize::MAX);
        let mut window = None;
        let interrupted = loop {
            let mut batch = vec![];
            let mut tasks = vec![];
            let mut stopped = false;
            for (index, candidate) in candidates.by_ref() {
                if should_stop() {
                    stopped = true;
                    break;
                }
                if blocked.is_some() || search.target.is_some() {
                    self.validate(candidate)?;
                }
                if search
                    .target
                    .is_some_and(|target| !target.accepts(self, candidate))
                {
                    continue;
                }
                if let Some(blocked) = blocked {
                    result.blocked.get_or_insert_with(|| blocked.clone());
                    continue;
                }
                if self
                    .reach
                    .as_ref()
                    .is_some_and(|reach| reach.excludes(candidate.pose, &options.ik))
                {
                    self.validate(candidate)?;
                    result.failures.push(GripFailure {
                        candidate: index,
                        reason: GripStop::OutsideReach { at: candidate.pose },
                    });
                    continue;
                }
                let Some(mut split) = self.project_in(candidate, &mut window)? else {
                    result.failures.push(GripFailure {
                        candidate: index,
                        reason: GripStop::NoPart,
                    });
                    continue;
                };
                let world = u32::try_from(batch.len()).map_err(|_| {
                    batchplan::Error::Input(format!(
                        "a grip search has too many worlds in a batch: {}",
                        batch.len()
                    ))
                })?;
                let mut touches = std::mem::take(&mut split.scene.world.touch);
                let mut c = GripCandidate {
                    index,
                    candidate,
                    of: &split.scene.of,
                    touches: &mut touches,
                    split: &split,
                    world,
                    hand: self.hand_index,
                };
                let task = configure(&mut c);
                ensure_input!(
                    task.matches(&c),
                    "grip task {index} changed the source hand, world, grasp or collision body"
                );
                split.scene.world.touch = touches;
                tasks.push(task);
                batch.push((index, *candidate, split));
                if batch.len() == search.batch {
                    break;
                }
            }
            if stopped || should_stop() {
                break true;
            }
            if batch.is_empty() {
                break false;
            }
            let worlds: Vec<_> = batch
                .iter()
                .map(|(_, _, g)| g.scene.world.clone())
                .collect();
            let uploaded = self.device.upload(&worlds)?;
            result.batches += 1;
            let solved = T::run(self.device, &uploaded, &tasks, options)?;
            for (((index, grasp, grip), task), outcome) in
                batch.into_iter().zip(tasks).zip(solved.outcomes)
            {
                match outcome {
                    Ok(mut plans) => {
                        if result.plans.len() < enough {
                            let mut plan =
                                plans.pop().expect("a successful single grasp has one plan");
                            plan.candidate = index;
                            let mut contact = grip.scene.clone();
                            contact.world.touch.push(Touch::new(
                                grip.part,
                                self.device.robot().finger_links(self.hand_index),
                            ));
                            let lift = task.lift(&grip)?;
                            result.plans.push(GripPlan {
                                grasp,
                                plan,
                                locked: task.locks().first().cloned().unwrap_or_default(),
                                grip,
                                contact,
                                lift,
                            });
                        }
                    }
                    Err(mut failure) => {
                        let stop = failure
                            .candidates
                            .pop()
                            .expect("a failed single grasp has one stop");
                        result.failures.push(GripFailure {
                            candidate: index,
                            reason: GripStop::Motion {
                                stop,
                                of: grip.scene.of,
                            },
                        });
                    }
                }
            }
            if should_stop() {
                break true;
            }
            if result.plans.len() >= enough || candidates.len() == 0 {
                break false;
            }
        };
        result
            .plans
            .sort_by(|a, b| a.plan.duration().total_cmp(&b.plan.duration()));
        result.failures.sort_by_key(|f| f.candidate);
        Ok(if interrupted {
            GripOutcome::Interrupted(result)
        } else {
            GripOutcome::Complete(result)
        })
    }
}
