//! Planner output as policy-training demonstrations: nominal reaches plus recoveries from
//! perturbed states, timed within the robot's limits at randomized speeds.

use std::f32::consts::PI;

use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

use crate::device::{self, Device, Worlds};
use crate::error::{Result, ensure_input, input};
use crate::ik::{self, IkOptions, IkProblem};
use crate::rng::Rng;
use crate::robot::{AttachedObject, Robot};
use crate::spline;
use crate::task::{self, Held, Pick, Place, TaskOptions, TaskPlan, TaskResult};
use crate::timing::{HandSpeed, Trajectory};
use crate::tool_path::{self, ToolPath};
use crate::trajopt::{self, PlanOptions, PlanProblem, PlanResult};
use crate::types::{JointTrajectory, Pose, Solved};
use crate::world::{Obstacle, Touch, World};

/// How [`recovery_problems`] perturbs planned paths.
#[derive(Clone, Copy, Debug)]
pub struct RecoveryOptions {
    /// Perturbed states drawn per solved path; those that collide are dropped.
    pub per_trajectory: usize,
    /// Standard deviation of the joint-space perturbation (radians).
    pub sigma: f32,
    /// Range of path fractions where perturbations happen.
    pub phase: (f32, f32),
    /// Seed of the perturbations.
    pub rng_seed: u64,
}

impl Default for RecoveryOptions {
    fn default() -> Self {
        Self {
            per_trajectory: 2,
            sigma: 0.15,
            phase: (0.2, 0.8),
            rng_seed: 3,
        }
    }
}

/// A planning problem that starts from a perturbed state of a planned path.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Recovery {
    /// Index in `result.problems` of the problem whose best path was perturbed.
    pub parent: usize,
    /// Fraction of the nominal path where the perturbation happened.
    pub phase: f32,
    /// From the perturbed state to the parent's goal.
    pub problem: PlanProblem,
}

/// Simulates policy mistakes: perturbs states along the best path of each solved problem and
/// keeps the collision-free ones as problems toward the same goal. Planning them yields
/// demonstrations of recovering from off-nominal states, which raw planner output never contains.
pub fn recovery_problems(
    device: &Device,
    worlds: &Worlds,
    result: &PlanResult,
    o: &RecoveryOptions,
) -> Result<Vec<Recovery>> {
    let robot = device.robot();
    let n = robot.dof();
    let mut rng = Rng::new(o.rng_seed);
    let mut candidates = vec![];
    let mut on_path = vec![0.0; n];
    for Solved {
        index: parent,
        problem,
        solution: path,
    } in result.solved()
    {
        for _ in 0..o.per_trajectory {
            let phase = rng.range(o.phase.0, o.phase.1);
            spline::at_phase(path, n, phase, &mut on_path);
            let start: Vec<f32> = (0..n)
                .map(|j| {
                    (on_path[j] + o.sigma * rng.normal())
                        .clamp(robot.bounds(j).0, robot.bounds(j).1)
                })
                .collect();
            let next = PlanProblem::new(problem.world, start, problem.goal.clone());
            candidates.push(Recovery {
                parent,
                phase,
                problem: next,
            });
        }
    }
    let starts: Vec<f32> = candidates
        .iter()
        .flat_map(|c| c.problem.start.iter().copied())
        .collect();
    let item_world: Vec<u32> = candidates.iter().map(|c| c.problem.world).collect();
    let clear = device.clearance(worlds, &item_world, &starts)?;
    Ok(candidates
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| device::collision_free(clear[i]))
        .map(|(_, c)| c)
        .collect())
}

/// States to start episodes from, such as a reinforcement learner's reset bank: samples of
/// `demos` with the joint noise of `o`, at rest, kept where they are collision-free. A carried
/// object in a closed hand moves with that hand, in the grip the hand last holds it in alone;
/// samples where several hands hold it are skipped, since noise would pull them apart.
///
/// Each state is a one-sample demonstration with its sample's world, task, goal and grippers,
/// whose origin is `Origin::Recovery` with `parent` an index into `demos`; `npy::export` writes
/// them as it writes demonstrations. The check is conservative: an object no hand holds is an
/// obstacle even to fingers about to close around it, and a held one must clear the scene, so
/// states at a grasp or a set-down are rare.
pub fn reset_states(
    device: &Device,
    worlds: &[World],
    demos: &[Demonstration],
    o: &RecoveryOptions,
) -> Result<Vec<Demonstration>> {
    if demos.is_empty() {
        return Ok(vec![]);
    }
    let robot = device.robot();
    check_demos(robot, worlds, demos)?;
    let (lo, hi) = o.phase;
    ensure_input!(
        0.0 <= lo && lo <= hi && hi <= 1.0 && o.sigma.is_finite() && o.sigma >= 0.0,
        "reset states need phases within [0, 1] and a non-negative sigma, got {:?} and {}",
        o.phase,
        o.sigma
    );
    let (n, hands) = (robot.dof(), robot.hands());
    let mut rng = Rng::new(o.rng_seed);
    // Each state with the hand holding its object and the object in that hand's frame, and the
    // world it is checked in: the object where it now is, or left out when a hand holds it.
    let (mut states, mut scenes) = (vec![], vec![]);
    for (parent, d) in demos.iter().enumerate() {
        let t = &d.trajectory;
        let holding = |f: usize| -> Vec<usize> {
            (0..hands)
                .filter(|&h| d.gripper[f * hands + h] == 0.0)
                .collect()
        };
        // Each hand's grip from its last sample holding the object alone: where its gripper has
        // just closed, the object still sits where it was.
        let grips: Vec<Option<Pose>> = (0..hands)
            .map(|h| {
                let c = d.carried.as_ref()?;
                let f = (0..t.len()).rev().find(|&f| holding(f) == [h])?;
                Some(
                    robot
                        .ee_pose(&t.positions[f * n..(f + 1) * n], h)
                        .inverse()
                        .mul_pose(c.poses[f]),
                )
            })
            .collect();
        for _ in 0..o.per_trajectory {
            let phase = rng.range(lo, hi);
            let f = (phase * (t.len() - 1) as f32).round() as usize;
            let at = &t.positions[f * n..(f + 1) * n];
            let q: Vec<f32> = (0..n)
                .map(|j| {
                    (at[j] + o.sigma * rng.normal()).clamp(robot.bounds(j).0, robot.bounds(j).1)
                })
                .collect();
            let mut world = worlds[d.world as usize].clone();
            let mut held = None;
            let carried = match &d.carried {
                None => None,
                Some(c) => {
                    let pose = match holding(f)[..] {
                        [] => {
                            world.obstacles[c.obstacle] =
                                world.obstacles[c.obstacle].placed(c.poses[f]);
                            c.poses[f]
                        }
                        [h] => {
                            let in_hand = grips[h].expect("the hand holds it alone at this sample");
                            let object = world.remove(c.obstacle)?;
                            held = Some((h, object.placed(in_hand)));
                            robot.ee_pose(&q, h).mul_pose(in_hand)
                        }
                        _ => continue,
                    };
                    Some(Carried {
                        obstacle: c.obstacle,
                        poses: vec![pose],
                    })
                }
            };
            let trajectory = JointTrajectory {
                dof: n,
                dt: t.dt,
                duration: 0.0,
                positions: q,
                velocities: vec![0.0; n],
                accelerations: vec![0.0; n],
            };
            let state = Demonstration {
                origin: Origin::Recovery { parent, phase },
                world: d.world,
                task: d.task.clone(),
                goal: d.goal,
                trajectory,
                gripper: d.gripper[f * hands..(f + 1) * hands].to_vec(),
                carried,
            };
            states.push((state, held));
            scenes.push(world);
        }
    }

    // One check per hand and object in it, on a robot holding the object.
    let uploaded = device.upload(&scenes)?;
    let mut groups: Vec<(Hold, Vec<usize>)> = vec![];
    for (i, (_, held)) in states.iter().enumerate() {
        match groups.iter_mut().find(|(g, _)| same_hold(g, held)) {
            Some((_, members)) => members.push(i),
            None => groups.push((held.clone(), vec![i])),
        }
    }
    let mut free = vec![false; states.len()];
    for (held, members) in groups {
        let holder;
        let checker = match held {
            None => device,
            Some((h, object)) => {
                let ee = robot.ee_links[h];
                let touch_links = robot
                    .links
                    .iter()
                    .filter(|l| l.chain == robot.links[ee].chain)
                    .map(|l| l.name.clone())
                    .collect();
                holder = device.with_robot(&robot.attach(&AttachedObject {
                    name: "held object".into(),
                    link: robot.links[ee].name.clone(),
                    shapes: vec![object],
                    touch_links,
                    spheres: Default::default(),
                })?)?;
                &holder
            }
        };
        let item_world: Vec<u32> = members.iter().map(|&i| i as u32).collect();
        let q: Vec<f32> = members
            .iter()
            .flat_map(|&i| states[i].0.trajectory.positions.iter().copied())
            .collect();
        let clear = checker.clearance(&uploaded, &item_world, &q)?;
        for (k, &i) in members.iter().enumerate() {
            free[i] = device::collision_free(clear[k]);
        }
    }
    Ok(states
        .into_iter()
        .zip(free)
        .filter(|&(_, free)| free)
        .map(|((state, _), _)| state)
        .collect())
}

/// A hand and the object it holds, placed in the hand's frame; `None` when nothing is held.
type Hold = Option<(usize, Obstacle)>;

/// Whether two holds are one hand holding the same shape, placed alike to within 0.1 mm and a few
/// milliradians: planned poses differ by rounding. (`Quat::angle_between` turns the rounding of a
/// quaternion's norm into milliradians.)
fn same_hold(a: &Hold, b: &Hold) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some((h, x)), Some((k, y))) => {
            let (p, r) = (x.pose(), y.pose());
            h == k
                && x.placed(Pose::IDENTITY) == y.placed(Pose::IDENTITY)
                && p.position.distance(r.position) < 1e-4
                && p.rotation.dot(r.rotation).abs() > 1.0 - 1e-6
        }
        _ => false,
    }
}

/// Where a demonstration starts.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Origin {
    /// From a start state toward the goal.
    Nominal,
    /// Starts from a perturbed state of demonstration `parent`, `phase` of the way along it: a
    /// replanned recovery (`parent` an index into the same output), or that state alone
    /// (`reset_states`, `parent` an index into the demonstrations it perturbs).
    Recovery {
        /// The demonstration it branches from.
        parent: usize,
        /// Fraction of the parent's path where it starts.
        phase: f32,
    },
}

impl Origin {
    /// The demonstration a recovery branches from; `None` for nominal demonstrations.
    #[must_use]
    pub fn parent(&self) -> Option<usize> {
        match *self {
            Origin::Nominal => None,
            Origin::Recovery { parent, .. } => Some(parent),
        }
    }
}

/// The task of every reach demonstration.
pub const REACH_TASK: &str = "Move the gripper to the target pose.";

/// One time-parameterized episode in `worlds[world]`.
#[derive(Clone, Debug)]
pub struct Demonstration {
    /// How the episode starts.
    pub origin: Origin,
    /// The world the episode happens in.
    pub world: u32,
    /// What the episode does, in words; exporters index the distinct tasks.
    pub task: String,
    /// The pose the episode drives toward: the IK frame's for a reach, the object's for
    /// pick-and-place.
    pub goal: Pose,
    /// The robot's joints, sampled every `trajectory.dt` seconds.
    pub trajectory: JointTrajectory,
    /// Each hand's gripper opening at each sample (`[samples, hands]`): 1 open, 0 closed.
    pub gripper: Vec<f32>,
    /// An obstacle of the world that moves during the episode, such as a carried object.
    pub carried: Option<Carried>,
}

/// An obstacle that moves during an episode: its index among the world's obstacles and its pose
/// (the obstacle's own frame: centre and rotation) at each sample.
#[derive(Clone, Debug)]
pub struct Carried {
    /// Its index among the world's obstacles.
    pub obstacle: usize,
    /// Its pose at each sample of the trajectory.
    pub poses: Vec<Pose>,
}

/// How [`demonstrations`] solves, plans, perturbs and times its episodes.
#[derive(Clone, Copy, Debug)]
pub struct DemoOptions {
    /// IK for the goals.
    pub ik: IkOptions,
    /// Planning, nominal and recovery.
    pub plan: PlanOptions,
    /// Perturbations for the recoveries.
    pub recovery: RecoveryOptions,
    /// Standard deviation of the joint noise added to the default pose for start states (radians).
    /// Starts that collide fall back to the default pose.
    pub start_noise: f32,
    /// Each demonstration's speed scale (a fraction of the fastest timing within the robot's
    /// limits) is drawn uniformly from this range.
    pub speed_scale: (f32, f32),
    /// Sample period of the trajectories (seconds).
    pub dt: f32,
    /// Seed of the start noise and speed scales.
    pub rng_seed: u64,
}

impl Default for DemoOptions {
    fn default() -> Self {
        Self {
            ik: IkOptions::default(),
            plan: PlanOptions::default(),
            recovery: RecoveryOptions::default(),
            start_noise: 0.3,
            speed_scale: (0.6, 1.0),
            dt: 0.05,
            rng_seed: 7,
        }
    }
}

/// Reach-to-pose demonstrations for each goal, followed by recoveries from perturbed states of
/// those demonstrations. Goals without a collision-free IK solution or plan are skipped.
pub fn demonstrations(
    device: &Device,
    worlds: &Worlds,
    goals: &[IkProblem],
    o: &DemoOptions,
) -> Result<Vec<Demonstration>> {
    let robot = device.robot();
    let n = robot.dof();
    let (slowest, fastest) = o.speed_scale;
    ensure_input!(
        slowest > 0.0 && slowest <= fastest && fastest <= 1.0,
        "speed scales must satisfy 0 < slowest <= fastest <= 1, got {:?}",
        o.speed_scale
    );
    ensure_input!(
        o.dt.is_finite() && o.dt > 0.0,
        "dt must be positive, got {}",
        o.dt
    );
    if let Some(g) = goals
        .iter()
        .position(|g| g.targets.first().is_none_or(Option::is_none))
    {
        return Err(input!(
            "goal {g}: reach demonstrations target the first hand"
        ));
    }
    let mut rng = Rng::new(o.rng_seed);
    let ik = ik::solve_ik(device, worlds, goals, &o.ik)?;

    let mut starts: Vec<f32> = (0..goals.len())
        .flat_map(|_| {
            (0..n)
                .map(|j| {
                    (robot.default_q[j] + o.start_noise * rng.normal())
                        .clamp(robot.bounds(j).0, robot.bounds(j).1)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let item_world: Vec<u32> = goals.iter().map(|g| g.world).collect();
    let start_clear = device.clearance(worlds, &item_world, &starts)?;
    let colliding = (0..goals.len()).filter(|&g| !device::collision_free(start_clear[g]));
    for g in colliding {
        starts[g * n..(g + 1) * n].copy_from_slice(&robot.default_q);
    }
    // Nominal plan problem p reaches the target of IK problem goal_of[p].
    let (goal_of, problems): (Vec<usize>, Vec<PlanProblem>) = ik
        .solved()
        .map(|s| {
            let start = starts[s.index * n..(s.index + 1) * n].to_vec();
            (
                s.index,
                PlanProblem::new(s.problem.world, start, s.solution),
            )
        })
        .unzip();
    let nominal = trajopt::plan(device, worlds, &problems, &o.plan)?;
    let recoveries = recovery_problems(device, worlds, &nominal, &o.recovery)?;
    let recovery_plans: Vec<PlanProblem> = recoveries.iter().map(|r| r.problem.clone()).collect();
    let recovered = trajopt::plan(device, worlds, &recovery_plans, &o.plan)?;

    // Speed scales are drawn in order, then the paths are timed in parallel (see `timed`).
    let mut scale = || rng.range(o.speed_scale.0, o.speed_scale.1);
    let nominal_paths: Vec<(&[f32], f32)> =
        nominal.solved().map(|s| (s.solution, scale())).collect();
    let recovery_paths: Vec<(&[f32], f32)> =
        recovered.solved().map(|s| (s.solution, scale())).collect();
    let mut nominal_timed = timed(robot, &nominal_paths, o.dt)?.into_iter();
    let mut recovery_timed = timed(robot, &recovery_paths, o.dt)?.into_iter();
    let reach = |origin, world, goal, trajectory: JointTrajectory| Demonstration {
        origin,
        world,
        task: REACH_TASK.into(),
        goal,
        gripper: vec![1.0; trajectory.len() * robot.hands()],
        trajectory,
        carried: None,
    };
    let mut demos = vec![];
    // Demonstration index of each solved nominal problem, for recovery parents.
    let mut demo_of = vec![usize::MAX; problems.len()];
    for s in nominal.solved() {
        demo_of[s.index] = demos.len();
        let goal = goals[goal_of[s.index]].targets[0].expect("reach goals target the first hand");
        demos.push(reach(
            Origin::Nominal,
            s.problem.world,
            goal,
            nominal_timed.next().expect("one per path"),
        ));
    }
    for s in recovered.solved() {
        let rec = &recoveries[s.index];
        let parent = demo_of[rec.parent];
        let goal = demos[parent].goal;
        let trajectory = recovery_timed.next().expect("one per path");
        demos.push(reach(
            Origin::Recovery {
                parent,
                phase: rec.phase,
            },
            s.problem.world,
            goal,
            trajectory,
        ));
    }
    Ok(demos)
}

/// The shared sample period of demonstrations to export, after checking that each one belongs to
/// one of `worlds`, has the robot's joints, holds whole samples with a gripper opening per hand
/// each and any carried obstacle's pose each, and shares that period.
pub(crate) fn check_demos(robot: &Robot, worlds: &[World], demos: &[Demonstration]) -> Result<f32> {
    ensure_input!(!demos.is_empty(), "no demonstrations to export");
    let dt = demos[0].trajectory.dt;
    for (i, d) in demos.iter().enumerate() {
        let t = &d.trajectory;
        ensure_input!(
            (d.world as usize) < worlds.len(),
            "demonstration {i} is in world {}, past the end",
            d.world
        );
        ensure_input!(
            t.dof == robot.dof(),
            "demonstration {i} has {} joints, the robot has {}",
            t.dof,
            robot.dof()
        );
        ensure_input!(
            !t.positions.is_empty()
                && t.positions.len() == t.velocities.len()
                && t.positions.len() % t.dof == 0,
            "demonstration {i} does not hold whole samples"
        );
        ensure_input!(t.dt == dt, "all demonstrations must share one dt");
        ensure_input!(
            d.gripper.len() == t.len() * robot.hands()
                && d.gripper.iter().all(|g| (0.0..=1.0).contains(g)),
            "demonstration {i} needs a gripper opening in [0, 1] per hand per sample"
        );
        if let Some(c) = &d.carried {
            let obstacles = worlds[d.world as usize].obstacles.len();
            ensure_input!(
                c.obstacle < obstacles,
                "demonstration {i} carries obstacle {} of {obstacles}",
                c.obstacle
            );
            ensure_input!(
                c.poses.len() == t.len(),
                "demonstration {i} needs one carried pose per sample"
            );
        }
    }
    Ok(dt)
}

/// One pick-and-place task: pick up the cuboid `object` of `worlds[world]` from above and set it
/// down with its centre and rotation at `place`, or turned onto itself where it cannot be set down
/// so (half a turn, or a quarter for a square footprint: the box looks the same). Both must be
/// upright (their z axis vertical).
#[derive(Clone, Debug)]
pub struct PickPlace {
    /// The world holding the object.
    pub world: u32,
    /// The object's index among the world's obstacles: a cuboid.
    pub object: usize,
    /// Where the object's centre and rotation end up, or turned onto itself about its z axis.
    pub place: Pose,
    /// Obstacles the object rests on where it lies and where it is set down, by index in the
    /// world: it may touch them as it is lifted off and as it is set down.
    pub support: Vec<usize>,
}

/// How [`pick_and_place`] and [`handover`] grasp, move and time their episodes.
#[derive(Clone, Debug)]
pub struct PickPlaceOptions {
    /// Picking the object up and setting it down ([`crate::pick`], [`crate::place`]): IK, the
    /// planned paths, the straight moves and the hand's speed on them. By default the hand is
    /// uncapped, so the straight moves go as fast as the joints allow, as the planned paths do,
    /// before their speed scales; and a move in or away may be shortened to 30% of its length
    /// where an obstacle blocks the rest: a move in then starts nearer, the planned path going
    /// round the obstacle to it, and a move away ends sooner. A handover's moves between the two
    /// use these too, but its straight moves must cover their whole length.
    pub task: TaskOptions,
    /// How far above the grasp and the place pose the gripper moves in from and backs off to
    /// (meters).
    pub approach: f32,
    /// How far below the object's top the gripper holds it (meters).
    pub grasp_depth: f32,
    /// The widest object the gripper closes on (meters).
    pub max_width: f32,
    /// How long closing or opening the gripper takes (seconds).
    pub gripper_time: f32,
    /// Each motion is slowed by a speed scale drawn uniformly from this range.
    pub speed_scale: (f32, f32),
    /// Sample period of the trajectories (seconds).
    pub dt: f32,
    /// Seed of the speed scales.
    pub rng_seed: u64,
}

impl Default for PickPlaceOptions {
    fn default() -> Self {
        Self {
            task: TaskOptions {
                tool_speed: f32::INFINITY,
                min_fraction: 0.3,
                ..TaskOptions::default()
            },
            approach: 0.1,
            grasp_depth: 0.02,
            max_width: 0.08,
            gripper_time: 0.5,
            speed_scale: (0.6, 1.0),
            dt: 0.05,
            rng_seed: 11,
        }
    }
}

/// Pick-and-place demonstrations, one per task that every step succeeds for; the others are
/// skipped. The robot's IK frame is the grasp point, its z axis pointing out of the gripper and its
/// y axis along the fingers' closing direction (as the Panda's `ee_link` in the examples).
///
/// Each episode is a [`crate::pick`] by the first hand from the default configuration, then a
/// [`crate::place`] from where it ends, each batched over the tasks: a planned path to above the
/// object; straight down to the grasp, the fingers touching the object; close; straight up with
/// it, touching what it rests on; a planned path holding it to above the place pose; straight down
/// onto its supports there; open; straight back up. The grasp is top-down, across the object's
/// narrower side, the hand facing as at the default configuration; tasks that cannot be picked
/// so are tried again with the hand turned half way round, and tasks that cannot be set down at
/// `place` with the object turned onto itself there ([`crate::place_poses`]), each in a second
/// batch only as large as the failures. Each motion is slowed by its own speed scale. The
/// demonstration records the gripper's opening and where the object is at every sample, and a
/// task such as "Pick up the box and put it 20 cm to the left."
pub fn pick_and_place(
    device: &Device,
    worlds: &[World],
    tasks: &[PickPlace],
    o: &PickPlaceOptions,
) -> Result<Vec<Demonstration>> {
    let robot = device.robot();
    let (n, hands) = (robot.dof(), robot.hands());
    check_episodes(o)?;
    let arm = Arm::of(robot, 0);
    let mut objects = vec![];
    let mut picks = vec![];
    for (i, t) in tasks.iter().enumerate() {
        let (object, half) = cuboid(worlds, i, t.world, t.object, o)?;
        ensure_input!(
            upright(&t.place),
            "task {i}: the place pose must be finite and upright, got {:?}",
            t.place
        );
        let grasp = top_down(object, half, o.grasp_depth, 0.0, &arm);
        picks.push(pick_up(
            t.world,
            &robot.default_q,
            0,
            t.object,
            grasp,
            &t.support,
            o,
        ));
        objects.push(object);
    }
    let uploaded = device.upload(worlds)?;
    let pick = |batch: &[Pick]| task::pick(device, &uploaded, batch, &o.task);
    let picked = with_fallback(&picks, turned_grasps, pick)?;

    // Set down in each task's world without its object (index `i`).
    let tasks_objects = tasks.iter().map(|t| (t.world, t.object));
    let clear = device.upload(&without(worlds, tasks_objects)?)?;
    let mut places = vec![];
    let mut holding = vec![];
    for (i, up) in picked.into_iter().enumerate() {
        let Some((_, up)) = up else {
            continue;
        };
        let t = &tasks[i];
        let held = up.held.clone().expect("a pick holds what it picked up");
        places.push(set_down(
            i,
            up.end(),
            None,
            held,
            t.place,
            t.object,
            &t.support,
            o,
        ));
        holding.push((i, up));
    }
    let near = near_seeds(&o.task);
    let place = |batch: &[Place]| task::place(device, &clear, batch, &near);
    let placed = with_fallback(&places, turned_poses, place)?;
    let done: Vec<(usize, TaskPlan, TaskPlan, Pose)> = holding
        .into_iter()
        .zip(placed)
        .filter_map(|((i, up), down)| {
            let (place, down) = down?;
            let at = place.poses[down.candidate];
            Some((i, up, down, at))
        })
        .collect();

    // Speed scales are drawn in order, then the motions are sampled on the device's threads.
    let mut rng = Rng::new(o.rng_seed);
    let motions: Vec<(&Trajectory, f32)> = done
        .iter()
        .flat_map(|(_, up, down, _)| {
            [&up.transit, &up.approach, &up.depart].into_iter().chain([
                &down.transit,
                &down.approach,
                &down.depart,
            ])
        })
        .map(|t| (t, rng.range(o.speed_scale.0, o.speed_scale.1)))
        .collect();
    let mut sampled = sampled(device, &motions, o.dt)?.into_iter();
    let hold = ((o.gripper_time / o.dt).round() as usize).max(1);
    let mut demos = vec![];
    for (i, up, _, at) in &done {
        let mut next = || sampled.next().expect("six motions per task done");
        let [transit, approach, lift, carry, lower, retreat] = [(); 6].map(|_| next());
        // The first hand opens and closes; the others stay open.
        let (open, closed) = (
            vec![1.0; hands],
            (0..hands)
                .map(|h| if h == 0 { 0.0 } else { 1.0 })
                .collect::<Vec<_>>(),
        );
        let mut episode = Episode::new(n, hands, o.dt);
        episode.append(&transit, &open);
        episode.append(&approach, &open);
        let grasped = episode.len();
        episode.hold(hold, &open, &closed);
        episode.append(&lift, &closed);
        episode.append(&carry, &closed);
        episode.append(&lower, &closed);
        let released = episode.len();
        episode.hold(hold, &closed, &open);
        episode.append(&retreat, &open);

        let in_hand = up
            .held
            .as_ref()
            .expect("a pick holds what it picked up")
            .grasp
            .inverse();
        let q = |f: usize| &episode.trajectory.positions[f * n..(f + 1) * n];
        let carried = |f: usize| robot.ee_pose(q(f), 0).mul_pose(in_hand);
        let set_down = carried(released - 1);
        let poses = (0..episode.len())
            .map(|f| match f {
                f if f < grasped + hold => objects[*i],
                f if f < released => carried(f),
                _ => set_down,
            })
            .collect();
        let t = &tasks[*i];
        demos.push(Demonstration {
            origin: Origin::Nominal,
            world: t.world,
            task: describe(objects[*i].position, t.place.position),
            goal: *at,
            trajectory: episode.trajectory,
            gripper: episode.gripper,
            carried: Some(Carried {
                obstacle: t.object,
                poses,
            }),
        });
    }
    Ok(demos)
}

/// One handover: hand `from` picks up the cuboid `object` of `worlds[world]` and brings it to
/// `meet`, where hand `to` takes it; then hand `to` sets it down at `place`. Each hand holds one
/// end of the object's longer horizontal side, from above, the end toward its own arm at `meet`.
/// The object, `meet` and `place` must be upright (their z axis vertical).
#[derive(Clone, Debug)]
pub struct Handover {
    /// The world holding the object.
    pub world: u32,
    /// The object's index among the world's obstacles: a cuboid.
    pub object: usize,
    /// The hand that picks the object up.
    pub from: usize,
    /// The hand that takes it and sets it down.
    pub to: usize,
    /// Where the object is handed over (its centre and rotation), held by both hands.
    pub meet: Pose,
    /// Where the taker sets it down (its centre and rotation), or turned onto itself about its z
    /// axis where it cannot set it down so.
    pub place: Pose,
    /// Obstacles the object rests on where it lies and where it is set down, by index in the
    /// world: it may touch them as it is lifted off and as it is set down.
    pub support: Vec<usize>,
}

/// How far along the longer side, as a fraction of its half length, each hand holds the object.
const HANDOVER_GRIP: f32 = 0.75;

/// IK seeds where the hand lets go: IK starts from the configuration the carry starts at, which
/// nearly always converges, and the others are spares.
const NEAR_SEEDS: usize = 4;

/// Handover demonstrations between two hands, one per task that every step succeeds for; the
/// others are skipped. The hands' IK frames are as `pick_and_place` needs them, and the options are
/// its options.
///
/// Each episode is batched over the tasks, step by step:
/// - the giver picks the object up ([`crate::pick`]) while the taker waits;
/// - both arms move at once: the giver carries the object to `meet` while the taker comes to
///   above its end, so they arrive together;
/// - the taker moves straight down onto its end (the giver waits) and closes; the giver opens and
///   backs straight off (the taker waits);
/// - the taker sets the object down ([`crate::place`]), the giver going home on the way at the
///   same time ([`Place::idle`](crate::Place::idle)) and waiting there.
///
/// As in `pick_and_place`, a pick that fails is tried again with the giver's hand turned half way
/// round, and a place with the object turned onto itself. The straight moves between must cover
/// their whole length: the taker's runs forward to its grasp.
///
/// Between the pick and the place, the object rides in the giver's hand (`Robot::attach`) to the
/// meeting pose, then stands there while the taker's fingers move onto it and the giver's hand
/// leaves it, both of which may touch it. The demonstration records both grippers and where the
/// object is at every sample, and a task such as "Pass the box from the left hand to the right hand
/// and put it down."
pub fn handover(
    device: &Device,
    worlds: &[World],
    tasks: &[Handover],
    o: &PickPlaceOptions,
) -> Result<Vec<Demonstration>> {
    let robot = device.robot();
    let (n, hands) = (robot.dof(), robot.hands());
    check_episodes(o)?;
    let arms: Vec<Arm> = (0..hands).map(|h| Arm::of(robot, h)).collect();
    let home = robot.default_q.clone();
    // Per task: the object where it lies, the giver's pick, and the taker's grasp in the object's
    // frame.
    let mut objects = vec![];
    let mut picks = vec![];
    let mut takes = vec![];
    for (i, t) in tasks.iter().enumerate() {
        ensure_input!(
            t.from < hands && t.to < hands && t.from != t.to,
            "task {i}: hands {} and {} of {hands}: a handover needs two different hands",
            t.from,
            t.to
        );
        let (object, half) = cuboid(worlds, i, t.world, t.object, o)?;
        ensure_input!(
            upright(&t.meet) && upright(&t.place),
            "task {i}: the meeting and place poses must be finite and upright, got {:?} and {:?}",
            t.meet,
            t.place
        );
        // The end toward each arm at `meet`: the giver's in the object's frame.
        let long = if half.x <= half.y { Vec3::Y } else { Vec3::X };
        let toward = (t.meet.rotation * long)
            .dot(arms[t.from].base - t.meet.position)
            .signum();
        let along = toward * HANDOVER_GRIP * half.x.max(half.y);
        let giver = top_down(object, half, o.grasp_depth, along, &arms[t.from]);
        picks.push(pick_up(
            t.world, &home, t.from, t.object, giver, &t.support, o,
        ));
        takes.push(top_down(t.meet, half, o.grasp_depth, -along, &arms[t.to]));
        objects.push(object);
    }
    let uploaded = device.upload(worlds)?;
    let pick = |batch: &[Pick]| task::pick(device, &uploaded, batch, &o.task);
    let picked = with_fallback(&picks, turned_grasps, pick)?;
    // Everything after the pick happens in each task's world without its object (index `i`).
    let tasks_objects = tasks.iter().map(|t| (t.world, t.object));
    let clear_worlds = without(worlds, tasks_objects)?;
    let clear = device.upload(&clear_worlds)?;

    /// A task on its way from the pick to the place.
    struct Exchange {
        task: usize,
        up: TaskPlan,
        /// The giver's grasp in the object's frame.
        giver: Pose,
        /// Its world while the object stands at the meeting pose.
        standing: u32,
        /// Both arms to the meeting pose, then the taker in and the giver off, as control points.
        meet: Vec<f32>,
        take: Vec<f32>,
        leave: Vec<f32>,
    }
    let mut going: Vec<Exchange> = picked
        .into_iter()
        .enumerate()
        .filter_map(|(task, up)| {
            let (_, up) = up?;
            let giver = up
                .held
                .as_ref()
                .expect("a pick holds what it picked up")
                .grasp;
            Some(Exchange {
                task,
                up,
                giver,
                standing: 0,
                meet: vec![],
                take: vec![],
                leave: vec![],
            })
        })
        .collect();

    // Both hands at the meeting pose, the taker above its end, from where the pick ends.
    let meeting: Vec<IkProblem> = going
        .iter()
        .map(|e| {
            let t = &tasks[e.task];
            let mut targets = vec![None; hands];
            targets[t.from] = Some(t.meet.mul_pose(e.giver));
            targets[t.to] = Some(task::behind(t.meet.mul_pose(takes[e.task]), o.approach));
            IkProblem {
                world: e.task as u32,
                targets,
                seed: Some(e.up.end().to_vec()),
            }
        })
        .collect();
    let met = ik::solve_ik(device, &clear, &meeting, &o.task.ik)?;
    for (k, e) in going.iter_mut().enumerate() {
        if let Some(goal) = met.nearest(k, e.up.end()) {
            e.meet = goal.to_vec();
        }
    }
    going.retain(|e| !e.meet.is_empty());

    // There both at once, holding the object as the giver grasps it: a device per grasp.
    let mut holders: Vec<(Held, Device, Vec<usize>)> = vec![];
    for (k, e) in going.iter().enumerate() {
        let held = e.up.held.clone().expect("a pick holds what it picked up");
        match holders.iter_mut().find(|(h, _, _)| *h == held) {
            Some((_, _, alike)) => alike.push(k),
            None => {
                let holder = device.with_robot(&robot.attach(&held.attachment(robot)?)?)?;
                holders.push((held, holder, vec![k]));
            }
        }
    }
    let mut planned = vec![false; going.len()];
    for (_, holder, alike) in &holders {
        let problems: Vec<PlanProblem> = alike
            .iter()
            .map(|&k| {
                let e = &going[k];
                PlanProblem::new(e.task as u32, e.up.end(), e.meet.as_slice())
            })
            .collect();
        for solved in trajopt::plan(holder, &clear, &problems, &o.task.plan)?.solved() {
            let k = alike[solved.index];
            going[k].meet = solved.solution.to_vec();
            planned[k] = true;
        }
    }
    let mut planned = planned.into_iter();
    going.retain(|_| planned.next().expect("a flag per task going"));

    // The object stands where the giver holds it at the meeting pose, and both hands may touch it:
    // the giver's, and the taker's fingers.
    let mut standing = vec![];
    for e in &mut going {
        let t = &tasks[e.task];
        let at = robot
            .ee_pose(last(&e.meet, n), t.from)
            .mul_pose(e.giver.inverse());
        let mut w = clear_worlds[e.task].clone();
        w.obstacles
            .push(worlds[t.world as usize].obstacles[t.object].placed(at));
        let links = robot
            .hand_links(t.from)
            .into_iter()
            .chain(task::fingers(robot, t.to));
        w.touch.push(Touch::new(w.obstacles.len() - 1, links));
        e.standing = standing.len() as u32;
        standing.push(w);
    }
    let standing = device.upload(&standing)?;
    // The taker straight down onto its end, then the giver straight back off it.
    let straight = |paths: Vec<ToolPath>| -> Result<Vec<Option<Vec<f32>>>> {
        let moved = tool_path::follow(device, &standing, &paths, &o.task.tool_path)?;
        let whole = moved.paths.into_iter().zip(moved.fraction);
        Ok(whole
            .map(|(path, fraction)| (fraction == 1.0 && !path.is_empty()).then_some(path))
            .collect())
    };
    let path = |e: &Exchange, hand: usize, start: &[f32], to: Pose| ToolPath {
        world: e.standing,
        start: start.to_vec(),
        hand,
        waypoints: vec![to],
    };
    let taking = going.iter().map(|e| {
        let object = standing[e.standing as usize]
            .obstacles
            .last()
            .expect("the object stands there")
            .pose();
        let to = object.mul_pose(takes[e.task]);
        path(e, tasks[e.task].to, last(&e.meet, n), to)
    });
    let mut taken = straight(taking.collect())?.into_iter();
    going.retain_mut(|e| match taken.next().flatten() {
        Some(take) => {
            e.take = take;
            true
        }
        None => false,
    });
    let leaving = going.iter().map(|e| {
        let (from, at) = (tasks[e.task].from, last(&e.take, n));
        path(
            e,
            from,
            at,
            task::behind(robot.ee_pose(at, from), o.approach),
        )
    });
    let mut left = straight(leaving.collect())?.into_iter();
    going.retain_mut(|e| match left.next().flatten() {
        Some(leave) => {
            e.leave = leave;
            true
        }
        None => false,
    });

    // The taker sets it down, the giver going home on the way.
    let places: Vec<Place> = going
        .iter()
        .map(|e| {
            let t = &tasks[e.task];
            let held = Held {
                hand: t.to,
                object: worlds[t.world as usize].obstacles[t.object].placed(Pose::IDENTITY),
                grasp: takes[e.task],
            };
            let start = last(&e.leave, n);
            set_down(
                e.task,
                start,
                Some(&home),
                held,
                t.place,
                t.object,
                &t.support,
                o,
            )
        })
        .collect();
    let near = near_seeds(&o.task);
    let place = |batch: &[Place]| task::place(device, &clear, batch, &near);
    let placed = with_fallback(&places, turned_poses, place)?;
    let done: Vec<(Exchange, TaskPlan, Pose)> = going
        .into_iter()
        .zip(placed)
        .filter_map(|(e, down)| {
            let (place, down) = down?;
            let at = place.poses[down.candidate];
            Some((e, down, at))
        })
        .collect();

    // The exchange's motions are timed as the tasks' are, then every motion is slowed by a speed
    // scale drawn in order and sampled on the device's threads.
    let speed = |hand: usize| HandSpeed {
        hand,
        linear: o.task.tool_speed,
        angular: f32::INFINITY,
    };
    let exchanges: Vec<[Trajectory; 3]> = device.install(|| {
        done.par_iter()
            .map(|(e, _, _)| {
                let t = &tasks[e.task];
                Ok([
                    Trajectory::try_new(robot, &e.meet, 1.0)?,
                    Trajectory::with_hand_speed(robot, &e.take, &speed(t.to))?,
                    Trajectory::with_hand_speed(robot, &e.leave, &speed(t.from))?,
                ])
            })
            .collect::<Result<_>>()
    })?;
    let mut rng = Rng::new(o.rng_seed);
    let motions: Vec<(&Trajectory, f32)> = done
        .iter()
        .zip(&exchanges)
        .flat_map(|((e, down, _), exchange)| {
            [&e.up.transit, &e.up.approach, &e.up.depart]
                .into_iter()
                .chain(exchange)
                .chain([&down.transit, &down.approach, &down.depart])
        })
        .map(|t| (t, rng.range(o.speed_scale.0, o.speed_scale.1)))
        .collect();
    let mut sampled = sampled(device, &motions, o.dt)?.into_iter();
    let hold = ((o.gripper_time / o.dt).round() as usize).max(1);
    let mut demos = vec![];
    for (e, _, at) in &done {
        let t = &tasks[e.task];
        let mut next = || sampled.next().expect("nine motions per task done");
        let [
            transit,
            approach,
            lift,
            to_meet,
            take,
            leave,
            carry,
            lower,
            retreat,
        ] = [(); 9].map(|_| next());
        // Both grippers, open (1) or closed (0): (giver, taker).
        let grippers = |giver: f32, taker: f32| -> Vec<f32> {
            (0..hands)
                .map(|h| {
                    if h == t.from {
                        giver
                    } else if h == t.to {
                        taker
                    } else {
                        1.0
                    }
                })
                .collect()
        };
        let mut episode = Episode::new(n, hands, o.dt);
        episode.append(&transit, &grippers(1.0, 1.0));
        episode.append(&approach, &grippers(1.0, 1.0));
        let grasped = episode.len();
        episode.hold(hold, &grippers(1.0, 1.0), &grippers(0.0, 1.0));
        episode.append(&lift, &grippers(0.0, 1.0));
        episode.append(&to_meet, &grippers(0.0, 1.0));
        episode.append(&take, &grippers(0.0, 1.0));
        episode.hold(hold, &grippers(0.0, 1.0), &grippers(0.0, 0.0));
        episode.hold(hold, &grippers(0.0, 0.0), &grippers(1.0, 0.0));
        let passed = episode.len();
        episode.append(&leave, &grippers(1.0, 0.0));
        episode.append(&carry, &grippers(1.0, 0.0));
        episode.append(&lower, &grippers(1.0, 0.0));
        let released = episode.len();
        episode.hold(hold, &grippers(1.0, 0.0), &grippers(1.0, 1.0));
        episode.append(&retreat, &grippers(1.0, 1.0));

        let (giver, taker) = (e.giver.inverse(), takes[e.task].inverse());
        let q = |f: usize| &episode.trajectory.positions[f * n..(f + 1) * n];
        let held = |f: usize| {
            if f < passed {
                robot.ee_pose(q(f), t.from).mul_pose(giver)
            } else {
                robot.ee_pose(q(f), t.to).mul_pose(taker)
            }
        };
        let set_down = held(released - 1);
        let poses = (0..episode.len())
            .map(|f| match f {
                f if f < grasped + hold => objects[e.task],
                f if f < released => held(f),
                _ => set_down,
            })
            .collect();
        demos.push(Demonstration {
            origin: Origin::Nominal,
            world: t.world,
            task: describe_handover(robot, t.from, t.to),
            goal: *at,
            trajectory: episode.trajectory,
            gripper: episode.gripper,
            carried: Some(Carried {
                obstacle: t.object,
                poses,
            }),
        });
    }
    Ok(demos)
}

/// "Pass the box from the left hand to the right hand and put it down.", the hands named after
/// their parts' prefixes, else numbered.
fn describe_handover(robot: &Robot, from: usize, to: usize) -> String {
    let name = |h: usize| match robot.hand_prefix(h) {
        Some(prefix) => format!("the {} hand", prefix.trim_end_matches(['_', '/', '-', '.'])),
        None => format!("hand {h}"),
    };
    format!(
        "Pass the box from {} to {} and put it down.",
        name(from),
        name(to)
    )
}

/// Each path timed at its speed scale and sampled every `dt`, in parallel on the current rayon
/// pool: timing is most of a generator's work once its planning is batched.
fn timed(robot: &Robot, paths: &[(&[f32], f32)], dt: f32) -> Result<Vec<JointTrajectory>> {
    paths
        .par_iter()
        .map(|&(path, speed_scale)| Trajectory::try_new(robot, path, speed_scale)?.sample(1.0 / dt))
        .collect()
}

/// Each motion slowed by its speed scale and sampled every `dt`, on `device`'s threads: sampling
/// is much of what is left once the motions are planned.
fn sampled(
    device: &Device,
    motions: &[(&Trajectory, f32)],
    dt: f32,
) -> Result<Vec<JointTrajectory>> {
    device.install(|| {
        motions
            .par_iter()
            .map(|&(t, speed_scale)| {
                let slowed = Trajectory {
                    knot_interval: t.knot_interval / speed_scale,
                    ..t.clone()
                };
                slowed.sample(1.0 / dt)
            })
            .collect()
    })
}

/// That the options make episodes: speed scales in (0, 1], positive sizes and times.
fn check_episodes(o: &PickPlaceOptions) -> Result<()> {
    let (slowest, fastest) = o.speed_scale;
    ensure_input!(
        slowest > 0.0 && slowest <= fastest && fastest <= 1.0,
        "speed scales must satisfy 0 < slowest <= fastest <= 1, got {:?}",
        o.speed_scale
    );
    let positive = |v: f32| v.is_finite() && v > 0.0;
    ensure_input!(
        positive(o.dt)
            && positive(o.approach)
            && positive(o.max_width)
            && o.gripper_time >= 0.0
            && o.grasp_depth >= 0.0,
        "episodes need a positive dt, approach and max_width and a non-negative gripper_time and grasp_depth, got {}, {}, {}, {} and {}",
        o.dt,
        o.approach,
        o.max_width,
        o.gripper_time,
        o.grasp_depth
    );
    Ok(())
}

/// Task `i`'s object, obstacle `object` of world `world`, as its pose and half sizes: an upright
/// cuboid narrow enough for the gripper.
fn cuboid(
    worlds: &[World],
    i: usize,
    world: u32,
    object: usize,
    o: &PickPlaceOptions,
) -> Result<(Pose, Vec3)> {
    let w = worlds
        .get(world as usize)
        .ok_or_else(|| input!("task {i}: world {world} of {}", worlds.len()))?;
    let Some(&Obstacle::Cuboid {
        center,
        half_extents: half,
        rotation,
    }) = w.obstacles.get(object)
    else {
        return Err(input!(
            "task {i}: obstacle {object} of world {world} is not a cuboid"
        ));
    };
    let pose = Pose {
        position: center,
        rotation,
    };
    ensure_input!(
        upright(&pose),
        "task {i}: the object must be upright, got rotation {rotation}"
    );
    let width = 2.0 * half.x.min(half.y);
    ensure_input!(
        width <= o.max_width,
        "task {i}: the object is {width} m across, wider than the gripper's {} m",
        o.max_width
    );
    Ok((pose, half))
}

/// Whether `pose` is finite, with a unit rotation that keeps its z axis vertical.
fn upright(pose: &Pose) -> bool {
    let r = pose.rotation;
    pose.position.is_finite()
        && r.is_finite()
        && (r.length() - 1.0).abs() < 1e-3
        && (r * Vec3::Z).dot(Vec3::Z) > 1.0 - 1e-4
}

/// Each task run, then those that fail run again with `others` in their place, in a second batch
/// only as large as the failures: per task, the task as it got through and its quickest plan. The
/// grasp or pose asked for wins wherever it gets through.
fn with_fallback<T: Clone>(
    tasks: &[T],
    others: fn(&T) -> Option<T>,
    run: impl Fn(&[T]) -> Result<TaskResult<T>>,
) -> Result<Vec<Option<(T, TaskPlan)>>> {
    let quickest = |result: TaskResult<T>| -> Vec<Option<(T, TaskPlan)>> {
        let outcomes = result.problems.into_iter().zip(result.outcomes);
        outcomes
            .map(|(t, outcome)| Some((t, outcome.ok()?.into_iter().next()?)))
            .collect()
    };
    let mut done = quickest(run(tasks)?);
    let (again, which): (Vec<T>, Vec<usize>) = (0..tasks.len())
        .filter(|&k| done[k].is_none())
        .filter_map(|k| Some((others(&tasks[k])?, k)))
        .unzip();
    if !again.is_empty() {
        for (got, k) in quickest(run(&again)?).into_iter().zip(which) {
            done[k] = got;
        }
    }
    Ok(done)
}

/// `o` with IK keeping `NEAR_SEEDS` seeds.
fn near_seeds(o: &TaskOptions) -> TaskOptions {
    TaskOptions {
        ik: IkOptions {
            seeds: o.ik.seeds.min(NEAR_SEEDS),
            ..o.ik
        },
        ..o.clone()
    }
}

/// A pick's grasps half a turn round about the hand's z axis: closing the other way round.
fn turned_grasps(pick: &Pick) -> Option<Pick> {
    let turn = Quat::from_rotation_z(PI);
    let grasps = pick
        .grasps
        .iter()
        .map(|g| Pose::new(g.position, g.rotation * turn));
    Some(Pick {
        grasps: grasps.collect(),
        ..pick.clone()
    })
}

/// A place's pose turned onto itself (`place_poses`: half a turn for a box, quarter turns for a
/// square footprint), if the object has such turns.
fn turned_poses(place: &Place) -> Option<Place> {
    let turns = task::place_poses(&place.held.object, place.poses[0], 1);
    (turns.len() > 1).then(|| Place {
        poses: turns[1..].to_vec(),
        ..place.clone()
    })
}

/// Hand `hand` picking up obstacle `object` of world `world` from `start` by `grasp`, moving in and
/// lifting `o.approach`, off `support`.
fn pick_up(
    world: u32,
    start: &[f32],
    hand: usize,
    object: usize,
    grasp: Pose,
    support: &[usize],
    o: &PickPlaceOptions,
) -> Pick {
    Pick {
        hand,
        approach: o.approach,
        lift: Vec3::Z * o.approach,
        support: support.to_vec(),
        ..Pick::new(world, start, object, vec![grasp])
    }
}

/// Setting `held` down at `pose` from `start` in world `world`, one without the object (obstacle
/// `object` of the world it came from, whose indices `support` uses), lowering it and backing off
/// `o.approach`, the joints only other hands move going to `idle` on the way.
#[expect(
    clippy::too_many_arguments,
    reason = "the two generators set down alike from their own tasks"
)]
fn set_down(
    world: usize,
    start: &[f32],
    idle: Option<&[f32]>,
    held: Held,
    pose: Pose,
    object: usize,
    support: &[usize],
    o: &PickPlaceOptions,
) -> Place {
    let renumbered = support.iter().map(|&s| if s > object { s - 1 } else { s });
    Place {
        idle: idle.map(<[f32]>::to_vec),
        lower: Vec3::Z * -o.approach,
        retreat: o.approach,
        support: renumbered.collect(),
        ..Place::new(world as u32, start, held, vec![pose])
    }
}

/// Each task's world (`(world, object)`) without its object.
fn without(worlds: &[World], tasks: impl Iterator<Item = (u32, usize)>) -> Result<Vec<World>> {
    tasks
        .map(|(world, object)| {
            let mut w = worlds[world as usize].clone();
            w.remove(object)?;
            Ok(w)
        })
        .collect()
}

/// The last configuration of control points `path`, of `n` joints each.
fn last(path: &[f32], n: usize) -> &[f32] {
    &path[path.len() - n..]
}

/// Where a hand's arm stands, and which way its hand faces at the default configuration: its x
/// axis away from the arm's base (the Panda's) or toward it (the reBot's). Grasps face the same
/// way, so they need no half turn of the wrist.
struct Arm {
    base: Vec3,
    outward: bool,
}

impl Arm {
    fn of(robot: &Robot, hand: usize) -> Self {
        let fk = robot.fk(&robot.default_q);
        let ee = robot.ee_links[hand];
        let base = robot.links[ee]
            .chain
            .first()
            .map_or(Vec3::ZERO, |&i| fk.pos[i]);
        let outward = fk.rot[ee].x_axis.dot((fk.pos[ee] - base).with_z(0.0)) >= 0.0;
        Self { base, outward }
    }
}

/// A top-down grasp of the upright cuboid of half sizes `half` at `object`, in its frame: `along`
/// meters from its center along the longer horizontal side, `depth` below its top, closing across
/// the narrower side, the hand facing as `arm`'s does. Built in the object's frame, so objects of
/// one size held alike share a `Held`, and with it a holder device.
fn top_down(object: Pose, half: Vec3, depth: f32, along: f32, arm: &Arm) -> Pose {
    let (narrow, long) = if half.x <= half.y {
        (Vec3::X, Vec3::Y)
    } else {
        (Vec3::Y, Vec3::X)
    };
    let z = -Vec3::Z;
    let sideways = (object.rotation * narrow).cross(z);
    let y = if (sideways.dot((object.position - arm.base).with_z(0.0)) >= 0.0) == arm.outward {
        narrow
    } else {
        -narrow
    };
    Pose {
        position: long * along + Vec3::Z * (half.z - depth.min(half.z)),
        rotation: Quat::from_mat3(&Mat3::from_cols(y.cross(z), y, z)),
    }
}

/// "Pick up the box and put it 20 cm to the left.": the move along its larger horizontal axis,
/// in the robot base's frame (x forward, y left), to the nearest 5 cm.
fn describe(from: Vec3, to: Vec3) -> String {
    let d = to - from;
    let (along, direction) = if d.x.abs() >= d.y.abs() {
        (d.x, if d.x >= 0.0 { "forward" } else { "back" })
    } else {
        (
            d.y,
            if d.y >= 0.0 {
                "to the left"
            } else {
                "to the right"
            },
        )
    };
    let cm = ((along.abs() * 20.0).round() * 5.0).max(5.0);
    format!("Pick up the box and put it {cm} cm {direction}.")
}

/// A demonstration being assembled from timed segments that start and end at rest, with each
/// hand's gripper opening per sample.
struct Episode {
    trajectory: JointTrajectory,
    gripper: Vec<f32>,
    hands: usize,
}

impl Episode {
    fn new(dof: usize, hands: usize, dt: f32) -> Self {
        let trajectory = JointTrajectory {
            dof,
            dt,
            duration: 0.0,
            positions: vec![],
            velocities: vec![],
            accelerations: vec![],
        };
        Self {
            trajectory,
            gripper: vec![],
            hands,
        }
    }

    fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Appends `segment` with each hand's gripper at `openings`; its first sample, at rest where
    /// the episode ends, is dropped after the first segment.
    fn append(&mut self, segment: &JointTrajectory, openings: &[f32]) {
        let n = self.trajectory.dof;
        let skip = usize::from(!self.trajectory.positions.is_empty());
        let t = &mut self.trajectory;
        t.positions
            .extend_from_slice(&segment.positions[skip * n..]);
        t.velocities
            .extend_from_slice(&segment.velocities[skip * n..]);
        t.accelerations
            .extend_from_slice(&segment.accelerations[skip * n..]);
        while self.gripper.len() < t.positions.len() / n * self.hands {
            self.gripper.extend_from_slice(openings);
        }
        t.duration = (t.len() - 1) as f32 * t.dt;
    }

    /// Holds the arms still for `samples` samples while each hand's gripper moves from `from` to
    /// `to`.
    fn hold(&mut self, samples: usize, from: &[f32], to: &[f32]) {
        let n = self.trajectory.dof;
        let t = &mut self.trajectory;
        let last = t.positions[t.positions.len() - n..].to_vec();
        for k in 1..=samples {
            t.positions.extend_from_slice(&last);
            t.velocities.extend(std::iter::repeat_n(0.0, n));
            t.accelerations.extend(std::iter::repeat_n(0.0, n));
            let u = k as f32 / samples as f32;
            self.gripper
                .extend(from.iter().zip(to).map(|(a, b)| a + (b - a) * u));
        }
        t.duration = (t.len() - 1) as f32 * t.dt;
    }
}
