//! Planner output as policy-training demonstrations: nominal reaches plus recoveries from
//! perturbed states, timed within the robot's limits at randomized speeds.

use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

use crate::device::{self, Device, Worlds};
use crate::error::{Result, ensure_input, input};
use crate::ik::{self, IkOptions, IkProblem, IkResult};
use crate::rng::Rng;
use crate::robot::{AttachedObject, Robot};
use crate::spline;
use crate::timing::Trajectory;
use crate::trajopt::{self, PlanOptions, PlanProblem, PlanResult};
use crate::types::{JointTrajectory, Pose, Solved};
use crate::world::{Obstacle, World};

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
/// down with its centre and rotation at `place`. Both must be upright (their z axis vertical).
#[derive(Clone, Debug)]
pub struct PickPlace {
    /// The world holding the object.
    pub world: u32,
    /// The object's index among the world's obstacles: a cuboid.
    pub object: usize,
    /// Where the object's centre and rotation end up.
    pub place: Pose,
}

/// How [`pick_and_place`] and [`handover`] grasp, plan and time their episodes.
#[derive(Clone, Copy, Debug)]
pub struct PickPlaceOptions {
    /// IK for the grasps and the poses between them.
    pub ik: IkOptions,
    /// Planning the segments.
    pub plan: PlanOptions,
    /// How far above the grasp and the place pose the gripper approaches from and retreats to
    /// (meters).
    pub approach: f32,
    /// How far below the object's top the gripper holds it (meters).
    pub grasp_depth: f32,
    /// The widest object the gripper closes on (meters).
    pub max_width: f32,
    /// How long closing or opening the gripper takes (seconds).
    pub gripper_time: f32,
    /// Each segment's speed scale is drawn uniformly from this range.
    pub speed_scale: (f32, f32),
    /// Sample period of the trajectories (seconds).
    pub dt: f32,
    /// Seed of the speed scales.
    pub rng_seed: u64,
}

impl Default for PickPlaceOptions {
    fn default() -> Self {
        Self {
            ik: IkOptions::default(),
            plan: PlanOptions::default(),
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
/// Each episode chains planned segments, all from rest to rest: approach a pose above the object
/// (which is an obstacle) from the default configuration; descend to the grasp (the object no
/// longer an obstacle, as the fingers close around it); close the gripper; lift back up; transfer
/// to above the place pose holding the object (`Robot::attach`, touching only the links fixed to
/// the wrist); lower it; open; retreat. The grasp is top-down, across the object's narrower side.
/// The demonstration records the gripper's opening and where the object is at every sample, and
/// a task such as "Pick up the box and put it 20 cm to the left."
pub fn pick_and_place(
    device: &Device,
    worlds: &[World],
    tasks: &[PickPlace],
    o: &PickPlaceOptions,
) -> Result<Vec<Demonstration>> {
    let robot = device.robot();
    let (n, hands) = (robot.dof(), robot.hands());
    // The first hand does the work; the joints only the others move stay put.
    let first = |pose: Pose| -> Vec<Option<Pose>> {
        (0..hands).map(|h| (h == 0).then_some(pose)).collect()
    };
    let hold = robot.idle_joints(0);
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
        "pick-and-place needs positive dt, approach and max_width and non-negative gripper_time and grasp_depth"
    );
    let upright = |r: Quat| (r * Vec3::Z).dot(Vec3::Z) > 1.0 - 1e-4;
    let arm = Arm::of(robot, 0);
    let mut grasps = vec![];
    for (i, t) in tasks.iter().enumerate() {
        let world = worlds
            .get(t.world as usize)
            .ok_or_else(|| input!("task {i}: world {} of {}", t.world, worlds.len()))?;
        let Some(&Obstacle::Cuboid {
            center,
            half_extents: half,
            rotation,
        }) = world.obstacles.get(t.object)
        else {
            return Err(input!(
                "task {i}: obstacle {} of world {} is not a cuboid",
                t.object,
                t.world
            ));
        };
        ensure_input!(
            upright(rotation) && upright(t.place.rotation),
            "task {i}: the object and the place pose must be upright"
        );
        ensure_input!(
            2.0 * half.x.min(half.y) <= o.max_width,
            "task {i}: the object is wider than the gripper opens"
        );
        grasps.push(Grasp::new(
            Pose {
                position: center,
                rotation,
            },
            half,
            o.grasp_depth,
            &arm,
        ));
    }

    // Each task's world as given (index i) and without its object (index tasks + i).
    let without = |t: &PickPlace| -> Result<World> {
        let mut w = worlds[t.world as usize].clone();
        w.remove(t.object)?;
        Ok(w)
    };
    let scene: Vec<World> = tasks
        .iter()
        .map(|t| Ok(worlds[t.world as usize].clone()))
        .chain(tasks.iter().map(without))
        .collect::<Result<_>>()?;
    let uploaded = device.upload(&scene)?;
    let clear = |i: usize| (tasks.len() + i) as u32;
    let up = Vec3::Z * o.approach;
    let at = |p: Pose, lift: Vec3| Pose::new(p.position + lift, p.rotation);
    let place_tool: Vec<Pose> = tasks
        .iter()
        .zip(&grasps)
        .map(|(t, g)| g.tool_for(t.place))
        .collect();

    // Above the object, then the grasp from there.
    let above: Vec<IkProblem> = grasps
        .iter()
        .enumerate()
        .map(|(i, g)| IkProblem {
            world: i as u32,
            targets: first(at(g.tool, up)),
            seed: None,
        })
        .collect();
    let above = ik::solve_ik(device, &uploaded, &above, &o.ik)?;
    // The solution nearest the start, not the one with the most clearance: every later pose is
    // seeded from this one, and a far branch (joints near opposite limits) makes the arm swing
    // around for moves of a few centimetres.
    let above: Vec<Option<Vec<f32>>> = (0..tasks.len())
        .map(|i| above.nearest(i, &robot.default_q).map(<[f32]>::to_vec))
        .collect();
    let mut ok: Vec<Option<[Vec<f32>; 4]>> = (0..tasks.len()).map(|_| None).collect();
    let seeded = |ik: &IkResult, i: usize, from: &[f32]| ik.nearest(i, from).map(<[f32]>::to_vec);
    // The grasp, and the pose above the place pose, both from above the object, in one batch.
    // Above the place pose the object only hangs below the hand, as it did when lifted, so the
    // bare robot solves it; the transfer's plan, which holds the object, checks the pose before
    // planning to it. (Solved per object size, holding it, this took a GPU call per size.)
    let grasp_and_above_place: Vec<IkProblem> = grasps
        .iter()
        .enumerate()
        .map(|(i, g)| IkProblem {
            world: clear(i),
            targets: first(g.tool),
            seed: above[i].clone(),
        })
        .chain((0..tasks.len()).map(|i| IkProblem {
            world: clear(i),
            targets: first(at(place_tool[i], up)),
            seed: above[i].clone(),
        }))
        .collect();
    let grasp = ik::solve_ik(device, &uploaded, &grasp_and_above_place, &o.ik)?;
    let above_place: Vec<Option<Vec<f32>>> = (0..tasks.len())
        .map(|i| {
            above[i]
                .as_deref()
                .and_then(|from| seeded(&grasp, tasks.len() + i, from))
        })
        .collect();

    // Holding each distinct object size (the grasp fixes where the object sits in the IK frame).
    let ee = robot.ee_links[0];
    let touch: Vec<String> = robot
        .links
        .iter()
        .filter(|l| l.chain == robot.links[ee].chain)
        .map(|l| l.name.clone())
        .collect();
    let mut holders: Vec<(Vec3, Device)> = vec![];
    for g in &grasps {
        if holders.iter().all(|(half, _)| *half != g.half) {
            let held = robot.attach(&AttachedObject {
                name: "held object".into(),
                link: robot.links[ee].name.clone(),
                shapes: vec![Obstacle::Cuboid {
                    center: g.object_in_tool.position,
                    half_extents: g.half,
                    rotation: g.object_in_tool.rotation,
                }],
                touch_links: touch.clone(),
                spheres: Default::default(),
            })?;
            holders.push((g.half, device.with_robot(&held)?));
        }
    }

    // The place pose, from above it.
    let place: Vec<IkProblem> = (0..tasks.len())
        .map(|i| IkProblem {
            world: clear(i),
            targets: first(place_tool[i]),
            seed: above_place[i].clone(),
        })
        .collect();
    let place = ik::solve_ik(device, &uploaded, &place, &o.ik)?;
    for i in 0..tasks.len() {
        let q1 = above[i].clone();
        let q2 = q1.as_deref().and_then(|q1| seeded(&grasp, i, q1));
        let q3 = above_place[i].clone();
        let q4 = q3.as_deref().and_then(|q3| seeded(&place, i, q3));
        if let (Some(q1), Some(q2), Some(q3), Some(q4)) = (q1, q2, q3, q4) {
            ok[i] = Some([q1, q2, q3, q4]);
        }
    }

    // The planned segments, in batches over the tasks still going: approach, descend, transfer
    // (holding the object, one batch per object size) and lower.
    let start = robot.default_q.clone();
    let mut paths: [Vec<Vec<f32>>; 4] = Default::default();
    let mut live: Vec<usize> = (0..tasks.len()).filter(|&i| ok[i].is_some()).collect();
    for (s, segment) in paths.iter_mut().enumerate() {
        *segment = vec![vec![]; tasks.len()];
        let ends = |i: usize| {
            let q = ok[i]
                .as_ref()
                .expect("live tasks have their configurations");
            match s {
                0 => (start.clone(), q[0].clone()),
                1 => (q[0].clone(), q[1].clone()),
                2 => (q[0].clone(), q[2].clone()),
                _ => (q[2].clone(), q[3].clone()),
            }
        };
        let groups: Vec<(&Device, Vec<usize>)> = if s == 2 {
            holders
                .iter()
                .map(|(half, held)| {
                    (
                        held,
                        live.iter()
                            .copied()
                            .filter(|&i| grasps[i].half == *half)
                            .collect(),
                    )
                })
                .collect()
        } else {
            vec![(device, live.clone())]
        };
        let mut next = vec![];
        for (d, group) in groups {
            let problems: Vec<PlanProblem> = group
                .iter()
                .map(|&i| {
                    let (start, goal) = ends(i);
                    PlanProblem {
                        world: if s == 0 { i as u32 } else { clear(i) },
                        start,
                        goal,
                        start_motion: None,
                        hold: hold.clone(),
                    }
                })
                .collect();
            for solved in trajopt::plan(d, &uploaded, &problems, &o.plan)?.solved() {
                segment[group[solved.index]] = solved.solution.to_vec();
                next.push(group[solved.index]);
            }
        }
        next.sort_unstable();
        live = next;
    }

    // Speed scales are drawn in order, then the segments are timed in parallel (see `timed`).
    let mut rng = Rng::new(o.rng_seed);
    let segments: Vec<(&[f32], f32)> = live
        .iter()
        .flat_map(|&i| paths.iter().map(move |segment| segment[i].as_slice()))
        .map(|path| (path, rng.range(o.speed_scale.0, o.speed_scale.1)))
        .collect();
    let mut segments = timed(robot, &segments, o.dt)?.into_iter();
    let hold = ((o.gripper_time / o.dt).round() as usize).max(1);
    let mut demos = vec![];
    for i in live {
        let mut next = || segments.next().expect("four segments per live task");
        let (approach, descend, transfer, lower) = (next(), next(), next(), next());
        // The first hand opens and closes; the others stay open.
        let (open, closed) = (
            vec![1.0; hands],
            (0..hands)
                .map(|h| if h == 0 { 0.0 } else { 1.0 })
                .collect::<Vec<_>>(),
        );
        let mut episode = Episode::new(n, hands, o.dt);
        episode.append(&approach, &open);
        episode.append(&descend, &open);
        let grasped = episode.len();
        episode.hold(hold, &open, &closed);
        episode.append(&reversed(&descend), &closed);
        episode.append(&transfer, &closed);
        episode.append(&lower, &closed);
        let released = episode.len();
        episode.hold(hold, &closed, &open);
        episode.append(&reversed(&lower), &open);

        let g = &grasps[i];
        let carried = |q: &[f32]| robot.ee_pose(q, 0).mul_pose(g.object_in_tool);
        let last_held = carried(&episode.trajectory.positions[(released - 1) * n..released * n]);
        let poses = (0..episode.len())
            .map(|f| match f {
                f if f < grasped + hold => g.object,
                f if f < released => carried(&episode.trajectory.positions[f * n..(f + 1) * n]),
                _ => last_held,
            })
            .collect();
        demos.push(Demonstration {
            origin: Origin::Nominal,
            world: tasks[i].world,
            task: describe(g.object.position, tasks[i].place.position),
            goal: tasks[i].place,
            trajectory: episode.trajectory,
            gripper: episode.gripper,
            carried: Some(Carried {
                obstacle: tasks[i].object,
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
    /// Where the taker sets it down (its centre and rotation).
    pub place: Pose,
}

/// IK seeds for a step seeded with the configuration before it, such as a grasp from just above
/// it: that seed almost always succeeds, and the others are spares.
const NEAR_SEEDS: usize = 4;

/// How far along the longer side, as a fraction of its half length, each hand holds the object.
const HANDOVER_GRIP: f32 = 0.75;

/// Handover demonstrations between two hands, one per task that every step succeeds for; the
/// others are skipped. The hands' IK frames are as `pick_and_place` needs them, and the options are
/// its options.
///
/// Each episode chains planned segments from rest to rest, every one batched over the tasks:
/// - the giver approaches above its end of the object and descends to it (the taker waits), closes
///   and lifts it back up;
/// - both arms move at once: the giver carries the object to `meet` while the taker comes to above
///   its end, so they arrive together;
/// - the taker descends onto its end (the giver waits) and closes; the giver opens and lifts off
///   (the taker waits);
/// - both arms move at once again: the giver returns to the default configuration while the taker
///   carries the object to above `place`;
/// - the taker lowers it (the giver waits), opens and retreats.
///
/// A waiting arm's joints are held (`PlanProblem::hold`). Plans hold the object in the hand that
/// has it (`Robot::attach`), touching only the links fixed to either wrist, except the lowering,
/// which plans without it as `pick_and_place`'s does: the object only hangs below the fingers. The
/// demonstration records both grippers and where the object is at every sample, and a task such as
/// "Pass the box from the left hand to the right hand and put it down."
pub fn handover(
    device: &Device,
    worlds: &[World],
    tasks: &[Handover],
    o: &PickPlaceOptions,
) -> Result<Vec<Demonstration>> {
    let robot = device.robot();
    let (n, hands) = (robot.dof(), robot.hands());
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
        "handovers need positive dt, approach and max_width and non-negative gripper_time and grasp_depth"
    );
    let arms: Vec<Arm> = (0..hands).map(|h| Arm::of(robot, h)).collect();
    let upright = |r: Quat| (r * Vec3::Z).dot(Vec3::Z) > 1.0 - 1e-4;
    // Per task: the giver's grasp on the object where it lies, and the taker's at `meet`.
    let mut grasps = vec![];
    for (i, t) in tasks.iter().enumerate() {
        ensure_input!(
            t.from < hands && t.to < hands && t.from != t.to,
            "task {i}: hands {} and {} of {hands}: a handover needs two different hands",
            t.from,
            t.to
        );
        let world = worlds
            .get(t.world as usize)
            .ok_or_else(|| input!("task {i}: world {} of {}", t.world, worlds.len()))?;
        let Some(&Obstacle::Cuboid {
            center,
            half_extents: half,
            rotation,
        }) = world.obstacles.get(t.object)
        else {
            return Err(input!(
                "task {i}: obstacle {} of world {} is not a cuboid",
                t.object,
                t.world
            ));
        };
        ensure_input!(
            upright(rotation) && upright(t.meet.rotation) && upright(t.place.rotation),
            "task {i}: the object, the meeting pose and the place pose must be upright"
        );
        ensure_input!(
            2.0 * half.x.min(half.y) <= o.max_width,
            "task {i}: the object is wider than the grippers open"
        );
        // The end toward each arm at `meet`: the giver's in the object's frame.
        let long = if half.x <= half.y { Vec3::Y } else { Vec3::X };
        let toward = (t.meet.rotation * long)
            .dot(arms[t.from].base - t.meet.position)
            .signum();
        let along = toward * HANDOVER_GRIP * half.x.max(half.y);
        let object = Pose {
            position: center,
            rotation,
        };
        let giver = Grasp::at(object, half, o.grasp_depth, along, &arms[t.from]);
        let taker = Grasp::at(t.meet, half, o.grasp_depth, -along, &arms[t.to]);
        grasps.push((giver, taker));
    }

    // Each task's world as given (index i) and without its object (index tasks + i).
    let without = |t: &Handover| -> Result<World> {
        let mut w = worlds[t.world as usize].clone();
        w.remove(t.object)?;
        Ok(w)
    };
    let scene: Vec<World> = tasks
        .iter()
        .map(|t| Ok(worlds[t.world as usize].clone()))
        .chain(tasks.iter().map(without))
        .collect::<Result<_>>()?;
    let uploaded = device.upload(&scene)?;
    let clear = |i: usize| (tasks.len() + i) as u32;
    let up = Vec3::Z * o.approach;
    let at = |p: Pose| Pose::new(p.position + up, p.rotation);
    let only = |h: usize, pose: Pose| -> Vec<Option<Pose>> {
        (0..hands).map(|k| (k == h).then_some(pose)).collect()
    };
    // The joints only hand `h` moves, which wait while the other hand works.
    let own = |h: usize| -> Vec<usize> {
        let others: Vec<usize> = (0..hands)
            .filter(|&k| k != h)
            .flat_map(|k| robot.hand_joints(k))
            .collect();
        robot
            .hand_joints(h)
            .into_iter()
            .filter(|j| !others.contains(j))
            .collect()
    };
    let nearest = |ik: &IkResult, k: usize, from: &[f32]| ik.nearest(k, from).map(<[f32]>::to_vec);
    let home = robot.default_q.clone();
    // The giver's configuration with the waiting taker's joints from `taker`.
    let with_taker = |t: &Handover, q: &[f32], taker: &[f32]| -> Vec<f32> {
        let mut out = q.to_vec();
        for &j in &own(t.to) {
            out[j] = taker[j];
        }
        out
    };

    // Configurations, in six IK batches: above the giver's end; its grasp; both hands at the
    // meeting pose (the taker above its end); the taker's grasp there; the giver lifted off it,
    // and the taker above the place pose with the giver back home; the place pose. All but the
    // first and the meeting pose are seeded with the configuration before (`NEAR_SEEDS`).
    let n_tasks = tasks.len();
    let near_options = IkOptions {
        seeds: o.ik.seeds.min(NEAR_SEEDS),
        ..o.ik
    };
    let ik = |problems: Vec<IkProblem>| ik::solve_ik(device, &uploaded, &problems, &o.ik);
    let ik_near =
        |problems: Vec<IkProblem>| ik::solve_ik(device, &uploaded, &problems, &near_options);
    let above = ik((0..n_tasks)
        .map(|i| IkProblem {
            world: i as u32,
            targets: only(tasks[i].from, at(grasps[i].0.tool)),
            seed: None,
        })
        .collect())?;
    let c1: Vec<Option<Vec<f32>>> = (0..n_tasks).map(|i| nearest(&above, i, &home)).collect();
    let meet_tool = |i: usize| grasps[i].0.tool_for(tasks[i].meet);
    let grasp = ik_near(
        (0..n_tasks)
            .map(|i| IkProblem {
                world: clear(i),
                targets: only(tasks[i].from, grasps[i].0.tool),
                seed: c1[i].clone(),
            })
            .collect(),
    )?;
    let c2: Vec<Option<Vec<f32>>> = (0..n_tasks)
        .map(|i| c1[i].as_deref().and_then(|q| nearest(&grasp, i, q)))
        .collect();
    let meeting = ik((0..n_tasks)
        .map(|i| {
            let mut targets = only(tasks[i].from, meet_tool(i));
            targets[tasks[i].to] = Some(at(grasps[i].1.tool));
            IkProblem {
                world: clear(i),
                targets,
                seed: c1[i].clone(),
            }
        })
        .collect())?;
    let c3: Vec<Option<Vec<f32>>> = (0..n_tasks)
        .map(|i| c1[i].as_deref().and_then(|q| nearest(&meeting, i, q)))
        .collect();
    let third = ik_near(
        (0..n_tasks)
            .map(|i| IkProblem {
                world: clear(i),
                targets: only(tasks[i].to, grasps[i].1.tool),
                seed: c3[i].clone(),
            })
            .collect(),
    )?;
    let c4: Vec<Option<Vec<f32>>> = (0..n_tasks)
        .map(|i| c3[i].as_deref().and_then(|q| nearest(&third, i, q)))
        .collect();
    let place_tool = |i: usize| grasps[i].1.tool_for(tasks[i].place);
    // Above the place pose: the taker's arm from the meeting grasp, the giver's from home.
    let homed: Vec<Option<Vec<f32>>> = (0..n_tasks)
        .map(|i| c4[i].as_deref().map(|q| with_taker(&tasks[i], &home, q)))
        .collect();
    let fourth = ik_near(
        (0..n_tasks)
            .map(|i| IkProblem {
                world: clear(i),
                targets: only(tasks[i].from, at(meet_tool(i))),
                seed: c4[i].clone(),
            })
            .chain((0..n_tasks).map(|i| IkProblem {
                world: clear(i),
                targets: only(tasks[i].to, at(place_tool(i))),
                seed: homed[i].clone(),
            }))
            .collect(),
    )?;
    let c5: Vec<Option<Vec<f32>>> = (0..n_tasks)
        .map(|i| c4[i].as_deref().and_then(|q| nearest(&fourth, i, q)))
        .collect();
    let c6: Vec<Option<Vec<f32>>> = (0..n_tasks)
        .map(|i| {
            homed[i]
                .as_deref()
                .and_then(|q| nearest(&fourth, n_tasks + i, q))
        })
        .collect();
    let fifth = ik_near(
        (0..n_tasks)
            .map(|i| IkProblem {
                world: clear(i),
                targets: only(tasks[i].to, place_tool(i)),
                seed: c6[i].clone(),
            })
            .collect(),
    )?;
    let c7: Vec<Option<Vec<f32>>> = (0..n_tasks)
        .map(|i| c6[i].as_deref().and_then(|q| nearest(&fifth, i, q)))
        .collect();
    let configs: Vec<Option<[Vec<f32>; 8]>> = (0..n_tasks)
        .map(|i| {
            let c = [&c1, &c2, &c3, &c4, &c5, &c6, &c7].map(|c| c[i].clone());
            if c.iter().all(Option::is_some) {
                let [c1, c2, c3, c4, c5, c6, c7] = c.map(Option::unwrap);
                Some([home.clone(), c1, c2, c3, c4, c5, c6, c7])
            } else {
                None
            }
        })
        .collect();

    // Robots holding each object size in either hand; both wrists may touch it.
    let touch: Vec<String> = robot
        .links
        .iter()
        .filter(|l| {
            robot
                .ee_links
                .iter()
                .any(|&ee| l.chain == robot.links[ee].chain)
        })
        .map(|l| l.name.clone())
        .collect();
    let mut holders: Vec<(Vec3, usize, Device)> = vec![];
    for (t, (giver, taker)) in tasks.iter().zip(&grasps) {
        for (hand, grasp) in [(t.from, giver), (t.to, taker)] {
            if holders
                .iter()
                .all(|(half, h, _)| (*half, *h) != (grasp.half, hand))
            {
                let held = robot.attach(&AttachedObject {
                    name: "held object".into(),
                    link: robot.links[robot.ee_links[hand]].name.clone(),
                    shapes: vec![Obstacle::Cuboid {
                        center: grasp.object_in_tool.position,
                        half_extents: grasp.half,
                        rotation: grasp.object_in_tool.rotation,
                    }],
                    touch_links: touch.clone(),
                    spheres: Default::default(),
                })?;
                holders.push((grasp.half, hand, device.with_robot(&held)?));
            }
        }
    }

    // The planned segments, each in batches over the tasks still going: (from, to) configuration,
    // who holds the object (None: no one), and which hand waits (None: both move).
    enum Holder {
        Nobody,
        Giver,
        Taker,
    }
    let segments: [(usize, usize, Holder, Option<bool>); 7] = [
        // The giver approaches its end, then descends to it; the taker waits.
        (0, 1, Holder::Nobody, Some(true)),
        (1, 2, Holder::Nobody, Some(true)),
        // To the meeting pose, both at once.
        (1, 3, Holder::Giver, None),
        // The taker descends onto its end; the giver waits.
        (3, 4, Holder::Giver, Some(false)),
        // The giver lifts off; the taker waits.
        (4, 5, Holder::Taker, Some(true)),
        // The giver home, the taker above the place pose, both at once.
        (5, 6, Holder::Taker, None),
        // The taker lowers it, hanging below the fingers as in `pick_and_place`; the giver waits.
        (6, 7, Holder::Nobody, Some(false)),
    ];
    let mut paths: Vec<Vec<Vec<f32>>> = vec![vec![vec![]; n_tasks]; segments.len()];
    let mut live: Vec<usize> = (0..n_tasks).filter(|&i| configs[i].is_some()).collect();
    for (s, (from, to, holder, taker_waits)) in segments.iter().enumerate() {
        let hand_of = |i: usize| match holder {
            Holder::Nobody => None,
            Holder::Giver => Some((grasps[i].0.half, tasks[i].from)),
            Holder::Taker => Some((grasps[i].1.half, tasks[i].to)),
        };
        let mut groups: Vec<(&Device, Vec<usize>)> = vec![];
        for &i in &live {
            let d = match hand_of(i) {
                None => device,
                Some(key) => {
                    &holders
                        .iter()
                        .find(|(half, h, _)| (*half, *h) == key)
                        .expect("a holder per size and hand")
                        .2
                }
            };
            match groups.iter_mut().find(|(g, _)| std::ptr::eq(*g, d)) {
                Some((_, group)) => group.push(i),
                None => groups.push((d, vec![i])),
            }
        }
        let mut next = vec![];
        for (d, group) in groups {
            let problems: Vec<PlanProblem> = group
                .iter()
                .map(|&i| {
                    let c = configs[i]
                        .as_ref()
                        .expect("live tasks have their configurations");
                    let hold = match taker_waits {
                        Some(true) => own(tasks[i].to),
                        Some(false) => own(tasks[i].from),
                        None => vec![],
                    };
                    PlanProblem {
                        world: if s == 0 { i as u32 } else { clear(i) },
                        start: c[*from].clone(),
                        goal: c[*to].clone(),
                        start_motion: None,
                        hold,
                    }
                })
                .collect();
            for solved in trajopt::plan(d, &uploaded, &problems, &o.plan)?.solved() {
                paths[s][group[solved.index]] = solved.solution.to_vec();
                next.push(group[solved.index]);
            }
        }
        next.sort_unstable();
        live = next;
    }

    // Speed scales are drawn in order, then the segments are timed in parallel (see `timed`).
    let mut rng = Rng::new(o.rng_seed);
    let timings: Vec<(&[f32], f32)> = live
        .iter()
        .flat_map(|&i| paths.iter().map(move |segment| segment[i].as_slice()))
        .map(|path| (path, rng.range(o.speed_scale.0, o.speed_scale.1)))
        .collect();
    let mut timed_segments = timed(robot, &timings, o.dt)?.into_iter();
    let hold = ((o.gripper_time / o.dt).round() as usize).max(1);
    let mut demos = vec![];
    for i in live {
        let t = &tasks[i];
        let mut next = || timed_segments.next().expect("seven segments per live task");
        let [approach, descend, to_meet, take, lift_off, carry, lower] = [(); 7].map(|_| next());
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
        episode.append(&approach, &grippers(1.0, 1.0));
        episode.append(&descend, &grippers(1.0, 1.0));
        let grasped = episode.len();
        episode.hold(hold, &grippers(1.0, 1.0), &grippers(0.0, 1.0));
        episode.append(&reversed(&descend), &grippers(0.0, 1.0));
        episode.append(&to_meet, &grippers(0.0, 1.0));
        episode.append(&take, &grippers(0.0, 1.0));
        episode.hold(hold, &grippers(0.0, 1.0), &grippers(0.0, 0.0));
        episode.hold(hold, &grippers(0.0, 0.0), &grippers(1.0, 0.0));
        let passed = episode.len();
        episode.append(&lift_off, &grippers(1.0, 0.0));
        episode.append(&carry, &grippers(1.0, 0.0));
        episode.append(&lower, &grippers(1.0, 0.0));
        let released = episode.len();
        episode.hold(hold, &grippers(1.0, 0.0), &grippers(1.0, 1.0));
        episode.append(&reversed(&lower), &grippers(1.0, 1.0));

        let (giver, taker) = &grasps[i];
        let q = |f: usize| &episode.trajectory.positions[f * n..(f + 1) * n];
        let held = |f: usize| {
            if f < passed {
                robot.ee_pose(q(f), t.from).mul_pose(giver.object_in_tool)
            } else {
                robot.ee_pose(q(f), t.to).mul_pose(taker.object_in_tool)
            }
        };
        let set_down = held(released - 1);
        let poses = (0..episode.len())
            .map(|f| match f {
                f if f < grasped + hold => giver.object,
                f if f < released => held(f),
                _ => set_down,
            })
            .collect();
        demos.push(Demonstration {
            origin: Origin::Nominal,
            world: t.world,
            task: describe_handover(robot, t.from, t.to),
            goal: t.place,
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

/// A top-down grasp of an upright cuboid.
struct Grasp {
    object: Pose,
    half: Vec3,
    /// The IK frame at the grasp.
    tool: Pose,
    /// The object's pose in the IK frame while held.
    object_in_tool: Pose,
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

impl Grasp {
    fn new(object: Pose, half: Vec3, depth: f32, arm: &Arm) -> Self {
        Self::at(object, half, depth, 0.0, arm)
    }

    /// A top-down grasp `along` metres from the centre along the longer horizontal side, closing
    /// across the narrower one, the hand facing as `arm`'s does.
    fn at(object: Pose, half: Vec3, depth: f32, along: f32, arm: &Arm) -> Self {
        let z = -Vec3::Z;
        let (narrow, long) = if half.x <= half.y {
            (Vec3::X, Vec3::Y)
        } else {
            (Vec3::Y, Vec3::X)
        };
        let mut y = object.rotation * narrow;
        if (y.cross(z).dot((object.position - arm.base).with_z(0.0)) >= 0.0) != arm.outward {
            y = -y;
        }
        let rotation = Quat::from_mat3(&Mat3::from_cols(y.cross(z), y, z));
        let position = object.position
            + object.rotation * (long * along)
            + Vec3::Z * (half.z - depth.min(half.z));
        let tool = Pose { position, rotation };
        Self {
            object,
            half,
            tool,
            object_in_tool: tool.inverse().mul_pose(object),
        }
    }

    /// The IK frame that holds the object at `place`.
    fn tool_for(&self, place: Pose) -> Pose {
        place.mul_pose(self.object_in_tool.inverse())
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

/// `t` played backwards.
fn reversed(t: &JointTrajectory) -> JointTrajectory {
    let n = t.dof;
    let rows = |v: &[f32], sign: f32| {
        v.chunks(n)
            .rev()
            .flat_map(|r| r.iter().map(move |x| sign * x))
            .collect()
    };
    JointTrajectory {
        positions: rows(&t.positions, 1.0),
        velocities: rows(&t.velocities, -1.0),
        accelerations: rows(&t.accelerations, 1.0),
        ..t.clone()
    }
}
