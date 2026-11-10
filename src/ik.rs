//! Batched collision-aware inverse kinematics: damped least squares from many seeds per target,
//! with the collision gradient projected into the Jacobian null space. A problem gives each hand of
//! the robot a target or none: each step solves every targeted hand's 6 x 6 system where the
//! hands are on separate arms, and their targets as one stacked system where the hands share
//! joints (a mobile base, a torso), and joints no targeted hand moves stay where the seed has
//! them.

use crate::error::{Result, ensure_input, input};

use crate::device::{self, CollisionWeights, Contact, Device, Worlds};
use crate::held::{self, Held};
use crate::rng::Rng;
use crate::robot::Robot;
use crate::types::{Pose, Solved};

/// How [`solve_ik`] searches.
#[derive(Clone, Copy, Debug)]
pub struct IkOptions {
    /// Starting configurations per problem: the problem's seed or the robot's default
    /// configuration, then random ones within the joint limits.
    pub seeds: usize,
    /// Damped-least-squares steps per seed, at most: one that has come to rest takes no more.
    pub iterations: u32,
    /// Of `iterations`, the steps every seed takes before only `kept` of each problem's go on;
    /// 0 (or `iterations` and more) for every seed to take them all. Most seeds that will arrive
    /// have by then, and the rest of the steps go to the few that are closing in.
    pub screening: u32,
    /// Seeds per problem that go on past `screening`: of those not yet at their targets within
    /// the tolerances and clear of collision, the nearest, the collision-free first. The others
    /// are returned as the screening left them.
    pub kept: usize,
    /// Damped-least-squares damping (lambda).
    pub damping: f32,
    /// Meters of position error equivalent to one radian of rotation error.
    pub rotation_weight: f32,
    /// Largest joint change per iteration (radians).
    pub max_step: f32,
    /// Step along the null-space-projected collision gradient.
    pub collision_step: f32,
    /// The collision cost that gradient comes from.
    pub collision: CollisionWeights,
    /// Largest position error of a successful seed (meters).
    pub position_tolerance: f32,
    /// Largest rotation error of a successful seed (radians).
    pub rotation_tolerance: f32,
    /// Seed of the random starting configurations. Each problem draws its own from it and its
    /// targets and seed, so a problem's starting configurations do not depend on the others in its
    /// batch.
    pub rng_seed: u64,
}

impl Default for IkOptions {
    fn default() -> Self {
        Self {
            seeds: 64,
            iterations: 30,
            screening: 8,
            kept: 4,
            damping: 0.02,
            rotation_weight: 0.5,
            max_step: 1.0,
            collision_step: 0.05,
            collision: CollisionWeights {
                world: 100.0,
                self_collision: 100.0,
                margin: 0.01,
                self_margin: 0.005,
            },
            position_tolerance: 1e-3,
            rotation_tolerance: 1e-2,
            rng_seed: 1,
        }
    }
}

/// Target poses for a robot's hands, in one world.
#[derive(Clone, Debug)]
pub struct IkProblem {
    /// The world's index in the batch.
    pub world: u32,
    /// Each hand's target pose (see `RobotOptions::ee_links`), or `None` to hold the joints only
    /// that hand moves where the seed has them, as an arm that waits.
    pub targets: Vec<Option<Pose>>,
    /// Where seed 0 starts instead of the robot's default configuration: the current state, for a
    /// target close to it (see [`IkResult::nearest`]). Every seed starts held joints here.
    pub seed: Option<Vec<f32>>,
    /// Joints that stay where the seed has them, in every seed, even where a targeted hand moves
    /// on them: a mobile base placed beside a table while its arms reach.
    pub hold: Vec<usize>,
    /// An object a hand holds, as a pick leaves it: checked as part of the robot, as attaching it
    /// ([`Held::attachment`]) would check it. `None` holds nothing.
    pub holding: Option<Held>,
}

impl IkProblem {
    /// A one-handed robot's hand at `target` in world `world`, seeded from its default
    /// configuration, holding nothing.
    #[must_use]
    pub fn new(world: u32, target: Pose) -> Self {
        Self {
            world,
            targets: vec![Some(target)],
            seed: None,
            hold: Vec::new(),
            holding: None,
        }
    }
}

/// Per-seed results for `problems`; items are problem-major (`item = problem * seeds + seed`).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct IkResult {
    /// The problems, in order.
    pub problems: Vec<IkProblem>,
    /// Joints per configuration.
    pub dof: usize,
    /// Seeds per problem.
    pub seeds: usize,
    /// Each seed's final configuration, `[problems, seeds, dof]`.
    pub q: Vec<f32>,
    /// Per item: meters from the target, the worst over the targeted hands.
    pub position_error: Vec<f32>,
    /// Per item: radians from the target, the worst over the targeted hands.
    pub rotation_error: Vec<f32>,
    /// Per item: meters from the nearest obstacle (see
    /// [`Evaluation::world_clearance`](crate::Evaluation::world_clearance)).
    pub world_clearance: Vec<f32>,
    /// Per item: meters between the nearest checked pair of the robot's links.
    pub self_clearance: Vec<f32>,
    /// Per item: within both tolerances, and collision-free.
    pub success: Vec<bool>,
}

impl IkResult {
    /// The results for groups of `problems` (`parts`, each with its problems' indices), as one.
    fn gather(
        problems: &[IkProblem],
        dof: usize,
        seeds: usize,
        parts: Vec<(Vec<usize>, Self)>,
    ) -> Self {
        let mut parts = parts;
        if parts.len() == 1 && parts[0].0.len() == problems.len() {
            let (_, whole) = parts.remove(0);
            return whole;
        }
        let items = problems.len() * seeds;
        let mut out = Self {
            problems: problems.to_vec(),
            dof,
            seeds,
            q: vec![0.0; items * dof],
            position_error: vec![0.0; items],
            rotation_error: vec![0.0; items],
            world_clearance: vec![0.0; items],
            self_clearance: vec![0.0; items],
            success: vec![false; items],
        };
        for (mine, part) in parts {
            for (k, &p) in mine.iter().enumerate() {
                let (to, from) = (p * seeds..(p + 1) * seeds, k * seeds..(k + 1) * seeds);
                out.q[to.start * dof..to.end * dof]
                    .copy_from_slice(&part.q[from.start * dof..from.end * dof]);
                out.position_error[to.clone()].copy_from_slice(&part.position_error[from.clone()]);
                out.rotation_error[to.clone()].copy_from_slice(&part.rotation_error[from.clone()]);
                out.world_clearance[to.clone()]
                    .copy_from_slice(&part.world_clearance[from.clone()]);
                out.self_clearance[to.clone()].copy_from_slice(&part.self_clearance[from.clone()]);
                out.success[to].copy_from_slice(&part.success[from]);
            }
        }
        out
    }

    /// Item `item`'s configuration.
    #[must_use]
    pub fn solution(&self, item: usize) -> &[f32] {
        &self.q[item * self.dof..(item + 1) * self.dof]
    }

    /// The successful seed with the most obstacle clearance.
    #[must_use]
    pub fn best(&self, problem: usize) -> Option<&[f32]> {
        (problem * self.seeds..(problem + 1) * self.seeds)
            .filter(|&i| self.success[i])
            .max_by(|&a, &b| self.world_clearance[a].total_cmp(&self.world_clearance[b]))
            .map(|i| self.solution(i))
    }

    /// The successful seed of `problem` nearest `q` (Euclidean in joint space), which keeps a small
    /// move on the same IK branch.
    #[must_use]
    pub fn nearest(&self, problem: usize, q: &[f32]) -> Option<&[f32]> {
        let distance = |i: usize| {
            self.solution(i)
                .iter()
                .zip(q)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        };
        (problem * self.seeds..(problem + 1) * self.seeds)
            .filter(|&i| self.success[i])
            .min_by(|&a, &b| distance(a).total_cmp(&distance(b)))
            .map(|i| self.solution(i))
    }

    /// Every problem with a successful seed, with its best configuration.
    pub fn solved(&self) -> impl Iterator<Item = Solved<'_, IkProblem>> {
        self.problems
            .iter()
            .enumerate()
            .filter_map(|(index, problem)| {
                Some(Solved {
                    index,
                    problem,
                    solution: self.best(index)?,
                })
            })
    }
}

/// Why a problem of an [`IkResult`] has no solution ([`failures`]).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum IkFailure {
    /// No seed reaches the targets within the tolerances, even without the push away from
    /// obstacles: they lie beyond the arm's reach or its joint limits, or need more seeds or
    /// iterations than were given.
    Unreachable,
    /// Seeds reach the targets, but only in collision: what the one with the most clearance
    /// collides with ([`Device::contacts`]).
    Collides {
        /// Its contacts, nearest first.
        contacts: Vec<Contact>,
    },
}

/// Why each problem of `result`, solved by [`solve_ik`] with `o`, has no solution: `None` where it
/// has one. The push away from obstacles can hold every seed off a pose that collides, so problems
/// no seed reached are solved once more without it (`collision_step` 0): a pose reached then
/// collides, one still not reached is out of reach. That solve costs what theirs did, so it runs
/// only when asked for, here.
pub fn failures(
    device: &Device,
    worlds: &Worlds,
    result: &IkResult,
    o: &IkOptions,
) -> Result<Vec<Option<IkFailure>>> {
    ensure_input!(
        result.seeds == o.seeds,
        "a result of {} seeds per problem, from options with {}",
        result.seeds,
        o.seeds
    );
    let problems = 0..result.problems.len();
    let failed: Vec<usize> = problems.filter(|&p| result.best(p).is_none()).collect();
    let unreached: Vec<usize> = failed
        .iter()
        .copied()
        .filter(|&p| reaching(result, p, o).is_none())
        .collect();
    let unpushed = IkOptions {
        collision_step: 0.0,
        ..*o
    };
    let retried = if unreached.is_empty() || o.collision_step == 0.0 {
        None
    } else {
        let again: Vec<IkProblem> = unreached
            .iter()
            .map(|&p| result.problems[p].clone())
            .collect();
        Some(solve_ik(device, worlds, &again, &unpushed)?)
    };
    // Per failed problem, the reaching configuration whose contacts it reports, if any.
    let mut colliding: Vec<(usize, u32, &[f32])> = vec![];
    for &p in &failed {
        let world = result.problems[p].world;
        if let Some(item) = reaching(result, p, o) {
            colliding.push((p, world, result.solution(item)));
        } else if let Some(again) = &retried {
            let k = unreached
                .iter()
                .position(|&u| u == p)
                .expect("problems no seed reached were solved again");
            if let Some(item) = reaching(again, k, &unpushed) {
                colliding.push((p, world, again.solution(item)));
            }
        }
    }
    // What each collides with, holding what its problem holds.
    let holds: Vec<Option<&Held>> = colliding
        .iter()
        .map(|&(p, _, _)| result.problems[p].holding.as_ref())
        .collect();
    let parts = held::by_hold(device, &holds, |on, mine| {
        let item_world: Vec<u32> = mine.iter().map(|&k| colliding[k].1).collect();
        let q: Vec<f32> = mine.iter().flat_map(|&k| colliding[k].2).copied().collect();
        on.contacts(worlds, &item_world, &q, 0.0)
    })?;
    let mut out: Vec<Option<IkFailure>> = vec![None; result.problems.len()];
    for &p in &failed {
        out[p] = Some(IkFailure::Unreachable);
    }
    for (mine, contacts) in parts {
        for (&k, contacts) in mine.iter().zip(contacts) {
            out[colliding[k].0] = Some(IkFailure::Collides { contacts });
        }
    }
    Ok(out)
}

/// The seed of `problem` with the most clearance among those within `o`'s tolerances, collision-free
/// or not.
fn reaching(result: &IkResult, problem: usize, o: &IkOptions) -> Option<usize> {
    let clearance = |k: usize| result.world_clearance[k].min(result.self_clearance[k]);
    (problem * result.seeds..(problem + 1) * result.seeds)
        .filter(|&k| {
            result.position_error[k] < o.position_tolerance
                && result.rotation_error[k] < o.rotation_tolerance
        })
        .max_by(|&a, &b| clearance(a).total_cmp(&clearance(b)))
}

/// Seed 0 starts from the problem's seed or else the robot's default configuration, the rest with
/// the joints the targeted hands move drawn uniformly within joint limits. Problems that hold an
/// object ([`IkProblem::holding`]) run as a batch for each object and grasp, in parallel, on the
/// device's hardware; `device` must not hold one in that hand already.
pub fn solve_ik(
    device: &Device,
    worlds: &Worlds,
    problems: &[IkProblem],
    o: &IkOptions,
) -> Result<IkResult> {
    check(device.robot(), worlds, problems, o)?;
    let holds: Vec<Option<&Held>> = problems.iter().map(|p| p.holding.as_ref()).collect();
    let parts = held::by_hold(device, &holds, |on, items| {
        solved(on, worlds, &held::subset(problems, items), o)
    })?;
    Ok(IkResult::gather(
        problems,
        device.robot().dof(),
        o.seeds,
        parts,
    ))
}

impl IkOptions {
    /// An `Error::Input` naming the first option IK cannot run with; `what` names the options in
    /// it (`IK`, or `plan`'s `goal_ik`).
    pub(crate) fn check(&self, what: &str) -> Result<()> {
        ensure_input!(self.seeds > 0, "{what} needs at least one seed");
        ensure_input!(
            self.kept > 0,
            "{what} needs at least one seed kept past the screening"
        );
        let positive = |v: f32| v.is_finite() && v > 0.0;
        ensure_input!(
            positive(self.max_step)
                && positive(self.rotation_weight)
                && self.damping.is_finite()
                && self.damping >= 0.0,
            "{what} needs a positive max_step and rotation_weight and a non-negative damping"
        );
        ensure_input!(
            self.collision_step.is_finite(),
            "{what} needs a finite collision_step, got {}",
            self.collision_step
        );
        Ok(())
    }
}

/// Whether `problems` and `o` can be solved for `robot` in `worlds`.
fn check(robot: &Robot, worlds: &Worlds, problems: &[IkProblem], o: &IkOptions) -> Result<()> {
    let n = robot.dof();
    o.check("IK")?;
    let hands = robot.hands();
    for (i, p) in problems.iter().enumerate() {
        ensure_input!(
            p.targets.len() == hands && p.targets.iter().any(Option::is_some),
            "IK problem {i}: needs a target, or None, for each of the robot's {hands} hands, and at least one target"
        );
        ensure_input!(
            (p.world as usize) < worlds.len(),
            "IK problem {i} references world {} of {}",
            p.world,
            worlds.len()
        );
        ensure_input!(
            p.targets.iter().flatten().all(Pose::is_unit),
            "IK problem {i}: targets need a finite position and a unit-quaternion rotation"
        );
        if let Some(seed) = &p.seed {
            ensure_input!(
                seed.len() == n && seed.iter().all(|v| v.is_finite()),
                "IK problem {i}: the seed needs {n} finite values"
            );
        }
        if let Some(&j) = p.hold.iter().find(|&&j| j >= n) {
            return Err(input!(
                "IK problem {i}: holds joint {j} of a robot with {n}"
            ));
        }
        if let Some(held) = &p.holding {
            held.check(robot, worlds, &format!("IK problem {i}"))?;
        }
    }
    Ok(())
}

/// Seeds (items) under which a batch is seeded on the calling thread: handing fewer to the
/// device's pool costs more than it saves.
const SEEDED_ALONE: usize = 4096;

/// [`solve_ik`] for `problems`, checked, holding what `device` holds.
fn solved(
    device: &Device,
    worlds: &Worlds,
    problems: &[IkProblem],
    o: &IkOptions,
) -> Result<IkResult> {
    let robot = device.robot();
    let n = robot.dof();
    let hands = robot.hands();
    let items = problems.len() * o.seeds;
    // Each problem's targets once: its seeds share them.
    let targets: Vec<Option<Pose>> = problems.iter().flat_map(|p| p.targets.clone()).collect();
    let item_world: Vec<u32> = problems
        .iter()
        .flat_map(|p| std::iter::repeat_n(p.world, o.seeds))
        .collect();
    // Per item and joint, whether it is held; empty when no problem holds one.
    let holding = problems.iter().any(|p| !p.hold.is_empty());
    let flags = problems
        .iter()
        .flat_map(|p| (0..o.seeds * n).map(|i| p.hold.contains(&(i % n))));
    let held: Vec<bool> = if holding { flags.collect() } else { vec![] };
    let hand_joints: Vec<Vec<usize>> = (0..hands).map(|h| robot.hand_joints(h)).collect();
    // A problem's seeds: its own start, then random configurations of the joints its targets
    // move, drawn from its own generator, so they do not depend on where it is in the batch.
    let seed = |p: &IkProblem, seeds: &mut [f32]| {
        let mut moved = vec![false; n];
        let targeted = p.targets.iter().enumerate().filter(|(_, t)| t.is_some());
        for (h, _) in targeted {
            for &j in &hand_joints[h] {
                moved[j] = true;
            }
        }
        for &j in &p.hold {
            moved[j] = false;
        }
        let start = p.seed.as_deref().unwrap_or(&robot.default_q);
        let aims = p.targets.iter().flat_map(|t| match t {
            Some(Pose {
                position: v,
                rotation: r,
            }) => [1.0, v.x, v.y, v.z, r.x, r.y, r.z, r.w],
            None => [0.0; 8],
        });
        let mut rng = Rng::for_problem(o.rng_seed, aims.chain(start.iter().copied()));
        for (s, q) in seeds.chunks_mut(n).enumerate() {
            q.copy_from_slice(start);
            if s > 0 {
                let drawn = (0..n).filter(|&j| moved[j]);
                for j in drawn {
                    q[j] = rng.range(robot.lower[j], robot.upper[j]);
                }
            }
        }
    };
    let mut q = vec![0.0; items * n];
    let each = q.chunks_mut(o.seeds * n).zip(problems);
    if items < SEEDED_ALONE {
        for (seeds, p) in each {
            seed(p, seeds);
        }
    } else {
        use rayon::prelude::*;
        device.install(|| {
            let each = q.par_chunks_mut(o.seeds * n).zip(problems);
            each.for_each(|(seeds, p)| seed(p, seeds));
        });
    }
    // A robot without collision spheres has nothing to push away from: its push is zero, and
    // leaving the step out leaves out its solve.
    let unpushed = IkOptions {
        collision_step: 0.0,
        ..*o
    };
    let stepped = if robot.spheres.is_empty() {
        &unpushed
    } else {
        o
    };
    let staged = o.screening > 0 && o.screening < o.iterations && o.kept < o.seeds;
    let first = IkOptions {
        iterations: if staged { o.screening } else { o.iterations },
        ..*stepped
    };
    let mut solved = device.ik(worlds, &item_world, &targets, &held, &mut q, &first)?;
    if staged {
        // Of each problem's seeds that have not arrived, the `kept` nearest their targets, the
        // collision-free first, take the remaining steps as a batch of their own. A problem with
        // fewer left takes arrived ones along, which stay at rest, so every problem sends as many.
        let rank = |s: &[f32; 4]| {
            let clear = device::collision_free([s[2], s[3]]);
            let arrived = clear && s[0] < o.position_tolerance && s[1] < o.rotation_tolerance;
            (arrived, !clear, s[0] + o.rotation_weight * s[1])
        };
        let choose = |p: usize, chosen: &mut [usize]| {
            let mut best: Vec<((bool, bool, f32), usize)> = Vec::with_capacity(o.kept);
            let first = p * o.seeds;
            let seeds = solved[first..first + o.seeds].iter().enumerate();
            for (s, seed) in seeds {
                let (ranked, item) = (rank(seed), first + s);
                // Items come in order, so one that ties an earlier one goes after it.
                let at = best.partition_point(|&((arrived, blocked, error), _)| {
                    (arrived, blocked)
                        .cmp(&(ranked.0, ranked.1))
                        .then(error.total_cmp(&ranked.2))
                        .is_le()
                });
                if at < o.kept {
                    best.truncate(o.kept - 1);
                    best.insert(at, (ranked, item));
                }
            }
            for (slot, &(_, item)) in chosen.iter_mut().zip(&best) {
                *slot = item;
            }
        };
        let mut chosen = vec![0; problems.len() * o.kept];
        if items < SEEDED_ALONE {
            for (p, slots) in chosen.chunks_mut(o.kept).enumerate() {
                choose(p, slots);
            }
        } else {
            use rayon::prelude::*;
            device.install(|| {
                let each = chosen.par_chunks_mut(o.kept).enumerate();
                each.for_each(|(p, slots)| choose(p, slots));
            });
        }
        let mut refined: Vec<f32> = Vec::with_capacity(chosen.len() * n);
        let mut refined_held = Vec::with_capacity(if holding { chosen.len() * n } else { 0 });
        for &item in &chosen {
            refined.extend_from_slice(&q[item * n..(item + 1) * n]);
            if holding {
                refined_held.extend_from_slice(&held[item * n..(item + 1) * n]);
            }
        }
        let refined_world: Vec<u32> = chosen.iter().map(|&item| item_world[item]).collect();
        let rest = IkOptions {
            iterations: o.iterations - o.screening,
            ..*stepped
        };
        let again = device.ik(
            worlds,
            &refined_world,
            &targets,
            &refined_held,
            &mut refined,
            &rest,
        )?;
        for ((&item, row), seed) in chosen.iter().zip(again).zip(refined.chunks(n)) {
            q[item * n..(item + 1) * n].copy_from_slice(seed);
            solved[item] = row;
        }
    }
    let mut result = IkResult {
        problems: problems.to_vec(),
        dof: n,
        seeds: o.seeds,
        q,
        position_error: Vec::with_capacity(items),
        rotation_error: Vec::with_capacity(items),
        world_clearance: Vec::with_capacity(items),
        self_clearance: Vec::with_capacity(items),
        success: Vec::with_capacity(items),
    };
    for [position, rotation, world, own] in solved {
        result.position_error.push(position);
        result.rotation_error.push(rotation);
        result.world_clearance.push(world);
        result.self_clearance.push(own);
        result.success.push(
            position < o.position_tolerance
                && rotation < o.rotation_tolerance
                && device::collision_free([world, own]),
        );
    }
    Ok(result)
}
