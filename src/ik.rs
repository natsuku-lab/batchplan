//! Batched collision-aware inverse kinematics: damped least squares from many seeds per target,
//! with the collision gradient projected into the Jacobian null space. A problem gives each hand of
//! the robot a target or none: each step solves every targeted hand's 6 x 6 system, which is exact
//! for hands on separate arms (hands whose chains share joints add their steps there), and joints
//! no targeted hand moves stay where the seed has them.

use crate::error::{Result, ensure_input};

use crate::device::{self, CollisionWeights, Device, Worlds};
use crate::rng::Rng;
use crate::types::{Pose, Solved};

/// How [`solve_ik`] searches.
#[derive(Clone, Copy, Debug)]
pub struct IkOptions {
    /// Starting configurations per problem: the problem's seed or the robot's default
    /// configuration, then random ones within the joint limits.
    pub seeds: usize,
    /// Damped-least-squares steps per seed.
    pub iterations: u32,
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
    /// targets and seed, so a problem's solutions do not depend on the others in its batch.
    pub rng_seed: u64,
}

impl Default for IkOptions {
    fn default() -> Self {
        Self {
            seeds: 32,
            iterations: 60,
            damping: 0.02,
            rotation_weight: 0.5,
            max_step: 0.3,
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
}

impl IkProblem {
    /// A one-handed robot's hand at `target` in world `world`, seeded from its default
    /// configuration.
    #[must_use]
    pub fn new(world: u32, target: Pose) -> Self {
        Self {
            world,
            targets: vec![Some(target)],
            seed: None,
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

/// Seed 0 starts from the problem's seed or else the robot's default configuration, the rest with
/// the joints the targeted hands move drawn uniformly within joint limits.
pub fn solve_ik(
    device: &Device,
    worlds: &Worlds,
    problems: &[IkProblem],
    o: &IkOptions,
) -> Result<IkResult> {
    let robot = device.robot();
    let n = robot.dof();
    ensure_input!(o.seeds > 0, "need at least one seed");
    let positive = |v: f32| v.is_finite() && v > 0.0;
    ensure_input!(
        positive(o.max_step)
            && positive(o.rotation_weight)
            && o.damping.is_finite()
            && o.damping >= 0.0,
        "IK needs a positive max_step and rotation_weight and a non-negative damping"
    );
    let hands = robot.hands();
    for (i, p) in problems.iter().enumerate() {
        ensure_input!(
            p.targets.len() == hands && p.targets.iter().any(Option::is_some),
            "IK problem {i}: needs a target, or None, for each of the robot's {hands} hands, and at least one target"
        );
        for target in p.targets.iter().flatten() {
            let (position, rotation) = (target.position, target.rotation);
            ensure_input!(
                position.is_finite()
                    && rotation.is_finite()
                    && (rotation.length() - 1.0).abs() < 1e-3,
                "IK problem {i}: targets need a finite position and a unit-quaternion rotation"
            );
        }
        if let Some(seed) = &p.seed {
            ensure_input!(
                seed.len() == n && seed.iter().all(|v| v.is_finite()),
                "IK problem {i}: the seed needs {n} finite values"
            );
        }
    }
    let items = problems.len() * o.seeds;
    let mut q = Vec::with_capacity(items * n);
    let mut item_world = Vec::with_capacity(items);
    let mut targets = Vec::with_capacity(items * hands);
    let hand_joints: Vec<Vec<usize>> = (0..hands).map(|h| robot.hand_joints(h)).collect();
    let mut moved = vec![false; n];
    for p in problems {
        moved.fill(false);
        let targeted = p.targets.iter().enumerate().filter(|(_, t)| t.is_some());
        for (h, _) in targeted {
            for &j in &hand_joints[h] {
                moved[j] = true;
            }
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
        for s in 0..o.seeds {
            if s == 0 {
                q.extend_from_slice(start);
            } else {
                q.extend((0..n).map(|j| {
                    if moved[j] {
                        rng.range(robot.lower[j], robot.upper[j])
                    } else {
                        start[j]
                    }
                }));
            }
            item_world.push(p.world);
            targets.extend_from_slice(&p.targets);
        }
    }
    let err = device.ik(worlds, &item_world, &targets, &mut q, o)?;
    let clear = device.clearance(worlds, &item_world, &q)?;
    let success = (0..items)
        .map(|i| {
            err[i][0] < o.position_tolerance
                && err[i][1] < o.rotation_tolerance
                && device::collision_free(clear[i])
        })
        .collect();
    Ok(IkResult {
        problems: problems.to_vec(),
        dof: n,
        seeds: o.seeds,
        q,
        position_error: err.iter().map(|e| e[0]).collect(),
        rotation_error: err.iter().map(|e| e[1]).collect(),
        world_clearance: clear.iter().map(|c| c[0]).collect(),
        self_clearance: clear.iter().map(|c| c[1]).collect(),
        success,
    })
}
