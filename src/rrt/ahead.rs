//! Threads with nothing left to do checking a long RRT-Connect search's next edges before it asks.
//!
//! A search is sequential: each sample extends trees that every earlier sample grew. Yet most
//! samples extend from the node they would have a few dozen samples earlier (98% of them, 32
//! samples ahead, in the Panda benchmark's longest searches), so once a batch has no searches left
//! to start, its idle threads help those that have drawn `JOIN_AFTER` samples, and a search alone
//! summons helpers into the pool once it has drawn `ALONE_AFTER`. Helpers forecast the extension
//! each coming sample makes, and the connection that follows it, on copies of the search's trees,
//! and check them. Which tree a sample extends
//! depends on the trees' sizes when they are near the ratio that decides it, so a forecast then
//! covers both. The search takes a forecast's answer only for the very query it would have asked
//! (the same edges, bit for bit, keeping the same clearance), so it finds what it would alone.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::{Halton, Tree, connect_edges, place, reach, steps_to, swaps, tube_radius};
use crate::edges::{self, EdgeQuery, Keep, Search, Wanted};
use crate::robot::Robot;

/// Samples past the search's that helpers forecast, at most: further ahead, the trees they copy
/// are staler, and fewer forecasts match.
const LOOKAHEAD: usize = 64;
/// Forecasts kept, one per sample: room for every sample helpers may be forecasting.
const SLOTS: usize = 2 * LOOKAHEAD;
/// Helpers per search, at most.
pub(crate) const HELPERS: usize = 8;
/// Samples a search of a batch draws before helpers join it. Most end sooner, and helpers that
/// would only wait beside them, spinning, slow them down: from 250 samples, short batches planned
/// 5% slower on an M4 Pro. From 1,000, the Ryzen's 32 threads ran out of searches to start while
/// the batch's unsolvable search was short of it, and left it unhelped.
const JOIN_AFTER: usize = 500;
/// Samples a search alone draws before it summons helpers: they start on sleeping threads, and the
/// call waits for them to end, which costs a search that ends soon after.
const ALONE_AFTER: usize = 4000;

/// What a search shares with the threads helping it, and their forecasts.
#[derive(Default)]
pub(crate) struct Ahead {
    /// Whether the search has drawn `JOIN_AFTER` samples (1) or `ALONE_AFTER` (2), and whether it has
    /// summoned helpers.
    reached: AtomicUsize,
    summoned: AtomicBool,
    /// Set when the first helper joins: until then the search shares nothing.
    joined: AtomicBool,
    helpers: AtomicUsize,
    /// Set once the search is done sampling.
    over: AtomicBool,
    /// The sample the search is at, shifted left one, and whether the start's tree extends then.
    at: AtomicUsize,
    /// The next sample to forecast.
    next: AtomicUsize,
    /// The number of the base forecasts are made from (none yet at 0), and the nodes logged since.
    epoch: AtomicUsize,
    logged: AtomicUsize,
    shared: Mutex<Shared>,
    slots: OnceLock<Box<[Mutex<Slot>]>>,
}

#[derive(Default)]
struct Shared {
    base: Option<Arc<Base>>,
    /// Nodes the trees gained since the base: into the start's tree (or the goals'), and the
    /// parent, with their coordinates in `coords`.
    added: Vec<(bool, Option<usize>)>,
    coords: Vec<f32>,
}

/// The search as helpers start from it: its trees, the start's then the goals', and what places
/// and checks its samples. `halton` is where it was before the first sample.
pub(super) struct Base {
    pub(super) trees: [Tree; 2],
    pub(super) halton: Halton,
    pub(super) world: u32,
    pub(super) range: f32,
    pub(super) resolution: f32,
    pub(super) max_iterations: usize,
    pub(super) keep: Option<Arc<Keep>>,
    pub(super) start: Vec<f32>,
    pub(super) line_goal: Vec<f32>,
    pub(super) across: Vec<f32>,
    pub(super) held: Vec<bool>,
}

/// One tree's forecast for a sample: its extension's edges and answer, then, if the extension is
/// valid, its connection's.
#[derive(Clone, Default)]
struct Forecast {
    made: bool,
    extend: Vec<f32>,
    extended: usize,
    connect: Vec<f32>,
    connected: usize,
}

/// A sample's forecasts, extending the start's tree and the goals'.
struct Slot {
    sample: usize,
    epoch: usize,
    forecasts: [Forecast; 2],
}

impl Ahead {
    /// The search draws sample `sample`.
    pub(super) fn sample(&self, sample: usize) {
        if sample == JOIN_AFTER {
            self.reached.store(1, Ordering::Relaxed);
        } else if sample == ALONE_AFTER {
            self.reached.store(2, Ordering::Relaxed);
        }
    }

    /// Whether a search alone summons helpers now: it has drawn `ALONE_AFTER` samples, and has not
    /// summoned them yet.
    pub(crate) fn summon(&self) -> bool {
        self.reached.load(Ordering::Relaxed) >= 2 && !self.summoned.swap(true, Ordering::Relaxed)
    }

    /// Whether a helper has joined, so the search should share.
    pub(super) fn joined(&self) -> bool {
        self.joined.load(Ordering::Relaxed)
    }

    /// Starts helpers over from `base`.
    pub(super) fn publish(&self, base: Base) {
        let mut shared = self.shared.lock().expect("a helper panicked");
        shared.base = Some(Arc::new(base));
        shared.added.clear();
        shared.coords.clear();
        self.logged.store(0, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.over.store(false, Ordering::Release);
    }

    /// The search is at `sample`, extending the start's tree if `start_first`.
    pub(super) fn at(&self, sample: usize, start_first: bool) {
        self.at.store(sample << 1 | usize::from(start_first), Ordering::Release);
    }

    /// Node `i` of `tree`, the start's if `start_tree`, was added.
    pub(super) fn add(&self, start_tree: bool, tree: &Tree, i: usize) {
        let mut shared = self.shared.lock().expect("a helper panicked");
        shared.added.push((start_tree, tree.parent[i]));
        tree.push_node(i, &mut shared.coords);
        self.logged.store(shared.added.len(), Ordering::Release);
    }

    /// The forecast answer to `sample`'s extension of the start's tree (if `start_first`) or the
    /// goals', or (`connect`) its connection, if a helper asked exactly `edges`.
    pub(super) fn answer(&self, sample: usize, start_first: bool, edges: &[f32], connect: bool) -> Option<usize> {
        let slot = self.slots.get()?[sample % SLOTS].lock().expect("a helper panicked");
        let forecast = &slot.forecasts[usize::from(!start_first)];
        if slot.sample != sample || slot.epoch != self.epoch.load(Ordering::Acquire) || !forecast.made {
            return None;
        }
        let (asked, answer) =
            if connect { (&forecast.connect, forecast.connected) } else { (&forecast.extend, forecast.extended) };
        let same = asked.len() == edges.len() && asked.iter().zip(edges).all(|(a, b)| a.to_bits() == b.to_bits());
        same.then_some(answer)
    }

    /// The search is done sampling: helpers leave.
    pub(crate) fn finish(&self) {
        self.over.store(true, Ordering::Release);
    }

    /// A helper for the search, unless it is done sampling, has drawn fewer than `JOIN_AFTER`
    /// samples, or has helpers enough. With `yields`, it runs the pool's other work every so often
    /// and while it waits, such as a repair's samples: a pool thread that helps one search of a
    /// batch should not hold up another.
    pub(crate) fn helper<'a>(self: &Arc<Self>, robot: &'a Robot, yields: bool) -> Option<Helper<'a>> {
        if self.over.load(Ordering::Acquire) || self.reached.load(Ordering::Relaxed) == 0 {
            return None;
        }
        if self.helpers.fetch_add(1, Ordering::AcqRel) >= HELPERS {
            self.helpers.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        self.slots.get_or_init(|| {
            (0..SLOTS)
                .map(|_| Mutex::new(Slot { sample: usize::MAX, epoch: 0, forecasts: Default::default() }))
                .collect()
        });
        self.joined.store(true, Ordering::Relaxed);
        let n = robot.dof();
        Some(Helper {
            robot,
            ahead: self.clone(),
            epoch: 0,
            base: None,
            trees: [Tree::new(&robot.continuous), Tree::new(&robot.continuous)],
            replayed: 0,
            added: vec![],
            coords: vec![],
            sample: 0,
            todo: [None; 2],
            from_start: true,
            asked: Asked::Nothing,
            yields,
            rounds: 0,
            forecasts: Default::default(),
            new: Vec::with_capacity(n),
            target: vec![0.0; n],
            unit: vec![],
            steps: vec![],
        })
    }
}

/// What a helper asked last.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Asked {
    Nothing,
    Extension,
    /// The extension was valid; its connection is next.
    Valid,
    Connection,
}

/// A thread forecasting a search's samples, run as a search of its own.
pub(crate) struct Helper<'a> {
    robot: &'a Robot,
    ahead: Arc<Ahead>,
    /// The base it copied, and its trees, the start's then the goals', with the nodes added since
    /// replayed.
    epoch: usize,
    base: Option<Arc<Base>>,
    trees: [Tree; 2],
    replayed: usize,
    added: Vec<(bool, Option<usize>)>,
    coords: Vec<f32>,
    /// The sample under forecast, the trees still to extend for it (the start's if true), the one
    /// extending, and what it asked.
    sample: usize,
    todo: [Option<bool>; 2],
    from_start: bool,
    asked: Asked,
    /// Whether it runs the pool's other work (see `Ahead::helper`), and its rounds of looking for a
    /// sample to forecast.
    yields: bool,
    rounds: usize,
    forecasts: [Forecast; 2],
    new: Vec<f32>,
    target: Vec<f32>,
    unit: Vec<f32>,
    steps: Vec<f32>,
}

impl Drop for Helper<'_> {
    fn drop(&mut self) {
        self.ahead.helpers.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Helper<'_> {
    /// Brings the trees up to the search's, as far as it has logged; false while it shares none.
    fn sync(&mut self) -> bool {
        let epoch = self.ahead.epoch.load(Ordering::Acquire);
        if epoch == 0 {
            return false;
        }
        if epoch == self.epoch && self.ahead.logged.load(Ordering::Acquire) == self.replayed {
            return true;
        }
        let n = self.robot.dof();
        let shared = self.ahead.shared.lock().expect("a search panicked");
        let epoch = self.ahead.epoch.load(Ordering::Acquire);
        let Some(base) = &shared.base else { return false };
        if epoch != self.epoch {
            let base = base.clone();
            drop(shared);
            self.trees = base.trees.clone();
            self.base = Some(base);
            (self.epoch, self.replayed) = (epoch, 0);
            return self.sync();
        }
        self.added.clear();
        self.added.extend_from_slice(&shared.added[self.replayed..]);
        self.coords.clear();
        self.coords.extend_from_slice(&shared.coords[self.replayed * n..]);
        drop(shared);
        for (&(start_tree, parent), q) in self.added.iter().zip(self.coords.chunks(n)) {
            self.trees[usize::from(!start_tree)].add(q, parent);
        }
        self.replayed += self.added.len();
        true
    }

    /// Claims the next sample to forecast and the trees to extend for it, if the search is not too
    /// far behind: the one taking its turn then, as the search's would at the copies' sizes, and
    /// the other too if the search's own sizes could change whose turn it is.
    fn claim(&mut self) -> bool {
        let at = self.ahead.at.load(Ordering::Acquire);
        let (current, mut from_start) = (at >> 1, at & 1 == 1);
        let mut next = self.ahead.next.load(Ordering::Relaxed);
        let sample = loop {
            let sample = next.max(current + 1);
            if sample > current + LOOKAHEAD {
                return false;
            }
            match self.ahead.next.compare_exchange_weak(next, sample + 1, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break sample,
                Err(now) => next = now,
            }
        };
        let (start, goal) = (self.trees[0].len() as f32, self.trees[1].len() as f32);
        for _ in current + 1..=sample {
            let (a, b) = if from_start { (start, goal) } else { (goal, start) };
            if swaps(a, b) {
                from_start = !from_start;
            }
        }
        // The trees grow by about a node a sample: near twice the other's size, either may extend.
        let (small, large) = (start.min(goal), start.max(goal));
        let margin = (sample - current) as f32 + 1.0;
        let sure = large >= 2.0 * (small + margin) || large + margin < 2.0 * small;
        self.sample = sample;
        self.todo = [Some(from_start), (!sure).then_some(!from_start)];
        self.forecasts.iter_mut().for_each(|f| f.made = false);
        true
    }

    /// The extension query of the next tree to forecast this sample from, if any is left.
    fn extend(&mut self, buffer: &mut Vec<f32>) -> Option<EdgeQuery> {
        let from_start = self.todo.iter_mut().find_map(Option::take)?;
        self.from_start = from_start;
        let (robot, base) = (self.robot, self.base.as_ref().expect("synced"));
        let target = &mut self.target;
        base.halton.point(base.halton.index + self.sample as u64 + 1, target);
        // The tube's samples come first: while it lasts, as many were drawn before as this one's index.
        let tube = tube_radius(robot, &base.across, base.max_iterations, self.sample);
        place(robot, &base.start, &base.line_goal, &base.across, &base.held, tube, target, &mut self.unit);
        let tree = &mut self.trees[usize::from(!from_start)];
        let (near, d) = tree.nearest(target, &robot.continuous);
        reach(robot, base.range, tree, near, d, target, &mut self.new);
        let forecast = &mut self.forecasts[usize::from(!from_start)];
        forecast.extend.clear();
        tree.push_node(near, &mut forecast.extend);
        forecast.extend.extend_from_slice(&self.new);
        self.asked = Asked::Extension;
        let mut edges = edges::reuse(buffer);
        edges.extend_from_slice(&forecast.extend);
        Some(query(base, edges))
    }

    /// Hands the sample's forecasts to the search.
    fn record(&self) {
        let slots = self.ahead.slots.get().expect("slots before helpers");
        let mut slot = slots[self.sample % SLOTS].lock().expect("a search panicked");
        slot.sample = self.sample;
        slot.epoch = self.epoch;
        slot.forecasts.clone_from(&self.forecasts);
    }
}

impl Search for Helper<'_> {
    fn request(&mut self, buffer: &mut Vec<f32>) -> Option<EdgeQuery> {
        if self.asked == Asked::Valid {
            self.asked = Asked::Connection;
            let mut edges = edges::reuse(buffer);
            edges.extend_from_slice(&self.forecasts[usize::from(!self.from_start)].connect);
            return Some(query(self.base.as_ref().expect("synced"), edges));
        }
        if let Some(query) = self.extend(buffer) {
            return Some(query);
        }
        if self.asked != Asked::Nothing {
            self.record();
            self.asked = Asked::Nothing;
        }
        let mut waited = 0u32;
        loop {
            if self.ahead.over.load(Ordering::Acquire) {
                return None;
            }
            // Every so often, other work in the pool first.
            self.rounds += 1;
            if self.yields && self.rounds.is_multiple_of(16) {
                while rayon::yield_now() == Some(rayon::Yield::Executed) {}
            }
            if self.sync() && self.claim() {
                return self.extend(buffer);
            }
            // The search is behind, or shares nothing yet: other work in the pool first, such as
            // the samples of a search's repair.
            if self.yields && rayon::yield_now() == Some(rayon::Yield::Executed) {
                continue;
            }
            waited += 1;
            if waited < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    fn answer(&mut self, answer: usize) {
        let forecast = &mut self.forecasts[usize::from(!self.from_start)];
        match self.asked {
            Asked::Extension if answer == 1 => {
                let robot = self.robot;
                let base = self.base.as_ref().expect("synced");
                let other = &mut self.trees[usize::from(self.from_start)];
                let (node, d) = other.nearest(&self.new, &robot.continuous);
                steps_to(robot, base.range, &self.new, other, node, d, &mut self.target, &mut self.steps);
                forecast.connect.clear();
                forecast.connect.extend_from_slice(&self.new);
                connect_edges(&self.steps, robot.dof(), &mut forecast.connect);
                forecast.extended = answer;
                self.asked = Asked::Valid;
            }
            Asked::Extension => {
                forecast.connect.clear();
                (forecast.extended, forecast.made) = (answer, true);
            }
            Asked::Connection => (forecast.connected, forecast.made) = (answer, true),
            Asked::Nothing | Asked::Valid => unreachable!("answers follow requests"),
        }
    }
}

/// A query of `edges` as the search's own: whether they are free from the first, at its
/// resolution, keeping its clearance.
fn query(base: &Base, edges: Vec<f32>) -> EdgeQuery {
    EdgeQuery {
        world: base.world,
        edges,
        wanted: Some(Wanted::Prefix),
        spacing: base.resolution,
        keep: base.keep.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu;
    use crate::edges::EdgeScratch;
    use crate::lanes::Lanes;
    use crate::robot::{CollisionModel, RobotOptions};
    use crate::rrt::{Connect, RrtOptions, RrtProblem, Sampling};
    use crate::world::{Obstacle, World};
    use glam::{Quat, Vec3};
    use std::collections::HashMap;

    fn panda() -> Robot {
        let fingers = [("panda_finger_joint1".to_string(), 0.04), ("panda_finger_joint2".to_string(), 0.04)];
        let options = RobotOptions {
            lock_joints: HashMap::from(fingers),
            collision_model: Some(CollisionModel::load("assets/franka/panda_collision.json").unwrap()),
            ..Default::default()
        };
        Robot::load("assets/franka/franka_panda.urdf", &options).unwrap()
    }

    /// MotionBenchMaker's `dresser_neutral_goal` problem 53, as robometrics packages it (rounded):
    /// the goal deep in a drawer, which RRT-Connect does not reach, both trees growing until the
    /// samples run out.
    fn dresser() -> (World, RrtProblem) {
        // Box sizes, centres and turns about z (the quaternion's w and z).
        let boxes: [([f32; 3], [f32; 3], [f32; 2]); 35] = [
            ([1.1726693, 0.3247525, 0.01], [0.6136891, 0.0490191, 0.005], [0.8007928, 0.5989415]),
            ([1.1726693, 0.3247525, 0.01], [0.6136891, 0.0490191, 0.7995072], [0.8007928, 0.5989415]),
            ([0.5813346, 0.004, 0.7845072], [0.8199162, 0.2965029, 0.4022536], [0.8007928, 0.5989415]),
            ([0.5813346, 0.3247525, 0.01], [0.5301518, -0.2346015, 0.4022536], [0.8007928, 0.5989415]),
            ([0.5813346, 0.019, 0.3872536], [0.3652785, -0.1860399, 0.6008804], [0.8007928, 0.5989415]),
            ([0.5733346, 0.2882773, 0.004], [0.5126573, -0.2294487, 0.4092536], [0.8007928, 0.5989415]),
            ([0.004, 0.2882773, 0.3872536], [0.5942168, 0.0474572, 0.6008804], [0.8007928, 0.5989415]),
            ([0.004, 0.2882773, 0.3872536], [0.4310978, -0.5063546, 0.6008804], [0.8007928, 0.5989415]),
            ([0.5813346, 0.004, 0.3872536], [0.6528416, -0.2707384, 0.6008804], [0.8007928, 0.5989415]),
            ([0.01, 0.3247525, 0.3872536], [0.5301518, -0.2346015, 0.2036268], [0.8007928, 0.5989415]),
            ([0.2856673, 0.3247525, 0.01], [0.5719205, -0.0927912, 0.2036268], [0.8007928, 0.5989415]),
            ([0.2856673, 0.019, 0.1886268], [0.1547153, 0.0300919, 0.3029402], [0.8007928, 0.5989415]),
            ([0.01, 0.3247525, 0.8045072], [0.7807637, 0.6162605, 0.4022536], [0.8007928, 0.5989415]),
            ([0.2776673, 0.2882773, 0.004], [0.3020941, -0.0133169, 0.2106268], [0.8007928, 0.5989415]),
            ([0.004, 0.2882773, 0.1886268], [0.3418849, 0.1217787, 0.3029402], [0.8007928, 0.5989415]),
            ([0.004, 0.2882773, 0.1886268], [0.2623032, -0.1484124, 0.3029402], [0.8007928, 0.5989415]),
            ([0.2856673, 0.004, 0.1886268], [0.4422784, -0.0546066, 0.3029402], [0.8007928, 0.5989415]),
            ([0.2856673, 0.019, 0.1886268], [0.4070471, -0.0442296, 0.1043134], [0.8007928, 0.5989415]),
            ([0.2776673, 0.2882773, 0.004], [0.5544259, -0.0876384, 0.012], [0.8007928, 0.5989415]),
            ([0.004, 0.2882773, 0.1886268], [0.5942168, 0.0474572, 0.1043134], [0.8007928, 0.5989415]),
            ([0.004, 0.2882773, 0.1886268], [0.514635, -0.2227339, 0.1043134], [0.8007928, 0.5989415]),
            ([0.2856673, 0.004, 0.1886268], [0.6946103, -0.1289281, 0.1043134], [0.8007928, 0.5989415]),
            ([0.2856673, 0.019, 0.3872536], [0.071178, -0.2535288, 0.2036268], [0.8007928, 0.5989415]),
            ([0.01, 0.3247525, 0.8045072], [0.4466146, -0.5182222, 0.4022536], [0.8007928, 0.5989415]),
            ([0.2776673, 0.2882773, 0.004], [0.2185568, -0.2969375, 0.012], [0.8007928, 0.5989415]),
            ([0.004, 0.2882773, 0.3872536], [0.2583477, -0.161842, 0.2036268], [0.8007928, 0.5989415]),
            ([0.004, 0.2882773, 0.3872536], [0.1787659, -0.4320331, 0.2036268], [0.8007928, 0.5989415]),
            ([0.2856673, 0.004, 0.3872536], [0.3587411, -0.3382273, 0.2036268], [0.8007928, 0.5989415]),
            ([1.1926693, 0.01, 0.8045072], [0.7646532, 0.0045544, 0.4022536], [0.8007928, 0.5989415]),
            ([0.01, 0.3247525, 0.7845072], [0.6136891, 0.0490191, 0.4022536], [0.8007928, 0.5989415]),
            ([0.5813346, 0.019, 0.7845072], [0.5323531, 0.3812014, 0.4022536], [0.8007928, 0.5989415]),
            ([0.5733346, 0.2882773, 0.004], [0.6797318, 0.3377927, 0.012], [0.8007928, 0.5989415]),
            ([0.004, 0.2882773, 0.7845072], [0.7612914, 0.6146985, 0.4022536], [0.8007928, 0.5989415]),
            ([0.004, 0.2882773, 0.7845072], [0.5981723, 0.0608868, 0.4022536], [0.8007928, 0.5989415]),
            ([0.3, 0.25, 0.8], [-0.05, 0.0, -0.4], [1.0, 0.0]),
        ];
        let obstacles = boxes
            .iter()
            .map(|&(dims, center, [w, z])| Obstacle::Cuboid {
                center: Vec3::from(center),
                half_extents: Vec3::from(dims) * 0.5,
                rotation: Quat::from_xyzw(0.0, 0.0, z, w).normalize(),
            })
            .collect();
        let problem = RrtProblem {
            world: 0,
            start: vec![2.6544309, 0.9331489, 2.5985923, -2.8740572, -0.468277, 1.9738969, -2.3363396],
            goals: vec![vec![0.1406762, -0.7326517, 0.1564918, -2.1809819, 0.1102163, 1.5412756, 0.9102062]],
            hold: vec![],
        };
        (World { obstacles }, problem)
    }

    /// The search as `Planning` runs it, keeping a clearance and then without it, alone or with
    /// `helpers` threads helping: its trees (the start's, then the goals') and path, bit for bit,
    /// with the samples it drew and the queries helpers answered.
    fn search(robot: &Robot, world: &World, problem: &RrtProblem, helpers: usize) -> (Vec<u32>, usize, usize) {
        let (lanes, worlds) = (Lanes::new(robot), cpu::prepare(std::slice::from_ref(world)));
        let keeping = RrtOptions { clearance: 0.015, max_iterations: 1000, ..RrtOptions::default() };
        let ahead = Arc::new(Ahead::default());
        let mut connect = Connect::new(robot, problem, &keeping, Sampling::NearLine, ahead.clone());
        std::thread::scope(|s| {
            ahead.reached.store(1, Ordering::Relaxed);
            for mut helper in (0..helpers).map(|_| ahead.helper(robot, false).expect("room for helpers")) {
                let (lanes, worlds) = (&lanes, &worlds);
                s.spawn(move || {
                    edges::run(&mut helper, robot, lanes, worlds, None, false, None, &mut EdgeScratch::new(lanes))
                });
            }
            let mut scratch = EdgeScratch::new(&lanes);
            edges::run(&mut connect, robot, &lanes, &worlds, None, false, None, &mut scratch);
            connect.relax(3000);
            edges::run(&mut connect, robot, &lanes, &worlds, None, false, None, &mut scratch);
            ahead.finish();
        });
        (fingerprint(&connect), connect.tube_samples + connect.iterations, connect.forecast_answers)
    }

    /// A search's trees (the start's, then the goals') and path, bit for bit.
    fn fingerprint(connect: &Connect) -> Vec<u32> {
        let [a, b] = &connect.trees;
        let trees = if connect.start_first { [a, b] } else { [b, a] };
        let mut bits: Vec<u32> = vec![];
        for tree in trees {
            bits.extend(tree.coords.iter().flatten().map(|v| v.to_bits()));
            bits.extend(tree.parent.iter().map(|p| p.map_or(u32::MAX, |p| p as u32)));
        }
        bits.extend(connect.path.iter().flatten().map(|v| v.to_bits()));
        bits
    }

    /// The tests that keep threads busy run one at a time, so each has the machine's cores.
    static ALONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Helpers change how fast a search runs, never what it finds: a long search grows the same
    /// trees with three helping as alone, though helpers answer many of its queries (half as many
    /// as it draws samples on 32 threads, a twentieth with two cores shared with other tests).
    #[test]
    fn a_helped_search_grows_the_trees_it_would_alone() {
        let _alone = ALONE.lock().unwrap_or_else(|e| e.into_inner());
        let robot = panda();
        let (world, problem) = dresser();
        let (alone, samples, _) = search(&robot, &world, &problem, 0);
        let (helped, _, answered) = search(&robot, &world, &problem, 3);
        assert!(samples > 3000, "the search drew only {samples} samples");
        assert!(alone == helped, "helpers changed the search");
        assert!(answered > samples / 50, "helpers answered {answered} queries over {samples} samples");
    }

    /// The scheduler's helpers change nothing either. A long search grows the trees it grows
    /// unhelped when the scheduler runs it in a batch, whose threads help it once the other
    /// searches end (shorter, so it has drawn `JOIN_AFTER` samples by then), and when it runs alone,
    /// summoning helpers after `ALONE_AFTER` samples; helpers answer queries both times.
    #[test]
    fn the_scheduler_helps_long_searches_in_batches_and_alone() {
        let _alone = ALONE.lock().unwrap_or_else(|e| e.into_inner());
        let robot = panda();
        let (world, problem) = dresser();
        let (lanes, worlds) = (Lanes::new(&robot), cpu::prepare(std::slice::from_ref(&world)));
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let long = RrtOptions { max_iterations: 6000, ..RrtOptions::default() };
        let short = RrtOptions { max_iterations: 1000, ..RrtOptions::default() };
        let connect = |o: &RrtOptions| Connect::new(&robot, &problem, o, Sampling::NearLine, Arc::default());
        let mut alone = connect(&long);
        edges::run(&mut alone, &robot, &lanes, &worlds, None, false, None, &mut EdgeScratch::new(&lanes));
        let unhelped = fingerprint(&alone);
        assert_eq!(alone.forecast_answers, 0);
        assert!(alone.tube_samples + alone.iterations > ALONE_AFTER + 1000, "the search ends too soon");

        let mut batch = [connect(&long), connect(&short), connect(&short), connect(&short)];
        let mut searches: Vec<&mut dyn Search> = batch.iter_mut().map(|c| c as &mut dyn Search).collect();
        cpu::run_searches(&pool, &mut searches, &robot, &lanes, &worlds, None);
        assert!(fingerprint(&batch[0]) == unhelped, "helpers in a batch changed the search");
        assert!(batch[0].forecast_answers > 0, "no helper answered in the batch");

        let mut one = connect(&long);
        cpu::run_searches(&pool, &mut [&mut one as &mut dyn Search], &robot, &lanes, &worlds, None);
        assert!(fingerprint(&one) == unhelped, "helpers of a search alone changed it");
        assert!(one.forecast_answers > 0, "no helper answered the search alone");
    }
}
