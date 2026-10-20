//! Sampling-based searches (RRT-Connect, simplification, validating a spline) ask one question of
//! the world: whether straight joint-space edges, or given configurations, are collision-free. An
//! edge is checked at `R·n` evenly spaced
//! states, `t = m / (R·n)` for `m = 1..=R·n` (the start excluded, the end included), with
//! `R = LANES` and `n` the fewest passes that keep them `spacing` apart. They are visited as VAMP's
//! rake does, `R` states spread over the edge at a time, each pass stepping back, and the first
//! collision ends the check. Searches are sequential, so each runs to its end on one thread of the
//! CPU, with the robot's SIMD collision checks (`lanes`), on either device; threads with no search
//! left to start may check a long one's next edges before it asks (`rrt::Ahead`). A search can also
//! ask for a path to be optimized, on that thread, before it checks it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use fearless_simd::{Simd, dispatch};

use crate::cpu::{self, Prepared};
use crate::lanes::{LANES, LaneScratch, Lanes};
use crate::robot::Robot;
use crate::rrt::{Ahead, distance};
use crate::trajopt::PlanOptions;

/// What a query asks of its edges, checked in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Wanted {
    /// How many edges, from the first, are valid.
    Prefix,
    /// The index of the first valid edge.
    FirstValid,
}

/// Edges `[k, 2, dof]` (each its start then its end) in one world, checked `spacing` apart, or
/// configurations `[k, dof]` (`wanted` is `None`), whose answer is how many from the first are
/// collision-free, counted in groups of `LANES` (`k` when all are). With `keep`, edges and
/// configurations must also keep a clearance from the world.
#[derive(Clone, Debug)]
pub(crate) struct EdgeQuery {
    pub(crate) world: u32,
    pub(crate) edges: Vec<f32>,
    pub(crate) wanted: Option<Wanted>,
    pub(crate) spacing: f32,
    pub(crate) keep: Option<Arc<Keep>>,
}

/// A clearance from the world that states keep, shrinking to zero toward a path's ends: a state
/// `d` from the nearest end (joint-space distance) keeps `clearance * min(1, d / ramp)`. Paths
/// start and end where they must, often in contact, such as at a grasp.
#[derive(Clone, Debug)]
pub(crate) struct Keep {
    pub(crate) clearance: f32,
    pub(crate) ramp: f32,
    pub(crate) ends: Vec<Vec<f32>>,
}

impl Keep {
    fn margin(&self, q: &[f32]) -> f32 {
        let nearest = self.ends.iter().map(|end| distance(q, end)).fold(f32::INFINITY, f32::min);
        self.clearance * (nearest / self.ramp).min(1.0)
    }

    /// Whether every state of the edge from `a` to `b` is beyond the ramp from every end, so keeps
    /// the full clearance: no end comes within the ramp of the segment, with room for rounding.
    fn beyond_ramp(&self, a: &[f32], b: &[f32]) -> bool {
        let span: f32 = a.iter().zip(b).map(|(x, y)| (y - x) * (y - x)).sum();
        self.ends.iter().all(|end| {
            let along: f32 = a.iter().zip(b).zip(end).map(|((x, y), e)| (e - x) * (y - x)).sum();
            let t = if span > 0.0 { (along / span).clamp(0.0, 1.0) } else { 0.0 };
            let gap: f32 = a.iter().zip(b).zip(end).map(|((x, y), e)| (x + (y - x) * t - e).powi(2)).sum();
            gap.sqrt() >= self.ramp * 1.001
        })
    }
}

impl EdgeQuery {
    pub(crate) fn count(&self, n: usize) -> usize {
        self.edges.len() / (2 * n)
    }

    pub(crate) fn edge(&self, i: usize, n: usize) -> (&[f32], &[f32]) {
        let e = &self.edges[2 * i * n..2 * (i + 1) * n];
        e.split_at(n)
    }
}

/// A search that needs edges checked: RRT-Connect or simplification of one problem.
pub(crate) trait Search: Send {
    /// The edges to check next, written into `buffer`'s allocation (see `reuse`), which the driver
    /// hands back once they are answered; `None` once the search is over, or wants a path optimized.
    fn request(&mut self, buffer: &mut Vec<f32>) -> Option<EdgeQuery>;
    /// The answer to the last request.
    fn answer(&mut self, answer: usize);
    /// A path to optimize in place before the next request, asked when `request` has none.
    fn optimize(&mut self) -> Option<Optimize<'_>> {
        None
    }
    /// What the search shares with threads that would help it along (see `rrt::Ahead`), asked
    /// before it runs.
    fn help(&self) -> Option<Arc<Ahead>> {
        None
    }
}

/// A path's B-spline control points to improve with `options.iterations` steps of trajectory
/// optimization in `world`.
pub(crate) struct Optimize<'a> {
    pub(crate) world: u32,
    pub(crate) path: &'a mut [f32],
    pub(crate) options: PlanOptions,
}

/// `buffer`'s allocation, emptied, for a query's edges: searches fill it instead of allocating
/// one per query, which, across every thread of a batch, the allocator would serialize.
pub(crate) fn reuse(buffer: &mut Vec<f32>) -> Vec<f32> {
    let mut edges = std::mem::take(buffer);
    edges.clear();
    edges
}

/// Working memory for checking edges on one thread.
pub(crate) struct EdgeScratch {
    lanes: LaneScratch,
    states: Vec<f32>,
    /// The edges of the searches this thread runs, one query after another.
    buffer: Vec<f32>,
}

impl EdgeScratch {
    pub(crate) fn new(lanes: &Lanes) -> Self {
        Self { lanes: lanes.scratch(), states: vec![0.0; LANES * lanes.dof()], buffer: vec![] }
    }
}

/// Queries between the driver's calls to `pause`.
const POLL: usize = 64;

/// Runs `search` to its end, or to `deadline`, checking its edges in `worlds` and optimizing the
/// paths it asks to on this thread. With `spread`, an optimization also hands its samples to the
/// current pool's idle threads, counted in `spread` while it runs so that threads helping other
/// searches take its samples first (see `rrt::Ahead::helper`). `pause`, if given, is asked every
/// `POLL` queries whether to return early, between queries; true if it did, and a later call goes
/// on with the search (where a search alone stops to summon help, see `cpu::run_searches`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    search: &mut dyn Search,
    robot: &Robot,
    lanes: &Lanes,
    worlds: &[Vec<Prepared>],
    deadline: Option<Instant>,
    spread: Option<&AtomicUsize>,
    pause: Option<&dyn Fn() -> bool>,
    scratch: &mut EdgeScratch,
) -> bool {
    // One dispatch for the whole search, so the checks inline at the CPU's SIMD level.
    dispatch!(lanes.level(), simd => run_at(simd, search, robot, lanes, worlds, deadline, spread, pause, scratch))
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn run_at<S: Simd>(
    simd: S,
    search: &mut dyn Search,
    robot: &Robot,
    lanes: &Lanes,
    worlds: &[Vec<Prepared>],
    deadline: Option<Instant>,
    spread: Option<&AtomicUsize>,
    pause: Option<&dyn Fn() -> bool>,
    scratch: &mut EdgeScratch,
) -> bool {
    let mut asked = 0usize;
    while deadline.is_none_or(|d| Instant::now() < d) {
        asked += 1;
        if pause.is_some_and(|pause| asked.is_multiple_of(POLL) && pause()) {
            return true;
        }
        let Some(query) = search.request(&mut scratch.buffer) else {
            let Some(o) = search.optimize() else { return false };
            if let Some(optimizing) = spread {
                optimizing.fetch_add(1, Ordering::Relaxed);
            }
            cpu::optimize_path(robot, &worlds[o.world as usize], o.path, &o.options, deadline, spread.is_some());
            if let Some(optimizing) = spread {
                optimizing.fetch_sub(1, Ordering::Relaxed);
            }
            continue;
        };
        let world = &worlds[query.world as usize];
        let n = lanes.dof();
        // Loops rather than closures: a closure is a function of its own, without the dispatched
        // SIMD level, and the checks inside it would not inline.
        let answer = match query.wanted {
            // Edges in order, stopping at the answer: the first invalid one for a prefix, the
            // first valid one otherwise; the count when none stops the scan.
            Some(wanted) => {
                let count = query.count(n);
                let mut answer = count;
                for i in 0..count {
                    if edge_free(simd, lanes, world, query.edge(i, n), &query, scratch)
                        == (wanted == Wanted::FirstValid)
                    {
                        answer = i;
                        break;
                    }
                }
                answer
            }
            None => {
                let mut answer = query.edges.len() / n;
                for (group, q) in query.edges.chunks(LANES * n).enumerate() {
                    let mut margin = [0.0; LANES];
                    if let Some(keep) = &query.keep {
                        for (m, c) in margin.iter_mut().zip(q.chunks(n)) {
                            *m = keep.margin(c);
                        }
                    }
                    if !lanes.keeps(simd, world, q, margin, &mut scratch.lanes) {
                        answer = group * LANES;
                        break;
                    }
                }
                answer
            }
        };
        search.answer(answer);
        scratch.buffer = query.edges;
    }
    false
}

/// Whether the edge from `a` to `b` is collision-free at its states `query.spacing` apart, and
/// keeps the query's clearance.
#[inline(always)]
fn edge_free<S: Simd>(
    simd: S,
    lanes: &Lanes,
    world: &[Prepared],
    (a, b): (&[f32], &[f32]),
    query: &EdgeQuery,
    s: &mut EdgeScratch,
) -> bool {
    let n = lanes.dof();
    let passes = passes(a, b, query.spacing);
    let total = LANES * passes;
    // Away from the path's ends, every state keeps the full clearance.
    let full = query.keep.as_ref().filter(|k| k.beyond_ramp(a, b)).map(|k| k.clearance);
    for p in 0..passes {
        let mut margin = [full.unwrap_or(0.0); LANES];
        for (l, out) in s.states.chunks_mut(n).enumerate() {
            state(a, b, (l + 1) * passes - p, total, out);
            if let (Some(keep), None) = (&query.keep, full) {
                margin[l] = keep.margin(out);
            }
        }
        if !lanes.keeps(simd, world, &s.states, margin, &mut s.lanes) {
            return false;
        }
    }
    true
}

/// The passes an edge from `a` to `b` takes: `LANES` states each.
pub(crate) fn passes(a: &[f32], b: &[f32], spacing: f32) -> usize {
    ((distance(a, b) / (LANES as f32 * spacing)).ceil() as usize).max(1)
}

/// State `m` of `total` along the edge from `a` to `b`, into `out`.
#[inline]
pub(crate) fn state(a: &[f32], b: &[f32], m: usize, total: usize, out: &mut [f32]) {
    let t = m as f32 / total as f32;
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = x + (y - x) * t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, collision_free};
    use crate::rng::Rng;
    use crate::robot::{CollisionModel, Robot, RobotOptions};
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

    fn devices(robot: &Robot) -> Vec<Device> {
        let mut out = vec![Device::cpu(robot).unwrap()];
        match Device::gpu(robot) {
            Ok(gpu) => out.push(gpu),
            Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
            Err(e) => panic!("no GPU: {e}"),
        }
        out
    }

    struct Asker {
        queries: Vec<EdgeQuery>,
        answers: Vec<usize>,
    }

    impl Search for Asker {
        fn request(&mut self, _: &mut Vec<f32>) -> Option<EdgeQuery> {
            self.queries.get(self.answers.len()).cloned()
        }

        fn answer(&mut self, answer: usize) {
            self.answers.push(answer);
        }
    }

    /// Answers on every device match a check of every state, and each other.
    #[test]
    fn every_device_answers_edge_queries_alike() {
        let robot = panda();
        let n = robot.dof();
        let world = World {
            obstacles: vec![
                Obstacle::cuboid(Vec3::new(0.5, 0.0, 0.4), Vec3::new(0.05, 0.3, 0.2)),
                Obstacle::sphere(Vec3::new(0.0, 0.5, 0.5), 0.15),
                Obstacle::Capsule {
                    center: Vec3::new(-0.3, -0.3, 0.6),
                    rotation: Quat::from_rotation_x(1.0),
                    radius: 0.06,
                    half_length: 0.2,
                },
            ],
        };
        let spacing = 0.03;
        let mut rng = Rng::new(9);
        let mut random = || -> Vec<f32> { (0..n).map(|j| rng.range(robot.lower[j], robot.upper[j])).collect() };
        let queries: Vec<EdgeQuery> = (0..400)
            .map(|i| {
                let k = 1 + i % 4;
                let edges: Vec<f32> = (0..k).flat_map(|_| [random(), random()].concat()).collect();
                let wanted = if i % 2 == 0 { Wanted::Prefix } else { Wanted::FirstValid };
                EdgeQuery { world: (i % 2) as u32, edges, wanted: Some(wanted), spacing, keep: None }
            })
            .collect();
        let mut answers = vec![];
        for device in devices(&robot) {
            let worlds = device.upload(&[world.clone(), World::default()]).unwrap();
            // Four searches asking their queries in turn, run at once.
            let mut askers: Vec<Asker> =
                queries.chunks(100).map(|q| Asker { queries: q.to_vec(), answers: vec![] }).collect();
            let mut searches: Vec<&mut dyn Search> = askers.iter_mut().map(|a| a as &mut dyn Search).collect();
            device.search(&worlds, &mut searches, None).unwrap();
            let got: Vec<usize> = askers.into_iter().flat_map(|a| a.answers).collect();
            // Every state of every edge, checked one by one.
            let (mut valid_edges, mut invalid_edges) = (0, 0);
            for (query, &answer) in queries.iter().zip(&got) {
                let valid: Vec<bool> = (0..query.count(n))
                    .map(|i| {
                        let (a, b) = query.edge(i, n);
                        let total = LANES * passes(a, b, spacing);
                        let mut q = vec![0.0; total * n];
                        for (m, out) in q.chunks_mut(n).enumerate() {
                            state(a, b, m + 1, total, out);
                        }
                        let clear = device.clearance(&worlds, &vec![query.world; total], &q).unwrap();
                        clear.into_iter().all(collision_free)
                    })
                    .collect();
                valid_edges += valid.iter().filter(|&&v| v).count();
                invalid_edges += valid.iter().filter(|&&v| !v).count();
                let expected = match query.wanted.unwrap() {
                    Wanted::Prefix => valid.iter().take_while(|&&v| v).count(),
                    Wanted::FirstValid => valid.iter().position(|&v| v).unwrap_or(valid.len()),
                };
                assert_eq!(answer, expected, "{}: {query:?}", device.name());
            }
            assert!(valid_edges > 50 && invalid_edges > 50, "{valid_edges} valid and {invalid_edges} invalid edges");
            answers.push(got);
        }
        if let [cpu, gpu] = &answers[..] {
            assert_eq!(cpu, gpu, "the devices answer differently");
        }
    }

    /// An edge that passes within the clearance of an obstacle fails to keep it, unless the
    /// clearance has ramped down near the path's ends; without a clearance it is free.
    #[test]
    fn edges_keep_their_clearance_away_from_the_ends() {
        let robot = panda();
        let q = robot.default_q().to_vec();
        let hand = robot.link_pose(&q, "panda_hand").unwrap().position;
        // A ball 8 mm from the hand's spheres, measured on the device.
        let ball = |gap: f32| World { obstacles: vec![Obstacle::sphere(hand + Vec3::new(0.0, 0.0, 0.2 + gap), 0.1)] };
        let cpu = Device::cpu(&robot).unwrap();
        let probe = cpu.upload(&[ball(0.0)]).unwrap();
        let touching = cpu.clearance(&probe, &[0], &q).unwrap()[0][0];
        let world = ball(0.008 - touching);
        let near: Vec<f32> = q.iter().map(|v| v + 1e-3).collect();
        let far: Vec<f32> = q.iter().map(|v| v + 2.0).collect();
        let keep = |ends: Vec<Vec<f32>>| Some(Arc::new(Keep { clearance: 0.01, ramp: 0.5, ends }));
        let query = |keep| EdgeQuery {
            world: 0,
            edges: [&q[..], &near].concat(),
            wanted: Some(Wanted::Prefix),
            spacing: 0.01,
            keep,
        };
        let queries = vec![query(None), query(keep(vec![far.clone()])), query(keep(vec![q.clone(), far]))];
        for device in devices(&robot) {
            let worlds = device.upload(std::slice::from_ref(&world)).unwrap();
            let clear = device.clearance(&worlds, &[0], &q).unwrap()[0][0];
            assert!(
                clear > 0.005 && clear < 0.01,
                "{}: the edge should pass 5-10 mm from the ball, not {clear}",
                device.name()
            );
            let mut asker = Asker { queries: queries.clone(), answers: vec![] };
            device.search(&worlds, &mut [&mut asker as &mut dyn Search], None).unwrap();
            assert_eq!(asker.answers, [1, 0, 1], "{}: free, too close, near an end", device.name());
        }
    }
}
