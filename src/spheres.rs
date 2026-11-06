//! Collision spheres fitted to link geometry, and the self-collision link pairs not worth checking.
//!
//! Spheres follow the voxel method cuRobo uses for robot models: interior grid points become
//! candidate spheres that touch the surface, and a greedy cover picks the candidates that reach
//! the most surface samples. The overhang allowed while covering is each link's tolerance, scaled
//! by one factor searched for across the whole robot, and a final inflation grows spheres until
//! every sample is inside one. Samples no candidate reaches, on parts too thin for the grid, get
//! candidates of their own along their inward normals, and dense samples the inflation would grow
//! a sphere more than twice the overhang to reach join the samples the cover is chosen on: so no
//! sphere of closed geometry reaches farther than twice its link's scaled tolerance, which falls as
//! the budget grows. Where the budget reaches the tolerances, it holds every link within its own.
//! Parts thinner than `MIN_TANGENT_DEPTH`, and vertices whose averaged normal leads out of the
//! geometry, get no candidates of their own and may reach a little past those bounds.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use glam::{Quat, UVec3, Vec3};
use parry3d::math::Pose;
use parry3d::query;
use parry3d::shape::{TriMesh as ParryMesh, TriMeshFlags};
use rayon::prelude::*;

use crate::description::{self, TriMesh};
use crate::rng::Rng;
use crate::robot::Robot;
use crate::world::Obstacle;

/// Which link geometry spheres are fitted to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SphereGeometry {
    /// The links' collision geometry.
    #[default]
    Collision,
    /// Their visual geometry, for descriptions whose collision geometry is coarse or missing.
    Visual,
}

/// How spheres are fitted to a robot's links.
#[derive(Clone, Copy, Debug)]
pub struct SphereOptions {
    /// Spheres for the whole robot.
    pub budget: usize,
    /// How far spheres may reach beyond the geometry (meters) before the fit spends more of the
    /// budget. When the budget cannot reach it, the fit uses the smallest overhang it can.
    pub tolerance: f32,
    /// Largest spacing of the interior grid that proposes sphere centers (meters). Small links
    /// use a finer grid.
    pub voxel_size: f32,
    /// The geometry the spheres cover.
    pub geometry: SphereGeometry,
    /// Random configurations used to find link pairs that never collide.
    pub self_collision_samples: usize,
    /// Seed of the fit's random choices and those configurations.
    pub rng_seed: u64,
}

impl SphereOptions {
    /// Why these options cannot fit spheres, if they cannot.
    pub(crate) fn check(&self) -> Result<(), String> {
        let positive = |v: f32| v.is_finite() && v > 0.0;
        if self.budget == 0 || !positive(self.tolerance) || !positive(self.voxel_size) {
            return Err(format!(
                "sphere fitting needs a budget of at least one sphere and positive tolerance and voxel size, got {self:?}"
            ));
        }
        Ok(())
    }
}

impl Default for SphereOptions {
    fn default() -> Self {
        Self {
            budget: 64,
            tolerance: 0.005,
            voxel_size: 0.01,
            geometry: SphereGeometry::Collision,
            self_collision_samples: 10_000,
            rng_seed: 1,
        }
    }
}

const SURFACE_SAMPLES: usize = 1500;
/// Surface each sample every final sphere set must contain stands for (square meters): one per
/// 3 x 3 mm, from `DENSE_SAMPLES` to `MAX_DENSE_SAMPLES` per link. Surface between them may lie
/// outside the spheres by about their spacing where a sphere crosses it steeply.
const DENSE_AREA: f32 = 9e-6;
const DENSE_SAMPLES: usize = 20_000;
const MAX_DENSE_SAMPLES: usize = 80_000;
const MAX_VERTEX_SAMPLES: usize = 1500;
const MAX_CANDIDATES: usize = 1500;
const MAX_VOXELS: usize = 200_000;
/// Bisection steps of the overhang search.
const SEARCH_STEPS: usize = 16;
/// Bisection steps of a search near an earlier one's scale, within a factor of 2 of it.
const NEAR_STEPS: usize = 10;
/// Refinements of a cover, and searches after refining, at most.
const MAX_REFINEMENTS: usize = 8;
/// The shallowest center a sample's own candidate is looked for at (meters).
const MIN_TANGENT_DEPTH: f32 = 1e-4;
/// How far, in overhangs, a sphere may grow past what a cover reaches where the budget holds the
/// tolerances back. Holding growth to one overhang raises the overhang the budget allows for every
/// sphere, and a cover with tighter spheres grown in a few places checks fewer false contacts:
/// fitted to the Panda at 64 spheres, one overhang left a fifth of MotionBenchMaker's problems with
/// both ends free, against two fifths with two.
const BOUND_GROWTH: f32 = 2.0;

/// Area-weighted random points on the surface of `meshes`, each with its triangle's outward unit
/// normal.
fn surface_samples(meshes: &[TriMesh], count: usize, rng: &mut Rng) -> Vec<(Vec3, Vec3)> {
    let triangles: Vec<[Vec3; 3]> = meshes
        .iter()
        .flat_map(|m| m.triangles.iter().map(|&t| m.corners(t)))
        .collect();
    let mut cumulative = Vec::with_capacity(triangles.len());
    let mut total = 0.0;
    for &t in &triangles {
        total += description::triangle_area(t);
        cumulative.push(total);
    }
    if total <= 0.0 {
        return vec![];
    }
    (0..count)
        .map(|_| {
            let pick = rng.uniform() * total;
            let [a, b, c] = triangles[cumulative
                .partition_point(|&x| x < pick)
                .min(triangles.len() - 1)];
            let (r1, r2) = (rng.uniform().sqrt(), rng.uniform());
            let point = a * (1.0 - r1) + b * (r1 * (1.0 - r2)) + c * (r1 * r2);
            (point, (b - a).cross(c - a).normalize_or_zero())
        })
        .collect()
}

/// Every vertex of `meshes` with its outward unit normal: the area-weighted mean of its
/// triangles' normals.
fn vertex_normals(meshes: &[TriMesh]) -> Vec<(Vec3, Vec3)> {
    let mut out = vec![];
    for m in meshes {
        let mut normals = vec![Vec3::ZERO; m.vertices.len()];
        for &t in &m.triangles {
            let [a, b, c] = m.corners(t);
            let n = (b - a).cross(c - a);
            for i in t {
                normals[i as usize] += n;
            }
        }
        let vertices = m.vertices.iter().zip(normals);
        out.extend(vertices.map(|(&v, n)| (v, n.normalize_or_zero())));
    }
    out
}

/// Whether the sphere `(c, r)` reaches `p` within `overhang` beyond its radius: the one test the
/// cover, its refinement and the samples' own candidates share, so a candidate made to reach a
/// sample is one the cover counts as reaching it.
#[inline]
fn reaches(p: Vec3, (c, r): (Vec3, f32), overhang: f32) -> bool {
    (p - c).length_squared() <= (r + overhang) * (r + overhang)
}

/// The largest sphere centered at `p` inside any of `solids`, if `p` is inside one.
fn inscribed(solids: &[ParryMesh], p: Vec3) -> Option<(Vec3, f32)> {
    use parry3d::query::PointQuery;

    solids
        .iter()
        .filter_map(|s| {
            let proj = s.project_local_point(p, false);
            proj.is_inside.then(|| (p - proj.point).length())
        })
        .reduce(f32::max)
        .map(|r| (p, r))
}

/// One link's surface samples and candidate spheres (center, largest radius inside the geometry).
struct LinkFit {
    /// Points the cover is chosen to reach: mesh vertices and random surface points, then the
    /// dense samples covers missed (`refined_cover`).
    samples: Vec<Vec3>,
    /// Each sample's outward unit normal (zero where the surface has none).
    normals: Vec<Vec3>,
    /// Surface points the final spheres must contain, with their outward unit normals.
    dense: Vec<(Vec3, Vec3)>,
    /// Per dense sample, whether it has joined `samples`.
    joined: Vec<bool>,
    candidates: Vec<(Vec3, f32)>,
    /// The geometry, as parry tests points against it.
    solids: Vec<ParryMesh>,
    /// The largest candidate's radius, before the candidates are thinned out.
    largest: f32,
    /// Whether its loops run on the current rayon pool, or on the calling thread alone.
    parallel: bool,
}

impl LinkFit {
    fn new(meshes: &[TriMesh], o: &SphereOptions, rng: &mut Rng, parallel: bool) -> Option<Self> {
        let vertices = vertex_normals(meshes);
        let stride = vertices.len().div_ceil(MAX_VERTEX_SAMPLES).max(1);
        let mut sampled: Vec<(Vec3, Vec3)> = vertices.into_iter().step_by(stride).collect();
        sampled.extend(surface_samples(meshes, SURFACE_SAMPLES, rng));
        if sampled.is_empty() {
            return None;
        }
        let (samples, normals): (Vec<Vec3>, Vec<Vec3>) = sampled.into_iter().unzip();
        let area: f32 = meshes
            .iter()
            .flat_map(|m| {
                m.triangles
                    .iter()
                    .map(|&t| description::triangle_area(m.corners(t)))
            })
            .sum();
        let count = ((area / DENSE_AREA) as usize).clamp(DENSE_SAMPLES, MAX_DENSE_SAMPLES);
        let dense = surface_samples(meshes, count, rng);
        let flags = TriMeshFlags::MERGE_DUPLICATE_VERTICES
            | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES
            | TriMeshFlags::ORIENTED;
        let solids: Vec<ParryMesh> = meshes
            .iter()
            .filter_map(|m| {
                ParryMesh::with_flags(m.vertices.clone(), m.triangles.clone(), flags).ok()
            })
            .collect();
        let (lo, hi) = samples
            .iter()
            .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), &p| {
                (lo.min(p), hi.max(p))
            });
        let extent = (hi - lo).max(Vec3::splat(1e-4));
        let mut voxel = o.voxel_size.min(extent.min_element() / 4.0).max(1e-3);
        let cells = |voxel: f32| (extent / voxel).ceil().max(Vec3::ONE).as_uvec3();
        while cells(voxel).element_product() as usize > MAX_VOXELS {
            voxel *= 1.25;
        }
        let n = cells(voxel);
        let cell = |i: u32| {
            let (x, y, z) = (i % n.x, i / n.x % n.y, i / (n.x * n.y));
            inscribed(
                &solids,
                lo + (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * voxel,
            )
        };
        let cells = 0..n.element_product();
        let mut candidates: Vec<(Vec3, f32)> = if parallel {
            cells.into_par_iter().filter_map(cell).collect()
        } else {
            cells.filter_map(cell).collect()
        };
        if candidates.is_empty() {
            // Open or degenerate geometry has no interior; grow spheres from the surface instead.
            candidates = samples.iter().map(|&p| (p, 0.0)).collect();
        }
        let largest = candidates.iter().map(|c| c.1).fold(0.0, f32::max);
        let stride = candidates.len().div_ceil(MAX_CANDIDATES).max(1);
        Some(Self {
            samples,
            normals,
            joined: vec![false; dense.len()],
            dense,
            candidates: candidates.into_iter().step_by(stride).collect(),
            solids,
            largest,
            parallel,
        })
    }

    /// How `spheres` (`[x, y, z, radius]`, in the meshes' frame) fit this geometry: their largest
    /// overhang, the share of the dense samples outside every one, and the geometry's thickness.
    fn measure(&self, link: &str, spheres: &[[f32; 4]]) -> LinkQualification {
        use parry3d::query::PointQuery;

        // How deep `p` lies inside the geometry: negative outside.
        let depth = |p: Vec3| {
            inscribed(&self.solids, p).map_or_else(
                || {
                    -self
                        .solids
                        .iter()
                        .map(|s| s.distance_to_local_point(p, false))
                        .fold(f32::INFINITY, f32::min)
                },
                |(_, r)| r,
            )
        };
        let overhang = spheres
            .iter()
            .map(|s| s[3] - depth(Vec3::new(s[0], s[1], s[2])))
            .fold(0.0, f32::max);
        let inside = |p: &&Vec3| {
            spheres
                .iter()
                .any(|s| p.distance(Vec3::new(s[0], s[1], s[2])) <= s[3] * (1.0 + 1e-5))
        };
        let outside = self.dense.iter().filter(|(p, _)| !inside(&p)).count();
        LinkQualification {
            link: link.to_string(),
            spheres: spheres.len(),
            overhang,
            uncovered: outside as f32 / self.dense.len().max(1) as f32,
            thickness: 2.0 * self.largest,
        }
    }

    /// Greedy cover: candidates reaching each sample within `overhang` beyond their radius,
    /// picked by how many uncovered samples they reach, up to `limit` of them. Returns the picks
    /// and the samples they leave: without a limit, those no candidate reaches.
    fn greedy(&self, overhang: f32, limit: usize) -> (Vec<usize>, Vec<usize>) {
        let words = self.samples.len().div_ceil(64);
        // Samples by x, so each candidate tests only those within its reach along x.
        let mut by_x: Vec<usize> = (0..self.samples.len()).collect();
        by_x.sort_by(|&a, &b| self.samples[a].x.total_cmp(&self.samples[b].x));
        let xs: Vec<f32> = by_x.iter().map(|&i| self.samples[i].x).collect();
        let reached = |&(c, r): &(Vec3, f32)| {
            let mut bits = vec![0u64; words];
            let first = xs.partition_point(|&x| x < c.x - (r + overhang));
            let last = xs.partition_point(|&x| x <= c.x + (r + overhang));
            for &i in &by_x[first..last] {
                if reaches(self.samples[i], (c, r), overhang) {
                    bits[i / 64] |= 1 << (i % 64);
                }
            }
            bits
        };
        let cover: Vec<Vec<u64>> = if self.parallel {
            self.candidates.par_iter().map(reached).collect()
        } else {
            self.candidates.iter().map(reached).collect()
        };
        let mut uncovered = vec![!0u64; words];
        if !self.samples.len().is_multiple_of(64) {
            uncovered[words - 1] = (1u64 << (self.samples.len() % 64)) - 1;
        }
        // Lazy greedy (Minoux): a candidate's gain only falls as samples get covered, so a gain
        // computed earlier bounds its gain now, and the top of the heap, recomputed and still on
        // top, is the best. Ties go to the larger sphere, then the lower index, as a full scan's.
        let gain = |i: usize, uncovered: &[u64]| -> u32 {
            cover[i]
                .iter()
                .zip(uncovered)
                .map(|(c, u)| (c & u).count_ones())
                .sum()
        };
        // Radii are distances, never negative, so their bits order as they do.
        let key = |i: usize, gain: u32| (gain, self.candidates[i].1.to_bits(), Reverse(i));
        let mut heap: BinaryHeap<(u32, u32, Reverse<usize>)> = (0..self.candidates.len())
            .map(|i| key(i, gain(i, &uncovered)))
            .filter(|k| k.0 > 0)
            .collect();
        let mut picks = vec![];
        while picks.len() < limit
            && let Some((bound, _, Reverse(best))) = heap.pop()
        {
            let now = gain(best, &uncovered);
            if now < bound {
                if now > 0 {
                    heap.push(key(best, now));
                }
                continue;
            }
            picks.push(best);
            for (u, c) in uncovered.iter_mut().zip(&cover[best]) {
                *u &= !c;
            }
        }
        let left = (0..self.samples.len())
            .filter(|&i| uncovered[i / 64] & (1 << (i % 64)) != 0)
            .collect();
        (picks, left)
    }

    /// A greedy cover at `overhang` that reaches every dense sample within `growth` times
    /// `overhang`, at most `MAX_REFINEMENTS` times refined: samples no candidate reaches get
    /// candidates of their own ([`LinkFit::tangent`]), and dense samples the cover leaves farther
    /// join the samples.
    fn refined_cover(&mut self, overhang: f32, growth: f32) -> Vec<usize> {
        let any_reaches =
            |spheres: &[(Vec3, f32)], p: Vec3| spheres.iter().any(|&s| reaches(p, s, overhang));
        let mut picks = vec![];
        for _ in 0..=MAX_REFINEMENTS {
            let left;
            (picks, left) = self.greedy(overhang, usize::MAX);
            // One candidate reaches many neighbouring samples: those an added one reaches get
            // none.
            let mut added: Vec<(Vec3, f32)> = vec![];
            for i in left {
                let (p, n) = (self.samples[i], self.normals[i]);
                if !any_reaches(&added, p)
                    && let Some(candidate) = self.tangent(p, n, overhang)
                {
                    added.push(candidate);
                }
            }
            if !added.is_empty() {
                self.candidates.extend(added);
                continue;
            }
            // Dense samples that joined and are still missed are those no candidate reaches.
            let picked: Vec<(Vec3, f32)> = picks.iter().map(|&i| self.candidates[i]).collect();
            let missed = |&i: &usize| {
                let p = self.dense[i].0;
                !self.joined[i] && !picked.iter().any(|&s| reaches(p, s, growth * overhang))
            };
            let dense = 0..self.dense.len();
            let missed: Vec<usize> = if self.parallel {
                dense.into_par_iter().filter(missed).collect()
            } else {
                dense.filter(missed).collect()
            };
            if missed.is_empty() {
                break;
            }
            for i in missed {
                let (p, n) = self.dense[i];
                self.samples.push(p);
                self.normals.push(n);
                self.joined[i] = true;
            }
        }
        picks
    }

    /// The largest sphere inside the geometry that reaches surface point `p` within `overhang`
    /// beyond its radius, as `greedy` tests it, centered along its inward normal `-n`: the deepest
    /// center that does, halving the depth from the larger of the link's largest inscribed radius
    /// and the overhang, down to `MIN_TANGENT_DEPTH`. `None` where `n` is zero or no such center
    /// lies inside, as where a vertex's averaged normal leads out of the geometry.
    fn tangent(&self, p: Vec3, n: Vec3, overhang: f32) -> Option<(Vec3, f32)> {
        if n == Vec3::ZERO {
            return None;
        }
        let mut depth = self.largest.max(overhang).max(MIN_TANGENT_DEPTH);
        loop {
            if let Some(candidate) = inscribed(&self.solids, p - n * depth)
                && reaches(p, candidate, overhang)
            {
                return Some(candidate);
            }
            if depth <= MIN_TANGENT_DEPTH {
                return None;
            }
            depth = (0.5 * depth).max(MIN_TANGENT_DEPTH);
        }
    }

    /// The picked spheres, grown until every sample is inside one, and the largest growth. Each
    /// sample outside them grows the pick that reaches it with the least overhang, judged by the
    /// picks' original radii so growth cannot snowball onto one sphere.
    fn spheres(&self, picks: &[usize]) -> (Vec<[f32; 4]>, f32) {
        // A cover that picked nothing still needs one sphere to grow: the largest candidate.
        let largest = (0..self.candidates.len())
            .max_by(|&a, &b| self.candidates[a].1.total_cmp(&self.candidates[b].1));
        let picks: Vec<usize> = if picks.is_empty() {
            largest.into_iter().collect()
        } else {
            picks.to_vec()
        };
        let mut spheres: Vec<(Vec3, f32)> = picks.iter().map(|&i| self.candidates[i]).collect();
        let dense = self.dense.iter().map(|(p, _)| p);
        for &s in self.samples.iter().chain(dense) {
            if spheres.iter().any(|&(c, r)| (s - c).length() <= r) {
                continue;
            }
            let overhang = |a: usize| (s - spheres[a].0).length() - self.candidates[picks[a]].1;
            let j = (0..spheres.len())
                .min_by(|&a, &b| overhang(a).total_cmp(&overhang(b)))
                .expect("a sphere");
            spheres[j].1 = (s - spheres[j].0).length();
        }
        let growth = picks
            .iter()
            .zip(&spheres)
            .map(|(&i, s)| s.1 - self.candidates[i].1)
            .fold(0.0, f32::max);
        (
            spheres
                .into_iter()
                .map(|(c, r)| [c.x, c.y, c.z, r])
                .collect(),
            growth,
        )
    }
}

/// How well a robot's collision spheres fit its geometry ([`Robot::qualify`](crate::Robot::qualify)):
/// a robot that loads may still have spheres unfit to plan with, such as fingers inside spheres too
/// large to grasp small objects.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Qualification {
    /// Every link with both geometry and spheres, in the robot's order.
    pub links: Vec<LinkQualification>,
    /// Per hand: how far its spheres reach past its IK frame ([`Robot::tip`](crate::Robot::tip)).
    pub tips: Vec<f32>,
}

/// How well one link's spheres fit its geometry.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct LinkQualification {
    /// The link's name.
    pub link: String,
    /// How many collision spheres it has.
    pub spheres: usize,
    /// How far its spheres reach beyond its geometry, at most (meters): a sphere's radius less
    /// how deep its center lies inside.
    pub overhang: f32,
    /// The share of its geometry's surface outside every sphere, from fresh samples, one per
    /// 3 x 3 mm of it (20,000 to 80,000): up to about 2% for fitted spheres, which cover their own
    /// samples and leave slivers about a millimeter deep between them where held tight, and more
    /// where a hand-tuned model leaves gaps.
    pub uncovered: f32,
    /// How thick its geometry is where thickest (meters): the largest sphere inside it, across. A
    /// link thinner than its spheres' overhang is mostly sphere.
    pub thickness: f32,
}

/// How `spheres` fit `meshes` (both in the link's frame), or `None` where the meshes have no
/// surface. The same samples on every call, seeded by `o`.
pub(crate) fn measure(
    link: &str,
    meshes: &[TriMesh],
    spheres: &[[f32; 4]],
    o: &SphereOptions,
) -> Option<LinkQualification> {
    // Samples the fit did not draw: its rng starts at `o.rng_seed`.
    let fit = LinkFit::new(
        meshes,
        o,
        &mut Rng::new(o.rng_seed.wrapping_add(1)),
        /* parallel */ true,
    )?;
    Some(fit.measure(link, spheres))
}

/// The smallest scale, from 1 up, of every link's tolerance (`tolerances`, per link) at which the
/// greedy covers of all links fit in `o.budget` spheres. `near`, the scale of an earlier search of
/// the same links, narrows the search to around it.
fn search_scale(
    fits: &[Option<LinkFit>],
    tolerances: &[f32],
    o: &SphereOptions,
    near: Option<f32>,
) -> f32 {
    let fits_budget = |scale: f32| {
        let mut total = 0;
        for (f, &tolerance) in fits.iter().zip(tolerances) {
            let Some(f) = f else { continue };
            let (picks, left) = f.greedy(tolerance * scale, o.budget + 1 - total);
            total += picks.len() + usize::from(!left.is_empty());
            if total > o.budget {
                return false;
            }
        }
        true
    };
    if fits_budget(1.0) {
        return 1.0;
    }
    // A scale at which one sphere covers any link.
    let mut top = 1.0f32;
    for (f, &tolerance) in fits.iter().zip(tolerances) {
        let Some(f) = f else { continue };
        let (lo_corner, hi_corner) = f
            .samples
            .iter()
            .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(a, b), &p| {
                (a.min(p), b.max(p))
            });
        top = top.max((hi_corner - lo_corner).length() / tolerance);
    }
    let (mut lo, mut hi, steps) = match near {
        Some(near) => {
            let (mut lo, mut hi) = ((0.5 * near).max(1.0), (2.0 * near).min(top));
            while hi < top && !fits_budget(hi) {
                (lo, hi) = (hi, (2.0 * hi).min(top));
            }
            while lo > 1.0 && fits_budget(lo) {
                (lo, hi) = ((0.5 * lo).max(1.0), lo);
            }
            (lo, hi, NEAR_STEPS)
        }
        None => (1.0, top, SEARCH_STEPS),
    };
    for _ in 0..steps {
        let mid = (lo * hi).sqrt();
        if fits_budget(mid) { hi = mid } else { lo = mid }
    }
    hi
}

/// Spheres fitted to links ([`fit`]).
pub(crate) struct Fitted {
    /// Per link, spheres `[x, y, z, radius]` in its frame.
    pub(crate) spheres: Vec<Vec<[f32; 4]>>,
    /// How far the farthest reaches beyond the surface it touches (meters).
    pub(crate) overhang: f32,
    /// What every link's tolerance was scaled by for the budget: 1 where it reached them.
    pub(crate) scale: f32,
}

/// Spheres for each link's meshes (in the link frame), at most `o.budget` in total. Each closed
/// link's overhang is its tolerance (`tolerances`, per link) when the budget allows, otherwise at
/// most twice its tolerance scaled up by the smallest factor the budget can achieve, but for the
/// parts the module doc names. With `parallel`,
/// the work runs on the current rayon pool; without, on the calling thread alone, without
/// registering it with any pool.
pub(crate) fn fit(
    links: &[Vec<TriMesh>],
    tolerances: &[f32],
    o: &SphereOptions,
    parallel: bool,
) -> Fitted {
    debug_assert!(o.check().is_ok(), "callers check the options");
    debug_assert_eq!(links.len(), tolerances.len());
    let mut rng = Rng::new(o.rng_seed);
    let mut fits: Vec<Option<LinkFit>> = links
        .iter()
        .map(|m| LinkFit::new(m, o, &mut rng, parallel))
        .collect();

    // Cover every link at the scale the budget allows, refining each cover until spheres grow by
    // at most `BOUND_GROWTH` overhangs to reach every dense sample. Refining adds samples, so the
    // covers may outgrow the budget: then search again from that scale, which only grows.
    let cover = |fits: &mut [Option<LinkFit>], scale: f32, growth: f32| -> Vec<Vec<usize>> {
        fits.iter_mut()
            .zip(tolerances)
            .map(|(f, &tolerance)| {
                f.as_mut()
                    .map_or(vec![], |f| f.refined_cover(tolerance * scale, growth))
            })
            .collect()
    };
    let total = |picks: &[Vec<usize>]| picks.iter().map(Vec::len).sum::<usize>();
    let mut scale = search_scale(&fits, tolerances, o, None);
    let mut picks: Option<Vec<Vec<usize>>> = None;
    for _ in 0..MAX_REFINEMENTS {
        let covers = cover(&mut fits, scale, BOUND_GROWTH);
        if total(&covers) <= o.budget {
            picks = Some(covers);
            break;
        }
        scale = search_scale(&fits, tolerances, o, Some(scale));
    }
    // Past the refinements, the covers the last search counted, which keep to the budget.
    let mut picks = picks.unwrap_or_else(|| {
        fits.iter()
            .zip(tolerances)
            .map(|(f, &tolerance)| {
                f.as_ref()
                    .map_or(vec![], |f| f.greedy(tolerance * scale, usize::MAX).0)
            })
            .collect()
    });
    // Where the budget reaches the tolerances, it may hold every link within its own.
    if scale == 1.0 {
        let strict = cover(&mut fits, 1.0, 1.0);
        if total(&strict) <= o.budget {
            picks = strict;
        }
    }
    let fitted: Vec<(Vec<[f32; 4]>, f32)> = fits
        .iter()
        .zip(&picks)
        .map(|(f, picks)| f.as_ref().map_or((vec![], 0.0), |f| f.spheres(picks)))
        .collect();
    Fitted {
        overhang: fitted.iter().map(|f| f.1).fold(0.0, f32::max),
        spheres: fitted.into_iter().map(|f| f.0).collect(),
        scale,
    }
}

/// Spheres for an attached object's `shapes` (in its link's frame) without fitting, if every shape
/// is a box or a sphere: a sphere is itself, and a box splits into a grid of cells, each covered by
/// the sphere through its corners. That sphere reaches past the box's faces by its radius less the
/// cell's smallest half size, so the grid with the fewest spheres within `o.tolerance` is taken, or
/// where the budget cannot reach it, the least overhang the budget can. Boxes share the budget the
/// spheres leave. Microseconds, where fitting a box takes milliseconds.
pub(crate) fn cover_primitives(shapes: &[Obstacle], o: &SphereOptions) -> Option<Vec<[f32; 4]>> {
    if shapes
        .iter()
        .any(|s| !matches!(s, Obstacle::Cuboid { .. } | Obstacle::Sphere { .. }))
    {
        return None;
    }
    let boxes = shapes
        .iter()
        .filter(|s| matches!(s, Obstacle::Cuboid { .. }))
        .count();
    let per_box = o
        .budget
        .checked_sub(shapes.len() - boxes)?
        .checked_div(boxes)
        .unwrap_or(0);
    if boxes > 0 && per_box == 0 {
        return None;
    }
    let mut spheres = vec![];
    for shape in shapes {
        match *shape {
            Obstacle::Sphere { center, radius } => {
                spheres.push([center.x, center.y, center.z, radius])
            }
            Obstacle::Cuboid {
                center,
                half_extents,
                rotation,
            } => {
                let cells = box_cells(half_extents, o.tolerance, per_box);
                let cell = half_extents / cells.as_vec3();
                // A hair wide, so rounding never leaves a corner uncovered.
                let radius = cell.length() * (1.0 + 1e-6);
                for i in 0..cells.x {
                    for j in 0..cells.y {
                        for k in 0..cells.z {
                            let at = (UVec3::new(2 * i + 1, 2 * j + 1, 2 * k + 1).as_vec3() * cell)
                                - half_extents;
                            let p = center + rotation * at;
                            spheres.push([p.x, p.y, p.z, radius]);
                        }
                    }
                }
            }
            _ => unreachable!("only boxes and spheres reach here"),
        }
    }
    Some(spheres)
}

/// The cells per axis splitting a box of `half` extents for `cover_primitives`: the fewest within
/// `tolerance` of overhang and `budget` spheres, else the least overhang within the budget.
fn box_cells(half: Vec3, tolerance: f32, budget: usize) -> UVec3 {
    let mut best: Option<(bool, usize, f32, UVec3)> = None;
    for x in 1..=budget {
        for y in 1..=budget / x {
            for z in 1..=budget / (x * y) {
                let cells = UVec3::new(x as u32, y as u32, z as u32);
                let cell = half / cells.as_vec3();
                let overhang = cell.length() - cell.min_element();
                let candidate = (overhang <= tolerance, x * y * z, overhang, cells);
                let better = |b: &(bool, usize, f32, UVec3)| match (candidate.0, b.0) {
                    (true, false) => true,
                    (false, true) => false,
                    (true, true) => (candidate.1, candidate.2) < (b.1, b.2),
                    (false, false) => (candidate.2, candidate.1) < (b.2, b.1),
                };
                if best.as_ref().is_none_or(better) {
                    best = Some(candidate);
                }
            }
        }
    }
    best.map_or(UVec3::ONE, |b| b.3)
}

/// Link pairs not worth checking for self-collision, by the rules of MoveIt's setup assistant:
/// adjacent links, pairs whose spheres already overlap at the default configuration, and pairs
/// whose geometry never touches across random configurations. The last rule uses the meshes, not
/// the spheres, so the spheres' overhang does not keep pairs that cannot actually collide.
pub(crate) fn ignore_pairs(
    robot: &Robot,
    geometry: &[Vec<TriMesh>],
    o: &SphereOptions,
) -> BTreeMap<String, Vec<String>> {
    let links = robot.links.len();
    let solids: Vec<Option<ParryMesh>> = geometry
        .iter()
        .map(|meshes| {
            let mut vertices = vec![];
            let mut triangles = vec![];
            for m in meshes {
                let base = vertices.len() as u32;
                vertices.extend(&m.vertices);
                triangles.extend(m.triangles.iter().map(|t| t.map(|i| i + base)));
            }
            ParryMesh::with_flags(vertices, triangles, TriMeshFlags::MERGE_DUPLICATE_VERTICES).ok()
        })
        .collect();
    // Pairs as (lower index, higher index); a link's parent always has the lower index.
    let mut ignore = BTreeSet::new();
    let solid_links = (0..links).filter(|&l| solids[l].is_some());
    for l in solid_links {
        let mut p = robot.links[l].parent;
        while let Some(i) = p {
            if solids[i].is_some() {
                ignore.insert((i, l));
                break;
            }
            p = robot.links[i].parent;
        }
    }
    let pairs: Vec<(usize, usize)> = (0..links)
        .flat_map(|a| (a + 1..links).map(move |b| (a, b)))
        .filter(|&(a, b)| solids[a].is_some() && solids[b].is_some() && !ignore.contains(&(a, b)))
        .collect();
    let touching = |q: &[f32]| -> Vec<bool> {
        let fk = robot.fk(q);
        let pose = |i: usize| Pose::from_parts(fk.pos[i], Quat::from_mat3(&fk.rot[i]));
        pairs
            .iter()
            .map(|&(a, b)| {
                let (sa, sb) = (
                    solids[a].as_ref().expect("filtered"),
                    solids[b].as_ref().expect("filtered"),
                );
                query::intersection_test(&pose(a), sa, &pose(b), sb)
                    .is_ok_and(|hit| hit.intersecting)
            })
            .collect()
    };
    let fk = robot.fk(&robot.default_q);
    let centers: Vec<Vec3> = robot
        .spheres
        .iter()
        .map(|s| fk.rot[s.link] * s.center + fk.pos[s.link])
        .collect();
    let mut at_default = BTreeSet::new();
    for (a, sa) in robot.spheres.iter().enumerate() {
        for (b, sb) in robot.spheres.iter().enumerate().skip(a + 1) {
            let reach = sa.radius + sa.self_buffer + sb.radius + sb.self_buffer;
            if sa.link != sb.link && (centers[a] - centers[b]).length() < reach {
                at_default.insert((sa.link.min(sb.link), sa.link.max(sb.link)));
            }
        }
    }
    let ever = (0..o.self_collision_samples as u64)
        .into_par_iter()
        .map(|k| {
            let mut rng = Rng::new(
                o.rng_seed
                    .wrapping_add(k.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            );
            let q: Vec<f32> = (0..robot.dof())
                .map(|j| rng.range(robot.lower[j], robot.upper[j]))
                .collect();
            touching(&q)
        })
        .reduce(
            || vec![false; pairs.len()],
            |a, b| a.iter().zip(&b).map(|(x, y)| x | y).collect(),
        );
    for (k, &pair) in pairs.iter().enumerate() {
        if at_default.contains(&pair) || !ever[k] {
            ignore.insert(pair);
        }
    }
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (a, b) in ignore {
        out.entry(robot.links[a].name.clone())
            .or_default()
            .push(robot.links[b].name.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::description::Geometry;

    /// The candidate a full scan of every candidate's gain picks next: the most samples, ties to
    /// the larger sphere, then the lower index.
    fn full_scan(fit: &LinkFit, overhang: f32, limit: usize) -> Vec<usize> {
        let reached: Vec<Vec<bool>> = fit
            .candidates
            .iter()
            .map(|&(c, r)| {
                let reach = r + overhang;
                fit.samples
                    .iter()
                    .map(|&s| (s - c).length_squared() <= reach * reach)
                    .collect()
            })
            .collect();
        let mut uncovered = vec![true; fit.samples.len()];
        let mut picks = vec![];
        while picks.len() < limit {
            let gain = |i: usize| {
                (0..uncovered.len())
                    .filter(|&s| uncovered[s] && reached[i][s])
                    .count()
            };
            let (best, most) = (0..fit.candidates.len())
                .map(|i| (i, gain(i)))
                .max_by(|a, b| {
                    let radius = |i: usize| fit.candidates[i].1;
                    a.1.cmp(&b.1)
                        .then(radius(a.0).total_cmp(&radius(b.0)))
                        .then(b.0.cmp(&a.0))
                })
                .expect("the shapes have interior candidates");
            if most == 0 {
                break;
            }
            for (s, covered) in uncovered.iter_mut().enumerate() {
                *covered &= !reached[best][s];
            }
            picks.push(best);
        }
        picks
    }

    /// Lazy greedy picks what a full scan does, at every overhang and limit: a box and a thin
    /// cylinder give candidates of many sizes, and spheres of one size tie.
    #[test]
    fn lazy_greedy_picks_as_a_full_scan_does() {
        let mut mesh = Geometry::Box {
            half: Vec3::new(0.04, 0.02, 0.01),
        }
        .mesh()
        .expect("a box meshes");
        let mut fin = Geometry::Cylinder {
            radius: 0.004,
            half_length: 0.03,
        }
        .mesh()
        .expect("a cylinder meshes");
        for v in &mut fin.vertices {
            *v += Vec3::new(0.0, 0.0, 0.04);
        }
        mesh.append(fin);
        let o = SphereOptions::default();
        let fit = LinkFit::new(&[mesh], &o, &mut Rng::new(3), /* parallel */ false)
            .expect("the shapes have surface");
        for overhang in [0.0005, 0.002, 0.01, 0.05] {
            for limit in [3, usize::MAX] {
                let (picks, _) = fit.greedy(overhang, limit);
                assert!(!picks.is_empty(), "every overhang needs a sphere at least");
                assert_eq!(
                    picks,
                    full_scan(&fit, overhang, limit),
                    "overhang {overhang}"
                );
            }
        }
    }
}
