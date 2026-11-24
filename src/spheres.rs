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
//! the budget grows. Where the budget reaches the tolerances, what it has left holds spheres
//! tighter: the least growth past them it affords, down to none, every link within its own.
//! Parts thinner than `MIN_TANGENT_DEPTH`, and vertices whose averaged normal leads out of the
//! geometry, get no candidates of their own and may reach a little past those bounds.
//!
//! Inside a closed mesh is where its surface winds around the point (`winding`): shells that
//! overlap are their union, a shell is solid whichever way it faces, and one facing inward inside
//! one facing outward is a cavity. The grid's cells and the depth of a sphere's center are told so.
//! A mesh with open edges has no inside, and the side of its nearest face stands in for it. So it
//! does for a sample's own candidate, a little way in from its face: a shell that faces inward
//! and is thinner than the grid has those on its outside, as it had before the grid was told so.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::f32::consts::{PI, TAU};
use std::sync::Arc;

use glam::{Quat, UVec3, Vec3, Vec4};
use parry3d::math::Pose;
use parry3d::query;
use parry3d::shape::{TriMesh as ParryMesh, TriMeshFlags};
use rayon::prelude::*;

use crate::description::{self, TriMesh};
use crate::rng::Rng;
use crate::robot::Robot;
use crate::winding::{Crossings, Surface};
use crate::world::{self, Obstacle};

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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereOptions {
    /// Spheres for the whole robot, at most. It caps the robot's own spheres only: an object it
    /// holds ([`AttachedObject::spheres`](crate::AttachedObject::spheres),
    /// [`Held::spheres`](crate::Held::spheres)) takes as many as its `tolerance` does.
    pub budget: usize,
    /// How far spheres may reach beyond the geometry (meters) before the fit spends more of the
    /// budget. When the budget cannot reach it, the fit uses the smallest overhang it can. An
    /// object a robot holds always reaches it, however many spheres that takes, and its spheres
    /// hold all of it, but for rounding, unless `max_overhang` holds them back: for it, the
    /// tolerance is the knob.
    pub tolerance: f32,
    /// The farthest any sphere may reach beyond the geometry (meters), whatever the budget: each
    /// sphere's radius is at most how deep its center lies inside plus this
    /// ([`LinkQualification::overhang`]), so `Some(0.0)` keeps every sphere inside. Spheres are
    /// chosen as without it and then held to it, which leaves surface outside them wherever
    /// `tolerance`, as the budget scales it, is the larger: [`Robot::qualify`](crate::Robot::qualify)
    /// reports how deep as each link's `gap`. The geometry must be closed, every edge shared by
    /// as many triangles one way as the other, with no shell facing inward, and a collision model
    /// given with the options is held to it as a fitted one is. A center's depth is its distance to the nearest face of a
    /// mesh that winds around it, a face buried in another shell among them, so it is never more
    /// than the true depth. Depths are measured in the geometry as the fit sees it, a round shape
    /// as its mesh of 24 sides, which lies inside it by up to 0.9% of a cylinder's radius and 1.7%
    /// of a ball's or a capsule's; a held shape covered without fitting is held to the shape
    /// itself, so `qualify` reads that much more for it. `None`: spheres cover the surface and
    /// reach as far past it as the budget makes them.
    pub max_overhang: Option<f32>,
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
        if let Some(limit) = self.max_overhang
            && !(limit.is_finite() && limit >= 0.0)
        {
            return Err(format!(
                "max_overhang must be finite and not negative, got {limit}"
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
            max_overhang: None,
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
/// Searches of the scale after refining covers, at most.
const MAX_SEARCHES: usize = 8;
/// The shallowest center a sample's own candidate is looked for at (meters).
const MIN_TANGENT_DEPTH: f32 = 1e-4;
/// How far, in overhangs, a sphere may grow past what a cover reaches where the budget holds the
/// tolerances back. Holding growth to one overhang raises the overhang the budget allows for every
/// sphere, and a cover with tighter spheres grown in a few places checks fewer false contacts:
/// fitted to the Panda at 64 spheres, one overhang left a fifth of MotionBenchMaker's problems with
/// both ends free, against two fifths with two.
const BOUND_GROWTH: f32 = 2.0;
/// How near the least growth the budget affords its search comes, where the budget reaches the
/// tolerances: covers that reach within it of their tolerances stay as they are.
const GROWTH_PRECISION: f32 = 1.0 / 16.0;

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

/// Whether the sphere `(c, r)` reaches `p` within `overhang` beyond its radius: the scalar reference
/// for the cover, its refinement and the samples' own candidates.
#[inline]
fn reaches(p: Vec3, (c, r): (Vec3, f32), overhang: f32) -> bool {
    (p - c).length_squared() <= (r + overhang) * (r + overhang)
}

/// Four spheres checked together, with the same rounding and boundary as [`reaches`].
struct SphereCover {
    groups: Vec<[Vec4; 4]>,
}

impl SphereCover {
    fn new(candidates: &[(Vec3, f32)], picks: &[usize], overhang: f32) -> Self {
        let groups = picks
            .chunks(4)
            .map(|chunk| {
                // Repeat a real sphere in the tail so unused lanes cannot create a hit.
                let spheres: [_; 4] =
                    std::array::from_fn(|i| candidates[chunk[i.min(chunk.len() - 1)]]);
                let x = Vec4::from_array(spheres.map(|s| s.0.x));
                let y = Vec4::from_array(spheres.map(|s| s.0.y));
                let z = Vec4::from_array(spheres.map(|s| s.0.z));
                let radius = Vec4::from_array(spheres.map(|s| s.1 + overhang));
                [x, y, z, radius * radius]
            })
            .collect();
        Self { groups }
    }

    fn contains(&self, p: Vec3) -> bool {
        self.groups.iter().any(|&[x, y, z, radius_squared]| {
            let dx = Vec4::splat(p.x) - x;
            let dy = Vec4::splat(p.y) - y;
            let dz = Vec4::splat(p.z) - z;
            (dx * dx + dy * dy + dz * dz).cmple(radius_squared).any()
        })
    }
}

/// The largest sphere centered at `p` inside any of `solids`, if `p` is inside one, as the side of
/// the nearest face tells it: for points a little way in from a face ([`LinkFit::tangent`]).
fn inscribed(solids: &[Solid], p: Vec3) -> Option<(Vec3, f32)> {
    use parry3d::query::PointQuery;

    solids
        .iter()
        .filter_map(|s| {
            let proj = s.mesh.project_local_point(p, false);
            proj.is_inside.then(|| (p - proj.point).length())
        })
        .reduce(f32::max)
        .map(|r| (p, r))
}

/// One link's surface samples and candidate spheres (center, largest radius inside the geometry).
#[derive(Clone)]
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
    /// The geometry, as points are tested against it, shared by a fit's copies.
    solids: Arc<[Solid]>,
    /// The largest candidate's radius, before the candidates are thinned out.
    largest: f32,
    /// Whether its loops run on the current rayon pool, or on the calling thread alone.
    parallel: bool,
    /// Whether its samples propose their medial balls, and covers drop the picks others make
    /// redundant.
    medial: bool,
}

impl LinkFit {
    /// With `surface`, points of the surface every final sphere set must contain in place of
    /// random dense samples, each sample also proposes its medial ball ([`LinkFit::medial`]) and
    /// covers drop redundant picks.
    fn new(
        meshes: &[TriMesh],
        o: &SphereOptions,
        rng: &mut Rng,
        parallel: bool,
        surface: Option<&[(Vec3, Vec3)]>,
    ) -> Option<Self> {
        use parry3d::query::PointQuery;

        let vertices = vertex_normals(meshes);
        let stride = vertices.len().div_ceil(MAX_VERTEX_SAMPLES).max(1);
        let mut sampled: Vec<(Vec3, Vec3)> = vertices.into_iter().step_by(stride).collect();
        sampled.extend(surface_samples(meshes, SURFACE_SAMPLES, rng));
        if sampled.is_empty() {
            return None;
        }
        let (samples, normals): (Vec<Vec3>, Vec<Vec3>) = sampled.into_iter().unzip();
        let dense = match surface {
            Some(surface) => surface.to_vec(),
            None => {
                let area: f32 = meshes
                    .iter()
                    .flat_map(|m| {
                        m.triangles
                            .iter()
                            .map(|&t| description::triangle_area(m.corners(t)))
                    })
                    .sum();
                let count = ((area / DENSE_AREA) as usize).clamp(DENSE_SAMPLES, MAX_DENSE_SAMPLES);
                surface_samples(meshes, count, rng)
            }
        };
        let medial = surface.is_some();
        let solids = solids(meshes);
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
        // The cells' centers along each axis.
        let along = |axis: usize| -> Vec<f32> {
            let cells = 0..n[axis];
            cells.map(|i| lo[axis] + (i as f32 + 0.5) * voxel).collect()
        };
        let (xs, ys, zs) = (along(0), along(1), along(2));
        // Where the rows of cells cross each closed solid, for what winds around each cell.
        let crossed = |s: &Solid| {
            s.closed
                .then(|| Crossings::new(s.mesh.vertices(), s.mesh.indices(), &ys, &zs, parallel))
        };
        let crossings: Vec<Option<Crossings>> = solids.iter().map(crossed).collect();
        // A row's cells that are inside a solid, each with the largest sphere inside one: only
        // those are measured to the surface, which is most of what a cell costs.
        let (xs, crossings) = (xs.as_slice(), crossings.as_slice());
        let row = |row: u32| -> Vec<(Vec3, f32)> {
            let (y, z) = (ys[(row % n.y) as usize], zs[(row / n.y) as usize]);
            let numbers = crossings
                .iter()
                .map(|c| c.as_ref().map(|c| c.numbers(row as usize, xs)));
            let mut wound: Vec<_> = numbers.collect();
            let cell = |&x: &f32| {
                let p = Vec3::new(x, y, z);
                let radius = |(s, wound): (&Solid, &mut Option<_>)| {
                    let inside = match wound {
                        Some(numbers) => Iterator::next(numbers) != Some(0),
                        None => true,
                    };
                    if !inside {
                        return None;
                    }
                    let proj = s.mesh.project_local_point(p, false);
                    (s.closed || proj.is_inside).then(|| (p - proj.point).length())
                };
                let radii = solids.iter().zip(&mut wound).filter_map(radius);
                radii.reduce(f32::max).map(|r| (p, r))
            };
            // Room for the whole row at once: a box's cells are all inside it.
            let mut inside = Vec::with_capacity(xs.len());
            inside.extend(xs.iter().filter_map(cell));
            inside
        };
        let rows = 0..n.y * n.z;
        let mut candidates: Vec<(Vec3, f32)> = if parallel {
            rows.into_par_iter().flat_map_iter(row).collect()
        } else {
            rows.flat_map(row).collect()
        };
        if candidates.is_empty() {
            // Open or degenerate geometry has no interior; grow spheres from the surface instead.
            candidates = samples.iter().map(|&p| (p, 0.0)).collect();
        }
        let largest = candidates.iter().map(|c| c.1).fold(0.0, f32::max);
        let stride = candidates.len().div_ceil(MAX_CANDIDATES).max(1);
        let mut fit = Self {
            samples,
            normals,
            joined: vec![false; dense.len()],
            dense,
            candidates: candidates.into_iter().step_by(stride).collect(),
            solids,
            largest,
            parallel,
            medial,
        };
        if medial {
            // No ball inside the geometry is wider than its bounds' narrowest side.
            let widest = 0.5 * extent.min_element();
            let ball = |(&p, &n): (&Vec3, &Vec3)| fit.medial(p, n, widest);
            let balls: Vec<(Vec3, f32)> = if parallel {
                let samples = fit.samples.par_iter().zip(&fit.normals);
                samples.filter_map(ball).collect()
            } else {
                fit.samples
                    .iter()
                    .zip(&fit.normals)
                    .filter_map(ball)
                    .collect()
            };
            fit.candidates.extend(balls);
        }
        Some(fit)
    }

    /// How `spheres` (`[x, y, z, radius]`, in the meshes' frame) fit this geometry: their largest
    /// overhang, the share of the dense samples outside every one and how far outside the
    /// farthest lies, and the geometry's thickness.
    fn measure(&self, link: &str, spheres: &[[f32; 4]]) -> LinkQualification {
        let overhang = overhang(&self.solids, spheres);
        let inside = |p: Vec3| {
            spheres
                .iter()
                .any(|s| p.distance(Vec3::new(s[0], s[1], s[2])) <= s[3] * (1.0 + 1e-5))
        };
        // How far `p` lies from the nearest sphere's surface.
        let beyond = |p: Vec3| {
            spheres
                .iter()
                .map(|s| p.distance(Vec3::new(s[0], s[1], s[2])) - s[3])
                .fold(f32::INFINITY, f32::min)
        };
        let outside = self.dense.iter().filter(|(p, _)| !inside(*p));
        let (count, gap) = outside.fold((0, 0.0f32), |(count, gap), &(p, _)| {
            (count + 1, gap.max(beyond(p)))
        });
        LinkQualification {
            link: link.to_string(),
            spheres: spheres.len(),
            overhang,
            uncovered: count as f32 / self.dense.len().max(1) as f32,
            gap,
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
            // Padded past rounding, so the window keeps every sample `reaches` accepts: a sample
            // left out would get its own candidate again every refinement.
            let reach = r + overhang;
            let pad = 4.0 * f32::EPSILON * (c.x.abs() + reach);
            let first = xs.partition_point(|&x| x < c.x - reach - pad);
            let last = xs.partition_point(|&x| x <= c.x + reach + pad);
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
        if self.medial {
            // A later pick can reach all an earlier one did: drop each pick, in the order they
            // were made, whose every sample another pick reaches.
            let reached_by = |pick: usize| {
                let words = cover[pick].iter().enumerate();
                words.flat_map(|(w, &word)| {
                    let bits = std::iter::successors((word != 0).then_some(word), |b| {
                        let rest = b & (b - 1);
                        (rest != 0).then_some(rest)
                    });
                    bits.map(move |b| 64 * w + b.trailing_zeros() as usize)
                })
            };
            let mut count = vec![0u32; self.samples.len()];
            for &pick in &picks {
                for i in reached_by(pick) {
                    count[i] += 1;
                }
            }
            picks.retain(|&pick| {
                let redundant = reached_by(pick).all(|i| count[i] > 1);
                if redundant {
                    for i in reached_by(pick) {
                        count[i] -= 1;
                    }
                }
                !redundant
            });
        }
        let left = (0..self.samples.len())
            .filter(|&i| uncovered[i / 64] & (1 << (i % 64)) != 0)
            .collect();
        (picks, left)
    }

    /// A greedy cover at `overhang` that reaches every dense sample within `growth` times
    /// `overhang`, refined until it does: samples no candidate reaches get candidates of their own
    /// ([`LinkFit::tangent`]), and dense samples the cover leaves farther join the samples. Each
    /// round adds candidates that reach samples none reached, or joins dense samples, so it ends.
    fn refined_cover(&mut self, overhang: f32, growth: f32) -> Vec<usize> {
        let any_reaches =
            |spheres: &[(Vec3, f32)], p: Vec3| spheres.iter().any(|&s| reaches(p, s, overhang));
        loop {
            let (picks, left) = self.greedy(overhang, usize::MAX);
            // One candidate reaches many neighbouring samples: those an added one reaches get
            // none.
            let mut added: Vec<(Vec3, f32)> = vec![];
            for i in left {
                let (p, n) = (self.samples[i], self.normals[i]);
                if any_reaches(&added, p) {
                    continue;
                }
                // Where no sphere inside reaches it, as at a vertex whose averaged normal leads out
                // of a part thinner than `MIN_TANGENT_DEPTH`, a cover that must hold its samples
                // centers one on it, so growing another to it never reaches past the overhang.
                let own = self.tangent(p, n, overhang);
                if let Some(candidate) = own.or_else(|| self.medial.then_some((p, 0.0))) {
                    added.push(candidate);
                }
            }
            if !added.is_empty() {
                self.candidates.extend(added);
                continue;
            }
            // Dense samples that joined and are still missed are those no candidate reaches.
            let cover = SphereCover::new(&self.candidates, &picks, growth * overhang);
            let missed = |&i: &usize| {
                let p = self.dense[i].0;
                !self.joined[i] && !cover.contains(p)
            };
            let dense = 0..self.dense.len();
            let missed: Vec<usize> = if self.parallel {
                dense.into_par_iter().filter(missed).collect()
            } else {
                dense.filter(missed).collect()
            };
            if missed.is_empty() {
                return picks;
            }
            for i in missed {
                let (p, n) = self.dense[i];
                self.samples.push(p);
                self.normals.push(n);
                self.joined[i] = true;
            }
        }
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

    /// The medial ball at surface point `p`: the largest inside the geometry that touches `p`,
    /// centered along its inward normal `-n` and no wider than `widest`, found by halving the
    /// interval of its radius down to `MIN_TANGENT_DEPTH` (the shrinking ball of Ma, Bae and Choi).
    /// Its center lies on the medial axis, where a sphere reaches the most surface for its
    /// overhang: the middle of a slab, which covers both its faces, or the bisector of an edge.
    /// Returned as its center and the depth of that center. `None` where `n` is zero.
    fn medial(&self, p: Vec3, n: Vec3, widest: f32) -> Option<(Vec3, f32)> {
        if n == Vec3::ZERO {
            return None;
        }
        // Whether the ball of radius `r` along the normal touches the surface at `p` alone.
        let inside = |r: f32| {
            let pad = ROUNDING * (p.abs().max_element() + r);
            inscribed(&self.solids, p - n * r).is_some_and(|(_, depth)| depth + pad >= r)
        };
        let (mut lo, mut hi) = (0.0f32, widest);
        while hi - lo > MIN_TANGENT_DEPTH {
            let mid = 0.5 * (lo + hi);
            if inside(mid) { lo = mid } else { hi = mid }
        }
        inscribed(&self.solids, p - n * lo).filter(|ball| ball.1 > MIN_TANGENT_DEPTH)
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

    /// Holds fitted `spheres` to `limit` past the geometry ([`SphereOptions::max_overhang`]): each
    /// radius is cut back to its center's depth plus the limit, and a sphere left with no more
    /// than rounding gives a center on the surface is dropped, so the samples they reached past
    /// that stay outside. Returns how far past the geometry the farthest still reaches. Out of
    /// line: inlined, it grew `fit` by two fifths, and fits without a limit that search for the
    /// least growth took 7 to 12% longer on x86.
    #[inline(never)]
    fn hold(&self, spheres: &mut Vec<[f32; 4]>, limit: f32) -> f32 {
        let corner = |far: f32, solid: &Solid| {
            let bounds = solid.mesh.local_aabb();
            far.max(bounds.mins.abs().max(bounds.maxs.abs()).max_element())
        };
        let least = ROUNDING * self.solids.iter().fold(0.0, corner);
        let depths = Depths::new(&self.solids);
        let mut reach = 0.0f32;
        spheres.retain_mut(|sphere| {
            let depth = depths.at(Vec3::new(sphere[0], sphere[1], sphere[2]));
            sphere[3] = sphere[3].min(depth + limit);
            let kept = sphere[3] > least;
            if kept {
                reach = reach.max(sphere[3] - depth);
            }
            kept
        });
        reach
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
    /// How far the farthest of those samples lies outside the spheres (meters), zero where none
    /// does: how deep an obstacle can reach into the link unseen. Up to about a millimeter for
    /// spheres fitted to an arm's links, and two or three on a stand's wide boxes; more where a
    /// hand-tuned model leaves gaps, or where [`SphereOptions::max_overhang`] holds spheres
    /// short of the surface, which leaves all of it outside them at zero.
    pub gap: f32,
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
        /* surface */ None,
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
            let (picks, left) = f.greedy(tolerance * scale, (o.budget - total).saturating_add(1));
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
    /// The largest of each link's overhang over its tolerance.
    pub(crate) relative_overhang: f32,
    /// What every link's tolerance was scaled by for the budget: 1 where it reached them.
    pub(crate) scale: f32,
}

/// Spheres for each link's meshes (in the link frame), at most `o.budget` in total. Where the
/// budget reaches every tolerance (`tolerances`, per link), each closed link's overhang is at most
/// its tolerance times the least growth, from 1 to `BOUND_GROWTH`, the budget affords; otherwise at
/// most twice its tolerance scaled up by the smallest factor the budget can achieve; but for the
/// parts the module doc names. A budget of `usize::MAX` reaches every tolerance: the covers pick
/// as many spheres as the tolerances take, at most one per sample. With `surface`, one link's
/// spheres contain those points of its surface in place of random samples, and its candidates
/// take in its samples' medial balls ([`LinkFit::new`]). With `parallel`, the work runs
/// on the current rayon pool; without, on the calling thread alone, without registering it with
/// any pool.
pub(crate) fn fit(
    links: &[Vec<TriMesh>],
    tolerances: &[f32],
    o: &SphereOptions,
    parallel: bool,
    surface: Option<&[(Vec3, Vec3)]>,
) -> Fitted {
    debug_assert!(o.check().is_ok(), "callers check the options");
    debug_assert_eq!(links.len(), tolerances.len());
    debug_assert!(
        surface.is_none() || links.len() == 1,
        "a surface is one link's"
    );
    let mut rng = Rng::new(o.rng_seed);
    let mut fits: Vec<Option<LinkFit>> = links
        .iter()
        .map(|m| LinkFit::new(m, o, &mut rng, parallel, surface))
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
    for _ in 0..MAX_SEARCHES {
        let covers = cover(&mut fits, scale, BOUND_GROWTH);
        if total(&covers) <= o.budget {
            picks = Some(covers);
            break;
        }
        scale = search_scale(&fits, tolerances, o, Some(scale));
    }
    // Past the searches, the covers the last one counted, which keep to the budget.
    let picks = picks.unwrap_or_else(|| {
        fits.iter()
            .zip(tolerances)
            .map(|(f, &tolerance)| {
                f.as_ref()
                    .map_or(vec![], |f| f.greedy(tolerance * scale, usize::MAX).0)
            })
            .collect()
    });
    let grown = |fits: &[Option<LinkFit>], picks: &[Vec<usize>]| -> Vec<(Vec<[f32; 4]>, f32)> {
        fits.iter()
            .zip(picks)
            .map(|(f, picks)| f.as_ref().map_or((vec![], 0.0), |f| f.spheres(picks)))
            .collect()
    };
    // The largest of each link's growth over its tolerance.
    let relative = |fitted: &[(Vec<[f32; 4]>, f32)]| {
        fitted
            .iter()
            .zip(tolerances)
            .map(|(f, &t)| f.1 / t)
            .fold(0.0, f32::max)
    };
    // Where the budget reaches the tolerances, the least growth it affords, down to none, below
    // what the covers in hand reach. Each try refines its own copy of those covers, so one past the
    // budget leaves no samples behind.
    let mut fitted = None;
    if scale == 1.0 {
        let within = |growth: f32| {
            let mut trial = fits.clone();
            let covers = cover(&mut trial, 1.0, growth);
            (total(&covers) <= o.budget).then_some((trial, covers))
        };
        let mut best = within(1.0);
        if best.is_none() {
            let loose = grown(&fits, &picks);
            let (mut lo, mut hi) = (1.0, relative(&loose));
            while hi - lo > GROWTH_PRECISION {
                let mid = 0.5 * (lo + hi);
                match within(mid) {
                    Some(t) => (best, hi) = (Some(t), mid),
                    None => lo = mid,
                }
            }
            if best.is_none() {
                fitted = Some(loose);
            }
        }
        if let Some((trial, covers)) = best {
            fitted = Some(grown(&trial, &covers));
        }
    }
    let mut fitted = fitted.unwrap_or_else(|| grown(&fits, &picks));
    if let Some(limit) = o.max_overhang {
        // The tries above share each link's geometry with `fits`.
        for (f, (spheres, reach)) in fits.iter().zip(&mut fitted) {
            if let Some(f) = f {
                *reach = f.hold(spheres, limit);
            }
        }
    }
    Fitted {
        overhang: fitted.iter().map(|f| f.1).fold(0.0, f32::max),
        relative_overhang: relative(&fitted),
        spheres: fitted.into_iter().map(|f| f.0).collect(),
        scale,
    }
}

/// Spheres for an attached object's `shapes` (in its link's frame) without fitting, where every
/// shape can be covered directly: a sphere is itself; a box splits into a grid of cells, each
/// covered by the sphere through its corners; a capsule is spheres along its axis; and a
/// cylinder that is thin or slender, a disc or a rod, is one layer of rings of cells or one line
/// of spheres where that reaches `o.tolerance` within the budget ([`cylinder_cover`]). A stout
/// cylinder is left to the fit, which spends fewer spheres on its inside; so is a distance grid. A
/// box's sphere reaches past its faces by its radius less the cell's smallest half size, so of the
/// covers within the tolerance the one of fewest spheres is taken; so for a capsule. The shapes but
/// the spheres share the budget the spheres leave, and where it has no cover of a shape within the
/// tolerance, the shapes are left to the fit too. Microseconds, where fitting takes milliseconds,
/// and within the tolerance on a thin wide shape in fewer spheres than a fit. Under
/// `o.max_overhang` the covers are those within the smaller of it and the tolerance, and where the
/// budget has none within it, the shapes are left to the fit, which holds its spheres to it.
pub(crate) fn cover_primitives(shapes: &[Obstacle], o: &SphereOptions) -> Option<Vec<[f32; 4]>> {
    if shapes.iter().any(|s| matches!(s, Obstacle::Sdf { .. })) {
        return None;
    }
    // Under a limit, a cover reaches no farther than it, or the shapes are left to the fit.
    let limit = o.max_overhang.unwrap_or(f32::INFINITY);
    let tolerance = o.tolerance.min(limit);
    let balls = shapes
        .iter()
        .filter(|s| matches!(s, Obstacle::Sphere { .. }))
        .count();
    let rest = shapes.len() - balls;
    let each = o.budget.checked_sub(balls)?.checked_div(rest).unwrap_or(0);
    if rest > 0 && each == 0 {
        return None;
    }
    let mut spheres = vec![];
    // A shape's spheres, given about its own axes, as it is placed.
    let placed = |spheres: &mut Vec<[f32; 4]>, center: Vec3, rotation: Quat, own: Vec<[f32; 4]>| {
        let rotation = world::unit(rotation);
        for [x, y, z, radius] in own {
            let p = center + rotation * Vec3::new(x, y, z);
            spheres.push([p.x, p.y, p.z, radius]);
        }
    };
    for shape in shapes {
        match *shape {
            Obstacle::Sphere { center, radius } => {
                spheres.push([center.x, center.y, center.z, radius]);
            }
            Obstacle::Cuboid {
                center,
                half_extents,
                rotation,
            } => {
                let cells = box_cells(half_extents, tolerance, each)?;
                let cell = half_extents / cells.as_vec3();
                // A hair wide, so rounding never leaves a corner uncovered.
                let radius = cell.length() * HAIR;
                if radius - cell.min_element() > limit {
                    return None;
                }
                let mut own = vec![];
                for i in 0..cells.x {
                    for j in 0..cells.y {
                        for k in 0..cells.z {
                            let at = (UVec3::new(2 * i + 1, 2 * j + 1, 2 * k + 1).as_vec3() * cell)
                                - half_extents;
                            own.push([at.x, at.y, at.z, radius]);
                        }
                    }
                }
                placed(&mut spheres, center, rotation, own);
            }
            Obstacle::Cylinder {
                center,
                rotation,
                radius,
                half_height,
            } => {
                let own = cylinder_cover(radius, half_height, tolerance, each)?;
                placed(&mut spheres, center, rotation, own);
            }
            Obstacle::Capsule {
                center,
                rotation,
                radius,
                half_length,
            } => {
                let own = capsule_cover(radius, half_length, tolerance, each)?;
                if own[0][3] - radius > limit {
                    return None;
                }
                placed(&mut spheres, center, rotation, own);
            }
            Obstacle::Sdf { .. } => unreachable!("a distance grid returns above"),
        }
    }
    Some(spheres)
}

/// What a covering sphere's radius is widened by, so rounding never leaves a corner uncovered.
const HAIR: f32 = 1.0 + 1e-6;

/// Spheres along a capsule's axis (z) that cover it: the fewest that reach no more than
/// `tolerance` past it, `None` where more than `budget` would. Evenly spaced from end to end,
/// each wide enough to reach the capsule's side halfway to the next.
fn capsule_cover(
    radius: f32,
    half_length: f32,
    tolerance: f32,
    budget: usize,
) -> Option<Vec<[f32; 4]>> {
    if half_length <= 0.0 {
        return Some(vec![[0.0, 0.0, 0.0, radius]]);
    }
    // With `count` spheres: half the spacing between two, and how far each reaches past the side.
    let half_gap = |count: usize| half_length / (count - 1).max(1) as f32;
    let covering = |count: usize| match count {
        1 => half_length + radius,
        _ => radius.hypot(half_gap(count)) * HAIR,
    };
    let count = (1..=budget).find(|&count| covering(count) - radius <= tolerance)?;
    if count == 1 {
        return Some(vec![[0.0, 0.0, 0.0, covering(1)]]);
    }
    let spheres = (0..count).map(|k| {
        let z = -half_length + 2.0 * half_gap(count) * k as f32;
        [0.0, 0.0, z, covering(count)]
    });
    Some(spheres.collect())
}

/// How wide a cylinder's cover makes its middle disc, in ring widths: a disc of that radius
/// needs a sphere about as large as a ring's cells do.
const MIDDLE: f32 = 0.75;

/// Spheres that cover a thin or a slender cylinder about its z axis and reach no more than
/// `tolerance` past it, the fewest of up to `budget`: one layer of a middle disc and rings of
/// equal width around it, each ring cut into equal cells, or one line of spheres along the axis.
/// A cell is covered by a sphere at the cylinder's middle height, on the cell's middle line,
/// through whichever of its corners are farthest; where along that line, toward the axis or the
/// rim, is what leaves the sphere reaching least past the cylinder's faces and side. Rings, and
/// how many cells a ring is cut into (one to three times what makes them square), are tried
/// over. `None` where neither reaches the tolerance within the budget: a stout cylinder needs
/// spheres of several sizes, which the fit finds.
fn cylinder_cover(
    radius: f32,
    half_height: f32,
    tolerance: f32,
    budget: usize,
) -> Option<Vec<[f32; 4]>> {
    // A line of `count` spheres along the axis, each through the rim of its slice.
    let line = |count: usize| {
        let half = half_height / count as f32;
        let sphere = radius.hypot(half) * HAIR;
        let past = (sphere - half).max(sphere - radius);
        let spheres = (0..count)
            .map(|k| [0.0, 0.0, -half_height + (2 * k + 1) as f32 * half, sphere])
            .collect::<Vec<_>>();
        (past, spheres)
    };
    // One layer's spheres as (distance from the axis, cells around it, radius), and how far
    // they reach past the cylinder, for `rings` rings (the middle disc one of them) whose cells
    // are `fine` times as many as would make them square.
    let layer = |rings: usize, fine: f32| {
        let half = half_height;
        // Past the faces, and past the side.
        let past = |from_axis: f32, sphere: f32| (sphere - half).max(from_axis + sphere - radius);
        let width = radius / (rings as f32 - 1.0 + MIDDLE);
        let middle = MIDDLE * width;
        let sphere = middle.hypot(half) * HAIR;
        let mut layer = vec![(0.0, 1, sphere)];
        let mut overhang = past(0.0, sphere);
        for ring in 1..rings {
            let inner = middle + (ring - 1) as f32 * width;
            let outer = if ring + 1 == rings {
                radius
            } else {
                inner + width
            };
            let around = (fine * TAU * (inner + outer) / 2.0 / width).ceil().max(3.0);
            let half_angle = PI / around;
            // The sphere through the cell's farthest corners, from `at` along its middle line.
            let through = |at: f32| {
                let to =
                    |corner: f32| at * at + corner * corner - 2.0 * at * corner * half_angle.cos();
                (to(inner).max(to(outer)) + half * half).sqrt() * HAIR
            };
            let steps = 32;
            let best = (0..=steps)
                .map(|k| inner + (outer - inner) * k as f32 / steps as f32)
                .map(|at| (at, through(at)))
                .min_by(|a, b| past(a.0, a.1).total_cmp(&past(b.0, b.1)))
                .expect("at least one place along the line");
            overhang = overhang.max(past(best.0, best.1));
            layer.push((best.0, around as usize, best.1));
        }
        (overhang, layer)
    };
    // The fewest spheres within the tolerance, of the lines and the layers.
    let lines = (1..=budget).map(line).find(|(past, _)| *past <= tolerance);
    let mut best = lines.map(|(_, spheres)| spheres);
    for rings in 2.. {
        let mut any = false;
        for fine in [1.0, 1.5, 2.0, 3.0] {
            let (overhang, layer) = layer(rings, fine);
            let count: usize = layer.iter().map(|ring| ring.1).sum();
            if count > budget {
                continue;
            }
            any = true;
            if overhang <= tolerance && best.as_ref().is_none_or(|b| count < b.len()) {
                let cells = layer.iter().flat_map(|&(from_axis, around, sphere)| {
                    (0..around).map(move |cell| {
                        let angle = TAU * cell as f32 / around as f32;
                        [
                            from_axis * angle.cos(),
                            from_axis * angle.sin(),
                            0.0,
                            sphere,
                        ]
                    })
                });
                best = Some(cells.collect());
            }
        }
        if !any {
            break;
        }
    }
    best
}

/// The cells per axis splitting a box of `half` extents for `cover_primitives`: the fewest within
/// `tolerance` of overhang and `budget` spheres, the least overhang of those, `None` where the
/// budget has none.
fn box_cells(half: Vec3, tolerance: f32, budget: usize) -> Option<UVec3> {
    let mut best: Option<(usize, f32, UVec3)> = None;
    for x in 1..=budget {
        for y in 1..=budget / x {
            for z in 1..=budget / (x * y) {
                let cells = UVec3::new(x as u32, y as u32, z as u32);
                let cell = half / cells.as_vec3();
                let overhang = cell.length() - cell.min_element();
                let candidate = (x * y * z, overhang, cells);
                let better = |b: &(usize, f32, UVec3)| (candidate.0, candidate.1) < (b.0, b.1);
                if overhang <= tolerance && best.as_ref().is_none_or(better) {
                    best = Some(candidate);
                }
            }
        }
    }
    best.map(|b| b.2)
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

// What a fit runs once for a link, or not at all without a limit, is kept here, after the
// functions it spends its time in: set among them, it took fits 1 to 8% longer in a one-unit
// build on x86.

/// One mesh of a link's geometry.
struct Solid {
    /// For distances to its surface, and the side of its nearest face.
    mesh: ParryMesh,
    /// Whether its surface is closed: every edge borders as many triangles that run along it one
    /// way as the other. What it winds around is then its inside (`winding`), shells that overlap
    /// their union. A mesh with open edges has none, and the side of its nearest face stands in.
    closed: bool,
}

/// `meshes` as points are tested against them, without those parry cannot build.
fn solids(meshes: &[TriMesh]) -> Arc<[Solid]> {
    let flags = TriMeshFlags::MERGE_DUPLICATE_VERTICES
        | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES
        | TriMeshFlags::ORIENTED;
    let solid = |m: &TriMesh| {
        let mesh = ParryMesh::with_flags(m.vertices.clone(), m.triangles.clone(), flags).ok()?;
        // Each edge as its two vertices, lower first, by which way a triangle runs along it.
        let (mut forward, mut backward) = (vec![], vec![]);
        for &[a, b, c] in mesh.indices() {
            for (from, to) in [(a, b), (b, c), (c, a)] {
                let edge = u64::from(from.min(to)) << 32 | u64::from(from.max(to));
                if from < to {
                    forward.push(edge);
                } else {
                    backward.push(edge);
                }
            }
        }
        forward.sort_unstable();
        backward.sort_unstable();
        Some(Solid {
            closed: forward == backward,
            mesh,
        })
    };
    meshes.iter().filter_map(solid).collect()
}

/// How far the farthest of `spheres` reaches beyond `solids`: its radius less its center's depth.
/// Out of line, so that [`LinkFit::measure`]'s loops over the samples are compiled as they were
/// without it: with the depths set up beside them they took 5 to 11% longer on an M4 Pro, in
/// fewer instructions.
#[inline(never)]
fn overhang(solids: &[Solid], spheres: &[[f32; 4]]) -> f32 {
    let depths = Depths::new(solids);
    spheres
        .iter()
        .map(|s| s[3] - depths.at(Vec3::new(s[0], s[1], s[2])))
        .fold(0.0, f32::max)
}

/// How deep points lie inside a link's solids, one at a time.
struct Depths<'a> {
    solids: &'a [Solid],
    /// Per solid that is closed, its surface, for what winds around a point.
    surfaces: Vec<Option<Surface<'a>>>,
    /// The corners of the box around the solids.
    bounds: (Vec3, Vec3),
}

impl<'a> Depths<'a> {
    fn new(solids: &'a [Solid]) -> Self {
        let surface = |s: &'a Solid| {
            s.closed
                .then(|| Surface::new(s.mesh.vertices(), s.mesh.indices()))
        };
        let around = |(lo, hi): (Vec3, Vec3), s: &Solid| {
            let bounds = s.mesh.local_aabb();
            (lo.min(bounds.mins), hi.max(bounds.maxs))
        };
        Self {
            solids,
            surfaces: solids.iter().map(surface).collect(),
            bounds: solids
                .iter()
                .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), around),
        }
    }

    /// How deep `p` lies inside the solids (meters): the radius of the largest sphere centered
    /// there that crosses no face of a solid it is inside, or where it is outside them all, its
    /// distance to the nearest, negative. Inside a closed solid is where its surface winds around
    /// the point, and inside one with open edges the inner side of its nearest face.
    fn at(&self, p: Vec3) -> f32 {
        use parry3d::query::PointQuery;

        let (mut deepest, mut nearest) = (None::<f32>, f32::INFINITY);
        for (s, surface) in self.solids.iter().zip(&self.surfaces) {
            let proj = s.mesh.project_local_point(p, false);
            let to_surface = (p - proj.point).length();
            let wound = |surface: &Surface| surface.number(p) != 0;
            if surface.as_ref().map_or(proj.is_inside, wound) {
                deepest = Some(deepest.map_or(to_surface, |d| d.max(to_surface)));
            }
            nearest = nearest.min(to_surface);
        }
        // A point is no nearer the solids than their box: far enough away, the distances to
        // their faces lose every digit.
        let (lo, hi) = self.bounds;
        let to_box = (lo - p).max(p - hi).max(Vec3::ZERO).length();
        deepest.unwrap_or_else(|| -nearest.max(to_box))
    }
}

/// Why spheres cannot be held within a distance of `meshes` ([`SphereOptions::max_overhang`]), if
/// they cannot. A mesh that is not closed has no inside for a sphere's depth to be measured in:
/// closed, as [`Solid::closed`], is every edge bordering as many triangles that run along it one
/// way as the other, once vertices at one place are merged and triangles that name a vertex
/// twice are dropped. And a shell that faces inward, a cavity's among them, is refused: a sample
/// on it has its own candidate by the side of its face ([`LinkFit::tangent`]), the wrong one, so
/// where it is thinner than the grid it would be left without spheres.
pub(crate) fn closed(meshes: &[TriMesh]) -> Result<(), String> {
    let flags = TriMeshFlags::MERGE_DUPLICATE_VERTICES | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES;
    for (k, mesh) in meshes.iter().enumerate() {
        let of = format!("mesh {k} of {}", meshes.len());
        let solid = ParryMesh::with_flags(mesh.vertices.clone(), mesh.triangles.clone(), flags)
            .map_err(|e| format!("{of} has no surface: {e}"))?;
        let faces = solid.indices();
        if faces.is_empty() {
            return Err(format!("{of} has no surface"));
        }
        // Per edge, the triangles that run along it from its lower vertex less those that run
        // back, and one of its triangles, to join shells by.
        let mut edges: BTreeMap<(u32, u32), (i32, usize)> = BTreeMap::new();
        // Each triangle's shell, as the lowest triangle joined to it across edges so far.
        let mut shell: Vec<usize> = (0..faces.len()).collect();
        let first = |shell: &[usize], mut t: usize| {
            while shell[t] != t {
                t = shell[t];
            }
            t
        };
        for (t, &[a, b, c]) in faces.iter().enumerate() {
            for (from, to) in [(a, b), (b, c), (c, a)] {
                let edge = edges.entry((from.min(to), from.max(to))).or_insert((0, t));
                edge.0 += if from < to { 1 } else { -1 };
                let (ours, theirs) = (first(&shell, t), first(&shell, edge.1));
                shell[ours.max(theirs)] = ours.min(theirs);
            }
        }
        let open = edges.values().filter(|edge| edge.0 != 0).count();
        if open > 0 {
            return Err(format!(
                "{of} is not closed: {open} of its {} edges border more triangles that run \
                 along them one way than the other",
                edges.len()
            ));
        }
        // Six times each shell's volume, negative where it faces inward.
        let mut volumes: BTreeMap<usize, f64> = BTreeMap::new();
        for (t, face) in faces.iter().enumerate() {
            let [a, b, c] = face.map(|i| solid.vertices()[i as usize].as_dvec3());
            *volumes.entry(first(&shell, t)).or_default() += a.dot(b.cross(c));
        }
        let inward = volumes.values().filter(|&&volume| volume < 0.0).count();
        if inward > 0 {
            return Err(format!(
                "{of} has {inward} of its {} shells facing inward, which get their spheres from \
                 the wrong side where they are thin",
                volumes.len()
            ));
        }
    }
    Ok(())
}

/// What rounding moves a depth measured in `f32` by, in units of the largest coordinate it is
/// measured from: a point's difference from the surface loses a unit in the last place per axis,
/// and a sphere placed by a pose a few more, so eight in all.
const ROUNDING: f32 = 8.0 * f32::EPSILON;

/// Why `spheres` (`[x, y, z, radius]`, in the frame of `meshes`, which are [`closed`]) do not all
/// keep within `max_overhang` of them, if one does not: the first that reaches farther.
pub(crate) fn within(
    meshes: &[TriMesh],
    spheres: &[[f32; 4]],
    max_overhang: f32,
) -> Result<(), String> {
    let solids = solids(meshes);
    let depths = Depths::new(&solids);
    let largest = meshes
        .iter()
        .flat_map(|m| &m.vertices)
        .fold(0.0, |largest: f32, v| largest.max(v.abs().max_element()));
    for (k, &[x, y, z, radius]) in spheres.iter().enumerate() {
        let center = Vec3::new(x, y, z);
        let depth = depths.at(center);
        let slack = ROUNDING * largest.max(center.abs().max_element() + radius);
        let reach = radius - depth;
        if reach.is_nan() || reach > max_overhang + slack {
            let lies = if depth < 0.0 {
                format!("its center {:.3} mm outside", -depth * 1e3)
            } else {
                format!("its center {:.3} mm deep", depth * 1e3)
            };
            return Err(format!(
                "sphere {k}, at {center} with radius {radius}, reaches {:.3} mm past the \
                 geometry, {lies}: `max_overhang` allows {:.3} mm",
                reach * 1e3,
                max_overhang * 1e3
            ));
        }
    }
    Ok(())
}

/// Spheres for an attached object's `shapes` (in its link's frame) by their surfaces' tiles, as
/// many as the tolerance takes, or the smaller of it and `o.max_overhang`: a thin cylinder's layer
/// ([`cylinder_layer`]), a capsule's line of spheres, a ball itself. Each holds its shape by
/// construction, a sphere through the corners of each tile. `None` where a shape has none: a box
/// or a stout cylinder, which the fit covers, or a distance grid.
pub(crate) fn cover_tiles(shapes: &[Obstacle], o: &SphereOptions) -> Option<Vec<[f32; 4]>> {
    let tolerance = o.tolerance.min(o.max_overhang.unwrap_or(f32::INFINITY));
    let mut spheres = vec![];
    for shape in shapes {
        let (center, rotation, own) = match *shape {
            Obstacle::Sphere { center, radius } => {
                (center, Quat::IDENTITY, vec![[0.0, 0.0, 0.0, radius]])
            }
            Obstacle::Cuboid {
                center,
                half_extents,
                rotation,
            } => (center, rotation, box_cover(half_extents, tolerance)?),
            Obstacle::Sdf { .. } => return None,
            Obstacle::Cylinder {
                center,
                rotation,
                radius,
                half_height,
            } => (
                center,
                rotation,
                cylinder_layer(radius, half_height, tolerance)?,
            ),
            Obstacle::Capsule {
                center,
                rotation,
                radius,
                half_length,
            } => {
                let own = capsule_cover(radius, half_length, tolerance, usize::MAX)?;
                (center, rotation, own)
            }
        };
        let rotation = world::unit(rotation);
        for [x, y, z, radius] in own {
            let p = center + rotation * Vec3::new(x, y, z);
            spheres.push([p.x, p.y, p.z, radius]);
        }
    }
    Some(spheres)
}

/// Spheres in one layer at a thin cylinder's middle height that cover it about its z axis and
/// reach no more than `tolerance` past it, `None` where it is too thick for any: rings from the
/// rim in, each cut into the fewest equal sectors whose spheres reach the tolerance, each sphere
/// on its sector's middle line through the farthest corners of the sector's tiles on both faces
/// (and on the side, for the outer ring), from where along that line it reaches least past the
/// cylinder. A sector's tiles lie inside a ball that holds their corners and is centered on its
/// middle line, the farthest point of a ring's sector or of the side's patch being a corner. The
/// outer ring is as wide as takes fewest spheres for the area it covers, and so is each ring
/// inside it, of widths in half tolerances; the middle one is a disc. A sphere at the middle
/// height reaches the side at its faces only where the cylinder is at most `2 (1 + sqrt(2))`
/// tolerances thick.
fn cylinder_layer(radius: f32, half_height: f32, tolerance: f32) -> Option<Vec<[f32; 4]>> {
    (tolerance > 0.0).then_some(())?;
    let h = half_height;
    // A ring's sphere from `inner` to `outer` in `around` sectors: where along the middle line,
    // its radius, and how far it reaches past the cylinder, at least.
    let sphere = |inner: f32, outer: f32, around: u32| {
        let half = PI / around as f32;
        let farthest = |at: f32| {
            let to = |r: f32| r * r + at * at - 2.0 * r * at * half.cos();
            (to(inner).max(to(outer)) + h * h).sqrt()
        };
        let past = |at: f32| farthest(at) - h.min(radius - at);
        let (mut low, mut high) = (0.0f32, outer);
        for _ in 0..48 {
            let (a, b) = (low + (high - low) / 3.0, high - (high - low) / 3.0);
            if past(a) <= past(b) {
                high = b;
            } else {
                low = a;
            }
        }
        let at = if around == 1 { 0.0 } else { 0.5 * (low + high) };
        (at, farthest(at), past(at))
    };
    // The fewest sectors of a ring whose spheres reach the tolerance; more never reach farther.
    let sectors = |inner: f32, outer: f32| {
        let most = ((TAU * outer / (0.25 * tolerance)).ceil() as u32).max(1);
        if sphere(inner, outer, most).2 > tolerance {
            return None;
        }
        let (mut low, mut high) = (1, most);
        while low < high {
            let mid = (low + high) / 2;
            if sphere(inner, outer, mid).2 <= tolerance {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        Some(low)
    };
    // From `outer` in, the ring of fewest sectors for its area, as (inner radius, sectors).
    let ring = |outer: f32| {
        let widths = (1..=12).map(|k| (k as f32 * 0.5 * tolerance).min(outer));
        let options = widths.filter_map(|w| {
            let inner = outer - w;
            let around = sectors(inner, outer)?;
            Some((
                inner,
                around,
                around as f32 / (outer * outer - inner * inner),
            ))
        });
        options.min_by(|a, b| a.2.total_cmp(&b.2))
    };
    let mut rings = vec![];
    let mut outer = radius;
    while outer > 0.0 {
        let (inner, around, _) = ring(outer)?;
        rings.push((inner.max(0.0), outer, around));
        outer = inner;
    }
    let mut spheres = vec![];
    for (inner, outer, around) in rings {
        let (at, _, past) = sphere(inner, outer, around);
        (past <= tolerance).then_some(())?;
        for k in 0..around {
            let middle = TAU * (k as f32 + 0.5) / around as f32;
            let half = PI / around as f32;
            let center = Vec3::new(at * middle.cos(), at * middle.sin(), 0.0);
            // Through the tiles' corners as placed, widened a hair.
            let mut farthest = 0.0f32;
            for r in [inner, outer] {
                for side in [-1.0, 1.0] {
                    let angle = middle + side * half;
                    for z in [-h, h] {
                        let corner = Vec3::new(r * angle.cos(), r * angle.sin(), z);
                        farthest = farthest.max(corner.distance(center));
                    }
                }
            }
            spheres.push([center.x, center.y, center.z, farthest * HAIR]);
        }
    }
    Some(spheres)
}

/// Points on the surfaces of `shapes`, as they are placed, with their outward unit normals, and
/// how far any point of those surfaces lies from the nearest of them at most, with what rounding
/// moves a point of that size: grids whose cells are at most a spacing across along each
/// direction of a face, a ring or a meridian, so every point lies within the spacing over
/// `sqrt(2)` of a corner of its cell. The spacing is the one at which `MAX_DENSE_SAMPLES` points
/// cover the surfaces' area, but no coarser than keeps that distance within half the `tolerance`.
/// A distance grid has no surface here.
pub(crate) fn surface_grid(shapes: &[Obstacle], tolerance: f32) -> (Vec<(Vec3, Vec3)>, f32) {
    use std::f32::consts::{FRAC_1_SQRT_2, FRAC_PI_2};

    /// Cells at most `spacing` across along `length`, one at least.
    fn cells(length: f32, spacing: f32) -> usize {
        ((length / spacing).ceil() as usize).max(1)
    }

    /// Points around a ring of radius `r` about z at height `z`, each with `normal(angle)`, at
    /// most `spacing` apart on a ring of radius `wide`: the widest of the cells it corners, so a
    /// point of them lies within the spacing over `sqrt(2)` of one.
    fn ring(
        out: &mut Vec<(Vec3, Vec3)>,
        spacing: f32,
        (r, wide): (f32, f32),
        z: f32,
        normal: impl Fn(f32) -> Vec3,
    ) {
        let around = if r > 0.0 {
            cells(TAU * wide, spacing).max(3)
        } else {
            1
        };
        for k in 0..around {
            let a = TAU * k as f32 / around as f32;
            out.push((Vec3::new(r * a.cos(), r * a.sin(), z), normal(a)));
        }
    }

    /// The side of a cylinder about z of `radius` from `-half` to `half`.
    fn side(out: &mut Vec<(Vec3, Vec3)>, spacing: f32, radius: f32, half: f32) {
        let along = cells(2.0 * half, spacing);
        for k in 0..=along {
            let z = half * (2.0 * k as f32 / along as f32 - 1.0);
            ring(out, spacing, (radius, radius), z, |a| {
                Vec3::new(a.cos(), a.sin(), 0.0)
            });
        }
    }

    /// A sphere's half about z of `radius` centered at height `z`, from its equator to its pole
    /// on the side of `up`, by its parallels.
    fn cap(out: &mut Vec<(Vec3, Vec3)>, spacing: f32, radius: f32, z: f32, up: f32) {
        let steps = cells(FRAC_PI_2 * radius, spacing);
        let polar = |k: usize| FRAC_PI_2 * k.min(steps) as f32 / steps as f32;
        for k in 0..=steps {
            let (s, c) = polar(k).sin_cos();
            let normal = |a: f32| Vec3::new(s * a.cos(), s * a.sin(), up * c);
            let wide = radius * polar(k + 1).sin();
            ring(
                out,
                spacing,
                (radius * s, wide),
                z + up * radius * c,
                normal,
            );
        }
    }

    let area: f32 = shapes
        .iter()
        .map(|shape| match *shape {
            Obstacle::Sphere { radius, .. } => 2.0 * TAU * radius * radius,
            Obstacle::Cuboid {
                half_extents: h, ..
            } => 8.0 * (h.x * h.y + h.y * h.z + h.z * h.x),
            Obstacle::Cylinder {
                radius,
                half_height,
                ..
            } => TAU * radius * (radius + 2.0 * half_height),
            Obstacle::Capsule {
                radius,
                half_length,
                ..
            } => 2.0 * TAU * radius * (radius + half_length),
            Obstacle::Sdf { .. } => 0.0,
        })
        .sum();
    let spacing = (area / MAX_DENSE_SAMPLES as f32)
        .sqrt()
        .min(0.5 * tolerance / FRAC_1_SQRT_2);
    let mut out = vec![];
    for shape in shapes {
        let mut own = vec![];
        let (center, rotation) = match *shape {
            Obstacle::Sphere { center, radius } => {
                for up in [-1.0, 1.0] {
                    cap(&mut own, spacing, radius, 0.0, up);
                }
                (center, Quat::IDENTITY)
            }
            Obstacle::Cuboid {
                center,
                half_extents: h,
                rotation,
            } => {
                for axis in 0..3 {
                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                    let (nu, nv) = (cells(2.0 * h[u], spacing), cells(2.0 * h[v], spacing));
                    for side in [-1.0, 1.0] {
                        let mut normal = Vec3::ZERO;
                        normal[axis] = side;
                        for i in 0..=nu {
                            for j in 0..=nv {
                                let mut p = normal * h[axis];
                                p[u] = h[u] * (2.0 * i as f32 / nu as f32 - 1.0);
                                p[v] = h[v] * (2.0 * j as f32 / nv as f32 - 1.0);
                                own.push((p, normal));
                            }
                        }
                    }
                }
                (center, rotation)
            }
            Obstacle::Cylinder {
                center,
                rotation,
                radius,
                half_height,
            } => {
                side(&mut own, spacing, radius, half_height);
                for up in [-1.0, 1.0] {
                    let rings = cells(radius, spacing);
                    for k in 0..=rings {
                        let r = |k: usize| radius * k.min(rings) as f32 / rings as f32;
                        let z = up * half_height;
                        ring(&mut own, spacing, (r(k), r(k + 1)), z, |_| Vec3::Z * up);
                    }
                }
                (center, rotation)
            }
            Obstacle::Capsule {
                center,
                rotation,
                radius,
                half_length,
            } => {
                side(&mut own, spacing, radius, half_length);
                for up in [-1.0, 1.0] {
                    cap(&mut own, spacing, radius, up * half_length, up);
                }
                (center, rotation)
            }
            Obstacle::Sdf { .. } => continue,
        };
        let rotation = world::unit(rotation);
        let placed = own
            .into_iter()
            .map(|(p, n)| (center + rotation * p, rotation * n));
        out.extend(placed);
    }
    let extent = out
        .iter()
        .map(|(p, _)| p.abs().max_element())
        .fold(0.0, f32::max);
    (out, spacing * FRAC_1_SQRT_2 + ROUNDING * extent)
}

/// Sphere centers of a box's medial cover in its axes sorted longest first, each with the radius it
/// needs where that is less than its depth and the tolerance allow.
type Centers = Vec<(f64, f64, f64, Option<f64>)>;

/// Deterministic enclosing sphere cover of the axis-aligned box `[-half_extents, half_extents]`
/// whose spheres reach at most `tolerance` past the box: recursive concentric medial shells
/// (corners on the 3-face ray, edges on the 45-degree bisector) closing onto a staggered
/// mid-plane sheet, with exact thin-box, hybrid-slab, slender-rod and single-sphere regimes.
///
/// All layout runs in `f64` for a tolerance shrunk by four times [`ROUNDING`] of the box's size,
/// and each emitted `f32` sphere takes the largest radius that still keeps `radius - depth <=
/// tolerance` in `f32`, absorbing casting and rotation rounding so enclosure holds strictly in
/// `f32`. Returns `None` only when `half_extents` is negative or non-finite, or `tolerance` does
/// not exceed the rounding slack.
fn box_cover(half_extents: Vec3, tolerance: f32) -> Option<Vec<[f32; 4]>> {
    use std::f64::consts::SQRT_2;

    if !half_extents.is_finite() || half_extents.min_element() < 0.0 || tolerance.is_nan() {
        return None;
    }
    if tolerance <= 0.0 {
        return None;
    }
    let mut permutation = [0usize, 1, 2];
    permutation.sort_by(|&i, &j| half_extents[j].total_cmp(&half_extents[i]));
    let sorted_half = [
        f64::from(half_extents[permutation[0]]),
        f64::from(half_extents[permutation[1]]),
        f64::from(half_extents[permutation[2]]),
    ];
    let [a, b, c] = sorted_half;
    let raw_tolerance = f64::from(tolerance);

    // Center casting (<= sqrt(3)/2 ulp), depth subtraction (<= 1 ulp), radius round-down (<= 2
    // ulp) and subsequent pose rotation (`ROUNDING` = 8 ulp) total under 12 ulp of `a + tolerance`;
    // 4 * ROUNDING (32 ulp) leaves strict positive enclosure headroom in `f32`.
    let slack = 4.0 * f64::from(ROUNDING) * (a + raw_tolerance);
    let effective_tolerance = raw_tolerance - slack;
    if effective_tolerance <= 0.0 {
        return None;
    }

    let emit = |centers: Centers| -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity(centers.len());
        for (x, y, z, exact_radius) in centers {
            let local = [
                x.clamp(-a, a) as f32,
                y.clamp(-b, b) as f32,
                z.clamp(-c, c) as f32,
            ];
            let depth = (half_extents[permutation[0]] - local[0].abs())
                .min(half_extents[permutation[1]] - local[1].abs())
                .min(half_extents[permutation[2]] - local[2].abs())
                .max(0.0);
            let mut radius = match exact_radius {
                Some(required) => ((required + 0.5 * slack) as f32).min(depth + tolerance),
                None => depth + tolerance,
            };
            while radius - depth > tolerance {
                radius = radius.next_down();
            }
            let mut unpermuted = [0.0f32; 3];
            for axis in 0..3 {
                unpermuted[permutation[axis]] = local[axis];
            }
            out.push([unpermuted[0], unpermuted[1], unpermuted[2], radius]);
        }
        out
    };

    let face_radius_squared =
        2.0 * effective_tolerance * c + effective_tolerance * effective_tolerance;
    let face_radius = face_radius_squared.sqrt();

    // Regime 0: a single sphere at the origin encloses the whole box.
    let diagonal = (a * a + b * b + c * c).sqrt();
    if diagonal <= c + effective_tolerance {
        return Some(emit(vec![(0.0, 0.0, 0.0, Some(diagonal))]));
    }

    // Regime 1: slender rod — a single line of spheres along the long axis. An end center on the
    // x-axis reaches corner (a, b, c) iff b^2 + c^2 <= face_radius^2.
    if b * b + c * c < face_radius_squared {
        let max_half_step = (face_radius_squared - b * b).sqrt();
        let count = (a / max_half_step).ceil().max(1.0) as usize;
        let half_step = a / count as f64;
        let max_x = a - c;
        let mut centers = Vec::with_capacity(count);
        for k in 0..count {
            let left = -a + (2 * k) as f64 * half_step;
            let right = -a + (2 * k + 2) as f64 * half_step;
            let x = (0.5 * (left + right)).clamp(-max_x, max_x);
            let max_dx = (x - left).abs().max((right - x).abs());
            let required = (max_dx * max_dx + b * b + c * c).sqrt();
            centers.push((x, 0.0, 0.0, Some(required)));
        }
        return Some(emit(centers));
    }

    // Corner medial ray d_x = d_y = d_z = t covers the corner iff 3*t^2 <= (t + tau)^2, giving
    // t <= (sqrt(3) + 1) / 2 * tau.
    let corner_inset = 0.5 * (3.0_f64.sqrt() + 1.0) * effective_tolerance;

    // Regime 2a (thin box, including all c <= tau): every sphere lies on z = 0 with depth c,
    // reducing to a 2D disk cover of [-a, a] x [-b, b] centered in [-(a-c), a-c] x [-(b-c), b-c].
    if c <= corner_inset {
        let sheet = cover_rect_2d_within(a, b, face_radius, a - c, b - c)?;
        let centers = sheet.into_iter().map(|(x, y)| (x, y, 0.0, None)).collect();
        return Some(emit(centers));
    }

    // One shared z = 0 edge row beats two separate edge rows iff its half-reach
    // sqrt(2*tau*c + tau^2 - c^2) exceeds half the single-edge optimum sqrt(2)*tau / 2, i.e.,
    // c < (1 + sqrt(6) / 2) * tau.
    let slab_merge_limit = (1.0 + 0.5 * 6.0_f64.sqrt()) * effective_tolerance;

    // Regime 2b (hybrid thin slab): 8 corner spheres + 4 merged z = 0 edge rows + z = 0 sheet.
    if c < slab_merge_limit {
        let mut centers = Vec::new();
        push_box_corners(
            &mut centers,
            a - corner_inset,
            b - corner_inset,
            c - corner_inset,
        );
        let slab_half_step = (face_radius_squared - c * c).sqrt();
        let x_edge_length = (2.0 * a - 4.0 * corner_inset).max(0.0);
        let y_edge_length = (2.0 * b - 4.0 * corner_inset).max(0.0);
        for (length, along_x) in [(x_edge_length, true), (y_edge_length, false)] {
            if length > 0.0 {
                let count = (length / (2.0 * slab_half_step)).ceil() as usize;
                let step = length / count as f64;
                for sign in [-1.0, 1.0] {
                    for k in 0..count {
                        let coord = -0.5 * length + (k as f64 + 0.5) * step;
                        if along_x {
                            centers.push((coord, sign * (b - c), 0.0, None));
                        } else {
                            centers.push((sign * (a - c), coord, 0.0, None));
                        }
                    }
                }
            }
        }
        let inner_a = (a - 2.0 * c).max(0.0);
        let inner_b = (b - 2.0 * c).max(0.0);
        if inner_a > 0.0 && inner_b > 0.0 {
            let sheet = cover_rect_2d_within(inner_a, inner_b, face_radius, a - c, b - c)?;
            for (x, y) in sheet {
                centers.push((x, y, 0.0, None));
            }
        }
        return Some(emit(centers));
    }

    // Regime 3 (general 3D box): recursive concentric medial shells closing onto z = 0.
    // On each axis, try the minimum edge count and exact geometric-threshold counts where the
    // bisector strip width crosses 2*corner_inset, c, or c - face_radius/sqrt(2) (capped at
    // 2*n_min + 1, since adding more than n_min spheres per edge always costs more than Shell 2).
    let edge_lengths = sorted_half.map(|extent| (2.0 * extent - 4.0 * corner_inset).max(0.0));
    let candidates_per_axis = [0usize, 1, 2].map(|axis| {
        edge_count_candidates(
            edge_lengths[axis],
            effective_tolerance,
            corner_inset,
            c,
            face_radius,
            SQRT_2,
        )
    });

    let mut best: Option<Centers> = None;
    for &nx in &candidates_per_axis[0] {
        for &ny in &candidates_per_axis[1] {
            for &nz in &candidates_per_axis[2] {
                if let Some(candidate) = build_general_box_cover(
                    sorted_half,
                    effective_tolerance,
                    face_radius,
                    corner_inset,
                    edge_lengths,
                    [nx, ny, nz],
                ) && best
                    .as_ref()
                    .is_none_or(|b_vec| candidate.len() < b_vec.len())
                {
                    best = Some(candidate);
                }
            }
        }
    }
    best.map(emit)
}

/// Candidate Sphere counts for one axis's Shell 1 edge rows: the minimum that covers the edge,
/// one step off the vertical tangent of `t(h)` at `h = sqrt(2)*tau`, and the exact counts where
/// the bisector strip width reaches `2*corner_inset`, `c`, or `c - face_radius/sqrt(2)`.
fn edge_count_candidates(
    length: f64,
    tau: f64,
    corner_inset: f64,
    min_half: f64,
    face_radius: f64,
    sqrt_2: f64,
) -> Vec<usize> {
    if length <= 0.0 {
        return vec![0];
    }
    let min_count = (length / (2.0 * sqrt_2 * tau)).ceil() as usize;
    // Shell 2's step delta_2 >= (sqrt(11/4) + 1/2)*tau > sqrt(2)*tau, so Shell 2 uses fewer than
    // `min_count` spheres per edge; bumping Shell 1 past 2*min_count + 1 can never win.
    let cap = 2 * min_count + 1;
    let mut counts = vec![min_count];
    if min_count < cap {
        counts.push(min_count + 1);
    }
    let max_bisector_width = 2.0 * (1.0 + sqrt_2) * tau;
    let target_widths = [
        2.0 * corner_inset,
        min_half,
        (min_half - face_radius / sqrt_2).max(0.0),
    ];
    for width in target_widths {
        if width > 2.0 * tau && width < max_bisector_width {
            let target_depth = 0.5 * width;
            let discriminant = 2.0 * tau * tau - (target_depth - tau) * (target_depth - tau);
            if discriminant > 0.0 {
                let target_half_step = discriminant.sqrt();
                let count = (length / (2.0 * target_half_step)).ceil() as usize;
                if count >= min_count && count <= cap && !counts.contains(&count) {
                    counts.push(count);
                }
            }
        }
    }
    counts.sort_unstable();
    counts
}

/// Emits the 8 sign combinations of `(x, y, z)` for a box's corners or corner-ray shell.
fn push_box_corners(out: &mut Centers, x: f64, y: f64, z: f64) {
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                out.push((sx * x, sy * y, sz * z, None));
            }
        }
    }
}

/// Emits the 4 edge-medial rows of `count` uniformly spaced spheres along `axis` over
/// `[-length/2, length/2]`, inset by `bisector_depth` from the other two faces.
fn push_box_edge_rows(
    out: &mut Centers,
    axis: usize,
    half: [f64; 3],
    length: f64,
    count: usize,
    bisector_depth: f64,
) {
    let j = (axis + 1) % 3;
    let k = (axis + 2) % 3;
    let step = length / count as f64;
    let offset_j = half[j] - bisector_depth;
    let offset_k = half[k] - bisector_depth;
    for sj in [-1.0, 1.0] {
        for sk in [-1.0, 1.0] {
            for index in 0..count {
                let mut point = [0.0f64; 3];
                point[axis] = -0.5 * length + (index as f64 + 0.5) * step;
                point[j] = sj * offset_j;
                point[k] = sk * offset_k;
                out.push((point[0], point[1], point[2], None));
            }
        }
    }
}

/// Builds the Regime 3 recursive medial-shell + staggered mid-plane cover for one choice of
/// Shell 1 edge counts `edge_counts`.
fn build_general_box_cover(
    half: [f64; 3],
    tau: f64,
    face_radius: f64,
    corner_inset: f64,
    edge_lengths: [f64; 3],
    edge_counts: [usize; 3],
) -> Option<Centers> {
    use std::f64::consts::SQRT_2;

    let [a, b, c] = half;
    let mut centers = Vec::new();

    // Shell 1: 8 corners and up to 12 edge rows along the 45-degree bisectors.
    push_box_corners(
        &mut centers,
        a - corner_inset,
        b - corner_inset,
        c - corner_inset,
    );
    let mut strip_widths = [2.0 * corner_inset; 3];
    for axis in 0..3 {
        if edge_counts[axis] == 0 {
            continue;
        }
        let j = (axis + 1) % 3;
        let k = (axis + 2) % 3;
        let half_step = edge_lengths[axis] / (2.0 * edge_counts[axis] as f64);
        let bisector_depth = (tau + (2.0 * tau * tau - half_step * half_step).max(0.0).sqrt())
            .min(half[j])
            .min(half[k]);
        let face_reach = (2.0 * tau * bisector_depth + tau * tau - half_step * half_step)
            .max(0.0)
            .sqrt();
        strip_widths[axis] = bisector_depth + face_reach;
        push_box_edge_rows(
            &mut centers,
            axis,
            half,
            edge_lengths[axis],
            edge_counts[axis],
            bisector_depth,
        );
    }

    // Exact guaranteed face insets after Shell 1: on face k (axes i, j), every point with
    // u_i <= inset_i or u_j <= inset_j is covered.
    let corner_span = 2.0 * corner_inset;
    let safe_inset = |along: f64, across: f64| {
        if along >= corner_span {
            across
        } else {
            across.min(corner_span)
        }
    };
    let z_inset_x = safe_inset(strip_widths[0], strip_widths[1]);
    let z_inset_y = safe_inset(strip_widths[1], strip_widths[0]);
    let y_inset_x = safe_inset(strip_widths[0], strip_widths[2]);
    let y_inset_z = safe_inset(strip_widths[2], strip_widths[0]);
    let x_inset_y = safe_inset(strip_widths[1], strip_widths[2]);
    let x_inset_z = safe_inset(strip_widths[2], strip_widths[1]);

    // Fast path F1: Shell 1 already covers all four side faces, leaving only the z = +-c core.
    let x_sides_done = x_inset_z >= c || x_inset_y >= b;
    let y_sides_done = y_inset_z >= c || y_inset_x >= a;
    if x_sides_done && y_sides_done {
        let rem_a = (a - z_inset_x).max(0.0);
        let rem_b = (b - z_inset_y).max(0.0);
        if rem_a <= 0.0 || rem_b <= 0.0 {
            return Some(centers);
        }
        if let Some(sheet) = cover_rect_2d_within(rem_a, rem_b, face_radius, a - c, b - c) {
            centers.extend(sheet.into_iter().map(|(x, y)| (x, y, 0.0, None)));
            return Some(centers);
        }
    }

    // Fast path F3: x-ends are done and y-sides have a narrow remaining strip; test closing both
    // y = +-b and z = +-c together with 1 or 2 shared staggered rows on z = 0.
    let y_strip = (c - y_inset_z).max(0.0);
    let mut best_tail: Option<Centers> = None;
    if x_sides_done && y_strip > 0.0 && y_strip <= face_radius / SQRT_2 {
        let z_rem_a = (a - z_inset_x).max(0.0);
        let y_rem_a = (a - y_inset_x).max(0.0);
        let z_rem_b = (b - z_inset_y).max(0.0);
        let shared_a = z_rem_a.max(y_rem_a);
        if shared_a > 0.0 {
            best_tail =
                try_shared_midplane_rows(half, tau, face_radius, shared_a, z_rem_b, y_strip);
        }
    }

    // General path: concentric slanted medial shells m = 2.. until u >= c - face_radius/sqrt(2).
    // From 2*tau*(u + delta) + tau^2 = 2*delta^2, v_1 = sqrt(tau*u + 3/4*tau^2) and each shell
    // advances delta by +tau while covering [u, u + 2*delta] on all six faces.
    let mut general_tail = Vec::new();
    let mut inset = strip_widths[0].min(strip_widths[1]).min(strip_widths[2]);
    let base_v = (tau * inset + 0.75 * tau * tau).sqrt();
    let mut shell = 2usize;
    let stop_inset = c - face_radius / SQRT_2;
    while inset < stop_inset {
        let delta = base_v + (shell as f64 - 1.5) * tau;
        let bisector_depth = inset + delta;
        let next_inset = inset + 2.0 * delta;
        push_box_corners(
            &mut general_tail,
            a - bisector_depth,
            b - bisector_depth,
            c - bisector_depth,
        );
        for axis in 0..3 {
            let length = (2.0 * half[axis] - 2.0 * next_inset).max(0.0);
            if length > 0.0 {
                let count = (length / (2.0 * delta)).ceil() as usize;
                push_box_edge_rows(&mut general_tail, axis, half, length, count, bisector_depth);
            }
        }
        inset = next_inset;
        shell += 1;
    }

    // Final z = 0 closure (Cases F1, F2, F4).
    if inset >= c {
        let rem_a = (a - inset).max(0.0);
        let rem_b = (b - inset).max(0.0);
        if rem_a > 0.0 && rem_b > 0.0 {
            let sheet = cover_rect_2d_within(rem_a, rem_b, face_radius, a - c, b - c)?;
            general_tail.extend(sheet.into_iter().map(|(x, y)| (x, y, 0.0, None)));
        }
    } else {
        let remaining_z = c - inset;
        let max_radius = c + tau;
        // Case F2: a single central sphere closes all 6 faces (exact for cubes).
        let rem_a = a - inset;
        let rem_b = b - inset;
        if rem_a * rem_a + rem_b * rem_b <= face_radius * face_radius
            && a * a + rem_b * rem_b + remaining_z * remaining_z <= max_radius * max_radius
            && b * b + rem_a * rem_a + remaining_z * remaining_z <= max_radius * max_radius
        {
            general_tail.push((0.0, 0.0, 0.0, None));
        } else {
            // Case F4: 4 merged corners + 4 merged z = 0 border rows + central staggered sheet.
            let inner_inset = 2.0 * c - inset;
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    general_tail.push((sx * (a - c), sy * (b - c), 0.0, None));
                }
            }
            let merged_half_step = (face_radius * face_radius - remaining_z * remaining_z)
                .max(0.0)
                .sqrt();
            let lx = (2.0 * a - 2.0 * inner_inset).max(0.0);
            let ly = (2.0 * b - 2.0 * inner_inset).max(0.0);
            if lx > 0.0 {
                let nx = (lx / (2.0 * merged_half_step)).ceil() as usize;
                let step = lx / nx as f64;
                for sign in [-1.0, 1.0] {
                    for k in 0..nx {
                        let x = -0.5 * lx + (k as f64 + 0.5) * step;
                        general_tail.push((x, sign * (b - c), 0.0, None));
                    }
                }
            }
            if ly > 0.0 {
                let ny = (ly / (2.0 * merged_half_step)).ceil() as usize;
                let step = ly / ny as f64;
                for sign in [-1.0, 1.0] {
                    for k in 0..ny {
                        let y = -0.5 * ly + (k as f64 + 0.5) * step;
                        general_tail.push((sign * (a - c), y, 0.0, None));
                    }
                }
            }
            let core_a = (a - inner_inset).max(0.0);
            let core_b = (b - inner_inset).max(0.0);
            if core_a > 0.0 && core_b > 0.0 {
                let sheet = cover_rect_2d_within(core_a, core_b, face_radius, a - c, b - c)?;
                general_tail.extend(sheet.into_iter().map(|(x, y)| (x, y, 0.0, None)));
            }
        }
    }

    match best_tail {
        Some(tail) if tail.len() < general_tail.len() => centers.extend(tail),
        _ => centers.extend(general_tail),
    }
    Some(centers)
}

/// Fast path F3: closes both the remaining `y = +-b` strips (`|x| <= span_a, |z| <= y_strip`) and
/// the `z = +-c` core (`|x| <= span_a, |y| <= z_rem_b`) with 1 centered row or 2 staggered rows on
/// `z = 0`, solving the optimal row offset `y_row` in closed form.
fn try_shared_midplane_rows(
    half: [f64; 3],
    tau: f64,
    face_radius: f64,
    span_a: f64,
    z_rem_b: f64,
    y_strip: f64,
) -> Option<Centers> {
    let [a, b, c] = half;
    let max_x = a - c;
    let max_y = b - c;
    let end_inset = (span_a - max_x).max(0.0);
    let r2 = face_radius * face_radius;
    if end_inset >= face_radius {
        return None;
    }
    let end_chord = (r2 - end_inset * end_inset).sqrt();
    let max_radius = c + tau;
    let side_budget = max_radius * max_radius - y_strip * y_strip;
    if side_budget <= 0.0 {
        return None;
    }

    let mut best: Option<Centers> = None;

    // Option 1: a single row on y = 0, z = 0 if it reaches both y = +-b and z = +-c.
    if z_rem_b < face_radius && b * b < side_budget {
        let half_reach = (r2 - z_rem_b * z_rem_b).min(side_budget - b * b).sqrt();
        if let Some(xs) = cover_segment_1d_within(span_a, half_reach, max_x) {
            best = Some(xs.into_iter().map(|x| (x, 0.0, 0.0, None)).collect());
        }
    }

    // Option 2: two staggered rows at y = +-y_row on z = 0.
    // The three half-step limits are h1 (outer z-face strip), h2 (middle staggered strip +
    // clamped end strip), and h3 (side face y = +-b). Since min(h1, h3) is non-decreasing and h2
    // is non-increasing, the maximum occurs at a kink or the exact crossing h1 = h2 or h3 = h2.
    let mut candidate_ys = vec![
        0.0,
        max_y,
        z_rem_b.clamp(0.0, max_y),
        (0.5 * end_chord).clamp(0.0, max_y),
        ((z_rem_b + end_chord) / 3.0).clamp(0.0, max_y),
    ];
    // Crossing h3(y)^2 = h2(y)^2 expands to:
    // 3*y^2 + (2*b - 4*end_chord)*y + (c^2 - y_strip^2 - b^2 + end_chord^2) = 0.
    let linear = 2.0 * b - 4.0 * end_chord;
    let constant = c * c - y_strip * y_strip - b * b + end_chord * end_chord;
    let discriminant = linear * linear - 12.0 * constant;
    if discriminant >= 0.0 {
        let root = discriminant.sqrt();
        candidate_ys.push(((-linear + root) / 6.0).clamp(0.0, max_y));
        candidate_ys.push(((-linear - root) / 6.0).clamp(0.0, max_y));
    }

    for y_row in candidate_ys {
        let outer_z = (z_rem_b - y_row).max(0.0);
        if outer_z >= face_radius {
            continue;
        }
        let h1_sq = r2 - outer_z * outer_z;
        let middle_gap = (2.0 * y_row - end_chord).max(0.0);
        if middle_gap >= face_radius {
            continue;
        }
        let h2_sq = r2 - middle_gap * middle_gap;
        let side_y = b - y_row;
        let h3_sq = side_budget - side_y * side_y;
        if h3_sq <= 0.0 {
            continue;
        }
        let half_reach = h1_sq.min(h2_sq).min(h3_sq).sqrt();
        if half_reach <= 0.0 {
            continue;
        }
        let count = (span_a / half_reach).ceil().max(1.0) as usize;
        let actual_half = span_a / count as f64;
        // Even-row centers sit at -span_a + (2*k + 1)*actual_half; they stay inside [-max_x, max_x]
        // iff actual_half >= end_inset.
        if actual_half < end_inset {
            continue;
        }
        let total = 2 * count + 1;
        if best
            .as_ref()
            .is_some_and(|existing| existing.len() <= total)
        {
            continue;
        }
        let mut pts = Vec::with_capacity(total);
        for k in 0..count {
            let x = -span_a + (2 * k + 1) as f64 * actual_half;
            pts.push((x, y_row, 0.0, None));
        }
        for k in 0..=count {
            let x = (-span_a + (2 * k) as f64 * actual_half).clamp(-max_x, max_x);
            pts.push((x, -y_row, 0.0, None));
        }
        best = Some(pts);
    }
    best
}

/// Fewest 1D centers inside `[-max_coord, max_coord]` whose intervals of half-width `half_reach`
/// cover `[-extent, extent]`.
fn cover_segment_1d_within(extent: f64, half_reach: f64, max_coord: f64) -> Option<Vec<f64>> {
    let inset = (extent - max_coord).max(0.0);
    if half_reach <= 0.0 || inset > half_reach {
        return None;
    }
    if extent <= half_reach {
        return Some(vec![0.0]);
    }
    let unconstrained = (extent / half_reach).ceil() as usize;
    let actual_half = extent / unconstrained as f64;
    if actual_half >= inset {
        let coords = (0..unconstrained)
            .map(|k| -extent + (2 * k + 1) as f64 * actual_half)
            .collect();
        return Some(coords);
    }
    // Pin the two outer centers at +-max_coord and divide [-max_coord, max_coord] evenly.
    let intervals = (max_coord / half_reach).ceil().max(1.0) as usize;
    let count = intervals + 1;
    let step = 2.0 * max_coord / intervals as f64;
    let coords = (0..count).map(|k| -max_coord + k as f64 * step).collect();
    Some(coords)
}

/// 2D disk cover of `[-a, a] x [-b, b]` with disk radius `radius` and every center constrained to
/// `[-max_x, max_x] x [-max_y, max_y]`, taking the fewest disks among 1 disk, rectangular grids
/// (with pinned outer rows/columns when constrained), and staggered grids in both orientations.
fn cover_rect_2d_within(
    a: f64,
    b: f64,
    radius: f64,
    max_x: f64,
    max_y: f64,
) -> Option<Vec<(f64, f64)>> {
    let r2 = radius * radius;
    if a * a + b * b <= r2 {
        return Some(vec![(0.0, 0.0)]);
    }
    let mut best: Option<Vec<(f64, f64)>> = None;
    for swapped in [false, true] {
        let (ra, rb, mx, my) = if swapped {
            (b, a, max_y, max_x)
        } else {
            (a, b, max_x, max_y)
        };
        let ex = (ra - mx).max(0.0);
        let ey = (rb - my).max(0.0);
        if ex * ex + ey * ey > r2 {
            continue;
        }
        let place = |x: f64, y: f64| if swapped { (y, x) } else { (x, y) };

        let rows_min = (rb / radius).floor() as usize + 1;
        let mut rows_max = ((2.0 * rb / radius).ceil() as usize + 3).max(rows_min);
        let max_by = (r2 - ex * ex).max(0.0).sqrt();
        if my > 0.0 && max_by > 0.0 {
            let pinned_rows = (my / max_by).ceil() as usize + 1;
            rows_max = rows_max.max(pinned_rows);
        }

        let end_chord = (r2 - ex * ex).max(0.0).sqrt();
        for rows in rows_min..=rows_max {
            // Rectangular grid with outer rows pinned to +-my if needed.
            let (ys, half_y) = if rows == 1 {
                (vec![0.0], rb)
            } else if rb / rows as f64 >= ey {
                let half_y = rb / rows as f64;
                let ys = (0..rows)
                    .map(|j| -rb + (2 * j + 1) as f64 * half_y)
                    .collect();
                (ys, half_y)
            } else {
                let half_y = ey.max(my / (rows - 1) as f64);
                let step = 2.0 * my / (rows - 1) as f64;
                let ys = (0..rows).map(|j| -my + j as f64 * step).collect();
                (ys, half_y)
            };
            if half_y < radius {
                let half_x = (r2 - half_y * half_y).sqrt();
                if let Some(xs) = cover_segment_1d_within(ra, half_x, mx) {
                    let total = rows * xs.len();
                    if best.as_ref().is_none_or(|existing| total < existing.len()) {
                        let mut pts = Vec::with_capacity(total);
                        for &y in &ys {
                            for &x in &xs {
                                pts.push(place(x, y));
                            }
                        }
                        best = Some(pts);
                    }
                }
            }

            // Staggered (hexagonal-style) grid for rows >= 2.
            // Setting outer margin q >= (2*rb - (rows - 1)*end_chord) / (rows + 1) guarantees
            // 2*d_in - end_chord <= q <= sqrt(r^2 - h_act^2), which simultaneously covers the
            // outer strips, interior staggered triangles, and clamped odd-row end strips.
            if rows >= 2 && end_chord > 0.0 {
                let q = ((2.0 * rb - (rows - 1) as f64 * end_chord) / (rows + 1) as f64)
                    .max(ey)
                    .max(0.0);
                if q < radius && q < rb {
                    let half_in = (rb - q) / (rows - 1) as f64;
                    let half_x = (r2 - q * q).sqrt();
                    if half_x > 0.0 {
                        let count = (ra / half_x).ceil().max(1.0) as usize;
                        let actual_half = ra / count as f64;
                        if actual_half >= ex {
                            let total = rows * count + rows / 2;
                            if best.as_ref().is_none_or(|existing| total < existing.len()) {
                                let mut pts = Vec::with_capacity(total);
                                for j in 0..rows {
                                    let y = -rb + q + 2.0 * j as f64 * half_in;
                                    if j % 2 == 0 {
                                        for k in 0..count {
                                            let x = -ra + (2 * k + 1) as f64 * actual_half;
                                            pts.push(place(x, y));
                                        }
                                    } else {
                                        for k in 0..=count {
                                            let x =
                                                (-ra + (2 * k) as f64 * actual_half).clamp(-mx, mx);
                                            pts.push(place(x, y));
                                        }
                                    }
                                }
                                best = Some(pts);
                            }
                        }
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::description::Geometry;

    #[test]
    fn packed_cover_matches_scalar_reaches() {
        let candidates: Vec<_> = (0..11)
            .map(|i| (Vec3::new(4.0 * i as f32, i as f32, -2.0 * i as f32), 0.25))
            .collect();
        for count in 0..=9 {
            let picks: Vec<usize> = (1..=count).rev().collect();
            for overhang in [0.0, 0.125] {
                let cover = SphereCover::new(&candidates, &picks, overhang);
                assert!(
                    !cover.contains(candidates[0].0),
                    "an empty cover or unused lane cannot cover an unpicked sphere"
                );
                for &i in &picks {
                    let (center, radius) = candidates[i];
                    let edge = center + Vec3::X * (radius + overhang);
                    assert_eq!(
                        picks
                            .iter()
                            .filter(|&&j| reaches(edge, candidates[j], overhang))
                            .count(),
                        1,
                        "each picked sphere must be the only hit at its boundary"
                    );
                    for x in [edge.x.next_down(), edge.x, edge.x.next_up()] {
                        let p = Vec3::new(x, edge.y, edge.z);
                        let expected = picks.iter().any(|&j| reaches(p, candidates[j], overhang));
                        assert_eq!(
                            cover.contains(p),
                            expected,
                            "count {count}, sphere {i}, {p:?}"
                        );
                    }
                }
            }
        }
        let p = Vec3::new(1.0, 2.0f32.powi(-12), 2.0f32.powi(-12));
        assert!(
            reaches(p, (Vec3::ZERO, 1.0), 0.0),
            "the scalar sum rounds to one"
        );
        assert!(
            p.x * p.x + (p.y * p.y + p.z * p.z) > 1.0,
            "regrouping the sum must change the result for this point"
        );
        let cover = SphereCover::new(&[(Vec3::ZERO, 1.0)], &[0], 0.0);
        assert!(
            cover.contains(p),
            "packing must preserve the scalar addition order"
        );
    }
    #[test]
    fn the_cover_counts_a_sample_its_own_candidate_reaches_at_the_edge_of_its_window() {
        // A box from x = -0.05 to just past 0.002, and a sample on that face whose own candidate
        // reaches it at exactly its radius and overhang, where an unpadded window along x ends
        // short of it.
        let mut mesh = Geometry::Box {
            half: Vec3::new(0.026, 0.05, 0.05),
        }
        .mesh()
        .expect("a box meshes");
        let face = f32::from_bits(0.002f32.to_bits() + 3);
        for v in &mut mesh.vertices {
            v.x = if v.x > 0.0 { face } else { -0.05 };
        }
        let mut fit = LinkFit::new(
            &[mesh],
            &SphereOptions::default(),
            &mut Rng::new(1),
            /* parallel */ false,
            /* surface */ None,
        )
        .expect("a box has a surface");
        let (p, overhang) = (Vec3::new(face, 0.026_000_006, 0.0), 0.001);
        let (c, r) = fit
            .tangent(p, Vec3::X, overhang)
            .expect("the face has room");
        assert!(
            reaches(p, (c, r), overhang) && p.x > c.x + (r + overhang),
            "the sample is not at the window's edge: {p:?}, {c:?}, {r}"
        );
        fit.samples = vec![p];
        fit.normals = vec![Vec3::X];
        fit.candidates = vec![(c, r)];
        let (picks, left) = fit.greedy(overhang, usize::MAX);
        assert!(left.is_empty() && picks == [0], "{picks:?} leave {left:?}");
    }

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
        let fit = LinkFit::new(
            &[mesh],
            &o,
            &mut Rng::new(3),
            /* parallel */ false,
            /* surface */ None,
        )
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

    /// A box's medial balls lie on its medial axis: one from the middle of a wide face on the box's
    /// middle plane, touching both wide faces, and one from near an edge on the edge's bisector.
    /// A cover from them keeps no pick whose every sample the others reach.
    #[test]
    fn medial_balls_lie_on_the_medial_axis_and_covers_keep_no_redundant_pick() {
        let half = Vec3::new(0.12, 0.0125, 0.025);
        let mesh = Geometry::Box { half }.mesh().expect("a box meshes");
        let o = SphereOptions::default();
        let (surface, _) = surface_grid(&[Obstacle::cuboid(Vec3::ZERO, half)], o.tolerance);
        let fit = LinkFit::new(
            &[mesh],
            &o,
            &mut Rng::new(3),
            /* parallel */ false,
            Some(&surface),
        )
        .expect("the box has surface");
        let middle = fit
            .medial(Vec3::new(0.0, half.y, 0.0), Vec3::Y, half.y)
            .expect("inside the box");
        assert!(
            middle.0.y.abs() <= MIN_TANGENT_DEPTH && (middle.1 - half.y).abs() <= MIN_TANGENT_DEPTH,
            "{middle:?}"
        );
        let edge = fit
            .medial(Vec3::new(0.0, half.y, half.z - 0.004), Vec3::Y, half.y)
            .expect("inside the box");
        let (from_side, from_top) = (half.y - edge.0.y, half.z - edge.0.z);
        assert!(
            (from_side - from_top).abs() <= MIN_TANGENT_DEPTH
                && (edge.1 - from_side.min(from_top)).abs() < 1e-6,
            "{edge:?}"
        );
        let (picks, _) = fit.greedy(o.tolerance, usize::MAX);
        assert!(picks.len() > 1);
        for &pick in &picks {
            let others: Vec<(Vec3, f32)> = picks
                .iter()
                .filter(|&&other| other != pick)
                .map(|&other| fit.candidates[other])
                .collect();
            let its_own = fit.samples.iter().any(|&p| {
                reaches(p, fit.candidates[pick], o.tolerance)
                    && !others.iter().any(|&other| reaches(p, other, o.tolerance))
            });
            assert!(its_own, "pick {pick} reaches no sample the others miss");
        }
    }

    /// The medial box cover on random boxes, 0.5 mm to 15 cm along each axis in any order, for
    /// tolerances of 0.5 to 10 mm: every point of a grid a quarter tolerance apart through each box
    /// lies inside a sphere, and no sphere reaches past the box farther than the tolerance. Boxes
    /// whose grid would take more than 400,000 points are left out.
    #[test]
    fn medial_box_covers_hold_random_boxes() {
        let mut state: u64 = 12345;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let (smallest, largest) = (0.0005f64.ln(), 0.15f64.ln());
        let (mut tried, mut failed) = (0, vec![]);
        for _ in 0..400 {
            let tolerance = 0.0005 + uniform() * 0.0095;
            let mut h = [0.0f64; 3];
            for extent in &mut h {
                *extent = (smallest + uniform() * (largest - smallest)).exp();
            }
            let step = tolerance / 4.0;
            let points: f64 = h.iter().map(|x| 2.0 * x / step).product();
            if points > 4e5 {
                continue;
            }
            tried += 1;
            let half = Vec3::new(h[0] as f32, h[1] as f32, h[2] as f32);
            let Some(spheres) = box_cover(half, tolerance as f32) else {
                failed.push((half, tolerance, "no cover".to_string()));
                continue;
            };
            let h = [half.x, half.y, half.z].map(f64::from);
            let depth = |s: &[f32; 4]| {
                (0..3)
                    .map(|k| h[k] - f64::from(s[k]).abs())
                    .fold(f64::INFINITY, f64::min)
            };
            let past = spheres
                .iter()
                .map(|s| f64::from(s[3]) - depth(s))
                .fold(f64::NEG_INFINITY, f64::max);
            let counts = h.map(|x| (2.0 * x / step).ceil() as usize + 1);
            let at =
                |k: usize, i: usize| -h[k] + 2.0 * h[k] * i as f64 / (counts[k] - 1).max(1) as f64;
            let mut gap = f64::NEG_INFINITY;
            for i in 0..counts[0] {
                for j in 0..counts[1] {
                    for k in 0..counts[2] {
                        let p = [at(0, i), at(1, j), at(2, k)];
                        let outside = |s: &[f32; 4]| {
                            let d2: f64 = (0..3).map(|a| (p[a] - f64::from(s[a])).powi(2)).sum();
                            d2.sqrt() - f64::from(s[3])
                        };
                        gap = gap.max(spheres.iter().map(outside).fold(f64::INFINITY, f64::min));
                    }
                }
            }
            if gap > 0.0 || past > tolerance + 1e-9 {
                failed.push((half, tolerance, format!("{gap:e} outside, {past:e} past")));
            }
        }
        assert!(tried > 100, "{tried} boxes tried");
        assert!(failed.is_empty(), "{failed:?}");
    }
}
