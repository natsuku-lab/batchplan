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
//! A mesh with open edges has no inside, and the side of its nearest face stands in for it, as it
//! does for a sample's own candidate, a little way in from its face.

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
    /// Spheres for the whole robot.
    pub budget: usize,
    /// How far spheres may reach beyond the geometry (meters) before the fit spends more of the
    /// budget. When the budget cannot reach it, the fit uses the smallest overhang it can.
    pub tolerance: f32,
    /// The farthest any sphere may reach beyond the geometry (meters), whatever the budget: each
    /// sphere's radius is at most how deep its center lies inside plus this
    /// ([`LinkQualification::overhang`]), so `Some(0.0)` keeps every sphere inside. Spheres are
    /// chosen as without it and then held to it, which leaves surface outside them wherever
    /// `tolerance`, as the budget scales it, is the larger: [`Robot::qualify`](crate::Robot::qualify)
    /// reports how deep as each link's `gap`. The geometry must be closed, every edge shared by
    /// as many triangles one way as the other, and a collision model given with the options is
    /// held to it as a fitted one is. A center's depth is its distance to the nearest face of a
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
}

impl LinkFit {
    fn new(meshes: &[TriMesh], o: &SphereOptions, rng: &mut Rng, parallel: bool) -> Option<Self> {
        use parry3d::query::PointQuery;

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
    /// overhang, the share of the dense samples outside every one and how far outside the
    /// farthest lies, and the geometry's thickness.
    fn measure(&self, link: &str, spheres: &[[f32; 4]]) -> LinkQualification {
        let depths = Depths::new(&self.solids);
        let overhang = spheres
            .iter()
            .map(|s| s[3] - depths.at(Vec3::new(s[0], s[1], s[2])))
            .fold(0.0, f32::max);
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
    /// The largest of each link's overhang over its tolerance.
    pub(crate) relative_overhang: f32,
    /// What every link's tolerance was scaled by for the budget: 1 where it reached them.
    pub(crate) scale: f32,
}

/// Spheres for each link's meshes (in the link frame), at most `o.budget` in total. Where the
/// budget reaches every tolerance (`tolerances`, per link), each closed link's overhang is at most
/// its tolerance times the least growth, from 1 to `BOUND_GROWTH`, the budget affords; otherwise at
/// most twice its tolerance scaled up by the smallest factor the budget can achieve; but for the
/// parts the module doc names. With `parallel`, the work runs on the current rayon pool; without,
/// on the calling thread alone, without registering it with any pool.
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
/// cylinder, or a thin one the budget cannot bring within the tolerance, is left to the fit,
/// which spends fewer spheres on its inside; so is a distance grid. A box's sphere reaches past
/// its faces by its radius less the cell's smallest half size, so of the covers within the
/// tolerance the one of fewest spheres is taken, or where the budget cannot reach it, the least
/// overhang the budget can; so for a capsule. The shapes but the spheres share the budget the
/// spheres leave. Microseconds, where fitting takes milliseconds, and within the tolerance on a
/// thin wide shape, where a fit of as many spheres is not. Under `o.max_overhang` the covers are
/// those within the smaller of it and the tolerance, and where the budget has none within it,
/// the shapes are left to the fit, which holds its spheres to it.
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
                let cells = box_cells(half_extents, tolerance, each);
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
                let own = capsule_cover(radius, half_length, tolerance, each);
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
/// `tolerance` past it, else `budget` of them. Evenly spaced from end to end, each wide enough
/// to reach the capsule's side halfway to the next.
fn capsule_cover(radius: f32, half_length: f32, tolerance: f32, budget: usize) -> Vec<[f32; 4]> {
    if half_length <= 0.0 {
        return vec![[0.0, 0.0, 0.0, radius]];
    }
    // With `count` spheres: half the spacing between two, and how far each reaches past the side.
    let half_gap = |count: usize| half_length / (count - 1).max(1) as f32;
    let covering = |count: usize| match count {
        1 => half_length + radius,
        _ => radius.hypot(half_gap(count)) * HAIR,
    };
    let within = (1..=budget).find(|&count| covering(count) - radius <= tolerance);
    let count = within.unwrap_or(budget);
    if count == 1 {
        return vec![[0.0, 0.0, 0.0, covering(1)]];
    }
    (0..count)
        .map(|k| {
            let z = -half_length + 2.0 * half_gap(count) * k as f32;
            [0.0, 0.0, z, covering(count)]
        })
        .collect()
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

// What a fit runs once for a link, or not at all without a limit, is kept here, after the
// functions it spends its time in: set among them, it took fits 2 to 8% longer in a one-unit
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

/// How deep points lie inside a link's solids, one at a time.
struct Depths<'a> {
    solids: &'a [Solid],
    /// Per solid that is closed, its surface, for what winds around a point.
    surfaces: Vec<Option<Surface<'a>>>,
}

impl<'a> Depths<'a> {
    fn new(solids: &'a [Solid]) -> Self {
        let surface = |s: &'a Solid| {
            s.closed
                .then(|| Surface::new(s.mesh.vertices(), s.mesh.indices()))
        };
        Self {
            solids,
            surfaces: solids.iter().map(surface).collect(),
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
        deepest.unwrap_or(-nearest)
    }
}

/// Why spheres cannot be held within a distance of `meshes` ([`SphereOptions::max_overhang`]), if
/// they cannot: a mesh that is not closed has no inside for a sphere's depth to be measured in.
/// Closed, as [`Solid::closed`]: every edge borders as many triangles that run along it one way
/// as the other, once vertices at one place are merged and triangles that name a vertex twice
/// are dropped. A triangle given twice the same way leaves its edges with one too many.
pub(crate) fn closed(meshes: &[TriMesh]) -> Result<(), String> {
    let flags = TriMeshFlags::MERGE_DUPLICATE_VERTICES | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES;
    for (k, mesh) in meshes.iter().enumerate() {
        let of = format!("mesh {k} of {}", meshes.len());
        let solid = ParryMesh::with_flags(mesh.vertices.clone(), mesh.triangles.clone(), flags)
            .map_err(|e| format!("{of} has no surface: {e}"))?;
        if solid.indices().is_empty() {
            return Err(format!("{of} has no surface"));
        }
        // Each triangle from its lowest vertex, so that one given again is the same three.
        let from_lowest = |&[a, b, c]: &[u32; 3]| {
            if a <= b && a <= c {
                [a, b, c]
            } else if b <= c {
                [b, c, a]
            } else {
                [c, a, b]
            }
        };
        let mut faces: Vec<[u32; 3]> = solid.indices().iter().map(from_lowest).collect();
        faces.sort_unstable();
        let again = faces.windows(2).filter(|pair| pair[0] == pair[1]).count();
        if again > 0 {
            return Err(format!(
                "{of} is not closed: it gives {again} of its {} faces again the same way",
                faces.len() - again
            ));
        }
        // Per edge, the triangles that run along it from its lower vertex less those that run back.
        let mut edges: BTreeMap<(u32, u32), i32> = BTreeMap::new();
        for [a, b, c] in faces {
            for (from, to) in [(a, b), (b, c), (c, a)] {
                *edges.entry((from.min(to), from.max(to))).or_default() +=
                    if from < to { 1 } else { -1 };
            }
        }
        let open = edges.values().filter(|&&balance| balance != 0).count();
        if open > 0 {
            return Err(format!(
                "{of} is not closed: {open} of its {} edges border a triangle on one side alone, \
                 or triangles that face opposite ways",
                edges.len()
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
        let mut fit = LinkFit::new(&[mesh], &SphereOptions::default(), &mut Rng::new(1), false)
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
