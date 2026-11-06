//! Collision spheres fitted to link geometry, and the self-collision link pairs not worth checking.
//!
//! Spheres follow the voxel method cuRobo uses for robot models: interior grid points become
//! candidate spheres that touch the surface, and a greedy cover picks the candidates that reach
//! the most surface samples. The overhang allowed while covering is each link's tolerance, scaled
//! by one factor searched for across the whole robot, and a final inflation grows spheres until
//! every sample is inside one.

use std::collections::{BTreeMap, BTreeSet};

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
/// Samples every final sphere set must contain; denser than the ones the cover is chosen from.
const DENSE_SAMPLES: usize = 20_000;
const MAX_VERTEX_SAMPLES: usize = 1500;
const MAX_CANDIDATES: usize = 1500;
const MAX_VOXELS: usize = 200_000;
/// Bisection steps of the overhang search.
const SEARCH_STEPS: usize = 24;

/// Area-weighted random points on the surface of `meshes`.
fn surface_samples(meshes: &[TriMesh], count: usize, rng: &mut Rng) -> Vec<Vec3> {
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
            a * (1.0 - r1) + b * (r1 * (1.0 - r2)) + c * (r1 * r2)
        })
        .collect()
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
    /// Points the cover is chosen to reach: mesh vertices and random surface points.
    samples: Vec<Vec3>,
    /// Surface points the final spheres must contain.
    dense: Vec<Vec3>,
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
        let vertices: Vec<Vec3> = meshes
            .iter()
            .flat_map(|m| m.vertices.iter().copied())
            .collect();
        let stride = vertices.len().div_ceil(MAX_VERTEX_SAMPLES).max(1);
        let mut samples: Vec<Vec3> = vertices.into_iter().step_by(stride).collect();
        samples.extend(surface_samples(meshes, SURFACE_SAMPLES, rng));
        if samples.is_empty() {
            return None;
        }
        let dense = surface_samples(meshes, DENSE_SAMPLES, rng);
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
        let outside = self.dense.len() - self.dense.iter().filter(inside).count();
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
    /// and whether they reach every sample.
    fn greedy(&self, overhang: f32, limit: usize) -> (Vec<usize>, bool) {
        let words = self.samples.len().div_ceil(64);
        let reached = |&(c, r): &(Vec3, f32)| {
            let mut bits = vec![0u64; words];
            for (i, s) in self.samples.iter().enumerate() {
                if (*s - c).length() <= r + overhang {
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
        let mut picks = vec![];
        while picks.len() < limit && uncovered.iter().any(|&u| u != 0) {
            let gain = |i: usize| {
                cover[i]
                    .iter()
                    .zip(&uncovered)
                    .map(|(c, u)| (c & u).count_ones())
                    .sum::<u32>()
            };
            let (best, best_gain) = (0..self.candidates.len())
                .map(|i| (i, gain(i)))
                .max_by(|a, b| {
                    let radius = |i: usize| self.candidates[i].1;
                    a.1.cmp(&b.1)
                        .then(radius(a.0).total_cmp(&radius(b.0)))
                        .then(b.0.cmp(&a.0))
                })
                .expect("at least one candidate");
            if best_gain == 0 {
                break;
            }
            picks.push(best);
            for (u, c) in uncovered.iter_mut().zip(&cover[best]) {
                *u &= !c;
            }
        }
        let complete = uncovered.iter().all(|&u| u == 0);
        (picks, complete)
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
        for &s in self.samples.iter().chain(&self.dense) {
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
    /// The share of its geometry's surface outside every sphere, from 20,000 fresh samples: a
    /// fraction of a percent for fitted spheres, which cover their own samples, and more where a
    /// hand-tuned model leaves gaps.
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
    let fit = LinkFit::new(
        meshes,
        o,
        &mut Rng::new(o.rng_seed),
        /* parallel */ true,
    )?;
    Some(fit.measure(link, spheres))
}

/// The smallest scale, from 1 up, of every link's tolerance (`tolerances`, per link) at which the
/// greedy covers of all links fit in `o.budget` spheres.
fn search_scale(fits: &[Option<LinkFit>], tolerances: &[f32], o: &SphereOptions) -> f32 {
    let needed = |scale: f32| {
        let mut total = 0;
        for (f, &tolerance) in fits.iter().zip(tolerances) {
            let Some(f) = f else { continue };
            let (picks, complete) = f.greedy(tolerance * scale, o.budget + 1 - total);
            total += picks.len() + usize::from(!complete);
            if total > o.budget {
                break;
            }
        }
        total
    };
    if needed(1.0) <= o.budget {
        return 1.0;
    }
    // Bisect between the tolerances and a scale at which one sphere covers any link.
    let (mut lo, mut hi) = (1.0f32, 1.0f32);
    for (f, &tolerance) in fits.iter().zip(tolerances) {
        let Some(f) = f else { continue };
        let (lo_corner, hi_corner) = f
            .samples
            .iter()
            .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(a, b), &p| {
                (a.min(p), b.max(p))
            });
        hi = hi.max((hi_corner - lo_corner).length() / tolerance);
    }
    for _ in 0..SEARCH_STEPS {
        let mid = 0.5 * (lo + hi);
        if needed(mid) <= o.budget {
            hi = mid
        } else {
            lo = mid
        }
    }
    hi
}

/// Spheres `[x, y, z, radius]` for each link's meshes (in the link frame), at most `o.budget` in
/// total, and how far the largest one reaches beyond the surface it touches. Each link's overhang
/// is its tolerance (`tolerances`, per link) when the budget allows, otherwise every tolerance
/// scaled up by the smallest factor the budget can achieve. With `parallel`, the work runs on the
/// current rayon pool; without, on the calling thread alone, without registering it with any pool.
pub(crate) fn fit(
    links: &[Vec<TriMesh>],
    tolerances: &[f32],
    o: &SphereOptions,
    parallel: bool,
) -> (Vec<Vec<[f32; 4]>>, f32) {
    debug_assert!(o.check().is_ok(), "callers check the options");
    debug_assert_eq!(links.len(), tolerances.len());
    let mut rng = Rng::new(o.rng_seed);
    let fits: Vec<Option<LinkFit>> = links
        .iter()
        .map(|m| LinkFit::new(m, o, &mut rng, parallel))
        .collect();
    let scale = search_scale(&fits, tolerances, o);
    let fitted: Vec<(Vec<[f32; 4]>, f32)> = fits
        .iter()
        .zip(tolerances)
        .map(|(f, &tolerance)| {
            f.as_ref().map_or((vec![], 0.0), |f| {
                f.spheres(&f.greedy(tolerance * scale, usize::MAX).0)
            })
        })
        .collect();
    let reach = fitted.iter().map(|f| f.1).fold(0.0, f32::max);
    (fitted.into_iter().map(|f| f.0).collect(), reach)
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
