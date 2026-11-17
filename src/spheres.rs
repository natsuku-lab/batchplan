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
    /// Keep centers inside the geometry and cap radii at the nearest surface distance plus
    /// `max_mesh_penetration`. The budget stays a hard maximum; uncovered geometry is allowed.
    pub enforce_mesh_bounds: bool,
    /// Hard maximum distance spheres may extend past the original mesh (meters) when
    /// `enforce_mesh_bounds` is enabled. Zero keeps spheres entirely inside closed geometry.
    pub max_mesh_penetration: f32,
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
        if !self.max_mesh_penetration.is_finite() || self.max_mesh_penetration < 0.0 {
            return Err(format!(
                "max_mesh_penetration must be finite and nonnegative, got {}",
                self.max_mesh_penetration
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
            enforce_mesh_bounds: false,
            max_mesh_penetration: 0.0,
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

/// Occupancy from oriented crossings, independent of the nearest-face inside heuristic.
/// Nonzero winding preserves overlapping outward shells (odd/even parity would not).
fn winding_inside(solid: &ParryMesh, point: Vec3) -> bool {
    use parry3d::partitioning::TraversalAction;
    use parry3d::query::{Ray, RayCast};

    let bounds = solid.local_aabb();
    let limit = 2.0 * (bounds.maxs - bounds.mins).length();
    let directions = [
        Vec3::new(1.0, 0.231, 0.177),
        Vec3::new(-0.371, 1.0, 0.529),
        Vec3::new(0.419, -0.217, 1.0),
    ];
    let mut inside = 0;
    for (index, direction) in directions.into_iter().enumerate() {
        let direction = direction.normalize();
        let ray = Ray::new(point, direction);
        let mut winding = 0i32;
        // Visit every intersected triangle once. Unlike repeated nearest-hit casts,
        // this needs no epsilon advance that could skip a very thin entry/exit pair.
        solid.bvh().traverse(|node| {
            if node.aabb().cast_local_ray(&ray, limit, true).is_none() {
                return TraversalAction::Prune;
            }
            if let Some(triangle) = node.leaf_data() {
                let triangle = solid.triangle(triangle);
                if triangle.cast_local_ray(&ray, limit, false).is_some() {
                    // RayIntersection.normal is ray-facing, not the original winding.
                    let facing = triangle.scaled_normal().dot(direction);
                    if facing > 0.0 {
                        winding += 1;
                    } else if facing < 0.0 {
                        winding -= 1;
                    }
                }
            }
            TraversalAction::Continue
        });
        inside += usize::from(winding != 0);
        if inside >= 2 {
            return true;
        }
        if index + 1 - inside >= 2 {
            return false;
        }
    }
    false
}

/// The largest sphere centered at `p` inside any of `solids`, if `p` is inside one.
fn inscribed(solids: &[ParryMesh], p: Vec3) -> Option<(Vec3, f32)> {
    use parry3d::query::PointQuery;

    solids
        .iter()
        .filter_map(|s| {
            // Oriented point queries can misclassify points far outside very thin solids.
            // Every interior point, and every sphere inside this solid, must also lie in
            // its AABB. Enforce both necessary conditions independently of that query.
            let bounds = s.local_aabb();
            let box_depth = (p - bounds.mins).min(bounds.maxs - p).min_element();
            if !box_depth.is_finite() || box_depth < 0.0 {
                return None;
            }
            let proj = s.project_local_point(p, false);
            if !proj.point.is_finite() {
                return None;
            }
            let radius = (p - proj.point).length();
            if !radius.is_finite() || !winding_inside(s, p) {
                return None;
            }
            Some(radius.min(box_depth))
        })
        .reduce(f32::max)
        .map(|r| (p, r))
}

/// A conservative bound against every original triangle, including component interfaces.
fn bounded_radius(solids: &[ParryMesh], point: Vec3, penetration: f32) -> Option<f32> {
    use parry3d::query::PointQuery;

    let (_, interior) = inscribed(solids, point)?;
    let nearest = solids
        .iter()
        .map(|solid| point.distance(solid.project_local_point(point, false).point))
        .fold(f32::INFINITY, f32::min);
    let limit = interior.min(nearest) + penetration;
    // Round inward, so serializing the f32 center/radius cannot increase the bound.
    let radius = limit.next_down();
    (radius.is_finite() && radius > 0.0).then_some(radius)
}

/// Validate bounded fits and supplied models against the original closed triangle surfaces.
pub(crate) fn validate_mesh_bounds(
    meshes: &[TriMesh],
    fitted: &[[f32; 4]],
    penetration: f32,
) -> Result<(), String> {
    if meshes.is_empty() {
        return if fitted.is_empty() {
            Ok(())
        } else {
            Err("bounded spheres need original collision geometry".into())
        };
    }
    let flags = TriMeshFlags::MERGE_DUPLICATE_VERTICES
        | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES
        | TriMeshFlags::ORIENTED;
    let mut solids = vec![];
    for mesh in meshes {
        if mesh.vertices.iter().any(|point| !point.is_finite()) {
            return Err("bounded sphere fitting needs finite mesh vertices".into());
        }
        let solid = ParryMesh::with_flags(mesh.vertices.clone(), mesh.triangles.clone(), flags)
            .map_err(|e| format!("invalid collision mesh for bounded spheres: {e}"))?;
        // Closed component cycles may touch and share edges. Each edge must still have
        // equally many forward/backward incidences; open or inconsistent winding is refused.
        let mut edges: BTreeMap<(u32, u32), (usize, i64)> = BTreeMap::new();
        for &[a, b, c] in solid.indices() {
            for (a, b) in [(a, b), (b, c), (c, a)] {
                let edge = edges.entry((a.min(b), a.max(b))).or_default();
                edge.0 += 1;
                edge.1 += if a < b { 1 } else { -1 };
            }
        }
        if edges.is_empty()
            || edges
                .values()
                .any(|&(count, balance)| count < 2 || balance != 0)
        {
            return Err("bounded sphere fitting requires closed, consistently wound meshes".into());
        }
        solids.push(solid);
    }
    if fitted.is_empty() {
        return Err(
            "bounded sphere fitting found no interior sphere for collision geometry".into(),
        );
    }
    for (index, &[x, y, z, radius]) in fitted.iter().enumerate() {
        let point = Vec3::new(x, y, z);
        let allowed = bounded_radius(&solids, point, penetration)
            .ok_or_else(|| format!("sphere {index} has no valid interior center"))?;
        if !radius.is_finite() || radius <= 0.0 || radius > allowed {
            return Err(format!(
                "sphere {index} radius {radius} exceeds the mesh penetration limit {penetration} (maximum radius {allowed})"
            ));
        }
    }
    Ok(())
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
    /// The geometry, as parry tests points against it, shared by a fit's copies.
    solids: Arc<[ParryMesh]>,
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
        let solids: Arc<[ParryMesh]> = meshes
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
        if candidates.is_empty() && o.enforce_mesh_bounds {
            // A thin closed part may fall between grid points. Search inward from its
            // surface instead; bounded mode never seeds growing spheres on the surface.
            let stride = samples.len().div_ceil(MAX_CANDIDATES).max(1);
            for (&point, &normal) in samples.iter().zip(&normals).step_by(stride) {
                if normal == Vec3::ZERO {
                    continue;
                }
                let mut depth = extent.max_element();
                while depth >= 1e-8 {
                    if let Some(candidate) = inscribed(&solids, point - normal * depth)
                        && candidate.1 > 0.0
                    {
                        candidates.push(candidate);
                        break;
                    }
                    depth *= 0.5;
                }
            }
        } else if candidates.is_empty() {
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
                        .map(|s| (p - s.project_local_point(p, false).point).length())
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
    fn spheres(&self, picks: &[usize], penetration: Option<f32>) -> (Vec<[f32; 4]>, f32) {
        // A cover that picked nothing still needs one sphere to grow: the largest candidate.
        let largest = (0..self.candidates.len())
            .max_by(|&a, &b| self.candidates[a].1.total_cmp(&self.candidates[b].1));
        let picks: Vec<usize> = if picks.is_empty() {
            largest.into_iter().collect()
        } else {
            picks.to_vec()
        };
        let mut spheres: Vec<(Vec3, f32)> = picks.iter().map(|&i| self.candidates[i]).collect();
        if spheres.is_empty() {
            return (vec![], 0.0);
        }
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
        // Cover/scaling chooses where to spend the budget. In bounded mode its scoring
        // slack never authorizes growth past the mesh: preserve the centers and cap each
        // selected radius independently, leaving coverage gaps instead of exceeding the cap.
        if let Some(penetration) = penetration {
            for (center, radius) in &mut spheres {
                *radius = bounded_radius(&self.solids, *center, penetration)
                    .map_or(0.0, |limit| radius.min(limit));
            }
        }
        let growth = picks
            .iter()
            .zip(&spheres)
            .map(|(&i, s)| s.1 - self.candidates[i].1)
            .fold(0.0, f32::max);
        (
            spheres
                .into_iter()
                .filter(|(_, r)| *r > 0.0)
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
            .map(|(f, picks)| {
                f.as_ref().map_or((vec![], 0.0), |f| {
                    f.spheres(
                        picks,
                        o.enforce_mesh_bounds.then_some(o.max_mesh_penetration),
                    )
                })
            })
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
    let fitted = fitted.unwrap_or_else(|| grown(&fits, &picks));
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
/// thin wide shape, where a fit of as many spheres is not.
pub(crate) fn cover_primitives(shapes: &[Obstacle], o: &SphereOptions) -> Option<Vec<[f32; 4]>> {
    if o.enforce_mesh_bounds {
        // Direct covers circumscribe cells and cannot honor an arbitrary hard cap.
        return None;
    }
    if shapes.iter().any(|s| matches!(s, Obstacle::Sdf { .. })) {
        return None;
    }
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
                let cells = box_cells(half_extents, o.tolerance, each);
                let cell = half_extents / cells.as_vec3();
                // A hair wide, so rounding never leaves a corner uncovered.
                let radius = cell.length() * HAIR;
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
                let own = cylinder_cover(radius, half_height, o.tolerance, each)?;
                placed(&mut spheres, center, rotation, own);
            }
            Obstacle::Capsule {
                center,
                rotation,
                radius,
                half_length,
            } => {
                let own = capsule_cover(radius, half_length, o.tolerance, each);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::description::Geometry;

    #[test]
    fn bounded_fits_keep_the_penetration_and_whole_robot_budget_hard() {
        let half = Vec3::new(0.08, 0.02, 0.01);
        let mesh = Geometry::Box { half }.mesh().expect("closed box");
        for penetration in [0.0, 0.002] {
            let options = SphereOptions {
                budget: 4,
                enforce_mesh_bounds: true,
                max_mesh_penetration: penetration,
                ..Default::default()
            };
            let links = vec![vec![mesh.clone()], vec![mesh.clone()]];
            let fitted = fit(&links, &[options.tolerance; 2], &options, false);
            assert!(fitted.spheres.iter().all(|spheres| !spheres.is_empty()));
            assert!(fitted.spheres.iter().map(Vec::len).sum::<usize>() <= options.budget);
            for spheres in &fitted.spheres {
                validate_mesh_bounds(std::slice::from_ref(&mesh), spheres, penetration)
                    .expect("every fitted sphere respects the original mesh");
                for &[x, y, z, radius] in spheres {
                    // Independent analytic distance to the six original box faces.
                    let depth = (half - Vec3::new(x, y, z).abs()).min_element();
                    assert!(depth >= 0.0, "center must stay inside the box");
                    assert!(
                        radius <= depth + penetration,
                        "radius {radius}, depth {depth}"
                    );
                }
            }
            let measured = measure(
                "box",
                std::slice::from_ref(&mesh),
                &fitted.spheres[0],
                &options,
            )
            .expect("qualification reports the remaining surface gaps");
            assert!(measured.uncovered > 0.0);
            assert!(measured.overhang <= penetration);
        }
        // The default remains the covering fitter: this tiny budget needs substantial growth.
        let legacy = SphereOptions {
            budget: 2,
            ..Default::default()
        };
        let fitted = fit(&[vec![mesh]], &[legacy.tolerance], &legacy, false);
        assert!(fitted.spheres[0].iter().any(|&[x, y, z, radius]| {
            radius > (half - Vec3::new(x, y, z).abs()).min_element() + 0.002
        }));
    }

    #[test]
    fn bounds_use_every_original_surface_and_refuse_open_meshes() {
        let half = Vec3::splat(0.02);
        let first = Geometry::Box { half }.mesh().expect("box");
        let mut second = first.clone();
        for point in &mut second.vertices {
            point.x += 0.01;
        }
        let meshes = [first.clone(), second];
        // This fits the first box but crosses an original face of the overlapping second box.
        assert!(validate_mesh_bounds(&meshes, &[[0.0, 0.0, 0.0, 0.015]], 0.0).is_err());
        validate_mesh_bounds(&meshes, &[[0.0, 0.0, 0.0, 0.009]], 0.0)
            .expect("sphere stays on the interior side of all original surfaces");
        let mut open = first;
        open.triangles.pop();
        let error = validate_mesh_bounds(&[open], &[[0.0, 0.0, 0.0, 0.001]], 0.0)
            .expect_err("open geometry cannot certify a bounded sphere");
        assert!(error.contains("closed"), "{error}");
    }

    #[test]
    fn bounded_thin_fits_find_inward_centers_and_invalid_limits_are_rejected() {
        let mesh = thin_tetrahedron();
        let options = SphereOptions {
            budget: 8,
            enforce_mesh_bounds: true,
            ..Default::default()
        };
        let fitted = fit(&[vec![mesh.clone()]], &[options.tolerance], &options, false);
        assert!(
            !fitted.spheres[0].is_empty(),
            "thin closed geometry needs interior seeds"
        );
        assert!(fitted.spheres[0].len() <= options.budget);
        validate_mesh_bounds(&[mesh], &fitted.spheres[0], 0.0).expect("thin fit stays inside");
        for invalid in [-0.001, f32::NAN, f32::INFINITY] {
            assert!(
                SphereOptions {
                    max_mesh_penetration: invalid,
                    ..options
                }
                .check()
                .is_err()
            );
        }
        assert!(
            cover_primitives(&[Obstacle::cuboid(Vec3::ZERO, Vec3::splat(0.1))], &options).is_none(),
            "bounded boxes must bypass circumscribed cell spheres"
        );
    }

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
    fn thin_tetrahedron() -> TriMesh {
        TriMesh {
            vertices: vec![
                Vec3::new(-0.05, -0.02, 0.0),
                Vec3::new(0.05, -0.02, 0.0),
                Vec3::new(0.0, 0.02, 0.0),
                Vec3::new(0.0, 0.0, 1e-5),
            ],
            triangles: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        }
    }

    #[test]
    fn an_oriented_sliver_cannot_claim_a_distant_point_or_hide_its_overhang() {
        // A closed 0.1 mm prism reproducing Parry's false interior classification.
        let mesh = TriMesh {
            vertices: vec![
                Vec3::new(0.04805384, 0.12926735, 0.09413833),
                Vec3::new(0.048095737, 0.12927745, 0.0951274),
                Vec3::new(0.048110154, 0.12929678, 0.095489144),
                Vec3::new(0.048209894, 0.12930256, 0.09548485),
                Vec3::new(0.048195478, 0.12928323, 0.095123105),
                Vec3::new(0.04815358, 0.12927313, 0.09413404),
            ],
            triangles: vec![
                [0, 1, 2],
                [3, 4, 5],
                [0, 5, 4],
                [0, 4, 1],
                [1, 4, 3],
                [1, 3, 2],
                [2, 3, 5],
                [2, 5, 0],
            ],
        };
        let point = Vec3::new(-0.13717227, -0.13240308, -0.080591865);
        let solid = ParryMesh::with_flags(
            mesh.vertices.clone(),
            mesh.triangles.clone(),
            TriMeshFlags::MERGE_DUPLICATE_VERTICES
                | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES
                | TriMeshFlags::ORIENTED,
        )
        .expect("closed sliver prism");
        assert!(inscribed(std::slice::from_ref(&solid), point).is_none());
        let qualification = measure(
            "sliver",
            &[mesh],
            &[[point.x, point.y, point.z, 1e-9]],
            &SphereOptions::default(),
        )
        .expect("sliver surface");
        assert!(
            qualification.overhang > 0.3,
            "distant sphere's actual overhang was hidden: {qualification:?}"
        );
    }

    #[test]
    fn the_retention_sliver_rejects_a_false_interior_point_within_its_aabb() {
        let mesh = TriMesh {
            vertices: vec![
                Vec3::new(-0.09312411, 0.10867787, 0.09173776),
                Vec3::new(-0.09360698, 0.107881434, 0.09120643),
                Vec3::new(-0.09330204, 0.10837083, 0.09153486),
                Vec3::new(-0.09328164, 0.1084163, 0.091448165),
                Vec3::new(-0.09358658, 0.1079269, 0.09111974),
                Vec3::new(-0.09310371, 0.108723335, 0.09165107),
            ],
            triangles: vec![
                [0, 1, 2],
                [3, 4, 5],
                [0, 5, 4],
                [0, 4, 1],
                [1, 4, 3],
                [1, 3, 2],
                [2, 3, 5],
                [2, 5, 0],
            ],
        };
        let solid = ParryMesh::with_flags(mesh.vertices, mesh.triangles, TriMeshFlags::ORIENTED)
            .expect("actual closed retention sliver");
        let point = Vec3::new(-0.09329422, 0.10790725, 0.09129702);
        let bounds = solid.local_aabb();
        assert!(point.cmpge(bounds.mins).all() && point.cmple(bounds.maxs).all());
        assert!(!winding_inside(&solid, point));
        assert!(inscribed(std::slice::from_ref(&solid), point).is_none());
    }

    #[test]
    fn a_point_inside_a_thin_mesh_bounds_but_outside_its_footprint_is_rejected() {
        let mesh = thin_tetrahedron();
        let solid = ParryMesh::with_flags(mesh.vertices, mesh.triangles, TriMeshFlags::ORIENTED)
            .expect("closed thin tetrahedron");
        assert!(inscribed(std::slice::from_ref(&solid), Vec3::new(0.04, 0.015, 1e-6)).is_none());
        assert!(inscribed(std::slice::from_ref(&solid), Vec3::new(0.0, 0.0, 1e-6)).is_some());
    }

    #[test]
    fn oriented_winding_rejects_concave_gaps_and_accepts_overlapping_solids() {
        use crate::description::Geometry;
        let cuboid = |center: Vec3, half: Vec3| {
            let mut mesh = Geometry::Box { half }.mesh().expect("box surface");
            for point in &mut mesh.vertices {
                *point += center;
            }
            mesh
        };
        let mut u = cuboid(Vec3::new(-0.04, 0.0, 0.0), Vec3::new(0.01, 0.05, 0.01));
        u.append(cuboid(
            Vec3::new(0.04, 0.0, 0.0),
            Vec3::new(0.01, 0.05, 0.01),
        ));
        u.append(cuboid(
            Vec3::new(0.0, -0.04, 0.0),
            Vec3::new(0.04, 0.01, 0.01),
        ));
        let solid = ParryMesh::with_flags(u.vertices, u.triangles, TriMeshFlags::ORIENTED)
            .expect("U-shaped component union");
        assert!(inscribed(std::slice::from_ref(&solid), Vec3::ZERO).is_none());
        assert!(inscribed(std::slice::from_ref(&solid), Vec3::new(-0.04, 0.0, 0.0)).is_some());
        let mut overlap = cuboid(Vec3::new(-0.005, 0.0, 0.0), Vec3::splat(0.02));
        overlap.append(cuboid(Vec3::new(0.005, 0.0, 0.0), Vec3::splat(0.02)));
        let solid =
            ParryMesh::with_flags(overlap.vertices, overlap.triangles, TriMeshFlags::ORIENTED)
                .expect("overlapping outward shells");
        assert!(
            winding_inside(&solid, Vec3::ZERO),
            "winding two is occupied"
        );
        let inside_one = Vec3::new(-0.016, 0.0, 0.0);
        assert!(
            winding_inside(&solid, inside_one),
            "a point inside one shell stays occupied near another shell's exterior face"
        );
        assert!(inscribed(std::slice::from_ref(&solid), inside_one).is_some());
    }

    #[test]
    fn touching_component_shells_do_not_fill_their_concave_gap() {
        use crate::description::Geometry;
        let mut joined = Geometry::Box {
            half: Vec3::new(0.02, 0.02, 0.01),
        }
        .mesh()
        .expect("first box");
        for point in &mut joined.vertices {
            *point += Vec3::new(-0.02, -0.02, 0.0);
        }
        let mut second = Geometry::Box {
            half: Vec3::new(0.02, 0.02, 0.01),
        }
        .mesh()
        .expect("touching box");
        for point in &mut second.vertices {
            *point += Vec3::new(0.02, 0.02, 0.0);
        }
        joined.append(second);
        let solid =
            ParryMesh::with_flags(joined.vertices, joined.triangles, TriMeshFlags::ORIENTED)
                .expect("touching closed components");
        assert!(!winding_inside(&solid, Vec3::new(-0.02, 0.02, 0.0)));
        assert!(winding_inside(&solid, Vec3::new(0.02, 0.02, 0.0)));
    }

    #[test]
    fn thin_solids_never_propose_spheres_outside_their_own_bounds() {
        let mesh = thin_tetrahedron();
        let solid = ParryMesh::with_flags(mesh.vertices, mesh.triangles, TriMeshFlags::ORIENTED)
            .expect("closed thin tetrahedron");
        for point in [
            Vec3::new(0.0, 0.0, 0.4),
            Vec3::new(0.3, 0.0, 1e-6),
            Vec3::new(0.0, -0.2, 1e-6),
        ] {
            assert!(inscribed(std::slice::from_ref(&solid), point).is_none());
        }
        let point = Vec3::new(0.0, 0.0, 1e-6);
        if let Some((center, radius)) = inscribed(std::slice::from_ref(&solid), point) {
            let bounds = solid.local_aabb();
            assert!((center - Vec3::splat(radius)).cmpge(bounds.mins).all());
            assert!((center + Vec3::splat(radius)).cmple(bounds.maxs).all());
        }
    }

    #[test]
    fn a_thin_solid_fit_does_not_turn_into_a_large_distant_sphere() {
        let mesh = thin_tetrahedron();
        let options = SphereOptions {
            budget: 64,
            tolerance: 0.005,
            ..SphereOptions::default()
        };
        let fitted = fit(&[vec![mesh]], &[options.tolerance], &options, false);
        assert!(!fitted.spheres[0].is_empty());
        assert!(fitted.spheres[0].len() <= options.budget);
        // Surface fallback and tangent growth may overhang a thin solid; they must not
        // inherit a hundreds-of-millimetres radius from a false interior point query.
        for sphere in &fitted.spheres[0] {
            assert!(sphere[3] < 0.05, "unbounded thin-solid sphere: {sphere:?}");
            assert!(sphere[..3].iter().all(|coordinate| coordinate.is_finite()));
        }
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
