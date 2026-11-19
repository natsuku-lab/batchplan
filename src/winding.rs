//! Which points of a grid a closed triangle surface winds around.
//!
//! The winding number of a closed surface around a point counts how often the surface encloses
//! it, with its facing: 1 inside a shell that faces out, 2 where two such shells overlap, 0
//! outside and in a cavity (a shell facing in, inside one facing out). Unlike the normal of the
//! nearest face, it is defined where shells overlap or cross.
//!
//! One ray along +x per row of the grid counts every point of the row ([`Crossings`]). Each
//! triangle is projected onto the y-z plane and tested against the rows it spans. Vertices and
//! rows are snapped to one integer lattice, so the tests are exact: a ray through an edge counts
//! once for the triangles that share it, a ray through a vertex once for the fan around it, and
//! a triangle seen edge-on never. The snap moves the surface by a billionth of the grid's size,
//! which only points on the surface can tell.
//!
//! [`Surface`] counts the same way around one point at a time.

use glam::Vec3;
use rayon::prelude::*;

/// Lattice steps across the rows and vertices, along y or z, whichever they span farther: few
/// enough that twice the product of two differences fits an `i64`.
const STEPS: f64 = (1u64 << 30) as f64;

/// Meshes of fewer triangles are crossed on the calling thread: handing so few to the pool costs
/// more than crossing them.
const CROSSED_ALONE: usize = 1 << 12;

/// A ray's crossing of a triangle.
struct Crossing {
    /// The ray: the row's y index plus the y count times its z index.
    row: u32,
    /// Where along the ray (meters, in the grid's frame).
    x: f64,
    /// 1 where the triangle faces +x, -1 where it faces -x.
    facing: i8,
}

/// Where the rays along the rows of a grid cross a triangle surface.
pub(crate) struct Crossings {
    /// Row by row, and along each row.
    crossings: Vec<Crossing>,
    /// Where each row's begin among them, and where the last row's end.
    starts: Vec<u32>,
}

impl Crossings {
    /// The crossings of the rows of a grid whose points have the coordinates `ys` and `zs` along
    /// those axes, each ascending: row `j + ys.len() * k` runs along x through `ys[j]` and
    /// `zs[k]`. `triangles` must index `vertices`. With `parallel`, a mesh of [`CROSSED_ALONE`]
    /// triangles or more is crossed on the current rayon pool; without, on the calling thread
    /// alone, whatever its size.
    pub(crate) fn new(
        vertices: &[Vec3],
        triangles: &[[u32; 3]],
        ys: &[f32],
        zs: &[f32],
        parallel: bool,
    ) -> Self {
        // One lattice for the rows and the vertices, from the least of each along y and z.
        let span = |rows: &[f32], axis: usize| {
            let of_vertices = vertices.iter().map(|v| v[axis]);
            let all = of_vertices.chain(rows.iter().copied());
            all.fold((f32::INFINITY, f32::NEG_INFINITY), |(least, most), v| {
                (least.min(v), most.max(v))
            })
        };
        let (in_y, in_z) = (span(ys, 1), span(zs, 2));
        let from = [in_y.0, in_z.0];
        let longest =
            (f64::from(in_y.1) - f64::from(in_y.0)).max(f64::from(in_z.1) - f64::from(in_z.0));
        let scale = if longest > 0.0 { STEPS / longest } else { 1.0 };
        let snap = |v: f32, from: f32| ((f64::from(v) - f64::from(from)) * scale).round() as i64;
        let rows_y: Vec<i64> = ys.iter().map(|&y| snap(y, from[0])).collect();
        let rows_z: Vec<i64> = zs.iter().map(|&z| snap(z, from[1])).collect();
        let count_y = ys.len() as u32;

        let (rows_y, rows_z) = (&rows_y, &rows_z);
        let crossed = |t: &[u32; 3]| {
            let corner = |i: u32| {
                let v = vertices[i as usize];
                ([snap(v.y, from[0]), snap(v.z, from[1])], f64::from(v.x))
            };
            let ((a, ax), (b, bx), (c, cx)) = (corner(t[0]), corner(t[1]), corner(t[2]));
            let area = edge(a, b, c);
            let rows = |of: &[i64], axis: usize| {
                let (lo, hi) = (
                    a[axis].min(b[axis]).min(c[axis]),
                    a[axis].max(b[axis]).max(c[axis]),
                );
                of.partition_point(|&r| r < lo)..of.partition_point(|&r| r <= hi)
            };
            // Edge-on, a triangle has no rows.
            let (in_y, in_z) = if area == 0 {
                (0..0, 0..0)
            } else {
                (rows(rows_y, 0), rows(rows_z, 1))
            };
            in_z.flat_map(move |k| {
                let in_y = in_y.clone();
                in_y.filter_map(move |j| {
                    let p = [rows_y[j], rows_z[k]];
                    let facing = area.signum();
                    let within = [(b, c), (c, a), (a, b)].map(|(u, v)| side(u, v, p) == facing);
                    if within != [true; 3] {
                        return None;
                    }
                    // The plane's x at the ray, from the corners' weights there.
                    let weights = [edge(b, c, p), edge(c, a, p), edge(a, b, p)];
                    let x =
                        (weights[0] as f64 * ax + weights[1] as f64 * bx + weights[2] as f64 * cx)
                            / area as f64;
                    Some(Crossing {
                        row: j as u32 + count_y * k as u32,
                        x,
                        facing: facing as i8,
                    })
                })
            })
        };
        let by_row = |a: &Crossing, b: &Crossing| a.row.cmp(&b.row).then(a.x.total_cmp(&b.x));
        let crossings = if triangles.len() < CROSSED_ALONE || !parallel {
            let mut crossings: Vec<Crossing> = triangles.iter().flat_map(crossed).collect();
            crossings.sort_unstable_by(by_row);
            crossings
        } else {
            let mut crossings: Vec<Crossing> =
                triangles.par_iter().flat_map_iter(crossed).collect();
            crossings.par_sort_unstable_by(by_row);
            crossings
        };
        let mut starts = vec![0u32; ys.len() * zs.len() + 1];
        for crossing in &crossings {
            starts[crossing.row as usize + 1] += 1;
        }
        for row in 1..starts.len() {
            starts[row] += starts[row - 1];
        }
        Self { crossings, starts }
    }

    /// The winding number of the surface around each point of `row` whose x is in `xs`,
    /// ascending: a point is wound by the crossings its ray still has ahead of it.
    pub(crate) fn numbers<'a>(
        &'a self,
        row: usize,
        xs: &'a [f32],
    ) -> impl Iterator<Item = i32> + 'a {
        let ahead = &self.crossings[self.starts[row] as usize..self.starts[row + 1] as usize];
        let mut left: i32 = ahead.iter().map(|c| i32::from(c.facing)).sum();
        let mut passed = 0;
        xs.iter().map(move |&x| {
            while passed < ahead.len() && ahead[passed].x <= f64::from(x) {
                left -= i32::from(ahead[passed].facing);
                passed += 1;
            }
            left
        })
    }
}

/// A triangle surface on the lattice of its own vertices, for the winding number around one point
/// at a time: what [`Crossings`] counts for a row through the point, without the rest of a grid.
pub(crate) struct Surface<'a> {
    triangles: &'a [[u32; 3]],
    /// Each vertex snapped along y and z, and its x.
    corners: Vec<([i64; 2], f64)>,
    /// The vertices' least y and z, where the lattice starts.
    from: [f32; 2],
    /// Lattice steps per meter.
    scale: f64,
    /// The vertices' greatest y and z on the lattice.
    most: [i64; 2],
}

impl<'a> Surface<'a> {
    /// The surface of `triangles`, which must index `vertices`.
    pub(crate) fn new(vertices: &[Vec3], triangles: &'a [[u32; 3]]) -> Self {
        let span = |axis: usize| {
            let along = vertices.iter().map(|v| v[axis]);
            along.fold((f32::INFINITY, f32::NEG_INFINITY), |(least, most), v| {
                (least.min(v), most.max(v))
            })
        };
        let (in_y, in_z) = (span(1), span(2));
        let from = [in_y.0, in_z.0];
        let longest =
            (f64::from(in_y.1) - f64::from(in_y.0)).max(f64::from(in_z.1) - f64::from(in_z.0));
        let scale = if longest > 0.0 { STEPS / longest } else { 1.0 };
        let snap = |v: f32, from: f32| ((f64::from(v) - f64::from(from)) * scale).round() as i64;
        let corner = |v: &Vec3| ([snap(v.y, from[0]), snap(v.z, from[1])], f64::from(v.x));
        Self {
            triangles,
            corners: vertices.iter().map(corner).collect(),
            from,
            scale,
            most: [snap(in_y.1, from[0]), snap(in_z.1, from[1])],
        }
    }

    /// The winding number of the surface around `p`: the facings of the triangles the ray from
    /// it along +x still crosses.
    pub(crate) fn number(&self, p: Vec3) -> i32 {
        let snap = |v: f32, from: f32| ((f64::from(v) - f64::from(from)) * self.scale).round();
        let on = [snap(p.y, self.from[0]), snap(p.z, self.from[1])];
        // Beside the surface along y or z, the ray crosses nothing (and a point that is not a
        // number is beside it).
        let beside = |axis: usize| !(0.0..=self.most[axis] as f64).contains(&on[axis]);
        if beside(0) || beside(1) {
            return 0;
        }
        let on = on.map(|v| v as i64);
        let mut number = 0;
        for t in self.triangles {
            let [(a, ax), (b, bx), (c, cx)] = t.map(|i| self.corners[i as usize]);
            let beside = |axis: usize| {
                on[axis] < a[axis].min(b[axis]).min(c[axis])
                    || on[axis] > a[axis].max(b[axis]).max(c[axis])
            };
            if beside(0) || beside(1) {
                continue;
            }
            let area = edge(a, b, c);
            let facing = area.signum();
            // Edge-on, a triangle is not crossed.
            if area == 0
                || [(b, c), (c, a), (a, b)]
                    .iter()
                    .any(|&(u, v)| side(u, v, on) != facing)
            {
                continue;
            }
            let weights = [edge(b, c, on), edge(c, a, on), edge(a, b, on)];
            let x = (weights[0] as f64 * ax + weights[1] as f64 * bx + weights[2] as f64 * cx)
                / area as f64;
            if x > f64::from(p.x) {
                number += facing as i32;
            }
        }
        number
    }
}

/// Twice the signed area of the triangle `u`, `v`, `p` in the y-z plane: positive where `p` is
/// left of the edge from `u` to `v`.
fn edge(u: [i64; 2], v: [i64; 2], p: [i64; 2]) -> i64 {
    (v[0] - u[0]) * (p[1] - u[1]) - (v[1] - u[1]) * (p[0] - u[0])
}

/// Which side of the edge from `u` to `v` the point `p` is on, 1 for left and -1 for right. A
/// point on the edge takes the side it has moved a hair along +y, then a smaller hair along +z:
/// the two triangles that share an edge then see it on opposite sides, and one of them holds it.
fn side(u: [i64; 2], v: [i64; 2], p: [i64; 2]) -> i64 {
    let on = edge(u, v, p);
    if on != 0 {
        on.signum()
    } else if v[1] != u[1] {
        (u[1] - v[1]).signum()
    } else {
        (v[0] - u[0]).signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// One point at a time counts what a row through the point counts: over closed and open
    /// soups, at random points and at points on the lattice's own lines through vertices, where
    /// a ray runs through edges and corners.
    #[test]
    fn a_surface_counts_around_a_point_what_crossings_count_along_its_row() {
        let mut rng = Rng::new(3);
        let mut differing = 0;
        let mut wound = 0;
        for soup in 0..40 {
            // Boxes, some turned inside out, some missing a face, overlapping at random.
            let mut vertices = vec![];
            let mut triangles: Vec<[u32; 3]> = vec![];
            for shell in 0..1 + soup % 3 {
                let center = Vec3::new(rng.range(-0.1, 0.1), rng.range(-0.1, 0.1), 0.0);
                let half = Vec3::new(rng.range(0.05, 0.2), rng.range(0.05, 0.2), 0.1);
                let base = vertices.len() as u32;
                for k in 0..8 {
                    let sign = |bit: u32| if k >> bit & 1 == 1 { 1.0 } else { -1.0 };
                    vertices.push(center + Vec3::new(sign(0), sign(1), sign(2)) * half);
                }
                let faces = [
                    [0, 4, 6, 2],
                    [1, 3, 7, 5],
                    [0, 1, 5, 4],
                    [2, 6, 7, 3],
                    [0, 2, 3, 1],
                    [4, 5, 7, 6],
                ];
                let open = usize::from(soup % 5 == 4);
                for [a, b, c, d] in faces.into_iter().skip(open) {
                    let [a, b, c, d] = [a, b, c, d].map(|corner| corner + base);
                    if (soup + shell) % 4 == 3 {
                        triangles.extend([[a, c, b], [a, d, c]]);
                    } else {
                        triangles.extend([[a, b, c], [a, c, d]]);
                    }
                }
            }
            let surface = Surface::new(&vertices, &triangles);
            for k in 0..400 {
                // Through a vertex along y, z or both, or anywhere in and around the boxes.
                let v = vertices[k % vertices.len()];
                let free = Vec3::new(
                    rng.range(-0.4, 0.4),
                    rng.range(-0.4, 0.4),
                    rng.range(-0.2, 0.2),
                );
                let p = match k % 4 {
                    0 => Vec3::new(free.x, v.y, v.z),
                    1 => Vec3::new(free.x, v.y, free.z),
                    2 => Vec3::new(free.x, free.y, v.z),
                    _ => free,
                };
                let row = Crossings::new(
                    &vertices,
                    &triangles,
                    &[p.y],
                    &[p.z],
                    /* parallel */ false,
                );
                let along = row.numbers(0, &[p.x]).next().expect("one point");
                let around = surface.number(p);
                differing += usize::from(along != around);
                wound += usize::from(around != 0);
            }
        }
        assert_eq!(differing, 0, "points counted differently");
        assert!(wound > 2000, "{wound} points inside a shell");
    }
}
