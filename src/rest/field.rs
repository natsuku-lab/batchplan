//! A surface as [`rests_on`](super::rests_on) reads it: its support as a height field seen from
//! above, and everything else as a second one that blocks.

use glam::{Vec2, Vec3};

use super::shape::Turned;
use crate::surface::Surface;

/// The most cells a surface is kept in: past it, `rests_on` asks for wider ones.
pub(super) const MOST_CELLS: f32 = (1 << 22) as f32;

/// A support point whose normal's z is under this faces down: it holds nothing up, and blocks.
/// 30 degrees under level, so that the noise of a wall's normals, which are level, does not
/// turn its points into undersides.
const FACING: f32 = -0.5;

/// A surface's points in cells seen from above, each cell with the highest point in it. The cells
/// are centred on whole multiples of their width in the surface's frame, so which cell a point
/// falls in does not depend on what else is in the surface or the batch.
pub(super) struct Field {
    /// The cells' width (meters).
    pub(super) cell: f32,
    /// The cell the first column and row hold, as a multiple of the width.
    first: [i32; 2],
    across: usize,
    rows: usize,
    /// Per cell, the highest support point in it, where it is; its height is negative infinity
    /// where the cell holds none.
    tops: Vec<Vec3>,
    /// Per cell, the height of the lowest support point in it: from there up to the highest the
    /// support was seen, as the side of a wall is.
    floors: Vec<f32>,
    /// Per cell, the highest point in it that is not the support's, or is the support's, faces
    /// down and has no support point of the cell over it.
    blocks: Vec<Vec3>,
}

impl Field {
    /// How many cells `cell` wide a support between `lower` and `upper`, seen from above, is
    /// kept in with the `margin` around it.
    pub(super) fn cells(lower: Vec2, upper: Vec2, cell: f32, margin: f32) -> f32 {
        ((upper - lower + 2.0 * margin) / cell + 3.0).element_product()
    }

    /// `s` read once, its support between `lower` and `upper` seen from above, in cells
    /// `cell` wide, with what else of it lies within `margin` of the support.
    pub(super) fn new(s: &Surface, lower: Vec2, upper: Vec2, cell: f32, margin: f32) -> Self {
        let first = ((lower - margin) / cell).round().as_ivec2() - 1;
        let last = ((upper + margin) / cell).round().as_ivec2() + 1;
        let (across, rows) = (
            (last.x - first.x + 1) as usize,
            (last.y - first.y + 1) as usize,
        );
        let empty = Vec3::new(0.0, 0.0, f32::NEG_INFINITY);
        let mut field = Self {
            cell,
            first: first.to_array(),
            across,
            rows,
            tops: vec![empty; across * rows],
            floors: vec![f32::INFINITY; across * rows],
            blocks: vec![empty; across * rows],
        };
        // The support first, then what blocks: a support point that faces down under a higher
        // one of its cell is the underside of what that one is the top of, as a bar seen all
        // round, and is no more in the way than the bar is.
        for holding in [true, false] {
            for (k, &p) in s.points.iter().enumerate() {
                let (column, row) = (field.column(p.x), field.row(p.y));
                if column < 0 || row < 0 || column >= across as i32 || row >= rows as i32 {
                    continue;
                }
                let cell = row as usize * across + column as usize;
                let support = s.object.is_empty() || s.object[k];
                let holds = support && s.normals[k].z >= FACING;
                if holds != holding || (support && !holds && p.z <= field.tops[cell].z) {
                    continue;
                }
                if holds {
                    field.floors[cell] = field.floors[cell].min(p.z);
                }
                let layer = if holds {
                    &mut field.tops
                } else {
                    &mut field.blocks
                };
                if p.z > layer[cell].z {
                    layer[cell] = p;
                }
            }
        }
        field
    }

    /// The column `x` falls in, counted from the first; outside the field it is not one.
    #[inline]
    fn column(&self, x: f32) -> i32 {
        (x / self.cell).round() as i32 - self.first[0]
    }

    #[inline]
    fn row(&self, y: f32) -> i32 {
        (y / self.cell).round() as i32 - self.first[1]
    }

    /// The columns and the rows whose cells may hold a point within `reach` of `at`, seen from
    /// above.
    #[inline]
    pub(super) fn window(&self, at: Vec2, reach: Vec2) -> [std::ops::Range<usize>; 2] {
        let clamp = |v: i32, most: usize| v.clamp(0, most as i32) as usize;
        [
            clamp(self.column(at.x - reach.x), self.across)
                ..clamp(self.column(at.x + reach.x) + 1, self.across),
            clamp(self.row(at.y - reach.y), self.rows)
                ..clamp(self.row(at.y + reach.y) + 1, self.rows),
        ]
    }

    /// How high the middle of `object` comes to rest set straight down over `at`: the least height
    /// at which none of the support is in it. Negative infinity where no support is under it.
    pub(super) fn set_down(&self, object: &Turned, at: Vec2) -> f32 {
        let [columns, rows] = self.window(at, object.reach);
        // No point lower than this under the height found so far can raise it.
        let deepest = object.depth;
        let mut height = f32::NEG_INFINITY;
        for row in rows {
            let cells = &self.tops[row * self.across..][columns.clone()];
            for p in cells {
                if p.z + deepest <= height {
                    continue;
                }
                if let Some((lower, _)) = object.span(p.truncate() - at) {
                    height = height.max(p.z - lower);
                }
            }
        }
        height
    }

    /// Whether `grown`, an object grown by the clearance it keeps, with its middle at `center`, is
    /// over everything that blocks: nothing that is not the support reaches into it or over it.
    pub(super) fn clear(&self, grown: &Turned, center: Vec3) -> bool {
        let at = center.truncate();
        let [columns, rows] = self.window(at, grown.reach);
        for row in rows {
            let cells = &self.blocks[row * self.across..][columns.clone()];
            for p in cells {
                if p.z <= center.z - grown.shape.radius() {
                    continue;
                }
                if let Some((lower, _)) = grown.span(p.truncate() - at)
                    && p.z > center.z + lower
                {
                    return false;
                }
            }
        }
        true
    }

    /// The highest support point in the cell at `column` and `row`; its height is negative
    /// infinity where the cell holds none.
    #[inline]
    pub(super) fn top(&self, column: usize, row: usize) -> Vec3 {
        self.tops[row * self.across + column]
    }

    /// The height of the lowest support point in the cell at `column` and `row`.
    #[inline]
    pub(super) fn floor(&self, column: usize, row: usize) -> f32 {
        self.floors[row * self.across + column]
    }

    /// The highest support point in the cell at `column` and `row`, if the field has that cell
    /// and it holds one.
    pub(super) fn top_at(&self, column: usize, row: usize) -> Option<Vec3> {
        let inside = column < self.across && row < self.rows;
        let top = inside.then(|| self.tops[row * self.across + column])?;
        (top.z > f32::NEG_INFINITY).then_some(top)
    }
}
