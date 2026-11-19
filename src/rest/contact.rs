//! What of the support pushes on an object set down on it, and which way.
//!
//! The support's points within [`touching`](super::RestOptions::touching) under the object are
//! taken patch by patch: the cells that touch one piece of its surface (a face, an end, a round
//! side) and adjoin each other. Whether a point so near pushes is not the point's to say: a can
//! laid on a table has points that near under 2 cm of its side, and a box stood on an edge has
//! them under both faces beside it. So the support under a patch is taken as the plane through
//! its points, as the engines take a contact's direction from the plane or the face a shape
//! lies on and only then ask which of its points are near enough to count (MuJoCo's and ODE's
//! plane colliders, the clipping of one box's face against another's). The piece touches with
//! all of the patch only where all of it that lies over that stretch of support stays within
//! `touching` of the plane, each point of the patch pushing where it is. Elsewhere the piece
//! touches along the edge, the line or the point of it that reaches farthest toward the plane: a
//! box on an edge at the edge's two ends, a can on its side at the two ends of its lowest line.
//! Either way the push is along the plane's normal, which many points give far better than any
//! one of them: the normals the surface came with are not read. A patch that lies away from that
//! edge is a ridge of the support under the piece, as the top of a lip under a face: its points
//! push where they are, along the object's own normal, the one side there that has one.
//!
//! The support is solid under its highest points, so the side of a step pushes too, level,
//! against what stands or leans beside it, as far down as it was seen.

use glam::Vec3;

use super::field::Field;
use super::shape::{Meets, Piece, ROUND, Turned};
use super::statics::Contact;

/// A support point under the object, near enough to tell how the support lies there.
#[derive(Clone, Copy)]
struct Near {
    point: Vec3,
    /// The point of the object's underside over it, in the surface's frame and in the object's.
    over: Vec3,
    local: Vec3,
    /// The unit direction out of the object there.
    out: Vec3,
    piece: Piece,
    /// How far under the object it lies, and whether that is within `touching`.
    gap: f32,
    touches: bool,
    /// Its cell in the window around the object, and in the field.
    cell: u32,
    column: u32,
    row: u32,
    /// The patch it is of: a point of it, which every point of the patch leads to.
    patch: u32,
}

/// What a thread keeps between the poses it reads contacts of.
#[derive(Default)]
pub(super) struct Touch {
    near: Vec<Near>,
    /// Per cell of the window around the object, the point in it, counted from one; none, zero.
    cells: Vec<u32>,
    patches: Vec<(u32, u32)>,
    /// The points of the patches under the object's round side, together.
    rounds: Vec<(u32, u32)>,
    fitted: Vec<Vec3>,
    around: Vec<(u32, u32)>,
    /// The points reached from a patch over one stretch of support, and those still to go on
    /// from.
    reached: Vec<bool>,
    front: Vec<u32>,
}

// The thresholds below are in `touching`s, so they follow the options; their values are chosen,
// not derived. `tests/rest.rs` holds each this far: halved, `AROUND` fails its dish and groove,
// `STEP` its steep slopes and `TIGHT` its log; doubled, and `ASIDE` both ways, no test tells.

/// How many times `touching` under the object a support point may lie and still tell how the
/// support lies under a patch too narrow to tell by itself.
const AROUND: f32 = 3.0;

/// How many times `touching` higher or lower than a patch a support point two cells from it may
/// lie and be of the same stretch of support.
const STEP: f32 = 2.5;

/// How much less squarely than another piece of the object a piece may face a plane, as the
/// difference of the cosines, and still be what meets it: past it, the piece stands beside the
/// one that does.
const ASIDE: f32 = 0.2;

/// How much of `touching` farther under the object than the nearest of a patch a point of it may
/// lie and be where the object meets the support's end.
const TIGHT: f32 = 0.25;

/// How near the edge it would touch along, in cells seen from above, a patch must come to touch
/// along it: in the edge's cell or one beside it, cornerwise too.
const REACH: f32 = std::f32::consts::SQRT_2;

impl Touch {
    /// The contacts of `object`, set down with its middle at `center`, with the support in
    /// `field`, into `contacts`.
    pub(super) fn contacts(
        &mut self,
        field: &Field,
        object: &Turned,
        center: Vec3,
        touching: f32,
        contacts: &mut Vec<Contact>,
    ) {
        contacts.clear();
        self.near.clear();
        let at = center.truncate();
        let [columns, rows] = field.window(at, object.reach + touching);
        let across = columns.len();
        self.cells.clear();
        self.cells.resize(across * rows.len(), 0);
        let lowest = center.z - object.shape.radius();
        for (j, row) in rows.enumerate() {
            for (i, column) in columns.clone().enumerate() {
                let point = field.top(column, row);
                if point.z < lowest {
                    continue;
                }
                let offset = point.truncate() - at;
                let Some((lower, _)) = object.span(offset) else {
                    let seen = field.floor(column, row);
                    beside(object, center, point, seen, touching, contacts);
                    continue;
                };
                let gap = center.z + lower - point.z;
                if gap > AROUND * touching {
                    continue;
                }
                let local = object.local(offset, lower);
                let (piece, out) = object.piece_at(local);
                let k = self.near.len() as u32;
                self.near.push(Near {
                    point,
                    over: point.with_z(center.z + lower),
                    local,
                    out,
                    piece,
                    gap,
                    touches: gap <= touching,
                    cell: (j * across + i) as u32,
                    column: column as u32,
                    row: row as u32,
                    patch: k,
                });
                self.cells[j * across + i] = k + 1;
                if gap > touching {
                    continue;
                }
                // Of one patch with the cells before it that touch the same piece.
                let before = [(-1, 0), (-1, -1), (0, -1), (1, -1)];
                for (di, dj) in before {
                    let (ni, nj) = (i as i32 + di, j as i32 + dj);
                    if ni < 0 || nj < 0 || ni >= across as i32 {
                        continue;
                    }
                    let other = self.cells[nj as usize * across + ni as usize];
                    if other == 0 {
                        continue;
                    }
                    let other = self.near[other as usize - 1];
                    if other.touches && other.piece == piece {
                        let (a, b) = (self.patch(k), self.patch(other.patch));
                        self.near[a.max(b) as usize].patch = a.min(b);
                    }
                }
            }
        }
        // The patches, each one's points together: in the order of their first points, which
        // the points come in already where there is one patch.
        self.patches.clear();
        let mut sorted = true;
        for k in 0..self.near.len() as u32 {
            if self.near[k as usize].touches {
                let patch = self.patch(k);
                sorted &= self.patches.last().is_none_or(|last| last.0 <= patch);
                self.patches.push((patch, k));
            }
        }
        if !sorted {
            self.patches.sort_unstable();
        }
        // A round side that lies on one plane of the support touches it once, however the
        // points near under it fall into patches: on a table read with noise those at the
        // fringe stand apart from the rest. They are of one plane where none lies farther than
        // `touching` off it: a lip at the foot of a slope is not of the slope's.
        let near = &self.near;
        self.rounds.clear();
        let rounds = self
            .patches
            .iter()
            .filter(|p| near[p.1 as usize].piece == ROUND);
        self.rounds.extend(rounds);
        let apart = self.rounds.windows(2).any(|pair| pair[0].0 != pair[1].0);
        let fitted = &mut self.fitted;
        let lies = apart
            && above(near, &self.rounds, object, field.cell, fitted)
                .is_some_and(|(by, off)| off <= touching && by >= object.shape.round() - touching);
        if lies {
            let cell = field.cell;
            round(near, &self.rounds, object, cell, touching, fitted, contacts);
        }
        let mut start = 0;
        while start < self.patches.len() {
            let patch = self.patches[start].0;
            let count = self.patches[start..]
                .iter()
                .take_while(|p| p.0 == patch)
                .count();
            if lies && self.near[self.patches[start].1 as usize].piece == ROUND {
                start += count;
                continue;
            }
            self.patch_contacts(
                field,
                object,
                start..start + count,
                across,
                touching,
                contacts,
            );
            start += count;
        }
    }

    /// The point every point of `k`'s patch leads to.
    fn patch(&mut self, k: u32) -> u32 {
        let mut root = k;
        while self.near[root as usize].patch != root {
            root = self.near[root as usize].patch;
        }
        let mut at = k;
        while at != root {
            at = std::mem::replace(&mut self.near[at as usize].patch, root);
        }
        root
    }

    /// The contacts of one patch, `members` of `self.patches`.
    fn patch_contacts(
        &mut self,
        field: &Field,
        object: &Turned,
        members: std::ops::Range<usize>,
        across: usize,
        touching: f32,
        contacts: &mut Vec<Contact>,
    ) {
        let members = &self.patches[members];
        let near = &self.near;
        let piece = near[members[0].1 as usize].piece;
        if piece == ROUND {
            let fitted = &mut self.fitted;
            return round(
                near, members, object, field.cell, touching, fitted, contacts,
            );
        }
        // Each point of the patch pushing where it is: along `along`, the support's direction
        // there, or where that is not known along the object's own normal.
        let in_place = |contacts: &mut Vec<Contact>, along: Option<Vec3>| {
            let own = members.iter().map(|&(_, k)| {
                let n = &near[k as usize];
                Contact {
                    point: n.point,
                    at: n.over,
                    push: along.filter(|a| a.dot(n.out) < 0.0).unwrap_or(-n.out),
                    piece: n.piece,
                }
            });
            contacts.extend(own);
        };
        // How the support lies under the patch: by its own points, or where they are too few or
        // too nearly in one line, with those around them.
        self.fitted.clear();
        let own = members.iter().map(|&(_, k)| near[k as usize].point);
        self.fitted.extend(own);
        let mut plane = fit(&self.fitted, field.cell);
        // Whether the patch's own points tell the plane: a patch too narrow to is a ridge of
        // the support, or the fringe of a wider stretch.
        let wide = plane.is_some();
        if plane.is_none() {
            self.around.clear();
            for &(_, k) in members {
                let n = &near[k as usize];
                for (dc, dr) in (-2..=2).flat_map(|dr| (-2..=2).map(move |dc| (dc, dr))) {
                    let (column, row) = (n.column as i32 + dc, n.row as i32 + dr);
                    if column >= 0 && row >= 0 {
                        self.around.push((column as u32, row as u32));
                    }
                }
            }
            self.around.sort_unstable();
            self.around.dedup();
            self.fitted.clear();
            // Of the same stretch of support: no step between them and the patch.
            let heights = members.iter().map(|&(_, k)| near[k as usize].point.z);
            let (low, high) = heights.fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), z| {
                (low.min(z), high.max(z))
            });
            let step = STEP * touching;
            for &(column, row) in &self.around {
                let Some(point) = field.top_at(column as usize, row as usize) else {
                    continue;
                };
                if point.z >= low - step && point.z <= high + step {
                    self.fitted.push(point);
                }
            }
            plane = fit(&self.fitted, field.cell);
            // Once more without what lies off that plane, as the top of a wall beside the patch.
            if let Some(normal) = plane {
                let count = self.fitted.len() as f32;
                let middle = self.fitted.iter().sum::<Vec3>() / count;
                let before = self.fitted.len();
                self.fitted
                    .retain(|p| (*p - middle).dot(normal).abs() <= touching);
                if self.fitted.len() < before {
                    plane = fit(&self.fitted, field.cell);
                }
            }
        }
        let Some(normal) = plane else {
            return in_place(contacts, None);
        };
        // What of the piece lies over that stretch of support, whether it touches or not: the
        // points under the piece that lie on the plane.
        let toward = object.frame.transpose() * -normal;
        let count = self.fitted.len() as f32;
        let middle = self.fitted.iter().sum::<Vec3>() / count;
        let on_plane =
            |n: &&Near| n.piece == piece && (n.point - middle).dot(normal).abs() <= touching;
        let over = || near.iter().filter(on_plane).map(|n| n.local);
        let lower = over().fold(Vec3::INFINITY, Vec3::min);
        let upper = over().fold(Vec3::NEG_INFINITY, Vec3::max);
        let grown = |lower: Vec3, upper: Vec3| [lower - 0.5 * field.cell, upper + 0.5 * field.cell];
        // The plane is met by the piece of the object that faces it, as a box's face is set
        // against a plane's. Under another piece are the points just past an edge of that one,
        // which seen from above lie under the side that rises from it: that side meets the
        // plane along the edge and no more.
        let (faces, less) = object.facing(piece, toward);
        let beside = less > ASIDE;
        let mut meets = object.meets(piece, toward, grown(lower, upper), touching, beside);
        if !matches!(meets, Meets::All) {
            // Not with all of it that lies over the plane anywhere: with all of it over this
            // stretch, then, the points that adjoin the patch's one after another.
            self.reached.clear();
            self.reached.resize(near.len(), false);
            self.front.clear();
            for &(_, k) in members {
                self.reached[k as usize] = true;
                self.front.push(k);
            }
            let rows = self.cells.len() / across;
            let (mut lower, mut upper) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
            while let Some(k) = self.front.pop() {
                let n = &near[k as usize];
                (lower, upper) = (lower.min(n.local), upper.max(n.local));
                let (i, j) = (
                    (n.cell as usize % across) as i32,
                    (n.cell as usize / across) as i32,
                );
                for (di, dj) in (-1..=1).flat_map(|dj| (-1..=1).map(move |di| (di, dj))) {
                    let (ni, nj) = (i + di, j + dj);
                    if ni < 0 || nj < 0 || ni >= across as i32 || nj >= rows as i32 {
                        continue;
                    }
                    let other = self.cells[nj as usize * across + ni as usize];
                    if other == 0 || self.reached[other as usize - 1] {
                        continue;
                    }
                    if on_plane(&&near[other as usize - 1]) {
                        self.reached[other as usize - 1] = true;
                        self.front.push(other - 1);
                    }
                }
            }
            meets = object.meets(piece, toward, grown(lower, upper), touching, beside);
        }
        let of = |k: u32| near[k as usize].local;
        // Where on the edge, the line or the point that meets the plane a point of the patch
        // touches, in the object's frame.
        let on = |local: Vec3| match meets {
            Meets::All => local,
            Meets::Point(at) => at,
            Meets::Line { at, axis, half } => {
                let mut on = at;
                on[axis] = local[axis].clamp(-half, half);
                on
            }
        };
        if matches!(meets, Meets::All) {
            return in_place(contacts, Some(normal));
        }
        // Whether the patch comes to it: away from it, it is a ridge of the support that touches.
        let reaches = members.iter().any(|&(_, k)| {
            let n = &near[k as usize];
            let there = object.frame * (on(n.local) - n.local);
            there.truncate().length() <= REACH * field.cell
        });
        if !reaches && wide {
            // The edge that would meet the plane lies past where the support ends: the piece
            // meets the plane where the support under it comes nearest, at its end.
            let nearest = members.iter().map(|&(_, k)| near[k as usize].gap);
            let nearest = nearest.fold(f32::INFINITY, f32::min);
            let tight = members.iter().filter_map(|&(_, k)| {
                let n = &near[k as usize];
                (n.gap <= nearest + TIGHT * touching).then_some(Contact {
                    point: n.point,
                    at: n.over,
                    push: normal,
                    piece: ROUND,
                })
            });
            return contacts.extend(tight);
        }
        if !reaches {
            return in_place(contacts, None);
        }
        // The edge is the facing piece's too, which gives it where it touches there.
        if beside && near.iter().any(|n| n.touches && n.piece == faces) {
            return;
        }
        let contact = |k: u32, push: Vec3| {
            let n = &near[k as usize];
            Contact {
                point: n.point,
                at: n.over + object.frame * (on(n.local) - n.local),
                push,
                piece: ROUND,
            }
        };
        match meets {
            Meets::All => {}
            Meets::Point(_) => contacts.push(contact(members[0].1, normal)),
            Meets::Line { axis, .. } => {
                // Its two ends, which carry what any of the line can; pushed square to it.
                let along = object.frame.col(axis);
                let push = (normal - along * normal.dot(along)).normalize_or(normal);
                let by = |a: &&(u32, u32), b: &&(u32, u32)| of(a.1)[axis].total_cmp(&of(b.1)[axis]);
                let first = members.iter().min_by(by).expect("a patch has a point");
                let last = members.iter().max_by(by).expect("a patch has a point");
                contacts.push(contact(first.1, push));
                if last.1 != first.1 {
                    contacts.push(contact(last.1, push));
                }
            }
        }
    }
}

/// The contacts of a patch under a round piece of `object`, the points `members` of `near`. A
/// round side has a normal of its own everywhere, so it is pushed where it comes nearest the
/// support, along its normal there, as the engines take the nearest points of two curved shapes
/// and the line between them: the points to either side that are as near as `touching` are the
/// side curving away, not more contact. A side that runs straight along its axis, as a can's,
/// touches along the line through that point, as far each way as the patch follows it. One that
/// curves both ways, as a ball's, touches a plane at that one point; where it reaches through
/// the plane of the patch's points by more than `touching`, as a ball set on a cup's rim does,
/// the support is no plane under it, and each point of the patch pushes where it is, along the
/// normal there.
fn round(
    near: &[Near],
    members: &[(u32, u32)],
    object: &Turned,
    cell: f32,
    touching: f32,
    fitted: &mut Vec<Vec3>,
    contacts: &mut Vec<Contact>,
) {
    let of = |k: u32| &near[k as usize];
    let nearest = members
        .iter()
        .map(|m| m.1)
        .min_by(|&a, &b| of(a).gap.total_cmp(&of(b).gap))
        .expect("a patch has a point");
    let tight = of(nearest);
    let contact = |k: u32, on: Vec3| Contact {
        point: of(k).point,
        at: tight.over + object.frame * (on - tight.local),
        push: -tight.out,
        piece: ROUND,
    };
    let Some((radius, half)) = object.straight(tight.local) else {
        let through = above(near, members, object, cell, fitted)
            .is_some_and(|(by, _)| by < object.shape.round() - touching);
        if !through {
            return contacts.push(contact(nearest, tight.local));
        }
        let own = members.iter().map(|m| of(m.1));
        return contacts.extend(own.map(|n| Contact {
            point: n.point,
            at: n.over,
            push: -n.out,
            piece: ROUND,
        }));
    };
    // The line, and the points of the patch that lie along it.
    let line = tight.local.truncate().normalize_or_zero() * radius;
    let along = members.iter().map(|m| m.1).filter(|&k| {
        let at = of(k).local;
        at.truncate().distance(line) <= REACH * cell && at.z.abs() <= half
    });
    let by = |a: &u32, b: &u32| of(*a).local.z.total_cmp(&of(*b).local.z);
    let (first, last) = (along.clone().min_by(by), along.max_by(by));
    let (Some(first), Some(last)) = (first, last) else {
        return contacts.push(contact(nearest, tight.local));
    };
    contacts.push(contact(first, line.extend(of(first).local.z)));
    if last != first {
        contacts.push(contact(last, line.extend(of(last).local.z)));
    }
}

/// How far over the plane through the support points `members` of `near` the middle of the round
/// side of `object` above them lies, its radius where it lies on the plane and less where it
/// reaches through it, and how far off the plane the farthest of the points is. None where
/// the points tell no plane.
fn above(
    near: &[Near],
    members: &[(u32, u32)],
    object: &Turned,
    cell: f32,
    fitted: &mut Vec<Vec3>,
) -> Option<(f32, f32)> {
    let of = |m: &(u32, u32)| &near[m.1 as usize];
    let tight = members
        .iter()
        .map(of)
        .min_by(|a, b| a.gap.total_cmp(&b.gap))?;
    fitted.clear();
    fitted.extend(members.iter().map(|m| of(m).point));
    let normal = fit(fitted, cell)?;
    let on = fitted.iter().sum::<Vec3>() / fitted.len() as f32;
    let middle = tight.over - tight.out * object.shape.round();
    let off = fitted.iter().map(|p| (*p - on).dot(normal).abs());
    Some(((middle - on).dot(normal), off.fold(0.0, f32::max)))
}

/// The unit normal, upward, of the plane through the support points `of`, by least squares over
/// their heights; none where they are fewer than three or spread no wider than one row of cells
/// can, which tells no plane.
fn fit(of: &[Vec3], cell: f32) -> Option<Vec3> {
    if of.len() < 3 {
        return None;
    }
    let count = of.len() as f64;
    let points = || of.iter().map(|p| p.as_dvec3());
    let mean = points().sum::<glam::DVec3>() / count;
    let (mut xx, mut xy, mut yy, mut xz, mut yz) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for p in points() {
        let d = p - mean;
        xx += d.x * d.x;
        xy += d.x * d.y;
        yy += d.y * d.y;
        xz += d.x * d.z;
        yz += d.y * d.z;
    }
    // The lesser spread, seen from above.
    let (half_sum, det) = (0.5 * (xx + yy), xx * yy - xy * xy);
    let least = half_sum - (half_sum * half_sum - det).max(0.0).sqrt();
    // Points anywhere in one row of cells spread at most a cell over the root of twelve.
    let row = f64::from(cell * cell) / 12.0;
    if least <= count * row {
        return None;
    }
    let (b, c) = ((xz * yy - yz * xy) / det, (yz * xx - xz * xy) / det);
    Some(glam::DVec3::new(-b, -c, 1.0).normalize().as_vec3())
}

/// The contact of `object`, its middle at `center`, with the side of the column of support under
/// `point`, a point beside it seen from above, where the support was seen down to `seen`: where
/// the column rises past the nearest of the object by half of `touching` or more, within
/// `touching` of it, and was seen that far down, it pushes level against it. The end of a board
/// with nothing seen under it is no wall.
fn beside(
    object: &Turned,
    center: Vec3,
    point: Vec3,
    seen: f32,
    touching: f32,
    contacts: &mut Vec<Contact>,
) {
    // The nearest of the object to the column, by turns from each to the other.
    let mut on_column = point.with_z(point.z.min(center.z));
    let mut on_object = center;
    for _ in 0..3 {
        on_object = center + object.nearest(on_column - center);
        on_column.z = on_object.z.min(point.z);
    }
    let apart = on_object - on_column;
    let below = on_column.z < seen - touching;
    if below || point.z - on_object.z < 0.5 * touching || apart.length() > touching {
        return;
    }
    let level = apart.truncate();
    let push = if level.length() > 1e-6 {
        level.normalize().extend(0.0)
    } else {
        let (_, out) = object.piece_at(object.frame.transpose() * (on_object - center));
        let Some(push) = (-out.truncate()).try_normalize() else {
            return;
        };
        push.extend(0.0)
    };
    contacts.push(Contact {
        point: on_column,
        at: on_object,
        push,
        piece: ROUND,
    });
}
