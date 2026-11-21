//! Telling one view's readings apart into parts: stretches of surface that no jump in depth, no
//! step and no sharp concave crease divides.
//!
//! The steps are those of depth-map segmentation for incremental mapping (Tateno, Tombari and
//! Navab 2015; Furrer et al. 2018, whose segments Voxblox++ fuses), with the convexity rule of
//! LCCP (Stein et al. 2014): things that touch or stand on one another meet in a concave crease,
//! while one convex thing's faces meet in convex edges, so readings are cut apart at the first
//! and kept together across the second.
//!
//! Everything is measured in inverse depth. There a flat surface is a plane over the image
//! however it leans, so fitting one is linear and exact, and a triangulating camera's noise is
//! the same at every depth.
//!
//! 1. **Jumps** ([`links`]). Neighboring readings are on one surface unless the depth between
//!    them steps by more than any surface through their own neighbors would.
//! 2. **Planes** ([`Planes`]). Each reading gets the plane through the readings around it, over
//!    a window as wide as the noise needs for the plane's lean to be told to a third of
//!    [`CREASE`], from sums kept over the whole image (Holzer et al. 2012). A window whose
//!    readings do not lie on one plane, as one across a step, gives none.
//! 3. **Seeds** ([`flood`]): the readings with planes, connected without crossing a jump,
//!    where they fill a window. A window across a crease or a step is not one plane, so the
//!    readings along it have none and keep the seeds on either side apart. Each seed grows over
//!    the readings beside it that lie on its plane ([`grow`]), and the two sides meet at the
//!    crease.
//! 4. **Slivers**: what is left lies on no seed's plane, as a rim or a wall seen edge on, too
//!    thin to hold a window. It is told apart by jumps alone, and is a part of its own where it
//!    holds a window ([`holding`]). One of readings on no shape that is narrower, however long,
//!    lies along the one part of such readings beside it, most of it within a window's width of
//!    it, and does not stand in front of it along a window's width of their border, is that
//!    part's edge, and its readings go to the parts beside them. One that only reaches that part
//!    or stands in front of it, or has no such part beside it, as a rod alone on a table, or lies
//!    between two, is a part where it fills a window, as is one of readings on a known shape.
//! 5. **Borders** ([`borders`]). Where two of these meet with no jump between, each one's plane
//!    is held against the other's surface, which is LCCP's test: each passing in front of the
//!    other is a convex edge, and they are one part, as are two planes that agree; each passing
//!    behind is a crease; one in front and one behind is a step, and the one in front hides the
//!    other. How much each part hides, and which parts meet in a crease, tell a hollow's floor
//!    from a thing of its own; which touch, with no jump between, what holds up a thing that
//!    hangs.
//! 6. **Round things** ([`round`]). A part that hides another across their border, as a bowl's
//!    outside does its inside along the near rim, is one part with it where the two are round
//!    about one axis and end in one rim, unless the view's masks say they are two things.
//! 7. **Masks** ([`masks::split`]). A part whose readings lie in several of the view's masks, or
//!    in a mask and in none, is divided along them: a side other than the part's largest is cut
//!    off where it fills a window and is as wide as one ([`holding`]), as a sliver is a part
//!    where it is, or lies in a mask and is most of what that mask covers in the view or the
//!    part's largest side lies in another mask. What a side hides and the creases it meets in
//!    are its own: each crossing of a border was counted under the mask its reading lies in
//!    ([`Side`]).
//!
//! A reading may lie on a known shape, no farther off it than the camera's noise lets one on it
//! read ([`Kind::Known`]), and be the shape's or a thing's: a plate's floor, the foot of a box.
//! The readings on a shape and those on none are told apart alike, and apart from each other,
//! so that what lies off the shapes comes out as it would without them:
//!
//! - A reading on no shape gets its plane from the readings on none around it; one on a shape,
//!   from all those around it. So the readings on a shape about the foot of what stands on it
//!   have no plane, and each falls to the surface it lies nearest ([`grow`]): the thing's,
//!   where it lies nearer the thing's plane than it lies over the shape.
//! - A seed is of readings on a shape or of readings on none, and no part of the one is made
//!   one with a part of the other, however their planes agree: a thing that rises from a table
//!   without a crease stays a thing. Where they meet as one surface is kept with their creases
//!   ([`Parts::meets`]), for a floor to be told by.
//! - Across what may read on a shape the planes tell no step, nor does one reading on a shape
//!   ([`borders`]).
//!
//! Which parts are a known shape's is for `look` to say, from where each stands in front of
//! another ([`Parts::hidden`]).

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::ops::Range;

use glam::Vec3;

use crate::view::{Kind, SIGMAS, Seen};
use crate::{masks, threads};

mod round;

/// The part of a pixel that is in none.
pub(crate) const NONE: u32 = u32::MAX;

/// The angle between two surfaces that a crease between two things is told by (radians): 10
/// degrees, the tolerance LCCP keeps an object's own shallow dents whole with. The windows are
/// sized to tell it ([`WINDOW`]).
const CREASE: f32 = 0.174_532_92;

/// How many of the noise's standard deviations a step in inverse depth must pass what the
/// readings beside it predict to be a jump: five deviations of that difference, which is of
/// three readings with weights 1, -2 and 1 (5 * sqrt(6)). Five, so that a window's worth of
/// readings holds no jump by chance: a window stops at the nearest one.
const JUMP: f32 = 12.247_449;

/// What a window's size must reach, as `(2r + 1) * sqrt(r * (r + 1))` for its radius `r` in
/// pixels, per unit of noise times focal length times depth: enough that the angle between two
/// windows' planes is told to within [`CREASE`] at three deviations, a third of it at one
/// (3 * sqrt(2) * sqrt(3) / CREASE).
const WINDOW: f32 = 7.348_469 / CREASE;

/// How far from its pixel's middle the point a reading is of lies, as a standard deviation in
/// pixels: anywhere in the pixel alike (1 / sqrt(12)). On a surface that leans, it reads as
/// depth noise.
const LATERAL: f32 = 0.288_675;

/// The neighbors a pixel is linked to going forward: east, south, southeast and southwest. The
/// other four are these from the neighbor's side.
const DIRECTIONS: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (-1, 1)];

/// One view's readings told apart.
pub(crate) struct Parts {
    /// Per pixel, its part, or [`NONE`].
    pub(crate) of: Vec<u32>,
    /// How many parts there are; they are numbered in the order the image's rows meet them.
    pub(crate) count: usize,
    /// Per pixel, the unit normal of its surface, toward the camera; zero where it has none.
    pub(crate) normals: Vec<Vec3>,
    /// Per pixel, the share it is of the window a plane is taken over at its depth: readings
    /// that add up to less than one window cannot be told from noise.
    pub(crate) share: Vec<f32>,
    /// Per part, how much of its border it is the nearer side of, in window widths: what it
    /// hides of the surfaces behind it.
    pub(crate) hides: Vec<f32>,
    /// Each place along a border where one reading hides the other, across a jump or a step:
    /// the reading in front, then the one behind. What `hides` is the sum of, reading by
    /// reading.
    pub(crate) hidden: Vec<(usize, usize)>,
    /// Per two parts that meet in a crease, the lower numbered first: how long the crease is,
    /// in window widths. Where one is of readings on a known shape and the other of readings on
    /// none, the border they meet in as one surface counts as well: the two are never one part.
    pub(crate) meets: BTreeMap<(u32, u32), f32>,
    /// Per two parts, the lower numbered first: how long a border with no jump in it they share,
    /// in window widths, where they touch. A crease is such a border, and so is one where either
    /// side has no plane to tell a crease by, as where a handle's post narrower than a window
    /// meets the front it stands out of.
    pub(crate) touches: BTreeMap<(u32, u32), f32>,
}

/// The parts of `seen`'s readings, those that lie in several of the view's `masks` (a number
/// per pixel, or empty) divided along them. What is done for each pixel by itself is done on up
/// to `threads` threads.
pub(crate) fn parts(seen: &Seen, masks: &[u32], threads: usize) -> Parts {
    let pixels = seen.kinds.len();
    let open = |i: usize| seen.kinds[i] == Kind::Open;
    let known = |i: usize| matches!(seen.kinds[i], Kind::Known(_));
    // Per pixel, the standard deviation of its reading in inverse depth; 0 without one.
    let noise: Vec<f32> = each_pixel(seen, threads, |i| {
        if seen.inverse[i] == 0.0 {
            0.0
        } else {
            seen.spread(i)
        }
    });
    let mut joined = vec![0; pixels];
    let band = band(seen, threads);
    threads::each_piece(
        threads,
        joined.chunks_mut(band).collect(),
        |number, rows| {
            links(seen, &noise, number * band, rows);
        },
    );
    let walker = Walker::new(seen, &joined);
    // The radius of the window the noise asks for at each reading's depth: the least whose size
    // reaches what it needs, or the largest the image holds, where none reaches.
    let largest = seen.width.min(seen.height).saturating_sub(1).max(1);
    let sizes: Vec<f32> = (1..largest)
        .map(|r| (2 * r + 1) as f32 * ((r * (r + 1)) as f32).sqrt())
        .collect();
    let wanted: Vec<usize> = each_pixel(seen, threads, |i| {
        if !seen.reads(i) {
            return 0;
        }
        let need = WINDOW * seen.sigma(i) / seen.spacing(i);
        let reaches = sizes.iter().position(|&size| size >= need);
        reaches.map_or(largest, |below| below + 1)
    });
    let planes = Planes::of(seen, &wanted, noise, threads);

    // Seeds: the readings with planes, connected without crossing a jump, where they fill a
    // window: those on no known shape among themselves, and those on one. Each grows over the
    // readings that lie on its plane.
    let width = |i: usize| (2 * wanted[i] + 1) as f32;
    let share: Vec<f32> = each_pixel(seen, threads, |i| 1.0 / (width(i) * width(i)));
    let mut of = vec![NONE; pixels];
    // Those of the readings on no shape are numbered first, as the slivers of such readings are
    // below: a part's number says which it is of.
    let seeded = |i: usize| open(i) && planes.has[i];
    let open_seeds = flood(&walker, &mut of, 0, &seeded, Some(&share));
    let seeded = |i: usize| known(i) && planes.has[i];
    let seeds = flood(&walker, &mut of, open_seeds, &seeded, Some(&share));
    let mut surface: Vec<usize> = (0..pixels).collect();
    grow(
        &walker,
        &planes,
        &mut of,
        &mut surface,
        /* on its plane */ true,
    );
    // Slivers: what is left, told apart by jumps alone.
    let open_slivers = flood(&walker, &mut of, seeds, &open, None);
    let all = flood(&walker, &mut of, open_slivers, &known, None);
    // Whether a part, by the number it was made with, is of readings on a known shape.
    let on_shape = |part: u32| (open_seeds..seeds).contains(&part) || part >= open_slivers;

    // Those that meet as one surface or in a convex edge are one part, named by its first:
    // but for a part of readings on a known shape and one of readings on none.
    let sides = borders(&walker, &planes, &of, &surface, masks, &width);
    let mut first: Vec<u32> = (0..all).collect();
    let find = |first: &mut [u32], part: u32| {
        let mut at = part;
        while first[at as usize] != at {
            first[at as usize] = first[first[at as usize] as usize];
            at = first[at as usize];
        }
        at
    };
    for (&(a, b), meeting) in &sides.meetings {
        // A window's width of border that is one surface or a convex edge makes them one,
        // whatever the rest of their border is: a bowl's wall meets its inside over the rim at
        // its sides, and stands in front of it at the near rim.
        if meeting.one >= 1.0 && meeting.one > meeting.crease && on_shape(a) == on_shape(b) {
            let (a, b) = (find(&mut first, a), find(&mut first, b));
            first[a.max(b) as usize] = a.min(b);
        }
    }
    // A set of slivers of readings on no shape is a part of its own where it holds a window
    // ([`holding`]): where it is as wide as a window that could have had a plane. One that holds
    // none is the edge of the part of such readings beside it, where one lies beside it, the set
    // does not stand in front of it, across a jump or a step, along a window's width of their
    // border, and more than half of its readings lie within a window's width of it in the image,
    // the reach of one plane: as a box's side seen edge on runs along its top and does not stand
    // in front of it. Its readings go to the parts beside them, whichever plane each lies
    // nearest, however many there are, as do those of any set that fills no window, which cannot
    // be told from noise whether or not it holds one cut by the image's edge. Every other set is a
    // part where it fills a window. One that only reaches that part, as a pencil whose tip meets a
    // box's corner, is not its edge; one that stands in front of it so is a thing before it, as a
    // slab a box stands on is, a rod leaning on it or a handle's post; where no such part lies
    // beside it, as by a rod alone on a table, it is a thin thing of its own; where two do, it
    // lies between two things, and of a reading with no plane the depth does not say whose edge it
    // is. And readings on a known shape are the shape's or a thing's by the stretch they make
    // (`look`), which is kept.
    //
    // The window's width is the unit borders are measured in, not a measured threshold: at two
    // widths of standing in front a pencil lying along the foot of a box's front would be taken
    // for its edge (`a_pencil_lying_along_a_boxs_front_is_not_the_boxs_on_noisy_depth`), but no
    // test tells one width from any length at all.
    let wholes: Vec<u32> = (of.iter())
        .map(|&part| {
            if part == NONE {
                NONE
            } else {
                find(&mut first, part)
            }
        })
        .collect();
    let mut sizes = vec![0.0; all as usize];
    // Per set of slivers, the first part of readings on no shape found beside it, and whether
    // another is. A set of seeds that holds one of such readings holds them alone, and is named
    // by it.
    let mut beside = vec![(NONE, false); all as usize];
    for (i, &whole) in wholes.iter().enumerate() {
        if whole == NONE {
            continue;
        }
        sizes[whole as usize] += share[i];
        if whole >= seeds {
            let by = walker.neighbors(i).map(|j| wholes[j]);
            for other in by.filter(|&other| other < open_seeds) {
                let (one, two) = &mut beside[whole as usize];
                if *one == NONE {
                    *one = other;
                } else if *one != other {
                    *two = true;
                }
            }
        }
    }
    // Per set of slivers, how much of its border it stands in front of that part along, across a
    // jump or a step, in window widths: a rod leaning on a box's front hides the front across a
    // jump all along it, and touches it without one only where it rests.
    let mut before = vec![0.0; all as usize];
    for &(front, back) in &sides.hidden {
        let whole = wholes[front];
        if whole != NONE && whole >= seeds && beside[whole as usize].0 == wholes[back] {
            before[whole as usize] += 1.0 / width(front);
        }
    }
    let holds = holding(seen, &wholes, &wanted, seeds..open_slivers);
    // Whether a set of slivers lies beside one part of readings on no shape alone, and does not
    // stand in front of it so: an edge of that part, if it lies along it.
    let alongside = |whole: u32| {
        let (one, two) = beside[whole as usize];
        whole >= seeds
            && whole < open_slivers
            && one != NONE
            && !two
            && before[whole as usize] < 1.0
    };
    // Of each such set, how many readings it has, and how many of them have a reading of that
    // part within a window's width in the image, the reach of one plane.
    let mut readings = vec![0usize; all as usize];
    let mut near = vec![0usize; all as usize];
    for (i, &whole) in wholes.iter().enumerate() {
        if whole == NONE || !alongside(whole) || holds[whole as usize] {
            continue;
        }
        readings[whole as usize] += 1;
        let part = beside[whole as usize].0;
        let reach = (2 * wanted[i] + 1) as isize;
        let (x, y) = ((i % seen.width) as isize, (i / seen.width) as isize);
        let at = |u: isize, v: isize| {
            let inside =
                u >= 0 && v >= 0 && (u as usize) < seen.width && (v as usize) < seen.height;
            inside && wholes[v as usize * seen.width + u as usize] == part
        };
        let found = (-reach..=reach).any(|dy| (-reach..=reach).any(|dx| at(x + dx, y + dy)));
        near[whole as usize] += usize::from(found);
    }
    for (part, &whole) in of.iter_mut().zip(&wholes) {
        if *part != NONE {
            let edge = alongside(whole) && 2 * near[whole as usize] > readings[whole as usize];
            let kept = sizes[whole as usize] >= 1.0 && (holds[whole as usize] || !edge);
            *part = if whole >= seeds && !kept { NONE } else { whole };
        }
    }
    grow(
        &walker,
        &planes,
        &mut of,
        &mut surface,
        /* on its plane */ false,
    );
    // A wall's outside and the inside it stands in front of are one part too, where they are
    // round about one axis and end in one rim, both of readings on no shape. Which reading hides
    // which was taken along the borders as they were; their parts are asked for as they are now,
    // a rim's slivers gone to the parts beside them. Two that lie in different masks stay two,
    // as the masks say.
    let lies = masks::lies(&of, all as usize, &share, masks);
    let shells = round::shells(
        seen,
        &planes,
        &of,
        all as usize,
        &on_shape,
        &width,
        &sides.hidden,
    );
    for (a, b) in shells {
        let apart = |a: u32, b: u32| a != 0 && b != 0 && a != b;
        if !apart(lies[a as usize], lies[b as usize]) {
            let (a, b) = (find(&mut first, a), find(&mut first, b));
            first[a.max(b) as usize] = a.min(b);
        }
    }
    for part in &mut of {
        if *part != NONE {
            *part = find(&mut first, *part);
        }
    }

    // The parts, numbered as the rows meet them.
    let mut renumbered = vec![NONE; all as usize];
    let mut count = 0;
    for part in &mut of {
        if *part != NONE {
            let new = &mut renumbered[*part as usize];
            if *new == NONE {
                *new = count;
                count += 1;
            }
            *part = *new;
        }
    }
    // Those the masks divide, and with each side the borders of its own readings.
    let mut count = count as usize;
    let cut = masks::split(&mut of, &mut count, &share, &wanted, masks, seen);
    let now = |first: &mut [u32], (piece, mask): Side| {
        let part = renumbered[find(first, piece) as usize];
        cut.get(&(part, mask)).copied().unwrap_or(part)
    };
    let mut hides = vec![0.0; count];
    for (&side, hidden) in &sides.hides {
        let part = now(&mut first, side);
        if part != NONE {
            hides[part as usize] += hidden;
        }
    }
    let mut meets: BTreeMap<(u32, u32), f32> = BTreeMap::new();
    for (&(a, b), crease) in &sides.creases {
        let (a, b) = (now(&mut first, a), now(&mut first, b));
        if a != b && a != NONE && b != NONE {
            *meets.entry((a.min(b), a.max(b))).or_default() += crease;
        }
    }
    let mut touches: BTreeMap<(u32, u32), f32> = BTreeMap::new();
    for (&(a, b), length) in &sides.touches {
        let (a, b) = (now(&mut first, a), now(&mut first, b));
        if a != b && a != NONE && b != NONE {
            *touches.entry((a.min(b), a.max(b))).or_default() += length;
        }
    }
    Parts {
        of,
        count,
        normals: planes.normals(seen, threads),
        share,
        hides,
        hidden: sides.hidden,
        meets,
        touches,
    }
}

/// Per set of `wholes` (each pixel's set, or [`NONE`]), whether it holds a window: whether some
/// reading of it has its whole window, of the radius `wanted` gives it and cut by the image's
/// edges as a plane's is, of readings of the set alone. Asked of the sets numbered in `slivers`.
/// Their readings are counted over every rectangle from the image's corner, and a window is
/// looked through only where all of it is such readings.
pub(crate) fn holding(
    seen: &Seen,
    wholes: &[u32],
    wanted: &[usize],
    slivers: Range<u32>,
) -> Vec<bool> {
    let (width, height) = (seen.width, seen.height);
    let sliver = |i: usize| slivers.contains(&wholes[i]);
    let stride = width + 1;
    let mut counts = vec![0u32; stride * (height + 1)];
    for y in 0..height {
        let mut row = 0;
        for x in 0..width {
            row += u32::from(sliver(y * width + x));
            counts[(y + 1) * stride + x + 1] = counts[y * stride + x + 1] + row;
        }
    }
    let sets = (wholes.iter().filter(|&&whole| whole != NONE).max()).map_or(0, |&most| most + 1);
    let mut holds = vec![false; sets as usize];
    for i in (0..wholes.len()).filter(|&i| sliver(i)) {
        let whole = wholes[i];
        if holds[whole as usize] {
            continue;
        }
        let (x, y, r) = (i % width, i / width, wanted[i]);
        let (left, right) = (x.saturating_sub(r), (x + r + 1).min(width));
        let (top, bottom) = (y.saturating_sub(r), (y + r + 1).min(height));
        let corner = |x: usize, y: usize| counts[y * stride + x];
        let count =
            corner(right, bottom) + corner(left, top) - corner(right, top) - corner(left, bottom);
        if count as usize == (right - left) * (bottom - top) {
            let rows = top..bottom;
            holds[whole as usize] = rows
                .into_iter()
                .all(|v| (left..right).all(|u| wholes[v * width + u] == whole));
        }
    }
    holds
}

/// Numbers, from `first` on, the sets of pixels in no part that `admit` lets in and that links
/// join, in the order the image's rows meet them, and returns the next number. With `shares`
/// (per pixel, its share of a window), a set that fills less than a window is left in no part.
fn flood(
    walker: &Walker<'_>,
    of: &mut [u32],
    first: u32,
    admit: &dyn Fn(usize) -> bool,
    shares: Option<&[f32]>,
) -> u32 {
    let mut count = first;
    let mut met = vec![false; of.len()];
    let mut members = vec![];
    for start in 0..of.len() {
        if met[start] || of[start] != NONE || !admit(start) {
            continue;
        }
        members.clear();
        members.push(start);
        met[start] = true;
        let mut next = 0;
        while next < members.len() {
            for j in walker.neighbors(members[next]) {
                if !met[j] && of[j] == NONE && admit(j) {
                    met[j] = true;
                    members.push(j);
                }
            }
            next += 1;
        }
        let fills = |shares: &[f32]| members.iter().map(|&i| shares[i]).sum::<f32>() >= 1.0;
        if shares.is_none_or(fills) {
            for &i in &members {
                of[i] = count;
            }
            count += 1;
        }
    }
    count
}

/// How many pixels a band of `seen`'s rows holds where its pixels are done in bands on `threads`
/// threads: all of them on one, and on more a few bands a thread, so that one whose rows hold
/// more to do holds up no other thread long.
fn band(seen: &Seen, threads: usize) -> usize {
    let rows = if threads > 1 {
        seen.height.div_ceil(4 * threads)
    } else {
        seen.height
    };
    (rows * seen.width).max(1)
}

/// `pixel` for each of `seen`'s pixels, in bands of rows on up to `threads` threads ([`band`]),
/// with the results in the pixels' order: each the same however many threads there are.
fn each_pixel<T: Clone + Default + Send>(
    seen: &Seen,
    threads: usize,
    pixel: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let mut values = vec![T::default(); seen.kinds.len()];
    let band = band(seen, threads);
    threads::each_piece(
        threads,
        values.chunks_mut(band).collect(),
        |number, values| {
            for (k, value) in values.iter_mut().enumerate() {
                *value = pixel(number * band + k);
            }
        },
    );
    values
}

/// Per pixel of the rows from pixel `first` on that `joined` holds, a bit for each of
/// [`DIRECTIONS`]: set where the pixel and its neighbor that way both read a surface (an open
/// reading or a known shape's) with no jump between them. `noise` is each reading's standard
/// deviation in inverse depth.
///
/// The step between two readings, in inverse depth, lies between the steps on either side of it
/// wherever one surface runs through all four, flat, creased or curved. A jump is a step outside
/// those by more than the noise makes of it.
fn links(seen: &Seen, noise: &[f32], first: usize, joined: &mut [u8]) {
    let (width, height) = (seen.width as isize, seen.height as isize);
    let depth = &seen.inverse;
    // The pixel at column `x` and row `y`, where the image has one and it reads a surface.
    let reading = |x: isize, y: isize| {
        let inside = x >= 0 && y >= 0 && x < width && y < height;
        inside
            .then(|| (y * width + x) as usize)
            .filter(|&i| seen.reads(i))
    };
    let rows = first / seen.width..(first + joined.len()) / seen.width;
    let pixels = rows.flat_map(|y| (0..width).map(move |x| (y as isize, x)));
    for ((y, x), links) in pixels.zip(joined) {
        let Some(i) = reading(x, y) else {
            continue;
        };
        for (bit, &(dx, dy)) in DIRECTIONS.iter().enumerate() {
            let Some(j) = reading(x + dx, y + dy) else {
                continue;
            };
            let before = reading(x - dx, y - dy);
            let after = reading(x + 2 * dx, y + 2 * dy);
            let beside = [
                before.map(|b| depth[i] - depth[b]),
                after.map(|a| depth[a] - depth[j]),
            ];
            let mut span = (f32::INFINITY, f32::NEG_INFINITY);
            for step in beside.into_iter().flatten() {
                span = (span.0.min(step), span.1.max(step));
            }
            // Two readings with none beside them: taken as a surface that faces the camera.
            if span.0 > span.1 {
                span = (0.0, 0.0);
            }
            let step = depth[j] - depth[i];
            let jump = JUMP * noise[i].max(noise[j]);
            if step >= span.0 - jump && step <= span.1 + jump {
                *links |= 1 << bit;
            }
        }
    }
}

/// Walks a view's readings along its links.
struct Walker<'a> {
    seen: &'a Seen,
    joined: &'a [u8],
    /// Per pixel, a bit for each of its eight neighbors it is joined to: for each of
    /// [`DIRECTIONS`] in turn the neighbor forward, then the one back.
    around: Vec<u8>,
    /// How far each of those neighbors lies from a pixel in the image's memory.
    strides: [isize; 8],
}

impl<'a> Walker<'a> {
    /// A walker over `seen`'s readings as `joined` links them ([`links`]).
    fn new(seen: &'a Seen, joined: &'a [u8]) -> Self {
        let mut around = vec![0; joined.len()];
        let mut strides = [0; 8];
        for (bit, &(dx, dy)) in DIRECTIONS.iter().enumerate() {
            let stride = dx + dy * seen.width as isize;
            strides[2 * bit] = stride;
            strides[2 * bit + 1] = -stride;
            for i in (0..joined.len()).filter(|&i| joined[i] & (1 << bit) != 0) {
                // A link is set only between two pixels of the image.
                around[i] |= 1 << (2 * bit);
                around[(i as isize + stride) as usize] |= 1 << (2 * bit + 1);
            }
        }
        Self {
            seen,
            joined,
            around,
            strides,
        }
    }

    /// The pixel one link from `i` along direction `bit`, forward or back, if they are joined.
    fn next(&self, i: usize, bit: usize, forward: bool) -> Option<usize> {
        let (dx, dy) = DIRECTIONS[bit];
        // A link is kept with the pixel it goes forward from.
        let (from, to) = if forward {
            (i, self.seen.step(i, (dx, dy))?)
        } else {
            let j = self.seen.step(i, (-dx, -dy))?;
            (j, j)
        };
        (self.joined[from] & (1 << bit) != 0).then_some(to)
    }

    /// The pixels joined to `i`, of its eight neighbors.
    fn neighbors(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let around = self.around[i];
        let joined = (0..self.strides.len()).filter(move |bit| around & (1 << bit) != 0);
        joined.map(move |bit| (i as isize + self.strides[bit]) as usize)
    }
}

/// Each reading's plane in inverse depth over the image: `c + a * du + b * dv` at `du` columns
/// and `dv` rows from the reading, fitted by least squares to the readings in the square window
/// around it: those on no known shape, for a reading on none, and all of them, for one on a
/// shape.
struct Planes {
    /// Per pixel `[c, a, b]`: its own reading and no slope where it has no plane.
    fits: Vec<[f32; 3]>,
    /// Per pixel, the covariance of `[c, a, b]`, its upper triangle row by row: empty of
    /// meaning where the pixel has no plane.
    covariances: Vec<[f32; 6]>,
    /// Per pixel, whether it has a plane: a reading whose window's readings lie on one.
    has: Vec<bool>,
    /// Per pixel, the standard deviation of its reading in inverse depth.
    noise: Vec<f32>,
    width: usize,
}

impl Planes {
    /// The planes of `seen`'s readings, each over a window of the radius `wanted`, with the
    /// standard deviation of each reading in inverse depth (`noise`, 0 without a reading).
    ///
    /// A window's sums come from sums kept over every rectangle from the image's corner, so each
    /// fit costs the same whatever the window's size. A reading's variance is the noise's and
    /// what a point anywhere in its pixel adds on a surface that leans ([`LATERAL`]). A window
    /// whose readings lie farther off its plane than that explains, at three deviations of
    /// their summed squares, is not of one flat surface and gives no plane.
    fn of(seen: &Seen, wanted: &[usize], noise: Vec<f32>, threads: usize) -> Self {
        let (width, height) = (seen.width, seen.height);
        let pixels = width * height;
        // 1, x, y, xx, xy, yy, w, xw, yw and ww, summed over the readings on no known shape, and
        // over all the readings.
        let stride = width + 1;
        let mut sums = vec![[[0f64; 10]; 2]; stride * (height + 1)];
        for y in 0..height {
            let mut row = [[0f64; 10]; 2];
            for x in 0..width {
                let i = y * width + x;
                if seen.reads(i) {
                    let (x, y, w) = (x as f64, y as f64, f64::from(seen.inverse[i]));
                    let terms = [1.0, x, y, x * x, x * y, y * y, w, x * w, y * w, w * w];
                    let from = usize::from(seen.kinds[i] != Kind::Open);
                    for row in &mut row[from..] {
                        *row = std::array::from_fn(|k| row[k] + terms[k]);
                    }
                }
                let above = sums[y * stride + x + 1];
                sums[(y + 1) * stride + x + 1] =
                    std::array::from_fn(|of| std::array::from_fn(|k| above[of][k] + row[of][k]));
            }
        }
        // Reading `i`'s plane and its covariance, if it has one.
        let fit = |i: usize| {
            if !seen.reads(i) {
                return None;
            }
            let (x, y, r) = (i % width, i / width, wanted[i]);
            let (left, right) = (x.saturating_sub(r), (x + r + 1).min(width));
            let (top, bottom) = (y.saturating_sub(r), (y + r + 1).min(height));
            let of = usize::from(seen.kinds[i] != Kind::Open);
            let corner = |x: usize, y: usize| sums[y * stride + x][of];
            let (a, b, c, d) = (
                corner(left, top),
                corner(right, top),
                corner(left, bottom),
                corner(right, bottom),
            );
            let [n, sx, sy, sxx, sxy, syy, sw, sxw, syw, sww]: [f64; 10] =
                std::array::from_fn(|k| d[k] - b[k] - c[k] + a[k]);
            // About the reading's own pixel.
            let (x, y) = (x as f64, y as f64);
            let (mx, my) = (sx - x * n, sy - y * n);
            let (mxx, mxy, myy) = (
                sxx - 2.0 * x * sx + x * x * n,
                sxy - x * sy - y * sx + x * y * n,
                syy - 2.0 * y * sy + y * y * n,
            );
            let (mxw, myw) = (sxw - x * sw, syw - y * sw);
            // The inverse of [[n, mx, my], [mx, mxx, mxy], [my, mxy, myy]], by its cofactors.
            let cofactors = [
                mxx * myy - mxy * mxy,
                my * mxy - mx * myy,
                mx * mxy - my * mxx,
                n * myy - my * my,
                mx * my - n * mxy,
                n * mxx - mx * mx,
            ];
            let determinant = n * cofactors[0] + mx * cofactors[1] + my * cofactors[2];
            // Readings on one line, or too few, lie on no one plane: on a line, the determinant
            // is what rounding leaves of its terms, under half of a double's digits of them.
            let spans = determinant > f64::EPSILON.sqrt() * n * mxx.max(1.0) * myy.max(1.0);
            if n < 4.0 || !spans {
                return None;
            }
            let inverse = cofactors.map(|cofactor| cofactor / determinant);
            let fit = [
                inverse[0] * sw + inverse[1] * mxw + inverse[2] * myw,
                inverse[1] * sw + inverse[3] * mxw + inverse[4] * myw,
                inverse[2] * sw + inverse[4] * mxw + inverse[5] * myw,
            ];
            let off = sww - (fit[0] * sw + fit[1] * mxw + fit[2] * myw);
            let lean = LATERAL as f64 * fit[1].hypot(fit[2]);
            let noise = f64::from(noise[i]);
            let variance = noise * noise + lean * lean;
            let free = n - 3.0;
            // Asked this way round, so that sums that are not numbers give no plane.
            let flat = off <= variance * (free + f64::from(SIGMAS) * (2.0 * free).sqrt());
            if !flat {
                return None;
            }
            Some((
                fit.map(|v| v as f32),
                inverse.map(|v| (v * variance) as f32),
            ))
        };
        let mut fits: Vec<[f32; 3]> = seen.inverse.iter().map(|&w| [w, 0.0, 0.0]).collect();
        let mut covariances = vec![[0.0; 6]; pixels];
        let mut has = vec![false; pixels];
        // In bands of rows.
        let band = band(seen, threads);
        let bands = (fits.chunks_mut(band))
            .zip(covariances.chunks_mut(band))
            .zip(has.chunks_mut(band));
        threads::each_piece(
            threads,
            bands.collect(),
            |number, ((fits, covariances), has)| {
                for k in 0..fits.len() {
                    if let Some((fit, covariance)) = fit(number * band + k) {
                        (fits[k], covariances[k], has[k]) = (fit, covariance, true);
                    }
                }
            },
        );
        Planes {
            fits,
            covariances,
            has,
            noise,
            width,
        }
    }

    /// Where pixel `at` lies from pixel `i`, in columns and rows.
    fn offset(&self, i: usize, at: usize) -> (f32, f32) {
        let column = |k: usize| (k % self.width) as f32;
        let row = |k: usize| (k / self.width) as f32;
        (column(at) - column(i), row(at) - row(i))
    }

    /// The inverse depth reading `i`'s plane has at pixel `at`.
    fn at(&self, i: usize, at: usize) -> f32 {
        let (du, dv) = self.offset(i, at);
        let [c, a, b] = self.fits[i];
        c + a * du + b * dv
    }

    /// The variance of [`Planes::at`]. A reading without a plane has its own noise and says
    /// nothing of the surface beside it.
    fn variance(&self, i: usize, at: usize) -> f32 {
        if !self.has[i] {
            return if i == at {
                self.noise[i] * self.noise[i]
            } else {
                f32::INFINITY
            };
        }
        let (du, dv) = self.offset(i, at);
        let [cc, ca, cb, aa, ab, bb] = self.covariances[i];
        cc + 2.0 * (ca * du + cb * dv + ab * du * dv) + aa * du * du + bb * dv * dv
    }

    /// The variance of one reading on reading `i`'s plane: the noise's, and what a point
    /// anywhere in its pixel adds on a surface that leans as the plane does.
    fn reading(&self, i: usize) -> f32 {
        let lean = LATERAL * self.fits[i][1].hypot(self.fits[i][2]);
        self.noise[i] * self.noise[i] + lean * lean
    }

    /// How far reading `i`'s plane passes over reading `j`'s at `j`, in inverse depth (so
    /// positive where it passes nearer the camera), and the standard deviation of that.
    fn passes(&self, i: usize, j: usize) -> Pass {
        Pass {
            over: self.at(i, j) - self.fits[j][0],
            deviation: (self.variance(i, j) + self.variance(j, j)).sqrt(),
        }
    }

    /// Per pixel, the unit normal of its plane, toward the camera; zero without a plane. On up
    /// to `threads` threads.
    fn normals(&self, seen: &Seen, threads: usize) -> Vec<Vec3> {
        each_pixel(seen, threads, |i| {
            if !self.has[i] {
                return Vec3::ZERO;
            }
            let [c, a, b] = self.fits[i];
            seen.facing((i % self.width, i / self.width), [a, b, c])
        })
    }
}

/// How far one reading's plane passes over another reading ([`Planes::passes`]).
#[derive(Clone, Copy)]
struct Pass {
    over: f32,
    deviation: f32,
}

impl Pass {
    /// Whether it passes clear of the other reading, past what noise makes.
    fn clear(self) -> bool {
        self.over.abs() > SIGMAS * self.deviation
    }

    /// As seen from the other reading.
    fn back(self) -> Self {
        Self {
            over: -self.over,
            ..self
        }
    }
}

/// Gives readings in no part to the parts beside them: always next the reading that lies
/// nearest the plane of a part it is joined to, so each side of a crease takes the readings on
/// its own surface and the two meet at the crease. `surface` says, per pixel, the reading whose
/// plane it joined (itself to begin with). With `on_plane`, a reading joins only a plane it
/// lies on, within three deviations of the plane and of one reading; the rest stay in no part.
/// A reading on no known shape joins no surface of readings on one, and a reading on a shape
/// joins a surface of readings on none only where it lies nearer its plane than over the shape.
fn grow(
    walker: &Walker<'_>,
    planes: &Planes,
    of: &mut [u32],
    surface: &mut [usize],
    on_plane: bool,
) {
    let seen = walker.seen;
    // An offer of a plane to a reading in 64 bits: how far off the reading lies from the plane in
    // inverse depth, whose bits order as distances do, none being negative, then a pixel's
    // number, which an image's fit in 32 bits as a part's do.
    assert!(
        of.len() as u64 <= 1 << 32,
        "an image of 2^32 pixels at most"
    );
    let key = |off: f32, pixel: usize| u64::from(off.to_bits()) << 32 | pixel as u64;
    let pixel = |key: u64| (key & u64::from(u32::MAX)) as usize;
    // Per reading in no part, the nearest offer made to it yet, with the reading whose plane it
    // is, so that of two as near the lower numbered. An offer no nearer can never be the one
    // taken, and is not made.
    let mut nearest = vec![u64::MAX; of.len()];
    // Each offer made, with the reading offered: nearest first, then by pixel, so the order does
    // not depend on how the heap breaks ties. A reading is taken at its nearest offer, the last
    // made to it.
    let mut next = BinaryHeap::new();
    let offer = |next: &mut BinaryHeap<_>, nearest: &mut [u64], j: usize, plane: usize| {
        let off = (seen.inverse[j] - planes.at(plane, j)).abs();
        if key(off, plane) >= nearest[j] {
            return;
        }
        // A surface of readings on a known shape takes none that lies on no shape. One of
        // readings on none takes a reading on a shape only where it lies nearer the plane than
        // it lies over the shape, its distance from the shape taken as along its ray: the shape
        // is there beside it, whether or not a part of it is.
        let over = seen.off[j].max(0.0) * seen.inverse[j].powi(2);
        let apart = match (seen.kinds[j] == Kind::Open, seen.kinds[plane] == Kind::Open) {
            (true, to_open) => !to_open,
            (false, to_open) => to_open && off >= over,
        };
        if apart {
            return;
        }
        let deviation = (planes.variance(plane, j) + planes.reading(plane)).sqrt();
        if !on_plane || off <= SIGMAS * deviation {
            nearest[j] = key(off, plane);
            next.push(Reverse(key(off, j)));
        }
    };
    let open = |of: &[u32], j: usize| of[j] == NONE && seen.reads(j);
    for j in (0..of.len()).filter(|&j| open(of, j)) {
        for i in walker.neighbors(j) {
            if of[i] != NONE {
                offer(&mut next, &mut nearest, j, surface[i]);
            }
        }
    }
    while let Some(Reverse(offered)) = next.pop() {
        let i = pixel(offered);
        if of[i] != NONE {
            continue;
        }
        // An open reading's nearest offer is the first of its entries to come up.
        debug_assert_eq!(offered >> 32, nearest[i] >> 32);
        let plane = pixel(nearest[i]);
        of[i] = of[plane];
        surface[i] = plane;
        for j in walker.neighbors(i) {
            if open(of, j) {
                offer(&mut next, &mut nearest, j, plane);
            }
        }
    }
}

/// How two parts meet where no jump lies between them, in window widths of border.
#[derive(Clone, Copy, Default)]
struct Meeting {
    /// As one surface or in a convex edge, where the planes say so past the noise: a convex
    /// edge, or two planes that agree to within one reading's noise.
    one: f32,
    /// In a concave crease.
    crease: f32,
}

/// The readings of one part that lie in one mask, as the part and the mask (0: in none): what a
/// view's masks may make a part of its own ([`masks::split`]). Without masks, a part.
type Side = (u32, u32);

/// What the parts' borders say.
struct Borders {
    /// Per two parts with a border that no jump lies in, the lower numbered first.
    meetings: HashMap<(u32, u32), Meeting>,
    /// Per side, how much of its part's border it is the nearer side of, in window widths.
    hides: BTreeMap<Side, f32>,
    /// Per two sides of different parts that meet in a crease, the lower first: how long the
    /// crease is, in window widths. With it, how long they meet as one surface where one is of
    /// readings on a known shape and the other of readings on none.
    creases: BTreeMap<(Side, Side), f32>,
    /// Per two sides of different parts that border with no jump between, the lower first: how
    /// long a border, in window widths.
    touches: BTreeMap<(Side, Side), f32>,
    /// Each place along a border where one reading hides the other: the reading in front, which
    /// is in a part, then the one behind.
    hidden: Vec<(usize, usize)>,
}

/// How the parts border one another. Lengths are in window widths (`width` gives a pixel's), so
/// that one window's width of border counts as 1.
///
/// Across a jump, the nearer part hides the other. Where no jump lies between two parts, each
/// one's plane is held against the other's surface where that plane was taken: both pass in
/// front for a convex edge, both behind for a crease, one in front and one behind for a step,
/// where the one in front hides the other. Planes that pass neither way and are told to within
/// one reading's noise are one surface; told more loosely, they say nothing. A sliver has no
/// plane: its reading is held against its neighbor's, and where it lies clear of that plane it
/// is a step, the nearer of the two hiding the other. No step is told between a surface of
/// readings on a known shape and one of readings on none, nor by one reading on a shape.
///
/// What a part hides and the creases it meets in are kept by side, each crossing under the mask
/// its reading lies in (`masks`, a number per pixel or empty): added up over a part's sides they
/// are the part's, and a side that a mask cuts off takes its own with it.
fn borders(
    walker: &Walker<'_>,
    planes: &Planes,
    of: &[u32],
    surface: &[usize],
    masks: &[u32],
    width: &dyn Fn(usize) -> f32,
) -> Borders {
    let seen = walker.seen;
    let side = |i: usize| (of[i], masks.get(i).copied().unwrap_or(0));
    let mut hides: BTreeMap<Side, f32> = BTreeMap::new();
    let mut creases: BTreeMap<(Side, Side), f32> = BTreeMap::new();
    let mut touches: BTreeMap<(Side, Side), f32> = BTreeMap::new();
    let mut meetings: HashMap<(u32, u32), Meeting> = HashMap::new();
    let mut hidden = vec![];
    let reads = |i: usize| matches!(seen.kinds[i], Kind::Open | Kind::Known(_));
    for i in 0..of.len() {
        // East and south alone: each pixel of a straight border is then crossed once.
        for (bit, &direction) in DIRECTIONS.iter().enumerate().take(2) {
            let Some(j) = seen.step(i, direction) else {
                continue;
            };
            let (a, b) = (of[i], of[j]);
            if a == b || !reads(i) || !reads(j) {
                continue;
            }
            // The reading in front, where one is.
            let mut front = None;
            if walker.next(i, bit, /* forward */ true).is_none() {
                // A jump: the reading with the larger inverse depth is in front.
                front = Some(if seen.inverse[i] > seen.inverse[j] {
                    i
                } else {
                    j
                });
            } else if a != NONE && b != NONE {
                let (x, y) = (side(i), side(j));
                *touches.entry((x.min(y), x.max(y))).or_default() += 1.0 / width(i);
                let (own, other) = (surface[i], surface[j]);
                // How the other's plane passes this one's surface, and this one's the other's:
                // in front where positive.
                let passes = match (planes.has[own], planes.has[other]) {
                    (true, true) => Some((planes.passes(other, own), planes.passes(own, other))),
                    // A sliver's reading against its neighbor's plane, counted both ways.
                    (true, false) => Some((planes.passes(own, j).back(), planes.passes(own, j))),
                    (false, true) => {
                        Some((planes.passes(other, i), planes.passes(other, i).back()))
                    }
                    (false, false) => None,
                };
                let meeting = meetings.entry((a.min(b), a.max(b))).or_default();
                let length = 1.0 / width(i);
                if let Some((over, under)) = passes {
                    let agree = |pass: Pass| !pass.clear() && pass.deviation <= planes.noise[i];
                    // Across what may read on a known shape, the planes tell no step: of one
                    // surface's readings on either side of it, those on the shape are the
                    // farther ones, so the planes of the two differ by that choice alone. Nor
                    // does one reading on a shape, with no plane of its own. What meets as one
                    // surface across it is kept as a crease is, the two being no one part.
                    let open = |i: usize| seen.kinds[i] == Kind::Open;
                    let told = match (open(own), open(other)) {
                        (true, true) => true,
                        (false, false) => planes.has[own] && planes.has[other],
                        _ => false,
                    };
                    let mut meet = |across: bool| {
                        if across {
                            let (a, b) = (side(i), side(j));
                            *creases.entry((a.min(b), a.max(b))).or_default() += length;
                        }
                    };
                    if over.clear() && under.clear() {
                        if over.over > 0.0 && under.over > 0.0 {
                            meeting.one += length;
                        } else if over.over < 0.0 && under.over < 0.0 {
                            meeting.crease += length;
                            meet(true);
                        } else if told {
                            front = Some(if over.over > 0.0 { j } else { i });
                        }
                    } else if agree(over) && agree(under) {
                        meeting.one += length;
                        meet(open(own) != open(other));
                    }
                }
            }
            if let Some(front) = front.filter(|&front| of[front] != NONE) {
                *hides.entry(side(front)).or_default() += 1.0 / width(front);
                hidden.push((front, if front == i { j } else { i }));
            }
        }
    }
    Borders {
        meetings,
        hides,
        creases,
        touches,
        hidden,
    }
}
