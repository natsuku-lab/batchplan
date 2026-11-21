//! One look: every camera's frame in, the scene out ([`look`]).
//!
//! 1. The robot and the readings outside the look's corners are left out of each view, and each
//!    other reading is held against the shapes already known ([`Seen`]).
//! 2. Each view's readings are told apart into parts ([`segment`]), which the view's masks
//!    divide where one part lies in several ([`masks::split`]).
//! 3. A part that lies on a known shape and stands in front of none of it is the shape's
//!    ([`Told::known`]). Of the others, the parts of one view that lie in one mask are one
//!    object ([`masks::joins`]), as are parts of different views that show one surface
//!    ([`same`]) and a hollow's floor with its walls ([`hollows`]): but never two parts of one
//!    view that lie in different masks ([`Sets`]). A known shape's part is such an object's all
//!    the same where it lies inside it ([`Told::keep`]). An object that hangs over space a view
//!    sees is one with what it touches, as a handle is with the front it stands out of
//!    ([`attach`]).
//! 4. Each object is fitted a shape and stood on what lies under it ([`Fit`], [`fit::under`]),
//!    or on the lower thing it touches where the look cannot see under its middle
//!    ([`attach`]). One that holds a hollow's floor and is square from above is a box open at
//!    the top, its room what the look shows of it ([`Fit::open`], [`fit::opened`]). Where one
//!    stands on another and the two take up one place, the lower ends where the upper begins
//!    ([`part`]).
//! 5. Each object that lies where one of the scene before did keeps its id ([`continues`]).
//! 6. What no view could see of the scene before stays as that scene had it ([`shown`]).

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;

use batchplan::{DepthImage, Device, Intrinsics, Obstacle, Pose};
use glam::{Vec2, Vec3};

use crate::error::{Result, ensure_input};
use crate::fit::{self, Fit};
use crate::scene::{Label, Object, Scene, Shape, Support};
use crate::segment::{self, NONE, Parts};
use crate::view::{Kind, SIGMAS, Seen};
use crate::{masks, threads};

/// The standard deviation of a camera's depth reading, by the depth it reads (meters):
/// `constant + linear * depth + quadratic * depth * depth`. A stereo or structured-light
/// camera's grows with the square of the depth ([`Noise::stereo`]). A time-of-flight camera's
/// is a constant and a part that grows with the depth. Raise it to cover what the camera's pose
/// and the known shapes are off by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Noise {
    /// The part that is the same at every depth (meters).
    pub constant: f32,
    /// The part that grows with the depth (meters per meter).
    pub linear: f32,
    /// The part that grows with the square of the depth (meters per square meter).
    pub quadratic: f32,
}

impl Noise {
    /// A stereo or structured-light camera's, from what it is built as: the error of its
    /// matching in pixels (`subpixel`) over its focal length in pixels times its baseline in
    /// meters, growing with the square of the depth. Which camera it is stays the caller's to
    /// say: nothing here stands for one that is not known.
    ///
    /// ```
    /// // A camera of 640 pixels' focal length and 5 cm between its imagers, matching to 0.08
    /// // of a pixel: 2.5 mm at one meter, 10 mm at two.
    /// let noise = miru::Noise::stereo(640.0, 0.05, 0.08);
    /// assert_eq!((noise.constant, noise.linear), (0.0, 0.0));
    /// assert!((noise.quadratic - 0.0025).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn stereo(focal: f32, baseline: f32, subpixel: f32) -> Noise {
        Noise {
            constant: 0.0,
            linear: 0.0,
            quadratic: subpixel / (focal * baseline),
        }
    }

    /// The standard deviation of a reading at `depth` (meters).
    pub(crate) fn at(&self, depth: f32) -> f32 {
        self.constant + (self.linear + self.quadratic * depth) * depth
    }
}

/// One camera's frame.
#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    /// The camera's depth, its pinhole model and where it stood in the look's frame when the
    /// frame was taken, as batchplan's grids, maps and masks take them: meters along the
    /// camera's z axis, row by row, with zero, a negative or a value that is not finite where a
    /// pixel has no reading.
    pub image: DepthImage<'a>,
    /// Its colour: red, green and blue, a byte each, pixel by pixel in the depth's order. Empty
    /// for a camera without colour.
    pub color: &'a [u8],
    /// How exact the camera's depth is, from its datasheet. Every tolerance comes from it: how
    /// far a reading may lie from a known shape and be on it, how large a step in depth is a
    /// jump, and how wide the windows are that a surface's lean is taken over, so how small a
    /// crease is seen. A simulated camera states the noise of the camera it stands for: stated
    /// far below that, every facet of a mesh is a surface of its own.
    pub noise: Noise,
    /// Per pixel, whether its reading is the robot's or of what a hand holds, as whatever knows
    /// the robot's shape and configuration says: the `on_robot` of the mask that
    /// [`DepthImage::robot_mask`] gives. Such a reading is no object's. Empty: none is.
    pub robot: &'a [bool],
    /// Per pixel, in the depth's order, the number of the thing a mask model says the pixel
    /// shows, and 0 where it says none: instance masks, from whatever model the caller runs. A
    /// number means a thing in this view only. Readings that the depth shows as one surface and
    /// the masks as two things, or as a thing and none, are two where those not in the larger
    /// have as many readings as the noise needs to tell a surface by, and are as wide as the
    /// window a plane is taken over, or lie in a mask that is mostly theirs, or lie in a mask
    /// where the larger lies in another: otherwise they are the fringe of a mask, which a model
    /// never draws exact to the pixel, and go with the larger. What the view shows mostly in one
    /// mask is one object, whatever the depth shows between its parts. What it shows mostly in
    /// two masks is two objects, whatever another view shows of them. The robot's readings and those outside the look's
    /// corners stay what they are whatever a mask says, and what reads as a known shape is a
    /// mask's thing only where it lies inside that thing, as a plate's floor does: the table
    /// that a loose mask takes in stays the table's. Empty for a view without a model.
    pub masks: &'a [u32],
}

/// What one look takes in: every camera's frame from one moment, and what is known without
/// looking. All of it is in one frame whose z axis is up, as a robot's base frame is.
#[derive(Clone, Copy, Debug)]
pub struct Look<'a> {
    /// The cameras' frames. A scene's labels come per view, in this order.
    pub views: &'a [View<'a>],
    /// The shapes the world is already known to hold, as a cell's table and walls: the obstacles
    /// of the world the robot plans in, in the look's frame. What reads on one and stands in
    /// front of none of it is its own ([`Label::Known`]), and no object is made of it; a reading
    /// on one is an object's where the object's own surface runs through it, as the foot of a box
    /// and a plate's floor do. A scene names them by their index here. A shape given as a
    /// distance grid is known no finer than its voxels: a reading is on it by the grid's own
    /// distances, which a camera that reads to less than a voxel tells from the shape's.
    pub known: &'a [Obstacle],
}

/// How [`look`] reads the frames.
#[derive(Clone, Copy, Debug)]
pub struct LookOptions {
    /// The lower corner of the box readings count in: one outside it is no reading. As far as
    /// the robot reaches is enough. Unbounded by default.
    pub lower_corner: Vec3,
    /// Its upper corner.
    pub upper_corner: Vec3,
}

impl Default for LookOptions {
    fn default() -> Self {
        Self {
            lower_corner: Vec3::NEG_INFINITY,
            upper_corner: Vec3::INFINITY,
        }
    }
}

/// What the cameras see in `look`, as objects: which readings are one thing, its shape, its
/// colour and what it stands on, and for each pixel of each view what it shows.
///
/// An object that lies where one of `previous` did keeps that one's id; any other takes a new
/// one. The first look of a session passes `Scene::default()`.
///
/// The look runs on `device`'s threads and reads nothing of its robot. The scene does not depend
/// on how many threads there are: each view is read in bands and told apart by itself, and what
/// is done per object or per two views (their measuring, which of them show one surface, whether
/// a view shows where an object stood) is done item by item, the results taken in the items'
/// order.
///
/// What the look cannot see stays as `previous` had it. A view cannot see an object where its
/// pixels are the robot's or have no reading, where what they read stands before the object, and
/// where the object lies outside its image. An object of `previous` that none continues stays,
/// unseen ([`Object::seen`]), unless some view shows more of where it stood than is hidden from
/// it; and one that is seen in part keeps its shape, unless a view shows more of the rest than is
/// hidden from it, or its readings lie outside that shape. So a look with no views gives the
/// objects of `previous`, none seen.
///
/// Readings are told apart by geometry: where the depth jumps, and where two surfaces meet in a
/// concave crease, as things that touch or stand on one another do. One convex thing is one
/// object whatever it touches, and so is a round hollow one, as a bowl, whose outside and inside
/// both show enough to be told round about one axis. A thing that nothing under it holds up, as a
/// handle or a shelf's top, is one object with what it touches, where a view sees through the
/// space under it. A thing with creases of its own whose pieces each stand on what is under
/// them, as a step, comes out in pieces, unless a mask says they are one thing
/// ([`View::masks`]). A thing whose floor shows among walls that rise around it, as a tray's or a
/// drawer's pulled out, is one object with them, and where it is square from above its shape is
/// a box open at the top ([`Shape::OpenBox`]): its room reaches as far as a look shows it, a wall
/// whose inner face no view shows being as thick as what hides the floor before it, and a later
/// look that sees in makes the wall thinner.
///
/// miru runs no model. A caller whose model needs prompts, points or boxes, looks once without
/// masks, takes the prompts from that scene's labels, runs its model and looks again, after the
/// same `previous`, with the masks it gave.
pub fn look(device: &Device, look: &Look<'_>, previous: &Scene, o: &LookOptions) -> Result<Scene> {
    ensure_input!(
        o.lower_corner.cmplt(o.upper_corner).all(),
        "a look's corners need lower < upper, got {} and {}",
        o.lower_corner,
        o.upper_corner
    );
    for (index, view) in look.views.iter().enumerate() {
        check(index, view)?;
    }
    device.install(|| checked(device, look, previous, o, rayon::current_num_threads()))
}

/// [`look`] of frames and options that have been checked, on `device`'s threads, of which there
/// are `threads`.
fn checked(
    device: &Device,
    look: &Look<'_>,
    previous: &Scene,
    o: &LookOptions,
    threads: usize,
) -> Result<Scene> {
    let seen = (look.views.iter()).map(|view| Seen::of(device, view, look.known, o, threads));
    let seen: Vec<Seen> = seen.collect::<Result<_>>()?;
    let parts: Vec<Parts> = threads::each(threads, seen.len(), |view| {
        let threads = threads::share(threads, seen.len(), view);
        segment::parts(&seen[view], look.views[view].masks, threads)
    });

    // Each object's parts, by view; an object is named by the first of its parts.
    let first: Vec<usize> = parts
        .iter()
        .scan(0, |next, p| {
            let first = *next;
            *next += p.count;
            Some(first)
        })
        .collect();
    // What each view's masks say are one thing first, which nothing refuses; then what the
    // views show of one another, where no view's masks say otherwise.
    let lies: Vec<Vec<u32>> = (parts.iter().zip(look.views))
        .map(|(parts, view)| masks::lies(&parts.of, parts.count, &parts.share, view.masks))
        .collect();
    let told = Told::new(look, &seen, &parts, &first, threads);
    let mut sets = Sets::new(&lies, told.known());
    for (lies, first) in lies.iter().zip(&first) {
        for mask in masks::joins(lies) {
            // With the first of them that is no known shape's alone: such a part joins none.
            if let Some(object) = mask.iter().find(|&&part| !sets.theirs[first + part]) {
                for part in &mask {
                    sets.join(first + object, first + part);
                }
            }
        }
    }
    same(&seen, &parts, &first, &mut sets, threads);
    let (floors, _) = told.gather(&mut sets, /* with floors */ true);
    // What each object says by itself, measured once; a known shape's part is no object's.
    let said: Vec<Option<Measured>> = threads::each(threads, floors.len(), |object| {
        let readings = &floors[object];
        (!sets.theirs[readings.set]).then(|| Measured::of(readings, previous))
    });
    let mut floor_parts = hollows(&mut sets, &told, &floors, &said);
    let (mut objects, mut labels) = told.gather(&mut sets, /* with floors */ false);
    let before: Vec<(&Readings, &Measured)> = (floors.iter().zip(&said))
        .filter_map(|(readings, said)| Some((readings, said.as_ref()?)))
        .collect();
    let mut measured = remeasure(&objects, &before, previous, threads);
    if told.keep(&mut sets, &objects, &measured) {
        let (after, shown) = told.gather(&mut sets, /* with floors */ false);
        let before: Vec<_> = objects.iter().zip(&measured).collect();
        measured = remeasure(&after, &before, previous, threads);
        (objects, labels) = (after, shown);
    }

    // Nothing was seen lower than the lowest reading, give or take its noise, so nothing lower
    // holds anything up.
    let mut floor = f32::INFINITY;
    for readings in &seen {
        for (i, kind) in readings.kinds.iter().enumerate() {
            if matches!(kind, Kind::Known(_) | Kind::Open) {
                floor = floor.min(readings.points[i].z - SIGMAS * readings.sigma(i));
            }
        }
    }
    // What hangs over free space is held up by what it touches, and what the look cannot see
    // under stands on what it touches.
    let (joined, stands) = attach(
        &mut sets, &told, &objects, &measured, &labels, floor, previous,
    );
    if joined {
        let (after, shown) = told.gather(&mut sets, /* with floors */ false);
        let before: Vec<_> = objects.iter().zip(&measured).collect();
        measured = remeasure(&after, &before, previous, threads);
        (objects, labels) = (after, shown);
    }
    // What the scene before had as one thing and the robot now divides is one thing still.
    if rejoin(&mut sets, &objects, &measured, &labels, &seen, previous) {
        let (after, shown) = told.gather(&mut sets, /* with floors */ false);
        let before: Vec<_> = objects.iter().zip(&measured).collect();
        measured = remeasure(&after, &before, previous, threads);
        (objects, labels) = (after, shown);
    }
    let mut fits: Vec<Fit> = measured.iter().map(|said| said.fit).collect();
    // Per object, the one it stands on that way.
    let mut held = vec![None; objects.len()];
    let index: HashMap<usize, usize> = (objects.iter().enumerate())
        .map(|(object, readings)| (readings.set, object))
        .collect();
    // A box a view sees into holds a floor too, whether or not the floor hides anything.
    let square: Vec<bool> = (fits.iter())
        .map(|fit| matches!(fit.shape(), Shape::Cuboid { .. }))
        .collect();
    told.sees_into(&labels, &objects, &square, &mut floor_parts, threads);
    // What holds a hollow's floor among walls and is a box from above is open at the top.
    let mut floored: Vec<usize> = (0..floor_parts.len())
        .filter(|&part| floor_parts[part])
        .filter_map(|part| index.get(&sets.find(part)).copied())
        .filter(|&object| matches!(fits[object].shape(), Shape::Cuboid { .. }))
        .collect();
    floored.sort_unstable();
    floored.dedup();
    let floored: Vec<(usize, Vec<fit::Reading>)> = if floored.is_empty() {
        vec![]
    } else {
        let readings = told.readings(&labels, &floor_parts, &floored);
        floored.into_iter().zip(readings).collect()
    };
    for (object, readings) in &floored {
        fits[*object].open(readings);
    }
    for (object, under) in stands {
        let (object, under) = (index[&sets.find(object)], index[&sets.find(under)]);
        // The two may be one thing since: what came back in pieces stands on nothing of its own.
        if object != under {
            held[object] = Some(under);
        }
    }

    // Each object's size as the look's readings alone give its shape, stood on what they show
    // under it and parted from it: what the shapes of `previous`, which were made so, are held
    // against.
    let alone: Vec<(Shape, f32)> = (fits.iter())
        .map(|fit| (fit.shape(), fit.lowest.z))
        .collect();
    let stood_alone = stand(&fits, &alone, look.known, floor, &held, &|_, _| false);
    let made: Vec<Shape> = stood_alone.iter().map(|(fit, _)| fit.shape()).collect();
    // Where one thing stands on another and the two take up one place, decided once, from what
    // this look's readings alone give: the same cuts make these sizes and the shapes at the end.
    let cuts = part(&made, &objects, &stood_alone);
    let sizes: Vec<f32> = (cut(&made, &cuts, &vec![None; fits.len()]).iter())
        .map(|shape| shape.outer().volume())
        .collect();

    // What stays as `previous` had it: the shape of an object seen in part, where its readings
    // lie in that shape and no view shows the rest of it; and the objects that none continues
    // and no view shows.
    let was = continues(&measured, &sizes, previous);
    let mut kept: Vec<Option<Shape>> = threads::each(threads, objects.len(), |object| {
        let before = previous.objects[was[object]?].shape;
        // Readings that add up to a window lie outside it: more than noise makes.
        let outgrown = objects[object].outside(&before) >= 1.0;
        (!outgrown && !shown(&before, Some(object), &seen, &labels)).then_some(before)
    });
    // A kept shape is opened where this look shows its hollow's floor, as a fit is.
    for (object, readings) in &floored {
        if let Some(before) = kept[*object] {
            kept[*object] = Some(fit::opened(&before, readings).unwrap_or(before));
        }
    }
    // One measured afresh keeps the room it had, as what a look cannot see stays.
    let carried: Vec<usize> = (0..objects.len())
        .filter(|&object| kept[object].is_none())
        .filter(|&object| {
            was[object].is_some_and(|before| {
                matches!(previous.objects[before].shape, Shape::OpenBox { .. })
            })
        })
        .collect();
    if !carried.is_empty() {
        let readings = told.readings(&labels, &floor_parts, &carried);
        for (object, readings) in carried.into_iter().zip(readings) {
            let before = was[object].expect("a carried object continues one");
            fits[object].carry(&previous.objects[before].shape, &readings);
        }
    }
    let unseen: Vec<usize> = (0..previous.objects.len())
        .filter(|before| !was.contains(&Some(*before)))
        .filter(|&before| !shown(&previous.objects[before].shape, None, &seen, &labels))
        .collect();

    // Every object's shape and the height of its lowest point: those seen, then those unseen.
    let bodies = |fits: &[Fit], kept: &[Option<Shape>]| -> Vec<(Shape, f32)> {
        (fits.iter().zip(kept))
            .map(|(fit, kept)| (kept.unwrap_or_else(|| fit.shape()), fit.lowest.z))
            .chain(unseen.iter().map(|&before| {
                let shape = previous.objects[before].shape;
                (shape, shape.lowest())
            }))
            .collect()
    };
    // An object no view saw still stands on what it stood on, so it holds none of that up: not
    // the object it stood on, nor one that object stood on in its turn.
    let seen_count = fits.len();
    let rested = |object: usize, body: usize| {
        let body = body.checked_sub(seen_count).map(|kept| unseen[kept]);
        (body.zip(was[object])).is_some_and(|(body, object)| stood_on(previous, body, object))
    };
    let mut stood = stand(
        &fits,
        &bodies(&fits, &kept),
        look.known,
        floor,
        &held,
        &rested,
    );
    let hide = told.hides_floors(&labels, &floor_parts);
    let closed = hold_what_stands(&objects, &mut fits, &mut kept, &stood, &hide);
    // A room widened or closed changes what the things in it stand on.
    let now = bodies(&fits, &kept);
    if (now.iter().zip(&closed))
        .any(|((shape, _), &closed)| closed || matches!(shape, Shape::OpenBox { .. }))
    {
        stood = stand(&fits, &now, look.known, floor, &held, &rested);
    }

    // The scene says the id after its newest, which must be a number too: the last id a `u32`
    // holds is never given.
    let new = was.iter().filter(|was| was.is_none()).count();
    ensure_input!(
        u32::try_from(new).is_ok_and(|new| previous.next_id.checked_add(new).is_some()),
        "the ids have run out: the scene before has a `next_id` of {} and the look {new} new \
         objects, more than a `u32` has left",
        previous.next_id
    );
    let mut next_id = previous.next_id;
    let shapes: Vec<Shape> = (kept.iter().zip(&stood))
        .map(|(kept, (fit, _))| kept.unwrap_or_else(|| fit.shape()))
        .collect();
    let shapes = cut(&shapes, &cuts, &kept);
    let mut all: Vec<Object> = (objects.iter().enumerate())
        .map(|(k, readings)| Object {
            id: was[k].map_or_else(
                || {
                    next_id += 1;
                    next_id - 1
                },
                |before| previous.objects[before].id,
            ),
            shape: shapes[k],
            // What showed its inside once is not made solid by a look that cannot see into it,
            // nor what was open at the top by a look that leaves it a box that covers it; one
            // open at the top shows the room it holds.
            solid: !closed[k]
                && (matches!(shapes[k], Shape::OpenBox { .. })
                    || (was[k].is_none_or(|before| {
                        let before = &previous.objects[before];
                        before.solid && !matches!(before.shape, Shape::OpenBox { .. })
                    }) && readings.solid(&shapes[k]))),
            color: readings.color(),
            on: stood[k].1,
            seen: true,
        })
        .collect();
    // Where an object of `previous` is in this scene, if it is.
    let now = |before: usize| {
        let continued = was.iter().position(|was| *was == Some(before));
        let carried = || Some(fits.len() + unseen.iter().position(|kept| *kept == before)?);
        continued.or_else(carried)
    };
    all.extend(unseen.iter().map(|&before| {
        let object = &previous.objects[before];
        Object {
            on: match object.on {
                Some(Support::Object(under)) => now(under).map(Support::Object),
                on => on,
            },
            seen: false,
            ..object.clone()
        }
    }));
    Ok(Scene {
        objects: all,
        labels,
        next_id,
    })
}

/// A look's views told apart into parts.
struct Told<'a> {
    look: &'a Look<'a>,
    seen: &'a [Seen],
    parts: &'a [Parts],
    /// Each view's first part among all the views'.
    first: &'a [usize],
    /// The views' rows in bands, which [`Told::gather`] reads on up to `threads` threads.
    bands: Vec<Band>,
    /// The parts that have readings, by their number among all the views', in the order the
    /// views show them: view by view, and in a view as the rows meet the first reading of each.
    order: Vec<usize>,
    threads: usize,
}

/// A band of one view's rows.
struct Band {
    view: usize,
    pixels: Range<usize>,
    /// Per part of the view, how many of its readings lie in the band.
    readings: Vec<usize>,
}

/// The object of a part that is none's ([`Told::gather`]).
const NO_OBJECT: usize = usize::MAX;

impl<'a> Told<'a> {
    /// The views of `look`, read as `seen`, told apart into `parts`, each view's first part
    /// among all the views' at `first`, on up to `threads` threads.
    fn new(
        look: &'a Look<'a>,
        seen: &'a [Seen],
        parts: &'a [Parts],
        first: &'a [usize],
        threads: usize,
    ) -> Self {
        // Each view's rows in as many bands as there are threads.
        let mut pixels: Vec<(usize, Range<usize>)> = vec![];
        for (view, seen) in seen.iter().enumerate() {
            let band = (seen.height.div_ceil(threads) * seen.width).max(1);
            let all = seen.kinds.len();
            let starts = (0..all).step_by(band);
            pixels.extend(starts.map(|start| (view, start..all.min(start + band))));
        }
        // Per band, per part of its view, how many of its readings lie in the band and where
        // the first does.
        let counted = threads::each(threads, pixels.len(), |band| {
            let (view, pixels) = &pixels[band];
            let (seen, of) = (&seen[*view], &parts[*view]);
            let (mut readings, mut firsts) = (vec![0; of.count], vec![usize::MAX; of.count]);
            let own = of.of[pixels.clone()]
                .iter()
                .zip(&seen.kinds[pixels.clone()]);
            for (i, (&part, kind)) in (pixels.start..).zip(own) {
                if part != NONE && matches!(kind, Kind::Known(_) | Kind::Open) {
                    let part = part as usize;
                    if readings[part] == 0 {
                        firsts[part] = i;
                    }
                    readings[part] += 1;
                }
            }
            (readings, firsts)
        });
        let mut order = vec![];
        for (view, of) in parts.iter().enumerate() {
            let mut met = vec![usize::MAX; of.count];
            let bands = (pixels.iter().zip(&counted)).filter(|((band, _), _)| *band == view);
            for (_, (_, firsts)) in bands {
                for (met, &first) in met.iter_mut().zip(firsts) {
                    *met = (*met).min(first);
                }
            }
            let mut shown: Vec<(usize, usize)> = (met.into_iter().enumerate())
                .filter(|&(_, at)| at != usize::MAX)
                .map(|(part, at)| (at, first[view] + part))
                .collect();
            shown.sort_unstable();
            order.extend(shown.into_iter().map(|(_, part)| part));
        }
        let bands = (pixels.into_iter().zip(counted))
            .map(|((view, pixels), (readings, _))| Band {
                view,
                pixels,
                readings,
            })
            .collect();
        Told {
            look,
            seen,
            parts,
            first,
            bands,
            order,
            threads,
        }
    }

    /// Each object's readings, in the order the views show the objects, each with the set its
    /// parts are; and per view and pixel what the pixel shows. A part that is a known shape's
    /// ([`Sets::theirs`]) is no object's: its readings show what a reading in no part does, the
    /// shape it lies on or nothing. With `floors`, such a part is an object of its own, for
    /// [`hollows`] to make a floor of.
    ///
    /// The bands are read on threads of their own, each into its own stretch of each object's
    /// readings, so that every object's are in the order the views and their rows show them.
    fn gather(&self, sets: &mut Sets, floors: bool) -> (Vec<Readings>, Vec<Vec<Label>>) {
        // Per part, its object, the objects numbered as the views show them; per object, its
        // set.
        let mut object_of = vec![NO_OBJECT; sets.first.len()];
        let mut index = vec![NO_OBJECT; sets.first.len()];
        let mut sets_of: Vec<usize> = vec![];
        for &part in &self.order {
            if floors || !sets.theirs[part] {
                let set = sets.find(part);
                if index[set] == NO_OBJECT {
                    index[set] = sets_of.len();
                    sets_of.push(set);
                }
                object_of[part] = index[set];
            }
        }
        let count = sets_of.len();
        // Per band, how many readings of each object lie in it.
        let within: Vec<Vec<usize>> = (self.bands.iter())
            .map(|band| {
                let mut within = vec![0; count];
                let first = self.first[band.view];
                for (part, &readings) in band.readings.iter().enumerate() {
                    if let Some(within) = within.get_mut(object_of[first + part]) {
                        *within += readings;
                    }
                }
                within
            })
            .collect();
        let mut objects: Vec<Readings> = (sets_of.into_iter().enumerate())
            .map(|(object, set)| {
                let all = within.iter().map(|within| within[object]).sum();
                Readings {
                    set,
                    points: vec![Vec3::ZERO; all],
                    slack: vec![0.0; all],
                    share: vec![0.0; all],
                    colors: ([0; 3], 0),
                }
            })
            .collect();
        let mut labels: Vec<Vec<Label>> = (self.seen.iter())
            .map(|seen| vec![Label::Nothing; seen.kinds.len()])
            .collect();
        // Per band, its labels, and per object the stretch of its readings the band fills.
        type Stretch<'b> = (&'b mut [Vec3], &'b mut [f32], &'b mut [f32]);
        let mut rest: Vec<&mut [Label]> = labels.iter_mut().map(Vec::as_mut_slice).collect();
        let mut pieces: Vec<(&mut [Label], Vec<Stretch<'_>>)> = (self.bands.iter())
            .map(|band| {
                let (own, after) =
                    std::mem::take(&mut rest[band.view]).split_at_mut(band.pixels.len());
                rest[band.view] = after;
                (own, Vec::with_capacity(count))
            })
            .collect();
        for (object, readings) in objects.iter_mut().enumerate() {
            let mut points = readings.points.as_mut_slice();
            let mut slack = readings.slack.as_mut_slice();
            let mut share = readings.share.as_mut_slice();
            for ((_, stretches), within) in pieces.iter_mut().zip(&within) {
                let (own_points, after) = std::mem::take(&mut points).split_at_mut(within[object]);
                points = after;
                let (own_slack, after) = std::mem::take(&mut slack).split_at_mut(within[object]);
                slack = after;
                let (own_share, after) = std::mem::take(&mut share).split_at_mut(within[object]);
                share = after;
                stretches.push((own_points, own_slack, own_share));
            }
        }
        // Per band, per object, the sum of its readings' colours there and how many had one.
        let mut colors = vec![vec![([0_u64; 3], 0_u64); count]; self.bands.len()];
        let work: Vec<_> = pieces.into_iter().zip(colors.iter_mut()).collect();
        threads::each_piece(
            self.threads,
            work,
            |number, ((labels, mut stretches), colors)| {
                let band = &self.bands[number];
                let (seen, parts) = (&self.seen[band.view], &self.parts[band.view]);
                let (color, first) = (self.look.views[band.view].color, self.first[band.view]);
                let mut next = vec![0; count];
                for (label, i) in labels.iter_mut().zip(band.pixels.clone()) {
                    let part = parts.of[i];
                    let object = if part == NONE {
                        NO_OBJECT
                    } else {
                        object_of[first + part as usize]
                    };
                    *label = match (seen.kinds[i], object) {
                        (Kind::Nothing, _) => Label::Nothing,
                        (Kind::Robot, _) => Label::Robot,
                        (Kind::Known(shape), NO_OBJECT) => Label::Known(shape),
                        (Kind::Open, NO_OBJECT) => Label::Nothing,
                        (Kind::Known(_) | Kind::Open, object) => {
                            let (points, slack, share) = &mut stretches[object];
                            let at = next[object];
                            next[object] += 1;
                            points[at] = seen.points[i];
                            slack[at] = SIGMAS * seen.sigma(i);
                            share[at] = parts.share[i];
                            if let Some(color) = color.get(3 * i..3 * i + 3) {
                                let (sums, had) = &mut colors[object];
                                for (sum, &byte) in sums.iter_mut().zip(color) {
                                    *sum += u64::from(byte);
                                }
                                *had += 1;
                            }
                            Label::Object(object)
                        }
                    };
                }
            },
        );
        for (object, readings) in objects.iter_mut().enumerate() {
            for colors in &colors {
                let (sums, had) = colors[object];
                for (sum, add) in readings.colors.0.iter_mut().zip(sums) {
                    *sum += add;
                }
                readings.colors.1 += had;
            }
        }
        (objects, labels)
    }

    /// Marks in `floors`, per part, the floors a view sees into.
    ///
    /// A solid box shows a camera above it nothing under its top, so what of a box faces up
    /// under walls of its own is the floor of a hollow. A part of an object that `square` says is
    /// fitted square is such a floor where more than half of its planes face up, more than half
    /// of their normal, and:
    ///
    /// - the object rises over it: its readings higher than the part's level by more than their
    ///   slack add up to a window or more. The level is the height more than half of the part's
    ///   readings lie at or under ([`fit::level`]): the strips of a wall that are read as the
    ///   floor's edges are of its part and rise with the rest. Fewer readings over it cannot be
    ///   told from noise, as those that stray over a box's top on a camera's depth;
    /// - none of what rises stands over the floor itself, inside the least rectangle from above
    ///   around what of the part lies at its level by more than its slack, but for less than a
    ///   window's worth: that rectangle is what a room over the floor spans
    ///   ([`fit::Fit::open`]). Walls stand around a floor; what rises out of the middle of one, as
    ///   a burner out of a stove's top, or beside a floor cut aslant, as the higher half of a
    ///   box that rests tilted, leaves no room a box can hold;
    /// - the floor takes up more than half of the object from above, by the least outlines
    ///   around the two: a box is open at the top where a view sees into most of it, and a notch
    ///   in a thing, as between a cabinet's handles, leaves it the box that covers it.
    ///
    /// [`hollows`] finds the floors that are things of their own and hide nothing, however
    /// small; these are a part of their object already, as a drawer's floor seen from over it
    /// with its walls.
    fn sees_into(
        &self,
        labels: &[Vec<Label>],
        objects: &[Readings],
        square: &[bool],
        floors: &mut [bool],
        threads: usize,
    ) {
        // The cheap question first, of every part of a square object: whether anything of the
        // object reads higher than the part's lowest by more than the least slack of the
        // object's readings. Most parts that face up are a box's top, with nothing over it.
        // Per object, its highest reading and its readings' least slack; per part, its object,
        // its planes, those that face up, and its lowest reading.
        let over: Vec<(f32, f32)> = (objects.iter())
            .map(|readings| {
                let highest = readings.points.iter().map(|point| point.z);
                let least = readings.slack.iter().copied();
                (
                    highest.fold(f32::MIN, f32::max),
                    least.fold(f32::MAX, f32::min),
                )
            })
            .collect();
        let parts = threads::each(threads, self.seen.len(), |view| {
            let (seen, of) = (&self.seen[view], &self.parts[view]);
            let next = self.first.get(view + 1).copied().unwrap_or(floors.len());
            let mut parts = vec![(usize::MAX, 0_u32, 0_u32, f32::MAX); next - self.first[view]];
            for (i, label) in labels[view].iter().enumerate() {
                let Label::Object(object) = *label else {
                    continue;
                };
                if !square[object] {
                    continue;
                }
                let (owner, all, up, lowest) = &mut parts[of.of[i] as usize];
                let normal = of.normals[i];
                *owner = object;
                *all += u32::from(normal != Vec3::ZERO);
                *up += u32::from(normal.z > 0.5);
                *lowest = lowest.min(seen.points[i].z);
            }
            parts
        })
        .concat();
        let may: Vec<usize> = (0..parts.len())
            .filter(|&part| {
                let (object, all, up, lowest) = parts[part];
                object != usize::MAX && 2 * up > all && over[object].0 > lowest + over[object].1
            })
            .collect();
        if may.is_empty() {
            return;
        }
        // Of the objects such a part is of, every part's readings: place, slack and share of a
        // window.
        let mut wanted = vec![false; objects.len()];
        for &part in &may {
            wanted[parts[part].0] = true;
        }
        let mut readings: Vec<Vec<(Vec3, f32, f32)>> = vec![Vec::new(); parts.len()];
        for (view, ((seen, of), labels)) in self.seen.iter().zip(self.parts).zip(labels).enumerate()
        {
            for (i, label) in labels.iter().enumerate() {
                if matches!(*label, Label::Object(object) if wanted[object]) {
                    let part = self.first[view] + of.of[i] as usize;
                    readings[part].push((seen.points[i], SIGMAS * seen.sigma(i), of.share[i]));
                }
            }
        }
        // Per object such a part is of, its parts.
        let mut of_object: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (part, &(object, ..)) in parts.iter().enumerate() {
            if object != usize::MAX && wanted[object] {
                of_object.entry(object).or_default().push(part);
            }
        }
        // The least outline around what lies at `points`, from above; and the farthest of
        // them toward -x, -y, +x and +y, which lie on it, counterclockwise.
        let outline = |points: &mut dyn Iterator<Item = Vec2>| fit::hull(points.collect());
        let farthest = |points: &mut dyn Iterator<Item = Vec2>| {
            let mut ends: Option<[Vec2; 4]> = None;
            for point in points {
                let ends = ends.get_or_insert([point; 4]);
                if point.x < ends[0].x {
                    ends[0] = point;
                }
                if point.y < ends[1].y {
                    ends[1] = point;
                }
                if point.x > ends[2].x {
                    ends[2] = point;
                }
                if point.y > ends[3].y {
                    ends[3] = point;
                }
            }
            ends.expect("a part has a reading")
        };
        // Per object, what its farthest readings span, which is no more than its outline's
        // area, and that area once a part needs it. The questions that take one pass over the
        // readings come before those that put them in order.
        let mut wholes: BTreeMap<usize, (f32, Option<f32>)> = BTreeMap::new();
        for part in may {
            let object = parts[part].0;
            let whole = || {
                of_object[&object]
                    .iter()
                    .flat_map(|&other| readings[other].iter())
            };
            let own = &readings[part];
            let (span, exact) = wholes.entry(object).or_insert_with(|| {
                let ends = farthest(&mut whole().map(|reading| reading.0.truncate()));
                (fit::area(&ends), None)
            });
            // The box around the part along x and y is no less than the floor's outline.
            let (least, most) =
                own.iter()
                    .fold((Vec2::MAX, Vec2::MIN), |(least, most), reading| {
                        let flat = reading.0.truncate();
                        (least.min(flat), most.max(flat))
                    });
            if 2.0 * (most - least).element_product() <= *span {
                continue;
            }
            let level = fit::level(own.iter().map(|reading| reading.0.z))
                .expect("a part of an object has a reading");
            // What of the object rises over the level: where, with its slack and share.
            let over: Vec<(Vec2, f32, f32)> = whole()
                .filter(|&&(point, slack, _)| point.z > level + slack)
                .map(|&(point, slack, share)| (point.truncate(), slack, share))
                .collect();
            if over.iter().map(|reading| reading.2).sum::<f32>() < 1.0 {
                continue;
            }
            // What of that stands over the floor, inside the least rectangle around it, which
            // a room over it would span: first inside what the floor's farthest readings span,
            // which lies in it.
            let at_level: Vec<Vec2> = (own.iter())
                .filter(|&&(point, slack, _)| point.z <= level + slack)
                .map(|reading| reading.0.truncate())
                .collect();
            let stands = |inside: &dyn Fn(Vec2, f32) -> bool| {
                let standing = over.iter().filter(|&&(at, slack, _)| inside(at, slack));
                standing.map(|reading| reading.2).sum::<f32>() >= 1.0
            };
            let ends = farthest(&mut at_level.iter().copied());
            if stands(&|at, slack| fit::holds(&ends, at, slack)) {
                continue;
            }
            let floor = fit::hull(at_level);
            let exact = *exact.get_or_insert_with(|| {
                fit::area(&outline(&mut whole().map(|reading| reading.0.truncate())))
            });
            if 2.0 * fit::area(&floor) <= exact {
                continue;
            }
            let (middle, half, yaw) = fit::rectangle(&floor);
            let turn = Vec2::from_angle(-yaw);
            floors[part] |=
                !stands(&|at, slack| (half - turn.rotate(at - middle).abs()).min_element() > slack);
        }
    }

    /// Per object of `objects`, its readings as the room of a box open at the top is measured from
    /// them ([`fit::Reading`]): the pixels `labels` give it. A reading is of the floor where its
    /// part is one that `floors`, per part, says is a hollow's floor, more than half of that
    /// part's planes face up, more than half of their normal, and it lies no higher than the
    /// part's level by more than its slack ([`fit::level`]): a wall's inner face that hides
    /// nothing can be taken for a floor, and faces sideways; and the strips of a wall that are
    /// read as a floor's edges are of its part, and rise over it.
    fn readings(
        &self,
        labels: &[Vec<Label>],
        floors: &[bool],
        objects: &[usize],
    ) -> Vec<Vec<fit::Reading>> {
        // Per object, its place among `objects`, where it is one of them.
        let mut slot = vec![usize::MAX; objects.iter().max().map_or(0, |most| most + 1)];
        for (place, &object) in objects.iter().enumerate() {
            slot[object] = place;
        }
        // Per object, its readings and each one's part; per part of a floor, its planes, how
        // many of them face up, and its readings' heights.
        let mut readings: Vec<Vec<(fit::Reading, usize)>> = vec![vec![]; objects.len()];
        let mut planes = vec![(0_usize, 0_usize); floors.len()];
        let mut heights: Vec<Vec<f32>> = vec![Vec::new(); floors.len()];
        for (view, ((seen, parts), labels)) in
            self.seen.iter().zip(self.parts).zip(labels).enumerate()
        {
            for (i, label) in labels.iter().enumerate() {
                let Label::Object(object) = *label else {
                    continue;
                };
                let Some(&slot) = slot.get(object).filter(|&&slot| slot != usize::MAX) else {
                    continue;
                };
                let part = self.first[view] + parts.of[i] as usize;
                let normal = parts.normals[i];
                if floors[part] {
                    heights[part].push(seen.points[i].z);
                    planes[part].0 += usize::from(normal != Vec3::ZERO);
                    planes[part].1 += usize::from(normal.z > 0.5);
                }
                let reading = fit::Reading {
                    point: seen.points[i],
                    normal,
                    slack: SIGMAS * seen.sigma(i),
                    share: parts.share[i],
                    floor: floors[part],
                };
                readings[slot].push((reading, part));
            }
        }
        // Per part of a floor that faces up, its level.
        let levels: Vec<Option<f32>> = (heights.into_iter().zip(&planes))
            .map(|(heights, &(all, up))| fit::level(heights.into_iter()).filter(|_| 2 * up > all))
            .collect();
        (readings.into_iter())
            .map(|readings| {
                (readings.into_iter())
                    .map(|(reading, part)| fit::Reading {
                        floor: reading.floor
                            && levels[part]
                                .is_some_and(|level| reading.point.z <= level + reading.slack),
                        ..reading
                    })
                    .collect()
            })
            .collect()
    }

    /// Per two objects of `labels`, how long a border the first stands in front of the second's
    /// floor along (a part that `floors`, per part, says is a hollow's floor), in window widths:
    /// what hides the floor stands on it or over it.
    fn hides_floors(
        &self,
        labels: &[Vec<Label>],
        floors: &[bool],
    ) -> BTreeMap<(usize, usize), f32> {
        let mut hide = BTreeMap::new();
        for (view, (parts, labels)) in self.parts.iter().zip(labels).enumerate() {
            for &(front, back) in &parts.hidden {
                let (Label::Object(over), Label::Object(under)) = (labels[front], labels[back])
                else {
                    continue;
                };
                let part = parts.of[back];
                if over != under && part != NONE && floors[self.first[view] + part as usize] {
                    *hide.entry((over, under)).or_default() += parts.share[front].sqrt();
                }
            }
        }
        hide
    }

    /// Per part of every view, whether it is a known shape's alone: before any object takes it
    /// for its own ([`Told::keep`]).
    ///
    /// What is a known shape's lies on it and stands in front of none of it: a surface cannot
    /// be in front of itself. So a part is a known shape's when more than half of its readings
    /// lie on known shapes and it stands in front of the shape it lies on most along less than
    /// a window's width of border: in front of readings that lie inside the shape, deeper than
    /// a reading on it may read. Of two readings on its surface, the noise does not say which
    /// is in front. A part with a window's worth of readings so deep inside known shapes is the
    /// shape's whatever it stands in front of: nothing else can be there.
    fn known(&self) -> Vec<bool> {
        let shapes = self.look.known.len();
        let mut theirs = vec![];
        for (seen, parts) in self.seen.iter().zip(self.parts) {
            // Per part: its readings; per known shape, those that lie on it and how much of the
            // part's border lies before readings deep inside it, in window widths; and how much
            // of the part lies deep inside known shapes, in windows.
            let mut readings = vec![0; parts.count];
            let mut on = vec![vec![0; shapes]; parts.count];
            let mut before = vec![vec![0.0; shapes]; parts.count];
            let mut inside = vec![0.0; parts.count];
            let deep = |i: usize| seen.off[i] < -SIGMAS * seen.sigma(i);
            for (i, &part) in parts.of.iter().enumerate() {
                if part != NONE {
                    readings[part as usize] += 1;
                    if let Kind::Known(shape) = seen.kinds[i] {
                        on[part as usize][shape] += 1;
                        if deep(i) {
                            inside[part as usize] += parts.share[i];
                        }
                    }
                }
            }
            for &(front, back) in &parts.hidden {
                let part = parts.of[front];
                // Of two readings on a shape's surface, the noise does not say which is in
                // front: both read where the surface may.
                if let Kind::Known(shape) = seen.kinds[back]
                    && part != NONE
                    && deep(back)
                {
                    before[part as usize][shape] += parts.share[front].sqrt();
                }
            }
            theirs.extend((0..parts.count).map(|part| {
                // Of two shapes with as many, the first.
                let most = (0..shapes).max_by_key(|&shape| (on[part][shape], Reverse(shape)));
                2 * on[part].iter().sum::<usize>() > readings[part]
                    && most.is_some_and(|most| before[part][most] < 1.0 || inside[part] >= 1.0)
            }));
        }
        theirs
    }

    /// Makes each part that is a known shape's alone the object's it was to be one with
    /// ([`Sets::asked`]), where more than half of its readings lie inside the shape of that
    /// object's other parts, and says whether any was made so: a plate's floor that reads on
    /// the table lies inside its rim, and the table a loose mask takes in with a bowl does not
    /// lie inside the bowl. Of several such objects, the first, where no view's masks say the
    /// two are not one thing. `objects` are those the sets give without such parts, and
    /// `measured` what each says by itself.
    fn keep(&self, sets: &mut Sets, objects: &[Readings], measured: &[Measured]) -> bool {
        let of: HashMap<usize, usize> = (objects.iter().enumerate())
            .map(|(object, readings)| (readings.set, object))
            .collect();
        // Per such part, the objects it was to be one with and how many of its readings lie
        // inside each.
        let mut with: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); sets.first.len()];
        for (part, other) in std::mem::take(&mut sets.asked) {
            with[part].insert(of[&sets.find(other)], 0);
        }
        let shapes: Vec<Shape> = measured.iter().map(|said| said.fit.shape()).collect();
        let mut readings = vec![0; sets.first.len()];
        for ((seen, parts), &first) in self.seen.iter().zip(self.parts).zip(self.first) {
            for (i, &part) in parts.of.iter().enumerate() {
                if part == NONE || with[first + part as usize].is_empty() {
                    continue;
                }
                let part = first + part as usize;
                readings[part] += 1;
                for (&object, within) in &mut with[part] {
                    let inside = shapes[object].distance(seen.points[i]);
                    *within += usize::from(inside <= SIGMAS * seen.sigma(i));
                }
            }
        }
        let mut kept = false;
        for (part, with) in with.iter().enumerate() {
            let inside = with
                .iter()
                .filter(|&(_, &within)| 2 * within > readings[part]);
            for (&object, _) in inside {
                // Unless a view's masks say the two are not one thing.
                sets.theirs[part] = false;
                if sets.join(part, objects[object].set) {
                    kept = true;
                    break;
                }
                sets.theirs[part] = true;
            }
        }
        kept
    }
}

/// Joins each object that is the floor of another's hollow to it.
///
/// A thing of its own hides something: somewhere along its outline it stands in front of what
/// lies behind it. A stretch of surface that hides nothing, lying inside the shape of an object
/// it meets in a crease, is the bottom of that object's hollow: a bowl's or a basket's floor,
/// which its walls rise around. What stands on such a floor hides part of it and stays apart.
///
/// `objects` are those the sets give with the known shapes' parts, and `said` what each says by
/// itself, nothing for a known shape's part. Says per part whether it is of a floor so joined.
fn hollows(
    sets: &mut Sets,
    told: &Told<'_>,
    objects: &[Readings],
    said: &[Option<Measured>],
) -> Vec<bool> {
    let of: HashMap<usize, usize> = (objects.iter().enumerate())
        .map(|(object, readings)| (readings.set, object))
        .collect();
    // Each part's set before any floor joins its walls, and per set whether it is a floor.
    let sets_before: Vec<usize> = (0..sets.first.len()).map(|part| sets.find(part)).collect();
    let mut joined = vec![false; sets_before.len()];
    // Per object, how much it hides; per two objects, how long a crease they meet in.
    let mut hides = vec![0.0; objects.len()];
    let mut meets: BTreeMap<(usize, usize), f32> = BTreeMap::new();
    for (parts, &first) in told.parts.iter().zip(told.first) {
        let mut object = |part: u32| of[&sets.find(first + part as usize)];
        for (part, hidden) in parts.hides.iter().enumerate() {
            hides[object(part as u32)] += hidden;
        }
        for (&(a, b), &length) in &parts.meets {
            let (a, b) = (object(a), object(b));
            if a != b {
                *meets.entry((a.min(b), a.max(b))).or_default() += length;
            }
        }
    }
    // A known shape's part is no hollow's walls.
    let shapes: Vec<Option<Shape>> = (said.iter())
        .map(|said| Some(said.as_ref()?.fit.shape()))
        .collect();
    for (floor, readings) in objects.iter().enumerate() {
        if hides[floor] >= 1.0 {
            continue;
        }
        // The objects it meets in a window's width of crease or more, and lies inside.
        let around = meets.iter().filter_map(|(&(a, b), &length)| {
            let other = if a == floor { b } else { a };
            let meets = (a == floor || b == floor) && length >= 1.0;
            let inside = |shape: &Shape| readings.within(shape).is_some();
            (meets && shapes[other].as_ref().is_some_and(inside)).then_some((length, other))
        });
        // The one it meets longest; of two as long, the first.
        let walls = around.max_by(|x, y| x.0.total_cmp(&y.0).then(y.1.cmp(&x.1)));
        if let Some((_, walls)) = walls
            && sets.join(readings.set, objects[walls].set)
        {
            joined[readings.set] = true;
        }
    }
    sets_before.iter().map(|&set| joined[set]).collect()
}

/// Joins each object that nothing holds up to the object that holds it, and says whether any was
/// joined: a handle to the front it stands out of.
///
/// An object rests on what its lowest reading lies on or in ([`support`]): any other object as its
/// own readings are fitted, joined to others in this look or not, but in a shape that is not solid
/// ([`Object::solid`]) only on its surface, within the reading's tolerance. Such a shape names the
/// room a tray, a bowl or a cabinet takes up, and says nothing of what it holds inside; a ledge
/// joined to a cabinet holds up what stands on it by its own shape. Otherwise it rests on what lies
/// under its middle, seen from above as its fitted shape is: a bowl whose foot overhangs the box it
/// stands on has the table under its lowest reading, and the box under its middle. It hangs where a
/// view sees through the space there, from the height of its lowest reading down to what lies under
/// its middle, again past what a shape that is not solid holds inside, or to `floor`, the lowest
/// the look reads. Of the points a pixel's width apart on that line, a view sees past those whose
/// pixel reads farther than the point by more than that reading's tolerance, [`SIGMAS`] deviations
/// of its noise; the others are hidden from it, where they lie outside its image or their pixel is
/// the robot's, has no reading or reads something nearer, but for those the object itself hides,
/// which say nothing of what holds it up: a handle that points at the camera hides most of what
/// lies under its middle. A view sees through the space where it sees past more of the points than
/// are hidden from it, and where no view does, the object stands for all the look can tell, as a
/// bowl hides all that lies under its middle. One that hangs is held up by what it touches, along a
/// border with no jump in it ([`Parts::touches`]) a window's width or more: by what reaches lower
/// than it, along the most border, where it touches any, as a ledge is by the cabinet and not by a
/// block that stands on it; else by what it touches along the most border, as a strip of a bowl's
/// outside, whose middle is over the bowl's hollow, is by the rim above it. It joins that object,
/// unless a view's masks say the two are two things ([`Sets::join`]), and the two rest by the
/// lowest and highest readings of the one that holds the other up: a sliver over free space that
/// hangs from a thing makes it hang no more than it did. Until none is joined, so that what a
/// handle joined, where it hangs in its turn, joins what holds it up.
///
/// Where no view sees through, something the look cannot see holds the object up, and it stands on
/// the object it touches along the most border of those that can, however short, for the touch says
/// only which of them it is: what holds a thing up is under it, so one whose lowest reading is
/// lower, which keeps [`support`] from finding the object under it in turn, and whose highest is
/// not above the object's lowest by more than that reading's tolerance, unlike a taller box beside
/// it. Such pairs are said too, each as a part of the object and a part of what it stands on (sets
/// of [`Sets`]); where an object touches none that can, it stands on what lies under its lowest
/// reading, solid or not: a block in a bowl stands on the bowl.
///
/// `objects` are those the sets give, `measured` what each says by itself and `labels` their
/// pixels; what the scene before has that no view saw holds things up, as in [`look`], and holds
/// what lies in it: a view that saw through it would show its place.
fn attach(
    sets: &mut Sets,
    told: &Told<'_>,
    objects: &[Readings],
    measured: &[Measured],
    labels: &[Vec<Label>],
    floor: f32,
    previous: &Scene,
) -> (bool, Vec<(usize, usize)>) {
    let of: HashMap<usize, usize> = (objects.iter().enumerate())
        .map(|(object, readings)| (readings.set, object))
        .collect();
    // Per two objects, how long a border they touch along. A known shape's part is no object's.
    let mut touches: BTreeMap<(usize, usize), f32> = BTreeMap::new();
    for (parts, &first) in told.parts.iter().zip(told.first) {
        let mut object = |part: u32| of.get(&sets.find(first + part as usize)).copied();
        for (&(a, b), &length) in &parts.touches {
            if let (Some(a), Some(b)) = (object(a), object(b))
                && a != b
            {
                *touches.entry((a.min(b), a.max(b))).or_default() += length;
            }
        }
    }
    let fits: Vec<Fit> = measured.iter().map(|said| said.fit).collect();
    let alone: Vec<(Shape, f32)> = (fits.iter())
        .map(|fit| (fit.shape(), fit.lowest.z))
        .collect();
    // The objects of the scene before that none of these continues and no view shows.
    let sizes: Vec<f32> = (stand(&fits, &alone, told.look.known, floor, &[], &|_, _| false).iter())
        .map(|(fit, _)| fit.shape().outer().volume())
        .collect();
    let was = continues(measured, &sizes, previous);
    let unseen: Vec<&Object> = (previous.objects.iter().enumerate())
        .filter(|&(before, object)| {
            !was.contains(&Some(before)) && !shown(&object.shape, None, told.seen, labels)
        })
        .map(|(_, object)| object)
        .collect();

    // Whether a view sees past more of the points straight under `from`, down to the height
    // `bottom` a pixel's width apart at `from`, than something other than the object hides from
    // it: `own` says, of an object of `objects`, whether it is the object or joined to it.
    let hangs = |from: Vec3, bottom: f32, own: &dyn Fn(usize) -> bool| {
        (told.seen.iter().zip(labels)).any(|(seen, labels)| {
            let eye = seen.eye();
            let points = ((from.z - bottom) / (seen.pixel() * eye.distance(from))).ceil();
            let points = points.max(1.0) as usize;
            let step = (from.z - bottom) / points as f32;
            let (mut past, mut hidden) = (0, 0);
            for k in 0..points {
                let point = from - Vec3::Z * step * (k as f32 + 0.5);
                let Some(j) = seen.project(point).filter(|&j| seen.reads(j)) else {
                    hidden += 1;
                    continue;
                };
                let beyond = eye.distance(seen.points[j]) - eye.distance(point);
                if beyond > SIGMAS * seen.sigma(j) {
                    past += 1;
                } else if !matches!(labels[j], Label::Object(object) if own(object)) {
                    hidden += 1;
                }
            }
            past > hidden
        })
    };

    // What lies under a thing is each object as its own readings are fitted, solid or not,
    // joined to others or not: a ledge joined to the cabinet it stands out of still holds up
    // what stands on it, and the shape fitted to the two together, which takes in the space
    // under the ledge, holds up nothing. Then what the scene before has that no view saw.
    let mut under: Vec<(Shape, f32)> = alone.clone();
    under.extend(
        unseen
            .iter()
            .map(|object| (object.shape, object.shape.lowest())),
    );
    // What no view saw holds up what lies in it, as it did: a view that saw through it would
    // show its place, and it would be gone.
    let firm: Vec<bool> = (objects.iter().zip(&alone))
        .map(|(readings, (shape, _))| readings.solid(shape))
        .chain(unseen.iter().map(|_| true))
        .collect();

    // The objects joined so far, each named by its first: per object its first, and per first
    // its objects, the lowest reading it rests by with that reading's tolerance and the height
    // of its highest, and the middle from above of its shape, which it rests on: fitted to its
    // objects' readings together where it is first needed after a join.
    let mut first: Vec<usize> = (0..objects.len()).collect();
    let mut members: Vec<Vec<usize>> = (0..objects.len()).map(|object| vec![object]).collect();
    let mut lowest: Vec<(Vec3, f32)> = (objects.iter())
        .map(|readings| {
            let points = &readings.points;
            let low = (0..points.len())
                .min_by(|&a, &b| points[a].z.total_cmp(&points[b].z))
                .expect("an object has a reading");
            (points[low], readings.slack[low])
        })
        .collect();
    let mut highest: Vec<f32> = (objects.iter())
        .map(|readings| {
            readings
                .points
                .iter()
                .map(|p| p.z)
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();
    let mut middles: Vec<Option<Vec2>> = (alone.iter())
        .map(|(shape, _)| Some(shape.middle().truncate()))
        .collect();
    let mut joined = false;
    loop {
        let heads: Vec<usize> = (0..objects.len()).filter(|&o| first[o] == o).collect();
        let (mut joins, mut stands) = (vec![], vec![]);
        for &head in &heads {
            let (from, slack) = lowest[head];
            // What lies under a point at the height of its lowest reading, of what is not the
            // object or joined to it: a shape that is not solid says nothing of what it holds
            // inside it, past that reading's tolerance.
            let below = |point: Vec3| {
                let left_out = |_: usize, body: usize| {
                    let own = body < objects.len() && first[body] == head;
                    own || (!firm[body] && under[body].0.distance(point) < -slack)
                };
                support(head, point, &under, told.look.known, floor, &left_out)
            };
            if below(from).is_some_and(|(_, drop)| drop == 0.0) {
                continue;
            }
            let middle = middles[head].get_or_insert_with(|| {
                let points: Vec<Vec3> = (members[head].iter())
                    .flat_map(|&object| objects[object].points.iter().copied())
                    .collect();
                Fit::of(&points).shape().middle().truncate()
            });
            let middle = middle.extend(from.z);
            let bottom = below(middle).map_or(floor, |(_, drop)| middle.z - drop);
            let hanging = hangs(middle, bottom, &|object| first[object] == head);
            // What holds a thing up reaches lower than it, as a cabinet does a ledge out of its
            // front, and not a block that stands on the ledge; what holds it from under it has
            // its top no higher than the thing's lowest reading, past that reading's tolerance.
            let lower = |other: usize| lowest[other].0.z < from.z;
            let holds = |other: usize| lower(other) && highest[other] <= from.z + slack;
            // The objects it touches, by their first; the one along the most border, a window's
            // width or more, and of two as long, the first.
            let mut along: BTreeMap<usize, f32> = BTreeMap::new();
            for (&(a, b), &length) in &touches {
                let (a, b) = (first[a], first[b]);
                if a != b && (a == head || b == head) {
                    *along.entry(a + b - head).or_default() += length;
                }
            }
            // Of those it touches along `least` windows' width or more that `of` lets in, the
            // one along the most border; of two as long, the first.
            let most = |least: f32, of: &dyn Fn(usize) -> bool| {
                (along.iter().map(|(&other, &length)| (other, length)))
                    .filter(|&(other, length)| length > 0.0 && length >= least && of(other))
                    .max_by(|x, y| x.1.total_cmp(&y.1).then(y.0.cmp(&x.0)))
                    .map(|(other, _)| other)
            };
            if hanging {
                // Of what reaches lower than it, where it touches any: a strip of a bowl's
                // outside, whose middle is over the bowl's hollow, touches the rim above it.
                let holder = most(1.0, &lower).or_else(|| most(1.0, &|_| true));
                joins.extend(holder.map(|other| (head, other)));
            } else if let Some(other) = most(0.0, &holds) {
                // Something the look cannot see holds it up; the touch says only which of what
                // can it is, however short.
                stands.push((objects[head].set, objects[other].set));
            }
        }
        let mut changed = false;
        for (hanging, holder) in joins {
            let (hanging, holder) = (first[hanging], first[holder]);
            let (a, b) = (hanging.min(holder), hanging.max(holder));
            if a == b || !sets.join(objects[a].set, objects[b].set) {
                continue;
            }
            // The two rest by what holds the one up: a piece over free space that hangs from a
            // thing makes it hang no more than it did.
            let rests = (lowest[holder], highest[holder]);
            for &object in &members[b] {
                first[object] = a;
            }
            let moved = std::mem::take(&mut members[b]);
            members[a].extend(moved);
            middles[a] = None;
            (lowest[a], highest[a]) = rests;
            changed = true;
        }
        if !changed {
            return (joined, stands);
        }
        joined = true;
    }
}

/// Where an object is parted from what it stands on or from what stands on it ([`part`]): the
/// height it now begins at and the one it now ends at, where these move, and its shape fitted
/// again from its readings between them.
#[derive(Clone, Copy, Default)]
struct Cut {
    bottom: Option<f32>,
    top: Option<f32>,
    fitted: Option<Shape>,
}

/// Where each object of `objects`, whose shapes are `shapes` and which stand as `stood` says, is
/// parted from what it stands on or from what stands on it: where one thing stands on another,
/// the two are not in one place, and the lower ends where the upper begins.
///
/// An object A that stands on an object B, whose shape reaches under the top of B's by more than
/// the tolerance of A's lowest reading and B's highest (three deviations of each, the root of the
/// sum of their squares), holds a few of B's readings, or B a few of A's: fewer than a window's
/// worth cannot be told from noise. So A's bottom is taken where a window's worth of its readings
/// is reached from below (a_lo), and B's top and bottom where a window's worth of its readings is
/// reached from above (b_hi) and from below (b_lo). Where b_hi lies over a_lo by more than their
/// tolerance, the two truly interleave, as a bowl in a tray or a block in a bowl, and both stay
/// as they are; so do a thing in an open box's room, every reading of which lies over the room
/// within its slack, though only its top above the walls shows; a lower whose top window is not
/// over its bottom window, of little more than a window, no thing under A; and an A with nothing
/// over b_hi. Otherwise A begins at b_hi, or at its own lowest reading where that is higher, and
/// B ends where the highest of what stands on it begins, A or another, so that nothing on it is
/// left over it or cut under its own readings: B is fitted again from its readings up to there,
/// within their slack; and where B so fitted is solid or open at the top, A is fitted again from
/// its readings from where it begins up. What lies inside a shape that is not solid says nothing
/// of whose it is: a piece of a thing in a bowl, its rim at the bowl's, keeps its readings under
/// it. A box open at the top keeps its outline and its room. A lower keeps its top where its
/// readings over where it would end, that no other object's shape holds, reach up to it: they lie
/// in no other's place. Every pair is decided against `shapes` as they come in, so that the order
/// the views name the objects in changes nothing. Then parting leaves none of an object's readings
/// out of every shape: each reaches past where it was cut as far as those of its readings lie that
/// no shape, as cut, holds; one that reaches back to where it was is not cut.
fn part(shapes: &[Shape], objects: &[Readings], stood: &[(Fit, Option<Support>)]) -> Vec<Cut> {
    // Two heights of an object's readings, a lower and a higher, each with the slack of the
    // reading there.
    type Heights = [(f32, f32); 2];
    let count = objects.len();
    // Where a window's worth of an object's readings is reached from below and from above. Once
    // an object.
    let mut windows: Vec<Option<Option<Heights>>> = vec![None; count];
    let mut window = |object: usize| {
        *windows[object].get_or_insert_with(|| {
            let readings = &objects[object];
            let heights = (readings.points.iter().zip(&readings.share)).map(|(p, s)| (p.z, *s));
            let (low, high) = fit::windows(heights)?;
            let slack = |at: f32| {
                let reached = readings.points.iter().position(|p| p.z == at);
                reached.map(|i| (at, readings.slack[i]))
            };
            Some([slack(low)?, slack(high)?])
        })
    };
    // Of each object, its lowest reading's height and slack and its highest's. Once an object.
    let mut extremes: Vec<Option<Heights>> = vec![None; count];
    let mut extreme = |object: usize| {
        *extremes[object].get_or_insert_with(|| {
            let readings = &objects[object];
            let (mut low, mut high) = ((f32::INFINITY, 0.0), (f32::NEG_INFINITY, 0.0));
            for (point, &slack) in readings.points.iter().zip(&readings.slack) {
                if point.z < low.0 {
                    low = (point.z, slack);
                }
                if point.z > high.0 {
                    high = (point.z, slack);
                }
            }
            [low, high]
        })
    };
    // Whether every one of `readings` lies over the room of `shape`, a box open at the top, seen
    // from above and within its slack.
    let in_room = |shape: &Shape, readings: &Readings| {
        let Shape::OpenBox {
            center,
            rotation,
            room_lower,
            room_upper,
            ..
        } = *shape
        else {
            return false;
        };
        (readings.points.iter().zip(&readings.slack)).all(|(point, slack)| {
            let own = (rotation.inverse() * (*point - center)).truncate();
            let (low, high) = (room_lower.truncate() - *slack, room_upper + *slack);
            own.cmpge(low).all() && own.cmple(high).all()
        })
    };
    // The pairs that part, each with the height it parts at.
    let mut pairs: Vec<(usize, usize, f32)> = vec![];
    for (upper, (_, on)) in stood.iter().enumerate().take(count) {
        let Some(Support::Object(lower)) = *on else {
            continue;
        };
        if lower >= count {
            continue;
        }
        let (sa, sb) = (shapes[upper], shapes[lower]);
        if sa.lowest() >= sb.highest() {
            continue;
        }
        let overlap = extreme(upper)[0].1.hypot(extreme(lower)[1].1);
        if sa.lowest() >= sb.highest() - overlap || in_room(&sb, &objects[upper]) {
            continue;
        }
        let (Some([a_lo, _]), Some([b_lo, b_hi])) = (window(upper), window(lower)) else {
            continue;
        };
        let interleave = b_hi.0 > a_lo.0 + a_lo.1.hypot(b_hi.1);
        if interleave || b_hi.0 <= b_lo.0 || b_hi.0 >= sa.highest() {
            continue;
        }
        pairs.push((upper, lower, b_hi.0));
    }
    // Where each thing on a lower that parts begins: one parted from it at its height or at its
    // own lowest reading, where that is higher; any other at its shape's bottom.
    let lowest = |k: usize| extremes[k].map_or(f32::INFINITY, |[low, _]| low.0);
    let begins = |upper: usize, lower: usize| {
        let pair = pairs.iter().find(|&&(a, b, _)| (a, b) == (upper, lower));
        pair.map_or_else(
            || shapes[upper].lowest(),
            |&(_, _, at)| at.max(lowest(upper)),
        )
    };
    // How high those of a lower's readings over `ends` reach that no other object's shape holds.
    let held_over = |lower: usize, ends: f32| {
        let readings = &objects[lower];
        let others = (shapes.iter().enumerate()).filter(|&(object, _)| object != lower);
        let unheld = |&(point, slack): &(&Vec3, &f32)| {
            point.z > ends + slack
                && others
                    .clone()
                    .all(|(_, shape)| shape.distance(*point) > *slack)
        };
        (readings.points.iter().zip(&readings.slack))
            .filter(unheld)
            .fold(ends, |high, (point, _)| high.max(point.z))
    };
    let (mut bottom, mut top) = (vec![None; count], vec![None::<f32>; count]);
    for &(_, lower, _) in &pairs {
        let on = |upper: &usize| stood[*upper].1 == Some(Support::Object(lower));
        let ends = ((0..count).filter(on)).fold(f32::NEG_INFINITY, |high, upper| {
            high.max(begins(upper, lower))
        });
        let highest = shapes[lower].highest();
        if ends < highest && held_over(lower, ends) < highest {
            top[lower] = Some(ends);
        }
    }
    // Each lower as it ends, fitted to its readings up to where it ends, and whether it is solid
    // then; one that keeps its top as it was.
    let mut below: Vec<Option<(Shape, bool)>> = vec![None; count];
    for &(upper, lower, at) in &pairs {
        let (b, sb) = (&objects[lower], shapes[lower]);
        let (_, solid) = *below[lower].get_or_insert_with(|| {
            let fitted = top[lower].map_or(sb, |ends| fitted(b, sb, sb.lowest(), ends));
            (
                fitted,
                matches!(fitted, Shape::OpenBox { .. }) || b.solid(&fitted),
            )
        });
        let begins = at.max(lowest(upper));
        if solid && begins != shapes[upper].lowest() {
            bottom[upper] = Some(begins);
        }
    }
    // Each cut object fitted again between its heights; then each reaches as far as those of its
    // readings past them that no shape so cut holds, and is fitted again where it does.
    let span = |k: usize| {
        let (low, high) = (shapes[k].lowest(), shapes[k].highest());
        (bottom[k].unwrap_or(low), top[k].unwrap_or(high))
    };
    let cut: Vec<Shape> = (0..count)
        .map(|k| match (bottom[k], top[k], below[k]) {
            (None, None, _) => shapes[k],
            (None, Some(_), Some((fitted, _))) => fitted,
            _ => fitted(&objects[k], shapes[k], span(k).0, span(k).1),
        })
        .collect();
    (0..count)
        .map(|k| {
            if bottom[k].is_none() && top[k].is_none() {
                return Cut::default();
            }
            let (mut low, mut high) = span(k);
            let readings = &objects[k];
            for (point, slack) in readings.points.iter().zip(&readings.slack) {
                let past = (bottom[k].is_some() && point.z < low - slack)
                    || (top[k].is_some() && point.z > high + slack);
                if past && cut.iter().all(|shape| shape.distance(*point) > *slack) {
                    (low, high) = (low.min(point.z), high.max(point.z));
                }
            }
            // Where it reaches back to where it was, nothing is cut.
            if (low, high) == (shapes[k].lowest(), shapes[k].highest()) {
                return Cut::default();
            }
            let reached = (low, high) != span(k);
            Cut {
                bottom: bottom[k].map(|_| low),
                top: top[k].map(|_| high),
                fitted: Some(if reached {
                    fitted(readings, shapes[k], low, high)
                } else {
                    cut[k]
                }),
            }
        })
        .collect()
}

/// `shapes` with `cuts` made ([`part`]), but for the shapes `kept` from the scene before.
fn cut(shapes: &[Shape], cuts: &[Cut], kept: &[Option<Shape>]) -> Vec<Shape> {
    (shapes.iter().enumerate())
        .map(|(k, &shape)| {
            let cut = cuts.get(k).copied().unwrap_or_default();
            if kept.get(k).is_some_and(Option::is_some) || (cut.bottom, cut.top) == (None, None) {
                return shape;
            }
            let bottom = cut.bottom.unwrap_or_else(|| shape.lowest());
            let top = cut.top.unwrap_or_else(|| shape.highest());
            // A box open at the top is cut, never fitted again; nor is one a look closed since.
            let outline = match (shape, cut.fitted) {
                (Shape::OpenBox { .. }, _) | (_, None | Some(Shape::OpenBox { .. })) => shape,
                (_, Some(fitted)) => fitted,
            };
            between(outline, bottom, top)
        })
        .collect()
}

/// `shape`, upright, between the heights `bottom` and `top`: its outline from above as it was,
/// and a box open at the top with its floor where it was.
fn between(shape: Shape, bottom: f32, top: f32) -> Shape {
    let (middle, half) = (0.5 * (bottom + top), 0.5 * (top - bottom));
    match shape {
        Shape::Cuboid {
            center,
            half_extents,
            rotation,
        } => Shape::Cuboid {
            center: center.truncate().extend(middle),
            half_extents: half_extents.truncate().extend(half),
            rotation,
        },
        Shape::Cylinder {
            center,
            rotation,
            radius,
            ..
        } => Shape::Cylinder {
            center: center.truncate().extend(middle),
            rotation,
            radius,
            half_height: half,
        },
        Shape::OpenBox {
            center,
            half_extents,
            rotation,
            room_lower,
            room_upper,
        } => Shape::OpenBox {
            center: center.truncate().extend(middle),
            half_extents: half_extents.truncate().extend(half),
            rotation,
            room_lower: (room_lower.truncate())
                .extend((center.z + room_lower.z - middle).clamp(-half, half)),
            room_upper,
        },
    }
}

/// `shape` fitted again from those of `readings` between the heights `bottom` and `top`, within
/// their slack, and between those heights; a box open at the top only cut to them.
fn fitted(readings: &Readings, shape: Shape, bottom: f32, top: f32) -> Shape {
    let own: Vec<Vec3> = (readings.points.iter().zip(&readings.slack))
        .filter(|&(point, slack)| point.z >= bottom - slack && point.z <= top + slack)
        .map(|(point, _)| *point)
        .collect();
    let refit = !own.is_empty() && !matches!(shape, Shape::OpenBox { .. });
    between(
        if refit { Fit::of(&own).shape() } else { shape },
        bottom,
        top,
    )
}

/// Why view `index` cannot be read, if it cannot.
fn check(index: usize, view: &View<'_>) -> Result<()> {
    let image = &view.image;
    let pixels = image.depth.len();
    ensure_input!(
        image.width > 0 && pixels.is_multiple_of(image.width),
        "view {index}: {pixels} depth values do not form rows of {}",
        image.width
    );
    let Intrinsics { fx, fy, cx, cy } = image.intrinsics;
    ensure_input!(
        fx > 0.0
            && fy > 0.0
            && fx.is_finite()
            && fy.is_finite()
            && cx.is_finite()
            && cy.is_finite(),
        "view {index}: invalid intrinsics {:?}",
        image.intrinsics
    );
    let Pose { position, rotation } = image.camera;
    ensure_input!(
        position.is_finite() && rotation.is_finite() && (rotation.length() - 1.0).abs() < 1e-3,
        "view {index}: the camera's pose must be finite with a unit rotation, got {:?}",
        image.camera
    );
    let Noise {
        constant,
        linear,
        quadratic,
    } = view.noise;
    let parts = [constant, linear, quadratic];
    ensure_input!(
        parts.iter().all(|part| part.is_finite() && *part >= 0.0)
            && parts.iter().any(|part| *part > 0.0),
        "view {index}: the camera's noise must be finite, not negative and not all zero, got {:?}",
        view.noise
    );
    let (bytes, flags, numbers) = (view.color.len(), view.robot.len(), view.masks.len());
    ensure_input!(
        bytes == 0 || bytes == 3 * pixels,
        "view {index}: {bytes} bytes of colour for {pixels} pixels: three a pixel, or none"
    );
    ensure_input!(
        flags == 0 || flags == pixels,
        "view {index}: {flags} robot flags for {pixels} pixels: one a pixel, or none"
    );
    ensure_input!(
        numbers == 0 || numbers == pixels,
        "view {index}: {numbers} mask numbers for {pixels} pixels: one a pixel, or none"
    );
    Ok(())
}

/// One object's readings from every view.
struct Readings {
    /// The set its parts are.
    set: usize,
    points: Vec<Vec3>,
    /// Per point, how far off a surface it may read and be on it (meters).
    slack: Vec<f32>,
    /// Per point, the share it is of a window at its depth ([`Parts::share`]).
    share: Vec<f32>,
    /// The sum of its readings' colours, and how many had one.
    colors: ([u64; 3], u64),
}

impl Readings {
    /// Whether `shape`, its own, stands for the thing: whether more of its readings lie on the
    /// shape's surface than off it, each as far as its slack: inside it, or outside, where the
    /// shape leaves a reading out, as one parted from it as another's. A box's and a can's lie on
    /// their shapes' faces; most of a bowl's or a tray's lie inside theirs, its inside and its
    /// floor.
    fn solid(&self, shape: &Shape) -> bool {
        let (mut on, mut off) = (0, 0);
        for (point, slack) in self.points.iter().zip(&self.slack) {
            if shape.distance(*point).abs() <= *slack {
                on += 1;
            } else {
                off += 1;
            }
        }
        on > off
    }

    /// The mean colour of the readings that had one.
    fn color(&self) -> Option<[u8; 3]> {
        let (sums, count) = self.colors;
        (count > 0).then(|| sums.map(|sum| (sum / count) as u8))
    }

    /// How many of its readings lie in `shape` or on it, where more than half of them do.
    fn within(&self, shape: &Shape) -> Option<usize> {
        let distance = shape.distances();
        let (mut within, mut outside) = (0, 0);
        for (point, slack) in self.points.iter().zip(&self.slack) {
            if distance(*point) <= *slack {
                within += 1;
            } else {
                outside += 1;
                // Once half lie outside, more than half cannot lie in it.
                if 2 * outside >= self.points.len() {
                    return None;
                }
            }
        }
        (2 * within > self.points.len()).then_some(within)
    }

    /// How much of it lies outside `shape`, in windows: fewer of its readings than fill one
    /// cannot be told from noise.
    fn outside(&self, shape: &Shape) -> f32 {
        let distance = shape.distances();
        let readings = self.points.iter().zip(&self.slack).zip(&self.share);
        readings
            .filter(|&((point, slack), _)| distance(*point) > *slack)
            .map(|(_, share)| share)
            .sum()
    }
}

/// What an object's readings say by themselves, whatever they are joined to after.
#[derive(Clone)]
struct Measured {
    fit: Fit,
    /// Per object of the scene before, how many of the readings its shape holds, where it holds
    /// more than half of them ([`continues`]).
    holds: Vec<Option<usize>>,
}

impl Measured {
    fn of(readings: &Readings, previous: &Scene) -> Self {
        Self {
            fit: Fit::of(&readings.points),
            holds: if previous.objects.is_empty() {
                vec![]
            } else {
                Self::holds(readings, previous)
            },
        }
    }

    /// Per object of `previous`, how many of `readings` its shape holds, where it holds more than
    /// half of them.
    ///
    /// Of the box around the readings and their most slack: a shape whose middle lies farther
    /// from that box than the shape reaches, that slack and what rounding can make of the
    /// distances holds none of them, each outside it. Rounding makes a distance off by fewer than
    /// thirty-two of a float's steps of the lengths it is taken from.
    fn holds(readings: &Readings, previous: &Scene) -> Vec<Option<usize>> {
        let (low, high) = (readings.points.iter()).fold(
            (Vec3::INFINITY, Vec3::NEG_INFINITY),
            |(low, high), point| (low.min(*point), high.max(*point)),
        );
        let slack = readings.slack.iter().copied().fold(0.0, f32::max);
        let apart = |shape: &Shape| {
            let (middle, Some(reach)) = (shape.middle(), shape.reach()) else {
                return false;
            };
            let lengths = middle.length() + low.abs().max(high.abs()).length() + reach;
            let rounding = 32.0 * f32::EPSILON * lengths;
            middle.distance(middle.clamp(low, high)) > reach + slack + rounding
        };
        let holds = previous.objects.iter().map(|before| {
            let shape = before.shape.outer();
            if apart(&shape) {
                None
            } else {
                readings.within(&shape)
            }
        });
        holds.collect()
    }
}

/// What each of `objects` says by itself: what an object of `before` said where it has the
/// object's set and as many readings, which it then holds all of, since sets only grow by joining;
/// else measured anew.
fn remeasure(
    objects: &[Readings],
    before: &[(&Readings, &Measured)],
    previous: &Scene,
    threads: usize,
) -> Vec<Measured> {
    threads::each(threads, objects.len(), |object| {
        let readings = &objects[object];
        let same = before.iter().find(|(before, _)| {
            before.set == readings.set && before.points.len() == readings.points.len()
        });
        same.map_or_else(
            || Measured::of(readings, previous),
            |(_, said)| (*said).clone(),
        )
    })
}

/// Which parts are one object. A view's masks say which of its parts are not: no set holds two
/// parts of one view that lie in different masks.
struct Sets {
    /// Per part, a part of its set before it, or itself for the set's first.
    first: Vec<usize>,
    /// Per set, at its first part: the masks its parts lie in, each as its view and its number.
    /// One a view at most.
    lies: Vec<Vec<(usize, u32)>>,
    /// Per part, whether it is a known shape's alone ([`Told::known`]). Such a part is in no
    /// set but its own, so that the table makes no one object of two things that each have
    /// something to do with it.
    theirs: Vec<bool>,
    /// The joins asked of such parts: the part, and a part of the object it was to be one with.
    asked: Vec<(usize, usize)>,
}

impl Sets {
    /// Every part a set of its own. `lies` has per view, per part, the mask it lies in or 0
    /// ([`masks::lies`]), the views' parts in the order they are numbered in, and `theirs`
    /// whether each is a known shape's alone.
    fn new(lies: &[Vec<u32>], theirs: Vec<bool>) -> Self {
        let mut parts: Vec<Vec<(usize, u32)>> = vec![];
        for (view, lies) in lies.iter().enumerate() {
            let masked = |&mask: &u32| (mask != 0).then_some((view, mask));
            parts.extend(lies.iter().map(|mask| masked(mask).into_iter().collect()));
        }
        Self {
            first: (0..parts.len()).collect(),
            lies: parts,
            theirs,
            asked: vec![],
        }
    }

    /// The set `part` is in, named by the first of its parts.
    fn find(&mut self, part: usize) -> usize {
        let mut at = part;
        while self.first[at] != at {
            self.first[at] = self.first[self.first[at]];
            at = self.first[at];
        }
        at
    }

    /// Makes the sets of `a` and `b` one and says so, or leaves them two where that one would
    /// hold parts of one view that lie in different masks: two things, that view's masks say.
    /// Where one of the two is a known shape's alone, they stay two and the join is kept as
    /// asked ([`Told::keep`]); where both are, nothing is made of it.
    fn join(&mut self, a: usize, b: usize) -> bool {
        match (self.theirs[a], self.theirs[b]) {
            (false, false) => {}
            (true, false) => self.asked.push((a, b)),
            (false, true) => self.asked.push((b, a)),
            (true, true) => return false,
        }
        if self.theirs[a] || self.theirs[b] {
            return false;
        }
        let (a, b) = (self.find(a), self.find(b));
        let (first, other) = (a.min(b), a.max(b));
        if first == other {
            return true;
        }
        // The masks the one lies in and the other does not: in a view where the other lies in
        // a mask too, that is two masks of one view.
        let kept = &self.lies[first];
        let new: Vec<(usize, u32)> = (self.lies[other].iter())
            .filter(|lies| !kept.contains(lies))
            .copied()
            .collect();
        if (new.iter()).any(|&(view, _)| kept.iter().any(|lies| lies.0 == view)) {
            return false;
        }
        self.first[other] = first;
        self.lies[other].clear();
        self.lies[first].extend(new);
        true
    }
}

/// Makes one set of the parts of different views that show one surface: a part is another
/// view's part where more than half of what the other view also reads of it, a patch at least,
/// reads on that part's surface. A reading reads on it within three deviations of its difference
/// from the other view's reading there: the two cameras' errors are independent, so that is the
/// root of the sum of their squares, not their sum. `first` is each view's first part among all
/// the views'. Each two views are held against each other on up to `threads` threads.
///
/// Where a view's masks say two such parts are not one ([`Sets::join`]), which of them stand
/// depends on the order they are made in: from the two that show the most surface in common
/// down, so that a part is first one with what it shares most with.
fn same(seen: &[Seen], parts: &[Parts], first: &[usize], sets: &mut Sets, threads: usize) {
    // Every two views, each held against the other.
    let pairs: Vec<(usize, usize)> = (0..seen.len())
        .flat_map(|a| {
            (0..seen.len())
                .filter(move |&b| b != a)
                .map(move |b| (a, b))
        })
        .collect();
    // The parts to make one, and how much of the first reads on the second, in patches.
    let pair = |(a, b): (usize, usize)| {
        let mut joins: Vec<(f32, usize, usize)> = vec![];
        // Per part of `a`, how much of it `b` reads too, and the parts of `b` it reads on with
        // how much of it reads on each. Both in patches.
        let mut shared = vec![0.0; parts[a].count];
        let mut on: Vec<Vec<(u32, f32)>> = vec![vec![]; parts[a].count];
        for (i, &part) in parts[a].of.iter().enumerate() {
            if part == NONE {
                continue;
            }
            let point = seen[a].points[i];
            let Some(j) = seen[b]
                .project(point)
                .filter(|&j| seen[b].inverse[j] != 0.0)
            else {
                continue;
            };
            let between = point - seen[b].points[j];
            let slack = SIGMAS * seen[a].sigma(i).hypot(seen[b].sigma(j));
            let normal = parts[b].normals[j];
            let agree = if normal == Vec3::ZERO {
                between.length() <= slack + seen[b].spacing(j)
            } else {
                between.dot(normal).abs() <= slack
            };
            if !agree {
                continue;
            }
            let share = parts[a].share[i];
            shared[part as usize] += share;
            let (other, on) = (parts[b].of[j], &mut on[part as usize]);
            if other != NONE {
                match on.iter_mut().find(|(on, _)| *on == other) {
                    Some((_, patches)) => *patches += share,
                    None => on.push((other, share)),
                }
            }
        }
        for (part, on) in on.iter().enumerate() {
            for &(other, patches) in on {
                if patches >= 1.0 && 2.0 * patches > shared[part] {
                    joins.push((patches, first[a] + part, first[b] + other as usize));
                }
            }
        }
        joins
    };
    let mut joins = threads::each(threads, pairs.len(), |k| pair(pairs[k])).concat();
    joins.sort_by(|x, y| y.0.total_cmp(&x.0).then((x.1, x.2).cmp(&(y.1, y.2))));
    for (_, part, other) in joins {
        sets.join(part, other);
    }
}

/// Widens the room of each box open at the top among `objects` (the shapes `kept` from the scene
/// before, or else their `fits`) to hold what stands in it, and closes it where something rests
/// in its solids: says which it closed, each now a box that covers the thing. `stood` is what each
/// object stands on, as the shapes before were, and `hide` how long a border each stands in front
/// of another's floor along ([`Told::hides_floors`]).
///
/// A thing whose lowest reading lies in the box, on or over the floor's top and nearer it than
/// the box's top ([`fit::stands_in`]), and that stands in front of the floor along a window's
/// width or more, stands on the floor or on what stands there: the floor runs under it, and no
/// wall stands where it does, so the room reaches over its outline ([`fit::widened`]). A wall
/// whose inner face no view shows is put where the floor is seen to stop, and what hides the floor
/// there, as a bowl does, is in the room; a piece of a wall that the look sees as an object of its
/// own hides none of the floor before it. Then a thing that rests on the box with its lowest
/// reading inside the box's solids by more than its slack stands where the box says nothing can:
/// the box does not stand for the thing, as one inflated by stray readings over what stands beside
/// it, and covers it instead.
fn hold_what_stands(
    objects: &[Readings],
    fits: &mut [Fit],
    kept: &mut [Option<Shape>],
    stood: &[(Fit, Option<Support>)],
    hide: &BTreeMap<(usize, usize), f32>,
) -> Vec<bool> {
    let mut closed = vec![false; objects.len()];
    let shape =
        |fits: &[Fit], kept: &[Option<Shape>], k: usize| kept[k].unwrap_or_else(|| fits[k].shape());
    // Each object's lowest reading and how far off a surface it may read.
    let lowest: Vec<(Vec3, f32)> = (objects.iter())
        .map(|readings| {
            let points = &readings.points;
            let low = (0..points.len())
                .min_by(|&a, &b| points[a].z.total_cmp(&points[b].z))
                .expect("an object has a reading");
            (points[low], readings.slack[low])
        })
        .collect();
    for k in 0..objects.len() {
        let open = shape(fits, kept, k);
        if !matches!(open, Shape::OpenBox { .. }) {
            continue;
        }
        let standing: Vec<Shape> = (0..objects.len())
            .filter(|&j| {
                j != k
                    && fit::stands_in(&open, lowest[j].0, lowest[j].1)
                    && hide.get(&(j, k)).is_some_and(|&border| border >= 1.0)
            })
            .map(|j| shape(fits, kept, j))
            .collect();
        match &mut kept[k] {
            Some(shape) => *shape = fit::widened(shape, &standing),
            None => fits[k].widen(&standing),
        }
        // As it stands: down to what holds it up.
        let widened = kept[k].unwrap_or_else(|| {
            let mut fit = stood[k].0;
            fit.widen(&standing);
            fit.shape()
        });
        let rests_inside = (0..objects.len()).any(|j| {
            j != k
                && stood[j].1 == Some(Support::Object(k))
                && widened.distance(lowest[j].0) < -lowest[j].1
        });
        if rests_inside {
            match &mut kept[k] {
                Some(shape) => *shape = shape.outer(),
                None => fits[k].close(),
            }
            closed[k] = true;
        }
    }
    closed
}

/// Whether object `over` of `scene` stood on its object `under`, itself or through what it
/// stood on.
fn stood_on(scene: &Scene, over: usize, under: usize) -> bool {
    let mut at = over;
    // No chain is longer than the scene has objects, whatever a caller made of the scene.
    for _ in 0..scene.objects.len() {
        match scene.objects.get(at).and_then(|object| object.on) {
            Some(Support::Object(next)) if next == under => return true,
            Some(Support::Object(next)) => at = next,
            _ => return false,
        }
    }
    false
}

/// Each fit stood on what holds it up among `bodies` and the known shapes, and what that is:
/// for an object that `held` says stands on another ([`attach`]), that one, down to its top,
/// and for the rest [`support`], which says what `rested` is. `held` may be shorter than
/// `fits`, or empty. What stands on another is higher than it, so [`support`] never finds it
/// under that other.
fn stand(
    fits: &[Fit],
    bodies: &[(Shape, f32)],
    known: &[Obstacle],
    floor: f32,
    held: &[Option<usize>],
    rested: &dyn Fn(usize, usize) -> bool,
) -> Vec<(Fit, Option<Support>)> {
    let stood = fits.iter().enumerate().map(|(object, fit)| {
        let mut fit = *fit;
        if let Some(under) = held.get(object).copied().flatten() {
            fit.stand_on(bodies[under].0.highest());
            return (fit, Some(Support::Object(under)));
        }
        let on = support(object, fit.lowest, bodies, known, floor, rested);
        if let Some((_, drop)) = on {
            fit.stand_on(fit.lowest.z - drop);
        }
        (fit, on.map(|(support, _)| support))
    });
    stood.collect()
}

/// What holds object `object` up, whose lowest reading is `from`, and how far under that
/// reading it lies (0 where the reading is inside it): the nearest straight under it, of the
/// known shapes and of the other objects whose own lowest reading is lower, since what holds a
/// thing up is under it. `bodies` has every object's shape and the height of its lowest reading,
/// or of its shape's lowest point for one no view saw. A body that `rested(object, body)` says
/// stands on the object is left out: what stands on a thing does not hold it up. Of two as
/// near, an object before a known shape, and the smaller object.
fn support(
    object: usize,
    from: Vec3,
    bodies: &[(Shape, f32)],
    known: &[Obstacle],
    floor: f32,
    rested: &dyn Fn(usize, usize) -> bool,
) -> Option<(Support, f32)> {
    let lower = |other: &usize| {
        let (own, others) = (from.z, bodies[*other].1);
        others < own || (others == own && *other < object)
    };
    let under = (0..bodies.len()).filter(lower);
    let objects = under
        .filter(|&other| !rested(object, other))
        .filter_map(|other| {
            let shape = bodies[other].0;
            let drop = fit::under(|p| shape.distance(p), from, floor)?;
            Some((drop, shape.volume(), Support::Object(other)))
        });
    let shapes = known.iter().enumerate().filter_map(|(index, shape)| {
        let drop = fit::under(|p| shape.distance(p).value, from, floor)?;
        Some((drop, f32::INFINITY, Support::Known(index)))
    });
    let nearest = objects
        .chain(shapes)
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)))?;
    Some((nearest.2, nearest.0))
}

/// Per object, the object of `previous` it continues, whose id it keeps: the one it lies where.
/// An object lies where a previous object did whose shape holds more than half of its readings:
/// of several, as a thing in a tray lies in the tray's shape and in its own, the one nearest
/// its own size, `sizes` giving each object's as a volume. A box open at the top holds and is as
/// large as its box ([`Shape::outer`]): a piece of a thing that lies in its room lies where the
/// thing does. Where several lie in one previous object, the one nearest its size continues it,
/// as the larger of the two halves of what measured as one thing does, and the others the next
/// nearest they lie where: an object and a previous object it lies where are paired nearest in
/// size first; of two pairs as near, the one with more of the object's readings in the previous
/// object's shape, then the first. So a drawer whose floor was a thing of its own, and whose
/// walls were the cabinet's, lies where the cabinet did and where the floor did, and continues
/// the floor while the cabinet, nearer the cabinet's size, continues the cabinet, however many
/// readings the drawer has in the cabinet's shape.
fn continues(measured: &[Measured], sizes: &[f32], previous: &Scene) -> Vec<Option<usize>> {
    // Each object and each previous object it lies where: how far from the object's size that
    // one's is, the logarithm of their ratio, and how many of its readings that one holds.
    let mut pairs: Vec<(f32, usize, usize, usize)> = vec![];
    for (object, (said, size)) in measured.iter().zip(sizes).enumerate() {
        for (before, (was, within)) in previous.objects.iter().zip(&said.holds).enumerate() {
            if let Some(within) = *within {
                let apart = (was.shape.outer().volume() / size).ln().abs();
                pairs.push((apart, within, object, before));
            }
        }
    }
    pairs.sort_by(|a, b| {
        (a.0.total_cmp(&b.0))
            .then(b.1.cmp(&a.1))
            .then((a.2, a.3).cmp(&(b.2, b.3)))
    });
    let mut was = vec![None; measured.len()];
    let mut continued = vec![false; previous.objects.len()];
    for (_, _, object, before) in pairs {
        if was[object].is_none() && !continued[before] {
            was[object] = Some(before);
            continued[before] = true;
        }
    }
    was
}

/// Makes one object of the pieces a thing of the scene before comes back in under the robot,
/// and says whether it made any. With the robot's arm over a thing, the thing comes back as
/// several objects: one continues it ([`continues`]) and the rest would be new, though nothing
/// moved. A piece is an object that continues none and more than half of whose readings lie in
/// the solid of an earlier object's shape, the one holding the most of them, where another
/// object continues that one: in the walls and the floor of a box open at the top, not in its
/// room, where what was put into it lies. Only a shape that stood for its thing has such a solid
/// ([`Object::solid`]): a covering box has room in it for other things.
///
/// A piece is one object with the object that continues its thing where the robot may hide
/// what joins them: in some view the two lie at one stretch of what the view does not read
/// about the robot ([`unread`]). Elsewhere the look shows the piece apart, as it does a thing
/// set down in what a shape claims beyond its thing, or one of two things taken for one that is
/// pushed away. And not where a view reads more of where the earlier shape would show as
/// something else than as the pieces and that object ([`sight`]). What a view cannot see says
/// nothing. What reads as something else is held against what reads as the thing and not, as in
/// [`shown`], against what is hidden, because a shape may claim more than its thing: the box of
/// a cabinet with a drawer pulled out claims the room under the drawer, where the table reads in
/// every look, by more than an arm hides. A view's masks that tell a piece from the rest keep it
/// apart ([`Sets::join`]).
fn rejoin(
    sets: &mut Sets,
    objects: &[Readings],
    measured: &[Measured],
    labels: &[Vec<Label>],
    seen: &[Seen],
    previous: &Scene,
) -> bool {
    if previous.objects.is_empty() {
        return false;
    }
    let sizes: Vec<f32> = (measured.iter())
        .map(|said| said.fit.shape().outer().volume())
        .collect();
    let was = continues(measured, &sizes, previous);
    // Per earlier object, the object that continues it.
    let mut now = vec![None; previous.objects.len()];
    for (object, was) in was.iter().enumerate() {
        if let Some(before) = *was {
            now[before] = Some(object);
        }
    }
    // Per object that continues one, the pieces of what it continues.
    let mut pieces: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for piece in (0..objects.len()).filter(|&object| was[object].is_none()) {
        let held = (previous.objects.iter().enumerate()).filter_map(|(before, was)| {
            Some((objects[piece].within(&was.shape)?, Reverse(before)))
        });
        // The earlier object that holds the most of it, of two as full the first: its piece
        // where that one's shape stood for it and an object continues it, and no other's.
        if let Some((_, Reverse(before))) = held.max()
            && previous.objects[before].solid
            && let Some(whole) = now[before]
        {
            pieces.entry(whole).or_default().push(piece);
        }
    }
    if pieces.is_empty() {
        return false;
    }
    let unread: Vec<_> = (seen.iter().zip(labels))
        .map(|(seen, labels)| unread(seen, labels))
        .collect();
    let mut joined = false;
    for (whole, mut pieces) in pieces {
        // Only those the robot may hide the join of.
        pieces.retain(|&piece| {
            unread.iter().any(|at| {
                let mut stretches = at.range((piece, 0)..=(piece, usize::MAX));
                stretches.any(|&(_, stretch)| at.contains(&(whole, stretch)))
            })
        });
        if pieces.is_empty() {
            continue;
        }
        let before = was[whole].expect("it continues one");
        // The views as they would show the thing were the pieces its own.
        let own = |label: &Label| match label {
            Label::Object(object) if pieces.contains(object) => Label::Object(whole),
            other => *other,
        };
        let as_one: Vec<Vec<Label>> = (labels.iter())
            .map(|view| view.iter().map(own).collect())
            .collect();
        let shape = &previous.objects[before].shape;
        let apart = seen.iter().zip(&as_one).any(|(seen, labels)| {
            let sight = sight(shape, Some(whole), seen, labels);
            sight.otherwise > sight.own
        });
        if apart {
            continue;
        }
        for piece in pieces {
            joined |= sets.join(objects[piece].set, objects[whole].set);
        }
    }
    joined
}

/// The stretches of what a view does not read about the robot, and the objects that lie at each,
/// as pairs of an object and a stretch: behind such a stretch the robot may hide what joins the
/// objects on either side of it. A stretch is the robot's pixels and those without a reading
/// that hang together with them, as the shadow the robot casts in a stereo camera's depth does;
/// an object lies at it where one of its pixels is beside one of the stretch's.
fn unread(seen: &Seen, labels: &[Label]) -> BTreeSet<(usize, usize)> {
    let beside = |i: usize| {
        let (x, y) = (i % seen.width, i / seen.width);
        let left = (x > 0).then(|| i - 1);
        let right = (x + 1 < seen.width).then(|| i + 1);
        let up = (y > 0).then(|| i - seen.width);
        let down = (y + 1 < seen.height).then(|| i + seen.width);
        [left, right, up, down].into_iter().flatten()
    };
    let dark = |i: usize| seen.kinds[i] == Kind::Robot || seen.inverse[i] == 0.0;
    // Per pixel, the stretch it is of, and per stretch, whether one of the robot's is in it.
    let mut of: Vec<Option<usize>> = vec![None; labels.len()];
    let mut robots = Vec::new();
    let mut open = Vec::new();
    for start in 0..labels.len() {
        if !dark(start) || of[start].is_some() {
            continue;
        }
        let stretch = robots.len();
        robots.push(false);
        of[start] = Some(stretch);
        open.push(start);
        while let Some(i) = open.pop() {
            robots[stretch] |= seen.kinds[i] == Kind::Robot;
            for j in beside(i) {
                if dark(j) && of[j].is_none() {
                    of[j] = Some(stretch);
                    open.push(j);
                }
            }
        }
    }
    let mut at = BTreeSet::new();
    for (i, label) in labels.iter().enumerate() {
        if let Label::Object(object) = *label {
            let stretches = beside(i).filter_map(|j| of[j]);
            at.extend(
                stretches
                    .filter(|&stretch| robots[stretch])
                    .map(|stretch| (object, stretch)),
            );
        }
    }
    at
}

/// Whether some view shows where `shape`, the shape an object had in the scene before, would
/// show now, otherwise than as object `own`, the one that continues it if one does: whether more
/// of what the view would show of the shape is read on it or past it than is hidden from the
/// view ([`sight`]). Then the object is not there as it was.
fn shown(shape: &Shape, own: Option<usize>, seen: &[Seen], labels: &[Vec<Label>]) -> bool {
    seen.iter().zip(labels).any(|(seen, labels)| {
        let sight = sight(shape, own, seen, labels);
        sight.otherwise > sight.hidden
    })
}

/// How a view reads where a shape would show, as the angles each part takes up from the camera,
/// which is by the pixel.
struct Sight {
    /// Read as the object the shape is held against.
    own: f32,
    /// Read on the shape, in it or past it, as anything else.
    otherwise: f32,
    /// What the view cannot see.
    hidden: f32,
}

/// How the view `seen` reads where `shape`, the shape an object had in the scene before, would
/// show now, held against object `own`, the one that continues it if one does.
///
/// A view shows the part of a shape's surface that faces its camera. A point of that part is
/// hidden from the view where it lies outside the image, where its pixel is the robot's or has
/// no reading, and where the pixel reads something that stands before the shape: outside it by
/// more than the reading's noise allows, and nearer the camera. Elsewhere the pixel reads on the
/// shape, in it or past it, and the view shows what is there now: `own`, which is the object,
/// seen, and says nothing of the rest; or something else.
fn sight(shape: &Shape, own: Option<usize>, seen: &Seen, labels: &[Label]) -> Sight {
    let eye = seen.eye();
    let mut sight = Sight {
        own: 0.0,
        otherwise: 0.0,
        hidden: 0.0,
    };
    // A view tells no more across a face than it has pixels across.
    let most = seen.width.max(seen.height);
    let distance = shape.distances();
    shape.faces(eye, seen.pixel(), most, |point, angle| {
        let Some(i) = seen.project(point) else {
            sight.hidden += angle;
            return;
        };
        if own.is_some_and(|own| labels[i] == Label::Object(own)) {
            sight.own += angle;
            return;
        }
        let reading = seen.points[i];
        let unread = seen.kinds[i] == Kind::Robot || seen.inverse[i] == 0.0;
        let covered = unread
            || (distance(reading) > SIGMAS * seen.sigma(i)
                && eye.distance_squared(reading) < eye.distance_squared(point));
        if covered {
            sight.hidden += angle;
        } else {
            sight.otherwise += angle;
        }
    });
    sight
}
