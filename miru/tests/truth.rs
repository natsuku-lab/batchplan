//! The truth table: scenes built from known things, each read through three cameras, and what
//! `miru::look` makes of every thing held against what the scene was built from. For the table:
//!
//! ```text
//! cargo test --release -p miru --test truth -- --nocapture
//! ```
//!
//! **The cameras** stand where a robot's do, the same two for every scene: the tests' `EYE`, 78 cm
//! from the table's middle and 40 degrees over it, as a camera on a stand over a robot's table;
//! and a nearer one, 40 cm off and 55 degrees over it, turned 30 degrees from the first, as a hand
//! carries one over what it is about to take. Both read through the tests' pinhole, so that what
//! differs between the cameras is their depth alone. A stereo camera's depth goes through
//! batchplan's filter (`DepthImage::filtered`) before it becomes a view, at the noise the camera
//! states and with every other option the library's default: the README makes that the first step
//! of a look with batchplan, for it drops most of what such a camera reads between a thing's edge
//! and what lies behind it, which `look` would take for the thing, and it is how such depth reaches
//! `look` in use. The default camera's exact depth reads nothing between two surfaces and is not
//! filtered. Each scene is read three times:
//!
//! - `exact`: both views by the default camera, the renderer's exact depth stating the tests'
//!   noise. It is what `tests/look.rs` holds the crate to: what the geometry's rules make of the
//!   truth itself.
//! - `D435`: both by an Intel RealSense D435, by its datasheet (`common::Frame::sensed`): one
//!   stereo camera's errors alone, at both distances.
//! - `D435+D405`: the far view by the D435 and the near one by a D405, as a robot carries them
//!   and as the recorded sets were taken. Its far view is the `D435` reading's, draw for draw, so
//!   what moves between the two follows from the near camera alone: the D405's noise, nearly
//!   three times the D435's through one pinhole, and how much nearer it reads.
//!
//! **A row** is one thing of a scene through one of the three: how many objects it came out as
//! (those holding more than a window's worth of its pixels in a view, the window `src/segment.rs`
//! takes a plane over at the thing's depth); then, of the object holding most of its pixels, its
//! shape's kind, its size across from above (the lesser first) and its height less the thing's
//! (mm), how far its middle lies from the thing's (mm), what it stands on, whether it is solid,
//! the share of the thing's pixels it holds and, on a second look, whether it kept its id.
//!
//! **What is asserted** is each property of each row, but the gaps below. The allowance is the
//! camera's: a face may lie off by three deviations of the noise the view states at the thing's
//! middle plus a pixel's width there, the larger of the two views', so a size by two faces' and a
//! middle by one face's along each axis. Of the share, the thing's pixels that hold it to the
//! object are those that read, lie farther than a face's allowance from every other thing and
//! have no other thing beside them: the rest the camera's own noise, shadows and jitter may give
//! to something else. A listed gap that holds fails the test as well: it is taken off the list,
//! and asserted from then on.
//!
//! **Gaps**: the rows that do not hold today, each with what the object holding most of the
//! thing shows where it does not: its sizes across, across and tall, with in brackets how much
//! larger they are than the thing's and the allowance; how far its middle is off, with the
//! allowance; how much of the thing it holds, with the least the camera lets it. On exact depth
//! the knob's dial and bar are one object, 57 mm wide where the bar is 25. Through the stereo
//! cameras most boxes are larger than their things by more than the noise reaches, and not
//! solid: in `a box`, a few of the readings the jitter puts between the box's far edge and the
//! table behind it are kept by the filter and are the box's object, and its shape reaches out to
//! them. A plate 5 mm thick on a box is one object with the box through every camera, the box's
//! shape as tall as the two, so a plate set down on a box reads gone at the next look while it
//! stands there (`tests/memory.rs`); and two boxes touching are one object through every camera,
//! so of a box set down against another one reads gone.
//!
//! | Scene | Camera | Thing | Shows |
//! |---|---|---|---|
//! | a box | D435 | box | 131 x 157 x 67 mm (+51 +37 +7; 24), 31 mm off (20), not solid |
//! | a box | D435+D405 | box | 137 x 157 x 67 mm (+57 +37 +7; 24), 32 mm off (20), not solid |
//! | a turned box | D435 | box | 105 x 145 x 66 mm (+25 +25 +6; 24), not solid |
//! | a turned box | D435+D405 | box | 105 x 152 x 66 mm (+25 +32 +6; 24) |
//! | an upright can | D435 | can | a box, not solid |
//! | an upright can | D435+D405 | can | a box, not solid |
//! | two boxes apart | D435 | box 1 | 65 x 135 x 67 mm (+5 +75 +7; 24), 31 mm off (20), not solid |
//! | two boxes apart | D435 | box 2 | 95 x 115 x 56 mm (+35 +35 +6; 24), 21 mm off (21), not solid |
//! | two boxes apart | D435+D405 | box 1 | 73 x 136 x 67 mm (+13 +76 +7; 24), 31 mm off (20), not solid |
//! | two boxes apart | D435+D405 | box 2 | 115 x 118 x 56 mm (+55 +38 +6; 24), 26 mm off (21) |
//! | a box on a box | D435 | lower | 152 x 204 x 70 mm (+12 +64 +10; 24), 26 mm off (20), not solid |
//! | a box on a box | D435 | upper | 3 objects, not solid, holds 0.63 (0.87) |
//! | a box on a box | D435+D405 | lower | 179 x 204 x 63 mm (+39 +64 +3; 24), 31 mm off (20) |
//! | a box on a box | D435+D405 | upper | 2 objects, 87 x 107 x 43 mm (+27 +27 +3; 22), not solid |
//! | a can on a box | D435 | box | 152 x 171 x 65 mm (+12 +31 +5; 24), not solid |
//! | a can on a box | D435 | can | 2 objects, a box, not solid |
//! | a can on a box | D435+D405 | box | 171 x 174 x 63 mm (+31 +34 +3; 24) |
//! | a can on a box | D435+D405 | can | a box, not solid |
//! | a bowl | D435 | bowl | 2 objects, a box, 35 mm off (21), holds 0.75 (0.88) |
//! | a bowl | D435+D405 | bowl | 2 objects, a box, 38 mm off (21) |
//! | a tray | D435 | tray | 4 objects, 203 x 225 x 20 mm (-37 -15 -40; 24), on object 2, holds 0.49 (0.87) |
//! | a tray | D435+D405 | tray | 4 objects, a box, 209 x 226 x 29 mm (-31 -14 -31; 24), on object 1, holds 0.48 (0.88) |
//! | a block in a tray | D435 | tray | 5 objects, a box, 208 x 225 x 16 mm (-32 -15 -44; 24), 26 mm off (20), not solid, holds 0.50 (0.86) |
//! | a block in a tray | D435 | block | 47 x 65 x 35 mm (+7 +25 +5; 24), not solid |
//! | a block in a tray | D435+D405 | tray | 4 objects, a box, 211 x 228 x 15 mm (-29 -12 -45; 24), 25 mm off (20), on object 2, holds 0.49 (0.87) |
//! | a block in a tray | D435+D405 | block | 2 objects, 48 x 66 x 29 mm (+8 +26 -1; 24), holds 0.68 (0.80) |
//! | a knob on a box | exact | box | not solid |
//! | a knob on a box | exact | dial | a box, 57 x 90 x 48 mm (-33 -0 +38; 7), 25 mm off (6), not solid |
//! | a knob on a box | exact | bar | 57 x 90 x 48 mm (+32 -0 +10; 6), 17 mm off (6), on box, not solid |
//! | a knob on a box | D435 | box | 220 x 278 x 72 mm (+20 +78 +12; 24), 33 mm off (20), not solid |
//! | a knob on a box | D435 | dial | a box, 220 x 278 x 72 mm (+130 +188 +62; 22), 44 mm off (19), on the table, not solid |
//! | a knob on a box | D435 | bar | 2 objects, 93 x 134 x 60 mm (+68 +44 +22; 22), 29 mm off (19), not solid |
//! | a knob on a box | D435+D405 | box | 243 x 278 x 72 mm (+43 +78 +12; 24), 37 mm off (20), not solid |
//! | a knob on a box | D435+D405 | dial | a box, 243 x 278 x 72 mm (+153 +188 +62; 22), 47 mm off (19), on the table, not solid |
//! | a knob on a box | D435+D405 | bar | 94 x 134 x 60 mm (+69 +44 +22; 22), 29 mm off (19), not solid |
//! | two boxes, one moved | D435 | box 1 | 67 x 145 x 65 mm (+7 +85 +5; 28), 34 mm off (24), not solid, keeps its id |
//! | two boxes, one moved | D435 | box 2 | 2 objects, 92 x 112 x 55 mm (+32 +32 +5; 24), 25 mm off (21), not solid, holds 0.53 (0.89) |
//! | two boxes, one moved | D435+D405 | box 1 | 75 x 145 x 65 mm (+15 +85 +5; 28), 34 mm off (24), not solid, keeps its id |
//! | two boxes, one moved | D435+D405 | box 2 | 118 x 119 x 56 mm (+58 +39 +6; 24), 27 mm off (21), not solid |
//! | a can on a box, again | D435 | box | 151 x 172 x 69 mm (+11 +32 +9; 24), not solid |
//! | a can on a box, again | D435 | can | 2 objects, a box, not solid |
//! | a can on a box, again | D435+D405 | box | 172 x 174 x 63 mm (+32 +34 +3; 24), not solid |
//! | a can on a box, again | D435+D405 | can | a box, not solid |
//! | a plate on a box | exact | box | 140 x 140 x 65 mm (+0 +0 +5; 7), not solid |
//! | a plate on a box | exact | plate | the box's object: 140 x 140 x 65 mm (+80 +80 +60; 7), 30 mm off (6), on the table, not solid |
//! | a plate on a box | D435 | box | 159 x 219 x 71 mm (+19 +79 +11; 24), 35 mm off (20), not solid |
//! | a plate on a box | D435 | plate | the box's object: 159 x 219 x 71 mm (+99 +159 +66; 23), 44 mm off (19), on the table, not solid |
//! | a plate on a box | D435+D405 | box | 160 x 219 x 71 mm (+20 +79 +11; 24), 35 mm off (20), not solid |
//! | a plate on a box | D435+D405 | plate | the box's object: 160 x 219 x 71 mm (+100 +159 +66; 23), 44 mm off (19), on the table, not solid |
//! | two boxes touching | exact | box 1 | the two boxes' object: 80 x 240 x 60 mm (+0 +120 +0; 7), 60 mm off (6) |
//! | two boxes touching | exact | box 2 | the two boxes' object: 80 x 240 x 60 mm (+0 +120 +0; 7), 60 mm off (6) |
//! | two boxes touching | D435 | box 1 | the two boxes' object: 158 x 262 x 68 mm (+78 +142 +8; 24), 71 mm off (20), not solid |
//! | two boxes touching | D435 | box 2 | the two boxes' object: 158 x 262 x 68 mm (+78 +142 +8; 24), 64 mm off (20), not solid |
//! | two boxes touching | D435+D405 | box 1 | the two boxes' object: 158 x 289 x 68 mm (+78 +169 +8; 24), 83 mm off (20), not solid |
//! | two boxes touching | D435+D405 | box 2 | the two boxes' object: 158 x 289 x 68 mm (+78 +169 +8; 24), 52 mm off (20), not solid |

#[allow(dead_code, reason = "each test file uses part of it")]
mod common;

use std::fmt::Write as _;

use batchplan::DepthFilter;
use common::{
    CAMERAS, EYE, Frame, MIDDLE, NEAR, Stereo, Thing, a_box, block, can, device, known, looking,
    sizes, standing_bowl, table, thin_tray, tray_with_block, turned,
};
use glam::{Quat, Vec3};
use miru::{Label, Look, LookOptions, Noise, Scene, Shape, Support};

/// What a row says of a thing, each asserted unless it is a listed gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Property {
    /// It is one object.
    Objects,
    /// Its object's shape is of its kind.
    Kind,
    /// Its object's sizes are its own, within two faces' allowance.
    Size,
    /// Its object's middle is its own, within a face's allowance along each axis.
    Middle,
    /// Its object stands on what it stands on.
    On,
    /// Its object is solid where it is.
    Solid,
    /// Its object holds as many of its pixels as the camera lets it.
    Share,
    /// On a second look, its object kept its id where it stayed and took a new one where it
    /// moved.
    Id,
}

/// The rows that do not hold today, a line each: scene, camera and thing, and the properties of it
/// that do not, by the names the table prints. The file's doc has what each shows.
const GAPS: &str = "
a box | D435 | box | size middle solid
a box | D435+D405 | box | size middle solid
a turned box | D435 | box | size solid
a turned box | D435+D405 | box | size
an upright can | D435 | can | kind solid
an upright can | D435+D405 | can | kind solid
two boxes apart | D435 | box 1 | size middle solid
two boxes apart | D435 | box 2 | size middle solid
two boxes apart | D435+D405 | box 1 | size middle solid
two boxes apart | D435+D405 | box 2 | size middle
a box on a box | D435 | lower | size middle solid
a box on a box | D435 | upper | objects solid share
a box on a box | D435+D405 | lower | size middle
a box on a box | D435+D405 | upper | objects size solid
a can on a box | D435 | box | size solid
a can on a box | D435 | can | objects kind solid
a can on a box | D435+D405 | box | size
a can on a box | D435+D405 | can | kind solid
a bowl | D435 | bowl | objects kind middle share
a bowl | D435+D405 | bowl | objects kind middle
a tray | D435 | tray | objects size on share
a tray | D435+D405 | tray | objects kind size on share
a block in a tray | D435 | tray | objects kind size middle solid share
a block in a tray | D435 | block | size solid
a block in a tray | D435+D405 | tray | objects kind size middle on share
a block in a tray | D435+D405 | block | objects size share
a knob on a box | exact | box | solid
a knob on a box | exact | dial | kind size middle solid
a knob on a box | exact | bar | size middle on solid
a knob on a box | D435 | box | size middle solid
a knob on a box | D435 | dial | kind size middle on solid
a knob on a box | D435 | bar | objects size middle solid
a knob on a box | D435+D405 | box | size middle solid
a knob on a box | D435+D405 | dial | kind size middle on solid
a knob on a box | D435+D405 | bar | size middle solid
two boxes, one moved | D435 | box 1 | size middle solid id
two boxes, one moved | D435 | box 2 | objects size middle solid share
two boxes, one moved | D435+D405 | box 1 | size middle solid id
two boxes, one moved | D435+D405 | box 2 | size middle solid
a can on a box, again | D435 | box | size solid
a can on a box, again | D435 | can | objects kind solid
a can on a box, again | D435+D405 | box | size solid
a can on a box, again | D435+D405 | can | kind solid
a plate on a box | exact | box | solid
a plate on a box | exact | plate | size middle on solid
a plate on a box | D435 | box | size middle solid
a plate on a box | D435 | plate | size middle on solid
a plate on a box | D435+D405 | box | size middle solid
a plate on a box | D435+D405 | plate | size middle on solid
two boxes touching | exact | box 1 | size middle
two boxes touching | exact | box 2 | size middle
two boxes touching | D435 | box 1 | size middle solid
two boxes touching | D435 | box 2 | size middle solid
two boxes touching | D435+D405 | box 1 | size middle solid
two boxes touching | D435+D405 | box 2 | size middle solid
";

/// The kind of a shape a look measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Box,
    Cylinder,
    OpenBox,
}

impl Kind {
    fn of(shape: &Shape) -> Kind {
        match shape {
            Shape::Cuboid { .. } => Kind::Box,
            Shape::Cylinder { .. } => Kind::Cylinder,
            Shape::OpenBox { .. } => Kind::OpenBox,
            _ => panic!("a shape the table does not know: {shape:?}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Box => "box",
            Kind::Cylinder => "cylinder",
            Kind::OpenBox => "open box",
        }
    }
}

/// One thing of a scene, as it was built.
struct Truth {
    name: &'static str,
    /// The renderer's things it is made of, by their index in the scene: the table is 0.
    parts: Vec<usize>,
    kind: Kind,
    middle: Vec3,
    /// Its size across from above, the lesser first, and its height (meters).
    size: [f32; 3],
    /// What it stands on: another truth of the scene, or the table.
    on: Option<usize>,
    solid: bool,
    /// On a second look, whether it stayed where it was, and so keeps its id.
    stays: bool,
}

/// A solid box of `half` sizes about `middle`, made of the renderer's things `parts`, on the
/// table and staying where it is.
fn a_box_truth(name: &'static str, parts: &[usize], middle: Vec3, half: Vec3) -> Truth {
    Truth {
        name,
        parts: parts.to_vec(),
        kind: Kind::Box,
        middle,
        size: [
            2.0 * half.x.min(half.y),
            2.0 * half.x.max(half.y),
            2.0 * half.z,
        ],
        on: None,
        solid: true,
        stays: true,
    }
}

/// A solid upright cylinder about `middle`, the renderer's thing `part`, on the table and
/// staying where it is.
fn a_cylinder_truth(
    name: &'static str,
    part: usize,
    middle: Vec3,
    radius: f32,
    half: f32,
) -> Truth {
    Truth {
        kind: Kind::Cylinder,
        size: [2.0 * radius, 2.0 * radius, 2.0 * half],
        ..a_box_truth(name, &[part], middle, Vec3::ZERO)
    }
}

impl Truth {
    /// The truth standing on its scene's truth `under` instead.
    fn on(self, under: usize) -> Truth {
        Truth {
            on: Some(under),
            ..self
        }
    }
}

/// A scene: the things on the table and the truths a look is held against, and for a scene
/// looked at twice, the things and truths of the second look, which is the one held.
struct Case {
    name: &'static str,
    things: Vec<Thing>,
    truths: Vec<Truth>,
    then: Option<(Vec<Thing>, Vec<Truth>)>,
}

impl Case {
    fn once(name: &'static str, things: Vec<Thing>, truths: Vec<Truth>) -> Case {
        Case {
            name,
            things,
            truths,
            then: None,
        }
    }
}

/// Two boxes whose middles are 16 cm apart across the table, as things 1 and 2: a 6 cm cube, its
/// middle `x` along the table, and one 6 by 8 cm and 5 cm tall.
fn two_boxes(x: f32) -> (Vec<Thing>, Vec<Truth>) {
    let (first, second) = (Vec3::new(x, -0.08, 0.03), Vec3::new(0.5, 0.08, 0.025));
    let (first_half, second_half) = (Vec3::splat(0.03), Vec3::new(0.03, 0.04, 0.025));
    (
        vec![
            block(first, first_half, [200, 30, 30]),
            block(second, second_half, [30, 160, 30]),
        ],
        vec![
            a_box_truth("box 1", &[1], first, first_half),
            a_box_truth("box 2", &[2], second, second_half),
        ],
    )
}

/// A can 6 cm across and 6 cm tall on a box 14 cm square and 6 cm tall, as things 1 and 2.
fn can_on_a_box() -> (Vec<Thing>, Vec<Truth>) {
    let (under, half) = (Vec3::new(0.5, 0.0, 0.03), Vec3::new(0.07, 0.07, 0.03));
    let over = Vec3::new(0.5, 0.0, 0.09);
    (
        vec![
            block(under, half, [200, 200, 30]),
            can(over, 0.03, 0.03, [20, 20, 20]),
        ],
        vec![
            a_box_truth("box", &[1], under, half),
            a_cylinder_truth("can", 2, over, 0.03, 0.03).on(0),
        ],
    )
}

/// The catalogue: each shape the crate tells, alone, beside, on or in another, and looked at
/// again unchanged and after a thing moved.
fn cases() -> Vec<Case> {
    let middle = Vec3::new(0.5, 0.0, 0.03);
    let mut cases = vec![];
    // A box, 8 by 12 cm and 6 cm tall.
    let half = Vec3::new(0.04, 0.06, 0.03);
    cases.push(Case::once(
        "a box",
        vec![a_box()],
        vec![a_box_truth("box", &[1], middle, half)],
    ));
    // The same box turned 30 degrees about the vertical.
    let turn = Quat::from_rotation_z(30f32.to_radians());
    cases.push(Case::once(
        "a turned box",
        vec![turned(block(middle, half, [200, 30, 30]), middle, turn)],
        vec![a_box_truth("box", &[1], middle, half)],
    ));
    // An upright can, 7 cm across and 10 cm tall.
    let at = Vec3::new(0.5, 0.0, 0.05);
    cases.push(Case::once(
        "an upright can",
        vec![can(at, 0.035, 0.05, [30, 30, 200])],
        vec![a_cylinder_truth("can", 1, at, 0.035, 0.05)],
    ));
    let (things, truths) = two_boxes(0.5);
    cases.push(Case::once("two boxes apart", things, truths));
    // A box 6 by 8 cm and 4 cm tall on one 14 cm square and 6 cm tall.
    let (under, under_half) = (middle, Vec3::new(0.07, 0.07, 0.03));
    let (over, over_half) = (Vec3::new(0.5, 0.0, 0.08), Vec3::new(0.03, 0.04, 0.02));
    cases.push(Case::once(
        "a box on a box",
        vec![
            block(under, under_half, [200, 200, 30]),
            block(over, over_half, [200, 30, 30]),
        ],
        vec![
            a_box_truth("lower", &[1], under, under_half),
            a_box_truth("upper", &[2], over, over_half).on(0),
        ],
    ));
    let (things, truths) = can_on_a_box();
    cases.push(Case::once("a can on a box", things, truths));
    // A bowl 11.6 cm across at its rim and 5 cm tall, its wall 4 mm through: it is not solid.
    let bowl = Truth {
        solid: false,
        ..a_cylinder_truth("bowl", 1, MIDDLE + 0.025 * Vec3::Z, 0.058, 0.025)
    };
    cases.push(Case::once(
        "a bowl",
        vec![standing_bowl(MIDDLE)],
        vec![bowl],
    ));
    // A tray 24 cm square and 6 cm tall, its walls and floor 1 cm through: things 1 to 5.
    let tray = || Truth {
        kind: Kind::OpenBox,
        ..a_box_truth(
            "tray",
            &[1, 2, 3, 4, 5],
            middle,
            Vec3::new(0.12, 0.12, 0.03),
        )
    };
    cases.push(Case::once("a tray", thin_tray(), vec![tray()]));
    // The tray with a block 4 cm square and 3 cm tall on its floor, in its middle.
    let block_half = Vec3::new(0.02, 0.02, 0.015);
    let in_it = a_box_truth("block", &[6], Vec3::new(0.5, 0.0, 0.025), block_half).on(0);
    cases.push(Case::once(
        "a block in a tray",
        tray_with_block(0.5),
        vec![tray(), in_it],
    ));
    // A stove's knob: a bar 9 by 2.5 cm and 3.8 cm tall on a round dial 9 cm across and 1 cm
    // tall, on a box 20 cm square and 6 cm tall.
    let (stove, stove_half) = (middle, Vec3::new(0.1, 0.1, 0.03));
    let dial = Vec3::new(0.5, 0.0, 0.065);
    let (bar, bar_half) = (Vec3::new(0.5, 0.0, 0.089), Vec3::new(0.0125, 0.045, 0.019));
    cases.push(Case::once(
        "a knob on a box",
        vec![
            block(stove, stove_half, [200, 200, 200]),
            can(dial, 0.045, 0.005, [60, 60, 60]),
            block(bar, bar_half, [20, 20, 20]),
        ],
        vec![
            a_box_truth("box", &[1], stove, stove_half),
            a_cylinder_truth("dial", 2, dial, 0.045, 0.005).on(0),
            a_box_truth("bar", &[3], bar, bar_half).on(1),
        ],
    ));
    // The two boxes again after the first moved 10 cm away from the far camera, clear of where it
    // stood.
    let (things, truths) = two_boxes(0.5);
    let (moved, mut after) = two_boxes(0.4);
    after[0].stays = false;
    cases.push(Case {
        name: "two boxes, one moved",
        things,
        truths,
        then: Some((moved, after)),
    });
    // The can on the box again, as it was.
    let (things, truths) = can_on_a_box();
    cases.push(Case {
        name: "a can on a box, again",
        things,
        truths,
        then: Some(can_on_a_box()),
    });
    // A plate 6 cm square and 5 mm thick on a box 14 cm square and 6 cm tall: a thing set down on
    // another (`Act::LetGo` in `tests/memory.rs`).
    let (under, under_half) = (middle, Vec3::new(0.07, 0.07, 0.03));
    let (plate, plate_half) = (Vec3::new(0.5, 0.0, 0.0625), Vec3::new(0.03, 0.03, 0.0025));
    cases.push(Case::once(
        "a plate on a box",
        vec![
            block(under, under_half, [200, 200, 30]),
            block(plate, plate_half, [30, 200, 30]),
        ],
        vec![
            a_box_truth("box", &[1], under, under_half),
            a_box_truth("plate", &[2], plate, plate_half).on(0),
        ],
    ));
    // Two boxes 8 by 12 cm and 6 cm tall side by side, touching: a thing set down or pushed
    // against another (`tests/memory.rs`).
    let (first, second) = (Vec3::new(0.5, -0.06, 0.03), Vec3::new(0.5, 0.06, 0.03));
    cases.push(Case::once(
        "two boxes touching",
        vec![
            block(first, half, [200, 30, 30]),
            block(second, half, [30, 30, 200]),
        ],
        vec![
            a_box_truth("box 1", &[1], first, half),
            a_box_truth("box 2", &[2], second, half),
        ],
    ));
    cases
}

/// The scene's things after the table, thing 0.
fn on_the_table(things: Vec<Thing>) -> Vec<Thing> {
    std::iter::once(table()).chain(things).collect()
}

/// The far view's frame and the near one's of `things`, as the renderer gives the truth.
fn render(things: &[Thing]) -> [Frame; 2] {
    [EYE, NEAR].map(|eye| Frame::of(things, looking(eye, MIDDLE)))
}

/// `rendered` as `camera` reads it with the draws of `seed`, through batchplan's filter at the
/// noise the camera states; as it is for the default camera.
fn read(rendered: &Frame, camera: Option<Stereo>, seed: u64) -> Frame {
    let Some(camera) = camera else {
        return rendered.clone();
    };
    let mut frame = rendered.clone().sensed(camera, seed);
    let Noise {
        constant,
        linear,
        quadratic,
    } = frame.noise;
    let filter = DepthFilter {
        noise: [constant, linear, quadratic],
        ..Default::default()
    };
    frame.depth = (frame.view().image.filtered(&filter)).expect("a sound image");
    frame
}

/// The scene `miru::look` makes of the two views, with the table known, after `previous`.
fn look(frames: &[Frame; 2], previous: &Scene) -> Scene {
    let views = frames.each_ref().map(Frame::view);
    let known = [known(&table())];
    let look = Look {
        views: &views,
        known: &known,
    };
    miru::look(device(), &look, previous, &LookOptions::default()).expect("a sound look")
}

/// The standard deviation of a reading at `depth` by the noise a view states (meters).
fn deviation(noise: Noise, depth: f32) -> f32 {
    noise.constant + (noise.linear + noise.quadratic * depth) * depth
}

/// How many readings a window holds where a view stating `noise` reads at `depth` through the
/// tests' pinhole: the square of pixels `src/segment.rs` takes a plane over there, the least
/// whose `(2r + 1) sqrt(r (r + 1))` reaches its `WINDOW`, 3 sqrt(6) over LCCP's 10 degrees, times
/// a reading's deviation over the spacing of two readings.
fn window(noise: Noise, depth: f32) -> usize {
    let need = 7.348_469 / 0.174_532_92 * deviation(noise, depth) / (depth / common::K.fx);
    let reaches = |r: &usize| (2 * r + 1) as f32 * ((r * (r + 1)) as f32).sqrt() >= need;
    let radius = (1..).find(reaches).expect("windows grow without end");
    (2 * radius + 1).pow(2)
}

/// Where pixel `i` of `frame` reads, in the look's frame.
fn point(frame: &Frame, i: usize) -> Vec3 {
    let (k, z) = (frame.intrinsics, frame.depth[i]);
    let (u, v) = ((i % frame.width) as f32, (i / frame.width) as f32);
    let ray = Vec3::new((u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z);
    frame.camera.rotation * ray + frame.camera.position
}

/// What a look made of one truth, over both views.
struct Measured {
    /// Per view, how many pixels show it.
    shown: [usize; 2],
    /// Per object, how many of those pixels it holds.
    held: Vec<usize>,
    /// How many objects hold more than a window's worth of them in a view.
    objects: usize,
    /// The object that holds the most of them, the first of those that hold as many, if any
    /// holds one.
    best: Option<usize>,
    /// How many of them the camera lets an object hold for certain: those that read, that lie
    /// farther than a face's allowance from every other thing, and that no other thing borders.
    clear: usize,
    /// How far one of the truth's faces may lie off (meters): three deviations of the noise a
    /// view states at its middle, and a pixel's width there, the larger of the views that show
    /// it.
    face: f32,
}

/// What `scene` made of `truth`, one of `things`, whose views are `read` and whose truth the
/// renderer gave in `rendered`.
fn measure(
    truth: &Truth,
    things: &[Thing],
    rendered: &[Frame; 2],
    read: &[Frame; 2],
    scene: &Scene,
) -> Measured {
    let mut measured = Measured {
        shown: [0; 2],
        held: vec![0; scene.objects.len()],
        objects: 0,
        best: None,
        clear: 0,
        face: 0.0,
    };
    let mut many = vec![false; scene.objects.len()];
    for view in 0..2 {
        let (truth_frame, frame) = (&rendered[view], &read[view]);
        let camera = truth_frame.camera;
        let depth = (camera.rotation.inverse() * (truth.middle - camera.position)).z;
        let face = 3.0 * deviation(frame.noise, depth) + depth / frame.intrinsics.fx;
        let its = |i: usize| truth_frame.shows[i].is_some_and(|k| truth.parts.contains(&k));
        let others: Vec<&Thing> = (things.iter().enumerate())
            .filter(|(k, _)| !truth.parts.contains(k))
            .map(|(_, thing)| thing)
            .collect();
        let width = frame.width as isize;
        let height = frame.depth.len() as isize / width;
        let bordered = |i: usize| {
            let (u, v) = ((i % frame.width) as isize, (i / frame.width) as isize);
            let around = (-1..=1).flat_map(|dv| (-1..=1).map(move |du| (u + du, v + dv)));
            let mut around = around.filter(|&(x, y)| (x, y) != (u, v));
            around.any(|(x, y)| {
                !(0..width).contains(&x)
                    || !(0..height).contains(&y)
                    || !its((y * width + x) as usize)
            })
        };
        let mut held = vec![0; scene.objects.len()];
        for i in (0..frame.depth.len()).filter(|&i| its(i)) {
            measured.shown[view] += 1;
            if let Label::Object(object) = scene.labels[view][i] {
                held[object] += 1;
            }
            let p = point(truth_frame, i);
            if frame.depth[i] > 0.0
                && others.iter().all(|thing| (thing.distance)(p) > face)
                && !bordered(i)
            {
                measured.clear += 1;
            }
        }
        if measured.shown[view] > 0 {
            measured.face = measured.face.max(face);
        }
        let window = window(frame.noise, depth);
        for (object, &count) in held.iter().enumerate() {
            measured.held[object] += count;
            many[object] |= count > window;
        }
    }
    measured.objects = many.iter().filter(|&&many| many).count();
    let most = measured.held.iter().copied().max().unwrap_or(0);
    measured.best = (most > 0).then(|| {
        let first = measured.held.iter().position(|&held| held == most);
        first.expect("the most is one of them")
    });
    measured
}

/// One property of one row: what it shows, and whether it holds.
struct Cell {
    property: Property,
    text: String,
    holds: bool,
}

/// What `support` names: the table, a truth whose object it is, or another object.
fn name(support: Option<Support>, truths: &[Truth], measured: &[Measured]) -> String {
    match support {
        None => "nothing".into(),
        Some(Support::Known(0)) => "table".into(),
        Some(Support::Object(object)) => {
            let truth = (truths.iter().zip(measured)).find(|(_, m)| m.best == Some(object));
            truth.map_or_else(
                || format!("object {object}"),
                |(truth, _)| truth.name.into(),
            )
        }
        Some(other) => format!("{other:?}"),
    }
}

/// The properties of truth `k`'s row in `scene`, and on a second look, of its id against the scene
/// before and what was measured there.
fn cells(
    truths: &[Truth],
    measured: &[Measured],
    k: usize,
    scene: &Scene,
    before: Option<(&Scene, &[Measured])>,
) -> Vec<Cell> {
    let (truth, m) = (&truths[k], &measured[k]);
    let cell = |property, text: String, holds| Cell {
        property,
        text,
        holds,
    };
    let mut cells = vec![cell(
        Property::Objects,
        m.objects.to_string(),
        m.objects == 1,
    )];
    let mut rest = vec![
        Property::Kind,
        Property::Size,
        Property::Middle,
        Property::On,
        Property::Solid,
        Property::Share,
    ];
    if before.is_some() {
        rest.push(Property::Id);
    }
    let Some(best) = m.best else {
        cells.extend(
            rest.into_iter()
                .map(|property| cell(property, "-".into(), false)),
        );
        return cells;
    };
    let object = &scene.objects[best];
    let kind = Kind::of(&object.shape);
    let holds = kind == truth.kind;
    let text = if holds {
        kind.name().into()
    } else {
        format!("{} ({})", kind.name(), truth.kind.name())
    };
    cells.push(cell(Property::Kind, text, holds));
    let ([narrow, wide], middle, half_height) = sizes(&object.shape);
    let size = [2.0 * narrow, 2.0 * wide, 2.0 * half_height];
    let off: Vec<f32> = (size.iter().zip(truth.size))
        .map(|(s, t)| 1000.0 * (s - t))
        .collect();
    let reach = 2000.0 * m.face;
    let text = format!(
        "{:+.0} {:+.0} {:+.0} /{reach:.0} = {:.0} {:.0} {:.0}",
        off[0],
        off[1],
        off[2],
        1000.0 * size[0],
        1000.0 * size[1],
        1000.0 * size[2]
    );
    cells.push(cell(
        Property::Size,
        text,
        off.iter().all(|o| o.abs() <= reach),
    ));
    let (apart, reach) = (
        1000.0 * middle.distance(truth.middle),
        1000.0 * 3f32.sqrt() * m.face,
    );
    let text = format!("{apart:.0} /{reach:.0}");
    cells.push(cell(Property::Middle, text, apart <= reach));
    let wanted = match truth.on {
        None => Some(Support::Known(0)),
        Some(under) => measured[under].best.map(Support::Object),
    };
    let holds = wanted.is_some() && object.on == wanted;
    let mut text = name(object.on, truths, measured);
    if !holds {
        let under = truth.on.map_or("table", |under| truths[under].name);
        write!(text, " ({under})").expect("a string takes it");
    }
    cells.push(cell(Property::On, text, holds));
    let yes = |solid: bool| if solid { "yes" } else { "no" };
    let holds = object.solid == truth.solid;
    let text = if holds {
        yes(object.solid).into()
    } else {
        format!("{} ({})", yes(object.solid), yes(truth.solid))
    };
    cells.push(cell(Property::Solid, text, holds));
    let shown = (m.shown[0] + m.shown[1]) as f32;
    let (share, least) = (m.held[best] as f32 / shown, m.clear as f32 / shown);
    cells.push(cell(
        Property::Share,
        format!("{share:.2} /{least:.2}"),
        share >= least,
    ));
    if let Some((scene_before, measured_before)) = before {
        let was = measured_before[k]
            .best
            .map(|was| scene_before.objects[was].id);
        let kept = was == Some(object.id);
        let text = if kept { "kept" } else { "new" };
        cells.push(cell(Property::Id, text.into(), kept == truth.stays));
    }
    cells
}

impl Property {
    const ALL: [Property; 8] = [
        Property::Objects,
        Property::Kind,
        Property::Size,
        Property::Middle,
        Property::On,
        Property::Solid,
        Property::Share,
        Property::Id,
    ];

    /// Its name in the table's header and in [`GAPS`], and how wide its column is.
    fn column(self) -> (&'static str, usize) {
        match self {
            Property::Objects => ("objects", 9),
            Property::Kind => ("kind", 24),
            Property::Size => ("size", 36),
            Property::Middle => ("middle", 15),
            Property::On => ("on", 19),
            Property::Solid => ("solid", 14),
            Property::Share => ("share", 16),
            Property::Id => ("id", 10),
        }
    }
}

/// The rows [`GAPS`] lists: scene, camera and thing, and the properties of each.
fn gaps() -> Vec<([&'static str; 3], Vec<Property>)> {
    let lines = GAPS.lines().filter(|line| !line.is_empty());
    lines
        .map(|line| {
            let fields: Vec<&str> = line.split(" | ").collect();
            let [scene, camera, thing, properties] = fields[..] else {
                panic!("a gap of four fields: {line}");
            };
            let property = |name: &str| {
                let named = Property::ALL.into_iter().find(|p| p.column().0 == name);
                named.unwrap_or_else(|| panic!("{name} names no property: {line}"))
            };
            let properties = properties.split_whitespace().map(property).collect();
            ([scene, camera, thing], properties)
        })
        .collect()
}

#[test]
fn what_a_look_makes_of_known_things_through_three_cameras() {
    // Per camera: rows, rows with a gap, properties asserted, properties that are gaps.
    let mut counts = [[0; 4]; 3];
    let mut wrong = vec![];
    let gaps = gaps();
    let mut met: Vec<Vec<bool>> = gaps.iter().map(|gap| vec![false; gap.1.len()]).collect();
    println!(
        "pixels: the thing's in the far view + the near one. size: across from above (the lesser \
         first), across and tall, the object's less the thing's /allowance = the object's own \
         (mm). middle: how far off /allowance (mm). share: of the thing's pixels the object \
         holds /the least the camera lets it. In brackets: the truth."
    );
    let mut header = format!(
        "{:<26}{:<11}{:<7}{:<11}",
        "scene", "camera", "thing", "pixels"
    );
    for property in Property::ALL {
        let (name, width) = property.column();
        write!(header, "{name:<width$}").expect("a string takes it");
    }
    println!("{}", header.trim_end());
    for (number, case) in cases().into_iter().enumerate() {
        let things = on_the_table(case.things);
        let rendered = render(&things);
        let then = (case.then).map(|(things, truths)| {
            let things = on_the_table(things);
            (render(&things), things, truths)
        });
        for (c, (camera, far, near)) in CAMERAS.into_iter().enumerate() {
            let seed = |look: u64, view: u64| 100 * number as u64 + 10 * look + view;
            let views = |rendered: &[Frame; 2], look: u64| {
                [
                    read(&rendered[0], far, seed(look, 0)),
                    read(&rendered[1], near, seed(look, 1)),
                ]
            };
            let first = views(&rendered, 0);
            let scene = look(&first, &Scene::default());
            let measure_all = |truths: &[Truth], things, rendered, read, scene| -> Vec<Measured> {
                let each = truths.iter();
                each.map(|truth| measure(truth, things, rendered, read, scene))
                    .collect()
            };
            let measured = measure_all(&case.truths, &things, &rendered, &first, &scene);
            let (truths, scene, measured, before) = match &then {
                None => (&case.truths, scene, measured, None),
                Some((rendered_then, things_then, truths_then)) => {
                    let second = views(rendered_then, 1);
                    let after = look(&second, &scene);
                    let measured_then =
                        measure_all(truths_then, things_then, rendered_then, &second, &after);
                    (truths_then, after, measured_then, Some((scene, measured)))
                }
            };
            for k in 0..truths.len() {
                let before = before.as_ref().map(|(scene, m)| (scene, m.as_slice()));
                let cells = cells(truths, &measured, k, &scene, before);
                let shown = measured[k].shown;
                let mut line = format!(
                    "{:<26}{camera:<11}{:<7}{:<11}",
                    case.name,
                    truths[k].name,
                    format!("{}+{}", shown[0], shown[1])
                );
                let mut gapped = false;
                let row = [case.name, camera, truths[k].name];
                let listed = gaps.iter().position(|gap| gap.0 == row);
                for cell in &cells {
                    let key = (row, cell.property.column().0);
                    let gap = listed.and_then(|listed| {
                        let at = gaps[listed].1.iter().position(|&p| p == cell.property)?;
                        met[listed][at] = true;
                        Some(at)
                    });
                    let mark = match (cell.holds, gap.is_some()) {
                        (true, false) => "",
                        (false, true) => " GAP",
                        (false, false) => " FAIL",
                        (true, true) => " CLOSED",
                    };
                    if !cell.holds && gap.is_none() || cell.holds && gap.is_some() {
                        wrong.push(format!("{key:?}{mark}: {}", cell.text));
                    }
                    counts[c][2 + usize::from(gap.is_some())] += 1;
                    gapped |= gap.is_some();
                    let text = format!("{}{mark}", cell.text);
                    write!(line, "{text:<width$}", width = cell.property.column().1)
                        .expect("a string takes it");
                }
                counts[c][0] += 1;
                counts[c][1] += usize::from(gapped);
                println!("{}", line.trim_end());
            }
        }
    }
    println!();
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let [rows, gapped, asserted, gaps] = counts[c];
        println!(
            "{camera:<11}{rows} rows, {} whole and {gapped} with a gap; {asserted} properties \
             asserted, {gaps} gaps",
            rows - gapped
        );
    }
    for (gap, met) in gaps.iter().zip(&met) {
        let unmet = (gap.1.iter().zip(met)).filter(|(_, met)| !**met);
        wrong.extend(unmet.map(|(property, _)| format!("{gap:?}: {property:?} names no row")));
    }
    assert!(
        wrong.is_empty(),
        "rows that do not hold, and gaps that do or name nothing:\n{}",
        wrong.join("\n")
    );
}
