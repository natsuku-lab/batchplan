//! Rests of what a hand holds on what was seen, through `rests_on`: on surfaces sampled from
//! places whose rests are known (a table, a table among other things, a basket, two bars, a
//! slope, a post, an edge), every rest found stands where the rule says it does and none stands
//! where it does not; then a rendered scene's rests set down through `place`.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashSet;
use std::f32::consts::{FRAC_PI_2, TAU};

use batchplan::glam::{Mat3, Quat, Vec2, Vec3};
use batchplan::rng::Rng;
use batchplan::{
    DepthImage, Device, Error, Fall, Held, Intrinsics, Obstacle, Place, Pose, Rest, RestFailure,
    RestOptions, RestProblem, Surface, TaskOptions, World, falls_into, rest_failures, rests_on,
};

/// Points 2 mm apart, as a depth camera reads a surface under a meter.
const SPACING: f32 = 0.002;

/// A box 4 by 6 by 5 cm.
const HALF: Vec3 = Vec3::new(0.02, 0.03, 0.025);

fn device() -> Device {
    Device::cpu(&common::panda().unwrap()).unwrap()
}

fn a_box() -> Obstacle {
    Obstacle::cuboid(Vec3::ZERO, HALF)
}

/// A flat patch through `middle` with normal `normal`, `half` wide along each of `across`, the
/// support's or not.
fn face(
    middle: Vec3,
    normal: Vec3,
    across: [Vec3; 2],
    half: [f32; 2],
    support: bool,
    s: &mut Surface,
) {
    let steps = |h: f32| (2.0 * h / SPACING).round() as i32;
    for i in 0..=steps(half[0]) {
        for j in 0..=steps(half[1]) {
            let (u, v) = (i as f32 * SPACING - half[0], j as f32 * SPACING - half[1]);
            s.points.push(middle + across[0] * u + across[1] * v);
            s.normals.push(normal);
            s.object.push(support);
        }
    }
}

/// A level patch with its top at `z`, from `lower` to `upper` seen from above.
fn level(lower: Vec2, upper: Vec2, z: f32, support: bool, s: &mut Surface) {
    let (middle, half) = ((lower + upper) / 2.0, (upper - lower) / 2.0);
    face(
        middle.extend(z),
        Vec3::Z,
        [Vec3::X, Vec3::Y],
        half.to_array(),
        support,
        s,
    );
}

/// A box's top and sides, as cameras around and above it see them.
fn block(center: Vec3, half: Vec3, support: bool, s: &mut Surface) {
    level(
        (center - half).truncate(),
        (center + half).truncate(),
        center.z + half.z,
        support,
        s,
    );
    for axis in 0..2 {
        for sign in [1.0, -1.0] {
            let normal = Vec3::AXES[axis] * sign;
            let along = Vec3::AXES[1 - axis];
            let half_across = [half[1 - axis], half.z];
            face(
                center + normal * half[axis],
                normal,
                [along, Vec3::Z],
                half_across,
                support,
                s,
            );
        }
    }
}

/// A table 30 cm square with its top at 0, all of it the support.
fn table() -> Surface {
    let mut s = Surface::default();
    level(Vec2::splat(-0.15), Vec2::splat(0.15), 0.0, true, &mut s);
    s
}

fn rests(object: &Obstacle, surface: &Surface, o: &RestOptions) -> Vec<Rest> {
    rests_turned(object, Quat::IDENTITY, surface, o)
}

/// `object`'s rests on `surface`, turned by `rotation` at the first.
fn rests_turned(
    object: &Obstacle,
    rotation: Quat,
    surface: &Surface,
    o: &RestOptions,
) -> Vec<Rest> {
    let problem = RestProblem {
        rotation,
        ..RestProblem::new(object.clone(), 0)
    };
    rests_on(&device(), &[problem], std::slice::from_ref(surface), o)
        .unwrap()
        .remove(0)
}

fn up(r: &Rest) -> Vec3 {
    r.pose.rotation * Vec3::Z
}

/// How far `p` is from `object` resting at `r`.
fn distance(object: &Obstacle, r: &Rest, p: Vec3) -> f32 {
    object.placed(r.pose).distance(p).value
}

#[test]
fn a_box_rests_upright_on_a_table_at_each_of_its_own_turns() {
    let o = RestOptions::default();
    let found = rests(&a_box(), &table(), &o);
    assert!(!found.is_empty());
    let lean = HALF.z * o.tipping.tan();
    for r in &found {
        // On the table, upright, its middle over the table by what it tips before it falls.
        assert!((r.pose.position.z - HALF.z).abs() < 1e-4, "{r:?}");
        assert!(up(r).z > 0.9999, "{r:?}");
        assert!(
            r.pose.position.truncate().abs().max_element() <= 0.15 - lean + 1e-4,
            "{r:?}"
        );
        assert!(r.tipping >= o.tipping && r.support.len() >= 3, "{r:?}");
        for c in &r.support {
            assert!(
                c.z.abs() < 1e-6 && distance(&a_box(), r, *c).abs() < 1e-4,
                "{r:?}"
            );
        }
    }
    // Some hang past the table's edge, their middles over it.
    let past = |r: &Rest| {
        let corners = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
        corners.into_iter().any(|(x, y)| {
            let corner = r.pose.position + r.pose.rotation * (Vec3::new(x, y, -1.0) * HALF);
            corner.truncate().abs().max_element() > 0.15 + 1e-3
        })
    };
    assert!(found.iter().any(past));
    // Each of the eight turns comes. A box's half turn sets it down as it was, so it rests at a
    // turn and at the one half way round in the same places, in the same order, turned each its
    // own way; a square box at every quarter turn so, and an upright can at every turn. Beside
    // a block that is in the way, where a box rests depends on how it is turned.
    let mut beside = table();
    block(
        Vec3::new(0.03, 0.0, 0.04),
        Vec3::new(0.03, 0.03, 0.04),
        false,
        &mut beside,
    );
    let by_turn = |object: &Obstacle| {
        let mut turns: Vec<(Vec3, Vec<Vec3>)> = vec![];
        for r in rests(object, &beside, &o) {
            let along = r.pose.rotation * Vec3::X;
            match turns.iter_mut().find(|t| t.0.dot(along) > 0.9999) {
                Some(turn) => turn.1.push(r.pose.position),
                None => turns.push((along, vec![r.pose.position])),
            }
        }
        turns
    };
    let square = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.02));
    let can = Obstacle::cylinder(Vec3::ZERO, 0.03, 0.05);
    for (object, alike) in [(a_box(), 4), (square, 2), (can, 1)] {
        let turns = by_turn(&object);
        assert_eq!(turns.len(), 8, "{object:?}");
        for k in 0..8 {
            assert_eq!(turns[k].1, turns[k % alike].1, "turn {k} of {object:?}");
        }
        if alike > 1 {
            assert_ne!(turns[1].1, turns[0].1, "{object:?}");
        }
    }
    // A can leaning 0.07 degrees is the same a turn of 45 degrees on, to the rounding turns are
    // told alike by, and not 90 degrees on: every turn has its rests still, none left to a turn
    // that is itself a copy.
    let can = Obstacle::cylinder(Vec3::ZERO, 0.03, 0.05);
    let leaning = Quat::from_rotation_x(0.07f32.to_radians());
    let mut each = [0; 8];
    for r in rests_turned(&can, leaning, &table(), &o) {
        let lean = (r.pose.rotation * Vec3::Z).truncate();
        let turn = (lean.y.atan2(lean.x) + FRAC_PI_2).rem_euclid(TAU) / (TAU / 8.0);
        each[turn.round() as usize % 8] += 1;
    }
    assert!(each.iter().all(|&n| n > 500), "{each:?}");
}

#[test]
fn on_a_measured_table_it_rests_on_its_highest_points() {
    // A table read with a millimeter of noise: the box rests on the highest points under it,
    // upright, within the noise of the top.
    let mut rng = Rng::new(3);
    let mut noisy = table();
    for p in &mut noisy.points {
        p.z += rng.range(-0.001, 0.001);
    }
    let found = rests(&a_box(), &noisy, &RestOptions::default());
    assert!(found.len() > 100);
    for r in &found {
        let under = noisy.points.iter().filter(|p| {
            let local = r.pose.rotation.inverse() * (**p - r.pose.position);
            local.x.abs() <= HALF.x && local.y.abs() <= HALF.y
        });
        let highest = under.map(|p| p.z).fold(f32::NEG_INFINITY, f32::max);
        assert!((r.pose.position.z - HALF.z - highest).abs() < 1e-5, "{r:?}");
        assert!(up(r).z > 0.9999, "{r:?}");
    }
}

#[test]
fn a_flat_bottom_on_a_top_measured_in_strips_tips_as_far_as_on_one_face() {
    // A table read in rows 4 mm apart, every other cell row empty, with two hundredths of a
    // millimeter of noise: each row is a stretch of support of its own, whose plane comes out a
    // little off its neighbors'. They are one face under the box's flat bottom still, and it
    // tips as far as on the table read whole: about its own edge, to the last points 1 mm
    // inside it, 20.8 degrees.
    let mut rng = Rng::new(7);
    let mut strips = Surface::default();
    for p in table().points {
        if ((p.y / SPACING).round() as i32) % 2 == 0 {
            strips.points.push(p + Vec3::Z * rng.range(-2e-5, 2e-5));
            strips.normals.push(Vec3::Z);
            strips.object.push(true);
        }
    }
    let tall = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.03, 0.05));
    let o = RestOptions {
        turns: 1,
        ..Default::default()
    };
    let found = rests(&tall, &strips, &o);
    let inside: Vec<&Rest> = found
        .iter()
        .filter(|r| r.pose.position.truncate().abs().max_element() < 0.1)
        .collect();
    assert!(inside.len() > 300, "{}", inside.len());
    let expected = (0.019f32 / 0.05).atan();
    for r in inside {
        assert!(
            (r.tipping - expected).abs() < 1e-3,
            "{} rad, not {expected}: {r:?}",
            r.tipping
        );
    }
}

#[test]
fn rests_keep_clear_of_what_is_not_the_support() {
    // A block 6 cm square and 8 cm tall on the table, which it may not touch.
    let mut s = table();
    let (center, half) = (Vec3::new(0.03, 0.0, 0.04), Vec3::new(0.03, 0.03, 0.04));
    let before = s.points.len();
    block(center, half, false, &mut s);
    let other: Vec<Vec3> = s.points[before..].to_vec();
    let o = RestOptions::default();
    let found = rests(&a_box(), &s, &o);
    assert!(!found.is_empty());
    for r in &found {
        assert!(
            (r.pose.position.z - HALF.z).abs() < 1e-4,
            "on the table alone: {r:?}"
        );
        let nearest = other.iter().map(|&p| distance(&a_box(), r, p));
        let nearest = nearest.fold(f32::INFINITY, f32::min);
        assert!(
            nearest >= o.clearance - 1e-4,
            "{nearest} m from the block: {r:?}"
        );
    }
    // Let near it, some come nearer than a centimeter.
    let close = RestOptions {
        clearance: 0.0,
        ..o
    };
    let near = rests(&a_box(), &s, &close);
    let nearest = |r: &Rest| {
        let each = other.iter().map(|&p| distance(&a_box(), r, p));
        each.fold(f32::INFINITY, f32::min)
    };
    assert!(
        near.iter()
            .any(|r| nearest(r) < 0.005 && nearest(r) > -1e-4)
    );
    // As the support, it is rested on.
    s.object = vec![];
    let on_it = rests(&a_box(), &s, &o);
    let top = center.z + half.z + HALF.z;
    assert!(on_it.iter().any(|r| (r.pose.position.z - top).abs() < 1e-4));
}

/// A basket 20 by 16 cm and 8 cm tall, its walls 1 cm thick, its floor's top 1 cm up: the floor
/// the support where `floor`, the walls where `walls`.
fn basket(floor: bool, walls: Option<bool>) -> Surface {
    let mut s = Surface::default();
    let (outer, inner) = (Vec2::new(0.1, 0.08), Vec2::new(0.09, 0.07));
    if floor {
        level(-inner, inner, 0.01, true, &mut s);
    }
    if let Some(support) = walls {
        let thickness = (outer - inner) / 2.0;
        for (axis, sign) in [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)] {
            let mut middle = Vec3::new(0.0, 0.0, 0.04);
            middle[axis] = sign * (inner[axis] + thickness[axis]);
            let mut half = Vec3::new(outer.x, outer.y, 0.04);
            half[axis] = thickness[axis];
            block(middle, half, support, &mut s);
        }
    }
    s
}

#[test]
fn a_basket_holds_what_is_set_on_its_floor_inside_its_walls() {
    // A can 6 cm across and 12 cm tall: across a wall's top 1 cm wide alone, its middle 5 mm in
    // from the edge and 6 cm up, it would tip at 4.8 degrees.
    let can = Obstacle::cylinder(Vec3::ZERO, 0.03, 0.06);
    let o = RestOptions::default();
    let inside = |r: &Rest| {
        r.pose
            .position
            .truncate()
            .abs()
            .cmple(Vec2::new(0.09, 0.07))
            .all()
    };
    let found = rests(&can, &basket(true, Some(false)), &o);
    assert!(!found.is_empty());
    for r in &found {
        // On the floor, a centimeter clear of the walls' insides.
        assert!((r.pose.position.z - 0.07).abs() < 1e-4, "{r:?}");
        let room = Vec2::new(0.09, 0.07) - 0.03 - o.clearance + 1e-4;
        assert!(r.pose.position.truncate().abs().cmple(room).all(), "{r:?}");
    }
    // With its walls the support too, it may stand on the rim where two walls meet it, never
    // across one wall alone.
    let found = rests(&can, &basket(true, Some(true)), &o);
    let rim = found
        .iter()
        .filter(|r| !inside(r) || r.pose.position.z > 0.08);
    let mut corners = 0;
    for r in rim {
        assert!((r.pose.position.z - 0.14).abs() < 1e-4, "{r:?}");
        let on = |axis: usize| r.support.iter().any(|c| c[axis].abs() > [0.09, 0.07][axis]);
        assert!(on(0) && on(1), "on one wall alone: {r:?}");
        corners += 1;
    }
    assert!(corners > 0);
    assert!(
        found
            .iter()
            .any(|r| (r.pose.position.z - 0.07).abs() < 1e-4)
    );
    // A floor no camera saw holds nothing: no rest inside.
    let unseen = rests(&can, &basket(false, Some(true)), &o);
    assert!(!unseen.is_empty());
    assert!(unseen.iter().all(|r| r.pose.position.z > 0.08));
    assert!(rests(&can, &basket(false, Some(false)), &o).is_empty());
}

/// Two bars 2 cm thick along y, their axes 8 cm apart at x = ±4 cm and 10 cm up, seen all round.
fn bars() -> Surface {
    bars_about(0.0)
}

/// [`bars`], the middle between them at x = `middle`.
fn bars_about(middle: f32) -> Surface {
    let mut s = Surface::default();
    let around = (TAU * 0.01 / SPACING).round() as usize;
    for side in [1.0, -1.0] {
        for k in 0..around {
            let angle = TAU * k as f32 / around as f32;
            let normal = Vec3::new(angle.cos(), 0.0, angle.sin());
            for j in 0..=(0.3 / SPACING) as usize {
                let y = j as f32 * SPACING - 0.15;
                s.points
                    .push(Vec3::new(middle + side * 0.04, y, 0.1) + normal * 0.01);
                s.normals.push(normal);
            }
        }
    }
    s
}

#[test]
fn bars_hold_what_spans_them_and_cradle_what_is_laid_between_them() {
    // A tray 12 by 6 cm and 4 cm tall, its long side across the bars: it rests across both, its
    // middle between them, and nowhere on one bar alone. A bar's points within `touching` of the
    // tray reach sqrt(2 r touching) to either side of its top: that far, a bar counts as touched.
    let band = (2.0f32 * 0.01 * 0.002).sqrt();
    let tray = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.06, 0.03, 0.02));
    let o = RestOptions {
        turns: 1,
        ..Default::default()
    };
    let found = rests(&tray, &bars(), &o);
    assert!(!found.is_empty());
    for r in &found {
        assert!(
            r.pose.position.x.abs() <= 0.06 - 0.04 + band,
            "on one bar: {r:?}"
        );
        // Level on both tops.
        assert!((r.pose.position.z - 0.13).abs() < 2e-3, "{r:?}");
        let on = |side: f32| {
            r.support
                .iter()
                .any(|c| (c.x - side * 0.04).abs() < 0.01 + 1e-3)
        };
        assert!(on(1.0) && on(-1.0), "{r:?}");
    }
    // Of the ten spots across the bars, those between them hold it; over one bar it tips.
    let across = |x: f32| found.iter().any(|r| (r.pose.position.x - x).abs() < 1e-3);
    assert!(across(-0.005) && across(0.005) && !across(0.035) && !across(-0.045));
    // A can 6.6 cm across laid down at each of its turns comes to rest between the bars,
    // touching both, its axis 4.3 cm from each bar's and along them: set down half a centimeter
    // to either side of the middle it lies on one bar, and rolls in. The spots to both sides
    // come to the one place, which comes once a turn; across the bars it would roll along them.
    let can = Obstacle::cylinder(Vec3::ZERO, 0.033, 0.08);
    let lying = Quat::from_rotation_x(FRAC_PI_2);
    let z = 0.1 + (0.043f32 * 0.043 - 0.04 * 0.04).sqrt();
    // Wherever the middle lies among the spots, and with spots as near together as `touching`,
    // where the one over the middle rests without rolling and its neighbors roll to it.
    for (middle, spacing) in [
        (0.0f32, 0.01f32),
        (0.0005, 0.01),
        (0.0035, 0.01),
        (0.001, 0.002),
    ] {
        let o = RestOptions {
            spacing,
            ..Default::default()
        };
        let cradled = rests_turned(&can, lying, &bars_about(middle), &o);
        assert!(cradled.len() >= 10, "{middle}: {}", cradled.len());
        for (k, r) in cradled.iter().enumerate() {
            // Touching both bars: the lowest place between the points seen of them is up to
            // 0.4 mm lower than between the bars themselves, which the points lie on 2 mm apart.
            for side in [1.0, -1.0] {
                let bar = Vec3::new(middle + side * 0.04, r.pose.position.y, 0.1);
                let gap = bar.distance(r.pose.position) - 0.043;
                assert!((-5e-4..5e-4).contains(&gap), "{gap} m from a bar: {r:?}");
            }
            // Its height follows from the two: half a millimeter nearer a bar's axis is 1.4 mm
            // lower between them.
            assert!(
                (r.pose.position.x - middle).abs() < 5e-4 && (r.pose.position.z - z).abs() < 1.4e-3,
                "{middle}: {r:?}"
            );
            assert!((r.pose.rotation * Vec3::Z).y.abs() > 0.9999, "{r:?}");
            let same_way = |other: &&Rest| other.pose.rotation == r.pose.rotation;
            for other in cradled[..k].iter().filter(same_way) {
                let apart = other.pose.position.distance(r.pose.position);
                assert!(
                    apart > 0.75 * spacing,
                    "{apart} m apart: {r:?} and {other:?}"
                );
            }
        }
        // Each place comes both ways round along the bars.
        let ways = |along: f32| {
            let that_way = |r: &&Rest| (r.pose.rotation * Vec3::Z).y * along > 0.0;
            cradled.iter().filter(that_way).count()
        };
        assert!(ways(1.0) > 0 && ways(1.0) == ways(-1.0));
    }
    // Stood on its end and leaning a few degrees, it is not rolled: on a table read with a
    // millimeter of noise, whatever rests it has are on the spots. Leaning 3 degrees, it tips
    // where it overhangs the rim, at spots all around which it tips too and where the noise makes
    // the table push two ways: the spots set closer about them, a cell apart, find rests within a
    // spacing of there too.
    let mut rng = Rng::new(11);
    let mut noisy = table();
    for p in &mut noisy.points {
        p.z += rng.range(-0.001, 0.001);
    }
    for degrees in [1.0f32, 3.0] {
        let leaning = Quat::from_rotation_x(degrees.to_radians());
        for r in rests_turned(&can, leaning, &noisy, &RestOptions::default()) {
            let at = r.pose.position.truncate();
            let from_a_spot = (at / 0.01 - 0.5).round() + 0.5;
            let off = (at - from_a_spot * 0.01).length();
            let cells = at / 0.002;
            let closer = degrees > 1.0
                && at.abs().max_element() + 0.033 + 0.01 > 0.15
                && (cells - cells.round()).abs().max_element() < 1e-3;
            assert!(off < 1e-5 || closer, "{off} m off its spot: {r:?}");
        }
    }
    // On a table it rolls, and has no rest.
    assert!(rests_turned(&can, lying, &table(), &RestOptions::default()).is_empty());
    let can = Obstacle::cylinder(Vec3::ZERO, 0.035, 0.08);
    // Standing on its end it does.
    assert!(!rests(&can, &table(), &o).is_empty());
}

/// A point of the support for every 2 mm of a circle of `radius` about the origin at height 0,
/// from `start` round by `sweep`.
fn arc(radius: f32, start: f32, sweep: f32, s: &mut Surface) {
    let steps = (radius * sweep / SPACING).ceil() as usize;
    for k in 0..=steps {
        let angle = start + sweep * k as f32 / steps as f32;
        s.points
            .push(Vec3::new(angle.cos(), angle.sin(), 0.0) * radius);
        s.normals.push(Vec3::Z);
        s.object.push(true);
    }
}

#[test]
fn a_ball_rests_on_a_rim_or_three_posts_under_it_and_rolls_off_a_table() {
    // A ball 6 cm across over a rim 4 cm across, as a cup's: the rim holds it from every side,
    // where a spot is near enough the rim's middle for the far side to lie within `touching`
    // under it. On three short stretches of that rim, as posts, it rests too.
    let ball = Obstacle::sphere(Vec3::ZERO, 0.03);
    let fine = RestOptions {
        spacing: 0.001,
        ..Default::default()
    };
    let mut rim = Surface::default();
    arc(0.02, 0.0, TAU, &mut rim);
    let mut posts = Surface::default();
    for k in 0..3 {
        arc(0.02, TAU * k as f32 / 3.0, 0.2, &mut posts);
    }
    for (what, s) in [("a rim", &rim), ("three posts", &posts)] {
        let found = rests(&ball, s, &fine);
        assert!(!found.is_empty(), "none on {what}");
        for r in &found {
            // Down on the nearest of it, and within `touching` of all it is said to touch.
            let from = |p: &Vec3| p.distance(r.pose.position) - 0.03;
            let nearest = s.points.iter().map(from).fold(f32::INFINITY, f32::min);
            assert!(nearest.abs() < 1e-4, "{nearest} m off {what}: {r:?}");
            for c in &r.support {
                assert!(from(c) < fine.touching, "{} m off {what}: {r:?}", from(c));
            }
            // Held from every side: what it touches reaches round its middle.
            let around = |c: &Vec3| c.y.atan2(c.x);
            let mut angles: Vec<f32> = r.support.iter().map(around).collect();
            angles.sort_by(f32::total_cmp);
            let gaps = (0..angles.len()).map(|k| {
                let next = angles[(k + 1) % angles.len()];
                (next - angles[k]).rem_euclid(TAU)
            });
            let widest = gaps.fold(0.0, f32::max);
            assert!(widest < std::f32::consts::PI, "open by {widest} rad: {r:?}");
        }
    }
    // On half the rim it rolls off the open side, and on a table, level or read with half a
    // millimeter of noise, it rolls any way.
    let mut half = Surface::default();
    arc(0.02, 0.0, 0.9 * std::f32::consts::PI, &mut half);
    assert!(rests(&ball, &half, &fine).is_empty());
    let o = RestOptions::default();
    assert!(rests(&ball, &table(), &o).is_empty());
    let mut rng = Rng::new(11);
    let mut noisy = table();
    for p in &mut noisy.points {
        p.z += rng.range(-0.0005, 0.0005);
    }
    assert!(rests(&ball, &noisy, &o).is_empty());
    // So does a can laid on its side there: the points at the fringe of what lies near under
    // it, apart from the rest in the noise, are the one table still.
    let can = Obstacle::cylinder(Vec3::ZERO, 0.033, 0.04);
    let lying = Quat::from_rotation_x(FRAC_PI_2);
    assert!(rests_turned(&can, lying, &noisy, &o).is_empty());
}

#[test]
fn near_where_the_caller_would_have_it_the_spots_are_a_cell_apart() {
    // A plate 14 cm across, its floor flat out to 5 cm and its rim rising 7 mm from there to
    // its edge, and a bowl given as a flat cylinder 11 cm across, wider than the floor: it
    // stands on the rim, evenly all round only within 3 mm of the middle, where the far side
    // of the ring lies within `touching` under it. The plate's middle is as far from the spots
    // as a place can be, 7 mm from the four around it: they miss it.
    let mut plate = Surface::default();
    for i in -35..=35 {
        for j in -35..=35 {
            let at = Vec2::new(i as f32, j as f32) * SPACING;
            let out = at.length();
            if out <= 0.07 {
                let rise = (out - 0.05).max(0.0) * 0.35;
                plate.points.push(at.extend(rise));
                plate.normals.push(Vec3::Z);
                plate.object.push(true);
            }
        }
    }
    let bowl = Obstacle::cylinder(Vec3::ZERO, 0.055, 0.025);
    let o = RestOptions::default();
    let problem = RestProblem::new(bowl, 0);
    let rests = |p: &RestProblem, s: &Surface| {
        rests_on(
            &device(),
            std::slice::from_ref(p),
            std::slice::from_ref(s),
            &o,
        )
        .unwrap()
        .remove(0)
    };
    let from_the_middle = |r: &Rest| r.pose.position.truncate().length();
    let plain = rests(&problem, &plate);
    assert!(plain.iter().all(|r| from_the_middle(r) > 0.01));
    // Told to look near the middle, it finds the bowl there, at every turn, standing on the
    // ring all round; and every rest it had without.
    let near = RestProblem {
        near: Some(Vec3::new(0.0, 0.0, 0.3)),
        ..problem.clone()
    };
    let found = rests(&near, &plate);
    let middle: Vec<&Rest> = found
        .iter()
        .filter(|r| from_the_middle(r) < 0.003)
        .collect();
    let mut turns: Vec<Quat> = middle.iter().map(|r| r.pose.rotation).collect();
    turns.dedup();
    assert_eq!(turns.len(), 8, "{} at {turns:?}", middle.len());
    for r in &middle {
        // As steady as the friction lets anything be, 26.6 degrees: on the ring all round.
        assert!(r.tipping > 26f32.to_radians(), "{r:?}");
        let around = |c: &Vec3| c.y.atan2(c.x);
        let mut angles: Vec<f32> = r.support.iter().map(around).collect();
        angles.sort_by(f32::total_cmp);
        let gaps = (0..angles.len()).map(|k| {
            let next = angles[(k + 1) % angles.len()];
            (next - angles[k]).rem_euclid(TAU)
        });
        assert!(gaps.fold(0.0, f32::max) < FRAC_PI_2, "{r:?}");
    }
    for r in &plain {
        assert!(found.contains(r), "lost: {r:?}");
    }
    // Near a point off the plate, as far off as may be, nothing more is tried.
    for far in [0.5f32, 1e7, f32::MAX] {
        let off = RestProblem {
            near: Some(Vec3::new(far, 0.5, 0.0)),
            ..problem.clone()
        };
        assert_eq!(rests(&off, &plate), plain, "{far}");
    }
    // On a table, where a box rests at every spot, it rests at every closer spot too, at each
    // turn: those a cell apart out to the spacing from the point, each a place of its own but
    // one that lies within a millimeter of a grid's spot, which is that spot's place.
    for point in [
        Vec2::ZERO,
        Vec2::new(0.005, 0.005),
        Vec2::new(-0.0412, 0.0273),
    ] {
        let about = RestProblem {
            near: Some(point.extend(0.0)),
            ..RestProblem::new(a_box(), 0)
        };
        let closer = rests(&about, &table());
        let at = |spot: Vec2| {
            let there = |r: &&Rest| r.pose.position.truncate().distance(spot) < 1e-5;
            closer.iter().filter(there).count()
        };
        let (mut own, mut farthest) = (0, 0.0f32);
        for i in -5..=5 {
            for j in -5..=5 {
                let aside = Vec2::new(i as f32, j as f32) * o.touching;
                let spot = point + aside;
                let grid = ((spot / o.spacing).floor() + 0.5) * o.spacing;
                // Just at the spacing, either way.
                if (aside.length() - o.spacing).abs() < 1e-5 {
                    continue;
                }
                if aside.length() > o.spacing {
                    let rests = if spot.distance(grid) < 1e-5 { 8 } else { 0 };
                    assert_eq!(at(spot), rests, "{spot} is past the spacing from {point}");
                } else if spot.distance(grid) > 0.0011 {
                    assert_eq!(at(spot), 8, "{spot} about {point}");
                    own += 1;
                    farthest = farthest.max(aside.length());
                } else if spot.distance(grid) < 0.0009 {
                    // Once a turn, at the grid's spot and not beside it.
                    let beside = if spot.distance(grid) < 1e-5 { 8 } else { 0 };
                    assert_eq!(
                        at(spot),
                        beside,
                        "{spot} is the place of the spot at {grid}"
                    );
                    assert_eq!(at(grid), 8, "{grid}");
                }
            }
        }
        assert!(
            own > 60 && farthest > 0.0085,
            "{own} to {farthest} m about {point}"
        );
    }
    // More closer spots than a call takes, and more cells than a surface is read in, are
    // input errors that say which.
    let mut wide = Surface::default();
    level(Vec2::ZERO, Vec2::splat(0.002), 0.0, true, &mut wide);
    level(Vec2::splat(3.0), Vec2::splat(3.002), 0.0, true, &mut wide);
    let spaced = RestOptions {
        spacing: 2.0,
        ..Default::default()
    };
    let about = RestProblem {
        near: Some(Vec3::splat(1.5)),
        ..RestProblem::new(a_box(), 0)
    };
    let refused = |p: &RestProblem, s: &Surface, o: &RestOptions| match rests_on(
        &device(),
        std::slice::from_ref(p),
        std::slice::from_ref(s),
        o,
    ) {
        Err(Error::Input(why)) => why,
        other => panic!("not refused: {:?}", other.map(|r| r[0].len())),
    };
    assert!(refused(&about, &wide, &spaced).starts_with("problem 0: "));
    level(Vec2::splat(5.0), Vec2::splat(5.002), 0.0, true, &mut wide);
    let plain = RestProblem::new(a_box(), 0);
    assert!(refused(&plain, &wide, &spaced).contains("wider cells"));
}

#[test]
fn an_object_hangs_past_an_edge_only_as_far_as_it_stands() {
    // Tipping 5 degrees about the edge before falling, the box's middle stays 2.2 mm in from it;
    // asked to tip 20 degrees, 9.1 mm. The table's side is seen too, and nothing that rests is
    // tilted over the edge.
    let mut edged = table();
    let side = [Vec3::Y, Vec3::Z];
    face(
        Vec3::new(0.15, 0.0, -0.03),
        Vec3::X,
        side,
        [0.15, 0.03],
        true,
        &mut edged,
    );
    for degrees in [5.0f32, 20.0] {
        let o = RestOptions {
            tipping: degrees.to_radians(),
            turns: 1,
            spacing: 0.002,
            ..Default::default()
        };
        let found = rests(&a_box(), &edged, &o);
        assert!(found.iter().all(|r| up(r).z > 0.9999));
        let inset = HALF.z * o.tipping.tan();
        let farthest = found
            .iter()
            .map(|r| r.pose.position.x)
            .fold(f32::MIN, f32::max);
        assert!(farthest <= 0.15 - inset + 1e-4, "{degrees}: {farthest}");
        assert!(
            farthest > 0.15 - inset - o.spacing - 1e-4,
            "{degrees}: {farthest}"
        );
        // Over the edge, how far it tips is how far in its middle is, where that is short of
        // where the friction gives out, 26 degrees and more: there it would slide.
        let over = found.iter().filter(|r| {
            let past = 0.15 - r.pose.position.x;
            past.atan2(HALF.z) < 26f32.to_radians() && r.pose.position.y.abs() < 0.1
        });
        let mut checked = 0;
        for r in over {
            let past = 0.15 - r.pose.position.x;
            assert!((r.tipping - past.atan2(HALF.z)).abs() < 1e-3, "{r:?}");
            assert!(r.tipping <= 0.5f32.atan() + 1e-4, "{r:?}");
            checked += 1;
        }
        assert!(checked > 0);
    }
}

#[test]
fn a_post_too_narrow_for_what_stands_on_it_holds_nothing() {
    // A box 4 cm square and 20 cm tall on a post: on one 1.5 cm across its middle is 7.5 mm in,
    // and it tips over at 4.3 degrees; on one 3 cm across, at 8.5.
    let tall = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.02, 0.1));
    let post = |half: f32| {
        let mut s = Surface::default();
        block(
            Vec3::new(0.0, 0.0, -0.05),
            Vec3::new(half, half, 0.05),
            true,
            &mut s,
        );
        s
    };
    let o = RestOptions::default();
    assert!(rests(&tall, &post(0.0075), &o).is_empty());
    let found = rests(&tall, &post(0.015), &o);
    assert!(!found.is_empty());
    for r in &found {
        let inset = 0.015 - r.pose.position.truncate().abs().max_element();
        assert!((r.tipping - inset.atan2(0.1)).abs() < 1e-3, "{r:?}");
    }
}

/// A slope rising `degrees` toward -y, 20 cm square, through the origin; with a lip along its foot
/// at y = 0.1, 1 cm high, where `lip`.
fn slope(degrees: f32, lip: bool) -> Surface {
    let mut s = Surface::default();
    let tilt = degrees.to_radians();
    let normal = Vec3::new(0.0, tilt.sin(), tilt.cos());
    let down = Vec3::new(0.0, tilt.cos(), -tilt.sin());
    face(
        Vec3::ZERO,
        normal,
        [Vec3::X, down],
        [0.1, 0.1],
        true,
        &mut s,
    );
    if lip {
        let foot = down * 0.1;
        face(
            foot + Vec3::Z * 0.005,
            -Vec3::Y,
            [Vec3::X, Vec3::Z],
            [0.1, 0.005],
            true,
            &mut s,
        );
        level(
            Vec2::new(-0.1, foot.y),
            Vec2::new(0.1, foot.y + 0.01),
            foot.z + 0.01,
            true,
            &mut s,
        );
    }
    s
}

/// The turn that lays an object's underside flat on a [`slope`] of `degrees`.
fn lying_on(degrees: f32) -> Quat {
    Quat::from_rotation_x(-degrees.to_radians())
}

#[test]
fn a_slope_holds_what_lies_on_it_by_friction_or_against_a_stop() {
    // A tile 6 cm square and 2 cm thick, lying flat on the slope.
    let tile = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.01));
    let o = RestOptions {
        turns: 1,
        ..Default::default()
    };
    let normal = |degrees: f32| {
        let tilt = degrees.to_radians();
        Vec3::new(0.0, tilt.sin(), tilt.cos())
    };
    // At 20 degrees friction of 0.5 holds it anywhere on the slope, flat on it.
    let gentle = rests_turned(&tile, lying_on(20.0), &slope(20.0, false), &o);
    assert!(gentle.len() > 50);
    for r in &gentle {
        assert!(up(r).dot(normal(20.0)) > 0.999, "{r:?}");
        let off = r.pose.position.dot(normal(20.0)) - 0.01;
        assert!(off.abs() < 5e-4, "{off} off the slope: {r:?}");
    }
    // Set down level it touches the slope along an edge, and has no rest.
    assert!(rests(&tile, &slope(20.0, false), &o).is_empty());
    // At 35 it slides off: no rest, but against a lip at its foot, where a spot comes within
    // `touching` of it.
    let steep = rests_turned(&tile, lying_on(35.0), &slope(35.0, false), &o);
    assert!(steep.is_empty(), "{:?}", &steep[..steep.len().min(3)]);
    let fine = RestOptions {
        spacing: 0.002,
        ..o
    };
    let held = rests_turned(&tile, lying_on(35.0), &slope(35.0, true), &fine);
    assert!(!held.is_empty());
    let tilt = 35f32.to_radians();
    let lip = Vec3::new(0.0, 0.1 * tilt.cos(), 0.01 - 0.1 * tilt.sin());
    for r in &held {
        // Flat on the slope, against the lip's top edge.
        // Within `touching` of the slope, where its end leans on the lip's edge.
        assert!(
            (r.pose.position.dot(normal(35.0)) - 0.01).abs() < o.touching,
            "{r:?}"
        );
        let edge = lip.with_x(r.pose.position.x);
        let gap = tile.placed(r.pose).distance(edge).value;
        assert!(gap.abs() < 2.5e-3, "{gap} m from the lip: {r:?}");
    }
    // Off the slope's side its edge falls away to a shelf 8 cm lower, as a rack's beside a
    // cabinet: the side neither holds the tile up nor keeps it from its rests on the slope. (On
    // the shelf, it may stand on its edge against the side face.)
    let mut edged = slope(35.0, true);
    // The slope's side, from its edge down to the shelf.
    let rows = (0.1 * 35f32.to_radians().cos() / SPACING) as i32;
    for j in -rows..=rows {
        let y = j as f32 * SPACING;
        let top = -y * 35f32.to_radians().tan();
        for k in 0..=((top + 0.12) / SPACING) as i32 {
            edged
                .points
                .push(Vec3::new(0.1, y, -0.12 + k as f32 * SPACING));
            edged.normals.push(Vec3::X);
            edged.object.push(true);
        }
    }
    level(
        Vec2::new(0.1, -0.09),
        Vec2::new(0.25, 0.09),
        -0.12,
        true,
        &mut edged,
    );
    let beside = rests_turned(&tile, lying_on(35.0), &edged, &fine);
    let on_the_slope = beside
        .iter()
        .filter(|r| r.support.iter().all(|c| c.z > -0.1));
    assert!(on_the_slope.clone().count() >= held.len() / 2);
    for r in on_the_slope {
        let edge = lip.with_x(r.pose.position.x);
        let gap = tile.placed(r.pose).distance(edge).value;
        assert!(gap.abs() < 2.5e-3, "{gap} m from the lip: {r:?}");
    }
    // With friction enough to hold it there tipped 5 degrees more, 45 degrees, it stays where it is
    // set on the steep slope too.
    let rough = RestOptions { friction: 1.0, ..o };
    assert!(rests_turned(&tile, lying_on(35.0), &slope(35.0, false), &rough).len() > 50);
}

#[test]
fn a_can_laid_across_a_slope_rolls_down_to_a_lip_and_off_without_one() {
    // A can laid level across a slope of 20 degrees rolls down it: with a lip at the foot it
    // comes to rest against the lip, from the spots within half their spacing up the slope of
    // there, and with none it has no rest. Upright on its end it would stand anywhere.
    let can = Obstacle::cylinder(Vec3::ZERO, 0.02, 0.05);
    let across = Quat::from_rotation_y(FRAC_PI_2);
    let o = RestOptions {
        turns: 1,
        ..Default::default()
    };
    assert!(rests_turned(&can, across, &slope(20.0, false), &o).is_empty());
    let against = rests_turned(&can, across, &slope(20.0, true), &o);
    assert!(against.len() >= 5, "{}", against.len());
    let tilt = 20f32.to_radians();
    let lip = Vec3::new(0.0, 0.1 * tilt.cos(), 0.01 - 0.1 * tilt.sin());
    for r in &against {
        // On the slope and at the lip's top edge.
        let normal = Vec3::new(0.0, tilt.sin(), tilt.cos());
        let off = r.pose.position.dot(normal) - 0.02;
        assert!(off.abs() < 5e-4, "{off} m off the slope: {r:?}");
        let gap = can
            .placed(r.pose)
            .distance(lip.with_x(r.pose.position.x))
            .value;
        assert!(gap.abs() < 5e-4, "{gap} m from the lip: {r:?}");
    }
}

#[test]
fn a_slope_steeper_than_the_friction_holds_nothing_without_a_stop() {
    // A surface pushes along its own normal, so a slope steeper than atan 0.5 (26.6 degrees) holds
    // nothing that has no stop below it: not a tile on a 30 degree slope, nor on 35 and 40 degree
    // slopes read with half a millimeter of noise, whose points just past and under its edges
    // would push it nearly straight up along its own normal there; nor a bar 4 mm square set down
    // upright, whose face is not flat on the slope and touches it along an edge.
    let o = RestOptions {
        turns: 1,
        ..Default::default()
    };
    let tile = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.01));
    let bar = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.05, 0.002, 0.002));
    let noisy = |degrees: f32| {
        let tilt = degrees.to_radians();
        let normal = Vec3::new(0.0, tilt.sin(), tilt.cos());
        let mut rng = Rng::new(3);
        let mut s = slope(degrees, false);
        for p in &mut s.points {
            *p += normal * rng.range(-0.0005, 0.0005);
        }
        s
    };
    for (what, object, s) in [
        ("tile on 30 degrees", &tile, slope(30.0, false)),
        ("tile on 35 degrees, noisy", &tile, noisy(35.0)),
        ("tile on 40 degrees, noisy", &tile, noisy(40.0)),
        ("bar on 30 degrees", &bar, slope(30.0, false)),
        ("bar on 45 degrees", &bar, slope(45.0, false)),
    ] {
        let found = rests(object, &s, &o);
        assert!(
            found.is_empty(),
            "{what}: {} rests, as {:?}",
            found.len(),
            found.first()
        );
    }
    // The friction holds it to within a degree of its angle, 26.6: at 26 and not at 27, asked to
    // tip no further. Asked to tip 5 degrees, not at 24.
    let barely = RestOptions { tipping: 0.0, ..o };
    assert!(rests_turned(&tile, lying_on(26.0), &slope(26.0, false), &barely).len() > 50);
    assert!(rests_turned(&tile, lying_on(27.0), &slope(27.0, false), &barely).is_empty());
    assert!(rests_turned(&tile, lying_on(24.0), &slope(24.0, false), &barely).len() > 50);
    assert!(rests_turned(&tile, lying_on(24.0), &slope(24.0, false), &o).is_empty());
    // Under the friction, the noisy slope holds the tile, and a lip holds it on the steep one.
    assert!(rests_turned(&tile, lying_on(20.0), &noisy(20.0), &o).len() > 50);
    let fine = RestOptions {
        spacing: 0.002,
        ..o
    };
    let mut lipped = noisy(35.0);
    lipped.join(slope(35.0, true));
    assert!(!rests_turned(&tile, lying_on(35.0), &lipped, &fine).is_empty());
    // So does a bare wall at its foot, which pushes level: with the slope's friction it holds.
    let tilt = 35f32.to_radians();
    let foot = Vec3::new(0.0, tilt.cos(), -tilt.sin()) * 0.1;
    let mut walled = slope(35.0, false);
    let wall = [0.1, 0.005];
    face(
        foot + Vec3::Z * 0.005,
        Vec3::NEG_Y,
        [Vec3::X, Vec3::Z],
        wall,
        true,
        &mut walled,
    );
    let against = rests_turned(&tile, lying_on(35.0), &walled, &fine);
    assert!(!against.is_empty());
    for r in &against {
        let low = r.support.iter().map(|c| c.z).fold(f32::INFINITY, f32::min);
        assert!(low < foot.z + 0.011, "held by the wall: {r:?}");
    }
}

/// LIBERO's wine rack beside its cabinet, as cameras measured them: a board
/// sloping `degrees` (30, as measured) up toward -y, 12 cm along and 25 cm wide, its side at x = 0 falling to the
/// cabinet's top 5 cm under the slope's foot, which runs on past a gap of 1 cm. All of it is the
/// support, points 2 mm apart.
fn rack_beside_a_cabinet(degrees: f32) -> (Surface, [f32; 2]) {
    let tilt = degrees.to_radians();
    let (normal, down) = (
        Vec3::new(0.0, tilt.sin(), tilt.cos()),
        Vec3::new(0.0, tilt.cos(), -tilt.sin()),
    );
    let mut s = Surface::default();
    face(
        Vec3::new(-0.125, 0.0, 0.0),
        normal,
        [Vec3::X, down],
        [0.125, 0.06],
        true,
        &mut s,
    );
    // The rack's side, down from the slope's edge to the cabinet's top.
    let (foot, cabinet) = (-0.06 * tilt.sin(), -0.06 * tilt.sin() - 0.05);
    let rows = (0.06 * tilt.cos() / SPACING) as i32;
    for j in -rows..=rows {
        let y = j as f32 * SPACING;
        let top = -y * tilt.tan();
        for k in 0..=((top - cabinet) / SPACING) as i32 {
            s.points
                .push(Vec3::new(0.0, y, cabinet + k as f32 * SPACING));
            s.normals.push(Vec3::X);
            s.object.push(true);
        }
    }
    level(
        Vec2::new(0.01, -0.1),
        Vec2::new(0.25, 0.1),
        cabinet,
        true,
        &mut s,
    );
    (s, [foot, cabinet])
}

#[test]
fn a_bottle_laid_on_the_cabinet_may_lie_against_the_side_of_the_rack_beside_it() {
    // A bottle laid down at each of its turns. On the cabinet's top against the rack's side it
    // is clear, held up and steady: a rest, though the normals it touches, the side's and the
    // cabinet's, make no one plane.
    let (s, [_, cabinet]) = rack_beside_a_cabinet(30.0);
    let bottle = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.019, 0.0234, 0.0786));
    let o = RestOptions::default();
    let found = rests_turned(&bottle, Quat::from_rotation_y(FRAC_PI_2), &s, &o);
    let on_the_rack = |p: &Vec3| p.x < 1e-3 && p.z > cabinet + 0.01;
    let on_the_cabinet = |p: &Vec3| (p.z - cabinet).abs() < 1e-3;
    let mut against = 0;
    for r in &found {
        // Nothing of either board in it, and every corner of what holds it up touching it.
        let placed = bottle.placed(r.pose);
        let deepest = s.points.iter().map(|&p| placed.distance(p).value);
        assert!(deepest.fold(f32::INFINITY, f32::min) > -o.touching, "{r:?}");
        for c in &r.support {
            assert!(
                placed.distance(*c).value.abs() <= o.touching + 1e-4,
                "{r:?}"
            );
        }
        if r.support.iter().any(on_the_rack) && r.support.iter().any(on_the_cabinet) {
            against += 1;
            assert!(r.tipping >= o.tipping, "{r:?}");
        }
    }
    assert!(
        against > 0,
        "none lies on the cabinet against the rack of {}",
        found.len()
    );
}

#[test]
fn a_support_seen_only_from_below_holds_nothing_up() {
    // A shelf's underside, its normals down, says nothing of its top; seen from above it does.
    let mut below = Surface::default();
    face(
        Vec3::ZERO,
        -Vec3::Z,
        [Vec3::X, Vec3::Y],
        [0.1, 0.1],
        true,
        &mut below,
    );
    assert!(rests(&a_box(), &below, &RestOptions::default()).is_empty());
    let mut above = below.clone();
    above.normals.fill(Vec3::Z);
    assert!(!rests(&a_box(), &above, &RestOptions::default()).is_empty());
    // Without normals, it is taken as it comes.
    above.normals.fill(Vec3::ZERO);
    assert!(!rests(&a_box(), &above, &RestOptions::default()).is_empty());
}

#[test]
fn a_support_seen_from_above_holds_whatever_its_normals_say() {
    // A board with a lip along it, read by a camera above it at a slant with enough noise that
    // some normals fitted over 5 mm face down: where the camera's place is known they are the
    // board's top, and the box rests where it does not when that place is not known.
    let board = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.01), Vec3::new(0.15, 0.12, 0.01));
    let lip = Obstacle::cuboid(Vec3::new(0.0, 0.08, 0.005), Vec3::new(0.15, 0.005, 0.005));
    let eye = Vec3::new(0.5, 0.0, 0.3);
    let s = measured(&[board, lip], &[0, 1], eye, Vec3::ZERO, 0.003);
    let down = (0..s.points.len()).filter(|&k| s.object[k] && s.normals[k].z < -0.5);
    assert!(down.count() > 20);
    let o = RestOptions::default();
    let known = rests(&a_box(), &s, &o);
    let mut unknown = s.clone();
    unknown.seen_from.clear();
    let without = rests(&a_box(), &unknown, &o);
    assert!(
        known.len() > without.len() + 100,
        "{} with where it was seen from, {} without",
        known.len(),
        without.len()
    );
    // A shelf seen from below holds nothing up, its place known or not: seen from a meter off,
    // its points lie about one to a cell, each the highest of its own.
    let shelf = Obstacle::cuboid(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.1, 0.1, 0.01));
    let from_below = Vec3::new(0.1, 0.0, 0.05);
    let mut under = seen(&[shelf], &[0], from_below, Vec3::new(0.0, 0.0, 1.0));
    assert!(under.points.len() > 1000);
    assert!(under.normals.iter().all(|n| n.z < -0.5));
    assert!(rests(&a_box(), &under, &o).is_empty());
    under.seen_from.fill(Vec3::INFINITY);
    assert!(rests(&a_box(), &under, &o).is_empty());
}

#[test]
fn the_first_rests_are_spread_over_the_support_the_steadiest_first() {
    let found = rests(&a_box(), &table(), &RestOptions::default());
    let first = &found[..8];
    let first_turn = found
        .iter()
        .filter(|r| (r.pose.rotation * Vec3::X).x > 0.9999);
    let steadiest = first_turn.map(|r| r.tipping).fold(0.0, f32::max);
    assert_eq!(first[0].tipping, steadiest);
    for (k, a) in first.iter().enumerate() {
        assert!(
            (a.pose.rotation * Vec3::X).x > 0.9999,
            "the first turn first: {a:?}"
        );
        for b in &first[..k] {
            assert!(
                a.pose.position.distance(b.pose.position) > 0.08,
                "{a:?} {b:?}"
            );
        }
    }
}

#[test]
fn rests_do_not_depend_on_the_threads_the_batch_or_the_device() {
    let robot = common::panda().unwrap();
    let o = RestOptions::default();
    let mut cluttered = table();
    block(
        Vec3::new(0.03, 0.0, 0.04),
        Vec3::new(0.03, 0.03, 0.04),
        false,
        &mut cluttered,
    );
    let can = Obstacle::cylinder(Vec3::ZERO, 0.035, 0.08);
    let tray = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.06, 0.03, 0.02));
    let tile = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.01));
    let surfaces = [
        cluttered,
        bars(),
        Surface::default(),
        slope(20.0, false),
        basket(true, Some(true)),
    ];
    let turned = |object: &Obstacle, surface: usize, rotation: Quat| RestProblem {
        rotation,
        ..RestProblem::new(object.clone(), surface)
    };
    // Two problems on the table, two on the bars, one each on the rest.
    let problems = [
        turned(&a_box(), 0, Quat::IDENTITY),
        turned(&tray, 1, Quat::IDENTITY),
        turned(&a_box(), 2, Quat::IDENTITY),
        turned(&tile, 3, lying_on(20.0)),
        turned(&can, 4, Quat::IDENTITY),
        turned(&can, 1, Quat::from_rotation_x(FRAC_PI_2)),
        turned(&tile, 0, Quat::from_rotation_z(0.3)),
        RestProblem {
            near: Some(Vec3::new(-0.081, 0.062, 0.5)),
            ..RestProblem::new(a_box(), 0)
        },
    ];
    let one = Device::cpu_threads(&robot, 1).unwrap();
    let alone: Vec<Vec<Rest>> = problems
        .iter()
        .map(|p| {
            let problem = RestProblem {
                surface: 0,
                ..p.clone()
            };
            let s = std::slice::from_ref(&surfaces[p.surface]);
            rests_on(&one, &[problem], s, &o).unwrap().remove(0)
        })
        .collect();
    assert!(alone[0].len() > 100 && !alone[1].is_empty() && alone[2].is_empty());
    assert!(!alone[3].is_empty() && !alone[4].is_empty());
    assert!(!alone[5].is_empty() && !alone[6].is_empty());
    assert!(alone[7].len() > alone[0].len());
    let mut devices = vec![one, Device::cpu(&robot).unwrap()];
    match Device::gpu(&robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    for device in &devices {
        // Sharing surfaces, and each with a surface of its own.
        let together = rests_on(device, &problems, &surfaces, &o).unwrap();
        assert_eq!(together, alone, "{}", device.name());
        let own: Vec<Surface> = problems
            .iter()
            .map(|p| surfaces[p.surface].clone())
            .collect();
        let apart: Vec<RestProblem> = (0..problems.len())
            .map(|k| RestProblem {
                surface: k,
                ..problems[k].clone()
            })
            .collect();
        assert_eq!(
            rests_on(device, &apart, &own, &o).unwrap(),
            alone,
            "{}",
            device.name()
        );
    }
}

#[test]
fn malformed_input_is_an_input_error() {
    let device = device();
    let good = table();
    let o = RestOptions::default();
    let refused = |p: &RestProblem, s: &Surface, o: &RestOptions| {
        let s = std::slice::from_ref(s);
        match rests_on(&device, std::slice::from_ref(p), s, o) {
            Err(Error::Input(why)) => Some(why),
            _ => None,
        }
    };
    let a = RestProblem::new(a_box(), 0);
    assert!(refused(&a, &good, &o).is_none());
    // Problems name what is wrong with them.
    let elsewhere = RestProblem::new(a_box(), 1);
    let both = rests_on(
        &device,
        &[a.clone(), elsewhere],
        std::slice::from_ref(&good),
        &o,
    );
    assert!(
        matches!(&both, Err(Error::Input(why)) if why.starts_with("problem 1: surface 1 of 1"))
    );
    let grid = batchplan::SdfGrid::try_new([2, 2, 2], 0.1, Vec3::ZERO, &[1.0; 8]).unwrap();
    let grid = RestProblem::new(Obstacle::from_grid(grid), 0);
    assert!(refused(&grid, &good, &o).is_some_and(|why| why.starts_with("problem 0")));
    let nan = RestProblem::new(
        Obstacle::cuboid(Vec3::ZERO, Vec3::new(f32::NAN, 0.1, 0.1)),
        0,
    );
    assert!(refused(&nan, &good, &o).is_some());
    let turned = RestProblem {
        rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 2.0),
        ..a.clone()
    };
    assert!(refused(&turned, &good, &o).is_some());
    let nowhere_near = RestProblem {
        near: Some(Vec3::new(0.0, f32::NAN, 0.0)),
        ..a.clone()
    };
    assert!(refused(&nowhere_near, &good, &o).is_some_and(|why| why.starts_with("problem 0")));
    // Malformed surfaces.
    let mut short = good.clone();
    short.normals.pop();
    assert!(
        refused(&a, &short, &o).is_some(),
        "a point without a normal"
    );
    let mut flags = good.clone();
    flags.object = vec![true];
    assert!(
        refused(&a, &flags, &o).is_some(),
        "flags for some points only"
    );
    let mut nowhere = good.clone();
    nowhere.points[3].y = f32::INFINITY;
    assert!(
        refused(&a, &nowhere, &o).is_some(),
        "a point that is not finite"
    );
    // Normals that are not unit, refused as `grasps_on` refuses them: at 0.4 of unit a table
    // would hold nothing up and say nothing.
    for scale in [0.4f32, 3.0] {
        let mut scaled = good.clone();
        scaled.normals.iter_mut().for_each(|n| *n *= scale);
        let why = refused(&a, &scaled, &o).unwrap_or_default();
        assert!(
            why.starts_with("surface 0: normal 0 must be unit or zero"),
            "{why}"
        );
    }
    for bad in [
        RestOptions { turns: 0, ..o },
        RestOptions { spacing: 0.0, ..o },
        RestOptions {
            touching: f32::NAN,
            ..o
        },
        RestOptions {
            clearance: -0.01,
            ..o
        },
        RestOptions {
            tipping: FRAC_PI_2,
            ..o
        },
        RestOptions {
            friction: f32::INFINITY,
            ..o
        },
    ] {
        assert!(refused(&a, &good, &bad).is_some(), "{bad:?}");
    }
    // Spots a micrometer apart over the table are too many.
    assert!(refused(&a, &good, &RestOptions { spacing: 1e-6, ..o }).is_some());
    // Of two surfaces too wide for the spots, the first is named, on one thread or many.
    let wide = |points: usize| Surface {
        points: (0..points)
            .map(|k| {
                Vec3::new(
                    20.0 * k as f32 / (points - 1) as f32,
                    20.0 * (k % 2) as f32,
                    0.0,
                )
            })
            .collect(),
        normals: vec![Vec3::Z; points],
        ..Default::default()
    };
    let wides = [wide(2_000_000), wide(2)];
    let problems = [RestProblem::new(a_box(), 0), RestProblem::new(a_box(), 1)];
    let robot = common::panda().unwrap();
    for device in [
        Device::cpu_threads(&robot, 1).unwrap(),
        Device::cpu(&robot).unwrap(),
    ] {
        let why = match rests_on(&device, &problems, &wides, &o) {
            Err(Error::Input(why)) => why,
            other => panic!("{other:?}"),
        };
        assert!(
            why.starts_with("surface 0: the support spans 20.00 by 20.00 m"),
            "{why}"
        );
    }
    // No problems, and a surface without points, give no rests.
    assert!(rests_on(&device, &[], &[], &o).unwrap().is_empty());
    assert!(rests(&a_box(), &Surface::default(), &o).is_empty());
}

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
const K: Intrinsics = Intrinsics {
    fx: 300.0,
    fy: 300.0,
    cx: 160.0,
    cy: 120.0,
};

/// A camera at `eye` looking at `target`, its image's x axis level.
fn looking(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).try_normalize().unwrap_or(Vec3::X);
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
}

/// What a camera at `eye` looking at `target` sees of `scene`: its points, those of `support`
/// (an index into `scene`) marked as the support.
fn seen(scene: &[Obstacle], support: &[usize], eye: Vec3, target: Vec3) -> Surface {
    measured(scene, support, eye, target, 0.0)
}

/// [`seen`], with every depth read off by up to `noise` (meters) either way.
fn measured(scene: &[Obstacle], support: &[usize], eye: Vec3, target: Vec3, noise: f32) -> Surface {
    let camera = looking(eye, target);
    let mut rng = Rng::new(11);
    let depth: Vec<f32> = (0..WIDTH * HEIGHT)
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let ray = Vec3::new((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0);
            let dir = camera.rotation * ray.normalize();
            let mut t = 0.0;
            while t < 2.0 {
                let d = scene
                    .iter()
                    .map(|o| o.distance(camera.position + dir * t).value)
                    .fold(f32::INFINITY, f32::min);
                if d < 1e-5 {
                    return t * ray.normalize().z;
                }
                t += d;
            }
            0.0
        })
        .map(|z| {
            if z > 0.0 && noise > 0.0 {
                z + rng.range(-noise, noise)
            } else {
                z
            }
        })
        .collect();
    let image = DepthImage {
        depth: &depth,
        width: WIDTH,
        intrinsics: K,
        camera,
    };
    let flags: Vec<bool> = (0..depth.len())
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let z = depth[i];
            let p = camera.rotation * Vec3::new((u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z, z)
                + camera.position;
            let nearest = (0..scene.len()).min_by(|&a, &b| {
                scene[a]
                    .distance(p)
                    .value
                    .total_cmp(&scene[b].distance(p).value)
            });
            z > 0.0 && nearest.is_some_and(|k| support.contains(&k))
        })
        .collect();
    Surface::from_depth(&device(), &image, &flags, None, 0.005).unwrap()
}

#[test]
fn a_basket_seen_from_the_front_alone_has_no_rest_inside_and_seen_from_above_has() {
    // A basket 16 cm square and 10 cm tall, its walls 1 cm thick, its floor 1 cm thick, on a
    // table: a camera in front and low sees its near wall and nothing of its floor.
    let table = Obstacle::cuboid(Vec3::new(0.5, 0.0, -0.01), Vec3::new(0.3, 0.3, 0.01));
    let mut scene = vec![
        table,
        Obstacle::cuboid(Vec3::new(0.5, 0.0, 0.005), Vec3::new(0.08, 0.08, 0.005)),
    ];
    for (x, y, hx, hy) in [
        (0.075, 0.0, 0.005, 0.08),
        (-0.075, 0.0, 0.005, 0.08),
        (0.0, 0.075, 0.08, 0.005),
        (0.0, -0.075, 0.08, 0.005),
    ] {
        scene.push(Obstacle::cuboid(
            Vec3::new(0.5 + x, y, 0.05),
            Vec3::new(hx, hy, 0.05),
        ));
    }
    let basket = [1, 2, 3, 4, 5];
    let front = seen(
        &scene,
        &basket,
        Vec3::new(0.0, 0.0, 0.12),
        Vec3::new(0.5, 0.0, 0.05),
    );
    let can = Obstacle::cylinder(Vec3::ZERO, 0.025, 0.04);
    let o = RestOptions::default();
    let inside = |r: &Rest| {
        (r.pose.position.truncate() - Vec2::new(0.5, 0.0))
            .abs()
            .max_element()
            < 0.07
    };
    let floor = |r: &Rest| (r.pose.position.z - 0.05).abs() < 2e-3;
    let from_the_front = rests(&can, &front, &o);
    assert!(!from_the_front.iter().any(|r| inside(r) && floor(r)));
    // Looking in from above, a wrist camera's view, its floor is seen and held.
    let mut both = front;
    both.join(seen(
        &scene,
        &basket,
        Vec3::new(0.5, 0.0, 0.5),
        Vec3::new(0.5, 0.0, 0.0),
    ));
    let from_above = rests(&can, &both, &o);
    let on_the_floor: Vec<&Rest> = from_above
        .iter()
        .filter(|r| inside(r) && floor(r))
        .collect();
    assert!(!on_the_floor.is_empty());
    for r in on_the_floor {
        // Upright, and within the walls, which as the support it may touch.
        assert!(up(r).z > 0.9999, "{r:?}");
        let off = (r.pose.position.truncate() - Vec2::new(0.5, 0.0))
            .abs()
            .max_element();
        assert!(off + 0.025 <= 0.07 + 1e-3, "{r:?}");
    }
}

#[test]
fn rests_found_on_what_a_camera_sees_are_set_down_by_place() {
    // A table and a block standing on it, seen from above and in front; the table is the
    // support. A box held from above is set down at the first rests, the table its support
    // in the world as the block's neighbour, and lands on it clear of the block.
    let robot = common::panda().unwrap();
    let device = Device::cpu(&robot).unwrap();
    let table = Obstacle::cuboid(Vec3::new(0.45, 0.0, -0.01), Vec3::new(0.15, 0.15, 0.01));
    let neighbour = Obstacle::cuboid(Vec3::new(0.45, 0.0, 0.05), Vec3::new(0.03, 0.03, 0.05));
    let scene = [table, neighbour];
    let mut s = seen(
        &scene,
        &[0],
        Vec3::new(0.45, 0.0, 0.6),
        Vec3::new(0.45, 0.0, 0.0),
    );
    s.join(seen(
        &scene,
        &[0],
        Vec3::new(0.95, 0.0, 0.4),
        Vec3::new(0.45, 0.0, 0.0),
    ));
    let o = RestOptions::default();
    let found = rests(&a_box(), &s, &o);
    assert!(found.len() > 20);
    for r in &found {
        assert!(
            (r.pose.position.z - HALF.z).abs() < 2e-3 && up(r).z > 0.999,
            "{r:?}"
        );
        let box_at = a_box().placed(r.pose);
        let block_points = s
            .points
            .iter()
            .zip(&s.object)
            .filter(|(_, support)| !**support);
        let nearest = block_points.map(|(p, _)| box_at.distance(*p).value);
        assert!(
            nearest.fold(f32::INFINITY, f32::min) >= o.clearance - 1e-4,
            "{r:?}"
        );
    }
    let held = Held {
        locked: Default::default(),
        spheres: Default::default(),
        hand: 0,
        object: a_box(),
        // From above, closing along its x axis, 2 cm in from its top.
        grasp: Pose::new(
            Vec3::Z * 0.005,
            Quat::from_mat3(&Mat3::from_cols(Vec3::Y, Vec3::X, Vec3::NEG_Z)),
        ),
    };
    let world = World::new(scene.to_vec());
    let worlds = device.upload(&[world]).unwrap();
    let poses: Vec<Pose> = found.iter().take(16).map(|r| r.pose).collect();
    let place = Place {
        support: vec![0],
        ..Place::new(0, robot.default_q(), held, poses)
    };
    let placed = batchplan::place(&device, &worlds, &[place], &TaskOptions::default()).unwrap();
    let plans = placed.outcomes[0].as_ref().expect("a rest is set down");
    assert!(!plans.is_empty());
}

/// How far the nearest sum of `columns`, each weighted by no less than nought, comes from `b`: Lawson
/// and Hanson's least squares over nonnegative weights.
fn unbalanced(columns: &[[f64; 6]], b: [f64; 6]) -> f64 {
    let dot = |a: &[f64; 6], b: &[f64; 6]| (0..6).map(|i| a[i] * b[i]).sum::<f64>();
    let left = |x: &[f64]| {
        let mut left = b;
        for (column, weight) in columns.iter().zip(x) {
            for i in 0..6 {
                left[i] -= column[i] * weight;
            }
        }
        left
    };
    // The weights of the columns `set` nearest `b`, by their normal equations.
    let nearest = |set: &[usize]| -> Vec<f64> {
        let n = set.len();
        let mut a = vec![vec![0.0f64; n + 1]; n];
        for (p, &i) in set.iter().enumerate() {
            for (q, &j) in set.iter().enumerate() {
                a[p][q] = dot(&columns[i], &columns[j]);
            }
            a[p][p] += 1e-12;
            a[p][n] = dot(&columns[i], &b);
        }
        for c in 0..n {
            let pivot = (c..n)
                .max_by(|&p, &q| a[p][c].abs().total_cmp(&a[q][c].abs()))
                .expect("a column below the diagonal");
            a.swap(c, pivot);
            let row = a[c].clone();
            if row[c].abs() < 1e-18 {
                continue;
            }
            for (r, other) in a.iter_mut().enumerate() {
                let f = other[c] / row[c];
                if r != c {
                    for k in c..=n {
                        other[k] -= f * row[k];
                    }
                }
            }
        }
        let solved = a.iter().enumerate().map(|(r, row)| {
            if row[r].abs() < 1e-18 {
                0.0
            } else {
                row[n] / row[r]
            }
        });
        solved.collect()
    };
    let mut x = vec![0.0f64; columns.len()];
    let mut passive = vec![false; columns.len()];
    for _ in 0..3 * columns.len() + 6 {
        let rest = left(&x);
        let free = (0..columns.len()).filter(|&j| !passive[j]);
        let next =
            free.max_by(|&p, &q| dot(&columns[p], &rest).total_cmp(&dot(&columns[q], &rest)));
        let Some(j) = next.filter(|&j| dot(&columns[j], &rest) > 1e-10) else {
            break;
        };
        passive[j] = true;
        for _ in 0..3 * columns.len() + 6 {
            let set: Vec<usize> = (0..columns.len()).filter(|&k| passive[k]).collect();
            let z = nearest(&set);
            if z.iter().all(|&v| v > 0.0) {
                for (&k, &v) in set.iter().zip(&z) {
                    x[k] = v;
                }
                break;
            }
            let steps = set.iter().zip(&z).filter(|&(_, &v)| v <= 0.0);
            let step = steps
                .map(|(&k, &v)| x[k] / (x[k] - v))
                .fold(f64::INFINITY, f64::min);
            for (&k, &v) in set.iter().zip(&z) {
                x[k] += step * (v - x[k]);
                if x[k] <= 1e-14 {
                    x[k] = 0.0;
                    passive[k] = false;
                }
            }
        }
    }
    let rest = left(&x);
    dot(&rest, &rest).sqrt()
}

/// Whether forces at the support points within `touching` of the box or cylinder `object`
/// resting at `r`, each inside its friction cone (sixteen rays of it), can carry its weight with
/// no torque about its middle: a check that shares nothing with `rests_on`. A point pushes along
/// the object's own normal where it lies against the inside of a face, half a millimeter from
/// its edges, and along its own elsewhere, as hard as it takes. Every point that near counts,
/// more than touch it, so a rest that fails this cannot hold.
fn holds(object: &Obstacle, r: &Rest, s: &Surface, o: &RestOptions) -> bool {
    // The face nearest a point in the object's frame: the way out of it, and whether the point
    // lies inside its edges.
    let face = |local: Vec3| match *object {
        Obstacle::Cuboid {
            half_extents: half, ..
        } => {
            let past = (local.abs() - half).to_array();
            let axis = (0..3)
                .max_by(|&a, &b| past[a].total_cmp(&past[b]))
                .expect("three axes");
            let inside = (0..3).all(|i| i == axis || local[i].abs() <= half[i] - 0.0005);
            (Vec3::AXES[axis] * local[axis].signum(), inside)
        }
        Obstacle::Cylinder {
            radius,
            half_height,
            ..
        } => {
            let across = local.truncate().length();
            if local.z.abs() - half_height >= across - radius {
                (Vec3::Z * local.z.signum(), across <= radius - 0.0005)
            } else {
                let out = (local.truncate() / across).extend(0.0);
                (out, local.z.abs() <= half_height - 0.0005)
            }
        }
        _ => panic!("the check is written for boxes and cylinders, got {object:?}"),
    };
    let placed = object.placed(r.pose);
    let cone = o.friction.atan();
    let mut rays: Vec<[f64; 6]> = vec![];
    for (k, &p) in s.points.iter().enumerate() {
        let distance = placed.distance(p).value;
        let support = s.object.is_empty() || s.object[k];
        if !support || distance > o.touching || distance < -1e-3 {
            continue;
        }
        let local = r.pose.rotation.inverse() * (p - r.pose.position);
        let (out, inside) = face(local);
        let push = if inside || s.normals[k] == Vec3::ZERO {
            -(r.pose.rotation * out)
        } else {
            s.normals[k]
        };
        let (e1, e2) = push.any_orthonormal_pair();
        for j in 0..16 {
            let around = TAU * j as f32 / 16.0;
            let force = push * cone.cos() + (e1 * around.cos() + e2 * around.sin()) * cone.sin();
            let torque = (p - r.pose.position).cross(force) / 0.05;
            let ray = [force.x, force.y, force.z, torque.x, torque.y, torque.z];
            rays.push(ray.map(f64::from));
        }
    }
    unbalanced(&rays, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) < 1e-3
}

#[test]
fn a_ramp_up_to_a_platform_holds_nothing_on_the_ramp() {
    // A ramp steeper than the friction rising to a level platform: a tile and a bottle laid down
    // rest on the platform, or from it down onto the ramp, and nowhere on the ramp alone, where
    // the platform's edge above pushes nothing up the slope.
    let tile = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.01));
    let bottle = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.019, 0.0234, 0.0786));
    let o = RestOptions::default();
    let once = RestOptions { turns: 1, ..o };
    for degrees in [30.0f32, 35.0, 45.0] {
        let angle = degrees.to_radians();
        let top = Vec3::new(0.0, -angle.cos(), angle.sin()) * 0.1;
        let mut s = slope(degrees, false);
        level(
            Vec2::new(-0.1, top.y - 0.1),
            Vec2::new(0.1, top.y),
            top.z,
            true,
            &mut s,
        );
        let laid = Quat::from_rotation_y(FRAC_PI_2);
        for (object, rotation, o) in [(&tile, Quat::IDENTITY, &once), (&bottle, laid, &o)] {
            let found = rests_turned(object, rotation, &s, o);
            assert!(found.len() >= 200, "{degrees}: {}", found.len());
            for r in &found {
                let on_the_platform = r.support.iter().any(|p| p.y <= top.y + 1e-3);
                assert!(on_the_platform, "{degrees}: on the ramp alone: {r:?}");
                assert!(holds(object, r, &s, o), "{degrees}: cannot hold: {r:?}");
            }
        }
    }
}

#[test]
fn a_board_a_camera_measured_holds_only_short_of_the_friction() {
    // A board sloping past the friction, read by a camera with up to a millimeter of noise, whose
    // normals come out degrees off: a tile rests nowhere on it, however its points and normals
    // scatter. At 20 degrees, 5 short of the friction less the tipping asked for, it rests.
    let tile = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.01));
    let o = RestOptions {
        turns: 1,
        ..Default::default()
    };
    let read = |degrees: f32, noise: f32| {
        let rotation = Quat::from_rotation_x(-degrees.to_radians());
        let board = Obstacle::Cuboid {
            center: Vec3::new(0.0, 0.0, -0.01),
            half_extents: Vec3::new(0.12, 0.12, 0.01),
            rotation,
        };
        let eye = Vec3::new(0.0, 0.25, 0.45);
        let s = measured(std::slice::from_ref(&board), &[0], eye, Vec3::ZERO, noise);
        // The normals are off the board's by degrees, half of them by more than one.
        let known = s.normals.iter().filter(|n| **n != Vec3::ZERO);
        let mut off: Vec<f32> = known
            .map(|n| n.dot(rotation * Vec3::Z).clamp(-1.0, 1.0).acos())
            .collect();
        off.sort_by(f32::total_cmp);
        assert!(off[off.len() / 2] > 1f32.to_radians(), "{degrees}: {off:?}");
        s
    };
    for (degrees, noise) in [(30.0, 0.0005), (35.0, 0.001), (40.0, 0.001)] {
        let found = rests_turned(&tile, lying_on(degrees), &read(degrees, noise), &o);
        assert!(found.is_empty(), "{degrees}: {} rests", found.len());
    }
    let gentle = read(20.0, 0.001);
    let found = rests_turned(&tile, lying_on(20.0), &gentle, &o);
    assert!(found.len() > 100, "{}", found.len());
    for r in &found {
        assert!(holds(&tile, r, &gentle, &o), "{r:?}");
    }
}

#[test]
fn a_log_holds_a_tile_only_as_far_round_as_the_friction_does() {
    // A log 20 cm thick along x, its top at 0: a tile laid tangent to it rests as far round from
    // the top as the friction holds it with 5 degrees to spare, and no farther.
    let radius = 0.1f32;
    let mut log = Surface::default();
    let around = (std::f32::consts::PI * radius / SPACING) as usize;
    for k in 0..=around {
        let angle = std::f32::consts::PI * k as f32 / around as f32;
        let normal = Vec3::new(0.0, angle.cos(), angle.sin());
        for j in 0..=100 {
            let x = j as f32 * SPACING - 0.1;
            log.points
                .push(Vec3::new(x, 0.0, -radius) + normal * radius);
            log.normals.push(normal);
        }
    }
    let tile = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.01));
    let o = RestOptions {
        turns: 1,
        ..Default::default()
    };
    for degrees in [0.0f32, 10.0, 20.0] {
        let found = rests_turned(&tile, lying_on(degrees), &log, &o);
        assert!(!found.is_empty(), "none at {degrees} degrees round");
        let (sin, cos) = degrees.to_radians().sin_cos();
        for r in &found {
            // Tangent to it, as near as the log stays within `touching` of the tile: 2 cm to
            // either side of where its normal is the tile's.
            let out = r.pose.position - Vec3::new(r.pose.position.x, 0.0, -radius);
            let off = out - Vec3::new(0.0, sin, cos) * (radius + 0.01);
            assert!(off.length() < 0.021, "{off} from tangent: {r:?}");
            assert!(holds(&tile, r, &log, &o), "{r:?}");
        }
    }
    for degrees in [25.0f32, 30.0] {
        let found = rests_turned(&tile, lying_on(degrees), &log, &o);
        assert!(
            found.is_empty(),
            "{} at {degrees} degrees round",
            found.len()
        );
    }
}

#[test]
fn every_rest_on_a_rack_holds_by_force_and_torque() {
    // The example's rack, 30 degrees with a lip 1 cm high along its foot, and a bottle laid
    // across it flat on the slope, steeper than the friction holds: it rests only against the
    // lip, and every rest holds by a check of its own.
    let angle = 30f32.to_radians();
    let (normal, down) = (
        Vec3::new(0.0, angle.sin(), angle.cos()),
        Vec3::new(0.0, angle.cos(), -angle.sin()),
    );
    let mut rack = Surface::default();
    face(
        Vec3::ZERO,
        normal,
        [Vec3::X, down],
        [0.15, 0.06],
        true,
        &mut rack,
    );
    let lip = down * 0.06 + Vec3::Z * 0.005;
    face(
        lip,
        -Vec3::Y,
        [Vec3::X, Vec3::Z],
        [0.15, 0.005],
        true,
        &mut rack,
    );
    let o = RestOptions::default();
    let laid = Quat::from_rotation_y(FRAC_PI_2);
    let bottle = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.019, 0.023, 0.079));
    let fine = RestOptions {
        turns: 1,
        spacing: 0.002,
        ..o
    };
    let found = rests_turned(&bottle, lying_on(30.0) * laid, &rack, &fine);
    assert!(found.len() >= 10, "{}", found.len());
    for r in &found {
        assert!(holds(&bottle, r, &rack, &fine), "{r:?}");
        // Its lower long edge at the lip's top edge.
        let edge = (down * 0.06 + Vec3::Z * 0.01).with_x(r.pose.position.x);
        let gap = bottle.placed(r.pose).distance(edge).value;
        assert!(gap.abs() < 2.5e-3, "{gap} m from the lip: {r:?}");
    }
    assert!(rests_turned(&bottle, laid, &rack, &fine).is_empty());

    // Beside the cabinet, at the slopes such racks are measured with, the bottle laid level
    // rests over the cabinet's top, and every rest holds.
    let bottle = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.019, 0.0234, 0.0786));
    for degrees in [30.0, 35.0, 40.0, 45.0] {
        let (s, _) = rack_beside_a_cabinet(degrees);
        let found = rests_turned(&bottle, laid, &s, &o);
        assert!(found.len() >= 1600, "{degrees}: {}", found.len());
        for r in &found {
            assert!(holds(&bottle, r, &s, &o), "{degrees}: {r:?}");
        }
    }
}

#[test]
fn a_wall_at_the_foot_of_a_steep_slope_holds_what_lies_against_it_whatever_its_height() {
    // A bare wall at the foot of a 35 degree slope, lower than a tile is thick and higher: the
    // tile laid flat on the slope rests against it, the wall's top under the tile's upper edge
    // where it is lower than 2.4 cm.
    let angle = 35f32.to_radians();
    let (normal, down) = (
        Vec3::new(0.0, angle.sin(), angle.cos()),
        Vec3::new(0.0, angle.cos(), -angle.sin()),
    );
    let tile = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.01));
    let foot = down * 0.1;
    for height in [0.006f32, 0.01, 0.03] {
        let mut s = slope(35.0, false);
        let middle = foot + Vec3::Z * (height / 2.0);
        let half = [0.1, height / 2.0];
        face(middle, -Vec3::Y, [Vec3::X, Vec3::Z], half, true, &mut s);
        {
            let o = RestOptions {
                turns: 1,
                spacing: 0.002,
                ..Default::default()
            };
            let found = rests_turned(&tile, lying_on(35.0), &s, &o);
            assert!(!found.is_empty(), "wall {height} m: none");
            for r in &found {
                assert!(up(r).dot(normal) > 0.999, "{r:?}");
                // Against the wall: its lower edge as far up the slope from the foot as the
                // wall's top holds its face, or its upper edge against a wall taller than it.
                let edge = r.pose.position + r.pose.rotation * Vec3::new(0.0, 0.03, -0.01);
                let from_the_foot = (foot - edge).dot(down);
                let against = (height * angle.sin()).min(0.02 * angle.tan());
                assert!(
                    (from_the_foot - against).abs() < o.touching + 1e-3,
                    "wall {height} m: {from_the_foot} m up from the foot: {r:?}"
                );
                assert!(holds(&tile, r, &s, &o), "wall {height} m: {r:?}");
            }
        }
    }
}

#[test]
fn noisy_normals_a_dish_a_groove_and_a_gentle_slope_hold_what_lies_in_them() {
    let tile = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.01));
    let o = RestOptions {
        turns: 1,
        ..Default::default()
    };
    // A level table whose normals are each up to 10 degrees off.
    let mut noisy = table();
    let mut rng = Rng::new(5);
    for n in &mut noisy.normals {
        let axis = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 0.0);
        let angle = rng.range(0.0, 10f32.to_radians());
        *n = Quat::from_axis_angle(axis.normalize_or(Vec3::X), angle) * *n;
    }
    let off = noisy
        .normals
        .iter()
        .filter(|n| n.z < 5f32.to_radians().cos());
    assert!(off.count() > noisy.normals.len() / 4);
    assert_eq!(rests(&tile, &noisy, &o).len(), 900);
    // A dish 12 cm across, a cap of a sphere of 30 cm.
    let mut dish = Surface::default();
    let radius = 0.3f32;
    for i in -30..=30 {
        for j in -30..=30 {
            let (x, y) = (i as f32 * SPACING, j as f32 * SPACING);
            let p = Vec3::new(x, y, radius - (radius * radius - x * x - y * y).sqrt());
            dish.points.push(p);
            dish.normals.push((Vec3::Z * radius - p).normalize());
            dish.object.push(true);
        }
    }
    // A level tile stands on its four corners over the dish's middle, and nowhere else.
    let in_the_dish = rests(&tile, &dish, &o);
    assert_eq!(in_the_dish.len(), 4);
    for r in &in_the_dish {
        assert!(r.pose.position.truncate().length() < 0.01, "{r:?}");
    }
    // A groove along x, its sides at 45 degrees: a box square across lies in it corner down, and
    // a can lies along it.
    let mut groove = Surface::default();
    for side in [1.0f32, -1.0] {
        let normal = Vec3::new(0.0, -side, 1.0).normalize();
        let along = Vec3::new(0.0, side, 1.0).normalize();
        face(
            along * 0.05,
            normal,
            [Vec3::X, along],
            [0.1, 0.05],
            true,
            &mut groove,
        );
    }
    let square = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.05, 0.015, 0.015));
    // Corner down, it rests wedged over the groove's line, where a spot comes within `touching`
    // of it; flat, it stands on two edges and rests there too, and nowhere off the line.
    let fine = RestOptions {
        spacing: 0.002,
        ..o
    };
    let corner_down = Quat::from_rotation_x(FRAC_PI_2 / 2.0);
    for rotation in [corner_down, Quat::IDENTITY] {
        let wedged = rests_turned(&square, rotation, &groove, &fine);
        assert!(wedged.len() >= 20, "{}", wedged.len());
        for r in &wedged {
            assert!(r.pose.position.y.abs() < 2e-3, "{r:?}");
        }
    }
    let can = Obstacle::cylinder(Vec3::ZERO, 0.02, 0.05);
    let along = Quat::from_rotation_y(FRAC_PI_2);
    let capsule = Obstacle::capsule(Vec3::ZERO, 0.02, 0.05);
    for round in [&can, &capsule] {
        let found = rests_turned(round, along, &groove, &fine);
        assert!(found.len() >= 18, "{}", found.len());
        for r in &found {
            assert!(r.pose.position.y.abs() < 2e-3, "{r:?}");
        }
    }
    // Upright, the capsule stands on its round end and has no rest on a table.
    assert!(rests(&capsule, &table(), &o).is_empty());
    // A slope of 20 degrees.
    assert_eq!(
        rests_turned(&tile, lying_on(20.0), &slope(20.0, false), &o).len(),
        360
    );
}

#[test]
fn a_flat_bottom_on_a_level_support_tips_as_far_as_the_general_balance_finds() {
    // A box flat on a level table is judged in closed form; set down a degree off level, within
    // `touching` of the table across its bottom, by the balance of forces every other rest is.
    // Along the table's edge, hanging past it by more and more, the two come to the same angle,
    // to that degree and the half a degree the balance is found to: about the edge where the box
    // hangs far past it, and about its own where it does not.
    let tall = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.03, 0.05));
    let o = RestOptions {
        turns: 1,
        ..Default::default()
    };
    let level = rests(&tall, &table(), &o);
    let off = Quat::from_rotation_x(1f32.to_radians());
    let tilted = rests_turned(&tall, off, &table(), &o);
    assert!(level.len() > 500 && tilted.len() == level.len());
    let mut margins = vec![];
    for closed in &level {
        let at = closed.pose.position.truncate();
        let same = |r: &&Rest| r.pose.position.truncate().distance(at) < 1e-4;
        let general = tilted
            .iter()
            .find(same)
            .expect("the same spot, set down tilted");
        assert!(up(closed).z == 1.0 && up(general).z < 0.9999);
        let found_to = 0.5f32.to_radians() + 1f32.to_radians();
        assert!(
            (general.tipping - closed.tipping).abs() < found_to + 1e-3,
            "{} and {} rad: {general:?}",
            general.tipping,
            closed.tipping
        );
        margins.push(closed.tipping);
    }
    // Its middle 5 mm and 15 mm in from the edge, and with the table under all of it, to the last
    // points 1 mm inside its own edges: 5.7, 16.7 and 20.8 degrees.
    for inset in [0.005f32, 0.015, 0.019] {
        let expected = (inset / 0.05).atan();
        let same = margins.iter().filter(|m| (*m - expected).abs() < 1e-3);
        assert!(same.count() > 20, "{expected} rad of {margins:?}");
    }
}

/// A pen 20 by 16 cm inside, its floor's top 1 cm up and its walls 1 cm thick and `height` over
/// the floor, all the support, the wall at +x with a gap `gap` wide in its middle, and nothing
/// outside it.
fn pen(height: f32, gap: f32) -> Surface {
    let mut s = Surface::default();
    let inner = Vec2::new(0.1, 0.08);
    level(-inner, inner, 0.01, true, &mut s);
    let (z, half) = (0.01 + height / 2.0, height / 2.0);
    let wall = |at: Vec2, half_across: Vec2, s: &mut Surface| {
        block(at.extend(z), half_across.extend(half), true, s);
    };
    wall(Vec2::new(-0.105, 0.0), Vec2::new(0.005, 0.09), &mut s);
    wall(Vec2::new(0.0, 0.085), Vec2::new(0.1, 0.005), &mut s);
    wall(Vec2::new(0.0, -0.085), Vec2::new(0.1, 0.005), &mut s);
    let piece = (0.09 - gap / 2.0) / 2.0;
    for sign in [1.0, -1.0] {
        let at = Vec2::new(0.105, sign * (0.09 - piece));
        wall(at, Vec2::new(0.005, piece), &mut s);
    }
    s
}

fn falls(object: &Obstacle, surface: &Surface, o: &RestOptions) -> Vec<Fall> {
    let problem = RestProblem::new(object.clone(), 0);
    falls_into(&device(), &[problem], std::slice::from_ref(surface), o)
        .unwrap()
        .remove(0)
}

#[test]
fn what_is_let_go_over_a_pen_falls_in_and_stays() {
    // A can 6 cm across and 8 cm tall over a pen with walls 8 cm tall. Grown by the clearance it
    // is 8 cm across: it is let go where that fits inside the walls, its bottom a centimeter
    // over their tops, and lands on the floor 8 cm under where it would leave.
    let can = Obstacle::cylinder(Vec3::ZERO, 0.03, 0.04);
    let o = RestOptions::default();
    let found = falls(&can, &pen(0.08, 0.0), &o);
    for f in &found {
        let at = f.pose.position;
        assert!((at.z - 0.04 - (0.09 + o.clearance)).abs() < 1e-4, "{f:?}");
        let room = Vec2::new(0.1, 0.08) - 0.03 - o.clearance + 1e-4;
        assert!(at.truncate().abs().cmple(room).all(), "{f:?}");
        assert!((f.sunk - 0.08).abs() < 1e-4, "{f:?}");
    }
    // Every spot where it fits, 12 by 8, at every turn; the middle first.
    assert_eq!(found.len(), 12 * 8 * 8);
    let first = found[0].pose.position.truncate();
    assert!(first.abs().max_element() < 0.01, "{first}");
    // Walls lower than its middle, a gap it fits through, a floor no camera saw and a table hold
    // nothing in. A gap narrower than it does.
    for (what, s) in [
        ("low walls", pen(0.03, 0.0)),
        ("a gap of 9 cm", pen(0.08, 0.09)),
        (
            "a gap of 6.5 cm, which it fits through and its clearance does not",
            pen(0.08, 0.065),
        ),
        ("no floor", basket(false, Some(true))),
        ("a table", table()),
    ] {
        let none = falls(&can, &s, &o);
        assert!(none.is_empty(), "{what}: {}", none.len());
    }
    assert_eq!(falls(&can, &pen(0.08, 0.04), &o).len(), found.len());
    // Its corner open, the walls to either side stopping 4 cm short of it, 5.7 cm apart across
    // it: too narrow for the can, but not for it worn down by half the spots' diagonal, 4.6 cm
    // across, which leaves that way between two spots that each touch a wall. It is taken as a
    // way out.
    let mut corner = Surface::default();
    level(
        Vec2::new(-0.1, -0.08),
        Vec2::new(0.1, 0.08),
        0.01,
        true,
        &mut corner,
    );
    let mut wall =
        |at: Vec2, half: Vec2| block(at.extend(0.05), half.extend(0.04), true, &mut corner);
    wall(Vec2::new(-0.105, 0.0), Vec2::new(0.005, 0.09));
    wall(Vec2::new(0.0, -0.085), Vec2::new(0.1, 0.005));
    wall(Vec2::new(0.105, -0.025), Vec2::new(0.005, 0.065));
    wall(Vec2::new(-0.025, 0.085), Vec2::new(0.085, 0.005));
    assert!(falls(&can, &corner, &o).is_empty());
    // Over a floor seen but for a hole 12 cm square in its middle, it is let go over the hole
    // too, as deep as it lands anywhere in the pen.
    let mut holed = Surface::default();
    for (lower, upper) in [
        (Vec2::new(-0.1, -0.08), Vec2::new(0.1, -0.06)),
        (Vec2::new(-0.1, 0.06), Vec2::new(0.1, 0.08)),
        (Vec2::new(-0.1, -0.06), Vec2::new(-0.06, 0.06)),
        (Vec2::new(0.06, -0.06), Vec2::new(0.1, 0.06)),
    ] {
        level(lower, upper, 0.01, true, &mut holed);
    }
    let mut walls = pen(0.08, 0.0);
    walls.points.retain(|p| p.z > 0.0105);
    let kept = walls.points.len();
    walls.normals.truncate(kept);
    walls.object.truncate(kept);
    holed.join(walls);
    let over_the_hole: Vec<Fall> = falls(&can, &holed, &o)
        .into_iter()
        .filter(|f| f.pose.position.truncate().abs().max_element() < 0.02)
        .collect();
    assert!(!over_the_hole.is_empty());
    for f in &over_the_hole {
        assert!((f.sunk - 0.08).abs() < 1e-4, "{f:?}");
    }
    // A plate 6 cm square and 6 mm thick lying flat, and a ball 2.4 cm across, slide out
    // through a gap a little wider than they are; in the closed pen they are held in. Worn down
    // by half the spots' diagonal, they keep their thickness.
    let plate = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.003));
    let ball = Obstacle::sphere(Vec3::ZERO, 0.012);
    let once = RestOptions { turns: 1, ..o };
    for (what, object, gap) in [("plate", &plate, 0.062), ("ball", &ball, 0.025)] {
        assert!(!falls(object, &pen(0.08, 0.0), &once).is_empty(), "{what}");
        let out = falls(object, &pen(0.08, gap), &once);
        assert!(out.is_empty(), "{what}: {}", out.len());
    }
    // A ball 1.6 cm across and a pencil standing have none: worn down that far, less than a
    // cell of the support is left of them, which could slip between the points seen of a wall.
    let pencil = Obstacle::cylinder(Vec3::ZERO, 0.004, 0.08);
    let small = Obstacle::sphere(Vec3::ZERO, 0.008);
    for thin in [&pencil, &small] {
        assert!(falls(thin, &pen(0.12, 0.0), &o).is_empty(), "{thin:?}");
    }
    // A box 4 by 6 cm comes at every turn, turned its own way.
    let boxed = falls(&a_box(), &pen(0.08, 0.0), &o);
    let mut turns: Vec<Quat> = boxed.iter().map(|f| f.pose.rotation).collect();
    turns.dedup();
    assert_eq!(turns.len(), 8);
}

#[test]
fn what_falls_onto_something_already_in_a_pen_may_not_stay() {
    // A block 6 by 6 cm and 7 cm tall stands in the pen, not the support. Over it the can lands
    // on it, its middle a centimeter under the walls' tops: let go there, it does not stay in.
    let can = Obstacle::cylinder(Vec3::ZERO, 0.03, 0.04);
    let o = RestOptions::default();
    let mut s = pen(0.08, 0.0);
    block(
        Vec3::new(-0.06, 0.0, 0.045),
        Vec3::new(0.03, 0.03, 0.035),
        false,
        &mut s,
    );
    let found = falls(&can, &s, &o);
    assert!(!found.is_empty());
    for f in &found {
        // Its footprint grown by the clearance off the block.
        let off = f.pose.position.truncate() - Vec2::new(-0.06, 0.0);
        let near = off.abs() - Vec2::splat(0.03);
        let gap = near.max(Vec2::ZERO).length();
        assert!(
            gap >= 0.03 + o.clearance - 1e-4,
            "{gap} m from the block: {f:?}"
        );
    }
}

#[test]
fn falls_do_not_depend_on_the_threads_the_batch_or_the_device() {
    let robot = common::panda().unwrap();
    let o = RestOptions::default();
    let can = Obstacle::cylinder(Vec3::ZERO, 0.03, 0.04);
    let surfaces = [pen(0.08, 0.0), basket(true, Some(true)), table()];
    let problems = [
        RestProblem::new(can.clone(), 0),
        RestProblem::new(a_box(), 1),
        RestProblem::new(can, 2),
        RestProblem {
            rotation: Quat::from_rotation_z(0.3),
            ..RestProblem::new(a_box(), 0)
        },
    ];
    let one = Device::cpu_threads(&robot, 1).unwrap();
    let alone: Vec<Vec<Fall>> = problems
        .iter()
        .map(|p| {
            let problem = RestProblem {
                surface: 0,
                ..p.clone()
            };
            let s = std::slice::from_ref(&surfaces[p.surface]);
            falls_into(&one, &[problem], s, &o).unwrap().remove(0)
        })
        .collect();
    assert!(!alone[0].is_empty() && !alone[1].is_empty() && alone[2].is_empty());
    assert!(!alone[3].is_empty());
    let mut devices = vec![one, Device::cpu(&robot).unwrap()];
    match Device::gpu(&robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    for device in &devices {
        let together = falls_into(device, &problems, &surfaces, &o).unwrap();
        assert_eq!(together, alone, "{}", device.name());
    }
    // Its input is checked as `rests_on` checks it.
    let elsewhere = RestProblem::new(a_box(), 3);
    let refused = falls_into(&devices[0], &[elsewhere], &surfaces, &o);
    assert!(matches!(refused, Err(Error::Input(why)) if why.starts_with("problem 0: surface 3")));
}

/// A wine bottle stood upright, 4.3 cm across and 15.8 cm tall.
fn bottle() -> Obstacle {
    Obstacle::cylinder(Vec3::ZERO, 0.0215, 0.079)
}

/// Whether `r`'s middle stands over a spot of the grid `o` spaces, seen from above.
fn on_the_grid(r: &Rest, o: &RestOptions) -> bool {
    let cells = r.pose.position.truncate() / o.spacing - 0.5;
    (cells - cells.round()).abs().max_element() < 1e-3
}

/// The centimeters along x, from the rack's own frame turned by `turn`, in which a rest's middle
/// stands.
fn centimeters(found: &[Rest], turn: Quat) -> HashSet<i32> {
    let along = |r: &Rest| (turn.inverse() * r.pose.position).x;
    found
        .iter()
        .map(|r| (along(r) * 100.0).floor() as i32)
        .collect()
}

#[test]
fn an_upright_bottle_rests_against_the_lip_of_a_sloping_rack() {
    // Stood upright on a slope of 30 degrees with a lip 1 cm high and 1 cm wide at its foot, the
    // bottle rests where its rim behind lies on the slope and its bottom on the lip's top, its
    // middle far enough in from the lip's edge to tip 5 degrees: a strip 2 mm deep, between the
    // rows of the grid a centimeter apart. The spots set closer about the grid's near misses
    // find it along the lip's length, and only there.
    let o = RestOptions::default();
    let rack = slope(30.0, true);
    let found = rests(&bottle(), &rack, &o);
    let foot = 0.1 * 30f32.to_radians().cos();
    let mut places = HashSet::new();
    for r in &found {
        assert!(holds(&bottle(), r, &rack, &o), "{r:?}");
        assert!(r.tipping >= o.tipping, "{r:?}");
        assert!(up(r).z > 0.9999, "{r:?}");
        assert!(!on_the_grid(r, &o), "{r:?}");
        // On the lattice the support's cells are centred on, each place once a turn.
        let cells = r.pose.position.truncate() / o.touching;
        assert!((cells - cells.round()).abs().max_element() < 1e-3, "{r:?}");
        let place = (
            cells.round().as_ivec2().to_array(),
            r.pose.rotation.to_array().map(f32::to_bits),
        );
        assert!(places.insert(place), "twice: {r:?}");
        let over = r.pose.position.y - foot;
        assert!((0.0..0.004).contains(&over), "over the lip's top: {r:?}");
    }
    let along = centimeters(&found, Quat::IDENTITY);
    assert!(
        along.len() >= 18,
        "{} of the lip's 20: {along:?}",
        along.len()
    );
    assert!(rests(&bottle(), &slope(30.0, false), &o).is_empty());
}

#[test]
fn a_lip_the_grid_meets_at_a_spot_is_found_along_its_length() {
    // The rack turned half a degree about the vertical: the grid's spots meet the strip where it
    // can stand in two places, and the spots set closer find the rest of it.
    let o = RestOptions::default();
    let turn = Quat::from_rotation_z(0.5f32.to_radians());
    let mut rack = slope(30.0, true);
    for p in &mut rack.points {
        *p = turn * *p;
    }
    for n in &mut rack.normals {
        *n = turn * *n;
    }
    let found = rests(&bottle(), &rack, &o);
    let on_grid = found.iter().filter(|r| on_the_grid(r, &o)).count();
    assert!(
        on_grid > 0 && on_grid < found.len() / 10,
        "{on_grid} of {}",
        found.len()
    );
    for r in &found {
        assert!(holds(&bottle(), r, &rack, &o), "{r:?}");
    }
    let along = centimeters(&found, turn);
    assert!(
        along.len() >= 18,
        "{} of the lip's 20: {along:?}",
        along.len()
    );
}

/// [`table`] with two blocks on it, the support's where `flagged`.
fn cluttered(flagged: bool) -> Surface {
    let mut s = table();
    let blocks = [
        (Vec3::new(0.03, 0.0, 0.04), Vec3::new(0.03, 0.03, 0.04)),
        (Vec3::new(-0.08, -0.07, 0.02), Vec3::new(0.02, 0.04, 0.02)),
    ];
    for (center, half) in blocks {
        block(center, half, flagged, &mut s);
    }
    s
}

#[test]
fn on_a_table_every_rest_stands_on_a_spot_of_the_grid() {
    // Where a box or a can rests at a spot, or tips at the table's rim beside one that does, as a
    // box 20 cm tall, nothing is set closer: as many rests as the grid alone found, each on a spot
    // of it.
    let o = RestOptions::default();
    let can = Obstacle::cylinder(Vec3::ZERO, 0.03, 0.05);
    let tall = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.02, 0.1));
    let scenes = [
        (
            table(),
            vec![(a_box(), 7200), (can.clone(), 7200), (tall, 6272)],
        ),
        (cluttered(false), vec![(a_box(), 4348), (can.clone(), 4368)]),
        (cluttered(true), vec![(a_box(), 5596), (can, 5536)]),
    ];
    for (k, (s, objects)) in scenes.iter().enumerate() {
        for (object, count) in objects {
            let found = rests(object, s, &o);
            assert_eq!(found.len(), *count, "scene {k}: {object:?}");
            assert!(found.iter().all(|r| on_the_grid(r, &o)), "scene {k}");
        }
    }
}

/// Why `problem` has no rest on `surface`, through `rests_on` and `rest_failures`.
fn failure(problem: &RestProblem, surface: &Surface, o: &RestOptions) -> Option<RestFailure> {
    let (device, problems) = (device(), std::slice::from_ref(problem));
    let surfaces = std::slice::from_ref(surface);
    let found = rests_on(&device, problems, surfaces, o).unwrap();
    rest_failures(&device, problems, surfaces, &found, o)
        .unwrap()
        .remove(0)
}

/// A post `2 half` square and 10 cm tall, its top at 0, the support.
fn post(half: f32) -> Surface {
    let mut s = Surface::default();
    let (center, half) = (Vec3::new(0.0, 0.0, -0.05), Vec3::new(half, half, 0.05));
    block(center, half, true, &mut s);
    s
}

/// A table 6 cm square with walls not the support 2.9 cm to either side of its middle along x:
/// a box 4 cm wide rests on it nowhere clear of them.
fn between_walls() -> (Surface, Vec<Vec3>) {
    let mut s = Surface::default();
    level(Vec2::splat(-0.03), Vec2::splat(0.03), 0.0, true, &mut s);
    let before = s.points.len();
    for x in [-0.034, 0.034] {
        let (center, half) = (Vec3::new(x, 0.0, 0.025), Vec3::new(0.005, 0.03, 0.025));
        block(center, half, false, &mut s);
    }
    let walls = s.points[before..].to_vec();
    (s, walls)
}

#[test]
fn a_problem_without_rests_says_how_near_the_search_came() {
    let o = RestOptions::default();
    let tall = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.02, 0.1));
    // Nothing of the support anywhere: none of it flagged, or seen only from below.
    let mut none = table();
    none.object.fill(false);
    let unsupported = Some(RestFailure::Unsupported);
    assert_eq!(
        failure(&RestProblem::new(a_box(), 0), &none, &o),
        unsupported
    );
    let mut below = table();
    below.normals.fill(-Vec3::Z);
    assert_eq!(
        failure(&RestProblem::new(a_box(), 0), &below, &o),
        unsupported
    );

    // On a slope steeper than the friction, laid flat on it: held nowhere, pushed along the
    // slope's normal, at the spot nearest the slope's middle, nothing in its way.
    let tile = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.01));
    let laid = RestProblem {
        rotation: lying_on(30.0),
        ..RestProblem::new(tile.clone(), 0)
    };
    let (sin, cos) = 30f32.to_radians().sin_cos();
    let normal = Vec3::new(0.0, sin, cos);
    let Some(RestFailure::Unheld {
        pose,
        support,
        push,
        blocked,
    }) = failure(&laid, &slope(30.0, false), &o)
    else {
        panic!("not unheld");
    };
    assert!(push.angle_between(normal) < 1f32.to_radians(), "{push}");
    assert!(blocked.is_none());
    assert!(pose.position.truncate().length() < 0.008, "{pose:?}");
    assert!(
        (pose.rotation * Vec3::Z).angle_between(normal) < 1e-3,
        "{pose:?}"
    );
    assert!(support.len() >= 3);
    for c in &support {
        assert!(c.dot(normal).abs() < 1e-3, "off the slope: {c}");
        assert!(
            tile.placed(pose).distance(*c).value.abs() < o.touching,
            "{c}"
        );
    }
    // An upright bottle there, the lip left unflagged and the place asked for at the lip: held
    // nowhere, and the lip's points in its way.
    let mut short = slope(30.0, true);
    let slope_points = slope(30.0, false).points.len();
    short.object[slope_points..].fill(false);
    let foot = 0.1 * cos;
    let at_the_lip = RestProblem {
        near: Some(Vec3::new(0.0, foot, 0.0)),
        ..RestProblem::new(bottle(), 0)
    };
    let Some(RestFailure::Unheld {
        pose,
        blocked: Some(by),
        ..
    }) = failure(&at_the_lip, &short, &o)
    else {
        panic!("not unheld with the lip in the way");
    };
    assert!(!by.support && by.point.y >= foot - 1e-3, "{by:?}");
    assert!(
        pose.position.truncate().distance(Vec2::new(0.0, foot)) < 0.012,
        "{pose:?}"
    );

    // A box 20 cm tall on a post 1.5 cm across: held upright and tipping, as far as its middle
    // is in from the post's edge allows, the steadiest of all, where the spots about the post's
    // middle find it (a flat top pushes one way: none are set closer there); its support the
    // post's top.
    let on_the_post = RestProblem {
        near: Some(Vec3::ZERO),
        ..RestProblem::new(tall.clone(), 0)
    };
    let Some(RestFailure::Tips {
        pose,
        support,
        tipping,
        blocked,
    }) = failure(&on_the_post, &post(0.0075), &o)
    else {
        panic!("not tipping");
    };
    let inset = 0.0075 - pose.position.truncate().abs().max_element();
    assert!(
        (tipping - inset.atan2(0.1)).abs() < 0.5f32.to_radians(),
        "{tipping} at {pose:?}"
    );
    assert!(
        tipping < o.tipping && inset > 0.007,
        "{tipping} at {pose:?}"
    );
    assert!(blocked.is_none());
    assert!(support.len() >= 3);
    for c in &support {
        assert!(c.z.abs() < 1e-6 && c.truncate().abs().max_element() <= 0.0075 + 1e-6);
    }

    // The bottle against the rack's lip, its rim on the slope and its bottom on the lip, asked to
    // tip 10 degrees: it tips, as far as the steadiest of its rests asked for 5 does, found as a
    // rest's is.
    let rack = slope(30.0, true);
    let steadiest = rests(&bottle(), &rack, &o)
        .iter()
        .map(|r| r.tipping)
        .fold(0.0, f32::max);
    let steadier = RestOptions {
        tipping: 10f32.to_radians(),
        ..o
    };
    assert!(
        steadiest > o.tipping && steadiest < steadier.tipping,
        "{steadiest}"
    );
    let Some(RestFailure::Tips { tipping, .. }) =
        failure(&RestProblem::new(bottle(), 0), &rack, &steadier)
    else {
        panic!("not tipping against the lip");
    };
    assert!(
        (tipping - steadiest).abs() < 0.5f32.to_radians(),
        "{tipping}, {steadiest}"
    );

    // A table between walls that are not the support: it would rest, but for a wall within the
    // clearance; and the box 20 cm tall, which also tips at the table's edges, comes to that too.
    let (walled, walls) = between_walls();
    for object in [a_box(), tall] {
        let Some(RestFailure::Blocked { rest, by }) =
            failure(&RestProblem::new(object.clone(), 0), &walled, &o)
        else {
            panic!("not blocked: {object:?}");
        };
        assert!(walls.contains(&by.point) && !by.support, "{by:?}");
        assert!(
            distance(&object, &rest, by.point) < o.clearance,
            "{by:?} {rest:?}"
        );
        assert!(
            rest.tipping >= o.tipping && rest.support.len() >= 3,
            "{rest:?}"
        );
        assert!((rest.pose.position.z - object_half_height(&object)).abs() < 1e-4);
    }
}

#[test]
fn why_none_is_the_same_alone_in_a_batch_and_on_any_device_and_says_nothing_where_one_rests() {
    let robot = common::panda().unwrap();
    let o = RestOptions::default();
    let tall = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.02, 0.1));
    let (walled, _) = between_walls();
    let surfaces = [table(), slope(30.0, false), post(0.0075), walled];
    let problems = [
        RestProblem::new(a_box(), 0),
        RestProblem {
            rotation: lying_on(30.0),
            ..RestProblem::new(a_box(), 1)
        },
        RestProblem::new(tall.clone(), 2),
        RestProblem::new(a_box(), 3),
        RestProblem::new(tall, 3),
    ];
    let one = Device::cpu_threads(&robot, 1).unwrap();
    let found = rests_on(&one, &problems, &surfaces, &o).unwrap();
    let alone: Vec<Option<RestFailure>> = (0..problems.len())
        .map(|k| {
            let s = std::slice::from_ref(&surfaces[problems[k].surface]);
            let p = RestProblem {
                surface: 0,
                ..problems[k].clone()
            };
            let found = rests_on(&one, std::slice::from_ref(&p), s, &o).unwrap();
            rest_failures(&one, &[p], s, &found, &o).unwrap().remove(0)
        })
        .collect();
    assert!(alone[0].is_none());
    assert!(matches!(alone[1], Some(RestFailure::Unheld { .. })));
    assert!(matches!(alone[2], Some(RestFailure::Tips { .. })));
    assert!(matches!(alone[3], Some(RestFailure::Blocked { .. })));
    assert!(matches!(alone[4], Some(RestFailure::Blocked { .. })));
    let mut devices = vec![one, Device::cpu(&robot).unwrap()];
    match Device::gpu(&robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    for device in &devices {
        let together = rest_failures(device, &problems, &surfaces, &found, &o).unwrap();
        assert_eq!(together, alone, "{}", device.name());
    }
    // Given no rests for a problem that has them, it says none either.
    let device = &devices[0];
    let unasked = rest_failures(device, &problems[..1], &surfaces, &[vec![]], &o).unwrap();
    assert_eq!(unasked, [None]);
    // Its input is checked as `rests_on`'s is, and the rests must be one list a problem.
    let short = rest_failures(device, &problems, &surfaces, &found[1..], &o);
    assert!(matches!(short, Err(Error::Input(why)) if why.contains("4 lists of rests for 5")));
    let elsewhere = RestProblem::new(a_box(), 4);
    let refused = rest_failures(device, &[elsewhere], &surfaces, &[vec![]], &o);
    assert!(matches!(refused, Err(Error::Input(why)) if why.starts_with("problem 0: surface 4")));
}

/// Half the height of an upright box.
fn object_half_height(object: &Obstacle) -> f32 {
    match *object {
        Obstacle::Cuboid { half_extents, .. } => half_extents.z,
        _ => panic!("a box, got {object:?}"),
    }
}
