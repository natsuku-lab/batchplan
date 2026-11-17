//! Grasps of a surface through `grasps_on`: surfaces sampled from shapes whose grasps are known
//! give those grasps and no others, each holding by friction on the shape itself; a wall is
//! grasped across its thickness; what was not seen from both sides, what does not face within the
//! friction cone and what has something else between the fingers is left out.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::f32::consts::TAU;

use batchplan::glam::{Mat3, Quat, Vec3};
use batchplan::{
    DepthImage, Device, Error, Grasp, GraspOptions, Intrinsics, Obstacle, Pose, Surface, grasps_on,
};

/// Points 2 mm apart: a fingertip's patch of 5 mm holds a few dozen.
const SPACING: f32 = 0.002;

fn device() -> Device {
    Device::cpu(&common::panda().unwrap()).unwrap()
}

/// A flat patch through `middle` with normal `normal`, `half` wide along each of `across`.
fn face(middle: Vec3, normal: Vec3, across: [Vec3; 2], half: [f32; 2], s: &mut Surface) {
    let steps = |h: f32| (2.0 * h / SPACING).round() as i32;
    for i in 0..=steps(half[0]) {
        for j in 0..=steps(half[1]) {
            let (u, v) = (i as f32 * SPACING - half[0], j as f32 * SPACING - half[1]);
            s.points.push(middle + across[0] * u + across[1] * v);
            s.normals.push(normal);
        }
    }
}

/// A box's faces, without those along the axes `missing` names.
fn box_surface(half: Vec3, missing: &[(usize, f32)]) -> Surface {
    let mut s = Surface::default();
    for axis in 0..3 {
        for sign in [1.0, -1.0] {
            if missing.contains(&(axis, sign)) {
                continue;
            }
            let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
            let normal = Vec3::AXES[axis] * sign;
            face(
                normal * half[axis],
                normal,
                [Vec3::AXES[a], Vec3::AXES[b]],
                [half[a], half[b]],
                &mut s,
            );
        }
    }
    s
}

/// An upright tube open at both ends: its outer and inner walls and its two rims.
fn tube_surface(outer: f32, inner: f32, half_height: f32) -> Surface {
    let mut s = Surface::default();
    let around = |r: f32| (TAU * r / SPACING).round() as usize;
    let up = (2.0 * half_height / SPACING).round() as usize;
    for (r, out) in [(outer, 1.0), (inner, -1.0)] {
        for k in 0..around(r) {
            let radial = Vec3::new((TAU * k as f32 / around(r) as f32).cos(), 0.0, 0.0)
                + Vec3::Y * (TAU * k as f32 / around(r) as f32).sin();
            for j in 0..=up {
                s.points
                    .push(radial * r + Vec3::Z * (j as f32 * SPACING - half_height));
                s.normals.push(radial * out);
            }
        }
    }
    let across = ((outer - inner) / SPACING).round() as usize;
    for end in [1.0, -1.0] {
        for i in 0..=across {
            let r = inner + (outer - inner) * i as f32 / across as f32;
            for k in 0..around(r) {
                let angle = TAU * k as f32 / around(r) as f32;
                s.points.push(Vec3::new(
                    r * angle.cos(),
                    r * angle.sin(),
                    end * half_height,
                ));
                s.normals.push(Vec3::Z * end);
            }
        }
    }
    s
}

fn grasps(surface: &Surface, o: &GraspOptions) -> Vec<Grasp> {
    grasps_on(&device(), 0, std::slice::from_ref(surface), o)
        .unwrap()
        .remove(0)
}

/// The grasp's closing line and the two points it closes on.
fn closes_on(g: &Grasp) -> (Vec3, [Vec3; 2]) {
    let line = g.pose.rotation * Vec3::Y;
    // The frame's y axis runs from the first to the second, through the frame or ahead of it.
    let between = g.contacts[1] - g.contacts[0];
    assert!(between.normalize().dot(line) > 0.9999, "{g:?}");
    let ahead = (g.contacts[0] + g.contacts[1]) / 2.0 - g.pose.position;
    let toward = g.pose.rotation * Vec3::Z;
    assert!(
        ahead.cross(toward).length() < 1e-5 && ahead.dot(toward) > -1e-5,
        "{g:?}"
    );
    (line, g.contacts)
}

#[test]
fn a_box_is_grasped_across_each_side_that_fits_and_no_other() {
    let half = Vec3::new(0.02, 0.03, 0.06);
    let o = GraspOptions::default();
    let found = grasps(&box_surface(half, &[]), &o);
    assert!(2.0 * half.z > o.opening && 2.0 * half.y < o.opening);
    let mut across = [0; 3];
    for g in &found {
        let (line, ends) = closes_on(g);
        let axis = line.abs().max_position();
        across[axis] += 1;
        // Along one axis, on the two faces it crosses, as wide as the box is there.
        assert!(line.abs()[axis] > 0.9999, "{g:?}");
        assert!((g.width() - 2.0 * half[axis]).abs() < 1e-4, "{g:?}");
        for end in ends {
            assert!((end[axis].abs() - half[axis]).abs() < 1e-4, "{g:?}");
            assert!(end.abs().cmple(half + 1e-4).all(), "{g:?}");
        }
        // The hand's frame is a rotation whose z axis is square to the line.
        let toward = g.pose.rotation * Vec3::Z;
        assert!(toward.dot(line).abs() < 1e-4 && g.pose.rotation.is_normalized());
    }
    assert!(across[0] > 0 && across[1] > 0, "{across:?}");
    assert_eq!(across[2], 0, "the box is too long to grasp along z");
    // From above among them, near its top alone: lower, the palm would be in the box, between
    // the fingers though it is.
    let above = |g: &&Grasp| (g.pose.rotation * Vec3::Z).dot(Vec3::NEG_Z) > 0.9999;
    let from_above: Vec<&Grasp> = found.iter().filter(above).collect();
    assert!(!from_above.is_empty());
    for g in &from_above {
        assert!(
            g.pose.position.z > half.z - 0.05,
            "the palm is in the box: {g:?}"
        );
    }
    assert!(
        found.iter().any(|g| g.pose.position.z < 0.0),
        "from its sides, lower too"
    );
}

#[test]
fn patches_the_hand_is_clear_around_are_grasped_from_every_turn_both_ways_round() {
    // Two plates 1 cm square with nothing around them: the hand is clear from every side.
    let o = GraspOptions::default();
    let found = grasps(&plates_of(0.04, 0.005), &o);
    assert!(!found.is_empty());
    assert_eq!(found.len() % (2 * o.turns), 0);
    for both in found.chunks(2) {
        let [a, b] = [both[0].pose, both[1].pose];
        assert_eq!(a.position, b.position);
        assert!((a.rotation * Vec3::Y).dot(b.rotation * Vec3::Y) < -0.9999);
        assert!((a.rotation * Vec3::Z).dot(b.rotation * Vec3::Z) > 0.9999);
        assert_eq!(both[0].contacts, [both[1].contacts[1], both[1].contacts[0]]);
    }
    // The first from above, the next a turn on.
    let toward = |k: usize| found[k].pose.rotation * Vec3::Z;
    assert!(toward(0).dot(Vec3::NEG_Z) > 0.9999, "{}", toward(0));
    let turned = toward(0).dot(toward(2));
    assert!(
        (turned - (TAU / o.turns as f32).cos()).abs() < 1e-4,
        "{turned}"
    );
}

#[test]
fn a_wall_is_grasped_across_its_thickness() {
    // A bowl's wall: 11 cm across, 8 mm thick, 5 cm tall, too wide for the hand to span.
    let (outer, inner, half_height) = (0.055, 0.047, 0.025);
    let o = GraspOptions::default();
    assert!(2.0 * outer > o.opening);
    let found = grasps(&tube_surface(outer, inner, half_height), &o);
    let mut sectors = [false; 8];
    let mut through = 0;
    for g in &found {
        let (line, ends) = closes_on(g);
        if line.z.abs() > 0.9999 {
            // Rim to rim through the wall.
            assert!((g.width() - 2.0 * half_height).abs() < 1e-3, "{g:?}");
            through += 1;
            continue;
        }
        // Across the wall: along the radius where it stands, one end on each side of it.
        let radial = g.pose.position.with_z(0.0).normalize();
        assert!(line.dot(radial).abs() > 0.995, "{g:?}");
        assert!((g.width() - (outer - inner)).abs() < 1e-3, "{g:?}");
        let mut radii = ends.map(|e| e.with_z(0.0).length());
        radii.sort_by(f32::total_cmp);
        assert!(
            (radii[0] - inner).abs() < 1e-3 && (radii[1] - outer).abs() < 1e-3,
            "{g:?}"
        );
        let angle = g.pose.position.y.atan2(g.pose.position.x).rem_euclid(TAU);
        sectors[(angle / TAU * 8.0) as usize % 8] = true;
    }
    assert!(
        sectors.iter().all(|&s| s),
        "all around the rim: {sectors:?}"
    );
    assert!(through > 0, "and from rim to rim, which fits too");
    // None spans the bowl.
    assert!(found.iter().all(|g| g.width() < 0.06));
}

#[test]
fn a_cup_is_grasped_around_it_and_across_its_wall_by_a_hand_that_fits_inside() {
    let (outer, inner, half_height) = (0.03, 0.026, 0.06);
    let cup = tube_surface(outer, inner, half_height);
    let o = GraspOptions::default();
    let found = grasps(&cup, &o);
    let width = |found: &[Grasp], w: f32| found.iter().any(|g| (g.width() - w).abs() < 1e-3);
    assert!(width(&found, 2.0 * outer), "around it");
    // The far wall's inside faces the line's start: no finger presses on it from there.
    assert!(
        !width(&found, outer + inner),
        "not onto the inside of its far wall"
    );
    // Open 8 cm, the finger inside would stand in the far wall, 5.2 cm away.
    assert!(
        !width(&found, outer - inner),
        "not across its wall, open wide"
    );
    // Open 3 cm, it fits inside.
    let robot = common::panda().unwrap();
    let fingers = ["panda_finger_joint1", "panda_finger_joint2"].map(|j| (j.to_string(), 0.015));
    let narrow = robot.relock(&HashMap::from(fingers)).unwrap();
    let device = Device::cpu(&narrow).unwrap();
    let o = GraspOptions { opening: 0.03, ..o };
    let found = grasps_on(&device, 0, &[cup], &o).unwrap().remove(0);
    assert!(width(&found, outer - inner), "across its wall, open 3 cm");
    assert!(!width(&found, 2.0 * outer));
}

#[test]
fn a_surface_with_nothing_seen_behind_it_gives_no_grasp() {
    let half = Vec3::new(0.02, 0.03, 0.02);
    let o = GraspOptions::default();
    // The box seen from +x and from above and below: its sides along y unseen.
    let seen = box_surface(half, &[(0, -1.0), (1, 1.0), (1, -1.0)]);
    let found = grasps(&seen, &o);
    // Top to bottom, both seen, and nothing along x or y.
    assert!(!found.is_empty());
    for g in &found {
        let (line, _) = closes_on(g);
        assert!(line.z.abs() > 0.9999, "{g:?}");
    }
    let all = grasps(&box_surface(half, &[]), &o);
    let along = |axis: usize| {
        all.iter()
            .filter(|g| closes_on(g).0.abs()[axis] > 0.9999)
            .count()
    };
    assert!(along(0) > 0 && along(1) > 0, "seen whole, it has them");
}

/// Two plates 4 cm square, `apart`, facing away from each other.
fn plates(apart: f32) -> Surface {
    plates_of(apart, 0.02)
}

/// Two square plates of half size `half`, `apart`, facing away from each other.
fn plates_of(apart: f32, half: f32) -> Surface {
    let mut s = Surface::default();
    for side in [1.0, -1.0] {
        let normal = Vec3::X * side;
        face(
            normal * apart / 2.0,
            normal,
            [Vec3::Y, Vec3::Z],
            [half, half],
            &mut s,
        );
    }
    s
}

#[test]
fn sides_farther_apart_than_the_fingers_open_or_too_near_to_be_two_are_left_out() {
    let o = GraspOptions::default();
    assert!(!grasps(&plates(o.opening - 0.002), &o).is_empty());
    // A little past the opening: still within a fingertip of where the search ends.
    assert!(grasps(&plates(o.opening + 0.002), &o).is_empty());
    // A sheet 1 mm thick is one surface seen with noise, not two sides to close on.
    assert!(!grasps(&plates(0.004), &o).is_empty());
    assert!(grasps(&plates(0.001), &o).is_empty());
}

/// An upright tube whose section is an ellipse of half axes `a` along x and `b` along y: its
/// outside alone.
fn oval(a: f32, b: f32, half_height: f32) -> Surface {
    let mut s = Surface::default();
    let around = (TAU * a / SPACING).round() as usize * 2;
    let up = (2.0 * half_height / SPACING).round() as usize;
    for k in 0..around {
        let angle = TAU * k as f32 / around as f32;
        let at = Vec3::new(a * angle.cos(), b * angle.sin(), 0.0);
        let normal = Vec3::new(at.x / (a * a), at.y / (b * b), 0.0).normalize();
        for j in 0..=up {
            s.points
                .push(at + Vec3::Z * (j as f32 * SPACING - half_height));
            s.normals.push(normal);
        }
    }
    s
}

#[test]
fn every_grasp_of_a_curved_side_holds_by_friction_on_the_shape_itself() {
    // Around an oval, a line in along one side's normal meets the other side askew nearly
    // everywhere: only lines inside both cones hold.
    let (a, b) = (0.035, 0.02);
    let o = GraspOptions {
        friction: 0.3,
        ..Default::default()
    };
    let found = grasps(&oval(a, b, 0.02), &o);
    assert!(!found.is_empty());
    let cone = o.friction.atan();
    let mut askew: f32 = 0.0;
    for g in &found {
        let (line, ends) = closes_on(g);
        for (end, out) in ends.into_iter().zip([-line, line]) {
            // On the oval, and within the cone of its normal there.
            let on = (end.x / a).powi(2) + (end.y / b).powi(2);
            assert!((on - 1.0).abs() < 0.05, "{g:?}");
            let normal = Vec3::new(end.x / (a * a), end.y / (b * b), 0.0).normalize();
            let off = normal.dot(out).clamp(-1.0, 1.0).acos();
            assert!(off < cone + 0.03, "{off} rad off a cone of {cone}: {g:?}");
            askew = askew.max(off);
        }
    }
    assert!(askew > 0.1, "some hold askew, within the cone: {askew}");
}

/// Two faces leaning toward each other by `lean` each, 4 cm apart at their middles.
fn wedge(lean: f32) -> Surface {
    let mut s = Surface::default();
    for side in [1.0, -1.0] {
        let normal = Vec3::new(side * lean.cos(), 0.0, lean.sin());
        let up = Vec3::new(-side * lean.sin(), 0.0, lean.cos());
        face(
            Vec3::X * side * 0.02,
            normal,
            [Vec3::Y, up],
            [0.02, 0.02],
            &mut s,
        );
    }
    s
}

#[test]
fn faces_hold_where_the_line_between_them_is_inside_both_friction_cones() {
    // Each face leans 20 degrees: the level line between them is 20 degrees off each normal.
    let lean = 20f32.to_radians();
    let holds = |friction: f32| {
        let o = GraspOptions {
            friction,
            ..Default::default()
        };
        grasps(&wedge(lean), &o)
    };
    // A cone of atan(0.5), 26.6 degrees, holds it; one of atan(0.3), 16.7 degrees, does not.
    let found = holds(0.5);
    assert!(!found.is_empty());
    for g in &found {
        let (line, ends) = closes_on(g);
        // On the faces themselves, along the line that splits their normals: level.
        assert!(line.x.abs() > 0.999, "{g:?}");
        for end in ends {
            let off = (end.x.abs() - 0.02) * lean.cos() + end.z * lean.sin();
            assert!(off.abs() < 1e-3, "{g:?}");
        }
    }
    assert!(holds(0.3).is_empty());
    // Faces that lean past the cone hold at no friction a fingertip has here.
    assert!(grasps(&wedge(35f32.to_radians()), &GraspOptions::default()).is_empty());
}

#[test]
fn something_else_between_the_fingers_leaves_a_grasp_out() {
    // Two plates 4 cm apart, and between them a block that is not the object's.
    let plates = |s: &mut Surface| {
        for side in [1.0, -1.0] {
            face(
                Vec3::X * side * 0.02,
                Vec3::X * side,
                [Vec3::Y, Vec3::Z],
                [0.02, 0.02],
                s,
            );
        }
    };
    let mut alone = Surface::default();
    plates(&mut alone);
    let o = GraspOptions::default();
    let free = grasps(&alone, &o);
    assert!(!free.is_empty());
    let mut blocked = alone.clone();
    blocked.object = vec![true; blocked.points.len()];
    // A sheet of something else between the plates, 1 cm over their middles.
    let before = blocked.points.len();
    face(
        Vec3::new(0.0, 0.0, 0.01),
        Vec3::Z,
        [Vec3::X, Vec3::Y],
        [0.01, 0.02],
        &mut blocked,
    );
    blocked.object.resize(blocked.points.len(), false);
    assert!(blocked.points.len() > before);
    let found = grasps(&blocked, &o);
    assert!(!found.is_empty() && found.len() < free.len());
    for g in &found {
        let off = (g.pose.position.z - 0.01).abs();
        assert!(off > o.contact - SPACING, "the sheet is between: {g:?}");
    }
    let near_the_sheet = |g: &&Grasp| (g.pose.position.z - 0.01).abs() < 0.002;
    assert!(
        free.iter().any(|g| near_the_sheet(&g)),
        "without it, there are grasps there"
    );
    // The same points as the object's are squeezed with it.
    blocked.object = vec![];
    assert_eq!(grasps(&blocked, &o).len(), free.len());
}

/// Marks every point of `s` so far as the object's, for points added after that are not.
fn the_object(s: &mut Surface) {
    s.object = vec![true; s.points.len()];
}

#[test]
fn a_finger_is_kept_out_of_what_stands_beside_the_object() {
    let o = GraspOptions::default();
    let alone = plates_of(0.04, 0.005);
    let free = grasps(&alone, &o).len();
    assert!(free > 0);
    // A wall facing the object, `at` from its middle along the closing line.
    let beside = |at: f32| {
        let mut s = alone.clone();
        the_object(&mut s);
        face(
            Vec3::X * at,
            Vec3::NEG_X,
            [Vec3::Y, Vec3::Z],
            [0.05, 0.05],
            &mut s,
        );
        s.object.resize(s.points.len(), false);
        grasps(&s, &o).len()
    };
    // Open 8 cm, a finger stands 4 cm out: in a wall 4.5 cm out, and clear of one 12 cm out.
    assert_eq!(beside(0.045), 0);
    assert_eq!(beside(0.12), free);
}

#[test]
fn fingers_may_be_in_what_they_close_on_and_nothing_else_of_the_object() {
    // Plates a millimeter nearer than the fingers open: the open fingers' spheres reach them.
    let o = GraspOptions::default();
    let wide = plates(o.opening - 0.001);
    assert!(!grasps(&wide, &o).is_empty());
    // Something else nearly flush with one small plate, just over it and under the palm: a
    // finger coming down past it would be in it, between the plates' planes though it is.
    let small = plates_of(o.opening - 0.001, 0.005);
    let mut flush = small.clone();
    the_object(&mut flush);
    face(
        Vec3::new(o.opening / 2.0 + 0.001, 0.0, 0.0145),
        Vec3::NEG_X,
        [Vec3::Y, Vec3::Z],
        [0.02, 0.0055],
        &mut flush,
    );
    flush.object.resize(flush.points.len(), false);
    let down = |g: &Grasp| (g.pose.rotation * Vec3::Z).dot(Vec3::NEG_Z);
    assert!(grasps(&small, &o).iter().any(|g| down(g) > 0.9));
    assert!(!grasps(&flush, &o).iter().any(|g| down(g) > 0.9));
    // A ledge of the object out past one plate, where that finger stands: no grasp.
    let mut ledged = plates_of(0.04, 0.005);
    face(
        Vec3::X * 0.045,
        Vec3::NEG_X,
        [Vec3::Y, Vec3::Z],
        [0.05, 0.05],
        &mut ledged,
    );
    assert!(grasps(&ledged, &o).is_empty());
}

#[test]
fn the_hand_stays_over_the_floor() {
    let robot = common::panda().unwrap();
    let tip = robot.tip(0);
    let surface = plates_of(0.04, 0.005);
    let anywhere = grasps(&surface, &GraspOptions::default());
    let floor = -0.01;
    let o = GraspOptions {
        floor: Some(floor),
        ..Default::default()
    };
    let found = grasps(&surface, &o);
    assert!(!found.is_empty() && found.len() < anywhere.len());
    let low = |g: &Grasp| {
        let (closing, toward) = (g.pose.rotation * Vec3::Y, g.pose.rotation * Vec3::Z);
        let tips = (g.pose.position + toward * tip).z - closing.z.abs() * o.opening / 2.0;
        g.pose.position.z.min(tips)
    };
    for g in &found {
        assert!(low(g) >= floor - 1e-5, "{g:?}");
    }
    assert!(anywhere.iter().any(|g| low(g) < floor - 1e-3));
    // The frame too, where the fingertips are higher: coming up from below.
    let above_them = GraspOptions {
        floor: Some(0.006),
        ..Default::default()
    };
    assert!(
        anywhere
            .iter()
            .any(|g| low(g) < 0.006 && g.pose.position.z + tip > 0.006)
    );
    assert!(grasps(&surface, &above_them).is_empty());
    // And the lower open finger, where the hand closes upright: 4 cm under the frame.
    let mut upright = Surface::default();
    for side in [1.0, -1.0] {
        face(
            Vec3::Z * side * 0.02,
            Vec3::Z * side,
            [Vec3::X, Vec3::Y],
            [0.005, 0.005],
            &mut upright,
        );
    }
    let over = |floor: f32| GraspOptions {
        floor: Some(floor),
        ..Default::default()
    };
    assert!(!grasps(&upright, &over(-0.05)).is_empty());
    assert!(grasps(&upright, &over(-0.03)).is_empty());
}

#[test]
fn fingertips_stay_off_the_bottom_of_a_bowl() {
    // The bowl of the wall test with a bottom 2 cm under its rim.
    let (outer, inner, half_height) = (0.055, 0.047, 0.025);
    let bottom = half_height - 0.02;
    let mut bowl = tube_surface(outer, inner, half_height);
    let rings = (inner / SPACING) as usize;
    for i in 0..rings {
        let r = i as f32 * SPACING;
        let around = ((TAU * r / SPACING) as usize).max(1);
        for k in 0..around {
            let angle = TAU * k as f32 / around as f32;
            bowl.points
                .push(Vec3::new(r * angle.cos(), r * angle.sin(), bottom));
            bowl.normals.push(Vec3::Z);
        }
    }
    let tip = common::panda().unwrap().tip(0);
    let found = grasps(&bowl, &GraspOptions::default());
    let across = |g: &&Grasp| (g.width() - (outer - inner)).abs() < 1e-3;
    let from_above = |g: &&Grasp| (g.pose.rotation * Vec3::Z).dot(Vec3::NEG_Z) > 0.9999;
    let rim: Vec<&Grasp> = found.iter().filter(across).filter(from_above).collect();
    assert!(!rim.is_empty(), "the rim is grasped from above");
    for g in &rim {
        let fingertips = g.pose.position.z - tip;
        assert!(fingertips > bottom - 1e-4, "in the bottom: {g:?}");
    }
    // Lower on the wall there are patches too, where the fingertips would be in the bottom.
    let lowest = rim
        .iter()
        .map(|g| g.contacts[0].z)
        .fold(f32::INFINITY, f32::min);
    assert!(lowest > bottom && lowest < half_height - 0.002, "{lowest}");
    // And none comes up through the bottom to the wall over it. (Under it the wall hangs free
    // here, and is grasped from below.)
    let through = |g: &Grasp| (g.pose.rotation * Vec3::Z).z > 0.5 && g.contacts[0].z > bottom;
    assert!(!found.iter().filter(across).any(through));
    let under = |g: &Grasp| (g.pose.rotation * Vec3::Z).z > 0.5 && g.contacts[0].z < bottom;
    assert!(found.iter().filter(across).any(under));
    // With less of the fingers asked to overlap the wall, the hand is drawn back up from where
    // its fingertips would be in the bottom: its frame over the line it closes on.
    let shallow = GraspOptions {
        overlap: 0.002,
        ..Default::default()
    };
    let drawn = |g: &Grasp| g.pose.position.z - g.contacts[0].z;
    let found = grasps(&bowl, &shallow);
    let back = found
        .iter()
        .filter(across)
        .filter(from_above)
        .map(drawn)
        .fold(0.0, f32::max);
    assert!(
        back > 0.001 && back <= tip - shallow.overlap + 1e-5,
        "{back}"
    );
    for g in rim {
        assert!(drawn(g) <= tip - 0.01 + 1e-5, "{g:?}");
    }
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
    let right = forward.cross(Vec3::Z).normalize();
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
}

/// z depth per pixel by sphere tracing the scene's distance functions; 0 where a ray hits nothing.
fn render(scene: &[Obstacle], camera: Pose) -> Vec<f32> {
    (0..WIDTH * HEIGHT)
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let ray = Vec3::new((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0);
            let dir = camera.rotation * ray.normalize();
            let mut t = 0.0;
            while t < 2.0 {
                let p = camera.position + dir * t;
                let d = scene
                    .iter()
                    .map(|o| o.distance(p).value)
                    .fold(f32::INFINITY, f32::min);
                if d < 1e-5 {
                    return t * ray.normalize().z;
                }
                t += d;
            }
            0.0
        })
        .collect()
}

/// A box 4 by 6 by 5 cm on a table, and what a camera at `eye` sees of them: the surface, the
/// box's pixels marked as the object's.
fn box_on_a_table(eye: Vec3) -> Surface {
    let table = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.01), Vec3::new(0.5, 0.5, 0.01));
    let block = Obstacle::cuboid(Vec3::new(0.0, 0.0, 0.025), Vec3::new(0.02, 0.03, 0.025));
    let camera = looking(eye, Vec3::new(0.0, 0.0, 0.025));
    let depth = render(&[table, block.clone()], camera);
    let image = DepthImage {
        depth: &depth,
        width: WIDTH,
        intrinsics: K,
        camera,
    };
    let object: Vec<bool> = (0..depth.len())
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let z = depth[i];
            let p = camera.rotation * Vec3::new((u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z, z)
                + camera.position;
            z > 0.0 && block.distance(p).value < 5e-4 && p.z > 1e-3
        })
        .collect();
    assert!(
        object.iter().filter(|&&o| o).count() > 500,
        "the camera sees the box"
    );
    Surface::from_depth(&image, &object, None, 0.005).unwrap()
}

#[test]
fn a_depth_image_gives_the_surfaces_it_sees_with_normals_toward_the_camera() {
    let eye = Vec3::new(0.35, 0.0, 0.35);
    let seen = box_on_a_table(eye);
    assert_eq!(seen.points.len(), seen.normals.len());
    assert_eq!(seen.points.len(), seen.object.len());
    let (mut side, mut top, mut unknown, mut object) = (0, 0, 0, 0);
    for i in 0..seen.points.len() {
        let (p, n) = (seen.points[i], seen.normals[i]);
        if !seen.object[i] {
            assert_eq!(n, Vec3::ZERO, "only the object's normals are found");
            continue;
        }
        object += 1;
        if n == Vec3::ZERO {
            unknown += 1;
            continue;
        }
        assert!(
            (n.length() - 1.0).abs() < 1e-4 && n.dot(eye - p) > 0.0,
            "{n} at {p}"
        );
        // Away from the box's edges, where a patch lies on one face, the face's own normal.
        let inside = p.y.abs() < 0.024 && p.x.abs() < 0.014;
        if (p.x - 0.02).abs() < 1e-3 && (0.006..0.044).contains(&p.z) && p.y.abs() < 0.024 {
            assert!(n.dot(Vec3::X) > 0.999, "{n} at {p}");
            side += 1;
        } else if (p.z - 0.05).abs() < 1e-3 && inside {
            assert!(n.dot(Vec3::Z) > 0.999, "{n} at {p}");
            top += 1;
        }
    }
    assert!(side > 100 && top > 100, "{side} on the side, {top} on top");
    assert!(
        unknown * 20 < object,
        "{unknown} of {object} without a normal"
    );
    // The far side was not seen: nothing there.
    assert!(
        seen.points
            .iter()
            .all(|p| p.x > -0.019 || p.z < 1e-3 || p.z > 0.049)
    );
}

#[test]
fn what_two_cameras_see_between_them_is_grasped_and_what_one_sees_alone_is_not() {
    let o = GraspOptions::default();
    let (near, far) = (Vec3::new(0.35, 0.0, 0.35), Vec3::new(-0.35, 0.0, 0.35));
    // One camera sees one side and the top: nothing to close on.
    let one = box_on_a_table(near);
    assert!(grasps(&one, &o).is_empty());
    // With the other's view of the far side, the box is grasped across the 4 cm between them,
    // clear of the table.
    let mut both = one.clone();
    both.join(box_on_a_table(far));
    assert_eq!(both.object.len(), both.points.len());
    let found = grasps(&both, &o);
    assert!(!found.is_empty());
    let tip = common::panda().unwrap().tip(0);
    // Within the friction cone of the faces' own normals, and a little for the normals found
    // near the box's edges, which lean toward the next face.
    let cone = (o.friction.atan() + 0.05).cos();
    let mut square = 0;
    for g in &found {
        let (line, ends) = closes_on(g);
        assert!(line.x.abs() > cone, "{g:?}");
        square += usize::from(line.x.abs() > 0.999);
        for end in ends {
            assert!((end.x.abs() - 0.02).abs() < 0.002, "{g:?}");
            assert!(
                end.y.abs() < 0.031 && (-0.001..0.051).contains(&end.z),
                "{g:?}"
            );
        }
        assert!((g.width() * line.x.abs() - 0.04).abs() < 0.002, "{g:?}");
        let toward = g.pose.rotation * Vec3::Z;
        let lowest = (g.pose.position + toward * tip).z.min(g.pose.position.z);
        assert!(lowest > -1e-3, "in the table: {g:?}");
    }
    assert!(
        square * 2 > found.len(),
        "most of them square across it: {square} of {}",
        found.len()
    );
    assert!(
        found.iter().any(|g| (g.pose.rotation * Vec3::Z).z < -0.99),
        "from above"
    );
}

#[test]
fn points_get_the_normals_of_the_planes_they_lie_on_toward_where_they_were_seen_from() {
    // Two faces of a box, seen from over its corner.
    let half = Vec3::new(0.02, 0.03, 0.02);
    let mut faces = Surface::default();
    face(
        Vec3::X * half.x,
        Vec3::X,
        [Vec3::Y, Vec3::Z],
        [half.y, half.z],
        &mut faces,
    );
    face(
        Vec3::Z * half.z,
        Vec3::Z,
        [Vec3::X, Vec3::Y],
        [half.x, half.y],
        &mut faces,
    );
    let eye = Vec3::new(0.3, 0.0, 0.3);
    let found = Surface::from_points(&faces.points, eye, 0.005).unwrap();
    assert_eq!(found.points, faces.points);
    assert!(found.object.is_empty());
    let mut known = 0;
    for i in 0..found.points.len() {
        let (p, n) = (found.points[i], found.normals[i]);
        // Within the radius of the edge the two faces share, the plane is neither's.
        let edge = (p.x - half.x).abs() < 0.006 && (p.z - half.z).abs() < 0.006;
        if !edge {
            assert!(n.dot(faces.normals[i]) > 0.999, "{n} at {p}");
            known += 1;
        }
        assert!(n == Vec3::ZERO || n.dot(eye - p) > 0.0, "{n} at {p}");
    }
    assert!(known > 500);
    // Seen from the other side, the same planes face the other way.
    let behind = Surface::from_points(&faces.points, -eye, 0.005).unwrap();
    assert!(behind.normals[0].dot(faces.normals[0]) < -0.999);
    // Points on one line, and too few, lie on no plane one could name.
    let line: Vec<Vec3> = (0..10).map(|k| Vec3::X * k as f32 * 0.001).collect();
    let on_a_line = Surface::from_points(&line, eye, 0.005).unwrap();
    assert!(on_a_line.normals.iter().all(|&n| n == Vec3::ZERO));
    let two = Surface::from_points(&line[..2], eye, 0.005).unwrap();
    assert_eq!(two.normals, [Vec3::ZERO; 2]);
    // And a surface without normals gives no grasp, its points kept all the same.
    assert!(grasps(&on_a_line, &GraspOptions::default()).is_empty());
    assert!(matches!(
        Surface::from_points(&line, eye, 0.0),
        Err(Error::Input(_))
    ));
}

#[test]
fn joined_surfaces_keep_which_points_are_the_objects() {
    let mut plain = plates(0.04);
    let count = plain.points.len();
    plain.join(plates(0.03));
    assert_eq!(plain.points.len(), 2 * count);
    assert!(
        plain.object.is_empty(),
        "every point the object's, as before"
    );
    let mut marked = plates(0.04);
    marked.object = vec![false; count];
    let mut joined = plates(0.03);
    joined.join(marked.clone());
    assert_eq!(joined.object.len(), 2 * count);
    assert!(
        joined.object[..count].iter().all(|&o| o) && joined.object[count..].iter().all(|&o| !o)
    );
    marked.join(plates(0.03));
    assert!(
        marked.object[..count].iter().all(|&o| !o) && marked.object[count..].iter().all(|&o| o)
    );
    assert_eq!(marked.normals.len(), 2 * count);
}

#[test]
fn malformed_surfaces_and_options_are_input_errors() {
    let device = device();
    let good = box_surface(Vec3::splat(0.02), &[]);
    let o = GraspOptions::default();
    let refused = |s: &Surface, hand: usize, o: &GraspOptions| {
        matches!(
            grasps_on(&device, hand, std::slice::from_ref(s), o),
            Err(Error::Input(_))
        )
    };
    assert!(!refused(&good, 0, &o));
    assert!(refused(&good, 1, &o), "a hand the robot lacks");
    let mut short = good.clone();
    short.normals.pop();
    assert!(refused(&short, 0, &o), "a point without a normal");
    let mut long = good.clone();
    long.normals[0] *= 2.0;
    assert!(refused(&long, 0, &o), "a normal that is not unit");
    let mut nowhere = good.clone();
    nowhere.points[0].x = f32::NAN;
    assert!(refused(&nowhere, 0, &o), "a point that is not finite");
    let mut flags = good.clone();
    flags.object = vec![true];
    assert!(refused(&flags, 0, &o), "object flags for some points only");
    for bad in [
        GraspOptions { friction: 0.0, ..o },
        GraspOptions { contact: 0.0, ..o },
        GraspOptions {
            opening: f32::INFINITY,
            ..o
        },
        GraspOptions { turns: 0, ..o },
    ] {
        assert!(refused(&good, 0, &bad), "{bad:?}");
    }
    // No surfaces, and a surface without points, give no grasps.
    assert!(grasps_on(&device, 0, &[], &o).unwrap().is_empty());
    assert!(grasps(&Surface::default(), &o).is_empty());
}
