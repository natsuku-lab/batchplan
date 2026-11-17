//! Grasps of a surface through `grasps_on`: surfaces sampled from shapes whose grasps are known
//! give those grasps and no others, each holding by friction on the shape itself; a wall is
//! grasped across its thickness; what was not seen from both sides, what does not face within the
//! friction cone and what has something else between the fingers is left out.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f32::consts::TAU;

use batchplan::glam::Vec3;
use batchplan::{Device, Error, Grasp, GraspOptions, Surface, grasps_on};

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
    let half = line * g.width / 2.0;
    (line, [g.pose.position - half, g.pose.position + half])
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
        assert!((g.width - 2.0 * half[axis]).abs() < 1e-4, "{g:?}");
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
    // Each pair of patches from `turns` directions, both ways round, the first from above.
    assert_eq!(found.len() % (2 * o.turns), 0);
    let first = found[0].pose.rotation * Vec3::Z;
    assert!(first.dot(Vec3::NEG_Z) > 0.9999, "{first}");
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
            assert!((g.width - 2.0 * half_height).abs() < 1e-3, "{g:?}");
            through += 1;
            continue;
        }
        // Across the wall: along the radius where it stands, one end on each side of it.
        let radial = g.pose.position.with_z(0.0).normalize();
        assert!(line.dot(radial).abs() > 0.995, "{g:?}");
        assert!((g.width - (outer - inner)).abs() < 1e-3, "{g:?}");
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
    assert!(found.iter().all(|g| g.width < 0.06));
}

#[test]
fn a_cup_the_hand_spans_is_grasped_around_it_too() {
    let (outer, inner, half_height) = (0.03, 0.026, 0.06);
    let found = grasps(
        &tube_surface(outer, inner, half_height),
        &GraspOptions::default(),
    );
    let width = |w: f32| found.iter().any(|g| (g.width - w).abs() < 1e-3);
    assert!(width(outer - inner), "across its wall");
    assert!(width(2.0 * outer), "around it");
    // The far wall's inside faces the line's start: no finger presses on it from there.
    assert!(!width(outer + inner), "not onto the inside of its far wall");
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
