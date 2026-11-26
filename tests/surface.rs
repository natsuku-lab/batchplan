//! Grasps of a surface through `grasps_on`: surfaces sampled from shapes whose grasps are known
//! give those grasps and no others, each holding by friction on the shape itself; a wall is
//! grasped across its thickness; what was not seen from both sides, what does not face within the
//! friction cone and what has something else between the fingers is left out; with a shell, a
//! wall seen from one side is grasped where the cameras saw past it, and nothing behind a solid.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::f32::consts::TAU;

use batchplan::glam::{Mat3, Quat, Vec2, Vec3};
use batchplan::rng::Rng;
use batchplan::{
    DepthImage, Device, Error, Grasp, GraspOptions, Gripper, GripperCalibration, HandSphere,
    Intrinsics, Obstacle, Pose, Robot, Screw, Surface, grasps_on,
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

/// The Panda's hand with the widths and friction `o` names: its pads `o.opening` apart as loaded,
/// closing to nothing, and pressing with `o.friction`.
fn calibration(o: &GraspOptions) -> GripperCalibration {
    GripperCalibration {
        friction: o.friction,
        ..GripperCalibration::new(o.opening, 0.0, common::panda_closed())
    }
}

/// Hand 0 of `device`'s robot, a Panda, as [`calibration`] has it for `o`.
fn gripper(device: &Device, o: &GraspOptions) -> Gripper {
    device.robot().gripper(0, &calibration(o)).unwrap()
}

fn grasps(surface: &Surface, o: &GraspOptions) -> Vec<Grasp> {
    let device = device();
    grasps_on(
        &device,
        &gripper(&device, o),
        std::slice::from_ref(surface),
        o,
    )
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

/// Material must fit between both pads at once and clear the rest of the hand.
#[test]
fn the_closing_region_requires_two_pads_and_a_clear_gap_for_the_whole_ball() {
    let device = device();
    let o = GraspOptions::default();
    let open = gripper(&device, &o);
    let hand = open.spheres();
    let g = grasps(&plates_of(0.04, 0.005), &o)[0];
    let to_world = |p| g.pose.position + g.pose.rotation * p;
    let radius = 0.001;
    let at = hand
        .iter()
        .filter(|s| s.side < 0.0)
        .map(|s| Vec3::new(s.center.x, 0.0, s.center.z))
        .find(|p| g.closes_on(&open, to_world(*p), radius))
        .expect("both finger silhouettes contain a ball at least 2 mm wide");
    let p = to_world(at);
    // The Panda with spheres of its own: `change` given its collision model.
    let panda_with = |change: &dyn Fn(&mut batchplan::CollisionModel)| {
        let mut options = common::panda_options();
        change(
            options
                .collision_model
                .as_mut()
                .expect("the Panda's spheres"),
        );
        let robot = Robot::load(common::asset("franka/franka_panda.urdf"), &options).unwrap();
        robot.gripper(0, &calibration(&o)).unwrap()
    };
    for finger in ["panda_leftfinger", "panda_rightfinger"] {
        let one = panda_with(&|model| {
            model.spheres.remove(finger);
        });
        assert!(
            !g.closes_on(&one, p, radius),
            "the pad of {finger} is missing"
        );
    }
    let largest = hand
        .iter()
        .filter(|s| s.side != 0.0)
        .map(|s| s.radius)
        .fold(0.0, f32::max);
    assert!(largest < g.opening * 0.5);
    assert!(
        !g.closes_on(&open, p, largest + radius),
        "the ball extends past the pads"
    );
    // A sphere of the palm's at `at`, placed in the palm link's frame.
    let robot = device.robot();
    let q = robot.default_q();
    let palm = robot.link_pose(q, "panda_hand").expect("the Panda's palm");
    let in_palm = palm.inverse().mul_pose(robot.ee_pose(q, 0));
    let center = in_palm.position + in_palm.rotation * at;
    let palm_radius = hand
        .iter()
        .find(|s| s.side == 0.0)
        .expect("the hand has a palm")
        .radius;
    let blocked = panda_with(&|model| {
        let spheres = model
            .spheres
            .get_mut("panda_hand")
            .expect("the palm's spheres");
        spheres.push([center.x, center.y, center.z, palm_radius]);
    });
    assert!(
        !g.closes_on(&blocked, p, radius),
        "the palm occupies the closing gap"
    );
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
    // Every pair from above first, then every pair a turn to one side, a turn to the other, and
    // so on round to from below.
    let pairs = found.len() / (2 * o.turns);
    let toward = |k: usize| found[k].pose.rotation * Vec3::Z;
    for turn in 0..o.turns {
        let around = turn.div_ceil(2) as f32 * TAU / o.turns as f32;
        for k in 0..2 * pairs {
            let down = toward(turn * 2 * pairs + k).dot(Vec3::NEG_Z);
            assert!((down - around.cos()).abs() < 1e-4, "turn {turn}: {down}");
        }
    }
    // The two sides of a turn are opposite ways round the line.
    let sides = toward(2 * pairs).dot(toward(4 * pairs));
    assert!(
        (sides - (2.0 * TAU / o.turns as f32).cos()).abs() < 1e-4,
        "{sides}"
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
    let found = grasps_on(&device, &gripper(&device, &o), &[cup], &o)
        .unwrap()
        .remove(0);
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
    // Faces that lean less than half a degree, with next to no friction: along the line that
    // splits their normals, 0.007 rad off each, inside a cone of 0.01.
    let slight = 0.4f32.to_radians();
    let o = GraspOptions {
        friction: 0.01,
        ..Default::default()
    };
    let found = grasps(&wedge(slight), &o);
    assert!(!found.is_empty());
    for g in &found {
        let line = closes_on(g).0;
        assert!(line.x.abs() > (0.0105f32 - slight).cos(), "{g:?}");
    }
    // Faces that lean past the cone hold at no friction a fingertip has here.
    assert!(grasps(&wedge(35f32.to_radians()), &GraspOptions::default()).is_empty());
}

/// Grasps of a surface take the hand's widths and its pads' friction from the gripper, not the
/// options: options that name others change nothing, and a gripper that names others changes what
/// holds. The wedge whose faces lean 20 degrees holds in a cone of atan(0.5) and not of atan(0.3),
/// and plates 6.4 cm apart fit fingers 8 cm apart but neither 3 cm nor a capacity of 6 cm.
#[test]
fn grasps_of_a_surface_take_widths_and_friction_from_the_gripper() {
    let device = device();
    let o = GraspOptions::default();
    let hand = gripper(&device, &o);
    let (wedge, plates) = (wedge(20f32.to_radians()), plates_of(0.064, 0.005));
    let on = |gripper: &Gripper, s: &Surface, o: &GraspOptions| {
        let found = grasps_on(&device, gripper, std::slice::from_ref(s), o);
        found.unwrap().remove(0)
    };
    let elsewhere = GraspOptions {
        opening: 0.03,
        friction: 0.3,
        ..o
    };
    for s in [&wedge, &plates] {
        let found = on(&hand, s, &o);
        assert!(!found.is_empty());
        assert_eq!(on(&hand, s, &elsewhere), found);
    }
    let rough = GripperCalibration {
        friction: 0.3,
        ..calibration(&o)
    };
    let small = GripperCalibration {
        capacity: Some(0.06),
        ..calibration(&o)
    };
    let build = |c: GripperCalibration| device.robot().gripper(0, &c).unwrap();
    assert!(on(&build(rough), &wedge, &o).is_empty());
    assert!(on(&build(small), &plates, &o).is_empty());
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
    let mut blocked = alone;
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
    // A wall facing the object, `at` from its middle along the closing line, on either side.
    let beside = |at: f32| {
        let mut s = alone.clone();
        the_object(&mut s);
        face(
            Vec3::X * at,
            Vec3::X * -at.signum(),
            [Vec3::Y, Vec3::Z],
            [0.05, 0.05],
            &mut s,
        );
        s.object.resize(s.points.len(), false);
        grasps(&s, &o).len()
    };
    // Open 8 cm, a finger stands 4 cm out: in a wall 4.5 cm out, and clear of one 12 cm out.
    assert_eq!(beside(0.045), 0);
    assert_eq!(beside(-0.045), 0);
    assert_eq!(beside(0.12), free);
    assert_eq!(beside(-0.12), free);
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

/// The Panda's spheres in its IK frame with its fingers `opening` apart, as the robot relocked
/// there has them: those within 12 cm of the frame.
fn hand_open(opening: f32) -> Vec<(Vec3, f32)> {
    let fingers = ["panda_finger_joint1", "panda_finger_joint2"];
    let locked: HashMap<String, f32> = fingers
        .iter()
        .map(|joint| (joint.to_string(), opening / 2.0))
        .collect();
    let robot = common::panda().unwrap().relock(&locked).unwrap();
    let q = robot.default_q().to_vec();
    let frame = robot.ee_pose(&q, 0).inverse();
    let near = robot.spheres(&q).into_iter().filter_map(|s| {
        let at = frame.rotation * Vec3::new(s[0], s[1], s[2]) + frame.position;
        (at.length() < 0.12).then_some((at, s[3]))
    });
    near.collect()
}

/// Whether `hand` at `g`, and its way in, lie clear of every point of `s` but the object's
/// between the two contacts; with `others`, of every point that is not the object's.
fn clear_at(hand: &[(Vec3, f32)], g: &Grasp, s: &Surface, approach: f32, others: bool) -> bool {
    let (closing, toward) = (g.pose.rotation * Vec3::Y, g.pose.rotation * Vec3::Z);
    hand_at(hand, g).all(|(center, radius)| {
        s.points.iter().enumerate().all(|(i, &point)| {
            let object = s.object.is_empty() || s.object[i];
            let across = (point - g.pose.position).dot(closing).abs();
            if object && (others || across <= g.width() / 2.0 + 0.0025) {
                return true;
            }
            let back = (center - point).dot(toward).clamp(0.0, approach);
            (point - (center - toward * back)).length() > radius
        })
    })
}

#[test]
fn measured_grasps_use_the_loaded_gap_and_only_commandable_openings() {
    let loaded = 0.070;
    let capacity = 0.060;
    let locks = ["panda_finger_joint1", "panda_finger_joint2"]
        .map(|joint| (joint.to_string(), loaded / 2.0))
        .into_iter()
        .collect();
    let device = Device::cpu(&common::panda().unwrap().relock(&locks).unwrap()).unwrap();
    let options = GraspOptions {
        opening: loaded,
        narrow: true,
        ..GraspOptions::default()
    };
    let at_capacity = |capacity| {
        let calibration = GripperCalibration {
            capacity: Some(capacity),
            ..calibration(&options)
        };
        device.robot().gripper(0, &calibration)
    };
    let search = |s: &Surface, capacity| {
        let gripper = at_capacity(capacity)?;
        grasps_on(&device, &gripper, std::slice::from_ref(s), &options)
            .map(|mut sets| sets.remove(0))
    };
    let wide = plates_of(0.064, 0.005);
    assert!(!search(&wide, loaded).unwrap().is_empty());
    assert!(
        search(&wide, capacity).unwrap().is_empty(),
        "pair exceeds capacity"
    );
    for invalid in [0.0, -0.01, loaded + 0.001, f32::NAN] {
        assert!(matches!(search(&wide, invalid), Err(Error::Input(_))));
    }
    // Plates a surface's noise to each finger narrower than the passage's opening, the nearest
    // the fingers may come in.
    let mut s = plates_of(0.039, 0.005);
    let free = search(&s, capacity).unwrap();
    assert!(!free.is_empty() && free.iter().all(|g| g.opening == loaded));
    // This reading is outside half the capacity but inside the loaded finger gap.
    let point = Vec3::new(-0.031, -0.00525, -0.01025);
    let angles = |close_on| {
        let options = GraspOptions {
            turns: 3,
            close_on,
            ..options
        };
        let gripper = at_capacity(capacity).unwrap();
        grasps_on(&device, &gripper, std::slice::from_ref(&s), &options)
            .unwrap()
            .remove(0)
    };
    let coarse = angles(None);
    let refined = angles(Some(point));
    assert!(refined.starts_with(&coarse));
    let hand = at_capacity(capacity).unwrap();
    let added = &refined[coarse.len()..];
    assert!(
        added.iter().any(|g| {
            let middle = g.contacts[0].midpoint(g.contacts[1]);
            (point - middle).dot(g.pose.rotation * Vec3::Y).abs() > capacity / 2.0
        }),
        "the loaded jaws admit a reading beyond half the graspable width"
    );
    assert!(added.iter().all(|g| g.closes_on(&hand, point, 0.0)));
    the_object(&mut s);
    for sign in [-1.0, 1.0] {
        face(
            Vec3::X * sign * 0.044,
            Vec3::X * -sign,
            [Vec3::Y, Vec3::Z],
            [0.05, 0.01],
            &mut s,
        );
    }
    s.object.resize(s.points.len(), false);
    let found = search(&s, capacity).unwrap();
    assert!(
        found.iter().any(|g| (g.opening - 0.045).abs() < 1e-6),
        "{found:?}"
    );
    let wrong = hand_open(0.055);
    assert!(
        found.iter().any(|g| (g.opening - 0.045).abs() < 1e-6
            && !clear_at(&wrong, g, &s, options.approach, false)),
        "at least one passage blocks a 7.5 mm slide but admits a 12.5 mm slide per finger"
    );
    for g in &found {
        assert!(g.opening == loaded || g.opening < capacity);
        assert!(clear_at(
            &hand_open(g.opening),
            g,
            &s,
            options.approach,
            false
        ));
    }
    // A capacity left out is the loaded gap.
    assert_eq!(
        search(&s, loaded).unwrap(),
        grasps_on(&device, &gripper(&device, &options), &[s], &options).unwrap()[0]
    );
}

/// Fingers that may come in narrower (`GraspOptions::narrow`) take, of sixteenths of the opening
/// the Panda relocked there is clear at, way in included, the one farthest from any at which it
/// would be in what is not the object's, as far past open as the fingers close; of those as far,
/// the one with the most room, as many steps from the first it would be in anything at farther
/// apart as from the first nearer, the last a surface's noise off each patch (2.5 mm), the nearest
/// a finger may come in: one nearer may be on its patch's wall before it closes. And of those the
/// widest. So between walls in the open fingers' way, or just past them, the fingers come in
/// that far off the plates; beside a ledge of the object itself they stand midway between it and
/// there; what is not the
/// object's past one open finger is found though the object's own walls fill every opening past
/// the other; and with nothing within their travel they come in open, every grasp as without
/// the option.
#[test]
fn fingers_that_may_come_in_narrower_keep_farthest_from_what_stands_around() {
    let o = GraspOptions::default();
    let narrow = GraspOptions { narrow: true, ..o };
    let alone = plates_of(0.02, 0.005);
    let open = grasps(&alone, &o);
    assert!(!open.is_empty() && open.iter().all(|g| g.opening == o.opening));
    assert_eq!(grasps(&alone, &narrow), open);
    // The openings they may come in at, widest first: the whole one and each sixteenth less.
    let openings = gripper(&device(), &o).openings();
    assert_eq!((openings.len(), openings[0]), (16, o.opening));
    let sixteenth = |pair: &[f32]| (pair[0] - pair[1] - o.opening / 16.0).abs() < 1e-6;
    assert!(openings.windows(2).all(sixteenth), "{openings:?}");
    // Walls facing the plates, low enough that a hand from above has only its fingertips
    // between them: something else's, or the object's own.
    let between = |walls: &[f32], own: &[f32]| {
        let mut s = alone.clone();
        the_object(&mut s);
        for (walls, theirs) in [(own, true), (walls, false)] {
            for &at in walls {
                face(
                    Vec3::X * at,
                    Vec3::X * -at.signum(),
                    [Vec3::Y, Vec3::Z],
                    [0.05, 0.01],
                    &mut s,
                );
            }
            s.object.resize(s.points.len(), theirs);
        }
        s
    };
    let step = o.opening / 16.0;
    let nearest = 0.02 + 2.0 * 0.0025;
    let most = ((o.opening - nearest) / step).round() as i32;
    let hands: Vec<Vec<(Vec3, f32)>> = (-most..=most)
        .map(|k| hand_open(o.opening - k as f32 * step))
        .collect();
    // The grasps of the plates among `walls`, each at the opening the rule gives the Panda, and
    // the opening of one from above.
    let from_above = |walls: &[f32], own: &[f32]| {
        let s = between(walls, own);
        let found = grasps(&s, &narrow);
        assert!(found.len() >= grasps(&s, &o).len() && !found.is_empty());
        for g in &found {
            let clear = |others: bool| -> Vec<bool> {
                let at = |hand: &Vec<_>| clear_at(hand, g, &s, o.approach, others);
                hands.iter().map(at).collect()
            };
            let (free, apart) = (clear(false), clear(true));
            let free = |k: i32| free[(k + most) as usize];
            let from_others = |k: i32| {
                let theirs = (-most..=most).filter(|&j| !apart[(j + most) as usize]);
                theirs.map(|j| (j - k).abs()).min().unwrap_or(i32::MAX)
            };
            let room = |k: i32| {
                let wider = (-most..k).rev().find(|&j| !free(j));
                let nearer = (k + 1..=most).find(|&j| !free(j)).unwrap_or(most + 1);
                wider.map_or(i32::MAX, |j| k - j).min(nearer - k)
            };
            let best = (0..=most)
                .filter(|&k| free(k))
                .max_by_key(|&k| (from_others(k), room(k), -k))
                .unwrap_or_else(|| panic!("the Panda is clear at no opening for {g:?}"));
            let expected = o.opening - best as f32 * step;
            assert!(
                (g.opening - expected).abs() < 1e-6,
                "{g:?} for {expected}, clear at {:?}",
                (0..=most).filter(|&k| free(k)).collect::<Vec<_>>()
            );
            assert!(
                openings.contains(&g.opening),
                "{} of {openings:?}",
                g.opening
            );
        }
        let down = found.iter().find(|g| (g.pose.rotation * Vec3::Z).z < -0.99);
        let opening = down.expect("a grasp from above").opening;
        (found, opening)
    };
    // In the open fingers' way, and 4 mm past them: a surface's noise off the plates, which are
    // 2 cm apart.
    for walls in [[0.045, -0.045], [0.04, -0.06], [0.065, -0.065]] {
        let in_the_way = grasps(&between(&walls, &[]), &o).is_empty();
        assert_eq!(in_the_way, walls[0] < 0.06);
        let (_, opening) = from_above(&walls, &[]);
        assert!((opening - nearest).abs() < 1e-6, "{walls:?}: {opening}");
    }
    // Farther than the fingers could reach: as without the option.
    let far = [0.11, -0.11];
    let (found, opening) = from_above(&far, &[]);
    assert_eq!(found, grasps(&between(&far, &[]), &o));
    assert_eq!(opening, o.opening);
    // The object's own ledge where one open finger stands: midway between it and the plate.
    let (_, opening) = from_above(&[], &[0.045]);
    assert!((opening - 0.035).abs() < 1e-6, "{opening}");
    // The object's own walls all the way past one open finger, and a wall of something else
    // 4 mm past the other: what is not the object's is found though the object's fill every
    // opening past open, and the fingers come in as near the plates as they may, whichever finger
    // is on which side.
    let (found, _) = from_above(&[-0.065], &[0.062, 0.075, 0.088]);
    let down = found
        .iter()
        .filter(|g| (g.pose.rotation * Vec3::Z).z < -0.99);
    let openings: Vec<f32> = down.map(|g| g.opening).collect();
    assert!(openings.len() >= 2 && openings.iter().all(|w| (w - nearest).abs() < 1e-6));
}

/// A grasp the open hand has stays where it is with fingers that may come in narrower: a stud
/// under one open fingertip draws the open hand back from the plates, and narrower fingers,
/// which clear it at full depth, come in to that grasp where the open hand does. Where the open
/// hand is clear at no depth, with the stud too tall to draw back from, the hand comes in
/// narrower at full depth.
#[test]
fn fingers_that_may_come_in_narrower_leave_the_open_hands_grasps_where_they_are() {
    // A hand may be drawn back until 4 mm of its fingers are left on the plates.
    let o = GraspOptions {
        overlap: 0.004,
        ..Default::default()
    };
    let narrow = GraspOptions { narrow: true, ..o };
    // A stud where the +x open fingertip of a hand from above comes down, its top `under` the
    // plates' middle.
    let with_stud = |under: f32| {
        let mut s = plates_of(0.02, 0.005);
        the_object(&mut s);
        face(
            Vec3::new(0.05, 0.0, -under),
            Vec3::Z,
            [Vec3::X, Vec3::Y],
            [0.01, 0.01],
            &mut s,
        );
        s.object.resize(s.points.len(), false);
        s
    };
    let from_above = |g: &&Grasp| (g.pose.rotation * Vec3::Z).z < -0.99;
    let back = |g: &Grasp| {
        let middle = (g.contacts[0] + g.contacts[1]) / 2.0;
        (middle - g.pose.position).length()
    };
    let low = with_stud(0.008);
    let open = grasps(&low, &o);
    let drawn_back = open.iter().filter(from_above).filter(|g| back(g) > 1e-3);
    assert!(drawn_back.count() > 0, "the stud draws the open hand back");
    let found = grasps(&low, &narrow);
    for g in &open {
        assert!(
            found.iter().any(|n| n.pose == g.pose),
            "the open hand's grasp stays where it is: {g:?}"
        );
    }
    // Too tall to draw back from: the open hand has no grasp from above, and narrower fingers
    // come in at full depth.
    let tall = with_stud(-0.004);
    assert_eq!(grasps(&tall, &o).iter().filter(from_above).count(), 0);
    let found = grasps(&tall, &narrow);
    let deep = found.iter().filter(from_above).filter(|g| back(g) < 1e-6);
    assert!(deep.clone().count() > 0);
    assert!(deep.into_iter().all(|g| g.opening < o.opening - 0.02));
}

/// With a `clearance` the hand keeps that far from what it does not close on, its way in included:
/// plates grasped from above with something else's face a millimeter under where the open
/// fingertips come down, as a drawer's front behind the handle a hand grips. The points alone
/// find the hand clear at full depth, where a planner whose grids err toward collision refuses
/// it; with 2.5 mm of clearance the hand from above is drawn back to where it has that, and no
/// grasp comes nearer on its way in, with the fingers open or come in narrower.
#[test]
fn a_hand_keeps_its_clearance_from_what_it_does_not_close_on() {
    let tip = common::panda().unwrap().tip(0);
    let mut s = plates_of(0.02, 0.005);
    the_object(&mut s);
    face(
        Vec3::Z * -(tip + 0.001),
        Vec3::Z,
        [Vec3::X, Vec3::Y],
        [0.08, 0.08],
        &mut s,
    );
    s.object.resize(s.points.len(), false);
    let clearance = 0.0025;
    let grown = |opening: f32| -> Vec<(Vec3, f32)> {
        let hand = hand_open(opening);
        let wider = hand
            .iter()
            .map(|&(at, radius)| (at, radius + clearance - 1e-4));
        wider.collect()
    };
    // How low the hand from above comes: its frame's height in the lowest of those grasps.
    let from_above = |g: &&Grasp| (g.pose.rotation * Vec3::Z).z < -0.99;
    let lowest = |found: &[Grasp]| {
        let down = found.iter().filter(from_above).map(|g| g.pose.position.z);
        down.fold(f32::INFINITY, f32::min)
    };
    for narrow in [false, true] {
        let o = GraspOptions {
            narrow,
            ..Default::default()
        };
        let near = grasps(&s, &o);
        let kept = grasps(&s, &GraspOptions { clearance, ..o });
        let roomy = |g: &Grasp| clear_at(&grown(g.opening), g, &s, o.approach, false);
        // Without it some grasp from above comes nearer than that.
        assert!(
            near.iter().filter(from_above).any(|g| !roomy(g)),
            "narrow {narrow}"
        );
        // With it none does, and the hand from above is drawn back, its grasps kept.
        assert!(kept.iter().all(roomy), "narrow {narrow}");
        let (low, back) = (lowest(&near), lowest(&kept));
        assert!(
            back.is_finite() && back > low + 1e-4,
            "narrow {narrow}: {low} {back}"
        );
    }
    // Clearance is a distance: none below 0.
    let o = GraspOptions {
        clearance: -0.001,
        ..Default::default()
    };
    let device = device();
    let refused = grasps_on(&device, &gripper(&device, &o), std::slice::from_ref(&s), &o);
    assert!(matches!(refused, Err(Error::Input(_))));
}

/// Narrower fingers stand higher where the hand closes upright: over a floor the open fingers
/// would be under, a grasp comes in with its fingers as far over it as they can be, on the plates.
#[test]
fn fingers_that_come_in_narrower_stay_over_the_floor() {
    let tip = common::panda().unwrap().tip(0);
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
    let over = GraspOptions {
        floor: Some(-0.03),
        ..Default::default()
    };
    assert!(grasps(&upright, &over).is_empty());
    let narrow = GraspOptions {
        narrow: true,
        ..over
    };
    let found = grasps(&upright, &narrow);
    assert!(!found.is_empty());
    for g in &found {
        let (closing, toward) = (g.pose.rotation * Vec3::Y, g.pose.rotation * Vec3::Z);
        let tips = (g.pose.position + toward * tip).z - closing.z.abs() * g.opening / 2.0;
        assert!(tips >= -0.03 - 1e-6, "{g:?}");
        assert!((g.opening - 0.045).abs() < 1e-6, "{g:?}");
    }
    // A floor only fingers a surface's noise off the plates clear leaves those grasps, and one
    // nearer the plates none: the fingers come in no nearer.
    let just = |floor: f32| GraspOptions {
        floor: Some(floor),
        ..narrow
    };
    assert_eq!(grasps(&upright, &just(-0.023)).len(), found.len());
    assert!(grasps(&upright, &just(-0.022)).is_empty());
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
    let distance = |p: Vec3| {
        scene
            .iter()
            .map(|o| o.distance(p).value)
            .fold(f32::INFINITY, f32::min)
    };
    render_with(distance, camera)
}

/// z depth per pixel by sphere tracing `distance`; 0 where a ray hits nothing.
fn render_with(distance: impl Fn(Vec3) -> f32, camera: Pose) -> Vec<f32> {
    (0..WIDTH * HEIGHT)
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let ray = Vec3::new((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0);
            let dir = camera.rotation * ray.normalize();
            let mut t = 0.0;
            while t < 2.0 {
                let p = camera.position + dir * t;
                let d = distance(p);
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
    box_read_farther(eye, 0.0)
}

/// As `box_on_a_table`, each reading `farther` (meters) than the scene is.
fn box_read_farther(eye: Vec3, farther: f32) -> Surface {
    let table = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.01), Vec3::new(0.5, 0.5, 0.01));
    let block = Obstacle::cuboid(Vec3::new(0.0, 0.0, 0.025), Vec3::new(0.02, 0.03, 0.025));
    let camera = looking(eye, Vec3::new(0.0, 0.0, 0.025));
    let mut depth = render(&[table, block.clone()], camera);
    for reading in depth.iter_mut().filter(|z| **z > 0.0) {
        *reading += farther;
    }
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
            z > 0.0 && block.distance(p).value < 5e-4 + farther && p.z > 1e-3
        })
        .collect();
    assert!(
        object.iter().filter(|&&o| o).count() > 500,
        "the camera sees the box"
    );
    Surface::from_depth(&device(), &image, &object, None, 0.005).unwrap()
}

#[test]
fn a_depth_image_gives_the_surfaces_it_sees_with_normals_toward_the_camera() {
    let eye = Vec3::new(0.35, 0.0, 0.35);
    let seen = box_on_a_table(eye);
    assert_eq!(seen.points.len(), seen.normals.len());
    assert_eq!(seen.points.len(), seen.object.len());
    assert_eq!(seen.seen_from, vec![eye; seen.points.len()]);
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
fn without_a_shell_what_two_cameras_see_between_them_is_grasped_and_what_one_sees_alone_is_not() {
    let o = without_shell();
    let (near, far) = (Vec3::new(0.35, 0.0, 0.35), Vec3::new(-0.35, 0.0, 0.35));
    // One camera sees one side and the top: without a shell, nothing to close on.
    let one = box_on_a_table(near);
    assert!(grasps(&one, &o).is_empty());
    // With the other's view of the far side, the box is grasped across the 4 cm between them,
    // clear of the table.
    let mut both = one;
    both.join(box_on_a_table(far));
    assert_eq!(both.object.len(), both.points.len());
    let found = grasps(&both, &o);
    assert!(!found.is_empty());
    // Every side the box is grasped across seen, a shell adds nothing.
    assert_eq!(grasps(&both, &with_shell(0.015)), found);
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

/// A bowl as LIBERO's, standing on z = 0: 11 cm across its rim and 5 cm tall, its wall `wall`
/// thick, flared 26 degrees from upright and rounded at the rim. In its profile (radius, height),
/// its wall's middle and its bottom's, which it lies within half the wall of.
fn bowl_profile(wall: f32) -> [(Vec2, Vec2); 2] {
    let half = wall / 2.0;
    let edge = Vec2::new(0.03, half);
    let rim = Vec2::new(0.055 - half, 0.05 - half);
    [(edge, rim), (Vec2::new(0.0, half), edge)]
}

/// How far `p` lies from the bowl of `wall`, negative inside it.
fn bowl_distance(p: Vec3, wall: f32) -> f32 {
    let at = Vec2::new(p.truncate().length(), p.z);
    let from = |(a, b): (Vec2, Vec2)| {
        let t = ((at - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
        at.distance(a + (b - a) * t)
    };
    let [side, bottom] = bowl_profile(wall);
    from(side).min(from(bottom)) - wall / 2.0
}

/// Which side of the middle of the bowl's wall `p` lies on, and how far: positive outside.
fn outside_the_wall(p: Vec3, wall: f32) -> f32 {
    let [(edge, rim), _] = bowl_profile(wall);
    let up = (rim - edge).normalize();
    (Vec2::new(p.truncate().length(), p.z) - edge).perp_dot(up)
}

/// What cameras at `eyes` see of the bowl of `wall` on a table, the bowl's readings marked as the
/// object's.
fn bowl_seen(wall: f32, eyes: &[Vec3]) -> Surface {
    bowl_read_farther(wall, eyes, 0.0)
}

/// As `bowl_seen`, each reading `farther` (meters) than the scene is.
fn bowl_read_farther(wall: f32, eyes: &[Vec3], farther: f32) -> Surface {
    let table = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.01), Vec3::new(0.5, 0.5, 0.01));
    let mut seen = Surface::default();
    for &eye in eyes {
        let camera = looking(eye, Vec3::new(0.0, 0.0, 0.025));
        let mut depth = render_with(
            |p| bowl_distance(p, wall).min(table.distance(p).value),
            camera,
        );
        for reading in depth.iter_mut().filter(|z| **z > 0.0) {
            *reading += farther;
        }
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
                let p = camera.rotation
                    * Vec3::new((u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z, z)
                    + camera.position;
                z > 0.0 && bowl_distance(p, wall) < 5e-4 + farther && p.z > 1e-3
            })
            .collect();
        seen.join(Surface::from_depth(&device(), &image, &object, None, 0.005).unwrap());
    }
    seen
}

/// Over the bowl, as a wrist camera 40 cm above it: its rim and inner wall, not its outer wall,
/// which flares out of sight under the rim.
const ABOVE: Vec3 = Vec3::new(0.02, 0.0, 0.42);

/// In front of the bowl and over it, as a camera on the robot's base: its near wall's outside and
/// its far wall's inside.
const FRONT: Vec3 = Vec3::new(0.45, 0.0, 0.4);

/// A thin wall's: a LIBERO bowl's.
const THIN: f32 = 0.003;

/// The Panda's hand's spheres in its IK frame, open as the Panda is loaded: the spheres within
/// 12 cm of the frame at its default configuration.
fn hand() -> Vec<(Vec3, f32)> {
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let frame = robot.ee_pose(&q, 0).inverse();
    let near = robot.spheres(&q).into_iter().filter_map(|s| {
        let at = frame.rotation * Vec3::new(s[0], s[1], s[2]) + frame.position;
        (at.length() < 0.12).then_some((at, s[3]))
    });
    let near: Vec<(Vec3, f32)> = near.collect();
    assert!(near.len() > 10, "{} spheres", near.len());
    near
}

/// `hand`'s spheres at grasp `g`.
fn hand_at(hand: &[(Vec3, f32)], g: &Grasp) -> impl Iterator<Item = (Vec3, f32)> {
    let pose = g.pose;
    hand.iter()
        .map(move |&(at, r)| (pose.rotation * at + pose.position, r))
}

fn with_shell(shell: f32) -> GraspOptions {
    GraspOptions {
        shell: Some(shell),
        ..Default::default()
    }
}

fn without_shell() -> GraspOptions {
    GraspOptions {
        shell: None,
        ..Default::default()
    }
}

#[test]
fn a_rim_seen_from_above_is_grasped_across_its_wall_with_a_shell() {
    let seen = bowl_seen(THIN, &[ABOVE]);
    // The camera sees the rim and the inner wall, and nothing of the outer wall under the rim.
    let ours = |i: &usize| seen.object[*i];
    let inner = (0..seen.points.len())
        .filter(ours)
        .filter(|&i| outside_the_wall(seen.points[i], THIN) < 0.0)
        .count();
    let outer = (0..seen.points.len())
        .filter(ours)
        .filter(|&i| outside_the_wall(seen.points[i], THIN) > 0.0 && seen.points[i].z < 0.045)
        .count();
    assert!(
        inner > 1000 && outer == 0,
        "{inner} inside, {outer} outside"
    );
    assert!(seen.seen_from.iter().all(|&from| from == ABOVE));
    assert!(grasps(&seen, &without_shell()).is_empty());
    let o = with_shell(0.015);
    let found = grasps(&seen, &o);
    let hand = hand();
    let mut sectors = [false; 8];
    for g in &found {
        let (line, [first, second]) = closes_on(g);
        // From the inner wall seen out through the wall, to where the camera saw past it.
        let side = [first, second].map(|p| outside_the_wall(p, THIN));
        let (seen_end, beyond) = if side[0] < side[1] {
            (first, second)
        } else {
            (second, first)
        };
        assert!(
            outside_the_wall(seen_end, THIN) < 0.0 && outside_the_wall(beyond, THIN) > 0.0,
            "{g:?}"
        );
        assert!(bowl_distance(seen_end, THIN).abs() < 1.5e-3, "{g:?}");
        assert!(bowl_distance(beyond, THIN) > 0.0, "{g:?}");
        assert!(g.width() <= o.shell.unwrap() + 1e-6, "{g:?}");
        let through = (0..=20).map(|k| first + (second - first) * k as f32 / 20.0);
        assert!(
            through.map(|p| bowl_distance(p, THIN)).any(|d| d < 0.0),
            "{g:?}"
        );
        // Along the wall's normal there, near the rim.
        let radial = seen_end.with_z(0.0).normalize();
        assert!(
            line.with_z(0.0).normalize().dot(radial).abs() > 0.99,
            "{g:?}"
        );
        assert!(seen_end.z > 0.035, "{g:?}");
        // The open hand is out of the bowl, outer wall and all, which no camera saw.
        for (center, radius) in hand_at(&hand, g) {
            assert!(bowl_distance(center, THIN) > radius - 1e-3, "{g:?}");
        }
        let angle = seen_end.y.atan2(seen_end.x).rem_euclid(TAU);
        sectors[(angle / TAU * 8.0) as usize % 8] = true;
    }
    assert!(
        sectors.iter().all(|&s| s),
        "all around the rim: {sectors:?}"
    );
    let down = |g: &&Grasp| (g.pose.rotation * Vec3::Z).dot(Vec3::NEG_Z) > 0.8;
    assert!(found.iter().filter(down).count() > 16, "many from above");
}

/// A wall seen from one side that leans is closed on level too, where that is inside its
/// friction cone: the flared bowl's inner wall leans 27 degrees, so fingers that hold with a
/// friction of 2 close level across its rim all around, the hand coming straight down, each
/// grasp from the inner wall seen to a point outside the wall at its height, the open hand out
/// of the bowl; these follow none of the grasps along the wall's normal out of place, and
/// fingers with a friction of 0.3, whose cone the level line is outside, get none.
#[test]
fn a_leaning_wall_seen_from_one_side_is_closed_on_level_within_its_friction_cone() {
    let seen = bowl_seen(THIN, &[ABOVE]);
    let with_friction = |friction: f32| GraspOptions {
        friction,
        ..with_shell(0.015)
    };
    let level = |g: &&Grasp| (g.pose.rotation * Vec3::Y).z.abs() < 1e-4;
    let found = grasps(&seen, &with_friction(2.0));
    let hand = hand();
    let mut sectors = [false; 8];
    let mut from_above = 0;
    for g in found.iter().filter(level) {
        assert!(g.one_sided, "{g:?}");
        let (_, [first, second]) = closes_on(g);
        let side = [first, second].map(|p| outside_the_wall(p, THIN));
        let (seen_end, beyond) = if side[0] < side[1] {
            (first, second)
        } else {
            (second, first)
        };
        // From the inner wall seen, level and out through the wall, to where the camera saw
        // past it.
        assert!(bowl_distance(seen_end, THIN).abs() < 1.5e-3, "{g:?}");
        assert!(outside_the_wall(beyond, THIN) > 0.0, "{g:?}");
        assert!(bowl_distance(beyond, THIN) > 0.0, "{g:?}");
        assert!((seen_end.z - beyond.z).abs() < 1e-4, "{g:?}");
        let radial = seen_end.with_z(0.0).normalize();
        assert!((beyond - seen_end).normalize().dot(radial) > 0.99, "{g:?}");
        for (center, radius) in hand_at(&hand, g) {
            assert!(bowl_distance(center, THIN) > radius - 1e-3, "{g:?}");
        }
        if (g.pose.rotation * Vec3::Z).z < -0.9999 {
            from_above += 1;
            let angle = seen_end.y.atan2(seen_end.x).rem_euclid(TAU);
            sectors[(angle / TAU * 8.0) as usize % 8] = true;
        }
    }
    assert!(
        from_above > 16 && sectors.iter().all(|&s| s),
        "{from_above} straight down, around the rim: {sectors:?}"
    );
    // Along the wall's normal no hand comes straight down: it leans as the wall does.
    let leaning = found.iter().filter(|g| !level(g));
    let steepest = leaning
        .map(|g| (g.pose.rotation * Vec3::Z).z)
        .fold(0.0, f32::min);
    assert!(steepest > -0.95, "{steepest}");
    // The grasps along the normal are those found without the level ones, in their order.
    let slippery = grasps(&seen, &with_friction(0.3));
    assert!(!slippery.is_empty() && !slippery.iter().any(|g| level(&g)));
}

/// Points on the whole of the bowl of `wall`, inside and out, its rim's round top and its bottom,
/// about a millimeter apart: what a finger could meet of it, seen or not.
fn bowl_points(wall: f32) -> Vec<Vec3> {
    let half = wall / 2.0;
    let mut profile = vec![];
    for (a, b) in bowl_profile(wall) {
        let along = (b - a).normalize();
        let steps = ((b - a).length() / 1e-3).ceil() as usize;
        for k in 0..=steps {
            let at = a + (b - a) * (k as f32 / steps as f32);
            profile.extend([at + along.perp() * half, at - along.perp() * half]);
        }
    }
    let [(edge, rim), _] = bowl_profile(wall);
    let up = (rim - edge).normalize();
    for k in 0..=8 {
        let angle = k as f32 / 8.0 * TAU / 2.0;
        profile.push(rim + (up.perp() * angle.cos() + up * angle.sin()) * half);
    }
    let mut points = vec![];
    for at in profile {
        let around = ((TAU * at.x / 1e-3).ceil() as usize).max(1);
        for k in 0..around {
            let angle = k as f32 / around as f32 * TAU;
            points.push(Vec3::new(at.x * angle.cos(), at.x * angle.sin(), at.y));
        }
    }
    points
}

/// How far past its own patch, along its closing line, the most of `points` a finger of `hand`
/// comes within its spheres, each `inset` less wide, of as the fingers close from `g`'s opening
/// onto the patches: 0 for nothing past them, a finger meeting first what it closes on.
fn met_past_the_patches(
    g: &Grasp,
    hand: &[HandSphere],
    gap: f32,
    points: &[Vec3],
    inset: f32,
) -> f32 {
    let closing = g.pose.rotation * Vec3::Y;
    let width = g.width();
    let mut most = 0.0f32;
    for sphere in hand.iter().filter(|sphere| sphere.side != 0.0) {
        let center = g.pose.position + g.pose.rotation * sphere.center;
        let at = |opening: f32| center - closing * (sphere.side * (gap - opening) / 2.0);
        let (a, b) = (at(g.opening), at(width));
        for &p in points {
            let past = (p - g.pose.position).dot(closing).abs() - width / 2.0;
            if past <= most {
                continue;
            }
            let t = ((p - a).dot(b - a) / (b - a).length_squared().max(1e-12)).clamp(0.0, 1.0);
            if p.distance(a + (b - a) * t) <= sphere.radius - inset {
                most = past;
            }
        }
    }
    most
}

/// With `first_touch`, a grasp whose fingers meet the object before their patches comes after every
/// one whose fingers meet it first there, and without it the same grasps come as before: closing
/// level deep across the flared bowl's wall, seen from in front and over it as a robot's cameras
/// see a bowl on a cabinet, the finger on the side the wall leans over meets the wall above its
/// patch, a finger's length from the other contact, and the bowl turns in the hand as it is lifted
/// (LIBERO's spatial task 9: the jaw read 8.7 mm at the close and 3.5 mm lifted, the bowl 15
/// degrees over). Held to the bowl's true shape, its unseen outside too: within each turn of each
/// lot, two seen sides and one, the grasps whose fingers meet the bowl more than two patches'
/// radius past their own come after all whose fingers meet it no more than a surface's noise past,
/// fingers that come in narrower swept from where they come in; and so of plates with a ledge
/// beside one. From above still comes before a turn to either side: a hand turned 44 degrees about
/// its closing line that closed cleanly on a bowl's rim came before every late grasp from above,
/// and closed on nothing.
#[test]
fn grasps_whose_fingers_meet_the_object_before_their_patches_come_after_the_rest() {
    let seen = bowl_seen(THIN, &[FRONT, ABOVE]);
    let o = GraspOptions {
        friction: 2.0,
        first_touch: true,
        ..with_shell(0.08)
    };
    let found = grasps(&seen, &o);
    // Without it, the same grasps in the order they came before, which puts some that meet the
    // bowl past their patches before some that meet it there.
    let plain = grasps(
        &seen,
        &GraspOptions {
            first_touch: false,
            ..o
        },
    );
    let sorted = |gs: &[Grasp]| {
        let mut keys: Vec<_> = gs.iter().map(|g| format!("{g:?}")).collect();
        keys.sort();
        keys
    };
    assert_eq!(sorted(&found), sorted(&plain));
    let default = GraspOptions {
        friction: 2.0,
        ..with_shell(0.08)
    };
    assert_eq!(plain, grasps(&seen, &default), "off unless asked for");
    let robot = common::panda().unwrap();
    let hand = robot.hand_spheres(0);
    let bowl = bowl_points(THIN);
    let lots = first_touch_order(&plain, &hand, o.opening, &bowl, 0.0);
    assert!(
        lots.iter().any(|(_, lot)| lot.out_of_order.is_some()),
        "{lots:?}"
    );
    // A late grasp comes before a clean one of a later turn too, which stays where it is: grasp
    // for grasp, each lot's turns are those that came there without it.
    assert!(
        lots.iter().any(|(_, lot)| lot.late_before_a_later_turn),
        "{lots:?}"
    );
    let turns = |found: &[Grasp]| {
        first_touch_order(found, &hand, o.opening, &bowl, 0.0).map(|(_, lot)| lot.steps)
    };
    assert_eq!(turns(&found), turns(&plain), "from above before a turn");
    // Every grasp of two seen sides of this bowl closes onto its patches first; of one, level
    // grasps across the leaning wall are among the late. Fingers that may come in narrower are
    // swept from where they come in.
    for narrow in [false, true] {
        let o = GraspOptions { narrow, ..o };
        let [(_, both), (_, one)] =
            first_touch_order(&grasps(&seen, &o), &hand, o.opening, &bowl, 0.0);
        assert!(both.clean > 0 && both.out_of_order.is_none(), "{both:?}");
        let narrower = !narrow || (one.narrower_clean > 0 && one.narrower_late > 0);
        assert!(
            one.clean > 0 && one.late_level > 0 && one.out_of_order.is_none() && narrower,
            "narrow {narrow}: {one:?}"
        );
    }
    // Two plates seen whole and a ledge standing out 5 mm off one along half its length, as a lip
    // over a patch: of these grasps of two seen sides, those whose finger meets the ledge come after
    // those whose fingers meet the plates first. The patches stand in cells a patch wide: of the
    // fingers that pass near the ledge's edges, only those farther than half a cell's diagonal
    // in it or out of it are held to it.
    let mut ledged = plates_of(0.02, 0.02);
    let ledge = Vec3::new(0.0225, 0.01, 0.01);
    face(
        ledge,
        Vec3::Z,
        [Vec3::X, Vec3::Y],
        [0.0075, 0.01],
        &mut ledged,
    );
    the_object(&mut ledged);
    let found = grasps(&ledged, &o);
    let cell = o.contact * 3f32.sqrt() / 2.0;
    let [(_, both), (_, one)] = first_touch_order(&found, &hand, o.opening, &ledged.points, cell);
    assert!(
        both.clean > 0 && both.late > 0 && both.out_of_order.is_none() && one.clean + one.late == 0,
        "{both:?}"
    );
}

/// Of one lot of grasps: how many meet what they close on no more than a surface's noise past
/// their patches (`clean`), how many meet it more than two patches' radius past (`late`), how
/// many level ones (`late_level`), of those less open than the hand, how many of each kind, the
/// first late grasp that comes before a clean one of its own turn, with how far past each meets
/// it (millimeters), whether a late grasp comes before a clean one of a later turn, and how many
/// steps round its closing line each grasp's hand is turned from above (`steps`).
#[derive(Debug)]
struct Lot {
    clean: usize,
    late: usize,
    late_level: usize,
    narrower_clean: usize,
    narrower_late: usize,
    out_of_order: Option<(usize, f32, usize, f32)>,
    late_before_a_later_turn: bool,
    steps: Vec<usize>,
}

/// How many steps round its closing line, of the default `turns` to a round, `g`'s hand is turned
/// from above, to either side: 0 from above, and where the line is upright.
fn steps_from_above(g: &Grasp) -> usize {
    let (closing, toward) = (g.pose.rotation * Vec3::Y, g.pose.rotation * Vec3::Z);
    let above = Vec3::NEG_Z.reject_from_normalized(closing).try_normalize();
    let step = TAU / GraspOptions::default().turns as f32;
    above.map_or(0, |above| {
        (toward.angle_between(above) / step).round() as usize
    })
}

/// `found`'s two lots, two seen sides and one, each as a [`Lot`] held to the true shape's `points`,
/// what was not seen of it too: clean where the fingers' spheres, each `tolerance` wider, meet
/// nothing past, late where they meet it each `tolerance` less wide.
fn first_touch_order(
    found: &[Grasp],
    hand: &[HandSphere],
    gap: f32,
    points: &[Vec3],
    tolerance: f32,
) -> [(bool, Lot); 2] {
    let (noise, late) = (0.0025, 2.0 * GraspOptions::default().contact);
    let level = |g: &Grasp| (g.pose.rotation * Vec3::Y).z.abs() < 1e-4;
    [false, true].map(|one_sided| {
        let lot: Vec<&Grasp> = found.iter().filter(|g| g.one_sided == one_sided).collect();
        let met = |inset: f32| -> Vec<f32> {
            let met = |g: &&Grasp| met_past_the_patches(g, hand, gap, points, inset);
            lot.iter().map(met).collect()
        };
        let (touched, into) = (met(-tolerance), met(tolerance));
        let is_clean: Vec<bool> = touched.iter().map(|&past| past <= noise).collect();
        let is_late: Vec<bool> = into.iter().map(|&past| past > late).collect();
        let count = |keep: &dyn Fn(&Grasp, bool, bool) -> bool| {
            let each = lot.iter().zip(is_clean.iter().zip(&is_late));
            each.filter(|&(g, (&clean, &late))| keep(g, clean, late))
                .count()
        };
        // A lot comes turn by turn: from above, then a step to one side, a step to the other, two
        // steps to one side, and so on. So a run of grasps as many steps from above is one turn,
        // or the two to either side one after the other, parted somewhere.
        let steps: Vec<usize> = lot.iter().map(|g| steps_from_above(g)).collect();
        let disordered = |from: usize, to: usize| {
            let first = (from..to).find(|&i| is_late[i])?;
            let clean = (from..to).rfind(|&i| is_clean[i])?;
            (first < clean).then(|| (first, into[first] * 1e3, clean, touched[clean] * 1e3))
        };
        let mut out_of_order = None;
        let mut from = 0;
        while from < lot.len() {
            let next = (from..lot.len()).find(|&i| steps[i] != steps[from]);
            let to = next.unwrap_or(lot.len());
            let one_turn = steps[from] == 0 || 2 * steps[from] == GraspOptions::default().turns;
            let mut parts = if one_turn { to..=to } else { from..=to };
            if !parts.any(|at| disordered(from, at).is_none() && disordered(at, to).is_none()) {
                out_of_order = out_of_order.or_else(|| disordered(from, to));
            }
            from = to;
        }
        let late_before_a_later_turn = (0..lot.len())
            .any(|i| is_late[i] && (i..lot.len()).any(|j| is_clean[j] && steps[j] > steps[i]));
        let narrower = |g: &Grasp| g.opening < gap;
        let lot = Lot {
            clean: count(&|_, clean, _| clean),
            late: count(&|_, _, late| late),
            late_level: count(&|g, _, late| level(g) && late),
            narrower_clean: count(&|g, clean, _| narrower(g) && clean),
            narrower_late: count(&|g, _, late| narrower(g) && late),
            out_of_order,
            late_before_a_later_turn,
            steps,
        };
        (one_sided, lot)
    })
}

#[test]
fn nothing_is_assumed_behind_a_solid_or_a_wall_thicker_than_the_shell() {
    // A box seen from above alone: no ray goes into it, and none past its sides within the
    // opening, so nothing is assumed behind any of it.
    let box_seen = box_on_a_table(ABOVE);
    let block = Obstacle::cuboid(Vec3::new(0.0, 0.0, 0.025), Vec3::new(0.02, 0.03, 0.025));
    assert!(grasps(&box_seen, &with_shell(0.015)).is_empty());
    assert!(grasps(&box_seen, &with_shell(0.08)).is_empty());
    // Seen from in front and over it, past its far side within the opening: closed on across
    // the whole of it, its far contact out of it, and the hand past that contact, where the far
    // finger stands in space the camera saw past, clear of it. (A finger may be in what it closes
    // on, and of the faces no camera saw the hand knows nothing.)
    let front = box_on_a_table(Vec3::new(0.35, 0.0, 0.35));
    let o = with_shell(0.08);
    let found = grasps(&front, &o);
    let whole: Vec<&Grasp> = found.iter().filter(|g| g.one_sided).collect();
    assert!(whole.len() > 100, "{}", whole.len());
    let hand = hand();
    let mut far_side = 0;
    for g in &whole {
        assert!(
            g.width() > 0.04 && g.width() <= o.opening,
            "across all of it: {g:?}"
        );
        let [a, b] = g.contacts.map(|c| block.distance(c).value);
        let (seen, unseen) = if a < b {
            (g.contacts[0], g.contacts[1])
        } else {
            (g.contacts[1], g.contacts[0])
        };
        assert!(a.min(b).abs() < 1e-3 && a.max(b) > 0.0, "{g:?}");
        let line = (unseen - seen).normalize();
        let beyond = |&(center, _): &(Vec3, f32)| (center - unseen).dot(line) > 0.0;
        for (center, radius) in hand_at(&hand, g).filter(beyond) {
            assert!(block.distance(center).value > radius - 1e-3, "{g:?}");
            far_side += 1;
        }
    }
    assert!(far_side > whole.len(), "{far_side}");
    // A wall is never taken thicker than the fingers open: a shell past it is one at it, an
    // infinite one too.
    let past = grasps(&front, &with_shell(0.2));
    assert_eq!(past, grasps(&front, &with_shell(o.opening)));
    assert_eq!(past, grasps(&front, &with_shell(f32::INFINITY)));
    assert!(
        found.iter().any(|g| g.width() > o.opening - 1e-3),
        "some reach the opening"
    );
    // A wall 2 cm thick seen from above: thicker than a shell of 1.5 cm, not than one of 3.
    let thick = bowl_seen(0.02, &[ABOVE]);
    assert!(grasps(&thick, &with_shell(0.015)).is_empty());
    let across = grasps(&thick, &with_shell(0.03));
    assert!(!across.is_empty());
    assert!(across.iter().all(|g| g.width() > 0.02), "across the wall");
    // And the thin wall's other side shows only past where the rim hides it: no grasp within
    // 4 mm.
    assert!(grasps(&bowl_seen(THIN, &[ABOVE]), &with_shell(0.004)).is_empty());
}

/// By default a wall seen from one side is closed on as thick as the fingers open, however wide
/// that is: the rim seen from above and the box seen from in front, which have no grasp with the
/// shell turned off, have those of a shell at the opening, and at an opening of 10 cm more than
/// those of a shell of 8 cm.
#[test]
fn by_default_a_wall_seen_from_one_side_is_closed_on_as_thick_as_the_fingers_open() {
    let rim = bowl_seen(THIN, &[ABOVE]);
    let front = box_on_a_table(Vec3::new(0.35, 0.0, 0.35));
    for opening in [0.08, 0.1] {
        let o = GraspOptions {
            opening,
            ..Default::default()
        };
        for seen in [&rim, &front] {
            assert!(grasps(seen, &GraspOptions { shell: None, ..o }).is_empty());
            let found = grasps(seen, &o);
            assert!(found.iter().any(|g| g.one_sided), "{opening} m");
            let at_the_opening = GraspOptions {
                shell: Some(opening),
                ..o
            };
            assert_eq!(found, grasps(seen, &at_the_opening), "{opening} m");
        }
    }
    let wide = GraspOptions {
        opening: 0.1,
        ..Default::default()
    };
    let eight = GraspOptions {
        shell: Some(0.08),
        ..wide
    };
    assert!(grasps(&front, &wide).len() > grasps(&front, &eight).len());
}

#[test]
fn looks_joined_from_one_place_are_one_view_however_many() {
    // A box past which the camera sees, its look cropped to around the box, and that look
    // joined with itself 100 and 400 times, as a harness that keeps looking from one place
    // would join them.
    let whole = box_on_a_table(Vec3::new(0.35, 0.0, 0.35));
    let near = |i: &usize| whole.points[*i].truncate().abs().max_element() < 0.12;
    let kept: Vec<usize> = (0..whole.points.len()).filter(near).collect();
    let look = Surface {
        points: kept.iter().map(|&i| whole.points[i]).collect(),
        normals: kept.iter().map(|&i| whole.normals[i]).collect(),
        object: kept.iter().map(|&i| whole.object[i]).collect(),
        seen_from: kept.iter().map(|&i| whole.seen_from[i]).collect(),
    };
    let o = with_shell(0.08);
    let one_sided = |s: &Surface| -> Vec<f32> {
        let found = grasps(s, &o);
        let mut widths: Vec<f32> = found
            .iter()
            .filter(|g| g.one_sided)
            .map(Grasp::width)
            .collect();
        widths.sort_by(f32::total_cmp);
        widths
    };
    let once = one_sided(&look);
    assert!(
        once.len() > 100,
        "{} of {} points",
        once.len(),
        look.points.len()
    );
    for k in [100, 400] {
        let mut looks = Surface::default();
        for _ in 0..k {
            looks.join(look.clone());
        }
        // As many, as wide but for how a patch's mean over k copies of its points rounds.
        let found = one_sided(&looks);
        assert_eq!(found.len(), once.len(), "{k} looks");
        let apart = found
            .iter()
            .zip(&once)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(apart < 1e-6, "{k} looks: widths {apart} m apart");
    }
}

#[test]
fn two_looks_from_one_place_show_what_one_look_does() {
    // The bowl seen from in front and over it, and a box past which the camera sees: once, and
    // joined with a second look from the same place whose readings are a little farther, as a
    // harness that looks twice gives them. The two looks' readings of a pixel lie along one ray.
    let one_sided = |s: &Surface, o: &GraspOptions| -> Vec<f32> {
        let found = grasps(s, o);
        let mut widths: Vec<f32> = found
            .iter()
            .filter(|g| g.one_sided)
            .map(Grasp::width)
            .collect();
        widths.sort_by(f32::total_cmp);
        widths
    };
    let o = with_shell(0.015);
    let once = one_sided(&bowl_seen(THIN, &[FRONT]), &o);
    assert!(once.len() > 100, "{}", once.len());
    let median = |widths: &[f32]| widths[widths.len() / 2];
    for farther in [0.0, 1e-4, 1e-3] {
        let mut twice = bowl_seen(THIN, &[FRONT]);
        twice.join(bowl_read_farther(THIN, &[FRONT], farther));
        let found = one_sided(&twice, &o);
        // None fewer: within a surface's noise of the first look, as many; a millimeter
        // farther, the second look's points are a layer of their own and seed more.
        let (n, m) = (once.len(), found.len());
        assert!(m * 20 >= n * 19, "{m} against {n}, {farther} m farther");
        if farther < 1e-3 {
            assert!(m * 20 <= n * 21, "{m} against {n}, {farther} m farther");
        }
        // As wide, but for how much farther the second look reads and a step of the search for
        // where the wall ends (half a millimeter).
        let apart = (median(&found) - median(&once)).abs();
        assert!(
            apart <= farther + 5e-4 + 1e-6,
            "{apart} m, {farther} m farther"
        );
    }
    // The box's faces lie on the edges of the cells seeds are drawn from, so a second look's
    // seeds fall in cells of their own: none fewer, each across the whole box.
    let eye = Vec3::new(0.35, 0.0, 0.35);
    let o = with_shell(0.08);
    let once = one_sided(&box_on_a_table(eye), &o);
    assert!(once.len() > 100, "{}", once.len());
    for farther in [0.0, 1e-4, 1e-3] {
        let mut twice = box_on_a_table(eye);
        twice.join(box_read_farther(eye, farther));
        let found = one_sided(&twice, &o);
        let (n, m) = (once.len(), found.len());
        assert!(m * 20 >= n * 19, "{m} against {n}, {farther} m farther");
        assert!(found[0] > 0.04, "{} m", found[0]);
    }
}

#[test]
fn walls_seen_from_one_side_come_after_those_seen_from_both_and_change_nothing_else() {
    // From in front and over it, the near wall is seen from both sides, the far wall's outside
    // not at all.
    let both = bowl_seen(THIN, &[FRONT, ABOVE]);
    let without = grasps(&both, &without_shell());
    let with = grasps(&both, &with_shell(0.015));
    assert!(!without.is_empty() && with.len() > without.len());
    assert_eq!(with[..without.len()], without[..]);
    // Each says whether one of its contacts was not seen.
    assert!(without.iter().all(|g| !g.one_sided));
    assert!(with[without.len()..].iter().all(|g| g.one_sided));
    // From four sides, where each camera sees both sides of the wall in front of it the shell
    // adds nothing: only between them, where too little of the outside was seen to press on.
    let turned = |k: usize| Quat::from_rotation_z(k as f32 * TAU / 4.0) * FRONT;
    let around = bowl_seen(THIN, &[turned(0), turned(1), turned(2), turned(3)]);
    let without = grasps(&around, &without_shell());
    let with = grasps(&around, &with_shell(0.015));
    assert!(!without.is_empty() && with.len() > without.len());
    assert_eq!(with[..without.len()], without[..]);
    for g in &with[without.len()..] {
        let middle = (g.contacts[0] + g.contacts[1]) / 2.0;
        let from_a_camera = middle.y.atan2(middle.x).rem_euclid(TAU / 4.0);
        let from_a_camera = from_a_camera.min(TAU / 4.0 - from_a_camera);
        assert!(from_a_camera > 30f32.to_radians(), "{g:?}");
    }
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
fn readings_of_the_robot_are_not_surfaces() {
    // The hand's spheres as the scene, seen from in front of the robot: with its mask, the image
    // shows nothing of them, and the table behind is still seen.
    let robot = common::panda().unwrap();
    let q = robot.default_q().to_vec();
    let hand = robot.ee_pose(&q, 0).position;
    let spheres: Vec<[f32; 4]> = robot
        .spheres(&q)
        .into_iter()
        .filter(|s| Vec3::new(s[0], s[1], s[2]).distance(hand) < 0.12)
        .collect();
    assert!(spheres.len() > 10);
    let mut scene: Vec<Obstacle> = spheres
        .iter()
        .map(|s| Obstacle::Sphere {
            center: Vec3::new(s[0], s[1], s[2]),
            radius: s[3],
        })
        .collect();
    scene.push(Obstacle::cuboid(
        hand - Vec3::Z * 0.3,
        Vec3::new(0.5, 0.5, 0.01),
    ));
    let camera = looking(hand + Vec3::new(0.4, 0.0, 0.3), hand);
    let depth = render(&scene, camera);
    let image = DepthImage {
        depth: &depth,
        width: WIDTH,
        intrinsics: K,
        camera,
    };
    let on_the_robot = |surface: &Surface| {
        let near = |p: &&Vec3| {
            spheres
                .iter()
                .any(|s| p.distance(Vec3::new(s[0], s[1], s[2])) < s[3] + 1e-3)
        };
        surface.points.iter().filter(near).count()
    };
    let unmasked = Surface::from_depth(&device(), &image, &[], None, 0.005).unwrap();
    assert!(on_the_robot(&unmasked) > 500, "the camera sees the hand");
    let mask = image.robot_mask(&robot, &q, 0.005, &[]).unwrap();
    let masked = Surface::from_depth(&device(), &image, &[], Some(&mask), 0.005).unwrap();
    assert_eq!(on_the_robot(&masked), 0);
    assert!(masked.points.len() > 1000, "the table is still seen");
    assert!(masked.object.is_empty() && masked.normals.iter().any(|&n| n != Vec3::ZERO));
    // Object flags for some pixels only, and a radius that is not a length, are refused.
    for (object, radius) in [(&[true][..], 0.005), (&[][..], 0.0), (&[][..], f32::NAN)] {
        assert!(matches!(
            Surface::from_depth(&device(), &image, object, None, radius),
            Err(Error::Input(_))
        ));
    }
}

/// An image's rows are read on the device's threads: on one thread and on all of them the
/// surface is the same, point for point in the image's order, with or without object flags.
#[test]
fn a_surface_reads_the_same_on_one_thread_and_on_many() {
    let robot = common::panda().unwrap();
    let (one, many) = (Device::cpu_threads(&robot, 1).unwrap(), device());
    // A wall with a step across it and readings missing in a band, noise on every reading.
    let mut rng = Rng::new(3);
    let depth: Vec<f32> = (0..WIDTH * HEIGHT)
        .map(|i| {
            let (u, v) = (i % WIDTH, i / WIDTH);
            let wall = if u < WIDTH / 2 { 0.4 } else { 0.5 };
            let noise = rng.range(-0.0005, 0.0005);
            if (HEIGHT / 3..HEIGHT / 3 + 9).contains(&v) {
                0.0
            } else {
                wall + noise
            }
        })
        .collect();
    let image = DepthImage {
        depth: &depth,
        width: WIDTH,
        intrinsics: K,
        camera: Pose::new(Vec3::new(0.1, -0.2, 0.3), Quat::from_rotation_x(0.3)),
    };
    let object: Vec<bool> = (0..depth.len()).map(|i| i % 3 != 0).collect();
    for flags in [&[][..], &object[..]] {
        let alone = Surface::from_depth(&one, &image, flags, None, 0.005).unwrap();
        let together = Surface::from_depth(&many, &image, flags, None, 0.005).unwrap();
        assert!(alone == together, "the surfaces differ");
        assert_eq!(alone.points.len(), WIDTH * (HEIGHT - 9));
        assert_eq!(alone.object.len(), flags.len().min(alone.points.len()));
        // In the image's order: each point is its pixel's reading.
        let read = image.points(None).unwrap();
        let read: Vec<Vec3> = read.into_iter().flatten().collect();
        assert!(
            alone.points == read,
            "the points are not in the image's order"
        );
        let normals = alone.normals.iter().filter(|n| **n != Vec3::ZERO).count();
        assert!(normals > WIDTH * HEIGHT / 2, "{normals} normals");
    }
}

#[test]
fn a_normal_is_the_plane_through_every_reading_within_the_radius() {
    // A wall half a meter ahead, read with a millimeter of noise: over the 5 mm around a
    // reading, six pixels here, the noise averages out; over its nearest pixels alone it
    // would not.
    let camera = Pose::new(Vec3::ZERO, Quat::IDENTITY);
    let mut rng = Rng::new(7);
    let depth: Vec<f32> = (0..WIDTH * HEIGHT)
        .map(|_| 0.5 + rng.range(-0.001, 0.001))
        .collect();
    let image = DepthImage {
        depth: &depth,
        width: WIDTH,
        intrinsics: K,
        camera,
    };
    let seen = Surface::from_depth(&device(), &image, &[], None, 0.005).unwrap();
    assert_eq!(seen.points.len(), WIDTH * HEIGHT);
    let known: Vec<f32> = seen
        .normals
        .iter()
        .filter(|&&n| n != Vec3::ZERO)
        .map(|n| n.dot(Vec3::NEG_Z).clamp(-1.0, 1.0).acos())
        .collect();
    assert!(
        known.len() * 10 > 9 * WIDTH * HEIGHT,
        "{} with a normal",
        known.len()
    );
    let off = known.iter().sum::<f32>() / known.len() as f32;
    assert!(off < 0.12, "{off} rad off the wall's normal on average");
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
    // Flags for some of a surface's points only are not made up for by joining it: refused
    // alone, it is refused joined.
    let mut partly = plates(0.04);
    partly.object = vec![false];
    let mut joined = Surface::default();
    joined.join(partly);
    assert_eq!(joined.object.len(), 1);
    let (device, o) = (device(), GraspOptions::default());
    assert!(matches!(
        grasps_on(&device, &gripper(&device, &o), &[joined], &o),
        Err(Error::Input(_))
    ));
    marked.join(plates(0.03));
    assert!(
        marked.object[..count].iter().all(|&o| !o) && marked.object[count..].iter().all(|&o| o)
    );
    assert_eq!(marked.normals.len(), 2 * count);
    assert!(marked.seen_from.is_empty(), "nowhere known, as before");
    // Where the points were seen from, kept where known and not known (NaN) where not.
    let eye = Vec3::new(0.3, 0.0, 0.3);
    let mut seen = Surface::from_points(&plates(0.04).points, eye, 0.005).unwrap();
    assert_eq!(seen.seen_from, vec![eye; count]);
    seen.join(plates(0.03));
    let mut unseen = plates(0.03);
    unseen.join(Surface::from_points(&plates(0.04).points, -eye, 0.005).unwrap());
    for (joined, known) in [(&seen, 0..count), (&unseen, count..2 * count)] {
        assert_eq!(joined.seen_from.len(), 2 * count);
        for (k, from) in joined.seen_from.iter().enumerate() {
            assert_eq!(known.contains(&k), from.is_finite(), "{k}: {from}");
        }
    }
    assert!(seen.seen_from[0] == eye && unseen.seen_from[count] == -eye);
    // What a side did not say is not known, and a joined surface is still equal to itself.
    assert_eq!(seen, seen.clone());
    assert_eq!(unseen, unseen.clone());
}

#[test]
fn grasps_do_not_depend_on_the_threads_the_batch_or_the_device() {
    let robot = common::panda().unwrap();
    let o = GraspOptions::default();
    let surfaces = [
        tube_surface(0.055, 0.047, 0.025),
        box_surface(Vec3::new(0.02, 0.03, 0.06), &[]),
        Surface::default(),
        plates(0.04),
    ];
    let one = Device::cpu_threads(&robot, 1).unwrap();
    let hand = gripper(&one, &o);
    let alone: Vec<Vec<Grasp>> = surfaces
        .iter()
        .map(|s| {
            grasps_on(&one, &hand, std::slice::from_ref(s), &o)
                .unwrap()
                .remove(0)
        })
        .collect();
    assert!(alone[0].len() > 100 && alone[1].len() > 100 && alone[2].is_empty());
    let mut devices = vec![one, Device::cpu(&robot).unwrap()];
    match Device::gpu(&robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    for device in &devices {
        let together = grasps_on(device, &hand, &surfaces, &o).unwrap();
        assert_eq!(together, alone, "{}", device.name());
    }
    // Walls seen from one side too.
    let o = with_shell(0.015);
    let surfaces = [
        bowl_seen(THIN, &[ABOVE]),
        surfaces[1].clone(),
        bowl_seen(THIN, &[FRONT]),
        box_on_a_table(ABOVE),
    ];
    let alone: Vec<Vec<Grasp>> = surfaces
        .iter()
        .map(|s| {
            grasps_on(&devices[0], &hand, std::slice::from_ref(s), &o)
                .unwrap()
                .remove(0)
        })
        .collect();
    assert!(alone[0].len() > 100 && alone[2].len() > 100);
    for device in &devices {
        let together = grasps_on(device, &hand, &surfaces, &o).unwrap();
        assert_eq!(together, alone, "{}", device.name());
    }
    // And fingers that come in narrower, as some of these do.
    let o = GraspOptions { narrow: true, ..o };
    let alone: Vec<Vec<Grasp>> = surfaces
        .iter()
        .map(|s| {
            grasps_on(&devices[0], &hand, std::slice::from_ref(s), &o)
                .unwrap()
                .remove(0)
        })
        .collect();
    assert!(alone.iter().flatten().any(|g| g.opening < o.opening));
    for device in &devices {
        let together = grasps_on(device, &hand, &surfaces, &o).unwrap();
        assert_eq!(together, alone, "{}", device.name());
    }
}

#[test]
fn the_first_grasps_are_spread_over_the_object() {
    // Around a bowl's rim, the first eight grasps of its wall are from above, each well away
    // from the others, in most of its eight eighths.
    let (outer, inner, half_height) = (0.055, 0.047, 0.025);
    let found = grasps(
        &tube_surface(outer, inner, half_height),
        &GraspOptions::default(),
    );
    let across: Vec<&Grasp> = found
        .iter()
        .filter(|g| (g.width() - (outer - inner)).abs() < 1e-3)
        .step_by(2)
        .collect();
    let eighth = |g: &&Grasp| {
        let angle = g.pose.position.y.atan2(g.pose.position.x).rem_euclid(TAU);
        (angle / TAU * 8.0) as usize % 8
    };
    let mut first: Vec<usize> = across[..8].iter().map(eighth).collect();
    first.sort_unstable();
    first.dedup();
    assert!(first.len() >= 6, "the first eight stand in {first:?}");
    for (k, a) in across[..8].iter().enumerate() {
        assert!(
            (a.pose.rotation * Vec3::Z).dot(Vec3::NEG_Z) > 0.9999,
            "{a:?}"
        );
        for b in &across[..k] {
            assert!(
                a.pose.position.distance(b.pose.position) > 0.02,
                "{a:?} {b:?}"
            );
        }
    }
}

#[test]
fn malformed_surfaces_and_options_are_input_errors() {
    let device = device();
    let good = box_surface(Vec3::splat(0.02), &[]);
    let o = GraspOptions::default();
    let refused = |s: &Surface, hand: usize, o: &GraspOptions| {
        let gripper = device.robot().gripper(hand, &calibration(o));
        matches!(
            gripper.and_then(|g| grasps_on(&device, &g, std::slice::from_ref(s), o)),
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
    let mut places = good.clone();
    places.seen_from = vec![Vec3::Z];
    assert!(
        refused(&places, 0, &o),
        "where some points only were seen from"
    );
    // Where a point was seen from may be unknown.
    let mut unknown = good.clone();
    unknown.seen_from = vec![Vec3::NAN; good.points.len()];
    assert!(!refused(&unknown, 0, &with_shell(0.01)));
    for bad in [
        GraspOptions { friction: 0.0, ..o },
        GraspOptions { contact: 0.0, ..o },
        GraspOptions { overlap: 0.0, ..o },
        GraspOptions {
            overlap: f32::NEG_INFINITY,
            ..o
        },
        GraspOptions {
            floor: Some(f32::NAN),
            ..o
        },
        GraspOptions {
            approach: -0.1,
            ..o
        },
        GraspOptions {
            opening: f32::INFINITY,
            ..o
        },
        GraspOptions { turns: 0, ..o },
        GraspOptions {
            near: Some(Vec3::new(0.0, f32::NAN, 0.0)),
            ..o
        },
        with_shell(0.0),
        with_shell(-0.01),
        with_shell(f32::NAN),
        with_shell(f32::NEG_INFINITY),
    ] {
        assert!(refused(&good, 0, &bad), "{bad:?}");
    }
    // No surfaces, and a surface without points, give no grasps.
    assert!(
        grasps_on(&device, &gripper(&device, &o), &[], &o)
            .unwrap()
            .is_empty()
    );
    assert!(grasps(&Surface::default(), &o).is_empty());
}

/// How many turns from above a grasp comes, of `turns` round its closing line, either way.
fn turns_from_above(g: &Grasp, turns: usize) -> usize {
    let toward = g.pose.rotation * Vec3::Z;
    let angle = toward.angle_between(Vec3::NEG_Z);
    (angle / (TAU / turns as f32)).round() as usize
}

/// The middle between the two points a grasp closes on.
fn middle(g: &Grasp) -> Vec3 {
    (g.contacts[0] + g.contacts[1]) / 2.0
}

#[test]
fn grasps_near_a_point_come_nearest_first_turn_by_turn() {
    // A bar 2 cm wide, 16 cm long and 10 cm tall: grasped across its width alone, all along it.
    let half = Vec3::new(0.08, 0.01, 0.05);
    let s = box_surface(half, &[]);
    let o = GraspOptions::default();
    let spread = grasps(&s, &o);
    assert!(spread.len() > 4 * o.turns, "{}", spread.len());
    let point = Vec3::new(0.05, 0.0, 0.05);
    let near = grasps(
        &s,
        &GraspOptions {
            near: Some(point),
            ..o
        },
    );
    // The same grasps, in another order.
    let key = |g: &Grasp| {
        let (p, r) = (g.pose.position, g.pose.rotation);
        [
            p.x,
            p.y,
            p.z,
            r.x,
            r.y,
            r.z,
            r.w,
            g.contacts[0].x,
            g.contacts[1].x,
        ]
    };
    let sorted = |found: &[Grasp]| {
        let mut found = found.to_vec();
        found.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
        found
    };
    assert_eq!(sorted(&spread), sorted(&near));
    assert_ne!(spread, near);
    // Turn by turn as without the point, from above first, and within a turn the pairs nearest
    // it first. From outside, a turn to one side is told from one as far to the other only by
    // which way round its pair was found, so the tier of a turn either way may start over once.
    let tiers = |found: &[Grasp]| {
        let each = found.iter().map(|g| turns_from_above(g, o.turns));
        each.collect::<Vec<usize>>()
    };
    assert_eq!(tiers(&spread), tiers(&near));
    assert_eq!(turns_from_above(&near[0], o.turns), 0);
    let tier = tiers(&near);
    let mut start = 0;
    while start < near.len() {
        let end = (start..near.len()).find(|&k| tier[k] != tier[start]);
        let end = end.unwrap_or(near.len());
        let run = &near[start..end];
        let over = run
            .windows(2)
            .filter(|w| middle(&w[0]).distance(point) > middle(&w[1]).distance(point) + 1e-6);
        let sides = if tier[start] == 0 || 2 * tier[start] == o.turns {
            0
        } else {
            1
        };
        assert!(over.count() <= sides, "turn {} from above", tier[start]);
        start = end;
    }
    // The first grasps lie at the point, where without it they spread along the bar.
    assert!(middle(&near[0]).truncate().distance(point.truncate()) < 2.0 * SPACING);
    let far = |found: &[Grasp]| {
        let first = found.iter().take(2 * 4).map(middle);
        first.map(|m| m.x).fold(f32::NEG_INFINITY, f32::max)
            - found
                .iter()
                .take(2 * 4)
                .map(|g| middle(g).x)
                .fold(f32::INFINITY, f32::min)
    };
    assert!(
        far(&spread) > 0.08 && far(&near) < 0.02,
        "{} and {}",
        far(&spread),
        far(&near)
    );
}

#[test]
fn a_grasps_moment_about_a_line_is_what_friction_its_patches_and_a_push_give() {
    // A knob's bar 2.5 cm wide, 9 cm long and 3 cm tall, its axis upright through its middle:
    // grasped across its width, the fingertips (friction 0.5, patches 5 mm) give it 0.5 x 1.25 cm
    // each by friction and 5 mm each by where they press on their patches, 2.25 cm a newton,
    // wherever along the bar; off the axis, the hand pushing with up to both fingers' force gives
    // twice as much again as the grip is far from the axis.
    let half = Vec3::new(0.045, 0.0125, 0.015);
    let o = GraspOptions::default();
    let found = grasps(&box_surface(half, &[]), &o);
    for screw in [Some(Screw::about(Vec3::ZERO, Vec3::Z)), None] {
        let mut ranked = found.clone();
        o.order_for_turn(&mut ranked, screw);
        let moments: Vec<_> = ranked
            .iter()
            .map(|g| {
                let axis = screw
                    .unwrap_or_else(|| Screw::about(g.pose.position, g.pose.rotation * Vec3::Z));
                g.moment(&axis, &o)
            })
            .collect();
        assert!(
            moments[0] > *moments.last().unwrap(),
            "the bar has grips with unequal leverage"
        );
        assert!(
            moments.windows(2).all(|m| m[0] >= m[1]),
            "strongest grip first"
        );
    }
    let across: Vec<&Grasp> = found
        .iter()
        .filter(|g| (g.pose.rotation * Vec3::Y).y.abs() > 0.9999)
        .collect();
    // The axis through the middle of the grasps nearest the bar's middle.
    let x = |g: &&Grasp| middle(g).x;
    let nearest = across
        .iter()
        .map(x)
        .min_by(|a, b| a.abs().total_cmp(&b.abs()));
    let axis = Screw::about(Vec3::X * nearest.expect("grasps across the bar"), Vec3::Z);
    let (mut at_axis, mut off_axis) = (0, 0);
    for g in &across {
        let off = middle(g).x - axis.point.x;
        let moment = g.moment(&axis, &o);
        let expected = 0.5 * 0.025 + 2.0 * 0.005 + 2.0 * off.abs();
        assert!((moment - expected).abs() < 1e-5, "{moment} for {g:?}");
        if off.abs() < 1e-5 {
            assert!((moment - 0.0225).abs() < 1e-6, "{moment}");
            at_axis += 1;
        }
        if (off.abs() - 0.03).abs() < 1e-5 {
            assert!((moment - 0.0825).abs() < 1e-5, "{moment}");
            off_axis += 1;
        }
    }
    assert!(at_axis > 0 && off_axis > 0, "{at_axis} and {off_axis}");
    // Closing along the line: the patches and the push give nothing, and friction gives each
    // fingertip's slide across the line at its distance from it and its twist about the line, a
    // patch pressed evenly twisting by 2/3 of friction times its radius.
    let plates = plates_of(0.03, 0.02);
    for g in grasps(&plates, &o) {
        let line = g.pose.rotation * Vec3::Y;
        let along = Screw::about(Vec3::new(0.0, 0.01, -0.02), line);
        let from = (middle(&g) - along.point).reject_from(line).length();
        let twist = 2.0 / 3.0 * o.friction * o.contact;
        let expected = 2.0 * o.friction * from + 2.0 * twist;
        assert!((g.moment(&along, &o) - expected).abs() < 1e-5, "{g:?}");
    }
    // An axis 5 mm across the bar from the middle of the grips: friction can give the farther
    // finger's slide 5 mm more, which a push along the closing line works against one way round,
    // and the moment is that way's: 2.25 cm and | 2 x 3 cm - 0.5 cm | 3 cm along the bar.
    let aside = Screw::about(axis.point + Vec3::Y * 0.005, Vec3::Z);
    for g in &across {
        let off = middle(g).x - axis.point.x;
        let expected = 0.0225 + (2.0 * off.abs() - 0.005).abs();
        assert!((g.moment(&aside, &o) - expected).abs() < 1e-5, "{g:?}");
    }
    // Either way round the line alike.
    let reversed = Screw::about(axis.point, Vec3::NEG_Z);
    for g in &across {
        assert!((g.moment(&reversed, &o) - g.moment(&axis, &o)).abs() < 1e-6);
    }
}

#[test]
fn the_grasp_closing_region_uses_a_rigid_frame() {
    let robot = common::panda().unwrap();
    let hand = robot
        .gripper(0, &calibration(&GraspOptions::default()))
        .unwrap();
    let mut surface = Surface::default();
    for side in [-1.0, 1.0] {
        face(
            Vec3::new(0.0, side * 0.01, 0.10),
            side * Vec3::Y,
            [Vec3::X, Vec3::Z],
            [0.012; 2],
            &mut surface,
        );
    }
    let grasp = grasps(&surface, &GraspOptions::default())[0];
    let mut rounded = grasp;
    rounded.pose.rotation *= 0.9992;
    assert!(rounded.pose.is_unit());
    // A ball grazing the side of a Panda pad at the open grasp.
    let local = Vec3::new(-0.00909, -0.03838, 0.00606);
    let point = grasp.pose.position + grasp.pose.rotation * local;
    assert!(!grasp.closes_on(&hand, point, 0.0008));
    assert!(!rounded.closes_on(&hand, point, 0.0008));

    let mut wider = grasp;
    wider.opening += 0.01;
    let between = (grasp.contacts[0] + grasp.contacts[1]) * 0.5;
    assert!(grasp.closes_on(&hand, between, 0.0));
    assert!(
        !wider.closes_on(&hand, between, 0.0),
        "the loaded hand cannot open to this width"
    );
}

#[test]
fn refining_approaches_for_a_point_keeps_the_original_grasps_and_closes_on_the_point() {
    let device = device();
    let surface = box_surface(Vec3::new(0.08, 0.01, 0.05), &[]);
    let hand = gripper(&device, &GraspOptions::default());
    let at = Vec3::new(0.051, 0.01, 0.049);
    let get = |o: &GraspOptions| {
        grasps_on(&device, &hand, std::slice::from_ref(&surface), o)
            .unwrap()
            .remove(0)
    };
    for narrow in [false, true] {
        let options = GraspOptions {
            turns: 4,
            narrow,
            ..Default::default()
        };
        let original = get(&options);
        let extra = get(&GraspOptions {
            close_on: Some(at),
            ..options
        });
        assert!(
            !original.is_empty(),
            "the bar has measured antipodal contacts"
        );
        assert!(
            extra.starts_with(&original),
            "existing successes keep their rank"
        );
        let added = &extra[original.len()..];
        assert!(
            !added.is_empty(),
            "intermediate rolls cover the named point"
        );
        assert!(added.iter().all(|g| g.closes_on(&hand, at, 0.0)));
        assert!(
            added.iter().all(|g| !original.contains(g)),
            "coarse rolls are not tried twice"
        );
        assert!(
            original.iter().any(|g| added.iter().any(|other| {
                let displacement = g.pose.position - other.pose.position;
                g.contacts == other.contacts
                    && g.pose.rotation == other.pose.rotation
                    && displacement.dot(g.pose.rotation * Vec3::Z) > 0.001
                    && displacement.cross(g.pose.rotation * Vec3::Z).length() < 1e-6
            })),
            "the same contact pair and roll admit a shallower insertion"
        );
        let far = get(&GraspOptions {
            close_on: Some(Vec3::splat(2.0)),
            ..options
        });
        assert_eq!(far, original, "neither pad can cover the distant point");
    }
    let invalid = GraspOptions {
        close_on: Some(Vec3::NAN),
        ..Default::default()
    };
    assert!(matches!(
        grasps_on(&device, &hand, &[surface], &invalid),
        Err(Error::Input(_))
    ));
}

#[test]
fn insertion_overlap_comes_from_both_fingers() {
    let surface = plates_of(0.045, 0.005);
    let options = GraspOptions {
        turns: 3,
        close_on: Some(Vec3::new(0.0, -0.00525, -0.01025)),
        ..Default::default()
    };
    let short = common::panda().unwrap().tip(0);
    for extended in ["panda_hand", "panda_leftfinger"] {
        let mut model = common::panda_options();
        let spheres = &mut model.collision_model.as_mut().unwrap().spheres;
        spheres
            .get_mut(extended)
            .unwrap()
            .push([0.10, 0.0, 0.16, 0.005]);
        let robot =
            batchplan::Robot::load(common::asset("franka/franka_panda.urdf"), &model).unwrap();
        let device = Device::cpu_threads(&robot, 4).unwrap();
        assert!(
            robot.tip(0) > short + 0.04,
            "the extension reaches past the ordinary fingertips"
        );
        let grasps = grasps_on(
            &device,
            &gripper(&device, &options),
            std::slice::from_ref(&surface),
            &options,
        )
        .unwrap()
        .remove(0);
        let mut alternatives = 0;
        for (i, g) in grasps.iter().enumerate() {
            if grasps[..i].iter().any(|before| {
                before.contacts == g.contacts
                    && before.pose.rotation == g.pose.rotation
                    && before.pose.position != g.pose.position
            }) {
                alternatives += 1;
                let middle = g.contacts[0].midpoint(g.contacts[1]);
                let back = (middle - g.pose.position).dot(g.pose.rotation * Vec3::Z);
                assert!(
                    short - back >= options.overlap - 1e-6,
                    "both fingers must overlap the patches"
                );
            }
        }
        assert!(
            alternatives > 0,
            "the fingers reach more than one insertion"
        );
    }
}
