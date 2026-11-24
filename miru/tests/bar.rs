//! A measured bar above a disk and surfaces missing one of those two features.

use batchplan::{Pose, Surface};
use glam::{Quat, Vec3};
use miru::BarOnDisk;

const DEPTH: f32 = 0.005;
const RESOLUTION: f32 = 0.0025;

fn sample(whole_disk: bool, split_bar: bool) -> Surface {
    let mut surface = Surface::default();
    let mut point = |x: f32, y: f32, z: f32, target: bool| {
        surface.points.push(Vec3::new(x, y, z));
        surface.object.push(target);
    };
    for i in -50..=50 {
        for j in -50_i32..=50 {
            let (x, y) = (i as f32 * 0.001, j as f32 * 0.001);
            if x.hypot(y) < 0.044 && (whole_disk || x > 0.004) {
                point(x, y, 0.0, false);
            }
            if x.abs() <= 0.032 && y.abs() <= 0.009 && (!split_bar || y.abs() > 0.004) {
                point(x, y, 0.023, true);
                if j.abs() == 8 {
                    for k in 0..23 {
                        point(x, y, k as f32 * 0.001, true);
                    }
                }
            }
        }
    }
    surface
}

#[test]
fn opposite_grips_over_a_measured_disk_follow_the_surface_frame() {
    let mut s = sample(true, false);
    let frame = Pose::new(Vec3::new(0.18, 0.09, 0.35), Quat::from_rotation_y(1.1));
    let up = frame.rotation * Vec3::Z;
    for p in &mut s.points {
        *p = frame.position + frame.rotation * *p;
    }
    let bar = BarOnDisk::measure(&s, up, DEPTH, RESOLUTION)
        .unwrap()
        .expect("the entire disk and a single raised crest were sampled");
    assert!(bar.center.distance(frame.position) < 0.001);
    assert!((bar.radius - 0.044).abs() < 0.001);
    assert!((bar.width - 0.018).abs() < 0.003);
    for g in &bar.grasps {
        assert!(g.position.distance(frame.position + up * (0.023 - DEPTH)) < 0.001);
        assert!((g.rotation * Vec3::Z).dot(up) < -0.999);
        assert!((g.rotation * Vec3::Y).dot(frame.rotation * Vec3::Y).abs() > 0.999);
    }
    assert!((bar.grasps[0].rotation * Vec3::Y).dot(bar.grasps[1].rotation * Vec3::Y) < -0.999);
}

#[test]
fn a_hidden_disk_two_bars_and_an_intervening_surface_have_no_single_grip() {
    for s in [sample(false, false), sample(true, true)] {
        assert!(
            BarOnDisk::measure(&s, Vec3::Z, DEPTH, RESOLUTION)
                .unwrap()
                .is_none()
        );
    }
    let mut s = sample(true, false);
    s.points.push(Vec3::new(0.0, 0.0, 0.025));
    s.object.push(false);
    assert!(
        BarOnDisk::measure(&s, Vec3::Z, DEPTH, RESOLUTION)
            .unwrap()
            .is_none()
    );
}
