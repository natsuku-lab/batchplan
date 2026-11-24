//! Visible empty space distinguishes a scalloped plate from an intact disk.

#[allow(
    dead_code,
    reason = "this test uses the renderer and its stereo camera models"
)]
mod common;
use batchplan::{DepthFilter, Surface};
use common::{CAMERAS, EYE, Frame, MIDDLE, NEAR, Thing, block, can, device, looking};
use glam::Vec3;
use miru::BarOnDisk;
fn things(kind: u32) -> Vec<Thing> {
    let mut things = vec![
        block(
            MIDDLE - Vec3::Z * 0.004,
            Vec3::new(0.12, 0.12, 0.004),
            [80; 3],
        ),
        block(
            MIDDLE + Vec3::Z * 0.0235,
            Vec3::new(0.032, 0.009, 0.0115),
            [30; 3],
        ),
    ];
    if kind == 100 {
        things.push(block(
            MIDDLE + Vec3::Z * 0.006,
            Vec3::new(0.16, 0.16, 0.006),
            [150; 3],
        ));
    } else {
        let mut disk = can(MIDDLE + Vec3::Z * 0.006, 0.044, 0.006, [150; 3]);
        if kind > 0 {
            for i in 0..kind {
                let a = std::f32::consts::TAU * (i as f32) / (kind as f32);
                let hole = can(
                    MIDDLE + Vec3::new(0.041 * a.cos(), 0.041 * a.sin(), 0.006),
                    0.009,
                    0.012,
                    [150; 3],
                );
                disk = common::without(disk, hole);
            }
        }
        things.push(disk);
    }
    things
}
#[test]
fn a_measured_disk_requires_an_intact_face_and_a_visible_edge() {
    for pegs in [0, 12, 100] {
        let things = things(pegs);
        let raw = [EYE, NEAR].map(|eye| Frame::of(&things, looking(eye, MIDDLE + Vec3::Z * 0.02)));
        for (name, far, near) in CAMERAS {
            let mut surface = Surface::default();
            let mut widths = vec![];
            for (i, (frame, cam)) in raw.iter().zip([far, near]).enumerate() {
                let mut f = frame.clone();
                if let Some(cam) = cam {
                    f = f.sensed(cam, 12 + i as u64);
                    let n = f.noise;
                    let filter = DepthFilter {
                        noise: [n.constant, n.linear, n.quadratic],
                        ..Default::default()
                    };
                    f.depth = f.view().image.filtered(&filter).unwrap();
                }
                if pegs == 100 {
                    let k = f.intrinsics;
                    for (i, z) in f.depth.iter_mut().enumerate() {
                        let ray = Vec3::new(
                            ((i % f.width) as f32 - k.cx) / k.fx,
                            ((i / f.width) as f32 - k.cy) / k.fy,
                            1.0,
                        );
                        let point = f.camera.position + f.camera.rotation * (ray * *z);
                        if f.shows[i] == Some(2)
                            && point
                                .truncate()
                                .distance(MIDDLE.truncate() + glam::Vec2::X * 0.004)
                                > 0.044
                        {
                            *z = 0.0;
                        }
                    }
                }
                let flags: Vec<_> = f.shows.iter().map(|x| *x == Some(1)).collect();
                for (z, t) in f.depth.iter().zip(&flags) {
                    if *t && *z > 0.0 {
                        let n = f.noise;
                        widths.push(
                            (z / f.intrinsics.fx)
                                .max(n.constant + n.linear * z + n.quadratic * z * z),
                        );
                    }
                }
                surface.join(
                    Surface::from_depth(device(), &f.view().image, &flags, None, 0.005).unwrap(),
                );
            }
            widths.sort_by(f32::total_cmp);
            let tol = widths[widths.len() / 2];
            let out = BarOnDisk::measure(&surface, Vec3::Z, 0.005, tol).unwrap();
            if pegs != 0 {
                assert!(
                    out.is_none(),
                    "{name} accepted scene {pegs} as a circular face"
                );
            } else if name == "exact" {
                let face =
                    out.expect("the intact disk is visible at the exact camera's resolution");
                assert!(face.center.distance(MIDDLE + Vec3::Z * 0.012) < tol);
                assert!((face.radius - 0.044).abs() < tol);
            }
        }
    }
}
