//! A raised bar over a measured circular face.

use std::collections::HashMap;

use batchplan::{Device, Gripper, Pose, Surface, SurfaceGraspOptions};
use glam::{DMat3, DVec3, Mat3, Quat, Vec2, Vec3};

use crate::closure::Sweep;
use crate::error::{Result, ensure_input};
use crate::fit;
use crate::memory::Kept;
use crate::scene::Label;

/// A narrow crest on the selected object over a circular outline in the surrounding readings.
/// Its centre is the centre of the visible disk. Whether anything turns about it needs independent
/// knowledge of the mechanism. The bar and the disk may even be one stationary part.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct BarOnDisk {
    /// Centre of the measured circular face in the surface's frame.
    pub center: Vec3,
    /// Radius of the measured circular outline, in meters.
    pub radius: f32,
    /// Width of the measured crest across the grip, in meters.
    pub width: f32,
    /// Opposite top-down hand poses across the crest, at the insertion asked for. Each must be
    /// checked with the actual hand, aperture and scene.
    pub grasps: Vec<Pose>,
}

impl BarOnDisk {
    /// Measures a bar along an assumed outward normal and a disk below it. Only target points
    /// contribute to the bar. All surface points can contribute to the disk, regardless of owner.
    /// Neither the estimate nor a missing estimate changes the collision world or its touches.
    /// `resolution` is the spatial tolerance of the surface in meters. `insertion` is how far
    /// below the crest to put the hand frame. Rays in `seen_from` must show space beyond the rim.
    /// Returns none where either feature is unresolved.
    pub fn measure(
        surface: &Surface,
        outward: Vec3,
        insertion: f32,
        resolution: f32,
    ) -> Result<Option<Self>> {
        ensure_input!(
            outward.is_normalized(),
            "invalid outward unit normal {outward:?}"
        );
        ensure_input!(
            insertion.is_finite() && insertion > 0.0,
            "invalid insertion {insertion}"
        );
        ensure_input!(
            resolution.is_finite() && resolution > 0.0,
            "invalid resolution {resolution}"
        );
        ensure_input!(
            surface.object.is_empty() || surface.object.len() == surface.points.len(),
            "{} object flags for {} surface points",
            surface.object.len(),
            surface.points.len()
        );
        ensure_input!(
            surface.points.iter().all(|p| p.is_finite()),
            "nonfinite surface point"
        );
        ensure_input!(
            surface.seen_from.is_empty() || surface.seen_from.len() == surface.points.len(),
            "{} ray origins for {} points",
            surface.seen_from.len(),
            surface.points.len()
        );
        let (u, v) = outward.any_orthonormal_pair();
        let basis = Mat3::from_cols(u, v, outward);
        let local = basis.transpose();
        let target: Vec<_> = surface
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| surface.object.is_empty() || surface.object[*i])
            .map(|(_, p)| local * *p)
            .collect();
        if target.len() < 3 {
            return Ok(None);
        }
        let mut heights: Vec<_> = target.iter().map(|p| p.z).collect();
        heights.sort_unstable_by(f32::total_cmp);
        let low = quantile(&heights, 0.01);
        let top = quantile(&heights, 0.995);
        if top - low <= insertion + 2.0 * resolution {
            return Ok(None);
        }
        let crest: Vec<_> = target
            .iter()
            .filter(|p| p.z >= top - insertion)
            .map(|p| p.truncate())
            .collect();
        let Some(bar) = crest_box(&crest, resolution) else {
            return Ok(None);
        };
        // A second object measured through the middle of the proposed contact rules it out.
        let foreign = surface
            .points
            .iter()
            .zip(&surface.object)
            .filter(|(_, ours)| !**ours);
        for (point, _) in foreign {
            let p = local * *point;
            let offset = p.truncate() - bar.center;
            if p.z >= top - insertion
                && p.z <= top + resolution
                && offset.dot(bar.along).abs() < bar.length * 0.5 - resolution
                && offset.dot(bar.along.perp()).abs() < bar.width * 0.5 - resolution
            {
                return Ok(None);
            }
        }
        let near: Vec<_> = surface
            .points
            .iter()
            .map(|p| local * *p)
            .filter(|p| p.truncate().distance_squared(bar.center) < bar.length * bar.length)
            .collect();
        let mut floor: Vec<_> = near
            .iter()
            .map(|p| p.z)
            .filter(|&z| z >= low - resolution && z < low + (top - low) / 3.0)
            .collect();
        floor.sort_unstable_by(f32::total_cmp);
        let Some(level) = densest(&floor, 2.0 * resolution) else {
            return Ok(None);
        };
        if top - insertion - level <= 2.0 * resolution {
            return Ok(None);
        }
        let layer: Vec<Vec2> = near
            .iter()
            .filter(|p| (p.z - level).abs() <= resolution)
            .map(|p| p.truncate())
            .collect();
        let hull = fit::hull(layer.clone());
        // A search boundary is not evidence of a circular edge.
        let edge: Vec<_> = hull
            .into_iter()
            .filter(|p| p.distance(bar.center) < bar.length - 2.0 * resolution)
            .collect();
        let Some((center, radius)) = circle(&edge, resolution) else {
            return Ok(None);
        };
        if !measured_face(&layer, center, radius, resolution) {
            return Ok(None);
        }
        // A missing depth boundary can cut a perfect circle out of an unbroken slab.
        // Require observed rays that pass outside the measured rim and reach below the face.
        let sectors = (std::f32::consts::TAU * radius / (2.0 * resolution)).ceil() as usize;
        let mut exterior = vec![false; sectors];
        for (p, eye) in surface.points.iter().zip(&surface.seen_from) {
            if !eye.is_finite() {
                continue;
            }
            let end = local * *p;
            if end.z >= level - resolution {
                continue;
            }
            let start = local * *eye;
            if start.z <= level + resolution {
                continue;
            }
            let cross = start.lerp(end, (start.z - level) / (start.z - end.z));
            let offset = cross.truncate() - center;
            let distance = offset.length();
            if distance < radius - 2.0 * resolution {
                return Ok(None);
            }
            if distance >= radius + resolution && distance <= radius + 3.0 * resolution {
                let theta = offset.to_angle().rem_euclid(std::f32::consts::TAU);
                let bin = (theta * sectors as f32 / std::f32::consts::TAU) as usize;
                exterior[bin.min(sectors - 1)] = true;
            }
        }
        if exterior.iter().filter(|seen| **seen).count() <= sectors / 2 {
            return Ok(None);
        }
        let across = bar.along.perp();
        let off = center - bar.center;
        if off.dot(across).abs() > bar.width / 2.0
            || off.dot(bar.along).abs() > bar.length / 2.0
            || radius <= bar.width / 2.0 + resolution
        {
            return Ok(None);
        }
        let x = basis * bar.along.extend(0.0);
        let rotation = Quat::from_mat3(&Mat3::from_cols(x, (-outward).cross(x), -outward));
        let at = basis * bar.center.extend(top - insertion);
        Ok(Some(Self {
            center: basis * center.extend(level),
            radius,
            width: bar.width,
            grasps: vec![
                Pose::new(at, rotation),
                Pose::new(at, rotation * Quat::from_rotation_z(std::f32::consts::PI)),
            ],
        }))
    }
}

struct Bar {
    center: Vec2,
    along: Vec2,
    width: f32,
    length: f32,
}

fn quantile(sorted: &[f32], fraction: f32) -> f32 {
    let index = (sorted.len() - 1) as f32 * fraction;
    let i = index as usize;
    sorted[i] + (sorted[(i + 1).min(sorted.len() - 1)] - sorted[i]) * index.fract()
}

fn crest_box(points: &[Vec2], resolution: f32) -> Option<Bar> {
    if points.len() < 3 {
        return None;
    }
    let mean = points.iter().sum::<Vec2>() / points.len() as f32;
    let mut covariance = Vec2::ZERO;
    for p in points {
        let d = *p - mean;
        covariance += Vec2::new(d.x * d.x - d.y * d.y, 2.0 * d.x * d.y);
    }
    let along = Vec2::from_angle(0.5 * covariance.y.atan2(covariance.x));
    let limits = |direction: Vec2| {
        let mut offsets: Vec<_> = points.iter().map(|p| (*p - mean).dot(direction)).collect();
        offsets.sort_unstable_by(f32::total_cmp);
        let (a, b) = (quantile(&offsets, 0.01), quantile(&offsets, 0.99));
        let gap = offsets
            .windows(2)
            .any(|s| s[0] >= a && s[1] <= b && s[1] - s[0] > 2.0 * resolution);
        (!gap).then_some(((a + b) / 2.0, b - a))
    };
    let (x, length) = limits(along)?;
    let (y, width) = limits(along.perp())?;
    (width > 2.0 * resolution && length > 2.0 * width).then_some(Bar {
        center: mean + along * x + along.perp() * y,
        along,
        width,
        length,
    })
}

fn densest(sorted: &[f32], band: f32) -> Option<f32> {
    let mut end = 0;
    let mut best = 0..0;
    for (start, &height) in sorted.iter().enumerate() {
        while end < sorted.len() && sorted[end] <= height + band {
            end += 1;
        }
        if end - start > best.len() {
            best = start..end;
        }
    }
    (best.len() >= 3).then(|| quantile(&sorted[best], 0.5))
}

// An arc of corners alone can come from disconnected posts. Over more than half the circle,
// require actual face readings across its outer half, at the resolution at which it was fitted.
fn measured_face(points: &[Vec2], center: Vec2, radius: f32, resolution: f32) -> bool {
    let spacing = 2.0 * resolution;
    let sectors = (std::f32::consts::TAU * radius / spacing).ceil() as usize;
    let steps = (radius * 0.5 / spacing).ceil() as usize;
    if sectors < 4
        || steps == 0
        || sectors.saturating_mul(steps) > 4usize.saturating_mul(points.len())
    {
        return false;
    }
    let mut cells = HashMap::<(i32, i32), Vec<Vec2>>::new();
    for p in points {
        let q = *p - center;
        let at = (q / spacing).floor().as_ivec2();
        cells.entry((at.x, at.y)).or_default().push(q);
    }
    let reached = |p: Vec2| {
        let at = (p / spacing).floor().as_ivec2();
        (-1..=1).any(|dy| {
            (-1..=1).any(|dx| {
                cells.get(&(at.x + dx, at.y + dy)).is_some_and(|points| {
                    points
                        .iter()
                        .any(|q| q.distance_squared(p) <= 2.0 * resolution * resolution)
                })
            })
        })
    };
    let mut seen = 0;
    for i in 0..sectors {
        let along = Vec2::from_angle(std::f32::consts::TAU * i as f32 / sectors as f32);
        if (0..=steps).all(|j| {
            let distance = radius * 0.5 + (radius * 0.5 - resolution) * (j as f32 / steps as f32);
            reached(along * distance)
        }) {
            seen += 1;
        }
    }
    seen > sectors / 2
}

fn circle(points: &[Vec2], resolution: f32) -> Option<(Vec2, f32)> {
    if points.len() < 3 {
        return None;
    }
    let mean = points.iter().map(|p| p.as_dvec2()).sum::<glam::DVec2>() / points.len() as f64;
    let mut lhs = DMat3::ZERO;
    let mut rhs = DVec3::ZERO;
    for p in points {
        let d = p.as_dvec2() - mean;
        let a = (2.0 * d).extend(1.0);
        lhs += DMat3::from_cols(a * a.x, a * a.y, a * a.z);
        rhs += a * d.length_squared();
    }
    if lhs.determinant() <= 0.0 {
        return None;
    }
    let solution = lhs.inverse() * rhs;
    let center = (mean + solution.truncate()).as_vec2();
    let mut distances: Vec<_> = points.iter().map(|p| p.distance(center)).collect();
    distances.sort_unstable_by(f32::total_cmp);
    let radius = quantile(&distances, 0.5);
    if !radius.is_finite() || distances.iter().any(|&d| (d - radius).abs() > resolution) {
        return None;
    }
    let mut angles: Vec<_> = points.iter().map(|p| (*p - center).to_angle()).collect();
    angles.sort_unstable_by(f32::total_cmp);
    let gaps = angles
        .windows(2)
        .map(|w| w[1] - w[0])
        .chain([angles[0] + std::f32::consts::TAU - angles[angles.len() - 1]]);
    let mut supported = 0.0;
    for gap in gaps {
        if gap >= std::f32::consts::FRAC_PI_2 {
            return None;
        }
        if radius * (1.0 - (gap * 0.5).cos()) <= resolution {
            supported += gap;
        }
    }
    (supported > std::f32::consts::PI).then_some((center, radius))
}

impl Kept {
    /// Measures a bar over a disk and checks its proposed poses against this look's depth.
    /// `gripper` is the hand, from `Robot::gripper` for `device`'s robot: it closes from its
    /// loaded gap to its closed one, its jaws sliding (a gripper whose jaws do not is an
    /// `Error::Input`), and its capacity limits the graspable width. `depth` places the IK frame
    /// below the crest; the options' `overlap` bounds fingertip insertion, and their `approach`
    /// and `floor` bound the approach. The returned poses use the fully open hand. Both closing
    /// sweeps must encounter target readings above the disk and clear other measured material.
    /// Plan every motion against the current world. A circular face supplies a centre hypothesis
    /// for a centred control. Stationary bars on circular faces have exactly the same appearance.
    pub fn bar_on_disk(
        &self,
        device: &Device,
        id: u32,
        outward: Vec3,
        gripper: &Gripper,
        depth: f32,
        options: &SurfaceGraspOptions,
    ) -> Result<Option<BarOnDisk>> {
        let (robot, hand) = (device.robot(), gripper.hand());
        ensure_input!(
            hand < robot.hands(),
            "invalid hand {hand} for {} hands",
            robot.hands()
        );
        ensure_input!(
            gripper.sliding(),
            "hand {hand}'s gripper does not slide (Gripper::sliding): the closure is checked \
             with each jaw sliding along the hand's y axis"
        );
        ensure_input!(
            outward.is_normalized() && depth.is_finite() && depth > 0.0,
            "invalid outward direction {outward:?} or insertion {depth}"
        );
        ensure_input!(
            options.overlap.is_finite()
                && options.overlap > 0.0
                && options.approach.is_finite()
                && options.approach >= 0.0
                && options.floor.is_none_or(f32::is_finite),
            "invalid overlap {}, approach {} or floor {:?}",
            options.overlap,
            options.approach,
            options.floor
        );
        let Some(target) = self
            .scene()
            .objects
            .iter()
            .position(|object| object.id == id)
        else {
            return Ok(None);
        };
        let Some(resolution) = self.bar_resolution(Label::Object(target)) else {
            return Ok(None);
        };
        let Some(surface) = self.surface(device, id)? else {
            return Ok(None);
        };
        measured(&surface, resolution, outward, gripper, depth, options)
    }
}

/// [`Kept::bar_on_disk`] on the target's surface as [`Kept::surface`] gives it, at the look's
/// resolution for it ([`Kept::bar_resolution`]), its inputs checked.
pub(crate) fn measured(
    surface: &Surface,
    resolution: f32,
    outward: Vec3,
    gripper: &Gripper,
    depth: f32,
    options: &SurfaceGraspOptions,
) -> Result<Option<BarOnDisk>> {
    let Some(mut bar) = BarOnDisk::measure(surface, outward, depth, resolution)? else {
        return Ok(None);
    };
    if bar.width + 2.0 * resolution > gripper.capacity() || depth + gripper.tip() < options.overlap
    {
        return Ok(None);
    }
    let sweep = Sweep::new(gripper, options.approach);
    bar.grasps.retain(|&pose| {
        hand_fits(
            surface, pose, bar.center, outward, &sweep, options, resolution,
        )
    });
    Ok((!bar.grasps.is_empty()).then_some(bar))
}

fn hand_fits(
    surface: &Surface,
    pose: Pose,
    disk: Vec3,
    outward: Vec3,
    sweep: &Sweep<'_>,
    options: &SurfaceGraspOptions,
    resolution: f32,
) -> bool {
    let depth_limit = (pose.position - disk).dot(outward) - resolution;
    crate::closure::inspect(surface, pose, sweep, options, depth_limit)
        == crate::ClosureEvidence::BothSides
}

#[cfg(test)]
mod tests {
    use batchplan::{CollisionModel, GripperCalibration, Robot, RobotOptions};

    use super::*;

    #[test]
    fn stops_at_the_measured_closed_gap() {
        let r = 0.003;
        let loaded = 0.080;
        let closed = 0.002;
        let urdf = r#"<robot name="test"><link name="base"/><link name="palm"/><link name="left"/><link name="right"/>
 <joint name="arm" type="revolute"><parent link="base"/><child link="palm"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
 <joint name="l" type="prismatic"><parent link="palm"/><child link="left"/><origin xyz="0 0.003 0"/><axis xyz="0 1 0"/><limit lower="0.001" upper="0.04" effort="1" velocity="1"/></joint>
 <joint name="r" type="prismatic"><parent link="palm"/><child link="right"/><origin xyz="0 -0.003 0"/><axis xyz="0 -1 0"/><limit lower="0.001" upper="0.04" effort="1" velocity="1"/></joint></robot>"#;
        let model = CollisionModel {
            spheres: ["left", "right"]
                .into_iter()
                .map(|n| (n.to_string(), vec![[0., 0., 0., r]]))
                .collect(),
            ..Default::default()
        };
        let robot = Robot::from_urdf(
            urdf,
            &RobotOptions {
                lock_joints: ["l", "r"]
                    .into_iter()
                    .map(|n| (n.to_string(), loaded / 2.))
                    .collect(),
                collision_model: Some(model),
                ee_links: vec!["palm".into()],
                finger_links: std::collections::HashMap::from([(
                    "palm".into(),
                    vec!["left".into(), "right".into()],
                )]),
                ..Default::default()
            },
        )
        .unwrap();
        let locks: HashMap<String, f32> = ["l", "r"]
            .into_iter()
            .map(|n| (n.to_string(), closed / 2.))
            .collect();
        // The hand closing to `closed_gap`.
        let gripper = |closed_gap| {
            let calibration = GripperCalibration {
                capacity: Some(0.016),
                ..GripperCalibration::new(loaded, closed_gap, locks.clone())
            };
            robot.gripper(0, &calibration).unwrap()
        };
        let measured = gripper(closed);
        assert!(measured.sliding() && measured.opening() == loaded);
        let s = Surface {
            points: vec![Vec3::new(0., -0.0006, 0.), Vec3::new(0., 0.0006, 0.)],
            object: vec![true, true],
            ..Surface::default()
        };
        let options = SurfaceGraspOptions {
            approach: 0.020,
            ..Default::default()
        };
        let check = |gripper| {
            hand_fits(
                &s,
                Pose::IDENTITY,
                Vec3::Z * 0.020,
                Vec3::NEG_Z,
                &Sweep::new(&gripper, options.approach),
                &options,
                0.00015,
            )
        };
        assert!(check(gripper(0.0)));
        assert!(!check(measured));
    }
}
