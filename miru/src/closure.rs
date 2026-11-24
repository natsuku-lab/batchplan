//! What a parallel hand approaching open and closing would intercept in a retained observation.

use std::ops::RangeInclusive;

use batchplan::{Device, GraspOptions, HandSphere, Pose, Surface};
use glam::Vec3;

use crate::error::{Result, ensure_input};
use crate::memory::Kept;
use crate::scene::Label;

/// Evidence from the observed endpoints and the supplied floor, for one proposed hand pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClosureEvidence {
    /// The open approach intersects a reading, the hand crosses the floor, or the closing
    /// sweep intersects a foreign reading or makes contact outside the fingers.
    Obstructed,
    /// At least one finger's closing sweep lacks a target reading on its own side.
    Incomplete,
    /// Both fingers sweep through target readings on their own sides without an observed
    /// obstruction. Hidden material, contact normals and whether the part can move are unknown.
    BothSides,
}

impl Kept {
    /// Inspects fully open, straight approaches and symmetric closures at supplied hand poses.
    /// All poses are in the observation frame, in input order. Supply the calibrated closed and
    /// loaded widths of a hand qualified by `Robot::parallel_jaw_opening`. The loaded robot's
    /// hand spheres describe the open pose. Only `approach` and `floor` are used from `options`.
    /// No visible reading for `id` gives `None`.
    ///
    /// This is a measurement of the latest endpoints. Continue checking the scene's obstacles,
    /// unknown space and the whole motion, including rotation and withdrawal, with the planner.
    pub fn observed_closures(
        &self,
        device: &Device,
        id: u32,
        hand: usize,
        gaps: RangeInclusive<f32>,
        poses: &[Pose],
        options: &GraspOptions,
    ) -> Result<Option<Vec<ClosureEvidence>>> {
        let robot = device.robot();
        ensure_input!(
            hand < robot.hands(),
            "invalid hand {hand} for {} hands",
            robot.hands()
        );
        let (closed, loaded) = (*gaps.start(), *gaps.end());
        ensure_input!(
            closed >= 0.0 && closed < loaded && loaded.is_finite(),
            "invalid closed and loaded widths {closed} and {loaded}"
        );
        ensure_input!(
            options.approach.is_finite()
                && options.approach >= 0.0
                && options.floor.is_none_or(f32::is_finite),
            "invalid approach {} or floor {:?}",
            options.approach,
            options.floor
        );
        for (index, pose) in poses.iter().enumerate() {
            ensure_input!(
                pose.position.is_finite() && pose.rotation.is_normalized(),
                "invalid hand pose {index}: {pose:?}"
            );
        }
        let scene = self.scene();
        let Some(index) = scene.objects.iter().position(|object| object.id == id) else {
            return Ok(None);
        };
        let label = Label::Object(index);
        if !scene.labels.iter().any(|labels| labels.contains(&label)) {
            return Ok(None);
        }
        if poses.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let spheres = robot.hand_spheres(hand);
        let mut surface = Surface::default();
        for (i, labels) in scene.labels.iter().enumerate() {
            let image = self
                .image(i)
                .expect("a retained image for every label image");
            let readings = image.points(self.robot(i))?;
            for (point, named) in readings.into_iter().zip(labels) {
                if let Some(point) = point {
                    surface.points.push(point);
                    surface.object.push(*named == label);
                }
            }
        }
        let evidence = poses
            .iter()
            .map(|&pose| {
                inspect(
                    &surface,
                    pose,
                    &spheres,
                    gaps.clone(),
                    options,
                    f32::INFINITY,
                )
            })
            .collect();
        Ok(Some(evidence))
    }
}

pub(crate) fn inspect(
    surface: &Surface,
    pose: Pose,
    hand: &[HandSphere],
    gaps: RangeInclusive<f32>,
    options: &GraspOptions,
    depth_limit: f32,
) -> ClosureEvidence {
    let stroke = gaps.end() - gaps.start();
    let inverse = pose.rotation.conjugate();
    let approach = Vec3::Z * options.approach;
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for s in hand {
        let closed = s.center - Vec3::Y * s.side * stroke * 0.5;
        let start = s.center - approach;
        for at in [closed, s.center, start] {
            if options
                .floor
                .is_some_and(|z| pose.position.z + (pose.rotation * at).z - s.radius < z)
            {
                return ClosureEvidence::Obstructed;
            }
            lo = lo.min(at - s.radius);
            hi = hi.max(at + s.radius);
        }
    }
    let mut contacted = [false; 2];
    for (i, p) in surface.points.iter().enumerate() {
        let at = inverse * (*p - pose.position);
        if at.cmplt(lo).any() || at.cmpgt(hi).any() {
            continue;
        }
        let ours = surface.object.is_empty() || surface.object[i];
        for s in hand {
            let d = at - s.center;
            let open = Vec3::new(d.x, d.y, d.z - d.z.clamp(-options.approach, 0.0));
            if open.length_squared() <= s.radius * s.radius {
                return ClosureEvidence::Obstructed;
            }
            let end = -s.side * stroke * 0.5;
            let close = Vec3::new(d.x, d.y - d.y.clamp(end.min(0.0), end.max(0.0)), d.z);
            if close.length_squared() <= s.radius * s.radius {
                if !s.finger || s.side == 0.0 || !ours || at.z >= depth_limit {
                    return ClosureEvidence::Obstructed;
                }
                if at.y * s.side > 0.0 {
                    contacted[usize::from(s.side > 0.0)] = true;
                }
            }
        }
    }
    if contacted == [true, true] {
        ClosureEvidence::BothSides
    } else {
        ClosureEvidence::Incomplete
    }
}
