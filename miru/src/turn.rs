//! Grips to turn a control by, proposed from three sources and ranked by what the retained
//! endpoints show of each closure.

use std::f32::consts::FRAC_1_SQRT_2;
use std::ops::RangeInclusive;

use batchplan::{Device, GraspOptions, Pose, Screw};
use glam::Vec3;

use crate::bar;
use crate::closure::ClosureEvidence;
use crate::error::{Result, ensure_input};
use crate::memory::Kept;
use crate::scene::Label;

/// Where a grip was proposed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GripSource {
    /// Across a raised bar over a measured disk ([`Kept::bar_on_disk`]).
    Bar,
    /// A grasp of the object's measured surface ([`Kept::surface`],
    /// `batchplan::grasps_on_with_capacity`).
    Surface,
    /// A grasp from above of the object's fitted shape (`batchplan::grasps`).
    Shape,
}

/// One hand pose to turn a control from.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct TurnGrip {
    /// The hand's IK frame, in the observation's frame.
    pub pose: Pose,
    /// The aperture the hand is locked at as it approaches (meters): `GraspOptions::opening`
    /// for the hand as loaded, fully open, or a surface grasp's own `Grasp::opening`.
    pub opening: f32,
    /// What proposed it.
    pub source: GripSource,
    /// What the retained endpoints show of its open approach and closure
    /// ([`Kept::observed_closures`] at this pose).
    pub evidence: ClosureEvidence,
}

/// Grips planned together.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct TurnGroup {
    /// In the order to try them.
    pub grips: Vec<TurnGrip>,
    /// What to plan the group with (`Turn::enough`): `Some(1)` where its order is a preference,
    /// `None` where its grips are alike and the quickest should win.
    pub enough: Option<usize>,
}

/// The grips [`Kept::turn_grips`] proposes, and the line to turn about.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct TurnGrips {
    /// Through the measured disk's centre where a bar over a disk resolves, else through the
    /// middle of the object's fitted shape; along the caller's `up`.
    pub screw: Screw,
    /// Plan them in order, a group only where every earlier one found no turn. A group with no
    /// grips is left out.
    pub groups: Vec<TurnGroup>,
}

impl Kept {
    /// Proposes hand poses to turn object `id` about the axis along `up`, the control's outward
    /// unit normal, with hand `hand` of `device`'s robot, in groups to plan in order:
    ///
    /// 1. **Bar**: the poses [`Kept::bar_on_disk`] gives along `up`, in its order, fully open;
    ///    `enough: Some(1)`.
    /// 2. **Shape, both sides**: `batchplan::grasps` of the object's fitted shape at the loaded
    ///    width (`gaps.end()`), those whose z axis points within 45 degrees of `-up`, as the
    ///    shape's approaches are its axes, 90 degrees apart; of these, those whose closure shows
    ///    [`ClosureEvidence::BothSides`]. Fully open; `enough: None`.
    /// 3. **The rest**: the grasps of the object's measured surface
    ///    (`batchplan::grasps_on_with_capacity` with `capacity`), ordered by the moment they
    ///    exert about the screw (`GraspOptions::order_for_turn`), those showing `BothSides` first,
    ///    each at its own opening; then the grasps of group 2's shape whose closure is
    ///    [`ClosureEvidence::Incomplete`], fully open. `enough: Some(1)`.
    ///
    /// Fully open is `options.opening`, the width a surface grasp gives for the hand as loaded.
    /// Every proposal's evidence is [`Kept::observed_closures`]'s at that pose, with these gaps
    /// and options: what the latest endpoints show of an open approach and a symmetric closure,
    /// nothing of hidden material, contact normals or whether the control turns about the axis.
    /// `gaps` are the calibrated closed and loaded widths, as for [`Kept::bar_on_disk`]; `options`
    /// is what those calls take, its `depth` the insertion below a bar's crest and in the
    /// shape's grasps, which grasps of the surface do not read. Plan every motion against the
    /// current world and guard it in execution: a grip here is a proposal.
    ///
    /// None where no label of the look names the object. Out-of-range input is an
    /// `Error::Input` naming the value.
    #[expect(
        clippy::too_many_arguments,
        reason = "the object, the calibrated hand and the turn's axis, as the calls it composes take them"
    )]
    pub fn turn_grips(
        &self,
        device: &Device,
        id: u32,
        hand: usize,
        gaps: RangeInclusive<f32>,
        capacity: f32,
        up: Vec3,
        options: &GraspOptions,
    ) -> Result<Option<TurnGrips>> {
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
            capacity.is_finite()
                && capacity > 0.0
                && capacity <= options.opening
                && options.opening <= loaded,
            "invalid capacity {capacity} for an opening of {} and a loaded width of {loaded}",
            options.opening
        );
        ensure_input!(up.is_normalized(), "invalid up direction {up:?}");
        ensure_input!(
            options.depth.is_finite()
                && options.depth > 0.0
                && options.overlap.is_finite()
                && options.overlap > 0.0
                && options.approach.is_finite()
                && options.approach >= 0.0
                && options.floor.is_none_or(f32::is_finite),
            "invalid insertion {}, overlap {}, approach {} or floor {:?}",
            options.depth,
            options.overlap,
            options.approach,
            options.floor
        );
        // What `batchplan::grasps` refuses besides a shape no grasp fits.
        ensure_input!(
            options.turns > 0 && options.tip.is_finite() && (0.0..1.0).contains(&options.along),
            "invalid turns {}, tip {} or along {}",
            options.turns,
            options.tip,
            options.along
        );
        let scene = self.scene();
        let Some(index) = scene.objects.iter().position(|object| object.id == id) else {
            return Ok(None);
        };
        let label = Label::Object(index);
        let Some(surface) = self.surface(device, id)? else {
            return Ok(None);
        };
        let bar = match self.bar_resolution(label) {
            Some(resolution) => {
                bar::measured(robot, &surface, resolution, up, hand, gaps.clone(), options)?
            }
            None => None,
        };
        let shape = scene.objects[index].shape.obstacle();
        let placed = shape.pose();
        let screw = Screw::about(bar.as_ref().map_or(placed.position, |bar| bar.center), up);
        let loaded_options = GraspOptions {
            opening: loaded,
            ..*options
        };
        // With its options checked, `grasps` refuses only a shape no grasp fits.
        let shaped: Vec<Pose> = batchplan::grasps(&shape, &loaded_options)
            .unwrap_or_default()
            .into_iter()
            .map(|grasp| placed.mul_pose(grasp))
            .filter(|pose| (pose.rotation * Vec3::Z).dot(up) < -FRAC_1_SQRT_2)
            .collect();
        let found = batchplan::grasps_on_with_capacity(
            device,
            hand,
            std::slice::from_ref(&surface),
            options,
            capacity,
        )?;
        let mut seen = found.into_iter().next().unwrap_or_default();
        options.order_for_turn(&mut seen, Some(screw));
        let bars = bar.map(|bar| bar.grasps).unwrap_or_default();
        let poses: Vec<Pose> = (bars.iter().chain(&shaped).copied())
            .chain(seen.iter().map(|grasp| grasp.pose))
            .collect();
        let spheres = robot.hand_spheres(hand);
        let evidence = self.closures(device, label, &spheres, gaps, &poses, options)?;
        let (bar_evidence, evidence) = evidence.split_at(bars.len());
        let (shape_evidence, seen_evidence) = evidence.split_at(shaped.len());
        let grip = |pose, opening, source, evidence| TurnGrip {
            pose,
            opening,
            source,
            evidence,
        };
        let open = options.opening;
        let opposed = (bars.iter().zip(bar_evidence))
            .map(|(&pose, &evidence)| grip(pose, open, GripSource::Bar, evidence))
            .collect();
        let shapes = shaped
            .iter()
            .zip(shape_evidence)
            .map(|(&pose, &evidence)| grip(pose, open, GripSource::Shape, evidence));
        let (centred, partial): (Vec<_>, Vec<_>) =
            shapes.partition(|grip| grip.evidence == ClosureEvidence::BothSides);
        let surfaces = (seen.iter().zip(seen_evidence)).map(|(grasp, &evidence)| {
            grip(grasp.pose, grasp.opening, GripSource::Surface, evidence)
        });
        let (both, others): (Vec<_>, Vec<_>) =
            surfaces.partition(|grip| grip.evidence == ClosureEvidence::BothSides);
        let incomplete = partial
            .into_iter()
            .filter(|grip| grip.evidence == ClosureEvidence::Incomplete);
        let rest = both.into_iter().chain(others).chain(incomplete).collect();
        let groups = [(opposed, Some(1)), (centred, None), (rest, Some(1))]
            .into_iter()
            .filter(|(grips, _)| !grips.is_empty())
            .map(|(grips, enough)| TurnGroup { grips, enough })
            .collect();
        Ok(Some(TurnGrips { screw, groups }))
    }
}
