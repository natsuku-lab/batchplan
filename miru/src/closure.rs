//! What a parallel hand approaching open and closing would intercept in a retained observation.

use batchplan::{Device, GraspOptions, Gripper, HandSphere, Pose, Surface};
use glam::{Mat3, Vec3};

use crate::error::{Result, ensure_input};
use crate::memory::Kept;
use crate::scene::Label;
use crate::threads;

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
    /// All poses are in the observation frame, in input order. `gripper` is the hand, from
    /// `Robot::gripper` for `device`'s robot: its spheres describe the open pose, and it closes
    /// from its loaded gap to its closed one, its jaws sliding: a gripper whose jaws do not is an
    /// `Error::Input`. Only `approach` and `floor` are used from `options`.
    /// On `device`'s threads, each pose's evidence the same on any number of them. No visible
    /// reading for `id` gives `None`.
    ///
    /// This is a measurement of the latest endpoints. Continue checking the scene's obstacles,
    /// unknown space and the whole motion, including rotation and withdrawal, with the planner.
    pub fn observed_closures(
        &self,
        device: &Device,
        id: u32,
        gripper: &Gripper,
        poses: &[Pose],
        options: &GraspOptions,
    ) -> Result<Option<Vec<ClosureEvidence>>> {
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
        Ok(Some(self.closures(device, label, gripper, poses, options)?))
    }

    /// [`inspect`] at each of `poses`, against the retained endpoints off the robot, those `label`
    /// names flagged as the target's, on `device`'s threads. The endpoints are read once, and
    /// only those within the box about every pose's swept hand are kept, which are all that
    /// `inspect` reads.
    pub(crate) fn closures(
        &self,
        device: &Device,
        label: Label,
        gripper: &Gripper,
        poses: &[Pose],
        options: &GraspOptions,
    ) -> Result<Vec<ClosureEvidence>> {
        let sweep = Sweep::new(gripper, options.approach);
        let (lo, hi) = (sweep.lo, sweep.hi);
        let (mid, half) = ((lo + hi) * 0.5, (hi - lo) * 0.5);
        let mut low = Vec3::splat(f32::INFINITY);
        let mut high = Vec3::splat(f32::NEG_INFINITY);
        for pose in poses {
            let center = pose.position + pose.rotation * mid;
            let turned = Mat3::from_quat(pose.rotation);
            let reach = Mat3::from_cols(
                turned.x_axis.abs(),
                turned.y_axis.abs(),
                turned.z_axis.abs(),
            ) * half;
            // What `inspect`'s move into the hand's frame and this box's corners round off.
            let size = pose.position.abs().max_element() + lo.abs().max(hi.abs()).max_element();
            let pad = Vec3::splat(rounding(size));
            low = low.min(center - reach - pad);
            high = high.max(center + reach + pad);
        }
        let mut surface = Surface::default();
        for (i, labels) in self.scene().labels.iter().enumerate() {
            let image = self
                .image(i)
                .expect("a retained image for every label image");
            let readings = image.points(self.robot(i))?;
            for (point, named) in readings.into_iter().zip(labels) {
                if let Some(point) = point
                    && point.cmpge(low).all()
                    && point.cmple(high).all()
                {
                    surface.points.push(point);
                    surface.object.push(*named == label);
                }
            }
        }
        Ok(device.install(|| {
            threads::each(rayon::current_num_threads(), poses.len(), |k| {
                inspect(&surface, poses[k], &sweep, options, f32::INFINITY)
            })
        }))
    }
}

/// Where hand sphere `s` stands in the hand's frame with the fingers closed by `stroke`, open,
/// and open `approach` back along the hand's z axis.
fn stations(s: &HandSphere, stroke: f32, approach: f32) -> [Vec3; 3] {
    let closed = s.center - Vec3::Y * s.side * stroke * 0.5;
    [closed, s.center, s.center - Vec3::Z * approach]
}

/// The box in the hand's frame that holds hand sphere `s` at each of its [`stations`].
fn reach(s: &HandSphere, stroke: f32, approach: f32) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for at in stations(s, stroke, approach) {
        lo = lo.min(at - s.radius);
        hi = hi.max(at + s.radius);
    }
    (lo, hi)
}

/// How far rounding may move a point of coordinates up to `size` (meters) through a rotation and
/// a few sums: a few units in the last place; sixteen of them bound it.
fn rounding(size: f32) -> f32 {
    16.0 * f32::EPSILON * size
}

/// A hand as it comes in open and closes its fingers, in its own frame: the box about every
/// sphere's sweep, and a grid over the box listing per cell the spheres whose box ([`reach`])
/// comes within rounding of it. A reading in a cell can only be in the sweep of the spheres it
/// lists, so [`inspect`] tests no others.
pub(crate) struct Sweep<'a> {
    hand: &'a [HandSphere],
    stroke: f32,
    approach: f32,
    lo: Vec3,
    hi: Vec3,
    cell: f32,
    dims: [usize; 3],
    /// Where each cell's spheres start in `spheres`, and one past the last cell's end.
    starts: Vec<usize>,
    spheres: Vec<usize>,
}

impl<'a> Sweep<'a> {
    /// `gripper`'s hand closing from its loaded gap to its closed one after coming in open along
    /// `approach` (meters).
    pub(crate) fn new(gripper: &'a Gripper, approach: f32) -> Self {
        let (hand, stroke) = (gripper.spheres(), gripper.opening() - gripper.closed_gap());
        let boxes: Vec<(Vec3, Vec3)> = hand.iter().map(|s| reach(s, stroke, approach)).collect();
        let (lo, hi) = boxes.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), (low, high)| (lo.min(*low), hi.max(*high)),
        );
        // As many cells as spheres: a reading then meets about as many spheres as one sphere's
        // box covers cells, rather than every sphere of the hand. And along no axis more than one
        // past their count, so that a box drawn out along one axis, as by a long approach, holds
        // at most the cube of that in cells. A box of no volume is one cell.
        let count = hand.len().max(1) as f32;
        let cell = ((hi - lo).element_product() / count)
            .cbrt()
            .max((hi - lo).max_element() / count);
        let cell = if cell.is_finite() && cell > 0.0 {
            cell
        } else {
            f32::INFINITY
        };
        let dims = ((hi - lo) / cell)
            .ceil()
            .to_array()
            .map(|n| (n as usize).max(1));
        let pad = rounding(lo.abs().max(hi.abs()).max_element());
        let index = |at: Vec3| -> [usize; 3] {
            let at = ((at - lo) / cell).max(Vec3::ZERO).to_array();
            std::array::from_fn(|axis| (at[axis] as usize).min(dims[axis] - 1))
        };
        let cells = |(low, high): &(Vec3, Vec3)| {
            let (first, last) = (index(*low - pad), index(*high + pad));
            (first[2]..=last[2]).flat_map(move |z| {
                (first[1]..=last[1]).flat_map(move |y| (first[0]..=last[0]).map(move |x| [x, y, z]))
            })
        };
        let flat = |[x, y, z]: [usize; 3]| x + dims[0] * (y + dims[1] * z);
        let mut starts = vec![0; dims[0] * dims[1] * dims[2] + 1];
        for reached in &boxes {
            for at in cells(reached) {
                starts[flat(at) + 1] += 1;
            }
        }
        for i in 1..starts.len() {
            starts[i] += starts[i - 1];
        }
        let mut filled = starts.clone();
        let mut spheres = vec![0; starts[starts.len() - 1]];
        for (k, reached) in boxes.iter().enumerate() {
            for at in cells(reached) {
                let slot = &mut filled[flat(at)];
                spheres[*slot] = k;
                *slot += 1;
            }
        }
        Self {
            hand,
            stroke,
            approach,
            lo,
            hi,
            cell,
            dims,
            starts,
            spheres,
        }
    }

    /// The spheres whose sweep may hold `at`, a point of the box.
    fn near(&self, at: Vec3) -> &[usize] {
        let at = ((at - self.lo) / self.cell).max(Vec3::ZERO).to_array();
        let [x, y, z]: [usize; 3] =
            std::array::from_fn(|axis| (at[axis] as usize).min(self.dims[axis] - 1));
        let i = x + self.dims[0] * (y + self.dims[1] * z);
        &self.spheres[self.starts[i]..self.starts[i + 1]]
    }
}

pub(crate) fn inspect(
    surface: &Surface,
    pose: Pose,
    sweep: &Sweep<'_>,
    options: &GraspOptions,
    depth_limit: f32,
) -> ClosureEvidence {
    let (hand, stroke) = (sweep.hand, sweep.stroke);
    let inverse = pose.rotation.conjugate();
    for s in hand {
        for at in stations(s, stroke, sweep.approach) {
            if options
                .floor
                .is_some_and(|z| pose.position.z + (pose.rotation * at).z - s.radius < z)
            {
                return ClosureEvidence::Obstructed;
            }
        }
    }
    let (lo, hi) = (sweep.lo, sweep.hi);
    let mut contacted = [false; 2];
    for (i, p) in surface.points.iter().enumerate() {
        let at = inverse * (*p - pose.position);
        if at.cmplt(lo).any() || at.cmpgt(hi).any() {
            continue;
        }
        let ours = surface.object.is_empty() || surface.object[i];
        for &k in sweep.near(at) {
            let s = &hand[k];
            let d = at - s.center;
            let open = Vec3::new(d.x, d.y, d.z - d.z.clamp(-sweep.approach, 0.0));
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
