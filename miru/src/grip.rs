//! A look's measured material split at the closing region of a grasp.
use std::sync::Weak;

use batchplan::{
    DepthImage, Device, Grasp, GraspOptions, HandSphere, Held, IkOptions, Obstacle, Occlusion,
    OccupancyMap, PickBody, Pose, RobotMask, SdfOptions,
};
use glam::Vec3;

use crate::error::{Result, ensure_input};
use crate::memory::{Map, Memory, Observed, Of, Planned, Step};
use crate::scene::Label;

/// A world for one candidate grip. Its obstacle order applies to every index here and to
/// contacts reported while executing the candidate.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Gripped {
    /// The split world, without any touch permissions.
    pub scene: Planned,
    /// The obstacle containing only material between the fingers. The fingers may touch this
    /// obstacle while reaching and closing.
    pub part: usize,
    /// The ID of the object whose readings the grasp closes on.
    pub owner: u32,
    /// The hand's pose expressed in the contact part's frame, as a pick or turn takes it.
    pub grasp: Pose,
    /// The whole body to carry for a pick, including the other obstacle indices to remove
    /// on departure. Its enclosing shape fills any hollow interior while carried. None if
    /// moving this object would also displace another object that stands on it.
    pub body: Option<PickBody>,
    pickup: Pickup,
}

#[derive(Clone, Debug)]
struct Pickup {
    source: Weak<Map>,
    hand: usize,
    id: u32,
    shape: Obstacle,
    grasp: Pose,
    retained: Vec<Obstacle>,
    fixtures: Vec<(usize, Obstacle)>,
}

/// Prepared evidence for grasps of one object and hand. Project candidates one at a time with
/// [`Self::project`] to bound storage independently of the number of candidates.
#[derive(Debug)]
pub struct GripWorlds<'a> {
    observed: &'a Observed,
    pub(crate) device: &'a Device,
    pub(crate) hand_index: usize,
    loaded_opening: f32,
    hand: Vec<HandSphere>,
    id: u32,
    index: usize,
    base: Planned,
    maps: Vec<OccupancyMap>,
    mine: Vec<Vec3>,
    other: Vec<Vec3>,
    views: Vec<(DepthImage<'a>, Option<&'a RobotMask>)>,
    fixtures: Vec<(usize, Obstacle)>,
    divisions: u32,
}

impl Observed {
    /// Prepares the step's world and the current look's labelled readings once for all grasps
    /// of `id`. Give the same hand and robot opening used by `grasps_on`. An absent object,
    /// occupied hand or memory from another look or after an act gives an input error.
    pub fn grip_worlds<'a>(
        &'a self,
        device: &'a Device,
        step: &Step<'_>,
        hand: usize,
        id: u32,
        loaded_opening: f32,
    ) -> Result<GripWorlds<'a>> {
        ensure_input!(
            loaded_opening.is_finite() && loaded_opening > 0.0,
            "grip opening must be finite and positive, got {loaded_opening}"
        );
        let memory = &self.memory;
        ensure_input!(
            memory.current(&self.kept.map),
            "the grip on object {id} needs the memory of the look that saw it"
        );
        ensure_input!(
            hand < device.robot().hands(),
            "grip hand {hand} of {}",
            device.robot().hands()
        );
        ensure_input!(
            hand < memory.hands(),
            "grip hand {hand} of the memory's {}",
            memory.hands()
        );
        ensure_input!(
            memory.held(hand).is_none(),
            "hand {hand} already holds an object"
        );
        let index = memory.objects().iter().position(|o| o.id == id);
        let Some(index) = index else {
            return Err(batchplan::Error::Input(format!(
                "the grip names absent object {id}"
            )));
        };
        let mut into = step.into.to_vec();
        into.push(id);
        let mut scene = memory.world(
            device,
            &Step {
                into: &into,
                ..*step
            },
        )?;
        let object = &memory.objects()[index];
        let mut i = 0;
        while i < scene.of.len() {
            if matches!(scene.of[i], Of::Region(seen) | Of::Object(seen) if seen == id) {
                scene.of.remove(i);
                scene.world.remove(i)?;
            } else {
                i += 1;
            }
        }
        let hand_index = hand;
        let hand = device.robot().hand_spheres(hand);
        let (mut mine, mut other) = (vec![], vec![]);
        let mut fixtures = vec![Vec::new(); memory.known.len()];
        let mut unassigned = vec![];
        for (image, mask, labels) in self.kept.views() {
            let points = device.install(|| image.points(Some(mask)))?;
            for (p, label) in points.into_iter().zip(labels) {
                if let Some(p) = p {
                    if *label == Label::Object(index) {
                        mine.push(p);
                    } else {
                        other.push(p);
                        if let Label::Known(fixture) = *label {
                            fixtures[fixture].push(p);
                        } else {
                            unassigned.push(p);
                        }
                    }
                }
            }
        }
        let cover = object.shape.obstacle();
        let voxel = memory.map.voxel;
        // Resolve a fingertip patch: at this spacing the maximum growth of a grid built from
        // depth (two voxel diagonals) is at most the radius used for its surface normals.
        let divisions = (2.0 * 3f32.sqrt() * voxel / GraspOptions::default().contact).ceil() as u32;
        let fine = SdfOptions {
            voxel: voxel / divisions as f32,
            padding: memory.padding,
        };
        let mut maps = vec![];
        let mut fixture_parts = vec![];
        if object.solid {
            for solid in object.shape.solids() {
                let mut map = OccupancyMap::from_shape(&solid, &fine)?;
                if let Some(mut ambiguous) =
                    map.take_readings(&other, &[], memory.padding, |_, _| true)?
                {
                    for (fixture, points) in fixtures.iter().enumerate() {
                        if points.is_empty() {
                            continue;
                        }
                        let mut keep = mine.clone();
                        keep.extend_from_slice(&unassigned);
                        for (k, other) in fixtures.iter().enumerate() {
                            if k != fixture {
                                keep.extend_from_slice(other);
                            }
                        }
                        if let Some(on_fixture) =
                            ambiguous.take_readings(points, &keep, memory.padding, |_, _| true)?
                        {
                            let k = add_map(
                                device,
                                &mut scene,
                                &on_fixture,
                                Occlusion::Free,
                                Of::Known(fixture),
                            )?;
                            fixture_parts.push((fixture, scene.world.obstacles[k].clone()));
                        }
                    }
                    if ambiguous.counts().occupied > 0 {
                        add_map(
                            device,
                            &mut scene,
                            &ambiguous,
                            Occlusion::Free,
                            Of::Region(id),
                        )?;
                    }
                }
                maps.push(map);
            }
        } else {
            maps.push(memory.map.fused.part(&cover, memory.padding)?);
        }
        let views = self
            .kept
            .views()
            .map(|(image, mask, _)| (image, Some(mask)))
            .collect();
        Ok(GripWorlds {
            observed: self,
            device,
            hand_index,
            loaded_opening,
            hand,
            id,
            index,
            base: scene,
            maps,
            mine,
            other,
            views,
            fixtures: fixture_parts,
            divisions,
        })
    }
}

impl GripWorlds<'_> {
    pub(crate) fn validate(&self, g: &Grasp) -> Result<()> {
        let id = self.id;
        ensure_input!(
            g.pose.is_unit()
                && g.contacts.iter().all(|p| p.is_finite())
                && g.opening > 0.0
                && g.opening <= self.loaded_opening,
            "grasp of object {id} needs finite contacts, a rigid pose and an opening within {} m, got {g:?}",
            self.loaded_opening
        );
        Ok(())
    }

    pub(crate) fn closes_on(&self, grasp: &Grasp, point: Vec3) -> bool {
        grasp.closes_on(&self.hand, self.loaded_opening, point, 0.0)
    }

    /// Splits only this grasp's measured closing region. Returns `None` when there is no
    /// attributable occupied cell there or more than one solid supplies the contact. The
    /// scene has no touch permissions. Keep its indices and the chosen opening together when
    /// planning or reporting collisions. Drop an unsuccessful result before projecting the next.
    pub fn project(&self, g: &Grasp) -> Result<Option<Gripped>> {
        let id = self.id;
        self.validate(g)?;
        let hand = &self.hand;
        if self.mine.is_empty()
            || !hand.iter().any(|s| s.side < 0.0)
            || !hand.iter().any(|s| s.side > 0.0)
        {
            return Ok(None);
        }
        let device = self.device;
        let memory = &self.observed.memory;
        let index = self.index;
        let object = &memory.objects()[index];
        let voxel = memory.map.voxel;
        let cover = object.shape.obstacle();
        let mut scene = self.base.clone();
        let views: Vec<_> = self
            .views
            .iter()
            .map(|(image, mask)| (image, *mask))
            .collect();
        let pose = Pose::new(g.pose.position, g.pose.rotation.normalize());
        let mut other_parts = vec![];
        let mut pieces = vec![];
        for source in &self.maps {
            let mut owner = source.clone();
            let mut bound = None;
            let mut local = if object.solid {
                owner
            } else {
                // The coarse grid can grow toward the hand by two voxel diagonals. Refine
                // the whole volume its spheres check out to that distance and the margin.
                let halo = 2.0 * 3f32.sqrt() * voxel + IkOptions::default().collision.margin;
                let mut lower = Vec3::INFINITY;
                let mut upper = Vec3::NEG_INFINITY;
                for sphere in hand {
                    let p = pose.position + pose.rotation * sphere.center;
                    lower = lower.min(p - (sphere.radius + halo));
                    upper = upper.max(p + (sphere.radius + halo));
                }
                let region = Obstacle::cuboid((lower + upper) * 0.5, (upper - lower) * 0.5);
                let local = owner.part(&region, 0.0)?;
                // A selected coarse cube can extend a diagonal past the box.
                bound = Some(Obstacle::cuboid(
                    (lower + upper) * 0.5,
                    (upper - lower) * 0.5 + 2.0 * crate::memory::voxel_reach(voxel),
                ));
                owner.free(&region, crate::memory::voxel_reach(voxel))?;
                if let Some(outside) =
                    owner.take_readings(&self.mine, &self.other, memory.padding, |_, _| true)?
                {
                    other_parts.push(add_map(
                        device,
                        &mut scene,
                        &outside,
                        Occlusion::Free,
                        Of::Object(id),
                    )?);
                }
                let counts = owner.counts();
                if counts.occupied + counts.unobserved > 0 {
                    add_map(
                        device,
                        &mut scene,
                        &owner,
                        Occlusion::Occupied,
                        Of::Region(id),
                    )?;
                }
                local.refined(device, &views, self.divisions)?
            };
            if let Some(part) =
                local.take_readings(&self.mine, &self.other, memory.padding, |p, r| {
                    g.closes_on(hand, self.loaded_opening, p, r)
                })?
            {
                pieces.push(part);
            }
            if !object.solid
                && let Some(outside) =
                    local.take_readings(&self.mine, &self.other, memory.padding, |_, _| true)?
            {
                other_parts.push(add_map(
                    device,
                    &mut scene,
                    &outside,
                    Occlusion::Free,
                    Of::Object(id),
                )?);
            }
            let counts = local.counts();
            if counts.occupied + counts.unobserved > 0 {
                let padded = match &bound {
                    Some(bound) => local.part(bound, memory.padding)?,
                    None => local,
                };
                let at = add_map(
                    device,
                    &mut scene,
                    &padded,
                    Occlusion::Occupied,
                    Of::Region(id),
                )?;
                if object.solid {
                    other_parts.push(at);
                }
            }
        }
        // A contact that would allow the grasp to touch several disconnected solids with
        // one index is ambiguous.
        if pieces.len() != 1 {
            return Ok(None);
        }
        let part = add_map(
            device,
            &mut scene,
            &pieces[0],
            Occlusion::Free,
            Of::Gripped(id),
        )?;
        let part_pose = scene.world.obstacles[part].pose();
        let grasp = part_pose.inverse().mul_pose(pose);
        // Preserve the collision material a normal whole-room pickup would clear. A solid's
        // original model leaves with it; any foreign evidence inside it stays.
        let retained = (scene.of.iter().enumerate())
            .filter(|(k, of)| **of == Of::Region(id) && !other_parts.contains(k))
            .map(|(k, _)| scene.world.obstacles[k].clone())
            .collect();
        let pickup = Pickup {
            source: self.observed.kept.map.clone(),
            hand: self.hand_index,
            id,
            shape: cover.clone(),
            grasp: cover.pose().inverse().mul_pose(part_pose.mul_pose(grasp)),
            retained,
            fixtures: self.fixtures.clone(),
        };
        let body = (memory.standing(index).len() == 1).then(|| PickBody {
            shape: cover.clone(),
            other_parts,
        });
        Ok(Some(Gripped {
            scene,
            part,
            owner: id,
            grasp,
            body,
            pickup,
        }))
    }
}

impl Observed {
    /// The memory after a pick of one of this look's gripped parts. Pass the hold the pick
    /// returned, with its fingers locked as they closed. Only identified body material
    /// leaves. Where an occupied or unexplored part of the room stayed in the lift world,
    /// its collision shape is marked occupied in the fused map. This coarse projection errs
    /// toward collision; later looks update it with the map's usual weighing.
    pub fn picked(&self, grip: &Gripped, held: &Held) -> Result<Memory> {
        let picked = &grip.pickup;
        let id = picked.id;
        ensure_input!(
            self.memory.current(&picked.source) && self.memory.current(&self.kept.map),
            "the pick of object {id} needs the same look that supplied its grip"
        );
        let index = self
            .memory
            .objects()
            .iter()
            .position(|o| o.id == id)
            .expect("the selected object is present in the unchanged source memory");
        ensure_input!(
            self.memory.standing(index).len() == 1,
            "the pick of object {id} would displace an object standing on it"
        );
        ensure_input!(
            held.hand == picked.hand
                && picked.matches_grasp(held.grasp)
                && held.object == picked.shape.placed(Pose::IDENTITY),
            "the hold of object {id} must use its selected grip, body and hand {}",
            picked.hand
        );
        self.memory.after_grip(
            index,
            held,
            &picked.shape,
            &picked.retained,
            &picked.fixtures,
        )
    }
}

fn add_map(
    device: &Device,
    scene: &mut Planned,
    map: &OccupancyMap,
    unknown: Occlusion,
    of: Of,
) -> Result<usize> {
    let grid = device.install(|| map.grid(device, unknown))?;
    let index = scene.of.len();
    scene.world.obstacles.push(Obstacle::from_grid(grid));
    scene.of.push(of);
    Ok(index)
}

impl Pickup {
    /// The same physical frame after composing and inverting poses in single precision.
    /// Account for the coordinates subtracted to form their relative translation, and for the
    /// two signs of a unit quaternion.
    fn matches_grasp(&self, pose: Pose) -> bool {
        if !pose.is_unit() {
            return false;
        }
        let scale = 1.0 + self.shape.pose().position.length() + self.grasp.position.length();
        // Each rotation of a point takes two cross products, a scale and two additions.
        let error = 32.0 * f32::EPSILON;
        let a = pose.rotation.normalize();
        let b = self.grasp.rotation.normalize();
        pose.position
            .abs_diff_eq(self.grasp.position, scale * error)
            && (a.abs_diff_eq(b, error) || a.abs_diff_eq(-b, error))
    }
}
