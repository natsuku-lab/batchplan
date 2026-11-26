//! A look's measured material split at the closing region of a grasp.
use std::sync::Weak;

use batchplan::{
    DepthImage, Device, Grasp, GraspOptions, Gripper, HandReach, Held, IkOptions, Obstacle,
    Occlusion, OccupancyMap, PickBody, Pose, RobotMask, SdfOptions,
};
use glam::Vec3;

use crate::error::{Result, ensure_input, input};
use crate::grip_plan::SupportedObjects;
use crate::memory::{Map, Memory, Observed, Of, Planned, Step};
use crate::scene::{Label, Support};

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
    pub(crate) gripper: &'a Gripper,
    pub(crate) blocked: Option<SupportedObjects>,
    pub(crate) reach: Option<HandReach>,
    support: Option<SupportRoom>,
    id: u32,
    index: usize,
    base: Planned,
    maps: Vec<OccupancyMap>,
    coarse: Option<Attributed>,
    mine: Vec<Vec3>,
    other: Vec<Vec3>,
    views: Vec<(DepthImage<'a>, Option<&'a RobotMask>)>,
    fixtures: Vec<(usize, Obstacle)>,
    divisions: u32,
}

/// Independently modeled target returns are masked in the support's depth views.
#[derive(Debug)]
struct SupportRoom {
    index: usize,
    masks: Vec<RobotMask>,
}

/// The latest support window needed by a sequence of grasps. Its centre is snapped to the
/// same lattice for a candidate projected alone or in a batch.
#[derive(Debug)]
pub(crate) struct SupportWindow {
    center: Vec3,
    scene: Planned,
}

/// The coarse region before choosing which hand-sized window to refine.
#[derive(Debug)]
struct Attributed {
    stationary: OccupancyMap,
    moving: Option<OccupancyMap>,
}

impl Observed {
    /// Prepares the step's world and the current look's labelled readings once for all grasps
    /// of `id`. Give the gripper used by `grasps_on`. An absent object, occupied hand or memory
    /// from another look or after an act gives an input error.
    pub fn grip_worlds<'a>(
        &'a self,
        device: &'a Device,
        step: &Step<'_>,
        gripper: &'a Gripper,
        id: u32,
    ) -> Result<GripWorlds<'a>> {
        let hand = gripper.hand();
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
            return Err(input!("the grip names absent object {id}"));
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
        let (mut mine, mut other) = (vec![], vec![]);
        let mut fixtures = vec![Vec::new(); memory.known.len()];
        let mut unassigned = vec![];
        let mut support = match object.on {
            Some(Support::Object(index)) if object.solid && !memory.objects()[index].solid => {
                Some(SupportRoom {
                    index,
                    masks: Vec::new(),
                })
            }
            _ => None,
        };
        let solids = if object.solid {
            object.shape.solids()
        } else {
            Vec::new()
        };
        for (image, mask, labels) in self.kept.views() {
            let points = device.install(|| image.points(Some(mask)))?;
            if let Some(room) = &mut support {
                let mut background = mask.clone();
                let readings = labels.iter().zip(&points).zip(&mut background.on_robot);
                for ((label, point), masked) in readings {
                    if *label == Label::Object(index)
                        && point.is_some_and(|p| solids.iter().any(|s| s.distance(p).value <= 0.0))
                    {
                        *masked = true;
                    }
                }
                room.masks.push(background);
            }
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
            for solid in solids {
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
        let views: Vec<_> = self
            .kept
            .views()
            .map(|(image, mask, _)| (image, Some(mask)))
            .collect();
        let coarse = if object.solid {
            None
        } else {
            let images: Vec<_> = views.iter().map(|(image, mask)| (image, *mask)).collect();
            let mut stationary = maps[0].clone();
            let moving = stationary.take_surfaces(
                device,
                &images,
                |v, i| self.kept.scene().labels[v][i] == Label::Object(index),
                memory.padding,
                |_, _| true,
            )?;
            Some(Attributed { stationary, moving })
        };
        let supports: Vec<_> = memory
            .standing(index)
            .into_iter()
            .filter(|&i| i != index)
            .map(|i| memory.objects()[i].id)
            .collect();
        let blocked = (!supports.is_empty()).then_some(SupportedObjects {
            object: id,
            supports,
        });
        Ok(GripWorlds {
            observed: self,
            device,
            gripper,
            blocked,
            reach: None,
            support,
            id,
            index,
            base: scene,
            maps,
            coarse,
            mine,
            other,
            views,
            fixtures: fixture_parts,
            divisions,
        })
    }
}

impl GripWorlds<'_> {
    /// The same worlds, checking the hand's kinematic outer bound before projecting each candidate
    /// for [`Self::plan`]. A certified failure keeps its original index. Shared options are
    /// validated up front. The task factory is called only for projected candidates. Chains without
    /// a bound use the ordinary search.
    #[must_use]
    pub fn with_reach_bound(mut self) -> Self {
        self.reach = self.device.robot().hand_reach(self.gripper.hand());
        self
    }

    pub(crate) fn validate(&self, g: &Grasp) -> Result<()> {
        let id = self.id;
        ensure_input!(
            g.pose.is_unit()
                && g.contacts.iter().all(|p| p.is_finite())
                && g.opening > 0.0
                && g.opening <= self.gripper.opening(),
            "grasp of object {id} needs finite contacts, a rigid pose and an opening within {} m, got {g:?}",
            self.gripper.opening()
        );
        Ok(())
    }

    pub(crate) fn closes_on(&self, grasp: &Grasp, point: Vec3) -> bool {
        grasp.closes_on(self.gripper, point, 0.0)
    }

    /// A hollow support stays collision geometry throughout the grip. Near the closing patches
    /// a coarse cell may fill the gap between the support and the hand. Refine that part
    /// from every current camera, retaining the other coarse cells and the parent's unseen space.
    fn support_scene(
        &self,
        at: Vec3,
        views: &[(&DepthImage<'_>, Option<&RobotMask>)],
        window: &mut Option<SupportWindow>,
    ) -> Result<Planned> {
        let memory = &self.observed.memory;
        let Some(room) = &self.support else {
            return Ok(self.base.clone());
        };
        let support = &memory.objects()[room.index];
        let cover = support.shape.obstacle();
        // The fine tile, including the rounding to coarse cells, need never have more samples
        // per axis than the robot's entire map. Large hands retain the coarse map farther out.
        let voxel = memory.map.voxel;
        let budget = 2.0 * self.device.robot().reach() / self.divisions as f32;
        let span = self
            .gripper
            .opening()
            .min((budget - 4.0 * voxel) / 3.0)
            .max(voxel);
        let center = (at / span).round() * span;
        if cover.distance(at).value > span {
            return Ok(self.base.clone());
        }
        if let Some(cached) = window.as_ref().filter(|w| w.center == center) {
            return Ok(cached.scene.clone());
        }
        let mut scene = self.base.clone();
        let Some(slot) = scene
            .of
            .iter()
            .position(|of| matches!(*of, Of::Object(id) | Of::Region(id) if id == support.id))
        else {
            return Ok(scene);
        };
        // The window extends at least one tile span past the contact midpoint. Its boundary
        // lies on the coarse map before subdivision.
        let half = Vec3::splat(1.5 * span);
        let region = Obstacle::cuboid(center, half);
        let radius = crate::memory::voxel_reach(memory.map.voxel);
        let mut outside = memory.room_map(&cover)?;
        let background: Vec<_> = views
            .iter()
            .zip(&room.masks)
            .map(|((image, _), mask)| (*image, Some(mask)))
            .collect();
        let local =
            outside
                .part(&region, 0.0)?
                .refined(self.device, &background, self.divisions)?;
        outside.free(&region, radius)?;
        let bounds = Obstacle::cuboid(center, half + 2.0 * radius);
        let inside = local.part(&bounds, memory.padding)?;
        let mut first = true;
        for part in [outside, inside] {
            let counts = part.counts();
            if counts.occupied + counts.unobserved > 0 {
                let grid = self
                    .device
                    .install(|| part.grid(self.device, Occlusion::Occupied))?;
                let obstacle = Obstacle::from_grid(grid);
                if first {
                    scene.world.obstacles[slot] = obstacle;
                    scene.of[slot] = Of::Region(support.id);
                    first = false;
                } else {
                    scene.world.obstacles.push(obstacle);
                    scene.of.push(Of::Region(support.id));
                }
            }
        }
        *window = Some(SupportWindow {
            center,
            scene: scene.clone(),
        });
        Ok(scene)
    }

    /// Splits only this grasp's measured closing region. Returns `None` when there is no
    /// attributable occupied cell there or more than one solid supplies the contact. The
    /// scene has no touch permissions. Keep its indices and the chosen opening together when
    /// planning or reporting collisions. Drop an unsuccessful result before projecting the next.
    pub fn project(&self, g: &Grasp) -> Result<Option<Gripped>> {
        self.project_in(g, &mut None)
    }

    pub(crate) fn project_in(
        &self,
        g: &Grasp,
        window: &mut Option<SupportWindow>,
    ) -> Result<Option<Gripped>> {
        let id = self.id;
        self.validate(g)?;
        let hand = self.gripper.spheres();
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
        let views: Vec<_> = self
            .views
            .iter()
            .map(|(image, mask)| (image, *mask))
            .collect();
        let pose = Pose::new(g.pose.position, g.pose.rotation.normalize());
        let mut scene =
            self.support_scene(g.contacts[0].midpoint(g.contacts[1]), &views, window)?;
        let mut other_parts = vec![];
        let mut pieces = vec![];
        for source in &self.maps {
            let mut bound = None;
            let mut local = if let Some(coarse) = &self.coarse {
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
                let local = source.part(&region, 0.0)?;
                // A selected coarse cube can extend a diagonal past the box.
                bound = Some(Obstacle::cuboid(
                    (lower + upper) * 0.5,
                    (upper - lower) * 0.5 + 2.0 * crate::memory::voxel_reach(voxel),
                ));
                let radius = crate::memory::voxel_reach(voxel);
                if let Some(mut outside) = coarse.moving.clone() {
                    outside.free(&region, radius)?;
                    if outside.counts().occupied > 0 {
                        other_parts.push(add_map(
                            device,
                            &mut scene,
                            &outside,
                            Occlusion::Free,
                            Of::Object(id),
                        )?);
                    }
                }
                let mut owner = coarse.stationary.clone();
                owner.free(&region, radius)?;
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
            } else {
                source.clone()
            };
            let closes = |p, r| g.closes_on(self.gripper, p, r);
            if object.solid {
                if let Some(part) =
                    local.take_readings(&self.mine, &self.other, memory.padding, closes)?
                {
                    pieces.push(part);
                }
            } else if let Some(mut material) = local.take_surfaces(
                device,
                &views,
                |view, pixel| {
                    self.observed.kept.scene().labels[view][pixel] == Label::Object(index)
                },
                memory.padding,
                |_, _| true,
            )? {
                let readings = material.occupied();
                if let Some(part) =
                    material.take_readings(&readings, &[], memory.padding, closes)?
                {
                    pieces.push(part);
                }
                if material.counts().occupied > 0 {
                    other_parts.push(add_map(
                        device,
                        &mut scene,
                        &material,
                        Occlusion::Free,
                        Of::Object(id),
                    )?);
                }
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
            hand: self.gripper.hand(),
            id,
            shape: cover.clone(),
            grasp: cover.pose().inverse().mul_pose(part_pose.mul_pose(grasp)),
            retained,
            fixtures: self.fixtures.clone(),
        };
        let body = self.blocked.is_none().then(|| PickBody {
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
