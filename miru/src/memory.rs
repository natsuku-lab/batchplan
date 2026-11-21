//! One look through batchplan and miru together, as the README's "With batchplan" takes it step
//! by step, and what carries from one look to the next ([`Memory`]).
//!
//! Each thing has one memory: a shape, or batchplan's map of what the cameras read
//! ([`OccupancyMap`]), never both. [`observe`] is a look: each camera's depth filtered, the robot
//! and what its hands hold masked, [`look`](crate::look()) after the memory's scene, and what no
//! shape holds fused into the map.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use batchplan::{
    DepthFilter, DepthImage, Device, GraspOptions, Held, IkOptions, Intrinsics, Obstacle,
    Occlusion, OccupancyMap, Pose, Robot, RobotMask, SdfOptions, Surface, Weighing, World,
};
use glam::Vec3;

use crate::error::{Result, ensure_input};
use crate::look::{Look, LookOptions, Noise, View};
use crate::scene::{Label, Object, Scene, Support};
use crate::view::SIGMAS;

/// One depth camera's frame, and what is known of the camera: facts, so there is no default.
#[derive(Clone, Copy, Debug)]
pub struct Camera<'a> {
    /// The depth as the camera read it, not filtered ([`observe`] filters it), its pinhole and
    /// where the camera stood when it took the frame, in the robot's frame.
    pub image: DepthImage<'a>,
    /// Its colour, as [`View::color`].
    pub color: &'a [u8],
    /// How exact its depth is, from its datasheet, as [`View::noise`]: what the filter, the look
    /// and the margin of the shapes hold its readings to. All zeros is refused: a camera stated
    /// without noise has its flat surfaces filtered away and gives the shapes no margin.
    pub noise: Noise,
    /// How many pixels beside a jump in depth the camera reads toward the other surface, which
    /// its depth method sets ([`RobotMask::with_band`]): none for exact depth, one for a sensor
    /// that samples each pixel's disparity a fraction of a pixel off, half its window for a
    /// stereo block matcher.
    pub band: usize,
    /// What its pose and the known shapes are off by (meters), which no noise covers: 0 where
    /// both are exact.
    pub off_by: f32,
    /// Per pixel, the number of the thing a mask model says it shows, as [`View::masks`].
    pub masks: &'a [u32],
}

/// Where the robot stood when the frames were taken.
#[derive(Clone, Copy, Debug)]
pub struct Stance<'a> {
    /// Its joints, in the order of the device's robot's `joint_names()`.
    pub q: &'a [f32],
    /// Its locked joints as they were read, as `Robot::relock` takes them: the fingers where they
    /// stood. Empty: as the robot was loaded.
    pub locked: &'a HashMap<String, f32>,
    /// How far past the robot's collision spheres a reading of it may lie and still be the
    /// robot's (meters): what the spheres are off the links' surfaces by, a fact of the robot's
    /// model.
    pub padding: f32,
}

/// How a [`Memory`] keeps what the cameras read.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryOptions {
    /// How much each look counts in the map: batchplan's default, `Weighing::Evidence`, or
    /// `Weighing::Latest` for a caller that looks now and then and acts on each look (batchplan's
    /// `docs/perception.md`, "Fused maps").
    pub weighing: Weighing,
}

/// What the robot did, said once, to the memory that holds the thing ([`Memory::after`]).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
#[allow(
    missing_docs,
    reason = "each act's doc says what its fields are: an object's id, a hand's hold, a pose and a \
              distance in the robot's frame (meters)"
)]
pub enum Act {
    /// A hand took object `id` and holds it as `held`. It leaves the objects, and so does what
    /// stands on it, in turn; the hand holds it, and each look's mask leaves it out. Where it was
    /// not solid the map held it: the room it took up is free there, as far as a voxel reaches,
    /// and so is that of each thing that was not solid and left with it.
    Picked { id: u32, held: Held },
    /// The hand that holds object `id` set it down, its frame (its shape's middle, turned as the
    /// shape was) at `at`: where a place plan's held object lies when it is let go. It is among
    /// the objects again with its id, its shape placed there, unseen, and the next look answers
    /// one of three for it:
    /// - seen there: it is among the memory's objects, and some label of the look's scene names
    ///   it;
    /// - its place not seen: it is among the memory's objects, and no label names it;
    /// - its place seen and empty: it is not among the memory's objects.
    ///
    /// Where it is not solid the map holds it, and the room it takes up where it lies is never
    /// seen, as a push leaves it, until a look reads it there.
    LetGo { id: u32, at: Pose },
    /// The hand that holds object `id` lost it where nothing knows: it dropped, slipped, or was
    /// let go in the air. It is in no hand and among no objects, and the next look finds it under
    /// whatever id it gives.
    Lost { id: u32 },
    /// Object `id` and what stands on it, in turn, were pushed by `by`: their shapes move with
    /// them, unseen, and the next look keeps their ids where it finds them there. Where one was
    /// not solid the map held it: the room it took up is never seen, where it stood and where it
    /// stands.
    Pushed { id: u32, by: Vec3 },
    /// Something of object `id` was drawn out by `by`, as a drawer out of a cabinet: the object
    /// stays as it was until a look measures it afresh. Where it is not solid the map held it: the
    /// room it takes up is never seen, where it stood and moved by `by` (batchplan's
    /// `docs/perception.md`: "a pull or a push of something the map holds").
    Drew { id: u32, by: Vec3 },
}

/// What carries from one look to the next: the objects and the ids given, what each hand holds,
/// and the map of what the cameras read that no shape holds. A value: a look gives a new one
/// ([`observe`]), as do acts ([`Memory::after`]).
#[derive(Clone, Debug)]
pub struct Memory {
    /// The shapes the world is known to hold, as [`Look::known`].
    known: Arc<[Obstacle]>,
    /// The objects and the next id, without labels: what the next look takes as `previous`.
    scene: Scene,
    /// Per hand of the robot, what it holds.
    hands: Vec<Option<Holding>>,
    /// The map, shared by the memories made from it until a look fuses another.
    map: Arc<Map>,
    /// What the acts since the map was fused tell it, in order.
    marks: Vec<Mark>,
    /// How far from the robot's frame it reaches (meters): the look's corners and the map's.
    reach: f32,
    /// How far past an object's room its part of the map reaches (meters), as far as a grid must
    /// be exact: the collision margin and the robot's largest sphere.
    padding: f32,
}

/// What one hand holds, and the object it was when it was taken.
#[derive(Clone, Debug)]
struct Holding {
    id: u32,
    held: Held,
    object: Object,
}

/// What an act tells the map, kept until a look fuses it: the voxels within a voxel's reach of
/// `shape` are free (`OccupancyMap::free`), or never seen (`OccupancyMap::clear`).
#[derive(Clone, Debug, PartialEq)]
struct Mark {
    shape: Obstacle,
    free: bool,
}

impl Mark {
    /// Tells `map`, of `voxel`.
    fn on(&self, map: &mut OccupancyMap, voxel: f32) -> Result<()> {
        let reach = voxel_reach(voxel);
        match self.free {
            true => map.free(&self.shape, reach),
            false => map.clear(&self.shape, reach),
        }
    }
}

/// The map of what the cameras read, its voxel, and the parts of it made for the rooms objects
/// take up, kept for every memory that shares the map.
#[derive(Debug)]
struct Map {
    fused: OccupancyMap,
    voxel: f32,
    regions: Mutex<Vec<Region>>,
}

impl Map {
    fn new(fused: OccupancyMap, voxel: f32) -> Map {
        Map {
            fused,
            voxel,
            regions: Mutex::default(),
        }
    }

    /// The parts made so far. A part is whole once pushed, so one a panicking thread left is
    /// whole too.
    fn regions(&self) -> MutexGuard<'_, Vec<Region>> {
        self.regions.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The map's part in the room an object takes up, after the marks a memory had, gridded.
#[derive(Debug)]
struct Region {
    cover: Obstacle,
    marks: Vec<Mark>,
    grid: Option<Obstacle>,
}

impl Memory {
    /// Nothing seen yet, in a cell whose fixtures are `known` (as [`Look::known`]), with a map of
    /// everything as far as the device's robot reaches from its frame (`Robot::reach`) in voxels
    /// of batchplan's default size (`SdfOptions::default().voxel`). A robot whose reach a map of
    /// that size cannot hold is an `Error::Input`.
    pub fn try_new(device: &Device, known: Vec<Obstacle>, o: &MemoryOptions) -> Result<Memory> {
        let robot = device.robot();
        let reach = robot.reach();
        let voxel = SdfOptions::default().voxel;
        let corner = Vec3::splat(reach);
        let fused = OccupancyMap::try_new(-corner, corner, voxel)?.with_weighing(o.weighing);
        // A grid is exact where its box reaches past every surface by the collision margin and
        // the robot's largest sphere (batchplan's `docs/perception.md`, "Padding").
        let spheres = robot.spheres(robot.default_q());
        let largest = spheres.iter().map(|sphere| sphere[3]).fold(0.0, f32::max);
        Ok(Memory {
            known: known.into(),
            scene: Scene::default(),
            hands: vec![None; robot.hands()],
            map: Arc::new(Map::new(fused, voxel)),
            marks: vec![],
            reach,
            padding: IkOptions::default().collision.margin + largest,
        })
    }

    /// The objects as the last look left them, in its order: an object's `on` indexes these.
    #[must_use]
    pub fn objects(&self) -> &[Object] {
        &self.scene.objects
    }

    /// The least id no object has had: the one the next new object takes.
    #[must_use]
    pub fn next_id(&self) -> u32 {
        self.scene.next_id
    }

    /// What hand `hand` holds, with the id it had as an object; none where it holds nothing.
    #[must_use]
    pub fn held(&self, hand: usize) -> Option<(u32, &Held)> {
        let holding = self.hands.get(hand)?.as_ref()?;
        Some((holding.id, &holding.held))
    }

    /// The memory after `acts`, in order ([`Act`]): the objects and the hands at once, the map's
    /// share when the map is next read, by [`Memory::world`] or the next look. The map is not
    /// copied: memories after different acts share it, and the parts made of it for the rooms
    /// objects take up. An id no object has, a let-go or a loss of what no hand holds, a pick into
    /// a hand that holds something already or that the robot has not, and a pose or a distance
    /// that is no number are an `Error::Input` naming the act.
    pub fn after(&self, acts: &[Act]) -> Result<Memory> {
        let mut memory = self.clone();
        for (k, act) in acts.iter().enumerate() {
            memory.act(k, act)?;
        }
        Ok(memory)
    }

    /// The memory after act `k`, `act`.
    fn act(&mut self, k: usize, act: &Act) -> Result<()> {
        match act {
            Act::Picked { id, held } => {
                let index = self.index(k, *id)?;
                let hand = held.hand;
                ensure_input!(
                    hand < self.hands.len(),
                    "act {k} picks object {id} into hand {hand}, of the robot's {}",
                    self.hands.len()
                );
                if let Some(holding) = &self.hands[hand] {
                    return Err(batchplan::Error::Input(format!(
                        "act {k} picks object {id} into hand {hand}, which holds object {} already",
                        holding.id
                    )));
                }
                let object = self.scene.objects[index].clone();
                for taken in self.take(index) {
                    if !taken.solid {
                        let shape = taken.shape.obstacle();
                        self.marks.push(Mark { shape, free: true });
                    }
                }
                let held = held.clone();
                self.hands[hand] = Some(Holding {
                    id: *id,
                    held,
                    object,
                });
            }
            Act::LetGo { id, at } => {
                ensure_input!(
                    at.is_unit(),
                    "act {k} lets go of object {id} at {at:?}, which is no finite pose with a \
                     unit rotation"
                );
                let object = self.release(k, *id)?;
                let shape = object.shape.placed(*at);
                // The map held a thing that is not solid where it lies now as what was there.
                if !object.solid {
                    let shape = shape.obstacle();
                    self.marks.push(Mark { shape, free: false });
                }
                self.scene.objects.push(Object {
                    shape,
                    on: None,
                    ..object
                });
            }
            Act::Lost { id } => {
                self.release(k, *id)?;
            }
            Act::Pushed { id, by } => {
                ensure_input!(
                    by.is_finite(),
                    "act {k} pushes object {id} by {by}, which is no distance"
                );
                let index = self.index(k, *id)?;
                for moved in self.standing(index) {
                    let object = &mut self.scene.objects[moved];
                    let pose = object.shape.pose();
                    let shape = object
                        .shape
                        .placed(Pose::new(pose.position + *by, pose.rotation));
                    if !object.solid {
                        for was in [object.shape.obstacle(), shape.obstacle()] {
                            self.marks.push(Mark {
                                shape: was,
                                free: false,
                            });
                        }
                    }
                    object.shape = shape;
                }
            }
            Act::Drew { id, by } => {
                ensure_input!(
                    by.is_finite(),
                    "act {k} draws object {id} by {by}, which is no distance"
                );
                let object = &self.scene.objects[self.index(k, *id)?];
                if !object.solid {
                    let pose = object.shape.pose();
                    let drawn = object
                        .shape
                        .placed(Pose::new(pose.position + *by, pose.rotation));
                    for shape in [object.shape.obstacle(), drawn.obstacle()] {
                        self.marks.push(Mark { shape, free: false });
                    }
                }
            }
        }
        Ok(())
    }

    /// Where object `id` is among the objects, for act `k`.
    fn index(&self, k: usize, id: u32) -> Result<usize> {
        let index = self.scene.objects.iter().position(|object| object.id == id);
        index.ok_or_else(|| {
            batchplan::Error::Input(format!(
                "act {k} names object {id}, which the memory does not hold"
            ))
        })
    }

    /// The object the hand that holds `id` took, which no hand holds after this, for act `k`.
    fn release(&mut self, k: usize, id: u32) -> Result<Object> {
        let holds = |holding: &Option<Holding>| holding.as_ref().is_some_and(|h| h.id == id);
        let hand = self.hands.iter().position(holds);
        let holding = hand.and_then(|hand| self.hands[hand].take());
        let holding = holding.ok_or_else(|| {
            batchplan::Error::Input(format!(
                "act {k} lets go of object {id}, which no hand holds"
            ))
        })?;
        Ok(holding.object)
    }

    /// Object `index` and the objects that stand on it, in turn, in their order.
    fn standing(&self, index: usize) -> Vec<usize> {
        let mut standing = vec![index];
        // Each round takes in what stands on one taken in before, until none does.
        let mut grew = true;
        while grew {
            grew = false;
            for (k, object) in self.scene.objects.iter().enumerate() {
                let on =
                    matches!(object.on, Some(Support::Object(under)) if standing.contains(&under));
                if on && !standing.contains(&k) {
                    standing.push(k);
                    grew = true;
                }
            }
        }
        standing.sort_unstable();
        standing
    }

    /// Takes object `index` and what stands on it, in turn, out of the objects, what the rest
    /// stand on numbered as they are left; the objects taken, in their order.
    fn take(&mut self, index: usize) -> Vec<Object> {
        let gone = self.standing(index);
        let mut now = vec![None; self.scene.objects.len()];
        let (mut kept, mut taken) = (vec![], vec![]);
        let objects = std::mem::take(&mut self.scene.objects)
            .into_iter()
            .enumerate();
        for (k, object) in objects {
            match gone.contains(&k) {
                true => taken.push(object),
                false => {
                    now[k] = Some(kept.len());
                    kept.push(object);
                }
            }
        }
        for object in &mut kept {
            if let Some(Support::Object(under)) = object.on {
                object.on = now[under].map(Support::Object);
            }
        }
        self.scene.objects = kept;
        taken
    }

    /// The world to plan `step` against on `device`, every obstacle named in [`Planned::of`]:
    ///
    /// 1. The known shapes, in their order, so that what was found to touch them still does.
    /// 2. Each object, in the order of [`Memory::objects`]: one that is `solid` its solids
    ///    ([`Shape::solids`](crate::Shape::solids)); one that is not the room it takes up, its
    ///    shape's box or cylinder, unless the step goes into it: it is named in `into`, or that
    ///    room holds some of the robot's spheres at a configuration of `at`, or of the hand's
    ///    placed by a pose of `reach` (`Robot::hand_spheres`), the robot holding what the
    ///    memory's hands hold: each hold's fingers relocked where it locks them and its thing
    ///    attached (`Held::attachment`), so that a thing carried into a room takes the step into
    ///    it. Then it is its part of the map as the cameras saw into it, never-seen space solid
    ///    (`OccupancyMap::part`, gridded `Occlusion::Occupied`), or the room it takes up where
    ///    nothing in that part is solid.
    /// 3. What the map holds outside every room an object takes up, as one grid with never-seen
    ///    space free, where it holds anything there.
    ///
    /// A part reaches past its object's room by the collision margin of batchplan's
    /// `IkOptions::default()` and the robot's largest sphere: as far as a grid must be exact
    /// (batchplan's `docs/perception.md`, "Padding"). An id in `into` that no object has, a
    /// configuration of `at` that is not one of the robot's, and a pose of `reach` of a hand the
    /// robot has not or that is no number are an `Error::Input`.
    pub fn world(&self, device: &Device, step: &Step<'_>) -> Result<Planned> {
        let robot = device.robot();
        for id in step.into {
            ensure_input!(
                self.scene.objects.iter().any(|object| object.id == *id),
                "the step goes into object {id}, which the memory does not hold"
            );
        }
        for (k, q) in step.at.iter().enumerate() {
            ensure_input!(
                q.len() == robot.dof() && q.iter().all(|value| value.is_finite()),
                "configuration {k} of the step needs a finite value for each of the robot's {} \
                 joints, got {q:?}",
                robot.dof()
            );
        }
        for (k, (hand, pose)) in step.reach.iter().enumerate() {
            ensure_input!(
                *hand < robot.hands() && pose.is_unit(),
                "pose {k} of the step needs one of the robot's {} hands and a finite pose with a \
                 unit rotation, got hand {hand} at {pose:?}",
                robot.hands()
            );
        }
        // The robot's spheres where the step takes it, holding what the hands hold.
        let holding = match step.at.is_empty() && step.reach.is_empty() {
            true => Cow::Borrowed(robot),
            false => device.install(|| self.holding(robot))?,
        };
        let mut spheres: Vec<[f32; 4]> =
            (step.at.iter()).flat_map(|q| holding.spheres(q)).collect();
        for &(hand, pose) in step.reach {
            let placed = (holding.hand_spheres(hand).into_iter()).map(|sphere| {
                let center = pose.position + pose.rotation * sphere.center;
                [center.x, center.y, center.z, sphere.radius]
            });
            spheres.extend(placed);
        }
        let mut obstacles = self.known.to_vec();
        let mut of: Vec<Of> = (0..self.known.len()).map(Of::Known).collect();
        let mut covers = vec![];
        for object in &self.scene.objects {
            if object.solid {
                let solids = object.shape.solids();
                of.extend(std::iter::repeat_n(Of::Object(object.id), solids.len()));
                obstacles.extend(solids);
                continue;
            }
            let cover = object.shape.obstacle();
            let holds =
                |&[x, y, z, radius]: &[f32; 4]| cover.distance(Vec3::new(x, y, z)).value < radius;
            let into = step.into.contains(&object.id) || spheres.iter().any(holds);
            let region = match into {
                true => self.region(device, &cover)?,
                false => None,
            };
            match region {
                Some(region) => {
                    obstacles.push(region);
                    of.push(Of::Region(object.id));
                }
                None => {
                    obstacles.push(cover.clone());
                    of.push(Of::Object(object.id));
                }
            }
            covers.push(cover);
        }
        if let Some(rest) = self.unexplained(device, &covers)? {
            obstacles.push(rest);
            of.push(Of::Unexplained);
        }
        Ok(Planned {
            world: World::new(obstacles),
            of,
        })
    }

    /// `robot` holding what the memory's hands hold: each hold's fingers relocked where it locks
    /// them and its thing attached to its hand.
    fn holding<'a>(&self, robot: &'a Robot) -> Result<Cow<'a, Robot>> {
        let mut holding = Cow::Borrowed(robot);
        for held in self.hands.iter().flatten().map(|holding| &holding.held) {
            if !held.locked.is_empty() {
                holding = Cow::Owned(holding.relock(&held.locked)?);
            }
            let attachment = held.attachment(&holding)?;
            holding = Cow::Owned(holding.attach(&attachment)?);
        }
        Ok(holding)
    }

    /// The map in `cover` as the cameras saw into it, gridded with never-seen space solid: solid
    /// where a reading landed or nothing saw past, free where a camera saw in. None where nothing
    /// in it is solid.
    ///
    /// Made once for a room and the acts since the map was fused: memories that share the map and
    /// have had the same acts since, whatever their hands hold, share it.
    fn region(&self, device: &Device, cover: &Obstacle) -> Result<Option<Obstacle>> {
        let made = |region: &&Region| region.cover == *cover && region.marks == self.marks;
        if let Some(region) = self.map.regions().iter().find(made) {
            return Ok(region.grid.clone());
        }
        let mut part = self.map.fused.part(cover, self.padding)?;
        for mark in &self.marks {
            mark.on(&mut part, self.map.voxel)?;
        }
        let counts = part.counts();
        let grid = match counts.occupied + counts.unobserved {
            0 => None,
            _ => Some(Obstacle::from_grid(part.grid(device, Occlusion::Occupied)?)),
        };
        self.map.regions().push(Region {
            cover: cover.clone(),
            marks: self.marks.clone(),
            grid: grid.clone(),
        });
        Ok(grid)
    }

    /// What the map holds outside every one of `covers`, gridded with never-seen space free, as
    /// the space around a scene is (batchplan's `docs/perception.md`, "One memory for each
    /// thing"): the part of the map over the occupied voxels that reach into no cover, each cover
    /// freed in it as far as a voxel reaches into it. None where no occupied voxel lies outside
    /// them.
    fn unexplained(&self, device: &Device, covers: &[Obstacle]) -> Result<Option<Obstacle>> {
        let reach = voxel_reach(self.map.voxel);
        // An act's mark leaves nothing occupied within a voxel's reach of its shape either.
        let marked = self.marks.iter().map(|mark| &mark.shape);
        let shapes: Vec<&Obstacle> = covers.iter().chain(marked).collect();
        let outside = |p: &Vec3| shapes.iter().all(|shape| shape.distance(*p).value > reach);
        let points: Vec<Vec3> = self
            .map
            .fused
            .occupied()
            .into_iter()
            .filter(outside)
            .collect();
        let Some(&first) = points.first() else {
            return Ok(None);
        };
        let (lower, upper) = (points.iter()).fold((first, first), |(lower, upper), p| {
            (lower.min(*p), upper.max(*p))
        });
        let bounds = Obstacle::cuboid(0.5 * (lower + upper), 0.5 * (upper - lower) + reach);
        let mut part = self.map.fused.part(&bounds, self.padding)?;
        for mark in &self.marks {
            mark.on(&mut part, self.map.voxel)?;
        }
        for cover in covers {
            part.free(cover, reach)?;
        }
        Ok(Some(Obstacle::from_grid(
            part.grid(device, Occlusion::Free)?,
        )))
    }
}

/// Where a step of a plan goes among the objects: a thing that is not solid is the room it takes
/// up unless the step goes into it ([`Memory::world`]). A step that goes nowhere is the default.
#[derive(Clone, Copy, Debug, Default)]
pub struct Step<'a> {
    /// Objects the step acts inside, by id: a place into a container, the owner of a handle a
    /// pull draws.
    pub into: &'a [u32],
    /// Configurations the step starts from or passes, each in the order of the device's robot's
    /// `joint_names()`.
    pub at: &'a [&'a [f32]],
    /// Poses a hand goes to: the hand, and where its IK frame is.
    pub reach: &'a [(usize, Pose)],
}

/// The world to plan a step against ([`Memory::world`]), and what each of its obstacles is.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Planned {
    /// The obstacles, with no touches: which links may touch which obstacle is the caller's to
    /// say, by `of`.
    pub world: World,
    /// Per obstacle of `world`, in its order, what it stands for.
    pub of: Vec<Of>,
}

/// What an obstacle of a [`Planned`] world stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Of {
    /// A known shape, by its index among the memory's.
    Known(usize),
    /// An object, by its id: a solid of its shape where it is `solid`, else the room it takes up.
    Object(u32),
    /// An object that is not solid, by its id, as the cameras saw into it: its part of the map.
    Region(u32),
    /// What the map holds that no shape and no object's room explains.
    Unexplained,
}

/// What a look gives: the memory after it, and what it keeps of its views.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Observed {
    /// The memory the next look and act start from.
    pub memory: Memory,
    /// The look's views and scene, for as long as the caller keeps them.
    pub kept: Kept,
}

/// What one look keeps for later: its scene, whose labels name its objects, and each camera's
/// filtered depth and the robot's mask, as the look read them.
#[derive(Clone, Debug)]
pub struct Kept {
    scene: Scene,
    views: Vec<KeptView>,
}

/// One camera's depth as a look read it.
#[derive(Clone, Debug)]
struct KeptView {
    depth: Vec<f32>,
    width: usize,
    intrinsics: Intrinsics,
    camera: Pose,
    robot: RobotMask,
}

impl KeptView {
    fn image(&self) -> DepthImage<'_> {
        DepthImage {
            depth: &self.depth,
            width: self.width,
            intrinsics: self.intrinsics,
            camera: self.camera,
        }
    }
}

impl Kept {
    /// The look's scene: its objects as the look made them, before any act, and per camera, per
    /// pixel, what the pixel shows. A label's object is an index into these objects, not into a
    /// later memory's.
    #[must_use]
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Camera `camera`'s depth as the look read it: filtered, with the camera's pinhole and pose.
    /// None past the look's cameras.
    #[must_use]
    pub fn image(&self, camera: usize) -> Option<DepthImage<'_>> {
        self.views.get(camera).map(KeptView::image)
    }

    /// Which of camera `camera`'s readings are the robot's or what its hands hold: the mask the
    /// look was given, at the stance's padding. None past the look's cameras.
    #[must_use]
    pub fn robot(&self, camera: usize) -> Option<&RobotMask> {
        self.views.get(camera).map(|view| &view.robot)
    }

    /// Object `id`'s readings over the look's views as one surface, for its grasps and rests
    /// (`grasps_on`, `rests_on`), as the README's "With batchplan" takes them in step 6: each
    /// camera's filtered depth, a point for each reading off the robot, those the look's labels
    /// give the object flagged as its, each with the normal of a fingertip's patch about it
    /// (`GraspOptions::default().contact`), the views joined (`Surface::from_depth`,
    /// `Surface::join`). On `device`'s threads. None where no label of the look names the object:
    /// an id the look has not, or one it kept unseen.
    pub fn surface(&self, device: &Device, id: u32) -> Result<Option<Surface>> {
        let Some(object) = self.scene.objects.iter().position(|object| object.id == id) else {
            return Ok(None);
        };
        let label = Label::Object(object);
        if !self
            .scene
            .labels
            .iter()
            .flatten()
            .any(|named| *named == label)
        {
            return Ok(None);
        }
        let radius = GraspOptions::default().contact;
        let mut joined: Option<Surface> = None;
        for (view, labels) in self.views.iter().zip(&self.scene.labels) {
            let flags: Vec<bool> = labels.iter().map(|named| *named == label).collect();
            let image = view.image();
            let surface = Surface::from_depth(device, &image, &flags, Some(&view.robot), radius)?;
            match &mut joined {
                Some(joined) => joined.join(surface),
                None => joined = Some(surface),
            }
        }
        Ok(joined)
    }
}

/// One look, after `memory`, through batchplan and [`look`](crate::look()) as the README's "With
/// batchplan" takes it, camera by camera:
///
/// 1. The depth is filtered at the camera's noise (`DepthImage::filtered`, batchplan's
///    `DepthFilter` otherwise).
/// 2. The robot is masked at `at`, its locked joints relocked and holding what the memory's hands
///    hold (`DepthImage::robot_mask`, at the stance's padding): the view's `robot`.
/// 3. [`look`](crate::look()) is given the views, the memory's known shapes and objects, and the
///    corners of the map. A camera's noise is raised there by what it is off by, a third of it
///    for each of the three deviations a reading may lie from a shape and be on it, as
///    [`Noise`] says to.
/// 4. The shapes are the known ones and the solids of each object that is `solid`, and the
///    readings within three of the camera's deviations and what it is off by of one, the robot's
///    and what lies within the camera's band behind any of them are left out of the map
///    (`RobotMask::with_shapes` and `with_band`); the rest is fused (`OccupancyMap::integrate`).
///    The map's mask is padded past what a grid of it grows toward collision, 2√3 of its voxels,
///    where that is more than the stance's padding.
///
/// It runs on `device`'s threads, and reads the device's robot: its joints, its spheres and how
/// far it reaches. With no cameras it is a look that sees nothing: the memory as it was. A camera
/// stated with no noise, or one that is negative or no number, is an `Error::Input`, as are
/// stances and frames batchplan's calls refuse.
pub fn observe(
    device: &Device,
    cameras: &[Camera<'_>],
    at: &Stance<'_>,
    memory: &Memory,
) -> Result<Observed> {
    for (index, camera) in cameras.iter().enumerate() {
        check(index, camera)?;
    }
    let robot = device.robot();
    ensure_input!(
        at.q.len() == robot.dof(),
        "the stance has {} joint values, the robot {} joints",
        at.q.len(),
        robot.dof()
    );
    ensure_input!(
        at.q.iter().all(|value| value.is_finite()),
        "the stance's joints must be numbers, got {:?}",
        at.q
    );
    ensure_input!(
        at.padding.is_finite() && at.padding >= 0.0,
        "the stance's padding must be finite and not negative, got {}",
        at.padding
    );
    let robot = match at.locked.is_empty() {
        true => Cow::Borrowed(robot),
        false => Cow::Owned(robot.relock(at.locked)?),
    };
    if cameras.is_empty() {
        let kept = Kept {
            scene: memory.scene.clone(),
            views: vec![],
        };
        return Ok(Observed {
            memory: memory.clone(),
            kept,
        });
    }
    let holding: Vec<Held> = (memory.hands.iter().flatten())
        .map(|holding| holding.held.clone())
        .collect();

    let filtered: Vec<Vec<f32>> = (cameras.iter())
        .map(|camera| {
            let filter = DepthFilter {
                noise: coefficients(camera.noise),
                ..Default::default()
            };
            camera.image.filtered(&filter)
        })
        .collect::<Result<_>>()?;
    let images: Vec<DepthImage> = (cameras.iter().zip(&filtered))
        .map(|(camera, depth)| DepthImage {
            depth,
            ..camera.image
        })
        .collect();
    let masks: Vec<RobotMask> = device.install(|| {
        (images.iter())
            .map(|image| image.robot_mask(&robot, at.q, at.padding, &holding))
            .collect::<Result<_>>()
    })?;
    let views: Vec<View> = (cameras.iter().zip(&images).zip(&masks))
        .map(|((camera, image), mask)| View {
            image: *image,
            color: camera.color,
            noise: raised(camera),
            robot: &mask.on_robot,
            masks: camera.masks,
        })
        .collect();
    let corner = Vec3::splat(memory.reach);
    let options = LookOptions {
        lower_corner: -corner,
        upper_corner: corner,
    };
    let look = Look {
        views: &views,
        known: &memory.known,
    };
    let scene = crate::look(device, &look, &memory.scene, &options)?;

    let shapes: Vec<Obstacle> = (memory.known.iter().cloned())
        .chain(
            (scene.objects.iter())
                .filter(|object| object.solid)
                .flat_map(|object| object.shape.solids()),
        )
        .collect();
    let voxel = memory.map.voxel;
    let padding = map_padding(voxel);
    let mut fused = memory.map.fused.clone();
    for mark in &memory.marks {
        mark.on(&mut fused, voxel)?;
    }
    for ((camera, image), mask) in cameras.iter().zip(&images).zip(&masks) {
        // One mask where the stance's padding reaches past the map's.
        let mask = match padding > at.padding {
            true => device.install(|| image.robot_mask(&robot, at.q, padding, &holding))?,
            false => mask.clone(),
        };
        let margin = coefficients(raised(camera)).map(|coefficient| SIGMAS * coefficient);
        let mask = (mask.with_shapes(device, image, &shapes, margin)?).with_band(
            device,
            image,
            margin,
            camera.band,
        )?;
        fused.integrate(device, image, Some(&mask))?;
    }

    let views = (filtered.into_iter().zip(cameras).zip(masks))
        .map(|((depth, camera), robot)| KeptView {
            depth,
            width: camera.image.width,
            intrinsics: camera.image.intrinsics,
            camera: camera.image.camera,
            robot,
        })
        .collect();
    let memory = Memory {
        scene: Scene {
            objects: scene.objects.clone(),
            labels: vec![],
            next_id: scene.next_id,
        },
        map: Arc::new(Map::new(fused, voxel)),
        marks: vec![],
        ..memory.clone()
    };
    Ok(Observed {
        memory,
        kept: Kept { scene, views },
    })
}

/// Whether camera `index` is stated as a camera is: its noise and what it is off by.
fn check(index: usize, camera: &Camera<'_>) -> Result<()> {
    let noise = coefficients(camera.noise);
    ensure_input!(
        noise.iter().all(|c| c.is_finite() && *c >= 0.0) && noise.iter().any(|c| *c > 0.0),
        "camera {index}: its noise must be finite, not negative and not all zeros, got {:?}",
        camera.noise
    );
    ensure_input!(
        camera.off_by.is_finite() && camera.off_by >= 0.0,
        "camera {index}: what it is off by must be finite and not negative, got {}",
        camera.off_by
    );
    Ok(())
}

/// A noise's coefficients, as batchplan takes them: the constant, the linear and the quadratic.
fn coefficients(noise: Noise) -> [f32; 3] {
    [noise.constant, noise.linear, noise.quadratic]
}

/// A camera's noise raised by what it is off by: a third of it for each of the three deviations
/// ([`SIGMAS`]) a reading may lie from a shape and be on it, so that the look holds its readings
/// to the shapes as far as the map's mask does.
fn raised(camera: &Camera<'_>) -> Noise {
    Noise {
        constant: camera.noise.constant + camera.off_by / SIGMAS,
        ..camera.noise
    }
}

/// How far past the robot's spheres a reading fused into a map of `voxel` is the robot's
/// (meters): past what a grid of the map grows toward collision, so that the robot's surface
/// never meets the space it hides (batchplan's `docs/perception.md`, "Masking the robot").
///
/// A grid made of a map grows by at most 2√3 voxels past a reading, as batchplan's `sdf.rs` builds
/// it. A free point of the grid stores at most its distance from the middle of the nearest
/// occupied voxel less √3 voxels: less half a voxel's diagonal for the voxel's cube, and half a
/// diagonal again so that interpolation never reads farther than the geometry
/// (`finishing_table`). At a point farther than √3 voxels from every occupied middle the eight
/// points of its cell are free, for they lie within √3 voxels of it, and trilinear interpolation
/// reads there at most √3/2 voxels less than the distance at the point: the mean distance of the
/// eight from it, distance being 1-Lipschitz. So the solid lies within 3√3/2 voxels of an occupied
/// voxel's middle, and that middle within √3/2 of the reading that filled the voxel, where a
/// pixel's footprint is no wider than a voxel: out to 3 m through the tests' pinhole of 300
/// pixels, while a coarse array, as an 8x8 time-of-flight sensor's, is wider and grows a grid
/// farther. For 1 cm voxels, 3.46 cm.
fn map_padding(voxel: f32) -> f32 {
    2.0 * 3f32.sqrt() * voxel
}

/// How far a voxel's cube reaches from its middle, half its diagonal (meters): a voxel whose
/// middle lies that near a shape reaches into it, as `OccupancyMap::part` and
/// `RobotMask::with_shapes` take it.
fn voxel_reach(voxel: f32) -> f32 {
    0.5 * 3f32.sqrt() * voxel
}
