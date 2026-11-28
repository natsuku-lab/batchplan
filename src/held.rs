//! A hand holding an object ([`Held`]), for every call that solves, plans or follows with one: each
//! object fitted with spheres once, a device holding it on the same hardware, and a call's items
//! grouped by what they hold, a batch for each. Hands that hold one object together
//! ([`Held::also`]) keep their places on it: the constraints that say so ([`pinned`]) and the
//! poses that follow from one hand's ([`Held::place_hands`]) are made here, for each such call.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};

use glam::{Mat3, Vec3};

use crate::constraint::{self, HandConstraint};
use crate::device::{Device, Worlds};
use crate::error::{Result, ensure_input, input};
use crate::robot::{self, AttachedObject, Fk, Robot};
use crate::spheres::SphereOptions;
use crate::types::Pose;
use crate::world::Obstacle;

/// An object a hand holds, or several hands together.
#[derive(Clone, Debug, PartialEq)]
pub struct Held {
    /// The hand holding it: with several, the one whose frame the object is fixed to.
    pub hand: usize,
    /// Its shape in its own frame: centered at the origin, unrotated.
    pub object: Obstacle,
    /// The hand's IK frame in the object's frame: the grasp that holds it.
    pub grasp: Pose,
    /// Other hands that hold it too, each by its grasp, as two hands carry a tray: none, `hand`
    /// holds it alone. The object is rigid between the hands that hold it. It may touch the links
    /// of each ([`Robot::hand_links`]), and every call that holds it keeps each hand at its grasp:
    /// a pose given for one of them places the others ([`IkProblem::targets`](crate::IkProblem),
    /// [`Goal::Poses`](crate::Goal)), a configuration given (a start, a goal) must have them
    /// there, and every state of a path is checked to, as under a [`HandConstraint`] that pins
    /// each to its place in `hand`'s frame. A hand that lets go leaves this list; the object stays
    /// fixed to `hand`.
    pub also: Vec<Grip>,
    /// Other links of the robot that may touch it, by name, beside those of the hands that hold
    /// it: the links of a hand that has let go of it and has still to back off, or of one that
    /// works on what another holds ([`Robot::hand_links`]), or an arm it rests against. Contact
    /// between them and the object is no collision in any call that holds it; they are not kept
    /// anywhere. Empty: none.
    pub touch_links: Vec<String>,
    /// Joints the robot was loaded with locked (`RobotOptions::lock_joints`), at the values they
    /// keep while it holds the object, as [`Robot::relock`] takes them: the fingers where they
    /// closed on it. Every call that holds it checks the robot so; joints not named keep their
    /// values. A joint that carries a hand (a torso) is refused: some steps solve on the robot as
    /// loaded. To attach the object yourself, relock first, then attach [`Held::attachment`].
    /// Empty: as loaded.
    pub locked: HashMap<String, f32>,
    /// How spheres cover it: they hold all of it, no point of its surface outside them but for
    /// rounding, and reach no more than `tolerance` (5 mm by default) past it, however many
    /// spheres that takes, which `budget` does not cap. A gap narrower than what its spheres reach
    /// past it reads as a contact: a smaller tolerance fits closer, with more spheres. Its own
    /// `max_overhang` holds them, whatever the robot was loaded with, and then leaves some of it
    /// outside them, but for a box, a cylinder or a capsule, which their tiles hold whole within
    /// the smaller of the two unless that is zero, which no tiles keep.
    pub spheres: SphereOptions,
}

impl Held {
    /// The object as `robot` holds it, for [`Robot::attach`]: fixed to the hand's IK frame as the
    /// link `"held by hand {hand}"`, and free to touch the links ([`Robot::hand_links`]) of every
    /// hand that holds it and its [`Held::touch_links`].
    pub fn attachment(&self, robot: &Robot) -> Result<AttachedObject> {
        let hands = self.hands().map(|(hand, _)| hand);
        if let Some(hand) = hands.clone().find(|&hand| hand >= robot.hands()) {
            return Err(input!("hand {hand} of {}", robot.hands()));
        }
        Ok(AttachedObject {
            name: held_link(self.hand),
            link: robot.links[robot.ee_links[self.hand]].name.clone(),
            shapes: vec![self.object.placed(self.grasp.rigid().inverse())],
            touch_links: hands
                .flat_map(|hand| robot.hand_links(hand))
                .map(String::from)
                .chain(self.touch_links.iter().cloned())
                .collect(),
            spheres: self.spheres,
        })
    }

    /// Every hand that holds it with its grasp, `hand` first.
    pub(crate) fn hands(&self) -> impl Iterator<Item = (usize, Pose)> + Clone + '_ {
        let also = self.also.iter().map(|grip| (grip.hand, grip.grasp));
        std::iter::once((self.hand, self.grasp)).chain(also)
    }

    /// The other hands that hold it with hand `hand`, each with its place in that hand's frame,
    /// as their grasps put them: none where `hand` does not hold it, or holds it alone.
    pub(crate) fn partners(&self, hand: usize) -> impl Iterator<Item = (usize, Pose)> + '_ {
        let own = self.hands().find(|&(holder, _)| holder == hand);
        own.into_iter().flat_map(move |(_, grasp)| {
            let to_object = grasp.rigid().inverse();
            let others = self.hands().filter(move |&(other, _)| other != hand);
            others.map(move |(other, grasp)| (other, to_object.mul_pose(grasp.rigid())))
        })
    }

    /// Where each other hand that holds it stays in `hand`'s frame, pinned at its grasp.
    fn pins(&self) -> impl Iterator<Item = HandConstraint> + '_ {
        self.partners(self.hand)
            .map(|(hand, place)| HandConstraint {
                relative_to: Some(self.hand),
                position: Vec3::ZERO,
                tilt: 0.0,
                twist: 0.0,
                ..HandConstraint::new(hand, place)
            })
    }

    /// An `Error::Input` where hands hold it together and hand `hand`, which is none of them,
    /// would follow a path on joints that move one of them, but for those the path holds
    /// (`hold`): it would pull them off their grasps. For `what`.
    pub(crate) fn check_moved_by(
        &self,
        robot: &Robot,
        hand: usize,
        hold: &[usize],
        what: &str,
    ) -> Result<()> {
        if self.also.is_empty() || self.hands().any(|(holder, _)| holder == hand) {
            return Ok(());
        }
        let moved = robot.hand_joints(hand);
        let moves = |holder: usize| {
            let own = robot.hand_joints(holder);
            own.iter().any(|j| moved.contains(j) && !hold.contains(j))
        };
        if let Some((holder, _)) = self.hands().find(|&(holder, _)| moves(holder)) {
            return Err(input!(
                "{what}: hand {hand} follows a path on joints that move hand {holder}, which \
                 holds an object with another hand: follow with one of the hands that hold it, \
                 or hold those joints"
            ));
        }
        Ok(())
    }

    /// An `Error::Input` where more hands than one hold it, for `what`, which moves one hand's.
    pub(crate) fn check_one_hand(&self, what: &str) -> Result<()> {
        ensure_input!(
            self.also.is_empty(),
            "{what}: hands {} and {} hold the object together, and this call moves what one hand \
             holds",
            self.hand,
            self.also.first().map_or(self.hand, |grip| grip.hand)
        );
        Ok(())
    }

    /// Why `q` does not hold it, if it does not: which of the other hands is off its grasp where
    /// `hand` is at its own, and by how much. None for one hand's hold.
    pub(crate) fn off_its_grasps(&self, robot: &Robot, q: &[f32]) -> Option<String> {
        self.pins().find_map(|pin| {
            let off = constraint::breach(robot, &[pin], q)?;
            Some(format!(
                "hand {} is not at its grasp of the object hand {} holds: {off}",
                pin.hand, self.hand
            ))
        })
    }

    /// Places the hands that hold it in `targets` (a pose or none per hand of `robot`), for
    /// `what`: from the first of them given a pose, which stays as given, the others are where
    /// their grasps put them. An `Error::Input` where one given a pose of its own breaks its pin
    /// ([`Held::also`]), as a configuration with the hands there would; nothing where none has a
    /// pose, or one hand holds it. The hold is one `robot` can take ([`Held::check_on`]).
    pub(crate) fn place_hands(
        &self,
        robot: &Robot,
        targets: &mut [Option<Pose>],
        what: &str,
    ) -> Result<()> {
        if self.also.is_empty() {
            return Ok(());
        }
        let given = self.hands().find_map(|(hand, grasp)| {
            let target = targets.get(hand).copied().flatten()?;
            Some((hand, target.rigid().mul_pose(grasp.rigid().inverse())))
        });
        let Some((first, object)) = given else {
            return Ok(());
        };
        // The hands as given, and those not given where their grasps put them.
        let mut fk = Fk::new(robot.links.len());
        for (hand, grasp) in self.hands() {
            let given = targets.get(hand).copied().flatten();
            let at = given.map_or_else(|| object.mul_pose(grasp.rigid()), Pose::rigid);
            let link = robot.ee_links[hand];
            (fk.pos[link], fk.rot[link]) = (at.position, Mat3::from_quat(at.rotation));
        }
        let off = self.pins().find_map(|pin| {
            let off = constraint::breach_at(robot, &[pin], &fk)?;
            Some((pin.hand, off))
        });
        if let Some((hand, off)) = off {
            return Err(input!(
                "{what}: the pose given hand {hand} is off its grasp of the object hand {} holds \
                 ({off}): give one of the hands that hold it a pose, or poses that keep their \
                 grasps",
                self.hand
            ));
        }
        for (hand, grasp) in self.hands().filter(|&(hand, _)| hand != first) {
            if let Some(target) = targets.get_mut(hand) {
                *target = Some(object.mul_pose(grasp.rigid()));
            }
        }
        Ok(())
    }

    /// Whether `robot`, which must not hold one in that hand already, can hold it in `worlds`: an
    /// `Error::Input` naming `what` holds it where not.
    pub(crate) fn check(&self, robot: &Robot, worlds: &Worlds, what: &str) -> Result<()> {
        self.check_on(robot, what)?;
        // Worlds name the link after the robot's own where they let what a hand comes to hold
        // touch something (`Touch::held`), or were uploaded for a robot holding something.
        let link = held_link(self.hand);
        if let Some(was) = worlds.link_name(robot.links.len())
            && was != link
        {
            return Err(input!(
                "{what}: holds an object in hand {}, but these worlds let '{was}' touch obstacles, the link after the robot's own: name what hand {} holds in them (`Touch::held`), or upload them without that touch",
                self.hand,
                self.hand
            ));
        }
        Ok(())
    }

    /// Whether `robot`, which must not hold one in that hand already, can hold it, whatever the
    /// worlds: an `Error::Input` naming `what` holds it where not.
    pub(crate) fn check_on(&self, robot: &Robot, what: &str) -> Result<()> {
        ensure_input!(
            self.hand < robot.hands(),
            "{what}: holds an object in hand {} of {}",
            self.hand,
            robot.hands()
        );
        let link = held_link(self.hand);
        ensure_input!(
            robot.links.iter().all(|l| l.name != link),
            "{what}: the robot already has '{link}': pass the device that does not hold it"
        );
        for (k, grip) in self.also.iter().enumerate() {
            ensure_input!(
                grip.hand < robot.hands()
                    && grip.hand != self.hand
                    && self.also[..k].iter().all(|other| other.hand != grip.hand),
                "{what}: hand {} holds the object too, which must be another of the robot's {} \
                 hands than hand {}, named once",
                grip.hand,
                robot.hands(),
                self.hand
            );
            ensure_input!(
                grip.grasp.is_unit(),
                "{what}: hand {}'s grasp of the held object must be finite with a unit rotation, \
                 got {:?}",
                grip.hand,
                grip.grasp
            );
            let holds = held_link(grip.hand);
            ensure_input!(
                robot.links.iter().all(|l| l.name != holds),
                "{what}: hand {} holds the object too, and the robot already has '{holds}': pass \
                 the device that does not hold it",
                grip.hand
            );
        }
        if let Some(unknown) =
            (self.touch_links.iter()).find(|name| robot.links.iter().all(|l| l.name != **name))
        {
            return Err(input!(
                "{what}: the held object may touch '{unknown}', which is no link of the robot"
            ));
        }
        check_locked(robot, &self.locked, what)?;
        self.spheres
            .check()
            .map_err(|e| input!("{what}: the held object's spheres: {e}"))?;
        ensure_input!(
            !matches!(self.object, Obstacle::Sdf { .. }),
            "{what}: the held object is a distance grid, which cannot be held"
        );
        // In its own frame, as it is attached.
        self.object
            .placed(Pose::IDENTITY)
            .check()
            .map_err(|e| input!("{what}: the held object: {e}, got {:?}", self.object))?;
        ensure_input!(
            self.grasp.is_unit(),
            "{what}: the held grasp must be finite with a unit rotation, got {:?}",
            self.grasp
        );
        Ok(())
    }

    /// `robot` holding the object as [`Held::attachment`] attaches it, with `spheres` fitted to
    /// the object in its own frame and placed by the grasp, as attaching it would place them.
    fn holding(&self, robot: &Robot, spheres: &[[f32; 4]]) -> Result<Robot> {
        let placed = robot::placed_spheres(spheres, self.grasp.rigid().inverse());
        if self.locked.is_empty() {
            return robot.attach_with(&self.attachment(robot)?, || Ok(placed));
        }
        let relocked = robot.relock(&self.locked)?;
        relocked.attach_with(&self.attachment(&relocked)?, || Ok(placed))
    }
}

/// That `locked` (a task's or a hold's `locked`) names joints `robot` was loaded with locked and
/// can take per task, at values it can: none carries a hand, whose IK, plans and constraints the
/// call solves on the robot as loaded. An `Error::Input` naming `what` holds them where not.
pub(crate) fn check_locked(robot: &Robot, locked: &HashMap<String, f32>, what: &str) -> Result<()> {
    if locked.is_empty() {
        return Ok(());
    }
    if let Some((joint, hand)) = locked
        .keys()
        .find_map(|j| robot.hand_carried_by(j).map(|h| (j, h)))
    {
        return Err(input!(
            "{what}: locked joint '{joint}' carries hand {hand}; only joints off the hands' \
             chains, as fingers, take other values per task or hold"
        ));
    }
    robot
        .relock(locked)
        .map(|_| ())
        .map_err(|e| input!("{what}: its locked joints: {e}"))
}

/// The name of the link an object held by `hand` is attached as.
pub(crate) fn held_link(hand: usize) -> String {
    format!("held by hand {hand}")
}

/// Runs `run` on each group of items that hold alike (`holds`: per item, what it holds), with the
/// items' indices in order: on `device` for those holding nothing, and on a device holding the
/// object for the rest, every object fitted once. Groups run on the device's threads, as many at
/// a time as it has (`Device::in_turns`); a call with one group runs it on the calling thread,
/// and one with no items runs `run` once on `device` with none.
pub(crate) fn by_hold<R: Send>(
    device: &Device,
    holds: &[Option<&Held>],
    run: impl Fn(&Device, &[usize]) -> Result<R> + Sync,
) -> Result<Vec<(Vec<usize>, R)>> {
    // Group 0 holds nothing; group k + 1 holds `distinct[k]`.
    let mut distinct: Vec<&Held> = vec![];
    let mut groups: Vec<Vec<usize>> = vec![vec![]];
    for (i, hold) in holds.iter().enumerate() {
        let group = match hold {
            None => 0,
            Some(held) => match distinct.iter().position(|d| d == held) {
                Some(k) => k + 1,
                None => {
                    distinct.push(held);
                    groups.push(vec![]);
                    distinct.len()
                }
            },
        };
        groups[group].push(i);
    }
    let holders = holders(device, device.robot(), &distinct)?;
    let on = |group: usize| match group {
        0 => device,
        k => &holders[k - 1],
    };
    let mut groups: Vec<(usize, Vec<usize>)> = groups
        .into_iter()
        .enumerate()
        .filter(|(_, items)| !items.is_empty())
        .collect();
    if groups.len() <= 1 {
        let (group, items) = groups.pop().unwrap_or_default();
        let result = run(on(group), &items)?;
        return Ok(vec![(items, result)]);
    }
    let results = device.in_turns(&mut groups, |_, (group, items)| run(on(*group), items))?;
    let groups = groups.into_iter().map(|(_, items)| items);
    Ok(groups.zip(results).collect())
}

/// `items` of `all`, in order: `all` itself where they are every one.
pub(crate) fn subset<'a, T: Clone>(all: &'a [T], items: &[usize]) -> Cow<'a, [T]> {
    if items.len() == all.len() {
        Cow::Borrowed(all)
    } else {
        Cow::Owned(items.iter().map(|&i| all[i].clone()).collect())
    }
}

/// A device for `robot` holding each of `helds`, built in parallel, its spheres as [`Fitted`]
/// fits them.
pub(crate) fn holders(device: &Device, robot: &Robot, helds: &[&Held]) -> Result<Vec<Device>> {
    if helds.is_empty() {
        return Ok(vec![]);
    }
    let fitted = Fitted::new(device, robot, helds)?;
    let mut helds = helds.to_vec();
    device.in_turns(&mut helds, |k, held| fitted.holder(device, robot, k, held))
}

/// How many objects' spheres [`Fits`] keeps: those of the latest held.
const KEPT: usize = 64;

/// Spheres fitted to held objects, in their own frames, kept by the devices that share hardware:
/// fitting a cylinder or a capsule takes milliseconds, and every call holding it would fit it
/// again. The least recently used goes first past [`KEPT`]. Calls that fit one object alike at
/// once, as two groups of one call do, wait for the first: it is fitted once.
#[derive(Default)]
pub(crate) struct Fits(Mutex<VecDeque<Fit>>);

/// An object, the options it was fitted by and its spheres, once fitted.
type Fit = (Obstacle, SphereOptions, Slot);

/// Where one object's spheres are kept, held while they are fitted.
type Slot = Arc<Mutex<Option<Arc<[[f32; 4]]>>>>;

impl Fits {
    /// The slot kept for `object` fitted by `options`, now the most recently used: a new, empty
    /// one where there is none.
    fn slot(&self, object: &Obstacle, options: &SphereOptions) -> Slot {
        // A panic elsewhere cannot leave the list half changed: each change is one call on it.
        let mut kept = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let fit = match kept
            .iter()
            .position(|(o, f, _)| o == object && f == options)
        {
            Some(k) => kept.remove(k).expect("a position in the list"),
            None => {
                if kept.len() == KEPT {
                    kept.pop_front();
                }
                (object.clone(), *options, Slot::default())
            }
        };
        let slot = fit.2.clone();
        kept.push_back(fit);
        slot
    }

    /// The spheres kept for `object` fitted by `options`, now the most recently used.
    #[cfg(test)]
    fn get(&self, object: &Obstacle, options: &SphereOptions) -> Option<Arc<[[f32; 4]]>> {
        let kept = self.0.lock().expect("no test panics holding the list");
        let known = kept.iter().any(|(o, f, _)| o == object && f == options);
        drop(kept);
        let slot = known.then(|| self.slot(object, options))?;
        slot.lock().expect("no fit panicked").clone()
    }

    /// Keeps `spheres` for `object` fitted by `options`.
    #[cfg(test)]
    fn keep(&self, object: Obstacle, options: SphereOptions, spheres: Arc<[[f32; 4]]>) {
        let slot = self.slot(&object, &options);
        *slot.lock().expect("no fit panicked") = Some(spheres);
    }

    /// `object`'s spheres in its own frame, fitted by `options`: as kept, or fitted and kept.
    pub(crate) fn spheres(
        &self,
        object: &Obstacle,
        options: &SphereOptions,
    ) -> Result<Arc<[[f32; 4]]>> {
        let slot = self.slot(object, options);
        // A fit that panicked left its slot empty, as it found it: the next call fits again.
        let mut kept = slot.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(spheres) = &*kept {
            return Ok(spheres.clone());
        }
        let fitted: Arc<[[f32; 4]]> =
            robot::object_spheres(std::slice::from_ref(object), options)?.into();
        *kept = Some(fitted.clone());
        Ok(fitted)
    }
}

/// Held objects' spheres: each object fitted once, in its own frame, to be placed by each grasp of
/// it where attaching each would fit them again.
pub(crate) struct Fitted {
    /// Per object, its spheres.
    spheres: Vec<Arc<[[f32; 4]]>>,
    /// Per held object, which of them it has.
    object_of: Vec<usize>,
}

impl Fitted {
    /// Each object among `helds` as the device keeps it ([`Fits`]), or fitted once, in parallel.
    pub(crate) fn new(device: &Device, robot: &Robot, helds: &[&Held]) -> Result<Self> {
        // Each object at its own origin, as `Held::attachment` places it, with the options it fits
        // by: one object held with two sets of options is fitted twice.
        let mut objects: Vec<(Obstacle, SphereOptions)> = vec![];
        let mut object_of = Vec::with_capacity(helds.len());
        for held in helds {
            let object = held.object.placed(Pose::IDENTITY);
            let options = held.attachment(robot)?.spheres;
            let fitted_so = |(o, f): &(Obstacle, SphereOptions)| *o == object && *f == options;
            object_of.push(match objects.iter().position(fitted_so) {
                Some(k) => k,
                None => {
                    objects.push((object, options));
                    objects.len() - 1
                }
            });
        }
        let fits = device.fits();
        let spheres = device.in_turns(&mut objects, |_, (object, options)| {
            fits.spheres(object, options)
        })?;
        Ok(Self { spheres, object_of })
    }

    /// A device for `robot` holding `held`, the `k`th of the held objects fitted.
    pub(crate) fn holder(
        &self,
        device: &Device,
        robot: &Robot,
        k: usize,
        held: &Held,
    ) -> Result<Device> {
        device.with_robot(&held.holding(robot, &self.spheres[self.object_of[k]])?)
    }
}

/// `constraints` with, where more hands than one hold `holding`, the pins that keep each at its
/// grasp ([`Held::also`]), but for those `constraints` state already.
pub(crate) fn pinned<'a>(
    holding: Option<&Held>,
    constraints: &'a [HandConstraint],
) -> Cow<'a, [HandConstraint]> {
    let pins = holding.into_iter().flat_map(Held::pins);
    let mut pins = pins.filter(|pin| !constraints.contains(pin)).peekable();
    if pins.peek().is_none() {
        return Cow::Borrowed(constraints);
    }
    Cow::Owned(constraints.iter().copied().chain(pins).collect())
}

/// Another hand's hold of an object a hand holds ([`Held::also`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grip {
    /// The hand.
    pub hand: usize,
    /// Its IK frame in the object's frame: the grasp it holds the object by.
    pub grasp: Pose,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use glam::{Quat, Vec3};

    use super::*;
    use crate::robot::{CollisionModel, RobotOptions};

    fn panda() -> Robot {
        let fingers = [
            ("panda_finger_joint1".to_string(), 0.04),
            ("panda_finger_joint2".to_string(), 0.04),
        ];
        let options = RobotOptions {
            ee_links: vec!["ee_link".into()],
            lock_joints: HashMap::from(fingers),
            collision_model: Some(
                CollisionModel::load("assets/franka/panda_collision.json").unwrap(),
            ),
            ..Default::default()
        };
        Robot::load("assets/franka/franka_panda.urdf", &options).unwrap()
    }

    /// What a hand comes to hold is the link after the robot's own: with 64 of its own, the
    /// first of a touch set's second word, which a world that names it makes room for.
    #[test]
    fn a_held_objects_touch_fits_past_a_robot_of_64_links() {
        let mut robot = panda();
        let base = robot.links[0].name.clone();
        for k in robot.links.len()..64 {
            let ball = AttachedObject {
                name: format!("ball {k}"),
                link: base.clone(),
                shapes: vec![Obstacle::sphere(Vec3::new(0.0, 0.0, -1.0), 0.01)],
                touch_links: vec![],
                spheres: SphereOptions::default(),
            };
            robot = robot.attach(&ball).expect("a ball attaches");
        }
        assert_eq!(robot.links.len(), 64);
        let slab = Obstacle::cuboid(Vec3::new(0.5, 0.0, 0.2), Vec3::splat(0.1));
        let world = crate::World {
            touch: vec![crate::Touch::held(0, 0)],
            ..crate::World::new(vec![slab.clone(), slab])
        };
        let sets = world.touch_sets(&robot).expect("the touch resolves");
        assert!(
            sets.contains(0, 64),
            "what the hand holds may touch the first slab"
        );
        let others = (0..64).any(|link| sets.contains(0, link) || sets.contains(1, link));
        assert!(
            !others && !sets.contains(1, 64),
            "and nothing else may touch either"
        );
    }

    /// Devices on one GPU or thread pool fit a held object once and keep it, the least recently
    /// used going first.
    #[test]
    fn devices_on_one_hardware_fit_a_held_object_once() {
        let robot = panda();
        let device = Device::cpu(&robot).unwrap();
        let can = Held {
            touch_links: vec![],
            also: vec![],
            locked: Default::default(),
            spheres: Default::default(),
            hand: 0,
            object: Obstacle::cylinder(Vec3::ZERO, 0.03, 0.06),
            grasp: Pose::IDENTITY,
        };
        let first = Fitted::new(&device, &robot, &[&can]).unwrap();
        let holder = holders(&device, &robot, &[&can]).unwrap().remove(0);
        let again = Fitted::new(&holder, &robot, &[&can]).unwrap();
        assert!(Arc::ptr_eq(&first.spheres[0], &again.spheres[0]));
        // Used again just before it would go, it stays; left unused, it goes.
        let options = can.attachment(&robot).unwrap().spheres;
        let ball = |k: usize| Obstacle::sphere(Vec3::ZERO, 0.01 + 1e-3 * k as f32);
        let fits = device.fits();
        for k in 0..KEPT - 1 {
            fits.keep(ball(k), options, Arc::from(vec![]));
        }
        assert!(fits.get(&can.object, &options).is_some());
        fits.keep(ball(KEPT), options, Arc::from(vec![]));
        assert!(fits.get(&can.object, &options).is_some());
        assert!(fits.get(&ball(0), &options).is_none());
        for k in 0..KEPT {
            fits.keep(ball(KEPT + 1 + k), options, Arc::from(vec![]));
        }
        assert!(fits.get(&can.object, &options).is_none());
    }

    /// Pick and place hold each object as attaching [`Held::attachment`] holds it, collision model
    /// and all: a cylinder and a capsule, each at two grasps whose rotations are not their own
    /// inverses, fitted once each.
    #[test]
    fn holders_hold_an_object_as_attaching_it_does() {
        let robot = panda();
        let device = Device::cpu(&robot).unwrap();
        let objects = [
            Obstacle::Cylinder {
                center: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                radius: 0.03,
                half_height: 0.06,
            },
            Obstacle::Capsule {
                center: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                radius: 0.025,
                half_length: 0.04,
            },
        ];
        let grasps = [
            Pose::new(
                Vec3::new(0.01, -0.02, 0.03),
                Quat::from_rotation_x(1.1) * Quat::from_rotation_y(0.4),
            ),
            Pose::new(
                Vec3::new(0.0, 0.01, -0.02),
                Quat::from_rotation_z(2.0) * Quat::from_rotation_x(0.7),
            ),
        ];
        let helds: Vec<Held> = objects
            .iter()
            .flat_map(|object| {
                grasps.map(|grasp| Held {
                    touch_links: vec![],
                    also: vec![],
                    locked: Default::default(),
                    spheres: Default::default(),
                    hand: 0,
                    object: object.clone(),
                    grasp,
                })
            })
            .collect();
        let held_refs: Vec<&Held> = helds.iter().collect();
        let holding = holders(&device, &robot, &held_refs).unwrap();
        for (held, holder) in helds.iter().zip(&holding) {
            let attached = robot.attach(&held.attachment(&robot).unwrap()).unwrap();
            assert!(
                holder.robot().collision_model() == attached.collision_model(),
                "{held:?}"
            );
        }
    }

    /// A grasp whose rotation is a thousandth off unit holds the object where the unit one
    /// does, attached to the robot and placed for a problem that holds it.
    #[test]
    fn a_grasp_not_quite_unit_holds_the_object_where_the_unit_one_does() {
        let robot = panda();
        let turn = Quat::from_rotation_y(1.6) * Quat::from_rotation_x(0.4);
        let own = [[0.0, 0.0, 0.03, 0.03], [0.0, 0.0, -0.03, 0.03]];
        let spheres = |rotation: Quat| {
            let held = Held {
                touch_links: vec![],
                also: vec![],
                locked: Default::default(),
                spheres: Default::default(),
                hand: 0,
                object: Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.02, 0.05)),
                grasp: Pose::new(Vec3::new(0.0, 0.0, 0.28), rotation),
            };
            let attached = robot.attach(&held.attachment(&robot).unwrap()).unwrap();
            let holding = held.holding(&robot, &own).unwrap();
            [attached, holding].map(|r| r.collision_model().spheres[&held_link(0)].clone())
        };
        let unit = spheres(turn);
        for norm in [0.9991f32, 1.0009] {
            for (unit, scaled) in unit.iter().zip(spheres(turn * norm)) {
                assert!(unit[0][..3].iter().any(|x| x.abs() > 0.2), "{unit:?}");
                assert_eq!(unit.len(), scaled.len());
                for (u, s) in unit.iter().zip(&scaled) {
                    let apart = u.iter().zip(s).map(|(a, b)| (a - b).abs());
                    assert!(
                        apart.fold(0.0, f32::max) < 2e-5,
                        "norm {norm}: {u:?} became {s:?}"
                    );
                }
            }
        }
    }

    /// Calls that fit one object alike at once, as two groups of one call do, fit it once: each
    /// gets the same spheres, which the first fitted and the others waited for.
    #[test]
    fn an_object_fitted_at_once_alike_is_fitted_once() {
        let fits = Fits::default();
        let bowl = Obstacle::cylinder(glam::Vec3::ZERO, 0.056, 0.025);
        let options = SphereOptions::default();
        let fitted: Vec<Arc<[[f32; 4]]>> = std::thread::scope(|scope| {
            let fitting: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| fits.spheres(&bowl, &options).unwrap()))
                .collect();
            let joined = fitting.into_iter().map(|h| h.join().expect("a fit"));
            joined.collect()
        });
        assert!(fitted.windows(2).all(|w| Arc::ptr_eq(&w[0], &w[1])));
    }
}
