//! A hand holding an object ([`Held`]), for every call that solves, plans or follows with one: each
//! object fitted with spheres once, a device holding it on the same hardware, and a call's items
//! grouped by what they hold, a batch for each.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

use rayon::prelude::*;

use crate::device::{Device, Worlds};
use crate::error::{Result, ensure_input, input};
use crate::robot::{self, AttachedObject, Robot};
use crate::spheres::SphereOptions;
use crate::types::Pose;
use crate::world::Obstacle;

/// An object a hand holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Held {
    /// The hand holding it.
    pub hand: usize,
    /// Its shape in its own frame: centered at the origin, unrotated.
    pub object: Obstacle,
    /// The hand's IK frame in the object's frame: the grasp that holds it.
    pub grasp: Pose,
}

impl Held {
    /// The object as `robot` holds it, for [`Robot::attach`]: fixed to the hand's IK frame as the
    /// link `"held by hand {hand}"`, and free to touch the hand's links ([`Robot::hand_links`]).
    pub fn attachment(&self, robot: &Robot) -> Result<AttachedObject> {
        ensure_input!(
            self.hand < robot.hands(),
            "hand {} of {}",
            self.hand,
            robot.hands()
        );
        Ok(AttachedObject {
            name: held_link(self.hand),
            link: robot.links[robot.ee_links[self.hand]].name.clone(),
            shapes: vec![self.object.placed(self.grasp.inverse())],
            touch_links: robot
                .hand_links(self.hand)
                .into_iter()
                .map(String::from)
                .collect(),
            spheres: Default::default(),
        })
    }

    /// Whether `robot`, which must not hold one in that hand already, can hold it in `worlds`: an
    /// `Error::Input` naming `what` holds it where not.
    pub(crate) fn check(&self, robot: &Robot, worlds: &Worlds, what: &str) -> Result<()> {
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
        // Worlds whose touches were resolved for a robot holding something name its link.
        if let Some(was) = worlds.link_name(robot.links.len())
            && was != link
        {
            return Err(input!(
                "{what}: holds an object in hand {}, but these worlds let links touch obstacles and were uploaded for a robot holding '{was}': upload them to a device holding nothing, or one in hand {}",
                self.hand,
                self.hand
            ));
        }
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
        let placed = robot::placed_spheres(spheres, self.grasp.inverse());
        robot.attach_with(&self.attachment(robot)?, || Ok(placed))
    }
}

/// The name of the link an object held by `hand` is attached as.
pub(crate) fn held_link(hand: usize) -> String {
    format!("held by hand {hand}")
}

/// Runs `run` on each group of items that hold alike (`holds`: per item, what it holds), with the
/// items' indices in order: on `device` for those holding nothing, and on a device holding the
/// object for the rest, every object fitted once. Groups run in parallel on the device's threads;
/// a call with one group runs it on the calling thread, and one with no items runs `run` once on
/// `device` with none.
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
    device.install(|| {
        groups
            .into_par_iter()
            .map(|(group, items)| {
                let result = run(on(group), &items)?;
                Ok((items, result))
            })
            .collect()
    })
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
    device.install(|| {
        let each = helds.par_iter().enumerate();
        each.map(|(k, held)| fitted.holder(device, robot, k, held))
            .collect()
    })
}

/// How many objects' spheres [`Fits`] keeps: those of the latest held.
const KEPT: usize = 64;

/// Spheres fitted to held objects, in their own frames, kept by the devices that share hardware:
/// fitting a cylinder or a capsule takes milliseconds, and every call holding it would fit it again.
/// The least recently used goes first past [`KEPT`].
#[derive(Default)]
pub(crate) struct Fits(Mutex<VecDeque<Fit>>);

/// An object, the options it was fitted by and its spheres.
type Fit = (Obstacle, SphereOptions, Arc<[[f32; 4]]>);

impl Fits {
    /// The spheres kept for `object` fitted by `options`, now the most recently used.
    fn get(&self, object: &Obstacle, options: &SphereOptions) -> Option<Arc<[[f32; 4]]>> {
        // A panic elsewhere cannot leave the list half changed: each change is one call on it.
        let mut kept = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let k = kept
            .iter()
            .position(|(o, f, _)| o == object && f == options)?;
        let fit = kept.remove(k)?;
        let spheres = fit.2.clone();
        kept.push_back(fit);
        Some(spheres)
    }

    /// Keeps `spheres` for `object` fitted by `options`.
    fn keep(&self, object: Obstacle, options: SphereOptions, spheres: Arc<[[f32; 4]]>) {
        let mut kept = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if kept.iter().all(|(o, f, _)| *o != object || *f != options) {
            if kept.len() == KEPT {
                kept.pop_front();
            }
            kept.push_back((object, options, spheres));
        }
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
        // by.
        let mut objects: Vec<(Obstacle, SphereOptions)> = vec![];
        let mut object_of = Vec::with_capacity(helds.len());
        for held in helds {
            let object = held.object.placed(Pose::IDENTITY);
            let options = held.attachment(robot)?.spheres;
            object_of.push(match objects.iter().position(|(o, _)| *o == object) {
                Some(k) => k,
                None => {
                    objects.push((object, options));
                    objects.len() - 1
                }
            });
        }
        let fits = device.fits();
        let spheres = device.install(|| {
            objects
                .par_iter()
                .map(|(object, options)| {
                    if let Some(spheres) = fits.get(object, options) {
                        return Ok(spheres);
                    }
                    let fitted: Arc<[[f32; 4]]> =
                        robot::object_spheres(std::slice::from_ref(object), options)?.into();
                    fits.keep(object.clone(), *options, fitted.clone());
                    Ok(fitted)
                })
                .collect::<Result<_>>()
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

    /// Devices on one GPU or thread pool fit a held object once and keep it, the least recently
    /// used going first.
    #[test]
    fn devices_on_one_hardware_fit_a_held_object_once() {
        let robot = panda();
        let device = Device::cpu(&robot).unwrap();
        let can = Held {
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
}
