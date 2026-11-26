//! Turning things about a line (`turn`): a knob, a bar standing proud of a round dial, seen by two
//! depth cameras, its grips taken from what they saw and ordered by the moment they give about its
//! axis, turned a quarter turn; a cap unscrewed half a turn along its thread; a lever swung 60
//! degrees about its hinge; a valve's wheel turned 45 degrees by its rim; then 64 knobs in 64
//! worlds in one call. For each, how many grips, how many get through and where the rest stop, and
//! the call's time, the quickest of five, and the time to the first plan (`Turn::enough`); for the
//! knob, how many grips make the turn from the configuration nearest the start against from any IK
//! finds at the grip, and how many configurations IK finds at a grip. The 64 knobs are turned to
//! their first plans.
//!
//!   cargo run --release --example turns            # on the CPU
//!   cargo run --release --example turns -- --gpu   # on the GPU
//!   cargo run --release --example turns -- --fixtures  # a drawer draw and a hinged door swing

#[path = "common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, PI, TAU};
use std::time::Instant;

use anyhow::Result;
use batchplan::rng::Rng;
use batchplan::{
    DepthImage, Device, Gripper, GripperCalibration, Held, IkProblem, Intrinsics, Obstacle, Pose,
    Robot, Screw, ShapeGraspOptions, Surface, SurfaceGraspOptions, TaskFailure, TaskOptions,
    TaskPlan, TaskResult, TaskStep, ToolPath, ToolPathOptions, Touch, Turn, World, Worlds,
};
use glam::{Mat3, Quat, Vec2, Vec3};
use rayon::prelude::*;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const RUNS: usize = 5;

/// A camera at `eye` looking at `target`, in OpenCV axes (x right, y down, z forward).
fn look_at(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).normalize();
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
}

/// z depth per pixel by sphere tracing the scene's distance function; 0 where a ray hits nothing.
fn render(distance: impl Fn(Vec3) -> f32 + Sync, camera: Pose, k: Intrinsics) -> Vec<f32> {
    (0..WIDTH * HEIGHT)
        .into_par_iter()
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let ray = Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
            let dir = camera.rotation * ray.normalize();
            let mut t = 0.0;
            while t < 2.0 {
                let p = camera.position + dir * t;
                let d = distance(p);
                if d < 1e-5 {
                    return t * ray.normalize().z;
                }
                t += d;
            }
            0.0
        })
        .collect()
}

fn table() -> Obstacle {
    Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02))
}

/// One thing to turn: its world, which obstacle it is and what it turns on, the grips to try in
/// the order to try them (in its frame) and the fingers' joints at each, the line it turns about
/// and how far.
struct Thing {
    world: World,
    object: usize,
    support: Vec<usize>,
    grips: Vec<Pose>,
    locked: Vec<HashMap<String, f32>>,
    screw: Screw,
    angle: f32,
}

/// Read a handle from the scene through both cameras, flagging the handle's surface pixels.
fn seen_bar(
    device: &Device,
    scene: &[Obstacle],
    bar: &Obstacle,
    cameras: [Pose; 2],
) -> Result<Surface> {
    let k = Intrinsics {
        fx: 600.0,
        fy: 600.0,
        cx: WIDTH as f32 / 2.0,
        cy: HEIGHT as f32 / 2.0,
    };
    let mut seen = Surface::default();
    for camera in cameras {
        let depth = render(
            |p| {
                let each = scene.iter().map(|o| o.distance(p).value);
                each.fold(f32::INFINITY, f32::min)
            },
            camera,
            k,
        );
        let image = DepthImage {
            depth: &depth,
            width: WIDTH,
            intrinsics: k,
            camera,
        };
        // The bar's pixels, as a segmentation would give them.
        let object: Vec<bool> = (0..depth.len())
            .map(|i| {
                let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
                let z = depth[i];
                let p = camera.rotation
                    * Vec3::new((u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z)
                    + camera.position;
                z > 0.0 && bar.distance(p).value < 5e-4
            })
            .collect();
        seen.join(Surface::from_depth(device, &image, &object, None, 0.005)?);
    }
    Ok(seen)
}

/// A knob at `at` on the table: a box 24 x 20 x 10 cm, a dial 9 cm across and 1 cm thick on it,
/// and a bar 9 x 2.5 cm standing 3 cm proud of the dial, turned `yaw` about the vertical. Its
/// grips are taken from two depth images, one from either side across the bar, with the bar's
/// readings flagged, and ordered by the moment they give about the knob's axis, the most first;
/// the fingers are locked at each grip's opening, each standing at half of it.
fn knob(device: &Device, at: Vec2, yaw: f32) -> Result<Thing> {
    let console = Obstacle::cuboid(at.extend(0.05), Vec3::new(0.12, 0.1, 0.05));
    let dial = Obstacle::cylinder(at.extend(0.105), 0.045, 0.005);
    let half = Vec3::new(0.045, 0.0125, 0.015);
    let bar = Obstacle::cuboid(Vec3::ZERO, half).placed(Pose::new(
        at.extend(0.11 + half.z),
        Quat::from_rotation_z(yaw),
    ));
    let scene = [table(), console, dial, bar.clone()];
    let middle = bar.pose().position;
    let across = Quat::from_rotation_z(yaw) * Vec3::Y;
    let cameras =
        [1.0, -1.0].map(|side| look_at(middle + across * side * 0.3 + Vec3::Z * 0.3, middle));
    let seen = seen_bar(device, &scene, &bar, cameras)?;
    // The fingers come in as narrow as keeps them off the dial, above its top.
    let o = SurfaceGraspOptions {
        narrow: true,
        floor: Some(0.11),
        ..SurfaceGraspOptions::default()
    };
    let screw = Screw::about(at.extend(0.11), Vec3::Z);
    // The Panda's pads 8 cm apart as its fingers are held open, touching closed.
    let calibration = GripperCalibration::new(0.08, 0.0, common::panda_closed());
    let gripper = device.robot().gripper(0, &calibration)?;
    let mut found =
        batchplan::grasps_on(device, &gripper, std::slice::from_ref(&seen), &o)?.remove(0);
    o.order_for_turn(&gripper, &mut found, Some(screw));
    let to_bar = bar.pose().inverse();
    let fingers = |opening: f32| {
        let each = ["panda_finger_joint1", "panda_finger_joint2"];
        each.map(|j| (j.to_string(), opening / 2.0)).into()
    };
    Ok(Thing {
        world: World::new(scene.to_vec()),
        object: 3,
        support: vec![2],
        grips: found.iter().map(|g| to_bar.mul_pose(g.pose)).collect(),
        locked: found.iter().map(|g| fingers(g.opening)).collect(),
        screw,
        angle: FRAC_PI_2,
    })
}

/// The Panda's hand of `robot` for grasps of a shape, its pads 8 cm apart: they need no fingers
/// that slide.
fn shape_hand(robot: &Robot) -> Result<Gripper> {
    Ok(robot.gripper(0, &GripperCalibration::new(0.08, 0.0, HashMap::new()))?)
}

/// A cap 4 cm across and 2.5 cm tall on a bottle 7 cm across and 15 cm tall, its thread 5 mm a
/// turn, unscrewed half a turn: anticlockwise from above, up 2.5 mm.
fn cap(robot: &Robot) -> Result<Thing> {
    let bottle = Obstacle::cylinder(Vec3::new(0.45, 0.2, 0.075), 0.035, 0.075);
    let cap = Obstacle::cylinder(Vec3::new(0.45, 0.2, 0.1625), 0.02, 0.0125);
    let o = ShapeGraspOptions {
        floor: Some(0.15),
        ..ShapeGraspOptions::default()
    };
    Ok(Thing {
        grips: batchplan::grasps(&cap, &shape_hand(robot)?, &o)?,
        locked: vec![],
        world: World::new(vec![table(), bottle, cap]),
        object: 2,
        support: vec![1],
        screw: Screw {
            point: Vec3::new(0.45, 0.2, 0.15),
            direction: Vec3::Z,
            pitch: 0.005 / TAU,
        },
        angle: PI,
    })
}

/// A lever on a post 15 cm tall, its hinge upright through the post: an arm 15 cm long, 2 cm wide
/// and 1.5 cm thick, swung 60 degrees.
fn lever(robot: &Robot) -> Result<Thing> {
    let hinge = Vec3::new(0.45, -0.2, 0.15);
    let post = Obstacle::cylinder(Vec3::new(hinge.x, hinge.y, 0.07), 0.015, 0.07);
    let arm = Obstacle::cuboid(
        hinge + Vec3::new(0.06, 0.0, 0.0075),
        Vec3::new(0.075, 0.01, 0.0075),
    );
    let o = ShapeGraspOptions::default();
    Ok(Thing {
        grips: batchplan::grasps(&arm, &shape_hand(robot)?, &o)?,
        locked: vec![],
        world: World::new(vec![table(), post, arm]),
        object: 2,
        support: vec![1],
        screw: Screw::about(hinge, Vec3::Z),
        angle: FRAC_PI_3,
    })
}

/// A valve's wheel 20 cm across and 2 cm thick on a pipe 25 cm tall, gripped on its rim and
/// turned 45 degrees.
fn wheel(robot: &Robot) -> Result<Thing> {
    let pipe = Obstacle::cylinder(Vec3::new(0.5, 0.15, 0.12), 0.02, 0.12);
    let rim = Obstacle::cylinder(Vec3::new(0.5, 0.15, 0.25), 0.1, 0.01);
    let o = ShapeGraspOptions::default();
    Ok(Thing {
        grips: batchplan::grasps(&rim, &shape_hand(robot)?, &o)?,
        locked: vec![],
        world: World::new(vec![table(), pipe, rim]),
        object: 2,
        support: vec![1],
        screw: Screw::about(Vec3::new(0.5, 0.15, 0.25), Vec3::Z),
        angle: FRAC_PI_4,
    })
}

/// Where the candidates of `outcomes` that did not get through stopped, counted by step.
fn stops(outcomes: &[std::result::Result<Vec<TaskPlan>, TaskFailure>]) -> String {
    let stops: Vec<TaskStep> = outcomes
        .iter()
        .filter_map(|o| o.as_ref().err())
        .flat_map(|f| f.candidates.iter().map(|c| c.step))
        .collect();
    let steps = [
        (TaskStep::Start, "start"),
        (TaskStep::Reach, "reach"),
        (TaskStep::Turn, "turn"),
        (TaskStep::Approach, "in"),
        (TaskStep::Depart, "away"),
        (TaskStep::Transit, "transit"),
    ];
    let each = steps.iter().map(|&(step, name)| {
        let count = stops.iter().filter(|&&s| s == step).count();
        format!("{name} {count}")
    });
    each.collect::<Vec<String>>().join(", ")
}

/// The best of `RUNS` timings of `f`, in milliseconds, and its last result.
fn timed<T>(mut f: impl FnMut() -> Result<T>) -> Result<(f64, T)> {
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..RUNS {
        let t = Instant::now();
        last = Some(f()?);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    Ok((best, last.expect("at least one run")))
}

/// Each task of `result` with only the grips that did not get through, if any: a call with them
/// says where each stopped.
fn the_rest(result: &TaskResult<Turn>) -> Vec<Turn> {
    let each = result.problems.iter().zip(&result.outcomes).map(|(t, o)| {
        let through: Vec<usize> = match o {
            Ok(plans) => plans.iter().map(|p| p.candidate).collect(),
            Err(_) => vec![],
        };
        let left: Vec<usize> = (0..t.grasps.len())
            .filter(|g| !through.contains(g))
            .collect();
        Turn {
            grasps: left.iter().map(|&g| t.grasps[g]).collect(),
            locked: left
                .iter()
                .filter_map(|&g| t.locked.get(g).cloned())
                .collect(),
            ..t.clone()
        }
    });
    each.collect()
}

/// Of `thing`'s grips, how many make the turn from the configuration IK finds nearest the start,
/// and how many from any it finds, followed as `turn` follows them: on the robot with its fingers
/// where the grip locks them, holding the object in its world without it; and how many
/// configurations IK finds at a grip, the fewest, the median and the most, over the grips that
/// have one, and how many those are.
fn from_nearest(
    robot: &Robot,
    device: &Device,
    thing: &Thing,
    o: &TaskOptions,
) -> Result<([usize; 2], [usize; 4])> {
    let start = robot.default_q();
    let mut touching = thing.world.clone();
    touching
        .touch
        .push(Touch::new(thing.object, robot.finger_links(0)));
    let mut without = thing.world.clone();
    let object = without.remove(thing.object)?;
    let shifted = |s: usize| if s > thing.object { s - 1 } else { s };
    let touches = thing.support.iter().map(|&s| Touch::held(shifted(s), 0));
    without.touch.extend(touches);
    let worlds = device.upload(&[touching, without])?;
    let placed = thing.world.obstacles[thing.object].pose();
    let along = ToolPathOptions {
        precision: o.tool_path.precision / 2.0,
        ..o.tool_path.clone()
    };
    // The grips by where they lock the fingers, each lot on the robot relocked so.
    let mut lots: Vec<(HashMap<String, f32>, Vec<usize>)> = vec![];
    for g in 0..thing.grips.len() {
        let locked = thing.locked.get(g).cloned().unwrap_or_default();
        match lots.iter_mut().find(|(l, _)| *l == locked) {
            Some((_, grips)) => grips.push(g),
            None => lots.push((locked, vec![g])),
        }
    }
    let (mut nearest, mut any) = (
        vec![false; thing.grips.len()],
        vec![false; thing.grips.len()],
    );
    let mut found_at: Vec<usize> = vec![];
    for (locked, grips) in &lots {
        let on = match locked.is_empty() {
            true => device.with_robot(robot)?,
            false => device.with_robot(&robot.relock(locked)?)?,
        };
        let at: Vec<Pose> = grips
            .iter()
            .map(|&g| placed.mul_pose(thing.grips[g]))
            .collect();
        let problems: Vec<IkProblem> = at
            .iter()
            .map(|&t| IkProblem {
                world: 0,
                targets: vec![Some(t)],
                seed: Some(start.to_vec()),
                hold: vec![],
                holding: None,
            })
            .collect();
        let solved = batchplan::solve_ik(&on, &worlds, &problems, &o.ik)?;
        let (mut paths, mut whose) = (vec![], vec![]);
        for (k, (&g, &pose)) in grips.iter().zip(&at).enumerate() {
            let waypoints = thing.screw.path(pose, thing.angle, along.precision)?;
            let distance = |i: usize| -> f32 {
                let q = solved.solution(i);
                q.iter().zip(start).map(|(a, b)| (a - b).powi(2)).sum()
            };
            let mut found: Vec<usize> = (k * solved.seeds..(k + 1) * solved.seeds)
                .filter(|&i| solved.success[i])
                .collect();
            found.sort_by(|&a, &b| distance(a).total_cmp(&distance(b)));
            if !found.is_empty() {
                found_at.push(found.len());
            }
            let held = Held {
                hand: 0,
                object: object.placed(Pose::IDENTITY),
                grasp: thing.grips[g],
                locked: HashMap::new(),
                spheres: Default::default(),
            };
            for (rank, &i) in found.iter().enumerate() {
                paths.push(ToolPath {
                    world: 1,
                    start: solved.solution(i).to_vec(),
                    hand: 0,
                    waypoints: waypoints.clone(),
                    hold: vec![],
                    holding: Some(held.clone()),
                });
                whose.push((g, rank == 0));
            }
        }
        let followed = batchplan::follow(&on, &worlds, &paths, &along)?;
        for (&(g, first), &f) in whose.iter().zip(&followed.fraction) {
            if f == 1.0 {
                any[g] = true;
                nearest[g] |= first;
            }
        }
    }
    let count = |v: &[bool]| v.iter().filter(|&&b| b).count();
    found_at.sort_unstable();
    let spread = match found_at.len() {
        0 => [0; 4],
        k => [found_at[0], found_at[k / 2], found_at[k - 1], k],
    };
    Ok(([count(&nearest), count(&any)], spread))
}

/// Each of `things` as a task of a world of its own, by its grips in their order, `enough` plans
/// of each to do.
fn tasks_of(robot: &Robot, things: &[Thing], enough: Option<usize>) -> Vec<Turn> {
    let each = things.iter().enumerate().map(|(w, t)| Turn {
        screw: Some(t.screw),
        support: t.support.clone(),
        locked: t.locked.clone(),
        enough,
        ..Turn::new(
            w as u32,
            robot.default_q(),
            t.object,
            t.grips.clone(),
            t.angle,
        )
    });
    each.collect()
}

/// Turns `thing` by every grip in one call, the quickest of five, and says how many got through
/// and where the rest stopped; then the call that stops at its first plan, as a caller that runs
/// one asks.
fn report(
    name: &str,
    device: &Device,
    robot: &Robot,
    thing: &Thing,
    o: &TaskOptions,
) -> Result<()> {
    let world = thing.world.clone();
    let uploaded: Worlds = device.upload(std::slice::from_ref(&world))?;
    let tasks = tasks_of(robot, std::slice::from_ref(thing), None);
    let (ms, result) = timed(|| Ok(batchplan::turn(device, &uploaded, &tasks, o)?))?;
    let first = tasks_of(robot, std::slice::from_ref(thing), Some(1));
    let (first_ms, _) = timed(|| Ok(batchplan::turn(device, &uploaded, &first, o)?))?;
    // The grips that did not get through, alone, say where each stopped: a task's plans leave
    // the others out.
    let rest: Vec<Turn> = the_rest(&result);
    let stopped = match rest[0].grasps.is_empty() {
        true => vec![],
        false => batchplan::turn(device, &uploaded, &rest, o)?.outcomes,
    };
    let through = result.outcomes[0].as_ref().map_or(0, Vec::len);
    println!(
        "  {name}: {} grips, {through} through, the rest stopping at {}: {ms:.1} ms; {first_ms:.1} ms to the first plan",
        tasks[0].grasps.len(),
        stops(&stopped)
    );
    Ok(())
}

/// The same handle on two fixtures: a drawer that comes straight out and a door whose hinge
/// stands behind and to one side of its grip. Two cameras on opposite sides of the handle's
/// height see the pads that close above and below it.
fn fixtures(device: &Device) -> Result<()> {
    let middle = Vec3::new(0.49, 0.04, 0.22);
    let bar = Obstacle::cuboid(middle, Vec3::new(0.012, 0.055, 0.012));
    let face = Obstacle::cuboid(Vec3::new(0.57, 0.0, 0.18), Vec3::new(0.018, 0.19, 0.18));
    let scene = vec![table(), face, bar.clone()];
    let cameras = [-1.0, 1.0]
        .map(|side| look_at(middle + Vec3::new(-0.24, side * 0.08, side * 0.15), middle));
    let surface = seen_bar(device, &scene, &bar, cameras)?;
    let options = SurfaceGraspOptions {
        near: Some(middle),
        approach: 0.05,
        ..Default::default()
    };
    // The Panda's pads 8 cm apart as its fingers are held open, touching closed.
    let calibration = GripperCalibration::new(0.08, 0.0, common::panda_closed());
    let gripper = device.robot().gripper(0, &calibration)?;
    let found = batchplan::grasps_on(device, &gripper, &[surface], &options)?.remove(0);
    let grasped: Vec<Pose> = found
        .iter()
        .map(|g| bar.pose().inverse().mul_pose(g.pose))
        .collect();
    let pulls = [batchplan::Pull {
        enough: Some(1),
        approach: options.approach,
        ..batchplan::Pull::new(
            0,
            device.robot().default_q(),
            2,
            grasped.clone(),
            Vec3::NEG_X,
            0.08,
        )
    }];
    let hinge = Vec3::new(0.55, -0.16, middle.z);
    let turns = [Turn {
        stays: true,
        screw: Some(Screw::about(hinge, Vec3::Z)),
        enough: Some(1),
        approach: options.approach,
        ..Turn::new(0, device.robot().default_q(), 2, grasped, FRAC_PI_4)
    }];
    let worlds = device.upload(&[World::new(scene)])?;
    let o = TaskOptions::default();
    let draw = batchplan::pull(device, &worlds, &pulls, &o)?
        .outcomes
        .remove(0)
        .map_err(|failure| anyhow::anyhow!("{failure}"))?;
    let swing = batchplan::turn(device, &worlds, &turns, &o)?
        .outcomes
        .remove(0)
        .map_err(|failure| anyhow::anyhow!("{failure}"))?;
    println!(
        "{} measured grasps; 8 cm drawer draw in {:.2} s, 45 degree door swing in {:.2} s",
        found.len(),
        draw[0].duration(),
        swing[0].duration()
    );
    Ok(())
}

fn main() -> Result<()> {
    let gpu = std::env::args().any(|a| a == "--gpu");
    let robot = common::panda()?;
    let device = if gpu {
        Device::gpu(&robot)?
    } else {
        Device::cpu(&robot)?
    };
    println!("{}", device.name());
    if std::env::args().any(|a| a == "--fixtures") {
        return fixtures(&device);
    }
    let o = TaskOptions::default();
    // A first call builds what a device builds once.
    let knob_a = knob(&device, Vec2::new(0.5, 0.0), 0.4)?;
    let mut warm = tasks_of(&robot, std::slice::from_ref(&knob_a), Some(1)).remove(0);
    warm.grasps.truncate(1);
    warm.locked.truncate(1);
    batchplan::turn(
        &device,
        &device.upload(std::slice::from_ref(&knob_a.world))?,
        &[warm],
        &o,
    )?;
    report("a knob, a quarter turn", &device, &robot, &knob_a, &o)?;
    let ([nearest, any], [fewest, median, most, grips]) =
        from_nearest(&robot, &device, &knob_a, &o)?;
    println!(
        "    of its grips, {nearest} make the turn from the configuration nearest the start, {any} from some configuration IK finds there"
    );
    println!(
        "    IK finds {fewest} to {most} configurations at a grip, the median {median}, over the {grips} grips it finds any at"
    );
    report(
        "a cap, half a turn up its thread",
        &device,
        &robot,
        &cap(&robot)?,
        &o,
    )?;
    report(
        "a lever, 60 degrees about its hinge",
        &device,
        &robot,
        &lever(&robot)?,
        &o,
    )?;
    report(
        "a wheel by its rim, 45 degrees",
        &device,
        &robot,
        &wheel(&robot)?,
        &o,
    )?;
    // 64 knobs at random places and turns on the table, each a task of a world of its own, every
    // grip of each given in the order of its moment, the first plan of each to do.
    let mut rng = Rng::new(11);
    let knobs: Vec<Thing> = (0..64)
        .map(|_| {
            let at = Vec2::new(rng.range(0.4, 0.6), rng.range(-0.3, 0.3));
            knob(&device, at, rng.range(-PI, PI))
        })
        .collect::<Result<_>>()?;
    let worlds: Vec<World> = knobs.iter().map(|k| k.world.clone()).collect();
    let uploaded = device.upload(&worlds)?;
    let tasks = tasks_of(&robot, &knobs, Some(1));
    let (ms, result) = timed(|| Ok(batchplan::turn(&device, &uploaded, &tasks, &o)?))?;
    let done = result.outcomes.iter().filter(|o| o.is_ok()).count();
    let grips: usize = tasks.iter().map(|t| t.grasps.len()).sum();
    println!(
        "  64 knobs in 64 worlds, a quarter turn each: {grips} grips, {done} knobs turned, the rest's grips stopping at {}: {ms:.1} ms to the first plan of each",
        stops(&result.outcomes)
    );
    Ok(())
}
