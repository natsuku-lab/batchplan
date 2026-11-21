//! Turning things about a line (`turn`): a knob, a bar standing proud of a round dial, seen by two
//! depth cameras, its grips taken from what they saw and ordered by the moment they give about its
//! axis, turned a quarter turn; a cap unscrewed half a turn along its thread; a lever swung 60
//! degrees about its hinge; a valve's wheel turned 45 degrees by its rim; then 64 knobs in 64
//! worlds in one call. For each, how many grips, how many get through and where the rest stop, and
//! the call's time, the quickest of five, and the time to the first plan (`Turn::enough`); for the
//! knob, how many grips make the turn from the configuration nearest the start against from any IK
//! finds at the grip. The 64 knobs are turned to their first plans.
//!
//!   cargo run --release --example turns            # on the CPU
//!   cargo run --release --example turns -- --gpu   # on the GPU

#[path = "common/mod.rs"]
mod common;

use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, PI, TAU};
use std::time::Instant;

use anyhow::Result;
use batchplan::rng::Rng;
use batchplan::{
    DepthImage, Device, GraspOptions, IkProblem, Intrinsics, Obstacle, Pose, Robot, Screw, Surface,
    TaskFailure, TaskOptions, TaskPlan, TaskResult, TaskStep, ToolPath, ToolPathOptions, Touch,
    Turn, World, Worlds,
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

/// One thing to turn: its world, which obstacle it is, the grips to try in the order to try them
/// (in its frame), the line it turns about and how far.
struct Thing {
    world: World,
    object: usize,
    grips: Vec<Pose>,
    screw: Screw,
    angle: f32,
}

/// A knob at `at` on the table: a box 24 x 20 x 10 cm, a dial 9 cm across and 1 cm thick on it,
/// and a bar 9 x 2.5 cm standing 3 cm proud of the dial, turned `yaw` about the vertical. Its
/// grips are taken from two depth images, one from either side across the bar, with the bar's
/// readings flagged,
/// and ordered by the moment they give about the knob's axis, the most first.
fn knob(device: &Device, at: Vec2, yaw: f32) -> Result<Thing> {
    let console = Obstacle::cuboid(at.extend(0.05), Vec3::new(0.12, 0.1, 0.05));
    let dial = Obstacle::cylinder(at.extend(0.105), 0.045, 0.005);
    let half = Vec3::new(0.045, 0.0125, 0.015);
    let bar = Obstacle::cuboid(Vec3::ZERO, half).placed(Pose::new(
        at.extend(0.11 + half.z),
        Quat::from_rotation_z(yaw),
    ));
    let scene = [table(), console, dial, bar.clone()];
    let k = Intrinsics {
        fx: 600.0,
        fy: 600.0,
        cx: WIDTH as f32 / 2.0,
        cy: HEIGHT as f32 / 2.0,
    };
    let middle = bar.pose().position;
    let mut seen = Surface::default();
    // One camera on either side of the bar, across it, 30 cm out and 30 cm up.
    let across = Quat::from_rotation_z(yaw) * Vec3::Y;
    for side in [1.0, -1.0] {
        let camera = look_at(middle + across * side * 0.3 + Vec3::Z * 0.3, middle);
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
    // The fingers come in as narrow as keeps them off the dial, above its top.
    let o = GraspOptions {
        narrow: true,
        floor: Some(0.11),
        ..GraspOptions::default()
    };
    let screw = Screw::about(at.extend(0.11), Vec3::Z);
    let mut found = batchplan::grasps_on(device, 0, std::slice::from_ref(&seen), &o)?.remove(0);
    found.sort_by(|a, b| b.moment(&screw, &o).total_cmp(&a.moment(&screw, &o)));
    let to_bar = bar.pose().inverse();
    Ok(Thing {
        world: World::new(scene.to_vec()),
        object: 3,
        grips: found.iter().map(|g| to_bar.mul_pose(g.pose)).collect(),
        screw,
        angle: FRAC_PI_2,
    })
}

/// A cap 4 cm across and 2.5 cm tall on a bottle 7 cm across and 15 cm tall, its thread 5 mm a
/// turn, unscrewed half a turn: anticlockwise from above, up 2.5 mm.
fn cap(robot: &Robot) -> Result<Thing> {
    let bottle = Obstacle::cylinder(Vec3::new(0.45, 0.2, 0.075), 0.035, 0.075);
    let cap = Obstacle::cylinder(Vec3::new(0.45, 0.2, 0.1625), 0.02, 0.0125);
    let o = GraspOptions {
        floor: Some(0.15),
        tip: robot.tip(0),
        ..GraspOptions::default()
    };
    Ok(Thing {
        grips: batchplan::grasps(&cap, &o)?,
        world: World::new(vec![table(), bottle, cap]),
        object: 2,
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
    let o = GraspOptions {
        tip: robot.tip(0),
        ..GraspOptions::default()
    };
    Ok(Thing {
        grips: batchplan::grasps(&arm, &o)?,
        world: World::new(vec![table(), post, arm]),
        object: 2,
        screw: Screw::about(hinge, Vec3::Z),
        angle: FRAC_PI_3,
    })
}

/// A valve's wheel 20 cm across and 2 cm thick on a pipe 25 cm tall, gripped on its rim and
/// turned 45 degrees.
fn wheel(robot: &Robot) -> Result<Thing> {
    let pipe = Obstacle::cylinder(Vec3::new(0.5, 0.15, 0.12), 0.02, 0.12);
    let rim = Obstacle::cylinder(Vec3::new(0.5, 0.15, 0.25), 0.1, 0.01);
    let o = GraspOptions {
        tip: robot.tip(0),
        ..GraspOptions::default()
    };
    Ok(Thing {
        grips: batchplan::grasps(&rim, &o)?,
        world: World::new(vec![table(), pipe, rim]),
        object: 2,
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
        let left = (0..t.grasps.len()).filter(|g| !through.contains(g));
        Turn {
            grasps: left.map(|g| t.grasps[g]).collect(),
            ..t.clone()
        }
    });
    each.collect()
}

/// Of `thing`'s grips, how many make the turn from the configuration IK finds nearest the start,
/// and how many from any it finds, followed as `turn` follows them.
fn from_nearest(
    robot: &Robot,
    device: &Device,
    thing: &Thing,
    o: &TaskOptions,
) -> Result<[usize; 2]> {
    let start = robot.default_q();
    let mut world = thing.world.clone();
    world
        .touch
        .push(Touch::new(thing.object, robot.finger_links(0)));
    let worlds = device.upload(&[world])?;
    let placed = thing.world.obstacles[thing.object].pose();
    let at: Vec<Pose> = thing.grips.iter().map(|&g| placed.mul_pose(g)).collect();
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
    let solved = batchplan::solve_ik(device, &worlds, &problems, &o.ik)?;
    let along = ToolPathOptions {
        precision: o.tool_path.precision / 2.0,
        ..o.tool_path.clone()
    };
    let (mut paths, mut whose) = (vec![], vec![]);
    for (g, &pose) in at.iter().enumerate() {
        let waypoints = thing.screw.path(pose, thing.angle, along.precision)?;
        let distance = |i: usize| -> f32 {
            let q = solved.solution(i);
            q.iter().zip(start).map(|(a, b)| (a - b).powi(2)).sum()
        };
        let mut found: Vec<usize> = (g * solved.seeds..(g + 1) * solved.seeds)
            .filter(|&i| solved.success[i])
            .collect();
        found.sort_by(|&a, &b| distance(a).total_cmp(&distance(b)));
        for (k, &i) in found.iter().enumerate() {
            paths.push(ToolPath {
                world: 0,
                start: solved.solution(i).to_vec(),
                hand: 0,
                waypoints: waypoints.clone(),
                hold: vec![],
                holding: None,
            });
            whose.push((g, k == 0));
        }
    }
    let followed = batchplan::follow(device, &worlds, &paths, &along)?;
    let (mut nearest, mut any) = (vec![false; at.len()], vec![false; at.len()]);
    for (&(g, first), &f) in whose.iter().zip(&followed.fraction) {
        if f == 1.0 {
            any[g] = true;
            nearest[g] |= first;
        }
    }
    let count = |v: &[bool]| v.iter().filter(|&&b| b).count();
    Ok([count(&nearest), count(&any)])
}

/// Each of `things` as a task of a world of its own, by its grips in their order, `enough` plans
/// of each to do.
fn tasks_of(robot: &Robot, things: &[Thing], enough: Option<usize>) -> Vec<Turn> {
    let each = things.iter().enumerate().map(|(w, t)| Turn {
        screw: Some(t.screw),
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

fn main() -> Result<()> {
    let gpu = std::env::args().any(|a| a == "--gpu");
    let robot = common::panda()?;
    let device = if gpu {
        Device::gpu(&robot)?
    } else {
        Device::cpu(&robot)?
    };
    println!("{}", device.name());
    let o = TaskOptions::default();
    // A first call builds what a device builds once.
    let knob_a = knob(&device, Vec2::new(0.5, 0.0), 0.4)?;
    let warm = Turn::new(
        0,
        robot.default_q(),
        knob_a.object,
        knob_a.grips[..1].to_vec(),
        0.1,
    );
    batchplan::turn(
        &device,
        &device.upload(std::slice::from_ref(&knob_a.world))?,
        &[warm],
        &o,
    )?;
    report("a knob, a quarter turn", &device, &robot, &knob_a, &o)?;
    let [nearest, any] = from_nearest(&robot, &device, &knob_a, &o)?;
    println!(
        "    of its grips, {nearest} make the turn from the configuration nearest the start, {any} from some configuration IK finds there"
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
