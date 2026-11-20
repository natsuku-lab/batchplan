//! Where a held object rests on what cameras see, and what finding it costs. A box on a table
//! among other things, seen in two 640x480 depth images; a can in a basket seen from above, and
//! where to let it go to fall in; a tray laid across two bars, and a can laid along them, which
//! rolls in between; a bottle laid flat on a sloping rack against the lip at its foot; a bottle
//! stood upright on that rack, which the bare lip does not hold and a lip with a top does, found
//! by looking closer, with why it has none where it has none; then 64 boxes of as many heights on
//! the one table in one call, which reads the table once.
//!
//! cargo run --release --example surface_rests [-- threads] [--dump dir]
//!
//! With `--dump`, each scene's points and rests are written to `dir`, for
//! `scripts/validate_rests.py` to let go of each rest in MuJoCo and see that it stays.

#[path = "common/mod.rs"]
mod common;

use std::f32::consts::{FRAC_PI_2, TAU};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use batchplan::{
    DepthImage, Device, Intrinsics, Obstacle, Pose, Rest, RestOptions, RestProblem, Surface,
    falls_into, rest_failures, rests_on,
};
use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

/// A camera at `eye` looking at `target`, in OpenCV axes (x right, y down, z forward).
fn look_at(eye: Vec3, target: Vec3) -> Pose {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Z).try_normalize().unwrap_or(Vec3::X);
    let down = forward.cross(right);
    Pose::new(eye, Quat::from_mat3(&Mat3::from_cols(right, down, forward)))
}

/// What a camera at `eye` looking at `target` sees of `scene`, the points of `support` (indices
/// into `scene`) marked as the support, as a segmentation would mark them.
fn seen(
    device: &Device,
    scene: &[Obstacle],
    support: &[usize],
    eye: Vec3,
    target: Vec3,
) -> Result<Surface> {
    let k = Intrinsics {
        fx: 600.0,
        fy: 600.0,
        cx: WIDTH as f32 / 2.0,
        cy: HEIGHT as f32 / 2.0,
    };
    let camera = look_at(eye, target);
    let point = |i: usize, z: f32| {
        let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
        camera.rotation * Vec3::new((u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z)
            + camera.position
    };
    let depth: Vec<f32> = (0..WIDTH * HEIGHT)
        .into_par_iter()
        .map(|i| {
            let (u, v) = ((i % WIDTH) as f32, (i / WIDTH) as f32);
            let ray = Vec3::new((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0);
            let dir = camera.rotation * ray.normalize();
            let mut t = 0.0;
            while t < 2.0 {
                let p = camera.position + dir * t;
                let d = scene
                    .iter()
                    .map(|o| o.distance(p).value)
                    .fold(f32::INFINITY, f32::min);
                if d < 1e-5 {
                    return t * ray.normalize().z;
                }
                t += d;
            }
            0.0
        })
        .collect();
    let flags: Vec<bool> = (0..depth.len())
        .map(|i| {
            let p = point(i, depth[i]);
            let distance = |j: usize| scene[j].distance(p).value;
            let nearest = (0..scene.len()).min_by(|&a, &b| distance(a).total_cmp(&distance(b)));
            depth[i] > 0.0 && nearest.is_some_and(|j| support.contains(&j))
        })
        .collect();
    let image = DepthImage {
        depth: &depth,
        width: WIDTH,
        intrinsics: k,
        camera,
    };
    Ok(Surface::from_depth(device, &image, &flags, None, 0.005)?)
}

/// Points 2 mm apart around a bar of radius `r` along y at `(x, z)`, 30 cm long, with normals.
fn bar(x: f32, z: f32, r: f32, s: &mut Surface) {
    let around = (TAU * r / 0.002).round() as usize;
    for k in 0..around {
        let angle = TAU * k as f32 / around as f32;
        let normal = Vec3::new(angle.cos(), 0.0, angle.sin());
        for j in 0..=150 {
            s.points
                .push(Vec3::new(x, j as f32 * 0.002 - 0.15, z) + normal * r);
            s.normals.push(normal);
        }
    }
}

/// A plane through `middle` with normal `normal`, `half` wide along `across`, points 2 mm apart.
fn plane(middle: Vec3, normal: Vec3, across: [Vec3; 2], half: [f32; 2], s: &mut Surface) {
    let steps = |h: f32| (2.0 * h / 0.002).round() as i32;
    for i in 0..=steps(half[0]) {
        for j in 0..=steps(half[1]) {
            let (u, v) = (i as f32 * 0.002 - half[0], j as f32 * 0.002 - half[1]);
            s.points.push(middle + across[0] * u + across[1] * v);
            s.normals.push(normal);
        }
    }
}

/// A sum over every number of `rests`, to tell whether two builds find the same.
fn sum(rests: &[Rest]) -> f64 {
    let of = |v: Vec3| f64::from(v.x) + 2.0 * f64::from(v.y) + 3.0 * f64::from(v.z);
    let each = rests.iter().enumerate().map(|(k, r)| {
        let turn = r.pose.rotation * Vec3::new(1.0, 2.0, 3.0);
        let support: f64 = r.support.iter().map(|&c| of(c)).sum();
        (k % 7 + 1) as f64 * (of(r.pose.position) + of(turn) + support + f64::from(r.tipping))
    });
    each.sum()
}

/// Writes a scene and its rests to `dir`, under the first word of `what`: the support's points and
/// the rests as little-endian `f32`s (three to a point; a rest's position, its rotation as x, y,
/// z, w, and its tipping), and the object and the options as JSON.
fn write(
    dir: &Path,
    what: &str,
    s: &Surface,
    object: &Obstacle,
    rests: &[Rest],
    o: &RestOptions,
) -> Result<()> {
    let name = what.split([' ', ',']).nth(1).unwrap_or("scene");
    let support = (0..s.points.len()).filter(|&k| s.object.is_empty() || s.object[k]);
    // What holds the object up, as `rests_on` reads it: not a point that faces down and was not
    // seen from above.
    let from_above = |k: usize| {
        s.seen_from
            .get(k)
            .is_some_and(|f| f.is_finite() && f.z > s.points[k].z)
    };
    let facing = support.filter(|&k| s.normals[k].z >= -0.5 || from_above(k));
    let points: Vec<u8> = facing
        .flat_map(|k| s.points[k].to_array())
        .flat_map(f32::to_le_bytes)
        .collect();
    std::fs::write(dir.join(format!("{name}.points.f32")), points)?;
    let rows = rests.iter().flat_map(|r| {
        let (p, q) = (r.pose.position, r.pose.rotation);
        [p.x, p.y, p.z, q.x, q.y, q.z, q.w, r.tipping]
    });
    let rows: Vec<u8> = rows.flat_map(f32::to_le_bytes).collect();
    std::fs::write(dir.join(format!("{name}.rests.f32")), rows)?;
    let shape = match *object {
        Obstacle::Cuboid { half_extents, .. } => {
            format!(r#"{{"box": {:?}}}"#, half_extents.to_array())
        }
        Obstacle::Cylinder {
            radius,
            half_height,
            ..
        } => format!(r#"{{"cylinder": [{radius}, {half_height}]}}"#),
        Obstacle::Sphere { radius, .. } => format!(r#"{{"sphere": [{radius}]}}"#),
        Obstacle::Capsule {
            radius,
            half_length,
            ..
        } => format!(r#"{{"capsule": [{radius}, {half_length}]}}"#),
        _ => anyhow::bail!("a distance grid has no rests"),
    };
    let about = format!(
        r#"{{"what": {what:?}, "object": {shape}, "friction": {}, "touching": {}}}"#,
        o.friction, o.touching
    );
    std::fs::write(dir.join(format!("{name}.json")), about)?;
    Ok(())
}

/// The quickest of `rounds` runs of `run`, in milliseconds, and its last result.
fn quickest<T>(rounds: usize, mut run: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..rounds {
        let start = Instant::now();
        last = Some(run());
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
    }
    (best, last.expect("at least one round"))
}

fn main() -> Result<()> {
    let robot = common::panda()?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dump = args.iter().position(|a| a == "--dump");
    let dump = dump.and_then(|k| args.get(k + 1)).map(PathBuf::from);
    let device = match args.first().filter(|a| !a.starts_with("--")) {
        Some(threads) => Device::cpu_threads(&robot, threads.parse()?)?,
        None => Device::cpu(&robot)?,
    };
    if let Some(dir) = &dump {
        std::fs::create_dir_all(dir)?;
    }
    let o = RestOptions::default();
    let report = |what: &str,
                  s: &Surface,
                  object: Obstacle,
                  rotation: Quat,
                  o: &RestOptions|
     -> Result<Vec<Rest>> {
        let problems = [RestProblem {
            rotation,
            ..RestProblem::new(object, 0)
        }];
        let surfaces = std::slice::from_ref(s);
        let (ms, found) = quickest(5, || rests_on(&device, &problems, surfaces, o));
        let found = found?.remove(0);
        if let Some(dir) = &dump {
            write(dir, what, s, &problems[0].object, &found, o)?;
        }
        let support = if s.object.is_empty() {
            s.points.len()
        } else {
            s.object.iter().filter(|&&f| f).count()
        };
        println!(
            "{what}: {} points, {support} the support's: {} rests in {ms:.1} ms (sum {:.6})",
            s.points.len(),
            found.len(),
            sum(&found)
        );
        Ok(found)
    };

    // A box 4 by 6 by 5 cm set down on a table 60 cm square among three others, seen from the
    // robot's side and from above.
    let table = Obstacle::cuboid(Vec3::new(0.5, 0.0, -0.01), Vec3::new(0.3, 0.3, 0.01));
    let mut scene = vec![table];
    for (x, y, h) in [(0.45, 0.1, 0.05), (0.6, -0.05, 0.08), (0.4, -0.15, 0.03)] {
        scene.push(Obstacle::cuboid(
            Vec3::new(x, y, h),
            Vec3::new(0.03, 0.04, h),
        ));
    }
    let middle = Vec3::new(0.5, 0.0, 0.0);
    let mut clutter = seen(&device, &scene, &[0], Vec3::new(0.0, 0.0, 0.6), middle)?;
    clutter.join(seen(
        &device,
        &scene,
        &[0],
        Vec3::new(0.5, 0.0, 0.8),
        middle,
    )?);
    let block = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.02, 0.03, 0.025));
    report(
        "a table among three blocks in two depth images",
        &clutter,
        block,
        Quat::IDENTITY,
        &o,
    )?;

    // A can in a basket 20 cm square, seen from above as a wrist camera looks in.
    let mut basket = vec![Obstacle::cuboid(
        Vec3::new(0.5, 0.0, 0.005),
        Vec3::new(0.1, 0.1, 0.005),
    )];
    for (x, y, hx, hy) in [
        (0.095, 0.0, 0.005, 0.1),
        (-0.095, 0.0, 0.005, 0.1),
        (0.0, 0.095, 0.1, 0.005),
        (0.0, -0.095, 0.1, 0.005),
    ] {
        basket.push(Obstacle::cuboid(
            Vec3::new(0.5 + x, y, 0.06),
            Vec3::new(hx, hy, 0.06),
        ));
    }
    let inside = seen(
        &device,
        &basket,
        &[0, 1, 2, 3, 4],
        Vec3::new(0.5, 0.0, 0.5),
        middle,
    )?;
    let can = Obstacle::cylinder(Vec3::ZERO, 0.033, 0.04);
    report(
        "a basket seen from above",
        &inside,
        can.clone(),
        Quat::IDENTITY,
        &o,
    )?;
    // Where to let that can go over the basket for it to fall in and stay.
    let into = [RestProblem::new(can.clone(), 0)];
    let basket_seen = std::slice::from_ref(&inside);
    let (ms, falls) = quickest(5, || falls_into(&device, &into, basket_seen, &o));
    let falls = falls?.remove(0);
    let first = falls.first().map(|f| f.pose.position);
    println!(
        "the can let go over that basket to fall in: {} places in {ms:.1} ms, the first at {first:?}",
        falls.len()
    );

    // A tray 12 by 6 cm laid across two bars 2 cm thick, 8 cm apart.
    let mut bars = Surface::default();
    bar(0.04, 0.1, 0.01, &mut bars);
    bar(-0.04, 0.1, 0.01, &mut bars);
    let tray = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.06, 0.03, 0.02));
    report(
        "two bars, a tray laid across them",
        &bars,
        tray,
        Quat::IDENTITY,
        &o,
    )?;

    // A can laid along those bars: set down beside the middle between them it lies on one of
    // them, and rolls in to lie on both.
    let lying = Quat::from_rotation_x(FRAC_PI_2);
    report("a can laid along two bars", &bars, can, lying, &o)?;

    // A bottle 16 cm long laid across a rack sloping 30 degrees, a lip 1 cm high at its foot:
    // turned to lie flat on the slope, which is steeper than the friction holds, it rests only
    // against the lip, where spots 2 mm apart come within `touching` of it. Turned about the
    // vertical it would not lie flat: the one turn is tried.
    let tilt = 30f32.to_radians();
    let (normal, down) = (
        Vec3::new(0.0, tilt.sin(), tilt.cos()),
        Vec3::new(0.0, tilt.cos(), -tilt.sin()),
    );
    let mut rack = Surface::default();
    plane(Vec3::ZERO, normal, [Vec3::X, down], [0.15, 0.06], &mut rack);
    let foot = down * 0.06;
    plane(
        foot + Vec3::Z * 0.005,
        Vec3::NEG_Y,
        [Vec3::X, Vec3::Z],
        [0.15, 0.005],
        &mut rack,
    );
    let bottle = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.019, 0.023, 0.079));
    let flat_across = Quat::from_rotation_x(-tilt) * Quat::from_rotation_y(FRAC_PI_2);
    let fine = RestOptions {
        turns: 1,
        spacing: 0.002,
        ..o
    };
    report(
        "a sloping rack with a lip, a bottle laid flat on it",
        &rack,
        bottle,
        flat_across,
        &fine,
    )?;

    // A wine bottle stood upright on that rack, the default options. Against the bare lip it has
    // no rest: for its rim behind to reach the slope its middle would stand past the lip. With
    // the lip's top 1 cm wide, as a rack's lip has, it stands against the lip, its rim behind on
    // the slope and its bottom on the lip's top, in a strip 2 mm deep between the spots' rows,
    // which looking closer about the spots where it nearly holds finds along the lip.
    let wine = Obstacle::cylinder(Vec3::ZERO, 0.0215, 0.079);
    let bare = report(
        "a bare lip of that rack, a bottle stood upright on it",
        &rack,
        wine.clone(),
        Quat::IDENTITY,
        &o,
    )?;
    let upright = [RestProblem::new(wine.clone(), 0)];
    let racks = std::slice::from_ref(&rack);
    let (ms, why) = quickest(5, || {
        rest_failures(&device, &upright, racks, std::slice::from_ref(&bare), &o)
    });
    println!("  why none, in {ms:.1} ms: {:?}", why?[0]);
    let mut topped = rack.clone();
    plane(
        foot + Vec3::new(0.0, 0.005, 0.01),
        Vec3::Z,
        [Vec3::X, Vec3::Y],
        [0.15, 0.005],
        &mut topped,
    );
    report(
        "an upright bottle against the lip of that rack, its top 1 cm wide",
        &topped,
        wine,
        Quat::IDENTITY,
        &o,
    )?;

    // 64 boxes 4 by 6 cm and 2 to 8.3 cm tall on the one table, as a call asks what of many held
    // objects rests where: the table is read once for all of them.
    let boxes: Vec<RestProblem> = (0..64)
        .map(|k| {
            let half = Vec3::new(0.02, 0.03, 0.01 + 0.0005 * k as f32);
            RestProblem::new(Obstacle::cuboid(Vec3::ZERO, half), 0)
        })
        .collect();
    let table = std::slice::from_ref(&clutter);
    let (ms, found) = quickest(3, || rests_on(&device, &boxes, table, &o));
    let rests: usize = found?.iter().map(Vec::len).sum();
    println!(
        "64 boxes on that table in one call: {rests} rests in {ms:.1} ms, {:.1} ms a box",
        ms / 64.0
    );
    Ok(())
}
