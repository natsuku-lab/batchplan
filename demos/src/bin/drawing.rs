//! UR5e arms drawing shapes with a pen: each arm's pen goes down onto its table, along a shape's
//! edges and back up in straight lines (`batchplan::follow`), every arm in one call. The pen may
//! touch the table (`World::touch`); with `--no-touch` every pen stops just above it instead. Each
//! motion is timed with the pen under 15 cm/s (`Trajectory::with_hand_speed`). Writes the motions
//! to `<out_dir>/drawing.json` (joint positions and the pen tip, 30 times a second) with the pen's
//! URDF beside it, for a viewer to replay.
//!
//!   cargo run --release --bin drawing -- ../data/drawing          # 20 arms
//!   cargo run --release --bin drawing -- ../data/drawing 64 --no-touch

use std::f32::consts::PI;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use batchplan::glam::Vec3;
use batchplan::rng::Rng;
use batchplan::{
    Device, HandSpeed, Obstacle, Pose, Robot, RobotOptions, ToolPath, ToolPathOptions, Touch,
    Trajectory, World,
};
use batchplan_demos::ASSETS;

/// A 12 cm pen on the flange, its tip the IK frame. Its collision cylinder is 12 mm thick and starts
/// 3 cm off the flange: fitted to the whole pen, or to a thicker one, its spheres come within a
/// millimetre of the wrist's, and the arm stops for self-collision as it draws.
const PEN: &str = r#"  <link name="pen">
    <visual><origin xyz="0 0 0.06"/><geometry><cylinder radius="0.006" length="0.12"/></geometry></visual>
    <collision><origin xyz="0 0 0.075"/><geometry><cylinder radius="0.006" length="0.09"/></geometry></collision>
  </link>
  <joint name="tool0-pen" type="fixed"><parent link="tool0"/><child link="pen"/></joint>
  <link name="pen_tip"/>
  <joint name="pen-pen_tip" type="fixed"><parent link="pen"/><child link="pen_tip"/><origin xyz="0 0 0.12"/></joint>
"#;
/// Samples per second.
const HZ: f32 = 30.0;
/// How high the pen comes down from and goes back up to, above a shape's first corner.
const ABOVE: f32 = 0.06;

/// A triangle, square, pentagon, hexagon or five-pointed star, 14-22 cm across, somewhere the arm
/// reaches with the pen pointing down.
fn shape(rng: &mut Rng) -> Vec<Vec3> {
    let center = Vec3::new(rng.range(0.38, 0.52), rng.range(-0.18, 0.18), 0.0);
    let radius = rng.range(0.07, 0.11);
    let turn = rng.range(0.0, 2.0 * PI);
    let (corners, star) = match (rng.uniform() * 5.0) as usize {
        kind @ 0..=3 => (kind + 3, false),
        _ => (5, true),
    };
    let mut points = vec![];
    for i in 0..corners {
        let a = turn + 2.0 * PI * i as f32 / corners as f32;
        points.push(center + radius * Vec3::new(a.cos(), a.sin(), 0.0));
        if star {
            let b = a + PI / corners as f32;
            points.push(center + 0.42 * radius * Vec3::new(b.cos(), b.sin(), 0.0));
        }
    }
    points
}

fn main() -> Result<()> {
    let (mut out, mut arms, mut touch) = (None, 20, true);
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--no-touch" => touch = false,
            _ if out.is_none() => out = Some(PathBuf::from(arg)),
            _ => arms = arg.parse()?,
        }
    }
    let out = out.context("usage: drawing <out_dir> [arms=20] [--no-touch]")?;

    let urdf = std::fs::read_to_string(format!("{ASSETS}/ur5e/ur_description/urdf/ur5e.urdf"))?
        .replace("</robot>", &format!("{PEN}</robot>"));
    let options = RobotOptions {
        ee_links: vec!["pen_tip".into()],
        package_dirs: vec![format!("{ASSETS}/ur5e").into()],
        ..Default::default()
    };
    let robot = Robot::from_urdf(&urdf, &options)?;
    // The UR's home, the pen pointing down over the table.
    let home = vec![0.0, -PI / 2.0, PI / 2.0, -PI / 2.0, -PI / 2.0, 0.0];
    let ready = robot.ee_pose(&home, 0);
    ensure!(
        ready.position.x > 0.0 && (ready.rotation * Vec3::Z).z < -0.99,
        "the pen points down over the table at home, not {ready:?}"
    );

    let table = Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let world = World {
        touch: if touch {
            vec![Touch::new(0, robot.hand_links(0))]
        } else {
            vec![]
        },
        ..World::new(vec![table])
    };
    let device = Device::cpu(&robot)?;
    let worlds = device.upload(&vec![world; arms])?;
    let mut rng = Rng::new(11);
    let shapes: Vec<Vec<Vec3>> = (0..arms).map(|_| shape(&mut rng)).collect();
    let at = |p: Vec3| Pose::new(p, ready.rotation);
    let paths: Vec<ToolPath> = (0..arms)
        .map(|k| {
            let corners = &shapes[k];
            let above = at(corners[0] + Vec3::Z * ABOVE);
            let around = corners.iter().chain(&corners[..1]).map(|&c| at(c));
            ToolPath {
                world: k as u32,
                start: home.clone(),
                hand: 0,
                waypoints: [above]
                    .into_iter()
                    .chain(around)
                    .chain([above, ready])
                    .collect(),
            }
        })
        .collect();
    let started = Instant::now();
    let result = batchplan::follow(&device, &worlds, &paths, &ToolPathOptions::default())?;
    println!(
        "{arms} drawings followed in {:.1} ms on {}",
        started.elapsed().as_secs_f64() * 1e3,
        device.name()
    );

    let speed = HandSpeed {
        hand: 0,
        linear: 0.15,
        angular: 1.0,
    };
    let mut drawings = vec![];
    for (k, path) in result.paths.iter().enumerate() {
        if let Some(why) = result.stopped[k] {
            println!(
                "arm {k}: stopped {:.0}% of the way, {why:?}",
                100.0 * result.fraction[k]
            );
            continue;
        }
        let samples = Trajectory::with_hand_speed(&robot, path, &speed)?.sample(HZ)?;
        let positions: Vec<&[f32]> = samples.positions.chunks(robot.dof()).collect();
        let tip: Vec<[f32; 3]> = positions
            .iter()
            .map(|q| robot.ee_pose(q, 0).position.to_array())
            .collect();
        drawings.push(serde_json::json!({
            "corners": shapes[k].iter().map(|c| c.to_array()).collect::<Vec<_>>(),
            "positions": positions,
            "tip": tip,
        }));
    }
    println!("{} of {arms} drawn", drawings.len());
    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join("ur5e_pen.urdf"), &urdf)?;
    let json = serde_json::json!({ "dt": 1.0 / HZ, "drawings": drawings });
    std::fs::write(out.join("drawing.json"), serde_json::to_string(&json)?)?;
    println!("wrote {}", out.display());
    Ok(())
}
