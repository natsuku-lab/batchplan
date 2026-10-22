//! Two reBot B601 arms handing blocks to each other, from batchplan, played back in Rerun.
//!
//! The scenes are `examples/handover`'s: a table with a block lying in front of one arm and a few
//! small boxes. One `datagen::handover` call plans every episode: the giver picks the block up,
//! both arms move at once to where they meet, the taker grasps its end, the giver lets go and
//! returns home while the taker carries the block to the place pose and sets it down. Every segment
//! is planned for every scene in one batch. The viewer shows every scene at once and the first
//! episode's joints and grippers over time.
//!
//!   ../scripts/fetch_rebot.sh ../data/rebot
//!   cargo run --release --bin handover                            # 16 scenes, in the viewer
//!   cargo run --release --bin handover -- 36 --gpu               # 36 scenes, planned on the GPU
//!   cargo run --release --bin handover -- 16 --save handover.rrd  # writes a recording instead

#[path = "../../../examples/handover/rebot.rs"]
mod rebot;

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use batchplan::datagen::{Demonstration, Handover, handover};
use batchplan::glam::Vec3;
use batchplan::rng::Rng;
use batchplan::*;
use batchplan_demos::*;
use rebot::{FINGERS_OPEN, scene, two_rebots};

/// Where `scripts/fetch_rebot.sh` puts the arm.
const REBOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/rebot");
/// The arms' prefixes, hand 0 then hand 1.
const PREFIXES: [&str; 2] = ["left_", "right_"];
/// Distance between neighbouring scenes in the grid, metres.
const SPACING: f32 = 1.5;
/// Samples per second of the episodes, and of the playback.
const HZ: f32 = 30.0;

fn main() -> Result<()> {
    let args = Args::parse()?;
    let count = args.count.unwrap_or(16);
    let robot = two_rebots(Path::new(REBOT))?;
    // Small batches plan sooner on the CPU, which pays no per-call GPU cost.
    let device = args.device(&robot, false)?;
    let mut rng = Rng::new(5);
    let (worlds, tasks): (Vec<World>, Vec<Handover>) =
        (0..count).map(|i| scene(i as u32, &mut rng)).unzip();
    let options = rebot::options(HZ);
    // The first call on a GPU compiles its kernels; time the second.
    handover(&device, &worlds[..1], &tasks[..1], &options)?;
    let t = Instant::now();
    let demos = handover(&device, &worlds, &tasks, &options)?;
    let planned = t.elapsed();

    let featured = demos.first().context("no handover succeeded")?;
    let summary = format!(
        "## {} of {count} handovers between two reBot B601 arms\n\n\
         Planned in {:.0} ms on {}: IK for each grasp, both hands at once where they meet, and \
         every pose between; then seven motions per episode, each planned for every scene in one \
         batch: the giver picks the block up while the taker waits, both arms move at once to the \
         meeting pose, the taker grasps its end, the giver lets go and goes home while the taker \
         carries the block to its place, and sets it down.\n\n\
         Scene {}: *{}*",
        demos.len(),
        planned.as_secs_f64() * 1e3,
        device.name(),
        featured.world,
        featured.task,
    );
    println!(
        "{}",
        summary
            .replace("## ", "")
            .replace('*', "")
            .replace("\n\n", "\n")
    );

    let rec = args.recording("batchplan_handover")?;
    rec.log_static("summary", &rerun::TextDocument::from_markdown(summary))?;
    let mut episodes: Vec<Option<&Demonstration>> = vec![None; count];
    for d in &demos {
        episodes[d.world as usize] = Some(d);
    }
    let offsets = grid(count, SPACING);
    for (i, task) in tasks.iter().enumerate() {
        log_scene(&rec, &robot, &worlds[i], task, episodes[i], offsets[i], i)?;
    }
    let meshes = RobotMeshes::log(&rec, "world/rebot", &robot, Path::new(REBOT))?;

    // Every robot moves at once, on one timeline; each episode holds its last sample.
    let n = robot.dof();
    let frames = demos.iter().map(|d| d.trajectory.len()).max().unwrap_or(1);
    for f in 0..frames {
        rec.set_duration_secs("time", f as f32 / HZ);
        let mut robots = vec![];
        for (i, episode) in episodes.iter().enumerate() {
            let task = &tasks[i];
            let (q, openings) = match episode {
                Some(d) => {
                    let s = f.min(d.trajectory.len() - 1);
                    if let Some(c) = &d.carried {
                        rec.log(format!("world/scene_{i}/block"), &transform(&c.poses[s]))?;
                    }
                    (
                        &d.trajectory.positions[s * n..][..n],
                        [d.gripper[2 * s], d.gripper[2 * s + 1]],
                    )
                }
                None => (robot.default_q(), [1.0, 1.0]),
            };
            let Obstacle::Cuboid { half_extents, .. } = worlds[i].obstacles[task.object] else {
                unreachable!()
            };
            let fingers = rebot_fingers(&robot, q, openings, half_extents.x.min(half_extents.y));
            robots.push((offsets[i], meshes.link_poses(&robot, q, fingers)?));
        }
        meshes.log_robots(&rec, &robots)?;
        if f < featured.trajectory.len() {
            let q = &featured.trajectory.positions[f * n..][..n];
            for (j, name) in robot.joint_names().iter().enumerate() {
                let (arm, joint) = name.split_once('_').context("a prefixed joint")?;
                rec.log(
                    format!("{arm}/{joint}"),
                    &rerun::Scalars::single(q[j] as f64),
                )?;
            }
            for (h, prefix) in PREFIXES.iter().enumerate() {
                let arm = prefix.trim_end_matches('_');
                rec.log(
                    format!("{arm}/gripper"),
                    &rerun::Scalars::single(featured.gripper[2 * f + h] as f64),
                )?;
            }
        }
    }
    if args.save.is_none() {
        println!("Opened in the Rerun viewer; press play on the time panel.");
    }
    Ok(())
}

/// What does not move in scene `i`: the table and boxes, the block where it starts if nothing
/// carries it, the meeting and place poses, and each hand's path.
fn log_scene(
    rec: &rerun::RecordingStream,
    robot: &Robot,
    world: &World,
    task: &Handover,
    episode: Option<&Demonstration>,
    offset: Vec3,
    i: usize,
) -> Result<()> {
    let scene = format!("world/scene_{i}");
    rec.log_static(
        scene.as_str(),
        &rerun::Transform3D::from_translation(offset.to_array()),
    )?;
    let fixed: Vec<(usize, &Obstacle)> = world
        .obstacles
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != task.object)
        .collect();
    rec.log_static(
        format!("{scene}/obstacles"),
        &boxes(&fixed, |k| if k == 0 { wood() } else { slate() }),
    )?;
    let Obstacle::Cuboid {
        center,
        half_extents,
        rotation,
    } = world.obstacles[task.object]
    else {
        anyhow::bail!("the block is not a box");
    };
    rec.log_static(
        format!("{scene}/block"),
        &rerun::Boxes3D::from_half_sizes([half_extents.to_array()])
            .with_colors([rerun::Color::from_rgb(220, 60, 50)])
            .with_fill_mode(rerun::FillMode::Solid),
    )?;
    // A carried block's pose is logged at every sample; static data would hide it.
    if !episode.is_some_and(|d| d.carried.is_some()) {
        rec.log_static(
            format!("{scene}/block"),
            &transform(&Pose {
                position: center,
                rotation,
            }),
        )?;
    }
    let color = if episode.is_some() {
        rerun::Color::from_rgb(40, 200, 90)
    } else {
        rerun::Color::from_rgb(230, 60, 60)
    };
    for (name, pose) in [("place", task.place), ("meet", task.meet)] {
        rec.log_static(
            format!("{scene}/{name}"),
            &rerun::Boxes3D::from_centers_and_half_sizes(
                [pose.position.to_array()],
                [half_extents.to_array()],
            )
            .with_quaternions([quaternion(pose.rotation)])
            .with_colors([if name == "place" {
                color
            } else {
                rerun::Color::from_rgb(160, 160, 170)
            }]),
        )?;
    }
    if let Some(d) = episode {
        let hands = [
            rerun::Color::from_rgb(70, 130, 230),
            rerun::Color::from_rgb(240, 150, 40),
        ];
        for (h, hand_color) in hands.into_iter().enumerate() {
            let path: Vec<[f32; 3]> = d
                .trajectory
                .positions
                .chunks(robot.dof())
                .map(|q| robot.ee_pose(q, h).position.to_array())
                .collect();
            rec.log_static(
                format!("{scene}/{}hand_path", PREFIXES[h]),
                &rerun::LineStrips3D::new([path])
                    .with_radii([0.0015])
                    .with_colors([hand_color]),
            )?;
        }
    }
    Ok(())
}

/// Moves each arm's fingers to its gripper's opening (1 open, 0 closed) at `q`; the planning model
/// holds them open. Closed fingers stop at the block, `stop` from the middle.
fn rebot_fingers(
    robot: &Robot,
    q: &[f32],
    openings: [f32; 2],
    stop: f32,
) -> impl Fn(&str, Pose) -> Pose {
    // The fingers slide along the grasp point's y axis, the right one mirroring the left.
    let shift: Vec<Vec3> = (0..2)
        .map(|h| {
            let closed = stop.min(FINGERS_OPEN);
            let finger = closed + openings[h] * (FINGERS_OPEN - closed);
            robot.ee_pose(q, h).rotation * Vec3::Y * (finger - FINGERS_OPEN)
        })
        .collect();
    move |name: &str, mut pose: Pose| {
        for (h, prefix) in PREFIXES.iter().enumerate() {
            match name.strip_prefix(prefix) {
                Some("finger_left_link") => pose.position += shift[h],
                Some("finger_right_link") => pose.position -= shift[h],
                _ => {}
            }
        }
        pose
    }
}
