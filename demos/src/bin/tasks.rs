//! Whole tasks in Rerun: in each scene the Panda picks a box out of clutter (`batchplan::pick`,
//! by any grasp `batchplan::grasps` proposes) and sets it down at a free spot
//! (`batchplan::place`, at any of the box's symmetric turns there). Two calls plan every scene:
//! each IK, straight move and planned path is one batch over every scene's candidates. Two scenes
//! cannot be done, a lid over the box and a crate over the spot, and say why.
//!
//!   cargo run --release --bin tasks                      # 16 scenes, in the viewer
//!   cargo run --release --bin tasks -- 36 --gpu          # 36 scenes, planned on the GPU
//!   cargo run --release --bin tasks -- 16 --save tasks.rrd  # writes a recording instead

use std::f32::consts::PI;
use std::time::Instant;

use anyhow::Result;
use batchplan::glam::{Quat, Vec3};
use batchplan::rng::Rng;
use batchplan::{
    GraspOptions, Obstacle, Pick, Place, Pose, Robot, TaskOptions, TaskPlan, Trajectory, World,
};
use batchplan_demos::{Args, OPEN, RobotMeshes};

/// Distance between neighbouring scenes in the grid, metres.
const SPACING: f32 = 1.8;
/// Samples per second of the motions, and of the playback.
const HZ: f32 = 30.0;
/// How long closing or opening the gripper takes, in samples.
const GRIP: usize = 15;
/// The box to move, among each scene's obstacles.
const BOX: usize = 1;

/// A table, a box to move (obstacle 1), 2-4 more boxes of clutter, and a free spot to set the box
/// down at. Scene 3 has a lid just over the box, which it cannot be lifted past; scene 7 a crate
/// over its spot.
fn scene(i: usize, rng: &mut Rng) -> (World, Pose) {
    let table = Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let half = Vec3::new(
        rng.range(0.015, 0.03),
        rng.range(0.02, 0.035),
        rng.range(0.025, 0.045),
    );
    let at = |rng: &mut Rng, side: f32| {
        Pose::new(
            Vec3::new(rng.range(0.35, 0.6), side * rng.range(0.15, 0.4), half.z),
            Quat::from_rotation_z(rng.range(-PI, PI)),
        )
    };
    let side = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
    let (from, to) = (at(rng, side), at(rng, -side));
    let mut obstacles = vec![table, Obstacle::cuboid(Vec3::ZERO, half).placed(from)];
    while obstacles.len() < 4 + (rng.uniform() * 3.0) as usize {
        let size = Vec3::new(
            rng.range(0.02, 0.05),
            rng.range(0.02, 0.05),
            rng.range(0.03, 0.15),
        );
        let side = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        let pose = at(rng, side);
        let clutter = Obstacle::cuboid(Vec3::ZERO, Vec3::new(size.x, size.y, size.z))
            .placed(Pose::new(pose.position.with_z(size.z), pose.rotation));
        let clear = |p: Vec3| clutter.distance(p).value > 0.1;
        if clear(from.position) && clear(to.position) {
            obstacles.push(clutter);
        }
    }
    match i {
        3 => obstacles.push(Obstacle::cuboid(
            from.position.with_z(2.0 * half.z + 0.008),
            Vec3::new(0.008, 0.008, 0.004),
        )),
        7 => obstacles.push(Obstacle::cuboid(
            to.position.with_z(0.06),
            Vec3::new(0.08, 0.08, 0.06),
        )),
        _ => {}
    }
    (World::new(obstacles), to)
}

/// One scene's motion at `HZ`: each configuration, the gripper's opening and where the box is.
struct Episode {
    q: Vec<Vec<f32>>,
    opening: Vec<f32>,
    object: Vec<Pose>,
}

impl Episode {
    fn new() -> Self {
        Self {
            q: vec![],
            opening: vec![],
            object: vec![],
        }
    }

    /// `t` sampled, the gripper `opening` throughout and the box at `object(q)`.
    fn run(
        &mut self,
        robot: &Robot,
        t: &Trajectory,
        opening: f32,
        object: &dyn Fn(&[f32]) -> Pose,
    ) -> Result<()> {
        let samples = t.sample(HZ)?;
        for q in samples.positions.chunks(robot.dof()) {
            self.q.push(q.to_vec());
            self.opening.push(opening);
            self.object.push(object(q));
        }
        Ok(())
    }

    /// The gripper moving from `from` to `to` where the last motion ended, the box at `object`.
    fn grip(&mut self, from: f32, to: f32, object: Pose) {
        let q = self.q.last().cloned().unwrap_or_default();
        for k in 1..=GRIP {
            self.q.push(q.clone());
            self.opening
                .push(from + (to - from) * k as f32 / GRIP as f32);
            self.object.push(object);
        }
    }
}

/// Pick then place, as one episode: the box where it stood, carried in the hand, then where it
/// was set down.
fn episode(robot: &Robot, start: Pose, picked: &TaskPlan, placed: &TaskPlan) -> Result<Episode> {
    let held = picked.held.as_ref().expect("a pick leaves its object held");
    let carried = |q: &[f32]| robot.ee_pose(q, held.hand).mul_pose(held.grasp.inverse());
    let set_down = placed
        .placed
        .as_ref()
        .expect("a place leaves its object placed")
        .pose();
    let closed = match held.object {
        Obstacle::Cuboid { half_extents, .. } => half_extents.y,
        _ => OPEN,
    };
    let mut e = Episode::new();
    let still = |pose: Pose| move |_: &[f32]| pose;
    e.run(robot, &picked.transit, OPEN, &still(start))?;
    e.run(robot, &picked.approach, OPEN, &still(start))?;
    e.grip(OPEN, closed, start);
    e.run(robot, &picked.depart, closed, &carried)?;
    e.run(robot, &placed.transit, closed, &carried)?;
    e.run(robot, &placed.approach, closed, &carried)?;
    e.grip(closed, OPEN, set_down);
    e.run(robot, &placed.depart, OPEN, &still(set_down))?;
    Ok(e)
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let count = args.count.unwrap_or(16);
    let robot = batchplan_demos::panda()?;
    let device = args.device(&robot, false)?;
    let mut rng = Rng::new(11);
    let (worlds, spots): (Vec<World>, Vec<Pose>) = (0..count).map(|i| scene(i, &mut rng)).unzip();
    let o = TaskOptions::default();

    let t = Instant::now();
    let uploaded = device.upload(&worlds)?;
    let picks = worlds
        .iter()
        .enumerate()
        .map(|(w, world)| {
            let object = world.obstacles[BOX].placed(Pose::IDENTITY);
            Ok(Pick {
                support: vec![0],
                ..Pick::new(
                    w as u32,
                    robot.default_q(),
                    BOX,
                    batchplan::grasps(&object, &GraspOptions::default())?,
                )
            })
        })
        .collect::<Result<Vec<Pick>>>()?;
    let picked = batchplan::pick(&device, &uploaded, &picks, &o)?;
    let picking = t.elapsed();
    // Each picked box set down at its spot, in its world without it.
    let t = Instant::now();
    let (mut after, mut places, mut placing_of) = (vec![], vec![], vec![None; count]);
    for (w, outcome) in picked.outcomes.iter().enumerate() {
        let Ok(plans) = outcome else { continue };
        let mut world = worlds[w].clone();
        let object = world.remove(BOX)?;
        let held = plans[0]
            .held
            .clone()
            .expect("a pick leaves its object held");
        placing_of[w] = Some(places.len());
        places.push(Place {
            support: vec![0],
            ..Place::new(
                after.len() as u32,
                plans[0].end(),
                held,
                batchplan::place_poses(&object, spots[w], 8),
            )
        });
        after.push(world);
    }
    let placed = batchplan::place(&device, &device.upload(&after)?, &places, &o)?;
    let placing = t.elapsed();

    // Each scene's episode, or why it has none.
    let mut episodes: Vec<Option<Episode>> = vec![];
    let mut notes = vec![];
    for w in 0..count {
        let start = worlds[w].obstacles[BOX].pose();
        let note = match (
            &picked.outcomes[w],
            placing_of[w].map(|p| &placed.outcomes[p]),
        ) {
            (Err(why), _) => {
                episodes.push(None);
                format!("no pick: {why}")
            }
            (Ok(_), Some(Err(why))) => {
                episodes.push(None);
                format!("no place: {why}")
            }
            (Ok(picks), Some(Ok(places))) => {
                let e = episode(&robot, start, &picks[0], &places[0])?;
                episodes.push(Some(e));
                format!(
                    "{:.1} s up by grasp {} ({} of {} got through), {:.1} s down at turn {} ({} of {})",
                    picks[0].duration(),
                    picks[0].candidate,
                    picks.len(),
                    picked.problems[w].grasps.len(),
                    places[0].duration(),
                    places[0].candidate,
                    places.len(),
                    placed.problems[placing_of[w].expect("placed")].poses.len()
                )
            }
            (Ok(_), None) => unreachable!("every pick that got through was placed"),
        };
        notes.push(note);
    }
    let done = episodes.iter().flatten().count();
    let summary = format!(
        "## {done} of {count} boxes moved\n\n\
         Picked in {:.0} ms and set down in {:.0} ms on {}: two calls, each running every scene's \
         candidates through IK, the straight moves and the planned paths in batches.\n\n{}",
        picking.as_secs_f64() * 1e3,
        placing.as_secs_f64() * 1e3,
        device.name(),
        notes
            .iter()
            .enumerate()
            .map(|(w, n)| format!("- scene {w}: {n}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    println!("{}", summary.replace("## ", "").replace("\n\n", "\n"));

    let rec = args.recording("batchplan_tasks")?;
    rec.log_static("summary", &rerun::TextDocument::from_markdown(summary))?;
    let offsets = batchplan_demos::grid(count, SPACING);
    for w in 0..count {
        let scene = format!("world/scene_{w}");
        rec.log_static(
            scene.as_str(),
            &rerun::Transform3D::from_translation(offsets[w].to_array()),
        )?;
        let fixed: Vec<(usize, &Obstacle)> = worlds[w]
            .obstacles
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != BOX)
            .collect();
        rec.log_static(
            format!("{scene}/obstacles"),
            &batchplan_demos::boxes(&fixed, |k| {
                if k == 0 {
                    batchplan_demos::wood()
                } else {
                    batchplan_demos::slate()
                }
            }),
        )?;
        let Obstacle::Cuboid { half_extents, .. } = worlds[w].obstacles[BOX] else {
            unreachable!("every scene's box is a cuboid")
        };
        let color = if episodes[w].is_some() {
            rerun::Color::from_rgb(40, 200, 90)
        } else {
            rerun::Color::from_rgb(230, 60, 60)
        };
        rec.log_static(
            format!("{scene}/object"),
            &rerun::Boxes3D::from_half_sizes([half_extents.to_array()])
                .with_colors([rerun::Color::from_rgb(235, 120, 40)])
                .with_fill_mode(rerun::FillMode::Solid),
        )?;
        rec.log_static(
            format!("{scene}/spot"),
            &rerun::Boxes3D::from_centers_and_half_sizes(
                [spots[w].position.to_array()],
                [half_extents.to_array()],
            )
            .with_quaternions([batchplan_demos::quaternion(spots[w].rotation)])
            .with_colors([color]),
        )?;
        rec.log_static(
            format!("{scene}/note"),
            &rerun::Points3D::new([[0.4, 0.0, 0.7]])
                .with_labels([notes[w].as_str()])
                .with_colors([color]),
        )?;
        match &episodes[w] {
            Some(e) => {
                let hand: Vec<[f32; 3]> =
                    e.q.iter()
                        .map(|q| robot.ee_pose(q, 0).position.to_array())
                        .collect();
                rec.log_static(
                    format!("{scene}/hand_path"),
                    &rerun::LineStrips3D::new([hand])
                        .with_radii([0.002])
                        .with_colors([color]),
                )?;
            }
            None => rec.log_static(
                format!("{scene}/object"),
                &batchplan_demos::transform(&worlds[w].obstacles[BOX].pose()),
            )?,
        }
    }
    let meshes: RobotMeshes = batchplan_demos::panda_meshes(&rec, "world/panda", &robot)?;
    // Every robot moves at once, on one timeline; each episode holds its last sample.
    let frames = episodes
        .iter()
        .flatten()
        .map(|e| e.q.len())
        .max()
        .unwrap_or(1);
    for f in 0..frames {
        rec.set_duration_secs("time", f as f32 / HZ);
        let mut robots = vec![];
        for (w, e) in episodes.iter().enumerate() {
            let (q, opening) = match e {
                Some(e) => {
                    let s = f.min(e.q.len() - 1);
                    rec.log(
                        format!("world/scene_{w}/object"),
                        &batchplan_demos::transform(&e.object[s]),
                    )?;
                    (e.q[s].as_slice(), e.opening[s])
                }
                None => (robot.default_q(), OPEN),
            };
            let fingers = batchplan_demos::panda_fingers(&robot, q, opening)?;
            robots.push((offsets[w], meshes.link_poses(&robot, q, fingers)?));
        }
        meshes.log_robots(&rec, &robots)?;
    }
    if args.save.is_none() {
        println!("Opened in the Rerun viewer; press play on the time panel.");
    }
    Ok(())
}
