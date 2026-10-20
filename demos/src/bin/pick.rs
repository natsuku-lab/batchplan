//! Pick-and-place episodes from batchplan, played back in Rerun.
//!
//! Each scene is a table with two posts and a box to move to the other side. One
//! `datagen::pick_and_place` call plans every episode: IK for the grasp and the place pose, then
//! approach, descend, close, lift, carry, lower, open and retreat, each segment planned for every
//! scene in one batch and timed within the Panda's limits. The viewer shows every scene at once,
//! the box riding in the gripper, and the first episode's joints over time.
//!
//!   cargo run --release --bin pick                      # 16 scenes, in the viewer
//!   cargo run --release --bin pick -- 36 --gpu         # 36 scenes, planned on the GPU
//!   cargo run --release --bin pick -- 16 --save pick.rrd  # writes a recording instead

use std::f32::consts::PI;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use batchplan::datagen::{PickPlace, PickPlaceOptions, pick_and_place};
use batchplan::glam::{Quat, Vec3};
use batchplan::rng::Rng;
use batchplan::*;
use batchplan_demos::*;

/// Distance between neighbouring scenes in the grid, metres.
const SPACING: f32 = 1.8;
/// Samples per second of the episodes, and of the playback.
const HZ: f32 = 30.0;

fn main() -> Result<()> {
    let args = Args::parse()?;
    let count = args.count.unwrap_or(16);
    let robot = panda()?;
    // Small batches plan sooner on the CPU, which pays no per-call GPU cost.
    let device = args.device(&robot, false)?;
    let mut rng = Rng::new(3);
    let (worlds, tasks): (Vec<World>, Vec<PickPlace>) =
        (0..count).map(|i| pick_place_scene(i as u32, &mut rng)).unzip();
    let mut options = PickPlaceOptions { dt: 1.0 / HZ, ..Default::default() };
    // Trajectory optimization smooths the carries the search had to bend around obstacles; without
    // it, about one carry in six stops at a corner of its searched path.
    options.plan.seeds = 2;
    // The first call on a GPU compiles its kernels; time the second.
    pick_and_place(&device, &worlds[..1], &tasks[..1], &options)?;
    let t = Instant::now();
    let demos = pick_and_place(&device, &worlds, &tasks, &options)?;
    let planned = t.elapsed();

    let featured = demos.first().context("no task succeeded")?;
    let blocked = worlds.iter().zip(&tasks).filter(|(w, t)| straight_carry_blocked(w, t)).count();
    let summary = format!(
        "## {} of {count} pick-and-place episodes\n\n\
         In {blocked} of the {count} scenes, a wall, posts, an arch or a cluster of boxes stands across the straight \
         carry, so the robot must plan around it.\n\n\
         Planned in {:.0} ms on {}: IK for every grasp and place pose, then 4 planned motions per episode \
         (the lift and the retreat retrace the descent and the lowering), each planned for every scene in one batch, \
         smoothed by trajectory optimization and timed within the Panda's limits.\n\n\
         Scene {}: *{}*",
        demos.len(),
        planned.as_secs_f64() * 1e3,
        device.name(),
        featured.world,
        featured.task,
    );
    println!("{}", summary.replace("## ", "").replace("*", "").replace("\n\n", "\n"));

    let rec = args.recording("batchplan_pick_and_place")?;
    rec.log_static("summary", &rerun::TextDocument::from_markdown(summary))?;
    let scenes = PickScenes::log(&rec, &robot, &worlds, &tasks, &demos, SPACING)?;

    let samples = featured.trajectory.len();

    // Every robot moves at once, on one timeline.
    let n = robot.dof();
    for f in 0..scenes.frames() {
        rec.set_duration_secs("time", f as f32 / HZ);
        scenes.log_frame(&rec, f, Some(featured.world as usize))?;
        if f < samples {
            let q = &featured.trajectory.positions[f * n..][..n];
            for (j, name) in robot.joint_names().iter().enumerate() {
                rec.log(format!("joints/{name}"), &rerun::Scalars::single(q[j] as f64))?;
            }
            rec.log("joints/gripper", &rerun::Scalars::single(featured.gripper[f] as f64))?;
        }
    }
    if args.save.is_none() {
        println!("Opened in the Rerun viewer; press play on the time panel.");
    }
    Ok(())
}

/// What stands between a pick-and-place task's two spots, so the straight carry is blocked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Between {
    /// A low wall to lift the box over.
    Wall,
    /// A row of tall posts to weave through or around.
    Posts,
    /// Two pillars and a beam across them: under or around.
    Arch,
    /// A cluster of boxes of different heights.
    Cluster,
}

/// A table with a small box to move from one side of the robot to the other, something standing
/// between the two spots (see [`Between`]), and 2-4 more boxes of clutter, none touching either
/// spot. The box is of a random size.
fn pick_place_scene(world: u32, rng: &mut Rng) -> (World, PickPlace) {
    let table = Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let half = Vec3::new(rng.range(0.015, 0.03), rng.range(0.025, 0.04), rng.range(0.02, 0.04));
    let side = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
    let spot = |rng: &mut Rng, side: f32| Vec3::new(rng.range(0.35, 0.6), side * rng.range(0.2, 0.38), half.z);
    let (pick, place) = (spot(rng, side), spot(rng, -side));
    let upright = |center: Vec3, half_extents: Vec3, yaw: f32| Obstacle::Cuboid {
        center,
        half_extents,
        rotation: Quat::from_rotation_z(yaw),
    };
    // Between the spots: where the straight carry crosses y = 0.
    let x = pick.x + (place.x - pick.x) * pick.y / (pick.y - place.y);
    let mut between =
        match [Between::Wall, Between::Posts, Between::Arch, Between::Cluster][(rng.uniform() * 4.0) as usize % 4] {
            Between::Wall => {
                let height = rng.range(0.08, 0.2);
                vec![upright(
                    Vec3::new(x, 0.0, height),
                    Vec3::new(rng.range(0.12, 0.2), 0.015, height),
                    rng.range(-0.3, 0.3),
                )]
            }
            Between::Posts => (0..3)
                .map(|k| {
                    let height = rng.range(0.2, 0.4);
                    upright(
                        Vec3::new(x + (k as f32 - 1.0) * 0.1, rng.range(-0.03, 0.03), height),
                        Vec3::new(0.025, 0.025, height),
                        0.0,
                    )
                })
                .collect(),
            Between::Arch => {
                let (height, span) = (rng.range(0.25, 0.35), rng.range(0.1, 0.16));
                vec![
                    upright(Vec3::new(x - span, 0.0, height), Vec3::new(0.02, 0.02, height), 0.0),
                    upright(Vec3::new(x + span, 0.0, height), Vec3::new(0.02, 0.02, height), 0.0),
                    upright(Vec3::new(x, 0.0, 2.0 * height + 0.02), Vec3::new(span + 0.03, 0.03, 0.02), 0.0),
                ]
            }
            Between::Cluster => (0..4)
                .map(|_| {
                    let h = Vec3::new(rng.range(0.03, 0.06), rng.range(0.03, 0.06), rng.range(0.05, 0.18));
                    upright(Vec3::new(x + rng.range(-0.1, 0.1), rng.range(-0.06, 0.06), h.z), h, rng.range(0.0, PI))
                })
                .collect(),
        };
    // Clutter elsewhere, kept 8 cm from either spot and from the robot's base.
    let mut clutter = vec![];
    while clutter.len() < 2 + (rng.uniform() * 3.0) as usize {
        let h = Vec3::new(rng.range(0.02, 0.05), rng.range(0.02, 0.05), rng.range(0.03, 0.15));
        let c = Vec3::new(rng.range(0.3, 0.8), rng.range(-0.6, 0.6), h.z);
        let candidate = upright(c, h, rng.range(0.0, PI));
        let clear = |p: Vec3| candidate.distance(p).0 > 0.08 + half.max_element();
        if clear(pick) && clear(place) && candidate.distance(Vec3::new(0.0, 0.0, 0.3)).0 > 0.25 {
            clutter.push(candidate);
        }
    }
    let mut obstacles = vec![table];
    obstacles.append(&mut between);
    obstacles.append(&mut clutter);
    obstacles.push(Obstacle::Cuboid {
        center: pick,
        half_extents: half,
        rotation: Quat::from_rotation_z(rng.range(-1.2, 1.2)),
    });
    let place = Pose::new(place, Quat::from_rotation_z(rng.range(-1.2, 1.2)));
    let object = obstacles.len() - 1;
    (World { obstacles }, PickPlace { world, object, place })
}

/// Whether the straight carry, 10 cm above the table from the box to where it goes, passes through
/// an obstacle of `world` other than the box: whether the task needs planning around something.
fn straight_carry_blocked(world: &World, task: &PickPlace) -> bool {
    let Obstacle::Cuboid { center, .. } = world.obstacles[task.object] else { return false };
    let lift = Vec3::new(0.0, 0.0, 0.1);
    let (from, to) = (center + lift, task.place.position + lift);
    (0..=50).any(|k| {
        let p = from.lerp(to, k as f32 / 50.0);
        world.obstacles.iter().enumerate().any(|(i, o)| i != task.object && i != 0 && o.distance(p).0 < 0.0)
    })
}

/// Pick-and-place episodes laid out in a grid of scenes, one robot each.
struct PickScenes<'a> {
    robot: &'a Robot,
    worlds: &'a [World],
    tasks: &'a [PickPlace],
    /// Each scene's episode, if its task succeeded.
    episodes: Vec<Option<&'a datagen::Demonstration>>,
    offsets: Vec<Vec3>,
    meshes: RobotMeshes,
}

impl<'a> PickScenes<'a> {
    /// Logs what does not move: each scene's table, posts and place target, the box's start
    /// where nothing carries it, the hand's path, and the robot's meshes.
    fn log(
        rec: &rerun::RecordingStream,
        robot: &'a Robot,
        worlds: &'a [World],
        tasks: &'a [PickPlace],
        demos: &'a [datagen::Demonstration],
        spacing: f32,
    ) -> Result<Self> {
        let mut episodes = vec![None; tasks.len()];
        for d in demos {
            episodes[d.world as usize] = Some(d);
        }
        let offsets = grid(tasks.len(), spacing);
        for (i, task) in tasks.iter().enumerate() {
            let world = &worlds[task.world as usize];
            let scene = format!("world/scene_{i}");
            rec.log_static(scene.as_str(), &rerun::Transform3D::from_translation(offsets[i].to_array()))?;
            let fixed: Vec<(usize, &Obstacle)> =
                world.obstacles.iter().enumerate().filter(|(k, _)| *k != task.object).collect();
            rec.log_static(format!("{scene}/obstacles"), &boxes(&fixed, |k| if k == 0 { wood() } else { slate() }))?;
            let Obstacle::Cuboid { center, half_extents, rotation } = world.obstacles[task.object] else {
                bail!("the object to move is not a box");
            };
            rec.log_static(
                format!("{scene}/object"),
                &rerun::Boxes3D::from_half_sizes([half_extents.to_array()])
                    .with_colors([rerun::Color::from_rgb(235, 120, 40)])
                    .with_fill_mode(rerun::FillMode::Solid),
            )?;
            // A carried box's pose is logged at every sample; static data would hide it.
            let episode: Option<&datagen::Demonstration> = episodes[i];
            if !episode.is_some_and(|d| d.carried.is_some()) {
                rec.log_static(format!("{scene}/object"), &transform(&Pose { position: center, rotation }))?;
            }
            let color = if episode.is_some() {
                rerun::Color::from_rgb(40, 200, 90)
            } else {
                rerun::Color::from_rgb(230, 60, 60)
            };
            rec.log_static(
                format!("{scene}/place"),
                &rerun::Boxes3D::from_centers_and_half_sizes(
                    [task.place.position.to_array()],
                    [half_extents.to_array()],
                )
                .with_quaternions([quaternion(task.place.rotation)])
                .with_colors([color]),
            )?;
            if let Some(d) = episode {
                let hand: Vec<[f32; 3]> = d
                    .trajectory
                    .positions
                    .chunks(robot.dof())
                    .map(|q| robot.ee_pose(q, 0).position.to_array())
                    .collect();
                rec.log_static(
                    format!("{scene}/hand_path"),
                    &rerun::LineStrips3D::new([hand]).with_radii([0.002]).with_colors([color]),
                )?;
            }
        }
        let meshes = panda_meshes(rec, "world/panda", robot)?;
        Ok(Self { robot, worlds, tasks, episodes, offsets, meshes })
    }

    /// Samples in the longest episode.
    fn frames(&self) -> usize {
        self.episodes.iter().flatten().map(|d| d.trajectory.len()).max().unwrap_or(1)
    }

    /// Every robot and carried box at sample `f` (each episode holds its last sample). The hand
    /// of scene `hand_of` is also logged as an entity, for a camera mounted on it.
    fn log_frame(&self, rec: &rerun::RecordingStream, f: usize, hand_of: Option<usize>) -> Result<()> {
        let n = self.robot.dof();
        let mut robots = vec![];
        for (i, episode) in self.episodes.iter().enumerate() {
            let (q, opening) = match episode {
                Some(d) => {
                    let s = f.min(d.trajectory.len() - 1);
                    if let Some(c) = &d.carried {
                        rec.log(format!("world/scene_{i}/object"), &transform(&c.poses[s]))?;
                    }
                    // The fingers close until they meet the box, across its narrower side.
                    let world = &self.worlds[self.tasks[i].world as usize];
                    let Obstacle::Cuboid { half_extents, .. } = world.obstacles[self.tasks[i].object] else {
                        unreachable!()
                    };
                    (
                        &d.trajectory.positions[s * n..][..n],
                        (d.gripper[s] * OPEN).max(half_extents.x.min(half_extents.y)),
                    )
                }
                None => (self.robot.default_q(), OPEN),
            };
            if hand_of == Some(i) {
                let hand = self.robot.link_pose(q, "panda_hand").context("the Panda's hand")?;
                rec.log(format!("world/scene_{i}/robot/panda_hand"), &transform(&hand))?;
            }
            let fingers = panda_fingers(self.robot, q, opening)?;
            robots.push((self.offsets[i], self.meshes.link_poses(self.robot, q, fingers)?));
        }
        self.meshes.log_robots(rec, &robots)
    }
}
