//! Random tabletop worlds shared by the examples and tests.
#![allow(dead_code, reason = "each example and test uses part of it")]

use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI};

use batchplan::datagen::PickPlace;
use batchplan::rng::Rng;
use batchplan::{
    CollisionModel, CollisionWeights, Device, HandConstraint, IkOptions, IkProblem, Obstacle, Part,
    PlanProblem, Pose, Robot, RobotOptions, World,
};
use glam::{Quat, Vec3};

pub fn asset(path: &str) -> String {
    format!("{}/assets/{path}", env!("CARGO_MANIFEST_DIR"))
}

/// The Franka Panda with its fingers held open, a 10 cm tool frame below the hand as the IK
/// frame, and hand-tuned collision spheres converted from cuRobo.
pub fn panda_options() -> RobotOptions {
    let fingers = [
        ("panda_finger_joint1".to_string(), 0.04),
        ("panda_finger_joint2".to_string(), 0.04),
    ];
    RobotOptions {
        ee_links: vec!["ee_link".into()],
        lock_joints: HashMap::from(fingers),
        default_q: Some(vec![0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]),
        collision_model: Some(
            CollisionModel::load(asset("franka/panda_collision.json")).expect("Panda spheres"),
        ),
        ..Default::default()
    }
}

pub fn panda() -> Result<Robot, batchplan::Error> {
    Robot::load(asset("franka/franka_panda.urdf"), &panda_options())
}

/// Two Pandas 70 cm apart along y, facing +x, as `left_` and `right_` parts: hands 0 and 1, each
/// at its ready pose with its fingers held open.
pub fn two_pandas() -> Result<Robot, batchplan::Error> {
    let one = panda_options();
    let arm = |prefix: &str, y: f32| Part {
        path: asset("franka/franka_panda.urdf").into(),
        prefix: prefix.into(),
        base: Pose::new(Vec3::new(0.0, y, 0.0), Quat::IDENTITY),
    };
    let parts = [arm("left_", 0.35), arm("right_", -0.35)];
    let spheres = one.collision_model.as_ref().expect("Panda spheres");
    let mut options = RobotOptions {
        ee_links: vec!["left_ee_link".into(), "right_ee_link".into()],
        lock_joints: ["left_", "right_"]
            .iter()
            .flat_map(|p| {
                one.lock_joints
                    .iter()
                    .map(move |(j, v)| (format!("{p}{j}"), *v))
            })
            .collect(),
        collision_model: Some(CollisionModel::assemble(&[
            ("left_", spheres),
            ("right_", spheres),
        ])),
        ..Default::default()
    };
    // Joints come breadth first, the arms interleaved: each takes its ready pose by name.
    let names = Robot::assemble(&parts, &options)?.joint_names().to_vec();
    let single = Robot::load(asset("franka/franka_panda.urdf"), &one)?;
    let ready = |name: &str| {
        let joint = name
            .trim_start_matches("left_")
            .trim_start_matches("right_");
        single
            .joint_names()
            .iter()
            .position(|j| j == joint)
            .map(|k| single.default_q()[k])
    };
    options.default_q = Some(
        names
            .iter()
            .map(|n| ready(n).expect("a Panda joint"))
            .collect(),
    );
    Robot::assemble(&parts, &options)
}

/// A table at z = 0 (the robot base height) with 2-6 random boxes in front of the robot.
pub fn tabletop(rng: &mut Rng) -> World {
    let mut obstacles = vec![Obstacle::cuboid(
        Vec3::new(0.4, 0.0, -0.02),
        Vec3::new(0.7, 0.8, 0.02),
    )];
    let boxes = 2 + (rng.uniform() * 5.0) as usize;
    for _ in 0..boxes {
        let half = Vec3::new(
            rng.range(0.02, 0.08),
            rng.range(0.02, 0.08),
            rng.range(0.03, 0.2),
        );
        obstacles.push(Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.3, 0.75), rng.range(-0.5, 0.5), half.z),
            half_extents: half,
            rotation: Quat::from_rotation_z(rng.range(0.0, PI)),
        });
    }
    World::new(obstacles)
}

/// A top-down grasp pose (gripper z axis pointing down) in free space above the table.
pub fn grasp_target(world: &World, rng: &mut Rng) -> Pose {
    loop {
        let p = Vec3::new(
            rng.range(0.3, 0.7),
            rng.range(-0.45, 0.45),
            rng.range(0.05, 0.4),
        );
        if world.obstacles.iter().all(|o| o.distance(p).value > 0.06) {
            let rotation = Quat::from_rotation_z(rng.range(-PI, PI)) * Quat::from_rotation_x(PI);
            return Pose {
                position: p,
                rotation,
            };
        }
    }
}

/// A table under any robot and three posts within its reach; none of them touch the default pose.
pub fn tabletop_for(robot: &Robot, cpu: &Device, rng: &mut Rng) -> World {
    let mut reach = 0.0f32;
    for _ in 0..500 {
        let q: Vec<f32> = (0..robot.dof())
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect();
        reach = reach.max(robot.ee_pose(&q, 0).position.truncate().length());
    }
    let model = robot.collision_model();
    let floor = model
        .spheres
        .iter()
        .flat_map(|(link, spheres)| {
            let pose = robot
                .link_pose(robot.default_q(), link)
                .expect("the collision model's links are the robot's");
            spheres.iter().map(move |s| {
                (pose.position + pose.rotation * Vec3::new(s[0], s[1], s[2])).z - s[3]
            })
        })
        .fold(f32::INFINITY, f32::min);
    let table = Obstacle::cuboid(
        Vec3::new(0.0, 0.0, floor - 0.01 - reach * 0.05),
        Vec3::new(2.0 * reach, 2.0 * reach, reach * 0.05),
    );
    let mut world = World::new(vec![table]);
    while world.obstacles.len() < 4 {
        let (angle, radius) = (rng.range(-3.1, 3.1), rng.range(0.4, 0.8) * reach);
        let half = Vec3::new(0.06, 0.06, 0.3) * reach;
        let candidate = Obstacle::Cuboid {
            center: Vec3::new(
                radius * angle.cos(),
                radius * angle.sin(),
                floor - 0.01 + half.z,
            ),
            half_extents: half,
            rotation: Quat::from_rotation_z(rng.range(0.0, 3.1)),
        };
        let mut trial = world.clone();
        trial.obstacles.push(candidate);
        let uploaded = cpu
            .upload(std::slice::from_ref(&trial))
            .expect("boxes upload");
        if cpu
            .evaluate(&uploaded, &[0], robot.default_q(), &CollisionWeights::NONE)
            .expect("one configuration in one world")
            .collision_free(0)
        {
            world = trial;
        }
    }
    world
}

/// A table with its top at z = 0, two tall boxes, and a small box to move across the table: the
/// world and the task. The small box is the world's last obstacle.
pub fn pick_place_scene(world: u32, rng: &mut Rng) -> (World, PickPlace) {
    let table = Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let mut obstacles = vec![table];
    for side in [-1.0, 1.0] {
        obstacles.push(Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.6, 0.7), side * rng.range(0.3, 0.45), 0.15),
            half_extents: Vec3::new(0.04, 0.04, 0.15),
            rotation: Quat::from_rotation_z(rng.range(0.0, 3.0)),
        });
    }
    let half = Vec3::new(
        rng.range(0.015, 0.03),
        rng.range(0.025, 0.04),
        rng.range(0.02, 0.04),
    );
    let spot = |rng: &mut Rng, side: f32| {
        Vec3::new(rng.range(0.35, 0.55), side * rng.range(0.1, 0.3), half.z)
    };
    let side = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
    obstacles.push(Obstacle::Cuboid {
        center: spot(rng, side),
        half_extents: half,
        rotation: Quat::from_rotation_z(rng.range(-1.2, 1.2)),
    });
    let place = Pose::new(
        spot(rng, -side),
        Quat::from_rotation_z(rng.range(-1.2, 1.2)),
    );
    let object = obstacles.len() - 1;
    (
        World::new(obstacles),
        PickPlace {
            world,
            object,
            place,
            support: vec![0],
        },
    )
}

/// UR5e motions from the default pose (arm straight out over the table) to far goals in tabletop
/// worlds; `above_table` keeps the shoulder within the half turn above the table.
pub fn far_reaches(
    robot: &Robot,
    cpu: &Device,
    count: usize,
    above_table: bool,
) -> (Vec<World>, Vec<PlanProblem>) {
    let mut rng = Rng::new(14);
    let scene: Vec<World> = (0..4).map(|_| tabletop_for(robot, cpu, &mut rng)).collect();
    let worlds = cpu.upload(&scene).expect("boxes upload");
    let mut problems = vec![];
    while problems.len() < count {
        let world = problems.len() as u32 % 4;
        let mut goal: Vec<f32> = (0..robot.dof())
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect();
        if above_table {
            goal[1] = rng.range(-3.0, -0.1);
        }
        let far = goal
            .iter()
            .zip(robot.default_q())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            > 4.0;
        if far
            && cpu
                .evaluate(&worlds, &[world], &goal, &CollisionWeights::NONE)
                .expect("one configuration in one world")
                .collision_free(0)
        {
            problems.push(PlanProblem::new(world, robot.default_q(), goal));
        }
    }
    (scene, problems)
}

/// 64 tabletop reaches from the default pose, the worlds they live in, and their IK goals.
pub fn reaches(robot: &Robot) -> (Vec<World>, Vec<PlanProblem>) {
    let mut rng = Rng::new(41);
    let scene: Vec<World> = (0..64).map(|_| tabletop(&mut rng)).collect();
    let cpu = Device::cpu(robot).expect("the CPU device starts");
    let goals: Vec<IkProblem> = scene
        .iter()
        .enumerate()
        .map(|(i, w)| IkProblem::new(i as u32, grasp_target(w, &mut rng)))
        .collect();
    let ik = batchplan::solve_ik(
        &cpu,
        &cpu.upload(&scene).expect("boxes upload"),
        &goals,
        &IkOptions::default(),
    )
    .expect("IK takes tabletop targets");
    let problems = ik
        .solved()
        .map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution))
        .collect();
    (scene, problems)
}

/// `reaches` with a ball where each hand passes halfway along its straight line, so every search
/// detours and optimization has work to do; problems whose ends the ball touches are dropped.
pub fn blocked_reaches(robot: &Robot) -> (Vec<World>, Vec<PlanProblem>) {
    let (mut scene, problems) = reaches(robot);
    for p in &problems {
        let middle: Vec<f32> = p
            .start
            .iter()
            .zip(&p.goal)
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let center = robot.ee_pose(&middle, 0).position;
        scene[p.world as usize]
            .obstacles
            .push(Obstacle::sphere(center, 0.1));
    }
    let cpu = Device::cpu(robot).expect("the CPU device starts");
    let worlds = cpu.upload(&scene).expect("boxes and balls upload");
    let free = |p: &PlanProblem| {
        let ends = [&p.start[..], &p.goal[..]].concat();
        let e = cpu
            .evaluate(&worlds, &[p.world, p.world], &ends, &CollisionWeights::NONE)
            .expect("two configurations in one world");
        e.collision_free(0) && e.collision_free(1)
    };
    let problems = problems.into_iter().filter(free).collect();
    (scene, problems)
}

/// The zone the upright reaches start and end in (meters).
pub const ZONE_MIN: Vec3 = Vec3::new(0.30, -0.45, 0.25);
pub const ZONE_MAX: Vec3 = Vec3::new(0.75, 0.45, 0.70);

/// The upright reaches' scene: a floor under the robot, and a pillar in the zone between the
/// start of `upright_reaches` and many of its goals.
pub fn pillar_scene() -> World {
    let size = ZONE_MAX - ZONE_MIN;
    let height = 0.85 * ZONE_MAX.z;
    let floor = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.05), Vec3::new(1.5, 1.5, 0.05));
    let pillar = Obstacle::cylinder(
        Vec3::new(ZONE_MIN.x + 0.12 * size.x, 0.0, 0.5 * height),
        0.06,
        0.5 * height,
    );
    World::new(vec![floor, pillar])
}

/// The Panda's gripper pointing straight down, turned `yaw` about the vertical.
pub fn upright(yaw: f32) -> Quat {
    Quat::from_rotation_x(PI) * Quat::from_rotation_z(yaw)
}

/// Where `upright_reaches` starts the gripper: in a corner of the zone, between the pillar and the
/// robot's right.
pub fn zone_corner() -> Vec3 {
    ZONE_MIN + 0.12 * (ZONE_MAX - ZONE_MIN)
}

/// Reaches with the gripper upright in `pillar_scene` (world 0 of `worlds`): from the zone's corner,
/// solved by IK nearest the robot's ready pose, to `count` places in the zone (`upright_goals`).
/// The start, and the goals IK reached.
pub fn upright_reaches(
    device: &Device,
    worlds: &batchplan::Worlds,
    count: usize,
    rng: &mut Rng,
) -> Result<(Vec<f32>, Vec<Vec<f32>>), batchplan::Error> {
    let ready = device.robot().default_q();
    let first = IkProblem {
        seed: Some(ready.to_vec()),
        ..IkProblem::new(0, Pose::new(zone_corner(), upright(0.0)))
    };
    let start = batchplan::solve_ik(device, worlds, &[first], &IkOptions::default())?
        .nearest(0, ready)
        .expect("the start is in reach")
        .to_vec();
    let goals = upright_goals(device, worlds, &start, count, rng)?;
    Ok((start, goals))
}

/// `count` places in the zone at random turns about the vertical, the gripper upright, each solved
/// by IK from `start` and the solution nearest it kept: the goals IK reached.
pub fn upright_goals(
    device: &Device,
    worlds: &batchplan::Worlds,
    start: &[f32],
    count: usize,
    rng: &mut Rng,
) -> Result<Vec<Vec<f32>>, batchplan::Error> {
    let inset = 0.12 * (ZONE_MAX - ZONE_MIN);
    let targets: Vec<IkProblem> = (0..count)
        .map(|_| {
            let at = Vec3::new(
                rng.range(ZONE_MIN.x + inset.x, ZONE_MAX.x - inset.x),
                rng.range(ZONE_MIN.y + inset.y, ZONE_MAX.y - inset.y),
                rng.range(ZONE_MIN.z + inset.z, ZONE_MAX.z - inset.z),
            );
            IkProblem {
                seed: Some(start.to_vec()),
                ..IkProblem::new(0, Pose::new(at, upright(rng.range(-PI, PI))))
            }
        })
        .collect();
    let ik = batchplan::solve_ik(device, worlds, &targets, &IkOptions::default())?;
    Ok((0..count)
        .filter_map(|i| ik.nearest(i, start).map(<[f32]>::to_vec))
        .collect())
}

/// The scene of `tray_carries`: a block 25 cm tall in front of two arms (`two_pandas`).
pub fn tray_scene() -> World {
    World::new(vec![Obstacle::cuboid(
        Vec3::new(0.5, 0.0, 0.125),
        Vec3::new(0.06, 0.4, 0.125),
    )])
}

/// A tray at `at` turned `yaw` about the vertical.
pub fn tray(at: Vec3, yaw: f32) -> Pose {
    Pose::new(at, Quat::from_rotation_z(yaw))
}

/// Two arms (`two_pandas`) holding a tray at `tray`, each gripper pointing down and closing across
/// an edge 12 cm either side of its middle, its narrow side to the other's and turned half a turn
/// from it, so that one hand's place in the other's frame turns as well: the left hand at its
/// edge, and the right at `right` in the left hand's frame, or else at its own edge. The IK
/// solution in world 0 of `worlds` nearest `seed`, if any.
pub fn tray_grip(
    device: &Device,
    worlds: &batchplan::Worlds,
    tray: Pose,
    right: Option<Pose>,
    seed: &[f32],
) -> Result<Option<Vec<f32>>, batchplan::Error> {
    let grasp = |side: f32| Pose::new(Vec3::new(0.0, 0.12 * side, 0.0), upright(FRAC_PI_2 * side));
    let left = tray.mul_pose(grasp(1.0));
    let right = right.map_or_else(|| tray.mul_pose(grasp(-1.0)), |r| left.mul_pose(r));
    let problem = IkProblem {
        world: 0,
        targets: vec![Some(left), Some(right)],
        seed: Some(seed.to_vec()),
        hold: vec![],
        holding: None,
    };
    let solved = batchplan::solve_ik(device, worlds, &[problem], &IkOptions::default())?;
    Ok(solved.nearest(0, seed).map(<[f32]>::to_vec))
}

/// Two arms carrying a tray (`tray_carries`).
pub struct TrayCarries {
    pub start: Vec<f32>,
    /// The right hand's place in the left hand's frame at the start.
    pub place: Pose,
    /// The goals IK reached keeping it.
    pub goals: Vec<Vec<f32>>,
}

/// Two arms carrying a tray over the block of `tray_scene` (world 0 of `worlds`), from below its
/// top to places beyond it at turns about the vertical, where no straight walk gets.
pub fn tray_carries(
    device: &Device,
    worlds: &batchplan::Worlds,
) -> Result<TrayCarries, batchplan::Error> {
    let robot = device.robot();
    let at = tray(Vec3::new(0.3, 0.0, 0.2), 0.0);
    let start =
        tray_grip(device, worlds, at, None, robot.default_q())?.expect("the tray is in reach");
    let place = robot
        .ee_pose(&start, 0)
        .inverse()
        .mul_pose(robot.ee_pose(&start, 1));
    let mut goals = vec![];
    for (at, yaw) in [
        (Vec3::new(0.62, 0.0, 0.32), 0.0),
        (Vec3::new(0.62, 0.05, 0.36), 0.3),
        (Vec3::new(0.6, -0.05, 0.34), -0.3),
        (Vec3::new(0.65, 0.0, 0.4), 0.15),
        (Vec3::new(0.66, 0.0, 0.3), 0.0),
    ] {
        goals.extend(tray_grip(
            device,
            worlds,
            tray(at, yaw),
            Some(place),
            &start,
        )?);
    }
    Ok(TrayCarries {
        start,
        place,
        goals,
    })
}

/// A tray carried level by two arms: the right hand pinned to `place` in the left hand's frame,
/// the left within `tilt` of straight down.
pub fn carried_level(place: Pose, tilt: f32) -> [HandConstraint; 2] {
    let pinned = HandConstraint {
        relative_to: Some(0),
        position: Vec3::ZERO,
        tilt: 0.0,
        twist: 0.0,
        ..HandConstraint::new(1, place)
    };
    let level = HandConstraint {
        tilt,
        ..HandConstraint::new(0, Pose::new(Vec3::ZERO, upright(0.0)))
    };
    [pinned, level]
}
