//! Random tabletop worlds shared by the examples and tests.
#![allow(dead_code)]

use std::f32::consts::PI;

use std::collections::HashMap;

use batchplan::rng::Rng;
use batchplan::{
    CollisionModel, CollisionWeights, Device, IkOptions, IkProblem, Obstacle, PlanProblem, Pose, Robot, RobotOptions,
    World, solve_ik,
};
use glam::{Quat, Vec3};

pub fn asset(path: &str) -> String {
    format!("{}/assets/{path}", env!("CARGO_MANIFEST_DIR"))
}

/// The Franka Panda with its fingers held open, a 10 cm tool frame below the hand as the IK
/// frame, and hand-tuned collision spheres converted from cuRobo.
pub fn panda_options() -> RobotOptions {
    let fingers = [("panda_finger_joint1".to_string(), 0.04), ("panda_finger_joint2".to_string(), 0.04)];
    RobotOptions {
        ee_links: vec!["ee_link".into()],
        lock_joints: HashMap::from(fingers),
        default_q: Some(vec![0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]),
        collision_model: Some(CollisionModel::load(asset("franka/panda_collision.json")).expect("Panda spheres")),
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
    let arm = |prefix: &str, y: f32| batchplan::Part {
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
            .flat_map(|p| one.lock_joints.iter().map(move |(j, v)| (format!("{p}{j}"), *v)))
            .collect(),
        collision_model: Some(CollisionModel::assemble(&[("left_", spheres), ("right_", spheres)])),
        ..Default::default()
    };
    // Joints come breadth first, the arms interleaved: each takes its ready pose by name.
    let names = Robot::assemble(&parts, &options)?.joint_names().to_vec();
    let single = Robot::load(asset("franka/franka_panda.urdf"), &one)?;
    let ready = |name: &str| {
        let joint = name.trim_start_matches("left_").trim_start_matches("right_");
        single.joint_names().iter().position(|j| j == joint).map(|k| single.default_q()[k])
    };
    options.default_q = Some(names.iter().map(|n| ready(n).expect("a Panda joint")).collect());
    Robot::assemble(&parts, &options)
}

/// A table at z = 0 (the robot base height) with 2-6 random boxes in front of the robot.
pub fn tabletop(rng: &mut Rng) -> World {
    let mut obstacles = vec![Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02))];
    let boxes = 2 + (rng.uniform() * 5.0) as usize;
    for _ in 0..boxes {
        let half = Vec3::new(rng.range(0.02, 0.08), rng.range(0.02, 0.08), rng.range(0.03, 0.2));
        obstacles.push(Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.3, 0.75), rng.range(-0.5, 0.5), half.z),
            half_extents: half,
            rotation: Quat::from_rotation_z(rng.range(0.0, PI)),
        });
    }
    World { obstacles }
}

/// A top-down grasp pose (gripper z axis pointing down) in free space above the table.
pub fn grasp_target(world: &World, rng: &mut Rng) -> Pose {
    loop {
        let p = Vec3::new(rng.range(0.3, 0.7), rng.range(-0.45, 0.45), rng.range(0.05, 0.4));
        if world.obstacles.iter().all(|o| o.distance(p).0 > 0.06) {
            let rotation = Quat::from_rotation_z(rng.range(-PI, PI)) * Quat::from_rotation_x(PI);
            return Pose { position: p, rotation };
        }
    }
}

/// A table under any robot and three posts within its reach; none of them touch the default pose.
pub fn tabletop_for(robot: &Robot, cpu: &Device, rng: &mut Rng) -> World {
    let mut reach = 0.0f32;
    for _ in 0..500 {
        let q: Vec<f32> = (0..robot.dof()).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect();
        reach = reach.max(robot.ee_pose(&q, 0).position.truncate().length());
    }
    let model = robot.collision_model();
    let floor = model
        .spheres
        .iter()
        .flat_map(|(link, spheres)| {
            let pose = robot.link_pose(robot.default_q(), link).unwrap();
            spheres.iter().map(move |s| (pose.position + pose.rotation * Vec3::new(s[0], s[1], s[2])).z - s[3])
        })
        .fold(f32::INFINITY, f32::min);
    let table = Obstacle::cuboid(
        Vec3::new(0.0, 0.0, floor - 0.01 - reach * 0.05),
        Vec3::new(2.0 * reach, 2.0 * reach, reach * 0.05),
    );
    let mut world = World { obstacles: vec![table] };
    while world.obstacles.len() < 4 {
        let (angle, radius) = (rng.range(-3.1, 3.1), rng.range(0.4, 0.8) * reach);
        let half = Vec3::new(0.06, 0.06, 0.3) * reach;
        let candidate = Obstacle::Cuboid {
            center: Vec3::new(radius * angle.cos(), radius * angle.sin(), floor - 0.01 + half.z),
            half_extents: half,
            rotation: Quat::from_rotation_z(rng.range(0.0, 3.1)),
        };
        let mut trial = world.clone();
        trial.obstacles.push(candidate);
        let uploaded = cpu.upload(std::slice::from_ref(&trial)).unwrap();
        if cpu.evaluate(&uploaded, &[0], robot.default_q(), &CollisionWeights::NONE).unwrap().collision_free(0) {
            world = trial;
        }
    }
    world
}

/// A table with its top at z = 0, two tall boxes, and a small box to move across the table: the
/// world and the task. The small box is the world's last obstacle.
pub fn pick_place_scene(world: u32, rng: &mut Rng) -> (World, batchplan::datagen::PickPlace) {
    let table = Obstacle::cuboid(Vec3::new(0.4, 0.0, -0.02), Vec3::new(0.7, 0.8, 0.02));
    let mut obstacles = vec![table];
    for side in [-1.0, 1.0] {
        obstacles.push(Obstacle::Cuboid {
            center: Vec3::new(rng.range(0.6, 0.7), side * rng.range(0.3, 0.45), 0.15),
            half_extents: Vec3::new(0.04, 0.04, 0.15),
            rotation: Quat::from_rotation_z(rng.range(0.0, 3.0)),
        });
    }
    let half = Vec3::new(rng.range(0.015, 0.03), rng.range(0.025, 0.04), rng.range(0.02, 0.04));
    let spot = |rng: &mut Rng, side: f32| Vec3::new(rng.range(0.35, 0.55), side * rng.range(0.1, 0.3), half.z);
    let side = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
    obstacles.push(Obstacle::Cuboid {
        center: spot(rng, side),
        half_extents: half,
        rotation: Quat::from_rotation_z(rng.range(-1.2, 1.2)),
    });
    let place = Pose::new(spot(rng, -side), Quat::from_rotation_z(rng.range(-1.2, 1.2)));
    let object = obstacles.len() - 1;
    (World { obstacles }, batchplan::datagen::PickPlace { world, object, place })
}

/// UR5e motions from the default pose (arm straight out over the table) to far goals in tabletop
/// worlds; `above_table` keeps the shoulder within the half turn above the table.
pub fn far_reaches(robot: &Robot, cpu: &Device, count: usize, above_table: bool) -> (Vec<World>, Vec<PlanProblem>) {
    let mut rng = Rng::new(14);
    let scene: Vec<World> = (0..4).map(|_| tabletop_for(robot, cpu, &mut rng)).collect();
    let worlds = cpu.upload(&scene).unwrap();
    let mut problems = vec![];
    while problems.len() < count {
        let world = problems.len() as u32 % 4;
        let mut goal: Vec<f32> = (0..robot.dof()).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect();
        if above_table {
            goal[1] = rng.range(-3.0, -0.1);
        }
        let far = goal.iter().zip(robot.default_q()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt() > 4.0;
        if far && cpu.evaluate(&worlds, &[world], &goal, &CollisionWeights::NONE).unwrap().collision_free(0) {
            problems.push(PlanProblem::new(world, robot.default_q(), goal));
        }
    }
    (scene, problems)
}

/// 64 tabletop reaches from the default pose, the worlds they live in, and their IK goals.
pub fn reaches(robot: &Robot) -> (Vec<World>, Vec<PlanProblem>) {
    let mut rng = batchplan::rng::Rng::new(41);
    let scene: Vec<World> = (0..64).map(|_| tabletop(&mut rng)).collect();
    let cpu = Device::cpu(robot).unwrap();
    let goals: Vec<IkProblem> =
        scene.iter().enumerate().map(|(i, w)| IkProblem::new(i as u32, grasp_target(w, &mut rng))).collect();
    let ik = solve_ik(&cpu, &cpu.upload(&scene).unwrap(), &goals, &IkOptions::default()).unwrap();
    let problems = ik.solved().map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution)).collect();
    (scene, problems)
}

/// `reaches` with a ball where each hand passes halfway along its straight line, so every search
/// detours and optimization has work to do; problems whose ends the ball touches are dropped.
pub fn blocked_reaches(robot: &Robot) -> (Vec<World>, Vec<PlanProblem>) {
    let (mut scene, problems) = reaches(robot);
    for p in &problems {
        let middle: Vec<f32> = p.start.iter().zip(&p.goal).map(|(a, b)| 0.5 * (a + b)).collect();
        let center = robot.ee_pose(&middle, 0).position;
        scene[p.world as usize].obstacles.push(Obstacle::sphere(center, 0.1));
    }
    let cpu = Device::cpu(robot).unwrap();
    let worlds = cpu.upload(&scene).unwrap();
    let free = |p: &PlanProblem| {
        let ends = [&p.start[..], &p.goal[..]].concat();
        let e = cpu.evaluate(&worlds, &[p.world, p.world], &ends, &CollisionWeights::NONE).unwrap();
        e.collision_free(0) && e.collision_free(1)
    };
    let problems = problems.into_iter().filter(free).collect();
    (scene, problems)
}
