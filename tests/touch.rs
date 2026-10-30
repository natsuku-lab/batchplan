//! Links a world lets touch an obstacle (`World::touch`), on every device: evaluation, IK,
//! planning and the mesh check skip those pairs, and only those.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::{
    AttachedObject, CollisionWeights, Device, Error, IkOptions, IkProblem, MeshModel, Obstacle,
    PlanOptions, PlanProblem, Robot, SphereOptions, Touch, World,
};
use glam::Vec3;

fn panda() -> Robot {
    common::panda().unwrap()
}

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot).unwrap()];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

/// A bar between the open fingers at `q`, as a grasp holds it: through both finger pads.
fn grasped(robot: &Robot, q: &[f32]) -> Obstacle {
    let tool = robot.ee_pose(q, 0);
    Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.01, 0.06, 0.01)).placed(tool)
}

fn touching(obstacles: Vec<Obstacle>, obstacle: usize, links: &[&str]) -> World {
    World {
        touch: vec![Touch::new(obstacle, links.iter().copied())],
        ..World::new(obstacles)
    }
}

fn world_clearance(device: &Device, world: &World, q: &[f32]) -> f32 {
    let worlds = device.upload(std::slice::from_ref(world)).unwrap();
    device
        .evaluate(&worlds, &[0], q, &CollisionWeights::NONE)
        .unwrap()
        .world_clearance[0]
}

#[test]
fn hand_links_are_the_hand_and_its_fingers() {
    let robot = panda();
    let mut links = robot.hand_links(0);
    links.sort_unstable();
    assert_eq!(
        links,
        [
            "ee_link",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger"
        ]
    );
}

#[test]
fn links_touch_only_the_obstacles_the_world_lets_them() {
    let robot = panda();
    let q = robot.default_q().to_vec();
    let bar = grasped(&robot, &q);
    let hand = robot.hand_links(0);
    // An obstacle around the forearm, which no hand link reaches.
    let forearm = robot.link_pose(&q, "panda_link5").unwrap().position;
    let around_forearm = Obstacle::sphere(forearm, 0.08);
    for device in devices(&robot) {
        let name = device.name();
        let clearance = |world: &World| world_clearance(&device, world, &q);
        assert!(
            clearance(&World::new(vec![bar.clone()])) < 0.0,
            "{name}: the bar must collide with the fingers to test anything"
        );
        assert!(
            clearance(&touching(vec![bar.clone()], 0, &hand)) > 0.0,
            "{name}: the hand may touch the bar"
        );
        assert!(
            clearance(&touching(vec![bar.clone()], 0, &["panda_leftfinger"])) < 0.0,
            "{name}: the right finger may not touch it"
        );
        let both = vec![bar.clone(), around_forearm.clone()];
        assert!(
            clearance(&touching(both.clone(), 0, &hand)) < 0.0,
            "{name}: the forearm still collides with the other obstacle"
        );
        assert!(
            clearance(&touching(vec![around_forearm.clone()], 0, &hand)) < 0.0,
            "{name}: links not named still collide with a touched obstacle"
        );
    }
}

#[test]
fn ik_grasps_inside_an_object_only_where_the_hand_may_touch_it() {
    let robot = panda();
    let q = robot.default_q().to_vec();
    let bar = grasped(&robot, &q);
    let o = IkOptions {
        seeds: 8,
        ..Default::default()
    };
    let target = robot.ee_pose(&q, 0);
    for device in devices(&robot) {
        let name = device.name();
        let solved = |world: World| {
            let worlds = device.upload(&[world]).unwrap();
            let problem = IkProblem {
                seed: Some(q.clone()),
                ..IkProblem::new(0, target)
            };
            let result = batchplan::solve_ik(&device, &worlds, &[problem], &o).unwrap();
            result.best(0).is_some()
        };
        assert!(
            !solved(World::new(vec![bar.clone()])),
            "{name}: every solution puts the fingers in the bar"
        );
        assert!(
            solved(touching(vec![bar.clone()], 0, &robot.hand_links(0))),
            "{name}: with the hand allowed to touch it"
        );
    }
}

#[test]
fn plans_reach_into_an_object_only_where_the_hand_may_touch_it() {
    let robot = panda();
    let goal = robot.default_q().to_vec();
    let mut start = goal.clone();
    start[0] += 0.8;
    let bar = grasped(&robot, &goal);
    let o = PlanOptions::default();
    for device in devices(&robot) {
        let name = device.name();
        let planned = |world: &World| {
            let worlds = device.upload(std::slice::from_ref(world)).unwrap();
            let problem = PlanProblem::new(0, start.clone(), goal.clone());
            let result = batchplan::plan(&device, &worlds, &[problem], &o).unwrap();
            result.best(0).map(<[f32]>::to_vec)
        };
        assert!(
            planned(&World::new(vec![bar.clone()])).is_none(),
            "{name}: the goal has the fingers in the bar"
        );
        let world = touching(vec![bar.clone()], 0, &robot.hand_links(0));
        let path = planned(&world).unwrap_or_else(|| panic!("{name}: no path into the grasp"));
        let n = robot.dof();
        let end = &path[path.len() - n..];
        assert!(
            end.iter().zip(&goal).all(|(a, b)| (a - b).abs() < 1e-4),
            "{name}: the path ends at the grasp"
        );
    }
}

#[test]
fn mesh_checks_skip_what_the_world_lets_touch() {
    let robot = panda();
    let q = robot.default_q().to_vec();
    let model = MeshModel::try_new(&robot).unwrap();
    let bar = grasped(&robot, &q);
    let world = |w: World| model.clearance(&[w], &[0], &q).unwrap()[0].world;
    assert_eq!(
        world(World::new(vec![bar.clone()])),
        0.0,
        "the bar must pass through the finger meshes to test anything"
    );
    assert!(world(touching(vec![bar], 0, &robot.hand_links(0))) > 0.0);
}

#[test]
fn touches_name_links_and_obstacles_that_exist() {
    let robot = panda();
    let q = robot.default_q().to_vec();
    let bar = grasped(&robot, &q);
    let model = MeshModel::try_new(&robot).unwrap();
    let unknown = touching(vec![bar.clone()], 0, &["no_such_link"]);
    let past_the_end = touching(vec![bar], 1, &["panda_hand"]);
    for device in devices(&robot) {
        let name = device.name();
        for (world, says) in [
            (&unknown, "no_such_link"),
            (&past_the_end, "obstacle 1 of 1"),
        ] {
            let err = device.upload(std::slice::from_ref(world)).unwrap_err();
            assert!(
                matches!(&err, Error::Input(m) if m.contains(says)),
                "{name}: {err:?}"
            );
        }
    }
    let err = model.clearance(&[unknown], &[0], &q).unwrap_err();
    assert!(
        matches!(&err, Error::Input(m) if m.contains("no_such_link")),
        "{err:?}"
    );
}

#[test]
fn a_held_object_may_touch_what_the_world_lets_it() {
    let robot = panda();
    let q = robot.default_q().to_vec();
    // A box hanging below the fingertips, and a small table its bottom sinks into, out of the
    // arm's reach.
    let held = robot
        .attach(&AttachedObject {
            name: "held box".into(),
            link: "panda_hand".into(),
            shapes: vec![Obstacle::cuboid(
                Vec3::new(0.0, 0.0, 0.17),
                Vec3::new(0.03, 0.03, 0.05),
            )],
            touch_links: vec!["panda_leftfinger".into(), "panda_rightfinger".into()],
            spheres: SphereOptions::default(),
        })
        .unwrap();
    let bottom = held.link_pose(&q, "panda_hand").unwrap();
    let bottom = bottom.position + bottom.rotation * Vec3::new(0.0, 0.0, 0.22);
    let table = Obstacle::cuboid(
        bottom - Vec3::new(0.0, 0.0, 0.05 - 0.01),
        Vec3::new(0.06, 0.06, 0.05),
    );
    for device in devices(&robot) {
        let name = device.name();
        let holder = device.with_robot(&held).unwrap();
        let clearance = |world: &World| world_clearance(&holder, world, &q);
        assert!(
            clearance(&World::new(vec![table.clone()])) < 0.0,
            "{name}: the box must sink into the table to test anything"
        );
        let resting = touching(vec![table.clone()], 0, &["held box"]);
        assert!(clearance(&resting) > 0.0, "{name}: the box may rest on it");
        // Only the device holding the box has a link of that name.
        let err = device.upload(&[resting]).unwrap_err();
        assert!(
            matches!(&err, Error::Input(m) if m.contains("held box")),
            "{name}: {err:?}"
        );
    }
}

#[test]
fn worlds_without_touches_serialize_as_before() {
    let bar = Obstacle::cuboid(Vec3::ZERO, Vec3::splat(0.1));
    let plain = serde_json::to_value(World::new(vec![bar.clone()])).unwrap();
    assert!(plain.get("touch").is_none(), "{plain}");
    let world = touching(vec![bar], 0, &["panda_hand"]);
    let back: World = serde_json::from_value(serde_json::to_value(&world).unwrap()).unwrap();
    assert_eq!(back.touch, world.touch);
    let old: World = serde_json::from_value(plain).unwrap();
    assert!(old.touch.is_empty());
}
