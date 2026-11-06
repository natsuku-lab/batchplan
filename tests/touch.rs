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
fn removing_an_obstacle_keeps_every_touch_on_its_obstacle() {
    let ball = |x: f32| Obstacle::sphere(Vec3::new(x, 0.0, 0.0), 0.1);
    let mut world = World {
        touch: vec![
            Touch::new(0, ["a"]),
            Touch::new(1, ["b"]),
            Touch::new(2, ["c"]),
        ],
        ..World::new(vec![ball(0.0), ball(1.0), ball(2.0)])
    };
    assert_eq!(world.remove(1).unwrap(), ball(1.0));
    assert_eq!(world.obstacles, vec![ball(0.0), ball(2.0)]);
    assert_eq!(
        world.touch,
        vec![Touch::new(0, ["a"]), Touch::new(1, ["c"])]
    );
    assert_eq!(world.remove(0).unwrap(), ball(0.0));
    assert_eq!(world.touch, vec![Touch::new(0, ["c"])]);
    assert!(matches!(world.remove(1), Err(Error::Input(_))));
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

#[test]
fn links_that_touch_their_table_in_every_configuration_are_found() {
    let robot = Robot::load(
        common::asset("so101/so101_new_calib.urdf"),
        &Default::default(),
    )
    .unwrap();
    // A table whose top is a centimeter above the base's foot, through the spheres of the link
    // that turns on the base; a post beside the arm that some motions reach.
    let table = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.19), Vec3::new(0.5, 0.5, 0.2));
    let post = Obstacle::cuboid(Vec3::new(0.25, 0.0, 0.1), Vec3::new(0.02, 0.02, 0.1));
    // A block the gripper is in at the default configuration, and only near it.
    let at_rest = robot.link_pose(robot.default_q(), "gripper_link").unwrap();
    let block = Obstacle::sphere(at_rest.position, 0.03);
    // A slab under the table, 3.5 cm below the shoulder's spheres whatever the arm does.
    let below = Obstacle::cuboid(Vec3::new(0.0, 0.0, -0.33), Vec3::new(0.5, 0.5, 0.3));
    let with_block = World::new(vec![table.clone(), post.clone(), block, below]);
    assert_eq!(
        with_block.always_touching(&robot),
        vec![Touch::new(0, ["shoulder_link"])]
    );
    let world = World::new(vec![table, post]);
    let touches = world.always_touching(&robot);
    assert_eq!(touches, vec![Touch::new(0, ["shoulder_link"])]);
    // Without them, nothing can be planned from rest; with them, it can.
    let start = robot.default_q().to_vec();
    let mut goal = start.clone();
    goal[0] += 1.0;
    let problem = PlanProblem::new(0, start, goal);
    let mounted = World {
        touch: touches,
        ..world.clone()
    };
    for d in devices(&robot) {
        let worlds = d.upload(&[world.clone(), mounted.clone()]).unwrap();
        let o = PlanOptions::default();
        let bare = batchplan::plan(&d, &worlds, std::slice::from_ref(&problem), &o).unwrap();
        assert!(
            matches!(
                bare.failures[0],
                Some(batchplan::PlanFailure::StartCollides { .. })
            ),
            "{}: {:?}",
            d.name(),
            bare.failures[0]
        );
        let on_table = PlanProblem {
            world: 1,
            ..problem.clone()
        };
        let planned = batchplan::plan(&d, &worlds, &[on_table], &o).unwrap();
        assert!(planned.best(0).is_some(), "{}", d.name());
    }
    // Touches the world already has are not found again.
    assert!(mounted.always_touching(&robot).is_empty());
    // A turntable turns without limit, its table always in a slab level with it.
    let dir = std::env::temp_dir().join(format!("batchplan-turntable-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("turntable.urdf"), TURNTABLE).unwrap();
    let turntable = Robot::load(dir.join("turntable.urdf"), &Default::default()).unwrap();
    assert_eq!(turntable.continuous(), [true, false]);
    let slab = Obstacle::cuboid(Vec3::new(0.0, 0.0, 0.04), Vec3::new(1.0, 1.0, 0.1));
    let touches = World::new(vec![slab]).always_touching(&turntable);
    assert!(
        touches.len() == 1 && touches[0].links.iter().any(|l| l == "table"),
        "{touches:?}"
    );
}

/// A turntable arm: a continuous joint about z carrying a 40 cm table along x, and an arm.
const TURNTABLE: &str = r#"<robot name="turntable">
  <link name="base"/>
  <link name="table"><collision><origin xyz="0.2 0 0.05"/><geometry><box size="0.4 0.06 0.06"/></geometry></collision></link>
  <link name="arm"><collision><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.05 0.05"/></geometry></collision></link>
  <joint name="turn" type="continuous"><parent link="base"/><child link="table"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/></joint>
  <joint name="shoulder" type="revolute"><parent link="table"/><child link="arm"/><origin xyz="0.4 0 0.05"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" velocity="2" effort="1"/></joint>
</robot>"#;

#[test]
fn fingers_named_in_the_options_are_the_hands_fingers() {
    let path = common::asset("so101/so101_new_calib.urdf");
    let frame = "gripper_frame_link";
    let bare = batchplan::RobotOptions {
        ee_links: vec![frame.into()],
        ..Default::default()
    };
    // The SO-101's fixed jaw is part of its palm's link: only the moving jaw is below it.
    let robot = Robot::load(&path, &bare).unwrap();
    assert_eq!(robot.finger_links(0), ["moving_jaw_so101_v1_link"]);
    let jaws = ["gripper_link", "moving_jaw_so101_v1_link"];
    let named = batchplan::RobotOptions {
        finger_links: [(frame.to_string(), jaws.map(String::from).to_vec())].into(),
        collision_model: Some(robot.collision_model().clone()),
        ..bare.clone()
    };
    let named = Robot::load(&path, &named).unwrap();
    assert_eq!(named.finger_links(0), jaws);
    let mut hand = named.hand_links(0);
    hand.sort_unstable();
    assert_eq!(hand, [frame, "gripper_link", "moving_jaw_so101_v1_link"]);
    // A block against the fixed jaw, between the jaws: the named fingers may touch it.
    let q = robot.default_q();
    let palm = robot.link_pose(q, "gripper_link").unwrap();
    let tool = robot.ee_pose(q, 0);
    let block = Obstacle::sphere(palm.position.lerp(tool.position, 0.5), 0.015);
    for (robot, free) in [(&robot, false), (&named, true)] {
        let device = Device::cpu(robot).unwrap();
        let world = touching(vec![block.clone()], 0, &robot.finger_links(0));
        assert_eq!(
            world_clearance(&device, &world, q) >= 0.0,
            free,
            "{:?}",
            robot.finger_links(0)
        );
    }
    // Names that are not a hand, or not a link, are load errors.
    for (hand, link) in [("wrist_link", "gripper_link"), (frame, "jaw")] {
        let wrong = batchplan::RobotOptions {
            finger_links: [(hand.to_string(), vec![link.to_string()])].into(),
            ..bare.clone()
        };
        let err = Robot::load(&path, &wrong).unwrap_err();
        assert!(
            matches!(err, Error::Load { .. })
                && err
                    .to_string()
                    .contains(if hand == frame { link } else { hand }),
            "{err}"
        );
    }
}
