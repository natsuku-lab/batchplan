//! Attached objects (MoveIt's attach and detach) and devices that switch robots without
//! re-uploading worlds. GPU parts are skipped when no adapter is available unless
//! `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::*;
use glam::{Quat, Vec3};

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot).unwrap()];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

/// A box held between the Panda's open fingers, wide enough to press into them.
fn held_box(touch_links: &[&str]) -> AttachedObject {
    AttachedObject {
        name: "box".into(),
        link: "panda_hand".into(),
        shapes: vec![Obstacle::Cuboid {
            center: Vec3::new(0.0, 0.0, 0.1),
            half_extents: Vec3::new(0.02, 0.045, 0.03),
            rotation: Quat::IDENTITY,
        }],
        touch_links: touch_links.iter().map(|s| s.to_string()).collect(),
        spheres: SphereOptions { budget: 12, ..Default::default() },
    }
}

const FINGERS: [&str; 2] = ["panda_leftfinger", "panda_rightfinger"];

#[test]
fn attached_objects_move_with_their_link_and_skip_touch_links() {
    let robot = common::panda().unwrap();
    let held = robot.attach(&held_box(&FINGERS)).unwrap();
    let mut rng = Rng::new(2);
    for _ in 0..20 {
        let q: Vec<f32> = (0..robot.dof()).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect();
        let (object, hand) = (held.link_pose(&q, "box").unwrap(), robot.link_pose(&q, "panda_hand").unwrap());
        let rotation = (glam::Mat3::from_quat(object.rotation) - glam::Mat3::from_quat(hand.rotation)).abs();
        assert!(
            (object.position - hand.position).length() < 1e-6 && rotation.to_cols_array().iter().all(|&v| v < 1e-6)
        );
    }
    let cpu = Device::cpu(&robot).unwrap();
    let pose = robot.default_q();
    // An obstacle between the open fingers misses the bare hand but hits the box.
    let hand = robot.link_pose(pose, "panda_hand").unwrap();
    let between = World {
        obstacles: vec![Obstacle::Sphere {
            center: hand.position + hand.rotation * Vec3::new(0.0, 0.0, 0.1),
            radius: 0.01,
        }],
    };
    let worlds = cpu.upload(&[between]).unwrap();
    let evaluate = |d: &Device| d.evaluate(&worlds, &[0], pose, &CollisionWeights::NONE).unwrap();
    let (bare, holding) = (evaluate(&cpu), evaluate(&cpu.with_robot(&held).unwrap()));
    assert!(bare.world_clearance[0] > 0.0 && holding.world_clearance[0] < 0.0, "{bare:?} vs {holding:?}");
    // The box presses into the fingers: a collision unless they are touch links.
    assert!(holding.self_clearance[0] >= 0.0, "touch links still collide: {holding:?}");
    let gripping = robot.attach(&held_box(&[])).unwrap();
    assert!(evaluate(&cpu.with_robot(&gripping).unwrap()).self_clearance[0] < 0.0, "the box misses the fingers");
    // Collision gradients reach the joints through the box's frame.
    let holder = cpu.with_robot(&held).unwrap();
    let w = CollisionWeights { world: 1000.0, self_collision: 0.0, margin: 0.03, self_margin: 0.0 };
    let cost = |q: &[f32]| holder.evaluate(&worlds, &[0], q, &w).unwrap();
    let at = cost(pose);
    assert!(at.cost[0] > 1.0, "the box should press into the obstacle");
    let h = 1e-3;
    for j in 0..robot.dof() {
        let (mut up, mut down) = (pose.to_vec(), pose.to_vec());
        up[j] += h;
        down[j] -= h;
        let fd = (cost(&up).cost[0] - cost(&down).cost[0]) / (2.0 * h);
        assert!(
            (fd - at.grad[j]).abs() < 2e-2 * at.grad.iter().fold(1.0f32, |m, g| m.max(g.abs())),
            "joint {j}: {fd} vs {}",
            at.grad[j]
        );
    }
    // Detaching gives back the robot as it was.
    let detached = held.detach("box").unwrap();
    assert_eq!(detached.collision_model(), robot.collision_model());
    assert!(detached.link_pose(pose, "box").is_none());
    assert_eq!(evaluate(&cpu.with_robot(&detached).unwrap()).self_clearance, bare.self_clearance);
}

#[test]
fn attaching_reports_what_is_wrong() {
    let robot = common::panda().unwrap();
    let input = |object: AttachedObject| matches!(robot.attach(&object), Err(Error::Input(_)));
    assert!(input(AttachedObject { link: "gripper".into(), ..held_box(&FINGERS) }), "unknown link");
    assert!(input(AttachedObject { name: "panda_link3".into(), ..held_box(&FINGERS) }), "a taken name");
    assert!(input(AttachedObject { touch_links: vec!["thumb".into()], ..held_box(&FINGERS) }), "unknown touch link");
    let held = robot.attach(&held_box(&FINGERS)).unwrap();
    assert!(matches!(
        held.attach(&AttachedObject { name: "lid".into(), link: "box".into(), ..held_box(&[]) }),
        Err(Error::Input(_))
    ));
    assert!(matches!(robot.detach("box"), Err(Error::Input(_))));
}

#[test]
fn attaching_leaves_the_calling_thread_out_of_rayon() {
    // Outside a pool, the spheres are fitted on the calling thread; registering it with a rayon
    // pool would leave every parallel loop it starts later running on that one thread.
    let robot = common::panda().unwrap();
    std::thread::spawn(move || {
        robot.attach(&held_box(&FINGERS)).unwrap();
        assert_eq!(rayon::current_thread_index(), None, "attaching made the caller a rayon worker");
    })
    .join()
    .unwrap();
}

#[test]
fn devices_for_an_attached_robot_plan_in_worlds_already_uploaded() {
    let robot = common::panda().unwrap();
    let held = robot.attach(&held_box(&FINGERS)).unwrap();
    let mut rng = Rng::new(5);
    let scene: Vec<World> = (0..8).map(|_| common::tabletop(&mut rng)).collect();
    let goals: Vec<IkProblem> = (0..8)
        .map(|w| IkProblem {
            world: w,
            targets: vec![Some(common::grasp_target(&scene[w as usize], &mut rng))],
            seed: None,
        })
        .collect();
    let checker = Device::cpu(&held).unwrap();
    let on_checker = checker.upload(&scene).unwrap();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let holding = d.with_robot(&held).unwrap();
        let ik = solve_ik(&holding, &worlds, &goals, &IkOptions::default()).unwrap();
        let problems: Vec<PlanProblem> = ik
            .solved()
            .map(|s| PlanProblem {
                world: s.problem.world,
                start: held.default_q().to_vec(),
                goal: s.solution.to_vec(),
                start_motion: None,
                hold: vec![],
            })
            .collect();
        let result = plan(&holding, &worlds, &problems, &PlanOptions::default()).unwrap();
        assert!(result.solved().count() >= 6, "{}: planned {} of 8", d.name(), result.solved().count());
        // Every plan keeps the box clear of the tables, checked independently.
        let n = held.dof();
        let (mut q, mut item_world) = (vec![], vec![]);
        for s in result.solved() {
            let trajectory = Trajectory::new(&held, s.solution, 1.0).unwrap();
            let samples = trajectory.sample(32.0 / trajectory.knot_interval).unwrap().positions;
            item_world.extend(std::iter::repeat_n(s.problem.world, samples.len() / n));
            q.extend(samples);
        }
        let e = checker.evaluate(&on_checker, &item_world, &q, &CollisionWeights::NONE).unwrap();
        let worst = e.world_clearance.iter().chain(&e.self_clearance).fold(f32::INFINITY, |m, &v| m.min(v));
        assert!(worst > -2e-3, "{}: a plan penetrates by {worst}", d.name());
        // The derived device sees the box exactly as a device made for the held robot does.
        let mine = holding.evaluate(&worlds, &item_world, &q, &CollisionWeights::NONE).unwrap();
        let differs = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(differs(&mine.world_clearance, &e.world_clearance) < 1e-4, "{}", d.name());
        assert!(differs(&mine.self_clearance, &e.self_clearance) < 1e-4, "{}", d.name());
    }
}

#[test]
fn held_boxes_and_balls_are_covered_without_fitting() {
    let robot = common::panda().unwrap();
    let rotation = Quat::from_rotation_z(0.4) * Quat::from_rotation_x(0.3);
    let (center, half) = (Vec3::new(0.0, 0.01, 0.08), Vec3::new(0.02, 0.035, 0.028));
    // A tolerance the default budget can reach for this box.
    let options = SphereOptions { tolerance: 0.008, ..Default::default() };
    let held = robot
        .attach(&AttachedObject {
            shapes: vec![Obstacle::Cuboid { center, half_extents: half, rotation }],
            spheres: options,
            ..held_box(&FINGERS)
        })
        .unwrap();
    let spheres = &held.collision_model().spheres["box"];
    assert!(spheres.len() <= options.budget, "{} spheres", spheres.len());
    // Every point of the box is inside a sphere, and no sphere reaches past it by more than the
    // tolerance.
    let mut rng = Rng::new(9);
    let to_box = |p: Vec3| rotation.inverse() * (p - center);
    for _ in 0..4000 {
        let p = center
            + rotation * Vec3::new(rng.range(-half.x, half.x), rng.range(-half.y, half.y), rng.range(-half.z, half.z));
        assert!(spheres.iter().any(|s| p.distance(Vec3::new(s[0], s[1], s[2])) <= s[3]), "{p} is uncovered");
    }
    for s in spheres {
        for _ in 0..200 {
            let dir = Vec3::new(rng.normal(), rng.normal(), rng.normal()).normalize();
            let local = to_box(Vec3::new(s[0], s[1], s[2]) + dir * s[3]);
            let outside = (local.abs() - half).max(Vec3::ZERO).length();
            assert!(outside <= options.tolerance + 1e-5, "a sphere reaches {outside} past the box");
        }
    }
    // A ball is its own sphere.
    let ball = robot
        .attach(&AttachedObject { shapes: vec![Obstacle::Sphere { center, radius: 0.03 }], ..held_box(&FINGERS) })
        .unwrap();
    assert_eq!(ball.collision_model().spheres["box"], vec![[center.x, center.y, center.z, 0.03]]);
}
