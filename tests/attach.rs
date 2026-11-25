//! Attached objects (MoveIt's attach and detach) and devices that switch robots without
//! re-uploading worlds. GPU parts are skipped when no adapter is available unless
//! `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::{
    AttachedObject, CollisionWeights, Device, Error, IkOptions, IkProblem, Obstacle, PlanOptions,
    PlanProblem, Pose, Robot, SphereOptions, Trajectory, World,
};
use glam::{Mat3, Quat, Vec2, Vec3};

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
        shapes: vec![Obstacle::cuboid(
            Vec3::new(0.0, 0.0, 0.1),
            Vec3::new(0.02, 0.045, 0.03),
        )],
        touch_links: touch_links.iter().map(|s| s.to_string()).collect(),
        spheres: SphereOptions {
            budget: 12,
            ..Default::default()
        },
    }
}

const FINGERS: [&str; 2] = ["panda_leftfinger", "panda_rightfinger"];

#[test]
fn attached_objects_move_with_their_link_and_skip_touch_links() {
    let robot = common::panda().unwrap();
    let held = robot.attach(&held_box(&FINGERS)).unwrap();
    let mut rng = Rng::new(2);
    for _ in 0..20 {
        let q: Vec<f32> = (0..robot.dof())
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect();
        let (object, hand) = (
            held.link_pose(&q, "box").unwrap(),
            robot.link_pose(&q, "panda_hand").unwrap(),
        );
        let rotation = (Mat3::from_quat(object.rotation) - Mat3::from_quat(hand.rotation)).abs();
        assert!(
            (object.position - hand.position).length() < 1e-6
                && rotation.to_cols_array().iter().all(|&v| v < 1e-6)
        );
    }
    let cpu = Device::cpu(&robot).unwrap();
    let pose = robot.default_q();
    // An obstacle between the open fingers misses the bare hand but hits the box.
    let hand = robot.link_pose(pose, "panda_hand").unwrap();
    let between = World::new(vec![Obstacle::sphere(
        hand.position + hand.rotation * Vec3::new(0.0, 0.0, 0.1),
        0.01,
    )]);
    let worlds = cpu.upload(&[between]).unwrap();
    let evaluate = |d: &Device| {
        d.evaluate(&worlds, &[0], pose, &CollisionWeights::NONE)
            .unwrap()
    };
    let (bare, holding) = (evaluate(&cpu), evaluate(&cpu.with_robot(&held).unwrap()));
    assert!(
        bare.world_clearance[0] > 0.0 && holding.world_clearance[0] < 0.0,
        "{bare:?} vs {holding:?}"
    );
    // The box presses into the fingers: a collision unless they are touch links.
    assert!(
        holding.self_clearance[0] >= 0.0,
        "touch links still collide: {holding:?}"
    );
    let gripping = robot.attach(&held_box(&[])).unwrap();
    assert!(
        evaluate(&cpu.with_robot(&gripping).unwrap()).self_clearance[0] < 0.0,
        "the box misses the fingers"
    );
    // Collision gradients reach the joints through the box's frame.
    let holder = cpu.with_robot(&held).unwrap();
    let w = CollisionWeights {
        world: 1000.0,
        self_collision: 0.0,
        margin: 0.03,
        self_margin: 0.0,
    };
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
            (fd - at.gradient[j]).abs()
                < 2e-2 * at.gradient.iter().fold(1.0f32, |m, g| m.max(g.abs())),
            "joint {j}: {fd} vs {}",
            at.gradient[j]
        );
    }
    // Detaching gives back the robot as it was.
    let detached = held.detach("box").unwrap();
    assert_eq!(detached.collision_model(), robot.collision_model());
    assert!(detached.link_pose(pose, "box").is_none());
    assert_eq!(
        evaluate(&cpu.with_robot(&detached).unwrap()).self_clearance,
        bare.self_clearance
    );
}

#[test]
fn attaching_reports_what_is_wrong() {
    let robot = common::panda().unwrap();
    let input = |object: AttachedObject| matches!(robot.attach(&object), Err(Error::Input(_)));
    assert!(
        input(AttachedObject {
            link: "gripper".into(),
            ..held_box(&FINGERS)
        }),
        "unknown link"
    );
    assert!(
        input(AttachedObject {
            name: "panda_link3".into(),
            ..held_box(&FINGERS)
        }),
        "a taken name"
    );
    assert!(
        input(AttachedObject {
            touch_links: vec!["thumb".into()],
            ..held_box(&FINGERS)
        }),
        "unknown touch link"
    );
    let shape = |shape: Obstacle| AttachedObject {
        shapes: vec![shape],
        ..held_box(&FINGERS)
    };
    assert!(
        input(shape(Obstacle::cuboid(
            Vec3::ZERO,
            Vec3::new(f32::NAN, 0.02, 0.02)
        ))),
        "a size that is not finite"
    );
    assert!(
        input(shape(Obstacle::Cylinder {
            center: Vec3::ZERO,
            rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 0.95),
            radius: 0.03,
            half_height: 0.05,
        })),
        "a rotation that is not a unit quaternion"
    );
    let held = robot.attach(&held_box(&FINGERS)).unwrap();
    assert!(matches!(
        held.attach(&AttachedObject {
            name: "lid".into(),
            link: "box".into(),
            ..held_box(&[])
        }),
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
        assert_eq!(
            rayon::current_thread_index(),
            None,
            "attaching made the caller a rayon worker"
        );
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
        .map(|w| IkProblem::new(w, common::grasp_target(&scene[w as usize], &mut rng)))
        .collect();
    let checker = Device::cpu(&held).unwrap();
    let on_checker = checker.upload(&scene).unwrap();
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let holding = d.with_robot(&held).unwrap();
        let ik = batchplan::solve_ik(&holding, &worlds, &goals, &IkOptions::default()).unwrap();
        let problems: Vec<PlanProblem> = ik
            .solved()
            .map(|s| PlanProblem::new(s.problem.world, held.default_q(), s.solution))
            .collect();
        let result =
            batchplan::plan(&holding, &worlds, &problems, &PlanOptions::default()).unwrap();
        assert!(
            result.solved().count() >= 6,
            "{}: planned {} of 8",
            d.name(),
            result.solved().count()
        );
        // Every plan keeps the box clear of the tables, checked independently.
        let n = held.dof();
        let (mut q, mut item_world) = (vec![], vec![]);
        for s in result.solved() {
            let trajectory = Trajectory::try_new(&held, s.solution, 1.0).unwrap();
            let samples = trajectory
                .sample(32.0 / trajectory.knot_interval)
                .unwrap()
                .positions;
            item_world.extend(std::iter::repeat_n(s.problem.world, samples.len() / n));
            q.extend(samples);
        }
        let e = checker
            .evaluate(&on_checker, &item_world, &q, &CollisionWeights::NONE)
            .unwrap();
        let worst = e
            .world_clearance
            .iter()
            .chain(&e.self_clearance)
            .fold(f32::INFINITY, |m, &v| m.min(v));
        assert!(worst > -2e-3, "{}: a plan penetrates by {worst}", d.name());
        // The derived device sees the box exactly as a device made for the held robot does.
        let mine = holding
            .evaluate(&worlds, &item_world, &q, &CollisionWeights::NONE)
            .unwrap();
        let differs = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        assert!(
            differs(&mine.world_clearance, &e.world_clearance) < 1e-4,
            "{}",
            d.name()
        );
        assert!(
            differs(&mine.self_clearance, &e.self_clearance) < 1e-4,
            "{}",
            d.name()
        );
    }
}

/// A single shape is fitted in its own frame, so its spheres move with it: a cylinder held upright
/// and tilted in the hand has the same spheres, tilted.
#[test]
fn a_held_shape_has_the_same_spheres_wherever_it_sits_on_its_link() {
    let robot = common::panda().unwrap();
    let upright = Obstacle::Cylinder {
        center: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        radius: 0.03,
        half_height: 0.08,
    };
    let pose = Pose::new(
        Vec3::new(0.0, 0.02, 0.1),
        Quat::from_rotation_x(1.2) * Quat::from_rotation_z(0.3),
    );
    let spheres = |shape: Obstacle| {
        let held = robot
            .attach(&AttachedObject {
                shapes: vec![shape],
                spheres: SphereOptions::default(),
                ..held_box(&FINGERS)
            })
            .unwrap();
        held.collision_model().spheres["box"].clone()
    };
    let (own, turned) = (spheres(upright.clone()), spheres(upright.placed(pose)));
    assert!(own.len() > 1, "{own:?}");
    assert_eq!(own.len(), turned.len());
    for (o, t) in own.iter().zip(&turned) {
        let moved = pose.position + pose.rotation * Vec3::new(o[0], o[1], o[2]);
        assert!(
            moved.distance(Vec3::new(t[0], t[1], t[2])) < 1e-6 && (o[3] - t[3]).abs() < 1e-7,
            "{o:?} moved to {moved}, but the turned shape has {t:?}"
        );
    }
}

#[test]
fn held_boxes_and_balls_are_covered_without_fitting() {
    let robot = common::panda().unwrap();
    let rotation = Quat::from_rotation_z(0.4) * Quat::from_rotation_x(0.3);
    let (center, half) = (Vec3::new(0.0, 0.01, 0.08), Vec3::new(0.02, 0.035, 0.028));
    // A tolerance the default budget can reach for this box.
    let options = SphereOptions {
        tolerance: 0.008,
        ..Default::default()
    };
    let held = robot
        .attach(&AttachedObject {
            shapes: vec![Obstacle::Cuboid {
                center,
                half_extents: half,
                rotation,
            }],
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
            + rotation
                * Vec3::new(
                    rng.range(-half.x, half.x),
                    rng.range(-half.y, half.y),
                    rng.range(-half.z, half.z),
                );
        assert!(
            spheres
                .iter()
                .any(|s| p.distance(Vec3::new(s[0], s[1], s[2])) <= s[3]),
            "{p} is uncovered"
        );
    }
    for s in spheres {
        for _ in 0..200 {
            let dir = Vec3::new(rng.normal(), rng.normal(), rng.normal()).normalize();
            let local = to_box(Vec3::new(s[0], s[1], s[2]) + dir * s[3]);
            let outside = (local.abs() - half).max(Vec3::ZERO).length();
            assert!(
                outside <= options.tolerance + 1e-5,
                "a sphere reaches {outside} past the box"
            );
        }
    }
    // A ball is its own sphere.
    let ball = robot
        .attach(&AttachedObject {
            shapes: vec![Obstacle::sphere(center, 0.03)],
            ..held_box(&FINGERS)
        })
        .unwrap();
    assert_eq!(
        ball.collision_model().spheres["box"],
        vec![[center.x, center.y, center.z, 0.03]]
    );
}

/// Held shapes under a limit on how far their spheres reach (`SphereOptions::max_overhang`):
/// covered whole where the budget has a cover within it, or a box by its medial cover past it,
/// else fitted and held to it.
#[test]
fn held_shapes_keep_within_a_limit_on_overhang() {
    let robot = common::panda().unwrap();
    let rotation = Quat::from_rotation_z(0.4) * Quat::from_rotation_x(0.3);
    let (center, half) = (Vec3::new(0.0, 0.01, 0.08), Vec3::new(0.02, 0.035, 0.028));
    // The box's spheres in its own frame, held with `options`.
    let boxed = |options: SphereOptions| -> Vec<[f32; 4]> {
        let held = robot
            .attach(&AttachedObject {
                shapes: vec![Obstacle::Cuboid {
                    center,
                    half_extents: half,
                    rotation,
                }],
                spheres: options,
                ..held_box(&FINGERS)
            })
            .unwrap();
        let own = |s: &[f32; 4]| {
            let p = rotation.inverse() * (Vec3::new(s[0], s[1], s[2]) - center);
            [p.x, p.y, p.z, s[3]]
        };
        let spheres = &held.collision_model().spheres["box"];
        assert!(!spheres.is_empty());
        spheres.iter().map(own).collect()
    };
    // How far the farthest reaches past the box, and whether they hold all of it.
    let past = |spheres: &[[f32; 4]]| {
        let reach = |s: &[f32; 4]| {
            let q = Vec3::new(s[0], s[1], s[2]).abs() - half;
            s[3] + q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
        };
        spheres.iter().map(reach).fold(f32::NEG_INFINITY, f32::max)
    };
    let whole = |spheres: &[[f32; 4]]| {
        let mut rng = Rng::new(9);
        (0..4000).all(|_| {
            let p = Vec3::new(
                rng.range(-half.x, half.x),
                rng.range(-half.y, half.y),
                rng.range(-half.z, half.z),
            );
            spheres
                .iter()
                .any(|s| p.distance(Vec3::new(s[0], s[1], s[2])) <= s[3])
        })
    };
    let limit = 0.003;
    let with = |budget: usize, max_overhang: Option<f32>| SphereOptions {
        budget,
        max_overhang,
        ..Default::default()
    };
    // 512 spheres have a cover within the limit, tighter than the tolerance asks for: the whole
    // box, each sphere within it.
    let free = boxed(with(512, None));
    assert!(
        whole(&free) && past(&free) > limit + 5e-4,
        "{}",
        past(&free)
    );
    let covered = boxed(with(512, Some(limit)));
    assert!(whole(&covered), "a cover within the limit leaves none out");
    assert!(past(&covered) <= limit + 1e-6, "{}", past(&covered));
    // 64 have none within the limit: the box's medial cover holds all of it within the limit
    // however many spheres that takes, as without it within the tolerance.
    let free = boxed(with(64, None));
    assert!(
        whole(&free) && past(&free) > limit + 5e-4,
        "{}",
        past(&free)
    );
    let medial = boxed(with(64, Some(limit)));
    assert!(past(&medial) <= limit + 1e-6, "{}", past(&medial));
    assert!(
        whole(&medial),
        "a box's cover within the limit leaves none out"
    );
    // Held to nothing past it, every sphere is inside the box.
    let inside = boxed(with(64, Some(0.0)));
    assert!(past(&inside) <= 1e-6, "{}", past(&inside));

    // A capsule's spheres lie along its axis, a rod's too, and a plate's in a layer: each within
    // the limit of its shape, as its cover's own account gives it, and all of it covered.
    let (radius, half_length) = (0.02, 0.06);
    let capsule = |center, rotation| Obstacle::Capsule {
        center,
        rotation,
        radius: 0.02,
        half_length: 0.06,
    };
    let limit = 0.001;
    let free = held_round(
        &robot,
        capsule,
        with(64, None),
        /* with_a_ball */ false,
    );
    let held = held_round(
        &robot,
        capsule,
        with(64, Some(limit)),
        /* with_a_ball */ false,
    );
    let reach = |spheres: &[[f32; 4]]| spheres.iter().map(|s| s[3] - radius).fold(0.0, f32::max);
    assert!(reach(&free) > limit + 5e-4, "{}", reach(&free));
    assert!(reach(&held) <= limit + 1e-6 && held.len() > free.len());
    assert!(
        held.iter()
            .all(|s| s[0].hypot(s[1]) < 1e-6 && s[2].abs() <= half_length + 1e-6)
    );
    // Two spheres along the axis cannot keep it: fitted, they are held to it off the axis too.
    let few = held_round(
        &robot,
        capsule,
        with(2, Some(limit)),
        /* with_a_ball */ false,
    );
    let past = |s: &[f32; 4]| {
        let along = s[2].clamp(-half_length, half_length);
        Vec3::new(s[0], s[1], s[2] - along).length() + s[3] - radius
    };
    let farthest = few.iter().map(past).fold(f32::NEG_INFINITY, f32::max);
    assert!(!few.is_empty() && farthest <= limit + 1e-6, "{farthest}");
    // Held to nothing past it, no spheres along the axis can keep it: fitted, they are inside it.
    let inside = held_round(
        &robot,
        capsule,
        with(64, Some(0.0)),
        /* with_a_ball */ false,
    );
    let farthest = inside.iter().map(past).fold(f32::NEG_INFINITY, f32::max);
    assert!(!inside.is_empty() && farthest <= 1e-6, "{farthest}");
    let plate = |center, rotation| Obstacle::Cylinder {
        center,
        rotation,
        radius: 0.0695,
        half_height: 0.009,
    };
    let limit = 0.004;
    let held = held_round(
        &robot,
        plate,
        with(512, Some(limit)),
        /* with_a_ball */ false,
    );
    let past = cylinder_overhang(&held, 0.0695, 0.009);
    assert!(past <= limit + 1e-5, "{past} m past the plate");
    // A ball is its own sphere at any limit.
    let ball = robot
        .attach(&AttachedObject {
            shapes: vec![Obstacle::sphere(center, 0.03)],
            spheres: with(64, Some(0.0)),
            ..held_box(&FINGERS)
        })
        .unwrap();
    assert_eq!(
        ball.collision_model().spheres["box"],
        vec![[center.x, center.y, center.z, 0.03]]
    );
    // A sheet too thin for a center inside it has no sphere within it, and says so.
    let sheet = AttachedObject {
        shapes: vec![Obstacle::cuboid(center, Vec3::new(0.03, 0.03, 2e-5))],
        spheres: with(64, Some(0.0)),
        ..held_box(&FINGERS)
    };
    let refused = robot.attach(&sheet).map(|_| ());
    assert!(
        matches!(&refused, Err(Error::Input(m)) if m.contains("'box': no sphere fits within `max_overhang`")),
        "{refused:?}"
    );
    // Shapes of no thickness have no inside for a sphere to be held in: a sheet and a disc.
    // Their two sides are closed and wind around nothing.
    let flat = [
        Obstacle::cuboid(center, Vec3::new(0.03, 0.03, 0.0)),
        Obstacle::Cylinder {
            center,
            rotation,
            radius: 0.05,
            half_height: 0.0,
        },
    ];
    for shape in flat {
        for budget in [16, 64] {
            let held = AttachedObject {
                shapes: vec![shape.clone()],
                spheres: with(budget, Some(0.0)),
                ..held_box(&FINGERS)
            };
            let refused = robot.attach(&held).map(|_| ());
            assert!(
                matches!(&refused, Err(Error::Input(m)) if m.starts_with("'box': no sphere fits within `max_overhang`")),
                "{shape:?}: {refused:?}"
            );
            // Without a limit they are covered, as before, and with one to reach past them
            // within, by spheres centered on them and no larger than it.
            let free = AttachedObject {
                spheres: with(budget, None),
                ..held.clone()
            };
            robot.attach(&free).unwrap();
            let limit = 0.002;
            let thin = AttachedObject {
                spheres: with(budget, Some(limit)),
                ..held
            };
            let attached = robot.attach(&thin).unwrap();
            let spheres = &attached.collision_model().spheres["box"];
            assert!(!spheres.is_empty());
            for s in spheres {
                let off = (rotation.inverse() * (Vec3::new(s[0], s[1], s[2]) - center)).z;
                let rotated = matches!(shape, Obstacle::Cylinder { .. });
                let off = if rotated { off } else { s[2] - center.z };
                assert!(off.abs() < 1e-5 && s[3] <= limit + 1e-6, "{s:?}");
            }
        }
    }
    let bad = AttachedObject {
        spheres: with(64, Some(-1.0)),
        ..held_box(&FINGERS)
    };
    let refused = robot.attach(&bad).map(|_| ());
    assert!(
        matches!(&refused, Err(Error::Input(m)) if m.contains("max_overhang must be finite")),
        "{refused:?}"
    );
}

/// A round shape held with `options`, alone or with a small ball beside it, and the shape's
/// spheres in its own frame (its axis along z).
fn held_round(
    robot: &Robot,
    shape: fn(Vec3, Quat) -> Obstacle,
    options: SphereOptions,
    with_a_ball: bool,
) -> Vec<[f32; 4]> {
    let rotation = Quat::from_rotation_z(0.4) * Quat::from_rotation_x(0.3);
    let center = Vec3::new(0.0, 0.01, 0.12);
    let mut shapes = vec![shape(center, rotation)];
    if with_a_ball {
        shapes.push(Obstacle::sphere(center + Vec3::X * 0.2, 0.01));
    }
    let held = robot
        .attach(&AttachedObject {
            shapes,
            spheres: options,
            ..held_box(&FINGERS)
        })
        .unwrap();
    let own = |s: &[f32; 4]| {
        let p = rotation.inverse() * (Vec3::new(s[0], s[1], s[2]) - center);
        [p.x, p.y, p.z, s[3]]
    };
    let spheres = &held.collision_model().spheres["box"];
    let its = spheres.len() - usize::from(with_a_ball);
    spheres[..its].iter().map(own).collect()
}

/// That `spheres` hold every point of a cylinder of `radius` and `half_height` about z, and how
/// far the farthest reaches past its side and its faces.
fn cylinder_overhang(spheres: &[[f32; 4]], radius: f32, half_height: f32) -> f32 {
    let mut rng = Rng::new(11);
    for _ in 0..6000 {
        let (angle, from_axis) = (
            rng.range(0.0, std::f32::consts::TAU),
            radius * rng.uniform().sqrt(),
        );
        let p = Vec3::new(
            from_axis * angle.cos(),
            from_axis * angle.sin(),
            rng.range(-half_height, half_height),
        );
        let inside = |s: &&[f32; 4]| p.distance(Vec3::new(s[0], s[1], s[2])) <= s[3];
        assert!(spheres.iter().any(|s| inside(&s)), "{p} is uncovered");
    }
    let past = |s: &[f32; 4]| {
        let side = Vec3::new(s[0], s[1], 0.0).length() + s[3] - radius;
        side.max(s[2].abs() + s[3] - half_height)
    };
    spheres.iter().map(past).fold(0.0, f32::max)
}

#[test]
fn held_discs_rods_and_capsules_are_covered_without_fitting() {
    let robot = common::panda().unwrap();
    // A plate 14 cm across and 18 mm thick: one layer of spheres at its middle height, within
    // the tolerance. 256 cover it directly in a layer of rings of one width, 145 spheres, and its
    // tiles' layer of rings of the widths and sectors that take fewest is taken, fewer: a bigger
    // budget never costs more. A fit of as many spheres is not within the tolerance.
    let plate = |center, rotation| Obstacle::Cylinder {
        center,
        rotation,
        radius: 0.0695,
        half_height: 0.009,
    };
    let fine = SphereOptions {
        budget: 256,
        ..Default::default()
    };
    let spheres = held_round(&robot, plate, fine, /* with_a_ball */ false);
    assert!(spheres.len() < 145, "{} spheres", spheres.len());
    assert!(spheres.iter().all(|s| s[2].abs() < 1e-6), "one layer");
    let past = cylinder_overhang(&spheres, 0.0695, 0.009);
    assert!(past <= fine.tolerance + 1e-5, "{past} m past the plate");
    // Covered with a ball beside it, as the two are placed, its spheres are where they were:
    // turned as the plate is.
    let beside = held_round(&robot, plate, fine, /* with_a_ball */ true);
    assert_eq!(beside.len(), spheres.len());
    for (a, b) in beside.iter().zip(&spheres) {
        let apart = Vec3::new(a[0], a[1], a[2]).distance(Vec3::new(b[0], b[1], b[2]));
        assert!(
            apart < 1e-5 && (a[3] - b[3]).abs() < 1e-7,
            "{a:?} against {b:?}"
        );
    }
    // The default 64 have no direct layer within the tolerance, and do not cap the plate's
    // spheres: it takes the same layer.
    let coarse = held_round(
        &robot,
        plate,
        SphereOptions::default(),
        /* with_a_ball */ false,
    );
    assert_eq!(coarse, spheres);
    // A rod 8 mm across and 14 cm long: a line of spheres along its axis.
    let rod = |center, rotation| Obstacle::Cylinder {
        center,
        rotation,
        radius: 0.004,
        half_height: 0.07,
    };
    let spheres = held_round(
        &robot,
        rod,
        SphereOptions::default(),
        /* with_a_ball */ false,
    );
    assert!(
        spheres.iter().all(|s| s[0].hypot(s[1]) < 1e-6),
        "on its axis"
    );
    let past = cylinder_overhang(&spheres, 0.004, 0.07);
    assert!(
        past <= 0.005 + 1e-5 && spheres.len() < 20,
        "{past} m, {} spheres",
        spheres.len()
    );
    // A can 6.6 cm across and 12 cm tall is neither: its fit spends fewer spheres on its inside.
    let can = |center, rotation| Obstacle::Cylinder {
        center,
        rotation,
        radius: 0.033,
        half_height: 0.06,
    };
    let spheres = held_round(
        &robot,
        can,
        SphereOptions::default(),
        /* with_a_ball */ false,
    );
    let off_axis = spheres.iter().any(|s| s[0].hypot(s[1]) > 1e-3);
    assert!(
        off_axis && spheres.iter().any(|s| s[2].abs() > 1e-3),
        "fitted"
    );
    // A capsule: spheres along its axis from end to end, the fewest within the tolerance.
    let capsule = |center, rotation| Obstacle::Capsule {
        center,
        rotation,
        radius: 0.02,
        half_length: 0.05,
    };
    let spheres = held_round(
        &robot,
        capsule,
        SphereOptions::default(),
        /* with_a_ball */ false,
    );
    assert!(
        spheres.iter().all(|s| s[0].hypot(s[1]) < 1e-6),
        "on its axis"
    );
    let ends = spheres
        .iter()
        .map(|s| s[2])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!((ends - 0.05).abs() < 1e-6, "to its ends: {ends}");
    let mut rng = Rng::new(5);
    for _ in 0..4000 {
        // A point of the capsule: within its radius of a point of its axis.
        let on_axis = Vec3::Z * rng.range(-0.05, 0.05);
        let dir = Vec3::new(rng.normal(), rng.normal(), rng.normal()).normalize();
        let p = on_axis + dir * 0.02 * rng.uniform().cbrt();
        let inside = |s: &&[f32; 4]| p.distance(Vec3::new(s[0], s[1], s[2])) <= s[3];
        assert!(spheres.iter().any(|s| inside(&s)), "{p} is uncovered");
    }
    let past = spheres.iter().map(|s| s[3] - 0.02).fold(0.0, f32::max);
    assert!(past <= 0.005 + 1e-6, "{past} m past the capsule");
    let fewer = spheres.len() - 1;
    let gap = 0.05 / (fewer - 1) as f32;
    assert!(
        0.02f32.hypot(gap) - 0.02 > 0.005,
        "one sphere fewer would reach past it"
    );
}

/// A held object's spheres reach its tolerance whatever its budget: the tolerance is the knob.
/// A bowl 11.2 cm across and 5.1 cm deep, which 64 fitted spheres left 9.1 mm past its side,
/// takes as many as the default 5 mm does, the same at budgets of 1, 256 and 1,024, and fewer than
/// the 129 a fit from the grid's candidates alone took to reach 5 mm; so do a box and a capsule
/// that 64 cannot cover directly within it. A box that 256 cover directly with a grid of cells
/// takes its medial cover of fewer spheres at every budget.
#[test]
fn a_held_objects_spheres_reach_its_tolerance_whatever_its_budget() {
    let robot = common::panda().unwrap();
    let options = |budget: usize| SphereOptions {
        budget,
        ..Default::default()
    };
    // How far spheres reach past the side or the faces of a cylinder about z, as
    // `examples/held_spheres.rs` measures it.
    let past = |spheres: &[[f32; 4]], radius: f32, half_height: f32| {
        let past = |s: &[f32; 4]| {
            let side = Vec3::new(s[0], s[1], 0.0).length() + s[3] - radius;
            side.max(s[2].abs() + s[3] - half_height)
        };
        spheres.iter().map(past).fold(f32::NEG_INFINITY, f32::max)
    };
    let bowl = |center, rotation| Obstacle::Cylinder {
        center,
        rotation,
        radius: 0.05603,
        half_height: 0.02537,
    };
    let default = SphereOptions::default();
    let spheres = held_round(&robot, bowl, default, /* with_a_ball */ false);
    let reach = past(&spheres, 0.05603, 0.02537);
    assert!(
        reach <= default.tolerance && default.budget < spheres.len() && spheres.len() < 129,
        "{} spheres reach {reach} m past the bowl",
        spheres.len()
    );
    for budget in [1, 256, 1024] {
        let other = held_round(&robot, bowl, options(budget), /* with_a_ball */ false);
        assert_eq!(other, spheres, "at a budget of {budget}");
    }
    let attached = |shape: Obstacle, spheres: SphereOptions| {
        let held = robot
            .attach(&AttachedObject {
                shapes: vec![shape],
                spheres,
                ..held_box(&FINGERS)
            })
            .unwrap();
        held.collision_model().spheres["box"].clone()
    };
    let half = Vec3::new(0.02, 0.035, 0.028);
    let boxed = attached(Obstacle::cuboid(Vec3::ZERO, half), default);
    let reach = |s: &[f32; 4]| {
        let q = Vec3::new(s[0], s[1], s[2]).abs() - half;
        s[3] + q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
    };
    let reach = boxed.iter().map(reach).fold(f32::NEG_INFINITY, f32::max);
    assert!(
        reach <= default.tolerance,
        "{} spheres reach {reach} m past the box",
        boxed.len()
    );
    // A capsule 1 cm across and 1.21 m long: a line of spheres along its axis, which a fit within
    // the tolerance does not spend fewer on.
    let (radius, half_length) = (0.005, 0.6);
    let rod = attached(
        Obstacle::Capsule {
            center: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            radius,
            half_length,
        },
        default,
    );
    let reach = |s: &[f32; 4]| {
        let along = s[2].clamp(-half_length, half_length);
        Vec3::new(s[0], s[1], s[2] - along).length() + s[3] - radius
    };
    let reach = rod.iter().map(reach).fold(f32::NEG_INFINITY, f32::max);
    let on_axis = rod.iter().all(|s| s[0].hypot(s[1]) < 1e-6);
    assert!(
        reach <= default.tolerance && on_axis,
        "{} spheres reach {reach} m past the capsule",
        rod.len()
    );
    // A box 5 by 5 by 7 cm, which 256 cover directly with a grid of cells within it, takes its
    // medial cover of fewer spheres at any budget: a bigger budget never costs more.
    let block = Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.025, 0.025, 0.035));
    let medial = attached(block.clone(), default);
    for budget in [1, 256, 1024] {
        assert_eq!(
            attached(block.clone(), options(budget)),
            medial,
            "at {budget}"
        );
    }
}

/// Boxes a held object's medial cover takes: the blocks `examples/handover/` passes, a 6 cm cube,
/// a cereal box and a box thinner than either tolerance.
const MEDIAL_BOXES: [(&str, Vec3); 7] = [
    ("block 1", Vec3::new(0.12, 0.0125, 0.025)),
    ("block 2", Vec3::new(0.125, 0.016, 0.028)),
    ("block 3", Vec3::new(0.13, 0.015, 0.027)),
    ("block 4", Vec3::new(0.14, 0.0175, 0.03)),
    ("a 6 cm cube", Vec3::new(0.03, 0.03, 0.03)),
    ("a cereal box", Vec3::new(0.095, 0.035, 0.135)),
    ("a thin box", Vec3::new(0.07, 0.05, 0.0005)),
];

/// A held object's spheres hold all of it and reach no farther past it than the tolerance, on the
/// shapes that test that hardest: a bowl and two plates at 5 and 1 mm, held by rings along their
/// medial structure; a slab 2 mm thick, where a sphere crosses its faces steeply; a sheet 0.08 mm
/// thick, too thin for a center inside it to reach its faces; a bar 12 mm square and 60 cm long
/// at a budget of 8; a disc 1 cm thick at 0.4 mm; a mug, a cylinder with a box through its side; a
/// wheel's rim, which a layer of rings covers; and the boxes a medial cover takes, at 5 and 1 mm.
/// Points of each surface at random, its corners, its edges and rims, and its rims at the middles
/// of 24 sectors each lie inside a sphere, and every sphere's radius and its center's signed
/// distance to the shapes sum to no more than the tolerance.
#[test]
fn a_held_objects_spheres_hold_all_of_it() {
    use std::f32::consts::TAU;

    let robot = common::panda().unwrap();
    let options = |budget: usize, tolerance: f32| SphereOptions {
        budget,
        tolerance,
        ..Default::default()
    };
    let cylinder = |radius, half_height| Obstacle::cylinder(Vec3::ZERO, radius, half_height);
    let cuboid = |half: Vec3| Obstacle::cuboid(Vec3::ZERO, half);
    let mug = vec![
        cylinder(0.04, 0.05),
        Obstacle::cuboid(Vec3::new(0.045, 0.0, 0.0), Vec3::new(0.02, 0.006, 0.03)),
    ];
    let cases = [
        (
            "a bowl",
            vec![cylinder(0.05603, 0.02537)],
            options(64, 0.005),
        ),
        (
            "a bowl at 1 mm",
            vec![cylinder(0.05603, 0.02537)],
            options(64, 0.001),
        ),
        (
            "a slab",
            vec![cuboid(Vec3::new(0.1, 0.1, 0.001))],
            options(64, 0.005),
        ),
        (
            "a sheet",
            vec![cuboid(Vec3::new(0.025, 0.025, 4e-5))],
            options(64, 0.005),
        ),
        (
            "a bar",
            vec![cuboid(Vec3::new(0.3, 0.006, 0.006))],
            options(8, 0.005),
        ),
        ("a disc", vec![cylinder(0.03, 0.005)], options(64, 0.0004)),
        ("a mug", mug, options(64, 0.005)),
        ("a plate", vec![cylinder(0.0695, 0.009)], options(64, 0.005)),
        (
            "a plate at 1 mm",
            vec![cylinder(0.0695, 0.009)],
            options(64, 0.001),
        ),
        (
            "goal 5's plate",
            vec![cylinder(0.06860558, 0.009563851)],
            options(64, 0.005),
        ),
        (
            "goal 5's plate at 1 mm",
            vec![cylinder(0.06860558, 0.009563851)],
            options(64, 0.001),
        ),
        (
            "a wheel's rim",
            vec![cylinder(0.1, 0.01)],
            options(64, 0.005),
        ),
    ]
    .into_iter()
    .chain(MEDIAL_BOXES.into_iter().flat_map(|(name, half)| {
        [0.005, 0.001].map(|tolerance| (name, vec![cuboid(half)], options(64, tolerance)))
    }));
    let mut rng = Rng::new(17);
    for (name, shapes, spheres) in cases {
        let held = robot
            .attach(&AttachedObject {
                shapes: shapes.clone(),
                spheres,
                ..held_box(&FINGERS)
            })
            .unwrap();
        let fitted = held.collision_model().spheres["box"].clone();
        // Points of each shape's surface outside every other shape.
        let mut points = vec![];
        for shape in &shapes {
            let mut on = vec![];
            match *shape {
                Obstacle::Cylinder {
                    center,
                    radius,
                    half_height: h,
                    ..
                } => {
                    let rim = |a: f32| center + Vec3::new(radius * a.cos(), radius * a.sin(), 0.0);
                    for _ in 0..60_000 {
                        let a = rng.range(0.0, TAU);
                        let z = if rng.uniform() < 0.5 { -h } else { h };
                        on.push(rim(a) + Vec3::Z * rng.range(-h, h));
                        on.push(center + (rim(a) - center) * rng.uniform().sqrt() + Vec3::Z * z);
                        on.push(rim(a) + Vec3::Z * z);
                    }
                    // The middles of the sectors of a mesh of 24 sides, where it lies deepest.
                    for k in 0..24 {
                        let a = TAU * (k as f32 + 0.5) / 24.0;
                        for step in 0..=40 {
                            on.push(rim(a) + Vec3::Z * h * (step as f32 / 20.0 - 1.0));
                        }
                    }
                }
                Obstacle::Cuboid {
                    center,
                    half_extents: half,
                    ..
                } => {
                    for corner in 0..8 {
                        let sign = |bit: i32| if corner & bit == 0 { -1.0 } else { 1.0 };
                        on.push(center + half * Vec3::new(sign(1), sign(2), sign(4)));
                    }
                    for _ in 0..100_000 {
                        let mut p = Vec3::new(
                            rng.range(-half.x, half.x),
                            rng.range(-half.y, half.y),
                            rng.range(-half.z, half.z),
                        );
                        let axis = (rng.uniform() * 3.0) as usize % 3;
                        p[axis] = if rng.uniform() < 0.5 {
                            -half[axis]
                        } else {
                            half[axis]
                        };
                        on.push(center + p);
                        let other = (axis + 1 + (rng.uniform() * 2.0) as usize % 2) % 3;
                        p[other] = if rng.uniform() < 0.5 {
                            -half[other]
                        } else {
                            half[other]
                        };
                        on.push(center + p);
                    }
                }
                _ => unreachable!(),
            }
            let others: Vec<&Obstacle> = shapes.iter().filter(|o| *o != shape).collect();
            let outside_others = |p: &Vec3| others.iter().all(|o| o.distance(*p).value >= 0.0);
            points.extend(on.into_iter().filter(outside_others));
        }
        // How far the farthest point lies outside every sphere.
        let outside = |p: &Vec3| {
            let to = |s: &[f32; 4]| p.distance(Vec3::new(s[0], s[1], s[2])) - s[3];
            fitted.iter().map(to).fold(f32::INFINITY, f32::min)
        };
        let gap = points.iter().map(outside).fold(0.0, f32::max);
        // How far past the shapes the farthest sphere reaches: no more than past the one its
        // center lies deepest in.
        let past = |s: &[f32; 4]| {
            let c = Vec3::new(s[0], s[1], s[2]);
            let depth = shapes.iter().map(|o| o.distance(c).value);
            s[3] + depth.fold(f32::INFINITY, f32::min)
        };
        let reach = fitted.iter().map(past).fold(f32::NEG_INFINITY, f32::max);
        assert!(
            gap <= 1e-6 && reach <= spheres.tolerance + 1e-6,
            "{name}: {} spheres leave {gap} m of it outside and reach {reach} m past it",
            fitted.len()
        );
    }
}

/// A held shape's rotation a thousandth off a unit quaternion, which is accepted, turns it as the
/// unit one does: a long board and a long rod, each fitted in its own frame, keep their spheres
/// where they were. Together, and the board with a ball, they are fitted as they are placed, where
/// the rounding of a rotation can tip which spheres the fit picks: their spheres reach no farther
/// past the shapes as the unit rotation places them than the tolerance.
#[test]
fn a_held_shape_turned_by_a_rotation_not_quite_unit_is_not_stretched() {
    let robot = common::panda().unwrap();
    let turn = Quat::from_rotation_y(1.6) * Quat::from_rotation_x(0.4);
    let shapes = |rotation: Quat| {
        let board = Obstacle::Cuboid {
            center: Vec3::new(0.0, 0.0, 0.1),
            half_extents: Vec3::new(0.6, 0.03, 0.01),
            rotation,
        };
        let rod = Obstacle::Cylinder {
            center: Vec3::new(0.0, 0.05, 0.1),
            rotation,
            radius: 0.02,
            half_height: 0.6,
        };
        let ball = Obstacle::Sphere {
            center: Vec3::new(0.0, 0.0, 0.2),
            radius: 0.03,
        };
        [
            vec![board.clone()],
            vec![rod.clone()],
            vec![board.clone(), rod],
            vec![board, ball],
        ]
    };
    let spheres = |shapes: Vec<Obstacle>| {
        let held = robot
            .attach(&AttachedObject {
                shapes,
                spheres: SphereOptions::default(),
                ..held_box(&FINGERS)
            })
            .unwrap();
        held.collision_model().spheres["box"].clone()
    };
    // How far the farthest of `spheres` reaches past the board and the rod or the ball, as
    // `turn` places them.
    let past = |spheres: &[[f32; 4]], ball: bool| {
        let reach = |s: &[f32; 4]| {
            let c = Vec3::new(s[0], s[1], s[2]);
            let local = |center: Vec3| turn.inverse() * (c - center);
            let q = local(Vec3::new(0.0, 0.0, 0.1)).abs() - Vec3::new(0.6, 0.03, 0.01);
            let board = q.max(Vec3::ZERO).length() + q.max_element().min(0.0);
            let other = if ball {
                c.distance(Vec3::new(0.0, 0.0, 0.2)) - 0.03
            } else {
                let p = local(Vec3::new(0.0, 0.05, 0.1));
                let d = Vec2::new(p.truncate().length() - 0.02, p.z.abs() - 0.6);
                d.max(Vec2::ZERO).length() + d.max_element().min(0.0)
            };
            s[3] + board.min(other)
        };
        spheres.iter().map(reach).fold(f32::NEG_INFINITY, f32::max)
    };
    let tolerance = SphereOptions::default().tolerance;
    for norm in [0.9991f32, 1.0009] {
        let sets = shapes(turn).into_iter().zip(shapes(turn * norm));
        for (k, (unit, scaled)) in sets.enumerate() {
            let (unit, scaled) = (spheres(unit), spheres(scaled));
            let reach = unit
                .iter()
                .map(|s| Vec3::new(s[0], s[1], s[2] - 0.1).length())
                .fold(0.0, f32::max);
            assert!(reach > 0.5, "the spheres reach {reach} from the hand");
            if k >= 2 {
                let ball = k == 3;
                let (unit, scaled) = (past(&unit, ball), past(&scaled, ball));
                assert!(
                    unit <= tolerance + 1e-5 && scaled <= tolerance + 1e-5,
                    "norm {norm}: {scaled} m past them, against {unit} m"
                );
                continue;
            }
            assert_eq!(unit.len(), scaled.len());
            for (u, s) in unit.iter().zip(&scaled) {
                let apart = u.iter().zip(s).map(|(a, b)| (a - b).abs());
                assert!(
                    apart.fold(0.0, f32::max) < 2e-5,
                    "norm {norm}: {u:?} became {s:?}"
                );
            }
        }
    }
}

/// What a hand holds is among its spheres (`Robot::hand_spheres`): a ball fixed to the hand adds
/// its spheres to the hand's, each placed by the hand's pose where the robot holding it has it.
#[test]
fn a_hands_spheres_take_in_what_it_holds() {
    let robot = common::panda().unwrap();
    let ball = AttachedObject {
        name: "ball".into(),
        link: "panda_hand".into(),
        shapes: vec![Obstacle::sphere(Vec3::new(0.0, 0.03, 0.12), 0.02)],
        touch_links: FINGERS.iter().map(|s| s.to_string()).collect(),
        spheres: SphereOptions::default(),
    };
    let holding = robot.attach(&ball).unwrap();
    let q = holding.default_q();
    // Attached objects are links after every one of the robot's, so their spheres come last.
    let everywhere = holding.spheres(q);
    let held = &everywhere[robot.spheres(q).len()..];
    assert!(!held.is_empty(), "the ball has spheres");
    let hand = holding.hand_spheres(0);
    assert_eq!(hand.len(), robot.hand_spheres(0).len() + held.len());
    let frame = holding.ee_pose(q, 0);
    for &[x, y, z, radius] in held {
        let at = |sphere: &batchplan::HandSphere| {
            let center = frame.position + frame.rotation * sphere.center;
            center.distance(Vec3::new(x, y, z)) < 1e-5 && (sphere.radius - radius).abs() < 1e-6
        };
        assert!(
            hand.iter().any(at),
            "the ball's sphere at {x}, {y}, {z} is no hand sphere"
        );
    }
}

/// Checks dense deterministic samples of a solid cuboid (corners, edges, faces, and interior)
/// for strict `f32` enclosure (gap 0) and per-sphere exact overhang via `Obstacle::distance`.
fn check_box_cover_samples(
    name: &str,
    shape: &Obstacle,
    half: Vec3,
    pose: Pose,
    spheres: &[[f32; 4]],
    reach_limit: f32,
    rng: &mut Rng,
) {
    assert!(!spheres.is_empty(), "{name}: empty sphere cover");
    let to_world = |local: Vec3| pose.position + pose.rotation * local;
    let mut points = Vec::new();

    // All 8 corners.
    for corner in 0..8 {
        let sign = |bit: i32| if corner & bit == 0 { -1.0 } else { 1.0 };
        points.push(to_world(half * Vec3::new(sign(1), sign(2), sign(4))));
    }
    // Dense uniform samples along all 12 edges.
    for axis in 0..3 {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        for su in [-1.0, 1.0] {
            for sv in [-1.0, 1.0] {
                for step in 0..=200 {
                    let t = 2.0 * step as f32 / 200.0 - 1.0;
                    let mut p = Vec3::ZERO;
                    p[axis] = t * half[axis];
                    p[u] = su * half[u];
                    p[v] = sv * half[v];
                    points.push(to_world(p));
                }
            }
        }
    }
    // Face centers and box center (catching any missing mid-plane sheet).
    points.push(to_world(Vec3::ZERO));
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            let mut p = Vec3::ZERO;
            p[axis] = sign * half[axis];
            points.push(to_world(p));
        }
    }
    // Fresh deterministic samples across faces, edges, and solid interior.
    for _ in 0..40_000 {
        let mut p = Vec3::new(
            rng.range(-half.x, half.x),
            rng.range(-half.y, half.y),
            rng.range(-half.z, half.z),
        );
        points.push(to_world(p));
        let axis = (rng.uniform() * 3.0) as usize % 3;
        p[axis] = if rng.uniform() < 0.5 {
            -half[axis]
        } else {
            half[axis]
        };
        points.push(to_world(p));
        let other = (axis + 1 + (rng.uniform() * 2.0) as usize % 2) % 3;
        p[other] = if rng.uniform() < 0.5 {
            -half[other]
        } else {
            half[other]
        };
        points.push(to_world(p));
    }

    let outside = |p: &Vec3| {
        spheres
            .iter()
            .map(|s| p.distance(Vec3::new(s[0], s[1], s[2])) - s[3])
            .fold(f32::INFINITY, f32::min)
    };
    let gap = points.iter().map(outside).fold(0.0f32, f32::max);
    let reach = spheres
        .iter()
        .map(|s| s[3] + shape.distance(Vec3::new(s[0], s[1], s[2])).value)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        gap <= 0.0,
        "{name} ({} spheres): leaves {gap:e} m outside every sphere",
        spheres.len()
    );
    assert!(
        reach <= reach_limit,
        "{name} ({} spheres): reaches {reach} m past box (limit {reach_limit})",
        spheres.len()
    );
}

/// Held cuboids that `box_cells` cannot cover within the budget use the closed-form medial cover:
/// strict solid enclosure (gap 0) and exact overhang `<= tolerance` on the four handover blocks,
/// the 6 cm cube, the cereal box, and a thin box with `c < tolerance` at both 5 mm and 1 mm,
/// axis-aligned and rotated, using far fewer spheres than the fitted cover.
#[test]
fn held_cuboids_enclose_the_solid_box_within_tolerance_at_5mm_and_1mm() {
    let robot = common::panda().unwrap();
    let cases: [(&str, Vec3, usize, usize); 7] = [
        (
            "block 1 (240x25x50 mm)",
            Vec3::new(0.120, 0.0125, 0.025),
            95,
            924,
        ),
        (
            "block 2 (250x32x56 mm)",
            Vec3::new(0.125, 0.016, 0.028),
            109,
            1020,
        ),
        (
            "block 3 (260x30x54 mm)",
            Vec3::new(0.130, 0.015, 0.027),
            112,
            1047,
        ),
        (
            "block 4 (280x35x60 mm)",
            Vec3::new(0.140, 0.0175, 0.030),
            121,
            1125,
        ),
        ("6 cm cube", Vec3::new(0.030, 0.030, 0.030), 45, 484),
        (
            "cereal box (190x65x280 mm)",
            Vec3::new(0.095, 0.0325, 0.140),
            238,
            1973,
        ),
        (
            "thin box (140x100x1 mm)",
            Vec3::new(0.070, 0.050, 0.0005),
            195,
            2863,
        ),
    ];
    let mut rng = Rng::new(23);
    for tolerance in [0.005f32, 0.001] {
        for &(name, half, max_at_5mm, max_at_1mm) in &cases {
            let shape = Obstacle::cuboid(Vec3::ZERO, half);
            let held = robot
                .attach(&AttachedObject {
                    shapes: vec![shape.clone()],
                    spheres: SphereOptions {
                        tolerance,
                        ..Default::default()
                    },
                    ..held_box(&FINGERS)
                })
                .unwrap();
            let spheres = &held.collision_model().spheres["box"];
            let label = format!("{name} @ {:.0} mm", tolerance * 1e3);
            check_box_cover_samples(
                &label,
                &shape,
                half,
                Pose::IDENTITY,
                spheres,
                tolerance,
                &mut rng,
            );
            let max_spheres = if tolerance == 0.005 {
                max_at_5mm
            } else {
                max_at_1mm
            };
            assert!(
                spheres.len() <= max_spheres,
                "{label}: got {} spheres, expected <= {max_spheres}",
                spheres.len()
            );
        }
    }
    // Rotated and translated box: still strictly enclosed, with overhang within `f32` rotation
    // rounding (`8 * f32::EPSILON * scale`).
    let pose = Pose::new(
        Vec3::new(0.02, -0.01, 0.09),
        (Quat::from_rotation_x(0.7) * Quat::from_rotation_y(1.1) * Quat::from_rotation_z(0.5))
            .normalize(),
    );
    let half = Vec3::new(0.095, 0.0325, 0.140);
    let shape = Obstacle::Cuboid {
        center: pose.position,
        half_extents: half,
        rotation: pose.rotation,
    };
    let held = robot
        .attach(&AttachedObject {
            shapes: vec![shape.clone()],
            spheres: SphereOptions::default(),
            ..held_box(&FINGERS)
        })
        .unwrap();
    let spheres = &held.collision_model().spheres["box"];
    let scale = pose.position.abs().max_element() + half.length() + 0.005;
    check_box_cover_samples(
        "rotated cereal box @ 5 mm",
        &shape,
        half,
        pose,
        spheres,
        0.005 + 8.0 * f32::EPSILON * scale,
        &mut rng,
    );
}

/// Sweeps thin-box, hybrid-slab, slender-rod, cube, and general regime boundaries (`c / tau`
/// across `[0, 5]`) through `Robot::attach` to verify strict enclosure and overhang everywhere.
#[test]
fn held_cuboids_cover_every_thickness_regime_and_boundary() {
    let robot = common::panda().unwrap();
    let tolerance = 0.005f32;
    let ratios = [
        0.0f32, 0.01, 0.2, 0.5, 0.9, 1.0, 1.2, 1.35, 1.366, 1.37, 1.5, 1.8, 2.1, 2.22, 2.23, 2.5,
        3.0, 4.0, 5.0,
    ];
    let mut rng = Rng::new(31);
    for &ratio in &ratios {
        let c = ratio * tolerance;
        let shapes = [
            ("wide", Vec3::new(0.09, 0.06, c)),
            ("narrow", Vec3::new(0.11, (c + 0.004).max(0.014), c)),
            ("cube-like", Vec3::new(c.max(0.01), c.max(0.01), c)),
        ];
        for (kind, half) in shapes {
            let shape = Obstacle::cuboid(Vec3::ZERO, half);
            let held = robot
                .attach(&AttachedObject {
                    shapes: vec![shape.clone()],
                    spheres: SphereOptions {
                        budget: 1,
                        tolerance,
                        ..Default::default()
                    },
                    ..held_box(&FINGERS)
                })
                .unwrap();
            let spheres = &held.collision_model().spheres["box"];
            let label = format!("{kind} c/tau={ratio} ({half:?})");
            check_box_cover_samples(
                &label,
                &shape,
                half,
                Pose::IDENTITY,
                spheres,
                tolerance,
                &mut rng,
            );
        }
    }
}

/// Points about 1 mm apart over the surface of a capsule, a box or a cylinder about z at the origin:
/// rings along a side, a disc's rows and columns, a hemisphere's rings, a face's rows and columns.
fn surface_points(shape: &Obstacle) -> Vec<Vec3> {
    use std::f32::consts::{FRAC_PI_2, TAU};
    let step = 0.001;
    let parts = |length: f32| (length / step).ceil().max(1.0) as usize;
    let mut points = vec![];
    let mut ring = |r: f32, z: f32| {
        let around = parts(TAU * r);
        let at = |i: usize| TAU * i as f32 / around as f32;
        points.extend((0..around).map(|i| Vec3::new(r * at(i).cos(), r * at(i).sin(), z)));
    };
    let mut side = |r: f32, half: f32| {
        let up = parts(2.0 * half);
        for k in 0..=up {
            ring(r, -half + 2.0 * half * k as f32 / up as f32);
        }
    };
    match *shape {
        Obstacle::Capsule {
            radius,
            half_length,
            ..
        } => {
            side(radius, half_length);
            let rings = parts(FRAC_PI_2 * radius);
            for k in 1..=rings {
                let polar = FRAC_PI_2 * k as f32 / rings as f32;
                for end in [-1.0, 1.0] {
                    ring(
                        radius * polar.cos(),
                        end * (half_length + radius * polar.sin()),
                    );
                }
            }
        }
        Obstacle::Cylinder {
            radius,
            half_height,
            ..
        } => {
            side(radius, half_height);
            let n = parts(radius) as i32;
            for i in -n..=n {
                for j in -n..=n {
                    let (x, y) = (i as f32 * step, j as f32 * step);
                    if x.hypot(y) <= radius {
                        points.extend([-1.0, 1.0].map(|end| Vec3::new(x, y, end * half_height)));
                    }
                }
            }
        }
        Obstacle::Cuboid { half_extents, .. } => {
            for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
                let (na, nb) = (parts(2.0 * half_extents[a]), parts(2.0 * half_extents[b]));
                for i in 0..=na {
                    for j in 0..=nb {
                        for end in [-1.0, 1.0] {
                            let mut p = Vec3::ZERO;
                            p[a] = half_extents[a] * (2.0 * i as f32 / na as f32 - 1.0);
                            p[b] = half_extents[b] * (2.0 * j as f32 / nb as f32 - 1.0);
                            p[c] = end * half_extents[c];
                            points.push(p);
                        }
                    }
                }
            }
        }
        _ => unreachable!("a capsule, a cylinder or a box"),
    }
    points
}

/// Under a limit of zero, which no tiles' spheres keep, a held capsule, box and cylinder are fitted
/// inside them, and leave no more of their surface outside than the fit within the budget did
/// before (`b6ffaaa`): of points about 1 mm apart over it, at least as many within 1 mm of a
/// sphere, and none farther outside than its farthest. Those are this test's on `b6ffaaa`: 35
/// spheres in the capsule, 26.05% within 1 mm, 5.12 mm at most; 64 in the box, 29.65%, 7.45 mm; 64
/// in the bowl, 15.42%, 9.76 mm.
#[test]
fn held_shapes_at_a_limit_of_zero_leave_no_more_outside_than_before() {
    let robot = common::panda().unwrap();
    let capsule: fn(Vec3, Quat) -> Obstacle = |center, rotation| Obstacle::Capsule {
        center,
        rotation,
        radius: 0.02,
        half_length: 0.05,
    };
    let boxed: fn(Vec3, Quat) -> Obstacle = |center, rotation| Obstacle::Cuboid {
        center,
        half_extents: Vec3::new(0.02, 0.035, 0.028),
        rotation,
    };
    let bowl: fn(Vec3, Quat) -> Obstacle = |center, rotation| Obstacle::Cylinder {
        center,
        rotation,
        radius: 0.05603,
        half_height: 0.02537,
    };
    let inside = SphereOptions {
        max_overhang: Some(0.0),
        ..Default::default()
    };
    let cases = [
        ("capsule", capsule, (0.2605, 0.00513)),
        ("box", boxed, (0.2965, 0.00745)),
        ("bowl", bowl, (0.1542, 0.00976)),
    ];
    for (name, shape, (share, farthest)) in cases {
        let spheres = held_round(&robot, shape, inside, /* with_a_ball */ false);
        let points = surface_points(&shape(Vec3::ZERO, Quat::IDENTITY));
        let outside = |p: &Vec3| {
            let to = |s: &[f32; 4]| p.distance(Vec3::new(s[0], s[1], s[2])) - s[3];
            spheres.iter().map(to).fold(f32::INFINITY, f32::min)
        };
        let gaps: Vec<f32> = points.iter().map(outside).collect();
        let near = gaps.iter().filter(|&&gap| gap <= 0.001).count() as f32 / gaps.len() as f32;
        let most = gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            near >= share && most <= farthest,
            "{name}: {near} within 1 mm (before, {share}), {most} m outside at most \
             (before, {farthest})"
        );
    }
}
