//! Robots loaded from descriptions: mimic joints, fitted collision spheres, self-collision pairs,
//! collision-model files, SRDF import, package resolution and load errors. GPU parts are skipped
//! when no adapter is available unless `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use batchplan::rng::Rng;
use batchplan::{
    CollisionModel, CollisionWeights, Device, Error, JointLimits, Obstacle, PlanOptions,
    PlanProblem, Robot, RobotOptions, SphereOptions, Trajectory, World,
};
use glam::{DMat3, DVec3, Mat3, Quat, Vec3};
use mesh_loader::Loader;

const UR5E: &str = "ur5e/ur_description/urdf/ur5e.urdf";
const SO101: &str = "so101/so101_new_calib.urdf";
const GRIPPER: &str = "robotiq_2f85/robotiq_description/urdf/robotiq_2f_85.urdf";

fn load(path: &str) -> Robot {
    Robot::load(common::asset(path), &RobotOptions::default()).unwrap()
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

fn random_q(robot: &Robot, rng: &mut Rng) -> Vec<f32> {
    (0..robot.dof())
        .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
        .collect()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("batchplan-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// World poses of every link of a URDF, composed independently of the library in f64. Mimic
/// joints take `multiplier * leader + offset`; other moving joints take `values[name]`.
fn reference_poses(path: &Path, values: &HashMap<String, f64>) -> HashMap<String, (DMat3, DVec3)> {
    let urdf = urdf_rs::read_file(path).unwrap();
    let value = |j: &urdf_rs::Joint| match &j.mimic {
        Some(m) => m.multiplier.unwrap_or(1.0) * values[&m.joint] + m.offset.unwrap_or(0.0),
        None => values.get(&j.name).copied().unwrap_or(0.0),
    };
    let children: Vec<&str> = urdf.joints.iter().map(|j| j.child.link.as_str()).collect();
    let root = urdf
        .links
        .iter()
        .find(|l| !children.contains(&l.name.as_str()))
        .unwrap();
    let mut poses = HashMap::from([(root.name.clone(), (DMat3::IDENTITY, DVec3::ZERO))]);
    while poses.len() < urdf.links.len() {
        for j in &urdf.joints {
            let Some(&(pr, pp)) = poses.get(&j.parent.link) else {
                continue;
            };
            let [r, p, y] = j.origin.rpy.0;
            let rot = pr
                * DMat3::from_rotation_z(y)
                * DMat3::from_rotation_y(p)
                * DMat3::from_rotation_x(r);
            let pos = pp + pr * DVec3::from_array(j.origin.xyz.0);
            let axis = DVec3::from_array(j.axis.xyz.0).normalize();
            let pose = match j.joint_type {
                urdf_rs::JointType::Revolute | urdf_rs::JointType::Continuous => {
                    (rot * DMat3::from_axis_angle(axis, value(j)), pos)
                }
                urdf_rs::JointType::Prismatic => (rot, pos + rot * axis * value(j)),
                _ => (rot, pos),
            };
            poses.insert(j.child.link.clone(), pose);
        }
    }
    poses
}

#[test]
fn mimic_joints_follow_their_leader() {
    let robot = load(GRIPPER);
    assert_eq!(robot.joint_names(), ["robotiq_85_left_knuckle_joint"]);
    assert_eq!((robot.lower(), robot.upper()), (&[0.0][..], &[0.8][..]));
    let urdf = urdf_rs::read_file(common::asset(GRIPPER)).unwrap();
    let mut rng = Rng::new(3);
    for _ in 0..50 {
        let q = random_q(&robot, &mut rng);
        let expected = reference_poses(
            Path::new(&common::asset(GRIPPER)),
            &HashMap::from([(robot.joint_names()[0].clone(), q[0] as f64)]),
        );
        for link in &urdf.links {
            let pose = robot.link_pose(&q, &link.name).unwrap();
            let (r, p) = expected[&link.name];
            assert!(
                (pose.position.as_dvec3() - p).length() < 1e-5,
                "{} position at q = {q:?}",
                link.name
            );
            let rot_err = (DMat3::from_quat(pose.rotation.as_dquat()) - r)
                .to_cols_array()
                .iter()
                .fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(
                rot_err < 1e-5,
                "{} rotation off by {rot_err} at q = {q:?}",
                link.name
            );
        }
    }
}

#[test]
fn mimic_limits_and_velocities_constrain_the_leader() {
    let dir = scratch("mimic");
    let urdf = r#"<robot name="mimic">
      <link name="base"/><link name="a"/><link name="b"/><link name="c"/>
      <joint name="leader" type="revolute"><parent link="base"/><child link="a"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="2" effort="1"/></joint>
      <joint name="double" type="revolute"><parent link="base"/><child link="b"/><axis xyz="0 1 0"/>
        <limit lower="-0.5" upper="1.5" velocity="1" effort="1"/><mimic joint="leader" multiplier="2" offset="0.5"/></joint>
      <joint name="flipped" type="prismatic"><parent link="a"/><child link="c"/><axis xyz="1 0 0"/>
        <limit lower="-0.6" upper="0.9" velocity="5" effort="1"/><mimic joint="leader" multiplier="-1"/></joint>
    </robot>"#;
    std::fs::write(dir.join("mimic.urdf"), urdf).unwrap();
    let robot = Robot::load(dir.join("mimic.urdf"), &RobotOptions::default()).unwrap();
    // double = 2 q + 0.5 in [-0.5, 1.5] -> q in [-0.5, 0.5]; flipped = -q in [-0.6, 0.9] -> q in
    // [-0.9, 0.6].
    assert_eq!((robot.lower(), robot.upper()), (&[-0.5][..], &[0.5][..]));
    // double moves twice as fast as the leader, so the leader may move at most 0.5.
    assert_eq!(robot.max_velocity(), [0.5]);
    let q = [0.3];
    let flipped = robot.link_pose(&q, "c").unwrap().position;
    assert!(
        (flipped - Vec3::new(-0.3 * 0.3f32.cos(), -0.3 * 0.3f32.sin(), 0.0)).length() < 1e-6,
        "{flipped}"
    );
    let double = robot.link_pose(&q, "b").unwrap().rotation;
    assert!(double.angle_between(Quat::from_rotation_y(1.1)) < 1e-6);
}

/// A sphere where the 2F-85's fingers meet as they close.
fn between_the_fingers(robot: &Robot) -> World {
    let tip = |side: &str, q: f32| {
        robot
            .link_pose(&[q], &format!("robotiq_85_{side}_finger_tip_link"))
            .unwrap()
            .position
    };
    let center = 0.5 * (tip("left", 0.4) + tip("right", 0.4));
    World::new(vec![Obstacle::sphere(
        center + Vec3::new(0.0, 0.0, 0.02),
        0.015,
    )])
}

#[test]
fn mimic_collision_gradients_match_finite_differences() {
    let robot = load(GRIPPER);
    let cpu = Device::cpu(&robot).unwrap();
    let worlds = cpu.upload(&[between_the_fingers(&robot)]).unwrap();
    let w = CollisionWeights {
        world: 1000.0,
        self_collision: 1000.0,
        margin: 0.02,
        self_margin: 0.01,
    };
    let cost = |q: f32| cpu.evaluate(&worlds, &[0], &[q], &w).unwrap().cost[0];
    let (mut checked, mut worst) = (0, 0.0f32);
    for k in 0..60 {
        let q = 0.05 + 0.7 * k as f32 / 59.0;
        let e = cpu.evaluate(&worlds, &[0], &[q], &w).unwrap();
        if e.cost[0] < 1e-3 {
            continue;
        }
        let fd = |h: f32| (cost(q + h) - cost(q - h)) / (2.0 * h);
        let err = (fd(1e-3) - e.gradient[0])
            .abs()
            .min((fd(2e-4) - e.gradient[0]).abs())
            / e.gradient[0].abs().max(1.0);
        worst = worst.max(err);
        checked += 1;
    }
    assert!(
        checked > 20,
        "only {checked} configurations touch the obstacle"
    );
    assert!(worst < 2e-2, "gradient off by {worst} of its size");
}

#[test]
fn mimic_gripper_evaluates_the_same_on_every_device() {
    let robot = load(GRIPPER);
    let scene = [between_the_fingers(&robot)];
    let q: Vec<f32> = (0..2000).map(|k| 0.8 * k as f32 / 1999.0).collect();
    let item_world = vec![0; q.len()];
    let w = CollisionWeights {
        world: 1000.0,
        self_collision: 1000.0,
        margin: 0.02,
        self_margin: 0.01,
    };
    let devices = devices(&robot);
    let reference = devices[0]
        .evaluate(&devices[0].upload(&scene).unwrap(), &item_world, &q, &w)
        .unwrap();
    assert!(
        reference.cost.iter().filter(|&&c| c > 0.0).count() > 500,
        "too few configurations carry cost"
    );
    for d in &devices[1..] {
        let e = d
            .evaluate(&d.upload(&scene).unwrap(), &item_world, &q, &w)
            .unwrap();
        for (i, qi) in q.iter().enumerate() {
            assert!(
                (e.world_clearance[i] - reference.world_clearance[i]).abs() < 1e-4,
                "{}: clearance at q = {qi}",
                d.name(),
            );
            assert!(
                (e.self_clearance[i] - reference.self_clearance[i]).abs() < 1e-4,
                "{}: self clearance at q = {qi}",
                d.name(),
            );
            let scale = reference.gradient[i].abs().max(1.0);
            assert!(
                (e.gradient[i] - reference.gradient[i]).abs() / scale < 1e-3,
                "{}: gradient at q = {qi}",
                d.name()
            );
        }
    }
}

/// Collision geometry of every link of a URDF as triangles in the link frame, loaded
/// independently of the library. `package://` paths resolve below `package_root`'s parent.
fn link_triangles(path: &str, package_root: &Path) -> HashMap<String, Vec<[Vec3; 3]>> {
    let urdf = urdf_rs::read_file(common::asset(path)).unwrap();
    let dir = Path::new(&common::asset(path))
        .parent()
        .unwrap()
        .to_path_buf();
    let mut out = HashMap::new();
    for link in &urdf.links {
        let mut triangles = vec![];
        for c in &link.collision {
            let urdf_rs::Geometry::Mesh { filename, scale } = &c.geometry else {
                panic!("test robots use meshes")
            };
            let file = match filename.strip_prefix("package://") {
                Some(rest) => package_root.parent().unwrap().join(rest),
                None => dir.join(filename),
            };
            let scale = scale.map_or(Vec3::ONE, |s| Vec3::from_array(s.0.map(|v| v as f32)));
            let [r, p, y] = c.origin.rpy.0.map(|v| v as f32);
            let rot =
                Mat3::from_rotation_z(y) * Mat3::from_rotation_y(p) * Mat3::from_rotation_x(r);
            let trans = Vec3::from_array(c.origin.xyz.0.map(|v| v as f32));
            for m in Loader::default().load(&file).unwrap().meshes {
                let v: Vec<Vec3> = m
                    .vertices
                    .iter()
                    .map(|&v| rot * (Vec3::from(v) * scale) + trans)
                    .collect();
                triangles.extend(m.faces.iter().map(|f| f.map(|i| v[i as usize])));
            }
        }
        if !triangles.is_empty() {
            out.insert(link.name.clone(), triangles);
        }
    }
    out
}

#[test]
fn fitted_spheres_contain_the_link_surfaces() {
    let ur_root = PathBuf::from(common::asset("ur5e/ur_description"));
    for (path, root) in [(UR5E, ur_root.clone()), (SO101, ur_root)] {
        let robot = load(path);
        let model = robot.collision_model();
        let total: usize = model.spheres.values().map(Vec::len).sum();
        assert!(
            total <= SphereOptions::default().budget,
            "{path}: {total} spheres"
        );
        let mut rng = Rng::new(99);
        let mut worst = 0.0f32;
        for (link, triangles) in link_triangles(path, &root) {
            let spheres = &model.spheres[&link];
            for _ in 0..3000 {
                let [a, b, c] =
                    triangles[(rng.uniform() * triangles.len() as f32) as usize % triangles.len()];
                let (r1, r2) = (rng.uniform().sqrt(), rng.uniform());
                let p = a * (1.0 - r1) + b * (r1 * (1.0 - r2)) + c * (r1 * r2);
                let outside = spheres
                    .iter()
                    .map(|s| (p - Vec3::new(s[0], s[1], s[2])).length() - s[3])
                    .fold(f32::INFINITY, f32::min);
                worst = worst.max(outside);
            }
        }
        eprintln!(
            "{path}: {total} spheres; worst surface point outside them by {:.2} mm",
            worst * 1e3
        );
        assert!(
            worst < 2e-3,
            "{path}: surface reaches {worst} m outside the spheres"
        );
    }
}

#[test]
fn fitting_is_deterministic() {
    assert_eq!(load(UR5E).collision_model(), load(UR5E).collision_model());
}

#[test]
fn collision_models_round_trip_through_files() {
    let fitted = load(SO101);
    let file = scratch("model").join("so101_collision.json");
    fitted.collision_model().save(&file).unwrap();
    let options = RobotOptions {
        collision_model: Some(CollisionModel::load(&file).unwrap()),
        ..Default::default()
    };
    let reloaded = Robot::load(common::asset(SO101), &options).unwrap();
    assert_eq!(reloaded.collision_model(), fitted.collision_model());
    let mut rng = Rng::new(4);
    let q: Vec<f32> = (0..200).flat_map(|_| random_q(&fitted, &mut rng)).collect();
    let item_world = vec![0; 200];
    let evaluate = |r: &Robot| {
        let cpu = Device::cpu(r).unwrap();
        cpu.evaluate(
            &cpu.upload(&[World::default()]).unwrap(),
            &item_world,
            &q,
            &CollisionWeights::NONE,
        )
        .unwrap()
    };
    assert_eq!(
        evaluate(&fitted).self_clearance,
        evaluate(&reloaded).self_clearance
    );
}

#[test]
fn srdf_disabled_pairs_are_not_checked() {
    use std::fmt::Write;

    let pairs = ["panda_link5", "panda_link6", "panda_link7", "panda_hand"];
    let mut srdf = String::from("<robot name=\"panda\">\n");
    for link in pairs {
        writeln!(
            srdf,
            "  <disable_collisions link1=\"panda_link0\" link2=\"{link}\" reason=\"Never\"/>"
        )
        .unwrap();
    }
    srdf += "</robot>\n";
    let plain = common::panda().unwrap();
    let options = RobotOptions {
        srdf: Some(srdf),
        ..common::panda_options()
    };
    let with_srdf = Robot::load(common::asset("franka/franka_panda.urdf"), &options).unwrap();
    assert!(
        with_srdf.collision_model().self_collision_ignore["panda_link0"]
            .contains(&"panda_hand".to_string())
    );
    let mut rng = Rng::new(8);
    let q: Vec<f32> = (0..5000).flat_map(|_| random_q(&plain, &mut rng)).collect();
    let item_world = vec![0; 5000];
    let clearance = |r: &Robot| {
        let cpu = Device::cpu(r).unwrap();
        cpu.evaluate(
            &cpu.upload(&[World::default()]).unwrap(),
            &item_world,
            &q,
            &CollisionWeights::NONE,
        )
        .unwrap()
        .self_clearance
    };
    let (a, b) = (clearance(&plain), clearance(&with_srdf));
    assert!(
        a.iter().zip(&b).all(|(x, y)| y >= x),
        "disabling pairs can only increase self clearance"
    );
    let changed = a.iter().zip(&b).filter(|(x, y)| y > x).count();
    assert!(
        changed > 0,
        "no configuration had a disabled pair as its closest"
    );
}

#[test]
fn packages_resolve_from_package_dirs_or_by_searching_upward() {
    // The UR5e URDF sits inside its package, so `package://ur_description/...` resolves upward.
    let ur5e = load(UR5E);
    assert_eq!(ur5e.dof(), 6);
    // The default IK frame is the link with the most actuated joints above it.
    let q = [0.3, -1.0, 1.2, -0.4, 0.9, 0.2];
    assert_eq!(
        ur5e.ee_pose(&q, 0),
        ur5e.link_pose(&q, "wrist_3_link").unwrap()
    );

    // A copy outside its package needs the package's parent in package_dirs.
    let dir = scratch("packages");
    std::fs::copy(common::asset(GRIPPER), dir.join("gripper.urdf")).unwrap();
    let err = Robot::load(dir.join("gripper.urdf"), &RobotOptions::default()).unwrap_err();
    assert!(
        format!("{err:#}").contains("package 'robotiq_description'"),
        "{err:#}"
    );
    let options = RobotOptions {
        package_dirs: vec![common::asset("robotiq_2f85").into()],
        ..Default::default()
    };
    assert_eq!(
        Robot::load(dir.join("gripper.urdf"), &options)
            .unwrap()
            .dof(),
        1
    );
}

#[test]
fn a_urdf_given_as_text_loads_the_robot_its_file_does() {
    let spheres = SphereOptions {
        self_collision_samples: 500,
        ..Default::default()
    };
    let options = RobotOptions {
        package_dirs: vec![common::asset("ur5e").into()],
        spheres,
        ..Default::default()
    };
    let file = Robot::load(common::asset(UR5E), &options).unwrap();
    let text = Robot::from_urdf(
        &std::fs::read_to_string(common::asset(UR5E)).unwrap(),
        &options,
    )
    .unwrap();
    assert_eq!(text.joint_names(), file.joint_names());
    assert_eq!(
        (text.lower(), text.upper(), text.max_velocity()),
        (file.lower(), file.upper(), file.max_velocity())
    );
    // Spheres fitted to the same meshes: the packages resolved to the same files.
    assert!(!file.collision_model().spheres.is_empty());
    assert_eq!(text.collision_model(), file.collision_model());
    let mut rng = Rng::new(3);
    for _ in 0..20 {
        let q = random_q(&file, &mut rng);
        assert_eq!(text.ee_pose(&q, 0), file.ee_pose(&q, 0));
    }

    // Text has no directory to search upward from or to resolve relative paths against.
    let ur5e = std::fs::read_to_string(common::asset(UR5E)).unwrap();
    let err = Robot::from_urdf(&ur5e, &RobotOptions::default()).unwrap_err();
    assert!(
        matches!(err, Error::Input(_)) && format!("{err}").contains("package 'ur_description'"),
        "{err}"
    );
    let so101 = std::fs::read_to_string(common::asset(SO101)).unwrap();
    let err = Robot::from_urdf(&so101, &RobotOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("relative"), "{err}");
}

#[test]
fn joint_limits_replace_the_descriptions_motion_limits() {
    let panda = common::asset("franka/franka_panda.urdf");
    let plain = Robot::load(&panda, &common::panda_options()).unwrap();
    let given = |entries: Vec<(&str, JointLimits)>| -> HashMap<String, JointLimits> {
        entries
            .into_iter()
            .map(|(n, l)| (n.to_string(), l))
            .collect()
    };
    let limits = given(vec![
        (
            "panda_joint1",
            JointLimits {
                velocity: Some(0.5),
                ..Default::default()
            },
        ),
        (
            "panda_joint2",
            JointLimits {
                acceleration: Some(1.0),
                jerk: Some(4.0),
                ..Default::default()
            },
        ),
    ]);
    let scaled = RobotOptions {
        joint_limits: limits.clone(),
        velocity_scale: 0.5,
        ..common::panda_options()
    };
    let robot = Robot::load(&panda, &scaled).unwrap();
    // Replaced, then scaled like every other limit.
    assert_eq!(robot.max_velocity()[0], 0.25);
    assert_eq!(
        (robot.max_acceleration()[1], robot.max_jerk()[1]),
        (1.0, 4.0)
    );
    for j in 1..7 {
        assert_eq!(
            robot.max_velocity()[j],
            plain.max_velocity()[j] * 0.5,
            "joint {j}"
        );
    }
    let others = (0..7).filter(|&j| j != 1);
    for j in others {
        assert_eq!(
            robot.max_acceleration()[j],
            plain.max_acceleration()[j],
            "joint {j}"
        );
        assert_eq!(robot.max_jerk()[j], plain.max_jerk()[j], "joint {j}");
    }

    // A move of the slowed joint takes at least as long as its limit allows.
    let slow = Robot::load(
        &panda,
        &RobotOptions {
            joint_limits: limits,
            ..common::panda_options()
        },
    )
    .unwrap();
    let (start, mut goal) = (plain.default_q().to_vec(), plain.default_q().to_vec());
    goal[0] += 1.0;
    // Three equal control points at each end: the path starts and ends at rest.
    let line: Vec<f32> = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
        .into_iter()
        .flat_map(|s| start.iter().zip(&goal).map(move |(a, b)| a + (b - a) * s))
        .collect();
    let duration = |r: &Robot| Trajectory::try_new(r, &line, 1.0).unwrap().duration();
    assert!(duration(&plain) < 1.0 / 0.5, "{}", duration(&plain));
    assert!(duration(&slow) >= 1.0 / 0.5, "{}", duration(&slow));

    // A mimic joint's limits bound its leader through the mimic's multiplier.
    let gripper = |joint_limits: HashMap<String, JointLimits>| {
        Robot::load(
            common::asset(GRIPPER),
            &RobotOptions {
                joint_limits,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(gripper(HashMap::new()).max_velocity(), [0.5]);
    let mimic = given(vec![(
        "robotiq_85_right_inner_knuckle_joint",
        JointLimits {
            position: Some([-0.5, 0.0]),
            velocity: Some(0.2),
            ..Default::default()
        },
    )]);
    let limited = gripper(mimic);
    assert_eq!(
        (limited.max_velocity(), limited.upper()),
        ([0.2].as_slice(), [0.5].as_slice())
    );

    let err = |joint_limits| {
        format!(
            "{:#}",
            Robot::load(
                &panda,
                &RobotOptions {
                    joint_limits,
                    ..common::panda_options()
                }
            )
            .unwrap_err()
        )
    };
    let message = err(given(vec![(
        "panda_joint3",
        JointLimits {
            velocity: Some(0.0),
            ..Default::default()
        },
    )]));
    assert!(
        message.contains("'panda_joint3' a velocity limit of 0"),
        "{message}"
    );
    let message = err(given(vec![(
        "panda_joint3",
        JointLimits {
            position: Some([1.0, -1.0]),
            ..Default::default()
        },
    )]));
    assert!(
        message.contains("'panda_joint3' the range [1, -1]"),
        "{message}"
    );
}

#[test]
fn load_errors_name_the_problem() {
    let err = |path: &str, o: RobotOptions| format!("{:#}", Robot::load(path, &o).unwrap_err());
    let panda = common::asset("franka/franka_panda.urdf");
    let message = err(
        &panda,
        RobotOptions {
            ee_links: vec!["hand".into()],
            ..common::panda_options()
        },
    );
    assert!(message.contains("unknown ee_link 'hand'"), "{message}");
    let message = err(
        &panda,
        RobotOptions {
            lock_joints: HashMap::from([("finger".into(), 0.0)]),
            ..common::panda_options()
        },
    );
    assert!(message.contains("unknown joint 'finger'"), "{message}");
    let message = err(
        &panda,
        RobotOptions {
            default_q: Some(vec![0.0; 6]),
            ..common::panda_options()
        },
    );
    assert!(message.contains("default_q has 6 values"), "{message}");
    let every_joint = (1..=7)
        .map(|j| (format!("panda_joint{j}"), 0.0))
        .chain(common::panda_options().lock_joints);
    let message = err(
        &panda,
        RobotOptions {
            lock_joints: every_joint.collect(),
            ..common::panda_options()
        },
    );
    assert!(message.contains("no actuated joints"), "{message}");
    let exact = SphereOptions {
        tolerance: 0.0,
        ..Default::default()
    };
    let message = err(
        &common::asset(SO101),
        RobotOptions {
            spheres: exact,
            ..Default::default()
        },
    );
    assert!(message.contains("positive tolerance"), "{message}");
    let message = err("robot.sdf", RobotOptions::default());
    assert!(
        message.contains("unsupported robot description format"),
        "{message}"
    );
    // Hosts can match on the kind, which names the file.
    let kind = Robot::load("robot.sdf", &RobotOptions::default()).unwrap_err();
    assert!(
        matches!(&kind, Error::Load { path, .. } if path.ends_with("robot.sdf")),
        "{kind:?}"
    );
    let mut model = common::panda_options().collision_model.unwrap();
    model
        .spheres
        .insert("gripper".into(), vec![[0.0, 0.0, 0.0, 0.1]]);
    let message = err(
        &panda,
        RobotOptions {
            collision_model: Some(model),
            ..common::panda_options()
        },
    );
    assert!(message.contains("link 'gripper'"), "{message}");
}

#[test]
fn every_test_arm_loads_spherizes_and_plans() {
    let panda = RobotOptions {
        collision_model: None,
        ..common::panda_options()
    };
    let arms = [
        (UR5E, RobotOptions::default()),
        (
            "menagerie/universal_robots_ur5e/ur5e.xml",
            RobotOptions::default(),
        ),
        (SO101, RobotOptions::default()),
        ("franka/franka_panda.urdf", panda),
    ];
    for (path, options) in arms {
        let robot = Robot::load(common::asset(path), &options).unwrap();
        let cpu = Device::cpu(&robot).unwrap();
        let mut rng = Rng::new(12);
        let scene: Vec<World> = (0..4)
            .map(|_| common::tabletop_for(&robot, &cpu, &mut rng))
            .collect();
        let worlds = cpu.upload(&scene).unwrap();
        // Goals within 1.5 rad of the default pose per joint: tabletop motions. Goals across the
        // UR5e's full +-2 pi range swing the arm through the table, which trajectory optimization
        // alone does not escape.
        let mut problems = vec![];
        while problems.len() < 12 {
            let world = problems.len() as u32 % 4;
            let goal: Vec<f32> = (0..robot.dof())
                .map(|j| {
                    let d = robot.default_q()[j];
                    rng.range(
                        (d - 1.5).max(robot.lower()[j]),
                        (d + 1.5).min(robot.upper()[j]),
                    )
                })
                .collect();
            let far: f32 = goal
                .iter()
                .zip(robot.default_q())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            if far > 1.0
                && cpu
                    .evaluate(&worlds, &[world], &goal, &CollisionWeights::NONE)
                    .unwrap()
                    .collision_free(0)
            {
                problems.push(PlanProblem::new(world, robot.default_q(), goal));
            }
        }
        for d in devices(&robot) {
            let result = batchplan::plan(
                &d,
                &d.upload(&scene).unwrap(),
                &problems,
                &PlanOptions::default(),
            )
            .unwrap();
            let solved = result.solved().count();
            eprintln!(
                "{path} on {}: planned {solved}/{}",
                d.name(),
                problems.len()
            );

            assert!(
                solved >= 9,
                "{path} on {}: planned only {solved} of {}",
                d.name(),
                problems.len()
            );
        }
    }
}

/// Evaluates 500 random configurations among `scene`'s obstacles on every device, checks that the
/// devices agree on clearances and gradients and that enough configurations carry cost, and
/// returns the devices.
fn devices_agree(robot: &Robot, scene: &[World]) -> Vec<Device> {
    let mut rng = Rng::new(17);
    let q: Vec<f32> = (0..500).flat_map(|_| random_q(robot, &mut rng)).collect();
    let item_world = vec![0; 500];
    let w = CollisionWeights {
        world: 1.0,
        self_collision: 1.0,
        margin: 0.05,
        self_margin: 0.02,
    };
    let devices = devices(robot);
    let reference = devices[0]
        .evaluate(&devices[0].upload(scene).unwrap(), &item_world, &q, &w)
        .unwrap();
    assert!(
        reference.cost.iter().filter(|&&c| c > 0.0).count() > 50,
        "too few configurations carry cost"
    );
    for d in &devices {
        let e = d
            .evaluate(&d.upload(scene).unwrap(), &item_world, &q, &w)
            .unwrap();
        for i in 0..500 {
            assert!(
                (e.world_clearance[i] - reference.world_clearance[i]).abs() < 1e-4,
                "{}: clearance",
                d.name()
            );
            assert!(
                (e.self_clearance[i] - reference.self_clearance[i]).abs() < 1e-4,
                "{}: self clearance",
                d.name()
            );
        }
        let scale = reference
            .gradient
            .iter()
            .fold(1.0f32, |m, g| m.max(g.abs()));
        let worst = e
            .gradient
            .iter()
            .zip(&reference.gradient)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst / scale < 1e-3,
            "{}: gradient off by {worst}",
            d.name()
        );
    }
    devices
}

/// A planar-ish snake: `joints` revolute joints, alternating between two axes, each carrying a
/// 10 cm capsule-like cylinder.
fn snake_urdf(joints: usize) -> String {
    use std::fmt::Write;

    let mut urdf = String::from("<robot name=\"snake\">\n  <link name=\"base\"/>\n");
    for i in 0..joints {
        let (parent, axis) = (
            if i == 0 {
                "base".to_string()
            } else {
                format!("l{}", i - 1)
            },
            ["0 1 0", "0 0 1"][i % 2],
        );
        writeln!(
            urdf,
            "  <link name=\"l{i}\"><collision><origin xyz=\"0 0 0.05\"/><geometry><cylinder radius=\"0.02\" length=\"0.1\"/></geometry></collision></link>\n\
             <joint name=\"j{i}\" type=\"revolute\"><parent link=\"{parent}\"/><child link=\"l{i}\"/>\
             <origin xyz=\"0 0 {}\"/><axis xyz=\"{axis}\"/><limit lower=\"-1.5\" upper=\"1.5\" velocity=\"2\" effort=\"1\"/></joint>",
            if i == 0 { 0.0 } else { 0.1 }
        )
        .unwrap();
    }
    urdf + "</robot>\n"
}

#[test]
fn robots_past_the_old_kernel_limits_work_on_every_device() {
    // 40 actuated joints, 41 links and more than 128 spheres: past every former fixed limit
    // (16 joints, 32 links, 128 spheres).
    let dir = scratch("snake");
    std::fs::write(dir.join("snake.urdf"), snake_urdf(40)).unwrap();
    let options = RobotOptions {
        spheres: SphereOptions {
            budget: 200,
            ..Default::default()
        },
        ..Default::default()
    };
    let robot = Robot::load(dir.join("snake.urdf"), &options).unwrap();
    let spheres: usize = robot.collision_model().spheres.values().map(Vec::len).sum();
    assert!(
        robot.dof() == 40 && spheres > 128,
        "{} joints, {spheres} spheres",
        robot.dof()
    );
    let scene = [World::new(vec![Obstacle::cuboid(
        Vec3::new(0.6, 0.0, 1.2),
        Vec3::splat(0.3),
    )])];
    for d in devices_agree(&robot, &scene) {
        let worlds = d.upload(&scene).unwrap();
        let start = vec![0.0; 40];
        let mut goal = vec![0.0; 40];
        goal[0] = 1.0;
        let problem = PlanProblem::new(0, start, goal);
        let result = batchplan::plan(&d, &worlds, &[problem], &PlanOptions::default()).unwrap();
        assert!(
            result.best(0).is_some(),
            "{}: no plan for the snake",
            d.name()
        );
    }
}

/// A one-link arm on a `kind` joint ("floating" or "planar", normal z) from `world`.
fn mounted_arm_urdf(kind: &str) -> String {
    format!(
        r#"<robot name="mounted">
  <link name="world"/>
  <link name="base"><collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision></link>
  <link name="arm"><collision><origin xyz="0 0 0.25"/><geometry><cylinder radius="0.03" length="0.5"/></geometry></collision></link>
  <joint name="mount" type="{kind}"><parent link="world"/><child link="base"/><origin xyz="0 0 0.5"/><axis xyz="0 0 1"/></joint>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 0.05"/>
    <axis xyz="0 1 0"/><limit lower="-2" upper="2" velocity="2" effort="1"/></joint>
</robot>"#
    )
}

#[test]
fn floating_and_planar_joints_become_one_axis_joints() {
    let dir = scratch("compound");
    let floating = [
        "mount_x", "mount_y", "mount_z", "mount_rx", "mount_ry", "mount_rz",
    ];
    for (kind, joints) in [
        ("floating", &floating[..]),
        ("planar", &["mount_x", "mount_y", "mount_theta"][..]),
    ] {
        let path = dir.join(format!("{kind}.urdf"));
        std::fs::write(&path, mounted_arm_urdf(kind)).unwrap();
        // Translations have no limits of their own.
        let err = Robot::load(&path, &RobotOptions::default())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("mount_x") && err.contains("joint_limits"),
            "{err}"
        );
        let joint_limits = ["mount_x", "mount_y", "mount_z"]
            .into_iter()
            .filter(|n| joints.contains(n))
            .map(|n| {
                (
                    n.to_string(),
                    JointLimits {
                        position: Some([-1.0, 1.0]),
                        ..Default::default()
                    },
                )
            })
            .collect();
        let options = RobotOptions {
            joint_limits,
            ee_links: vec!["arm".into()],
            ..Default::default()
        };
        let robot = Robot::load(&path, &options).unwrap();
        assert_eq!(robot.joint_names()[..joints.len()], *joints, "{kind}");
        assert_eq!(robot.dof(), joints.len() + 1);
        // Sliding along x moves the arm along x; the last rotation turns it about z.
        let mut q = vec![0.0; robot.dof()];
        let home = robot.ee_pose(&q, 0);
        q[0] = 0.3;
        let slid = robot.ee_pose(&q, 0);
        assert!(
            (slid.position - home.position - Vec3::new(0.3, 0.0, 0.0)).length() < 1e-5,
            "{kind}: {slid:?}"
        );
        q[0] = 0.0;
        q[joints.len() - 1] = 0.7;
        let turned = robot.ee_pose(&q, 0);
        let expected = Mat3::from_quat(Quat::from_rotation_z(0.7) * home.rotation);
        let off = (Mat3::from_quat(turned.rotation) - expected)
            .to_cols_array()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(off < 1e-5, "{kind}: rotation off by {off}");
        // Every device sees the same robot and can move its base.
        let scene = [World::new(vec![Obstacle::cuboid(
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::splat(0.6),
        )])];
        for d in devices_agree(&robot, &scene) {
            let worlds = d.upload(&[World::default()]).unwrap();
            let mut goal = vec![0.0; robot.dof()];
            goal[0] = 0.8;
            let problem = PlanProblem::new(0, vec![0.0; robot.dof()], goal);
            let result = batchplan::plan(&d, &worlds, &[problem], &PlanOptions::default()).unwrap();
            assert!(
                result.best(0).is_some(),
                "{}: no plan for the {kind} base",
                d.name()
            );
        }
    }
}
