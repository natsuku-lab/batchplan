//! Robots loaded from descriptions: mimic joints, fitted collision spheres, self-collision pairs,
//! collision-model files, SRDF import, package resolution and load errors. GPU parts are skipped
//! when no adapter is available unless `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use batchplan::rng::Rng;
use batchplan::{
    CollisionModel, CollisionWeights, Device, Error, GripperCalibration, JointLimits, Obstacle,
    Part, PlanOptions, PlanProblem, Pose, Robot, RobotOptions, SphereOptions, Trajectory, World,
};
use glam::{DMat3, DVec3, Mat3, Quat, Vec3};
use mesh_loader::Loader;

const UR5E: &str = "ur5e/ur_description/urdf/ur5e.urdf";
const SO101: &str = "so101/so101_new_calib.urdf";
const GRIPPER: &str = "robotiq_2f85/robotiq_description/urdf/robotiq_2f_85.urdf";
/// A robot's stand: boxes whose faces are wide against a tolerance of centimeters.
const STAND: &str = r#"<robot name="stand">
  <link name="base"/>
  <link name="feet"><collision><geometry><box size="0.77 0.7 0.31"/></geometry></collision></link>
  <link name="box"><collision><origin xyz="0 0 0.5"/><geometry><box size="0.22 0.4 0.53"/></geometry></collision></link>
  <joint name="turn" type="revolute"><parent link="base"/><child link="feet"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
  <joint name="tilt" type="revolute"><parent link="feet"/><child link="box"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
</robot>"#;

fn load(path: &str) -> Robot {
    Robot::load(common::asset(path), &RobotOptions::default()).unwrap()
}

/// Hand 0 of `robot` as a gripper with these widths and closed joints: its opening where its jaws
/// slide, `None` where they do not.
fn sliding_opening(
    robot: &Robot,
    loaded_gap: f32,
    capacity: f32,
    closed_gap: f32,
    closed: &HashMap<String, f32>,
) -> Option<f32> {
    let calibration = GripperCalibration {
        capacity: Some(capacity),
        ..GripperCalibration::new(loaded_gap, closed_gap, closed.clone())
    };
    let gripper = robot.gripper(0, &calibration).unwrap();
    gripper.sliding().then_some(gripper.opening())
}

#[test]
fn parallel_jaw_motion_matches_the_calibrated_stroke_on_both_sides() {
    let panda = common::panda().unwrap();
    let closed = HashMap::from([
        ("panda_finger_joint1".into(), 0.0),
        ("panda_finger_joint2".into(), 0.0),
    ]);
    let gap = |loaded, capacity, locks| sliding_opening(&panda, loaded, capacity, 0.0, locks);
    assert_eq!(gap(0.080, 0.060, &closed), Some(0.080));
    assert_eq!(gap(0.080f32.next_down(), 0.080, &closed), Some(0.080));
    assert_eq!(gap(0.060, 0.060, &closed), None);
    let two_mm = closed.keys().map(|name| (name.clone(), 0.001)).collect();
    assert_eq!(
        sliding_opening(&panda, 0.080, 0.016, 0.002, &two_mm),
        Some(0.080)
    );
    let past_stop = closed.keys().map(|name| (name.clone(), -0.01)).collect();
    assert_eq!(gap(0.100, 0.060, &past_stop), None);
    let one = HashMap::from([("panda_finger_joint1".into(), 0.0)]);
    assert_eq!(gap(0.080, 0.060, &one), None);
    let calibration = |capacity, friction| GripperCalibration {
        capacity: Some(capacity),
        friction,
        ..GripperCalibration::new(0.08, 0.0, closed.clone())
    };
    assert!(panda.gripper(1, &calibration(0.060, 0.5)).is_err());
    assert!(panda.gripper(0, &calibration(0.081, 0.5)).is_err());
    // A capacity no wider than the closed gap, below it or where it and the closed gap are one
    // step under the loaded gap, which the opening then snaps to.
    let below = GripperCalibration {
        capacity: Some(0.01),
        ..GripperCalibration::new(0.08, 0.02, closed.clone())
    };
    let step = 0.08f32.next_down();
    let at_closed = GripperCalibration {
        capacity: Some(step),
        ..GripperCalibration::new(0.08, step, closed.clone())
    };
    for (wrong, value) in [(below, "0.01"), (at_closed, "0.07999999")] {
        match panda.gripper(0, &wrong) {
            Err(Error::Input(message)) => assert!(message.contains(value), "{message}"),
            other => panic!("{wrong:?}: {other:?}"),
        }
    }
    for friction in [0.0, -0.5, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(
                panda.gripper(0, &calibration(0.060, friction)),
                Err(Error::Input(_))
            ),
            "friction {friction}"
        );
    }
}

#[test]
fn an_unchanged_torso_lock_does_not_change_jaw_qualification() {
    let mut options = common::panda_options();
    options.lock_joints.insert("panda_joint1".into(), 0.0);
    options.default_q.as_mut().unwrap().remove(0);
    let robot = Robot::load(common::asset("franka/franka_panda.urdf"), &options).unwrap();
    let mut closed = HashMap::from([
        ("panda_finger_joint1".into(), 0.0),
        ("panda_finger_joint2".into(), 0.0),
    ]);
    let opening = |locks: &HashMap<String, f32>| sliding_opening(&robot, 0.080, 0.060, 0.0, locks);
    assert_eq!(opening(&closed), Some(0.080));
    closed.insert("panda_joint1".into(), 0.0);
    assert_eq!(opening(&closed), Some(0.080));
    // A joint closed where it was loaded stays as loaded at every opening.
    let calibration = GripperCalibration::new(0.08, 0.0, closed.clone());
    let locked = robot.gripper(0, &calibration).unwrap().locked_at(0.04);
    assert!(
        locked.len() == 2 && !locked.contains_key("panda_joint1"),
        "{locked:?}"
    );
    closed.insert("panda_joint1".into(), 0.2);
    assert_eq!(opening(&closed), None);
}

#[test]
fn jaw_axes_and_coupled_limits_are_checked_even_far_from_the_origin() {
    let urdf = r#"<robot name="jaws">
      <link name="base"/><link name="palm"/><link name="left"/><link name="right"/><link name="rack"/>
      <joint name="base_slide" type="prismatic"><parent link="base"/><child link="palm"/>
        <axis xyz="1 0 0"/><limit effort="1" velocity="1" lower="0" upper="100000"/></joint>
      <joint name="l" type="prismatic"><parent link="palm"/><child link="left"/>
        <axis xyz="0 1 0"/><limit effort="1" velocity="1" lower="0" upper="0.04"/></joint>
      <joint name="r" type="prismatic"><parent link="palm"/><child link="right"/>
        <axis xyz="0 -1 0"/><limit effort="1" velocity="1" lower="0" upper="0.04"/></joint>
      <joint name="coupled" type="prismatic"><parent link="base"/><child link="rack"/>
        <axis xyz="0 0 1"/><limit effort="1" velocity="1" lower="0" upper="0.05"/>
        <mimic joint="l"/></joint></robot>"#;
    let mut model = CollisionModel::default();
    for link in ["palm", "left", "right"] {
        model
            .spheres
            .insert(link.into(), vec![[0.0, 0.0, 0.0, 0.001]]);
    }
    let closed = HashMap::from([("l".into(), 0.0), ("r".into(), 0.0)]);
    let check = |text: &str, position| {
        let options = RobotOptions {
            ee_links: vec!["palm".into()],
            lock_joints: HashMap::from([("l".into(), 0.04), ("r".into(), 0.04)]),
            default_q: Some(vec![position]),
            collision_model: Some(model.clone()),
            ..Default::default()
        };
        let robot = Robot::from_urdf(text, &options).unwrap();
        sliding_opening(&robot, 0.08, 0.06, 0.0, &closed)
    };
    let skew = urdf
        .replace("0 1 0", "0.01 1 0")
        .replace("0 -1 0", "-0.01 -1 0");
    for position in [0.0, 65536.0] {
        assert_eq!(check(urdf, position), Some(0.08));
        assert_eq!(check(&skew, position), None);
    }
    let limited = urdf.replace(
        "lower=\"0\" upper=\"0.05\"",
        "lower=\"0.01\" upper=\"0.05\"",
    );
    assert_eq!(check(&limited, 0.0), None, "the coupled rack hits its stop");
}

/// Nothing reads what a link looks like, so a visual mesh that is not there keeps no robot from
/// loading.
#[test]
fn a_robot_loads_without_a_visual_mesh_it_names() {
    let urdf = include_str!("../assets/franka/franka_panda.urdf").replace(
        "</robot>",
        r#"<link name="badge"><visual><geometry>
             <mesh filename="package://nowhere/badge.stl"/></geometry></visual></link>
           <joint name="badge_joint" type="fixed"><parent link="panda_link0"/>
             <child link="badge"/></joint></robot>"#,
    );
    let urdf = urdf.replace("meshes/", &common::asset("franka/meshes/"));
    let robot = Robot::from_urdf(&urdf, &common::panda_options()).expect("no visual mesh is read");
    assert_eq!(robot.dof(), common::panda().unwrap().dof());
}

/// A rotating mimic can return to its initial orientation at closure even though it sweeps a
/// sphere sideways in between. The two endpoint sphere sets alone cannot establish a slide.
#[test]
fn a_revolving_finger_mimic_cannot_qualify_as_a_slide() {
    let mut options = common::panda_options();
    let model = options.collision_model.as_mut().unwrap();
    model
        .spheres
        .insert("tip".into(), vec![[0.01, 0.0, 0.0, 0.003]]);
    let urdf = include_str!("../assets/franka/franka_panda.urdf").replace(
        "</robot>",
        r#"<link name="tip"/>
           <joint name="swing" type="revolute"><parent link="panda_leftfinger"/><child link="tip"/>
             <axis xyz="0 0 1"/><limit lower="0" upper="7" effort="1" velocity="1"/>
             <mimic joint="panda_finger_joint1" multiplier="157.07963268"/>
           </joint></robot>"#,
    );
    let urdf = urdf.replace("meshes/", &common::asset("franka/meshes/"));
    let robot = Robot::from_urdf(&urdf, &options).unwrap();
    let closed = HashMap::from([
        ("panda_finger_joint1".into(), 0.0),
        ("panda_finger_joint2".into(), 0.0),
    ]);
    let shut = robot.relock(&closed).unwrap();
    let open = robot.hand_spheres(0);
    assert!(open.iter().any(|s| s.radius == 0.003 && s.side != 0.0));
    assert!(
        open.iter()
            .zip(shut.hand_spheres(0))
            .all(|(a, b)| { (a.center - Vec3::Y * a.side * 0.040).distance(b.center) < 1e-6 })
    );
    assert_eq!(sliding_opening(&robot, 0.080, 0.080, 0.0, &closed), None);
}

/// Openings turn into the finger joints and back between the closed and the loaded gap, not the
/// capacity, where the two differ: the Panda relocked 6.15 cm open and grasping up to 5 cm, as a
/// TIAGo's hand, each finger standing at half the opening, the ends exact.
#[test]
fn openings_convert_between_the_closed_and_loaded_gaps() {
    let (loaded, capacity) = (0.0615, 0.05);
    let fingers = |at: f32| -> HashMap<String, f32> {
        let each = ["panda_finger_joint1", "panda_finger_joint2"];
        each.map(|joint| (joint.to_string(), at)).into()
    };
    let robot = common::panda()
        .unwrap()
        .relock(&fingers(loaded / 2.0))
        .unwrap();
    let calibration = GripperCalibration {
        capacity: Some(capacity),
        ..GripperCalibration::new(loaded, 0.0, fingers(0.0))
    };
    let gripper = robot.gripper(0, &calibration).unwrap();
    assert!(gripper.sliding());
    assert_eq!(
        (gripper.opening(), gripper.capacity(), gripper.closed_gap()),
        (loaded, capacity, 0.0)
    );
    assert_eq!(gripper.locked_at(loaded), fingers(loaded / 2.0));
    assert_eq!(gripper.locked_at(0.0), fingers(0.0));
    assert_eq!(gripper.opening_of(&fingers(loaded / 2.0)), loaded);
    assert_eq!(gripper.opening_of(&fingers(0.0)), 0.0);
    let openings = gripper.openings();
    assert_eq!((openings.len(), openings[0]), (16, loaded));
    for (k, &opening) in openings.iter().enumerate().skip(1) {
        let sixteenths = capacity * (16 - k) as f32 / 16.0;
        assert!((opening - sixteenths).abs() < 1e-7, "{k}: {opening}");
    }
    for &opening in &openings {
        let locked = gripper.locked_at(opening);
        assert!(
            locked.len() == 2 && locked.values().all(|q| (q - opening / 2.0).abs() < 1e-7),
            "{opening}: {locked:?}"
        );
        assert!((gripper.opening_of(&locked) - opening).abs() < 1e-7);
    }
    // Past either gap, the nearer; a joint not named stands as loaded.
    assert_eq!(gripper.locked_at(0.1), fingers(loaded / 2.0));
    assert_eq!(gripper.locked_at(-0.01), fingers(0.0));
    assert_eq!(gripper.opening_of(&fingers(0.05)), loaded);
    let one = HashMap::from([("panda_finger_joint1".to_string(), 0.0)]);
    assert_eq!(gripper.opening_of(&one), loaded / 2.0);
    assert_eq!(gripper.opening_of(&HashMap::new()), loaded);
}

/// A hand whose one jaw turns on a hinge and whose other is part of the palm, as the SO-101's:
/// its gripper turns widths into the hinge's angle and back, between closed at the closed gap and
/// as loaded at the loaded gap, but its jaws do not slide, so fingers that come in narrower are
/// refused, saying why, and the hand as loaded is not.
#[test]
fn a_hinged_jaw_converts_widths_but_does_not_slide() {
    let urdf = r#"<robot name="hinge">
      <link name="base"/><link name="palm"/><link name="jaw"/>
      <joint name="arm" type="revolute"><parent link="base"/><child link="palm"/>
        <axis xyz="0 0 1"/><limit effort="1" velocity="1" lower="-3" upper="3"/></joint>
      <joint name="hinge" type="revolute"><parent link="palm"/><child link="jaw"/>
        <origin xyz="0 0.0235 0"/><axis xyz="1 0 0"/>
        <limit effort="1" velocity="1" lower="-0.2" upper="1.7"/></joint></robot>"#;
    let mut model = CollisionModel::default();
    model
        .spheres
        .insert("palm".into(), vec![[0.0, -0.03, 0.0, 0.005]]);
    model
        .spheres
        .insert("jaw".into(), vec![[0.0, 0.005, 0.0, 0.005]]);
    let options = RobotOptions {
        ee_links: vec!["palm".into()],
        finger_links: HashMap::from([("palm".into(), vec!["jaw".into()])]),
        lock_joints: HashMap::from([("hinge".into(), 0.6)]),
        default_q: Some(vec![0.0]),
        collision_model: Some(model),
        ..Default::default()
    };
    let robot = Robot::from_urdf(urdf, &options).unwrap();
    let (loaded, capacity) = (0.047, 0.04);
    let calibration = GripperCalibration {
        capacity: Some(capacity),
        ..GripperCalibration::new(loaded, 0.0, HashMap::from([("hinge".into(), -0.1)]))
    };
    let gripper = robot.gripper(0, &calibration).unwrap();
    assert!(!gripper.sliding());
    let angle = |opening: f32| gripper.locked_at(opening)["hinge"];
    assert_eq!((angle(loaded), angle(0.0)), (0.6, -0.1));
    let at_capacity = -0.1 + 0.7 * capacity / loaded;
    assert!(
        (angle(capacity) - at_capacity).abs() < 1e-6,
        "{}",
        angle(capacity)
    );
    for opening in [0.0, 0.01, capacity, loaded] {
        let back = gripper.opening_of(&gripper.locked_at(opening));
        assert!((back - opening).abs() < 1e-6, "{opening}: {back}");
    }
    let device = Device::cpu(&robot).unwrap();
    let surface = batchplan::Surface {
        points: vec![Vec3::new(0.0, -0.01, 0.0), Vec3::new(0.0, 0.01, 0.0)],
        normals: vec![Vec3::NEG_Y, Vec3::Y],
        ..Default::default()
    };
    let narrow = batchplan::GraspOptions {
        narrow: true,
        ..Default::default()
    };
    let surfaces = std::slice::from_ref(&surface);
    match batchplan::grasps_on(&device, &gripper, surfaces, &narrow) {
        Err(Error::Input(message)) => assert!(message.contains("slide"), "{message}"),
        other => panic!("narrowing a hinged jaw: {other:?}"),
    }
    let open = batchplan::GraspOptions::default();
    assert!(batchplan::grasps_on(&device, &gripper, surfaces, &open).is_ok());
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

/// Every link's pose in `a` and `b` at `q`, within 1e-6.
fn same_poses(a: &Robot, b: &Robot, q: &[f32], links: &[&str]) {
    for link in links {
        let (pa, pb) = (a.link_pose(q, link).unwrap(), b.link_pose(q, link).unwrap());
        assert!(
            (pa.position - pb.position).length() < 1e-6
                && pa.rotation.dot(pb.rotation).abs() > 1.0 - 1e-6,
            "{link}: {pa:?} against {pb:?}"
        );
    }
}

/// Relocking moves the links a locked joint carries, its mimic joints' links included, to where a
/// robot loaded with the new value has them, and leaves the other locked joints alone.
#[test]
fn relocked_joints_move_their_links_as_a_robot_loaded_there() {
    let dir = scratch("relock");
    let urdf = r#"<robot name="relock">
      <link name="base"/><link name="a"/><link name="b"/><link name="c"/><link name="d"/>
      <joint name="leader" type="revolute"><parent link="base"/><child link="a"/><axis xyz="0 0 1"/>
        <origin xyz="0.1 0 0.2"/><limit lower="-1" upper="1" velocity="2" effort="1"/></joint>
      <joint name="double" type="revolute"><parent link="base"/><child link="b"/><axis xyz="0 1 0"/>
        <origin xyz="0 0.3 0" rpy="0.2 0 0"/>
        <limit lower="-2" upper="2" velocity="1" effort="1"/><mimic joint="leader" multiplier="2" offset="0.5"/></joint>
      <joint name="flipped" type="prismatic"><parent link="a"/><child link="c"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" velocity="5" effort="1"/><mimic joint="leader" multiplier="-1"/></joint>
      <joint name="free" type="revolute"><parent link="c"/><child link="d"/><axis xyz="0 1 0"/>
        <origin xyz="0 0 0.1"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
    </robot>"#;
    let path = dir.join("relock.urdf");
    std::fs::write(&path, urdf).unwrap();
    // The collision model fitted with the leader free, as relocking it needs.
    let free = Robot::load(&path, &RobotOptions::default()).unwrap();
    let locked = |value: f32| RobotOptions {
        lock_joints: HashMap::from([("leader".into(), value)]),
        collision_model: Some(free.collision_model().clone()),
        ..Default::default()
    };
    let robot = Robot::load(&path, &locked(0.2)).unwrap();
    assert_eq!(robot.joint_names(), ["free"]);
    let relocked = robot
        .relock(&HashMap::from([("leader".into(), -0.4)]))
        .unwrap();
    let loaded = Robot::load(&path, &locked(-0.4)).unwrap();
    let links = ["base", "a", "b", "c", "d"];
    same_poses(&relocked, &loaded, &[0.7], &links);
    // Each mimic joint at its own value of the leader's: double = 2 (-0.4) + 0.5 = -0.3 about y,
    // flipped = 0.4 along a's x.
    let b = relocked.link_pose(&[0.7], "b").unwrap();
    let turned = Quat::from_rotation_x(0.2) * Quat::from_rotation_y(-0.3);
    assert!(b.rotation.dot(turned).abs() > 1.0 - 1e-6, "{b:?}");
    let c = relocked.link_pose(&[0.7], "c").unwrap().position;
    let slid = Vec3::new(0.1, 0.0, 0.2) + Quat::from_rotation_z(-0.4) * Vec3::new(0.4, 0.0, 0.0);
    assert!((c - slid).length() < 1e-6, "{c} against {slid}");
    let moved = robot.link_pose(&[0.7], "d").unwrap().position;
    let there = relocked.link_pose(&[0.7], "d").unwrap().position;
    assert!(
        (moved - there).length() > 0.05,
        "the lock moves d: {moved} against {there}"
    );

    // The Panda with its first joint and fingers locked: relocking one finger and the first joint
    // keeps the other finger, and moves the spheres with their links.
    let panda = |joint1: f32, finger: f32| {
        let mut o = common::panda_options();
        o.lock_joints.insert("panda_joint1".into(), joint1);
        o.default_q = None;
        o.lock_joints.insert("panda_finger_joint1".into(), finger);
        Robot::load(common::asset("franka/franka_panda.urdf"), &o).unwrap()
    };
    let relocked = panda(0.3, 0.04)
        .relock(&HashMap::from([
            ("panda_joint1".into(), -0.8),
            ("panda_finger_joint1".into(), 0.01),
        ]))
        .unwrap();
    let loaded = panda(-0.8, 0.01);
    let q = [-0.6, 0.2, 0.4, -2.0, 0.3, 1.4];
    let names = [
        "panda_link1",
        "panda_link4",
        "ee_link",
        "panda_leftfinger",
        "panda_rightfinger",
    ];
    same_poses(&relocked, &loaded, &q, &names);
    let (a, b) = (relocked.spheres(&q), loaded.spheres(&q));
    assert_eq!(a.len(), b.len());
    for (sa, sb) in a.iter().zip(&b) {
        let gap = (0..4).map(|k| (sa[k] - sb[k]).abs()).fold(0.0, f32::max);
        assert!(gap < 1e-6, "{sa:?} against {sb:?}");
    }

    let message = |joints: HashMap<String, f32>| format!("{}", robot.relock(&joints).unwrap_err());
    let free = message(HashMap::from([("free".into(), 0.1)]));
    assert!(free.contains("'free' was not locked"), "{free}");
    let mimic = message(HashMap::from([("double".into(), 0.1)]));
    assert!(mimic.contains("'double' was not locked"), "{mimic}");
    let nan = message(HashMap::from([("leader".into(), f32::NAN)]));
    assert!(nan.contains("not a finite value"), "{nan}");
}

/// A model fitted with a joint locked ignores pairs found at that value alone, so the joint stays
/// where it was locked: relocking it is refused, and so is loading the model with it free or locked
/// elsewhere, through a model saved and loaded again and two arms assembled from it too. The same
/// robot loaded with a model fitted with the joint free relocks it.
#[test]
fn joints_locked_when_the_model_was_fitted_stay_there() {
    let path = common::asset(SO101);
    let locked = |value: Option<f32>, model: Option<CollisionModel>| RobotOptions {
        lock_joints: value
            .map(|v| ("elbow_flex".to_string(), v))
            .into_iter()
            .collect(),
        collision_model: model,
        ..Default::default()
    };
    let fitted = Robot::load(&path, &locked(Some(0.3), None)).unwrap();
    assert_eq!(
        fitted.collision_model().locked_joints,
        BTreeMap::from([("elbow_flex".to_string(), 0.3)])
    );
    let saved = scratch("locked_model").join("model.json");
    fitted.collision_model().save(&saved).unwrap();
    let model = CollisionModel::load(&saved).unwrap();
    // The same text without a file, and text that is no model refused as input.
    let text = model.to_json();
    assert_eq!(text, std::fs::read_to_string(&saved).unwrap());
    assert_eq!(CollisionModel::from_json(&text).unwrap(), model);
    let cut = CollisionModel::from_json(&text[..text.len() / 2]);
    assert!(matches!(cut, Err(batchplan::Error::Input(_))), "{cut:?}");
    let reloaded = Robot::load(&path, &locked(Some(0.3), Some(model.clone()))).unwrap();
    let moved = HashMap::from([("elbow_flex".into(), -1.0)]);
    for robot in [&fitted, &reloaded] {
        let message = format!("{}", robot.relock(&moved).unwrap_err());
        assert!(
            message.contains("'elbow_flex' was locked at 0.3 when the collision model was fitted"),
            "{message}"
        );
    }
    for elsewhere in [None, Some(-1.0)] {
        let message = format!(
            "{:#}",
            Robot::load(&path, &locked(elsewhere, Some(model.clone()))).unwrap_err()
        );
        assert!(
            message.contains("fitted with joint 'elbow_flex' locked at 0.3"),
            "{message}"
        );
    }

    // Two arms from the model: each arm's joint, by its prefixed name.
    let arm = |prefix: &str, y: f32| Part {
        path: path.clone().into(),
        prefix: prefix.into(),
        base: Pose::new(Vec3::new(0.0, y, 0.0), Quat::IDENTITY),
    };
    let parts = [arm("left_", 0.2), arm("right_", -0.2)];
    let both = |left: f32| RobotOptions {
        lock_joints: HashMap::from([
            ("left_elbow_flex".into(), left),
            ("right_elbow_flex".into(), 0.3),
        ]),
        collision_model: Some(CollisionModel::assemble(&[
            ("left_", &model),
            ("right_", &model),
        ])),
        ..Default::default()
    };
    let pair = Robot::assemble(&parts, &both(0.3)).unwrap();
    let message = format!(
        "{}",
        pair.relock(&HashMap::from([("left_elbow_flex".into(), -1.0)]))
            .unwrap_err()
    );
    assert!(
        message.contains("'left_elbow_flex' was locked at 0.3"),
        "{message}"
    );
    let message = format!("{:#}", Robot::assemble(&parts, &both(-1.0)).unwrap_err());
    assert!(
        message.contains("joint 'left_elbow_flex' locked at 0.3"),
        "{message}"
    );

    let free = Robot::load(&path, &RobotOptions::default()).unwrap();
    assert!(free.collision_model().locked_joints.is_empty());
    let robot = Robot::load(
        &path,
        &locked(Some(0.3), Some(free.collision_model().clone())),
    )
    .unwrap();
    assert!(robot.relock(&moved).is_ok());
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
    let devices = common::devices(&robot);
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
fn link_triangles(path: &str, package_root: &Path) -> BTreeMap<String, Vec<[Vec3; 3]>> {
    let urdf = urdf_rs::read_file(common::asset(path)).unwrap();
    let dir = Path::new(&common::asset(path))
        .parent()
        .unwrap()
        .to_path_buf();
    let mut out = BTreeMap::new();
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

/// A binary STL may start with "solid": its 80-byte header is free text, which CAD exporters fill
/// with "solid <name>". Thirteen triangles put a carriage return in the count, which ends that
/// "solid" line for a parser reading ASCII. It loads as its ASCII twin.
#[test]
fn binary_stls_that_start_with_solid_load_as_their_ascii_twin() {
    use std::fmt::Write;
    let corners = |i: usize| Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, (i >> 2) as f32);
    let quads = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ];
    let mut triangles: Vec<[Vec3; 3]> = quads
        .iter()
        .flat_map(|&[a, b, c, d]| [[a, b, c], [a, c, d]])
        .map(|t| t.map(|i| 0.1 * corners(i)))
        .collect();
    triangles.push(triangles[0]);
    assert_eq!(triangles.len(), 13, "13 is a carriage return");
    let mut binary = format!("{:<80}", "solid exported cube").into_bytes();
    binary.extend(13u32.to_le_bytes());
    let mut ascii = String::from("solid exported cube\n");
    for t in &triangles {
        binary.extend([0.0f32; 3].iter().flat_map(|v| v.to_le_bytes()));
        binary.extend(
            t.iter()
                .flat_map(|v| v.to_array())
                .flat_map(f32::to_le_bytes),
        );
        binary.extend([0, 0]);
        ascii += "facet normal 0 0 0\nouter loop\n";
        for v in t {
            writeln!(ascii, "vertex {} {} {}", v.x, v.y, v.z).expect("writing to a String");
        }
        ascii += "endloop\nendfacet\n";
    }
    ascii += "endsolid exported cube\n";
    let dir = scratch("solid-stl");
    std::fs::write(dir.join("binary.stl"), binary).unwrap();
    std::fs::write(dir.join("ascii.stl"), ascii).unwrap();
    let robot = |mesh: &str| {
        let urdf = format!(
            r#"<robot name="cube"><link name="base"/>
              <joint name="j" type="revolute"><parent link="base"/><child link="cube"/>
                <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
              <link name="cube"><collision><geometry><mesh filename="{}"/></geometry></collision></link>
            </robot>"#,
            dir.join(mesh).display()
        );
        let path = dir.join(format!("{mesh}.urdf"));
        std::fs::write(&path, urdf).unwrap();
        Robot::load(&path, &RobotOptions::default())
    };
    let twin = robot("ascii.stl").unwrap();
    assert!(
        !twin.collision_model().spheres["cube"].is_empty(),
        "the cube is fitted with spheres"
    );
    let binary = robot("binary.stl").unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(binary.collision_model(), twin.collision_model());
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
            lock_joints: HashMap::from([("panda_joint4".into(), f32::NAN)]),
            ..common::panda_options()
        },
    );
    assert!(
        message.contains("lock joint 'panda_joint4' at NaN, not a finite value"),
        "{message}"
    );
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
    // A sphere that reaches less than nothing, or nowhere, would check as farther from what it
    // touches than it is.
    for (sphere, value) in [
        ([0.0, 0.0, 0.0, -0.05], "-0.05"),
        ([f32::NAN, 0.0, 0.0, 0.05], "NaN"),
    ] {
        let mut model = common::panda_options().collision_model.unwrap();
        model.spheres.insert("panda_hand".into(), vec![sphere]);
        let message = err(
            &panda,
            RobotOptions {
                collision_model: Some(model),
                ..common::panda_options()
            },
        );
        assert!(
            message.contains("of link 'panda_hand'") && message.contains(value),
            "{message}"
        );
    }
    // A buffer may shrink the hand's spheres (the smallest is 2.2 cm in radius), not past nothing.
    for (buffer, refused) in [(f32::INFINITY, true), (-0.05, true), (-0.01, false)] {
        let mut model = common::panda_options().collision_model.unwrap();
        model
            .self_collision_buffer
            .insert("panda_hand".into(), buffer);
        let options = RobotOptions {
            collision_model: Some(model),
            ..common::panda_options()
        };
        if refused {
            let message = err(&panda, options);
            let named = format!("got {buffer}");
            assert!(
                message.contains("buffer of link 'panda_hand'") && message.contains(&named),
                "{message}"
            );
        } else {
            Robot::load(&panda, &options).unwrap();
        }
    }
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
        for d in common::devices(&robot) {
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
    let devices = common::devices(robot);
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

#[test]
fn spheres_are_the_collision_model_placed_at_the_configuration() {
    let robot = load(UR5E);
    let mut rng = Rng::new(3);
    let q = random_q(&robot, &mut rng);
    let mut expected = vec![];
    for link in robot.collision_model().spheres.keys() {
        let pose = robot.link_pose(&q, link).unwrap();
        for s in &robot.collision_model().spheres[link] {
            let c = pose.position + pose.rotation * Vec3::new(s[0], s[1], s[2]);
            expected.push([c.x, c.y, c.z, s[3]]);
        }
    }
    let mut placed = robot.spheres(&q);
    assert_eq!(placed.len(), expected.len());
    let key = |s: &[f32; 4]| (s[0].to_bits(), s[1].to_bits(), s[2].to_bits());
    placed.sort_by_key(key);
    expected.sort_by_key(key);
    for (a, b) in placed.iter().zip(&expected) {
        let gap = Vec3::from_slice(&a[..3]).distance(Vec3::from_slice(&b[..3]));
        assert!(gap < 1e-5 && a[3] == b[3], "{a:?} against {b:?}");
    }
}

#[test]
fn reach_bounds_every_sphere_and_stays_near_the_farthest_one() {
    let dir = scratch("reach");
    let planar = dir.join("planar.urdf");
    std::fs::write(&planar, mounted_arm_urdf("planar")).unwrap();
    // The same base on a slope: its slides rise and fall as they go.
    let sloped = dir.join("sloped.urdf");
    let slope = mounted_arm_urdf("planar").replace(
        r#"<origin xyz="0 0 0.5"/>"#,
        r#"<origin xyz="0 0 0.5" rpy="0.7 0 0"/>"#,
    );
    assert!(slope.contains("rpy"), "the base turns");
    std::fs::write(&sloped, slope).unwrap();
    let slide = JointLimits {
        position: Some([-1.0, 1.0]),
        ..Default::default()
    };
    let on_a_base = RobotOptions {
        joint_limits: [("mount_x".into(), slide), ("mount_y".into(), slide)].into(),
        ..Default::default()
    };
    let robots = [
        ("UR5e", load(UR5E)),
        ("SO-101", load(SO101)),
        ("Panda", common::panda().unwrap()),
        ("two Pandas", common::two_pandas().unwrap()),
        ("the 2F-85", load(GRIPPER)),
        ("a planar base", Robot::load(&planar, &on_a_base).unwrap()),
        ("a sloped base", Robot::load(&sloped, &on_a_base).unwrap()),
    ];
    let mut rng = Rng::new(5);
    for (name, robot) in robots {
        let reach = robot.reach();
        let far = |q: &[f32]| {
            robot
                .spheres(q)
                .iter()
                .map(|s| Vec3::from_slice(&s[..3]).length() + s[3])
                .fold(0.0, f32::max)
        };
        // The farthest of random configurations, then climbed toward farther ones.
        let mut best = robot.default_q().to_vec();
        for _ in 0..2000 {
            let q = random_q(&robot, &mut rng);
            if far(&q) > far(&best) {
                best = q;
            }
        }
        for k in 0..4000 {
            let step = 0.5 * (1.0 - k as f32 / 4000.0);
            let q: Vec<f32> = best
                .iter()
                .enumerate()
                .map(|(j, v)| {
                    let (lo, hi) = (robot.lower()[j], robot.upper()[j]);
                    (v + step * (hi - lo).min(6.3) * rng.range(-0.5, 0.5)).clamp(lo, hi)
                })
                .collect();
            if far(&q) > far(&best) {
                best = q;
            }
        }
        let farthest = far(&best);
        assert!(farthest > 0.0, "{name} has spheres");
        assert!(
            farthest <= reach + 1e-5,
            "{name}: a sphere reaches {farthest} m, past the bound of {reach} m"
        );
        assert!(
            reach <= 1.12 * farthest,
            "{name}: a bound of {reach} m for spheres reaching {farthest} m"
        );
    }
}

#[test]
fn qualification_measures_how_the_spheres_fit_each_link() {
    let robot = load(SO101);
    let fitted = robot.qualify().unwrap();
    let with_geometry = [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "lower_arm_link",
        "wrist_link",
        "gripper_link",
        "moving_jaw_so101_v1_link",
    ];
    let named: Vec<&str> = fitted.links.iter().map(|l| l.link.as_str()).collect();
    assert_eq!(named, with_geometry);
    // The fit's own account of its overhang, to the tenth of a millimeter it gives.
    let source = robot.collision_model().source.clone().unwrap();
    let reported: f32 = source
        .split("up to ")
        .nth(1)
        .unwrap()
        .split(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let largest = fitted.links.iter().map(|l| l.overhang).fold(0.0, f32::max);
    assert!(
        (largest * 1e3 - reported).abs() < 0.06,
        "{largest} m against {source}"
    );
    for l in &fitted.links {
        assert_eq!(l.spheres, robot.collision_model().spheres[&l.link].len());
        assert!(l.uncovered < 0.005 && l.thickness > 0.01, "{l:?}");
    }
    // Spheres halved leave most of the surface out and reach less far; one moved off the link
    // reaches as far as it moved.
    let mut model = robot.collision_model().clone();
    for s in model.spheres.get_mut("upper_arm_link").unwrap() {
        s[3] *= 0.5;
    }
    model.spheres.get_mut("wrist_link").unwrap()[0][2] += 0.05;
    let hand_tuned = Robot::load(
        common::asset(SO101),
        &RobotOptions {
            collision_model: Some(model),
            ..Default::default()
        },
    )
    .unwrap()
    .qualify()
    .unwrap();
    let link = |q: &batchplan::Qualification, name: &str| {
        q.links.iter().find(|l| l.link == name).unwrap().clone()
    };
    let (before, after) = (
        link(&fitted, "upper_arm_link"),
        link(&hand_tuned, "upper_arm_link"),
    );
    assert!(
        after.uncovered > 0.3 && after.overhang < before.overhang,
        "{after:?}"
    );
    let (before, after) = (link(&fitted, "wrist_link"), link(&hand_tuned, "wrist_link"));
    assert!(
        after.overhang > before.overhang + 0.02,
        "{before:?} then {after:?}"
    );
    assert_eq!(link(&fitted, "base_link"), link(&hand_tuned, "base_link"));
}

#[test]
fn a_hands_tip_is_how_far_its_spheres_reach_past_its_frame() {
    let robot = common::panda().unwrap();
    let q = robot.default_q();
    let frame = robot.ee_pose(q, 0);
    let mut farthest = f32::NEG_INFINITY;
    for link in robot.hand_links(0) {
        let Some(spheres) = robot.collision_model().spheres.get(link) else {
            continue;
        };
        let pose = robot.link_pose(q, link).unwrap();
        for s in spheres {
            let center = pose.position + pose.rotation * Vec3::new(s[0], s[1], s[2]);
            let along = (frame.rotation * Vec3::Z).dot(center - frame.position);
            farthest = farthest.max(along + s[3]);
        }
    }
    assert!(
        farthest > 0.0,
        "the fingertips reach past the frame between them"
    );
    assert!((robot.tip(0) - farthest).abs() < 1e-6);
    assert_eq!(robot.qualify().unwrap().tips, [robot.tip(0)]);
}

/// Fitted spheres keep every link within its tolerance when the budget allows it, and otherwise
/// reach less far as the budget grows: the SO-101's jaws, held to 1 mm, are thin enough that a
/// cover chosen on too few samples or candidates misses parts of them, which spheres then grow to
/// reach, and a stand's wide faces take many refinements of a cover to hold.
#[test]
fn fitted_spheres_keep_each_links_tolerance_and_reach_less_as_the_budget_grows() {
    let jaws = ["gripper_link", "moving_jaw_so101_v1_link"];
    let fit = |budget: usize, jaw: f32| {
        let options = RobotOptions {
            spheres: SphereOptions {
                budget,
                ..Default::default()
            },
            sphere_tolerances: jaws.iter().map(|l| (l.to_string(), jaw)).collect(),
            ..Default::default()
        };
        let robot = Robot::load(common::asset(SO101), &options).unwrap();
        let spheres: usize = robot.collision_model().spheres.values().map(Vec::len).sum();
        assert!(spheres <= budget, "{spheres} spheres");
        (spheres, robot.qualify().unwrap().links)
    };
    let tolerance = |link: &str, jaw: f32| {
        if jaws.contains(&link) {
            jaw
        } else {
            SphereOptions::default().tolerance
        }
    };
    // A budget that reaches the tolerances: every link keeps its own.
    let (spheres, links) = fit(2048, 0.003);
    assert!(
        spheres < 2048,
        "the budget was the limit: {spheres} spheres"
    );
    for l in &links {
        assert!(
            l.overhang <= tolerance(&l.link, 0.003) + 1e-5,
            "{} reaches {:.2} mm past its geometry",
            l.link,
            l.overhang * 1e3
        );
    }
    // A box with a fin 0.15 mm thick, thinner than the grid and than a quarter of the tolerance:
    // the fin's samples get candidates of their own, and keep to it too.
    let dir = scratch("fin");
    let urdf = r#"<robot name="fin">
      <link name="base"/>
      <link name="body">
        <collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision>
        <collision><origin xyz="0.003 0.055 0"/><geometry><box size="0.00015 0.06 0.04"/></geometry></collision>
      </link>
      <joint name="turn" type="revolute"><parent link="base"/><child link="body"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
    </robot>"#;
    std::fs::write(dir.join("fin.urdf"), urdf).unwrap();
    let options = RobotOptions {
        spheres: SphereOptions {
            budget: 256,
            ..Default::default()
        },
        ..Default::default()
    };
    let fin = Robot::load(dir.join("fin.urdf"), &options).unwrap();
    let spheres: usize = fin.collision_model().spheres.values().map(Vec::len).sum();
    assert!(spheres < 256, "the budget was the limit: {spheres} spheres");
    let body = &fin.qualify().unwrap().links[0];
    assert!(
        body.overhang <= SphereOptions::default().tolerance + 1e-5,
        "the finned box reaches {:.2} mm past its geometry",
        body.overhang * 1e3
    );
    // At a seed whose box takes 18 refinements of its cover.
    let dir = scratch("stand");
    std::fs::write(dir.join("stand.urdf"), STAND).unwrap();
    let options = RobotOptions {
        spheres: SphereOptions {
            budget: 1024,
            tolerance: 0.02,
            rng_seed: 13,
            ..Default::default()
        },
        ..Default::default()
    };
    let stand = Robot::load(dir.join("stand.urdf"), &options).unwrap();
    let spheres: usize = stand.collision_model().spheres.values().map(Vec::len).sum();
    assert!(
        spheres < 1024,
        "the budget was the limit: {spheres} spheres"
    );
    for l in stand.qualify().unwrap().links {
        assert!(
            l.overhang <= 0.02 + 1e-5,
            "the stand's {} reaches {:.1} mm past its geometry",
            l.link,
            l.overhang * 1e3
        );
    }
    // Budgets that do not: the jaws held to 1 mm reach less far with every doubling.
    let mut before = f32::INFINITY;
    for budget in [128, 256, 512] {
        let (_, links) = fit(budget, 0.001);
        let jaw = links
            .iter()
            .filter(|l| jaws.contains(&l.link.as_str()))
            .map(|l| l.overhang)
            .fold(0.0, f32::max);
        assert!(
            jaw < 0.9 * before,
            "at {budget} spheres the jaws reach {:.2} mm, at half as many {:.2} mm",
            jaw * 1e3,
            before * 1e3
        );
        before = jaw;
    }
}

/// Where the budget holds the tolerances back, no link's spheres reach farther than twice its
/// tolerance scaled by what the fit reports: on two wide thin plates, whose few cover samples lie
/// centimeters apart, spheres grown to take in the dense samples between them would reach farther.
#[test]
fn spheres_reach_at_most_twice_their_scaled_tolerance() {
    let dir = scratch("plates");
    let urdf = r#"<robot name="plates">
      <link name="base"/>
      <link name="wide"><collision><geometry><box size="0.9 0.6 0.004"/></geometry></collision></link>
      <link name="tall"><collision><origin xyz="0 0 0.3"/><geometry><box size="0.6 0.004 0.3"/></geometry></collision></link>
      <joint name="turn" type="revolute"><parent link="base"/><child link="wide"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
      <joint name="tilt" type="revolute"><parent link="wide"/><child link="tall"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
    </robot>"#;
    std::fs::write(dir.join("plates.urdf"), urdf).unwrap();
    for budget in [256, 1024] {
        let options = RobotOptions {
            spheres: SphereOptions {
                budget,
                ..Default::default()
            },
            ..Default::default()
        };
        let robot = Robot::load(dir.join("plates.urdf"), &options).unwrap();
        let source = robot.collision_model().source.clone().unwrap();
        let scale: f32 = source
            .split("scaled by ")
            .nth(1)
            .and_then(|s| s.split(' ').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("no scale in {source:?}"));
        assert!(scale > 1.0, "the budget reached the tolerances: {source}");
        let bound = 2.0 * scale * SphereOptions::default().tolerance;
        for l in robot.qualify().unwrap().links {
            assert!(
                l.overhang <= 1.01 * bound,
                "at {budget} spheres {} reaches {:.1} mm, past twice {scale} times its tolerance",
                l.link,
                l.overhang * 1e3
            );
        }
    }
}

/// Where the budget reaches the tolerances but cannot hold every link within its own, what it has
/// left past the loosest covers holds the spheres tighter, and the fit's `source` gives the largest
/// of each link's reach over its own tolerance: on a stand, whose wide faces take many spheres to
/// hold.
#[test]
fn a_budget_past_the_tolerances_holds_spheres_tighter() {
    let dir = scratch("stand-budget");
    std::fs::write(dir.join("stand.urdf"), STAND).unwrap();
    // Spheres, each link's reach over its tolerance, and the source.
    let fit = |budget: usize, feet: f32, top: f32| {
        let options = RobotOptions {
            spheres: SphereOptions {
                budget,
                ..Default::default()
            },
            sphere_tolerances: [("feet".to_string(), feet), ("box".to_string(), top)].into(),
            ..Default::default()
        };
        let robot = Robot::load(dir.join("stand.urdf"), &options).unwrap();
        let spheres: usize = robot.collision_model().spheres.values().map(Vec::len).sum();
        assert!(
            spheres <= budget,
            "{spheres} spheres over a budget of {budget}"
        );
        let reach: Vec<f32> = robot
            .qualify()
            .unwrap()
            .links
            .iter()
            .map(|l| l.overhang / options.sphere_tolerances[&l.link])
            .collect();
        (
            spheres,
            reach,
            robot.collision_model().source.clone().unwrap(),
        )
    };
    let farthest = |reach: &[f32]| reach.iter().copied().fold(0.0, f32::max);
    let (fewer, before, source) = fit(384, 0.02, 0.02);
    assert!(
        source.contains("scaled by 1.00"),
        "the budget held the tolerances back: {source}"
    );
    let (spheres, after, _) = fit(512, 0.02, 0.02);
    // The least growth 512 spheres afford is about 1.21.
    assert!(
        farthest(&after) > 1.05 && farthest(&after) < 1.3,
        "512 spheres reach {after:?} times their tolerances"
    );
    assert!(
        spheres > fewer && farthest(&after) < 0.9 * farthest(&before),
        "{spheres} spheres reach {after:?} times their tolerances, {fewer} reach {before:?}"
    );
    // Each link its own tolerance, one way round and the other: the source names the larger.
    for (feet, top) in [(0.04, 0.01), (0.01, 0.04)] {
        let (_, reach, source) = fit(512, feet, top);
        let relative: f32 = source
            .split("and up to ")
            .nth(1)
            .and_then(|s| s.split(' ').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("no relative overhang in {source:?}"));
        assert!(
            (reach[0] - reach[1]).abs() > 0.05 && (relative - farthest(&reach)).abs() < 0.01,
            "{source}, where the links reach {reach:?} times their tolerances"
        );
    }
}

#[test]
fn links_given_a_tighter_tolerance_get_spheres_that_reach_less_far() {
    let jaws = ["gripper_link", "moving_jaw_so101_v1_link"];
    let fit = |tolerances: &[(&str, f32)]| {
        let options = RobotOptions {
            spheres: SphereOptions {
                budget: 256,
                ..Default::default()
            },
            sphere_tolerances: tolerances
                .iter()
                .map(|&(l, t)| (l.to_string(), t))
                .collect(),
            ..Default::default()
        };
        let robot = Robot::load(common::asset(SO101), &options).unwrap();
        let spheres: usize = robot.collision_model().spheres.values().map(Vec::len).sum();
        assert!(spheres <= 256, "{spheres} spheres");
        robot.qualify().unwrap().links
    };
    let even = fit(&[]);
    let tight = fit(&jaws.map(|l| (l, 0.002)));
    let reach = |links: &[batchplan::LinkQualification], of_jaws: bool| {
        links
            .iter()
            .filter(|l| jaws.contains(&l.link.as_str()) == of_jaws)
            .map(|l| l.overhang)
            .collect::<Vec<f32>>()
    };
    let most = |v: Vec<f32>| v.into_iter().fold(0.0, f32::max);
    let least = |v: Vec<f32>| v.into_iter().fold(f32::INFINITY, f32::min);
    // The jaws reach less far than any other link, and than they did, at the arm's cost.
    assert!(
        most(reach(&tight, true)) < 0.75 * least(reach(&tight, false)),
        "{tight:?}"
    );
    assert!(
        most(reach(&tight, true)) < least(reach(&even, true)),
        "{even:?} {tight:?}"
    );
    assert!(most(reach(&tight, false)) > most(reach(&even, false)));
    // Tolerances name links of the robot, and are positive.
    for (link, tolerance) in [("jaw", 0.002), ("gripper_link", 0.0)] {
        let options = RobotOptions {
            sphere_tolerances: [(link.to_string(), tolerance)].into(),
            ..Default::default()
        };
        let err = Robot::load(common::asset(SO101), &options).unwrap_err();
        assert!(
            matches!(err, Error::Load { .. }) && err.to_string().contains(link),
            "{err}"
        );
    }
}

/// A part's base places its part by its rotation made unit, and is checked as other poses are:
/// a rotation a thousandth off unit puts the arm where the unit one does, and one that is not a
/// rotation is refused.
#[test]
fn a_parts_base_turns_its_part_and_is_checked() {
    let one = common::panda_options();
    let assembled = |base: Pose| {
        let part = Part {
            path: common::asset("franka/franka_panda.urdf").into(),
            prefix: "left_".into(),
            base,
        };
        let locks = one.lock_joints.iter();
        let options = RobotOptions {
            lock_joints: locks.map(|(j, v)| (format!("left_{j}"), *v)).collect(),
            collision_model: Some(CollisionModel::assemble(&[(
                "left_",
                one.collision_model.as_ref().unwrap(),
            )])),
            ..Default::default()
        };
        Robot::assemble(&[part], &options).map(|robot| robot.spheres(robot.default_q()))
    };
    let (at, turn) = (Vec3::new(0.0, 0.35, 0.0), Quat::from_rotation_z(2.5));
    let unit = assembled(Pose::new(at, turn)).unwrap();
    let reach = unit
        .iter()
        .map(|s| Vec3::new(s[0], s[1], s[2]).distance(at));
    assert!(
        reach.fold(0.0, f32::max) > 0.5,
        "the arm reaches far from its base"
    );
    for norm in [0.9991f32, 1.0009] {
        let scaled = assembled(Pose::new(at, turn * norm)).unwrap();
        for (u, s) in unit.iter().zip(&scaled) {
            let apart = u.iter().zip(s).map(|(a, b)| (a - b).abs());
            assert!(
                apart.fold(0.0, f32::max) < 2e-5,
                "norm {norm}: {u:?} became {s:?}"
            );
        }
    }
    let bad = [
        Pose::new(at, turn * 2.0),
        Pose::new(at, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
        Pose::new(Vec3::NAN, turn),
    ];
    for base in bad {
        let refused = assembled(base);
        assert!(
            matches!(&refused, Err(Error::Input(m)) if m.contains("base")),
            "{base:?}: {refused:?}"
        );
    }
}

/// How deep `p` lies inside a box of `half` extents about `center`: negative outside.
fn depth_in_box(p: Vec3, center: Vec3, half: Vec3) -> f32 {
    let q = (p - center).abs() - half;
    -(q.max(Vec3::ZERO).length() + q.max_element().min(0.0))
}

#[test]
fn spheres_held_to_an_overhang_reach_no_farther_and_the_gap_says_what_they_leave() {
    // The stand's two boxes, as the URDF places them.
    let boxes = [
        ("feet", Vec3::ZERO, Vec3::new(0.385, 0.35, 0.155)),
        ("box", Vec3::new(0.0, 0.0, 0.5), Vec3::new(0.11, 0.2, 0.265)),
    ];
    let fit = |max_overhang: Option<f32>| {
        let options = RobotOptions {
            spheres: SphereOptions {
                budget: 48,
                max_overhang,
                ..Default::default()
            },
            ..Default::default()
        };
        Robot::from_urdf(STAND, &options).unwrap()
    };
    let free = fit(None).qualify().unwrap().links;
    for limit in [0.0, 0.002, 0.02] {
        let robot = fit(Some(limit));
        let model = robot.collision_model();
        let count: usize = model.spheres.values().map(Vec::len).sum();
        assert!(count > 0 && count <= 48, "{count} spheres");
        let source = model.source.as_deref().unwrap();
        let held = format!("each held to {:.1} mm", limit * 1e3);
        assert!(source.contains(&held), "{source}");
        let links = robot.qualify().unwrap().links;
        // The fit's own account of how far its spheres reach is what they reach, held: the limit.
        let reported: f32 = source
            .split("up to ")
            .nth(1)
            .and_then(|s| s.split(' ').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("no reach in {source:?}"));
        let largest = links.iter().map(|l| l.overhang).fold(0.0, f32::max);
        assert!(
            (largest - limit).abs() < 1e-5 && (reported - limit * 1e3).abs() < 0.06,
            "{largest} m against {source}"
        );
        for ((&(name, center, half), l), free) in boxes.iter().zip(&links).zip(&free) {
            assert_eq!((l.link.as_str(), free.link.as_str()), (name, name));
            // The limit is one the budget does not reach by itself.
            assert!(free.overhang > limit + 0.005, "{free:?}");
            let spheres = &model.spheres[name];
            assert!(!spheres.is_empty());
            for &[x, y, z, radius] in spheres {
                let reach = radius - depth_in_box(Vec3::new(x, y, z), center, half);
                assert!(
                    radius > 0.0 && reach <= limit + 1e-6,
                    "{name}: a sphere reaches {reach} past the box, limit {limit}"
                );
            }
            assert!(l.overhang <= limit + 1e-6, "{l:?}");
            // How far the box's corners and its faces' centers lie outside the spheres, at most:
            // the gap is measured on finer samples, so it is at least as deep, and not far off.
            let outside = |p: Vec3| {
                spheres
                    .iter()
                    .map(|s| p.distance(Vec3::new(s[0], s[1], s[2])) - s[3])
                    .fold(f32::INFINITY, f32::min)
            };
            let marks = (0..27).map(|k| {
                let at = Vec3::new((k % 3) as f32, (k / 3 % 3) as f32, (k / 9) as f32) - 1.0;
                center + at * half
            });
            let deepest = marks
                .filter(|&p| depth_in_box(p, center, half).abs() < 1e-6)
                .map(outside)
                .fold(0.0, f32::max);
            assert!(
                deepest > 0.01 && l.gap >= deepest - 1e-6 && l.gap < deepest + 0.01,
                "{name}: gap {} against {deepest} at the corners",
                l.gap
            );
            assert!(l.gap > free.gap + 0.01 && l.uncovered > 0.5, "{l:?}");
        }
    }
    // Real meshes, each link held: the SO-101.
    let limit = 0.002;
    let options = RobotOptions {
        spheres: SphereOptions {
            budget: 256,
            max_overhang: Some(limit),
            ..Default::default()
        },
        ..Default::default()
    };
    let free = load(SO101).qualify().unwrap().links;
    let robot = Robot::load(common::asset(SO101), &options).unwrap();
    let links = robot.qualify().unwrap().links;
    assert_eq!(links.len(), free.len());
    assert!(free.iter().all(|l| l.overhang > 2.0 * limit), "{free:?}");
    for l in &links {
        assert!(l.overhang <= limit + 1e-6 && l.gap > 0.0, "{l:?}");
    }
}

/// Boxes (center, half extents) written as one OBJ in `dir`, the first without its faces in
/// `dropped` (of -x, +x, -y, +y, -z, +z) and with those in `twice` given twice. Returns its path.
fn boxes_obj(
    dir: &Path,
    name: &str,
    boxes: &[(Vec3, Vec3)],
    dropped: &[usize],
    twice: &[usize],
) -> PathBuf {
    // Each face's corners, counterclockwise seen from outside.
    let faces = [
        [1, 5, 7, 3],
        [2, 4, 8, 6],
        [1, 2, 6, 5],
        [3, 7, 8, 4],
        [1, 3, 4, 2],
        [5, 6, 8, 7],
    ];
    let mut lines = vec![];
    for (n, &(center, half)) in boxes.iter().enumerate() {
        for k in 0..8 {
            let sign = |bit: usize| if k >> bit & 1 == 1 { 1.0 } else { -1.0 };
            let v = center + Vec3::new(sign(0), sign(1), sign(2)) * half;
            lines.push(format!("v {} {} {}", v.x, v.y, v.z));
        }
        for (k, face) in faces.into_iter().enumerate() {
            let [a, b, c, d] = face.map(|corner| corner + 8 * n);
            let times = match n {
                0 if dropped.contains(&k) => 0,
                0 if twice.contains(&k) => 2,
                _ => 1,
            };
            for _ in 0..times {
                lines.push(format!("f {a} {b} {c}\nf {a} {c} {d}"));
            }
        }
    }
    let path = dir.join(name);
    std::fs::write(&path, lines.join("\n")).unwrap();
    path
}

/// A robot of one link, `body`, with `collision` inside its `<collision>` element.
fn one_link(collision: &str) -> String {
    format!(
        r#"<robot name="held"><link name="base"/>
          <link name="body"><collision>{collision}</collision></link>
          <joint name="turn" type="revolute"><parent link="base"/><child link="body"/>
            <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
        </robot>"#
    )
}

/// [`one_link`] with the mesh file at `path` as its geometry.
fn meshed(path: &Path) -> String {
    one_link(&format!(
        r#"<geometry><mesh filename="file://{}"/></geometry>"#,
        path.display()
    ))
}

/// A link of two closed shells in one mesh, a plate 20.9 × 6.1 cm over another with 4 cm between
/// them: the space between is outside the link, at its middle and under the plates' edges and
/// the diagonals their faces are split along.
#[test]
fn the_space_between_two_shells_of_a_link_is_outside_it() {
    let half = Vec3::new(0.1045, 0.0305, 0.01);
    let plates = [(Vec3::Z * 0.03, half), (Vec3::Z * -0.03, half)];
    let depth = |p: Vec3| {
        let of = |&(center, half): &(Vec3, Vec3)| depth_in_box(p, center, half);
        plates.iter().map(of).fold(f32::NEG_INFINITY, f32::max)
    };
    let dir = scratch("two-shells");
    let urdf = meshed(&boxes_obj(&dir, "plates.obj", &plates, &[], &[]));
    let given = |sphere: [f32; 4], max_overhang: Option<f32>| RobotOptions {
        collision_model: Some(CollisionModel {
            spheres: BTreeMap::from([("body".to_string(), vec![sphere])]),
            ..Default::default()
        }),
        spheres: SphereOptions {
            max_overhang,
            ..Default::default()
        },
        ..Default::default()
    };
    for (x, y) in [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (-0.25, 0.25),
    ] {
        let center = Vec3::new(x * half.x, y * half.y, 0.0);
        let sphere = [center.x, center.y, 0.0, 0.0196];
        let refused = Robot::from_urdf(&urdf, &given(sphere, Some(0.0))).map(|_| ());
        assert!(
            matches!(&refused, Err(Error::Input(m)) if m.contains("its center 20.000 mm outside")),
            "{center}: {refused:?}"
        );
        let robot = Robot::from_urdf(&urdf, &given(sphere, None)).unwrap();
        let overhang = robot.qualify().unwrap().links[0].overhang;
        assert!((overhang - 0.0396).abs() < 1e-5, "{center}: {overhang}");
    }
    // Fitted, every sphere's center is in a plate, and held to nothing past them, all of it.
    for max_overhang in [None, Some(0.0)] {
        let options = RobotOptions {
            collision_model: None,
            ..given([0.0; 4], max_overhang)
        };
        let robot = Robot::from_urdf(&urdf, &options).unwrap();
        let spheres = &robot.collision_model().spheres["body"];
        assert!(spheres.len() > 8, "{} spheres", spheres.len());
        for &[x, y, z, radius] in spheres {
            let deep = depth(Vec3::new(x, y, z));
            assert!(deep > 0.0, "a center {} mm outside the plates", -deep * 1e3);
            if max_overhang.is_some() {
                assert!(radius <= deep + 1e-6, "{radius} at a depth of {deep}");
            }
        }
    }
}

/// Two boxes that overlap, in one mesh: inside is their union. A sphere in the first, beside the
/// face of the second that is buried in it, is inside the link, where the side of that nearest
/// face would have it outside.
#[test]
fn shells_that_overlap_in_one_mesh_are_their_union() {
    let half = Vec3::new(0.05, 0.04, 0.03);
    let boxes = [(Vec3::ZERO, half), (Vec3::X * 0.06, half)];
    let dir = scratch("overlap");
    let urdf = meshed(&boxes_obj(&dir, "overlap.obj", &boxes, &[], &[]));
    let depth = |p: Vec3| {
        let of = |&(center, half): &(Vec3, Vec3)| depth_in_box(p, center, half);
        boxes.iter().map(of).fold(f32::NEG_INFINITY, f32::max)
    };
    let given = |sphere: [f32; 4], max_overhang: Option<f32>| RobotOptions {
        collision_model: Some(CollisionModel {
            spheres: BTreeMap::from([("body".to_string(), vec![sphere])]),
            ..Default::default()
        }),
        spheres: SphereOptions {
            max_overhang,
            ..Default::default()
        },
        ..Default::default()
    };
    // 5 mm from the second box's buried face, at x = 1 cm, and 3 cm deep in the first.
    let beside = [0.005, 0.0, 0.0, 0.004];
    let robot = Robot::from_urdf(&urdf, &given(beside, None)).unwrap();
    let overhang = robot.qualify().unwrap().links[0].overhang;
    assert_eq!(overhang, 0.0, "a sphere inside the first box");
    Robot::from_urdf(&urdf, &given(beside, Some(0.0))).unwrap();
    // In both at once, and outside both: held by the nearest face, refused past it.
    Robot::from_urdf(&urdf, &given([0.03, 0.0, 0.0, 0.02], Some(0.0))).unwrap();
    let refused = Robot::from_urdf(&urdf, &given([0.12, 0.0, 0.0, 0.005], Some(0.0))).map(|_| ());
    assert!(
        matches!(&refused, Err(Error::Input(m)) if m.contains("its center 10.000 mm outside")),
        "{refused:?}"
    );
    // Fitted, the spheres cover the union, and held to nothing past it, stay in it.
    for max_overhang in [None, Some(0.0)] {
        let options = RobotOptions {
            collision_model: None,
            ..given([0.0; 4], max_overhang)
        };
        let robot = Robot::from_urdf(&urdf, &options).unwrap();
        let spheres = &robot.collision_model().spheres["body"];
        for &[x, y, z, radius] in spheres {
            let deep = depth(Vec3::new(x, y, z));
            assert!(
                deep > -1e-6,
                "a center {} mm outside the boxes",
                -deep * 1e3
            );
            if max_overhang.is_some() {
                assert!(radius <= deep + 1e-6, "{radius} at a depth of {deep}");
            }
        }
    }
}

#[test]
fn a_limit_on_overhang_holds_supplied_models_and_names_what_it_refuses() {
    let half = Vec3::new(0.05, 0.04, 0.03);
    let boxed = one_link(r#"<geometry><box size="0.1 0.08 0.06"/></geometry>"#);
    let limit = 0.002;
    let options = |spheres: Vec<[f32; 4]>, link: &str| RobotOptions {
        collision_model: Some(CollisionModel {
            spheres: BTreeMap::from([(link.to_string(), spheres)]),
            ..Default::default()
        }),
        spheres: SphereOptions {
            max_overhang: Some(limit),
            ..Default::default()
        },
        ..Default::default()
    };
    let refused = |urdf: &str, o: &RobotOptions| match Robot::from_urdf(urdf, o) {
        Err(Error::Input(message)) => message,
        other => panic!("expected an input error, got {other:?}"),
    };
    // Inside, and past a face by the limit: held. A hair more, or a center outside: refused,
    // the sphere named with how far it reaches.
    Robot::from_urdf(&boxed, &options(vec![[0.0, 0.0, 0.0, 0.03]], "body")).unwrap();
    Robot::from_urdf(&boxed, &options(vec![[0.0, 0.0, 0.0, 0.032]], "body")).unwrap();
    let message = refused(
        &boxed,
        &options(
            vec![[0.0, 0.0, 0.0, 0.03], [0.0, 0.0, 0.01, 0.0225]],
            "body",
        ),
    );
    assert!(
        message.contains("link 'body': sphere 1,")
            && message.contains("reaches 2.500 mm past the geometry")
            && message.contains("20.000 mm deep")
            && message.contains("`max_overhang` allows 2.000 mm"),
        "{message}"
    );
    let message = refused(&boxed, &options(vec![[0.06, 0.0, 0.0, 0.001]], "body"));
    assert!(
        message.contains("sphere 0,") && message.contains("its center 10.000 mm outside"),
        "{message}"
    );
    let message = refused(&boxed, &options(vec![[0.0, 0.0, 0.0, f32::NAN]], "body"));
    assert!(message.contains("sphere 0,"), "{message}");
    // Without a limit the same model loads, as it always has.
    let mut free = options(vec![[0.0, 0.0, 0.01, 0.0225]], "body");
    free.spheres.max_overhang = None;
    Robot::from_urdf(&boxed, &free).unwrap();
    // Spheres on a link without geometry have nothing to be held to.
    let message = refused(&boxed, &options(vec![[0.0, 0.0, 0.0, 0.01]], "base"));
    assert!(
        message.contains("link 'base' has spheres") && message.contains("no geometry"),
        "{message}"
    );

    // A fit under the limit is a model the limit holds, here and with the link's geometry and
    // the spheres both moved by one pose, which rounds each center anew.
    let fit = RobotOptions {
        collision_model: None,
        ..options(vec![], "body")
    };
    let fitted = Robot::from_urdf(&boxed, &fit).unwrap();
    let spheres = fitted.collision_model().spheres["body"].clone();
    let at_the_limit = spheres.iter().filter(|&&[x, y, z, radius]| {
        (radius - depth_in_box(Vec3::new(x, y, z), Vec3::ZERO, half) - limit).abs() < 1e-6
    });
    assert!(at_the_limit.count() > 8, "few spheres reach the limit");
    Robot::from_urdf(&boxed, &options(spheres.clone(), "body")).unwrap();
    let mut rng = Rng::new(5);
    for _ in 0..20 {
        let rpy = Vec3::new(
            rng.range(-3.0, 3.0),
            rng.range(-1.5, 1.5),
            rng.range(-3.0, 3.0),
        );
        let xyz = Vec3::new(
            rng.range(-0.5, 0.5),
            rng.range(-0.5, 0.5),
            rng.range(-0.5, 0.5),
        );
        // A URDF's rpy turns about fixed x, then y, then z.
        let turn = Quat::from_rotation_z(rpy.z)
            * Quat::from_rotation_y(rpy.y)
            * Quat::from_rotation_x(rpy.x);
        let moved = one_link(&format!(
            r#"<origin xyz="{} {} {}" rpy="{} {} {}"/><geometry><box size="0.1 0.08 0.06"/></geometry>"#,
            xyz.x, xyz.y, xyz.z, rpy.x, rpy.y, rpy.z
        ));
        let placed = spheres
            .iter()
            .map(|&[x, y, z, radius]| {
                let c = xyz + turn * Vec3::new(x, y, z);
                [c.x, c.y, c.z, radius]
            })
            .collect();
        Robot::from_urdf(&moved, &options(placed, "body")).unwrap();
    }

    // Geometry that is not closed has no inside to hold spheres in: refused with the link and
    // its open edges, whether the model is given or fitted, and so is a face given twice, which
    // leaves its edges with a triangle too many.
    let dir = scratch("overhang");
    let whole = [(Vec3::ZERO, half)];
    let closed = meshed(&boxes_obj(&dir, "closed.obj", &whole, &[], &[]));
    let open = meshed(&boxes_obj(&dir, "open.obj", &whole, &[5], &[]));
    let doubled = meshed(&boxes_obj(&dir, "doubled.obj", &whole, &[], &[1]));
    let inside = options(vec![[0.0, 0.0, 0.0, 0.03]], "body");
    Robot::from_urdf(&closed, &inside).unwrap();
    for o in [&inside, &fit] {
        let message = refused(&open, o);
        assert!(
            message.contains("link 'body': mesh 0 of 1 is not closed: 4 of its 17 edges"),
            "{message}"
        );
        let message = refused(&doubled, o);
        assert!(
            message.contains("link 'body': mesh 0 of 1 is not closed: 4 of its 18 edges"),
            "{message}"
        );
    }
    // The whole box given twice is closed: it winds around its inside twice.
    let twice = meshed(&boxes_obj(
        &dir,
        "twice.obj",
        &[whole[0], whole[0]],
        &[],
        &[],
    ));
    Robot::from_urdf(&twice, &inside).unwrap();
    Robot::from_urdf(&twice, &fit).unwrap();
    // The spheres a fit under the limit gives `urdf`'s link, as (center, radius).
    let fitted = |urdf: &str| -> Vec<(Vec3, f32)> {
        let robot = Robot::from_urdf(urdf, &fit).unwrap();
        let spheres = &robot.collision_model().spheres["body"];
        assert!(!spheres.is_empty());
        let sphere = |s: &[f32; 4]| (Vec3::new(s[0], s[1], s[2]), s[3]);
        spheres.iter().map(sphere).collect()
    };
    // A box whose every face is given both ways, as a mesh drawn from both sides is, is closed
    // and winds around nothing, whichever copy comes first: a sphere in it is outside, and
    // fitted spheres sit on its faces, no larger than the limit.
    let text = std::fs::read_to_string(dir.join("closed.obj")).unwrap();
    // Backward from its last corner, so the copies start at different ones.
    let reversed = |line: &str| {
        let corners: Vec<&str> = line.split(' ').collect();
        format!("f {} {} {}", corners[3], corners[2], corners[1])
    };
    let faces = text.lines().filter(|line| line.starts_with('f'));
    let (forward, backward): (Vec<String>, Vec<String>) =
        faces.map(|line| (line.to_string(), reversed(line))).unzip();
    let corners: Vec<&str> = text.lines().filter(|line| line.starts_with('v')).collect();
    for (name, faces) in [
        ("both.obj", [forward.clone(), backward.clone()].concat()),
        ("both-reversed-first.obj", [backward, forward].concat()),
    ] {
        let path = dir.join(name);
        std::fs::write(&path, [corners.join("\n"), faces.join("\n")].join("\n")).unwrap();
        let message = refused(&meshed(&path), &inside);
        assert!(
            message.contains("sphere 0,") && message.contains("its center 30.000 mm outside"),
            "{name}: {message}"
        );
        for (center, radius) in fitted(&meshed(&path)) {
            let off = depth_in_box(center, Vec3::ZERO, half).abs();
            assert!(
                off < 1e-5 && radius <= limit + 1e-6,
                "{name}: {radius} at {center}"
            );
        }
    }
    // Two boxes that share a face are one box twice as tall: a sphere in the upper one is held
    // by its depth there, to the face they share at the nearest, and fitted spheres keep within
    // the limit of the whole.
    let stacked = [(Vec3::ZERO, half), (Vec3::Z * 2.0 * half.z, half)];
    let stacked = meshed(&boxes_obj(&dir, "stacked.obj", &stacked, &[], &[]));
    Robot::from_urdf(&stacked, &inside).unwrap();
    Robot::from_urdf(&stacked, &options(vec![[0.0, 0.0, 0.05, 0.022]], "body")).unwrap();
    let message = refused(&stacked, &options(vec![[0.0, 0.0, 0.05, 0.023]], "body"));
    assert!(message.contains("its center 20.000 mm deep"), "{message}");
    let tall = Vec3::new(half.x, half.y, 2.0 * half.z);
    for (center, radius) in fitted(&stacked) {
        let reach = radius - depth_in_box(center, Vec3::Z * half.z, tall);
        assert!(
            reach <= limit + 1e-6,
            "{radius} at {center} reaches {reach}"
        );
    }
    // A cube beside the box is its own shell: a sphere beside it is outside, one at its middle
    // is in it, and fitted spheres keep to the two. Facing inward it is as solid, and without a
    // limit is fitted so, every center in the box or the cube; under a limit it is refused, since
    // a shell that faces inward gets its samples' candidates from the wrong side.
    let apart = [(Vec3::ZERO, half), (Vec3::X * 0.2, Vec3::splat(0.02))];
    let text = std::fs::read_to_string(boxes_obj(&dir, "apart.obj", &apart, &[], &[])).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let faces = lines.iter().filter(|line| line.starts_with('f')).count();
    let mut seen = 0;
    let inverted: Vec<String> = lines
        .iter()
        .map(|line| {
            seen += usize::from(line.starts_with('f'));
            // The cube's faces are the second half.
            if line.starts_with('f') && seen > faces / 2 {
                reversed(line)
            } else {
                line.to_string()
            }
        })
        .collect();
    let path = dir.join("inverted.obj");
    std::fs::write(&path, inverted.join("\n")).unwrap();
    let depth = |p: Vec3| {
        let of = |&(center, half): &(Vec3, Vec3)| depth_in_box(p, center, half);
        apart.iter().map(of).fold(f32::NEG_INFINITY, f32::max)
    };
    let beside = options(vec![[0.225, 0.0, 0.0, 0.005]], "body");
    let within = options(vec![[0.2, 0.0, 0.0, 0.02]], "body");
    let outward = meshed(&dir.join("apart.obj"));
    let message = refused(&outward, &beside);
    assert!(message.contains("its center 5.000 mm outside"), "{message}");
    Robot::from_urdf(&outward, &within).unwrap();
    let spheres = fitted(&outward);
    for &(center, radius) in &spheres {
        let reach = radius - depth(center);
        assert!(
            reach <= limit + 1e-6,
            "{radius} at {center} reaches {reach}"
        );
    }
    let in_the_cube = |&&(center, _): &&(Vec3, f32)| center.x > 0.1;
    assert!(
        spheres.iter().filter(in_the_cube).count() > 2,
        "the cube has spheres"
    );
    for o in [&beside, &within, &fit] {
        let message = refused(&meshed(&path), o);
        assert!(
            message.contains("link 'body': mesh 0 of 1 has 1 of its 2 shells facing inward"),
            "{message}"
        );
    }
    for urdf in [outward, meshed(&path)] {
        let mut free = fit.clone();
        free.spheres.max_overhang = None;
        let robot = Robot::from_urdf(&urdf, &free).unwrap();
        let spheres = &robot.collision_model().spheres["body"];
        assert!(spheres.iter().any(|s| s[0] > 0.1), "the cube has spheres");
        for &[x, y, z, _] in spheres {
            let deep = depth(Vec3::new(x, y, z));
            assert!(deep > -1e-6, "a center {} mm outside", -deep * 1e3);
        }
        // Measured without a limit, a sphere beside the cube reaches its radius and the 5 mm.
        free.collision_model = beside.collision_model.clone();
        let overhang = Robot::from_urdf(&urdf, &free)
            .unwrap()
            .qualify()
            .unwrap()
            .links[0]
            .overhang;
        assert!((overhang - 0.01).abs() < 1e-5, "{overhang}");
    }
    // A sphere farther away than its distance can be told to a digit is outside all the same.
    let message = refused(&closed, &options(vec![[1e20, 0.0, 0.0, 0.01]], "body"));
    assert!(
        message.contains("sphere 0,") && message.contains("outside"),
        "{message}"
    );
    let mut free = fit;
    free.spheres.max_overhang = None;
    Robot::from_urdf(&open, &free).unwrap();
    Robot::from_urdf(&doubled, &free).unwrap();
    // A box open at one end along x has no surface that winds around its middle, and is fitted
    // by the side of its nearest face, as thick as the box: 6 cm.
    let open_along_x = meshed(&boxes_obj(&dir, "open-x.obj", &whole, &[1], &[]));
    let robot = Robot::from_urdf(&open_along_x, &free).unwrap();
    let thickness = robot.qualify().unwrap().links[0].thickness;
    assert!((thickness - 2.0 * half.z).abs() < 0.012, "{thickness}");

    // The options are checked whether spheres are fitted or given, as input, by every loader.
    let path = dir.join("stand.urdf");
    std::fs::write(&path, STAND).unwrap();
    let bad = [
        SphereOptions {
            max_overhang: Some(-0.001),
            ..Default::default()
        },
        SphereOptions {
            max_overhang: Some(f32::NAN),
            ..Default::default()
        },
        SphereOptions {
            max_overhang: Some(f32::INFINITY),
            ..Default::default()
        },
        SphereOptions {
            budget: 0,
            ..Default::default()
        },
    ];
    for spheres in bad {
        for collision_model in [None, Some(CollisionModel::default())] {
            let o = RobotOptions {
                collision_model,
                spheres,
                ..Default::default()
            };
            let part = Part {
                path: path.clone(),
                prefix: String::new(),
                base: Pose::IDENTITY,
            };
            for result in [
                Robot::load(&path, &o),
                Robot::from_urdf(STAND, &o),
                Robot::assemble(&[part], &o),
            ] {
                assert!(
                    matches!(&result, Err(Error::Input(m)) if m.contains("max_overhang") || m.contains("budget")),
                    "{spheres:?}: {result:?}"
                );
            }
        }
    }
}

#[test]
fn a_link_too_thin_for_a_sphere_inside_it_is_refused_by_name() {
    // A plate 0.05 mm thick: no center the fit can place lies inside it.
    let urdf = r#"<robot name="plate"><link name="base"/>
      <link name="plate"><collision><geometry><box size="0.1 0.1 0.00005"/></geometry></collision></link>
      <joint name="turn" type="revolute"><parent link="base"/><child link="plate"/>
        <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
    </robot>"#;
    let options = |max_overhang: Option<f32>| RobotOptions {
        spheres: SphereOptions {
            max_overhang,
            ..Default::default()
        },
        ..Default::default()
    };
    let refused =
        Robot::from_urdf(urdf, &options(Some(0.0))).map(|robot| robot.collision_model().clone());
    assert!(
        matches!(&refused, Err(Error::Input(m)) if m.contains("link 'plate': no sphere fits within `max_overhang` (0 m)")),
        "{refused:?}"
    );
    // A ball of no radius has no surface at all, and no limit would help it.
    let point = urdf.replace(
        r#"<box size="0.1 0.1 0.00005"/>"#,
        r#"<sphere radius="0"/>"#,
    );
    assert_ne!(point, urdf);
    let refused = Robot::from_urdf(&point, &options(Some(0.002))).map(|_| ());
    assert!(
        matches!(&refused, Err(Error::Input(m)) if m.contains("link 'plate': mesh 0 of 1 has no surface")),
        "{refused:?}"
    );
    // With room past it, or without a limit, it is covered.
    for max_overhang in [Some(0.002), None] {
        let robot = Robot::from_urdf(urdf, &options(max_overhang)).unwrap();
        let l = &robot.qualify().unwrap().links[0];
        assert!(
            l.spheres > 0 && l.overhang <= max_overhang.unwrap_or(1.0) + 1e-6,
            "{l:?}"
        );
    }
}

#[test]
fn an_outer_reach_bound_keeps_fk_poses_and_accounts_for_ik_tolerance() {
    let options = batchplan::IkOptions::default();
    let mut rng = Rng::new(407);
    for robot in [
        common::panda().unwrap(),
        load(UR5E),
        common::two_pandas().unwrap(),
    ] {
        for hand in 0..robot.hands() {
            let bound = robot.hand_reach(hand).expect("revolute chain");
            for _ in 0..128 {
                let pose = robot.ee_pose(&random_q(&robot, &mut rng), hand);
                assert!(!bound.excludes(pose, &options));
                let far = Pose::new(pose.position + Vec3::X * 10.0, pose.rotation);
                assert!(bound.excludes(far, &options));
            }
        }
    }

    let dir = scratch("outer-reach");
    let path = dir.join("arm.urdf");
    let arm = r#"<robot name="arm">
      <link name="base"/><link name="one"/><link name="two"/><link name="three"/><link name="tcp"/>
      <joint name="a" type="revolute"><parent link="base"/><child link="one"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="b" type="revolute"><parent link="one"/><child link="two"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="c" type="revolute"><parent link="two"/><child link="three"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="tip" type="fixed"><parent link="three"/><child link="tcp"/><origin xyz="0.5 0 0"/></joint>
    </robot>"#;
    std::fs::write(&path, arm).unwrap();
    let options = RobotOptions {
        ee_links: vec!["tcp".into()],
        ..Default::default()
    };
    let robot = Robot::load(&path, &options).unwrap();
    let bound = robot.hand_reach(0).unwrap();
    let straight = robot.ee_pose(&[0.0; 3], 0);
    assert_eq!(straight.position, Vec3::X * 2.5);
    let tight = batchplan::IkOptions {
        position_tolerance: 1e-6,
        rotation_tolerance: 1e-6,
        ..Default::default()
    };
    assert!(!bound.excludes(straight, &tight));
    let translated = Pose::new(straight.position + Vec3::X * 0.003, straight.rotation);
    assert!(bound.excludes(translated, &tight));
    assert!(!bound.excludes(
        translated,
        &batchplan::IkOptions {
            position_tolerance: 0.004,
            ..tight
        }
    ));
    // IK accepts nearly unit quaternions and uses their matrix directly. Normalizing such a
    // target for the wrist-circle calculation may move it outside a tight angular tolerance.
    let loose_rotation = Pose::new(translated.position, straight.rotation * 0.9992);
    assert!(loose_rotation.is_unit());
    assert!(!bound.excludes(loose_rotation, &tight));
    let rotated = Pose::new(straight.position, Quat::from_rotation_y(0.1));
    assert!(bound.excludes(rotated, &tight));
    assert!(!bound.excludes(
        rotated,
        &batchplan::IkOptions {
            rotation_tolerance: 0.11,
            ..tight
        }
    ));
    let locked = RobotOptions {
        lock_joints: HashMap::from([("a".into(), 0.0)]),
        ..options.clone()
    };
    assert!(Robot::load(&path, &locked).unwrap().hand_reach(0).is_none());
    std::fs::write(
        &path,
        arm.replacen("axis xyz=\"0 0 1\"", "axis xyz=\"0 0 0\"", 1),
    )
    .unwrap();
    assert!(
        Robot::load(&path, &options)
            .unwrap()
            .hand_reach(0)
            .is_none()
    );
    std::fs::write(&path, arm.replacen("revolute", "prismatic", 1)).unwrap();
    assert!(
        Robot::load(&path, &options)
            .unwrap()
            .hand_reach(0)
            .is_none()
    );

    let skewed = r#"<mujoco><worldbody><body name="a" xyaxes="1 1 0 1 1.0000002 0">
      <joint name="aa" type="hinge" axis="0 0 1"/><body name="b">
      <joint name="bb" type="hinge" axis="0 0 1"/><body name="tcp" pos="1 0 0"/>
      </body></body></worldbody></mujoco>"#;
    let path = dir.join("skewed.xml");
    std::fs::write(&path, skewed).unwrap();
    assert!(
        Robot::load(&path, &options)
            .unwrap()
            .hand_reach(0)
            .is_none()
    );
}
