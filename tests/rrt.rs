//! RRT-Connect, shortcutting and smoothing, and the planner searching with them. GPU parts are
//! skipped when no adapter is available unless `BATCHPLAN_REQUIRE_GPU=1`.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::rrt::connect;
use batchplan::shortcut::shortcut;
use batchplan::{
    CollisionWeights, Device, Obstacle, PlanOptions, PlanProblem, Robot, RobotOptions, RrtOptions,
    RrtProblem, ShortcutOptions, Trajectory, World, Worlds,
};
use glam::Vec3;

const UR5E: &str = "ur5e/ur_description/urdf/ur5e.urdf";

fn devices(robot: &Robot) -> Vec<Device> {
    let mut devices = vec![Device::cpu(robot).unwrap()];
    match Device::gpu(robot) {
        Ok(gpu) => devices.push(gpu),
        Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => eprintln!("skipping GPU: {e}"),
        Err(e) => panic!("no GPU: {e}"),
    }
    devices
}

fn searches(problems: &[PlanProblem]) -> Vec<RrtProblem> {
    problems
        .iter()
        .map(|p| RrtProblem {
            world: p.world,
            start: p.start.clone(),
            goals: vec![p.goal.clone()],
            hold: vec![],
        })
        .collect()
}

fn length(path: &[f32], n: usize) -> f32 {
    path.chunks(n)
        .zip(path.chunks(n).skip(1))
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .sum()
}

/// Every edge of every path sampled at `resolution` on the CPU: the worst clearance.
fn worst_clearance(cpu: &Device, worlds: &Worlds, paths: &[(u32, &[f32])], resolution: f32) -> f32 {
    let n = cpu.robot().dof();
    let (mut q, mut item_world) = (vec![], vec![]);
    for &(world, path) in paths {
        for (a, b) in path.chunks(n).zip(path.chunks(n).skip(1)) {
            let d = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt();
            let count = (d / resolution).ceil().max(1.0) as usize;
            for k in 0..=count {
                let t = k as f32 / count as f32;
                q.extend(a.iter().zip(b).map(|(x, y)| x + (y - x) * t));
                item_world.push(world);
            }
        }
    }
    let e = cpu
        .evaluate(worlds, &item_world, &q, &CollisionWeights::NONE)
        .unwrap();
    e.world_clearance
        .iter()
        .chain(&e.self_clearance)
        .fold(f32::INFINITY, |m, &v| m.min(v))
}

#[test]
fn rrt_paths_are_collision_free_under_a_denser_check() {
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let (scene, problems) = common::far_reaches(&robot, &cpu, 12, true);
    let on_cpu = cpu.upload(&scene).unwrap();
    let o = RrtOptions::default();
    for d in devices(&robot) {
        let found = connect(&d, &d.upload(&scene).unwrap(), &searches(&problems), &o).unwrap();
        let mut paths = vec![];
        for (p, path) in problems.iter().zip(&found.paths) {
            let Some(path) = path else { continue };
            assert!(
                path[..n] == p.start[..] && path[path.len() - n..] == p.goal[..],
                "{}: wrong ends",
                d.name()
            );
            let longest = path
                .chunks(n)
                .zip(path.chunks(n).skip(1))
                .map(|(a, b)| {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).powi(2))
                        .sum::<f32>()
                        .sqrt()
                })
                .fold(0.0f32, f32::max);
            // Only a straight line from start to goal, or one bent through a via point or two, may
            // have edges longer than the range.
            assert!(
                path.len() <= 4 * n || longest <= o.range * 1.0001,
                "{}: an edge of {longest}",
                d.name()
            );
            paths.push((p.world, &path[..]));
        }
        eprintln!(
            "{}: RRT-Connect found {} of {} paths",
            d.name(),
            paths.len(),
            problems.len()
        );
        assert!(
            paths.len() >= 10,
            "{}: found only {} paths",
            d.name(),
            paths.len()
        );
        let worst = worst_clearance(&cpu, &on_cpu, &paths, o.resolution / 4.0);
        assert!(worst > -1e-3, "{}: an edge penetrates by {worst}", d.name());
    }
}

#[test]
fn shortcutting_never_lengthens_a_path() {
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let (scene, problems) = common::far_reaches(&robot, &cpu, 12, true);
    let worlds = cpu.upload(&scene).unwrap();
    let found = connect(&cpu, &worlds, &searches(&problems), &RrtOptions::default()).unwrap();
    let (world, raw): (Vec<u32>, Vec<Vec<f32>>) = problems
        .iter()
        .zip(found.paths)
        .filter_map(|(p, path)| Some((p.world, path?)))
        .unzip();
    let mut short = raw.clone();
    let o = ShortcutOptions::default();
    shortcut(&cpu, &worlds, &world, &mut short, &o).unwrap();
    let (mut before, mut after) = (0.0, 0.0);
    for (a, b) in raw.iter().zip(&short) {
        assert!(
            length(b, n) <= length(a, n),
            "shortcutting lengthened a path: {} > {}",
            length(b, n),
            length(a, n)
        );
        assert!(
            a[..n] == b[..n] && a[a.len() - n..] == b[b.len() - n..],
            "shortcutting moved an end"
        );
        (before, after) = (before + length(a, n), after + length(b, n));
    }
    let straight: f32 = raw
        .iter()
        .map(|p| length(&[&p[..n], &p[p.len() - n..]].concat(), n))
        .sum();
    eprintln!(
        "total length {before:.1} -> {after:.1} over {} paths; straight lines {straight:.1}",
        raw.len()
    );
    // Straight lines from start to goal bound the length from below.
    assert!(
        after < before && after < 1.1 * straight,
        "shortcutting barely helped: {before} -> {after}"
    );
    let paths: Vec<(u32, &[f32])> = world
        .iter()
        .copied()
        .zip(short.iter().map(|p| &p[..]))
        .collect();
    let worst = worst_clearance(&cpu, &worlds, &paths, o.resolution / 4.0);
    assert!(worst > -1e-3, "a shortcut penetrates by {worst}");
}

#[test]
fn fixed_seeds_give_fixed_results() {
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let (scene, problems) = common::far_reaches(&robot, &cpu, 6, true);
    let worlds = cpu.upload(&scene).unwrap();
    let run = |rrt: RrtOptions| {
        let found = connect(&cpu, &worlds, &searches(&problems), &rrt).unwrap();
        let (world, mut paths): (Vec<u32>, Vec<Vec<f32>>) = problems
            .iter()
            .zip(found.paths)
            .filter_map(|(p, path)| Some((p.world, path?)))
            .unzip();
        shortcut(
            &cpu,
            &worlds,
            &world,
            &mut paths,
            &ShortcutOptions::default(),
        )
        .unwrap();
        paths
    };
    let rrt = RrtOptions::default();
    let first = run(rrt);
    assert!(!first.is_empty());
    assert_eq!(first, run(rrt));
    assert_ne!(
        first,
        run(RrtOptions {
            rng_seed: rrt.rng_seed + 1,
            ..rrt
        }),
        "the seed changes nothing"
    );
}

#[test]
fn plan_solves_every_problem_rrt_connect_reaches() {
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let (scene, problems) = common::far_reaches(&robot, &cpu, 24, false);
    let on_cpu = cpu.upload(&scene).unwrap();
    let mut solved = vec![];
    for d in devices(&robot) {
        let worlds = d.upload(&scene).unwrap();
        let result = batchplan::plan(&d, &worlds, &problems, &PlanOptions::default()).unwrap();
        let reachable = connect(&d, &worlds, &searches(&problems), &RrtOptions::default()).unwrap();
        let rrt = reachable.paths.iter().flatten().count();
        eprintln!(
            "{}: planned {}/24, RRT-Connect {rrt}/24",
            d.name(),
            result.solved().count()
        );
        // Many goals lie a turn away past the table, where only their equivalents can be reached.
        assert!(
            result.solved().count() >= 20,
            "{}: planned only {} of 24",
            d.name(),
            result.solved().count()
        );
        for (i, path) in reachable.paths.iter().enumerate() {
            assert!(
                path.is_none() || result.best(i).is_some(),
                "{}: RRT-Connect reaches problem {i}; the plan does not",
                d.name()
            );
        }
        let stops: usize = result
            .solved()
            .map(|s| {
                let cp: Vec<&[f32]> = s.solution.chunks(n).collect();
                (1..cp.len() - 1)
                    .filter(|&t| cp[t - 1] == cp[t] && cp[t] == cp[t + 1])
                    .count()
                    - 2
            })
            .sum();
        eprintln!("MEASURE {}: stops {stops}", d.name());
        // Every plan, timed and sampled 4x denser than validation, under the CPU model.
        let (mut q, mut item_world) = (vec![], vec![]);
        for s in result.solved() {
            let trajectory = Trajectory::new(&robot, s.solution, 1.0).unwrap();
            trajectory.check(&robot).unwrap();
            let samples = trajectory
                .sample(32.0 / trajectory.knot_interval)
                .unwrap()
                .positions;
            item_world.extend(std::iter::repeat_n(s.problem.world, samples.len() / n));
            q.extend(samples);
        }
        let e = cpu
            .evaluate(&on_cpu, &item_world, &q, &CollisionWeights::NONE)
            .unwrap();
        let worst = e
            .world_clearance
            .iter()
            .chain(&e.self_clearance)
            .fold(f32::INFINITY, |m, &v| m.min(v));
        assert!(worst > -2e-3, "{}: a plan penetrates by {worst}", d.name());
        solved.push(result.paths);
    }
    // Searches run on the CPU's checks on either device, so the plans are the same.
    if let [cpu, gpu] = &solved[..] {
        assert_eq!(cpu, gpu, "the devices plan differently");
    }
}

/// A turntable arm: a continuous joint about z carrying a 40 cm arm along x, and a shoulder.
const TURNTABLE: &str = r#"<robot name="turntable">
  <link name="base"/>
  <link name="table"><collision><origin xyz="0.2 0 0.05"/><geometry><box size="0.4 0.06 0.06"/></geometry></collision></link>
  <link name="arm"><collision><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.05 0.05"/></geometry></collision></link>
  <joint name="turn" type="continuous"><parent link="base"/><child link="table"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/></joint>
  <joint name="shoulder" type="revolute"><parent link="table"/><child link="arm"/><origin xyz="0.4 0 0.05"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" velocity="2" effort="1"/></joint>
</robot>"#;

/// How far each joint moves along `path` (`[k, dof]` waypoints or control points).
fn travel(path: &[f32], n: usize) -> Vec<f32> {
    (0..n)
        .map(|j| {
            path.chunks(n)
                .zip(path.chunks(n).skip(1))
                .map(|(a, b)| (b[j] - a[j]).abs())
                .sum()
        })
        .collect()
}

#[test]
fn continuous_joints_turn_the_short_way_unless_it_is_blocked() {
    let dir = std::env::temp_dir().join(format!("batchplan-turntable-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("turntable.urdf"), TURNTABLE).unwrap();
    let robot = Robot::load(dir.join("turntable.urdf"), &RobotOptions::default()).unwrap();
    assert_eq!(robot.continuous(), [true, false]);
    // From 2.6 rad to -2.6 rad is 1.08 rad through pi, or 5.2 rad the long way.
    let problem = PlanProblem::new(0, vec![2.6, 0.0], vec![-2.6, 0.0]);
    let short = std::f32::consts::TAU - 5.2;
    // A fin along -x, beyond the turntable but within the arm's reach: across the short way only
    // (the arm points along -x at pi).
    let wall = World {
        obstacles: vec![Obstacle::Cuboid {
            center: Vec3::new(-0.55, 0.0, 0.15),
            half_extents: Vec3::new(0.1, 0.01, 0.25),
            rotation: glam::Quat::IDENTITY,
        }],
    };
    for d in devices(&robot) {
        let worlds = d.upload(&[World::default(), wall.clone()]).unwrap();
        let open = batchplan::plan(
            &d,
            &worlds,
            std::slice::from_ref(&problem),
            &PlanOptions::default(),
        )
        .unwrap();
        let path = open
            .best(0)
            .unwrap_or_else(|| panic!("{}: no plan", d.name()));
        let turned = travel(path, 2)[0];
        assert!(
            (turned - short).abs() < 0.05,
            "{}: turned {turned} rad, not the short way {short}",
            d.name()
        );
        // The trajectory runs past pi, as a continuous joint may.
        Trajectory::new(&robot, path, 1.0)
            .unwrap()
            .check(&robot)
            .unwrap();
        // The final pose is the goal's.
        let (end, goal) = (
            robot.ee_pose(&path[path.len() - 2..], 0),
            robot.ee_pose(&problem.goal, 0),
        );
        assert!(
            (end.position - goal.position).length() < 1e-4,
            "{}: ends at {end:?}",
            d.name()
        );

        let blocked = PlanProblem {
            world: 1,
            ..problem.clone()
        };
        let around = batchplan::plan(&d, &worlds, &[blocked], &PlanOptions::default()).unwrap();
        let path = around
            .best(0)
            .unwrap_or_else(|| panic!("{}: no plan around the wall", d.name()));
        let turned = travel(path, 2)[0];
        assert!(
            turned > 4.0,
            "{}: turned {turned} rad; the short way is blocked",
            d.name()
        );

        // With the fin along +x instead, only the way through pi is open, and RRT-Connect finds
        // it: its distances and steps wrap.
        let fin = |x: f32| World {
            obstacles: vec![Obstacle::Cuboid {
                center: Vec3::new(x, 0.0, 0.15),
                half_extents: Vec3::new(0.1, 0.01, 0.25),
                rotation: glam::Quat::IDENTITY,
            }],
        };
        let worlds = d.upload(&[fin(0.55)]).unwrap();
        let search = RrtProblem {
            world: 0,
            start: problem.start.clone(),
            goals: vec![problem.goal.clone()],
            hold: vec![],
        };
        let found = connect(&d, &worlds, &[search], &RrtOptions::default()).unwrap();
        let path = found.paths[0]
            .as_ref()
            .unwrap_or_else(|| panic!("{}: RRT-Connect found no way through pi", d.name()));
        let turned = travel(path, 2)[0];
        assert!(
            turned < 5.0,
            "{}: RRT-Connect turned {turned} rad, the blocked long way",
            d.name()
        );
    }
}

#[test]
fn ur5e_reaches_a_goal_a_turn_away_the_other_way_round() {
    // The goal's shoulder pan is 4 rad, past a wall at +90 degrees; the same pose at 4 - 2 pi lies
    // the other way round. Within the joint's interval the goal cannot be reached.
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let mut start = robot.default_q().to_vec();
    start[0] = 0.0;
    let mut goal = start.clone();
    goal[0] = 4.0;
    let wall = World {
        obstacles: vec![Obstacle::Cuboid {
            center: Vec3::new(0.0, 0.6, 0.3),
            half_extents: Vec3::new(0.05, 0.3, 0.6),
            rotation: glam::Quat::IDENTITY,
        }],
    };
    for d in devices(&robot) {
        let worlds = d.upload(std::slice::from_ref(&wall)).unwrap();
        let problem = PlanProblem::new(0, start.clone(), goal.clone());
        let result = batchplan::plan(&d, &worlds, &[problem], &PlanOptions::default()).unwrap();
        let path = result
            .best(0)
            .unwrap_or_else(|| panic!("{}: no plan", d.name()));
        let n = robot.dof();
        let end = &path[path.len() - n..];
        assert!(
            (end[0] - (4.0 - std::f32::consts::TAU)).abs() < 1e-5,
            "{}: ends with pan {}",
            d.name(),
            end[0]
        );
        let (reached, wanted) = (robot.ee_pose(end, 0), robot.ee_pose(&goal, 0));
        assert!(
            (reached.position - wanted.position).length() < 1e-4,
            "{}: ends at {reached:?}",
            d.name()
        );
        assert!(
            path.chunks(n).all(|q| q[0] <= 0.0),
            "{}: the pan turns toward the wall",
            d.name()
        );
    }
}

/// A straight line an obstacle only grazes is bent around it rather than given up for a search's
/// detour: the path keeps to the line's route, through a via point or two beside it, and is barely
/// longer.
/// Shortcuts between whole configurations keep a joint that swings out and back on a detour;
/// straightening each joint in turn removes it.
#[test]
fn shortcutting_straightens_a_joint_that_swings_back() {
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let start = robot.default_q().to_vec();
    let mut goal = start.clone();
    goal[0] += 2.0;
    goal[2] -= 0.8;
    // A small ball where the hand passes halfway along the straight line, and a detour over it on
    // which the wrist turns 1.5 rad out and back.
    let middle: Vec<f32> = start
        .iter()
        .zip(&goal)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let ball = Obstacle::sphere(robot.ee_pose(&middle, 0).position, 0.02);
    let worlds = cpu
        .upload(&[World {
            obstacles: vec![ball],
        }])
        .unwrap();
    let mut over = middle.clone();
    over[1] -= 0.4;
    over[n - 1] += 1.5;
    let detour = [start.clone(), over, goal.clone()].concat();
    let o = ShortcutOptions {
        straighten: true,
        ..Default::default()
    };
    let blocked = cpu
        .evaluate(&worlds, &[0], &middle, &CollisionWeights::NONE)
        .unwrap();
    assert!(!blocked.collision_free(0), "the straight line must collide");
    assert!(
        worst_clearance(&cpu, &worlds, &[(0, &detour)], o.resolution / 4.0) > o.clearance,
        "the detour must be clear"
    );
    let mut paths = vec![detour];
    shortcut(&cpu, &worlds, &[0], &mut paths, &o).unwrap();
    let wrist = travel(&paths[0], n)[n - 1];
    // None; 0.84 rad without straightening, smoothing alone.
    assert!(
        wrist < 0.1,
        "the wrist turns {wrist} rad, for none from start to goal"
    );
    assert!(
        worst_clearance(&cpu, &worlds, &[(0, &paths[0])], o.resolution / 4.0) > -1e-3,
        "the shortcut path collides"
    );
}

#[test]
fn a_grazed_straight_line_is_bent_around_the_obstacle() {
    let robot = Robot::load(common::asset(UR5E), &RobotOptions::default()).unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let start = robot.default_q().to_vec();
    let mut goal = start.clone();
    goal[0] += 2.0;
    goal[2] -= 0.8;
    // A small ball where the hand passes halfway along the line.
    let middle: Vec<f32> = start
        .iter()
        .zip(&goal)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let ball = Obstacle::sphere(robot.ee_pose(&middle, 0).position, 0.02);
    let worlds = cpu
        .upload(&[World {
            obstacles: vec![ball],
        }])
        .unwrap();
    let blocked = cpu
        .evaluate(&worlds, &[0], &middle, &CollisionWeights::NONE)
        .unwrap();
    assert!(!blocked.collision_free(0), "the straight line must collide");
    let problem = RrtProblem {
        world: 0,
        start: start.clone(),
        goals: vec![goal.clone()],
        hold: vec![],
    };
    let result = connect(&cpu, &worlds, &[problem], &RrtOptions::default()).unwrap();
    let path = result.paths[0].as_ref().expect("a path");
    let straight = length(&[start, goal].concat(), n);
    assert!(path.len() / n <= 4, "{} waypoints", path.len() / n);
    assert!(
        length(path, n) < 1.1 * straight,
        "{} rad against {straight} straight",
        length(path, n)
    );
}

/// Random Panda reaches with a ball where the hand passes halfway along the straight line: samples
/// drawn near the line first keep the planned paths near it (3.8% longer on average, two more than
/// 10% longer, against 7.3% and nine when samples come from the joint ranges at once).
#[test]
fn plans_around_a_ball_on_the_straight_line_stay_near_it() {
    let robot = common::panda().unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let mut rng = Rng::new(3);
    let (mut scene, mut problems) = (vec![], vec![]);
    while problems.len() < 48 {
        let start: Vec<f32> = (0..n)
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect();
        let goal: Vec<f32> = (0..n)
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect();
        let middle: Vec<f32> = start
            .iter()
            .zip(&goal)
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let ball = Obstacle::sphere(robot.ee_pose(&middle, 0).position, 0.2);
        let world = World {
            obstacles: vec![ball],
        };
        let one = cpu.upload(std::slice::from_ref(&world)).unwrap();
        let ends = cpu
            .evaluate(
                &one,
                &[0, 0],
                &[&start[..], &goal[..]].concat(),
                &CollisionWeights::NONE,
            )
            .unwrap();
        if ends.collision_free(0) && ends.collision_free(1) {
            problems.push(PlanProblem::new(scene.len() as u32, start, goal));
            scene.push(world);
        }
    }
    let result = batchplan::plan(
        &cpu,
        &cpu.upload(&scene).unwrap(),
        &problems,
        &PlanOptions::default(),
    )
    .unwrap();
    let excess: Vec<f32> = result
        .solved()
        .map(|s| {
            let straight = length(&[&s.problem.start[..], &s.problem.goal[..]].concat(), n);
            let dense = Trajectory::new(&robot, s.solution, 1.0)
                .unwrap()
                .sample(100.0)
                .unwrap()
                .positions;
            length(&dense, n) / straight - 1.0
        })
        .collect();
    assert_eq!(excess.len(), problems.len(), "every reach has a path");
    let mean = excess.iter().sum::<f32>() / excess.len() as f32;
    let far = excess.iter().filter(|&&e| e > 0.1).count();
    assert!(
        mean < 0.055,
        "paths are {:.1}% longer than straight on average",
        100.0 * mean
    );
    assert!(
        far <= 5,
        "{far} paths are more than 10% longer than straight"
    );
}

/// MotionBenchMaker's `bookshelf_tall_panda` problem 1, as robometrics packages it: the search near
/// the straight line finds a path that no spline follows, and the problem is searched again over
/// the joint ranges, which finds one that does.
#[test]
fn a_path_no_spline_follows_is_searched_again() {
    let robot = common::panda().unwrap();
    let cpu = Device::cpu(&robot).unwrap();
    let pose = |p: [f32; 7]| {
        (
            Vec3::new(p[0], p[1], p[2]),
            glam::Quat::from_xyzw(p[4], p[5], p[6], p[3]).normalize(),
        )
    };
    let shelf = [0.4287125, 1.1404362, 0.0, 0.83603096, 0.0, 0.0, 0.5486823];
    let mut obstacles = vec![];
    for (dims, at) in [
        ([1.2, 1.0, 0.04], [0.0, 0.0, 0.0]),
        ([1.2, 1.0, 0.04], [0.0, 0.0, 0.3]),
        ([1.2, 1.0, 0.04], [0.0, 0.0, 0.6]),
        ([1.2, 1.0, 0.04], [0.0, 0.0, 0.9]),
        ([1.3, 0.04, 1.64], [0.4587154, -0.1989478, 0.12]),
        ([1.3, 0.04, 1.64], [-0.4587153, 0.1989478, 0.12]),
    ] {
        let (center, rotation) = pose([
            shelf[0] + at[0],
            shelf[1] + at[1],
            at[2],
            shelf[3],
            0.0,
            0.0,
            shelf[6],
        ]);
        obstacles.push(Obstacle::Cuboid {
            center,
            half_extents: Vec3::from(dims) * 0.5,
            rotation,
        });
    }
    let (center, rotation) = pose([-0.05, 0.0, -0.4, 1.0, 0.0, 0.0, 0.0]);
    obstacles.push(Obstacle::Cuboid {
        center,
        half_extents: Vec3::new(0.15, 0.125, 0.4),
        rotation,
    });
    for at in [
        [0.71416974, 0.9076316, 0.68],
        [0.4019689, 0.8250349, 0.68],
        [0.63633084, 0.5053906, 0.68],
        [0.5747014, 0.96811984, 0.08],
        [0.61309785, 0.733467, 0.08],
        [0.0788708, 0.7471645, 0.08],
        [0.12357195, 1.1637776, 0.38],
        [0.31628948, 0.86219464, 0.38],
        [0.08531671, 0.7443689, 0.38],
    ] {
        let (center, rotation) = pose([at[0], at[1], at[2], shelf[3], 0.0, 0.0, shelf[6]]);
        obstacles.push(Obstacle::Cylinder {
            center,
            rotation,
            radius: 0.01,
            half_height: 0.07,
        });
    }
    let problem = PlanProblem::new(
        0,
        vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        vec![
            -1.9269282,
            -1.358336,
            1.0875098,
            -0.10738423,
            -0.6509739,
            3.3858628,
            0.35020632,
        ],
    );
    let worlds = cpu.upload(&[World { obstacles }]).unwrap();
    let result = batchplan::plan(&cpu, &worlds, &[problem], &PlanOptions::default()).unwrap();
    assert!(result.best(0).is_some(), "no path");
}
