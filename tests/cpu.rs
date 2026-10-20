//! CPU device checks: kinematics against published parameters, obstacle distances, analytic gradients, IK.

#[path = "../examples/common/mod.rs"]
mod common;

use std::f64::consts::FRAC_PI_2;

use batchplan::rng::Rng;
use batchplan::*;
use glam::{DMat3, DVec3, Quat, Vec3};

fn panda() -> Robot {
    common::panda().unwrap()
}

fn random_q(robot: &Robot, rng: &mut Rng) -> Vec<f32> {
    (0..robot.dof()).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect()
}

/// Flange pose from Franka's published modified-DH table, independent of the URDF parser.
fn franka_dh_flange(q: &[f32]) -> (DMat3, DVec3) {
    let table = [
        (0.0, 0.333, 0.0),
        (0.0, 0.0, -FRAC_PI_2),
        (0.0, 0.316, FRAC_PI_2),
        (0.0825, 0.0, FRAC_PI_2),
        (-0.0825, 0.384, -FRAC_PI_2),
        (0.0, 0.0, FRAC_PI_2),
        (0.088, 0.0, FRAC_PI_2),
    ];
    let (mut r, mut p) = (DMat3::IDENTITY, DVec3::ZERO);
    for (i, &(a, d, alpha)) in table.iter().enumerate() {
        // T_i = RotX(alpha) * TransX(a) * RotZ(q) * TransZ(d)
        p += r * DVec3::new(a, 0.0, 0.0);
        r *= DMat3::from_rotation_x(alpha) * DMat3::from_rotation_z(q[i] as f64);
        p += r * DVec3::new(0.0, 0.0, d);
    }
    p += r * DVec3::new(0.0, 0.0, 0.107);
    (r, p)
}

#[test]
fn fk_matches_franka_dh_parameters() {
    let robot = panda();
    assert_eq!(robot.dof(), 7);
    let mut rng = Rng::new(7);
    for _ in 0..200 {
        let q = random_q(&robot, &mut rng);
        let pose = robot.link_pose(&q, "panda_link8").unwrap();
        let (r, p) = franka_dh_flange(&q);
        assert!((pose.position.as_dvec3() - p).length() < 1e-5, "position mismatch at {q:?}");
        let r_err =
            (DMat3::from_quat(pose.rotation.as_dquat()) - r).to_cols_array().iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(r_err < 1e-5, "rotation mismatch {r_err} at {q:?}");
    }
}

/// One obstacle of every kind, rotated off the world axes.
fn every_obstacle_kind() -> Vec<Obstacle> {
    let rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.4, -0.7, 1.1);
    let center = Vec3::new(0.1, -0.2, 0.3);
    vec![
        Obstacle::Cuboid { center, half_extents: Vec3::new(0.1, 0.2, 0.05), rotation },
        Obstacle::sphere(center, 0.15),
        Obstacle::Cylinder { center, rotation, radius: 0.12, half_height: 0.2 },
        Obstacle::Capsule { center, rotation, radius: 0.08, half_length: 0.15 },
    ]
}

#[test]
fn obstacle_distances_are_exact_and_differentiable() {
    let mut rng = Rng::new(17);
    for o in every_obstacle_kind() {
        let (mut outside, mut inside) = (0, 0);
        for _ in 0..2000 {
            let p = Vec3::new(rng.range(-0.3, 0.5), rng.range(-0.6, 0.2), rng.range(-0.1, 0.7));
            let (d, g) = o.distance(p);
            assert!((g.length() - 1.0).abs() < 1e-4, "{o:?}: gradient {g} at {p} is not a unit vector");
            // Stepping back along the gradient by the distance lands on the surface.
            let (on_surface, _) = o.distance(p - d * g);
            assert!(on_surface.abs() < 1e-4, "{o:?}: {p} - {d} * {g} is {on_surface} from the surface");
            // The distance is only piecewise smooth (edges, the medial axis inside), so accept
            // agreement with finite differences at either of two step sizes.
            let fd_error = |h: f32| {
                let fd = Vec3::new(
                    o.distance(p + Vec3::X * h).0 - o.distance(p - Vec3::X * h).0,
                    o.distance(p + Vec3::Y * h).0 - o.distance(p - Vec3::Y * h).0,
                    o.distance(p + Vec3::Z * h).0 - o.distance(p - Vec3::Z * h).0,
                ) / (2.0 * h);
                (fd - g).length()
            };
            let err = fd_error(1e-3).min(fd_error(1e-4));
            assert!(err < 2e-2 || d < 0.0, "{o:?}: gradient {g} off by {err} at {p}");
            if d > 0.0 { outside += 1 } else { inside += 1 }
        }
        assert!(outside > 100 && inside > 20, "{o:?}: {outside} samples outside, {inside} inside");
    }
}

#[test]
fn round_obstacle_distances_match_closed_forms() {
    let center = Vec3::new(1.0, 2.0, 3.0);
    let cylinder = Obstacle::cylinder(center, 0.5, 1.0);
    let capsule = Obstacle::capsule(center, 0.5, 1.0);
    let cases = [
        (&cylinder, Vec3::new(0.8, 0.0, 0.2), 0.3), // beside the side wall
        (&cylinder, Vec3::new(0.0, 0.0, 1.4), 0.4), // above the cap
        (&cylinder, Vec3::new(0.8, 0.0, 1.4), (0.09f32 + 0.16).sqrt()), // past the rim
        (&cylinder, Vec3::new(0.3, 0.0, 0.0), -0.2), // inside, nearest the wall
        (&capsule, Vec3::new(0.0, 0.0, 1.9), 0.4),  // above the end cap
        (&capsule, Vec3::new(0.6, 0.0, 1.8), 0.5),  // diagonal from the end
        (&capsule, Vec3::new(0.2, 0.0, -0.3), -0.3), // inside the shaft
    ];
    for (o, offset, expected) in cases {
        let (d, _) = o.distance(center + offset);
        assert!((d - expected).abs() < 1e-5, "{o:?} at offset {offset}: {d}, expected {expected}");
    }
    // Rotating the obstacle and the query point together leaves the distance unchanged.
    let r = Quat::from_rotation_y(0.9);
    let rotated = Obstacle::Cylinder { center, rotation: r, radius: 0.5, half_height: 1.0 };
    assert!((rotated.distance(center + r * Vec3::new(0.8, 0.0, 1.4)).0 - 0.5).abs() < 1e-5);
}

#[test]
fn default_pose_is_collision_free_on_table() {
    let robot = panda();
    let cpu = Device::cpu(&robot).unwrap();
    let world = common::tabletop(&mut Rng::new(1));
    let table_only = World { obstacles: world.obstacles[..1].to_vec() };
    // Clearances are exact up to the margins.
    let w = CollisionWeights { margin: 0.1, self_margin: 0.1, ..CollisionWeights::NONE };
    let e = cpu.evaluate(&cpu.upload(&[table_only]).unwrap(), &[0], robot.default_q(), &w).unwrap();
    assert!(e.world_clearance[0] > 0.05, "world clearance {}", e.world_clearance[0]);
    assert!(e.self_clearance[0] > 0.0, "self clearance {}", e.self_clearance[0]);
}

#[test]
fn collision_gradient_matches_finite_differences() {
    let robot = panda();
    let cpu = Device::cpu(&robot).unwrap();
    let n = robot.dof();
    let w = CollisionWeights { world: 1000.0, self_collision: 1000.0, margin: 0.05, self_margin: 0.02 };
    let mut rng = Rng::new(11);
    let worlds: Vec<World> = (0..8).map(|_| common::tabletop(&mut rng)).collect();
    let worlds = cpu.upload(&worlds).unwrap();
    let mut checked = 0;
    for trial in 0..400 {
        let q = random_q(&robot, &mut rng);
        let world = trial as u32 % 8;
        let e = cpu.evaluate(&worlds, &[world], &q, &w).unwrap();
        if e.cost[0] < 1e-3 {
            continue;
        }
        let g = &e.grad;
        let norm = g.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
        // The cost is only piecewise smooth (box faces, hinge activation), so a finite difference
        // straddling a kink is wrong at that step size; accept agreement at either step size.
        let rel_err = |h: f32| {
            let fd: Vec<f32> = (0..n)
                .map(|j| {
                    let (mut qp, mut qm) = (q.clone(), q.clone());
                    qp[j] += h;
                    qm[j] -= h;
                    let cp = cpu.evaluate(&worlds, &[world], &qp, &w).unwrap().cost[0];
                    let cm = cpu.evaluate(&worlds, &[world], &qm, &w).unwrap().cost[0];
                    (cp - cm) / (2.0 * h)
                })
                .collect();
            (0..n).map(|j| (fd[j] - g[j]).powi(2)).sum::<f32>().sqrt() / norm
        };
        let err = rel_err(2e-4).min(rel_err(5e-5));
        assert!(err < 2e-2, "gradient mismatch {err} (|g| = {norm}) at {q:?}, grad {g:?}");
        checked += 1;
    }
    assert!(checked > 50, "only {checked} colliding samples");
}

#[test]
fn ik_reaches_targets_from_collision_free_configurations() {
    let robot = panda();
    let cpu = Device::cpu(&robot).unwrap();
    let mut rng = Rng::new(5);
    let empty = cpu.upload(&[World::default()]).unwrap();
    let mut problems = vec![];
    while problems.len() < 64 {
        let q = random_q(&robot, &mut rng);
        if cpu.evaluate(&empty, &[0], &q, &CollisionWeights::NONE).unwrap().collision_free(0) {
            problems.push(IkProblem::new(0, robot.ee_pose(&q, 0)));
        }
    }
    let o = IkOptions { seeds: 16, ..Default::default() };
    let result = solve_ik(&cpu, &empty, &problems, &o).unwrap();
    let solved = (0..problems.len()).filter(|&p| result.best(p).is_some()).count();
    assert!(solved >= 60, "solved {solved}/64");
    for (p, problem) in problems.iter().enumerate() {
        if let Some(q) = result.best(p) {
            assert!(
                (robot.ee_pose(q, 0).position - problem.targets[0].unwrap().position).length() < o.position_tolerance
            );
        }
    }
}
