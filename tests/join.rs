//! Joining motions without stopping (`join`), on every device: the joined motion turns within the
//! blend radius of each join and is the motions themselves elsewhere, finishes sooner than the
//! motions one after another and never later, keeps the limits and the sequence's constraints, and
//! refuses sequences whose motions do not meet. Falling back from blends that collide is tested in `src/join.rs`.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::{
    Device, Error, HandConstraint, JoinOptions, JoinResult, PlanOptions, PlanProblem, Pose, Robot,
    Sequence, ToolPath, ToolPathOptions, Trajectory, World, Worlds,
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

/// The radius the tests join at: 5 cm.
const BLEND: f32 = 0.05;

/// `paths` joined within `blend` at every join.
fn join(device: &Device, worlds: &Worlds, paths: Vec<Vec<f32>>, blend: f32) -> JoinResult {
    let sequence = Sequence {
        world: 0,
        blends: vec![blend; paths.len() - 1],
        paths,
        constraints: vec![],
    };
    batchplan::join(device, worlds, &[sequence], &JoinOptions::default()).unwrap()
}

/// The hand moved `down` meters down from the ready pose, then `side` sideways: two straight
/// motions with a right-angled corner between them. Returns them and the hand's start, corner and
/// end.
fn corner(device: &Device, worlds: &Worlds, down: f32, side: f32) -> (Vec<Vec<f32>>, [Vec3; 3]) {
    let robot = device.robot();
    let start = robot.ee_pose(robot.default_q(), 0);
    let corner = start.position - Vec3::new(0.0, 0.0, down);
    let end = corner + Vec3::new(0.0, side, 0.0);
    let o = ToolPathOptions::default();
    let mut from = robot.default_q().to_vec();
    let mut paths = vec![];
    for to in [corner, end] {
        let path = ToolPath {
            world: 0,
            start: from.clone(),
            hand: 0,
            waypoints: vec![Pose::new(to, start.rotation)],
            holding: None,
        };
        let followed = batchplan::follow(device, worlds, &[path], &o).unwrap();
        assert_eq!(followed.fraction[0], 1.0, "{:?}", followed.stopped[0]);
        let p = followed.paths[0].clone();
        from = p[p.len() - robot.dof()..].to_vec();
        paths.push(p);
    }
    (paths, [start.position, corner, end])
}

/// The configurations a path passes through, timed and sampled at 500 Hz.
fn samples(robot: &Robot, path: &[f32]) -> Vec<f32> {
    let trajectory = Trajectory::try_new(robot, path, 1.0).unwrap();
    trajectory.sample(500.0).unwrap().positions
}

/// The distance from `p` to the segment from `a` to `b`.
fn off_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
    p.distance(a + (b - a) * t)
}

/// Seconds the paths take, each timed alone.
fn duration(robot: &Robot, paths: &[Vec<f32>]) -> f32 {
    paths
        .iter()
        .map(|p| Trajectory::try_new(robot, p, 1.0).unwrap().duration())
        .sum()
}

#[test]
fn joined_motions_turn_within_the_blend_and_finish_sooner() {
    let robot = panda();
    let precision = ToolPathOptions::default().precision;
    for device in devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[World::default()]).unwrap();
        let (paths, [start, corner, end]) = self::corner(&device, &worlds, 0.1, 0.1);
        let joined = join(&device, &worlds, paths.clone(), BLEND);
        let radius = joined.blends[0][0].expect("a free corner blends");
        assert!(radius > 0.0 && radius <= BLEND, "{name}: blend {radius} m");
        assert_eq!(
            joined.paths[0].len(),
            1,
            "{name}: one motion, without stopping"
        );
        let path = &joined.paths[0][0];

        let trajectory = Trajectory::try_new(&robot, path, 1.0).unwrap();
        trajectory.check(&robot).unwrap();
        let apart = duration(&robot, &paths);
        assert!(
            trajectory.duration() < apart,
            "{name}: {} s joined, {apart} s apart",
            trajectory.duration()
        );

        // Beyond the blend radius of the corner, the hand is on the two lines; within it, it turns
        // without passing through the corner itself.
        let mut nearest = f32::INFINITY;
        for q in samples(&robot, path).chunks(robot.dof()) {
            let p = robot.ee_pose(q, 0).position;
            nearest = nearest.min(p.distance(corner));
            if p.distance(corner) > radius + 1e-4 {
                let off = off_segment(p, start, corner).min(off_segment(p, corner, end));
                assert!(off <= precision + 1e-4, "{name}: {off} m off the lines");
            }
        }
        assert!(
            nearest > 1e-3,
            "{name}: the hand passes {nearest} m from the corner"
        );
    }
}

#[test]
fn a_zero_blend_stops_at_every_join() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let (paths, _) = corner(&device, &worlds, 0.1, 0.1);
    let joined = join(&device, &worlds, paths.clone(), 0.0);
    assert_eq!(joined.blends[0], vec![None]);
    assert_eq!(joined.paths[0], paths, "the motions, one after another");
}

#[test]
fn turning_back_never_finishes_later_than_stopping() {
    // Out and straight back, and motions that turn by 116 and 153 degrees in joint space: blends
    // through such turns can be slower than stopping, and are kept only where they are not.
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let out = vec![0.6, -0.8, 0.2, -2.2, 0.1, 1.7, 0.5];
    let sequences = [
        vec![
            robot.default_q().to_vec(),
            out.clone(),
            robot.default_q().to_vec(),
        ],
        vec![
            robot.default_q().to_vec(),
            out,
            vec![-0.5, -0.5, -0.3, -1.8, 0.4, 1.9, 1.2],
            vec![0.2, -1.0, 0.5, -2.6, -0.3, 1.4, 0.3],
        ],
    ];
    for stops_at in sequences {
        let problems: Vec<PlanProblem> = stops_at
            .windows(2)
            .map(|w| PlanProblem::new(0, w[0].clone(), w[1].clone()))
            .collect();
        let plans = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap();
        let paths: Vec<Vec<f32>> = plans.solved().map(|s| s.solution.to_vec()).collect();
        assert_eq!(paths.len(), problems.len());
        for blend in [0.02, 0.05, 0.1, 0.2, 0.4] {
            let joined = join(&device, &worlds, paths.clone(), blend);
            let (joined, apart) = (duration(&robot, &joined.paths[0]), duration(&robot, &paths));
            assert!(
                joined <= apart,
                "blend {blend} m: {joined} s joined, {apart} s apart"
            );
        }
    }
}

#[test]
fn planned_motions_join_within_the_limits_alike_on_every_device() {
    let robot = panda();
    let n = robot.dof();
    // A sweep across the front, each motion carrying on roughly where the last was going.
    let stops_at = [
        vec![-1.2, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8],
        vec![-0.4, -0.3, 0.1, -1.9, 0.1, 1.7, 0.7],
        vec![0.4, -0.2, 0.2, -1.8, 0.2, 1.8, 0.6],
        vec![1.2, -0.4, 0.3, -2.0, 0.3, 1.9, 0.5],
    ];
    let problems: Vec<PlanProblem> = stops_at
        .windows(2)
        .map(|w| PlanProblem::new(0, w[0].clone(), w[1].clone()))
        .collect();
    let mut first: Option<JoinResult> = None;
    for device in devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[World::default()]).unwrap();
        let plans = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap();
        let paths: Vec<Vec<f32>> = plans.solved().map(|s| s.solution.to_vec()).collect();
        assert_eq!(paths.len(), 3, "{name}: every motion plans");
        let joined = join(&device, &worlds, paths.clone(), BLEND);
        assert_eq!(joined.blends[0].len(), 2);
        assert!(
            joined.blends[0].iter().all(Option::is_some),
            "{name}: {:?}",
            joined.blends[0]
        );
        assert_eq!(
            joined.paths[0].len(),
            1,
            "{name}: one motion, without stopping"
        );
        let path = &joined.paths[0][0];
        assert_eq!(
            &path[..n],
            &stops_at[0][..],
            "{name}: starts where the first starts"
        );
        assert_eq!(
            &path[path.len() - n..],
            &paths[2][paths[2].len() - n..],
            "{name}"
        );
        let trajectory = Trajectory::try_new(&robot, path, 1.0).unwrap();
        trajectory.check(&robot).unwrap();
        let apart = duration(&robot, &paths);
        assert!(
            trajectory.duration() < 0.9 * apart,
            "{name}: {} s, {apart} s apart",
            trajectory.duration()
        );
        match &first {
            None => first = Some(joined),
            Some(cpu) => {
                assert_eq!(joined.paths, cpu.paths, "{name}: joins as the CPU does");
                assert_eq!(joined.blends, cpu.blends, "{name}");
            }
        }
    }
}

#[test]
fn malformed_sequences_are_input_errors() {
    let robot = panda();
    let n = robot.dof();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let (paths, _) = corner(&device, &worlds, 0.1, 0.1);
    let o = JoinOptions::default();
    let input = |sequence: Sequence, o: &JoinOptions| {
        matches!(
            batchplan::join(&device, &worlds, &[sequence], o),
            Err(Error::Input(_))
        )
    };
    let sequence = |paths: Vec<Vec<f32>>| Sequence {
        world: 0,
        blends: vec![BLEND; paths.len().saturating_sub(1)],
        paths,
        constraints: vec![],
    };

    let mut apart = paths.clone();
    apart[1].iter_mut().take(3 * n).for_each(|v| *v += 0.1);
    assert!(input(sequence(apart), &o), "motions that do not meet");
    for k in 0..2 {
        let mut moving = paths.clone();
        moving[k][n] += 0.01;
        assert!(input(sequence(moving), &o), "path {k} starts moving");
    }
    let mut short = paths.clone();
    short[0].pop();
    assert!(
        input(sequence(short), &o),
        "a path that is not whole configurations"
    );
    assert!(input(sequence(vec![]), &o), "no paths");
    assert!(
        input(
            Sequence {
                world: 1,
                ..sequence(paths.clone())
            },
            &o
        ),
        "a world out of range"
    );
    for blends in [vec![], vec![BLEND; 2]] {
        let wrong = Sequence {
            blends,
            ..sequence(paths.clone())
        };
        assert!(input(wrong, &o), "a radius for each path, not each join");
    }
    for radius in [-0.01, f32::NAN] {
        let wide = Sequence {
            blends: vec![radius],
            ..sequence(paths.clone())
        };
        assert!(input(wide, &o), "a blend radius of {radius}");
    }
    let elsewhere = Sequence {
        constraints: vec![HandConstraint::new(1, Pose::IDENTITY)],
        ..sequence(paths.clone())
    };
    assert!(
        input(elsewhere, &o),
        "a constraint on a hand the robot lacks"
    );
    for resolution in [f32::INFINITY, f32::NAN, 1e-30, 0.0] {
        let unchecked = JoinOptions { resolution };
        assert!(
            input(sequence(paths.clone()), &unchecked),
            "a resolution of {resolution}"
        );
    }
}

#[test]
fn longer_sequences_never_finish_later_than_their_motions() {
    // Random planned sequences of three to five motions on the SO-101: each join's blend is weighed
    // against the stretch so far, so no stretch of joined motions takes longer than they do apart.
    let robot = Robot::load(
        common::asset("so101/so101_new_calib.urdf"),
        &Default::default(),
    )
    .unwrap();
    let n = robot.dof();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let mut rng = batchplan::rng::Rng::new(3);
    let mut random = || -> Vec<f32> {
        (0..n)
            .map(|j| rng.range(robot.lower()[j], robot.upper()[j]))
            .collect()
    };
    let mut sequences = vec![];
    for k in 0..60 {
        let stops: Vec<Vec<f32>> = std::iter::once(robot.default_q().to_vec())
            .chain((0..3 + k % 3).map(|_| random()))
            .collect();
        let problems: Vec<PlanProblem> = stops
            .windows(2)
            .map(|w| PlanProblem::new(0, w[0].clone(), w[1].clone()))
            .collect();
        let plans = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap();
        let paths: Option<Vec<Vec<f32>>> = (0..problems.len())
            .map(|i| plans.best(i).map(<[f32]>::to_vec))
            .collect();
        sequences.extend(paths);
    }
    assert!(
        sequences.len() >= 20,
        "{} sequences planned",
        sequences.len()
    );
    for blend in [0.02, 0.05, 0.1] {
        let sequences: Vec<Sequence> = sequences
            .iter()
            .map(|paths| Sequence {
                world: 0,
                paths: paths.clone(),
                blends: vec![blend; paths.len() - 1],
                constraints: vec![],
            })
            .collect();
        let joined =
            batchplan::join(&device, &worlds, &sequences, &JoinOptions::default()).unwrap();
        for (i, (stretches, sequence)) in joined.paths.iter().zip(&sequences).enumerate() {
            let (joined, apart) = (
                duration(&robot, stretches),
                duration(&robot, &sequence.paths),
            );
            assert!(
                joined <= apart,
                "blend {blend} m, sequence {i} of {} motions: {joined} s joined, {apart} s apart",
                sequence.paths.len()
            );
        }
    }
}

/// Planned paths through `stops`, one per pair.
fn planned(device: &Device, worlds: &Worlds, stops: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let problems: Vec<PlanProblem> = stops
        .windows(2)
        .map(|w| PlanProblem::new(0, w[0].clone(), w[1].clone()))
        .collect();
    let plans = batchplan::plan(device, worlds, &problems, &PlanOptions::default()).unwrap();
    let paths: Vec<Vec<f32>> = plans.solved().map(|s| s.solution.to_vec()).collect();
    assert_eq!(paths.len(), problems.len(), "every motion plans");
    paths
}

#[test]
fn a_kept_stop_ends_one_stretch_at_rest_and_starts_the_next() {
    // A sweep that blends, then a motion of one span, which leaves no room for a blend.
    let robot = panda();
    let n = robot.dof();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let mut paths = planned(
        &device,
        &worlds,
        &[
            vec![-1.2, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8],
            vec![-0.4, -0.3, 0.1, -1.9, 0.1, 1.7, 0.7],
            vec![0.4, -0.2, 0.2, -1.8, 0.2, 1.8, 0.6],
        ],
    );
    let end = paths[1][paths[1].len() - n..].to_vec();
    let on: Vec<f32> = end
        .iter()
        .enumerate()
        .map(|(j, v)| v + if j == 0 { 0.01 } else { 0.0 })
        .collect();
    let middle: Vec<f32> = end.iter().zip(&on).map(|(a, b)| 0.5 * (a + b)).collect();
    paths.push([end.repeat(3), middle, on.repeat(3)].concat());
    let joined = join(&device, &worlds, paths.clone(), BLEND);
    assert!(
        matches!(joined.blends[0][..], [Some(_), None]),
        "{:?}",
        joined.blends[0]
    );
    let stretches = &joined.paths[0];
    assert_eq!(stretches.len(), 2);
    assert_eq!(stretches[1], paths[2], "a lone motion is as given");
    assert_eq!(
        &stretches[0][stretches[0].len() - n..],
        &paths[2][..n],
        "they meet"
    );
    for stretch in stretches {
        Trajectory::try_new(&robot, stretch, 1.0)
            .unwrap()
            .check(&robot)
            .unwrap();
    }
}

#[test]
fn each_join_turns_within_its_own_radius() {
    // The sweep of three motions, joined at radii that differ per join, as each item of a Pilz
    // sequence gives its own.
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let paths = planned(
        &device,
        &worlds,
        &[
            vec![-1.2, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8],
            vec![-0.4, -0.3, 0.1, -1.9, 0.1, 1.7, 0.7],
            vec![0.4, -0.2, 0.2, -1.8, 0.2, 1.8, 0.6],
            vec![1.2, -0.4, 0.3, -2.0, 0.3, 1.9, 0.5],
        ],
    );
    let sequences: Vec<Sequence> = [vec![BLEND, 0.0], vec![0.0, BLEND], vec![0.2, BLEND]]
        .into_iter()
        .map(|blends| Sequence {
            world: 0,
            paths: paths.clone(),
            blends,
            constraints: vec![],
        })
        .collect();
    let joined = batchplan::join(&device, &worlds, &sequences, &JoinOptions::default()).unwrap();
    for (sequence, (blends, stretches)) in sequences
        .iter()
        .zip(joined.blends.iter().zip(&joined.paths))
    {
        let asked = &sequence.blends;
        for (k, (&radius, &got)) in asked.iter().zip(blends).enumerate() {
            match got {
                None => assert_eq!(radius, 0.0, "join {k} of {asked:?} stops"),
                Some(got) => assert!(
                    got > 0.0 && got <= radius,
                    "join {k} of {asked:?} turns within {got} m"
                ),
            }
        }
        let stops = asked.iter().filter(|&&r| r == 0.0).count();
        assert_eq!(
            stretches.len(),
            1 + stops,
            "{asked:?}: a stretch per stop and one more"
        );
    }
}

#[test]
fn a_stop_separates_the_joins_either_side_of_it() {
    // Random planned motions joined at radii [r, 0, r] join as the two sequences either side of
    // the stop, joined alone: how one side is split to time it never depends on the other.
    let robot = panda();
    let n = robot.dof();
    let device = Device::cpu(&robot).unwrap();
    let worlds = device.upload(&[World::default()]).unwrap();
    let mut rng = batchplan::rng::Rng::new(5);
    let mut compared = 0;
    for trial in 0..24 {
        let stops: Vec<Vec<f32>> = (0..5)
            .map(|_| {
                (0..n)
                    .map(|j| {
                        let (lower, upper) = (robot.lower()[j], robot.upper()[j]);
                        let middle = 0.5 * (lower + upper);
                        rng.range(
                            middle - 0.3 * (upper - lower),
                            middle + 0.3 * (upper - lower),
                        )
                    })
                    .collect()
            })
            .collect();
        let problems: Vec<PlanProblem> = stops
            .windows(2)
            .map(|w| PlanProblem::new(0, w[0].clone(), w[1].clone()))
            .collect();
        let plans = batchplan::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap();
        let Some(paths) = (0..4)
            .map(|i| plans.best(i).map(<[f32]>::to_vec))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let r = [0.02, 0.05, 0.1][trial % 3];
        let sequence = |paths: &[Vec<f32>], blends: Vec<f32>| Sequence {
            world: 0,
            paths: paths.to_vec(),
            blends,
            constraints: vec![],
        };
        let sequences = [
            sequence(&paths, vec![r, 0.0, r]),
            sequence(&paths[..2], vec![r]),
            sequence(&paths[2..], vec![r]),
        ];
        let joined =
            batchplan::join(&device, &worlds, &sequences, &JoinOptions::default()).unwrap();
        let apart: Vec<Vec<f32>> = joined.paths[1]
            .iter()
            .chain(&joined.paths[2])
            .cloned()
            .collect();
        let blends: Vec<Option<f32>> = [joined.blends[1][0], None, joined.blends[2][0]].to_vec();
        assert_eq!(joined.blends[0], blends, "trial {trial}, {r} m");
        assert_eq!(joined.paths[0], apart, "trial {trial}, {r} m");
        compared += 1;
    }
    assert!(compared >= 16, "{compared} trials planned");
}

/// Blends keep the sequence's constraints: motions planned with the gripper within 2° of straight
/// down, joined 30 cm wide, stay within it once timed, where the same blends without the
/// constraint tilt it past (to 11.0° on the Ryzen, 4.3° on the M4 Pro). The goals are seed 4's:
/// once splines passed through points projected inside the cone, seed 3's unconstrained blends
/// stayed within 1.14°, inside it, on the Ryzen and on its Radeon (2026-11-03). Blends between
/// motions that keep the cone stray past it only where they are wide.
#[test]
fn blends_keep_the_sequence_constraints() {
    let robot = panda();
    let down = common::upright(0.0);
    let upright = HandConstraint {
        tilt: 2f32.to_radians(),
        ..HandConstraint::new(0, Pose::new(Vec3::ZERO, down))
    };
    let tilt = |q: &[f32]| (robot.ee_pose(q, 0).rotation * Vec3::Z).angle_between(down * Vec3::Z);
    // The same stops for every device: IK on the GPU lands a little elsewhere.
    let cpu = Device::cpu(&robot).unwrap();
    let scene = common::pillar_scene();
    let reaching = cpu.upload(std::slice::from_ref(&scene)).unwrap();
    let (start, goals) = common::upright_reaches(&cpu, &reaching, 24, &mut Rng::new(4)).unwrap();
    let mut stops = vec![start];
    stops.extend(goals.into_iter().take(19));
    for device in devices(&robot) {
        let worlds = device.upload(std::slice::from_ref(&scene)).unwrap();
        let problems: Vec<PlanProblem> = stops
            .windows(2)
            .map(|w| PlanProblem {
                constraints: vec![upright],
                ..PlanProblem::new(0, w[0].clone(), w[1].clone())
            })
            .collect();
        let planned =
            batchplan::plan(&device, &worlds, &problems, &PlanOptions::default()).unwrap();
        let paths: Vec<Vec<f32>> = (0..problems.len())
            .map(|i| planned.best(i).expect("an upright motion").to_vec())
            .collect();
        let most = |constraints: Vec<HandConstraint>| {
            let sequence = Sequence {
                world: 0,
                blends: vec![0.3; paths.len() - 1],
                paths: paths.clone(),
                constraints,
            };
            let joined =
                batchplan::join(&device, &worlds, &[sequence], &JoinOptions::default()).unwrap();
            assert!(
                joined.blends[0].iter().any(Option::is_some),
                "nothing blended"
            );
            joined.paths[0]
                .iter()
                .flat_map(|p| samples(&robot, p))
                .collect::<Vec<f32>>()
                .chunks(robot.dof())
                .map(tilt)
                .fold(0.0, f32::max)
        };
        let (kept, free) = (most(vec![upright]), most(vec![]));
        assert!(
            kept <= upright.tilt + 1e-5 && free > upright.tilt + 1e-3,
            "{}: the joined motion tilts {:.3}° with the constraint and {:.3}° without",
            device.name(),
            kept.to_degrees(),
            free.to_degrees()
        );
    }
}
