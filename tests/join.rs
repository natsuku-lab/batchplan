//! Joining motions without stopping (`join`), on every device: the joined motion turns within the
//! blend radius of each join and is the motions themselves elsewhere, finishes sooner than the
//! motions one after another and never later, keeps the limits, and refuses sequences whose
//! motions do not meet. Falling back from blends that collide is tested in `src/join.rs`.

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::{
    Device, Error, JoinOptions, JoinResult, PlanOptions, PlanProblem, Pose, Robot, Sequence,
    ToolPath, ToolPathOptions, Trajectory, World, Worlds,
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

fn join(device: &Device, worlds: &Worlds, paths: Vec<Vec<f32>>, o: &JoinOptions) -> JoinResult {
    let sequence = Sequence { world: 0, paths };
    batchplan::join(device, worlds, &[sequence], o).unwrap()
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
    let o = JoinOptions::default();
    let precision = ToolPathOptions::default().precision;
    for device in devices(&robot) {
        let name = device.name();
        let worlds = device.upload(&[World::default()]).unwrap();
        let (paths, [start, corner, end]) = self::corner(&device, &worlds, 0.1, 0.1);
        let joined = join(&device, &worlds, paths.clone(), &o);
        let radius = joined.blends[0][0].expect("a free corner blends");
        assert!(
            radius > 0.0 && radius <= o.blend,
            "{name}: blend {radius} m"
        );
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
    let o = JoinOptions {
        blend: 0.0,
        ..Default::default()
    };
    let joined = join(&device, &worlds, paths.clone(), &o);
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
            let o = JoinOptions {
                blend,
                ..Default::default()
            };
            let joined = join(&device, &worlds, paths.clone(), &o);
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
        let joined = join(&device, &worlds, paths.clone(), &JoinOptions::default());
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
    let sequence = |paths: Vec<Vec<f32>>| Sequence { world: 0, paths };

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
                paths: paths.clone()
            },
            &o
        ),
        "a world out of range"
    );
    let wide = JoinOptions {
        blend: -0.01,
        ..Default::default()
    };
    assert!(input(sequence(paths.clone()), &wide), "a negative blend");
    let coarse = JoinOptions {
        resolution: 0.0,
        ..Default::default()
    };
    assert!(input(sequence(paths), &coarse), "no resolution");
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
        sequences.extend(paths.map(|paths| Sequence { world: 0, paths }));
    }
    assert!(
        sequences.len() >= 20,
        "{} sequences planned",
        sequences.len()
    );
    for blend in [0.02, 0.05, 0.1] {
        let o = JoinOptions {
            blend,
            ..Default::default()
        };
        let joined = batchplan::join(&device, &worlds, &sequences, &o).unwrap();
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
    let joined = join(&device, &worlds, paths.clone(), &JoinOptions::default());
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
