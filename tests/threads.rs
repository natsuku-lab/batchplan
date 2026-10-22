//! The CPU device runs batched work only on threads it starts itself. This file holds a single
//! test, so no other test starts threads beside it. Threads are counted on Linux and macOS.
#![cfg(any(target_os = "linux", target_os = "macos"))]

#[path = "../examples/common/mod.rs"]
mod common;

use batchplan::rng::Rng;
use batchplan::*;

#[cfg(target_os = "linux")]
fn threads() -> usize {
    std::fs::read_dir("/proc/self/task").unwrap().count()
}

#[cfg(target_os = "macos")]
fn threads() -> usize {
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as i32;
    let pid = std::process::id() as i32;
    // SAFETY: `info` is a writable `proc_taskinfo` of exactly `size` bytes.
    let written =
        unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDTASKINFO, 0, (&raw mut info).cast(), size) };
    assert_eq!(written, size, "proc_pidinfo failed");
    info.pti_threadnum as usize
}

#[test]
fn a_one_thread_device_works_on_its_one_thread() {
    let at_start = threads();
    let robot = common::panda().unwrap();
    let device = Device::cpu_threads(&robot, 1).unwrap();
    assert!(device.name().contains("(1 threads)"), "{}", device.name());
    assert_eq!(
        threads(),
        at_start + 1,
        "the device starts exactly one worker"
    );
    let mut rng = Rng::new(3);
    let scene: Vec<World> = (0..2).map(|_| common::tabletop(&mut rng)).collect();
    let worlds = device.upload(&scene).unwrap();
    let goals: Vec<IkProblem> = (0..2)
        .map(|w| IkProblem::new(w, common::grasp_target(&scene[w as usize], &mut rng)))
        .collect();
    let ik = solve_ik(&device, &worlds, &goals, &IkOptions::default()).unwrap();
    let problems: Vec<PlanProblem> = ik
        .solved()
        .map(|s| PlanProblem::new(s.problem.world, robot.default_q(), s.solution))
        .collect();
    assert!(!problems.is_empty());
    assert!(
        plan(&device, &worlds, &problems, &PlanOptions::default())
            .unwrap()
            .solved()
            .count()
            > 0
    );
    let searches: Vec<RrtProblem> = problems
        .iter()
        .map(|p| RrtProblem {
            world: p.world,
            start: p.start.clone(),
            goals: vec![p.goal.clone()],
            hold: vec![],
        })
        .collect();
    let found = rrt::connect(&device, &worlds, &searches, &RrtOptions::default()).unwrap();
    let (world, mut paths): (Vec<u32>, Vec<Vec<f32>>) = problems
        .iter()
        .zip(found.paths)
        .filter_map(|(p, path)| Some((p.world, path?)))
        .unzip();
    assert!(!paths.is_empty());
    shortcut::shortcut(
        &device,
        &worlds,
        &world,
        &mut paths,
        &ShortcutOptions::default(),
    )
    .unwrap();
    // Picking something up mid-task fits its spheres on the calling thread.
    let held = robot
        .attach(&AttachedObject {
            name: "box".into(),
            link: "panda_hand".into(),
            shapes: vec![Obstacle::sphere(glam::Vec3::new(0.0, 0.0, 0.1), 0.03)],
            touch_links: vec!["panda_leftfinger".into(), "panda_rightfinger".into()],
            spheres: SphereOptions {
                budget: 4,
                ..Default::default()
            },
        })
        .unwrap();
    device.with_robot(&held).unwrap();
    assert_eq!(
        threads(),
        at_start + 1,
        "batched work started other threads"
    );
}
