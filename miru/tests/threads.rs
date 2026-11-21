//! A look runs only on its device's threads. This file holds a single test, so no other test
//! starts threads beside it. Threads are counted on Linux and macOS.
#![cfg(any(target_os = "linux", target_os = "macos"))]

#[allow(dead_code, reason = "each test file uses part of it")]
mod common;

use batchplan::Intrinsics;
use common::{Frame, block, device_of, known, looking, table};
use glam::Vec3;
use miru::{Look, LookOptions, Scene};

#[cfg(target_os = "linux")]
fn threads() -> usize {
    std::fs::read_dir("/proc/self/task").unwrap().count()
}

#[cfg(target_os = "macos")]
fn threads() -> usize {
    // SAFETY: `proc_taskinfo` is a C struct of integers, for which all zero bytes are a value.
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
fn a_look_on_a_one_thread_device_works_on_its_one_thread() {
    let at_start = threads();
    let device = device_of(Some(1));
    assert_eq!(
        threads(),
        at_start + 1,
        "the device starts exactly one worker"
    );
    // Two views of a box on the table, the table known, at half the tests' usual size.
    let things = [
        table(),
        block(
            Vec3::new(0.5, 0.0, 0.03),
            Vec3::new(0.04, 0.06, 0.03),
            [200, 30, 30],
        ),
    ];
    let k = Intrinsics {
        fx: 150.0,
        fy: 150.0,
        cx: 79.5,
        cy: 59.5,
    };
    let target = Vec3::new(0.5, 0.0, 0.0);
    let frames = [Vec3::new(1.1, 0.0, 0.5), Vec3::new(0.5, 0.6, 0.5)]
        .map(|eye| Frame::through(&things, looking(eye, target), k, (160, 120)));
    let views = frames.each_ref().map(Frame::view);
    let shapes = [known(&things[0])];
    let look = Look {
        views: &views,
        known: &shapes,
    };
    let scene = miru::look(&device, &look, &Scene::default(), &LookOptions::default());
    let scene = scene.expect("a sound look");
    assert_eq!(scene.objects.len(), 1, "{:?}", scene.objects);
    assert_eq!(threads(), at_start + 1, "the look started other threads");
}
