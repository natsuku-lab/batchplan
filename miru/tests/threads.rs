//! A look, and the closures measured in what it kept, run only on their device's threads. This
//! file holds a single test, so no other test starts threads beside it. Threads are counted on
//! Linux and macOS.
#![cfg(any(target_os = "linux", target_os = "macos"))]

#[allow(dead_code, reason = "each test file uses part of it")]
mod common;

use std::collections::HashMap;

use batchplan::{Device, GraspOptions, Intrinsics, Pose};
use common::{Frame, block, device_of, known, looking, panda, table};
use glam::{Quat, Vec3};
use miru::{Camera, Look, LookOptions, Memory, MemoryOptions, Scene, Stance, observe};

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

    // The same views through `observe` on a device of the Panda's, and the closures of its hand
    // at two poses over the box in what that look kept.
    let robot = panda();
    let device = Device::cpu_threads(&robot, 1).expect("a CPU device");
    assert_eq!(
        threads(),
        at_start + 2,
        "the Panda's device starts exactly one worker"
    );
    let cameras: Vec<Camera> = (frames.iter())
        .map(|frame| Camera {
            image: frame.view().image,
            color: &frame.color,
            noise: frame.noise,
            band: 0,
            off_by: 0.0,
            masks: &frame.masks,
        })
        .collect();
    let memory = Memory::try_new(&device, shapes.to_vec(), &MemoryOptions::default());
    let locked = HashMap::new();
    let stance = Stance {
        q: robot.default_q(),
        locked: &locked,
        padding: 0.0,
    };
    let seen = observe(&device, &cameras, &stance, &memory.expect("a memory"));
    let kept = seen.expect("a sound look").kept;
    let [object] = &kept.scene().objects[..] else {
        panic!("one object: {:?}", kept.scene().objects);
    };
    let down = Quat::from_rotation_x(std::f32::consts::PI);
    let poses = [0.055, 0.2].map(|z| Pose::new(Vec3::new(0.5, 0.0, z), down));
    let closures = kept.observed_closures(
        &device,
        object.id,
        0,
        0.0..=0.08,
        &poses,
        &GraspOptions::default(),
    );
    let closures = closures.expect("sound poses").expect("the box is seen");
    assert_eq!(closures.len(), poses.len());
    assert_eq!(
        threads(),
        at_start + 2,
        "the look or the closures started other threads"
    );
}
