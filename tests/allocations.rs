//! Batched calls build each problem's search on the thread that runs it, so the memory a search
//! grows belongs to that thread's allocator arena. Built on the calling thread, every thread of a
//! batch waited for the caller's arena lock. A global allocator of this file's own counts the
//! calling thread's allocations.

#[path = "../examples/common/mod.rs"]
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use batchplan::rrt::{RrtOptions, RrtProblem, connect};
use batchplan::shortcut::shortcut;
use batchplan::{Device, PlanOptions};

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: every call goes to the system allocator unchanged; only a thread-local count is added.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|a| a.set(a.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|a| a.set(a.get() + 1));
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations `f` makes on the calling thread.
fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn batches_build_their_searches_on_the_threads_that_run_them() {
    let robot = common::panda().unwrap();
    let (scene, problems) = common::reaches(&robot);
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let worlds = device.upload(&scene).unwrap();
    let (few, many) = (8, 40);
    // The calling thread's allocations per problem beyond the first `few`.
    let per_problem = |small: usize, large: usize| (large - small) as f64 / (many - few) as f64;

    let o = PlanOptions::default();
    let small =
        allocations(|| drop(batchplan::plan(&device, &worlds, &problems[..few], &o).unwrap()));
    let large =
        allocations(|| drop(batchplan::plan(&device, &worlds, &problems[..many], &o).unwrap()));
    let plans = per_problem(small, large);

    let searches: Vec<RrtProblem> = problems[..many]
        .iter()
        .map(|p| RrtProblem {
            world: p.world,
            start: p.start.clone(),
            goals: vec![p.goal.clone()],
            hold: vec![],
        })
        .collect();
    let small = allocations(|| {
        drop(connect(&device, &worlds, &searches[..few], &RrtOptions::default()).unwrap())
    });
    let large =
        allocations(|| drop(connect(&device, &worlds, &searches, &RrtOptions::default()).unwrap()));
    let searched = per_problem(small, large);

    let found = connect(&device, &worlds, &searches, &RrtOptions::default()).unwrap();
    let (world, paths): (Vec<u32>, Vec<Vec<f32>>) = searches
        .iter()
        .zip(found.paths)
        .filter_map(|(p, path)| Some((p.world, path?)))
        .unzip();
    assert!(
        paths.len() >= many - 4,
        "{} of {many} searches found a path",
        paths.len()
    );
    let shortcuts = |k: usize| {
        let mut paths = paths[..k].to_vec();
        allocations(|| {
            shortcut(
                &device,
                &worlds,
                &world[..k],
                &mut paths,
                &Default::default(),
            )
            .unwrap()
        })
    };
    let shortened = (shortcuts(paths.len()) - shortcuts(few)) as f64 / (paths.len() - few) as f64;

    // Built on their threads: 3, 4 and 1, the results' copies of the problems; built on the
    // calling thread: 32, 16 and 7.
    assert!(
        plans <= 8.0,
        "plan allocates {plans:.1} per problem on the calling thread"
    );
    assert!(
        searched <= 8.0,
        "connect allocates {searched:.1} per problem on the calling thread"
    );
    assert!(
        shortened <= 3.0,
        "shortcut allocates {shortened:.1} per path on the calling thread"
    );
}
