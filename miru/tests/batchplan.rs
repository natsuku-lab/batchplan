//! A look's output through batchplan's own calls, on a real robot's device: the Franka Panda
//! with the collision spheres the library's examples use, its arm reaching over a table with a
//! box, a can and a tray on it. The fixtures and the solids of the solid objects upload as a
//! `World`; `RobotMask::with_shapes` takes them for the view's image and flags the readings the
//! scene labelled as theirs; the scene's labels, as flags, are what `Surface::from_depth` takes;
//! the arm's mask from `DepthImage::robot_mask` is the view's `robot`, and no object is made of
//! the arm. All of it on a CPU device of one thread, of all of them, and on the GPU's where there
//! is one (miru's `gpu` feature; skipped with a message where there is none, unless
//! `BATCHPLAN_REQUIRE_GPU` is set), where the scene is the same.

#[allow(dead_code, reason = "each test file uses part of it")]
mod common;

use batchplan::{Surface, World};
use common::{
    EYE, Frame, MIDDLE, NOISE, arm, block, can, devices, known, looking, panda, table, thin_tray,
};
use glam::Vec3;
use miru::{Label, Look, LookOptions, Scene};

#[test]
fn a_looks_output_goes_through_batchplans_calls_on_a_robots_device() {
    let robot = panda();
    // The hand 24 cm over the table, 58 cm out and 21 cm to the left, beside the tray: the arm
    // crosses the view's far half.
    let q = [0.35, 0.3, 0.0, -1.9, 0.0, 2.2, 0.8];
    // Things 1 to 7: a box, a can, the tray's floor and four walls; thing 8 the arm.
    let (the_box, the_can) = (Vec3::new(0.5, -0.2, 0.03), Vec3::new(0.55, 0.2, 0.05));
    let mut things = vec![
        table(),
        block(the_box, Vec3::new(0.04, 0.06, 0.03), [200, 30, 30]),
        can(the_can, 0.035, 0.05, [30, 30, 200]),
    ];
    things.extend(thin_tray());
    things.push(arm(&robot, &q));
    let arm_thing = things.len() - 1;
    // Each thing's parts: the box, the can and the tray.
    let parts = [1..2, 2..3, 3..8];
    let mut frame = Frame::of(&things, looking(EYE, MIDDLE));
    let on_arm: Vec<usize> = (0..frame.shows.len())
        .filter(|&i| frame.shows[i] == Some(arm_thing))
        .collect();
    assert!(on_arm.len() > 1000, "{} pixels show the arm", on_arm.len());

    // The robot's mask, padded by three deviations of the view's noise at the arm's farthest
    // reading: the arm is drawn as its spheres, and the depth reads them to within its noise.
    let farthest = on_arm.iter().map(|&i| frame.depth[i]).fold(0.0, f32::max);
    let padding = 3.0 * NOISE.quadratic * farthest * farthest;
    let mask = frame.view().image.robot_mask(&robot, &q, padding, &[]);
    let mask = mask.expect("a sound mask");
    frame.robot.clone_from(&mask.on_robot);
    assert!(
        on_arm.iter().all(|&i| frame.robot[i]),
        "the mask holds the arm"
    );

    let known = [known(&table())];
    let view = frame.view();
    let image = view.image;
    let views = [view];
    let look = Look {
        views: &views,
        known: &known,
    };
    let mut first: Option<Scene> = None;
    for (name, device) in devices(&robot) {
        let scene = miru::look(&device, &look, &Scene::default(), &LookOptions::default());
        let scene = scene.expect("a sound look");
        let same = first.get_or_insert_with(|| scene.clone());
        assert_eq!(&scene, same, "the scene on {name}");

        // No object is made of the arm: every pixel its mask holds is the robot's.
        let labels = &scene.labels[0];
        let robots = (0..labels.len()).filter(|&i| frame.robot[i]);
        assert!(robots.clone().all(|i| labels[i] == Label::Robot), "{name}");
        // The box, the can and the tray are an object each, and solid: their shapes stand for
        // them.
        let objects: Vec<usize> = (parts.clone())
            .map(|thing| {
                let its = |i: usize| frame.shows[i].is_some_and(|part| thing.contains(&part));
                let shows = (0..labels.len()).filter(|&i| its(i));
                let held = shows.filter_map(|i| match labels[i] {
                    Label::Object(object) => Some(object),
                    _ => None,
                });
                let mut held: Vec<usize> = held.collect();
                held.sort_unstable();
                held.dedup();
                assert_eq!(
                    held.len(),
                    1,
                    "{name}: things {thing:?} in objects {held:?}"
                );
                held[0]
            })
            .to_vec();
        assert_eq!(scene.objects.len(), 3, "{name}: {:?}", scene.objects);
        assert!(scene.objects.iter().all(|object| object.solid), "{name}");

        // The fixtures and the solids of the solid objects are a world on the robot's device,
        // which holds each thing where it is and the tray's room free.
        let solids = (scene.objects.iter()).flat_map(|object| object.shape.solids());
        let shapes: Vec<_> = known.iter().cloned().chain(solids).collect();
        let world = World::new(shapes.clone());
        device.upload(&[world]).expect("a world the device takes");
        let inside = |p: Vec3| {
            shapes[1..]
                .iter()
                .any(|shape| shape.distance(p).value < 0.0)
        };
        let floor = Vec3::new(0.5, 0.0, 0.005);
        assert!(
            inside(the_box) && inside(the_can) && inside(floor),
            "{name}"
        );
        assert!(
            !inside(Vec3::new(0.5, 0.0, 0.04)),
            "{name}: the tray's room is free"
        );

        // Those shapes, held by the mask within three deviations of the view's noise, take every
        // reading the scene labelled as theirs: the table's and the objects'.
        let margin = [
            3.0 * NOISE.constant,
            3.0 * NOISE.linear,
            3.0 * NOISE.quadratic,
        ];
        let held = mask.clone().with_shapes(&device, &image, &shapes, margin);
        let held = held.expect("shapes the mask takes");
        let theirs =
            (0..labels.len()).filter(|&i| matches!(labels[i], Label::Known(_) | Label::Object(_)));
        let left: Vec<usize> = theirs.filter(|&i| !held.on_robot[i]).collect();
        assert!(
            left.is_empty(),
            "{name}: {} readings the shapes do not take",
            left.len()
        );

        // Each object's labels, as flags, give a surface whose object points are its readings,
        // each within three deviations of the view's noise of its thing. The surface has a point
        // per reading off the robot, row by row.
        let off_robot = (0..labels.len()).filter(|&i| frame.depth[i] > 0.0 && !mask.on_robot[i]);
        let off_robot: Vec<usize> = off_robot.collect();
        for (thing, &object) in parts.iter().zip(&objects) {
            let flags = labels.iter().map(|label| *label == Label::Object(object));
            let flags: Vec<bool> = flags.collect();
            let surface = Surface::from_depth(&device, &image, &flags, Some(&mask), 0.005);
            let surface = surface.expect("a sound surface");
            assert_eq!(surface.points.len(), off_robot.len(), "{name}");
            let mut points = 0;
            for (k, &i) in off_robot.iter().enumerate() {
                let its = labels[i] == Label::Object(object);
                assert_eq!(surface.object[k], its, "{name}: pixel {i}");
                if its {
                    let p = surface.points[k];
                    let off = (thing.clone()).map(|part| (things[part].distance)(p).abs());
                    let reach = 3.0 * NOISE.quadratic * frame.depth[i] * frame.depth[i];
                    assert!(
                        off.fold(f32::INFINITY, f32::min) <= reach,
                        "{name}: pixel {i}"
                    );
                    points += 1;
                }
            }
            assert!(
                points > 100,
                "{name}: things {thing:?} have {points} points"
            );
        }
    }
}
