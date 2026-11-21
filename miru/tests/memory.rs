//! The look as one call (`miru::observe`) and the memory it carries from look to look, through
//! the interface, on scenes rendered from known things with a Franka Panda before them, its arm
//! drawn as its collision spheres. What is claimed of a scene is read the three ways
//! `tests/truth.rs` reads its scenes (`common::CAMERAS`): both views by the renderer's exact
//! depth, both by a D435, and the far one by a D435 and the near one by a D405. A stereo camera's
//! depth is given as it reads it, unfiltered: `observe` filters it. Where a stereo camera breaks a
//! claim that holds on exact depth, the claim's camera is listed in [`GAPS`] with what it shows.

#[allow(dead_code, reason = "each test file uses part of it")]
mod common;

use std::collections::HashMap;

use batchplan::{DepthFilter, DepthImage, Device, Robot, RobotMask};
use common::{
    CAMERAS, EYE, Frame, MIDDLE, NEAR, Stereo, Thing, arm, block, can, devices, known, looking,
    panda, table, thin_tray,
};
use glam::Vec3;
use miru::{
    Camera, Label, Look, LookOptions, Memory, MemoryOptions, Noise, Scene, Stance, observe,
};

/// The Panda's stance with its fingertips 2.3 cm over the top of the can of
/// [`reaching_over_a_tray`], its hand turned as `tests/batchplan.rs` holds it and the arm across
/// the far view: what reads of the can's top lies nearer the fingers than a map's mask pads them.
const OVER_THE_CAN: [f32; 7] = [0.349, 0.389, 0.0, -2.091, 0.0, 2.480, 0.799];

/// The claims a stereo camera breaks, a line each: the test, the camera, and what it shows.
const GAPS: &str = "
";

/// Whether `test`'s claim through `camera` is a listed gap.
fn gap(test: &str, camera: &str) -> bool {
    let lines = GAPS.lines().filter(|line| !line.is_empty());
    let fields = lines.map(|line| line.split(" | ").collect::<Vec<_>>());
    fields
        .into_iter()
        .any(|fields| fields[..2] == [test, camera])
}

/// Asserts what `test` found through each camera: that it holds where no gap is listed, and that
/// a listed gap still is one.
fn verdict(test: &str, found: &[(&str, Result<(), String>)]) {
    let wrong: Vec<String> = (found.iter())
        .filter_map(|(camera, found)| match (found, gap(test, camera)) {
            (Ok(()), false) | (Err(_), true) => None,
            (Ok(()), true) => Some(format!("{camera}: a listed gap holds")),
            (Err(why), false) => Some(format!("{camera}: {why}")),
        })
        .collect();
    assert!(wrong.is_empty(), "{test}:\n{}", wrong.join("\n"));
}

/// What a test found through one camera: what it found wrong, or nothing.
fn outcome(wrong: &[String]) -> Result<(), String> {
    match wrong.is_empty() {
        true => Ok(()),
        false => Err(wrong.join("; ")),
    }
}

/// The far view's frame and the near one's of `things`, as the renderer gives the truth.
fn render(things: &[Thing]) -> [Frame; 2] {
    [EYE, NEAR].map(|eye| Frame::of(things, looking(eye, MIDDLE)))
}

/// `rendered` as camera set `c` of [`CAMERAS`] reads it with the draws of `seed`, unfiltered, and
/// each camera's band: none for exact depth, and one for the tests' stereo cameras, which read
/// each pixel's disparity a fraction of a pixel off (`RobotMask::with_band`'s rule).
fn read(rendered: &[Frame; 2], c: usize, seed: u64) -> ([Frame; 2], [usize; 2]) {
    let (_, far, near) = CAMERAS[c];
    let one = |frame: &Frame, stereo: Option<Stereo>, seed: u64| match stereo {
        Some(stereo) => (frame.clone().sensed(stereo, seed), 1),
        None => (frame.clone(), 0),
    };
    let ((far, far_band), (near, near_band)) = (
        one(&rendered[0], far, seed),
        one(&rendered[1], near, seed + 1),
    );
    ([far, near], [far_band, near_band])
}

/// The standard deviation of a reading at `depth` by `noise` (meters).
fn deviation(noise: Noise, depth: f32) -> f32 {
    noise.constant + (noise.linear + noise.quadratic * depth) * depth
}

/// How far past its spheres the arm, thing `arm` of the scene, reads in `frames`: three
/// deviations of each frame's noise at the arm's farthest pixel, the larger of the two. The arm is
/// drawn as its spheres, so they are off its surface by nothing, and each camera reads them to
/// within its noise.
fn padding(rendered: &[Frame; 2], frames: &[Frame; 2], arm: usize) -> f32 {
    let each = (rendered.iter().zip(frames)).map(|(truth, frame)| {
        let on_arm = (0..truth.depth.len()).filter(|&i| truth.shows[i] == Some(arm));
        let farthest = on_arm.map(|i| truth.depth[i]).fold(0.0, f32::max);
        3.0 * deviation(frame.noise, farthest)
    });
    each.fold(0.0, f32::max)
}

/// `frames` as the cameras `observe` takes, each with its band, its pose and the known shapes
/// off by `off_by`.
fn cameras(frames: &[Frame; 2], bands: [usize; 2], off_by: f32) -> Vec<Camera<'_>> {
    (frames.iter().zip(bands))
        .map(|(frame, band)| Camera {
            image: frame.view().image,
            color: &frame.color,
            noise: frame.noise,
            band,
            off_by,
            masks: &frame.masks,
        })
        .collect()
}

/// A memory of nothing seen, in a cell that knows its table.
fn memory(device: &Device) -> Memory {
    let known = vec![known(&table())];
    Memory::try_new(device, known, &MemoryOptions::default()).expect("a memory")
}

/// The scene of `tests/batchplan.rs`: the table, a box, a can and the thin tray, and the arm at
/// `q`, the last thing.
fn reaching_over_a_tray(robot: &Robot, q: &[f32]) -> Vec<Thing> {
    let mut things = vec![
        table(),
        block(
            Vec3::new(0.5, -0.2, 0.03),
            Vec3::new(0.04, 0.06, 0.03),
            [200, 30, 30],
        ),
        can(Vec3::new(0.55, 0.2, 0.05), 0.035, 0.05, [30, 30, 200]),
    ];
    things.extend(thin_tray());
    things.push(arm(robot, q));
    things
}

/// The README's "With batchplan", steps 1 to 4, written out for `frames` after `previous`: each
/// frame filtered at its noise, the robot masked at `q` and `padding`, and a look with the corners
/// as far as the robot reaches. The scene, and each frame's filtered depth and mask.
fn by_steps(
    device: &Device,
    frames: &[Frame; 2],
    q: &[f32],
    padding: f32,
    previous: &Scene,
) -> (Scene, Vec<Vec<f32>>, Vec<RobotMask>) {
    let robot = device.robot();
    let mut depths = vec![];
    let mut masks = vec![];
    for frame in frames {
        let image = frame.view().image;
        let Noise {
            constant,
            linear,
            quadratic,
        } = frame.noise;
        let filter = DepthFilter {
            noise: [constant, linear, quadratic],
            ..Default::default()
        };
        let depth = image.filtered(&filter).expect("a sound image");
        let image = DepthImage {
            depth: &depth,
            ..image
        };
        let mask = device.install(|| image.robot_mask(robot, q, padding, &[]));
        masks.push(mask.expect("a sound mask"));
        depths.push(depth);
    }
    let views: Vec<miru::View> = (frames.iter().zip(&depths).zip(&masks))
        .map(|((frame, depth), mask)| miru::View {
            image: DepthImage {
                depth,
                ..frame.view().image
            },
            robot: &mask.on_robot,
            ..frame.view()
        })
        .collect();
    let known = [known(&table())];
    let look = Look {
        views: &views,
        known: &known,
    };
    let corner = Vec3::splat(robot.reach());
    let options = LookOptions {
        lower_corner: -corner,
        upper_corner: corner,
    };
    let scene = miru::look(device, &look, previous, &options).expect("a sound look");
    (scene, depths, masks)
}

/// On the scene of `tests/batchplan.rs`, through each camera and on each device, `observe` gives
/// the scene the README's steps give, and keeps the depth they filter and the mask they make; the
/// scene is the same on every device.
#[test]
fn a_look_is_the_readmes_steps_in_one_call() {
    let robot = panda();
    let things = reaching_over_a_tray(&robot, &OVER_THE_CAN);
    let rendered = render(&things);
    let devices = devices(&robot);
    let locked = HashMap::new();
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let (frames, bands) = read(&rendered, c, 10 * c as u64);
        let padding = padding(&rendered, &frames, things.len() - 1);
        let stance = Stance {
            q: &OVER_THE_CAN,
            locked: &locked,
            padding,
        };
        let cameras = cameras(&frames, bands, 0.0);
        let mut wrong = vec![];
        let mut first: Option<Scene> = None;
        for (name, device) in &devices {
            let seen = observe(device, &cameras, &stance, &memory(device)).expect("a look");
            let (scene, depths, masks) =
                by_steps(device, &frames, &OVER_THE_CAN, padding, &Scene::default());
            let kept = &seen.kept;
            if kept.scene() != &scene {
                wrong.push(format!("{name}: another scene than the steps'"));
            }
            if seen.memory.objects() != scene.objects.as_slice()
                || seen.memory.next_id() != scene.next_id
            {
                wrong.push(format!("{name}: the memory's objects are not the scene's"));
            }
            for (view, (depth, mask)) in depths.iter().zip(&masks).enumerate() {
                let image = kept.image(view).expect("a kept view");
                let robot = kept.robot(view).expect("a kept mask");
                if image.depth != depth.as_slice() || robot.on_robot != mask.on_robot {
                    wrong.push(format!(
                        "{name}: view {view} is not kept as the steps read it"
                    ));
                }
            }
            let same = first.get_or_insert_with(|| kept.scene().clone());
            if kept.scene() != same {
                wrong.push(format!("{name}: another scene than on the first device"));
            }
        }
        assert!(
            first.is_some_and(|scene| scene.objects.len() >= 3),
            "{camera}: the box, the can and the tray are seen"
        );
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_look_is_the_readmes_steps_in_one_call", &found);
}

/// A camera stated with no noise is refused, naming it: its flat surfaces would be filtered away
/// and its shapes given no margin.
#[test]
fn a_camera_stated_without_noise_is_refused() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let things = reaching_over_a_tray(&robot, &OVER_THE_CAN);
    let frames = render(&things);
    let mut cameras = cameras(&frames, [0, 0], 0.0);
    cameras[1].noise = Noise {
        constant: 0.0,
        linear: 0.0,
        quadratic: 0.0,
    };
    let locked = HashMap::new();
    let stance = Stance {
        q: &OVER_THE_CAN,
        locked: &locked,
        padding: 0.0,
    };
    let refused = observe(&device, &cameras, &stance, &memory(&device));
    let Err(batchplan::Error::Input(why)) = refused else {
        panic!("a camera without noise is taken: {refused:?}");
    };
    assert!(why.contains("camera 1"), "{why}");
}

/// A look with no cameras sees nothing: the memory comes back as it was, and nothing is kept of a
/// view.
#[test]
fn a_look_with_no_cameras_sees_nothing() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let things = reaching_over_a_tray(&robot, &OVER_THE_CAN);
    let rendered = render(&things);
    let locked = HashMap::new();
    let stance = Stance {
        q: &OVER_THE_CAN,
        locked: &locked,
        padding: padding(&rendered, &rendered, things.len() - 1),
    };
    let cameras = cameras(&rendered, [0, 0], 0.0);
    let first = observe(&device, &cameras, &stance, &memory(&device)).expect("a look");
    assert!(first.memory.objects().len() >= 3, "the first look sees");
    let none = observe(&device, &[], &stance, &first.memory).expect("a look of nothing");
    assert_eq!(none.memory.objects(), first.memory.objects());
    assert_eq!(none.memory.next_id(), first.memory.next_id());
    assert!(none.kept.scene().labels.is_empty(), "no view's labels");
    assert!(none.kept.image(0).is_none() && none.kept.robot(0).is_none());
}

/// The far and the near camera stated 5 mm over where they read from: with `off_by` 5 mm the table
/// is the table's, through each camera, and no object is made of it; with 0, on exact depth, whose
/// three deviations at the table are under 5 mm, the table's readings are an object of their own.
#[test]
fn a_camera_off_its_pose_is_held_to_what_it_is_off_by() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let things = vec![table(), common::a_box(), arm(&robot, robot.default_q())];
    let rendered = render(&things);
    let locked = HashMap::new();
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let (mut frames, bands) = read(&rendered, c, 100 + 10 * c as u64);
        let stance = Stance {
            q: robot.default_q(),
            locked: &locked,
            padding: padding(&rendered, &frames, things.len() - 1),
        };
        for frame in &mut frames {
            frame.camera.position.z += 0.005;
        }
        // Whether an object of the look holds more of the table's pixels than of anything else's.
        let table_is_an_object = |off_by: f32| {
            let cameras = cameras(&frames, bands, off_by);
            let seen = observe(&device, &cameras, &stance, &memory(&device)).expect("a look");
            let scene = seen.kept.scene();
            (0..scene.objects.len()).any(|object| {
                let [mut table, mut other] = [0, 0];
                for (labels, truth) in scene.labels.iter().zip(&rendered) {
                    for (label, shows) in labels.iter().zip(&truth.shows) {
                        if *label == Label::Object(object) {
                            match shows {
                                Some(0) => table += 1,
                                _ => other += 1,
                            }
                        }
                    }
                }
                table > other
            })
        };
        let mut wrong = vec![];
        if table_is_an_object(0.005) {
            wrong.push("off by 5 mm and stated so, the table is an object".to_string());
        }
        if c == 0 && !table_is_an_object(0.0) {
            wrong.push("off by 5 mm and stated exact, the table is the table's".to_string());
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_camera_off_its_pose_is_held_to_what_it_is_off_by", &found);
}
