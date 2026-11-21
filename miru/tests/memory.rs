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

use batchplan::{DepthFilter, DepthImage, Device, Obstacle, Robot, RobotMask};
use common::{
    CAMERAS, EYE, Frame, MIDDLE, NEAR, Stereo, Thing, arm, block, bowl, can, devices, known,
    looking, panda, table, thin_tray,
};
use glam::Vec3;
use miru::{
    Camera, Label, Look, LookOptions, Memory, MemoryOptions, Noise, Observed, Of, Planned, Scene,
    Shape, Stance, Step, observe,
};

/// The Panda's stance with its fingertips 2.3 cm over the top of the can of
/// [`reaching_over_a_tray`], its hand turned as `tests/batchplan.rs` holds it and the arm across
/// the far view: what reads of the can's top lies nearer the fingers than a map's mask pads them.
const OVER_THE_CAN: [f32; 7] = [0.349, 0.389, 0.0, -2.091, 0.0, 2.480, 0.799];

/// The Panda's stance with its open fingers in the middle of [`wide_bowl`], their tips 5.4 cm
/// over its floor and about 5 cm from its wall, its hand turned as `tests/batchplan.rs` holds it.
const IN_BOWL: [f32; 7] = [0.115, 0.361, -0.103, -2.387, 0.094, 2.745, 0.382];

/// The claims a stereo camera breaks, a line each: the test, the camera, the complaint the test
/// makes through it, and what the scene shows. A complaint through a camera that is not listed for
/// it fails the test, as does a listed one that no longer holds.
const GAPS: &str = "
a_world_is_the_known_shapes_then_each_object | D435 | the map holds what no shape explains | readings of the table that land past three deviations of the camera's noise from it: 121 points of the grid deep in solid, 119 of them nearest the table, none farther than 3.2 cm from a thing
a_world_is_the_known_shapes_then_each_object | D435+D405 | the map holds what no shape explains | as through the D435: 35 points of the grid deep in solid, 33 of them nearest the table, and one 40 cm from any thing
a_camera_off_its_pose_is_held_to_what_it_is_off_by | D435 | the map holds what no shape explains | the table is the table's, but the map holds readings of it past three deviations of the noise, as many as with the cameras stated where they stand: 148 points of the grid deep in solid against 149, all within 1.2 cm of the table
a_camera_off_its_pose_is_held_to_what_it_is_off_by | D435+D405 | the map holds what no shape explains | as through the D435: 67 points deep in solid against 68
";

/// The complaint listed for `test` through `camera`, if one is.
fn gap(test: &str, camera: &str) -> Option<&'static str> {
    let lines = GAPS.lines().filter(|line| !line.is_empty());
    let fields = lines.map(|line| line.split(" | ").collect::<Vec<_>>());
    let listed = fields
        .into_iter()
        .find(|fields| fields[..2] == [test, camera]);
    listed.map(|fields| fields[2])
}

/// Asserts what `test` found through each camera: that it holds where no gap is listed, that
/// where one is it fails with the listed complaint alone (on every device it ran on), and that the
/// gap still is one.
fn verdict(test: &str, found: &[(&str, Result<(), String>)]) {
    let wrong: Vec<String> = (found.iter())
        .filter_map(|(camera, found)| match (found, gap(test, camera)) {
            (Ok(()), None) => None,
            (Err(why), Some(listed)) if why.split("; ").all(|one| one.ends_with(listed)) => None,
            (Ok(()), Some(_)) => Some(format!("{camera}: a listed gap holds")),
            (Err(why), _) => Some(format!("{camera}: {why}")),
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

/// A bowl at the table's middle, 28 cm across at its rim and 8 cm tall, its floor 20 cm across
/// and its wall 4 mm through: room for the Panda's open fingers.
fn wide_bowl() -> Thing {
    bowl(MIDDLE, 0.10, 0.14, 0.08, 0.004, [40, 40, 40])
}

/// `things`, the arm at `q` the last of them, as camera set `c` reads them with the draws of
/// `seed`, looked at once after a memory of nothing on `device`, the cameras' poses and the known
/// shapes off by `off_by`.
fn first_look(
    device: &Device,
    things: &[Thing],
    rendered: &[Frame; 2],
    (c, seed): (usize, u64),
    q: &[f32],
    off_by: f32,
) -> Observed {
    let (frames, bands) = read(rendered, c, seed);
    let locked = HashMap::new();
    let stance = Stance {
        q,
        locked: &locked,
        padding: padding(rendered, &frames, things.len() - 1),
    };
    let cameras = cameras(&frames, bands, off_by);
    observe(device, &cameras, &stance, &memory(device)).expect("a look")
}

/// The object of `scene` that holds the most of thing `thing`'s pixels, if any holds one.
fn holder(scene: &Scene, rendered: &[Frame; 2], thing: usize) -> Option<usize> {
    let mut held = vec![0; scene.objects.len()];
    for (labels, truth) in scene.labels.iter().zip(rendered) {
        let its = (labels.iter().zip(&truth.shows)).filter(|(_, shows)| **shows == Some(thing));
        for (label, _) in its {
            if let Label::Object(object) = label {
                held[*object] += 1;
            }
        }
    }
    let most = held.iter().copied().max().filter(|&most| most > 0)?;
    held.iter().position(|&count| count == most)
}

/// The room `shape` takes up, as an obstacle: a box open at the top's box, and a box or a
/// cylinder itself, field for field.
fn cover(shape: &Shape) -> Obstacle {
    match *shape {
        Shape::Cuboid {
            center,
            half_extents,
            rotation,
        }
        | Shape::OpenBox {
            center,
            half_extents,
            rotation,
            ..
        } => Obstacle::Cuboid {
            center,
            half_extents,
            rotation,
        },
        Shape::Cylinder {
            center,
            rotation,
            radius,
            half_height,
        } => Obstacle::Cylinder {
            center,
            rotation,
            radius,
            half_height,
        },
        _ => panic!("a shape these tests do not know: {shape:?}"),
    }
}

/// What is wrong with `planned` as the world `memory` gives a step that goes into nothing: the
/// known table first, then each object's solids where it is solid and the room it takes up where
/// it is not, in the memory's order, and what nothing explains last, if anything.
fn laid_out(memory: &Memory, planned: &Planned) -> Vec<String> {
    let mut expected: Vec<(Of, Obstacle)> = vec![(Of::Known(0), known(&table()))];
    for object in memory.objects() {
        let shapes = match object.solid {
            true => object.shape.solids(),
            false => vec![cover(&object.shape)],
        };
        expected.extend(
            shapes
                .into_iter()
                .map(|shape| (Of::Object(object.id), shape)),
        );
    }
    let laid: Vec<(Of, Obstacle)> = (planned.of.iter().copied())
        .zip(planned.world.obstacles.iter().cloned())
        .collect();
    let shapes = laid.len() - usize::from(planned.of.last() == Some(&Of::Unexplained));
    match laid[..shapes] == expected[..] && planned.world.touch.is_empty() {
        true => vec![],
        false => vec![format!("laid out as {:?}", planned.of)],
    }
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
/// is the table's, through each camera, no object is made of it and the map holds nothing no shape
/// explains; with 0, on exact depth, whose three deviations at the table are under 5 mm, the
/// table's readings are an object of their own.
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
        // Whether an object of the look holds more of the table's pixels than of anything else's,
        // and whether the map holds what no shape explains.
        let look = |off_by: f32| {
            let cameras = cameras(&frames, bands, off_by);
            let seen = observe(&device, &cameras, &stance, &memory(&device)).expect("a look");
            let planned = seen
                .memory
                .world(&device, &Step::default())
                .expect("a world");
            let scene = seen.kept.scene();
            let table_is_an_object = (0..scene.objects.len()).any(|object| {
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
            });
            (table_is_an_object, planned.of.contains(&Of::Unexplained))
        };
        let mut wrong = vec![];
        let (table_is_an_object, unexplained) = look(0.005);
        if table_is_an_object {
            wrong.push("off by 5 mm and stated so, the table is an object".to_string());
        }
        if unexplained {
            wrong.push(
                "off by 5 mm and stated so, the map holds what no shape explains".to_string(),
            );
        }
        if c == 0 && !look(0.0).0 {
            wrong.push("off by 5 mm and stated exact, the table is the table's".to_string());
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_camera_off_its_pose_is_held_to_what_it_is_off_by", &found);
}

/// On the scene of `tests/batchplan.rs`, a world for a step that goes into nothing is the known
/// table, then each object's solids, or the room it takes up where it is not solid; what a shape
/// explains is not in the map, so nothing is left unexplained; the arm at its stance touches
/// nothing; and the world is the same on every device.
#[test]
fn a_world_is_the_known_shapes_then_each_object() {
    let robot = panda();
    let things = reaching_over_a_tray(&robot, &OVER_THE_CAN);
    let rendered = render(&things);
    let devices = devices(&robot);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let mut wrong = vec![];
        let mut first: Option<Planned> = None;
        for (name, device) in &devices {
            let seen = first_look(
                device,
                &things,
                &rendered,
                (c, 10 * c as u64),
                &OVER_THE_CAN,
                0.0,
            );
            let planned = seen
                .memory
                .world(device, &Step::default())
                .expect("a world");
            wrong.extend(
                laid_out(&seen.memory, &planned)
                    .into_iter()
                    .map(|why| format!("{name}: {why}")),
            );
            if planned.of.contains(&Of::Unexplained) {
                wrong.push(format!("{name}: the map holds what no shape explains"));
            }
            let worlds = device.upload(std::slice::from_ref(&planned.world));
            let worlds = worlds.expect("a world the device takes");
            let contacts = device.contacts(&worlds, &[0], &OVER_THE_CAN, 0.0);
            let contacts = contacts.expect("the arm's contacts");
            if !contacts[0].is_empty() {
                wrong.push(format!("{name}: the arm touches {:?}", contacts[0]));
            }
            let same = first.get_or_insert_with(|| planned.clone());
            if planned.world.obstacles != same.world.obstacles || planned.of != same.of {
                wrong.push(format!("{name}: another world than on the first device"));
            }
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_world_is_the_known_shapes_then_each_object", &found);
}

/// A bowl is not solid: for a step that goes into nothing it is the room it takes up, solid in its
/// middle; named in `into` it is its part of the map, free in its middle, where the cameras saw
/// in, and solid at its wall; and it is that part as well where the robot at a configuration of
/// `at`, or the hand at a pose of `reach`, stands in its room, and not where the robot stands
/// clear of it.
#[test]
fn a_thing_that_is_not_solid_is_its_room_unless_a_step_goes_into_it() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let things = vec![table(), wide_bowl(), arm(&robot, away)];
    let rendered = render(&things);
    let (room, wall) = (Vec3::new(0.5, 0.0, 0.05), Vec3::new(0.5, 0.12, 0.04));
    let in_bowl = robot.ee_pose(&IN_BOWL, 0);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let Some(bowl) = holder(seen.kept.scene(), &rendered, 1) else {
            found.push((*camera, Err("no object holds the bowl".to_string())));
            continue;
        };
        let object = &seen.kept.scene().objects[bowl];
        let id = object.id;
        // What the bowl's object is in the world for `step`.
        let of = |step: &Step<'_>| {
            let planned = seen.memory.world(&device, step).expect("a world");
            let at = planned
                .of
                .iter()
                .position(|of| matches!(of, Of::Object(o) | Of::Region(o) if *o == id));
            let at = at.expect("every object is in the world");
            (planned.of[at], planned.world.obstacles[at].clone())
        };
        let mut wrong = vec![];
        if object.solid {
            wrong.push("the bowl is solid".to_string());
        }
        let (of_nothing, shape) = of(&Step::default());
        if of_nothing != Of::Object(id)
            || shape != cover(&object.shape)
            || shape.distance(room).value >= 0.0
        {
            wrong.push(format!("into nothing it is {of_nothing:?}, not its room"));
        }
        let (of_into, region) = of(&Step {
            into: &[id],
            ..Step::default()
        });
        if of_into != Of::Region(id)
            || region.distance(room).value <= 0.0
            || region.distance(wall).value >= 0.0
        {
            wrong.push(format!("named in `into` it is {of_into:?}, {:.3} m from its middle and {:.3} m from its wall", region.distance(room).value, region.distance(wall).value));
        }
        let (of_at, _) = of(&Step {
            at: &[&IN_BOWL],
            ..Step::default()
        });
        if of_at != Of::Region(id) {
            wrong.push(format!("with the arm in it it is {of_at:?}"));
        }
        let (of_reach, _) = of(&Step {
            reach: &[(0, in_bowl)],
            ..Step::default()
        });
        if of_reach != Of::Region(id) {
            wrong.push(format!("with the hand in it it is {of_reach:?}"));
        }
        let (of_away, _) = of(&Step {
            at: &[away],
            ..Step::default()
        });
        if of_away != Of::Object(id) {
            wrong.push(format!("with the arm away it is {of_away:?}"));
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict(
        "a_thing_that_is_not_solid_is_its_room_unless_a_step_goes_into_it",
        &found,
    );
}

/// The hand in the wide bowl as the cameras look: the space it hides in the bowl is solid in the
/// bowl's part of the map, never seen, and the robot at its stance touches none of it, nor
/// anything else: the map's mask is padded past what its grid grows toward collision.
#[test]
fn the_arms_shadow_in_the_map_does_not_reach_the_arm() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let things = vec![table(), wide_bowl(), arm(&robot, &IN_BOWL)];
    let rendered = render(&things);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(
            &device,
            &things,
            &rendered,
            (c, 10 * c as u64),
            &IN_BOWL,
            0.0,
        );
        let step = Step {
            at: &[&IN_BOWL],
            ..Step::default()
        };
        let planned = seen.memory.world(&device, &step).expect("a world");
        let mut wrong = vec![];
        if !planned.of.iter().any(|of| matches!(of, Of::Region(_))) {
            wrong.push("no region holds the arm".to_string());
        }
        let worlds = device.upload(std::slice::from_ref(&planned.world));
        let worlds = worlds.expect("a world the device takes");
        let contacts = device
            .contacts(&worlds, &[0], &IN_BOWL, 0.0)
            .expect("the arm's contacts");
        if !contacts[0].is_empty() {
            let named: Vec<String> = (contacts[0].iter())
                .map(|contact| {
                    format!(
                        "{contact:?} of {:?}",
                        match contact {
                            batchplan::Contact::World { obstacle, .. } =>
                                Some(planned.of[*obstacle]),
                            _ => None,
                        }
                    )
                })
                .collect();
            wrong.push(format!("the arm touches {}", named.join(", ")));
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("the_arms_shadow_in_the_map_does_not_reach_the_arm", &found);
}

/// What the map holds that no shape and no object's room explains is one grid, the last
/// obstacle: solid at a crumb 3 mm across and a pebble 1 cm across wherever the look makes no
/// object of one and it stands clear of the table by more than the table's margin, and free in the
/// bowl's room and at its wall, in the box and over the table.
#[test]
fn what_no_room_explains_is_the_unexplained_grid_and_nothing_else() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let (crumb, pebble) = (Vec3::new(0.35, 0.3, 0.0015), Vec3::new(0.75, -0.25, 0.005));
    let the_box = Vec3::new(0.35, -0.25, 0.03);
    let things = vec![
        table(),
        common::standing_bowl(MIDDLE),
        block(the_box, Vec3::new(0.04, 0.06, 0.03), [200, 30, 30]),
        block(crumb, Vec3::splat(0.0015), [200, 30, 200]),
        block(pebble, Vec3::splat(0.005), [200, 30, 200]),
        arm(&robot, away),
    ];
    let rendered = render(&things);
    let free = [
        ("the bowl's room", Vec3::new(0.5, 0.0, 0.03)),
        ("the bowl's wall", Vec3::new(0.5, 0.043, 0.025)),
        ("the box", the_box),
        ("the table", Vec3::new(0.6, 0.25, 0.005)),
    ];
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let (frames, _) = read(&rendered, c, 10 * c as u64);
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let planned = seen
            .memory
            .world(&device, &Step::default())
            .expect("a world");
        let grid = (planned.of.last() == Some(&Of::Unexplained))
            .then(|| planned.world.obstacles.last())
            .flatten();
        let solid_at = |p: Vec3| grid.is_some_and(|grid| grid.distance(p).value < 0.0);
        let mut wrong = vec![];
        let mut held = 0;
        for (thing, (name, at, half)) in [
            (3, ("the crumb", crumb, 0.0015)),
            (4, ("the pebble", pebble, 0.005)),
        ] {
            // Clear of the table by more than three deviations of every camera that sees it.
            let reach = (frames.iter())
                .map(|frame| 3.0 * deviation(frame.noise, at.distance(frame.camera.position)))
                .fold(0.0, f32::max);
            let unmade = holder(seen.kept.scene(), &rendered, thing).is_none();
            if unmade && 2.0 * half > reach {
                held += 1;
                if !solid_at(at) {
                    wrong.push(format!("{name} is no object and not held"));
                }
            }
        }
        assert!(
            c > 0 || held > 0,
            "on exact depth the map holds something no room explains"
        );
        for (name, at) in free {
            if solid_at(at) {
                wrong.push(format!("the grid holds {name}"));
            }
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict(
        "what_no_room_explains_is_the_unexplained_grid_and_nothing_else",
        &found,
    );
}
