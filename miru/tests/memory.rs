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

use batchplan::{
    DepthFilter, DepthImage, Device, Gripper, GripperCalibration, Held, IkOptions, IkProblem,
    Obstacle, Pose, Robot, RobotMask, Screw, ShapeGraspOptions, SphereOptions, SurfaceGraspOptions,
    Weighing, World,
};
use common::{
    CAMERAS, EYE, Frame, MIDDLE, NEAR, Stereo, Thing, arm, block, bowl, can, devices, known,
    looking, panda, panda_closed, panda_gripper, panda_with_hands, table, thin_tray,
};
use glam::{Mat3, Quat, Vec3};
use miru::{
    Act, Camera, ClosureEvidence, GripSource, Kept, Label, Look, LookOptions, Memory,
    MemoryOptions, Noise, Observed, Of, Planned, Scene, Shape, Stance, Step, Support, TurnGrips,
    TurnGroup, Under, observe,
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
a_world_is_the_known_shapes_then_each_object | D435 | the map holds what no shape explains | readings of the table that land past three deviations of the camera's noise from it, nearest the table and within a few centimetres of a thing
a_world_is_the_known_shapes_then_each_object | D435+D405 | the map holds what no shape explains | as through the D435
a_camera_off_its_pose_is_held_to_what_it_is_off_by | D435 | the map holds what no shape explains | the table is the table's, but the map holds readings of it past three deviations of the noise, as it does with the cameras stated where they stand
a_camera_off_its_pose_is_held_to_what_it_is_off_by | D435+D405 | the map holds what no shape explains | as through the D435
a_plate_let_go_on_a_box_is_seen_there | exact | let go on the box, it reads gone | the look makes the plate one object with the box, which keeps the box's id and grows by the plate's 5 mm: `look`'s, as in `tests/truth.rs`'s \"a plate on a box\"
a_plate_let_go_on_a_box_is_seen_there | D435 | let go on the box, it reads gone | as on exact depth
a_plate_let_go_on_a_box_is_seen_there | D435+D405 | the plate on the table is no object | to the D405 the plate on the table is the table's, so there is nothing to pick
what_stood_on_a_picked_thing_is_in_a_memory | exact | the nearest obstacle is 0.090 m from its middle | the hand and the masks hold the box alone, the can left the objects with it, and the map never held it: the nearest obstacle is the table
what_stood_on_a_picked_thing_is_in_a_memory | D435 | the nearest obstacle is 0.034 m from its middle | as on exact depth, the nearest obstacle the grid of what nothing explains
what_stood_on_a_picked_thing_is_in_a_memory | D435+D405 | the nearest obstacle is 0.090 m from its middle | as on exact depth
a_thing_let_go_against_another_keeps_both_ids | exact | B, which did not move, reads gone | the look makes the two boxes one object, which keeps A's id: `look`'s, as in `tests/truth.rs`'s \"two boxes touching\"
a_thing_let_go_against_another_keeps_both_ids | D435 | B, which did not move, reads gone | as on exact depth
a_thing_let_go_against_another_keeps_both_ids | D435+D405 | A, let go there, reads gone | the two boxes are one object, which keeps B's id
the_band_takes_what_a_camera_reads_beside_the_arm | D435 | the map holds 2 of 15 | the band gives up an outline read nearer than the arm by more than the margin, and what lies within it of a reading it keeps; without the band the map holds 10
the_band_takes_what_a_camera_reads_beside_the_arm | D435+D405 | the map holds 8 of 20 | as through the D435; without the band the map holds 13
what_becomes_a_shape_leaves_the_map | D435 | thing 2, a solid object now, is held by the map | the pebble's own readings from the near look lie past three deviations of that camera's noise from its shape: a first look by the near camera alone holds them too, as the table's are held; freed 3 cm farther, it is held still
what_becomes_a_shape_leaves_the_map | D435+D405 | thing 2, a solid object now, is held by the map | as through the D435, the near look a D405's
a_knob_is_turned_by_its_bar_first | D435 | no bar over the disk resolves | the outline of the dial's top in the camera's readings fits no circle within the readings' resolution, so `BarOnDisk::measure` finds no disk under the bar
a_knob_is_turned_by_its_bar_first | D435+D405 | no bar over the disk resolves | as through the D435
a_box_seen_from_one_side_is_turned_from_above_its_middle | D435+D405 | no grip from above its middle in 1 groups | the box is fitted 14 x 6 cm and not solid, and a reading not the box's, 15 mm past its unseen side and 2 cm over the table, lies in both closing sweeps from above its fitted middle
a_post_in_the_closing_gap_rules_the_grip_out | D435+D405 | no grip from above its middle | as in `a_box_seen_from_one_side_is_turned_from_above_its_middle`, without the post
a_shelf_in_the_way_in_rules_the_grip_out | D435+D405 | no grip from above its middle | as in `a_box_seen_from_one_side_is_turned_from_above_its_middle`, without the shelf
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

/// A bowl standing at `foot`, 28 cm across at its rim and 8 cm tall, its floor 20 cm across and
/// its wall 4 mm through: room for the Panda's open fingers.
fn wide_bowl(foot: Vec3) -> Thing {
    bowl(foot, 0.10, 0.14, 0.08, 0.004, [40, 40, 40])
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

/// `observe`'s steps 1 to 4, written out for `frames` after `previous`: each
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
/// the scene its steps give written out, and keeps the depth they filter and the mask they make;
/// the scene is the same on every device.
#[test]
fn observe_gives_its_steps_written_out() {
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
    verdict("observe_gives_its_steps_written_out", &found);
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
    let remembered = first.memory.objects()[0].id;
    for id in [remembered, u32::MAX] {
        let lifts = none
            .kept
            .extractions(id, 0.01)
            .expect("an optional measurement");
        assert!(lifts.is_empty(), "no current reading of object {id}");
        for invalid in [-0.01, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                none.kept.extractions(id, invalid),
                Err(batchplan::Error::Input(_))
            ));
        }
    }
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
    let things = vec![table(), wide_bowl(MIDDLE), arm(&robot, away)];
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
    let things = vec![table(), wide_bowl(MIDDLE), arm(&robot, &IN_BOWL)];
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

/// Where the box of the acts' scene stands: 8 by 12 cm and 6 cm tall, on the table.
const BOX: Vec3 = Vec3::new(0.45, -0.15, 0.03);

/// The box's half sizes.
const BOX_HALF: Vec3 = Vec3::new(0.04, 0.06, 0.03);

/// Where the bowl of the acts' scene stands: the tests' 4 mm bowl, on the table.
const BOWL: Vec3 = Vec3::new(0.55, 0.15, 0.0);

/// The acts' scene: the table, the box at `the_box` where it is on the table, the bowl at
/// `the_bowl` where it is, and the arm at `q`, the last thing; with `holding`, a thing the hand
/// holds after the arm.
fn acts_scene(
    robot: &Robot,
    q: &[f32],
    the_box: Option<Vec3>,
    the_bowl: Option<Vec3>,
    holding: Option<Thing>,
) -> Vec<Thing> {
    let mut things = vec![table()];
    things.extend(the_box.map(|at| block(at, BOX_HALF, [200, 30, 30])));
    things.extend(the_bowl.map(common::standing_bowl));
    things.push(arm(robot, q));
    things.extend(holding);
    things
}

/// How hand 0 holds the object `id` of `memory`: its shape at the origin of its own frame, turned
/// as the hand is, the hand's frame 2 cm above its middle along its z axis.
fn grasped(memory: &Memory, id: u32) -> Held {
    let object = memory.objects().iter().find(|object| object.id == id);
    let object = object.expect("the memory holds the object");
    let at_origin = match cover(&object.shape) {
        Obstacle::Cylinder {
            radius,
            half_height,
            ..
        } => Obstacle::cylinder(Vec3::ZERO, radius, half_height),
        Obstacle::Cuboid { half_extents, .. } => Obstacle::cuboid(Vec3::ZERO, half_extents),
        shape => panic!("a shape these tests do not hold: {shape:?}"),
    };
    Held {
        hand: 0,
        object: at_origin,
        grasp: Pose::new(Vec3::new(0.0, 0.0, -0.02), Quat::IDENTITY),
        locked: HashMap::new(),
        spheres: SphereOptions::default(),
    }
}

/// The id of the object of `memory`, after a look whose scene is `scene`, that holds the most of
/// thing `thing`'s pixels.
fn id_of(scene: &Scene, rendered: &[Frame; 2], thing: usize) -> Option<u32> {
    holder(scene, rendered, thing).map(|object| scene.objects[object].id)
}

/// Whether an object of `scene` holds more of thing `thing`'s pixels than of anything else's.
fn an_object_of(scene: &Scene, rendered: &[Frame; 2], thing: usize) -> bool {
    (0..scene.objects.len()).any(|object| {
        let [mut its, mut other] = [0, 0];
        for (labels, truth) in scene.labels.iter().zip(rendered) {
            for (label, shows) in labels.iter().zip(&truth.shows) {
                if *label == Label::Object(object) {
                    match *shows == Some(thing) {
                        true => its += 1,
                        false => other += 1,
                    }
                }
            }
        }
        its > other
    })
}

/// A second look, with the arm at `q`, after `memory`, at `things` read as camera set `c` with
/// the draws of `seed`.
fn second_look(
    device: &Device,
    things: &[Thing],
    arm: usize,
    (c, seed): (usize, u64),
    q: &[f32],
    memory: &Memory,
) -> ([Frame; 2], Observed) {
    let rendered = render(things);
    let (frames, bands) = read(&rendered, c, seed);
    let locked = HashMap::new();
    let stance = Stance {
        q,
        locked: &locked,
        padding: padding(&rendered, &frames, arm),
    };
    let cameras = cameras(&frames, bands, 0.0);
    let seen = observe(device, &cameras, &stance, memory).expect("a look");
    (rendered, seen)
}

/// The box picked: the hand holds it, it is among no objects, and a second look with it in the
/// hand, in both views, makes no object of it.
#[test]
fn a_picked_thing_in_the_hand_is_no_object() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let things = acts_scene(&robot, away, Some(BOX), None, None);
    let rendered = render(&things);
    // The grasp is oblique to both sides, whichever one the fitted box names as its x axis.
    let hand_in_box = Pose::new(
        Vec3::new(0.0, 0.0, -0.02),
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
    );
    let hand_at_pick = Pose::new(BOX, Quat::IDENTITY).mul_pose(hand_in_box);
    let carried = robot
        .ee_pose(&OVER_THE_CAN, 0)
        .mul_pose(hand_in_box.inverse());
    let holding = common::turned(
        block(carried.position, BOX_HALF, [200, 30, 30]),
        carried.position,
        carried.rotation,
    );
    let lifted = acts_scene(&robot, &OVER_THE_CAN, None, None, Some(holding));
    let arm = lifted.len() - 2;
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let scene = seen.kept.scene();
        let Some(index) = holder(scene, &rendered, 1) else {
            found.push((*camera, Err("no object holds the box".to_string())));
            continue;
        };
        let object = &scene.objects[index];
        let id = object.id;
        let mut held = grasped(&seen.memory, id);
        held.grasp = cover(&object.shape).pose().inverse().mul_pose(hand_at_pick);
        let picked = (seen.memory)
            .after(&[Act::Picked {
                id,
                held: held.clone(),
            }])
            .expect("a pick");
        let mut wrong = vec![];
        if picked.held(0) != Some((id, &held)) || picked.objects().iter().any(|o| o.id == id) {
            wrong.push("the hand does not hold the box alone".to_string());
        }
        let (rendered, after) = second_look(
            &device,
            &lifted,
            arm,
            (c, 10 * c as u64 + 5),
            &OVER_THE_CAN,
            &picked,
        );
        let shown: usize = (rendered.iter())
            .map(|frame| frame.shows.iter().filter(|s| **s == Some(arm + 1)).count())
            .sum();
        assert!(
            shown > 500,
            "{camera}: the box in the hand shows in {shown} pixels"
        );
        if an_object_of(after.kept.scene(), &rendered, arm + 1) {
            wrong.push("the box in the hand is an object".to_string());
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_picked_thing_in_the_hand_is_no_object", &found);
}

/// The bowl, which is not solid, picked: the map held it, and the room it took up is free in it
/// at once, and after a second look that sees its place empty: what the map holds unexplained is
/// not solid where its wall stood.
#[test]
fn a_picked_thing_that_is_not_solid_leaves_the_map() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let things = acts_scene(&robot, away, None, Some(BOWL), None);
    let rendered = render(&things);
    let wall = BOWL + Vec3::new(0.0, 0.043, 0.025);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let Some(bowl) = holder(seen.kept.scene(), &rendered, 1) else {
            found.push((*camera, Err("no object holds the bowl".to_string())));
            continue;
        };
        let object = &seen.kept.scene().objects[bowl];
        assert!(!object.solid, "{camera}: the bowl is not solid");
        let held = grasped(&seen.memory, object.id);
        let picked = (seen.memory)
            .after(&[Act::Picked {
                id: object.id,
                held,
            }])
            .expect("a pick");
        // Whether the map holds what nothing explains where the bowl's wall stood.
        let held_at_wall = |memory: &Memory| {
            let planned = memory.world(&device, &Step::default()).expect("a world");
            let grid = (planned.of.last() == Some(&Of::Unexplained))
                .then(|| planned.world.obstacles.last())
                .flatten();
            grid.is_some_and(|grid| grid.distance(wall).value < 0.0)
        };
        let mut wrong = vec![];
        if held_at_wall(&picked) {
            wrong.push("picked, the map holds its wall".to_string());
        }
        let things = acts_scene(&robot, away, None, None, None);
        let (_, after) = second_look(
            &device,
            &things,
            things.len() - 1,
            (c, 10 * c as u64 + 5),
            away,
            &picked,
        );
        if held_at_wall(&after.memory) {
            wrong.push("picked and its place seen empty, the map holds its wall".to_string());
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_picked_thing_that_is_not_solid_leaves_the_map", &found);
}

/// The box picked and let go 15 cm along the table: a second look answers one of three. Seen
/// there, it keeps its id and a label names it; let go where neither camera sees, it keeps its id
/// and no label names it; let go where the cameras see nothing, it is gone.
#[test]
fn a_thing_let_go_is_seen_there_or_unseen_or_gone() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let things = acts_scene(&robot, away, Some(BOX), None, None);
    let rendered = render(&things);
    let (along, unseen) = (Vec3::new(0.0, 0.15, 0.0), Vec3::new(0.6, 0.45, 0.0));
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let Some(id) = id_of(seen.kept.scene(), &rendered, 1) else {
            found.push((*camera, Err("no object holds the box".to_string())));
            continue;
        };
        let shape = seen
            .memory
            .objects()
            .iter()
            .find(|o| o.id == id)
            .expect("the box")
            .shape;
        let frame = cover(&shape).pose();
        let held = grasped(&seen.memory, id);
        let picked = (seen.memory)
            .after(&[Act::Picked { id, held }])
            .expect("a pick");
        // The memory after letting the box go moved by `by`, and a second look with the box there,
        // or nowhere.
        let let_go = |by: Vec3, there: bool| {
            let at = Pose::new(frame.position + by, frame.rotation);
            let memory = picked
                .after(&[Act::LetGo { id, at, on: None }])
                .expect("a let-go");
            let things = acts_scene(&robot, away, there.then_some(BOX + by), None, None);
            let (rendered, after) = second_look(
                &device,
                &things,
                things.len() - 1,
                (c, 10 * c as u64 + 5),
                away,
                &memory,
            );
            let shows_the_box = |shows: &&Option<usize>| there && **shows == Some(1);
            let shown: usize = (rendered.iter())
                .map(|frame| frame.shows.iter().filter(shows_the_box).count())
                .sum();
            let kept = after.memory.objects().iter().position(|o| o.id == id);
            let named = kept.is_some_and(|object| {
                (after.kept.scene().labels.iter().flatten())
                    .any(|label| *label == Label::Object(object))
            });
            (shown, kept.is_some(), named)
        };
        let mut wrong = vec![];
        let (shown, kept, named) = let_go(along, true);
        assert!(
            shown > 500,
            "{camera}: the box let go shows in {shown} pixels"
        );
        if !(kept && named) {
            wrong.push(format!("seen there, it is kept: {kept}, named: {named}"));
        }
        let (shown, kept, named) = let_go(unseen, true);
        assert_eq!(shown, 0, "{camera}: the box let go out of view shows");
        if !(kept && !named) {
            wrong.push(format!(
                "let go out of view, it is kept: {kept}, named: {named}"
            ));
        }
        let (_, kept, _) = let_go(along, false);
        if kept {
            wrong.push("its place seen empty, it is kept".to_string());
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_thing_let_go_is_seen_there_or_unseen_or_gone", &found);
}

/// The box picked and lost: no hand holds it and it is among no objects; a second look that sees
/// it on the table elsewhere finds it under a new id.
#[test]
fn a_lost_thing_is_found_under_a_new_id() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let things = acts_scene(&robot, away, Some(BOX), None, None);
    let rendered = render(&things);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let Some(id) = id_of(seen.kept.scene(), &rendered, 1) else {
            found.push((*camera, Err("no object holds the box".to_string())));
            continue;
        };
        let held = grasped(&seen.memory, id);
        let lost = (seen.memory)
            .after(&[Act::Picked { id, held }, Act::Lost { id }])
            .expect("a pick and a loss");
        let mut wrong = vec![];
        if lost.held(0).is_some() || lost.objects().iter().any(|o| o.id == id) {
            wrong.push("lost, it is held or among the objects".to_string());
        }
        let things = acts_scene(
            &robot,
            away,
            Some(BOX + Vec3::new(0.0, 0.15, 0.0)),
            None,
            None,
        );
        let (rendered, after) = second_look(
            &device,
            &things,
            things.len() - 1,
            (c, 10 * c as u64 + 5),
            away,
            &lost,
        );
        match id_of(after.kept.scene(), &rendered, 1) {
            Some(found) if found >= lost.next_id() => {}
            other => wrong.push(format!(
                "seen again, it is {other:?}, from {}",
                lost.next_id()
            )),
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_lost_thing_is_found_under_a_new_id", &found);
}

/// The box pushed 10 cm along the table, clear of where it stood, keeps its id when a second look
/// finds it there. The bowl, which is not solid, pushed 12 cm, is never seen where it stands: its
/// part of the map there is solid in its middle until a look sees in.
#[test]
fn a_pushed_thing_keeps_its_id() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let things = acts_scene(&robot, away, Some(BOX), Some(BOWL), None);
    let rendered = render(&things);
    let (box_by, bowl_by) = (Vec3::new(-0.1, 0.0, 0.0), Vec3::new(0.0, 0.12, 0.0));
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let (Some(the_box), Some(bowl)) = (
            id_of(seen.kept.scene(), &rendered, 1),
            id_of(seen.kept.scene(), &rendered, 2),
        ) else {
            found.push((
                *camera,
                Err("no object holds the box or the bowl".to_string()),
            ));
            continue;
        };
        let mut wrong = vec![];
        let pushed = (seen.memory)
            .after(&[Act::Pushed {
                id: bowl,
                by: bowl_by,
            }])
            .expect("a push");
        let step = Step {
            into: &[bowl],
            ..Step::default()
        };
        let planned = pushed.world(&device, &step).expect("a world");
        let at = planned.of.iter().position(|of| *of == Of::Region(bowl));
        let middle = BOWL + bowl_by + Vec3::new(0.0, 0.0, 0.03);
        if !at.is_some_and(|at| planned.world.obstacles[at].distance(middle).value < 0.0) {
            wrong.push("pushed, the bowl's part of the map is free where it stands".to_string());
        }
        let pushed = (seen.memory)
            .after(&[Act::Pushed {
                id: the_box,
                by: box_by,
            }])
            .expect("a push");
        let things = acts_scene(&robot, away, Some(BOX + box_by), Some(BOWL), None);
        let (rendered, after) = second_look(
            &device,
            &things,
            things.len() - 1,
            (c, 10 * c as u64 + 5),
            away,
            &pushed,
        );
        let now = id_of(after.kept.scene(), &rendered, 1);
        if now != Some(the_box) {
            wrong.push(format!("pushed, the box is {now:?}, not {the_box}"));
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_pushed_thing_keeps_its_id", &found);
}

/// Acts on what the memory does not hold are refused, naming the act and the object or hand: an
/// id no object has, a let-go and a loss of what no hand holds, a pick into a hand that holds
/// something and into a hand the robot has not, and a let-go at a rotation that is not unit.
#[test]
fn acts_on_what_the_memory_does_not_hold_are_refused() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let things = acts_scene(&robot, away, Some(BOX), Some(BOWL), None);
    let rendered = render(&things);
    let seen = first_look(&device, &things, &rendered, (0, 0), away, 0.0);
    let ids = [1, 2].map(|thing| id_of(seen.kept.scene(), &rendered, thing).expect("an object"));
    let held = grasped(&seen.memory, ids[0]);
    let unknown = seen.memory.next_id();
    let refused = [
        (
            vec![Act::Pushed {
                id: unknown,
                by: Vec3::X,
            }],
            "act 0",
            unknown.to_string(),
        ),
        (
            vec![Act::LetGo {
                id: ids[0],
                at: Pose::IDENTITY,
                on: None,
            }],
            "act 0",
            ids[0].to_string(),
        ),
        (vec![Act::Lost { id: ids[1] }], "act 0", ids[1].to_string()),
        (
            vec![
                Act::Picked {
                    id: ids[0],
                    held: held.clone(),
                },
                Act::Picked {
                    id: ids[1],
                    held: grasped(&seen.memory, ids[1]),
                },
            ],
            "act 1",
            ids[0].to_string(),
        ),
        (
            vec![
                Act::Picked {
                    id: ids[0],
                    held: held.clone(),
                },
                Act::LetGo {
                    id: ids[0],
                    at: Pose::new(BOX, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
                    on: None,
                },
            ],
            "act 1",
            "unit rotation".to_string(),
        ),
        (
            vec![Act::Picked {
                id: ids[0],
                held: Held { hand: 1, ..held },
            }],
            "act 0",
            "hand 1".to_string(),
        ),
    ];
    for (acts, act, named) in refused {
        let after = seen.memory.after(&acts);
        let Err(batchplan::Error::Input(why)) = after else {
            panic!("{acts:?} are taken");
        };
        assert!(why.contains(act) && why.contains(&named), "{why}");
    }
}

/// Memories after acts that differ only in how the hand holds what it picked share the map: the
/// bowl's part of it, and the grid of what nothing explains (a crumb 3 mm across that no object
/// is made of), are each made once, for both.
#[test]
fn memories_after_different_holds_share_the_map() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let crumb = block(
        Vec3::new(0.35, 0.3, 0.0015),
        Vec3::splat(0.0015),
        [200, 30, 200],
    );
    let mut things = vec![table()];
    things.push(block(BOX, BOX_HALF, [200, 30, 30]));
    things.push(common::standing_bowl(BOWL));
    things.push(crumb);
    things.push(arm(&robot, away));
    let rendered = render(&things);
    let seen = first_look(&device, &things, &rendered, (0, 0), away, 0.0);
    let ids = [1, 2].map(|thing| id_of(seen.kept.scene(), &rendered, thing).expect("an object"));
    let held = grasped(&seen.memory, ids[0]);
    let turned = Held {
        grasp: Pose::new(
            held.grasp.position,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        ),
        ..held.clone()
    };
    let grid = |obstacle: &Obstacle| match obstacle {
        Obstacle::Sdf { grid, .. } => grid.clone(),
        other => panic!("a part of the map that is no grid: {other:?}"),
    };
    let grids: Vec<[std::sync::Arc<batchplan::SdfGrid>; 2]> = [held, turned]
        .into_iter()
        .map(|held| {
            let memory = (seen.memory)
                .after(&[Act::Picked { id: ids[0], held }])
                .expect("a pick");
            let step = Step {
                into: &[ids[1]],
                ..Step::default()
            };
            let planned = memory.world(&device, &step).expect("a world");
            let at = planned.of.iter().position(|of| *of == Of::Region(ids[1]));
            let region = &planned.world.obstacles[at.expect("the bowl is a region")];
            assert_eq!(
                planned.of.last(),
                Some(&Of::Unexplained),
                "the map holds the crumb"
            );
            let unexplained = planned.world.obstacles.last().expect("the crumb's grid");
            [grid(region), grid(unexplained)]
        })
        .collect();
    assert!(
        std::sync::Arc::ptr_eq(&grids[0][0], &grids[1][0]),
        "the bowl's part is made twice"
    );
    assert!(
        std::sync::Arc::ptr_eq(&grids[0][1], &grids[1][1]),
        "the grid of what nothing explains is made twice"
    );
}

/// A plate 5 mm thick, picked from the table and let go on the box's top, is seen there at the
/// next look: it keeps its id and a label names it. A look makes such a plate one object with the
/// box (`tests/truth.rs`, "a plate on a box"), so here it reads gone while it stands there.
#[test]
fn a_plate_let_go_on_a_box_is_seen_there() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let plate_half = Vec3::new(0.03, 0.03, 0.0025);
    let (on_table, on_box) = (
        Vec3::new(0.35, 0.25, 0.0025),
        BOX + Vec3::new(0.0, 0.0, BOX_HALF.z + 0.0025),
    );
    let scene = |plate: Vec3| {
        vec![
            table(),
            block(BOX, BOX_HALF, [200, 30, 30]),
            block(plate, plate_half, [30, 200, 30]),
            arm(&robot, away),
        ]
    };
    let things = scene(on_table);
    let rendered = render(&things);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let Some(id) = id_of(seen.kept.scene(), &rendered, 2) else {
            found.push((
                *camera,
                Err("the plate on the table is no object".to_string()),
            ));
            continue;
        };
        let shape = seen
            .memory
            .objects()
            .iter()
            .find(|o| o.id == id)
            .expect("the plate")
            .shape;
        let frame = cover(&shape).pose();
        let at = Pose::new(frame.position + (on_box - on_table), frame.rotation);
        let held = grasped(&seen.memory, id);
        let let_go =
            (seen.memory).after(&[Act::Picked { id, held }, Act::LetGo { id, at, on: None }]);
        let let_go = let_go.expect("a pick and a let-go");
        let things = scene(on_box);
        let (_, after) = second_look(
            &device,
            &things,
            things.len() - 1,
            (c, 10 * c as u64 + 5),
            away,
            &let_go,
        );
        let kept = after.memory.objects().iter().position(|o| o.id == id);
        let labels = after.kept.scene().labels.iter().flatten();
        let named =
            kept.is_some_and(|object| labels.clone().any(|label| *label == Label::Object(object)));
        let wrong = match (kept.is_some(), named) {
            (true, true) => vec![],
            (false, _) => vec!["let go on the box, it reads gone".to_string()],
            (true, false) => vec!["let go on the box, it reads unseen".to_string()],
        };
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_plate_let_go_on_a_box_is_seen_there", &found);
}

/// On the scene of `tests/batchplan.rs`, each object's surface from a kept look is `Kept::surface`
/// written out: each view's filtered depth with the object's labels as flags and the robot's mask,
/// a 5 mm patch, the views joined. It has a point per reading off the robot, view by view and row
/// by row, flagged where the look labels the reading the object's; and on exact depth each flagged
/// point lies within three deviations of the view's noise of its thing. An id the look has not has
/// no surface.
#[test]
fn a_kept_look_gives_an_objects_surface() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let things = reaching_over_a_tray(&robot, &OVER_THE_CAN);
    let rendered = render(&things);
    // Each object's things: the box, the can and the tray's floor and walls.
    let parts = [1..2, 2..3, 3..8];
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(
            &device,
            &things,
            &rendered,
            (c, 10 * c as u64),
            &OVER_THE_CAN,
            0.0,
        );
        let kept = &seen.kept;
        let scene = kept.scene();
        let mut wrong = vec![];
        for part in parts.clone() {
            let Some(object) = (part.clone())
                .filter_map(|thing| holder(scene, &rendered, thing))
                .next()
            else {
                wrong.push(format!("no object holds things {part:?}"));
                continue;
            };
            let id = scene.objects[object].id;
            let surface = kept.surface(&device, id).expect("a sound surface");
            let surface = surface.expect("a surface of an object the look names");
            let mut by_hand: Option<batchplan::Surface> = None;
            let mut k = 0;
            for (view, labels) in scene.labels.iter().enumerate() {
                let image = kept.image(view).expect("a kept view");
                let mask = kept.robot(view).expect("a kept mask");
                let flags: Vec<bool> = labels
                    .iter()
                    .map(|label| *label == Label::Object(object))
                    .collect();
                let one =
                    batchplan::Surface::from_depth(&device, &image, &flags, Some(mask), 0.005);
                let one = one.expect("a sound surface");
                let noise = rendered[view].noise;
                let off_robot =
                    (0..image.depth.len()).filter(|&i| image.depth[i] > 0.0 && !mask.on_robot[i]);
                for i in off_robot {
                    let its = labels[i] == Label::Object(object);
                    if surface.object.get(k) != Some(&its) {
                        wrong.push(format!(
                            "view {view}, pixel {i}: flagged {:?}, labelled {its}",
                            surface.object.get(k)
                        ));
                        break;
                    }
                    if its && c == 0 {
                        let p = surface.points[k];
                        let off = (part.clone())
                            .map(|thing| (things[thing].distance)(p).abs())
                            .fold(f32::INFINITY, f32::min);
                        if off > 3.0 * deviation(noise, image.depth[i]) {
                            wrong.push(format!("view {view}, pixel {i}: {off} m off its thing"));
                            break;
                        }
                    }
                    k += 1;
                }
                match &mut by_hand {
                    Some(joined) => joined.join(one),
                    None => by_hand = Some(one),
                }
            }
            let by_hand = by_hand.expect("a look of two views");
            if k != surface.points.len()
                || surface.points != by_hand.points
                || surface.normals != by_hand.normals
                || surface.object != by_hand.object
                || surface.seen_from != by_hand.seen_from
            {
                wrong.push(format!("things {part:?}: not the surface written out"));
            }
        }
        let unknown = kept
            .surface(&device, seen.memory.next_id())
            .expect("a sound call");
        if unknown.is_some() {
            wrong.push("an id the look has not has a surface".to_string());
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_kept_look_gives_an_objects_surface", &found);
}

/// The wide bowl, which is not solid, picked and let go over a crumb the map holds: a step into it
/// meets its wall where it stands. The map saw that place free before the bowl was there, and is
/// never seen there after the let-go, so the bowl's part of the map is solid at its wall.
#[test]
fn a_thing_let_go_that_is_not_solid_is_never_seen_where_it_lies() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let (from, to) = (Vec3::new(0.45, -0.17, 0.0), Vec3::new(0.5, 0.17, 0.0));
    let things = vec![
        table(),
        wide_bowl(from),
        block(
            to + Vec3::new(0.0, 0.0, 0.0015),
            Vec3::splat(0.0015),
            [200, 30, 200],
        ),
        arm(&robot, away),
    ];
    let rendered = render(&things);
    let wall = to + Vec3::new(0.0, 0.12, 0.04);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let Some(bowl) = id_of(seen.kept.scene(), &rendered, 1) else {
            found.push((*camera, Err("no object holds the bowl".to_string())));
            continue;
        };
        let object = seen
            .memory
            .objects()
            .iter()
            .find(|o| o.id == bowl)
            .expect("the bowl");
        assert!(!object.solid, "{camera}: the bowl is not solid");
        let frame = cover(&object.shape).pose();
        let held = grasped(&seen.memory, bowl);
        let at = Pose::new(frame.position + (to - from), frame.rotation);
        let let_go = (seen.memory)
            .after(&[
                Act::Picked { id: bowl, held },
                Act::LetGo {
                    id: bowl,
                    at,
                    on: None,
                },
            ])
            .expect("a pick and a let-go");
        let step = Step {
            into: &[bowl],
            ..Step::default()
        };
        let planned = let_go.world(&device, &step).expect("a world");
        let at = planned
            .of
            .iter()
            .position(|of| matches!(of, Of::Object(o) | Of::Region(o) if *o == bowl));
        let reads = at.map(|at| planned.world.obstacles[at].distance(wall).value);
        let wrong = match reads {
            Some(distance) if distance < 0.0 => vec![],
            _ => vec![format!("let go, its wall reads {reads:?}")],
        };
        found.push((*camera, outcome(&wrong)));
    }
    verdict(
        "a_thing_let_go_that_is_not_solid_is_never_seen_where_it_lies",
        &found,
    );
}

/// A stance whose joint is no number is refused, naming it: the robot's mask would leave the arm
/// out of none of the views, and the arm would be objects.
#[test]
fn a_stance_that_is_no_number_is_refused() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let things = reaching_over_a_tray(&robot, &OVER_THE_CAN);
    let rendered = render(&things);
    let cameras = cameras(&rendered, [0, 0], 0.0);
    let mut q = OVER_THE_CAN;
    q[3] = f32::NAN;
    let locked = HashMap::new();
    let stance = Stance {
        q: &q,
        locked: &locked,
        padding: padding(&rendered, &rendered, things.len() - 1),
    };
    let refused = observe(&device, &cameras, &stance, &memory(&device));
    let Err(batchplan::Error::Input(why)) = refused else {
        panic!("a stance that is no number is taken: {refused:?}");
    };
    assert!(why.contains("NaN"), "{why}");
}

/// The height of the top of `room`, a box or an upright cylinder.
fn top_of(room: &Obstacle) -> f32 {
    match *room {
        Obstacle::Cylinder {
            center,
            half_height,
            ..
        } => center.z + half_height,
        Obstacle::Cuboid {
            center,
            half_extents,
            ..
        } => center.z + half_extents.z,
        _ => panic!("a room these tests do not know: {room:?}"),
    }
}

/// Whether some of `spheres`, as `[x, y, z, radius]`, lies in `room`.
fn in_room(spheres: &[[f32; 4]], room: &Obstacle) -> bool {
    let holds = |&[x, y, z, radius]: &[f32; 4]| room.distance(Vec3::new(x, y, z)).value < radius;
    spheres.iter().any(holds)
}

/// The box picked and held over the wide bowl, the hand 5 mm clear of the bowl's room and the box
/// in it: a step that takes the hand there goes into the bowl, the bowl its part of the map,
/// whether the step reaches that pose or passes a configuration that puts the hand there.
#[test]
fn a_held_thing_carried_into_a_room_takes_the_step_into_it() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let the_box = Vec3::new(0.45, -0.3, 0.03);
    let things = vec![
        table(),
        wide_bowl(MIDDLE),
        block(the_box, BOX_HALF, [200, 30, 30]),
        arm(&robot, away),
    ];
    let rendered = render(&things);
    let down = robot.ee_pose(&IN_BOWL, 0).rotation;
    let worlds = device.upload(&[World::default()]).expect("an empty world");
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let ids = [1, 2].map(|thing| id_of(seen.kept.scene(), &rendered, thing));
        let [Some(bowl), Some(boxed)] = ids else {
            found.push((
                *camera,
                Err("no object holds the bowl or the box".to_string()),
            ));
            continue;
        };
        let object = seen
            .memory
            .objects()
            .iter()
            .find(|o| o.id == bowl)
            .expect("the bowl");
        let room = cover(&object.shape);
        let held = grasped(&seen.memory, boxed);
        let picked = (seen.memory)
            .after(&[Act::Picked {
                id: boxed,
                held: held.clone(),
            }])
            .expect("a pick");
        let over = top_of(&room) + robot.tip(0).max(0.0) + 0.005;
        let pose = Pose::new(Vec3::new(0.5, 0.0, over), down);
        let hand: Vec<[f32; 4]> = (robot.hand_spheres(0).into_iter())
            .map(|sphere| {
                let center = pose.position + pose.rotation * sphere.center;
                [center.x, center.y, center.z, sphere.radius]
            })
            .collect();
        let carried = pose.mul_pose(held.grasp.inverse()).position;
        assert!(
            !in_room(&hand, &room) && room.distance(carried).value < 0.0,
            "{camera}: the hand is clear of the bowl's room and the box in it"
        );
        let problem = IkProblem {
            seed: Some(IN_BOWL.to_vec()),
            ..IkProblem::new(0, pose)
        };
        let ik = IkOptions {
            seeds: 1,
            ..IkOptions::default()
        };
        let solved = batchplan::solve_ik(&device, &worlds, &[problem], &ik).expect("an IK call");
        let q = solved
            .solved()
            .next()
            .expect("the hand there")
            .solution
            .to_vec();
        assert!(
            !in_room(&robot.spheres(&q), &room),
            "{camera}: the arm is clear of the room"
        );
        // What the bowl is for `step`.
        let of = |step: &Step<'_>| {
            let planned = picked.world(&device, step).expect("a world");
            let mut of = planned.of.iter().copied();
            of.find(|of| matches!(of, Of::Object(o) | Of::Region(o) if *o == bowl))
        };
        let mut wrong = vec![];
        let reaching = Step {
            reach: &[(0, pose)],
            ..Step::default()
        };
        if of(&reaching) != Some(Of::Region(bowl)) {
            wrong.push(format!("reaching there, the bowl is {:?}", of(&reaching)));
        }
        let passing = Step {
            at: &[&q],
            ..Step::default()
        };
        if of(&passing) != Some(Of::Region(bowl)) {
            wrong.push(format!("passing there, the bowl is {:?}", of(&passing)));
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict(
        "a_held_thing_carried_into_a_room_takes_the_step_into_it",
        &found,
    );
}

/// A hold whose fingers closed on a cube 1 cm across, the hand pointing down beside the wide bowl,
/// its fingertips 1.5 cm under the top of the bowl's room and its frame 4 cm outside it, the
/// fingers opening toward it: open, a fingertip would be in the room; closed, as the hold locks
/// them, none is, the palm is over it and the cube outside, and a step that reaches there does not
/// go into the bowl.
#[test]
fn a_hold_keeps_its_fingers_where_it_closed_them() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let the_box = Vec3::new(0.45, -0.3, 0.03);
    let things = vec![
        table(),
        wide_bowl(MIDDLE),
        block(the_box, BOX_HALF, [200, 30, 30]),
        arm(&robot, away),
    ];
    let rendered = render(&things);
    let down = robot.ee_pose(&IN_BOWL, 0).rotation;
    let toward = (down * Vec3::Y).with_z(0.0).normalize();
    let closed = HashMap::from([
        ("panda_finger_joint1".to_string(), 0.0),
        ("panda_finger_joint2".to_string(), 0.0),
    ]);
    let placed = |robot: &Robot, pose: Pose| -> Vec<[f32; 4]> {
        (robot.hand_spheres(0).into_iter())
            .map(|sphere| {
                let center = pose.position + pose.rotation * sphere.center;
                [center.x, center.y, center.z, sphere.radius]
            })
            .collect()
    };
    let closed_robot = robot.relock(&closed).expect("the fingers closed");
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let ids = [1, 2].map(|thing| id_of(seen.kept.scene(), &rendered, thing));
        let [Some(bowl), Some(boxed)] = ids else {
            found.push((
                *camera,
                Err("no object holds the bowl or the box".to_string()),
            ));
            continue;
        };
        let object = seen
            .memory
            .objects()
            .iter()
            .find(|o| o.id == bowl)
            .expect("the bowl");
        let room = cover(&object.shape);
        let Obstacle::Cylinder { center, radius, .. } = room else {
            found.push((*camera, Err(format!("the bowl's room is {room:?}"))));
            continue;
        };
        let at = center.with_z(top_of(&room) - 0.015) - toward * (radius + 0.04);
        let pose = Pose::new(at, down);
        let held = Held {
            hand: 0,
            object: Obstacle::cuboid(Vec3::ZERO, Vec3::splat(0.005)),
            grasp: Pose::IDENTITY,
            locked: closed.clone(),
            spheres: SphereOptions::default(),
        };
        assert!(
            in_room(&placed(&robot, pose), &room)
                && !in_room(&placed(&closed_robot, pose), &room)
                && room.distance(pose.position).value > 0.005,
            "{camera}: open, a finger is in the room; closed, none is, and the cube is outside"
        );
        let picked = (seen.memory)
            .after(&[Act::Picked { id: boxed, held }])
            .expect("a pick");
        let step = Step {
            reach: &[(0, pose)],
            ..Step::default()
        };
        let planned = picked.world(&device, &step).expect("a world");
        let mut of = planned.of.iter().copied();
        let of = of.find(|of| matches!(of, Of::Object(o) | Of::Region(o) if *o == bowl));
        let wrong = match of {
            Some(Of::Object(_)) => vec![],
            other => vec![format!(
                "the hand's closed fingers beside it make the bowl {other:?}"
            )],
        };
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_hold_keeps_its_fingers_where_it_closed_them", &found);
}

/// A hold a restart brings back: a memory of nothing seen takes a pick of an id it never gave,
/// above its next id, raises its next id past it and holds it; a let-go makes an object of the
/// held box where it lies, and a second look that sees the box there keeps its id. A pick of an id
/// under the next id that no object has, and of the last id a `u32` has, are refused, and so is a
/// let-go of a hold the memory never saw whose shape is no box or cylinder, though it may be lost.
#[test]
fn a_hold_a_restart_brings_back_is_held_and_let_go_as_its_shape() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let fresh = memory(&device);
    let held = Held {
        hand: 0,
        object: Obstacle::cuboid(Vec3::ZERO, BOX_HALF),
        grasp: Pose::new(Vec3::new(0.0, 0.0, -0.02), Quat::IDENTITY),
        locked: HashMap::new(),
        spheres: SphereOptions::default(),
    };
    let id = fresh.next_id() + 7;
    let pick = |id: u32, held: &Held| Act::Picked {
        id,
        held: held.clone(),
    };
    let restored = fresh
        .after(&[pick(id, &held)])
        .expect("a hold brought back");
    assert_eq!(restored.held(0), Some((id, &held)), "the hand holds it");
    assert_eq!(restored.next_id(), id + 1, "no later id is its");
    assert!(restored.objects().is_empty(), "it is among no objects");
    let refused = |memory: &Memory, acts: &[Act], named: &str| {
        let Err(batchplan::Error::Input(why)) = memory.after(acts) else {
            panic!("{acts:?} are taken");
        };
        assert!(why.contains(named), "{why}");
    };
    refused(&fresh, &[pick(u32::MAX, &held)], &u32::MAX.to_string());
    refused(
        &restored,
        &[Act::Lost { id }, pick(id - 1, &held)],
        "under its next id",
    );
    let capsule = Held {
        object: Obstacle::capsule(Vec3::ZERO, 0.02, 0.05),
        ..held
    };
    let let_go = |id| Act::LetGo {
        id,
        at: Pose::new(BOX, Quat::IDENTITY),
        on: None,
    };
    refused(
        &fresh,
        &[pick(id, &capsule), let_go(id)],
        "no box or cylinder",
    );
    let lost = fresh.after(&[pick(id, &capsule), Act::Lost { id }]);
    assert!(
        lost.is_ok_and(|lost| lost.held(0).is_none()),
        "a hold of any shape may be lost"
    );

    let along = Vec3::new(0.0, 0.15, 0.0);
    let at = Pose::new(BOX + along, Quat::IDENTITY);
    let set_down = restored
        .after(&[Act::LetGo { id, at, on: None }])
        .expect("a let-go");
    let object = set_down.objects().iter().find(|o| o.id == id);
    assert!(
        object.is_some_and(|object| object.solid
            && cover(&object.shape) == Obstacle::cuboid(BOX + along, BOX_HALF)),
        "let go, it is the held box where it lies: {object:?}"
    );
    let things = acts_scene(&robot, away, Some(BOX + along), None, None);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let (_, after) = second_look(
            &device,
            &things,
            things.len() - 1,
            (c, 10 * c as u64 + 5),
            away,
            &set_down,
        );
        let kept = after.memory.objects().iter().position(|o| o.id == id);
        let labels = after.kept.scene().labels.iter().flatten();
        let named =
            kept.is_some_and(|object| labels.clone().any(|label| *label == Label::Object(object)));
        let wrong = match named {
            true => vec![],
            false => vec![format!(
                "seen where it was let go, it is kept: {}, named: {named}",
                kept.is_some()
            )],
        };
        found.push((*camera, outcome(&wrong)));
    }
    verdict(
        "a_hold_a_restart_brings_back_is_held_and_let_go_as_its_shape",
        &found,
    );
}

/// A can 6 cm across and 6 cm tall standing on a box 14 cm square and 6 cm tall, at `at` on the
/// table: the box, then the can.
fn can_on_a_box(at: Vec3) -> [Thing; 2] {
    [
        block(
            at + Vec3::new(0.0, 0.0, 0.03),
            Vec3::new(0.07, 0.07, 0.03),
            [200, 200, 30],
        ),
        can(at + Vec3::new(0.0, 0.0, 0.09), 0.03, 0.03, [20, 20, 20]),
    ]
}

/// Two cans on boxes, one in the table's middle and one beside it: a pick of the middle box takes
/// its can out of the objects with it, and what the other can stands on is its box among the
/// objects left, renumbered; a push of the other box into the middle after that pick moves its can
/// too, which a second look then finds there under its id.
#[test]
fn a_pick_takes_what_stands_on_the_thing_and_a_push_moves_it() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let (far, near) = (MIDDLE, Vec3::new(0.5, -0.26, 0.0));
    let scene = |near: Vec3| {
        let mut things = vec![table()];
        things.extend(can_on_a_box(far));
        things.extend(can_on_a_box(near));
        things.push(arm(&robot, away));
        things
    };
    let things = scene(near);
    let rendered = render(&things);
    // Into the middle, where a stack is told apart through every camera, the middle stack gone.
    let by = far - near;
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let held_by = |thing| holder(seen.kept.scene(), &rendered, thing);
        let [Some(far_box), Some(far_can), Some(near_box), Some(near_can)] =
            [1, 2, 3, 4].map(held_by)
        else {
            found.push((*camera, Err("no object holds a box or a can".to_string())));
            continue;
        };
        let objects = seen.memory.objects();
        let on = |object: usize, under: usize| objects[object].on == Some(Support::Object(under));
        let mut wrong = vec![];
        if !(on(far_can, far_box) && on(near_can, near_box)) {
            wrong.push("a can does not stand on its box".to_string());
            found.push((*camera, outcome(&wrong)));
            continue;
        }
        assert!(
            far_box.min(far_can) < near_box,
            "{camera}: the middle stack comes before the other's box, so taking it renumbers it"
        );
        let id = |object: usize| objects[object].id;
        let held = grasped(&seen.memory, id(far_box));
        let picked = (seen.memory)
            .after(&[Act::Picked {
                id: id(far_box),
                held,
            }])
            .expect("a pick");
        let left = picked.objects();
        let index = |object: usize| left.iter().position(|o| o.id == id(object));
        if index(far_can).is_some() {
            wrong.push("picked, what stood on the box stays among the objects".to_string());
        }
        let stands = index(near_can).map(|can| left[can].on);
        if stands != Some(index(near_box).map(Support::Object)) {
            wrong.push(format!(
                "picked, the near can stands on {stands:?}, not the near box"
            ));
        }
        let pushed = picked
            .after(&[Act::Pushed {
                id: id(near_box),
                by,
            }])
            .expect("a push");
        let moved = pushed.objects().iter().find(|o| o.id == id(near_can));
        let was = objects[near_can].shape;
        if !moved
            .is_some_and(|o| cover(&o.shape).pose().position == cover(&was).pose().position + by)
        {
            wrong.push("pushed, the near box leaves its can where it stood".to_string());
        }
        let mut things = vec![table()];
        things.extend(can_on_a_box(near + by));
        things.push(arm(&robot, away));
        let (rendered, after) = second_look(
            &device,
            &things,
            things.len() - 1,
            (c, 10 * c as u64 + 5),
            away,
            &pushed,
        );
        let now = id_of(after.kept.scene(), &rendered, 2);
        if now != Some(id(near_can)) {
            wrong.push(format!(
                "pushed and seen again, the near can is {now:?}, not {}",
                id(near_can)
            ));
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict(
        "a_pick_takes_what_stands_on_the_thing_and_a_push_moves_it",
        &found,
    );
}

/// A can on a box, the box picked: what stood on it is in some memory, the objects, a hand's hold
/// or the map, so that a plan meets it where it is. It is in none: the hand holds the box alone,
/// the can left the objects with it, and the map never held it.
#[test]
fn what_stood_on_a_picked_thing_is_in_a_memory() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let at = Vec3::new(0.5, 0.0, 0.0);
    let mut things = vec![table()];
    things.extend(can_on_a_box(at));
    things.push(arm(&robot, away));
    let rendered = render(&things);
    let can_middle = at + Vec3::new(0.0, 0.0, 0.09);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let Some(the_box) = id_of(seen.kept.scene(), &rendered, 1) else {
            found.push((*camera, Err("no object holds the box".to_string())));
            continue;
        };
        let held = grasped(&seen.memory, the_box);
        let picked = (seen.memory)
            .after(&[Act::Picked { id: the_box, held }])
            .expect("a pick");
        let planned = picked.world(&device, &Step::default()).expect("a world");
        let nearest = (planned.world.obstacles.iter())
            .map(|obstacle| obstacle.distance(can_middle).value)
            .fold(f32::INFINITY, f32::min);
        let wrong = match nearest < 0.0 {
            true => vec![],
            false => vec![format!(
                "picked, nothing holds the can: the nearest obstacle is {nearest:.3} m from its middle"
            )],
        };
        found.push((*camera, outcome(&wrong)));
    }
    verdict("what_stood_on_a_picked_thing_is_in_a_memory", &found);
}

/// Box A let go flush against box B, which did not move: a second look keeps both ids, a label
/// naming each. A look makes two boxes touching one object (`tests/truth.rs`, "two boxes
/// touching"), so one of the two reads gone.
#[test]
fn a_thing_let_go_against_another_keeps_both_ids() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let (a, b) = (Vec3::new(0.45, -0.2, 0.03), Vec3::new(0.45, 0.1, 0.03));
    let to = Vec3::new(b.x, b.y - 2.0 * BOX_HALF.y, b.z);
    let scene = |a: Vec3| {
        vec![
            table(),
            block(b, BOX_HALF, [30, 30, 200]),
            block(a, BOX_HALF, [200, 30, 30]),
            arm(&robot, away),
        ]
    };
    let things = scene(a);
    let rendered = render(&things);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let ids = [1, 2].map(|thing| id_of(seen.kept.scene(), &rendered, thing));
        let [Some(b_id), Some(a_id)] = ids else {
            found.push((*camera, Err("no object holds a box".to_string())));
            continue;
        };
        let shape = seen
            .memory
            .objects()
            .iter()
            .find(|o| o.id == a_id)
            .expect("A")
            .shape;
        let frame = cover(&shape).pose();
        let held = grasped(&seen.memory, a_id);
        let at = Pose::new(frame.position + (to - a), frame.rotation);
        let let_go = (seen.memory)
            .after(&[
                Act::Picked { id: a_id, held },
                Act::LetGo {
                    id: a_id,
                    at,
                    on: None,
                },
            ])
            .expect("a pick and a let-go");
        let things = scene(to);
        let (_, after) = second_look(
            &device,
            &things,
            things.len() - 1,
            (c, 10 * c as u64 + 5),
            away,
            &let_go,
        );
        let named = |id: u32| {
            let object = after.memory.objects().iter().position(|o| o.id == id);
            let labels = after.kept.scene().labels.iter().flatten();
            object.is_some_and(|object| labels.clone().any(|label| *label == Label::Object(object)))
        };
        let mut wrong = vec![];
        if !named(a_id) {
            wrong.push("A, let go there, reads gone".to_string());
        }
        if !named(b_id) {
            wrong.push("B, which did not move, reads gone".to_string());
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_thing_let_go_against_another_keeps_both_ids", &found);
}

/// The Panda's fingers closed, the hand over the table in both views: the stance's locked joints
/// say so, and the robot's mask holds the fingers where they are; no object is made of the arm.
#[test]
fn a_stance_masks_the_fingers_where_they_were_read() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let closed = HashMap::from([
        ("panda_finger_joint1".to_string(), 0.0),
        ("panda_finger_joint2".to_string(), 0.0),
    ]);
    let shut = robot.relock(&closed).expect("the fingers closed");
    let things = vec![table(), arm(&shut, &OVER_THE_CAN)];
    let rendered = render(&things);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let (frames, bands) = read(&rendered, c, 10 * c as u64);
        let stance = Stance {
            q: &OVER_THE_CAN,
            locked: &closed,
            padding: padding(&rendered, &frames, 1),
        };
        let cameras = cameras(&frames, bands, 0.0);
        let seen = observe(&device, &cameras, &stance, &memory(&device)).expect("a look");
        let wrong = match an_object_of(seen.kept.scene(), &rendered, 1)
            || holder(seen.kept.scene(), &rendered, 1).is_some()
        {
            false => vec![],
            true => vec!["the closed fingers are an object's".to_string()],
        };
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_stance_masks_the_fingers_where_they_were_read", &found);
}

/// The 4 mm bowl, which is not solid, in a first look, and gone in a second that sees its place
/// empty: under `Weighing::Latest` the second look decides, and the map holds nothing where its
/// wall stood; under the default, `Weighing::Evidence`, one look through it does not outweigh one
/// that read it, and the map still holds its wall.
#[test]
fn latest_lets_a_look_decide_what_it_sees() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let things = acts_scene(&robot, away, None, Some(BOWL), None);
    let empty = acts_scene(&robot, away, None, None, None);
    let rendered = render(&things);
    let wall = BOWL + Vec3::new(0.0, 0.043, 0.025);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let mut wrong = vec![];
        for (weighing, holds) in [(Weighing::Latest, false), (Weighing::Evidence, true)] {
            let (frames, bands) = read(&rendered, c, 10 * c as u64);
            let locked = HashMap::new();
            let stance = Stance {
                q: away,
                locked: &locked,
                padding: padding(&rendered, &frames, things.len() - 1),
            };
            let o = MemoryOptions {
                weighing,
                ..MemoryOptions::default()
            };
            let memory = Memory::try_new(&device, vec![known(&table())], &o).expect("a memory");
            let seen = observe(&device, &cameras(&frames, bands, 0.0), &stance, &memory);
            let seen = seen.expect("a look");
            assert!(
                (seen.memory.objects().iter()).any(|object| !object.solid),
                "{camera}: the map holds the bowl"
            );
            let (_, after) = second_look(
                &device,
                &empty,
                empty.len() - 1,
                (c, 10 * c as u64 + 5),
                away,
                &seen.memory,
            );
            let planned = after
                .memory
                .world(&device, &Step::default())
                .expect("a world");
            let grid = (planned.of.last() == Some(&Of::Unexplained))
                .then(|| planned.world.obstacles.last())
                .flatten();
            let held = grid.is_some_and(|grid| grid.distance(wall).value < 0.0);
            if held != holds {
                wrong.push(format!(
                    "under {weighing:?}, the map holds the wall: {held}"
                ));
            }
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("latest_lets_a_look_decide_what_it_sees", &found);
}

/// The arm over the tray scene: what a camera reads in mid-air within one pixel of the arm's
/// outline, farther than three deviations from every surface, is the arm's by the camera's band,
/// and the map holds none of it; through a stereo camera it holds fewer of them with the band than
/// without. Exact depth reads none.
#[test]
fn the_band_takes_what_a_camera_reads_beside_the_arm() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let things = reaching_over_a_tray(&robot, &OVER_THE_CAN);
    let arm = things.len() - 1;
    let rendered = render(&things);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let (frames, bands) = read(&rendered, c, 10 * c as u64);
        let locked = HashMap::new();
        let stance = Stance {
            q: &OVER_THE_CAN,
            locked: &locked,
            padding: padding(&rendered, &frames, arm),
        };
        // Of the mid-air readings beside the arm's outline, how many, and how many the map holds,
        // with the cameras' bands as `bands`.
        let held = |bands: [usize; 2]| {
            let seen = observe(
                &device,
                &cameras(&frames, bands, 0.0),
                &stance,
                &memory(&device),
            );
            let seen = seen.expect("a look");
            let planned = seen
                .memory
                .world(&device, &Step::default())
                .expect("a world");
            let grid = (planned.of.last() == Some(&Of::Unexplained))
                .then(|| planned.world.obstacles.last())
                .flatten();
            let [mut mixed, mut held] = [0, 0];
            for (view, (truth, frame)) in rendered.iter().zip(&frames).enumerate() {
                let image = seen.kept.image(view).expect("a kept view");
                let (width, height) = (
                    truth.width as isize,
                    (truth.depth.len() / truth.width) as isize,
                );
                let arm_at = |x: isize, y: isize| {
                    (0..width).contains(&x)
                        && (0..height).contains(&y)
                        && truth.shows[(y * width + x) as usize] == Some(arm)
                };
                for i in 0..image.depth.len() {
                    let (u, v) = ((i % truth.width) as isize, (i / truth.width) as isize);
                    let beside = (-1..=1).any(|dv| (-1..=1).any(|du| arm_at(u + du, v + dv)));
                    if image.depth[i] <= 0.0 || truth.shows[i] == Some(arm) || !beside {
                        continue;
                    }
                    let (k, z) = (frame.intrinsics, image.depth[i]);
                    let ray = Vec3::new(
                        (u as f32 - k.cx) / k.fx * z,
                        (v as f32 - k.cy) / k.fy * z,
                        z,
                    );
                    let p = frame.camera.rotation * ray + frame.camera.position;
                    let off = (things.iter())
                        .map(|thing| (thing.distance)(p).abs())
                        .fold(f32::INFINITY, f32::min);
                    if off > 3.0 * deviation(frame.noise, z) {
                        mixed += 1;
                        held += usize::from(grid.is_some_and(|grid| grid.distance(p).value < 0.0));
                    }
                }
            }
            (mixed, held)
        };
        let (mixed, with) = held(bands);
        let mut wrong = vec![];
        if with > 0 {
            wrong.push(format!("the map holds {with} of {mixed}"));
        }
        if c > 0 {
            let (_, without) = held([0, 0]);
            if without <= with {
                wrong.push(format!("without the band it holds {without}, no more"));
            }
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("the_band_takes_what_a_camera_reads_beside_the_arm", &found);
}

/// A thing the map holds that becomes a shape: a crumb 3 mm across and a pebble 1 cm across that a
/// first look through both cameras makes no object of, so that the map holds them, and that a look
/// from 20 cm makes a solid object of. Under the default weighing that look does not outweigh the
/// first, yet the map holds nothing at either where it is solid: what became a shape was freed
/// before the fuse. Each thing is held so wherever a camera passes it from the map to a shape.
#[test]
fn what_becomes_a_shape_leaves_the_map() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let things_at = [
        (1, Vec3::new(0.35, 0.3, 0.0015), Vec3::splat(0.0015)),
        (2, Vec3::new(0.75, -0.25, 0.005), Vec3::splat(0.005)),
    ];
    let mut things = vec![table()];
    things.extend(things_at.map(|(_, at, half)| block(at, half, [200, 30, 200])));
    things.push(arm(&robot, away));
    let rendered = render(&things);
    let locked = HashMap::new();
    let mut found = vec![];
    for (c, (camera, _, near)) in CAMERAS.iter().enumerate() {
        let first = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        // What the map holds unexplained at `at`, where it holds anything.
        let held_at = |memory: &Memory, at: Vec3| {
            let planned = memory.world(&device, &Step::default()).expect("a world");
            let grid = (planned.of.last() == Some(&Of::Unexplained))
                .then(|| planned.world.obstacles.last())
                .flatten();
            grid.is_some_and(|grid| grid.distance(at).value < 0.0)
        };
        let mut wrong = vec![];
        let mut passed = 0;
        for (thing, at, _) in things_at {
            if holder(first.kept.scene(), &rendered, thing).is_some() || !held_at(&first.memory, at)
            {
                continue;
            }
            // A look from 20 cm over the thing, a little aside, by the near camera of the set.
            let view = looking(at + Vec3::new(0.08, 0.0, 0.2), at);
            let truth = Frame::of(&things, view);
            let frame = match near {
                Some(stereo) => truth.clone().sensed(*stereo, 100 + 10 * c as u64),
                None => truth.clone(),
            };
            let frames = [frame.clone(), frame];
            let stance = Stance {
                q: away,
                locked: &locked,
                padding: padding(&rendered, &frames, things.len() - 1),
            };
            let near_camera = cameras(&frames, [usize::from(c > 0); 2], 0.0);
            let second = observe(&device, &near_camera[..1], &stance, &first.memory);
            let second = second.expect("a look");
            let shown = [truth.clone(), truth];
            let solid = holder(second.kept.scene(), &shown, thing)
                .is_some_and(|object| second.kept.scene().objects[object].solid);
            if !solid {
                continue;
            }
            passed += 1;
            if held_at(&second.memory, at) {
                wrong.push(format!(
                    "thing {thing}, a solid object now, is held by the map"
                ));
            }
        }
        assert!(
            passed > 0,
            "{camera}: some thing passes from the map to a shape"
        );
        found.push((*camera, outcome(&wrong)));
    }
    verdict("what_becomes_a_shape_leaves_the_map", &found);
}

/// A box picked and let go on what the caller set it down on: on the table, a known shape, it
/// stands on the table; on a second box, it stands on that box among the objects, so that a push
/// of the second box carries it and a pick of it takes it along. A known shape past those the
/// memory knows, an id no object has and the box's own id are refused, naming the act.
#[test]
fn a_thing_let_go_stands_on_what_it_was_set_down_on() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let (under, under_half) = (Vec3::new(0.5, 0.2, 0.03), Vec3::new(0.07, 0.07, 0.03));
    let things = vec![
        table(),
        block(BOX, BOX_HALF, [200, 30, 30]),
        block(under, under_half, [200, 200, 30]),
        arm(&robot, away),
    ];
    let rendered = render(&things);
    let seen = first_look(&device, &things, &rendered, (0, 0), away, 0.0);
    let [a, b] = [1, 2].map(|thing| id_of(seen.kept.scene(), &rendered, thing).expect("a box"));
    let held = grasped(&seen.memory, a);
    let shape = seen
        .memory
        .objects()
        .iter()
        .find(|o| o.id == a)
        .expect("the box")
        .shape;
    let frame = cover(&shape).pose();
    let on_table = Pose::new(frame.position + Vec3::new(0.0, -0.1, 0.0), frame.rotation);
    let on_box = Pose::new(
        under + Vec3::new(0.0, 0.0, under_half.z + BOX_HALF.z),
        frame.rotation,
    );
    let let_go = |at: Pose, on: Option<Under>| {
        (seen.memory).after(&[
            Act::Picked {
                id: a,
                held: held.clone(),
            },
            Act::LetGo { id: a, at, on },
        ])
    };
    let on = |memory: &Memory, id: u32| {
        let object = memory.objects().iter().find(|o| o.id == id);
        object.and_then(|object| object.on)
    };
    let tabled = let_go(on_table, Some(Under::Known(0))).expect("a let-go on the table");
    assert_eq!(on(&tabled, a), Some(Support::Known(0)), "on the table");
    let stacked = let_go(on_box, Some(Under::Object(b))).expect("a let-go on the box");
    let index = stacked
        .objects()
        .iter()
        .position(|o| o.id == b)
        .expect("the box under");
    assert_ne!(
        index, b as usize,
        "the box under has an index that is not its id"
    );
    assert_eq!(on(&stacked, a), Some(Support::Object(index)), "on the box");
    let by = Vec3::new(0.0, 0.1, 0.0);
    let pushed = stacked.after(&[Act::Pushed { id: b, by }]).expect("a push");
    let carried = pushed.objects().iter().find(|o| o.id == a);
    let carried = carried.map(|object| cover(&object.shape).pose().position);
    assert_eq!(
        carried,
        Some(on_box.position + by),
        "a push of the box under carries it"
    );
    let lifted = (stacked.after(&[Act::Picked {
        id: b,
        held: grasped(&stacked, b),
    }]))
    .expect("a pick");
    assert!(
        lifted.objects().iter().all(|o| o.id != a),
        "a pick of the box under takes it along"
    );
    let unknown = seen.memory.next_id();
    let refused = [
        (Under::Known(1), "known shape 1".to_string()),
        (Under::Object(unknown), format!("object {unknown}")),
        (Under::Object(a), "itself".to_string()),
    ];
    for (under, named) in refused {
        let Err(batchplan::Error::Input(why)) = let_go(on_box, Some(under.clone())) else {
            panic!("a let-go on {under:?} is taken");
        };
        assert!(why.contains("act 1") && why.contains(&named), "{why}");
    }
}

/// The wide bowl's rim has measured material between the fingers through every camera.
#[test]
fn a_grasp_of_the_rim_separates_its_own_readings() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let gripper = panda_gripper(&robot);
    let away = robot.default_q();
    let things = vec![table(), wide_bowl(MIDDLE), arm(&robot, away)];
    let rendered = render(&things);
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), away, 0.0);
        let id = id_of(seen.kept.scene(), &rendered, 1).expect("the bowl has an object");
        let surface = seen
            .kept
            .surface(&device, id)
            .unwrap()
            .expect("the bowl's surface");
        let options = batchplan::SurfaceGraspOptions {
            turns: 1,
            shell: Some(0.015),
            ..Default::default()
        };
        let gs = batchplan::grasps_on(&device, &gripper, &[surface], &options)
            .unwrap()
            .remove(0);
        let g = gs.first().expect("a grasp of the rim");
        let prepared = seen
            .grip_worlds(&device, &Step::default(), &gripper, id)
            .unwrap();
        // The nearest finger surface is 1 cm above even the highest reading. The coarse
        // hand window still contains the rim, but there is no material between the pads.
        let finger_tip = robot
            .hand_spheres(0)
            .iter()
            .filter(|s| s.side != 0.0)
            .map(|s| s.center.z + s.radius)
            .fold(f32::NEG_INFINITY, f32::max);
        let surface = seen.kept.surface(&device, id).unwrap().unwrap();
        let highest = surface
            .points
            .iter()
            .zip(&surface.object)
            .filter(|(_, own)| **own)
            .map(|(p, _)| p.z)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut above = *g;
        above.pose = Pose::new(
            Vec3::new(
                g.contacts[0].x,
                g.contacts[0].y,
                highest + 0.01 + finger_tip,
            ),
            Quat::from_rotation_x(std::f32::consts::PI),
        );
        assert!(
            prepared.project(&above).unwrap().is_none(),
            "{camera}: the fingers close in air"
        );
        let split = prepared
            .project(g)
            .unwrap()
            .expect("material between the fingers");
        assert_eq!(split.scene.of[split.part], Of::Gripped(id), "{camera}");
        assert!(
            split.scene.world.touch.is_empty(),
            "the projection adds no touch"
        );
        let part = &split.scene.world.obstacles[split.part];
        assert!(
            g.contacts
                .iter()
                .any(|p| part.distance(*p).value <= SurfaceGraspOptions::default().contact),
            "{camera}: its contact is on the selected patch"
        );
        let opposite = MIDDLE
            + Vec3::new(
                MIDDLE.x - g.contacts[0].x,
                MIDDLE.y - g.contacts[0].y,
                g.contacts[0].z,
            );
        let wall = split
            .scene
            .world
            .obstacles
            .iter()
            .enumerate()
            .any(|(k, o)| k != split.part && o.distance(opposite).value <= 0.0);
        assert!(
            wall,
            "{camera}: the opposite rim must still collide, at {opposite:?}"
        );
        assert!(
            part.distance(opposite).value > 0.08,
            "{camera}: only the gripped rim belongs to the part"
        );
        let index = seen
            .memory
            .objects()
            .iter()
            .position(|o| o.id == id)
            .unwrap();
        let supporting = seen
            .memory
            .objects()
            .iter()
            .any(|o| o.on == Some(Support::Object(index)));
        assert_eq!(
            split.body.is_none(),
            supporting,
            "{camera}: what stands on it prevents a whole-body pick"
        );
        if let Some(body) = &split.body {
            let held = Held {
                hand: 0,
                object: body.shape.placed(Pose::IDENTITY),
                grasp: body
                    .shape
                    .pose()
                    .inverse()
                    .mul_pose(part.pose().mul_pose(split.grasp)),
                locked: HashMap::new(),
                spheres: Default::default(),
            };
            let after = seen.picked(&split, &held).unwrap();
            let next = after.world(&device, &Step::default()).unwrap();
            let prior = seen
                .memory
                .after(&[Act::Picked { id, held }])
                .unwrap()
                .world(&device, &Step::default())
                .unwrap();
            let lo = MIDDLE - Vec3::new(0.14, 0.14, 0.0);
            let hi = MIDDLE + Vec3::new(0.14, 0.14, 0.08);
            let mut preserved = 0;
            for x in 0..16 {
                for y in 0..16 {
                    for z in 0..8 {
                        let fraction = Vec3::new(x as f32 / 15.0, y as f32 / 15.0, z as f32 / 7.0);
                        let p = lo + (hi - lo) * fraction;
                        let retained = split.scene.of.iter().enumerate().any(|(k, of)| {
                            *of == Of::Region(id)
                                && !body.other_parts.contains(&k)
                                && split.scene.world.obstacles[k].distance(p).value <= 0.0
                        });
                        if retained
                            && prior
                                .world
                                .obstacles
                                .iter()
                                .all(|o| o.distance(p).value > 0.0)
                        {
                            assert!(
                                next.world
                                    .obstacles
                                    .iter()
                                    .any(|o| o.distance(p).value <= 0.0),
                                "{camera}: the bowl would have cleared residual material at {p:?}"
                            );
                            preserved += 1;
                        }
                    }
                }
            }
            assert!(
                preserved > 0,
                "{camera}: the rim pickup left material in the old room"
            );
        }
    }
}

/// Two views on opposite sides of the same 4.2 by 5 by 7 cm box and its table. From 30 cm
/// up the support's readings separate from the box's. From 18 cm up two bottom-edge voxels
/// hold both and block the lift. The permitted finger contact is identical in both cases.
#[test]
fn the_split_world_reaches_a_grip_and_lifts_only_attributable_material() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let gripper = panda_gripper(&robot);
    let q = robot.default_q();
    let things = vec![
        table(),
        block(
            MIDDLE + Vec3::Z * 0.035,
            Vec3::new(0.021, 0.025, 0.035),
            [210, 30, 30],
        ),
        arm(&robot, q),
    ];
    for (height, mixed) in [(0.30, false), (0.18, true)] {
        let rendered = [-0.28, 0.28].map(|side| {
            Frame::of(
                &things,
                looking(Vec3::new(MIDDLE.x, side, height), MIDDLE + Vec3::Z * 0.04),
            )
        });
        let seen = first_look(&device, &things, &rendered, (0, 0), q, 0.0);
        let id = id_of(seen.kept.scene(), &rendered, 1).expect("the box has an object");
        let surface = seen
            .kept
            .surface(&device, id)
            .unwrap()
            .expect("the box's surface");
        let options = SurfaceGraspOptions {
            turns: 1,
            ..Default::default()
        };
        let gs = batchplan::grasps_on(&device, &gripper, std::slice::from_ref(&surface), &options)
            .unwrap()
            .remove(0);
        let g = gs.first().expect("the fingers can close on the box");
        let source = seen
            .grip_worlds(&device, &Step::default(), &gripper, id)
            .unwrap();
        assert!(seen.kept.is_current(&seen.memory));
        let mut air = *g;
        air.pose.position.z += 0.3;
        let candidates = [air, *g, *g];
        let configure = |c: &mut miru::GripCandidate<'_>| {
            c.touches.push(batchplan::Touch::new(0, ["panda_link0"]));
            batchplan::Pick {
                support: c
                    .of
                    .iter()
                    .enumerate()
                    .filter_map(|(k, of)| (*of == Of::Known(0)).then_some(k))
                    .collect(),
                transit: false,
                lift: Vec3::Z * 0.08,
                ..c.pick(q)
            }
        };
        let search = miru::GripSearch {
            target: Some(miru::GripTarget::Near {
                point: g.contacts[0].midpoint(g.contacts[1]),
                radius: 0.0,
            }),
            batch: 2,
            enough: Some(1),
        };
        let opts = batchplan::TaskOptions::default();
        let mut outside = *g;
        outside.pose.position.x += 5.0;
        let bounded = seen
            .grip_worlds(&device, &Step::default(), &gripper, id)
            .unwrap()
            .with_reach_bound();
        let miru::GripOutcome::Complete(rejected) = bounded
            .plan(
                &[outside],
                &search,
                &opts,
                || false,
                |_| -> batchplan::Pick { panic!("a certified rejection needs no projected task") },
            )
            .unwrap()
        else {
            panic!("an uninterrupted search completes");
        };
        assert_eq!(rejected.batches, 0);
        let invalid = batchplan::TaskOptions {
            min_fraction: 0.0,
            ..opts.clone()
        };
        assert!(
            bounded
                .plan(&[outside], &search, &invalid, || false, configure)
                .is_err()
        );

        assert!(
            matches!(rejected.failures[0].reason, miru::GripStop::OutsideReach { at }
            if at == outside.pose)
        );
        let miru::GripOutcome::Complete(mixed_search) = bounded
            .plan(&[outside, *g], &search, &opts, || false, configure)
            .unwrap()
        else {
            panic!("an uninterrupted search completes");
        };
        assert_eq!(mixed_search.batches, 1);
        if !mixed {
            assert_eq!(mixed_search.plans[0].plan.candidate, 1);
        }
        let miru::GripOutcome::Complete(mut found) = source
            .plan(&candidates, &search, &opts, || false, configure)
            .unwrap()
        else {
            panic!("an uninterrupted search completes");
        };
        assert_eq!(
            found.batches, 1,
            "two projected candidates fit into one batch"
        );
        assert_eq!(found.failures[0].candidate, 0);
        assert!(matches!(found.failures[0].reason, miru::GripStop::NoPart));
        if mixed {
            assert!(found.plans.is_empty());
            assert_eq!(found.failures.len(), candidates.len());
            let miru::GripStop::Motion { stop, of } = &found.failures[1].reason else {
                panic!("the mixed boundary stops the departure");
            };
            assert_eq!(
                stop.step,
                batchplan::TaskStep::Depart,
                "the grasp and approach were clear"
            );
            assert!(
                stop.contacts.iter().any(|contact| matches!(contact,
                batchplan::Contact::World { obstacle, .. } if of[*obstacle] == Of::Region(id))),
                "unattributed boundary material must remain in the lift world"
            );
            continue;
        }
        let winner = found.plans.pop().expect("a whole pick");
        assert_eq!(winner.plan.candidate, 1, "the earlier feasible input wins");
        assert_eq!(winner.grasp.opening, g.opening);
        let hold = batchplan::Hold::new(Pose::IDENTITY, g.opening, &gripper);
        assert_eq!(
            winner.locked, hold.locked,
            "held with the fingers at its opening"
        );
        let transit = winner.world(batchplan::TaskStep::Transit);
        let contact = winner.world(batchplan::TaskStep::Approach);
        let depart = winner.world(batchplan::TaskStep::Depart);
        assert_eq!(
            transit.world.touch,
            [batchplan::Touch::new(0, ["panda_link0"])]
        );
        assert_eq!(
            contact.world.touch[1],
            batchplan::Touch::new(winner.grip.part, robot.finger_links(0))
        );
        assert!(depart.of.iter().all(|of| *of == Of::Known(0)));
        assert!(
            depart
                .of
                .iter()
                .enumerate()
                .all(|(i, _)| depart.world.touch.contains(&batchplan::Touch::held(i, 0)))
        );
        let split = winner.grip;
        let plan = &winner.plan;
        let body = split.body.as_ref().expect("nothing stands on the box");
        assert!(
            !body.other_parts.is_empty(),
            "the picked part is smaller than the body"
        );

        // The third candidate would lift the least, but two earlier ones already succeeded.
        let search = miru::GripSearch {
            enough: Some(2),
            batch: 3,
            target: None,
        };
        let miru::GripOutcome::Complete(first_two) = source
            .plan(
                &[*g; 3],
                &search,
                &opts,
                || false,
                |c| {
                    let mut pick = configure(c);
                    pick.lift = Vec3::Z * [0.08, 0.04, 0.02][c.index];
                    pick
                },
            )
            .unwrap()
        else {
            panic!("an uninterrupted search completes");
        };
        let indices: Vec<_> = first_two.plans.iter().map(|p| p.plan.candidate).collect();
        assert_eq!(
            indices,
            [1, 0],
            "the first two successes sort by their durations"
        );
        assert!(first_two.plans[0].plan.duration() < first_two.plans[1].plan.duration());
        // A candidate narrower than the loaded gap is held with the fingers at its opening.
        let mut narrowed = *g;
        narrowed.opening = gripper.openings()[1];
        let mut held = vec![];
        let planned = source.plan(
            &[narrowed],
            &search,
            &opts,
            || false,
            |c| {
                let pick = configure(c);
                held.push(pick.grasps[0].clone());
                pick
            },
        );
        assert!(planned.is_ok(), "{planned:?}");
        assert_eq!(held.len(), 1, "one candidate, one task");
        assert_eq!(held[0].locked, gripper.locked_at(narrowed.opening));
        assert!(!held[0].locked.is_empty() && held[0].cost == 0.0);
        // A fixed part uses one contact world from the approach through the departure.
        // The neighboring fixtures and their obstacle numbers persist in every phase.
        let fixed = miru::GripSearch::default();
        let miru::GripOutcome::Complete(draw) = source
            .plan(
                &[*g],
                &fixed,
                &opts,
                || false,
                |c| {
                    let mut task = batchplan::Pull {
                        transit: false,
                        ..c.pull(q, Vec3::Z, 0.01)
                    };
                    task.grasps[0].locked =
                        HashMap::from([("panda_finger_joint1".to_string(), 0.04)]);
                    task
                },
            )
            .unwrap()
        else {
            panic!("a completed search");
        };
        assert_eq!(
            draw.plans[0].locked,
            HashMap::from([("panda_finger_joint1".to_string(), 0.04)]),
            "an exact per-joint command survives independently of the nominal opening"
        );
        let draw = &draw.plans.first().expect("a full draw along z").plan;
        assert!(draw.pull.is_some() && draw.turn.is_none() && draw.held.is_none());
        let miru::GripOutcome::Complete(turn) = source
            .plan(
                &[*g],
                &fixed,
                &opts,
                || false,
                |c| batchplan::Turn {
                    transit: false,
                    ..c.turn(
                        q,
                        Some(batchplan::Screw {
                            point: g.pose.position + Vec3::X * 0.04,
                            direction: Vec3::Z,
                            pitch: 0.0,
                        }),
                        0.01,
                    )
                },
            )
            .unwrap()
        else {
            panic!("a completed search");
        };
        let turned = turn
            .plans
            .first()
            .expect("the fixed part allows a small angle");
        assert!(turned.plan.turn.is_some() && turned.plan.pull.is_none());
        assert_eq!(
            turned.world(batchplan::TaskStep::Turn).world.touch,
            turned.world(batchplan::TaskStep::Depart).world.touch
        );
        let point = surface
            .points
            .iter()
            .zip(&surface.object)
            .find_map(|(p, &own)| (own && g.closes_on(&gripper, *p, 0.0)).then_some(*p))
            .expect("at least one measured point between both pads");
        let search = miru::GripSearch {
            target: Some(miru::GripTarget::Point(point)),
            batch: 1,
            enough: Some(2),
        };
        let mut invalid = *g;
        invalid.pose.position.z = f32::NAN;
        assert!(
            source
                .plan(&[invalid], &search, &opts, || false, configure)
                .is_err(),
            "a point selector must report invalid source geometry before testing membership"
        );
        let queried = std::cell::Cell::new(false);
        let mut polls_after_config = 0;
        let cancelled = || {
            if queried.get() {
                polls_after_config += 1;
            }
            polls_after_config >= 2
        };
        let miru::GripOutcome::Interrupted(partial) = source
            .plan(&candidates, &search, &opts, cancelled, |c| {
                assert_eq!(
                    c.index, 1,
                    "the fingers in air do not close on the asked point"
                );
                queried.set(true);
                configure(c)
            })
            .unwrap()
        else {
            panic!("a cancellation after the first batch must remain explicit");
        };
        assert_eq!(partial.plans.len(), 1);
        assert!(
            partial.failures.is_empty(),
            "the point selector excluded the displaced grasp"
        );
        let held = plan.held.as_ref().expect("the pick holds the box");
        assert_eq!(held.object, body.shape.placed(Pose::IDENTITY));
        let after = seen.picked(&split, held).unwrap();
        assert!(!seen.kept.is_current(&after));
        assert_eq!(after.held(0).expect("a hold").0, id);
        let next_world = after.world(&device, &Step::default()).unwrap();
        let fixture_parts = split
            .scene
            .of
            .iter()
            .filter(|of| **of == Of::Known(0))
            .count();
        assert!(
            fixture_parts > 1,
            "the box and table had a distinct fixture-only patch"
        );
        assert_eq!(
            next_world
                .of
                .iter()
                .filter(|of| **of == Of::Known(0))
                .count(),
            fixture_parts,
            "the extra fixture cells keep their identity for the next held-only support contact"
        );
        assert!(
            !next_world.of.contains(&Of::Unexplained),
            "pure table evidence must not leave an unattributed obstacle under the lifted box"
        );
        let empty = rendered.clone().map(|mut f| {
            f.depth.fill(0.0);
            f
        });
        let cameras = cameras(&empty, [0; 2], 0.0);
        let blank = observe(
            &device,
            &cameras,
            &Stance {
                q,
                locked: &HashMap::new(),
                padding: 0.0,
            },
            &after,
        )
        .unwrap();
        let next_world = blank.memory.world(&device, &Step::default()).unwrap();
        assert_eq!(
            next_world
                .of
                .iter()
                .filter(|of| **of == Of::Known(0))
                .count(),
            fixture_parts,
            "a blind camera cannot remove fixture evidence"
        );
        let mut wrong = held.clone();
        wrong.grasp.position.x += 0.01;
        assert!(
            seen.picked(&split, &wrong).is_err(),
            "a different grip cannot transfer the body"
        );
        let other = first_look(&device, &things, &rendered, (0, 0), q, 0.0);
        assert!(
            other.picked(&split, held).is_err(),
            "identical pixels from a new look are a different world"
        );
        let mut stale = seen.clone();
        stale.memory = after;
        assert!(
            stale
                .grip_worlds(&device, &Step::default(), &gripper, id)
                .is_err(),
            "an act invalidates the old picture"
        );
        let mut fixed = split.scene.world.clone();
        fixed
            .touch
            .push(batchplan::Touch::new(split.part, robot.finger_links(0)));
        let fixed = device.upload(&[fixed]).unwrap();
        let movement = batchplan::ToolPath {
            world: 0,
            start: plan.approach.end(),
            hand: 0,
            waypoints: vec![Pose::new(g.pose.position + Vec3::Z * 0.01, g.pose.rotation)],
            hold: vec![],
            holding: None,
        };
        let followed = batchplan::follow(
            &device,
            &fixed,
            &[movement],
            &batchplan::ToolPathOptions::default(),
        )
        .unwrap();
        assert_eq!(
            followed.fraction,
            [1.0],
            "the fixed part permits the straight draw"
        );
    }
}

/// A bar on a round dial over a panel, as the knob in the truth table. Select a grip whose
/// measured contact is on the protruding bar, and check the panel and the other end of the
/// dial independently of which objects the camera made of them.
#[test]
fn gripping_the_knob_leaves_the_panel_and_dial_in_collision() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let gripper = panda_gripper(&robot);
    let q = robot.default_q();
    let bar = MIDDLE + Vec3::Z * 0.089;
    let half = Vec3::new(0.0125, 0.045, 0.019);
    let things = vec![
        table(),
        block(
            MIDDLE + Vec3::Z * 0.03,
            Vec3::new(0.10, 0.10, 0.03),
            [200; 3],
        ),
        can(MIDDLE + Vec3::Z * 0.065, 0.045, 0.005, [60; 3]),
        block(bar, half, [20; 3]),
        arm(&robot, q),
    ];
    let rendered = [-0.27, 0.27].map(|side| {
        Frame::of(
            &things,
            looking(Vec3::new(MIDDLE.x + side, -0.04, 0.25), bar),
        )
    });
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 10 * c as u64), q, 0.0);
        let id = id_of(seen.kept.scene(), &rendered, 3).expect("the bar is visible");
        let surface = seen.kept.surface(&device, id).unwrap().unwrap();
        let options = SurfaceGraspOptions {
            turns: 8,
            shell: Some(0.015),
            ..Default::default()
        };
        let gs = batchplan::grasps_on(&device, &gripper, &[surface], &options)
            .unwrap()
            .remove(0);
        let prepared = seen
            .grip_worlds(&device, &Step::default(), &gripper, id)
            .unwrap();
        let grips = gs.iter().filter(|g| {
            g.contacts
                .iter()
                .any(|p| (*p - bar).abs().cmple(half + options.contact).all() && p.z > bar.z)
        });
        let split = grips
            .filter_map(|g| prepared.project(g).unwrap())
            .next()
            .unwrap_or_else(|| panic!("{camera}: no measured contact on the upper bar"));
        for p in [
            MIDDLE + Vec3::new(0.09, 0.0, 0.06),
            MIDDLE + Vec3::new(0.035, 0.0, 0.07),
        ] {
            assert!(
                split.scene.world.obstacles[split.part].distance(p).value > 0.0,
                "{camera}: the bar's contact patch includes the panel or dial at {p:?}"
            );
            assert!(
                split
                    .scene
                    .world
                    .obstacles
                    .iter()
                    .enumerate()
                    .any(|(i, o)| i != split.part && o.distance(p).value <= 0.0),
                "{camera}: no obstacle covers the panel or dial at {p:?}"
            );
        }
    }
}

/// Two separate physical objects: lifting the lower box would carry the can standing on it.
#[test]
fn a_supported_can_refuses_the_entire_pick_before_planning_candidates() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 1).unwrap();
    let gripper = panda_gripper(&robot);
    let q = robot.default_q();
    let mut things = vec![
        table(),
        block(
            MIDDLE + Vec3::Z * 0.035,
            Vec3::new(0.021, 0.025, 0.035),
            [210, 30, 30],
        ),
        can(MIDDLE + Vec3::Z * 0.09, 0.015, 0.02, [20; 3]),
        arm(&robot, q),
    ];
    let render = [-0.28, 0.28].map(|side| {
        Frame::of(
            &things,
            looking(Vec3::new(MIDDLE.x, side, 0.30), MIDDLE + Vec3::Z * 0.04),
        )
    });
    let seen = first_look(&device, &things, &render, (0, 0), q, 0.0);
    let base = id_of(seen.kept.scene(), &render, 1).expect("the lower box");
    let can = id_of(seen.kept.scene(), &render, 2).expect("the upper can");
    assert_ne!(base, can);
    let options = SurfaceGraspOptions {
        turns: 1,
        shell: Some(0.015),
        ..Default::default()
    };
    // Keep the same candidate before and after the can is set down on the box.
    things.remove(2);
    let bare_render = render
        .each_ref()
        .map(|view| Frame::of(&things, view.camera));
    let empty = first_look(&device, &things, &bare_render, (0, 0), q, 0.0);
    let box_id = id_of(empty.kept.scene(), &bare_render, 1).expect("the bare box");
    let surface = empty.kept.surface(&device, box_id).unwrap().unwrap();
    let options = SurfaceGraspOptions {
        turns: 1,
        ..options
    };
    let g = batchplan::grasps_on(&device, &gripper, &[surface], &options)
        .unwrap()
        .remove(0)
        .into_iter()
        .next()
        .expect("the bare box has a grip");
    let source = seen
        .grip_worlds(&device, &Step::default(), &gripper, base)
        .unwrap();
    let inputs = [g; 5];
    let run = |grasps: &[batchplan::Grasp], search: &miru::GripSearch, stop: bool| {
        source.plan(
            grasps,
            search,
            &batchplan::TaskOptions::default(),
            || stop,
            |_| -> batchplan::Pick { panic!("a blocked pick must never request a motion") },
        )
    };
    let search = miru::GripSearch::default();
    let miru::GripOutcome::Complete(result) = run(&inputs, &search, false).unwrap() else {
        panic!("the search was not interrupted");
    };
    let blocked = result.blocked.expect("one reason for the entire pick");
    assert_eq!(blocked.object, base);
    assert_eq!(blocked.supports, [can]);
    assert!(result.plans.is_empty() && result.failures.is_empty());
    assert_eq!(result.batches, 0);
    assert!(matches!(
        run(&inputs, &search, true).unwrap(),
        miru::GripOutcome::Interrupted(_)
    ));
    let elsewhere = miru::GripSearch {
        target: Some(miru::GripTarget::Near {
            point: MIDDLE + Vec3::Z,
            radius: 0.0,
        }),
        ..search
    };
    let miru::GripOutcome::Complete(empty) = run(&inputs, &elsewhere, false).unwrap() else {
        panic!("the search was not interrupted");
    };
    assert!(
        empty.blocked.is_none(),
        "no candidate was eligible for the target"
    );
    let mut bad = inputs;
    bad[4].opening = f32::NAN;
    assert!(
        run(&bad, &search, false).is_err(),
        "a refusal cannot hide invalid input"
    );
}

#[test]
fn a_world_point_names_only_readings_at_its_depth_and_keeps_disagreement() {
    let robot = panda();
    let device = Device::cpu(&robot).unwrap();
    let point = Vec3::new(0.45, 0.40, 0.20);
    let upper = point + Vec3::Z * 0.02;
    let known = [point + Vec3::Z * 0.001, upper - Vec3::Z * 0.001]
        .map(|p| Obstacle::cuboid(p, Vec3::new(0.1, 0.1, 0.001)));
    let memory = Memory::try_new(&device, known.to_vec(), &MemoryOptions::default()).unwrap();
    let depths = [0.5; 17 * 17];
    let camera = |at, rotation| Camera {
        image: DepthImage {
            width: 17,
            depth: &depths,
            camera: Pose::new(at, rotation),
            intrinsics: batchplan::Intrinsics {
                fx: 100.0,
                fy: 100.0,
                cx: 8.0,
                cy: 8.0,
            },
        },
        noise: Noise {
            constant: 0.002,
            linear: 0.0,
            quadratic: 0.0,
        },
        off_by: 0.006,
        color: &[],
        masks: &[],
        band: 0,
    };
    let below = camera(point - 0.5 * Vec3::Z, Quat::IDENTITY);
    let above = camera(
        upper + 0.5 * Vec3::Z,
        Quat::from_rotation_x(std::f32::consts::PI),
    );
    let stance = Stance {
        q: robot.default_q(),
        locked: &HashMap::new(),
        padding: 0.0,
    };
    let observed = observe(&device, &[below, above, below], &stance, &memory).unwrap();
    let kept = observed.kept;
    assert_eq!(kept.scene().labels[0][8 * 17 + 8], Label::Known(0));
    assert_eq!(kept.scene().labels[1][8 * 17 + 8], Label::Known(1));
    assert_eq!(kept.labels_at(point).unwrap(), [Label::Known(0)]);
    assert_eq!(kept.labels_at(upper).unwrap(), [Label::Known(1)]);
    assert_eq!(
        kept.labels_at(point.midpoint(upper)).unwrap(),
        [Label::Known(0), Label::Known(1)]
    );
    assert!(kept.labels_at(upper + Vec3::Z * 0.03).unwrap().is_empty());
    assert!(kept.labels_at(point + Vec3::X).unwrap().is_empty());
    assert!(kept.labels_at(Vec3::NAN).is_err());

    let unusable = [1e-22; 17 * 17];
    let camera = Camera {
        image: DepthImage {
            depth: &unusable,
            ..below.image
        },
        ..below
    };
    let observed = observe(&device, &[camera], &stance, &memory).unwrap();
    assert_eq!(
        observed.kept.image(0).unwrap().depth[8 * 17 + 8],
        unusable[0]
    );
    assert_eq!(observed.kept.scene().labels[0][8 * 17 + 8], Label::Nothing);
    let near_camera = below.image.camera.position + 0.002 * Vec3::Z;
    assert!(observed.kept.labels_at(near_camera).unwrap().is_empty());
}

/// A turned tray whose fitted world heights are reconstructed independently still gives a lift.
#[test]
fn a_block_resting_flush_in_a_turned_tray_has_an_extraction() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).expect("a CPU device");
    for raised in [0.0, 0.75] {
        assert_flush_tray_extraction(&device, raised);
    }
}

fn assert_flush_tray_extraction(device: &Device, raised: f32) {
    let at = Vec3::new(0.5, 0.0, 0.0);
    let raised = Vec3::Z * raised;
    let yaw = Quat::from_rotation_z(0.65);
    let mut standing = thin_tray();
    let target = standing.len() + 1;
    standing.push(block(
        at + Vec3::new(-0.028, 0.014, 0.032),
        Vec3::new(0.023, 0.018, 0.022),
        [130; 3],
    ));
    let mut things = vec![block(
        at + raised - Vec3::Z * 0.01,
        Vec3::new(0.5, 0.5, 0.01),
        [150; 3],
    )];
    let memory = Memory::try_new(device, vec![known(&things[0])], &MemoryOptions::default())
        .expect("a memory with the raised table");
    for thing in standing {
        things.push(Thing {
            color: thing.color,
            solid: None,
            distance: Box::new(move |p| (thing.distance)(at + yaw.inverse() * (p - at - raised))),
        });
    }
    let rendered = [EYE, Vec3::new(0.52, -0.055, 0.55)].map(|eye| {
        Frame::of(&things, looking(eye + raised, MIDDLE + raised))
            .masked(|i| if i == target { 2 } else { u32::from(i > 0) })
    });
    let stance = Stance {
        q: device.robot().default_q(),
        locked: &HashMap::new(),
        padding: 0.0,
    };
    let obs =
        observe(device, &cameras(&rendered, [0; 2], 0.0), &stance, &memory).expect("a sound look");
    let scene = obs.kept.scene();
    let (index, root) = scene
        .objects
        .iter()
        .enumerate()
        .find(|(_, o)| matches!(o.shape, Shape::OpenBox { .. }))
        .expect("the raised tray opening is seen");
    let child = scene
        .objects
        .iter()
        .find(|o| o.on == Some(Support::Object(index)))
        .expect("the block is seen standing on the tray");
    let lifts = obs
        .kept
        .extractions(child.id, 0.005)
        .expect("a sound margin");
    assert_eq!(
        lifts.len(),
        1,
        "the block is flush with the measured floor at translation {raised}"
    );
    assert_eq!(lifts[0].container, root.id);
    assert!(
        lifts[0].lift.distance(Vec3::Z * 0.055) < 0.0001,
        "the 50 mm room's depth plus the 5 mm margin"
    );
}

/// A second camera hits an occupied bowl voxel although its bead return ends in the next voxel.
#[test]
fn a_foreign_projective_hit_on_the_opposite_rim_stays_in_the_lift_world() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).expect("a CPU device");
    let gripper = panda_gripper(&robot);
    let q = robot.default_q();
    let cell = Vec3::new(0.52242327, -0.13757662, 0.08242336);
    let endpoint = Vec3::new(0.5235493, -0.14448546, cell.z);
    let center = endpoint - 0.001 * Vec3::Z;
    let bead = Obstacle::sphere(center, 0.001);
    let things = [
        table(),
        wide_bowl(MIDDLE),
        Thing {
            color: [130, 80, 112],
            solid: Some(bead.clone()),
            distance: Box::new(move |p| (p - center).length() - 0.001),
        },
        arm(&robot, q),
    ];
    let views = render(&things);
    let mut cams = cameras(&views, [0; 2], 0.0);
    let k = batchplan::Intrinsics {
        fx: 13.0,
        fy: 13.0,
        cx: 1.0,
        cy: 1.0,
    };
    let pose = Pose::new(
        endpoint + 0.25 * Vec3::Z,
        Quat::from_rotation_x(std::f32::consts::PI),
    );
    let mut sparse = Frame::through(&things, pose, k, (3, 3));
    assert_eq!(
        sparse.shows[4],
        Some(2),
        "the central pixel returns from the known bead"
    );
    for (i, z) in sparse.depth.iter_mut().enumerate() {
        if i != 4 {
            *z = 0.0;
        }
    }
    cams.push(Camera {
        image: sparse.view().image,
        color: &[],
        noise: common::NOISE,
        band: 0,
        off_by: 0.0,
        masks: &[],
    });
    let mem = Memory::try_new(
        &device,
        vec![known(&table()), bead],
        &MemoryOptions::default(),
    )
    .unwrap();
    let locked = HashMap::new();
    let seen = observe(
        &device,
        &cams,
        &Stance {
            q,
            locked: &locked,
            padding: 0.0,
        },
        &mem,
    )
    .unwrap();
    assert_eq!(seen.kept.scene().labels[2][4], Label::Known(1));
    let id = id_of(seen.kept.scene(), &views, 1).expect("the bowl is seen");
    let options = SurfaceGraspOptions {
        turns: 1,
        shell: Some(0.015),
        ..Default::default()
    };
    let surface = seen.kept.surface(&device, id).unwrap().unwrap();
    let choices = batchplan::grasps_on(&device, &gripper, &[surface], &options)
        .unwrap()
        .remove(0);
    let grasp = choices
        .iter()
        .max_by(|a, b| {
            a.pose
                .position
                .distance(cell)
                .total_cmp(&b.pose.position.distance(cell))
        })
        .unwrap();
    assert!(
        grasp.pose.position.distance(cell) > 0.20,
        "the hit is outside the refined hand window"
    );
    let prepared = seen
        .grip_worlds(&device, &Step::default(), &gripper, id)
        .unwrap();
    let split = prepared
        .project(grasp)
        .unwrap()
        .expect("a measured rim surface");
    let body = split
        .body
        .as_ref()
        .expect("the bowl supports no other object");
    let retained = split
        .scene
        .world
        .obstacles
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != split.part && !body.other_parts.contains(k))
        .map(|(_, obstacle)| obstacle.distance(cell).value)
        .fold(f32::INFINITY, f32::min);
    assert!(
        retained <= 0.0,
        "the bowl must leave the mixed coarse cell occupied on lift, got {retained} m"
    );
}

/// A block stands in a bowl. Two views see the block and the bowl but miss a patch on top
/// of the block. A third view can contribute one known bead in that patch.
#[test]
fn a_grip_resolves_its_support_without_copying_the_block_or_losing_a_foreign_hit() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 4).unwrap();
    let gripper = panda_gripper(&robot);
    let q = robot.default_q();
    let center = MIDDLE + Vec3::new(0.024, 0.004, 0.033);
    let bead_center = MIDDLE + Vec3::new(0.035, 0.005, 0.06255);
    let radius = 0.00055;
    let bead = Obstacle::sphere(bead_center, radius);
    let witness = bead_center + Vec3::Z * radius;
    let things = [
        table(),
        wide_bowl(MIDDLE),
        block(center, Vec3::new(0.021, 0.025, 0.029), [100, 10, 50]),
        Thing {
            color: [20, 150, 10],
            solid: Some(bead.clone()),
            distance: Box::new(move |p| p.distance(bead_center) - radius),
        },
        arm(&robot, q),
    ];
    let mut frames = [-0.28, 0.28].map(|side| {
        let camera = looking(
            MIDDLE + Vec3::new(0.0, side, 0.30),
            center + Vec3::Z * 0.007,
        );
        Frame::of(&things, camera).masked(|i| match i {
            1 | 2 => i as u32,
            _ => 0,
        })
    });
    for frame in &mut frames {
        let points = frame.view().image.points(None).unwrap();
        for (depth, point) in frame.depth.iter_mut().zip(points) {
            if point.is_some_and(|p| p.distance(witness) < 0.011) {
                *depth = 0.0;
            }
        }
    }
    let intrinsics = batchplan::Intrinsics {
        fx: 110.0,
        fy: 110.0,
        cx: 1.0,
        cy: 1.0,
    };
    let down = Pose::new(
        witness + Vec3::Z * 0.20,
        Quat::from_rotation_x(std::f32::consts::PI),
    );
    let mut third = Frame::through(&things, down, intrinsics, (3, 3));
    assert_eq!(third.shows[4], Some(3), "the center ray reaches the bead");
    let hit = third.depth[4];
    third.depth.fill(0.0);
    third.depth[4] = hit;

    let memory = Memory::try_new(
        &device,
        vec![known(&table()), bead],
        &MemoryOptions::default(),
    )
    .unwrap();
    let locks = HashMap::new();
    let stance = Stance {
        q,
        locked: &locks,
        padding: 0.0,
    };
    for sees_bead in [false, true] {
        let mut views = cameras(&frames, [0; 2], 0.0);
        if sees_bead {
            views.push(Camera {
                image: third.view().image,
                color: &[],
                noise: common::NOISE,
                band: 0,
                off_by: 0.0,
                masks: &[],
            });
        }
        let seen = observe(&device, &views, &stance, &memory).unwrap();
        let scene = seen.kept.scene();
        if sees_bead {
            assert_eq!(
                scene.labels[2][4],
                Label::Known(1),
                "this hit belongs to the bead"
            );
        }
        let id = id_of(scene, &frames, 2).expect("the two views see the block");
        let target = scene.objects.iter().find(|o| o.id == id).unwrap();
        let Some(Support::Object(parent)) = target.on else {
            panic!("the block rests in the bowl")
        };
        let bowl = &scene.objects[parent];
        assert!(target.solid && !bowl.solid);
        let options = SurfaceGraspOptions {
            turns: 1,
            ..Default::default()
        };
        let surface = seen.kept.surface(&device, id).unwrap().unwrap();
        let grasps = batchplan::grasps_on(&device, &gripper, &[surface], &options)
            .unwrap()
            .remove(0);
        let source = seen
            .grip_worlds(&device, &Step::default(), &gripper, id)
            .unwrap();
        let split = grasps
            .iter()
            .find_map(|g| source.project(g).unwrap())
            .expect("the block has a measured grip");
        let distance = |world: &Planned, at| {
            world
                .of
                .iter()
                .zip(&world.world.obstacles)
                .filter(|(of, _)| matches!(**of, Of::Object(id) | Of::Region(id) if id == bowl.id))
                .map(|(_, obstacle)| obstacle.distance(at).value)
                .fold(f32::INFINITY, f32::min)
        };
        let ordinary = seen
            .memory
            .world(
                &device,
                &Step {
                    into: &[bowl.id],
                    ..Default::default()
                },
            )
            .unwrap();
        let on_block = MIDDLE + Vec3::new(0.035, 0.020, 0.062);
        assert!(
            distance(&ordinary, on_block) > 0.01,
            "the map excludes the block"
        );
        assert!(
            distance(&split.scene, on_block) > 0.0,
            "the block must stay out of the stationary support"
        );
        assert!(split.scene.world.touch.is_empty());
        assert_eq!(
            distance(&split.scene, witness) <= 0.0,
            sees_bead,
            "only the third view contributes the foreign hit"
        );
        assert!(
            distance(&split.scene, MIDDLE + Vec3::new(0.0, -0.134, 0.070)) <= 0.0,
            "the far rim stays in collision"
        );
    }
}

#[test]
fn proposed_jaw_closures_distinguish_target_foreign_and_missing_readings() {
    use miru::ClosureEvidence::{BothSides, Incomplete, Obstructed};

    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let gripper = panda_gripper(&robot);
    let q = robot.default_q().to_vec();
    let middle = Vec3::new(0.5, 0.0, 0.05);
    let bar = || block(middle, Vec3::new(0.022, 0.009, 0.025), [180, 30, 30]);
    let things = [table(), bar(), arm(&robot, &q)];
    let views = |things: &[Thing]| {
        [-0.35, 0.35].map(|y| Frame::of(things, looking(middle + Vec3::new(0.08, y, 0.2), middle)))
    };
    let rendered = views(&things);
    let pin = block(
        middle + Vec3::Y * 0.026,
        Vec3::new(0.010, 0.002, 0.010),
        [30, 180, 30],
    );
    let pin_obstacle = known(&pin);
    let guard = views(&[table(), bar(), pin, arm(&robot, &q)]);
    let fixed = Memory::try_new(
        &device,
        vec![known(&things[0]), pin_obstacle],
        &MemoryOptions::default(),
    )
    .unwrap();
    let hand = robot.hand_spheres(0);
    let tip = hand
        .iter()
        .filter(|s| s.finger)
        .max_by(|a, b| a.center.z.total_cmp(&b.center.z))
        .unwrap();
    let down = Quat::from_rotation_x(std::f32::consts::PI);
    let at = middle - down * Vec3::new(tip.center.x, 0.0, tip.center.z);
    let poses = [
        Pose::new(at, down),
        Pose::new(at + Vec3::X * 0.10, down),
        Pose::new(at - Vec3::Z * middle.z, down),
    ];
    let options = SurfaceGraspOptions {
        approach: 0.06,
        ..Default::default()
    };
    for (c, (name, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 31), &q, 0.0);
        let id = id_of(seen.kept.scene(), &rendered, 1).expect("the rendered bar is visible");
        let closures = |id, poses: &[Pose]| {
            seen.kept
                .observed_closures(&device, id, &gripper, poses, &options)
        };
        assert_eq!(
            closures(id, &poses).unwrap().unwrap(),
            [BothSides, Incomplete, Obstructed],
            "{name}"
        );
        let (frames, bands) = read(&guard, c, 31);
        let locked = HashMap::new();
        let stance = Stance {
            q: &q,
            locked: &locked,
            padding: 0.0,
        };
        let with_guard = observe(&device, &cameras(&frames, bands, 0.0), &stance, &fixed).unwrap();
        let kept = &with_guard.kept;
        let id = id_of(kept.scene(), &guard, 1).expect("the bar remains visible beside the pin");
        assert!(
            kept.scene()
                .labels
                .iter()
                .flatten()
                .any(|label| *label == Label::Known(1)),
            "the foreign pin is visible on {name}"
        );
        assert_eq!(
            kept.observed_closures(&device, id, &gripper, &poses[..1], &options)
                .unwrap()
                .unwrap(),
            [Obstructed],
            "the moving finger sweeps into the stationary pin on {name}"
        );
        assert!(closures(u32::MAX, &poses).unwrap().is_none());
        assert!(closures(id, &[Pose::new(Vec3::NAN, down)]).is_err());
    }
}

/// An approach longer than the scene gives what one 10 m long does, however long: 1e10 m, 1e30 m
/// or the largest a float holds. Over the box, the hand turned down closes on it and the hand
/// turned up comes in through it from below, which a 10 cm approach does not reach.
#[test]
fn a_far_approach_gives_what_one_past_every_reading_gives() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let gripper = turn_gripper(&robot);
    let q = robot.default_q().to_vec();
    let things = vec![table(), a_control(), arm(&robot, &q)];
    let rendered = one_side(&things);
    let seen = first_look(&device, &things, &rendered, (0, 7), &q, 0.0);
    let id = id_of(seen.kept.scene(), &rendered, 1).expect("the box is visible");
    let down = Quat::from_rotation_x(std::f32::consts::PI);
    let poses = [
        Pose::new(Vec3::new(0.5, 0.0, 0.025), down),
        Pose::new(Vec3::new(0.5, 0.0, 0.5), Quat::IDENTITY),
    ];
    let evidence = |approach| {
        let options = SurfaceGraspOptions {
            approach,
            ..Default::default()
        };
        let found = seen
            .kept
            .observed_closures(&device, id, &gripper, &poses, &options);
        found.unwrap().expect("the box is seen")
    };
    let past = evidence(10.0);
    assert_eq!(
        past,
        [ClosureEvidence::BothSides, ClosureEvidence::Obstructed],
        "the far approach reaches the box from below"
    );
    assert_ne!(evidence(0.1), past, "a short approach does not");
    for approach in [1e30, 1e10, f32::MAX] {
        assert_eq!(evidence(approach), past, "approach {approach}");
    }
}

/// The calibrated widths and the graspable width the turn tests state for the Panda's hand: its
/// pads 8 cm apart as loaded, closing to nothing, and a hand said to grasp up to 7.5 cm, so that
/// the loaded width, at which the hand comes in fully open, and the capacity differ.
const TURN_GAPS: std::ops::RangeInclusive<f32> = 0.0..=0.08;
const TURN_CAPACITY: f32 = 0.075;

/// The Panda's hand as the turn tests calibrate it: [`TURN_GAPS`] and [`TURN_CAPACITY`].
fn turn_gripper(robot: &Robot) -> Gripper {
    let calibration = GripperCalibration {
        capacity: Some(TURN_CAPACITY),
        ..GripperCalibration::new(*TURN_GAPS.end(), *TURN_GAPS.start(), panda_closed())
    };
    robot.gripper(0, &calibration).expect("the Panda's hand")
}

/// The options the turn tests ask grips with, as batchplan-orchestrator asks a turn's: the hand
/// frame 5 mm below a crest, one direction about each closing line, fingers that may come in
/// narrower and walls closed across from one side.
fn turn_options() -> (ShapeGraspOptions, SurfaceGraspOptions) {
    let shape = ShapeGraspOptions {
        depth: 0.005,
        turns: 1,
        ..Default::default()
    };
    let surface = SurfaceGraspOptions {
        turns: 1,
        approach: 0.06,
        shell: Some(TURN_CAPACITY),
        narrow: true,
        ..Default::default()
    };
    (shape, surface)
}

/// One grip as the turn tests compare them: its pose, opening, source and evidence.
type Grip = (Pose, f32, GripSource, ClosureEvidence);

/// A group as the turn tests compare them: its grips and its `enough`.
type Group = (Vec<Grip>, Option<usize>);

/// `turn`'s groups as the turn tests compare them.
fn grips_of(turn: &TurnGrips) -> Vec<Group> {
    let group = |g: &TurnGroup| {
        let grips = g
            .grips
            .iter()
            .map(|g| (g.pose, g.opening, g.source, g.evidence));
        (grips.collect(), g.enough)
    };
    turn.groups.iter().map(group).collect()
}

/// What `Kept::turn_grips` is made of for hand 0 turning about +z, by the calls it composes: the
/// screw's point, the groups as [`grips_of`] gives them, and whether the grasps of the surface
/// moved when those closing on both sides were put first.
fn composed(
    kept: &Kept,
    device: &Device,
    id: u32,
    (shape_options, surface_options): (&ShapeGraspOptions, &SurfaceGraspOptions),
) -> (Vec3, Vec<Group>, bool) {
    use ClosureEvidence::{BothSides, Incomplete};
    let gripper = turn_gripper(device.robot());
    let evidence = |poses: &[Pose]| {
        let found = kept.observed_closures(device, id, &gripper, poses, surface_options);
        found.unwrap().expect("the object is seen")
    };
    let object = (kept.scene().objects.iter()).find(|o| o.id == id);
    let shape = cover(&object.expect("the object").shape);
    let placed = shape.pose();
    let depth = shape_options.depth;
    let bar = kept.bar_on_disk(device, id, Vec3::Z, &gripper, depth, surface_options);
    let bar = bar.unwrap();
    let screw = Screw::about(
        bar.as_ref().map_or(placed.position, |bar| bar.center),
        Vec3::Z,
    );
    let open = gripper.opening();
    let mut groups = vec![];
    if let Some(bar) = &bar {
        let grips = (bar.grasps.iter().zip(evidence(&bar.grasps)))
            .map(|(&pose, e)| (pose, open, GripSource::Bar, e));
        groups.push((grips.collect(), Some(1)));
    }
    let from_above: Vec<Pose> = (batchplan::grasps(&shape, &gripper, shape_options))
        .unwrap_or_default()
        .into_iter()
        .map(|grasp| placed.mul_pose(grasp))
        .filter(|pose| (pose.rotation * Vec3::Z).z < -0.99)
        .collect();
    let shaped: Vec<Grip> = (from_above.iter().zip(evidence(&from_above)))
        .map(|(&pose, e)| (pose, open, GripSource::Shape, e))
        .collect();
    let centred: Vec<Grip> = shaped
        .iter()
        .copied()
        .filter(|g| g.3 == BothSides)
        .collect();
    if !centred.is_empty() {
        groups.push((centred, None));
    }
    let surface = kept
        .surface(device, id)
        .unwrap()
        .expect("the object is seen");
    let found = batchplan::grasps_on(device, &gripper, &[surface], surface_options);
    let mut seen = found.unwrap().remove(0);
    surface_options.order_for_turn(&gripper, &mut seen, Some(screw));
    let poses: Vec<Pose> = seen.iter().map(|grasp| grasp.pose).collect();
    let surfaces: Vec<Grip> = (seen.iter().zip(evidence(&poses)))
        .map(|(grasp, e)| (grasp.pose, grasp.opening, GripSource::Surface, e))
        .collect();
    let mut rest: Vec<Grip> = surfaces
        .iter()
        .copied()
        .filter(|g| g.3 == BothSides)
        .collect();
    rest.extend(surfaces.iter().copied().filter(|g| g.3 != BothSides));
    let moved = rest[..] != surfaces[..];
    rest.extend(shaped.iter().copied().filter(|g| g.3 == Incomplete));
    if !rest.is_empty() {
        groups.push((rest, Some(1)));
    }
    (screw.point, groups, moved)
}

/// A bar standing on a dial over a panel, as in `gripping_the_knob_leaves_the_panel_and_dial_in_
/// collision`, and the arm at `q`, the last thing; the bar is thing 3.
fn a_knob(robot: &Robot, q: &[f32]) -> Vec<Thing> {
    vec![
        table(),
        block(
            MIDDLE + Vec3::Z * 0.03,
            Vec3::new(0.10, 0.10, 0.03),
            [200; 3],
        ),
        can(MIDDLE + Vec3::Z * 0.065, 0.045, 0.005, [60; 3]),
        block(
            MIDDLE + Vec3::Z * 0.089,
            Vec3::new(0.0125, 0.045, 0.019),
            [20; 3],
        ),
        arm(robot, q),
    ]
}

/// A knob is turned by its bar first: the grips across the bar over the dial, about the dial's
/// centre; then those from above the bar's fitted box that close on both its sides, which
/// batchplan plans alike; then the grasps of its surface, those closing on both sides first. On
/// every camera the groups are what the calls they are made of give, in that order.
#[test]
fn a_knob_is_turned_by_its_bar_first() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let q = robot.default_q().to_vec();
    let things = a_knob(&robot, &q);
    let bar = MIDDLE + Vec3::Z * 0.089;
    let rendered = [-0.27, 0.27].map(|side| {
        Frame::of(
            &things,
            looking(Vec3::new(MIDDLE.x + side, -0.04, 0.25), bar),
        )
    });
    let (shape_options, surface_options) = turn_options();
    let gripper = turn_gripper(&robot);
    assert_ne!(
        gripper.capacity(),
        gripper.opening(),
        "the capacity must differ from the loaded width to tell them apart"
    );
    let mut moved = false;
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 7), &q, 0.0);
        let id = id_of(seen.kept.scene(), &rendered, 3).expect("the bar is visible");
        let turn = seen.kept.turn_grips(
            &device,
            id,
            &gripper,
            Vec3::Z,
            &shape_options,
            &surface_options,
        );
        let turn = turn.unwrap().expect("the bar is seen");
        let (point, groups, reordered) =
            composed(&seen.kept, &device, id, (&shape_options, &surface_options));
        assert_eq!(turn.screw, Screw::about(point, Vec3::Z), "{camera}");
        assert_eq!(grips_of(&turn), groups, "{camera}");
        moved |= reordered;
        let mut wrong = vec![];
        match seen
            .kept
            .bar_on_disk(
                &device,
                id,
                Vec3::Z,
                &gripper,
                shape_options.depth,
                &surface_options,
            )
            .unwrap()
        {
            Some(disk) => {
                let first = &turn.groups[0];
                let poses: Vec<Pose> = first.grips.iter().map(|grip| grip.pose).collect();
                assert_eq!(poses, disk.grasps, "{camera}: the bar's grips come first");
                assert!(first.grips.iter().all(|g| g.opening == gripper.opening()));
                assert_eq!(first.enough, Some(1));
                assert_eq!(turn.screw.point, disk.center, "{camera}: about the dial");
            }
            None => wrong.push("no bar over the disk resolves".to_string()),
        }
        found.push((*camera, outcome(&wrong)));
    }
    assert!(
        moved,
        "some camera's grasps must close on both sides after others in the order of their moments"
    );
    verdict("a_knob_is_turned_by_its_bar_first", &found);
}

/// A box 86 x 24 x 30 mm, the stove's control of batchplan-orchestrator's failed turn, standing
/// on the table: the hand closes across its narrow width only.
fn a_control() -> Thing {
    block(
        Vec3::new(0.5, 0.0, 0.015),
        Vec3::new(0.043, 0.012, 0.015),
        [180, 30, 30],
    )
}

/// The control and the arm at `q`, seen from its long side at -y and its ends, never its far side.
fn one_side(things: &[Thing]) -> [Frame; 2] {
    let middle = Vec3::new(0.5, 0.0, 0.015);
    [0.35, -0.35].map(|x| Frame::of(things, looking(middle + Vec3::new(x, -0.3, 0.25), middle)))
}

/// A box seen from one side has no surface grasp of two seen contacts, but the hand from above
/// its middle closes on readings of its top at both edges: those grips are the second group,
/// fully open and planned alike, and the surface's grasps follow in the last.
#[test]
fn a_box_seen_from_one_side_is_turned_from_above_its_middle() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let gripper = turn_gripper(&robot);
    let q = robot.default_q().to_vec();
    let things = vec![table(), a_control(), arm(&robot, &q)];
    let rendered = one_side(&things);
    let (shape_options, surface_options) = turn_options();
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 7), &q, 0.0);
        let id = id_of(seen.kept.scene(), &rendered, 1).expect("the box is visible");
        let surface = seen.kept.surface(&device, id).unwrap().unwrap();
        let grasps = batchplan::grasps_on(&device, &gripper, &[surface], &surface_options);
        let grasps = grasps.unwrap().remove(0);
        assert!(
            !grasps.is_empty() && grasps.iter().all(|grasp| grasp.one_sided),
            "{camera}: the box is seen from one side"
        );
        let turn = seen.kept.turn_grips(
            &device,
            id,
            &gripper,
            Vec3::Z,
            &shape_options,
            &surface_options,
        );
        let turn = turn.unwrap().expect("the box is seen");
        let (point, groups, _) =
            composed(&seen.kept, &device, id, (&shape_options, &surface_options));
        assert_eq!(turn.screw, Screw::about(point, Vec3::Z), "{camera}");
        assert_eq!(grips_of(&turn), groups, "{camera}");
        let last = turn.groups.last().expect("the surface's grasps");
        let surfaces = last
            .grips
            .iter()
            .filter(|g| g.source == GripSource::Surface);
        assert_eq!(
            surfaces.count(),
            grasps.len(),
            "{camera}: every grasp of the surface"
        );
        let mut wrong = vec![];
        match &turn.groups[..] {
            [centred, _] if centred.enough.is_none() => {
                let [a, b] = &centred.grips[..] else {
                    panic!("{camera}: a pair, both ways round: {:?}", centred.grips);
                };
                for grip in [a, b] {
                    let axes = Mat3::from_quat(grip.pose.rotation);
                    assert_eq!(grip.source, GripSource::Shape);
                    assert_eq!(grip.evidence, ClosureEvidence::BothSides);
                    assert_eq!(grip.opening, gripper.opening(), "{camera}: fully open");
                    assert!(axes.z_axis.z < -0.999, "{camera}: from above");
                    assert!(axes.y_axis.y.abs() > 0.999, "{camera}: across its width");
                }
                assert!(
                    a.pose.rotation.dot(b.pose.rotation).abs() < 1e-3,
                    "both ways round"
                );
            }
            groups => wrong.push(format!(
                "no grip from above its middle in {} groups",
                groups.len()
            )),
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict(
        "a_box_seen_from_one_side_is_turned_from_above_its_middle",
        &found,
    );
}

/// The poses of `turn`'s grips from above the fitted shape, in every group.
fn shape_grips(turn: &TurnGrips) -> Vec<Pose> {
    let grips = turn.groups.iter().flat_map(|g| &g.grips);
    grips
        .filter(|g| g.source == GripSource::Shape)
        .map(|g| g.pose)
        .collect()
}

/// A post fixed beside the box, between its long side and where the open finger stands over it:
/// the fingers closing from above the box's middle would sweep into it, so those grips are in no
/// group, while without it they are the second group.
#[test]
fn a_post_in_the_closing_gap_rules_the_grip_out() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let gripper = turn_gripper(&robot);
    let q = robot.default_q().to_vec();
    let post = block(
        Vec3::new(0.5, 0.03, 0.015),
        Vec3::new(0.003, 0.003, 0.015),
        [30, 180, 30],
    );
    let fixed = Memory::try_new(
        &device,
        vec![known(&table()), known(&post)],
        &MemoryOptions::default(),
    )
    .unwrap();
    let bare = vec![table(), a_control(), arm(&robot, &q)];
    let guarded = vec![table(), a_control(), post, arm(&robot, &q)];
    let (shape_options, surface_options) = turn_options();
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let rendered = one_side(&bare);
        let seen = first_look(&device, &bare, &rendered, (c, 7), &q, 0.0);
        let id = id_of(seen.kept.scene(), &rendered, 1).expect("the box is visible");
        let turn = seen.kept.turn_grips(
            &device,
            id,
            &gripper,
            Vec3::Z,
            &shape_options,
            &surface_options,
        );
        let centred = shape_grips(&turn.unwrap().expect("the box is seen"));
        let rendered = one_side(&guarded);
        let (frames, bands) = read(&rendered, c, 7);
        let locked = HashMap::new();
        let stance = Stance {
            q: &q,
            locked: &locked,
            padding: padding(&rendered, &frames, guarded.len() - 1),
        };
        let seen = observe(&device, &cameras(&frames, bands, 0.0), &stance, &fixed).unwrap();
        let kept = &seen.kept;
        assert!(
            kept.scene()
                .labels
                .iter()
                .flatten()
                .any(|label| *label == Label::Known(1)),
            "{camera}: the post is seen"
        );
        let id = id_of(kept.scene(), &rendered, 1).expect("the box is visible beside the post");
        let turn = kept.turn_grips(
            &device,
            id,
            &gripper,
            Vec3::Z,
            &shape_options,
            &surface_options,
        );
        let turn = turn.unwrap().expect("the box is seen");
        let (point, groups, _) = composed(kept, &device, id, (&shape_options, &surface_options));
        assert_eq!(turn.screw, Screw::about(point, Vec3::Z), "{camera}");
        assert_eq!(grips_of(&turn), groups, "{camera}");
        let mut wrong = vec![];
        if centred.is_empty() {
            wrong.push("no grip from above its middle".to_string());
        }
        let ruled_out = kept.observed_closures(&device, id, &gripper, &centred, &surface_options);
        let ruled_out = ruled_out.unwrap().expect("the box is seen");
        assert!(
            ruled_out.iter().all(|e| *e == ClosureEvidence::Obstructed),
            "{camera}: the fingers close into the post: {ruled_out:?}"
        );
        let kept_grips = shape_grips(&turn);
        assert!(
            centred.iter().all(|pose| kept_grips
                .iter()
                .all(|kept| kept.position.distance(pose.position) > 0.005)),
            "{camera}: a grip into the post is in a group: {kept_grips:?}"
        );
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_post_in_the_closing_gap_rules_the_grip_out", &found);
}

/// A shelf 6 cm square and 6 mm thick, 17.5 cm over the box: clear of the open hand over the
/// box's middle and of its fingers' sweeps, but in the hand's straight way in from above. The
/// grips from above the box's middle, which close on both its sides without the shelf, come in
/// through it, so they are in no group.
#[test]
fn a_shelf_in_the_way_in_rules_the_grip_out() {
    use ClosureEvidence::{BothSides, Obstructed};

    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let gripper = turn_gripper(&robot);
    let q = robot.default_q().to_vec();
    let shelf = block(
        Vec3::new(0.5, 0.0, 0.175),
        Vec3::new(0.03, 0.03, 0.003),
        [30, 180, 30],
    );
    let bare = vec![table(), a_control(), arm(&robot, &q)];
    let sheltered = vec![table(), a_control(), shelf, arm(&robot, &q)];
    let (shape_options, surface_options) = turn_options();
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let rendered = one_side(&bare);
        let seen = first_look(&device, &bare, &rendered, (c, 7), &q, 0.0);
        let id = id_of(seen.kept.scene(), &rendered, 1).expect("the box is visible");
        let turn = seen.kept.turn_grips(
            &device,
            id,
            &gripper,
            Vec3::Z,
            &shape_options,
            &surface_options,
        );
        let centred = shape_grips(&turn.unwrap().expect("the box is seen"));
        let rendered = one_side(&sheltered);
        let seen = first_look(&device, &sheltered, &rendered, (c, 7), &q, 0.0);
        let kept = &seen.kept;
        assert!(
            holder(kept.scene(), &rendered, 2).is_some(),
            "{camera}: the shelf is seen"
        );
        let id = id_of(kept.scene(), &rendered, 1).expect("the box is visible under the shelf");
        let evidence = |options: &SurfaceGraspOptions| {
            let found = kept.observed_closures(&device, id, &gripper, &centred, options);
            found.unwrap().expect("the box is seen")
        };
        let through = evidence(&surface_options);
        assert!(
            through.iter().all(|e| *e == Obstructed),
            "{camera}: the hand comes in through the shelf: {through:?}"
        );
        let no_way_in = SurfaceGraspOptions {
            approach: 0.0,
            ..surface_options
        };
        let closing = evidence(&no_way_in);
        assert!(
            closing.iter().all(|e| *e == BothSides),
            "{camera}: the shelf is clear of the open hand and its fingers' sweeps: {closing:?}"
        );
        let turn = kept.turn_grips(
            &device,
            id,
            &gripper,
            Vec3::Z,
            &shape_options,
            &surface_options,
        );
        let kept_grips = shape_grips(&turn.unwrap().expect("the box is seen"));
        assert!(
            centred.iter().all(|pose| kept_grips
                .iter()
                .all(|kept| kept.position.distance(pose.position) > 0.005)),
            "{camera}: a grip through the shelf is in a group: {kept_grips:?}"
        );
        let mut wrong = vec![];
        if centred.is_empty() {
            wrong.push("no grip from above its middle".to_string());
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("a_shelf_in_the_way_in_rules_the_grip_out", &found);
}

/// Two boxes seen from one side: one 86 x 77 x 30 mm, its narrow width between the capacity of
/// 7.5 cm and the loaded width of 8 cm, and the control, 86 x 24 x 30 mm. No grip from above
/// either fitted shape closes across more than the capacity, and each grip's opening is the
/// loaded width: fully open.
#[test]
fn grips_from_above_open_as_loaded_and_close_across_no_more_than_the_capacity() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let gripper = turn_gripper(&robot);
    let q = robot.default_q().to_vec();
    let wide = block(
        Vec3::new(0.5, 0.0, 0.015),
        Vec3::new(0.043, 0.0385, 0.015),
        [180, 30, 30],
    );
    let (shape_options, surface_options) = turn_options();
    let (mut between, mut shaped) = (0, 0);
    for thing in [wide, a_control()] {
        let things = vec![table(), thing, arm(&robot, &q)];
        let rendered = one_side(&things);
        for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
            let seen = first_look(&device, &things, &rendered, (c, 7), &q, 0.0);
            let scene = seen.kept.scene();
            let id = id_of(scene, &rendered, 1).expect("the box is visible");
            let object = scene.objects.iter().find(|object| object.id == id);
            let shape = cover(&object.expect("the box").shape);
            let Obstacle::Cuboid {
                half_extents,
                rotation,
                ..
            } = shape
            else {
                panic!("{camera}: fitted as a box: {shape:?}");
            };
            let narrow = 2.0 * half_extents.x.min(half_extents.y);
            between += usize::from(gripper.capacity() < narrow && narrow <= gripper.opening());
            let turn = seen.kept.turn_grips(
                &device,
                id,
                &gripper,
                Vec3::Z,
                &shape_options,
                &surface_options,
            );
            let turn = turn.unwrap().expect("the box is seen");
            let grips = turn.groups.iter().flat_map(|group| &group.grips);
            for grip in grips.filter(|grip| grip.source == GripSource::Shape) {
                let closing = rotation.inverse() * (grip.pose.rotation * Vec3::Y);
                let across = 2.0 * closing.abs().dot(half_extents);
                assert!(
                    across <= gripper.capacity(),
                    "{camera}: a grip from above closes across {across} m, the hand grasps {} m",
                    gripper.capacity()
                );
                assert_eq!(grip.opening, gripper.opening(), "{camera}: fully open");
                shaped += 1;
            }
        }
    }
    assert!(
        between > 0 && shaped > 0,
        "some camera must fit the wide box between the capacity and the loaded width, and see a \
         grip from above a shape: {between} and {shaped}"
    );
}

/// A control 86 x 24 x 40 mm standing on a known panel slanted 20 degrees toward the cameras,
/// turned about the panel's normal. Its shape is fitted upright, so the grips from above it come
/// straight down, 20 degrees off the line it turns about: none of those is proposed.
#[test]
fn no_grip_from_above_is_off_the_axis_by_a_slant() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let gripper = turn_gripper(&robot);
    let q = robot.default_q().to_vec();
    let slant = Quat::from_rotation_x(20f32.to_radians());
    let normal = slant * Vec3::Z;
    let (panel_middle, panel_half) = (Vec3::new(0.5, 0.0, 0.10), Vec3::new(0.15, 0.10, 0.02));
    let control_half = Vec3::new(0.043, 0.012, 0.02);
    let middle = panel_middle + normal * (panel_half.z + control_half.z);
    let things = vec![
        table(),
        common::turned(
            block(panel_middle, panel_half, [200; 3]),
            panel_middle,
            slant,
        ),
        common::turned(block(middle, control_half, [180, 30, 30]), middle, slant),
        arm(&robot, &q),
    ];
    let rendered = [0.35, -0.35]
        .map(|x| Frame::of(&things, looking(middle + Vec3::new(x, -0.3, 0.25), middle)));
    let panel = Obstacle::Cuboid {
        center: panel_middle,
        half_extents: panel_half,
        rotation: slant,
    };
    let fixed = Memory::try_new(
        &device,
        vec![known(&table()), panel],
        &MemoryOptions::default(),
    )
    .unwrap();
    let (shape_options, surface_options) = turn_options();
    let mut straight_down = 0;
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let (frames, bands) = read(&rendered, c, 7);
        let locked = HashMap::new();
        let stance = Stance {
            q: &q,
            locked: &locked,
            padding: padding(&rendered, &frames, things.len() - 1),
        };
        let seen = observe(&device, &cameras(&frames, bands, 0.0), &stance, &fixed).unwrap();
        let scene = seen.kept.scene();
        let id = id_of(scene, &rendered, 2).expect("the control is visible");
        let object = scene.objects.iter().find(|object| object.id == id);
        let shape = cover(&object.expect("the control").shape);
        let placed = shape.pose();
        let from_shape = batchplan::grasps(&shape, &gripper, &shape_options).unwrap_or_default();
        straight_down += (from_shape.iter())
            .filter(|grasp| (placed.mul_pose(**grasp).rotation * Vec3::Z).z < -0.999)
            .count();
        let turn = seen.kept.turn_grips(
            &device,
            id,
            &gripper,
            normal,
            &shape_options,
            &surface_options,
        );
        let turn = turn.unwrap().expect("the control is seen");
        for pose in shape_grips(&turn) {
            let along = -(pose.rotation * Vec3::Z).dot(normal);
            assert!(
                along > 0.99,
                "{camera}: a grip from above {:.1} degrees off the axis",
                along.acos().to_degrees()
            );
        }
    }
    assert!(
        straight_down > 0,
        "the control's fitted shape must have grips straight down, off the axis by the slant"
    );
}

/// The knob's bar with a floor in both kinds of grasps: the shape's 2 mm under where its grips from
/// above put the fingertips, the surface's 2 mm over it, so that those grips' open hands cross the
/// surface's floor. Every grip `turn_grips` gives whose closure is not obstructed keeps the open
/// hand over that floor; with the surface's floor off, some grip from above would not.
#[test]
fn turn_grips_keep_the_open_hand_over_the_floor() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let q = robot.default_q().to_vec();
    let things = a_knob(&robot, &q);
    let bar = MIDDLE + Vec3::Z * 0.089;
    let rendered = [-0.27, 0.27].map(|side| {
        Frame::of(
            &things,
            looking(Vec3::new(MIDDLE.x + side, -0.04, 0.25), bar),
        )
    });
    let gripper = turn_gripper(&robot);
    let (shape_options, surface_options) = turn_options();
    let tips = bar.z + 0.019 - shape_options.depth - gripper.tip();
    let floor = tips + 0.002;
    let shape_options = ShapeGraspOptions {
        floor: Some(tips - 0.002),
        ..shape_options
    };
    let floored = SurfaceGraspOptions {
        floor: Some(floor),
        ..surface_options
    };
    let over = |pose: &Pose| {
        let lowest =
            |s: &batchplan::HandSphere| (pose.position + pose.rotation * s.center).z - s.radius;
        gripper.spheres().iter().all(|s| lowest(s) >= floor - 1e-5)
    };
    let (mut kept, mut crossing) = (0, 0);
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(&device, &things, &rendered, (c, 7), &q, 0.0);
        let id = id_of(seen.kept.scene(), &rendered, 3).expect("the bar is visible");
        let grips = |o: &SurfaceGraspOptions| {
            let turn = seen
                .kept
                .turn_grips(&device, id, &gripper, Vec3::Z, &shape_options, o);
            let turn = turn.unwrap().expect("the bar is seen");
            let each = turn.groups.into_iter().flat_map(|group| group.grips);
            each.collect::<Vec<_>>()
        };
        for grip in grips(&floored) {
            if grip.evidence != ClosureEvidence::Obstructed {
                assert!(
                    over(&grip.pose),
                    "{camera}: under the floor at {floor}: {grip:?}"
                );
                kept += 1;
            }
        }
        let unfloored = grips(&surface_options);
        let from_above = |g: &&miru::TurnGrip| g.source == GripSource::Shape && !over(&g.pose);
        crossing += unfloored.iter().filter(from_above).count();
    }
    assert!(
        kept > 0 && crossing > 0,
        "{kept} grips kept over the floor, {crossing} from above across it without it"
    );
}

/// An object no label names has no grips, and what is out of range is refused, naming the value,
/// before the look is read, so for an object it does not show as well: by `turn_grips` and by the
/// calls it is made of, `bar_on_disk` and `observed_closures`, options of the shape's grasps and of
/// the surface's alike. A gripper of a hand the device's robot lacks is refused, and one whose jaws
/// do not slide.
#[test]
fn turn_grips_need_the_object_seen_and_input_in_range() {
    let robot = panda();
    let device = Device::cpu_threads(&robot, 2).unwrap();
    let q = robot.default_q().to_vec();
    let things = vec![table(), a_control(), arm(&robot, &q)];
    let rendered = one_side(&things);
    let seen = first_look(&device, &things, &rendered, (0, 7), &q, 0.0);
    let id = id_of(seen.kept.scene(), &rendered, 1).expect("the box is visible");
    let (shape_options, surface_options) = turn_options();
    let gripper = turn_gripper(&robot);
    let kept = &seen.kept;
    assert!(
        kept.turn_grips(
            &device,
            id,
            &gripper,
            Vec3::Z,
            &shape_options,
            &surface_options
        )
        .unwrap()
        .is_some()
    );
    assert!(
        kept.turn_grips(
            &device,
            u32::MAX,
            &gripper,
            Vec3::Z,
            &shape_options,
            &surface_options
        )
        .unwrap()
        .is_none()
    );
    // What a calibration contradicts, `Robot::gripper` refuses (batchplan's `tests/robot.rs`).
    // The same hand, but hand 1 of a Panda with another at its fourth link: one whose jaws slide,
    // on a device whose robot has no hand 1.
    let two = panda_with_hands(&["panda_link4", "ee_link"]);
    let elsewhere = two
        .gripper(1, &GripperCalibration::new(0.08, 0.0, panda_closed()))
        .expect("the second hand's gripper");
    let still = robot
        .gripper(0, &GripperCalibration::new(0.08, 0.0, HashMap::new()))
        .expect("the hand as a gripper whose fingers stay");
    assert!(elsewhere.sliding() && !still.sliding());
    let stray = SurfaceGraspOptions {
        approach: -0.5,
        ..surface_options
    };
    let unturned = SurfaceGraspOptions {
        turns: 0,
        ..surface_options
    };
    let pointless = SurfaceGraspOptions {
        contact: f32::NAN,
        ..surface_options
    };
    let unturned_shape = ShapeGraspOptions {
        turns: 0,
        ..shape_options
    };
    let (up, off) = (Vec3::Z, Vec3::new(0.0, 0.0, 0.5));
    let down = Quat::from_rotation_x(std::f32::consts::PI);
    let poses = [Pose::new(Vec3::new(0.5, 0.0, 0.1), down)];
    let lost = [Pose::new(Vec3::NAN, down)];
    // Refused before the look is read: for an object it does not show as well.
    for id in [id, u32::MAX] {
        let turn = |g: &Gripper, up: Vec3, o: &SurfaceGraspOptions| {
            kept.turn_grips(&device, id, g, up, &shape_options, o)
                .map(|_| ())
        };
        let shaped = |o: &ShapeGraspOptions| {
            kept.turn_grips(&device, id, &gripper, up, o, &surface_options)
                .map(|_| ())
        };
        let bar = |g: &Gripper, out: Vec3, o: &SurfaceGraspOptions| {
            kept.bar_on_disk(&device, id, out, g, shape_options.depth, o)
                .map(|_| ())
        };
        let closures = |g: &Gripper, poses: &[Pose], o: &SurfaceGraspOptions| {
            kept.observed_closures(&device, id, g, poses, o).map(|_| ())
        };
        let refused = [
            ("turn", turn(&elsewhere, up, &surface_options), "hand 1"),
            ("turn", turn(&still, up, &surface_options), "slide"),
            ("turn", turn(&gripper, up, &stray), "-0.5"),
            ("turn", turn(&gripper, up, &unturned), "turns 0, contact"),
            ("turn", turn(&gripper, up, &pointless), "contact NaN"),
            ("turn", shaped(&unturned_shape), "turns 0, tip"),
            ("turn", turn(&gripper, off, &surface_options), "0.5"),
            ("bar", bar(&elsewhere, up, &surface_options), "hand 1"),
            ("bar", bar(&still, up, &surface_options), "slide"),
            ("bar", bar(&gripper, up, &stray), "-0.5"),
            ("bar", bar(&gripper, off, &surface_options), "0.5"),
            (
                "closures",
                closures(&elsewhere, &poses, &surface_options),
                "hand 1",
            ),
            (
                "closures",
                closures(&still, &poses, &surface_options),
                "slide",
            ),
            ("closures", closures(&gripper, &poses, &stray), "-0.5"),
            (
                "closures",
                closures(&gripper, &lost, &surface_options),
                "NaN",
            ),
        ];
        for (call, result, value) in refused {
            let message = result.expect_err("out of range").to_string();
            assert!(
                message.contains(value),
                "{call} {id}: {message} names {value}"
            );
        }
    }
}

/// Readings outside the corners a memory was given are in no look and no map, and what lies
/// inside is seen as without them: of two blocks either side of x = 0.5, a memory bounded there
/// keeps the near one as the unbounded memory has it and plans through where the far one stands.
#[test]
fn what_lies_outside_a_memorys_corners_is_not_seen() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    let away = robot.default_q();
    let (near, far) = (Vec3::new(0.35, -0.25, 0.03), Vec3::new(0.65, 0.25, 0.03));
    let half = Vec3::new(0.04, 0.06, 0.03);
    let things = vec![
        table(),
        block(near, half, [200, 30, 30]),
        block(far, half, [30, 30, 200]),
        arm(&robot, away),
    ];
    let rendered = render(&things);
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let (frames, bands) = read(&rendered, c, 10 * c as u64);
        let locked = HashMap::new();
        let stance = Stance {
            q: away,
            locked: &locked,
            padding: padding(&rendered, &frames, things.len() - 1),
        };
        let seen = |o: &MemoryOptions| {
            let memory = Memory::try_new(&device, vec![known(&table())], o).expect("a memory");
            let seen = observe(&device, &cameras(&frames, bands, 0.0), &stance, &memory);
            seen.expect("a look").memory
        };
        let at = |memory: &Memory, at: Vec3| -> Vec<Shape> {
            let objects = memory.objects().iter();
            objects
                .filter(|object| (object.shape.solids().iter()).any(|s| s.distance(at).value < 0.0))
                .map(|object| object.shape)
                .collect()
        };
        let whole = seen(&MemoryOptions::default());
        let bounded = seen(&MemoryOptions {
            upper_corner: Vec3::new(0.5, f32::INFINITY, f32::INFINITY),
            ..MemoryOptions::default()
        });
        let planned = bounded.world(&device, &Step::default()).expect("a world");
        let solid = (planned.world.obstacles.iter()).any(|o| o.distance(far).value < 0.0);
        let mut wrong = vec![];
        if at(&whole, near).len() != 1 || at(&whole, far).len() != 1 {
            wrong.push(format!(
                "the scene does not show both blocks: {:?}",
                whole.objects()
            ));
        }
        if at(&bounded, near) != at(&whole, near) {
            wrong.push(format!(
                "the near block is {:?} within the corners and {:?} without",
                at(&bounded, near),
                at(&whole, near)
            ));
        }
        if !at(&bounded, far).is_empty() || bounded.objects().len() != 1 {
            wrong.push(format!("past x = 0.5 it keeps {:?}", bounded.objects()));
        }
        if solid {
            wrong.push(format!(
                "its world stands where the far block does: {:?}",
                planned.of
            ));
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict("what_lies_outside_a_memorys_corners_is_not_seen", &found);
}

/// A robot whose base travels reaches farther than a map of the default voxel size holds, and its
/// memory is refused, naming the corners; with corners that bound it the memory is made. Corners
/// that are no numbers, or that leave no room within the reach, are refused, naming them.
#[test]
fn a_robot_that_reaches_far_keeps_a_map_within_corners() {
    let urdf = r#"<robot name="far">
      <link name="base"/>
      <joint name="slide" type="prismatic">
        <parent link="base"/><child link="cart"/><axis xyz="1 0 0"/>
        <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
      </joint>
      <link name="cart">
        <collision><geometry><box size="0.3 0.3 0.3"/></geometry></collision>
      </link>
      <joint name="swing" type="revolute">
        <parent link="cart"/><child link="arm"/><origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
        <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
      </joint>
      <link name="arm">
        <collision><origin xyz="0.4 0 0"/><geometry><box size="0.8 0.1 0.1"/></geometry></collision>
      </link>
      <joint name="tip" type="fixed">
        <parent link="arm"/><child link="hand"/><origin xyz="0.8 0 0"/>
      </joint>
      <link name="hand"/>
    </robot>"#;
    let options = batchplan::RobotOptions {
        ee_links: vec!["hand".to_string()],
        ..Default::default()
    };
    let robot = Robot::from_urdf(urdf, &options).expect("a cart with an arm");
    let device = Device::cpu(&robot).expect("a CPU device");
    let reach = robot.reach();
    assert!(
        reach > 2.1,
        "the cart carries the arm {reach} m: past a map"
    );
    let made = |lower_corner: Vec3, upper_corner: Vec3| {
        let o = MemoryOptions {
            lower_corner,
            upper_corner,
            ..MemoryOptions::default()
        };
        Memory::try_new(&device, vec![], &o).map(|_| ())
    };
    let (none, all) = (Vec3::NEG_INFINITY, Vec3::INFINITY);
    let Err(batchplan::Error::Input(whole)) = made(none, all) else {
        panic!("its whole reach is too large a map");
    };
    assert!(whole.contains("corners"), "{whole} says what bounds it");
    // The floor it stands on and as high as its arm swings: 1.3 m of the 4.5 m its reach spans.
    let floor = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, -0.1);
    let ceiling = Vec3::new(f32::INFINITY, f32::INFINITY, 1.2);
    made(floor, ceiling).expect("a map from the floor to over the arm");
    let refused = [
        (Vec3::new(0.0, f32::NAN, 0.0), all, "NaN"),
        (
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 0.5),
            "no room",
        ),
        (Vec3::splat(3.0), all, "no room"),
    ];
    for (lower, upper, value) in refused {
        let Err(batchplan::Error::Input(message)) = made(lower, upper) else {
            panic!("corners {lower} and {upper} make a map");
        };
        assert!(message.contains(value), "{message} names {value}");
    }
}

/// A place into what stands at a point does not hang on which objects the look made of it: a box
/// whose tall far wall a mask model's masks give to another thing than the rest comes out in two
/// objects, as a cabinet and the drawer pulled out of it do, and the wall's object, which a
/// caller that pulled it would name, stands beside the hollow: nothing is let go over its own
/// readings. Every reading about the box's middle (`Kept::surface_about`) shows the hollow, and a
/// block let go over it falls in. The surface holds the readings within its reach and none
/// beyond, and flags none.
#[test]
fn a_fall_into_a_box_in_pieces_is_found_in_every_reading_about_it() {
    let robot = panda();
    let device = Device::cpu(&robot).expect("a CPU device");
    // The open hand reaches into the box, so the arm reads within the surface's reach.
    let reaching = &IN_BOWL;
    let grey = [150, 150, 150];
    let things = vec![
        table(),
        block(
            Vec3::new(0.5, 0.0, 0.005),
            Vec3::new(0.10, 0.10, 0.005),
            grey,
        ),
        block(
            Vec3::new(0.605, 0.0, 0.025),
            Vec3::new(0.005, 0.11, 0.025),
            grey,
        ),
        block(
            Vec3::new(0.5, 0.105, 0.025),
            Vec3::new(0.10, 0.005, 0.025),
            grey,
        ),
        block(
            Vec3::new(0.5, -0.105, 0.025),
            Vec3::new(0.10, 0.005, 0.025),
            grey,
        ),
        block(
            Vec3::new(0.38, 0.0, 0.06),
            Vec3::new(0.02, 0.11, 0.06),
            grey,
        ),
        arm(&robot, reaching),
    ];
    let mut rendered = render(&things);
    // The masks say the tall far wall is one thing and the rest of the box another.
    for frame in &mut rendered {
        let mask = |shows: &Option<usize>| match shows {
            Some(5) => 1,
            Some(1..=4) => 2,
            _ => 0,
        };
        frame.masks = frame.shows.iter().map(mask).collect();
    }
    let (middle, reach) = (Vec3::new(0.5, 0.0, 0.05), 0.2);
    let problem = batchplan::RestProblem {
        near: Some(middle),
        ..batchplan::RestProblem::new(Obstacle::cuboid(Vec3::ZERO, Vec3::new(0.03, 0.03, 0.02)), 0)
    };
    let falls = |surface: &batchplan::Surface| {
        let options = batchplan::RestOptions::default();
        let problems = std::slice::from_ref(&problem);
        let found =
            batchplan::falls_into(&device, problems, std::slice::from_ref(surface), &options);
        found.expect("falls").remove(0)
    };
    let mut found = vec![];
    for (c, (camera, ..)) in CAMERAS.iter().enumerate() {
        let seen = first_look(
            &device,
            &things,
            &rendered,
            (c, 10 * c as u64),
            reaching,
            0.0,
        );
        let scene = seen.kept.scene();
        let mut wrong = vec![];
        let about = seen.kept.surface_about(&device, middle, reach);
        let about = about.expect("a sound point").expect("a look with views");
        let outside = (about.points.iter())
            .filter(|p| (**p - middle).truncate().length() > reach)
            .count();
        if outside > 0 || !about.object.is_empty() {
            wrong.push(format!(
                "{outside} readings beyond its reach, {} flags",
                about.object.len()
            ));
        }
        // The arm is within the reach, and none of its readings is in the surface: the surface
        // holds what an object's surface holds there, flagged or not.
        let on_arm: usize = (0..2)
            .map(|view| {
                let image = seen.kept.image(view).expect("a kept view");
                let mask = seen.kept.robot(view).expect("a kept mask");
                let points = image.points(None).expect("a sound image");
                let within = |point: &Option<Vec3>| {
                    point.is_some_and(|point| (point - middle).truncate().length() <= reach)
                };
                let robots = points.iter().zip(&mask.on_robot);
                robots.filter(|(point, on)| **on && within(point)).count()
            })
            .sum();
        if on_arm == 0 {
            wrong.push("the arm reads nowhere within the reach".to_string());
        }
        let (floor, wall) = (holder(scene, &rendered, 1), holder(scene, &rendered, 5));
        let Some(wall) = wall else {
            found.push((
                *camera,
                Err("no object holds the box's far wall".to_string()),
            ));
            continue;
        };
        let id = scene.objects[wall].id;
        let own = seen.kept.surface(&device, id).expect("a surface");
        let own = own.expect("the far wall's object is in view");
        let within = (own.points.iter())
            .filter(|p| (**p - middle).truncate().length() <= reach)
            .count();
        if within != about.points.len() {
            wrong.push(format!(
                "{} readings about the point, {within} of the look's within its reach",
                about.points.len()
            ));
        }
        let into = falls(&about);
        let over = (into.iter()).all(|fall| {
            let at = fall.pose.position - middle;
            at.x.abs() < 0.1 && at.y.abs() < 0.1
        });
        if into.is_empty() || !over {
            wrong.push(format!(
                "{} falls in every reading about it, all over its floor: {over}",
                into.len()
            ));
        }
        // What the test is of: the look gives the floor to another object than the far wall,
        // and over the wall's own readings nothing is let go.
        let alone = falls(&own).len();
        if floor == Some(wall) || alone > 0 {
            wrong.push(format!(
                "the floor's object is the far wall's: {}, whose readings alone give {alone} falls",
                floor == Some(wall)
            ));
        }
        found.push((*camera, outcome(&wrong)));
    }
    verdict(
        "a_fall_into_a_box_in_pieces_is_found_in_every_reading_about_it",
        &found,
    );
    // A point or a reach that is no number, or no reach at all, is refused by name.
    let seen = first_look(&device, &things, &rendered, (0, 0), reaching, 0.0);
    for (at, reach, value) in [
        (Vec3::new(f32::NAN, 0.0, 0.0), 0.2, "NaN"),
        (middle, 0.0, "and 0"),
        (middle, f32::INFINITY, "inf"),
    ] {
        let refused = seen.kept.surface_about(&device, at, reach);
        let Err(batchplan::Error::Input(why)) = refused else {
            panic!("a surface about {at} within {reach}");
        };
        assert!(why.contains(value), "{why} names {value}");
    }
}
