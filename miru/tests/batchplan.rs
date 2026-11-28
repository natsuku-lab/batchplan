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

use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;

use batchplan::{
    Device, Held, IkOptions, IkProblem, Obstacle, Pick, PickBody, Place, Pose, RestOptions,
    RestProblem, Surface, TaskOptions, World, Worlds,
};
use common::{
    EYE, Frame, MIDDLE, NEAR, NOISE, arm, block, can, devices, held_from_above, known, looking,
    panda, sizes, table, thin_tray, two_pandas,
};
use glam::{Mat3, Quat, Vec3};
use miru::{
    Act, Camera, Label, Look, LookOptions, Memory, MemoryOptions, Object, Observed, Of, Planned,
    Scene, Shape, Stance, Step, observe,
};

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

/// What two cameras, the far and the near, read of `things` on exact depth, the arms the last
/// thing: the frames, and the arms' padding, three deviations of the noise at their farthest
/// reading.
fn read(things: &[common::Thing]) -> ([Frame; 2], f32) {
    let frames = [EYE, NEAR].map(|eye| Frame::of(things, looking(eye, MIDDLE)));
    let arms = things.len() - 1;
    let farthest = (frames.iter())
        .flat_map(|frame| {
            (0..frame.depth.len())
                .filter(|&i| frame.shows[i] == Some(arms))
                .map(|i| frame.depth[i])
        })
        .fold(0.0, f32::max);
    (frames, 3.0 * NOISE.quadratic * farthest * farthest)
}

/// `frames` as the cameras `observe` takes.
fn cameras(frames: &[Frame; 2]) -> Vec<Camera<'_>> {
    (frames.iter())
        .map(|frame| Camera {
            image: frame.view().image,
            color: &frame.color,
            noise: frame.noise,
            band: 0,
            off_by: 0.0,
            masks: &frame.masks,
        })
        .collect()
}

/// The object of `seen`'s look whose shape's middle lies within 2 cm of `middle`.
fn object_at(seen: &Observed, middle: Vec3) -> &Object {
    let objects = seen.kept.scene().objects.iter();
    let mut near = objects.filter(|object| sizes(&object.shape).1.distance(middle) < 0.02);
    near.next()
        .unwrap_or_else(|| panic!("an object at {middle}"))
}

/// What hand `hand` holds after batchplan's pick of obstacle `object` of `worlds`' world 0 from
/// `start`, by `grasp` in that obstacle's frame, lifting `body` where given off the table, the
/// world's first obstacle: the hold the library makes of the memory's world.
fn picked(
    device: &Device,
    worlds: &Worlds,
    start: &[f32],
    (hand, object, grasp): (usize, usize, Pose),
    body: Option<PickBody>,
) -> Held {
    let pick = Pick {
        hand,
        body,
        support: vec![0],
        ..Pick::new(0, start, object, [grasp])
    };
    let mut picked = batchplan::pick(device, worlds, &[pick], &TaskOptions::default());
    let plans = picked.as_mut().expect("a sound pick").outcomes.remove(0);
    let plans = plans.unwrap_or_else(|failure| panic!("hand {hand} picks: {failure}"));
    plans[0].held.clone().expect("the pick's hold")
}

/// The configuration nearest the ready pose with each hand at its target, in a world of the
/// table alone.
fn holding_at(device: &Device, ready: &[f32], targets: [Pose; 2]) -> Vec<f32> {
    let table_only = device.upload(&[World::new(vec![known(&table())])]);
    let table_only = table_only.expect("a world");
    let problem = IkProblem {
        seed: Some(ready.to_vec()),
        targets: targets.map(Some).to_vec(),
        ..IkProblem::new(0, Pose::IDENTITY)
    };
    let solved = batchplan::solve_ik(device, &table_only, &[problem], &IkOptions::default());
    let solved = solved.expect("a sound problem");
    solved.nearest(0, ready).expect("both hands reach").to_vec()
}

/// The indices of the obstacles of `planned` that stand for object `id`.
fn of_object(planned: &Planned, id: u32) -> Vec<usize> {
    let of = planned.of.iter().enumerate();
    of.filter(|(_, of)| **of == Of::Object(id))
        .map(|(k, _)| k)
        .collect()
}

/// The flow of a place onto what another hand holds, planned before anything moves, from one
/// look: two Pandas at their ready pose before a table with a plank and a cube on it, seen by two
/// cameras. Each is held as batchplan's pick of it in the memory's world holds it: the plank by
/// hand 0 from above near one end, which is to hold it level in the air, the cube by hand 1 over
/// it. The plank's surface where hand 0's hold has it there (`Kept::surface_held`) gives the
/// cube rests on its top face near its far end (`rests_on`); those, as the releases of a place of
/// the cube onto the plank (`Place::onto`), give a plan in the world the memory gives after both
/// picks, the plan's hold of the plank carrying the cube. On a CPU device of one thread, of all of
/// them, and on the GPU's where there is one.
#[test]
fn a_place_onto_what_a_hand_holds_is_planned_from_one_look() {
    let robot = two_pandas();
    let ready = robot.default_q().to_vec();
    let (plank_at, plank_half) = (Vec3::new(0.55, 0.15, 0.015), Vec3::new(0.03, 0.15, 0.015));
    let cube_at = Vec3::new(0.55, -0.15, 0.02);
    let things = vec![
        table(),
        block(plank_at, plank_half, [200, 30, 30]),
        block(cube_at, Vec3::splat(0.02), [30, 30, 200]),
        arm(&robot, &ready),
    ];
    let (frames, padding) = read(&things);
    let locked = HashMap::new();
    let stance = Stance {
        q: &ready,
        locked: &locked,
        padding,
    };
    for (name, device) in devices(&robot) {
        let memory = Memory::try_new(&device, vec![known(&table())], &MemoryOptions::default());
        let seen = observe(
            &device,
            &cameras(&frames),
            &stance,
            &memory.expect("a memory"),
        );
        let seen = seen.expect("a look");
        let (plank, cube) = (object_at(&seen, plank_at), object_at(&seen, cube_at));
        let planned = seen
            .memory
            .world(&device, &Step::default())
            .expect("a world");
        let worlds = device.upload(std::slice::from_ref(&planned.world));
        let worlds = worlds.expect("a world");
        let (plank_parts, cube_parts) =
            (of_object(&planned, plank.id), of_object(&planned, cube.id));
        let (&[plank_index], &[cube_index]) = (&plank_parts[..], &cube_parts[..]) else {
            panic!("{name}: the plank and the cube are a solid each");
        };
        let from_above = held_from_above(&plank.shape, 0, 0.11).grasp;
        let holds_plank = picked(&device, &worlds, &ready, (0, plank_index, from_above), None);
        let from_above = held_from_above(&cube.shape, 1, 0.0).grasp;
        let holds_cube = picked(&device, &worlds, &ready, (1, cube_index, from_above), None);

        // Hand 0 holds the plank level 45 cm up between the arms, its length along y and the hand
        // at its end toward hand 0's arm; hand 1 holds the cube 30 cm to the right and higher.
        let plank_half = match &holds_plank.object {
            Obstacle::Cuboid { half_extents, .. } => *half_extents,
            other => panic!("{name}: the pick holds the plank as {other:?}"),
        };
        let lengthwise = match plank_half.x > plank_half.y {
            true => Quat::from_rotation_z(FRAC_PI_2),
            false => Quat::IDENTITY,
        };
        let in_the_air = Pose::new(Vec3::new(0.35, 0.0, 0.45), lengthwise);
        let cube_over = Pose::new(Vec3::new(0.35, -0.3, 0.6), Quat::IDENTITY);
        let start = holding_at(
            &device,
            &ready,
            [
                in_the_air.mul_pose(holds_plank.grasp),
                cube_over.mul_pose(holds_cube.grasp),
            ],
        );

        let surface = seen
            .kept
            .surface_held(&device, plank.id, &holds_plank, &start);
        let surface = surface
            .expect("a hold of a box")
            .expect("the plank's readings");
        let carried = robot
            .ee_pose(&start, 0)
            .mul_pose(holds_plank.grasp.inverse());
        let toward_the_hand =
            carried.rotation.inverse() * (robot.ee_pose(&start, 0).position - carried.position);
        let far_end =
            carried.position - carried.rotation * (toward_the_hand * Vec3::new(1.0, 1.0, 0.0));
        let problem = RestProblem {
            near: Some(far_end),
            ..RestProblem::new(holds_cube.object.clone(), 0)
        };
        let rests = batchplan::rests_on(&device, &[problem], &[surface], &RestOptions::default());
        let rests = rests.expect("sound problems").remove(0);
        assert!(!rests.is_empty(), "{name}: nothing rests on the plank");
        let half = sizes(&cube.shape).2;
        let top = carried.position.z + plank_half.z;
        let first = &rests[..rests.len().min(8)];
        for rest in first {
            let on = carried.inverse().mul_pose(rest.pose).position;
            assert!(
                (rest.pose.position.z - half - top).abs() < 0.003
                    && on.x.abs() < plank_half.x
                    && on.y.abs() < plank_half.y,
                "{name}: a rest at {:?}, off the plank's top at {top}",
                rest.pose
            );
        }

        // The world after both picks: the table and what the map holds that no shape explains.
        let both = [
            Act::Picked {
                id: plank.id,
                held: holds_plank.clone(),
            },
            Act::Picked {
                id: cube.id,
                held: holds_cube.clone(),
            },
        ];
        let after = seen.memory.after(&both).expect("both picks");
        let planned = after.world(&device, &Step::default()).expect("a world");
        let worlds = device.upload(&[planned.world]).expect("a world");
        let place = Place {
            onto: Some(holds_plank.clone()),
            ..Place::new(0, start.clone(), holds_cube.clone(), first)
        };
        let placed = batchplan::place(&device, &worlds, &[place], &TaskOptions::default());
        let plans = placed.expect("a sound place").outcomes.remove(0);
        let plans = plans.unwrap_or_else(|failure| panic!("{name}: {failure}"));
        let carrying = plans[0].held.as_ref().expect("the hold of the plank after");
        assert_eq!(
            carrying.carries.len(),
            1,
            "{name}: the plank carries the cube"
        );
    }
}

/// The flow of a place into what another hand holds, planned before anything moves, from one
/// look: two Pandas at their ready pose before a table with the thin tray and a 3 cm cube on it.
/// Hand 0 picks the tray by its wall toward that arm, the whole tray lifted with it (batchplan's
/// pick of the wall with the tray as its body), to hold it level in the air; hand 1 picks the
/// cube, to hold it beside and over it. The tray's surface where hand 0's hold has it there
/// (`Kept::surface_held`) gives falls of the cube into it toward its far side from hand 0
/// (`falls_into`), each landing inside it;
/// those, as the releases of a place of the cube into the tray (`Place::onto`), give a plan in
/// the world the memory gives after both picks. On exact depth, on a CPU device of one thread,
/// of all of them, and on the GPU's where there is one.
#[test]
fn a_place_into_what_a_hand_holds_is_planned_from_one_look() {
    let robot = two_pandas();
    let ready = robot.default_q().to_vec();
    let cube_at = Vec3::new(0.45, -0.25, 0.015);
    let mut things = vec![table()];
    things.extend(thin_tray());
    things.push(block(cube_at, Vec3::splat(0.015), [30, 30, 200]));
    things.push(arm(&robot, &ready));
    let (frames, padding) = read(&things);
    let locked = HashMap::new();
    let stance = Stance {
        q: &ready,
        locked: &locked,
        padding,
    };
    // Hand 0 closes on the top of the tray's wall toward its arm, across it, from above.
    let at_the_wall = Pose::new(
        Vec3::new(0.5, 0.115, 0.04),
        Quat::from_mat3(&Mat3::from_cols(Vec3::NEG_X, Vec3::Y, Vec3::NEG_Z)),
    );
    for (name, device) in devices(&robot) {
        let memory = Memory::try_new(&device, vec![known(&table())], &MemoryOptions::default());
        let seen = observe(
            &device,
            &cameras(&frames),
            &stance,
            &memory.expect("a memory"),
        );
        let seen = seen.expect("a look");
        let tray = object_at(&seen, Vec3::new(0.5, 0.0, 0.03));
        assert!(
            matches!(tray.shape, Shape::OpenBox { .. }),
            "{name}: the tray is a box open at the top on exact depth"
        );
        let cube = object_at(&seen, cube_at);
        let planned = seen
            .memory
            .world(&device, &Step::default())
            .expect("a world");
        let worlds = device.upload(std::slice::from_ref(&planned.world));
        let worlds = worlds.expect("a world");
        let parts = of_object(&planned, tray.id);
        let obstacles = &planned.world.obstacles;
        let nearest = |at: Vec3| move |&k: &usize| obstacles[k].distance(at).value;
        let wall = *parts
            .iter()
            .min_by(|a, b| {
                nearest(at_the_wall.position)(a).total_cmp(&nearest(at_the_wall.position)(b))
            })
            .expect("the tray's solids");
        let body = PickBody {
            shape: room_of(&tray.shape),
            other_parts: parts.iter().copied().filter(|&k| k != wall).collect(),
        };
        let grasp = obstacles[wall].pose().inverse().mul_pose(at_the_wall);
        let holds_tray = picked(&device, &worlds, &ready, (0, wall, grasp), Some(body));
        let &[cube_index] = &of_object(&planned, cube.id)[..] else {
            panic!("{name}: the cube is one solid");
        };
        // Hand 1 closes on the cube's upper half from above, along x, so that it stands narrow
        // toward hand 0 and its fingertips stay over the tray's rim where it lets go.
        let cube_room = room_of(&cube.shape).pose();
        let along_x = Quat::from_mat3(&Mat3::from_cols(Vec3::Y, Vec3::X, Vec3::NEG_Z));
        let upper = Pose::new(cube_room.position + Vec3::Z * 0.01, along_x);
        let from_above = cube_room.inverse().mul_pose(upper);
        let holds_cube = picked(&device, &worlds, &ready, (1, cube_index, from_above), None);

        // Hand 0 holds the tray level 30 cm up, as it stood; hand 1 holds the cube beside it,
        // higher, as it stood.
        let room = room_of(&tray.shape).pose();
        let in_the_air = Pose::new(Vec3::new(0.45, 0.0, 0.3), room.rotation);
        let cube_over = Pose::new(Vec3::new(0.4, -0.3, 0.5), cube_room.rotation);
        let start = holding_at(
            &device,
            &ready,
            [
                in_the_air.mul_pose(holds_tray.grasp),
                cube_over.mul_pose(holds_cube.grasp),
            ],
        );

        let surface = seen
            .kept
            .surface_held(&device, tray.id, &holds_tray, &start);
        let surface = surface
            .expect("a hold of a box")
            .expect("the tray's readings");
        // Where the hold has the tray, and its foot's middle in the true tray.
        let carried = robot
            .ee_pose(&start, 0)
            .mul_pose(holds_tray.grasp.inverse());
        let foot = carried
            .mul_pose(room.inverse())
            .mul_pose(Pose::new(Vec3::new(0.5, 0.0, 0.0), Quat::IDENTITY));
        // Toward its far side from hand 0, where hand 1 has room beside it.
        let far_side = foot.position + Vec3::new(0.0, -0.06, 0.03);
        let problem = RestProblem {
            near: Some(far_side),
            rotation: cube_room.rotation,
            ..RestProblem::new(holds_cube.object.clone(), 0)
        };
        let falls = batchplan::falls_into(&device, &[problem], &[surface], &RestOptions::default());
        let falls = falls.expect("sound problems").remove(0);
        assert!(!falls.is_empty(), "{name}: nothing falls into the tray");
        let first = &falls[..falls.len().min(8)];
        for fall in first {
            let lands = foot.inverse().mul_pose(Pose::new(
                fall.pose.position - Vec3::Z * fall.drop,
                Quat::IDENTITY,
            ));
            let local = lands.position;
            assert!(
                local.x.abs() < 0.11 - 0.015
                    && local.y.abs() < 0.11 - 0.015
                    && (local.z - 0.025).abs() < 0.005,
                "{name}: a fall lands at {local} in the tray's frame"
            );
        }

        let both = [
            Act::Picked {
                id: tray.id,
                held: holds_tray.clone(),
            },
            Act::Picked {
                id: cube.id,
                held: holds_cube.clone(),
            },
        ];
        let after = seen.memory.after(&both).expect("both picks");
        let planned = after.world(&device, &Step::default()).expect("a world");
        let worlds = device.upload(&[planned.world]).expect("a world");
        let place = Place {
            onto: Some(holds_tray.clone()),
            ..Place::new(0, start.clone(), holds_cube.clone(), first)
        };
        let placed = batchplan::place(&device, &worlds, &[place], &TaskOptions::default());
        let plans = placed.expect("a sound place").outcomes.remove(0);
        let plans = plans.unwrap_or_else(|failure| panic!("{name}: {failure}"));
        let carrying = plans[0].held.as_ref().expect("the hold of the tray after");
        assert_eq!(
            carrying.carries.len(),
            1,
            "{name}: the tray carries the cube"
        );
    }
}

/// The room `shape` takes up, as an obstacle: a box open at the top's box, and a box or a
/// cylinder itself.
fn room_of(shape: &Shape) -> Obstacle {
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
