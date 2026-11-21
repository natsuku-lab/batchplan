//! The stereo cameras the tests read scenes through (`common::Frame::sensed`): each of their
//! errors as the datasheet and the geometry of two imagers give it, on depth images whose truth is
//! known: a wall at one depth, and a box before it.

#[allow(dead_code, reason = "each test file uses part of it")]
mod common;

use batchplan::Pose;
use common::{D405, D435, Frame, HEIGHT, K, NOISE, Stereo, WIDTH};
use glam::{Quat, Vec3};

/// How many standard errors a statistic of a frame's readings may lie from what the camera states:
/// a camera read as stated lies further once in 1.7 million frames.
const ERRORS: f64 = 5.0;

/// A frame of the tests' pinhole whose depth is `depth`, row by row, showing nothing.
fn frame(depth: Vec<f32>) -> Frame {
    let pixels = depth.len();
    Frame {
        camera: Pose::new(Vec3::ZERO, Quat::IDENTITY),
        width: WIDTH,
        intrinsics: K,
        depth,
        color: vec![0; 3 * pixels],
        shows: vec![None; pixels],
        robot: vec![],
        masks: vec![],
        noise: NOISE,
    }
}

/// A wall `z` meters from the camera, square on.
fn wall(z: f32) -> Frame {
    frame(vec![z; WIDTH * HEIGHT])
}

/// A wall at `far` with a box before it at `near` over columns 150 to 249 of rows 80 to 159.
fn box_before_a_wall(near: f32, far: f32) -> Frame {
    let mut depth = vec![far; WIDTH * HEIGHT];
    for row in depth.chunks_mut(WIDTH).take(160).skip(80) {
        row[150..250].fill(near);
    }
    frame(depth)
}

/// `camera` with neither matching noise nor jitter: what the geometry of its two imagers alone
/// leaves.
fn quiet(camera: Stereo) -> Stereo {
    Stereo {
        subpixel: 0.0,
        jitter: 0.0,
        ..camera
    }
}

/// A point's disparity at depth `z` to `camera` through the tests' pinhole (pixels).
fn disparity_at(camera: Stereo, z: f32) -> f32 {
    (f64::from(K.fx) * camera.baseline) as f32 / z
}

#[test]
fn a_wall_reads_with_the_deviation_the_camera_states() {
    for (camera, z) in [
        (D435, 0.5),
        (D435, 1.0),
        (D435, 2.0),
        (D405, 0.2),
        (D405, 0.5),
    ] {
        let read = wall(z).sensed(camera, 1);
        assert_eq!(
            read.noise,
            camera.noise(K.fx),
            "{camera:?} states its noise"
        );
        assert_eq!(
            read.depth,
            wall(z).sensed(camera, 1).depth,
            "the same every time"
        );
        assert_ne!(
            read.depth,
            wall(z).sensed(camera, 2).depth,
            "a seed of its own"
        );
        // The readings right of the margin the second imager's view leaves, which the shadow's
        // test holds the camera to.
        let disparity = disparity_at(camera, z);
        let readings: Vec<f64> = (read.depth.chunks(WIDTH))
            .flat_map(|row| row[disparity.ceil() as usize + 1..].iter())
            .map(|&depth| f64::from(depth))
            .collect();
        assert!(
            readings.iter().all(|&depth| depth > 0.0),
            "{camera:?} at {z} m"
        );
        let n = readings.len() as f64;
        let mean = readings.iter().sum::<f64>() / n;
        let deviation =
            (readings.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        // The stated deviation, and the rounding to a whole count of the unit, a uniform error of
        // unit^2 / 12 in variance (Sheppard's correction). Depth is the disparity's inverse, so
        // its mean lies past the wall by the disparity's relative variance, to second order.
        let stated = f64::from(read.noise.quadratic * z * z);
        let unit = camera.unit;
        let expected = (stated * stated + unit * unit / 12.0).sqrt();
        let relative = camera.subpixel / f64::from(disparity);
        let middle = f64::from(z) * (1.0 + relative * relative);
        assert!(
            (mean - middle).abs() < ERRORS * expected / n.sqrt(),
            "{camera:?} at {z} m: the mean of {n} readings is {mean}, from {middle}"
        );
        assert!(
            (deviation - expected).abs() < ERRORS * expected / (2.0 * (n - 1.0)).sqrt(),
            "{camera:?} at {z} m: {n} readings deviate by {deviation}, from {expected}"
        );
    }
}

#[test]
fn nothing_reads_nearer_than_the_least_distance_and_everything_just_past_it_does() {
    for camera in [D435, D405] {
        let least = camera.least(K.fx);
        // Five deviations of the matching's error to either side of the farthest disparity the
        // camera searches.
        let reach = 5.0 * camera.subpixel;
        let at = |disparity: f64| (f64::from(K.fx) * camera.baseline / disparity) as f32;
        let (nearer, past) = (
            at(camera.disparities + reach),
            at(camera.disparities - reach),
        );
        assert!(nearer < least && least < past);
        for z in [0.5 * least, nearer] {
            let read = wall(z).sensed(camera, 3);
            assert!(
                read.depth.iter().all(|&depth| depth == 0.0),
                "{camera:?} at {z} m"
            );
        }
        // Every pixel the second imager sees reads: those right of the margin its view leaves.
        let read = wall(past).sensed(camera, 3);
        let margin = disparity_at(camera, past).ceil() as usize + 1;
        let rows = read.depth.chunks(WIDTH);
        assert!(
            rows.flat_map(|row| &row[margin..])
                .all(|&depth| depth > 0.0),
            "{camera:?} at {past} m"
        );
    }
}

#[test]
fn a_near_box_leaves_a_shadow_on_its_left_as_wide_as_its_disparity_is_more() {
    let camera = quiet(D435);
    let (near, far) = (0.6, 1.1);
    let read = box_before_a_wall(near, far).sensed(camera, 4);
    let reads = |row: usize, column: usize| read.depth[row * WIDTH + column] > 0.0;
    // Where a column lands in the second imager is rounded, and a point covers the column it
    // lands on and the next: the band is as wide as the difference to a pixel.
    let difference = disparity_at(camera, near) - disparity_at(camera, far);
    for row in 80..160 {
        let shadow = (0..150)
            .rev()
            .take_while(|&column| !reads(row, column))
            .count();
        assert!(
            (shadow as f32 - difference).abs() <= 1.0,
            "row {row}: a shadow of {shadow} columns where the disparity is {difference} more"
        );
        assert!(
            reads(row, 150) && reads(row, 249) && reads(row, 250),
            "none right of it"
        );
    }
    // Over and under the box the wall reads from the margin of the second imager's view on,
    // and the box and the wall read whole counts of the unit, as the truth is.
    let margin = disparity_at(camera, far);
    for row in [0, 79, 160, HEIGHT - 1] {
        for column in 0..WIDTH {
            assert_eq!(
                reads(row, column),
                column as f32 > margin - 0.5,
                "{row} {column}"
            );
        }
    }
    assert_eq!(
        (read.depth[120 * WIDTH + 200], read.depth[120 * WIDTH + 300]),
        (near, far)
    );
}

#[test]
fn jitter_reads_between_the_surfaces_by_an_edge_and_nowhere_else() {
    let camera = Stereo {
        subpixel: 0.0,
        ..D435
    };
    let (near, far) = (0.6, 1.1);
    let read = box_before_a_wall(near, far).sensed(camera, 5);
    // A pixel reads its neighbours' disparity at most a pixel past where it reads, which lies
    // under five deviations of the jitter off among the frame's 2 x 76,800 draws: the largest
    // of so many normal draws is about sqrt(2 ln n) = 4.9 deviations.
    let reach = (5.0 * camera.jitter + 1.0).ceil() as usize;
    let by_an_edge = |row: usize, column: usize| {
        let (row, column) = (row as isize, column as isize);
        let near = |at: isize, edge: isize| (at - edge).abs() <= reach as isize;
        (near(column, 150) || near(column, 249)) && (80 - 4..160 + 4).contains(&row)
            || (near(row, 80) || near(row, 159)) && (150 - 4..250 + 4).contains(&column)
    };
    let mut between = 0;
    for (i, &depth) in read.depth.iter().enumerate() {
        if depth != 0.0 && depth != near && depth != far {
            assert!(
                by_an_edge(i / WIDTH, i % WIDTH),
                "a reading of {depth} at pixel {i}"
            );
            assert!(
                depth > near && depth < far,
                "a reading of {depth} at pixel {i}"
            );
            between += 1;
        }
    }
    assert!(
        between > 50,
        "{between} readings between the box and the wall"
    );
}

#[test]
fn a_reading_is_a_whole_count_of_the_unit_and_none_past_sixteen_bits() {
    for (camera, z) in [(D435, 0.7), (D405, 0.3)] {
        let read = wall(z).sensed(camera, 6);
        let counts: Vec<f32> = (read.depth.iter())
            .filter(|&&depth| depth > 0.0)
            .map(|depth| depth / camera.unit as f32)
            .collect();
        assert!(
            counts
                .iter()
                .all(|count| (count - count.round()).abs() < 1e-2),
            "{camera:?} at {z} m"
        );
        // The noise spans many counts: the unit is finer than it.
        let mut whole: Vec<i64> = counts.iter().map(|count| count.round() as i64).collect();
        whole.sort_unstable();
        whole.dedup();
        assert!(
            whole.len() > 5,
            "{camera:?} at {z} m reads {} counts",
            whole.len()
        );
    }
    // 20 m is 20,000 counts of a D435's 1 mm and 200,000 of a D405's 0.1 mm, which 16 bits do
    // not hold: both read its disparity, under a pixel, and only the D435 gives a reading.
    let (held, past) = (wall(20.0).sensed(D435, 7), wall(20.0).sensed(D405, 7));
    let margin = disparity_at(D435, 20.0).ceil() as usize + 1;
    let rows = held.depth.chunks(WIDTH);
    assert!(
        rows.flat_map(|row| &row[margin..])
            .all(|&depth| depth > 0.0)
    );
    assert!(past.depth.iter().all(|&depth| depth == 0.0));
}
