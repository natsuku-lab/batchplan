//! Recorded looks, as batchplan-orchestrator's benchmark writes them (`bpo-bench looks`): a set is
//! a folder of runs, and a run is one scene with its looks in order.
//!
//! A run's `cell.json` has the shapes its world is known to hold, its `fixtures`: each a `pose`
//! and a `shape`, a `box` of `half_extents` or a `cylinder` of `radius` and `half_height` along
//! its z axis. A look is a folder named by its number, with:
//!
//! - `look.json`: its `cameras`, each with a `name`, `width`, `height`, `intrinsics` (fx, fy, cx,
//!   cy), `position` and `rotation` (x, y, z, w; the robot's base frame, OpenCV axes) and `noise`
//!   (c0, c1, c2 of its depth's standard deviation, c0 + c1 z + c2 z^2 meters at z meters), and
//!   the `objects` the executor that recorded the look measured, each with its `center`;
//! - per camera `color_<camera>.npy` (u8, h x w x 3) and `depth_<camera>.npy` (f32, meters along
//!   the optical axis, 0 for no reading);
//! - what that executor made of each pixel: `robot_<camera>.npy` (bool, whether it took the
//!   reading as the robot's or of what a hand holds) and `object_<camera>.npy` (u16, the one of
//!   `objects` it names there by its index, 65535 for none);
//! - the simulator's truth: `truth.json`, whose `bodies` each say the `object` they are part of,
//!   if any, and whose `frames` each have the truth `objects`' poses by `name`; and per camera
//!   `body_<camera>.npy` (u16, the body each pixel shows, 65535 for none).

use std::fs;
use std::path::Path;

use batchplan::{Device, Intrinsics, Obstacle, Pose, Robot, RobotOptions};
use glam::{Quat, Vec2, Vec3};
use miru::Noise;
use serde_json::Value;

/// What went wrong reading a recording.
pub type Error = Box<dyn std::error::Error>;

/// The label of a pixel that shows no body, or that the executor names no object under.
const NOTHING: u16 = u16::MAX;

/// One camera's frame of a recorded look. What is per pixel is row by row, as the depth is.
pub struct Camera {
    pub name: String,
    pub width: usize,
    pub intrinsics: Intrinsics,
    /// The camera in the robot's base frame, OpenCV axes.
    pub pose: Pose,
    /// Its depth noise, as its cell states it.
    pub noise: Noise,
    pub depth: Vec<f32>,
    pub color: Vec<u8>,
    /// Per pixel, whether the executor that recorded the look took the reading as the robot's or
    /// of what a hand holds: `miru::View::robot`.
    pub robot: Vec<bool>,
    /// Per pixel, the object that executor names there, by its index among those it measured.
    pub measured: Vec<Option<usize>>,
    /// Per pixel, the truth object the simulator shows there, by its index in `Recorded::truth`.
    pub truth: Vec<Option<usize>>,
}

/// One recorded look.
pub struct Recorded {
    pub cameras: Vec<Camera>,
    /// How many objects the executor that recorded the look measured.
    pub measured: usize,
    /// Per object that executor measured, its middle seen from above.
    pub centers: Vec<Vec2>,
    /// The names of the simulator's objects, sorted as the scorecard sorts them.
    pub truth: Vec<String>,
    /// Per one of them, its origin seen from above, as the first of the look's frames poses it.
    pub origins: Vec<Vec2>,
}

/// The `count` values of a `.npy` file of little-endian `f32`, `u16`, `u8` or `bool`, in its
/// order.
fn npy<const N: usize, T>(
    path: &Path,
    count: usize,
    of: fn([u8; N]) -> T,
) -> Result<Vec<T>, Error> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" || bytes[6] != 1 {
        return Err(format!("{}: not a version 1 .npy file", path.display()).into());
    }
    let header = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let data = bytes.get(header..).unwrap_or_default();
    if data.len() != N * count {
        let found = data.len();
        return Err(format!("{}: {found} bytes, not {count} of {N}", path.display()).into());
    }
    let (values, _) = data.as_chunks::<N>();
    Ok(values.iter().map(|&value| of(value)).collect())
}

/// A JSON array of numbers.
fn numbers(value: &Value) -> Vec<f32> {
    let values = value.as_array().into_iter().flatten();
    values
        .map(|v| v.as_f64().unwrap_or(f64::NAN) as f32)
        .collect()
}

/// A pose given as `position` and `rotation` (x, y, z, w).
fn pose(value: &Value) -> Result<Pose, Error> {
    let (p, r) = (numbers(&value["position"]), numbers(&value["rotation"]));
    let (&[x, y, z], &[i, j, k, w]) = (&p[..], &r[..]) else {
        return Err(format!("no pose of a position and a rotation: {value}").into());
    };
    Ok(Pose::new(Vec3::new(x, y, z), Quat::from_xyzw(i, j, k, w)))
}

fn json(path: &Path) -> Result<Value, Error> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// The truth object one of the simulator's bodies is part of, if any.
fn part(body: &Value) -> Option<&str> {
    body["object"].as_str().filter(|of| !of.is_empty())
}

impl Recorded {
    /// The look recorded in `folder`, with its truth.
    pub fn load(folder: &Path) -> Result<Self, Error> {
        let look = json(&folder.join("look.json"))?;
        let truth = json(&folder.join("truth.json"))?;
        let cameras = look["cameras"].as_array();
        let cameras = cameras.ok_or("look.json has no cameras")?;
        let objects = look["objects"].as_array();
        let objects = objects.ok_or("look.json has no objects")?;
        let measured = objects.len();
        let above = |value: &Value| match numbers(value)[..] {
            [x, y, _] => Ok(Vec2::new(x, y)),
            _ => Err(format!("no point of three coordinates: {value}")),
        };
        let centers = objects.iter().map(|object| above(&object["center"]));
        let centers = centers.collect::<Result<_, _>>()?;
        let bodies = truth["bodies"].as_array();
        let bodies = bodies.ok_or("truth.json has no bodies")?;
        let mut names: Vec<&str> = bodies.iter().filter_map(part).collect();
        names.sort_unstable();
        names.dedup();
        let frame = truth["frames"]
            .as_object()
            .and_then(|frames| frames.values().next());
        let poses = frame.and_then(|frame| frame["truth"]["objects"].as_array());
        let poses = poses.ok_or("truth.json has no frame with its objects' poses")?;
        let origins = names.iter().map(|name| {
            let pose = poses
                .iter()
                .find(|pose| pose["name"].as_str() == Some(name));
            let pose = pose.ok_or_else(|| format!("truth.json has no pose of {name}"))?;
            above(&pose["position"])
        });
        let origins = origins.collect::<Result<_, _>>()?;
        let object: Vec<Option<usize>> = (bodies.iter())
            .map(|body| names.iter().position(|name| Some(*name) == part(body)))
            .collect();
        let cameras = cameras.iter().map(|camera| {
            let name = camera["name"].as_str().ok_or("a camera without a name")?;
            let size = |side: &str| camera[side].as_u64().map(|pixels| pixels as usize);
            let (width, height) = size("width")
                .zip(size("height"))
                .ok_or("a camera without a width and a height")?;
            let [fx, fy, cx, cy] = numbers(&camera["intrinsics"])[..] else {
                return Err(format!("{name} has no intrinsics of fx, fy, cx, cy").into());
            };
            let [constant, linear, quadratic] = numbers(&camera["noise"])[..] else {
                return Err(format!("{name} has no noise of three coefficients").into());
            };
            let file = |what: &str| folder.join(format!("{what}_{name}.npy"));
            let pixels = width * height;
            let named = npy(&file("object"), pixels, u16::from_le_bytes)?;
            let beyond = |o: &&u16| **o != NOTHING && usize::from(**o) >= measured;
            if let Some(object) = named.iter().find(beyond) {
                return Err(format!("{name} names object {object} of {measured}").into());
            }
            let body = npy(&file("body"), pixels, u16::from_le_bytes)?;
            let beyond = |b: &&u16| **b != NOTHING && usize::from(**b) >= bodies.len();
            if let Some(body) = body.iter().find(beyond) {
                return Err(format!("{name} shows body {body} of {}", bodies.len()).into());
            }
            Ok(Camera {
                name: name.to_owned(),
                width,
                intrinsics: Intrinsics { fx, fy, cx, cy },
                pose: pose(camera)?,
                noise: Noise {
                    constant,
                    linear,
                    quadratic,
                },
                depth: npy(&file("depth"), pixels, f32::from_le_bytes)?,
                color: npy(&file("color"), 3 * pixels, u8::from_le_bytes)?,
                robot: npy(&file("robot"), pixels, |[flag]: [u8; 1]| flag != 0)?,
                measured: (named.iter())
                    .map(|&o| (o != NOTHING).then_some(usize::from(o)))
                    .collect(),
                truth: (body.iter())
                    .map(|&b| object.get(usize::from(b)).copied().flatten())
                    .collect(),
            })
        });
        Ok(Self {
            cameras: cameras.collect::<Result<_, Error>>()?,
            measured,
            centers,
            truth: names.into_iter().map(String::from).collect(),
            origins,
        })
    }
}

/// The fixtures of the run recorded in `folder`, in its cell's order, as the obstacles a world
/// keeps them as: what a look knows.
pub fn fixtures(folder: &Path) -> Result<Vec<Obstacle>, Error> {
    let cell = json(&folder.join("cell.json"))?;
    let fixtures = cell["fixtures"].as_array();
    let fixtures = fixtures.ok_or("cell.json has no fixtures")?.iter();
    let shapes = fixtures.map(|fixture| {
        let Pose {
            position: center,
            rotation,
        } = pose(&fixture["pose"])?;
        let (cuboid, cylinder) = (&fixture["shape"]["box"], &fixture["shape"]["cylinder"]);
        let half_extents = numbers(&cuboid["half_extents"]);
        let round = (cylinder["radius"].as_f64()).zip(cylinder["half_height"].as_f64());
        let shape = match (&half_extents[..], round) {
            (&[x, y, z], _) => Obstacle::Cuboid {
                center,
                half_extents: Vec3::new(x, y, z),
                rotation,
            },
            (_, Some((radius, half_height))) => Obstacle::Cylinder {
                center,
                rotation,
                radius: radius as f32,
                half_height: half_height as f32,
            },
            _ => return Err(format!("a fixture neither box nor cylinder: {fixture}").into()),
        };
        Ok(shape)
    });
    shapes.collect()
}

/// A device of `threads` threads, or of as many as the machine has, for looks to run on. A look
/// reads nothing of a device's robot, and a recording has none to give, so this one's is a single
/// joint.
pub fn device(threads: Option<usize>) -> Result<Device, Error> {
    let urdf = r#"<robot name="none"><link name="base"/><link name="tip"/>
        <joint name="joint" type="revolute"><parent link="base"/><child link="tip"/>
        <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint></robot>"#;
    let robot = Robot::from_urdf(urdf, &RobotOptions::default())?;
    Ok(match threads {
        Some(threads) => Device::cpu_threads(&robot, threads)?,
        None => Device::cpu(&robot)?,
    })
}
