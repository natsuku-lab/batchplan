//! What depth cameras show, as objects: which pixels are one thing, what shape it has and what
//! it rests on, and which object is which from one look to the next.
//!
//! miru is the perception [batchplan](https://github.com/natsuku-lab/batchplan) leaves out.
//! batchplan turns depth images into a collision world and plans on what the cameras saw; miru
//! says what the depth shows. It is built on batchplan, which does not depend on it: a view is
//! batchplan's [`DepthImage`](batchplan::DepthImage) with what a look needs besides, the shapes
//! the world is known to hold are its [`Obstacle`](batchplan::Obstacle)s, an object's solids
//! come back as obstacles, and a look runs on a [`Device`](batchplan::Device)'s threads.
//!
//! - **One call.** [`look`] takes every camera's posed depth and colour, where the robot is in
//!   each image and the shapes the world already knows, and returns a [`Scene`]: the objects,
//!   and for each pixel of each view what it shows ([`Label`]).
//! - **State in the result.** A scene is passed to the next look, which keeps each object's id
//!   while it stays where it was, and keeps what it cannot see as the scene had it. There is no
//!   handle of miru's to keep and nothing to configure at run time.
//! - **Geometry first.** Objects are told apart by where the depth jumps and where surfaces meet
//!   in a concave crease, at the scale the cameras' noise allows. No model is needed and no
//!   coordinates are ever asked of one. Where a caller has a mask model's instance masks, they
//!   come in as a number per pixel ([`View::masks`]) and join or divide what the depth alone
//!   cannot; miru runs no model.
//! - **The same on any machine's threads.** A scene does not depend on how many threads the
//!   device has: each piece of a look's work is one thread's alone, and the results are taken in
//!   the pieces' order.
//!
//! ```
//! use batchplan::glam::{Quat, Vec3};
//! use batchplan::{DepthImage, Device, Intrinsics, Obstacle, Pose};
//! use miru::{Label, Look, LookOptions, Noise, Scene, View};
//!
//! # let urdf = r#"<robot name="none"><link name="base"/><link name="tip"/>
//! #     <joint name="joint" type="revolute"><parent link="base"/><child link="tip"/>
//! #     <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint></robot>"#;
//! # let robot = batchplan::Robot::from_urdf(urdf, &batchplan::RobotOptions::default())?;
//! // The device the robot's motions are planned on: a look runs on its threads.
//! let device = Device::cpu(&robot)?;
//! // A camera 1 m over a table, looking straight down at it: from your driver.
//! let depth = vec![1.0; 64 * 48];
//! let view = View {
//!     image: DepthImage {
//!         depth: &depth,
//!         width: 64,
//!         intrinsics: Intrinsics { fx: 60.0, fy: 60.0, cx: 31.5, cy: 23.5 },
//!         camera: Pose::new(Vec3::new(0.5, 0.0, 1.0), Quat::from_rotation_x(std::f32::consts::PI)),
//!     },
//!     color: &[],
//!     // A stereo camera's, from its datasheet: 640 pixels' focal length, 5 cm between its
//!     // imagers, matching to 0.08 of a pixel; 2.5 mm at 1 m, growing with the depth squared.
//!     noise: Noise::stereo(640.0, 0.05, 0.08),
//!     robot: &[],
//!     // No mask model's numbers: geometry alone.
//!     masks: &[],
//! };
//! // The table is known, as the world the robot plans in has it: its top is the plane z = 0.
//! let table = Obstacle::cuboid(Vec3::new(0.5, 0.0, -0.02), Vec3::new(0.6, 0.6, 0.02));
//! let look = Look { views: &[view], known: &[table] };
//! let scene = miru::look(&device, &look, &Scene::default(), &LookOptions::default())?;
//! assert!(scene.objects.is_empty());
//! assert!(scene.labels[0].iter().all(|label| *label == Label::Known(0)));
//! # Ok::<(), batchplan::Error>(())
//! ```

// batchplan's rules: from Google's Rust style guide, documented public items, `#[must_use]` pure
// functions, and `expect` with its reason rather than `unwrap` outside tests; and internals
// `pub(crate)`, no `unsafe`, and no environment variables read (clippy.toml).
#![warn(missing_docs, missing_debug_implementations, unreachable_pub)]
#![deny(unsafe_code)]
#![warn(clippy::must_use_candidate, clippy::return_self_not_must_use)]
#![cfg_attr(not(test), warn(clippy::unwrap_used, clippy::disallowed_methods))]

/// Compiles the code in the README.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct Readme;

mod error;
mod fit;
mod look;
mod masks;
mod memory;
mod scene;
mod segment;
mod threads;
mod view;

pub use look::{Look, LookOptions, Noise, View, look};
pub use memory::{
    Camera, Kept, Memory, MemoryOptions, Observed, Of, Planned, Stance, Step, observe,
};
pub use scene::{Label, Object, Scene, Shape, Support};
