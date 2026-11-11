//! wgpu backend: one WGSL source runs on Vulkan (AMD, NVIDIA, Intel), Metal and DX12.

use std::any::Any;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use wgpu::util::DeviceExt;

use crate::constraint::{CONSTRAINT_WEIGHT, Region};
use crate::cpu::{self, CpuWorlds};
use crate::device::{Backend, CollisionWeights, Evaluation, Worlds};
use crate::edges::Search;
use crate::error::{Error, Result};
use crate::ik::IkOptions;
use crate::lanes::Lanes;
use crate::robot::{JointKind, Robot};
use crate::sdf::{self, DepthImage, FAR, Layout, Occlusion, Occupancy, RobotMask, SdfGrid};
use crate::trajopt::{LINE_SEARCH, MAX_HISTORY, PlanOptions};
use crate::types::{JointPaths, Pose};
use crate::world::{self, Obstacle, World};

const WORKGROUP: u32 = 64;
const _: () = assert!(
    LINE_SEARCH.len() <= 4,
    "traj_search prices the line-search steps in the components of one vec4"
);
/// Robots with at most this many pairs of links checked for self-collision get one block of
/// generated code per pair, with every frame in registers; more pairs loop at run time over frames
/// copied into arrays, so the code grows with links rather than pairs.
const UNROLLED_LINK_PAIRS: usize = 64;
/// Lanes per configuration in the collision kernels built for small batches (`LANES` in
/// kernels.wgsl).
const SHARED_LANES: u32 = 8;
/// Invocations a collision dispatch should reach: batches that reach it with one lane per
/// configuration use the single-lane kernels.
const LANE_TARGET: usize = 1 << 15;

/// Lanes per configuration for a dispatch of `configurations`.
fn lanes_for(configurations: usize) -> u32 {
    if configurations * (SHARED_LANES as usize) <= LANE_TARGET {
        SHARED_LANES
    } else {
        1
    }
}

/// IK configurations per collision sphere up to which IK that pushes away from obstacles keeps
/// `SHARED_LANES` lanes past the `lanes_for` bound. Splitting a configuration's checks pays up to
/// a batch size that grows with the robot's collision work: the crossover was about 26
/// configurations per sphere on a Radeon 8060S at 128 spheres and 32–40 at 507, and 34 to past 32
/// on an M4 Pro, so robots with many spheres keep their lanes past 4,096 configurations
/// (docs/benchmarks.md#how-the-speed-came).
const IK_SHARED_PER_SPHERE: usize = 25;

/// Lanes per configuration for IK that pushes away from obstacles, on `configurations` of a robot
/// with `spheres` collision spheres.
fn ik_lanes(configurations: usize, spheres: usize) -> u32 {
    if configurations <= IK_SHARED_PER_SPHERE * spheres {
        SHARED_LANES
    } else {
        lanes_for(configurations)
    }
}
/// Storage buffers in bind group 0: bindings 1..=9 are read-only, 10..=12 read-write (see
/// kernels.wgsl).
const READ_ONLY_STORAGE: u32 = 9;
const READ_WRITE_STORAGE: u32 = 3;
/// Upper bounds on work per queue submission, to stay clear of driver watchdogs.
const EVAL_CHUNK: usize = 1 << 18;
const IK_ITERS_PER_SUBMIT: u32 = 16;
/// Steps of IK per pass and per submission at most, however few items take them: a step of a
/// batch too small to fill the GPU takes as long as one of a batch that does.
const IK_MOST_STEPS: u32 = 64;
const IK_CHUNK: usize = 1 << 16;
/// The largest buffer kept for later calls, and how many are.
const SPARE_BYTES: u64 = 1 << 16;
const SPARE_BUFFERS: usize = 64;
const TRAJ_ITERS_PER_SUBMIT: u32 = 8;
const TRAJ_CHUNK: usize = 1 << 13;

/// A `vec4<f32>` on the shader side.
type Vec4 = [f32; 4];

/// Declares a struct shared with the kernels once: the `#[repr(C)]` Rust struct and its WGSL
/// declaration come from the same field list, so host and shader layouts cannot drift apart.
/// Field types must also be WGSL type names (`u32`, `i32`, `f32` or `Vec4`).
macro_rules! shader_struct {
    ($(#[$doc:meta])* $rust:ident => $wgsl:ident { $($field:ident: $ty:ident),* $(,)? }) => {
        $(#[$doc])*
        #[repr(C)]
        #[derive(Clone, Copy, Default, Pod, Zeroable)]
        struct $rust {
            $($field: $ty),*
        }

        impl $rust {
            const WGSL: &'static str =
                concat!("struct ", stringify!($wgsl), " {\n", $("    ", stringify!($field), ": ", stringify!($ty), ",\n",)* "}\n");
        }
    };
}

shader_struct! {
    /// Per-call constants (`P` in kernels.wgsl).
    GpuParams => Params {
        n_dof: u32, n_items: u32, points: u32, iterations: u32,
        w_world: f32, w_self: f32, margin: f32, self_margin: f32,
        acceleration_weight: f32, velocity_weight: f32, initial_step: f32, history: u32,
        damping: f32, rotation_weight: f32, max_step: f32, collision_step: f32,
        samples: u32, constraints: u32, constraint_weight: f32, held: u32,
        per_target: u32, first_seed: u32, kept: u32, chosen: u32,
        position_tolerance: f32, rotation_tolerance: f32, pad0: u32, pad1: u32,
    }
}

shader_struct! {
    /// One kinematic link: joint origin rotation columns and translation, and joint axis. A moving
    /// joint's value is its curve of `q[dof]`, `trans.w + axis.w x + curve.x x² + curve.y x³ +
    /// curve.z x⁴` (`x` itself unless it mimics another joint).
    /// `bound` is the sphere (center in the link frame, radius) around the link's collision spheres
    /// and their self-collision buffers, which are spheres
    /// `first_sphere..first_sphere + n_spheres`.
    /// The tree itself is written into the generated kernels (`robot_wgsl`).
    GpuLink => Link {
        c0: Vec4, c1: Vec4, c2: Vec4, trans: Vec4, axis: Vec4, curve: Vec4, bound: Vec4,
        first_sphere: u32, n_spheres: u32, pad0: u32, pad1: u32,
    }
}

shader_struct! {
    /// Collision sphere: center xyz and radius in `c`.
    GpuSphere => Sphere { c: Vec4, self_buf: f32, pad0: u32, pad1: u32, pad2: u32 }
}

shader_struct! {
    /// One step of building a grid (`G` in grids.wgsl). The grid: `origin` (w: the voxel size) and
    /// `nx * ny * nz = points` points. A depth image: the camera's position, the rows of its
    /// rotation's inverse (`to_camera*`) and its columns (`from_camera*`), the intrinsics (fx, fy,
    /// cx, cy), its size, whether robot depths are given and whether hidden space is occupied.
    /// `map` marks a map update. A transform pass runs along the `lines` of `axis`.
    GpuGridParams => GridParams {
        origin: Vec4, camera: Vec4, intrinsics: Vec4,
        to_camera0: Vec4, to_camera1: Vec4, to_camera2: Vec4,
        from_camera0: Vec4, from_camera1: Vec4, from_camera2: Vec4,
        nx: u32, ny: u32, nz: u32, points: u32,
        axis: u32, lines: u32, width: u32, height: u32,
        has_robot: u32, behind: u32, map: u32, spheres: u32,
    }
}

/// Obstacle kinds as stored in `GpuObstacle::center.w`.
const CUBOID: u32 = 0;
const SPHERE: u32 = 1;
const CYLINDER: u32 = 2;
const CAPSULE: u32 = 3;
const SDF: u32 = 4;

shader_struct! {
    /// `center.w` is the kind (`CUBOID`, `SPHERE`, ...). `half` holds the half extents of a cuboid,
    /// the radius (x) and half height or half length (y) of the round kinds, or a distance grid's
    /// origin (xyz) and voxel size (w). r0..r2 are the world-from-local rotation columns. A grid
    /// has `nx * ny * nz` points, two per `world_data` word from word `grid`. The links that may
    /// touch the obstacle are a bit set in `world_data` from word `r0.w` (a float, exact below
    /// 2^24, since touch sets come first): its length in words, then link `l` bit `l % 32` of the
    /// `l / 32`-th word. Word 0 holds no set, so `r0.w` is 0 when no link may touch the obstacle.
    GpuObstacle => Obstacle {
        center: Vec4, half: Vec4, r0: Vec4, r1: Vec4, r2: Vec4,
        nx: u32, ny: u32, nz: u32, grid: u32,
    }
}

/// Rounds of a loop written out at most: longer ones stay loops.
const WRITTEN_OUT: usize = 32;

/// `source` with its loops of a known count written out: the body of each
/// `for (var x = 0u; x < count; x++) {` once per round, with `x` its number, where `count` is a
/// number, the joints (`n`, `P.n_dof` or `MAX_DOF`: `dof`), `HANDS` (`hands`) or `LINE_STEPS`, up
/// to `WRITTEN_OUT`, and the body neither breaks nor continues. naga counts every loop's rounds
/// to rule out endless ones, and these loops are short and many: counting them costs IK a quarter
/// of its time. The source keeps its loops, to read beside `cpu.rs`.
fn loops_written_out(source: &str, dof: usize, hands: usize) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let count = |bound: &str| {
        let rounds = match bound {
            "n" | "P.n_dof" | "MAX_DOF" => dof,
            "HANDS" => hands,
            "LINE_STEPS" => LINE_SEARCH.len(),
            literal => literal.strip_suffix('u')?.parse().ok()?,
        };
        (rounds <= WRITTEN_OUT).then_some(rounds)
    };
    let mut out = String::with_capacity(source.len() * 2);
    write_out(&lines, &count, &mut out);
    out
}

/// The variable and rounds of a loop [`loops_written_out`] writes out, from its first line.
fn counted<'a>(line: &'a str, count: &dyn Fn(&str) -> Option<usize>) -> Option<(&'a str, usize)> {
    let rest = line.trim_start().strip_prefix("for (var ")?;
    let (variable, rest) = rest.split_once(" = 0u; ")?;
    let rest = rest.strip_prefix(variable)?.strip_prefix(" < ")?;
    let (bound, rest) = rest.split_once("; ")?;
    (rest == format!("{variable}++) {{")).then_some((variable, count(bound)?))
}

/// `text` with `name`, where it stands alone (no letter, digit, underscore or member dot beside
/// it), replaced by `value`, appended to `out`.
fn renamed(text: &str, name: &str, value: &str, out: &mut String) {
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut from = 0;
    while let Some(found) = text[from..].find(name) {
        let (start, end) = (from + found, from + found + name.len());
        let before = text[..start].chars().next_back();
        let alone = !before.is_some_and(|c| word(c) || c == '.')
            && !text[end..].chars().next().is_some_and(word);
        out.push_str(&text[from..start]);
        out.push_str(if alone { value } else { name });
        from = end;
    }
    out.push_str(&text[from..]);
}

/// [`loops_written_out`] for `lines`, appended to `out`; `count` gives a bound's rounds.
fn write_out(lines: &[&str], count: &dyn Fn(&str) -> Option<usize>, out: &mut String) {
    let depth = |line: &str| {
        let braces = |brace: char| line.chars().filter(|&c| c == brace).count() as isize;
        braces('{') - braces('}')
    };
    let mut at = 0;
    while at < lines.len() {
        let line = lines[at];
        at += 1;
        let Some((variable, rounds)) = counted(line, count) else {
            out.push_str(line);
            out.push('\n');
            continue;
        };
        // The body: the lines up to the brace that closes the loop.
        let (first, mut open) = (at, 1);
        while open > 0 {
            open += depth(lines[at]);
            at += 1;
        }
        let (body, close) = (&lines[first..at - 1], lines[at - 1]);
        let mut inner = String::new();
        write_out(body, count, &mut inner);
        let leaves = |l: &&str| l.contains("break;") || l.contains("continue;");
        if body.iter().any(leaves) {
            out.extend([line, "\n", &inner, close, "\n"]);
            continue;
        }
        let indent = &line[..line.len() - line.trim_start().len()];
        for round in 0..rounds {
            out.extend([indent, "{\n"]);
            renamed(&inner, variable, &format!("{round}u"), out);
            out.extend([indent, "}\n"]);
        }
    }
}

/// Spheres on a link of the robot's own, and sphere pairs of a pair of them, from which their
/// loops take two rounds at a time: naga counts a loop's rounds, and a link of many spheres has
/// many. No robot measured is slower from these counts than with one round at a time; from 16
/// and 64, cuRobo's Panda, whose hand has 18 spheres and 72 pairs with each of four links, ran
/// 1% slower, and from 24 and 64 a UR5e 1% (docs/benchmarks.md#how-the-speed-came). What a
/// robot holds is left out whatever its size, so that holding another object builds no kernels.
const PAIRED_SPHERES: u32 = 24;
const PAIRED_PAIRS: u32 = 128;

/// `source` with each `for (var x = a; x < b; x += LANES) {` taking two rounds at a time, the
/// second where it is still below `b`: the same rounds in the same order, half as many counted.
fn rounds_paired(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = String::with_capacity(source.len() * 2);
    let header = |line: &str| {
        let rest = line.trim_start().strip_prefix("for (var ")?;
        let (variable, rest) = rest.split_once(" = ")?;
        let (_, rest) = rest.split_once("; ")?;
        let rest = rest.strip_prefix(variable)?.strip_prefix(" < ")?;
        let (bound, rest) = rest.split_once("; ")?;
        let strided = rest == format!("{variable} += LANES) {{");
        strided.then(|| (variable.to_owned(), bound.to_owned()))
    };
    let mut at = 0;
    while at < lines.len() {
        let line = lines[at];
        at += 1;
        let Some((variable, bound)) = header(line) else {
            out.extend([line, "\n"]);
            continue;
        };
        // The body: the lines up to the brace that closes the loop.
        let (first, mut open) = (at, 1);
        while open > 0 {
            let braces = |brace: char| lines[at].matches(brace).count() as isize;
            open += braces('{') - braces('}');
            at += 1;
        }
        let body = lines[first..at - 1].join("\n") + "\n";
        let (head, close) = (line.trim_end_matches("LANES) {"), lines[at - 1]);
        let second = format!("({variable} + LANES)");
        out.extend([head, "2u * LANES) {\n{\n", &body, "}\n"]);
        out.extend(["if (", &second, " < ", &bound, ") {\n"]);
        renamed(&body, &variable, &second, &mut out);
        out.extend(["}\n", close, "\n"]);
    }
    out
}

/// One hand's damped least squares for a robot of `dof` joints, every sum written out term by
/// term: the loops they would be are short and many, and naga counts each loop's iterations to
/// rule out endless ones, which cost IK a quarter of its time. Mirrors, in `cpu.rs`, the columns
/// `hand_jacobian` leaves out, `hand_system`, `chol6`, `chol6_solve`, and `add_jt` as `ik_step`
/// uses it for the step and for the push, in their order of operations.
fn least_squares_wgsl(dof: usize) -> String {
    use std::fmt::Write;
    let jac = |r: usize, j: usize| format!("jac[{}u]", r * dof + j);
    let chol = |i: usize, j: usize| format!("chol[{}u]", i * 6 + j);
    let mut w = String::from(
        "\n// With joints held (P.held), zeroes the columns of jac `moving` leaves out: theirs.\nfn hold_columns() {\n    if (P.held == 1u) {\n",
    );
    for j in 0..dof {
        let _ = writeln!(w, "        if (!moving[{j}u]) {{");
        for r in 0..6 {
            let _ = writeln!(w, "            {} = 0.0;", jac(r, j));
        }
        w += "        }\n";
    }
    w += "    }\n}\n\n// Factors the damped normal matrix in `chol` in place (Cholesky, lower).\nfn chol6() {\n    var s: f32;\n";
    for i in 0..6 {
        for j in 0..=i {
            let _ = writeln!(w, "    s = {};", chol(i, j));
            for k in 0..j {
                let _ = writeln!(w, "    s -= {} * {};", chol(i, k), chol(j, k));
            }
            if i == j {
                let _ = writeln!(w, "    {} = sqrt(max(s, 1e-12));", chol(i, j));
            } else {
                let _ = writeln!(w, "    {} = s / {};", chol(i, j), chol(j, j));
            }
        }
    }
    w += "}\n\n// (J J^T + damping^2 I)^-1 b, by the factors in `chol`.\nfn chol6_solve(b: array<f32, 6>) -> array<f32, 6> {\n    var s: f32;\n";
    for i in 0..6 {
        let _ = writeln!(w, "    s = b[{i}u];");
        for k in 0..i {
            let _ = writeln!(w, "    s -= {} * y{k};", chol(i, k));
        }
        let _ = writeln!(w, "    let y{i} = s / {};", chol(i, i));
    }
    for i in (0..6).rev() {
        let _ = writeln!(w, "    s = y{i};");
        for k in i + 1..6 {
            let _ = writeln!(w, "    s -= {} * x{k};", chol(k, i));
        }
        let _ = writeln!(w, "    let x{i} = s / {};", chol(i, i));
    }
    w += "    return array<f32, 6>(x0, x1, x2, x3, x4, x5);\n}\n\n// Hand h's Jacobian into `jac`, and its damped normal matrix, factored, into `chol`.\nfn hand_system(h: u32) {\n    ee_jacobian(h);\n    hold_columns();\n    var s: f32;\n";
    for r in 0..6 {
        for c in 0..=r {
            w += "    s = 0.0;\n";
            for j in 0..dof {
                let _ = writeln!(w, "    s += {} * {};", jac(r, j), jac(c, j));
            }
            if r == c {
                w += "    s += P.damping * P.damping;\n";
            }
            let _ = writeln!(w, "    {} = s;\n    {} = s;", chol(r, c), chol(c, r));
        }
    }
    w += "    chol6();\n}\n\n// Hand h's step toward its target, J^T (J J^T + damping^2 I)^-1 e, into `dq`.\nfn pose_step(h: u32, dq: ptr<function, array<f32, MAX_DOF>>) {\n    let ep = goal_pos[h].xyz - ee_pos(h);\n    let eo = rot_log(goal_rot[h] * transpose(ee_rot(h))).xyz * P.rotation_weight;\n    hand_system(h);\n    let y = chol6_solve(array<f32, 6>(ep.x, ep.y, ep.z, eo.x, eo.y, eo.z));\n    var s: f32;\n";
    let transposed = |w: &mut String, x: &str, out: &str| {
        for j in 0..dof {
            w.push_str("    s = 0.0;\n");
            for r in 0..6 {
                let _ = writeln!(w, "    s += {} * {x}[{r}u];", jac(r, j));
            }
            let _ = writeln!(w, "    (*{out})[{j}u] += s;");
        }
    };
    transposed(&mut w, "y", "dq");
    w += "}\n\n// J^T (J J^T + damping^2 I)^-1 J grad for the system in jac and chol, into `push`.\nfn null_part(push: ptr<function, array<f32, MAX_DOF>>) {\n    var s: f32;\n    var jg: array<f32, 6>;\n";
    for r in 0..6 {
        w += "    s = 0.0;\n";
        for j in 0..dof {
            let _ = writeln!(w, "    s += {} * grad[{j}u];", jac(r, j));
        }
        let _ = writeln!(w, "    jg[{r}u] = s;");
    }
    w += "    let z = chol6_solve(jg);\n";
    transposed(&mut w, "z", "push");
    w += "}\n";
    w
}

/// The robot-specific part of the kernels: forward kinematics, the end effector's Jacobian and
/// the collision cost, written out link by link. Each link's frame (`rot_i`, `pos_i`) and collision
/// wrench (`force_i`, `moment_i`, about the world origin) is its own private variable indexed only
/// by constants, so it stays in registers; private arrays indexed at run time end up in scratch
/// memory on some drivers. Sphere and sphere-pair ranges are read from the buffers, so robots that
/// differ only in the spheres of what they hold share kernels, as do robots whose own links'
/// spheres stay on one side of `PAIRED_SPHERES` and `PAIRED_PAIRS`.
fn robot_wgsl(robot: &Robot) -> String {
    use std::fmt::Write;
    let links = &robot.links;
    let dof = robot.dof();
    let mut w = format!(
        "const MAX_DOF: u32 = {dof}u;\nconst JAC_LEN: u32 = {}u;\n",
        6 * dof
    );
    // Joint motion per unit of the actuated joint: the multiplier for a linear curve, the curve's
    // slope at the current configuration (computed by fk) for others.
    let nonlinear = |i: usize| {
        links[i]
            .joint
            .actuation()
            .is_some_and(|(_, c)| !c.is_linear())
    };
    let rate = |i: usize| {
        if nonlinear(i) {
            format!("rate_{i}")
        } else {
            format!("links[{i}u].axis.w")
        }
    };
    for i in 0..links.len() {
        let _ = writeln!(
            w,
            "var<private> rot_{i}: mat3x3<f32>;\nvar<private> pos_{i}: vec3<f32>;"
        );
        let _ = writeln!(
            w,
            "var<private> force_{i}: vec3<f32>;\nvar<private> moment_{i}: vec3<f32>;"
        );
        if nonlinear(i) {
            let _ = writeln!(w, "var<private> rate_{i}: f32;");
        }
    }

    w += "\n// Every link's frame, root first.\nfn fk() {\n";
    for (i, l) in links.iter().enumerate() {
        let (prot, ppos) = l.parent.map_or((String::new(), String::new()), |p| {
            (format!("rot_{p} * "), format!(" + pos_{p}"))
        });
        let _ = writeln!(w, "    {{\n        let l = links[{i}u];");
        let _ = writeln!(
            w,
            "        let jrot = {prot}mat3x3<f32>(l.c0.xyz, l.c1.xyz, l.c2.xyz);"
        );
        let _ = writeln!(w, "        let jpos = {prot}l.trans.xyz{ppos};");
        if let Some((dof, _)) = l.joint.actuation() {
            if nonlinear(i) {
                let _ = writeln!(w, "        let x = q[{dof}u];");
                w += "        let v = l.trans.w + x * (l.axis.w + x * (l.curve.x + x * (l.curve.y + x * l.curve.z)));\n";
                let _ = writeln!(
                    w,
                    "        rate_{i} = l.axis.w + x * (2.0 * l.curve.x + x * (3.0 * l.curve.y + x * 4.0 * l.curve.z));"
                );
            } else {
                let _ = writeln!(w, "        let v = l.axis.w * q[{dof}u] + l.trans.w;");
            }
        }
        let _ = match l.joint {
            JointKind::Revolute { .. } => {
                writeln!(
                    w,
                    "        rot_{i} = jrot * rodrigues(l.axis.xyz, v);\n        pos_{i} = jpos;"
                )
            }
            JointKind::Prismatic { .. } => {
                writeln!(
                    w,
                    "        rot_{i} = jrot;\n        pos_{i} = jpos + jrot * l.axis.xyz * v;"
                )
            }
            JointKind::Fixed => writeln!(w, "        rot_{i} = jrot;\n        pos_{i} = jpos;"),
        };
        w += "    }\n";
    }
    w += "}\n";

    // Hand h's frame, Jacobian and joints, each a switch with hand 0 as the default case.
    let hands = &robot.ee_links;
    let _ = writeln!(w, "\nconst HANDS: u32 = {}u;", hands.len());
    // IK solves hands that share joints as one system (`ik_step`); its buffers stay a single
    // value for other robots, whose kernels never touch them.
    let stacked = robot.hands_share_joints();
    let stack = if stacked { 6 * hands.len() } else { 6 };
    let _ = writeln!(
        w,
        "const STACKED: bool = {stacked};\nconst STACK: u32 = {stack}u;\nconst STACK_JAC: u32 = {}u;\nconst STACK_CHOL: u32 = {}u;",
        if stacked { stack * dof } else { 1 },
        if stacked { stack * stack } else { 1 },
    );
    let cases = |w: &mut String, body: &dyn Fn(&mut String, usize)| {
        w.push_str("    switch h {\n");
        for (h, &ee) in hands.iter().enumerate().skip(1) {
            let _ = writeln!(w, "        case {h}u: {{");
            body(w, ee);
            w.push_str("        }\n");
        }
        w.push_str("        default: {\n");
        body(w, hands[0]);
        w.push_str("        }\n    }\n");
    };
    w += "\nfn ee_pos(h: u32) -> vec3<f32> {\n";
    cases(&mut w, &|w, ee| {
        let _ = writeln!(w, "            return pos_{ee};");
    });
    w += "}\n\nfn ee_rot(h: u32) -> mat3x3<f32> {\n";
    cases(&mut w, &|w, ee| {
        let _ = writeln!(w, "            return rot_{ee};");
    });
    w += "}\n\n// Hand h's Jacobian into `jac` (6 x MAX_DOF, row-major; rotation rows scaled by\n";
    w += "// P.rotation_weight), its moving joints root first, mimic joints adding into their leader.\n";
    w += "fn ee_jacobian(h: u32) {\n";
    for k in 0..6 * dof {
        let _ = writeln!(w, "    jac[{k}u] = 0.0;");
    }
    cases(&mut w, &|w, ee| {
        for (i, d) in robot.chain(ee) {
            let _ = writeln!(
                w,
                "            {{\n            let m = {};\n            let a = rot_{i} * links[{i}u].axis.xyz;",
                rate(i)
            );
            if matches!(links[i].joint, JointKind::Prismatic { .. }) {
                w.push_str("            let jp = m * a;\n            let jo = vec3<f32>(0.0);\n");
            } else {
                let _ = writeln!(
                    w,
                    "            let jp = m * cross(a, pos_{ee} - pos_{i});\n            let jo = m * a * P.rotation_weight;"
                );
            }
            for (r, c) in ["jp.x", "jp.y", "jp.z", "jo.x", "jo.y", "jo.z"]
                .iter()
                .enumerate()
            {
                let _ = writeln!(w, "            jac[{r}u * MAX_DOF + {d}u] += {c};");
            }
            w.push_str("            }\n");
        }
    });
    w += "}\n";
    w += &least_squares_wgsl(dof);
    w += "\n// Marks in `moving` the joints hand h moves.\nfn mark_moving(h: u32) {\n";
    cases(&mut w, &|w, ee| {
        for (_, d) in robot.chain(ee) {
            let _ = writeln!(w, "            moving[{d}u] = true;");
        }
    });
    w += "}\n";

    w += r"
// Collision cost of the configuration last passed to fk(); with `gradient`, adds d(cost)/dq into
// `grad`. Returns (cost, world clearance, self clearance). Callers pass `gradient` as a constant,
// so the compiler drops the gradient work from cost-only kernels. Lane `lane` of LANES takes every
// LANES-th sphere of a link and every LANES-th sphere pair of a pair of links: all lanes run the
// same code on different spheres. Each returns its part; group_collision and group_grad combine
// the lanes.
fn collision(
    world: u32,
    w_world: f32,
    w_self: f32,
    margin: f32,
    self_margin: f32,
    gradient: bool,
    lane: u32,
) -> vec3<f32> {
    var cost = 0.0;
    var wmin = FAR;
    var smin = FAR;
    var touched = false;
    let range = world_ranges[world];
    if (gradient) {
";
    for i in 0..links.len() {
        let _ = writeln!(
            w,
            "        force_{i} = vec3<f32>(0.0);\n        moment_{i} = vec3<f32>(0.0);"
        );
    }
    w += "    }\n";
    w += "    // An obstacle farther from a link's bounding sphere than the margin cannot add cost through\n";
    w += "    // the link's spheres; the gap bounds their clearance from below. Distance grids are\n";
    w += "    // interpolated, not exact, so their spheres are always checked. Links no joint moves are not\n";
    w += "    // checked against the world (see `collision` in cpu.rs).\n";
    w += "    let world_gate = max(margin, 0.0);\n";
    let checked = (0..links.len())
        .filter(|&i| robot.sphere_ranges[i][1] > 0 && !robot.links[i].chain.is_empty());
    for i in checked {
        // The link's word and bit in a touch set.
        let (word, bit) = (i / 32, 1u32 << (i % 32));
        let mut block = String::new();
        let _ = write!(
            block,
            r"    {{
        let bl = links[{i}u].bound;
        let bc = rot_{i} * bl.xyz + pos_{i};
        for (var k = 0u; k < range.y; k++) {{
            let o = obstacles[range.x + k];
            let touch = u32(o.r0.w);
            if (touch != 0u && {word}u < world_data[touch]
                && (world_data[touch + {word}u + 1u] & {bit}u) != 0u) {{
                continue;
            }}
            if (u32(o.center.w) != SDF) {{
                let gap = obstacle_distance(o, bc).w - bl.w;
                if (gap > world_gate) {{
                    wmin = min(wmin, gap);
                    continue;
                }}
            }}
            let first = links[{i}u].first_sphere;
            for (var s = first + lane; s < first + links[{i}u].n_spheres; s += LANES) {{
                let sp = spheres[s];
                let c = rot_{i} * sp.c.xyz + pos_{i};
                let dg = obstacle_distance(o, c);
                let d = dg.w - sp.c.w;
                wmin = min(wmin, d);
                let pen = margin - d;
                if (pen > 0.0) {{
                    cost += w_world * pen * pen;
                    if (gradient) {{
                        let f = -2.0 * w_world * pen * dg.xyz;
                        force_{i} += f;
                        moment_{i} += cross(c, f);
                        touched = true;
                    }}
                }}
            }}
        }}
    }}
"
        );
        if i < robot.own_links() && robot.sphere_ranges[i][1] >= PAIRED_SPHERES {
            block = rounds_paired(&block);
        }
        w += &block;
    }
    w += "    // Likewise for link pairs whose bounding spheres are farther apart than the self margin.\n";
    w += "    let gate = max(self_margin, 0.0);\n";
    // One link pair: `ends` index its two links in the links buffer; `frame(e)` names end e's
    // rotation and position, `wrench(e)` its force and moment accumulators.
    type Names<'names> = &'names dyn Fn(usize) -> (String, String);
    let pair = |l: &str, ends: [&str; 2], frame: Names, wrench: Names| {
        let [a, b] = ends;
        let ((ra, pa), (rb, pb)) = (frame(0), frame(1));
        let ((fa, ma), (fb, mb)) = (wrench(0), wrench(1));
        format!(
            r"        let ba = links[{a}].bound;
        let bb = links[{b}].bound;
        let gap = length({ra} * ba.xyz + {pa} - {rb} * bb.xyz - {pb}) - ba.w - bb.w;
        if (gap > gate) {{
            smin = min(smin, gap);
        }} else {{
            let span = pairs[2u * {l} + 1u];
            for (var k = span.x + lane; k < span.x + span.y; k += LANES) {{
                let pr = pairs[k];
                let sa = spheres[pr.x];
                let sb = spheres[pr.y];
                let ca = {ra} * sa.c.xyz + {pa};
                let cb = {rb} * sb.c.xyz + {pb};
                let diff = ca - cb;
                let dist = length(diff);
                let d = dist - (sa.c.w + sa.self_buf) - (sb.c.w + sb.self_buf);
                smin = min(smin, d);
                let pen = self_margin - d;
                if (pen > 0.0) {{
                    cost += w_self * pen * pen;
                    if (gradient) {{
                        var u = vec3<f32>(1.0, 0.0, 0.0);
                        if (dist > 1e-9) {{
                            u = diff / dist;
                        }}
                        let g = 2.0 * w_self * pen * u;
                        {fa} -= g;
                        {ma} -= cross(ca, g);
                        {fb} += g;
                        {mb} += cross(cb, g);
                        touched = true;
                    }}
                }}
            }}
        }}
"
        )
    };
    let pairs = robot.self_link_pairs.len();
    if pairs <= UNROLLED_LINK_PAIRS {
        for (l, lp) in robot.self_link_pairs.iter().enumerate() {
            let link = [lp.a, lp.b];
            let frame = |e: usize| (format!("rot_{}", link[e]), format!("pos_{}", link[e]));
            let wrench = |e: usize| (format!("force_{}", link[e]), format!("moment_{}", link[e]));
            let mut body = pair(
                &format!("{l}u"),
                [&format!("{}u", lp.a), &format!("{}u", lp.b)],
                &frame,
                &wrench,
            );
            let own = (lp.a.max(lp.b) as usize) < robot.own_links();
            if own && lp.count >= PAIRED_PAIRS {
                body = rounds_paired(&body);
            }
            let _ = write!(w, "    {{\n{body}    }}\n");
        }
    } else {
        // Too many pairs to write out: frames go into arrays indexed at run time.
        let nl = links.len();
        let _ = writeln!(
            w,
            "    var frot: array<mat3x3<f32>, {nl}>;\n    var fpos: array<vec3<f32>, {nl}>;"
        );
        let _ = writeln!(
            w,
            "    var fforce: array<vec3<f32>, {nl}>;\n    var fmoment: array<vec3<f32>, {nl}>;"
        );
        for i in 0..nl {
            let _ = writeln!(w, "    frot[{i}u] = rot_{i};\n    fpos[{i}u] = pos_{i};");
        }
        let end = ["ab.x", "ab.y"];
        let frame = |e: usize| (format!("frot[{}]", end[e]), format!("fpos[{}]", end[e]));
        let wrench = |e: usize| {
            (
                format!("fforce[{}]", end[e]),
                format!("fmoment[{}]", end[e]),
            )
        };
        let body = pair("l", end, &frame, &wrench);
        let _ = write!(
            w,
            "    for (var l = 0u; l < {pairs}u; l++) {{\n        let ab = pairs[2u * l];\n{body}    }}\n"
        );
        w += "    if (gradient) {\n";
        for i in 0..nl {
            let _ = writeln!(
                w,
                "        force_{i} += fforce[{i}u];\n        moment_{i} += fmoment[{i}u];"
            );
        }
        w += "    }\n";
    }
    w += "    // Each link's wrench reaches the joints at and above it, leaves first.\n";
    w += "    if (gradient && touched) {\n";
    for (i, l) in links.iter().enumerate().rev() {
        let _ = match l.joint {
            JointKind::Revolute { dof, .. } => writeln!(
                w,
                "        grad[{dof}u] += {} * dot(rot_{i} * links[{i}u].axis.xyz, moment_{i} - cross(pos_{i}, force_{i}));",
                rate(i)
            ),
            JointKind::Prismatic { dof, .. } => {
                writeln!(
                    w,
                    "        grad[{dof}u] += {} * dot(rot_{i} * links[{i}u].axis.xyz, force_{i});",
                    rate(i)
                )
            }
            JointKind::Fixed => Ok(()),
        };
        if let Some(p) = l.parent {
            let _ = writeln!(
                w,
                "        force_{p} += force_{i};\n        moment_{p} += moment_{i};"
            );
        }
    }
    w += "    }\n    return vec3<f32>(cost, wmin, smin);\n}\n";
    w
}

fn v4(v: Vec3, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

#[derive(Clone)]
pub(crate) struct GpuBackend {
    robot: Robot,
    info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    kernels: Arc<Kernels>,
    /// Kernels built on this device, by their robot-specific source. Every backend derived with
    /// `with_robot` shares it, so attaching and detaching objects compiles each robot shape once.
    built: Arc<Mutex<HashMap<String, Arc<Kernels>>>>,
    buffers: RobotBuffers,
    /// The grid kernels, built on first use and shared like `built`.
    grids: Arc<OnceLock<GridKernels>>,
    grid_buffers: Arc<Mutex<GridBuffers>>,
    /// The robot prepared for the CPU's SIMD collision checks, which answer searches' edge checks.
    lanes: Arc<Lanes>,
    /// The device's CPU work (searches, preparing depth images) runs here, never on rayon's global
    /// pool, so a host's own parallel work cannot hold it past a time budget.
    pool: Arc<ThreadPool>,
    /// Small buffers of calls that are over, for the next small call: making a dozen costs a
    /// small IK call more than filling them. Shared by the backends `with_robot` derives.
    spare: Arc<Mutex<Vec<wgpu::Buffer>>>,
}

/// Building distance grids (grids.wgsl), which needs no robot.
struct GridKernels {
    layout: wgpu::BindGroupLayout,
    depth_occupancy: wgpu::ComputePipeline,
    depth_hits: wgpu::ComputePipeline,
    map_update: wgpu::ComputePipeline,
    edt_pass: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
}

impl GridKernels {
    fn new(device: &wgpu::Device, info: &wgpu::AdapterInfo) -> Result<Self> {
        use crate::sdf::{CLAMP, HIT, MISS, SHADOWED, UNOBSERVED};
        let source = [
            "alias Vec4 = vec4<f32>;\n",
            &format!("const WORKGROUP: u32 = {WORKGROUP}u;\n"),
            &format!(
                "const HIT: i32 = {HIT};\nconst MISS: i32 = {MISS};\nconst UNOBSERVED: i32 = {UNOBSERVED};\nconst SHADOWED: i32 = {SHADOWED};\n"
            ),
            &format!("const CLAMP_LO: i32 = {};\nconst CLAMP_HI: i32 = {};\n", CLAMP.0, CLAMP.1),
            GpuGridParams::WGSL,
            include_str!("grids.wgsl"),
        ]
        .concat();
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grids"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![entry(0, wgpu::BufferBindingType::Uniform)];
        entries.extend(
            (1..=2).map(|b| entry(b, wgpu::BufferBindingType::Storage { read_only: true })),
        );
        entries.extend(
            (3..=9).map(|b| entry(b, wgpu::BufferBindingType::Storage { read_only: false })),
        );
        entries.push(entry(
            10,
            wgpu::BufferBindingType::Storage { read_only: true },
        ));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grids"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let kernels = Self {
            depth_occupancy: pipeline("depth_occupancy"),
            depth_hits: pipeline("depth_hits"),
            map_update: pipeline("map_update"),
            edt_pass: pipeline("edt_pass"),
            finish: pipeline("finish"),
            layout,
        };
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(Error::Gpu(format!(
                "{} ({:?}) cannot build the grid kernels: {e}",
                info.name, info.backend
            )));
        }
        Ok(kernels)
    }
}

/// Buffers for building grids, kept for the next build: wgpu zero-fills new buffers, which took
/// longer than the distance transform itself. Each grows to the largest size asked for.
#[derive(Default)]
struct GridBuffers(HashMap<&'static str, wgpu::Buffer>);

impl GridBuffers {
    /// Bindings 1 to 10 of grids.wgsl, for grids of `points` points, images of `pixels` pixels,
    /// robot masks of `robot` words and finishing tables of `table` entries.
    fn bindings(
        &mut self,
        device: &wgpu::Device,
        points: usize,
        pixels: usize,
        robot: usize,
        table: usize,
    ) -> [&wgpu::Buffer; 10] {
        let words = [
            ("depth", pixels),
            ("robot", robot),
            ("bits", points.div_ceil(32)),
            ("to solid", points),
            ("to free", points),
            ("next solid", points),
            ("next free", points),
            ("envelope", points),
            ("log-odds", points.div_ceil(4)),
            ("finishing table", table),
        ];
        for (name, len) in words {
            if self
                .0
                .get(name)
                .is_none_or(|b| b.size() < (len.max(4) * 4) as u64)
            {
                self.0.insert(name, storage_zeroed(device, name, len));
            }
        }
        words.map(|(name, _)| &self.0[name])
    }
}

/// The compiled kernels for one robot shape.
struct Kernels {
    layout0: wgpu::BindGroupLayout,
    /// One lane per configuration, for large batches and IK without the push (`ik_lanes`).
    single: CollisionKernels,
    /// SHARED_LANES per configuration, for batches too small to fill the GPU with one, and IK of
    /// robots with many spheres (`ik_lanes`).
    shared: CollisionKernels,
    ik_select: wgpu::ComputePipeline,
    traj_search: wgpu::ComputePipeline,
    traj_grad: wgpu::ComputePipeline,
    lbfgs_direction: wgpu::ComputePipeline,
    /// The module and layout every pipeline is built from, for those built when first needed.
    module: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    /// The trajectory kernels that keep constraints, for one lane and SHARED_LANES, built the
    /// first time a constrained path is optimized.
    constrained: OnceLock<[TrajKernels; 2]>,
}

/// The per-sample trajectory kernels, built to keep constraints.
struct TrajKernels {
    traj_costs: wgpu::ComputePipeline,
    traj_samples: wgpu::ComputePipeline,
}

/// The kernels that check collision, built for one number of lanes per configuration.
struct CollisionKernels {
    evaluate: wgpu::ComputePipeline,
    clearance: wgpu::ComputePipeline,
    ik: wgpu::ComputePipeline,
    traj_costs: wgpu::ComputePipeline,
    traj_samples: wgpu::ComputePipeline,
}

impl Kernels {
    /// The collision kernels for `lanes` (from `lanes_for` or `ik_lanes`).
    fn collision(&self, lanes: u32) -> &CollisionKernels {
        if lanes == 1 {
            &self.single
        } else {
            &self.shared
        }
    }

    /// The per-sample trajectory kernels for `lanes`: costs, then gradients, keeping constraints
    /// with `constrained`.
    fn traj(
        &self,
        device: &wgpu::Device,
        info: &wgpu::AdapterInfo,
        lanes: u32,
        constrained: bool,
    ) -> Result<[&wgpu::ComputePipeline; 2]> {
        if !constrained {
            let c = self.collision(lanes);
            return Ok([&c.traj_costs, &c.traj_samples]);
        }
        let built = match self.constrained.get() {
            Some(built) => built,
            None => {
                let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
                let traj = |lanes: u32| TrajKernels {
                    traj_costs: pipeline(
                        device,
                        &self.layout,
                        &self.module,
                        "traj_costs",
                        lanes,
                        true,
                    ),
                    traj_samples: pipeline(
                        device,
                        &self.layout,
                        &self.module,
                        "traj_samples",
                        lanes,
                        true,
                    ),
                };
                let built = [traj(1), traj(SHARED_LANES)];
                if let Some(e) = pollster::block_on(scope.pop()) {
                    return Err(Error::Gpu(format!(
                        "{} ({:?}) cannot build the constrained trajectory kernels: {e}",
                        info.name, info.backend
                    )));
                }
                self.constrained.get_or_init(|| built)
            }
        };
        let t = &built[usize::from(lanes != 1)];
        Ok([&t.traj_costs, &t.traj_samples])
    }
}

/// The pipeline for kernel `entry` with `lanes` per configuration, keeping constraints with
/// `constrained` (`LANES`, `CONSTRAINED` in kernels.wgsl).
fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
    lanes: u32,
    constrained: bool,
) -> wgpu::ComputePipeline {
    let lanes = ("LANES", f64::from(lanes));
    let constants = [lanes, ("CONSTRAINED", 1.0)];
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(layout),
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: if constrained {
                &constants
            } else {
                &constants[..1]
            },
            // Every kernel writes its workgroup memory before reading it.
            zero_initialize_workgroup_memory: false,
        },
        cache: None,
    })
}

/// The robot as the kernels read it.
#[derive(Clone)]
struct RobotBuffers {
    links: wgpu::Buffer,
    spheres: wgpu::Buffer,
    pairs: wgpu::Buffer,
    limits: wgpu::Buffer,
}

impl RobotBuffers {
    fn new(device: &wgpu::Device, robot: &Robot) -> Self {
        let gpu_links: Vec<GpuLink> = robot
            .links
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let (axis, c) = match l.joint {
                    JointKind::Fixed => (Vec3::ZERO, [0.0; 5]),
                    JointKind::Revolute { axis, curve, .. }
                    | JointKind::Prismatic { axis, curve, .. } => (axis, curve.0),
                };
                GpuLink {
                    c0: v4(l.origin.rot.x_axis, 0.0),
                    c1: v4(l.origin.rot.y_axis, 0.0),
                    c2: v4(l.origin.rot.z_axis, 0.0),
                    trans: v4(l.origin.trans, c[0]),
                    axis: v4(axis, c[1]),
                    curve: [c[2], c[3], c[4], 0.0],
                    bound: robot.link_bounds[i],
                    first_sphere: robot.sphere_ranges[i][0],
                    n_spheres: robot.sphere_ranges[i][1],
                    ..Default::default()
                }
            })
            .collect();
        let gpu_spheres: Vec<GpuSphere> = robot
            .spheres
            .iter()
            .map(|s| GpuSphere {
                c: v4(s.center, s.radius),
                self_buf: s.self_buffer,
                ..Default::default()
            })
            .collect();
        // Each link pair as (a, b), then (first, count) of the sphere pairs that follow.
        let skip = 2 * robot.self_link_pairs.len() as u32;
        let gpu_pairs: Vec<[u32; 2]> = robot
            .self_link_pairs
            .iter()
            .flat_map(|lp| [[lp.a, lp.b], [skip + lp.first, lp.count]])
            .chain(robot.self_pairs.iter().copied())
            .collect();
        // Continuous joints are unbounded: clamping to infinite limits leaves them as they are.
        let gpu_limits: Vec<[f32; 2]> = (0..robot.dof()).map(|j| robot.bounds(j).into()).collect();

        Self {
            links: storage(device, "links", &gpu_links),
            spheres: storage(device, "spheres", &gpu_spheres),
            pairs: storage(device, "pairs", &gpu_pairs),
            limits: storage(device, "limits", &gpu_limits),
        }
    }
}

/// The kernels for `robot`'s shape, built on `device` unless `built` already holds them.
fn kernels_for(
    device: &wgpu::Device,
    info: &wgpu::AdapterInfo,
    built: &Mutex<HashMap<String, Arc<Kernels>>>,
    robot: &Robot,
) -> Result<Arc<Kernels>> {
    let robot_source = robot_wgsl(robot);
    let mut built = built
        .lock()
        .expect("no thread panics while holding the kernel cache");
    if let Some(kernels) = built.get(&robot_source) {
        return Ok(kernels.clone());
    }
    let counts = (robot.dof(), robot.hands());
    let kernels = Arc::new(build_kernels(device, info, &robot_source, counts)?);
    built.insert(robot_source, kernels.clone());
    Ok(kernels)
}

fn build_kernels(
    device: &wgpu::Device,
    info: &wgpu::AdapterInfo,
    robot_source: &str,
    (dof, hands): (usize, usize),
) -> Result<Kernels> {
    // Constants and shared structs are generated from their Rust definitions.
    let source = [
        "alias Vec4 = vec4<f32>;\n",
        &format!("const CUBOID: u32 = {CUBOID}u;\nconst SPHERE: u32 = {SPHERE}u;\n"),
        &format!("const CYLINDER: u32 = {CYLINDER}u;\nconst CAPSULE: u32 = {CAPSULE}u;\nconst SDF: u32 = {SDF}u;\n"),
        &format!("const WORKGROUP: u32 = {WORKGROUP}u;\n"),
        &format!("const IK_REST: f32 = {:?};\n", cpu::IK_REST),
        &format!(
            "const MAX_HISTORY: u32 = {MAX_HISTORY}u;\nconst LINE_STEPS: u32 = {}u;\n\
             fn line_search(c: u32) -> f32 {{\n    var steps = array<f32, {}>({});\n    return steps[c];\n}}\n",
            LINE_SEARCH.len(),
            LINE_SEARCH.len(),
            LINE_SEARCH.map(|a| format!("{a:?}")).join(", ")
        ),
        GpuParams::WGSL,
        GpuLink::WGSL,
        GpuSphere::WGSL,
        GpuObstacle::WGSL,
        robot_source,
        &loops_written_out(include_str!("kernels.wgsl"), dof, hands),
        include_str!("grid.wgsl"),
    ]
    .concat();
    // Turn shader and pipeline validation failures into errors instead of panics.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let descriptor = wgpu::ShaderModuleDescriptor {
        label: Some("batchplan kernels"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    };
    let module = device.create_shader_module(descriptor);

    let buffer_entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = vec![buffer_entry(0, wgpu::BufferBindingType::Uniform)];
    let (ro, rw) = (
        1..=READ_ONLY_STORAGE,
        READ_ONLY_STORAGE + 1..=READ_ONLY_STORAGE + READ_WRITE_STORAGE,
    );
    entries
        .extend(ro.map(|b| buffer_entry(b, wgpu::BufferBindingType::Storage { read_only: true })));
    entries
        .extend(rw.map(|b| buffer_entry(b, wgpu::BufferBindingType::Storage { read_only: false })));
    let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("main"),
        entries: &entries,
    });
    let pl_main = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout0)],
        immediate_size: 0,
    });
    let pipeline =
        |entry: &str, lanes: u32| pipeline(device, &pl_main, &module, entry, lanes, false);
    let collision = |lanes: u32| CollisionKernels {
        evaluate: pipeline("evaluate_main", lanes),
        clearance: pipeline("clearance_main", lanes),
        ik: pipeline("ik_main", lanes),
        traj_costs: pipeline("traj_costs", lanes),
        traj_samples: pipeline("traj_samples", lanes),
    };
    // Drivers compile each pipeline on the calling thread; the two sets compile side by side.
    let (single, shared) = std::thread::scope(|s| {
        let shared = s.spawn(|| collision(SHARED_LANES));
        (
            collision(1),
            shared.join().expect("pipeline creation does not panic"),
        )
    });
    let ik_select = pipeline("ik_select", 1);
    let traj_search = pipeline("traj_search", 1);
    let traj_grad = pipeline("traj_grad", 1);
    let lbfgs_direction = pipeline("lbfgs_direction", 1);
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(Error::Gpu(format!(
            "{} ({:?}) cannot build the kernels: {e}",
            info.name, info.backend
        )));
    }

    Ok(Kernels {
        layout0,
        single,
        shared,
        ik_select,
        traj_search,
        traj_grad,
        lbfgs_direction,
        module,
        layout: pl_main,
        constrained: OnceLock::new(),
    })
}

/// Worlds in device memory: every world's obstacles, each world's `[first, count]` range of them,
/// and the distance grids they use, each stored once.
struct GpuWorlds {
    obstacles: wgpu::Buffer,
    ranges: wgpu::Buffer,
    /// Distance grids and touch sets (`GpuObstacle`).
    data: wgpu::Buffer,
    /// The worlds as the CPU checks them, for searches.
    cpu: CpuWorlds,
}

/// Per-call buffers bound alongside the robot buffers.
struct CallBuffers<'call> {
    params: &'call wgpu::Buffer,
    worlds: &'call GpuWorlds,
    item_world: &'call wgpu::Buffer,
    targets: &'call wgpu::Buffer,
    q: &'call wgpu::Buffer,
    aux: &'call wgpu::Buffer,
    out: &'call wgpu::Buffer,
}

/// Whether an adapter is DX12's software renderer (WARP, "Microsoft Basic Render Driver"), which
/// every Windows machine has. On a four-core machine it took 185 s to warm up and had not
/// finished one pick after 82 minutes, where the CPU device picks in 16 ms. Mesa's llvmpipe, the
/// software Vulkan the tests run on, is no such adapter.
fn software_dx12(info: &wgpu::AdapterInfo) -> bool {
    info.backend == wgpu::Backend::Dx12 && info.device_type == wgpu::DeviceType::Cpu
}

/// Which of `adapters`, each with its limits, runs the kernels: the first whose name contains
/// `name` (any if `None`) and whose limits suffice, discrete GPUs before integrated ones before
/// the rest. OpenGL adapters are skipped, and DX12's software renderer unless it is named.
fn choose(adapters: &[(wgpu::AdapterInfo, wgpu::Limits)], name: Option<&str>) -> Result<usize> {
    let wanted = name.map(str::to_lowercase);
    let rank = |t: wgpu::DeviceType| match t {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        wgpu::DeviceType::Other => 4,
    };
    let mut rejected = vec![];
    let mut accepted = vec![];
    for (index, (info, _)) in adapters.iter().enumerate() {
        match &wanted {
            _ if info.backend == wgpu::Backend::Gl => {}
            Some(wanted) if !info.name.to_lowercase().contains(wanted) => {}
            None if software_dx12(info) => rejected.push(format!(
                "{} ({:?}) is a software renderer, which did not finish a pick in over an \
                 hour: it is used only when named",
                info.name, info.backend
            )),
            _ => accepted.push(index),
        }
    }
    accepted.sort_by_key(|&index| rank(adapters[index].0.device_type));
    for index in accepted {
        let (info, limits) = &adapters[index];
        let unmet = unmet_limits(limits);
        if unmet.is_empty() {
            return Ok(index);
        }
        rejected.push(format!(
            "{} ({:?}) lacks {}",
            info.name,
            info.backend,
            unmet.join(", ")
        ));
    }
    Err(match (&rejected[..], name) {
        ([], Some(name)) => Error::Gpu(format!(
            "no Vulkan, Metal or DX12 adapter named like '{name}'"
        )),
        ([], None) => Error::Gpu("no Vulkan, Metal or DX12 adapter found".into()),
        _ => Error::Gpu(format!(
            "no GPU adapter can run the kernels:\n  {}",
            rejected.join("\n  ")
        )),
    })
}

impl GpuBackend {
    /// Uses the adapter `choose` picks among those this machine has, by `name` if given.
    pub(crate) fn new(robot: &Robot, name: Option<&str>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let mut all = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let described: Vec<_> = all.iter().map(|a| (a.get_info(), a.limits())).collect();
        let adapter = all.swap_remove(choose(&described, name)?);
        let info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("batchplan"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter_limits,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| Error::Gpu(format!("requesting GPU device: {e}")))?;

        let built = Arc::new(Mutex::new(HashMap::new()));
        let kernels = kernels_for(&device, &info, &built, robot)?;
        let buffers = RobotBuffers::new(&device, robot);
        let (grids, grid_buffers) = Default::default();
        let lanes = Arc::new(Lanes::new(robot));
        let pool = ThreadPoolBuilder::new()
            .thread_name(|i| format!("batchplan-gpu-{i}"))
            .build()
            .map_err(|e| Error::Threads(format!("starting the GPU device's threads: {e}")))?;
        let pool = Arc::new(pool);
        Ok(Self {
            robot: robot.clone(),
            info,
            device,
            queue,
            kernels,
            built,
            buffers,
            grids,
            grid_buffers,
            lanes,
            pool,
            spare: Arc::default(),
        })
    }

    fn params(&self, n_items: usize, w: &CollisionWeights) -> GpuParams {
        GpuParams {
            n_dof: self.robot.dof() as u32,
            n_items: n_items as u32,
            w_world: w.world,
            w_self: w.self_collision,
            margin: w.margin,
            self_margin: w.self_margin,
            ..Default::default()
        }
    }

    fn upload_worlds(&self, worlds: &[World]) -> Result<GpuWorlds> {
        let cpu = cpu::prepare(&self.robot, worlds)?;
        let mut obstacles = vec![];
        let mut ranges: Vec<[u32; 2]> = vec![];
        // Touch sets after a word that holds none, then each distinct distance grid once.
        let (mut touches, mut words): (Vec<u32>, Vec<u32>) = (vec![0], vec![]);
        let mut grid_offsets: HashMap<*const SdfGrid, u32> = HashMap::new();
        for (s, prepared) in worlds.iter().zip(&cpu) {
            ranges.push([obstacles.len() as u32, s.obstacles.len() as u32]);
            for (k, o) in s.obstacles.iter().enumerate() {
                let rotated = |kind: u32, center: Vec3, half: [f32; 4], rotation: Quat| {
                    let r = Mat3::from_quat(rotation);
                    GpuObstacle {
                        center: v4(center, kind as f32),
                        half,
                        r0: v4(r.x_axis, 0.0),
                        r1: v4(r.y_axis, 0.0),
                        r2: v4(r.z_axis, 0.0),
                        ..Default::default()
                    }
                };
                obstacles.push(match *o {
                    Obstacle::Cuboid {
                        center,
                        half_extents,
                        rotation,
                    } => rotated(CUBOID, center, v4(half_extents, 0.0), rotation),
                    Obstacle::Sphere { center, radius } => {
                        rotated(SPHERE, center, [radius, 0.0, 0.0, 0.0], Quat::IDENTITY)
                    }
                    Obstacle::Cylinder {
                        center,
                        rotation,
                        radius,
                        half_height,
                    } => rotated(CYLINDER, center, [radius, half_height, 0.0, 0.0], rotation),
                    Obstacle::Capsule {
                        center,
                        rotation,
                        radius,
                        half_length,
                    } => rotated(CAPSULE, center, [radius, half_length, 0.0, 0.0], rotation),
                    Obstacle::Sdf {
                        ref grid,
                        center,
                        rotation,
                    } => {
                        let offset = *grid_offsets.entry(Arc::as_ptr(grid)).or_insert_with(|| {
                            let offset = words.len() as u32;
                            words.extend(grid.values.chunks(2).map(|pair| {
                                u32::from(pair[0])
                                    | u32::from(pair.get(1).copied().unwrap_or(0)) << 16
                            }));
                            offset
                        });
                        let [nx, ny, nz] = grid.dims;
                        GpuObstacle {
                            nx,
                            ny,
                            nz,
                            grid: offset,
                            ..rotated(SDF, center, v4(grid.origin, grid.voxel), rotation)
                        }
                    }
                });
                let set = prepared.touch.row(k);
                if set.iter().any(|&w| w != 0) {
                    let last = obstacles.len() - 1;
                    obstacles[last].r0[3] = touches.len() as f32;
                    touches.push(2 * set.len() as u32);
                    touches.extend(set.iter().flat_map(|&w| [w as u32, (w >> 32) as u32]));
                }
            }
        }
        if touches.len() > 1 << 24 {
            return Err(Error::Gpu(format!(
                "the worlds' touch sets take {} words, more than their offsets can address",
                touches.len()
            )));
        }
        for o in obstacles.iter_mut().filter(|o| o.center[3] == SDF as f32) {
            o.grid += touches.len() as u32;
        }
        touches.append(&mut words);
        let words = touches;
        let limits = self.device.limits();
        let bytes = (words.len() * 4) as u64;
        if bytes
            > limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size)
        {
            return Err(Error::Gpu(format!(
                "the worlds' distance grids take {} MiB, more than {} can bind",
                bytes >> 20,
                self.info.name
            )));
        }
        Ok(GpuWorlds {
            obstacles: storage(&self.device, "obstacles", &obstacles),
            ranges: storage(&self.device, "world ranges", &ranges),
            data: storage(&self.device, "world data", &words),
            cpu,
        })
    }

    fn uniform(&self, params: &GpuParams) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// A buffer of `usage` with room for `bytes`: one kept from an earlier call where a small one
    /// fits (its contents are whatever that call left), or a new one.
    fn lend(&self, usage: wgpu::BufferUsages, bytes: u64) -> wgpu::Buffer {
        let bytes = bytes.max(16);
        if bytes <= SPARE_BYTES {
            let mut spare = self.spare.lock().expect("no panic holds the spare buffers");
            let fits = |b: &wgpu::Buffer| b.usage() == usage && b.size() >= bytes;
            if let Some(i) = spare.iter().position(fits) {
                return spare.swap_remove(i);
            }
        }
        let size = if bytes <= SPARE_BYTES {
            bytes.next_power_of_two().max(256)
        } else {
            bytes
        };
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Keeps the small ones of `buffers`, whose work is over, for later calls.
    fn give_back(&self, buffers: impl IntoIterator<Item = wgpu::Buffer>) {
        let mut spare = self.spare.lock().expect("no panic holds the spare buffers");
        for b in buffers {
            if b.size() <= SPARE_BYTES && spare.len() < SPARE_BUFFERS {
                spare.push(b);
            }
        }
    }

    fn bind_main(&self, b: &CallBuffers) -> wgpu::BindGroup {
        let buffers = [
            b.params,
            &self.buffers.links,
            &self.buffers.spheres,
            &self.buffers.pairs,
            &self.buffers.limits,
            &b.worlds.obstacles,
            &b.worlds.ranges,
            b.item_world,
            b.targets,
            &b.worlds.data,
            b.q,
            b.aux,
            b.out,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.kernels.layout0,
            entries: &entries,
        })
    }

    fn submit_pass(&self, f: impl FnOnce(&mut wgpu::ComputePass)) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            f(&mut pass);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Blocks until the GPU has finished all submitted work.
    fn wait(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| Error::Gpu(e.to_string()))
    }

    /// Submits `enc` with copies of `parts` (buffers and their lengths in `T`s) into one staging
    /// buffer, and reads them all back after one wait.
    fn read_after<T: Pod, const N: usize>(
        &self,
        enc: wgpu::CommandEncoder,
        parts: [(&wgpu::Buffer, usize); N],
    ) -> Result<[Vec<T>; N]> {
        self.read_into(enc, parts, |parts: [&[T]; N]| parts.map(<[T]>::to_vec))
    }

    /// [`Self::read_after`], handing the parts to `take` where they are mapped rather than
    /// copying each out first.
    fn read_into<T: Pod, R, const N: usize>(
        &self,
        mut enc: wgpu::CommandEncoder,
        parts: [(&wgpu::Buffer, usize); N],
        take: impl FnOnce([&[T]; N]) -> R,
    ) -> Result<R> {
        let size = std::mem::size_of::<T>() as u64;
        let total: usize = parts.iter().map(|&(_, len)| len).sum();
        let bytes = (total as u64 * size).max(size);
        let staging = self.lend(
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            bytes,
        );
        let mut offset = 0;
        for &(buf, len) in &parts {
            enc.copy_buffer_to_buffer(buf, 0, &staging, offset, len as u64 * size);
            offset += len as u64 * size;
        }
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        self.wait()?;
        rx.recv()
            .map_err(|e| Error::Gpu(e.to_string()))?
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let mapped = staging
            .slice(..bytes)
            .get_mapped_range()
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let values: &[T] = bytemuck::cast_slice(&mapped);
        let mut start = 0;
        let out = take(parts.map(|(_, len)| {
            let part = &values[start..start + len];
            start += len;
            part
        }));
        drop(mapped);
        staging.unmap();
        self.give_back([staging]);
        Ok(out)
    }

    fn read<T: Pod>(&self, buf: &wgpu::Buffer, len: usize) -> Result<Vec<T>> {
        let enc = self.device.create_command_encoder(&Default::default());
        let [out] = self.read_after(enc, [(buf, len)])?;
        Ok(out)
    }

    /// Runs the `kernel` of the collision kernels once per configuration, which writes `stride`
    /// floats each.
    fn per_configuration(
        &self,
        kernel: fn(&CollisionKernels) -> &wgpu::ComputePipeline,
        stride: usize,
        worlds: &GpuWorlds,
        item_world: &[u32],
        q: &[f32],
        w: &CollisionWeights,
    ) -> Result<Vec<f32>> {
        let items = item_world.len();
        let params = self.uniform(&self.params(items, w));
        let world_buf = storage(&self.device, "item world", item_world);
        let no_targets = storage::<[f32; 4]>(&self.device, "unused targets", &[]);
        let no_aux = storage::<f32>(&self.device, "unused aux", &[]);
        let q_buf = storage(&self.device, "q", q);
        let out = storage_zeroed(&self.device, "out", items * stride);
        let bg = self.bind_main(&CallBuffers {
            params: &params,
            worlds,
            item_world: &world_buf,
            targets: &no_targets,
            q: &q_buf,
            aux: &no_aux,
            out: &out,
        });
        self.submit_pass(|pass| {
            let lanes = lanes_for(items);
            pass.set_pipeline(kernel(self.kernels.collision(lanes)));
            pass.set_bind_group(0, &bg, &[]);
            dispatch(pass, items * lanes as usize);
        });
        self.read(&out, items * stride)
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> String {
        format!(
            "gpu ({}, {:?}, {})",
            self.info.name, self.info.backend, self.info.driver_info
        )
    }

    fn pool(&self) -> &ThreadPool {
        &self.pool
    }

    fn robot(&self) -> &Robot {
        &self.robot
    }

    fn upload(&self, worlds: &[World]) -> Result<Box<dyn Any + Send + Sync>> {
        Ok(Box::new(self.upload_worlds(worlds)?))
    }

    fn with_robot(&self, robot: &Robot) -> Result<Box<dyn Backend>> {
        let kernels = kernels_for(&self.device, &self.info, &self.built, robot)?;
        let buffers = RobotBuffers::new(&self.device, robot);
        let lanes = Arc::new(Lanes::new(robot));
        Ok(Box::new(Self {
            robot: robot.clone(),
            kernels,
            buffers,
            lanes,
            ..self.clone()
        }))
    }

    fn evaluate(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        q: &[f32],
        w: &CollisionWeights,
    ) -> Result<Evaluation> {
        let worlds: &GpuWorlds = worlds.prepared();
        let n = self.robot.dof();
        let stride = 3 + n;
        let mut out = Evaluation::default();
        for (chunk_world, chunk_q) in item_world.chunks(EVAL_CHUNK).zip(q.chunks(EVAL_CHUNK * n)) {
            let raw =
                self.per_configuration(|k| &k.evaluate, stride, worlds, chunk_world, chunk_q, w)?;
            for row in raw.chunks(stride) {
                out.world_clearance.push(row[0]);
                out.self_clearance.push(row[1]);
                out.cost.push(row[2]);
                out.gradient.extend_from_slice(&row[3..]);
            }
        }
        Ok(out)
    }

    fn clearance(&self, worlds: &Worlds, item_world: &[u32], q: &[f32]) -> Result<Vec<[f32; 2]>> {
        let worlds: &GpuWorlds = worlds.prepared();
        let n = self.robot.dof();
        let mut out = Vec::with_capacity(item_world.len());
        for (chunk_world, chunk_q) in item_world.chunks(EVAL_CHUNK).zip(q.chunks(EVAL_CHUNK * n)) {
            let raw = self.per_configuration(
                |k| &k.clearance,
                2,
                worlds,
                chunk_world,
                chunk_q,
                &CollisionWeights::NONE,
            )?;
            out.extend(raw.chunks(2).map(|c| [c[0], c[1]]));
        }
        Ok(out)
    }

    fn ik(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        targets: &[Option<Pose>],
        held: &[bool],
        q: &mut [f32],
        o: &IkOptions,
    ) -> Result<Vec<[f32; 4]>> {
        let worlds: &GpuWorlds = worlds.prepared();
        let (n, hands) = (self.robot.dof(), self.robot.hands());
        // Items to a problem, whose targets they share.
        let seeds = (item_world.len() * hands)
            .checked_div(targets.len())
            .unwrap_or(1);
        let mut errors = Vec::with_capacity(item_world.len());
        // Seeds are chosen among a problem's (`IkOptions::screens`), so a chunk then holds whole
        // problems.
        let chunk = if o.screens() {
            IK_CHUNK / seeds * seeds
        } else {
            IK_CHUNK
        };
        let chunks = item_world
            .chunks(chunk)
            .zip(q.chunks_mut(chunk * n))
            .enumerate();
        for (c, (chunk_world, chunk_q)) in chunks {
            // The targets of the problems the chunk's items belong to, the first of which it may
            // enter partway through its seeds.
            let (start, end) = (c * chunk, c * chunk + chunk_world.len());
            let problems = start / seeds..(end - 1) / seeds + 1;
            let chunk_targets = &targets[problems.start * hands..problems.end * hands];
            let chunk_held = held
                .get(start * n..start * n + chunk_q.len())
                .unwrap_or_default();
            let call = (chunk_world, chunk_targets, chunk_held);
            let group = (seeds, start % seeds);
            self.ik_chunk(worlds, call, group, chunk_q, o, &mut errors)?;
        }
        Ok(errors)
    }

    fn trajopt(
        &self,
        worlds: &Worlds,
        item_world: &[u32],
        item_region: &[Option<Arc<Region>>],
        paths: &mut JointPaths,
        o: &PlanOptions,
        deadline: Option<Instant>,
    ) -> Result<()> {
        let worlds: &GpuWorlds = worlds.prepared();
        let (points, n) = (paths.points, paths.dof);
        if item_world.is_empty() || o.iterations == 0 {
            return Ok(());
        }
        // Paths are optimized in chunks whose L-BFGS state fits one binding and whose work per
        // submission stays well under driver watchdogs.
        let limits = self.device.limits();
        let bytes = 4 * lbfgs_stride(points, n, o) as u64;
        let fits = (limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / bytes)
            .max(1) as usize;
        let chunk = fits.min(TRAJ_CHUNK);
        for ((chunk_world, chunk_region), chunk_paths) in item_world
            .chunks(chunk)
            .zip(item_region.chunks(chunk))
            .zip(paths.positions.chunks_mut(chunk * points * n))
        {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            self.trajopt_chunk(worlds, chunk_world, chunk_region, chunk_paths, o, deadline)?;
        }
        Ok(())
    }

    fn grid_values(&self, grid: &Layout, occupancy: Occupancy) -> Result<Vec<u16>> {
        let kernels = self.grid_kernels()?;
        let mut buffers = self
            .grid_buffers
            .lock()
            .expect("no thread panics while holding the grid buffers");
        let points = grid.points();
        let (pixels, robot) = match occupancy {
            Occupancy::Seen { image, robot, .. } => {
                (image.depth.len(), robot.map_or(0, mask_words))
            }
            Occupancy::Given(_) => (0, 0),
        };
        let bindings = buffers.bindings(
            &self.device,
            points,
            pixels,
            robot,
            grid.diagonal() as usize + 1,
        );
        let (params, mut steps) = match occupancy {
            Occupancy::Given(occupied) => {
                let bits: Vec<u32> = occupied
                    .par_chunks(32)
                    .map(|word| {
                        word.iter()
                            .enumerate()
                            .fold(0, |w, (b, &o)| w | u32::from(o) << b)
                    })
                    .collect();
                self.queue
                    .write_buffer(bindings[2], 0, bytemuck::cast_slice(&bits));
                (grid_params(grid, None, None, Occlusion::Free), vec![])
            }
            Occupancy::Seen {
                image,
                robot,
                behind,
            } => {
                self.upload_image(&bindings, image, robot);
                let params = grid_params(grid, Some(image), robot, behind);
                (
                    params,
                    vec![
                        (&kernels.depth_occupancy, params, points.div_ceil(32)),
                        (&kernels.depth_hits, params, pixels),
                    ],
                )
            }
        };
        steps.extend(transform_steps(kernels, grid, &params));
        self.run_grid(kernels, &bindings, &steps);
        // The table is built while the transform runs.
        let table = sdf::finishing_table(grid.voxel, grid.diagonal());
        self.queue
            .write_buffer(bindings[9], 0, bytemuck::cast_slice(&table));
        self.run_grid(
            kernels,
            &bindings,
            &[(&kernels.finish, params, points.div_ceil(2))],
        );
        let words: Vec<u32> = self.read(bindings[4], points.div_ceil(2))?;
        if self.read::<u32>(bindings[3], 1)?[0] == FAR {
            return Err(sdf::one_kind(self.read::<u32>(bindings[2], 1)?[0] & 1 == 1));
        }
        Ok(bytemuck::cast_slice::<u32, u16>(&words)[..points].to_vec())
    }

    fn checked<'w>(&self, worlds: &'w Worlds) -> &'w CpuWorlds {
        &worlds.prepared::<GpuWorlds>().cpu
    }

    fn build_all(&self) -> Result<()> {
        for lanes in [1, SHARED_LANES] {
            self.kernels
                .traj(&self.device, &self.info, lanes, /* constrained */ true)?;
        }
        self.grid_kernels().map(|_| ())
    }

    fn search(&self, worlds: &Worlds, searches: &mut [&mut dyn Search], deadline: Option<Instant>) {
        let cpu = self.checked(worlds);
        crate::cpu::run_searches(
            &self.pool,
            searches,
            &self.robot,
            &self.lanes,
            cpu,
            deadline,
        );
    }

    fn integrate(
        &self,
        grid: &Layout,
        log_odds: &mut [i8],
        image: &DepthImage,
        robot: Option<&RobotMask>,
    ) -> Result<()> {
        let kernels = self.grid_kernels()?;
        let mut buffers = self
            .grid_buffers
            .lock()
            .expect("no thread panics while holding the grid buffers");
        let bindings = buffers.bindings(
            &self.device,
            grid.points(),
            image.depth.len(),
            robot.map_or(0, mask_words),
            0,
        );
        self.upload_image(&bindings, image, robot);
        let mut words = bytemuck::cast_slice::<i8, u8>(log_odds).to_vec();
        words.resize(log_odds.len().next_multiple_of(4), 0);
        self.queue.write_buffer(bindings[8], 0, &words);
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.clear_buffer(bindings[2], 0, None);
        self.queue.submit([enc.finish()]);
        let params = GpuGridParams {
            map: 1,
            ..grid_params(grid, Some(image), robot, Occlusion::Free)
        };
        let steps = [
            (&kernels.depth_hits, params, image.depth.len()),
            (&kernels.map_update, params, grid.points().div_ceil(4)),
        ];
        self.run_grid(kernels, &bindings, &steps);
        let words: Vec<u32> = self.read(bindings[8], grid.points().div_ceil(4))?;
        log_odds.copy_from_slice(&bytemuck::cast_slice::<u32, i8>(&words)[..log_odds.len()]);
        Ok(())
    }
}

impl GpuBackend {
    fn grid_kernels(&self) -> Result<&GridKernels> {
        if let Some(kernels) = self.grids.get() {
            return Ok(kernels);
        }
        let kernels = GridKernels::new(&self.device, &self.info)?;
        Ok(self.grids.get_or_init(|| kernels))
    }

    /// Uploads an image's readings (0 for pixels without one) and its robot mask: depths (huge
    /// where the robot is not seen), then the grown spheres.
    fn upload_image(
        &self,
        bindings: &[&wgpu::Buffer; 10],
        image: &DepthImage,
        robot: Option<&RobotMask>,
    ) {
        let readings: Vec<f32> = self.pool.install(|| {
            image
                .depth
                .par_iter()
                .map(|&z| if z.is_finite() && z > 0.0 { z } else { 0.0 })
                .collect()
        });
        self.queue
            .write_buffer(bindings[0], 0, bytemuck::cast_slice(&readings));
        if let Some(robot) = robot {
            let mut words: Vec<f32> = self
                .pool
                .install(|| robot.depth.par_iter().map(|&r| r.min(f32::MAX)).collect());
            words.extend(robot.spheres.iter().flat_map(|s| s.to_array()));
            self.queue
                .write_buffer(bindings[1], 0, bytemuck::cast_slice(&words));
        }
    }

    /// Runs each `(kernel, params, invocations)` step in order, then waits.
    fn run_grid(
        &self,
        kernels: &GridKernels,
        bindings: &[&wgpu::Buffer; 10],
        steps: &[(&wgpu::ComputePipeline, GpuGridParams, usize)],
    ) {
        let groups: Vec<_> = steps
            .iter()
            .map(|(_, params, _)| {
                let params = self.uniform_of(params);
                let entries: Vec<_> = std::iter::once(&params)
                    .chain(bindings.iter().copied())
                    .enumerate()
                    .map(|(i, buf)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: buf.as_entire_binding(),
                    })
                    .collect();
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &kernels.layout,
                    entries: &entries,
                })
            })
            .collect();
        self.submit_pass(|pass| {
            for ((kernel, _, invocations), group) in steps.iter().zip(&groups) {
                pass.set_pipeline(kernel);
                pass.set_bind_group(0, group, &[]);
                dispatch(pass, *invocations);
            }
        });
    }

    fn uniform_of<T: Pod>(&self, value: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("grid params"),
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }
}

/// The words of a robot mask in binding 2 of grids.wgsl.
fn mask_words(robot: &RobotMask) -> usize {
    robot.depth.len() + 4 * robot.spheres.len()
}

/// The grid kernels' parameters for `grid`, and for an image seen with `robot`'s mask.
fn grid_params(
    grid: &Layout,
    image: Option<&DepthImage>,
    robot: Option<&RobotMask>,
    behind: Occlusion,
) -> GpuGridParams {
    let [nx, ny, nz] = grid.dims.to_array();
    let mut params = GpuGridParams {
        origin: v4(grid.origin, grid.voxel),
        nx,
        ny,
        nz,
        points: grid.points() as u32,
        has_robot: u32::from(robot.is_some()),
        spheres: robot.map_or(0, |r| r.spheres.len() as u32),
        behind: match behind {
            Occlusion::Free => 0,
            Occlusion::Occupied => 1,
            Occlusion::OccupiedExceptRobotShadow => 2,
        },
        ..Default::default()
    };
    if let Some(image) = image {
        let from_camera = Mat3::from_quat(image.camera.rotation);
        let to_camera = from_camera.transpose();
        let k = image.intrinsics;
        params.camera = v4(image.camera.position, 0.0);
        params.intrinsics = [k.fx, k.fy, k.cx, k.cy];
        [params.to_camera0, params.to_camera1, params.to_camera2] =
            [0, 1, 2].map(|i| v4(to_camera.row(i), 0.0));
        [
            params.from_camera0,
            params.from_camera1,
            params.from_camera2,
        ] = [0, 1, 2].map(|i| v4(from_camera.col(i), 0.0));
        params.width = image.width as u32;
        params.height = (image.depth.len() / image.width) as u32;
    }
    params
}

/// The three passes of the distance transform, one per axis.
fn transform_steps<'a>(
    kernels: &'a GridKernels,
    grid: &Layout,
    params: &GpuGridParams,
) -> [(&'a wgpu::ComputePipeline, GpuGridParams, usize); 3] {
    let dims = grid.dims.to_array();
    [0, 1, 2].map(|axis| {
        let lines = grid.points() / dims[axis] as usize;
        (
            &kernels.edt_pass,
            GpuGridParams {
                axis: axis as u32,
                lines: lines as u32,
                ..*params
            },
            lines,
        )
    })
}

/// Floats of L-BFGS state per path; must match `lbfgs_stride` in kernels.wgsl.
fn lbfgs_stride(points: usize, n: usize, o: &PlanOptions) -> usize {
    let samples = (points - 3) * o.samples_per_span;
    samples * n + LINE_SEARCH.len() * samples + (4 + 2 * o.history) * points * n + 6
}

impl GpuBackend {
    /// IK on one chunk: its items' worlds, targets and held joints (`call`), in place on `q`.
    fn ik_chunk(
        &self,
        worlds: &GpuWorlds,
        (item_world, targets, held): (&[u32], &[Option<Pose>], &[bool]),
        (seeds, first): (usize, usize),
        q: &mut [f32],
        o: &IkOptions,
        results: &mut Vec<[f32; 4]>,
    ) -> Result<()> {
        let items = item_world.len();
        if items == 0 {
            return Ok(());
        }
        // Lanes split the collision checks, which IK makes only to push away from obstacles.
        let lanes = if o.collision_step > 0.0 {
            ik_lanes(items, self.robot.spheres.len())
        } else {
            1
        };
        let mut params = self.params(items, &o.collision);
        params.damping = o.damping;
        params.rotation_weight = o.rotation_weight;
        params.max_step = o.max_step;
        params.collision_step = o.collision_step;
        params.held = u32::from(!held.is_empty());
        params.per_target = seeds as u32;
        params.first_seed = first as u32;
        // A small call's buffers are ones kept from earlier calls, filled through the queue;
        // a large one's are made around their data, which saves a copy.
        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let small = std::mem::size_of_val(q) as u64 <= SPARE_BYTES;
        let lent = RefCell::new(Vec::new());
        let lend = |usage: wgpu::BufferUsages, bytes: &[u8], room: usize| {
            let buffer = self.lend(usage, room as u64);
            if !bytes.is_empty() {
                self.queue.write_buffer(&buffer, 0, bytes);
            }
            lent.borrow_mut().push(buffer.clone());
            buffer
        };
        let world_buf = if small {
            lend(storage_usage, bytemuck::cast_slice(item_world), 4 * items)
        } else {
            storage(&self.device, "item world", item_world)
        };
        // Per problem (`P.per_target` items, the first entered at its seed `P.first_seed`) and
        // hand: the position with w = 1 when the hand has a target, then the rotation's columns.
        let target_data: Vec<[f32; 4]> = targets
            .iter()
            .flat_map(|t| match t {
                Some(t) => {
                    let r = Mat3::from_quat(t.rotation);
                    [
                        v4(t.position, 1.0),
                        v4(r.x_axis, 0.0),
                        v4(r.y_axis, 0.0),
                        v4(r.z_axis, 0.0),
                    ]
                }
                None => [[0.0; 4]; 4],
            })
            .collect();
        let target_buf = if small {
            let bytes: &[u8] = bytemuck::cast_slice(&target_data);
            lend(storage_usage, bytes, bytes.len())
        } else {
            storage(&self.device, "targets", &target_data)
        };
        // Per item and joint, 1 where it is held (`P.held`).
        let flags: Vec<f32> = held.iter().map(|&h| f32::from(u8::from(h))).collect();
        let (held_buf, q_buf) = if small {
            let (flags, seeded): (&[u8], &[u8]) =
                (bytemuck::cast_slice(&flags), bytemuck::cast_slice(q));
            (
                lend(storage_usage, flags, flags.len()),
                lend(storage_usage, seeded, seeded.len()),
            )
        } else {
            (
                storage(&self.device, "held", &flags),
                storage(&self.device, "q", q),
            )
        };
        // Past the items' results in both outputs, the seeds chosen to go on (`P.chosen`).
        let slots = if o.screens() {
            items / seeds * o.kept
        } else {
            0
        };
        // Every float of both that is read is written first.
        let floats = items * 2 + slots;
        let (out, clear_out) = if small {
            (
                lend(storage_usage, &[], 4 * floats),
                lend(storage_usage, &[], 4 * floats),
            )
        } else {
            (
                storage_zeroed(&self.device, "ik errors", floats),
                storage_zeroed(&self.device, "clearance", floats),
            )
        };
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let bind =
            |params: &GpuParams, q: &wgpu::Buffer, aux: &wgpu::Buffer, out: &wgpu::Buffer| {
                let bytes = bytemuck::bytes_of(params);
                self.bind_main(&CallBuffers {
                    params: &lend(uniform_usage, bytes, bytes.len()),
                    worlds,
                    item_world: &world_buf,
                    targets: &target_buf,
                    q,
                    aux,
                    out,
                })
            };
        // Passes share a submission up to the work of `IK_ITERS_PER_SUBMIT` steps of a full chunk
        // and `IK_MOST_STEPS` steps, so a small call is one submission; everything is read back
        // through one buffer after one wait.
        let budget = IK_ITERS_PER_SUBMIT as usize * IK_CHUNK;
        let mut enc = self.device.create_command_encoder(&Default::default());
        // The open submission's work (steps times configurations) and steps.
        let mut open = (0usize, 0u32);
        // `steps` IK steps of the call's `configurations` as `params` has them (`chosen`: where
        // their indices are, or 0). No steps still write the errors.
        let mut stage = |enc: &mut wgpu::CommandEncoder,
                         params: &mut GpuParams,
                         lanes: u32,
                         configurations: usize,
                         steps: u32| {
            let most = (budget / configurations) as u32;
            let most = most.clamp(IK_ITERS_PER_SUBMIT, IK_MOST_STEPS);
            let mut left = steps;
            loop {
                params.iterations = most.min(left);
                let work = params.iterations as usize * configurations;
                if open.1 > 0
                    && (open.0 + work > budget || open.1 + params.iterations > IK_MOST_STEPS)
                {
                    let next = self.device.create_command_encoder(&Default::default());
                    self.queue.submit([std::mem::replace(enc, next).finish()]);
                    open = (0, 0);
                }
                let bg = bind(params, &q_buf, &held_buf, &out);
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.kernels.collision(lanes).ik);
                pass.set_bind_group(0, &bg, &[]);
                dispatch(&mut pass, configurations * lanes as usize);
                open = (open.0 + work, open.1 + params.iterations);
                left -= params.iterations;
                if left == 0 {
                    break;
                }
            }
        };
        // The clearances of the call's `configurations` (`chosen`, as above), from the
        // configurations already on the GPU.
        let clearances = |enc: &mut wgpu::CommandEncoder, configurations: usize, chosen: u32| {
            let mut clear = self.params(configurations, &CollisionWeights::NONE);
            clear.chosen = chosen;
            let bg = bind(&clear, &q_buf, &held_buf, &clear_out);
            let lanes = lanes_for(configurations);
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernels.collision(lanes).clearance);
            pass.set_bind_group(0, &bg, &[]);
            dispatch(&mut pass, configurations * lanes as usize);
        };
        if slots == 0 {
            stage(&mut enc, &mut params, lanes, items, o.iterations);
            clearances(&mut enc, items, 0);
        } else {
            // Every seed's screening; then each seed finds its place among its problem's, from
            // the errors and clearances; then those chosen take the other steps. The choice counts
            // collisions where the steps push away from them (`IkOptions::pushes`): elsewhere the
            // kernel leaves the clearances unread, which a small call's buffer still has from
            // the call before, and all are found at the end.
            stage(&mut enc, &mut params, lanes, items, o.screening);
            if o.pushes() {
                clearances(&mut enc, items, 0);
            }
            let chosen = (items * 2) as u32;
            let mut choice = self.params(items, &CollisionWeights::NONE);
            choice.per_target = seeds as u32;
            choice.kept = o.kept as u32;
            choice.chosen = chosen;
            choice.rotation_weight = o.rotation_weight;
            choice.position_tolerance = o.position_tolerance;
            choice.rotation_tolerance = o.rotation_tolerance;
            choice.collision_step = f32::from(u8::from(o.pushes()));
            // The errors and the clearances where the kernel reads and writes, and the
            // configurations where it writes nothing.
            let choice_bg = bind(&choice, &out, &clear_out, &q_buf);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.kernels.ik_select);
                pass.set_bind_group(0, &choice_bg, &[]);
                dispatch(&mut pass, items);
            }
            let lanes = if o.pushes() {
                ik_lanes(slots, self.robot.spheres.len())
            } else {
                1
            };
            params.n_items = slots as u32;
            params.chosen = chosen;
            stage(
                &mut enc,
                &mut params,
                lanes,
                slots,
                o.iterations - o.screening,
            );
            if o.pushes() {
                clearances(&mut enc, slots, chosen);
            } else {
                clearances(&mut enc, items, 0);
            }
        }
        let parts = [
            (&q_buf, q.len()),
            (&out, items * 2),
            (&clear_out, items * 2),
        ];
        self.read_into(enc, parts, |[solved, errors, clear]: [&[f32]; 3]| {
            q.copy_from_slice(solved);
            let items = errors.chunks(2).zip(clear.chunks(2));
            results.extend(items.map(|(e, c)| [e[0], e[1], c[0], c[1]]));
        })?;
        self.give_back(lent.take());
        Ok(())
    }

    /// Optimizes the paths `traj` holds, one per item of `item_world`, each keeping its constraints
    /// (`item_region`) if it has any.
    fn trajopt_chunk(
        &self,
        worlds: &GpuWorlds,
        item_world: &[u32],
        item_region: &[Option<Arc<Region>>],
        traj: &mut [f32],
        o: &PlanOptions,
        deadline: Option<Instant>,
    ) -> Result<()> {
        let items = item_world.len();
        let points = traj.len() / (items * self.robot.dof());
        let samples = items * (points - 3) * o.samples_per_span;
        let mut params = self.params(items, &o.collision);
        params.points = points as u32;
        params.samples = o.samples_per_span as u32;
        params.acceleration_weight = o.acceleration_weight;
        params.velocity_weight = o.velocity_weight;
        params.initial_step = o.initial_step;
        params.history = o.history as u32;
        let slots = item_region.iter().flatten().map(|r| r.limits.len()).max();
        let slots = slots.unwrap_or(0);
        params.constraints = slots as u32;
        params.constraint_weight = CONSTRAINT_WEIGHT;
        // The constraints' gradients read the hands' Jacobians unscaled.
        params.rotation_weight = 1.0;
        let lanes = lanes_for(samples);
        let [traj_costs, traj_samples] =
            self.kernels
                .traj(&self.device, &self.info, lanes, slots > 0)?;
        let params_buf = self.uniform(&params);
        let world_buf = storage(&self.device, "item world", item_world);
        let constraints = storage(
            &self.device,
            "constraints",
            &constraint_slots(item_region, slots),
        );
        let q_buf = storage(&self.device, "trajectories", traj);
        let aux = storage_zeroed(
            &self.device,
            "L-BFGS state",
            items * lbfgs_stride(points, self.robot.dof(), o),
        );
        let out = storage::<f32>(&self.device, "unused out", &[]);
        let bg = self.bind_main(&CallBuffers {
            params: &params_buf,
            worlds,
            item_world: &world_buf,
            targets: &constraints,
            q: &q_buf,
            aux: &aux,
            out: &out,
        });
        // The first round only prices the seeds: their directions are still zero.
        let rounds = o.iterations + 1;
        let mut k = 0;
        while k < rounds {
            let end = (k + TRAJ_ITERS_PER_SUBMIT).min(rounds);
            self.submit_pass(|pass| {
                pass.set_bind_group(0, &bg, &[]);
                for round in k..end {
                    pass.set_pipeline(traj_costs);
                    dispatch(pass, samples * LINE_SEARCH.len() * lanes as usize);
                    pass.set_pipeline(&self.kernels.traj_search);
                    dispatch_groups(pass, items);
                    if round + 1 == rounds {
                        break;
                    }
                    pass.set_pipeline(traj_samples);
                    dispatch(pass, samples * lanes as usize);
                    pass.set_pipeline(&self.kernels.traj_grad);
                    dispatch(pass, items * (points - 6));
                    pass.set_pipeline(&self.kernels.lbfgs_direction);
                    dispatch_groups(pass, items);
                }
            });
            k = end;
            // Waiting for the GPU is only needed to keep a deadline.
            if let Some(d) = deadline {
                self.wait()?;
                if Instant::now() >= d {
                    break;
                }
            }
        }
        traj.copy_from_slice(&self.read(&q_buf, traj.len())?);
        Ok(())
    }
}

/// Each item's constraints as `constraint_cost` in kernels.wgsl reads them, `slots` per item of
/// `CONSTRAINT_VEC4S` (5): the centre and the hand, the half extents and the hand it is relative to
/// (-1 for none), and the reference rotation's columns with the tilt bound, the twist bound and 1.
/// Free bounds are `world::FAR`, which every state keeps; empty slots are zero.
fn constraint_slots(item_region: &[Option<Arc<Region>>], slots: usize) -> Vec<Vec4> {
    let mut out = vec![[0.0; 4]; item_region.len() * slots * 5];
    let finite = |v: f32| v.min(world::FAR);
    for (item, region) in item_region.iter().enumerate() {
        for (k, l) in region.iter().flat_map(|r| &r.limits).enumerate() {
            let at = (item * slots + k) * 5;
            let frame = l.frame.map_or(-1.0, |f| f.hand as f32);
            let h = l.half.to_array().map(finite);
            let r = l.reference;
            out[at..at + 5].copy_from_slice(&[
                v4(l.center, l.hand as f32),
                [h[0], h[1], h[2], frame],
                v4(r.x_axis, finite(l.tilt)),
                v4(r.y_axis, finite(l.twist)),
                v4(r.z_axis, 1.0),
            ]);
        }
    }
    out
}

/// Adapter limits the kernels need beyond what wgpu guarantees; empty when the adapter suffices.
fn unmet_limits(l: &wgpu::Limits) -> Vec<String> {
    let storage = READ_ONLY_STORAGE + READ_WRITE_STORAGE;
    let mut unmet = vec![];
    if l.max_storage_buffers_per_shader_stage < storage {
        unmet.push(format!(
            "{storage} storage buffers per shader stage (has {})",
            l.max_storage_buffers_per_shader_stage
        ));
    }
    if l.max_compute_workgroup_size_x < WORKGROUP
        || l.max_compute_invocations_per_workgroup < WORKGROUP
    {
        unmet.push(format!("{WORKGROUP}-wide compute workgroups"));
    }
    unmet
}

fn dispatch(pass: &mut wgpu::ComputePass, threads: usize) {
    dispatch_groups(pass, threads.div_ceil(WORKGROUP as usize));
}

/// Dispatches `groups` workgroups, folded into two dimensions past the per-dimension limit.
fn dispatch_groups(pass: &mut wgpu::ComputePass, groups: usize) {
    let groups = groups as u32;
    if groups == 0 {
        return;
    }
    let x = groups.min(65535);
    pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
}

/// A storage buffer holding `data`, padded to at least one element: WGSL requires a binding to
/// hold one element of its array type even when the batch has none (e.g. a world with no
/// obstacles).
fn storage<T: Pod>(device: &wgpu::Device, label: &str, data: &[T]) -> wgpu::Buffer {
    // Padded only where there is too little to bind: large batches are not copied for it.
    let least = std::mem::size_of::<T>().max(16);
    let mut bytes = Cow::Borrowed(bytemuck::cast_slice::<T, u8>(data));
    if bytes.len() < least {
        bytes.to_mut().resize(least, 0);
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn storage_zeroed(device: &wgpu::Device, label: &str, floats: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (floats.max(4) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn building_everything_builds_the_kernels_first_use_would() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/ur5e/ur_description/urdf/ur5e.urdf"
        );
        let robot = Robot::load(path, &crate::RobotOptions::default()).expect("the UR5e loads");
        let gpu = match GpuBackend::new(&robot, None) {
            Ok(gpu) => gpu,
            Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => {
                eprintln!("skipping GPU: {e}");
                return;
            }
            Err(e) => panic!("no GPU: {e}"),
        };
        assert!(gpu.kernels.constrained.get().is_none() && gpu.grids.get().is_none());
        gpu.build_all().expect("the kernels build");
        assert!(gpu.kernels.constrained.get().is_some() && gpu.grids.get().is_some());
    }

    /// What a robot holds changes none of its kernels' code, however many spheres cover it: a
    /// device warmed up holding one object builds nothing when it holds another.
    #[test]
    fn a_held_object_of_any_size_shares_the_kernels() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/ur5e/ur_description/urdf/ur5e.urdf"
        );
        let robot = Robot::load(path, &crate::RobotOptions::default()).expect("the UR5e loads");
        let holding = |shape: crate::Obstacle| {
            let held = robot
                .attach(&crate::AttachedObject {
                    name: "box".into(),
                    link: "tool0".into(),
                    shapes: vec![shape],
                    touch_links: vec![],
                    spheres: crate::SphereOptions::default(),
                })
                .expect("the box attaches");
            let spheres = held.sphere_ranges.last().expect("the box's link")[1];
            (spheres, robot_wgsl(&held))
        };
        let ball = crate::Obstacle::sphere(Vec3::ZERO, 0.02);
        let cube = crate::Obstacle::cuboid(Vec3::ZERO, Vec3::splat(0.1));
        let ((few, small), (many, large)) = (holding(ball), holding(cube));
        assert!(
            few < PAIRED_SPHERES && many >= PAIRED_SPHERES,
            "{few} and {many} spheres lie on one side of where a link's loops pair up"
        );
        assert!(small == large, "the larger box changes the kernels' code");
    }

    /// A lane's loop over a link's spheres takes its rounds two at a time, the second where the
    /// link still has a sphere for it; other loops stay.
    #[test]
    fn a_strided_loop_takes_its_rounds_in_pairs() {
        let source = "\
for (var k = 0u; k < range.y; k++) {
    for (var s = first + lane; s < first + count; s += LANES) {
        let sp = spheres[s];
        cost += sp.c.w;
    }
}
";
        let paired = rounds_paired(source);
        assert!(
            paired.contains("for (var k = 0u; k < range.y; k++) {"),
            "{paired}"
        );
        assert!(
            paired.contains("for (var s = first + lane; s < first + count; s += 2u * LANES) {"),
            "{paired}"
        );
        assert!(paired.contains("let sp = spheres[s];"), "{paired}");
        assert!(
            paired.contains("if ((s + LANES) < first + count) {"),
            "{paired}"
        );
        // The second round reads its own sphere; `sp` and `spheres` keep their names.
        assert!(
            paired.contains("let sp = spheres[(s + LANES)];"),
            "{paired}"
        );
        assert_eq!(paired.matches("cost += sp.c.w;").count(), 2, "{paired}");
        let braces = |brace: char| paired.matches(brace).count();
        assert_eq!(braces('{'), braces('}'));
    }

    /// Loops of a known count are written out, their variable numbered where it stands alone;
    /// loops that leave early, and loops of other counts, stay.
    #[test]
    fn loops_of_a_known_count_are_written_out() {
        let source = "\
fn f() {
    for (var j = 0u; j < n; j++) {
        jac[j] = cross(sj.c, obj) * 2.0;
    }
    for (var h = 0u; h < HANDS; h++) {
        for (var j = 0u; j < P.n_dof; j++) {
            x[h * 3u + j] = 0.0;
        }
    }
    for (var c = 0u; c < LINE_STEPS; c++) {
        if (skip) {
            continue;
        }
        for (var j = 0u; j < MAX_DOF; j++) {
            y[j] = f32(c);
        }
    }
    for (var c = 0u; c < 2u; c++) {
        part[c] += sp.c.w * f32(c);
    }
    for (var k = 0u; k < range.y; k++) {
        z[k] = 1.0;
    }
}
";
        let written = loops_written_out(source, 2, 2);
        // The joints' loop: `j` alone is numbered, not in `jac`, `sj`, `obj` or the member `.c`.
        assert!(
            written.contains("jac[0u] = cross(sj.c, obj) * 2.0;"),
            "{written}"
        );
        assert!(
            written.contains("jac[1u] = cross(sj.c, obj) * 2.0;"),
            "{written}"
        );
        // A loop in a loop, both written out.
        for (h, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let line = format!("x[{h}u * 3u + {j}u] = 0.0;");
            assert!(written.contains(&line), "{line} is missing from {written}");
        }
        // A loop that continues stays, and the loop inside it is written out with its variable.
        assert!(
            written.contains("for (var c = 0u; c < LINE_STEPS; c++) {"),
            "{written}"
        );
        assert!(written.contains("y[0u] = f32(c);") && written.contains("y[1u] = f32(c);"));
        // A variable named as a member is: the member keeps its name.
        assert!(
            written.contains("part[1u] += sp.c.w * f32(1u);"),
            "{written}"
        );
        // A count only known at run time stays a loop.
        assert!(
            written.contains("for (var k = 0u; k < range.y; k++) {"),
            "{written}"
        );
        let loops = |text: &str| text.matches("for (").count();
        assert_eq!((loops(source), loops(&written)), (7, 2), "{written}");
        // The kernels have such loops, and none over the joints is left in what is built.
        let kernels = include_str!("kernels.wgsl");
        let over_joints = "for (var j = 0u; j < n; j++) {";
        assert!(kernels.matches(over_joints).count() > 10);
        let built = loops_written_out(kernels, 7, 1);
        assert!(!built.contains(over_joints) && !built.contains("j < P.n_dof; j++"));
        // `n` is the joints wherever the kernels name it: loops to `n` are written out as such.
        let named = kernels.matches("let n = ").count();
        assert_eq!(named, kernels.matches("let n = P.n_dof;").count());
        assert!(named > 5);
        // Braces still pair up.
        let braces = |text: &str, brace: char| text.matches(brace).count();
        assert_eq!(braces(&built, '{'), braces(&built, '}'));
    }

    /// A robot relocked (`Robot::relock`) differs only in the link origins the kernels read from
    /// buffers, so a device for it reuses the kernels built.
    #[test]
    fn a_relocked_robot_shares_the_kernels_built() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/ur5e/ur_description/urdf/ur5e.urdf"
        );
        let free = Robot::load(path, &crate::RobotOptions::default()).expect("the UR5e loads");
        let o = crate::RobotOptions {
            lock_joints: HashMap::from([("shoulder_pan_joint".into(), 0.0)]),
            collision_model: Some(free.collision_model().clone()),
            ..Default::default()
        };
        let robot = Robot::load(path, &o).expect("the UR5e loads");
        let gpu = match GpuBackend::new(&robot, None) {
            Ok(gpu) => gpu,
            Err(e) if std::env::var("BATCHPLAN_REQUIRE_GPU").is_err() => {
                eprintln!("skipping GPU: {e}");
                return;
            }
            Err(e) => panic!("no GPU: {e}"),
        };
        let relocked = robot
            .relock(&HashMap::from([("shoulder_pan_joint".into(), 1.2)]))
            .expect("the joint was locked");
        let shared =
            kernels_for(&gpu.device, &gpu.info, &gpu.built, &relocked).expect("the kernels build");
        assert!(Arc::ptr_eq(&shared, &gpu.kernels));
    }

    /// An adapter as wgpu describes it, with limits that suffice or WebGPU's defaults.
    fn adapter(
        name: &str,
        backend: wgpu::Backend,
        device_type: wgpu::DeviceType,
        suffices: bool,
    ) -> (wgpu::AdapterInfo, wgpu::Limits) {
        let info = wgpu::AdapterInfo {
            name: name.into(),
            ..wgpu::AdapterInfo::new(device_type, backend)
        };
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: if suffices { 16 } else { 8 },
            ..wgpu::Limits::defaults()
        };
        (info, limits)
    }

    /// DX12's software renderer, which every Windows machine has, is chosen only by name: a
    /// machine with nothing else gets an error that says so, and the CPU device. A real adapter
    /// goes before it, Mesa's software Vulkan is still chosen, and discrete GPUs come first.
    #[test]
    fn the_software_renderer_of_dx12_is_chosen_only_by_name() {
        use wgpu::{Backend, DeviceType};

        let warp = adapter(
            "Microsoft Basic Render Driver",
            Backend::Dx12,
            DeviceType::Cpu,
            true,
        );
        let radeon = adapter("AMD Radeon", Backend::Dx12, DeviceType::IntegratedGpu, true);
        let geforce = adapter(
            "NVIDIA GeForce",
            Backend::Vulkan,
            DeviceType::DiscreteGpu,
            true,
        );
        let weak = adapter(
            "Old Intel",
            Backend::Vulkan,
            DeviceType::IntegratedGpu,
            false,
        );
        let llvmpipe = adapter("llvmpipe (LLVM 20)", Backend::Vulkan, DeviceType::Cpu, true);
        let message =
            |adapters: &[(wgpu::AdapterInfo, wgpu::Limits)], name: Option<&str>| match choose(
                adapters, name,
            ) {
                Err(Error::Gpu(message)) => message,
                other => panic!("an adapter was chosen, or another error: {other:?}"),
            };
        let alone = [warp.clone()];
        let said = message(&alone, None);
        assert!(
            said.contains("Basic Render Driver") && said.contains("only when named"),
            "{said}"
        );
        assert_eq!(choose(&alone, Some("basic render")).unwrap(), 0);
        let said = message(&[warp.clone(), weak.clone()], None);
        assert!(
            said.contains("only when named") && said.contains("Old Intel"),
            "{said}"
        );
        assert!(said.contains("storage buffers"), "{said}");
        assert_eq!(message(&[], None), "no Vulkan, Metal or DX12 adapter found");
        let said = message(&alone, Some("radeon"));
        assert_eq!(said, "no Vulkan, Metal or DX12 adapter named like 'radeon'");
        assert_eq!(choose(&[warp.clone(), radeon.clone()], None).unwrap(), 1);
        assert_eq!(choose(&[warp, radeon, geforce], None).unwrap(), 2);
        assert_eq!(choose(&[weak, llvmpipe], None).unwrap(), 1);
    }

    #[test]
    fn baseline_webgpu_limits_are_rejected_with_the_reason() {
        let unmet = unmet_limits(&wgpu::Limits::defaults());
        assert_eq!(unmet, ["12 storage buffers per shader stage (has 8)"]);
    }
}
