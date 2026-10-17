//! Closed kinematic loops become polynomial mimics. A tree of joints cannot hold a loop closed, so
//! each loop's passive joints (those no motor drives) follow its driver: at driver values across
//! its range they take their static equilibrium, the least spring energy that closes the loop within
//! the joints' limits (what MuJoCo settles to without contact), and a quartic fitted to those values
//! becomes each passive joint's mimic curve.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use glam::{DVec3, Vec3};

use crate::description::{Curve, LoopClosure, Mimic, RobotDescription};
use crate::robot::Robot;

/// Largest gap the fitted curves may leave between a loop's two points, in meters.
const CLOSURE_TOLERANCE: f64 = 5e-4;
/// Driver values solved and fitted per loop.
const SAMPLES: usize = 64;
/// Weight of the closure residual (per m²) against spring energy (N·m): a millimetre of gap
/// outweighs any joint spring.
const CLOSURE_WEIGHT: f64 = 1e10;
/// Weight pulling each solve toward the previous one, which settles what neither closure nor
/// springs decide.
const STAY: f64 = 1e-6;

/// One loop in `tree`, a robot built from the description with every joint planned.
struct Loop {
    links: [usize; 2],
    anchors: [Vec3; 2],
    driver: usize,
    /// The passive joints' dofs in `tree`, with their spring stiffness and rest value.
    passive: Vec<(usize, f64, f64)>,
}

/// `desc` with every passive joint of its loops a mimic of the loop's driver. `tree` is `desc` with
/// the loops left open and every joint planned.
pub(crate) fn close_loops(desc: &RobotDescription, tree: &Robot) -> Result<RobotDescription> {
    let mut out = desc.clone();
    let joint_of: HashMap<&str, usize> = desc.joints.iter().enumerate().map(|(i, j)| (j.name.as_str(), i)).collect();
    let mut claimed: HashMap<usize, &str> = HashMap::new();
    for closure in &desc.loops {
        let l = find_loop(desc, tree, closure, &joint_of)?;
        for &(dof, ..) in &l.passive {
            if let Some(other) = claimed.insert(dof, &closure.name) {
                bail!("joint '{}' is passive in two loops, '{other}' and '{}'", tree.joint_names()[dof], closure.name);
            }
        }
        let curves = solve(tree, &l).with_context(|| format!("closing loop '{}'", closure.name))?;
        let driver = tree.joint_names()[l.driver].clone();
        let (lo, hi) = (tree.lower()[l.driver], tree.upper()[l.driver]);
        for (&(dof, ..), curve) in l.passive.iter().zip(curves) {
            let j = &mut out.joints[joint_of[tree.joint_names()[dof].as_str()]];
            // The solve kept the joint within its limits; where the fit overshoots them (a joint held
            // at a stop, then released, is not a quartic), its limits widen to the curve instead of
            // narrowing the driver's range.
            for k in 0..=4 * SAMPLES {
                let v = curve.value(lo + (hi - lo) * k as f32 / (4 * SAMPLES) as f32);
                (j.lower, j.upper) = (j.lower.min(v), j.upper.max(v));
            }
            j.mimic = Some(Mimic { joint: driver.clone(), curve });
        }
    }
    Ok(out)
}

fn find_loop(
    desc: &RobotDescription,
    tree: &Robot,
    closure: &LoopClosure,
    joint_of: &HashMap<&str, usize>,
) -> Result<Loop> {
    let link = |name: &str| {
        tree.links
            .iter()
            .position(|l| l.name == name)
            .with_context(|| format!("loop '{}' names link '{name}', which is not part of the robot", closure.name))
    };
    let links = [link(&closure.links[0])?, link(&closure.links[1])?];
    // The moving joints above one link and not the other close the loop.
    let (a, b) = (&tree.links[links[0]].chain, &tree.links[links[1]].chain);
    let joints: Vec<usize> =
        a.iter().filter(|i| !b.contains(i)).chain(b.iter().filter(|i| !a.contains(i))).copied().collect();
    let mut dofs: Vec<usize> =
        joints.iter().map(|&i| tree.links[i].joint.actuation().expect("chain links move").0).collect();
    dofs.sort_unstable();
    dofs.dedup();
    let desc_joint = |dof: usize| &desc.joints[joint_of[tree.joint_names()[dof].as_str()]];
    let (driven, passive): (Vec<usize>, Vec<usize>) = dofs.into_iter().partition(|&d| desc_joint(d).actuated);
    let driver = match driven[..] {
        [d] => d,
        [] => bail!("loop '{}' has no actuated joint to drive it", closure.name),
        _ => bail!(
            "loop '{}' is driven by several joints ({}), which is not supported",
            closure.name,
            driven.iter().map(|&d| tree.joint_names()[d].as_str()).collect::<Vec<_>>().join(", ")
        ),
    };
    ensure!(!passive.is_empty(), "loop '{}' has no passive joint to close it", closure.name);
    let passive =
        passive.into_iter().map(|d| (d, desc_joint(d).stiffness as f64, desc_joint(d).spring_ref as f64)).collect();
    // A second anchor left open is wherever the first is with every joint at zero.
    let second = closure.anchors.1.unwrap_or_else(|| {
        let fk = tree.fk(&vec![0.0; tree.dof()]);
        let world = fk.rot[links[0]] * closure.anchors.0 + fk.pos[links[0]];
        fk.rot[links[1]].transpose() * (world - fk.pos[links[1]])
    });
    Ok(Loop { links, anchors: [closure.anchors.0, second], driver, passive })
}

/// The gap between the loop's two points at `q`, and its derivative by each passive joint.
fn gap(tree: &Robot, l: &Loop, q: &[f32]) -> (DVec3, Vec<DVec3>) {
    let fk = tree.fk(q);
    let point = |e: usize| fk.rot[l.links[e]] * l.anchors[e] + fk.pos[l.links[e]];
    let (p0, p1) = (point(0), point(1));
    // A point on link `link` moves with each moving joint above it (mimics adding into their leader).
    let rate = |e: usize, p: Vec3, dof: usize| {
        tree.chain(l.links[e])
            .filter(|&(_, d)| d == dof)
            .map(|(i, _)| {
                (fk.rate(i, &tree.links[i].joint.actuation().expect("chain links move").1) * fk.dpoint(i, p)).as_dvec3()
            })
            .sum::<DVec3>()
    };
    let derivatives = l.passive.iter().map(|&(dof, ..)| rate(0, p0, dof) - rate(1, p1, dof)).collect();
    ((p0 - p1).as_dvec3(), derivatives)
}

/// The passive joints' equilibrium with the driver at `q[l.driver]`, starting from `q`; written
/// into `q`. Projected Levenberg-Marquardt on closure, springs and staying put.
fn settle(tree: &Robot, l: &Loop, q: &mut [f32]) -> f64 {
    let m = l.passive.len();
    let previous: Vec<f64> = l.passive.iter().map(|&(d, ..)| q[d] as f64).collect();
    let cost = |q: &[f32]| {
        let (g, _) = gap(tree, l, q);
        let springs: f64 = l.passive.iter().map(|&(d, k, rest)| k * (q[d] as f64 - rest).powi(2)).sum();
        let stay: f64 = l.passive.iter().zip(&previous).map(|(&(d, ..), p)| STAY * (q[d] as f64 - p).powi(2)).sum();
        CLOSURE_WEIGHT * g.length_squared() + springs + stay
    };
    let mut damping = 1e-3;
    let mut current = cost(q);
    for _ in 0..200 {
        let (g, dg) = gap(tree, l, q);
        // Normal equations of the residuals sqrt(W) gap, sqrt(k)(p - rest) and sqrt(STAY)(p - previous).
        let mut a = vec![0.0f64; m * m];
        let mut b = vec![0.0f64; m];
        for i in 0..m {
            for j in 0..m {
                a[i * m + j] = CLOSURE_WEIGHT * dg[i].dot(dg[j]);
            }
            let (d, k, rest) = l.passive[i];
            a[i * m + i] += k + STAY;
            b[i] = -(CLOSURE_WEIGHT * dg[i].dot(g) + k * (q[d] as f64 - rest) + STAY * (q[d] as f64 - previous[i]));
        }
        let mut accepted = false;
        while damping < 1e12 {
            let mut damped = a.clone();
            for i in 0..m {
                damped[i * m + i] *= 1.0 + damping;
            }
            let Some(step) = solve_spd(&damped, &b, m) else {
                damping *= 10.0;
                continue;
            };
            let mut trial = q.to_vec();
            for (i, &(d, ..)) in l.passive.iter().enumerate() {
                let (lo, hi) = tree.bounds(d);
                trial[d] = (q[d] as f64 + step[i]).clamp(lo as f64, hi as f64) as f32;
            }
            let next = cost(&trial);
            if next < current {
                q.copy_from_slice(&trial);
                (current, damping, accepted) = (next, (damping * 0.3).max(1e-9), true);
                break;
            }
            damping *= 10.0;
        }
        if !accepted {
            break;
        }
    }
    gap(tree, l, q).0.length()
}

/// Each passive joint's curve over the driver's range.
fn solve(tree: &Robot, l: &Loop) -> Result<Vec<Curve>> {
    let (lo, hi) = (tree.lower()[l.driver] as f64, tree.upper()[l.driver] as f64);
    let at = |k: usize| lo + (hi - lo) * k as f64 / SAMPLES as f64;
    // Sweep out from the sample nearest zero, where the description assembles the loop.
    let start = (0..=SAMPLES).min_by(|&a, &b| at(a).abs().total_cmp(&at(b).abs())).expect("samples exist");
    let mut values = vec![vec![0.0f64; SAMPLES + 1]; l.passive.len()];
    for sweep in [(start..=SAMPLES).collect::<Vec<_>>(), (0..=start).rev().collect()] {
        let mut q = vec![0.0f32; tree.dof()];
        for k in sweep {
            q[l.driver] = at(k) as f32;
            let left = settle(tree, l, &mut q);
            ensure!(
                left < CLOSURE_TOLERANCE / 10.0,
                "no closed configuration with the driver at {:.4}: {left:.2e} m apart",
                at(k)
            );
            for (v, &(d, ..)) in values.iter_mut().zip(&l.passive) {
                v[k] = q[d] as f64;
            }
        }
    }
    let xs: Vec<f64> = (0..=SAMPLES).map(at).collect();
    let curves: Vec<Curve> = values.iter().map(|v| fit_quartic(&xs, v)).collect();
    // The fitted curves must close the loop between the samples too.
    let mut q = vec![0.0f32; tree.dof()];
    for k in 0..=4 * SAMPLES {
        let x = lo + (hi - lo) * k as f64 / (4 * SAMPLES) as f64;
        q[l.driver] = x as f32;
        for (c, &(d, ..)) in curves.iter().zip(&l.passive) {
            q[d] = c.value(x as f32);
        }
        let left = gap(tree, l, &q).0.length();
        ensure!(
            left <= CLOSURE_TOLERANCE,
            "its passive joints do not follow a quartic of '{}': {left:.2e} m apart at {x:.4}",
            tree.joint_names()[l.driver]
        );
    }
    Ok(curves)
}

/// Least-squares quartic through `(xs, ys)`, fitted on `xs` mapped to [-1, 1] for conditioning.
fn fit_quartic(xs: &[f64], ys: &[f64]) -> Curve {
    let (lo, hi) = (xs[0], xs[xs.len() - 1]);
    let (mid, half) = ((lo + hi) / 2.0, ((hi - lo) / 2.0).max(1e-12));
    let mut a = [0.0f64; 25];
    let mut b = [0.0f64; 5];
    for (&x, &y) in xs.iter().zip(ys) {
        let t = (x - mid) / half;
        let powers = [1.0, t, t * t, t * t * t, t * t * t * t];
        for i in 0..5 {
            for j in 0..5 {
                a[i * 5 + j] += powers[i] * powers[j];
            }
            b[i] += powers[i] * y;
        }
    }
    let c = solve_spd(&a, &b, 5).expect("five distinct samples make the fit well posed");
    // Back from t = (x - mid) / half to x: expand each c_k t^k.
    let mut out = [0.0f64; 5];
    for (k, &ck) in c.iter().enumerate() {
        // (x - mid)^k / half^k = sum_j binom(k, j) x^j (-mid)^(k - j) / half^k
        for (j, o) in out.iter_mut().enumerate().take(k + 1) {
            let binom = (1..=j).fold(1.0, |acc, i| acc * (k - j + i) as f64 / i as f64);
            *o += ck * binom * (-mid).powi((k - j) as i32) / half.powi(k as i32);
        }
    }
    Curve(out.map(|v| v as f32))
}

/// Solves `a x = b` for symmetric positive definite `a` (`n` x `n`, row-major) by Cholesky.
fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s = a[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        y[i] = (b[i] - (0..i).map(|k| l[i * n + k] * y[k]).sum::<f64>()) / l[i * n + i];
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        x[i] = (y[i] - (i + 1..n).map(|k| l[k * n + i] * x[k]).sum::<f64>()) / l[i * n + i];
    }
    Some(x)
}
