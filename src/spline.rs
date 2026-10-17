//! Uniform cubic B-splines, the shape of every planned path. Span `k` of a spline blends control
//! points `k..k + 4`; a path's first three and last three control points are equal, so it starts
//! and ends at rest. `basis` in kernels.wgsl mirrors `basis`.

/// Weights of the four control points of a span for the position at `u` in [0, 1].
#[inline]
pub(crate) fn basis(u: f32) -> [f32; 4] {
    let (u2, u3, v) = (u * u, u * u * u, 1.0 - u);
    [v * v * v / 6.0, (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0, (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0, u3 / 6.0]
}

/// Weights for the first derivative with respect to `u`.
#[inline]
pub(crate) fn basis_d1(u: f32) -> [f32; 4] {
    let v = 1.0 - u;
    [-0.5 * v * v, 0.5 * (3.0 * u * u - 4.0 * u), 0.5 * (-3.0 * u * u + 2.0 * u + 1.0), 0.5 * u * u]
}

/// Weights for the second derivative with respect to `u`.
#[inline]
pub(crate) fn basis_d2(u: f32) -> [f32; 4] {
    [1.0 - u, 3.0 * u - 2.0, 1.0 - 3.0 * u, u]
}

/// Weights for the third derivative, constant along a span.
pub(crate) const BASIS_D3: [f32; 4] = [-1.0, 3.0, -3.0, 1.0];

/// Where along span `s` (of `per_span`) collision is sampled during optimization.
#[inline]
pub(crate) fn sample_u(s: usize, per_span: usize) -> f32 {
    (s as f32 + 0.5) / per_span as f32
}

/// `Σ weights[i] * cp[span + i]` into `out`, for control points `cp` (`[points, dof]`).
#[inline]
pub(crate) fn blend(cp: &[f32], dof: usize, span: usize, weights: [f32; 4], out: &mut [f32]) {
    for (j, o) in out.iter_mut().enumerate().take(dof) {
        *o = (0..4).map(|i| weights[i] * cp[(span + i) * dof + j]).sum();
    }
}

/// The spline at `per_span` evenly spaced points of every span plus its exact end, written into
/// `out`, which holds `[(points - 3) * per_span + 1, dof]`.
pub(crate) fn dense(cp: &[f32], dof: usize, per_span: usize, out: &mut [f32]) {
    let spans = cp.len() / dof - 3;
    for span in 0..spans {
        for s in 0..per_span {
            let row = (span * per_span + s) * dof;
            blend(cp, dof, span, basis(s as f32 / per_span as f32), &mut out[row..row + dof]);
        }
    }
    out[spans * per_span * dof..].copy_from_slice(&cp[cp.len() - dof..]);
}

/// The spline sampled for a collision check: every span at `per_span` evenly spaced points, or at
/// more where it moves farther, so neighbouring samples lie at most `spacing` apart (joint-space
/// distance), as edges are checked; then its exact end. `starts` gets the index of each span's
/// first sample, then the end's.
pub(crate) fn checked(
    cp: &[f32],
    dof: usize,
    per_span: usize,
    spacing: f32,
    out: &mut Vec<f32>,
    starts: &mut Vec<usize>,
) {
    let spans = cp.len() / dof - 3;
    let gap = |a: usize, b: usize| (0..dof).map(|j| (cp[b * dof + j] - cp[a * dof + j]).powi(2)).sum::<f32>().sqrt();
    out.clear();
    starts.clear();
    for span in 0..spans {
        // A span moves at most three times its Bezier control polygon's longest edge per unit of
        // its parameter: |P2 - P0| / 2, |P2 - P1| or |P3 - P1| / 2.
        let speed = (0.5 * gap(span, span + 2)).max(gap(span + 1, span + 2)).max(0.5 * gap(span + 1, span + 3));
        let count = per_span.max((speed / spacing).ceil() as usize);
        starts.push(out.len() / dof);
        for s in 0..count {
            let row = out.len();
            out.resize(row + dof, 0.0);
            blend(cp, dof, span, basis(s as f32 / count as f32), &mut out[row..]);
        }
    }
    starts.push(out.len() / dof);
    out.extend_from_slice(&cp[cp.len() - dof..]);
}

/// The point a fraction `phase` in [0, 1] of the way along the spline.
pub(crate) fn at_phase(cp: &[f32], dof: usize, phase: f32, out: &mut [f32]) {
    let spans = cp.len() / dof - 3;
    let x = phase.clamp(0.0, 1.0) * spans as f32;
    let span = (x as usize).min(spans - 1);
    blend(cp, dof, span, basis(x - span as f32), out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checked samples lie no farther apart than the spacing where spans are long, keep `per_span`
    /// where they are short, start each span where `starts` says, and end at the spline's end.
    #[test]
    fn checked_samples_are_spaced_as_edges_are() {
        let dof = 2;
        // Spans from 1 cm to 3 rad long, with sharp turns between them.
        let cp: Vec<f32> =
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 3.0, 0.0, 3.0, 2.0, 0.5, 2.5, 0.5, 2.5, 0.5, 2.5].to_vec();
        let (per_span, spacing) = (8, 1.0 / 32.0);
        let (mut out, mut starts) = (vec![], vec![]);
        checked(&cp, dof, per_span, spacing, &mut out, &mut starts);
        let samples: Vec<&[f32]> = out.chunks(dof).collect();
        let spans = cp.len() / dof - 3;
        assert_eq!(starts.len(), spans + 1);
        assert_eq!(*samples.last().unwrap(), &cp[cp.len() - dof..]);
        let gap = |a: &[f32], b: &[f32]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        let widest = samples.windows(2).map(|w| gap(w[0], w[1])).fold(0.0, f32::max);
        assert!(widest <= spacing * 1.0001, "samples {widest} apart");
        for span in 0..spans {
            let count = starts[span + 1] - starts[span];
            assert!(count >= per_span, "span {span} has {count} samples");
            let mut at = [0.0; 2];
            blend(&cp, dof, span, basis(0.0), &mut at);
            assert!(gap(samples[starts[span]], &at) < 1e-6, "span {span} starts elsewhere");
        }
        // Short spans keep the fixed count: the first span moves 1 cm.
        assert_eq!(starts[1] - starts[0], per_span);
    }
}
