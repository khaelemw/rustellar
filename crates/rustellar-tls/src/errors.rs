//! What TLS does not give: standard errors of the best fit, each transit's
//! own depth, and the odd and even transits' depths compared.
//!
//! The model is `m = d s(x)` with `x = (t - T0 - n P) / D` for the `n`th
//! transit. Near its least, the chi-squared `sum w (r - m)²` is a quadratic
//! in the four parameters `(P, T0, D, d)` with curvature `2 F`, where `F`
//! is the Fisher matrix `sum w J Jᵀ` and `J` the model's derivatives:
//!
//! ```text
//! dm/dd = s,  dm/dT0 = -d s'(x) / D,  dm/dP = -n d s'(x) / D,
//! dm/dD = -x d s'(x) / D
//! ```
//!
//! The covariance is `F⁻¹` (errors taken as given, as the chi-squared
//! does), and the standard errors the roots of its diagonal. Each transit's
//! depth is the least-squares depth of its points alone, with error
//! `1 / sqrt(sum w s²)`.

use crate::fit::{Points, Position};
use crate::template::Template;

/// Standard errors of the best fit's parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Errors {
    pub(crate) period: f64,
    /// Of the mid-transit time the fit gives (the first transit's).
    pub(crate) t0: f64,
    pub(crate) duration: f64,
    pub(crate) depth: f64,
}

/// One transit's own depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Transit {
    /// The transit's number from the fit's mid-transit time, and its time.
    pub(crate) number: i64,
    pub(crate) time: f64,
    /// The least-squares depth of its points alone, and its error.
    pub(crate) depth: f64,
    pub(crate) error: f64,
    /// Its points with the template above zero.
    pub(crate) points: usize,
}

/// The odd and even transits' depths, each fitted alone, and how many of
/// their errors apart they are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OddEven {
    pub(crate) odd: f64,
    pub(crate) odd_error: f64,
    pub(crate) even: f64,
    pub(crate) even_error: f64,
    pub(crate) sigma: f64,
}

/// The template's slope at `x`, by central differences: the shape is
/// smooth inside the transit, and only points within `H` of a contact feel
/// its kink there.
fn slope(template: &Template, x: f64) -> f64 {
    const H: f64 = 1e-6;
    (template.shape(x + H) - template.shape(x - H)) / (2.0 * H)
}

/// The transit number and argument of the template at time `t`.
fn place(position: Position, t: f64) -> (f64, f64) {
    let n = ((t - position.t0) / position.period + 0.5).floor();
    let x = (t - n.mul_add(position.period, position.t0)) / position.duration;
    (n, x)
}

/// The standard errors of the fit of `template` at `position` with `depth`
/// to `points`, or `None` where the curvature does not fix all four
/// parameters (too few points in transit).
pub(crate) fn errors(
    points: &Points,
    template: &Template,
    position: Position,
    depth: f64,
) -> Option<Errors> {
    let mut fisher = [[0.0; 4]; 4];
    for (&t, &w) in points.time.iter().zip(&points.weight) {
        let (n, x) = place(position, t);
        if x.abs() >= 0.5 {
            continue;
        }
        let s = template.shape(x);
        let ds = depth * slope(template, x) / position.duration;
        let j = [-n * ds, -ds, -x * ds, s];
        for (row, &a) in fisher.iter_mut().zip(&j) {
            for (cell, &b) in row.iter_mut().zip(&j) {
                *cell += w * a * b;
            }
        }
    }
    let covariance = invert(fisher)?;
    let error = |i: usize| covariance[i][i].sqrt();
    Some(Errors {
        period: error(0),
        t0: error(1),
        duration: error(2),
        depth: error(3),
    })
}

/// The inverse of a symmetric positive definite matrix, by Cholesky; `None`
/// if it is not positive definite.
fn invert(a: [[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    // a = l lᵀ, l lower triangular.
    let mut l = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..=i {
            let sum = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if sum.is_nan() || sum <= 0.0 {
                    return None;
                }
                l[i][i] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // Each column of the inverse: l lᵀ x = e.
    let mut inverse = [[0.0; 4]; 4];
    for c in 0..4 {
        let mut y = [0.0; 4];
        for i in 0..4 {
            let e = if i == c { 1.0 } else { 0.0 };
            y[i] = (e - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
        }
        let mut x = [0.0; 4];
        for i in (0..4).rev() {
            x[i] = (y[i] - (i + 1..4).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
        }
        for (row, &v) in inverse.iter_mut().zip(&x) {
            row[c] = v;
        }
    }
    Some(inverse)
}

/// Each transit's own depth at `position`, in time order: those with
/// points where the template is above zero.
pub(crate) fn transits(points: &Points, template: &Template, position: Position) -> Vec<Transit> {
    let mut found: Vec<(i64, f64, f64, usize)> = Vec::new();
    for ((&t, &r), &w) in points.time.iter().zip(&points.residual).zip(&points.weight) {
        let (n, x) = place(position, t);
        let s = template.shape(x);
        if s <= 0.0 {
            continue;
        }
        #[expect(clippy::cast_possible_truncation, reason = "a transit number")]
        let n = n as i64;
        match found.last_mut() {
            Some(last) if last.0 == n => {
                last.1 += w * r * s;
                last.2 += w * s * s;
                last.3 += 1;
            }
            _ => found.push((n, w * r * s, w * s * s, 1)),
        }
    }
    found
        .into_iter()
        .map(|(number, rs, ss, count)| {
            #[expect(clippy::cast_precision_loss, reason = "a transit number")]
            let time = position.period.mul_add(number as f64, position.t0);
            Transit {
                number,
                time,
                depth: rs / ss,
                error: ss.sqrt().recip(),
                points: count,
            }
        })
        .collect()
}

/// The odd and even transits' depths from `transits`, or `None` without
/// both.
pub(crate) fn odd_even(transits: &[Transit]) -> Option<OddEven> {
    // The depth of a set of transits fitted together is their depths'
    // mean weighted by 1 / error².
    let mean = |odd: bool| {
        let (mut sum, mut weights) = (0.0, 0.0);
        for t in transits.iter().filter(|t| (t.number % 2 != 0) == odd) {
            let w = t.error.powi(-2);
            sum += w * t.depth;
            weights += w;
        }
        (weights > 0.0).then(|| (sum / weights, weights.sqrt().recip()))
    };
    let ((odd, odd_error), (even, even_error)) = (mean(true)?, mean(false)?);
    Some(OddEven {
        odd,
        odd_error,
        even,
        even_error,
        sigma: (odd - even).abs() / odd_error.hypot(even_error),
    })
}
