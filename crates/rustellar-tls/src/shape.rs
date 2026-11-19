//! The best fit's transit shape: the planet's radius ratio `k` and impact
//! parameter `b` with the period, mid-transit time and duration, fitted
//! together where the search's template fixes `k` and `b`.
//!
//! The model is the template's planet, free: for the `n`th transit,
//! `x = (t - T0 - n P) / D` with `D` the time from first to last contact,
//! the separation `z = sqrt(b² + ((1 + k)² - b²) (2x)²)` and the dimming
//! `1 - F(z)` for `|x| < 1/2`, else 0, with `F` the flux of
//! `rustellar-model` and the template's limb darkening. Its depth is
//! `1 - F(b)` itself, not a template's shape scaled, so the fit gives the
//! planet's depth and duration without the template's bias.
//!
//! Fitted by Levenberg–Marquardt (derivatives by central differences) from
//! the search's best fit, from a few impact parameters: `k` and `b` trade
//! against each other (a grazing small planet and a central large one dim
//! alike), and the chi-squared can hold a least in each.
//!
//! Where the cadence does not resolve ingress and egress (minutes long for
//! small planets, against 30-minute points), `k` and `b` trade along a
//! curved valley: their errors from the curvature can then be far too
//! small, and the duration and depth are no better than the template's. Where it is resolved (2-minute points, a strong transit),
//! they are several times better.

use rustellar_model::{LimbDarkening, RadiusRatio, flux};

use crate::errors::invert;
use crate::fit::{Points, Position};

/// The fitted transit, with standard errors where the curvature fixes them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Shape {
    pub(crate) period: f64,
    pub(crate) t0: f64,
    /// From first to last contact (T14), in days.
    pub(crate) duration: f64,
    pub(crate) k: f64,
    pub(crate) b: f64,
    /// The dimming at mid-transit, `1 - F(b)`.
    pub(crate) depth: f64,
    /// The semi-major axis in stellar radii, from the period, duration,
    /// `k` and `b` on a circular orbit seen nearly edge on.
    pub(crate) a: f64,
    pub(crate) chi2: f64,
    /// Standard errors of period, T0, duration, k and b, or `None` where
    /// the curvature does not fix them all.
    pub(crate) errors: Option<[f64; 5]>,
}

/// The parameters: period, T0, duration, k, b.
type Params = [f64; 5];

/// The largest radius ratio fitted: a body as large as the star.
const K_MAX: f64 = 1.0;

/// Impact parameters the fit starts from.
const STARTS: [f64; 4] = [0.1, 0.45, 0.75, 0.9];

/// The dimming of the model `p` at time `t`.
fn model(p: &Params, ld: LimbDarkening, t: f64) -> f64 {
    let [period, t0, duration, k, b] = *p;
    let n = ((t - t0) / period + 0.5).floor();
    let two_x = 2.0 * (t - n.mul_add(period, t0)) / duration;
    if two_x.abs() >= 1.0 {
        return 0.0;
    }
    let reach = 1.0 + k;
    let chord = (reach - b) * (reach + b);
    let z = b.mul_add(b, chord * two_x * two_x).sqrt();
    let k = RadiusRatio::new(k).expect("a radius ratio in (0, 1]");
    1.0 - flux(z, k, ld)
}

/// Whether `p` is a transiting planet the model takes.
fn valid(p: &Params) -> bool {
    let [period, _, duration, k, b] = *p;
    period > 0.0
        && duration > 0.0
        && duration < period
        && k > 0.0
        && k <= K_MAX
        && b >= 0.0
        && b < 1.0 + k
}

/// The steps for the derivatives: small against each parameter's scale.
fn steps(p: &Params, span: f64) -> Params {
    let [period, _, duration, k, _] = *p;
    let h = 1e-4 * duration;
    [h * period / span.max(period), h, h, 1e-4 * k, 1e-4]
}

/// The chi-squared of `p` less the flat model's: only points it dims count.
fn chi2(points: &Points, p: &Params, ld: LimbDarkening) -> f64 {
    let mut sum = 0.0;
    for ((&t, &r), &w) in points.time.iter().zip(&points.residual).zip(&points.weight) {
        let m = model(p, ld, t);
        if m != 0.0 {
            sum += w * m * (m - 2.0 * r);
        }
    }
    sum
}

/// The normal equations at `p`: `Jᵀ W J` and `Jᵀ W (r - m)`.
fn normal(points: &Points, p: &Params, ld: LimbDarkening, h: &Params) -> ([Params; 5], Params) {
    let (mut a, mut g) = ([[0.0; 5]; 5], [0.0; 5]);
    // Only points near a transit move with the parameters.
    let near = {
        let mut wide = *p;
        wide[2] = (p[2] * 1.5).min(p[0] * 0.99);
        wide[4] = 0.0;
        wide
    };
    for ((&t, &r), &w) in points.time.iter().zip(&points.residual).zip(&points.weight) {
        if model(&near, ld, t) == 0.0 {
            continue;
        }
        let mut j = [0.0; 5];
        for (i, ji) in j.iter_mut().enumerate() {
            let (mut up, mut down) = (*p, *p);
            up[i] += h[i];
            down[i] -= h[i];
            // b at 0: the model is even in b.
            down[4] = down[4].abs();
            *ji = (model(&up, ld, t) - model(&down, ld, t)) / (2.0 * h[i]);
        }
        let d = r - model(p, ld, t);
        for (row, (&ji, gi)) in a.iter_mut().zip(j.iter().zip(&mut g)) {
            *gi += w * ji * d;
            for (cell, &jk) in row.iter_mut().zip(&j) {
                *cell += w * ji * jk;
            }
        }
    }
    (a, g)
}

/// Levenberg–Marquardt from `start`: the least found and its chi-squared.
fn descend(points: &Points, start: Params, ld: LimbDarkening, span: f64) -> (Params, f64) {
    let mut p = start;
    let mut current = chi2(points, &p, ld);
    let mut lambda = 1e-3;
    for _ in 0..200 {
        let h = steps(&p, span);
        let (a, g) = normal(points, &p, ld, &h);
        let mut improved = false;
        while lambda < 1e12 {
            let mut damped = a;
            for (i, row) in damped.iter_mut().enumerate() {
                row[i] *= 1.0 + lambda;
            }
            // A step that cannot be solved for is damped further, as one
            // that does not lower the chi-squared.
            let step = invert(damped).map(|inverse| {
                let mut next = p;
                for (n, row) in next.iter_mut().zip(&inverse) {
                    *n += row.iter().zip(&g).map(|(x, y)| x * y).sum::<f64>();
                }
                next[4] = next[4].abs();
                next
            });
            if let Some(next) = step.filter(valid) {
                let tried = chi2(points, &next, ld);
                if tried < current {
                    let gain = current - tried;
                    (p, current) = (next, tried);
                    lambda = (lambda / 10.0).max(1e-12);
                    improved = gain > 1e-9 * current.abs().max(1.0);
                    break;
                }
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    (p, current)
}

/// The radius ratio whose transit at impact parameter `b` dims by `depth`
/// at mid-transit, by a few fixed-point steps from `sqrt(depth)`.
fn k_for(depth: f64, b: f64, ld: LimbDarkening) -> f64 {
    // k stays in (0, 1]: each step scales it by a positive factor, capped;
    // a planet too small to dim anything in doubles is sent to the cap.
    let mut k = depth.max(1e-8).sqrt().min(K_MAX);
    for _ in 0..8 {
        let radius = RadiusRatio::new(k).expect("a radius ratio in (0, 1]");
        let dims = 1.0 - flux(b, radius, ld);
        k = (k * (depth / dims.max(f64::MIN_POSITIVE)).sqrt()).min(K_MAX);
    }
    k
}

/// The transit shape fitted to `points` from the search's best fit at
/// `start` with mid-transit `depth`, with limb darkening `ld`; `None` if no
/// start gives a planet that dims any point.
pub(crate) fn shape(
    points: &Points,
    start: Position,
    depth: f64,
    ld: LimbDarkening,
) -> Option<Shape> {
    let span = points.time[points.time.len() - 1] - points.time[0];
    let flat: f64 = points
        .residual
        .iter()
        .zip(&points.weight)
        .map(|(&r, &w)| w * r * r)
        .sum();
    let mut best: Option<(Params, f64)> = None;
    for b in STARTS {
        let k = k_for(depth, b, ld);
        let p = [start.period, start.t0, start.duration, k, b];
        if !valid(&p) || chi2(points, &p, ld) >= 0.0 {
            continue;
        }
        let found = descend(points, p, ld, span);
        if best.is_none_or(|b| found.1 < b.1) {
            best = Some(found);
        }
    }
    let (p, chi2) = best?;
    let [period, t0, duration, k, b] = p;
    let radius = RadiusRatio::new(k).expect("a radius ratio in (0, 1]");
    let (a_matrix, _) = normal(points, &p, ld, &steps(&p, span));
    let errors = invert(a_matrix).map(|c| [0, 1, 2, 3, 4].map(|i| c[i][i].sqrt()));
    let chord = ((1.0 + k) * (1.0 + k) - b * b).sqrt();
    Some(Shape {
        period,
        t0,
        duration,
        k,
        b,
        depth: 1.0 - flux(b, radius, ld),
        // The chord 2 chord stellar radii crossed in D at speed 2 pi a / P.
        a: chord * period / (std::f64::consts::PI * duration),
        chi2: flat + chi2,
        errors,
    })
}
