//! Levenberg-Marquardt least squares (Levenberg 1944; Marquardt 1963), in
//! the form of More (1978): the Jacobian factored by Householder QR, never
//! the normal equations, each damped step solved from `R` by Givens
//! rotations, and the parameters scaled by the Jacobian's column norms. The
//! damping follows Nielsen (1999), and each step can carry the geodesic
//! acceleration of Transtrum & Sethna (2012), a second-order correction
//! that follows a curved valley in fewer steps.
//!
//! Steps that cross a bound are projected back onto it (as in projected
//! Levenberg-Marquardt, Kanzow, Yamashita & Fukushima 2004), so a fit can
//! slide along a bound rather than stall at it. The Jacobian is by central
//! differences, falling back to one side where the other leaves the
//! feasible region.

/// Why a fit stopped.
///
/// # Examples
///
/// ```
/// use rustellar_fit::Termination;
///
/// assert_ne!(Termination::Converged, Termination::MaxIterations);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Termination {
    /// A full Gauss-Newton step would move the parameters by less than
    /// `1e-7 |r| + 1e-10` of their standard errors, `|r|` the norm of the
    /// weighted residuals (a few millionths of a standard error for a good
    /// fit): the minimum to well within its uncertainty.
    Converged,
    /// No step lowered chi-squared before the steps fell to the parameters'
    /// rounding. The usual reason is a minimum on a bound, where the
    /// Gauss-Newton step keeps pointing past it: limb darkening at the
    /// edge of its valid range, or `b = 0`, where the flux depends on
    /// `b^2` and the Jacobian's column for `b` vanishes. The parameters are
    /// then the minimum within the bounds.
    PrecisionLimit,
    /// The iteration limit was reached first.
    MaxIterations,
}

/// Residuals `f(x)`, weighted, whose squared sum is minimised.
pub(crate) trait Residuals {
    /// How many residuals there are.
    fn len(&self) -> usize;

    /// The residuals at `x` into `out`, or `false` where `x` is outside
    /// the feasible region (`out` is then not used).
    fn eval(&mut self, x: &[f64], out: &mut [f64]) -> bool;

    /// Moves `x` onto the feasible region's boundary where it has crossed
    /// it, so steps can slide along a bound rather than stall at it.
    fn project(&self, x: &mut [f64]);
}

/// How the solver runs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Settings {
    pub max_iterations: usize,
    pub accelerate: bool,
}

/// Where the solver stopped.
#[derive(Debug, Clone)]
pub(crate) struct Solution {
    /// The parameters.
    pub x: Vec<f64>,
    /// The residuals there.
    pub f: Vec<f64>,
    /// `R` of the Jacobian's QR there, row-major and upper triangular.
    pub r: Vec<f64>,
    /// The steps accepted.
    pub iterations: usize,
    /// Every evaluation of the residuals, the Jacobian's included.
    pub evaluations: usize,
    pub termination: Termination,
}

/// The converged test's tolerance, in standard errors per unit of the
/// residual norm: above the Jacobian's differencing error, which grows with
/// the residuals.
const TOLERANCE: f64 = 1e-7;
/// The converged test's floor, in standard errors: above the rounding of
/// residuals that are all but zero.
const FLOOR: f64 = 1e-10;
/// The geodesic acceleration's finite-difference step, along the velocity
/// (Transtrum & Sethna 2012).
const ACCEL_STEP: f64 = 0.1;
/// The largest acceleration taken, against the velocity, in the scaled norm.
const ACCEL_RATIO: f64 = 0.75;

/// Minimises `sum f(x)^2` from `x0`, which must be feasible, with the
/// Jacobian's columns differenced over `steps`.
pub(crate) fn solve(
    p: &mut dyn Residuals,
    x0: Vec<f64>,
    steps: &[f64],
    settings: Settings,
) -> Solution {
    let m = p.len();
    let n = x0.len();
    let mut x = x0;
    let mut f = vec![0.0; m];
    let feasible = p.eval(&x, &mut f);
    debug_assert!(feasible, "the starting point is feasible");
    let mut evaluations = 1;

    let mut jac = vec![0.0; m * n];
    let mut qr = vec![0.0; m * n];
    let mut trial = vec![0.0; m];
    let mut spare = vec![0.0; m];
    let mut x_try = vec![0.0; n];
    let mut scale = vec![0.0_f64; n];
    let mut mu = 1e-3;
    let mut nu = 2.0;
    let mut iterations = 0;

    loop {
        evaluations += jacobian(p, &x, &f, steps, &mut jac, &mut trial, &mut spare);
        qr.copy_from_slice(&jac);
        let (r, c) = householder(&mut qr, m, n, &f);
        for (j, s) in scale.iter_mut().enumerate() {
            *s = (*s).max(norm(&jac[j * m..(j + 1) * m]));
        }
        let unit = scale.iter().map(|&s| if s > 0.0 { s } else { 1.0 });
        let d: Vec<f64> = unit.collect();

        let termination = if norm(&c) <= TOLERANCE.mul_add(norm(&f), FLOOR) {
            Some(Termination::Converged)
        } else if iterations == settings.max_iterations {
            Some(Termination::MaxIterations)
        } else {
            None
        };
        if let Some(termination) = termination {
            return Solution {
                x,
                f,
                r,
                iterations,
                evaluations,
                termination,
            };
        }

        // Damped steps until one lowers chi-squared.
        loop {
            let (s, mut delta) = damped(&r, &c, &d, mu);
            if settings.accelerate {
                for ((xt, &xi), &v) in x_try.iter_mut().zip(&x).zip(&delta) {
                    *xt = ACCEL_STEP.mul_add(v, xi);
                }
                evaluations += 1;
                if p.eval(&x_try, &mut trial) {
                    let accel = acceleration(&jac, &s, &f, &trial, &delta, m);
                    if 2.0 * scaled_norm(&accel, &d) <= ACCEL_RATIO * scaled_norm(&delta, &d) {
                        for (dj, aj) in delta.iter_mut().zip(&accel) {
                            *dj = 0.5f64.mul_add(*aj, *dj);
                        }
                    }
                }
            }
            let stalled = delta
                .iter()
                .zip(&x)
                .zip(steps)
                .all(|((dj, xj), hj)| dj.abs() <= f64::EPSILON * (xj.abs() + hj));
            if stalled {
                return Solution {
                    x,
                    f,
                    r,
                    iterations,
                    evaluations,
                    termination: Termination::PrecisionLimit,
                };
            }
            for ((xt, &xi), &dj) in x_try.iter_mut().zip(&x).zip(&delta) {
                *xt = xi + dj;
            }
            p.project(&mut x_try);
            for ((dj, &xt), &xi) in delta.iter_mut().zip(&x_try).zip(&x) {
                *dj = xt - xi;
            }
            let moved = x_try != x;
            if moved && {
                evaluations += 1;
                p.eval(&x_try, &mut trial)
            } {
                let actual = 0.5 * (dot(&f, &f) - dot(&trial, &trial));
                if actual > 0.0 {
                    // A gain ratio above 1, or a prediction of no gain,
                    // counts as 1: Nielsen's update is the same for both.
                    let predicted = predicted_reduction(&r, &c, &delta);
                    let rho = actual / predicted.max(actual);
                    x.copy_from_slice(&x_try);
                    std::mem::swap(&mut f, &mut trial);
                    mu *= (1.0 / 3.0f64).max(1.0 - (2.0f64.mul_add(rho, -1.0)).powi(3));
                    nu = 2.0;
                    iterations += 1;
                    break;
                }
            }
            mu *= nu;
            nu *= 2.0;
        }
    }
}

/// The Jacobian of `f` at `x` into `jac` (column-major, `m` by `n`), by
/// central differences, or one side where the other is infeasible; a
/// column with neither side feasible is zero. Returns the evaluations.
fn jacobian(
    p: &mut dyn Residuals,
    x: &[f64],
    f: &[f64],
    steps: &[f64],
    jac: &mut [f64],
    plus: &mut [f64],
    minus: &mut [f64],
) -> usize {
    let m = f.len();
    let mut xs = x.to_vec();
    for (j, &h) in steps.iter().enumerate() {
        // The steps as represented, so rounding does not bias the slope.
        xs[j] = x[j] + h;
        let up = xs[j] - x[j];
        let has_up = p.eval(&xs, plus);
        xs[j] = x[j] - h;
        let down = x[j] - xs[j];
        let has_down = p.eval(&xs, minus);
        xs[j] = x[j];
        let col = &mut jac[j * m..(j + 1) * m];
        match (has_up, has_down) {
            (true, true) => {
                let width = up + down;
                for ((c, a), b) in col.iter_mut().zip(&*plus).zip(&*minus) {
                    *c = (a - b) / width;
                }
            }
            (true, false) => {
                for ((c, a), b) in col.iter_mut().zip(&*plus).zip(f) {
                    *c = (a - b) / up;
                }
            }
            (false, true) => {
                for ((c, a), b) in col.iter_mut().zip(f).zip(&*minus) {
                    *c = (a - b) / down;
                }
            }
            (false, false) => col.fill(0.0),
        }
    }
    2 * steps.len()
}

/// Householder QR of `a` (column-major, `m` by `n`, overwritten): `R`
/// (row-major `n` by `n`) and the first `n` entries of `Q^T f`.
fn householder(a: &mut [f64], m: usize, n: usize, f: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut r = vec![0.0; n * n];
    let mut qtf = f.to_vec();
    for j in 0..n {
        let (left, right) = a.split_at_mut((j + 1) * m);
        let v = &mut left[j * m + j..];
        let length = norm(v);
        if length == 0.0 {
            // Nothing to reflect: R's row is the columns' entries as they
            // stand, with a zero pivot.
            for k in j + 1..n {
                r[j * n + k] = right[(k - j - 1) * m + j];
            }
            continue;
        }
        let alpha = if v[0] > 0.0 { -length } else { length };
        v[0] -= alpha;
        // |v|^2 = 2 |a| (|a| + |a_0|).
        let vv = length * (length + (v[0] + alpha).abs()) * 2.0;
        r[j * n + j] = alpha;
        for k in j + 1..n {
            let col = &mut right[(k - j - 1) * m + j..(k - j) * m];
            let s = 2.0 * dot(v, col) / vv;
            for (ci, vi) in col.iter_mut().zip(&*v) {
                *ci = (-s).mul_add(*vi, *ci);
            }
            r[j * n + k] = col[0];
        }
        let tail = &mut qtf[j..];
        let s = 2.0 * dot(v, tail) / vv;
        for (ti, vi) in tail.iter_mut().zip(&*v) {
            *ti = (-s).mul_add(*vi, *ti);
        }
    }
    qtf.truncate(n);
    (r, qtf)
}

/// The damped step `-(R^T R + mu D^2)^-1 R^T c`, by Givens rotations of the
/// rows `sqrt(mu) D` into `R` (More 1978): the triangle `S` with
/// `S^T S = R^T R + mu D^2`, and the step.
fn damped(r: &[f64], c: &[f64], d: &[f64], mu: f64) -> (Vec<f64>, Vec<f64>) {
    let n = c.len();
    let mut s = r.to_vec();
    let mut z = c.to_vec();
    let mut row = vec![0.0; n];
    let root = mu.sqrt();
    for j in 0..n {
        row.fill(0.0);
        row[j] = root * d[j];
        let mut extra = 0.0;
        for k in j..n {
            if row[k] == 0.0 {
                continue;
            }
            let skk = s[k * n + k];
            let (cos, sin) = if skk.abs() < row[k].abs() {
                let cot = skk / row[k];
                let sin = 1.0 / cot.mul_add(cot, 1.0).sqrt();
                (sin * cot, sin)
            } else {
                let tan = row[k] / skk;
                let cos = 1.0 / tan.mul_add(tan, 1.0).sqrt();
                (cos, cos * tan)
            };
            s[k * n + k] = cos.mul_add(skk, sin * row[k]);
            let zk = z[k];
            z[k] = cos.mul_add(zk, sin * extra);
            extra = (-sin).mul_add(zk, cos * extra);
            for i in k + 1..n {
                let ski = s[k * n + i];
                s[k * n + i] = cos.mul_add(ski, sin * row[i]);
                row[i] = (-sin).mul_add(ski, cos * row[i]);
            }
        }
    }
    let mut step = z;
    back_substitute(&s, &mut step);
    for v in &mut step {
        *v = -*v;
    }
    (s, step)
}

/// The geodesic acceleration (Transtrum & Sethna 2012): the second
/// directional derivative of `f` along `v` by differences, `f_vv`, and
/// `-(S^T S)^-1 J^T f_vv`.
fn acceleration(jac: &[f64], s: &[f64], f: &[f64], f_h: &[f64], v: &[f64], m: usize) -> Vec<f64> {
    let n = v.len();
    let scale = 2.0 / ACCEL_STEP;
    let mut g = vec![0.0; n];
    for (i, (&fi, &fh)) in f.iter().zip(f_h).enumerate() {
        let jv: f64 = (0..n).map(|j| jac[j * m + i] * v[j]).sum();
        let fvv = scale * ((fh - fi) / ACCEL_STEP - jv);
        for (j, gj) in g.iter_mut().enumerate() {
            *gj = jac[j * m + i].mul_add(fvv, *gj);
        }
    }
    // S^T y = -g, then S a = y.
    for i in 0..n {
        let sum: f64 = (0..i).map(|k| s[k * n + i] * g[k]).sum();
        g[i] = (-g[i] - sum) / s[i * n + i];
    }
    back_substitute(s, &mut g);
    g
}

/// Solves `S x = b` in place for upper triangular `S`.
fn back_substitute(s: &[f64], b: &mut [f64]) {
    let n = b.len();
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|k| s[i * n + k] * b[k]).sum();
        b[i] = (b[i] - sum) / s[i * n + i];
    }
}

/// The reduction of `|f|^2 / 2` the linear model predicts for `delta`:
/// `-c.(R delta) - |R delta|^2 / 2`.
fn predicted_reduction(r: &[f64], c: &[f64], delta: &[f64]) -> f64 {
    let n = c.len();
    let mut total = 0.0;
    for i in 0..n {
        let u: f64 = (i..n).map(|k| r[i * n + k] * delta[k]).sum();
        total -= c[i].mul_add(u, 0.5 * u * u);
    }
    total
}

fn scaled_norm(v: &[f64], d: &[f64]) -> f64 {
    v.iter()
        .zip(d)
        .map(|(a, b)| (a * b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}
