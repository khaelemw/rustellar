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
//! slide along a bound rather than stall at it. The Jacobian is the
//! residuals' own where they give one, else by central differences, falling
//! back to one side where the other leaves the feasible region.

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
    /// fit): the minimum to well within its uncertainty. At a minimum on a
    /// limb-darkening bound, the step along the bound.
    Converged,
    /// No step lowered chi-squared before the steps fell to the parameters'
    /// rounding. The usual reason is `b = 0`, where the flux depends on
    /// `b^2` and the Jacobian's column for `b` vanishes, so the minimum is
    /// flat to first order. The parameters are then the minimum.
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

    /// The Jacobian at `x` into `jac` (column-major), where the residuals
    /// know it better than by differencing them: `false` to have it
    /// differenced.
    fn jacobian(&mut self, _x: &[f64], _jac: &mut [f64]) -> bool {
        false
    }

    /// The bounds `x` lies on, each as its inward normal (`n` long; for a
    /// curved bound, its tangent's): a step leaving one is kept to it.
    fn bounds(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        Vec::new()
    }
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
    /// Every evaluation of the residuals, differenced Jacobians' included.
    pub evaluations: usize,
    /// The Jacobians the residuals gave themselves.
    pub jacobians: usize,
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
/// Iterations in a row pressing against a bound before steps follow it.
const PRESSED: usize = 2;

/// Minimises `sum f(x)^2` from `x0`, which must be feasible, with the
/// Jacobian the residuals' own, or its columns differenced over `steps`.
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
    let mut jacobians = 0;
    // Iterations in a row whose step would leave a bound.
    let mut pressed = 0;

    loop {
        if p.jacobian(&x, &mut jac) {
            jacobians += 1;
        } else {
            evaluations += jacobian(p, &x, &f, steps, &mut jac, &mut trial, &mut spare);
        }
        qr.copy_from_slice(&jac);
        let (r, c) = householder(&mut qr, m, n, &f);
        for (j, s) in scale.iter_mut().enumerate() {
            *s = (*s).max(norm(&jac[j * m..(j + 1) * m]));
        }
        let unit = scale.iter().map(|&s| if s > 0.0 { s } else { 1.0 });
        let d: Vec<f64> = unit.collect();

        // The step's direction at the current damping says which bounds
        // it would leave; the test is then the step along them.
        let (_, delta) = damped(&r, &c, &d, mu);
        let (_, face) = onto_face(p, &x, &r, &c, &d, mu, delta);
        pressed = if face.is_some() { pressed + 1 } else { 0 };
        let gain = face
            .as_ref()
            .map_or_else(|| norm(&c), |z| face_gain(&r, &c, z));
        let termination = if gain <= TOLERANCE.mul_add(norm(&f), FLOOR) {
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
                jacobians,
                termination,
            };
        }

        // Damped steps until one lowers chi-squared.
        loop {
            let (s, delta) = damped(&r, &c, &d, mu);
            // A bound passed on the way is crossed by projection; one the
            // fit keeps pressing against is followed, so the other
            // parameters move as they would along it.
            let (mut delta, face) = if pressed >= PRESSED {
                onto_face(p, &x, &r, &c, &d, mu, delta)
            } else {
                (delta, None)
            };
            // The acceleration is for the free step; along a bound, none.
            if settings.accelerate && face.is_none() {
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
                    jacobians,
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

/// The rows checked at once for being all zero in a Householder column.
const CHUNK: usize = 64;

/// Householder QR of `a` (column-major, `m` by `n`, overwritten): `R`
/// (row-major `n` by `n`) and the first `n` entries of `Q^T f`.
///
/// Each reflection touches only the rows where its column is nonzero, the
/// pivot's aside: out of transit only the baseline's column is, so the
/// cost is in the points in transit. Exact zeros add nothing to a sum, so
/// the result is the same to the bit as reflecting every row.
fn householder(a: &mut [f64], m: usize, n: usize, f: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut r = vec![0.0; n * n];
    let mut qtf = f.to_vec();
    let mut rows = Vec::with_capacity(m);
    for j in 0..n {
        let (left, right) = a.split_at_mut((j + 1) * m);
        let v = &mut left[j * m..(j + 1) * m];
        rows.clear();
        rows.push(j);
        // In chunks, so long runs of zeros are passed over quickly.
        for (c, chunk) in v[j + 1..].chunks(CHUNK).enumerate() {
            if chunk.iter().fold(0.0f64, |s, x| s.max(x.abs())) != 0.0 {
                let start = j + 1 + c * CHUNK;
                rows.extend((start..start + chunk.len()).filter(|&i| v[i] != 0.0));
            }
        }
        let length = rows.iter().map(|&i| v[i] * v[i]).sum::<f64>().sqrt();
        if length == 0.0 {
            // Nothing to reflect: R's row is the columns' entries as they
            // stand, with a zero pivot.
            for k in j + 1..n {
                r[j * n + k] = right[(k - j - 1) * m + j];
            }
            continue;
        }
        let alpha = if v[j] > 0.0 { -length } else { length };
        v[j] -= alpha;
        // |v|^2 = 2 |a| (|a| + |a_0|).
        let vv = length * (length + (v[j] + alpha).abs()) * 2.0;
        r[j * n + j] = alpha;
        for k in j + 1..n {
            let col = &mut right[(k - j - 1) * m..(k - j) * m];
            reflect(v, &rows, vv, col);
            r[j * n + k] = col[j];
        }
        reflect(v, &rows, vv, &mut qtf);
    }
    qtf.truncate(n);
    (r, qtf)
}

/// `x - 2 v (v . x) / vv` on `rows`, where `v` is nonzero.
fn reflect(v: &[f64], rows: &[usize], vv: f64, x: &mut [f64]) {
    let s = 2.0 * rows.iter().map(|&i| v[i] * x[i]).sum::<f64>() / vv;
    for &i in rows {
        x[i] = (-s).mul_add(v[i], x[i]);
    }
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
    // Column by column, as the Jacobian is stored.
    let mut fvv = vec![0.0; m];
    for (j, &vj) in v.iter().enumerate() {
        for (a, &c) in fvv.iter_mut().zip(&jac[j * m..(j + 1) * m]) {
            *a += c * vj;
        }
    }
    for ((a, &fi), &fh) in fvv.iter_mut().zip(f).zip(f_h) {
        *a = scale * ((fh - fi) / ACCEL_STEP - *a);
    }
    let mut g: Vec<f64> = (0..n)
        .map(|j| {
            jac[j * m..(j + 1) * m]
                .iter()
                .zip(&fvv)
                .fold(0.0, |gj, (&c, &a)| c.mul_add(a, gj))
        })
        .collect();
    // S^T y = -g, then S a = y.
    for i in 0..n {
        let sum: f64 = (0..i).map(|k| s[k * n + i] * g[k]).sum();
        g[i] = (-g[i] - sum) / s[i * n + i];
    }
    back_substitute(s, &mut g);
    g
}

/// The damped step kept to the bounds `x` lies on that `delta` would
/// leave, and those bounds' null space (`None` if it leaves none): the
/// bounds taken one round at a time, as a step along some can leave
/// others.
fn onto_face(
    p: &dyn Residuals,
    x: &[f64],
    r: &[f64],
    c: &[f64],
    d: &[f64],
    mu: f64,
    mut delta: Vec<f64>,
) -> (Vec<f64>, Option<Vec<Vec<f64>>>) {
    let bounds = p.bounds(x);
    let mut held: Vec<&[f64]> = Vec::new();
    let mut face = None;
    loop {
        let leaving: Vec<&[f64]> = bounds
            .iter()
            .map(Vec::as_slice)
            .filter(|b| !held.contains(b) && dot(b, &delta) < 0.0)
            .collect();
        if leaving.is_empty() {
            return (delta, face);
        }
        held.extend(leaving);
        let z = null_space(&held, x.len());
        delta = face_step(r, c, d, mu, &z);
        face = Some(z);
    }
}

/// `v` less its parts along `basis` (orthonormal), twice over for
/// orthogonality, and normalised; `None` if nothing is left of it.
fn orthogonal(mut v: Vec<f64>, basis: &[Vec<f64>]) -> Option<Vec<f64>> {
    let before = norm(&v);
    for _ in 0..2 {
        for u in basis {
            let along = dot(u, &v);
            for (vi, ui) in v.iter_mut().zip(u) {
                *vi = (-along).mul_add(*ui, *vi);
            }
        }
    }
    let after = norm(&v);
    (after > 1e-10 * before).then(|| v.iter().map(|vi| vi / after).collect())
}

/// An orthonormal basis of the vectors square to every one of `normals`
/// (`n` long), by Gram-Schmidt.
pub(crate) fn null_space(normals: &[&[f64]], n: usize) -> Vec<Vec<f64>> {
    let mut all: Vec<Vec<f64>> = Vec::new();
    for normal in normals {
        if let Some(u) = orthogonal(normal.to_vec(), &all) {
            all.push(u);
        }
    }
    let rank = all.len();
    for i in 0..n {
        let mut e = vec![0.0; n];
        e[i] = 1.0;
        if let Some(u) = orthogonal(e, &all) {
            all.push(u);
        }
    }
    all.split_off(rank)
}

/// `R Z` (column-major, `n` by `z.len()`), with `r` row-major upper
/// triangular.
fn times_r(r: &[f64], z: &[Vec<f64>], n: usize) -> Vec<f64> {
    z.iter()
        .flat_map(|zj| (0..n).map(move |i| (i..n).map(|k| r[i * n + k] * zj[k]).sum::<f64>()))
        .collect()
}

/// The damped step within the span of `z`: `Z y`, `y` minimising
/// `|R Z y + c|^2 + mu |D Z y|^2`, by Householder QR of the stacked rows.
fn face_step(r: &[f64], c: &[f64], d: &[f64], mu: f64, z: &[Vec<f64>]) -> Vec<f64> {
    let n = c.len();
    let rz = times_r(r, z, n);
    let root = mu.sqrt();
    let mut a = vec![0.0; 2 * n * z.len()];
    for (j, zj) in z.iter().enumerate() {
        let column = &mut a[j * 2 * n..(j + 1) * 2 * n];
        column[..n].copy_from_slice(&rz[j * n..(j + 1) * n]);
        for i in 0..n {
            column[n + i] = root * d[i] * zj[i];
        }
    }
    let mut rhs = c.to_vec();
    rhs.resize(2 * n, 0.0);
    let (s, mut y) = householder(&mut a, 2 * n, z.len(), &rhs);
    back_substitute(&s, &mut y);
    let mut delta = vec![0.0; n];
    for (yj, zj) in y.iter().zip(z) {
        for (di, zi) in delta.iter_mut().zip(zj) {
            *di = (-yj).mul_add(*zi, *di);
        }
    }
    delta
}

/// What a full Gauss-Newton step within the span of `z` would gain, in
/// the same units as `|c|` for the free step: `|Q^T c|` for the QR of
/// `R Z`.
fn face_gain(r: &[f64], c: &[f64], z: &[Vec<f64>]) -> f64 {
    let n = c.len();
    let mut rz = times_r(r, z, n);
    norm(&householder(&mut rz, n, z.len(), c).1)
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
