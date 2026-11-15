//! Complete elliptic integrals, from Carlson's symmetric forms.
//!
//! The symmetric integrals use the duplication algorithm of Carlson (1995,
//! Numerical Algorithms 10, 13; DLMF 19.36). Each step replaces the arguments
//! with ones closer to their mean, without changing the integral beyond a known
//! factor; once they are close enough, a short Taylor series finishes it.
//!
//! The Legendre integrals use the parameter convention (`m`, not the modulus
//! `sqrt(m)`), and the third kind is
//! `Pi(n | m) = integral from 0 to pi/2 of dt / ((1 - n sin^2 t) sqrt(1 - m sin^2 t))`.

// Target relative error of the truncated series.
const TOLERANCE: f64 = 1e-16;

/// Carlson's `R_F(x, y, z)`: symmetric in all three arguments.
///
/// Arguments must be non-negative, with at most one of them zero.
pub(crate) fn carlson_rf(x: f64, y: f64, z: f64) -> f64 {
    let (x0, y0) = (x, y);
    let (mut x, mut y, mut z) = (x, y, z);

    let a0 = (x + y + z) / 3.0;
    let q =
        (3.0 * TOLERANCE).powf(-1.0 / 6.0) * (a0 - x).abs().max((a0 - y).abs()).max((a0 - z).abs());
    let mut a = a0;
    // 4^-n after n steps; dividing by 4 is exact in binary.
    let mut scale = 1.0;

    while scale * q > a.abs() {
        let lambda = duplication_lambda(x, y, z);
        x = (x + lambda) / 4.0;
        y = (y + lambda) / 4.0;
        z = (z + lambda) / 4.0;
        a = (a + lambda) / 4.0;
        scale /= 4.0;
    }

    let big_x = (a0 - x0) * scale / a;
    let big_y = (a0 - y0) * scale / a;
    let big_z = -big_x - big_y;
    let e2 = big_x * big_y - big_z * big_z;
    let e3 = big_x * big_y * big_z;

    (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0 - 3.0 * e2 * e3 / 44.0) / a.sqrt()
}

/// Carlson's `R_D(x, y, z)`: symmetric in `x` and `y` only.
///
/// Arguments must be non-negative, `z` positive, and at most one of `x` and
/// `y` zero.
pub(crate) fn carlson_rd(x: f64, y: f64, z: f64) -> f64 {
    let (x0, y0) = (x, y);
    let (mut x, mut y, mut z) = (x, y, z);

    let a0 = (x + y + 3.0 * z) / 5.0;
    let q =
        (TOLERANCE / 4.0).powf(-1.0 / 6.0) * (a0 - x).abs().max((a0 - y).abs()).max((a0 - z).abs());
    let mut a = a0;
    let mut scale = 1.0;
    let mut sum = 0.0;

    while scale * q > a.abs() {
        let lambda = duplication_lambda(x, y, z);
        sum += scale / (z.sqrt() * (z + lambda));
        x = (x + lambda) / 4.0;
        y = (y + lambda) / 4.0;
        z = (z + lambda) / 4.0;
        a = (a + lambda) / 4.0;
        scale /= 4.0;
    }

    let big_x = (a0 - x0) * scale / a;
    let big_y = (a0 - y0) * scale / a;
    let big_z = -(big_x + big_y) / 3.0;
    let xy = big_x * big_y;
    let z2 = big_z * big_z;
    let e2 = xy - 6.0 * z2;
    let e3 = (3.0 * xy - 8.0 * z2) * big_z;
    let e4 = 3.0 * (xy - z2) * z2;
    let e5 = xy * z2 * big_z;

    scale * a.powf(-1.5) * series_d_j(e2, e3, e4, e5) + 3.0 * sum
}

/// Carlson's `R_J(x, y, z, p)`: symmetric in `x`, `y` and `z`.
///
/// Arguments must be non-negative, `p` positive, and at most one of `x`, `y`
/// and `z` zero.
pub(crate) fn carlson_rj(x: f64, y: f64, z: f64, p: f64) -> f64 {
    let (x0, y0, z0) = (x, y, z);
    let (mut x, mut y, mut z, mut p) = (x, y, z, p);

    let a0 = (x + y + z + 2.0 * p) / 5.0;
    let delta = (p - x) * (p - y) * (p - z);
    let q = (TOLERANCE / 4.0).powf(-1.0 / 6.0)
        * (a0 - x)
            .abs()
            .max((a0 - y).abs())
            .max((a0 - z).abs())
            .max((a0 - p).abs());
    let mut a = a0;
    let mut scale = 1.0;
    let mut sum = 0.0;

    while scale * q > a.abs() {
        let lambda = duplication_lambda(x, y, z);
        let sqrt_p = p.sqrt();
        let d = (sqrt_p + x.sqrt()) * (sqrt_p + y.sqrt()) * (sqrt_p + z.sqrt());
        let e = scale * scale * scale * delta / (d * d);
        sum += scale / d * carlson_rc_one(e);
        x = (x + lambda) / 4.0;
        y = (y + lambda) / 4.0;
        z = (z + lambda) / 4.0;
        p = (p + lambda) / 4.0;
        a = (a + lambda) / 4.0;
        scale /= 4.0;
    }

    let big_x = (a0 - x0) * scale / a;
    let big_y = (a0 - y0) * scale / a;
    let big_z = (a0 - z0) * scale / a;
    let big_p = -(big_x + big_y + big_z) / 2.0;
    let xyz = big_x * big_y * big_z;
    let p2 = big_p * big_p;
    let e2 = big_x * big_y + big_x * big_z + big_y * big_z - 3.0 * p2;
    let e3 = xyz + 2.0 * e2 * big_p + 4.0 * p2 * big_p;
    let e4 = (2.0 * xyz + e2 * big_p + 3.0 * p2 * big_p) * big_p;
    let e5 = xyz * p2;

    scale * a.powf(-1.5) * series_d_j(e2, e3, e4, e5) + 6.0 * sum
}

/// Complete elliptic integral of the first kind, `K(m)`, from `mc = 1 - m`.
pub(crate) fn ellip_k(mc: f64) -> f64 {
    carlson_rf(0.0, mc, 1.0)
}

/// Complete elliptic integral of the second kind, `E(m)`.
///
/// Takes `m` and `mc = 1 - m` separately, so callers can pass an `mc` worked
/// out without the cancellation in `1.0 - m`. `k_m` is `K(m)` from
/// [`ellip_k`], passed in so that it is computed once when `E` and `Pi` are
/// both needed.
pub(crate) fn ellip_e(k_m: f64, m: f64, mc: f64) -> f64 {
    k_m - m / 3.0 * carlson_rd(0.0, mc, 1.0)
}

/// Complete elliptic integral of the third kind, `Pi(n | m)`, for `n < 1`,
/// from `mc = 1 - m`.
///
/// For large negative `n` the result is small and comes from cancelling
/// `K(m)` against a term of nearly the same size, so its error is about
/// `1e-16 * K(m)` absolute rather than relative. `k_m` is `K(m)`, as for
/// [`ellip_e`].
pub(crate) fn ellip_pi(k_m: f64, n: f64, mc: f64) -> f64 {
    k_m + n / 3.0 * carlson_rj(0.0, mc, 1.0, 1.0 - n)
}

// The `lambda` of one duplication step.
fn duplication_lambda(x: f64, y: f64, z: f64) -> f64 {
    let (sx, sy, sz) = (x.sqrt(), y.sqrt(), z.sqrt());
    sx * sy + sx * sz + sy * sz
}

// The Taylor series that finishes `R_D` and `R_J`; both use the same one.
fn series_d_j(e2: f64, e3: f64, e4: f64, e5: f64) -> f64 {
    1.0 - 3.0 * e2 / 14.0 + e3 / 6.0 + 9.0 * e2 * e2 / 88.0 - 3.0 * e4 / 22.0 - 9.0 * e2 * e3 / 52.0
        + 3.0 * e5 / 26.0
}

// Carlson's `R_C(1, 1 + e)` for `e > -1`, in closed form: `atan(sqrt(e)) /
// sqrt(e)` for `e > 0`, and `atanh(sqrt(-e)) / sqrt(-e)` for `e < 0`. Both
// tend to 1 as `e` goes to 0 and keep full relative precision for small `e`,
// so only `e = 0` needs its own case. `e < 0` comes from `Pi(n | m)` with
// `n > 0`, which the transit formulas never use.
fn carlson_rc_one(e: f64) -> f64 {
    if e == 0.0 {
        return 1.0;
    }

    let root = e.abs().sqrt();
    if e > 0.0 {
        root.atan() / root
    } else {
        root.atanh() / root
    }
}
