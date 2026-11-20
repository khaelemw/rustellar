//! The lens where a planet of radius `p` at distance `z` overlaps the star (of
//! radius 1), when the two edges cross: `|1 - p| < z < 1 + p`.
//!
//! The covered part of the star is bounded by an arc of the star's edge,
//! subtending `2 kappa1` at the star's centre, and an arc of the planet's
//! edge, subtending `2 kappa0` at the planet's centre. By Green's theorem, any
//! integral of a function of the distance `r` from the star's centre over the
//! lens is an integral along those two arcs. Along the planet's arc, with `phi`
//! the angle at the planet's centre measured from the star's direction and
//! `d = z - p`,
//!
//! ```text
//! r^2 = d^2 + 4 z p sin^2(phi / 2)
//! ```
//!
//! is a sum of two terms that are never negative, so the integrals keep full
//! precision even for a planet much larger than the star, where the textbook
//! formulas cancel terms of size `p^2` (area) or `p^4` (the `r^2` integral).
//! The arc integrals of `sin^2(phi / 2)` and its square are [`arc_1`] and
//! [`arc_2`].

/// The two half-angles of the lens, in `[0, pi]`.
#[derive(Clone, Copy)]
pub(crate) struct Lens {
    /// Half the angle the planet's arc subtends at the planet's centre.
    pub kappa0: f64,
    /// Half the angle the star's arc subtends at the star's centre.
    pub kappa1: f64,
}

impl Lens {
    /// The lens for `|1 - p| < z < 1 + p`.
    ///
    /// The half-angles come from `atan2` of their sines and cosines, scaled by
    /// `2 p z` and `2 z`:
    ///
    /// ```text
    /// s  = sqrt((1 + p - z)(1 + p + z)(z - 1 + p)(z + 1 - p))
    /// kappa0 = atan2(s, p^2 + z^2 - 1)      kappa1 = atan2(s, 1 - p^2 + z^2)
    /// ```
    ///
    /// Each factor of `s` is an exact sum of the inputs ([`sum3`]), and `s` is
    /// a product of square roots so that it cannot underflow when `z` is near
    /// the smallest double. The cosine parts are written around `p - 1` or
    /// `z - 1`, which are exact when `p` or `z` is near 1, so they keep their
    /// absolute precision where the lens is tiny (`z` near 0 with `p` near 1).
    pub(crate) fn new(z: f64, p: f64) -> Self {
        let s = (sum3(1.0, p, -z) * sum3(1.0, p, z)).max(0.0).sqrt()
            * sum3(z, -1.0, p).max(0.0).sqrt()
            * sum3(z, 1.0, -p).max(0.0).sqrt();

        // (p^2 + z^2 - 1, 1 - p^2 + z^2), from an exact difference when one of
        // p or z is within a factor of 2 of 1; otherwise neither is small.
        let (cos0, cos1) = if (0.5..=2.0).contains(&p) {
            let p2_minus_1 = (p - 1.0) * (p + 1.0);
            (p2_minus_1 + z * z, z * z - p2_minus_1)
        } else if (0.5..=2.0).contains(&z) {
            let z2_minus_1 = (z - 1.0) * (z + 1.0);
            (z2_minus_1 + p * p, 2.0 + z2_minus_1 - p * p)
        } else {
            (p * p + z * z - 1.0, 1.0 - p * p + z * z)
        };

        Self {
            kappa0: s.atan2(cos0),
            kappa1: s.atan2(cos1),
        }
    }
}

/// `a + b + c` with a single rounding (to within a few units in the last
/// place), however much the terms cancel: Knuth's two-sum on each addition,
/// with the two rounding errors added back at the end.
pub(crate) fn sum3(a: f64, b: f64, c: f64) -> f64 {
    let (s, e) = two_sum(a, b);
    let (t, f) = two_sum(s, c);
    t + (e + f)
}

// The rounded sum and its exact rounding error (Knuth 1969).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let b_virtual = s - a;
    let a_virtual = s - b_virtual;
    (s, (a - a_virtual) + (b - b_virtual))
}

// Below this angle the arc integrals use their Taylor series, which have no
// cancellation; above it the closed forms lose at most a factor of 6 (arc_1)
// or 40 (arc_2) to cancellation, so a few units in the last place.
const SERIES_BELOW: f64 = 1.0;

/// `integral from -k to k of sin^2(phi / 2) dphi = k - sin k`.
///
/// For small `k` this is about `k^3 / 6`, so the closed form would cancel;
/// below 1 the Taylor series is summed instead.
pub(crate) fn arc_1(k: f64) -> f64 {
    if k >= SERIES_BELOW {
        return k - k.sin();
    }

    // k^3/3! - k^5/5! + k^7/7! - ...
    let k2 = k * k;
    let mut term = k * k2 / 6.0;
    let mut total = 0.0_f64;
    let mut n = 1.0;
    while term.abs() > f64::EPSILON * 1e-3 * total.abs() {
        total += term;
        term *= -k2 / ((2.0 * n + 2.0) * (2.0 * n + 3.0));
        n += 1.0;
    }
    total
}

/// `integral from -k to k of sin^4(phi / 2) dphi
///  = (3 k - 4 sin k + sin(2 k) / 2) / 4`.
///
/// For small `k` this is about `k^5 / 40`, so below 1 the Taylor series is
/// summed: `sum over n >= 2 of (-1)^n (4^n - 4) k^(2n+1) / (4 (2n+1)!)`.
pub(crate) fn arc_2(k: f64) -> f64 {
    if k >= SERIES_BELOW {
        return arc_2_closed(k, k.sin());
    }
    arc_2_series(k)
}

/// Both arcs at `k`, with one `sin k` between them where they take it.
pub(crate) fn arcs(k: f64) -> (f64, f64) {
    if k >= SERIES_BELOW {
        let sin_k = k.sin();
        return (k - sin_k, arc_2_closed(k, sin_k));
    }
    (arc_1(k), arc_2_series(k))
}

/// [`arc_2`]'s closed form, given `sin k`.
fn arc_2_closed(k: f64, sin_k: f64) -> f64 {
    (3.0 * k - 4.0 * sin_k + 0.5 * (2.0 * k).sin()) / 4.0
}

/// [`arc_2`]'s series, for `k` below 1.
fn arc_2_series(k: f64) -> f64 {
    // Terms (-1)^n k^(2n+1) / (2n+1)! and their weights 4^n - 4, from n = 2.
    let k2 = k * k;
    let mut power = k * k2 * k2 / 120.0;
    let mut four_n = 16.0;
    let mut total = 0.0_f64;
    let mut n = 2.0;
    loop {
        let term = power * (four_n - 4.0);
        total += term;
        if term.abs() <= f64::EPSILON * 1e-3 * total.abs() {
            break;
        }
        power *= -k2 / ((2.0 * n + 2.0) * (2.0 * n + 3.0));
        four_n *= 4.0;
        n += 1.0;
    }
    total / 4.0
}
