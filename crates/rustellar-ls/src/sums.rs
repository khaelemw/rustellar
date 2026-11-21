//! Compensated sums and the exact phase of a frequency at a time.

use std::f64::consts::PI;

/// A running sum with its rounding error carried (Neumaier 1974): exact to
/// about one rounding whatever the order or the cancellation.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Sum {
    sum: f64,
    carry: f64,
}

impl Sum {
    /// Adds `x`.
    #[inline]
    pub(crate) fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.carry += (self.sum - t) + x;
        } else {
            self.carry += (x - t) + self.sum;
        }
        self.sum = t;
    }

    /// The sum.
    #[inline]
    pub(crate) fn value(self) -> f64 {
        self.sum + self.carry
    }

    /// This sum over `by` to about twice the digits of an `f64`: the
    /// quotient and its remainder's share, `(hi, lo)`.
    pub(crate) fn quotient(self, by: Self) -> (f64, f64) {
        let (n_hi, n_lo) = two_sum(self.sum, self.carry);
        let (d_hi, d_lo) = two_sum(by.sum, by.carry);
        let hi = n_hi / d_hi;
        // n - hi d, its leading part exact by the fused product.
        let r = (-hi).mul_add(d_hi, n_hi) + (n_lo - hi * d_lo);
        (hi, r / d_hi)
    }
}

/// `a + b` as the rounded sum and its exact error (Knuth).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// `cos(2 pi f t)` and `sin(2 pi f t)`, each as a whole base (-1, 0 or 1)
/// and a small part known to a few roundings of itself however small:
/// `((cos base, cos part), (sin base, sin part))`.
///
/// The phase `f t` in cycles comes from the exact product (one rounding at
/// the end), reduced to the nearest quarter cycle `q / 4`, so `r = f t - q
/// / 4` is in `[-1/8, 1/8]` and exact to that rounding. From `r`'s half
/// angle, `sin(2 pi r) = 2 sin(pi r) cos(pi r)` and `cos(2 pi r) - 1 = -2
/// sin^2(pi r)`, each exact to its own digits; the quarter turns only swap
/// and negate them. So a large time loses no phase, and a sine or cosine
/// near 0 or near 1 keeps its digits.
#[inline]
pub(crate) fn phase(f: f64, t: f64) -> ((f64, f64), (f64, f64)) {
    let hi = f * t;
    let lo = f.mul_add(t, -hi);
    let quarters = (4.0 * hi).round();
    // Exact: `hi` is within an eighth of `quarters / 4`.
    let r = (hi - 0.25 * quarters) + lo;
    let (sh, ch) = (PI * r).sin_cos();
    let (s, v) = (2.0 * sh * ch, -2.0 * sh * sh);
    // cos and sin of theta + q pi / 2.
    match quarters.rem_euclid(4.0) {
        0.0 => ((1.0, v), (0.0, s)),
        1.0 => ((0.0, -s), (1.0, v)),
        2.0 => ((-1.0, -v), (0.0, -s)),
        _ => ((0.0, s), (-1.0, -v)),
    }
}
