//! Compensated sums and the exact phase of a frequency at a time.

use std::f64::consts::TAU;

/// A running sum with its rounding error carried (Knuth's two-sum, as
/// Neumaier 1974 but without a branch): exact to about one rounding
/// whatever the order or the cancellation.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Sum {
    sum: f64,
    carry: f64,
}

impl Sum {
    /// Adds `x`.
    #[inline]
    pub(crate) fn add(&mut self, x: f64) {
        let (sum, error) = two_sum(self.sum, x);
        self.sum = sum;
        self.carry += error;
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
        // n - hi d, its leading part exact by the two-product.
        let (p, e) = two_product(hi, d_hi);
        let r = ((n_hi - p) - e) + (n_lo - hi * d_lo);
        (hi, r / d_hi)
    }
}

/// `a + b` as the rounded sum and its exact error (Knuth).
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// `a` as two halves of 26 bits that sum to it exactly (Veltkamp), so
/// their products are exact.
#[inline]
pub(crate) fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0 * a;
    let hi = c - (c - a);
    (hi, a - hi)
}

/// `a b` as the rounded product and its exact error (Dekker 1971), from
/// the halves of each: no fused multiply-add needed.
#[inline]
pub(crate) fn two_product_split(
    a: f64,
    (ah, al): (f64, f64),
    b: f64,
    (bh, bl): (f64, f64),
) -> (f64, f64) {
    let p = a * b;
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// `a b` as the rounded product and its exact error.
#[inline]
pub(crate) fn two_product(a: f64, b: f64) -> (f64, f64) {
    two_product_split(a, split(a), b, split(b))
}

/// 1.5 2^52: adding and subtracting it rounds to a whole number.
const ROUNDER: f64 = 6_755_399_441_055_744.0;

/// `cos(2 pi f t)` and `sin(2 pi f t)`, each as a whole base (-1, 0 or 1)
/// and a small part known to a few roundings of itself however small:
/// `((cos base, cos part), (sin base, sin part))`, from `f` and `t` and
/// their halves ([`split`]).
///
/// The phase `f t` in cycles comes from the exact product (one rounding at
/// the end), reduced to the nearest quarter cycle `q / 4`, so `r = f t - q
/// / 4` is in `[-1/8, 1/8]` and exact to that rounding. Then
/// `sin(2 pi r)` and `cos(2 pi r) - 1` come from fdlibm's minimax
/// polynomials for `|x| <= pi / 4` ([`kernel`]), each to a few roundings
/// of itself; the quarter turns only swap and negate them. So a large time
/// loses no phase, and a sine or cosine near 0 or near 1 keeps its digits.
#[inline]
pub(crate) fn phase(
    f: f64,
    f_halves: (f64, f64),
    t: f64,
    t_halves: (f64, f64),
) -> ((f64, f64), (f64, f64)) {
    let (hi, lo) = two_product_split(f, f_halves, t, t_halves);
    let four = 4.0 * hi;
    // The nearest whole number by the rounder below 2^52; at or above it
    // `four` is whole already. No call and no branch, so this vectorises.
    let rounded = (four + ROUNDER) - ROUNDER;
    let quarters = if four.abs() < 4_503_599_627_370_496.0 {
        rounded
    } else {
        four
    };
    // Exact: `hi` is within an eighth of `quarters / 4`.
    let r = (hi - 0.25 * quarters) + lo;
    let (s, v) = kernel(TAU * r);
    // The quadrant as q in -2..=2 (quarters less the nearest multiple of
    // 4, exact), and cos(q pi / 2), sin(q pi / 2) from exact polynomials
    // in it: no table and no branch.
    let q = quarters - 4.0 * ((0.25 * quarters + ROUNDER) - ROUNDER);
    let u = q * q;
    let cb = (6.0 - 7.0 * u + u * u) / 6.0;
    let sb = q * (4.0 - u) / 3.0;
    // cos and sin of theta + q pi / 2.
    ((cb, cb * v - sb * s), (sb, sb * v + cb * s))
}

/// `sin x` and `cos x - 1` for `|x| <= pi / 4`, each within about one
/// rounding of itself: the minimax polynomials of fdlibm's
/// `__kernel_sin` and `__kernel_cos` (Sun Microsystems 1993), the second
/// without its leading 1. The coefficients are fdlibm's, bit for bit
/// (its decimal forms round to them).
#[inline]
pub(crate) fn kernel(x: f64) -> (f64, f64) {
    const S1: f64 = f64::from_bits(0xbfc5_5555_5555_5549);
    const S2: f64 = f64::from_bits(0x3f81_1111_1110_f8a6);
    const S3: f64 = f64::from_bits(0xbf2a_01a0_19c1_61d5);
    const S4: f64 = f64::from_bits(0x3ec7_1de3_57b1_fe7d);
    const S5: f64 = f64::from_bits(0xbe5a_e5e6_8a2b_9ceb);
    const S6: f64 = f64::from_bits(0x3de5_d93a_5acf_d57c);
    const C1: f64 = f64::from_bits(0x3fa5_5555_5555_554c);
    const C2: f64 = f64::from_bits(0xbf56_c16c_16c1_5177);
    const C3: f64 = f64::from_bits(0x3efa_01a0_19cb_1590);
    const C4: f64 = f64::from_bits(0xbe92_7e4f_809c_52ad);
    const C5: f64 = f64::from_bits(0x3e21_ee9e_bdb4_b1c4);
    const C6: f64 = f64::from_bits(0xbda8_fae9_be88_38d4);
    let z = x * x;
    let sine = x + z * x * (S1 + z * (S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))));
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    (sine, z * r - 0.5 * z)
}
