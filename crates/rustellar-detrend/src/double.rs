//! Double-double arithmetic (a double and its remainder) and powers of two,
//! with no fused multiply-add, so the same bits on every platform.

/// A double-double: `hi + lo`, `|lo|` at most half an ulp of `hi`.
pub(crate) type Double = (f64, f64);

/// `sum + term`, the sum's low part gathering the errors.
pub(crate) fn accumulate(sum: Double, term: Double) -> Double {
    let (hi, error) = two_sum(sum.0, term.0);
    (hi, sum.1 + error + term.1)
}

pub(crate) fn negate(a: Double) -> Double {
    (-a.0, -a.1)
}

pub(crate) fn add(a: Double, b: Double) -> Double {
    let (sum, error) = two_sum(a.0, b.0);
    two_sum(sum, error + a.1 + b.1)
}

pub(crate) fn multiply(a: Double, b: Double) -> Double {
    let (product, error) = two_product(a.0, b.0);
    two_sum(product, error + a.0 * b.1 + a.1 * b.0)
}

/// Half of `a`: exact but where `lo` is subnormal.
pub(crate) fn half(a: Double) -> Double {
    (a.0 / 2.0, a.1 / 2.0)
}

/// `a b` and its rounding error, exactly (Dekker's product: no fused
/// multiply-add, which most x86-64 builds would call in software). Exact for
/// the scaled values here, far from overflow and underflow.
pub(crate) fn two_product(a: f64, b: f64) -> (f64, f64) {
    two_product_halves(a, b, split(b))
}

/// [`two_product`] with `b` already split.
pub(crate) fn two_product_halves(a: f64, b: f64, (b_hi, b_lo): (f64, f64)) -> (f64, f64) {
    let product = a * b;
    let (a_hi, a_lo) = split(a);
    let error = ((a_hi * b_hi - product) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (product, error)
}

/// `x` as two halves of 26 bits each (Veltkamp's split).
pub(crate) fn split(x: f64) -> (f64, f64) {
    let t = 134_217_729.0 * x; // 2^27 + 1
    let hi = t - (t - x);
    (hi, x - hi)
}

/// `a + b` and its rounding error, exactly (Knuth's two-sum).
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    (sum, (a - a_virtual) + (b - b_virtual))
}

/// `floor(log2 x)` for finite positive `x`.
pub(crate) fn exponent(x: f64) -> i32 {
    let bits = x.to_bits();
    let biased = (bits >> 52) as i32;
    if biased == 0 {
        -1074 + 63 - bits.leading_zeros() as i32
    } else {
        biased - 1023
    }
}

/// 2^`power`, for `power` from -1074 to 1023.
pub(crate) fn power_of_two(power: i32) -> f64 {
    debug_assert!((-1074..=1023).contains(&power));
    if power >= -1022 {
        f64::from_bits(((power + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (power + 1074))
    }
}

/// `x` 2^`power`, in steps that stay within the doubles' exponents.
pub(crate) fn scale(mut x: f64, mut power: i32) -> f64 {
    while power > 1000 {
        x *= power_of_two(1000);
        power -= 1000;
    }
    while power < -1000 {
        x *= power_of_two(-1000);
        power += 1000;
    }
    x * power_of_two(power)
}
