//! Tukey's biweight location of a window's fluxes, to full precision.
//!
//! With the median `M` and `MAD = median |x_i - M|`, the biweight location
//! `T` solves `sum psi((x_i - T) / (c MAD)) = 0`, where `psi(u) = u (1 -
//! u^2)^2` for `|u| < 1` and 0 beyond: the root that wotan's iteration
//! `T <- T + sum w_i d_i / sum w_i` (`d_i = x_i - T`, `w_i = (1 - u_i^2)^2`)
//! reaches from the median. `MAD = 0` gives `M`.
//!
//! Here: the median and MAD exactly (as double-doubles, a double and its
//! remainder), wotan's iteration until its step stops shrinking, Newton's
//! method on the residual `sum w_i d_i` evaluated in double-double, and last
//! the two doubles either side of the residual's sign change, the nearer
//! taken: the root correctly rounded wherever the residual's sign is
//! certain. Each window is first scaled by a power of two (exact) so that
//! its largest magnitude is between 1 and 2: no deviation overflows and none
//! loses bits to the subnormals.

use crate::location::{middle, middle_by, midpoint};

/// Steps of wotan's iteration at most, before Newton's method takes over.
const ITERATIONS: usize = 1000;

/// Newton steps at most; from where the iteration stalls, two or three do.
const NEWTON_STEPS: usize = 32;

/// Steps of one double at most when looking for the sign change.
const SEARCH_STEPS: usize = 64;

/// A double-double: `hi + lo`, `|lo|` at most half an ulp of `hi`.
type Double = (f64, f64);

/// The biweight location of `values` (finite, at least one) with tuning
/// constant `c` (positive, in MADs), reordering and scaling `values`;
/// `deviations` is scratch.
pub(crate) fn biweight(values: &mut [f64], deviations: &mut Vec<Double>, c: f64) -> f64 {
    debug_assert!(!values.is_empty() && c > 0.0);
    let (a, b) = middle(values);
    let largest = values.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if largest == 0.0 {
        return midpoint(a, b);
    }
    let e = exponent(largest);
    for x in values.iter_mut() {
        *x = scale(*x, -e);
    }
    // The exact median: halving the sum's two parts is exact.
    let (sum, error) = two_sum(scale(a, -e), scale(b, -e));
    let median = (sum / 2.0, error / 2.0);
    // The deviations, ordered exactly by (hi, lo) as rounding is monotone.
    deviations.clear();
    deviations.extend(values.iter().map(|&x| {
        let (d, error) = two_sum(x, -median.0);
        let d = two_sum(d, error - median.1);
        if d.0 < 0.0 { (-d.0, -d.1) } else { d }
    }));
    let (low, high) = middle_by(deviations, |p, q| {
        p.0.total_cmp(&q.0).then(p.1.total_cmp(&q.1))
    });
    let mad = half(add(low, high));
    if mad.0 == 0.0 {
        return midpoint(a, b);
    }
    let s = {
        let (hi, lo) = two_product(c, mad.0);
        two_sum(hi, lo + c * mad.1)
    };
    let values = &*values;

    // wotan's iteration, until its step stops shrinking.
    let mut t = median.0;
    let mut last = f64::INFINITY;
    for _ in 0..ITERATIONS {
        let here = sums(values, t, s.0);
        if here.weight == 0.0 {
            break;
        }
        let step = here.residual / here.weight;
        if step.abs() >= last {
            break;
        }
        last = step.abs();
        t += step;
    }

    // Newton's method on the accurate residual, while it shrinks.
    let mut at_t = residual(values, t, s);
    for _ in 0..NEWTON_STEPS {
        let slope = sums(values, t, s.0).slope;
        if slope <= 0.0 || at_t == 0.0 {
            break;
        }
        let next = t + at_t / slope;
        let at_next = residual(values, next, s);
        if next == t || at_next.abs() >= at_t.abs() {
            break;
        }
        (t, at_t) = (next, at_next);
    }

    // The nearer double either side of the sign change, among those exact
    // once scaled back (multiples of `fine`).
    let fine = power_of_two((-1074 - e).max(-1074));
    let t = nearest_root(t, fine, &|t| residual(values, t, s));
    scale(t, e)
}

/// The residual `sum w_i d_i` at `t` for `c MAD = s`, each term and the sum
/// in double-double: wrong only in the last bits of a double-double.
fn residual(values: &[f64], t: f64, s: Double) -> f64 {
    let (mut hi, mut lo) = (0.0, 0.0);
    for &x in values {
        let d = two_sum(x, -t);
        // u = d / s: the quotient and its correction.
        let u_hi = d.0 / s.0;
        let (p, p_error) = two_product(u_hi, s.0);
        let u_lo = (((d.0 - p) - p_error) + d.1 - u_hi * s.1) / s.0;
        if u_hi.abs() >= 1.0 {
            continue;
        }
        let u = (u_hi, u_lo);
        let v = add((1.0, 0.0), negate(multiply(u, u)));
        let term = multiply(multiply(v, v), d);
        let (sum, error) = two_sum(hi, term.0);
        hi = sum;
        lo += error + term.1;
    }
    hi + lo
}

fn negate(a: Double) -> Double {
    (-a.0, -a.1)
}

fn add(a: Double, b: Double) -> Double {
    let (sum, error) = two_sum(a.0, b.0);
    two_sum(sum, error + a.1 + b.1)
}

fn multiply(a: Double, b: Double) -> Double {
    let (product, error) = two_product(a.0, b.0);
    two_sum(product, error + a.0 * b.1 + a.1 * b.0)
}

/// Half of `a`: exact but where `lo` is subnormal.
fn half(a: Double) -> Double {
    (a.0 / 2.0, a.1 / 2.0)
}

/// The sums at `t`: the residual `sum w_i d_i` (compensated), the weight
/// `sum w_i` and the slope `sum (1 - u^2)(1 - 5 u^2)` (minus the residual's
/// derivative in `t`).
struct Sums {
    residual: f64,
    weight: f64,
    slope: f64,
}

fn sums(values: &[f64], t: f64, s: f64) -> Sums {
    let (mut residual, mut compensation) = (0.0, 0.0);
    let (mut weight, mut slope) = (0.0, 0.0);
    for &x in values {
        let d = x - t;
        let u = d / s;
        if u.abs() < 1.0 {
            let v = 1.0 - u * u;
            let w = v * v;
            let (sum, error) = two_sum(residual, w * d);
            (residual, compensation) = (sum, compensation + error);
            weight += w;
            slope += v * (1.0 - 5.0 * u * u);
        }
    }
    Sums {
        residual: residual + compensation,
        weight,
        slope,
    }
}

/// From `t` near a root of the decreasing `residual`, the doubles (multiples
/// of `fine`, a power of two) either side of its sign change, and of those
/// the nearer by linear interpolation, ties to even. `t` (snapped to the
/// multiples of `fine`) if there is no sign change within reach.
pub(crate) fn nearest_root(t: f64, fine: f64, residual: &dyn Fn(f64) -> f64) -> f64 {
    // Off the multiples only where the doubles are finer than `fine`, so
    // `t / fine` is small there.
    let t = if on_doubles(t, fine) {
        t
    } else {
        (t / fine).round() * fine
    };
    let at_t = residual(t);
    if at_t == 0.0 {
        return t;
    }
    // Towards the root: up while the residual is positive.
    let up = at_t > 0.0;
    let (mut near, mut at_near) = (t, at_t);
    for _ in 0..SEARCH_STEPS {
        let far = step(near, up, fine);
        let at_far = residual(far);
        if at_far == 0.0 {
            return far;
        }
        if (at_far > 0.0) != up {
            // The root at `near + fraction (far - near)`.
            let fraction = at_near / (at_near - at_far);
            let near_is_even = if on_doubles(near, fine) {
                near.to_bits() & 1 == 0
            } else {
                (near / fine).rem_euclid(2.0) == 0.0
            };
            return if fraction < 0.5 || (fraction == 0.5 && near_is_even) {
                near
            } else {
                far
            };
        }
        (near, at_near) = (far, at_far);
    }
    t
}

/// The next double from `t`, up or down.
fn next_double(t: f64, up: bool) -> f64 {
    if t == 0.0 {
        let tiny = f64::from_bits(1);
        if up { tiny } else { -tiny }
    } else if (t > 0.0) == up {
        f64::from_bits(t.to_bits() + 1)
    } else {
        f64::from_bits(t.to_bits() - 1)
    }
}

/// Whether the doubles next to `t` are at least `fine` apart, so that each
/// is a multiple of `fine`.
fn on_doubles(t: f64, fine: f64) -> bool {
    (next_double(t, true) - t).abs() >= fine && (next_double(t, false) - t).abs() >= fine
}

/// The next multiple of `fine` among the doubles from `t` (one), up or down.
fn step(t: f64, up: bool, fine: f64) -> f64 {
    let next = next_double(t, up);
    if (next - t).abs() >= fine {
        next
    } else if up {
        t + fine
    } else {
        t - fine
    }
}

/// `a b` and its rounding error, exactly (a fused multiply-add).
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let product = a * b;
    (product, a.mul_add(b, -product))
}

/// `a + b` and its rounding error, exactly (Knuth's two-sum).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    (sum, (a - a_virtual) + (b - b_virtual))
}

/// `floor(log2 x)` for finite positive `x`.
fn exponent(x: f64) -> i32 {
    let bits = x.to_bits();
    let biased = (bits >> 52) as i32;
    if biased == 0 {
        -1074 + 63 - bits.leading_zeros() as i32
    } else {
        biased - 1023
    }
}

/// 2^`power`, for `power` from -1074 to 1023.
fn power_of_two(power: i32) -> f64 {
    debug_assert!((-1074..=1023).contains(&power));
    if power >= -1022 {
        f64::from_bits(((power + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (power + 1074))
    }
}

/// `x` 2^`power`, in steps that stay within the doubles' exponents.
fn scale(mut x: f64, mut power: i32) -> f64 {
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
