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

use crate::location::midpoint;

/// Steps of wotan's iteration at most, before Newton's method takes over.
const ITERATIONS: usize = 1000;

/// wotan's iteration hands over to Newton's once its step is below this
/// part of c MAD.
const HANDOVER: f64 = 1.0 / 1024.0;

/// Newton steps at most; from where the iteration stalls, two or three do.
const NEWTON_STEPS: usize = 32;

/// Steps of one double at most when looking for the sign change.
const SEARCH_STEPS: usize = 64;

/// c MAD is scaled to at least 2^FLOOR.
const FLOOR: i32 = -500;

/// A double-double: `hi + lo`, `|lo|` at most half an ulp of `hi`.
type Double = (f64, f64);

/// Scratch space for [`biweight_sorted`], kept between windows so that no
/// window allocates.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    scaled: Vec<f64>,
}

/// The biweight location of `values` (finite, at least one) with tuning
/// constant `c` (positive, in MADs), sorting `values`.
#[cfg(test)]
pub(crate) fn biweight(values: &mut [f64], scratch: &mut Scratch, c: f64) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    biweight_sorted(values, scratch, c)
}

/// The biweight location of `sorted` (finite, at least one, ascending by
/// `total_cmp`) with tuning constant `c` (in MADs, from 2^-100 to 2^100).
///
/// Every sum runs over the values in this order, so the result depends on
/// the values alone, not on the order they came in.
pub(crate) fn biweight_sorted(sorted: &[f64], scratch: &mut Scratch, c: f64) -> f64 {
    debug_assert!(!sorted.is_empty() && (power_of_two(-100)..=power_of_two(100)).contains(&c));
    debug_assert!(sorted.is_sorted_by(|a, b| a.total_cmp(b).is_le()));
    let n = sorted.len();
    let (a, b) = (sorted[(n - 1) / 2], sorted[n / 2]);
    let largest = sorted[0].abs().max(sorted[n - 1].abs());
    if largest == 0.0 {
        return midpoint(a, b);
    }
    let mut e = exponent(largest);
    let (median, s) = loop {
        scratch.scaled.clear();
        scratch.scaled.extend(sorted.iter().map(|&x| scale(x, -e)));
        // The exact median: halving the sum's two parts is exact.
        let (sum, error) = two_sum(scale(a, -e), scale(b, -e));
        let median = (sum / 2.0, error / 2.0);
        let mad = mad(&scratch.scaled, median);
        if mad.0 == 0.0 {
            return midpoint(a, b);
        }
        let (hi, lo) = two_product(c, mad.0);
        let s = two_sum(hi, lo + c * mad.1);
        // c MAD far below the largest magnitude (a tight cluster beside
        // distant values): scaled up, so that 1 / (c MAD) and c MAD's low
        // part stay normal. A nonzero MAD is at least 2^-1075 of the largest
        // magnitude, so for c from 2^-100 to 2^100 c MAD is at least
        // 2^-1175 of it and the values grow to 2^677 at most.
        if s.0 >= power_of_two(FLOOR) {
            break (median, s);
        }
        e += exponent(s.0) - FLOOR;
    };
    let values = &scratch.scaled[..];
    let inverse = 1.0 / s.0;

    // wotan's iteration from the median, while its step shrinks, until the
    // step is a small part of c MAD: well inside the root's basin.
    let mut t = median.0;
    let mut last = f64::INFINITY;
    for _ in 0..ITERATIONS {
        let here = steer(values, t, inverse);
        if here.weight == 0.0 {
            break;
        }
        let step = here.residual / here.weight;
        if step.abs() >= last {
            break;
        }
        last = step.abs();
        t += step;
        if last <= HANDOVER * s.0 {
            break;
        }
    }

    // Newton's method, while the residual shrinks.
    let mut here = steer(values, t, inverse);
    for _ in 0..NEWTON_STEPS {
        if here.slope <= 0.0 || here.residual == 0.0 {
            break;
        }
        // Within two doubles the search below takes over.
        let next = t + here.residual / here.slope;
        if (next - t).abs() <= 2.0 * gap(t) {
            break;
        }
        let there = steer(values, next, inverse);
        if there.residual.abs() >= here.residual.abs() {
            break;
        }
        (t, here) = (next, there);
    }

    let precise = |t: f64| residual(values, t, s, inverse);
    let cheap = |t: f64| {
        let (value, bound) = bounded(values, t, inverse);
        Estimate { value, bound }
    };
    let mut at_t = cheap(t);
    if !at_t.certain() {
        // The sums cannot tell the residual from zero here: Newton's method
        // on the double-double residual (the slope as last found: near the
        // root it hardly changes), while it shrinks.
        at_t = Estimate::exact(precise(t));
        for _ in 0..NEWTON_STEPS {
            if here.slope <= 0.0 || at_t.value == 0.0 {
                break;
            }
            let next = t + at_t.value / here.slope;
            if (next - t).abs() <= 2.0 * gap(t) {
                break;
            }
            let at_next = precise(next);
            if at_next.abs() >= at_t.value.abs() {
                break;
            }
            (t, at_t) = (next, Estimate::exact(at_next));
        }
    }

    // The nearer double either side of the sign change, among those exact
    // once scaled back (multiples of `fine`).
    let fine = power_of_two((-1074 - e).max(-1074));
    let t = nearest_root(t, at_t, fine, &cheap, &precise);
    scale(t, e)
}

/// `x - median`, exactly as a double-double (to the last bits of one).
fn deviation(x: f64, median: Double) -> Double {
    let (d, error) = two_sum(x, -median.0);
    two_sum(d, error - median.1)
}

/// The median absolute deviation of `sorted` from `median`, as a
/// double-double: the deviations are taken outward from the median, the
/// smaller first, up to the middle one (or two).
fn mad(sorted: &[f64], median: Double) -> Double {
    let n = sorted.len();
    // Below `below` (and from `above`) the values not yet taken; at first
    // those below the median and the rest.
    let split = sorted.partition_point(|&x| deviation(x, median).0 < 0.0);
    let (mut below, mut above) = (split, split);
    let size = |x: f64| {
        let d = deviation(x, median);
        if d.0 < 0.0 { negate(d) } else { d }
    };
    // The next deviation either side, once each.
    let mut left = (below > 0).then(|| size(sorted[below - 1]));
    let mut right = (above < n).then(|| size(sorted[above]));
    let mut next = || {
        // The smaller of the two, exactly ordered.
        let take_left = match (left, right) {
            (Some(l), Some(r)) => l.0.total_cmp(&r.0).then(l.1.total_cmp(&r.1)).is_le(),
            (l, _) => l.is_some(),
        };
        let taken = if take_left {
            below -= 1;
            std::mem::replace(&mut left, (below > 0).then(|| size(sorted[below - 1])))
        } else {
            above += 1;
            std::mem::replace(&mut right, (above < n).then(|| size(sorted[above])))
        };
        // n / 2 + 1 of n deviations are taken: there is always one.
        taken.unwrap_or_default()
    };
    for _ in 0..(n - 1) / 2 {
        next();
    }
    let low = next();
    let high = if n % 2 == 1 { low } else { next() };
    half(add(low, high))
}

/// The values with `|u| = |x - t| / s` below 1, `u` computed as the sums do
/// (`(x - t) inverse`, `inverse = 1 / s`): a run of the sorted values, as `u`
/// never decreases with `x`.
fn inside(sorted: &[f64], t: f64, inverse: f64) -> &[f64] {
    let start = sorted.partition_point(|&x| (x - t) * inverse <= -1.0);
    let end = sorted.partition_point(|&x| (x - t) * inverse < 1.0);
    &sorted[start..end]
}

/// Independent sums, so that the passes over the values can run in
/// parallel lanes; the lanes are added in a fixed order.
const LANES: usize = 4;

/// The residual `sum w_i d_i` at `t` for `c MAD = s` (`inverse` its
/// reciprocal), each term and the sum in double-double: wrong only in the
/// last bits of a double-double.
fn residual(values: &[f64], t: f64, s: Double, inverse: f64) -> f64 {
    let s_halves = split(s.0);
    let term = |x: f64| {
        let d = two_sum(x, -t);
        // u = d / s: the quotient and its correction.
        let u_hi = d.0 * inverse;
        let (p, p_error) = two_product_halves(u_hi, s.0, s_halves);
        let u_lo = (((d.0 - p) - p_error) + d.1 - u_hi * s.1) * inverse;
        let u = (u_hi, u_lo);
        let v = add((1.0, 0.0), negate(multiply(u, u)));
        multiply(multiply(v, v), d)
    };
    let mut lanes = [(0.0, 0.0); LANES];
    let inside = inside(values, t, inverse);
    let mut chunks = inside.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (lane, &x) in lanes.iter_mut().zip(chunk) {
            *lane = accumulate(*lane, term(x));
        }
    }
    for (lane, &x) in lanes.iter_mut().zip(chunks.remainder()) {
        *lane = accumulate(*lane, term(x));
    }
    let total = add(add(lanes[0], lanes[1]), add(lanes[2], lanes[3]));
    total.0 + total.1
}

/// `sum + term`, the sum's low part gathering the errors.
fn accumulate(sum: Double, term: Double) -> Double {
    let (hi, error) = two_sum(sum.0, term.0);
    (hi, sum.1 + error + term.1)
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

/// The plain sums at `t` that steer the iteration: the residual
/// `sum w_i d_i`, the weight `sum w_i` and the slope
/// `sum (1 - u^2)(1 - 5 u^2)` (minus the residual's derivative in `t`).
#[derive(Clone, Copy, Default)]
struct Steer {
    residual: f64,
    weight: f64,
    slope: f64,
}

fn steer(values: &[f64], t: f64, inverse: f64) -> Steer {
    // Four lanes of each sum, side by side, so that they run as vectors.
    let (mut residual, mut weight, mut slope) = ([0.0; LANES], [0.0; LANES], [0.0; LANES]);
    let inside = inside(values, t, inverse);
    let mut chunks = inside.chunks_exact(LANES);
    for chunk in &mut chunks {
        for lane in 0..LANES {
            let d = chunk[lane] - t;
            let u = d * inverse;
            let v = 1.0 - u * u;
            let w = v * v;
            residual[lane] += w * d;
            weight[lane] += w;
            slope[lane] += v * (1.0 - 5.0 * u * u);
        }
    }
    for (lane, &x) in chunks.remainder().iter().enumerate() {
        let d = x - t;
        let u = d * inverse;
        let v = 1.0 - u * u;
        let w = v * v;
        residual[lane] += w * d;
        weight[lane] += w;
        slope[lane] += v * (1.0 - 5.0 * u * u);
    }
    let total = |a: [f64; LANES]| (a[0] + a[1]) + (a[2] + a[3]);
    Steer {
        residual: total(residual),
        weight: total(weight),
        slope: total(slope),
    }
}

/// The rounding errors of a term `w d` of the residual, in units of
/// `eps |d| (1 - u^2)`: `u` within 4 eps (the reciprocal, the product,
/// `d`'s low part and c MAD's), `1 - u^2` within 9 eps u^2 + eps (1 - u^2),
/// its square and the product with `d` 22 in all; doubled.
const TERM_ERROR: f64 = 44.0;

/// The residual `sum w_i d_i` at `t`, compensated and with `d_i` exact, and
/// a bound on its error.
fn bounded(values: &[f64], t: f64, inverse: f64) -> (f64, f64) {
    // Each lane: the residual's high part, the errors gathered, and
    // sum |d| (1 - u^2) for the bound.
    let term = |lane: &mut [f64; 3], x: f64| {
        let (d, d_low) = two_sum(x, -t);
        let u = d * inverse;
        let v = 1.0 - u * u;
        let w = v * v;
        let (sum, error) = two_sum(lane[0], w * d);
        lane[0] = sum;
        lane[1] += error + w * d_low;
        lane[2] += d.abs() * v;
    };
    let mut lanes = [[0.0; 3]; LANES];
    let inside = inside(values, t, inverse);
    let mut chunks = inside.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (lane, &x) in lanes.iter_mut().zip(chunk) {
            term(lane, x);
        }
    }
    for (lane, &x) in lanes.iter_mut().zip(chunks.remainder()) {
        term(lane, x);
    }
    let total = |k: usize| (lanes[0][k] + lanes[1][k]) + (lanes[2][k] + lanes[3][k]);
    let high = add(
        two_sum(lanes[0][0], lanes[1][0]),
        two_sum(lanes[2][0], lanes[3][0]),
    );
    let residual = high.0 + (high.1 + total(1));
    // The terms', the summation's (compensated: a few eps of the sum and
    // eps^2 of the terms, well inside the terms' share) and the last
    // rounding's.
    let bound = f64::EPSILON * (TERM_ERROR * total(2) + 2.0 * residual.abs());
    (residual, bound)
}

/// A residual and a bound on its error.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Estimate {
    pub(crate) value: f64,
    pub(crate) bound: f64,
}

impl Estimate {
    fn exact(value: f64) -> Self {
        Self { value, bound: 0.0 }
    }

    /// Whether the sign is certain.
    fn certain(self) -> bool {
        self.value.abs() > self.bound
    }
}

/// From `t` near a root of the decreasing `residual`, the doubles (multiples
/// of `fine`, a power of two) either side of its sign change, and of those
/// the nearer by linear interpolation, ties to even. `t` (snapped to the
/// multiples of `fine`) if there is no sign change within reach.
///
/// The residual comes from `cheap`, with a bound on its error, and from
/// `precise` (taken as exact) wherever that bound leaves a sign, or which
/// side is nearer, in doubt. `at_t` is the residual at `t`, known already.
pub(crate) fn nearest_root(
    t: f64,
    at_t: Estimate,
    fine: f64,
    cheap: &dyn Fn(f64) -> Estimate,
    precise: &dyn Fn(f64) -> f64,
) -> f64 {
    let settle = |t: f64, at: Estimate| {
        if at.certain() {
            at
        } else {
            Estimate::exact(precise(t))
        }
    };
    // Off the multiples only where the doubles are finer than `fine`, so
    // `t / fine` is small there.
    let (t, at_t) = if on_doubles(t, fine) {
        (t, at_t)
    } else {
        let snapped = (t / fine).round() * fine;
        (snapped, cheap(snapped))
    };
    let at_t = settle(t, at_t);
    if at_t.value == 0.0 {
        return t;
    }
    // Towards the root: up while the residual is positive.
    let up = at_t.value > 0.0;
    let (mut near, mut at_near) = (t, at_t);
    for _ in 0..SEARCH_STEPS {
        let far = step(near, up, fine);
        let at_far = settle(far, cheap(far));
        if at_far.value == 0.0 {
            return far;
        }
        if (at_far.value > 0.0) != up {
            // Which is nearer: certain if the sizes differ by more than the
            // bounds, else both precisely.
            let doubt = at_near.bound + at_far.bound;
            let (at_near, at_far) = if (at_near.value.abs() - at_far.value.abs()).abs() > doubt {
                (at_near.value, at_far.value)
            } else {
                (precise(near), precise(far))
            };
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

/// The larger gap from `t` to the doubles either side.
fn gap(t: f64) -> f64 {
    (next_double(t, true) - t)
        .abs()
        .max((next_double(t, false) - t).abs())
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

/// `a b` and its rounding error, exactly (Dekker's product: no fused
/// multiply-add, which most x86-64 builds would call in software). Exact for
/// the scaled values here, far from overflow and underflow.
fn two_product(a: f64, b: f64) -> (f64, f64) {
    two_product_halves(a, b, split(b))
}

/// [`two_product`] with `b` already split.
fn two_product_halves(a: f64, b: f64, (b_hi, b_lo): (f64, f64)) -> (f64, f64) {
    let product = a * b;
    let (a_hi, a_lo) = split(a);
    let error = ((a_hi * b_hi - product) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (product, error)
}

/// `x` as two halves of 26 bits each (Veltkamp's split).
fn split(x: f64) -> (f64, f64) {
    let t = 134_217_729.0 * x; // 2^27 + 1
    let hi = t - (t - x);
    (hi, x - hi)
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
