//! Huber's proposal 2: the location and scale of a window estimated
//! together, as statsmodels' `Huber` (and so wotan's `method='huber'`)
//! does, but solved exactly instead of iterated to a tolerance.
//!
//! With `N` values, tuning constant `c`, and `gamma` the expected value of
//! `min(Z^2, c^2)` for a standard normal `Z` (so that `s` estimates the
//! standard deviation of normal data), the location `mu` and scale `s`
//! solve
//!
//! ```text
//! sum clip(x_i, mu - c s, mu + c s) = N mu
//! sum min((x_i - mu)^2, (c s)^2)   = (N - 1) gamma s^2
//! ```
//!
//! For the points split into those below `mu - c s`, inside, and above
//! `mu + c s` (contiguous in a sorted window), the first equation makes `mu`
//! linear in `s` and the second is then a quadratic in `s`. So: split,
//! solve, split again at the solution, until the split holds (in doubles);
//! then the final split's quadratic in double-double, and `mu` rounded to
//! the nearest double.
//!
//! `s = 0` solves both equations for any window (every point clipped to
//! `mu`), but is the solution only where no positive scale does (see
//! `solve`). The two
//! equations are where `sum s rho((x_i - mu) / s) + (N - 1) gamma s / 2`
//! is least (Huber 1981), a function convex in `mu` and `s` together, so
//! the scale equation's excess at each `s`, with the best `mu` for it,
//! falls as `s` grows. Where the splits do not settle, bisection on `s`
//! finds the split instead.

use crate::double::{
    Double, accumulate, add, exponent, multiply, negate, scale, two_product, two_sum,
};

/// Rounds of split and solve, at most, before bisection takes over; two or
/// three usually settle.
pub(crate) const ROUNDS: usize = 64;

/// `sqrt(2 / pi)` as a double-double.
const SQRT_2_OVER_PI: Double = (0.797_884_560_802_865_4, -4.984_654_404_555_46e-17);

/// Scratch space for [`huber_joint`], kept between windows.
#[derive(Debug, Default)]
pub(crate) struct JointScratch {
    scaled: Vec<f64>,
    deviations: Vec<f64>,
}

/// `gamma(c)`, the expected value of `min(Z^2, c^2)` for a standard normal
/// `Z`, for `c` from `1e-30` to `1e30`: to about 1e-30 for `c` up to 6
/// (a series), and about 1e-23 above (one less the normal tail, in doubles,
/// below 4e-9 there).
pub(crate) fn gamma(c: f64) -> Double {
    let c2 = two_product(c, c);
    if c <= 6.0 {
        // c^2 - sqrt(2 / pi) sum_k (-1)^k 2 c^(2k + 3) / (2^k k! (2k + 1) (2k + 3))
        let mut power = multiply(c2, (c, 0.0));
        let mut sum = (0.0, 0.0);
        for k in 0..400_u32 {
            if k > 0 {
                power = divide(multiply(power, c2), (f64::from(2 * k), 0.0));
            }
            let term = divide(
                (2.0 * power.0, 2.0 * power.1),
                (f64::from((2 * k + 1) * (2 * k + 3)), 0.0),
            );
            sum = add(sum, if k % 2 == 0 { term } else { negate(term) });
            if term.0 <= sum.0.abs() * f64::EPSILON * f64::EPSILON * 1e-3 {
                break;
            }
        }
        add(c2, negate(multiply(SQRT_2_OVER_PI, sum)))
    } else {
        // 1 - 2 phi(c) ((1 - c^2) R(c) + c), R the Mills ratio by its
        // continued fraction; phi underflows to 0 far beyond where the
        // tail matters.
        let mut fraction = c;
        for k in (1..=80_u32).rev() {
            fraction = c + f64::from(k) / fraction;
        }
        let phi = (-0.5 * c * c).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let tail = 2.0 * phi * ((1.0 - c * c) / fraction + c);
        two_sum(1.0, -tail)
    }
}

/// Huber's proposal 2 location of `sorted` (finite, at least one, ascending
/// by `total_cmp`) with tuning constant `c` (from `1e-30` to `1e30`) and
/// `gamma` [`gamma(c)`](gamma).
pub(crate) fn huber_joint(
    sorted: &[f64],
    scratch: &mut JointScratch,
    c: f64,
    gamma: Double,
) -> f64 {
    joint(sorted, scratch, c, gamma, ROUNDS).0
}

/// The location and scale, with at most `rounds` rounds of split and solve
/// before statsmodels' update.
pub(crate) fn joint(
    sorted: &[f64],
    scratch: &mut JointScratch,
    c: f64,
    gamma: Double,
    rounds: usize,
) -> (f64, f64) {
    let (first, last) = (sorted[0], sorted[sorted.len() - 1]);
    if first == last {
        return (first, 0.0);
    }
    // Scaled by a power of two (exactly) so the largest magnitude is
    // between 1 and 2: no square overflows or loses bits to the subnormals.
    let power = exponent(first.abs().max(last.abs()));
    scratch.scaled.clear();
    scratch
        .scaled
        .extend(sorted.iter().map(|&x| scale(x, -power)));
    let x = &scratch.scaled[..];
    let split = settle(x, &mut scratch.deviations, c, gamma.0, rounds);
    let (mu, s) = precise(x, split, c, gamma);
    (scale(mu, power), scale(s, power))
}

/// How many values are below `mu - k` and above `mu + k`.
fn split(x: &[f64], mu: f64, k: f64) -> (usize, usize) {
    let below = x.partition_point(|&v| v < mu - k);
    let to = x.partition_point(|&v| v <= mu + k);
    (below, x.len() - to)
}

/// The solution `(mu, s)` for the split `(below, above)`, in doubles, if it
/// has one with `s` at least 0.
#[expect(clippy::cast_precision_loss, reason = "window counts")]
fn solve(x: &[f64], (below, above): (usize, usize), c: f64, gamma: f64) -> Option<(f64, f64)> {
    let n = x.len();
    let (inside, power) = match own_scale(&x[below..n - above]) {
        Inside::Empty => return None,
        // Scale 0 only where no positive scale solves the equations: as s
        // falls to 0, mu sits (c s (above - below) / m) from the tied value
        // (if that is within c s), so the scale equation's sum over s^2
        // tends to c^2 ((above - below)^2 / m + n - m).
        Inside::Equal(value) => {
            let m = (n - below - above) as f64;
            let pull = above as f64 - below as f64;
            let limit = c * c * (pull * pull / m + (n as f64 - m));
            let zero = pull.abs() <= m && limit <= (n - 1) as f64 * gamma;
            return zero.then_some((value, 0.0));
        }
        Inside::Spread(power) => (&x[below..n - above], power),
    };
    let m = inside.len() as f64;
    let a = inside.iter().map(|&v| scale(v, -power)).sum::<f64>() / m;
    let (s1, s2) = inside.iter().fold((0.0, 0.0), |(s1, s2), &v| {
        let d = scale(v, -power) - a;
        (s1 + d, s2 + d * d)
    });
    let b = c * (above as f64 - below as f64) / m;
    let qa = (n - 1) as f64 * gamma - m * b * b - (below + above) as f64 * c * c;
    let qb = 2.0 * b * s1;
    let root = (qb * qb + 4.0 * qa * s2).sqrt();
    let s = if qb >= 0.0 {
        2.0 * s2 / (qb + root)
    } else {
        (root - qb) / (2.0 * qa)
    };
    (s > 0.0 && s.is_finite()).then(|| (scale(a + b * s, power), scale(s, power)))
}

/// The points inside a split.
enum Inside {
    Empty,
    /// All the same value.
    Equal(f64),
    /// Not all the same: the power of two that scales their largest
    /// magnitude to between 1 and 2. They enter the equations only through
    /// their own sums (those outside only by their count), so they are
    /// scaled by it apart from the window: no square of a deviation
    /// underflows, however far below the window's largest value they are.
    Spread(i32),
}

fn own_scale(inside: &[f64]) -> Inside {
    match (inside.first(), inside.last()) {
        (Some(&first), Some(&last)) if first == last => Inside::Equal(first),
        (Some(&first), Some(&last)) => Inside::Spread(exponent(first.abs().max(last.abs()))),
        _ => Inside::Empty,
    }
}

/// The split at the solution: split and solve, from statsmodels' start (the
/// median and the MAD over 0.6745), until the split holds; then, if
/// `rounds` run out first, by bisection.
fn settle(
    x: &[f64],
    deviations: &mut Vec<f64>,
    c: f64,
    gamma: f64,
    rounds: usize,
) -> (usize, usize) {
    let n = x.len();
    let median = 0.5 * (x[(n - 1) / 2] + x[n / 2]);
    deviations.clear();
    deviations.extend(x.iter().map(|v| (v - median).abs()));
    deviations.sort_unstable_by(f64::total_cmp);
    let mad = 0.5 * (deviations[(n - 1) / 2] + deviations[n / 2]);
    // The range where the MAD is 0.
    let s = if mad > 0.0 {
        mad / 0.674_489_750_196_081_7
    } else {
        x[n - 1] - x[0]
    };
    let mut at = split(x, median, c * s);
    for _ in 0..rounds {
        match solve(x, at, c, gamma) {
            Some((m, t)) if split(x, m, c * t) == at => return at,
            Some((m, t)) => at = split(x, m, c * t),
            None => break,
        }
    }
    bisect(x, c, gamma)
}

/// The split at the solution by bisection on `s`, each `s` with its own
/// best `mu` ([`location`]): the scale equation's excess falls as `s`
/// grows, so the last `s` with an excess is the solution's, or 0 if the
/// smallest has none.
#[expect(clippy::cast_precision_loss, reason = "window counts")]
fn bisect(x: &[f64], c: f64, gamma: f64) -> (usize, usize) {
    let n = x.len();
    let excess = |s: f64| {
        let mu = location(x, c * s);
        let u = |v: f64| (v - mu) / s;
        let sum: f64 = x.iter().map(|&v| (u(v) * u(v)).min(c * c)).sum();
        sum > (n - 1) as f64 * gamma
    };
    // Every |x - mu| is at most 4 (the window scaled to below 2), so above
    // 8 / sqrt(gamma) the sum is below 16 n gamma / 64: no excess.
    let (mut low, mut high) = (1_u64, (8.0 / gamma.sqrt()).to_bits());
    if !excess(f64::from_bits(low)) {
        return split(x, location(x, 0.0), 0.0);
    }
    // Positive doubles are in the order of their bits.
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if excess(f64::from_bits(middle)) {
            low = middle;
        } else {
            high = middle;
        }
    }
    let s = f64::from_bits(high);
    split(x, location(x, c * s), c * s)
}

/// Huber's location of `x` (sorted) for the fixed half-width `k`: where
/// `sum clip(x_i - mu, -k, k)` changes sign, by bisection on the doubles
/// from `x[0]` to the last; the median's lower middle for `k` 0.
fn location(x: &[f64], k: f64) -> f64 {
    let n = x.len();
    if k == 0.0 {
        return x[(n - 1) / 2];
    }
    let below = |mu: f64| x.iter().map(|&v| (v - mu).clamp(-k, k)).sum::<f64>() > 0.0;
    let (mut low, mut high) = (ordered(x[0]), ordered(x[n - 1]));
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if below(unordered(middle)) {
            low = middle;
        } else {
            high = middle;
        }
    }
    unordered(high)
}

/// The doubles as integers in their order (negative ones reversed).
fn ordered(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    }
}

fn unordered(key: u64) -> f64 {
    f64::from_bits(if key >> 63 == 1 {
        key & !(1 << 63)
    } else {
        !key
    })
}

/// The solution for the split `(below, above)` in double-double: `mu`
/// rounded to the nearest double, and `s`.
#[expect(clippy::cast_precision_loss, reason = "window counts")]
fn precise(x: &[f64], (below, above): (usize, usize), c: f64, gamma: Double) -> (f64, f64) {
    let n = x.len();
    let inside = &x[below..n - above];
    // A split from `settle` has points inside (the median if not).
    let Inside::Spread(power) = own_scale(inside) else {
        return (inside.first().copied().unwrap_or(x[n / 2]), 0.0);
    };
    let m = (inside.len() as f64, 0.0);
    let a = divide(
        inside.iter().fold((0.0, 0.0), |sum, &v| {
            accumulate(sum, (scale(v, -power), 0.0))
        }),
        m,
    );
    let (s1, s2) = inside
        .iter()
        .fold(((0.0, 0.0), (0.0, 0.0)), |(s1, s2), &v| {
            let d = add((scale(v, -power), 0.0), negate(a));
            (accumulate(s1, d), accumulate(s2, multiply(d, d)))
        });
    let b = divide(two_product(c, above as f64 - below as f64), m);
    let qa = add(
        multiply(((n - 1) as f64, 0.0), gamma),
        negate(add(
            multiply(m, multiply(b, b)),
            multiply(((below + above) as f64, 0.0), two_product(c, c)),
        )),
    );
    let qb = multiply((2.0, 0.0), multiply(b, s1));
    let root = square_root(add(
        multiply(qb, qb),
        multiply((4.0, 0.0), multiply(qa, s2)),
    ));
    let s = if qb.0 >= 0.0 {
        divide((2.0 * s2.0, 2.0 * s2.1), add(qb, root))
    } else {
        divide(add(root, negate(qb)), (2.0 * qa.0, 2.0 * qa.1))
    };
    (scale(add(a, multiply(b, s)).0, power), scale(s.0, power))
}

/// `a / b` in double-double.
pub(crate) fn divide(a: Double, b: Double) -> Double {
    let q1 = a.0 / b.0;
    let r = add(a, negate(multiply((q1, 0.0), b)));
    let q2 = r.0 / b.0;
    let r = add(r, negate(multiply((q2, 0.0), b)));
    let (hi, lo) = two_sum(q1, q2);
    two_sum(hi, lo + r.0 / b.0)
}

/// `sqrt(a)` in double-double, for `a` at least 0 (0 for a negative `a`,
/// which rounding alone can give).
pub(crate) fn square_root(a: Double) -> Double {
    if a.0 <= 0.0 {
        return (0.0, 0.0);
    }
    let x = a.0.sqrt();
    let r = add(a, negate(two_product(x, x)));
    two_sum(x, r.0 / (2.0 * x))
}
