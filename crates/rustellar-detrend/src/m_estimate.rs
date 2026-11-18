//! M-estimators of location of a window's fluxes, to full precision:
//! Tukey's biweight, Welsch's, Andrews' sine wave and Huber's.
//!
//! With the median `M` and `MAD = median |x_i - M|`, the location `T` solves
//! `sum psi((x_i - T) / (c MAD)) = 0` (see [`crate::psi`]): the root that
//! wotan's iteration `T <- T + sum w_i d_i / sum w_i` (`d_i = x_i - T`, `w_i`
//! the weights) reaches from the median. `MAD = 0` gives `M`.
//!
//! Here: the median and MAD exactly (as double-doubles, a double and its
//! remainder), wotan's iteration until its step stops shrinking, Newton's
//! method on the residual `sum w_i d_i`, and last the two doubles either side
//! of the residual's sign change, the nearer taken: the root correctly
//! rounded wherever the residual's sign is certain. The residual is summed
//! plainly with a bound on its error, and in double-double where that bound
//! leaves the sign in doubt. Each window is first scaled by a power of two
//! (exact) so that its largest magnitude is between 1 and 2: no deviation
//! overflows and none loses bits to the subnormals.

use crate::{
    double::{
        Double, accumulate, add, exponent, half, negate, power_of_two, scale, split, two_product,
        two_product_halves, two_sum,
    },
    location::midpoint,
    psi::{Doubles, Isa, LANES, Lanes, Plain, Psi, two_sum_lanes},
};

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

/// Scratch space for [`solve`], kept between windows so that no
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
    solve::<crate::psi::Biweight>(values, scratch, c)
}

/// The location of `sorted` (finite, at least one, ascending by `total_cmp`)
/// for the psi function `P` with tuning constant `c` (in MADs, from 2^-100
/// to 2^100).
///
/// Every sum runs over the values in this order, so the result depends on
/// the values alone, not on the order they came in.
///
/// The same bits whichever instructions the processor has: built twice, for
/// x86-64's baseline and with AVX2 (wider vectors; no fused multiply-add,
/// which would round differently), the second used where the processor has
/// it.
pub(crate) fn solve<P: Psi>(sorted: &[f64], scratch: &mut Scratch, c: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if let Some(avx2) = wide() {
        // SAFETY: an `Avx2` exists only where the processor has AVX2.
        return unsafe { solve_wide::<P>(avx2, sorted, scratch, c) };
    }
    solve_with::<P, Plain>(Plain, sorted, scratch, c)
}

/// AVX2's vectors, if the processor has them (and the tests have not turned
/// them off).
#[cfg(target_arch = "x86_64")]
pub(crate) fn wide() -> Option<crate::psi::Avx2> {
    #[cfg(test)]
    if NARROW.with(std::cell::Cell::get) {
        return None;
    }
    crate::psi::Avx2::new()
}

#[cfg(all(test, target_arch = "x86_64"))]
thread_local! {
    /// Set by the tests to run the baseline build.
    pub(crate) static NARROW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`solve`] built with AVX2.
///
/// # Safety
///
/// The processor must have AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn solve_wide<P: Psi>(
    avx2: crate::psi::Avx2,
    sorted: &[f64],
    scratch: &mut Scratch,
    c: f64,
) -> f64 {
    solve_with::<P, crate::psi::Avx2>(avx2, sorted, scratch, c)
}

/// [`solve`], inlined into each build together with all it calls.
#[inline(always)]
fn solve_with<P: Psi, V: Isa>(v: V, sorted: &[f64], scratch: &mut Scratch, c: f64) -> f64 {
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
        let (residual, weight) = iterate::<P, V>(v, values, t, inverse);
        if weight == 0.0 {
            break;
        }
        let step = residual / weight;
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
    let mut here = newton::<P, V>(v, values, t, inverse);
    for _ in 0..NEWTON_STEPS {
        if here.slope <= 0.0 || here.residual == 0.0 {
            break;
        }
        // Within two doubles the search below takes over.
        let next = t + here.residual / here.slope;
        if (next - t).abs() <= 2.0 * gap(t) {
            break;
        }
        let there = newton::<P, V>(v, values, next, inverse);
        if there.residual.abs() >= here.residual.abs() {
            break;
        }
        (t, here) = (next, there);
    }

    let sums = Sums::<P, V> {
        values,
        s,
        inverse,
        isa: v,
        psi: std::marker::PhantomData,
    };
    let mut at_t = sums.cheap(t);
    if !at_t.certain() {
        // The sums cannot tell the residual from zero here: Newton's method
        // on the double-double residual, while it shrinks. The first step
        // takes the slope as last found; then the secant through the last
        // two residuals, as the sums' slope can be off (a point at the edge
        // of a cut counted on the wrong side) and near a root at zero, where
        // the doubles are dense, a slow approach would stop far from it.
        at_t = Estimate::exact(sums.precise(t));
        let mut slope = here.slope;
        for _ in 0..NEWTON_STEPS {
            if slope <= 0.0 || !slope.is_finite() || at_t.value == 0.0 {
                break;
            }
            let next = t + at_t.value / slope;
            if (next - t).abs() <= 2.0 * gap(t) {
                break;
            }
            let at_next = sums.precise(next);
            if at_next.abs() >= at_t.value.abs() {
                break;
            }
            slope = (at_t.value - at_next) / (next - t);
            (t, at_t) = (next, Estimate::exact(at_next));
        }
    }

    // The nearer double either side of the sign change, among those exact
    // once scaled back (multiples of `fine`).
    let fine = power_of_two((-1074 - e).max(-1074));
    let t = nearest_root(t, at_t, fine, &sums);
    scale(t, e)
}

/// `x - median`, exactly as a double-double (to the last bits of one).
#[inline(always)]
fn deviation(x: f64, median: Double) -> Double {
    let (d, error) = two_sum(x, -median.0);
    two_sum(d, error - median.1)
}

/// The median absolute deviation of `sorted` from `median`, as a
/// double-double: the deviations are taken outward from the median, the
/// smaller first, up to the middle one (or two).
#[inline(always)]
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

/// The values with `|u| = |x - t| / s` below `P`'s reach, `u` computed as
/// the sums do (`(x - t) inverse`, `inverse = 1 / s`): a run of the sorted
/// values, as `u` never decreases with `x`. All of them for an infinite
/// reach.
#[inline(always)]
fn inside<P: Psi>(sorted: &[f64], t: f64, inverse: f64) -> &[f64] {
    let start = sorted.partition_point(|&x| (x - t) * inverse <= -P::REACH);
    let end = sorted.partition_point(|&x| (x - t) * inverse < P::REACH);
    &sorted[start..end]
}

/// The residual `sum w_i d_i` at `t` for `c MAD = s` (`inverse` its
/// reciprocal), each term and the sum in double-double: wrong only in the
/// last bits of a double-double.
#[inline(always)]
fn residual<P: Psi, V: Isa>(v: V, values: &[f64], t: f64, s: Double, inverse: f64) -> f64 {
    let s_halves = split(s.0);
    // Value `k` goes to lane `k % LANES`, the lanes summed in order.
    let mut lanes = [(0.0, 0.0); LANES];
    let inside = inside::<P>(values, t, inverse);
    let mut blocks = inside.chunks_exact(BLOCK);
    for block in &mut blocks {
        let x = std::array::from_fn(|g| run(&block[g * LANES..]));
        let terms = terms::<P, V>(v, x, t, s, s_halves, inverse);
        for group in terms {
            for (l, lane) in lanes.iter_mut().enumerate() {
                *lane = accumulate(*lane, (group.0[l], group.1[l]));
            }
        }
    }
    // The rest padded with `t` (whose terms are not added).
    let rest = blocks.remainder();
    if !rest.is_empty() {
        let x = std::array::from_fn(|g| {
            std::array::from_fn(|l| rest.get(g * LANES + l).copied().unwrap_or(t))
        });
        let terms = terms::<P, V>(v, x, t, s, s_halves, inverse);
        for k in 0..rest.len() {
            let (group, l) = (terms[k / LANES], k % LANES);
            lanes[l] = accumulate(lanes[l], (group.0[l], group.1[l]));
        }
    }
    let total = add(add(lanes[0], lanes[1]), add(lanes[2], lanes[3]));
    total.0 + total.1
}

/// Groups of [`LANES`] values whose double-double terms are worked out
/// side by side.
const GROUPS: usize = 4;
const BLOCK: usize = GROUPS * LANES;

/// The terms `w(u) d` of [`GROUPS`] groups of values in double-double,
/// `d = x - t` exactly and `u = d / s` with its correction.
#[inline(always)]
fn terms<P: Psi, V: Isa>(
    v: V,
    x: [Lanes; GROUPS],
    t: f64,
    s: Double,
    s_halves: Double,
    inverse: f64,
) -> [Doubles; GROUPS] {
    let d: [Doubles; GROUPS] = std::array::from_fn(|g| two_sum_lanes(x[g], [-t; LANES]));
    let u = std::array::from_fn(|g| {
        let d = d[g];
        let u_hi: Lanes = std::array::from_fn(|l| d.0[l] * inverse);
        let u_lo = std::array::from_fn(|l| {
            let (p, p_error) = two_product_halves(u_hi[l], s.0, s_halves);
            (((d.0[l] - p) - p_error) + d.1[l] - u_hi[l] * s.1) * inverse
        });
        (u_hi, u_lo)
    });
    P::terms(v, u, d, s)
}

/// The values at or near `t` (see [`inside`]) lane by lane: whole runs of
/// [`LANES`], then the rest (if any) padded with `t` itself, with `1` in
/// `real` for each real lane and `0` for padding. A padded lane's deviation
/// is 0, so it adds nothing to a sum of `w d` (nor to one of `|d| m`); the
/// other sums take `real` times their terms.
#[inline(always)]
fn lanes_of<P: Psi>(
    values: &[f64],
    t: f64,
    inverse: f64,
) -> (std::slice::ChunksExact<'_, f64>, Option<(Lanes, Lanes)>) {
    let chunks = inside::<P>(values, t, inverse).chunks_exact(LANES);
    let rest = chunks.remainder();
    let padded = (!rest.is_empty()).then(|| {
        (
            std::array::from_fn(|l| rest.get(l).copied().unwrap_or(t)),
            std::array::from_fn(|l| if l < rest.len() { 1.0 } else { 0.0 }),
        )
    });
    (chunks, padded)
}

/// A run of [`LANES`] values as lanes.
#[inline(always)]
fn run(chunk: &[f64]) -> Lanes {
    std::array::from_fn(|l| chunk[l])
}

/// Lanes added in a fixed order.
#[inline(always)]
fn total(a: Lanes) -> f64 {
    (a[0] + a[1]) + (a[2] + a[3])
}

/// The plain sums at `t` that steer wotan's iteration, with the rough
/// weights: the residual `sum w_i d_i` and the weight `sum w_i`.
#[inline(always)]
fn iterate<P: Psi, V: Isa>(v: V, values: &[f64], t: f64, inverse: f64) -> (f64, f64) {
    #[inline(always)]
    fn step<P: Psi, V: Isa>(
        v: V,
        x: Lanes,
        real: Lanes,
        t: f64,
        inverse: f64,
        sums: &mut [Lanes; 2],
    ) {
        let d: Lanes = std::array::from_fn(|l| x[l] - t);
        let w = P::rough(v, std::array::from_fn(|l| d[l] * inverse));
        for l in 0..LANES {
            sums[0][l] += w[l] * d[l];
            sums[1][l] += w[l] * real[l];
        }
    }
    let mut sums = [[0.0; LANES]; 2];
    let (runs, rest) = lanes_of::<P>(values, t, inverse);
    for chunk in runs {
        step::<P, V>(v, run(chunk), [1.0; LANES], t, inverse, &mut sums);
    }
    if let Some((x, real)) = rest {
        step::<P, V>(v, x, real, t, inverse, &mut sums);
    }
    (total(sums[0]), total(sums[1]))
}

/// The plain sums at `t` for Newton's method: the residual `sum w_i d_i`
/// and the slope `sum psi'(u_i)` (minus the residual's derivative in `t`).
#[derive(Clone, Copy)]
struct Newton {
    residual: f64,
    slope: f64,
}

#[inline(always)]
fn newton<P: Psi, V: Isa>(v: V, values: &[f64], t: f64, inverse: f64) -> Newton {
    #[inline(always)]
    fn step<P: Psi, V: Isa>(
        v: V,
        x: Lanes,
        real: Lanes,
        t: f64,
        inverse: f64,
        sums: &mut [Lanes; 2],
    ) {
        let d: Lanes = std::array::from_fn(|l| x[l] - t);
        let (w, _, slope) = P::weigh(v, std::array::from_fn(|l| d[l] * inverse));
        for l in 0..LANES {
            sums[0][l] += w[l] * d[l];
            sums[1][l] += slope[l] * real[l];
        }
    }
    let mut sums = [[0.0; LANES]; 2];
    let (runs, rest) = lanes_of::<P>(values, t, inverse);
    for chunk in runs {
        step::<P, V>(v, run(chunk), [1.0; LANES], t, inverse, &mut sums);
    }
    if let Some((x, real)) = rest {
        step::<P, V>(v, x, real, t, inverse, &mut sums);
    }
    Newton {
        residual: total(sums[0]),
        slope: total(sums[1]),
    }
}

/// The residual `sum w_i d_i` at `t`, compensated and with `d_i` exact, and
/// a bound on its error.
#[inline(always)]
fn bounded<P: Psi, V: Isa>(v: V, values: &[f64], t: f64, inverse: f64) -> (f64, f64) {
    // The residual's high parts, the errors gathered, and sum |d| m(u) for
    // the bound.
    #[inline(always)]
    fn step<P: Psi, V: Isa>(v: V, x: Lanes, _: Lanes, t: f64, inverse: f64, sums: &mut [Lanes; 3]) {
        let exact: [(f64, f64); LANES] = std::array::from_fn(|l| two_sum(x[l], -t));
        let (w, m, _) = P::weigh(v, std::array::from_fn(|l| exact[l].0 * inverse));
        for l in 0..LANES {
            let (d, d_low) = exact[l];
            let (sum, error) = two_sum(sums[0][l], w[l] * d);
            sums[0][l] = sum;
            sums[1][l] += error + w[l] * d_low;
            sums[2][l] += d.abs() * m[l];
        }
    }
    let mut sums = [[0.0; LANES]; 3];
    let (runs, rest) = lanes_of::<P>(values, t, inverse);
    for chunk in runs {
        step::<P, V>(v, run(chunk), [1.0; LANES], t, inverse, &mut sums);
    }
    if let Some((x, real)) = rest {
        step::<P, V>(v, x, real, t, inverse, &mut sums);
    }
    let high = add(
        two_sum(sums[0][0], sums[0][1]),
        two_sum(sums[0][2], sums[0][3]),
    );
    let residual = high.0 + (high.1 + total(sums[1]));
    // The terms', the summation's (compensated: a few eps of the sum and
    // eps^2 of the terms, well inside the terms' share) and the last
    // rounding's.
    let bound = f64::EPSILON * (P::TERM_ERROR * total(sums[2]) + 2.0 * residual.abs());
    (residual, bound)
}

/// The residual two ways, for [`nearest_root`].
pub(crate) trait Residual {
    /// With a bound on its error.
    fn cheap(&self, t: f64) -> Estimate;
    /// Taken as exact.
    fn precise(&self, t: f64) -> f64;
}

/// The sums of one window: methods rather than closures, so that they are
/// built into the solver's wide build with all else.
struct Sums<'a, P, V> {
    values: &'a [f64],
    s: Double,
    inverse: f64,
    isa: V,
    psi: std::marker::PhantomData<P>,
}

impl<P: Psi, V: Isa> Residual for Sums<'_, P, V> {
    #[inline(always)]
    fn cheap(&self, t: f64) -> Estimate {
        let (value, bound) = bounded::<P, V>(self.isa, self.values, t, self.inverse);
        Estimate { value, bound }
    }

    #[inline(always)]
    fn precise(&self, t: f64) -> f64 {
        residual::<P, V>(self.isa, self.values, t, self.s, self.inverse)
    }
}

/// A cheap and a precise residual given as functions.
#[cfg(test)]
impl<C: Fn(f64) -> Estimate, R: Fn(f64) -> f64> Residual for (C, R) {
    fn cheap(&self, t: f64) -> Estimate {
        (self.0)(t)
    }

    fn precise(&self, t: f64) -> f64 {
        (self.1)(t)
    }
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
/// The residual comes from `residual.cheap`, with a bound on its error, and
/// from `residual.precise` (taken as exact) wherever that bound leaves a
/// sign, or which side is nearer, in doubt. `at_t` is the residual at `t`,
/// known already.
#[inline(always)]
pub(crate) fn nearest_root(t: f64, at_t: Estimate, fine: f64, residual: &impl Residual) -> f64 {
    let settle = |t: f64, at: Estimate| {
        if at.certain() {
            at
        } else {
            Estimate::exact(residual.precise(t))
        }
    };
    // Off the multiples only where the doubles are finer than `fine`, so
    // `t / fine` is small there.
    let (t, at_t) = if on_doubles(t, fine) {
        (t, at_t)
    } else {
        let snapped = (t / fine).round() * fine;
        (snapped, residual.cheap(snapped))
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
        let at_far = settle(far, residual.cheap(far));
        if at_far.value == 0.0 {
            return far;
        }
        if (at_far.value > 0.0) != up {
            // Which is nearer: certain if the sizes differ by more than the
            // bounds, else both precisely (each unless known so already).
            let doubt = at_near.bound + at_far.bound;
            let (at_near, at_far) = if (at_near.value.abs() - at_far.value.abs()).abs() > doubt {
                (at_near.value, at_far.value)
            } else {
                let exact = |t: f64, at: Estimate| {
                    if at.bound == 0.0 {
                        at.value
                    } else {
                        residual.precise(t)
                    }
                };
                (exact(near, at_near), exact(far, at_far))
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
#[inline(always)]
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
#[inline(always)]
fn gap(t: f64) -> f64 {
    (next_double(t, true) - t)
        .abs()
        .max((next_double(t, false) - t).abs())
}

/// Whether the doubles next to `t` are at least `fine` apart, so that each
/// is a multiple of `fine`.
#[inline(always)]
fn on_doubles(t: f64, fine: f64) -> bool {
    (next_double(t, true) - t).abs() >= fine && (next_double(t, false) - t).abs() >= fine
}

/// The next multiple of `fine` among the doubles from `t` (one), up or down.
#[inline(always)]
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
