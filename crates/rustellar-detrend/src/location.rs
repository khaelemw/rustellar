//! The locations a window's fluxes are summarised by.

use std::cmp::Ordering;

/// What a window's fluxes are summarised by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Location {
    /// The median: the middle value, or the correctly rounded midpoint of
    /// the two middle values.
    Median,
    /// The mean, correctly rounded.
    Mean,
    /// Tukey's biweight with tuning constant `c` (in MADs; wotan's is 5),
    /// to full precision.
    Biweight(f64),
}

/// The median of `values` (finite, at least one), reordering them.
///
/// For an even count, the midpoint of the two middle values correctly
/// rounded: `(a + b) / 2` (numpy's, to the bit), or `a / 2 + b / 2` where
/// `a + b` overflows.
#[cfg(test)]
pub(crate) fn median(values: &mut [f64]) -> f64 {
    // For an odd count, the midpoint of the middle value with itself: it.
    let (lower, upper) = middle(values);
    midpoint(lower, upper)
}

/// The two middle values of `values` (finite, at least one), lower first,
/// reordering them; the same value twice for an odd count.
#[cfg(test)]
pub(crate) fn middle(values: &mut [f64]) -> (f64, f64) {
    middle_by(values, f64::total_cmp)
}

/// The two middle items of `items` (at least one) in the order `order`,
/// lower first, reordering them; the same item twice for an odd count.
#[cfg(test)]
pub(crate) fn middle_by<T: Copy>(
    items: &mut [T],
    order: impl Fn(&T, &T) -> Ordering + Copy,
) -> (T, T) {
    let n = items.len();
    debug_assert!(n > 0);
    let (below, &mut upper, _) = items.select_nth_unstable_by(n / 2, order);
    if n % 2 == 1 {
        return (upper, upper);
    }
    let lower = below
        .iter()
        .copied()
        .max_by(order)
        .expect("an even count has an item below the middle");
    (lower, upper)
}

/// `(a + b) / 2` correctly rounded, for `a <= b` finite.
pub(crate) fn midpoint(a: f64, b: f64) -> f64 {
    debug_assert!(a.total_cmp(&b) != Ordering::Greater);
    let sum = a + b;
    // Rounding the sum once is correct rounding of the midpoint: halving is
    // exact unless the result is subnormal, and then the sum was exact.
    if sum.is_finite() {
        sum / 2.0
    } else {
        a / 2.0 + b / 2.0
    }
}
