//! The locations a window's fluxes are summarised by.

use std::cmp::Ordering;

/// What a window's fluxes are summarised by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Location {
    /// The median: the middle value, or the correctly rounded midpoint of
    /// the two middle values.
    Median,
    /// The mean, correctly rounded.
    Mean,
}

/// The median of `values` (finite, at least one), reordering them.
///
/// For an even count, the midpoint of the two middle values correctly
/// rounded: `(a + b) / 2` (numpy's, to the bit), or `a / 2 + b / 2` where
/// `a + b` overflows.
pub(crate) fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    debug_assert!(n > 0);
    let middle = n / 2;
    let (below, &mut upper, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    if n % 2 == 1 {
        return upper;
    }
    let lower = below
        .iter()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .expect("an even count has a value below the middle");
    midpoint(lower, upper)
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
