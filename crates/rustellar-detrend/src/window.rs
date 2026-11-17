//! Which points each point's trend is made from: the light curve split into
//! segments at its gaps, and for each point the range of points within half
//! a window of it, in its segment and not masked. Index work only, no
//! statistics.
//!
//! Every comparison of a time difference with a length (a gap with the break
//! tolerance, a distance with half the window, a distance from a segment's
//! end with the edge cutoff) is decided exactly, as if in real numbers, so a
//! point exactly on an edge is always on the same side of it.

use std::{cmp::Ordering, ops::Range};

/// Which ends of a window hold the points exactly half a window away.
///
/// # Examples
///
/// ```
/// use rustellar_detrend::Edges;
///
/// assert_eq!(Edges::default(), Edges::Symmetric);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Edges {
    /// Both: the window is symmetric about its point.
    #[default]
    Symmetric,
    /// The earlier only, as in wotan: `[t - w/2, t + w/2)`.
    HalfOpen,
}

/// How the windows are made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WindowOptions {
    /// The window's length, in days: greater than zero and finite.
    pub(crate) window: f64,
    /// The longest gap within a segment, in days; a longer one starts a new
    /// segment. Infinite for none.
    pub(crate) break_tolerance: f64,
    /// Points closer than this, in days, to either end of their segment get
    /// no window. At most half the window: a longer cutoff counts as half.
    pub(crate) edge_cutoff: f64,
    /// Which window ends hold the points exactly half a window away.
    pub(crate) edges: Edges,
}

/// The windows of a light curve's points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Windows {
    /// The points' segments, in order, covering every point.
    pub(crate) segments: Vec<Range<usize>>,
    /// The indices of the points not masked, in order.
    pub(crate) kept: Vec<usize>,
    /// For each point, the range of `kept` in its window, or `None` where
    /// the edge cutoff leaves it out. Empty where every point within reach
    /// is masked.
    pub(crate) windows: Vec<Option<Range<usize>>>,
}

/// The sign of `a - b - c`, exactly, for `a >= b` finite and `c >= 0` (less
/// for an infinite `c`).
///
/// `a - b = d + e` and `d - c = f + g` exactly (two-sums), so `a - b - c =
/// f + g + e`. If `g` is zero, `f + e` rounded has the sign of the exact sum.
/// Otherwise `d - c` is not exact, so `d` and `c` are more than a factor of
/// two apart (Sterbenz) and `|f| >= max(d, c) / 2`; then `|g| <= |f| 2^-52`
/// and `|e| <= d 2^-53 <= |f| 2^-52`, so `f`'s sign is the sum's.
pub(crate) fn exceeds(a: f64, b: f64, c: f64) -> Ordering {
    debug_assert!(a >= b && c >= 0.0);
    if c == f64::INFINITY {
        return Ordering::Less;
    }
    let (d, e) = two_difference(a, b);
    let (f, g) = two_difference(d, c);
    let sign = if g == 0.0 { f + e } else { f };
    sign.partial_cmp(&0.0).expect("finite")
}

/// `x - y` and its rounding error: `x - y = s + e` exactly (Knuth).
fn two_difference(x: f64, y: f64) -> (f64, f64) {
    let s = x - y;
    let z = s - x;
    let e = (x - (s - z)) - (y + z);
    (s, e)
}

/// The segments of `times` (strictly increasing): split wherever the gap
/// between neighbours is longer than `break_tolerance`.
pub(crate) fn segments(times: &[f64], break_tolerance: f64) -> Vec<Range<usize>> {
    let mut segments = Vec::new();
    let mut start = 0;
    for k in 1..times.len() {
        if exceeds(times[k], times[k - 1], break_tolerance) == Ordering::Greater {
            segments.push(start..k);
            start = k;
        }
    }
    if !times.is_empty() {
        segments.push(start..times.len());
    }
    segments
}

/// The windows of the points at `times` (strictly increasing), leaving out
/// those where `masked` is true.
pub(crate) fn windows(times: &[f64], masked: Option<&[bool]>, options: &WindowOptions) -> Windows {
    debug_assert!(options.window > 0.0 && options.window.is_finite());
    debug_assert!(masked.is_none_or(|m| m.len() == times.len()));
    let half = options.window / 2.0;
    let cutoff = options.edge_cutoff.clamp(0.0, half);
    let segments = segments(times, options.break_tolerance);
    let kept: Vec<usize> = (0..times.len())
        .filter(|&k| !masked.is_some_and(|m| m[k]))
        .collect();
    let mut windows = Vec::with_capacity(times.len());
    // The kept points' range in each segment, found as the segments go by.
    let mut first_kept = 0;
    for segment in &segments {
        let (first, last) = (times[segment.start], times[segment.end - 1]);
        let end_kept = first_kept + kept[first_kept..].partition_point(|&k| k < segment.end);
        let (mut lo, mut hi) = (first_kept, first_kept);
        for i in segment.clone() {
            let t = times[i];
            // Within the cutoff of either end: no window.
            if exceeds(t, first, cutoff) == Ordering::Less
                || exceeds(last, t, cutoff) == Ordering::Less
            {
                windows.push(None);
                continue;
            }
            // The first kept point no more than half a window before.
            while lo < end_kept
                && times[kept[lo]] < t
                && exceeds(t, times[kept[lo]], half) == Ordering::Greater
            {
                lo += 1;
            }
            // Past the last kept point within half a window after.
            hi = hi.max(lo);
            while hi < end_kept && {
                let u = times[kept[hi]];
                u <= t
                    || match (exceeds(u, t, half), options.edges) {
                        (Ordering::Less, _) | (Ordering::Equal, Edges::Symmetric) => true,
                        (Ordering::Equal, Edges::HalfOpen) | (Ordering::Greater, _) => false,
                    }
            } {
                hi += 1;
            }
            windows.push(Some(lo..hi));
        }
        first_kept = end_kept;
    }
    Windows {
        segments,
        kept,
        windows,
    }
}
