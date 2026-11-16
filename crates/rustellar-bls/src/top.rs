//! The periods whose power must be exact for a search's peaks and best box
//! to be the exact search's, when the power elsewhere is known only to
//! within a bound.
//!
//! With `g` the power known to within `e` of the exact `P` at each period,
//! and a threshold `t`: outside `A = {i : g_i + e_i >= t}` both `P` and `g`
//! are below `t`. Make `P` exact on `A` and the periods next to it. Then on
//! `A` the merged power is `P`, and so is whether each period is a local
//! maximum, which looks only at its neighbours; outside `A` it is below `t`
//! either way. So the local maxima at or above `t` are the same in the
//! merged power as in `P`, with the same values, and if there are at least
//! the wanted number of them, the highest are the same, in the same order.
//! That is checked once `P` is exact on `A`; if there are too few, `t` is
//! lowered and more periods made exact. The highest power is in `A` too
//! once `t` is at most the highest `g_i - e_i`.

use crate::BlsError;

/// Writes the exact power of the periods given into the power.
pub(crate) type Exact<'a> = dyn FnMut(&[usize], &mut [f64]) -> Result<(), BlsError> + 'a;

/// The bound on the power known only approximately: this fraction of it...
pub(crate) const RELATIVE: f64 = 1.0 / (1u64 << 30) as f64;
/// ...and never less than this fraction of the highest.
const FLOOR: f64 = 1.0 / (1u64 << 20) as f64;

/// Makes `power` exact, by `exact`, at the periods that decide the `count`
/// highest local maxima of the power and the highest power (as
/// [`peaks`](crate::refine::peaks) finds them), so that both are the same
/// as with the exact power everywhere. Before, `power` is exact where
/// `known` is true and within [`RELATIVE`] of the exact elsewhere.
///
/// `exact` is given the periods to make exact, each once (in order within
/// a call), and writes their exact power into `power`.
pub(crate) fn make_top_exact(
    power: &mut [f64],
    known: &[bool],
    count: usize,
    exact: &mut Exact<'_>,
) -> Result<(), BlsError> {
    let n = power.len();
    let approximate: Vec<f64> = power.to_vec();
    let highest = approximate.iter().copied().fold(0.0, f64::max);
    let bound = |i: usize| {
        if known[i] {
            0.0
        } else {
            RELATIVE * approximate[i].max(FLOOR * highest)
        }
    };
    let mut done: Vec<bool> = known.to_vec();
    let mut make = |wanted: &dyn Fn(usize) -> bool, power: &mut [f64]| {
        let mut todo = Vec::new();
        let near = |j: usize| wanted(j) || (j > 0 && wanted(j - 1)) || (j + 1 < n && wanted(j + 1));
        for (i, done) in done.iter_mut().enumerate() {
            if !*done && near(i) {
                todo.push(i);
                *done = true;
            }
        }
        if todo.is_empty() {
            Ok(())
        } else {
            exact(&todo, power)
        }
    };
    if highest <= 0.0 {
        // No box with a dip, to within the bound: every period exact.
        return make(&|_| true, power);
    }

    // The highest power: at or above the highest lower bound.
    let lowest_best = (0..n)
        .map(|i| approximate[i] - bound(i))
        .fold(f64::NEG_INFINITY, f64::max);
    make(&|i| approximate[i] + bound(i) >= lowest_best, power)?;
    if count == 0 {
        return Ok(());
    }

    // The peaks: from the `count`th highest local maximum of the
    // approximate power, less its bound twice.
    let top = crate::refine::peaks(&approximate, count);
    let mut threshold = if top.len() == count {
        let at = top[count - 1];
        approximate[at] - 2.0 * bound(at)
    } else {
        0.0
    };
    loop {
        let t = threshold;
        if t <= 0.0 {
            return make(&|_| true, power);
        }
        make(&|i| approximate[i] + bound(i) >= t, power)?;
        let found = (0..n)
            .filter(|&i| {
                approximate[i] + bound(i) >= t
                    && power[i] >= t
                    && power[i] > 0.0
                    && (i == 0 || power[i] > power[i - 1])
                    && (i + 1 == n || power[i] >= power[i + 1])
            })
            .count();
        if found >= count {
            return Ok(());
        }
        // Too few: lower the threshold, to zero once it is far below the
        // highest power.
        threshold = if t < FLOOR * RELATIVE * highest {
            0.0
        } else {
            0.5 * t
        };
    }
}
