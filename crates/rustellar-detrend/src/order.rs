//! Locations from the order of a window's fluxes, each correctly rounded:
//! the trimmed and winsorized means, and the Hodges-Lehmann estimator.

use std::cmp::Ordering;

use crate::{double::two_sum, exact_sum::ExactSum};

/// How many values a proportion `p` (from 0 to below 1/2) of `n` cuts from
/// each end: `floor(p n)` of the rounded product, as scipy and wotan, so
/// that 0.3 of 10 is 3 (the double nearest 0.3 is just below it, so the
/// exact product is just below 3).
pub(crate) fn cut(p: f64, n: usize) -> usize {
    debug_assert!((0.0..0.5).contains(&p));
    (p * n as f64).floor() as usize
}

/// The mean of `sorted` (ascending, at least one) without the `floor(p n)`
/// smallest and largest values, correctly rounded.
pub(crate) fn trimmed_mean(sorted: &[f64], p: f64, sum: &mut ExactSum) -> f64 {
    let n = sorted.len();
    let g = cut(p, n);
    sum.clear();
    for &x in &sorted[g..n - g] {
        sum.add(x);
    }
    sum.mean((n - 2 * g) as u64)
}

/// The mean of `sorted` (ascending, at least one) with the `floor(p n)`
/// smallest values replaced by the next, and the largest by the one before,
/// correctly rounded.
pub(crate) fn winsorized_mean(sorted: &[f64], p: f64, sum: &mut ExactSum) -> f64 {
    let n = sorted.len();
    let g = cut(p, n);
    sum.clear();
    for &x in &sorted[g..n - g] {
        sum.add(x);
    }
    for _ in 0..g {
        sum.add(sorted[g]);
        sum.add(sorted[n - 1 - g]);
    }
    sum.mean(n as u64)
}

/// `a + b` exactly, as its rounded sum and the remainder: two such pairs
/// compare as the exact sums do (equal sums round alike, and a smaller
/// rounded sum means a smaller exact one).
fn pair(a: f64, b: f64) -> (f64, f64) {
    two_sum(a, b)
}

fn compare(p: (f64, f64), q: (f64, f64)) -> Ordering {
    p.0.total_cmp(&q.0).then(p.1.total_cmp(&q.1))
}

/// Scratch space for [`hodges_lehmann`], kept between windows.
#[derive(Debug, Default)]
pub(crate) struct Rows {
    low: Vec<usize>,
    high: Vec<usize>,
    below: Vec<usize>,
    at_most: Vec<usize>,
}

/// The Hodges-Lehmann location of `sorted` (ascending, finite, at least
/// one): the median of the `n (n + 1) / 2` averages `(x_i + x_j) / 2`,
/// `i <= j`, correctly rounded.
///
/// The averages form a matrix whose rows (fixed `i`, `j` from `i`) and
/// columns ascend; the middle one is selected without forming it (Monahan
/// 1984): each round a pivot from the candidates left, each row's count
/// below and up to it by one walk, and the candidates cut to one side.
/// O(n) a round and O(log n) rounds expected.
pub(crate) fn hodges_lehmann(sorted: &[f64], rows: &mut Rows, sum: &mut ExactSum) -> f64 {
    let n = sorted.len();
    let total = n * (n + 1) / 2;
    let rank = (total - 1) / 2;
    let (first, at_most) = select(sorted, rows, rank);
    sum.clear();
    sum.add(first.0);
    sum.add(first.1);
    if total % 2 == 1 {
        return sum.mean(2);
    }
    // The next average up: the same, or the least above it.
    let second = if at_most > rank + 1 {
        first
    } else {
        (0..n)
            .filter(|&i| i + rows.at_most[i] < n)
            .map(|i| (sorted[i], sorted[i + rows.at_most[i]]))
            .min_by(|&p, &q| compare(pair(p.0, p.1), pair(q.0, q.1)))
            .expect("an average above the middle")
    };
    sum.add(second.0);
    sum.add(second.1);
    sum.mean(4)
}

/// The pair whose sum is the `rank`-th smallest (from 0), and how many sums
/// are at most it; `rows.at_most[i]` the count in row `i`.
fn select(sorted: &[f64], rows: &mut Rows, rank: usize) -> ((f64, f64), usize) {
    let n = sorted.len();
    // Candidates in row i: columns low[i] .. high[i] (exclusive).
    rows.low.clear();
    rows.low.extend(0..n);
    rows.high.clear();
    rows.high.resize(n, n);
    rows.below.resize(n, 0);
    rows.at_most.resize(n, 0);
    let mut random = 0x2545_f491_4f6c_dd1d_u64;
    loop {
        let candidates: usize = (0..n).map(|i| rows.high[i] - rows.low[i]).sum();
        debug_assert!(candidates > 0);
        // A pivot among the candidates, by a fixed sequence: any choice
        // gives the same answer, only the number of rounds changes.
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let mut pick = (random % candidates as u64) as usize;
        let mut row = 0;
        while pick >= rows.high[row] - rows.low[row] {
            pick -= rows.high[row] - rows.low[row];
            row += 1;
        }
        let pivot_pair = (sorted[row], sorted[rows.low[row] + pick]);
        let pivot = pair(pivot_pair.0, pivot_pair.1);
        // Each row's count below and up to the pivot (columns from i); the
        // boundary column falls as the row rises.
        let (mut below_total, mut at_most_total) = (0, 0);
        let (mut j_below, mut j_at_most) = (n, n);
        for i in 0..n {
            while j_below > i && compare(pair(sorted[i], sorted[j_below - 1]), pivot).is_ge() {
                j_below -= 1;
            }
            while j_at_most > i && compare(pair(sorted[i], sorted[j_at_most - 1]), pivot).is_gt() {
                j_at_most -= 1;
            }
            rows.below[i] = j_below.max(i) - i;
            rows.at_most[i] = j_at_most.max(i) - i;
            below_total += rows.below[i];
            at_most_total += rows.at_most[i];
        }
        if rank < below_total {
            // Below the pivot.
            for i in 0..n {
                rows.high[i] = rows.high[i].min(i + rows.below[i]);
                rows.low[i] = rows.low[i].min(rows.high[i]);
            }
        } else if rank < at_most_total {
            return (pivot_pair, at_most_total);
        } else {
            // Above it.
            for i in 0..n {
                rows.low[i] = rows.low[i].max(i + rows.at_most[i]);
                rows.high[i] = rows.high[i].max(rows.low[i]);
            }
        }
    }
}
