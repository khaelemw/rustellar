//! The fast binned search at one period.
//!
//! Fold the light curve at the period and drop the points into `m` equal
//! phase bins that tile the orbit exactly, as in Kovács, Zucker & Mazeh
//! (2002), from the phase of the first point. `m` is the fewest bins no wider
//! than `min(D) / oversample`, so a bin is `b = P / m`. Each duration becomes
//! a whole number of bins, `k = round(D / b)`, and a box is `k` neighbouring
//! bins. Running totals of the weight and the weighted flux, over the bins
//! twice round the orbit, give any box's sums in and out of transit as
//! differences of two totals, so every start bin and every duration costs
//! one step each.
//!
//! astropy instead uses bins of exactly `min(D) / oversample`, so its last bin
//! is a part bin, and a box through it is shorter than `k b`. Here every box
//! is exactly `k b` long, wherever it starts, so it holds a set of points that
//! a box of that duration can hold. astropy also pads only `oversample` bins
//! past the end, on top of the last real bin; going twice round the orbit, a
//! box of any length can start in any bin, and no bin is lost.

use crate::{
    BlsError, BoxFit,
    box_fit::{Weighted, offset_from_transit},
    sums::{Sum, between},
};

/// The best box of a binned search at one period.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the search reads this; not written yet")
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BinnedBox {
    /// Its fit, from the bins' sums.
    pub(crate) fit: BoxFit,
    /// The bin width, days: the period over the number of bins.
    pub(crate) bin: f64,
    /// The first bin, counted from the bin holding the first point.
    pub(crate) start: usize,
    /// How many bins long.
    pub(crate) bins: usize,
    /// Which of the durations searched.
    pub(crate) duration: usize,
    /// The mid-transit time, BTJD: the middle of the box.
    pub(crate) transit_time: f64,
}

/// The best box at `period` days over `durations` days, with bins no wider
/// than `min(durations) / oversample`. The first time is the bins' phase
/// zero.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the search calls this; not written yet")
)]
pub(crate) fn best_binned(
    times: &[f64],
    weighted: &Weighted,
    period: f64,
    durations: &[f64],
    oversample: usize,
) -> Result<Option<BinnedBox>, BlsError> {
    let n = times.len();
    let shortest = durations.iter().copied().fold(f64::INFINITY, f64::min);
    let m = ((period / (shortest / oversample as f64)).ceil() as usize).max(1);
    let bin = period / m as f64;

    let mut weight = vec![Sum::default(); m];
    let mut flux = vec![Sum::default(); m];
    let mut count = vec![0_usize; m];
    for (k, &t) in times.iter().enumerate() {
        let mut phase = offset_from_transit(t, times[0], period);
        if phase < 0.0 {
            phase += period;
        }
        // A phase that rounds up to m bins belongs in the last.
        let index = ((phase / bin) as usize).min(m - 1);
        weight[index].add(weighted.weights[k]);
        flux[index].add(weighted.centred[k]);
        count[index] += 1;
    }

    // Running totals over the bins twice round.
    let mut weight_totals = Vec::with_capacity(2 * m + 1);
    let mut flux_totals = Vec::with_capacity(2 * m + 1);
    let mut count_totals = Vec::with_capacity(2 * m + 1);
    let (mut w, mut f, mut c) = (Sum::default(), Sum::default(), 0);
    weight_totals.push(w);
    flux_totals.push(f);
    count_totals.push(c);
    for k in 0..2 * m {
        w.add(weight[k % m].value());
        f.add(flux[k % m].value());
        c += count[k % m];
        weight_totals.push(w);
        flux_totals.push(f);
        count_totals.push(c);
    }

    let mut best: Option<BinnedBox> = None;
    for (duration, &d) in durations.iter().enumerate() {
        let bins = (d / bin).round() as usize;
        // A box of every bin holds every point. (A box of no bins holds none,
        // and is skipped below.)
        if bins >= m {
            continue;
        }
        for start in 0..m {
            let end = start + bins;
            let n_in = count_totals[end] - count_totals[start];
            if n_in == 0 || n_in == n {
                continue;
            }
            let inside = [
                between(weight_totals[start], weight_totals[end]),
                between(flux_totals[start], flux_totals[end]),
            ];
            let outside = [
                between(weight_totals[end], weight_totals[start + m]),
                between(flux_totals[end], flux_totals[start + m]),
            ];
            let fit = BoxFit::from_sums(weighted.mean, inside, outside, n_in)?;
            if fit.depth > 0.0 && best.is_none_or(|b| fit.delta_chi2 > b.fit.delta_chi2) {
                best = Some(BinnedBox {
                    fit,
                    bin,
                    start,
                    bins,
                    duration,
                    transit_time: times[0] + (start as f64 + 0.5 * bins as f64) * bin,
                });
            }
        }
    }
    Ok(best)
}
