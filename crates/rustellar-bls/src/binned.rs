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
//! Each bin's own sums are plain sums of its few points, rounding at the size
//! of the bin; the running totals over the bins are compensated, so a box's
//! sums are about as accurate as adding up its own points, however many
//! points the light curve has. (astropy's plain running totals round at the
//! size of the whole light curve's sums.)
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
    totals::{Orbit, Total, box_sums},
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
    let shortest = durations.iter().copied().fold(f64::INFINITY, f64::min);
    let m = ((period / (shortest / oversample as f64)).ceil() as usize).max(1);
    let bin = period / m as f64;

    let mut bins = vec![Bin::default(); m];
    for (k, &t) in times.iter().enumerate() {
        let mut phase = offset_from_transit(t, times[0], period);
        if phase < 0.0 {
            phase += period;
        }
        // A phase that rounds up to m bins belongs in the last.
        let slot = &mut bins[((phase / bin) as usize).min(m - 1)];
        slot.weight += weighted.weights[k];
        slot.flux += weighted.centred[k];
        slot.count += 1;
    }

    // Running totals over the bins twice round.
    let mut totals = Vec::with_capacity(2 * m + 1);
    let mut total = Total::default();
    totals.push(total);
    for b in bins.iter().chain(&bins) {
        total.weight.add(b.weight);
        total.flux.add(b.flux);
        totals.push(total);
    }

    // The best box's statistic, length, start and duration; its fit is built
    // once, at the end.
    let mut best: Option<(f64, usize, usize, usize)> = None;
    let whole = Orbit::new(&totals[0], &totals[m]);
    for (duration, &d) in durations.iter().enumerate() {
        let length = (d / bin).round() as usize;
        // A box of every bin holds every point. (A box of no bins holds none,
        // and is skipped below.)
        if length >= m {
            continue;
        }
        // The totals at each box's start, end, and start one orbit on.
        let ends = totals[..m]
            .iter()
            .zip(&totals[length..length + m])
            .zip(&totals[m..2 * m]);
        for (start, ((first, end), wrap)) in ends.enumerate() {
            let delta_chi2 = whole.statistic(first, end, wrap)?;
            if delta_chi2 > best.map_or(0.0, |b| b.0) {
                best = Some((delta_chi2, length, start, duration));
            }
        }
    }
    let Some((_, length, start, duration)) = best else {
        return Ok(None);
    };
    let (first, end) = (&totals[start], &totals[start + length]);
    let n_in = (start..start + length).map(|k| bins[k % m].count).sum();
    let [inside, outside] = box_sums(first, end, &totals[start + m]);
    Ok(Some(BinnedBox {
        fit: BoxFit::from_sums(weighted.mean, inside, outside, n_in)?,
        bin,
        start,
        bins: length,
        duration,
        transit_time: times[0] + (start as f64 + 0.5 * length as f64) * bin,
    }))
}

/// One bin's plain sums of the weights and centred fluxes, and its number of
/// points.
#[derive(Debug, Clone, Copy, Default)]
struct Bin {
    weight: f64,
    flux: f64,
    count: usize,
}
