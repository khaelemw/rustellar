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
    BlsError, BoxFit, Objective,
    box_fit::{Weighted, offset_from_transit},
    totals::{Orbit, Total, box_sums},
};

/// The best box of a binned search at one period.
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
    /// The middle of the box, days after the first time: the middle of the
    /// range of times at which a box `bins` bins long holds the same points.
    pub(crate) centre: f64,
}

/// The bins and running totals of a binned search, kept between periods so
/// that a search allocates nothing per period.
#[derive(Debug, Clone, Default)]
pub(crate) struct Scratch {
    bins: Vec<Bin>,
    totals: Vec<Total>,
    scores: Vec<f64>,
}

impl Scratch {
    /// Room for up to `bins` bins, so no period that needs no more grows it.
    pub(crate) fn with_capacity(bins: usize) -> Self {
        Self {
            bins: Vec::with_capacity(bins),
            totals: Vec::with_capacity(2 * bins + 1),
            scores: Vec::with_capacity(bins),
        }
    }
}

/// The best box by `objective` at `period` days over `durations` days, with
/// bins no wider than `min(durations) / oversample`. The first time is the
/// bins' phase zero.
pub(crate) fn best_binned(
    times: &[f64],
    weighted: &Weighted,
    period: f64,
    durations: &[f64],
    oversample: usize,
    objective: Objective,
    scratch: &mut Scratch,
) -> Result<Option<BinnedBox>, BlsError> {
    let m = (bin_count(period, durations, oversample) as usize).max(1);
    let bin = period / m as f64;

    let Scratch {
        bins,
        totals,
        scores,
    } = scratch;
    bins.clear();
    bins.resize(m, Bin::default());
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
        slot.low = slot.low.min(phase);
        slot.high = slot.high.max(phase);
    }

    // Running totals over the bins twice round.
    totals.clear();
    let mut total = Total::default();
    totals.push(total);
    for b in bins.iter().chain(bins.iter()) {
        total.weight.add(b.weight);
        total.flux.add(b.flux);
        totals.push(total);
    }

    // The best box's score, length, start and duration; its fit is built
    // once, at the end.
    let mut best: Option<(f64, usize, usize, usize)> = None;
    let whole = Orbit::new(&totals[0], &totals[m], objective);
    for (duration, &d) in durations.iter().enumerate() {
        let length = (d / bin).round() as usize;
        // A box of every bin holds every point. (A box of no bins holds none,
        // and is skipped below.)
        if length >= m {
            continue;
        }
        // Every start's score at once, then the first of the highest.
        scores.clear();
        scores.resize(m, 0.0);
        whole.statistics(totals, length, scores)?;
        for (start, &score) in scores.iter().enumerate() {
            if score > best.map_or(0.0, |b| b.0) {
                best = Some((score, length, start, duration));
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
        centre: centre(bins, start, length, bin, period),
    }))
}

/// The middle of the range of times at which a box `length` bins long holds
/// the points of bins `start..start + length`, in days after the first time.
///
/// Its left edge `a` holds them, with the box open at both ends, when
/// `previous <= a < first` and `last < a + D <= next`, `first` and `last`
/// being the box's first and last points and `previous` and `next` its
/// neighbours outside. The bins themselves hold a point on the box's left
/// edge, so the box at the bins' own edges might not: the first point always
/// sits on the left edge of a box that starts at bin 0. The range is never
/// empty: `previous` and `last - D` are below the bins' left edge, and
/// `first` and `next - D` not below it. A box with a dip holds some points and
/// leaves some out, so all four exist.
fn centre(bins: &[Bin], start: usize, length: usize, bin: f64, period: f64) -> f64 {
    let m = bins.len();
    let span = length as f64 * bin;
    // Bin `j` of the bins continued round the orbit both ways, and how far
    // round, in days.
    let at = |j: usize| (&bins[j % m], (j / m) as f64 * period);
    let full = |j: &usize| at(*j).0.count > 0;
    let low = |j: usize| at(j).0.low + at(j).1;
    let high = |j: usize| at(j).0.high + at(j).1;
    // Indices start one orbit on, so the bins before the box are not below 0.
    let start = start + m;
    let inside = start..start + length;
    let first = low(inside.clone().find(full).expect("a point in transit"));
    let last = high(inside.rev().find(full).expect("a point in transit"));
    let next = low((start + length..start + m)
        .find(full)
        .expect("a point out of transit"));
    let previous = high(
        (start + length - m..start)
            .rev()
            .find(full)
            .expect("a point out of transit"),
    );
    let lower = previous.max(last - span);
    let upper = first.min(next - span);
    0.5 * (lower + upper) + 0.5 * span - period
}

/// The number of bins at `period` days: the fewest no wider than
/// `min(durations) / oversample`.
pub(crate) fn bin_count(period: f64, durations: &[f64], oversample: usize) -> f64 {
    let shortest = durations.iter().copied().fold(f64::INFINITY, f64::min);
    (period / (shortest / oversample as f64)).ceil()
}

/// One bin's plain sums of the weights and centred fluxes, and its number of
/// points.
#[derive(Debug, Clone, Copy)]
struct Bin {
    weight: f64,
    flux: f64,
    count: usize,
    /// The lowest and highest phase in the bin.
    low: f64,
    high: f64,
}

impl Default for Bin {
    fn default() -> Self {
        Self {
            weight: 0.0,
            flux: 0.0,
            count: 0,
            low: f64::INFINITY,
            high: f64::NEG_INFINITY,
        }
    }
}
