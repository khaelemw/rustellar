//! Exact refinement of the peaks of a binned search.
//!
//! The binned search gives the power at every period, each box rounded to
//! the bins. The highest peaks are then searched again exactly, in two steps:
//!
//! 1. at their own period and the periods either side, over every
//!    mid-transit time and each of the durations: the best duration;
//! 2. at those periods again, over every duration between the neighbours of
//!    the best one, so the box is not rounded to the duration grid either. A box
//!    shorter than the transit can sit anywhere inside it, and one longer
//!    anywhere round it, with much the same fit: the mid-transit time is
//!    then uncertain by up to half the difference, and the depth is diluted.
//!    Each edge is put halfway between the last point in transit and the
//!    first point out.
//!
//! The power spectrum itself stays binned: the exact search fits noise
//! better than the binned one does, so refining only the peaks would raise
//! them against the rest of the spectrum and inflate any detection statistic
//! measured from it.

use rustellar_core::LightCurve;

use crate::{BlsError, BoxFit, box_fit::Weighted, exact::Folded};

/// A peak of the binned power, refined by the exact search.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the search reads this; not written yet")
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Refined {
    /// The index of the peak's period.
    pub(crate) peak: usize,
    /// The index of the best period near it.
    pub(crate) period: usize,
    /// The duration, days.
    pub(crate) duration: f64,
    /// The mid-transit time, BTJD.
    pub(crate) transit_time: f64,
    /// The box's fit.
    pub(crate) fit: BoxFit,
}

/// The indices of the `count` highest local maxima of `power`, highest first
/// (the first of equal ones first). A local maximum is above the point
/// before it and not below the point after it, so a flat top counts once.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the search calls this; not written yet")
)]
pub(crate) fn peaks(power: &[f64], count: usize) -> Vec<usize> {
    let n = power.len();
    let mut peaks: Vec<usize> = (0..n)
        .filter(|&i| {
            (i == 0 || power[i] > power[i - 1]) && (i + 1 == n || power[i] >= power[i + 1])
        })
        .collect();
    peaks.sort_by(|&a, &b| power[b].total_cmp(&power[a]).then(a.cmp(&b)));
    peaks.truncate(count);
    peaks
}

/// Each peak searched exactly at its period and the `reach` periods either
/// side, over the durations, then at those periods over every duration
/// between the best one's neighbours: the best box of each, in the peaks'
/// order. Peaks with no box with a dip are left out.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the search calls this; not written yet")
)]
pub(crate) fn refine(
    lc: &LightCurve,
    weighted: &Weighted,
    periods: &[f64],
    durations: &[f64],
    peaks: &[usize],
    reach: usize,
) -> Result<Vec<Refined>, BlsError> {
    let first = lc.time()[0].as_btjd();
    let mut refined = Vec::with_capacity(peaks.len());
    for &peak in peaks {
        let near = peak.saturating_sub(reach)..(peak + reach + 1).min(periods.len());
        let folds: Vec<Folded> = near
            .clone()
            .map(|k| Folded::new(lc, weighted, periods[k]))
            .collect();
        // The best duration on the grid, over the periods near the peak.
        let mut grid: Option<(usize, f64)> = None;
        for folded in &folds {
            if let Some((duration, _, fit)) = folded.best_of(weighted, durations)? {
                if grid.is_none_or(|g| fit.delta_chi2 > g.1) {
                    grid = Some((duration, fit.delta_chi2));
                }
            }
        }
        let Some((duration, _)) = grid else {
            continue;
        };
        // Every duration between its neighbours on the grid, at each period
        // (and shorter than it).
        let d = durations[duration];
        let below = durations
            .iter()
            .copied()
            .filter(|&x| x < d)
            .reduce(f64::max)
            .unwrap_or(d);
        let mut best: Option<Refined> = None;
        for (period, folded) in near.zip(&folds) {
            let p = periods[period];
            if d >= p {
                continue;
            }
            let above = durations
                .iter()
                .copied()
                .filter(|&x| x > d && x < p)
                .reduce(f64::min)
                .unwrap_or(d);
            if let Some((centre, duration, fit)) = folded.best_between(weighted, below, above)? {
                if best.is_none_or(|b| fit.delta_chi2 > b.fit.delta_chi2) {
                    best = Some(Refined {
                        peak,
                        period,
                        duration,
                        transit_time: first + centre,
                        fit,
                    });
                }
            }
        }
        refined.extend(best);
    }
    Ok(refined)
}
