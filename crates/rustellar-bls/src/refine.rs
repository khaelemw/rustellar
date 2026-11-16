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

use crate::{BlsError, BoxFit, Objective, box_fit::Weighted, exact::Folded, parallel::map};

/// A peak of the binned power, refined by the exact search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Refined {
    /// The index of the peak's period.
    pub(crate) peak: usize,
    /// The index of the best period near it.
    pub(crate) period: usize,
    /// The duration, days.
    pub(crate) duration: f64,
    /// The mid-transit time, days after the first time.
    pub(crate) centre: f64,
    /// The box's fit.
    pub(crate) fit: BoxFit,
}

/// The indices of the `count` highest local maxima of `power` above zero,
/// highest first (the first of equal ones first). A local maximum is above
/// the point before it and not below the point after it, so a flat top
/// counts once.
///
/// Only the highest `count` are kept as it goes, in one allocation of at
/// most `count`, so the work is linear in the periods and the memory does
/// not grow with them.
pub(crate) fn peaks(power: &[f64], count: usize) -> Vec<usize> {
    let n = power.len();
    let mut peaks: Vec<usize> = Vec::with_capacity(count.min(n));
    for i in 0..n {
        let maximum = power[i] > 0.0
            && (i == 0 || power[i] > power[i - 1])
            && (i + 1 == n || power[i] >= power[i + 1]);
        if !maximum {
            continue;
        }
        // After the equal ones, which come earlier.
        let at = peaks.partition_point(|&k| power[k] >= power[i]);
        if at < count {
            if peaks.len() == count {
                peaks.pop();
            }
            peaks.insert(at, i);
        }
    }
    peaks
}

/// Each peak searched exactly at its period and the `reach` periods either
/// side, over the durations, then at those periods over every duration
/// between the best one's neighbours: the best box of each, in the peaks'
/// order, each the best by `objective`. Peaks with no box with a dip are
/// left out.
///
/// In each step every period near every peak is a job of its own, spread
/// over `threads` threads; the outcomes are taken in order, so the result is
/// the same on any number.
#[expect(clippy::too_many_arguments, reason = "the search's settings")]
pub(crate) fn refine(
    times: &[f64],
    weighted: &Weighted,
    periods: &[f64],
    durations: &[f64],
    peaks: &[usize],
    reach: usize,
    objective: Objective,
    threads: usize,
) -> Result<Vec<Refined>, BlsError> {
    let near = |peak: usize| peak.saturating_sub(reach)..(peak + reach + 1).min(periods.len());

    // Step 1: each fold's best box over the durations. The folds are not
    // kept: step 2 folds again, so at most one fold a thread is held (a fold
    // is about 70 bytes a point).
    let jobs: Vec<usize> = peaks.iter().flat_map(|&peak| near(peak)).collect();
    let folds = map(&jobs, threads, &|&period| {
        let folded = Folded::new(times, weighted, periods[period]);
        let grid = folded.best_of(weighted, durations, objective)?;
        Ok(grid.map(|(duration, _, fit)| (duration, objective.of(&fit))))
    })
    .into_iter()
    .collect::<Result<Vec<Option<(usize, f64)>>, BlsError>>()?;

    // Step 2: at each period near a peak, every duration between the
    // neighbours on the grid of that peak's best duration (and shorter than
    // the period). A job is the peak, the period and the range.
    let mut jobs: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut end = 0;
    for &peak in peaks {
        let near = near(peak);
        let here = end..end + near.len();
        end = here.end;
        // The best duration on the grid, over the periods near the peak.
        let mut grid: Option<(usize, f64)> = None;
        for &best in &folds[here.clone()] {
            if let Some((duration, score)) = best {
                if grid.is_none_or(|g| score > g.1) {
                    grid = Some((duration, score));
                }
            }
        }
        let Some((duration, _)) = grid else {
            continue;
        };
        let d = durations[duration];
        let below = durations
            .iter()
            .copied()
            .filter(|&x| x < d)
            .reduce(f64::max)
            .unwrap_or(d);
        for period in near {
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
            jobs.push((peak, period, below, above));
        }
    }
    let boxes = map(&jobs, threads, &|&(peak, period, below, above)| {
        let found = Folded::new(times, weighted, periods[period])
            .best_between(weighted, below, above, objective)?;
        Ok(found.map(|(centre, duration, fit)| Refined {
            peak,
            period,
            duration,
            centre,
            fit,
        }))
    });

    // Each peak's best, the first of equal ones. A peak's jobs are together.
    let mut refined: Vec<Refined> = Vec::with_capacity(peaks.len());
    for found in boxes {
        let Some(found) = found? else {
            continue;
        };
        match refined.last_mut() {
            Some(last) if last.peak == found.peak => {
                if objective.of(&found.fit) > objective.of(&last.fit) {
                    *last = found;
                }
            }
            _ => refined.push(found),
        }
    }
    Ok(refined)
}
