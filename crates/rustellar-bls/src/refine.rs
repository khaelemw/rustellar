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
//!    first point out;
//! 3. between the best period and halfway to its neighbours on the grid,
//!    over the same durations: eight periods across that stretch, then eight
//!    across the step about the best of them, and so on, until a step moves
//!    the last point by less than the time between points. A slightly wrong
//!    period moves the transits far from the middle of the data, and a grid
//!    fine enough not to is too slow to search everywhere. A period is only
//!    taken if its box is strictly better, so the box is never worse than
//!    on the grid.
//!
//! The power spectrum itself stays binned: the exact search fits noise
//! better than the binned one does, so refining only the peaks would raise
//! them against the rest of the spectrum and inflate any detection statistic
//! measured from it.

use crate::{BlsError, BoxFit, Objective, box_fit::Weighted, exact::Folded, parallel::map};

/// A fold's best box over the durations: which duration, its middle and its
/// fit, as [`Folded::best_of`] gives it.
pub(crate) type BestOf = Result<Option<(usize, f64, BoxFit)>, BlsError>;
/// A fold's best box over a range of durations: its middle, its duration
/// and its fit, as [`Folded::best_between`] gives it.
pub(crate) type BestBetween = Result<Option<(f64, f64, BoxFit)>, BlsError>;

/// Another way to search folds, many at once, to the same results as
/// [`Folded`]'s own searches; `None` where a fold is to be searched on the
/// CPU.
pub(crate) trait Folds: Sync {
    /// At each period, [`Folded::best_of`] over the durations.
    fn best_of(
        &self,
        periods: &[f64],
        durations: &[f64],
        objective: Objective,
    ) -> Vec<Option<BestOf>>;

    /// For each `[period, shortest, longest, beat]`,
    /// [`Folded::best_between`].
    fn best_between(&self, jobs: &[[f64; 4]], objective: Objective) -> Vec<Option<BestBetween>>;
}

/// A peak of the binned power, refined by the exact search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Refined {
    /// The index of the peak's period.
    pub(crate) peak: usize,
    /// The index of the best period near it on the grid.
    pub(crate) period: usize,
    /// The best period itself, days: within half a step of the grid's.
    pub(crate) days: f64,
    /// The duration, days.
    pub(crate) duration: f64,
    /// The mid-transit time, days after the first time.
    pub(crate) centre: f64,
    /// The box's fit.
    pub(crate) fit: BoxFit,
    /// The objective of the best box at the grid's periods (step 2), by
    /// which peaks are compared: step 3 tries more periods at some peaks
    /// than at others, so its boxes would favour those.
    pub(crate) grid: f64,
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
/// between the best one's neighbours, then at periods between the best one
/// and halfway to its neighbours: the best box of each, in the peaks' order,
/// each the best by `objective`. Peaks with no box with a dip are left out.
///
/// In each step every period near every peak is a job of its own, spread
/// over `threads` threads; the outcomes are taken in order, so the result is
/// the same on any number.
#[cfg(test)]
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
    refine_with(
        times, weighted, periods, durations, peaks, reach, objective, threads, None,
    )
}

/// As [`refine`], the folds searched by `folds` where it takes them (to the
/// same result).
#[expect(clippy::too_many_arguments, reason = "the search's settings")]
pub(crate) fn refine_with(
    times: &[f64],
    weighted: &Weighted,
    periods: &[f64],
    durations: &[f64],
    peaks: &[usize],
    reach: usize,
    objective: Objective,
    threads: usize,
    folds: Option<&dyn Folds>,
) -> Result<Vec<Refined>, BlsError> {
    // Each fold's search by `folds`, or else on the CPU.
    let best_of = |at: &[f64]| -> Vec<BestOf> {
        let mut found = folds.map_or_else(
            || (0..at.len()).map(|_| None).collect(),
            |f| f.best_of(at, durations, objective),
        );
        let todo: Vec<usize> = (0..at.len()).filter(|&k| found[k].is_none()).collect();
        let searched = map(&todo, threads, &|&k| {
            Folded::new(times, weighted, at[k]).best_of(weighted, durations, objective)
        });
        for (k, result) in todo.into_iter().zip(searched) {
            found[k] = Some(result);
        }
        found.into_iter().map(|f| f.expect("searched")).collect()
    };
    let best_between = |jobs: &[[f64; 4]]| -> Vec<BestBetween> {
        let mut found = folds.map_or_else(
            || (0..jobs.len()).map(|_| None).collect(),
            |f| f.best_between(jobs, objective),
        );
        let todo: Vec<usize> = (0..jobs.len()).filter(|&k| found[k].is_none()).collect();
        let searched = map(&todo, threads, &|&k| {
            let [p, shortest, longest, beat] = jobs[k];
            Folded::new(times, weighted, p)
                .best_between(weighted, shortest, longest, objective, beat)
        });
        for (k, result) in todo.into_iter().zip(searched) {
            found[k] = Some(result);
        }
        found.into_iter().map(|f| f.expect("searched")).collect()
    };
    let near = |peak: usize| peak.saturating_sub(reach)..(peak + reach + 1).min(periods.len());

    // Step 1: each fold's best box over the durations. The folds are not
    // kept: step 2 folds again, so at most one fold a thread is held (a fold
    // is about 70 bytes a point).
    let jobs: Vec<f64> = peaks
        .iter()
        .flat_map(|&peak| near(peak))
        .map(|period| periods[period])
        .collect();
    let folds = best_of(&jobs)
        .into_iter()
        .map(|grid| Ok(grid?.map(|(duration, _, fit)| (duration, objective.of(&fit)))))
        .collect::<Result<Vec<Option<(usize, f64)>>, BlsError>>()?;

    // Step 2: at each period near a peak, every duration between the
    // neighbours on the grid of that peak's best duration (and shorter than
    // the period). A job is the peak, the period and the range.
    let mut jobs: Vec<(usize, usize, f64, f64, f64)> = Vec::new();
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
        let Some((duration, score)) = grid else {
            continue;
        };
        // At the period of that box, every duration in the range below holds
        // its run too, so the peak's best here scores at least as much: only
        // a box beating it can be the peak's best.
        let beat = objective.beat(score);
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
            jobs.push((peak, period, below, above, beat));
        }
    }
    let searches: Vec<[f64; 4]> = jobs
        .iter()
        .map(|&(_, period, below, above, beat)| [periods[period], below, above, beat])
        .collect();
    let boxes =
        best_between(&searches)
            .into_iter()
            .zip(&jobs)
            .map(|(found, &(peak, period, _, _, _))| {
                Ok(found?.map(|(centre, duration, fit)| Refined {
                    peak,
                    period,
                    days: periods[period],
                    duration,
                    centre,
                    fit,
                    grid: objective.of(&fit),
                }))
            });

    // Each peak's best, the first of equal ones, with its duration range. A
    // peak's jobs are together.
    let mut refined: Vec<(Refined, f64, f64)> = Vec::with_capacity(peaks.len());
    for (found, &(_, _, below, above, _)) in boxes.zip(&jobs) {
        let Some(found) = found? else {
            continue;
        };
        match refined.last_mut() {
            Some(last) if last.0.peak == found.peak => {
                if objective.of(&found.fit) > objective.of(&last.0.fit) {
                    *last = (found, below, above);
                }
            }
            _ => refined.push((found, below, above)),
        }
    }

    // Step 3: periods between each best one and halfway to its neighbours.
    // (A box with a dip needs two points or more.)
    if refined.is_empty() {
        return Ok(Vec::new());
    }
    let spacing = spacing(times);
    let span = times[times.len() - 1] - times[0];
    let mut stretches: Vec<(f64, f64)> = refined
        .iter()
        .map(|(r, _, _)| {
            let p = periods[r.period];
            let halfway =
                |k: Option<usize>| k.and_then(|k| periods.get(k)).map_or(p, |&q| 0.5 * (p + q));
            (
                halfway(r.period.checked_sub(1)),
                halfway(Some(r.period + 1)),
            )
        })
        .collect();
    for _ in 0..LEVELS {
        // Each peak still to refine, and its eight periods across its
        // stretch, but for the best so far.
        let mut jobs: Vec<(usize, f64)> = Vec::new();
        for (k, (&(low, high), (r, _, above))) in stretches.iter().zip(&refined).enumerate() {
            let step = (high - low) / f64::from(STEPS);
            if span * step < spacing * r.days {
                continue;
            }
            for j in 0..=STEPS {
                let p = low + step * f64::from(j);
                if p != r.days && *above < p {
                    jobs.push((k, p));
                }
            }
        }
        if jobs.is_empty() {
            break;
        }
        // Only a box beating the best so far is kept.
        let searches: Vec<[f64; 4]> = jobs
            .iter()
            .map(|&(k, p)| {
                let (best, below, above) = refined[k];
                [p, below, above, objective.beat(objective.of(&best.fit))]
            })
            .collect();
        let boxes = best_between(&searches);
        for (found, &(k, p)) in boxes.into_iter().zip(&jobs) {
            let Some((centre, duration, fit)) = found? else {
                continue;
            };
            let best = &mut refined[k].0;
            if objective.of(&fit) > objective.of(&best.fit) {
                *best = Refined {
                    days: p,
                    duration,
                    centre,
                    fit,
                    ..*best
                };
            }
        }
        // The next stretch: a step either side of the best period.
        for (k, stretch) in stretches.iter_mut().enumerate() {
            let step = (stretch.1 - stretch.0) / f64::from(STEPS);
            let p = refined[k].0.days;
            *stretch = ((p - step).max(stretch.0), (p + step).min(stretch.1));
        }
    }
    Ok(refined.into_iter().map(|(r, _, _)| r).collect())
}

/// Periods across each stretch in step 3, and the most times it narrows.
const STEPS: u32 = 8;
const LEVELS: usize = 12;

/// The median time between consecutive points, days, of two or more points
/// (strictly increasing): a step in period that moves the last point by
/// less than this cannot change which points a box holds by much.
fn spacing(times: &[f64]) -> f64 {
    let mut gaps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    let middle = gaps.len() / 2;
    *gaps.select_nth_unstable_by(middle, f64::total_cmp).1
}
