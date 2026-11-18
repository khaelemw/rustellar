//! The fast search turned round (Foreman-Mackey et al. 2015; Smith et al.
//! 2025): the template's sums are taken once, at a grid of mid-transit times
//! across the light curve, and each period then adds up those of its
//! transits.
//!
//! The fit is additive over transits: the least-squares depth and gain at a
//! position need only `S_rs = sum w r s` and `S_ss = sum w s²` over all the
//! points, and each transit's points add their own part. So for each
//! duration `D` (in days, the same at every period) the light curve is
//! swept once, and the two sums kept with the template's centre at each of
//! a grid of times `D / cells` apart: the cells, about `cells * N` terms in
//! all. A trial position at period `P` then sums, for each of its transits,
//! the cell nearest its centre: `T / P` terms for a light curve spanning
//! `T`, whatever the number of points.
//!
//! Each transit's centre is moved to its nearest cell, by at most
//! `D / (2 cells)`; with twice as many cells per duration as trial times
//! that is half the trial times' own spacing. The cost per period and
//! duration is about the trial times times the transits, `steps * T / D`,
//! against about `steps * N` for the fold: less by the points in one
//! transit, `N D / T`.

use std::num::NonZeroUsize;
use std::thread;

use crate::fast::{MIN_STEPS, PeriodBest, Settings, Table};
use crate::fit::Points;

/// The template's sums with its centre at each of an even grid of times,
/// for one duration.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Cells {
    pub(crate) duration: f64,
    /// The time of the first cell, and the time between cells.
    pub(crate) start: f64,
    pub(crate) step: f64,
    pub(crate) rs: Vec<f64>,
    pub(crate) ss: Vec<f64>,
}

impl Cells {
    /// The cells of `duration` for `points`, `per_duration` to a duration:
    /// from half a duration and a cell before the first time to as much
    /// after the last; made on `threads` threads, the same on any number.
    pub(crate) fn new(
        points: &Points,
        table: &Table,
        duration: f64,
        per_duration: usize,
        threads: NonZeroUsize,
    ) -> Self {
        let time = &points.time;
        let (first, last) = (time[0], time[time.len() - 1]);
        #[expect(clippy::cast_precision_loss, reason = "a small count")]
        let step = duration / per_duration as f64;
        let half = 0.5 * duration;
        let start = first - half - step;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a positive count of cells"
        )]
        let count = ((last + half + step - start) / step).ceil() as usize + 1;
        let (mut rs, mut ss) = (vec![0.0; count], vec![0.0; count]);
        let chunk = count.div_ceil(threads.get());
        thread::scope(|scope| {
            for (part, (rs, ss)) in rs.chunks_mut(chunk).zip(ss.chunks_mut(chunk)).enumerate() {
                scope.spawn(move || {
                    let centre = |m: usize| {
                        #[expect(clippy::cast_precision_loss, reason = "a cell index")]
                        let m = m as f64;
                        step.mul_add(m, start)
                    };
                    let from = part * chunk;
                    let mut lo = time.partition_point(|&t| t <= centre(from) - half);
                    let mut hi = lo;
                    let samples = table.intervals();
                    let to_sample = samples / duration;
                    for (j, (r, s2)) in rs.iter_mut().zip(ss.iter_mut()).enumerate() {
                        let centre = centre(from + j);
                        while lo < time.len() && time[lo] <= centre - half {
                            lo += 1;
                        }
                        hi = hi.max(lo);
                        while hi < time.len() && time[hi] < centre + half {
                            hi += 1;
                        }
                        // Table samples from x = -1/2: (t - centre) / D + 1/2.
                        let shift = 0.5f64.mul_add(samples, -centre * to_sample);
                        for ((&t, &residual), &w) in time[lo..hi]
                            .iter()
                            .zip(&points.residual[lo..hi])
                            .zip(&points.weight[lo..hi])
                        {
                            let s = table.at_sample(t.mul_add(to_sample, shift));
                            *r += w * residual * s;
                            *s2 += w * s * s;
                        }
                    }
                });
            }
        });
        Self {
            duration,
            start,
            step,
            rs,
            ss,
        }
    }

    /// The sums at the cell nearest `t`, which must lie within half a
    /// duration of the light curve.
    fn at(&self, t: f64) -> (f64, f64) {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "t lies within the cells"
        )]
        let m = ((t - self.start) / self.step) as usize;
        #[expect(clippy::cast_precision_loss, reason = "a cell index")]
        let f = (t - self.start) / self.step - m as f64;
        (
            f.mul_add(self.rs[m + 1] - self.rs[m], self.rs[m]),
            f.mul_add(self.ss[m + 1] - self.ss[m], self.ss[m]),
        )
    }
}

/// Cells per trial mid-transit time's spacing: each transit's centre
/// interpolated between cells a quarter of that apart, which loses about as
/// little of the gain as the fold's exact sums.
pub(crate) const CELLS_PER_STEP: usize = 4;

/// A period's best so far, with its gain over the flat model.
#[derive(Debug, Clone, Copy)]
struct Running {
    best: PeriodBest,
    gain: f64,
}

/// What the search needs besides the periods.
pub(crate) struct Search<'a> {
    pub(crate) points: &'a Points,
    pub(crate) table: &'a Table,
    /// The trial durations in days, increasing.
    pub(crate) durations: &'a [f64],
    /// Cells per trial mid-transit time's spacing.
    pub(crate) cells_per_step: usize,
    pub(crate) settings: Settings,
    /// The flat model's chi-squared, `sum w r²`.
    pub(crate) flat_chi2: f64,
    /// The light curve's first time and its span, in days.
    pub(crate) origin: f64,
    pub(crate) span: f64,
}

impl Search<'_> {
    /// Whether `duration` is searched at `period`: between the shortest and
    /// longest transit the stars allow there, the longest widened by
    /// `(n + 1) / n` for `n` the periods in the span (as the fold does), and
    /// shorter than the period.
    pub(crate) fn allows(&self, period: f64, duration: f64) -> bool {
        let settings = self.settings;
        let shortest = settings.stars.shortest(settings.constants, period) * period;
        let orbits = self.span / period;
        let longest =
            settings.stars.longest(settings.constants, period) * period * (orbits + 1.0) / orbits;
        duration >= shortest && duration <= longest && duration < period
    }

    /// The durations searched at `period`, increasing.
    pub(crate) fn durations_at(&self, period: f64) -> impl Iterator<Item = f64> + '_ {
        self.durations
            .iter()
            .copied()
            .filter(move |&d| self.allows(period, d))
    }

    /// The sums at trial phase `phase` (from the first time): each
    /// transit's centre within half a duration of the light curve, one a
    /// period early when the phase is within half a duration of the
    /// period's end.
    fn trial(&self, cells: &Cells, period: f64, phase: f64) -> (f64, f64) {
        let half = 0.5 * cells.duration;
        let end = self.origin + self.span + half;
        let mut centre = if phase >= period - half {
            self.origin + phase - period
        } else {
            self.origin + phase
        };
        let (mut rs, mut ss) = (0.0, 0.0);
        while centre <= end {
            let (r, s) = cells.at(centre);
            rs += r;
            ss += s;
            centre += period;
        }
        (rs, ss)
    }

    /// One duration's trial mid-transit times at `period`, tiling the
    /// period (as many as the fold's), each summing its transits' cells.
    fn search(&self, cells: &Cells, period: f64, running: &mut Running) {
        let duration = cells.duration;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "a positive count of trial times"
        )]
        let per_duration = |steps: usize| ((period / duration) * steps as f64).ceil() as usize;
        let count = per_duration(self.settings.steps)
            .min(self.points.time.len().max(per_duration(MIN_STEPS)));
        #[expect(clippy::cast_precision_loss, reason = "a count")]
        let step = period / count as f64;
        #[expect(clippy::cast_precision_loss, reason = "a trial index")]
        let phase = |k: usize| k as f64 * step;
        let consider = |k: usize, running: &mut Running| {
            let (rs, ss) = self.trial(cells, period, phase(k));
            if rs > 0.0 {
                let depth = rs / ss;
                let gain = rs * depth;
                if depth > self.settings.min_depth && gain > running.gain {
                    *running = Running {
                        best: PeriodBest {
                            period,
                            chi2: self.flat_chi2 - gain,
                            duration,
                            t0: self.origin + phase(k),
                            depth,
                        },
                        gain,
                    };
                }
            }
        };
        for k in 0..count {
            consider(k, running);
        }
    }

    /// The best fit at each of `periods`, on `threads` threads: the same
    /// results, to the bit, for any number. One duration at a time, in
    /// increasing duration (so each period's best is the one its own order
    /// of durations finds): its cells made, every period searched with
    /// them, and dropped.
    pub(crate) fn periods(&self, periods: &[f64], threads: NonZeroUsize) -> Vec<PeriodBest> {
        let mut running: Vec<Running> = periods
            .iter()
            .map(|&period| Running {
                best: PeriodBest {
                    period,
                    chi2: self.flat_chi2,
                    duration: self.durations.first().copied().unwrap_or(0.0),
                    t0: self.origin,
                    depth: 0.0,
                },
                gain: 0.0,
            })
            .collect();
        let chunk = periods.len().div_ceil(threads.get()).max(1);
        for &duration in self.durations {
            if !periods.iter().any(|&p| self.allows(p, duration)) {
                continue;
            }
            let cells = Cells::new(
                self.points,
                self.table,
                duration,
                self.cells_per_step * self.settings.steps,
                threads,
            );
            let cells = &cells;
            thread::scope(|scope| {
                for (part, best) in periods.chunks(chunk).zip(running.chunks_mut(chunk)) {
                    scope.spawn(move || {
                        for (&period, running) in part.iter().zip(best) {
                            if self.allows(period, duration) {
                                self.search(cells, period, running);
                            }
                        }
                    });
                }
            });
        }
        running.into_iter().map(|r| r.best).collect()
    }
}
