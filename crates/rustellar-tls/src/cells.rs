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
//! the cells either side of its centre, interpolated linearly: `T / P`
//! terms for a light curve spanning `T`, whatever the number of points.
//!
//! The trial times are a whole number of cells apart, so a transit falls
//! the same fraction of a cell from the cells for every trial time: each
//! transit in turn adds its share to all the trial times, reading the cells
//! in order. The cost per period and duration is about the trial times
//! times the transits, `steps * T / D`, against about `steps * N` for the
//! fold: less by the points in one transit, `N D / T`.

use std::num::NonZeroUsize;
use std::thread;

#[cfg(feature = "cuda")]
use crate::cuda::{Gpu, Loaded};
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
    /// Where the cells of `duration` are for points at `time`,
    /// `per_duration` to a duration: the first cell's time, the time
    /// between cells, and how many.
    pub(crate) fn grid(time: &[f64], duration: f64, per_duration: usize) -> (f64, f64, usize) {
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
        (start, step, count)
    }

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
        let (start, step, count) = Self::grid(time, duration, per_duration);
        let half = 0.5 * duration;
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
}

/// Cells per trial mid-transit time's spacing: each transit's centre
/// interpolated between cells a quarter of that apart, which loses about as
/// little of the gain as the fold's exact sums.
pub(crate) const CELLS_PER_STEP: usize = 4;

/// Where one period's trial mid-transit times are, from the light curve's
/// first time: `count` of them, `delta` cells (`spacing` days) apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Trials {
    pub(crate) delta: usize,
    pub(crate) spacing: f64,
    pub(crate) count: usize,
}

/// Each trial time's sums, reused from one period to the next.
#[derive(Debug, Default)]
struct Scratch {
    rs: Vec<f64>,
    ss: Vec<f64>,
}

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
    /// The GPU to search the periods on, if any.
    #[cfg(feature = "cuda")]
    pub(crate) gpu: Option<&'a Gpu>,
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
    #[cfg(test)]
    pub(crate) fn durations_at(&self, period: f64) -> impl Iterator<Item = f64> + '_ {
        self.durations
            .iter()
            .copied()
            .filter(move |&d| self.allows(period, d))
    }

    /// Where one duration's trial mid-transit times are at `period`: a
    /// whole number of cells `delta` apart, tiling the period, `D / steps`
    /// apart (`cells_per_step` cells), or further when the points are
    /// sparser in phase than that, as far as `D / MIN_STEPS` (the fold's
    /// rule).
    pub(crate) fn trials(&self, step: f64, period: f64) -> Trials {
        #[expect(clippy::cast_precision_loss, reason = "a count")]
        let points = self.points.time.len() as f64;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a positive count of cells"
        )]
        let delta = ((period / points / step) as usize)
            .min(self.cells_per_step * self.settings.steps / MIN_STEPS)
            .max(self.cells_per_step);
        #[expect(clippy::cast_precision_loss, reason = "a small count")]
        let spacing = delta as f64 * step;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a positive count of trial times"
        )]
        let count = (period / spacing).ceil() as usize;
        Trials {
            delta,
            spacing,
            count,
        }
    }

    /// The times a transit's centre may be at, within half a duration of
    /// the light curve.
    pub(crate) fn window(&self, duration: f64) -> (f64, f64) {
        let half = 0.5 * duration;
        (self.origin - half, self.origin + self.span + half)
    }

    /// Each trial time's sums at `period` into `scratch`: the transits
    /// taken one at a time, oldest first, each adding its share to every
    /// trial time it is in. The trial times read cells `delta` apart, with
    /// the same fraction between cells for them all.
    fn sums(&self, cells: &Cells, period: f64, trials: Trials, scratch: &mut Scratch) {
        let Trials {
            delta,
            spacing,
            count,
        } = trials;
        let (rs, ss) = (&mut scratch.rs, &mut scratch.ss);
        rs.clear();
        rs.resize(count, 0.0);
        ss.clear();
        ss.resize(count, 0.0);
        let (from, to) = self.window(cells.duration);
        let mut orbit = -1.0;
        loop {
            let base = period.mul_add(orbit, self.origin);
            if base > to {
                break;
            }
            orbit += 1.0;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to the trial times"
            )]
            let (first, last) = (
                ((from - base) / spacing).ceil().max(0.0) as usize,
                (((to - base) / spacing).floor() as usize).min(count - 1),
            );
            // Cell index of the transit at trial time k: u + k delta.
            let u = (base - cells.start) / cells.step;
            let whole = u.floor();
            let f = u - whole;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a cell index, less at most one period's cells"
            )]
            let whole = whole as isize;
            for k in first..=last {
                #[expect(
                    clippy::cast_possible_wrap,
                    clippy::cast_sign_loss,
                    reason = "inside the cells for the trial times kept"
                )]
                let j = (whole + (k * delta) as isize) as usize;
                rs[k] += f.mul_add(cells.rs[j + 1] - cells.rs[j], cells.rs[j]);
                ss[k] += f.mul_add(cells.ss[j + 1] - cells.ss[j], cells.ss[j]);
            }
        }
    }

    /// One duration's trial times at `period`, the period's best kept.
    fn search(&self, cells: &Cells, period: f64, running: &mut Running, scratch: &mut Scratch) {
        let trials = self.trials(cells.step, period);
        self.sums(cells, period, trials, scratch);
        for (k, (&r, &s)) in scratch.rs.iter().zip(&scratch.ss).enumerate() {
            if r > 0.0 {
                let depth = r / s;
                let gain = r * depth;
                if depth > self.settings.min_depth && gain > running.gain {
                    self.keep(running, cells.duration, trials, k, gain, depth);
                }
            }
        }
    }

    /// Trial time `k` of `trials` as the period's best.
    fn keep(
        &self,
        running: &mut Running,
        duration: f64,
        trials: Trials,
        k: usize,
        gain: f64,
        depth: f64,
    ) {
        #[expect(clippy::cast_precision_loss, reason = "a trial index")]
        let t0 = trials.spacing.mul_add(k as f64, self.origin);
        *running = Running {
            best: PeriodBest {
                period: running.best.period,
                chi2: self.flat_chi2 - gain,
                duration,
                t0,
                depth,
            },
            gain,
        };
    }

    /// The best fit at each of `periods`, on `threads` threads: the same
    /// results, to the bit, for any number. One duration at a time, in
    /// increasing duration (so each period's best is the one its own order
    /// of durations finds): its cells made, every period searched with
    /// them, and dropped. With a GPU, the periods are searched there, to
    /// the same results; a GPU that fails leaves the rest to the CPU.
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
        // The points and the table on the GPU, once.
        #[cfg(feature = "cuda")]
        let mut gpu = self
            .gpu
            .and_then(|gpu| gpu.load(self.points, self.table).ok());
        let chunk = periods.len().div_ceil(threads.get()).max(1);
        for &duration in self.durations {
            if !periods.iter().any(|&p| self.allows(p, duration)) {
                continue;
            }
            #[cfg(feature = "cuda")]
            if let Some(loaded) = &gpu {
                if self.on_gpu(loaded, duration, periods, &mut running).is_ok() {
                    continue;
                }
                gpu = None;
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
                        let mut scratch = Scratch::default();
                        for (&period, running) in part.iter().zip(best) {
                            if self.allows(period, duration) {
                                self.search(cells, period, running, &mut scratch);
                            }
                        }
                    });
                }
            });
        }
        running.into_iter().map(|r| r.best).collect()
    }

    /// One duration's cells made and its periods searched on the GPU, each
    /// period's best kept as [`Search::search`] keeps it.
    #[cfg(feature = "cuda")]
    fn on_gpu(
        &self,
        loaded: &Loaded<'_>,
        duration: f64,
        periods: &[f64],
        running: &mut [Running],
    ) -> Result<(), String> {
        let per_duration = self.cells_per_step * self.settings.steps;
        let (_, step, _) = Cells::grid(&self.points.time, duration, per_duration);
        let (index, jobs): (Vec<usize>, Vec<(f64, Trials)>) = periods
            .iter()
            .enumerate()
            .filter(|&(_, &p)| self.allows(p, duration))
            .map(|(i, &p)| (i, (p, self.trials(step, p))))
            .unzip();
        let found = loaded.search(self, duration, per_duration, &jobs)?;
        for ((i, (_, trials)), found) in index.into_iter().zip(jobs).zip(found) {
            if let Some((k, gain, depth)) = found {
                if gain > running[i].gain {
                    self.keep(&mut running[i], duration, trials, k, gain, depth);
                }
            }
        }
        Ok(())
    }
}
