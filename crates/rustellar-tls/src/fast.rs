//! The fast search: at each period, every duration fitted at a grid of
//! mid-transit times, with the template looked up in a table.
//!
//! For one period `P` and duration `D`, the trial mid-transit times are `B`
//! evenly spaced times tiling the period, `h = P / B` apart, at least
//! `steps` per duration. Each point lies within half a duration of about
//! `steps` of them, and adds its terms `w r s` and `w s²` to each, with `s`
//! the template at its own time from that trial (not at a bin's centre).
//! So the cost per duration is about `N steps`, whatever `B`, and the sums
//! at each trial time are those of the exact fit there, up to the table's
//! interpolation. There are no more trial times than points, on average,
//! unless that leaves fewer than [`MIN_STEPS`] per duration, so a short
//! transit in a sparse light curve costs `N` times the points it holds,
//! like TLS's template laid on the samples.

use std::num::NonZeroUsize;
use std::thread;

use crate::durations::{Constants, Stars};
use crate::fit::Points;
use crate::template::Template;

/// The template sampled at `M + 1` evenly spaced points across `[-1/2,
/// 1/2]`, linearly interpolated between them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Table {
    values: Vec<f64>,
}

impl Table {
    /// `samples` intervals across the template.
    pub(crate) fn new(template: &Template, samples: usize) -> Self {
        #[expect(clippy::cast_precision_loss, reason = "a sample index")]
        let values = (0..=samples)
            .map(|j| template.shape(j as f64 / samples as f64 - 0.5))
            .collect();
        Self { values }
    }

    /// The samples, `intervals + 1` of them from `x = -1/2` to `1/2`.
    #[cfg(feature = "cuda")]
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }

    /// The number of intervals, as a double.
    pub(crate) fn intervals(&self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "a sample count")]
        let intervals = (self.values.len() - 1) as f64;
        intervals
    }

    /// The template at `x`, interpolated: 0 for `|x| >= 1/2`.
    #[cfg(test)]
    pub(crate) fn at(&self, x: f64) -> f64 {
        self.at_sample((x + 0.5) * self.intervals())
    }

    /// The template at `u` samples from `x = -1/2`, interpolated.
    pub(crate) fn at_sample(&self, u: f64) -> f64 {
        if !(u > 0.0 && u < self.intervals()) {
            return 0.0;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "0 < u < intervals"
        )]
        let j = u as usize;
        #[expect(clippy::cast_precision_loss, reason = "a sample index")]
        let f = u - j as f64;
        f.mul_add(self.values[j + 1] - self.values[j], self.values[j])
    }
}

/// The fewest trial mid-transit times per duration, however few points
/// there are.
pub(crate) const MIN_STEPS: usize = 10;

/// How the fast search is run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Settings {
    /// Trial mid-transit times per duration, at least.
    pub(crate) steps: usize,
    /// The smallest depth a fit may have to count (TLS's
    /// `transit_depth_min`).
    pub(crate) min_depth: f64,
    /// The stars the durations at each period are limited to.
    pub(crate) stars: Stars,
    pub(crate) constants: Constants,
}

/// The best fit found at one period.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PeriodBest {
    pub(crate) period: f64,
    /// The chi-squared: the flat model's less the best gain.
    pub(crate) chi2: f64,
    /// The best fit's duration and mid-transit time, in days, and depth.
    /// The depth is 0, and the others the first trial's, when nothing fits.
    pub(crate) duration: f64,
    pub(crate) t0: f64,
    pub(crate) depth: f64,
}

/// What one period's search needs besides the period.
pub(crate) struct Search<'a> {
    pub(crate) points: &'a Points,
    pub(crate) table: &'a Table,
    /// The trial durations as fractions of the period, increasing.
    pub(crate) fractions: &'a [f64],
    pub(crate) settings: Settings,
    /// The flat model's chi-squared, `sum w r²`.
    pub(crate) flat_chi2: f64,
    /// The light curve's first time and its span, in days.
    pub(crate) origin: f64,
    pub(crate) span: f64,
}

/// Buffers reused from one period to the next.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    /// The points' offsets from the first time, folded into `[0, P)`, with
    /// their indices, sorted by offset.
    folded: Vec<(f64, usize)>,
    /// The sweep, in order of offset: the points near the fold's end a
    /// period early, all of them, and those near its start a period late;
    /// their offsets, `w r` and `w`.
    offset: Vec<f64>,
    weighted_residual: Vec<f64>,
    weight: Vec<f64>,
}

impl Search<'_> {
    /// The durations searched at `period`: those of the grid between the
    /// shortest and longest transit the stars allow there. The longest is
    /// widened by `(n + 1) / n` for `n` the number of periods in the span,
    /// as TLS does, since a transit at each end of the light curve makes
    /// the span look one period longer than the transits it holds.
    pub(crate) fn durations(&self, period: f64) -> impl Iterator<Item = f64> + '_ {
        let settings = self.settings;
        let shortest = settings.stars.shortest(settings.constants, period);
        let orbits = self.span / period;
        let longest = settings.stars.longest(settings.constants, period) * (orbits + 1.0) / orbits;
        self.fractions
            .iter()
            .filter(move |&&q| q >= shortest && q <= longest && q < 1.0)
            .map(move |&q| q * period)
    }

    /// The best fit at `period`.
    ///
    /// The points are sorted by phase, and each trial time's sums taken
    /// over the run of points within half a duration of it, a run that only
    /// moves forward from one trial time to the next. A transit across the
    /// fold's ends is covered by sweeping the points within the longest
    /// half duration of the end a period early, before the rest, and those
    /// near the start a period late, after.
    pub(crate) fn period(&self, period: f64, scratch: &mut Scratch) -> PeriodBest {
        let points = self.points;
        let mut best = PeriodBest {
            period,
            chi2: self.flat_chi2,
            duration: self.fractions.first().map_or(0.0, |q| q * period),
            t0: self.origin,
            depth: 0.0,
        };
        let Some(longest) = self.durations(period).last() else {
            return best;
        };

        scratch.folded.clear();
        scratch.folded.extend(
            points
                .time
                .iter()
                .enumerate()
                .map(|(i, &t)| (((t - self.origin) / period).rem_euclid(1.0) * period, i)),
        );
        scratch
            .folded
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let folded = &scratch.folded;
        let reach = 0.5 * longest;
        let tail = folded.partition_point(|p| p.0 <= period - reach);
        let head = folded.partition_point(|p| p.0 < reach);
        let sweep = folded[tail..]
            .iter()
            .map(|&(o, i)| (o - period, i))
            .chain(folded.iter().copied())
            .chain(folded[..head].iter().map(|&(o, i)| (o + period, i)));
        let (offset, weighted_residual, weight) = (
            &mut scratch.offset,
            &mut scratch.weighted_residual,
            &mut scratch.weight,
        );
        offset.clear();
        weighted_residual.clear();
        weight.clear();
        for (o, i) in sweep {
            offset.push(o);
            weighted_residual.push(points.weight[i] * points.residual[i]);
            weight.push(points.weight[i]);
        }

        let mut best_gain = 0.0;
        let samples = self.table.intervals();
        for duration in self.durations(period) {
            let count = {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss,
                    reason = "a positive count of trial times"
                )]
                let per_duration =
                    |steps: usize| ((period / duration) * steps as f64).ceil() as usize;
                // No more trial times than points, on average, unless that
                // leaves fewer than `MIN_STEPS` per duration: closer than
                // the points' own spacing in phase they gain little.
                per_duration(self.settings.steps)
                    .min(points.time.len().max(per_duration(MIN_STEPS)))
            };
            #[expect(clippy::cast_precision_loss, reason = "a count")]
            let step = period / count as f64;
            let half = 0.5 * duration;
            let to_sample = samples / duration;
            let (mut lo, mut hi) = (0, 0);
            for k in 0..count {
                #[expect(clippy::cast_precision_loss, reason = "a trial index")]
                let centre = k as f64 * step;
                while lo < offset.len() && offset[lo] <= centre - half {
                    lo += 1;
                }
                hi = hi.max(lo);
                while hi < offset.len() && offset[hi] < centre + half {
                    hi += 1;
                }
                // Table samples from x = -1/2: (o - centre) / D + 1/2.
                let shift = 0.5f64.mul_add(samples, -centre * to_sample);
                let (mut rs, mut ss) = (0.0, 0.0);
                for i in lo..hi {
                    let s = self.table.at_sample(offset[i].mul_add(to_sample, shift));
                    rs += weighted_residual[i] * s;
                    ss += weight[i] * s * s;
                }
                if rs > 0.0 {
                    let depth = rs / ss;
                    let gain = rs * depth;
                    if depth > self.settings.min_depth && gain > best_gain {
                        best_gain = gain;
                        best = PeriodBest {
                            period,
                            chi2: self.flat_chi2 - gain,
                            duration,
                            t0: self.origin + centre,
                            depth,
                        };
                    }
                }
            }
        }
        best
    }

    /// The best fit at each of `periods`, on `threads` threads: the same
    /// results, to the bit, for any number.
    pub(crate) fn periods(&self, periods: &[f64], threads: NonZeroUsize) -> Vec<PeriodBest> {
        let chunk = periods.len().div_ceil(threads.get()).max(1);
        thread::scope(|scope| {
            let handles: Vec<_> = periods
                .chunks(chunk)
                .map(|part| {
                    scope.spawn(move || {
                        let mut scratch = Scratch::default();
                        part.iter()
                            .map(|&p| self.period(p, &mut scratch))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("a search thread panicked"))
                .collect()
        })
    }
}
