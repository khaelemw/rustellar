//! The fast search: at each period, every duration fitted at a grid of
//! mid-transit times, with the template looked up in a table.
//!
//! For one period `P` and duration `D`, the trial mid-transit times are `B`
//! evenly spaced times tiling the period, `h = P / B` apart, at least
//! `steps` per duration. Each point lies within half a duration of about
//! `steps` of them, and adds its terms `w r s` and `w s²` to each, with `s`
//! the template at its own time from that trial (not at a bin's centre).
//! So the cost per duration is about `N steps + B`, and the sums at each
//! trial time are those of the exact fit there, up to the table's
//! interpolation.

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

    /// The template at `x`, interpolated: 0 for `|x| >= 1/2`.
    pub(crate) fn at(&self, x: f64) -> f64 {
        let intervals = self.values.len() - 1;
        #[expect(clippy::cast_precision_loss, reason = "a sample count")]
        let u = (x + 0.5) * intervals as f64;
        if !(u > 0.0 && u < intervals as f64) {
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

/// Sums reused from one duration to the next.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    offset: Vec<f64>,
    rs: Vec<f64>,
    ss: Vec<f64>,
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
    pub(crate) fn period(&self, period: f64, scratch: &mut Scratch) -> PeriodBest {
        let points = self.points;
        scratch.offset.clear();
        scratch.offset.extend(
            points
                .time
                .iter()
                .map(|&t| ((t - self.origin) / period).rem_euclid(1.0) * period),
        );
        let mut best = PeriodBest {
            period,
            chi2: self.flat_chi2,
            duration: self.fractions.first().map_or(0.0, |q| q * period),
            t0: self.origin,
            depth: 0.0,
        };
        let mut best_gain = 0.0;
        for duration in self.durations(period) {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss,
                reason = "a positive count of trial times"
            )]
            let count = ((period / duration) * self.settings.steps as f64).ceil() as usize;
            #[expect(clippy::cast_precision_loss, reason = "a count")]
            let step = period / count as f64;
            scratch.rs.clear();
            scratch.rs.resize(count, 0.0);
            scratch.ss.clear();
            scratch.ss.resize(count, 0.0);
            let half = 0.5 * duration;
            for (i, &d) in scratch.offset.iter().enumerate() {
                let (w, r) = (points.weight[i], points.residual[i]);
                // Trial times k step with |d - k step| < D/2.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "within a period's trial times of 0"
                )]
                let first = ((d - half) / step).floor() as i64 + 1;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "within a period's trial times of 0"
                )]
                let last = ((d + half) / step).ceil() as i64 - 1;
                for k in first..=last {
                    #[expect(clippy::cast_precision_loss, reason = "a trial index")]
                    let s = self.table.at((d - k as f64 * step) / duration);
                    if s > 0.0 {
                        #[expect(
                            clippy::cast_possible_truncation,
                            clippy::cast_possible_wrap,
                            reason = "reduced modulo the count"
                        )]
                        let slot = k.rem_euclid(count as i64) as usize;
                        scratch.rs[slot] += w * r * s;
                        scratch.ss[slot] += w * s * s;
                    }
                }
            }
            for (k, (&rs, &ss)) in scratch.rs.iter().zip(&scratch.ss).enumerate() {
                if rs > 0.0 {
                    let depth = rs / ss;
                    let gain = rs * depth;
                    if depth > self.settings.min_depth && gain > best_gain {
                        best_gain = gain;
                        #[expect(clippy::cast_precision_loss, reason = "a trial index")]
                        let t0 = self.origin + k as f64 * step;
                        best = PeriodBest {
                            period,
                            chi2: self.flat_chi2 - gain,
                            duration,
                            t0,
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
