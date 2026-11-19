//! Standard errors by bootstrap: the best fit's model plus its residuals
//! resampled, fitted again many times; the errors are the fits' spread.
//!
//! Where the signal is weak the chi-squared is lumpy (points crossing the
//! template's edges) and its curvature's errors are too small, by about 25%
//! at signal to noise 12, with long tails. The bootstrap measures how the
//! fit itself scatters. The residuals are resampled in blocks a transit
//! long (a moving block bootstrap), so noise correlated over hours is
//! resampled as it is, not as white noise.
//!
//! Each resample is drawn from its own seeded generator, so the errors are
//! the same on every run and any number of threads.

use std::num::NonZeroUsize;
use std::thread;

use crate::fit::{Points, Position, fit};
use crate::refine::{Reach, refine};
use crate::template::Shape;

/// How far, in the curvature's standard errors, a resample's fit may move
/// from the best fit.
const REACH: f64 = 8.0;

/// The seed of the first resample's generator.
const SEED: u64 = 0x005E_ED0F_B007;

/// The bootstrap's standard errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Spread {
    pub(crate) period: f64,
    pub(crate) t0: f64,
    pub(crate) duration: f64,
    pub(crate) depth: f64,
}

/// SplitMix64: small, fast, and enough for drawing blocks.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A whole number in `0..n` (`n` well below 2^32, so the bias is nil).
    fn below(&mut self, n: usize) -> usize {
        #[expect(clippy::cast_possible_truncation, reason = "below n, a usize")]
        let i = (self.next() % n as u64) as usize;
        i
    }
}

/// The sample standard deviation of `values` (at least two).
fn spread(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let (mut n, mut sum) = (0.0, 0.0);
    for v in values.clone() {
        n += 1.0;
        sum += v;
    }
    let mean = sum / n;
    let squares: f64 = values.map(|v| (v - mean) * (v - mean)).sum();
    (squares / (n - 1.0)).sqrt()
}

/// The standard errors of the fit of `template` at `best` with `depth` to
/// `points`, by `count` resamples (at least 2), refitted with `sampled`
/// from the best fit within `REACH` curvature errors `sigma` (period, t0,
/// duration), on `threads` threads.
#[expect(clippy::too_many_arguments, reason = "the fit and how to resample it")]
pub(crate) fn bootstrap(
    points: &Points,
    sampled: &(impl Shape + Sync),
    template: &(impl Shape + Sync),
    best: Position,
    depth: f64,
    sigma: [f64; 3],
    count: usize,
    threads: NonZeroUsize,
) -> Spread {
    let model: Vec<f64> = points
        .time
        .iter()
        .map(|&t| depth * template.shape(best.x(t)))
        .collect();
    let residuals: Vec<f64> = points
        .residual
        .iter()
        .zip(&model)
        .map(|(r, m)| r - m)
        .collect();
    let n = points.time.len();
    let span = points.time[n - 1] - points.time[0];
    // A transit's worth of points, on average.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a count of points, positive and below n"
    )]
    let block = ((n as f64 * best.duration / span).round() as usize).clamp(1, n);
    let reach = Reach {
        t0: REACH * sigma[1],
        duration: 1.0 + REACH * sigma[2] / best.duration,
        period: REACH * sigma[0],
    };
    // One resample's fit: (period, t0, duration, depth).
    let resample = |index: u64| -> [f64; 4] {
        let mut random = Random(SEED.wrapping_add(index));
        let mut residual = Vec::with_capacity(n);
        while residual.len() < n {
            let start = random.below(n - block + 1);
            let take = block.min(n - residual.len());
            residual.extend(
                model[residual.len()..residual.len() + take]
                    .iter()
                    .zip(&residuals[start..start + take])
                    .map(|(m, r)| m + r),
            );
        }
        let again = Points {
            time: points.time.clone(),
            residual,
            weight: points.weight.clone(),
        };
        let p = refine(&again, sampled, best, reach).position;
        let depth = fit(&again, template, p).depth;
        // The first mid-transit time may be a period on from the best's.
        let t0 = p.t0 - ((p.t0 - best.t0) / p.period).round() * p.period;
        [p.period, t0, p.duration, depth]
    };
    let chunk = count.div_ceil(threads.get()).max(1);
    let fits: Vec<[f64; 4]> = thread::scope(|scope| {
        let handles: Vec<_> = (0..count)
            .collect::<Vec<_>>()
            .chunks(chunk)
            .map(|part| {
                let part = part.to_vec();
                let resample = &resample;
                scope.spawn(move || {
                    part.into_iter()
                        .map(|i| resample(i as u64))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a bootstrap thread panicked"))
            .collect()
    });
    let column = |i: usize| spread(fits.iter().map(move |f| f[i]));
    Spread {
        period: column(0),
        t0: column(1),
        duration: column(2),
        depth: column(3),
    }
}
