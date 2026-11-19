//! The best fit's signal to noise: TLS's, for white noise, and the pink
//! one of Pont, Zucker & Queloz (2006), for noise correlated over the
//! transit's duration.
//!
//! TLS's (`stats.snr`) is the mean dimming in transit over the scatter out
//! of it, times the root of the points in transit: `δ / σ_w sqrt(n)` (with
//! `δ` measured here from the mean level out of transit, not 1). That
//! counts every point as independent. Where the noise is correlated over
//! hours (a star's spots and granulation, an instrument's drifts), the `n`
//! points of `N` transits are not, and the pink signal to noise is
//!
//! ```text
//! δ / sqrt(σ_w² / n + σ_r² / N)
//! ```
//!
//! with `σ_r` the correlated part on the transit's timescale: out of
//! transit, the residuals averaged over runs of `m` consecutive points, as
//! many as a transit has, scatter by `σ_D`, and white noise alone would give
//! `σ_w / sqrt(m)`, so `σ_r² = max(σ_D² - σ_w² / m, 0)`. A run ends early at
//! a gap or a transit (more than a duration from its first point) and is
//! then left out.

use crate::fit::{Points, Position, Sum};
use crate::template::Template;

/// The best fit's signal to noise and the noise it is measured against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Snr {
    /// TLS's, for white noise.
    pub(crate) white: f64,
    /// For noise correlated over the duration.
    pub(crate) pink: f64,
    /// The scatter out of transit, `σ_w`.
    pub(crate) white_noise: f64,
    /// The correlated noise on the duration's timescale, `σ_r`.
    pub(crate) red_noise: f64,
    /// The mean dimming in transit, `δ`.
    pub(crate) depth: f64,
}

/// The mean and the standard deviation (with `n - 1`) of `values`.
fn mean_std(values: &[f64]) -> (f64, f64) {
    #[expect(clippy::cast_precision_loss, reason = "a count")]
    let n = values.len() as f64;
    let mut sum = Sum::default();
    values.iter().for_each(|&v| sum.add(v));
    let mean = sum.value() / n;
    let mut squares = Sum::default();
    values
        .iter()
        .for_each(|&v| squares.add((v - mean) * (v - mean)));
    (mean, (squares.value() / (n - 1.0)).sqrt())
}

/// The signal to noise of `template` at `position` in `points`, or `None`
/// with fewer than two points in transit or out of it, or fewer than two
/// full runs out of transit.
pub(crate) fn snr(points: &Points, template: &Template, position: Position) -> Option<Snr> {
    let (mut inside, mut outside) = (Vec::new(), Vec::new());
    let mut transits = 0_usize;
    let mut last = f64::NAN;
    for (&t, &r) in points.time.iter().zip(&points.residual) {
        if template.shape(position.x(t)) > 0.0 {
            inside.push(r);
            let number = ((t - position.t0) / position.period + 0.5).floor();
            if number != last {
                transits += 1;
                last = number;
            }
        } else {
            outside.push((t, r));
        }
    }
    if inside.len() < 2 || outside.len() < 2 {
        return None;
    }
    let per_transit = (inside.len() / transits).max(1);
    // The means of the full runs of `per_transit` points out of transit.
    let mut means = Vec::new();
    let (mut first, mut sum, mut count) = (f64::NEG_INFINITY, 0.0, 0);
    for &(t, r) in &outside {
        if t - first > position.duration {
            (first, sum, count) = (t, 0.0, 0);
        }
        sum += r;
        count += 1;
        if count == per_transit {
            #[expect(clippy::cast_precision_loss, reason = "a count")]
            means.push(sum / per_transit as f64);
            (first, sum, count) = (f64::NEG_INFINITY, 0.0, 0);
        }
    }
    if means.len() < 2 {
        return None;
    }
    let residuals: Vec<f64> = outside.iter().map(|o| o.1).collect();
    // Measured from the level out of transit, where TLS takes 1.
    let (within, _) = mean_std(&inside);
    let (level, white_noise) = mean_std(&residuals);
    let depth = within - level;
    let (_, binned) = mean_std(&means);
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let (n, transits, m) = (inside.len() as f64, transits as f64, per_transit as f64);
    let red_noise = binned
        .mul_add(binned, -white_noise * white_noise / m)
        .max(0.0)
        .sqrt();
    Some(Snr {
        white: depth / white_noise * n.sqrt(),
        pink: depth / (white_noise * white_noise / n + red_noise * red_noise / transits).sqrt(),
        white_noise,
        red_noise,
        depth,
    })
}
