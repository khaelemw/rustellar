//! TLS's spectra from the chi-squared at each period (Hippke & Heller 2019,
//! section 2.4): the signal residue SR, the signal detection efficiency SDE,
//! and the power spectrum with its slow trend taken off by a running median.
//!
//! The definitions are TLS's (`stats.spectra`), with means and standard
//! deviations from compensated sums.

use crate::fit::Sum;

/// The kernel of the running median, before it is made odd, per unit of
/// the period grid's oversampling factor (TLS's `SDE_MEDIAN_KERNEL_SIZE`).
const MEDIAN_KERNEL: usize = 30;

/// The spectra of a search.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Spectra {
    /// `min(chi2) / chi2`: 1 at the best period, in `(0, 1]`.
    pub(crate) sr: Vec<f64>,
    /// SR less its mean, scaled so that its peak is `sde_raw`.
    pub(crate) power_raw: Vec<f64>,
    /// `power_raw` less its running median, shifted to mean 0 and scaled
    /// so that its peak is `sde`; `power_raw` itself for short spectra.
    pub(crate) power: Vec<f64>,
    /// `(1 - mean(SR)) / std(SR)`.
    pub(crate) sde_raw: f64,
    /// The peak of `power` over its standard deviation.
    pub(crate) sde: f64,
}

fn mean(values: &[f64]) -> f64 {
    let mut sum = Sum::default();
    for &v in values {
        sum.add(v);
    }
    #[expect(clippy::cast_precision_loss, reason = "a count")]
    let n = values.len() as f64;
    sum.value() / n
}

/// The population standard deviation (numpy's default, `ddof = 0`).
fn std(values: &[f64]) -> f64 {
    let m = mean(values);
    let mut sum = Sum::default();
    for &v in values {
        sum.add((v - m) * (v - m));
    }
    #[expect(clippy::cast_precision_loss, reason = "a count")]
    let n = values.len() as f64;
    (sum.value() / n).sqrt()
}

fn max(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// The median of each run of `kernel` values (odd), padded to the input's
/// length with the first and last medians, as TLS's `running_median`: half
/// the missing values (rounded down) in front.
pub(crate) fn running_median(values: &[f64], kernel: usize) -> Vec<f64> {
    debug_assert!(kernel % 2 == 1 && values.len() >= kernel);
    let mut window = Vec::with_capacity(kernel);
    let medians: Vec<f64> = values
        .windows(kernel)
        .map(|run| {
            window.clear();
            window.extend_from_slice(run);
            *window.select_nth_unstable_by(kernel / 2, f64::total_cmp).1
        })
        .collect();
    let missing = values.len() - medians.len();
    let front = missing / 2;
    let mut out = vec![medians[0]; front];
    out.extend_from_slice(&medians);
    out.resize(values.len(), medians[medians.len() - 1]);
    out
}

/// The spectra of the chi-squared at each period, for a period grid with
/// `oversampling` periods per transit duration. `chi2` must hold at least
/// two values, positive and finite, not all equal.
pub(crate) fn spectra(chi2: &[f64], oversampling: usize) -> Spectra {
    let least = chi2.iter().copied().fold(f64::INFINITY, f64::min);
    let sr: Vec<f64> = chi2.iter().map(|&c| least / c).collect();
    let sr_mean = mean(&sr);
    let sde_raw = (1.0 - sr_mean) / std(&sr);
    let shifted: Vec<f64> = sr.iter().map(|&v| v - sr_mean).collect();
    let scale = sde_raw / max(&shifted);
    let power_raw: Vec<f64> = shifted.iter().map(|&v| v * scale).collect();

    let kernel = (oversampling * MEDIAN_KERNEL) | 1;
    let (power, sde) = if power_raw.len() > 2 * kernel {
        let median = running_median(&power_raw, kernel);
        let detrended: Vec<f64> = power_raw.iter().zip(&median).map(|(p, m)| p - m).collect();
        let m = mean(&detrended);
        let centred: Vec<f64> = detrended.iter().map(|&v| v - m).collect();
        let sde = max(&centred) / std(&centred);
        let scale = sde / max(&centred);
        (centred.iter().map(|&v| v * scale).collect(), sde)
    } else {
        (power_raw.clone(), sde_raw)
    };
    Spectra {
        sr,
        power_raw,
        power,
        sde_raw,
        sde,
    }
}

/// Half the width of the peak of `power` at half its height, in the units of
/// `periods` (increasing): from the first period below half the peak on
/// one side to the first on the other. Infinite when either side never
/// drops to half (TLS reads past the start of the array instead, item 9).
pub(crate) fn period_uncertainty(periods: &[f64], power: &[f64]) -> f64 {
    let peak = power
        .iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |top, (i, &p)| if p > top.1 { (i, p) } else { top },
        )
        .0;
    let half = 0.5 * power[peak];
    let upper = (peak + 1..power.len()).find(|&i| power[i] <= half);
    let lower = (0..peak).rev().find(|&i| power[i] <= half);
    match (lower, upper) {
        (Some(lower), Some(upper)) => 0.5 * (periods[upper] - periods[lower]),
        _ => f64::INFINITY,
    }
}
