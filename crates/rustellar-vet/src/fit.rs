use rustellar_core::LightCurve;

use crate::shape::Profile;
use crate::sum::Sum;
use crate::{Candidate, Shape, VetError};

/// A transit's depth fitted at a fixed period, mid-transit time and duration.
///
/// The model is `baseline * (1 - depth * shape)`, the shape 1 at
/// mid-transit and 0 outside the transit ([`Shape`]), fitted by weighted
/// least squares (weights `1 / flux_err^2`) in its two parameters. The
/// errors are the fit's standard errors: from the covariance of the two
/// parameters, the depth's through its ratio to the baseline. They are
/// right for white noise with the errors given; correlated noise needs more
/// (later phases).
///
/// The depth is a fraction of the baseline, positive for a dip: 1e-3 is a
/// dimming of 0.1% at mid-transit (for a box, throughout the transit).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Fit {
    /// The fractional dimming at mid-transit.
    pub depth: f64,
    /// Its standard error.
    pub depth_err: f64,
    /// The flux out of transit.
    pub baseline: f64,
    /// Its standard error.
    pub baseline_err: f64,
    /// The chi-squared of the fit.
    pub chi2: f64,
    /// How much lower it is than a flat line's: the depth's significance
    /// squared, `(depth / depth_err)^2` to first order.
    pub delta_chi2: f64,
    /// The points where the shape is above 0.
    pub points_in_transit: usize,
}

impl Fit {
    /// The depth's significance: depth over its error.
    #[must_use]
    pub fn snr(&self) -> f64 {
        self.depth / self.depth_err
    }
}

/// The depth of `candidate`'s transit in `lc`, for `shape`.
///
/// # Errors
///
/// - [`VetError::NoPointsInTransit`] if no point falls where the shape is
///   above 0, or [`VetError::NoPointsOutOfTransit`] if none falls outside:
///   the depth or the baseline would be unknown.
/// - [`VetError::DurationTooLongForShape`] for a limb-darkened shape whose
///   duration is more than half the period.
/// - [`VetError::BaselineNotPositive`] if the fitted baseline is zero or
///   negative, so no fractional depth exists.
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, LightCurve, Shape, TimeDelta, fit_depth};
///
/// // A 0.5% box dip, 0.1 days long, every 3 days, every 30 minutes.
/// let time: Vec<f64> = (0..576).map(|i| f64::from(i) / 48.0).collect();
/// let flux = time
///     .iter()
///     .map(|&t| if ((t - 0.75 + 1.5).rem_euclid(3.0) - 1.5).abs() < 0.05 { 0.995 } else { 1.0 })
///     .collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 576])?;
///
/// let candidate = Candidate::new(
///     TimeDelta::from_days(3.0)?,
///     Epoch::from_btjd(0.75)?,
///     TimeDelta::from_days(0.1)?,
/// )?;
/// let fit = fit_depth(&lc, &candidate, Shape::BOX)?;
/// assert!((fit.depth - 0.005).abs() < 1e-15);
/// assert_eq!(fit.points_in_transit, 20);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn fit_depth(lc: &LightCurve, candidate: &Candidate, shape: Shape) -> Result<Fit, VetError> {
    let period = candidate.period().as_days();
    let profile = shape.profile(period, candidate.duration().as_days())?;
    let shapes = shapes(lc, candidate, &profile);
    solve(lc.flux(), lc.flux_err(), &shapes)
}

/// The shape at each point of `lc` for `candidate`.
pub(crate) fn shapes(lc: &LightCurve, candidate: &Candidate, profile: &Profile) -> Vec<f64> {
    let period = candidate.period().as_days();
    let t0 = candidate.transit_time().as_btjd();
    lc.time()
        .iter()
        .map(|t| profile.at(phase(t.as_btjd() - t0, period)))
        .collect()
}

/// `dt` days from a mid-transit time, as days from the nearest one: in
/// `[-period / 2, period / 2)`.
pub(crate) fn phase(dt: f64, period: f64) -> f64 {
    let half = 0.5 * period;
    (dt + half).rem_euclid(period) - half
}

/// The weighted least-squares fit of `flux = alpha + beta * shape`, by
/// centred sums (two passes, each compensated), then `depth = -beta /
/// alpha`.
pub(crate) fn solve(flux: &[f64], flux_err: &[f64], shapes: &[f64]) -> Result<Fit, VetError> {
    let points_in_transit = shapes.iter().filter(|&&s| s > 0.0).count();
    if points_in_transit == 0 {
        return Err(VetError::NoPointsInTransit);
    }
    if points_in_transit == shapes.len() {
        return Err(VetError::NoPointsOutOfTransit);
    }
    let weights: Vec<f64> = flux_err.iter().map(|e| 1.0 / (e * e)).collect();
    let (mut w, mut ws, mut wy) = (Sum::default(), Sum::default(), Sum::default());
    for ((&wi, &s), &y) in weights.iter().zip(shapes).zip(flux) {
        w.add(wi);
        ws.add(wi * s);
        wy.add(wi * y);
    }
    let (w, s_mean, y_mean) = (w.get(), ws.get() / w.get(), wy.get() / w.get());
    let (mut sss, mut ssy, mut syy) = (Sum::default(), Sum::default(), Sum::default());
    for ((&wi, &s), &y) in weights.iter().zip(shapes).zip(flux) {
        let (ds, dy) = (s - s_mean, y - y_mean);
        sss.add(wi * ds * ds);
        ssy.add(wi * ds * dy);
        syy.add(wi * dy * dy);
    }
    let (sss, ssy, syy) = (sss.get(), ssy.get(), syy.get());
    let beta = ssy / sss;
    let alpha = y_mean - beta * s_mean;
    if alpha <= 0.0 {
        return Err(VetError::BaselineNotPositive);
    }
    let var_beta = 1.0 / sss;
    let var_alpha = 1.0 / w + s_mean * s_mean / sss;
    let cov = -s_mean / sss;
    let depth = -beta / alpha;
    // The delta method on depth = -beta / alpha.
    let (d_beta, d_alpha) = (-1.0 / alpha, beta / (alpha * alpha));
    let var_depth =
        d_beta * d_beta * var_beta + d_alpha * d_alpha * var_alpha + 2.0 * d_beta * d_alpha * cov;
    let delta_chi2 = beta * ssy;
    Ok(Fit {
        depth,
        depth_err: var_depth.sqrt(),
        baseline: alpha,
        baseline_err: var_alpha.sqrt(),
        chi2: (syy - delta_chi2).max(0.0),
        delta_chi2,
        points_in_transit,
    })
}
