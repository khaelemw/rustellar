use rustellar_core::{Epoch, LightCurve, TimeDelta};

use crate::{Candidate, Fit, Shape, VetError, fit_depth};

/// The fit at each period near a candidate's, and the best.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PeriodSearch {
    /// The candidate at the best period and mid-transit time (near the
    /// middle of the light curve), with the candidate's duration and the
    /// fitted depth.
    pub best: Candidate,
    /// The fit there.
    pub fit: Fit,
    /// Each period tried, in days, in order.
    pub periods: Vec<f64>,
    /// The drop in chi-squared at each, at its best mid-transit time: NaN
    /// where no dip (depth above 0) could be fitted.
    pub delta_chi2: Vec<f64>,
}

/// The candidate's period searched within `width` of it (a fraction: 0.02
/// for 2%), each period pivoting about the candidate's mid-transit time,
/// that time searched within half a duration, and the duration held: the
/// period and time that lower the chi-squared most. The result's time is
/// the transit nearest the middle of the light curve, where it is best
/// known.
///
/// The periods step by `duration * period / (4 * span)`, so from one to the
/// next the transits at either end of the light curve move by an eighth of
/// a duration; the times by an eighth of a duration, nine of them. The cost
/// grows with the span over the duration; a width of a few percent on years
/// of data means tens of thousands of fits.
///
/// Useful on an alias's row of [`aliases`](crate::aliases): a search's
/// period is exact for the alias it found, but its ratios to the others are
/// only near-exact.
///
/// # Errors
///
/// - [`VetError::WidthOutOfRange`] unless `0 < width <= 0.5`.
/// - [`VetError::NoPointsInTransit`] if no period gives a dip, and as
///   [`fit_depth`] at the best.
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, LightCurve, Shape, TimeDelta, search_period};
///
/// // A 0.3% box dip every 3.01 days; the candidate at 3 days.
/// let time: Vec<f64> = (0..3888).map(|i| f64::from(i) / 144.0).collect();
/// let flux = time
///     .iter()
///     .map(|&t| {
///         let dip = ((t - 0.75 + 1.505).rem_euclid(3.01) - 1.505).abs() < 0.05;
///         1.0 - if dip { 3e-3 } else { 0.0 }
///     })
///     .collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 3888])?;
/// let candidate = Candidate::new(
///     TimeDelta::from_days(3.0)?,
///     Epoch::from_btjd(0.75)?,
///     TimeDelta::from_days(0.1)?,
/// )?;
/// let found = search_period(&lc, &candidate, Shape::BOX, 0.01)?;
/// assert!((found.best.period().as_days() - 3.01).abs() < 0.002);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn search_period(
    lc: &LightCurve,
    candidate: &Candidate,
    shape: Shape,
    width: f64,
) -> Result<PeriodSearch, VetError> {
    if !(width > 0.0 && width <= 0.5) {
        return Err(VetError::WidthOutOfRange);
    }
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    let time = lc.time();
    let (first, last) = (time[0].as_btjd(), time[time.len() - 1].as_btjd());
    // The candidate's transit nearest the middle, by number.
    let t0 = candidate.transit_time().as_btjd();
    let n = ((0.5 * (first + last) - t0) / period).round();
    let span = (last - first).max(duration);
    let step = duration * period / (4.0 * span);
    #[expect(clippy::cast_possible_truncation, reason = "a count of steps")]
    let steps = (width * period / step).ceil() as i64;
    let at = |p: f64, t: f64| {
        Candidate::new(
            TimeDelta::from_days(p).expect("finite"),
            Epoch::from_btjd(t).expect("finite"),
            candidate.duration(),
        )
    };
    let (mut periods, mut delta_chi2) = (vec![], vec![]);
    let mut best: Option<(f64, f64, f64)> = None;
    for k in -steps..=steps {
        #[expect(clippy::cast_precision_loss, reason = "a step number")]
        let p = period + k as f64 * step;
        let mut top = f64::NAN;
        for j in -4..=4 {
            let t = t0 + n * p + f64::from(j) * duration / 8.0;
            let Some(gain) = at(p, t)
                .and_then(|c| fit_depth(lc, &c, shape))
                .ok()
                .filter(|f| f.depth > 0.0)
                .map(|f| f.delta_chi2)
            else {
                continue;
            };
            // `max` passes over the NaN of no fit yet.
            top = top.max(gain);
            if best.is_none_or(|(g, _, _)| gain > g) {
                best = Some((gain, p, t));
            }
        }
        periods.push(p);
        delta_chi2.push(top);
    }
    let (_, p, t) = best.ok_or(VetError::NoPointsInTransit)?;
    let best = at(p, t)?;
    let fit = fit_depth(lc, &best, shape)?;
    Ok(PeriodSearch {
        best: best.with_depth(fit.depth)?,
        fit,
        periods,
        delta_chi2,
    })
}
