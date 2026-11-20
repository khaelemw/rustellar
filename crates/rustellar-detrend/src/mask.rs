//! The mask of a candidate's transits, so a second detrending leaves them
//! out of every window.

use rustellar_core::{Epoch, LightCurve, TimeDelta};

use crate::DetrendError;

/// True at each point within half `duration` of a mid-transit time `t0 + n
/// period`, for any whole `n`: the points to leave out of every window when
/// detrending again around a candidate, so the trend follows the star and
/// not the transit.
///
/// A point exactly half a duration from a mid-transit time is not masked.
/// Pass a duration a little longer than the transit's own if its period or
/// mid-transit time is uncertain.
///
/// # Errors
///
/// [`DetrendError::NotPositive`] if `period` or `duration` is zero or
/// negative, with `name` `"period"` or `"duration"`.
///
/// # Examples
///
/// ```
/// use rustellar_detrend::{Detrend, Epoch, LightCurve, TimeDelta, transit_mask};
///
/// // Every 10 minutes for 4 days, with a transit every 2 days from day 0.5.
/// let time: Vec<f64> = (0..576).map(|i| f64::from(i) / 144.0).collect();
/// let flux = time
///     .iter()
///     .map(|&t| if ((t - 0.5) / 2.0 + 0.05).rem_euclid(1.0) < 0.1 { 0.99 } else { 1.0 })
///     .collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 576])?;
///
/// let mask = transit_mask(
///     &lc,
///     TimeDelta::from_days(2.0)?,
///     Epoch::from_btjd(0.5)?,
///     TimeDelta::from_days(0.25)?,
/// )?;
/// assert!(mask[72] && mask[360] && !mask[0]);
/// let flat = Detrend::default().flatten(&lc, Some(&mask))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn transit_mask(
    lc: &LightCurve,
    period: TimeDelta,
    t0: Epoch,
    duration: TimeDelta,
) -> Result<Vec<bool>, DetrendError> {
    let (period, half) = (period.as_days(), 0.5 * duration.as_days());
    if period <= 0.0 {
        return Err(DetrendError::NotPositive { name: "period" });
    }
    if half <= 0.0 {
        return Err(DetrendError::NotPositive { name: "duration" });
    }
    Ok(lc
        .time()
        .iter()
        .map(|&t| {
            // The time from the nearest mid-transit time, in periods.
            let phase = (((t - t0).as_days() / period) + 0.5).rem_euclid(1.0) - 0.5;
            phase.abs() * period < half
        })
        .collect())
}
