//! Trial periods for a transit search: any list, checked once when built,
//! or Ofir's (2014) optimal grid, dense where transits are short and sparse
//! where they are long, so it needs far fewer periods than an even
//! frequency grid for the same sensitivity.

use std::f64::consts::{PI, TAU};

use crate::{GridError, LightCurve, TimeDelta};

/// The trial periods: at least one, each greater than zero.
///
/// Any list will do, in any order, such as a pipeline's own log-spaced grid
/// or a zoom around a peak.
///
/// # Examples
///
/// ```
/// use rustellar_core::Periods;
///
/// let periods = Periods::from_days(&[0.5, 1.0, 2.0, 4.0])?;
/// assert_eq!(periods.min().as_days(), 0.5);
/// # Ok::<(), rustellar_core::GridError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Periods(Vec<TimeDelta>);

impl Periods {
    /// Checks and keeps the periods.
    ///
    /// # Errors
    ///
    /// - [`GridError::NoPeriods`] if `periods` is empty.
    /// - [`GridError::PeriodNotPositive`] if any is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{GridError, Periods, TimeDelta};
    ///
    /// let periods = Periods::new(vec![TimeDelta::from_days(3.0)?])?;
    /// assert_eq!(periods.as_slice().len(), 1);
    /// assert_eq!(Periods::new(vec![]), Err(GridError::NoPeriods));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(periods: Vec<TimeDelta>) -> Result<Self, GridError> {
        if periods.is_empty() {
            return Err(GridError::NoPeriods);
        }
        if periods.iter().any(|period| period.as_days() <= 0.0) {
            return Err(GridError::PeriodNotPositive);
        }
        Ok(Self(periods))
    }

    /// Periods from a list of days.
    ///
    /// # Errors
    ///
    /// - [`GridError::NonFinite`] (`name: "period"`) if any is NaN or infinite.
    ///   Every value is checked for this first.
    /// - Otherwise as [`Periods::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{GridError, Periods};
    ///
    /// assert!(Periods::from_days(&[1.0, 2.0]).is_ok());
    /// assert_eq!(Periods::from_days(&[0.0]), Err(GridError::PeriodNotPositive));
    /// ```
    pub fn from_days(days: &[f64]) -> Result<Self, GridError> {
        let periods = days
            .iter()
            .map(|&d| TimeDelta::from_days(d).map_err(|_| GridError::NonFinite { name: "period" }))
            .collect::<Result<_, _>>()?;
        Self::new(periods)
    }

    /// The periods, in the order given.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::Periods;
    ///
    /// let periods = Periods::from_days(&[3.0, 1.0])?;
    /// assert_eq!(periods.as_slice()[1].as_days(), 1.0);
    /// # Ok::<(), rustellar_core::GridError>(())
    /// ```
    #[must_use]
    pub fn as_slice(&self) -> &[TimeDelta] {
        &self.0
    }

    /// The shortest period.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::Periods;
    ///
    /// assert_eq!(Periods::from_days(&[3.0, 1.0])?.min().as_days(), 1.0);
    /// # Ok::<(), rustellar_core::GridError>(())
    /// ```
    #[must_use]
    pub fn min(&self) -> TimeDelta {
        self.0
            .iter()
            .copied()
            .reduce(|best, p| if p < best { p } else { best })
            .expect("checked non-empty when built")
    }
}

/// The Sun's mass parameter `G M`, IAU 2015 Resolution B3 nominal value, in
/// m³ s⁻².
const GM_SUN: f64 = 1.327_124_4e20;
/// The Sun's radius, IAU 2015 Resolution B3 nominal value, in metres.
const R_SUN: f64 = 6.957e8;
const SECONDS_PER_DAY: f64 = 86_400.0;
/// The most periods a grid may hold, about 800 MB; far above any real search.
const MAXIMUM_COUNT: f64 = 1e8;

/// Settings for Ofir's (2014) optimal period grid.
///
/// A central transit on a circular orbit lasts a fraction
///
/// ```text
/// q = (2π)^(2/3) / π · R / (G M)^(1/3) · f^(2/3)
/// ```
///
/// of the orbit, at orbital frequency `f` around a star of radius `R` and
/// mass `M`. Over a baseline `S`, a step `df` in frequency moves a transit's
/// phase by `S df`, so to keep it within `q / oversampling_factor` the step is
/// `df = A f^(2/3)` with
///
/// ```text
/// A = (2π)^(2/3) / π · R / (G M)^(1/3) / (S · oversampling_factor)
/// ```
///
/// Solving for `f` gives the grid in closed form, `f(x) = (A/3 · x + C)³` for
/// `x = 1, 2, …`, with `C = f_min^(1/3) − A/3` so that `f(1) = f_min`, up to
/// `N = (f_max^(1/3) − f_min^(1/3) + A/3) · 3 / A`, where `f(N) = f_max`.
///
/// The defaults are those of Ofir and TLS: a Sun-like star, `oversampling_factor`
/// 3, and at least 2 transits. Unset, the longest period is `baseline /
/// minimum_n_transit`, which shows that many transits at any phase, and the
/// shortest is the orbit at the Roche limit, `a = 3 R`:
///
/// ```text
/// f_max = 1 / (2π) · sqrt(G M / (3 R)³)
/// ```
///
/// Differences from TLS (`transitleastsquares.period_grid`), on purpose:
///
/// - The solar `G M` and `R` are the IAU 2015 nominal values; TLS uses
///   `G = 6.673e-11` with a solar mass of `1.989e30` kg, and a radius of
///   `695,508` km, which moves its periods by a few parts in 10,000.
/// - A set period range moves the grid's edges, as in the paper; TLS builds
///   the default grid and cuts it, so a longest period beyond
///   `baseline / minimum_n_transit` does nothing.
/// - The grid stops at `f_max`; TLS adds one more period, below the Roche
///   limit.
/// - Where TLS quietly changes a setting (clamping the star's radius and
///   mass, or falling back to a Sun-like star and dropping the period range
///   when the grid has under 100 periods), this returns the grid it was asked
///   for, or an error.
///
/// # Examples
///
/// ```
/// use rustellar_core::{LightCurve, OfirPeriod};
///
/// // One TESS sector, a Sun-like star.
/// let lc = LightCurve::from_days(vec![1325.293, 1353.177], vec![1.0; 2], vec![1e-3; 2])?;
/// let periods = OfirPeriod::default().periods(&lc)?;
/// assert_eq!(periods.as_slice().len(), 2552);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OfirPeriod {
    /// The star's radius, in solar radii.
    pub stellar_radius: f64,
    /// The star's mass, in solar masses.
    pub stellar_mass: f64,
    /// The shortest period, or the orbit at the Roche limit if `None`.
    pub minimum_period: Option<TimeDelta>,
    /// The longest period, or `baseline / minimum_n_transit` if `None`.
    pub maximum_period: Option<TimeDelta>,
    /// How many steps to take within one transit's phase drift: above 1 is
    /// finer (more periods). Ofir suggests 2 to 5.
    pub oversampling_factor: f64,
    /// The fewest transits a period must show within the baseline, for the
    /// default longest period. At least 2.
    pub minimum_n_transit: u32,
}

impl Default for OfirPeriod {
    fn default() -> Self {
        Self {
            stellar_radius: 1.0,
            stellar_mass: 1.0,
            minimum_period: None,
            maximum_period: None,
            oversampling_factor: 3.0,
            minimum_n_transit: 2,
        }
    }
}

impl OfirPeriod {
    /// The period grid for a light curve, shortest period first.
    ///
    /// The number of periods grows only as the baseline: about 2,500 for a
    /// 28-day light curve and a Sun-like star, where astropy's automatic grid
    /// over the same range needs tens of thousands.
    ///
    /// # Errors
    ///
    /// In this order:
    ///
    /// - [`GridError::TooFewTransits`] if `minimum_n_transit` is below 2.
    /// - [`GridError::NonFinite`] or [`GridError::NotPositive`] (`name`
    ///   `"stellar_radius"`, `"stellar_mass"` or `"oversampling_factor"`)
    ///   for a bad setting.
    /// - [`GridError::PeriodNotPositive`] if a set `minimum_period` or
    ///   `maximum_period` is zero or negative.
    /// - [`GridError::NoBaseline`] if the light curve has a single time.
    /// - [`GridError::PeriodRangeEmpty`] if the shortest period is above the
    ///   longest.
    /// - [`GridError::GridTooLarge`] if the grid would hold more than 100
    ///   million periods.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{GridError, LightCurve, OfirPeriod, TimeDelta};
    ///
    /// let lc = LightCurve::from_days(vec![1325.293, 1353.177], vec![1.0; 2], vec![1e-3; 2])?;
    ///
    /// // The pipeline's range, 0.5 to 15 days.
    /// let range = OfirPeriod {
    ///     minimum_period: Some(TimeDelta::from_days(0.5)?),
    ///     maximum_period: Some(TimeDelta::from_days(15.0)?),
    ///     ..OfirPeriod::default()
    /// };
    /// assert_eq!(range.periods(&lc)?.as_slice().len(), 2835);
    ///
    /// let no_star = OfirPeriod { stellar_mass: 0.0, ..OfirPeriod::default() };
    /// assert_eq!(no_star.periods(&lc), Err(GridError::NotPositive { name: "stellar_mass" }));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn periods(&self, lc: &LightCurve) -> Result<Periods, GridError> {
        if self.minimum_n_transit < 2 {
            return Err(GridError::TooFewTransits);
        }
        let radius = positive(self.stellar_radius, "stellar_radius")?;
        let mass = positive(self.stellar_mass, "stellar_mass")?;
        let oversampling = positive(self.oversampling_factor, "oversampling_factor")?;
        let set_minimum = self.minimum_period.map(positive_days).transpose()?;
        let set_maximum = self.maximum_period.map(positive_days).transpose()?;

        // Time is strictly increasing, so the baseline is last minus first.
        let time = lc.time();
        let baseline = (time[time.len() - 1] - time[0]).as_days();
        if baseline <= 0.0 {
            return Err(GridError::NoBaseline);
        }

        // SI units throughout: metres, seconds, hertz.
        let r = radius * R_SUN;
        let gm = mass * GM_SUN;
        let span = baseline * SECONDS_PER_DAY;

        let minimum_frequency = match set_maximum {
            Some(days) => 1.0 / (days * SECONDS_PER_DAY),
            None => f64::from(self.minimum_n_transit) / span,
        };
        let maximum_frequency = match set_minimum {
            Some(days) => 1.0 / (days * SECONDS_PER_DAY),
            None => {
                let roche = 3.0 * r;
                (gm / (roche * roche * roche)).sqrt() / TAU
            }
        };
        if minimum_frequency > maximum_frequency {
            return Err(GridError::PeriodRangeEmpty);
        }

        // (2π)^(2/3) as the cube root of (2π)², and cube roots by `cbrt`,
        // which is closer to the true root than `powf(1.0 / 3.0)`.
        let a = (TAU * TAU).cbrt() / PI * r / gm.cbrt() / (span * oversampling);
        let c = minimum_frequency.cbrt() - a / 3.0;
        let n_opt = (maximum_frequency.cbrt() - minimum_frequency.cbrt() + a / 3.0) * 3.0 / a;
        if n_opt.is_nan() || n_opt > MAXIMUM_COUNT {
            return Err(GridError::GridTooLarge);
        }

        // At least one: with equal edges, rounding can put n_opt just below 1.
        // The value is finite, in [0, 1e8], so the conversion is exact.
        let count = (n_opt.floor() as usize).max(1);

        // Highest frequency, so shortest period, first. The cube as two
        // products, not `powi`, so it rounds the same on every platform.
        let periods = (1..=count)
            .rev()
            .map(|x| {
                let u = a / 3.0 * x as f64 + c;
                TimeDelta::from_days(1.0 / (u * u * u) / SECONDS_PER_DAY)
                    .expect("frequencies between the two edges, both finite and positive")
            })
            .collect();

        Ok(Periods(periods))
    }
}

fn positive(value: f64, name: &'static str) -> Result<f64, GridError> {
    if !value.is_finite() {
        return Err(GridError::NonFinite { name });
    }
    if value <= 0.0 {
        return Err(GridError::NotPositive { name });
    }
    Ok(value)
}

// A set period in days, which must be greater than zero.
fn positive_days(period: TimeDelta) -> Result<f64, GridError> {
    let days = period.as_days();
    if days <= 0.0 {
        return Err(GridError::PeriodNotPositive);
    }
    Ok(days)
}
