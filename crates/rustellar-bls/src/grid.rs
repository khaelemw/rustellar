//! The trial durations and periods a search runs over, checked once when
//! built so the search itself cannot fail.
//!
//! Two differences from astropy, on purpose: astropy silently flips a negative
//! duration or period to positive, and checks that the longest duration is
//! shorter than the shortest period only when the search runs. Here a
//! negative value is an error, and the duration check runs when a search is
//! set up ([`check_durations_fit`]).

use rustellar_core::{Periods, TimeDelta};

use crate::BlsError;

/// The trial durations of the box: at least one, each greater than zero.
///
/// They can be in any order, and repeats are allowed, as in astropy.
///
/// # Examples
///
/// ```
/// use rustellar_bls::Durations;
///
/// let durations = Durations::from_days(&[0.05, 0.1, 0.2])?;
/// assert_eq!(durations.min().as_days(), 0.05);
/// assert_eq!(durations.max().as_days(), 0.2);
/// # Ok::<(), rustellar_bls::BlsError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Durations(Vec<TimeDelta>);

impl Durations {
    /// Checks and keeps the durations.
    ///
    /// # Errors
    ///
    /// - [`BlsError::NoDurations`] if `durations` is empty.
    /// - [`BlsError::DurationNotPositive`] if any is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::{BlsError, Durations};
    /// use rustellar_core::{Periods, TimeDelta};
    ///
    /// let durations = Durations::new(vec![TimeDelta::from_hours(3.0)?])?;
    /// assert_eq!(durations.as_slice().len(), 1);
    /// assert_eq!(Durations::new(vec![]), Err(BlsError::NoDurations));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(durations: Vec<TimeDelta>) -> Result<Self, BlsError> {
        check_positive(
            &durations,
            BlsError::NoDurations,
            BlsError::DurationNotPositive,
        )?;
        Ok(Self(durations))
    }

    /// Durations from a list of days, as most light-curve code holds them.
    ///
    /// # Errors
    ///
    /// - [`BlsError::NonFinite`] (`name: "duration"`) if any is NaN or
    ///   infinite. Every value is checked for this first.
    /// - Otherwise as [`Durations::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::{BlsError, Durations};
    ///
    /// assert!(Durations::from_days(&[0.05, 0.1]).is_ok());
    /// assert_eq!(
    ///     Durations::from_days(&[0.1, f64::NAN]),
    ///     Err(BlsError::NonFinite { name: "duration" })
    /// );
    /// ```
    pub fn from_days(days: &[f64]) -> Result<Self, BlsError> {
        Self::new(to_time_deltas(days, "duration")?)
    }

    /// The durations, in the order given.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::Durations;
    ///
    /// let durations = Durations::from_days(&[0.2, 0.1])?;
    /// assert_eq!(durations.as_slice()[0].as_days(), 0.2);
    /// # Ok::<(), rustellar_bls::BlsError>(())
    /// ```
    #[must_use]
    pub fn as_slice(&self) -> &[TimeDelta] {
        &self.0
    }

    /// The shortest duration, which sets the phase resolution of the search.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::Durations;
    ///
    /// assert_eq!(Durations::from_days(&[0.2, 0.1])?.min().as_days(), 0.1);
    /// # Ok::<(), rustellar_bls::BlsError>(())
    /// ```
    #[must_use]
    pub fn min(&self) -> TimeDelta {
        extreme(&self.0, |a, b| b < a)
    }

    /// The longest duration, which must be shorter than the shortest period.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::Durations;
    ///
    /// assert_eq!(Durations::from_days(&[0.2, 0.1])?.max().as_days(), 0.2);
    /// # Ok::<(), rustellar_bls::BlsError>(())
    /// ```
    #[must_use]
    pub fn max(&self) -> TimeDelta {
        extreme(&self.0, |a, b| b > a)
    }
}

/// Checks that every box fits in every orbit: the longest duration must be
/// shorter than the shortest period. A box as long as the orbit would cover
/// every point, leaving no out-of-transit level to compare with.
///
/// # Errors
///
/// [`BlsError::DurationNotShorterThanPeriod`] if the longest duration is at
/// least the shortest period.
pub(crate) fn check_durations_fit(
    periods: &Periods,
    durations: &Durations,
) -> Result<(), BlsError> {
    if durations.max() >= periods.min() {
        return Err(BlsError::DurationNotShorterThanPeriod);
    }
    Ok(())
}

// A set period in days, which must be greater than zero.
pub(crate) fn positive_days(period: TimeDelta) -> Result<f64, BlsError> {
    let days = period.as_days();
    if days <= 0.0 {
        return Err(BlsError::PeriodNotPositive);
    }
    Ok(days)
}

// Days to time deltas, failing on the first NaN or infinity.
fn to_time_deltas(days: &[f64], name: &'static str) -> Result<Vec<TimeDelta>, BlsError> {
    days.iter()
        .map(|&d| TimeDelta::from_days(d).map_err(|_| BlsError::NonFinite { name }))
        .collect()
}

fn check_positive(
    values: &[TimeDelta],
    empty: BlsError,
    not_positive: BlsError,
) -> Result<(), BlsError> {
    if values.is_empty() {
        return Err(empty);
    }
    if values.iter().any(|value| value.as_days() <= 0.0) {
        return Err(not_positive);
    }
    Ok(())
}

// The value that `replaces` every other: `TimeDelta` is `PartialOrd` only, so
// `Iterator::min` does not apply. Callers hold a non-empty list.
fn extreme(values: &[TimeDelta], replaces: fn(TimeDelta, TimeDelta) -> bool) -> TimeDelta {
    values
        .iter()
        .copied()
        .reduce(|best, value| if replaces(best, value) { value } else { best })
        .expect("checked non-empty when built")
}
