use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::TimeError;

const HOURS_PER_DAY: f64 = 24.0;

/// A length of time, such as an orbital period or a transit duration.
///
/// Stored in days. It may be zero or negative.
///
/// The constructors reject NaN and infinity, but the operators do not check
/// their results, the same as for `f64`. Multiplying by NaN or infinity, or
/// adding values near `f64::MAX`, gives a non-finite delta, and dividing by a
/// zero delta gives NaN or infinity. Realistic times never come close to
/// overflowing.
///
/// # Examples
///
/// ```
/// use rustellar_core::TimeDelta;
///
/// let period = TimeDelta::from_hours(36.0)?;
/// assert_eq!(period.as_days(), 1.5);
/// # Ok::<(), rustellar_core::TimeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TimeDelta {
    days: f64,
}

impl TimeDelta {
    fn new(days: f64) -> Result<Self, TimeError> {
        if !days.is_finite() {
            return Err(TimeError::NonFinite);
        }

        Ok(Self { days })
    }

    /// Makes a time delta from a number of days.
    ///
    /// # Errors
    ///
    /// [`TimeError::NonFinite`] if `days` is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{TimeDelta, TimeError};
    ///
    /// let period = TimeDelta::from_days(2.5)?;
    /// assert_eq!(period.as_hours(), 60.0);
    /// assert_eq!(TimeDelta::from_days(f64::INFINITY), Err(TimeError::NonFinite));
    /// # Ok::<(), TimeError>(())
    /// ```
    pub fn from_days(days: f64) -> Result<Self, TimeError> {
        Self::new(days)
    }

    /// Makes a time delta from a number of hours.
    ///
    /// # Errors
    ///
    /// [`TimeError::NonFinite`] if `hours` is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::TimeDelta;
    ///
    /// let duration = TimeDelta::from_hours(3.0)?;
    /// assert_eq!(duration.as_days(), 0.125);
    /// # Ok::<(), rustellar_core::TimeError>(())
    /// ```
    pub fn from_hours(hours: f64) -> Result<Self, TimeError> {
        Self::new(hours / HOURS_PER_DAY)
    }

    /// The time delta in days.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::TimeDelta;
    ///
    /// assert_eq!(TimeDelta::from_hours(12.0)?.as_days(), 0.5);
    /// # Ok::<(), rustellar_core::TimeError>(())
    /// ```
    #[must_use]
    pub fn as_days(self) -> f64 {
        self.days
    }

    /// The time delta in hours.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::TimeDelta;
    ///
    /// assert_eq!(TimeDelta::from_days(0.25)?.as_hours(), 6.0);
    /// # Ok::<(), rustellar_core::TimeError>(())
    /// ```
    #[must_use]
    pub fn as_hours(self) -> f64 {
        self.days * HOURS_PER_DAY
    }
}

impl Add for TimeDelta {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            days: self.days + rhs.days,
        }
    }
}

impl Sub for TimeDelta {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            days: self.days - rhs.days,
        }
    }
}

impl Div for TimeDelta {
    type Output = f64;

    fn div(self, rhs: Self) -> Self::Output {
        self.days / rhs.days
    }
}

impl Mul<f64> for TimeDelta {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            days: self.days * rhs,
        }
    }
}

impl Neg for TimeDelta {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { days: -self.days }
    }
}

/// A point in time, in days on the time scale of the data.
///
/// The crates only ever use differences between epochs, so any scale
/// counted in days works: BJD_TDB, or an offset form of it such as TESS's
/// BTJD (BJD_TDB - 2457000), Kepler's BKJD (BJD_TDB - 2454833) or MJD.
/// Keep every epoch of one analysis on the same scale; nothing converts
/// between scales.
///
/// A full BJD of about 2.46 million days leaves room for times to about
/// 40 microseconds only. Subtract a constant first (as BTJD and BKJD do)
/// to keep sub-millisecond precision.
///
/// As with [`TimeDelta`], the constructor rejects NaN and infinity but the
/// operators do not check their results, so adding or subtracting a
/// non-finite delta, or one near `f64::MAX`, gives a non-finite epoch, and
/// subtracting two epochs near `f64::MAX` gives a non-finite delta.
///
/// # Examples
///
/// ```
/// use rustellar_core::{Epoch, TimeDelta};
///
/// let t0 = Epoch::from_days(1325.5)?;
/// let later = t0 + TimeDelta::from_days(2.0)?;
/// assert_eq!(later.as_days(), 1327.5);
/// assert_eq!((later - t0).as_days(), 2.0);
/// # Ok::<(), rustellar_core::TimeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Epoch {
    days: f64,
}

impl Epoch {
    /// Makes an epoch from a time in days, on the data's own scale.
    ///
    /// # Errors
    ///
    /// [`TimeError::NonFinite`] if `days` is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{Epoch, TimeError};
    ///
    /// assert_eq!(Epoch::from_days(3000.25)?.as_days(), 3000.25);
    /// assert_eq!(Epoch::from_days(f64::NAN), Err(TimeError::NonFinite));
    /// # Ok::<(), TimeError>(())
    /// ```
    pub fn from_days(days: f64) -> Result<Self, TimeError> {
        if !days.is_finite() {
            return Err(TimeError::NonFinite);
        }
        Ok(Self { days })
    }

    /// The epoch in days, on the scale it was made on.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::Epoch;
    ///
    /// assert_eq!(Epoch::from_days(1325.5)?.as_days(), 1325.5);
    /// # Ok::<(), rustellar_core::TimeError>(())
    /// ```
    #[must_use]
    pub fn as_days(self) -> f64 {
        self.days
    }
}

impl Sub for Epoch {
    type Output = TimeDelta;

    fn sub(self, rhs: Self) -> Self::Output {
        TimeDelta {
            days: self.days - rhs.days,
        }
    }
}

impl Add<TimeDelta> for Epoch {
    type Output = Self;

    fn add(self, rhs: TimeDelta) -> Self::Output {
        Self {
            days: self.days + rhs.days,
        }
    }
}

impl Sub<TimeDelta> for Epoch {
    type Output = Self;

    fn sub(self, rhs: TimeDelta) -> Self::Output {
        Self {
            days: self.days - rhs.days,
        }
    }
}
