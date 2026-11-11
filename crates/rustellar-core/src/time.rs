use std::ops::{Add, Div, Neg, Sub};

use crate::TimeError;

const BTJD_OFFSET: f64 = 2_457_000.0;
const HOURS_PER_DAY: f64 = 24.0;

/// A length of time, such as an orbital period or a transit duration.
///
/// Stored in days. It may be zero or negative, but never NaN or infinite.
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
    pub fn from_days(days: f64) -> Result<Self, TimeError> {
        Self::new(days)
    }

    /// Makes a time delta from a number of hours.
    ///
    /// # Errors
    ///
    /// [`TimeError::NonFinite`] if `hours` is NaN or infinite.
    pub fn from_hours(hours: f64) -> Result<Self, TimeError> {
        Self::new(hours / HOURS_PER_DAY)
    }

    /// The time delta in days.
    pub fn as_days(self) -> f64 {
        self.days
    }

    /// The time delta in hours.
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

impl Neg for TimeDelta {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { days: -self.days }
    }
}

/// A point in time on the barycentric (BJD_TDB) time scale.
///
/// Stored as BTJD, the TESS convention: BJD_TDB minus 2457000. Converting from
/// BJD is exact for any realistic date, but converting back to BJD rounds to
/// about 40 microseconds, because the larger number has less precision left
/// for the fraction of a day.
///
/// There is no conversion from JD or UTC, as that needs the target's position
/// on the sky.
///
/// # Examples
///
/// ```
/// use rustellar_core::Epoch;
///
/// let t = Epoch::from_bjd(2_460_000.5)?;
/// assert_eq!(t.as_btjd(), 3000.5);
/// # Ok::<(), rustellar_core::TimeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Epoch {
    btjd: f64,
}

impl Epoch {
    fn new(btjd: f64) -> Result<Self, TimeError> {
        if !btjd.is_finite() {
            return Err(TimeError::NonFinite);
        }

        Ok(Self { btjd })
    }

    /// Makes an epoch from a BTJD value.
    ///
    /// # Errors
    ///
    /// [`TimeError::NonFinite`] if `btjd` is NaN or infinite.
    pub fn from_btjd(btjd: f64) -> Result<Self, TimeError> {
        Self::new(btjd)
    }

    /// Makes an epoch from a BJD_TDB value.
    ///
    /// # Errors
    ///
    /// [`TimeError::NonFinite`] if `bjd` is NaN or infinite.
    pub fn from_bjd(bjd: f64) -> Result<Self, TimeError> {
        Self::new(bjd - BTJD_OFFSET)
    }

    /// The epoch as BTJD.
    pub fn as_btjd(self) -> f64 {
        self.btjd
    }

    /// The epoch as BJD_TDB.
    pub fn as_bjd(self) -> f64 {
        self.btjd + BTJD_OFFSET
    }
}

impl Sub for Epoch {
    type Output = TimeDelta;

    fn sub(self, rhs: Self) -> Self::Output {
        TimeDelta {
            days: self.btjd - rhs.btjd,
        }
    }
}

impl Add<TimeDelta> for Epoch {
    type Output = Self;

    fn add(self, rhs: TimeDelta) -> Self::Output {
        Self {
            btjd: self.btjd + rhs.days,
        }
    }
}

impl Sub<TimeDelta> for Epoch {
    type Output = Self;

    fn sub(self, rhs: TimeDelta) -> Self::Output {
        Self {
            btjd: self.btjd - rhs.days,
        }
    }
}
