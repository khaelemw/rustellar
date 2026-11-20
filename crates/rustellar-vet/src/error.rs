use std::fmt;

/// Why a candidate could not be built or vetted.
///
/// New variants may be added in minor releases, so a `match` on this type
/// needs a wildcard arm.
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, TimeDelta, VetError};
///
/// let err = Candidate::new(
///     TimeDelta::from_days(2.0)?,
///     Epoch::from_btjd(1.0)?,
///     TimeDelta::from_days(3.0)?,
/// )
/// .unwrap_err();
/// assert_eq!(err, VetError::DurationNotShorterThanPeriod);
/// assert_eq!(err.to_string(), "the duration must be shorter than the period");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VetError {
    /// A length that must be greater than zero is zero or negative.
    NotPositive {
        /// Which: `"period"` or `"duration"`.
        name: &'static str,
    },
    /// The transit lasts as long as the period or longer.
    DurationNotShorterThanPeriod,
    /// The depth is NaN or infinite.
    DepthNotFinite,
}

impl fmt::Display for VetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { name } => write!(f, "{name} must be greater than zero"),
            Self::DurationNotShorterThanPeriod => {
                write!(f, "the duration must be shorter than the period")
            }
            Self::DepthNotFinite => write!(f, "the depth must be a finite number"),
        }
    }
}

impl std::error::Error for VetError {}
