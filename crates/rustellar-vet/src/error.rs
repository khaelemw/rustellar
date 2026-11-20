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
    /// A limb-darkened shape's impact parameter is at least `1 + k`: the
    /// planet would miss the star.
    NoTransit,
    /// A limb-darkened shape was asked for a duration of more than half the
    /// period, which no circular orbit gives.
    DurationTooLongForShape,
    /// No point of the light curve falls in transit.
    NoPointsInTransit,
    /// Every point of the light curve falls in transit, so there is no
    /// baseline.
    NoPointsOutOfTransit,
    /// The fitted baseline is zero or negative, so no fractional depth
    /// exists.
    BaselineNotPositive,
    /// A period ratio with a zero numerator or denominator.
    RatioZero,
    /// An alias's period, the candidate's times a ratio, is too large to
    /// hold.
    AliasPeriodNotFinite,
    /// A period search's width is not above 0 and at most 0.5.
    WidthOutOfRange,
    /// A suspect period's relative tolerance is not at least 0 and below 1.
    ToleranceOutOfRange,
    /// A time window ends before it starts.
    WindowReversed,
}

impl fmt::Display for VetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { name } => write!(f, "{name} must be greater than zero"),
            Self::DurationNotShorterThanPeriod => {
                write!(f, "the duration must be shorter than the period")
            }
            Self::DepthNotFinite => write!(f, "the depth must be a finite number"),
            Self::NoTransit => write!(
                f,
                "the impact parameter must be below 1 + k for the planet to cross the star"
            ),
            Self::DurationTooLongForShape => write!(
                f,
                "a limb-darkened shape needs a duration of at most half the period"
            ),
            Self::NoPointsInTransit => write!(f, "no point of the light curve is in transit"),
            Self::NoPointsOutOfTransit => {
                write!(f, "no point of the light curve is out of transit")
            }
            Self::BaselineNotPositive => write!(f, "the fitted baseline is not above zero"),
            Self::RatioZero => write!(
                f,
                "a period ratio needs a numerator and denominator above zero"
            ),
            Self::AliasPeriodNotFinite => write!(f, "the alias's period is too large to hold"),
            Self::WidthOutOfRange => {
                write!(f, "a period search's width must be above 0 and at most 0.5")
            }
            Self::ToleranceOutOfRange => {
                write!(
                    f,
                    "a suspect period's tolerance must be at least 0 and below 1"
                )
            }
            Self::WindowReversed => write!(f, "a time window must not end before it starts"),
        }
    }
}

impl std::error::Error for VetError {}
