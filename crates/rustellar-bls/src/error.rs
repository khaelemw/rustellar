use std::fmt;

/// Why a search's inputs could not be built.
///
/// New variants may be added in minor releases, so a `match` on this type needs
/// a wildcard arm.
///
/// # Examples
///
/// ```
/// use rustellar_bls::{BlsError, Periods};
///
/// let err = Periods::from_days(&[1.0, -2.0]).unwrap_err();
/// assert_eq!(err, BlsError::PeriodNotPositive);
/// assert_eq!(err.to_string(), "periods must be greater than zero");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlsError {
    /// A value is NaN or infinite.
    NonFinite {
        /// Which input, such as `"duration"` or `"period"`.
        name: &'static str,
    },
    /// The list of durations is empty.
    NoDurations,
    /// A duration is zero or negative.
    DurationNotPositive,
    /// The list of periods is empty.
    NoPeriods,
    /// A period is zero or negative.
    PeriodNotPositive,
    /// The longest duration is not shorter than the shortest period, so the
    /// box would cover a whole orbit.
    DurationNotShorterThanPeriod,
    /// The light curve has a single time, so it spans no baseline to build a
    /// period grid from.
    NoBaseline,
    /// [`AutoPeriod::minimum_n_transit`](crate::AutoPeriod::minimum_n_transit)
    /// is below 2.
    TooFewTransits,
    /// [`AutoPeriod::frequency_factor`](crate::AutoPeriod::frequency_factor)
    /// is zero or negative.
    FrequencyFactorNotPositive,
    /// The shortest period of a grid is above its longest.
    PeriodRangeEmpty,
    /// The frequency step is so large that the grid would reach a frequency
    /// of zero or below.
    GridTooCoarse,
    /// A setting that must be greater than zero is zero or negative.
    NotPositive {
        /// Which setting, such as `"stellar_radius"`.
        name: &'static str,
    },
    /// The grid would hold more than 100 million periods.
    GridTooLarge,
    /// A box holds none of the light curve's points.
    NoPointsInTransit,
    /// A box holds every one of the light curve's points, leaving no
    /// out-of-transit level.
    NoPointsOutOfTransit,
    /// A weight `1 / flux_err^2`, or a sum of weights or of their products
    /// with the flux, is too large for an `f64`.
    Overflow,
}

impl fmt::Display for BlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { name } => write!(f, "{name} must be finite"),
            Self::NoDurations => write!(f, "at least one duration is needed"),
            Self::DurationNotPositive => write!(f, "durations must be greater than zero"),
            Self::NoPeriods => write!(f, "at least one period is needed"),
            Self::PeriodNotPositive => write!(f, "periods must be greater than zero"),
            Self::DurationNotShorterThanPeriod => write!(
                f,
                "the longest duration must be shorter than the shortest period"
            ),
            Self::NoBaseline => write!(f, "the light curve must span more than one time"),
            Self::TooFewTransits => write!(f, "minimum_n_transit must be at least 2"),
            Self::FrequencyFactorNotPositive => {
                write!(f, "frequency_factor must be greater than zero")
            }
            Self::PeriodRangeEmpty => {
                write!(f, "the minimum period must not be above the maximum period")
            }
            Self::GridTooCoarse => write!(
                f,
                "frequency_factor is too large for the period range: the grid reaches zero frequency"
            ),
            Self::NotPositive { name } => write!(f, "{name} must be greater than zero"),
            Self::NoPointsInTransit => write!(f, "the box holds none of the points"),
            Self::NoPointsOutOfTransit => {
                write!(f, "the box holds every point, leaving none out of transit")
            }
            Self::Overflow => write!(
                f,
                "the weights 1 / flux_err^2, or their sums, are too large for an f64"
            ),
            Self::GridTooLarge => write!(
                f,
                "the period grid would hold more than 100 million periods"
            ),
        }
    }
}

impl std::error::Error for BlsError {}
