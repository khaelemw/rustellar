use std::fmt;

/// Why a fit could not be set up or run.
///
/// New variants may be added in minor releases, so a `match` on this type
/// needs a wildcard arm.
///
/// # Examples
///
/// ```
/// use rustellar_fit::FitError;
///
/// let err = FitError::NotPositive { name: "baseline" };
/// assert_eq!(err.to_string(), "baseline must be greater than zero");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FitError {
    /// A value is NaN or infinite.
    NonFinite {
        /// Which input, such as `"baseline"`.
        name: &'static str,
    },
    /// A value that must be greater than zero is zero or negative.
    NotPositive {
        /// Which input, such as `"baseline"`.
        name: &'static str,
    },
    /// Every parameter is held fixed, so there is nothing to fit.
    NothingToFit,
    /// Fewer points than free parameters, so the fit is not determined.
    TooFewPoints {
        /// The fewest points needed: one more than the free parameters.
        needed: usize,
    },
    /// The starting parameters put no point in transit, so the transit's
    /// shape cannot be fitted.
    NoPointsInTransit,
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { name } => write!(f, "{name} must be finite"),
            Self::NotPositive { name } => write!(f, "{name} must be greater than zero"),
            Self::NothingToFit => write!(f, "at least one parameter must be free"),
            Self::TooFewPoints { needed } => write!(f, "at least {needed} points are needed"),
            Self::NoPointsInTransit => {
                write!(f, "the starting parameters put no point in transit")
            }
        }
    }
}

impl std::error::Error for FitError {}
