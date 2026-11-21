use std::fmt;

/// Why a periodogram could not be set up or computed.
///
/// New variants may be added in minor releases, so a `match` on this type
/// needs a wildcard arm.
///
/// # Examples
///
/// ```
/// use rustellar_ls::{Frequencies, LsError};
///
/// let err = Frequencies::from_per_day(&[0.5, -1.0]).unwrap_err();
/// assert_eq!(err, LsError::NotPositive { name: "frequency" });
/// assert_eq!(err.to_string(), "frequency must be greater than zero");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LsError {
    /// A value is NaN or infinite.
    NonFinite {
        /// Which input, such as `"frequency"`.
        name: &'static str,
    },
    /// A value that must be greater than zero is zero or negative.
    NotPositive {
        /// Which input, such as `"frequency"` or `"samples_per_peak"`.
        name: &'static str,
    },
    /// No frequency was given.
    NoFrequencies,
    /// The light curve has a single time, so it spans no baseline.
    NoBaseline,
    /// The lowest frequency of a grid is above its highest.
    FrequencyRangeEmpty,
    /// The grid would hold more than a billion frequencies.
    GridTooLarge,
    /// The reference chi-squared is zero (the flux is constant, or zero
    /// without a mean), so no power is defined.
    NoVariation,
    /// The fast method was given frequencies that are not an even grid.
    NotUniform,
}

impl fmt::Display for LsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { name } => write!(f, "{name} must be finite"),
            Self::NotPositive { name } => write!(f, "{name} must be greater than zero"),
            Self::NoFrequencies => write!(f, "at least one frequency is needed"),
            Self::NoBaseline => write!(f, "the light curve must span more than one time"),
            Self::FrequencyRangeEmpty => write!(
                f,
                "the minimum frequency must not be above the maximum frequency"
            ),
            Self::GridTooLarge => write!(
                f,
                "the frequency grid would hold more than a billion frequencies"
            ),
            Self::NoVariation => write!(f, "the flux must vary about its reference fit"),
            Self::NotUniform => write!(f, "the fast method needs an even grid of frequencies"),
        }
    }
}

impl std::error::Error for LsError {}
