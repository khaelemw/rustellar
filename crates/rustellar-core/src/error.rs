use std::fmt;

/// Why a light curve could not be built or folded.
///
/// Returned by [`LightCurve::new`](crate::LightCurve::new) and
/// [`LightCurve::fold`](crate::LightCurve::fold). New variants may be added in
/// minor releases, so a `match` on this type needs a wildcard arm.
///
/// It implements [`std::error::Error`], so `?` converts it into a
/// `Box<dyn Error>` or any other error type that accepts standard errors.
///
/// # Examples
///
/// ```
/// use std::error::Error;
///
/// use rustellar_core::LightCurve;
///
/// fn load() -> Result<LightCurve, Box<dyn Error>> {
///     let lc = LightCurve::new(vec![2.0, 1.0], vec![1.0, 1.0], vec![0.1, 0.1])?;
///     Ok(lc)
/// }
///
/// let err = load().unwrap_err();
/// assert_eq!(err.to_string(), "time values must be strictly increasing");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LightCurveError {
    /// The arrays have no points.
    Empty,
    /// `time`, `flux`, and `flux_err` do not all have the same length.
    LengthMismatch,
    /// A time is not greater than the one before it (out of order or repeated).
    NotSorted,
    /// A time, flux, or flux error is NaN or infinite.
    NonFinite,
    /// A flux error is zero or negative.
    NonPositiveError,
    /// The fold period is zero, negative, NaN, or infinite.
    InvalidPeriod,
    /// The fold reference time `t0` is NaN or infinite.
    InvalidT0,
}

impl fmt::Display for LightCurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "light curve has no points"),
            Self::LengthMismatch => {
                write!(f, "time, flux, and flux_err must have the same length")
            }
            Self::NotSorted => write!(f, "time values must be strictly increasing"),
            Self::NonFinite => write!(f, "time, flux, and flux_err must all be finite"),
            Self::NonPositiveError => write!(f, "every flux_err value must be greater than zero"),
            Self::InvalidPeriod => write!(f, "fold period must be finite and greater than zero"),
            Self::InvalidT0 => write!(f, "fold reference time t0 must be finite"),
        }
    }
}

impl std::error::Error for LightCurveError {}
