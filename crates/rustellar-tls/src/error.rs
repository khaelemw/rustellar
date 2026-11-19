use std::fmt;

use rustellar_core::GridError;

/// Why a search could not be run.
///
/// New variants may be added in minor releases, so a `match` on this type
/// needs a wildcard arm.
///
/// # Examples
///
/// ```
/// use rustellar_tls::{SearchOptions, Tls, TlsError};
/// use rustellar_tls::LightCurve;
///
/// let lc = LightCurve::from_btjd(vec![1.0], vec![1.0], vec![1e-3])?;
/// let err = Tls::new(&lc).unwrap_err();
/// assert_eq!(err, TlsError::NoBaseline);
/// assert_eq!(err.to_string(), "the light curve must span more than one time");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TlsError {
    /// A setting is NaN or infinite.
    NonFinite {
        /// Which setting, such as `"min_depth"`.
        name: &'static str,
    },
    /// A setting that must be greater than zero is zero or negative.
    NotPositive {
        /// Which setting, such as `"stars.radius_min"`.
        name: &'static str,
    },
    /// The light curve has a single time, so it spans no baseline.
    NoBaseline,
    /// A weight `1 / flux_err^2` is too large or too small for an `f64` to
    /// hold to full precision.
    OutOfRange,
    /// [`SearchOptions::duration_step`](crate::SearchOptions::duration_step)
    /// is not above 1, so the durations would not grow.
    DurationStepNotAboveOne,
    /// A range of [`Stars`](crate::Stars) has its least above its greatest.
    StarRangeEmpty,
    /// A [`Template::Planet`](crate::Template::Planet) does not transit: its
    /// impact parameter is negative, or not below `1 + k`, or it dims
    /// nothing.
    TemplateNotTransiting,
    /// The GPU asked for with [`Device::Gpu`](crate::Device::Gpu) cannot be
    /// used: there is no such GPU, the CUDA driver or NVRTC library is
    /// missing, or the `cuda` feature is off.
    GpuUnavailable,
    /// The periods could not be made: a [`Periods`](crate::Periods) list or
    /// an [`OfirPeriod`](crate::OfirPeriod) grid failed, so `?` works on
    /// them in a function returning `TlsError`.
    Grid(GridError),
}

impl fmt::Display for TlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { name } => write!(f, "{name} must be finite"),
            Self::NotPositive { name } => write!(f, "{name} must be greater than zero"),
            Self::NoBaseline => f.write_str("the light curve must span more than one time"),
            Self::OutOfRange => f.write_str("a weight is out of the range an f64 holds precisely"),
            Self::DurationStepNotAboveOne => f.write_str("duration_step must be above 1"),
            Self::StarRangeEmpty => {
                f.write_str("a range of stars has its least above its greatest")
            }
            Self::TemplateNotTransiting => f.write_str("the template's planet does not transit"),
            Self::GpuUnavailable => f.write_str("the GPU asked for cannot be used"),
            Self::Grid(error) => write!(f, "the periods: {error}"),
        }
    }
}

impl std::error::Error for TlsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Grid(error) => Some(error),
            _ => None,
        }
    }
}

/// # Examples
///
/// ```
/// use rustellar_tls::{GridError, Periods, TlsError};
///
/// fn periods() -> Result<Periods, TlsError> {
///     Ok(Periods::from_days(&[])?)
/// }
/// assert_eq!(periods(), Err(TlsError::Grid(GridError::NoPeriods)));
/// ```
impl From<GridError> for TlsError {
    fn from(error: GridError) -> Self {
        Self::Grid(error)
    }
}
