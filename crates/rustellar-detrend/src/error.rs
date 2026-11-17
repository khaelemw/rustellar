use std::fmt;

/// Why a light curve could not be detrended.
///
/// New variants may be added in minor releases, so a `match` on this type needs
/// a wildcard arm.
///
/// # Examples
///
/// ```
/// use rustellar_core::{LightCurve, TimeDelta};
/// use rustellar_detrend::{Detrend, DetrendError};
///
/// let lc = LightCurve::from_btjd(vec![1.0, 2.0], vec![1.0, 0.99], vec![0.01; 2])?;
/// let detrend = Detrend::default().with_window(TimeDelta::from_days(0.0)?);
/// let err = detrend.trend(&lc, None).unwrap_err();
/// assert_eq!(err, DetrendError::NotPositive { name: "window" });
/// assert_eq!(err.to_string(), "window must be greater than zero");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DetrendError {
    /// A length that must be greater than zero is zero or negative.
    NotPositive {
        /// Which setting: `"window"` or `"break_tolerance"`.
        name: &'static str,
    },
    /// The edge cutoff is negative.
    NegativeEdgeCutoff,
    /// An M-estimator's tuning constant `c` is not between `1e-30` and
    /// `1e30` (or is NaN).
    TuningOutOfRange,
    /// A trimmed or winsorized mean's proportion is not from 0 to below 1/2
    /// (or is NaN).
    ProportionOutOfRange,
    /// The mask's length is not the light curve's.
    MaskLengthMismatch,
    /// [`Detrend::flatten`](crate::Detrend::flatten) found no point with a
    /// trend (every window cut off or masked, or every flattened value out
    /// of range), so there is no light curve to return.
    NoPointsLeft,
}

impl fmt::Display for DetrendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { name } => write!(f, "{name} must be greater than zero"),
            Self::NegativeEdgeCutoff => write!(f, "edge_cutoff must not be negative"),
            Self::TuningOutOfRange => {
                write!(f, "the tuning constant c must be between 1e-30 and 1e30")
            }
            Self::ProportionOutOfRange => {
                write!(f, "the proportion must be at least 0 and below 0.5")
            }
            Self::MaskLengthMismatch => {
                write!(
                    f,
                    "the mask must have one entry per point of the light curve"
                )
            }
            Self::NoPointsLeft => write!(f, "no point has a trend to flatten by"),
        }
    }
}

impl std::error::Error for DetrendError {}
