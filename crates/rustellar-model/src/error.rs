use std::fmt;

/// Why transit parameters could not be built.
///
/// New variants may be added in minor releases, so a `match` on this type needs
/// a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamError {
    /// A parameter is NaN or infinite.
    NonFinite {
        /// Which parameter, for example `"k"`.
        name: &'static str,
    },
    /// The radius ratio `k` is zero or negative.
    RadiusRatioNotPositive,
    /// The radius ratio `k` is above [`RadiusRatio::MAX`](crate::RadiusRatio::MAX).
    RadiusRatioTooLarge,
    /// The impact parameter `b` is negative.
    ImpactParameterNegative,
    /// The scaled semi-major axis `a_rs` is 1 or less (the orbit is inside the
    /// star).
    SemiMajorAxisTooSmall,
    /// The period is zero or negative.
    PeriodNotPositive,
    /// The impact parameter `b` is not less than `a_rs`, so no inclination
    /// gives it.
    ImpactParameterTooLarge,
    /// The inclination is not in `(0, 90]` degrees.
    InclinationOutOfRange,
    /// An exposure's duration is zero or negative.
    ExposureNotPositive,
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { name } => write!(f, "{name} must be finite"),
            Self::RadiusRatioNotPositive => {
                write!(f, "radius ratio k must be greater than zero")
            }
            Self::RadiusRatioTooLarge => write!(f, "radius ratio k must be at most 100"),
            Self::ImpactParameterNegative => write!(f, "impact parameter b must not be negative"),
            Self::SemiMajorAxisTooSmall => {
                write!(f, "scaled semi-major axis a_rs must be greater than 1")
            }
            Self::PeriodNotPositive => write!(f, "period must be greater than zero"),
            Self::ImpactParameterTooLarge => {
                write!(f, "impact parameter b must be less than a_rs")
            }
            Self::InclinationOutOfRange => {
                write!(
                    f,
                    "inclination must be greater than 0 and at most 90 degrees"
                )
            }
            Self::ExposureNotPositive => write!(f, "exposure duration must be greater than zero"),
        }
    }
}

impl std::error::Error for ParamError {}

/// Why limb-darkening coefficients were rejected.
///
/// New variants may be added in minor releases, so a `match` on this type needs
/// a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LimbDarkeningError {
    /// A coefficient is NaN or infinite.
    NonFinite,
    /// The coefficients make the intensity negative somewhere on the stellar
    /// disk.
    NegativeIntensity,
}

impl fmt::Display for LimbDarkeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "limb-darkening coefficients must be finite"),
            Self::NegativeIntensity => write!(
                f,
                "limb-darkening coefficients give negative intensity on the stellar disk"
            ),
        }
    }
}

impl std::error::Error for LimbDarkeningError {}

/// An output buffer whose length does not match the number of times.
///
/// Returned by the `light_curve_into` methods, which write one flux per time
/// into a buffer the caller owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct LengthMismatch {
    /// The number of times.
    pub times: usize,
    /// The length of the output buffer.
    pub output: usize,
}

impl fmt::Display for LengthMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "output has length {} but there are {} times",
            self.output, self.times
        )
    }
}

impl std::error::Error for LengthMismatch {}
