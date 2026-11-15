use crate::error::ParamError;

/// Planet radius divided by star radius, `k = Rp / Rs`.
///
/// Always finite, greater than zero and at most [`RadiusRatio::MAX`]. Values
/// above 1 are allowed: the geometry still works for a body larger than the
/// star, as in an eclipsing binary.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RadiusRatio(f64);

impl RadiusRatio {
    /// The largest radius ratio accepted, 100.
    ///
    /// Above this the analytic flux loses accuracy: its rounding error grows
    /// roughly as `k^3 * 1e-15`, reaching about `2e-9` at 100 and `1e-6` at
    /// 1000. Planets have `k` below about 0.3, and eclipsing binaries rarely
    /// exceed 50.
    pub const MAX: f64 = 100.0;

    /// Makes a radius ratio.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if `k` is NaN or infinite.
    /// - [`ParamError::RadiusRatioNotPositive`] if `k` is zero or negative.
    /// - [`ParamError::RadiusRatioTooLarge`] if `k` is above
    ///   [`RadiusRatio::MAX`].
    pub fn new(k: f64) -> Result<Self, ParamError> {
        if !k.is_finite() {
            return Err(ParamError::NonFinite { name: "k" });
        }

        if k <= 0.0 {
            return Err(ParamError::RadiusRatioNotPositive);
        }

        if k > Self::MAX {
            return Err(ParamError::RadiusRatioTooLarge);
        }

        Ok(Self(k))
    }

    /// The radius ratio as a number.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// Impact parameter `b`: the smallest distance between the planet and star
/// centres on the sky, in stellar radii.
///
/// Always finite and zero or more. `b = 0` is a central transit; `b` above
/// `1 + k` means the planet misses the star.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
// `pub(super)` lets `TransitParams::from_inclination_deg` build one directly
// after its own range check; everything else goes through `new`.
pub struct ImpactParameter(pub(super) f64);

impl ImpactParameter {
    /// Makes an impact parameter.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if `b` is NaN or infinite.
    /// - [`ParamError::ImpactParameterNegative`] if `b` is negative.
    pub fn new(b: f64) -> Result<Self, ParamError> {
        if !b.is_finite() {
            return Err(ParamError::NonFinite { name: "b" });
        }

        if b < 0.0 {
            return Err(ParamError::ImpactParameterNegative);
        }

        Ok(Self(b))
    }

    /// The impact parameter in stellar radii.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// Orbital semi-major axis divided by star radius, `a_rs = a / Rs`.
///
/// Always finite and greater than 1, so the orbit is outside the star.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ScaledSemiMajorAxis(f64);

impl ScaledSemiMajorAxis {
    /// Makes a scaled semi-major axis.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if `a_rs` is NaN or infinite.
    /// - [`ParamError::SemiMajorAxisTooSmall`] if `a_rs` is 1 or less.
    pub fn new(a_rs: f64) -> Result<Self, ParamError> {
        if !a_rs.is_finite() {
            return Err(ParamError::NonFinite { name: "a_rs" });
        }

        if a_rs <= 1.0 {
            return Err(ParamError::SemiMajorAxisTooSmall);
        }

        Ok(Self(a_rs))
    }

    /// The scaled semi-major axis in stellar radii.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}
