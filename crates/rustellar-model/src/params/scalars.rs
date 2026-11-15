use crate::error::ParamError;

/// Planet radius divided by star radius, `k = Rp / Rs`.
///
/// Always finite, greater than zero and at most [`RadiusRatio::MAX`]. Values
/// above 1 are allowed: the geometry still works for a body larger than the
/// star, as in an eclipsing binary.
///
/// # Examples
///
/// ```
/// use rustellar_model::RadiusRatio;
///
/// // A Jupiter around a Sun-like star.
/// let k = RadiusRatio::new(0.1)?;
/// assert_eq!(k.get(), 0.1);
/// # Ok::<(), rustellar_model::ParamError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RadiusRatio(f64);

impl RadiusRatio {
    /// The largest radius ratio accepted, 10,000.
    ///
    /// Above `k = 1` the analytic flux's rounding error grows in proportion to
    /// `k`, about `1.5e-16 k`, so `1.5e-12` here; this is as far as it is
    /// tested. Planets have `k` below about 0.3, and eclipsing binaries rarely
    /// exceed 50.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_model::{ParamError, RadiusRatio};
    ///
    /// assert!(RadiusRatio::new(RadiusRatio::MAX).is_ok());
    /// assert_eq!(RadiusRatio::new(2e4), Err(ParamError::RadiusRatioTooLarge));
    /// ```
    pub const MAX: f64 = 1e4;

    /// Makes a radius ratio.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if `k` is NaN or infinite.
    /// - [`ParamError::RadiusRatioNotPositive`] if `k` is zero or negative.
    /// - [`ParamError::RadiusRatioTooLarge`] if `k` is above
    ///   [`RadiusRatio::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_model::{ParamError, RadiusRatio};
    ///
    /// assert!(RadiusRatio::new(0.1).is_ok());
    /// assert!(RadiusRatio::new(2.0).is_ok()); // a body larger than the star
    /// assert_eq!(RadiusRatio::new(0.0), Err(ParamError::RadiusRatioNotPositive));
    /// assert_eq!(RadiusRatio::new(f64::NAN), Err(ParamError::NonFinite { name: "k" }));
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_model::RadiusRatio;
    ///
    /// assert_eq!(RadiusRatio::new(0.1)?.get(), 0.1);
    /// # Ok::<(), rustellar_model::ParamError>(())
    /// ```
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
///
/// # Examples
///
/// ```
/// use rustellar_model::ImpactParameter;
///
/// let central = ImpactParameter::new(0.0)?;
/// assert_eq!(central.get(), 0.0);
/// # Ok::<(), rustellar_model::ParamError>(())
/// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_model::{ImpactParameter, ParamError};
    ///
    /// assert!(ImpactParameter::new(0.5).is_ok());
    /// assert!(ImpactParameter::new(1.5).is_ok()); // misses the star, still valid
    /// assert_eq!(ImpactParameter::new(-0.1), Err(ParamError::ImpactParameterNegative));
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_model::ImpactParameter;
    ///
    /// assert_eq!(ImpactParameter::new(0.5)?.get(), 0.5);
    /// # Ok::<(), rustellar_model::ParamError>(())
    /// ```
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// Orbital semi-major axis divided by star radius, `a_rs = a / Rs`.
///
/// Always finite and greater than 1, so the orbit is outside the star.
///
/// # Examples
///
/// ```
/// use rustellar_model::ScaledSemiMajorAxis;
///
/// // A hot Jupiter: about ten stellar radii out.
/// let a_rs = ScaledSemiMajorAxis::new(10.0)?;
/// assert_eq!(a_rs.get(), 10.0);
/// # Ok::<(), rustellar_model::ParamError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ScaledSemiMajorAxis(f64);

impl ScaledSemiMajorAxis {
    /// Makes a scaled semi-major axis.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if `a_rs` is NaN or infinite.
    /// - [`ParamError::SemiMajorAxisTooSmall`] if `a_rs` is 1 or less.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_model::{ParamError, ScaledSemiMajorAxis};
    ///
    /// assert!(ScaledSemiMajorAxis::new(1.5).is_ok());
    /// assert_eq!(ScaledSemiMajorAxis::new(0.9), Err(ParamError::SemiMajorAxisTooSmall));
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_model::ScaledSemiMajorAxis;
    ///
    /// assert_eq!(ScaledSemiMajorAxis::new(10.0)?.get(), 10.0);
    /// # Ok::<(), rustellar_model::ParamError>(())
    /// ```
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}
