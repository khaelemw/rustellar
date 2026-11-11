use rustellar_core::{Epoch, TimeDelta};

use crate::error::{LimbDarkeningError, ParamError};

/// Planet radius divided by star radius, `k = Rp / Rs`.
///
/// Always finite and greater than zero. Values above 1 are allowed: the
/// geometry still works for a body larger than the star.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RadiusRatio(f64);

impl RadiusRatio {
    /// Makes a radius ratio.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if `k` is NaN or infinite.
    /// - [`ParamError::RadiusRatioNotPositive`] if `k` is zero or negative.
    pub fn new(k: f64) -> Result<Self, ParamError> {
        if !k.is_finite() {
            return Err(ParamError::NonFinite { name: "k" });
        }

        if k <= 0.0 {
            return Err(ParamError::RadiusRatioNotPositive);
        }

        Ok(Self(k))
    }

    /// The radius ratio as a number.
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
pub struct ImpactParameter(f64);

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
    pub fn get(self) -> f64 {
        self.0
    }
}

/// How the star's brightness falls from the centre of its disk to the edge.
///
/// The intensity is `I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2`, where `mu` is 1
/// at the disk centre and 0 at the edge. Uniform is `u1 = u2 = 0`, and linear
/// is `u2 = 0`.
///
/// Coefficients are rejected only if they make the intensity negative
/// somewhere on the disk. Negative `u1` (brighter towards the edge) is allowed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimbDarkening(Law);

// Private, so a law can only be made through the checked constructors.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Law {
    Uniform,
    Linear { u1: f64 },
    Quadratic { u1: f64, u2: f64 },
}

impl LimbDarkening {
    /// A star with the same brightness everywhere on its disk.
    pub fn uniform() -> Self {
        Self(Law::Uniform)
    }

    /// The linear law, `I(mu) = 1 - u1 (1 - mu)`.
    ///
    /// # Errors
    ///
    /// - [`LimbDarkeningError::NonFinite`] if `u1` is NaN or infinite.
    /// - [`LimbDarkeningError::NegativeIntensity`] if `u1` is greater than 1.
    pub fn linear(u1: f64) -> Result<Self, LimbDarkeningError> {
        check_intensity(u1, 0.0)?;
        Ok(Self(Law::Linear { u1 }))
    }

    /// The quadratic law, `I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2`.
    ///
    /// # Errors
    ///
    /// - [`LimbDarkeningError::NonFinite`] if `u1` or `u2` is NaN or infinite.
    /// - [`LimbDarkeningError::NegativeIntensity`] if the intensity is
    ///   negative anywhere on the disk.
    pub fn quadratic(u1: f64, u2: f64) -> Result<Self, LimbDarkeningError> {
        check_intensity(u1, u2)?;
        Ok(Self(Law::Quadratic { u1, u2 }))
    }

    /// The law as quadratic coefficients `(u1, u2)`: `(0, 0)` for uniform and
    /// `(u1, 0)` for linear.
    pub fn quadratic_coefficients(self) -> (f64, f64) {
        match self.0 {
            Law::Uniform => (0.0, 0.0),
            Law::Linear { u1 } => (u1, 0.0),
            Law::Quadratic { u1, u2 } => (u1, u2),
        }
    }
}

// Checks that `I(x) = 1 - u1 x - u2 x^2` is not negative for `x = 1 - mu` in
// `[0, 1]`. `I(0) = 1`, so the minimum is at the edge (`x = 1`) unless the
// curve opens upwards (`u2 < 0`) with its lowest point inside the range.
fn check_intensity(u1: f64, u2: f64) -> Result<(), LimbDarkeningError> {
    if !u1.is_finite() || !u2.is_finite() {
        return Err(LimbDarkeningError::NonFinite);
    }

    let negative_at_edge = u1 + u2 > 1.0;

    // Lowest point at x = -u1 / (2 u2), with value 1 + u1^2 / (4 u2).
    let lowest_inside = u2 < 0.0 && u1 > 0.0 && u1 < -2.0 * u2;
    let negative_inside = lowest_inside && u1 * u1 + 4.0 * u2 > 0.0;

    if negative_at_edge || negative_inside {
        return Err(LimbDarkeningError::NegativeIntensity);
    }

    Ok(())
}

/// Everything needed to model one planet's transit on a circular orbit.
///
/// Each value is checked when its own type is made; [`new`](Self::new) adds the
/// checks that need more than one value. Once built, a `TransitParams` is always
/// valid, so the model code does not check it again.
///
/// # Examples
///
/// ```
/// use rustellar_core::{Epoch, TimeDelta};
/// use rustellar_model::{
///     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
/// };
///
/// let params = TransitParams::new(
///     Epoch::from_btjd(0.0)?,
///     TimeDelta::from_days(3.0)?,
///     RadiusRatio::new(0.1)?,
///     ScaledSemiMajorAxis::new(10.0)?,
///     ImpactParameter::new(0.3)?,
///     LimbDarkening::quadratic(0.3, 0.25)?,
/// )?;
/// assert_eq!(params.b().get(), 0.3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitParams {
    t0: Epoch,
    period: TimeDelta,
    k: RadiusRatio,
    a_rs: ScaledSemiMajorAxis,
    b: ImpactParameter,
    ld: LimbDarkening,
}

impl TransitParams {
    /// Builds transit parameters.
    ///
    /// `t0` is the time of mid-transit. `b` may be larger than `1 + k`: that is
    /// a valid orbit where the planet misses the star.
    ///
    /// # Errors
    ///
    /// The checks run in this order, and the first one that fails is returned:
    ///
    /// 1. [`ParamError::PeriodNotPositive`] if `period` is zero or negative.
    /// 2. [`ParamError::ImpactParameterTooLarge`] if `b` is not less than
    ///    `a_rs`, since no inclination gives that orbit.
    pub fn new(
        t0: Epoch,
        period: TimeDelta,
        k: RadiusRatio,
        a_rs: ScaledSemiMajorAxis,
        b: ImpactParameter,
        ld: LimbDarkening,
    ) -> Result<Self, ParamError> {
        if period.as_days() <= 0.0 {
            return Err(ParamError::PeriodNotPositive);
        }

        if b.get() >= a_rs.get() {
            return Err(ParamError::ImpactParameterTooLarge);
        }

        Ok(Self {
            t0,
            period,
            k,
            a_rs,
            b,
            ld,
        })
    }

    /// Builds transit parameters from the orbital inclination instead of `b`.
    ///
    /// `inc_deg` is in degrees, where 90 is edge-on (a central transit). The
    /// impact parameter is `b = a_rs * cos(inc)`.
    ///
    /// # Errors
    ///
    /// - [`ParamError::NonFinite`] if `inc_deg` is NaN or infinite.
    /// - [`ParamError::InclinationOutOfRange`] if `inc_deg` is not greater than
    ///   0 and at most 90.
    /// - Any error from [`new`](Self::new).
    pub fn from_inclination_deg(
        t0: Epoch,
        period: TimeDelta,
        k: RadiusRatio,
        a_rs: ScaledSemiMajorAxis,
        inc_deg: f64,
        ld: LimbDarkening,
    ) -> Result<Self, ParamError> {
        if !inc_deg.is_finite() {
            return Err(ParamError::NonFinite {
                name: "inclination",
            });
        }

        if inc_deg <= 0.0 || inc_deg > 90.0 {
            return Err(ParamError::InclinationOutOfRange);
        }

        // Built directly: for an inclination in (0, 90], the cosine is positive
        // and finite, so `b` already meets `ImpactParameter`'s rules.
        let b = ImpactParameter(a_rs.get() * inc_deg.to_radians().cos());

        Self::new(t0, period, k, a_rs, b, ld)
    }

    /// Time of mid-transit.
    pub fn t0(self) -> Epoch {
        self.t0
    }

    /// Orbital period.
    pub fn period(self) -> TimeDelta {
        self.period
    }

    /// Planet to star radius ratio.
    pub fn k(self) -> RadiusRatio {
        self.k
    }

    /// Semi-major axis in stellar radii.
    pub fn a_rs(self) -> ScaledSemiMajorAxis {
        self.a_rs
    }

    /// Impact parameter in stellar radii.
    pub fn b(self) -> ImpactParameter {
        self.b
    }

    /// Orbital inclination in degrees, from `b` and `a_rs`. 90 is edge-on.
    pub fn inclination_deg(self) -> f64 {
        (self.b.get() / self.a_rs.get()).acos().to_degrees()
    }

    /// Limb-darkening law of the star.
    pub fn limb_darkening(self) -> LimbDarkening {
        self.ld
    }
}
