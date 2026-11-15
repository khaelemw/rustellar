use std::f64::consts::PI;

use rustellar_core::{Epoch, TimeDelta};

use super::{Contacts, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis};
use crate::{
    Exposure,
    error::{LengthMismatch, ParamError},
    flux,
};

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
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{Epoch, TimeDelta};
    /// use rustellar_model::{
    ///     ImpactParameter, LimbDarkening, ParamError, RadiusRatio, ScaledSemiMajorAxis,
    ///     TransitParams,
    /// };
    ///
    /// let result = TransitParams::new(
    ///     Epoch::from_btjd(0.0)?,
    ///     TimeDelta::from_days(3.0)?,
    ///     RadiusRatio::new(0.1)?,
    ///     ScaledSemiMajorAxis::new(10.0)?,
    ///     ImpactParameter::new(12.0)?, // farther than the orbit itself
    ///     LimbDarkening::uniform(),
    /// );
    /// assert_eq!(result, Err(ParamError::ImpactParameterTooLarge));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{Epoch, TimeDelta};
    /// use rustellar_model::{LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams};
    ///
    /// let params = TransitParams::from_inclination_deg(
    ///     Epoch::from_btjd(0.0)?,
    ///     TimeDelta::from_days(3.0)?,
    ///     RadiusRatio::new(0.1)?,
    ///     ScaledSemiMajorAxis::new(10.0)?,
    ///     88.0,
    ///     LimbDarkening::quadratic(0.3, 0.25)?,
    /// )?;
    /// // b = a_rs cos(i)
    /// assert!((params.b().get() - 10.0 * 88_f64.to_radians().cos()).abs() < 1e-15);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// assert_eq!(params.t0().as_btjd(), 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn t0(self) -> Epoch {
        self.t0
    }

    /// Orbital period.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// assert_eq!(params.period().as_days(), 3.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn period(self) -> TimeDelta {
        self.period
    }

    /// Planet to star radius ratio.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// // The depth is roughly k squared.
    /// let depth = params.k().get().powi(2);
    /// assert!((depth - 0.01).abs() < 1e-15);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn k(self) -> RadiusRatio {
        self.k
    }

    /// Semi-major axis in stellar radii.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// assert_eq!(params.a_rs().get(), 10.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn a_rs(self) -> ScaledSemiMajorAxis {
        self.a_rs
    }

    /// Impact parameter in stellar radii.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// assert_eq!(params.b().get(), 0.3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn b(self) -> ImpactParameter {
        self.b
    }

    /// Orbital inclination in degrees, from `b` and `a_rs`. 90 is edge-on.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// // cos(i) = b / a_rs = 0.03
    /// assert!((params.inclination_deg() - 88.281).abs() < 1e-3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn inclination_deg(self) -> f64 {
        (self.b.get() / self.a_rs.get()).acos().to_degrees()
    }

    /// Limb-darkening law of the star.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// assert_eq!(params.limb_darkening().quadratic_coefficients(), (0.3, 0.25));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn limb_darkening(self) -> LimbDarkening {
        self.ld
    }

    /// Orbital phase angle at time `t`, in radians.
    ///
    /// In `[-pi, pi)`, with 0 at mid-transit and `-pi` half an orbit away.
    /// Times many orbits from `t0` keep full precision, because whole orbits
    /// are removed before scaling to radians.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// use std::f64::consts::PI;
    ///
    /// assert_eq!(params.phase_angle(Epoch::from_btjd(0.0)?), 0.0);
    /// // A quarter of the way round the 3-day orbit.
    /// assert!((params.phase_angle(Epoch::from_btjd(0.75)?) - PI / 2.0).abs() < 1e-15);
    /// // Half an orbit away is -pi, never +pi.
    /// assert_eq!(params.phase_angle(Epoch::from_btjd(1.5)?), -PI);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn phase_angle(self, t: Epoch) -> f64 {
        2.0 * PI * (((t - self.t0) / self.period + 0.5).rem_euclid(1.0) - 0.5)
    }

    /// Distance between the planet and star centres on the sky at time `t`, in
    /// stellar radii.
    ///
    /// Equal to `b` at mid-transit. The value is the same half an orbit later,
    /// when the planet is behind the star, so check [`in_front`](Self::in_front)
    /// before treating a small separation as a transit.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// // At mid-transit the separation is the impact parameter.
    /// assert!((params.separation(Epoch::from_btjd(0.0)?) - 0.3).abs() < 1e-15);
    /// // A quarter orbit later the planet is a_rs from the star on the sky.
    /// assert!((params.separation(Epoch::from_btjd(0.75)?) - 10.0).abs() < 1e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn separation(self, t: Epoch) -> f64 {
        let (sin_phi, cos_phi) = self.phase_angle(t).sin_cos();
        let a = self.a_rs.get();
        let b = self.b.get();

        // z = sqrt(a^2 sin^2(phi) + b^2 cos^2(phi))
        (a * sin_phi).hypot(b * cos_phi)
    }

    /// Whether the planet is between the observer and the star at time `t`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// assert!(params.in_front(Epoch::from_btjd(0.0)?));
    /// assert!(!params.in_front(Epoch::from_btjd(1.5)?)); // behind the star
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn in_front(self, t: Epoch) -> bool {
        self.phase_angle(t).cos() > 0.0
    }

    /// Total transit duration, from first to fourth contact.
    ///
    /// `None` if the planet misses the star (`b >= 1 + k`). If the orbit is so
    /// close that the planet overlaps the star for its whole near-side half
    /// orbit (`a_rs < 1 + k`), this is half the period.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// let t14 = params.t14().expect("this planet transits");
    /// assert!((t14.as_hours() - 2.431).abs() < 1e-3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn t14(self) -> Option<TimeDelta> {
        let edge = 1.0 + self.k.get();

        if self.b.get() >= edge {
            return None;
        }

        self.contact_duration(edge)
    }

    /// Duration of the flat bottom, from second to third contact, when the
    /// planet is fully inside the stellar disk.
    ///
    /// `None` for a grazing transit (`b >= 1 - k`), which also covers a planet
    /// at least as large as the star (`k >= 1`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// let t23 = params.t23().expect("not grazing");
    /// assert!(t23 < params.t14().unwrap());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn t23(self) -> Option<TimeDelta> {
        let edge = 1.0 - self.k.get();

        if self.b.get() >= edge {
            return None;
        }

        self.contact_duration(edge)
    }

    // Time between the two points where the separation equals `edge`
    // (Winn 2010, eqs. 14 and 15). Callers have checked `b < edge`, so the
    // value under the square root is positive.
    fn contact_duration(self, edge: f64) -> Option<TimeDelta> {
        let a = self.a_rs.get();
        let b = self.b.get();

        // Above 1 only when a_rs < 1 + k; asin(1) gives half the period.
        let ratio = ((edge * edge - b * b) / (a * a - b * b)).min(1.0);
        let days = self.period.as_days() / PI * ratio.sqrt().asin();

        TimeDelta::from_days(days).ok()
    }

    /// The four contact points of the transit, as offsets from mid-transit.
    ///
    /// `None` if the planet misses the star. Second and third contact are
    /// `None` for a grazing transit, as for [`t23`](Self::t23).
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// let contacts = params.contacts().expect("this planet transits");
    /// let ingress = contacts.t2.unwrap() - contacts.t1;
    /// assert!(ingress.as_hours() > 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn contacts(self) -> Option<Contacts> {
        let half_t14 = self.t14()? * 0.5;
        let half_t23 = self.t23().map(|t23| t23 * 0.5);

        Some(Contacts {
            t1: -half_t14,
            t2: half_t23.map(|half| -half),
            t3: half_t23,
            t4: half_t14,
        })
    }

    /// Mid-transit times from `start` to `end`, both included, in time order.
    ///
    /// Empty if no transit falls in the window, or if `end` is before `start`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::{Epoch, TimeDelta};
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// // A 27-day TESS sector starting at BTJD 1.
    /// let start = Epoch::from_btjd(1.0)?;
    /// let end = Epoch::from_btjd(28.0)?;
    /// let mid_times: Vec<f64> = params.transit_times(start, end).map(|t| t.as_btjd()).collect();
    /// assert_eq!(mid_times, [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn transit_times(self, start: Epoch, end: Epoch) -> impl Iterator<Item = Epoch> {
        let t0 = self.t0;
        let period = self.period;

        // Orbit numbers that could fall in the window, widened by one at each
        // end. Rounding in the division can put a transit lying exactly on
        // `start` or `end` just outside the range, so the filter below decides.
        let first = ((start - t0) / period).floor() as i64;
        let last = ((end - t0) / period).ceil() as i64;

        (first..=last)
            .map(move |n| t0 + period * n as f64)
            .filter(move |&t| start <= t && t <= end)
    }

    /// Flux of the star at time `t`, relative to its flux out of transit.
    ///
    /// 1 when the planet is clear of the star or behind it. Otherwise the
    /// analytic [`flux`] at the planet's separation.
    ///
    /// # Accuracy
    ///
    /// Compared with batman 2.5.1 (Kreidberg 2015) at 25,160 times over 840
    /// transits with quadratic limb darkening (`k` from 0.005 to 0.29, `b`
    /// from 0 to 1.05, `a_rs` from 1.5 to 100), the largest difference is
    /// `2.1e-8`. That difference is batman's: at the points where it is
    /// largest, [`flux_ring_sum`](crate::flux_ring_sum) with 100,000 rings
    /// agrees with this result to `2e-15`, and batman's paper states 0.03 ppm
    /// (`3e-8`) for its quadratic model.
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
    ///
    /// let mid_transit = params.flux_at(Epoch::from_btjd(0.0)?);
    /// assert!((mid_transit - 0.988_527_81).abs() < 1e-8);
    ///
    /// // Half an orbit later the planet is behind the star.
    /// assert_eq!(params.flux_at(Epoch::from_btjd(1.5)?), 1.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn flux_at(self, t: Epoch) -> f64 {
        // The separation is the same half an orbit later, when the planet is
        // behind the star, so this check must come first.
        if !self.in_front(t) {
            return 1.0;
        }

        flux(self.separation(t), self.k, self.ld)
    }

    /// Flux at each of `times`, as for [`flux_at`](Self::flux_at), and with
    /// the same accuracy.
    ///
    /// The result has one value per time, in the same order. The times do not
    /// need to be sorted, so the times of a
    /// [`LightCurve`](rustellar_core::LightCurve) can be passed straight in.
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
    ///
    /// let times = [0.0, 0.02, 0.04, 0.06].map(|t| Epoch::from_btjd(t).unwrap());
    /// let flux = params.light_curve(&times);
    ///
    /// let expected = [0.988_527_81, 0.988_925_66, 0.991_248_45, 1.0];
    /// for (f, e) in flux.iter().zip(expected) {
    ///     assert!((f - e).abs() < 1e-8);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn light_curve(self, times: &[Epoch]) -> Vec<f64> {
        times.iter().map(|&t| self.flux_at(t)).collect()
    }

    /// Writes the flux at each of `times` into `out`, as for
    /// [`light_curve`](Self::light_curve), without allocating.
    ///
    /// For repeated evaluation, as in a fit or a sampler, reuse one buffer
    /// across calls, or see [`TransitModel`](crate::TransitModel).
    ///
    /// # Errors
    ///
    /// [`LengthMismatch`] if `out` is not the same length as `times`; `out`
    /// is then left unchanged.
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
    ///
    /// let times = [0.0, 0.06].map(|t| Epoch::from_btjd(t).unwrap());
    /// let mut flux = [0.0; 2];
    /// params.light_curve_into(&times, &mut flux)?;
    ///
    /// assert!((flux[0] - 0.988_527_81).abs() < 1e-8);
    /// assert_eq!(flux[1], 1.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn light_curve_into(self, times: &[Epoch], out: &mut [f64]) -> Result<(), LengthMismatch> {
        check_lengths(times, out)?;
        for (f, &t) in out.iter_mut().zip(times) {
            *f = self.flux_at(t);
        }
        Ok(())
    }

    /// Mean flux over an exposure whose middle is at `t`.
    ///
    /// The average of [`flux_at`](Self::flux_at) over the instants of
    /// `exposure` (the midpoint rule; see [`Exposure`]). With one sample this
    /// is exactly `flux_at(t)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    ///
    /// use rustellar_core::{Epoch, TimeDelta};
    /// use rustellar_model::{
    ///     Exposure, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis,
    ///     TransitParams,
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
    /// let exposure = Exposure::new(TimeDelta::from_hours(0.5)?, NonZeroUsize::new(15).unwrap())?;
    ///
    /// // Centred on fourth contact, the planet only touches the star, but half
    /// // the exposure is spent in egress.
    /// let t4 = params.t0() + params.contacts().unwrap().t4;
    /// assert!(params.flux_at(t4) > 1.0 - 1e-6);
    /// assert!(params.binned_flux_at(t4, exposure) < 1.0 - 1e-4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn binned_flux_at(self, t: Epoch, exposure: Exposure) -> f64 {
        let sum: f64 = exposure.offsets().map(|dt| self.flux_at(t + dt)).sum();
        sum / exposure.samples().get() as f64
    }
}

// The error for an output buffer that does not match the times.
pub(crate) fn check_lengths<T>(times: &[T], out: &[f64]) -> Result<(), LengthMismatch> {
    if times.len() == out.len() {
        Ok(())
    } else {
        Err(LengthMismatch {
            times: times.len(),
            output: out.len(),
        })
    }
}
