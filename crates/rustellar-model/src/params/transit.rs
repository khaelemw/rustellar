use std::{f64::consts::PI, num::NonZeroUsize};

use rustellar_core::{Epoch, TimeDelta};

use super::{Contacts, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis};
use crate::{error::ParamError, flux_ring_sum};

// Rings per point for the light curve, until the analytic formula replaces the
// ring sum. 1000 keeps typical planets within about 1e-9 of the exact flux.
// A `const` unwrap runs at compile time, so 0 here would fail to build.
const LIGHT_CURVE_RINGS: NonZeroUsize = NonZeroUsize::new(1000).unwrap();

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

    /// Orbital phase angle at time `t`, in radians.
    ///
    /// In `[-pi, pi)`, with 0 at mid-transit and `-pi` half an orbit away.
    /// Times many orbits from `t0` keep full precision, because whole orbits
    /// are removed before scaling to radians.
    pub fn phase_angle(self, t: Epoch) -> f64 {
        2.0 * PI * (((t - self.t0) / self.period + 0.5).rem_euclid(1.0) - 0.5)
    }

    /// Distance between the planet and star centres on the sky at time `t`, in
    /// stellar radii.
    ///
    /// Equal to `b` at mid-transit. The value is the same half an orbit later,
    /// when the planet is behind the star, so check [`in_front`](Self::in_front)
    /// before treating a small separation as a transit.
    pub fn separation(self, t: Epoch) -> f64 {
        let (sin_phi, cos_phi) = self.phase_angle(t).sin_cos();
        let a = self.a_rs.get();
        let b = self.b.get();

        // z = sqrt(a^2 sin^2(phi) + b^2 cos^2(phi))
        (a * sin_phi).hypot(b * cos_phi)
    }

    /// Whether the planet is between the observer and the star at time `t`.
    pub fn in_front(self, t: Epoch) -> bool {
        self.phase_angle(t).cos() > 0.0
    }

    /// Total transit duration, from first to fourth contact.
    ///
    /// `None` if the planet misses the star (`b >= 1 + k`). If the orbit is so
    /// close that the planet overlaps the star for its whole near-side half
    /// orbit (`a_rs < 1 + k`), this is half the period.
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
    /// 1 when the planet is clear of the star or behind it. For now the flux
    /// comes from [`flux_ring_sum`](crate::flux_ring_sum) with 1000 rings,
    /// which is within about `1e-9` of the exact value for typical planets;
    /// a faster analytic formula will replace it with the same results.
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
    pub fn flux_at(self, t: Epoch) -> f64 {
        // The separation is the same half an orbit later, when the planet is
        // behind the star, so this check must come first.
        if !self.in_front(t) {
            return 1.0;
        }

        flux_ring_sum(self.separation(t), self.k, self.ld, LIGHT_CURVE_RINGS)
    }

    /// Flux at each of `times`, as for [`flux_at`](Self::flux_at).
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
    pub fn light_curve(self, times: &[Epoch]) -> Vec<f64> {
        times.iter().map(|&t| self.flux_at(t)).collect()
    }
}
