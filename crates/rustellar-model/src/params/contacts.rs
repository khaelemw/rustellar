use rustellar_core::TimeDelta;

/// The four contact points of a transit, as offsets from mid-transit.
///
/// First and fourth contact are when the planet's edge touches the star's
/// edge from outside; second and third are when the planet is just fully
/// inside the stellar disk. The orbit is symmetric, so `t1 = -t4` and
/// `t2 = -t3`.
///
/// Made by [`TransitParams::contacts`](crate::TransitParams::contacts).
///
/// More fields may be added in minor releases, so it cannot be built or
/// matched exhaustively outside this crate.
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
/// assert_eq!(contacts.t1, -contacts.t4);
/// assert!(contacts.t2.is_some()); // not grazing
/// assert_eq!(contacts.t4 - contacts.t1, params.t14().unwrap());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Contacts {
    /// First contact: the transit starts. Negative.
    pub t1: TimeDelta,
    /// Second contact: the planet is fully inside the disk. Negative, or
    /// `None` for a grazing transit.
    pub t2: Option<TimeDelta>,
    /// Third contact: the planet starts to leave the disk. Positive, or `None`
    /// for a grazing transit.
    pub t3: Option<TimeDelta>,
    /// Fourth contact: the transit ends. Positive.
    pub t4: TimeDelta,
}
