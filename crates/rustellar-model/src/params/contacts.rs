use rustellar_core::TimeDelta;

/// The four contact points of a transit, as offsets from mid-transit.
///
/// First and fourth contact are when the planet's edge touches the star's
/// edge from outside; second and third are when the planet is just fully
/// inside the stellar disk. The orbit is symmetric, so `t1 = -t4` and
/// `t2 = -t3`.
///
/// Made by [`TransitParams::contacts`](crate::TransitParams::contacts).
#[derive(Debug, Clone, Copy, PartialEq)]
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
