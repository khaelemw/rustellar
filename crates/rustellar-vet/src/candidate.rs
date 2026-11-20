use rustellar_core::{Epoch, TimeDelta};

use crate::VetError;

/// A transit candidate to vet: a period, a mid-transit time and a duration
/// from any search, with its depth if the search gave one.
///
/// Built by [`Candidate::new`], which checks it, or converted from a search's
/// result: `rustellar-bls`'s [`BestBox`](rustellar_bls::BestBox) with the
/// `bls` feature, `rustellar-tls`'s [`Candidate`](rustellar_tls::Candidate)
/// with the `tls` feature.
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, TimeDelta};
///
/// let candidate = Candidate::new(
///     TimeDelta::from_days(4.7)?,
///     Epoch::from_btjd(1326.3)?,
///     TimeDelta::from_hours(2.9)?,
/// )?
/// .with_depth(1e-3)?;
/// assert_eq!(candidate.period().as_days(), 4.7);
/// assert_eq!(candidate.depth(), Some(1e-3));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    period: TimeDelta,
    transit_time: Epoch,
    duration: TimeDelta,
    depth: Option<f64>,
}

impl Candidate {
    /// A candidate with no depth.
    ///
    /// # Errors
    ///
    /// - [`VetError::NotPositive`] if `period` or `duration` is zero or
    ///   negative.
    /// - [`VetError::DurationNotShorterThanPeriod`] if `duration` is at least
    ///   `period`.
    pub fn new(
        period: TimeDelta,
        transit_time: Epoch,
        duration: TimeDelta,
    ) -> Result<Self, VetError> {
        if period.as_days() <= 0.0 {
            return Err(VetError::NotPositive { name: "period" });
        }
        if duration.as_days() <= 0.0 {
            return Err(VetError::NotPositive { name: "duration" });
        }
        if duration.as_days() >= period.as_days() {
            return Err(VetError::DurationNotShorterThanPeriod);
        }
        Ok(Self {
            period,
            transit_time,
            duration,
            depth: None,
        })
    }

    /// The same candidate with `depth`, the fractional dimming in transit
    /// (positive for a dip).
    ///
    /// # Errors
    ///
    /// [`VetError::DepthNotFinite`] if `depth` is NaN or infinite.
    pub fn with_depth(self, depth: f64) -> Result<Self, VetError> {
        if !depth.is_finite() {
            return Err(VetError::DepthNotFinite);
        }
        Ok(Self {
            depth: Some(depth),
            ..self
        })
    }

    /// The period.
    #[must_use]
    pub const fn period(&self) -> TimeDelta {
        self.period
    }

    /// A mid-transit time; any transit of the candidate will do.
    #[must_use]
    pub const fn transit_time(&self) -> Epoch {
        self.transit_time
    }

    /// The duration, from first to last contact.
    #[must_use]
    pub const fn duration(&self) -> TimeDelta {
        self.duration
    }

    /// The depth the search gave, if any.
    #[must_use]
    pub const fn depth(&self) -> Option<f64> {
        self.depth
    }
}

/// The best box of a `rustellar-bls` search: its period, mid-transit time,
/// duration and depth.
#[cfg(feature = "bls")]
#[cfg_attr(docsrs, doc(cfg(feature = "bls")))]
impl TryFrom<&rustellar_bls::BestBox> for Candidate {
    type Error = VetError;

    fn try_from(best: &rustellar_bls::BestBox) -> Result<Self, VetError> {
        let t = &best.transit;
        Self::new(t.period(), t.transit_time(), t.duration())?.with_depth(best.fit.depth)
    }
}

/// The best candidate of a `rustellar-tls` search: its period, mid-transit
/// time, duration and depth.
#[cfg(feature = "tls")]
#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
impl TryFrom<&rustellar_tls::Candidate> for Candidate {
    type Error = VetError;

    fn try_from(best: &rustellar_tls::Candidate) -> Result<Self, VetError> {
        Self::new(best.period, best.transit_time, best.duration)?.with_depth(best.depth)
    }
}
