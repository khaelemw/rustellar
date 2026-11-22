//! Detrending around a search's candidate: detrend, search, mask the
//! candidate's transits, detrend again (and, if asked, search again, until
//! the search finds the same transits).

use std::num::NonZeroUsize;

use rustellar_core::{Epoch, LightCurve, TimeDelta};

use crate::{Detrend, DetrendError, Flattened, transit_mask};

/// How much longer than the candidate's duration [`Masking`]'s mask is, by
/// default. Searches report durations from a coarse grid and mid-transit
/// times a few points off, so a mask the transit's own length would leave
/// its ingress or egress in some windows.
pub const MASK_PER_DURATION: f64 = 1.5;

/// How long the windows are around the mask, in candidate durations, by
/// default: long enough to keep a few unmasked points either side of the
/// mask, short enough to follow the star's variability.
pub const MASKED_WINDOW_PER_DURATION: f64 = 2.5;

/// A transit a search found: its period, a mid-transit time and its
/// duration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// The period.
    pub period: TimeDelta,
    /// A mid-transit time.
    pub t0: Epoch,
    /// The duration, first to last contact.
    pub duration: TimeDelta,
}

/// The settings of [`Detrend::flatten_around`]: the mask's and the
/// windows' lengths, in candidate durations, and the most searches.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use rustellar_detrend::Masking;
///
/// let masking = Masking::default()
///     .with_mask_per_duration(2.0)
///     .with_window_per_duration(3.0)
///     .with_rounds(NonZeroUsize::new(3).unwrap());
/// assert_eq!(masking.rounds().get(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Masking {
    mask_per_duration: f64,
    window_per_duration: f64,
    rounds: NonZeroUsize,
}

impl Default for Masking {
    /// [`MASK_PER_DURATION`], [`MASKED_WINDOW_PER_DURATION`] and one
    /// round: on transits over spotted stars, searching again after the
    /// mask kept no more of the signal and moved some right periods to
    /// wrong ones.
    fn default() -> Self {
        Self {
            mask_per_duration: MASK_PER_DURATION,
            window_per_duration: MASKED_WINDOW_PER_DURATION,
            rounds: NonZeroUsize::MIN,
        }
    }
}

impl Masking {
    /// With the mask `factor` times the candidate's duration.
    #[must_use]
    pub fn with_mask_per_duration(mut self, factor: f64) -> Self {
        self.mask_per_duration = factor;
        self
    }

    /// With windows (or knot spacing) `factor` times the candidate's
    /// duration once its transits are masked. Longer than the mask, so
    /// every masked point keeps unmasked neighbours in its window.
    #[must_use]
    pub fn with_window_per_duration(mut self, factor: f64) -> Self {
        self.window_per_duration = factor;
        self
    }

    /// With at most `rounds` searches.
    #[must_use]
    pub fn with_rounds(mut self, rounds: NonZeroUsize) -> Self {
        self.rounds = rounds;
        self
    }

    /// The mask's length in candidate durations.
    #[must_use]
    pub fn mask_per_duration(&self) -> f64 {
        self.mask_per_duration
    }

    /// The windows' length in candidate durations, once masked.
    #[must_use]
    pub fn window_per_duration(&self) -> f64 {
        self.window_per_duration
    }

    /// The most searches.
    #[must_use]
    pub fn rounds(&self) -> NonZeroUsize {
        self.rounds
    }
}

/// A light curve flattened around the transits its search found, from
/// [`Detrend::flatten_around`].
#[derive(Debug, Clone, PartialEq)]
pub struct Masked {
    flattened: Flattened,
    candidate: Option<Candidate>,
    mask: Option<Vec<bool>>,
    searches: usize,
    converged: bool,
}

impl Masked {
    /// The light curve flattened with the last mask (or with none, if the
    /// first search found nothing).
    #[must_use]
    pub fn flattened(&self) -> &Flattened {
        &self.flattened
    }

    /// The flattened light curve, taken out.
    #[must_use]
    pub fn into_flattened(self) -> Flattened {
        self.flattened
    }

    /// The candidate whose transits were masked last, if any.
    #[must_use]
    pub fn candidate(&self) -> Option<Candidate> {
        self.candidate
    }

    /// The last mask, one entry per point of the original light curve.
    #[must_use]
    pub fn mask(&self) -> Option<&[bool]> {
        self.mask.as_deref()
    }

    /// How many times the search ran.
    #[must_use]
    pub fn searches(&self) -> usize {
        self.searches
    }

    /// Whether the last search found nothing or the transits already
    /// masked (the same mask), rather than running out of rounds or
    /// finding a mask that leaves no point.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.converged
    }
}

impl Detrend {
    /// `lc` flattened around the transits `search` finds in it, so the
    /// trend passes under them and leaves their whole depth.
    ///
    /// Flattens `lc` with these settings, then up to
    /// [`rounds`](Masking::rounds) times: searches the flattened curve;
    /// stops if it finds nothing, or the same transits as last time (the
    /// same mask), or a mask that leaves no point (one at least as long as
    /// the period: then the last flattening stands); otherwise masks the
    /// candidate's transits over
    /// [`mask_per_duration`](Masking::mask_per_duration) of its duration
    /// ([`transit_mask`]) and flattens `lc` again, with windows (or knot
    /// spacing) [`window_per_duration`](Masking::window_per_duration) of its
    /// duration.
    ///
    /// By default once: the search runs on the blind flattening, and its
    /// candidate is measured on the masked one. A search on a curve masked
    /// around a candidate favours that candidate, so it confirms the
    /// candidate rather than finds another; more rounds only refine its
    /// mid-transit time and duration.
    ///
    /// `search` is any transit search, such as a BLS or TLS one, returning
    /// its best candidate or `None`.
    ///
    /// # Errors
    ///
    /// Those of [`flatten`](Self::flatten) and [`transit_mask`] (a
    /// candidate's period or duration not positive), and
    /// [`MaskNotInWindow`](DetrendError::MaskNotInWindow) if the mask factor
    /// is not positive or not below the window factor (checked first).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_detrend::{Candidate, Detrend, Epoch, LightCurve, Masking, TimeDelta};
    ///
    /// // Every 10 minutes for 4 days, with a transit every 2 days from day
    /// // 0.5, on a straight trend.
    /// let time: Vec<f64> = (0..576).map(|i| f64::from(i) / 144.0).collect();
    /// let flux = time
    ///     .iter()
    ///     .map(|&t| {
    ///         let dip = ((t - 0.5) / 2.0 + 0.05).rem_euclid(1.0) < 0.1;
    ///         (1.0 + 1e-3 * t) * if dip { 0.99 } else { 1.0 }
    ///     })
    ///     .collect();
    /// let lc = LightCurve::from_days(time, flux, vec![1e-3; 576])?;
    ///
    /// // A stand-in for a real search: always the same transit.
    /// let search = |_: &LightCurve| {
    ///     Some(Candidate {
    ///         period: TimeDelta::from_days(2.0).ok()?,
    ///         t0: Epoch::from_days(0.5).ok()?,
    ///         duration: TimeDelta::from_days(0.2).ok()?,
    ///     })
    /// };
    /// let detrend = Detrend::default().with_window(TimeDelta::from_days(0.5)?);
    /// let done = detrend.flatten_around(&lc, &Masking::default(), search)?;
    /// assert_eq!(done.searches(), 1);
    /// assert_eq!(done.mask().map(|m| m[72]), Some(true));
    /// // Mid-transit keeps its whole depth.
    /// let flux = done.flattened().light_curve().flux();
    /// assert!((flux[72] - 0.99).abs() < 1e-9);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flatten_around<F>(
        &self,
        lc: &LightCurve,
        masking: &Masking,
        mut search: F,
    ) -> Result<Masked, DetrendError>
    where
        F: FnMut(&LightCurve) -> Option<Candidate>,
    {
        let (mask_factor, window_factor) = (masking.mask_per_duration, masking.window_per_duration);
        // Written so NaN fails too.
        if !(mask_factor > 0.0 && mask_factor < window_factor) {
            return Err(DetrendError::MaskNotInWindow);
        }
        let mut done = Masked {
            flattened: self.flatten(lc, None)?,
            candidate: None,
            mask: None,
            searches: 0,
            converged: false,
        };
        for _ in 0..masking.rounds.get() {
            done.searches += 1;
            let Some(candidate) = search(done.flattened.light_curve()) else {
                done.converged = true;
                break;
            };
            let mask = transit_mask(
                lc,
                candidate.period,
                candidate.t0,
                candidate.duration * mask_factor,
            )?;
            if done.mask.as_ref() == Some(&mask) {
                done.converged = true;
                break;
            }
            // A mask as long as the period leaves nothing to follow the
            // star with: keep the last flattening.
            if mask.iter().all(|&m| m) {
                break;
            }
            done.flattened = self
                .with_window(candidate.duration * window_factor)
                .flatten(lc, Some(&mask))?;
            done.candidate = Some(candidate);
            done.mask = Some(mask);
        }
        Ok(done)
    }
}
