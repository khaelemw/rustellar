//! What a search found: the best fit at each period, the spectra, and the
//! best candidate with what is known about it.

use rustellar_bls::Device;
use rustellar_core::{Epoch, TimeDelta};

/// A time in days as an [`Epoch`]; the light curve's times are finite, and
/// so is every time the search derives from them.
pub(crate) fn epoch(btjd: f64) -> Epoch {
    Epoch::from_btjd(btjd).expect("a finite time")
}

/// A span in days as a [`TimeDelta`].
pub(crate) fn span(days: f64) -> TimeDelta {
    TimeDelta::from_days(days).expect("a finite span")
}

/// The best fit of the template at one trial period.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PeriodFit {
    /// The trial period.
    pub period: TimeDelta,
    /// The best fit's duration, from first to last contact.
    pub duration: TimeDelta,
    /// The best fit's mid-transit time: the first within one period after
    /// the light curve's first time.
    pub transit_time: Epoch,
    /// The best fit's depth: the dimming at mid-transit. 0 where no fit
    /// dims (the other values then those of the first trial).
    pub depth: f64,
    /// The best fit's chi-squared; the flat model's where no fit dims.
    pub chi2: f64,
}

/// The spectra of a search, indexed like [`TlsResult::periods`], as TLS
/// defines them.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Spectra {
    /// The signal residue `min(chi2) / chi2`: 1 at the best period, in
    /// `(0, 1]`.
    pub sr: Vec<f64>,
    /// SR less its mean, scaled so that its peak is `sde_raw`.
    pub power_raw: Vec<f64>,
    /// `power_raw` less its running median, scaled so that its peak is
    /// `sde`: the spectrum to read peaks from.
    pub power: Vec<f64>,
    /// `(1 - mean(SR)) / std(SR)`.
    pub sde_raw: f64,
    /// The signal detection efficiency: the peak of `power` over its
    /// standard deviation. TLS suggests 7 or more for a detection in white
    /// noise; see [`SignalToNoise`] for correlated noise.
    pub sde: f64,
}

/// Standard errors of the candidate's parameters, from the chi-squared's
/// curvature (the Fisher matrix): right where the signal is strong, lower
/// bounds where it is weak.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Uncertainties {
    /// Of the period.
    pub period: TimeDelta,
    /// Of the mid-transit time.
    pub transit_time: TimeDelta,
    /// Of the duration.
    pub duration: TimeDelta,
    /// Of the depth.
    pub depth: f64,
}

/// One transit's own depth: the template's depth fitted to its points
/// alone.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TransitDepth {
    /// The transit's number, counted from the candidate's transit time.
    pub number: i64,
    /// Its mid-transit time.
    pub time: Epoch,
    /// Its depth: the dimming at mid-transit.
    pub depth: f64,
    /// The depth's standard error.
    pub error: f64,
    /// Its points in transit.
    pub points: usize,
}

/// The odd and even transits' depths, each set fitted alone: an eclipsing
/// binary at twice the period shows two depths.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OddEven {
    /// The odd transits' depth, each weighted by its error.
    pub odd: f64,
    /// Its standard error.
    pub odd_error: f64,
    /// The even transits' depth (the candidate's own transit is number 0,
    /// even).
    pub even: f64,
    /// Its standard error.
    pub even_error: f64,
    /// How many combined standard errors apart the two depths are.
    pub sigma: f64,
}

/// The candidate's signal to noise, for white noise and for noise
/// correlated over the transit's duration.
///
/// `white` is TLS's: the mean dimming in transit over the scatter out of
/// it, times the root of the points in transit, as if every point were
/// independent. `pink` (Pont, Zucker & Queloz 2006) adds the correlated
/// part, measured from the scatter of means of out-of-transit runs as long
/// as a transit, which no number of points in one transit averages away.
/// On injected transits its threshold for a given false-alarm rate is the
/// same in white and correlated noise, where `white`'s is not.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SignalToNoise {
    /// For white noise, as TLS's.
    pub white: f64,
    /// For noise correlated over the duration.
    pub pink: f64,
    /// The scatter of the points out of transit.
    pub white_noise: f64,
    /// The correlated noise on the transit's timescale.
    pub red_noise: f64,
    /// The mean dimming in transit, from the level out of transit.
    pub depth: f64,
}

/// The candidate's transit with the planet's size and impact parameter
/// fitted, where the template fixes them: its duration and depth without
/// the template's bias.
///
/// Where the cadence resolves ingress and egress (2-minute points, a strong
/// transit) these are several times closer to the truth than the
/// template's. Where it does not (30-minute points, small planets), `k` and
/// `b` trade against each other, and their errors, from the curvature, can
/// be far too small.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TransitShape {
    /// The period.
    pub period: TimeDelta,
    /// The mid-transit time.
    pub transit_time: Epoch,
    /// From first to last contact.
    pub duration: TimeDelta,
    /// The planet's radius over the star's.
    pub k: f64,
    /// The impact parameter, in stellar radii.
    pub b: f64,
    /// The dimming at mid-transit.
    pub depth: f64,
    /// The orbit's radius in stellar radii, for a circular orbit.
    pub a: f64,
    /// The chi-squared of the fit.
    pub chi2: f64,
    /// Standard errors from the curvature, where it fixes them all.
    pub errors: Option<ShapeUncertainties>,
}

/// Standard errors of a [`TransitShape`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ShapeUncertainties {
    /// Of the period.
    pub period: TimeDelta,
    /// Of the mid-transit time.
    pub transit_time: TimeDelta,
    /// Of the duration.
    pub duration: TimeDelta,
    /// Of the radius ratio.
    pub k: f64,
    /// Of the impact parameter.
    pub b: f64,
}

/// The best candidate: the highest peaks of the spectrum refined, the best
/// of them by chi-squared, and what is known about it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Candidate {
    /// The period.
    pub period: TimeDelta,
    /// Its mid-transit time: the first after the light curve's first time
    /// less half a period.
    pub transit_time: Epoch,
    /// From first to last contact.
    pub duration: TimeDelta,
    /// The template's depth: the dimming at mid-transit.
    pub depth: f64,
    /// The chi-squared of the fit.
    pub chi2: f64,
    /// The points the template covers.
    pub points_in_transit: usize,
    /// The half width at half maximum of the spectrum's highest peak
    /// (TLS's `period_uncertainty`), or `None` where the peak does not fall
    /// to half on both sides within the periods searched.
    pub period_uncertainty: Option<TimeDelta>,
    /// Standard errors, where the curvature fixes them.
    pub errors: Option<Uncertainties>,
    /// Each transit with points, in time order.
    pub transits: Vec<TransitDepth>,
    /// The odd and even transits compared, where there are both.
    pub odd_even: Option<OddEven>,
    /// The signal to noise, where there are at least two points in transit
    /// and out of it.
    pub snr: Option<SignalToNoise>,
    /// The transit with the planet's size and impact parameter free.
    pub shape: Option<TransitShape>,
}

/// The result of a search.
///
/// # Examples
///
/// See [`Tls::search`](crate::Tls::search).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TlsResult {
    pub(crate) periods: Vec<PeriodFit>,
    pub(crate) spectra: Option<Spectra>,
    pub(crate) best: Option<Candidate>,
    pub(crate) gpu: Option<usize>,
}

impl TlsResult {
    /// The best fit at each trial period, in increasing period.
    #[must_use]
    pub fn periods(&self) -> &[PeriodFit] {
        &self.periods
    }

    /// The spectra, or `None` where no period has a fit that dims.
    #[must_use]
    pub fn spectra(&self) -> Option<&Spectra> {
        self.spectra.as_ref()
    }

    /// The best candidate, or `None` where no period has a fit that dims.
    #[must_use]
    pub fn best(&self) -> Option<&Candidate> {
        self.best.as_ref()
    }

    /// Where the search ran: [`Device::Gpu`] with the GPU's index, or
    /// [`Device::Cpu`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::{Device, TlsResult};
    ///
    /// assert_eq!(TlsResult::default().device(), Device::Cpu);
    /// ```
    #[must_use]
    pub fn device(&self) -> Device {
        self.gpu.map_or(Device::Cpu, Device::Gpu)
    }
}
