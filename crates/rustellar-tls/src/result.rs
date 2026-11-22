//! What a search found: the best fit at each period, the spectra, and the
//! best candidate with what is known about it.

use rustellar_core::Device;
use rustellar_core::{Epoch, TimeDelta};

/// A time in days as an [`Epoch`]; the light curve's times are finite, and
/// so is every time the search derives from them.
pub(crate) fn epoch(days: f64) -> Epoch {
    Epoch::from_days(days).expect("a finite time")
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

/// Standard errors of the candidate's parameters: from the chi-squared's
/// curvature ([`Candidate::errors`]) or by bootstrap
/// ([`Candidate::bootstrap_errors`]).
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
    /// The template's, from first to last contact. Where the transit's
    /// ingress is longer than the template's (a large or grazing planet)
    /// this comes out short; [`Candidate::duration_estimate`] corrects it.
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
    /// Standard errors from the chi-squared's curvature (the Fisher matrix),
    /// where it fixes them all: right where the signal is strong, too small
    /// where it is weak (about 25% at signal to noise 12) or the noise is
    /// correlated.
    pub errors: Option<Uncertainties>,
    /// Standard errors by bootstrap: the spread of the fit repeated on its
    /// model plus its residuals resampled in blocks a transit long
    /// ([`SearchOptions::bootstrap`](crate::SearchOptions::bootstrap)
    /// times). The curvature's where the signal is strong; where it is weak
    /// or the noise is correlated, the fit's real scatter. On injected
    /// transits at signal to noise 8 to 12, the truth is within 2 of them
    /// 91% to 98% of the time, against 81% to 94% for the curvature's; in
    /// noise correlated over hours 95% to 97%, against 68% to 85%. `None`
    /// without the curvature's or resamples.
    pub bootstrap_errors: Option<Uncertainties>,
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

/// How much lower the shape's chi-squared must be than the template's for
/// [`Candidate::duration_estimate`] to take its duration.
pub const SHAPE_CHI2_GAIN: f64 = 1.0;

impl Candidate {
    /// The duration from first to last contact: the [`shape`](Self::shape)
    /// fit's where it lowers the chi-squared by more than
    /// [`SHAPE_CHI2_GAIN`], else the template's [`duration`](Self::duration).
    ///
    /// The template's shape is fixed, so on a transit with a longer ingress
    /// it fits a shorter duration: by 12% on radius ratios of 0.1 to 0.2
    /// and 26% on grazing transits. The shape fit removes that bias where
    /// the data tell the shapes apart; where they do not (a small planet
    /// at 30 minutes), its radius ratio and impact parameter trade against
    /// each other and its duration scatters more than the template's.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_model::{
    ///     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// };
    /// use rustellar_tls::{Epoch, LightCurve, Periods, SearchOptions, TimeDelta, Tls};
    ///
    /// // A planet of radius ratio 0.15 every 2.5 days, every 10 minutes.
    /// let planet = TransitParams::new(
    ///     Epoch::from_days(1.0)?,
    ///     TimeDelta::from_days(2.5)?,
    ///     RadiusRatio::new(0.15)?,
    ///     ScaledSemiMajorAxis::new(8.0)?,
    ///     ImpactParameter::new(0.5)?,
    ///     LimbDarkening::quadratic(0.4804, 0.1867)?,
    /// )?;
    /// let time: Vec<f64> = (0..2160).map(|i| f64::from(i) / 144.0).collect();
    /// let flux = time
    ///     .iter()
    ///     .map(|&t| planet.flux_at(Epoch::from_days(t).unwrap()))
    ///     .collect();
    /// let lc = LightCurve::from_days(time, flux, vec![1e-3; 2160])?;
    /// let periods = Periods::from_days(&[2.45, 2.5, 2.55])?;
    /// let options = SearchOptions::default().with_bootstrap(0);
    ///
    /// let best = Tls::new(&lc)?.search(&periods, &options)?.best().cloned().unwrap();
    /// let truth = planet.t14().expect("a transit").as_days();
    /// let error = |d: TimeDelta| (d.as_days() / truth - 1.0).abs();
    /// assert!(error(best.duration) > 0.05);
    /// assert!(error(best.duration_estimate()) < 0.01);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn duration_estimate(&self) -> TimeDelta {
        match self.shape {
            Some(shape) if self.chi2 - shape.chi2 > SHAPE_CHI2_GAIN => shape.duration,
            _ => self.duration,
        }
    }
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
