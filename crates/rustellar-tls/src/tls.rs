//! The search: a light curve checked once, searched with any periods and
//! options.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use rustellar_core::LightCurve;
use rustellar_core::{Device, Periods};

use crate::TlsError;
use crate::durations::Constants;
use crate::fit::Points;
use crate::options::{Method, SearchOptions};
use crate::result::{
    Candidate, OddEven, PeriodFit, ShapeUncertainties, SignalToNoise, Spectra, TlsResult,
    TransitDepth, TransitShape, Uncertainties, epoch, span,
};
use crate::search::{self, Options, run};

/// Intervals in the template's lookup table for the fast search.
const TABLE_SAMPLES: usize = 4096;

/// A transit least squares search of one light curve.
///
/// [`Tls::new`] checks the light curve once; [`Tls::search`] then fits the
/// template at every trial period, duration and mid-transit time, builds
/// the spectra, refines the highest peaks, and describes the best
/// candidate.
///
/// The flux must be normalised, 1 out of transit (detrended, as
/// transitleastsquares expects): the template dims from 1, so a light curve
/// around any other level fits nothing.
///
/// # Examples
///
/// ```
/// use rustellar_tls::{Periods, SearchOptions, Tls};
/// use rustellar_tls::LightCurve;
///
/// // 10 days at 30 minutes with a 1% transit, 0.1 days long, every 2.5
/// // days from day 1.
/// let time: Vec<f64> = (0..480).map(|i| f64::from(i) / 48.0).collect();
/// let flux: Vec<f64> = time
///     .iter()
///     .map(|&t| {
///         let phase = (t - 1.0 + 1.25).rem_euclid(2.5) - 1.25;
///         if phase.abs() < 0.05 { 0.99 } else { 1.0 }
///     })
///     .collect();
/// let lc = LightCurve::from_days(time, flux, vec![1e-3; 480])?;
///
/// let periods: Vec<f64> = (0..200).map(|i| 1.5 + 0.01 * f64::from(i)).collect();
/// let result = Tls::new(&lc)?.search(&Periods::from_days(&periods)?, &SearchOptions::default())?;
///
/// let best = result.best().expect("a transit");
/// assert!((best.period.as_days() - 2.5).abs() < 0.01);
/// assert!(result.spectra().unwrap().sde > 7.0);
/// assert_eq!(result.periods().len(), 200);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Tls {
    points: Points,
}

impl Tls {
    /// The light curve's times, residuals `1 - flux` and weights
    /// `1 / flux_err^2`.
    ///
    /// # Errors
    ///
    /// - [`TlsError::NoBaseline`] if the light curve has a single time.
    /// - [`TlsError::OutOfRange`] if a weight is too large or too small for
    ///   an `f64` to hold to full precision, or the chi-squared of the flat
    ///   model overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::{Tls, TlsError};
    /// use rustellar_tls::LightCurve;
    ///
    /// let lc = LightCurve::from_days(vec![0.0, 1.0], vec![1.0, 1.0], vec![1e-200, 1e-3])?;
    /// assert_eq!(Tls::new(&lc).unwrap_err(), TlsError::OutOfRange);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(lc: &LightCurve) -> Result<Self, TlsError> {
        let points = Points::new(lc);
        if points.time.len() < 2 {
            return Err(TlsError::NoBaseline);
        }
        if !points.weight.iter().all(|w| w.is_normal()) || !points.flat_chi2().is_finite() {
            return Err(TlsError::OutOfRange);
        }
        Ok(Self { points })
    }

    /// The search over `periods` (in any order; the results are in
    /// increasing period).
    ///
    /// At each period the template is fitted at every trial duration and
    /// mid-transit time ([`SearchOptions::steps`] per duration), its depth
    /// by weighted least squares, and the best fit's chi-squared kept. The
    /// durations run by [`SearchOptions::duration_step`] from the shortest
    /// transit of a small planet across the smallest star to the longest of
    /// a large planet across the largest ([`SearchOptions::stars`]), at
    /// most 12% of the period. The spectra follow TLS. The
    /// [`SearchOptions::refine_peaks`] highest peaks are then refined in
    /// period, duration and time together, from their own and a few
    /// neighbouring periods, and the best by chi-squared described.
    ///
    /// # Errors
    ///
    /// - [`TlsError::NonFinite`] for a setting that is NaN or infinite.
    /// - [`TlsError::NotPositive`] for a range of stars not above zero.
    /// - [`TlsError::StarRangeEmpty`], [`TlsError::DurationStepNotAboveOne`]
    ///   or [`TlsError::TemplateNotTransiting`] for those settings.
    /// - [`TlsError::GpuUnavailable`] if [`SearchOptions::device`] is
    ///   [`Device::Gpu`] and that GPU cannot be used. A GPU that fails
    ///   during the search leaves the rest to the CPU, with the same
    ///   results.
    ///
    /// # Examples
    ///
    /// See [`Tls`].
    pub fn search(
        &self,
        periods: &Periods,
        options: &SearchOptions,
    ) -> Result<TlsResult, TlsError> {
        options.check()?;
        let template = options.template.build()?;
        let stars = options.stars.build()?;
        let device = match options.device {
            #[cfg(feature = "cuda")]
            Device::Auto => search::Device::Gpu(0),
            #[cfg(not(feature = "cuda"))]
            Device::Auto => search::Device::Cpu,
            #[cfg(feature = "cuda")]
            Device::Gpu(index) => {
                crate::cuda::Gpu::new(index).map_err(|_| TlsError::GpuUnavailable)?;
                search::Device::Gpu(index)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Gpu(_) => return Err(TlsError::GpuUnavailable),
            // The CPU, and anything a later rustellar-core adds.
            _ => search::Device::Cpu,
        };
        let internal = Options {
            template,
            table_samples: TABLE_SAMPLES,
            steps: options.steps.get(),
            min_depth: options.min_depth,
            stars,
            constants: Constants::IAU,
            duration_step: options.duration_step,
            oversampling: options.oversampling.get(),
            refine_peaks: options.refine_peaks,
            threads: options.threads,
            method: match options.method {
                Method::Cells => search::Method::Cells,
                Method::Fold => search::Method::Fold,
            },
            device,
            bootstrap: options.bootstrap,
        };
        let mut days: Vec<f64> = periods.as_slice().iter().map(|p| p.as_days()).collect();
        days.sort_by(f64::total_cmp);
        let outcome = run(&self.points, &days, internal);
        Ok(result(outcome))
    }

    /// Many light curves searched with the same `options`, each over its
    /// own periods: the results in the order given, each exactly what
    /// [`Tls::search`] gives for that curve.
    ///
    /// [`SearchOptions::threads`] is shared out: a quarter as many
    /// searches run at once as there are threads (at least one), each on
    /// its share, and on a GPU they share it. That keeps the cores and the
    /// GPU busier than one search at a time: on 16 threads it is about 1.4
    /// times the speed of a loop on the CPU and 1.6 times on a GPU (see
    /// the README).
    ///
    /// # Errors
    ///
    /// Each search's own, as [`Tls::search`]'s.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::{LightCurve, Periods, SearchOptions, Tls};
    ///
    /// // Three light curves, a 1% dip every 2.5, 3 and 3.5 days.
    /// let curve = |period: f64| -> Result<Tls, Box<dyn std::error::Error>> {
    ///     let time: Vec<f64> = (0..480).map(|i| f64::from(i) / 48.0).collect();
    ///     let flux = time
    ///         .iter()
    ///         .map(|&t| if (t - 1.0).rem_euclid(period) < 0.1 { 0.99 } else { 1.0 })
    ///         .collect();
    ///     Ok(Tls::new(&LightCurve::from_days(time, flux, vec![1e-3; 480])?)?)
    /// };
    /// let periods: Vec<f64> = (0..300).map(|i| 1.5 + 0.01 * f64::from(i)).collect();
    /// let periods = Periods::from_days(&periods)?;
    /// let jobs = [
    ///     (curve(2.5)?, periods.clone()),
    ///     (curve(3.0)?, periods.clone()),
    ///     (curve(3.5)?, periods),
    /// ];
    /// let results = Tls::search_many(&jobs, &SearchOptions::default().with_bootstrap(0));
    /// for (result, truth) in results.into_iter().zip([2.5, 3.0, 3.5]) {
    ///     let best = result?.best().expect("a transit").period.as_days();
    ///     assert!((best - truth).abs() < 0.01);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn search_many(
        jobs: &[(Self, Periods)],
        options: &SearchOptions,
    ) -> Vec<Result<TlsResult, TlsError>> {
        let threads = options.threads.get();
        let at_once = (threads / THREADS_PER_SEARCH).clamp(1, jobs.len().max(1));
        let each = options
            .with_threads(NonZeroUsize::new((threads / at_once).max(1)).expect("at least one"));
        let next = AtomicUsize::new(0);
        let mut results: Vec<Option<Result<TlsResult, TlsError>>> =
            (0..jobs.len()).map(|_| None).collect();
        let found: Vec<Vec<(usize, Result<TlsResult, TlsError>)>> = thread::scope(|scope| {
            let workers: Vec<_> = (0..at_once)
                .map(|_| {
                    scope.spawn(|| {
                        let mut mine = vec![];
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some((tls, periods)) = jobs.get(i) else {
                                break mine;
                            };
                            mine.push((i, tls.search(periods, &each)));
                        }
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|w| w.join().expect("a search thread panicked"))
                .collect()
        });
        for (i, result) in found.into_iter().flatten() {
            results[i] = Some(result);
        }
        results
            .into_iter()
            .map(|r| r.expect("every job searched"))
            .collect()
    }
}

/// The threads each of [`Tls::search_many`]'s searches gets: on 16
/// threads, four searches of four threads were the fastest split on the
/// CPU (one thread each was the slowest) and as fast as any on a GPU.
const THREADS_PER_SEARCH: usize = 4;

/// The search's outcome in public types.
fn result(outcome: search::Outcome) -> TlsResult {
    let periods = outcome
        .periods
        .iter()
        .map(|p| PeriodFit {
            period: span(p.period),
            duration: span(p.duration),
            transit_time: epoch(p.t0),
            depth: p.depth,
            chi2: p.chi2,
        })
        .collect();
    let spectra = outcome.spectra.map(|s| Spectra {
        sr: s.sr,
        power_raw: s.power_raw,
        power: s.power,
        sde_raw: s.sde_raw,
        sde: s.sde,
    });
    let best = outcome.best.map(|b| Candidate {
        period: span(b.position.period),
        transit_time: epoch(b.position.t0),
        duration: span(b.position.duration),
        depth: b.fit.depth,
        chi2: b.fit.chi2,
        points_in_transit: b.fit.in_transit,
        period_uncertainty: outcome
            .period_uncertainty
            .is_finite()
            .then(|| span(outcome.period_uncertainty)),
        errors: outcome.errors.map(|e| Uncertainties {
            period: span(e.period),
            transit_time: span(e.t0),
            duration: span(e.duration),
            depth: e.depth,
        }),
        bootstrap_errors: outcome.bootstrap.map(|e| Uncertainties {
            period: span(e.period),
            transit_time: span(e.t0),
            duration: span(e.duration),
            depth: e.depth,
        }),
        transits: outcome
            .transits
            .iter()
            .map(|t| TransitDepth {
                number: t.number,
                time: epoch(t.time),
                depth: t.depth,
                error: t.error,
                points: t.points,
            })
            .collect(),
        odd_even: outcome.odd_even.map(|o| OddEven {
            odd: o.odd,
            odd_error: o.odd_error,
            even: o.even,
            even_error: o.even_error,
            sigma: o.sigma,
        }),
        snr: outcome.snr.map(|s| SignalToNoise {
            white: s.white,
            pink: s.pink,
            white_noise: s.white_noise,
            red_noise: s.red_noise,
            depth: s.depth,
        }),
        shape: outcome.shape.map(|s| TransitShape {
            period: span(s.period),
            transit_time: epoch(s.t0),
            duration: span(s.duration),
            k: s.k,
            b: s.b,
            depth: s.depth,
            a: s.a,
            chi2: s.chi2,
            errors: s.errors.map(|e| ShapeUncertainties {
                period: span(e[0]),
                transit_time: span(e[1]),
                duration: span(e[2]),
                k: e[3],
                b: e[4],
            }),
        }),
    });
    TlsResult {
        periods,
        spectra,
        best,
        gpu: outcome.gpu,
    }
}
