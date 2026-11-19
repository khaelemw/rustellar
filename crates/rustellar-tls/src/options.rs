//! How a search is run: the template, the stars the durations cover, and
//! the search's own settings.

use std::num::NonZeroUsize;
use std::thread;

use rustellar_bls::Device;
use rustellar_model::{LimbDarkening, RadiusRatio};

use crate::TlsError;
use crate::durations;
use crate::template;

/// The transit shape fitted at every period, duration and mid-transit time.
///
/// # Examples
///
/// ```
/// use rustellar_tls::{SearchOptions, Template};
/// use rustellar_model::{LimbDarkening, RadiusRatio};
///
/// // A hot Jupiter across an M dwarf, seen near the limb.
/// let planet = Template::Planet {
///     k: RadiusRatio::new(0.15)?,
///     b: 0.6,
///     ld: LimbDarkening::quadratic(0.3, 0.3)?,
/// };
/// let options = SearchOptions::default().with_template(planet);
/// assert_eq!(options.template, planet);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum Template {
    /// TLS's default: an Earth-like planet (`k = 0.03`) across a Sun-like
    /// star, at impact parameter 0.318, with quadratic limb darkening
    /// `(0.4804, 0.1867)`.
    #[default]
    Default,
    /// TLS's grazing template: the default planet at impact parameter 0.99.
    Grazing,
    /// TLS's box: a steep trapezoid (`k = 0.1`, central, no limb
    /// darkening), as BLS's box but for its edges.
    Box,
    /// A planet of radius ratio `k` at impact parameter `b` across a star
    /// with limb darkening `ld`.
    Planet {
        /// The planet's radius over the star's.
        k: RadiusRatio,
        /// The impact parameter, in stellar radii: 0 for a central transit,
        /// below `1 + k` for any transit.
        b: f64,
        /// The star's limb darkening.
        ld: LimbDarkening,
    },
}

impl Template {
    /// The template, or [`TlsError::TemplateNotTransiting`].
    pub(crate) fn build(self) -> Result<template::Template, TlsError> {
        match self {
            Self::Default => Ok(template::Template::tls_default()),
            Self::Grazing => Ok(template::Template::tls_grazing()),
            Self::Box => Ok(template::Template::tls_box()),
            Self::Planet { k, b, ld } => {
                template::Template::new(k, b, ld).ok_or(TlsError::TemplateNotTransiting)
            }
        }
    }
}

/// The range of host stars the trial durations cover, in solar radii and
/// masses: at each period, from a small planet's central transit across the
/// smallest, lightest star to a large planet's (twice Jupiter's radius)
/// across the largest, heaviest one, as TLS.
///
/// # Examples
///
/// ```
/// use rustellar_tls::Stars;
///
/// // Sun-like stars only: fewer, longer durations, a faster search.
/// let sunlike = Stars { radius_min: 0.8, radius_max: 1.2, mass_min: 0.8, mass_max: 1.2 };
/// assert_eq!(Stars::default().radius_min, 0.13);
/// # let _ = sunlike;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stars {
    /// The smallest radius, in solar radii. Default 0.13.
    pub radius_min: f64,
    /// The largest radius, in solar radii. Default 3.5.
    pub radius_max: f64,
    /// The least mass, in solar masses. Default 0.1.
    pub mass_min: f64,
    /// The greatest mass, in solar masses. Default 1.
    pub mass_max: f64,
}

impl Default for Stars {
    fn default() -> Self {
        let tls = durations::Stars::TLS;
        Self {
            radius_min: tls.r_min,
            radius_max: tls.r_max,
            mass_min: tls.m_min,
            mass_max: tls.m_max,
        }
    }
}

impl Stars {
    /// The range checked, as the durations use it.
    pub(crate) fn build(self) -> Result<durations::Stars, TlsError> {
        let values = [
            ("stars.radius_min", self.radius_min),
            ("stars.radius_max", self.radius_max),
            ("stars.mass_min", self.mass_min),
            ("stars.mass_max", self.mass_max),
        ];
        for (name, value) in values {
            if !value.is_finite() {
                return Err(TlsError::NonFinite { name });
            }
            if value <= 0.0 {
                return Err(TlsError::NotPositive { name });
            }
        }
        if self.radius_min > self.radius_max || self.mass_min > self.mass_max {
            return Err(TlsError::StarRangeEmpty);
        }
        Ok(durations::Stars {
            r_min: self.radius_min,
            r_max: self.radius_max,
            m_min: self.mass_min,
            m_max: self.mass_max,
        })
    }
}

/// How each period's best fit is found. Both find the same fits to within
/// the trial times' spacing; on 300 injected transits they are as close to
/// the truth as each other.
///
/// # Examples
///
/// ```
/// use rustellar_tls::{Method, SearchOptions};
///
/// // TLS's own way, for comparing with it.
/// let options = SearchOptions::default().with_method(Method::Fold);
/// assert_eq!(options.method, Method::Fold);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Method {
    /// The template's sums once per duration at a grid of times, each
    /// period then summing its transits': about five times faster than
    /// [`Method::Fold`] on one thread, and the only one on a GPU.
    #[default]
    Cells,
    /// TLS's way: each period's light curve folded and the template swept
    /// along it, over durations that are fractions of the period. On the
    /// CPU only.
    Fold,
}

/// How to run a search.
///
/// Start from [`SearchOptions::default`], TLS's defaults, and change what
/// you need with the `with_` methods; new settings may be added in minor
/// releases.
///
/// # Examples
///
/// ```
/// use rustellar_tls::{Device, SearchOptions};
/// use std::num::NonZeroUsize;
///
/// let options = SearchOptions::default()
///     .with_min_depth(50e-6)
///     .with_threads(NonZeroUsize::MIN)
///     .with_device(Device::Cpu);
/// assert_eq!(options.duration_step, 1.1);
/// assert_eq!(options.refine_peaks, 5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SearchOptions {
    /// The transit shape. Default [`Template::Default`].
    pub template: Template,
    /// The host stars the durations cover. Default TLS's.
    pub stars: Stars,
    /// The least depth (dimming at mid-transit) a fit counts with. Default
    /// 10 ppm.
    pub min_depth: f64,
    /// Each trial duration this factor times the one before; above 1.
    /// Default 1.1.
    pub duration_step: f64,
    /// Trial mid-transit times per duration. Default 50.
    pub steps: NonZeroUsize,
    /// The period grid's oversampling factor, which sets the width of the
    /// running median the spectrum is flattened by (TLS's
    /// `oversampling_factor`). Default 3.
    pub oversampling: NonZeroUsize,
    /// How many of the spectrum's highest peaks are refined. Default 5; at
    /// least 1 is refined where any period has a fit.
    pub refine_peaks: usize,
    /// How each period's best fit is found. Default [`Method::Cells`].
    pub method: Method,
    /// How many threads the search runs on. Default all available cores.
    /// The results are the same, to the bit, whatever the number.
    pub threads: NonZeroUsize,
    /// Where the cells search runs. Default [`Device::Auto`]: an NVIDIA GPU
    /// if the `cuda` feature is on and one can be used, otherwise the CPU,
    /// with the same results to the bit.
    pub device: Device,
    /// How many times the best fit's residuals are resampled and the fit
    /// repeated for [`Candidate::bootstrap_errors`](crate::Candidate::bootstrap_errors).
    /// Default 100; below 2, none. The resamples are seeded: the same on
    /// every run and any number of threads.
    pub bootstrap: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            template: Template::Default,
            stars: Stars::default(),
            min_depth: 10e-6,
            duration_step: 1.1,
            steps: NonZeroUsize::new(50).expect("not zero"),
            oversampling: NonZeroUsize::new(3).expect("not zero"),
            refine_peaks: 5,
            method: Method::Cells,
            threads: thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
            device: Device::Auto,
            bootstrap: 100,
        }
    }
}

impl SearchOptions {
    /// With `template` fitted.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::{SearchOptions, Template};
    ///
    /// let options = SearchOptions::default().with_template(Template::Grazing);
    /// assert_eq!(options.template, Template::Grazing);
    /// ```
    #[must_use]
    pub fn with_template(mut self, template: Template) -> Self {
        self.template = template;
        self
    }

    /// With durations for `stars`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::{SearchOptions, Stars};
    ///
    /// let stars = Stars { radius_max: 1.5, ..Stars::default() };
    /// let options = SearchOptions::default().with_stars(stars);
    /// assert_eq!(options.stars.radius_max, 1.5);
    /// ```
    #[must_use]
    pub fn with_stars(mut self, stars: Stars) -> Self {
        self.stars = stars;
        self
    }

    /// With fits shallower than `depth` not counted.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::SearchOptions;
    ///
    /// let options = SearchOptions::default().with_min_depth(1e-4);
    /// assert_eq!(options.min_depth, 1e-4);
    /// ```
    #[must_use]
    pub fn with_min_depth(mut self, depth: f64) -> Self {
        self.min_depth = depth;
        self
    }

    /// With each trial duration `step` times the one before.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::SearchOptions;
    ///
    /// let options = SearchOptions::default().with_duration_step(1.05);
    /// assert_eq!(options.duration_step, 1.05);
    /// ```
    #[must_use]
    pub fn with_duration_step(mut self, step: f64) -> Self {
        self.duration_step = step;
        self
    }

    /// With `steps` trial mid-transit times per duration.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::SearchOptions;
    /// use std::num::NonZeroUsize;
    ///
    /// let options = SearchOptions::default().with_steps(NonZeroUsize::new(100).unwrap());
    /// assert_eq!(options.steps.get(), 100);
    /// ```
    #[must_use]
    pub fn with_steps(mut self, steps: NonZeroUsize) -> Self {
        self.steps = steps;
        self
    }

    /// With the running median sized for a grid oversampled `factor` times.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::SearchOptions;
    /// use std::num::NonZeroUsize;
    ///
    /// let options = SearchOptions::default().with_oversampling(NonZeroUsize::new(5).unwrap());
    /// assert_eq!(options.oversampling.get(), 5);
    /// ```
    #[must_use]
    pub fn with_oversampling(mut self, factor: NonZeroUsize) -> Self {
        self.oversampling = factor;
        self
    }

    /// With the `count` highest peaks refined.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::SearchOptions;
    ///
    /// let options = SearchOptions::default().with_refine_peaks(10);
    /// assert_eq!(options.refine_peaks, 10);
    /// ```
    #[must_use]
    pub fn with_refine_peaks(mut self, count: usize) -> Self {
        self.refine_peaks = count;
        self
    }

    /// With each period's best fit found by `method`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::{Method, SearchOptions};
    ///
    /// let options = SearchOptions::default().with_method(Method::Fold);
    /// assert_eq!(options.method, Method::Fold);
    /// ```
    #[must_use]
    pub fn with_method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    /// With the search on `threads` threads.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::SearchOptions;
    /// use std::num::NonZeroUsize;
    ///
    /// let options = SearchOptions::default().with_threads(NonZeroUsize::MIN);
    /// assert_eq!(options.threads.get(), 1);
    /// ```
    #[must_use]
    pub fn with_threads(mut self, threads: NonZeroUsize) -> Self {
        self.threads = threads;
        self
    }

    /// With the search run on `device`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::{Device, SearchOptions};
    ///
    /// let options = SearchOptions::default().with_device(Device::Cpu);
    /// assert_eq!(options.device, Device::Cpu);
    /// ```
    #[must_use]
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// With the bootstrap's errors from `count` resamples; below 2, none.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_tls::SearchOptions;
    ///
    /// // A survey's first pass: the curvature's errors only.
    /// let options = SearchOptions::default().with_bootstrap(0);
    /// assert_eq!(options.bootstrap, 0);
    /// ```
    #[must_use]
    pub fn with_bootstrap(mut self, count: usize) -> Self {
        self.bootstrap = count;
        self
    }

    /// The settings checked.
    pub(crate) fn check(&self) -> Result<(), TlsError> {
        if !self.min_depth.is_finite() {
            return Err(TlsError::NonFinite { name: "min_depth" });
        }
        if !self.duration_step.is_finite() {
            return Err(TlsError::NonFinite {
                name: "duration_step",
            });
        }
        if self.duration_step <= 1.0 {
            return Err(TlsError::DurationStepNotAboveOne);
        }
        Ok(())
    }
}
