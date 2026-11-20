//! The public interface: [`Detrend`]'s settings, [`Detrend::trend`] and
//! [`Detrend::flatten`].

use std::{num::NonZeroUsize, thread};

use rustellar_core::{LightCurve, TimeDelta};

use crate::{
    DetrendError,
    huber_joint::gamma,
    location::Location,
    spline::{self, Fit},
    trend::trend,
    window::{Edges, WindowOptions},
};

/// Fewer points than this per thread are not worth a thread's start.
const POINTS_PER_THREAD: usize = 1024;

/// The window [`Detrend::with_window_for_transit`] sets, in transit
/// durations: 2.5, the middle of the range a blind search did best after,
/// so a transit somewhat longer than expected still keeps most of its depth.
///
/// Around a candidate, mask its transits ([`transit_mask`](crate::transit_mask)):
/// then any window keeps the whole depth, and one of 2 durations leaves
/// little of the star's variability.
pub const WINDOW_PER_DURATION: f64 = 2.5;

/// How the trend is made: a location of each window's fluxes (sliding),
/// or a spline fitted to each segment with knots a window apart.
///
/// New methods may be added in minor releases, so a `match` on this type
/// needs a wildcard arm.
///
/// # Examples
///
/// ```
/// use rustellar_detrend::Method;
///
/// assert_eq!(Method::default(), Method::Biweight { c: 5.0 });
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Method {
    /// Tukey's biweight location (Hippke et al. 2019), solved to full
    /// precision: correctly rounded wherever the sign of its defining sum
    /// can be told from zero. `c` is the tuning constant in median absolute
    /// deviations, from `1e-30` to `1e30`: points more than `c` MADs from
    /// the location get no weight. wotan's default, and this crate's, is 5.
    Biweight {
        /// The tuning constant, in median absolute deviations.
        c: f64,
    },
    /// The median: the middle flux, or the correctly rounded midpoint of the
    /// two middle fluxes.
    Median,
    /// The mean, correctly rounded (summed exactly).
    Mean,
    /// Welsch's M-estimator (Dennis and Welsch 1978): weights
    /// `exp(-u^2 / 2)`, `u` in units of `c` MADs, from `1e-30` to `1e30`;
    /// wotan's default is 2.11. Solved as the biweight is. Unlike wotan's,
    /// it gives every point some weight: wotan's also drops points beyond
    /// `c` MADs, which can leave its defining sum with no root.
    Welsch {
        /// The tuning constant, in median absolute deviations.
        c: f64,
    },
    /// Andrews' sine wave (Andrews et al. 1972): `psi(u) = sin u` for
    /// `|u| < pi`, `u` in units of `c` MADs, from `1e-30` to `1e30`; wotan's
    /// default is 1.339. Solved as the biweight is.
    AndrewsSineWave {
        /// The tuning constant, in median absolute deviations.
        c: f64,
    },
    /// Huber's M-estimator (Huber 1964), iterated to its root:
    /// `psi(u) = u` for `|u| < 1`, else its sign, `u` in units of `c` MADs,
    /// from `1e-30` to `1e30`; 1.28 as wotan's `huber_psi`, which takes a
    /// single step from the median instead.
    Huber {
        /// The tuning constant, in median absolute deviations.
        c: f64,
    },
    /// Huber's proposal 2 (Huber 1964): the location and the scale `s`
    /// estimated together, points beyond `c s` from the location counted
    /// as if at `c s`, `s` normalised to the standard deviation of normal
    /// data (with `n - 1`). The estimator of statsmodels' `Huber` and
    /// wotan's `method='huber'` (both with `c` 1.5, `c` here from `1e-30`
    /// to `1e30`), but solved exactly where they stop at a tolerance: the
    /// location to about 1e-30 of the solution, so the nearest double
    /// almost always. A window whose values are mostly equal can have
    /// scale 0, and then the location is that value.
    HuberJoint {
        /// The tuning constant, in scales.
        c: f64,
    },
    /// The Hodges-Lehmann estimator: the median of the averages of every
    /// pair of fluxes (each with itself too), correctly rounded.
    HodgesLehmann,
    /// The mean without the `floor(proportion n)` smallest and largest
    /// fluxes, correctly rounded. `proportion` from 0 to below 1/2; wotan's
    /// default is 0.1.
    TrimmedMean {
        /// The proportion cut from each end.
        proportion: f64,
    },
    /// The mean with the `floor(proportion n)` smallest fluxes replaced by
    /// the next, and the largest by the one before, correctly rounded.
    /// `proportion` from 0 to below 1/2. (wotan's replaces one more at the
    /// low end than at the high.)
    WinsorizedMean {
        /// The proportion replaced at each end.
        proportion: f64,
    },
    /// A cubic spline fitted to each segment by least squares, not a
    /// sliding window: knots evenly spaced, at most a window apart (as many
    /// intervals as the segment's length over the window, rounded up, and
    /// no more than its unmasked points). Masked points are left out of the
    /// fit and get the spline's value; where too few points pin the spline
    /// down (an interval with none nearby), it is kept smooth across. The
    /// fit scipy's `LSQUnivariateSpline` makes on the same knots, to about
    /// an ulp of the fluxes' scale where scipy's is off by tens. Not
    /// robust: an outlier pulls the spline toward it. The edge cutoff
    /// applies as for the windows.
    Spline,
    /// [`Spline`](Self::Spline) made robust by clipping: refitted without
    /// the points more than `sigma` robust standard deviations from it
    /// (1.4826 times the median absolute deviation of the residuals), on
    /// either side, every point judged afresh each time, until the points
    /// kept repeat (at most 50 refits). Flares and dips both drop out,
    /// so a transit pulls the trend less. `sigma` from `1e-30` to `1e30`;
    /// 3 is usual. wotan's `rspline` clips only points below the spline,
    /// by the standard deviation, at 2.
    RobustSpline {
        /// The clipping threshold, in robust standard deviations.
        sigma: f64,
    },
}

impl Default for Method {
    fn default() -> Self {
        Self::Biweight { c: 5.0 }
    }
}

/// Where the light curve is split into segments that are detrended apart.
///
/// # Examples
///
/// ```
/// use rustellar_core::TimeDelta;
/// use rustellar_detrend::{BreakTolerance, Detrend};
///
/// let detrend = Detrend::default()
///     .with_break_tolerance(BreakTolerance::Fixed(TimeDelta::from_days(1.0)?));
/// assert_eq!(Detrend::default().break_tolerance, BreakTolerance::HalfWindow);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BreakTolerance {
    /// At gaps longer than half the window (wotan's default).
    #[default]
    HalfWindow,
    /// At gaps longer than this, which must be greater than zero.
    Fixed(TimeDelta),
    /// Nowhere: the whole light curve is one segment.
    Never,
}

/// How to detrend a light curve: a time-windowed sliding filter.
///
/// For each point, the fluxes of the points within half a window either
/// side, in its segment and not masked, are summarised by the
/// [`method`](Self::method); that is the trend at the point. The light
/// curve is split into segments at gaps longer than the
/// [`break_tolerance`](Self::break_tolerance), and points closer than the
/// [`edge_cutoff`](Self::edge_cutoff) to either end of their segment get no
/// trend.
///
/// Every comparison of times is exact, so a point exactly half a window
/// away is always in or always out. The results are the same, to the bit,
/// on any number of threads and on any platform.
///
/// Start from [`Detrend::default`] and change what you need with the
/// `with_` methods; new settings may be added in minor releases.
///
/// # Examples
///
/// ```
/// use rustellar_core::{LightCurve, TimeDelta};
/// use rustellar_detrend::{Detrend, Method};
///
/// // Two days at 2-minute cadence: a slow drift and one dip.
/// let time: Vec<f64> = (0..1440).map(|i| i as f64 / 720.0).collect();
/// let flux: Vec<f64> = time
///     .iter()
///     .map(|&t| 1.0 + 0.01 * t - if (1.0..1.05).contains(&t) { 0.005 } else { 0.0 })
///     .collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 1440])?;
///
/// let detrend = Detrend::default().with_window(TimeDelta::from_days(0.5)?);
/// let flat = detrend.flatten(&lc, None)?;
/// // The drift is gone, the dip is kept.
/// let flux = flat.light_curve().flux();
/// assert!((flux[500] - 1.0).abs() < 1e-12);
/// assert!(flux[740] < 0.996);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Detrend {
    /// What each window's fluxes are summarised by. Default
    /// [`Method::Biweight`] with `c = 5`.
    pub method: Method,
    /// The window's length, greater than zero. Default 0.5 days (wotan's
    /// for the biweight).
    pub window: TimeDelta,
    /// Where the light curve is split into segments. Default
    /// [`BreakTolerance::HalfWindow`].
    pub break_tolerance: BreakTolerance,
    /// Points closer than this to either end of their segment get no trend;
    /// at most half the window counts. Not negative. Default zero.
    pub edge_cutoff: TimeDelta,
    /// Whether the points exactly half a window after a point are in its
    /// window. Default [`Edges::Symmetric`]: they are.
    pub edges: Edges,
    /// How many threads to share the points among. Default all available
    /// cores. The results are the same, to the bit, whatever the number.
    pub threads: NonZeroUsize,
}

impl Default for Detrend {
    fn default() -> Self {
        Self {
            method: Method::default(),
            window: TimeDelta::from_days(0.5).expect("finite"),
            break_tolerance: BreakTolerance::HalfWindow,
            edge_cutoff: TimeDelta::from_days(0.0).expect("finite"),
            edges: Edges::Symmetric,
            threads: thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
        }
    }
}

impl Detrend {
    /// With `method`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_detrend::{Detrend, Method};
    ///
    /// let detrend = Detrend::default().with_method(Method::Median);
    /// assert_eq!(detrend.method, Method::Median);
    /// ```
    #[must_use]
    pub fn with_method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    /// With windows `window` long.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::TimeDelta;
    /// use rustellar_detrend::Detrend;
    ///
    /// let detrend = Detrend::default().with_window(TimeDelta::from_hours(18.0)?);
    /// assert_eq!(detrend.window.as_days(), 0.75);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn with_window(mut self, window: TimeDelta) -> Self {
        self.window = window;
        self
    }

    /// With windows [`WINDOW_PER_DURATION`] times `duration` long: for a
    /// search for transits up to `duration` long.
    ///
    /// A window too short follows a transit and takes some of its depth;
    /// one too long leaves the star's variability in, which raises false
    /// peaks. On 400 injected transits on spotted stars, a blind BLS search
    /// found the right period most often after windows of 2 to 3 transit
    /// durations, and less often on either side.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::TimeDelta;
    /// use rustellar_detrend::Detrend;
    ///
    /// let detrend = Detrend::default().with_window_for_transit(TimeDelta::from_hours(4.0)?);
    /// assert_eq!(detrend.window.as_hours(), 10.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn with_window_for_transit(self, duration: TimeDelta) -> Self {
        self.with_window(duration * WINDOW_PER_DURATION)
    }

    /// With segments split at `break_tolerance`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_detrend::{BreakTolerance, Detrend};
    ///
    /// let detrend = Detrend::default().with_break_tolerance(BreakTolerance::Never);
    /// assert_eq!(detrend.break_tolerance, BreakTolerance::Never);
    /// ```
    #[must_use]
    pub fn with_break_tolerance(mut self, break_tolerance: BreakTolerance) -> Self {
        self.break_tolerance = break_tolerance;
        self
    }

    /// With no trend within `edge_cutoff` of a segment's ends.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::TimeDelta;
    /// use rustellar_detrend::Detrend;
    ///
    /// let detrend = Detrend::default().with_edge_cutoff(TimeDelta::from_days(0.1)?);
    /// assert_eq!(detrend.edge_cutoff.as_days(), 0.1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn with_edge_cutoff(mut self, edge_cutoff: TimeDelta) -> Self {
        self.edge_cutoff = edge_cutoff;
        self
    }

    /// With window ends as `edges` says.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_detrend::{Detrend, Edges};
    ///
    /// let detrend = Detrend::default().with_edges(Edges::HalfOpen);
    /// assert_eq!(detrend.edges, Edges::HalfOpen);
    /// ```
    #[must_use]
    pub fn with_edges(mut self, edges: Edges) -> Self {
        self.edges = edges;
        self
    }

    /// On `threads` threads.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_detrend::Detrend;
    /// use std::num::NonZeroUsize;
    ///
    /// let detrend = Detrend::default().with_threads(NonZeroUsize::MIN);
    /// assert_eq!(detrend.threads.get(), 1);
    /// ```
    #[must_use]
    pub fn with_threads(mut self, threads: NonZeroUsize) -> Self {
        self.threads = threads;
        self
    }

    /// The trend at each point of `lc`, or `None` where the point has no
    /// window (within the edge cutoff) or its window holds no point (all
    /// masked).
    ///
    /// Points where `mask` is true are left out of every window, but still
    /// get a trend from their unmasked neighbours: mask the transits, so
    /// the trend passes under them.
    ///
    /// # Errors
    ///
    /// The checks run in this order, and the first that fails is returned:
    ///
    /// 1. [`TuningOutOfRange`](DetrendError::TuningOutOfRange) if an
    ///    M-estimator's `c` or a robust spline's `sigma` is not between
    ///    `1e-30` and `1e30`, or
    ///    [`ProportionOutOfRange`](DetrendError::ProportionOutOfRange) if a
    ///    trimmed or winsorized mean's proportion is not from 0 to below 1/2.
    /// 2. [`NotPositive`](DetrendError::NotPositive) if the window, or a
    ///    fixed break tolerance, is zero or negative.
    /// 3. [`NegativeEdgeCutoff`](DetrendError::NegativeEdgeCutoff) if the
    ///    edge cutoff is negative.
    /// 4. [`MaskLengthMismatch`](DetrendError::MaskLengthMismatch) if `mask`
    ///    is not as long as `lc`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{LightCurve, TimeDelta};
    /// use rustellar_detrend::{Detrend, Method};
    ///
    /// let lc = LightCurve::from_btjd(
    ///     vec![0.0, 0.125, 0.25, 0.375, 0.5],
    ///     vec![1.0, 2.0, 3.0, 4.0, 50.0],
    ///     vec![0.1; 5],
    /// )?;
    /// let detrend = Detrend::default()
    ///     .with_method(Method::Median)
    ///     .with_window(TimeDelta::from_days(0.25)?);
    /// // Each point and its neighbours either side.
    /// let trend = detrend.trend(&lc, None)?;
    /// assert_eq!(trend, [Some(1.5), Some(2.0), Some(3.0), Some(4.0), Some(27.0)]);
    /// // The last point masked: left out of its neighbour's window.
    /// let masked = [false, false, false, false, true];
    /// let trend = detrend.trend(&lc, Some(&masked))?;
    /// assert_eq!(trend[3], Some(3.5));
    /// assert_eq!(trend[4], Some(4.0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn trend(
        &self,
        lc: &LightCurve,
        mask: Option<&[bool]>,
    ) -> Result<Vec<Option<f64>>, DetrendError> {
        let (options, estimator) = self.check()?;
        if mask.is_some_and(|m| m.len() != lc.len()) {
            return Err(DetrendError::MaskLengthMismatch);
        }
        let times: Vec<f64> = lc.time().iter().map(|t| t.as_btjd()).collect();
        Ok(match estimator {
            Estimator::Window(location) => {
                let threads = self.threads.get().min(lc.len().div_ceil(POINTS_PER_THREAD));
                trend(&times, lc.flux(), mask, &options, location, threads)
            }
            Estimator::Spline(how) => spline::trend(&times, lc.flux(), mask, &options, how),
        })
    }

    /// `lc` divided by its trend: each flux and flux error divided by the
    /// trend at its point (the error by its magnitude), with the trend.
    ///
    /// Points without a trend (see [`trend`](Self::trend)) are left out of
    /// the flattened light curve, and so are any whose quotients are not
    /// finite or whose error is not greater than zero (a trend of zero, or
    /// one far below the flux).
    ///
    /// # Errors
    ///
    /// Those of [`trend`](Self::trend), then
    /// [`NoPointsLeft`](DetrendError::NoPointsLeft) if no point is left.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{LightCurve, TimeDelta};
    /// use rustellar_detrend::{Detrend, Method};
    ///
    /// let lc = LightCurve::from_btjd(
    ///     vec![0.0, 0.125, 0.25, 0.375],
    ///     vec![2.0, 2.0, 1.0, 2.0],
    ///     vec![0.2; 4],
    /// )?;
    /// let detrend = Detrend::default()
    ///     .with_method(Method::Median)
    ///     .with_window(TimeDelta::from_days(0.25)?)
    ///     .with_edge_cutoff(TimeDelta::from_days(0.0625)?);
    /// let flat = detrend.flatten(&lc, None)?;
    /// // The two ends are cut off.
    /// assert_eq!(flat.indices(), [1, 2]);
    /// assert_eq!(flat.light_curve().flux(), [1.0, 0.5]);
    /// assert_eq!(flat.light_curve().flux_err(), [0.1, 0.1]);
    /// assert_eq!(flat.trend(), [None, Some(2.0), Some(2.0), None]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flatten(
        &self,
        lc: &LightCurve,
        mask: Option<&[bool]>,
    ) -> Result<Flattened, DetrendError> {
        let trend = self.trend(lc, mask)?;
        let mut indices = Vec::new();
        let (mut time, mut flux, mut flux_err) = (Vec::new(), Vec::new(), Vec::new());
        for (i, level) in trend.iter().enumerate() {
            let Some(level) = *level else { continue };
            let (f, e) = (lc.flux()[i] / level, lc.flux_err()[i] / level.abs());
            if f.is_finite() && e.is_finite() && e > 0.0 {
                indices.push(i);
                time.push(lc.time()[i]);
                flux.push(f);
                flux_err.push(e);
            }
        }
        if indices.is_empty() {
            return Err(DetrendError::NoPointsLeft);
        }
        // A subset of a light curve's times, and every value checked above.
        let light_curve = LightCurve::new(time, flux, flux_err).expect("a valid light curve");
        Ok(Flattened {
            light_curve,
            indices,
            trend,
        })
    }

    /// The settings checked, as the windows' options and the estimator.
    fn check(&self) -> Result<(WindowOptions, Estimator), DetrendError> {
        let tuning = |c: f64| {
            if (1e-30..=1e30).contains(&c) {
                Ok(c)
            } else {
                Err(DetrendError::TuningOutOfRange)
            }
        };
        let proportion = |p: f64| {
            if (0.0..0.5).contains(&p) {
                Ok(p)
            } else {
                Err(DetrendError::ProportionOutOfRange)
            }
        };
        let window_of = |location| Estimator::Window(location);
        let estimator = match self.method {
            Method::Biweight { c } => window_of(Location::Biweight(tuning(c)?)),
            Method::Welsch { c } => window_of(Location::Welsch(tuning(c)?)),
            Method::AndrewsSineWave { c } => window_of(Location::Andrews(tuning(c)?)),
            Method::Huber { c } => window_of(Location::Huber(tuning(c)?)),
            Method::HuberJoint { c } => window_of(Location::HuberJoint(tuning(c)?, gamma(c))),
            Method::Median => window_of(Location::Median),
            Method::Mean => window_of(Location::Mean),
            Method::HodgesLehmann => window_of(Location::HodgesLehmann),
            Method::TrimmedMean { proportion: p } => window_of(Location::Trimmed(proportion(p)?)),
            Method::WinsorizedMean { proportion: p } => {
                window_of(Location::Winsorized(proportion(p)?))
            }
            Method::Spline => Estimator::Spline(Fit::Plain),
            Method::RobustSpline { sigma } => Estimator::Spline(Fit::Clipped(tuning(sigma)?)),
        };
        let window = self.window.as_days();
        if window <= 0.0 {
            return Err(DetrendError::NotPositive { name: "window" });
        }
        let break_tolerance = match self.break_tolerance {
            BreakTolerance::HalfWindow => window / 2.0,
            BreakTolerance::Fixed(tolerance) if tolerance.as_days() <= 0.0 => {
                return Err(DetrendError::NotPositive {
                    name: "break_tolerance",
                });
            }
            BreakTolerance::Fixed(tolerance) => tolerance.as_days(),
            BreakTolerance::Never => f64::INFINITY,
        };
        let edge_cutoff = self.edge_cutoff.as_days();
        if edge_cutoff < 0.0 {
            return Err(DetrendError::NegativeEdgeCutoff);
        }
        let options = WindowOptions {
            window,
            break_tolerance,
            edge_cutoff,
            edges: self.edges,
        };
        Ok((options, estimator))
    }
}

/// What a checked [`Method`] runs.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Estimator {
    /// A location of each point's window.
    Window(Location),
    /// A spline per segment.
    Spline(Fit),
}

/// A light curve divided by its trend, from [`Detrend::flatten`].
///
/// # Examples
///
/// ```
/// use rustellar_core::LightCurve;
/// use rustellar_detrend::Detrend;
///
/// let lc = LightCurve::from_btjd(vec![0.0, 0.1, 0.2], vec![4.0, 4.0, 4.0], vec![0.4; 3])?;
/// let flat = Detrend::default().flatten(&lc, None)?;
/// assert_eq!(flat.light_curve().flux(), [1.0, 1.0, 1.0]);
/// let lc = flat.into_light_curve();
/// assert_eq!(lc.flux_err(), [0.1, 0.1, 0.1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Flattened {
    light_curve: LightCurve,
    indices: Vec<usize>,
    trend: Vec<Option<f64>>,
}

impl Flattened {
    /// The flattened light curve: the points with a trend, each flux and
    /// flux error divided by it.
    #[must_use]
    pub fn light_curve(&self) -> &LightCurve {
        &self.light_curve
    }

    /// The flattened light curve, taken out.
    #[must_use]
    pub fn into_light_curve(self) -> LightCurve {
        self.light_curve
    }

    /// For each point of the flattened light curve, its index in the
    /// original.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// The trend at each point of the original light curve, as from
    /// [`Detrend::trend`].
    #[must_use]
    pub fn trend(&self) -> &[Option<f64>] {
        &self.trend
    }
}
