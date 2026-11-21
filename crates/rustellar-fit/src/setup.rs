//! What a fit starts from: the light curve, the starting parameters, which
//! are free, and how each point is measured.

use rustellar_core::LightCurve;
use rustellar_model::{Exposure, TransitParams};

use crate::{FitError, Parameter};

/// A transit fit set up: the light curve, the starting parameters and
/// baseline, which [`Parameter`]s are free (all by default) and, for long
/// cadences, each point's [`Exposure`].
///
/// # Examples
///
/// ```
/// use rustellar_fit::{Parameter, TransitFit};
/// use rustellar_model::{
///     Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis,
///     TimeDelta, TransitParams,
/// };
///
/// let start = TransitParams::new(
///     Epoch::from_days(1.0)?,
///     TimeDelta::from_days(3.0)?,
///     RadiusRatio::new(0.1)?,
///     ScaledSemiMajorAxis::new(10.0)?,
///     ImpactParameter::new(0.3)?,
///     LimbDarkening::quadratic(0.4, 0.25)?,
/// )?;
/// let time: Vec<f64> = (0..1000).map(|i| 0.01 * f64::from(i)).collect();
/// let lc = LightCurve::from_days(time, vec![1.0; 1000], vec![1e-3; 1000])?;
///
/// let fit = TransitFit::new(&lc, start)
///     .with_fixed(Parameter::U1)
///     .with_fixed(Parameter::U2)
///     .with_baseline(1.0)?;
/// assert_eq!(fit.free().len(), 6);
/// assert!(!fit.is_free(Parameter::U1));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct TransitFit<'a> {
    pub(crate) lc: &'a LightCurve,
    pub(crate) start: TransitParams,
    pub(crate) baseline: f64,
    pub(crate) free: [bool; 8],
    pub(crate) exposure: Option<Exposure>,
    pub(crate) max_iterations: usize,
    pub(crate) accelerate: bool,
    pub(crate) bounds: LimbDarkeningBounds,
}

/// Which limb-darkening coefficients a fit may reach.
///
/// For the quadratic law `I(mu) / I(1) = 1 - u1 (1 - mu) - u2 (1 - mu)^2`:
///
/// - [`Valid`](Self::Valid), the default: any the model accepts, the
///   intensity nowhere negative. Limb brightening (`u1 < 0`) is allowed,
///   as some stars and some fits to noise show it.
/// - [`Physical`](Self::Physical): the intensity also nowhere rising
///   towards the limb, Kipping's (2013) triangle `u1 + u2 <= 1`,
///   `u1 >= 0`, `u1 + 2 u2 >= 0`, for a star that darkens to its limb.
///
/// The bounds apply to the coefficients that are free; a fixed one keeps
/// its value, and the free one is held to what the bounds allow beside it.
///
/// # Examples
///
/// ```
/// use rustellar_fit::LimbDarkeningBounds;
///
/// assert_eq!(LimbDarkeningBounds::default(), LimbDarkeningBounds::Valid);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum LimbDarkeningBounds {
    /// The intensity nowhere negative.
    #[default]
    Valid,
    /// The intensity nowhere negative nor rising towards the limb.
    Physical,
}

impl<'a> TransitFit<'a> {
    /// A fit of `lc` from `start`, a baseline of 1, every parameter free,
    /// each point an instant.
    #[must_use]
    pub const fn new(lc: &'a LightCurve, start: TransitParams) -> Self {
        Self {
            lc,
            start,
            baseline: 1.0,
            free: [true; 8],
            exposure: None,
            max_iterations: 200,
            accelerate: true,
            bounds: LimbDarkeningBounds::Valid,
        }
    }

    /// The starting baseline: the flux out of transit.
    ///
    /// # Errors
    ///
    /// [`FitError::NonFinite`] or [`FitError::NotPositive`] unless it is
    /// finite and above zero.
    pub fn with_baseline(self, baseline: f64) -> Result<Self, FitError> {
        if !baseline.is_finite() {
            return Err(FitError::NonFinite { name: "baseline" });
        }
        if baseline <= 0.0 {
            return Err(FitError::NotPositive { name: "baseline" });
        }
        Ok(Self { baseline, ..self })
    }

    /// `parameter` held at its starting value.
    #[must_use]
    pub const fn with_fixed(mut self, parameter: Parameter) -> Self {
        self.free[parameter.index()] = false;
        self
    }

    /// `parameter` fitted (as every parameter is by default).
    #[must_use]
    pub const fn with_free(mut self, parameter: Parameter) -> Self {
        self.free[parameter.index()] = true;
        self
    }

    /// Each point the mean over `exposure` (Kipping 2010), for long
    /// cadences.
    #[must_use]
    pub const fn with_exposure(self, exposure: Exposure) -> Self {
        Self {
            exposure: Some(exposure),
            ..self
        }
    }

    /// At most `iterations` steps (200 by default).
    #[must_use]
    pub const fn with_max_iterations(self, iterations: usize) -> Self {
        Self {
            max_iterations: iterations,
            ..self
        }
    }

    /// Whether each step carries the geodesic acceleration of Transtrum &
    /// Sethna (2012), a second-order correction that follows curved
    /// valleys, such as `a / R*` against `b`, in fewer steps (on by
    /// default).
    #[must_use]
    pub const fn with_acceleration(self, accelerate: bool) -> Self {
        Self { accelerate, ..self }
    }

    /// The limb-darkening coefficients the fit may reach
    /// ([`LimbDarkeningBounds::Valid`] by default). A start outside them is
    /// moved to their nearest point.
    #[must_use]
    pub const fn with_limb_darkening_bounds(self, bounds: LimbDarkeningBounds) -> Self {
        Self { bounds, ..self }
    }

    /// The limb-darkening bounds.
    #[must_use]
    pub const fn limb_darkening_bounds(&self) -> LimbDarkeningBounds {
        self.bounds
    }

    /// Whether `parameter` is fitted.
    #[must_use]
    pub const fn is_free(&self, parameter: Parameter) -> bool {
        self.free[parameter.index()]
    }

    /// The fitted parameters, in [`Parameter::ALL`]'s order.
    #[must_use]
    pub fn free(&self) -> Vec<Parameter> {
        Parameter::ALL
            .into_iter()
            .filter(|&p| self.is_free(p))
            .collect()
    }

    /// The light curve being fitted.
    #[must_use]
    pub const fn light_curve(&self) -> &'a LightCurve {
        self.lc
    }

    /// The most steps the fit takes.
    #[must_use]
    pub const fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// The starting parameters.
    #[must_use]
    pub const fn start(&self) -> TransitParams {
        self.start
    }

    /// The starting baseline.
    #[must_use]
    pub const fn baseline(&self) -> f64 {
        self.baseline
    }

    /// The exposure, if set.
    #[must_use]
    pub const fn exposure(&self) -> Option<Exposure> {
        self.exposure
    }

    /// Whether the set-up can be fitted, as the fit itself checks first.
    ///
    /// # Errors
    ///
    /// - [`FitError::NothingToFit`] if every parameter is fixed.
    /// - [`FitError::TooFewPoints`] unless there are more points than free
    ///   parameters.
    /// - [`FitError::NoPointsInTransit`] if a shape parameter is free but
    ///   the starting parameters put no point in transit.
    pub fn check(&self) -> Result<(), FitError> {
        let free = self.free();
        if free.is_empty() {
            return Err(FitError::NothingToFit);
        }
        if self.lc.len() <= free.len() {
            return Err(FitError::TooFewPoints {
                needed: free.len() + 1,
            });
        }
        let shape = free.iter().any(|&p| p != Parameter::Baseline);
        let flux = |t| match self.exposure {
            Some(exposure) => self.start.binned_flux_at(t, exposure),
            None => self.start.flux_at(t),
        };
        if shape && self.lc.time().iter().all(|&t| flux(t) >= 1.0) {
            return Err(FitError::NoPointsInTransit);
        }
        Ok(())
    }
}
