//! What a fit gives back.

use std::f64::consts::PI;

use rustellar_model::TransitParams;

use crate::lm::Solution;
use crate::problem::params_from;
use crate::{Covariance, Estimate, Parameter, Termination};

/// The gravitational constant (CODATA 2018), in m^3 kg^-1 s^-2.
const G: f64 = 6.674_30e-11;
const SECONDS_PER_DAY: f64 = 86_400.0;

/// A finished fit: the best parameters with their standard errors and
/// covariance, the quantities derived from them, the chi-squared, and how
/// the fit got there.
///
/// The errors are to first order, from the fit's linearisation at the
/// minimum, and take the flux errors as given (see
/// [`Covariance::scaled`] to scale them by the scatter). An error is
/// infinite where the fit leaves its value unconstrained.
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
/// let truth = TransitParams::new(
///     Epoch::from_days(0.5)?,
///     TimeDelta::from_days(3.0)?,
///     RadiusRatio::new(0.1)?,
///     ScaledSemiMajorAxis::new(10.0)?,
///     ImpactParameter::new(0.3)?,
///     LimbDarkening::quadratic(0.4, 0.25)?,
/// )?;
/// let time: Vec<f64> = (0..1000).map(|i| 0.001 * f64::from(i)).collect();
/// let mut flux = Vec::new();
/// for &t in &time {
///     flux.push(truth.flux_at(Epoch::from_days(t)?));
/// }
/// let lc = LightCurve::from_days(time, flux, vec![1e-3; 1000])?;
///
/// let result = TransitFit::new(&lc, truth)
///     .with_fixed(Parameter::Period)
///     .run()?;
/// let k = result.estimate(Parameter::RadiusRatio);
/// assert!((k.value() - 0.1).abs() < 1e-9);
/// assert!(k.error() > 1e-4 && k.error() < 1e-2);
/// // A fixed parameter has no error.
/// assert_eq!(result.estimate(Parameter::Period).error(), 0.0);
/// let density = result.stellar_density(); // in g/cm^3
/// assert!((density.value() - 2.10).abs() < 0.01);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct FitResult {
    params: TransitParams,
    /// Every parameter's value, `b` as its size.
    values: [f64; 8],
    /// The differencing steps of the derived quantities, over `free`.
    steps: Vec<f64>,
    free: Vec<Parameter>,
    covariance: Option<Covariance>,
    chi_squared: f64,
    points: usize,
    iterations: usize,
    evaluations: usize,
    jacobians: usize,
    termination: Termination,
}

impl FitResult {
    /// The result at `values` (with `b` signed as fitted), over `free`,
    /// with the solver's `steps` and `solution`.
    pub(crate) fn new(
        mut values: [f64; 8],
        free: Vec<Parameter>,
        steps: Vec<f64>,
        solution: Solution,
    ) -> Self {
        let mut covariance = Covariance::from_r(free.clone(), &solution.r);
        // Reported as |b|, so its covariances change sign with b.
        if values[4] < 0.0 {
            values[4] = -values[4];
            let place = free.iter().position(|&p| p == Parameter::ImpactParameter);
            if let (Some(c), Some(i)) = (covariance.as_mut(), place) {
                c.flip(i);
            }
        }
        let params = params_from(&values).expect("the solver keeps to feasible points");
        Self {
            params,
            values,
            steps,
            free,
            covariance,
            chi_squared: solution.f.iter().map(|r| r * r).sum(),
            points: solution.f.len(),
            iterations: solution.iterations,
            evaluations: solution.evaluations,
            jacobians: solution.jacobians,
            termination: solution.termination,
        }
    }

    /// The best transit parameters.
    #[must_use]
    pub const fn params(&self) -> TransitParams {
        self.params
    }

    /// The best baseline.
    #[must_use]
    pub const fn baseline(&self) -> f64 {
        self.values[7]
    }

    /// One parameter's best value (its starting value if it was fixed).
    #[must_use]
    pub const fn value(&self, parameter: Parameter) -> f64 {
        self.values[parameter.index()]
    }

    /// One parameter's best value and standard error (zero if it was
    /// fixed).
    #[must_use]
    pub fn estimate(&self, parameter: Parameter) -> Estimate {
        let error = self
            .covariance
            .as_ref()
            .map_or(f64::INFINITY, |c| c.standard_error(parameter));
        let error = if self.free.contains(&parameter) {
            error
        } else {
            0.0
        };
        Estimate::new(self.value(parameter), error)
    }

    /// The fitted parameters' covariance, or `None` where the fit leaves
    /// some combination of them unconstrained (the errors are then
    /// infinite).
    #[must_use]
    pub const fn covariance(&self) -> Option<&Covariance> {
        self.covariance.as_ref()
    }

    /// The transit's depth: one less the flux at mid-transit, against the
    /// baseline, with limb darkening (not `k^2`).
    #[must_use]
    pub fn depth(&self) -> Estimate {
        self.derived(|p| Some(1.0 - p.flux_at(p.t0())))
            .expect("every transit has a depth")
    }

    /// The total duration, first to fourth contact, in days (Seager &
    /// Mallen-Ornelas 2003); `None` if the planet misses the star.
    #[must_use]
    pub fn t14(&self) -> Option<Estimate> {
        self.derived(|p| p.t14().map(|t| t.as_days()))
    }

    /// The full duration, second to third contact, in days; `None` for a
    /// grazing transit, which has no second contact.
    #[must_use]
    pub fn t23(&self) -> Option<Estimate> {
        self.derived(|p| p.t23().map(|t| t.as_days()))
    }

    /// The orbit's inclination, in degrees: 90 is edge-on.
    #[must_use]
    pub fn inclination_deg(&self) -> Estimate {
        self.derived(|p| Some(p.inclination_deg()))
            .expect("every orbit has an inclination")
    }

    /// The star's mean density in g/cm^3, `3 pi (a / R*)^3 / (G P^2)`
    /// (Seager & Mallen-Ornelas 2003): for a circular orbit and a planet of
    /// negligible mass against the star's.
    #[must_use]
    pub fn stellar_density(&self) -> Estimate {
        self.derived(|p| {
            let period = p.period().as_days() * SECONDS_PER_DAY;
            let kg_per_m3 = 3.0 * PI * p.a_rs().get().powi(3) / (G * period * period);
            Some(kg_per_m3 / 1000.0)
        })
        .expect("every orbit has a density")
    }

    /// A quantity of the parameters with its error to first order, the
    /// gradient by central differences (one-sided where the quantity is
    /// undefined on the other side).
    fn derived(&self, f: impl Fn(TransitParams) -> Option<f64>) -> Option<Estimate> {
        let at = |values: &[f64; 8]| params_from(values).and_then(&f);
        let value = at(&self.values)?;
        let Some(covariance) = &self.covariance else {
            return Some(Estimate::new(value, f64::INFINITY));
        };
        let gradient: Vec<f64> = self
            .free
            .iter()
            .zip(&self.steps)
            .map(|(p, &h)| {
                let (mut up, mut down) = (self.values, self.values);
                up[p.index()] += h;
                down[p.index()] -= h;
                match (at(&up), at(&down)) {
                    (Some(a), Some(b)) => (a - b) / (2.0 * h),
                    (Some(a), None) => (a - value) / h,
                    (None, Some(b)) => (value - b) / h,
                    (None, None) => 0.0,
                }
            })
            .collect();
        Some(Estimate::new(value, covariance.propagate(&gradient).sqrt()))
    }

    /// The parameters that were fitted, in [`Parameter::ALL`]'s order.
    #[must_use]
    pub fn free(&self) -> &[Parameter] {
        &self.free
    }

    /// The weighted sum of squared residuals at the best fit.
    #[must_use]
    pub const fn chi_squared(&self) -> f64 {
        self.chi_squared
    }

    /// Chi-squared over the degrees of freedom: points less free
    /// parameters.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn reduced_chi_squared(&self) -> f64 {
        self.chi_squared / (self.points - self.free.len()) as f64
    }

    /// The points fitted.
    #[must_use]
    pub const fn points(&self) -> usize {
        self.points
    }

    /// The steps the fit took.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// The Jacobians from the geometry (see the crate's docs): one an
    /// iteration, unless `k` was too near its limits to difference, when
    /// the light curve was differenced instead.
    #[must_use]
    pub const fn jacobians(&self) -> usize {
        self.jacobians
    }

    /// The light curves computed: one a trial step, plus two a free
    /// parameter for each Jacobian made by differencing them.
    #[must_use]
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Why the fit stopped.
    #[must_use]
    pub const fn termination(&self) -> Termination {
        self.termination
    }
}
