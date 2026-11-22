//! Transit posteriors: the likelihood of a light curve under the transit
//! model, the parameters of a fit's set-up sampled with priors.

use std::cell::RefCell;
use std::f64::consts::PI;

use rustellar_fit::{
    Epoch, FitResult, ImpactParameter, LimbDarkening, Parameter, RadiusRatio, ScaledSemiMajorAxis,
    TimeDelta, TransitFit, TransitParams,
};
use rustellar_model::TransitModel;

use crate::rng::Rng;
use crate::{LogProbability, McmcError};

thread_local! {
    static BUFFER: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Newton's constant, m^3 kg^-1 s^-2 (CODATA 2018), as `rustellar-fit`.
const G: f64 = 6.674_30e-11;
const SECONDS_PER_DAY: f64 = 86_400.0;

/// One coordinate of a [`TransitPosterior`]'s samples.
///
/// The limb darkening is sampled as Kipping's (2013) `q1` and `q2` when
/// both coefficients are free (uniform in both is uniform over every
/// physical pair), and as `u1` or `u2` when only one is. The eccentricity
/// is sampled as `sqrt(e) cos w` and `sqrt(e) sin w` when both are free
/// (uniform in both is uniform in `e` and `w`; Eastman, Gaudi & Agol 2013),
/// and as `e cos w` or `e sin w` when only one is.
///
/// # Examples
///
/// ```
/// use rustellar_mcmc::transit::Coordinate;
///
/// assert_eq!(Coordinate::Q1.name(), "q1");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Coordinate {
    /// The time of mid-transit, in days.
    MidTime,
    /// The orbital period, in days.
    Period,
    /// The planet's radius over the star's, `k`.
    RadiusRatio,
    /// The orbit's radius over the star's, `a / R*`.
    SemiMajorAxis,
    /// The impact parameter `b`, from 0 to `1 + k` (transiting).
    ImpactParameter,
    /// The linear limb-darkening coefficient, when `u2` is fixed.
    U1,
    /// The quadratic limb-darkening coefficient, when `u1` is fixed.
    U2,
    /// Kipping's `q1 = (u1 + u2)^2`, from 0 to 1.
    Q1,
    /// Kipping's `q2 = u1 / (2 (u1 + u2))`, from 0 to 1.
    Q2,
    /// `e cos w`, when `e sin w` is fixed.
    ECosW,
    /// `e sin w`, when `e cos w` is fixed.
    ESinW,
    /// `sqrt(e) cos w`.
    SqrtECosW,
    /// `sqrt(e) sin w`.
    SqrtESinW,
    /// The flux out of transit.
    Baseline,
}

impl Coordinate {
    /// A short name: `t0`, `period`, `k`, `a_rs`, `b`, `u1`, `u2`, `q1`,
    /// `q2`, `e_cos_w`, `e_sin_w`, `sqrt_e_cos_w`, `sqrt_e_sin_w` or
    /// `baseline`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MidTime => "t0",
            Self::Period => "period",
            Self::RadiusRatio => "k",
            Self::SemiMajorAxis => "a_rs",
            Self::ImpactParameter => "b",
            Self::U1 => "u1",
            Self::U2 => "u2",
            Self::Q1 => "q1",
            Self::Q2 => "q2",
            Self::ECosW => "e_cos_w",
            Self::ESinW => "e_sin_w",
            Self::SqrtECosW => "sqrt_e_cos_w",
            Self::SqrtESinW => "sqrt_e_sin_w",
            Self::Baseline => "baseline",
        }
    }
}

/// A prior on one coordinate, its density normalised.
///
/// # Examples
///
/// ```
/// use rustellar_mcmc::transit::Prior;
///
/// let p = Prior::normal(1.0, 0.5)?;
/// assert!((p.log_density(1.0) + (0.5 * (2.0 * std::f64::consts::PI).ln() + 0.5f64.ln())).abs() < 1e-15);
/// assert_eq!(Prior::uniform(0.0, 2.0)?.log_density(3.0), f64::NEG_INFINITY);
/// # Ok::<(), rustellar_mcmc::McmcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Prior {
    /// Uniform from `low` to `high`.
    Uniform {
        /// The lower end.
        low: f64,
        /// The upper end.
        high: f64,
    },
    /// Normal about `mean` with standard deviation `sd`.
    Normal {
        /// The mean.
        mean: f64,
        /// The standard deviation.
        sd: f64,
    },
    /// Uniform in the logarithm from `low` to `high`, both above zero.
    LogUniform {
        /// The lower end.
        low: f64,
        /// The upper end.
        high: f64,
    },
}

impl Prior {
    /// Uniform from `low` to `high`.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidPrior`] unless both are finite and `low` is
    /// below `high`.
    pub fn uniform(low: f64, high: f64) -> Result<Self, McmcError> {
        if !(low.is_finite() && high.is_finite() && low < high) {
            return Err(McmcError::InvalidPrior { name: "uniform" });
        }
        Ok(Self::Uniform { low, high })
    }

    /// Normal about `mean` with standard deviation `sd`.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidPrior`] unless both are finite and `sd` is
    /// above zero.
    pub fn normal(mean: f64, sd: f64) -> Result<Self, McmcError> {
        if !(mean.is_finite() && sd.is_finite() && sd > 0.0) {
            return Err(McmcError::InvalidPrior { name: "normal" });
        }
        Ok(Self::Normal { mean, sd })
    }

    /// Uniform in the logarithm from `low` to `high`.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidPrior`] unless both are finite, `low` is above
    /// zero and below `high`.
    pub fn log_uniform(low: f64, high: f64) -> Result<Self, McmcError> {
        if !(low.is_finite() && high.is_finite() && 0.0 < low && low < high) {
            return Err(McmcError::InvalidPrior {
                name: "log-uniform",
            });
        }
        Ok(Self::LogUniform { low, high })
    }

    /// The log of the density at `x`: minus infinity outside the support.
    #[must_use]
    pub fn log_density(&self, x: f64) -> f64 {
        match *self {
            Self::Uniform { low, high } => {
                if (low..=high).contains(&x) {
                    -(high - low).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            Self::Normal { mean, sd } => {
                let z = (x - mean) / sd;
                (-0.5 * z).mul_add(z, -0.5 * (2.0 * PI).ln()) - sd.ln()
            }
            Self::LogUniform { low, high } => {
                if (low..=high).contains(&x) {
                    -x.ln() - (high / low).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }
}

/// The values of every fit parameter, in [`Parameter::ALL`]'s order.
type Values = [f64; 10];

pub(crate) fn slot(p: Parameter) -> usize {
    Parameter::ALL
        .iter()
        .position(|&q| q == p)
        .expect("every parameter is listed")
}

/// The posterior of a transit's parameters given a light curve: the
/// Gaussian likelihood of the flux with its errors under the transit model
/// (with the set-up's exposure), times the priors.
///
/// The parameters sampled are the set-up's free ones, as [`Coordinate`]s;
/// the fixed ones keep the set-up's starting values. Without a prior, each
/// coordinate is uniform over its range: `k` and `a / R*` above what the
/// model allows, `b` from 0 to `1 + k`, `q1` and `q2` from 0 to 1, the
/// eccentricity below 1 with the planet outside the star, the baseline
/// above zero. [`with_prior`](Self::with_prior) adds a prior on any
/// coordinate, and [`with_stellar_density`](Self::with_stellar_density)
/// a normal prior on the star's density from `a / R*` and the period
/// (Seager & Mallen-Ornelas 2003).
///
/// # Examples
///
/// ```
/// use rustellar_fit::{Parameter, TransitFit};
/// use rustellar_mcmc::transit::{Coordinate, Prior, TransitPosterior};
/// use rustellar_mcmc::LogProbability;
/// use rustellar_model::{
///     Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis,
///     TimeDelta, TransitParams,
/// };
///
/// let truth = TransitParams::new(
///     Epoch::from_days(0.0)?,
///     TimeDelta::from_days(3.0)?,
///     RadiusRatio::new(0.1)?,
///     ScaledSemiMajorAxis::new(10.0)?,
///     ImpactParameter::new(0.3)?,
///     LimbDarkening::quadratic(0.4, 0.25)?,
/// )?;
/// let time: Vec<f64> = (0..400).map(|i| -0.2 + 0.001 * f64::from(i)).collect();
/// let flux = time
///     .iter()
///     .map(|&t| truth.flux_at(Epoch::from_days(t).unwrap()))
///     .collect();
/// let lc = LightCurve::from_days(time, flux, vec![1e-3; 400])?;
/// let fit = TransitFit::new(&lc, truth).with_fixed(Parameter::Period);
/// let posterior = TransitPosterior::new(&fit)?
///     .with_prior(Coordinate::RadiusRatio, Prior::uniform(0.0, 0.5)?)?;
/// assert_eq!(posterior.coordinates().len(), 7);
/// let x = posterior.point(truth, 1.0);
/// assert_eq!(posterior.log_probability(&x), -(0.5f64).ln());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct TransitPosterior<'a> {
    model: TransitModel,
    flux: &'a [f64],
    weight: Vec<f64>,
    coordinates: Vec<Coordinate>,
    priors: Vec<Option<Prior>>,
    fixed: Values,
    density: Option<(f64, f64)>,
}

impl<'a> TransitPosterior<'a> {
    /// The posterior of `fit`'s free parameters, from its light curve and
    /// exposure; the fixed parameters keep its starting values.
    ///
    /// # Errors
    ///
    /// [`McmcError::NoDimensions`] if no parameter is free.
    pub fn new(fit: &TransitFit<'a>) -> Result<Self, McmcError> {
        let lc = fit.light_curve();
        let times = lc.time().to_vec();
        let model = match fit.exposure() {
            Some(exposure) => TransitModel::with_exposure(times, exposure),
            None => TransitModel::new(times),
        };
        let free = |p| fit.is_free(p);
        let mut coordinates = Vec::new();
        for (p, c) in [
            (Parameter::MidTime, Coordinate::MidTime),
            (Parameter::Period, Coordinate::Period),
            (Parameter::RadiusRatio, Coordinate::RadiusRatio),
            (Parameter::SemiMajorAxis, Coordinate::SemiMajorAxis),
            (Parameter::ImpactParameter, Coordinate::ImpactParameter),
        ] {
            if free(p) {
                coordinates.push(c);
            }
        }
        match (free(Parameter::U1), free(Parameter::U2)) {
            (true, true) => coordinates.extend([Coordinate::Q1, Coordinate::Q2]),
            (true, false) => coordinates.push(Coordinate::U1),
            (false, true) => coordinates.push(Coordinate::U2),
            (false, false) => {}
        }
        match (free(Parameter::ECosW), free(Parameter::ESinW)) {
            (true, true) => coordinates.extend([Coordinate::SqrtECosW, Coordinate::SqrtESinW]),
            (true, false) => coordinates.push(Coordinate::ECosW),
            (false, true) => coordinates.push(Coordinate::ESinW),
            (false, false) => {}
        }
        if free(Parameter::Baseline) {
            coordinates.push(Coordinate::Baseline);
        }
        if coordinates.is_empty() {
            return Err(McmcError::NoDimensions);
        }
        Ok(Self {
            model,
            flux: lc.flux(),
            weight: lc.flux_err().iter().map(|e| e.recip()).collect(),
            priors: vec![None; coordinates.len()],
            coordinates,
            fixed: values_of(fit.start(), fit.baseline()),
            density: None,
        })
    }

    /// A prior on `coordinate`, on top of its range.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidPrior`] if `coordinate` is not sampled.
    pub fn with_prior(mut self, coordinate: Coordinate, prior: Prior) -> Result<Self, McmcError> {
        let i = self
            .coordinates
            .iter()
            .position(|&c| c == coordinate)
            .ok_or(McmcError::InvalidPrior {
                name: coordinate.name(),
            })?;
        self.priors[i] = Some(prior);
        Ok(self)
    }

    /// A normal prior on the star's density, `3 pi (a / R*)^3 / (G P^2)`,
    /// in g/cm^3: `mean` with standard deviation `sd`.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidPrior`] unless both are finite and `sd` is
    /// above zero.
    pub fn with_stellar_density(mut self, mean: f64, sd: f64) -> Result<Self, McmcError> {
        if !(mean.is_finite() && sd.is_finite() && sd > 0.0) {
            return Err(McmcError::InvalidPrior {
                name: "stellar density",
            });
        }
        self.density = Some((mean, sd));
        Ok(self)
    }

    /// The coordinates sampled, in order.
    #[must_use]
    pub fn coordinates(&self) -> &[Coordinate] {
        &self.coordinates
    }

    /// The point of the coordinates for `params` and `baseline` (the free
    /// parameters' values, in the coordinates' terms).
    #[must_use]
    pub fn point(&self, params: TransitParams, baseline: f64) -> Vec<f64> {
        let v = values_of(params, baseline);
        self.coordinates
            .iter()
            .map(|&c| coordinate_of(c, &v))
            .collect()
    }

    /// The transit's parameters and baseline at `x`, or `None` outside
    /// their ranges.
    #[must_use]
    pub fn params_at(&self, x: &[f64]) -> Option<(TransitParams, f64)> {
        let v = self.values_at(x)?;
        Some((params_from(&v)?, v[slot(Parameter::Baseline)]))
    }

    /// The star's density in g/cm^3 at `x`, from `a / R*` and the period.
    #[must_use]
    pub fn stellar_density(&self, x: &[f64]) -> Option<f64> {
        self.values_at(x).map(|v| density(&v))
    }

    /// `walkers` starting points drawn from the fit's best parameters and
    /// covariance (a normal in the fit's parameters), each inside the
    /// posterior's support. Where the fit is degenerate (an eccentric orbit
    /// without a density prior, say) its covariance reaches far outside
    /// the support, so the normal is narrowed tenfold, up to six times,
    /// until the walkers fit: a small ball about the best fit, as samplers
    /// are usually started.
    ///
    /// # Errors
    ///
    /// [`McmcError::NoCovariance`] if the fit has none or its parameters
    /// differ from the posterior's set-up; [`McmcError::OutsideSupport`]
    /// if at every width a hundred draws per walker leave too few inside.
    pub fn walkers(
        &self,
        result: &FitResult,
        walkers: usize,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>, McmcError> {
        let cov = result.covariance().ok_or(McmcError::NoCovariance)?;
        let params = cov.parameters();
        let best = values_of(result.params(), result.baseline());
        if params != self.fit_parameters().as_slice() {
            return Err(McmcError::NoCovariance);
        }
        let n = params.len();
        let mut matrix = vec![0.0; n * n];
        for (i, &p) in params.iter().enumerate() {
            for (j, &q) in params.iter().enumerate() {
                matrix[i * n + j] = cov.get(p, q);
            }
        }
        let lower = cholesky(&matrix, n).ok_or(McmcError::NoCovariance)?;
        let mut rng = Rng::new(seed);
        let mut most = 0;
        let mut scale = 1.0;
        for _ in 0..7 {
            let mut out = Vec::with_capacity(walkers);
            for _ in 0..100 * walkers {
                let z: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
                let mut v = best;
                for (i, &p) in params.iter().enumerate() {
                    let step: f64 = (0..=i).map(|j| lower[i * n + j] * z[j]).sum();
                    v[slot(p)] += scale * step;
                }
                let x: Vec<f64> = self
                    .coordinates
                    .iter()
                    .map(|&c| coordinate_of(c, &v))
                    .collect();
                if self.log_probability(&x).is_finite() {
                    out.push(x);
                    if out.len() == walkers {
                        return Ok(out);
                    }
                }
            }
            most = most.max(out.len());
            scale *= 0.1;
        }
        Err(McmcError::OutsideSupport { walker: most })
    }

    /// The fit parameters behind the coordinates, in [`Parameter::ALL`]'s
    /// order.
    fn fit_parameters(&self) -> Vec<Parameter> {
        let mut out: Vec<Parameter> = self
            .coordinates
            .iter()
            .flat_map(|&c| match c {
                Coordinate::MidTime => vec![Parameter::MidTime],
                Coordinate::Period => vec![Parameter::Period],
                Coordinate::RadiusRatio => vec![Parameter::RadiusRatio],
                Coordinate::SemiMajorAxis => vec![Parameter::SemiMajorAxis],
                Coordinate::ImpactParameter => vec![Parameter::ImpactParameter],
                Coordinate::U1 | Coordinate::Q1 => vec![Parameter::U1],
                Coordinate::U2 | Coordinate::Q2 => vec![Parameter::U2],
                Coordinate::ECosW | Coordinate::SqrtECosW => vec![Parameter::ECosW],
                Coordinate::ESinW | Coordinate::SqrtESinW => vec![Parameter::ESinW],
                Coordinate::Baseline => vec![Parameter::Baseline],
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Every parameter's value at `x`, or `None` outside the coordinates'
    /// ranges.
    fn values_at(&self, x: &[f64]) -> Option<Values> {
        values_from_coordinates(&self.coordinates, x, self.fixed)
    }
}

impl LogProbability for TransitPosterior<'_> {
    fn log_probability(&self, x: &[f64]) -> f64 {
        let Some(v) = self.values_at(x) else {
            return f64::NEG_INFINITY;
        };
        let Some(params) = params_from(&v) else {
            return f64::NEG_INFINITY;
        };
        let mut lp: f64 = self
            .priors
            .iter()
            .zip(x)
            .filter_map(|(p, &value)| p.map(|p| p.log_density(value)))
            .sum();
        if let Some((mean, sd)) = self.density {
            let z = (density(&v) - mean) / sd;
            lp -= 0.5 * z * z;
        }
        if lp == f64::NEG_INFINITY {
            return lp;
        }
        let baseline = v[slot(Parameter::Baseline)];
        // Each thread's buffer for the model, kept between calls.
        let chi2 = BUFFER.with_borrow_mut(|model| {
            model.resize(self.flux.len(), 0.0);
            // The lengths match by construction.
            let _ = self.model.light_curve_into(params, model);
            model
                .iter()
                .zip(self.flux)
                .zip(&self.weight)
                .map(|((&m, &y), &w)| {
                    let r = baseline.mul_add(m, -y) * w;
                    r * r
                })
                .sum::<f64>()
        });
        (-0.5f64).mul_add(chi2, lp)
    }
}

/// Every parameter's value with `coordinates` at `x` and the rest at
/// `fixed`, or `None` outside the coordinates' ranges.
pub(crate) fn values_from_coordinates(
    coordinates: &[Coordinate],
    x: &[f64],
    fixed: Values,
) -> Option<Values> {
    let mut v = fixed;
    let (mut q, mut s) = ((None, None), (None, None));
    for (&c, &value) in coordinates.iter().zip(x) {
        let at = match c {
            Coordinate::MidTime => Parameter::MidTime,
            Coordinate::Period => Parameter::Period,
            Coordinate::RadiusRatio => Parameter::RadiusRatio,
            Coordinate::SemiMajorAxis => Parameter::SemiMajorAxis,
            Coordinate::ImpactParameter => Parameter::ImpactParameter,
            Coordinate::U1 => Parameter::U1,
            Coordinate::U2 => Parameter::U2,
            Coordinate::ECosW => Parameter::ECosW,
            Coordinate::ESinW => Parameter::ESinW,
            Coordinate::Baseline => Parameter::Baseline,
            Coordinate::Q1 => {
                q.0 = Some(value);
                continue;
            }
            Coordinate::Q2 => {
                q.1 = Some(value);
                continue;
            }
            Coordinate::SqrtECosW => {
                s.0 = Some(value);
                continue;
            }
            Coordinate::SqrtESinW => {
                s.1 = Some(value);
                continue;
            }
        };
        v[slot(at)] = value;
    }
    if let (Some(q1), Some(q2)) = q {
        if !((0.0..=1.0).contains(&q1) && (0.0..=1.0).contains(&q2)) {
            return None;
        }
        let root = q1.sqrt();
        v[slot(Parameter::U1)] = 2.0 * root * q2;
        v[slot(Parameter::U2)] = root * 2.0f64.mul_add(-q2, 1.0);
    }
    if let (Some(c), Some(s)) = s {
        let root = c.hypot(s);
        v[slot(Parameter::ECosW)] = root * c;
        v[slot(Parameter::ESinW)] = root * s;
    }
    let (k, b) = (
        v[slot(Parameter::RadiusRatio)],
        v[slot(Parameter::ImpactParameter)],
    );
    if !(0.0..1.0 + k).contains(&b) || v[slot(Parameter::Baseline)] <= 0.0 {
        return None;
    }
    Some(v)
}

/// Every parameter's value for `params` and `baseline`.
fn values_of(params: TransitParams, baseline: f64) -> Values {
    let (u1, u2) = params.limb_darkening().quadratic_coefficients();
    let e = params.eccentricity();
    let (e_cos, e_sin) = if e == 0.0 {
        (0.0, 0.0)
    } else {
        let (s, c) = params.periastron_deg().to_radians().sin_cos();
        (e * c, e * s)
    };
    [
        params.t0().as_days(),
        params.period().as_days(),
        params.k().get(),
        params.a_rs().get(),
        params.b().get(),
        u1,
        u2,
        e_cos,
        e_sin,
        baseline,
    ]
}

/// The coordinate's value given every parameter's.
pub(crate) fn coordinate_of(c: Coordinate, v: &Values) -> f64 {
    let (u1, u2) = (v[slot(Parameter::U1)], v[slot(Parameter::U2)]);
    let (ec, es) = (v[slot(Parameter::ECosW)], v[slot(Parameter::ESinW)]);
    let root_e = ec.hypot(es).sqrt();
    match c {
        Coordinate::MidTime => v[slot(Parameter::MidTime)],
        Coordinate::Period => v[slot(Parameter::Period)],
        Coordinate::RadiusRatio => v[slot(Parameter::RadiusRatio)],
        Coordinate::SemiMajorAxis => v[slot(Parameter::SemiMajorAxis)],
        Coordinate::ImpactParameter => v[slot(Parameter::ImpactParameter)],
        Coordinate::U1 => u1,
        Coordinate::U2 => u2,
        Coordinate::Q1 => (u1 + u2) * (u1 + u2),
        // Any q2 at q1 = 0, where it has no effect: a half.
        Coordinate::Q2 => {
            if u1 + u2 == 0.0 {
                0.5
            } else {
                u1 / (2.0 * (u1 + u2))
            }
        }
        Coordinate::ECosW => ec,
        Coordinate::ESinW => es,
        Coordinate::SqrtECosW => {
            if root_e == 0.0 {
                0.0
            } else {
                ec / root_e
            }
        }
        Coordinate::SqrtESinW => {
            if root_e == 0.0 {
                0.0
            } else {
                es / root_e
            }
        }
        Coordinate::Baseline => v[slot(Parameter::Baseline)],
    }
}

/// The transit's parameters, or `None` where the model refuses them.
fn params_from(v: &Values) -> Option<TransitParams> {
    let [t0, period, k, a_rs, b, u1, u2, e_cos, e_sin, _] = *v;
    let circular = TransitParams::new(
        Epoch::from_days(t0).ok()?,
        TimeDelta::from_days(period).ok()?,
        RadiusRatio::new(k).ok()?,
        ScaledSemiMajorAxis::new(a_rs).ok()?,
        ImpactParameter::new(b).ok()?,
        LimbDarkening::quadratic(u1, u2).ok()?,
    )
    .ok()?;
    if e_cos == 0.0 && e_sin == 0.0 {
        return Some(circular);
    }
    circular
        .with_eccentricity(e_cos.hypot(e_sin), e_sin.atan2(e_cos).to_degrees())
        .ok()
}

/// The star's density in g/cm^3 from `a / R*` and the period.
pub(crate) fn density(v: &Values) -> f64 {
    let period = v[slot(Parameter::Period)] * SECONDS_PER_DAY;
    3.0 * PI * v[slot(Parameter::SemiMajorAxis)].powi(3) / (G * period * period) / 1000.0
}

/// The lower Cholesky factor of a symmetric `n` by `n` matrix (row-major),
/// or `None` unless it is positive definite.
pub(crate) fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                let d = a[i * n + i] - sum;
                if d <= 0.0 || !d.is_finite() {
                    return None;
                }
                l[i * n + i] = d.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    Some(l)
}
