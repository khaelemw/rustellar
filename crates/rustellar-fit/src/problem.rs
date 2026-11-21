//! The transit fit as weighted residuals over the free parameters.

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitModel, TransitParams,
};

use crate::jacobian::jacobian;
use crate::lm::Residuals;
use crate::{Parameter, TransitFit};

/// A differencing step's size against its parameter's scale: `eps^(1/3)`,
/// which balances truncation against rounding for central differences.
pub(crate) const STEP: f64 = 6.055_454_452_393_343e-6;

/// The residuals `(baseline * model - flux) / err` of a set-up, over its
/// free parameters; the fixed ones keep their starting values.
pub(crate) struct Problem<'a> {
    model: TransitModel,
    /// The times in days, for the Jacobian.
    pub(crate) times: Vec<f64>,
    /// Each point's sample offsets in days: `[0]` for instants.
    pub(crate) offsets: Vec<f64>,
    flux: &'a [f64],
    pub(crate) weight: Vec<f64>,
    /// Every parameter's value, the free ones overwritten on each call.
    values: [f64; 8],
    pub(crate) free: Vec<usize>,
}

impl<'a> Problem<'a> {
    pub(crate) fn new(fit: &TransitFit<'a>, values: [f64; 8]) -> Self {
        let times = fit.lc.time().to_vec();
        let days = times.iter().map(|t| t.as_days()).collect();
        let (model, offsets) = match fit.exposure {
            Some(exposure) => (
                TransitModel::with_exposure(times, exposure),
                exposure.offsets().map(TimeDelta::as_days).collect(),
            ),
            None => (TransitModel::new(times), vec![0.0]),
        };
        Self {
            model,
            times: days,
            offsets,
            flux: fit.lc.flux(),
            weight: fit.lc.flux_err().iter().map(|e| e.recip()).collect(),
            values,
            free: fit.free().iter().map(|p| p.index()).collect(),
        }
    }

    /// The free parameters' values.
    pub(crate) fn free_values(&self) -> Vec<f64> {
        self.free.iter().map(|&i| self.values[i]).collect()
    }

    /// Every value with the free ones from `x`.
    pub(crate) fn values_at(&self, x: &[f64]) -> [f64; 8] {
        let mut values = self.values;
        for (&i, &v) in self.free.iter().zip(x) {
            values[i] = v;
        }
        values
    }
}

impl Residuals for Problem<'_> {
    fn len(&self) -> usize {
        self.flux.len()
    }

    fn eval(&mut self, x: &[f64], out: &mut [f64]) -> bool {
        let values = self.values_at(x);
        let baseline = values[Parameter::Baseline.index()];
        let Some(params) = params_from(&values) else {
            return false;
        };
        if baseline <= 0.0 {
            return false;
        }
        // The lengths match by construction.
        let _ = self.model.light_curve_into(params, out);
        for ((o, &y), &w) in out.iter_mut().zip(self.flux).zip(&self.weight) {
            *o = baseline.mul_add(*o, -y) * w;
        }
        true
    }

    fn jacobian(&mut self, x: &[f64], jac: &mut [f64]) -> bool {
        jacobian(self, &self.values_at(x), jac)
    }

    fn project(&self, x: &mut [f64]) {
        let mut v = self.values_at(x);
        let [_, period, k, a_rs, b, u1, u2, baseline] = &mut v;
        *period = period.max(f64::MIN_POSITIVE);
        *baseline = baseline.max(f64::MIN_POSITIVE);
        *k = k.max(f64::EPSILON);
        *a_rs = a_rs.max(1.0 + f64::EPSILON);
        if b.abs() >= *a_rs {
            *b = (*a_rs * (1.0 - f64::EPSILON)).copysign(*b);
        }
        // The limb's intensity, 1 - u1 - u2, at zero, by the nearest point
        // of u1 + u2 = 1 (exactly 1 in floating point, as u1 + (1 - u1)).
        let excess = *u1 + *u2 - 1.0;
        if excess > 0.0 {
            *u1 -= 0.5 * excess;
            *u2 = 1.0 - *u1;
        }
        // For u2 < -1 the intensity first reaches zero inside the disc,
        // where u1^2 + 4 u2 = 0: past u1 = 2 sqrt(-u2), to that curve's
        // nearest point.
        if *u2 < -1.0 && *u1 > 2.0 * (-*u2).sqrt() {
            (*u1, *u2) = onto_inner_edge(*u1, *u2);
        }
        for (xi, &i) in x.iter_mut().zip(&self.free) {
            *xi = v[i];
        }
    }
}

/// The point of the curve `(2t, -t^2)`, `t >= 1`, nearest `(u1, u2)`, a
/// point beyond it with `u2 < -1`: the root of `t^3 + (u2 + 2) t - u1`,
/// which rises and is convex from `t = sqrt(-u2)`, where it is negative,
/// so Newton's method from there converges from above after one step.
/// `u2` is `-(u1 / 2)^2` as rounded, so `u1^2 + 4 u2` is exactly zero.
pub(crate) fn onto_inner_edge(u1: f64, u2: f64) -> (f64, f64) {
    let mut t = (-u2).sqrt();
    for _ in 0..100 {
        let step = t.mul_add(t * t, (u2 + 2.0).mul_add(t, -u1)) / (3.0 * t).mul_add(t, u2 + 2.0);
        let next = t - step;
        if next >= t && t > (-u2).sqrt() {
            break;
        }
        t = next;
    }
    let u1 = 2.0 * t.max(1.0);
    (u1, -(0.25 * (u1 * u1)))
}

/// The transit parameters of `values`, with `|b|` as `b` (the flux depends
/// on `b^2` alone), or `None` where they are not a valid transit.
pub(crate) fn params_from(values: &[f64; 8]) -> Option<TransitParams> {
    let [t0, period, k, a_rs, b, u1, u2, _] = *values;
    TransitParams::new(
        Epoch::from_days(t0).ok()?,
        TimeDelta::from_days(period).ok()?,
        RadiusRatio::new(k).ok()?,
        ScaledSemiMajorAxis::new(a_rs).ok()?,
        ImpactParameter::new(b.abs()).ok()?,
        LimbDarkening::quadratic(u1, u2).ok()?,
    )
    .ok()
}

/// Every parameter's value from transit parameters and a baseline.
pub(crate) fn values_of(params: TransitParams, baseline: f64) -> [f64; 8] {
    let (u1, u2) = params.limb_darkening().quadratic_coefficients();
    [
        params.t0().as_days(),
        params.period().as_days(),
        params.k().get(),
        params.a_rs().get(),
        params.b().get(),
        u1,
        u2,
        baseline,
    ]
}
