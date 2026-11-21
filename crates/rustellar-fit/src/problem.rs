//! The transit fit as weighted residuals over the free parameters.

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitModel, TransitParams,
};

use crate::lm::Residuals;
use crate::{Parameter, TransitFit};

/// The residuals `(baseline * model - flux) / err` of a set-up, over its
/// free parameters; the fixed ones keep their starting values.
pub(crate) struct Problem<'a> {
    model: TransitModel,
    flux: &'a [f64],
    weight: Vec<f64>,
    /// Every parameter's value, the free ones overwritten on each call.
    values: [f64; 8],
    free: Vec<usize>,
}

impl<'a> Problem<'a> {
    pub(crate) fn new(fit: &TransitFit<'a>, values: [f64; 8]) -> Self {
        let times = fit.lc.time().to_vec();
        let model = match fit.exposure {
            Some(exposure) => TransitModel::with_exposure(times, exposure),
            None => TransitModel::new(times),
        };
        Self {
            model,
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
        for (xi, &i) in x.iter_mut().zip(&self.free) {
            *xi = v[i];
        }
    }
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
