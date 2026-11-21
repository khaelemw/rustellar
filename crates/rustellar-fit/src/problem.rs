//! The transit fit as weighted residuals over the free parameters.

use std::thread;

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitModel, TransitParams,
};

use crate::jacobian::jacobian;
use crate::lm::Residuals;
use crate::parameter::{BASE, COUNT};
use crate::{LimbDarkeningBounds, Parameter, TransitFit};

/// A differencing step's size against its parameter's scale: `eps^(1/3)`,
/// which balances truncation against rounding for central differences.
pub(crate) const STEP: f64 = 6.055_454_452_393_343e-6;

/// Points are shared out over threads in blocks this long, dealt in turn,
/// so each thread has its share of the points in transit.
pub(crate) const BLOCK: usize = 256;

/// The points of thread `t` of `threads`, out of `m`.
pub(crate) fn rows(t: usize, threads: usize, m: usize) -> impl Iterator<Item = usize> {
    (t * BLOCK..m)
        .step_by(threads * BLOCK)
        .flat_map(move |start| start..(start + BLOCK).min(m))
}

/// One thread's points for the model: their indices, a model at their
/// times, and the flux there.
struct Part {
    rows: Vec<usize>,
    model: TransitModel,
    flux: Vec<f64>,
}

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
    values: [f64; COUNT],
    pub(crate) free: Vec<usize>,
    /// Whether the limb darkening is held to Kipping's triangle.
    physical: bool,
    /// Each thread's points, beyond the first thread.
    parts: Vec<Part>,
    /// Each thread's Jacobian rows with shape derivatives.
    pub(crate) rows_out: Vec<Vec<(usize, [f64; COUNT])>>,
    /// The rows with shape derivatives in the last Jacobian.
    pub(crate) active: Vec<usize>,
    /// Whether the last Jacobian was this crate's, so only its rows in
    /// transit need clearing.
    pub(crate) written: bool,
}

impl<'a> Problem<'a> {
    pub(crate) fn new(fit: &TransitFit<'a>, values: [f64; COUNT]) -> Self {
        let times = fit.lc.time().to_vec();
        let days = times.iter().map(|t| t.as_days()).collect();
        let threads = fit.threads.get();
        let model_of = |times: Vec<Epoch>| match fit.exposure {
            Some(exposure) => TransitModel::with_exposure(times, exposure),
            None => TransitModel::new(times),
        };
        let parts = if threads > 1 {
            (0..threads)
                .map(|t| {
                    let rows: Vec<usize> = rows(t, threads, times.len()).collect();
                    Part {
                        model: model_of(rows.iter().map(|&i| times[i]).collect()),
                        flux: vec![0.0; rows.len()],
                        rows,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
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
            physical: fit.bounds == LimbDarkeningBounds::Physical,
            parts,
            rows_out: vec![Vec::new(); threads],
            active: Vec::new(),
            written: false,
        }
    }

    /// The free parameters' values.
    pub(crate) fn free_values(&self) -> Vec<f64> {
        self.free.iter().map(|&i| self.values[i]).collect()
    }

    /// Every value with the free ones from `x`.
    pub(crate) fn values_at(&self, x: &[f64]) -> [f64; COUNT] {
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
        let baseline = values[BASE];
        let Some(params) = params_from(&values) else {
            return false;
        };
        if baseline <= 0.0 {
            return false;
        }
        // The lengths match by construction.
        if self.parts.is_empty() {
            let _ = self.model.light_curve_into(params, out);
        } else {
            thread::scope(|scope| {
                for part in &mut self.parts {
                    scope.spawn(move || {
                        let _ = part.model.light_curve_into(params, &mut part.flux);
                    });
                }
            });
            for part in &self.parts {
                for (&i, &v) in part.rows.iter().zip(&part.flux) {
                    out[i] = v;
                }
            }
        }
        for ((o, &y), &w) in out.iter_mut().zip(self.flux).zip(&self.weight) {
            *o = baseline.mul_add(*o, -y) * w;
        }
        true
    }

    fn jacobian(&mut self, x: &[f64], jac: &mut [f64]) -> bool {
        let values = self.values_at(x);
        let given = jacobian(self, &values, jac);
        // Differenced instead: every entry may be set.
        self.written &= given;
        given
    }

    fn bounds(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let v = self.values_at(x);
        let (u1, u2) = (v[Parameter::U1.index()], v[Parameter::U2.index()]);
        let at = |p: Parameter| self.free.iter().position(|&i| i == p.index());
        let (i1, i2) = (at(Parameter::U1), at(Parameter::U2));
        let mut out = Vec::new();
        let mut push = |g1: f64, g2: f64| {
            let mut normal = vec![0.0; x.len()];
            if let Some(i) = i1 {
                normal[i] = g1;
            }
            if let Some(i) = i2 {
                normal[i] = g2;
            }
            if normal.iter().any(|&g| g != 0.0) {
                out.push(normal);
            }
        };
        // On a bound to within the rounding of the projection onto it.
        let near = 4.0 * f64::EPSILON;
        if u1 + u2 >= 1.0 - near {
            push(-1.0, -1.0);
        }
        // The triangle's other edges; the valid range's curved inner edge is
        // left to the projection, as a step along its tangent leaves it.
        if self.physical {
            if u1 <= near {
                push(1.0, 0.0);
            }
            if 2.0f64.mul_add(u2, u1) <= near {
                push(1.0, 2.0);
            }
        }
        out
    }

    fn project(&self, x: &mut [f64]) {
        let mut v = self.values_at(x);
        let [_, period, k, a_rs, b, u1, u2, e_cos, e_sin, baseline] = &mut v;
        *period = period.max(f64::MIN_POSITIVE);
        *baseline = baseline.max(f64::MIN_POSITIVE);
        *k = k.max(f64::EPSILON);
        *a_rs = a_rs.max(1.0 + f64::EPSILON);
        if b.abs() >= *a_rs {
            *b = (*a_rs * (1.0 - f64::EPSILON)).copysign(*b);
        }
        // The periastron outside the star, a (1 - e) > 1, by shrinking e
        // with w kept.
        let e = e_cos.hypot(*e_sin);
        let most = (1.0 - a_rs.recip()) * (1.0 - 4.0 * f64::EPSILON);
        if e > most {
            *e_cos *= most / e;
            *e_sin *= most / e;
        }
        let u_free = (
            self.free.contains(&Parameter::U1.index()),
            self.free.contains(&Parameter::U2.index()),
        );
        if self.physical {
            (*u1, *u2) = onto_physical(*u1, *u2, u_free);
            for (xi, &i) in x.iter_mut().zip(&self.free) {
                *xi = v[i];
            }
            return;
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

/// The nearest point to `(u1, u2)` of Kipping's (2013) triangle, with
/// corners `(0, 0)`, `(2, -1)` and `(0, 1)`, moving only the free ones
/// (`free`: whether `u1`, `u2` are): with both, the nearest point of the
/// triangle; with one, the nearest point of the triangle's chord through
/// the fixed one, if it has one.
pub(crate) fn onto_physical(u1: f64, u2: f64, free: (bool, bool)) -> (f64, f64) {
    match free {
        (true, true) => onto_triangle([u1, u2], [[0.0, 0.0], [2.0, -1.0], [0.0, 1.0]]).into(),
        (true, false) => {
            let (lo, hi) = ((-2.0 * u2).max(0.0), 1.0 - u2);
            (if lo <= hi { u1.clamp(lo, hi) } else { u1 }, u2)
        }
        (false, true) => {
            let (lo, hi) = (-0.5 * u1, 1.0 - u1);
            (
                u1,
                if u1 >= 0.0 && lo <= hi {
                    u2.clamp(lo, hi)
                } else {
                    u2
                },
            )
        }
        (false, false) => (u1, u2),
    }
}

/// The nearest point to `p` of the triangle `abc` (Ericson 2005, 5.1.5):
/// `p` itself inside, else on the nearest edge or corner.
fn onto_triangle(p: [f64; 2], [a, b, c]: [[f64; 2]; 3]) -> [f64; 2] {
    let sub = |x: [f64; 2], y: [f64; 2]| [x[0] - y[0], x[1] - y[1]];
    let dot = |x: [f64; 2], y: [f64; 2]| x[0].mul_add(y[0], x[1] * y[1]);
    let along = |x: [f64; 2], d: [f64; 2], t: f64| [t.mul_add(d[0], x[0]), t.mul_add(d[1], x[1])];
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1.mul_add(d4, -(d3 * d2));
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return along(a, ab, d1 / (d1 - d3));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5.mul_add(d2, -(d1 * d6));
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return along(a, ac, d2 / (d2 - d6));
    }
    let va = d3.mul_add(d6, -(d5 * d4));
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return along(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    // Inside.
    p
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
pub(crate) fn params_from(values: &[f64; COUNT]) -> Option<TransitParams> {
    let [t0, period, k, a_rs, b, u1, u2, e_cos, e_sin, _] = *values;
    let circular = TransitParams::new(
        Epoch::from_days(t0).ok()?,
        TimeDelta::from_days(period).ok()?,
        RadiusRatio::new(k).ok()?,
        ScaledSemiMajorAxis::new(a_rs).ok()?,
        ImpactParameter::new(b.abs()).ok()?,
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

/// Every parameter's value from transit parameters and a baseline.
pub(crate) fn values_of(params: TransitParams, baseline: f64) -> [f64; COUNT] {
    let (u1, u2) = params.limb_darkening().quadratic_coefficients();
    let e = params.eccentricity();
    let (w_sin, w_cos) = params.periastron_deg().to_radians().sin_cos();
    [
        params.t0().as_days(),
        params.period().as_days(),
        params.k().get(),
        params.a_rs().get(),
        params.b().get(),
        u1,
        u2,
        e * w_cos,
        e * w_sin,
        baseline,
    ]
}
