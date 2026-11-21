//! rustellar-fit against batman 2.5.1 and PyTransit 2.6.18 with scipy
//! 1.17.1's `least_squares` on the same six light curves (a hot Jupiter
//! without and with noise, the same at BJD-sized times, a small planet, a
//! grazing transit and long cadence), every parameter free from the same
//! start: scipy's `lm` with a 2-point Jacobian, `trf` with 2-point and
//! 3-point Jacobians, `trf` with tolerances of 1e-15, and PyTransit with
//! `trf`, 3-point.
//!
//! Each answer is judged by this crate's model: its chi-squared there, so
//! every method is measured on the same scale, less the lowest any method
//! reached. The curves are batman's flux with seeded Gaussian noise; long
//! cadence is averaged over each exposure by the midpoint rule (Kipping
//! 2010).

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use rustellar_fit::{
    Epoch, Exposure, FitResult, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio,
    ScaledSemiMajorAxis, Termination, TimeDelta, TransitFit, TransitParams,
};
use rustellar_model::TransitModel;

const CASES: &str = include_str!("reference/fit_cases.csv");
const POINTS: &str = include_str!("reference/fit_points.csv");
const METHODS: &str = include_str!("reference/fit_methods.csv");

/// The parameters of a circular orbit, in the order the CSV files list
/// them.
const CIRCULAR: [Parameter; 8] = [
    Parameter::MidTime,
    Parameter::Period,
    Parameter::RadiusRatio,
    Parameter::SemiMajorAxis,
    Parameter::ImpactParameter,
    Parameter::U1,
    Parameter::U2,
    Parameter::Baseline,
];

/// A parameter's place in the CSV files' values.
fn slot(p: Parameter) -> usize {
    CIRCULAR.iter().position(|&q| q == p).unwrap()
}

struct Case {
    sigma: f64,
    fixed: Vec<Parameter>,
    exposure: Option<Exposure>,
    truth: [f64; 8],
    start: [f64; 8],
}

struct Method {
    name: String,
    values: [f64; 8],
    own_chi2: f64,
    evaluations: u32,
    errors: [f64; 8],
}

fn eight(fields: &[&str]) -> [f64; 8] {
    std::array::from_fn(|i| fields[i].parse().unwrap())
}

fn cases() -> BTreeMap<String, Case> {
    CASES
        .lines()
        .skip(1)
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let duration: f64 = f[2].parse().unwrap();
            let samples: usize = f[3].parse().unwrap();
            let exposure = (samples > 1).then(|| {
                Exposure::new(
                    TimeDelta::from_days(duration).unwrap(),
                    NonZeroUsize::new(samples).unwrap(),
                )
                .unwrap()
            });
            let fixed = CIRCULAR
                .into_iter()
                .filter(|p| f[4].split(';').any(|n| n == p.name()))
                .collect();
            let case = Case {
                sigma: f[1].parse().unwrap(),
                fixed,
                exposure,
                truth: eight(&f[5..13]),
                start: eight(&f[13..21]),
            };
            (f[0].to_string(), case)
        })
        .collect()
}

fn points() -> BTreeMap<String, (Vec<f64>, Vec<f64>)> {
    let mut out: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for line in POINTS.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let entry = out.entry(f[0].to_string()).or_default();
        entry.0.push(f[1].parse().unwrap());
        entry.1.push(f[2].parse().unwrap());
    }
    out
}

fn methods() -> BTreeMap<String, Vec<Method>> {
    let mut out: BTreeMap<String, Vec<Method>> = BTreeMap::new();
    for line in METHODS.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        out.entry(f[0].to_string()).or_default().push(Method {
            name: f[1].to_string(),
            values: eight(&f[2..10]),
            own_chi2: f[10].parse().unwrap(),
            evaluations: f[11].parse().unwrap(),
            errors: eight(&f[14..22]),
        });
    }
    out
}

/// The transit parameters of `v` (with `|b|`), or `None` where they are
/// not a valid transit (such as limb darkening that is negative at the
/// limb).
fn params(v: &[f64; 8]) -> Option<TransitParams> {
    TransitParams::new(
        Epoch::from_days(v[0]).ok()?,
        TimeDelta::from_days(v[1]).ok()?,
        RadiusRatio::new(v[2]).ok()?,
        ScaledSemiMajorAxis::new(v[3]).ok()?,
        ImpactParameter::new(v[4].abs()).ok()?,
        LimbDarkening::quadratic(v[5], v[6]).ok()?,
    )
    .ok()
}

/// Chi-squared of `v` by this crate's model.
fn chi_squared(lc: &LightCurve, case: &Case, v: &[f64; 8]) -> Option<f64> {
    let p = params(v)?;
    let times = lc.time().to_vec();
    let mut model = match case.exposure {
        Some(e) => TransitModel::with_exposure(times, e),
        None => TransitModel::new(times),
    };
    let flux = model.light_curve(p);
    Some(
        flux.iter()
            .zip(lc.flux())
            .map(|(m, y)| ((v[7] * m - y) / case.sigma).powi(2))
            .sum(),
    )
}

/// The largest distance of `v` from the fit in its standard errors, the
/// mid-time to the nearest transit.
fn distance(result: &FitResult, v: &[f64; 8]) -> f64 {
    CIRCULAR
        .iter()
        .map(|&p| {
            let mut d = v[slot(p)] - result.value(p);
            if p == Parameter::MidTime {
                let period = result.value(Parameter::Period);
                d -= (d / period).round() * period;
            }
            if p == Parameter::ImpactParameter {
                d = v[4].abs() - result.value(p);
            }
            let e = result.estimate(p).error();
            if e > 0.0 { (d / e).abs() } else { 0.0 }
        })
        .fold(0.0, f64::max)
}

#[test]
fn against_batman_and_pytransit() {
    let cases = cases();
    let points = points();
    let methods = methods();
    for (name, case) in &cases {
        let (time, flux) = &points[name];
        let lc = LightCurve::from_days(time.clone(), flux.clone(), vec![case.sigma; time.len()])
            .unwrap();
        let mut fit = TransitFit::new(&lc, params(&case.start).unwrap());
        if let Some(e) = case.exposure {
            fit = fit.with_exposure(e);
        }
        for &p in &case.fixed {
            fit = fit.with_fixed(p);
        }
        let result = fit.run().unwrap();
        // Converged, or at the minimum on a bound (b = 0 here).
        assert!(
            matches!(
                result.termination(),
                Termination::Converged | Termination::PrecisionLimit
            ),
            "{name}"
        );
        let ours = result.chi_squared();
        let values = std::array::from_fn(|i| result.value(CIRCULAR[i]));
        let check = chi_squared(&lc, case, &values).unwrap();
        assert!((check - ours).abs() <= 1e-9 * ours.max(1.0), "{name}");
        eprintln!(
            "{name}: chi2 {ours:.10}, {:?}, {} evaluations",
            result.termination(),
            result.evaluations()
        );

        for m in &methods[name] {
            // By this crate's model, or by the method's own where its
            // answer is outside this model's valid range.
            let (excess, scale) = match chi_squared(&lc, case, &m.values) {
                Some(c) => (c - ours, "ours"),
                None => (m.own_chi2 - ours, "own"),
            };
            assert!(
                excess >= -1e-9 * ours.max(1.0),
                "{name}, {}: {excess}",
                m.name
            );
            // The standard errors, where the method reached the same
            // minimum, t0 where it is reported at the same transit.
            let same_epoch = ((m.values[0] - result.value(Parameter::MidTime))
                / result.value(Parameter::Period))
            .round()
                == 0.0;
            let off = distance(&result, &m.values);
            let errors = CIRCULAR
                .iter()
                .filter(|&&p| result.estimate(p).error() > 0.0)
                .filter(|&&p| p != Parameter::MidTime || same_epoch)
                // Not b at 0, where its error to first order is meaningless.
                .filter(|&&p| {
                    let b = result.estimate(Parameter::ImpactParameter);
                    p != Parameter::ImpactParameter || b.value() > b.error()
                })
                .map(|&p| (m.errors[slot(p)] / result.estimate(p).error() - 1.0).abs())
                .fold(0.0, f64::max);
            // At b = 0 the errors swing with b through its correlations, so
            // they are compared only away from it.
            let b = result.estimate(Parameter::ImpactParameter);
            if off < 1e-2 && b.value() > b.error() {
                assert!(errors < 1e-2, "{name}, {}: errors {errors}", m.name);
            }
            eprintln!(
                "  {:26} excess chi2 {excess:9.2e} ({scale:4})  off by {off:9.2e} sd  {:5} evaluations  errors within {errors:.0e}",
                m.name, m.evaluations,
            );
        }
        if name.ends_with("noiseless") {
            // batman's flux, so the gap between its model and this one.
            for p in CIRCULAR {
                let want = case.truth[slot(p)];
                let mut d = result.value(p) - want;
                if p == Parameter::MidTime {
                    d -= (d / want.max(1.0)).round() * case.truth[1];
                }
                assert!(d.abs() < 2e-5, "{p}: {d}");
            }
        }
    }
}
