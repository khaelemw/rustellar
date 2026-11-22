//! `rustellar-fit` against batman and PyTransit with scipy's
//! `least_squares`: one thread, the light curves and starts of
//! rustellar-fit's reference test (a hot Jupiter without and with noise,
//! the same at BJD-sized times, a small planet, a grazing transit and long
//! cadence), every parameter free but the limb darkening of the small
//! planet and long cadence. Each fit is timed from its set-up to its
//! result, as the references' from the model's set-up to scipy's answer.

use std::fmt::Write;
use std::num::NonZeroUsize;
use std::time::Instant;

use rustellar_core::{Epoch, LightCurve, TimeDelta};
use rustellar_fit::{
    Exposure, ImpactParameter, LimbDarkening, Parameter, RadiusRatio, ScaledSemiMajorAxis,
    TransitFit, TransitParams,
};

use super::{Reference, best, milliseconds, package, seconds};

const CASES: &str = include_str!("../../../rustellar-fit/tests/reference/fit_cases.csv");
const POINTS: &str = include_str!("../../../rustellar-fit/tests/reference/fit_points.csv");

/// The references' methods, as the script names them in each case.
pub const METHODS: [(&str, &str); 5] = [
    ("batman", "lm 2-point"),
    ("batman", "trf 2-point"),
    ("batman", "trf 3-point"),
    ("batman", "tight"),
    ("PyTransit", "PyTransit"),
];

/// The references' times, as the script that ran them recorded them.
#[must_use]
pub fn reference() -> &'static str {
    include_str!("../../data/speed_fit.csv")
}

/// One case: its name, light curve, start, exposure and fixed parameters.
pub struct Case {
    pub name: String,
    pub lc: LightCurve,
    pub start: TransitParams,
    pub exposure: Option<Exposure>,
    pub fixed: Vec<Parameter>,
}

impl Case {
    /// The fit's set-up.
    #[must_use]
    pub fn set_up(&self) -> TransitFit<'_> {
        let mut fit = TransitFit::new(&self.lc, self.start);
        if let Some(e) = self.exposure {
            fit = fit.with_exposure(e);
        }
        for &p in &self.fixed {
            fit = fit.with_fixed(p);
        }
        fit
    }
}

fn number(field: &str) -> f64 {
    field.parse().expect("a number")
}

/// The reference test's six cases.
///
/// # Panics
///
/// Never: the files are the reference test's.
#[must_use]
pub fn cases() -> Vec<Case> {
    CASES
        .lines()
        .skip(1)
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let (time, flux): (Vec<f64>, Vec<f64>) = POINTS
                .lines()
                .skip(1)
                .filter_map(|l| {
                    let p: Vec<&str> = l.split(',').collect();
                    (p[0] == f[0]).then(|| (number(p[1]), number(p[2])))
                })
                .unzip();
            let err = vec![number(f[1]); time.len()];
            let samples: usize = f[3].parse().expect("a count");
            let exposure = (samples > 1).then(|| {
                Exposure::new(
                    TimeDelta::from_days(number(f[2])).expect("positive"),
                    NonZeroUsize::new(samples).expect("positive"),
                )
                .expect("valid")
            });
            let v: Vec<f64> = f[13..21].iter().map(|s| number(s)).collect();
            let start = TransitParams::new(
                Epoch::from_days(v[0]).expect("finite"),
                TimeDelta::from_days(v[1]).expect("positive"),
                RadiusRatio::new(v[2]).expect("valid"),
                ScaledSemiMajorAxis::new(v[3]).expect("valid"),
                ImpactParameter::new(v[4]).expect("valid"),
                LimbDarkening::quadratic(v[5], v[6]).expect("valid"),
            )
            .expect("valid");
            Case {
                name: f[0].to_owned(),
                lc: LightCurve::from_days(time, flux, err).expect("valid"),
                start,
                exposure,
                fixed: Parameter::ALL
                    .into_iter()
                    .filter(|p| f[4].split(';').any(|n| n == p.name()))
                    .collect(),
            }
        })
        .collect()
}

/// One time: a case's name and seconds.
pub type Timing = (String, f64);

/// Every case timed: the best of `repeats` repeats, each the mean of as
/// many fits as take about `budget` seconds (as the reference script).
#[must_use]
pub fn run(cases: &[Case], repeats: usize, budget: f64) -> Vec<Timing> {
    cases
        .iter()
        .map(|case| {
            let fit = || case.set_up().run().expect("a fit");
            let start = Instant::now();
            let _ = fit();
            let once = start.elapsed().as_secs_f64().max(1e-6);
            #[expect(clippy::cast_possible_truncation, reason = "a small count")]
            #[expect(clippy::cast_sign_loss, reason = "positive")]
            let number = ((budget / once) as u32).max(1);
            (case.name.clone(), best(repeats, number, fit))
        })
        .collect()
}

/// The README's table: each case's fit by this crate and by each
/// reference method.
#[must_use]
pub fn report(timings: &[Timing], references: &[Reference]) -> String {
    let batman = package(references, "batman");
    let pytransit = package(references, "PyTransit");
    let mut out = format!(
        "| Case | rustellar-fit | {batman} lm 2-point | trf 2-point | trf 3-point | trf tight | {pytransit} trf 3-point |\n\
         |---|--:|--:|--:|--:|--:|--:|\n"
    );
    let names: Vec<String> = CASES
        .lines()
        .skip(1)
        .filter_map(|l| l.split(',').next().map(str::to_owned))
        .collect();
    for name in names {
        let ours = milliseconds(timings.iter().find(|(c, _)| *c == name).map(|&(_, s)| s));
        write!(out, "| {} | {ours} |", name.replace('_', " ")).expect("a string");
        for (package, method) in METHODS {
            let theirs = milliseconds(seconds(references, package, &format!("{name} {method}")));
            write!(out, " {theirs} |").expect("a string");
        }
        out.push('\n');
    }
    out
}
