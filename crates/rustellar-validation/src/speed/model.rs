//! `rustellar-model` against batman, PyTransit and exoplanet-core: one
//! thread, the hot Jupiter of the crate's quick start (P = 3 d, k = 0.1,
//! a/R* = 10, b = 0.3, u = (0.4, 0.25)), times spread evenly over ±0.1 days
//! around mid-transit, so about half are in transit.

use std::fmt::Write;
use std::num::NonZeroUsize;

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    Exposure, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitModel,
    TransitParams, flux,
};

use super::{Reference, best, package, seconds, short};

/// The light curves' sizes.
pub const SIZES: [usize; 4] = [100, 1_000, 10_000, 100_000];
/// Samples per Kepler long-cadence exposure (29.4 minutes), 1,000 times.
pub const SAMPLES: [usize; 3] = [5, 15, 30];

/// The references' times, as the script that ran them recorded them.
#[must_use]
pub fn reference() -> &'static str {
    include_str!("../../data/speed_model.csv")
}

/// One time: a case (`curve 100`, `flux 100`, `exposure 5`) and seconds.
pub type Timing = (String, f64);

fn hot_jupiter() -> TransitParams {
    TransitParams::new(
        Epoch::from_btjd(0.0).expect("finite"),
        TimeDelta::from_days(3.0).expect("positive"),
        RadiusRatio::new(0.1).expect("valid"),
        ScaledSemiMajorAxis::new(10.0).expect("valid"),
        ImpactParameter::new(0.3).expect("valid"),
        LimbDarkening::quadratic(0.4, 0.25).expect("valid"),
    )
    .expect("valid")
}

/// The `i`-th of `n` times, in days from mid-transit.
fn day(i: usize, n: usize) -> f64 {
    let small = |v: usize| f64::from(u32::try_from(v).expect("small"));
    -0.1 + 0.2 * small(i) / small(n - 1)
}

fn times(n: usize) -> Vec<Epoch> {
    (0..n)
        .map(|i| Epoch::from_btjd(day(i, n)).expect("finite"))
        .collect()
}

/// Calls per repeat for `n` times, as the reference script: 100,000 / n.
fn number(n: usize) -> u32 {
    u32::try_from((100_000 / n).max(1)).expect("small")
}

/// Every case timed, `repeats` repeats each (seven for the README).
#[must_use]
pub fn run(repeats: usize) -> Vec<Timing> {
    let params = hot_jupiter();
    let (k, ld) = (params.k(), params.limb_darkening());
    let mut out = vec![];
    for n in SIZES {
        let mut model = TransitModel::new(times(n));
        let t = best(repeats, number(n), || model.light_curve(params).len());
        out.push((format!("curve {n}"), t));
        // The separations at the same times, as exoplanet-core takes them.
        let z: Vec<f64> = (0..n)
            .map(|i| {
                let phi = 2.0 * std::f64::consts::PI * day(i, n) / 3.0;
                (10.0 * phi.sin()).hypot(0.3 * phi.cos())
            })
            .collect();
        let t = best(repeats, number(n), || {
            z.iter().map(|&z| flux(z, k, ld)).collect::<Vec<_>>()
        });
        out.push((format!("flux {n}"), t));
    }
    for samples in SAMPLES {
        let exposure = Exposure::new(
            TimeDelta::from_hours(29.4 / 60.0).expect("positive"),
            NonZeroUsize::new(samples).expect("not zero"),
        )
        .expect("valid");
        let mut model = TransitModel::with_exposure(times(1_000), exposure);
        let t = best(repeats, 20, || model.light_curve(params).len());
        out.push((format!("exposure {samples}"), t));
    }
    out
}

/// A row's cells: ours where `name` is None, else the reference's.
fn cells(
    timings: &[Timing],
    references: &[Reference],
    name: Option<&str>,
    cases: impl Iterator<Item = String>,
) -> String {
    cases
        .map(|case| {
            short(name.map_or_else(
                || timings.iter().find(|(c, _)| *c == case).map(|&(_, s)| s),
                |name| seconds(references, name, &case),
            ))
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

/// The README's two tables: light curves by size, then exposures.
#[must_use]
pub fn report(timings: &[Timing], references: &[Reference]) -> String {
    let mut out =
        String::from("| | 100 times | 1,000 | 10,000 | 100,000 |\n|---|--:|--:|--:|--:|\n");
    let rows = [
        ("rustellar-model".to_owned(), None, "curve"),
        (package(references, "batman"), Some("batman"), "curve"),
        (package(references, "PyTransit"), Some("PyTransit"), "curve"),
        (
            "rustellar-model, flux only (no orbit)".to_owned(),
            None,
            "flux",
        ),
        (
            format!(
                "{}, flux only (no orbit)",
                package(references, "exoplanet-core")
            ),
            Some("exoplanet-core"),
            "flux",
        ),
    ];
    for (label, name, kind) in rows {
        let row = cells(
            timings,
            references,
            name,
            SIZES.iter().map(|n| format!("{kind} {n}")),
        );
        writeln!(out, "| {label} | {row} |").expect("a string");
    }
    out.push_str(
        "\nKepler long cadence (29.4 min), 1,000 times:\n\n\
         | Samples per exposure | 5 | 15 | 30 |\n|---|--:|--:|--:|\n",
    );
    for (label, name) in [
        ("rustellar-model".to_owned(), None),
        (package(references, "batman"), Some("batman")),
        (package(references, "PyTransit"), Some("PyTransit")),
    ] {
        let row = cells(
            timings,
            references,
            name,
            SAMPLES.iter().map(|s| format!("exposure {s}")),
        );
        writeln!(out, "| {label} | {row} |").expect("a string");
    }
    out
}
