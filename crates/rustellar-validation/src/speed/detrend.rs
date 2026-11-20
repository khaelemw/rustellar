//! `rustellar-detrend` against wotan, on the survey recipe's light curves
//! (the same seeds as the crate's survey tests and benchmark): a Kepler
//! quarter with a 1-day window, and a TESS sector at 2 minutes and at 20
//! seconds with half-day windows. wotan runs on one thread.

use std::fmt::Write;
use std::num::NonZeroUsize;

use rustellar_core::LightCurve;
use rustellar_detrend::Method;

#[expect(dead_code, reason = "only the recipe")]
#[allow(
    clippy::pedantic,
    clippy::nursery,
    reason = "the detrend crate's own test recipe, shared as it is"
)]
#[path = "../../../rustellar-detrend/tests/common/recipe.rs"]
mod recipe;

use recipe::{Curve, curves};

use super::{Reference, best, package, seconds, significant};

/// The references' times, as the script that ran them recorded them.
#[must_use]
pub fn reference() -> &'static str {
    include_str!("../../data/speed_detrend.csv")
}

/// The methods on the 2-minute sector, with wotan's defaults: a case name
/// and its label.
pub const METHODS: [(&str, &str); 10] = [
    ("median", "Median"),
    ("mean", "Mean"),
    ("biweight", "Biweight"),
    ("welsch", "Welsch"),
    ("andrews", "Andrews' sine wave"),
    ("huber", "Huber (fixed scale)"),
    ("huber-joint", "Huber's proposal 2"),
    ("hodges-lehmann", "Hodges-Lehmann"),
    ("trimmed", "Trimmed mean"),
    ("winsorized", "Winsorized mean"),
];

fn method(name: &str) -> Method {
    match name {
        "median" => Method::Median,
        "mean" => Method::Mean,
        "welsch" => Method::Welsch { c: 2.11 },
        "andrews" => Method::AndrewsSineWave { c: 1.339 },
        "huber" => Method::Huber { c: 1.28 },
        "huber-joint" => Method::HuberJoint { c: 1.5 },
        "hodges-lehmann" => Method::HodgesLehmann,
        "trimmed" => Method::TrimmedMean { proportion: 0.1 },
        "winsorized" => Method::WinsorizedMean { proportion: 0.1 },
        _ => Method::Biweight { c: 5.0 },
    }
}

/// A light curve of the recipe, with the settings of its own.
pub struct Set {
    curve: Curve,
    pub lc: LightCurve,
}

/// The three light curves: `kepler`, `tess` and `tess 20 s`.
#[must_use]
pub fn light_curves() -> Vec<Set> {
    let all = curves();
    let named = |name: &str| {
        all.iter()
            .find(|c| c.name == name)
            .expect("in the recipe")
            .clone()
    };
    let tess = named("tess");
    let fast = Curve {
        name: "tess 20 s".to_owned(),
        cadence: tess.cadence / 6.0,
        n: tess.n * 6,
        ..tess.clone()
    };
    [named("kepler"), tess, fast]
        .into_iter()
        .map(|c| {
            let built = c.build();
            let lc = c.light_curve(&built.time, &built.flux);
            Set { curve: c, lc }
        })
        .collect()
}

/// One time: a case (`biweight tess 16`: the curve and threads; `method
/// median`, one thread) and seconds.
pub type Timing = (String, f64);

/// The biweight on each of `sets`, on one thread and on `threads`, then
/// every method on one thread on `sets[methods]`; the best of `repeats`
/// calls each.
#[must_use]
pub fn run(repeats: usize, threads: NonZeroUsize, sets: &[Set], methods: usize) -> Vec<Timing> {
    let mut out = vec![];
    for Set { curve, lc } in sets {
        for t in [NonZeroUsize::MIN, threads] {
            let detrend = curve.detrend().with_threads(t);
            let s = best(repeats, 1, || detrend.trend(lc, None).expect("valid"));
            out.push((format!("biweight {} {t}", curve.name), s));
        }
    }
    let Set { curve, lc } = &sets[methods];
    for (name, _) in METHODS {
        let detrend = curve
            .detrend()
            .with_method(method(name))
            .with_threads(NonZeroUsize::MIN);
        let s = best(repeats, 1, || detrend.trend(lc, None).expect("valid"));
        out.push((format!("method {name}"), s));
    }
    out
}

fn ours(timings: &[Timing], case: &str) -> String {
    timings
        .iter()
        .find(|(c, _)| c == case)
        .map_or_else(|| "-".to_owned(), |&(_, s)| significant(s, 3))
}

fn theirs(references: &[Reference], case: &str) -> String {
    seconds(references, "wotan", case).map_or_else(|| "-".to_owned(), |s| significant(s, 3))
}

/// The README's two tables, in seconds: the biweight on each light curve,
/// then each method on the 2-minute sector.
#[must_use]
pub fn report(timings: &[Timing], references: &[Reference], threads: NonZeroUsize) -> String {
    let wotan = package(references, "wotan");
    let mut out = format!(
        "| Biweight | {wotan} | rustellar-detrend, 1 thread | rustellar-detrend, {threads} threads |\n\
         |---|--:|--:|--:|\n"
    );
    for (name, label) in [
        ("kepler", "Kepler quarter"),
        ("tess", "TESS, 2 minutes"),
        ("tess 20 s", "TESS, 20 seconds"),
    ] {
        writeln!(
            out,
            "| {label} | {} | {} | {} |",
            theirs(references, &format!("biweight {name}")),
            ours(timings, &format!("biweight {name} 1")),
            ours(timings, &format!("biweight {name} {threads}")),
        )
        .expect("a string");
    }
    write!(
        out,
        "\nEach method on the 2-minute sector, one thread (wotan's default settings):\n\n\
         | | {wotan} | rustellar-detrend |\n|---|--:|--:|\n"
    )
    .expect("a string");
    for (name, label) in METHODS {
        let case = format!("method {name}");
        writeln!(
            out,
            "| {label} | {} | {} |",
            theirs(references, &case),
            ours(timings, &case)
        )
        .expect("a string");
    }
    out
}
