//! `rustellar-bls` against astropy's `BoxLeastSquares` (`fast`,
//! oversampling 10, one thread), on the survey recipe's light curves (the
//! same seeds as the crate's benchmark): one TESS sector at 2 minutes,
//! thirteen sectors, and four years of Kepler at 29.4 minutes; 12 durations
//! from 0.01 to 0.35 days, the SNR objective, no refined peaks.

use std::fmt::Write;
use std::num::NonZeroUsize;

use rustellar_bls::{Bls, Durations, Objective, OfirPeriod, Periods, SearchOptions};
use rustellar_core::{LightCurve, TimeDelta};

use crate::recipe::Survey;

use super::{Reference, best, package, seconds, significant};

const TWO_MINUTES: f64 = 2.0 / 1440.0;

/// The references' times, as the script that ran them recorded them.
#[must_use]
pub fn reference() -> &'static str {
    include_str!("../../data/speed_bls.csv")
}

/// TESS sectors of 2-minute points: two orbits of 9,000 points, a day
/// apart, sectors 27.5 days apart.
fn sectors(seed: u64, count: u32) -> LightCurve {
    let segments = (0..count)
        .flat_map(|k| {
            let start = 1325.0 + 27.5 * f64::from(k);
            [(start, 9000), (start + 9000.0 * TWO_MINUTES + 1.0, 9000)]
        })
        .collect();
    Survey {
        seed,
        segments,
        cadence: TWO_MINUTES,
        noise: 1e-3,
        keep: 1.0,
        period: 4.7,
        tc: 1326.3,
        duration: 0.12,
        depth: 1e-3,
    }
    .light_curve()
}

/// Four years of 29.4-minute points: 16 quarters of 89 days, a day apart,
/// 5% of the points lost.
fn kepler() -> LightCurve {
    Survey {
        seed: 79,
        segments: (0..16)
            .map(|q| (1325.0 + 90.0 * f64::from(q), 4360))
            .collect(),
        cadence: 0.0204,
        noise: 3e-4,
        keep: 0.95,
        period: 37.1,
        tc: 1340.2,
        duration: 0.25,
        depth: 2e-4,
    }
    .light_curve()
}

/// Ofir's grid for a Sun-like star, oversampling 3, from 0.6 days to
/// `longest` (or half the baseline).
fn ofir(lc: &LightCurve, longest: Option<f64>) -> Periods {
    OfirPeriod {
        minimum_period: Some(TimeDelta::from_days(0.6).expect("positive")),
        maximum_period: longest.map(|d| TimeDelta::from_days(d).expect("positive")),
        ..OfirPeriod::default()
    }
    .periods(lc)
    .expect("a grid")
}

/// A case: its name (as in the reference file), the light curve and the
/// periods.
pub struct Case {
    pub name: &'static str,
    pub lc: LightCurve,
    pub periods: Periods,
}

/// The cases' names, with their labels in the README.
pub const NAMES: [(&str, &str); 4] = [
    ("1 sector 10000 periods", "1 sector"),
    ("1 sector Ofir", "1 sector, Ofir"),
    ("13 sectors Ofir", "13 sectors, Ofir"),
    ("Kepler 4 years Ofir to 100 d", "Kepler, Ofir"),
];

/// The README's cases.
#[must_use]
pub fn cases() -> Vec<Case> {
    let sector = sectors(77, 1);
    let thirteen = sectors(78, 13);
    let kepler = kepler();
    let log: Vec<f64> = (0..10_000)
        .map(|k| 0.5 * 30.0_f64.powf(f64::from(k) / 9999.0))
        .collect();
    vec![
        Case {
            name: NAMES[0].0,
            lc: sector.clone(),
            periods: Periods::from_days(&log).expect("valid"),
        },
        Case {
            name: NAMES[1].0,
            periods: ofir(&sector, None),
            lc: sector,
        },
        Case {
            name: NAMES[2].0,
            periods: ofir(&thirteen, None),
            lc: thirteen,
        },
        Case {
            name: NAMES[3].0,
            periods: ofir(&kepler, Some(100.0)),
            lc: kepler,
        },
    ]
}

/// One time: a case and its threads (`1 sector Ofir 16`), and seconds; and
/// each case's number of periods (`periods 1 sector Ofir`).
pub type Timing = (String, f64);

/// Each case timed on one thread and on `threads`, the best of three
/// searches (one where points times periods pass a billion, as astropy's).
#[must_use]
pub fn run(cases: &[Case], threads: NonZeroUsize) -> Vec<Timing> {
    let days: Vec<f64> = (0..12).map(|k| 0.01 + 0.34 * f64::from(k) / 11.0).collect();
    let durations = Durations::from_days(&days).expect("valid");
    let mut out = vec![];
    for case in cases {
        let bls = Bls::new(&case.lc).expect("valid");
        let count = case.periods.as_slice().len();
        let repeats = repeats(case.lc.len(), count);
        for t in [NonZeroUsize::MIN, threads] {
            let options = SearchOptions::default()
                .with_objective(Objective::Snr)
                .with_refine_peaks(0)
                .with_threads(t);
            let s = best(repeats, 1, || {
                bls.search(&case.periods, &durations, &options)
                    .expect("valid")
            });
            out.push((format!("{} {t}", case.name), s));
        }
        out.push((
            format!("periods {}", case.name),
            f64::from(u32::try_from(count).expect("small")),
        ));
    }
    out
}

/// Searches timed for `points` and `periods`: three, or one where their
/// product passes a billion (minutes a search), as astropy was timed.
#[must_use]
pub const fn repeats(points: usize, periods: usize) -> usize {
    if points * periods > 1_000_000_000 {
        1
    } else {
        3
    }
}

fn ours(timings: &[Timing], case: &str) -> Option<f64> {
    timings.iter().find(|(c, _)| c == case).map(|&(_, s)| s)
}

fn cell(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |s| significant(s, 3))
}

/// The README's table, in seconds per search.
#[must_use]
pub fn report(timings: &[Timing], references: &[Reference], threads: NonZeroUsize) -> String {
    let astropy = package(references, "astropy");
    let mut out = format!(
        "| | Periods | {astropy} | rustellar-bls, 1 thread | rustellar-bls, {threads} threads |\n\
         |---|--:|--:|--:|--:|\n"
    );
    for (name, label) in NAMES {
        let periods =
            ours(timings, &format!("periods {name}")).map_or_else(|| "-".to_owned(), thousands);
        writeln!(
            out,
            "| {label} | {periods} | {} | {} | {} |",
            cell(seconds(references, "astropy", name)),
            cell(ours(timings, &format!("{name} 1"))),
            cell(ours(timings, &format!("{name} {threads}"))),
        )
        .expect("a string");
    }
    out
}

/// A whole number with commas between thousands: `166,108`.
#[must_use]
pub fn thousands(value: f64) -> String {
    let digits = format!("{value:.0}");
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}
