//! `rustellar-tls` against transitleastsquares (its defaults, every core),
//! on the survey recipe's light curves around 1 (the same seeds as the
//! crate's benchmark): one sector at 30 minutes, one and two sectors at 2
//! minutes, a box transit every 4.7 days in white noise, Ofir's grid with
//! TLS's defaults.

use std::fmt::Write;
use std::num::NonZeroUsize;

use rustellar_core::LightCurve;
use rustellar_tls::{Device, Method, OfirPeriod, Periods, SearchOptions, Tls};

use crate::recipe::Survey;

use super::bls::thousands;
use super::{Reference, best, package, seconds, significant};

const TWO_MINUTES: f64 = 2.0 / 1440.0;
const THIRTY_MINUTES: f64 = 1.0 / 48.0;

/// The references' times, as the script that ran them recorded them.
#[must_use]
pub fn reference() -> &'static str {
    include_str!("../../data/speed_tls.csv")
}

/// `count` TESS sectors 27.5 days apart, each two orbits of `n` points a
/// day apart, around 1.
fn sectors(seed: u64, count: u32, cadence: f64, n: u32) -> LightCurve {
    let segments = (0..count)
        .flat_map(|k| {
            let start = 1325.0 + 27.5 * f64::from(k);
            [(start, n), (start + f64::from(n) * cadence + 1.0, n)]
        })
        .collect();
    let lc = Survey {
        seed,
        segments,
        cadence,
        noise: 1e-3,
        keep: 1.0,
        period: 4.7,
        tc: 1326.3,
        duration: 0.12,
        depth: 1e-3,
    }
    .light_curve();
    let time = lc.time().iter().map(|t| t.as_btjd()).collect();
    let flux = lc.flux().iter().map(|f| 1.0 + f).collect();
    LightCurve::from_btjd(time, flux, lc.flux_err().to_vec()).expect("valid")
}

/// A case: its name (as in the reference file), the light curve, its
/// periods and the repeats (three; transitleastsquares had three on the
/// 30-minute sector and one on the others, which take it 13 and 38 seconds,
/// where ours take one to four and vary more on the GPU).
pub struct Case {
    pub name: &'static str,
    pub lc: LightCurve,
    pub periods: Periods,
    pub repeats: usize,
}

/// The cases' names, with their labels in the README.
pub const NAMES: [(&str, &str); 3] = [
    ("1 sector 30 min", "1 sector, 30 min"),
    ("1 sector 2 min", "1 sector, 2 min"),
    ("2 sectors 2 min", "2 sectors, 2 min"),
];

/// The README's cases, over Ofir's grid with TLS's defaults.
#[must_use]
pub fn cases() -> Vec<Case> {
    [
        (sectors(80, 1, THIRTY_MINUTES, 620), 3),
        (sectors(77, 1, TWO_MINUTES, 9000), 3),
        (sectors(78, 2, TWO_MINUTES, 9000), 3),
    ]
    .into_iter()
    .zip(NAMES)
    .map(|((lc, repeats), (name, _))| Case {
        name,
        periods: OfirPeriod::default().periods(&lc).expect("a grid"),
        lc,
        repeats,
    })
    .collect()
}

/// The runs: a name and the options on `threads`. The GPU's only with the
/// `cuda` feature.
fn runs(threads: NonZeroUsize) -> Vec<(&'static str, SearchOptions)> {
    let cpu = SearchOptions::default()
        .with_threads(threads)
        .with_device(Device::Cpu)
        .with_bootstrap(0);
    #[cfg_attr(not(feature = "cuda"), expect(unused_mut, reason = "no GPU run"))]
    let mut runs = vec![
        ("cells", cpu),
        ("bootstrap", cpu.with_bootstrap(100)),
        ("fold", cpu.with_method(Method::Fold)),
    ];
    #[cfg(feature = "cuda")]
    runs.push(("gpu", cpu.with_device(Device::Gpu(0))));
    runs
}

/// One time: a case and a run (`1 sector 2 min cells`), and seconds; and
/// each case's points and periods (`points 1 sector 2 min`).
pub type Timing = (String, f64);

fn count(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("small"))
}

/// Each case and run timed on `threads`, the best of the case's repeats.
#[must_use]
pub fn run(cases: &[Case], threads: NonZeroUsize) -> Vec<Timing> {
    let mut out = vec![];
    for case in cases {
        let tls = Tls::new(&case.lc).expect("valid");
        for (run, options) in runs(threads) {
            let s = best(case.repeats, 1, || {
                tls.search(&case.periods, &options).expect("valid")
            });
            out.push((format!("{} {run}", case.name), s));
        }
        out.push((format!("points {}", case.name), count(case.lc.len())));
        out.push((
            format!("periods {}", case.name),
            count(case.periods.as_slice().len()),
        ));
    }
    out
}

fn ours(timings: &[Timing], case: &str) -> Option<f64> {
    timings.iter().find(|(c, _)| c == case).map(|&(_, s)| s)
}

fn cell(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |s| significant(s, 3))
}

fn whole(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), thousands)
}

/// The README's table, in seconds per search on `threads`; then `Fold`.
#[must_use]
pub fn report(timings: &[Timing], references: &[Reference], threads: NonZeroUsize) -> String {
    let name = package(references, "transitleastsquares");
    let mut out = format!(
        "Every search on {threads} threads.\n\n\
         | | Points | Periods | {name} | rustellar-tls, no bootstrap | default (bootstrap 100) | GPU, no bootstrap |\n\
         |---|--:|--:|--:|--:|--:|--:|\n"
    );
    let mut fold = vec![];
    for (case, label) in NAMES {
        writeln!(
            out,
            "| {label} | {} | {} | {} | {} | {} | {} |",
            whole(ours(timings, &format!("points {case}"))),
            whole(ours(timings, &format!("periods {case}"))),
            cell(seconds(references, "transitleastsquares", case)),
            cell(ours(timings, &format!("{case} cells"))),
            cell(ours(timings, &format!("{case} bootstrap"))),
            cell(ours(timings, &format!("{case} gpu"))),
        )
        .expect("a string");
        fold.push(cell(ours(timings, &format!("{case} fold"))));
    }
    writeln!(
        out,
        "\n`Method::Fold`, transitleastsquares' own way: {} seconds.",
        fold.join(", ")
    )
    .expect("a string");
    out
}
