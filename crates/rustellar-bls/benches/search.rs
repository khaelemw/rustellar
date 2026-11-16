// Speed of the search, for the numbers in the README. The light curves and
// grids are those timed for astropy in the same table (the shared survey
// recipe, the same seeds), so the comparison is like for like.
//
// Run with `cargo bench -p rustellar-bls`.

#[expect(dead_code, reason = "only the survey recipe")]
#[path = "../src/tests/recipe.rs"]
mod recipe;

use std::{hint::black_box, num::NonZeroUsize, thread, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use recipe::Survey;
use rustellar_bls::{Bls, Durations, Objective, OfirPeriod, Periods, SearchOptions};
use rustellar_core::{LightCurve, TimeDelta};

const TWO_MINUTES: f64 = 2.0 / 1440.0;

// TESS sectors of 2-minute points: two orbits of 9,000 points, a day apart,
// sectors 27.5 days apart.
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

// Four years of Kepler-like long cadence (29.4 minutes): 16 quarters of 89
// days, a day apart, 5% of points lost.
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

// 12 durations, 0.01 to 0.35 days.
fn durations() -> Durations {
    let days: Vec<f64> = (0..12).map(|k| 0.01 + 0.34 * f64::from(k) / 11.0).collect();
    Durations::from_days(&days).unwrap()
}

// Ofir's grid for a Sun-like star, oversampling 3, from 0.6 days to
// `longest` (or half the baseline).
fn ofir(lc: &LightCurve, longest: Option<f64>) -> Periods {
    OfirPeriod {
        minimum_period: Some(TimeDelta::from_days(0.6).unwrap()),
        maximum_period: longest.map(|d| TimeDelta::from_days(d).unwrap()),
        ..OfirPeriod::default()
    }
    .periods(lc)
    .unwrap()
}

fn options(threads: usize, refine: usize) -> SearchOptions {
    SearchOptions::default()
        .with_objective(Objective::Snr)
        .with_refine_peaks(refine)
        .with_threads(NonZeroUsize::new(threads).unwrap())
}

fn search(c: &mut Criterion) {
    let all = thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let durations = durations();
    let sector = sectors(77, 1);
    let log: Vec<f64> = (0..10_000)
        .map(|k| 0.5 * 30.0_f64.powf(f64::from(k) / 9999.0))
        .collect();
    let thirteen = sectors(78, 13);
    let kepler = kepler();
    let cases = [
        (
            "1 sector, 10,000 periods",
            &sector,
            Periods::from_days(&log).unwrap(),
        ),
        ("1 sector, Ofir", &sector, ofir(&sector, None)),
        ("13 sectors, Ofir", &thirteen, ofir(&thirteen, None)),
        (
            "Kepler 4 years, Ofir to 100 d",
            &kepler,
            ofir(&kepler, Some(100.0)),
        ),
    ];
    let mut group = c.benchmark_group("search");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(1));
    for (name, lc, periods) in &cases {
        let bls = Bls::new(lc).unwrap();
        // One thread too on one sector; the large curves take minutes on one.
        let runs: &[(usize, usize)] =
            if lc.time().len() < 100_000 && periods.as_slice().len() < 20_000 {
                &[(1, 0), (all, 0), (all, 5)]
            } else {
                &[(all, 0), (all, 5)]
            };
        for &(threads, refine) in runs {
            let id = BenchmarkId::new(*name, format!("{threads} threads, refine {refine}"));
            let options = options(threads, refine);
            group.bench_function(id, |b| {
                b.iter(|| {
                    bls.search(black_box(periods), &durations, &options)
                        .unwrap()
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, search);
criterion_main!(benches);
