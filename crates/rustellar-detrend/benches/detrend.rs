// Speed of the detrender, for the numbers in the README. The light curves
// are the survey recipe's (the same seeds as the survey tests), and wotan is
// timed on the same ones, so the comparison is like for like.
//
// Run with `cargo bench -p rustellar-detrend`.

#[expect(dead_code, reason = "only the recipe")]
#[path = "../tests/common/recipe.rs"]
mod recipe;

use std::{hint::black_box, num::NonZeroUsize, thread, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use recipe::{Curve, curves};
use rustellar_core::LightCurve;
use rustellar_detrend::Method;

// A Kepler quarter (long cadence, 1 d window), a TESS sector at 2 minutes
// (0.5 d), and the same sector at 20 seconds.
fn light_curves() -> Vec<(Curve, LightCurve)> {
    let all = curves();
    let named = |name: &str| all.iter().find(|c| c.name == name).unwrap().clone();
    let tess = named("tess");
    let fast = Curve {
        name: "tess 20 s".to_string(),
        cadence: tess.cadence / 6.0,
        n: tess.n * 6,
        ..tess.clone()
    };
    [named("kepler"), tess, fast]
        .into_iter()
        .map(|c| {
            let b = c.build();
            let lc = c.light_curve(&b.time, &b.flux);
            (c, lc)
        })
        .collect()
}

fn detrend(c: &mut Criterion) {
    let all = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let sets = light_curves();
    let mut group = c.benchmark_group("biweight");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    for (curve, lc) in &sets {
        for threads in [NonZeroUsize::MIN, all] {
            let detrend = curve.detrend().with_threads(threads);
            group.bench_with_input(
                BenchmarkId::new(format!("{} ({} points)", curve.name, lc.len()), threads),
                lc,
                |b, lc| b.iter(|| black_box(detrend.trend(lc, None).unwrap())),
            );
        }
    }
    group.finish();

    // Each method on the 2-minute sector, one thread; wotan's defaults.
    let (curve, lc) = &sets[1];
    let mut group = c.benchmark_group("methods");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    for (name, method) in [
        ("median", Method::Median),
        ("mean", Method::Mean),
        ("biweight", Method::Biweight { c: 5.0 }),
        ("welsch", Method::Welsch { c: 2.11 }),
        ("andrews", Method::AndrewsSineWave { c: 1.339 }),
        ("huber", Method::Huber { c: 1.28 }),
        ("hodges-lehmann", Method::HodgesLehmann),
        ("trimmed", Method::TrimmedMean { proportion: 0.1 }),
        ("winsorized", Method::WinsorizedMean { proportion: 0.1 }),
    ] {
        let detrend = curve
            .detrend()
            .with_method(method)
            .with_threads(NonZeroUsize::MIN);
        group.bench_function(name, |b| {
            b.iter(|| black_box(detrend.trend(lc, None).unwrap()));
        });
    }
    group.finish();
}

criterion_group!(benches, detrend);
criterion_main!(benches);
