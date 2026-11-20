// Speed of the search, for the numbers in the README. The light curves are
// the shared survey recipe's (the same seeds), and the periods Ofir's grid
// with TLS's defaults, so transitleastsquares can be timed on the same data
// (its grid starts a little shorter, from older solar constants: 2,438
// periods against 2,436 on the 30-minute sector).
//
// Run with `cargo bench -p rustellar-tls`, or with `--features cuda` for
// the GPU too.

#[expect(dead_code, reason = "only the survey recipe")]
#[path = "../../rustellar-bls/src/tests/recipe.rs"]
mod recipe;

use std::{hint::black_box, num::NonZeroUsize, thread, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use recipe::Survey;
use rustellar_core::LightCurve;
use rustellar_tls::{Device, Method, OfirPeriod, SearchOptions, Tls};

const TWO_MINUTES: f64 = 2.0 / 1440.0;
const THIRTY_MINUTES: f64 = 1.0 / 48.0;

// `count` TESS sectors 27.5 days apart, each two orbits of `n` points a
// day apart, around 1 (a transit every 4.7 days, 1e-3 deep).
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
    let time = lc.time().iter().map(|t| t.as_days()).collect();
    let flux = lc.flux().iter().map(|f| 1.0 + f).collect();
    LightCurve::from_days(time, flux, lc.flux_err().to_vec()).unwrap()
}

fn search(c: &mut Criterion) {
    let all =
        NonZeroUsize::new(thread::available_parallelism().map_or(1, NonZeroUsize::get)).unwrap();
    let cases = [
        ("1 sector, 30 min", sectors(80, 1, THIRTY_MINUTES, 620)),
        ("1 sector, 2 min", sectors(77, 1, TWO_MINUTES, 9000)),
        ("2 sectors, 2 min", sectors(78, 2, TWO_MINUTES, 9000)),
    ];
    let cpu = SearchOptions::default()
        .with_threads(all)
        .with_device(Device::Cpu)
        .with_bootstrap(0);
    let mut runs = vec![
        ("cells", cpu),
        ("cells, bootstrap 100", cpu.with_bootstrap(100)),
        ("fold", cpu.with_method(Method::Fold)),
    ];
    if cfg!(feature = "cuda") {
        runs.push(("cells, GPU", cpu.with_device(Device::Gpu(0))));
    }
    let mut group = c.benchmark_group("search");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(1));
    for (name, lc) in &cases {
        let tls = Tls::new(lc).unwrap();
        let periods = OfirPeriod::default().periods(lc).unwrap();
        for (run, options) in &runs {
            let id = BenchmarkId::new(
                *name,
                format!(
                    "{run}, {} threads, {} periods",
                    all,
                    periods.as_slice().len()
                ),
            );
            group.bench_function(id, |b| {
                b.iter(|| tls.search(black_box(&periods), options).unwrap())
            });
        }
    }
    group.finish();
}

criterion_group!(benches, search);
criterion_main!(benches);
