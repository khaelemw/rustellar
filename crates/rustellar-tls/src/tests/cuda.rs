// The cells search on the GPU against the CPU's: each period's best the
// same to the bit. Where there is no GPU (or no CUDA driver), each test
// prints why and passes, unless RUSTELLAR_REQUIRE_GPU is set: then no GPU
// is a failure.

use std::num::NonZeroUsize;

use rustellar_core::LightCurve;

use super::{Random, light_curve};
use crate::cells::{CELLS_PER_STEP, Search};
use crate::cuda::{FAIL_NEXT, Gpu};
use crate::durations::{self, Constants, Stars};
use crate::fast::{PeriodBest, Settings, Table};
use crate::fit::{Points, Position};
use crate::search::{Device, Method, Options, Outcome, run};
use crate::template::Template;

// ==========================
// Helper Functions
// ==========================

/// The first GPU, or why there is none.
fn gpu() -> Option<Gpu> {
    match Gpu::new(0) {
        Ok(gpu) => Some(gpu),
        Err(why) => {
            assert!(
                std::env::var_os("RUSTELLAR_REQUIRE_GPU").is_none(),
                "no GPU ({why}) with RUSTELLAR_REQUIRE_GPU set"
            );
            println!("skipped: no GPU ({why})");
            None
        }
    }
}

/// A transit at 3.1 days in `span` days of points every `cadence` days.
fn points(seed: u64, span: f64, cadence: f64) -> Points {
    let mut random = Random::new(seed);
    light_curve(
        &mut random,
        &Template::tls_default(),
        Position {
            period: 3.1,
            duration: 0.12,
            t0: 1.3,
        },
        2e-3,
        1e-3,
        (0.0, span, cadence),
    )
}

/// The periods from 0.5 to 13 days, evenly in frequency.
fn periods(count: usize) -> Vec<f64> {
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    (0..count)
        .map(|i| 1.0 / (2.0 - 1.923 * i as f64 / (count - 1) as f64))
        .collect()
}

/// The search of `points` over the TLS stars' durations, on `gpu` if any.
fn search<'a>(
    points: &'a Points,
    table: &'a Table,
    durations: &'a [f64],
    steps: usize,
    gpu: Option<&'a Gpu>,
) -> Search<'a> {
    let origin = points.time[0];
    Search {
        points,
        table,
        durations,
        cells_per_step: CELLS_PER_STEP,
        settings: Settings {
            steps,
            min_depth: 10e-6,
            stars: Stars::TLS,
            constants: Constants::IAU,
        },
        flat_chi2: points.flat_chi2(),
        origin,
        span: points.time[points.time.len() - 1] - origin,
        gpu,
    }
}

/// The best at each period, on the CPU and on `gpu`.
fn on_both(points: &Points, steps: usize, gpu: &Gpu) -> (Vec<PeriodBest>, Vec<PeriodBest>) {
    let table = Table::new(&Template::tls_default(), 4096);
    let periods = periods(400);
    let span = points.time[points.time.len() - 1] - points.time[0];
    let durations = durations::grid_days(
        Constants::IAU,
        Stars::TLS,
        periods[0],
        periods[periods.len() - 1],
        span,
        1.1,
    );
    let threads = NonZeroUsize::new(3).unwrap();
    let cpu = search(points, &table, &durations, steps, None).periods(&periods, threads);
    let on_gpu = search(points, &table, &durations, steps, Some(gpu)).periods(&periods, threads);
    (cpu, on_gpu)
}

// ==========================
// Tests
// ==========================

// Every period's best is the CPU's, to the bit: dense and sparse cadences,
// few and many trial times.
#[test]
fn each_periods_best_is_the_cpus() {
    let Some(gpu) = gpu() else { return };
    for (seed, span, cadence) in [
        (1, 27.0, 1.0 / 48.0),
        (2, 27.0, 1.0 / 720.0),
        (3, 60.0, 0.1),
    ] {
        let points = points(seed, span, cadence);
        for steps in [10, 50] {
            let (cpu, on_gpu) = on_both(&points, steps, &gpu);
            assert!(cpu.iter().any(|p| p.depth > 0.0));
            for (c, g) in cpu.iter().zip(&on_gpu) {
                assert_eq!(
                    c.chi2.to_bits(),
                    g.chi2.to_bits(),
                    "{seed} {steps}: {c:?} {g:?}"
                );
                assert_eq!(c, g, "{seed} {steps}");
            }
        }
    }
}

// Every point above 1: nothing fits at any period, on the GPU as on the
// CPU.
#[test]
fn nothing_fits_anywhere() {
    let Some(gpu) = gpu() else { return };
    let mut points = points(7, 27.0, 1.0 / 48.0);
    for r in &mut points.residual {
        *r = -r.abs() - 1e-4;
    }
    let (cpu, on_gpu) = on_both(&points, 10, &gpu);
    assert!(cpu.iter().all(|p| p.depth == 0.0));
    assert_eq!(cpu, on_gpu);
}

// A GPU that fails leaves the rest of the search to the CPU, with the same
// results.
#[test]
fn a_gpu_that_fails_leaves_the_search_to_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let points = points(4, 27.0, 1.0 / 48.0);
    let (cpu, _) = on_both(&points, 20, &gpu);
    FAIL_NEXT.with(|fail| fail.set(true));
    let (_, fell_back) = on_both(&points, 20, &gpu);
    assert!(!FAIL_NEXT.with(std::cell::Cell::get));
    assert_eq!(cpu, fell_back);
}

// No periods to search: nothing launched, nothing found.
#[test]
fn no_periods_find_nothing() {
    let Some(gpu) = gpu() else { return };
    let points = points(5, 10.0, 1.0 / 48.0);
    let table = Table::new(&Template::tls_default(), 4096);
    let durations = [0.1];
    let s = search(&points, &table, &durations, 10, Some(&gpu));
    let loaded = gpu.load(&points, &table).unwrap();
    assert_eq!(loaded.search(&s, 0.1, 40, &[]), Ok(Vec::new()));
    assert!(format!("{gpu:?}").starts_with("Gpu"));
}

// Where there is no CUDA library at all (RUSTELLAR_EXPECT_NO_CUDA, as on
// CI), setting up a GPU is an error that says so, and the whole search runs
// on the CPU.
#[test]
fn without_cuda_a_gpu_is_an_error() {
    if std::env::var_os("RUSTELLAR_EXPECT_NO_CUDA").is_none() {
        println!("skipped: RUSTELLAR_EXPECT_NO_CUDA is not set");
        return;
    }
    let why = Gpu::new(0).unwrap_err();
    assert!(why.contains("not installed"), "{why}");
    let points = points(8, 10.0, 1.0 / 48.0);
    let lc = LightCurve::from_btjd(
        points.time.clone(),
        points.residual.iter().map(|r| 1.0 - r).collect(),
        vec![1e-3; points.time.len()],
    )
    .unwrap();
    let periods = periods(50);
    let options = |device| Options {
        method: Method::Cells,
        device,
        ..Options::tls()
    };
    // No GPU set up: the CPU's search, and said so.
    assert_eq!(
        run(&Points::new(&lc), &periods, options(Device::Gpu(0))),
        run(&Points::new(&lc), &periods, options(Device::Cpu))
    );
}

// A GPU that is not there is an error, not a panic.
#[test]
fn a_missing_gpu_is_an_error() {
    assert!(Gpu::new(4096).is_err());
    // Asked again, the same.
    assert!(Gpu::new(4096).is_err());
}

// The whole search with the GPU is the CPU's, and with a GPU that is not
// there, the CPU's too.
#[test]
fn the_whole_search_is_the_cpus() {
    let Some(_) = gpu() else { return };
    let points = points(6, 27.0, 1.0 / 48.0);
    let lc = LightCurve::from_btjd(
        points.time.clone(),
        points.residual.iter().map(|r| 1.0 - r).collect(),
        vec![1e-3; points.time.len()],
    )
    .unwrap();
    let periods = periods(300);
    let options = |device| Options {
        method: Method::Cells,
        device,
        ..Options::tls()
    };
    let cpu = run(&Points::new(&lc), &periods, options(Device::Cpu));
    let gpu = run(&Points::new(&lc), &periods, options(Device::Gpu(0)));
    assert_eq!(gpu.gpu, Some(0));
    assert_eq!(Outcome { gpu: None, ..gpu }, cpu);
    // No such GPU: the CPU, and said so.
    assert_eq!(
        run(&Points::new(&lc), &periods, options(Device::Gpu(4096))),
        cpu
    );
}
