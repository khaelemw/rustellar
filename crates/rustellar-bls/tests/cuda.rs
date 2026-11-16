// The search with its binned step on the GPU against the CPU's: the same
// peaks, refined boxes and best box to the bit, and the power and boxes at
// every period to a rounding. Skipped, with the reason printed, where there
// is no GPU.
#![cfg(feature = "cuda")]

#[expect(dead_code, reason = "only the survey recipe")]
#[path = "../src/tests/recipe.rs"]
mod recipe;

use std::num::NonZeroUsize;

use recipe::Survey;
use rustellar_bls::{
    Bls, BlsError, BlsResult, Device, Durations, Objective, OfirPeriod, Periods, SearchOptions,
};
use rustellar_core::{LightCurve, TimeDelta};

// ==========================
// Helper Functions
// ==========================

const TWO_MINUTES: f64 = 2.0 / 1440.0;

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

fn kepler(quarters: u32) -> LightCurve {
    Survey {
        seed: 79,
        segments: (0..quarters)
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

fn durations() -> Durations {
    let days: Vec<f64> = (0..12).map(|k| 0.01 + 0.34 * f64::from(k) / 11.0).collect();
    Durations::from_days(&days).unwrap()
}

fn ofir(lc: &LightCurve, longest: Option<f64>) -> Periods {
    OfirPeriod {
        minimum_period: Some(TimeDelta::from_days(0.6).unwrap()),
        maximum_period: longest.map(|d| TimeDelta::from_days(d).unwrap()),
        ..OfirPeriod::default()
    }
    .periods(lc)
    .unwrap()
}

// The same peaks and best box to the bit; at every period the same box
// within a rounding, or (where two boxes are that close) another as high.
fn same_search(cpu: &BlsResult, gpu: &BlsResult, what: &str) {
    assert_eq!(gpu.peaks(), cpu.peaks(), "{what}: peaks");
    assert_eq!(gpu.best(), cpu.best(), "{what}: best");
    assert_eq!(gpu.power().len(), cpu.power().len());
    let mut others = 0;
    for (k, (g, c)) in gpu.boxes().iter().zip(cpu.boxes()).enumerate() {
        let (g_power, c_power) = (gpu.power()[k], cpu.power()[k]);
        assert!(
            (g_power - c_power).abs() <= 1e-12 * c_power,
            "{what} {k}: power {g_power} against {c_power}"
        );
        match (g, c) {
            (None, None) => {}
            (Some(g), Some(c)) => {
                if g.transit != c.transit || g.fit.n_in != c.fit.n_in {
                    others += 1;
                }
            }
            _ => panic!("{what} {k}: a box against none"),
        }
    }
    println!(
        "{what}: {others} of {} periods report another box as high",
        cpu.power().len()
    );
    assert!(
        others * 1000 <= cpu.power().len(),
        "{what}: {others} other boxes"
    );
}

fn compare(lc: &LightCurve, periods: &Periods, what: &str) {
    let bls = Bls::new(lc).unwrap();
    for objective in [
        Objective::DeltaChi2,
        Objective::Snr,
        Objective::AstropyLikelihood,
    ] {
        for refine in [0, 5] {
            let options = SearchOptions::default()
                .with_objective(objective)
                .with_refine_peaks(refine)
                .with_threads(NonZeroUsize::new(8).unwrap());
            let cpu = bls
                .search(periods, &durations(), &options.with_device(Device::Cpu))
                .unwrap();
            assert_eq!(cpu.device(), Device::Cpu);
            let on_gpu =
                match bls.search(periods, &durations(), &options.with_device(Device::Gpu(0))) {
                    Ok(found) => found,
                    Err(BlsError::GpuUnavailable) => {
                        println!("skipped: no GPU");
                        return;
                    }
                    Err(error) => panic!("{error}"),
                };
            assert_eq!(on_gpu.device(), Device::Gpu(0));
            same_search(
                &cpu,
                &on_gpu,
                &format!("{what} {objective:?} refine {refine}"),
            );
        }
    }
}

// ==========================
// Tests
// ==========================

#[test]
fn one_sector_is_the_cpus_search() {
    let lc = sectors(77, 1);
    compare(&lc, &ofir(&lc, None), "1 sector");
}

#[test]
fn three_sectors_are_the_cpus_search() {
    let lc = sectors(78, 3);
    compare(&lc, &ofir(&lc, None), "3 sectors");
}

// Few points: the folds' sort all in shared memory (under 2048 points), and
// with one step between (2048 to 4096).
#[test]
fn small_curves_are_the_cpus_search() {
    for (seed, points) in [(80, 700), (81, 3000)] {
        let lc = Survey {
            seed,
            segments: vec![(1325.0, points)],
            cadence: 0.01,
            noise: 1e-3,
            keep: 0.9,
            period: 2.3,
            tc: 1325.7,
            duration: 0.1,
            depth: 2e-3,
        }
        .light_curve();
        compare(&lc, &ofir(&lc, None), &format!("{points} points"));
    }
}

// Long periods, whose bins go a tile at a time.
#[test]
fn kepler_years_are_the_cpus_search() {
    let lc = kepler(6);
    compare(&lc, &ofir(&lc, Some(150.0)), "Kepler 6 quarters");
}

// Asking for a GPU there is not, as on a machine with none or with no CUDA
// driver, is an error, never a panic; the default searches on the CPU
// instead, with the same peaks.
#[test]
fn a_missing_gpu_is_an_error_or_the_cpu() {
    let lc = sectors(82, 1);
    let bls = Bls::new(&lc).unwrap();
    let periods = ofir(&lc, Some(5.0));
    let options = SearchOptions::default();
    assert_eq!(
        bls.search(
            &periods,
            &durations(),
            &options.with_device(Device::Gpu(999))
        ),
        Err(BlsError::GpuUnavailable)
    );
    let cpu = bls
        .search(&periods, &durations(), &options.with_device(Device::Cpu))
        .unwrap();
    let auto = bls.search(&periods, &durations(), &options).unwrap();
    println!("the default ran on {:?}", auto.device());
    assert!(matches!(auto.device(), Device::Cpu | Device::Gpu(0)));
    assert_eq!(auto.peaks(), cpu.peaks());
}
