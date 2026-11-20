// The folds made and searched on the GPU against the CPU's: every fold taken
// by the GPU, and each best box the same to the bit. Where there is no GPU
// (or no CUDA driver), each test prints why and passes, unless
// RUSTELLAR_REQUIRE_GPU is set: then no GPU is a failure.

use rustellar_core::LightCurve;

use super::recipe::Survey;
use crate::{
    Bls, BlsError, BlsResult, Device, Durations, Objective, Periods, SearchOptions,
    cuda::{CudaError, FAIL_NEXT, Gpu},
    exact::Folded,
    refine::Folds,
};

// ==========================
// Helper Functions
// ==========================

const OBJECTIVES: [Objective; 3] = [
    Objective::DeltaChi2,
    Objective::Snr,
    Objective::AstropyLikelihood,
];

fn curve(seed: u64, points: u32, cadence: f64) -> Bls {
    let half = points / 2;
    let lc = Survey {
        seed,
        segments: vec![(1325.0, half), (1326.0 + cadence * f64::from(half), half)],
        cadence,
        noise: 1e-3,
        keep: 0.95,
        period: 2.3,
        tc: 1325.7,
        duration: 0.1,
        depth: 2e-3,
    }
    .light_curve();
    Bls::new(&lc).unwrap()
}

// The GPU for `bls`, or why there is none.
fn gpu(bls: &Bls) -> Option<Gpu> {
    match Gpu::new(bls, 0) {
        Ok(gpu) => Some(gpu),
        Err(CudaError::Device(why)) => {
            assert!(
                std::env::var_os("RUSTELLAR_REQUIRE_GPU").is_none(),
                "no GPU ({why}) with RUSTELLAR_REQUIRE_GPU set"
            );
            println!("skipped: no GPU ({why})");
            None
        }
        Err(error) => panic!("{error:?}"),
    }
}

// Periods round the transit's and its aliases, and across the rest; two
// shorter than the longest durations, which their folds skip.
fn periods() -> Vec<f64> {
    let mut periods: Vec<f64> = (0..30).map(|k| 0.7 + 0.31 * f64::from(k)).collect();
    periods.extend([0.2, 0.3]);
    for p in [2.3, 4.6, 1.15] {
        periods.extend((0..5).map(|k| p * (1.0 + 1e-4 * f64::from(k - 2))));
    }
    periods
}

fn durations() -> Vec<f64> {
    (0..12).map(|k| 0.01 + 0.34 * f64::from(k) / 11.0).collect()
}

// ==========================
// Tests
// ==========================

// Few points (the sort all in shared memory), more (a step between), and
// many.
#[test]
fn best_boxes_over_the_durations_are_the_cpus() {
    for (seed, points, cadence) in [(90, 700, 0.02), (91, 3000, 0.005), (92, 20000, 0.0014)] {
        let bls = curve(seed, points, cadence);
        let Some(gpu) = gpu(&bls) else {
            return;
        };
        let (times, weighted) = (&bls.times, &bls.weighted);
        let (periods, durations) = (periods(), durations());
        for objective in OBJECTIVES {
            let found = gpu.folds.best_of(&periods, &durations, objective);
            for (k, &p) in periods.iter().enumerate() {
                let on_gpu = found[k].expect("the GPU takes every fold");
                let on_cpu =
                    Folded::new(times, weighted, p).best_of(weighted, &durations, objective);
                assert_eq!(on_gpu, on_cpu, "{points} points, {objective:?}, P {p}");
            }
        }
    }
}

// A range of durations, with scores to beat that some folds' best boxes
// beat and some do not.
#[test]
fn best_boxes_over_a_range_are_the_cpus() {
    for (seed, points, cadence) in [(93, 700, 0.02), (94, 20000, 0.0014)] {
        let bls = curve(seed, points, cadence);
        let Some(gpu) = gpu(&bls) else {
            return;
        };
        let (times, weighted) = (&bls.times, &bls.weighted);
        for objective in OBJECTIVES {
            let mut jobs = Vec::new();
            for (k, &p) in periods().iter().enumerate() {
                let (shortest, longest) = (0.04 + 0.01 * (k % 5) as f64, 0.18);
                let best = Folded::new(times, weighted, p)
                    .best_between(weighted, shortest, longest, objective, 0.0)
                    .unwrap()
                    .map_or(0.0, |(_, _, fit)| objective.of(&fit));
                for scale in [0.0, 0.5, 1.0, 2.0] {
                    jobs.push([p, shortest, longest, objective.beat(best * scale)]);
                }
            }
            let found = gpu.folds.best_between(&jobs, objective);
            let mut none = 0;
            for (job, on_gpu) in jobs.iter().zip(found) {
                let [p, shortest, longest, beat] = *job;
                let on_gpu = on_gpu.expect("the GPU takes every fold");
                let on_cpu = Folded::new(times, weighted, p)
                    .best_between(weighted, shortest, longest, objective, beat);
                assert_eq!(on_gpu, on_cpu, "{points} points, {objective:?}, {job:?}");
                none += usize::from(matches!(on_cpu, Ok(None)));
            }
            // Some found and some not.
            assert!(none > 0 && none < jobs.len(), "{none} of {}", jobs.len());
            // A range of durations not shorter than the period: the CPU's.
            let found = gpu
                .folds
                .best_between(&[[0.03, 0.04, 0.05, 0.0]], objective);
            assert!(found[0].is_none(), "{found:?}");
        }
    }
}

// Where there is no CUDA library at all (RUSTELLAR_EXPECT_NO_CUDA, as on CI),
// setting up a GPU is an error that says so, never a panic.
#[test]
fn without_cuda_a_gpu_is_an_error() {
    if std::env::var_os("RUSTELLAR_EXPECT_NO_CUDA").is_none() {
        println!("skipped: RUSTELLAR_EXPECT_NO_CUDA is not set");
        return;
    }
    let bls = curve(95, 700, 0.02);
    match Gpu::new(&bls, 0) {
        Err(CudaError::Device(why)) => println!("no GPU: {why}"),
        other => panic!("{:?}", other.map(|_| "a GPU")),
    }
}

// A GPU that fails during a search: the default searches again on the CPU,
// with the CPU's result to the bit, asking for that GPU is an error, and the
// GPU is used again at the next search.
#[test]
fn a_gpu_that_fails_leaves_the_search_to_the_cpu() {
    let bls = curve(96, 700, 0.02);
    if gpu(&bls).is_none() {
        return;
    }
    let mut days: Vec<f64> = periods().into_iter().filter(|&p| p > 0.5).collect();
    days.sort_by(f64::total_cmp);
    let periods = Periods::from_days(&days).unwrap();
    let durations = Durations::from_days(&durations()).unwrap();
    let options = SearchOptions::default();
    let on = |device| options.with_device(device);
    let cpu = bls.search(&periods, &durations, &on(Device::Cpu)).unwrap();

    FAIL_NEXT.with(|fail| fail.set(true));
    let auto = bls.search(&periods, &durations, &options).unwrap();
    assert_eq!(auto.device(), Device::Cpu);
    assert_eq!(auto, cpu);

    FAIL_NEXT.with(|fail| fail.set(true));
    assert_eq!(
        bls.search(&periods, &durations, &on(Device::Gpu(0))),
        Err(BlsError::GpuUnavailable)
    );

    let again = bls
        .search(&periods, &durations, &on(Device::Gpu(0)))
        .unwrap();
    assert_eq!(again.device(), Device::Gpu(0));
    assert_eq!(again.peaks(), cpu.peaks());
}

// `bls`'s search on GPU 0 and on the CPU, or `None` where there is no GPU.
fn on_both(
    bls: &Bls,
    periods: &[f64],
    durations: &[f64],
    options: SearchOptions,
) -> Option<[Result<BlsResult, BlsError>; 2]> {
    gpu(bls)?;
    let (periods, durations) = (
        Periods::from_days(periods).unwrap(),
        Durations::from_days(durations).unwrap(),
    );
    let on = |device| bls.search(&periods, &durations, &options.with_device(device));
    Some([on(Device::Gpu(0)), on(Device::Cpu)])
}

// Every flux the same: no box has a dip at any period, on the GPU as on the
// CPU.
#[test]
fn a_flat_curve_has_no_box() {
    let time: Vec<f64> = (0..2000).map(|i| 0.01 * f64::from(i)).collect();
    let lc = LightCurve::from_days(time, vec![1.0; 2000], vec![1e-3; 2000]).unwrap();
    let bls = Bls::new(&lc).unwrap();
    let Some([gpu, cpu]) = on_both(
        &bls,
        &[1.1, 2.3, 4.7],
        &[0.05, 0.1],
        SearchOptions::default(),
    ) else {
        return;
    };
    let (gpu, cpu) = (gpu.unwrap(), cpu.unwrap());
    assert_eq!(gpu.device(), Device::Gpu(0));
    assert!(gpu.boxes().iter().all(Option::is_none));
    assert_eq!(gpu.power(), cpu.power());
    assert_eq!(gpu.peaks(), cpu.peaks());
}

// Bins too narrow for the GPU's times: every period searched on the CPU, so
// the whole result is the CPU's.
#[test]
fn narrow_bins_are_searched_on_the_cpu() {
    let bls = curve(97, 3000, 0.005);
    let Some([gpu, cpu]) = on_both(&bls, &[1.1, 2.3, 4.7], &[0.0004], SearchOptions::default())
    else {
        return;
    };
    let (gpu, cpu) = (gpu.unwrap(), cpu.unwrap());
    assert_eq!(gpu.device(), Device::Gpu(0));
    assert_eq!(gpu.power(), cpu.power());
    assert_eq!(gpu.boxes(), cpu.boxes());
    assert_eq!(gpu.peaks(), cpu.peaks());
}

// The search's errors come back through the GPU as on the CPU: too many bins,
// found before the GPU's work, and a statistic that overflows, found in it.
#[test]
fn errors_are_the_cpus() {
    let bls = curve(98, 700, 0.02);
    let Some([gpu, cpu]) = on_both(&bls, &[5000.0], &[0.001], SearchOptions::default()) else {
        return;
    };
    assert_eq!(cpu, Err(BlsError::TooManyBins));
    assert_eq!(gpu, cpu);
    let periods = Periods::from_days(&[5000.0]).unwrap();
    let durations = Durations::from_days(&[0.001]).unwrap();
    assert_eq!(
        bls.search(&periods, &durations, &SearchOptions::default()),
        Err(BlsError::TooManyBins)
    );

    // Weights of 1e300 and a dip of 1e5.
    let time: Vec<f64> = (0..40).map(|i| 0.25 * f64::from(i)).collect();
    let flux: Vec<f64> = (0..40)
        .map(|i| if i % 9 == 4 { 0.0 } else { 1e5 })
        .collect();
    let lc = LightCurve::from_days(time, flux, vec![1e-150; 40]).unwrap();
    let bls = Bls::new(&lc).unwrap();
    for options in [
        SearchOptions::default(),
        SearchOptions::default().with_threads(std::num::NonZeroUsize::MIN),
    ] {
        let Some([gpu, cpu]) = on_both(&bls, &[1.5, 2.25, 3.0], &[0.3, 0.5], options) else {
            return;
        };
        assert_eq!(cpu, Err(BlsError::OutOfRange));
        assert_eq!(gpu, cpu);
    }
}

// A light curve whose GPU is set up says so when printed.
#[test]
fn a_bls_with_a_gpu_prints_it() {
    let bls = curve(99, 700, 0.02);
    let Some([gpu, _]) = on_both(&bls, &[2.3], &[0.1], SearchOptions::default()) else {
        return;
    };
    gpu.unwrap();
    let printed = format!("{bls:?}");
    assert!(printed.contains("Gpu"), "{printed}");
}
