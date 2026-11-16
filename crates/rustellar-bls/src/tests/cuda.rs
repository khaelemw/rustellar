// The folds made and searched on the GPU against the CPU's: every fold taken
// by the GPU, and each best box the same to the bit. Where there is no GPU
// (or no CUDA driver), each test prints why and passes.

use super::recipe::Survey;
use crate::{
    Bls, Objective,
    cuda::{CudaError, Gpu},
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
            println!("skipped: no GPU ({why})");
            None
        }
        Err(error) => panic!("{error:?}"),
    }
}

// Periods round the transit's and its aliases, and across the rest.
fn periods() -> Vec<f64> {
    let mut periods: Vec<f64> = (0..30).map(|k| 0.7 + 0.31 * f64::from(k)).collect();
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
        }
    }
}
