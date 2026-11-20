// The search through the public API: worked example 3 against astropy's slow
// search and the exact search, the same results on any number of threads,
// what each objective reports, and the errors.

use std::num::NonZeroUsize;

use rstest::rstest;
use rustellar_bls::{
    BestBox, Bls, BlsError, BlsResult, Durations, Objective, Periods, SearchOptions, TransitBox,
};
use rustellar_core::{LightCurve, TimeDelta};

// ==========================
// Helper Functions
// ==========================

const EXAMPLE_3: &str = include_str!("reference/example_3_points.csv");
const SLOW_POWER: &str = include_str!("reference/slow_power.csv");
const DURATIONS: [f64; 3] = [0.06, 0.12, 0.2];

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

fn example_3() -> LightCurve {
    let (mut time, mut flux, mut flux_err) = (vec![], vec![], vec![]);
    for v in rows(EXAMPLE_3) {
        time.push(v[0].parse().unwrap());
        flux.push(v[1].parse().unwrap());
        flux_err.push(v[2].parse().unwrap());
    }
    LightCurve::from_btjd(time, flux, flux_err).unwrap()
}

// astropy's slow search: each period and its snr.
fn slow_power() -> Vec<(f64, f64)> {
    rows(SLOW_POWER)
        .map(|v| (v[0].parse().unwrap(), v[1].parse().unwrap()))
        .collect()
}

fn periods(days: &[f64]) -> Periods {
    Periods::from_days(days).unwrap()
}

fn durations(days: &[f64]) -> Durations {
    Durations::from_days(days).unwrap()
}

fn threads(n: usize) -> SearchOptions {
    SearchOptions::default().with_threads(NonZeroUsize::new(n).unwrap())
}

fn search_example_3(options: &SearchOptions) -> BlsResult {
    let days: Vec<f64> = slow_power().iter().map(|s| s.0).collect();
    Bls::new(&example_3())
        .unwrap()
        .search(&periods(&days), &durations(&DURATIONS), options)
        .unwrap()
}

fn days(value: f64) -> TimeDelta {
    TimeDelta::from_days(value).unwrap()
}

// ==========================
// Worked example 3
// ==========================

// astropy's slow search at 300 periods from 1 to 6 days: the refined best box
// is at the transit's 3-day period and at least as good as the best astropy
// finds anywhere.
#[test]
fn worked_example_3() {
    let result = search_example_3(&threads(1).with_objective(Objective::Snr));
    let theirs = slow_power()
        .iter()
        .map(|s| s.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let best = result.best().unwrap();
    assert!(
        (best.transit.period().as_days() - 3.0).abs() < 0.02,
        "{best:?}"
    );
    assert!(
        best.fit.snr >= theirs * (1.0 - 1e-12),
        "{best:?} vs {theirs}"
    );
    assert_eq!(result.peaks()[0].best, *best);
}

// Each refined peak is at least as good as the exact search at each period
// within its reach and each duration on the grid.
#[test]
fn peaks_beat_the_exact_search_on_the_grid() {
    let lc = example_3();
    let result = search_example_3(&threads(1));
    let all: Vec<f64> = slow_power().iter().map(|s| s.0).collect();
    assert_eq!(result.peaks().len(), 5);
    for peak in result.peaks() {
        let i = peak.peak_index;
        assert!(peak.period_index.abs_diff(i) <= 2);
        for &p in &all[i.saturating_sub(2)..(i + 3).min(all.len())] {
            for d in DURATIONS {
                if let Some(exact) = TransitBox::best(&lc, days(p), days(d)).unwrap() {
                    assert!(
                        peak.best.fit.delta_chi2 >= exact.fit.delta_chi2 * (1.0 - 1e-13),
                        "peak {i}, P {p}, D {d}"
                    );
                }
            }
        }
        // The box refits to itself.
        let refit = peak.best.transit.fit(&lc).unwrap();
        assert_eq!(refit.n_in, peak.best.fit.n_in);
        assert!((refit.delta_chi2 / peak.best.fit.delta_chi2 - 1.0).abs() < 1e-12);
    }
}

// The peaks come highest binned power first.
#[test]
fn peaks_in_order_of_power() {
    let result = search_example_3(&threads(1));
    let power = result.power();
    for pair in result.peaks().windows(2) {
        assert!(power[pair[0].peak_index] >= power[pair[1].peak_index]);
    }
}

// ==========================
// The binned boxes
// ==========================

// Each box's power is its fit's objective, its duration is a searched one
// rounded to a whole number of bins, it starts within one period after the
// first time, and it fits to itself.
#[rstest]
fn boxes_match_the_power(
    #[values(Objective::DeltaChi2, Objective::Snr, Objective::AstropyLikelihood)]
    objective: Objective,
) {
    let lc = example_3();
    let first = lc.time()[0];
    let result = search_example_3(&threads(1).with_objective(objective));
    assert_eq!(result.boxes().len(), 300);
    for (k, (&power, found)) in result.power().iter().zip(result.boxes()).enumerate() {
        let found = found.unwrap();
        assert_eq!(power, objective.of(&found.fit), "period {k}");
        let p = found.transit.period().as_days();
        let bin = p / (p / (0.06 / 10.0)).ceil();
        let d = found.transit.duration().as_days();
        assert!(
            DURATIONS.iter().any(|&x| (d - x).abs() <= 0.5 * bin),
            "period {k}: {d}"
        );
        let after = (found.transit.transit_time() - first).as_days();
        assert!((0.0..=p).contains(&after), "period {k}: {after}");
        let refit = found.transit.fit(&lc).unwrap();
        assert_eq!(refit.n_in, found.fit.n_in, "period {k}");
        assert!((refit.delta_chi2 / found.fit.delta_chi2 - 1.0).abs() < 1e-10);
    }
}

// delta_chi2 and snr rank boxes the same way, so they pick the same boxes.
#[test]
fn snr_picks_the_same_boxes_as_delta_chi2() {
    let delta_chi2 = search_example_3(&threads(1));
    let snr = search_example_3(&threads(1).with_objective(Objective::Snr));
    assert_eq!(delta_chi2.boxes(), snr.boxes());
}

// An unweighted search, as astropy's with no errors: one error for every
// point. Its value only scales the statistics (here by a power of two, so
// exactly): the boxes and depths are the same, and each depth is the plain
// mean out of the box less the plain mean in it.
#[test]
fn one_error_for_every_point_is_unweighted() {
    let lc = example_3();
    let with_error = |sigma: f64| {
        let time = lc.time().iter().map(|t| t.as_btjd()).collect();
        LightCurve::from_btjd(time, lc.flux().to_vec(), vec![sigma; lc.len()]).unwrap()
    };
    let search = |lc: &LightCurve| {
        let days: Vec<f64> = slow_power().iter().map(|s| s.0).collect();
        Bls::new(lc)
            .unwrap()
            .search(&periods(&days), &durations(&DURATIONS), &threads(1))
            .unwrap()
    };
    let (unit, small) = (with_error(1.0), with_error(1.0 / 512.0));
    let (a, b) = (search(&unit), search(&small));
    for (x, y) in a.boxes().iter().zip(b.boxes()) {
        let (x, y) = (x.unwrap(), y.unwrap());
        assert_eq!(x.transit, y.transit);
        assert_eq!(x.fit.depth, y.fit.depth);
        assert_eq!(x.fit.delta_chi2 * 262_144.0, y.fit.delta_chi2);
        assert_eq!(x.fit.depth_err / 512.0, y.fit.depth_err);
    }
    let best = a.best().unwrap();
    let (mut inside, mut outside) = (vec![], vec![]);
    for (t, f) in unit.time().iter().zip(unit.flux()) {
        if best.transit.contains(*t) {
            inside.push(*f);
        } else {
            outside.push(*f);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let plain = mean(&outside) - mean(&inside);
    assert!(
        (best.fit.depth - plain).abs() < 1e-15,
        "{} {plain}",
        best.fit.depth
    );
}

// ==========================
// Threads and memory
// ==========================

// Contiguous chunks of periods, one per thread, give the same results to the
// bit.
#[rstest]
fn same_on_any_number_of_threads(#[values(2, 3, 8, 301)] n: usize) {
    let one = search_example_3(&threads(1));
    let many = search_example_3(&threads(n));
    let bits = |r: &BlsResult| r.power().iter().map(|p| p.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&one), bits(&many));
    assert_eq!(one, many);
}

// A result reused for searches of other sizes gives what a fresh one does.
#[test]
fn search_into_reuses_a_result() {
    let lc = example_3();
    let bls = Bls::new(&lc).unwrap();
    let options = threads(2);
    let mut reused = BlsResult::default();
    for count in [300, 7, 120, 1] {
        let days: Vec<f64> = (0..count)
            .map(|k| 1.0 + 5.0 * f64::from(k) / 300.0)
            .collect();
        let (p, d) = (periods(&days), durations(&DURATIONS));
        bls.search_into(&p, &d, &options, &mut reused).unwrap();
        assert_eq!(reused, bls.search(&p, &d, &options).unwrap(), "{count}");
    }
}

// ==========================
// Options
// ==========================

#[test]
fn no_refinement() {
    let result = search_example_3(&threads(1).with_refine_peaks(0));
    assert!(result.peaks().is_empty());
    // The best is then the binned box at the highest power.
    let power = result.power();
    let top = (0..power.len())
        .reduce(|a, b| if power[b] > power[a] { b } else { a })
        .unwrap();
    assert_eq!(result.best(), result.boxes()[top].as_ref());
}

#[test]
fn refine_one_period_only() {
    let result = search_example_3(&threads(1).with_refine_reach(0).with_refine_peaks(2));
    assert_eq!(result.peaks().len(), 2);
    for peak in result.peaks() {
        assert_eq!(peak.period_index, peak.peak_index);
    }
}

#[test]
fn default_options() {
    let options = SearchOptions::default();
    assert_eq!(options.oversample.get(), 10);
    assert_eq!(options.objective, Objective::DeltaChi2);
    assert_eq!((options.refine_peaks, options.refine_reach), (5, 2));
    assert_eq!(
        options.threads,
        std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
    );
}

// ==========================
// Edge cases and errors
// ==========================

#[test]
fn flat_flux_has_no_dip() {
    let time: Vec<f64> = (0..300).map(|i| 0.05 * f64::from(i)).collect();
    let lc = LightCurve::from_btjd(time, vec![1.0; 300], vec![1e-3; 300]).unwrap();
    let result = Bls::new(&lc)
        .unwrap()
        .search(
            &periods(&[1.0, 2.0, 3.0]),
            &durations(&[0.1]),
            &SearchOptions::default(),
        )
        .unwrap();
    assert_eq!(result.power(), [0.0; 3]);
    assert_eq!(result.boxes(), [None::<BestBox>; 3]);
    assert!(result.peaks().is_empty());
    assert_eq!(result.best(), None);
}

#[test]
fn one_point() {
    let lc = LightCurve::from_btjd(vec![1325.0], vec![1.0], vec![1e-3]).unwrap();
    let result = Bls::new(&lc)
        .unwrap()
        .search(
            &periods(&[1.0]),
            &durations(&[0.1]),
            &SearchOptions::default(),
        )
        .unwrap();
    assert_eq!(result.best(), None);
}

#[test]
fn duration_not_shorter_than_period() {
    let bls = Bls::new(&example_3()).unwrap();
    assert_eq!(
        bls.search(
            &periods(&[1.0, 3.0]),
            &durations(&[0.1, 1.0]),
            &SearchOptions::default()
        ),
        Err(BlsError::DurationNotShorterThanPeriod)
    );
}

// A million bins is the limit: 122.0703125 days in bins of 2^-13 days, each
// value exact.
#[rstest]
#[case(122.070_312_5, Ok(()))]
#[case(122.070_4, Err(BlsError::TooManyBins))]
#[case(1e300, Err(BlsError::TooManyBins))]
fn too_many_bins(#[case] longest: f64, #[case] expected: Result<(), BlsError>) {
    let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![1.0, 0.9], vec![1e-3; 2]).unwrap();
    let options = SearchOptions::default()
        .with_oversample(NonZeroUsize::new(8).unwrap())
        .with_threads(NonZeroUsize::MIN);
    let outcome = Bls::new(&lc)
        .unwrap()
        .search(
            &periods(&[1.0, longest]),
            &durations(&[0.000_976_562_5]),
            &options,
        )
        .map(|_| ());
    assert_eq!(outcome, expected);
}

#[test]
fn weights_out_of_range() {
    let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![1.0, 1.0], vec![1e-200, 1e-3]).unwrap();
    assert_eq!(Bls::new(&lc).unwrap_err(), BlsError::OutOfRange);
}

// Weights of 1e300 and a dip of 1e5: the statistic overflows on whichever
// thread meets it, and the search stops.
#[rstest]
fn statistics_that_overflow(#[values(1, 3)] n: usize) {
    let lc =
        LightCurve::from_btjd(vec![0.0, 1.0, 2.0], vec![1e5, 0.0, 1e5], vec![1e-150; 3]).unwrap();
    assert_eq!(
        Bls::new(&lc).unwrap().search(
            &periods(&[5.0, 10.0, 20.0]),
            &durations(&[0.5]),
            &threads(n)
        ),
        Err(BlsError::OutOfRange)
    );
}

// Only the refinement's second step tries the box that holds the point of
// weight 1e300 alone in its run from 0.12 to 0.32 days: it needs a duration
// above 0.2 days and no more than the 0.26 days between that run's
// neighbours. The binned search at 0.2 and 0.3 days fits, and refinement
// stops the search.
#[rstest]
fn refinement_that_overflows(#[values(1, 3)] n: usize) {
    let lc = LightCurve::from_btjd(
        vec![0.09, 0.12, 0.14, 0.32, 0.35, 0.58, 0.73],
        vec![1e5, 0.0, 0.0, -1e5, 1e5, -1e5, 1e5],
        vec![1.0, 1e-150, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let bls = Bls::new(&lc).unwrap();
    let (p, d) = (periods(&[1.0]), durations(&[0.2, 0.3]));
    assert!(bls.search(&p, &d, &threads(n).with_refine_peaks(0)).is_ok());
    assert_eq!(bls.search(&p, &d, &threads(n)), Err(BlsError::OutOfRange));
}

// Without the `cuda` feature: the default searches on the CPU, and asking
// for a GPU is an error.
#[cfg(not(feature = "cuda"))]
#[test]
fn without_the_cuda_feature_the_search_is_on_the_cpu() {
    use rustellar_bls::Device;

    let time: Vec<f64> = (0..400).map(|i| 0.05 * f64::from(i)).collect();
    let lc = LightCurve::from_btjd(time, vec![1.0; 400], vec![1e-3; 400]).unwrap();
    let bls = Bls::new(&lc).unwrap();
    let periods = Periods::from_days(&[1.5, 2.0, 2.5]).unwrap();
    let durations = Durations::from_days(&[0.1]).unwrap();
    let options = SearchOptions::default();
    let auto = bls.search(&periods, &durations, &options).unwrap();
    assert_eq!(auto.device(), Device::Cpu);
    assert_eq!(
        bls.search(&periods, &durations, &options.with_device(Device::Gpu(0))),
        Err(BlsError::GpuUnavailable)
    );
}
