//! Against astropy's `BoxLeastSquares`, through the public search: equal to
//! its fast search where astropy is right, at full depth where its
//! wrap-around goes wrong, never below its slow search, and closer to
//! injected transits than its best box.

#[expect(dead_code, reason = "the astropy tests use only some of the recipes")]
#[path = "../src/tests/recipe.rs"]
mod recipe;

use std::num::NonZeroUsize;

use recipe::{Recipe, check_samples};
use rustellar_bls::{BestBox, Bls, Device, Durations, Objective, Periods, SearchOptions};
use rustellar_core::LightCurve;

const FAST_CURVES: &str = include_str!("reference/fast_curves.csv");
const FAST_POWER: &str = include_str!("reference/fast_power.csv");
const WRAP_CASES: &str = include_str!("reference/wrap_cases.csv");
const INJECTED: &str = include_str!("reference/injected.csv");

// ==========================
// Helper Functions
// ==========================

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

fn options(oversample: usize, refine_peaks: usize) -> SearchOptions {
    SearchOptions::default()
        .with_oversample(NonZeroUsize::new(oversample).unwrap())
        .with_objective(Objective::DeltaChi2)
        .with_refine_peaks(refine_peaks)
        .with_refine_reach(2)
        .with_threads(NonZeroUsize::MIN)
        .with_device(Device::Cpu)
}

/// The best binned box at one period.
fn binned(lc: &LightCurve, period: f64, durations: &[f64], oversample: usize) -> BestBox {
    Bls::new(lc)
        .unwrap()
        .search(
            &Periods::from_days(&[period]).unwrap(),
            &Durations::from_days(durations).unwrap(),
            &options(oversample, 0),
        )
        .unwrap()
        .boxes()[0]
        .expect("a box with a dip")
}

/// The box's duration in bins of `bin` days: a whole number of them.
fn in_bins(b: &BestBox, bin: f64) -> usize {
    let bins = b.transit.duration().as_days() / bin;
    assert!((bins - bins.round()).abs() < 1e-9, "{b:?}: not whole bins");
    bins.round() as usize
}

/// The box's start and end, in days of phase from the first point (its
/// mid-transit time is within one period after it).
fn phase_span(lc: &LightCurve, b: &BestBox) -> (f64, f64) {
    let centre = b.transit.transit_time().as_btjd() - lc.time()[0].as_btjd();
    let half = 0.5 * b.transit.duration().as_days();
    (centre - half, centre + half)
}

// The box reported, fitted point by point, holds the same points with the
// same fit, though points may sit exactly on its bins' edges.
fn holds_its_own_points(lc: &LightCurve, ours: &BestBox) {
    let fit = ours.transit.fit(lc).unwrap();
    assert_eq!(fit.n_in, ours.fit.n_in, "{ours:?}");
    assert!(
        (fit.delta_chi2 / ours.fit.delta_chi2 - 1.0).abs() < 1e-12,
        "{fit:?} vs {ours:?}"
    );
}

// How far astropy's plain sums can be from the exact snr, relative: each of
// its sums over `n` points can be off by about n rounding errors of the
// largest |flux|, and the depth is a difference of two such means. Eight
// times that, plus a floor for our own rounding.
fn astropy_tolerance(lc: &LightCurve, depth: f64) -> f64 {
    let largest = lc.flux().iter().fold(0.0_f64, |m, y| m.max(y.abs()));
    1e-12 + 8.0 * lc.len() as f64 * f64::EPSILON * largest / depth.abs()
}

fn fast_curve(fields: &[&str]) -> LightCurve {
    let lc = Recipe {
        seed: fields[1].parse().unwrap(),
        n: 1350,
        t_start: 1325.0,
        cadence: 0.02,
        noise: 1e-3,
        period: fields[2].parse().unwrap(),
        tc: fields[3].parse().unwrap(),
        duration: fields[4].parse().unwrap(),
        depth: fields[5].parse().unwrap(),
    }
    .light_curve();
    check_samples(&lc, fields[6]);
    lc
}

struct WrapCase<'a> {
    name: &'a str,
    lc: LightCurve,
    // The transit's duration.
    duration: f64,
    // astropy's (snr, depth) from the fast and the slow method.
    fast: (f64, f64),
    slow: (f64, f64),
}

fn wrap_cases() -> Vec<WrapCase<'static>> {
    let rows: Vec<Vec<&str>> = rows(WRAP_CASES).collect();
    rows.chunks(2)
        .map(|pair| {
            let (fast, slow) = (&pair[0], &pair[1]);
            assert_eq!((fast[4], slow[4]), ("fast", "slow"));
            let lc = Recipe {
                seed: fast[1].parse().unwrap(),
                n: 15_000,
                t_start: 1325.0,
                cadence: 0.002,
                noise: 1e-4,
                period: 3.0,
                tc: fast[2].parse().unwrap(),
                duration: fast[3].parse().unwrap(),
                depth: 0.01,
            }
            .light_curve();
            check_samples(&lc, fast[9]);
            let numbers = |v: &[&str]| (v[5].parse().unwrap(), v[6].parse().unwrap());
            WrapCase {
                name: fast[0],
                lc,
                duration: fast[3].parse().unwrap(),
                fast: numbers(fast),
                slow: numbers(slow),
            }
        })
        .collect()
}

// Bins of 3/512 d, 512 to the orbit, as astropy's.
const WRAP_DURATIONS: [f64; 3] = [3.0 / 64.0, 3.0 / 32.0, 3.0 / 16.0];
const WRAP_OVERSAMPLE: usize = 8;

// Bins of 1/256 d, and periods of a whole number of them: astropy's bins.
const FAST_DURATIONS: [f64; 4] = [0.031_25, 0.08, 0.12, 0.16];
const FAST_OVERSAMPLE: usize = 8;

// ==========================
// Tests
// ==========================

#[test]
fn recipe_reproduces_the_reference_light_curves() {
    for v in rows(FAST_CURVES) {
        fast_curve(&v);
    }
    assert_eq!(wrap_cases().len(), 4);
}

// 4 light curves (a transit mid-phase, one centred on the first point, one
// just before it, and noise only) at 250 periods from 0.8 to 9 days, each a
// whole number of astropy's bins, so its bins are these bins. Where its best
// box is clean (clear of the last real bin, which its padding overwrites,
// and of the wrap-around, which it pads too short), that box is one this
// search tries too, with the same sums: ours is at least as good, and as
// good to astropy's rounding unless ours is a box astropy cannot try, by
// phase zero or the last bin (to within the 0.02 d between points, as the
// box reported is centred among the times that hold its points).
// Maximising the snr maximises delta_chi2, its square.
#[test]
fn agrees_with_astropy_fast_away_from_phase_zero() {
    let curves: Vec<LightCurve> = rows(FAST_CURVES).map(|v| fast_curve(&v)).collect();
    let (mut equal, mut better) = (0, 0);
    let bin = 1.0 / 256.0;
    for v in rows(FAST_POWER) {
        let lc = &curves[v[0].parse::<usize>().unwrap()];
        let period: f64 = v[1].parse().unwrap();
        let ours = binned(lc, period, &FAST_DURATIONS, FAST_OVERSAMPLE);
        holds_its_own_points(lc, &ours);
        let (snr, depth): (f64, f64) = (v[2].parse().unwrap(), v[3].parse().unwrap());
        let m: usize = v[8].parse().unwrap();
        // astropy's bins are ours: 1/256 d, m to the period.
        assert_eq!(period / bin, m as f64);
        in_bins(&ours, bin);
        let tolerance = astropy_tolerance(lc, depth);

        if v[9] == "1" {
            assert!(
                ours.fit.snr >= snr * (1.0 - tolerance),
                "P {period}: {} below astropy's {snr}",
                ours.fit.snr
            );
            if (ours.fit.snr / snr - 1.0).abs() < tolerance {
                equal += 1;
            } else {
                let (start, end) = phase_span(lc, &ours);
                let slack = 0.02 + bin;
                assert!(
                    start < slack || end > (m - 1) as f64 * bin - slack,
                    "P {period}: {} above {snr} away from phase zero and the last bin",
                    ours.fit.snr
                );
                better += 1;
            }
        }
    }
    // Most periods are compared exactly (947 of 1,000); at some, the best box
    // straddles phase zero, where astropy cannot put it (20 better).
    assert!(equal > 900, "{equal} equal");
    assert!(better >= 15, "{better} better");
}

// astropy's recorded failures, so a test fails if a future astropy fixes
// them (then the docs need updating).
#[test]
fn astropy_fast_fails_at_phase_zero() {
    for case in wrap_cases() {
        let (fast, slow) = (case.fast, case.slow);
        match case.name {
            "mid_phase" => assert!((fast.0 / slow.0 - 1.0).abs() < 1e-12, "{fast:?} {slow:?}"),
            // The last bin's points never count as in transit: no box has a
            // dip, and astropy leaves the depth, duration and time as whatever
            // was in memory (here a NaN depth).
            "last_bin" => assert_eq!(fast.0, f64::NEG_INFINITY),
            // The box cannot straddle phase zero: the depth is 8% low.
            "phase_zero_short" => assert!(fast.1 < 0.93 * slow.1, "{fast:?} {slow:?}"),
            // Longer boxes fare worse: 76% of the snr, 77% of the depth.
            "phase_zero_long" => assert!(fast.0 < 0.77 * slow.0, "{fast:?} {slow:?}"),
            name => panic!("unexpected case {name}"),
        }
    }
}

// astropy's slow search tries a box every D / oversample from the first point,
// which for these durations (8, 16 and 32 bins) is every 1, 2 or 4 bins, with
// the box's edges on bin edges: every box it tries, this search tries too.
// So this search is never below it, and it finds every transit, at full depth.
#[test]
fn full_depth_where_astropy_fast_is_wrong() {
    for case in wrap_cases() {
        let ours = binned(&case.lc, 3.0, &WRAP_DURATIONS, WRAP_OVERSAMPLE);
        in_bins(&ours, 3.0 / 512.0);
        let tolerance = astropy_tolerance(&case.lc, case.slow.1);
        assert!(
            ours.fit.snr >= case.slow.0 * (1.0 - tolerance),
            "{}: {} vs slow {}",
            case.name,
            ours.fit.snr,
            case.slow.0
        );
        assert!(
            ours.fit.snr >= case.fast.0 * (1.0 - tolerance),
            "{}",
            case.name
        );
        if case.name != "last_bin" {
            // A box centred on the transit, which holds a part `overlap / box`
            // of it, and leaves the rest among the other points, measures a
            // depth of 0.01 (overlap / box - rest / (P - box)): 0.00996 for
            // phase_zero_long, whose box is 0.0125 d short. The depth's error
            // is 3e-6 to 6e-6; the seeds are fixed, so this never flakes.
            let length = ours.transit.duration().as_days();
            let overlap = case.duration.min(length);
            let expected = 0.01 * (overlap / length - (case.duration - overlap) / (3.0 - length));
            assert!(
                (ours.fit.depth - expected).abs() < 1.5e-5,
                "{}: {} vs {expected}",
                case.name,
                ours.fit.depth
            );
        }
    }
}

// ==========================
// Injected transits
// ==========================

// How far a reported box is from the injected one: the mid-transit time at
// the transit nearest the middle of the data (where a slightly wrong period
// moves it least), and the depth; ours, then astropy's.
struct Errors {
    time: [f64; 2],
    depth: [f64; 2],
}

// `t`'s distance from the nearest transit of period `p` through `tc`.
fn offset_from_transit(t: f64, tc: f64, p: f64) -> f64 {
    let x = t - tc;
    x - (x / p).round() * p
}

fn injected() -> Vec<Errors> {
    let durations = Durations::from_days(&[0.04, 0.06, 0.08, 0.11, 0.15, 0.2]).unwrap();
    let middle = 1325.0 + 13.5;
    let at_middle = |t0: f64, p: f64| t0 + ((middle - t0) / p).round() * p;
    rows(INJECTED)
        .map(|v| {
            let f = |i: usize| v[i].parse::<f64>().unwrap();
            let (period, tc, depth) = (f(2), f(3), f(5));
            let lc = Recipe {
                seed: v[1].parse().unwrap(),
                n: 2000,
                t_start: 1325.0,
                cadence: 0.0135,
                noise: 1e-3,
                period,
                tc,
                duration: f(4),
                depth,
            }
            .light_curve();
            check_samples(&lc, v[11]);
            // The reference data's periods, operation for operation.
            let periods: Vec<f64> = (0..201)
                .map(|k| period * (1.0 + 0.02 * f64::from(k - 100) / 100.0))
                .collect();
            // The binned power's highest peak, refined two periods either
            // side.
            let result = Bls::new(&lc)
                .unwrap()
                .search(
                    &Periods::from_days(&periods).unwrap(),
                    &durations,
                    &options(10, 1),
                )
                .unwrap();
            let ours = result.best().expect("a box with a dip");
            let ours_time = at_middle(
                ours.transit.transit_time().as_btjd(),
                ours.transit.period().as_days(),
            );
            let theirs_time = at_middle(f(7), f(6));
            Errors {
                time: [ours_time, theirs_time].map(|t| offset_from_transit(t, tc, period).abs()),
                depth: [ours.fit.depth, f(8)].map(|d| (d - depth).abs()),
            }
        })
        .collect()
}

fn median(mut x: Vec<f64>) -> f64 {
    x.sort_by(f64::total_cmp);
    0.5 * (x[(x.len() - 1) / 2] + x[x.len() / 2])
}

fn largest(x: &[f64]) -> f64 {
    x.iter().copied().fold(0.0, f64::max)
}

// 144 box transits, each searched at 201 periods within 2% of its own, the
// binned power's highest peak refined two periods either side, against
// astropy's best box on the same light curves and periods. Three groups of
// 48: snr 8 to 20 at random phases, snr 50 to 150 at random phases, and snr
// 8 to 20 within 0.03 d of the first point's phase.
//
// Measured (median, and largest, error; days for times):
//
// | group      | time, ours        | time, astropy     | depth, ours | depth, astropy |
// |------------|-------------------|-------------------|-------------|----------------|
// | random     | 1.05e-3 (5.8e-3)  | 2.13e-3 (1.27e-2) | 5.0e-5      | 5.8e-5         |
// | bright     | 7.1e-4 (4.2e-3)   | 2.77e-3 (9.6e-3)  | 7.7e-5      | 2.40e-4        |
// | phase zero | 6.7e-4 (3.8e-3)   | 3.24e-3 (2.06e-2) | 6.6e-5      | 1.08e-4        |
//
// At snr 8 to 20 the noise sets much of the error; when it does not, the
// grid of durations does (a box shorter than the transit can sit anywhere in
// it), and astropy's phase bins, and near phase zero its wrap-around.
#[test]
fn closer_to_injected_transits_than_astropy() {
    let errors = injected();
    assert_eq!(errors.len(), 144);
    for (name, group) in [
        ("random", &errors[..48]),
        ("bright", &errors[48..96]),
        ("phase zero", &errors[96..]),
    ] {
        let time = |k: usize| group.iter().map(|e| e.time[k]).collect::<Vec<f64>>();
        let depth = |k: usize| group.iter().map(|e| e.depth[k]).collect::<Vec<f64>>();
        let (ours, theirs) = (median(time(0)), median(time(1)));
        assert!(ours < 0.6 * theirs, "{name}: time {ours} vs {theirs}");
        assert!(
            largest(&time(0)) < largest(&time(1)),
            "{name}: largest time"
        );
        let (ours, theirs) = (median(depth(0)), median(depth(1)));
        assert!(ours < theirs, "{name}: depth {ours} vs {theirs}");
        if name == "bright" {
            assert!(ours < 0.5 * theirs, "{name}: depth {ours} vs {theirs}");
        }
    }
}
