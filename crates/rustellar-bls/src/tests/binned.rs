// The binned search: equal to fitting each box point by point where every
// point sits on a bin edge, equal to astropy's fast search where astropy is
// right, at full depth where astropy's wrap-around goes wrong, never below
// astropy's slow search, and never above the exact search.

use rustellar_core::{Epoch, LightCurve, TimeDelta};

use super::recipe::{Random, Recipe, check_samples};
use crate::{
    BlsError, TransitBox,
    binned::{BinnedBox, best_binned},
    box_fit::Weighted,
};

// ==========================
// Helper Functions
// ==========================

const FAST_CURVES: &str = include_str!("../../tests/reference/fast_curves.csv");
const FAST_POWER: &str = include_str!("../../tests/reference/fast_power.csv");
const WRAP_CASES: &str = include_str!("../../tests/reference/wrap_cases.csv");

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

fn binned(lc: &LightCurve, period: f64, durations: &[f64], oversample: usize) -> Option<BinnedBox> {
    let times: Vec<f64> = lc.time().iter().map(|t| t.as_btjd()).collect();
    best_binned(
        &times,
        &Weighted::new(lc).unwrap(),
        period,
        durations,
        oversample,
    )
    .unwrap()
}

fn exact(lc: &LightCurve, period: f64, duration: f64) -> Option<f64> {
    TransitBox::best(
        lc,
        TimeDelta::from_days(period).unwrap(),
        TimeDelta::from_days(duration).unwrap(),
    )
    .unwrap()
    .map(|b| b.fit.delta_chi2)
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
// Point by point, where points sit on bin edges
// ==========================

// Times on a grid of 1/64 day, the bin width, and a period of 160 bins: every
// phase is a whole number of bins, exactly. A box of k bins from bin n then
// holds exactly the points a box of duration k/64 centred at bin n + (k-1)/2
// holds, so the binned search must find the best of those boxes, fitted
// point by point.
#[test]
fn equals_fitting_each_box_point_by_point() {
    let bin = 1.0 / 64.0;
    let period = 160.0 * bin;
    let durations = [4.0 * bin, 8.0 * bin, 16.0 * bin];
    let mut rng = Random(5);
    for _ in 0..40 {
        // About 300 of the grid times over 20 days, with a dip near a random
        // phase.
        let centre = rng.unit() * period;
        let (mut time, mut flux, mut err) = (vec![], vec![], vec![]);
        for j in 0..1280 {
            if rng.unit() < 0.25 {
                let t = 1325.0 + f64::from(j) * bin;
                let phase = (f64::from(j) * bin) % period;
                let dip = (phase - centre).abs() < 0.08;
                time.push(t);
                flux.push(1e-3 * (rng.unit() - 0.5) - if dip { 2e-3 } else { 0.0 });
                err.push(1e-3 * (0.5 + rng.unit()));
            }
        }
        let lc = LightCurve::from_btjd(time, flux, err).unwrap();
        let ours = binned(&lc, period, &durations, 4).unwrap();

        let first = lc.time()[0].as_btjd();
        let mut best = 0.0_f64;
        for &d in &durations {
            let k = (d / bin).round();
            for n in 0..160 {
                let centre = first + (f64::from(n) + 0.5 * (k - 1.0)) * bin;
                let transit = TransitBox::new(
                    TimeDelta::from_days(period).unwrap(),
                    TimeDelta::from_days(d).unwrap(),
                    Epoch::from_btjd(centre).unwrap(),
                )
                .unwrap();
                if let Ok(fit) = transit.fit(&lc) {
                    if fit.depth > 0.0 {
                        best = best.max(fit.delta_chi2);
                    }
                }
            }
        }
        assert!(
            (ours.fit.delta_chi2 / best - 1.0).abs() < 1e-13,
            "{} vs {best}",
            ours.fit.delta_chi2
        );
    }
}

// Four bins of a quarter of a day: a box as long as the period would hold
// every point, so only the one-bin boxes are tried.
#[test]
fn a_box_of_every_bin_is_skipped() {
    let time: Vec<f64> = (0..40).map(|i| 0.1 * f64::from(i)).collect();
    let flux = time
        .iter()
        .map(|t| if t % 1.0 < 0.2 { 0.99 } else { 1.0 })
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 40]).unwrap();
    let both = binned(&lc, 1.0, &[0.25, 1.0], 1).unwrap();
    assert_eq!(both, binned(&lc, 1.0, &[0.25], 1).unwrap());
    assert_eq!((both.start, both.bins, both.duration), (0, 1, 0));
}

// Weights of 1e300 and a dip of 1e5: the likelihood is about 1e310.
#[test]
fn statistics_that_overflow() {
    let lc =
        LightCurve::from_btjd(vec![0.0, 1.0, 2.0], vec![1e5, 0.0, 1e5], vec![1e-150; 3]).unwrap();
    let times = [0.0, 1.0, 2.0];
    assert_eq!(
        best_binned(&times, &Weighted::new(&lc).unwrap(), 10.0, &[0.5], 1),
        Err(BlsError::OutOfRange)
    );
}

// ==========================
// The shared recipe
// ==========================

#[test]
fn recipe_reproduces_the_generators_light_curves() {
    for v in rows(FAST_CURVES) {
        fast_curve(&v);
    }
    assert_eq!(wrap_cases().len(), 4);
}

// ==========================
// Against astropy's fast search
// ==========================

// 4 light curves (a transit mid-phase, one centred on the first point, one
// just before it, and noise only) at 250 periods from 0.8 to 9 days, each a
// whole number of astropy's bins, so its bins are these bins. Where its best
// box is clean (clear of the last real bin, which its padding overwrites,
// and of the wrap-around, which it pads too short), that box is one this
// search tries too, with the same sums: ours is at least as good. Where ours
// is clean as well, each search tried the other's box, so the two are equally
// good, to astropy's rounding. Maximising the snr maximises delta_chi2, its
// square.
#[test]
fn agrees_with_astropy_fast_away_from_phase_zero() {
    let curves: Vec<LightCurve> = rows(FAST_CURVES).map(|v| fast_curve(&v)).collect();
    let (mut equal, mut better) = (0, 0);
    for v in rows(FAST_POWER) {
        let lc = &curves[v[0].parse::<usize>().unwrap()];
        let period: f64 = v[1].parse().unwrap();
        let ours = binned(lc, period, &FAST_DURATIONS, FAST_OVERSAMPLE).unwrap();
        let (snr, depth): (f64, f64) = (v[2].parse().unwrap(), v[3].parse().unwrap());
        let m: usize = v[8].parse().unwrap();
        assert_eq!(ours.bin, period / m as f64);
        assert_eq!(ours.bin, 1.0 / 256.0);
        let tolerance = astropy_tolerance(lc, depth);

        if v[9] == "1" {
            assert!(
                ours.fit.snr >= snr * (1.0 - tolerance),
                "P {period}: {} below astropy's {snr}",
                ours.fit.snr
            );
            // Clear of the last bin, as astropy's clean boxes are.
            if ours.start + ours.bins < m {
                assert!(
                    (ours.fit.snr / snr - 1.0).abs() < tolerance,
                    "P {period}: {} vs {snr}",
                    ours.fit.snr
                );
                equal += 1;
            } else if ours.fit.snr > snr * (1.0 + tolerance) {
                better += 1;
            }
        }
    }
    // Most periods are compared exactly (947 of 1,000); at some, the best box
    // straddles phase zero, where astropy cannot put it (20 better).
    assert!(equal > 900, "{equal} equal");
    assert!(better >= 15, "{better} better");
}

// ==========================
// Where astropy's wrap-around goes wrong
// ==========================

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
        let ours = binned(&case.lc, 3.0, &WRAP_DURATIONS, WRAP_OVERSAMPLE).unwrap();
        assert_eq!(ours.bin, 3.0 / 512.0);
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
            let length = ours.bins as f64 * ours.bin;
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
// Bounded by the exact search
// ==========================

// The bins tile the orbit, so every box of k bins is k b long wherever it
// starts, and holds what a box of duration k b holds (slide it left by less
// than the gap before its first point): the binned search is never above the
// exact search at duration k b. 10,000 cases: 25 light curves, 100 periods,
// 4 durations each.
#[test]
fn never_above_the_exact_search() {
    let mut rng = Random(9);
    let mut cases = 0;
    for curve in 0..25 {
        let period = 0.8 + 3.0 * rng.unit();
        let lc = Recipe {
            seed: 1000 + curve,
            n: 300,
            t_start: 1325.0,
            cadence: 0.05,
            noise: 1e-3,
            period,
            tc: 1325.0 + period * rng.unit(),
            duration: 0.05 + 0.1 * rng.unit(),
            depth: 2e-3 * rng.unit(),
        }
        .light_curve();
        for _ in 0..100 {
            let trial = 0.5 + 4.0 * rng.unit();
            for d in [0.04, 0.07, 0.1, 0.15] {
                cases += 1;
                // One duration at a time: bins of at most d / 5, and boxes of
                // 5 bins, since d / b < 5 + d / P rounds to 5.
                let Some(ours) = binned(&lc, trial, &[d], 5) else {
                    continue;
                };
                assert_eq!(ours.bins, 5);
                let exact = exact(&lc, trial, 5.0 * ours.bin)
                    .expect("the exact search finds every set the binned one does");
                assert!(
                    ours.fit.delta_chi2 <= exact * (1.0 + 1e-12),
                    "P {trial}, D {d}: {} above {exact}",
                    ours.fit.delta_chi2
                );
            }
        }
    }
    assert_eq!(cases, 10_000);
}

// How far below the exact search: a noiseless transit of depth 0.01 and
// duration T, sampled every 1/1024 day with equal errors, searched with a box
// of length L = k b. Put a box of length L where it overlaps the transit by
// `o`: its depth is 0.01 (o / L - (T - o) / (P - L)), and its snr is that
// times a factor that depends on L alone. The exact search overlaps by
// min(T, L). The binned box's start is on a grid of b, so it can come within
// (b - |T - L|) / 2 of an edge where it overlaps fully, if that is positive:
// it overlaps by at least min(T, L) - max(0, b - |T - L|) / 2. Counting
// points rather than lengths moves each count by at most a point at each end
// in each transit, so the lengths are given two cadences of slack. For
// T = L that is an snr of about 1 - 1 / (2 k) of the exact search's: 95% at
// k = 10.
#[test]
fn below_the_exact_search_by_at_most_half_a_bin() {
    let cadence = 1.0 / 1024.0;
    let time: Vec<f64> = (0..10_240)
        .map(|i| 1325.0 + cadence * f64::from(i))
        .collect();
    let mut rng = Random(79);
    for oversample in [2, 3, 5, 10] {
        for _ in 0..30 {
            let period = 1.0 + 3.0 * rng.unit();
            let transit = 0.05 + 0.1 * rng.unit();
            let tc = 1325.0 + period * rng.unit();
            let flux = time
                .iter()
                .map(|t| {
                    let r = (t - tc) / period;
                    let f = r - r.floor();
                    if f.min(1.0 - f) * period < 0.5 * transit {
                        -1e-2
                    } else {
                        0.0
                    }
                })
                .collect();
            let lc = LightCurve::from_btjd(time.clone(), flux, vec![1e-3; time.len()]).unwrap();
            let duration = transit * (0.6 + 0.8 * rng.unit());
            let ours = binned(&lc, period, &[duration], oversample).unwrap();
            let length = ours.bins as f64 * ours.bin;
            let exact = exact(&lc, period, length).unwrap();

            let depth = |o: f64| o / length - (transit - o) / (period - length);
            let full = transit.min(length);
            let short = full - (ours.bin - (transit - length).abs()).max(0.0) / 2.0;
            let bound = depth(short - 2.0 * cadence) / depth(full + 2.0 * cadence);
            let ratio = (ours.fit.delta_chi2 / exact).sqrt();
            assert!(
                ratio >= bound && ratio <= 1.0 + 1e-12,
                "oversample {oversample}, P {period}: {ratio} vs {bound}"
            );
        }
    }
}

#[test]
fn flat_flux_has_no_dip() {
    let time: Vec<f64> = (0..50).map(|i| 0.1 * f64::from(i)).collect();
    let lc = LightCurve::from_btjd(time, vec![1.0; 50], vec![1e-3; 50]).unwrap();
    assert_eq!(binned(&lc, 1.3, &[0.2], 4), None);
}

// Most of the weight in one short stretch of phase, with the dip there: the
// box holds more than half the weight, so the weight out of it is taken from
// the running totals rather than from the orbit's weight less the box's. The
// fit is the one-box fit of the same points, and the exact search finds the
// same box.
#[test]
fn a_box_with_most_of_the_weight() {
    let mut time = vec![];
    let mut flux = vec![];
    let mut err = vec![];
    for i in 0..40 {
        // Ten points an orbit at phases 0.40 to 0.49 with small errors, and
        // two at 0.1 and 0.8 with large ones.
        let orbit = f64::from(i / 12);
        let k = i % 12;
        let (phase, f, e) = match k {
            0 => (0.1, 1.0, 1e-2),
            11 => (0.8, 1.0, 1e-2),
            _ => (0.39 + 0.01 * f64::from(k), 0.99, 1e-3),
        };
        time.push(1325.0 + orbit + phase);
        flux.push(f);
        err.push(e);
    }
    let lc = LightCurve::from_btjd(time, flux, err).unwrap();
    let ours = binned(&lc, 1.0, &[0.25], 5).unwrap();
    let total: f64 = lc.flux_err().iter().map(|e| 1.0 / (e * e)).sum();
    assert!(ours.fit.weight_in > 0.5 * total, "{ours:?}");
    assert!((ours.fit.depth - 0.01).abs() < 1e-12, "{ours:?}");
    let exact = exact(&lc, 1.0, 0.25).unwrap();
    assert!((ours.fit.delta_chi2 / exact - 1.0).abs() < 1e-13);
}

// One point of weight 1e300 in the box and one of weight 1 out of it, 1e5
// apart: delta_chi2 is about 1e10, but the likelihood, 0.5 W_in depth^2, is
// about 5e309. The search picks the box by delta_chi2; its fit overflows.
#[test]
fn best_box_whose_fit_overflows() {
    let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![0.0, 1e5], vec![1e-150, 1.0]).unwrap();
    let times = [0.0, 1.0];
    assert_eq!(
        best_binned(&times, &Weighted::new(&lc).unwrap(), 10.0, &[0.5], 1),
        Err(BlsError::OutOfRange)
    );
    assert_eq!(
        TransitBox::best(
            &lc,
            TimeDelta::from_days(10.0).unwrap(),
            TimeDelta::from_days(0.5).unwrap()
        ),
        Err(BlsError::OutOfRange)
    );
}
