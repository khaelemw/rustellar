// The exact search at one period and duration: the same best box as exact
// rational arithmetic, never worse than astropy's slow search, and the edge
// cases.

use rstest::rstest;
use rustellar_bls::{BestBox, BlsError, TransitBox};
use rustellar_core::{Epoch, LightCurve, TimeDelta};

// ==========================
// Helper Functions
// ==========================

const EXACT_CASES: &str = include_str!("reference/exact_cases.csv");
const EXACT_POINTS: &str = include_str!("reference/exact_points.csv");
const EXAMPLE_3: &str = include_str!("reference/example_3_points.csv");
const SLOW_POWER: &str = include_str!("reference/slow_power.csv");

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

fn days(value: f64) -> TimeDelta {
    TimeDelta::from_days(value).unwrap()
}

fn best(lc: &LightCurve, period: f64, duration: f64) -> Option<BestBox> {
    TransitBox::best(lc, days(period), days(duration)).unwrap()
}

fn relative(a: f64, b: f64) -> f64 {
    (a / b - 1.0).abs()
}

struct ExactCase {
    lc: LightCurve,
    period: f64,
    duration: f64,
    // The best delta_chi2 and which points it holds, or None.
    best: Option<(f64, Vec<bool>)>,
}

fn exact_cases() -> Vec<ExactCase> {
    let points: Vec<Vec<&str>> = rows(EXACT_POINTS).collect();
    rows(EXACT_CASES)
        .map(|v| {
            let case = v[0];
            let (mut time, mut flux, mut flux_err) = (vec![], vec![], vec![]);
            for p in points.iter().filter(|p| p[0] == case) {
                time.push(p[1].parse().unwrap());
                flux.push(p[2].parse().unwrap());
                flux_err.push(p[3].parse().unwrap());
            }
            ExactCase {
                lc: LightCurve::from_btjd(time, flux, flux_err).unwrap(),
                period: v[1].parse().unwrap(),
                duration: v[2].parse().unwrap(),
                best: (!v[3].is_empty()).then(|| {
                    (
                        v[3].parse().unwrap(),
                        v[4].chars().map(|c| c == '1').collect(),
                    )
                }),
            }
        })
        .collect()
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

fn in_transit(lc: &LightCurve, transit: &TransitBox) -> Vec<bool> {
    lc.time().iter().map(|&t| transit.contains(t)).collect()
}

// ==========================
// Against exact arithmetic
// ==========================

#[test]
fn reference_files_are_complete() {
    let cases = exact_cases();
    assert_eq!(cases.len(), 240);
    for case in &cases {
        assert!((6..=20).contains(&case.lc.len()));
    }
}

// 240 data sets of 6 to 20 points; the last 40 on an exact grid of times,
// where points lie exactly one duration apart in phase.
#[test]
fn same_best_box_as_exact_arithmetic() {
    for (i, case) in exact_cases().iter().enumerate() {
        let ours = best(&case.lc, case.period, case.duration);
        match (&ours, &case.best) {
            (None, None) => {}
            (Some(ours), Some((delta_chi2, mask))) => {
                assert!(
                    relative(ours.fit.delta_chi2, *delta_chi2) < 1e-13,
                    "case {i}: {} vs {delta_chi2}",
                    ours.fit.delta_chi2
                );
                assert_eq!(&in_transit(&case.lc, &ours.transit), mask, "case {i}");
                assert_eq!(ours.fit.n_in, mask.iter().filter(|&&m| m).count());
            }
            _ => panic!("case {i}: {ours:?} vs {:?}", case.best),
        }
    }
}

// The box returned, fitted afresh, gives the same statistics.
#[test]
fn best_box_refits_to_itself() {
    for (i, case) in exact_cases().iter().enumerate() {
        if let Some(best) = best(&case.lc, case.period, case.duration) {
            let refit = best.transit.fit(&case.lc).unwrap();
            assert_eq!(refit.n_in, best.fit.n_in, "case {i}");
            assert!(
                relative(refit.delta_chi2, best.fit.delta_chi2) < 1e-13,
                "case {i}"
            );
            assert!(relative(refit.depth, best.fit.depth) < 1e-12, "case {i}");
        }
    }
}

// No mid-transit time on a fine grid does better.
#[test]
fn no_grid_point_does_better() {
    for case in exact_cases().iter().take(60) {
        let Some(ours) = best(&case.lc, case.period, case.duration) else {
            continue;
        };
        let start = case.lc.time()[0].as_btjd();
        for k in 0..2_000 {
            let t0 = start + case.period * f64::from(k) / 2_000.0;
            let transit = TransitBox::new(
                days(case.period),
                days(case.duration),
                Epoch::from_btjd(t0).unwrap(),
            )
            .unwrap();
            if let Ok(fit) = transit.fit(&case.lc) {
                if fit.depth > 0.0 {
                    assert!(fit.delta_chi2 <= ours.fit.delta_chi2 * (1.0 + 1e-13));
                }
            }
        }
    }
}

// ==========================
// Against astropy's slow search
// ==========================

// The worked example: astropy's box, fitted here, gives astropy's numbers.
#[test]
fn worked_example_3_box() {
    let transit = TransitBox::new(
        days(3.0),
        days(0.12),
        Epoch::from_btjd(1.302_942_952_053_283).unwrap(),
    )
    .unwrap();
    let fit = transit.fit(&example_3()).unwrap();
    assert!(
        relative(fit.likelihood, 348.187_970_700_163_9) < 1e-12,
        "{fit:?}"
    );
    assert!(
        relative(fit.depth, 0.005_078_553_094_469_474) < 1e-12,
        "{fit:?}"
    );
    assert!(relative(fit.snr, 25.482_750_349_124_515) < 1e-12, "{fit:?}");
}

// At P = 3 and D = 0.12, the exact search finds that box or a better one.
#[test]
fn worked_example_3_search() {
    let ours = best(&example_3(), 3.0, 0.12).unwrap();
    assert!(
        ours.fit.snr >= 25.482_750_349_124_515 * (1.0 - 1e-12),
        "{ours:?}"
    );
}

// At each of 300 periods from 1 to 6 days, the best snr over the durations
// 0.06, 0.12 and 0.2 is never below astropy's slow search, which tries a
// grid of mid-transit times inside the same space.
#[test]
fn never_below_astropy_slow() {
    let lc = example_3();
    let mut better = 0;
    for v in rows(SLOW_POWER) {
        let period: f64 = v[0].parse().unwrap();
        let theirs: f64 = v[1].parse().unwrap();
        let ours = [0.06, 0.12, 0.2]
            .iter()
            .filter_map(|&d| best(&lc, period, d))
            .map(|b| b.fit.snr)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            ours >= theirs * (1.0 - 1e-12),
            "P {period}: {ours} vs {theirs}"
        );
        if ours > theirs * (1.0 + 1e-9) {
            better += 1;
        }
    }
    // A phase grid misses the best box at most periods: 214 of 300 here.
    assert!(better >= 200, "better at {better} of 300 periods");
}

// ==========================
// Edge cases
// ==========================

// A single transit in two days of data, searched at a 10-day period.
#[test]
fn single_transit() {
    let time: Vec<f64> = (0..200).map(|i| 1325.0 + 0.01 * f64::from(i)).collect();
    let flux = time
        .iter()
        .map(|t| {
            if (t - 1325.8).abs() < 0.05 {
                0.995
            } else {
                1.0
            }
        })
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 200]).unwrap();
    let ours = best(&lc, 10.0, 0.1).unwrap();
    assert!((ours.fit.depth - 0.005).abs() < 1e-12, "{ours:?}");
    assert!((ours.transit.transit_time().as_btjd() - 1325.8).abs() < 0.01);
}

// Two stretches with a gap of a week between, a transit in each.
#[test]
fn data_with_a_gap() {
    let time: Vec<f64> = (0..400)
        .map(|i| {
            let t = 1325.0 + 0.01 * f64::from(i);
            if i < 200 { t } else { t + 7.0 }
        })
        .collect();
    // A dip over the 7 points within 0.035 days of 1325.5 + 2.5 n, and a box
    // that holds 7 points at most.
    let flux = time
        .iter()
        .map(|t| {
            let r = (t - 1325.5).rem_euclid(2.5);
            if r.min(2.5 - r) < 0.035 { 0.996 } else { 1.0 }
        })
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 400]).unwrap();
    let ours = best(&lc, 2.5, 0.065).unwrap();
    assert_eq!(ours.fit.n_in, 14);
    assert!((ours.fit.depth - 0.004).abs() < 1e-12, "{ours:?}");
}

// Every point at the same phase: any box holds all of them or none.
#[test]
fn every_box_holds_every_point() {
    let lc = LightCurve::from_btjd(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 0.9, 1.1, 1.0],
        vec![1e-3; 4],
    )
    .unwrap();
    assert_eq!(best(&lc, 1.0, 0.5), None);
}

#[test]
fn flat_flux_has_no_dip() {
    let time: Vec<f64> = (0..50).map(f64::from).collect();
    let lc = LightCurve::from_btjd(time, vec![1.0; 50], vec![1e-3; 50]).unwrap();
    assert_eq!(best(&lc, 7.3, 0.5), None);
}

#[test]
fn one_point() {
    let lc = LightCurve::from_btjd(vec![1325.0], vec![1.0], vec![1e-3]).unwrap();
    assert_eq!(best(&lc, 1.0, 0.1), None);
}

#[test]
fn bad_box() {
    let lc = example_3();
    assert_eq!(
        TransitBox::best(&lc, days(1.0), days(1.0)),
        Err(BlsError::DurationNotShorterThanPeriod)
    );
}

// A weight that has lost its digits stops the search before it starts,
// rather than part way through at the first box that holds only that point.
#[rstest]
fn weights_out_of_range(#[values(1e-160, 1e155, 1e200)] bad: f64) {
    let lc = LightCurve::from_btjd(
        vec![0.0, 1.0, 2.0],
        vec![1.0, 0.9, 1.0],
        vec![bad, 1.0, 1.0],
    )
    .unwrap();
    assert_eq!(
        TransitBox::best(&lc, days(10.0), days(0.5)),
        Err(BlsError::OutOfRange)
    );
}

// Weights of 1e300 and a dip of 1e5: the likelihood is about 1e310.
#[test]
fn statistics_that_overflow() {
    let lc =
        LightCurve::from_btjd(vec![0.0, 1.0, 2.0], vec![1e5, 0.0, 1e5], vec![1e-150; 3]).unwrap();
    assert_eq!(
        TransitBox::best(&lc, days(10.0), days(0.5)),
        Err(BlsError::OutOfRange)
    );
}

// Points at 0, 1, 2 and 3 days and a 3-day box: the only box that holds just
// the dip at 1 and 2 runs from 0 to 3, its edges on the points either side.
#[test]
fn only_one_mid_transit_time() {
    let lc = LightCurve::from_btjd(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 0.9, 0.9, 1.0],
        vec![1e-3; 4],
    )
    .unwrap();
    let ours = best(&lc, 10.0, 3.0).unwrap();
    assert_eq!(ours.fit.n_in, 2);
    assert_eq!(ours.transit.transit_time().as_btjd(), 1.5);
    assert_eq!(in_transit(&lc, &ours.transit), [false, true, true, false]);
}
