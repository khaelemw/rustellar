// Refinement: the peaks are the local maxima, highest first, and each refined
// box is the best the exact search finds over its periods and durations,
// searched one by one.

use rstest::rstest;
use rustellar_core::{Epoch, LightCurve, TimeDelta};

use super::recipe::{Recipe, check_samples};
use crate::{
    Objective, TransitBox,
    binned::{Scratch, best_binned},
    box_fit::{Weighted, offset_from_transit},
    refine::{peaks, refine},
};

fn btjd(lc: &LightCurve) -> Vec<f64> {
    lc.time().iter().map(|t| t.as_btjd()).collect()
}

#[rstest]
#[case::several(&[1.0, 3.0, 2.0, 5.0, 5.0, 4.0, 0.0, 7.0], 2, &[7, 3])]
#[case::all_of_them(&[1.0, 3.0, 2.0, 5.0, 5.0, 4.0, 0.0, 7.0], 9, &[7, 3, 1])]
#[case::ends(&[5.0, 1.0, 2.0], 5, &[0, 2])]
#[case::equal_heights(&[0.0, 2.0, 0.0, 2.0, 0.0], 5, &[1, 3])]
#[case::flat(&[1.0, 1.0, 1.0], 5, &[0])]
#[case::zero(&[0.0, 0.0, 1.0, 0.0], 5, &[2])]
#[case::all_zero(&[0.0, 0.0], 5, &[])]
#[case::no_dip_anywhere(&[f64::NEG_INFINITY, 0.5, f64::NEG_INFINITY], 5, &[1])]
#[case::one(&[4.0], 1, &[0])]
#[case::empty(&[], 3, &[])]
#[case::none_wanted(&[1.0, 2.0], 0, &[])]
fn local_maxima(#[case] power: &[f64], #[case] count: usize, #[case] expected: &[usize]) {
    assert_eq!(peaks(power, count), expected);
}

fn days(value: f64) -> TimeDelta {
    TimeDelta::from_days(value).unwrap()
}

// A transit of depth 3e-3 every 3.1 days in 27 days of data, the binned
// power at 400 periods from 1 to 10 days, and its three highest peaks refined
// two periods either side. Each refined box beats the exact search run at
// each of those periods and durations in turn, is the best of a fine scan of
// durations about the best one at each period, and fits to itself.
#[test]
fn equals_the_exact_search_at_each_period() {
    let lc = Recipe {
        seed: 21,
        n: 1350,
        t_start: 1325.0,
        cadence: 0.02,
        noise: 1e-3,
        period: 3.1,
        tc: 1326.2,
        duration: 0.1,
        depth: 3e-3,
    }
    .light_curve();
    let times: Vec<f64> = lc.time().iter().map(|t| t.as_btjd()).collect();
    let weighted = Weighted::new(&lc).unwrap();
    let periods: Vec<f64> = (0..400).map(|i| 1.0 + 9.0 * f64::from(i) / 399.0).collect();
    let durations = [0.05, 0.08, 0.12, 0.16];
    let mut scratch = Scratch::default();
    let power: Vec<f64> = periods
        .iter()
        .map(|&p| {
            best_binned(
                &times,
                &weighted,
                p,
                &durations,
                8,
                Objective::DeltaChi2,
                &mut scratch,
            )
            .unwrap()
            .map_or(0.0, |b| b.fit.delta_chi2)
        })
        .collect();
    let top = peaks(&power, 3);
    let refined = refine(
        &btjd(&lc),
        &weighted,
        &periods,
        &durations,
        &top,
        2,
        Objective::DeltaChi2,
    )
    .unwrap();
    assert_eq!(refined.len(), 3);

    for (r, &peak) in refined.iter().zip(&top) {
        assert_eq!(r.peak, peak);
        let mut brute = (0.0, 0, 0);
        for (period, &p) in periods.iter().enumerate().skip(peak - 2).take(5) {
            for (duration, &d) in durations.iter().enumerate() {
                if let Some(b) = TransitBox::best(&lc, days(p), days(d)).unwrap() {
                    if b.fit.delta_chi2 > brute.0 {
                        brute = (b.fit.delta_chi2, period, duration);
                    }
                }
            }
        }
        // Step 2 does at least as well as the grid, with a duration between
        // the grid's best one's neighbours.
        assert!(r.fit.delta_chi2 >= brute.0 * (1.0 - 1e-13), "peak {peak}");
        let (below, above) = match brute.2 {
            0 => (durations[0], durations[1]),
            3 => (durations[2], durations[3]),
            k => (durations[k - 1], durations[k + 1]),
        };
        assert!((below..=above).contains(&r.duration), "peak {peak}: {r:?}");
        // The exact search at 200 durations across that range, at each of
        // the five periods, finds nothing better, and the same at best: the
        // sets a duration can hold change every few thousandths of a day
        // here, much more slowly than this.
        let mut scan = (0.0_f64, 0);
        for (period, &p) in periods.iter().enumerate().skip(peak - 2).take(5) {
            for k in 0..=200 {
                let d = below + (above - below) * f64::from(k) / 200.0;
                if let Some(b) = TransitBox::best(&lc, days(p), days(d)).unwrap() {
                    if b.fit.delta_chi2 > scan.0 {
                        scan = (b.fit.delta_chi2, period);
                    }
                }
            }
        }
        assert_eq!(r.period, scan.1, "peak {peak}");
        assert!(
            (r.fit.delta_chi2 / scan.0 - 1.0).abs() < 1e-13,
            "peak {peak}: {} vs {}",
            r.fit.delta_chi2,
            scan.0
        );
        let refit = TransitBox::new(
            days(periods[r.period]),
            days(r.duration),
            Epoch::from_btjd(times[0] + r.centre).unwrap(),
        )
        .unwrap()
        .fit(&lc)
        .unwrap();
        assert_eq!(refit.n_in, r.fit.n_in);
        assert!((refit.delta_chi2 / r.fit.delta_chi2 - 1.0).abs() < 1e-13);
    }
    // The highest peak is the transit's period, and its refined depth is the
    // injected one, to the noise.
    let best = refined[0];
    assert!((periods[best.period] - 3.1).abs() < 0.03, "{best:?}");
    assert!((best.fit.depth - 3e-3).abs() < 3.0 * best.fit.depth_err);
}

// Near either end of the periods, the periods either side are cut off.
#[test]
fn reach_stops_at_the_ends() {
    let time: Vec<f64> = (0..200).map(|i| 1325.0 + 0.05 * f64::from(i)).collect();
    let flux = time
        .iter()
        .map(|t| if (t - 1325.0) % 2.0 < 0.1 { 0.99 } else { 1.0 })
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 200]).unwrap();
    let weighted = Weighted::new(&lc).unwrap();
    let periods = [1.9, 2.0, 2.1];
    // The 3-day duration is longer than every period, and is skipped.
    let refined = refine(
        &btjd(&lc),
        &weighted,
        &periods,
        &[0.1, 3.0],
        &[0, 2],
        5,
        Objective::DeltaChi2,
    )
    .unwrap();
    assert_eq!(refined.len(), 2);
    for r in &refined {
        assert_eq!((r.period, r.duration), (1, 0.1));
    }
    // Flat flux: no box has a dip, and no peak is kept.
    let flat = LightCurve::from_btjd(
        lc.time().iter().map(|t| t.as_btjd()).collect(),
        vec![1.0; 200],
        vec![1e-3; 200],
    )
    .unwrap();
    let weighted = Weighted::new(&flat).unwrap();
    assert!(
        refine(
            &btjd(&flat),
            &weighted,
            &periods,
            &[0.1],
            &[1],
            1,
            Objective::DeltaChi2
        )
        .unwrap()
        .is_empty()
    );
}

// Daily points with a dip every fifth day, near periods of 0.4, 1 and 5
// days: at 1 day every point has the same phase, so no box has a dip, and at
// 0.4 days the half-day box is longer than the period. Only 5 days has a box.
#[test]
fn periods_near_a_peak_without_a_box() {
    let time: Vec<f64> = (0..20).map(f64::from).collect();
    let flux = (0..20)
        .map(|i| if i % 5 == 0 { 0.99 } else { 1.0 })
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 20]).unwrap();
    let weighted = Weighted::new(&lc).unwrap();
    let refined = refine(
        &btjd(&lc),
        &weighted,
        &[0.4, 1.0, 5.0],
        &[0.5],
        &[1],
        1,
        Objective::DeltaChi2,
    )
    .unwrap();
    assert_eq!(refined.len(), 1);
    assert_eq!((refined[0].period, refined[0].fit.n_in), (2, 4));
}

// Weights of 1e300 and a dip of 1e5: every box's delta_chi2 overflows.
#[test]
fn statistics_that_overflow() {
    let lc =
        LightCurve::from_btjd(vec![0.0, 1.0, 2.0], vec![1e5, 0.0, 1e5], vec![1e-150; 3]).unwrap();
    let weighted = Weighted::new(&lc).unwrap();
    assert_eq!(
        refine(
            &btjd(&lc),
            &weighted,
            &[10.0],
            &[0.5],
            &[0],
            0,
            Objective::DeltaChi2
        ),
        Err(crate::BlsError::OutOfRange)
    );
}

// ==========================
// Injected transits, against astropy
// ==========================

const INJECTED: &str = include_str!("../../tests/reference/injected.csv");

// How far a reported box is from the injected one: the mid-transit time at
// the transit nearest the middle of the data (where a slightly wrong period
// moves it least), and the depth.
struct Errors {
    time: [f64; 2],
    depth: [f64; 2],
}

fn injected() -> Vec<Errors> {
    let durations = [0.04, 0.06, 0.08, 0.11, 0.15, 0.2];
    let middle = 1325.0 + 13.5;
    let at_middle = |t0: f64, p: f64| t0 + ((middle - t0) / p).round() * p;
    INJECTED
        .lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| {
            let v: Vec<&str> = line.split(',').collect();
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
            let times: Vec<f64> = lc.time().iter().map(|t| t.as_btjd()).collect();
            let weighted = Weighted::new(&lc).unwrap();
            // The generator's periods, operation for operation.
            let periods: Vec<f64> = (0..201)
                .map(|k| period * (1.0 + 0.02 * f64::from(k - 100) / 100.0))
                .collect();
            let mut scratch = Scratch::default();
            let power: Vec<f64> = periods
                .iter()
                .map(|&p| {
                    best_binned(
                        &times,
                        &weighted,
                        p,
                        &durations,
                        10,
                        Objective::DeltaChi2,
                        &mut scratch,
                    )
                    .unwrap()
                    .map_or(0.0, |b| b.fit.delta_chi2)
                })
                .collect();
            let ours = refine(
                &btjd(&lc),
                &weighted,
                &periods,
                &durations,
                &peaks(&power, 1),
                2,
                Objective::DeltaChi2,
            )
            .unwrap()[0];
            let ours_time = at_middle(times[0] + ours.centre, periods[ours.period]);
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
// | random     | 1.05e-3 (5.8e-3)  | 2.19e-3 (1.27e-2) | 5.0e-5      | 5.9e-5         |
// | bright     | 7.2e-4 (4.2e-3)   | 2.82e-3 (9.6e-3)  | 7.9e-5      | 2.41e-4        |
// | phase zero | 6.8e-4 (3.8e-3)   | 3.36e-3 (2.06e-2) | 6.7e-5      | 1.15e-4        |
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

// A box that only the second step tries can overflow too. The run from 0.12
// to 0.32 days holds the point of weight 1e300 and spans 0.2 days, so only a
// box longer than 0.2 days, and no longer than the 0.26 days between its
// neighbours, holds it; the grid's 0.2 and 0.3 days never do, and their best
// box fits.
#[test]
fn second_step_overflows() {
    let lc = LightCurve::from_btjd(
        vec![0.09, 0.12, 0.14, 0.32, 0.35, 0.58, 0.73],
        vec![1e5, 0.0, 0.0, -1e5, 1e5, -1e5, 1e5],
        vec![1.0, 1e-150, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let weighted = Weighted::new(&lc).unwrap();
    let folded = crate::exact::Folded::new(&btjd(&lc), &weighted, 1.0);
    assert!(
        folded
            .best_of(&weighted, &[0.2, 0.3], Objective::DeltaChi2)
            .unwrap()
            .is_some()
    );
    assert_eq!(
        refine(
            &btjd(&lc),
            &weighted,
            &[1.0],
            &[0.2, 0.3],
            &[0],
            0,
            Objective::DeltaChi2
        ),
        Err(crate::BlsError::OutOfRange)
    );
}
