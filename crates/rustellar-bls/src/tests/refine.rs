// Refinement: the peaks are the local maxima, highest first, and each refined
// box is the best the exact search finds over its periods and durations,
// searched one by one.

use rstest::rstest;
use rustellar_core::{Epoch, LightCurve, TimeDelta};

use super::recipe::Recipe;
use crate::{
    Objective, TransitBox,
    binned::{Scratch, best_binned},
    box_fit::Weighted,
    refine::{peaks, refine},
};

fn days_of(lc: &LightCurve) -> Vec<f64> {
    lc.time().iter().map(|t| t.as_days()).collect()
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
// each of those periods and durations in turn, and a fine scan of durations
// about the best one at each period; is the best of that scan at its own
// period, within half a step of the grid's; and fits to itself.
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
    let times: Vec<f64> = lc.time().iter().map(|t| t.as_days()).collect();
    let weighted = Weighted::new(&lc).unwrap();
    let periods: Vec<f64> = (0..400).map(|i| 1.0 + 9.0 * f64::from(i) / 399.0).collect();
    let durations = [0.05, 0.08, 0.12, 0.16];
    let mut scratch = Scratch::default();
    let power: Vec<f64> = periods
        .iter()
        .map(|&p| {
            best_binned(
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
        &days_of(&lc),
        &weighted,
        &periods,
        &durations,
        &top,
        2,
        Objective::DeltaChi2,
        1,
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
        // Step 3 does at least as well, at a period within half a step of
        // the grid's, where the same scan of durations finds nothing better.
        assert!(r.fit.delta_chi2 >= scan.0 * (1.0 - 1e-13), "peak {peak}");
        let step = periods[1] - periods[0];
        assert!(
            (r.days - periods[r.period]).abs() <= 0.5 * step,
            "peak {peak}"
        );
        let mut own = 0.0_f64;
        for k in 0..=200 {
            let d = below + (above - below) * f64::from(k) / 200.0;
            if let Some(b) = TransitBox::best(&lc, days(r.days), days(d)).unwrap() {
                own = own.max(b.fit.delta_chi2);
            }
        }
        assert!(
            (r.fit.delta_chi2 / own - 1.0).abs() < 1e-13,
            "peak {peak}: {} vs {own}",
            r.fit.delta_chi2
        );
        let refit = TransitBox::new(
            days(r.days),
            days(r.duration),
            Epoch::from_days(times[0] + r.centre).unwrap(),
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
    assert!((best.days - 3.1).abs() < 0.03, "{best:?}");
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
    let lc = LightCurve::from_days(time, flux, vec![1e-3; 200]).unwrap();
    let weighted = Weighted::new(&lc).unwrap();
    let periods = [1.9, 2.0, 2.1];
    // The 3-day duration is longer than every period, and is skipped.
    let refined = refine(
        &days_of(&lc),
        &weighted,
        &periods,
        &[0.1, 3.0],
        &[0, 2],
        5,
        Objective::DeltaChi2,
        1,
    )
    .unwrap();
    assert_eq!(refined.len(), 2);
    for r in &refined {
        assert_eq!((r.period, r.duration), (1, 0.1));
    }
    // Flat flux: no box has a dip, and no peak is kept.
    let flat = LightCurve::from_days(
        lc.time().iter().map(|t| t.as_days()).collect(),
        vec![1.0; 200],
        vec![1e-3; 200],
    )
    .unwrap();
    let weighted = Weighted::new(&flat).unwrap();
    assert!(
        refine(
            &days_of(&flat),
            &weighted,
            &periods,
            &[0.1],
            &[1],
            1,
            Objective::DeltaChi2,
            1
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
    let lc = LightCurve::from_days(time, flux, vec![1e-3; 20]).unwrap();
    let weighted = Weighted::new(&lc).unwrap();
    let refined = refine(
        &days_of(&lc),
        &weighted,
        &[0.4, 1.0, 5.0],
        &[0.5],
        &[1],
        1,
        Objective::DeltaChi2,
        1,
    )
    .unwrap();
    assert_eq!(refined.len(), 1);
    assert_eq!((refined[0].period, refined[0].fit.n_in), (2, 4));
}

// Daily points with a dip every third day, on a grid of 0.5 and 1.5 days:
// the box at 1.5 days holds the dips. Step 3 searches from halfway to 0.5
// days, 1 day, where every point has the same phase and no box has a dip,
// so it keeps the best of the other periods it tries.
#[test]
fn refined_period_without_a_box() {
    let time: Vec<f64> = (0..30).map(f64::from).collect();
    let flux = (0..30)
        .map(|i| if i % 3 == 0 { 0.99 } else { 1.0 })
        .collect();
    let lc = LightCurve::from_days(time, flux, vec![1e-3; 30]).unwrap();
    let weighted = Weighted::new(&lc).unwrap();
    let refined = refine(
        &days_of(&lc),
        &weighted,
        &[0.5, 1.5],
        &[0.2],
        &[1],
        1,
        Objective::DeltaChi2,
        1,
    )
    .unwrap();
    assert_eq!(refined.len(), 1);
    let r = refined[0];
    assert_eq!((r.period, r.fit.n_in), (1, 10));
    assert!((1.0..=1.5).contains(&r.days), "{r:?}");
}

// Weights of 1e300 and a dip of 1e5: every box's delta_chi2 overflows.
#[test]
fn statistics_that_overflow() {
    let lc =
        LightCurve::from_days(vec![0.0, 1.0, 2.0], vec![1e5, 0.0, 1e5], vec![1e-150; 3]).unwrap();
    let weighted = Weighted::new(&lc).unwrap();
    assert_eq!(
        refine(
            &days_of(&lc),
            &weighted,
            &[10.0],
            &[0.5],
            &[0],
            0,
            Objective::DeltaChi2,
            1
        ),
        Err(crate::BlsError::OutOfRange)
    );
}

// ==========================
// Injected transits, against astropy
// ==========================

// A box that only the second step tries can overflow too. The run from 0.12
// to 0.32 days holds the point of weight 1e300 and spans 0.2 days, so only a
// box longer than 0.2 days, and no longer than the 0.26 days between its
// neighbours, holds it; the grid's 0.2 and 0.3 days never do, and their best
// box fits.
#[test]
fn second_step_overflows() {
    let lc = LightCurve::from_days(
        vec![0.09, 0.12, 0.14, 0.32, 0.35, 0.58, 0.73],
        vec![1e5, 0.0, 0.0, -1e5, 1e5, -1e5, 1e5],
        vec![1.0, 1e-150, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let weighted = Weighted::new(&lc).unwrap();
    let folded = crate::exact::Folded::new(&days_of(&lc), &weighted, 1.0);
    assert!(
        folded
            .best_of(&weighted, &[0.2, 0.3], Objective::DeltaChi2)
            .unwrap()
            .is_some()
    );
    assert_eq!(
        refine(
            &days_of(&lc),
            &weighted,
            &[1.0],
            &[0.2, 0.3],
            &[0],
            0,
            Objective::DeltaChi2,
            1
        ),
        Err(crate::BlsError::OutOfRange)
    );
}

// Every period near every peak is a job of its own: the same refined boxes,
// to the bit, on any number of threads, with peaks whose reaches overlap and
// one at each end of the periods.
#[rstest]
fn same_on_any_number_of_threads(#[values(2, 3, 7, 64)] threads: usize) {
    let lc = Recipe {
        seed: 22,
        n: 600,
        t_start: 1325.0,
        cadence: 0.03,
        noise: 1e-3,
        period: 2.3,
        tc: 1325.7,
        duration: 0.1,
        depth: 2e-3,
    }
    .light_curve();
    let times = days_of(&lc);
    let weighted = Weighted::new(&lc).unwrap();
    let periods: Vec<f64> = (0..60).map(|i| 1.5 + 0.03 * f64::from(i)).collect();
    let durations = [0.05, 0.1, 0.2];
    let top = [27, 28, 0, 59, 30, 12];
    let one = refine(
        &times,
        &weighted,
        &periods,
        &durations,
        &top,
        3,
        Objective::Snr,
        1,
    )
    .unwrap();
    assert_eq!(one.len(), top.len());
    let many = refine(
        &times,
        &weighted,
        &periods,
        &durations,
        &top,
        3,
        Objective::Snr,
        threads,
    )
    .unwrap();
    for (a, b) in one.iter().zip(&many) {
        assert_eq!((a.peak, a.period), (b.peak, b.period));
        assert_eq!(
            [a.duration, a.centre, a.fit.snr].map(f64::to_bits),
            [b.duration, b.centre, b.fit.snr].map(f64::to_bits)
        );
    }
    assert_eq!(one, many);
}
