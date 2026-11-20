//! `Bls::search_whitened` against its definition: each duration searched
//! alone, each box's power and fit divided by the information of its own
//! duration (here `1 + edge / D`; the square root of it for the SNR), and
//! the best duration kept at each period.

use std::num::NonZeroUsize;

use rustellar_bls::{
    BestBox, Bls, BlsError, BlsResult, Device, Durations, Objective, Periods, SearchOptions,
};
use rustellar_core::{LightCurve, TimeDelta};

/// Every 30 minutes for 20 days: a 0.005 dip lasting 0.15 days every 3
/// days, in noise (a fixed generator).
fn curve() -> LightCurve {
    let mut state = 3_u64;
    let time: Vec<f64> = (0..960).map(|i| f64::from(i) / 48.0).collect();
    let flux = time
        .iter()
        .map(|t| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[expect(clippy::cast_precision_loss, reason = "a fixed generator")]
            let noise = 2e-3 * ((state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5);
            let dip = if (t % 3.0 - 1.0).abs() < 0.075 {
                0.995
            } else {
                1.0
            };
            dip + noise
        })
        .collect();
    LightCurve::from_days(time, flux, vec![1e-3; 960]).unwrap()
}

fn grid() -> (Periods, Durations) {
    let periods: Vec<f64> = (0..300).map(|k| 1.5 + 0.01 * f64::from(k)).collect();
    (
        Periods::from_days(&periods).unwrap(),
        Durations::from_days(&[0.08, 0.15, 0.25]).unwrap(),
    )
}

fn options(objective: Objective) -> SearchOptions {
    SearchOptions::default()
        .with_objective(objective)
        .with_threads(NonZeroUsize::new(2).unwrap())
        .with_device(Device::Cpu)
}

/// The definition, from plain searches of one duration each.
fn by_hand(bls: &Bls, edge: f64, objective: Objective) -> (Vec<f64>, Vec<Option<BestBox>>) {
    let (periods, durations) = grid();
    let mut power = vec![0.0; periods.as_slice().len()];
    let mut boxes = vec![None; power.len()];
    for d in durations.as_slice() {
        let one = Durations::new(vec![*d]).unwrap();
        let run = bls.search(&periods, &one, &options(objective)).unwrap();
        for (k, p) in run.power().iter().enumerate() {
            // Each box by its own duration (whole phase bins).
            let Some(found) = run.boxes()[k] else {
                continue;
            };
            let factor = 1.0 + edge / found.transit.duration().as_days();
            let scale = if objective == Objective::Snr {
                factor.sqrt()
            } else {
                factor
            };
            if p / scale > power[k] {
                power[k] = p / scale;
                boxes[k] = Some(found);
            }
        }
    }
    (power, boxes)
}

fn whitened(bls: &Bls, edge: f64, objective: Objective) -> BlsResult {
    let (periods, durations) = grid();
    bls.search_whitened(&periods, &durations, &options(objective), |d: TimeDelta| {
        1.0 + edge / d.as_days()
    })
    .unwrap()
}

#[test]
fn power_and_boxes_by_definition() {
    let bls = Bls::new(&curve()).unwrap();
    for objective in [
        Objective::DeltaChi2,
        Objective::Snr,
        Objective::AstropyLikelihood,
    ] {
        for edge in [0.0, 0.05, 0.4] {
            let result = whitened(&bls, edge, objective);
            let (power, boxes) = by_hand(&bls, edge, objective);
            assert_eq!(result.power(), power.as_slice(), "{objective:?} {edge}");
            for (got, raw) in result.boxes().iter().zip(&boxes) {
                assert_eq!(got.is_some(), raw.is_some());
                let (Some(got), Some(raw)) = (got, raw) else {
                    continue;
                };
                assert_eq!(got.transit, raw.transit);
                let f = 1.0 + edge / raw.transit.duration().as_days();
                let fit = got.fit;
                assert_eq!(fit.delta_chi2, raw.fit.delta_chi2 / f);
                assert_eq!(fit.likelihood, raw.fit.likelihood / f);
                assert_eq!(fit.depth, raw.fit.depth / f);
                assert_eq!(fit.depth_err, raw.fit.depth_err / f.sqrt());
                assert_eq!(fit.snr, raw.fit.snr / f.sqrt());
                assert_eq!(fit.flux_out, raw.fit.flux_out);
                assert_eq!(fit.flux_in, raw.fit.flux_out - fit.depth);
                assert_eq!((fit.n_in, fit.weight_in), (raw.fit.n_in, raw.fit.weight_in));
            }
        }
    }
}

#[test]
fn peaks_highest_first_one_a_period() {
    let bls = Bls::new(&curve()).unwrap();
    let result = whitened(&bls, 0.1, Objective::DeltaChi2);
    let peaks = result.peaks();
    assert!(!peaks.is_empty() && peaks.len() <= 5);
    assert!(
        peaks
            .windows(2)
            .all(|w| w[0].grid_objective >= w[1].grid_objective)
    );
    let mut seen: Vec<usize> = peaks.iter().map(|p| p.peak_index).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), peaks.len());
    let best = result.best().unwrap();
    assert!((best.transit.period().as_days() - 3.0).abs() < 0.01);
    // The peaks' fits are scaled as the boxes'.
    let p = &peaks[0];
    let f = 1.0 + 0.1 / p.best.transit.duration().as_days();
    assert!(
        (p.best.fit.snr * p.best.fit.snr - p.best.fit.delta_chi2).abs()
            < 1e-9 * p.best.fit.delta_chi2
    );
    assert!(f > 1.0);
    // No refinement: no peaks.
    let (periods, durations) = grid();
    let none = bls
        .search_whitened(
            &periods,
            &durations,
            &options(Objective::DeltaChi2).with_refine_peaks(0),
            |d: TimeDelta| 1.0 + 0.1 / d.as_days(),
        )
        .unwrap();
    assert!(none.peaks().is_empty());
    assert!(none.best().is_some());
}

#[test]
fn a_factor_not_above_zero_is_out_of_range() {
    let bls = Bls::new(&curve()).unwrap();
    let (periods, durations) = grid();
    for bad in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert_eq!(
            bls.search_whitened(&periods, &durations, &options(Objective::Snr), |_| bad),
            Err(BlsError::OutOfRange)
        );
    }
    // Above zero at the durations searched but not at the boxes found
    // (whole phase bins, a little shorter or longer).
    let picky = |d: TimeDelta| {
        if [0.08, 0.15, 0.25].contains(&d.as_days()) {
            1.0
        } else {
            -1.0
        }
    };
    assert_eq!(
        bls.search_whitened(&periods, &durations, &options(Objective::Snr), picky),
        Err(BlsError::OutOfRange)
    );
    // The search's own errors pass through.
    let short = Periods::from_days(&[0.2]).unwrap();
    assert!(
        bls.search_whitened(&short, &durations, &options(Objective::Snr), |_| 1.0)
            .is_err()
    );
}
