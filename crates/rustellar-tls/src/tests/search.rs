// The whole search: an injected transit recovered from Ofir's period grid,
// nothing to fit, the peaks it refines, and threads.

use std::num::NonZeroUsize;

use rustellar_bls::OfirPeriod;
use rustellar_core::{Epoch, LightCurve, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

use super::Random;
use crate::search::{Options, peaks, run};

/// 27 days at 30 minutes, a transit of radius ratio `k` (from the model
/// crate, not the template), noise `noise`.
fn injected(random: &mut Random, k: f64, period: f64, t0: f64, noise: f64) -> LightCurve {
    let time: Vec<f64> = (0..1296).map(|i| f64::from(i) / 48.0).collect();
    let params = TransitParams::new(
        Epoch::from_btjd(t0).unwrap(),
        TimeDelta::from_days(period).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(12.0).unwrap(),
        ImpactParameter::new(0.2).unwrap(),
        LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
    )
    .unwrap();
    let epochs: Vec<Epoch> = time.iter().map(|&t| Epoch::from_btjd(t).unwrap()).collect();
    let flux: Vec<f64> = params
        .light_curve(&epochs)
        .into_iter()
        .map(|f| {
            let n: f64 = (0..4).map(|_| random.uniform() - 0.5).sum::<f64>() * 3_f64.sqrt();
            f + noise * n
        })
        .collect();
    LightCurve::from_btjd(time, flux, vec![noise; 1296]).unwrap()
}

fn periods(lc: &LightCurve) -> Vec<f64> {
    OfirPeriod::default()
        .periods(lc)
        .unwrap()
        .as_slice()
        .iter()
        .map(|p| p.as_days())
        .collect()
}

#[test]
fn an_injected_transit_is_recovered() {
    let mut random = Random::new(51);
    for (k, period, t0) in [(0.04, 3.7, 1.3), (0.03, 5.9, 4.4), (0.05, 1.37, 0.2)] {
        let lc = injected(&mut random, k, period, t0, 4e-4);
        let grid = periods(&lc);
        let mut options = Options::tls();
        options.threads = NonZeroUsize::new(4).unwrap();
        let outcome = run(&lc, &grid, options);
        let best = outcome.best.unwrap();
        let spectra = outcome.spectra.unwrap();
        let p = best.position;
        assert!((p.period / period - 1.0).abs() < 1e-3, "{p:?}");
        let off = (p.t0 - t0).rem_euclid(period);
        assert!(off.min(period - off) < 0.01, "{p:?}");
        assert!(spectra.sde > 9.0, "{}", spectra.sde);
        assert!(outcome.period_uncertainty > 0.0 && outcome.period_uncertainty < 0.05 * period);
        assert_eq!(outcome.periods.len(), grid.len());
        // The model's depth at mid-transit, 1 - F(b), near k² (limb
        // darkening deepens the middle).
        assert!((best.fit.depth / (k * k) - 1.0).abs() < 0.4, "{best:?}");
    }
}

#[test]
fn nothing_to_fit_gives_no_spectra() {
    // Every flux above 1: no depth anywhere.
    let time: Vec<f64> = (0..500).map(|i| f64::from(i) / 20.0).collect();
    let lc = LightCurve::from_btjd(time, vec![1.001; 500], vec![1e-3; 500]).unwrap();
    let outcome = run(&lc, &[1.0, 1.5, 2.0, 3.0], Options::tls());
    assert!(outcome.spectra.is_none() && outcome.best.is_none());
    assert_eq!(outcome.period_uncertainty, f64::INFINITY);
    assert!(outcome.periods.iter().all(|p| p.depth == 0.0));
}

#[test]
fn one_period_is_refined_within_rounding_of_it() {
    let mut random = Random::new(52);
    let lc = injected(&mut random, 0.05, 2.5, 0.7, 3e-4);
    let mut options = Options::tls();
    options.min_depth = 0.0;
    // The peak at the first period and at the last: refined within the
    // neighbouring period, and to the true one.
    for grid in [[2.5, 2.6], [2.4, 2.5]] {
        let best = run(&lc, &grid, options).best.unwrap();
        assert!((best.position.period - 2.5).abs() < 1e-3, "{best:?}");
    }
}

#[test]
fn the_peaks_are_local_maxima_highest_first() {
    let power = [3.0, 1.0, 5.0, 5.0, 2.0, 7.0, 0.0, 4.0];
    // A plateau's peak is its first value.
    assert_eq!(peaks(&power, 10), vec![5, 2, 7, 0]);
    assert_eq!(peaks(&power, 2), vec![5, 2]);
    assert_eq!(peaks(&[1.0], 3), vec![0]);
    assert!(peaks(&power, 0).is_empty());
}

#[test]
fn the_same_outcome_on_any_number_of_threads() {
    let mut random = Random::new(53);
    let lc = injected(&mut random, 0.04, 3.1, 0.9, 5e-4);
    let grid: Vec<f64> = periods(&lc).into_iter().step_by(7).collect();
    let base = run(&lc, &grid, Options::tls());
    for threads in [2, 5] {
        let mut options = Options::tls();
        options.threads = NonZeroUsize::new(threads).unwrap();
        assert_eq!(run(&lc, &grid, options), base, "{threads}");
    }
}
