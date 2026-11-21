// The periodogram's API: setup, threads, normalisations against each
// other, and invariances.

use std::f64::consts::TAU;
use std::num::NonZeroUsize;

use rustellar_ls::{Frequencies, LightCurve, LombScargle, LsError, Mean, Normalization};

/// `n` uneven times with a sinusoid of `f` cycles per day and a fixed
/// wiggle as noise, from `offset`, with errors `error`.
fn curve(n: u32, f: f64, offset: f64, error: f64) -> LightCurve {
    let time: Vec<f64> = (0..n)
        .map(|i| offset + 0.13 * f64::from(i) + 0.02 * f64::from((i * i) % 5))
        .collect();
    let flux = (0..n)
        .zip(&time)
        .map(|(i, t)| {
            let noise = 0.3 * f64::from((i * 7919) % 13) / 13.0 - 0.15;
            1.0 + (TAU * f * (t - offset)).sin() + noise
        })
        .collect();
    LightCurve::from_days(time, flux, vec![error; n as usize]).unwrap()
}

fn one(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

#[test]
fn constant_flux_has_no_power() {
    let lc = LightCurve::from_days(vec![0.0, 1.0, 2.5], vec![2.0; 3], vec![0.1; 3]).unwrap();
    for mean in [Mean::Floating, Mean::Subtracted] {
        assert_eq!(
            LombScargle::new(&lc, mean).unwrap_err(),
            LsError::NoVariation
        );
    }
    // Without a mean a constant is a signal, at zero frequency.
    assert!(LombScargle::new(&lc, Mean::Zero).is_ok());
    let zero = LightCurve::from_days(vec![0.0, 1.0], vec![0.0; 2], vec![0.1; 2]).unwrap();
    assert_eq!(
        LombScargle::new(&zero, Mean::Zero).unwrap_err(),
        LsError::NoVariation
    );
    assert_eq!(
        LsError::NoVariation.to_string(),
        "the flux must vary about its reference fit"
    );
}

#[test]
fn threads_change_nothing() {
    let lc = curve(400, 0.7, 0.0, 0.1);
    let grid = Frequencies::uniform(0.01, 0.003, 1001).unwrap();
    let ls = LombScargle::new(&lc, Mean::Floating).unwrap();
    assert!(ls.threads().get() >= 1);
    let all = ls.power(&grid, Normalization::Standard);
    for threads in [1, 3, 2000] {
        let ls = ls.clone().with_threads(one(threads));
        assert_eq!(ls.threads().get(), threads);
        assert_eq!(ls.power(&grid, Normalization::Standard), all);
    }
    let (f, p) = all.best();
    assert!((f - 0.7).abs() < 0.002 && p > 0.5, "{f} {p}");
    assert_eq!(all.frequencies(), &grid);
}

#[test]
fn normalizations_agree() {
    let lc = curve(300, 1.3, 0.0, 0.2);
    let grid = Frequencies::from_per_day(&[0.4, 1.3, 2.0, 0.05]).unwrap();
    for mean in [Mean::Floating, Mean::Subtracted, Mean::Zero] {
        let ls = LombScargle::new(&lc, mean).unwrap();
        let get = |n| ls.power(&grid, n).power().to_vec();
        let standard = get(Normalization::Standard);
        let model = get(Normalization::Model);
        let log = get(Normalization::Log);
        let psd = get(Normalization::Psd);
        for k in 0..4 {
            let p = standard[k];
            assert!((0.0..=1.0).contains(&p));
            assert!((model[k] - p / (1.0 - p)).abs() < 1e-12 * model[k].max(1.0));
            assert!((log[k] + (-p).ln_1p()).abs() < 1e-12 * log[k].max(1.0));
            assert!((psd[k] - 0.5 * p * ls.chi2_ref()).abs() < 1e-12 * psd[k].max(1.0));
        }
    }
}

#[test]
fn psd_scales_with_the_errors() {
    let grid = Frequencies::uniform(0.1, 0.1, 30).unwrap();
    let one = LombScargle::new(&curve(200, 0.9, 0.0, 1.0), Mean::Floating).unwrap();
    let two = LombScargle::new(&curve(200, 0.9, 0.0, 2.0), Mean::Floating).unwrap();
    assert_eq!(two.chi2_ref(), one.chi2_ref() / 4.0);
    let (a, b) = (
        one.power(&grid, Normalization::Psd),
        two.power(&grid, Normalization::Psd),
    );
    for (x, y) in a.power().iter().zip(b.power()) {
        assert!((x / 4.0 - y).abs() <= 1e-15 * x, "{x} {y}");
    }
    // The other normalisations do not depend on the errors' scale.
    assert_eq!(
        one.power(&grid, Normalization::Standard),
        two.power(&grid, Normalization::Standard)
    );
}

#[test]
fn large_times_lose_nothing() {
    // The same data near 0 and shifted by 2^21 days (each time exact in
    // both: steps of 1/64 day): the powers agree to rounding. astropy's
    // slow method loses 1e-10 of the peak at such times.
    let time: Vec<f64> = (0..500_u32)
        .map(|i| f64::from(i * 9 + (i * i) % 5) / 64.0)
        .collect();
    let flux: Vec<f64> = (0..500_u32)
        .zip(&time)
        .map(|(i, t)| (TAU * 1.7 * t).sin() + 0.1 * f64::from((i * 7919) % 13))
        .collect();
    let shifted: Vec<f64> = time.iter().map(|t| t + 2_097_152.0).collect();
    assert!(
        shifted
            .iter()
            .zip(&time)
            .all(|(s, t)| s - 2_097_152.0 == *t)
    );
    let near = LightCurve::from_days(time, flux.clone(), vec![0.1; 500]).unwrap();
    let far = LightCurve::from_days(shifted, flux, vec![0.1; 500]).unwrap();
    let grid = Frequencies::uniform(0.05, 0.01, 400).unwrap();
    for mean in [Mean::Floating, Mean::Subtracted, Mean::Zero] {
        let near = LombScargle::new(&near, mean).unwrap();
        let far = LombScargle::new(&far, mean).unwrap();
        assert!(far.centre() > 2e6 && near.centre() < 100.0);
        let (a, b) = (
            near.power(&grid, Normalization::Standard),
            far.power(&grid, Normalization::Standard),
        );
        let worst = a
            .power()
            .iter()
            .zip(b.power())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-15, "{mean:?} {worst:e}");
    }
}

#[test]
fn best_is_the_first_of_a_tie() {
    let lc = curve(100, 0.5, 0.0, 0.1);
    let ls = LombScargle::new(&lc, Mean::Floating).unwrap();
    let p = ls.power(
        &Frequencies::from_per_day(&[0.5, 0.2, 0.5]).unwrap(),
        Normalization::Log,
    );
    assert_eq!(p.best(), (0.5, p.power()[0]));
    assert_eq!(p.power()[0], p.power()[2]);
    assert_eq!(p.normalization(), Normalization::Log);
    assert_eq!(ls.mean(), Mean::Floating);
}
