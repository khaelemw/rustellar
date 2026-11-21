// The fast periodogram's API: its errors, every mean and normalisation
// against the exact periodogram, and the frequencies it computes exactly.

use std::f64::consts::TAU;
use std::num::NonZeroUsize;

use rustellar_ls::{Frequencies, LightCurve, LombScargle, LsError, Mean, Normalization};

/// `n` uneven times with a sinusoid of 1.3 cycles per day and a wiggle.
fn curve(n: u32) -> LightCurve {
    let time: Vec<f64> = (0..n)
        .map(|i| 0.07 * f64::from(i) + 0.01 * f64::from((i * i) % 7))
        .collect();
    let flux = (0..n)
        .zip(&time)
        .map(|(i, t)| {
            let noise = 0.4 * f64::from((i * 7919) % 17) / 17.0 - 0.2;
            2.0 + (TAU * 1.3 * t).sin() + noise
        })
        .collect();
    let error = (0..n).map(|i| 0.1 + 0.01 * f64::from(i % 3)).collect();
    LightCurve::from_days(time, flux, error).unwrap()
}

const NORMALIZATIONS: [Normalization; 4] = [
    Normalization::Standard,
    Normalization::Model,
    Normalization::Log,
    Normalization::Psd,
];

/// The largest difference between the fast and exact powers.
fn worst(ls: &LombScargle, grid: &Frequencies, n: Normalization, tolerance: f64) -> f64 {
    let fast = ls.power_fast(grid, n, tolerance).unwrap();
    let exact = ls.power(grid, n);
    assert_eq!(fast.frequencies(), exact.frequencies());
    assert_eq!(fast.normalization(), n);
    fast.power()
        .iter()
        .zip(exact.power())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn bad_tolerances_and_listed_grids() {
    let ls = LombScargle::new(&curve(50), Mean::Floating).unwrap();
    let grid = Frequencies::uniform(0.1, 0.01, 100).unwrap();
    let s = Normalization::Standard;
    for (tolerance, name) in [(f64::NAN, "nan"), (f64::INFINITY, "inf")] {
        assert_eq!(
            ls.power_fast(&grid, s, tolerance).unwrap_err(),
            LsError::NonFinite { name: "tolerance" },
            "{name}"
        );
    }
    for tolerance in [0.0, -1e-6] {
        assert_eq!(
            ls.power_fast(&grid, s, tolerance).unwrap_err(),
            LsError::NotPositive { name: "tolerance" }
        );
    }
    let listed = Frequencies::from_per_day(&[0.1, 0.2, 0.4]).unwrap();
    let error = ls.power_fast(&listed, s, 1e-6).unwrap_err();
    assert_eq!(error, LsError::NotUniform);
    assert_eq!(
        error.to_string(),
        "the fast method needs an even grid of frequencies"
    );
}

#[test]
fn every_mean_and_normalization_within_tolerance() {
    let lc = curve(700);
    let grid = Frequencies::uniform(0.05, 0.0037, 3001).unwrap();
    for mean in [Mean::Floating, Mean::Subtracted, Mean::Zero] {
        let ls = LombScargle::new(&lc, mean).unwrap();
        for tolerance in [1e-3, 1e-7, 1e-11] {
            // Standard within the tolerance; the others scaled by theirs.
            assert!(
                worst(&ls, &grid, Normalization::Standard, tolerance) <= tolerance,
                "{mean:?} {tolerance}"
            );
            for n in NORMALIZATIONS {
                let exact = ls.power(&grid, n);
                let top = exact.power().iter().copied().fold(0.0, f64::max);
                assert!(
                    worst(&ls, &grid, n, tolerance) <= 1e3 * tolerance * top.max(1.0),
                    "{mean:?} {n:?} {tolerance}"
                );
            }
        }
    }
}

#[test]
fn degenerate_and_strict_frequencies_computed_exactly() {
    let ls = LombScargle::new(&curve(300), Mean::Floating).unwrap();
    // The lowest frequencies, where the fit's terms nearly coincide.
    let grid = Frequencies::uniform(1e-9, 1e-4, 200).unwrap();
    assert!(worst(&ls, &grid, Normalization::Standard, 1e-9) <= 1e-9);
    // A tolerance below the transforms' reach: every power exact.
    let grid = Frequencies::uniform(0.2, 0.01, 300).unwrap();
    let fast = ls
        .power_fast(&grid, Normalization::Standard, 1e-17)
        .unwrap();
    let exact = ls.power(&grid, Normalization::Standard);
    assert_eq!(fast.power(), exact.power());
}

#[test]
fn single_frequency_and_any_threads() {
    let ls = LombScargle::new(&curve(200), Mean::Floating).unwrap();
    let one = Frequencies::uniform(1.3, 0.01, 1).unwrap();
    assert!(worst(&ls, &one, Normalization::Standard, 1e-8) <= 1e-8);
    let grid = Frequencies::uniform(0.01, 0.002, 2000).unwrap();
    let all = ls.power_fast(&grid, Normalization::Standard, 1e-8).unwrap();
    for threads in [1, 3, 64] {
        let ls = ls.clone().with_threads(NonZeroUsize::new(threads).unwrap());
        let some = ls.power_fast(&grid, Normalization::Standard, 1e-8).unwrap();
        assert_eq!(some.power(), all.power(), "{threads}");
    }
}
