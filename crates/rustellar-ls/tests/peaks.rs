// Peaks and the fitted sinusoid: the API, every mean and normalisation,
// the grid's ends, and degenerate frequencies.

use std::f64::consts::TAU;

use rustellar_ls::{Frequencies, LightCurve, LombScargle, LsError, Mean, Normalization};

/// Two sinusoids (0.37 and 1.9 cycles per day) on uneven times, errors
/// varying.
fn curve() -> LightCurve {
    let time: Vec<f64> = (0..400)
        .map(|i| 0.11 * f64::from(i) + 0.03 * f64::from((i * i) % 7))
        .collect();
    let flux = time
        .iter()
        .enumerate()
        .map(|(i, t)| {
            #[expect(clippy::cast_precision_loss, reason = "small")]
            let noise = 0.05 * ((i * 7919) % 13) as f64 / 13.0;
            2.0 + 0.8 * (TAU * 0.37 * t + 0.3).cos() + 0.2 * (TAU * 1.9 * t).sin() + noise
        })
        .collect();
    let error = (0..400).map(|i| 0.05 + 0.01 * f64::from(i % 4)).collect();
    LightCurve::from_days(time, flux, error).unwrap()
}

#[test]
fn model_needs_a_positive_frequency() {
    let ls = LombScargle::new(&curve(), Mean::Floating).unwrap();
    assert_eq!(
        ls.model(f64::NAN).unwrap_err(),
        LsError::NonFinite { name: "frequency" }
    );
    assert_eq!(
        ls.model(0.0).unwrap_err(),
        LsError::NotPositive { name: "frequency" }
    );
}

#[test]
fn model_for_each_mean() {
    let lc = curve();
    let weights: Vec<f64> = lc.flux_err().iter().map(|e| 1.0 / (e * e)).collect();
    let total: f64 = weights.iter().sum();
    let mean = weights
        .iter()
        .zip(lc.flux())
        .map(|(w, y)| w * y)
        .sum::<f64>()
        / total;
    for m in [Mean::Floating, Mean::Subtracted, Mean::Zero] {
        let ls = LombScargle::new(&lc, m).unwrap();
        let model = ls.model(0.37).unwrap();
        assert_eq!(model.frequency(), 0.37);
        assert_eq!(model.reference(), ls.centre());
        assert_eq!(model.amplitude(), model.cosine().hypot(model.sine()));
        assert_eq!(model.phase(), model.sine().atan2(model.cosine()));
        assert!(model.amplitude_error() > 0.0 && model.amplitude_error() < 0.01);
        assert!(model.phase_error() > 0.0 && model.phase_error() < 0.01);
        // The fit's chi-squared left is the periodogram's.
        let chi2: f64 = lc
            .time()
            .iter()
            .zip(lc.flux())
            .zip(&weights)
            .map(|((t, y), w)| w * (y - model.value(t.as_days())).powi(2))
            .sum();
        let grid = Frequencies::from_per_day(&[0.37]).unwrap();
        let reduction = ls.power(&grid, Normalization::Psd).power()[0] * 2.0;
        assert!((ls.chi2_ref() - reduction - chi2).abs() <= 1e-9 * ls.chi2_ref());
        match m {
            Mean::Floating => {
                assert!((model.offset() - 2.025).abs() < 0.01);
                assert!(model.offset_error() > (1.0 / total).sqrt());
            }
            Mean::Subtracted => {
                assert!((model.offset() - mean).abs() <= 1e-14 * mean);
                assert!(
                    ((model.offset_error() - (1.0 / total).sqrt()) / model.offset_error()).abs()
                        < 1e-14
                );
            }
            _ => {
                assert_eq!(model.offset(), 0.0);
                assert_eq!(model.offset_error(), 0.0);
            }
        }
    }
}

#[test]
fn degenerate_model_has_no_errors() {
    // Even sampling at its Nyquist frequency: the sine is zero at every
    // point, so only the cosine is fitted.
    let time: Vec<f64> = (0..64).map(|i| 0.5 * f64::from(i)).collect();
    let flux = (0..64)
        .map(|i| 1.0 + 0.1 * f64::from(i % 2) + 0.01 * f64::from(i % 5))
        .collect();
    let lc = LightCurve::from_days(time, flux, vec![0.01; 64]).unwrap();
    let model = LombScargle::new(&lc, Mean::Floating)
        .unwrap()
        .model(1.0)
        .unwrap();
    assert!(model.amplitude() > 0.04);
    assert_eq!(model.amplitude_error(), f64::INFINITY);
    assert_eq!(model.phase_error(), f64::INFINITY);
    assert_eq!(model.offset_error(), f64::INFINITY);
}

#[test]
fn peaks_in_every_normalization() {
    let ls = LombScargle::new(&curve(), Mean::Floating).unwrap();
    let grid = Frequencies::uniform(0.05, 0.004, 600).unwrap();
    for n in [
        Normalization::Standard,
        Normalization::Model,
        Normalization::Log,
        Normalization::Psd,
    ] {
        let p = ls.power(&grid, n);
        let peaks = ls.peaks(&p, 2);
        assert_eq!(peaks.len(), 2);
        assert!((peaks[0].frequency() - 0.37).abs() < 2e-3, "{n:?}");
        assert!((peaks[1].frequency() - 1.9).abs() < 2e-3, "{n:?}");
        assert_eq!(peaks[0].period(), 1.0 / peaks[0].frequency());
        for peak in &peaks {
            // Its power is the periodogram's there, and above the grid's.
            let there = Frequencies::from_per_day(&[peak.frequency()]).unwrap();
            assert_eq!(peak.power(), ls.power(&there, n).power()[0]);
            assert!(peak.power() >= p.best().1 || peak.frequency() > 1.0);
        }
    }
    // None asked, and every peak when asked for more.
    let p = ls.power(&grid, Normalization::Standard);
    assert!(ls.peaks(&p, 0).is_empty());
    let all = ls.peaks(&p, 10_000);
    assert!(all.len() > 2 && all.len() < 600);
    assert!(all.windows(2).all(|w| w[0].power() >= w[1].power()));
}

#[test]
fn peaks_at_the_grid_ends_stay_there() {
    let ls = LombScargle::new(&curve(), Mean::Floating).unwrap();
    // Rising to the end, below 0.37: the peak is the last frequency.
    let below = Frequencies::uniform(0.360, 0.001, 8).unwrap();
    let p = ls.power(&below, Normalization::Standard);
    let peak = ls.peaks(&p, 1)[0];
    assert_eq!(peak.frequency(), below.get(7));
    assert_eq!(peak.power(), p.power()[7]);
    // Falling from the start, above 0.37: the first.
    let above = Frequencies::uniform(0.373, 0.001, 5).unwrap();
    let p = ls.power(&above, Normalization::Standard);
    assert_eq!(ls.peaks(&p, 1)[0].frequency(), 0.373);
    // One frequency.
    let one = Frequencies::from_per_day(&[0.36]).unwrap();
    let p = ls.power(&one, Normalization::Standard);
    assert_eq!(ls.peaks(&p, 3)[0].frequency(), 0.36);
}
