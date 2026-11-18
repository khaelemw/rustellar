// The spectra against transitleastsquares 1.32's on fixed chi-squared
// arrays, the running median against a sort, and the period uncertainty,
// including where TLS reads past the start of its array.

use super::Random;
use crate::spectra::{period_uncertainty, running_median, spectra};

/// Integers from 1000 to 1099.9 in an order without pattern, with a dip
/// at 150 (or the middle, for short arrays): the same in TLS's test.
fn chi2(n: usize) -> Vec<f64> {
    let mut c: Vec<f64> = (0..n)
        .map(|k| 1000.0 + ((k * 7919) % 1000) as f64 / 10.0)
        .collect();
    if n > 150 {
        (c[149], c[150], c[151]) = (850.0, 700.0, 850.0);
    } else {
        c[n / 2] = 700.0;
    }
    c
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

// transitleastsquares 1.32's `spectra` and `period_uncertainty` (periods
// 1 + 0.01 k): n, oversampling, SDE_raw, SDE, the peak's index, power at
// the first, peak and last periods, power_raw at the peak, uncertainty.
type Case = (usize, usize, f64, f64, usize, f64, f64, f64, f64, f64);
const TLS: [Case; 3] = [
    (
        400,
        3,
        12.237_310_799_128_231,
        11.997_615_551_176_867,
        150,
        1.177_309_022_238_828_7,
        11.997_615_551_176_867,
        -0.451_731_061_931_935_74,
        12.237_310_799_128_233,
        0.009_999_999_999_999_787,
    ),
    // Too short for the running median: the power is the raw power.
    (
        100,
        3,
        8.695_471_803_414_415,
        8.695_471_803_414_415,
        50,
        0.756_948_773_003_830_1,
        8.695_471_803_414_415,
        -0.897_843_979_624_327_3,
        8.695_471_803_414_415,
        0.010_000_000_000_000_009,
    ),
    (
        400,
        1,
        12.237_310_799_128_231,
        12.930_889_999_532_702,
        150,
        1.410_099_193_026_056_7,
        12.930_889_999_532_702,
        -0.414_757_148_790_423_9,
        12.237_310_799_128_233,
        0.009_999_999_999_999_787,
    ),
];

#[test]
fn tls_spectra_to_rounding() {
    for (n, over, sde_raw, sde, peak, first, top, last, raw_top, uncertainty) in TLS {
        let s = spectra(&chi2(n), over);
        assert!(
            close(s.sde_raw, sde_raw),
            "{n} {over}: {} {sde_raw}",
            s.sde_raw
        );
        assert!(close(s.sde, sde), "{n} {over}: {} {sde}", s.sde);
        assert!(
            close(s.power[0], first) && close(s.power[peak], top) && close(s.power[n - 1], last)
        );
        assert!(close(s.power_raw[peak], raw_top));
        assert_eq!(s.sr[peak], 1.0);
        let periods: Vec<f64> = (0..n).map(|k| 1.0 + 0.01 * k as f64).collect();
        assert!(close(period_uncertainty(&periods, &s.power), uncertainty));
    }
}

#[test]
fn the_spectras_shapes() {
    let c = chi2(400);
    let s = spectra(&c, 3);
    // SR in (0, 1], 1 only at the least chi-squared.
    assert!(s.sr.iter().all(|&v| v > 0.0 && v <= 1.0));
    assert_eq!(s.sr.iter().filter(|&&v| v == 1.0).count(), 1);
    // The peaks are SDE_raw and SDE.
    let max = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(close(max(&s.power_raw), s.sde_raw));
    assert!(close(max(&s.power), s.sde));
    // The power has mean 0.
    let mean = s.power.iter().sum::<f64>() / 400.0;
    assert!(mean.abs() < 1e-12, "{mean}");
    // Scaling every chi-squared changes nothing.
    let scaled: Vec<f64> = c.iter().map(|v| v * 1e-7).collect();
    let t = spectra(&scaled, 3);
    assert!(close(t.sde, s.sde) && close(t.sde_raw, s.sde_raw));
}

#[test]
fn the_running_median_is_a_sorted_windows_middle() {
    let mut random = Random::new(41);
    for _ in 0..200 {
        let n = 1 + (random.next_u64() % 300) as usize;
        let kernel = 1 + 2 * (random.next_u64() % (n as u64).div_ceil(2)) as usize;
        let values: Vec<f64> = (0..n).map(|_| (random.next_u64() % 50) as f64).collect();
        let got = running_median(&values, kernel);
        assert_eq!(got.len(), n);
        let medians: Vec<f64> = values
            .windows(kernel)
            .map(|w| {
                let mut w = w.to_vec();
                w.sort_by(f64::total_cmp);
                w[kernel / 2]
            })
            .collect();
        let front = (n - medians.len()) / 2;
        for (i, &m) in got.iter().enumerate() {
            let expected = medians[i.saturating_sub(front).min(medians.len() - 1)];
            assert_eq!(m, expected, "{n} {kernel} {i}");
        }
    }
}

#[test]
fn the_uncertainty_is_infinite_where_the_peak_never_halves() {
    let periods: Vec<f64> = (0..10).map(f64::from).collect();
    // TLS 1.32 gives -1.5 here: its walk down wraps to the array's end.
    let power = [5.0, 9.0, 8.0, 7.0, 6.0, 1.0, 1.0, 1.0, 1.0, 6.0];
    assert_eq!(period_uncertainty(&periods, &power), f64::INFINITY);
    // And at the last period, no upper side (TLS: infinite too).
    assert_eq!(
        period_uncertainty(&periods[..4], &[1.0, 2.0, 3.0, 9.0]),
        f64::INFINITY
    );
    // Halving on both sides: from the first at or below half on each side.
    let power = [0.0, 4.0, 4.5, 9.0, 4.6, 4.5, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(period_uncertainty(&periods, &power), 0.5 * (5.0 - 2.0));
    // Ties at the peak: the first.
    let power = [0.0, 9.0, 9.0, 0.0];
    assert_eq!(period_uncertainty(&periods[..4], &power), 0.5 * (3.0 - 0.0));
}
