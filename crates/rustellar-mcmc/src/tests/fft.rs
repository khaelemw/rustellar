// The FFT's autocovariance sums against the direct sums, at lengths that
// are and are not powers of two.

use crate::fft::autocovariance_sums;

#[test]
fn against_direct_sums() {
    assert!(autocovariance_sums(&[]).is_empty());
    for n in [1usize, 2, 3, 5, 8, 64, 100, 257] {
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7919 % 101) as f64).sin() + 0.01 * i as f64)
            .collect();
        let mean = x.iter().sum::<f64>() / n as f64;
        let got = autocovariance_sums(&x);
        assert_eq!(got.len(), n);
        let scale = x
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            .max(1e-300);
        for (t, g) in got.iter().enumerate() {
            let want: f64 = (0..n - t).map(|i| (x[i] - mean) * (x[i + t] - mean)).sum();
            assert!(
                (g - want).abs() <= 1e-14 * scale,
                "n {n} lag {t}: {g} vs {want}"
            );
        }
    }
}
