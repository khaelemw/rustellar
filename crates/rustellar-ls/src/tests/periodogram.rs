use rustellar_core::LightCurve;

use crate::periodogram::{Buffer, has_avx2, solve};
use crate::{LombScargle, Mean};

#[test]
fn solve_full_rank() {
    // [[2, 1], [1, 3]] x = [3, 5]: x = (4 / 5, 7 / 5).
    let (b, g) = solve(2.0, 1.0, 3.0, 3.0, 5.0, 2.0);
    assert!((b - 0.8).abs() < 1e-15 && (g - 1.4).abs() < 1e-15);
    // A term far smaller than the other but uncorrelated is still fitted.
    let (b, g) = solve(1.0, 0.0, 1e-40, 1.0, 1e-20, 2.0);
    assert!((b - 1.0).abs() < 1e-15 && (g - 1e20).abs() < 1e5);
}

#[test]
fn solve_rank_one_and_zero() {
    // 5 e e^T with e = (2, 1) / sqrt 5, and with e = (1, 2) / sqrt 5: the
    // solution is e (e . rhs) / 5.
    for (a, b, d, e) in [(4.0, 2.0, 1.0, (2.0, 1.0)), (1.0, 2.0, 4.0, (1.0, 2.0))] {
        let (u, v) = (2.0, 1.0);
        let (x, y) = solve(a, b, d, u, v, 10.0);
        let along = (e.0 * u + e.1 * v) / 25.0;
        assert!((x - along * e.0).abs() < 1e-15 && (y - along * e.1).abs() < 1e-15);
    }
    assert_eq!(solve(0.0, 0.0, 0.0, 1.0, 1.0, 3.0), (0.0, 0.0));
}

#[test]
fn vectors_change_no_bit() {
    let time: Vec<f64> = (0..203_u32)
        .map(|i| 0.37 * f64::from(i) + 0.01 * f64::from(i % 3))
        .collect();
    let flux: Vec<f64> = (0..203_u32).map(|i| f64::from((i * 31) % 17)).collect();
    let lc = LightCurve::from_days(time, flux, vec![0.5; 203]).unwrap();
    for mean in [Mean::Floating, Mean::Zero] {
        let ls = LombScargle::new(&lc, mean).unwrap();
        let mut buffer = Buffer::new(lc.len());
        for f in [0.001, 0.3, 1.351_351_351_351_351_3, 7.0] {
            let plain = ls.fit(f, true, false, &mut buffer);
            let wide = ls.fit(f, true, has_avx2(), &mut buffer);
            assert_eq!(plain.reduction.to_bits(), wide.reduction.to_bits(), "{f}");
            assert_eq!(plain.chi2.to_bits(), wide.chi2.to_bits(), "{f}");
        }
    }
}
