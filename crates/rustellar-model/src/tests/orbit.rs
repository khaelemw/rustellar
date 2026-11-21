// Kepler's equation over the whole range of eccentricities and mean
// anomalies, and the sine and cosine of exact multiples of 90 degrees.

use crate::params::{kepler, sin_cos_deg};

#[test]
fn kepler_residual_is_rounding() {
    // A fixed, simple generator, so the test is the same every run.
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..200_000 {
        let e = 0.999 * next();
        let m = std::f64::consts::PI * (2.0 * next() - 1.0);
        let big_e = kepler(m, e);
        let residual = (-e).mul_add(big_e.sin(), big_e) - m;
        assert!(
            residual.abs() <= 4.0 * f64::EPSILON * m.abs().max(1.0),
            "{e} {m}"
        );
    }
    assert_eq!(kepler(0.0, 0.5), 0.0);
    assert_eq!(kepler(1.0, 0.0), 1.0);
    assert_eq!(kepler(std::f64::consts::PI, 0.9), std::f64::consts::PI);
}

#[test]
fn quarter_turns_are_exact() {
    assert_eq!(sin_cos_deg(0.0), (0.0, 1.0));
    assert_eq!(sin_cos_deg(90.0), (1.0, -0.0));
    assert_eq!(sin_cos_deg(180.0), (-0.0, -1.0));
    assert_eq!(sin_cos_deg(270.0), (-1.0, 0.0));
    assert_eq!(sin_cos_deg(-90.0), (-1.0, 0.0));
    for deg in [-359.0, -123.4, -45.0, 30.0, 45.0, 200.0, 721.5] {
        let (s, c) = sin_cos_deg(deg);
        let (ws, wc) = f64::to_radians(deg).sin_cos();
        assert!((s - ws).abs() < 1e-14 && (c - wc).abs() < 1e-14, "{deg}");
    }
}
