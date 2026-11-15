use std::f64::consts::FRAC_PI_2;

use rstest::rstest;

use crate::elliptic::{carlson_rd, carlson_rf, carlson_rj, ellip_e, ellip_k, ellip_pi};

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    let error = ((actual - expected) / expected).abs();
    assert!(
        error < tolerance,
        "{actual} vs {expected}: relative error {error:e}"
    );
}

// Test values from Carlson (1995), table 1.
#[test]
fn carlson_reference_values() {
    assert_relative(carlson_rf(1.0, 2.0, 0.0), 1.311_028_777_146_1, 1e-13);
    assert_relative(carlson_rd(0.0, 2.0, 1.0), 1.797_210_352_103_4, 1e-13);
    assert_relative(carlson_rj(0.0, 1.0, 2.0, 3.0), 0.776_886_237_785_82, 1e-13);
}

// Reference values from scipy.special.
#[test]
fn legendre_reference_values() {
    assert_relative(ellip_k(0.5), 1.854_074_677_301_372, 1e-14);
    assert_relative(ellip_e(0.5, 0.5), 1.350_643_881_047_675_5, 1e-14);
    assert_relative(ellip_pi(0.3, 0.5), 2.250_376_821_943_946_4, 1e-14);
}

#[rstest]
#[case(1e-6)]
#[case(0.5)]
#[case(1.0)]
#[case(7.0)]
#[case(1e6)]
fn equal_arguments_have_closed_forms(#[case] x: f64) {
    assert_relative(carlson_rf(x, x, x), x.powf(-0.5), 1e-14);
    assert_relative(carlson_rd(x, x, x), x.powf(-1.5), 1e-14);
    assert_relative(carlson_rj(x, x, x, x), x.powf(-1.5), 1e-14);
}

#[test]
fn symmetric_in_the_documented_arguments() {
    let (x, y, z) = (0.3, 1.7, 4.2);
    let rf = carlson_rf(x, y, z);
    for (a, b, c) in [(x, z, y), (y, x, z), (y, z, x), (z, x, y), (z, y, x)] {
        assert_relative(carlson_rf(a, b, c), rf, 1e-14);
    }
    assert_relative(carlson_rd(y, x, z), carlson_rd(x, y, z), 1e-14);
    assert_relative(carlson_rj(z, x, y, 5.0), carlson_rj(x, y, z, 5.0), 1e-14);
}

#[test]
fn scaling_the_arguments_scales_the_integral() {
    let (x, y, z, p, s) = (0.3, 1.7, 4.2, 5.0, 9.0);
    assert_relative(
        carlson_rf(s * x, s * y, s * z),
        carlson_rf(x, y, z) / 3.0,
        1e-14,
    );
    assert_relative(
        carlson_rd(s * x, s * y, s * z),
        carlson_rd(x, y, z) / 27.0,
        1e-14,
    );
    assert_relative(
        carlson_rj(s * x, s * y, s * z, s * p),
        carlson_rj(x, y, z, p) / 27.0,
        1e-14,
    );
}

#[test]
fn legendre_integrals_at_m_zero() {
    assert_relative(ellip_k(1.0), FRAC_PI_2, 1e-15);
    assert_relative(ellip_e(0.0, 1.0), FRAC_PI_2, 1e-15);
    // Absolute tolerance: for large negative n, Pi is small and its error
    // scales with K = pi / 2, not with Pi.
    for n in [-1e6, -3.0, -0.5, 0.0, 0.5, 0.9] {
        let error = (ellip_pi(n, 1.0) - FRAC_PI_2 / (1.0 - n).sqrt()).abs();
        assert!(error < 1e-15, "n {n}: error {error:e}");
    }
}

#[rstest]
#[case(1e-10)]
#[case(0.5)]
#[case(1.0 - 1e-10)]
fn pi_with_n_zero_is_k(#[case] m: f64) {
    assert_relative(ellip_pi(0.0, 1.0 - m), ellip_k(1.0 - m), 1e-15);
}

// E(m) K(1 - m) + E(1 - m) K(m) - K(m) K(1 - m) = pi / 2.
#[rstest]
#[case(1e-10)]
#[case(0.1)]
#[case(0.5)]
#[case(0.9)]
#[case(1.0 - 1e-10)]
fn legendre_relation_holds(#[case] m: f64) {
    let mc = 1.0 - m;
    let (k, kc) = (ellip_k(mc), ellip_k(m));
    let (e, ec) = (ellip_e(m, mc), ellip_e(mc, m));
    assert_relative(e * kc + ec * k - k * kc, FRAC_PI_2, 1e-14);
}

// The transit formulas take m close to 1 and n far below 0.
#[test]
fn finite_and_positive_at_transit_extremes() {
    for mc in [1e-12, 1e-6, 0.5, 1.0] {
        let m = 1.0 - mc;
        assert!(ellip_k(mc).is_finite() && ellip_k(mc) >= FRAC_PI_2);
        let e = ellip_e(m, mc);
        assert!(e > 1.0 - 1e-15 && e < FRAC_PI_2 + 1e-15, "mc {mc}: E {e}");
        for n in [-1e6, -1.0, -1e-6] {
            let pi = ellip_pi(n, mc);
            assert!(
                pi.is_finite() && pi > 0.0 && pi < ellip_k(mc),
                "n {n}, m {m}"
            );
        }
    }
}
