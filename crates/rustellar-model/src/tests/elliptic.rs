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
    assert_relative(
        ellip_e(ellip_k(0.5), 0.5, 0.5),
        1.350_643_881_047_675_5,
        1e-14,
    );
    assert_relative(
        ellip_pi(ellip_k(0.5), 0.3, 0.5),
        2.250_376_821_943_946_4,
        1e-14,
    );
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
    assert_relative(ellip_e(ellip_k(1.0), 0.0, 1.0), FRAC_PI_2, 1e-15);
    // Absolute tolerance: for large negative n, Pi is small and its error
    // scales with K = pi / 2, not with Pi.
    for n in [-1e6, -3.0, -0.5, 0.0, 0.5, 0.9] {
        let error = (ellip_pi(ellip_k(1.0), n, 1.0) - FRAC_PI_2 / (1.0 - n).sqrt()).abs();
        assert!(error < 1e-15, "n {n}: error {error:e}");
    }
}

#[rstest]
#[case(1e-10)]
#[case(0.5)]
#[case(1.0 - 1e-10)]
fn pi_with_n_zero_is_k(#[case] m: f64) {
    assert_relative(
        ellip_pi(ellip_k(1.0 - m), 0.0, 1.0 - m),
        ellip_k(1.0 - m),
        1e-15,
    );
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
    let (e, ec) = (ellip_e(ellip_k(mc), m, mc), ellip_e(ellip_k(m), mc, m));
    assert_relative(e * kc + ec * k - k * kc, FRAC_PI_2, 1e-14);
}

// The transit formulas take m close to 1 and n far below 0.
#[test]
fn finite_and_positive_at_transit_extremes() {
    for mc in [1e-12, 1e-6, 0.5, 1.0] {
        let m = 1.0 - mc;
        assert!(ellip_k(mc).is_finite() && ellip_k(mc) >= FRAC_PI_2);
        let e = ellip_e(ellip_k(mc), m, mc);
        assert!(e > 1.0 - 1e-15 && e < FRAC_PI_2 + 1e-15, "mc {mc}: E {e}");
        for n in [-1e6, -1.0, -1e-6] {
            let pi = ellip_pi(ellip_k(mc), n, mc);
            assert!(
                pi.is_finite() && pi > 0.0 && pi < ellip_k(mc),
                "n {n}, m {m}"
            );
        }
    }
}

// Reference values over the parameter range the transit model uses, and
// beyond it (n > 0). scipy's integrals are independent implementations;
// `m = 1 - mc` is computed the same way here and in the generator.
// (mc, K, E) from scipy.special.ellipkm1 and ellipe
const K_E_TABLE: [(f64, f64, f64); 7] = [
    (1.0, FRAC_PI_2, FRAC_PI_2),
    (0.999999999999, 1.5707963267952894, 1.5707963267945038),
    (0.9, 1.6124413487202192, 1.5307576368977633),
    (0.5, 1.8540746773013719, 1.3506438810476755),
    (0.1, 2.5780921133481733, 1.1047747327040733),
    (1e-06, 8.294051463615439, 1.0000038970261722),
    (1e-12, 15.201804919087715, 1.0000000000073508),
];
// (n, mc, Pi) from scipy.special.elliprf and elliprj
const PI_TABLE: [(f64, f64, f64); 42] = [
    (-1000000.0, 1.0, 0.0015707955413981622),
    (-100.0, 1.0, 0.15630007634061638),
    (-1.0, 1.0, 1.1107207345395915),
    (-1e-06, 1.0, 1.5707955413973222),
    (0.3, 1.0, 1.877460709222638),
    (0.9, 1.0, 4.967294132898052),
    (-1000000.0, 0.999999999999, 0.0015707955413961638),
    (-100.0, 0.999999999999, 0.1563000763406237),
    (-1.0, 0.999999999999, 1.1107207345398216),
    (-1e-06, 0.999999999999, 1.570795541397715),
    (0.3, 0.999999999999, 1.8774607092231492),
    (0.9, 0.999999999999, 4.967294132899938),
    (-1000000.0, 0.9, 0.001570877146563232),
    (-100.0, 0.9, 0.1570382071634968),
    (-1.0, 0.9, 1.1350336427288368),
    (-1e-06, 0.9, 1.6124405318837165),
    (0.3, 0.9, 1.9317578940285607),
    (0.9, 0.9, 5.169473402157696),
    (-1000000.0, 0.5, 0.0015712985793014944),
    (-100.0, 0.5, 0.1609257334226124),
    (-1.0, 0.5, 1.2731273667496823),
    (-1e-06, 0.5, 1.8540736704405567),
    (0.3, 0.5, 2.2503768219439464),
    (0.9, 0.5, 6.425573644195659),
    (-1000000.0, 0.1, 0.0015722681508281156),
    (-100.0, 0.1, 0.1702261930661333),
    (-1.0, 0.1, 1.664766936048353),
    (-1e-06, 0.1, 2.5780904763302104),
    (0.3, 0.1, 3.234766174311374),
    (0.9, 0.1, 11.047747327040735),
    (-1000000.0, 1e-06, 0.0015780887966112545),
    (-100.0, 1e-06, 0.2277754893172208),
    (-1.0, 1e-06, 4.539723937424347),
    (-1e-06, 1e-06, 8.29404416956754),
    (0.3, 1e-06, 11.3673381364981),
    (0.9, 1e-06, 65.68948127125059),
    (-1000000.0, 1e-12, 0.0015849965470557947),
    (-100.0, 1e-12, 0.29616912536750917),
    (-1.0, 1e-12, 7.993601541240843),
    (-1e-06, 1e-12, 15.201790717296666),
    (0.3, 1e-12, 21.235555316460992),
    (0.9, 1e-12, 134.76675134872696),
];

#[test]
fn k_and_e_match_scipy_over_parameter_range() {
    for (mc, k, e) in K_E_TABLE {
        let m = 1.0 - mc;
        assert_relative(ellip_k(mc), k, 1e-14);
        assert_relative(ellip_e(ellip_k(mc), m, mc), e, 1e-14);
    }
}

// For large negative n, Pi is K minus a term of nearly the same size (see
// `ellip_pi`), so its error scales with K rather than with Pi; elsewhere it is
// relative to Pi.
#[test]
fn pi_matches_scipy_over_parameter_range() {
    for (n, mc, pi) in PI_TABLE {
        let k = ellip_k(mc);
        let error = (ellip_pi(k, n, mc) - pi).abs();
        let tolerance = (1e-15 * k).max(1e-14 * pi.abs());
        assert!(error < tolerance, "n {n}, mc {mc}: error {error:e}");
    }
}
