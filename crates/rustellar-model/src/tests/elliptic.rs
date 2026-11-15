use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use rstest::rstest;

use crate::elliptic::cel_e_d;

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    let error = ((actual - expected) / expected).abs();
    assert!(
        error < tolerance,
        "{actual} vs {expected}: relative error {error:e}"
    );
}

// The Legendre integrals as special cases of cel, from the complementary
// modulus kc = sqrt(1 - m).
fn ellip_k(kc: f64) -> f64 {
    cel_e_d(kc, 1.0, 1.0, 1.0).0
}

fn ellip_e(kc: f64) -> f64 {
    cel_e_d(kc, 1.0, 1.0, 1.0).1
}

fn ellip_d(kc: f64) -> f64 {
    cel_e_d(kc, 1.0, 1.0, 1.0).2
}

// Pi(n | m) = cel(kc, 1 - n, 1, 1), for n < 1.
fn ellip_pi(n: f64, kc: f64) -> f64 {
    let sqrt_p = (1.0 - n).sqrt();
    cel_e_d(kc, sqrt_p, 1.0, 1.0 / sqrt_p).0
}

// Reference values from mpmath at 60 digits.
#[test]
fn legendre_reference_values() {
    let kc = 0.5_f64.sqrt();
    assert_relative(ellip_k(kc), 1.854_074_677_301_372, 4e-16);
    assert_relative(ellip_e(kc), 1.350_643_881_047_675_5, 4e-16);
    assert_relative(ellip_pi(0.3, kc), 2.250_376_821_943_946_4, 4e-16);
}

// At kc = 1 (m = 0) every integral has a closed form.
#[test]
fn integrals_at_m_zero() {
    assert_relative(ellip_k(1.0), FRAC_PI_2, 1e-16);
    assert_relative(ellip_e(1.0), FRAC_PI_2, 1e-16);
    assert_relative(ellip_d(1.0), FRAC_PI_4, 1e-16);
    for n in [-1e12, -1e6, -3.0, -0.5, 0.0, 0.5, 0.9, 0.999_999] {
        assert_relative(ellip_pi(n, 1.0), FRAC_PI_2 / (1.0 - n).sqrt(), 4e-16);
    }
}

#[rstest]
#[case(1e-10)]
#[case(0.5)]
#[case(1.0 - 1e-10)]
fn pi_with_n_zero_is_k(#[case] m: f64) {
    let kc = (1.0 - m).sqrt();
    assert_relative(ellip_pi(0.0, kc), ellip_k(kc), 4e-16);
}

// D = (E - kc^2 K) / k^2, away from k = 0 where that form cancels.
#[rstest]
#[case(0.01)]
#[case(0.3)]
#[case(0.5)]
#[case(0.9)]
fn d_is_e_minus_kc2_k_over_k2(#[case] kc: f64) {
    let k2 = (1.0 - kc) * (1.0 + kc);
    assert_relative(
        ellip_d(kc),
        (ellip_e(kc) - kc * kc * ellip_k(kc)) / k2,
        2e-15,
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
    let (kc, kc_prime) = ((1.0 - m).sqrt(), m.sqrt());
    let (k, k_prime) = (ellip_k(kc), ellip_k(kc_prime));
    let (e, e_prime) = (ellip_e(kc), ellip_e(kc_prime));
    assert_relative(e * k_prime + e_prime * k - k * k_prime, FRAC_PI_2, 2e-15);
}

// cel is linear in a and b.
#[test]
fn linear_in_a_and_b() {
    let (kc, sqrt_p) = (0.37, 0.81);
    let cel = |a: f64, b: f64| cel_e_d(kc, sqrt_p, a, b / sqrt_p).0;
    let (a, b) = (1.7, -0.6);
    assert_relative(cel(a, b), a * cel(1.0, 0.0) + b * cel(0.0, 1.0), 1e-15);
}

// As kc goes to 0, K grows like ln(4 / kc) and E and D tend to 1, with
// corrections of order kc^2 ln(kc). The transit formulas reach kc near 1e-10
// next to second and third contact.
#[rstest]
#[case(1e-10)]
#[case(1e-50)]
#[case(1e-300)]
fn finite_as_kc_goes_to_zero(#[case] kc: f64) {
    assert_relative(ellip_k(kc), (4.0 / kc).ln(), 1e-15);
    assert_relative(ellip_e(kc), 1.0, 1e-15);
    assert_relative(ellip_d(kc), 1.0, 1e-15);
}

// As p goes to 0 with b / sqrt(p) = c fixed, cel(kc, p, 0, b) tends to
// c pi / (2 kc): the limit the transit formulas reach next to the line z = p,
// where they pass sqrt(p) down to 1e-100 of the scale.
#[rstest]
#[case(0.2)]
#[case(0.9)]
fn general_integral_as_p_goes_to_zero(#[case] kc: f64) {
    let c = 1.3;
    for sqrt_p in [1e-20, 1e-50, 1e-100] {
        let (general, _, _) = cel_e_d(kc, sqrt_p, 0.0, c);
        assert_relative(general, c * PI / (2.0 * kc), 1e-15);
    }
}

// Reference values over the parameter range the transit model uses, from
// mpmath's ellipk, ellipe and ellippi at 60 digits, each rounded once to the
// nearest double. (scipy's elliprj, used here before, is off by up to 3e-12
// for large negative n.)
// (mc, K, E), with m = 1 - mc
const K_E_TABLE: [(f64, f64, f64); 7] = [
    (1.0, FRAC_PI_2, FRAC_PI_2),
    (0.999999999999, 1.5707963267952894, 1.570796326794504),
    (0.9, 1.6124413487202194, 1.5307576368977631),
    (0.5, 1.8540746773013719, 1.3506438810476755),
    (0.1, 2.5780921133481733, 1.1047747327040733),
    (1e-06, 8.29405146361544, 1.000003897026172),
    (1e-12, 15.201804919087715, 1.000000000007351),
];
// (n, mc, Pi(n | 1 - mc))
const PI_TABLE: [(f64, f64, f64); 42] = [
    (-1000000.0, 1.0, 0.0015707955413973222),
    (-100.0, 1.0, 0.15630007634061657),
    (-1.0, 1.0, 1.1107207345395915),
    (-1e-06, 1.0, 1.5707955413973222),
    (0.3, 1.0, 1.877460709222638),
    (0.9, 1.0, 4.967294132898051),
    (-1000000.0, 0.999999999999, 0.001570795541397323),
    (-100.0, 0.999999999999, 0.15630007634062365),
    (-1.0, 0.999999999999, 1.1107207345398216),
    (-1e-06, 0.999999999999, 1.570795541397715),
    (0.3, 0.999999999999, 1.8774607092231492),
    (0.9, 0.999999999999, 4.967294132899937),
    (-1000000.0, 0.9, 0.0015708771465632201),
    (-100.0, 0.9, 0.15703820716349703),
    (-1.0, 0.9, 1.1350336427288368),
    (-1e-06, 0.9, 1.6124405318837165),
    (0.3, 0.9, 1.9317578940285607),
    (0.9, 0.9, 5.1694734021576965),
    (-1000000.0, 0.5, 0.0015712985793004195),
    (-100.0, 0.5, 0.16092573342261243),
    (-1.0, 0.5, 1.2731273667496825),
    (-1e-06, 0.5, 1.8540736704405572),
    (0.3, 0.5, 2.250376821943947),
    (0.9, 0.5, 6.425573644195659),
    (-1000000.0, 0.1, 0.0015722681508276764),
    (-100.0, 0.1, 0.1702261930661335),
    (-1.0, 0.1, 1.6647669360483535),
    (-1e-06, 0.1, 2.5780904763302104),
    (0.3, 0.1, 3.2347661743113743),
    (0.9, 0.1, 11.047747327040733),
    (-1000000.0, 1e-06, 0.0015780887966068112),
    (-100.0, 1e-06, 0.2277754893172189),
    (-1.0, 1e-06, 4.5397239374243465),
    (-1e-06, 1e-06, 8.29404416956754),
    (0.3, 1e-06, 11.367338136498098),
    (0.9, 1e-06, 65.68948127125057),
    (-1000000.0, 1e-12, 0.0015849965470507632),
    (-100.0, 1e-12, 0.2961691253675035),
    (-1.0, 1e-12, 7.993601541240842),
    (-1e-06, 1e-12, 15.201790717296664),
    (0.3, 1e-12, 21.235555316460992),
    (0.9, 1e-12, 134.7667513487269),
];

#[test]
fn k_and_e_match_reference_values_over_parameter_range() {
    for (mc, k, e) in K_E_TABLE {
        let kc = mc.sqrt();
        assert_relative(ellip_k(kc), k, 1e-15);
        assert_relative(ellip_e(kc), e, 1e-15);
    }
}

// cel gives Pi directly, with no cancellation against K, so its error is
// relative to Pi even for large negative n.
#[test]
fn pi_matches_reference_values_over_parameter_range() {
    for (n, mc, pi) in PI_TABLE {
        assert_relative(ellip_pi(n, mc.sqrt()), pi, 2e-15);
    }
}
