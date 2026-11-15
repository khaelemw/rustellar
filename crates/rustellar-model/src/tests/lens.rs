use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, PI};

use rstest::rstest;

use crate::lens::{Lens, arc_1, arc_2, sum3};

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    let error = ((actual - expected) / expected).abs();
    assert!(
        error <= tolerance,
        "{actual} vs {expected}: relative error {error:e}"
    );
}

// ==========================
// sum3
// ==========================

// Each sum is exactly a double, so sum3 must return it exactly, where the
// plain left-to-right sum can lose it entirely.
#[rstest]
#[case::small_after_large(1e16, 1.0, -1e16, 1.0)]
#[case::subnormal(5e-324, 1.0, -1.0, 5e-324)]
#[case::smallest_step_below_one(1.0, -1.0, f64::EPSILON / 2.0, f64::EPSILON / 2.0)]
// 0.1 + 0.9 - 1 is exactly 2^-55 in doubles.
#[case::decimal_inputs(0.1, -1.0, 0.9, 2f64.powi(-55))]
#[case::cancels_to_zero(0.75, -1.0, 0.25, 0.0)]
fn sum3_is_exact(#[case] a: f64, #[case] b: f64, #[case] c: f64, #[case] exact: f64) {
    assert_eq!(sum3(a, b, c), exact);
}

// ==========================
// Lens
// ==========================

// Two unit circles one radius apart cross at 60 degrees from either centre.
#[test]
fn equal_circles_one_radius_apart() {
    let lens = Lens::new(1.0, 1.0);
    assert_relative(lens.kappa0, FRAC_PI_3, 3e-16);
    assert_relative(lens.kappa1, FRAC_PI_3, 3e-16);
}

// With p = 3/4 and z = 5/4, the radii to each crossing point are at right
// angles (3-4-5), so cos kappa1 = 4/5 and cos kappa0 = 3/5.
#[test]
fn right_angle_crossing() {
    let lens = Lens::new(1.25, 0.75);
    assert_relative(lens.kappa1, 0.8_f64.acos(), 2e-16);
    assert_relative(lens.kappa0, 0.6_f64.acos(), 2e-16);
}

// A planet of radius 1 almost on the star's centre: cos kappa = z / 2 for
// both, so the circles cross nearly at right angles to the line of centres,
// however small z is, down to the smallest double.
#[rstest]
fn nearly_coincident_circles(#[values(1e-8, 1e-200, 5e-324)] z: f64) {
    let lens = Lens::new(z, 1.0);
    assert_relative(lens.kappa0, (0.5 * z).acos(), 3e-16);
    assert_relative(lens.kappa1, (0.5 * z).acos(), 3e-16);
    assert_relative(lens.kappa0, FRAC_PI_2, z);
}

// At first contact the lens closes from outside (both angles 0); at the inner
// contact line it closes on the star's arc for a small planet (kappa1 = 0,
// kappa0 = pi) and on the planet's for a large one (kappa0 = 0, kappa1 = pi).
#[rstest]
#[case::first_contact(1.5, 0.5, 0.0, 0.0)]
#[case::inner_small(0.5, 0.5, PI, 0.0)]
#[case::inner_large(1.0, 2.0, 0.0, PI)]
fn contact_lines(#[case] z: f64, #[case] p: f64, #[case] kappa0: f64, #[case] kappa1: f64) {
    let lens = Lens::new(z, p);
    assert_eq!(lens.kappa0, kappa0);
    assert_eq!(lens.kappa1, kappa1);
}

// ==========================
// arc_1 and arc_2
// ==========================

// (k, k - sin k, (3 k - 4 sin k + sin 2k / 2) / 4) from mpmath at 50 digits,
// each rounded once to the nearest double. 1 - 2^-53 and 1 sit on either side
// of the switch from series to closed form.
const ARCS: [(f64, f64, f64); 9] = [
    (
        1e-8,
        1.666_666_666_666_666_8e-25,
        2.500_000_000_000_000_3e-42,
    ),
    (1e-3, 1.666_666_583_333_335_5e-10, 2.499_999_702_380_97e-17),
    (
        0.25,
        0.002_596_040_745_477_070_2,
        2.423_307_100_244_543_6e-5,
    ),
    (0.5, 0.020_574_461_395_796_998, 0.000_758_334_496_784_063),
    (
        1.0 - f64::EPSILON / 2.0,
        0.158_529_015_192_103_44,
        0.022_191_193_545_313_695,
    ),
    (1.0, 0.158_529_015_192_103_5, 0.022_191_193_545_313_705),
    (1.5, 0.502_505_013_395_945_6, 0.145_145_014_403_428_97),
    (2.0, 1.090_702_573_174_318_3, 0.496_102_261_260_827_3),
    (PI, PI, 2.356_194_490_192_345),
];

#[test]
fn arcs_match_reference_values() {
    for (k, one, two) in ARCS {
        assert_relative(arc_1(k), one, 4e-16);
        assert_relative(arc_2(k), two, 8e-16);
    }
}

#[test]
fn arcs_are_zero_at_zero() {
    assert_eq!(arc_1(0.0), 0.0);
    assert_eq!(arc_2(0.0), 0.0);
}

// Far below 1 the leading terms are all there is, and they must not
// underflow into 0 early or lose their relative precision.
#[rstest]
fn arcs_keep_relative_precision_when_tiny(#[values(1e-20, 1e-60)] k: f64) {
    assert_relative(arc_1(k), k * k * k / 6.0, 4e-16);
    assert_relative(arc_2(k), k.powi(5) / 40.0, 4e-16);
}

// Both are integrals of non-negative functions, so they increase with k.
#[test]
fn arcs_increase() {
    let mut previous = (0.0, 0.0);
    for i in 1..=4000 {
        let k = PI * f64::from(i) / 4000.0;
        let now = (arc_1(k), arc_2(k));
        assert!(now.0 > previous.0 && now.1 > previous.1, "k = {k}");
        previous = now;
    }
}
