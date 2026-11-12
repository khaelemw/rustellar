use std::num::NonZeroUsize;

use rstest::rstest;
use rustellar_model::{LimbDarkening, RadiusRatio, flux_ring_sum, flux_uniform};

// ==========================
// Helper Functions
// ==========================

fn rings(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn k(k: f64) -> RadiusRatio {
    RadiusRatio::new(k).unwrap()
}

// Quadratic coefficients typical of a Sun-like star in the optical.
fn sun_like() -> LimbDarkening {
    LimbDarkening::quadratic(0.3, 0.25).unwrap()
}

// ==========================
// Tests
// ==========================

// Expected values were computed independently with 20,000 rings. Using the
// same count here makes the comparison exact up to rounding.
#[rstest]
#[case::centre(0.0, 0.988_358_291_4)]
#[case::inside(0.5, 0.988_887_014_9)]
#[case::second_contact(0.9, 0.991_439_093_9)]
#[case::on_limb(1.0, 0.996_328_453_8)]
#[case::partial_outer(1.05, 0.998_702_273_4)]
fn ring_sum_matches_reference_values(#[case] z: f64, #[case] expected: f64) {
    let flux = flux_ring_sum(z, k(0.1), sun_like(), rings(20_000));

    assert!((flux - expected).abs() < 1e-9);
}

#[test]
fn four_rings_match_reference_value() {
    // z = 0.5, k = 0.1: four rings in equal steps of asin(r) from r = 0.4 to
    // 0.6. Small enough to work through ring by ring.
    let flux = flux_ring_sum(0.5, k(0.1), sun_like(), rings(4));

    assert!((flux - 0.988_887_305).abs() < 1e-8);
}

// 1000 rings is the count the light curve will use, so check it is within
// 1e-9 of the converged values everywhere, including on the limb.
#[rstest]
#[case::centre(0.0)]
#[case::inside(0.5)]
#[case::second_contact(0.9)]
#[case::on_limb(1.0)]
#[case::partial_outer(1.05)]
fn thousand_rings_are_within_1e9(#[case] z: f64) {
    let coarse = flux_ring_sum(z, k(0.1), sun_like(), rings(1000));
    let fine = flux_ring_sum(z, k(0.1), sun_like(), rings(20_000));

    assert!((coarse - fine).abs() < 1e-9, "z = {z}: {coarse} vs {fine}");
}

// A planet larger than the star, just past full coverage, with the darkest
// edge allowed: the brightness changes fastest under the planet's edge here.
// Equal steps in r were 8 ppm off at 1000 rings; this must be within 1 ppm.
#[rstest]
#[case::linear_dark_edge(LimbDarkening::linear(1.0).unwrap())]
#[case::sun_like(sun_like())]
#[case::steep_quadratic(LimbDarkening::quadratic(2.0, -1.0).unwrap())]
fn thousand_rings_meet_1ppm_for_large_planet(#[case] ld: LimbDarkening) {
    for z in [0.5044, 0.6, 1.0, 2.0] {
        let coarse = flux_ring_sum(z, k(1.5), ld, rings(1000));
        let fine = flux_ring_sum(z, k(1.5), ld, rings(50_000));

        assert!((coarse - fine).abs() < 1e-6, "z = {z}: {coarse} vs {fine}");
    }
}

// With no limb darkening the sum adds up the planet's area ring by ring, and
// the total is the same overlap the uniform formula gives in one step.
#[rstest]
fn uniform_ring_sum_matches_flux_uniform(
    #[values(0.01, 0.1, 0.5, 1.0, 1.5)] k_value: f64,
    #[values(1, 10, 1000)] n: usize,
) {
    for i in 0..=200 {
        let z = (1.0 + k_value + 0.1) * f64::from(i) / 200.0;
        let ring = flux_ring_sum(z, k(k_value), LimbDarkening::uniform(), rings(n));
        let exact = flux_uniform(z, k(k_value));

        assert!((ring - exact).abs() < 1e-12, "z = {z}: {ring} vs {exact}");
    }
}

// For a small planet fully on the disk, the blocked light is about the
// planet's area times the brightness under it. The error grows towards the
// limb, to about 2e-4 at z = 0.9 for k = 0.1.
#[rstest]
#[case::centre(0.0, 1e-5)]
#[case::inside(0.5, 2e-5)]
#[case::second_contact(0.9, 2e-4)]
fn ring_sum_matches_small_planet_estimate(#[case] z: f64, #[case] tolerance: f64) {
    let (u1, u2) = (0.3, 0.25);
    let x = 1.0 - (1.0 - z * z).sqrt();
    let intensity = 1.0 - u1 * x - u2 * x * x;
    let estimate = 1.0 - 0.01 * intensity / (1.0 - u1 / 3.0 - u2 / 6.0);

    let flux = flux_ring_sum(z, k(0.1), sun_like(), rings(1000));

    assert!((flux - estimate).abs() < tolerance, "{flux} vs {estimate}");
}

#[test]
fn limb_darkened_transit_is_deeper_at_centre_than_near_limb() {
    let centre = flux_ring_sum(0.0, k(0.1), sun_like(), rings(1000));
    let near_limb = flux_ring_sum(0.9, k(0.1), sun_like(), rings(1000));

    assert!(centre < near_limb);
}

#[rstest]
#[case::first_contact(1.1, 0.1)]
#[case::clear(1.2, 0.1)]
#[case::larger_first_contact(2.5, 1.5)]
fn ring_sum_is_exactly_one_once_clear(#[case] z: f64, #[case] k_value: f64) {
    assert_eq!(flux_ring_sum(z, k(k_value), sun_like(), rings(1000)), 1.0);
}

#[rstest]
#[case::centred(0.0, 1.5)]
#[case::still_covering(0.5, 1.5)]
#[case::equal_size_centred(0.0, 1.0)]
fn ring_sum_is_exactly_zero_when_star_is_covered(#[case] z: f64, #[case] k_value: f64) {
    assert_eq!(flux_ring_sum(z, k(k_value), sun_like(), rings(20_000)), 0.0);
}

#[test]
fn ring_sum_stays_between_zero_and_one() {
    let laws = [
        sun_like(),
        LimbDarkening::linear(1.0).unwrap(),
        LimbDarkening::linear(-0.5).unwrap(),
        LimbDarkening::quadratic(2.0, -1.0).unwrap(),
    ];

    for ld in laws {
        for k_value in [0.01, 0.1, 0.5, 1.0, 1.5] {
            let z_max = 1.0 + k_value + 0.1;

            for i in 0..=500 {
                let z = z_max * f64::from(i) / 500.0;
                let flux = flux_ring_sum(z, k(k_value), ld, rings(100));

                assert!(
                    (0.0..=1.0).contains(&flux),
                    "{ld:?}, k = {k_value}, z = {z}: {flux}"
                );
            }
        }
    }
}

#[rstest]
#[case::inside(-0.5)]
#[case::on_limb(-1.0)]
#[case::far(-5.0)]
fn negative_separation_matches_positive(#[case] z: f64) {
    let negative = flux_ring_sum(z, k(0.1), sun_like(), rings(100));
    let positive = flux_ring_sum(-z, k(0.1), sun_like(), rings(100));

    assert_eq!(negative, positive);
}

#[test]
fn nan_separation_gives_nan_flux() {
    assert!(flux_ring_sum(f64::NAN, k(0.1), sun_like(), rings(100)).is_nan());
}
