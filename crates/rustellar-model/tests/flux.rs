// Tests for the analytic flux, mostly against the ring sum, which computes the
// same integral by brute force.

use std::{f64::consts::PI, num::NonZeroUsize};

use rstest::rstest;
use rustellar_model::{LimbDarkening, RadiusRatio, flux, flux_ring_sum, flux_uniform};

// ==========================
// Helper Functions
// ==========================

// Ring count for the reference flux; within about 1e-10 of exact here.
const RINGS: NonZeroUsize = NonZeroUsize::new(20_000).unwrap();

// Allowed difference from the ring sum: its own error, with room.
const RING_TOLERANCE: f64 = 1e-9;

fn k(k: f64) -> RadiusRatio {
    RadiusRatio::new(k).unwrap()
}

fn sun_like() -> LimbDarkening {
    LimbDarkening::quadratic(0.4, 0.25).unwrap()
}

// A spread of limb darkening, including the edges of what is allowed.
fn laws() -> [LimbDarkening; 5] {
    [
        LimbDarkening::uniform(),
        LimbDarkening::linear(1.0).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
        LimbDarkening::quadratic(0.1, 0.6).unwrap(),
        LimbDarkening::quadratic(2.0, -1.0).unwrap(),
    ]
}

// Separations from the centre to just past first contact, plus every line
// where the formula changes: z = 0, k, 1 - k, 1 + k and k - 1.
fn separations(k: f64) -> Vec<f64> {
    let mut z: Vec<f64> = (0..=50)
        .map(|i| (1.0 + k) * 1.02 * i as f64 / 50.0)
        .collect();
    z.extend([0.0, k, (1.0 - k).abs(), 1.0 + k, (k - 1.0).max(0.0)]);
    z
}

// The lines between the formula's regions, for continuity checks.
fn lines(k: f64) -> Vec<f64> {
    let mut lines = vec![k, 1.0 + k];
    if k < 1.0 {
        lines.push(1.0 - k);
    } else {
        lines.push(k - 1.0);
    }
    lines.into_iter().filter(|&z| z > 0.0).collect()
}

// ==========================
// Tests
// ==========================

#[rstest]
#[case(0.0001)]
#[case(0.005)]
#[case(0.01)]
#[case(0.05)]
#[case(0.1)]
#[case(0.2)]
#[case(0.5)]
#[case(0.9)]
#[case(1.0)]
#[case(1.5)]
fn matches_ring_sum(#[case] radius_ratio: f64) {
    for ld in laws() {
        for z in separations(radius_ratio) {
            let analytic = flux(z, k(radius_ratio), ld);
            let rings = flux_ring_sum(z, k(radius_ratio), ld, RINGS);
            assert!(
                (analytic - rings).abs() < RING_TOLERANCE,
                "k {radius_ratio}, z {z}, {ld:?}: {analytic} vs {rings}"
            );
        }
    }
}

#[rstest]
#[case(0.01)]
#[case(0.1)]
#[case(0.5)]
#[case(1.5)]
fn uniform_star_matches_flux_uniform(#[case] radius_ratio: f64) {
    for z in separations(radius_ratio) {
        let analytic = flux(z, k(radius_ratio), LimbDarkening::uniform());
        let uniform = flux_uniform(z, k(radius_ratio));
        assert!(
            (analytic - uniform).abs() < 1e-15,
            "z {z}: {analytic} vs {uniform}"
        );
    }
}

// Approaching each line from both sides, the flux must head smoothly to its
// value on the line: no jumps from a wrong case or a missing 2/3.
#[rstest]
#[case(0.01)]
#[case(0.1)]
#[case(0.25)]
#[case(0.5)]
#[case(0.75)]
#[case(1.0)]
#[case(1.5)]
fn continuous_across_case_lines(#[case] radius_ratio: f64) {
    let ld = sun_like();
    for line in lines(radius_ratio) {
        let on_line = flux(line, k(radius_ratio), ld);
        for d in [1e-3, 1.01e-5, 9.9e-6, 1e-7, 1e-9, 1e-11, 1e-13] {
            for z in [line - d, line + d] {
                let near = flux(z, k(radius_ratio), ld);
                // The flux changes by at most a few times d over d, plus
                // rounding.
                assert!(
                    (near - on_line).abs() < 3.0 * d + 1e-15,
                    "k {radius_ratio}, line {line}, z {z}: {near} vs {on_line}"
                );
            }
        }
    }
}

#[rstest]
#[case(0.01)]
#[case(0.1)]
#[case(0.5)]
#[case(1.0)]
#[case(1.5)]
fn stays_between_zero_and_one(#[case] radius_ratio: f64) {
    for ld in laws() {
        for i in 0..=500 {
            let z = (1.0 + radius_ratio) * i as f64 / 500.0;
            let f = flux(z, k(radius_ratio), ld);
            assert!((0.0..=1.0).contains(&f), "k {radius_ratio}, z {z}: {f}");
        }
    }
}

#[test]
fn exactly_one_out_of_transit() {
    assert_eq!(flux(1.1, k(0.1), sun_like()), 1.0);
    assert_eq!(flux(3.0, k(0.1), sun_like()), 1.0);
    assert_eq!(flux(f64::INFINITY, k(0.1), sun_like()), 1.0);
}

#[test]
fn exactly_zero_when_star_is_covered() {
    assert_eq!(flux(0.0, k(1.5), sun_like()), 0.0);
    assert_eq!(flux(0.5, k(1.5), sun_like()), 0.0);
}

#[test]
fn negative_separation_is_the_same_distance() {
    for z in [0.0, 0.05, 0.3, 0.9, 1.05] {
        assert_eq!(flux(-z, k(0.1), sun_like()), flux(z, k(0.1), sun_like()));
    }
}

#[test]
fn nan_separation_gives_nan() {
    assert!(flux(f64::NAN, k(0.1), sun_like()).is_nan());
}

// Small planet: the planet covers a patch of nearly constant brightness, so
// 1 - F is close to k^2 I(z) / disk_integral, with an error of order k^4.
#[test]
fn small_planet_blocks_its_area_times_local_brightness() {
    let ld = sun_like();
    let radius_ratio = 0.001;
    for z in [0.0_f64, 0.3, 0.6, 0.9] {
        let mu = (1.0 - z * z).sqrt();
        let intensity = 1.0 - 0.4 * (1.0 - mu) - 0.25 * (1.0 - mu).powi(2);
        let expected = radius_ratio * radius_ratio * intensity / ld.disk_integral();
        let blocked = 1.0 - flux(z, k(radius_ratio), ld);
        assert!(
            (blocked - expected).abs() < 1e-10,
            "z {z}: {blocked} vs {expected}"
        );
    }
}

// ==========================
// Closed forms from Mandel & Agol (2002)
// ==========================

// Flux from the three blocked moments over pi (B0, B1, B2) for coefficients
// (u1, u2): 1 - [(1 - c2) B0 + c2 B1 - c4 B2] / (1 - u1/3 - u2/6).
fn from_moments(b0: f64, b1: f64, b2: f64, u1: f64, u2: f64) -> f64 {
    let c2 = u1 + 2.0 * u2;
    let c4 = -u2;
    1.0 - ((1.0 - c2) * b0 + c2 * b1 - c4 * b2) / (1.0 - u1 / 3.0 - u2 / 6.0)
}

// Planet centred on the star: it covers whole rings, so every moment is
// elementary. B0 = k^2, B1 = (2/3)(1 - (1 - k^2)^(3/2)), B2 = k^4 / 2.
#[rstest]
#[case(0.01)]
#[case(0.1)]
#[case(0.5)]
#[case(0.9)]
#[case(0.999)]
fn centred_planet_matches_closed_form(#[case] radius_ratio: f64) {
    let k2 = radius_ratio * radius_ratio;
    let b1 = 2.0 / 3.0 * (1.0 - (1.0 - k2).powf(1.5));
    for (u1, u2) in [(0.4, 0.25), (1.0, 0.0), (0.1, 0.6), (2.0, -1.0)] {
        let ld = LimbDarkening::quadratic(u1, u2).unwrap();
        let expected = from_moments(k2, b1, k2 * k2 / 2.0, u1, u2);
        let actual = flux(0.0, k(radius_ratio), ld);
        assert!(
            (actual - expected).abs() < 1e-14,
            "k {radius_ratio}, u ({u1}, {u2}): {actual} vs {expected}"
        );
    }
}

// k = z = 1/2: the planet's edge passes through the star's centre and touches
// the limb from inside. Mandel & Agol give B1 = 1/3 - 4 / (9 pi); the planet
// is fully inside, so B0 = k^2 = 1/4 and B2 = (k^2 / 2)(k^2 + 2 z^2) = 3/32.
#[test]
fn half_size_planet_through_centre_matches_closed_form() {
    let b1 = 1.0 / 3.0 - 4.0 / (9.0 * PI);
    for (u1, u2) in [(0.4, 0.25), (1.0, 0.0), (2.0, -1.0)] {
        let ld = LimbDarkening::quadratic(u1, u2).unwrap();
        let expected = from_moments(0.25, b1, 3.0 / 32.0, u1, u2);
        let actual = flux(0.5, k(0.5), ld);
        assert!(
            (actual - expected).abs() < 1e-14,
            "u ({u1}, {u2}): {actual} vs {expected}"
        );
    }
}

// Fully inside a uniform star the planet blocks exactly its own area.
#[rstest]
#[case(0.0)]
#[case(0.3)]
#[case(0.75)]
fn uniform_star_fully_inside_blocks_planet_area(#[case] z: f64) {
    assert_eq!(flux(z, k(0.25), LimbDarkening::uniform()), 1.0 - 0.0625);
}

// ==========================
// Invariants
// ==========================

// The blocked light, (1 - F) times the disk integral, is linear in (u1, u2)
// because the intensity is.
#[rstest]
#[case(0.05)]
#[case(0.3)]
#[case(1.5)]
fn blocked_light_is_linear_in_coefficients(#[case] radius_ratio: f64) {
    let blocked = |z: f64, u1: f64, u2: f64| {
        let ld = LimbDarkening::quadratic(u1, u2).unwrap();
        (1.0 - flux(z, k(radius_ratio), ld)) * ld.disk_integral()
    };
    for z in separations(radius_ratio) {
        let base = blocked(z, 0.0, 0.0);
        let per_u1 = blocked(z, 1.0, 0.0) - base;
        let per_u2 = blocked(z, 0.0, 1.0) - base;
        for (u1, u2) in [(0.4, 0.25), (0.1, 0.6), (2.0, -1.0), (-0.3, 0.5)] {
            let expected = base + u1 * per_u1 + u2 * per_u2;
            let actual = blocked(z, u1, u2);
            assert!(
                (actual - expected).abs() < 1e-13,
                "z {z}, u ({u1}, {u2}): {actual} vs {expected}"
            );
        }
    }
}

// A larger planet at the same separation covers a superset of the star, so it
// never lets more light through.
#[test]
fn larger_planet_never_brightens() {
    let radius_ratios: Vec<f64> = (1..=60).map(|i| 0.025 * f64::from(i)).collect();
    for ld in laws() {
        for i in 0..=100 {
            let z = 2.6 * f64::from(i) / 100.0;
            let fluxes: Vec<f64> = radius_ratios.iter().map(|&r| flux(z, k(r), ld)).collect();
            for (pair, r) in fluxes.windows(2).zip(&radius_ratios) {
                assert!(pair[1] <= pair[0] + 1e-15, "z {z}, k {r}: {pair:?}");
            }
        }
    }
}

// When the intensity falls from centre to limb (u1 >= 0 and u1 + 2 u2 >= 0),
// moving the planet outwards never dims the star. (Anderson's theorem: the
// overlap of two radially decreasing profiles falls with their distance.)
#[rstest]
#[case(0.01)]
#[case(0.1)]
#[case(0.5)]
#[case(1.5)]
fn moving_outwards_never_dims_a_darkened_star(#[case] radius_ratio: f64) {
    let darkened = [
        LimbDarkening::uniform(),
        LimbDarkening::linear(1.0).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
        LimbDarkening::quadratic(2.0, -1.0).unwrap(),
    ];
    for ld in darkened {
        let mut previous = 0.0;
        for i in 0..=2000 {
            let z = (1.0 + radius_ratio) * 1.01 * f64::from(i) / 2000.0;
            let f = flux(z, k(radius_ratio), ld);
            assert!(f >= previous - 1e-15, "{ld:?}, z {z}: {f} after {previous}");
            previous = f;
        }
    }
}

// ==========================
// Recorded edge behaviour
// ==========================

// Points where rounding took the flux of an earlier version up to 1.5e-12
// outside [0, 1]: just inside first contact, and near full cover for large k.
#[rstest]
#[case(0.008_218_304_526_170_69, 1.008_218_304_526_167)]
#[case(5.130_142_744_967_692, 6.130_142_744_967_677)]
#[case(10.518_758_779_314_913, 9.518_758_779_314_92)]
#[case(8.679_067_557_546_007, 7.679_067_557_648_099)]
#[case(1.176_093_250_154_705_7, 0.176_093_250_154_707_17)]
fn stays_in_range_where_rounding_overshoots(#[case] radius_ratio: f64, #[case] z: f64) {
    for ld in laws() {
        let f = flux(z, k(radius_ratio), ld);
        assert!((0.0..=1.0).contains(&f), "{ld:?}: {f:e}");
    }
}

// More rings, since for large k the ring sum's own error grows.
const MORE_RINGS: NonZeroUsize = NonZeroUsize::new(200_000).unwrap();

// From vanishingly small radius ratios up to 100, where the ring sum's own
// error sets the tolerance. The analytic flux is within 1e-15 max(1, k) of
// exact up to k = 1e4 (tests/exact_reference.rs).
#[rstest]
#[case(1e-300, RING_TOLERANCE)]
#[case(1e-12, RING_TOLERANCE)]
#[case(1e-6, RING_TOLERANCE)]
#[case(20.0, RING_TOLERANCE)]
#[case(100.0, RING_TOLERANCE)]
fn accurate_for_extreme_radius_ratios(#[case] radius_ratio: f64, #[case] tolerance: f64) {
    let ld = sun_like();
    let start = (radius_ratio - 1.0).max(0.0);
    for i in 0..=40 {
        let z = start + (1.0 + radius_ratio - start) * f64::from(i) / 40.0;
        let analytic = flux(z, k(radius_ratio), ld);
        assert!((0.0..=1.0).contains(&analytic), "z {z}: {analytic}");
        let rings = flux_ring_sum(z, k(radius_ratio), ld, MORE_RINGS);
        assert!(
            (analytic - rings).abs() < tolerance,
            "k {radius_ratio}, z {z}: {analytic} vs {rings}"
        );
    }
}

#[test]
fn signed_zero_and_subnormal_separations() {
    let ld = sun_like();
    let centred = flux(0.0, k(0.1), ld);
    assert_eq!(flux(-0.0, k(0.1), ld), centred);
    assert!((flux(f64::MIN_POSITIVE, k(0.1), ld) - centred).abs() < 1e-15);
    assert!((flux(5e-324, k(0.1), ld) - centred).abs() < 1e-15);
    assert_eq!(flux(f64::NEG_INFINITY, k(0.1), ld), 1.0);
}
