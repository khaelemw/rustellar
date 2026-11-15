// Tests for the analytic flux, mostly against the ring sum, which computes the
// same integral by brute force.

use std::num::NonZeroUsize;

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
        for d in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13] {
            for z in [line - d, line + d] {
                let near = flux(z, k(radius_ratio), ld);
                // The flux changes by at most a few times d over d, plus the
                // up-to-1e-8 error of the limiting formula at z = k.
                assert!(
                    (near - on_line).abs() < 3.0 * d + 2e-8,
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
