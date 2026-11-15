use std::{f64::consts::PI, num::NonZeroUsize};

use rstest::rstest;

use crate::{
    LimbDarkening, RadiusRatio,
    analytic::{
        blocked_mu_centre_line, blocked_mu_centred, blocked_mu_inner_contact, blocked_mu_inside,
        blocked_mu_partial, blocked_r2,
    },
    flux_ring_sum,
    uniform::overlap_area,
};

// At this count the ring-sum moments are within about 1e-10 of the exact
// values for p up to 1.5 (the error falls as 1 / rings^2).
const RINGS: NonZeroUsize = NonZeroUsize::new(20_000).unwrap();

// Allowed difference from the ring sum: its own error, with some room.
const RING_TOLERANCE: f64 = 2e-10;

// B1 from the ring sum. Linear limb darkening with u1 = 1 has intensity mu and
// disk integral 2/3, so the blocked fraction (1 - F) is B1 / (2/3).
fn ring_b1(z: f64, p: f64) -> f64 {
    let ld = LimbDarkening::linear(1.0).unwrap();
    (1.0 - ring_flux(z, p, ld)) * 2.0 / 3.0
}

// B2 from the ring sum. Quadratic limb darkening with (2, -1) has intensity
// mu^2 = 1 - r^2 and disk integral 1/2, so (1 - F) / 2 is B0 - B2.
fn ring_b2(z: f64, p: f64) -> f64 {
    let ld = LimbDarkening::quadratic(2.0, -1.0).unwrap();
    overlap_area(z, p, 1.0) / PI - (1.0 - ring_flux(z, p, ld)) / 2.0
}

fn ring_flux(z: f64, p: f64, ld: LimbDarkening) -> f64 {
    flux_ring_sum(z, RadiusRatio::new(p).unwrap(), ld, RINGS)
}

// `count` separations evenly inside (lo, hi), endpoints excluded, skipping
// those within 1e-3 of z = p, where the general formulas lose precision.
fn separations(lo: f64, hi: f64, count: usize, p: f64) -> impl Iterator<Item = f64> {
    (0..count)
        .map(move |i| lo + (hi - lo) * (i as f64 + 0.5) / count as f64)
        .filter(move |z| (z - p).abs() > 1e-3)
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    let difference = (actual - expected).abs();
    assert!(
        difference < tolerance,
        "{what}: {actual} vs {expected}, difference {difference:e}"
    );
}

// B2

#[rstest]
#[case(0.01)]
#[case(0.1)]
#[case(0.3)]
#[case(0.5)]
#[case(0.75)]
#[case(1.0)]
#[case(1.5)]
fn blocked_r2_matches_ring_sum(#[case] p: f64) {
    for i in 0..=40 {
        let z = (1.0 + p) * i as f64 / 40.0;
        assert_close(
            blocked_r2(z, p),
            ring_b2(z, p),
            RING_TOLERANCE,
            &format!("z {z}"),
        );
    }
}

#[test]
fn blocked_r2_outside_is_zero() {
    assert_eq!(blocked_r2(1.1, 0.1), 0.0);
    assert_eq!(blocked_r2(5.0, 0.1), 0.0);
}

#[test]
fn blocked_r2_covering_is_half() {
    assert_eq!(blocked_r2(0.0, 1.5), 0.5);
    assert_eq!(blocked_r2(0.5, 1.5), 0.5);
}

#[test]
fn blocked_r2_inside_is_closed_form() {
    // (p^2 / 2)(p^2 + 2 z^2) with p = 1/4, z = 1/2: (1/32)(1/16 + 1/2).
    assert_eq!(blocked_r2(0.5, 0.25), 0.017_578_125);
}

#[rstest]
#[case(0.25)]
#[case(0.75)]
fn blocked_r2_is_continuous_at_inner_contact(#[case] p: f64) {
    let on_line = blocked_r2(1.0 - p, p);
    for d in [1e-6, 1e-9, 1e-12] {
        assert_close(
            blocked_r2(1.0 - p + d, p),
            on_line,
            3.0 * d + 1e-15,
            "above",
        );
        assert_close(
            blocked_r2(1.0 - p - d, p),
            on_line,
            3.0 * d + 1e-15,
            "below",
        );
    }
}

// B1, general formulas

#[rstest]
#[case(0.01)]
#[case(0.1)]
#[case(0.3)]
#[case(0.45)]
fn blocked_mu_inside_matches_ring_sum(#[case] p: f64) {
    for z in separations(0.0, 1.0 - p, 30, p) {
        assert_close(
            blocked_mu_inside(z, p),
            ring_b1(z, p),
            RING_TOLERANCE,
            &format!("z {z}"),
        );
    }
}

#[rstest]
#[case(0.01)]
#[case(0.1)]
#[case(0.5)]
#[case(0.9)]
#[case(1.0)]
#[case(1.5)]
fn blocked_mu_partial_matches_ring_sum(#[case] p: f64) {
    for z in separations((1.0 - p).abs(), 1.0 + p, 30, p) {
        assert_close(
            blocked_mu_partial(z, p),
            ring_b1(z, p),
            RING_TOLERANCE,
            &format!("z {z}"),
        );
    }
}

#[test]
fn blocked_mu_inside_covers_the_centre_on_both_sides() {
    // The 2/3 added for z < p: the value must not jump across z = p.
    let p = 0.25;
    let below = blocked_mu_inside(p - 1e-4, p);
    let above = blocked_mu_inside(p + 1e-4, p);
    assert_close(below, above, 1e-4, "across z = p");
}

#[test]
fn blocked_mu_partial_is_zero_at_first_contact() {
    let p = 0.1;
    assert_close(
        blocked_mu_partial(1.0 + p - 1e-12, p),
        0.0,
        1e-12,
        "first contact",
    );
}

// B1, special lines

#[rstest]
#[case(0.01)]
#[case(0.25)]
#[case(0.5)]
#[case(0.9)]
fn blocked_mu_centred_matches_ring_sum(#[case] p: f64) {
    assert_close(
        blocked_mu_centred(p),
        ring_b1(0.0, p),
        RING_TOLERANCE,
        "z = 0",
    );
}

#[rstest]
#[case(0.1)]
#[case(0.25)]
fn blocked_mu_centred_is_the_limit_of_inside(#[case] p: f64) {
    assert_close(
        blocked_mu_inside(1e-6, p),
        blocked_mu_centred(p),
        1e-10,
        "z -> 0",
    );
}

#[rstest]
#[case(0.01)]
#[case(0.25)]
#[case(0.4)]
#[case(0.5)]
#[case(0.6)]
#[case(0.75)]
#[case(1.2)]
#[case(2.0)]
fn blocked_mu_centre_line_matches_ring_sum(#[case] p: f64) {
    assert_close(
        blocked_mu_centre_line(p),
        ring_b1(p, p),
        RING_TOLERANCE,
        "z = p",
    );
}

#[rstest]
#[case(0.25)]
#[case(0.75)]
#[case(1.5)]
fn blocked_mu_centre_line_is_the_limit_of_the_general_formulas(#[case] p: f64) {
    let general = if p < 0.5 {
        blocked_mu_inside
    } else {
        blocked_mu_partial
    };
    let on_line = blocked_mu_centre_line(p);
    // B1 changes by about 0.4 d at distance d from the line.
    for d in [1e-4, 1e-6] {
        assert_close(general(p + d, p), on_line, d, "above");
        assert_close(general(p - d, p), on_line, d, "below");
    }
}

#[test]
fn blocked_mu_centre_line_is_continuous_at_half() {
    let at_half = blocked_mu_centre_line(0.5);
    assert_eq!(at_half, 1.0 / 3.0 - 4.0 / (9.0 * PI));
    for d in [1e-6, 1e-9, 1e-11] {
        assert_close(blocked_mu_centre_line(0.5 - d), at_half, d, "below");
        assert_close(blocked_mu_centre_line(0.5 + d), at_half, d, "above");
    }
}

// Binary-exact p, so that 1.0 - p is exactly the line.
#[rstest]
#[case(0.125)]
#[case(0.25)]
#[case(0.375)]
#[case(0.625)]
#[case(0.75)]
fn blocked_mu_inner_contact_matches_ring_sum(#[case] p: f64) {
    let z = 1.0 - p;
    assert_close(
        blocked_mu_inner_contact(p),
        ring_b1(z, p),
        RING_TOLERANCE,
        "z = 1 - p",
    );
}

#[rstest]
#[case(0.125)]
#[case(0.25)]
#[case(0.75)]
fn blocked_mu_inner_contact_is_the_limit_of_the_general_formulas(#[case] p: f64) {
    let on_line = blocked_mu_inner_contact(p);
    // The general formulas converge with an error of about 3 d.
    for d in [1e-6, 1e-9, 1e-12] {
        let z = 1.0 - p;
        assert_close(blocked_mu_inside(z - d, p), on_line, 5.0 * d, "inside");
        assert_close(blocked_mu_partial(z + d, p), on_line, 5.0 * d, "partial");
    }
}
