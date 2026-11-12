// Tests for `overlap_area` with a disk radius other than 1, which the ring
// sum relies on and the public `flux_uniform` never uses.

use std::f64::consts::PI;

use rstest::rstest;

use crate::uniform::overlap_area;

#[rstest]
#[case::outside(0.5)]
#[case::inside(0.05)]
fn zero_radius_disk_has_no_overlap(#[case] z: f64) {
    assert_eq!(overlap_area(z, 0.1, 0.0), 0.0);
}

#[rstest]
#[case::small_disk_inside(0.1, 0.5, 0.2)]
#[case::small_disk_partial(0.4, 0.5, 0.2)]
#[case::large_disk_partial(1.0, 0.1, 1.2)]
#[case::touching(0.7, 0.5, 0.2)]
fn overlap_is_symmetric_in_the_two_radii(#[case] z: f64, #[case] k: f64, #[case] big_r: f64) {
    let one_way = overlap_area(z, k, big_r);
    let other_way = overlap_area(z, big_r, k);

    assert!(
        (one_way - other_way).abs() < 1e-14,
        "{one_way} vs {other_way}"
    );
}

// Scaling every length by `s` scales the area by `s^2`.
#[rstest]
fn overlap_scales_with_area(#[values(0.25, 0.5, 2.0)] s: f64) {
    for i in 0..=100 {
        let z = 1.2 * f64::from(i) / 100.0;
        let scaled = overlap_area(s * z, s * 0.3, s * 0.8);
        let expected = s * s * overlap_area(z, 0.3, 0.8);

        assert!((scaled - expected).abs() < 1e-13, "s = {s}, z = {z}");
    }
}

#[test]
fn overlap_never_exceeds_either_disk() {
    for big_r in [0.1, 0.5, 1.0] {
        for i in 0..=200 {
            let z = 1.5 * f64::from(i) / 200.0;
            let area = overlap_area(z, 0.3, big_r);
            let smaller = PI * big_r.min(0.3).powi(2);

            assert!(
                (0.0..=smaller + 1e-15).contains(&area),
                "R = {big_r}, z = {z}: {area}"
            );
        }
    }
}
