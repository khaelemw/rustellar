// Tests for crate-only items, which the integration tests in `tests/` at the
// crate root cannot reach.

use rstest::rstest;

use crate::LimbDarkening;

// Every law `quadratic` accepts on a grid of coefficients, wide enough to
// include limb brightening and laws that are rejected.
fn accepted_laws() -> Vec<LimbDarkening> {
    let mut laws = Vec::new();

    for i in -10..=20 {
        for j in -10..=10 {
            let u1 = f64::from(i) / 10.0;
            let u2 = f64::from(j) / 10.0;

            if let Ok(ld) = LimbDarkening::quadratic(u1, u2) {
                laws.push(ld);
            }
        }
    }

    assert!(laws.len() > 100, "grid too sparse: {}", laws.len());
    laws
}

#[rstest]
#[case::uniform_centre(LimbDarkening::uniform(), 1.0, 1.0)]
#[case::uniform_edge(LimbDarkening::uniform(), 0.0, 1.0)]
#[case::linear_middle(LimbDarkening::linear(0.6).unwrap(), 0.5, 0.7)]
#[case::linear_edge(LimbDarkening::linear(0.6).unwrap(), 0.0, 0.4)]
#[case::quadratic_centre(LimbDarkening::quadratic(0.3, 0.25).unwrap(), 1.0, 1.0)]
#[case::quadratic_middle(LimbDarkening::quadratic(0.3, 0.25).unwrap(), 0.5, 0.7875)]
#[case::quadratic_edge(LimbDarkening::quadratic(0.3, 0.25).unwrap(), 0.0, 0.45)]
#[case::limb_brightening_edge(LimbDarkening::linear(-0.2).unwrap(), 0.0, 1.2)]
fn intensity_matches_reference_values(
    #[case] ld: LimbDarkening,
    #[case] mu: f64,
    #[case] expected: f64,
) {
    assert!((ld.intensity(mu) - expected).abs() < 1e-12);
}

#[test]
fn intensity_is_exactly_one_at_centre() {
    for ld in accepted_laws() {
        assert_eq!(ld.intensity(1.0), 1.0, "{ld:?}");
    }
}

// The constructors promise the intensity is never negative on the disk;
// this checks `intensity` keeps that promise.
#[test]
fn intensity_is_never_negative_on_disk() {
    for ld in accepted_laws() {
        for i in 0..=1000 {
            let mu = f64::from(i) / 1000.0;
            let intensity = ld.intensity(mu);

            assert!(intensity >= -1e-15, "{ld:?} at mu = {mu}: {intensity}");
        }
    }
}

// The disk integral over pi is 2 * integral of I(mu) mu dmu from 0 to 1.
// The integrand is a cubic in mu, and Simpson's rule on one panel is exact
// for cubics, so this checks the closed form against `intensity`.
#[test]
fn disk_integral_matches_integral_of_intensity() {
    for ld in accepted_laws() {
        let g = |mu: f64| 2.0 * ld.intensity(mu) * mu;
        let simpson = (g(0.0) + 4.0 * g(0.5) + g(1.0)) / 6.0;

        assert!((ld.disk_integral() - simpson).abs() < 1e-12, "{ld:?}");
    }
}
