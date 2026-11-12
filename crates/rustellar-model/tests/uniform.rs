use rstest::rstest;
use rustellar_model::{RadiusRatio, flux_uniform};

// ==========================
// Helper Functions
// ==========================

fn flux(z: f64, k: f64) -> f64 {
    flux_uniform(z, RadiusRatio::new(k).unwrap())
}

// The neighbouring floats of a positive number. `f64::next_up` and
// `next_down` need Rust 1.86, above this crate's minimum.
fn next_up(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1)
}

fn next_down(x: f64) -> f64 {
    f64::from_bits(x.to_bits() - 1)
}

// ==========================
// Tests
// ==========================

// Expected values were computed independently with the lens formula.
#[rstest]
#[case::centre(0.0, 0.1, 0.99)]
#[case::inside(0.5, 0.1, 0.99)]
#[case::second_contact(0.9, 0.1, 0.99)]
#[case::partial_inner(0.95, 0.1, 0.992_026_64)]
#[case::partial_on_limb(1.0, 0.1, 0.995_106_13)]
#[case::partial_outer(1.05, 0.1, 0.998_111_44)]
#[case::equal_size_centre(0.0, 1.0, 0.0)]
#[case::equal_size_partial(0.5, 1.0, 0.314_962_36)]
#[case::equal_size_on_limb(1.0, 1.0, 0.608_997_78)]
#[case::larger_centre(0.0, 1.5, 0.0)]
#[case::larger_still_covering(0.5, 1.5, 0.0)]
#[case::larger_partial(1.0, 1.5, 0.258_299_94)]
#[case::larger_partial_outer(2.0, 1.5, 0.841_657_40)]
fn flux_matches_reference_values(#[case] z: f64, #[case] k: f64, #[case] expected: f64) {
    assert!((flux(z, k) - expected).abs() < 1e-8);
}

// At and beyond first contact the planet blocks nothing, so the flux must be
// exactly 1, not just close to it.
#[rstest]
#[case::first_contact(1.1, 0.1)]
#[case::clear(1.2, 0.1)]
#[case::far(10.0, 0.1)]
#[case::larger_first_contact(2.5, 1.5)]
fn flux_is_exactly_one_once_clear(#[case] z: f64, #[case] k: f64) {
    assert_eq!(flux(z, k), 1.0);
}

// Each place where the formula switches branch, compared with the floats on
// either side. A jump here would mean a wrong `<` or `<=`, or lost precision
// near the switch.
#[rstest]
#[case::second_contact(0.9, 0.1)]
#[case::first_contact(1.1, 0.1)]
#[case::half_size_second_contact(0.5, 0.5)]
#[case::equal_size_first_contact(2.0, 1.0)]
#[case::larger_stops_covering(0.5, 1.5)]
#[case::larger_first_contact(2.5, 1.5)]
fn flux_is_continuous_at_branch_switches(#[case] z: f64, #[case] k: f64) {
    let at = flux(z, k);

    assert!((flux(next_down(z), k) - at).abs() < 1e-9);
    assert!((flux(next_up(z), k) - at).abs() < 1e-9);
}

#[test]
fn flux_stays_between_zero_and_one() {
    for k in [0.01, 0.1, 0.5, 1.0, 1.5] {
        let z_max = 1.0 + k + 0.1;

        for i in 0..=10_000 {
            let z = z_max * f64::from(i) / 10_000.0;
            let f = flux(z, k);

            assert!((0.0..=1.0).contains(&f), "k = {k}, z = {z}: flux {f}");
        }
    }
}

#[test]
fn flux_falls_as_planet_moves_onto_star() {
    // From first contact to second contact the flux only goes down.
    let k = 0.1;
    let mut previous = flux(1.0 + k, k);

    for i in 1..=1000 {
        let z = 1.0 + k - 2.0 * k * f64::from(i) / 1000.0;
        let f = flux(z, k);

        assert!(f <= previous, "z = {z}: {f} > {previous}");
        previous = f;
    }
}

// Without this, a large negative z passes the "fully inside" check and gives
// a full transit depth for a planet nowhere near the star.
#[rstest]
#[case::inside(-0.5, 0.1)]
#[case::partial(-1.0, 0.1)]
#[case::far(-5.0, 0.1)]
#[case::larger_partial(-2.0, 1.5)]
fn negative_separation_matches_positive(#[case] z: f64, #[case] k: f64) {
    assert_eq!(flux(z, k), flux(-z, k));
}

#[test]
fn nan_separation_gives_nan_flux() {
    assert!(flux(f64::NAN, 0.1).is_nan());
}
