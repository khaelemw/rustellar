use std::f64::consts::{FRAC_PI_2, PI};

use rstest::rstest;
use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

// ==========================
// Helper Functions
// ==========================

const PERIOD_DAYS: f64 = 3.0;

// A realistic TESS-era mid-transit time (BTJD), so the subtraction inside
// `phase_angle` works on real-sized numbers.
const T0: f64 = 3_000.123_456;

// Example A from the guide (3.4): a_rs = 10, b = 0.3, P = 3 d.
fn example_a(t0: f64) -> TransitParams {
    with_b(t0, 0.3)
}

fn with_b(t0: f64, b: f64) -> TransitParams {
    orbit(t0, 0.1, 10.0, b)
}

fn orbit(t0: f64, k: f64, a_rs: f64, b: f64) -> TransitParams {
    TransitParams::new(
        Epoch::from_btjd(t0).unwrap(),
        TimeDelta::from_days(PERIOD_DAYS).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(a_rs).unwrap(),
        ImpactParameter::new(b).unwrap(),
        LimbDarkening::uniform(),
    )
    .unwrap()
}

fn at(t: f64) -> Epoch {
    Epoch::from_btjd(t).unwrap()
}

// ==========================
// Tests
// ==========================

#[rstest]
#[case::mid_transit(0.0, 0.0, 0.3)]
#[case::early(0.05, 0.104_720, 1.087_031)]
#[case::later(0.10, 0.209_440, 2.099_723)]
#[case::quarter_orbit(0.75, FRAC_PI_2, 10.0)]
#[case::half_orbit(1.50, -PI, 0.3)]
fn geometry_matches_example_a(#[case] t: f64, #[case] phase: f64, #[case] separation: f64) {
    let params = example_a(0.0);

    assert!((params.phase_angle(at(t)) - phase).abs() < 1e-6);
    assert!((params.separation(at(t)) - separation).abs() < 1e-6);
}

#[rstest]
#[case::mid_transit(0.0, true)]
#[case::early(0.05, true)]
#[case::later(0.10, true)]
#[case::half_orbit(1.50, false)]
#[case::just_after_quarter(0.76, false)]
#[case::just_before_three_quarters(2.24, false)]
#[case::just_after_three_quarters(2.26, true)]
fn in_front_matches_side_of_orbit(#[case] t: f64, #[case] expected: bool) {
    assert_eq!(example_a(0.0).in_front(at(t)), expected);
}

#[rstest]
#[case::central(0.0)]
#[case::typical(0.3)]
#[case::grazing(1.05)]
#[case::missing_star(5.0)]
fn separation_at_t0_is_b(#[case] b: f64) {
    let params = with_b(T0, b);

    assert!((params.separation(at(T0)) - b).abs() < 1e-12);
    assert!(params.in_front(at(T0)));
}

#[test]
fn behind_star_half_an_orbit_later() {
    // Same separation as mid-transit, but on the far side of the star.
    let params = example_a(T0);
    let t = at(T0 + PERIOD_DAYS / 2.0);

    assert!((params.separation(t) - 0.3).abs() < 1e-9);
    assert!(!params.in_front(t));
}

#[rstest]
#[case::forward(1000.0)]
#[case::backward(-1000.0)]
fn separation_repeats_every_period(#[case] orbits: f64) {
    let params = example_a(T0);
    let near = params.separation(at(T0 + 0.05));
    let far = params.separation(at(T0 + orbits * PERIOD_DAYS + 0.05));

    assert!((far - near).abs() < 1e-9, "{near} vs {far}");
}

#[test]
fn separation_is_symmetric_about_t0() {
    let params = example_a(T0);

    for i in 1..100 {
        let dt = f64::from(i) * 0.013;
        let before = params.separation(at(T0 - dt));
        let after = params.separation(at(T0 + dt));
        assert!(
            (before - after).abs() < 1e-9,
            "dt = {dt}: {before} vs {after}"
        );
    }
}

#[test]
fn phase_angle_stays_in_range() {
    let params = example_a(0.0);

    for i in -2000..=2000 {
        let t = f64::from(i) * 0.37;
        let phase = params.phase_angle(at(t));
        assert!(
            (-PI..PI).contains(&phase),
            "t = {t}: phase {phase} is outside [-pi, pi)"
        );
    }
}

#[test]
fn durations_match_example_b() {
    // Example B from the guide (3.4): k = 0.1, b = 0.3, a_rs = 10, P = 3 d.
    let params = example_a(T0);
    let t14 = params.t14().unwrap();
    let t23 = params.t23().unwrap();

    assert!((t14.as_days() - 0.101_295_62).abs() < 1e-8);
    assert!((t23.as_days() - 0.081_162_64).abs() < 1e-8);
    assert!((t14.as_hours() - 2.431_095).abs() < 1e-6);
    assert!((t23.as_hours() - 1.947_903).abs() < 1e-6);
    assert!((t23 / t14 - 0.801_245).abs() < 1e-6);
}

#[rstest]
#[case::central(0.1, 0.0)]
#[case::typical(0.1, 0.3)]
#[case::near_edge(0.1, 0.85)]
#[case::large_planet(0.3, 0.5)]
fn contacts_land_on_the_stellar_edge(#[case] k: f64, #[case] b: f64) {
    // Half a duration after mid-transit, the planet's edge must touch the
    // star's edge: separation 1 + k at fourth contact, 1 - k at third.
    //
    // Uses t0 = 0: near BTJD 3000 an f64 time only resolves about 5e-13 days,
    // which moves the planet by about 1e-11 stellar radii, more than the
    // 1e-12 this checks.
    let params = orbit(0.0, k, 10.0, b);
    let t0 = at(0.0);
    let fourth = params.separation(t0 + params.t14().unwrap() * 0.5);
    let third = params.separation(t0 + params.t23().unwrap() * 0.5);

    assert!(
        (fourth - (1.0 + k)).abs() < 1e-12,
        "fourth contact at {fourth}"
    );
    assert!(
        (third - (1.0 - k)).abs() < 1e-12,
        "third contact at {third}"
    );
}

#[rstest]
#[case::central(0.1, 0.0, true, true)]
#[case::typical(0.1, 0.3, true, true)]
#[case::grazing(0.1, 0.95, true, false)]
#[case::grazing_at_flat_bottom_limit(0.1, 0.9, true, false)]
#[case::touching_only(0.1, 1.1, false, false)]
#[case::missing_star(0.1, 1.2, false, false)]
#[case::planet_larger_than_star(1.5, 0.0, true, false)]
fn durations_exist_only_when_contacts_do(
    #[case] k: f64,
    #[case] b: f64,
    #[case] has_t14: bool,
    #[case] has_t23: bool,
) {
    let params = orbit(T0, k, 10.0, b);

    assert_eq!(params.t14().is_some(), has_t14);
    assert_eq!(params.t23().is_some(), has_t23);
}

#[test]
fn grazing_duration_matches_guide() {
    // b = 1.0, k = 0.1 from the guide's example B edge cases.
    let t14 = orbit(T0, 0.1, 10.0, 1.0).t14().unwrap();
    assert!((t14.as_hours() - 1.055_913).abs() < 1e-6);
}

#[test]
fn close_orbit_lasts_half_the_period() {
    // a_rs < 1 + k: the planet overlaps the star for its whole near-side half
    // orbit.
    let t14 = orbit(T0, 0.1, 1.05, 0.3).t14().unwrap();
    assert!((t14.as_days() - PERIOD_DAYS / 2.0).abs() < 1e-12);
}

#[test]
fn flat_bottom_is_shorter_than_transit() {
    for i in 0..90 {
        let b = f64::from(i) * 0.01;
        let params = with_b(T0, b);
        let t14 = params.t14().unwrap();
        let t23 = params.t23().unwrap();
        assert!(
            t23 < t14,
            "b = {b}: t23 {t23:?} is not shorter than t14 {t14:?}"
        );
    }
}

#[test]
fn contacts_match_example_b() {
    let contacts = example_a(T0).contacts().unwrap();
    let half_t14 = 0.101_295_62 / 2.0;
    let half_t23 = 0.081_162_64 / 2.0;

    assert!((contacts.t1.as_days() + half_t14).abs() < 1e-8);
    assert!((contacts.t2.unwrap().as_days() + half_t23).abs() < 1e-8);
    assert!((contacts.t3.unwrap().as_days() - half_t23).abs() < 1e-8);
    assert!((contacts.t4.as_days() - half_t14).abs() < 1e-8);
}

#[rstest]
#[case::central(0.1, 0.0)]
#[case::typical(0.1, 0.3)]
#[case::near_edge(0.1, 0.85)]
#[case::large_planet(0.3, 0.5)]
fn every_contact_lands_on_the_stellar_edge(#[case] k: f64, #[case] b: f64) {
    // t0 = 0 for the same precision reason as above.
    let params = orbit(0.0, k, 10.0, b);
    let contacts = params.contacts().unwrap();
    let t0 = at(0.0);
    let outer = [contacts.t1, contacts.t4];
    let inner = [contacts.t2.unwrap(), contacts.t3.unwrap()];

    for offset in outer {
        let z = params.separation(t0 + offset);
        assert!((z - (1.0 + k)).abs() < 1e-12, "{offset:?}: z = {z}");
    }
    for offset in inner {
        let z = params.separation(t0 + offset);
        assert!((z - (1.0 - k)).abs() < 1e-12, "{offset:?}: z = {z}");
    }
}

#[rstest]
#[case::central(0.0)]
#[case::typical(0.3)]
#[case::near_edge(0.85)]
fn contacts_are_symmetric_and_ordered(#[case] b: f64) {
    let contacts = with_b(T0, b).contacts().unwrap();
    let t2 = contacts.t2.unwrap();
    let t3 = contacts.t3.unwrap();

    assert_eq!(contacts.t1, -contacts.t4);
    assert_eq!(t2, -t3);
    assert!(contacts.t1 < t2 && t2 < t3 && t3 < contacts.t4);
}

#[test]
fn contacts_agree_with_durations() {
    let params = example_a(T0);
    let contacts = params.contacts().unwrap();

    assert_eq!(contacts.t4 - contacts.t1, params.t14().unwrap());
    assert_eq!(
        contacts.t3.unwrap() - contacts.t2.unwrap(),
        params.t23().unwrap()
    );
}

#[test]
fn grazing_transit_has_only_outer_contacts() {
    let contacts = with_b(T0, 0.95).contacts().unwrap();

    assert!(contacts.t1.as_days() < 0.0);
    assert!(contacts.t4.as_days() > 0.0);
    assert_eq!(contacts.t2, None);
    assert_eq!(contacts.t3, None);
}

#[test]
fn missing_the_star_has_no_contacts() {
    assert_eq!(with_b(T0, 1.2).contacts(), None);
}

#[rstest]
#[case::from_t0(-0.1, 10.0, &[0, 1, 2, 3])]
#[case::before_t0(-10.0, -0.1, &[-3, -2, -1])]
#[case::ends_on_transits(6.0, 15.0, &[2, 3, 4, 5])]
#[case::single_instant(21.0, 21.0, &[7])]
#[case::far_from_t0(30_000.0, 30_007.0, &[10_000, 10_001, 10_002])]
#[case::between_transits(0.5, 2.5, &[])]
#[case::end_before_start(10.0, 0.0, &[])]
fn transit_times_cover_the_window(
    #[case] start_days: f64,
    #[case] end_days: f64,
    #[case] orbits: &[i64],
) {
    // Window ends are in days from t0. The expected times are built the same
    // way as in the model, so they can be compared exactly.
    let params = example_a(T0);
    let times: Vec<Epoch> = params
        .transit_times(at(T0 + start_days), at(T0 + end_days))
        .collect();
    let expected: Vec<Epoch> = orbits
        .iter()
        .map(|&n| params.t0() + params.period() * n as f64)
        .collect();

    assert_eq!(times, expected);
}

#[test]
fn transit_times_are_mid_transits() {
    let params = example_a(T0);

    for t in params.transit_times(at(T0 - 30.0), at(T0 + 30.0)) {
        assert!(params.phase_angle(t).abs() < 1e-9);
        assert!(params.in_front(t));
    }
}
