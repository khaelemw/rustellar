use rstest::rstest;
use rustellar_core::{Epoch, LightCurve, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

// ==========================
// Helper Functions
// ==========================

const PERIOD_DAYS: f64 = 3.0;

// A realistic TESS-era mid-transit time (BTJD), so the time arithmetic works
// on real-sized numbers.
const T0: f64 = 3_000.123_456;

// Reference transit: k = 0.1, a_rs = 10, b = 0.3, P = 3 d, quadratic limb
// darkening (0.3, 0.25). Expected values in these tests were computed
// independently.
fn reference_transit(t0: f64) -> TransitParams {
    TransitParams::new(
        Epoch::from_btjd(t0).unwrap(),
        TimeDelta::from_days(PERIOD_DAYS).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.3, 0.25).unwrap(),
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
#[case::mid_transit(0.00, 0.988_527_81)]
#[case::early(0.02, 0.988_925_66)]
#[case::near_second_contact(0.04, 0.991_248_45)]
#[case::after_fourth_contact(0.06, 1.0)]
#[case::well_clear(0.10, 1.0)]
#[case::behind_star(1.50, 1.0)]
fn flux_matches_reference_values(#[case] t: f64, #[case] expected: f64) {
    let flux = reference_transit(0.0).flux_at(at(t));

    assert!((flux - expected).abs() < 1e-8, "t = {t}: {flux}");
}

#[rstest]
#[case::after_fourth_contact(0.06)]
#[case::behind_star(1.50)]
fn flux_is_exactly_one_out_of_transit(#[case] t: f64) {
    assert_eq!(reference_transit(T0).flux_at(at(T0 + t)), 1.0);
}

// Half an orbit after mid-transit the separation equals b again, but the
// planet is behind the star, so nothing is blocked.
#[test]
fn planet_behind_star_blocks_nothing() {
    let params = reference_transit(T0);
    let behind = at(T0 + PERIOD_DAYS / 2.0);

    assert!((params.separation(behind) - 0.3).abs() < 1e-12);
    assert_eq!(params.flux_at(behind), 1.0);
}

#[rstest]
#[case::later(1000.0)]
#[case::earlier(-1000.0)]
fn flux_repeats_every_orbit(#[case] orbits: f64) {
    let params = reference_transit(T0);
    let first = params.flux_at(at(T0 + 0.02));
    let later = params.flux_at(at(T0 + orbits * PERIOD_DAYS + 0.02));

    assert!((first - later).abs() < 1e-9, "{first} vs {later}");
}

#[test]
fn transit_is_symmetric_about_mid_transit() {
    let params = reference_transit(T0);

    for i in 1..=60 {
        let dt = f64::from(i) * 0.001;
        let before = params.flux_at(at(T0 - dt));
        let after = params.flux_at(at(T0 + dt));

        assert!((before - after).abs() < 1e-12, "dt = {dt}");
    }
}

#[test]
fn transit_lasts_from_first_to_fourth_contact() {
    let params = reference_transit(T0);
    let contacts = params.contacts().unwrap();
    let t0 = params.t0();

    // Just inside the contacts the planet blocks some light; just outside it
    // blocks none.
    let nudge = TimeDelta::from_days(1e-6).unwrap();

    assert!(params.flux_at(t0 + contacts.t1 + nudge) < 1.0);
    assert!(params.flux_at(t0 + contacts.t4 - nudge) < 1.0);
    assert_eq!(params.flux_at(t0 + contacts.t1 - nudge), 1.0);
    assert_eq!(params.flux_at(t0 + contacts.t4 + nudge), 1.0);
}

#[test]
fn light_curve_matches_flux_at_each_time() {
    let params = reference_transit(T0);

    // Unsorted on purpose: the order of the output follows the input.
    let times = [0.04, -0.02, 0.0, 1.5, 0.06, 0.02].map(|dt| at(T0 + dt));
    let flux = params.light_curve(&times);

    assert_eq!(flux.len(), times.len());
    for (t, f) in times.iter().zip(&flux) {
        assert_eq!(*f, params.flux_at(*t));
    }
}

#[test]
fn light_curve_of_no_times_is_empty() {
    assert!(reference_transit(T0).light_curve(&[]).is_empty());
}

#[test]
fn light_curve_takes_times_from_a_light_curve() {
    let time: Vec<f64> = (0..50).map(|i| T0 - 0.1 + 0.004 * f64::from(i)).collect();
    let n = time.len();
    let observed = LightCurve::from_btjd(time, vec![1.0; n], vec![0.001; n]).unwrap();

    let model = reference_transit(T0).light_curve(observed.time());

    assert_eq!(model.len(), observed.len());
    assert!(model.iter().any(|&f| f < 1.0), "no point in transit");
    assert!(model.iter().all(|&f| (0.0..=1.0).contains(&f)));
}
