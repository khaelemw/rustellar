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

// The flux at a time is the flux at its separation when the planet is in
// front, and exactly 1 otherwise, over a whole orbit. In front, to rounding:
// `flux_at` squares and adds where `separation` uses `hypot`.
#[test]
fn flux_at_is_the_flux_at_the_separation_in_front() {
    let params = reference_transit(T0);
    for i in 0..30_000 {
        let t = at(T0 + f64::from(i) * 1e-4);
        if params.in_front(t) {
            let expected =
                rustellar_model::flux(params.separation(t), params.k(), params.limb_darkening());
            assert!((params.flux_at(t) - expected).abs() < 3e-16, "{t:?}");
        } else {
            assert_eq!(params.flux_at(t), 1.0, "{t:?}");
        }
    }
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

// The same transit observed far from `t0`, as when a Kepler-era ephemeris
// (around BTJD -1300) predicts TESS data (BTJD 3000 and later), against the
// same transit with `t0` moved next to it. The only difference allowed is the
// rounding of times of that size: about 4e-12 days at BTJD 30,000.
#[rstest]
#[case::kepler_to_tess(1433.0)]
#[case::ten_thousand_orbits_later(10_000.0)]
#[case::ten_thousand_orbits_earlier(-10_000.0)]
fn far_from_t0_matches_the_nearby_transit(#[case] orbits: f64) {
    let far = reference_transit(T0);
    let shift = TimeDelta::from_days(orbits * PERIOD_DAYS).unwrap();
    let near = reference_transit((at(T0) + shift).as_btjd());

    let mut worst: f64 = 0.0;
    for i in -100..=100 {
        let t = at(T0) + shift + TimeDelta::from_days(f64::from(i) * 6e-4).unwrap();
        worst = worst.max((far.flux_at(t) - near.flux_at(t)).abs());
    }
    assert!(worst < 1e-10, "{worst:e}");
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

// With limb darkening that dims towards the limb, the star only gets darker
// as the planet moves in from first contact to mid-transit.
#[test]
fn flux_falls_from_first_contact_to_mid_transit() {
    let params = reference_transit(T0);
    let t1 = params.contacts().unwrap().t1.as_days();
    let mut previous = 1.0;

    for i in 0..=500 {
        let t = T0 + t1 * (1.0 - f64::from(i) / 500.0);
        let flux = params.flux_at(at(t));

        assert!(flux <= previous + 1e-15, "t - t0 = {}", t - T0);
        previous = flux;
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
