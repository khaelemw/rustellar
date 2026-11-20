// Tests for averaging the flux over an exposure.

use std::num::NonZeroUsize;

use rstest::rstest;
use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    Exposure, ImpactParameter, LimbDarkening, ParamError, RadiusRatio, ScaledSemiMajorAxis,
    TransitParams,
};

// ==========================
// Helper Functions
// ==========================

const T0: f64 = 3_000.123_456;

// Kepler long cadence, in days.
const KEPLER_LONG: f64 = 29.4 / 1440.0;

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn samples(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn exposure(duration: f64, n: usize) -> Exposure {
    Exposure::new(days(duration), samples(n)).unwrap()
}

fn at(t: f64) -> Epoch {
    Epoch::from_btjd(t).unwrap()
}

// A hot Jupiter: T14 is about 2.4 hours.
fn transit(ld: LimbDarkening) -> TransitParams {
    TransitParams::new(
        at(T0),
        days(3.0),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        ld,
    )
    .unwrap()
}

fn sun_like() -> TransitParams {
    transit(LimbDarkening::quadratic(0.4, 0.25).unwrap())
}

// Times from before first contact to after fourth contact.
fn across_transit() -> impl Iterator<Item = f64> {
    (-60..=60).map(|i| T0 + f64::from(i) * 0.001)
}

// The exposure average by composite Simpson's rule over 20,000 intervals: an
// independent quadrature.
fn simpson_average(params: TransitParams, t: f64, duration: f64) -> f64 {
    let intervals = 20_000;
    let h = duration / f64::from(intervals);
    let start = t - duration / 2.0;
    let sum: f64 = (0..=intervals)
        .map(|i| {
            let weight = if i == 0 || i == intervals {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            weight * params.flux_at(at(start + f64::from(i) * h))
        })
        .sum();
    sum * h / 3.0 / duration
}

// ==========================
// Tests
// ==========================

#[test]
fn exposure_keeps_its_values() {
    let e = exposure(KEPLER_LONG, 15);
    assert_eq!(e.duration(), days(KEPLER_LONG));
    assert_eq!(e.samples().get(), 15);
}

#[rstest]
#[case::zero(0.0)]
#[case::negative_zero(-0.0)]
#[case::negative(-0.02)]
fn exposure_must_be_positive(#[case] duration: f64) {
    assert_eq!(
        Exposure::new(days(duration), samples(5)),
        Err(ParamError::ExposureNotPositive)
    );
}

#[test]
fn offsets_are_the_middles_of_equal_slices() {
    let offsets: Vec<f64> = exposure(1.0, 4).offsets().map(TimeDelta::as_days).collect();
    assert_eq!(offsets, [-0.375, -0.125, 0.125, 0.375]);
}

#[rstest]
#[case(1)]
#[case(2)]
#[case(7)]
#[case(30)]
fn offsets_are_symmetric_and_inside_the_exposure(#[case] n: usize) {
    let offsets: Vec<f64> = exposure(KEPLER_LONG, n)
        .offsets()
        .map(TimeDelta::as_days)
        .collect();
    assert_eq!(offsets.len(), n);
    for (early, late) in offsets.iter().zip(offsets.iter().rev()) {
        assert_eq!(*early, -late);
    }
    for pair in offsets.windows(2) {
        assert!(pair[0] < pair[1]);
    }
    assert!(offsets.iter().all(|o| o.abs() < KEPLER_LONG / 2.0));
}

#[test]
fn one_sample_is_the_flux_at_the_time_stamp() {
    let params = sun_like();
    let one = exposure(KEPLER_LONG, 1);
    for t in across_transit() {
        assert_eq!(params.binned_flux_at(at(t), one), params.flux_at(at(t)));
    }
}

#[test]
fn exactly_one_when_the_whole_exposure_is_out_of_transit() {
    let params = sun_like();
    for t in [T0 - 0.2, T0 + 0.2, T0 + 1.5] {
        assert_eq!(params.binned_flux_at(at(t), exposure(KEPLER_LONG, 30)), 1.0);
    }
}

// The average turns the time stamp's phase by each instant's angle rather
// than finding each instant's phase from its time, and skips exposures wholly
// out of transit. It is still the mean of `flux_at` over the instants: within
// 1e-14 at BTJD-sized times, and exactly 1 out of transit. At BJD-sized times
// (2.46 million days) a time stamp is only good to 5e-10 days, so the flux
// differs from the same phase at BTJD-sized times by up to 3e-10 either way,
// summed instant by instant or turned. Cases: a hot Jupiter, a grazing
// transit, a planet that misses the star, one so close it covers the star for
// half its orbit, and an exposure a third of the period; every 0.25 millidays
// over a whole orbit, and 1.3e-7 days on.
#[test]
fn the_mean_of_flux_at_its_instants() {
    let planet = |k: f64, a: f64, b: f64| {
        TransitParams::new(
            at(T0),
            days(3.0),
            RadiusRatio::new(k).unwrap(),
            ScaledSemiMajorAxis::new(a).unwrap(),
            ImpactParameter::new(b).unwrap(),
            LimbDarkening::quadratic(0.4, 0.25).unwrap(),
        )
        .unwrap()
    };
    let cases = [
        (sun_like(), exposure(KEPLER_LONG, 15)),
        (planet(0.1, 10.0, 1.05), exposure(KEPLER_LONG, 7)),
        (planet(0.1, 10.0, 1.2), exposure(KEPLER_LONG, 5)),
        (planet(0.5, 1.2, 0.3), exposure(KEPLER_LONG, 9)),
        (sun_like(), exposure(1.0, 30)),
    ];
    for (params, exposure) in cases {
        let n = exposure.samples().get() as f64;
        let mean = |t: f64| {
            exposure
                .offsets()
                .map(|dt| params.flux_at(at(t) + dt))
                .sum::<f64>()
                / n
        };
        for i in 0..12_000 {
            let t = T0 + f64::from(i) * 2.5e-4 - 0.75;
            for t in [t, t + 1.3e-7] {
                let binned = params.binned_flux_at(at(t), exposure);
                let expected = mean(t);
                if expected == 1.0 {
                    assert_eq!(binned, 1.0, "{t}");
                }
                assert!(
                    (binned - expected).abs() < 1e-14,
                    "{t}: {binned} {expected}"
                );
                // A whole number of orbits later, at BJD-sized times.
                let later = t + 2_457_000.0;
                let (binned_later, expected_later) =
                    (params.binned_flux_at(at(later), exposure), mean(later));
                assert!((binned_later - binned).abs() < 3e-10, "{later}");
                assert!((expected_later - expected).abs() < 3e-10, "{later}");
            }
        }
    }
}

// Inside the flat bottom of a uniform star's transit the flux is constant, so
// averaging changes nothing.
#[test]
fn flat_bottom_is_unchanged() {
    let params = transit(LimbDarkening::uniform());
    let flat = 1.0 - 0.01;
    for t in [T0 - 0.01, T0, T0 + 0.02] {
        let binned = params.binned_flux_at(at(t), exposure(KEPLER_LONG, 15));
        assert!((binned - flat).abs() < 1e-15, "t {t}: {binned}");
    }
}

#[test]
fn symmetric_about_mid_transit() {
    let params = sun_like();
    let e = exposure(KEPLER_LONG, 15);
    for i in 1..=60 {
        let dt = f64::from(i) * 0.001;
        let before = params.binned_flux_at(at(T0 - dt), e);
        let after = params.binned_flux_at(at(T0 + dt), e);
        assert!((before - after).abs() < 1e-12, "dt {dt}");
    }
}

// Averaging spreads the transit out: shallower in the middle, and dimming
// starts before the instantaneous first contact.
#[test]
fn averaging_makes_the_transit_shallower_and_longer() {
    let params = sun_like();
    let e = exposure(KEPLER_LONG, 30);
    assert!(params.binned_flux_at(at(T0), e) > params.flux_at(at(T0)));

    let t1 = params.t0() + params.contacts().unwrap().t1;
    let early = t1 - days(KEPLER_LONG / 4.0);
    assert_eq!(params.flux_at(early), 1.0);
    assert!(params.binned_flux_at(early, e) < 1.0);
}

// Averaging over a box keeps the area of the transit: the total light lost
// over the whole transit is the same with or without it.
#[test]
fn keeps_the_area_of_the_transit() {
    let params = sun_like();
    let e = exposure(KEPLER_LONG, 15);
    let step = 1e-4;
    let (mut binned, mut instant) = (0.0, 0.0);
    for i in -1500..=1500 {
        let t = at(T0 + f64::from(i) * step);
        binned += (1.0 - params.binned_flux_at(t, e)) * step;
        instant += (1.0 - params.flux_at(t)) * step;
    }
    assert!(
        (binned - instant).abs() < 1e-6 * instant,
        "{binned} vs {instant}"
    );
}

// Kepler long cadence across a 2.4-hour transit: with 1000 samples the
// average agrees with an independent Simpson's-rule integral.
#[test]
fn matches_an_independent_quadrature() {
    let params = sun_like();
    let e = exposure(KEPLER_LONG, 1000);
    for t in across_transit().step_by(5) {
        let binned = params.binned_flux_at(at(t), e);
        let simpson = simpson_average(params, t, KEPLER_LONG);
        assert!(
            (binned - simpson).abs() < 1e-9,
            "t {t}: {binned} vs {simpson}"
        );
    }
}

// Doubling the samples cuts the error by about four: the midpoint rule is
// second order. Checked in mid-ingress, where the flux curves most, with a
// 2-minute exposure that stays clear of the contacts (a longer one spans
// them, and the kink in the flux there slows convergence).
#[test]
fn error_falls_as_one_over_samples_squared() {
    let params = sun_like();
    let contacts = params.contacts().unwrap();
    let t1 = contacts.t1.as_days();
    let t2 = contacts.t2.unwrap().as_days();
    let t = T0 + (t1 + t2) / 2.0;
    let short = 2.0 / 1440.0;
    let exact = simpson_average(params, t, short);
    let error = |n| (params.binned_flux_at(at(t), exposure(short, n)) - exact).abs();
    for n in [5, 10, 20] {
        let ratio = error(n) / error(2 * n);
        assert!((3.0..5.0).contains(&ratio), "{n} samples: ratio {ratio}");
    }
}

#[test]
fn stays_between_zero_and_one() {
    let params = sun_like();
    for n in [1, 2, 3, 15] {
        for t in across_transit() {
            let f = params.binned_flux_at(at(t), exposure(KEPLER_LONG, n));
            assert!((0.0..=1.0).contains(&f), "t {t}, {n} samples: {f}");
        }
    }
}
