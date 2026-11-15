// Tests for `TransitModel` and the `light_curve_into` methods.

use std::num::NonZeroUsize;

use rustellar_core::{Epoch, LightCurve, TimeDelta};
use rustellar_model::{
    Exposure, ImpactParameter, LengthMismatch, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis,
    TransitModel, TransitParams,
};

// ==========================
// Helper Functions
// ==========================

const T0: f64 = 3_000.123_456;

fn at(t: f64) -> Epoch {
    Epoch::from_btjd(t).unwrap()
}

fn params(k: f64, b: f64) -> TransitParams {
    TransitParams::new(
        at(T0),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(b).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
}

// Unsorted on purpose, with points in and out of transit.
fn times() -> Vec<Epoch> {
    [0.03, -0.05, 0.0, 0.2, -0.01, 1.5, 0.045]
        .map(|dt| at(T0 + dt))
        .to_vec()
}

fn kepler_long() -> Exposure {
    Exposure::new(
        TimeDelta::from_hours(29.4 / 60.0).unwrap(),
        NonZeroUsize::new(15).unwrap(),
    )
    .unwrap()
}

// ==========================
// Tests
// ==========================

#[test]
fn model_keeps_its_times_and_exposure() {
    let plain = TransitModel::new(times());
    assert_eq!(plain.times(), times().as_slice());
    assert_eq!(plain.exposure(), None);

    let binned = TransitModel::with_exposure(times(), kepler_long());
    assert_eq!(binned.exposure(), Some(kepler_long()));
}

#[test]
fn model_without_exposure_matches_light_curve() {
    let mut model = TransitModel::new(times());
    let p = params(0.1, 0.3);
    assert_eq!(model.light_curve(p), p.light_curve(&times()).as_slice());
}

#[test]
fn model_with_exposure_matches_binned_flux() {
    let mut model = TransitModel::with_exposure(times(), kepler_long());
    let p = params(0.1, 0.3);
    let expected: Vec<f64> = times()
        .into_iter()
        .map(|t| p.binned_flux_at(t, kepler_long()))
        .collect();
    assert_eq!(model.light_curve(p), expected.as_slice());
}

// Each call depends only on its own parameters, not on the call before.
#[test]
fn model_can_be_reused_for_many_parameters() {
    let mut model = TransitModel::with_exposure(times(), kepler_long());
    let sets = [params(0.1, 0.3), params(0.05, 0.9), params(0.2, 0.0)];
    let first: Vec<Vec<f64>> = sets
        .iter()
        .map(|&p| model.light_curve(p).to_vec())
        .collect();
    for (p, expected) in sets.iter().zip(&first).rev() {
        assert_eq!(model.light_curve(*p), expected.as_slice());
    }
}

#[test]
fn model_into_matches_model() {
    let mut model = TransitModel::with_exposure(times(), kepler_long());
    let p = params(0.1, 0.3);
    let mut out = vec![0.0; times().len()];
    model.light_curve_into(p, &mut out).unwrap();
    assert_eq!(out.as_slice(), model.light_curve(p));
}

#[test]
fn params_into_matches_light_curve() {
    let p = params(0.1, 0.3);
    let mut out = vec![0.0; times().len()];
    p.light_curve_into(&times(), &mut out).unwrap();
    assert_eq!(out, p.light_curve(&times()));
}

#[test]
fn wrong_output_length_is_an_error_and_leaves_output_alone() {
    let p = params(0.1, 0.3);
    let model = TransitModel::new(times());
    for len in [0, 6, 8] {
        let mut out = vec![-1.0; len];
        let check = |error: LengthMismatch| {
            assert_eq!((error.times, error.output), (7, len));
        };
        check(p.light_curve_into(&times(), &mut out).unwrap_err());
        check(model.light_curve_into(p, &mut out).unwrap_err());
        assert!(out.iter().all(|&f| f == -1.0));
    }
}

#[test]
fn length_mismatch_message() {
    let p = params(0.1, 0.3);
    let error = p.light_curve_into(&times(), &mut [0.0; 2]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "output has length 2 but there are 7 times"
    );
}

#[test]
fn no_times_gives_an_empty_light_curve() {
    let mut model = TransitModel::new(Vec::new());
    assert!(model.light_curve(params(0.1, 0.3)).is_empty());
    assert_eq!(params(0.1, 0.3).light_curve_into(&[], &mut []), Ok(()));
}

#[test]
fn model_takes_times_from_a_light_curve() {
    let lc = LightCurve::from_btjd(
        vec![T0 - 0.05, T0, T0 + 0.05],
        vec![1.0, 0.99, 1.0],
        vec![1e-4; 3],
    )
    .unwrap();
    let mut model = TransitModel::new(lc.time().to_vec());
    let p = params(0.1, 0.3);
    assert_eq!(model.light_curve(p), p.light_curve(lc.time()).as_slice());
}
