// Averaging over an exposure: the angles it turns the time stamp's phase
// through are the exposure's own offsets, as angles of the orbit.

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use rustellar_core::{Epoch, TimeDelta};

use crate::params::Averaging;
use crate::{
    Exposure, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

fn hot_jupiter() -> TransitParams {
    TransitParams::new(
        Epoch::from_btjd(0.0).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
}

// The non-negative angles the averaging visits, in order, from its first
// angle and its step, as `binned_flux_with` turns them.
fn angles(averaging: &Averaging) -> Vec<(f64, f64)> {
    let (step_sin, step_cos) = averaging.step;
    let (mut sin_theta, mut cos_theta) = averaging.first;
    let mut out = Vec::new();
    if averaging.samples % 2 == 1 {
        out.push((0.0, 1.0));
        (sin_theta, cos_theta) = (step_sin, step_cos);
    }
    for _ in 0..averaging.samples / 2 {
        out.push((sin_theta, cos_theta));
        (sin_theta, cos_theta) = (
            sin_theta.mul_add(step_cos, cos_theta * step_sin),
            cos_theta.mul_add(step_cos, -(sin_theta * step_sin)),
        );
    }
    out
}

#[test]
fn the_angles_are_the_exposures_offsets() {
    let params = hot_jupiter();
    for (hours, samples) in [(0.49, 1), (0.49, 2), (0.49, 15), (0.49, 30), (24.0, 31)] {
        let exposure = Exposure::new(
            TimeDelta::from_hours(hours).unwrap(),
            NonZeroUsize::new(samples).unwrap(),
        )
        .unwrap();
        let averaging = params.averaging(exposure);
        assert_eq!(averaging.samples, samples);
        // The offsets at and after the time stamp, as angles of the orbit.
        let expected: Vec<f64> = exposure
            .offsets()
            .map(|dt| 2.0 * PI * (dt / params.period()))
            .filter(|&angle| angle >= 0.0)
            .collect();
        let found = angles(&averaging);
        assert_eq!(found.len(), expected.len());
        for ((sin_theta, cos_theta), angle) in found.into_iter().zip(expected) {
            let (s, c) = angle.sin_cos();
            assert!((sin_theta - s).abs() < 1e-15, "{samples}: {angle}");
            assert!((cos_theta - c).abs() < 1e-15, "{samples}: {angle}");
        }
    }
}

// Past `clear` from mid-transit, every instant of the exposure is out of
// transit; inside it, the exposure's nearest instant can reach the transit.
#[test]
fn clear_is_half_the_transit_and_half_the_exposure() {
    let params = hot_jupiter();
    let exposure = Exposure::new(
        TimeDelta::from_hours(0.49).unwrap(),
        NonZeroUsize::new(15).unwrap(),
    )
    .unwrap();
    let averaging = params.averaging(exposure);
    let t14 = params.t14().unwrap();
    let half = |dt: TimeDelta| PI * (dt / params.period());
    let expected = half(t14) + half(exposure.duration());
    assert!(
        (averaging.clear - expected).abs() < 2e-6,
        "{}",
        averaging.clear
    );
    assert!(averaging.clear > expected);
}

// A light curve leaves out the sine and cosine beyond the transit's angle,
// where the flux is exactly 1: each value is `flux_at`'s to the bit,
// through first and last contact, for a planet that misses the star, and
// for one so close that the transit spans half the orbit.
#[test]
fn light_curves_are_flux_at_to_the_bit() {
    let at = |a_rs: f64, b: f64| {
        TransitParams::new(
            Epoch::from_btjd(0.0).unwrap(),
            TimeDelta::from_days(3.0).unwrap(),
            RadiusRatio::new(0.1).unwrap(),
            ScaledSemiMajorAxis::new(a_rs).unwrap(),
            ImpactParameter::new(b).unwrap(),
            LimbDarkening::quadratic(0.4, 0.25).unwrap(),
        )
        .unwrap()
    };
    for params in [hot_jupiter(), at(10.0, 1.2), at(1.05, 0.0)] {
        let edge = params.clear() / (2.0 * PI) * 3.0;
        let mut times: Vec<Epoch> = (0..=3000)
            .map(|i| Epoch::from_btjd(-3.0 + 2e-3 * f64::from(i)).unwrap())
            .collect();
        for offset in [-1e-9, 0.0, 1e-9] {
            for t in [edge + offset, -edge - offset] {
                times.push(Epoch::from_btjd(t).unwrap());
            }
        }
        let expected: Vec<f64> = times.iter().map(|&t| params.flux_at(t)).collect();
        assert_eq!(params.light_curve(&times), expected);
        let mut out = vec![0.0; times.len()];
        params.light_curve_into(&times, &mut out).unwrap();
        assert_eq!(out, expected);
        let mut model = crate::TransitModel::new(times);
        assert_eq!(model.light_curve(params), expected.as_slice());
    }
}
