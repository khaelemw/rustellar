use std::num::NonZeroUsize;

use crate::lm::Residuals;
use crate::parameter::COUNT;
use crate::problem::{Problem, values_of};
use crate::{
    Epoch, Exposure, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis,
    TimeDelta, TransitFit, TransitParams,
};

fn params(b: f64) -> TransitParams {
    TransitParams::new(
        Epoch::from_days(0.5).unwrap(),
        TimeDelta::from_days(2.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(6.0).unwrap(),
        ImpactParameter::new(b).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
}

/// A whole orbit, so in transit, out of it and behind the star; 0.5
/// exactly at mid-transit.
fn curve() -> LightCurve {
    let time: Vec<f64> = (0..2000).map(|i| 0.001 * f64::from(i)).collect();
    LightCurve::from_days(time, vec![1.0; 2000], vec![1e-3; 2000]).unwrap()
}

/// The largest gap between the geometric Jacobian and the differenced one
/// (central, steps of 1e-7 of each value's scale), against each column's
/// largest entry.
fn worst(fit: &TransitFit<'_>, values: [f64; COUNT]) -> f64 {
    let mut p = Problem::new(fit, values);
    let x = p.free_values();
    let (m, n) = (p.len(), x.len());
    let mut jac = vec![0.0; m * n];
    assert!(p.jacobian(&x, &mut jac));
    let (mut up, mut down) = (vec![0.0; m], vec![0.0; m]);
    let mut gap: f64 = 0.0;
    for j in 0..n {
        let h = 1e-7 * x[j].abs().max(0.1);
        let mut xs = x.clone();
        xs[j] = x[j] + h;
        assert!(p.eval(&xs, &mut up));
        xs[j] = x[j] - h;
        assert!(p.eval(&xs, &mut down));
        let column = &jac[j * m..(j + 1) * m];
        let scale = column.iter().fold(0.0f64, |s, c| s.max(c.abs()));
        // The b column is zero at b = 0: the flux goes with b^2.
        let scale = scale.max(1.0);
        for (i, c) in column.iter().enumerate() {
            let fd = (up[i] - down[i]) / (2.0 * h);
            gap = gap.max((c - fd).abs() / scale);
        }
    }
    gap
}

#[test]
fn matches_differencing() {
    let lc = curve();
    for b in [0.0, 0.3, 1.0] {
        let fit = TransitFit::new(&lc, params(b));
        // The sign of b carried through.
        for sign in [1.0, -1.0] {
            let mut values = values_of(params(b), 1.01);
            values[4] *= sign;
            let gap = worst(&fit, values);
            assert!(gap < 1e-6, "b {b}: {gap}");
        }
    }
}

#[test]
fn matches_differencing_over_an_exposure() {
    let lc = curve();
    let exposure = Exposure::new(
        TimeDelta::from_days(0.02).unwrap(),
        NonZeroUsize::new(5).unwrap(),
    )
    .unwrap();
    let fit = TransitFit::new(&lc, params(0.3)).with_exposure(exposure);
    let gap = worst(&fit, values_of(params(0.3), 1.0));
    // A sample within a step of a contact point, where the flux's
    // curvature is singular, differs most.
    assert!(gap < 1e-5, "{gap}");
}

#[test]
fn falls_back_at_the_largest_radius_ratio() {
    let lc = curve();
    let fit = TransitFit::new(&lc, params(0.3));
    let mut values = values_of(params(0.3), 1.0);
    values[2] = RadiusRatio::MAX;
    let mut p = Problem::new(&fit, values);
    let x = p.free_values();
    let mut jac = vec![0.0; p.len() * x.len()];
    assert!(!p.jacobian(&x, &mut jac));
}

#[test]
fn matches_differencing_in_a_grazing_close_orbit() {
    // a / R* inside the transit's reach: in front of the star half the
    // orbit, so no phase is clear of it. The flux steps where the planet
    // passes behind, a quarter of an orbit from mid-transit, so t0 keeps
    // that off the points.
    let close = TransitParams::new(
        Epoch::from_days(0.500_3).unwrap(),
        TimeDelta::from_days(2.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(1.05).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap();
    let lc = curve();
    let fit = TransitFit::new(&lc, close);
    let gap = worst(&fit, values_of(close, 1.0));
    assert!(gap < 1e-5, "{gap}");
}
