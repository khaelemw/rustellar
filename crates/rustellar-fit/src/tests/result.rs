use crate::lm::Solution;
use crate::problem::values_of;
use crate::{
    Epoch, FitResult, ImpactParameter, LimbDarkening, Parameter, RadiusRatio, ScaledSemiMajorAxis,
    Termination, TimeDelta, TransitParams,
};

fn start() -> TransitParams {
    TransitParams::new(
        Epoch::from_days(0.5).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
}

/// A result over every parameter with this `R` (8 by 8) and these steps.
fn result(values: [f64; 8], r: Vec<f64>, steps: Vec<f64>) -> FitResult {
    let solution = Solution {
        x: values.to_vec(),
        f: vec![0.0; 20],
        r,
        iterations: 1,
        evaluations: 1,
        termination: Termination::Converged,
    };
    FitResult::new(values, Parameter::ALL.to_vec(), steps, solution)
}

/// Upper triangular, with ones above a diagonal of tens.
fn coupled() -> Vec<f64> {
    (0..64)
        .map(|i| match (i / 8, i % 8) {
            (r, c) if r == c => 10.0,
            (r, c) if r < c => 1.0,
            _ => 0.0,
        })
        .collect()
}

#[test]
fn negative_b_is_reported_as_its_size() {
    let mut values = values_of(start(), 1.0);
    let plus = result(values, coupled(), vec![1e-6; 8]);
    values[4] = -0.3;
    let minus = result(values, coupled(), vec![1e-6; 8]);
    assert_eq!(minus.value(Parameter::ImpactParameter), 0.3);
    assert_eq!(minus.params(), plus.params());
    let (b, k) = (Parameter::ImpactParameter, Parameter::RadiusRatio);
    let (cp, cm) = (plus.covariance().unwrap(), minus.covariance().unwrap());
    assert!(cp.get(b, k) != 0.0);
    assert_eq!(cm.get(b, k), -cp.get(b, k));
    assert_eq!(cm.get(k, k), cp.get(k, k));
}

#[test]
fn unconstrained() {
    let r = vec![0.0; 64];
    let res = result(values_of(start(), 1.0), r, vec![1e-6; 8]);
    assert!(res.covariance().is_none());
    assert_eq!(res.estimate(Parameter::RadiusRatio).error(), f64::INFINITY);
    assert_eq!(res.depth().error(), f64::INFINITY);
}

#[test]
fn one_sided_and_missing_gradients() {
    let mut steps = vec![1e-6; 8];
    // k beyond its range either way, a_rs below 1 downwards, u1 past the
    // limb's zero upwards.
    steps[2] = 1e5;
    steps[3] = 9.5;
    steps[5] = 1.0;
    let res = result(values_of(start(), 1.0), coupled(), steps);
    let depth = res.depth();
    assert!(depth.error().is_finite() && depth.error() > 0.0);
}
