use crate::lm::Residuals;
use crate::problem::{Problem, params_from, values_of};
use crate::{
    Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TimeDelta,
    TransitFit, TransitParams,
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

fn curve() -> LightCurve {
    let time: Vec<f64> = (0..100).map(|i| 0.01 * f64::from(i)).collect();
    LightCurve::from_days(time, vec![1.0; 100], vec![1e-3; 100]).unwrap()
}

#[test]
fn projects_onto_the_feasible_region() {
    let lc = curve();
    let fit = TransitFit::new(&lc, start());
    let mut p = Problem::new(&fit, values_of(start(), 1.0));

    let mut x = [0.5, -1.0, -0.1, 0.5, -3.0, 0.9, 0.3, -2.0];
    p.project(&mut x);
    let [_, period, k, a_rs, b, u1, u2, baseline] = x;
    assert!(period > 0.0 && k > 0.0 && a_rs > 1.0 && baseline > 0.0);
    // b keeps its sign, inside the orbit.
    assert!(b < 0.0 && b.abs() < a_rs);
    // The excess of u1 + u2 over 1 taken off both equally.
    assert!((u1 - 0.8).abs() < 1e-15 && u1 + u2 <= 1.0);
    let mut out = vec![0.0; 100];
    assert!(p.eval(&x, &mut out));

    // A point already feasible is left alone.
    let mut y = values_of(start(), 1.0);
    p.project(&mut y);
    assert_eq!(y, values_of(start(), 1.0));
    // Onto the edge without rounding past it.
    for (u1, u2) in [
        (0.1 + 0.2, 0.7 + 3e-16),
        (-0.3, 1.7),
        (1.9, 0.1),
        (1e-17, 1.0 + 1e-15),
    ] {
        let mut z = values_of(start(), 1.0);
        (z[5], z[6]) = (u1, u2);
        p.project(&mut z);
        assert!(z[5] + z[6] <= 1.0 && (z[5] + z[6] - 1.0).abs() < 1e-15);
        assert!(p.eval(&z, &mut out));
    }
}

#[test]
fn infeasible_points() {
    let lc = curve();
    let fit = TransitFit::new(&lc, start());
    let mut p = Problem::new(&fit, values_of(start(), 1.0));
    let mut out = vec![0.0; 100];
    let good = values_of(start(), 1.0);
    for (i, bad) in [(7, 0.0), (2, -0.1), (3, 0.5), (5, 2.0), (1, 0.0)] {
        let mut x = good;
        x[i] = bad;
        assert!(!p.eval(&x, &mut out), "{i}");
    }
    // A negative b is |b|.
    let mut x = good;
    x[4] = -0.3;
    assert_eq!(params_from(&x), Some(start()));
}
