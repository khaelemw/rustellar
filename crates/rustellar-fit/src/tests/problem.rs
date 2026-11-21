use crate::lm::Residuals;
use crate::problem::{Problem, onto_inner_edge, onto_physical, params_from, values_of};
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

#[test]
fn onto_the_inner_edge() {
    // Points where the intensity is negative inside the disc (u2 < -1,
    // u1 > 2 sqrt(-u2)), and one past both edges.
    let lc = curve();
    let fit = TransitFit::new(&lc, start());
    let mut p = Problem::new(&fit, values_of(start(), 1.0));
    let mut out = vec![0.0; 100];
    for (u1, u2) in [
        (3.0, -1.5),
        (2.6, -1.5),
        (10.0, -4.0),
        (2.05, -1.01),
        (5.0, -3.0),
    ] {
        let mut x = values_of(start(), 1.0);
        (x[5], x[6]) = (u1, u2);
        p.project(&mut x);
        let (v1, v2) = (x[5], x[6]);
        assert!(p.eval(&x, &mut out), "({u1}, {u2}) to ({v1}, {v2})");
        assert!(v2 <= -1.0 && (v1 * v1 + 4.0 * v2).abs() < 1e-12);
    }
    // The nearest point of the curve, against a fine sample of it.
    for (u1, u2) in [(3.0, -1.5), (10.0, -4.0), (2.05, -1.01)] {
        let (v1, v2) = onto_inner_edge(u1, u2);
        let d = (v1 - u1).hypot(v2 - u2);
        let nearest = (0..400_000)
            .map(|i| 1.0 + 1e-5 * f64::from(i))
            .map(|t| (2.0 * t - u1).hypot(-t * t - u2))
            .fold(f64::INFINITY, f64::min);
        assert!(
            d <= nearest + 1e-9 && d >= nearest - 1e-6,
            "{d} against {nearest}"
        );
    }
}

#[test]
fn onto_kippings_triangle() {
    let both = (true, true);
    // Inside: unmoved.
    assert_eq!(onto_physical(0.4, 0.25, both), (0.4, 0.25));
    // Nearest the corners.
    assert_eq!(onto_physical(-1.0, -1.0, both), (0.0, 0.0));
    assert_eq!(onto_physical(3.0, -2.0, both), (2.0, -1.0));
    assert_eq!(onto_physical(-1.0, 2.0, both), (0.0, 1.0));
    // Onto each edge, square to it.
    let close =
        |(a, b): (f64, f64), (c, d): (f64, f64)| (a - c).abs() < 1e-15 && (b - d).abs() < 1e-15;
    assert!(close(onto_physical(-0.2, 0.3, both), (0.0, 0.3)));
    assert!(close(onto_physical(1.0, 1.0, both), (0.5, 0.5)));
    // u1 + 2 u2 = 0: (1, -1) to (1.2, -0.6).
    assert!(close(onto_physical(1.0, -1.0, both), (1.2, -0.6)));
    // One free: along the chord through the fixed one.
    assert_eq!(onto_physical(-0.2, 0.3, (true, false)), (0.0, 0.3));
    assert_eq!(onto_physical(0.9, 0.3, (true, false)), (0.7, 0.3));
    assert_eq!(onto_physical(0.5, -0.4, (true, false)), (0.8, -0.4));
    assert_eq!(onto_physical(0.4, 0.9, (false, true)), (0.4, 0.6));
    assert_eq!(onto_physical(0.4, -0.5, (false, true)), (0.4, -0.2));
    // No chord through the fixed one: unmoved.
    assert_eq!(onto_physical(-0.5, 1.5, (true, false)), (-0.5, 1.5));
    assert_eq!(onto_physical(-0.5, 0.2, (false, true)), (-0.5, 0.2));
    assert_eq!(onto_physical(2.5, 0.2, (false, true)), (2.5, 0.2));
    assert_eq!(onto_physical(-0.5, 0.2, (false, false)), (-0.5, 0.2));
}

#[test]
fn bounds_on_the_free_coefficients() {
    use crate::{LimbDarkeningBounds, Parameter};
    let lc = curve();
    let at = |u1: f64, u2: f64| {
        let mut v = values_of(start(), 1.0);
        v[5] = u1;
        v[6] = u2;
        v
    };
    // On u1 + u2 = 1 with u2 fixed: the normal is u1's alone.
    let fit = TransitFit::new(&lc, start()).with_fixed(Parameter::U2);
    let p = Problem::new(&fit, at(0.6, 0.4));
    let bounds = p.bounds(&p.free_values());
    assert_eq!(bounds.len(), 1);
    assert_eq!(bounds[0][5], -1.0);
    assert_eq!(bounds[0].iter().filter(|&&g| g != 0.0).count(), 1);
    // With u1 fixed instead, u2's alone.
    let fit = TransitFit::new(&lc, start()).with_fixed(Parameter::U1);
    let p = Problem::new(&fit, at(0.6, 0.4));
    let bounds = p.bounds(&p.free_values());
    assert_eq!(bounds[0][5], -1.0);
    // Both fixed: no bound to keep to.
    let fit = TransitFit::new(&lc, start())
        .with_fixed(Parameter::U1)
        .with_fixed(Parameter::U2);
    let p = Problem::new(&fit, at(0.6, 0.4));
    assert!(p.bounds(&p.free_values()).is_empty());
    // The triangle's corner (0, 0): two bounds.
    let fit =
        TransitFit::new(&lc, start()).with_limb_darkening_bounds(LimbDarkeningBounds::Physical);
    let p = Problem::new(&fit, at(0.0, 0.0));
    assert_eq!(p.bounds(&p.free_values()).len(), 2);
    // Inside: none.
    let p = Problem::new(&fit, at(0.4, 0.25));
    assert!(p.bounds(&p.free_values()).is_empty());
}
