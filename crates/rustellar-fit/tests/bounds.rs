// Limb-darkening bounds: a star brightening to its limb, fitted with and
// without Kipping's (2013) triangle.

use rustellar_fit::{
    Epoch, ImpactParameter, LightCurve, LimbDarkening, LimbDarkeningBounds, Parameter, RadiusRatio,
    ScaledSemiMajorAxis, Termination, TimeDelta, TransitFit, TransitParams,
};

fn params(u1: f64, u2: f64) -> TransitParams {
    TransitParams::new(
        Epoch::from_days(1.3).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(u1, u2).unwrap(),
    )
    .unwrap()
}

fn curve(truth: TransitParams) -> LightCurve {
    let time: Vec<f64> = (0..2880).map(|i| f64::from(i) * 2.0 / 1440.0).collect();
    let flux = time
        .iter()
        .map(|&t| truth.flux_at(Epoch::from_days(t).unwrap()))
        .collect();
    LightCurve::from_days(time, flux, vec![1e-4; 2880]).unwrap()
}

#[test]
fn limb_brightening_is_kept_out_by_physical_bounds() {
    // Brightening to the limb: u1 < 0, valid but not physical.
    let truth = params(-0.2, 0.3);
    let lc = curve(truth);
    let start = params(0.4, 0.25);

    let valid = TransitFit::new(&lc, start).run().unwrap();
    assert!((valid.value(Parameter::U1) + 0.2).abs() < 1e-7);
    assert!(valid.chi_squared() < 1e-10);

    let fit = TransitFit::new(&lc, start).with_limb_darkening_bounds(LimbDarkeningBounds::Physical);
    assert_eq!(fit.limb_darkening_bounds(), LimbDarkeningBounds::Physical);
    let physical = fit.run().unwrap();
    // On the edge u1 = 0, a minimum on a bound.
    assert!(matches!(
        physical.termination(),
        Termination::Converged | Termination::PrecisionLimit
    ));
    let (u1, u2) = (physical.value(Parameter::U1), physical.value(Parameter::U2));
    assert!(u1.abs() < 1e-9, "{u1}");
    assert!(u1 + u2 <= 1.0 && u1 + 2.0 * u2 >= 0.0);
    assert!(physical.chi_squared() > 1.0);
}

#[test]
fn a_start_outside_is_moved_inside() {
    let truth = params(0.4, 0.25);
    let lc = curve(truth);
    // u1 alone free, from limb brightening: moved to u1 = 0 first, then
    // fitted to the truth.
    let mut fit = TransitFit::new(&lc, params(-0.3, 0.25))
        .with_limb_darkening_bounds(LimbDarkeningBounds::Physical);
    for p in Parameter::ALL {
        if p != Parameter::U1 {
            fit = fit.with_fixed(p);
        }
    }
    let result = fit.run().unwrap();
    assert_eq!(result.termination(), Termination::Converged);
    assert!((result.value(Parameter::U1) - 0.4).abs() < 1e-9);
    // Inside already, the default bounds change nothing.
    let result = TransitFit::new(&lc, truth).run().unwrap();
    assert!((result.value(Parameter::U2) - 0.25).abs() < 1e-9);
}
