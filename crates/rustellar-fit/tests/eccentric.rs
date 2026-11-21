// Eccentric orbits: noiseless light curves recovered with the eccentricity
// held at its true value, and with e sin w fitted where a / R* is known.

use rustellar_fit::{
    Epoch, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio, ScaledSemiMajorAxis,
    Termination, TimeDelta, TransitFit, TransitParams,
};

fn params(k: f64, a: f64, b: f64, e: f64, w_deg: f64) -> TransitParams {
    TransitParams::new(
        Epoch::from_days(1.0).unwrap(),
        TimeDelta::from_days(5.0).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(a).unwrap(),
        ImpactParameter::new(b).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
    .with_eccentricity(e, w_deg)
    .unwrap()
}

/// Two transits at two-minute cadence, noiseless.
fn curve(truth: TransitParams) -> LightCurve {
    let time: Vec<f64> = (0..7200).map(|i| f64::from(i) / 720.0).collect();
    let flux = time
        .iter()
        .map(|&t| truth.flux_at(Epoch::from_days(t).unwrap()))
        .collect();
    LightCurve::from_days(time, flux, vec![1e-4; 7200]).unwrap()
}

#[test]
fn shape_with_the_eccentricity_held() {
    let truth = params(0.1, 15.0, 0.4, 0.3, 60.0);
    let lc = curve(truth);
    // Every other parameter off, the eccentricity at its true value.
    let start = params(0.105, 14.0, 0.5, 0.3, 60.0);
    let result = TransitFit::new(&lc, start).run().unwrap();
    assert_eq!(result.termination(), Termination::Converged);
    assert!(result.chi_squared() < 1e-12, "{}", result.chi_squared());
    let p = result.params();
    assert!((p.k().get() - 0.1).abs() < 1e-9);
    assert!((p.a_rs().get() - 15.0).abs() < 1e-7);
    assert!((p.b().get() - 0.4).abs() < 1e-7);
    assert!((p.eccentricity() - 0.3).abs() < 1e-15);
    // The durations follow the orbit.
    let t14 = result.params().t14().unwrap().as_days();
    assert!((t14 - truth.t14().unwrap().as_days()).abs() < 1e-9);
}

#[test]
fn e_sin_w_with_a_known() {
    // With a / R* known (from the star's density, say), the duration gives
    // e sin w; e cos w barely shows in a transit, so it stays fixed.
    let truth = params(0.1, 15.0, 0.4, 0.3, 60.0);
    let lc = curve(truth);
    let e_cos = 0.3 * 60f64.to_radians().cos();
    let start = params(0.102, 15.0, 0.45, 0.15, 0.0)
        .with_eccentricity(e_cos.hypot(0.1), 0.1f64.atan2(e_cos).to_degrees())
        .unwrap();
    let result = TransitFit::new(&lc, start)
        .with_fixed(Parameter::SemiMajorAxis)
        .with_free(Parameter::ESinW)
        .run()
        .unwrap();
    assert_eq!(result.termination(), Termination::Converged);
    let e_sin = 0.3 * 60f64.to_radians().sin();
    let have = result.estimate(Parameter::ESinW);
    assert!((have.value() - e_sin).abs() < 1e-7, "{have}");
    assert!(have.error() > 0.0 && have.error() < 0.01);
    assert!((result.value(Parameter::ECosW) - e_cos).abs() < 1e-15);
    assert!((result.params().eccentricity() - 0.3).abs() < 1e-7);
    assert!((result.params().periastron_deg() - 60.0).abs() < 1e-5);
}

#[test]
fn from_a_circular_start() {
    // e cos w and e sin w both free from zero, a known: the fit leaves the
    // circular orbit, where w is undefined, without trouble.
    let truth = params(0.1, 15.0, 0.4, 0.2, 100.0);
    let lc = curve(truth);
    let start = TransitParams::new(
        Epoch::from_days(1.0).unwrap(),
        TimeDelta::from_days(5.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(15.0).unwrap(),
        ImpactParameter::new(0.4).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap();
    let result = TransitFit::new(&lc, start)
        .with_fixed(Parameter::SemiMajorAxis)
        .with_fixed(Parameter::U1)
        .with_fixed(Parameter::U2)
        .with_free(Parameter::ECosW)
        .with_free(Parameter::ESinW)
        .run()
        .unwrap();
    assert_eq!(result.termination(), Termination::Converged);
    let (w_sin, w_cos) = 100f64.to_radians().sin_cos();
    assert!((result.value(Parameter::ECosW) - 0.2 * w_cos).abs() < 1e-6);
    assert!((result.value(Parameter::ESinW) - 0.2 * w_sin).abs() < 1e-6);
    // Found, as the curve is noiseless, but with e cos w hardly
    // constrained: at 1e-4 a point its standard error is far above 1.
    assert!(result.estimate(Parameter::ECosW).error() > 1.0);
    assert!(result.chi_squared() < 1e-9, "{}", result.chi_squared());
}
