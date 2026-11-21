use crate::{
    Epoch, FitError, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio,
    ScaledSemiMajorAxis, TimeDelta, TransitFit, TransitParams,
};

fn start(t0: f64) -> TransitParams {
    TransitParams::new(
        Epoch::from_days(t0).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
}

fn curve(n: u32, step: f64) -> LightCurve {
    let time = (0..n).map(|i| step * f64::from(i)).collect();
    LightCurve::from_days(time, vec![1.0; n as usize], vec![1e-3; n as usize]).unwrap()
}

#[test]
fn checks() {
    let lc = curve(100, 0.01);
    assert_eq!(TransitFit::new(&lc, start(0.5)).check(), Ok(()));
    // No point within the transit (the data span 1 day, the transit 1.5).
    assert_eq!(
        TransitFit::new(&lc, start(1.5)).check(),
        Err(FitError::NoPointsInTransit)
    );
    // The baseline alone needs no transit.
    let mut fit = TransitFit::new(&lc, start(1.5));
    for p in Parameter::ALL {
        fit = fit.with_fixed(p);
    }
    assert_eq!(fit.check(), Err(FitError::NothingToFit));
    assert_eq!(fit.with_free(Parameter::Baseline).check(), Ok(()));
    let few = curve(8, 0.1);
    assert_eq!(
        TransitFit::new(&few, start(0.4)).check(),
        Err(FitError::TooFewPoints { needed: 9 })
    );
}

#[test]
fn exposure_widens_the_transit() {
    use crate::Exposure;
    // The last point (0.99) lies 0.005 d before first contact; an exposure
    // of 0.04 d reaches 0.02 d past it, into the transit.
    let lc = curve(100, 0.01);
    let params = start(0.995 + 0.5 * start(0.0).t14().unwrap().as_days());
    assert_eq!(
        TransitFit::new(&lc, params).check(),
        Err(FitError::NoPointsInTransit)
    );
    let exposure = Exposure::new(
        TimeDelta::from_days(0.04).unwrap(),
        std::num::NonZeroUsize::new(7).unwrap(),
    )
    .unwrap();
    assert_eq!(
        TransitFit::new(&lc, params).with_exposure(exposure).check(),
        Ok(())
    );
}
