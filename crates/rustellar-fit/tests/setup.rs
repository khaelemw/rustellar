// The set-up's API: defaults, builders and errors.

use rustellar_fit::{
    Epoch, Exposure, FitError, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio,
    ScaledSemiMajorAxis, TimeDelta, TransitFit, TransitParams,
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

#[test]
fn defaults_and_builders() {
    let lc = LightCurve::from_days(vec![0.0, 0.5, 1.0], vec![1.0; 3], vec![1e-3; 3]).unwrap();
    let fit = TransitFit::new(&lc, start());
    // Every parameter but the eccentricity's.
    let eccentric = [Parameter::ECosW, Parameter::ESinW];
    let circular: Vec<Parameter> = Parameter::ALL
        .into_iter()
        .filter(|p| !eccentric.contains(p))
        .collect();
    assert_eq!(fit.free(), circular);
    let every = fit.clone().with_free(eccentric[0]).with_free(eccentric[1]);
    assert_eq!(every.free(), Parameter::ALL.to_vec());
    assert_eq!(fit.baseline(), 1.0);
    assert_eq!(fit.start(), start());
    assert_eq!(fit.exposure(), None);
    assert_eq!(fit.max_iterations(), 200);
    assert_eq!(fit.light_curve().len(), 3);
    let exposure = Exposure::new(
        TimeDelta::from_days(0.02).unwrap(),
        std::num::NonZeroUsize::new(5).unwrap(),
    )
    .unwrap();
    let fit = fit
        .with_fixed(Parameter::Period)
        .with_fixed(Parameter::U2)
        .with_free(Parameter::U2)
        .with_exposure(exposure)
        .with_max_iterations(10)
        .with_baseline(1.5)
        .unwrap();
    assert!(!fit.is_free(Parameter::Period) && fit.is_free(Parameter::U2));
    assert_eq!(fit.free().len(), 7);
    assert_eq!(fit.baseline(), 1.5);
    assert_eq!(fit.exposure(), Some(exposure));
    assert_eq!(fit.max_iterations(), 10);
}

#[test]
fn bad_baselines_and_messages() {
    let lc = LightCurve::from_days(vec![0.0, 1.0], vec![1.0; 2], vec![1e-3; 2]).unwrap();
    let fit = TransitFit::new(&lc, start());
    assert_eq!(
        fit.clone().with_baseline(f64::NAN).unwrap_err(),
        FitError::NonFinite { name: "baseline" }
    );
    assert_eq!(
        fit.with_baseline(0.0).unwrap_err(),
        FitError::NotPositive { name: "baseline" }
    );
    for (err, text) in [
        (
            FitError::NonFinite { name: "baseline" },
            "baseline must be finite",
        ),
        (
            FitError::NotPositive { name: "baseline" },
            "baseline must be greater than zero",
        ),
        (
            FitError::NothingToFit,
            "at least one parameter must be free",
        ),
        (
            FitError::TooFewPoints { needed: 9 },
            "at least 9 points are needed",
        ),
        (
            FitError::NoPointsInTransit,
            "the starting parameters put no point in transit",
        ),
    ] {
        assert_eq!(err.to_string(), text);
    }
}

#[test]
fn parameter_names() {
    let names: Vec<String> = Parameter::ALL.iter().map(ToString::to_string).collect();
    assert_eq!(
        names,
        [
            "t0", "P", "k", "a_rs", "b", "u1", "u2", "e_cos_w", "e_sin_w", "baseline"
        ]
    );
    for (i, p) in Parameter::ALL.into_iter().enumerate() {
        assert_eq!(p.index(), i);
    }
}
