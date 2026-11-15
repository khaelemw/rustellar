// Checks that each feature brings in its crate, and that the types are the
// same ones the crate itself exports, so values pass freely between code using
// `rustellar` and code using a tool crate directly.

use rustellar::{Epoch, LightCurve, TimeDelta};

// ==========================
// Tests
// ==========================

#[test]
fn core_types_are_always_available() {
    let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![1.0, 1.0], vec![0.1, 0.1]).unwrap();
    let folded = lc
        .fold(
            TimeDelta::from_days(2.0).unwrap(),
            Epoch::from_btjd(0.0).unwrap(),
        )
        .unwrap();
    assert_eq!(folded.phase(), &[-0.5, 0.0]);
}

#[test]
fn core_types_are_the_core_crate_types() {
    let t: rustellar_core::Epoch = Epoch::from_btjd(1.0).unwrap();
    assert_eq!(t.as_btjd(), 1.0);
}

#[cfg(feature = "model")]
#[test]
fn model_feature_brings_in_the_model() {
    use rustellar::model::{
        ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    };

    let params: rustellar_model::TransitParams = TransitParams::new(
        Epoch::from_btjd(0.0).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::uniform(),
    )
    .unwrap();
    // A uniform disk: the depth is the planet's area, k^2.
    let flux = params.flux_at(Epoch::from_btjd(0.0).unwrap());
    assert!((flux - 0.99).abs() < 1e-15, "{flux}");
}
