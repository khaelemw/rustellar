// Core's types re-exported here are core's, not copies: a value made
// through one path is accepted through the other.

use std::any::TypeId;

#[test]
fn core_types_are_re_exported_unchanged() {
    for (here, core) in [
        (
            TypeId::of::<rustellar_bls::LightCurve>(),
            TypeId::of::<rustellar_core::LightCurve>(),
        ),
        (
            TypeId::of::<rustellar_bls::FoldedLightCurve>(),
            TypeId::of::<rustellar_core::FoldedLightCurve>(),
        ),
        (
            TypeId::of::<rustellar_bls::Epoch>(),
            TypeId::of::<rustellar_core::Epoch>(),
        ),
        (
            TypeId::of::<rustellar_bls::TimeDelta>(),
            TypeId::of::<rustellar_core::TimeDelta>(),
        ),
        (
            TypeId::of::<rustellar_bls::LightCurveError>(),
            TypeId::of::<rustellar_core::LightCurveError>(),
        ),
        (
            TypeId::of::<rustellar_bls::TimeError>(),
            TypeId::of::<rustellar_core::TimeError>(),
        ),
    ] {
        assert_eq!(here, core);
    }
    let made_here =
        rustellar_bls::LightCurve::from_days(vec![0.0, 1.0], vec![1.0; 2], vec![1e-3; 2]);
    let _: rustellar_core::LightCurve = made_here.unwrap();
}
