// The shapes: the box's edges, and the limb-darkened transit against the
// model's own transit on the orbit the shape chose.

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

use crate::shape::Profile;
use crate::{Shape, VetError};

fn transit(k: f64, b: f64) -> Shape {
    Shape::transit(
        RadiusRatio::new(k).unwrap(),
        ImpactParameter::new(b).unwrap(),
        LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
    )
    .unwrap()
}

#[test]
fn the_box_is_one_strictly_inside() {
    let p = Shape::BOX.profile(3.0, 0.25).unwrap();
    assert_eq!(p.at(0.0), 1.0);
    assert_eq!(p.at(0.124_999), 1.0);
    assert_eq!(p.at(-0.124_999), 1.0);
    assert_eq!(p.at(0.125), 0.0);
    assert_eq!(p.at(-0.125), 0.0);
    assert_eq!(p.at(1.0), 0.0);
}

// The orbit chosen gives the model a first-to-last contact of exactly the
// duration asked for, and the model's flux on it is the shape times the
// central depth, at every phase.
#[test]
fn the_transit_is_the_models_on_its_orbit() {
    for (k, b, period, duration) in [
        (0.05, 0.5, 3.0, 0.12),
        (0.1, 0.0, 1.0, 0.08),
        (0.02, 0.9, 30.0, 0.3),
        (0.2, 1.1, 5.0, 0.1),
        (0.05, 0.3, 2.0, 0.99),
    ] {
        let ld = LimbDarkening::quadratic(0.4804, 0.1867).unwrap();
        let shape = transit(k, b);
        let Profile::Transit { a2, centre, .. } = shape.profile(period, duration).unwrap() else {
            panic!("a transit");
        };
        let params = TransitParams::new(
            Epoch::from_days(0.0).unwrap(),
            TimeDelta::from_days(period).unwrap(),
            RadiusRatio::new(k).unwrap(),
            ScaledSemiMajorAxis::new(a2.sqrt()).unwrap(),
            ImpactParameter::new(b).unwrap(),
            ld,
        )
        .unwrap();
        let t14 = params.t14().unwrap().as_days();
        assert!((t14 - duration).abs() < 1e-13 * duration, "{k} {b}: {t14}");
        let profile = shape.profile(period, duration).unwrap();
        for i in -60..=60 {
            let phase = f64::from(i) / 100.0 * duration;
            let model = 1.0 - params.flux_at(Epoch::from_days(phase).unwrap());
            let ours = profile.at(phase) * centre;
            assert!(
                (model - ours).abs() < 1e-15,
                "{k} {b} {phase}: {model} {ours}"
            );
        }
        assert_eq!(profile.at(0.0), 1.0);
    }
}

// From 1 at mid-transit, down to 0 at the contacts, the same either side.
#[test]
fn the_transit_falls_to_its_edges() {
    let profile = Shape::default().profile(3.0, 0.12).unwrap();
    let mut last = 1.0;
    for i in 0..=60 {
        let phase = 0.06 * f64::from(i) / 60.0;
        let s = profile.at(phase);
        assert_eq!(s, profile.at(-phase));
        assert!(s <= last && s >= 0.0, "{phase}: {s}");
        last = s;
    }
    assert_eq!(profile.at(0.06), 0.0);
    assert!(profile.at(0.059) > 0.0);
}

#[test]
fn the_shapes_checks() {
    let k = RadiusRatio::new(0.1).unwrap();
    let ld = LimbDarkening::uniform();
    assert_eq!(
        Shape::transit(k, ImpactParameter::new(1.1).unwrap(), ld),
        Err(VetError::NoTransit)
    );
    assert!(Shape::transit(k, ImpactParameter::new(1.099).unwrap(), ld).is_ok());
    assert_eq!(
        Shape::default().profile(1.0, 0.51).unwrap_err(),
        VetError::DurationTooLongForShape
    );
    assert!(Shape::default().profile(1.0, 0.5).is_ok());
    // A box takes any duration below the period.
    assert!(Shape::BOX.profile(1.0, 0.9).is_ok());
}
