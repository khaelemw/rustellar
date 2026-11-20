// The depth fit through the public API: a limb-darkened transit injected
// with rustellar-model and fitted back, a box, and what cannot be fitted.

use rustellar_model::{ScaledSemiMajorAxis, TransitParams};
use rustellar_vet::{
    Candidate, Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, Shape, TimeDelta,
    VetError, fit_depth,
};

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn at(t: f64) -> Epoch {
    Epoch::from_btjd(t).unwrap()
}

// 27 days every 10 minutes from BTJD 1500.
fn times() -> Vec<f64> {
    (0..3888).map(|i| 1500.0 + f64::from(i) / 144.0).collect()
}

// A planet of k 0.05 at b 0.5, a/R* 15, P 3.7 d, around a Sun-like star;
// its duration from the model.
fn planet() -> TransitParams {
    TransitParams::new(
        at(1501.3),
        days(3.7),
        RadiusRatio::new(0.05).unwrap(),
        ScaledSemiMajorAxis::new(15.0).unwrap(),
        ImpactParameter::new(0.5).unwrap(),
        LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
    )
    .unwrap()
}

// The default shape is this planet's: fitted at its own duration, the
// noiseless transit's depth comes back to rounding, whatever the baseline.
#[test]
fn a_limb_darkened_transit_comes_back() {
    let p = planet();
    let t = times();
    let duration = p.t14().unwrap();
    let centre = 1.0 - p.flux_at(at(1501.3));
    for scale in [1.0, 2e4, 3e-5] {
        let flux: Vec<f64> = t.iter().map(|&x| scale * p.flux_at(at(x))).collect();
        let lc = LightCurve::from_btjd(t.clone(), flux, vec![1e-3 * scale; t.len()]).unwrap();
        let candidate = Candidate::new(days(3.7), at(1501.3 + 2.0 * 3.7), duration).unwrap();
        let fit = fit_depth(&lc, &candidate, Shape::default()).unwrap();
        assert!(
            (fit.depth / centre - 1.0).abs() < 1e-11,
            "{scale}: {} {centre}",
            fit.depth
        );
        assert!((fit.baseline / scale - 1.0).abs() < 1e-13);
        assert!(fit.chi2 < 1e-12, "{}", fit.chi2);
        // Seven transits of 1.74 hours at 10 minutes, about 10.5 points each.
        assert_eq!(fit.points_in_transit, 75);
    }
}

// A box fitted to the same transit finds a shallower, average depth, and
// fits it worse.
#[test]
fn a_box_on_a_rounded_transit() {
    let p = planet();
    let t = times();
    let flux: Vec<f64> = t.iter().map(|&x| p.flux_at(at(x))).collect();
    let lc = LightCurve::from_btjd(t.clone(), flux, vec![1e-4; t.len()]).unwrap();
    let candidate = Candidate::new(days(3.7), at(1501.3), p.t14().unwrap()).unwrap();
    let shaped = fit_depth(&lc, &candidate, Shape::default()).unwrap();
    let boxed = fit_depth(&lc, &candidate, Shape::BOX).unwrap();
    assert!(boxed.depth < shaped.depth);
    assert!(boxed.depth > 0.6 * shaped.depth);
    assert!(boxed.chi2 > 1.0);
    assert_eq!(boxed.points_in_transit, shaped.points_in_transit);
}

#[test]
fn what_cannot_be_fitted() {
    let t: Vec<f64> = (0..100).map(|i| f64::from(i) / 100.0).collect();
    let lc = LightCurve::from_btjd(t, vec![1.0; 100], vec![1e-3; 100]).unwrap();
    // No point within a transit at 5.5 (the light curve covers one day).
    let late = Candidate::new(days(10.0), at(5.5), days(0.2)).unwrap();
    assert_eq!(
        fit_depth(&lc, &late, Shape::BOX),
        Err(VetError::NoPointsInTransit)
    );
    // Every point in transit.
    let long = Candidate::new(days(3.0), at(0.5), days(1.4)).unwrap();
    assert_eq!(
        fit_depth(&lc, &long, Shape::default()),
        Err(VetError::NoPointsOutOfTransit)
    );
    // Longer than half the period: a box only.
    let wide = Candidate::new(days(0.5), at(0.5), days(0.3)).unwrap();
    assert_eq!(
        fit_depth(&lc, &wide, Shape::default()),
        Err(VetError::DurationTooLongForShape)
    );
    assert!(fit_depth(&lc, &wide, Shape::BOX).is_ok());
}

#[test]
fn the_new_errors_say_what_is_wrong() {
    for (err, text) in [
        (
            VetError::NoTransit,
            "the impact parameter must be below 1 + k for the planet to cross the star",
        ),
        (
            VetError::DurationTooLongForShape,
            "a limb-darkened shape needs a duration of at most half the period",
        ),
        (
            VetError::NoPointsInTransit,
            "no point of the light curve is in transit",
        ),
        (
            VetError::NoPointsOutOfTransit,
            "no point of the light curve is out of transit",
        ),
        (
            VetError::BaselineNotPositive,
            "the fitted baseline is not above zero",
        ),
        (
            VetError::RatioZero,
            "a period ratio needs a numerator and denominator above zero",
        ),
        (
            VetError::AliasPeriodNotFinite,
            "the alias's period is too large to hold",
        ),
    ] {
        assert_eq!(err.to_string(), text);
    }
}
