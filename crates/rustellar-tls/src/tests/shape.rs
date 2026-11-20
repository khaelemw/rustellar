// The transit shape fitted with the radius ratio and impact parameter free.

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams, flux,
};

use super::Random;
use crate::fit::{Points, Position};
use crate::shape::shape;

fn ld() -> LimbDarkening {
    LimbDarkening::quadratic(0.4804, 0.1867).unwrap()
}

/// 27 days every `cadence` days, with `dimming(t)` and white noise.
fn points(random: &mut Random, cadence: f64, noise: f64, dimming: impl Fn(f64) -> f64) -> Points {
    let time: Vec<f64> = (0..)
        .map(|i| f64::from(i) * cadence)
        .take_while(|&t| t < 27.0)
        .collect();
    let residual = time
        .iter()
        .map(|&t| {
            let n: f64 = (0..4).map(|_| random.uniform() - 0.5).sum::<f64>() * 3_f64.sqrt();
            dimming(t) + noise * n
        })
        .collect();
    let weight = vec![1.0 / noise.max(1e-6).powi(2); time.len()];
    Points {
        time,
        residual,
        weight,
    }
}

/// The straight-chord model the fit uses, at period 3.1, T0 1.3, T14 0.12.
fn chord(k: f64, b: f64) -> impl Fn(f64) -> f64 {
    move |t| {
        let n = ((t - 1.3) / 3.1 + 0.5).floor();
        let two_x = 2.0 * (t - n.mul_add(3.1, 1.3)) / 0.12;
        if two_x.abs() >= 1.0 {
            return 0.0;
        }
        let z = b
            .mul_add(b, ((1.0 + k) * (1.0 + k) - b * b) * two_x * two_x)
            .sqrt();
        1.0 - flux(z, RadiusRatio::new(k).unwrap(), ld())
    }
}

// Without noise, from the template fit's kind of start (duration and time
// a little off, depth only), the planet is found to the derivatives'
// precision: k and b, central or grazing.
#[test]
fn a_noiseless_transit_is_fitted() {
    for (k, b) in [(0.05, 0.2), (0.08, 0.7), (0.1, 0.95)] {
        let mut random = Random::new(1);
        let p = points(&mut random, 1.0 / 720.0, 0.0, chord(k, b));
        let depth = 1.0 - flux(b, RadiusRatio::new(k).unwrap(), ld());
        let start = Position {
            period: 3.1 * (1.0 + 2e-5),
            duration: 0.11,
            t0: 1.302,
        };
        let s = shape(&p, start, depth * 0.9, ld()).unwrap();
        assert!((s.k / k - 1.0).abs() < 1e-5, "{k} {b} {s:?}");
        assert!((s.b - b).abs() < 1e-4, "{k} {b} {s:?}");
        assert!((s.duration / 0.12 - 1.0).abs() < 1e-6, "{s:?}");
        assert!((s.period / 3.1 - 1.0).abs() < 1e-9, "{s:?}");
        assert!((s.t0 - 1.3).abs() < 1e-7, "{s:?}");
        assert!((s.depth / depth - 1.0).abs() < 1e-5, "{s:?}");
        // The chi-squared at weights of 1e12: residuals about 1e-9.
        assert!(s.chi2 < 1e-3, "{s:?}");
    }
}

// A planet on its curved path (the model crate's), with noise: k, b, the
// duration and depth within a few errors, and a semi-major axis near the
// truth's.
#[test]
fn a_planet_is_fitted() {
    let (k, b, a) = (0.06, 0.4, 12.0);
    let params = TransitParams::new(
        Epoch::from_days(1.3).unwrap(),
        TimeDelta::from_days(3.1).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(a).unwrap(),
        ImpactParameter::new(b).unwrap(),
        ld(),
    )
    .unwrap();
    let t14 = params.t14().unwrap().as_days();
    let mut random = Random::new(2);
    let p = points(&mut random, 1.0 / 720.0, 5e-4, |t| {
        1.0 - params.flux_at(Epoch::from_days(t).unwrap())
    });
    let start = Position {
        period: 3.1,
        duration: t14 * 0.9,
        t0: 1.3,
    };
    let s = shape(&p, start, 3e-3, ld()).unwrap();
    let e = s.errors.unwrap();
    assert!((s.k - k).abs() < 3.0 * e[3], "{s:?}");
    assert!((s.b - b).abs() < 3.0 * e[4], "{s:?}");
    assert!((s.duration - t14).abs() < 3.0 * e[2], "{s:?} {t14}");
    assert!((s.a / a - 1.0).abs() < 0.1, "{s:?}");
}

// Nothing dims where the start puts the transits: no shape.
#[test]
fn nothing_to_fit() {
    let mut random = Random::new(3);
    let p = points(&mut random, 1.0 / 48.0, 1e-3, |_| 0.0);
    let away = Position {
        period: 100.0,
        duration: 0.1,
        t0: 60.0,
    };
    assert_eq!(shape(&p, away, 1e-3, ld()), None);
}
