// Eccentric orbits against mpmath at 40 digits (the sky separation, which
// side of the star the planet is on, the contact times) and batman 2.5.1's
// flux, on six orbits from e = 0.05 to 0.9; and the API's checks.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use rustellar_model::{
    Epoch, Exposure, ImpactParameter, LimbDarkening, ParamError, RadiusRatio, ScaledSemiMajorAxis,
    TimeDelta, TransitModel, TransitParams,
};

const POINTS: &str = include_str!("reference/eccentric_points.csv");
const CONTACTS: &str = include_str!("reference/eccentric_contacts.csv");

fn circular(a: f64, b: f64, k: f64, period: f64) -> TransitParams {
    TransitParams::new(
        Epoch::from_days(0.0).unwrap(),
        TimeDelta::from_days(period).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(a).unwrap(),
        ImpactParameter::new(b).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
}

/// Each orbit's parameters, by name.
fn orbits() -> BTreeMap<String, TransitParams> {
    POINTS
        .lines()
        .skip(1)
        .map(|line| {
            let f: Vec<f64> = line
                .split(',')
                .skip(1)
                .take(6)
                .map(|s| s.parse().unwrap())
                .collect();
            let params = circular(f[2], f[3], f[4], f[5])
                .with_eccentricity(f[0], f[1])
                .unwrap();
            (line.split(',').next().unwrap().to_owned(), params)
        })
        .collect()
}

#[test]
fn separation_and_side_against_mpmath() {
    let orbits = orbits();
    let mut worst: f64 = 0.0;
    for line in POINTS.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let params = orbits[f[0]];
        let t = Epoch::from_days(f[7].parse().unwrap()).unwrap();
        let z: f64 = f[8].parse().unwrap();
        // The orbit's angles are known to a few units in the last place
        // (the time's own rounding among them), so the separation to a
        // few of `a`'s.
        let a: f64 = f[3].parse().unwrap();
        let gap = (params.separation(t) - z).abs() / (a * f64::EPSILON);
        worst = worst.max(gap);
        assert!(gap < 16.0, "{line}: {}", params.separation(t));
        assert_eq!(params.in_front(t), f[9] == "1", "{line}");
    }
    eprintln!("separation within {worst:.1} a eps");
}

#[test]
fn contacts_against_mpmath() {
    let orbits = orbits();
    for line in CONTACTS.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let params = orbits[f[0]];
        let outer = (f[1].parse::<f64>().unwrap() - 1.0 - params.k().get()).abs() < 1e-12;
        let contacts = params.contacts().unwrap();
        let (before, after) = if outer {
            (Some(contacts.t1), Some(contacts.t4))
        } else {
            (contacts.t2, contacts.t3)
        };
        if f[2].is_empty() {
            assert!(before.is_none() && after.is_none(), "{line}");
            assert!(outer || params.t23().is_none());
            continue;
        }
        let (want_before, want_after): (f64, f64) = (f[2].parse().unwrap(), f[3].parse().unwrap());
        let (before, after) = (before.unwrap().as_days(), after.unwrap().as_days());
        assert!((before - want_before).abs() < 1e-11, "{line}: {before}");
        assert!((after - want_after).abs() < 1e-11, "{line}: {after}");
        let duration = if outer { params.t14() } else { params.t23() };
        assert!((duration.unwrap().as_days() - (want_after - want_before)).abs() < 2e-11);
    }
}

#[test]
fn flux_against_batman() {
    // batman solves Kepler's equation and integrates the flux to its own
    // tolerances; the gap is theirs.
    let orbits = orbits();
    let mut worst: f64 = 0.0;
    for line in POINTS.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let params = orbits[f[0]];
        let t = Epoch::from_days(f[7].parse().unwrap()).unwrap();
        let want: f64 = f[10].parse().unwrap();
        worst = worst.max((params.flux_at(t) - want).abs());
    }
    eprintln!("flux within {worst:.1e} of batman");
    assert!(worst < 1e-6);
}

#[test]
fn every_path_agrees_with_flux_at() {
    for params in orbits().into_values() {
        let span = params.period().as_days();
        let times: Vec<Epoch> = (0..4000)
            .map(|i| Epoch::from_days(span * (f64::from(i) / 4000.0 - 0.5)).unwrap())
            .collect();
        // The light curve skips the times clear of the star.
        let curve = params.light_curve(&times);
        for (&t, &f) in times.iter().zip(&curve) {
            assert_eq!(f, params.flux_at(t));
        }
        // An exposure's flux is the mean of its instants'.
        let exposure = Exposure::new(
            TimeDelta::from_days(0.01 * span).unwrap(),
            NonZeroUsize::new(7).unwrap(),
        )
        .unwrap();
        let mut model = TransitModel::with_exposure(times.clone(), exposure);
        let binned = model.light_curve(params);
        for (&t, &f) in times.iter().zip(binned.iter()).step_by(37) {
            let mean: f64 = exposure
                .offsets()
                .map(|o| params.flux_at(t + o))
                .sum::<f64>()
                / 7.0;
            assert!((f - mean).abs() < 1e-15);
            assert_eq!(f, params.binned_flux_at(t, exposure));
        }
    }
}

#[test]
fn checks_and_the_circular_case() {
    let c = circular(10.0, 0.3, 0.1, 3.0);
    assert_eq!(c.with_eccentricity(0.0, 45.0), Ok(c));
    assert_eq!(c.eccentricity(), 0.0);
    assert_eq!(c.periastron_deg(), 90.0);
    assert_eq!(
        c.with_eccentricity(f64::NAN, 0.0),
        Err(ParamError::NonFinite {
            name: "eccentricity"
        })
    );
    assert_eq!(
        c.with_eccentricity(0.1, f64::INFINITY),
        Err(ParamError::NonFinite { name: "omega" })
    );
    assert_eq!(
        c.with_eccentricity(1.0, 0.0),
        Err(ParamError::EccentricityOutOfRange)
    );
    assert_eq!(
        c.with_eccentricity(-0.1, 0.0),
        Err(ParamError::EccentricityOutOfRange)
    );
    assert_eq!(
        c.with_eccentricity(0.95, 0.0),
        Err(ParamError::PeriastronTooClose)
    );
    // b = 0.3 at a = 10 with e = 0.85, w = 90: cos i = 0.03 * 1.85 / 0.2775.
    let near = circular(10.0, 9.0, 0.1, 3.0);
    assert_eq!(
        near.with_eccentricity(0.5, 90.0),
        Err(ParamError::ImpactParameterTooLarge)
    );
    for err in [
        ParamError::EccentricityOutOfRange,
        ParamError::PeriastronTooClose,
    ] {
        assert!(!err.to_string().is_empty());
    }

    // cos i from b, e and w.
    let e = c.with_eccentricity(0.4, 60.0).unwrap();
    let cos_i = 0.3 / 10.0 * 0.4f64.mul_add(60f64.to_radians().sin(), 1.0) / 0.84;
    assert!((e.inclination_deg() - cos_i.acos().to_degrees()).abs() < 1e-12);
    assert_eq!(e.b().get(), 0.3);

    // An orbit whose planet misses the star: no contacts, flux 1.
    let miss = circular(10.0, 2.0, 0.1, 3.0)
        .with_eccentricity(0.2, 10.0)
        .unwrap();
    assert!(miss.contacts().is_none() && miss.t14().is_none() && miss.t23().is_none());
    assert_eq!(miss.flux_at(Epoch::from_days(0.0).unwrap()), 1.0);

    // A planet larger than the star has no inner contacts.
    let big = circular(10.0, 0.0, 1.5, 3.0)
        .with_eccentricity(0.2, 10.0)
        .unwrap();
    let contacts = big.contacts().unwrap();
    assert!(contacts.t2.is_none() && big.t23().is_none());
}
