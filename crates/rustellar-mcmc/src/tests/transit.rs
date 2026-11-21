// The transit posterior's mappings: Kipping's q1 and q2, sqrt(e) cos w and
// sqrt(e) sin w, the density, the Cholesky factor.

use rustellar_fit::Parameter;

use crate::transit::{Coordinate, cholesky, coordinate_of, density, slot, values_from_coordinates};

fn values(u1: f64, u2: f64, e_cos: f64, e_sin: f64) -> [f64; 10] {
    [0.0, 3.0, 0.1, 10.0, 0.3, u1, u2, e_cos, e_sin, 1.0]
}

#[test]
fn kipping_round_trip() {
    for (u1, u2) in [(0.4, 0.25), (0.0, 0.5), (1.0, 0.0), (0.2, -0.1), (0.7, 0.3)] {
        let v = values(u1, u2, 0.0, 0.0);
        let q = [
            coordinate_of(Coordinate::Q1, &v),
            coordinate_of(Coordinate::Q2, &v),
        ];
        let back = values_from_coordinates(&[Coordinate::Q1, Coordinate::Q2], &q, v).unwrap();
        assert!((back[slot(Parameter::U1)] - u1).abs() < 1e-15, "{u1} {u2}");
        assert!((back[slot(Parameter::U2)] - u2).abs() < 1e-15, "{u1} {u2}");
    }
    // No darkening: q2 a half, by convention.
    let v = values(0.0, 0.0, 0.0, 0.0);
    assert_eq!(coordinate_of(Coordinate::Q1, &v), 0.0);
    assert_eq!(coordinate_of(Coordinate::Q2, &v), 0.5);
    // Outside the unit square.
    for q in [[-0.1, 0.5], [1.1, 0.5], [0.5, -0.1], [0.5, 1.1]] {
        assert!(values_from_coordinates(&[Coordinate::Q1, Coordinate::Q2], &q, v).is_none());
    }
}

#[test]
fn eccentricity_round_trip() {
    for (e, w) in [(0.1f64, 30.0f64), (0.5, 200.0), (0.9, -80.0)] {
        let (s, c) = w.to_radians().sin_cos();
        let v = values(0.4, 0.25, e * c, e * s);
        let x = [
            coordinate_of(Coordinate::SqrtECosW, &v),
            coordinate_of(Coordinate::SqrtESinW, &v),
        ];
        assert!((x[0].hypot(x[1]) - e.sqrt()).abs() < 1e-15);
        let back = values_from_coordinates(&[Coordinate::SqrtECosW, Coordinate::SqrtESinW], &x, v)
            .unwrap();
        assert!((back[slot(Parameter::ECosW)] - e * c).abs() < 1e-15);
        assert!((back[slot(Parameter::ESinW)] - e * s).abs() < 1e-15);
    }
    let circular = values(0.4, 0.25, 0.0, 0.0);
    assert_eq!(coordinate_of(Coordinate::SqrtECosW, &circular), 0.0);
    assert_eq!(coordinate_of(Coordinate::SqrtESinW, &circular), 0.0);
}

#[test]
fn ranges() {
    let v = values(0.4, 0.25, 0.0, 0.0);
    let b = [Coordinate::ImpactParameter];
    assert!(values_from_coordinates(&b, &[1.09], v).is_some());
    assert!(values_from_coordinates(&b, &[1.1], v).is_none());
    assert!(values_from_coordinates(&b, &[-0.01], v).is_none());
    assert!(values_from_coordinates(&[Coordinate::Baseline], &[0.0], v).is_none());
}

#[test]
fn stellar_density() {
    // a / R* = 10 and P = 3 d: 3 pi 1000 / (G (3 * 86400 s)^2), in g/cm^3.
    let rho = density(&values(0.4, 0.25, 0.0, 0.0));
    let want = 3.0 * std::f64::consts::PI * 1000.0 / (6.674_30e-11 * 259_200.0f64.powi(2)) / 1000.0;
    assert!((rho - want).abs() < 1e-15 * want);
    // About the Sun's 1.41 g/cm^3 and a half.
    assert!((rho - 2.1018).abs() < 1e-4, "{rho}");
}

#[test]
fn cholesky_factor() {
    let a = [4.0, 2.0, 0.4, 2.0, 2.0, 0.5, 0.4, 0.5, 3.0];
    let l = cholesky(&a, 3).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let lt: f64 = (0..3).map(|k| l[i * 3 + k] * l[j * 3 + k]).sum();
            assert!((lt - a[i * 3 + j]).abs() < 1e-15);
        }
        for j in i + 1..3 {
            assert_eq!(l[i * 3 + j], 0.0);
        }
    }
    assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
    assert!(cholesky(&[f64::NAN], 1).is_none());
}
