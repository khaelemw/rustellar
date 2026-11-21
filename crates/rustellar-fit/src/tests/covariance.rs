use crate::{Covariance, Parameter};

const FREE: [Parameter; 2] = [Parameter::RadiusRatio, Parameter::ImpactParameter];

#[test]
fn inverse_of_r_transpose_r() {
    // R = [[2, 1], [0, 4]]: R^T R = [[4, 2], [2, 17]], inverse / 64.
    let c = Covariance::from_r(FREE.to_vec(), &[2.0, 1.0, 0.0, 4.0]).unwrap();
    let k = Parameter::RadiusRatio;
    let b = Parameter::ImpactParameter;
    assert!((c.get(k, k) - 17.0 / 64.0).abs() < 1e-15);
    assert!((c.get(k, b) + 2.0 / 64.0).abs() < 1e-15);
    assert!((c.get(b, b) - 4.0 / 64.0).abs() < 1e-15);
    let mut flipped = c.clone();
    flipped.flip(1);
    assert_eq!(flipped.get(k, b), -c.get(k, b));
    assert_eq!(flipped.get(b, b), c.get(b, b));
    // g^T C g for g = (1, 1).
    assert!((c.propagate(&[1.0, 1.0]) - 17.0 / 64.0).abs() < 1e-15);
}

#[test]
fn singular() {
    assert_eq!(
        Covariance::from_r(FREE.to_vec(), &[2.0, 1.0, 0.0, 0.0]),
        None
    );
    assert_eq!(
        Covariance::from_r(FREE.to_vec(), &[2.0, 1.0, 0.0, 1e-17]),
        None
    );
}
