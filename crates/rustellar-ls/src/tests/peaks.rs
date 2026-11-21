use crate::peaks::itp;

#[test]
fn itp_to_the_last_bit() {
    let root = itp(|x| x * x - 2.0, 1.0, 2.0, -1.0, 2.0);
    assert!((root - std::f64::consts::SQRT_2).abs() <= f64::EPSILON * 2.0);
    // Above zero first, and a zero hit exactly.
    let root = itp(|x| 2.0 - x * x, 1.0, 2.0, 1.0, -2.0);
    assert!((root - std::f64::consts::SQRT_2).abs() <= f64::EPSILON * 2.0);
    let root = itp(|x| x - 1.5, 1.0, 2.0, -0.5, 0.5);
    assert!((root - 1.5).abs() <= f64::EPSILON * 2.0);
}
