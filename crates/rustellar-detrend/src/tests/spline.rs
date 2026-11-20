// The knots, the basis and the triangle's corners.

use crate::spline::{Knots, Triangle};

#[test]
fn the_knots() {
    let k = Knots::new(10.0, 12.0, 0.5, 100);
    assert_eq!((k.start, k.step, k.intervals), (10.0, 0.5, 4));
    // Rounded up: spacing at most the window.
    assert_eq!(Knots::new(10.0, 12.0, 0.6, 100).intervals, 4);
    // No more intervals than points, and at least one.
    assert_eq!(Knots::new(10.0, 12.0, 1e-6, 7).intervals, 7);
    assert_eq!(Knots::new(10.0, 12.0, 1e-6, 0).intervals, 1);
    assert_eq!(Knots::new(10.0, 12.0, 1e9, 7).intervals, 1);
    // One time: an interval a window long.
    let one = Knots::new(3.0, 3.0, 0.5, 1);
    assert_eq!((one.start, one.step, one.intervals), (3.0, 0.5, 1));
    assert_eq!(one.coefficients(), 4);
}

#[test]
fn the_basis_sums_to_one() {
    let k = Knots::new(1400.0, 1427.3, 0.37, 1000);
    for i in 0..=1000 {
        let x = 1400.0 + 27.3 * f64::from(i) / 1000.0;
        let (j, b) = k.basis(x);
        assert!(j + 4 <= k.coefficients());
        assert!(b.iter().all(|&v| v >= 0.0));
        assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }
    // At a knot: 1/6, 2/3, 1/6, 0.
    let (j, b) = k.basis(1400.0 + k.step * 3.0);
    assert_eq!(j, 3);
    assert!((b[0] - 1.0 / 6.0).abs() < 1e-12 && (b[1] - 2.0 / 3.0).abs() < 1e-12);
    // The last point is in the last interval.
    assert_eq!(k.basis(1427.3).0, k.intervals - 1);
}

// Rows with zeros skip their rotations; a triangle as wide as its band
// solves a small system exactly.
#[test]
fn the_triangle() {
    let mut t = Triangle::new(2);
    t.add(0, [2.0, 0.0, 0.0, 0.0], 4.0);
    t.add(0, [0.0, 3.0, 0.0, 0.0], 9.0);
    assert_eq!(t.solve(), [2.0, 3.0]);
    let mut t = Triangle::new(4);
    for (first, h, y) in [
        (0, [1.0, 1.0, 0.0, 0.0], 3.0),
        (0, [1.0, -1.0, 0.0, 0.0], -1.0),
        (2, [1.0, 1.0, 0.0, 0.0], 7.0),
        (2, [1.0, -1.0, 0.0, 0.0], -1.0),
    ] {
        t.add(first, h, y);
    }
    let x = t.solve();
    for (x, want) in x.iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!((x - want).abs() < 1e-14, "{x}");
    }
}

// Values whose squares leave the normal range are scaled first.
#[test]
fn tiny_and_huge_rows() {
    for scale in [1e-200, 1e200] {
        let mut t = Triangle::new(1);
        t.add(0, [3.0 * scale, 0.0, 0.0, 0.0], 6.0 * scale);
        t.add(0, [4.0 * scale, 0.0, 0.0, 0.0], 8.0 * scale);
        let x = t.solve();
        assert!((x[0] - 2.0).abs() < 1e-15, "{x:?}");
    }
}
