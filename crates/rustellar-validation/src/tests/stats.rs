// The statistics, on values worked by hand.

use crate::{median, sign_test, timing};

#[test]
fn the_median() {
    assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
}

// 10 of 10: 2 / 2^10. 5 of 10: everything. 2 of 10 as 8 of 10:
// 2 (1 + 10 + 45) / 1024.
#[test]
fn the_sign_test() {
    assert_eq!(sign_test(10, 10), 2.0 / 1024.0);
    assert_eq!(sign_test(5, 10), 1.0);
    assert_eq!(sign_test(2, 10), 112.0 / 1024.0);
    assert_eq!(sign_test(8, 10), sign_test(2, 10));
}

// Transits at 1, 3, 5, 7 and 9 days. The right period 6 minutes late: 6
// minutes each. A period 1 minute long: 0, 1, 2, 3 and 4 minutes off.
#[test]
fn timing_errors() {
    let minute = 1.0 / 1440.0;
    assert!((timing(2.0, 1.0, 2.0, 1.0 + 6.0 * minute, 10.0) - 6.0).abs() < 1e-9);
    assert!((timing(2.0, 1.0, 2.0 + minute, 1.0, 10.0) - 2.0).abs() < 1e-9);
    // A first transit before the data starts is not counted.
    assert!((timing(2.0, -1.0, 2.0, -1.0 + 6.0 * minute, 10.0) - 6.0).abs() < 1e-9);
}
