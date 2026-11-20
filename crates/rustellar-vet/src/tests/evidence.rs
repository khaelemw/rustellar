// The median step between times.

use crate::evidence::median_step;

#[test]
fn the_median_step() {
    assert_eq!(median_step(&[3.0]), None);
    assert_eq!(median_step(&[3.0, 3.0]), None);
    assert_eq!(median_step(&[0.0, 1.0, 3.0]), Some(1.5));
    // Steps 1, 1, 2 and 6 (the repeat of 2 is no step).
    assert_eq!(median_step(&[4.0, 0.0, 1.0, 2.0, 2.0, 10.0]), Some(1.5));
    // Unsorted, a repeat ignored: steps 1, 1, 2 (no 0).
    assert_eq!(median_step(&[2.0, 0.0, 1.0, 1.0, 4.0]), Some(1.0));
}
