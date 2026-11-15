// The offset from the nearest mid-transit: exact where the answer is a
// representable number, and never more than half a period.

use rstest::rstest;

use crate::box_fit::offset_from_transit;

#[rstest]
#[case::at_t0(1.5, 1.5, 2.0, 0.0)]
#[case::orbits_later(3001.5, 1.5, 2.0, 0.0)]
#[case::orbits_earlier(-2.5, 1.5, 2.0, 0.0)]
#[case::after(3001.75, 1.5, 2.0, 0.25)]
#[case::before(3001.25, 1.5, 2.0, -0.25)]
#[case::before_t0(1.25, 1.5, 2.0, -0.25)]
fn exact_offsets(#[case] t: f64, #[case] t0: f64, #[case] p: f64, #[case] offset: f64) {
    assert_eq!(offset_from_transit(t, t0, p), offset);
}

// 0.1 is not a representable number, so 3000.1 - 0.1 is not 3000 exactly;
// the offset is the exact difference of the two floats, about -9e-14 days,
// which plain subtraction rounds away.
#[test]
fn keeps_what_plain_subtraction_loses() {
    let (t, t0, p) = (3000.1, 0.1, 1.0);
    let ours = offset_from_transit(t, t0, p);
    // 3000.1 is 3000.09999999999990905052982270717620849609375 and 0.1 is
    // 0.1000000000000000055511151231257827021181583404541015625.
    // Their difference less 3000, worked out in rational arithmetic, is this
    // float exactly.
    assert_eq!(ours, -9.095_502_129_241_595e-14);
    assert_eq!(t - t0 - 3000.0, 0.0);
}

#[test]
fn never_more_than_half_a_period() {
    for i in 0..10_000 {
        let t = 1325.0 + 0.037 * f64::from(i);
        let offset = offset_from_transit(t, 1.3, 0.731);
        assert!(offset.abs() <= 0.5 * 0.731, "{t}: {offset}");
    }
}
