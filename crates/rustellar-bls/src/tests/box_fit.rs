// The offset from the nearest mid-transit: exact where the answer is a
// representable number, and never more than half a period.

use rstest::rstest;

use crate::box_fit::{nearest, offset_from_transit};

// 2^-1000 and 2^950.
const TINY: f64 = 9.332_636_185_032_189e-302;
const HUGE: f64 = 9.516_908_214_257_812e285;

#[test]
fn tiny_and_huge_are_powers_of_two() {
    assert_eq!(TINY.to_bits(), (1023_u64 - 1000) << 52);
    assert_eq!(HUGE.to_bits(), (1023_u64 + 950) << 52);
}

#[rstest]
#[case::at_t0(1.5, 1.5, 2.0, 0.0)]
#[case::orbits_later(3001.5, 1.5, 2.0, 0.0)]
#[case::orbits_earlier(-2.5, 1.5, 2.0, 0.0)]
#[case::after(3001.75, 1.5, 2.0, 0.25)]
#[case::before(3001.25, 1.5, 2.0, -0.25)]
#[case::before_t0(1.25, 1.5, 2.0, -0.25)]
// Beyond 2^26 orbits, and for periods past 2^-900 or 2^900, the product of
// the orbits and the period comes from a fused multiply-add instead.
#[case::many_orbits(100_000_000.25, 0.0, 1.0, 0.25)]
#[case::tiny_period(1.25 * TINY, 0.0, TINY, 0.25 * TINY)]
#[case::huge_period(3.25 * HUGE, 0.0, HUGE, 0.25 * HUGE)]
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

// Halves go to the even neighbour; from 2^52 up, every double is whole.
#[rstest]
#[case::down(2.4, 2.0)]
#[case::up(2.6, 3.0)]
#[case::half_to_even_below(2.5, 2.0)]
#[case::half_to_even_above(3.5, 4.0)]
#[case::negative(-2.5, -2.0)]
#[case::small_negative(-0.4, -0.0)]
#[case::just_below_2_52(4_503_599_627_370_495.5, 4_503_599_627_370_496.0)]
#[case::large(1e17, 1e17)]
#[case::large_negative(-1e17, -1e17)]
fn nearest_whole_number(#[case] x: f64, #[case] expected: f64) {
    assert_eq!(nearest(x).to_bits(), expected.to_bits());
}
