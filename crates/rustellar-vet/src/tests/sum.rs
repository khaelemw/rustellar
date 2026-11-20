// The compensated sum keeps what a plain sum loses.

use crate::sum::Sum;

#[test]
fn small_terms_beside_a_large_one() {
    let mut s = Sum::default();
    for x in [1e16, 1.0, -1e16] {
        s.add(x);
    }
    assert_eq!(s.get(), 1.0);
    // The large term after the small ones too.
    let mut s = Sum::default();
    for x in [1.0, 1e16, 1.0, -1e16] {
        s.add(x);
    }
    assert_eq!(s.get(), 2.0);
}

#[test]
fn a_long_sum_of_tenths() {
    let mut s = Sum::default();
    let mut plain = 0.0_f64;
    for _ in 0..1_000_000 {
        s.add(0.1);
        plain += 0.1;
    }
    // 0.1 is 0.1000000000000000055511151231257827 in binary; a million of
    // them are 100000.0000000000055511..., nearer 100000 than the next
    // double (1.5e-11 above it).
    assert_eq!(s.get(), 100_000.0);
    assert!((plain - 100_000.0).abs() > 1e-6);
}
