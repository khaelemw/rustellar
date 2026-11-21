use crate::sums::{Sum, phase};

#[test]
fn sum_carries_rounding() {
    let mut s = Sum::default();
    for x in [1.0, 1e100, 1.0, -1e100] {
        s.add(x);
    }
    assert_eq!(s.value(), 2.0);
    let mut s = Sum::default();
    for _ in 0..10 {
        s.add(0.1);
    }
    assert_eq!(s.value(), 1.0);
}

#[test]
fn quotient_to_twice_the_digits() {
    let (mut n, mut d) = (Sum::default(), Sum::default());
    n.add(1.0);
    d.add(3.0);
    let (hi, lo) = n.quotient(d);
    // 1/3 = hi + lo with lo the next 53 bits: 3 (hi + lo) = 1 to 1e-32.
    assert_eq!(hi, 1.0 / 3.0);
    assert!((3.0f64.mul_add(hi, -1.0) + 3.0 * lo).abs() < 1e-32);
    // With carries in both.
    let (mut n, mut d) = (Sum::default(), Sum::default());
    for x in [1e16, 1.0, 1.0] {
        n.add(x);
    }
    for x in [1e16, 1.0] {
        d.add(x);
    }
    let (hi, lo) = n.quotient(d);
    // (1e16 + 2) / (1e16 + 1) = 1 + 1e-16 - 1e-32 + ...
    assert!(((hi - 1.0) + lo - 1e-16).abs() < 1e-31, "{hi} {lo:e}");
}

#[test]
fn phase_exact_at_large_times_and_quarter_cycles() {
    use std::f64::consts::TAU;
    // 0.1 is 0.1000000000000000055511151231257827: at t = 2.46e6 the
    // product is 246000.0000000000136557..., 1.3656e-11 cycles.
    let ((cb, cx), (sb, sx)) = phase(0.1, 2.46e6);
    let theta = TAU * 1.365_574_320_288_942_5e-11;
    assert_eq!((cb, sb), (1.0, 0.0));
    assert!((sx / theta - 1.0).abs() < 1e-15, "{sx:e}");
    assert!((cx / (-0.5 * theta * theta) - 1.0).abs() < 1e-15, "{cx:e}");
    // Just past half a cycle: 1 + 2^-52 at t = 0.5 is 2^-53 past 0.5.
    let ((cb, cx), (sb, sx)) = phase(1.0 + f64::EPSILON, 0.5);
    let theta = TAU * f64::EPSILON / 2.0;
    assert_eq!((cb, sb), (-1.0, 0.0));
    assert!((sx / -theta - 1.0).abs() < 1e-15, "{sx:e}");
    assert!(cx > 0.0 && cx < 1e-30);
    // Just past a quarter: the cosine near 0 keeps its digits.
    let ((cb, cx), (sb, sx)) = phase(1.0 + f64::EPSILON, 0.25);
    assert_eq!((cb, sb), (0.0, 1.0));
    assert!(
        (cx / -(TAU * f64::EPSILON / 4.0) - 1.0).abs() < 1e-15,
        "{cx:e}"
    );
    assert!(sx < 0.0 && sx > -1e-30);
    // Three quarters, and exact quarters.
    let ((cb, _), (sb, _)) = phase(0.75, 1.0);
    assert_eq!((cb, sb), (0.0, -1.0));
    assert_eq!(phase(0.5, 4.0), ((1.0, 0.0), (0.0, 0.0)));
    assert_eq!(phase(0.25, -3.0), ((0.0, -0.0), (1.0, 0.0)));
}
