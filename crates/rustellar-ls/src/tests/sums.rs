use crate::sums::{Sum, kernel, split};

fn phase(f: f64, t: f64) -> ((f64, f64), (f64, f64)) {
    crate::sums::phase(f, split(f), t, split(t))
}

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

#[test]
fn kernel_within_a_rounding() {
    // sin x and cos x - 1 at 40 digits, rounded once.
    for (x, sine, versine) in [
        (0.5, 0.479_425_538_604_203, -0.122_417_438_109_627_28),
        (
            -std::f64::consts::FRAC_PI_4,
            -0.707_106_781_186_547_5,
            -0.292_893_218_813_452_4,
        ),
        (1e-8, 1e-8, -5e-17),
        (
            0.123_456_789,
            0.123_143_415_182_310_87,
            -0.007_611_114_886_287_63,
        ),
        (-0.3, -0.295_520_206_661_339_55, -0.044_663_510_874_393_98),
    ] {
        let (s, v) = kernel(x);
        assert!((s - sine).abs() <= f64::EPSILON * sine.abs(), "{x}: {s}");
        assert!(
            (v - versine).abs() <= f64::EPSILON * versine.abs(),
            "{x}: {v}"
        );
    }
    // Against the standard library's across the range.
    for k in -1000..=1000 {
        let x = std::f64::consts::FRAC_PI_4 * f64::from(k) / 1000.0;
        let (s, _) = kernel(x);
        assert!(
            (s - x.sin()).abs() <= 2.0 * f64::EPSILON * x.sin().abs(),
            "{x}"
        );
    }
}

#[test]
fn phase_beyond_two_to_the_fifty_two() {
    // f t = 2^53 + 2: whole, so no phase at all.
    let ((cb, cx), (sb, sx)) = phase(2.0, 4_503_599_627_370_497.0);
    assert_eq!((cb, sb), (1.0, 0.0));
    assert_eq!((cx, sx), (0.0, 0.0));
}
