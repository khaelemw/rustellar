// Compensated sums: exact where plain addition is not, and within one
// rounding of the exact sum over a million random terms.

use rstest::rstest;

use crate::sums::{Sum, between};

// ==========================
// Helper Functions
// ==========================

// SplitMix64: a small, seeded generator, so the tests need no dependency.
struct Random(u64);

impl Random {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    // A signed 53-bit integer times 2^e, e from -20 to 20: a term of any sign
    // over twelve orders of magnitude, whose exact value is an integer
    // multiple of 2^-20.
    fn term(&mut self) -> (f64, i128) {
        let mantissa = (self.next_u64() >> 11) as i64 - (1 << 52);
        let exponent = (self.next_u64() % 41) as i32 - 20;
        let value = mantissa as f64 * 2f64.powi(exponent);
        let scaled = i128::from(mantissa) << (exponent + 20);
        (value, scaled)
    }
}

fn sum(terms: &[f64]) -> f64 {
    terms.iter().copied().collect::<Sum>().value()
}

fn ulps_apart(a: f64, b: f64) -> u64 {
    // Same sign throughout these tests, so the bit patterns are ordered.
    a.to_bits().abs_diff(b.to_bits())
}

// ==========================
// Exact where plain addition is not
// ==========================

#[rstest]
#[case::lost_one(&[1e16, 1.0, -1e16], 1.0)]
#[case::small_first(&[1.0, 1e100, 1.0, -1e100], 2.0)]
#[case::tenths(&[0.1; 10], 1.0)]
#[case::empty(&[], 0.0)]
fn exact_where_plain_addition_is_not(#[case] terms: &[f64], #[case] exact: f64) {
    assert_eq!(sum(terms), exact);
}

#[test]
fn plain_addition_gets_them_wrong() {
    assert_eq!(1e16 + 1.0 - 1e16, 0.0);
    assert_ne!([0.1; 10].iter().sum::<f64>(), 1.0);
}

#[test]
fn overflow_gives_infinity_not_nan() {
    assert_eq!(sum(&[f64::MAX, f64::MAX]), f64::INFINITY);
    assert_eq!(sum(&[f64::MAX, f64::MAX, -1.0]), f64::INFINITY);
}

// ==========================
// A million random terms
// ==========================

// The exact sum is kept as an integer count of 2^-20, which i128 holds with
// room to spare (each term below 2^93, a million of them below 2^113), and
// converted to f64 with one rounding.
#[test]
fn million_terms_within_one_rounding_of_the_exact_sum() {
    let mut rng = Random(1);
    let (terms, scaled): (Vec<f64>, Vec<i128>) = (0..1_000_000).map(|_| rng.term()).unzip();
    let exact = scaled.iter().sum::<i128>() as f64 * 2f64.powi(-20);

    let ours = sum(&terms);
    assert!(ulps_apart(ours, exact) <= 1, "{ours} vs {exact}");

    // Plain addition, for comparison, is far off.
    let plain: f64 = terms.iter().sum();
    assert!(ulps_apart(plain, exact) > 100, "{plain} vs {exact}");
}

// ==========================
// Sums between two running totals
// ==========================

// The terms between two totals near 1e16, summed far more finely than the
// totals themselves are spaced.
#[test]
fn between_keeps_small_differences() {
    let mut total = Sum::default();
    total.add(1e16);
    let earlier = total;
    for _ in 0..10 {
        total.add(0.1);
    }
    total.add(1.0);
    assert_eq!(between(earlier, total), 2.0);
    assert_eq!(between(earlier, earlier), 0.0);
}

// Stretches of a hundred thousand random terms, from running totals, within one
// rounding of its exact sum.
#[test]
fn between_matches_exact_sums_of_stretches() {
    let mut rng = Random(7);
    let (terms, scaled): (Vec<f64>, Vec<i128>) = (0..100_000).map(|_| rng.term()).unzip();
    let mut totals = vec![Sum::default()];
    let mut exact = vec![0_i128];
    for (x, k) in terms.iter().zip(&scaled) {
        let mut next = totals[totals.len() - 1];
        next.add(*x);
        totals.push(next);
        exact.push(exact[exact.len() - 1] + k);
    }
    for _ in 0..1_000 {
        let i = (rng.next_u64() % 100_001) as usize;
        let j = (rng.next_u64() % 100_001) as usize;
        let (i, j) = (i.min(j), i.max(j));
        let ours = between(totals[i], totals[j]);
        let truth = (exact[j] - exact[i]) as f64 * 2f64.powi(-20);
        let ulps = if ours == truth {
            0
        } else {
            ulps_apart(ours, truth)
        };
        assert!(
            ulps <= 1 || (ours - truth).abs() < 1e-30,
            "{i}..{j}: {ours} vs {truth}"
        );
    }
}
