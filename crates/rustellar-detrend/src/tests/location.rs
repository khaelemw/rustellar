// The median: the selection against a full sort, and the midpoint of the two
// middle values against the exact sum's correctly rounded mean.

use super::Random;
use crate::{
    exact_sum::ExactSum,
    location::{median, midpoint},
};

// ==========================
// Helper Functions
// ==========================

// A double of any exponent, subnormals included, either sign.
fn any_double(random: &mut Random) -> f64 {
    loop {
        let x = f64::from_bits(random.next());
        if x.is_finite() {
            return x;
        }
    }
}

fn exact_midpoint(a: f64, b: f64) -> f64 {
    let mut sum = ExactSum::default();
    sum.add(a);
    sum.add(b);
    sum.mean(2)
}

// ==========================
// Tests
// ==========================

#[test]
fn the_median_is_the_sorted_middle() {
    let mut random = Random(0x5eed_0101);
    for trial in 0..5_000 {
        let n = 1 + random.below(60) as usize;
        let values: Vec<f64> = (0..n)
            .map(|_| match trial % 3 {
                0 => random.unit(),
                // Few values: many ties.
                1 => random.below(4) as f64,
                _ => any_double(&mut random),
            })
            .collect();
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        let expected = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            exact_midpoint(sorted[n / 2 - 1], sorted[n / 2])
        };
        let mut scratch = values.clone();
        let found = median(&mut scratch);
        assert!(
            found == expected || found.to_bits() == expected.to_bits(),
            "{values:?}: {found} against {expected}"
        );
    }
}

#[test]
fn the_midpoint_is_correctly_rounded() {
    let mut random = Random(0x5eed_0102);
    for _ in 0..200_000 {
        let (x, y) = (any_double(&mut random), any_double(&mut random));
        let (a, b) = if x.total_cmp(&y).is_le() {
            (x, y)
        } else {
            (y, x)
        };
        let found = midpoint(a, b);
        let expected = exact_midpoint(a, b);
        assert!(
            found == expected,
            "{a:e}, {b:e}: {found:e} against {expected:e}"
        );
    }
}

#[test]
fn midpoints_at_the_extremes() {
    let (max, tiny) = (f64::MAX, f64::from_bits(1));
    // The sum overflows: halves added instead.
    assert_eq!(midpoint(max, max), max);
    assert_eq!(midpoint(1e308, 1.5e308), 1.25e308);
    assert_eq!(midpoint(-max, -max), -max);
    assert_eq!(midpoint(-max, max), 0.0);
    // Subnormal midpoints, midway: ties to even.
    assert_eq!(midpoint(tiny, 2.0 * tiny), 2.0 * tiny);
    assert_eq!(midpoint(0.0, tiny), 0.0);
    assert_eq!(midpoint(-tiny, 0.0), 0.0);
    assert_eq!(median(&mut [max, 1.0, max, -max]), midpoint(1.0, max));
    assert_eq!(median(&mut [3.0, 1.0, 2.0, 2.0]), 2.0);
    assert_eq!(median(&mut [5.0]), 5.0);
    assert_eq!(median(&mut [5.0, 4.0]), 4.5);
}
