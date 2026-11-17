// The exact sum: means of whole numbers checked for correct rounding against
// their neighbours in integer arithmetic, scaling by powers of two, adding
// and taking away across every exponent, and hand-built extremes.

use super::Random;
use crate::exact_sum::ExactSum;

// ==========================
// Helper Functions
// ==========================

// `x` as (signed mantissa, exponent): x = mantissa 2^exponent exactly.
fn split(x: f64) -> (i128, i32) {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = (bits & ((1 << 52) - 1)) as i128;
    let (mantissa, exponent) = if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1 << 52), biased - 1075)
    };
    (if bits >> 63 == 1 { -mantissa } else { mantissa }, exponent)
}

// The doubles either side of finite `x`, as bit steps (no `next_up` at the
// crate's minimum Rust).
fn neighbours(x: f64) -> (f64, f64) {
    let step = |x: f64, up: bool| {
        if x == 0.0 {
            let tiny = f64::from_bits(1);
            return if up { tiny } else { -tiny };
        }
        let bits = x.to_bits();
        let away = (x > 0.0) == up;
        f64::from_bits(if away { bits + 1 } else { bits - 1 })
    };
    (step(x, false), step(x, true))
}

// Whether `m` is `sum / count` correctly rounded, ties to even, for a whole
// `sum` (below 2^64) and `count` (below 2^10): its distance to the exact
// quotient against its neighbours', in a common unit.
fn correctly_rounded(m: f64, sum: i128, count: i128) -> bool {
    // A nonzero quotient is at least 2^-10: far from zero's neighbours.
    if m == 0.0 || sum == 0 {
        return m == 0.0 && sum == 0;
    }
    let (down, up) = neighbours(m);
    let parts = [split(down), split(m), split(up)];
    let unit = parts.iter().map(|p| p.1).min().unwrap().min(0);
    let distance = |(mantissa, exponent): (i128, i32)| {
        ((mantissa << (exponent - unit)) * count - (sum << -unit)).abs()
    };
    let [d_down, d_m, d_up] = parts.map(distance);
    let even = m.to_bits() & 1 == 0;
    (d_m < d_down || (d_m == d_down && even)) && (d_m < d_up || (d_m == d_up && even))
}

fn mean_of(values: &[f64]) -> f64 {
    let mut sum = ExactSum::default();
    for &x in values {
        sum.add(x);
    }
    sum.mean(values.len() as u64)
}

// A double of any exponent, subnormals included, either sign.
fn any_double(random: &mut Random) -> f64 {
    loop {
        let x = f64::from_bits(random.next());
        if x.is_finite() {
            return x;
        }
    }
}

// ==========================
// Tests
// ==========================

#[test]
fn means_of_whole_numbers_are_correctly_rounded() {
    let mut random = Random(0x5eed_0001);
    let mut ties = 0;
    for trial in 0..20_000 {
        // Pairs of wide whole numbers, whose halved sums are often midway;
        // or many narrower ones.
        let pair = trial % 4 == 0;
        let count = if pair { 2 } else { 1 + random.below(1000) };
        let bits = 1 + random.below(30);
        let values: Vec<f64> = (0..count)
            .map(|_| {
                let k = if pair {
                    (random.below(1 << 53) << random.below(10)) as f64
                } else {
                    random.below(1 << bits) as f64
                };
                if random.below(3) == 0 { -k } else { k }
            })
            .collect();
        let sum: i128 = values.iter().map(|&v| v as i128).sum();
        let m = mean_of(&values);
        assert!(correctly_rounded(m, sum, count as i128), "{values:?}: {m}");
        let (mantissa, exponent) = split(m);
        let rounded = exponent < -1 || mantissa << (exponent + 1) != sum;
        ties += usize::from(pair && rounded);
    }
    // Halved sums that needed rounding met.
    assert!(ties > 500, "{ties}");
}

#[test]
fn ties_go_to_even() {
    let big = 2f64.powi(53);
    // 2^52 + 1/2 and 2^52 + 3/2: midway, to the even neighbour.
    assert_eq!(mean_of(&[1.0, big]), 2f64.powi(52));
    assert_eq!(mean_of(&[3.0, big]), 2f64.powi(52) + 2.0);
    assert_eq!(mean_of(&[-3.0, -big]), -(2f64.powi(52) + 2.0));
    // Just past midway, from the bits below the 53 kept.
    let past = mean_of(&[1.0 + 2f64.powi(-52), big]);
    assert_eq!(past, 2f64.powi(52) + 1.0);
    // Just past midway from the remainder alone: (2^53 + 4/3) units of
    // 2^-1074, kept to even units, has no quotient bits below the half.
    let unit = f64::from_bits(1);
    let past = mean_of(&[3.0 * 2f64.powi(-1021), 4.0 * unit, 0.0]);
    assert_eq!(past, 2f64.powi(-1021) + 2.0 * unit);
    // Midway in the subnormals: 1/2 and 3/2 of the smallest.
    let tiny = f64::from_bits(1);
    assert_eq!(mean_of(&[tiny, 0.0]), 0.0);
    assert_eq!(mean_of(&[3.0 * tiny, 0.0]), 2.0 * tiny);
    assert_eq!(mean_of(&[-3.0 * tiny, 0.0]), -2.0 * tiny);
    assert_eq!(mean_of(&[5.0 * tiny, 0.0, 0.0]), 2.0 * tiny);
}

#[test]
fn means_scale_by_powers_of_two() {
    let mut random = Random(0x5eed_0002);
    for _ in 0..5_000 {
        let count = 1 + random.below(50);
        let values: Vec<f64> = (0..count)
            .map(|_| random.below(1 << 30) as f64 - (1 << 29) as f64)
            .collect();
        let m = mean_of(&values);
        let power = random.below(1990) as i32 - 1000;
        let scale = 2f64.powi(power);
        let scaled: Vec<f64> = values.iter().map(|v| v * scale).collect();
        // Exact while the mean stays normal (it is 0 or at least 2^-6).
        assert_eq!(mean_of(&scaled), m * scale, "{values:?} by 2^{power}");
    }
}

#[test]
fn adding_and_taking_away_is_exact() {
    let mut random = Random(0x5eed_0003);
    for _ in 0..500 {
        let count = 1 + random.below(200) as usize;
        let values: Vec<f64> = (0..count).map(|_| any_double(&mut random)).collect();
        let mut sum = ExactSum::default();
        for &x in &values {
            sum.add(x);
        }
        // The first half taken away, out of order: the rest's sum exactly.
        let half = count / 2;
        let mut order: Vec<usize> = (0..half).collect();
        for k in (1..half).rev() {
            order.swap(k, random.below(k as u64 + 1) as usize);
        }
        for &k in &order {
            sum.remove(values[k]);
        }
        let mut rest = ExactSum::default();
        for &x in &values[half..] {
            rest.add(x);
        }
        assert_eq!(sum, rest);
        for &x in &values[half..] {
            sum.remove(x);
        }
        assert_eq!(sum, ExactSum::default());
    }
}

#[test]
fn sums_past_the_largest_double() {
    let max = f64::MAX;
    assert_eq!(mean_of(&[1e308, 1e308]), 1e308);
    assert_eq!(mean_of(&[max; 1000]), max);
    assert_eq!(mean_of(&[-max; 1000]), -max);
    assert_eq!(mean_of(&[max, max, -max]), max / 3.0);
    let mut sum = ExactSum::default();
    for _ in 0..1_000_000 {
        sum.add(max);
    }
    assert_eq!(sum.mean(1_000_000), max);
    assert_eq!(sum.mean(2_000_000), max / 2.0);
}

#[test]
fn cancellation_loses_nothing() {
    assert_eq!(mean_of(&[1.0, 1e100, -1e100]), 1.0 / 3.0);
    assert_eq!(mean_of(&[1e300, 1e-300, -1e300]), 1e-300 / 3.0);
    assert_eq!(mean_of(&[f64::MAX, f64::from_bits(1), -f64::MAX]), 0.0);
    assert_eq!(
        mean_of(&[f64::MAX, f64::from_bits(3), -f64::MAX]),
        f64::from_bits(1)
    );
}

#[test]
fn zero_and_clear() {
    assert_eq!(mean_of(&[0.0]), 0.0);
    assert_eq!(mean_of(&[-0.0, 0.0]), 0.0);
    assert_eq!(mean_of(&[2.5, -2.5]), 0.0);
    let mut sum = ExactSum::default();
    sum.add(f64::MAX);
    sum.add(-1e-310);
    sum.clear();
    assert_eq!(sum, ExactSum::default());
    sum.add(7.0);
    assert_eq!(sum.mean(2), 3.5);
}
