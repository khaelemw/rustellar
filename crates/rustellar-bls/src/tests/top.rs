// Making the top of an approximate power exact: the peaks and the highest
// power are the exact power's, and few periods are made exact.

use super::recipe::Random;
use crate::{
    BlsError,
    refine::peaks,
    top::{RELATIVE, make_top_exact},
};

// ==========================
// Helper Functions
// ==========================

// A power with many local maxima: a random walk's absolute value, some
// periods zero (no dip), and two peaks equal to the last bit and two a
// rounding apart, near the top.
fn exact_power(seed: u64, n: usize) -> Vec<f64> {
    let mut random = Random(seed);
    let mut level: f64 = 50.0;
    let mut power: Vec<f64> = (0..n)
        .map(|_| {
            level = (level + 4.0 * (random.unit() - 0.5)).abs();
            if random.unit() < 0.02 { 0.0 } else { level }
        })
        .collect();
    let top = power.iter().copied().fold(0.0, f64::max);
    for (k, &at) in [n / 5, 2 * n / 5].iter().enumerate() {
        power[at] = top * 1.5;
        power[at + 3 * n / 10] = if k == 0 {
            top * 1.5
        } else {
            (top * 1.5) * (1.0 + f64::EPSILON)
        };
    }
    power
}

// Each value moved by up to `RELATIVE` of itself, the way that most often
// swaps neighbours.
fn approximate(power: &[f64], seed: u64) -> Vec<f64> {
    let mut random = Random(seed);
    power
        .iter()
        .map(|&p| p * (1.0 + 0.999 * RELATIVE * (2.0 * random.unit() - 1.0)))
        .collect()
}

fn highest(power: &[f64]) -> usize {
    (0..power.len()).fold(0, |b, i| if power[i] > power[b] { i } else { b })
}

// The approximate power made exact at the top: the periods made exact.
fn made_exact(exact: &[f64], power: &mut [f64], known: &[bool], count: usize) -> Vec<usize> {
    let mut asked = Vec::new();
    make_top_exact(
        power,
        known,
        count,
        &mut |todo: &[usize], power: &mut [f64]| {
            for &i in todo {
                power[i] = exact[i];
            }
            asked.extend_from_slice(todo);
            Ok::<(), BlsError>(())
        },
    )
    .unwrap();
    asked
}

// ==========================
// Tests
// ==========================

#[test]
fn peaks_and_highest_are_the_exact_powers() {
    for seed in 0..40 {
        let n = 4000;
        let exact = exact_power(seed, n);
        for count in [0, 1, 2, 5, 40] {
            let mut power = approximate(&exact, seed + 100);
            let asked = made_exact(&exact, &mut power, &vec![false; n], count);
            let (pa, pb) = (peaks(&power, count), peaks(&exact, count));
            assert_eq!(
                pa,
                pb,
                "{seed} {count}: {:?} {:?}",
                pa.iter()
                    .map(|&i| (power[i], power[i - 1], power[i + 1]))
                    .collect::<Vec<_>>(),
                pb.iter()
                    .map(|&i| (exact[i], exact[i - 1], exact[i + 1]))
                    .collect::<Vec<_>>()
            );
            assert_eq!(highest(&power), highest(&exact), "{seed} {count}");
            // Each once; few for a few peaks.
            let mut once = asked.clone();
            once.sort_unstable();
            once.dedup();
            assert_eq!(once.len(), asked.len(), "{seed} {count}");
            if count <= 5 {
                assert!(
                    asked.len() < n / 20,
                    "{seed} {count}: {} made exact",
                    asked.len()
                );
            }
        }
    }
}

#[test]
fn known_periods_are_not_made_exact_again() {
    let n = 2000;
    let exact = exact_power(7, n);
    let mut power = approximate(&exact, 8);
    let known: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    for i in (0..n).step_by(3) {
        power[i] = exact[i];
    }
    let asked = made_exact(&exact, &mut power, &known, 5);
    assert!(asked.iter().all(|&i| !known[i]));
    assert_eq!(peaks(&power, 5), peaks(&exact, 5));
}

#[test]
fn fewer_peaks_than_wanted_makes_every_period_exact() {
    // One peak only.
    let n = 300;
    let exact: Vec<f64> = (0..n).map(|i| 1.0 + (i.min(n - 1 - i)) as f64).collect();
    let mut power = approximate(&exact, 3);
    let asked = made_exact(&exact, &mut power, &vec![false; n], 2);
    assert_eq!(asked.len(), n);
    assert_eq!(power, exact);
}

#[test]
fn no_dip_anywhere_makes_every_period_exact() {
    let exact = vec![0.0; 50];
    let mut power = exact.clone();
    let asked = made_exact(&exact, &mut power, &[false; 50], 5);
    assert_eq!(asked.len(), 50);
}

#[test]
fn an_error_making_a_period_exact_is_returned() {
    let mut power = vec![1.0, 2.0, 1.0];
    let failed = make_top_exact(
        &mut power,
        &[false; 3],
        1,
        &mut |_: &[usize], _: &mut [f64]| Err(BlsError::OutOfRange),
    );
    assert_eq!(failed, Err(BlsError::OutOfRange));
}
