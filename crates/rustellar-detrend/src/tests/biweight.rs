// The biweight: against the root at 50 digits within a bound derived from
// the roundings (and wotan 1.10's error on the same windows), exact
// symmetries, the hand-built edges, and the sign-change search.

use super::Random;
use crate::{
    biweight::{biweight, nearest_root},
    location::{Location, midpoint},
    trend::trend,
    window::{Edges, WindowOptions, windows},
};

// ==========================
// Helper Functions
// ==========================

fn location(values: &[f64], c: f64) -> f64 {
    biweight(&mut values.to_vec(), &mut Vec::new(), c)
}

// The gap from `x` to the next double away from zero: its ulp.
fn ulp(x: f64) -> f64 {
    let x = x.abs();
    f64::from_bits(x.to_bits() + 1) - x
}

// `x` 2^`power`, exactly where the result is a double (in steps that stay
// within the doubles' exponents).
fn times_two_to(mut x: f64, mut power: i32) -> f64 {
    while power != 0 {
        let step = power.clamp(-1000, 1000);
        x *= 2f64.powi(step);
        power -= step;
    }
    x
}

// The bound's share for the roundings, in units of eps^2 (the residual is
// summed in double-double) times the window's sensitivity (see the
// reference): a residual term `w d` takes about eight roundings (u, u^2,
// 1 - u^2, the square, the product with d, the sum), as does c MAD.
const ROUNDINGS: f64 = 8.0;

// ==========================
// Tests
// ==========================

#[test]
fn the_root_at_fifty_digits() {
    let text = include_str!("../../tests/reference/biweight_windows.csv");
    let (mut windows_seen, mut nearest, mut certain, mut worst_ulps) = (0, 0, 0, 0.0f64);
    let (mut wotan_worse, mut wotan_worst_ulps, mut ours_worse) = (0, 0.0f64, 0);
    let mut kinds = std::collections::BTreeSet::new();
    for line in text.lines().filter(|l| !l.starts_with('#')).skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let kind = f[1];
        kinds.insert(kind);
        let number = |k: usize| f[k].parse::<f64>().unwrap();
        let (c, hi, lo, sensitivity, wotan) =
            (number(2), number(4), number(5), number(7), number(8));
        // The root, c MAD and sensitivity are given times 2^-e.
        let e: i32 = f[3].parse().unwrap();
        let values: Vec<f64> = f[9].split(';').map(|v| v.parse().unwrap()).collect();
        let ours = location(&values, c);
        // |x 2^-e - root|: x 2^-e - hi is exact (Sterbenz) wherever it
        // matters.
        let error = |x: f64| ((times_two_to(x, -e) - hi) - lo).abs();
        // The spacing of the doubles there, once scaled back.
        let unit = ulp(hi).max(times_two_to(f64::from_bits(1), (-e).max(0)));
        let rounding = ROUNDINGS * f64::EPSILON * f64::EPSILON * sensitivity;
        let bound = 0.5 * unit + rounding;
        assert!(
            error(ours) <= bound,
            "window {} ({kind}): {ours:e}, root {hi:e} + {lo:e}: {:.2} ulp, bound {:.2}",
            f[0],
            error(ours) / unit,
            bound / unit
        );
        windows_seen += 1;
        nearest += usize::from(error(ours) <= 0.5 * unit);
        // Where the roundings cannot move the sign change past a double:
        // the nearest double, always.
        if rounding <= 0.01 * unit {
            certain += 1;
            assert!(
                error(ours) <= 0.5 * unit,
                "window {}: {:.3} ulp",
                f[0],
                error(ours) / unit
            );
        }
        worst_ulps = worst_ulps.max(error(ours) / unit);
        if wotan.is_nan() {
            // wotan fails here (subnormal) or cannot finish (1e250).
        } else if error(wotan) > error(ours) {
            wotan_worse += 1;
        } else if error(wotan) < error(ours) {
            ours_worse += 1;
        }
        if !wotan.is_nan() {
            wotan_worst_ulps = wotan_worst_ulps.max(error(wotan) / unit);
        }
    }
    eprintln!(
        "{windows_seen} windows: {nearest} the nearest double ({certain} certain), worst {worst_ulps:.3e} ulp; \
         wotan worse in {wotan_worse}, better in {ours_worse}, worst {wotan_worst_ulps:.3e} ulp"
    );
    assert!(
        windows_seen > 800 && kinds.len() >= 14,
        "{windows_seen}, {kinds:?}"
    );
    // Certain but for roots within about 1e-19 of zero; wotan never nearer,
    // often far off.
    assert!(
        certain * 100 >= windows_seen * 98,
        "{certain} of {windows_seen}"
    );
    assert_eq!(ours_worse, 0);
    assert!(wotan_worse > windows_seen / 2 && wotan_worst_ulps > 1e6);
}

#[test]
fn powers_of_two_and_negation_are_exact() {
    let mut random = Random(0x5eed_0301);
    for _ in 0..2_000 {
        let n = 1 + random.below(40) as usize;
        let values: Vec<f64> = (0..n)
            .map(|_| 1.0 + 1e-3 * (random.unit() - 0.5) + 0.05 * f64::from(random.below(20) == 0))
            .collect();
        let c = 2.0 + 6.0 * random.unit();
        let t = location(&values, c);
        assert!(t.is_finite());
        let p = random.below(1800) as i32 - 900;
        let scale = 2f64.powi(p);
        let scaled: Vec<f64> = values.iter().map(|v| v * scale).collect();
        assert_eq!(location(&scaled, c), t * scale, "{values:?} by 2^{p}");
        let negated: Vec<f64> = values.iter().map(|v| -v).collect();
        assert_eq!(location(&negated, c), -t, "{values:?} negated");
        // The order of the values does not matter.
        let mut reversed = values.clone();
        reversed.reverse();
        assert_eq!(location(&reversed, c), t);
    }
}

#[test]
fn small_windows_and_no_spread() {
    // MAD 0: the median, whatever the rest.
    assert_eq!(location(&[1.0, 1.0, 1.0, 2.0, 30.0], 5.0), 1.0);
    assert_eq!(location(&[0.3; 4], 5.0), 0.3);
    assert_eq!(location(&[0.0, 0.0, -0.0], 5.0), 0.0);
    assert_eq!(location(&[0.7], 5.0), 0.7);
    // Two points: their midpoint, correctly rounded.
    let mut random = Random(0x5eed_0302);
    for _ in 0..10_000 {
        let a = (random.unit() - 0.5) * 10f64.powi(random.below(20) as i32 - 10);
        let b = (random.unit() - 0.5) * 10f64.powi(random.below(20) as i32 - 10);
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        assert_eq!(location(&[a, b], 5.0), midpoint(a, b), "{a:e}, {b:e}");
    }
    // Symmetric: the median.
    assert_eq!(location(&[-5.0, -1.0, 0.0, 1.0, 5.0], 5.0), 0.0);
    assert_eq!(location(&[0.9, 1.0, 1.1, 1.2], 5.0), midpoint(1.0, 1.1));
    // Subnormal and huge windows, beyond what doubles hold unscaled.
    let tiny = f64::from_bits(1);
    assert_eq!(location(&[tiny, 2.0 * tiny], 5.0), 2.0 * tiny);
    assert_eq!(location(&[3.0 * tiny, 0.0, 6.0 * tiny], 5.0), 3.0 * tiny);
    let max = f64::MAX;
    assert_eq!(location(&[-max, max, max], 5.0), max);
    assert_eq!(location(&[-max, 0.0, max], 5.0), 0.0);
}

#[test]
fn no_point_within_c_mads() {
    // c below 1: at the median no point is within c MAD, so no step.
    assert_eq!(location(&[0.0, 1.0, 2.0, 3.0], 0.5), 1.5);
}

#[test]
fn the_sign_change_and_ties() {
    let eps = f64::EPSILON;
    // Roots at 1 + 1.5 eps and 1 + 2.5 eps: midway, to the even double.
    let at = |root_in_eps: f64| move |t: f64| 2.0 * root_in_eps - 2.0 * (t - 1.0) / eps;
    assert_eq!(
        nearest_root(1.0, f64::from_bits(1), &at(1.5)),
        1.0 + 2.0 * eps
    );
    assert_eq!(
        nearest_root(1.0, f64::from_bits(1), &at(2.5)),
        1.0 + 2.0 * eps
    );
    assert_eq!(
        nearest_root(1.0 + 4.0 * eps, f64::from_bits(1), &at(2.5)),
        1.0 + 2.0 * eps
    );
    // Nearer one side or the other.
    assert_eq!(nearest_root(1.0, f64::from_bits(1), &at(1.25)), 1.0 + eps);
    assert_eq!(
        nearest_root(1.0, f64::from_bits(1), &at(1.75)),
        1.0 + 2.0 * eps
    );
    // A root on a double.
    assert_eq!(
        nearest_root(1.0, f64::from_bits(1), &at(3.0)),
        1.0 + 3.0 * eps
    );
    assert_eq!(nearest_root(1.0, f64::from_bits(1), &at(0.0)), 1.0);
    // Coarser than the doubles: multiples of 4 eps, from off them.
    let fine = 4.0 * eps;
    assert_eq!(nearest_root(1.0 + eps, fine, &at(5.0)), 1.0 + 4.0 * eps);
    assert_eq!(nearest_root(1.0 + eps, fine, &at(7.0)), 1.0 + 8.0 * eps);
    assert_eq!(nearest_root(1.0, fine, &at(6.0)), 1.0 + 8.0 * eps);
    assert_eq!(nearest_root(1.0, fine, &at(-6.0)), 1.0 - 8.0 * eps);
    // Down through zero, and no sign change within reach.
    let tiny = f64::from_bits(1);
    assert_eq!(
        nearest_root(2.0 * tiny, tiny, &|t| -(t / tiny) - 3.0),
        -3.0 * tiny
    );
    assert_eq!(
        nearest_root(2.0 * tiny, tiny, &|t| -(t / tiny) - 2.5),
        -2.0 * tiny
    );
    assert_eq!(nearest_root(-2.0 * tiny, tiny, &|t| -(t / tiny) + 0.5), 0.0);
    assert_eq!(nearest_root(1.0, f64::from_bits(1), &|_| 1.0), 1.0);
}

#[test]
fn the_trend_is_each_windows_biweight() {
    let mut random = Random(0x5eed_0303);
    let n = 400;
    let mut t = 0.0;
    let times: Vec<f64> = (0..n)
        .map(|_| {
            t += 0.01 * random.unit();
            t
        })
        .collect();
    let flux: Vec<f64> = (0..n).map(|_| 1.0 + 1e-3 * (random.unit() - 0.5)).collect();
    let masked: Vec<bool> = (0..n).map(|_| random.below(10) == 0).collect();
    let o = WindowOptions {
        window: 0.3,
        break_tolerance: f64::INFINITY,
        edge_cutoff: 0.1,
        edges: Edges::Symmetric,
    };
    let found = trend(&times, &flux, Some(&masked), &o, Location::Biweight(5.0));
    let w = windows(&times, Some(&masked), &o);
    let mut some = 0;
    for (i, window) in w.windows.iter().enumerate() {
        let expected = window.clone().filter(|r| !r.is_empty()).map(|r| {
            let values: Vec<f64> = w.kept[r].iter().map(|&k| flux[k]).collect();
            location(&values, 5.0)
        });
        assert_eq!(found[i], expected, "point {i}");
        some += usize::from(expected.is_some());
    }
    assert!(some > 300 && some < n, "{some}");
}
