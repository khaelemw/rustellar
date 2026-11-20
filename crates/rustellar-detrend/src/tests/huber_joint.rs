// Huber's proposal 2: gamma against mpmath at 60 digits, solutions against
// the two equations, bisection against split and solve, the small and
// degenerate windows, and statsmodels. (The survey at 50 digits: the
// integration tests.)

use super::{Random, awkward_windows};
use crate::double::{exponent, scale};
use crate::huber_joint::{JointScratch, ROUNDS, divide, gamma, huber_joint, joint, square_root};

// ==========================
// Helper Functions
// ==========================

// The two equations' residuals at (mu, s), relative to s: the clipped mean
// less mu, and the scale equation's sides' difference over its scale.
// Both in units of s, so nothing overflows.
fn residuals(x: &[f64], c: f64, (mu, s): (f64, f64)) -> (f64, f64) {
    let n = x.len() as f64;
    let u = |v: f64| (v - mu) / s;
    let clipped = x.iter().map(|&v| u(v).clamp(-c, c)).sum::<f64>() / n;
    let sum: f64 = x.iter().map(|&v| (u(v) * u(v)).min(c * c)).sum();
    let gamma = gamma(c).0;
    (clipped, sum / ((n - 1.0) * gamma) - 1.0)
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values
}

// ==========================
// Tests
// ==========================

#[test]
fn gamma_against_mpmath() {
    // (c, gamma's double, the rest) from mpmath at 60 digits.
    let cases = [
        (1e-3, 9.99468077012657e-07, 4.0475217675100494e-23),
        (0.5, 0.18512836514672018, -3.0520315489690503e-18),
        (1.345, 0.7101645482690485, 8.379542367610921e-19),
        (1.5, 0.77846521617447, 2.827749038953398e-17),
        (3.0, 0.9950072780344534, 3.426124230337118e-17),
        (6.0, 0.9999999961505409, 3.9585367587789275e-17),
        (6.5, 0.9999999998427769, 5.901331618487962e-18),
        (9.0, 1.0, -4.461779337425963e-19),
        (13.0, 1.0, -2.432797857301142e-38),
        (50.0, 1.0, 0.0),
    ];
    for (c, hi, lo) in cases {
        let (h, l) = gamma(c);
        let error = ((h - hi) + (l - lo)).abs() / hi;
        let bound = if c <= 3.0 { 1e-30 } else { 1e-23 };
        assert!(error < bound, "c {c}: {error:e}");
    }
    // Far below 1, c^2 less its next term (mpmath's closed form cancels).
    let c = 1e-30;
    let (h, l) = gamma(c);
    let expected = c * c - 2.0 / 3.0 * (2.0 / std::f64::consts::PI).sqrt() * c * c * c;
    assert!(((h + l) / expected - 1.0).abs() < 1e-15);
    assert_eq!(gamma(1e30), (1.0, 0.0));
}

#[test]
fn double_double_division_and_root() {
    let third = divide((1.0, 0.0), (3.0, 0.0));
    let back = crate::double::multiply(third, (3.0, 0.0));
    assert!(((back.0 - 1.0) + back.1).abs() < 1e-31);
    let root = square_root((2.0, 0.0));
    let square = crate::double::multiply(root, root);
    assert!(((square.0 - 2.0) + square.1).abs() < 1e-31);
    assert_eq!(square_root((0.0, 0.0)), (0.0, 0.0));
    assert_eq!(square_root((-1e-40, 0.0)), (0.0, 0.0));
}

#[test]
fn solutions_solve_both_equations() {
    let mut random = Random(0x6875_6265_7232);
    let mut scratch = JointScratch::default();
    for (values, _) in awkward_windows(&mut random, 4000) {
        // Scaled (exactly) to magnitudes about 1, so the answer is not
        // subnormal, with too few bits to check the equations to 1e-9.
        let largest = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let power = if largest > 0.0 { exponent(largest) } else { 0 };
        let x = sorted(values.iter().map(|&v| scale(v, -power)).collect());
        for c in [0.5, 1.5, 3.0] {
            let (mu, s) = joint(&x, &mut scratch, c, gamma(c), ROUNDS);
            assert!(mu.is_finite() && s >= 0.0, "{x:?}");
            assert!((x[0]..=x[x.len() - 1]).contains(&mu));
            // (A subnormal scale has too few bits for 1e-9.)
            if s > f64::MIN_POSITIVE && x.len() > 1 {
                let (a, b) = residuals(&x, c, (mu, s));
                assert!(a.abs() < 1e-9 && b.abs() < 1e-9, "c {c}: {a:e} {b:e} {x:?}");
            }
        }
    }
}

#[test]
fn bisection_alone_agrees() {
    // No split-and-solve rounds: bisection finds the split, the
    // same solution follows.
    let mut random = Random(0x6661_6c6c);
    let mut scratch = JointScratch::default();
    for (values, _) in awkward_windows(&mut random, 600) {
        let x = sorted(values);
        let g = gamma(1.5);
        let (mu, s) = joint(&x, &mut scratch, 1.5, g, ROUNDS);
        let (mu0, s0) = joint(&x, &mut scratch, 1.5, g, 0);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * s.max(f64::MIN_POSITIVE);
        assert!(close(mu, mu0) && close(s, s0), "{mu} {mu0}, {s} {s0} {x:?}");
    }
}

#[test]
fn small_and_degenerate_windows() {
    let mut scratch = JointScratch::default();
    let g = gamma(1.5);
    let mut location = |x: &[f64]| huber_joint(x, &mut scratch, 1.5, g);
    assert_eq!(location(&[3.5]), 3.5);
    assert_eq!(location(&[2.0, 2.0, 2.0]), 2.0);
    // Two values: their mean, with the scale their spread.
    assert_eq!(location(&[1.0, 2.0]), 1.5);
    // Mostly equal values: scale 0, the location their value.
    assert_eq!(location(&[1.0, 1.0, 1.0, 1.0, 5.0]), 1.0);
    // Symmetric: the middle.
    assert_eq!(location(&[-3.0, -1.0, 0.0, 1.0, 3.0]), 0.0);
    // Scaled by a power of two, scaled the same.
    let x = [0.98, 0.99, 1.0, 1.003, 1.01, 1.02, 1.5, 0.7];
    let x = sorted(x.to_vec());
    let big: Vec<f64> = x.iter().map(|v| v * 2f64.powi(600)).collect();
    let small: Vec<f64> = x.iter().map(|v| v * 2f64.powi(-600)).collect();
    let base = location(&x);
    assert_eq!(location(&big), base * 2f64.powi(600));
    assert_eq!(location(&small), base * 2f64.powi(-600));
}

#[test]
fn statsmodels_on_two_windows() {
    // statsmodels 0.15's Huber(c=1.5), tol 1e-15: the same to its
    // tolerance.
    let mut scratch = JointScratch::default();
    let g = gamma(1.5);
    for (x, mu, s) in [
        (
            vec![1.0, 2.0, 3.0, 4.0, 100.0],
            4.027470541352249,
            4.073254776939338,
        ),
        (
            vec![0.99, 1.01, 1.0, 0.98, 1.02, 1.5, 0.7, 1.003],
            1.0005,
            0.032578473759308536,
        ),
    ] {
        let (m, t) = joint(&sorted(x), &mut scratch, 1.5, g, ROUNDS);
        assert!(
            (m - mu).abs() < 1e-12 * s && (t / s - 1.0).abs() < 1e-12,
            "{m} {t}"
        );
    }
}
