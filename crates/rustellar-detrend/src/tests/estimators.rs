// Welsch, Andrews' sine wave, Huber, Hodges-Lehmann, trimmed and winsorized
// means: the series at 50 digits, brute force, exact symmetries, and the two
// builds. (Against the reference windows: the integration tests.)

use super::Random;
use crate::{
    exact_sum::ExactSum,
    m_estimate::{Scratch, solve},
    order::{Rows, cut, hodges_lehmann, trimmed_mean, winsorized_mean},
    psi::{Andrews, Huber, Psi, Welsch, cos, exp_negative, exp_negative_double, sinc, sinc_double},
};

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

fn m<P: Psi>(values: &[f64], c: f64) -> f64 {
    solve::<P>(&sorted(values), &mut Scratch::default(), c)
}

fn estimate(method: &str, values: &[f64], c: f64) -> f64 {
    let v = sorted(values);
    let mut sum = ExactSum::default();
    match method {
        "welsch" => m::<Welsch>(values, c),
        "andrews" => m::<Andrews>(values, c),
        "huber" => m::<Huber>(values, c),
        "hodges" => hodges_lehmann(&v, &mut Rows::default(), &mut sum),
        "trimmed" => trimmed_mean(&v, c, &mut sum),
        "winsorized" => winsorized_mean(&v, c, &mut sum),
        _ => unreachable!("{method}"),
    }
}

fn ulp(x: f64) -> f64 {
    let x = x.abs();
    if x < f64::MIN_POSITIVE {
        f64::from_bits(1)
    } else {
        f64::from_bits(x.to_bits() + 1) - x
    }
}

fn times_two_to(mut x: f64, mut power: i32) -> f64 {
    while power != 0 {
        let step = power.clamp(-1000, 1000);
        x *= 2f64.powi(step);
        power -= step;
    }
    x
}

// ==========================
// The weights' series
// ==========================

// (function, argument, value at 50 digits as hi + lo), from mpmath.
const SERIES: [(&str, f64, f64, f64); 13] = [
    ("exp", -0.5, 0.6065306597126334, -6.593178415491414e-19),
    ("exp", -1.0, 0.36787944117144233, -1.2428753672788363e-17),
    ("exp", -0.3466, 0.7070881069410189, -4.768387989392634e-17),
    ("exp", -700.25, 7.678723813110872e-305, 3.44e-321),
    ("exp", -1e-10, 0.9999999999, 8.279037096265651e-18),
    ("exp", -37.5, 5.175555005801869e-17, -2.3609618230840602e-33),
    ("sinc", 1.0, 0.8414709848078965, 1.776845092935536e-18),
    ("sinc", 0.25, 0.958851077208406, -1.0207939721112026e-17),
    ("sinc", 9.0, 0.04704000268662241, -1.766839340265652e-18),
    ("sinc", 9.8, 0.003544888814891375, -2.6826490374394688e-20),
    ("cos", 2.0, 0.15594369476537448, -6.372354595075329e-18),
    ("cos", 9.8, -0.9999384235442913, -3.612269714511659e-17),
    ("cos", 0.01, 0.9950041652780258, -5.457073892582271e-17),
];

#[test]
fn the_series_at_fifty_digits() {
    let eps = f64::EPSILON;
    for (name, x, hi, lo) in SERIES {
        let exact = hi + lo;
        match name {
            "exp" => {
                // Plain within 4 eps relative; double-double within 1e-30.
                assert!((exp_negative(x) - exact).abs() <= 4.0 * eps * hi, "{x}");
                let d = exp_negative_double((x, 0.0));
                let error = (d.0 - hi) + (d.1 - lo);
                assert!(
                    error.abs() <= 1e-30 * hi.max(1e-300) + 1e-320,
                    "{x}: {error:e}"
                );
            }
            "sinc" => {
                // Plain within 8 eps absolute, as the bound takes it.
                assert!((sinc(x) - exact).abs() <= 8.0 * eps, "{x}");
                let d = sinc_double((x, 0.0));
                assert!(((d.0 - hi) + (d.1 - lo)).abs() <= 1e-31, "{x}");
            }
            _ => assert!((cos(x) - exact).abs() <= 8.0 * eps, "{x}"),
        }
    }
    assert_eq!(exp_negative(-800.0), 0.0);
    assert_eq!(exp_negative_double((-800.0, 0.0)), (0.0, 0.0));
    assert_eq!(exp_negative(0.0), 1.0);
}

// Each plain coefficient is the double nearest its true value: the high
// part of the double-double tables (whose two parts come from 50-digit
// values), and for the cosine's (-1)^j / (2j)! = (2j + 1) (-1)^j / (2j + 1)!,
// that product of the sine's double-double, rounded once.
#[test]
fn the_coefficients_are_correctly_rounded() {
    use crate::{
        double::{multiply, two_sum},
        psi::{COS, EXP, EXP_DOUBLE, SINC, SINC_DOUBLE},
    };
    // The tables run from the highest power down; the plain ones up.
    for (j, c) in EXP.iter().enumerate().take(14) {
        assert_eq!(c.to_bits(), EXP_DOUBLE[27 - j].0.to_bits(), "exp {j}");
    }
    assert_eq!(EXP[14..], [0.0, 0.0]);
    for (j, c) in SINC.iter().enumerate() {
        assert_eq!(c.to_bits(), SINC_DOUBLE[24 - j].0.to_bits(), "sinc {j}");
    }
    for (j, c) in COS.iter().enumerate() {
        let product = multiply(SINC_DOUBLE[24 - j], ((2 * j + 1) as f64, 0.0));
        let (nearest, _) = two_sum(product.0, product.1);
        assert_eq!(c.to_bits(), nearest.to_bits(), "cos {j}");
    }
}

#[test]
fn the_series_across_their_range() {
    // Plain against double-double: within the errors the bounds assume.
    let eps = f64::EPSILON;
    for k in 0..=20_000 {
        let x = -(k as f64) / 20_000.0 * 40.0;
        let d = exp_negative_double((x, 0.0));
        assert!(
            (exp_negative(x) - d.0).abs() <= 4.0 * eps * d.0 + 1e-320,
            "{x}"
        );
        let y = k as f64 / 20_000.0 * std::f64::consts::PI.powi(2);
        let s = sinc_double((y, 0.0));
        assert!((sinc(y) - s.0).abs() <= 8.0 * eps, "{y}");
        // cos^2 + (u sinc)^2 = 1.
        let (c, u) = (cos(y), y.sqrt());
        assert!((c * c + (u * s.0).powi(2) - 1.0).abs() <= 16.0 * eps, "{y}");
    }
}

// ==========================
// References
// ==========================

// ==========================
// Brute force
// ==========================

#[test]
fn hodges_lehmann_by_brute_force() {
    let mut random = Random(0x5eed_0601);
    let mut rows = Rows::default();
    let mut sum = ExactSum::default();
    for case in 0..3000 {
        let n = 1 + random.below(60) as usize;
        let values: Vec<f64> = (0..n)
            .map(|_| match case % 4 {
                0 => random.below(6) as f64,
                1 => 1.0 + 1e-3 * (random.unit() - 0.5),
                2 => (random.unit() - 0.5) * 10f64.powi(random.below(40) as i32 - 20),
                _ => f64::from_bits(random.below(1 << 12)),
            })
            .collect();
        let v = sorted(&values);
        // Every average exactly (two-sums), sorted, the middle one or two.
        let mut pairs: Vec<(f64, f64, f64, f64)> = Vec::new();
        for i in 0..n {
            for j in i..n {
                let s = v[i] + v[j];
                let e = (v[i] - (s - (s - v[i]))) + (v[j] - (s - v[i]));
                pairs.push((s, e, v[i], v[j]));
            }
        }
        pairs.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.total_cmp(&q.1)));
        let total = pairs.len();
        let mut exact = ExactSum::default();
        let a = pairs[(total - 1) / 2];
        exact.add(a.2);
        exact.add(a.3);
        let expected = if total % 2 == 1 {
            exact.mean(2)
        } else {
            let b = pairs[total / 2];
            exact.add(b.2);
            exact.add(b.3);
            exact.mean(4)
        };
        let found = hodges_lehmann(&v, &mut rows, &mut sum);
        assert_eq!(found.to_bits(), expected.to_bits(), "{v:?}");
    }
}

#[test]
fn trimming_and_winsorizing_by_definition() {
    let mut random = Random(0x5eed_0602);
    let mut sum = ExactSum::default();
    for _ in 0..3000 {
        let n = 1 + random.below(50) as usize;
        let v = sorted(
            &(0..n)
                .map(|_| random.below(1000) as f64)
                .collect::<Vec<_>>(),
        );
        let p = [0.0, 0.1, 0.2, 0.25, 0.3, 0.49][random.below(6) as usize];
        let g = cut(p, n);
        // Integers: the sums are exact, the quotient rounds once.
        let kept: f64 = v[g..n - g].iter().sum();
        assert_eq!(trimmed_mean(&v, p, &mut sum), kept / (n - 2 * g) as f64);
        let winsorized = kept + g as f64 * (v[g] + v[n - 1 - g]);
        assert_eq!(winsorized_mean(&v, p, &mut sum), winsorized / n as f64);
    }
    // The rounded product, as scipy and wotan: 0.3 of 10 is 3.
    assert_eq!(cut(0.3, 10), 3);
    assert_eq!(cut(0.1, 10), 1);
    assert_eq!(cut(0.0, 10), 0);
    assert_eq!(cut(0.49, 3), 1);
    // wotan's winsorized mean of 0..19 at 0.1 is 9.65; symmetric data's is
    // the middle.
    let v: Vec<f64> = (0..20).map(f64::from).collect();
    assert_eq!(winsorized_mean(&v, 0.1, &mut sum), 9.5);
}

// ==========================
// Exact symmetries
// ==========================

#[test]
fn scaling_negation_and_order() {
    let mut random = Random(0x5eed_0603);
    let methods = [
        ("welsch", 2.11),
        ("andrews", 1.339),
        ("huber", 1.28),
        ("hodges", 0.0),
        ("trimmed", 0.1),
        ("winsorized", 0.2),
    ];
    for _ in 0..300 {
        let n = 1 + random.below(60) as usize;
        let values: Vec<f64> = (0..n)
            .map(|_| {
                1.0 + 1e-3 * (random.unit() - 0.5) + if random.below(10) == 0 { 0.05 } else { 0.0 }
            })
            .collect();
        let p = random.below(400) as i32 - 200;
        for (method, c) in methods {
            let found = estimate(method, &values, c);
            let scaled: Vec<f64> = values.iter().map(|&x| times_two_to(x, p)).collect();
            assert_eq!(
                estimate(method, &scaled, c),
                times_two_to(found, p),
                "{method} 2^{p}"
            );
            let negated: Vec<f64> = values.iter().map(|&x| -x).collect();
            assert_eq!(estimate(method, &negated, c), -found, "{method} negated");
            let mut reversed = values.clone();
            reversed.reverse();
            assert_eq!(
                estimate(method, &reversed, c).to_bits(),
                found.to_bits(),
                "{method}"
            );
        }
    }
}

#[test]
fn small_windows_and_no_spread() {
    for method in ["welsch", "andrews", "huber"] {
        assert_eq!(
            estimate(method, &[1.0, 1.0, 1.0, 2.0, 30.0], 2.0),
            1.0,
            "{method}"
        );
        assert_eq!(estimate(method, &[0.7], 2.0), 0.7);
        assert_eq!(estimate(method, &[0.0], 2.0), 0.0);
        // Symmetric: the middle.
        assert_eq!(estimate(method, &[-5.0, -1.0, 0.0, 1.0, 5.0], 2.0), 0.0);
        // A tight cluster beside distant values.
        let unit = times_two_to(1.0, -1070);
        let values = [-1.0, unit, 2.0 * unit, 3.0 * unit, 1.0];
        assert_eq!(estimate(method, &values, 2.0), 2.0 * unit, "{method}");
    }
    assert_eq!(estimate("hodges", &[0.7], 0.0), 0.7);
    assert_eq!(estimate("hodges", &[1.0, 2.0], 0.0), 1.5);
    let max = f64::MAX;
    assert_eq!(estimate("hodges", &[max, max], 0.0), max);
    assert_eq!(estimate("trimmed", &[max, max, -max], 0.0), max / 3.0);
}

// Huber's residual is linear between the points' entries into and exits
// from the clipped range, so its root is a ratio of exact sums: checked
// against it, in integers, where it falls within one segment.
#[test]
fn huber_is_the_exact_ratio() {
    let mut random = Random(0x5eed_0604);
    for _ in 0..2000 {
        let n = 3 + random.below(40) as usize;
        let values: Vec<f64> = (0..n).map(|_| random.below(64) as f64).collect();
        let c = [0.5, 1.0, 1.28, 2.0, 3.0][random.below(5) as usize];
        let found = estimate("huber", &values, c);
        // At T = found, clip(x - T, -s, s) sums to within a few roundings
        // of zero, and the exact root (a ratio) rounds to it.
        let v = sorted(&values);
        let median = (v[(n - 1) / 2] + v[n / 2]) / 2.0;
        let mut deviations: Vec<f64> = v.iter().map(|x| (x - median).abs()).collect();
        deviations.sort_by(f64::total_cmp);
        let mad = (deviations[(n - 1) / 2] + deviations[n / 2]) / 2.0;
        if mad == 0.0 {
            assert_eq!(found, median);
            continue;
        }
        let s = c * mad;
        // The segment at `found`: points inside, above and below.
        let (mut inside, mut sum, mut above, mut below) = (0.0, 0.0, 0.0, 0.0);
        for &x in &v {
            if x - found >= s {
                above += 1.0;
            } else if found - x >= s {
                below += 1.0;
            } else {
                inside += 1.0;
                sum += x;
            }
        }
        // T = (sum inside + s (above - below)) / inside, if consistent.
        let root = (sum + s * (above - below)) / inside;
        let consistent = v.iter().all(|&x| {
            let d = x - root;
            (d.abs() < s) == ((x - found).abs() < s)
        });
        if inside > 0.0 && consistent {
            assert!(
                (found - root).abs() <= 2.0 * ulp(root),
                "{v:?} c {c}: {found} {root}"
            );
        }
    }
}

// ==========================
// The two builds
// ==========================

// AVX2's weights, value for value the plain ones, across and beyond the
// ranges the solver meets (where the processor has AVX2).
#[cfg(target_arch = "x86_64")]
#[test]
fn the_wide_weights_are_the_plain_ones() {
    use crate::psi::{Avx2, Lanes, Plain};
    let Some(wide) = Avx2::new() else {
        return;
    };
    let same = |a: Lanes, b: Lanes, what: &str| {
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}: {a:?} {b:?}");
        }
    };
    let mut random = Random(0x5eed_0601);
    for k in 0..40_000 {
        // Sizes from 0 to beyond where e^(-u^2 / 2) underflows.
        let reach = [1e-8, 1.0, std::f64::consts::PI, 40.0][k % 4];
        let u: Lanes = std::array::from_fn(|_| (2.0 * random.unit() - 1.0) * reach);
        let (a, b) = (Welsch::weigh(Plain, u), Welsch::weigh(wide, u));
        same(a.0, b.0, "welsch w");
        same(a.1, b.1, "welsch m");
        same(a.2, b.2, "welsch slope");
        same(
            Welsch::rough(Plain, u),
            Welsch::rough(wide, u),
            "welsch rough",
        );
        let inside = std::f64::consts::PI - 1e-3;
        let u = u.map(|x| x.clamp(-inside, inside));
        let (a, b) = (Andrews::weigh(Plain, u), Andrews::weigh(wide, u));
        same(a.0, b.0, "andrews w");
        same(a.2, b.2, "andrews slope");
        same(
            Andrews::rough(Plain, u),
            Andrews::rough(wide, u),
            "andrews rough",
        );
    }
    let edges = [0.0, -0.0, 1e-300, 38.5, 38.6, 38.7, 39.0];
    let u: Lanes = std::array::from_fn(|i| edges[i]);
    same(Welsch::weigh(Plain, u).0, Welsch::weigh(wide, u).0, "edges");
    let u: Lanes = std::array::from_fn(|i| edges[i + 3]);
    same(Welsch::weigh(Plain, u).0, Welsch::weigh(wide, u).0, "edges");
}

// The double-double terms, four lanes in four groups at a time, by either
// build: the scalar ones to the bit (every lane takes the same steps).
#[test]
fn the_grouped_terms_are_the_scalar_ones() {
    use crate::psi::{Biweight, Doubles, Isa, Plain};
    fn check<P: Psi, V: Isa>(v: V, random: &mut Random, reach: f64, what: &str) {
        let mut pick = |size: f64| -> [Doubles; 4] {
            std::array::from_fn(|_| {
                let hi: [f64; 4] = std::array::from_fn(|_| (2.0 * random.unit() - 1.0) * size);
                let lo = std::array::from_fn(|i| hi[i] * f64::EPSILON * (random.unit() - 0.5));
                (hi, lo)
            })
        };
        let (mut u, d) = (pick(reach), pick(1e-3));
        // Zero, and (for Welsch) past where e^(-u^2 / 2) underflows.
        u[0].0[0] = 0.0;
        u[0].1[0] = 0.0;
        u[1].0[1] = reach;
        u[1].1[1] = 0.0;
        // Just inside the cut at |u| = 1: a high part of exactly 1 and a
        // tail toward zero, either sign.
        u[2].0[2] = 1.0;
        u[2].1[2] = -1e-17;
        u[2].0[3] = -1.0;
        u[2].1[3] = 1e-17;
        let s = (2.5e-3, 1e-20);
        let grouped = P::terms(v, u, d, s);
        for g in 0..4 {
            for l in 0..4 {
                let scalar = P::term((u[g].0[l], u[g].1[l]), (d[g].0[l], d[g].1[l]), s);
                let lane = (grouped[g].0[l], grouped[g].1[l]);
                assert_eq!(
                    (lane.0.to_bits(), lane.1.to_bits()),
                    (scalar.0.to_bits(), scalar.1.to_bits()),
                    "{what}: u {:?}",
                    (u[g].0[l], u[g].1[l])
                );
            }
        }
    }
    fn all<V: Isa>(v: V) {
        let mut random = Random(0x5eed_0702);
        for _ in 0..2_000 {
            check::<Biweight, V>(v, &mut random, 1.0, "biweight");
            check::<Welsch, V>(v, &mut random, 40.0, "welsch");
            check::<Welsch, V>(v, &mut random, 2.0, "welsch");
            check::<Andrews, V>(v, &mut random, std::f64::consts::PI - 1e-3, "andrews");
            check::<Huber, V>(v, &mut random, 2.0, "huber");
        }
    }
    all(Plain);
    #[cfg(target_arch = "x86_64")]
    if let Some(wide) = crate::psi::Avx2::new() {
        all(wide);
    }
}

// The solver's baseline build gives the bits its AVX2 build does, for each
// M-estimator, on awkward windows.
#[cfg(target_arch = "x86_64")]
#[test]
fn the_baseline_build_is_the_wide_one() {
    use crate::m_estimate::wide;
    use crate::{m_estimate::NARROW, psi::Biweight};
    NARROW.with(|n| n.set(true));
    assert!(wide().is_none(), "the baseline build not taken");
    NARROW.with(|n| n.set(false));
    let run = |narrow: bool, method: &str, values: &[f64], c: f64| {
        NARROW.with(|n| n.set(narrow));
        let found = match method {
            "biweight" => m::<Biweight>(values, c),
            _ => estimate(method, values, c),
        };
        NARROW.with(|n| n.set(false));
        found
    };
    let mut random = Random(0x5eed_0603);
    let mut windows = super::awkward_windows(&mut random, 1500);
    // Symmetric about 0, MAD m, and a point x at the double just below c m
    // (where c m is not a double): its |u| = x / (c m) is 1 less a tail, just
    // inside Huber's cut. The location is 0.
    let mut edges = 0;
    for m in [3.0, 5.0, 7.0, 11.0] {
        for c in [1.1, 1.28, 1.3, 1.7] {
            let x: f64 = c * m;
            let x = if c.mul_add(m, -x) < 0.0 {
                f64::from_bits(x.to_bits() - 1)
            } else {
                x
            };
            if c.mul_add(m, -x) > 0.0 {
                let values = vec![-x, -x, -m, -m, 0.0, m, m, x, x];
                assert_eq!(estimate("huber", &values, c), 0.0, "c {c}, m {m}");
                windows.push((values, c));
                edges += 1;
            }
        }
    }
    assert!(edges >= 8, "{edges}");
    // Hand-built: MAD 0, one and two points, points exactly at and a hair
    // either side of the biweight's cut (c MAD = 5), and clusters.
    let hand_built: [&[f64]; 18] = [
        &[1.0, 1.0, 1.0, 2.0, 3.0],
        &[0.3, 0.3, 0.3, 0.3],
        &[0.7],
        &[0.1, 0.2],
        &[1.0, 1.000_000_000_000_000_2],
        &[-3.0, 1e-17],
        &[-5.0, -1.0, 0.0, 1.0, 5.0],
        &[-5.0, -1.0, 0.0, 1.0, 4.999_999_999],
        &[-5.0, -1.0, 0.0, 1.0, 5.000_000_001],
        &[-4.999_999_999, -1.0, 0.0, 1.0, 5.0],
        &[-5.0, -1.0, 0.0, 1.0, 4.999_999_999_999_999],
        &[-5.0, -1.0, 0.0, 1.0, 5.000_000_000_000_001],
        &[-4.999_999_999_999_999, -1.0, 0.0, 1.0, 5.0],
        &[-5.000_000_000_000_001, -1.0, 0.0, 1.0, 5.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4],
        &[1.0, 1.001, 1.002, 1.003, 1.0045, 1.006, 1.2, 1.21],
        &[-1e300, -1.0, 0.0, 1.0, 1e300],
        &[f64::MIN_POSITIVE, 2.0 * f64::MIN_POSITIVE, 1.0],
    ];
    for values in hand_built {
        let mut values = values.to_vec();
        values.sort_by(f64::total_cmp);
        for c in [1.0, 5.0] {
            windows.push((values.clone(), c));
        }
    }
    for (values, c) in &windows {
        for method in ["biweight", "welsch", "andrews", "huber"] {
            assert_eq!(
                run(true, method, values, *c).to_bits(),
                run(false, method, values, *c).to_bits(),
                "{method} c {c}: {values:?}"
            );
        }
    }
}
