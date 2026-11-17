// Welsch, Andrews' sine wave, Huber, Hodges-Lehmann, trimmed and winsorized
// means: against 50-digit and exact references, brute force, exact
// symmetries, and wotan 1.10 measured the same way.

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

struct Row {
    kind: String,
    method: String,
    c: f64,
    e: i32,
    hi: f64,
    lo: f64,
    sensitivity: f64,
    wotan: f64,
    values: Vec<f64>,
}

fn rows() -> Vec<Row> {
    include_str!("../../tests/reference/estimator_windows.csv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            Row {
                kind: f[1].to_string(),
                method: f[2].to_string(),
                c: f[3].parse().unwrap(),
                e: f[4].parse().unwrap(),
                hi: f[5].parse().unwrap(),
                lo: f[6].parse().unwrap(),
                sensitivity: f[8].parse().unwrap(),
                wotan: f[9].parse().unwrap(),
                values: f[10].split(';').map(|v| v.parse().unwrap()).collect(),
            }
        })
        .collect()
}

#[test]
fn the_reference_windows() {
    let rows = rows();
    assert!(rows.len() > 1500);
    let eps = f64::EPSILON;
    let mut stats = std::collections::BTreeMap::<String, [usize; 4]>::new();
    for r in &rows {
        let found = estimate(&r.method, &r.values, r.c);
        let entry = stats.entry(r.method.clone()).or_default();
        entry[0] += 1;
        if matches!(r.method.as_str(), "welsch" | "andrews" | "huber") {
            // As the biweight's: the nearest double to the root wherever the
            // root is further from a midpoint than its rounding can move it.
            let scaled = times_two_to(found, -r.e);
            let error = ((scaled - r.hi) - r.lo).abs();
            let doubt = 8.0 * eps * eps * r.sensitivity;
            // The doubles' spacing, or the subnormals' once scaled back.
            let unit = ulp(r.hi).max(times_two_to(f64::from_bits(1), (-r.e).max(0)));
            assert!(
                error <= 0.5 * unit + doubt,
                "{} {} c {}: {scaled:e} for {:e} + {:e} ({:e})",
                r.kind,
                r.method,
                r.c,
                r.hi,
                r.lo,
                error
            );
            // An exact midpoint (the mean of two points, say) is a tie, and
            // ties go to the even double of the grid.
            let tie = doubt < 1e-6 * unit && (error - 0.5 * unit).abs() <= doubt;
            if tie {
                entry[1] += 1;
                assert_eq!(
                    (scaled / unit) % 2.0,
                    0.0,
                    "{} {}: the tie not to even",
                    r.kind,
                    r.method
                );
            }
            let certain = (error - 0.5 * unit).abs() > doubt;
            if certain {
                entry[1] += 1;
                assert!(
                    error <= 0.5 * unit,
                    "{} {}: not the nearest",
                    r.kind,
                    r.method
                );
            }
            if r.wotan.is_finite() {
                let theirs = (times_two_to(r.wotan, -r.e) - r.hi - r.lo).abs();
                entry[2] += usize::from(theirs > error);
                entry[3] += usize::from(theirs < error);
            }
        } else {
            // Exact: the correctly rounded value, to the bit.
            assert_eq!(
                found.to_bits(),
                r.hi.to_bits(),
                "{} {} {}: {found:e} for {:e}",
                r.kind,
                r.method,
                r.c,
                r.hi
            );
            entry[1] += 1;
            if r.wotan.is_finite() {
                entry[2] += usize::from(r.wotan != r.hi);
            }
        }
    }
    for (method, [n, certain, wotan_worse, wotan_better]) in &stats {
        println!(
            "{method}: {n} windows, {certain} certain, wotan worse in {wotan_worse}, better in {wotan_better}"
        );
        assert!(
            *certain * 100 >= *n * 97,
            "{method}: {certain} of {n} certain"
        );
        // Andrews is wotan's estimator too: wotan never nearer.
        if method == "andrews" || method == "hodges" || method == "trimmed" {
            assert_eq!(*wotan_better, 0, "{method}");
        }
    }
}

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

// The solver's baseline build gives the bits its AVX2 build does, on every
// reference window.
#[cfg(target_arch = "x86_64")]
#[test]
fn the_baseline_build_is_the_wide_one() {
    use crate::m_estimate::wide;
    use crate::{m_estimate::NARROW, psi::Biweight};
    NARROW.with(|n| n.set(true));
    assert!(wide().is_none(), "the baseline build not taken");
    NARROW.with(|n| n.set(false));
    let run = |narrow: bool, r: &Row| {
        NARROW.with(|n| n.set(narrow));
        let found = match r.method.as_str() {
            "welsch" | "andrews" | "huber" => estimate(&r.method, &r.values, r.c),
            _ => m::<Biweight>(&r.values, 5.0),
        };
        NARROW.with(|n| n.set(false));
        found
    };
    for r in &rows() {
        assert_eq!(
            run(true, r).to_bits(),
            run(false, r).to_bits(),
            "{} {} c {}",
            r.kind,
            r.method,
            r.c
        );
    }
}
