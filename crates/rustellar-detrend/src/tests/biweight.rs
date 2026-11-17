// The biweight: against the root at 50 digits within a bound derived from
// the roundings (and wotan 1.10's error on the same windows), exact
// symmetries, the hand-built edges, and the sign-change search.

use super::Random;
use crate::{
    biweight::{Estimate, Scratch, biweight, nearest_root},
    location::{Location, midpoint},
    trend::trend,
    window::{Edges, WindowOptions, windows},
};

// ==========================
// Helper Functions
// ==========================

fn location(values: &[f64], c: f64) -> f64 {
    biweight(&mut values.to_vec(), &mut Scratch::default(), c)
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
fn a_tight_cluster_beside_distant_values() {
    // The cluster symmetric about its middle, the distant values beyond
    // c MAD: the middle, wherever c MAD falls (subnormal, or underflowing
    // to zero at the window's first scale).
    for p in [-1074, -1070, -1060, -1030, -1000, -700, -400] {
        let unit = times_two_to(1.0, p);
        for c in [1e-30, 0.3, 5.0, 1e5] {
            let cluster = [unit, 2.0 * unit, 3.0 * unit];
            let mut values = cluster.to_vec();
            values.extend([-1.0, 1.0]);
            assert_eq!(location(&values, c), 2.0 * unit, "2^{p}, c {c}");
            values.extend(cluster.map(|x| x + 1e300).iter().chain(&[-1e300, -1e300]));
            values.extend(cluster);
            values.extend(cluster);
            values.extend(cluster);
            assert_eq!(location(&values, c), 2.0 * unit, "2^{p}, c {c}");
        }
    }
    // c MAD far beyond the spread: every weight 1, the mean.
    assert_eq!(
        location(&[1.0, 1.0, 1.0 + f64::EPSILON, 1.5, 1.5], 1e30),
        1.2
    );
}

#[test]
fn no_point_within_c_mads() {
    // c below 1: at the median no point is within c MAD, so no step.
    assert_eq!(location(&[0.0, 1.0, 2.0, 3.0], 0.5), 1.5);
}

// The larger gap from `t` to the doubles either side.
fn gap(t: f64) -> f64 {
    let toward = |up: bool| {
        if t == 0.0 {
            f64::from_bits(1)
        } else if (t > 0.0) == up {
            f64::from_bits(t.to_bits() + 1)
        } else {
            f64::from_bits(t.to_bits() - 1)
        }
    };
    (toward(true) - t).abs().max((toward(false) - t).abs())
}

// The sign-change search from `t`, the residual there computed.
// Twice: with the residual given exactly, and with a cheap estimate whose
// bound is too wide to settle anything, so that every sign and the nearer
// side come from the precise residual. Both must agree.
fn root(t: f64, fine: f64, residual: &dyn Fn(f64) -> f64) -> f64 {
    let exact = |t: f64| Estimate {
        value: residual(t),
        bound: 0.0,
    };
    let doubtful = |t: f64| Estimate {
        value: residual(t),
        bound: f64::INFINITY,
    };
    let sure = nearest_root(t, exact(t), fine, &exact, residual);
    let unsure = nearest_root(t, doubtful(t), fine, &doubtful, residual);
    assert_eq!(sure.to_bits(), unsure.to_bits());
    sure
}

#[test]
fn the_sign_change_and_ties() {
    let eps = f64::EPSILON;
    // Roots at 1 + 1.5 eps and 1 + 2.5 eps: midway, to the even double.
    let at = |root_in_eps: f64| move |t: f64| 2.0 * root_in_eps - 2.0 * (t - 1.0) / eps;
    assert_eq!(root(1.0, f64::from_bits(1), &at(1.5)), 1.0 + 2.0 * eps);
    assert_eq!(root(1.0, f64::from_bits(1), &at(2.5)), 1.0 + 2.0 * eps);
    assert_eq!(
        root(1.0 + 4.0 * eps, f64::from_bits(1), &at(2.5)),
        1.0 + 2.0 * eps
    );
    // Nearer one side or the other.
    assert_eq!(root(1.0, f64::from_bits(1), &at(1.25)), 1.0 + eps);
    assert_eq!(root(1.0, f64::from_bits(1), &at(1.75)), 1.0 + 2.0 * eps);
    // A root on a double.
    assert_eq!(root(1.0, f64::from_bits(1), &at(3.0)), 1.0 + 3.0 * eps);
    assert_eq!(root(1.0, f64::from_bits(1), &at(0.0)), 1.0);
    // Coarser than the doubles: multiples of 4 eps, from off them.
    let fine = 4.0 * eps;
    assert_eq!(root(1.0 + eps, fine, &at(5.0)), 1.0 + 4.0 * eps);
    assert_eq!(root(1.0 + eps, fine, &at(7.0)), 1.0 + 8.0 * eps);
    assert_eq!(root(1.0, fine, &at(6.0)), 1.0 + 8.0 * eps);
    assert_eq!(root(1.0, fine, &at(-6.0)), 1.0 - 8.0 * eps);
    // Down through zero, and no sign change within reach.
    let tiny = f64::from_bits(1);
    assert_eq!(root(2.0 * tiny, tiny, &|t| -(t / tiny) - 3.0), -3.0 * tiny);
    assert_eq!(root(2.0 * tiny, tiny, &|t| -(t / tiny) - 2.5), -2.0 * tiny);
    assert_eq!(root(-2.0 * tiny, tiny, &|t| -(t / tiny) + 0.5), 0.0);
    assert_eq!(root(1.0, f64::from_bits(1), &|_| 1.0), 1.0);
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
    let found = trend(&times, &flux, Some(&masked), &o, Location::Biweight(5.0), 3);
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

// The solver written plainly: every value visited in every sum (ascending,
// those with |u| < 1 dealt to four lanes in turn), the MAD by sorting the
// deviations, the exact product by a fused multiply-add. The sorted solver
// (runs of |u| < 1 found by bisection, the MAD by merging outward, Dekker's
// product) must give the same bits.
fn plain(values: &[f64], c: f64) -> f64 {
    type Double = (f64, f64);
    fn two_sum(a: f64, b: f64) -> Double {
        let s = a + b;
        let bv = s - a;
        (s, (a - (s - bv)) + (b - bv))
    }
    fn two_product(a: f64, b: f64) -> Double {
        let p = a * b;
        (p, a.mul_add(b, -p))
    }
    fn add(a: Double, b: Double) -> Double {
        let (s, e) = two_sum(a.0, b.0);
        two_sum(s, e + a.1 + b.1)
    }
    fn multiply(a: Double, b: Double) -> Double {
        let (p, e) = two_product(a.0, b.0);
        two_sum(p, e + a.0 * b.1 + a.1 * b.0)
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    let (a, b) = (v[(n - 1) / 2], v[n / 2]);
    let largest = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if largest == 0.0 {
        return midpoint(a, b);
    }
    // The binary exponent: 2^e <= x < 2^(e + 1).
    let exponent = |x: f64| {
        if x == 0.0 {
            return -1075;
        }
        let e = x.log2().floor() as i32;
        if times_two_to(x, -e) >= 2.0 {
            e + 1
        } else if times_two_to(x, -e) < 1.0 {
            e - 1
        } else {
            e
        }
    };
    let mut e = exponent(largest);
    let (y, median, s) = loop {
        let y: Vec<f64> = v.iter().map(|&x| times_two_to(x, -e)).collect();
        let (sum, error) = two_sum(times_two_to(a, -e), times_two_to(b, -e));
        let median = (sum / 2.0, error / 2.0);
        let mut deviations: Vec<Double> = y
            .iter()
            .map(|&x| {
                let (d, error) = two_sum(x, -median.0);
                let d = two_sum(d, error - median.1);
                if d.0 < 0.0 { (-d.0, -d.1) } else { d }
            })
            .collect();
        deviations.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.total_cmp(&q.1)));
        let total = add(deviations[(n - 1) / 2], deviations[n / 2]);
        let mad = (total.0 / 2.0, total.1 / 2.0);
        if mad.0 == 0.0 {
            return midpoint(a, b);
        }
        let (hi, lo) = two_product(c, mad.0);
        let s = two_sum(hi, lo + c * mad.1);
        // c MAD below 2^-500: scaled up to 2^-500.
        if s.0 >= times_two_to(1.0, -500) {
            break (y, median, s);
        }
        e += exponent(s.0) + 500;
    };
    let inverse = 1.0 / s.0;
    // (residual, weight, slope), plain
    let sums = |t: f64| {
        let mut lanes = [[0.0; 3]; 4];
        let mut k = 0;
        for &x in &y {
            let d = x - t;
            let u = d * inverse;
            if u.abs() < 1.0 {
                let v = 1.0 - u * u;
                let lane = &mut lanes[k % 4];
                lane[0] += v * v * d;
                lane[1] += v * v;
                lane[2] += v * (1.0 - 5.0 * u * u);
                k += 1;
            }
        }
        let total = |k: usize| (lanes[0][k] + lanes[1][k]) + (lanes[2][k] + lanes[3][k]);
        (total(0), total(1), total(2))
    };
    // The residual compensated, with its bound.
    let cheap = |t: f64| {
        let mut lanes = [[0.0; 3]; 4];
        let mut k = 0;
        for &x in &y {
            let (d, d_low) = two_sum(x, -t);
            let u = d * inverse;
            if u.abs() < 1.0 {
                let v = 1.0 - u * u;
                let lane = &mut lanes[k % 4];
                let (sum, error) = two_sum(lane[0], v * v * d);
                lane[0] = sum;
                lane[1] += error + v * v * d_low;
                lane[2] += d.abs() * v;
                k += 1;
            }
        }
        let total = |k: usize| (lanes[0][k] + lanes[1][k]) + (lanes[2][k] + lanes[3][k]);
        let high = add(
            two_sum(lanes[0][0], lanes[1][0]),
            two_sum(lanes[2][0], lanes[3][0]),
        );
        let value = high.0 + (high.1 + total(1));
        Estimate {
            value,
            bound: f64::EPSILON * (44.0 * total(2) + 2.0 * value.abs()),
        }
    };
    let residual = |t: f64| {
        let mut lanes = [(0.0, 0.0); 4];
        let mut k = 0;
        for &x in &y {
            let d = two_sum(x, -t);
            let u_hi = d.0 * inverse;
            if u_hi.abs() >= 1.0 {
                continue;
            }
            let (p, pe) = two_product(u_hi, s.0);
            let u_lo = (((d.0 - p) - pe) + d.1 - u_hi * s.1) * inverse;
            let u = (u_hi, u_lo);
            let uu = multiply(u, u);
            let v = add((1.0, 0.0), (-uu.0, -uu.1));
            let term = multiply(multiply(v, v), d);
            let lane: &mut Double = &mut lanes[k % 4];
            let (sum, error) = two_sum(lane.0, term.0);
            *lane = (sum, lane.1 + error + term.1);
            k += 1;
        }
        let total = add(add(lanes[0], lanes[1]), add(lanes[2], lanes[3]));
        total.0 + total.1
    };
    let mut t = median.0;
    let mut last = f64::INFINITY;
    for _ in 0..1000 {
        let (r, weight, _) = sums(t);
        if weight == 0.0 {
            break;
        }
        let step = r / weight;
        if step.abs() >= last {
            break;
        }
        last = step.abs();
        t += step;
        if last <= s.0 / 1024.0 {
            break;
        }
    }
    let mut here = sums(t);
    for _ in 0..32 {
        if here.2 <= 0.0 || here.0 == 0.0 {
            break;
        }
        let next = t + here.0 / here.2;
        if (next - t).abs() <= 2.0 * gap(t) {
            break;
        }
        let there = sums(next);
        if there.0.abs() >= here.0.abs() {
            break;
        }
        (t, here) = (next, there);
    }
    let mut at_t = cheap(t);
    if at_t.value.abs() <= at_t.bound {
        at_t = Estimate {
            value: residual(t),
            bound: 0.0,
        };
        for _ in 0..32 {
            if here.2 <= 0.0 || at_t.value == 0.0 {
                break;
            }
            let next = t + at_t.value / here.2;
            if (next - t).abs() <= 2.0 * gap(t) {
                break;
            }
            let at_next = residual(next);
            if at_next.abs() >= at_t.value.abs() {
                break;
            }
            (t, at_t) = (
                next,
                Estimate {
                    value: at_next,
                    bound: 0.0,
                },
            );
        }
    }
    let fine = times_two_to(f64::from_bits(1), (-e).max(0));
    times_two_to(nearest_root(t, at_t, fine, &cheap, &residual), e)
}

#[test]
fn the_sorted_solver_is_the_plain_one() {
    let mut random = Random(0x5eed_0304);
    let text = include_str!("../../tests/reference/biweight_windows.csv");
    let mut windows: Vec<(Vec<f64>, f64)> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (
                f[9].split(';').map(|v| v.parse().unwrap()).collect(),
                f[2].parse().unwrap(),
            )
        })
        .collect();
    for _ in 0..3000 {
        let n = 1 + random.below(80) as usize;
        let kind = random.below(5);
        let values = (0..n)
            .map(|_| match kind {
                0 => 1.0 + 1e-3 * (random.unit() - 0.5),
                1 => random.below(5) as f64,
                2 => (random.unit() - 0.5) * 10f64.powi(random.below(8) as i32 - 4),
                3 => {
                    f64::from_bits(1 + random.below(1 << 20))
                        * if random.below(2) == 0 { 1.0 } else { -1.0 }
                }
                // A tight cluster (subnormal or near it) beside distant
                // values: c MAD far below the largest magnitude.
                _ => {
                    if random.below(3) == 0 {
                        10f64.powi(random.below(600) as i32 - 300)
                    } else {
                        let bits = 1 << (1 + random.below(60));
                        f64::from_bits(random.below(bits))
                    }
                }
            })
            .collect();
        let c = match random.below(4) {
            0 => 1e-30,
            1 => 1e30,
            _ => 1.0 + 8.0 * random.unit(),
        };
        windows.push((values, c));
    }
    for (values, c) in &windows {
        let (found, expected) = (location(values, *c), plain(values, *c));
        assert_eq!(found.to_bits(), expected.to_bits(), "{values:?}, c {c}");
    }
}
