//! The reference windows through the public interface: each window's fluxes
//! as a light curve short enough that every point's window holds them all,
//! so the trend at a point is the method's value over the window. Against
//! roots at 50 digits and exact values (with wotan 1.10's error on the same
//! windows), numpy's medians and exact means of random light curves, and
//! wotan's own windows.

use std::collections::{BTreeMap, BTreeSet};

use rustellar_core::{LightCurve, TimeDelta};
use rustellar_detrend::{BreakTolerance, Detrend, Edges, Method};

const BIWEIGHT: &str = include_str!("reference/biweight_windows.csv");
const ESTIMATORS: &str = include_str!("reference/estimator_windows.csv");
const LOCATION_CASES: &str = include_str!("reference/location_cases.csv");
const LOCATION_POINTS: &str = include_str!("reference/location_points.csv");
const WOTAN_CASES: &str = include_str!("reference/wotan_window_cases.csv");
const WOTAN_POINTS: &str = include_str!("reference/wotan_window_points.csv");

// ==========================
// Helper Functions
// ==========================

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn rows(text: &'static str) -> impl Iterator<Item = Vec<&'static str>> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| l.split(',').collect())
}

/// `method` over `values`: the trend at the first point of a light curve of
/// those fluxes, 2^-10 days apart, under a window that holds every point.
fn over(method: Method, values: &[f64]) -> f64 {
    let n = values.len();
    let time: Vec<f64> = (0..n).map(|i| i as f64 / 1024.0).collect();
    let lc = LightCurve::from_btjd(time, values.to_vec(), vec![1.0; n]).unwrap();
    let detrend = Detrend::default()
        .with_method(method)
        .with_window(days(4.0 * (n as f64 / 1024.0) + 1.0))
        .with_break_tolerance(BreakTolerance::Never);
    detrend.trend(&lc, None).unwrap()[0].expect("a trend")
}

// The gap from `x` to the next double away from zero (the subnormals' for
// those below the normal range).
fn ulp(x: f64) -> f64 {
    let x = x.abs();
    if x < f64::MIN_POSITIVE {
        f64::from_bits(1)
    } else {
        f64::from_bits(x.to_bits() + 1) - x
    }
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

// The same double, or both zero (numpy may keep a zero's sign).
fn same(found: Option<f64>, expected: Option<f64>) -> bool {
    match (found, expected) {
        (Some(a), Some(b)) => a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0),
        (a, b) => a.is_none() && b.is_none(),
    }
}

fn break_tolerance(tolerance: f64) -> BreakTolerance {
    if tolerance == 0.0 {
        BreakTolerance::Never
    } else {
        BreakTolerance::Fixed(days(tolerance))
    }
}

// ==========================
// The biweight at 50 digits
// ==========================

#[test]
fn the_biweight_root_at_fifty_digits() {
    let (mut windows_seen, mut nearest, mut certain, mut worst_ulps) = (0, 0, 0, 0.0f64);
    let (mut wotan_worse, mut wotan_worst_ulps, mut ours_worse) = (0, 0.0f64, 0);
    let mut kinds = BTreeSet::new();
    for f in rows(BIWEIGHT) {
        let kind = f[1];
        kinds.insert(kind);
        let number = |k: usize| f[k].parse::<f64>().unwrap();
        let (c, hi, lo, sensitivity, wotan) =
            (number(2), number(4), number(5), number(7), number(8));
        // The root, c MAD and sensitivity are given times 2^-e.
        let e: i32 = f[3].parse().unwrap();
        let values: Vec<f64> = f[9].split(';').map(|v| v.parse().unwrap()).collect();
        let ours = over(Method::Biweight { c }, &values);
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

// ==========================
// The other estimators
// ==========================

fn method(name: &str, c: f64) -> Method {
    match name {
        "welsch" => Method::Welsch { c },
        "andrews" => Method::AndrewsSineWave { c },
        "huber" => Method::Huber { c },
        "hodges" => Method::HodgesLehmann,
        "trimmed" => Method::TrimmedMean { proportion: c },
        "winsorized" => Method::WinsorizedMean { proportion: c },
        _ => unreachable!("{name}"),
    }
}

#[test]
fn the_other_estimators_reference_windows() {
    let eps = f64::EPSILON;
    let mut stats = BTreeMap::<String, [usize; 4]>::new();
    let mut seen = 0;
    for f in rows(ESTIMATORS) {
        let (kind, name) = (f[1], f[2]);
        let number = |k: usize| f[k].parse::<f64>().unwrap();
        let (c, hi, lo, sensitivity, wotan) =
            (number(3), number(5), number(6), number(8), number(9));
        let e: i32 = f[4].parse().unwrap();
        let values: Vec<f64> = f[10].split(';').map(|v| v.parse().unwrap()).collect();
        let found = over(method(name, c), &values);
        seen += 1;
        let entry = stats.entry(name.to_string()).or_default();
        entry[0] += 1;
        if matches!(name, "welsch" | "andrews" | "huber") {
            // As the biweight's: the nearest double to the root wherever the
            // root is further from a midpoint than its rounding can move it.
            let scaled = times_two_to(found, -e);
            let error = ((scaled - hi) - lo).abs();
            let doubt = ROUNDINGS * eps * eps * sensitivity;
            // The doubles' spacing, or the subnormals' once scaled back.
            let unit = ulp(hi).max(times_two_to(f64::from_bits(1), (-e).max(0)));
            assert!(
                error <= 0.5 * unit + doubt,
                "{kind} {name} c {c}: {scaled:e} for {hi:e} + {lo:e} ({error:e})"
            );
            // An exact midpoint (the mean of two points, say) is a tie, and
            // ties go to the even double of the grid.
            let tie = doubt < 1e-6 * unit && (error - 0.5 * unit).abs() <= doubt;
            if tie {
                entry[1] += 1;
                assert_eq!(
                    (scaled / unit) % 2.0,
                    0.0,
                    "{kind} {name}: the tie not to even"
                );
            }
            let certain = (error - 0.5 * unit).abs() > doubt;
            if certain {
                entry[1] += 1;
                assert!(error <= 0.5 * unit, "{kind} {name}: not the nearest");
            }
            if wotan.is_finite() {
                let theirs = (times_two_to(wotan, -e) - hi - lo).abs();
                entry[2] += usize::from(theirs > error);
                entry[3] += usize::from(theirs < error);
            }
        } else {
            // Exact: the correctly rounded value, to the bit.
            assert_eq!(
                found.to_bits(),
                hi.to_bits(),
                "{kind} {name} {c}: {found:e} for {hi:e}"
            );
            entry[1] += 1;
            if wotan.is_finite() {
                entry[2] += usize::from(wotan != hi);
            }
        }
    }
    assert!(seen > 1500, "{seen}");
    for (name, [n, certain, wotan_worse, wotan_better]) in &stats {
        eprintln!(
            "{name}: {n} windows, {certain} certain, wotan worse in {wotan_worse}, better in {wotan_better}"
        );
        assert!(
            *certain * 100 >= *n * 97,
            "{name}: {certain} of {n} certain"
        );
        // Andrews is wotan's estimator too: wotan never nearer.
        if name == "andrews" || name == "hodges" || name == "trimmed" {
            assert_eq!(*wotan_better, 0, "{name}");
        }
    }
    assert_eq!(stats.len(), 6);
}

// ==========================
// numpy's median and the exact mean
// ==========================

#[test]
fn numpys_median_and_the_exact_mean() {
    let cases: Vec<Vec<&str>> = rows(LOCATION_CASES).collect();
    let mut points = rows(LOCATION_POINTS).peekable();
    let (mut checked, mut none) = (0, 0);
    let mut kinds = BTreeSet::new();
    for case in &cases {
        let id = case[0];
        kinds.insert(case[1]);
        let number = |k: usize| case[k].parse::<f64>().unwrap();
        let edges = match case[5] {
            "symmetric" => Edges::Symmetric,
            _ => Edges::HalfOpen,
        };
        let mut rows = Vec::new();
        while points.peek().is_some_and(|p| p[0] == id) {
            rows.push(points.next().unwrap());
        }
        assert_eq!(rows.len(), case[6].parse::<usize>().unwrap());
        let time: Vec<f64> = rows.iter().map(|r| r[2].parse().unwrap()).collect();
        let flux: Vec<f64> = rows.iter().map(|r| r[3].parse().unwrap()).collect();
        let masked: Vec<bool> = rows.iter().map(|r| r[4] == "1").collect();
        let lc = LightCurve::from_btjd(time, flux, vec![1.0; rows.len()]).unwrap();
        let detrend = Detrend::default()
            .with_window(days(number(2)))
            .with_break_tolerance(break_tolerance(number(3)))
            .with_edge_cutoff(days(number(4)))
            .with_edges(edges);
        let trend = |method: Method| {
            detrend
                .with_method(method)
                .trend(&lc, Some(&masked))
                .unwrap()
        };
        let (medians, means) = (trend(Method::Median), trend(Method::Mean));
        let value = |s: &str| (s != "nan").then(|| s.parse::<f64>().unwrap());
        for (i, row) in rows.iter().enumerate() {
            let (median, mean) = (value(row[5]), value(row[6]));
            assert!(
                same(medians[i], median),
                "case {id}, point {i}: median {:?} against {median:?}",
                medians[i]
            );
            assert!(
                same(means[i], mean),
                "case {id}, point {i}: mean {:?} against {mean:?}",
                means[i]
            );
            checked += 1;
            none += usize::from(median.is_none());
        }
    }
    assert_eq!(cases.len(), 30);
    assert_eq!(kinds.len(), 6);
    assert_eq!(checked, 2_850);
    // Edges cut and empty windows met, but most points have a trend.
    assert!(none > 100 && none < checked / 2, "{none}");
}

// ==========================
// wotan's windows
// ==========================

/// The exact mean of `f(k)` over the unmasked `k` in `points`, rounded once
/// (the sums are integers far below 2^53).
fn mean_of(points: &[usize], f: impl Fn(usize) -> f64) -> f64 {
    points.iter().map(|&k| f(k)).sum::<f64>() / points.len() as f64
}

// wotan 1.10's windows on random light curves (its own segmenting, masking
// and slider), as the first and last points of each: ours are the same
// half-open windows, but that wotan's never hold a segment's last point.
// Seen through the trend: with fluxes k + 1 and then (k + 1)^2 at point k,
// our mean is the mean over wotan's window, or over it and the next
// unmasked point (the segment's last).
#[test]
fn wotan_windows_but_the_last_point() {
    let cases: Vec<Vec<&str>> = rows(WOTAN_CASES).collect();
    let mut points = rows(WOTAN_POINTS).peekable();
    let (mut checked, mut last_point_differs) = (0, 0);
    for case in &cases {
        let id = case[0];
        let number = |k: usize| case[k].parse::<f64>().unwrap();
        let mut rows = Vec::new();
        while points.peek().is_some_and(|p| p[0] == id) {
            rows.push(points.next().unwrap());
        }
        let n = rows.len();
        assert_eq!(n, case[5].parse::<usize>().unwrap());
        let time: Vec<f64> = rows.iter().map(|r| r[2].parse().unwrap()).collect();
        let masked: Vec<bool> = rows.iter().map(|r| r[3] == "1").collect();
        let detrend = Detrend::default()
            .with_method(Method::Mean)
            .with_window(days(number(2)))
            .with_break_tolerance(break_tolerance(number(3)))
            .with_edge_cutoff(days(number(4)))
            .with_edges(Edges::HalfOpen);
        let linear = |k: usize| (k + 1) as f64;
        let square = |k: usize| linear(k) * linear(k);
        let trend = |f: &dyn Fn(usize) -> f64| {
            let flux: Vec<f64> = (0..n).map(f).collect();
            let lc = LightCurve::from_btjd(time.clone(), flux, vec![1.0; n]).unwrap();
            detrend.trend(&lc, Some(&masked)).unwrap()
        };
        let (by_linear, by_square) = (trend(&linear), trend(&square));
        for (i, row) in rows.iter().enumerate() {
            let first: i64 = row[4].parse().unwrap();
            let last: i64 = row[5].parse().unwrap();
            let count: usize = row[6].parse().unwrap();
            // wotan's window: its unmasked points from first to last.
            let theirs: Vec<usize> = if first < 0 {
                Vec::new()
            } else {
                (first as usize..=last as usize)
                    .filter(|&k| !masked[k])
                    .collect()
            };
            assert_eq!(theirs.len(), count, "case {id}, point {i}");
            let after = theirs.last().map_or(i, |&k| k + 1);
            let next = (after..n).find(|&k| !masked[k]);
            let with_next: Option<Vec<usize>> =
                next.map(|k| theirs.iter().copied().chain([k]).collect());
            let means = |set: &[usize]| {
                (!set.is_empty()).then(|| (mean_of(set, linear), mean_of(set, square)))
            };
            let found = by_linear[i].map(|a| (a, by_square[i].unwrap()));
            let same_bits = |e: Option<(f64, f64)>| {
                e.map(|(a, b)| (a.to_bits(), b.to_bits()))
                    == found.map(|(a, b)| (a.to_bits(), b.to_bits()))
            };
            if same_bits(means(&theirs)) {
            } else if with_next.as_deref().is_some_and(|w| same_bits(means(w))) {
                last_point_differs += 1;
            } else {
                panic!("case {id}, point {i}: {found:?} for wotan's {first}..={last}");
            }
            checked += 1;
        }
    }
    // Every point compared; the last-point difference seen often.
    assert_eq!(checked, 16_586);
    assert!(last_point_differs > 100, "{last_point_differs}");
}

// ==========================
// Exactly at the cuts
// ==========================

// Windows symmetric about 0 with MAD m and points at the double just below
// and just above c m (where c m is not a double): |u| a hair either side of
// Huber's cut at 1, decided exactly. By symmetry every method's location is
// 0. Huber's terms and the order estimators are exact, so theirs is 0 to
// the bit; the biweight's, Welsch's and Andrews' residuals carry the
// double-double's rounding (eps^2 times the window's spread, ~1e-30 here),
// within which a root at 0 cannot be told from its neighbours.
#[test]
fn symmetric_windows_with_points_at_the_cut() {
    let mut checked = 0;
    for m in [3.0, 5.0, 7.0, 11.0, 13.0] {
        for c in [1.1, 1.28, 1.3, 1.7, 2.11, 3.3] {
            let near: f64 = c * m;
            if c.mul_add(m, -near) == 0.0 {
                continue;
            }
            let (below, above) = if c.mul_add(m, -near) > 0.0 {
                (near, f64::from_bits(near.to_bits() + 1))
            } else {
                (f64::from_bits(near.to_bits() - 1), near)
            };
            let values = vec![
                -above, -below, -below, -m, -m, -1.0, 0.0, 1.0, m, m, below, below, above,
            ];
            for method in [
                Method::Huber { c },
                Method::Biweight { c },
                Method::Welsch { c },
                Method::AndrewsSineWave { c },
                Method::Median,
                Method::Mean,
                Method::HodgesLehmann,
                Method::TrimmedMean { proportion: 0.1 },
                Method::WinsorizedMean { proportion: 0.1 },
            ] {
                let found = over(method, &values);
                let exact = !matches!(
                    method,
                    Method::Biweight { .. }
                        | Method::Welsch { .. }
                        | Method::AndrewsSineWave { .. }
                );
                if exact {
                    assert_eq!(found, 0.0, "{method:?}, m {m}");
                } else {
                    assert!(found.abs() <= 1e-28, "{method:?}, m {m}: {found:e}");
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 150, "{checked}");
}
