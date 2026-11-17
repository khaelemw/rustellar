// The trend over windows: numpy's median and the exact mean (Fractions) of
// every window, the windows made with exact times; and the sliding sum
// against a fresh one per window.

use super::Random;
use crate::{
    exact_sum::ExactSum,
    location::Location,
    trend::trend,
    window::{Edges, WindowOptions, windows},
};

// ==========================
// Helper Functions
// ==========================

fn options(window: f64, break_tolerance: f64, edge_cutoff: f64, edges: Edges) -> WindowOptions {
    WindowOptions {
        window,
        break_tolerance,
        edge_cutoff,
        edges,
    }
}

// The same double, or both zero (numpy may keep a zero's sign).
fn same(found: Option<f64>, expected: Option<f64>) -> bool {
    match (found, expected) {
        (Some(a), Some(b)) => a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0),
        (a, b) => a.is_none() && b.is_none(),
    }
}

// ==========================
// Tests
// ==========================

#[test]
fn numpys_median_and_the_exact_mean() {
    let rows = |text: &'static str| {
        text.lines()
            .filter(|l| !l.starts_with('#'))
            .skip(1)
            .map(|l| l.split(',').collect::<Vec<&str>>())
    };
    let cases: Vec<Vec<&str>> =
        rows(include_str!("../../tests/reference/location_cases.csv")).collect();
    let mut points = rows(include_str!("../../tests/reference/location_points.csv")).peekable();
    let (mut checked, mut none) = (0, 0);
    let mut kinds = std::collections::BTreeSet::new();
    for case in &cases {
        let id = case[0];
        kinds.insert(case[1]);
        let window: f64 = case[2].parse().unwrap();
        let tolerance: f64 = case[3].parse().unwrap();
        let edge_cutoff: f64 = case[4].parse().unwrap();
        let edges = match case[5] {
            "symmetric" => Edges::Symmetric,
            _ => Edges::HalfOpen,
        };
        let mut rows = Vec::new();
        while points.peek().is_some_and(|p| p[0] == id) {
            rows.push(points.next().unwrap());
        }
        assert_eq!(rows.len(), case[6].parse::<usize>().unwrap());
        let times: Vec<f64> = rows.iter().map(|r| r[2].parse().unwrap()).collect();
        let flux: Vec<f64> = rows.iter().map(|r| r[3].parse().unwrap()).collect();
        let masked: Vec<bool> = rows.iter().map(|r| r[4] == "1").collect();
        let break_tolerance = if tolerance == 0.0 {
            f64::INFINITY
        } else {
            tolerance
        };
        let o = options(window, break_tolerance, edge_cutoff, edges);
        let medians = trend(&times, &flux, Some(&masked), &o, Location::Median);
        let means = trend(&times, &flux, Some(&masked), &o, Location::Mean);
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

#[test]
fn the_sliding_sum_is_a_fresh_sum() {
    let mut random = Random(0x5eed_0201);
    let mut longest = 0;
    for _ in 0..300 {
        let n = 1 + random.below(600) as usize;
        let mut t = 0.0;
        let times: Vec<f64> = (0..n)
            .map(|_| {
                // Small steps and the odd gap that splits a segment.
                t += if random.below(150) == 0 {
                    2.0
                } else {
                    0.05 * random.unit()
                };
                t
            })
            .collect();
        let flux: Vec<f64> = (0..n)
            .map(|_| (random.unit() - 0.5) * 10f64.powi(random.below(40) as i32 - 20))
            .collect();
        let masked: Vec<bool> = (0..n).map(|_| random.below(4) == 0).collect();
        let window = 0.05 + 6.0 * random.unit();
        let edges = if random.below(2) == 0 {
            Edges::Symmetric
        } else {
            Edges::HalfOpen
        };
        let o = options(window, 0.5, 0.2 * random.unit(), edges);
        let means = trend(&times, &flux, Some(&masked), &o, Location::Mean);
        let w = windows(&times, Some(&masked), &o);
        for (i, window) in w.windows.iter().enumerate() {
            let expected = window.clone().filter(|r| !r.is_empty()).map(|r| {
                let mut sum = ExactSum::default();
                for &k in &w.kept[r.clone()] {
                    sum.add(flux[k]);
                }
                sum.mean(r.len() as u64)
            });
            assert_eq!(means[i], expected, "point {i}");
            longest = longest.max(window.as_ref().map_or(0, |r| r.len()));
        }
    }
    // Long windows slid as well as short.
    assert!(longest > 100, "{longest}");
}

#[test]
fn no_window_no_trend() {
    let times = [0.0, 0.1, 0.2, 0.3, 0.4];
    let flux = [1.0, 2.0, 3.0, 4.0, 5.0];
    let o = options(0.25, f64::INFINITY, 0.0, Edges::Symmetric);
    for location in [Location::Median, Location::Mean] {
        // Everything masked: windows with no point.
        let all = [true; 5];
        assert_eq!(trend(&times, &flux, Some(&all), &o, location), [None; 5]);
        // The edge cutoff: no window at the ends.
        let cut = options(0.25, f64::INFINITY, 0.1, Edges::Symmetric);
        let found = trend(&times, &flux, None, &cut, location);
        assert_eq!(found, [None, Some(2.0), Some(3.0), Some(4.0), None]);
        // Every other point masked: the windows hold one or two points.
        let some = [false, true, false, true, false];
        let found = trend(&times, &flux, Some(&some), &o, location);
        assert_eq!(found, [1.0, 2.0, 3.0, 4.0, 5.0].map(Some));
    }
    assert!(trend(&[], &[], None, &o, Location::Mean).is_empty());
}
