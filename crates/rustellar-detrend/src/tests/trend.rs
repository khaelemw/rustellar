// The trend over windows: numpy's median and the exact mean (Fractions) of
// every window, the windows made with exact times; and the sliding sum
// against a fresh one per window.

use super::Random;
use crate::{
    biweight::{Scratch, biweight},
    exact_sum::ExactSum,
    location::{Location, median},
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
        let medians = trend(&times, &flux, Some(&masked), &o, Location::Median, 4);
        let means = trend(&times, &flux, Some(&masked), &o, Location::Mean, 2);
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
        let means = trend(&times, &flux, Some(&masked), &o, Location::Mean, 2);
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
        assert_eq!(trend(&times, &flux, Some(&all), &o, location, 1), [None; 5]);
        // The edge cutoff: no window at the ends.
        let cut = options(0.25, f64::INFINITY, 0.1, Edges::Symmetric);
        let found = trend(&times, &flux, None, &cut, location, 2);
        assert_eq!(found, [None, Some(2.0), Some(3.0), Some(4.0), None]);
        // Every other point masked: the windows hold one or two points.
        let some = [false, true, false, true, false];
        let found = trend(&times, &flux, Some(&some), &o, location, 1);
        assert_eq!(found, [1.0, 2.0, 3.0, 4.0, 5.0].map(Some));
    }
    assert!(trend(&[], &[], None, &o, Location::Mean, 4).is_empty());
}

// A light curve with gaps, masked points, outliers and a slow trend, from
// `seed`.
fn curve(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut random = Random(seed);
    let mut t = 1400.0;
    let mut times = Vec::with_capacity(n);
    let mut flux = Vec::with_capacity(n);
    for _ in 0..n {
        t += if random.below(300) == 0 {
            1.0
        } else {
            0.002 + 0.002 * random.unit()
        };
        times.push(t);
        let outlier = if random.below(50) == 0 { 0.02 } else { 0.0 };
        // A slow wave in plain arithmetic (no `sin`, whose last bits differ
        // between platforms' maths libraries).
        let phase = (t * 0.7).fract();
        flux.push(1.0 + 12e-3 * phase * (1.0 - phase) + 1e-3 * (random.unit() - 0.5) + outlier);
    }
    let masked = (0..n).map(|_| random.below(20) == 0).collect();
    (times, flux, masked)
}

const LOCATIONS: [Location; 4] = [
    Location::Mean,
    Location::Median,
    Location::Biweight(5.0),
    Location::Biweight(3.0),
];

#[test]
fn any_number_of_threads() {
    let (times, flux, masked) = curve(0x5eed_0401, 3000);
    for edges in [Edges::Symmetric, Edges::HalfOpen] {
        let o = options(0.5, 0.25, 0.1, edges);
        for location in LOCATIONS {
            let one = trend(&times, &flux, Some(&masked), &o, location, 1);
            for threads in [2, 3, 7, 16, 5000] {
                let many = trend(&times, &flux, Some(&masked), &o, location, threads);
                let same = one
                    .iter()
                    .zip(&many)
                    .all(|(a, b)| a.map(f64::to_bits) == b.map(f64::to_bits));
                assert!(same, "{location:?}, {threads} threads");
            }
        }
    }
}

#[test]
fn the_sliding_window_is_each_window_alone() {
    for seed in 0..8 {
        let (times, flux, masked) = curve(0x5eed_0410 + seed, 800);
        let edges = if seed % 2 == 0 {
            Edges::Symmetric
        } else {
            Edges::HalfOpen
        };
        let o = options(0.1 + 0.1 * seed as f64, 0.3, 0.02, edges);
        let w = windows(&times, Some(&masked), &o);
        for location in LOCATIONS {
            let found = trend(&times, &flux, Some(&masked), &o, location, 1);
            for (i, window) in w.windows.iter().enumerate() {
                let expected = window.clone().filter(|r| !r.is_empty()).map(|r| {
                    let mut values: Vec<f64> = w.kept[r].iter().map(|&k| flux[k]).collect();
                    match location {
                        Location::Mean => {
                            let mut sum = ExactSum::default();
                            values.iter().for_each(|&x| sum.add(x));
                            sum.mean(values.len() as u64)
                        }
                        Location::Median => median(&mut values),
                        Location::Biweight(c) => biweight(&mut values, &mut Scratch::default(), c),
                    }
                });
                assert_eq!(
                    found[i].map(f64::to_bits),
                    expected.map(f64::to_bits),
                    "{location:?}, point {i}"
                );
            }
        }
    }
}

// Every output bit of four light curves and every location, hashed (FNV-1a):
// a change anywhere, on any machine, changes it.
#[test]
fn fingerprint() {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |x: u64| {
        for byte in x.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    for seed in 0..4 {
        let (times, flux, masked) = curve(0x5eed_0420 + seed, 2000);
        let o = options(0.3, 0.15, 0.0, Edges::Symmetric);
        for location in LOCATIONS {
            for value in trend(&times, &flux, Some(&masked), &o, location, 4) {
                eat(value.map_or(u64::MAX, f64::to_bits));
            }
        }
    }
    assert_eq!(hash, 0x6c972f18869fc1b1, "fingerprint {hash:#018x}");
}
