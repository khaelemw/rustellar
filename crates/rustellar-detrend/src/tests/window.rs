// The segments and windows: the exact comparison against integer
// arithmetic, hand-built edges, a brute-force definition on times where
// doubles are exact, and wotan 1.10's windows with its two differences.

use std::{cmp::Ordering, ops::Range};

use super::Random;
use crate::window::{Edges, WindowOptions, Windows, exceeds, segments, windows};

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

// The points (indices) in each point's window, `None` where there is none.
fn points_in(w: &Windows) -> Vec<Option<Vec<usize>>> {
    w.windows
        .iter()
        .map(|r| r.clone().map(|r| w.kept[r].to_vec()))
        .collect()
}

// `x` as an integer number of 2^-82, exactly: for |x| in [2^-30, 2^30].
fn scaled(x: f64) -> i128 {
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1075;
    let mantissa = i128::from((bits & ((1 << 52) - 1)) | (1 << 52));
    let shift = exponent + 82;
    assert!((0..=126 - 53).contains(&shift), "{x} out of range");
    if x < 0.0 {
        -(mantissa << shift)
    } else {
        mantissa << shift
    }
}

// The windows by their definition, point against point, in exact integer
// arithmetic (none of the code's own): every time and length a whole number
// of 2^-82 days.
fn brute_force(
    times: &[f64],
    masked: &[bool],
    window: f64,
    break_tolerance: f64,
    edge_cutoff: f64,
    edges: Edges,
) -> Vec<Option<Vec<usize>>> {
    let exact = |x: f64| if x == 0.0 { 0 } else { scaled(x) };
    let n = times.len();
    let t: Vec<i128> = times.iter().map(|&x| exact(x)).collect();
    // Halving a double is exact.
    let half = exact(window / 2.0);
    let cutoff = exact(edge_cutoff).min(half);
    let tolerance = (break_tolerance != f64::INFINITY).then(|| exact(break_tolerance));
    let mut segment = vec![0; n];
    for k in 1..n {
        let split = tolerance.is_some_and(|tol| t[k] - t[k - 1] > tol);
        segment[k] = segment[k - 1] + usize::from(split);
    }
    (0..n)
        .map(|i| {
            let mine: Vec<usize> = (0..n).filter(|&j| segment[j] == segment[i]).collect();
            let (first, last) = (t[mine[0]], t[*mine.last().unwrap()]);
            if t[i] - first < cutoff || last - t[i] < cutoff {
                return None;
            }
            Some(
                mine.into_iter()
                    .filter(|&j| !masked[j])
                    .filter(|&j| {
                        let d = t[j] - t[i];
                        -half <= d && (d < half || (d == half && edges == Edges::Symmetric))
                    })
                    .collect(),
            )
        })
        .collect()
}

// ==========================
// Tests
// ==========================

// The sign of a - b - c against integer arithmetic, on values whose
// difference rounds, including near-ties.
#[test]
fn exceeds_is_exact() {
    let mut rng = Random(17);
    let mut rounded_wrong = 0;
    for _ in 0..200_000 {
        let magnitude =
            |rng: &mut Random| 2f64.powi(rng.below(50) as i32 - 25) * (1.0 + rng.unit());
        let (x, y) = (magnitude(&mut rng), magnitude(&mut rng));
        let (a, b) = if x >= y { (x, y) } else { (y, x) };
        if a == b {
            continue;
        }
        // c near a - b (a tie, a few ulps either side) or anywhere.
        let c = match rng.below(3) {
            0 => a - b,
            1 => {
                let near = (a - b).to_bits() as i64 + rng.below(9) as i64 - 4;
                f64::from_bits(near as u64)
            }
            _ => magnitude(&mut rng),
        };
        if c.is_nan() || c < 2f64.powi(-30) {
            continue;
        }
        let exact = (scaled(a) - scaled(b) - scaled(c)).cmp(&0);
        assert_eq!(exceeds(a, b, c), exact, "{a:e} - {b:e} - {c:e}");
        rounded_wrong += usize::from(((a - b) - c).partial_cmp(&0.0) != Some(exact));
    }
    // The plain difference would have been wrong often.
    assert!(rounded_wrong > 1000, "{rounded_wrong}");
    assert_eq!(exceeds(1.0, 0.5, 0.0), Ordering::Greater);
    assert_eq!(exceeds(1.0, 1.0, 0.0), Ordering::Equal);
    assert_eq!(exceeds(1.0, 0.0, f64::INFINITY), Ordering::Less);
}

// A gap exactly at the tolerance stays in its segment; just above it splits;
// an infinite tolerance never splits.
#[test]
fn gaps_at_the_tolerance() {
    // Gaps of 1, 0.5, 1.5 and 0.25.
    let times = [0.0, 1.0, 1.5, 3.0, 3.25];
    assert_eq!(segments(&times, 1.0), vec![0..3, 3..5]);
    let below = f64::from_bits(1.0f64.to_bits() - 1);
    assert_eq!(segments(&times, below), vec![0..1, 1..3, 3..5]);
    assert_eq!(segments(&times, 1.5), vec![0..5]);
    assert_eq!(segments(&times, f64::INFINITY), vec![0..5]);
    assert_eq!(segments(&[2.0], 0.1), vec![0..1]);
    assert_eq!(segments(&[], 0.1), Vec::<Range<usize>>::new());
}

// A gap whose rounded difference equals the tolerance but whose exact one is
// above it: split. 1 - 2^-60 rounds to 1.
#[test]
fn a_gap_that_rounds_to_the_tolerance() {
    let times = [2f64.powi(-60), 1.0 + 2f64.powi(-52)];
    // Exactly 1 + 2^-52 - 2^-60, above 1.
    assert_eq!(times[1] - times[0], 1.0 + 2f64.powi(-52));
    assert_eq!(segments(&times, 1.0), vec![0..1, 1..2]);
    let times = [2f64.powi(-60), 1.0];
    // Exactly 1 - 2^-60, below 1, though it rounds to 1.
    assert_eq!(segments(&times, 1.0 - 2f64.powi(-53)), vec![0..1, 1..2]);
    assert_eq!(segments(&times, 1.0), vec![0..2]);
    // Exactly 1 + 2^-60, above 1, though it rounds to 1: split.
    let times = [-(2f64.powi(-60)), 1.0];
    assert_eq!(times[1] - times[0], 1.0);
    assert_eq!(segments(&times, 1.0), vec![0..1, 1..2]);
}

// Points exactly half a window away: both in a symmetric window; only the
// earlier in a half-open one. The last point is in its neighbours' windows.
#[test]
fn points_exactly_half_a_window_away() {
    let times: Vec<f64> = (0..10).map(f64::from).collect();
    let symmetric = windows(
        &times,
        None,
        &options(2.0, f64::INFINITY, 0.0, Edges::Symmetric),
    );
    let half_open = windows(
        &times,
        None,
        &options(2.0, f64::INFINITY, 0.0, Edges::HalfOpen),
    );
    assert_eq!(symmetric.segments, vec![0..10]);
    assert_eq!(points_in(&symmetric)[4], Some(vec![3, 4, 5]));
    assert_eq!(points_in(&half_open)[4], Some(vec![3, 4]));
    assert_eq!(points_in(&symmetric)[0], Some(vec![0, 1]));
    assert_eq!(points_in(&symmetric)[8], Some(vec![7, 8, 9]));
    assert_eq!(points_in(&symmetric)[9], Some(vec![8, 9]));
    assert_eq!(points_in(&half_open)[9], Some(vec![8, 9]));
}

// A point just within half a window, though its rounded distance is exactly
// half a window: in, even in a half-open window.
#[test]
fn a_distance_that_rounds_to_half_a_window() {
    let times = [2f64.powi(-60), 1.0];
    let w = windows(
        &times,
        None,
        &options(2.0, f64::INFINITY, 0.0, Edges::HalfOpen),
    );
    assert_eq!(points_in(&w), vec![Some(vec![0, 1]), Some(vec![0, 1])]);
}

// One point: its own window. Every point masked: windows with no points.
#[test]
fn one_point_and_all_masked() {
    let w = windows(&[3.0], None, &options(1.0, 0.5, 0.0, Edges::Symmetric));
    assert_eq!(w.segments, vec![0..1]);
    assert_eq!(points_in(&w), vec![Some(vec![0])]);

    let times = [0.0, 0.1, 0.2, 5.0];
    let w = windows(
        &times,
        Some(&[true; 4]),
        &options(1.0, 0.5, 0.0, Edges::Symmetric),
    );
    assert!(w.kept.is_empty());
    assert_eq!(w.segments, vec![0..3, 3..4]);
    assert_eq!(w.windows, vec![Some(0..0); 4]);
}

// A masked point is in no window but has its own; a segment's windows never
// reach into the next.
#[test]
fn masks_and_segments() {
    let times = [0.0, 0.1, 0.2, 0.3, 1.0, 1.1];
    let masked = [false, true, false, false, false, false];
    let w = windows(
        &times,
        Some(&masked),
        &options(1.0, 0.5, 0.0, Edges::Symmetric),
    );
    assert_eq!(w.segments, vec![0..4, 4..6]);
    assert_eq!(w.kept, vec![0, 2, 3, 4, 5]);
    let points = points_in(&w);
    assert_eq!(points[1], Some(vec![0, 2, 3]));
    assert_eq!(points[3], Some(vec![0, 2, 3]));
    assert_eq!(points[4], Some(vec![4, 5]));
}

// The edge cutoff: a point exactly the cutoff from its segment's end keeps
// its window; nearer, none; a cutoff over half the window counts as half.
#[test]
fn edge_cutoff() {
    let times: Vec<f64> = (0..9).map(|k| 0.25 * f64::from(k)).collect();
    let w = windows(
        &times,
        None,
        &options(2.0, f64::INFINITY, 0.5, Edges::Symmetric),
    );
    let have: Vec<bool> = w.windows.iter().map(Option::is_some).collect();
    assert_eq!(
        have,
        [false, false, true, true, true, true, true, false, false]
    );
    let w = windows(
        &times,
        None,
        &options(1.0, f64::INFINITY, 50.0, Edges::Symmetric),
    );
    let have: Vec<bool> = w.windows.iter().map(Option::is_some).collect();
    assert_eq!(
        have,
        [false, false, true, true, true, true, true, false, false]
    );
}

// Random times on a grid of 2^-10 days, where every difference is exact, so
// the definition can be checked with plain doubles: many points sit exactly
// on window edges, gaps exactly at the tolerance, and points exactly the
// cutoff from their segment's ends.
#[test]
fn the_windows_are_their_definition() {
    let mut rng = Random(29);
    for case in 0..300 {
        let n = 1 + rng.below(80) as usize;
        let mut times = Vec::with_capacity(n);
        let mut t = 1000.0 + rng.below(1024) as f64 / 1024.0;
        for _ in 0..n {
            times.push(t);
            // Mostly small steps, some gaps.
            let step = if rng.below(10) == 0 {
                1 + rng.below(1024)
            } else {
                1 + rng.below(64)
            };
            t += step as f64 / 1024.0;
        }
        let masked: Vec<bool> = (0..n).map(|_| rng.below(5) == 0).collect();
        let window = (1 + rng.below(256)) as f64 / 512.0;
        let break_tolerance = match rng.below(3) {
            0 => f64::INFINITY,
            1 => window / 2.0,
            _ => (1 + rng.below(512)) as f64 / 1024.0,
        };
        let edge_cutoff = match rng.below(3) {
            0 => 0.0,
            1 => (rng.below(256)) as f64 / 1024.0,
            _ => 10.0,
        };
        for edges in [Edges::Symmetric, Edges::HalfOpen] {
            let w = windows(
                &times,
                Some(&masked),
                &options(window, break_tolerance, edge_cutoff, edges),
            );
            assert_eq!(
                points_in(&w),
                brute_force(&times, &masked, window, break_tolerance, edge_cutoff, edges),
                "case {case}, {edges:?}"
            );
            let covered: usize = w.segments.iter().map(Range::len).sum();
            assert_eq!(covered, n);
        }
    }
}

// Arbitrary doubles, from either side of zero, where differences of times
// round: windows, tolerances and cutoffs set to rounded differences of the
// times themselves, so the exact difference is a hair either side of them.
// Against the definition in integer arithmetic.
#[test]
fn the_windows_are_exact_where_differences_round() {
    let mut rng = Random(31);
    let mut near_ties = 0;
    for case in 0..400 {
        let n = 2 + rng.below(60) as usize;
        let mut times = Vec::with_capacity(n);
        let mut t = 2.0 * rng.unit() - 1.0;
        while times.len() < n {
            // Leave out times too near zero for the integer check.
            if t == 0.0 || t.abs() >= 2f64.powi(-30) {
                times.push(t);
            }
            t += if rng.below(8) == 0 {
                rng.unit()
            } else {
                0.08 * rng.unit() + 1e-9
            };
        }
        let masked: Vec<bool> = (0..n).map(|_| rng.below(6) == 0).collect();
        // A rounded difference of two of the times.
        let difference = |rng: &mut Random| {
            let i = rng.below(n as u64 - 1) as usize;
            let j = i + 1 + rng.below((n - i - 1).min(20) as u64) as usize;
            times[j] - times[i]
        };
        let window = if rng.below(2) == 0 {
            2.0 * difference(&mut rng)
        } else {
            0.001 + 0.5 * rng.unit()
        };
        let break_tolerance = match rng.below(3) {
            0 => f64::INFINITY,
            1 => difference(&mut rng),
            _ => window / 2.0,
        };
        let edge_cutoff = match rng.below(3) {
            0 => 0.0,
            1 => difference(&mut rng),
            _ => 10.0,
        };
        // How often the rounded distance is a tie that the exact one is not.
        for i in 0..n {
            for j in i + 1..n {
                let d = times[j] - times[i];
                if (d == window / 2.0 || d == break_tolerance || d == edge_cutoff)
                    && scaled_or_zero(times[j]) - scaled_or_zero(times[i]) != scaled_or_zero(d)
                {
                    near_ties += 1;
                }
            }
        }
        for edges in [Edges::Symmetric, Edges::HalfOpen] {
            let w = windows(
                &times,
                Some(&masked),
                &options(window, break_tolerance, edge_cutoff, edges),
            );
            assert_eq!(
                points_in(&w),
                brute_force(&times, &masked, window, break_tolerance, edge_cutoff, edges),
                "case {case}, {edges:?}"
            );
        }
    }
    // The rounding mattered: plain doubles would have had ties that are not.
    assert!(near_ties > 50, "{near_ties}");
}

fn scaled_or_zero(x: f64) -> i128 {
    if x == 0.0 { 0 } else { scaled(x) }
}

// wotan 1.10's windows on random light curves (its own segmenting, masking
// and slider): the same as our half-open windows, but that wotan's never
// hold a segment's last point.
#[test]
fn wotan_windows_but_the_last_point() {
    let rows = |text: &'static str| {
        text.lines()
            .filter(|l| !l.starts_with('#'))
            .skip(1)
            .map(|l| l.split(',').collect::<Vec<&str>>())
    };
    let cases: Vec<Vec<&str>> =
        rows(include_str!("../../tests/reference/wotan_window_cases.csv")).collect();
    let mut points = rows(include_str!(
        "../../tests/reference/wotan_window_points.csv"
    ))
    .peekable();
    let (mut checked, mut last_point_differs) = (0, 0);
    for case in &cases {
        let id = case[0];
        let window: f64 = case[2].parse().unwrap();
        let tolerance: f64 = case[3].parse().unwrap();
        let edge_cutoff: f64 = case[4].parse().unwrap();
        let mut rows = Vec::new();
        while points.peek().is_some_and(|p| p[0] == id) {
            rows.push(points.next().unwrap());
        }
        assert_eq!(rows.len(), case[5].parse::<usize>().unwrap());
        let times: Vec<f64> = rows.iter().map(|r| r[2].parse().unwrap()).collect();
        let masked: Vec<bool> = rows.iter().map(|r| r[3] == "1").collect();
        let break_tolerance = if tolerance == 0.0 {
            f64::INFINITY
        } else {
            tolerance
        };
        let w = windows(
            &times,
            Some(&masked),
            &options(window, break_tolerance, edge_cutoff, Edges::HalfOpen),
        );
        let segment_last: Vec<usize> = w
            .segments
            .iter()
            .flat_map(|s| s.clone().map(move |_| s.end - 1))
            .collect();
        for (i, row) in rows.iter().enumerate() {
            let ours = points_in(&w)[i].clone();
            // wotan's: ours without the segment's last point; no trend where
            // that leaves none.
            let theirs = ours.as_ref().map(|p| {
                p.iter()
                    .copied()
                    .filter(|&j| j != segment_last[i])
                    .collect::<Vec<usize>>()
            });
            let expected = match &theirs {
                Some(p) if !p.is_empty() => (p[0] as i64, *p.last().unwrap() as i64, p.len()),
                _ => (-1, -1, 0),
            };
            let found = (
                row[4].parse::<i64>().unwrap(),
                row[5].parse::<i64>().unwrap(),
                row[6].parse::<usize>().unwrap(),
            );
            assert_eq!(found, expected, "case {id}, point {i}");
            checked += 1;
            last_point_differs += usize::from(ours != theirs);
        }
    }
    // Every point compared; the last-point difference seen often.
    assert_eq!(checked, 16_586);
    assert!(last_point_differs > 100, "{last_point_differs}");
}
