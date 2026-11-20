//! The trend as a cubic spline fitted to each segment: knots evenly spaced
//! at most a window apart, the coefficients by least squares.
//!
//! The spline is a sum of uniform cubic B-splines. The least-squares system
//! is banded (each point touches four coefficients), so it is reduced to a
//! triangle by Givens rotations one point at a time, as FITPACK does: about
//! 16 operations a point and no normal equations, so the precision is that
//! of the data, not its square.

use std::cmp::Ordering;

use crate::{
    location::midpoint,
    window::{WindowOptions, exceeds, segments},
};

/// Steps of iterative refinement after the first solve.
const REFINEMENTS: usize = 2;

/// How a segment's spline is fitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Fit {
    /// Least squares over every unmasked point.
    Plain,
    /// Least squares, refitted without the points more than this many
    /// robust standard deviations from the spline, until they repeat.
    Clipped(f64),
}

/// The most refits a clipped spline makes.
pub(crate) const MAX_CLIPS: usize = 50;

/// The standard deviation of normal data over its median absolute
/// deviation: `1 / Phi^-1(3/4)`.
pub(crate) const MAD_TO_SD: f64 = 1.482_602_218_505_602;

/// Knots `step` apart from `start`, `intervals` of them: `intervals + 3`
/// coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Knots {
    pub(crate) start: f64,
    pub(crate) step: f64,
    pub(crate) intervals: usize,
}

impl Knots {
    /// The knots of a segment from `first` to `last` (`first <= last`):
    /// `ceil((last - first) / window)` intervals, at least one and at most
    /// `points`; for a segment of one time, one interval a window long.
    pub(crate) fn new(first: f64, last: f64, window: f64, points: usize) -> Self {
        let span = last - first;
        if span <= 0.0 {
            return Self {
                start: first,
                step: window,
                intervals: 1,
            };
        }
        let wanted = (span / window).ceil();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "positive, and capped by the point count"
        )]
        let intervals = if wanted >= points as f64 {
            points.max(1)
        } else {
            (wanted as usize).max(1)
        };
        #[expect(clippy::cast_precision_loss, reason = "a count of intervals")]
        let step = span / intervals as f64;
        Self {
            start: first,
            step,
            intervals,
        }
    }

    /// The number of coefficients.
    pub(crate) fn coefficients(&self) -> usize {
        self.intervals + 3
    }

    /// The first of the four coefficients that reach `x`, and the four
    /// B-splines' values there (they sum to 1).
    pub(crate) fn basis(&self, x: f64) -> (usize, [f64; 4]) {
        // `x - start` is exact for times of one magnitude (Sterbenz), and
        // `j step` is split exactly into its double and the rest, so `u` is
        // found to about an ulp however many intervals come before.
        let offset = x - self.start;
        let at = (offset / self.step).max(0.0);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "not negative, and clamped below"
        )]
        let j = (at.floor() as usize).min(self.intervals - 1);
        #[expect(clippy::cast_precision_loss, reason = "an interval number")]
        let jf = j as f64;
        let knot = jf * self.step;
        let rest = jf.mul_add(self.step, -knot);
        let u = (((offset - knot) - rest) / self.step).clamp(0.0, 1.0);
        let v = 1.0 - u;
        let (u2, u3) = (u * u, u * u * u);
        let b = [
            v * v * v / 6.0,
            (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
            (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
            u3 / 6.0,
        ];
        (j, b)
    }

    /// The spline with `coefficients` at `x`.
    pub(crate) fn value(&self, coefficients: &[f64], x: f64) -> f64 {
        let (j, b) = self.basis(x);
        b.iter()
            .zip(&coefficients[j..j + 4])
            .map(|(b, c)| b * c)
            .sum()
    }
}

/// The upper triangle of a banded least-squares system, rows added one at
/// a time: row `i` holds the triangle's entries from column `i` to `i + 3`.
#[derive(Debug, Clone)]
pub(crate) struct Triangle {
    rows: Vec<[f64; 4]>,
    rhs: Vec<f64>,
    /// The first column of the last row added: rows come in order of it.
    first: usize,
    /// Every rotation made, in order: its column, cosine and sine.
    rotations: Vec<(usize, f64, f64)>,
    /// For each row added, the end of its rotations in `rotations`.
    ends: Vec<usize>,
}

impl Triangle {
    pub(crate) fn new(columns: usize) -> Self {
        Self {
            rows: vec![[0.0; 4]; columns],
            rhs: vec![0.0; columns],
            first: 0,
            rotations: Vec::new(),
            ends: Vec::new(),
        }
    }

    /// The row with `h[k]` at column `first + k`, and right-hand side `y`,
    /// rotated in. Rows must come in order of `first`, so that no row of the
    /// triangle reaches past the new row's band.
    pub(crate) fn add(&mut self, first: usize, mut h: [f64; 4], mut y: f64) {
        debug_assert!(first >= self.first, "rows out of order");
        self.first = first;
        for k in 0..4 {
            let pivot = h[k];
            if pivot == 0.0 {
                continue;
            }
            let col = first + k;
            let row = &mut self.rows[col];
            let (c, s, d) = givens(row[0], pivot);
            row[0] = d;
            for l in k + 1..4 {
                let (a, b) = (h[l], row[l - k]);
                row[l - k] = c * b + s * a;
                h[l] = c * a - s * b;
            }
            let z = self.rhs[col];
            self.rhs[col] = c * z + s * y;
            y = c * y - s * z;
            self.rotations.push((col, c, s));
        }
        self.ends.push(self.rotations.len());
    }

    /// The right-hand sides `ys` (one for each row added, in order) taken
    /// through the same rotations: the triangle's right-hand side for them.
    pub(crate) fn rotate(&self, ys: impl Iterator<Item = f64>) -> Vec<f64> {
        let mut rhs = vec![0.0; self.rows.len()];
        let mut from = 0;
        for (&end, mut y) in self.ends.iter().zip(ys) {
            for &(col, c, s) in &self.rotations[from..end] {
                let z = rhs[col];
                rhs[col] = c * z + s * y;
                y = c * y - s * z;
            }
            from = end;
        }
        rhs
    }

    /// The largest diagonal entry's magnitude.
    fn largest(&self) -> f64 {
        self.rows.iter().fold(0.0, |m, r| m.max(r[0].abs()))
    }

    /// Whether a diagonal entry is too small beside the largest for its
    /// coefficient to be known from the data.
    fn deficient(&self) -> bool {
        let tiny = 1e-8 * self.largest();
        self.rows.iter().any(|r| r[0].abs() <= tiny)
    }

    /// The solution, by back substitution.
    pub(crate) fn solve(&self) -> Vec<f64> {
        self.solve_for(&self.rhs)
    }

    /// The solution for the right-hand side `rhs` (from
    /// [`rotate`](Self::rotate)).
    pub(crate) fn solve_for(&self, rhs: &[f64]) -> Vec<f64> {
        let n = self.rows.len();
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let r = &self.rows[i];
            let mut z = rhs[i];
            for l in 1..4.min(n - i) {
                z -= r[l] * x[i + l];
            }
            x[i] = z / r[0];
        }
        x
    }
}

/// The rotation taking `(a, b)` to `(d, 0)`: `c`, `s` and `d >= 0`, with
/// `d` the hypotenuse found without overflow or a platform's `hypot`
/// (scaled first where the squares would leave the normal range).
fn givens(a: f64, b: f64) -> (f64, f64, f64) {
    let q = a * a + b * b;
    let d = if (1e-290..1e290).contains(&q) {
        q.sqrt()
    } else {
        let m = a.abs().max(b.abs());
        let (x, y) = (a / m, b / m);
        m * (x * x + y * y).sqrt()
    };
    let inverse = 1.0 / d;
    (a * inverse, b * inverse, d)
}

/// The coefficients of the least-squares spline on `knots` through the
/// points `(x, y)` with weights `w` (the square roots of the least-squares
/// weights, above zero); at least one point.
///
/// Where the points leave a coefficient undetermined (a knot interval
/// without points nearby, or fewer points than coefficients), the spline
/// is also kept smooth there: differences of order three of the
/// coefficients (two for two points, one for one) are pulled toward zero
/// by rows a millionth of the largest diagonal entry, which leaves the
/// determined part all but untouched and bridges the rest with a low-order
/// polynomial.
pub(crate) fn fit(knots: &Knots, points: &[(f64, f64, f64)]) -> Vec<f64> {
    debug_assert!(!points.is_empty());
    let rows: Vec<(usize, [f64; 4])> = points.iter().map(|&(x, _, _)| knots.basis(x)).collect();
    let m = knots.coefficients();
    // The triangle of the rows (each point's, times its weight) and the
    // smoothing rows if any, merged in order of their first column: a row
    // must not meet a triangle row that reaches past its own band. Each
    // row's source in order: a point's index, or `None` for smoothing.
    let build = |smooth: Option<Smooth>| {
        let mut triangle = Triangle::new(m);
        let mut order = Vec::with_capacity(points.len() + m);
        let mut next = 0;
        for (i, (&(j, b), &(_, y, w))) in rows.iter().zip(points).enumerate() {
            if let Some(s) = smooth {
                while next <= j.min(s.last(m)) {
                    triangle.add(next, s.row(), 0.0);
                    order.push(None);
                    next += 1;
                }
            }
            triangle.add(j, b.map(|b| b * w), y * w);
            order.push(Some(i));
        }
        if let Some(s) = smooth {
            for i in next..=s.last(m) {
                triangle.add(i, s.row(), 0.0);
                order.push(None);
            }
        }
        (triangle, order)
    };
    let (mut triangle, mut order) = build(None);
    if triangle.deficient() {
        (triangle, order) = build(Some(Smooth::new(triangle.largest(), points.len())));
    }
    let mut c = triangle.solve();
    // Steps of refinement: the residuals found to about an ulp each (fused
    // multiply-adds), taken through the same rotations, and their
    // least-squares spline added on.
    for _ in 0..REFINEMENTS {
        let residual = |i: usize| {
            let (j, b) = rows[i];
            let (_, y, w) = points[i];
            let r = b
                .iter()
                .zip(&c[j..j + 4])
                .fold(y, |r, (b, c)| (-b).mul_add(*c, r));
            r * w
        };
        let rhs = triangle.rotate(order.iter().map(|o| o.map_or(0.0, residual)));
        let delta = triangle.solve_for(&rhs);
        for (c, d) in c.iter_mut().zip(delta) {
            *c += d;
        }
    }
    c
}

/// The spline fitted to `points` and refitted to those within `sigma`
/// robust standard deviations of it ([`MAD_TO_SD`] times the median
/// absolute deviation of every point's residual from their median), until
/// the points kept repeat, their scatter is zero, or [`MAX_CLIPS`] refits.
/// Every point is judged afresh each time, so one left out can come back.
pub(crate) fn clipped(knots: &Knots, points: &[(f64, f64, f64)], sigma: f64) -> Vec<f64> {
    let mut c = fit(knots, points);
    let mut inside = vec![true; points.len()];
    let mut residuals = vec![0.0; points.len()];
    let mut scratch = Vec::with_capacity(points.len());
    let mut subset = Vec::with_capacity(points.len());
    for _ in 0..MAX_CLIPS {
        for (r, &(x, y, _)) in residuals.iter_mut().zip(points) {
            *r = y - knots.value(&c, x);
        }
        let limit = sigma * MAD_TO_SD * mad(&residuals, &mut scratch);
        if limit == 0.0 {
            break;
        }
        let mut same = true;
        for (keep, r) in inside.iter_mut().zip(&residuals) {
            let now = r.abs() <= limit;
            same &= *keep == now;
            *keep = now;
        }
        if same {
            break;
        }
        subset.clear();
        subset.extend(
            points
                .iter()
                .zip(&inside)
                .filter(|(_, k)| **k)
                .map(|(p, _)| *p),
        );
        c = fit(knots, &subset);
    }
    c
}

/// The median absolute deviation of `values` (at least one, finite) from
/// their median, with `scratch` for room.
pub(crate) fn mad(values: &[f64], scratch: &mut Vec<f64>) -> f64 {
    scratch.clear();
    scratch.extend_from_slice(values);
    let centre = median(scratch);
    for v in scratch.iter_mut() {
        *v = (*v - centre).abs();
    }
    median(scratch)
}

/// The median of `values` (at least one, finite), reordering them.
fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    let (below, &mut upper, _) = values.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 1 {
        return upper;
    }
    let lower = below.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    midpoint(lower, upper)
}

/// Rows pulling the differences of the coefficients toward zero, of order
/// three (two for two points, one for one), each a millionth of the largest
/// diagonal entry.
#[derive(Debug, Clone, Copy)]
struct Smooth {
    weight: f64,
    order: usize,
}

impl Smooth {
    fn new(largest: f64, points: usize) -> Self {
        Self {
            weight: 1e-6 * largest,
            order: points.min(3),
        }
    }

    /// The row: the differences' coefficients times the weight.
    fn row(self) -> [f64; 4] {
        let d: [f64; 4] = match self.order {
            1 => [-1.0, 1.0, 0.0, 0.0],
            2 => [1.0, -2.0, 1.0, 0.0],
            _ => [-1.0, 3.0, -3.0, 1.0],
        };
        d.map(|d| d * self.weight)
    }

    /// The first column of the last row, for `m` coefficients.
    fn last(self, m: usize) -> usize {
        m - self.order - 1
    }
}

/// The trend at each point: the spline of its segment, or `None` within the
/// edge cutoff or in a segment with every point masked.
pub(crate) fn trend(
    times: &[f64],
    flux: &[f64],
    masked: Option<&[bool]>,
    options: &WindowOptions,
    how: Fit,
) -> Vec<Option<f64>> {
    let cutoff = options.edge_cutoff.clamp(0.0, options.window / 2.0);
    let mut out = vec![None; times.len()];
    let mut kept = Vec::new();
    for segment in segments(times, options.break_tolerance) {
        kept.clear();
        kept.extend(
            segment
                .clone()
                .filter(|&k| !masked.is_some_and(|m| m[k]))
                .map(|k| (times[k], flux[k], 1.0)),
        );
        if kept.is_empty() {
            continue;
        }
        let (first, last) = (times[segment.start], times[segment.end - 1]);
        let knots = Knots::new(first, last, options.window, kept.len());
        let coefficients = match how {
            Fit::Plain => fit(&knots, &kept),
            Fit::Clipped(sigma) => clipped(&knots, &kept, sigma),
        };
        for i in segment {
            let t = times[i];
            // Within the cutoff of either end: no trend.
            if exceeds(t, first, cutoff) != Ordering::Less
                && exceeds(last, t, cutoff) != Ordering::Less
            {
                out[i] = Some(knots.value(&coefficients, t));
            }
        }
    }
    out
}
