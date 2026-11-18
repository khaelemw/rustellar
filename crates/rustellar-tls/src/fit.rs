//! The template fitted at one position: one period, duration and
//! mid-transit time, its depth by weighted least squares.
//!
//! Out of transit the model is 1; in transit it is `1 - d s(x)`. With
//! residuals `r = 1 - y` and weights `w = 1 / dy²`, the chi-squared
//! `sum w (r - d s)²` is least at
//!
//! ```text
//! d* = sum w r s / sum w s²
//! ```
//!
//! and the chi-squared there is computed again from `d*`, term by term, not
//! as `sum w r² - (sum w r s)² / sum w s²`, which loses every digit when the
//! template fits the data well.

use rustellar_core::LightCurve;

use crate::template::Shape;

/// A sum of doubles with Neumaier's compensation: the rounding error of
/// each addition is kept and added back at the end, so the result is as if
/// summed in twice the precision, then rounded.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Sum {
    sum: f64,
    compensation: f64,
}

impl Sum {
    pub(crate) fn add(&mut self, x: f64) {
        let t = self.sum + x;
        self.compensation += if self.sum.abs() >= x.abs() {
            (self.sum - t) + x
        } else {
            (x - t) + self.sum
        };
        self.sum = t;
    }

    pub(crate) fn value(self) -> f64 {
        self.sum + self.compensation
    }
}

/// A light curve as the fit uses it: times in days, residuals `1 - y`, and
/// weights `1 / dy²`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Points {
    pub(crate) time: Vec<f64>,
    pub(crate) residual: Vec<f64>,
    pub(crate) weight: Vec<f64>,
}

impl Points {
    pub(crate) fn new(lc: &LightCurve) -> Self {
        Self {
            time: lc.time().iter().map(|t| t.as_btjd()).collect(),
            residual: lc.flux().iter().map(|y| 1.0 - y).collect(),
            weight: lc.flux_err().iter().map(|dy| 1.0 / (dy * dy)).collect(),
        }
    }

    /// The chi-squared of the flat model, `sum w r²`.
    pub(crate) fn flat_chi2(&self) -> f64 {
        let mut sum = Sum::default();
        for (&r, &w) in self.residual.iter().zip(&self.weight) {
            sum.add(w * r * r);
        }
        sum.value()
    }
}

/// Where a template sits: its period, duration and mid-transit time, in
/// days.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Position {
    pub(crate) period: f64,
    pub(crate) duration: f64,
    pub(crate) t0: f64,
}

impl Position {
    /// The template's argument at time `t`: the time from the nearest
    /// mid-transit, in `[-P/2, P/2)`, over the duration.
    pub(crate) fn x(self, t: f64) -> f64 {
        let phase = ((t - self.t0) / self.period + 0.5).rem_euclid(1.0) - 0.5;
        phase * self.period / self.duration
    }
}

/// The best fit at one position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Fit {
    /// The least-squares depth: the dimming at mid-transit. Zero when the
    /// points in transit are not on average below 1 (no transit fits).
    pub(crate) depth: f64,
    /// The chi-squared at that depth.
    pub(crate) chi2: f64,
    /// How many points the template covers (`s > 0`).
    pub(crate) in_transit: usize,
}

/// The template at `position`, fitted to `points`.
pub(crate) fn fit(points: &Points, template: &impl Shape, position: Position) -> Fit {
    let shapes: Vec<f64> = points
        .time
        .iter()
        .map(|&t| template.shape(position.x(t)))
        .collect();
    fit_shapes(points, &shapes)
}

/// The fit for given template values `shapes`, one per point.
pub(crate) fn fit_shapes(points: &Points, shapes: &[f64]) -> Fit {
    let (mut rs, mut ss) = (Sum::default(), Sum::default());
    let mut in_transit = 0;
    for ((&r, &w), &s) in points.residual.iter().zip(&points.weight).zip(shapes) {
        if s > 0.0 {
            in_transit += 1;
            rs.add(w * r * s);
            ss.add(w * s * s);
        }
    }
    let (rs, ss) = (rs.value(), ss.value());
    let depth = if rs > 0.0 { rs / ss } else { 0.0 };
    Fit {
        depth,
        chi2: chi2_shapes(points, shapes, depth),
        in_transit,
    }
}

/// The chi-squared for given template values at a given depth.
pub(crate) fn chi2_shapes(points: &Points, shapes: &[f64], depth: f64) -> f64 {
    let mut sum = Sum::default();
    for ((&r, &w), &s) in points.residual.iter().zip(&points.weight).zip(shapes) {
        let e = depth.mul_add(-s, r);
        sum.add(w * e * e);
    }
    sum.value()
}

/// The gain over the flat model of the template at `position`, `S_rs² /
/// S_ss` (0 where `S_rs <= 0`): the flat model's chi-squared less the best
/// fit's, without the second pass.
///
/// Only the points within half a duration of a transit can be in it, so
/// with the times increasing and the duration shorter than the period,
/// each transit's run of points is found by bisection and the rest are
/// not visited: about `N D / P` points, not `N`.
pub(crate) fn gain(points: &Points, template: &impl Shape, position: Position) -> f64 {
    let (mut rs, mut ss) = (Sum::default(), Sum::default());
    let mut add = |i: usize| {
        let s = template.shape(position.x(points.time[i]));
        if s > 0.0 {
            let w = points.weight[i];
            rs.add(w * points.residual[i] * s);
            ss.add(w * s * s);
        }
    };
    let time = &points.time;
    let Position {
        period,
        duration,
        t0,
    } = position;
    if duration < period {
        let half = 0.5 * duration;
        let (first, last) = (time[0], time[time.len() - 1]);
        let mut centre = period.mul_add(((first - half - t0) / period).floor(), t0);
        let mut from = 0;
        while centre - half <= last {
            let lo = from + time[from..].partition_point(|&t| t < centre - half);
            let hi = lo + time[lo..].partition_point(|&t| t <= centre + half);
            (lo..hi).for_each(&mut add);
            from = hi;
            centre += period;
        }
    } else {
        (0..time.len()).for_each(add);
    }
    let (rs, ss) = (rs.value(), ss.value());
    if rs > 0.0 { rs * (rs / ss) } else { 0.0 }
}
