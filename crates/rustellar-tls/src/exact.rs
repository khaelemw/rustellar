//! The oracle: the best mid-transit time at one period and duration, with
//! no grid in the end. Slow and plain, to check the fast search against.
//!
//! Minimising the chi-squared is maximising the gain over the flat model,
//! `S_rs² / S_ss` (where `S_rs > 0`), which needs only the points in
//! transit. The gain is continuous in the mid-transit time (the template is
//! 0 at contact), and smooth between the times a point enters or leaves the
//! transit. So it is taken on a dense grid of mid-transit times, then
//! polished by a golden-section search around the grid's best.

use crate::fit::{Fit, Points, Position, Sum, fit};
use crate::template::Template;

/// The best fit at one period and duration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Best {
    /// Its mid-transit time, within a period after the light curve's
    /// first time.
    pub(crate) t0: f64,
    pub(crate) fit: Fit,
}

/// The points folded on a period: offsets from the first time in `[0, P)`,
/// sorted, with the points' indices.
struct Folded {
    offset: Vec<f64>,
    index: Vec<usize>,
}

impl Folded {
    fn new(points: &Points, origin: f64, period: f64) -> Self {
        let mut pairs: Vec<(f64, usize)> = points
            .time
            .iter()
            .enumerate()
            .map(|(i, &t)| (((t - origin) / period).rem_euclid(1.0) * period, i))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        Self {
            offset: pairs.iter().map(|p| p.0).collect(),
            index: pairs.iter().map(|p| p.1).collect(),
        }
    }

    /// The indices of the points within `half` of offset `tau`, going round
    /// the fold.
    fn near(&self, tau: f64, half: f64, period: f64) -> impl Iterator<Item = usize> + '_ {
        let n = self.offset.len();
        let start = tau - half;
        let end = tau + half;
        let range = |low: f64, high: f64| {
            let a = self.offset.partition_point(|&o| o < low);
            let b = self.offset.partition_point(|&o| o <= high);
            a..b.max(a)
        };
        // The window, and its parts wrapped round either end of the fold.
        let middle = range(start.max(0.0), end.min(period));
        let before = if start < 0.0 {
            range(start + period, period)
        } else {
            n..n
        };
        let after = if end > period {
            range(0.0, end - period)
        } else {
            0..0
        };
        before
            .chain(middle)
            .chain(after)
            .map(move |k| self.index[k])
    }
}

/// The gain over the flat model of the template at `t0`.
fn gain(
    points: &Points,
    folded: &Folded,
    template: &Template,
    origin: f64,
    period: f64,
    duration: f64,
    tau: f64,
) -> f64 {
    let position = Position {
        period,
        duration,
        t0: origin + tau,
    };
    let (mut rs, mut ss) = (Sum::default(), Sum::default());
    for i in folded.near(tau, 0.5 * duration, period) {
        let s = template.shape(position.x(points.time[i]));
        let w = points.weight[i];
        rs.add(w * points.residual[i] * s);
        ss.add(w * s * s);
    }
    let (rs, ss) = (rs.value(), ss.value());
    if rs > 0.0 { rs * (rs / ss) } else { 0.0 }
}

/// The best mid-transit time at one period and duration: the best of a
/// grid of `steps` per duration, then polished. `points` must not be empty;
/// `period` and `duration` positive and finite, `duration < period`.
pub(crate) fn best(
    points: &Points,
    template: &Template,
    period: f64,
    duration: f64,
    steps: u32,
) -> Best {
    let origin = points.time[0];
    let folded = Folded::new(points, origin, period);
    let gain_at = |tau: f64| gain(points, &folded, template, origin, period, duration, tau);

    let step = duration / f64::from(steps);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a positive count, far below usize::MAX"
    )]
    let count = (period / step).ceil() as usize;
    let mut best_tau = 0.0;
    let mut best_gain = gain_at(0.0);
    for k in 1..count {
        #[expect(clippy::cast_precision_loss, reason = "a grid index")]
        let tau = k as f64 * step;
        let g = gain_at(tau);
        if g > best_gain {
            (best_tau, best_gain) = (tau, g);
        }
    }

    // Golden-section search for the largest gain within a step either side,
    // keeping the best time seen (the grid's, if nothing beats it).
    let mut top = (best_tau, best_gain);
    let mut at = |tau: f64| {
        let tau = tau.rem_euclid(period);
        let g = gain_at(tau);
        if g > top.1 {
            top = (tau, g);
        }
        g
    };
    let ratio = 0.5 * (5_f64.sqrt() - 1.0);
    let (mut a, mut b) = (best_tau - step, best_tau + step);
    let mut c = b - ratio * (b - a);
    let mut d = a + ratio * (b - a);
    let (mut gc, mut gd) = (at(c), at(d));
    while b - a > 4.0 * f64::EPSILON * period {
        if gc >= gd {
            (b, d, gd) = (d, c, gc);
            c = b - ratio * (b - a);
            gc = at(c);
        } else {
            (a, c, gc) = (c, d, gd);
            d = a + ratio * (b - a);
            gd = at(d);
        }
    }
    let tau = top.0;
    let t0 = origin + tau;
    Best {
        t0,
        fit: fit(
            points,
            template,
            Position {
                period,
                duration,
                t0,
            },
        ),
    }
}
