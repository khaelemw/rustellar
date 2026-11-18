//! Refinement of a peak: the exact fit's mid-transit time, duration and
//! period, each free to move continuously around where the grid put it.
//!
//! The gain is maximised over one parameter at a time by golden-section
//! search, round after round, until a round gains nothing. The mid-transit
//! time is held at the transit nearest the light curve's middle while the
//! period moves, so a change of period moves the transits at both ends
//! equally and the two barely trade off.

use crate::fit::{Fit, Points, Position, fit, gain};
use crate::template::Template;

/// How far each parameter may move from where it starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Reach {
    /// The mid-transit time, either way, in days.
    pub(crate) t0: f64,
    /// The duration, by up to this factor either way (more than 1).
    pub(crate) duration: f64,
    /// The period, either way, in days.
    pub(crate) period: f64,
}

/// A refined peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Refined {
    /// The mid-transit time is the first after the light curve's first
    /// time less half a period.
    pub(crate) position: Position,
    pub(crate) fit: Fit,
}

/// The largest of `f` on `[a, b]` by golden-section search, or `start` if
/// nothing found beats `f(start)`; with the value there.
fn golden(mut f: impl FnMut(f64) -> f64, start: f64, mut a: f64, mut b: f64) -> (f64, f64) {
    let mut top = (start, f(start));
    let mut at = |x: f64| {
        let y = f(x);
        if y > top.1 {
            top = (x, y);
        }
        y
    };
    let ratio = 0.5 * (5_f64.sqrt() - 1.0);
    let mut c = b - ratio * (b - a);
    let mut d = a + ratio * (b - a);
    let (mut fc, mut fd) = (at(c), at(d));
    while b - a > 4.0 * f64::EPSILON * a.abs().max(b.abs()) {
        if fc >= fd {
            (b, d, fd) = (d, c, fc);
            c = b - ratio * (b - a);
            fc = at(c);
        } else {
            (a, c, fc) = (c, d, fd);
            d = a + ratio * (b - a);
            fd = at(d);
        }
    }
    top
}

/// The peak at `start` refined within `reach` of it. `points` must not be
/// empty.
pub(crate) fn refine(
    points: &Points,
    template: &Template,
    start: Position,
    reach: Reach,
) -> Refined {
    let first = points.time[0];
    let middle = 0.5 * (first + points.time[points.time.len() - 1]);
    // The transit nearest the middle.
    let orbits = ((middle - start.t0) / start.period).round();
    let mut centre = start.period.mul_add(orbits, start.t0);
    let (mut period, mut duration) = (start.period, start.duration);
    let at = |period: f64, duration: f64, t0: f64| {
        gain(
            points,
            template,
            Position {
                period,
                duration,
                t0,
            },
        )
    };
    let mut best = at(period, duration, centre);
    // Each search starts from the best so far, so only the last gain is
    // needed: the round's.
    loop {
        let before = best;
        (centre, _) = golden(
            |t| at(period, duration, t),
            centre,
            centre - reach.t0,
            centre + reach.t0,
        );
        (duration, _) = golden(
            |d| at(period, d, centre),
            duration,
            start.duration / reach.duration,
            start.duration * reach.duration,
        );
        (period, best) = golden(
            |p| at(p, duration, centre),
            period,
            start.period - reach.period,
            start.period + reach.period,
        );
        // Rounds gaining less than rounding are not worth another.
        if best - before <= 1e-14 * best {
            break;
        }
    }
    // Back to the first transit after the first time less half a period.
    let back = ((centre - first) / period + 0.5).floor();
    let t0 = period.mul_add(-back, centre);
    let position = Position {
        period,
        duration,
        t0,
    };
    Refined {
        position,
        fit: fit(points, template, position),
    }
}
