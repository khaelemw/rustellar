//! Refinement of a peak: the exact fit's mid-transit time, duration and
//! period, each free to move continuously around where the grid put it.
//!
//! The gain is maximised over one parameter at a time by golden-section
//! search, then along the round's whole move, round after round, until a
//! round gains nothing worth having. The mid-transit
//! time is held at the transit nearest the light curve's middle while the
//! period moves, so a change of period moves the transits at both ends
//! equally and the two barely trade off.

use crate::fit::{Fit, Points, Position, fit, gain};
use crate::template::Template;

/// The most rounds of searches; far more than a peak needs.
const MAX_ROUNDS: usize = 100;

/// A round gaining less than this, relative to the gain, is the last: the
/// chi-squared is then within this of its least.
const TOLERANCE: f64 = 1e-12;

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
    for _ in 0..MAX_ROUNDS {
        let before = (best, centre, duration, period);
        // Each search starts from the best so far, so the last one's value
        // is the round's.
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
        // Then along the round's whole move, which follows a valley that
        // one parameter at a time can only zig-zag down (Powell's idea).
        let (dc, dd, dp) = (centre - before.1, duration - before.2, period - before.3);
        let along = |a: f64| {
            at(
                dp.mul_add(a, period),
                dd.mul_add(a, duration),
                dc.mul_add(a, centre),
            )
        };
        let (a, gained) = golden(along, 0.0, -1.0, 3.0);
        if gained > best {
            (centre, duration, period, best) = (
                dc.mul_add(a, centre),
                dd.mul_add(a, duration),
                dp.mul_add(a, period),
                gained,
            );
        }
        if best - before.0 <= TOLERANCE * best {
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
