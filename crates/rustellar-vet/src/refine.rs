use rustellar_core::{Epoch, LightCurve, TimeDelta};

use crate::fit::{fit_depth, phase};
use crate::sum::Sum;
use crate::{Candidate, Fit, Shape, VetError};

/// A candidate with its mid-transit time and duration refined, and its fit
/// there.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Refined {
    /// The candidate at the refined mid-transit time and duration, with the
    /// fitted depth.
    pub candidate: Candidate,
    /// The fit there.
    pub fit: Fit,
}

/// Grid steps across each range, then in each finer pass.
const STEPS: u32 = 40;
/// Finer passes, each a tenth of the last step either side of the best.
const PASSES: u32 = 2;

/// `candidate`'s mid-transit time and duration, refined at its period: the
/// dip (depth above 0) that lowers the chi-squared most, with the
/// mid-transit time within half a duration of the candidate's and the
/// duration from half to twice its own.
///
/// - For [`Shape::BOX`], exactly: the fit changes only where an edge
///   crosses a point, so every run of points in phase is scored (edges
///   half-way between neighbours), and the best is the optimum over every
///   time and duration in the ranges.
/// - For a limb-darkened shape, which varies smoothly: a grid of 41 times
///   by 41 durations (evenly in the logarithm), then twice a grid of the
///   same size around the best, each a tenth as wide.
///
/// The period stays the candidate's; so do the shape's other parameters.
///
/// # Errors
///
/// As [`fit_depth`], at the candidate itself; and
/// [`VetError::NoPointsInTransit`] if no dip is found in the ranges.
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, LightCurve, Shape, TimeDelta, refine};
///
/// // A 0.5% box dip from 0.70 to 0.80 every 3 days, every 30 minutes;
/// // the candidate a little off in time and duration.
/// let time: Vec<f64> = (0..576).map(|i| f64::from(i) / 48.0).collect();
/// let flux = time
///     .iter()
///     .map(|&t| if (t - 0.7).rem_euclid(3.0) < 0.1 { 0.995 } else { 1.0 })
///     .collect();
/// let lc = LightCurve::from_days(time, flux, vec![1e-3; 576])?;
/// let candidate = Candidate::new(
///     TimeDelta::from_days(3.0)?,
///     Epoch::from_days(0.73)?,
///     TimeDelta::from_days(0.07)?,
/// )?;
///
/// let refined = refine(&lc, &candidate, Shape::BOX)?;
/// assert!((refined.candidate.transit_time().as_days() - 0.75).abs() < 0.011);
/// assert!((refined.fit.depth - 0.005).abs() < 1e-15);
/// assert_eq!(refined.fit.points_in_transit, 20);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn refine(lc: &LightCurve, candidate: &Candidate, shape: Shape) -> Result<Refined, VetError> {
    // The candidate's own fit first, for its errors.
    fit_depth(lc, candidate, shape)?;
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    // The longest duration allowed: twice the candidate's, below the period
    // (half of it for a limb-darkened shape).
    let cap = if shape == Shape::BOX { 1.0 } else { 0.5 };
    let longest = (2.0 * duration).min(cap * period);
    let (shift, length) = if shape == Shape::BOX {
        best_run(lc, candidate, longest)
    } else {
        best_grid(lc, candidate, shape, longest)
    }
    .ok_or(VetError::NoPointsInTransit)?;
    let refined = at(candidate, shift, length)?;
    let fit = fit_depth(lc, &refined, shape)?;
    Ok(Refined {
        candidate: refined.with_depth(fit.depth)?,
        fit,
    })
}

/// The candidate moved by `shift` days with `length` days' duration.
fn at(candidate: &Candidate, shift: f64, length: f64) -> Result<Candidate, VetError> {
    let t0 = Epoch::from_days(candidate.transit_time().as_days() + shift).expect("finite");
    let duration = TimeDelta::from_days(length).expect("finite");
    Candidate::new(candidate.period(), t0, duration)
}

/// The best box over every run of points in phase: its shift and length.
fn best_run(lc: &LightCurve, candidate: &Candidate, longest: f64) -> Option<(f64, f64)> {
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    let (shortest, reach) = (0.5 * duration, 0.5 * duration);
    // Every point that any allowed box can hold: within half the longest
    // duration of a centre at most `reach` from the candidate's.
    let bound = (reach + 0.5 * longest).min(0.5 * period);
    let t0 = candidate.transit_time().as_days();
    // The points no box can hold, summed apart, so that a run's
    // complement is exactly the rest (zero for a run of every point).
    let (mut far_w, mut far_wy) = (Sum::default(), Sum::default());
    let mut near: Vec<(f64, f64, f64)> = vec![];
    for ((t, &y), &e) in lc.time().iter().zip(lc.flux()).zip(lc.flux_err()) {
        let w = 1.0 / (e * e);
        let p = phase(t.as_days() - t0, period);
        if p.abs() < bound {
            near.push((p, w, w * y));
        } else {
            far_w.add(w);
            far_wy.add(w * y);
        }
    }
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (far_w, far_wy) = (far_w.get(), far_wy.get());
    // Prefix sums of weights and weighted fluxes, compensated.
    let (mut pw, mut pwy) = (vec![0.0], vec![0.0]);
    let (mut sw, mut swy) = (Sum::default(), Sum::default());
    for &(_, w, wy) in &near {
        sw.add(w);
        swy.add(wy);
        pw.push(sw.get());
        pwy.push(swy.get());
    }
    let n = near.len();
    let (total_w, total_wy) = (far_w + pw[n], far_wy + pwy[n]);
    let mean = total_wy / total_w;
    // An edge half-way to the next point, or at the bound past the last.
    let left_edge = |i: usize| {
        if i == 0 {
            -bound
        } else {
            0.5 * (near[i - 1].0 + near[i].0)
        }
    };
    let right_edge = |j: usize| {
        if j + 1 == n {
            bound
        } else {
            0.5 * (near[j].0 + near[j + 1].0)
        }
    };
    let mut best: Option<(f64, f64, f64)> = None;
    for i in 0..n {
        // A run cannot start between two points at the same phase.
        if i > 0 && near[i - 1].0 == near[i].0 {
            continue;
        }
        let left = left_edge(i);
        for j in i..n {
            if j + 1 < n && near[j].0 == near[j + 1].0 {
                continue;
            }
            let right = right_edge(j);
            let length = right - left;
            if length > longest {
                break;
            }
            let centre = 0.5 * (left + right);
            if length < shortest || centre.abs() > reach {
                continue;
            }
            let w_in = pw[j + 1] - pw[i];
            let wy_in = pwy[j + 1] - pwy[i];
            let w_out = far_w + pw[i] + (pw[n] - pw[j + 1]);
            if w_in <= 0.0 || w_out <= 0.0 {
                continue;
            }
            let mean_in = wy_in / w_in;
            if mean_in >= mean {
                continue;
            }
            // The box's drop in chi-squared from a flat line.
            let gain = (mean_in - mean).powi(2) * w_in * total_w / w_out;
            if best.is_none_or(|(g, _, _)| gain > g) {
                best = Some((gain, centre, length));
            }
        }
    }
    best.map(|(_, centre, length)| (centre, length))
}

/// The best limb-darkened fit on a grid and finer grids around its best:
/// its shift and length.
///
/// Each trial is scored by the drop in chi-squared, `S_sy^2 / S_ss`, from
/// sums about the weighted mean flux: `S_sy = sum w s (y - mean)` and
/// `S_ss = sum w s^2 - (sum w s)^2 / W`. Only points with a shape above 0
/// enter them, so each trial sums the points in its own transit, found by
/// phase among those any trial can reach, and never the whole light curve.
fn best_grid(
    lc: &LightCurve,
    candidate: &Candidate,
    shape: Shape,
    longest: f64,
) -> Option<(f64, f64)> {
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    let t0 = candidate.transit_time().as_days();
    let (mut w_all, mut wy_all) = (Sum::default(), Sum::default());
    for (&y, &e) in lc.flux().iter().zip(lc.flux_err()) {
        let w = 1.0 / (e * e);
        w_all.add(w);
        wy_all.add(w * y);
    }
    let (w_all, n) = (w_all.get(), lc.len());
    let mean = wy_all.get() / w_all;
    // Every point any trial can reach (shift up to half the duration, half
    // the longest length either side), sorted by phase: phase, weight and
    // flux less the mean.
    let reach = 0.5 * duration + 0.5 * longest;
    let mut near: Vec<(f64, f64, f64)> = lc
        .time()
        .iter()
        .zip(lc.flux())
        .zip(lc.flux_err())
        .map(|((t, &y), &e)| (phase(t.as_days() - t0, period), 1.0 / (e * e), y - mean))
        .filter(|&(ph, _, _)| ph.abs() < reach)
        .collect();
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Where a trial's transit cannot wrap past half the period, only the
    // points within it are summed.
    let unwrapped = reach + 0.5 * duration < 0.5 * period;
    let gain = |shift: f64, length: f64| -> Option<f64> {
        let profile = shape.profile(period, length).ok()?;
        let (lo, hi) = if unwrapped {
            let half = 0.5 * length;
            (
                near.partition_point(|p| p.0 <= shift - half),
                near.partition_point(|p| p.0 < shift + half),
            )
        } else {
            (0, near.len())
        };
        let (mut ss, mut s1, mut sy) = (Sum::default(), Sum::default(), Sum::default());
        let mut points = 0;
        for &(ph, w, yc) in &near[lo..hi] {
            let s = profile.at(phase(ph - shift, period));
            if s > 0.0 {
                ss.add(w * s * s);
                s1.add(w * s);
                sy.add(w * s * yc);
                points += 1;
            }
        }
        // A fit needs points both in and out of transit.
        (points > 0 && points < n).then_some(())?;
        let (s1, sy) = (s1.get(), sy.get());
        let sss = ss.get() - s1 * s1 / w_all;
        let beta = sy / sss;
        let alpha = mean - beta * s1 / w_all;
        (beta < 0.0 && alpha > 0.0).then_some(sy * beta)
    };
    // Shift in days; length as its logarithm.
    let (mut shift_lo, mut shift_hi) = (-0.5 * duration, 0.5 * duration);
    let (mut log_lo, mut log_hi) = ((0.5 * duration).ln(), longest.ln());
    let mut best: Option<(f64, f64, f64)> = None;
    for _ in 0..=PASSES {
        let (shift_step, log_step) = (
            (shift_hi - shift_lo) / f64::from(STEPS),
            (log_hi - log_lo) / f64::from(STEPS),
        );
        for a in 0..=STEPS {
            let shift = shift_lo + shift_step * f64::from(a);
            for b in 0..=STEPS {
                let length = (log_lo + log_step * f64::from(b)).exp().min(longest);
                let Some(g) = gain(shift, length) else {
                    continue;
                };
                if best.is_none_or(|(bg, _, _)| g > bg) {
                    best = Some((g, shift, length));
                }
            }
        }
        let (_, shift, length) = best?;
        // A tenth as wide around the best, within the original ranges.
        shift_lo = (shift - shift_step).max(-0.5 * duration);
        shift_hi = (shift + shift_step).min(0.5 * duration);
        log_lo = (length.ln() - log_step).max((0.5 * duration).ln());
        log_hi = (length.ln() + log_step).min(longest.ln());
    }
    best.map(|(_, shift, length)| (shift, length))
}
