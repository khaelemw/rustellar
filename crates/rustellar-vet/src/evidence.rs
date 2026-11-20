use rustellar_core::{Epoch, LightCurve, TimeDelta};

use crate::fit::{phase, shapes};
use crate::grouped::{Grouped, solve};
use crate::{Candidate, Shape, VetError};

/// One predicted transit: where it falls, how many points it has, and its
/// own depth.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Transit {
    /// Its number, counted from the candidate's mid-transit time (0),
    /// negative before it.
    pub epoch: i64,
    /// Its predicted mid-transit time.
    pub time: Epoch,
    /// The points where the shape is above 0.
    pub points: usize,
    /// The points a transit would have at the light curve's typical
    /// cadence: the duration over the median step between points. `None`
    /// for a light curve with no step (one time).
    pub expected: Option<f64>,
    /// Its fractional depth at mid-transit, every transit fitted together
    /// with one shared baseline; `None` without points.
    pub depth: Option<f64>,
    /// Its standard error.
    pub depth_err: Option<f64>,
}

impl Transit {
    /// The share of the expected points it has: 1 for a transit wholly in
    /// the data, less where it falls partly in a gap or off an edge.
    #[must_use]
    pub fn coverage(&self) -> Option<f64> {
        #[expect(clippy::cast_precision_loss, reason = "a count of points")]
        self.expected.map(|e| self.points as f64 / e)
    }
}

/// The odd and even transits' depths compared: each group fitted with one
/// depth, both with one shared baseline.
///
/// The difference's error counts the two depths' correlation through the
/// shared baseline. Many points in each group make that small, but it is
/// never left out.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OddEven {
    /// The odd transits' depth.
    pub odd: f64,
    /// Its standard error.
    pub odd_err: f64,
    /// The even transits' depth (epoch 0 is even).
    pub even: f64,
    /// Its standard error.
    pub even_err: f64,
    /// Odd minus even.
    pub difference: f64,
    /// Its standard error.
    pub difference_err: f64,
    /// Odd transits with points.
    pub odd_transits: usize,
    /// Even transits with points.
    pub even_transits: usize,
}

impl OddEven {
    /// How many standard errors apart the two depths are: positive when
    /// the odd transits are the deeper.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        self.difference / self.difference_err
    }
}

/// The evidence transit by transit for one candidate.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Evidence {
    /// Every transit predicted within the light curve's span (its window
    /// overlapping first to last time), in order.
    pub transits: Vec<Transit>,
    /// The median step between consecutive times, if there is one.
    pub cadence: Option<TimeDelta>,
    /// The odd and even transits compared, where both have points.
    pub odd_even: Option<OddEven>,
}

impl Evidence {
    /// The transits with at least one point.
    #[must_use]
    pub fn with_data(&self) -> usize {
        self.transits.iter().filter(|t| t.points > 0).count()
    }
}

/// The evidence for `candidate` in `lc`, for `shape`: each predicted
/// transit's points, coverage and depth, and the odd and even transits
/// compared.
///
/// # Errors
///
/// - [`VetError::NoPointsInTransit`], [`VetError::NoPointsOutOfTransit`],
///   [`VetError::BaselineNotPositive`] and
///   [`VetError::DurationTooLongForShape`] as [`fit_depth`](crate::fit_depth).
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, LightCurve, Shape, TimeDelta, evidence};
///
/// // Box dips every 3 days, 0.4% and 0.6% deep in turn: an eclipsing
/// // binary at twice the period.
/// let time: Vec<f64> = (0..1728).map(|i| f64::from(i) / 144.0).collect();
/// let flux = time
///     .iter()
///     .map(|&t| {
///         let n = ((t - 0.75) / 3.0).round();
///         let deep = if n.rem_euclid(2.0) == 1.0 { 0.006 } else { 0.004 };
///         if (t - 0.75 - 3.0 * n).abs() < 0.05 { 1.0 - deep } else { 1.0 }
///     })
///     .collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 1728])?;
/// let candidate = Candidate::new(
///     TimeDelta::from_days(3.0)?,
///     Epoch::from_btjd(0.75)?,
///     TimeDelta::from_days(0.1)?,
/// )?;
///
/// let e = evidence(&lc, &candidate, Shape::BOX)?;
/// assert_eq!(e.transits.len(), 4);
/// let oe = e.odd_even.unwrap();
/// assert!((oe.difference - 0.002).abs() < 1e-12);
/// assert!(oe.sigma() > 5.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn evidence(
    lc: &LightCurve,
    candidate: &Candidate,
    shape: Shape,
) -> Result<Evidence, VetError> {
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    let t0 = candidate.transit_time().as_btjd();
    let profile = shape.profile(period, duration)?;
    let s = shapes(lc, candidate, &profile);
    let times: Vec<f64> = lc.time().iter().map(|t| t.as_btjd()).collect();
    let Layout { epochs, predicted } = layout(&times, candidate);
    let index = |n: i64| predicted.binary_search(&n).ok();
    // Each transit its own group.
    let group: Vec<Option<usize>> = epochs.iter().map(|&n| index(n)).collect();
    let each = solve(lc.flux(), lc.flux_err(), &s, &group, predicted.len())?;
    let cadence = median_step(&times);
    let expected = cadence.map(|c| duration / c);
    let transits = predicted
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            #[expect(clippy::cast_precision_loss, reason = "a transit number")]
            let centre = t0 + n as f64 * period;
            Transit {
                epoch: n,
                time: Epoch::from_btjd(centre).expect("finite"),
                points: each.points[k],
                expected,
                depth: each.depth(k),
                depth_err: each.cov_depth(k, k).map(f64::sqrt),
            }
        })
        .collect();
    // The odd (group 0) and even (group 1) transits.
    let parity: Vec<Option<usize>> = epochs
        .iter()
        .map(|&n| usize::from(n.rem_euclid(2) == 0).into())
        .collect();
    let odd_even = solve(lc.flux(), lc.flux_err(), &s, &parity, 2)
        .ok()
        .and_then(|fit| compare(&fit, &predicted, &each));
    Ok(Evidence {
        transits,
        cadence: cadence.map(|c| TimeDelta::from_days(c).expect("finite")),
        odd_even,
    })
}

/// The odd and even depths compared, where both groups have points.
fn compare(fit: &Grouped, predicted: &[i64], each: &Grouped) -> Option<OddEven> {
    let (odd, even) = (fit.depth(0)?, fit.depth(1)?);
    let (var_odd, var_even) = (fit.cov_depth(0, 0)?, fit.cov_depth(1, 1)?);
    let cov = fit.cov_depth(0, 1)?;
    let count = |even: bool| {
        predicted
            .iter()
            .zip(&each.points)
            .filter(|&(n, &p)| p > 0 && (n.rem_euclid(2) == 0) == even)
            .count()
    };
    Some(OddEven {
        odd,
        odd_err: var_odd.sqrt(),
        even,
        even_err: var_even.sqrt(),
        difference: odd - even,
        difference_err: (var_odd + var_even - 2.0 * cov).sqrt(),
        odd_transits: count(false),
        even_transits: count(true),
    })
}

/// The median of the positive steps between consecutive sorted times.
pub(crate) fn median_step(times: &[f64]) -> Option<f64> {
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut steps: Vec<f64> = sorted
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0.0)
        .collect();
    if steps.is_empty() {
        return None;
    }
    steps.sort_by(f64::total_cmp);
    let n = steps.len();
    Some(if n % 2 == 1 {
        steps[n / 2]
    } else {
        0.5 * (steps[n / 2 - 1] + steps[n / 2])
    })
}

/// Each point's transit number, and the transits predicted in the span.
pub(crate) struct Layout {
    /// Each point's transit number (the nearest mid-transit time's),
    /// counted from the candidate's.
    pub(crate) epochs: Vec<i64>,
    /// The transits whose window overlaps first to last time, in order.
    pub(crate) predicted: Vec<i64>,
}

/// The layout of `candidate`'s transits over `times` (days).
pub(crate) fn layout(times: &[f64], candidate: &Candidate) -> Layout {
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    let t0 = candidate.transit_time().as_btjd();
    let epochs: Vec<i64> = times
        .iter()
        .map(|&t| {
            let n = ((t - t0) - phase(t - t0, period)) / period;
            #[expect(clippy::cast_possible_truncation, reason = "a whole number of periods")]
            let n = n.round() as i64;
            n
        })
        .collect();
    let (first, last) = times
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &t| {
            (lo.min(t), hi.max(t))
        });
    #[expect(clippy::cast_possible_truncation, reason = "a whole number of periods")]
    let (lo, hi) = (
        ((first - t0 - 0.5 * duration) / period).floor() as i64,
        ((last - t0 + 0.5 * duration) / period).ceil() as i64,
    );
    let predicted = (lo..=hi)
        .filter(|&n| {
            #[expect(clippy::cast_precision_loss, reason = "a transit number")]
            let centre = t0 + n as f64 * period;
            centre + 0.5 * duration > first && centre - 0.5 * duration < last
        })
        .collect();
    Layout { epochs, predicted }
}
