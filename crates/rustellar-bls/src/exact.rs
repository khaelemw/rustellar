//! The exact search: the best box at one period and duration over every
//! mid-transit time, with no grid in phase. Slow and simple; the fast search
//! is checked against it.
//!
//! Fold the light curve at the period. A box whose left edge is at phase `a`
//! holds the points with `a < phase < a + D`, so as `a` slides round the
//! orbit the set of points in transit changes only when an edge meets a
//! point. Each set it takes is a run of neighbouring points in phase order,
//! `i..j`, and is possible exactly when
//!
//! ```text
//! phase[j-1] - phase[i] < D                 (the run fits in the box)
//! phase[j] - phase[i-1] >= D                (and the box fits between its neighbours)
//! phase[i-1] < phase[i], phase[j-1] < phase[j]
//! ```
//!
//! (with phases continued past one orbit). For each first point `i` the
//! possible ends `j` form a range that only moves forward as `i` does, so two
//! pointers find every possible run, about two per point, and running totals
//! give each run's sums.

use rustellar_core::{LightCurve, TimeDelta};

use crate::{
    BlsError, BoxFit, TransitBox,
    box_fit::{Weighted, offset_from_transit},
    totals::{Orbit, Total, box_sums},
};

/// The best box found by a search, and its fit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct BestBox {
    /// The box: the period and duration searched, and the best mid-transit
    /// time.
    pub transit: TransitBox,
    /// Its fit to the light curve.
    pub fit: BoxFit,
}

impl TransitBox {
    /// The box of this period and duration with the largest `delta_chi2`
    /// among those with a dip (`depth > 0`), over every mid-transit time.
    ///
    /// Every set of points a box can hold is tried, so no phase grid,
    /// astropy's included, can find a better one. Boxes that hold no point,
    /// or every point, have no fit and are skipped. The mid-transit time is
    /// the middle of the range of times that give the best set.
    ///
    /// Returns `None` if no box has a dip, for example if the flux is flat.
    ///
    /// # Errors
    ///
    /// - As [`TransitBox::new`] for a bad period or duration.
    /// - [`BlsError::OutOfRange`] as for [`TransitBox::fit`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::TransitBox;
    /// use rustellar_core::{LightCurve, TimeDelta};
    ///
    /// // Every 6 hours for 10 days, with a dip of 0.01 at 2.0 and 2.25 days into
    /// // each 5-day orbit.
    /// let time: Vec<f64> = (0..40).map(|i| 0.25 * f64::from(i)).collect();
    /// let flux = time
    ///     .iter()
    ///     .map(|t| if (t % 5.0 - 2.1).abs() < 0.2 { 0.99 } else { 1.0 })
    ///     .collect();
    /// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 40])?;
    ///
    /// let best = TransitBox::best(&lc, TimeDelta::from_days(5.0)?, TimeDelta::from_days(0.5)?)?
    ///     .expect("there is a dip");
    /// assert_eq!(best.fit.n_in, 4);
    /// assert!((best.fit.depth - 0.01).abs() < 1e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn best(
        lc: &LightCurve,
        period: TimeDelta,
        duration: TimeDelta,
    ) -> Result<Option<BestBox>, BlsError> {
        let reference = lc.time()[0];
        // Checks the period and duration; the mid-transit time is replaced.
        Self::new(period, duration, reference)?;
        let weighted = Weighted::new(lc)?;
        let folded = Folded::new(lc, &weighted, period.as_days());
        Ok(folded
            .best(&weighted, duration.as_days())?
            .map(|(centre, fit)| BestBox {
                transit: Self::new(
                    period,
                    duration,
                    reference + TimeDelta::from_days(centre).expect("finite"),
                )
                .expect("checked above"),
                fit,
            }))
    }
}

/// A light curve folded at one period, for the exact search at any duration.
pub(crate) struct Folded {
    period: f64,
    /// Phases from the first time, in `[-P/2, P/2]`, sorted, then again one
    /// orbit on, so a run can wrap round.
    phase: Vec<f64>,
    /// Running totals in that order: `totals[k]` is over the first `k`.
    totals: Vec<Total>,
}

impl Folded {
    pub(crate) fn new(lc: &LightCurve, weighted: &Weighted, period: f64) -> Self {
        let reference = lc.time()[0].as_btjd();
        let n = lc.len();
        let mut order: Vec<(f64, usize)> = lc
            .time()
            .iter()
            .enumerate()
            .map(|(k, t)| (offset_from_transit(t.as_btjd(), reference, period), k))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let phase = order
            .iter()
            .map(|o| o.0)
            .chain(order.iter().map(|o| o.0 + period))
            .collect();
        let mut totals = Vec::with_capacity(2 * n + 1);
        let mut total = Total::default();
        totals.push(total);
        for &(_, k) in order.iter().chain(&order) {
            total.weight.add(weighted.weights[k]);
            total.flux.add(weighted.centred[k]);
            totals.push(total);
        }
        Self {
            period,
            phase,
            totals,
        }
    }

    /// The best box over the durations, in days: which duration, the box's
    /// middle, in days from the first time, and its fit. The first of equal
    /// boxes is kept, and durations not shorter than the period are skipped.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the search calls this; not written yet")
    )]
    pub(crate) fn best_of(
        &self,
        weighted: &Weighted,
        durations: &[f64],
    ) -> Result<Option<(usize, f64, BoxFit)>, BlsError> {
        let mut best: Option<(usize, f64, BoxFit)> = None;
        for (k, &d) in durations.iter().enumerate() {
            if d >= self.period {
                continue;
            }
            if let Some((centre, fit)) = self.best(weighted, d)? {
                if best.is_none_or(|b| fit.delta_chi2 > b.2.delta_chi2) {
                    best = Some((k, centre, fit));
                }
            }
        }
        Ok(best)
    }

    /// The best box of duration `d` days: its middle, in days from the first
    /// time, and its fit. See [`TransitBox::best`].
    pub(crate) fn best(
        &self,
        weighted: &Weighted,
        d: f64,
    ) -> Result<Option<(f64, BoxFit)>, BlsError> {
        Ok(self
            .best_between(weighted, d, d)?
            .map(|(centre, _, fit)| (centre, fit)))
    }

    /// The best box of any duration from `shortest` to `longest` days: its
    /// middle, in days from the first time, its duration, and its fit.
    ///
    /// A run of points `i..j` is held by a box of duration `d` exactly when
    /// `d` is above the run's span, `phase[j-1] - phase[i]`, and not above
    /// the gap between its neighbours, `phase[j] - phase[i-1]`. So the runs
    /// to try are those that span less than `longest` and whose neighbours
    /// are at least `shortest` apart: two pointers again. Every duration in
    /// that range gives the same fit, so each edge is put halfway between
    /// the last point in and the first point out, the duration kept within
    /// the range, and the box centred in the times that hold the run.
    pub(crate) fn best_between(
        &self,
        weighted: &Weighted,
        shortest: f64,
        longest: f64,
    ) -> Result<Option<(f64, f64, BoxFit)>, BlsError> {
        let (p, phase, totals) = (self.period, &self.phase, &self.totals);
        let n = phase.len() / 2;
        let previous = |i: usize| {
            if i == 0 {
                phase[n - 1] - p
            } else {
                phase[i - 1]
            }
        };
        let orbit = Orbit::new(&totals[0], &totals[n]);
        // The best run's statistic, first point and one past its last.
        let mut best: Option<(f64, usize, usize)> = None;
        let (mut first_end, mut last_end) = (1, 1);
        for i in 0..n {
            let before = previous(i);
            // The first possible end: the neighbours are `shortest` apart.
            first_end = first_end.max(i + 1);
            while first_end < i + n && phase[first_end] - before < shortest {
                first_end += 1;
            }
            // One past the last possible end: the run spans under `longest`.
            last_end = last_end.max(i + 1);
            while last_end < i + n && phase[last_end] - phase[i] < longest {
                last_end += 1;
            }
            if before >= phase[i] {
                continue; // i and the point before it cannot be told apart
            }
            // A run of every point leaves none out of transit.
            for j in first_end..=last_end.min(i + n - 1) {
                if phase[j - 1] >= phase[j] {
                    continue;
                }
                let delta_chi2 = orbit.statistic(&totals[i], &totals[j], &totals[i + n])?;
                if delta_chi2 > best.map_or(0.0, |b| b.0) {
                    best = Some((delta_chi2, i, j));
                }
            }
        }
        let Some((_, i, j)) = best else {
            return Ok(None);
        };
        let [inside, outside] = box_sums(&totals[i], &totals[j], &totals[i + n]);
        let fit = BoxFit::from_sums(weighted.mean, inside, outside, j - i)?;
        // Edges halfway between the points either side, the duration kept
        // in range, and the left edges that give this run, and their middle.
        let before = previous(i);
        let halfway = 0.5 * (phase[j - 1] + phase[j]) - 0.5 * (before + phase[i]);
        let d = halfway.clamp(shortest, longest);
        let low = before.max(phase[j - 1] - d);
        let high = phase[i].min(phase[j] - d);
        let edge = if low < high { 0.5 * (low + high) } else { low };
        Ok(Some((edge + 0.5 * d, d, fit)))
    }
}
