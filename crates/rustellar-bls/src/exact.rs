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
    sums::{Sum, between},
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
    /// - [`BlsError::Overflow`] as for [`TransitBox::fit`].
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
        let (p, d) = (period.as_days(), duration.as_days());
        let n = lc.len();

        // Phases from the first time, in [-P/2, P/2], sorted, then continued
        // past one orbit so a run can wrap round.
        let mut order: Vec<(f64, usize)> = lc
            .time()
            .iter()
            .enumerate()
            .map(|(k, t)| (offset_from_transit(t.as_btjd(), reference.as_btjd(), p), k))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let phase: Vec<f64> = (0..2 * n)
            .map(|k| {
                if k < n {
                    order[k].0
                } else {
                    order[k - n].0 + p
                }
            })
            .collect();

        // Running totals of the weight and centred flux, in phase order.
        let mut weight_totals = vec![Sum::default()];
        let mut flux_totals = vec![Sum::default()];
        for k in 0..2 * n {
            let point = order[k % n].1;
            let (mut w, mut s) = (weight_totals[k], flux_totals[k]);
            w.add(weighted.weights[point]);
            s.add(weighted.centred[point]);
            weight_totals.push(w);
            flux_totals.push(s);
        }

        let mut best: Option<(f64, BoxFit)> = None;
        let (mut first_end, mut last_end) = (1, 1);
        for i in 0..n {
            let previous = if i == 0 {
                phase[n - 1] - p
            } else {
                phase[i - 1]
            };
            // The first possible end: the box must clear the point before i.
            first_end = first_end.max(i + 1);
            while first_end < i + n && phase[first_end] - previous < d {
                first_end += 1;
            }
            // One past the last possible end: the run must fit in the box.
            last_end = last_end.max(i + 1);
            while last_end < i + n && phase[last_end] - phase[i] < d {
                last_end += 1;
            }
            if previous >= phase[i] {
                continue; // i and the point before it cannot be told apart
            }
            // A run of every point leaves none out of transit.
            for j in first_end..=last_end.min(i + n - 1) {
                if phase[j - 1] >= phase[j] {
                    continue;
                }
                let inside = [
                    between(weight_totals[i], weight_totals[j]),
                    between(flux_totals[i], flux_totals[j]),
                ];
                let outside = [
                    between(weight_totals[j], weight_totals[i + n]),
                    between(flux_totals[j], flux_totals[i + n]),
                ];
                let fit = BoxFit::from_sums(weighted.mean, inside, outside, j - i)?;
                if fit.depth > 0.0 && best.is_none_or(|(_, b)| fit.delta_chi2 > b.delta_chi2) {
                    // The left edges that give this run, and their middle.
                    let low = previous.max(phase[j - 1] - d);
                    let high = phase[i].min(phase[j] - d);
                    let edge = if low < high { 0.5 * (low + high) } else { low };
                    best = Some((edge + 0.5 * d, fit));
                }
            }
        }

        Ok(best.map(|(centre, fit)| BestBox {
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
