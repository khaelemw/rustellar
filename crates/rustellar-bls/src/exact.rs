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

use std::ops::RangeInclusive;

use rustellar_core::{LightCurve, TimeDelta};

use crate::{
    BlsError, BoxFit, Objective, TransitBox,
    box_fit::{AtPeriod, Weighted},
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
        let times: Vec<f64> = lc.time().iter().map(|t| t.as_btjd()).collect();
        let folded = Folded::new(&times, &weighted, period.as_days());
        Ok(folded
            .best(&weighted, duration.as_days(), Objective::DeltaChi2)?
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
    /// `times` in BTJD, with weights `weighted`, folded at `period` days.
    pub(crate) fn new(times: &[f64], weighted: &Weighted, period: f64) -> Self {
        let n = times.len();
        // Each time's offset from the first, from its exact elapsed time.
        let at = AtPeriod::new(period);
        let elapsed = &weighted.elapsed;
        let last = elapsed.last().map_or(0.0, |&(s, _)| s);
        let mut order: Vec<(f64, usize)> = if at.short_up_to(last) {
            elapsed
                .iter()
                .enumerate()
                .map(|(k, &(s, e))| (at.offset_after_start_short(s, e), k))
                .collect()
        } else {
            elapsed
                .iter()
                .enumerate()
                .map(|(k, &(s, e))| (at.offset_after_start(s, e), k))
                .collect()
        };
        // By phase, then by point (the order a stable sort by phase gives).
        // In time order the phases are already sorted within each orbit, and
        // the stable sort merges those runs: faster over a few hundred
        // orbits or fewer, slower than the unstable sort over more. Either
        // gives the same order, the keys being all different.
        let by_phase = |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        if times[n - 1] - times[0] < ORBITS_TO_MERGE * period {
            order.sort_by(by_phase);
        } else {
            order.sort_unstable_by(by_phase);
        }
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

    /// The best box by `objective` over the durations, in days: which duration, the box's
    /// middle, in days from the first time, and its fit. The first of equal
    /// boxes is kept, and durations not shorter than the period are skipped.
    pub(crate) fn best_of(
        &self,
        weighted: &Weighted,
        durations: &[f64],
        objective: Objective,
    ) -> Result<Option<(usize, f64, BoxFit)>, BlsError> {
        let mut best: Option<(usize, f64, BoxFit)> = None;
        for (k, &d) in durations.iter().enumerate() {
            if d >= self.period {
                continue;
            }
            if let Some((centre, fit)) = self.best(weighted, d, objective)? {
                if best.is_none_or(|b| objective.of(&fit) > objective.of(&b.2)) {
                    best = Some((k, centre, fit));
                }
            }
        }
        Ok(best)
    }

    /// The best box by `objective` of duration `d` days: its middle, in days from the first
    /// time, and its fit. See [`TransitBox::best`].
    pub(crate) fn best(
        &self,
        weighted: &Weighted,
        d: f64,
        objective: Objective,
    ) -> Result<Option<(f64, BoxFit)>, BlsError> {
        Ok(self
            .best_between(weighted, d, d, objective)?
            .map(|(centre, _, fit)| (centre, fit)))
    }

    /// The best box by `objective` of any duration from `shortest` to `longest` days: its
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
        objective: Objective,
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
        let orbit = Orbit::new(&totals[0], &totals[n], objective);
        // The best run's score, first point and one past its last.
        let mut best: Option<(f64, usize, usize)> = None;
        let mut bounds: Option<Bounds> = None;
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
            let ends = first_end..=last_end.min(i + n - 1);
            if ends.clone().count() < LONG_RANGE {
                self.runs(&orbit, i, ends, &mut best)?;
            } else {
                // Many ends (dense data, a range of durations): skip the
                // blocks of ends that cannot beat the best run so far.
                let bounds = bounds.get_or_insert_with(|| Bounds::new(totals, n, &orbit));
                bounds.search(self, &orbit, i, ends, &mut best)?;
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

impl Folded {
    /// Scores the runs from `i` to each of `ends`, in order, keeping the
    /// first of the highest in `best`.
    fn runs(
        &self,
        orbit: &Orbit,
        i: usize,
        ends: RangeInclusive<usize>,
        best: &mut Option<(f64, usize, usize)>,
    ) -> Result<(), BlsError> {
        let (phase, totals) = (&self.phase, &self.totals);
        let n = phase.len() / 2;
        for j in ends {
            if phase[j - 1] >= phase[j] {
                continue;
            }
            let score = orbit.statistic(&totals[i], &totals[j], &totals[i + n])?;
            if score > best.map_or(0.0, |b| b.0) {
                *best = Some((score, i, j));
            }
        }
        Ok(())
    }
}

/// Below this many orbits in the data, a fold's phases are sorted by merging
/// the runs of each orbit.
const ORBITS_TO_MERGE: f64 = 256.0;

/// Below this many possible ends for one first point, every run is scored.
const LONG_RANGE: usize = 64;
/// Ends to a block, at each level.
const BLOCK: usize = 32;

/// Upper bounds on the score of the runs from one first point to a block of
/// ends, so that blocks that cannot beat the best run so far are skipped and
/// the result is the same as scoring every run.
///
/// With `F` and `V` the running totals of centred flux and weight, and `S`
/// and `W` the orbit's, a run `i..j` has `u = (F_j - F_i) - S (V_j - V_i) /
/// W = g_j - g_i`, with `g = F - (S / W) V`, and its score is `u^2 / D`,
/// `D = W_in (W - W_in) / W`, when `u < 0` (times `W / (2 W_out)` for
/// astropy's likelihood). Over a block of ends, `u >= min g - g_i` and
/// `W_in` lies between its values at the block's first and last end; `D`
/// is concave in `W_in`, so its least is at one of the two. The bound adds
/// margins far above the rounding of `g`, `W_in` and the statistic, so it
/// is never below a score the statistic computes.
struct Bounds {
    g: Vec<f64>,
    weight: Vec<f64>,
    /// `mins[level][k]`: the least `g` over ends `k BLOCK^(level+1) ..`.
    mins: Vec<Vec<f64>>,
    /// Margin on `u`, and on `W_in`.
    margin_u: f64,
    margin_w: f64,
    whole: f64,
    likelihood: bool,
}

impl Bounds {
    fn new(totals: &[Total], n: usize, orbit: &Orbit) -> Self {
        let whole = orbit.weight();
        let ratio = orbit.flux() / whole;
        let ends = &totals[..2 * n];
        let weight: Vec<f64> = ends.iter().map(|t| t.weight.value()).collect();
        let flux: Vec<f64> = ends.iter().map(|t| t.flux.value()).collect();
        let g: Vec<f64> = flux
            .iter()
            .zip(&weight)
            .map(|(f, v)| f - ratio * v)
            .collect();
        let largest_flux = flux.iter().fold(0.0_f64, |a, f| a.max(f.abs()));
        let largest_weight = weight.last().copied().unwrap_or(0.0);
        let mut mins = vec![block_mins(&g)];
        while mins.last().is_some_and(|m| m.len() > BLOCK) {
            let next = block_mins(mins.last().expect("a level"));
            mins.push(next);
        }
        Self {
            g,
            weight,
            mins,
            margin_u: 64.0
                * f64::EPSILON
                * (largest_flux + ratio.abs() * largest_weight + orbit.flux().abs()),
            margin_w: 64.0 * f64::EPSILON * largest_weight,
            whole,
            likelihood: orbit.likelihood(),
        }
    }

    /// The runs from `i` to each of `ends`, as [`Folded::runs`] scores them,
    /// skipping blocks whose bound is at most the best score so far.
    fn search(
        &self,
        folded: &Folded,
        orbit: &Orbit,
        i: usize,
        ends: RangeInclusive<usize>,
        best: &mut Option<(f64, usize, usize)>,
    ) -> Result<(), BlsError> {
        let top = self.mins.len() - 1;
        let size = BLOCK.pow(u32::try_from(top + 1).expect("few levels"));
        for k in ends.start() / size..=ends.end() / size {
            self.visit(folded, orbit, i, &ends, top, k, best)?;
        }
        Ok(())
    }

    #[expect(clippy::too_many_arguments, reason = "the search's state")]
    fn visit(
        &self,
        folded: &Folded,
        orbit: &Orbit,
        i: usize,
        ends: &RangeInclusive<usize>,
        level: usize,
        k: usize,
        best: &mut Option<(f64, usize, usize)>,
    ) -> Result<(), BlsError> {
        let size = BLOCK.pow(u32::try_from(level + 1).expect("few levels"));
        let low = (*ends.start()).max(k * size);
        let high = (*ends.end()).min((k + 1) * size - 1);
        if low > high {
            return Ok(());
        }
        let bound = self.bound(i, self.mins[level][k], low, high);
        if bound <= best.map_or(0.0, |b| b.0) {
            return Ok(());
        }
        if level == 0 {
            return folded.runs(orbit, i, low..=high, best);
        }
        for child in k * BLOCK..((k + 1) * BLOCK).min(self.mins[level - 1].len()) {
            self.visit(folded, orbit, i, ends, level - 1, child, best)?;
        }
        Ok(())
    }

    /// An upper bound on the score of the runs from `i` to ends `low..=high`,
    /// `least` the least `g` there. Infinite (never skipped) when the
    /// weight in the box could reach zero or the whole orbit, and NaN on
    /// overflow, which is never skipped either.
    fn bound(&self, i: usize, least: f64, low: usize, high: usize) -> f64 {
        let fall = self.g[i] - least + self.margin_u;
        if fall <= 0.0 {
            return 0.0;
        }
        let w = self.whole;
        let shortest = (self.weight[low] - self.weight[i]) - self.margin_w;
        let longest = (self.weight[high] - self.weight[i]) + self.margin_w;
        if shortest <= 0.0 || longest >= w {
            return f64::INFINITY;
        }
        let d = |x: f64| x * ((w - x) / w);
        let least_d = d(shortest).min(d(longest)) * (1.0 - 64.0 * f64::EPSILON);
        let mut bound = fall * (fall / least_d);
        if self.likelihood {
            bound *= 0.5 * w / (w - longest);
        }
        bound * (1.0 + 64.0 * f64::EPSILON)
    }
}

/// The least of each block of `BLOCK` values.
fn block_mins(values: &[f64]) -> Vec<f64> {
    values
        .chunks(BLOCK)
        .map(|c| c.iter().copied().fold(f64::INFINITY, f64::min))
        .collect()
}
