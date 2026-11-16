//! The search over a grid of periods: the binned search at every period, in
//! parallel, then the highest peaks refined exactly.

use std::num::NonZeroUsize;
use std::thread;

use rustellar_core::{Epoch, LightCurve, TimeDelta};

use crate::{
    BestBox, BlsError, BoxFit, Durations, Periods, TransitBox,
    binned::{Scratch, best_binned, bin_count},
    box_fit::Weighted,
    grid::check_durations_fit,
    parallel::on_threads,
    refine::{peaks, refine},
};

/// The most phase bins a search may use at one period: each takes about 90
/// bytes on each thread.
const MAX_BINS: f64 = 1e6;

/// What a search maximises, and reports as its power.
///
/// `delta_chi2` and `snr` rank boxes the same way (`delta_chi2 = snr^2`);
/// astropy's likelihood, `delta_chi2 W / (2 W_out)`, favours boxes that hold
/// more of the weight. See [`BoxFit`] for each.
///
/// # Examples
///
/// ```
/// use rustellar_bls::Objective;
///
/// assert_eq!(Objective::default(), Objective::DeltaChi2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Objective {
    /// The fall in chi-squared from a flat line to the box, as in Kovacs,
    /// Zucker & Mazeh (2002).
    #[default]
    DeltaChi2,
    /// The depth over its uncertainty.
    Snr,
    /// The log-likelihood that astropy's `BoxLeastSquares` reports as its
    /// power.
    AstropyLikelihood,
}

impl Objective {
    /// This objective's value for a fit.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::{Objective, TransitBox};
    /// use rustellar_core::{Epoch, LightCurve, TimeDelta};
    ///
    /// let lc = LightCurve::from_btjd(
    ///     vec![0.0, 1.0, 2.0, 3.0],
    ///     vec![1.0, 0.99, 1.0, 1.0],
    ///     vec![1e-3; 4],
    /// )?;
    /// let transit = TransitBox::new(
    ///     TimeDelta::from_days(10.0)?,
    ///     TimeDelta::from_days(0.5)?,
    ///     Epoch::from_btjd(1.0)?,
    /// )?;
    /// let fit = transit.fit(&lc)?;
    /// assert_eq!(Objective::Snr.of(&fit), fit.snr);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn of(self, fit: &BoxFit) -> f64 {
        match self {
            Self::DeltaChi2 => fit.delta_chi2,
            Self::Snr => fit.snr,
            Self::AstropyLikelihood => fit.likelihood,
        }
    }
}

/// How to run a search.
///
/// Start from [`SearchOptions::default`] and change what you need with the
/// `with_` methods; new settings may be added in minor releases.
///
/// # Examples
///
/// ```
/// use rustellar_bls::{Objective, SearchOptions};
/// use std::num::NonZeroUsize;
///
/// let options = SearchOptions::default()
///     .with_objective(Objective::Snr)
///     .with_refine_peaks(3)
///     .with_threads(NonZeroUsize::MIN);
/// assert_eq!(options.oversample.get(), 10);
/// assert_eq!(options.refine_peaks, 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SearchOptions {
    /// The phase bins per shortest duration: a bin is at most
    /// `min(durations) / oversample` wide. Default 10.
    pub oversample: NonZeroUsize,
    /// What the search maximises. Default [`Objective::DeltaChi2`].
    pub objective: Objective,
    /// How many of the highest peaks of the power to refine exactly. Default
    /// 5; 0 turns refinement off.
    pub refine_peaks: usize,
    /// How many periods either side of each peak to refine at. Default 2.
    pub refine_reach: usize,
    /// How many threads the binned search runs on. Default all available
    /// cores. The results are the same, to the bit, whatever the number.
    pub threads: NonZeroUsize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            oversample: NonZeroUsize::new(10).expect("not zero"),
            objective: Objective::DeltaChi2,
            refine_peaks: 5,
            refine_reach: 2,
            threads: thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
        }
    }
}

impl SearchOptions {
    /// With `oversample` phase bins per shortest duration.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::SearchOptions;
    /// use std::num::NonZeroUsize;
    ///
    /// let options = SearchOptions::default().with_oversample(NonZeroUsize::new(5).unwrap());
    /// assert_eq!(options.oversample.get(), 5);
    /// ```
    #[must_use]
    pub fn with_oversample(mut self, oversample: NonZeroUsize) -> Self {
        self.oversample = oversample;
        self
    }

    /// With `objective` maximised.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::{Objective, SearchOptions};
    ///
    /// let options = SearchOptions::default().with_objective(Objective::AstropyLikelihood);
    /// assert_eq!(options.objective, Objective::AstropyLikelihood);
    /// ```
    #[must_use]
    pub fn with_objective(mut self, objective: Objective) -> Self {
        self.objective = objective;
        self
    }

    /// With the `count` highest peaks refined; 0 turns refinement off.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::SearchOptions;
    ///
    /// let options = SearchOptions::default().with_refine_peaks(0);
    /// assert_eq!(options.refine_peaks, 0);
    /// ```
    #[must_use]
    pub fn with_refine_peaks(mut self, count: usize) -> Self {
        self.refine_peaks = count;
        self
    }

    /// With each peak refined at `reach` periods either side.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::SearchOptions;
    ///
    /// let options = SearchOptions::default().with_refine_reach(4);
    /// assert_eq!(options.refine_reach, 4);
    /// ```
    #[must_use]
    pub fn with_refine_reach(mut self, reach: usize) -> Self {
        self.refine_reach = reach;
        self
    }

    /// With the binned search on `threads` threads.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::SearchOptions;
    /// use std::num::NonZeroUsize;
    ///
    /// let options = SearchOptions::default().with_threads(NonZeroUsize::MIN);
    /// assert_eq!(options.threads.get(), 1);
    /// ```
    #[must_use]
    pub fn with_threads(mut self, threads: NonZeroUsize) -> Self {
        self.threads = threads;
        self
    }
}

/// A peak of the power, refined exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Peak {
    /// The index of the peak's period in the grid.
    pub peak_index: usize,
    /// The index of the period on the grid, within the refinement's reach
    /// of the peak, that holds the best box there; the box's own period is
    /// within half a step of it.
    pub period_index: usize,
    /// The best box near the peak, over every mid-transit time and every
    /// duration between the neighbours on the grid of the best duration
    /// there; at the best period on the grid, then refined between it and
    /// halfway to its neighbours.
    pub best: BestBox,
    /// The objective of the best box at the grid's periods, before the
    /// period is refined: the peaks are compared by this. The refinement
    /// tries more periods at some peaks than at others, so comparing the
    /// refined boxes would favour those.
    pub grid_objective: f64,
}

/// The result of a search: the power and best box at each period, indexed
/// like the periods, and the refined peaks.
///
/// # Examples
///
/// See [`Bls::search`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BlsResult {
    objective: Objective,
    power: Vec<f64>,
    boxes: Vec<Option<BestBox>>,
    peaks: Vec<Peak>,
}

impl BlsResult {
    /// The objective's value for the best binned box at each period, or 0 at
    /// a period with no box with a dip.
    #[must_use]
    pub fn power(&self) -> &[f64] {
        &self.power
    }

    /// The best binned box at each period, or `None` where no box has a dip.
    ///
    /// Each box is a whole number of phase bins long and starts on a bin
    /// edge, so its duration is the searched one rounded to the bins. Its
    /// mid-transit time is within one period after the light curve's first
    /// time.
    #[must_use]
    pub fn boxes(&self) -> &[Option<BestBox>] {
        &self.boxes
    }

    /// The refined peaks, highest binned power first. Peaks with no box with
    /// a dip are left out.
    #[must_use]
    pub fn peaks(&self) -> &[Peak] {
        &self.peaks
    }

    /// The best box found: the refined box of the peak with the highest
    /// [`grid_objective`](Peak::grid_objective), or, with no refined peaks,
    /// the binned box at the highest power (the first of equal ones). `None`
    /// if no box has a dip.
    #[must_use]
    pub fn best(&self) -> Option<&BestBox> {
        if self.peaks.is_empty() {
            first_best(self.objective, self.boxes.iter().flatten())
        } else {
            let first = self
                .peaks
                .iter()
                .fold(None::<&Peak>, |best, next| match best {
                    Some(b) if next.grid_objective <= b.grid_objective => Some(b),
                    _ => Some(next),
                });
            first.map(|p| &p.best)
        }
    }
}

/// A bin's cost in the binned search, in points: the totals and a score at
/// each of a dozen durations, against binning one point (measured).
const COST_A_BIN: f64 = 8.0;

/// The longest of `periods`, in days.
fn longest(periods: &[TimeDelta]) -> f64 {
    periods.iter().map(|p| p.as_days()).fold(0.0, f64::max)
}

/// The box with the highest objective, the first of equal ones.
fn first_best<'a>(
    objective: Objective,
    boxes: impl Iterator<Item = &'a BestBox>,
) -> Option<&'a BestBox> {
    boxes.fold(None, |best, next| match best {
        Some(b) if objective.of(&next.fit) <= objective.of(&b.fit) => Some(b),
        _ => Some(next),
    })
}

/// A light curve ready to search: its times and weights, computed once.
///
/// # Examples
///
/// ```
/// use rustellar_bls::{Bls, Durations, Periods, SearchOptions};
/// use rustellar_core::LightCurve;
///
/// // Every 30 minutes for 20 days, with a dip of 0.005 lasting 0.15 days
/// // every 3 days.
/// let time: Vec<f64> = (0..960).map(|i| f64::from(i) / 48.0).collect();
/// let flux = time
///     .iter()
///     .map(|t| if (t % 3.0 - 1.0).abs() < 0.075 { 0.995 } else { 1.0 })
///     .collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 960])?;
///
/// let periods: Vec<f64> = (0..200).map(|k| 2.0 + 0.01 * f64::from(k)).collect();
/// let result = Bls::new(&lc)?.search(
///     &Periods::from_days(&periods)?,
///     &Durations::from_days(&[0.1, 0.15, 0.2])?,
///     &SearchOptions::default(),
/// )?;
/// assert_eq!(result.power().len(), 200);
/// let best = result.best().expect("there is a dip");
/// assert!((best.transit.period().as_days() - 3.0).abs() < 0.006);
/// assert!((best.fit.depth - 0.005).abs() < 1e-4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Bls {
    first: Epoch,
    times: Vec<f64>,
    weighted: Weighted,
}

impl Bls {
    /// The light curve's times and weights `1 / flux_err^2`.
    ///
    /// # Errors
    ///
    /// [`BlsError::OutOfRange`] if a weight is too large or too small for an
    /// `f64` to hold to full precision, or the weighted flux overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::{Bls, BlsError};
    /// use rustellar_core::LightCurve;
    ///
    /// let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![1.0, 1.0], vec![1e-200, 1e-3])?;
    /// assert_eq!(Bls::new(&lc).unwrap_err(), BlsError::OutOfRange);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(lc: &LightCurve) -> Result<Self, BlsError> {
        Ok(Self {
            first: lc.time()[0],
            times: lc.time().iter().map(|t| t.as_btjd()).collect(),
            weighted: Weighted::new(lc)?,
        })
    }

    /// The best box at each period over the durations, binned, and the
    /// highest peaks refined exactly.
    ///
    /// At each period the light curve is folded into phase bins that tile
    /// the orbit, at most `min(durations) / oversample` wide, and every box a
    /// whole number of bins long, the nearest to each duration, is tried at
    /// every bin. Then each of the `refine_peaks` highest local maxima of the
    /// power is searched again exactly, at its period and the `refine_reach`
    /// periods either side: over every mid-transit time and every duration,
    /// then over every duration between the best one's neighbours on the
    /// grid, then at periods between the best one and halfway to its
    /// neighbours, ever closer, until a step moves the last point by less
    /// than the time between points. The power itself stays binned.
    ///
    /// # Errors
    ///
    /// - [`BlsError::DurationNotShorterThanPeriod`] if the longest duration
    ///   is not shorter than the shortest period.
    /// - [`BlsError::TooManyBins`] if the longest period would need more than
    ///   1 million phase bins.
    /// - [`BlsError::OutOfRange`] if a box's statistic overflows.
    ///
    /// # Examples
    ///
    /// See [`Bls`].
    pub fn search(
        &self,
        periods: &Periods,
        durations: &Durations,
        options: &SearchOptions,
    ) -> Result<BlsResult, BlsError> {
        let mut result = BlsResult::default();
        self.search_into(periods, durations, options, &mut result)?;
        Ok(result)
    }

    /// As [`Bls::search`], into `result`, reusing its memory: run with a
    /// result of the same size again, the search allocates the same few
    /// buffers whatever the number of periods.
    ///
    /// On an error `result` holds no meaningful values.
    ///
    /// # Errors
    ///
    /// As [`Bls::search`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::{Bls, BlsResult, Durations, Periods, SearchOptions};
    /// use rustellar_core::LightCurve;
    ///
    /// let time: Vec<f64> = (0..100).map(|i| 0.1 * f64::from(i)).collect();
    /// let lc = LightCurve::from_btjd(time, vec![1.0; 100], vec![1e-3; 100])?;
    /// let bls = Bls::new(&lc)?;
    /// let durations = Durations::from_days(&[0.2])?;
    /// let options = SearchOptions::default();
    ///
    /// let mut result = BlsResult::default();
    /// for start in [1.0, 2.0] {
    ///     let periods = Periods::from_days(&[start, start + 0.5])?;
    ///     bls.search_into(&periods, &durations, &options, &mut result)?;
    ///     // Flat flux: no box has a dip.
    ///     assert_eq!(result.power(), [0.0, 0.0]);
    ///     assert_eq!(result.best(), None);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_into(
        &self,
        periods: &Periods,
        durations: &Durations,
        options: &SearchOptions,
        result: &mut BlsResult,
    ) -> Result<(), BlsError> {
        check_durations_fit(periods, durations)?;
        let durations: Vec<f64> = durations.as_slice().iter().map(|d| d.as_days()).collect();
        let oversample = options.oversample.get();
        if bin_count(longest(periods.as_slice()), &durations, oversample) > MAX_BINS {
            return Err(BlsError::TooManyBins);
        }

        let periods = periods.as_slice();
        let n = periods.len();
        result.objective = options.objective;
        result.power.clear();
        result.power.resize(n, 0.0);
        result.boxes.clear();
        result.boxes.resize(n, None);
        result.peaks.clear();

        // Contiguous chunks of periods, one a thread, of about equal cost:
        // a period costs a pass over the points and a few over its bins
        // (`COST_A_BIN` points' worth each), so a chunk of the longest
        // periods holds fewer. The first error in period order: each chunk
        // stops at its first.
        let threads = options.threads.get().min(n);
        let points = self.times.len() as f64;
        let cost: Vec<f64> = periods
            .iter()
            .map(|p| points + COST_A_BIN * bin_count(p.as_days(), &durations, oversample))
            .collect();
        let total: f64 = cost.iter().sum();
        let mut ends = Vec::with_capacity(threads);
        let mut sum = 0.0;
        for (k, c) in cost.iter().enumerate() {
            sum += c;
            // The chunk ends where its share is reached, the last at n.
            if sum >= total * (ends.len() + 1) as f64 / threads as f64 && ends.len() + 1 < threads {
                ends.push(k + 1);
            }
        }
        ends.push(n);
        let mut chunks = Vec::with_capacity(threads);
        let (mut rest, mut power, mut boxes) =
            (periods, &mut result.power[..], &mut result.boxes[..]);
        let mut from = 0;
        for &end in &ends {
            let (p, p_rest) = rest.split_at(end - from);
            let (w, w_rest) = power.split_at_mut(end - from);
            let (b, b_rest) = boxes.split_at_mut(end - from);
            chunks.push(((p, w), b));
            (rest, power, boxes) = (p_rest, w_rest, b_rest);
            from = end;
        }
        let chunks = chunks.into_iter().filter(|((p, _), _)| !p.is_empty());
        on_threads(chunks, &|((periods, power), boxes)| {
            self.binned_chunk(
                periods,
                &durations,
                oversample,
                options.objective,
                power,
                boxes,
            )
        })
        .into_iter()
        .collect::<Result<(), BlsError>>()?;

        let top = peaks(&result.power, options.refine_peaks);
        if top.is_empty() {
            return Ok(());
        }
        let days: Vec<f64> = periods.iter().map(|p| p.as_days()).collect();
        let refined = refine(
            &self.times,
            &self.weighted,
            &days,
            &durations,
            &top,
            options.refine_reach,
            options.objective,
            options.threads.get(),
        )?;
        result.peaks.extend(refined.into_iter().map(|r| Peak {
            peak_index: r.peak,
            period_index: r.period,
            grid_objective: r.grid,
            best: self.best_box(
                TimeDelta::from_days(r.days).expect("a period"),
                r.duration,
                r.centre,
                r.fit,
            ),
        }));
        Ok(())
    }

    /// The binned search at each of `periods`, into `power` and `boxes`,
    /// stopping at the first error.
    fn binned_chunk(
        &self,
        periods: &[TimeDelta],
        durations: &[f64],
        oversample: usize,
        objective: Objective,
        power: &mut [f64],
        boxes: &mut [Option<BestBox>],
    ) -> Result<(), BlsError> {
        // Room for this chunk's longest period, so no period grows it.
        let most = bin_count(longest(periods), durations, oversample) as usize;
        let mut scratch = Scratch::with_capacity(most, durations.len());
        for ((&period, power), slot) in periods.iter().zip(power).zip(boxes) {
            let days = period.as_days();
            let found = best_binned(
                &self.weighted,
                days,
                durations,
                oversample,
                objective,
                &mut scratch,
            )?;
            if let Some(found) = found {
                *power = objective.of(&found.fit);
                *slot = Some(self.best_box(
                    period,
                    found.bin * found.bins as f64,
                    found.centre,
                    found.fit,
                ));
            }
        }
        Ok(())
    }

    /// The box of `period` and `duration` days centred `centre` days after
    /// the first time, moved by whole periods to within one period after it.
    fn best_box(&self, period: TimeDelta, duration: f64, centre: f64, fit: BoxFit) -> BestBox {
        let centre = centre.rem_euclid(period.as_days());
        BestBox {
            transit: TransitBox::new(
                period,
                TimeDelta::from_days(duration).expect("finite"),
                self.first + TimeDelta::from_days(centre).expect("finite"),
            )
            .expect("shorter than the period"),
            fit,
        }
    }
}
