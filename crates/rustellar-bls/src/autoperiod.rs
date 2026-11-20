//! astropy's automatic period grid (`BoxLeastSquares.autoperiod`), computed
//! the same way, so the periods are the same to the last bit.

use rustellar_core::{LightCurve, Periods, TimeDelta};

use crate::{BlsError, Durations, grid::positive_days};

/// Settings for astropy's automatic period grid.
///
/// The grid is even in frequency, from `1 / maximum_period` to
/// `1 / minimum_period`, with step
///
/// ```text
/// df = frequency_factor * min(duration) / baseline^2
/// ```
///
/// where the baseline is the time from the light curve's first point to its
/// last. A transit lasts a fraction `duration / period` of the orbit, so its
/// phase drifts out of place over the baseline much faster than a sine wave
/// would: the step is finer than a sine search's `1 / baseline` by the factor
/// `duration / baseline`.
///
/// The defaults are astropy's: no set range, at least 3 transits, and
/// `frequency_factor` 1. Unset, the shortest period is twice the longest
/// duration, and the longest is `baseline / (minimum_n_transit - 1)`, the
/// longest period that still shows `minimum_n_transit` transits.
///
/// Where astropy quietly carries on, this returns an error instead: a
/// minimum period above the maximum (astropy swaps them), a light curve with
/// a single time, and a step so coarse that the grid would reach zero or
/// negative frequencies.
///
/// # Examples
///
/// ```
/// use rustellar_bls::{AutoPeriod, Durations};
/// use rustellar_core::LightCurve;
///
/// let lc = LightCurve::from_days(vec![0.0, 13.5, 27.0], vec![1.0; 3], vec![1e-3; 3])?;
/// let durations = Durations::from_days(&[0.05, 0.1, 0.2])?;
///
/// let periods = AutoPeriod::default().periods(&lc, &durations)?;
/// assert_eq!(periods.as_slice().len(), 35_371);
/// assert_eq!(periods.as_slice()[0].as_days(), 0.4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoPeriod {
    /// The shortest period, or twice the longest duration if `None`.
    pub minimum_period: Option<TimeDelta>,
    /// The longest period, or `baseline / (minimum_n_transit - 1)` if `None`.
    pub maximum_period: Option<TimeDelta>,
    /// The fewest transits a period must show within the baseline, for the
    /// default longest period. At least 2.
    pub minimum_n_transit: u32,
    /// Scales the frequency step: below 1 is finer (more periods), above 1
    /// coarser.
    pub frequency_factor: f64,
}

impl Default for AutoPeriod {
    fn default() -> Self {
        Self {
            minimum_period: None,
            maximum_period: None,
            minimum_n_transit: 3,
            frequency_factor: 1.0,
        }
    }
}

impl AutoPeriod {
    /// The period grid for a light curve and the durations to be searched,
    /// shortest period first.
    ///
    /// The number of periods is about `(1 / minimum_period - 1 /
    /// maximum_period) / df`, which grows as the square of the baseline: about
    /// 35,000 for a 27-day light curve with the default settings and a
    /// shortest duration of 0.05 days.
    ///
    /// # Errors
    ///
    /// In this order:
    ///
    /// - [`BlsError::TooFewTransits`] if `minimum_n_transit` is below 2.
    /// - [`BlsError::NonFinite`] (`name: "frequency_factor"`) or
    ///   [`BlsError::FrequencyFactorNotPositive`] for a bad
    ///   `frequency_factor`.
    /// - [`BlsError::PeriodNotPositive`] if a set `minimum_period` or
    ///   `maximum_period` is zero or negative.
    /// - [`BlsError::NoBaseline`] if the light curve has a single time.
    /// - [`BlsError::PeriodRangeEmpty`] if the shortest period is above the
    ///   longest.
    /// - [`BlsError::GridTooCoarse`] if the step is so large that the last
    ///   frequency in the grid is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::{AutoPeriod, BlsError, Durations};
    /// use rustellar_core::{LightCurve, TimeDelta};
    ///
    /// let lc = LightCurve::from_days(vec![1325.3, 1353.2], vec![1.0; 2], vec![1e-3; 2])?;
    /// let durations = Durations::from_days(&[0.05, 0.1])?;
    ///
    /// // The pipeline's range, 0.5 to 15 days.
    /// let range = AutoPeriod {
    ///     minimum_period: Some(TimeDelta::from_days(0.5)?),
    ///     maximum_period: Some(TimeDelta::from_days(15.0)?),
    ///     ..AutoPeriod::default()
    /// };
    /// let periods = range.periods(&lc, &durations)?;
    /// assert_eq!(periods.min().as_days(), 0.5);
    ///
    /// let reversed = AutoPeriod {
    ///     minimum_period: Some(TimeDelta::from_days(15.0)?),
    ///     maximum_period: Some(TimeDelta::from_days(0.5)?),
    ///     ..AutoPeriod::default()
    /// };
    /// assert_eq!(reversed.periods(&lc, &durations), Err(BlsError::PeriodRangeEmpty));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn periods(&self, lc: &LightCurve, durations: &Durations) -> Result<Periods, BlsError> {
        if self.minimum_n_transit < 2 {
            return Err(BlsError::TooFewTransits);
        }
        if !self.frequency_factor.is_finite() {
            return Err(BlsError::NonFinite {
                name: "frequency_factor",
            });
        }
        if self.frequency_factor <= 0.0 {
            return Err(BlsError::FrequencyFactorNotPositive);
        }
        let set_minimum = self.minimum_period.map(positive_days).transpose()?;
        let set_maximum = self.maximum_period.map(positive_days).transpose()?;

        // Time is strictly increasing, so the baseline is last minus first.
        let time = lc.time();
        let baseline = (time[time.len() - 1] - time[0]).as_days();
        if baseline <= 0.0 {
            return Err(BlsError::NoBaseline);
        }

        // Each expression is written as astropy writes it, so that every
        // rounding is the same.
        let df = self.frequency_factor * durations.min().as_days() / (baseline * baseline);
        let minimum_period = set_minimum.unwrap_or(2.0 * durations.max().as_days());
        let maximum_period =
            set_maximum.unwrap_or(baseline / f64::from(self.minimum_n_transit - 1));
        if minimum_period > maximum_period {
            return Err(BlsError::PeriodRangeEmpty);
        }

        let minimum_frequency = 1.0 / maximum_period;
        let maximum_frequency = 1.0 / minimum_period;

        // numpy's round sends halves to the even neighbour; f64::round would
        // send them away from zero. The value is finite and not negative, so
        // the conversion to an integer is exact, and `k as f64` below is what
        // astropy's `np.arange(n)` gives.
        let steps = ((maximum_frequency - minimum_frequency) / df).round_ties_even() as usize;

        // The frequencies fall from the first to the last, so if the last
        // gives a finite, positive period, they all do.
        let last_frequency = maximum_frequency - df * steps as f64;
        if !(last_frequency > 0.0 && (1.0 / last_frequency).is_finite()) {
            return Err(BlsError::GridTooCoarse);
        }

        let periods = (0..=steps)
            .map(|k| {
                TimeDelta::from_days(1.0 / (maximum_frequency - df * k as f64))
                    .expect("between the first and last periods, both finite")
            })
            .collect();

        Ok(Periods::new(periods).expect("periods between two positive edges"))
    }
}
