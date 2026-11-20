//! The box search on a light curve whitened by an autoregressive model:
//! the matched filter for a box in that model's noise.

use rustellar_core::{Periods, TimeDelta};

use crate::{Bls, BlsError, BlsResult, BoxFit, Durations, Objective, SearchOptions};

impl Bls {
    /// The search for boxes on a light curve whitened and filtered back by
    /// an autoregressive model (`rustellar-detrend`'s `Whitened::matched`),
    /// with `information` how many times its points' information a box of
    /// each duration holds (its `Matched::information`): the matched filter
    /// for a box in the model's noise.
    ///
    /// A box sums the filtered fluxes as in [`Bls::search`], but in
    /// correlated noise a transit's ingress and egress carry more of its
    /// signal than its middle, so a box holds more information than its
    /// points. Each duration is searched on its own and each box's
    /// statistics divided by `information` of its own duration (a binned
    /// box is whole phase bins long, a refined one may be another duration
    /// near it): `delta_chi2`, the likelihood and the depth by it, `snr`
    /// and the depth's error by its square root. Then the best duration is
    /// kept at each period, and the peaks refined at each duration kept,
    /// the highest first. With `information` 1 everywhere this is
    /// [`Bls::search`] one duration at a time.
    ///
    /// # Errors
    ///
    /// As [`Bls::search`], and [`BlsError::OutOfRange`] if `information` is
    /// not above zero (and finite) for a duration searched or found.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_bls::{Bls, Durations, Periods, SearchOptions};
    /// use rustellar_core::{LightCurve, TimeDelta};
    ///
    /// // Every 30 minutes for 20 days, with a dip of 0.005 lasting 0.15
    /// // days every 3 days.
    /// let time: Vec<f64> = (0..960).map(|i| f64::from(i) / 48.0).collect();
    /// let flux = time
    ///     .iter()
    ///     .map(|t| if (t % 3.0 - 1.0).abs() < 0.075 { 0.995 } else { 1.0 })
    ///     .collect();
    /// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 960])?;
    ///
    /// let periods: Vec<f64> = (0..200).map(|k| 2.0 + 0.01 * f64::from(k)).collect();
    /// let periods = Periods::from_days(&periods)?;
    /// let durations = Durations::from_days(&[0.1, 0.15, 0.2])?;
    /// let options = SearchOptions::default();
    /// let bls = Bls::new(&lc)?;
    /// // As in AR(1) noise: a box worth 0.15 days more than its points.
    /// let information = |d: TimeDelta| 1.0 + 0.15 / d.as_days();
    /// let result = bls.search_whitened(&periods, &durations, &options, information)?;
    /// let best = result.best().expect("there is a dip");
    /// assert!((best.transit.period().as_days() - 3.0).abs() < 0.006);
    /// // A box weighs more than its points, so its power is lower.
    /// let plain = bls.search(&periods, &Durations::from_days(&[0.15])?, &options)?;
    /// let k = result.power().iter().position(|&p| p > 0.0).unwrap();
    /// assert!(result.power()[k] <= plain.power()[k]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_whitened<F>(
        &self,
        periods: &Periods,
        durations: &Durations,
        options: &SearchOptions,
        information: F,
    ) -> Result<BlsResult, BlsError>
    where
        F: Fn(TimeDelta) -> f64,
    {
        let mut result = BlsResult::default();
        let mut first = true;
        for &duration in durations.as_slice() {
            factor(&information, duration)?;
            let one = Durations::new(vec![duration])?;
            let mut run = self.search(periods, &one, options)?;
            scale(&mut run, &information)?;
            if first {
                result = run;
                first = false;
                continue;
            }
            for (k, (power, found)) in run.power.iter().zip(run.boxes).enumerate() {
                if *power > result.power[k] {
                    result.power[k] = *power;
                    result.boxes[k] = found;
                }
            }
            result.peaks.extend(run.peaks);
        }
        // The highest first (the earlier duration on a tie), one a period,
        // as many as asked for.
        result
            .peaks
            .sort_by(|a, b| b.grid_objective.total_cmp(&a.grid_objective));
        let mut seen = Vec::new();
        result.peaks.retain(|p| {
            let new = !seen.contains(&p.peak_index);
            seen.push(p.peak_index);
            new
        });
        result.peaks.truncate(options.refine_peaks);
        Ok(result)
    }
}

/// `information` of `duration`, or an error if it is not above zero (and
/// finite).
fn factor(information: &impl Fn(TimeDelta) -> f64, duration: TimeDelta) -> Result<f64, BlsError> {
    let factor = information(duration);
    if factor > 0.0 && factor.is_finite() {
        Ok(factor)
    } else {
        Err(BlsError::OutOfRange)
    }
}

/// `value` of `objective` for a box `factor` times as informative.
fn divided(objective: Objective, value: f64, factor: f64) -> f64 {
    match objective {
        Objective::Snr => value / factor.sqrt(),
        Objective::DeltaChi2 | Objective::AstropyLikelihood => value / factor,
    }
}

/// The run's power, boxes and peaks, each by its own box's duration.
fn scale(run: &mut BlsResult, information: &impl Fn(TimeDelta) -> f64) -> Result<(), BlsError> {
    let objective = run.objective;
    // A period with no box keeps its power (zero).
    let found = run.power.iter_mut().zip(&mut run.boxes);
    for (power, found) in found.filter_map(|(p, b)| b.as_mut().map(|b| (p, b))) {
        let f = factor(information, found.transit.duration())?;
        *power = divided(objective, *power, f);
        found.fit = found.fit.scaled(f);
    }
    for peak in &mut run.peaks {
        let f = factor(information, peak.best.transit.duration())?;
        peak.grid_objective = divided(objective, peak.grid_objective, f);
        peak.best.fit = peak.best.fit.scaled(f);
    }
    Ok(())
}

impl BoxFit {
    /// The fit of a box `factor` times as informative as its points: the
    /// depth and the statistics divided by it, the depth's error and `snr`
    /// by its square root.
    fn scaled(self, factor: f64) -> Self {
        let root = factor.sqrt();
        let depth = self.depth / factor;
        Self {
            flux_in: self.flux_out - depth,
            depth,
            depth_err: self.depth_err / root,
            snr: self.snr / root,
            delta_chi2: self.delta_chi2 / factor,
            likelihood: self.likelihood / factor,
            ..self
        }
    }
}
