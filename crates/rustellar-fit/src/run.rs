//! Running a set-up's fit.

use crate::lm::{self, Residuals, Settings};
use crate::problem::{Problem, STEP, values_of};
use crate::{FitError, FitResult, Parameter, TransitFit};

impl TransitFit<'_> {
    /// Fits the transit: Levenberg-Marquardt from the starting parameters
    /// to the least-squares minimum.
    ///
    /// Where the mid-transit time is free it is reported at the transit
    /// nearest the middle of the data, the epoch least correlated with the
    /// period. The impact parameter is fitted with its sign, since the flux
    /// depends only on its square, and reported as its size.
    ///
    /// # Errors
    ///
    /// Any error from [`check`](Self::check).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_fit::{Parameter, Termination, TransitFit};
    /// use rustellar_model::{
    ///     Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis,
    ///     TimeDelta, TransitParams,
    /// };
    ///
    /// let params = |k: f64| -> Result<TransitParams, Box<dyn std::error::Error>> {
    ///     Ok(TransitParams::new(
    ///         Epoch::from_days(0.5)?,
    ///         TimeDelta::from_days(3.0)?,
    ///         RadiusRatio::new(k)?,
    ///         ScaledSemiMajorAxis::new(10.0)?,
    ///         ImpactParameter::new(0.3)?,
    ///         LimbDarkening::quadratic(0.4, 0.25)?,
    ///     )?)
    /// };
    /// let truth = params(0.1)?;
    /// let time: Vec<f64> = (0..1000).map(|i| 0.001 * f64::from(i)).collect();
    /// let mut flux = Vec::new();
    /// for &t in &time {
    ///     flux.push(truth.flux_at(Epoch::from_days(t)?));
    /// }
    /// let lc = LightCurve::from_days(time, flux, vec![1e-3; 1000])?;
    ///
    /// // The radius ratio from a start 5% off, the rest held.
    /// let mut fit = TransitFit::new(&lc, params(0.105)?);
    /// for p in Parameter::ALL {
    ///     if p != Parameter::RadiusRatio {
    ///         fit = fit.with_fixed(p);
    ///     }
    /// }
    /// let result = fit.run()?;
    /// assert_eq!(result.termination(), Termination::Converged);
    /// assert!((result.value(Parameter::RadiusRatio) - 0.1).abs() < 1e-10);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn run(&self) -> Result<FitResult, FitError> {
        self.check()?;
        let mut values = values_of(self.start, self.baseline);
        let (t0, period) = (values[0], values[1]);
        let time = self.lc.time();
        let (first, last) = time
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), t| {
                (lo.min(t.as_days()), hi.max(t.as_days()))
            });
        if self.is_free(Parameter::MidTime) {
            let shift = ((0.5 * (first + last) - t0) / period).round();
            values[0] = shift.mul_add(period, t0);
        }
        let duration = self.start.t14().map_or(0.01 * period, |t14| t14.as_days());
        let scales = [
            duration,
            duration / ((last - first) / period).max(1.0),
            values[2],
            values[3],
            values[4].max(values[2]),
            1.0,
            1.0,
            values[7],
        ];

        let mut problem = Problem::new(self, values);
        let free = self.free();
        let steps: Vec<f64> = free.iter().map(|p| STEP * scales[p.index()]).collect();
        let mut x0 = problem.free_values();
        // Onto the limb-darkening bounds, where the start is outside them.
        problem.project(&mut x0);
        let settings = Settings {
            max_iterations: self.max_iterations,
            accelerate: self.accelerate,
        };
        let solution = lm::solve(&mut problem, x0, &steps, settings);

        let values = problem.values_at(&solution.x);
        Ok(FitResult::new(values, free, steps, solution))
    }
}
