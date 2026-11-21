/// A log-probability to sample: the log of a density known up to a
/// constant, over a vector of reals.
///
/// Return `f64::NEG_INFINITY` where the density is zero (outside a prior's
/// bounds, say); the sampler never moves there. Every closure
/// `Fn(&[f64]) -> f64` is one.
///
/// # Examples
///
/// ```
/// use rustellar_mcmc::LogProbability;
///
/// /// A normal with mean `mean` and unit variance, positive values only.
/// struct HalfNormal {
///     mean: f64,
/// }
///
/// impl LogProbability for HalfNormal {
///     fn log_probability(&self, x: &[f64]) -> f64 {
///         if x[0] < 0.0 {
///             return f64::NEG_INFINITY;
///         }
///         -0.5 * (x[0] - self.mean).powi(2)
///     }
/// }
///
/// let target = HalfNormal { mean: 1.0 };
/// assert_eq!(target.log_probability(&[1.0]), 0.0);
/// assert_eq!(target.log_probability(&[-1.0]), f64::NEG_INFINITY);
/// ```
pub trait LogProbability {
    /// The log-probability at `x`, up to a constant.
    fn log_probability(&self, x: &[f64]) -> f64;
}

impl<F: Fn(&[f64]) -> f64> LogProbability for F {
    fn log_probability(&self, x: &[f64]) -> f64 {
        self(x)
    }
}
