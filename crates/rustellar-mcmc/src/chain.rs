//! The chains a run produces.

use crate::diagnostics::{self, IntegratedTime};

/// Every walker's position and log-probability after each step of a run,
/// and how often each walker's proposals were accepted.
///
/// # Examples
///
/// ```
/// use rustellar_mcmc::Ensemble;
///
/// let target = |x: &[f64]| -0.5 * x[0] * x[0];
/// let start = (0..8).map(|i| vec![0.1 * f64::from(i) - 0.35]).collect();
/// let chain = Ensemble::new(start)?.with_seed(3).run(&target, 2000)?;
/// assert_eq!((chain.steps(), chain.walkers(), chain.dimensions()), (2000, 8, 1));
/// // Mean 0 and variance 1, after the first 200 steps.
/// assert!(chain.mean(200)[0].abs() < 0.15);
/// assert!((chain.covariance(200)[0] - 1.0).abs() < 0.2);
/// # Ok::<(), rustellar_mcmc::McmcError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    pub(crate) walkers: usize,
    pub(crate) dimensions: usize,
    pub(crate) start: Vec<f64>,
    /// Step by step, walker by walker.
    pub(crate) positions: Vec<f64>,
    pub(crate) log_probabilities: Vec<f64>,
    pub(crate) accepted: Vec<usize>,
    pub(crate) evaluations: usize,
}

impl Chain {
    /// The steps taken.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.log_probabilities.len() / self.walkers
    }

    /// The number of walkers.
    #[must_use]
    pub const fn walkers(&self) -> usize {
        self.walkers
    }

    /// The number of dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The log-probabilities worked out, the starting points' included.
    #[must_use]
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// A walker's position after step `step` (from 0).
    ///
    /// # Panics
    ///
    /// If `step` or `walker` is out of range.
    #[must_use]
    pub fn position(&self, step: usize, walker: usize) -> &[f64] {
        assert!(step < self.steps() && walker < self.walkers, "out of range");
        let at = (step * self.walkers + walker) * self.dimensions;
        &self.positions[at..at + self.dimensions]
    }

    /// A walker's log-probability after step `step`.
    ///
    /// # Panics
    ///
    /// If `step` or `walker` is out of range.
    #[must_use]
    pub fn log_probability(&self, step: usize, walker: usize) -> f64 {
        assert!(step < self.steps() && walker < self.walkers, "out of range");
        self.log_probabilities[step * self.walkers + walker]
    }

    /// One coordinate of one walker, step by step.
    #[must_use]
    pub fn trace(&self, walker: usize, dimension: usize) -> Vec<f64> {
        (0..self.steps())
            .map(|s| self.position(s, walker)[dimension])
            .collect()
    }

    /// The fraction of each walker's proposals accepted.
    #[must_use]
    pub fn acceptance_fractions(&self) -> Vec<f64> {
        #[allow(clippy::cast_precision_loss)]
        let steps = self.steps().max(1) as f64;
        #[allow(clippy::cast_precision_loss)]
        self.accepted.iter().map(|&a| a as f64 / steps).collect()
    }

    /// The fraction of all proposals accepted.
    #[must_use]
    pub fn acceptance_fraction(&self) -> f64 {
        let each = self.acceptance_fractions();
        #[allow(clippy::cast_precision_loss)]
        let n = each.len() as f64;
        each.iter().sum::<f64>() / n
    }

    /// Every walker's position after each step past the first `discard`
    /// (the burn-in), step by step.
    pub fn samples(&self, discard: usize) -> impl Iterator<Item = &[f64]> {
        let from = discard.min(self.steps()) * self.walkers * self.dimensions;
        self.positions[from..].chunks_exact(self.dimensions)
    }

    /// The samples' mean, past the first `discard` steps (NaN with none).
    #[must_use]
    pub fn mean(&self, discard: usize) -> Vec<f64> {
        let mut sum = vec![0.0; self.dimensions];
        let mut n = 0usize;
        for x in self.samples(discard) {
            for (s, v) in sum.iter_mut().zip(x) {
                *s += v;
            }
            n += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        sum.iter().map(|s| s / n as f64).collect()
    }

    /// The samples' covariance past the first `discard` steps, row by row
    /// (`dimensions` by `dimensions`), over `n - 1`.
    #[must_use]
    pub fn covariance(&self, discard: usize) -> Vec<f64> {
        let mean = self.mean(discard);
        let d = self.dimensions;
        let mut cov = vec![0.0; d * d];
        let mut n = 0usize;
        for x in self.samples(discard) {
            for i in 0..d {
                let di = x[i] - mean[i];
                for j in 0..=i {
                    cov[i * d + j] = di.mul_add(x[j] - mean[j], cov[i * d + j]);
                }
            }
            n += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let scale = (n as f64 - 1.0).recip();
        for i in 0..d {
            for j in 0..=i {
                cov[i * d + j] *= scale;
                cov[j * d + i] = cov[i * d + j];
            }
        }
        cov
    }

    /// Each walker's last position (its start, with no steps), to start
    /// another run from.
    #[must_use]
    pub fn last(&self) -> Vec<Vec<f64>> {
        let from = if self.steps() == 0 {
            &self.start[..]
        } else {
            let at = (self.steps() - 1) * self.walkers * self.dimensions;
            &self.positions[at..]
        };
        from.chunks_exact(self.dimensions)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// One parameter's draws past the first `discard` steps, walker by
    /// walker: the chains the [`diagnostics`] take.
    #[must_use]
    pub fn traces(&self, dimension: usize, discard: usize) -> Vec<Vec<f64>> {
        assert!(dimension < self.dimensions, "out of range");
        (0..self.walkers)
            .map(|w| {
                (discard.min(self.steps())..self.steps())
                    .map(|s| self.position(s, w)[dimension])
                    .collect()
            })
            .collect()
    }

    /// Each parameter's integrated autocorrelation time past the first
    /// `discard` steps, as emcee's `get_autocorr_time` (window step 5).
    ///
    /// # Panics
    ///
    /// If no steps are left.
    #[must_use]
    pub fn autocorrelation_times(&self, discard: usize) -> Vec<IntegratedTime> {
        self.per_dimension(discard, |c| diagnostics::integrated_time(c, 5.0))
    }

    /// Each parameter's rank-normalised split-R-hat past the first
    /// `discard` steps, the walkers taken as chains.
    #[must_use]
    pub fn rhat(&self, discard: usize) -> Vec<f64> {
        self.per_dimension(discard, diagnostics::rhat)
    }

    /// Each parameter's bulk effective sample size past the first
    /// `discard` steps, the walkers taken as chains.
    #[must_use]
    pub fn ess_bulk(&self, discard: usize) -> Vec<f64> {
        self.per_dimension(discard, diagnostics::ess_bulk)
    }

    /// Each parameter's tail effective sample size past the first
    /// `discard` steps, the walkers taken as chains.
    #[must_use]
    pub fn ess_tail(&self, discard: usize) -> Vec<f64> {
        self.per_dimension(discard, diagnostics::ess_tail)
    }

    fn per_dimension<T>(&self, discard: usize, f: impl Fn(&[Vec<f64>]) -> T) -> Vec<T> {
        (0..self.dimensions)
            .map(|d| f(&self.traces(d, discard)))
            .collect()
    }
}
