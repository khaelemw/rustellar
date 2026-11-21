//! What a fit gives back.

use rustellar_model::TransitParams;

use crate::lm::Solution;
use crate::problem::values_of;
use crate::{Parameter, Termination};

/// A finished fit: the best parameters, the chi-squared there, and how the
/// fit got there.
///
/// See [`TransitFit::run`](crate::TransitFit::run) for an example.
#[derive(Debug, Clone)]
pub struct FitResult {
    params: TransitParams,
    baseline: f64,
    free: Vec<Parameter>,
    chi_squared: f64,
    points: usize,
    iterations: usize,
    evaluations: usize,
    termination: Termination,
    /// `R` of the weighted Jacobian at the best fit, over the free
    /// parameters, row-major.
    #[allow(dead_code)]
    r: Vec<f64>,
}

impl FitResult {
    pub(crate) fn new(
        params: TransitParams,
        baseline: f64,
        free: Vec<Parameter>,
        solution: Solution,
    ) -> Self {
        Self {
            params,
            baseline,
            free,
            chi_squared: solution.f.iter().map(|r| r * r).sum(),
            points: solution.f.len(),
            iterations: solution.iterations,
            evaluations: solution.evaluations,
            termination: solution.termination,
            r: solution.r,
        }
    }

    /// The best transit parameters.
    #[must_use]
    pub const fn params(&self) -> TransitParams {
        self.params
    }

    /// The best baseline.
    #[must_use]
    pub const fn baseline(&self) -> f64 {
        self.baseline
    }

    /// One parameter's best value (its starting value if it was fixed).
    #[must_use]
    pub fn value(&self, parameter: Parameter) -> f64 {
        values_of(self.params, self.baseline)[parameter.index()]
    }

    /// The parameters that were fitted, in [`Parameter::ALL`]'s order.
    #[must_use]
    pub fn free(&self) -> &[Parameter] {
        &self.free
    }

    /// The weighted sum of squared residuals at the best fit.
    #[must_use]
    pub const fn chi_squared(&self) -> f64 {
        self.chi_squared
    }

    /// Chi-squared over the degrees of freedom: points less free
    /// parameters.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn reduced_chi_squared(&self) -> f64 {
        self.chi_squared / (self.points - self.free.len()) as f64
    }

    /// The points fitted.
    #[must_use]
    pub const fn points(&self) -> usize {
        self.points
    }

    /// The steps the fit took.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// The model evaluations, the Jacobians' included.
    #[must_use]
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Why the fit stopped.
    #[must_use]
    pub const fn termination(&self) -> Termination {
        self.termination
    }
}
