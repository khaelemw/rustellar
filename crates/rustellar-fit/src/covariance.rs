//! The fitted parameters' covariance, and estimates with their errors.

use std::fmt;

use crate::Parameter;

/// The covariance of the fitted parameters, `(J^T J)^-1` for the Jacobian
/// `J` of the weighted residuals at the best fit: from the fit's own `R`,
/// as `R^-1 R^-T`, so the conditioning is never squared.
///
/// It takes the flux errors as given. Where they may be off by a common
/// factor, [`scaled`](Self::scaled) by the reduced chi-squared gives the
/// covariance that factor implies.
///
/// See [`FitResult::covariance`](crate::FitResult::covariance) for an
/// example.
#[derive(Debug, Clone, PartialEq)]
pub struct Covariance {
    free: Vec<Parameter>,
    /// Row-major, over `free`.
    matrix: Vec<f64>,
}

impl Covariance {
    /// From `R` (row-major, upper triangular, over `free`), or `None` where
    /// a pivot is too small against the largest for `R` to be inverted.
    pub(crate) fn from_r(free: Vec<Parameter>, r: &[f64]) -> Option<Self> {
        let n = free.len();
        let largest = (0..n).map(|i| r[i * n + i].abs()).fold(0.0, f64::max);
        #[allow(clippy::cast_precision_loss)]
        let floor = n as f64 * f64::EPSILON * largest;
        if (0..n).any(|i| r[i * n + i].abs() <= floor) {
            return None;
        }
        // R^-1, upper triangular, by columns of R X = I.
        let mut inv = vec![0.0; n * n];
        for j in 0..n {
            for i in (0..=j).rev() {
                let start = if i == j { 1.0 } else { 0.0 };
                let sum: f64 = (i + 1..=j).map(|k| r[i * n + k] * inv[k * n + j]).sum();
                inv[i * n + j] = (start - sum) / r[i * n + i];
            }
        }
        // R^-1 R^-T.
        let mut matrix = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let c: f64 = (j..n).map(|k| inv[i * n + k] * inv[j * n + k]).sum();
                matrix[i * n + j] = c;
                matrix[j * n + i] = c;
            }
        }
        Some(Self { free, matrix })
    }

    /// The covariance's parameters, in [`Parameter::ALL`]'s order.
    #[must_use]
    pub fn parameters(&self) -> &[Parameter] {
        &self.free
    }

    fn place(&self, p: Parameter) -> Option<usize> {
        self.free.iter().position(|&q| q == p)
    }

    /// The covariance of `p` and `q`: zero if either was held fixed.
    #[must_use]
    pub fn get(&self, p: Parameter, q: Parameter) -> f64 {
        match (self.place(p), self.place(q)) {
            (Some(i), Some(j)) => self.matrix[i * self.free.len() + j],
            _ => 0.0,
        }
    }

    /// The standard error of `p`: zero if it was held fixed.
    #[must_use]
    pub fn standard_error(&self, p: Parameter) -> f64 {
        self.get(p, p).sqrt()
    }

    /// The correlation of `p` and `q`, from -1 to 1: zero if either was
    /// held fixed.
    #[must_use]
    pub fn correlation(&self, p: Parameter, q: Parameter) -> f64 {
        let scale = self.standard_error(p) * self.standard_error(q);
        if scale > 0.0 {
            self.get(p, q) / scale
        } else {
            0.0
        }
    }

    /// The covariance times `factor`: by the reduced chi-squared, the
    /// covariance with the flux errors scaled to the scatter about the fit.
    #[must_use]
    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            free: self.free.clone(),
            matrix: self.matrix.iter().map(|c| c * factor).collect(),
        }
    }

    /// The variance of a quantity with gradient `gradient` over
    /// [`parameters`](Self::parameters), to first order: `g^T C g`.
    pub(crate) fn propagate(&self, gradient: &[f64]) -> f64 {
        let n = self.free.len();
        (0..n)
            .map(|i| {
                let row: f64 = (0..n).map(|j| self.matrix[i * n + j] * gradient[j]).sum();
                gradient[i] * row
            })
            .sum()
    }

    /// The covariance with the parameter at `index` negated.
    pub(crate) fn flip(&mut self, index: usize) {
        let n = self.free.len();
        for k in 0..n {
            if k != index {
                self.matrix[index * n + k] = -self.matrix[index * n + k];
                self.matrix[k * n + index] = -self.matrix[k * n + index];
            }
        }
    }
}

/// A value with its standard error.
///
/// # Examples
///
/// ```
/// use rustellar_fit::Estimate;
///
/// let e = Estimate::new(0.1, 0.002);
/// assert_eq!((e.value(), e.error()), (0.1, 0.002));
/// assert_eq!(e.to_string(), "0.1 ± 0.002");
/// assert_eq!(format!("{e:.3}"), "0.100 ± 0.002");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    value: f64,
    error: f64,
}

impl Estimate {
    /// A value and its standard error.
    #[must_use]
    pub const fn new(value: f64, error: f64) -> Self {
        Self { value, error }
    }

    /// The value.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// The standard error: infinite where the fit leaves the value
    /// unconstrained.
    #[must_use]
    pub const fn error(self) -> f64 {
        self.error
    }
}

impl fmt::Display for Estimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match f.precision() {
            Some(p) => write!(f, "{:.p$} ± {:.p$}", self.value, self.error),
            None => write!(f, "{} ± {}", self.value, self.error),
        }
    }
}
