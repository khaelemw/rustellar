//! Compensated summation (Neumaier 1974), so that sums over a million points
//! keep the precision of a single addition.
//!
//! Plain addition rounds every partial sum, and the errors build up with the
//! number of terms. Neumaier's method keeps the rounding error of each
//! addition in a second number and adds it back at the end, so the result is
//! about as accurate as rounding the exact sum once, unless the terms cancel
//! almost completely.

/// A running sum and the rounding error lost so far.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Sum {
    sum: f64,
    compensation: f64,
}

impl Sum {
    /// Adds a term. The rounding error of `sum + x` is exactly
    /// `(sum - t) + x` when `sum` is the larger, and `(x - t) + sum`
    /// otherwise (Knuth's and Dekker's error-free transformation).
    pub(crate) fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.compensation += (self.sum - t) + x;
        } else {
            self.compensation += (x - t) + self.sum;
        }
        self.sum = t;
    }

    /// The sum, with the lost rounding error added back. An overflow gives
    /// the infinite (or NaN) plain sum, as plain addition would.
    pub(crate) fn value(self) -> f64 {
        if self.sum.is_finite() {
            self.sum + self.compensation
        } else {
            self.sum
        }
    }
}

impl FromIterator<f64> for Sum {
    fn from_iter<I: IntoIterator<Item = f64>>(terms: I) -> Self {
        let mut sum = Self::default();
        for x in terms {
            sum.add(x);
        }
        sum
    }
}
