//! Compensated summation (Neumaier 1974), so that sums over a million points
//! keep the precision of a single addition.
//!
//! Plain addition rounds every partial sum, and the errors build up with the
//! number of terms. Neumaier's method keeps the rounding error of each
//! addition in a second number and adds it back at the end, so the result is
//! about as accurate as rounding the exact sum once, unless the terms cancel
//! almost completely. Each error is found with Knuth's two-sum, which needs
//! no branch.

/// A running sum and the rounding error lost so far.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Sum {
    sum: f64,
    compensation: f64,
}

impl Sum {
    /// Adds a term. The rounding error of `t = sum + x` is exactly
    /// `(sum - (t - v)) + (x - v)` with `v = t - sum` (Knuth's two-sum),
    /// whichever term is the larger. Six additions and no branch: a branch
    /// on which is larger, as in Neumaier's form, is mispredicted half the
    /// time on fluxes of either sign.
    pub(crate) fn add(&mut self, x: f64) {
        let t = self.sum + x;
        let v = t - self.sum;
        self.compensation += (self.sum - (t - v)) + (x - v);
        self.sum = t;
    }

    /// A sum from its plain sum and compensation, as made elsewhere.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_parts([sum, compensation]: [f64; 2]) -> Self {
        Self { sum, compensation }
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

/// `later - earlier`, for two running totals over the same terms: the sum of
/// the terms added in between, about as accurate as rounding it once,
/// however large the totals.
///
/// The plain sums' difference is exact when the terms in between are small
/// beside the totals, as the two are then within a factor of two (Sterbenz);
/// otherwise it rounds once at about the size of the result. The difference
/// of the compensations then adds back what the plain sums lost.
pub(crate) fn between(earlier: Sum, later: Sum) -> f64 {
    (later.sum - earlier.sum) + (later.compensation - earlier.compensation)
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
