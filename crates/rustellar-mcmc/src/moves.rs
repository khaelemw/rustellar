//! How a walker moves: proposals from the other half of the ensemble.

use crate::McmcError;
use crate::rng::Rng;

/// A way of proposing a walker's next position from the positions of the
/// other half of the ensemble.
///
/// # Examples
///
/// ```
/// use rustellar_mcmc::Move;
///
/// assert_eq!(Move::default(), Move::stretch(2.0)?);
/// assert!(Move::stretch(1.0).is_err());
/// # Ok::<(), rustellar_mcmc::McmcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Move {
    /// Goodman & Weare's (2010) stretch move: along the line through the
    /// walker and another walker `c`, to `c + z (x - c)`, with `z` drawn
    /// from `g(z) ∝ 1 / sqrt(z)` on `[1 / a, a]`, accepted with
    /// probability `min(1, z^(d - 1) p(y) / p(x))`. The default, with
    /// `a = 2`.
    Stretch {
        /// The scale `a`, above 1: larger takes longer steps, accepted
        /// less often.
        a: f64,
    },
}

impl Default for Move {
    fn default() -> Self {
        Self::Stretch { a: 2.0 }
    }
}

impl Move {
    /// The stretch move with scale `a`.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidMove`] unless `a` is finite and above 1.
    pub fn stretch(a: f64) -> Result<Self, McmcError> {
        if !(a.is_finite() && a > 1.0) {
            return Err(McmcError::InvalidMove { name: "a" });
        }
        Ok(Self::Stretch { a })
    }
}

/// One walker's proposal, before its log-probability is known: where to,
/// and the log of the proposal's density ratio (the Hastings term) to add
/// to the log-probabilities' difference.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Proposal {
    pub(crate) to: Vec<f64>,
    pub(crate) log_ratio: f64,
}

impl Move {
    /// The proposal for the walker at `x`, from `others` (the other half's
    /// positions), drawing from `rng`.
    pub(crate) fn propose(&self, x: &[f64], others: &[&[f64]], rng: &mut Rng) -> Proposal {
        match *self {
            Self::Stretch { a } => {
                let c = others[index(rng, others.len())];
                let z = stretch_factor(rng.unit(), a);
                let to = x
                    .iter()
                    .zip(c)
                    .map(|(&xi, &ci)| z.mul_add(xi - ci, ci))
                    .collect();
                #[allow(clippy::cast_precision_loss)]
                let log_ratio = (x.len() - 1) as f64 * z.ln();
                Proposal { to, log_ratio }
            }
        }
    }
}

/// `z` from the uniform `u` in `[0, 1)`: the inverse of `g`'s cumulative
/// distribution, `sqrt(z)` uniform on `[1 / sqrt(a), sqrt(a)]`.
pub(crate) fn stretch_factor(u: f64, a: f64) -> f64 {
    let root = (a - 1.0).mul_add(u, 1.0);
    root * root / a
}

/// A uniform index below `n` (`n` at most 2^32): the high half of the
/// product of 64 random bits and `n`, biased by under `n / 2^64`.
pub(crate) fn index(rng: &mut Rng, n: usize) -> usize {
    #[allow(clippy::cast_possible_truncation)]
    let i = ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize;
    i
}
