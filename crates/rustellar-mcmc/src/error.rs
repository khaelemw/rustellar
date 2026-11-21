use std::fmt;

/// Why an ensemble could not be set up or run.
///
/// New variants may be added in minor releases, so a `match` on this type
/// needs a wildcard arm.
///
/// # Examples
///
/// ```
/// use rustellar_mcmc::McmcError;
///
/// let err = McmcError::TooFewWalkers { needed: 6 };
/// assert_eq!(err.to_string(), "at least 6 walkers are needed");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum McmcError {
    /// The walkers have no dimensions.
    NoDimensions,
    /// Fewer walkers than twice the dimensions, an even number: the
    /// ensemble moves in two halves, each needing more walkers than
    /// dimensions to span the space.
    TooFewWalkers {
        /// The fewest walkers needed.
        needed: usize,
    },
    /// An odd number of walkers, which cannot be split in two halves.
    OddWalkers,
    /// A walker's length differs from the first walker's.
    DimensionMismatch {
        /// The walker, counting from 0.
        walker: usize,
    },
    /// A walker's starting point holds a NaN or an infinity.
    NonFinite {
        /// The walker, counting from 0.
        walker: usize,
    },
    /// The walkers lie in a subspace (on a line in a plane, say), which an
    /// ensemble's moves never leave.
    Degenerate,
    /// A move's setting or weight is out of range, or no move is given.
    InvalidMove {
        /// Which setting, such as `"a"`.
        name: &'static str,
    },
    /// A walker starts where the log-probability is not finite (outside
    /// the prior, or NaN).
    OutsideSupport {
        /// The walker, counting from 0.
        walker: usize,
    },
    /// A prior's setting is out of range, or it is on a coordinate not
    /// sampled.
    InvalidPrior {
        /// Which prior or coordinate, such as `"normal"` or `"q1"`.
        name: &'static str,
    },
    /// A fit has no covariance to start walkers from (it could not be
    /// inverted), or one over other parameters than the posterior's.
    NoCovariance,
}

impl fmt::Display for McmcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDimensions => write!(f, "the walkers must have at least one dimension"),
            Self::TooFewWalkers { needed } => write!(f, "at least {needed} walkers are needed"),
            Self::OddWalkers => write!(f, "the number of walkers must be even"),
            Self::DimensionMismatch { walker } => {
                write!(f, "walker {walker} has a different length from walker 0")
            }
            Self::NonFinite { walker } => write!(f, "walker {walker} must be finite"),
            Self::Degenerate => write!(f, "the walkers must not lie in a subspace"),
            Self::InvalidMove { name } => write!(f, "the move's {name} is out of range"),
            Self::InvalidPrior { name } => write!(f, "the {name} prior is out of range"),
            Self::NoCovariance => write!(f, "the fit has no covariance over these parameters"),
            Self::OutsideSupport { walker } => {
                write!(
                    f,
                    "walker {walker} starts where the log-probability is not finite"
                )
            }
        }
    }
}

impl std::error::Error for McmcError {}
