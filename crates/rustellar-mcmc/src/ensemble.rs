//! The ensemble: its walkers' starting points, moves, seed and threads.

use std::num::NonZeroUsize;

use crate::rng::Rng;
use crate::{LogProbability, McmcError, Move};

/// An ensemble of walkers set up to sample: their starting points (checked
/// once), how they move (the [stretch move](Move::Stretch) by default),
/// the seed of the random numbers and the threads to run on.
///
/// The walkers move in two halves, each by the other's positions, so there
/// must be an even number of them, at least twice the dimensions, and they
/// must not lie in a subspace (which the moves never leave). Start them in
/// a small ball about a good point, such as a least-squares fit.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use rustellar_mcmc::{Ensemble, McmcError};
///
/// let start = vec![vec![0.0], vec![1.0]];
/// let ensemble = Ensemble::new(start)?
///     .with_seed(7)
///     .with_threads(NonZeroUsize::new(2).unwrap());
/// assert_eq!(ensemble.seed(), 7);
///
/// // One walker on a line is not enough.
/// assert_eq!(
///     Ensemble::new(vec![vec![0.0]]).unwrap_err(),
///     McmcError::TooFewWalkers { needed: 2 }
/// );
/// # Ok::<(), McmcError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Ensemble {
    start: Vec<Vec<f64>>,
    moves: Vec<(Move, f64)>,
    seed: u64,
    threads: NonZeroUsize,
}

impl Ensemble {
    /// An ensemble from each walker's starting point, a seed of 0 and one
    /// thread.
    ///
    /// # Errors
    ///
    /// - [`McmcError::NoDimensions`] if the walkers are empty.
    /// - [`McmcError::DimensionMismatch`] if a walker's length differs
    ///   from the first's.
    /// - [`McmcError::NonFinite`] if a starting point holds a NaN or an
    ///   infinity.
    /// - [`McmcError::TooFewWalkers`] with fewer than twice the
    ///   dimensions.
    /// - [`McmcError::OddWalkers`] with an odd number of walkers.
    /// - [`McmcError::Degenerate`] if the walkers lie in a subspace.
    pub fn new(start: Vec<Vec<f64>>) -> Result<Self, McmcError> {
        let dimensions = start.first().map_or(0, Vec::len);
        if dimensions == 0 {
            return Err(McmcError::NoDimensions);
        }
        for (walker, point) in start.iter().enumerate() {
            if point.len() != dimensions {
                return Err(McmcError::DimensionMismatch { walker });
            }
            if !point.iter().all(|v| v.is_finite()) {
                return Err(McmcError::NonFinite { walker });
            }
        }
        if start.len() < 2 * dimensions {
            return Err(McmcError::TooFewWalkers {
                needed: 2 * dimensions,
            });
        }
        if start.len() % 2 != 0 {
            return Err(McmcError::OddWalkers);
        }
        if rank(&start) < dimensions {
            return Err(McmcError::Degenerate);
        }
        Ok(Self {
            start,
            moves: vec![(Move::default(), 1.0)],
            seed: 0,
            threads: NonZeroUsize::MIN,
        })
    }

    /// Every step's walkers moved by `mv`.
    #[must_use]
    pub fn with_move(self, mv: Move) -> Self {
        Self {
            moves: vec![(mv, 1.0)],
            ..self
        }
    }

    /// Each half-step's move drawn from `moves` with probabilities in
    /// proportion to their weights, as emcee's mixtures of moves.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidMove`] if `moves` is empty or a weight is not
    /// finite and above zero.
    pub fn with_moves(self, moves: Vec<(Move, f64)>) -> Result<Self, McmcError> {
        if moves.is_empty() {
            return Err(McmcError::InvalidMove { name: "moves" });
        }
        if moves.iter().any(|&(_, w)| !(w.is_finite() && w > 0.0)) {
            return Err(McmcError::InvalidMove { name: "weight" });
        }
        Ok(Self { moves, ..self })
    }

    /// The moves and their weights.
    #[must_use]
    pub fn moves(&self) -> &[(Move, f64)] {
        &self.moves
    }

    /// The index of a move drawn by weight (no random number drawn for a
    /// single move).
    pub(crate) fn pick(&self, rng: &mut Rng) -> usize {
        if self.moves.len() == 1 {
            return 0;
        }
        let total: f64 = self.moves.iter().map(|&(_, w)| w).sum();
        let mut u = rng.unit() * total;
        // The last, where rounding carries `u` past every weight.
        let last = self.moves.len() - 1;
        self.moves
            .iter()
            .position(|&(_, w)| {
                let hit = u < w;
                u -= w;
                hit
            })
            .unwrap_or(last)
    }

    /// The seed of the random numbers: the same seed gives the same chain.
    #[must_use]
    pub fn with_seed(self, seed: u64) -> Self {
        Self { seed, ..self }
    }

    /// The threads the log-probabilities are worked out on.
    #[must_use]
    pub fn with_threads(self, threads: NonZeroUsize) -> Self {
        Self { threads, ..self }
    }

    /// The walkers' starting points.
    #[must_use]
    pub fn start(&self) -> &[Vec<f64>] {
        &self.start
    }

    /// The number of walkers.
    #[must_use]
    pub fn walkers(&self) -> usize {
        self.start.len()
    }

    /// The number of dimensions.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.start[0].len()
    }

    /// The seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The threads.
    #[must_use]
    pub const fn threads(&self) -> NonZeroUsize {
        self.threads
    }

    /// Whether every walker starts where `target` is finite, as a run
    /// checks first.
    ///
    /// # Errors
    ///
    /// [`McmcError::OutsideSupport`] for the first walker that does not.
    pub fn check<T: LogProbability + ?Sized>(&self, target: &T) -> Result<(), McmcError> {
        match self
            .start
            .iter()
            .position(|x| !target.log_probability(x).is_finite())
        {
            Some(walker) => Err(McmcError::OutsideSupport { walker }),
            None => Ok(()),
        }
    }
}

/// The dimension of the space the points' differences from their mean
/// span: Gram-Schmidt over the differences, each orthogonalised twice and
/// kept only where more than 1e-10 of its length is left.
pub(crate) fn rank(points: &[Vec<f64>]) -> usize {
    let n = points.len() as f64;
    let dimensions = points[0].len();
    let mean: Vec<f64> = (0..dimensions)
        .map(|j| points.iter().map(|p| p[j]).sum::<f64>() / n)
        .collect();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for p in points {
        let mut v: Vec<f64> = p.iter().zip(&mean).map(|(a, m)| a - m).collect();
        let size = norm(&v);
        if size == 0.0 {
            continue;
        }
        // Twice, for orthogonality to rounding ("twice is enough").
        for _ in 0..2 {
            for b in &basis {
                let d: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
                for (x, y) in v.iter_mut().zip(b) {
                    *x -= d * y;
                }
            }
        }
        let rest = norm(&v);
        if rest > 1e-10 * size {
            basis.push(v.iter().map(|x| x / rest).collect());
            if basis.len() == dimensions {
                break;
            }
        }
    }
    basis.len()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
