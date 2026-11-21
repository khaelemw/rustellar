//! How a walker moves: proposals from the other half of the ensemble.

use crate::McmcError;
use crate::rng::Rng;

/// A way of moving a walker using the positions of the other half of the
/// ensemble. Every move keeps the target's distribution. All but the
/// snooker move are affine-invariant: they behave the same on a target
/// stretched, sheared or rotated. The snooker move projects at right
/// angles, so it behaves the same on a target rotated, scaled evenly or
/// shifted, not stretched along one direction.
///
/// # Examples
///
/// ```
/// use rustellar_mcmc::Move;
///
/// assert_eq!(Move::default(), Move::stretch(2.0)?);
/// assert!(Move::stretch(1.0).is_err());
/// let mixture = [
///     (Move::differential_evolution(None, 1e-5)?, 0.8),
///     (Move::snooker(1.7)?, 0.2),
/// ];
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
    /// Goodman & Weare's (2010) walk move: a Gaussian step with the
    /// covariance of `subset` walkers of the other half drawn at random
    /// (all of them when `None`), as emcee's `WalkMove`.
    Walk {
        /// The walkers whose spread sets the step, at least 2; `None` for
        /// the whole other half.
        subset: Option<usize>,
    },
    /// Differential evolution (ter Braak 2006; Nelson et al. 2014): a step
    /// of `gamma (1 + sigma n)` times the difference of two walkers of the
    /// other half, `n` a standard normal, as emcee's `DEMove`.
    DifferentialEvolution {
        /// The step's scale; `None` for `2.38 / sqrt(2 d)`, the optimum
        /// for a Gaussian in `d` dimensions.
        gamma: Option<f64>,
        /// The relative spread of the scale (emcee's default, 1e-5).
        sigma: f64,
    },
    /// ter Braak & Vrugt's (2008) snooker move: along the line through the
    /// walker and another walker `z`, by `gamma` times the difference of
    /// two more walkers' projections on that line, accepted with
    /// probability `min(1, (|y - z| / |x - z|)^(d - 1) p(y) / p(x))`.
    /// Needs six walkers. Not affine-invariant: best mixed with another
    /// move, on parameters of similar scale.
    Snooker {
        /// The step's scale (ter Braak & Vrugt's 1.7).
        gamma: f64,
    },
    /// The ensemble slice sampler (Karamanis & Beutler 2021, as zeus):
    /// slice sampling (Neal 2003) along a direction from the other half
    /// times `mu`, stepped out and shrunk until a point inside the slice
    /// is found. Every step moves; the price is a varying number of
    /// log-probabilities per step.
    Slice {
        /// The initial slice width, in units of the direction.
        mu: f64,
        /// The steps over which `mu` is tuned (towards as many
        /// expansions as contractions, Karamanis & Beutler 2021), then
        /// fixed. Discard at least these steps.
        tune: usize,
        /// How the direction is drawn.
        direction: Direction,
    },
}

/// The direction of a slice move, drawn from the other half of the
/// ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Direction {
    /// The difference of two walkers (zeus's differential move).
    #[default]
    Differential,
    /// A Gaussian with the walkers' covariance (zeus's Gaussian move).
    Gaussian,
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
        Self::Stretch { a }.checked()
    }

    /// The walk move with `subset` walkers (`None` for all of the other
    /// half).
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidMove`] if `subset` is below 2.
    pub fn walk(subset: Option<usize>) -> Result<Self, McmcError> {
        Self::Walk { subset }.checked()
    }

    /// Differential evolution with scale `gamma` (`None` for
    /// `2.38 / sqrt(2 d)`) and relative spread `sigma`.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidMove`] unless `gamma` is finite and positive
    /// and `sigma` finite and not negative.
    pub fn differential_evolution(gamma: Option<f64>, sigma: f64) -> Result<Self, McmcError> {
        Self::DifferentialEvolution { gamma, sigma }.checked()
    }

    /// The snooker move with scale `gamma`.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidMove`] unless `gamma` is finite and positive.
    pub fn snooker(gamma: f64) -> Result<Self, McmcError> {
        Self::Snooker { gamma }.checked()
    }

    /// The slice move along differences of walkers, with initial width
    /// `mu` tuned over the first `tune` steps.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidMove`] unless `mu` is finite and positive.
    pub fn slice(mu: f64, tune: usize) -> Result<Self, McmcError> {
        Self::Slice {
            mu,
            tune,
            direction: Direction::Differential,
        }
        .checked()
    }

    /// The slice move along Gaussian directions with the other half's
    /// covariance, with initial width `mu` tuned over the first `tune`
    /// steps.
    ///
    /// # Errors
    ///
    /// [`McmcError::InvalidMove`] unless `mu` is finite and positive.
    pub fn slice_gaussian(mu: f64, tune: usize) -> Result<Self, McmcError> {
        Self::Slice {
            mu,
            tune,
            direction: Direction::Gaussian,
        }
        .checked()
    }

    /// The move, if its settings are in range.
    pub(crate) fn checked(self) -> Result<Self, McmcError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let bad = match self {
            Self::Stretch { a } => (!(a.is_finite() && a > 1.0)).then_some("a"),
            Self::Walk { subset } => subset.is_some_and(|s| s < 2).then_some("subset"),
            Self::DifferentialEvolution { gamma, sigma } => {
                if !gamma.is_none_or(positive) {
                    Some("gamma")
                } else if !(sigma.is_finite() && sigma >= 0.0) {
                    Some("sigma")
                } else {
                    None
                }
            }
            Self::Snooker { gamma } => (!positive(gamma)).then_some("gamma"),
            Self::Slice { mu, .. } => (!positive(mu)).then_some("mu"),
        };
        bad.map_or(Ok(self), |name| Err(McmcError::InvalidMove { name }))
    }

    /// The fewest walkers in all the move works with.
    pub(crate) fn walkers_needed(&self) -> usize {
        match *self {
            Self::Stretch { .. } => 2,
            Self::Walk { subset } => 2 * subset.unwrap_or(2),
            Self::DifferentialEvolution { .. } | Self::Slice { .. } => 4,
            Self::Snooker { .. } => 6,
        }
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
    /// positions), drawing from `rng`. Not for [`Move::Slice`], which is
    /// not a proposal.
    pub(crate) fn propose(&self, x: &[f64], others: &[&[f64]], rng: &mut Rng) -> Proposal {
        let d = x.len();
        #[allow(clippy::cast_precision_loss)]
        let dims = d as f64;
        match *self {
            Self::Stretch { a } => {
                let c = others[index(rng, others.len())];
                let z = stretch_factor(rng.unit(), a);
                let to = x
                    .iter()
                    .zip(c)
                    .map(|(&xi, &ci)| z.mul_add(xi - ci, ci))
                    .collect();
                Proposal {
                    to,
                    log_ratio: (dims - 1.0) * z.ln(),
                }
            }
            Self::Walk { subset } => {
                let picked = distinct(rng, others.len(), subset.unwrap_or(others.len()));
                let step = gaussian_step(others, &picked, rng);
                let to = x.iter().zip(step).map(|(a, b)| a + b).collect();
                Proposal { to, log_ratio: 0.0 }
            }
            Self::DifferentialEvolution { gamma, sigma } => {
                let pair = distinct(rng, others.len(), 2);
                let (c1, c2) = (others[pair[0]], others[pair[1]]);
                let g0 = gamma.unwrap_or_else(|| 2.38 / (2.0 * dims).sqrt());
                let g = g0 * sigma.mul_add(rng.normal(), 1.0);
                let to = x
                    .iter()
                    .zip(c1.iter().zip(c2))
                    .map(|(&xi, (&a, &b))| g.mul_add(b - a, xi))
                    .collect();
                Proposal { to, log_ratio: 0.0 }
            }
            Self::Snooker { gamma } => {
                let three = distinct(rng, others.len(), 3);
                let (z, z1, z2) = (others[three[0]], others[three[1]], others[three[2]]);
                let delta: Vec<f64> = x.iter().zip(z).map(|(a, b)| a - b).collect();
                let r2 = dot(&delta, &delta);
                if r2 == 0.0 {
                    // The walker on top of `z`: no line, no move.
                    return Proposal {
                        to: x.to_vec(),
                        log_ratio: f64::NEG_INFINITY,
                    };
                }
                let along: f64 = delta
                    .iter()
                    .zip(z1.iter().zip(z2))
                    .map(|(e, (a, b))| e * (a - b))
                    .sum();
                let k = gamma * along / r2;
                let to: Vec<f64> = x
                    .iter()
                    .zip(&delta)
                    .map(|(&xi, &e)| k.mul_add(e, xi))
                    .collect();
                let moved: Vec<f64> = to.iter().zip(z).map(|(a, b)| a - b).collect();
                Proposal {
                    log_ratio: 0.5 * (dims - 1.0) * (dot(&moved, &moved) / r2).ln(),
                    to,
                }
            }
            Self::Slice { .. } => unreachable!("the slice move is not a proposal"),
        }
    }
}

/// A slice move's direction from `others`, times `mu`.
pub(crate) fn slice_direction(
    direction: Direction,
    mu: f64,
    others: &[&[f64]],
    rng: &mut Rng,
) -> Vec<f64> {
    match direction {
        Direction::Differential => {
            let pair = distinct(rng, others.len(), 2);
            let (c1, c2) = (others[pair[0]], others[pair[1]]);
            c1.iter().zip(c2).map(|(a, b)| mu * (a - b)).collect()
        }
        Direction::Gaussian => {
            let all: Vec<usize> = (0..others.len()).collect();
            gaussian_step(others, &all, rng)
                .into_iter()
                .map(|v| mu * v)
                .collect()
        }
    }
}

/// `sum_j n_j (c_j - mean) / sqrt(s - 1)` over the `s` walkers `picked`,
/// the `n_j` standard normals: a Gaussian step with their covariance (over
/// `s - 1`).
fn gaussian_step(others: &[&[f64]], picked: &[usize], rng: &mut Rng) -> Vec<f64> {
    let d = others[0].len();
    #[allow(clippy::cast_precision_loss)]
    let s = picked.len() as f64;
    let mut mean = vec![0.0; d];
    for &j in picked {
        for (m, v) in mean.iter_mut().zip(others[j]) {
            *m += v / s;
        }
    }
    let scale = (s - 1.0).sqrt().recip();
    let mut step = vec![0.0; d];
    for &j in picked {
        let n = rng.normal() * scale;
        for ((t, c), m) in step.iter_mut().zip(others[j]).zip(&mean) {
            *t = n.mul_add(c - m, *t);
        }
    }
    step
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `k` distinct indices below `n`, in random order (a partial
/// Fisher-Yates shuffle).
pub(crate) fn distinct(rng: &mut Rng, n: usize, k: usize) -> Vec<usize> {
    let mut all: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + index(rng, n - i);
        all.swap(i, j);
    }
    all.truncate(k);
    all
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
