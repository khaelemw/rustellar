//! Ensemble Markov chain Monte Carlo: samples from a posterior given its
//! log-probability, with the diagnostics that say when the chains can be
//! trusted.
//!
//! The sampler is an ensemble of walkers moved together, as Goodman &
//! Weare's (2010) affine-invariant sampler and emcee (Foreman-Mackey et
//! al. 2013). Any log-probability over a vector of reals can be sampled;
//! nothing is tied to transits or to an instrument.
//!
//! **Status:** in development. So far:
//!
//! - [`LogProbability`]: what is sampled, for any closure
//!   `Fn(&[f64]) -> f64`.
//! - [`Ensemble`]: the walkers' starting points, checked once, a seed and
//!   a number of threads.
//! - [`Ensemble::run`]: the stretch move ([`Move`]) in two halves,
//!   reproducible from the seed on any number of threads, giving a
//!   [`Chain`] with its acceptance fractions, means and covariance.
//!
//! # Examples
//!
//! ```
//! use rustellar_mcmc::{Ensemble, LogProbability};
//!
//! // A standard normal in two dimensions.
//! let target = |x: &[f64]| -0.5 * x.iter().map(|v| v * v).sum::<f64>();
//! assert_eq!(target.log_probability(&[0.0, 0.0]), 0.0);
//!
//! // Four walkers spread about the origin.
//! let start = vec![
//!     vec![0.1, 0.0],
//!     vec![0.0, 0.1],
//!     vec![-0.1, 0.0],
//!     vec![0.0, -0.1],
//! ];
//! let ensemble = Ensemble::new(start)?.with_seed(42);
//! assert_eq!(ensemble.dimensions(), 2);
//! assert_eq!(ensemble.walkers(), 4);
//! ensemble.check(&target)?;
//! # Ok::<(), rustellar_mcmc::McmcError>(())
//! ```

mod chain;
mod ensemble;
mod error;
mod moves;
mod probability;
mod rng;
mod run;

#[cfg(test)]
mod tests;

pub use chain::Chain;
pub use ensemble::Ensemble;
pub use error::McmcError;
pub use moves::Move;
pub use probability::LogProbability;
