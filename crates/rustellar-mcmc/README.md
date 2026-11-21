# rustellar-mcmc

Ensemble Markov chain Monte Carlo in pure Rust: samples from a posterior
given its log-probability, the diagnostics that say when the chains can be
trusted, and transit posteriors built on `rustellar-model` and
`rustellar-fit`.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `mcmc` feature.

```toml
[dependencies]
# Through the umbrella, which most projects want:
rustellar = { version = "0.1", features = ["mcmc"] }
# Or this crate alone:
rustellar-mcmc = "0.1"
```

**Status:** in development, not yet released.

## Statement of need

A least-squares fit gives the best parameters and their errors to first
order. A transit's parameters are often far from that: a/R*, b and k trade
along a curved valley, b is bounded at zero, limb darkening is bounded by
physics, and grazing or faint transits have posteriors nothing like a
Gaussian. Sampling the posterior gives the real uncertainties and their
correlations. This crate will sample any log-probability with an ensemble
of walkers (Goodman & Weare 2010; Foreman-Mackey et al. 2013), say when the
chains have converged, and give transit posteriors with the usual priors,
faster per independent sample than the Python samplers.

## So far

- `LogProbability`: what is sampled, implemented for any closure
  `Fn(&[f64]) -> f64`.
- `Ensemble`: the walkers' starting points, checked once (an even number,
  at least twice the dimensions, finite, not in a subspace), a seed and a
  number of threads; `check` that every walker starts inside the support.
- `Ensemble::run`: Goodman & Weare's stretch move (`Move::Stretch`, scale
  2 by default, or any mixture of moves by weight), the walkers moved in
  two halves as emcee does. A half's random numbers are all drawn before
  its log-probabilities, so a seed gives the same chain on any number of
  threads. The chain is the image of itself under an affine map of the
  space, proposal for proposal. Gaussians in one, three (standard
  deviations from 1e-3 to 1e3, correlated) and ten dimensions are
  recovered: means and covariances within a few percent of their spread.
- `Chain`: every walker's position and log-probability after each step,
  acceptance fractions, means, covariance, traces, and the last positions
  to start another run from.
- The random numbers: xoshiro256** (Blackman & Vigna 2021) seeded through
  SplitMix64, matching the reference implementations' outputs, so a seed
  gives the same chain on every platform.

## Quick start

```rust
use rustellar_mcmc::{Ensemble, LogProbability};

let target = |x: &[f64]| -0.5 * x.iter().map(|v| v * v).sum::<f64>();
let start = vec![vec![0.1], vec![-0.1]];
let chain = Ensemble::new(start)?.with_seed(1).run(&target, 1000)?;
assert_eq!(chain.steps(), 1000);
# Ok::<(), rustellar_mcmc::McmcError>(())
```

## References

- Blackman, D. & Vigna, S. 2021, ACM TOMS, 47, 36
- Foreman-Mackey, D., Hogg, D. W., Lang, D. & Goodman, J. 2013, PASP, 125, 306
- Goodman, J. & Weare, J. 2010, CAMCoS, 5, 65

## License

MIT or Apache-2.0, at your option.
