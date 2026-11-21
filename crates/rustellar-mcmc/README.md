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
- The random numbers: xoshiro256** (Blackman & Vigna 2021) seeded through
  SplitMix64, matching the reference implementations' outputs, so a seed
  gives the same chain on every platform.

## Quick start

```rust
use rustellar_mcmc::{Ensemble, LogProbability};

let target = |x: &[f64]| -0.5 * x.iter().map(|v| v * v).sum::<f64>();
let start = vec![vec![0.1], vec![-0.1]];
let ensemble = Ensemble::new(start)?.with_seed(1);
ensemble.check(&target)?;
# Ok::<(), rustellar_mcmc::McmcError>(())
```

## References

- Blackman, D. & Vigna, S. 2021, ACM TOMS, 47, 36
- Foreman-Mackey, D., Hogg, D. W., Lang, D. & Goodman, J. 2013, PASP, 125, 306
- Goodman, J. & Weare, J. 2010, CAMCoS, 5, 65

## License

MIT or Apache-2.0, at your option.
