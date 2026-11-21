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
- `Ensemble::run`: the walkers split in two halves at random each step
  and moved half by half, as emcee does, with any mixture of moves by
  weight. A half's random numbers are all drawn before its
  log-probabilities, so a seed gives the same chain on any number of
  threads.
- `Move`: Goodman & Weare's stretch move (the default, scale 2) and walk
  move, differential evolution (ter Braak 2006, with Nelson et al.'s
  spread of the scale), ter Braak & Vrugt's snooker move, and the
  ensemble slice sampler (Karamanis & Beutler 2021) along differences of
  walkers or Gaussian directions with the other half's covariance,
  its width tuned over the first steps. All but the snooker move are
  affine-invariant: the chain is the image of itself under an affine map
  of the space, proposal for proposal (the snooker move under rotations,
  even scalings and shifts). Gaussians in one, three (standard deviations
  from 1e-3 to 1e3, correlated) and ten dimensions are recovered by each:
  means and covariances within a few percent of their spread.
- `Chain`: every walker's position and log-probability after each step,
  acceptance fractions, means, covariance, traces, and the last positions
  to start another run from.
- `diagnostics`: the integrated autocorrelation time with Sokal's
  automatic window (as emcee's `integrated_time`, with `trusted` for its
  50-times rule), and the rank-normalised split-R-hat and bulk and tail
  effective sample sizes (Vehtari et al. 2021, as ArviZ). Per parameter of
  a `Chain`, the walkers taken as chains, or on any equal-length chains.
  On 14 sets of chains (independent, autocorrelated up to 0.99,
  anticorrelated, unconverged, tied, heavy-tailed, drifting, too short,
  constant) they agree with emcee 3.1.6 and ArviZ 1.0 to within 5e-15
  relative. The normal quantile behind the rank normalisation (Wichura's
  AS 241) is within 4.9e-16 of 50-digit values, where scipy's is within
  6.2e-16.
- The random numbers: xoshiro256** (Blackman & Vigna 2021) seeded through
  SplitMix64, matching the reference implementations' outputs, so a seed
  gives the same chain on every platform.

## The moves against emcee

Autocorrelation times (`tests/study.rs`, run with `--ignored`): the
largest over the parameters, averaged over three seeds, with their spread.
A ten-dimensional normal (any Gaussian, for the affine-invariant moves)
with 40 walkers, 20,000 steps; Goodman & Weare's Rosenbrock density
`exp(-(100 (y - x^2)^2 + (1 - x)^2) / 20)` with 20 walkers, 50,000 steps.
emcee 3.1.6 with the same targets (vectorised); its walk move, at about a
millisecond per walker per step, ran once for 6,000 steps on the normal
only. One log-probability per walker per step makes the time also the
log-probabilities per independent sample; the slice moves take several.

| Target | Move | emcee tau | rustellar tau | log-probabilities per independent sample | emcee s per 1000 steps | rustellar s per 1000 steps |
|---|---|---|---|---|---|---|
| normal, 10-d | stretch | 123 ± 3 | 127 ± 2 | 127 | 0.30 | 0.0065 |
| | walk | 40 | 40 ± 1 | 40 | 6.0 | 0.046 |
| | differential evolution | 35 ± 1 | 35 ± 1 | 35 | 0.094 | 0.0080 |
| | DE 0.8, snooker 0.2 | | 40 ± 0.4 | 40 | | 0.0088 |
| | slice, differential | | 21 ± 0.2 | 102 | | 0.017 |
| | slice, Gaussian | | 21 ± 0.5 | 104 | | 0.052 |
| Rosenbrock | stretch | 2278 ± 272 | 2323 ± 154 | 2323 | 0.27 | 0.0025 |
| | walk | | 848 ± 138 | 848 | | 0.010 |
| | differential evolution | 908 ± 208 | 698 ± 93 | 698 | 0.28 | 0.0035 |
| | DE 0.8, snooker 0.2 | | 1001 ± 262 | 1001 | | 0.0038 |
| | slice, differential | | 1209 ± 56 | 9046 | | 0.0089 |
| | slice, Gaussian | | 1032 ± 171 | 6272 | | 0.014 |

Where both have the move, the times agree within their spread: the moves
are the same algorithms. emcee's snooker move is left out because it
samples the wrong distribution (its projection is scaled by the distance
to the third walker and its Hastings term halved; a five-dimensional
normal comes out with 42% to 78% of its variance, depending on its scale);
this crate's is checked to give the variance at every scale. On these
targets differential evolution needs the fewest log-probabilities per
independent sample; the slice moves take the fewest steps but several
log-probabilities each. The wall times measure the samplers' own
overhead on cheap targets; for transits, where the model dominates, see
the posterior's comparison.

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
- ter Braak, C. J. F. 2006, Statistics and Computing, 16, 239
- ter Braak, C. J. F. & Vrugt, J. A. 2008, Statistics and Computing, 18, 435
- Foreman-Mackey, D., Hogg, D. W., Lang, D. & Goodman, J. 2013, PASP, 125, 306
- Geyer, C. J. 1992, Statistical Science, 7, 473
- Goodman, J. & Weare, J. 2010, CAMCoS, 5, 65
- Karamanis, M. & Beutler, F. 2021, Statistics and Computing, 31, 61
- Neal, R. M. 2003, Annals of Statistics, 31, 705
- Nelson, B., Ford, E. B. & Payne, M. J. 2014, ApJS, 210, 11
- Sokal, A. 1997, in Functional Integration, NATO ASI Series B, 361, 131
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. 2021, Bayesian Analysis, 16, 667
- Wichura, M. J. 1988, Applied Statistics, 37, 477

## License

MIT or Apache-2.0, at your option.
