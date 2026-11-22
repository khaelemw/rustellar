# rustellar-mcmc

Ensemble Markov chain Monte Carlo in pure Rust. It samples a posterior
from its log-probability, provides diagnostics that tell you when the
chains can be trusted, and builds transit posteriors on `rustellar-model`
and `rustellar-fit`.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `mcmc` feature.

```toml
[dependencies]
# Through the umbrella crate (what most projects want):
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
correlations. This crate samples any log-probability with an ensemble of
walkers (Goodman & Weare 2010; Foreman-Mackey et al. 2013), tells you when
the chains have converged, and provides transit posteriors with the usual
priors, at a lower cost per independent sample than the Python samplers.

## What it does

- `LogProbability`: the function being sampled. It is implemented for any
  closure `Fn(&[f64]) -> f64`.
- `Ensemble`: the walkers' starting points, validated once (an even
  number of walkers, at least twice the dimension, all finite, and not
  confined to a subspace), plus a seed and a thread count. `check`
  confirms every walker starts where the log-probability is finite.
- `Ensemble::run`: each step, the walkers are split into two random
  halves and moved one half at a time, as in emcee, using any weighted
  mixture of moves. All of a half's random numbers are drawn before its
  log-probabilities are computed, so a seed gives the same chain on any
  number of threads. Threads are started once per run and pick up
  log-probabilities (or slice updates) one at a time, so one slow
  evaluation doesn't hold the others up.
- `Move`: Goodman & Weare's stretch move (the default, scale 2) and walk
  move; differential evolution (ter Braak 2006, with Nelson et al.'s
  random scale); ter Braak & Vrugt's snooker move; and the ensemble slice
  sampler (Karamanis & Beutler 2021), stepping along differences of
  walkers or along Gaussian directions drawn from the other half's
  covariance, with its width tuned over the first steps. Every move except
  the snooker move is affine-invariant: applying an affine map to the
  space maps the chain onto itself, proposal for proposal. (The snooker
  move is invariant under rotations, uniform scalings and shifts.) Each
  move recovers Gaussians in one, three (correlated, with standard
  deviations from 1e-3 to 1e3) and ten dimensions, with means and
  covariances within a few percent of their spread.
- `transit::TransitPosterior`: the posterior of a `rustellar-fit` set-up's
  free parameters: the Gaussian likelihood of the flux under the transit
  model (with the set-up's exposure), times the priors. When both
  parameters of a pair are free, limb darkening is sampled as Kipping's
  (2013) `q1` and `q2`, and eccentricity as `sqrt(e) cos w` and
  `sqrt(e) sin w` (Eastman, Gaudi & Agol 2013). Each coordinate is uniform
  over its valid range unless you give it a uniform, normal or
  log-uniform `Prior`, and you can add a normal prior on the star's
  density, which follows from `a / R*` and the period (Seager &
  Mallen-Ornelas 2003). `walkers` draws starting points from the fit's
  best parameters and covariance. Where the posterior is close to
  Gaussian, its spreads match the fit's errors to within a few percent.
- `Chain`: every walker's position and log-probability after each step,
  plus acceptance fractions, means, covariance, traces, and the final
  positions for starting another run.
- `diagnostics`: the integrated autocorrelation time with Sokal's
  automatic window (as in emcee's `integrated_time`, with `trusted` for
  its 50-times rule), and the rank-normalised split-R-hat and bulk and
  tail effective sample sizes (Vehtari et al. 2021, as in ArviZ). They
  work per parameter of a `Chain` (treating each walker as a chain) or on
  any set of equal-length chains. On 14 sets of chains (independent,
  autocorrelated up to 0.99, anticorrelated, unconverged, tied,
  heavy-tailed, drifting, too short, constant) they agree with emcee
  3.1.6 and ArviZ 1.0 to 5e-15 relative. The normal quantile used in the
  rank normalisation (Wichura's AS 241) is within 4.9e-16 of 50-digit
  values.
- Random numbers: xoshiro256** (Blackman & Vigna 2021), seeded through
  SplitMix64 and matching the reference implementations' output, so a
  seed gives the same chain on every platform.

## The moves against emcee

Autocorrelation times from `tests/study.rs` (run with `--ignored`): the
largest over the parameters, averaged over three seeds, with their
spread. The targets are a ten-dimensional normal (which stands for any
Gaussian, for the affine-invariant moves) with 40 walkers and 20,000
steps, and Goodman & Weare's Rosenbrock density
`exp(-(100 (y - x^2)^2 + (1 - x)^2) / 20)` with 20 walkers and 50,000
steps. emcee 3.1.6 ran the same targets (vectorised). Its walk move takes
about a millisecond per walker per step, so it ran only once, for 6,000
steps, on the normal. Most moves cost one log-probability per walker per
step, so tau is also the log-probabilities per independent sample; the
slice moves take several per step.

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

Where both packages have a move, the times agree within their spread, as
expected for the same algorithm. emcee's snooker move is left out because
it samples the wrong distribution: it scales its projection by the
distance to the third walker and halves the Hastings term, so a
five-dimensional normal comes out with only 42% to 78% of its true
variance, depending on its scale. This crate's snooker move is tested to
give the right variance at every scale. On these targets, differential
evolution needs the fewest log-probabilities per independent sample. The
slice moves take the fewest steps but several log-probabilities each. The
wall times measure each sampler's own overhead on cheap targets; for
transits, where the model dominates, see below.

## The slice moves against zeus

zeus 2.5.4 is Karamanis & Beutler's own ensemble slice sampler. It ran on
the same targets, walkers, steps and seeds as the table above, and on the
transit posterior below (40 walkers, 30,000 steps with the first 5,000
discarded, using batman for the model). zeus's differential move tunes
its width until expansions and contractions are roughly balanced, then
stops tuning. Its Gaussian move keeps a fixed width of twice the other
walkers' spread, which corresponds to a width of 2 here
(`Move::slice_gaussian(2.0, 0)`).

| Target | Move | zeus tau | rustellar tau | zeus log-probabilities per independent sample | rustellar | zeus s per 1000 steps | rustellar s per 1000 steps |
|---|---|---|---|---|---|---|---|
| normal, 10-d | differential, tuned | 20.8 ± 0.3 | 21 ± 0.2 | 102 | 102 | 1.2 | 0.017 |
| | Gaussian, width 2 | 20.5 ± 0.3 | 20.6 ± 0.2 | 102 | 102 | 1.4 | 0.054 |
| Rosenbrock | differential, tuned | 878 ± 16 | 1209 ± 56 | 6097 | 9046 | 1.5 | 0.0089 |
| | differential, width 2 | | 480 ± 73 | | 3789 | | 0.010 |
| | Gaussian, width 2 | 556 ± 90 | 626 ± 65 | 4280 | 4821 | 1.7 | 0.016 |
| transit | differential, tuned | 128 | 140 | 708 | 864 | | |
| | differential, width 1 | | 125 | | 787 | | |

With the width fixed, the two cost the same number of log-probabilities
per step to within 1.3% (6.96 and 6.88 at a width of 1 on the Rosenbrock
density, 4.877 and 4.874 on the normal), confirming they are the same
algorithm. Where they differ, it's because tuning settles on a different
width. zeus stops tuning after a few dozen steps, at a width that varies
from run to run (it picks pairs of walkers with Python's own `random`,
which numpy's seed doesn't control; an earlier run of the same study gave
1287 on the Rosenbrock density), whereas this crate tunes for `tune`
steps. Balancing expansions and contractions isn't optimal, though: on
the curved Rosenbrock density, a fixed width of 2 needs 2.4 times fewer
log-probabilities than the tuned width (zeus's tuned differential move
likewise needs more than its fixed-width Gaussian one), and on the normal
the fixed width costs nothing extra. On the transit, zeus needs 10% to
20% fewer log-probabilities per independent sample, but each one costs
four times as much. Its 43.6 ms per independent sample (one thread) is
3.3 times slower than this crate's tuned slice move on one thread
(13.1 ms) and 11 times slower than on four threads (3.9 ms). Both are
slower on this posterior than differential evolution.

## The transit posterior against emcee with batman

The test case is one hot Jupiter transit (500 points, 0.0006 d apart,
white noise of 1e-3; `k` 0.1, `a / R*` 10, `b` 0.3), with t0, k, a/R*, b,
q1, q2 and the baseline free under uniform priors. The reference is emcee
3.1.6 with batman 2.5.1, using the stretch move with 40 walkers for
60,000 steps: 2.4 million log-probabilities in 137 s, with
autocorrelation times from 170 steps (t0) to 620 (b). This crate uses
differential evolution with 40 walkers started from the fit's
covariance, for 30,000 steps with the first 5,000 discarded. Over 30
seeds, every 16th, 50th and 84th percentile of every coordinate and of
the star's density agrees with emcee's to within 0.093 of the posterior's
standard deviation, and every 2.5th and 97.5th to within 0.152, which is
what the two chains' Monte Carlo errors allow. The test
(`tests/transit.rs`) allows 0.15 and 0.3, because each platform's maths
library rounds differently and so draws a different chain from the same
seed.

## Grazing and eccentric transits against zeus

Two harder posteriors, each sampled with 40 walkers started from the fit,
using differential evolution with the snooker move a fifth of the time:

- A grazing transit (`k` 0.1, `a / R*` 10, `b` 1.0, noise of 5e-4, `k`
  uniform below 0.3): 200,000 steps, the first fifth discarded.
- An eccentric orbit (`e` 0.3, `w` 60 degrees, noise of 1e-3, the
  eccentricity free as sqrt(e) cos w and sqrt(e) sin w, the star's density
  known to 10%): 100,000 steps.

The references come from zeus 2.5.4 with batman 2.5.1, pooling four
chains of 40,000 steps for the grazing transit and eight of 120,000 for
the eccentric orbit. Each percentile comparison subtracts the reference's
own uncertainty at that percentile: how far its individual chains strayed
from the pooled result (`tests/reference/<case>_spread.csv`).

emcee 3.1.6 isn't used as the reference here. On the eccentric
posterior, four chains of its differential-evolution moves (300,000 steps
each) agreed with each other, but all of them missed a tail of the
distribution: q2 above 0.4, and sqrt(e) sin w below zero. zeus and this
crate both find that tail from any starting point, and their
log-probabilities agree to 0.002 at the same points. emcee's stretch move
mixes too slowly on either posterior.

Gaps between this crate's percentiles and the reference's, in the
reference's standard deviations and after subtracting its own spread,
over 30 seeds:

| | 16th to 84th, median / worst | 2.5th and 97.5th, median / worst | Test allows |
|---|---|---|---|
| Grazing | 0.003 / 0.081 | 0.010 / 0.03 | 0.15 / 0.3 |
| Eccentric | 0.19 / 0.37 | 0.34 / 0.74 | 0.5 / 1.0 |

The grazing figures leave out 2 of the 30 seeds. In each of those, one
walker spent the whole run stuck in a narrow region where the planet
fully transits (`a / R*` near 16, `b` near 0.76), which shifts the 97.5th
percentiles of `a / R*` and the density by up to 7 sigma. One of zeus's
four chains did the same, so the grazing test uses the median of three
seeds.

On the eccentric orbit, the lower tail of t0 and sqrt(e) sin w mix slowly
in both samplers; in zeus their autocorrelation times reach 10,000 to
27,000 steps.

## Time per independent sample

The hot Jupiter posterior again, with 40 walkers and 30,000 steps (the
first 5,000 discarded). Time per independent sample is the run time
divided by 40 × 25,000 / (largest autocorrelation time). emcee 3.1.6 runs
with batman 2.5.1 and with PyTransit 2.6.18 (its quadratic model, within
3e-8 of batman's), and this crate on 1 and 4 threads (`tests/study.rs`;
`cargo validate speed mcmc` reruns the table).

| Sampler | Move | tau | Seconds | ms per independent sample |
|---|---|---|---|---|
| emcee + batman | stretch | 552 | 71.2 | 39.3 |
| emcee + batman | DE | 239 | 66.2 | 15.8 |
| emcee + PyTransit | stretch | 543 | 44.5 | 24.2 |
| emcee + PyTransit | DE | 231 | 40.5 | 9.38 |
| this crate, 1 thread | stretch | 520 | 16.4 | 8.53 |
| this crate, 1 thread | DE | 241 | 15.0 | 3.61 |
| this crate, 1 thread | slice | 140 | 93.1 | 13.1 |
| zeus 2.5.4 + batman | slice | 128 | 339 | 43.6 |
| this crate, 4 threads | stretch | 520 | 5.78 | 3.01 |
| this crate, 4 threads | DE | 241 | 5.24 | 1.26 |
| this crate, 4 threads | slice | 140 | 27.8 | 3.90 |

The autocorrelation times match emcee's for each move, and each
log-probability costs a third of PyTransit's and a fifth of batman's
under emcee. So on one thread this crate needs 2.6 to 4.6 times less time
per independent sample, and on four threads 7 to 13 times less. The slice
move mixes best per step but needs 864 log-probabilities per independent
sample against differential evolution's 289, so on a cheap posterior like
this one differential evolution is faster.

## Threads

A transit posterior (t0, k, a/R*, b, q1, q2 and the baseline free; 40
walkers, the stretch move, 1000 steps; `tests/study.rs`) on a 16-thread
machine. The chains are bit-identical at every thread count.

| Points | 1 thread | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 500 | 0.55 s | 1.8 times faster | 2.9 | 2.9 | 1.4 |
| 5000 | 6.4 s | 1.9 | 3.6 | 5.6 | 5.8 |

Each half-step has only 20 log-probabilities to share out, so when they're
cheap the threads soon spend more time waiting than working. A few
threads help on short light curves, and more on long ones.

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
- Eastman, J., Gaudi, B. S. & Agol, E. 2013, PASP, 125, 83
- Foreman-Mackey, D., Hogg, D. W., Lang, D. & Goodman, J. 2013, PASP, 125, 306
- Geyer, C. J. 1992, Statistical Science, 7, 473
- Goodman, J. & Weare, J. 2010, CAMCoS, 5, 65
- Karamanis, M. & Beutler, F. 2021, Statistics and Computing, 31, 61
- Kipping, D. M. 2013, MNRAS, 435, 2152
- Neal, R. M. 2003, Annals of Statistics, 31, 705
- Nelson, B., Ford, E. B. & Payne, M. J. 2014, ApJS, 210, 11
- Seager, S. & Mallen-Ornelas, G. 2003, ApJ, 585, 1038
- Sokal, A. 1997, in Functional Integration, NATO ASI Series B, 361, 131
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. 2021, Bayesian Analysis, 16, 667
- Wichura, M. J. 1988, Applied Statistics, 37, 477

## License

MIT or Apache-2.0.
