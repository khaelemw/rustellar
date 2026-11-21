# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `LogProbability`, for any closure `Fn(&[f64]) -> f64`.
- `Ensemble`: walkers' starting points checked once, a seed and threads;
  `check` against a log-probability.
- `Ensemble::run`, `Move` (the stretch move, mixtures by weight) and
  `Chain`.
- `McmcError`.
- `Move::Walk`, `Move::DifferentialEvolution`, `Move::Snooker` and
  `Move::Slice` with its `Direction` (differential or Gaussian); the
  halves split at random each step.
- `transit`: `TransitPosterior`, `Coordinate`, `Prior`; walkers from a
  fit; a stellar density prior. `McmcError::InvalidPrior` and
  `NoCovariance`.
- A pool of threads started once per run, the work taken item by item.
- `diagnostics`: `integrated_time`, `rhat`, `ess_bulk`, `ess_tail`;
  `Chain::traces`, `autocorrelation_times`, `rhat`, `ess_bulk`, `ess_tail`.
