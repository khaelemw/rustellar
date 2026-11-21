# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `LogProbability`, for any closure `Fn(&[f64]) -> f64`.
- `Ensemble`: walkers' starting points checked once, a seed and threads;
  `check` against a log-probability.
- `McmcError`.
