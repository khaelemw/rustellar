# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `TransitFit`: a light curve, a starting `TransitParams` and a baseline,
  any `Parameter` held fixed, an optional exposure; checked once.
- `TransitFit::run`: Levenberg-Marquardt with geodesic acceleration and
  projected steps, giving a `FitResult` and its `Termination`.
- `Covariance` and `Estimate`: standard errors from `R`, and the depth,
  durations, inclination and stellar density with errors.
- `FitError`.
