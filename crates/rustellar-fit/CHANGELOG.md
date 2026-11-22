# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## 0.1.0 - 2026-11-22

### Added

- `TransitFit`: a light curve, a starting `TransitParams` and a baseline,
  any `Parameter` held fixed, an optional exposure; checked once.
- `TransitFit::run`: Levenberg-Marquardt with geodesic acceleration and
  projected steps, giving a `FitResult` and its `Termination`.
- `Covariance` and `Estimate`: standard errors from `R`, and the depth,
  durations, inclination and stellar density with errors.
- `FitError`.
- `FitResult::jacobians`: the Jacobians from the geometry, which replace
  differencing the light curve.
- A criterion bench on the reference test's six fits.
- `LimbDarkeningBounds` and `TransitFit::with_limb_darkening_bounds`:
  Kipping's (2013) triangle as an option; steps follow a bound the fit
  keeps pressing against, and converge on it.
- `TransitFit::with_threads`: the model and Jacobian over threads, the
  same to the bit as on one.
- `Parameter::ECosW` and `Parameter::ESinW`: eccentric orbits, fixed by
  default; the baseline stays the last parameter.

### Changed

- The Jacobian writes only the rows in transit, and the geodesic
  acceleration runs column by column: 10% to 25% faster on one thread.
