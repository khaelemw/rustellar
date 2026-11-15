# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `TransitParams`: transit parameters checked when built, from `b` or from the
  inclination. Methods:
  - orbit geometry: `phase_angle`, `separation`, `in_front`;
  - timing: `t14`, `t23`, `contacts`, `transit_times`;
  - flux: `flux_at`, `light_curve`, `light_curve_into`, `binned_flux_at`.
- `RadiusRatio` (at most `RadiusRatio::MAX`, 100), `ImpactParameter`,
  `ScaledSemiMajorAxis` and `LimbDarkening` (uniform, linear, quadratic).
- `flux`: the analytic flux of Mandel & Agol (2002) for quadratic limb
  darkening, accurate to about 1e-13 for planets.
- `flux_uniform`, an exact formula for a star with no limb darkening.
- `flux_ring_sum`, a numerical integration kept as an independent check.
- `Exposure`: averaging over each measurement's exposure by the midpoint rule.
- `TransitModel`: fixed times and an optional exposure, evaluated for many
  parameter sets without allocating.
- `ParamError`, `LimbDarkeningError` and `LengthMismatch`.
