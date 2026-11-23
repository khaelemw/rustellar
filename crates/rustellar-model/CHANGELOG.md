# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## 0.1.0 - 2026-11-22

### Added

- `TransitParams`: transit parameters checked when built, from `b` or from the
  inclination. Methods:
  - orbit geometry: `phase_angle`, `separation`, `in_front`;
  - timing: `t14`, `t23`, `contacts`, `transit_times`;
  - flux: `flux_at`, `light_curve`, `light_curve_into`, `binned_flux_at`.
- `RadiusRatio` (at most `RadiusRatio::MAX`, 10,000), `ImpactParameter`,
  `ScaledSemiMajorAxis` and `LimbDarkening` (uniform, linear, quadratic).
- `blocked_light`: the three parts of the blocked light that the flux is
  linear in, for every limb-darkening law at once.
- `flux`: the analytic flux for quadratic limb darkening (Mandel & Agol 2002),
  with `B1` from Agol, Luger & Foreman-Mackey (2020) and Bulirsch's `cel`,
  and the area and `r^2` moments by Green's theorem. Within 1e-15 of exact for
  `k` up to 1 and about 1.5e-16 k above, on every case line and at
  separations down to the smallest double.
- `flux_uniform`, an exact formula for a star with no limb darkening.
- `flux_ring_sum`, a numerical integration kept as an independent check.
- `Exposure`: averaging over each measurement's exposure by the midpoint rule.
- `TransitModel`: fixed times and an optional exposure, evaluated for many
  parameter sets without allocating.
- Eccentric orbits: `TransitParams::with_eccentricity`, `eccentricity` and
  `periastron_deg`; every geometry, timing and flux method follows the orbit.
- `ParamError`, `LimbDarkeningError` and `LengthMismatch`;
  `ParamError::EccentricityOutOfRange` and `PeriastronTooClose`.
