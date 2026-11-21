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
- `ParamError`, `LimbDarkeningError` and `LengthMismatch`.

### Changed

- Averaging over an exposure is about a third faster: one sine and cosine
  per time stamp, each instant's phase turned from it, and an exposure wholly
  out of transit not summed. Within 1e-14 of the instant-by-instant mean.
- `flux_at` takes one sine and cosine, and squares rather than `hypot`: 5%
  faster, the same to rounding.
- Light curves leave out the sine and cosine beyond the transit's angle,
  where the flux is exactly 1, and across the limb `flux` builds its lens
  and arcs once for all three integrals: light curves 19% faster, a flux
  across the limb 30% faster, every value the same to the bit.
