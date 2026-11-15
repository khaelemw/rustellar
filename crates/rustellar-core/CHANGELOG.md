# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `Epoch` (BJD_TDB, stored as BTJD) and `TimeDelta`, with arithmetic between
  them.
- `LightCurve`: time, flux and flux error, checked once when built, and
  `fold` into a `FoldedLightCurve`.
- `LightCurveError` and `TimeError`.
