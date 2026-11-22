# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## 0.1.0 - 2026-11-22

### Added

- `Epoch` (days on the data's own time scale: BJD_TDB, BTJD, BKJD, MJD or
  any other) and `TimeDelta`, with arithmetic between them.
- `LightCurve`: time, flux and flux error, checked once when built
  (`new`, or `from_days` from plain numbers), and `fold` into a
  `FoldedLightCurve`.
- `LightCurveError` and `TimeError`.
- `Periods`, `OfirPeriod` (Ofir's 2014 optimal period grid) and
  `GridError`, shared by the searches; `Device`, where a search runs.
