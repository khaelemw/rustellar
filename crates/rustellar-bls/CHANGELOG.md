# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `Durations`, `Periods` and `BlsError`: the search inputs, checked when
  built.
- `AutoPeriod`: astropy's automatic period grid, the same to the last bit,
  with errors where astropy quietly carries on with bad settings.
- `OfirPeriod`: Ofir's (2014) optimal period grid, with the IAU 2015 solar
  values, about 50 times fewer periods than `AutoPeriod` over a TESS
  sector.
