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
- `TransitBox` and `BoxFit`: one box fitted exactly, with compensated sums
  and an in-transit test right to about a picosecond far from the
  mid-transit time.
- `TransitBox::best` and `BestBox`: the exact search at one period and
  duration, over every mid-transit time; never below astropy's slow search.
- `Bls`, `SearchOptions`, `Objective`, `BlsResult` and `Peak`: the search
  over many periods. Bins that tile the orbit, the power and best box at
  every period, and the highest peaks refined exactly over every
  mid-transit time, a continuous range of durations, and periods between
  the grid's (ranked by their best box on the grid,
  `Peak::grid_objective`). Runs on all cores with the same results, to the
  bit, on any number; `search_into` reuses a result's memory.
- `BlsError::TooManyBins`.
