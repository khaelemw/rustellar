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
- `Device`, `SearchOptions::device`, `BlsResult::device` and
  `BlsError::GpuUnavailable`; with the feature `cuda` (Rust 1.88),
  `Bls::search` runs its binned step and its refinement of the peaks on an
  NVIDIA GPU where there is one (`Device::Auto`, the default), else on the
  CPU. Every point
  goes in the CPU's bin, the bins' sums are exact 64-bit fixed point, and
  every period that could decide the peaks or the best box is searched on
  the CPU; the refinement's folds are made and searched with the CPU's
  arithmetic in the CPU's order. The peaks and the best box are
  `Bls::search`'s to the bit. No GPU or CUDA driver is an error, never a
  panic.
