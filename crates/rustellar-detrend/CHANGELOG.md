# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `Detrend`, `Method`, `BreakTolerance`, `Edges` and `DetrendError`: a
  time-windowed sliding filter, its settings checked when used.
- `Detrend::trend`: the trend at each point, with an optional mask. The
  biweight correctly rounded wherever its sum's sign can be told from zero,
  the median and the mean correctly rounded, every window edge decided
  exactly; the same results, to the bit, on any number of threads.
- `Detrend::flatten` and `Flattened`: the light curve divided by its trend,
  with the original index of each point kept.
