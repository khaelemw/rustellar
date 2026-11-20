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
- `Method::HuberJoint`: Huber's proposal 2, the location and scale
  estimated together (statsmodels' `Huber`, wotan's `method='huber'`),
  solved exactly: each split of the points solved as a quadratic, the last
  in double-double, with bisection on the scale where the splits do not
  settle.
- `transit_mask`: the points within half a duration of a candidate's
  mid-transit times, to detrend again with its transits left out.
- `Detrend::with_window_for_transit` and `WINDOW_PER_DURATION` (2.5): a
  window from the longest transit duration searched for, chosen by a blind
  BLS search on 400 injected curves (in the README).
- More methods: Welsch's, Andrews' sine wave and Huber's M-estimators
  (correctly rounded wherever their sums' signs can be told from zero, as the
  biweight), and the Hodges-Lehmann estimator and the trimmed and winsorized
  means (correctly rounded). The plain series run four to a vector with AVX2
  where the processor has it, to the same bits as without, and so do the
  double-double sums that settle the last bit, sixteen terms side by side.
