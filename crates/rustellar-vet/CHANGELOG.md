# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `Candidate`: a period, mid-transit time, duration and optional depth,
  checked; converted from `rustellar-bls`'s `BestBox` (feature `bls`) and
  `rustellar-tls`'s `Candidate` (feature `tls`).
- `fit_depth`, `Fit` and `Shape`: a transit's depth at a fixed period,
  mid-transit time and duration by weighted least squares, with the
  baseline and both standard errors; for a box or a limb-darkened transit
  (`rustellar-model`'s flux on a circular orbit of the candidate's
  duration), centred two-pass compensated sums.
- `refine` and `Refined`: the mid-transit time (within half a duration)
  and duration (half to twice) refined at the candidate's period; exact
  for a box (every run of points in phase scored), a grid and two finer
  grids for a limb-darkened shape.
- `evidence`, `Evidence`, `Transit` and `OddEven`: every transit predicted
  in the light curve's span with its points, expected points (coverage) and
  own depth, and the odd and even transits' depths with the difference's
  error counting their correlation through the shared baseline; all fitted
  with one baseline, solved exactly through the normal equations' arrow
  structure.
- `signal_to_noise`, `SignalToNoise`, `TransitSnr` and `Noise`: the
  signal to noise, white and with the noise measured (errors scaled to the
  scatter out of transit and red noise from residuals binned on the
  transit's duration, after Pont, Zucker & Queloz 2006), for the candidate
  and each transit; the largest transit's share of the depth; and the
  chi-squared drop from one depth to a depth per transit.
- `aliases`, `aliases_at`, `Aliases`, `AliasRow`, `Alias`, `Ratio`,
  `Split` and `Depth`: the candidate at ratios of its period (a third to
  three times by default), each with its fit, evidence, signal to noise and
  chi-squared against the candidate's, and the depths at the transits both,
  either or neither period predicts, fitted with one shared baseline;
  `Display` prints the table.
- `secondary`, `Secondary` and `Dip`: a box of the candidate's duration
  at half the period and at every phase a duration or more from the
  transit, with the deepest dip, the strongest brightening, the scan's
  spread of significances and its number of independent trials.
- `VetError`, with `RatioZero` and `AliasPeriodNotFinite`.
