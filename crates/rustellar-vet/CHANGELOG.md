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
- `multiples` and `Multiple`: the transits split `n` ways for every `n`
  up to a limit, each group's depth with a shared baseline, the
  chi-squared test of equal depths, and the multiple of the period implied.
- `OddEven::depth_ratio` and `OddEven::half_period`.
- `Secondary::primary` and `Secondary::depth_ratio`.
- `ellipsoidal`, `Ellipsoidal` and `Harmonic`: sinusoids once and twice an
  orbit out of eclipse; `Candidate::duty_cycle`.
- `search_period` and `PeriodSearch`: the period and mid-transit time
  searched near a candidate's, with the spectrum.
- `flag_periods`, `SuspectPeriod` and `PeriodFlag`: the candidate's period
  matched to periods the user gives, at their multiples and fractions.
- `flag_times`, `TimeWindow` and `TimeFlag`: the predicted transits that
  fall in time windows the user gives.
- `VetError`, with `RatioZero`, `AliasPeriodNotFinite`,
  `WidthOutOfRange`, `ToleranceOutOfRange` and `WindowReversed`.
- `refine_with_threads`: `refine` with the limb-darkened grid's trials
  shared among threads, the same result as on one.
