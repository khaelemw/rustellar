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
- `VetError`.
