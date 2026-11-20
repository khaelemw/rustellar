# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `Tls`: transit least squares over any periods, with TLS's templates,
  durations, spectra and defaults, the highest peaks refined.
- `SearchOptions`, `Template`, `Stars`, `Method` (`Cells`, the default,
  or TLS's `Fold`), and `Device` with the `cuda` feature's GPU search.
- `TlsResult`: each period's fit, the spectra, and the best `Candidate`
  with standard errors (curvature and bootstrap), per-transit and odd/even depths, white and pink
  signal to noise, and its free transit shape.
- `TlsError`, with `TlsError::Grid` for the period grid's errors.
- Docs: the flux must be normalised to 1 out of transit.
- The core types re-exported, and the period grid and `Device` from
  `rustellar-core`: the crate no longer depends on `rustellar-bls`.
- `Method::Fold` bins the folded points at high cadence, the template a
  quadratic across each bin: 2.7 to 3.7 times faster at 2 minutes, every
  period's best within 0.2% of the unbinned one.
