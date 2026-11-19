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
  with standard errors, per-transit and odd/even depths, white and pink
  signal to noise, and its free transit shape.
- `TlsError`.
