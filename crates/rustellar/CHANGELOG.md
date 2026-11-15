# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- The `rustellar` crate, with the `rustellar-core` types at the top level.
  Each tool is a module, turned on by a feature of the same name.
- The `model` feature (transit light curves, from `rustellar-model`), and
  `full` for every tool.
- The `bls` feature (box least squares, from `rustellar-bls`; in
  development).
