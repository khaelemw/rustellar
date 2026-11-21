# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `Frequencies`: the trial frequencies in cycles per day, an even grid or
  any list, checked once; `AutoFrequency`, astropy's automatic grid
  (VanderPlas 2018) with its defaults and every setting overridable.
- `LombScargle`, `Mean`, `Normalization` and `Periodogram`: the
  periodogram with a floating, subtracted or no mean, in the standard,
  model, log and PSD normalisations; exact phases at any time, compensated
  centred sums, the mean to twice the digits, degenerate frequencies
  fitted along the term that holds; on all cores. About 7 ns per point and
  frequency on one thread: exact products by Veltkamp splitting, fdlibm's
  minimax sine and cosine on the reduced phase, AVX2 where available with
  the same bits.
- `LombScargle::power_fast`: the periodogram on an even grid to a stated
  tolerance by three non-uniform FFTs (exponential-of-semicircle kernel),
  each power's error bounded and computed exactly where the bound fails;
  2 to 5 times faster than astropy's fast method, which has no tolerance.
- `LombScargle::peaks` and `Peak`: the highest peaks refined off the grid
  to the zero of the power's derivative, by its own sums and the ITP
  method, to 1e-14 of a peak's width.
- `LombScargle::model` and `Sinusoid`: the fitted sinusoid's offset,
  amplitude and phase with standard errors, and its value at any time.
- `LombScargle::false_alarm_probability`, `false_alarm_level` and
  `FalseAlarm`: Baluev (2008), Davies, naive and single-frequency
  probabilities of the highest peak and their inverses, in logarithms, to
  1e-13 of 40 digits; checked against simulated noise.
- `LsError`, with `NotUniform`, `NotProbability` and `TooFewPoints`.
