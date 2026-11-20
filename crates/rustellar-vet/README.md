# rustellar-vet

Vetting transit candidates in pure Rust: the evidence that a search's
period is the real one, shown rather than decided.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `vet` feature.

```toml
[dependencies]
# Through the umbrella, which most projects want:
rustellar = { version = "0.1", features = ["vet"] }
# Or this crate alone:
rustellar-vet = "0.1"
```

On its own it re-exports the core types it takes and returns
(`LightCurve`, `Epoch`, `TimeDelta` and their errors), so
`rustellar-core` need not be listed, and the two always match.

**Status:** in development, not yet released.

## Statement of need

A transit search reports one best period. Sometimes the true period is a
simple ratio of it: half of it (every other predicted transit is empty),
double (every other real transit was skipped), or a third, three times,
two thirds or three halves. A weak signal's chi-squared often prefers the
wrong one, because the noise really does favour it. What tells them apart
is the evidence transit by transit: whether the odd and even transits have
the same depth, how many predicted transits fall where there are data, and
each transit's own depth.

This crate will give, for a candidate and each of its aliases, the fit and
that evidence as numbers with their errors, in a table, with no hidden
thresholds. It takes a `Candidate` from any search: from `rustellar-bls`
with the `bls` feature, from `rustellar-tls` with the `tls` feature, or
built from a period, a mid-transit time and a duration.

## So far

- `Candidate`: a period, mid-transit time, duration and optional depth.
- `fit_depth`: the depth at that period, mid-transit time and duration,
  with its standard error, for a limb-darkened shape (the default: a small
  planet half-way to the limb of a Sun-like star) or a box. The model is
  baseline times (1 - depth times shape), fitted by weighted least squares;
  the depth comes back to rounding from a noiseless transit, and its
  stated error matches the scatter of 4,000 noisy fits to within 4%.
- `refine`: the mid-transit time and duration that fit best near the
  candidate's, at its period. For a box this is exact (no point of a
  301 by 301 grid does better); for a limb-darkened shape it matches a
  dense grid's best to a part in a million.
- `evidence`: each predicted transit's points, coverage and own depth,
  and the odd and even transits' depths compared. The difference's error
  counts the two depths' correlation through their shared baseline; on
  1,000 noise draws of a planet the odd-even significance has mean 0 and
  spread 1 to within the draws' own scatter, and a candidate at half the
  true period or an eclipsing binary stands out by more than 10 sigma.
- `signal_to_noise`: the signal to noise with the errors given (white)
  and with the noise measured out of transit (red, after Pont, Zucker &
  Queloz 2006), for the candidate and for each transit; the largest
  transit's share of the depth; and a chi-squared test of whether every
  transit has one depth. In red noise as large as the white, over 300
  draws, the depth scatters three times its white error and within 12% of
  its red error; in white noise the depth test's chi-squared has the mean
  and variance it should.
- `aliases`: the candidate at a third, half, two thirds, one, three
  halves, two and three times its period (or any ratios), each with its
  fit, evidence and signal to noise, its chi-squared against the
  candidate's, and the depth at the transits only one of the two periods
  predicts; printed as a table. With a planet at each of those ratios of
  the period searched, the true one is the best row every time, by a
  chi-squared of more than 50.
- `secondary`: a box of the transit's duration slid over every phase at
  least a duration from the transit, the transit's points left out: the
  dip at half the period with its upper limit, the deepest dip and the
  strongest brightening, and the spread of the scan's significances (the
  light curve's own noise scale). Each box equals a box fit to 1e-12, and
  in white noise the significance at half the period has mean 0 and
  spread 1.

## Quick start

```rust
use rustellar_vet::{Candidate, Epoch, TimeDelta};

let candidate = Candidate::new(
    TimeDelta::from_days(4.7)?,
    Epoch::from_btjd(1326.3)?,
    TimeDelta::from_hours(2.9)?,
)?
.with_depth(1e-3)?;
assert_eq!(candidate.depth(), Some(1e-3));
# Ok::<(), Box<dyn std::error::Error>>(())
```

## License

MIT or Apache-2.0, at your option.
