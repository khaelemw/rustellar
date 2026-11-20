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
