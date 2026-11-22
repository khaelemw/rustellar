# rustellar-core

The light curve and time types that every rustellar crate shares, in pure
Rust with no dependencies.

Part of [rustellar](../../README.md). You rarely need to add it yourself:
each rustellar crate re-exports the core types it uses, and the
`rustellar` crate re-exports the light curve and time types.

```toml
[dependencies]
rustellar-core = "0.1"
```

**Status:** version 0.1. The API may still change before 1.0.

## What it does

- `Epoch`: a point in time, in days on the data's own time scale (BJD_TDB,
  BTJD, BKJD, MJD or any other offset). `TimeDelta`: a length of time,
  such as a period or a duration, from days or hours. Two epochs subtract
  to a `TimeDelta`, an epoch plus or minus a `TimeDelta` is an epoch, and
  `TimeDelta`s add, subtract, negate, scale by a number and divide into a
  ratio.
- `LightCurve`: matching time, flux and flux-error arrays, checked once
  when built. Times must be finite and strictly increasing, fluxes finite,
  and errors finite and above zero. Once a light curve exists it is valid,
  so nothing downstream checks it again.
- `LightCurve::fold`: each point's phase on a period, from -0.5 to 0.5
  around a mid-transit time, sorted by phase with its flux and error
  (`FoldedLightCurve`).
- `Periods`: the trial periods of a transit search, any list that is not
  empty and has no period at or below zero.
- `OfirPeriod`: Ofir's (2014) optimal period grid for a star's radius and
  mass, with transitleastsquares' defaults. Each step moves a transit's
  phase by a third of its duration.
- `Device`: where a search runs, on the CPU or a GPU.

Every constructor returns an error (`LightCurveError`, `TimeError`,
`GridError`) instead of panicking.

## Accuracy

`OfirPeriod` gives the same periods as the grids computed in Python with
transitleastsquares 1.32's formula and constants, to within 1e-14
relative. The last bits differ because each platform's maths library
rounds the cube root differently. Where transitleastsquares quietly
changes a setting (clamping the star's radius or mass, or falling back to
a Sun-like star when the grid would have fewer than 100 periods), this
crate returns the grid it was asked for, or an error.

## Quick start

```rust
use rustellar_core::{Epoch, LightCurve, OfirPeriod, TimeDelta};

let lc = LightCurve::from_days(
    vec![0.0, 0.5, 1.0, 1.5],
    vec![1.0, 0.99, 1.0, 0.99],
    vec![0.001; 4],
)?;

// Fold on a 1-day period with phase 0 at day 0.5: the two dips line up.
let folded = lc.fold(TimeDelta::from_days(1.0)?, Epoch::from_days(0.5)?)?;
assert_eq!(folded.flux(), &[1.0, 1.0, 0.99, 0.99]);

// The periods to search over one TESS sector, for a Sun-like star.
let sector = LightCurve::from_days(vec![1325.293, 1353.177], vec![1.0; 2], vec![1e-3; 2])?;
assert_eq!(OfirPeriod::default().periods(&sector)?.as_slice().len(), 2552);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## License

MIT or Apache-2.0.
