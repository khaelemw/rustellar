# rustellar-ls

The Lomb-Scargle periodogram for unevenly sampled series in pure Rust.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `ls` feature.

```toml
[dependencies]
# Through the umbrella, which most projects want:
rustellar = { version = "0.1", features = ["ls"] }
# Or this crate alone:
rustellar-ls = "0.1"
```

On its own it re-exports the core types it takes and returns
(`LightCurve`, `Epoch`, `TimeDelta` and their errors), so
`rustellar-core` need not be listed, and the two always match.

**Status:** in development, not yet released.

## Statement of need

The Lomb-Scargle periodogram is the standard first look for periodic
signals in unevenly sampled data: stellar rotation, pulsations, eclipsing
and ellipsoidal binaries, radial velocities. For each trial frequency a
sinusoid and a floating mean are fitted by weighted least squares
(Zechmeister & Kurster 2009), and the power is how much they lower
chi-squared.

This crate will give that periodogram exactly, and fast to a stated
tolerance, with its peaks refined between the grid's frequencies, the
best-fit sinusoid and false-alarm probabilities. Frequencies are in cycles
per day of the data's own time scale; nothing is tied to an instrument.

## So far

- `Frequencies`: an even grid or any list of frequencies, checked once.
- `AutoFrequency`: astropy's automatic grid (VanderPlas 2018), with its
  defaults and every setting overridable.

```rust
use rustellar_ls::{AutoFrequency, LightCurve};

// 400 points over 20 days; the grid from 0.01 to 5 cycles per day.
let time: Vec<f64> = (0..400).map(|i| 0.05 * f64::from(i)).collect();
let lc = LightCurve::from_days(time, vec![1.0; 400], vec![1e-3; 400])?;
let grid = AutoFrequency::default()
    .with_minimum(0.01)
    .with_maximum(5.0)
    .frequencies(&lc)?;
assert_eq!(grid.min(), 0.01);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## License

MIT or Apache-2.0, at your option.
