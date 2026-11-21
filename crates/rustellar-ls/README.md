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
- `LombScargle`: the periodogram with a floating mean (Zechmeister &
  Kurster 2009), the mean subtracted once, or none (`Mean`), in astropy's
  four normalisations (`Normalization`), exactly, on all cores.

## Accuracy

The power is the weighted least-squares fit at each frequency, computed
so that rounding does not grow with the data:

- each phase from the exact product of frequency and time, reduced to the
  nearest quarter cycle, so times of millions of days lose nothing;
- each sine and cosine as a whole base and a small part, so terms near 0
  or 1 keep their digits, and the sums centred on their means with
  compensated summation, so frequencies far below 1 / T lose nothing;
- the weighted mean subtracted to twice the digits of an `f64`;
- where the sinusoid's terms are the same at every point to rounding (the
  Nyquist frequency of even sampling), the term that is not is fitted.

Against the same least squares at 40 digits on six seeded cases (uneven
sampling, BJD-sized times, a near-perfect sinusoid, frequencies far below
1 / T, even sampling through its Nyquist frequency, five points), with
astropy 7.2's three methods on the same frequencies, the largest error
as a fraction of the periodogram's peak (floating mean, standard
normalisation):

| Case | rustellar-ls | astropy slow | astropy cython | astropy fast |
|---|--:|--:|--:|--:|
| Uneven sampling | 6e-16 | 5e-15 | 1e-14 | 3e-3 |
| BJD-sized times | 3e-16 | 2e-10 | 2e-10 | 3e-3 |
| Near-perfect sinusoid | 3e-16 | 5e-15 | 5e-15 | 5e-2 |
| Far below 1 / T | 8e-16 | 0.18 | 0.18 | 0.20 |
| Even sampling, Nyquist | 4e-16 | NaN | NaN | 0.51 |
| Five points | 6e-16 | 5e-15 | 1e-14 | 9e-2 |

Every mean and normalisation is within ten roundings of the peak (the
model and log normalisations, which divide by the chi-squared left, also
within a few roundings of the flux over the residuals). One frequency is
left out: five points at 2.5 cycles per day, where the fit's terms
depend on each other to 1e-15 and the exact answer rests on digits no
`f64` computation of them holds; there all four methods agree with each
other, not with 40 digits.

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
