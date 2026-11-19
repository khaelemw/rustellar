# rustellar-tls

Transit least squares in pure Rust: a search for a repeating transit with a
limb-darkened template instead of a box (Hippke & Heller 2019). At each trial
period and duration the template is fitted at every mid-transit time, its
depth by weighted least squares, and the best fit's chi-squared gives the
spectrum.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `tls` feature.

**Status:** in development, not yet released.

Rust 1.85 or later; the `cuda` feature (the search on an NVIDIA GPU, with
the CPU's results to the bit) needs Rust 1.88 or later, an NVIDIA driver
and the CUDA NVRTC library at run time.

## Example

```rust
use rustellar_tls::{OfirPeriod, SearchOptions, Tls};
use rustellar_core::LightCurve;

let lc = LightCurve::from_btjd(time, flux, flux_err)?;
let periods = OfirPeriod::default().periods(&lc)?;
let result = Tls::new(&lc)?.search(&periods, &SearchOptions::default())?;

if let Some(best) = result.best() {
    println!("P = {} d, SDE {}", best.period.as_days(), result.spectra().unwrap().sde);
}
```

`SearchOptions::default()` is TLS's defaults: its template, durations for
stars of 0.13 to 3.5 solar radii and 0.1 to 1 solar mass, a least depth of
10 ppm. It searches on every core, and on the GPU with `cuda` where there
is one.

## Beyond TLS

The best candidate also comes with:

- standard errors of period, transit time, duration and depth, from the
  chi-squared's curvature and by a moving block bootstrap; the bootstrap's
  hold where the signal is weak or the noise correlated over hours, where
  the curvature's are too small;
- each transit's own depth, and odd against even transits;
- a signal to noise for noise correlated over hours (Pont, Zucker & Queloz
  2006) beside TLS's white one;
- its shape with the planet's radius ratio and impact parameter fitted,
  for durations and depths without the template's bias where the cadence
  resolves ingress.

The search's own fit at each period is also refined: on injected
transits against transitleastsquares 1.32, the same transits are found,
with mid-transit times, durations and depths at least as close to the
truth, and every refined fit's chi-squared at or below that at TLS's
answer.

## References

- Hippke, M. & Heller, R. 2019, A&A, 623, A39
- Mandel, K. & Agol, E. 2002, ApJ, 580, L171
- Ofir, A. 2014, A&A, 561, A138
- Pont, F., Zucker, S. & Queloz, D. 2006, MNRAS, 373, 231

## License

MIT or Apache-2.0, at your option.
