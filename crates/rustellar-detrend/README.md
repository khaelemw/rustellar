# rustellar-detrend

Detrending light curves in pure Rust: the slow variation of a star and of
the instrument estimated and divided out before a transit search. The method
is a time-windowed sliding filter, by default Tukey's biweight (Hippke et al.
2019), solved to full double precision, with every point in the windows
that cover it and the same results on any number of threads. Welsch's,
Andrews' and Huber's M-estimators, the Hodges-Lehmann estimator, the
trimmed and winsorized means, the median and the mean are there too.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `detrend` feature.

**Status:** in development, not yet released.

## Statement of need

The time-windowed biweight is the detrender most transit searches use,
through wotan, a Python package. Its iteration stops at a tolerance, so its
trend is good to a few parts per million at best, and worse for fluxes in
counts or far from 1; a window edge falls on whichever side the rounding of
a time difference puts it.

This crate is that detrender for Rust. Its biweight and other M-estimators
are correctly rounded wherever the sign of the sum that defines them can be
told from zero; its median, mean, Hodges-Lehmann estimator and trimmed and
winsorized means are correctly rounded; and every window edge is decided
exactly. It runs on all cores with the same result, to the bit, on any
number of threads and any processor, and on one thread it is faster than
wotan.

## Quick start

```rust
use rustellar_core::{LightCurve, TimeDelta};
use rustellar_detrend::Detrend;

// Two days at 2-minute cadence: a slow drift and a 0.5% dip.
let time: Vec<f64> = (0..1440).map(|i| f64::from(i) / 720.0).collect();
let dip = |t: f64| (1.0..1.05).contains(&t);
let flux: Vec<f64> = time
    .iter()
    .map(|&t| (1.0 + 0.01 * t) * if dip(t) { 0.995 } else { 1.0 })
    .collect();
let lc = LightCurve::from_btjd(time.clone(), flux, vec![1e-3; 1440])?;

// Mask the dip so the trend passes under it.
let mask: Vec<bool> = time.iter().map(|&t| dip(t)).collect();
let detrend = Detrend::default().with_window(TimeDelta::from_days(0.5)?);
let flat = detrend.flatten(&lc, Some(&mask))?;

let flux = flat.light_curve().flux();
assert!((flux[100] - 1.0).abs() < 1e-3);
assert!((flux[740] - 0.995).abs() < 1e-5);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## References

- Andrews, D. F., Bickel, P. J., Hampel, F. R., Huber, P. J., Rogers, W. H. &
  Tukey, J. W. 1972, Robust Estimates of Location (Princeton University Press)
- Dennis, J. E. & Welsch, R. E. 1978, Communications in Statistics -
  Simulation and Computation, 7, 345
- Hippke, M., David, T. J., Mulders, G. D. & Heller, R. 2019, AJ, 158, 143
- Hodges, J. L. & Lehmann, E. L. 1963, Annals of Mathematical Statistics,
  34, 598
- Huber, P. J. 1964, Annals of Mathematical Statistics, 35, 73

## License

MIT or Apache-2.0, at your option.
