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

## Accuracy

**Survey light curves.** Five simulated curves with spots, drift, flares,
noise and transits (a TESS sector at 2-minute cadence, a fast rotator, one
in counts, a faint star at flux 1e-5, a Kepler quarter), 1,477 sampled
points against the biweight root at 50 digits. Each of wotan's is compared
with the root of its own window:

| | Nearest double | Worst relative error |
|---|--:|--:|
| rustellar-detrend | 1,477 of 1,477 | half a unit |
| wotan 1.10, normalised flux | 0 | 1.0e-6 |
| wotan 1.10, counts | 0 | 2.4e-11 |
| wotan 1.10, faint star | 0 | 4.6e-5 |

Huber's M-estimator (c 1.28) at the same points: this crate's the nearest
double at all 1,477; wotan's `huber_psi`, one step from the median, at
none, worst 3.2e-4 relative to its own window's root.

**Reference windows.** 909 biweight windows and 1,599 windows of the other
methods, at 50 digits or exactly: every M-estimate within half a unit plus
its rounding doubt, and the nearest double wherever that doubt is small
(all but a few windows with roots very near zero, ties and multiple roots);
the order estimators to the bit. wotan is never nearer.

**wotan's `huber`.** At the same 1,477 points, wotan's `huber` against the
exact solution of proposal 2's equations on its own window: the nearest
double at 7, worst relative error 1.9e-9, up to 1.8e-6 of the scale
(statsmodels stops at a tolerance of 1e-6 of the scale). It never fell back
to the median there.

**Detrend then search.** The five curves flattened by this crate's trend
and by wotan's, then searched with rustellar-bls: both find every transit
with the same period and mid-transit time, and depths that agree to 0.006.
The exact trend changes accuracy and speed, not detections.

## Speed

Seconds on an AMD Ryzen 7 5700G (8 cores, 16 threads), Rust 1.99; wotan
1.10 (numba 0.64, after its first call) runs on one thread. The light
curves are the survey's: a Kepler quarter (4,265 points, 1-day window), a
TESS sector at 2 minutes (18,139 points) and at 20 seconds (108,878), the
TESS windows half a day.

| Biweight | wotan | rustellar-detrend, 1 thread | rustellar-detrend, 16 threads |
|---|--:|--:|--:|
| Kepler quarter | 0.0104 | 0.0045 | 0.0013 |
| TESS, 2 minutes | 0.185 | 0.095 | 0.012 |
| TESS, 20 seconds | 6.01 | 3.09 | 0.35 |

Each method on the 2-minute sector, one thread (wotan's default settings):

| | wotan | rustellar-detrend |
|---|--:|--:|
| Median | 0.060 | 0.0022 |
| Mean | 0.0099 | 0.0046 |
| Biweight | 0.185 | 0.095 |
| Welsch | 0.331 | 0.216 |
| Andrews' sine wave | 0.333 | 0.210 |
| Huber | 10.6 (proposal 2) | 0.076 |
| Hodges-Lehmann | 9.76 | 0.97 |
| Trimmed mean | 0.202 | 0.018 |
| Winsorized mean | 0.202 | 0.021 |

wotan's `huber` is a different estimator: statsmodels' proposal 2, which
estimates the scale with the location (c 1.5); this crate's Huber uses a
fixed scale, c times the MAD (c 1.28, as wotan's `huber_psi`).

`cargo bench -p rustellar-detrend` runs these.

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
