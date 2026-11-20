# rustellar-detrend

Detrending light curves in pure Rust: the slow variation of a star and of
the instrument estimated and divided out before a transit search. The method
is a time-windowed sliding filter, by default Tukey's biweight (Hippke et al.
2019), solved to full double precision, with every point in the windows
that cover it and the same results on any number of threads. Welsch's,
Andrews' and Huber's M-estimators, Huber's proposal 2 (location and scale
together, as wotan's `huber`), the Hodges-Lehmann estimator, the trimmed
and winsorized means, the median and the mean are there too, and a
least-squares cubic spline fitted to each segment, plain, clipped or
Huber's.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `detrend` feature.

```toml
[dependencies]
# Through the umbrella, which most projects want:
rustellar = { version = "0.1", features = ["detrend"] }
# Or this crate alone:
rustellar-detrend = "0.1"
```

On its own it re-exports the core types it takes and returns
(`LightCurve`, `Epoch`, `TimeDelta` and their errors), so
`rustellar-core` need not be listed, and the two always match.

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
use rustellar_detrend::{Detrend, LightCurve, TimeDelta};

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

**Huber's proposal 2** (`Method::HuberJoint`, c 1.5), against the exact
solution of its two equations at 60 digits: this crate's the nearest
double at all 1,477 points; wotan's `huber` (statsmodels, which stops at a
tolerance of 1e-6 of the scale), against the solution on its own window,
at 7, worst relative error 1.9e-9. It never fell back to the median
there.

**The spline** (`Method::Spline`), against the exact least-squares
spline on the same knots (60 digits) on 60 light curves (40,291 points of
normalised flux, counts, values about zero and about 1e200, with gaps,
masks and edge cutoffs), in units of 2^-52 times the largest flux: this
crate's median error 1.4, scipy's `LSQUnivariateSpline` (FITPACK) 17.5.
Where masks leave the knots badly conditioned, scipy's worst is 9,128
and this crate's 113. The times are taken relative to the knots exactly
and the fit refined twice from residuals found with fused multiply-adds.

**The robust spline** (`Method::RobustSpline`), against the same clipping
loop run exactly at 60 digits on 40 light curves with flares, dips and
single outliers: the same points clipped in every case, and the trend
within 2 ulps of the data's scale; the loop with scipy's fits, 4 to 47.
wotan's `rspline` is a different loop: it clips only below the spline, so
flares stay in and on pure noise its trend sits 0.09 standard deviations
high (this crate's: unbiased, both sides clipped), and it always makes
ten fits.

**The Huber spline** (`Method::HuberSpline`, statsmodels' `RLM` with
`HuberT` on the B-splines), against its fixed point solved at 50 digits on
30 light curves with flares, dips and outliers: within 3 ulps of the data's
scale in 29, 71 in one badly conditioned case; statsmodels at its default
tolerance from 4 ulps to 1.1e10 (1e-6 relative), at 1e-14 from 2 to 40.
Newton's steps on which points are within c s (the estimating equations
are linear once those are known), not reweighting alone: one or two steps
where reweighting takes tens.

**Detrend then search.** The five curves flattened by this crate's trend
and by wotan's, then searched with rustellar-bls: both find every transit
with the same period and mid-transit time, and depths that agree to 0.006.
The exact trend changes accuracy and speed, not detections.

## Choosing the window

A window too short follows a transit and takes some of its depth; one too
long leaves the star's variability in, which raises false peaks. Hippke et
al. (2019) suggest about three times the transit's duration. Measured here
on 400 injected curves (a TESS sector at 10 minutes, white noise 1e-3;
limb-darkened transits 1 to 7 hours long, signal to noise 10 to 44 in white
noise alone for 80% of them; for the star, two sinusoids of 0.5 to 10 days
and 1e-4 to 5e-3), biweight throughout. The share of searches whose best
BLS peak is the injected period, and the medians of the depth kept and of
the signal to noise kept (as a fraction of the white-noise ideal):

| Window | Blind BLS: right period | Right, SDE above 7 | Depth kept | Signal to noise kept | Transits masked: signal to noise kept |
|---|--:|--:|--:|--:|--:|
| 1.5 durations | 0.70 | 0.45 | 0.37 | 0.37 | 1.00 |
| 2 | 0.77 | 0.53 | 0.53 | 0.53 | 0.99 |
| 2.5 | 0.77 | 0.50 | 0.64 | 0.63 | 0.98 |
| 3 | 0.75 | 0.46 | 0.71 | 0.69 | 0.97 |
| 4 | 0.68 | 0.37 | 0.80 | 0.77 | 0.96 |
| 6 | 0.61 | 0.26 | 0.89 | 0.83 | 0.95 |
| 8 | 0.56 | 0.23 | 0.92 | 0.84 | 0.92 |
| 12 | 0.49 | 0.17 | 0.95 | 0.81 | 0.87 |
| 0.5 days | 0.70 | 0.46 | 0.70 | 0.66 | 0.98 |

A longer window keeps more of the depth, yet the search does worse after
it: the variability it leaves in makes false peaks. With the transits
masked (over 1.2 durations) every window keeps the whole depth, and the
shorter it is the less variability it leaves. So, two steps:

1. **Search** after `Detrend::with_window_for_transit(longest)`: a window
   of 2.5 times the longest duration searched for (`WINDOW_PER_DURATION`),
   in the middle of the range the search did best after, so a transit
   somewhat longer than expected keeps most of its depth.
2. **Measure** a candidate after detrending again with its transits left
   out (`transit_mask`) and a window of two durations, which leaves a few
   points either side of a mask wider than the transit.

`cargo run --release -p rustellar-validation --bin compare -- window`
reruns the study.

## Speed

Seconds on an AMD Ryzen 7 5700G (8 cores, 16 threads), Rust 1.99; wotan
1.10 (numba 0.64, after its first call) runs on one thread. The light
curves are the survey's: a Kepler quarter (4,265 points, 1-day window), a
TESS sector at 2 minutes (18,139 points) and at 20 seconds (108,878), the
TESS windows half a day.

| Biweight | wotan 1.10 | rustellar-detrend, 1 thread | rustellar-detrend, 16 threads |
|---|--:|--:|--:|
| Kepler quarter | 0.0105 | 0.00430 | 0.00122 |
| TESS, 2 minutes | 0.180 | 0.0919 | 0.0130 |
| TESS, 20 seconds | 5.84 | 2.91 | 0.334 |

Each method on the 2-minute sector, one thread (wotan's default settings):

| | wotan 1.10 | rustellar-detrend |
|---|--:|--:|
| Median | 0.0582 | 0.00251 |
| Mean | 0.00958 | 0.00504 |
| Biweight | 0.180 | 0.0910 |
| Welsch | 0.321 | 0.205 |
| Andrews' sine wave | 0.323 | 0.198 |
| Huber (fixed scale) | - | 0.0731 |
| Huber's proposal 2 | 10.3 | 0.148 |
| Hodges-Lehmann | 9.19 | 0.933 |
| Trimmed mean | 0.196 | 0.0176 |
| Winsorized mean | 0.196 | 0.0204 |

The least-squares spline (`Method::Spline`, knots half a day apart) on the
same sector: scipy's `LSQUnivariateSpline` 0.00259, this crate 0.00244.
The robust spline (sigma 3): wotan's `rspline` 0.447, this crate 0.00616.
The Huber spline (c 1.345): statsmodels' `RLM` 0.765 (the design matrix
built beforehand), wotan's `hspline` (scikit-learn) 0.269, this crate
0.0114.

`Method::Huber` uses a fixed scale, c times the MAD (c 1.28, as wotan's
`huber_psi`); `Method::HuberJoint` estimates the scale with the location,
as wotan's `huber`.

`cargo run --release -p rustellar-validation --bin compare --
speed-detrend` prints these tables again, the rustellar-detrend columns
timed afresh beside wotan's recorded times; `cargo bench -p
rustellar-detrend` runs them alone.

## References

- Andrews, D. F., Bickel, P. J., Hampel, F. R., Huber, P. J., Rogers, W. H. &
  Tukey, J. W. 1972, Robust Estimates of Location (Princeton University Press)
- Dennis, J. E. & Welsch, R. E. 1978, Communications in Statistics -
  Simulation and Computation, 7, 345
- Hippke, M., David, T. J., Mulders, G. D. & Heller, R. 2019, AJ, 158, 143
- Hodges, J. L. & Lehmann, E. L. 1963, Annals of Mathematical Statistics,
  34, 598
- Huber, P. J. 1964, Annals of Mathematical Statistics, 35, 73
- Dierckx, P. 1993, Curve and Surface Fitting with Splines (Oxford
  University Press)

## License

MIT or Apache-2.0, at your option.
