# rustellar-bls

Box least squares (BLS) transit search in pure Rust. It computes the
periodogram of Kovacs, Zucker & Mazeh (2002), weighted by the flux errors
(give every point the same error for an unweighted search), and refits the
highest peaks exactly.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate.

```toml
[dependencies]
# Through the umbrella crate (what most projects want):
rustellar = { version = "0.1", features = ["bls"] }
# Or this crate alone:
rustellar-bls = "0.1"
```

This crate re-exports the core types it uses (`LightCurve`, `Epoch`,
`TimeDelta` and their errors), so you don't need to add `rustellar-core`
yourself or keep its version in step.

**Status:** version 0.1. The API may still change before 1.0.

## Statement of need

BLS is the standard first search for transiting planets in survey light
curves. The usual implementation, astropy's `BoxLeastSquares`, is a Python
package with a C kernel, so a Rust pipeline has to call into Python or write
its own. Its fast method also rounds each box to its phase bins, so the period,
mid-transit time and depth it reports are only as good as the bins.

This crate is that search in Rust. It depends only on `rustellar-core`
(plus `cudarc` if you turn on the GPU feature), uses all cores, and gives
bit-identical results on any number of threads. A binned search finds the
peaks, then each of the highest is searched again exactly: over every
mid-transit time, a continuous range of durations, and periods between the
grid points. Its accuracy is measured against astropy, transitleastsquares
and exact rational arithmetic, and every comparison is part of the test
suite.

## Quick start

```rust
use rustellar_bls::{Bls, Durations, LightCurve, Objective, OfirPeriod, SearchOptions};

// Every 30 minutes for 20 days, with a dip of 0.005 lasting 0.15 days
// every 3 days.
let time: Vec<f64> = (0..960).map(|i| f64::from(i) / 48.0).collect();
let flux = time
    .iter()
    .map(|t| if (t % 3.0 - 1.0).abs() < 0.075 { 0.995 } else { 1.0 })
    .collect();
let lc = LightCurve::from_days(time, flux, vec![1e-3; 960])?;

// Ofir's (2014) optimal period grid for a Sun-like star.
let periods = OfirPeriod::default().periods(&lc)?;
let result = Bls::new(&lc)?.search(
    &periods,
    &Durations::from_days(&[0.05, 0.1, 0.15, 0.2])?,
    &SearchOptions::default().with_objective(Objective::Snr),
)?;
let best = result.best().expect("there is a dip");
assert!((best.transit.period().as_days() - 3.0).abs() < 0.01);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`result.power()` is the periodogram and `result.peaks()` the refined peaks.
`AutoPeriod` gives astropy's automatic period grid, and `TransitBox` fits a
single box exactly.

## Accuracy

**Injection and recovery.** 4,000 box-shaped transits in two independent
sets, in simulated TESS, Kepler-like and ground-based light curves (signal
to noise 5 to 100). Both searches run blind over the same Ofir grid and
durations. A transit counts as recovered when the best period is within 1%.
Medians over the recovered transits:

| | Recovered | Period (relative) | Mid-transit time (days) | Depth (relative) |
|---|---|---|---|---|
| rustellar-bls, first 1,000 | 928 | 2.6e-4 | 1.9e-3 | 0.037 |
| astropy 7.2.0 (fast), first 1,000 | 917 | 3.4e-4 | 3.2e-3 | 0.055 |
| rustellar-bls, wide 3,000 | 2,777 | 2.2e-4 | 1.8e-3 | 0.044 |
| astropy 7.2.0 (fast), wide 3,000 | 2,768 | 3.4e-4 | 3.6e-3 | 0.062 |

The errors are lower than astropy's in every survey and signal-to-noise
group of both sets. Recovery is a tie: 51 transits only this crate found and
31 only astropy did, which is not significant at 1%. On a subset of 250,
transitleastsquares 1.32 recovered exactly as many; this crate's
mid-transit time and depth errors are lower (1.6e-3 days against 4.9e-3 on
the first set), and its period error is about the same.

**Limb-darkened transits.** Real transits are rounded, which favours
transitleastsquares. 1,000 transits from the Mandel & Agol model (quadratic
limb darkening, impact parameters 0 to 1), plus 1,000 light curves of pure
noise to set each search's detection threshold for 1% false alarms:

| | Recovered | Period | Mid-transit time (days) |
|---|---|---|---|
| rustellar-bls | 790 | 2.5e-4 | 1.8e-3 |
| astropy 7.2.0 (fast) | 790 | 2.7e-4 | 2.6e-3 |

On a subset of 252, transitleastsquares recovered 212 to this crate's 200,
since it fits a limb-darkened shape instead of a box. Its period error is
the same as this crate's (2.5e-4) and its mid-transit time error is larger
(2.8e-3 days against 1.5e-3).

**Against brute force.** On 36 survey light curves at 864 periods, the
exact search never finds lower power than astropy's brute-force (`slow`)
method, and finds higher power at 788 of them.

**Rounding.** On 234,000 and 1,008,000 points, the depth, signal-to-noise
ratio and power of the reported boxes are compared with their values in
exact rational arithmetic. This crate's largest relative error is 2.0e-15,
within a bound derived from the data. astropy's grows with the period, to
3.0e-13 at 400 days, because it adds up large running totals without
compensation.

## Speed

Seconds per search over 12 durations, on an AMD Ryzen 7 5700G (8 cores, 16
threads), Rust 1.99. astropy 7.2.0 (`fast`, oversampling 10) runs on one
thread. The rustellar-bls columns do not refine the peaks (`refine 0`).
Refining five takes 0.04 to 0.07 seconds more on one sector, 1.0 (16
threads) to 4.4 (1 thread) on thirteen, and 0.2 to 1.6 on Kepler. The
light curves are the survey's: one TESS sector at 2-minute cadence
(18,000 points), thirteen sectors (234,000) and four years of Kepler at
30 minutes (66,253). Ofir's grid runs from 0.6 days (to 100 days for
Kepler).

| | Periods | astropy | rustellar-bls, 1 thread | rustellar-bls, 16 threads |
|---|--:|--:|--:|--:|
| 1 sector | 10,000 | 4.23 | 1.40 | 0.201 |
| 1 sector, Ofir | 2,353 | 0.767 | 0.289 | 0.0413 |
| 13 sectors, Ofir | 42,692 | 88.8 | 47.6 | 7.24 |
| Kepler, Ofir | 166,108 | 200 | 81.7 | 12.2 |

`cargo validate speed bls` reruns the rustellar-bls columns and prints
them beside astropy's recorded times. `cargo bench -p rustellar-bls` runs
the 16-thread cases alone.

### On a GPU

With the `cuda` feature (Rust 1.88 or later, an NVIDIA driver and the CUDA
NVRTC library at run time), `Bls::search` runs the binned step and the
refinement of the peaks on the GPU. The peaks, refined boxes and best box
are bit-identical to the CPU's, and the rest of the power spectrum agrees
to a relative 1e-12. `SearchOptions::device` chooses where a search runs:

- `Device::Auto` (the default): the first GPU if the feature is on and a
  GPU is usable, otherwise the CPU, with no error. If the GPU fails partway
  through, the CPU finishes the search.
- `Device::Cpu`: always the CPU, so the power is bit-identical on every
  machine.
- `Device::Gpu(index)`: that GPU, or `BlsError::GpuUnavailable`.

`BlsResult::device` reports where a search ran.

How the results stay identical:

- Binned step: every point goes in the same bin as on the CPU, and the bin
  sums are exact 64-bit fixed point, so the order they are added in does
  not matter. Any period whose power could change the peaks or the best
  box is searched again on the CPU.
- Refinement: each fold is computed on the GPU with the same arithmetic in
  the same order as the CPU (compensated sums, one point after another),
  and each run of points is scored the same way. The CPU builds each best
  box from its run.

A missing GPU or CUDA driver never causes a panic. The kernels are
compiled once per program (a quarter of a second), and each light curve's
points are copied to the GPU on its first search there (tens of
milliseconds). Seconds per search on an RTX 3060 Ti, next to the same
16-thread CPU timed in the same run (a separate run from the table above,
so the CPU times differ from it by up to 15%):

| | GPU, refine 0 | GPU, refine 5 | 16 threads, refine 0 | 16 threads, refine 5 |
|---|--:|--:|--:|--:|
| 1 sector, Ofir | 0.012 | 0.046 | 0.047 | 0.115 |
| 13 sectors, Ofir | 0.94 | 1.35 | 8.3 | 9.3 |
| Kepler, Ofir | 1.82 | 1.95 | 12.8 | 13.0 |

For many light curves at once, run one search per thread (16 threads,
a new `Bls` for each curve, refine 5, six durations); the searches share
the GPU. Seconds per light curve:

| Light curve | Points | Periods | GPU | CPU |
|---|--:|--:|--:|--:|
| 30-minute sector | 1,162 | 2,438 | 0.005 | 0.010 |
| Two 30-minute sectors | 2,339 | 5,510 | 0.011 | 0.031 |
| Kepler quarter, half kept | 2,049 | 9,046 | 0.016 | 0.061 |
| 2-minute sector | 18,000 | 2,353 | 0.026 | 0.075 |

## Limits

- The power spectrum comes from the binned search. Only the peaks are
  refitted exactly, so a detection statistic measured from the power is
  not inflated.
- The errors are taken as white noise. For noise correlated over a few
  cadences, `Bls::search_whitened` searches a light curve filtered by an
  autoregressive model (`rustellar-detrend`'s `Whitened::matched`), the
  matched filter for a box in that noise. On 400 injected curves it
  finds 23% more transits than the plain search on white noise at the
  same false-alarm rate, and no more under AR(1) red noise (the study is
  in `rustellar-detrend`'s README). Otherwise, inflate the flux errors.
- Detection thresholds are left to you. For vetting statistics and alias
  checks, see `rustellar-vet`.

## References

- Collier Cameron, A., et al. 2006, MNRAS, 373, 799
- Hippke, M. & Heller, R. 2019, A&A, 623, A39
- Kovacs, G., Zucker, S. & Mazeh, T. 2002, A&A, 391, 369
- Mandel, K. & Agol, E. 2002, ApJ, 580, L171
- Ofir, A. 2014, A&A, 561, A138

## License

MIT or Apache-2.0.
