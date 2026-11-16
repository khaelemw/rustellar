# rustellar-bls

Box least squares (BLS) transit search in pure Rust: the periodogram of
Kovacs, Zucker & Mazeh (2002), weighted by the flux errors, with the
highest peaks refitted exactly.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate.

**Status:** in development, not yet released.

## Statement of need

BLS is the standard first search for transiting planets in survey light
curves. The usual implementation, astropy's `BoxLeastSquares`, is a Python
package with a C kernel, so a Rust pipeline has to call into Python or write
its own. Its fast method also rounds each box to its phase bins, so the period,
mid-transit time and depth it reports are only as good as the bins.

This crate is that search for Rust. It has no dependencies beyond
`rustellar-core` and runs on all cores with the same result, to the bit, on
any number of threads. The binned search finds the peaks. Each of the
highest is then searched again exactly: over every mid-transit time, a
continuous range of durations, and periods between the grid's. Its accuracy
is measured against astropy, transitleastsquares and exact rational
arithmetic, and every comparison runs in the test suite.

## Quick start

```rust
use rustellar_bls::{Bls, Durations, Objective, OfirPeriod, SearchOptions};
use rustellar_core::LightCurve;

// Every 30 minutes for 20 days, with a dip of 0.005 lasting 0.15 days
// every 3 days.
let time: Vec<f64> = (0..960).map(|i| f64::from(i) / 48.0).collect();
let flux = time
    .iter()
    .map(|t| if (t % 3.0 - 1.0).abs() < 0.075 { 0.995 } else { 1.0 })
    .collect();
let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 960])?;

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

`result.power()` is the periodogram, `result.peaks()` the refined peaks.
astropy's automatic grid is `AutoPeriod`; `TransitBox` fits one box exactly.

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
limb darkening, impact parameters 0 to 1), and 1,000 light curves of noise
alone to set each search's detection threshold at 1% false alarms:

| | Recovered | Period | Mid-transit time (days) |
|---|---|---|---|
| rustellar-bls | 790 | 2.5e-4 | 1.8e-3 |
| astropy 7.2.0 (fast) | 790 | 2.7e-4 | 2.6e-3 |

On a subset of 252, transitleastsquares recovered 212 to this crate's 200: it
fits a limb-darkened transit rather than a box. Its period error is the same
as this crate's (2.5e-4), and its mid-transit time error larger (2.8e-3 days
against 1.5e-3).

**Against brute force.** On 36 survey light curves at 864 periods, the
exact search is never below astropy's brute-force (`slow`) method, and
above it at 788.

**Rounding.** On 234,000 and 1,008,000 points, the depth, signal-to-noise
ratio and power of the reported boxes are compared with their values in
exact rational arithmetic. This crate's largest relative error is 2.0e-15,
within a bound derived from the data. astropy's grows with the period, to
3.0e-13 at 400 days, because it sums large running totals plainly.

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
| 1 sector | 10,000 | 4.19 | 1.60 | 0.23 |
| 1 sector, Ofir | 2,353 | 0.76 | 0.36 | 0.047 |
| 13 sectors, Ofir | 42,692 | 86.8 | 48.8 | 8.3 |
| Kepler, Ofir | 166,108 | 194.9 | 83.9 | 12.8 |

`cargo bench -p rustellar-bls` runs the 16-thread cases.

### On a GPU

With the `cuda` feature (Rust 1.88 or later, an NVIDIA driver and the CUDA
NVRTC library at run time), `Bls::search` runs the binned step and the
refinement of the peaks on the GPU. The peaks, refined boxes and best box
are the CPU's to the bit; the power elsewhere agrees to 1e-12 of itself.
`SearchOptions::device` chooses where a search runs:

- `Device::Auto` (the default): the first GPU where the feature is on and a
  GPU can be used, otherwise the CPU, with no error. A GPU that fails during
  a search leaves it to the CPU.
- `Device::Cpu`: the CPU, for power the same to the bit on every machine.
- `Device::Gpu(index)`: that GPU, or `BlsError::GpuUnavailable`.

`BlsResult::device` says where a search ran.

- Binned step: every point is put in the same bin as on the CPU, and the
  bins' sums are exact 64-bit fixed point, so their order does not matter.
  Every period whose power could change the peaks or the best box is
  searched again on the CPU.
- Refinement: each fold is made on the GPU with the CPU's arithmetic, in the
  CPU's order (the compensated sums one point after another), and every run
  of points is scored as the CPU scores it. The CPU builds each best box
  from its run.

No GPU or CUDA driver is never a panic. The kernels are compiled once a
program (a quarter of a second), and each light curve's points are copied
to the GPU at its first search there (tens of milliseconds). Seconds per
search on an RTX 3060 Ti beside the same 16-thread CPU:

| | GPU, refine 0 | GPU, refine 5 | 16 threads, refine 0 | 16 threads, refine 5 |
|---|--:|--:|--:|--:|
| 1 sector, Ofir | 0.012 | 0.046 | 0.047 | 0.115 |
| 13 sectors, Ofir | 0.94 | 1.35 | 8.3 | 9.3 |
| Kepler, Ofir | 1.82 | 1.95 | 12.8 | 13.0 |

Searching many light curves at once, each on one thread from 16 threads
(a new `Bls` each, refine 5, six durations), the GPU shared between them;
seconds a light curve:

| Light curve | Points | Periods | GPU | CPU |
|---|--:|--:|--:|--:|
| 30-minute sector | 1,162 | 2,438 | 0.005 | 0.010 |
| Two 30-minute sectors | 2,339 | 5,510 | 0.011 | 0.031 |
| Kepler quarter, half kept | 2,049 | 9,046 | 0.016 | 0.061 |
| 2-minute sector | 18,000 | 2,353 | 0.026 | 0.075 |

## Limits

- The power is the binned search's. Only the peaks are refitted exactly, so
  a detection statistic measured from the power is not inflated.
- The errors are taken as white noise. Red noise is not modelled; inflate
  the flux errors to allow for it.
- Detection thresholds, vetting statistics and alias checks are for the
  pipeline using the search.

## References

- Collier Cameron, A., et al. 2006, MNRAS, 373, 799
- Hippke, M. & Heller, R. 2019, A&A, 623, A39
- Kovacs, G., Zucker, S. & Mazeh, T. 2002, A&A, 391, 369
- Mandel, K. & Agol, E. 2002, ApJ, 580, L171
- Ofir, A. 2014, A&A, 561, A138

## License

MIT or Apache-2.0, at your option.
