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
- `LombScargle::power_fast`: the same on an even grid to a stated
  tolerance, by non-uniform FFTs, with every power checked against a
  bound on its error and computed exactly where the bound is not met.
- `LombScargle::peaks`: the highest peaks, each refined off the grid to
  where the power's derivative is zero (`Peak`).
- `LombScargle::model`: the sinusoid fitted at a frequency, its offset,
  amplitude and phase with their standard errors (`Sinusoid`).

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

The fast method's largest error in the standard normalisation on the same
cases (floating mean), against astropy's fast method above:

| Case | tolerance 1e-4 | 1e-8 | 1e-12 |
|---|--:|--:|--:|
| Uneven sampling | 5e-8 | 6e-12 | 6e-15 |
| BJD-sized times | 5e-8 | 6e-12 | 6e-15 |
| Near-perfect sinusoid | 2e-7 | 3e-11 | 7e-15 |
| Far below 1 / T | 3e-6 | 2e-10 | 3e-14 |
| Even sampling, Nyquist | 2e-7 | 7e-11 | 4e-15 |
| Five points | 2e-6 | 2e-10 | 3e-14 |

Each point's phase is exact before it is spread, so times of any size
lose nothing; the transforms (Leroy 2012, with the kernel of Barnett,
Magland & af Klinteberg 2019) are asked for a hundredth of the
tolerance, and each power's error is bounded from theirs and the fit's
conditioning. Where the bound fails (the lowest frequencies, where the
fit's terms nearly coincide, and degenerate ones such as the Nyquist
frequency of even sampling) that power is computed exactly instead.

### Peaks and the sinusoid

Each peak is moved to where the power's derivative is zero, from the
derivative's own sums (the same centred terms as the power), by the ITP
method (Oliveira & Takahashi 2020). Interpolating the power instead cannot
place a peak closer than about 1e-8 of its width, where its change falls
below its rounding. Against the maximum at 40 digits on the same cases
(each case's three highest peaks with every mean), the largest error in
the peak's frequency as a fraction of its width 1 / T:

| Case | rustellar-ls | parabola through three powers | grid's best (astropy) |
|---|--:|--:|--:|
| Uneven sampling | 1e-14 | 1e-2 | 6e-2 |
| BJD-sized times | 1e-14 | 1e-2 | 6e-2 |
| Near-perfect sinusoid | 2e-15 | 2e-5 | 3e-3 |
| Far below 1 / T | 4e-14 | 2e-5 | 1e-4 |
| Even sampling, Nyquist | 7e-15 | 5e-3 | 6e-2 |
| Five points | 2e-15 | 1e-1 | 9e-2 |

The power there is within 2e-15 of the exact power at the maximum. The
sinusoid there (offset, amplitude, phase and their standard errors) is
within 4e-15 of 40 digits on every case; astropy's `model_parameters`
gives an amplitude off by up to 5e-10 at BJD-sized times and by 100% far
below 1 / T, where the normal equations of its raw terms cancel.

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

## Speed

On one AMD Ryzen (16 threads), floating mean, standard normalisation, an
even grid of frequencies (`cargo bench -p rustellar-ls`), against astropy
7.2 on the same machine and sizes:

| Points x frequencies | exact, 1 thread | exact, all threads | astropy slow | astropy cython |
|---|--:|--:|--:|--:|
| 1,000 x 10,000 | 69 ms | 10 ms | 539 ms | 303 ms |
| 10,000 x 10,000 | 703 ms | 99 ms | 5,355 ms | 2,718 ms |

The exact method costs about 7 ns per point and frequency on one thread,
4 times less than astropy's cython method: the phase from two exact
products and fdlibm's minimax polynomials on the reduced angle (no
library sine, no fused multiply-add needed), compiled for AVX2 where the
processor has it (the same arithmetic, so the same bits either way).

The fast method, all threads, against astropy's fast method (which has no
tolerance; see Accuracy):

| Points x frequencies | tolerance 1e-6 | tolerance 1e-12 | astropy fast |
|---|--:|--:|--:|
| 1,000 x 10,000 | 3.2 ms | 3.6 ms | 7.4 ms |
| 10,000 x 10,000 | 4.4 ms | 5.1 ms | 9.8 ms |
| 10,000 x 100,000 | 33 ms | 34 ms | 81 ms |
| 100,000 x 100,000 | 42 ms | 45 ms | 125 ms |
| 1,000,000 x 1,000,000 | 381 ms | 404 ms | 2,019 ms |

It costs `O(n + m log m)` for `n` points and `m` frequencies, on our own
cache-blocked FFT, with the same bits on any number of threads.

## License

MIT or Apache-2.0, at your option.
