# rustellar-ls

The Lomb-Scargle periodogram for unevenly sampled series in pure Rust.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `ls` feature.

```toml
[dependencies]
# Through the umbrella crate (what most projects want):
rustellar = { version = "0.1", features = ["ls"] }
# Or this crate alone:
rustellar-ls = "0.1"
```

This crate re-exports the core types it uses (`LightCurve`, `Epoch`,
`TimeDelta` and their errors), so you don't need to add `rustellar-core`
yourself or keep its version in step.

**Status:** version 0.1. The API may still change before 1.0.

## Statement of need

The Lomb-Scargle periodogram is the standard first look for periodic
signals in unevenly sampled data: stellar rotation, pulsations, eclipsing
and ellipsoidal binaries, radial velocities. For each trial frequency a
sinusoid and a floating mean are fitted by weighted least squares
(Zechmeister & Kurster 2009), and the power is how much they lower
chi-squared.

This crate computes that periodogram exactly, or faster to a tolerance you
choose. It also refines peaks between grid frequencies, gives the best-fit
sinusoid, and computes false-alarm probabilities. Frequencies are in
cycles per day on the data's own time scale; nothing is tied to a
particular instrument.

## What it does

- `Frequencies`: an even grid or any list of frequencies, validated once.
- `AutoFrequency`: astropy's automatic grid (VanderPlas 2018), with the
  same defaults, and every setting can be overridden.
- `LombScargle`: the exact periodogram on all cores, with a floating mean
  (Zechmeister & Kurster 2009), the mean subtracted once, or no mean
  (`Mean`), in any of astropy's four normalisations (`Normalization`).
- `LombScargle::power_fast`: the same on an even grid, using non-uniform
  FFTs, to a tolerance you set. Every power is checked against a bound on
  its error, and recomputed exactly where the bound isn't met.
- `LombScargle::peaks`: the highest peaks, each refined off the grid to
  where the power's derivative is zero (`Peak`).
- `LombScargle::model`: the sinusoid fitted at a frequency, with its
  offset, amplitude and phase and their standard errors (`Sinusoid`).
- `LombScargle::false_alarm_probability` and `false_alarm_level`: the
  probability that noise alone would produce a highest peak this high,
  and the inverse. Methods (`FalseAlarm`): Baluev's (2008) approximation,
  the Davies bound, a naive count of independent frequencies, or a single
  frequency.

## Accuracy

The power is the weighted least-squares fit at each frequency, computed
so that rounding error doesn't grow with the data:

- Each phase comes from the exact product of frequency and time, reduced
  to the nearest quarter cycle, so times of millions of days lose no
  precision.
- Each sine and cosine is split into a base value and a small remainder,
  so terms near 0 or 1 keep their digits. Sums are centred on their means
  and use compensated summation, so frequencies far below 1 / T lose
  nothing either.
- The weighted mean is subtracted at twice `f64` precision.
- Where one of the sinusoid's terms is constant across all points (the
  Nyquist frequency of even sampling), only the other term is fitted.

This is compared with the same least squares at 40 digits on six seeded
cases (uneven sampling, BJD-sized times, a near-perfect sinusoid,
frequencies far below 1 / T, even sampling through its Nyquist frequency,
and five points), alongside astropy 7.2's three methods on the same
frequencies. The table gives the largest error as a fraction of the
periodogram's peak (floating mean, standard normalisation):

| Case | rustellar-ls | astropy slow | astropy cython | astropy fast |
|---|--:|--:|--:|--:|
| Uneven sampling | 6e-16 | 5e-15 | 1e-14 | 3e-3 |
| BJD-sized times | 3e-16 | 2e-10 | 2e-10 | 3e-3 |
| Near-perfect sinusoid | 3e-16 | 5e-15 | 5e-15 | 5e-2 |
| Far below 1 / T | 8e-16 | 0.18 | 0.18 | 0.20 |
| Even sampling, Nyquist | 4e-16 | NaN | NaN | 0.51 |
| Five points | 6e-16 | 5e-15 | 1e-14 | 9e-2 |

Every mean option and normalisation is within ten rounding errors of the
peak. (The model and log normalisations, which divide by the remaining
chi-squared, are also within a few rounding errors relative to the flux
over the residuals.) One frequency is left out: five points at 2.5
cycles per day, where the fit's terms are dependent to 1e-15 and the
exact answer depends on digits that no `f64` calculation keeps. There all
four methods agree with each other but not with the 40-digit value.

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

Each point's phase is computed exactly before it is spread onto the FFT
grid, so times of any size lose nothing. The transforms (Leroy 2012, with
the kernel of Barnett, Magland & af Klinteberg 2019) run at a hundredth
of the requested tolerance, and each power's error is bounded using
theirs and the fit's conditioning. Where the bound fails (the lowest
frequencies, where the fit's terms nearly coincide, and degenerate ones
such as the Nyquist frequency of even sampling), that power is computed
exactly instead.

### Peaks and the sinusoid

Each peak is moved to where the power's derivative is zero, using the
derivative's own sums (the same centred terms as the power) and the ITP
root finder (Oliveira & Takahashi 2020). Interpolating the power instead
can't place a peak closer than about 1e-8 of its width, because the
power's change there is smaller than its rounding error. Compared with the
40-digit maximum on the same cases (each case's three highest peaks, with
every mean option), the largest error in peak frequency as a fraction of
the peak width 1 / T:

| Case | rustellar-ls | parabola through three powers | grid's best (astropy) |
|---|--:|--:|--:|
| Uneven sampling | 1e-14 | 1e-2 | 6e-2 |
| BJD-sized times | 1e-14 | 1e-2 | 6e-2 |
| Near-perfect sinusoid | 2e-15 | 2e-5 | 3e-3 |
| Far below 1 / T | 4e-14 | 2e-5 | 1e-4 |
| Even sampling, Nyquist | 7e-15 | 5e-3 | 6e-2 |
| Five points | 2e-15 | 1e-1 | 9e-2 |

The power at the refined peak is within 2e-15 of the exact maximum. The
sinusoid there (offset, amplitude, phase and their standard errors) is
within 4e-15 of the 40-digit values in every case. astropy's
`model_parameters` gets the amplitude wrong by up to 5e-10 at BJD-sized
times, and by 100% far below 1 / T, where its normal equations lose all
precision to cancellation.

### False-alarm probabilities

Compared with Baluev's (2008) formulas at 40 digits, on the same cases
plus 100,000 evenly spaced points, for probabilities from 0.5 down to
1e-40 in every normalisation, the largest relative error:

| Method | probability | level | astropy probability | astropy level |
|---|--:|--:|--:|--:|
| Baluev | 9e-14 | 2e-14 | 6e-11 | 1e-8, fails below 1e-16 |
| Davies | 8e-14 | 2e-14 | 7e-11 | 1e-8, fails below 1e-16 |
| Naive | 1e-14 | 5e-15 | 4e-12 | 8e-4, fails below 1e-16 |
| Single | 1e-14 | 3e-15 | 4e-12 | 3e-12 |

Probabilities are computed in logarithms, so values far below 1e-16 keep
their precision. The gamma-function ratio uses its asymptotic series
rather than a difference of log-gammas, and each level is solved to the
last bits with ITP. With no mean, the degrees of freedom follow Baluev
(none for the reference fit, two with the sinusoid); astropy refuses this
case. In 4,000 simulations of Gaussian noise on 60 uneven points, the
fraction whose highest peak exceeds the Baluev level for 0.5, 0.1 and
0.01 is 0.464, 0.097 and 0.0105 (Baluev is an upper bound, and close for
small probabilities). For the naive level at 0.1 it is 0.24.

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

On an AMD Ryzen 7 5700G (8 cores, 16 threads), floating mean, standard
normalisation, against astropy 7.2 on the same machine and light curves
(`n` points over 30 days with a 2.7-day sinusoid, on an even grid from
0.01 per day in steps of 0.005), best of three. `cargo validate speed ls`
reruns these tables.

| Points x frequencies | exact, 1 thread | exact, 16 threads | astropy slow | astropy cython |
|---|--:|--:|--:|--:|
| 1,000 x 10,000 | 65 ms | 8.5 ms | 0.51 s | 0.23 s |
| 10,000 x 10,000 | 0.66 s | 88 ms | 5.4 s | 2.3 s |

The exact method costs about 7 ns per point and frequency on one thread,
a third of astropy's cython method. The phase comes from two exact
products, and the sine and cosine from fdlibm's minimax polynomials on
the reduced angle (no library sine, and no fused multiply-add needed).
It is compiled for AVX2 where the processor supports it, with the same
arithmetic, so the results are bit-identical either way.

The fast method on 16 threads, against astropy's fast method (which has no
tolerance; see Accuracy):

| Points x frequencies | tolerance 1e-6 | tolerance 1e-12 | astropy fast |
|---|--:|--:|--:|
| 1,000 x 10,000 | 3.1 ms | 3.1 ms | 7.7 ms |
| 10,000 x 10,000 | 3.7 ms | 4.4 ms | 9.4 ms |
| 10,000 x 100,000 | 28 ms | 29 ms | 80 ms |
| 100,000 x 100,000 | 37 ms | 41 ms | 0.12 s |
| 1,000,000 x 1,000,000 | 0.34 s | 0.37 s | 2.0 s |

It costs `O(n + m log m)` for `n` points and `m` frequencies, using this
crate's own cache-blocked FFT, with bit-identical results on any number
of threads. Frequencies below a few times 1 / T are computed exactly (see
Accuracy), which takes most of the time on the smallest grids. `cargo
bench -p rustellar-ls` times both methods alone.

## License

MIT or Apache-2.0.
