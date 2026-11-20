# rustellar-tls

Transit least squares in pure Rust: a search for a repeating transit with a
limb-darkened template instead of a box (Hippke & Heller 2019). At each trial
period and duration the template is fitted at every mid-transit time, its
depth by weighted least squares, and the best fit's chi-squared gives the
spectrum.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate.

```toml
[dependencies]
# Through the umbrella, which most projects want:
rustellar = { version = "0.1", features = ["tls"] }
# Or this crate alone:
rustellar-tls = "0.1"
```

On its own it re-exports the core types it takes and returns
(`LightCurve`, `Epoch`, `TimeDelta` and their errors), so
`rustellar-core` need not be listed, and the two always match.

**Status:** in development, not yet released.

Rust 1.85 or later; the `cuda` feature (the search on an NVIDIA GPU, with
the CPU's results to the bit) needs Rust 1.88 or later, an NVIDIA driver
and the CUDA NVRTC library at run time.

## Example

```rust
use rustellar_tls::{LightCurve, OfirPeriod, SearchOptions, Tls};

// Every 30 minutes for 12 days, with a dip of 0.005 lasting 0.1 days every
// 3 days.
let time: Vec<f64> = (0..576).map(|i| f64::from(i) / 48.0).collect();
let flux = time
    .iter()
    .map(|&t| if (t - 0.7).rem_euclid(3.0) < 0.1 { 0.995 } else { 1.0 })
    .collect();
let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 576])?;

// Ofir's (2014) optimal period grid for a Sun-like star.
let periods = OfirPeriod::default().periods(&lc)?;
let result = Tls::new(&lc)?.search(&periods, &SearchOptions::default())?;
let best = result.best().expect("there is a dip");
println!("P = {} d, SDE {}", best.period.as_days(), result.spectra().unwrap().sde);
assert!((best.period.as_days() - 3.0).abs() < 0.01);
# Ok::<(), Box<dyn std::error::Error>>(())
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

The search's own fit at each period is also refined, and every refined
fit's chi-squared is at or below that at transitleastsquares' answer.

## Accuracy

**Against transitleastsquares 1.32.** 764 injected limb-darkened transits
(radius ratio 0.015 to 0.045, period 1 to 9 days), each light curve built
the same way in Python and Rust, both searches on TLS's defaults over the
same period grid. A transit counts as found when the best period is within
1%. Medians over the transits both found, this crate's first:

| Light curves | Found | Period (relative) | Mid-transit time (min) | Duration (relative) | Depth (relative) |
|---|---|---|---|---|---|
| 300, 27 d at 30 min, some sparse | 244 / 245 | 3.8e-4 / 5.2e-4 | 6.1 / 9.0 | 0.055 / 0.093 | 0.061 / 0.065 |
| 200, 30 then 10 min | 165 / 161 | 2.9e-4 / 4.8e-4 | 4.7 / 7.2 | 0.048 / 0.34 | 0.052 / 0.055 |
| 200, errors 1 to 3 times the noise | 130 / 126 | 4.4e-4 / 5.8e-4 | 7.6 / 9.4 | 0.063 / 0.087 | 0.083 / 0.10 |
| 40, 27 d at 2 min | 32 / 32 | 2.5e-4 / 3.5e-4 | 3.0 / 4.1 | 0.019 / 0.048 | 0.047 / 0.059 |
| 24, two sectors at 2 min | 22 / 22 | 8.4e-5 / 1.6e-4 | 3.1 / 4.8 | 0.040 / 0.072 | 0.043 / 0.073 |

Every median is this crate's or lower. Where both found the transit, this
crate is closer to the truth more often in period and mid-transit time in
every set (sign test p ≤ 0.02 except the 40, too few to tell). Recovery is
a tie: 12 transits only this crate found and 5 only transitleastsquares
did (p 0.14), all at SDE 4.5 to 6.9, where the transit and the tallest
noise peak are about as high. `rustellar-validation` reruns this.

**Against a box search.** On `rustellar-bls`'s 1,000 limb-darkened
transits and 1,000 light curves of noise alone, each search's detection
threshold set at 1% false alarms:

| | Found | Period (relative) | Mid-transit time (min) |
|---|---|---|---|
| rustellar-tls | 838 | 1.2e-4 | 1.7 |
| rustellar-bls | 790 | 2.5e-4 | 2.6 |

The gain is largest from the ground (145 of 250 against 104). On the 252
of these transitleastsquares 1.32 was run on, it found 212 to this crate's
214, with mid-transit times of 4.0 minutes against 1.8.

## Speed

Seconds per search on an AMD Ryzen 7 5700G (8 cores, 16 threads) and an
RTX 3060 Ti, Rust 1.99, every search on 16 threads. transitleastsquares
1.32 (numpy 2.4) runs with its defaults, the best of its repeats. One box
transit every 4.7 days in white noise; the same grid (TLS's has 2 to 4
more short periods from older solar constants).

| | Points | Periods | transitleastsquares | rustellar-tls, no bootstrap | default (bootstrap 100) | GPU, no bootstrap |
|---|--:|--:|--:|--:|--:|--:|
| 1 sector, 30 min | 1,240 | 2,436 | 8.41 | 0.73 | 0.90 | 0.16 |
| 1 sector, 2 min | 18,000 | 2,349 | 12.7 | 1.52 | 1.93 | 0.79 |
| 2 sectors, 2 min | 36,000 | 5,411 | 36.6 | 3.99 | 4.74 | 1.40 |

Doing what transitleastsquares does (no bootstrap), that is 8 to 12 times
faster on the CPU and 16 to 53 times on the GPU; the bootstrap's errors add
20 to 27%.
`Method::Fold`, transitleastsquares' own way, takes 1.81, 14.6 and 52.0
seconds (at 2-minute cadence its points are binned, every period's best
within 0.1% of the unbinned search): use it only to check the default
against. `cargo bench -p
rustellar-tls` runs these (`--features cuda` for the GPU).

## References

- Hippke, M. & Heller, R. 2019, A&A, 623, A39
- Mandel, K. & Agol, E. 2002, ApJ, 580, L171
- Ofir, A. 2014, A&A, 561, A138
- Pont, F., Zucker, S. & Queloz, D. 2006, MNRAS, 373, 231

## License

MIT or Apache-2.0, at your option.
