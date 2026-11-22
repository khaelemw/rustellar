# rustellar-tls

Transit least squares in pure Rust (Hippke & Heller 2019). It searches a
light curve for a repeating transit using a limb-darkened transit shape
instead of a box. For each trial period and duration, it fits the shape at
every mid-transit time, solving for the depth by weighted least squares,
and the best fit's chi-squared goes into the periodogram.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate.

```toml
[dependencies]
# Through the umbrella crate (what most projects want):
rustellar = { version = "0.1", features = ["tls"] }
# Or this crate alone:
rustellar-tls = "0.1"
```

This crate re-exports the core types it uses (`LightCurve`, `Epoch`,
`TimeDelta` and their errors), so you don't need to add `rustellar-core`
yourself or keep its version in step.

**Status:** in development, not yet released.

Needs Rust 1.85 or later. The `cuda` feature runs the search on an NVIDIA
GPU with exactly the same results as the CPU. It needs Rust 1.88 or later,
plus an NVIDIA driver and the CUDA NVRTC library at run time.

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
let lc = LightCurve::from_days(time, flux, vec![1e-3; 576])?;

// Ofir's (2014) optimal period grid for a Sun-like star.
let periods = OfirPeriod::default().periods(&lc)?;
let result = Tls::new(&lc)?.search(&periods, &SearchOptions::default())?;
let best = result.best().expect("there is a dip");
println!("P = {} d, SDE {}", best.period.as_days(), result.spectra().unwrap().sde);
assert!((best.period.as_days() - 3.0).abs() < 0.01);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`SearchOptions::default()` uses transitleastsquares' defaults: its
template, durations for stars of 0.13 to 3.5 solar radii and 0.1 to 1
solar mass, and a minimum depth of 10 ppm. The search uses every CPU core,
or the GPU if the `cuda` feature is on and one is present.

## Beyond TLS

The best candidate also comes with:

- Uncertainties on period, transit time, duration and depth, two ways:
  from the curvature of the chi-squared, and from a moving block
  bootstrap. The bootstrap errors stay realistic when the signal is weak
  or the noise is correlated over hours, where the curvature errors come
  out too small.
- The depth of each individual transit, and odd versus even transits.
- A signal-to-noise ratio that accounts for noise correlated over hours
  (Pont, Zucker & Queloz 2006), alongside transitleastsquares' white-noise
  one.
- A fit of the transit shape with the radius ratio and impact parameter
  free. When the cadence resolves ingress, this gives durations and depths
  without the fixed template's bias; `Candidate::duration_estimate` takes
  its duration where it fits better than the template.

The best fit is also refined in period, duration and time together, and
its chi-squared is never worse than at transitleastsquares' answer.

## Accuracy

**Against transitleastsquares 1.32.** 764 injected limb-darkened transits
(radius ratio 0.015 to 0.045, period 1 to 9 days), each light curve built
the same way in Python and Rust, both searches on TLS's defaults over the
same period grid. A transit counts as found when the best period is within
1%. Medians over the transits both found, this crate's first:

| Light curves | Found | Period (relative) | Mid-transit time (min) | Duration (relative) | Depth (relative) |
|---|---|---|---|---|---|
| 300, 27 d at 30 min, some sparse | 244 / 245 | 3.8e-4 / 5.2e-4 | 6.1 / 9.0 | 0.066 / 0.093 | 0.061 / 0.065 |
| 200, 30 then 10 min | 165 / 161 | 2.9e-4 / 4.8e-4 | 4.7 / 7.2 | 0.051 / 0.34 | 0.052 / 0.055 |
| 200, errors 1 to 3 times the noise | 130 / 126 | 4.4e-4 / 5.8e-4 | 7.6 / 9.4 | 0.072 / 0.087 | 0.083 / 0.10 |
| 40, 27 d at 2 min | 32 / 32 | 2.5e-4 / 3.5e-4 | 3.0 / 4.1 | 0.019 / 0.048 | 0.047 / 0.059 |
| 24, two sectors at 2 min | 22 / 22 | 8.4e-5 / 1.6e-4 | 3.1 / 4.8 | 0.040 / 0.072 | 0.043 / 0.073 |

This crate's median error is the same or lower in every column. Where
both found the transit, this crate is closer to the truth more often in
period and mid-transit time in every set (sign test p ≤ 0.02, except the
set of 40, which is too small to tell). Recovery is a tie: 12 transits
were found only by this crate and 5 only by transitleastsquares (p 0.14),
all at SDE 4.5 to 6.9, where the transit is barely taller than the highest
noise peak. `cargo validate accuracy tls` reruns this.

**In harder regimes.** The same comparison on 780 more transits around a
star with the Sun's density, each set changing one thing:

| Light curves | Found | Period (relative) | Mid-transit time (min) | Duration (relative) | Depth (relative) |
|---|---|---|---|---|---|
| 100, periods 9 to 13.5 d (two or three transits) | 66 / 61 | 5.2e-4 / 6.5e-4 | 8.1 / 14 | 0.066 / 0.075 | 0.076 / 0.063 |
| 100, periods 0.25 to 1 d | 50 / 50 | 5.5e-4 / 7.7e-4 | 2.5 / 3.7 | 0.043 / 0.054 | 0.073 / 0.056 |
| 100, radius ratios 0.1 to 0.2 | 100 / 100 | 1.2e-4 / 4.0e-4 | 2.2 / 5.7 | 0.0047 / 0.073 | 0.0094 / 0.021 |
| 100, grazing | 93 / 90 | 2.0e-4 / 3.8e-4 | 3.1 / 5.7 | 0.094 / 0.15 | 0.087 / 0.15 |
| 100, eccentricities 0.1 to 0.7 | 82 / 82 | 3.0e-4 / 4.6e-4 | 5.7 / 8.2 | 0.054 / 0.074 | 0.058 / 0.072 |
| 40, 6 d at 20 s | 28 / 28 | 1.2e-3 / 2.3e-3 | 5.0 / 6.1 | 0.093 / 0.098 | 0.094 / 0.10 |
| 40, three blocks over 108 d | 29 / 30 | 5.1e-5 / 1.3e-4 | 3.6 / 8.0 | 0.049 / 0.26 | 0.046 / 0.047 |
| 100, flagged points, outliers, NaNs | 19 / 18 | 3.2e-4 / 4.0e-4 | 4.1 / 5.6 | 0.037 / 0.038 | 0.023 / 0.030 |
| 100, spotted stars, red noise, detrended | 65 / 66 | 2.9e-4 / 4.6e-4 | 5.2 / 7.7 | 0.090 / 0.062 | 0.20 / 0.21 |
| The same, detrended again around the candidate | 65 / 66 | 4.0e-4 / 4.6e-4 | 5.8 / 7.7 | 0.068 / 0.062 | 0.085 / 0.21 |

This crate is again better in period and mid-transit time in every set,
and in duration in every set but the detrended one. The duration is
`Candidate::duration_estimate`: the template's, or the shape fit's where
that fit lowers the chi-squared by more than 1. A fixed template fits deep
and grazing transits, whose ingress is longer than its own, with too short
a duration (by 12% and 26%), and the shape fit removes that bias.

On the detrended curves every duration comes out short, because the
biweight takes part of each transit's edges into the trend: this crate's
by 8%, transitleastsquares' by 4%. On evenly sampled curves
transitleastsquares' durations run 2 to 6% longer than this crate's, even
where this crate's are unbiased, so here they land closer. Detrending
again with the candidate's transits masked (`Detrend::flatten_around` in
`rustellar-detrend`) brings the duration level with transitleastsquares'
(closer in 31 of 64, p 0.90) and cuts the depth error from 0.20 to 0.085.
`cargo validate accuracy around` reruns that row.

Outliers of 5 to 20 sigma in 1% of the points break both searches, so
clip them before searching.

**Against a box search.** On `rustellar-bls`'s 1,000 limb-darkened
transits and 1,000 light curves of pure noise, with each search's
detection threshold set for 1% false alarms:

| | Found | Period (relative) | Mid-transit time (min) |
|---|---|---|---|
| rustellar-tls | 838 | 1.2e-4 | 1.7 |
| rustellar-bls | 790 | 2.5e-4 | 2.6 |

The gain is largest for ground-based data (145 of 250, against 104). On
the 252 of these that transitleastsquares 1.32 was also run on, it found
212 to this crate's 214, with a median mid-transit time error of 4.0
minutes against 1.8.

## Speed

Seconds per search on an AMD Ryzen 7 5700G (8 cores, 16 threads) and an
RTX 3060 Ti, Rust 1.99, with every search on 16 threads.
transitleastsquares 1.32 (numpy 2.4) runs with its defaults, best of
several repeats. The data is a box transit every 4.7 days in white noise,
searched over the same period grid (transitleastsquares' has 2 to 4 extra
short periods because it uses older solar constants).

| | Points | Periods | transitleastsquares | rustellar-tls, no bootstrap | default (bootstrap 100) | GPU, no bootstrap |
|---|--:|--:|--:|--:|--:|--:|
| 1 sector, 30 min | 1,240 | 2,436 | 8.82 | 0.728 | 0.880 | 0.154 |
| 1 sector, 2 min | 18,000 | 2,349 | 13.2 | 1.50 | 1.89 | 0.810 |
| 2 sectors, 2 min | 36,000 | 5,411 | 38.2 | 3.89 | 4.67 | 1.42 |

Doing the same work as transitleastsquares (no bootstrap), this crate is
9 to 12 times faster on the CPU and 16 to 57 times faster on the GPU. The
bootstrap errors add 20 to 26% to the time.

`Method::Fold` searches the way transitleastsquares does and takes 1.71,
9.77 and 32.3 seconds, still faster than transitleastsquares in every
case. At 2-minute cadence it bins the points, which keeps each period's
best fit within 0.2% of the unbinned search. It is there to check the
default method against, not for normal use.

### Many light curves

`Tls::search_many` searches many light curves with the same options, each
over its own periods, and gives each curve exactly the result
`Tls::search` would. It runs a quarter as many searches at once as there
are threads, four threads each, sharing the GPU if there is one. Seconds
per light curve, sixteen one-sector curves, no bootstrap, 16 threads:

| | Loop, CPU | `search_many`, CPU | Loop, GPU | `search_many`, GPU |
|---|--:|--:|--:|--:|
| 30-minute cadence | 0.759 | 0.557 | 0.186 | 0.117 |
| 2-minute cadence | 1.49 | 1.05 | 0.768 | 0.490 |

That is 1.4 times the speed of a loop on the CPU and 1.6 times on the
GPU. With the GPU, most of the remaining time is spent refining the
peaks, which runs on the CPU.

`cargo validate speed tls` reruns both tables, printing the rustellar-tls
columns beside transitleastsquares' recorded times (`cargo validate-gpu`
for the GPU columns). `cargo bench -p rustellar-tls` runs the Rust timings
alone.

## References

- Hippke, M. & Heller, R. 2019, A&A, 623, A39
- Mandel, K. & Agol, E. 2002, ApJ, 580, L171
- Ofir, A. 2014, A&A, 561, A138
- Pont, F., Zucker, S. & Queloz, D. 2006, MNRAS, 373, 231

## License

MIT or Apache-2.0.
