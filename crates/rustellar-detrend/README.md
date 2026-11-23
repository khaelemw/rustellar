# rustellar-detrend

Detrending light curves in pure Rust. It estimates the slow variation from
the star and the instrument and divides it out before a transit search.

The default method is a sliding time window with Tukey's biweight (Hippke
et al. 2019), solved to full double precision, with bit-identical results
on any number of threads. Also available:

- Welsch's, Andrews' and Huber's M-estimators;
- Huber's proposal 2 (location and scale estimated together, like wotan's
  `huber`);
- the Hodges-Lehmann estimator, trimmed and winsorized means, the median
  and the mean;
- a least-squares cubic spline fitted to each segment, either plain,
  with outliers clipped, or with Huber weights.

Part of [rustellar](https://github.com/khaelemw/rustellar). Use it on its own, or through the
`rustellar` crate with the `detrend` feature.

```toml
[dependencies]
# Through the umbrella crate (what most projects want):
rustellar = { version = "0.1", features = ["detrend"] }
# Or this crate alone:
rustellar-detrend = "0.1"
```

This crate re-exports the core types it uses (`LightCurve`, `Epoch`,
`TimeDelta` and their errors), so you don't need to add `rustellar-core`
yourself or keep its version in step.

**Status:** version 0.1. The API may still change before 1.0.

## Statement of need

The time-windowed biweight is the detrender most transit searches use,
via the Python package wotan. wotan's iteration stops at a tolerance, so
its trend is accurate to a few parts per million at best, and worse for
fluxes in counts or far from 1. Whether a point lands inside a window can
also depend on rounding in a time difference.

This crate is that detrender in Rust. Its biweight and other M-estimators
return the correctly rounded answer whenever the sign of the defining sum
can be distinguished from zero. The median, mean, Hodges-Lehmann estimator
and trimmed and winsorized means are always correctly rounded, and window
edges are decided exactly. It uses all cores, gives bit-identical results
on any thread count and any processor, and is faster than wotan even on
one thread.

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
let lc = LightCurve::from_days(time.clone(), flux, vec![1e-3; 1440])?;

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
in counts, a faint star at flux 1e-5, and a Kepler quarter). At 1,477
sampled points, each trend value is compared with the 50-digit biweight
solution. wotan's values are compared with the solution for wotan's own
window at each point:

| | Nearest double | Worst relative error |
|---|--:|--:|
| rustellar-detrend | 1,477 of 1,477 | half a unit |
| wotan 1.10, normalised flux | 0 | 1.0e-6 |
| wotan 1.10, counts | 0 | 2.4e-11 |
| wotan 1.10, faint star | 0 | 4.6e-5 |

For Huber's M-estimator (c 1.28) at the same points, this crate gives the
nearest double at all 1,477. wotan's `huber_psi`, which takes one step
from the median, gives it at none, with a worst error of 3.2e-4 relative
to its own window's solution.

**Reference windows.** On 909 biweight windows and 1,599 windows for the
other methods, compared with 50-digit or exact answers, every M-estimate
is within half a unit in the last place plus its rounding uncertainty,
and is the nearest double wherever that uncertainty is small (all but a
few windows with solutions very near zero, ties or multiple solutions).
The order-based estimators match exactly. wotan is never closer.

**Huber's proposal 2** (`Method::HuberJoint`, c 1.5), compared with the
60-digit solution of its two equations: this crate gives the nearest
double at all 1,477 points. wotan's `huber` (via statsmodels, which stops
at a tolerance of 1e-6 of the scale) does so at 7, with a worst relative
error of 1.9e-9. This crate never had to fall back to the median on these
points.

**The spline** (`Method::Spline`) is compared with the exact 60-digit
least-squares spline on the same knots, on 60 light curves (40,291 points
of normalised flux, counts, values around zero and around 1e200, with
gaps, masks and edge cutoffs). In units of 2^-52 times the largest flux,
this crate's median error is 1.4 and scipy's `LSQUnivariateSpline`
(FITPACK) 17.5. Where masks leave the knots badly conditioned, scipy's
worst is 9,128 and this crate's 113. Times are taken relative to the
knots exactly, and the fit is refined twice from residuals computed with
fused multiply-adds.

**The robust spline** (`Method::RobustSpline`) is compared with the same
clipping loop run at 60 digits, on 40 light curves with flares, dips and
single outliers. It clips exactly the same points in every case, and the
trend is within 2 ulps of the data's scale; the same loop using scipy's
fits is off by 4 to 47. wotan's `rspline` works differently: it clips only
points below the spline, so flares stay in, its trend on pure noise sits
0.09 standard deviations too high (this crate clips both sides and is
unbiased), and it always runs ten fits.

**The Huber spline** (`Method::HuberSpline`, equivalent to statsmodels'
`RLM` with `HuberT` on the B-splines) is compared with its fixed point
solved at 50 digits, on 30 light curves with flares, dips and outliers.
It is within 3 ulps of the data's scale in 29 of them, and 71 in one
badly conditioned case. statsmodels at its default tolerance is off by 4
ulps up to 1.1e10 (1e-6 relative), and at a tolerance of 1e-14 by 2 to
40. Once you know which points lie within c × s, the estimating equations
are linear, so this crate takes Newton steps on that set instead of only
reweighting: one or two steps where reweighting takes tens.

**The autoregressive fit** (`Autoregressive`) is compared with
statsmodels 0.15's `AutoReg` (trend "c") and with the same least squares
solved at 60 digits, on 48 series (autoregressive processes around 1,
with a leftover wave, in counts, and near a unit root). Measuring each
term of the prediction (the constant, and each coefficient times the
largest flux) against the largest term, this crate's error is at most
0.16 ulps with a median of 0; statsmodels' median is 161 and its worst
4,122. The order chosen by AIC matches `ar_select_order` in all 48. The
fit uses Givens rotations (without the platform's `hypot`, so results are
identical everywhere), then refines on residuals with exact products
until they stop shrinking; statsmodels uses a pseudo-inverse.

**Detrend then search.** The five curves were flattened with this crate's
trend and with wotan's, then searched with rustellar-bls. Both find every
transit with the same period and mid-transit time, and depths agree to
0.006. So the exact trend improves accuracy and speed, but doesn't change
what gets detected.

## Choosing the window

A window that's too short follows the transit and removes some of its
depth; one that's too long leaves stellar variability in, which creates
false peaks. Hippke et al. (2019) suggest about three times the transit
duration. This was measured on 400 injected curves: a TESS sector at
10-minute cadence with white noise of 1e-3; limb-darkened transits 1 to 7
hours long, 80% of them with a white-noise signal-to-noise of 10 to 44;
and stellar variability from two sinusoids with periods of 0.5 to 10 days
and amplitudes of 1e-4 to 5e-3. All use the biweight. The table shows how
often the best BLS peak is the injected period, and the median fraction
of depth and signal-to-noise kept (relative to the white-noise ideal):

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

A longer window keeps more of the depth, but the search does worse
afterwards, because the variability left in creates false peaks. With the
transits masked (over 1.2 durations), every window keeps the full depth,
and shorter windows leave less variability. So the recommended approach
has two steps:

1. **Search** after `Detrend::with_window_for_transit(longest)`, which
   uses a window of 2.5 times the longest duration you're searching for
   (`WINDOW_PER_DURATION`). That's in the middle of the range where the
   search did best, so a transit somewhat longer than expected still keeps
   most of its depth.
2. **Measure** the candidate after detrending again with its transits
   masked. `Detrend::flatten_around` does both steps with any search (see
   below).

`cargo validate accuracy window`
reruns the study.

## Masking a search's candidate

`Detrend::flatten_around(lc, &Masking::default(), search)` flattens the
light curve, runs `search` (any transit search, such as BLS or TLS, that
returns its best candidate's period, mid-transit time and duration),
masks the candidate's transits (`transit_mask`) and flattens again around
them. With `Masking::with_rounds` it repeats the search until it finds the
same transits twice. Results on the same 400 curves, starting from the
biweight at 2.5 durations and a blind BLS search. Mask and window sizes
are in units of the candidate's duration as the search found it (for the
"true transits" rows, the true duration):

| Flattening | Mask | Window | Depth kept | Signal to noise kept | 10th percentile | Search right |
|---|--:|--:|--:|--:|--:|--:|
| Blind | | 2.5 | 0.64 | 0.63 | 0.42 | 0.770 |
| Masked once | 1.2 | 2 | 1.00 | 0.97 | 0.60 | 0.767 |
| Masked once (default) | 1.5 | 2.5 | 1.01 | 0.98 | 0.67 | 0.770 |
| Masked once | 2 | 3 | 1.01 | 0.98 | 0.67 | 0.762 |
| Searched until settled | 1.2 | 2 | 1.00 | 0.99 | 0.58 | 0.765 |
| Searched until settled | 1.5 | 2.5 | 1.01 | 0.98 | 0.65 | 0.762 |
| True transits masked | 1.2 | 2 | 1.00 | 0.99 | 0.78 | 0.892 |
| True transits masked | 1.5 | 2.5 | 1.00 | 0.98 | 0.69 | 0.838 |

Masking once around the search's own candidate keeps the full depth and
98% of the signal-to-noise (against 63% without masking), as much as
masking the true transits. With a mask 1.5 durations long, the search's
approximate duration and mid-transit time cost nothing. The curves where
it does worse (the 10th percentile) are the ones where the search was
wrong to begin with, and masking can't fix that: a curve masked around a
candidate favours that candidate, so searching again never found the true
period where the blind search had missed it (0 of 92), and pushed a few
correct ones to aliases. Hence the default of one round: search blind,
then measure with the mask. The depth comes out about 1% too deep,
because the search also picks the mid-transit time and duration where the
noise happens to dip.

`cargo validate accuracy masking`
reruns the study.

## Whitening before the search

`Autoregressive::default().whiten(&flat)` fits an autoregressive model to
a flattened light curve (each flux predicted from up to 30 previous ones
within a run of regular cadence, with the order chosen by AIC) and
returns the residuals as `1 + e`. Noise correlated over a few cadences
comes out white, as in ARPS (Caceres et al. 2019). A transit comes out
shallower, scaled by the model's gain `1 - phi_1 - ... - phi_p`, with a
deeper first point.

A whitened transit no longer looks like a box, so there is also a matched
search. `Whitened::matched(&flat)` filters the flattened light curve back
through the model, so that summing a box over it gives the generalised
least-squares fit of a box in the model's noise.
`Matched::information(duration)` says how much more a box of that
duration is worth than its point count suggests: in correlated noise, a
transit's ingress and egress carry more of the signal than its middle.
`rustellar-bls`'s `Bls::search_whitened` takes both.

```rust,ignore
let white = Autoregressive::default().whiten(flat.light_curve())?;
let filtered = white.matched(flat.light_curve())?;
let result = Bls::new(filtered.light_curve())?.search_whitened(
    &periods,
    &durations,
    &options,
    |d| filtered.information(d),
)?;
```

This uses the window study's 400 curves, flattened (biweight at 2.5
durations) and searched, with white noise and with AR(1) noise at 0.5 and
0.8, each with and without its transit. The table shows how often the
search finds the transit above the SDE threshold that only 5% of the
transit-free curves reach. Columns: BLS on the flattened curve, BLS on
the whitened curve, and the matched search, with AIC choosing the order
up to 3, 10 or 30.

| Noise | Flattened | Whitened, up to 3 | Up to 10 | Up to 30 | Matched, up to 3 | Up to 10 | Up to 30 |
|---|--:|--:|--:|--:|--:|--:|--:|
| White | 0.453 | 0.370 | 0.398 | 0.512 | 0.323 | 0.422 | **0.555** |
| AR(1) 0.5 | **0.318** | 0.278 | 0.302 | 0.302 | 0.258 | 0.275 | 0.305 |
| AR(1) 0.8 | 0.235 | **0.242** | 0.225 | 0.228 | 0.225 | 0.203 | 0.185 |

With a long model, the matched search finds the most transits on white
noise, 23% more than searching the flattened curve, at the same false
alarm rate, because a long model removes what the window left of the
star. In red noise, neither whitening nor the matched search beats
searching the flattened curve: the window has already removed most of the
slow noise, and fitting the rest adds as much noise as it removes. At a
fixed SDE of 7 every whitened search looks better, but only because the
false alarms rise too (from 0.098 to 0.152 on white noise, matched up to
order 10). So whitening and the matched search are worth trying at order
30, but aren't a default step.

`cargo validate accuracy whitening`
reruns the study.

## Splines or the biweight

The splines are alternatives to the biweight, not replacements. On the
window study's 400 curves (transits on spotted stars, above), the table
gives each method's depth kept, signal-to-noise kept, and how often a
blind BLS search afterwards finds the injected period with SDE above 7,
with wotan's methods on the same curves for comparison:

| Method | Knots or window, in durations | Depth kept | Signal to noise kept | Search finds it |
|---|--:|--:|--:|--:|
| Biweight | 2.5 | 0.64 | 0.63 | 0.500 |
| Spline | 1.5 | 0.42 | 0.42 | 0.497 |
| Spline | 2.5 | 0.62 | 0.60 | 0.443 |
| Robust (clipped) spline | 1.5 | 0.43 | 0.42 | 0.453 |
| Huber spline | 1.5 | 0.43 | 0.42 | 0.445 |
| wotan biweight | 2.5 | 0.64 | 0.63 | 0.495 |
| wotan `rspline` | 1.5 | 0.72 | 0.71 | 0.422 |
| wotan `rspline` | 2.5 | 0.91 | 0.87 | 0.400 |
| wotan `hspline` | 1.5 | 0.47 | 0.46 | 0.468 |

The biweight stays the default for searching. The plain spline with knots
1.5 durations apart does just as well, and every spline searched best at
that spacing (1 and 4 durations did worse). wotan's `rspline` keeps the
most depth because it only clips points below the spline, which removes
the transit's own points from the fit. But it removes noise dips too, so
its search finds fewer transits. Clipping this crate's robust spline the
same way (2 sigma below, 3 above) gave the same result: depth kept 0.79 at
2.5 durations, search 0.385. The robust splines are meant for light
curves with flares and outliers, which these curves don't have.

`cargo validate accuracy methods`
reruns the study (its README says how wotan's columns were made).

## Speed

Seconds on an AMD Ryzen 7 5700G (8 cores, 16 threads), Rust 1.99; wotan
1.10 (numba 0.64, after its first call) runs on one thread. The light
curves are a Kepler quarter (4,265 points, 1-day window) and a TESS
sector at 2 minutes (18,139 points) and at 20 seconds (108,878 points),
both with half-day windows.

| Biweight | wotan 1.10 | rustellar-detrend, 1 thread | rustellar-detrend, 16 threads |
|---|--:|--:|--:|
| Kepler quarter | 0.0105 | 0.00430 | 0.00120 |
| TESS, 2 minutes | 0.180 | 0.0901 | 0.0109 |
| TESS, 20 seconds | 5.84 | 2.90 | 0.328 |

Each method on the 2-minute sector, one thread (wotan's default settings):

| | wotan 1.10 | rustellar-detrend |
|---|--:|--:|
| Median | 0.0582 | 0.00250 |
| Mean | 0.00958 | 0.00513 |
| Biweight | 0.180 | 0.0902 |
| Welsch | 0.321 | 0.206 |
| Andrews' sine wave | 0.323 | 0.198 |
| Huber (fixed scale) | - | 0.0727 |
| Huber's proposal 2 | 10.3 | 0.148 |
| Hodges-Lehmann | 9.19 | 0.872 |
| Trimmed mean | 0.196 | 0.0175 |
| Winsorized mean | 0.196 | 0.0215 |

On the same sector:

- Least-squares spline (`Method::Spline`, knots half a day apart): scipy's
  `LSQUnivariateSpline` 0.00259, this crate 0.00244.
- Robust spline (sigma 3): wotan's `rspline` 0.447, this crate 0.00616.
- Huber spline (c 1.345): statsmodels' `RLM` 0.765 (with the design
  matrix built beforehand), wotan's `hspline` (scikit-learn) 0.269, this
  crate 0.0114.
- Autoregressive fit (AIC up to order 30, including the residuals):
  statsmodels' `ar_select_order` plus refit 0.303, this crate 0.0102,
  since every order's fit comes from one factorisation.

`Method::Huber` uses a fixed scale of c times the MAD (c 1.28, like
wotan's `huber_psi`); `Method::HuberJoint` estimates the scale together
with the location, like wotan's `huber`.

`Detrend::flatten_many` flattens many light curves with the same
settings, each with its own mask, and gives each curve exactly the result
`Detrend::flatten` would. It flattens half as many curves at once as
there are threads, two threads each, so small curves, which share out
poorly among many threads, gain the most. The biweight, seconds per light
curve on 16 threads:

| | Curves | One after another | `flatten_many` |
|---|--:|--:|--:|
| Kepler quarter | 64 | 0.00136 | 0.000468 |
| TESS, 2 minutes | 32 | 0.0117 | 0.00934 |
| TESS, 20 seconds | 8 | 0.338 | 0.307 |

There is no GPU version: the estimators are solved in double precision
to be correctly rounded, and a consumer GPU's double precision (1/64 of
its single precision on an RTX 3060 Ti, about 0.25 TFLOPS) is below what
the eight cores of the CPU above reach.

`cargo validate speed detrend` reruns these tables, printing the
rustellar-detrend columns beside wotan's recorded times. `cargo bench -p
rustellar-detrend` runs the Rust timings alone.

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

MIT or Apache-2.0.
