# rustellar-vet

Vetting transit candidates in pure Rust. It reports the evidence for and
against a search's period being the real one, and leaves the decision to
you.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `vet` feature.

```toml
[dependencies]
# Through the umbrella crate (what most projects want):
rustellar = { version = "0.1", features = ["vet"] }
# Or this crate alone:
rustellar-vet = "0.1"
```

This crate re-exports the core types it uses (`LightCurve`, `Epoch`,
`TimeDelta` and their errors), so you don't need to add `rustellar-core`
yourself or keep its version in step.

**Status:** version 0.1. The API may still change before 1.0.

## Statement of need

A transit search reports one best period, but sometimes the true period is
a simple multiple or fraction of it: half (every other predicted transit
is empty), double (every other real transit was skipped), or a third,
three times, two thirds or three halves. For a weak signal the chi-squared
often prefers the wrong one, because the noise genuinely favours it. What
separates them is the evidence transit by transit: whether odd and even
transits have the same depth, how many predicted transits land on data,
and the depth of each one.

For a candidate and each of its aliases, this crate gives the fit and that
evidence as numbers with their errors, laid out in a table. It applies no
thresholds of its own. It takes a `Candidate` from any search: from
`rustellar-bls` with the `bls` feature, from `rustellar-tls` with the `tls`
feature, or built by hand from a period, a mid-transit time and a
duration.

## What it does

- `Candidate`: a period, mid-transit time, duration and optional depth.
  `Candidate::duty_cycle` gives the duration divided by the period.
- `fit_depth`: fits the transit depth and its standard error at the
  candidate's period, mid-transit time and duration. The model is
  baseline × (1 − depth × shape), fitted by weighted least squares, where
  the shape is a box or a limb-darkened transit (by default a small planet
  half-way to the limb of a Sun-like star, in the Kepler band). On a
  noiseless transit the depth comes back exact to rounding, and over 4,000
  noisy fits the stated error matches the actual scatter to within 4%.
- `refine`: finds the mid-transit time and duration that fit best near the
  candidate's, keeping its period. For a box this is exact (no point on a
  301 × 301 grid does better); for a limb-darkened shape it matches the
  best of a dense grid to one part in a million. `refine_with_threads`
  does the same on several threads, with the same result.
- `evidence`: for each predicted transit, the number of points, the
  coverage and the depth, plus a comparison of the odd and even transits'
  depths. The error on their difference allows for the correlation from
  their shared baseline. Over 1,000 noise draws of a planet, the odd-even
  significance has mean 0 and standard deviation 1 (within the draws'
  scatter), while a candidate at half the true period, or an eclipsing
  binary, stands out by more than 10 sigma.
- `signal_to_noise`: the signal-to-noise ratio using the given errors
  (white noise) and using the noise measured out of transit (red noise,
  after Pont, Zucker & Queloz 2006), for the candidate and for each
  transit. Also the largest single transit's share of the total depth, and
  a chi-squared test of whether all transits have the same depth. Over 300
  draws with red noise as large as the white, the depth scatters three
  times its white-noise error and within 12% of its red-noise error. In
  white noise the depth test's chi-squared has the expected mean and
  variance.
- `aliases`: refits the candidate at a third, half, two thirds, one, three
  halves, two and three times its period (or any ratios you give). Each
  row has its fit, evidence, signal-to-noise and chi-squared relative to
  the original candidate, plus the depth at transits that only one of the
  two periods predicts. It prints as a table. With a planet injected at
  each of those ratios, the true period is the best row every time, by a
  chi-squared margin of more than 50.
- `secondary`: slides a box the length of the transit across every phase
  at least one duration away from the transit, leaving the transit's
  points out. It reports the dip at half the period with an upper limit,
  the deepest dip, the strongest brightening, and the spread of the
  scan's significances (the light curve's own noise level). Each box
  matches a direct box fit to 1e-12, and in white noise the significance
  at half the period has mean 0 and standard deviation 1. A `depth_ratio`
  (secondary over primary) near 1 means the two eclipses are alike.
- `OddEven::half_period`: whether the shallower of the odd and even depths
  looks empty (near zero and well below the deeper one), which points to
  twice the period rather than an eclipsing binary. `depth_ratio` gives
  the number.
- `multiples`: splits the transits into `n` groups for every `n` up to a
  limit (transit `e` goes in group `e mod n`), fits each group's depth,
  tests whether the depths are equal, and reports which multiple of the
  period the deep groups point to. At twice or three times the true
  period it names the right multiple and starting group. For a real planet
  every split is equal, and over 300 noise draws the test's mean matches
  its degrees of freedom.
- `ellipsoidal`: fits sinusoids at the orbital period and half of it,
  with both eclipses masked: the signature of a close binary (and of
  reflection and Doppler beaming). Injected signals come back to 1e-12,
  and in white noise each term divided by its error is standard normal.
- `search_period`: searches period and mid-transit time within a few
  percent of a candidate's (or an alias's), in steps that move the
  outermost transits by an eighth of a duration, and returns the whole
  spectrum.
- `flag_periods` and `flag_times`: check the candidate against what you
  know about your data. A `SuspectPeriod` is a period a real signal is
  unlikely to share (a spacecraft orbit, a rotation, a day); it is matched
  at its multiples and fractions within a relative tolerance, and each
  match gives the ratio and offset. A `TimeWindow` is a span where a
  transit shouldn't fall (an instrument event, the edge of a gap, flagged
  data); each predicted transit that overlaps one is listed with the
  windows it hits. Nothing instrument-specific is built in: you supply the
  periods and windows. Both are tested against a brute-force check of
  every ratio and window.

## Pointing back to the true period

This uses the fourteen TLS comparison sets from `rustellar-benchmarks`:
the original five plus the harder regimes (long, ultra-short, deep,
grazing and eccentric transits, 20-second cadence, gaps, flagged points
and spotted stars). Every planet transitleastsquares 1.32 found was given
to `aliases` at a third, half, two thirds, three halves, twice and three
times its period, the way a search locked onto an alias would report it.
That makes 7,849 candidates, each with transitleastsquares' mid-transit
time and duration and the default limb-darkened shape. How often the row
with the lowest chi-squared was the true period
(`cargo validate accuracy aliases` reruns this):

| Signal to noise at the true period | Candidates | Pointed to the true period |
|---|---|---|
| below 7 | 1,323 | 1,260 (95.2%) |
| 7 to 10 | 1,684 | 1,684 (100%) |
| 10 to 20 | 2,959 | 2,959 (100%) |
| 20 to 50 | 1,046 | 1,046 (100%) |
| 50 and above | 763 | 763 (100%) |

The other 74 candidates had mid-transit times more than a duration away
from the true transits, and are left out.

transitleastsquares itself reported an alias on 72 curves. Of the 37 at
twice the true period (median signal-to-noise 12 at the true one), the
table points 28 back to the right period, and 6 of the 9 at three times.
The 26 at a third, half, two thirds or three halves have a median
signal-to-noise of only 2.8 to 4.2 at the true period, mostly too faint
for any method to settle, and the table gets 3 of them right. With one
depth and one baseline, the lowest chi-squared is also the highest
signal-to-noise, so choosing by either picks the same row.

## Against LEO-Vetter and exovetter

This crate, LEO-Vetter 1.2.0 and exovetter 0.0.16 were run on the same
620 simulated light curves (27 days at 2 minutes, a 3-day period, a
0.1-day box transit, noise of 1e-3 per point). The planets have a 3e-3
transit. One set of binaries alternates eclipses of 3e-3 and 2.7e-3 (3.8
standard errors apart); another has a 1e-3 secondary eclipse at half the
period and ellipsoidal variation of 2e-4. One set of planets has red
noise as large as the white, and one has gaps. The test
`tests/references.rs` builds the same curves and checks every number
below.

| | rustellar-vet | LEO-Vetter 1.2.0 | exovetter 0.0.16 |
|---|---|---|---|
| Odd-even, 200 planets: mean absolute significance (0.80) | 0.80 | 0.80 | 0.056 |
| Odd-even, 200 planets: spread of the significance (1) | 1.03 | 1.02 | |
| Odd-even, 100 binaries: mean significance | -3.81 | -3.66 | 0.21 (unsigned) |
| Secondary, 100 binaries: found at half the period | 100 | 100 | 100 |
| Secondary: mean significance (31 in white noise) | 30.4 | 14.0 | -7,469 |
| Ellipsoidal amplitude, 100 binaries (2e-4) | 1.99e-4 | 2.83e-4 | 2.77e-4 |
| Ellipsoidal above 3 sigma, 200 planets (1.1%: 2.2) | 2 | 8 | 4 |
| Depth scatter over its stated error, white noise (1) | 1.00 | 1.03 | |
| Depth scatter over its stated error, red noise (1) | 1.03 | 1.05 | |
| Transits with data, 20 curves with gaps | equal | equal | |

LEO-Vetter's odd-even test and depth errors are as well calibrated as this
crate's. Its secondary significance treats the binary's ellipsoidal
variation as red noise, which halves it compared with the white-noise
value. Both packages' sinusoid fits (SWEET) mask only the primary
eclipse, so the secondary pulls their ellipsoidal amplitude 40% too high;
this crate masks both eclipses. exovetter uses the scatter of single
points as the odd-even error instead of the error on the depth, so its
significance is too small by roughly the square root of the number of
points in a transit. Its ModShift significances are also not in standard
errors, because it convolves each bin's summed flux rather than its mean.
Neither package has an equivalent of the upper limits, the scan's spread
or the equal-depth test.

## Speed

Milliseconds on one thread of an AMD Ryzen 7 5700G, for a 3.7-day planet
in a TESS sector at 2-minute cadence (19,440 points) and in four years of
Kepler long cadence (70,128 points), with the default limb-darkened shape
and with a box. `cargo validate speed vet` reruns this table, and
`cargo bench -p rustellar-vet` runs the timings alone.

| Step | TESS, limb-darkened | TESS, box | Kepler, limb-darkened | Kepler, box |
|---|---|---|---|---|
| `fit_depth` | 0.22 | 0.19 | 0.83 | 0.71 |
| `refine` | 153 | 1.2 | 558 | 18 |
| `evidence` | 0.59 | 0.56 | 3.2 | 3.1 |
| `signal_to_noise` | 0.93 | 0.90 | 6.4 | 6.3 |
| `aliases` (seven ratios) | 16 | 15 | 91 | 87 |
| `secondary` | 0.59 | | 4.4 | |

The limb-darkened `refine` scores each of its 5,043 trials using only the
points inside that trial's transit, so its time doesn't grow with the
number of out-of-transit points. `refine_with_threads` shares those
trials among threads: on 16 threads it takes 16 ms on TESS and 57 ms on
Kepler, ten times faster than on one. There is no GPU version: nearly all
the time goes on the limb-darkened flux's sines, cosines and square roots,
which a GPU rounds differently, and a consumer GPU is slower than the CPU
in double precision.

## Quick start

```rust
use rustellar_vet::{Candidate, Epoch, TimeDelta};

let candidate = Candidate::new(
    TimeDelta::from_days(4.7)?,
    Epoch::from_days(1326.3)?,
    TimeDelta::from_hours(2.9)?,
)?
.with_depth(1e-3)?;
assert_eq!(candidate.depth(), Some(1e-3));
# Ok::<(), Box<dyn std::error::Error>>(())
```

## License

MIT or Apache-2.0.
