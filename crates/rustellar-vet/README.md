# rustellar-vet

Vetting transit candidates in pure Rust: the evidence that a search's
period is the real one, shown rather than decided.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `vet` feature.

```toml
[dependencies]
# Through the umbrella, which most projects want:
rustellar = { version = "0.1", features = ["vet"] }
# Or this crate alone:
rustellar-vet = "0.1"
```

On its own it re-exports the core types it takes and returns
(`LightCurve`, `Epoch`, `TimeDelta` and their errors), so
`rustellar-core` need not be listed, and the two always match.

**Status:** in development, not yet released.

## Statement of need

A transit search reports one best period. Sometimes the true period is a
simple ratio of it: half of it (every other predicted transit is empty),
double (every other real transit was skipped), or a third, three times,
two thirds or three halves. A weak signal's chi-squared often prefers the
wrong one, because the noise really does favour it. What tells them apart
is the evidence transit by transit: whether the odd and even transits have
the same depth, how many predicted transits fall where there are data, and
each transit's own depth.

This crate will give, for a candidate and each of its aliases, the fit and
that evidence as numbers with their errors, in a table, with no hidden
thresholds. It takes a `Candidate` from any search: from `rustellar-bls`
with the `bls` feature, from `rustellar-tls` with the `tls` feature, or
built from a period, a mid-transit time and a duration.

## So far

- `Candidate`: a period, mid-transit time, duration and optional depth.
- `fit_depth`: the depth at that period, mid-transit time and duration,
  with its standard error, for a limb-darkened shape (the default: a small
  planet half-way to the limb of a Sun-like star, in the Kepler band) or a
  box. The model is
  baseline times (1 - depth times shape), fitted by weighted least squares;
  the depth comes back to rounding from a noiseless transit, and its
  stated error matches the scatter of 4,000 noisy fits to within 4%.
- `refine`: the mid-transit time and duration that fit best near the
  candidate's, at its period. For a box this is exact (no point of a
  301 by 301 grid does better); for a limb-darkened shape it matches a
  dense grid's best to a part in a million.
- `evidence`: each predicted transit's points, coverage and own depth,
  and the odd and even transits' depths compared. The difference's error
  counts the two depths' correlation through their shared baseline; on
  1,000 noise draws of a planet the odd-even significance has mean 0 and
  spread 1 to within the draws' own scatter, and a candidate at half the
  true period or an eclipsing binary stands out by more than 10 sigma.
- `signal_to_noise`: the signal to noise with the errors given (white)
  and with the noise measured out of transit (red, after Pont, Zucker &
  Queloz 2006), for the candidate and for each transit; the largest
  transit's share of the depth; and a chi-squared test of whether every
  transit has one depth. In red noise as large as the white, over 300
  draws, the depth scatters three times its white error and within 12% of
  its red error; in white noise the depth test's chi-squared has the mean
  and variance it should.
- `aliases`: the candidate at a third, half, two thirds, one, three
  halves, two and three times its period (or any ratios), each with its
  fit, evidence and signal to noise, its chi-squared against the
  candidate's, and the depth at the transits only one of the two periods
  predicts; printed as a table. With a planet at each of those ratios of
  the period searched, the true one is the best row every time, by a
  chi-squared of more than 50.
- `secondary`: a box of the transit's duration slid over every phase at
  least a duration from the transit, the transit's points left out: the
  dip at half the period with its upper limit, the deepest dip and the
  strongest brightening, and the spread of the scan's significances (the
  light curve's own noise scale). Each box equals a box fit to 1e-12, and
  in white noise the significance at half the period has mean 0 and
  spread 1.
  Its `depth_ratio` (at half over the primary) near 1 means the two
  eclipses are alike.
- `OddEven::half_period`: whether the shallower of the odd and even
  depths looks empty (near zero and well below the deeper), which means
  twice the period rather than an eclipsing binary; `depth_ratio` gives
  the number.
- `multiples`: the transits split `n` ways for every `n` up to a limit
  (transit `e` in group `e mod n`), each group's depth, a chi-squared test
  of equal depths, and the multiple of the period the deep groups point
  to. At twice or three times the true period the right multiple and
  starting group are named; for a planet every split is equal, and the
  test's mean matches its degrees of freedom over 300 noise draws.
- `ellipsoidal`: sinusoids once and twice an orbit fitted out of both
  eclipses, the signature of a close binary (and of reflection and
  Doppler beaming). Injected terms come back to 1e-12; each term over its
  error is a standard normal in white noise. `Candidate::duty_cycle` gives
  the duration over the period.
- `search_period`: the period searched within a few percent of a
  candidate's (or an alias's), with the mid-transit time, in steps that
  move the outermost transits by an eighth of a duration; the whole
  spectrum is returned.
- `flag_periods` and `flag_times`: what you know about your data, checked
  against the candidate. A `SuspectPeriod` is a period a real signal is
  unlikely to share (a spacecraft's orbit, a rotation, a day), matched at
  its multiples and fractions within a relative tolerance; each match
  gives the ratio and the offset. A `TimeWindow` is a span a transit
  should not fall in (an instrument event, a gap's edge, flagged data);
  each predicted transit that overlaps one is listed with the windows it
  overlaps. Nothing is built in for any instrument: the periods and
  windows are yours. Checked against trying every ratio and every window.

## Pointing back to the true period

On the fourteen TLS comparison sets (`cargo validate accuracy aliases`
in `rustellar-benchmarks`: the original five and the harder regimes,
long, ultra-short, deep, grazing and eccentric transits, 20-second
cadence, gaps, flagged points and spotted stars), every planet
transitleastsquares 1.32 found was also given to `aliases` at a third,
half, two thirds, three halves, twice and three times its period, as a
search locked onto an alias would report it: 7,849 candidates in all,
with TLS's mid-transit time and duration and the default limb-darkened
shape. The row of lowest chi-squared was the true period:

| Signal to noise at the true period | Candidates | Pointed to the true period |
|---|---|---|
| below 7 | 1,323 | 1,260 (95.2%) |
| 7 to 10 | 1,684 | 1,684 (100%) |
| 10 to 20 | 2,959 | 2,959 (100%) |
| 20 to 50 | 1,046 | 1,046 (100%) |
| 50 and above | 763 | 763 (100%) |

(The 74 other candidates have mid-transit times more than a duration off
the true transits.) TLS itself reported an alias on 72 curves. On the 37
at twice the period (median signal to noise 12 at the true one) the
table points 28 back to it, and 6 of the 9 at three times; the 26 at a
third, half, two thirds or three halves have median signals to noise of
2.8 to 4.2 at the true period, mostly too faint for any table to settle, and the
table is right on 3 of them. For one depth and one
baseline the lowest chi-squared is the highest signal to noise, so
choosing by either gives the same row.

## Against LEO-Vetter and exovetter

On the same 620 simulated light curves (27 days at 2 minutes, a 3-day
period, a 0.1-day box transit, noise of 1e-3 a point), this crate,
LEO-Vetter 1.2.0 and exovetter 0.0.16. The planets have a 3e-3 transit;
one set of binaries has eclipses of 3e-3 and 2.7e-3 in turn (3.8 standard
errors apart), another a 1e-3 secondary eclipse at half the period and
ellipsoidal variation of 2e-4; one set of planets has red noise as large
as the white; one has gaps. The test (`tests/references.rs`) makes the
very same curves and checks every number below.

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
crate's. Its secondary significance counts the binary's ellipsoidal
variation as red noise, so it is half the white-noise value. Both packages'
sinusoid fits (SWEET) leave out only the primary eclipse, so the
secondary pulls their ellipsoidal amplitude 40% high; this crate leaves
out both eclipses. exovetter's odd-even error is the scatter of single
points rather than the error of the depth, so its significance is too
small by about the square root of the points in a transit, and its
ModShift significances are not in standard errors (it convolves each
bin's sum of flux, not its mean). The upper limits, the scan's spread and
the one-depth test have no counterpart in either.

## Speed

Milliseconds on one thread of an AMD Ryzen 7 5700G, for a 3.7-day planet
in a TESS sector at 2 minutes (19,440 points) and in four years of Kepler
long cadence (70,128 points), with the default limb-darkened shape and a
box (`cargo bench -p rustellar-vet`, or `cargo validate speed vet` for
this table):

| Step | TESS, limb-darkened | TESS, box | Kepler, limb-darkened | Kepler, box |
|---|---|---|---|---|
| `fit_depth` | 0.22 | 0.19 | 0.83 | 0.71 |
| `refine` | 153 | 1.2 | 558 | 18 |
| `evidence` | 0.59 | 0.56 | 3.2 | 3.1 |
| `signal_to_noise` | 0.93 | 0.90 | 6.4 | 6.3 |
| `aliases` (seven ratios) | 16 | 15 | 91 | 87 |
| `secondary` | 0.59 | | 4.4 | |

The limb-darkened `refine` scores its 5,043 trials on the points within
each trial's own transit only, so it does not grow with the points out of
transit.

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

MIT or Apache-2.0, at your option.
