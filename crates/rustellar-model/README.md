# rustellar-model

Transit light curves in pure Rust: how a star dims as a planet on a
circular or eccentric orbit passes in front of it. It uses the analytic
model of Mandel & Agol (2002) for uniform, linear and quadratic limb
darkening, with exposure averaging for long-cadence data (Kipping 2010).

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `model` feature.

```toml
[dependencies]
# Through the umbrella crate (what most projects want):
rustellar = { version = "0.1", features = ["model"] }
# Or this crate alone:
rustellar-model = "0.1"
```

This crate re-exports the core types it uses (`LightCurve`, `Epoch`,
`TimeDelta` and their errors), so you don't need to add `rustellar-core`
yourself or keep its version in step.

**Status:** version 0.1. The API may still change before 1.0.

## Statement of need

Fitting transits, searching for them and vetting candidates all evaluate a
transit model many times. The established implementations, batman (Kreidberg
2015), PyTransit (Parviainen 2015) and exoplanet-core (Agol, Luger &
Foreman-Mackey 2020), are Python packages with compiled kernels, so a Rust
pipeline has to call into Python or write its own model. This crate is that
model in Rust. It depends only on `rustellar-core`, its parameter types
reject invalid values when they're built, and its accuracy is measured
against all three packages and an exact numerical integral.

## Quick start

```rust
use rustellar_model::{
    Epoch, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TimeDelta,
    TransitParams,
};

let params = TransitParams::new(
    Epoch::from_days(1000.0)?,           // mid-transit time
    TimeDelta::from_days(3.0)?,          // orbital period
    RadiusRatio::new(0.1)?,              // planet radius / star radius
    ScaledSemiMajorAxis::new(10.0)?,     // orbit radius / star radius
    ImpactParameter::new(0.3)?,          // closest approach to the star's centre
    LimbDarkening::quadratic(0.3, 0.25)?,
)?;

let flux = params.flux_at(Epoch::from_days(1000.0)?);
let t14 = params.t14().expect("this planet transits");
assert!(flux < 0.99 && (t14.as_hours() - 2.431).abs() < 1e-3);
# Ok::<(), Box<dyn std::error::Error>>(())
```

For many parameter sets on the same times, as in a fit, use `TransitModel`,
which does no allocation per call. Add an `Exposure` to average over each
measurement's integration time.

## Accuracy

The flux is checked against exact values at 36,742 points: 60-digit
evaluations of Mandel & Agol's closed forms, which are a different
formulation from the one used here, and are cross-checked against direct
integration. There are 32,768 random points over the whole valid range, and
3,974 on and around every line where the formula changes case. The line
points sit 1 to 10 rounding steps from each line and up to 0.1 away, with
separations down to the smallest double.

| Radius ratio k | Worst difference from exact |
|---|---|
| 1e-6 to 1 | 6.7e-16 |
| 1 to 10 | 1.2e-15 |
| 10 to 100 | 1.4e-14 |
| 100 to 1,000 | 9.1e-14 |
| 1,000 to 10,000 | 9.6e-13 |

Above k = 1 the error grows in proportion to k, at about 1.5e-16 k.

The other packages are compared at 1,932 points, each a (k, z, u1, u2), with
many on and near the case lines. The differences are from an exact flux by
adaptive quadrature, good to about 5e-12. At the point where this crate
differs most from it, its value equals the 60-digit one to the last digit:

| | Worst difference | Points off by more than 1 ppm |
|---|---|---|
| rustellar-model | 5.0e-12 (the quadrature's own error) | 0 |
| exoplanet-core 0.4.0 | 0.73 (one point just below z = k) | 1 |
| batman 2.5.1 | 0.73 (on z = 1 − k for k > 0.5) | 49 |
| PyTransit 2.6.18 | 0.10, and NaN at 12 points (all on z = 1 − k) | 15 |

Away from those lines all four agree. batman and PyTransit agree to their
stated precision of a few 1e-8, because they use polynomial approximations to
the elliptic integrals. exoplanet-core agrees to 1.4e-15 for k < 1, and its
own error grows as k^3 above that, to 2.6e-10 at k = 100.

Light curves match batman's orbit geometry to 7e-11 in separation over 1,984
transits. Exposure averages match PyTransit sample for sample.

Eccentric orbits (`with_eccentricity`, Kepler's equation by Halley's method)
are checked on six orbits from e = 0.05 to 0.9 against 40-digit mpmath: sky
separations within 6 rounding steps of a/R*, contact times within 1e-11 days,
and the flux within 2.5e-8 of batman 2.5.1's, which is batman's own
tolerance. Every one of these comparisons runs in the test suite.

## Speed

The setup for these timings:

- one thread, on an AMD Ryzen 7 5700G desktop CPU, Rust 1.99, numpy 2.4;
- the hot Jupiter of the quick start, with u = (0.4, 0.25);
- times spread evenly over ±0.1 days around mid-transit, so about half are
  in transit.

`cargo validate speed model` reruns this crate's timings and prints them
beside the other packages' recorded times. `cargo bench -p
rustellar-model` runs the Rust timings alone.

| | 100 times | 1,000 | 10,000 | 100,000 |
|---|--:|--:|--:|--:|
| rustellar-model | 3.9 µs | 40 µs | 0.39 ms | 3.9 ms |
| batman 2.5.1 | 7.5 µs | 65 µs | 0.63 ms | 6.4 ms |
| PyTransit 2.6.18 | 8.7 µs | 50 µs | 0.46 ms | 4.6 ms |
| rustellar-model, flux only (no orbit) | 3.1 µs | 30 µs | 0.30 ms | 3.0 ms |
| exoplanet-core 0.4.0, flux only (no orbit) | 18 µs | 48 µs | 0.34 ms | 3.9 ms |

Kepler long cadence (29.4 min), 1,000 times:

| Samples per exposure | 5 | 15 | 30 |
|---|--:|--:|--:|
| rustellar-model | 0.17 ms | 0.50 ms | 1.0 ms |
| batman 2.5.1 | 0.33 ms | 0.98 ms | 2.0 ms |
| PyTransit 2.6.18 | 0.20 ms | 0.59 ms | 1.2 ms |

A single flux in transit takes 42 ns with the planet fully on the disk and
104 ns across the limb, at full double precision:

- **rustellar-model** computes all three elliptic integrals in one AGM
  iteration (Bulirsch's `cel`, as in Agol et al. 2020).
- **batman and PyTransit** use polynomial approximations good to about 1e-8.
- **exoplanet-core** uses the same formulation at full precision.

This crate is the fastest at every size. Its light curves are 1.18 to 2.2
times as fast as PyTransit's, and its flux alone is 1.13 to 5.8 times as
fast as exoplanet-core's (which takes separations as input, so skips the
orbit). Two things help: a time clearly outside the transit gets a flux of
exactly 1 without computing a sine and cosine, and across the limb the
three elliptic integrals share one calculation.

With exposure averaging it is also the fastest, 15 to 20% ahead of
PyTransit. Each exposure needs only one sine and cosine (the sample
phases are rotated from it), and an exposure entirely out of transit is
skipped. For the same accuracy, the midpoint rule used here needs about
half as many samples as batman's.

## Limits

The planet is assumed to emit no light, and the radius ratio can be at
most 10,000. On an eccentric orbit, `b` is the impact parameter at
mid-transit and `t0` the time of inferior conjunction, and periastron must
stay outside the star (`e < 1 - 1/a`).

## References

- Agol, E., Luger, R. & Foreman-Mackey, D. 2020, AJ, 159, 123
- Kipping, D. M. 2010, MNRAS, 408, 1758
- Kreidberg, L. 2015, PASP, 127, 1161
- Mandel, K. & Agol, E. 2002, ApJ, 580, L171
- Parviainen, H. 2015, MNRAS, 450, 3233
- Winn, J. N. 2010, in Exoplanets, ed. S. Seager (Univ. Arizona Press), 55

## License

MIT or Apache-2.0.
