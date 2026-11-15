# rustellar-model

Transit light curves in pure Rust: how a star dims as a planet on a circular
orbit passes in front of it. The model is the analytic one of Mandel & Agol
(2002) for uniform, linear and quadratic limb darkening, with exposure
averaging for long-cadence data (Kipping 2010).

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `model` feature.

**Status:** in development, not yet released.

## Statement of need

Fitting transits, searching for them and vetting candidates all evaluate a
transit model many times. The established implementations, batman (Kreidberg
2015), PyTransit (Parviainen 2015) and exoplanet-core (Agol, Luger &
Foreman-Mackey 2020), are Python packages with compiled kernels, so a Rust
pipeline has to call into Python or write its own model. This crate is that
model for Rust, with no dependencies, inputs checked by their types, and its
accuracy measured against all three packages and an exact numerical integral.

## Quick start

```rust
use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

let params = TransitParams::new(
    Epoch::from_btjd(1000.0)?,           // mid-transit time
    TimeDelta::from_days(3.0)?,          // orbital period
    RadiusRatio::new(0.1)?,              // planet radius / star radius
    ScaledSemiMajorAxis::new(10.0)?,     // orbit radius / star radius
    ImpactParameter::new(0.3)?,          // closest approach to the star's centre
    LimbDarkening::quadratic(0.3, 0.25)?,
)?;

let flux = params.flux_at(Epoch::from_btjd(1000.0)?);
let t14 = params.t14().expect("this planet transits");
```

For many parameter sets on the same times, as in a fit, use `TransitModel`,
which does no allocation per call. Add an `Exposure` to average over each
measurement's integration time.

## Accuracy

There are 1,932 points, each a (k, z, u1, u2), chosen to cover every region
of the formula, with many on and near the lines where it changes case. The
differences are from the exact flux (adaptive quadrature, good to about
1e-12):

| | Worst difference | Points off by more than 1 ppm |
|---|---|---|
| rustellar-model | 1.5e-11 | 0 |
| exoplanet-core 0.4.0 | 0.73 (one point just below z = k) | 1 |
| batman 2.5.1 | 0.73 (on z = 1 − k for k > 0.5) | 49 |
| PyTransit 2.6.18 | 0.10, and NaN at 12 points (all on z = 1 − k) | 15 |

Away from those lines all four agree. batman and PyTransit agree to their
stated precision of a few 1e-8, because they use polynomial approximations to
the elliptic integrals. exoplanet-core agrees to 1e-13. The largest
differences from exoplanet-core over 28,000 random points across the whole
valid range are:

- 1e-13 for k < 1;
- 1.4e-12 for k < 10;
- 1.5e-9 at 100, the largest k allowed.

Light curves match batman's orbit geometry to 7e-11 in separation over 1,984
transits. Exposure averages match PyTransit sample for sample. Every one of
these comparisons runs in the test suite.

## Speed

The setup for these timings:

- one thread, on an AMD Ryzen 5000-series laptop CPU;
- the hot Jupiter of the quick start, with u = (0.4, 0.25);
- times spread evenly over ±0.1 days around mid-transit, so about half are
  in transit.

`cargo bench -p rustellar-model` runs the Rust side.

| | 100 times | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| rustellar-model | 23 µs | 232 µs | 2.2 ms | 24 ms |
| batman 2.5.1 | 8 µs | 91 µs | 0.77 ms | 8.9 ms |
| PyTransit 2.6.18 | 12 µs | 57 µs | 0.64 ms | 5.6 ms |
| exoplanet-core 0.4.0 (flux only, no orbit) | 28 µs | 60 µs | 0.56 ms | 7.0 ms |

Kepler long cadence (29.4 min), 1,000 times:

| Samples per exposure | 5 | 15 | 30 |
|---|---|---|---|
| rustellar-model | 1.3 ms | 3.6 ms | 7.1 ms |
| batman 2.5.1 | 0.39 ms | 1.2 ms | 2.2 ms |
| PyTransit 2.6.18 | 0.24 ms | 0.70 ms | 1.3 ms |

This crate is currently 2.5 to 4 times slower than the others:

- **rustellar-model** computes three separate Carlson elliptic integrals for
  each point in transit, which takes about 450 ns.
- **batman and PyTransit** use polynomial approximations good to about 1e-8.
- **exoplanet-core** reaches full precision faster with the formulation of
  Agol et al. (2020).

Moving to that formulation is planned. At the same accuracy, the midpoint
rule used here for exposures needs about half as many samples as batman's
rule.

## Limits

Circular orbits only, no light from the planet, and a radius ratio of at most
100. See [SCOPE.md](SCOPE.md) for what is planned and what is left to other
crates.

## References

- Agol, E., Luger, R. & Foreman-Mackey, D. 2020, AJ, 159, 123
- Kipping, D. M. 2010, MNRAS, 408, 1758
- Kreidberg, L. 2015, PASP, 127, 1161
- Mandel, K. & Agol, E. 2002, ApJ, 580, L171
- Parviainen, H. 2015, MNRAS, 450, 3233
- Winn, J. N. 2010, in Exoplanets, ed. S. Seager (Univ. Arizona Press), 55

## License

MIT or Apache-2.0, at your option.
