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
| rustellar-model | 5.3 µs | 53 µs | 0.52 ms | 5.2 ms |
| batman 2.5.1 | 8 µs | 91 µs | 0.77 ms | 8.9 ms |
| PyTransit 2.6.18 | 12 µs | 57 µs | 0.64 ms | 5.6 ms |
| exoplanet-core 0.4.0 (flux only, no orbit) | 28 µs | 60 µs | 0.56 ms | 7.0 ms |

Kepler long cadence (29.4 min), 1,000 times:

| Samples per exposure | 5 | 15 | 30 |
|---|---|---|---|
| rustellar-model | 0.28 ms | 0.84 ms | 1.7 ms |
| batman 2.5.1 | 0.39 ms | 1.2 ms | 2.2 ms |
| PyTransit 2.6.18 | 0.24 ms | 0.70 ms | 1.3 ms |

A single flux in transit takes 43 ns with the planet fully on the disk and
150 ns across the limb, at full double precision:

- **rustellar-model** computes all three elliptic integrals in one AGM
  iteration (Bulirsch's `cel`, as in Agol et al. 2020).
- **batman and PyTransit** use polynomial approximations good to about 1e-8.
- **exoplanet-core** uses the same formulation at full precision.

This crate's light curves are the fastest of the four at every length. For
exposures PyTransit is 15 to 30% faster. At the same accuracy, the midpoint rule used
here needs about half as many samples as batman's rule.

## Limits

Circular orbits only, no light from the planet, and a radius ratio of at most
10,000.

## References

- Agol, E., Luger, R. & Foreman-Mackey, D. 2020, AJ, 159, 123
- Kipping, D. M. 2010, MNRAS, 408, 1758
- Kreidberg, L. 2015, PASP, 127, 1161
- Mandel, K. & Agol, E. 2002, ApJ, 580, L171
- Parviainen, H. 2015, MNRAS, 450, 3233
- Winn, J. N. 2010, in Exoplanets, ed. S. Seager (Univ. Arizona Press), 55

## License

MIT or Apache-2.0, at your option.
