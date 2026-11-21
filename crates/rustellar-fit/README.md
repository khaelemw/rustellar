# rustellar-fit

Least-squares transit fits in pure Rust: the transit model of
`rustellar-model` fitted to a light curve, from a starting guess to the
best parameters, their standard errors and derived quantities.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `fit` feature.

```toml
[dependencies]
# Through the umbrella, which most projects want:
rustellar = { version = "0.1", features = ["fit"] }
# Or this crate alone:
rustellar-fit = "0.1"
```

On its own it re-exports the core and model types it takes and returns, so
neither need be listed.

**Status:** in development, not yet released.

## Statement of need

A transit search gives a period, a mid-transit time, a depth and a
duration; the planet's radius ratio, orbit and limb darkening, with their
uncertainties, need a fit of the full model. This crate will give that
fit by Levenberg-Marquardt, exactly to its stopping rule, with standard
errors from the fit's own linearisation and the depth, durations,
inclination and stellar density derived from it.

## So far

- `TransitFit`: the light curve, the starting parameters and baseline,
  which parameters are free (`Parameter`), and each point's exposure.
- `TransitFit::run`: Levenberg-Marquardt (More 1978; Nielsen 1999) by
  Householder QR and Givens rotations, never the normal equations, with
  geodesic acceleration (Transtrum & Sethna 2012) and steps projected onto
  the bounds; a central-difference Jacobian. Noiseless light curves are
  recovered to 1e-9 or better from offset starts, grazing and long-cadence
  ones included.
- Standard errors and the covariance from the fit's own `R`, never the
  normal equations; the depth, durations, inclination and stellar density
  with errors to first order. On simulated noise the pulls of every
  parameter have mean 0 and spread 1 to within 0.2.

## Accuracy

The fit stops when a full Gauss-Newton step would move the parameters by
less than a few millionths of their standard errors, or, at a minimum on
a bound (`b = 0`, or limb darkening at the edge of its valid range), when
no step lowers chi-squared at all (`Termination::PrecisionLimit`). Steps
that cross a bound are projected back onto it, so the fit slides along a
bound rather than stopping at it or leaving the model's valid range.

Against batman 2.5.1 and PyTransit 2.6.18 with scipy 1.17.1's
`least_squares`, each fitting every parameter from the same start on the
same light curves (batman's flux with seeded Gaussian noise; limb
darkening held for the small planet and long cadence, where the data
leave it all but free). Each answer's chi-squared by this crate's model,
above this crate's; † by the method's own model, where its answer is
outside the valid range (negative intensity, `a / R*` below zero):

| Case | lm, 2-point | trf, 2-point | trf, 3-point | trf, 3-point, tolerances 1e-15 | PyTransit, trf, 3-point |
|---|--:|--:|--:|--:|--:|
| Hot Jupiter | 8e-8 | 2e-7 | 2e-7 | 4e-9 | 2e-7 |
| The same, BJD-sized times | 984 | 3.6† | 959† | 959† | 959† |
| Small planet | 167† | 167† | 167† | 167† | 167† |
| Grazing | 4e-5 | 4.5e4† | 4.5e4† | 4.5e4† | 4.5e4† |
| Long cadence | 1.4e-2 | 1.3e-2 | 1.4e-2 | 1.4e-2 | 2e-5 |

No method reaches a lower chi-squared on any case. Where the others find
the same minimum (the hot Jupiter), they stop 1e-4 to 5e-4 standard
errors short of it, against a few millionths here; their standard errors
agree with these to 6e-4. The rest is where they fail:

- **BJD-sized times:** scipy's difference step is relative to each
  parameter, so a mid-transit time near 2.46e6 days is stepped by a
  quarter of the transit and the Jacobian is wrong. Steps here are scaled
  to the transit's duration, not the epoch's size.
- **Small planet, grazing:** unbounded, the others walk out of the
  model's valid range (to `a / R*` of -15, or `k` of -0.02 with `u1` of
  -61) and stop there.
- **Long cadence:** batman's exposure average spreads its samples edge to
  edge rather than at the midpoints of equal slices (Kipping 2010), 40
  times further from the exact average at 15 samples; PyTransit averages
  as this crate does.

Noiseless light curves of this crate's model are recovered to 1e-9 or
better (`tests/recovery.rs`); batman's noiseless flux to within the gap
between the two models (batman's accuracy is about 2e-8).

## License

MIT or Apache-2.0, at your option.
