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
  the bounds. The Jacobian comes from the geometry: the chain rule
  through the planet's separation for the orbit's parameters, exact for
  limb darkening and baseline (the flux is linear in them), and only the
  separation's and radius ratio's derivatives of the blocked light
  differenced. Noiseless light curves are
  recovered to 1e-9 or better from offset starts, grazing and long-cadence
  ones included.
- `LimbDarkeningBounds`: limb darkening anywhere the model is valid (the
  default, limb brightening included), or held to Kipping's (2013)
  triangle of stars that darken to their limb. A fit pressing against a
  bound follows it: the step is solved within the bound, so the other
  parameters move as they should along it and the fit converges there.
- `TransitFit::with_threads`: the model and the Jacobian's rows over
  several threads, the same to the bit as on one. The solver's linear
  algebra stays on one thread, so threads pay only for very large light
  curves (a fifth off at a million points).
- Eccentric orbits: `e cos w` and `e sin w` as parameters, held at the
  start's values (a circular orbit, by default) because a transit alone
  hardly constrains them. Free `e sin w` when a / R* is known from the
  star's density; `e cos w` barely shows in a transit at all (its error
  is above 1 at 1e-4 a point). On an eccentric orbit the Jacobian is
  differenced.
- Standard errors and the covariance from the fit's own `R`, never the
  normal equations; the depth, durations, inclination and stellar density
  with errors to first order. On simulated noise the pulls of every
  parameter have mean 0 and spread 1 to within 0.2.

## Accuracy

The fit stops when a full Gauss-Newton step would move the parameters by
less than a few millionths of their standard errors, along the bound
where the minimum is on one (limb darkening at the edge of its range).
At `b = 0`, where the flux is flat in `b` to first order, it stops when
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

## Speed

The setup for these timings:

- one thread, on an AMD Ryzen 7 5700G desktop CPU, Rust 1.99, Python 3.12,
  numpy 2.4, scipy 1.17.1;
- the light curves and starts of the accuracy table above, each fit timed
  from its set-up to its result (the references' from the model's set-up
  to scipy's answer), the best of seven repeats.

`cargo run --release -p rustellar-benchmarks --bin compare -- speed-fit`
prints this table again, this crate timed afresh beside the references'
recorded times; `cargo bench -p rustellar-fit` runs the Rust side alone.

| Case | rustellar-fit | batman 2.5.1, lm, 2-point | trf, 2-point | trf, 3-point | trf, tolerances 1e-15 | PyTransit 2.6.18, trf, 3-point |
|---|--:|--:|--:|--:|--:|--:|
| Hot Jupiter, noiseless | 1.6 ms | 14 ms | 19 ms | 28 ms | 35 ms | 14 ms |
| Hot Jupiter | 1.4 ms | 12 ms | 11 ms | 19 ms | 31 ms | 9.6 ms |
| The same, BJD-sized times | 3.3 ms | 3.0 ms† | 20 ms† | 5.9 ms† | 10 ms† | 2.7 ms† |
| Small planet | 13 ms | 6.9 ms† | 17 ms† | 27 ms† | 56 ms† | 8.3 ms† |
| Grazing | 19 ms | 0.38 s | 17 ms† | 23 ms† | 39 ms† | 8.5 ms† |
| Long cadence | 8.5 ms | 0.20 s | 0.19 s | 0.29 s | 0.30 s | 19 ms |

† stopped far from the minimum (chi-squared 160 to 45,000 higher; see
the accuracy table), so not the same work. Where a method reaches the
same minimum this crate is 2.2 to 20 times as fast.

Most of that is the Jacobian. Differencing the light curve costs two
light curves a free parameter; here the flux depends on the parameters
only through the separation, the radius ratio and the limb darkening, so
the orbit's columns follow by the chain rule, limb darkening's and the
baseline's exactly (the flux is linear in them), and only two derivatives
of the blocked light ([`blocked_light`] in rustellar-model) are
differenced, at five evaluations a point in transit and none out of it.
The QR factorisation skips the rows out of transit too, where only the
baseline's column is nonzero.

[`blocked_light`]: https://docs.rs/rustellar-model/latest/rustellar_model/fn.blocked_light.html

## License

MIT or Apache-2.0, at your option.
