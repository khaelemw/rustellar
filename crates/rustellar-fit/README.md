# rustellar-fit

Least-squares transit fitting in pure Rust. It fits `rustellar-model`'s
transit model to a light curve, from a starting guess to the best
parameters, their standard errors and quantities derived from them.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `fit` feature.

```toml
[dependencies]
# Through the umbrella crate (what most projects want):
rustellar = { version = "0.1", features = ["fit"] }
# Or this crate alone:
rustellar-fit = "0.1"
```

This crate re-exports the core and model types it uses, so you don't need
to add `rustellar-core` or `rustellar-model` yourself.

**Status:** in development, not yet released.

## Statement of need

A transit search gives a period, a mid-transit time, a depth and a
duration. Getting the planet's radius ratio, orbit and limb darkening,
with uncertainties, takes a fit of the full model. This crate does that
fit by Levenberg-Marquardt, converged tightly to its stopping rule, with
standard errors from the fit's own linearisation and the depth,
durations, inclination and stellar density derived from the result.

## What it does

- `TransitFit`: the light curve, the starting parameters and baseline,
  which parameters are free (`Parameter`), and each point's exposure.
- `TransitFit::run`: Levenberg-Marquardt (More 1978; Nielsen 1999) using
  Householder QR and Givens rotations (never the normal equations), with
  geodesic acceleration (Transtrum & Sethna 2012) and steps projected onto
  the bounds. The Jacobian is mostly analytic: the orbital parameters'
  columns come from the chain rule through the planet's separation, the
  limb darkening and baseline columns are exact (the flux is linear in
  them), and only two derivatives of the blocked light are found by finite
  differences. Noiseless light curves are recovered to 1e-9 or better from
  offset starts, including grazing and long-cadence ones.
- `LimbDarkeningBounds`: allows limb darkening anywhere the model is valid
  (the default, which includes limb brightening), or restricts it to
  Kipping's (2013) triangle of physically limb-darkened stars. When the
  fit runs into a bound, it solves the step within the bound, so the other
  parameters keep adjusting along it and the fit still converges.
- `TransitFit::with_threads`: spreads the model and the Jacobian rows over
  several threads, with bit-identical results. The linear algebra stays on
  one thread, so extra threads only help for very large light curves (a
  fifth faster at a million points).
- Eccentric orbits: `e cos w` and `e sin w` are parameters, held at their
  starting values (circular by default) because a transit alone barely
  constrains them. Free `e sin w` when a / R* is known from the star's
  density; `e cos w` hardly affects a transit at all (its error is above 1
  at 1e-4 noise per point). On an eccentric orbit the Jacobian is found by
  finite differences.
- Standard errors and the covariance come from the fit's own `R` factor,
  not the normal equations. The depth, durations, inclination and stellar
  density are given with first-order errors. On simulated noise, every
  parameter's pulls have mean 0 and standard deviation 1 to within 0.2.

## Accuracy

The fit stops when a full Gauss-Newton step would move the parameters by
less than a few millionths of their standard errors, along the bound
where the minimum is on one (limb darkening at the edge of its range).
At `b = 0`, where the flux is flat in `b` to first order, it stops when
no step lowers chi-squared at all (`Termination::PrecisionLimit`). Steps
that cross a bound are projected back onto it, so the fit slides along
the bound instead of stopping there or leaving the model's valid range.

The comparison is against batman 2.5.1 and PyTransit 2.6.18 with scipy
1.17.1's `least_squares`, each fitting every parameter from the same
start on the same light curves (batman's flux with seeded Gaussian noise;
limb darkening is held fixed for the small planet and long cadence, where
the data barely constrain it). The table shows how much higher each
method's final chi-squared is than this crate's, computed with this
crate's model. † marks answers outside the model's valid range (negative
intensity, or `a / R*` below zero); those are computed with the method's
own model instead:

| Case | lm, 2-point | trf, 2-point | trf, 3-point | trf, 3-point, tolerances 1e-15 | PyTransit, trf, 3-point |
|---|--:|--:|--:|--:|--:|
| Hot Jupiter | 8e-8 | 2e-7 | 2e-7 | 4e-9 | 2e-7 |
| The same, BJD-sized times | 984 | 3.6† | 959† | 959† | 959† |
| Small planet | 167† | 167† | 167† | 167† | 167† |
| Grazing | 4e-5 | 4.5e4† | 4.5e4† | 4.5e4† | 4.5e4† |
| Long cadence | 1.4e-2 | 1.3e-2 | 1.4e-2 | 1.4e-2 | 2e-5 |

No method reaches a lower chi-squared in any case. Where the others find
the same minimum (the hot Jupiter), they stop 1e-4 to 5e-4 standard
errors short of it, against a few millionths here, and their standard
errors agree with these to 6e-4. The other cases are where they fail:

- **BJD-sized times:** scipy's difference step is relative to each
  parameter, so a mid-transit time near 2.46e6 days is stepped by a
  quarter of the transit and the Jacobian is wrong. This crate scales its
  steps to the transit's duration instead.
- **Small planet, grazing:** unbounded, the others walk out of the
  model's valid range (to `a / R*` of -15, or `k` of -0.02 with `u1` of
  -61) and stop there.
- **Long cadence:** batman places its exposure samples edge to edge
  instead of at the midpoints of equal slices (Kipping 2010), which puts
  it 40 times further from the exact average at 15 samples. PyTransit
  averages the same way as this crate.

Noiseless light curves from this crate's model are recovered to 1e-9 or
better (`tests/recovery.rs`), and batman's noiseless flux to within the
difference between the two models (batman is accurate to about 2e-8).

## Speed

The setup for these timings:

- one thread, on an AMD Ryzen 7 5700G desktop CPU, Rust 1.99, Python 3.12,
  numpy 2.4, scipy 1.17.1;
- the light curves and starting points of the accuracy table above, each
  fit timed from set-up to result (for the Python packages, from building
  the model to scipy's answer), best of seven repeats.

`cargo validate speed fit` reruns this crate's timings and prints them
beside the other packages' recorded times. `cargo bench -p rustellar-fit`
runs the Rust timings alone.

| Case | rustellar-fit | batman 2.5.1, lm, 2-point | trf, 2-point | trf, 3-point | trf, tolerances 1e-15 | PyTransit 2.6.18, trf, 3-point |
|---|--:|--:|--:|--:|--:|--:|
| Hot Jupiter, noiseless | 1.5 ms | 14 ms | 19 ms | 28 ms | 35 ms | 14 ms |
| Hot Jupiter | 1.3 ms | 12 ms | 11 ms | 19 ms | 31 ms | 9.6 ms |
| The same, BJD-sized times | 3.1 ms | 3.0 ms† | 20 ms† | 5.9 ms† | 10 ms† | 2.7 ms† |
| Small planet | 13 ms | 6.9 ms† | 17 ms† | 27 ms† | 56 ms† | 8.3 ms† |
| Grazing | 14 ms | 380 ms | 17 ms† | 23 ms† | 39 ms† | 8.5 ms† |
| Long cadence | 8.6 ms | 200 ms | 190 ms | 290 ms | 300 ms | 19 ms |

† stopped far from the minimum (chi-squared 160 to 45,000 higher; see
the accuracy table), so these didn't do the same work. Where a method
does reach the same minimum, this crate is 2.2 to 27 times as fast as the
fastest of them.

Most of the gain comes from the Jacobian. Finite differences cost two
light curves per free parameter. Here the flux depends on the parameters
only through the separation, the radius ratio and the limb darkening, so
the orbital columns follow from the chain rule, the limb darkening and
baseline columns are exact (the flux is linear in them), and only two
derivatives of the blocked light ([`blocked_light`] in rustellar-model)
need finite differences: five evaluations per in-transit point and none
outside transit. The QR factorisation also skips out-of-transit rows,
where only the baseline column is nonzero.

[`blocked_light`]: https://docs.rs/rustellar-model/latest/rustellar_model/fn.blocked_light.html

## License

MIT or Apache-2.0.
