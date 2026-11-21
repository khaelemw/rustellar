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

## License

MIT or Apache-2.0, at your option.
