# rustellar

Exoplanet and astronomy time-series tools in pure Rust. Each tool is its own
crate. This one brings in the tools you turn on as features, so a project
compiles only what it uses.

```toml
[dependencies]
rustellar = { version = "0.0", features = ["model"] }
```

| Feature | Crate | What it does |
|---|---|---|
| `bls` | `rustellar-bls` | Box least squares transit search (in development) |
| `detrend` | `rustellar-detrend` | Detrending: a sliding biweight to full precision (in development) |
| `model` | `rustellar-model` | Transit light curves (Mandel & Agol 2002), exposure averaging |
| `tls` | `rustellar-tls` | Transit least squares: a limb-darkened template search (in development) |
| `full` | | Every tool |

The shared types from `rustellar-core` (`LightCurve`, `Epoch`, `TimeDelta`)
are always included. Each new tool will be added as a feature; see the
[workspace README](../../README.md) for the ones planned.

**Status:** in development, not yet released.

## License

MIT or Apache-2.0, at your option.
