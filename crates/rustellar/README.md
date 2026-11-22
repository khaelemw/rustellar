# rustellar

Exoplanet and astronomy time-series tools in pure Rust. Each tool is its own
crate, and this one pulls in whichever you turn on as features, so you only
compile what you use.

```toml
[dependencies]
rustellar = { version = "0.1", features = ["tls", "vet"] }
```

| Feature | Crate | What it does |
|---|---|---|
| `model` | `rustellar-model` | Transit light curves (Mandel & Agol 2002) with exposure averaging |
| `fit` | `rustellar-fit` | Least-squares transit fitting |
| `bls` | `rustellar-bls` | Box least squares transit search |
| `tls` | `rustellar-tls` | Transit least squares search |
| `detrend` | `rustellar-detrend` | Detrending filters (biweight and others) |
| `ls` | `rustellar-ls` | Lomb-Scargle periodogram |
| `mcmc` | `rustellar-mcmc` | Ensemble MCMC and transit posteriors |
| `vet` | `rustellar-vet` | Vetting transit candidates |
| `full` | | All of the above |
| `cuda` | | Runs the `bls` and `tls` searches on an NVIDIA GPU (needs Rust 1.88 or later) |

`cuda` is not part of `full`. Without the CUDA driver at run time, the
searches fall back to the CPU.

The shared types from `rustellar-core` (`LightCurve`, `Epoch`, `TimeDelta`)
are always included.

**Status:** in development, not yet released.

## License

MIT or Apache-2.0.
