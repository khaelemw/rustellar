# rustellar

Exoplanet and astronomy time-series tools in pure Rust. Each tool is a small
crate with no dependencies and a written scope, tested against the
established Python packages.

**Status:** in development. Nothing is released yet.

Depend on the `rustellar` crate and turn on the tools you need as features,
or depend on a tool's own crate directly:

```toml
[dependencies]
rustellar = { version = "0.0", features = ["model"] }
```

| Crate | Feature | What it does | Status |
|---|---|---|---|
| [`rustellar`](crates/rustellar) | | Brings in the tools below by feature | in place |
| [`rustellar-core`](crates/rustellar-core) | always on | Light curves, folding, time types | done |
| [`rustellar-model`](crates/rustellar-model) | `model` | Transit light curves (Mandel & Agol 2002), exposure averaging | done |
| `rustellar-fit` | `fit` | Least-squares transit fit | planned |
| [`rustellar-bls`](crates/rustellar-bls) | `bls` | Box least squares transit search | in progress |
| [`rustellar-detrend`](crates/rustellar-detrend) | `detrend` | Detrending filters (biweight and others) | in progress |
| `rustellar-mcmc` | `mcmc` | Affine-invariant ensemble sampler | planned |
| `rustellar-ls` | `ls` | Lomb-Scargle periodogram | planned |
| `rustellar-tls` | `tls` | Transit least squares | planned |

## Development

Run the tests (all features, so the umbrella crate's tools are included too):

```sh
cargo test --workspace --all-features
```

The long studies (injection-recovery against other packages, minutes each)
are left out of that run; run them on their own:

```sh
cargo test --workspace --release -- --ignored
```

CI has no GPU, so run the GPU tests on a machine with an NVIDIA GPU before
pushing changes to the GPU code. These are cargo aliases (in
`.cargo/config.toml`); a missing GPU fails here, not skips:

```sh
cargo gpu-test
cargo gpu-test --test cuda -- --ignored
```

The second runs the heavy GPU comparisons. With
[cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov), the line
coverage of `rustellar-bls`, its GPU code included:

```sh
cargo gpu-coverage
```

Run the benchmarks:

```sh
cargo bench -p rustellar-model
```

The reference data the tests compare against is in each crate's
`tests/reference` folder. The published packages stay small: they hold the
unit tests, which need no files (their data is generated in the test), and
leave out the reference data, the integration tests and the benchmarks
(each crate's `exclude`). So a test that reads a reference file is an
integration test, in `tests/`, and the full suite runs from the source.

## License

MIT or Apache-2.0, at your option.
