# rustellar

Exoplanet and astronomy time-series tools in pure Rust. Each tool is a small
crate of its own, tested for accuracy and speed against the Python packages
astronomers already use.

**Status:** version 0.1. The API may still change before 1.0.

Depend on the `rustellar` crate and turn on the tools you need as features,
or depend on a tool's own crate directly:

```toml
[dependencies]
rustellar = { version = "0.1", features = ["tls", "vet"] }
```

| Crate | Feature | What it does | Status |
|---|---|---|---|
| [`rustellar`](crates/rustellar) | | Brings in the tools below by feature | done |
| [`rustellar-core`](crates/rustellar-core) | always on | Light curves, folding, time types | done |
| [`rustellar-model`](crates/rustellar-model) | `model` | Transit light curves (Mandel & Agol 2002), exposure averaging | done |
| [`rustellar-fit`](crates/rustellar-fit) | `fit` | Least-squares transit fit | done |
| [`rustellar-bls`](crates/rustellar-bls) | `bls` | Box least squares transit search | done |
| [`rustellar-detrend`](crates/rustellar-detrend) | `detrend` | Detrending filters (biweight and others) | done |
| [`rustellar-mcmc`](crates/rustellar-mcmc) | `mcmc` | Ensemble MCMC, diagnostics and transit posteriors | done |
| [`rustellar-ls`](crates/rustellar-ls) | `ls` | Lomb-Scargle periodogram, peaks, false-alarm probabilities | done |
| [`rustellar-tls`](crates/rustellar-tls) | `tls` | Transit least squares | done |
| [`rustellar-vet`](crates/rustellar-vet) | `vet` | Vetting candidates: aliases, odd and even depths, coverage | done |

## Development

Run the tests (all features, so the umbrella crate's tools are included too):

```sh
cargo test --workspace --all-features
```

The longer comparisons against other packages take minutes each, so they
are skipped by default. Run them with:

```sh
cargo test --workspace --release -- --ignored
```

CI has no GPU, so if you change the GPU code, run the GPU tests locally on a
machine with an NVIDIA GPU. These cargo aliases are defined in
`.cargo/config.toml`. If no GPU is found the tests fail instead of being
skipped, so a passing run means the GPU really was tested:

```sh
cargo gpu-test
cargo gpu-test --test cuda -- --ignored
```

The second runs the longer GPU comparisons. To see line coverage of
`rustellar-bls` including its GPU code (needs
[cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov)):

```sh
cargo gpu-coverage
```

The same for `rustellar-tls`:

```sh
cargo gpu-test-tls
cargo gpu-coverage-tls
```

Run the benchmarks:

```sh
cargo bench -p rustellar-model
cargo bench -p rustellar-bls
cargo bench -p rustellar-tls
cargo bench -p rustellar-ls
cargo bench -p rustellar-fit
cargo bench -p rustellar-vet
```

To check a crate's accuracy and speed against the Python packages (this
produces the tables in each crate's README; see
[`rustellar-benchmarks`](crates/rustellar-benchmarks), which is not
published):

```sh
cargo validate list                  # list the suites and how long each takes
cargo validate accuracy tls --quick  # a quick check of one crate
cargo validate speed fit mcmc        # speed tables (run on an idle machine)
cargo validate all --quick           # a quick run of everything
cargo validate-gpu accuracy tls      # the searches on an NVIDIA GPU
cargo validate speed --save --repeat 3
cargo validate diff 0.1.0 0.2.0      # what changed between two releases
```

Results for each release are saved in `crates/rustellar-benchmarks/results/`.

The reference data the tests compare against lives in each crate's
`tests/reference` folder. To keep the published packages small, they
contain only the unit tests, which generate their own data. Reference
data, integration tests and benchmarks are excluded, so any test that
reads a reference file goes in `tests/`, and the full suite only runs from
a clone of this repository.

## License

MIT or Apache-2.0.
