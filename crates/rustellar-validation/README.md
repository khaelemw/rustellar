# rustellar-validation

The rustellar crates compared with their Python references on the same
data, with one command. Not published: it depends on the libraries, never
the reverse. Each library's own reference tests stay in its crate
(`cargo test -p rustellar-bls`, for example).

## rustellar-tls against transitleastsquares 1.32

```sh
git clone <this repository> && cd rustellar
cargo run --release -p rustellar-validation --bin compare
cargo run --release -p rustellar-validation --bin compare -- many --limit 50
cargo run --release -p rustellar-validation --features cuda --bin compare
```

For each set, `compare` builds every injected light curve again from its
seed, searches it with `rustellar-tls` (TLS's defaults, over Ofir's period
grid), and prints how many transits each package finds within 1% of the
injected period. On the transits both find, it prints each package's median
error in period, mid-transit times, depth and duration, how often
`rustellar-tls` is the closer, and a two-sided sign test.

`data/tls_<set>.csv` holds each curve's injected transit and
transitleastsquares 1.32's result on it (`power()` with its defaults and
the flux errors given).

| Set | Curves | Light curve |
|---|---|---|
| `many` | 300 | 27 days at 30 minutes; in 30% of them a quarter of the points dropped |
| `mixed` | 200 | 13.5 days at 30 minutes, then 13 days at 10 minutes |
| `varying` | 200 | 27 days at 30 minutes, errors from 1 to 3 times the noise |
| `short` | 40 | 27 days at 2 minutes |
| `sectors` | 24 | two sectors at 2 minutes, 54 days less two one-day gaps |

Every transit: radius ratio 0.015 to 0.045, period 1 to 9 days, a = 12,
b = 0.2, quadratic limb darkening (0.4804, 0.1867). The noise per point is
the sum of four uniforms from SplitMix64(seed), less 2, times sqrt(3), times
the point's error, so both packages see the same light curves (the transit
from batman in Python and `rustellar-model` here, which agree to a few
parts in 10^8).

A run over every set takes tens of minutes on a desktop CPU, most of it the
two 2-minute sets; `--limit` makes a quick check.
