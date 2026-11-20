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

A run over every set takes about 4 minutes on 16 threads with the GPU (an
AMD Ryzen 7 5700G and an RTX 3060 Ti), most of it the two 2-minute sets;
`--limit` makes a quick check. The results are in `rustellar-tls`'s
README.

## rustellar-bls against rustellar-tls on limb-darkened transits

```sh
cargo run --release -p rustellar-validation --bin compare -- limb
cargo run --release -p rustellar-validation --bin compare -- limb --step 10
```

Box-shaped injections favour any BLS; real transits are rounded by limb
darkening, which TLS's template models. `limb` runs both searches on the
curves of `rustellar-bls`'s limb-darkened study: 1,000 transits (quadratic
limb darkening u = 0.4, 0.26, a Sun-like star, impact parameter uniform in
[0, 1)) in four survey families' noise, and 1,000 curves of noise alone. A
search detects a transit when its SDE beats all but 1% of its own
noise-only curves, so both run at the same false-alarm rate, and recovers it
within 1% of its period. The table gives recovery and median errors by
family, by signal to noise and in all, then transitleastsquares 1.32 on the
quarter of the curves it was run on (`rustellar-bls`'s
`tests/reference/limb_tls.csv`). `--step N` takes every N-th block of eight
curves (each family with a transit, then without). Every curve takes both
searches on one thread each; all 2,000 take about 8 minutes on the same
machine.

## How long a detrending window should be

```sh
cargo run --release -p rustellar-validation --bin compare -- window
```

400 limb-darkened transits on spotted, rotating stars, each light curve
detrended with the biweight at multiples of its transit's duration: the
depth and signal to noise each window keeps, with and without the transits
masked, and how often a blind BLS search afterwards finds the injected
period. The results, and the window they chose, are in
`rustellar-detrend`'s README. It takes about 80 seconds on 16 threads;
`--limit N` runs the first N curves only.

## Speed against the Python references

```sh
cargo run --release -p rustellar-validation --bin compare -- speed
cargo run --release -p rustellar-validation --bin compare -- speed-model
```

`speed` times every crate on the cases of its README's speed tables and
prints those tables; `speed-model`, `speed-detrend`, `speed-bls` and
`speed-tls` one crate each. The rustellar side is timed as the references
were: the best of several runs (seven for the model and the detrender,
three for a search, one where a search takes minutes). Nothing is timed by
default, as the numbers want a quiet machine; all four take about five
minutes, most of it the BLS searches on one thread.

The references' times are in `data/speed_<crate>.csv`, one row per case,
package (with its version) and seconds. They were run on the same machine
(an AMD Ryzen 7 5700G, 8 cores, 16 threads; Python 3.12, numpy 2.4) by
scripts kept outside the repository, each package on its own settings as
the README says: batman, PyTransit and exoplanet-core on one thread, wotan
on one thread after its first call, astropy's `BoxLeastSquares` on one
thread, transitleastsquares on every core. The rustellar columns are
always timed afresh, so on another machine they are that machine's while
the references' stay this one's: compare the ratios only on a similar
processor, or time the references there in the same way and replace the
files.
