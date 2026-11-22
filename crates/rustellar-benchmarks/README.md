# rustellar-benchmarks

The rustellar crates compared with their Python references on the same
data, with one command. Not published: it depends on the libraries, never
the reverse. Each library's own reference tests stay in its crate
(`cargo test -p rustellar-bls`, for example).

## Validating a crate

Clone the repository, then run the suites by crate with the `cargo
validate` alias (for `cargo run --release -p rustellar-benchmarks --bin
rustellar-bench --`):

```sh
git clone <this repository> && cd rustellar
cargo validate list
cargo validate accuracy tls --quick
cargo validate accuracy bls detrend
cargo validate speed fit mcmc
cargo validate all --quick
cargo validate speed --dry-run
```

| Command | Runs |
|---|---|
| `list` | a table of every suite: its crate, accuracy or speed, minutes in full and with `--quick`, what it compares |
| `accuracy [TARGET...]` | the accuracy suites of each crate (`tls`, `bls`, `detrend`, `vet`) or suite named; all by default |
| `speed [TARGET...]` | the speed tables of each crate named (`model`, `detrend`, `bls`, `tls`, `ls`, `fit`, `vet`, `mcmc`); all by default |
| `all` | every accuracy suite, then every speed table |
| `help` | the commands and options |

Options: `--quick` (20 curves a set, 8 of the limb-darkened study),
`--limit N`, `--step N`, `--threads N`, `--cpu` (searches off the GPU) and
`--dry-run` (list what would run). Every table prints as Markdown, so
`cargo validate accuracy tls > tls.md` keeps it. `cargo validate-gpu`
builds with the `cuda` feature, for the searches on an NVIDIA GPU.

The time estimates in `list` are for 16 CPU threads of an AMD Ryzen 7
5700G; the GPU makes the searches several times faster. Speed tables want
a quiet machine. The references' times there were recorded on that
processor, so elsewhere compare the ratios only, or time the references
on yours (see below) and replace the files.

`compare` (below) is the older entry point to the same suites, by set
name.

## Results for each release

Each release's results are kept in `results/<version>/` of this crate, one
Markdown file per suite (`accuracy-many.md`, `speed-fit.md`, ...) and a
card of the machine they ran on (`machine.md`: processor, threads,
system, compiler, commit, version, GPU, and each command run):

```sh
cargo validate-gpu accuracy --save          # every accuracy suite, saved
cargo validate speed --save --repeat 3      # every speed table, three runs each
cargo validate report                       # this version's saved tables
cargo validate diff 0.1.0 0.2.0             # what changed between two releases
cargo validate diff 0.1.0 results/mine      # or against any saved folder
```

`--save` writes to this version's folder and `--to DIR` anywhere else;
saving again adds to the folder and the card. With `--repeat N` each speed
suite runs N times and saves each time's best, with each time's spread
(the median's distance from the best) in `speed-<crate>.spread.csv`.
Before any speed suite a fixed loop is timed nine times, and a warning
printed when its times spread by more than 5%: something else is running.
Each timing starts with one call to warm the caches.

`diff` compares the two runs' tables cell by cell, at the same table, row
and column. A time counts as changed beyond `--tolerance` (default 10%)
or twice either run's spread, where that is larger; any other number or
text counts at any change, as the accuracy suites are seeded and so
repeat exactly on one machine. It prints each changed cell with its old
and new value, and how much slower or faster a time is. The references'
columns never change (they are read from `data/`), so a change there means
a reference file was replaced.

For a release: on a quiet machine, run every suite saved (`accuracy
--save` with the GPU if there is one, then `speed --save --repeat 3`),
`diff` against the last release, explain or fix every change, copy the
new tables into the crates' READMEs and commit the folder with the
release.

## rustellar-tls against transitleastsquares 1.32

```sh
git clone <this repository> && cd rustellar
cargo run --release -p rustellar-benchmarks --bin compare
cargo run --release -p rustellar-benchmarks --bin compare -- many --limit 50
cargo run --release -p rustellar-benchmarks --features cuda --bin compare
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
cargo run --release -p rustellar-benchmarks --bin compare -- limb
cargo run --release -p rustellar-benchmarks --bin compare -- limb --step 10
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
cargo run --release -p rustellar-benchmarks --bin compare -- window
```

400 limb-darkened transits on spotted, rotating stars, each light curve
detrended with the biweight at multiples of its transit's duration: the
depth and signal to noise each window keeps, with and without the transits
masked, and how often a blind BLS search afterwards finds the injected
period. The results, and the window they chose, are in
`rustellar-detrend`'s README. It takes about 80 seconds on 16 threads;
`--limit N` runs the first N curves only.

## Detrending methods

```sh
cargo run --release -p rustellar-benchmarks --bin compare -- methods
```

The same 400 curves detrended with the biweight at 2.5 transit durations
and with each spline (plain, clipped, Huber) at knots 1 to 4 durations
apart: depth and signal to noise kept, and the blind BLS search. wotan
1.10's `biweight`, `rspline` and `hspline` on the same curves are read
from `data/methods_wotan.csv` and `data/methods_wotan_search.csv`, made by
running wotan on the curves `--export-methods FILE` writes and searching
what it flattened with `--search-flattened IN OUT`, as the crate's own are
searched. The results are in `rustellar-detrend`'s README. About 90
seconds on 16 threads.

## Transit masking

```sh
cargo run --release -p rustellar-benchmarks --bin compare -- masking
```

The same 400 curves flattened blind (the biweight at 2.5 transit
durations), searched with BLS, and flattened again around the candidate
found (`Detrend::flatten_around`), once or searching again until the mask
repeats, with several mask and window lengths; beside them, the true
transits masked. Depth and signal to noise kept, and the last search's
result. The results are in `rustellar-detrend`'s README. About two
minutes on 16 threads.

## Whitening

```sh
cargo run --release -p rustellar-benchmarks --bin compare -- whitening
```

The same 400 curves with their white noise and with AR(1) noise of 0.5 and
0.8, each with its transit and without, flattened (the biweight at 2.5
durations), then whitened by an autoregressive model (AIC up to orders 3,
10 and 30); a BLS search on each, and the matched search
(`Bls::search_whitened`): detections at SDE above 7, false alarms,
and detections above the SDE 5% of the transit-free curves reach. The
results are in `rustellar-detrend`'s README. About three minutes on 16
threads.

## rustellar-vet's alias table on TLS's candidates

```sh
cargo run --release -p rustellar-benchmarks --bin compare -- vet
```

transitleastsquares 1.32's candidates on the five TLS sets, from their
reference files (no search is run). Where TLS found the injected period
the candidate is also given at a third, half, two thirds, three halves,
twice and three times it; where TLS found an alias, as it is. Each is
pointed at its alias table's lowest chi-squared, and the table counts how
often that is the true period, by ratio and by the signal to noise at the
true period. The results are in `rustellar-vet`'s README. It takes a few
seconds; `--limit N` uses each set's first N curves.

## Speed against the Python references

```sh
cargo run --release -p rustellar-benchmarks --bin compare -- speed
cargo run --release -p rustellar-benchmarks --bin compare -- speed-model
```

`speed` times every crate on the cases of its README's speed tables and
prints those tables; `speed-model`, `speed-detrend`, `speed-bls`,
`speed-tls`, `speed-ls`, `speed-fit`, `speed-vet` and `speed-mcmc` one
crate each (`cargo validate speed CRATE` does the same). The rustellar side is timed as the references
were: the best of several runs (seven for the model, the detrender and the fit,
three for a search, one where a search takes minutes). Nothing is timed by
default, as the numbers want a quiet machine; all eight take about ten
minutes, most of it the BLS searches on one thread and the mcmc chains.
`vet` has no reference column (no package runs the same checks); `mcmc`
gives each sampler's time per independent sample on rustellar-mcmc's
transit posterior.

The references' times are in `data/speed_<crate>.csv`, one row per case,
package (with its version) and seconds. They were run on the same machine
(an AMD Ryzen 7 5700G, 8 cores, 16 threads; Python 3.12, numpy 2.4) by
the scripts in [`references/`](references/README.md), each package on its own settings as
the README says: batman, PyTransit and exoplanet-core on one thread, wotan
on one thread after its first call, astropy's `BoxLeastSquares` on one
thread, transitleastsquares on every core, astropy's `LombScargle` as it
runs (one thread; its fast method uses numpy's FFT), batman's and
PyTransit's fits with scipy's `least_squares` on one thread. The rustellar columns are
always timed afresh, so on another machine they are that machine's while
the references' stay this one's: compare the ratios only on a similar
processor, or time the references there in the same way and replace the
files.

## Making the references again

Every reference file in the repository, the crates' `tests/reference/` and
this crate's `data/`, was made by a script in
[`references/`](references/README.md), with a `requirements.txt` pinning
the Python packages at the versions the files name. This crate is never
published, so the Python stays here: the science crates have none.
