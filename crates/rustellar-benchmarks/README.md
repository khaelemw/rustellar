# rustellar-benchmarks

Compares the rustellar crates with the Python packages they replace, on
the same data, with one command. This crate isn't published: it depends
on the other crates, and none of them depend on it. Each crate's own
reference tests stay in that crate (`cargo test -p rustellar-bls`, for
example).

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
5700G; the GPU makes the searches several times faster. Run the speed
tables on an idle machine. The Python packages' times were recorded on
that processor, so on other hardware compare only the ratios, or time the
Python packages on your machine (see below) and replace the files.

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

`--save` writes to this version's folder, and `--to DIR` writes anywhere
else; saving again adds to the folder and the card. With `--repeat N`,
each speed suite runs N times and saves the best time for each cell, with
its spread (the median's distance from the best) in
`speed-<crate>.spread.csv`. Before any speed suite, a fixed loop is timed
nine times, and a warning is printed if those times vary by more than 5%,
which means something else is using the machine. Each timing starts with
one untimed call to warm the caches.

`diff` compares two runs' tables cell by cell (same table, row and
column; a row is named by its first cell, or by as many of its first
cells as tell it apart). A time is a number with a unit, or a bare
number of seconds in a speed table (counts such as Points, Periods and
tau excepted). A time counts as changed if it moves by more than `--tolerance`
(default 10%) or twice either run's spread, whichever is larger. Any
other number or text counts as changed if it differs at all, since the
accuracy suites are seeded and repeat exactly on the same machine. It
prints each changed cell with its old and new value, and how much slower
or faster a time got. The Python packages' columns are read from `data/`
and never change, so a change there means a reference file was replaced.

For a release: on an idle machine, run and save every suite (`accuracy
--save`, with the GPU if there is one, then `speed --save --repeat 3`),
`diff` against the last release, explain or fix every change, copy the
new tables into the crates' READMEs, and commit the folder with the
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

In those sets every transit has radius ratio 0.015 to 0.045, period 1 to
9 days, a = 12, b = 0.2. The harder regimes give the star the Sun's
density (a = 4.206 P^(2/3)), b from 0 to 0.7, and each changes one thing:

| Set | Curves | Light curve |
|---|---|---|
| `long` | 100 | periods 9 to 13.5 days in 27 days: two or three transits |
| `ultrashort` | 100 | periods 0.25 to 1 day, 10 days at 10 minutes |
| `deep` | 100 | radius ratios 0.1 to 0.2 |
| `grazing` | 100 | radius ratios 0.05 to 0.12, b from 1 - k to 1 + k/2 |
| `eccentric` | 100 | eccentricities 0.1 to 0.7, any argument of periastron |
| `fast` | 40 | 6 days at 20 seconds |
| `gaps` | 40 | three 27-day blocks over 108 days, one missing, each with a gap |
| `flagged` | 100 | 2% of points flagged and dropped, 1% outliers of 5 to 20 errors, 1% NaN |
| `variable` | 100 | a spotted star (0.1% to 0.5%, 3 to 10 days) and red noise, detrended first (wotan's biweight and `rustellar-detrend`'s, the window three transit durations) |

`references/make_tls_regimes_reference.py` writes their files. Each row
also stores the number of points and the sum of the flux, and the tests
check that the curves built in Rust match. Every transit has quadratic
limb darkening (0.4804, 0.1867). The noise at each point is the sum of
four uniform draws from SplitMix64(seed), minus 2, times sqrt(3), times
the point's error, so both packages see identical light curves (the
transit comes from batman in Python and from `rustellar-model` here,
which agree to a few parts in 10^8).

`cargo validate accuracy around` runs one more set: the `variable` curves
flattened and searched as above, then flattened again with the
candidate's transits masked (`Detrend::flatten_around`) and searched
again, against transitleastsquares on the first flattening.

A run over every set with `cargo validate-gpu accuracy tls` takes about
5 minutes on 16 threads with the GPU (an AMD Ryzen 7 5700G and an RTX
3060 Ti), the longest the gaps and the 300-curve sets; `--limit` makes a
quick check. The results are in `rustellar-tls`'s README.

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

This takes transitleastsquares 1.32's candidates on the TLS sets from
their reference files (no search is run). Where transitleastsquares found
the injected period, the candidate is also tried at a third, half, two
thirds, three halves, twice and three times that period; where it found
an alias, the candidate is used as found. Each candidate's alias table
picks the row with the lowest chi-squared, and the output counts how
often that's the true period, by ratio and by signal-to-noise at the true
period. The results are in `rustellar-vet`'s README. It takes a few
seconds; `--limit N` uses each set's first N curves.

## Speed against the Python references

```sh
cargo run --release -p rustellar-benchmarks --bin compare -- speed
cargo run --release -p rustellar-benchmarks --bin compare -- speed-model
```

`speed` times every crate on the cases in its README's speed tables and
prints those tables; `speed-model`, `speed-detrend`, `speed-bls`,
`speed-tls`, `speed-ls`, `speed-fit`, `speed-vet` and `speed-mcmc` do one
crate each (as does `cargo validate speed CRATE`). The Rust side is timed
the same way the Python packages were: best of several runs (seven for
the model, detrending and fitting, three for a search, one where a
search takes minutes). Nothing is timed by default, since the numbers
need an idle machine. All eight take about ten minutes, mostly the
single-threaded BLS searches and the mcmc chains. `vet` has no Python
column, since no package runs the same checks; `mcmc` gives each
sampler's time per independent sample on rustellar-mcmc's transit
posterior.

The references' times are in `data/speed_<crate>.csv`, one row per case,
package (with its version) and seconds. They were run on the same machine
(an AMD Ryzen 7 5700G, 8 cores, 16 threads; Python 3.12, numpy 2.4) by
the scripts in [`references/`](references/README.md), each package with
the settings its crate's README gives: batman, PyTransit and
exoplanet-core on one thread, wotan on one thread after its first call,
astropy's `BoxLeastSquares` on one thread, transitleastsquares on every
core, astropy's `LombScargle` as it runs (one thread; its fast method
uses numpy's FFT), and batman's and PyTransit's fits with scipy's
`least_squares` on one thread. The rustellar columns are always timed
fresh, so on another machine they reflect that machine while the Python
columns still reflect this one. Compare only the ratios, and only on a
similar processor, or time the Python packages on your machine the same
way and replace the files.

## Making the references again

Every reference file in the repository, the crates' `tests/reference/` and
this crate's `data/`, was made by a script in
[`references/`](references/README.md), with a `requirements.txt` pinning
the Python packages at the versions the files name. This crate is never
published, so the Python stays here: the science crates have none.
