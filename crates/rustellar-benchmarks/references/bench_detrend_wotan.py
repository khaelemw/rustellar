"""wotan 1.10's speed on the light curves of rustellar-detrend's benchmark
(the survey recipe: a Kepler quarter, a TESS sector at 2 minutes and at 20
seconds), for the README's table. wotan runs on one thread.

Each timing is the best of `REPEATS` after a warm-up call (numba compiles
on the first); the Hodges-Lehmann and Huber estimators, slow, once.

Run: python crates/rustellar-benchmarks/references/bench_detrend_wotan.py [CSV]

With CSV (crates/rustellar-benchmarks/data/speed_detrend.csv), also writes
each time as case,package,seconds for the benchmarks crate's speed table.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_detrend_survey_reference import CURVES, FIELDS, light_curve  # noqa: E402
import importlib.metadata  # noqa: E402
from wotan import flatten  # noqa: E402

REPEATS = 3


def curves():
    all_ = {row[0]: dict(zip(FIELDS, row)) for row in CURVES}
    tess = all_["tess"]
    fast = dict(tess, name="tess 20 s", cadence=tess["cadence"] / 6.0, n=tess["n"] * 6)
    return [all_["kepler"], tess, fast]


def best(time_, flux, window, method, repeats=REPEATS, **kw):
    run = lambda: flatten(  # noqa: E731
        time_, flux, method=method, window_length=window, break_tolerance=window / 2,
        edge_cutoff=0, return_trend=True, **kw,
    )
    run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    rows = []
    name_ = f"wotan {importlib.metadata.version('wotan')}"
    sets = []
    for p in curves():
        t, f, _ = light_curve(p)
        sets.append((p, np.array(t), np.array(f)))
    for p, t, f in sets:
        sec = best(t, f, p['window'], 'biweight', cval=5)
        rows.append((f"biweight {p['name']}", name_, sec))
        print(f"biweight {p['name']} ({len(t)} points): {sec:.4f} s", flush=True)
    p, t, f = sets[1]
    for key, name, method, kw in [
        ("median", "median", "median", {}),
        ("mean", "mean", "mean", {}),
        ("biweight", "biweight", "biweight", {"cval": 5}),
        ("welsch", "welsch", "welsch", {"cval": 2.11}),
        ("andrews", "andrews", "andrewsinewave", {"cval": 1.339}),
        ("hodges-lehmann", "hodges-lehmann", "hodges", {}),
        ("huber-joint", "huber (statsmodels proposal 2)", "huber", {"cval": 1.5}),
        ("trimmed", "trimmed", "trim_mean", {"proportiontocut": 0.1}),
        ("winsorized", "winsorized", "winsorize", {"proportiontocut": 0.1}),
    ]:
        repeats = 1 if method in ("hodges", "huber") else REPEATS
        sec = best(t, f, p['window'], method, repeats, **kw)
        rows.append((f"method {key}", name_, sec))
        print(f"{name} {p['name']}: {sec:.4f} s", flush=True)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", newline="\n") as out:
            out.write("case,package,seconds\n")
            for case, package, sec in rows:
                out.write(f"{case},{package},{sec!r}\n")


if __name__ == "__main__":
    main()
