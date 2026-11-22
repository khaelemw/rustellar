"""wotan 1.10's biweight, rspline and hspline on the window study's curves,
for rustellar-benchmarks's methods table.

The curves come from the benchmarks crate:

    cargo run --release -p rustellar-benchmarks --bin compare -- \\
        --export-methods crates/rustellar-benchmarks/references/work/methods_curves.csv

Each is flattened with the window (or knot spacing) a multiple of its
transit's duration, breaks at half of it, and measured as the crate measures
its own: the depth a matched filter with the transit's shape finds, over the
injected one, and the signal to noise over the ideal (the shape in white
noise of 1e-3 alone), the scatter from the points out of transit.

Writes crates/rustellar-benchmarks/data/methods_wotan.csv, and the flattened
curves to crates/rustellar-benchmarks/references/work/methods_wotan_flat.csv, which

    cargo run --release -p rustellar-benchmarks --bin compare --         --search-flattened crates/rustellar-benchmarks/references/work/methods_wotan_flat.csv         crates/rustellar-benchmarks/data/methods_wotan_search.csv

searches as the crate searches its own.

Run: python crates/rustellar-benchmarks/references/make_methods_wotan.py
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
os.makedirs(WORK, exist_ok=True)
CURVES = os.path.join(WORK, "methods_curves.csv")
OUT = os.path.join(ROOT, "crates", "rustellar-benchmarks", "data", "methods_wotan.csv")
# The flattened curves, for the crate's blind search (compare --search-flattened).
FLAT = os.path.join(WORK, "methods_wotan_flat.csv")
NOISE = 1e-3
TRIALS = [
    ("biweight", 2.5),
    ("rspline", 1.5),
    ("rspline", 2.5),
    ("rspline", 4.0),
    ("hspline", 1.5),
    ("hspline", 2.5),
    ("hspline", 4.0),
]


def measure(item):
    from wotan import flatten

    seed, t14, t, f, s = item
    ideal = np.sqrt((s * s).sum()) / NOISE
    rows = []
    for method, factor in TRIALS:
        w = factor * t14
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat = flatten(t, f, method=method, window_length=w, break_tolerance=w / 2,
                           edge_cutoff=0)
        ok = np.isfinite(flat)
        fi, si = flat[ok], s[ok]
        inside = si > 0
        rs = ((1 - fi[inside]) * si[inside]).sum()
        ss = (si[inside] ** 2).sum()
        scatter = np.std(fi[~inside], ddof=1)
        depth = rs / ss
        rows.append((seed, method, factor, depth, depth * np.sqrt(ss) / scatter / ideal,
                     t[ok], fi))
    return rows


def main():
    data = np.genfromtxt(CURVES, delimiter=",", names=True)
    items = []
    for seed in np.unique(data["seed"]):
        d = data[data["seed"] == seed]
        items.append((int(seed), float(d["t14"][0]), d["time"], d["flux"], d["shape"]))
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(measure, items, chunksize=4))
    with open(FLAT, "w", newline="\n") as out:
        out.write("seed,method,factor,time,flux\n")
        for rows in results:
            for seed, method, factor, _, _, t, f in rows:
                for ti, fi in zip(t, f):
                    out.write(f"{seed},{method},{factor!r},{float(ti)!r},{float(fi)!r}\n")
    with open(OUT, "w", newline="\n") as out:
        out.write("seed,method,factor,depth,snr\n")
        for rows in results:
            for seed, method, factor, depth, snr, _, _ in rows:
                out.write(f"{seed},{method},{factor!r},{float(depth)!r},{float(snr)!r}\n")
    print(len(items), "curves")


if __name__ == "__main__":
    main()
