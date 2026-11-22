"""wotan 1.10's windows on random light curves, for rustellar-detrend's window
tests.

wotan's own code is run, segmenting, masking and sliding included: `flatten`
with method "mean", its numba slider swapped for the same function in plain
Python (`py_func`), and the slider's `mean` swapped for a recorder. Each
point's flux is its index plus one, so the fluxes a window averages are the
indices of the points in it (masked points are made negative by wotan and
dropped). The recorder notes the first and last index and the count, and
returns a call number, which comes back as the point's trend.

Writes crates/rustellar-detrend/tests/reference/wotan_window_cases.csv and
wotan_window_points.csv.

Run: python crates/rustellar-benchmarks/references/make_detrend_window_reference.py
"""

import os
import sys

import numpy as np

import wotan  # noqa: F401  (loads the modules swapped below)
import wotan.slider as slider

flatten_module = sys.modules["wotan.flatten"]
flatten = flatten_module.flatten

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-detrend", "tests", "reference")

calls = []


def record(f):
    calls.append((int(round(f.min())) - 1, int(round(f.max())) - 1, len(f)))
    return float(len(calls))  # never 0, which wotan turns into NaN


slider.mean = record
flatten_module.running_segment = slider.running_segment.py_func


def light_curve(rng, kind):
    """Times (strictly increasing) and a mask for a case of `kind`."""
    if kind == "survey":
        # Two orbits of 30-minute cadence with a mid-sector gap and a few
        # dropped points.
        cadence = 1 / 48
        t = np.concatenate([1325 + cadence * np.arange(300), 1333 + cadence * np.arange(300)])
        t = t[rng.random(len(t)) > 0.05]
    elif kind == "jitter":
        # Uneven spacing with gaps of every length around the tolerances.
        steps = rng.exponential(0.03, 400)
        gaps = rng.random(400) < 0.05
        steps[gaps] = rng.uniform(0.05, 1.5, gaps.sum())
        t = 1400 + np.cumsum(steps)
    else:
        # Few points, far apart: windows of one or none.
        t = 1500 + np.cumsum(rng.uniform(0.05, 0.8, 25))
    n = len(t)
    mask = rng.random(n) < 0.1
    # A masked run, like a masked transit.
    start = rng.integers(0, max(1, n - 20))
    mask[start : start + rng.integers(1, 20)] = True
    return t, mask


def main():
    rng = np.random.default_rng(20261117)
    cases, points = [], []
    case = 0
    for kind in ["survey", "jitter", "sparse"]:
        for w, window in enumerate([0.3, 0.5, 1.0]):
            for b, break_tolerance in enumerate([None, 0.2, 0]):
                for e, edge_cutoff in enumerate([0, 0.1, 5.0]):
                    # The survey curves are long: a third of the combinations
                    # (a Latin square: each pair of settings once).
                    if kind == "survey" and (w + b + e) % 3 != 0:
                        continue
                    t, mask = light_curve(rng, kind)
                    flux = np.arange(len(t), dtype=float) + 1
                    calls.clear()
                    _, trend = flatten(
                        t,
                        flux,
                        method="mean",
                        window_length=window,
                        break_tolerance=break_tolerance,
                        edge_cutoff=edge_cutoff,
                        mask=mask.copy(),
                        return_trend=True,
                    )
                    # wotan's break tolerance: half the window by default, and
                    # 0 for none.
                    tolerance = window / 2 if break_tolerance is None else break_tolerance
                    cases.append((case, kind, window, tolerance, edge_cutoff, len(t)))
                    for i in range(len(t)):
                        if np.isnan(trend[i]):
                            first, last, count = -1, -1, 0
                        else:
                            first, last, count = calls[int(trend[i]) - 1]
                        points.append((case, i, repr(float(t[i])), int(mask[i]), first, last, count))
                    case += 1
    header = (
        "# wotan 1.10's windows: flatten(method='mean') with its own segmenting, masking "
        "and slider (in plain Python), each flux the point's index plus one. "
        "break_tolerance 0 means none.\n"
    )
    with open(os.path.join(OUT, "wotan_window_cases.csv"), "w", newline="\n") as f:
        f.write(header)
        f.write("case,kind,window,break_tolerance,edge_cutoff,points\n")
        for row in cases:
            f.write(",".join(str(v) for v in row) + "\n")
    with open(os.path.join(OUT, "wotan_window_points.csv"), "w", newline="\n") as f:
        f.write(header)
        f.write("# first and last: the indices of the window's first and last points; -1 for no trend\n")
        f.write("case,point,time,masked,first,last,count\n")
        for row in points:
            f.write(",".join(str(v) for v in row) + "\n")
    print(len(cases), "cases,", len(points), "points")


if __name__ == "__main__":
    main()
