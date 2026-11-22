"""numpy's median and the exact mean of every window, for rustellar-detrend's
location tests.

The windows are made here independently of the crate: segments and window
edges decided with exact `Fraction` times. For each point's window of
unmasked fluxes: numpy's `median`, and the mean as `Fraction`s, rounded once
to a double (int / int is correctly rounded in Python).

Fluxes of several kinds: normalised, counts, either sign around zero, a wide
range of magnitudes (where naive sums lose everything), subnormals, and
values drawn from a few (ties in the median, exact means).

Writes crates/rustellar-detrend/tests/reference/location_cases.csv and
location_points.csv.

Run: python crates/rustellar-benchmarks/references/make_detrend_location_reference.py
"""

import os
from fractions import Fraction

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-detrend", "tests", "reference")


def times_and_mask(rng):
    n = int(rng.integers(20, 160))
    steps = rng.exponential(0.03, n)
    gaps = rng.random(n) < 0.05
    steps[gaps] = rng.uniform(0.05, 1.5, gaps.sum())
    t = 1400 + np.cumsum(steps)
    mask = rng.random(n) < 0.1
    return t, mask


def fluxes(rng, kind, n):
    if kind == "normalised":
        return 1 + 1e-3 * rng.standard_normal(n)
    if kind == "counts":
        return 1e5 + 300 * rng.standard_normal(n)
    if kind == "signs":
        return 1e-3 * rng.standard_normal(n)
    if kind == "wide":
        # Magnitudes from 1e-300 to 1e300, either sign; no pair sums past the
        # largest double, so numpy's midpoint stays finite.
        return rng.choice([-1, 1], n) * 10.0 ** rng.uniform(-300, 300, n)
    if kind == "subnormal":
        return rng.choice([-1, 1], n) * rng.integers(1, 2**40, n) * 2.0**-1074
    if kind == "ties":
        return rng.choice([0.5, 1.0, 1.0, 1.25, 3.0], n)
    raise ValueError(kind)


def windows(t, mask, window, tolerance, cutoff, symmetric):
    """Each point's window: the indices of the unmasked points, or None."""
    T = [Fraction(float(x)) for x in t]
    half = Fraction(window) / 2
    cut = min(Fraction(cutoff), half)
    tol = None if tolerance == 0 else Fraction(tolerance)
    n = len(T)
    segment = [0] * n
    for k in range(1, n):
        segment[k] = segment[k - 1] + (tol is not None and T[k] - T[k - 1] > tol)
    out = []
    for i in range(n):
        mine = [j for j in range(n) if segment[j] == segment[i]]
        if T[i] - T[mine[0]] < cut or T[mine[-1]] - T[i] < cut:
            out.append(None)
            continue
        inside = []
        for j in mine:
            d = T[j] - T[i]
            if not mask[j] and -half <= d and (d < half or (d == half and symmetric)):
                inside.append(j)
        out.append(inside)
    return out


def text(x):
    return "nan" if x is None else repr(float(x))


def main():
    rng = np.random.default_rng(20261118)
    cases, points = [], []
    case = 0
    for kind in ["normalised", "counts", "signs", "wide", "subnormal", "ties"]:
        for window, tolerance, cutoff, symmetric in [
            (0.3, 0.15, 0.0, True),
            (0.5, 0.0, 0.1, True),
            (1.0, 0.2, 0.0, False),
            (0.2, 0.1, 5.0, True),
            (0.6, 0.3, 0.05, False),
        ]:
            t, mask = times_and_mask(rng)
            f = fluxes(rng, kind, len(t))
            for i, inside in enumerate(windows(t, mask, window, tolerance, cutoff, symmetric)):
                if not inside:
                    median = mean = None
                else:
                    values = [float(f[j]) for j in inside]
                    median = float(np.median(np.array(values)))
                    exact = sum(Fraction(v) for v in values) / len(values)
                    mean = exact.numerator / exact.denominator
                points.append(
                    (case, i, repr(float(t[i])), repr(float(f[i])), int(mask[i]), text(median), text(mean))
                )
            edges = "symmetric" if symmetric else "half_open"
            cases.append((case, kind, window, tolerance, cutoff, edges, len(t)))
            case += 1
    header = (
        "# Each window's numpy median and exact mean (Fractions, rounded once), the windows "
        "made with exact Fraction times. break_tolerance 0 means none.\n"
    )
    with open(os.path.join(OUT, "location_cases.csv"), "w", newline="\n") as fh:
        fh.write(header)
        fh.write("case,kind,window,break_tolerance,edge_cutoff,edges,points\n")
        for row in cases:
            fh.write(",".join(str(v) for v in row) + "\n")
    with open(os.path.join(OUT, "location_points.csv"), "w", newline="\n") as fh:
        fh.write(header)
        fh.write("# median and mean nan where the point has no window or its window no point\n")
        fh.write("case,point,time,flux,masked,median,mean\n")
        for row in points:
            fh.write(",".join(str(v) for v in row) + "\n")
    print(len(cases), "cases,", len(points), "points")


if __name__ == "__main__":
    main()
