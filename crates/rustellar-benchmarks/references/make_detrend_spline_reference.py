"""scipy's least-squares cubic spline and an exact one, for rustellar-detrend's
`Method::Spline` tests.

Each light curve is split at gaps longer than the break tolerance (decided
with exact `Fraction` times); each segment's knots run from its first time to
its last, ceil(span / window) intervals (at most the unmasked points), evenly
spaced. On those knots:

- scipy's `LSQUnivariateSpline` (FITPACK, double precision) through the
  unmasked points, evaluated at every point;
- the same least-squares spline solved with mpmath at 60 digits (uniform
  cubic B-splines on knots start + j step, step the double), rounded once.

Segments whose points leave a coefficient undetermined are not written:
there the crate adds a smoothing term that scipy does not have.

Writes crates/rustellar-detrend/tests/reference/spline_cases.csv and
spline_points.csv.

Run: python crates/rustellar-benchmarks/references/make_detrend_spline_reference.py
"""

import math
import os
import time as clock
from fractions import Fraction

import mpmath as mp
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-detrend", "tests", "reference")
mp.mp.dps = 60


def light_curve(rng, kind):
    n = int(rng.integers(40, 1500))
    steps = np.full(n, 2.0 / 1440) if rng.random() < 0.5 else rng.exponential(0.01, n)
    gaps = rng.random(n) < 0.003
    steps[gaps] = rng.uniform(0.3, 3.0, gaps.sum())
    t = 1400 + np.cumsum(steps)
    mask = np.zeros(n, bool)
    if rng.random() < 0.5:
        # A few transits masked.
        for c in rng.uniform(t[0], t[-1], 3):
            mask |= np.abs(t - c) < rng.uniform(0.02, 0.1)
    x = t - t[0]
    smooth = (
        rng.uniform(-5e-3, 5e-3) * np.sin(2 * np.pi * x / rng.uniform(0.5, 6) + rng.uniform(0, 6))
        + rng.uniform(-2e-3, 2e-3) * x / max(x[-1], 1e-9)
    )
    noise = rng.uniform(1e-5, 2e-3) * rng.standard_normal(n)
    if kind == "normalised":
        y = 1 + smooth + noise
    elif kind == "counts":
        y = 2.3e4 * (1 + smooth + noise)
    elif kind == "signs":
        y = smooth + noise
    else:  # "huge": counts scaled up, where a careless sum overflows nothing
        y = 1e200 * (1 + smooth + noise)
    return t, y, mask


def segments(t, tolerance):
    T = [Fraction(float(v)) for v in t]
    tol = Fraction(tolerance) if math.isfinite(tolerance) else None
    out, start = [], 0
    for k in range(1, len(t)):
        if tol is not None and T[k] - T[k - 1] > tol:
            out.append((start, k))
            start = k
    out.append((start, len(t)))
    return out


def knots(first, last, window, points):
    span = last - first
    if span <= 0:
        return first, window, 1
    wanted = math.ceil(span / window)
    intervals = max(points, 1) if wanted >= points else max(wanted, 1)
    return first, span / intervals, intervals


def basis(start, step, intervals, x):
    """The four uniform cubic B-splines at x, in mpmath, and the first."""
    at = (mp.mpf(x) - mp.mpf(start)) / mp.mpf(step)
    j = min(max(int(mp.floor(at)), 0), intervals - 1)
    u = at - j
    u = min(max(u, mp.mpf(0)), mp.mpf(1))
    v = 1 - u
    return j, [v**3 / 6, (3 * u**3 - 6 * u**2 + 4) / 6, (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6, u**3 / 6]


def exact(start, step, intervals, xs, ys, at):
    m = intervals + 3
    A = mp.matrix(m, m)
    b = mp.matrix(m, 1)
    for x, y in zip(xs, ys):
        j, B = basis(start, step, intervals, x)
        for k in range(4):
            b[j + k] += B[k] * mp.mpf(y)
            for l in range(4):
                A[j + k, j + l] += B[k] * B[l]
    c = mp.lu_solve(A, b)
    out = []
    for x in at:
        j, B = basis(start, step, intervals, x)
        out.append(float(sum(B[k] * c[j + k] for k in range(4))))
    return out


def determined(start, step, intervals, xs):
    """Whether every coefficient is pinned down: full column rank, well
    conditioned."""
    m = intervals + 3
    D = np.zeros((len(xs), m))
    for i, x in enumerate(xs):
        j, B = basis(start, step, intervals, x)
        D[i, j : j + 4] = [float(v) for v in B]
    if len(xs) < m:
        return False
    s = np.linalg.svd(D, compute_uv=False)
    return s[-1] > 1e-6 * s[0]


def main():
    rng = np.random.default_rng(20261120)
    kinds = ["normalised", "counts", "signs", "huge"]
    cases, points = [], []
    scipy_seconds = 0.0
    case = 0
    while case < 60:
        kind = kinds[case % 4]
        t, y, mask = light_curve(rng, kind)
        window = float(rng.choice([0.25, 0.5, 1.0, 2.0, 0.37]))
        tolerance = float(rng.choice([window / 2, 1.0, math.inf]))
        cutoff = float(rng.choice([0.0, 0.0, 0.1]))
        segs = segments(t, tolerance)
        out_scipy = [None] * len(t)
        out_exact = [None] * len(t)
        ok = True
        for a, b in segs:
            kept = [k for k in range(a, b) if not mask[k]]
            if not kept:
                continue
            start, step, intervals = knots(float(t[a]), float(t[b - 1]), window, len(kept))
            xs, ys = t[kept], y[kept]
            if not determined(start, step, intervals, xs):
                ok = False
                break
            interior = [start + j * step for j in range(1, intervals)]
            began = clock.perf_counter()
            s = LSQUnivariateSpline(xs, ys, interior, bbox=[float(t[a]), float(t[b - 1])], k=3)
            vals = s(t[a:b])
            scipy_seconds += clock.perf_counter() - began
            ex = exact(start, step, intervals, xs, ys, t[a:b])
            half = Fraction(window) / 2
            cut = min(Fraction(cutoff), half)
            for i in range(a, b):
                T = Fraction(float(t[i]))
                if T - Fraction(float(t[a])) < cut or Fraction(float(t[b - 1])) - T < cut:
                    continue
                out_scipy[i] = float(vals[i - a])
                out_exact[i] = ex[i - a]
        if not ok:
            continue
        cases.append((case, kind, window, tolerance, cutoff))
        for i in range(len(t)):
            points.append((case, float(t[i]), float(y[i]), int(mask[i]), out_scipy[i], out_exact[i]))
        case += 1
    with open(os.path.join(OUT, "spline_cases.csv"), "w", newline="\n") as f:
        f.write(f"# scipy LSQUnivariateSpline total {scipy_seconds:.4f} s\n")
        f.write("case,kind,window,tolerance,cutoff\n")
        for c in cases:
            tol = "inf" if math.isinf(c[3]) else repr(c[3])
            f.write(f"{c[0]},{c[1]},{c[2]!r},{tol},{c[4]!r}\n")
    with open(os.path.join(OUT, "spline_points.csv"), "w", newline="\n") as f:
        f.write("case,time,flux,masked,scipy,exact\n")
        for p in points:
            s = "" if p[4] is None else repr(p[4])
            e = "" if p[5] is None else repr(p[5])
            f.write(f"{p[0]},{p[1]!r},{p[2]!r},{p[3]},{s},{e}\n")
    print(len(cases), "cases", len(points), "points", f"scipy {scipy_seconds:.3f} s")


if __name__ == "__main__":
    main()
