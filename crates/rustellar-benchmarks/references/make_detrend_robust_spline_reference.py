"""The clipped spline (rustellar-detrend's `Method::RobustSpline`) run exactly
and with scipy, for its tests.

Each segment's knots as for `Method::Spline` (from the unmasked points). The
loop: fit, residuals, limit = sigma * 1.482602218505602 * MAD of the
residuals (about their median); keep the points with |residual| <= limit;
refit; until the set repeats, the limit is zero, or 50 refits.

- exact: every fit, residual, median and comparison in mpmath at 60 digits;
  the trend rounded once;
- scipy: the same loop in doubles, LSQUnivariateSpline and numpy medians.

Light curves with flares and dips, so the clipping matters. Cases where a
kept set leaves a coefficient undetermined are not written.

Writes crates/rustellar-detrend/tests/reference/robust_spline_cases.csv and
robust_spline_points.csv.

Run: python crates/rustellar-benchmarks/references/make_detrend_robust_spline_reference.py
"""

import math
import os
import sys

import mpmath as mp
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_detrend_spline_reference import basis, determined, knots, segments  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-detrend", "tests", "reference")
mp.mp.dps = 60
MAD_TO_SD = 1.482602218505602
MAX_CLIPS = 50


class Undetermined(Exception):
    pass


def light_curve(rng):
    n = int(rng.integers(100, 1200))
    steps = np.full(n, 2.0 / 1440) if rng.random() < 0.5 else rng.exponential(0.01, n)
    gaps = rng.random(n) < 0.003
    steps[gaps] = rng.uniform(0.3, 3.0, gaps.sum())
    t = 1400 + np.cumsum(steps)
    mask = np.zeros(n, bool)
    if rng.random() < 0.3:
        c = rng.uniform(t[0], t[-1])
        mask |= np.abs(t - c) < 0.05
    x = t - t[0]
    y = 1 + rng.uniform(-5e-3, 5e-3) * np.sin(2 * np.pi * x / rng.uniform(0.5, 6) + rng.uniform(0, 6))
    sd = rng.uniform(1e-4, 2e-3)
    y = y + sd * rng.standard_normal(n)
    # Flares: sharp rise, exponential decay.
    for _ in range(int(rng.integers(0, 4))):
        t0 = rng.uniform(t[0], t[-1])
        after = t >= t0
        y[after] += rng.uniform(5, 50) * sd * np.exp(-(t[after] - t0) / rng.uniform(0.005, 0.03))
    # Dips: boxes a tenth of a day.
    for _ in range(int(rng.integers(0, 4))):
        t0 = rng.uniform(t[0], t[-1])
        y[np.abs(t - t0) < 0.05] -= rng.uniform(3, 20) * sd
    # Single outliers.
    hit = rng.random(n) < 0.005
    y[hit] += rng.choice([-1, 1], hit.sum()) * rng.uniform(5, 30, hit.sum()) * sd
    if rng.random() < 0.25:
        y = 2.3e4 * y
    return t, y, mask


def exact_fit(start, step, intervals, xs, ys):
    m = intervals + 3
    A = mp.matrix(m, m)
    b = mp.matrix(m, 1)
    rows = []
    for x, y in zip(xs, ys):
        j, B = basis(start, step, intervals, x)
        rows.append((j, B))
        for k in range(4):
            b[j + k] += B[k] * mp.mpf(y)
            for l in range(4):
                A[j + k, j + l] += B[k] * B[l]
    return mp.lu_solve(A, b)


def value(c, start, step, intervals, x):
    j, B = basis(start, step, intervals, x)
    return sum(B[k] * c[j + k] for k in range(4))


def median(values):
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def exact_loop(start, step, intervals, xs, ys, sigma):
    inside = [True] * len(xs)
    if not determined(start, step, intervals, xs):
        raise Undetermined
    c = exact_fit(start, step, intervals, xs, ys)
    refits = 0
    for _ in range(MAX_CLIPS):
        r = [mp.mpf(y) - value(c, start, step, intervals, x) for x, y in zip(xs, ys)]
        centre = median(r)
        limit = mp.mpf(sigma) * mp.mpf(MAD_TO_SD) * median([abs(v - centre) for v in r])
        if limit == 0:
            break
        now = [abs(v) <= limit for v in r]
        if now == inside:
            break
        inside = now
        kx = [x for x, k in zip(xs, inside) if k]
        ky = [y for y, k in zip(ys, inside) if k]
        if not determined(start, step, intervals, kx):
            raise Undetermined
        c = exact_fit(start, step, intervals, kx, ky)
        refits += 1
    return c, inside, refits


def scipy_loop(start, step, intervals, first, last, xs, ys, sigma):
    inner = [start + j * step for j in range(1, intervals)]
    fit = lambda x, y: LSQUnivariateSpline(x, y, inner, bbox=[first, last], k=3)  # noqa: E731
    s = fit(xs, ys)
    inside = np.ones(len(xs), bool)
    for _ in range(MAX_CLIPS):
        r = ys - s(xs)
        limit = sigma * MAD_TO_SD * np.median(np.abs(r - np.median(r)))
        if limit == 0:
            break
        now = np.abs(r) <= limit
        if np.array_equal(now, inside):
            break
        inside = now
        s = fit(xs[inside], ys[inside])
    return s, inside


def main():
    rng = np.random.default_rng(20261121)
    cases, points = [], []
    case = 0
    while case < 40:
        t, y, mask = light_curve(rng)
        window = float(rng.choice([0.25, 0.5, 1.0, 0.37]))
        sigma = float(rng.choice([2.0, 3.0, 3.0, 5.0]))
        tolerance = window / 2
        out_scipy = [None] * len(t)
        out_exact = [None] * len(t)
        clipped = [0] * len(t)
        try:
            for a, b in segments(t, tolerance):
                kept = [k for k in range(a, b) if not mask[k]]
                if not kept:
                    continue
                first, last = float(t[a]), float(t[b - 1])
                start, step, intervals = knots(first, last, window, len(kept))
                xs, ys = t[kept], y[kept]
                c, inside, _ = exact_loop(start, step, intervals, xs, ys, sigma)
                s, _ = scipy_loop(start, step, intervals, first, last, xs, ys, sigma)
                for i in range(a, b):
                    out_exact[i] = float(value(c, start, step, intervals, t[i]))
                    out_scipy[i] = float(s(t[i]))
                for k, keep in zip(kept, inside):
                    clipped[k] = 0 if keep else 1
        except Undetermined:
            continue
        cases.append((case, window, tolerance, sigma))
        for i in range(len(t)):
            points.append((case, float(t[i]), float(y[i]), int(mask[i]), clipped[i],
                           out_scipy[i], out_exact[i]))
        case += 1
        print(case, end=" ", flush=True)
    with open(os.path.join(OUT, "robust_spline_cases.csv"), "w", newline="\n") as f:
        f.write("case,window,tolerance,sigma\n")
        for c in cases:
            f.write(f"{c[0]},{c[1]!r},{c[2]!r},{c[3]!r}\n")
    with open(os.path.join(OUT, "robust_spline_points.csv"), "w", newline="\n") as f:
        f.write("case,time,flux,masked,clipped,scipy,exact\n")
        for p in points:
            s = "" if p[5] is None else repr(p[5])
            e = "" if p[6] is None else repr(p[6])
            f.write(f"{p[0]},{p[1]!r},{p[2]!r},{p[3]},{p[4]},{s},{e}\n")
    print()
    print(len(cases), "cases", len(points), "points")


if __name__ == "__main__":
    main()
