"""Splines' speed on the 2-minute TESS sector of rustellar-detrend's benchmark
(the survey recipe), for the README: scipy's LSQUnivariateSpline on the
crate's knots (segments split at half the window, ceil(span / window)
intervals each), and wotan's rspline and hspline at the same window.

Each timing is the best of 3 after a warm-up call; hspline (slow) once.

Run: python crates/rustellar-benchmarks/references/bench_detrend_spline.py
"""

import math
import os
import sys
import time

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_detrend_wotan import curves  # noqa: E402
from make_detrend_survey_reference import light_curve  # noqa: E402
from wotan import flatten  # noqa: E402


def scipy_spline(t, f, window):
    out = np.empty_like(f)
    cuts = np.flatnonzero(np.diff(t) > window / 2) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        x, y = t[a:b], f[a:b]
        span = x[-1] - x[0]
        n = min(len(x), max(1, math.ceil(span / window)))
        step = span / n
        inner = [x[0] + j * step for j in range(1, n)]
        out[a:b] = LSQUnivariateSpline(x, y, inner, k=3)(x)
    return out


def best(run, repeats=3):
    run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    p = curves()[1]
    t, f, _ = light_curve(p)
    t, f = np.array(t), np.array(f)
    w = p["window"]
    print(f"{p['name']} ({len(t)} points), window {w}")
    print(f"scipy LSQUnivariateSpline: {best(lambda: scipy_spline(t, f, w)):.5f} s")
    print(f"statsmodels RLM HuberT (design built outside the timing): ", end="")
    import statsmodels.api as sm
    from make_detrend_huber_spline_reference import design
    cuts = np.flatnonzero(np.diff(t) > w / 2) + 1
    parts = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        x = t[a:b]
        span = x[-1] - x[0]
        n = min(len(x), max(1, math.ceil(span / w)))
        parts.append((design(x[0], span / n, n, x), f[a:b]))
    fit = lambda: [sm.RLM(y, X, M=sm.robust.norms.HuberT(t=1.345)).fit() for X, y in parts]  # noqa: E731
    print(f"{best(fit):.5f} s", flush=True)
    for method, repeats in [("rspline", 3), ("hspline", 1)]:
        run = lambda: flatten(t, f, method=method, window_length=w,  # noqa: E731
                              break_tolerance=w / 2, edge_cutoff=0, return_trend=True)
        print(f"wotan {method}: {best(run, repeats):.5f} s", flush=True)


if __name__ == "__main__":
    main()
