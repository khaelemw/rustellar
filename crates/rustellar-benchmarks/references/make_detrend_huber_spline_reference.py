"""The Huber spline (rustellar-detrend's `Method::HuberSpline`) at its exact
fixed point and with statsmodels, for its tests.

Each segment's knots as for `Method::Spline`. The estimate: the spline whose
coefficients are the weighted least-squares fit with Huber weights
min(1, c s / |r|) of its own residuals r, s = 1.482602218505602 times the
median of |r| (statsmodels' `RLM` with `HuberT(c)` and its default `mad`
scale, which `RLM` takes about zero, refitted each iteration).

- exact: iterated in mpmath at 50 digits from the least-squares fit until
  the zones (inside c s, or beyond it on either side) and the residuals
  that make the two medians are settled, then the estimating equations
  solved with them held (linear in the coefficients and the scale), and
  checked to hold them; rounded once;
- statsmodels: `RLM(y, X, HuberT(c)).fit()` on the B-spline design, at its
  default tolerance (1e-8 on the deviance) and at 1e-14.

Writes crates/rustellar-detrend/tests/reference/huber_spline_cases.csv and
huber_spline_points.csv.

Run: python crates/rustellar-benchmarks/references/make_detrend_huber_spline_reference.py
"""

import os
import sys

import mpmath as mp
import numpy as np
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_detrend_spline_reference import basis, determined, knots, segments  # noqa: E402
from make_detrend_robust_spline_reference import light_curve, median, value  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-detrend", "tests", "reference")
mp.mp.dps = 50
MAD_TO_SD = 1.482602218505602


def weighted_fit(start, step, intervals, rows, ys, ws):
    m = intervals + 3
    A = mp.matrix(m, m)
    b = mp.matrix(m, 1)
    for (j, B), y, w in zip(rows, ys, ws):
        for k in range(4):
            b[j + k] += w * B[k] * mp.mpf(y)
            for l in range(4):
                A[j + k, j + l] += w * B[k] * B[l]
    return mp.lu_solve(A, b)


def order_indices(values):
    """The indices of the two middle values (the same twice for an odd
    count), by value."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    n = len(values)
    return order[(n - 1) // 2], order[n // 2]


def state(r, c):
    """What fixes the estimating equations near a solution: the indices of
    the residuals whose magnitudes make the median, their signs, and each
    point's zone (0 inside c s, +1 or -1 beyond it on that side)."""
    mag = [abs(v) for v in r]
    k1, k2 = order_indices(mag)
    signs = (1 if r[k1] >= 0 else -1, 1 if r[k2] >= 0 else -1)
    s = mp.mpf(MAD_TO_SD) * (mag[k1] + mag[k2]) / 2
    zone = [0 if abs(v) <= c * s else (1 if v > 0 else -1) for v in r]
    return (k1, k2, signs, tuple(zone)), s


def solve_state(start, step, intervals, rows, ys, c, st):
    """The coefficients and scale satisfying the estimating equations with
    the state `st` held: X_in^T (y_in - X_in b) + c s X_out^T zone = 0 and
    s = MAD_TO_SD * (sign_1 r_k1 + sign_2 r_k2) / 2, both linear in (b, s)."""
    k1, k2, signs, zone = st
    m = intervals + 3
    A = mp.matrix(m + 1, m + 1)
    rhs = mp.matrix(m + 1, 1)
    for i, ((j, B), y) in enumerate(zip(rows, ys)):
        if zone[i] == 0:
            for k in range(4):
                rhs[j + k] += B[k] * mp.mpf(y)
                for l in range(4):
                    A[j + k, j + l] += B[k] * B[l]
        else:
            for k in range(4):
                A[j + k, m] -= c * zone[i] * B[k]
    K = mp.mpf(MAD_TO_SD) / 2
    A[m, m] = 1
    for idx, w in ((k1, signs[0]), (k2, signs[1])):
        j, B = rows[idx]
        rhs[m] += K * w * mp.mpf(ys[idx])
        for k in range(4):
            A[m, j + k] += K * w * B[k]
    x = mp.lu_solve(A, rhs)
    return [x[i] for i in range(m)], x[m]


def exact_huber(start, step, intervals, xs, ys, c):
    c = mp.mpf(c)
    rows = [basis(start, step, intervals, x) for x in xs]
    coef = weighted_fit(start, step, intervals, rows, ys, [1] * len(xs))
    residuals = lambda co: [mp.mpf(y) - sum(B[k] * co[j + k] for k in range(4)) for (j, B), y in zip(rows, ys)]  # noqa: E731
    for it in range(400):
        r = residuals(coef)
        st, s = state(r, c)
        if s == 0:
            return coef, it
        # Try the exact finish from this state.
        sol, s2 = solve_state(start, step, intervals, rows, ys, c, st)
        st2, s3 = state(residuals(sol), c)
        if st2 == st and abs(s3 - s2) <= abs(s2) * mp.mpf("1e-40"):
            return sol, it
        ws = [1 if v == 0 else min(mp.mpf(1), c * s / abs(v)) for v in r]
        coef = weighted_fit(start, step, intervals, rows, ys, ws)
    raise RuntimeError("no convergence")


def design(start, step, intervals, xs):
    X = np.zeros((len(xs), intervals + 3))
    for i, x in enumerate(xs):
        j, B = basis(start, step, intervals, x)
        X[i, j:j + 4] = [float(v) for v in B]
    return X


def main():
    rng = np.random.default_rng(20261122)
    cases, points = [], []
    case = 0
    worst_iterations = 0
    while case < 30:
        t, y, mask = light_curve(rng)
        if len(t) > 700:
            continue
        window = float(rng.choice([0.25, 0.5, 1.0, 0.37]))
        c = float(rng.choice([1.0, 1.345, 1.345, 2.0]))
        tolerance = window / 2
        out = {k: [None] * len(t) for k in ("exact", "sm", "sm_tight")}
        ok = True
        for a, b in segments(t, tolerance):
            kept = [k for k in range(a, b) if not mask[k]]
            if not kept:
                continue
            start, step, intervals = knots(float(t[a]), float(t[b - 1]), window, len(kept))
            xs, ys = t[kept], y[kept]
            if not determined(start, step, intervals, xs):
                ok = False
                break
            coef, its = exact_huber(start, step, intervals, xs, ys, c)
            worst_iterations = max(worst_iterations, its)
            X = design(start, step, intervals, xs)
            Xall = design(start, step, intervals, t[a:b])
            fits = {
                "sm": sm.RLM(ys, X, M=sm.robust.norms.HuberT(t=c)).fit(),
                "sm_tight": sm.RLM(ys, X, M=sm.robust.norms.HuberT(t=c)).fit(tol=1e-14, maxiter=1000),
            }
            for i in range(a, b):
                out["exact"][i] = float(value(coef, start, step, intervals, t[i]))
            for key, f in fits.items():
                vals = Xall @ f.params
                for i in range(a, b):
                    out[key][i] = float(vals[i - a])
        if not ok:
            continue
        cases.append((case, window, tolerance, c))
        for i in range(len(t)):
            points.append((case, float(t[i]), float(y[i]), int(mask[i]),
                           out["sm"][i], out["sm_tight"][i], out["exact"][i]))
        case += 1
        print(case, end=" ", flush=True)
    with open(os.path.join(OUT, "huber_spline_cases.csv"), "w", newline="\n") as f:
        f.write("case,window,tolerance,c\n")
        for k in cases:
            f.write(f"{k[0]},{k[1]!r},{k[2]!r},{k[3]!r}\n")
    with open(os.path.join(OUT, "huber_spline_points.csv"), "w", newline="\n") as f:
        f.write("case,time,flux,masked,statsmodels,statsmodels_tight,exact\n")
        for p in points:
            cells = ["" if v is None else repr(v) for v in p[4:]]
            f.write(f"{p[0]},{p[1]!r},{p[2]!r},{p[3]},{','.join(cells)}\n")
    print()
    print(len(cases), "cases", len(points), "points; most iterations", worst_iterations)


if __name__ == "__main__":
    main()
