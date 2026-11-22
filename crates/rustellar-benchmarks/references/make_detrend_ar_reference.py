"""statsmodels' AutoReg and ar_select_order, and an exact least-squares fit,
for rustellar-detrend's `Autoregressive` tests.

Each case is a regular series (no gaps: statsmodels has no notion of them)
of an autoregressive process, flattened-light-curve-like (white noise and a
short leftover wave), counts, or close to a unit root, with:

- a fixed order: statsmodels `AutoReg(x, p, trend="c").fit()` parameters
  and sigma2, and the same least squares solved with mpmath at 60 digits
  (normal equations, rounded once);
- `ar_select_order(x, maxlag, ic="aic", trend="c")`: the order chosen, and
  the refit at it (statsmodels and exact).

Parameters are written space-separated, constant first.

Writes crates/rustellar-detrend/tests/reference/ar_cases.csv and
ar_points.csv.

Run: python crates/rustellar-benchmarks/references/make_detrend_ar_reference.py
"""

import os
import time as clock
import warnings

import mpmath as mp
import numpy as np
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-detrend", "tests", "reference")
mp.mp.dps = 60


def series(rng, kind):
    n = int(rng.integers(60, 3000))
    p = int(rng.integers(1, 5))
    if kind == "unit root":
        phi = np.array([0.98])
    else:
        # Stable: roots of the characteristic polynomial outside the circle.
        roots = rng.uniform(1.2, 4.0, p) * rng.choice([-1, 1], p)
        poly = np.poly(1 / roots)  # 1 - phi_1 z - ...
        phi = -poly[1:]
    e = rng.uniform(1e-4, 2e-3) * rng.standard_normal(n + 200)
    x = np.zeros(n + 200)
    for t in range(len(phi), n + 200):
        x[t] = phi @ x[t - len(phi):t][::-1] + e[t]
    x = x[200:]
    if kind == "flattened":
        k = np.arange(n)
        x = x + rng.uniform(0, 1e-3) * np.sin(2 * np.pi * k / rng.uniform(5, 60))
    x = 1 + x
    if kind == "counts":
        x = 2.3e4 * x
    return x


def exact(x, p):
    """Least squares of x_t on [1, x_(t-1), .., x_(t-p)], t >= p, exactly."""
    n = len(x)
    xs = [mp.mpf(float(v)) for v in x]
    cols = p + 1
    ata = mp.zeros(cols, cols)
    aty = mp.zeros(cols, 1)
    for t in range(p, n):
        row = [mp.mpf(1)] + [xs[t - 1 - k] for k in range(p)]
        for i in range(cols):
            aty[i] += row[i] * xs[t]
            for j in range(cols):
                ata[i, j] += row[i] * row[j]
    beta = mp.lu_solve(ata, aty)
    return [float(beta[i]) for i in range(cols)]


def join(values):
    return " ".join(repr(float(v)) for v in values)


def main():
    rng = np.random.default_rng(20261120)
    kinds = ["normalised", "flattened", "counts", "unit root"]
    cases, points = [], []
    spent = 0.0
    for case in range(48):
        kind = kinds[case % len(kinds)]
        x = series(rng, kind)
        fixed = int(rng.integers(0, 7))
        most = int(rng.integers(1, 13))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = clock.perf_counter()
            fit = AutoReg(x, fixed, trend="c").fit()
            chosen = ar_select_order(x, most, ic="aic", trend="c")
            refit = chosen.model.fit()
            spent += clock.perf_counter() - start
        lag = 0 if chosen.ar_lags is None else max(chosen.ar_lags)
        cases.append(
            f"{case},{kind},{fixed},{join(fit.params)},{join(exact(x, fixed))},"
            f"{float(fit.sigma2)!r},{most},{lag},{join(refit.params)},{join(exact(x, lag))}"
        )
        points.extend(f"{case},{float(v)!r}" for v in x)
    with open(os.path.join(OUT, "ar_cases.csv"), "w", newline="\n") as out:
        out.write(f"# statsmodels AutoReg + ar_select_order total {spent:.4f} s\n")
        out.write("case,kind,fixed,statsmodels,exact,sigma2,most,chosen,statsmodels_chosen,exact_chosen\n")
        out.write("\n".join(cases) + "\n")
    with open(os.path.join(OUT, "ar_points.csv"), "w", newline="\n") as out:
        out.write("case,flux\n")
        out.write("\n".join(points) + "\n")
    print(len(cases), "cases,", len(points), "points")


if __name__ == "__main__":
    main()
