"""wotan's `huber` (statsmodels' Huber proposal 2: location and scale
estimated jointly, c = 1.5) at the survey's sampled points, against the
exact solution of proposal 2's equations on wotan's own window.

The equations, at a solution (mu, s), with I the points where
|x - mu| <= c s, n_lo / n_hi the points below / above, N points, n = N - 1:

    mu = (sum_I x + c s (n_hi - n_lo)) / |I|    (clipped points sit at mu +- c s)
    n gamma s^2 = sum_I (x - mu)^2 + (n_lo + n_hi) (c s)^2

For a fixed partition, mu is linear in s and the second is a quadratic in s:
solved at 60 digits, then the partition checked against the solution and
redone until it holds. gamma as statsmodels defines it, at 60 digits.

Prints the comparison; writes nothing to the repo (the crate has no
proposal-2 estimator to test). Run from a file (multiprocessing):

    python crates/rustellar-benchmarks/references/make_detrend_huber_reference.py
"""

import importlib.util
import math
import os
import sys
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
from mpmath import mp, mpf, ncdf, npdf, sqrt
from statsmodels.robust.scale import Huber
from wotan import flatten

mp.dps = 60
C = 1.5
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


survey = load("make_detrend_survey_reference")

c = mpf(C)
tmp = 2 * ncdf(c) - 1
GAMMA = tmp + c**2 * (1 - tmp) - 2 * c * npdf(c)


def solve(x, mu, s):
    """The exact solution near (mu, s): partition, solve, repeat."""
    N = len(x)
    n = N - 1
    seen = set()
    for _ in range(200):
        inside = tuple(abs(v - mu) <= c * s for v in x)
        lo = sum(1 for v, k in zip(x, inside) if not k and v < mu)
        hi = sum(1 for v, k in zip(x, inside) if not k and v > mu)
        key = (inside, lo, hi)
        if key in seen:
            return None
        seen.add(key)
        xs = [v for v, k in zip(x, inside) if k]
        m = len(xs)
        A = sum(xs) / m
        B = c * (hi - lo) / m
        d = [v - A for v in xs]
        qa = n * GAMMA - (lo + hi) * c**2 - m * B**2
        qb = 2 * B * sum(d)
        qc = -sum(v * v for v in d)
        s_new = (-qb + sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
        mu_new = A + B * s_new
        again = tuple(abs(v - mu_new) <= c * s_new for v in x)
        lo2 = sum(1 for v, k in zip(x, again) if not k and v < mu_new)
        hi2 = sum(1 for v, k in zip(x, again) if not k and v > mu_new)
        if (again, lo2, hi2) == key:
            return mu_new, s_new
        mu, s = mu_new, s_new
    return None


def statsmodels(values):
    """statsmodels' Huber as wotan calls it, or None where it raises (wotan
    then takes the median)."""
    try:
        return Huber(maxiter=1000, tol=1e-6, c=C)(np.array(values))
    except Exception:
        return None


def ulp(v):
    v = abs(v)
    return math.nextafter(v, math.inf) - v


def sample(job):
    curve, i, values, wotan = job
    got = statsmodels(values)
    x = [mpf(v) for v in values]
    if got is None:
        start = (mpf(float(np.median(values))), mpf(float(np.median(np.abs(np.array(values) - np.median(values))) / 0.6744897501960817)))
    else:
        start = (mpf(float(got[0])), mpf(float(got[1])))
    root = solve(x, *start)
    if root is None:
        return (curve, i, None, None, None, got is None, wotan)
    mu, s = root
    # A fixed point of statsmodels' own update, to 40 digits (far below a double).
    nmu = sum(min(max(v, mu - c * s), mu + c * s) for v in x) / len(x)
    num = sum((v - nmu) ** 2 if abs(v - mu) <= c * s else (c * s) ** 2 for v in x)
    nscale = sqrt(num / ((len(x) - 1) * GAMMA))
    assert abs(nmu - mu) < mpf(10) ** -40 * s and abs(nscale - s) < mpf(10) ** -40 * s, (curve, i)
    error = abs(mpf(wotan) - mu)
    return (curve, i, float(error / mpf(ulp(float(mu)))), float(error / s), float(error / abs(mu)), got is None, wotan)


def main():
    jobs = []
    for row in survey.CURVES:
        p = dict(zip(survey.FIELDS, row))
        time, flux, _ = survey.light_curve(p)
        window = p["window"]
        trend = flatten(
            np.array(time), np.array(flux), method="huber", window_length=window,
            break_tolerance=window / 2, edge_cutoff=0, return_trend=True,
        )[1]
        half = Fraction(window) / 2
        starts = survey.segments(time, window / 2)
        for lo, hi in zip(starts, starts[1:]):
            for i in range(lo, hi):
                if i % p["every"]:
                    continue
                theirs = survey.wotans(time, lo, hi, i, half)
                jobs.append((p["name"], i, [flux[j] for j in theirs], float(trend[i])))
        print(p["name"], "done", flush=True)
    with Pool(4) as pool:
        rows = pool.map(sample, jobs, chunksize=4)
    unsolved = [r for r in rows if r[2] is None]
    fell = [r for r in rows if r[5]]
    good = [r for r in rows if r[2] is not None]
    for name in [p[0] for p in survey.CURVES]:
        mine = [r for r in good if r[0] == name]
        ulps = sorted(r[2] for r in mine)
        scale = sorted(r[3] for r in mine)
        rel = sorted(r[4] for r in mine)
        print(
            f"{name}: {len(mine)} windows; error in ulp median {ulps[len(ulps)//2]:.3e} max {ulps[-1]:.3e}; "
            f"in scales median {scale[len(scale)//2]:.2e} max {scale[-1]:.2e}; relative max {rel[-1]:.2e}; "
            f"nearest double {sum(1 for u in ulps if u <= 0.5)}"
        )
    print(len(rows), "windows,", len(unsolved), "unsolved,", len(fell), "fell back to the median")
    for r in fell[:10]:
        print("fallback", r)
    for r in unsolved[:10]:
        print("unsolved", r[:2])


if __name__ == "__main__":
    sys.exit(main())
