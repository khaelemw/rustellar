"""Huber's proposal 2 (location and scale jointly, c = 1.5, n - 1) at the
survey's sampled points, for rustellar-detrend's survey test: the exact
solution of the two equations on the crate's window at 60 digits
(`make_detrend_huber_reference.solve`, started from statsmodels' answer),
and wotan 1.10's `method='huber'` trend there with the exact solution on
wotan's own window.

Writes crates/rustellar-detrend/tests/reference/survey_huber_joint.csv. Run
from a file (multiprocessing):

    python crates/rustellar-benchmarks/references/make_detrend_survey_huber_joint_reference.py
"""

import importlib.util
import math
import os
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
from mpmath import mpf
from wotan import flatten

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "crates", "rustellar-detrend", "tests", "reference")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


survey = load("make_detrend_survey_reference")
estimators = load("make_detrend_estimators_reference")
huber = load("make_detrend_huber_reference")


def exact(values):
    """The 60-digit solution (mu, s) on `values`, from statsmodels' answer or,
    where statsmodels raises, the median and MAD."""
    got = huber.statsmodels(values)
    if got is None:
        med = float(np.median(values))
        start = (mpf(med), mpf(float(np.median(np.abs(np.array(values) - med)) / 0.6744897501960817)))
    else:
        start = (mpf(float(got[0])), mpf(float(got[1])))
    root = huber.solve([mpf(v) for v in values], *start)
    assert root is not None, "unsolved"
    return root


def sample(job):
    curve, i, values, wotan, theirs = job
    mu, s = exact(values)
    e = math.frexp(max(abs(v) for v in values))[1] - 1
    factor = mpf(2) ** -e
    hi, lo = estimators.hi_lo(mu * factor)
    if theirs is None:
        w_hi, w_lo = hi, lo
    else:
        w_hi, w_lo = estimators.hi_lo(exact(theirs)[0] * factor)
    return (
        f"{curve},{i},{e},{hi!r},{lo!r},{float(s * factor)!r},"
        f"{wotan!r},{int(theirs is None)},{w_hi!r},{w_lo!r},{len(values)}"
    )


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
                mine = survey.ours(time, lo, hi, i, half)
                theirs = survey.wotans(time, lo, hi, i, half)
                other = None if mine == theirs else [flux[j] for j in theirs]
                jobs.append((p["name"], i, [flux[j] for j in mine], float(trend[i]), other))
        print(p["name"], len(time), "points", flush=True)
    with Pool(4) as pool:
        rows = pool.map(sample, jobs, chunksize=4)
    with open(os.path.join(OUT, "survey_huber_joint.csv"), "w", newline="\n") as fh:
        fh.write(
            "# Sampled points: Huber's proposal 2 (c = 1.5, n - 1) on the crate's window, the exact "
            "solution at 60 digits (hi + lo) and its scale, times 2^-e; wotan 1.10's method='huber' trend "
            "there (unscaled), whether its window holds the same points, the exact solution on its window "
            "(times 2^-e) and how many points ours holds.\n"
        )
        fh.write("curve,index,e,root_hi,root_lo,scale,wotan,same_window,wotan_root_hi,wotan_root_lo,points\n")
        for r in rows:
            fh.write(r + "\n")
    print(len(rows), "windows")


if __name__ == "__main__":
    main()
