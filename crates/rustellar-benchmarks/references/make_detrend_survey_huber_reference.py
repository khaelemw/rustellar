"""Huber's M-estimator (fixed scale c MAD, c = 1.28) at the survey's sampled
points, for rustellar-detrend's survey test: the root of the crate's window
at 50 digits (as `make_detrend_estimators_reference.m_reference`), and
wotan 1.10's `huber_psi` trend (one step, c 1.28) with the root of wotan's
own window.

Writes crates/rustellar-detrend/tests/reference/survey_huber.csv. Run from a
file (multiprocessing):

    python crates/rustellar-benchmarks/references/make_detrend_survey_huber_reference.py
"""

import importlib.util
import math
import os
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
from mpmath import mpf
from wotan import flatten

C = 1.28
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "crates", "rustellar-detrend", "tests", "reference")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


survey = load("make_detrend_survey_reference")
estimators = load("make_detrend_estimators_reference")


def sample(job):
    curve, i, values, wotan, theirs = job
    root, s, sensitivity = estimators.m_reference("huber", values, C)
    e = math.frexp(max(abs(v) for v in values))[1] - 1
    factor = mpf(2) ** -e
    hi, lo = estimators.hi_lo(root * factor)
    if theirs is None:
        w_hi, w_lo = hi, lo
    else:
        w_hi, w_lo = estimators.hi_lo(estimators.m_reference("huber", theirs, C)[0] * factor)
    return (
        f"{curve},{i},{e},{hi!r},{lo!r},{float(s * factor)!r},{float(sensitivity * factor)!r},"
        f"{wotan!r},{int(theirs is None)},{w_hi!r},{w_lo!r},{len(values)}"
    )


def main():
    jobs = []
    for row in survey.CURVES:
        p = dict(zip(survey.FIELDS, row))
        time, flux, _ = survey.light_curve(p)
        window = p["window"]
        trend = flatten(
            np.array(time), np.array(flux), method="huber_psi", window_length=window,
            break_tolerance=window / 2, edge_cutoff=0, cval=C, return_trend=True,
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
    with open(os.path.join(OUT, "survey_huber.csv"), "w", newline="\n") as fh:
        fh.write(
            "# Sampled points: Huber's M-estimator (c = 1.28, scale c MAD) of the crate's window at 50 digits "
            "(hi + lo), c MAD and the root's sensitivity, all times 2^-e; wotan 1.10's huber_psi trend there "
            "(one step, unscaled), whether its window holds the same points, the root of its window (times 2^-e) "
            "and how many points ours holds.\n"
        )
        fh.write("curve,index,e,root_hi,root_lo,c_mad,sensitivity,wotan,same_window,wotan_root_hi,wotan_root_lo,points\n")
        for r in rows:
            fh.write(r + "\n")
    print(len(rows), "windows")


if __name__ == "__main__":
    main()
