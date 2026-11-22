"""Write reference exposure-averaged light curves from batman and PyTransit.

For seven kinds of transit and every common Kepler and TESS cadence, writes
crates/rustellar-model/tests/reference/exposure_points.csv with, at each time:

    batman     batman 2.5.1 with supersample_factor = samples and exp_time
               (its samples run from edge to edge of the exposure)
    pytransit  PyTransit 2.6.18 QuadraticModel with nsamples = samples and
               exptimes (samples in the middle of equal slices: the midpoint
               rule of Kipping 2010)
    dense      batman's instantaneous flux averaged over 2001 midpoint
               samples: the exposure average with negligible sampling error
               and no orbit approximation, for checking convergence

Run from the repository root:

    python crates/rustellar-benchmarks/references/make_exposure_reference.py

The output has no date in it, so running again with the same versions gives
identical files.
"""

import math
import platform
import sys
from pathlib import Path

import batman
import numpy as np
import pytransit
from pytransit import QuadraticModel

OUT = (
    Path(__file__).resolve().parents[3]
    / "crates" / "rustellar-model" / "tests" / "reference" / "exposure_points.csv"
)

T0 = 3000.123456
U = (0.4, 0.25)
# (k, b, a_rs, period): a hot Jupiter, a grazing one, a small planet on a
# longer orbit, a large planet on a tight orbit, an eclipsing binary, a
# long-period planet, and a short-period sub-Neptune.
SETS = [
    (0.1, 0.3, 10.0, 3.0),
    (0.1, 0.95, 10.0, 3.0),
    (0.02, 0.5, 20.0, 10.0),
    (0.2, 0.0, 4.0, 1.0),
    (0.5, 0.5, 8.0, 2.0),
    (0.05, 0.2, 100.0, 100.0),
    (0.03, 0.7, 6.0, 1.5),
]
# Exposures in seconds: TESS 20 s, Kepler short cadence, TESS 2 min, TESS FFI
# from 2022 (200 s), 2020 (10 min) and 2018 (30 min), and Kepler long cadence.
CADENCES = [20.0, 58.85, 120.0, 200.0, 600.0, 1800.0, 1765.5]
SAMPLES = [1, 2, 3, 5, 7, 15, 30]
DENSE = 2001
N_TIMES = 61


def half_t14(k, b, a_rs, period):
    ratio = min(((1 + k) ** 2 - b * b) / (a_rs * a_rs - b * b), 1.0)
    return period / math.pi * math.asin(math.sqrt(ratio)) / 2.0


def times_for(k, b, a_rs, period, exposure):
    half = half_t14(k, b, a_rs, period)
    window = half + exposure
    offsets = list(np.linspace(-window, window, N_TIMES)) + [-half, half]
    return np.array(sorted(T0 + dt for dt in offsets))


def batman_binned(k, b, a_rs, period, t, exposure, samples):
    params = batman.TransitParams()
    params.t0 = T0
    params.per = period
    params.rp = k
    params.a = a_rs
    params.inc = math.degrees(math.acos(b / a_rs))
    params.ecc = 0.0
    params.w = 90.0
    params.u = list(U)
    params.limb_dark = "quadratic"
    model = batman.TransitModel(params, t, supersample_factor=samples, exp_time=exposure)
    return model.light_curve(params)


def batman_dense(k, b, a_rs, period, t, exposure):
    offsets = exposure * ((np.arange(DENSE) + 0.5) / DENSE - 0.5)
    flat = (t[:, None] + offsets[None, :]).ravel()
    return batman_binned(k, b, a_rs, period, flat, exposure, 1).reshape(t.size, DENSE).mean(axis=1)


def pytransit_binned(k, b, a_rs, period, t, exposure, samples):
    model = QuadraticModel(interpolate=False)
    model.set_data(t, nsamples=[samples], exptimes=[exposure])
    return model.evaluate(k, np.array(U), T0, period, a_rs, math.acos(b / a_rs))


def main():
    rows = []
    for index, (k, b, a_rs, period) in enumerate(SETS):
        for cadence in CADENCES:
            exposure = cadence / 86400.0
            t = times_for(k, b, a_rs, period, exposure)
            dense = batman_dense(k, b, a_rs, period, t, exposure)
            for samples in SAMPLES:
                bat = batman_binned(k, b, a_rs, period, t, exposure, samples)
                pyt = pytransit_binned(k, b, a_rs, period, t, exposure, samples)
                for row in zip(t, bat, pyt, dense):
                    rows.append((index, k, b, a_rs, period, exposure, samples) + tuple(float(x) for x in row))

    header = [
        f"# {len(rows)} points: exposure-averaged flux, quadratic limb darkening u = {U}, circular orbits, t0 = {T0}.",
        f"# Sets (k, b, a_rs, period): {SETS}. Cadences (s): {CADENCES}. Samples: {SAMPLES}.",
        f"# batman: batman {batman.__version__}, supersample_factor = samples, exp_time = exposure.",
        f"# pytransit: PyTransit {pytransit.__version__} QuadraticModel(interpolate=False), nsamples = samples.",
        f"# dense: batman's unbinned flux averaged over {DENSE} midpoint samples.",
        f"# numpy {np.__version__}, Python {platform.python_version()}. Times and exposures in days (BTJD).",
        "set,k,b,a_rs,period,exposure,samples,t,batman,pytransit,dense",
    ]
    with OUT.open("w", newline="\n", encoding="ascii") as f:
        f.write("\n".join(header) + "\n")
        for row in rows:
            fields = [str(row[0])] + [repr(x) for x in row[1:6]] + [str(row[6])] + [repr(x) for x in row[7:]]
            f.write(",".join(fields) + "\n")
    print(f"wrote {len(rows)} points to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
