"""TLS 1.32 on injected transits with uneven sampling or varying errors,
and at 2-minute cadence, for rustellar-benchmarks. Writes
crates/rustellar-benchmarks/data/tls_<kind>.csv for kind "mixed", "varying", "short" or
"sectors".

Each curve's transit parameters are make_tls_many_reference.params(seed)
(k, period, t0, noise; `sparse` unused). Transit from batman, a = 12,
b = 0.2, quadratic (0.4804, 0.1867).

- mixed: 13.5 d at 30 min (i / 48, i < 648), then from day 14 at 10 min
  (14 + i / 144, i < 1872). Errors `noise` at 30 min and `noise sqrt(3)`
  at 10 min (the same precision per unit time).
- varying: 27 d at 30 min (i / 48, i < 1296), errors
  `noise (2 - cos(2 pi t / 13.5))`, from 1 to 3 times `noise`.
- short: 27 d at 2 min (i / 720, i < 19440), errors `noise sqrt(15)` (the
  same precision per unit time as 30 min).
- sectors: two sectors at 2 min, 54 d (i / 720, i < 38880) less the
  mid-sector gaps [13, 14) and [40, 41) days, errors `noise sqrt(15)`.

Noise from SplitMix64(seed): per point the sum of four uniforms less 2,
times sqrt(3), times the point's error. TLS gets the errors.

Run: python crates/rustellar-benchmarks/references/make_tls_uneven_reference.py kind [count] [processes]
"""
import csv
import math
import os
import sys
import warnings
from multiprocessing import Pool

import batman
import numpy as np

from make_tls_injection_reference import SplitMix64
from make_tls_many_reference import params

warnings.filterwarnings("ignore")

FIRST_SEED = {"mixed": 2000, "varying": 3000, "short": 4000, "sectors": 5000}


def times_and_errors(kind, noise):
    if kind == "mixed":
        t = np.concatenate([np.arange(648) / 48.0, 14.0 + np.arange(1872) / 144.0])
        dy = np.where(t < 14.0, noise, noise * math.sqrt(3.0))
    elif kind == "varying":
        t = np.arange(1296) / 48.0
        dy = noise * (2.0 - np.cos(2.0 * math.pi * t / 13.5))
    else:
        days = 27.0 if kind == "short" else 54.0
        t = np.arange(int(days * 720)) / 720.0
        if kind == "sectors":
            t = t[~(((t >= 13.0) & (t < 14.0)) | ((t >= 40.0) & (t < 41.0)))]
        dy = np.full(len(t), noise * math.sqrt(15.0))
    return t, dy


def light_curve(kind, seed):
    k, period, t0, noise, _ = params(seed)
    t, dy = times_and_errors(kind, noise)
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = t0, period, k, 12.0
    p.inc = float(np.degrees(np.arccos(0.2 / 12.0)))
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.4804, 0.1867], "quadratic"
    model = batman.TransitModel(p, t).light_curve(p)
    rng = SplitMix64(seed)
    y = np.empty(len(t))
    for i, f in enumerate(model):
        n = 0.0
        for _ in range(4):
            n += rng.uniform() - 0.5
        y[i] = f + dy[i] * (n * math.sqrt(3.0))
    return (k, period, t0, noise), t, y, dy


def one(job):
    kind, seed = job
    warnings.filterwarnings("ignore")
    from transitleastsquares import transitleastsquares
    (k, period, t0, noise), t, y, dy = light_curve(kind, seed)
    r = transitleastsquares(t, y, dy, verbose=False).power(
        use_threads=1, show_progress_bar=False, verbose=False)
    return [seed, repr(k), repr(period), repr(t0), repr(noise),
            repr(float(r.period)), repr(float(r.T0)), repr(1 - float(r.depth)),
            repr(float(r.duration)), repr(float(r.SDE))]


if __name__ == "__main__":
    kind = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    processes = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "data", f"tls_{kind}.csv")
    first = FIRST_SEED[kind]
    with Pool(processes) as pool:
        rows = pool.map(one, [(kind, s) for s in range(first, first + count)])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "k", "period", "t0", "noise", "tls_period", "tls_t0",
                    "tls_depth", "tls_duration", "tls_sde"])
        w.writerows(rows)
    print(out, len(rows))
