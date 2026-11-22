"""TLS 1.32 on injected transits in harder regimes, for rustellar-benchmarks.
Writes crates/rustellar-benchmarks/data/tls_<kind>.csv for each kind below.

Each curve's parameters from SplitMix64(seed + 1_000_000), in this order:
k, period, t0 (uniform in [0, period)), noise (uniform in [3e-4, 8e-4)),
b, then e and w (eccentric only), then amp, prot and phase (variable
only). Every range below is uniform. The star has the Sun's density,
a = 4.206 P^(2/3) (P in days), and quadratic limb darkening
(0.4804, 0.1867), TLS's default; b is the impact parameter at
mid-transit, so cos i = (b / a)(1 + e sin w) / (1 - e^2). Transit from
batman.

- long: k in [0.015, 0.045), P in [9, 13.5) (two or three transits),
  b in [0, 0.7); 27 d at 30 min.
- ultrashort: k as long, P in [0.25, 1), b as long; 10 d at 10 min,
  errors noise sqrt(3).
- deep: k in [0.1, 0.2), P in [1, 9), b as long; 27 d at 30 min.
- grazing: k in [0.05, 0.12), P in [1, 9), b = 1 - k + 1.5 k u, so the
  planet's disc crosses the limb; 27 d at 30 min.
- eccentric: k as long, P in [2, 9), b as long, e in [0.1, 0.7), w in
  [0, 360) degrees; 27 d at 30 min.
- fast: k as long, P in [0.7, 3), b as long; 6 d at 20 s, errors
  noise sqrt(90).
- gaps: k as long, P in [1, 20), b as long; three 27-day blocks at
  30 min starting at days 0, 54 and 81 (one block missing between the
  first two), each less a one-day gap from its day 13.
- flagged: k as long, P in [1, 9), b as long; 27 d at 30 min. After
  each point's noise one more uniform u: below 0.02 the point is flagged
  and dimmed by 0.01 (as a bad-quality point is); from 0.02 to 0.03 an
  outlier, the error times (5 + 15 s) added or taken away (two more
  uniforms: s, then the sign, taken away below 0.5); from 0.03 to 0.04
  the flux is NaN. Flagged points are dropped before the search; the
  NaNs are left for TLS to clean.
- variable: k in [0.02, 0.05), P in [1, 9), b as long, amp in
  [1e-3, 5e-3), prot in [3, 10) d, phase in [0, 2 pi); 27 d at 30 min.
  The flux is the transit times 1 + amp (sin(2 pi t / prot + phase)
  + 0.3 sin(4 pi t / prot + 2 phase)), plus white noise, plus red noise
  r_i = 0.8 r_(i-1) + 0.6 noise n_i (r_0 = noise n_0) from a second set
  of four uniforms per point. Detrended by wotan's biweight before the
  search, window three times the transit's duration and break tolerance
  half the window; TLS gets the detrended flux and errors.

Noise from SplitMix64(seed): per point the sum of four uniforms less 2,
times sqrt(3), times the point's error. TLS gets the errors. Each row
also keeps the curve's finite points before detrending and their flux
summed, so the Rust side can check it builds the same curve.

Run: python crates/rustellar-benchmarks/references/make_tls_regimes_reference.py [kind ...] [--count N] [--processes N]
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

warnings.filterwarnings("ignore")

FIRST_SEED = {"long": 6000, "ultrashort": 7000, "deep": 8000, "grazing": 9000,
              "eccentric": 10000, "fast": 11000, "gaps": 12000, "flagged": 13000,
              "variable": 14000}
COUNT = {"fast": 40, "gaps": 40}
PERIODS = {"long": (9.0, 13.5), "ultrashort": (0.25, 1.0), "eccentric": (2.0, 9.0),
           "fast": (0.7, 3.0), "gaps": (1.0, 20.0)}
RADII = {"deep": (0.1, 0.2), "grazing": (0.05, 0.12), "variable": (0.02, 0.05)}


def between(rng, lo, hi):
    return lo + (hi - lo) * rng.uniform()


def params(kind, seed):
    rng = SplitMix64(seed + 1_000_000)
    p = {"k": between(rng, *RADII.get(kind, (0.015, 0.045)))}
    p["period"] = between(rng, *PERIODS.get(kind, (1.0, 9.0)))
    p["t0"] = p["period"] * rng.uniform()
    p["noise"] = between(rng, 3e-4, 8e-4)
    u = rng.uniform()
    p["b"] = 1.0 - p["k"] + 1.5 * p["k"] * u if kind == "grazing" else 0.7 * u
    p["e"], p["w"] = 0.0, 90.0
    if kind == "eccentric":
        p["e"] = between(rng, 0.1, 0.7)
        p["w"] = 360.0 * rng.uniform()
    p["amp"], p["prot"], p["phase"] = 0.0, 0.0, 0.0
    if kind == "variable":
        p["amp"] = between(rng, 1e-3, 5e-3)
        p["prot"] = between(rng, 3.0, 10.0)
        p["phase"] = 2.0 * math.pi * rng.uniform()
    p["a"] = 4.206 * p["period"] ** (2.0 / 3.0)
    return p


def times_and_errors(kind, noise):
    if kind == "ultrashort":
        t = np.arange(1440) / 144.0
        return t, np.full(len(t), noise * math.sqrt(3.0))
    if kind == "fast":
        t = np.arange(25920) / 4320.0
        return t, np.full(len(t), noise * math.sqrt(90.0))
    if kind == "gaps":
        block = np.arange(1296) / 48.0
        block = block[~((block >= 13.0) & (block < 14.0))]
        t = np.concatenate([start + block for start in (0.0, 54.0, 81.0)])
        return t, np.full(len(t), noise)
    t = np.arange(1296) / 48.0
    return t, np.full(len(t), noise)


def batman_params(p):
    q = batman.TransitParams()
    q.t0, q.per, q.rp, q.a = p["t0"], p["period"], p["k"], p["a"]
    cos_i = p["b"] / p["a"] * (1 + p["e"] * math.sin(math.radians(p["w"]))) / (1 - p["e"] ** 2)
    q.inc = math.degrees(math.acos(cos_i))
    q.ecc, q.w, q.u, q.limb_dark = p["e"], p["w"], [0.4804, 0.1867], "quadratic"
    return q


def t14(p):
    """First to fourth contact of a circular orbit, in days."""
    inc = math.acos(p["b"] / p["a"])
    chord = math.sqrt(max((1 + p["k"]) ** 2 - p["b"] ** 2, 0.0))
    return p["period"] / math.pi * math.asin(min(chord / (p["a"] * math.sin(inc)), 1.0))


def unit(rng):
    n = 0.0
    for _ in range(4):
        n += rng.uniform() - 0.5
    return n * math.sqrt(3.0)


def light_curve(kind, seed):
    """The curve TLS searches: times, flux, errors, and the parameters."""
    p = params(kind, seed)
    t, dy = times_and_errors(kind, p["noise"])
    q = batman_params(p)
    model = batman.TransitModel(q, t).light_curve(q)
    rng = SplitMix64(seed)
    keep, y = [], np.empty(len(t))
    red = 0.0
    for i, f in enumerate(model):
        y[i] = f + dy[i] * unit(rng)
        kept = True
        if kind == "flagged":
            u = rng.uniform()
            if u < 0.02:
                y[i] -= 0.01
                kept = False
            elif u < 0.03:
                s = rng.uniform()
                sign = -1.0 if rng.uniform() < 0.5 else 1.0
                y[i] += sign * dy[i] * (5.0 + 15.0 * s)
            elif u < 0.04:
                y[i] = math.nan
        elif kind == "variable":
            ti = t[i]
            star = 1.0 + p["amp"] * (math.sin(2 * math.pi * ti / p["prot"] + p["phase"])
                                     + 0.3 * math.sin(4 * math.pi * ti / p["prot"] + 2 * p["phase"]))
            n = unit(rng)
            red = p["noise"] * n if i == 0 else 0.8 * red + 0.6 * p["noise"] * n
            y[i] = f * star + (y[i] - f) + red
        keep.append(kept)
    keep = np.array(keep)
    t, y, dy = t[keep], y[keep], dy[keep]
    # A check that the Rust side builds the same curve: its finite points
    # before any detrending, and their flux summed.
    p["points"] = int(np.isfinite(y).sum())
    p["flux_sum"] = float(np.nansum(y))
    p["window"] = 3.0 * t14(p) if kind == "variable" else 0.0
    if kind == "variable":
        from wotan import flatten
        w = p["window"]
        flat, trend = flatten(t, y, method="biweight", window_length=w,
                              break_tolerance=w / 2, edge_cutoff=0.0, return_trend=True)
        good = np.isfinite(flat)
        t, y, dy = t[good], flat[good], (dy / trend)[good]
    return p, t, y, dy


def one(job):
    kind, seed = job
    warnings.filterwarnings("ignore")
    from transitleastsquares import transitleastsquares
    p, t, y, dy = light_curve(kind, seed)
    r = transitleastsquares(t, y, dy, verbose=False).power(
        use_threads=1, show_progress_bar=False, verbose=False)
    return [seed] + [repr(float(p[c])) for c in COLUMNS] + [
        repr(float(r.period)), repr(float(r.T0)), repr(1 - float(r.depth)),
        repr(float(r.duration)), repr(float(r.SDE))]


COLUMNS = ["k", "period", "t0", "noise", "b", "a", "e", "w", "amp", "prot", "phase", "window",
           "points", "flux_sum"]


if __name__ == "__main__":
    args = sys.argv[1:]
    count, processes = None, 4
    if "--count" in args:
        i = args.index("--count")
        count = int(args[i + 1])
        del args[i:i + 2]
    if "--processes" in args:
        i = args.index("--processes")
        processes = int(args[i + 1])
        del args[i:i + 2]
    kinds = args or list(FIRST_SEED)
    here = os.path.dirname(os.path.abspath(__file__))
    for kind in kinds:
        n = count or COUNT.get(kind, 100)
        first = FIRST_SEED[kind]
        with Pool(processes) as pool:
            rows = pool.map(one, [(kind, s) for s in range(first, first + n)])
        out = os.path.join(here, "..", "data", f"tls_{kind}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed"] + COLUMNS + ["tls_period", "tls_t0", "tls_depth",
                                              "tls_duration", "tls_sde"])
            w.writerows(rows)
        print(out, len(rows), flush=True)
