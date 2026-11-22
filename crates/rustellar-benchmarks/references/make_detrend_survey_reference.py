"""Survey-scale references for rustellar-detrend.

Light curves like a TESS sector (two orbits at 2-minute cadence, with the
gap between them) and a Kepler quarter, with stellar variability (a
Catmull-Rom spline through random knots a quarter of a rotation apart, and a
slow drift), noise (a sum of four uniforms), flares (a rational decay) and box
transits. The recipe uses only + - * / and floor, so the crate's tests build
the same light curves to the bit from the parameters written here.

For each curve:
- wotan 1.10's biweight trend (`flatten`, the same window, break tolerance
  half a window), at every point as wotan / model - 1 to six digits (for the
  detrend-then-search comparison), and in full at sampled points;
- at sampled points, the biweight root of the crate's window (symmetric,
  exact time differences, segments split at gaps over half a window) at 50
  digits, as make_detrend_biweight_reference.py, and whether wotan's window
  (half open, without its segment's last point, its edges compared in
  floating point) holds the same points, and if not, the root of wotan's
  window at 50 digits too: wotan's value is compared with its own window's
  root.

Writes crates/rustellar-detrend/tests/reference/survey_{curves,windows,wotan}.csv.

Run: python crates/rustellar-benchmarks/references/make_detrend_survey_reference.py
"""

import bisect
import math
import os
import struct
import sys
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
from mpmath import mpf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_detrend_biweight_reference import hi_lo, reference  # noqa: E402
from wotan import flatten  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-detrend", "tests", "reference")
C = 5.0
MASK64 = (1 << 64) - 1


class Random:
    """SplitMix64."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        x = self.state
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
        return x ^ (x >> 31)

    def unit(self):
        # A multiple of 2^-53 in [0, 1), exactly.
        return float(self.next_u64() >> 11) / float(1 << 53)


FIELDS = [
    "name", "seed", "n", "t_start", "cadence", "keep", "gap_start", "gap", "scale",
    "rotation", "amplitude", "drift", "noise", "flare_rate", "flare", "tau",
    "period", "tc", "duration", "depth", "window", "every",
]

TESS = 1.0 / 720.0
CURVES = [
    # A TESS sector: rotation 3.1 d, 1% spots, a 1 ppt transit.
    ("tess", 1, 19440, 1325.29, TESS, 0.97, 13.1, 1.0, 1.0,
     3.1, 0.01, 0.004, 1.5e-3, 1e-3, 0.01, 0.01, 4.7, 1.9, 0.12, 1e-3, 0.5, 61),
    # A fast rotator under a shorter window.
    ("fast rotator", 2, 19440, 1352.71, TESS, 0.97, 13.3, 0.9, 1.0,
     1.3, 0.01, -0.003, 1.5e-3, 1e-3, 0.01, 0.01, 3.3, 0.8, 0.09, 3e-3, 0.3, 61),
    # In counts.
    ("counts", 3, 19440, 1379.0, TESS, 0.97, 12.9, 1.1, 2.3e4,
     2.3, 0.005, 0.002, 1e-3, 1e-3, 0.01, 0.01, 6.1, 2.4, 0.15, 1e-3, 0.5, 61),
    # Faint (flux in units where wotan's absolute tolerance is coarse).
    ("faint", 4, 19440, 1406.13, TESS, 0.97, 13.0, 1.0, 1e-5,
     4.2, 0.008, 0.001, 1.5e-3, 1e-3, 0.01, 0.01, 5.3, 1.1, 0.13, 2e-3, 0.5, 61),
    # A Kepler quarter at long cadence, a 1 d window.
    ("kepler", 5, 4360, 131.51, 0.0204335, 0.99, 44.0, 1.0, 1.0,
     11.0, 0.004, 0.001, 2e-4, 5e-4, 0.005, 0.02, 9.3, 3.2, 0.2, 5e-4, 1.0, 15),
]


def in_box(x, period, tc, duration):
    k = math.floor((x - tc) / period + 0.5)
    r = (x - tc) - float(k) * period
    return abs(r) < duration / 2.0


def light_curve(p):
    """(time, flux, model) of the kept cadences, written as the crate's
    tests write it, operation for operation."""
    knots_rng = Random(p["seed"] ^ 0x5EED5EED5EED5EED)
    span = p["cadence"] * float(p["n"])
    h = p["rotation"] / 4.0
    knots = [p["amplitude"] * (2.0 * knots_rng.unit() - 1.0) for _ in range(int(math.floor(span / h)) + 4)]
    rng = Random(p["seed"])
    flares = []
    time, flux, model = [], [], []
    for i in range(p["n"]):
        u_keep, n1, n2, n3, n4, u_flare, u_amp = (rng.unit() for _ in range(7))
        x = p["cadence"] * float(i)
        if u_flare < p["flare_rate"]:
            flares.append((x, p["flare"] * (0.2 + u_amp)))
        if u_keep >= p["keep"] or (p["gap_start"] <= x and x < p["gap_start"] + p["gap"]):
            continue
        s = x / h + 1.0
        k = math.floor(s)
        f = s - float(k)
        p0, p1, p2, p3 = knots[k - 1], knots[k], knots[k + 1], knots[k + 2]
        spline = 0.5 * (
            2.0 * p1
            + f * ((p2 - p0) + f * ((2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) + f * (3.0 * (p1 - p2) + p3 - p0)))
        )
        m = p["scale"] * (1.0 + spline + p["drift"] * (x / span))
        flare = 0.0
        for x0, a in flares:
            if x - x0 < 30.0 * p["tau"]:
                q = 1.0 + (x - x0) / p["tau"]
                flare += a / (q * q)
        noise = p["noise"] * (((n1 + n2) + (n3 + n4)) - 2.0)
        dip = p["depth"] if in_box(x, p["period"], p["tc"], p["duration"]) else 0.0
        time.append(p["t_start"] + x)
        flux.append(m * (((1.0 + noise) - dip) + flare))
        model.append(m)
    return time, flux, model


def fingerprint(time, flux):
    """FNV-1a over the bits of every time and flux."""
    h = 0xCBF29CE484222325
    for t, f in zip(time, flux):
        for v in (t, f):
            h = ((h ^ struct.unpack("<Q", struct.pack("<d", v))[0]) * 0x100000001B3) & MASK64
    return h


def segments(time, tolerance):
    """Start index of each segment (and the end): split where the gap is
    longer than `tolerance`, exactly."""
    starts = [0]
    for k in range(1, len(time)):
        if Fraction(time[k]) - Fraction(time[k - 1]) > Fraction(tolerance):
            starts.append(k)
    return starts + [len(time)]


def ours(time, lo, hi, i, half):
    """The crate's window: indices in [lo, hi) within half a window of i,
    exactly."""
    a = bisect.bisect_left(time, time[i] - 2 * half, lo, hi)
    b = bisect.bisect_right(time, time[i] + 2 * half, lo, hi)
    ti = Fraction(time[i])
    return [j for j in range(a, b) if abs(Fraction(time[j]) - ti) <= half]


def wotans(time, lo, hi, i, half):
    """wotan's window, in floating point as `running_segment`: from the
    first point not before t - w/2 to before the first not before t + w/2,
    the end index stopping at the segment's last point."""
    start = lo
    while time[start] < time[i] - float(half):
        start += 1
    end = lo
    while time[end] < time[i] + float(half) and end < hi - 1:
        end += 1
    return list(range(start, end))


def sample(job):
    curve, i, values, wotan, theirs = job
    root, s, sensitivity = reference(values, C)
    e = math.frexp(max(abs(v) for v in values))[1] - 1
    factor = mpf(2) ** -e
    hi, lo = hi_lo(root * factor)
    # The root of wotan's window, where it differs (same scale).
    w_hi, w_lo = (hi, lo) if theirs is None else hi_lo(reference(theirs, C)[0] * factor)
    return (
        f"{curve},{i},{e},{hi!r},{lo!r},{float(s * factor)!r},{float(sensitivity * factor)!r},"
        f"{wotan!r},{int(theirs is None)},{w_hi!r},{w_lo!r},{len(values)}"
    )


def main():
    curve_rows, wotan_rows, jobs = [], [], []
    for row in CURVES:
        p = dict(zip(FIELDS, row))
        time, flux, model = light_curve(p)
        window = p["window"]
        trend = flatten(
            np.array(time), np.array(flux), method="biweight", window_length=window,
            break_tolerance=window / 2, edge_cutoff=0, cval=C, return_trend=True,
        )[1]
        wotan_rows.append(p["name"] + "," + ";".join("%.6e" % (trend[k] / model[k] - 1) for k in range(len(time))))
        half = Fraction(window) / 2
        starts = segments(time, window / 2)
        for lo, hi in zip(starts, starts[1:]):
            for i in range(lo, hi):
                if i % p["every"]:
                    continue
                mine = ours(time, lo, hi, i, half)
                theirs = wotans(time, lo, hi, i, half)
                other = None if mine == theirs else [flux[j] for j in theirs]
                jobs.append((p["name"], i, [flux[j] for j in mine], float(trend[i]), other))
        # A fingerprint of the curve, for the tests to check theirs against.
        curve_rows.append(
            ",".join(p[k] if k == "name" else repr(p[k]) for k in FIELDS)
            + f",{len(time)},{len(starts) - 1},{fingerprint(time, flux)}"
        )
        print(p["name"], len(time), "points,", len(starts) - 1, "segments", flush=True)
    with Pool() as pool:
        windows = pool.map(sample, jobs, chunksize=4)
    with open(os.path.join(OUT, "survey_curves.csv"), "w", newline="\n") as fh:
        fh.write(
            "# The survey light curves' recipes (see the tests), each with its number of points, "
            "segments and a hash of its times and fluxes (FNV-1a over their bits).\n"
        )
        fh.write(",".join(FIELDS) + ",points,segments,fingerprint\n")
        for r in curve_rows:
            fh.write(r + "\n")
    with open(os.path.join(OUT, "survey_windows.csv"), "w", newline="\n") as fh:
        fh.write(
            "# Sampled points: the biweight root (c = 5) of the crate's window at 50 digits (hi + lo), "
            "c MAD and the root's sensitivity, all times 2^-e; wotan 1.10's trend there (unscaled), "
            "whether its window holds the same points, the root of its window (times 2^-e) and how many points ours holds.\n"
        )
        fh.write("curve,index,e,root_hi,root_lo,c_mad,sensitivity,wotan,same_window,wotan_root_hi,wotan_root_lo,points\n")
        for r in windows:
            fh.write(r + "\n")
    with open(os.path.join(OUT, "survey_wotan.csv"), "w", newline="\n") as fh:
        fh.write("# wotan 1.10's biweight trend over each survey curve, as trend / model - 1.\n")
        for r in wotan_rows:
            fh.write(r + "\n")
    print(len(windows), "windows,", sum(1 for j in jobs if j[4] is not None), "where wotan's differs")


if __name__ == "__main__":
    main()
