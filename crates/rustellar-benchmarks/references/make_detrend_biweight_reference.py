"""The biweight location of many windows at 50 digits, with wotan's, for
rustellar-detrend's biweight tests.

For each window of fluxes: wotan's iteration (T <- T + sum w d / sum w)
from the exact median and MAD in 50-digit mpmath, run until the step is
below 1e-45 relative: the root. With it, the exact c MAD and the
sensitivity of the root to rounding (for the test's bound), and wotan
1.10's `location_iter` on the same values.

Windows:
- every third window of random light curves (normalised flux, counts,
  either sign around zero, a transit, outliers, subnormal and 1e±250
  scales), made with exact Fraction times as the crate's symmetric windows;
- survey-like windows: a day of TESS 2-minute cadence (720 points) with
  rotation, noise and a transit; 20-second cadence (4,320);
- hand-built: MAD = 0, one and two points, points at |u| exactly 1 and just
  either side, a window split in two clusters.

Writes crates/rustellar-detrend/tests/reference/biweight_windows.csv.

Run: python crates/rustellar-benchmarks/references/make_detrend_biweight_reference.py
"""

import math
import os
from fractions import Fraction

import numpy as np
from mpmath import mp, mpf

from wotan.location_estimates import location_iter

mp.dps = 50
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-detrend", "tests", "reference")


def exact_median(xs):
    s = sorted(xs)
    k = len(s)
    return s[k // 2] if k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2


def reference(values, c):
    """(root, c MAD, sensitivity), all mpf; the root is the median where
    MAD = 0."""
    xs = [mpf(float(v)) for v in values]
    m = exact_median(xs)
    mad = exact_median([abs(x - m) for x in xs])
    if mad == 0:
        return m, mpf(0), mpf(0)
    s = c * mad
    t = m
    for _ in range(100_000):
        num = den = mpf(0)
        for x in xs:
            u = (x - t) / s
            if abs(u) < 1:
                w = (1 - u * u) ** 2
                num += (x - t) * w
                den += w
        if den == 0:
            break
        step = num / den
        t += step
        if abs(step) <= mpf(10) ** -45 * (abs(t) + s):
            break
    else:
        raise RuntimeError("no convergence")
    # How far the root moves for errors of a few roundings in each term
    # (sum over |u| < 1 of |d|) and in c MAD (sum |psi'(u) u| s), over the
    # residual's slope sum psi'(u).
    terms = slope = mpf(0)
    for x in xs:
        u = (x - t) / s
        if abs(u) < 1:
            terms += abs(x - t) + abs((1 - u * u) * (1 - 5 * u * u) * u) * s
            slope += (1 - u * u) * (1 - 5 * u * u)
    sensitivity = terms / slope if slope > 0 else mpf("inf")
    return t, s, sensitivity


def hi_lo(x):
    hi = float(x)
    return hi, float(x - mpf(hi))


def windows_of(t, mask, window):
    """The crate's symmetric windows with exact times, no gaps: the
    unmasked indices within half a window of each point."""
    T = [Fraction(float(x)) for x in t]
    half = Fraction(window) / 2
    out = []
    for i in range(len(T)):
        out.append([j for j in range(len(T)) if not mask[j] and abs(T[j] - T[i]) <= half])
    return out


def light_curve(rng, kind, n):
    t = np.sort(rng.uniform(0, 3, n))
    base = 1 + 1e-3 * rng.standard_normal(n) + 2e-3 * np.sin(2 * np.pi * t / 1.7)
    if kind == "normalised":
        f = base
    elif kind == "counts":
        f = 1e5 * base
    elif kind == "signs":
        f = 1e-3 * rng.standard_normal(n) + 2e-4 * np.sin(t)
    elif kind == "transit":
        f = base - 4e-3 * (np.abs(t - 1.5) < 0.1)
    elif kind == "outliers":
        f = base.copy()
        hit = rng.random(n) < 0.05
        f[hit] += 0.02 * rng.standard_normal(hit.sum())
    elif kind == "subnormal":
        f = base * 2.0**-1060
    elif kind == "subnormal top":
        # Just below the normals: the subnormal grid one bit coarser than a
        # double's, where rounding twice goes wrong.
        f = base * 2.0**-1023
    elif kind == "huge":
        f = base * 1e250
    elif kind == "tiny":
        f = (base - 1) * 1e-250
    else:
        raise ValueError(kind)
    return t, f, rng.random(n) < 0.1


def survey(rng, k, cadence_name):
    t = np.arange(k) / k  # a day
    f = (
        1
        + 3e-3 * np.sin(2 * np.pi * t / 2.3 + rng.uniform(0, 6))
        + 1e-3 * rng.standard_normal(k)
        - 1e-3 * (np.abs(t - rng.uniform(0.2, 0.8)) < 0.05)
    )
    hit = rng.random(k) < 0.01
    f[hit] += 0.01 * rng.standard_normal(hit.sum())
    return cadence_name, f


def hand_built():
    c = 5.0
    out = []
    out.append(("mad0", [1.0, 1.0, 1.0, 2.0, 3.0]))
    out.append(("mad0", [0.3, 0.3, 0.3, 0.3]))
    out.append(("one", [0.7]))
    out.append(("two", [0.1, 0.2]))
    out.append(("two", [1.0, 1.0 + 2.0**-52]))
    out.append(("two", [-3.0, 1e-17]))
    # MAD 1 and c = 5: points at u = -1 and +1 exactly (weight 0), and just
    # inside or outside.
    out.append(("edge", [-5.0, -1.0, 0.0, 1.0, 5.0]))
    for d in [1e-9, 1e-15, 2.0**-50]:
        out.append(("edge", [-5.0, -1.0, 0.0, 1.0, 5.0 - d]))
        out.append(("edge", [-5.0, -1.0, 0.0, 1.0, 5.0 + d]))
        out.append(("edge", [-5.0 + d, -1.0, 0.0, 1.0, 5.0]))
    out.append(("clusters", [0.0] * 6 + [1.0] * 5 + [0.4]))
    out.append(("clusters", [1.0, 1.001, 1.002, 1.003, 1.0045, 1.006, 1.2, 1.21]))
    return [(kind, values, c) for kind, values in out]


def row(case, kind, values, c):
    root, s, sensitivity = reference(values, c)
    # Everything scaled by 2^-e (the largest magnitude between 1 and 2), as
    # the crate does, so that hi + lo holds subnormal roots exactly.
    e = math.frexp(max(abs(v) for v in values))[1] - 1 if any(values) else 0
    factor = mpf(2) ** -e
    root, s, sensitivity = root * factor, s * factor, sensitivity * factor
    hi, lo = hi_lo(root)
    # wotan's tolerance is absolute (1e-6): at 1e250 its step never gets
    # there; on subnormals 1 / (c MAD) overflows. NaN for those.
    wotan = float("nan")
    if kind not in ("huge", "subnormal", "subnormal top"):
        wotan = float(location_iter(np.array(values, dtype=np.float64), c, 1))
    return (
        f"{case},{kind},{c!r},{e},{hi!r},{lo!r},{float(s)!r},{float(sensitivity)!r},{wotan!r},"
        + ";".join(repr(float(v)) for v in values)
    )


def main():
    rng = np.random.default_rng(20261119)
    rows = []
    case = 0
    kinds = ["normalised", "counts", "signs", "transit", "outliers", "subnormal", "subnormal top", "huge", "tiny"]
    for kind in kinds:
        for window, c in [(0.3, 5.0), (0.8, 5.0), (0.5, 3.0)]:
            n = int(rng.integers(60, 140))
            t, f, mask = light_curve(rng, kind, n)
            for i, inside in enumerate(windows_of(t, mask, window)):
                if i % 3 == 0 and inside:
                    rows.append(row(case, kind, [float(f[j]) for j in inside], c))
                    case += 1
    for k, name in [(720, "survey 2 min"), (4320, "survey 20 s")]:
        for _ in range(4 if k == 720 else 2):
            kind, f = survey(rng, k, name)
            rows.append(row(case, kind, list(f), 5.0))
            case += 1
    for kind, values, c in hand_built():
        rows.append(row(case, kind, values, c))
        case += 1
    with open(os.path.join(OUT, "biweight_windows.csv"), "w", newline="\n") as fh:
        fh.write(
            "# The biweight root at 50 digits (hi + lo), c MAD and the root's sensitivity to "
            "rounding (see the tests), all times 2^-e (e = floor log2 of the largest "
            "magnitude); wotan 1.10's location_iter (unscaled); the window's fluxes.\n"
        )
        fh.write("window,kind,c,e,root_hi,root_lo,c_mad,sensitivity,wotan,values\n")
        for r in rows:
            fh.write(r + "\n")
    print(len(rows), "windows")


if __name__ == "__main__":
    main()
