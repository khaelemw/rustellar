"""The other location estimators of many windows, exactly or at 50 digits,
with wotan's, for rustellar-detrend's tests.

- Welsch (textbook: exp(-u^2/2), no cut), Andrews' sine wave (sin u for
  |u| < pi) and Huber (u clipped at 1), u = (x - T) / (c MAD): wotan's
  iteration T <- T + sum w d / sum w from the exact median and MAD at 50
  digits, then Newton, to 1e-45 relative: the root. With c MAD and the
  root's sensitivity to rounding, as for the biweight.
- Hodges-Lehmann (median of (x_i + x_j) / 2, i <= j), trimmed and
  winsorized means (floor(p n) at each end): exact, in Fractions, then
  correctly rounded (Python's float(Fraction)).
- wotan 1.10 on the same values: location_iter (welsch with its cut,
  andrewsinewave), huber_psi (one step), hodges, trim_mean, winsorize_mean;
  NaN where it fails or is not comparable.

Windows from the biweight generator's light curves (every fifth window),
survey-like windows, and hand-built ones.

Writes crates/rustellar-detrend/tests/reference/estimator_windows.csv.

Run: python crates/rustellar-benchmarks/references/make_detrend_estimators_reference.py
"""

import math
import os
import sys
from fractions import Fraction

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_detrend_biweight_reference import (  # noqa: E402
    exact_median,
    hi_lo,
    light_curve,
    survey,
    windows_of,
)
from wotan.location_estimates import (  # noqa: E402
    hodges,
    huber_psi,
    location_iter,
    trim_mean,
    winsorize_mean,
)

mp.dps = 50
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-detrend", "tests", "reference")


def weight(method, u):
    """(w, psi') at u."""
    if method == "welsch":
        w = mp.exp(-u * u / 2)
        return w, w * (1 - u * u)
    if method == "andrews":
        if abs(u) >= mp.pi:
            return mpf(0), mpf(0)
        if u == 0:
            return mpf(1), mpf(1)
        return mp.sin(u) / u, mp.cos(u)
    if method == "huber":
        if abs(u) < 1:
            return mpf(1), mpf(1)
        return 1 / abs(u), mpf(0)
    raise ValueError(method)


def m_reference(method, values, c):
    xs = [mpf(float(v)) for v in values]
    m = exact_median(xs)
    mad = exact_median([abs(x - m) for x in xs])
    if mad == 0:
        return m, mpf(0), mpf(0)
    s = c * mad
    t = m
    tolerance = lambda: mpf(10) ** -45 * (abs(t) + s)  # noqa: E731
    for _ in range(20_000):
        num = den = mpf(0)
        for x in xs:
            w, _ = weight(method, (x - t) / s)
            num += w * (x - t)
            den += w
        if den == 0:
            break
        step = num / den
        t += step
        if abs(step) <= mpf(10) ** -12 * s:
            break
    # Newton to the root.
    for _ in range(200):
        r = slope = mpf(0)
        for x in xs:
            w, dpsi = weight(method, (x - t) / s)
            r += w * (x - t)
            slope += dpsi
        if slope <= 0:
            break
        step = r / slope
        t += step
        if abs(step) <= tolerance():
            break
    else:
        raise RuntimeError("no convergence")
    terms = slope = mpf(0)
    for x in xs:
        u = (x - t) / s
        w, dpsi = weight(method, u)
        terms += abs(x - t) * w + abs(dpsi * u) * s
        slope += dpsi
    sensitivity = terms / slope if slope > 0 else mpf("inf")
    return t, s, sensitivity


def cut(p, n):
    # floor of the rounded product, as scipy and wotan: 0.3 of 10 is 3.
    return math.floor(p * n)


def exact_reference(method, values, p):
    xs = sorted(Fraction(float(v)) for v in values)
    n = len(xs)
    if method == "trimmed":
        g = cut(p, n)
        kept = xs[g : n - g]
        return float(sum(kept) / len(kept))
    if method == "winsorized":
        g = cut(p, n)
        ys = [xs[g]] * g + xs[g : n - g] + [xs[n - 1 - g]] * g
        return float(sum(ys) / n)
    if method == "hodges":
        averages = sorted((xs[i] + xs[j]) / 2 for i in range(n) for j in range(i, n))
        return float(exact_median(averages))
    raise ValueError(method)


def wotan(method, values, c):
    a = np.array(values, dtype=np.float64)
    try:
        with np.errstate(all="ignore"):
            if method == "welsch":
                return float(location_iter(a, c, 3))
            if method == "andrews":
                return float(location_iter(a, c, 2))
            if method == "huber":
                return float(huber_psi(a, c))
            if method == "hodges":
                return float(hodges(a)) if len(a) <= 800 else float("nan")
            if method == "trimmed":
                return float(trim_mean(a, c)) if c > 0 else float("nan")
            if method == "winsorized":
                return float(winsorize_mean(a, c)) if c > 0 else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")
    raise ValueError(method)


SETTINGS = [
    ("welsch", 2.11),
    ("welsch", 1.0),
    ("andrews", 1.339),
    ("andrews", 3.0),
    ("huber", 1.28),
    ("huber", 2.5),
    ("hodges", 0.0),
    ("trimmed", 0.1),
    ("trimmed", 0.25),
    ("winsorized", 0.1),
    ("winsorized", 0.3),
]
M_METHODS = ("welsch", "andrews", "huber")


def row(case, kind, method, c, values):
    if method in M_METHODS:
        root, s, sensitivity = m_reference(method, values, c)
        e = math.frexp(max(abs(v) for v in values))[1] - 1 if any(values) else 0
        factor = mpf(2) ** -e
        root, s, sensitivity = root * factor, s * factor, sensitivity * factor
        hi, lo = hi_lo(root)
    else:
        e, s, sensitivity = 0, mpf(0), mpf(0)
        hi, lo = exact_reference(method, values, c), 0.0
    w = float("nan")
    if kind not in ("huge", "subnormal", "subnormal top"):
        w = wotan(method, values, c)
    return (
        f"{case},{kind},{method},{c!r},{e},{hi!r},{lo!r},{float(s)!r},{float(sensitivity)!r},{w!r},"
        + ";".join(repr(float(v)) for v in values)
    )


def hand_built():
    return [
        ("mad0", [1.0, 1.0, 1.0, 2.0, 3.0]),
        ("one", [0.7]),
        ("two", [0.1, 0.2]),
        ("two", [1.0, 1.0 + 2.0**-52]),
        ("three", [0.0, 1.0, 10.0]),
        ("edge", [-5.0, -1.0, 0.0, 1.0, 5.0]),
        ("edge", [-3.0, -1.0, 0.0, 1.0, 3.0 - 1e-12]),
        ("ties", [2.0] * 5 + [3.0] * 5),
        ("clusters", [0.0] * 6 + [1.0] * 5 + [0.4]),
        ("skew", [0.0, 1, 2, 3, 4, 5, 6, 7, 50, 100]),
    ]


def main():
    rng = np.random.default_rng(20261117)
    rows = []
    case = 0
    kinds = ["normalised", "counts", "signs", "transit", "outliers", "subnormal", "subnormal top", "huge", "tiny"]
    for method, c in SETTINGS:
        for kind in kinds:
            n = int(rng.integers(40, 110))
            t, f, mask = light_curve(rng, kind, n)
            for i, inside in enumerate(windows_of(t, mask, 0.5)):
                if i % 5 == 0 and inside:
                    rows.append(row(case, kind, method, c, [float(f[j]) for j in inside]))
                    case += 1
        kind, f = survey(rng, 360, "survey 4 min")
        rows.append(row(case, kind, method, c, list(f)))
        case += 1
        for kind, values in hand_built():
            rows.append(row(case, kind, method, c, values))
            case += 1
    with open(os.path.join(OUT, "estimator_windows.csv"), "w", newline="\n") as fh:
        fh.write(
            "# Welsch, Andrews and Huber: the root at 50 digits (hi + lo), c MAD and the root's "
            "sensitivity to rounding, all times 2^-e (e = floor log2 of the largest magnitude). "
            "Hodges-Lehmann, trimmed and winsorized means: exact, correctly rounded (hi; e = 0). "
            "c is the tuning constant or the proportion. wotan 1.10's value (unscaled; NaN where it "
            "fails); the window's fluxes.\n"
        )
        fh.write("window,kind,method,c,e,root_hi,root_lo,c_mad,sensitivity,wotan,values\n")
        for r in rows:
            fh.write(r + "\n")
    print(len(rows), "windows")


if __name__ == "__main__":
    main()
