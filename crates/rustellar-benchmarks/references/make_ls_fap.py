"""Exact (mpmath, 40 digits) false-alarm probabilities and levels of the
Lomb-Scargle periodogram's highest peak (Baluev 2008, Table 1 and
equations 5 and 6), with astropy 7.2's, for rustellar-ls's tests (outside
the repo). Uses the cases of make_ls_reference.py, plus "large": 100,000
even times 0.01 days apart with unit errors.

Degrees of freedom: (dH, dK) = (1, 3) with a floating or subtracted mean,
(0, 2) without (astropy refuses data that are not centred, so its columns
are nan there). The maximum frequency is the case's grid's highest.

For each case, mean (floating, zero), normalisation, method and
probability p: the exact level z at which the method's probability is p;
z rounded to float64; the exact probability at that float z; astropy's
probability at it and its level for p (nan where it fails).

Writes crates/rustellar-ls/tests/reference/ls_fap.csv.

Run: python crates/rustellar-benchmarks/references/make_ls_fap.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from astropy.timeseries.periodograms.lombscargle import _statistics as st  # noqa: E402

from make_ls_reference import OUT, cases  # noqa: E402

mp.mp.dps = 40

NORMS = ["standard", "model", "log", "psd"]
METHODS = ["baluev", "davies", "naive", "single"]
PS = ["0.5", "1e-2", "1e-6", "1e-15", "1e-40"]


def all_cases():
    out = [(name, t, y, dy, f0 + df * (nf - 1)) for name, t, y, dy, (f0, df, nf) in cases()]
    t = 0.01 * np.arange(100_000)
    out.append(("large", t, np.zeros_like(t), np.ones_like(t), 5.0))
    return out


class Null:
    def __init__(self, t, dy, fmax, mean, norm, method):
        n = len(t)
        dh, dk = (1, 3) if mean == "floating" else (0, 2)
        self.nh, self.nk = mp.mpf(n - dh), mp.mpf(n - dk)
        w = [1 / mp.mpf(e) ** 2 for e in dy]
        ts = [mp.mpf(v) for v in t]
        sw = mp.fsum(w)
        mt = mp.fsum(a * b for a, b in zip(w, ts)) / sw
        var = mp.fsum(a * (b - mt) ** 2 for a, b in zip(w, ts)) / sw
        fmax = mp.mpf(fmax)
        self.w = fmax * mp.sqrt(4 * mp.pi * var)
        self.indep = fmax * (ts[-1] - ts[0])
        self.norm, self.method = norm, method

    def gamma(self, m):
        return mp.sqrt(2 / m) * mp.gamma(m / 2) / mp.gamma((m - 1) / 2)

    def single(self, z):
        nk = self.nk
        if self.norm == "standard":
            return (1 - z) ** (nk / 2)
        if self.norm == "model":
            return (1 + z) ** (-nk / 2)
        if self.norm == "log":
            return mp.exp(-nk * z / 2)
        return mp.exp(-z)

    def tau(self, z):
        nh, nk, w = self.nh, self.nk, self.w
        if self.norm == "standard":
            return self.gamma(nh) * w * (1 - z) ** ((nk - 1) / 2) * mp.sqrt(nh * z / 2)
        if self.norm == "model":
            return self.gamma(nk) * w * (1 + z) ** (-nk / 2) * mp.sqrt(nk * z / 2)
        if self.norm == "log":
            return self.gamma(nk) * w * mp.exp(-z * (nk - mp.mpf(0.5)) / 2) * mp.sqrt(nk * mp.sinh(z / 2))
        return w * mp.exp(-z) * mp.sqrt(z)

    def fap(self, z):
        z = mp.mpf(z)
        if z <= 0:
            return mp.mpf(1)
        if self.norm == "standard" and z >= 1:
            return mp.mpf(0)
        s = self.single(z)
        if self.method == "single":
            return s
        # In expm1 and log1p: 1 - (1 - s) e^-tau cancels at 40 digits too
        # once s and tau are near 1e-40.
        if self.method == "naive":
            return -mp.expm1(self.indep * mp.log1p(-s))
        tau = self.tau(z)
        if self.method == "davies":
            return min(s + tau, mp.mpf(1))
        return -mp.expm1(-tau) + s * mp.exp(-tau)

    def single_level(self, p):
        k = 2 / self.nk
        if self.norm == "standard":
            return -mp.expm1(k * mp.log(p))
        if self.norm == "model":
            return mp.expm1(-k * mp.log(p))
        if self.norm == "log":
            return -k * mp.log(p)
        return -mp.log(p)

    def level(self, p):
        p = mp.mpf(p)
        if self.method == "single":
            return self.single_level(p)
        if self.method == "naive":
            return self.single_level(-mp.expm1(mp.log1p(-p) / self.indep))
        lo = self.single_level(p)
        hi = mp.mpf(1) if self.norm == "standard" else 2 * lo + 1
        while self.fap(hi) >= p:
            hi *= 2
        # Bisection to the working precision, then a polish.
        for _ in range(200):
            mid = (lo + hi) / 2
            if self.fap(mid) >= p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def astropy(fn, *args):
    try:
        with np.errstate(all="ignore"):
            v = float(np.asarray(fn(*args)).ravel()[0])
        return repr(v)
    except Exception:  # noqa: BLE001
        return "nan"


def main():
    path = os.path.join(OUT, "ls_fap.csv")
    with open(path, "w", newline="\n") as fp:
        fp.write("case,mean,normalization,method,fmax,p,z_exact,z,fap_exact,astropy_fap,astropy_level\n")
        for name, t, y, dy, fmax in all_cases():
            trel = t - t[0]
            for mean in ["floating", "zero"]:
                for norm in NORMS:
                    for method in METHODS:
                        null = Null(t, dy, fmax, mean, norm, method)
                        for p in PS:
                            z_exact = null.level(p)
                            z = float(z_exact)
                            fap = null.fap(z)
                            if mean == "floating":
                                af = astropy(st.false_alarm_probability, z, fmax, trel, y, dy, norm, method)
                                al = astropy(st.false_alarm_level, float(p), fmax, trel, y, dy, norm, method)
                            else:
                                af = al = "nan"
                            fp.write(
                                f"{name},{mean},{norm},{method},{float(fmax)!r},{p},{mp.nstr(z_exact, 25)},"
                                f"{z!r},{mp.nstr(fap, 25)},{af},{al}\n"
                            )
            print(name, flush=True)


if __name__ == "__main__":
    main()
