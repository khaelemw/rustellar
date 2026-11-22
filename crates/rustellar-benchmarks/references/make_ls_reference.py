"""Exact (mpmath, 40 digits) Lomb-Scargle powers and astropy 7.2's slow,
cython and fast powers, for rustellar-ls's tests.

Cases (seeded): uneven sampling with a sinusoid in white noise and
varying errors; the same at BJD-sized times; a near-perfect sinusoid; low
frequencies only; even sampling with the Nyquist frequency on the grid
(a degenerate fit); a few points.

For each case, mean model and frequency, the exact reduction in
chi-squared and the chi-squared left, with raw weights 1 / dy^2 (Gram-Schmidt in the weighted inner
product at 40 digits, dropping a basis vector whose residual norm is below
1e-30 of the weights' sqrt(sum w), every term being at most 1), and chi2_ref. astropy's powers are its standard
normalisation. `independence` is how close the fit's terms come to
depending on each other (the smallest of each kept term's norm after
orthogonalising over before): near 1e-16 the fit is degenerate to f64
rounding.

Mean models: floating (fit_mean True), subtracted (fit_mean False,
center_data True), zero (both False).

Writes crates/rustellar-ls/tests/reference/ls_points.csv and
ls_powers.csv.

Run: python crates/rustellar-benchmarks/references/make_ls_reference.py
"""

import os

import mpmath as mp
import numpy as np
from astropy.timeseries import LombScargle

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-ls", "tests", "reference")
mp.mp.dps = 40

MEANS = {
    "floating": dict(fit_mean=True, center_data=True),
    "subtracted": dict(fit_mean=False, center_data=True),
    "zero": dict(fit_mean=False, center_data=False),
}


def cases():
    rng = np.random.default_rng(20261121)
    out = []
    # Uneven, a sinusoid in noise, errors varying by a factor of 3.
    n = 180
    t = np.sort(rng.uniform(0, 30, n))
    dy = rng.uniform(0.5e-3, 1.5e-3, n)
    y = 1 + 2e-3 * np.sin(2 * np.pi * t / 2.7 + 0.4) + dy * rng.standard_normal(n)
    out.append(("uneven", t, y, dy, (0.004, 0.006, 600)))
    # The same at BJD-sized times.
    out.append(("bjd", t + 2460123.25, y, dy, (0.004, 0.006, 600)))
    # A near-perfect sinusoid: the model and log normalisations' hard case.
    t2 = np.sort(rng.uniform(0, 10, 120))
    y2 = 5 + 0.3 * np.cos(2 * np.pi * 1.3 * t2) + 1e-9 * rng.standard_normal(120)
    out.append(("perfect", t2, y2, np.full(120, 1e-9), (1.25, 1e-3, 101)))
    # Low frequencies only, far below 1 / T: cancellation in the sums.
    out.append(("low", t, y, dy, (1e-6, 2e-5, 300)))
    # Even sampling: the Nyquist frequency (index 100) and its multiple
    # are on the grid, and there the sine is zero at every point.
    t3 = 0.5 * np.arange(64)
    y3 = 1 + 1e-2 * rng.standard_normal(64)
    out.append(("even", t3, y3, np.full(64, 1e-2), (0.005, 0.005, 400)))
    # A few points.
    t4 = np.array([0.0, 0.7, 1.9, 3.2, 4.0])
    y4 = np.array([1.0, 1.3, 0.8, 1.1, 0.95])
    out.append(("few", t4, y4, np.array([0.1, 0.2, 0.1, 0.15, 0.1]), (0.05, 0.05, 80)))
    return out


def exact(t, y, dy, f, mean):
    w = [1 / mp.mpf(e) ** 2 for e in dy]
    ys = [mp.mpf(v) for v in y]
    if mean != "zero":
        mu = mp.fsum(wi * yi for wi, yi in zip(w, ys)) / mp.fsum(w)
        ys = [v - mu for v in ys]
    two_pi_f = 2 * mp.pi * mp.mpf(f)
    basis = []
    if mean == "floating":
        basis.append([mp.mpf(1)] * len(t))
    basis.append([mp.cos(two_pi_f * mp.mpf(ti)) for ti in t])
    basis.append([mp.sin(two_pi_f * mp.mpf(ti)) for ti in t])

    def dot(a, b):
        return mp.fsum(wi * ai * bi for wi, ai, bi in zip(w, a, b))

    chi2_ref = dot(ys, ys)
    # Each term is at most 1 in size: a residual norm below 1e-30 of the
    # weights' is 40-digit rounding (sin(pi k) is not zero at 40 digits),
    # one above it a real term however small.
    scale = mp.sqrt(mp.fsum(w))
    # With a floating mean the reference fit is the mean itself, which the
    # centring already removed: the reduction is over the sinusoid only.
    kept = []
    # How close the kept terms come to depending on each other: each
    # one's norm after orthogonalising over before, the smallest.
    independence = mp.mpf(1)
    for v in basis:
        before = mp.sqrt(dot(v, v))
        for _ in range(2):
            for q in kept:
                c = dot(v, q)
                v = [a - c * b for a, b in zip(v, q)]
        norm = mp.sqrt(dot(v, v))
        if norm > mp.mpf("1e-30") * scale:
            kept.append([a / norm for a in v])
            independence = min(independence, norm / before)
    reduction = mp.fsum(dot(ys, q) ** 2 for q in kept)
    if mean == "floating":
        # The mean's own share (zero up to rounding, the data centred).
        reduction -= dot(ys, kept[0]) ** 2
    return reduction, chi2_ref, independence


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ls_points.csv"), "w", newline="\n") as fp:
        fp.write("case,t,y,dy\n")
        for name, t, y, dy, _ in cases():
            for a, b, c in zip(t, y, dy):
                fp.write(f"{name},{float(a)!r},{float(b)!r},{float(c)!r}\n")
    with open(os.path.join(OUT, "ls_powers.csv"), "w", newline="\n") as fp:
        fp.write("case,mean,f0,df,k,reduction,chi2,chi2_ref,independence,slow,cython,fast\n")
        for name, t, y, dy, (f0, df, nf) in cases():
            freq = f0 + df * np.arange(nf)
            for mean, flags in MEANS.items():
                ls = LombScargle(t, y, dy, **flags)
                slow = ls.power(freq, method="slow", assume_regular_frequency=True)
                cy = ls.power(freq, method="cython", assume_regular_frequency=True)
                fast = ls.power(freq, method="fast", assume_regular_frequency=True)
                for k, f in enumerate(freq):
                    red, ref, ind = exact(t, y, dy, f, mean)
                    fp.write(
                        f"{name},{mean},{float(f0)!r},{float(df)!r},{k},{mp.nstr(red, 25)},{mp.nstr(ref - red, 25)},"
                        f"{mp.nstr(ref, 25)},{mp.nstr(ind, 5)},{float(slow[k])!r},{float(cy[k])!r},{float(fast[k])!r}\n"
                    )
            print(name, flush=True)


if __name__ == "__main__":
    main()
