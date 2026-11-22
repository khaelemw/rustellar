"""Exact (mpmath, 40 digits) Lomb-Scargle peaks and best-fit sinusoids,
with astropy 7.2's model parameters, for rustellar-ls's tests (outside the
repo). Uses the cases and the exact reduction of make_ls_reference.py.

For each case and mean model, the three highest peaks of the grid's exact
power (from ls_powers.csv; local maxima, the grid's ends included when
above their neighbour), less any where the fit is singular. Each
is refined to where the power's derivative is zero within one grid step,
on the side the derivative at the grid peak points to; with no change of
sign there it stays on the grid. At the refined frequency: the exact
standard power, the parabola through the grid's three powers (the usual
interpolation), and the sinusoid fitted by weighted least squares at 40
digits,

    y = offset + a cos(2 pi f (t - tc)) + b sin(2 pi f (t - tc)),

tc = t[0] + 0.5 (t[-1] - t[0]) in float64 (as rustellar-ls), with the
standard errors from (X^T W X)^-1, W = 1 / dy^2, and the amplitude and
phase (y = offset + A cos(2 pi f (t - tc) - phase)) with theirs by first
order propagation. The subtracted mean's offset is the weighted mean,
with its standard error; the zero mean's is zero.

astropy: model_parameters at the refined frequency, its offset() and the
constant term (offset + theta0 with a floating mean), and the amplitude.

Writes crates/rustellar-ls/tests/reference/ls_models.csv.

Run: python crates/rustellar-benchmarks/references/make_ls_models.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from astropy.timeseries import LombScargle  # noqa: E402

from make_ls_reference import MEANS, OUT, cases, exact  # noqa: E402

mp.mp.dps = 40


def power(t, y, dy, f, mean):
    red, ref, _ = exact(t, y, dy, f, mean)
    return red / ref


def derivative(t, y, dy, f, mean):
    return mp.diff(lambda x: power(t, y, dy, x, mean), mp.mpf(f))


def peaks(p):
    n = len(p)
    out = []
    for k in range(n):
        left = k == 0 or p[k] > p[k - 1]
        right = k == n - 1 or p[k] >= p[k + 1]
        if left and right and np.isfinite(p[k]):
            out.append(k)
    out.sort(key=lambda k: -p[k])
    return out[:3]


def refine(t, y, dy, freq, k, mean):
    fk = mp.mpf(float(freq[k]))
    g = derivative(t, y, dy, fk, mean)
    other = k + 1 if g > 0 else k - 1
    if g == 0 or other < 0 or other >= len(freq):
        return fk
    fo = mp.mpf(float(freq[other]))
    go = derivative(t, y, dy, fo, mean)
    if (g > 0) == (go > 0):
        return fk
    a, b = (fk, fo) if fk < fo else (fo, fk)
    return mp.findroot(lambda x: derivative(t, y, dy, x, mean), (a, b), solver="anderson")


def model(t, y, dy, f, mean, tc):
    w = [1 / mp.mpf(e) ** 2 for e in dy]
    ys = [mp.mpf(v) for v in y]
    two_pi_f = 2 * mp.pi * f
    cols = []
    mu = mp.mpf(0)
    if mean != "zero":
        mu = mp.fsum(wi * yi for wi, yi in zip(w, ys)) / mp.fsum(w)
    if mean == "floating":
        cols.append([mp.mpf(1)] * len(t))
    else:
        ys = [v - mu for v in ys]
    cols.append([mp.cos(two_pi_f * (mp.mpf(ti) - mp.mpf(tc))) for ti in t])
    cols.append([mp.sin(two_pi_f * (mp.mpf(ti) - mp.mpf(tc))) for ti in t])
    m = len(cols)
    a = mp.matrix(m, m)
    rhs = mp.matrix(m, 1)
    for i in range(m):
        rhs[i] = mp.fsum(wi * ci * yi for wi, ci, yi in zip(w, cols[i], ys))
        for j in range(m):
            a[i, j] = mp.fsum(wi * ci * cj for wi, ci, cj in zip(w, cols[i], cols[j]))
    cov = a ** -1
    x = cov * rhs
    if mean == "floating":
        offset, var_offset = x[0], cov[0, 0]
        i = 1
    else:
        offset = mu
        var_offset = 1 / mp.fsum(w) if mean == "subtracted" else mp.mpf(0)
        i = 0
    ca, cb = x[i], x[i + 1]
    vaa, vbb, vab = cov[i, i], cov[i + 1, i + 1], cov[i, i + 1]
    amp = mp.hypot(ca, cb)
    phase = mp.atan2(cb, ca)
    var_amp = (ca**2 * vaa + cb**2 * vbb + 2 * ca * cb * vab) / amp**2
    var_phase = (cb**2 * vaa + ca**2 * vbb - 2 * ca * cb * vab) / amp**4
    return offset, ca, cb, amp, phase, mp.sqrt(var_offset), mp.sqrt(var_amp), mp.sqrt(var_phase)


def grid_powers():
    out = {}
    with open(os.path.join(OUT, "ls_powers.csv")) as fp:
        next(fp)
        for line in fp:
            r = line.strip().split(",")
            out.setdefault((r[0], r[1]), []).append(float(mp.mpf(r[5]) / mp.mpf(r[7])))
    return out


def main():
    exact_powers = grid_powers()
    path = os.path.join(OUT, "ls_models.csv")
    with open(path, "w", newline="\n") as fp:
        fp.write(
            "case,mean,f0,df,nf,k,f_peak,power,parabola,offset,cosine,sine,amplitude,phase,"
            "offset_err,amplitude_err,phase_err,astropy_constant,astropy_amplitude\n"
        )
        for name, t, y, dy, (f0, df, nf) in cases():
            freq = f0 + df * np.arange(nf)
            tc = float(t[0] + 0.5 * (t[-1] - t[0]))
            for mean, flags in MEANS.items():
                ls = LombScargle(t, y, dy, **flags)
                p = np.array(exact_powers[(name, mean)])
                for k in peaks(p):
                    f = refine(t, y, dy, freq, k, mean)
                    pw = power(t, y, dy, f, mean)
                    if 0 < k < nf - 1:
                        y0, y1, y2 = p[k - 1], p[k], p[k + 1]
                        shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
                        para = float(freq[k] + shift * df)
                    else:
                        para = float(freq[k])
                    try:
                        off, ca, cb, amp, ph, eo, ea, ep = model(t, y, dy, f, mean, tc)
                    except ZeroDivisionError:
                        continue
                    theta = ls.model_parameters(float(f))
                    const = ls.offset() + (theta[0] if mean == "floating" else 0.0)
                    astro_amp = float(np.hypot(theta[-2], theta[-1]))
                    vals = [off, ca, cb, amp, ph, eo, ea, ep]
                    fp.write(
                        f"{name},{mean},{float(f0)!r},{float(df)!r},{nf},{k},{mp.nstr(f, 25)},"
                        f"{mp.nstr(pw, 25)},{para!r},"
                        + ",".join(mp.nstr(v, 25) for v in vals)
                        + f",{float(const)!r},{astro_amp!r}\n"
                    )
            print(name, flush=True)


if __name__ == "__main__":
    main()
