"""Transit fits by batman 2.5.1 and PyTransit 2.6.18 with scipy 1.17.1's
least_squares, on deterministic light curves, for rustellar-fit's
reference test.

Each case's light curve: the truth's flux by batman (instantaneous, or the
mean over each exposure by the midpoint rule of Kipping 2010, batman
evaluated at the samples and averaged here), plus Gaussian noise from
SplitMix64 (Steele, Lea & Flood 2014) and Box-Muller, seeded per case.

Each method fits every parameter (t0, P, k, a/R*, b, u1, u2, baseline)
but the case's fixed ones (limb darkening, held at the truth, for the small
planet and long cadence, where the data leave it all but free) from the
case's start, residuals (baseline * model - flux) / err, x_scale='jac'
(More's scaling, as rustellar-fit):

- batman, scipy 'lm' (MINPACK), 2-point Jacobian (its only choice);
- batman, scipy 'trf', 2-point and 3-point;
- batman, scipy 'trf', 3-point, tolerances 1e-15 (the best it can do);
- PyTransit (QuadraticModel, exact, not interpolated), 'trf', 3-point.

Long cadence: batman with its own supersample_factor and exp_time (as a
user would), PyTransit with nsamples and exptimes.

Records per method: the parameters, chi-squared by its own model, model
evaluations (every call, the Jacobian's included), seconds (one run;
rustellar-benchmarks times them properly), scipy's status, and standard
errors from its final Jacobian, (J^T J)^-1.

Writes crates/rustellar-fit/tests/reference/fit_cases.csv,
fit_points.csv and fit_methods.csv.

Run: python crates/rustellar-benchmarks/references/make_fit_reference.py
"""

import math
import os
import time

import batman
import numpy as np
from pytransit import QuadraticModel
from scipy.optimize import least_squares

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "crates", "rustellar-fit", "tests", "reference",
)

NAMES = ["t0", "P", "k", "a_rs", "b", "u1", "u2", "baseline"]


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def uniform(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        return ((z >> 11) + 1) / float(1 << 53)

    def normal(self):
        u, v = self.uniform(), self.uniform()
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


# name, truth, start, (first time, step, points), sigma, noise seed (None:
# noiseless), exposure (days, samples) or None, fixed parameters
CASES = [
    ("hot_jupiter_noiseless",
     [1.3, 3.0, 0.1, 10.0, 0.3, 0.4, 0.25, 1.0],
     [1.302, 3.0005, 0.105, 9.0, 0.4, 0.3, 0.2, 1.0],
     (0.0, 5.0 / 1440.0, 1440), 1e-3, None, None, []),
    ("hot_jupiter",
     [1.3, 3.0, 0.1, 10.0, 0.3, 0.4, 0.25, 1.0],
     [1.302, 3.0005, 0.105, 9.0, 0.4, 0.3, 0.2, 1.0],
     (0.0, 5.0 / 1440.0, 1440), 1e-3, 1, None, []),
    ("bjd_times",
     [2459001.3, 3.0, 0.1, 10.0, 0.3, 0.4, 0.25, 1.0],
     [2459001.31, 3.001, 0.09, 12.0, 0.1, 0.5, 0.1, 1.0],
     (2459000.0, 5.0 / 1440.0, 1440), 1e-3, 2, None, []),
    ("small_planet",
     [5.0, 11.0, 0.02, 25.0, 0.5, 0.45, 0.2, 1.0],
     [5.02, 11.01, 0.025, 20.0, 0.2, 0.45, 0.2, 1.0],
     (0.0, 10.0 / 1440.0, 3888), 2e-4, 3, None, ["u1", "u2"]),
    ("grazing",
     [1.0, 2.0, 0.12, 6.0, 1.02, 0.4, 0.25, 1.0],
     [1.0, 2.0, 0.15, 6.5, 0.95, 0.4, 0.25, 1.0],
     (0.0, 2.0 / 1440.0, 2880), 1e-4, 4, None, []),
    ("long_cadence",
     [3.0, 4.5123, 0.08, 12.0, 0.4, 0.4, 0.25, 1.0],
     [3.01, 4.514, 0.09, 10.0, 0.2, 0.4, 0.25, 1.0],
     (0.0, 30.0 / 1440.0, 1440), 1e-3, 5, (30.0 / 1440.0, 15), ["u1", "u2"]),
]


def batman_params(x):
    t0, per, k, a, b, u1, u2, _ = x
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = t0, per, k, a
    p.inc = math.degrees(math.acos(max(-1.0, min(1.0, b / a))))
    p.ecc, p.w = 0.0, 90.0
    p.limb_dark, p.u = "quadratic", [u1, u2]
    return p


def times(first, step, n):
    return np.array([first + step * i for i in range(n)])


def truth_flux(x, t, exposure):
    """The truth's flux by batman, averaged by the midpoint rule."""
    if exposure is None:
        m = batman.TransitModel(batman_params(x), t)
        return x[7] * m.light_curve(batman_params(x))
    duration, samples = exposure
    offsets = [duration * ((2 * i + 1 - samples) / (2 * samples)) for i in range(samples)]
    total = np.zeros_like(t)
    for dt in offsets:
        m = batman.TransitModel(batman_params(x), t + dt)
        total += m.light_curve(batman_params(x))
    return x[7] * total / samples


class Batman:
    def __init__(self, t, exposure):
        p = batman_params([1.0, 3.0, 0.1, 10.0, 0.3, 0.4, 0.25, 1.0])
        if exposure is None:
            self.m = batman.TransitModel(p, t)
        else:
            self.m = batman.TransitModel(
                p, t, supersample_factor=exposure[1], exp_time=exposure[0])

    def __call__(self, x):
        return x[7] * self.m.light_curve(batman_params(x))


class PyTransit:
    def __init__(self, t, exposure):
        self.m = QuadraticModel()
        if exposure is None:
            self.m.set_data(t)
        else:
            self.m.set_data(t, exptimes=[exposure[0]], nsamples=[exposure[1]])

    def __call__(self, x):
        t0, per, k, a, b, u1, u2, base = x
        inc = math.acos(max(-1.0, min(1.0, b / a)))
        flux = self.m.evaluate(k, [u1, u2], t0, per, a, inc)
        return base * np.asarray(flux, dtype=float).ravel()


METHODS = [
    ("batman lm 2-point", Batman, dict(method="lm", jac="2-point")),
    ("batman trf 2-point", Batman, dict(method="trf", jac="2-point")),
    ("batman trf 3-point", Batman, dict(method="trf", jac="3-point")),
    ("batman trf 3-point tight", Batman,
     dict(method="trf", jac="3-point", ftol=1e-15, xtol=1e-15, gtol=1e-15,
          max_nfev=100000)),
    ("PyTransit trf 3-point", PyTransit, dict(method="trf", jac="3-point")),
]


def fit(model, flux, sigma, start, fixed, options):
    free = [i for i, n in enumerate(NAMES) if n not in fixed]
    calls = [0]

    def full(y):
        x = np.array(start, dtype=float)
        x[free] = y
        return x

    def residuals(y):
        calls[0] += 1
        return (model(full(y)) - flux) / sigma

    began = time.perf_counter()
    res = least_squares(residuals, np.array(start, dtype=float)[free], x_scale="jac",
                        **options)
    seconds = time.perf_counter() - began
    chi2 = float(np.sum(residuals(res.x) ** 2))
    jac = res.jac
    errors = np.zeros(len(NAMES))
    try:
        errors[free] = np.sqrt(np.abs(np.diag(np.linalg.inv(jac.T @ jac))))
    except np.linalg.LinAlgError:
        errors[free] = np.inf
    return full(res.x), chi2, calls[0], seconds, res.status, errors


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fit_cases.csv"), "w", newline="\n") as cases, \
            open(os.path.join(OUT, "fit_points.csv"), "w", newline="\n") as points, \
            open(os.path.join(OUT, "fit_methods.csv"), "w", newline="\n") as methods:
        cases.write("case,sigma,exposure_days,samples,fixed,"
                    + ",".join("truth_" + n for n in NAMES) + ","
                    + ",".join("start_" + n for n in NAMES) + "\n")
        points.write("case,time,flux\n")
        methods.write("case,method," + ",".join(NAMES)
                      + ",chi2,evaluations,seconds,status,"
                      + ",".join("err_" + n for n in NAMES) + "\n")
        for name, truth, start, grid, sigma, seed, exposure, fixed in CASES:
            t = times(*grid)
            flux = truth_flux(truth, t, exposure)
            if seed is not None:
                noise = SplitMix64(seed)
                flux = flux + sigma * np.array([noise.normal() for _ in t])
            exp_days, samples = exposure if exposure else (0.0, 1)
            cases.write(f"{name},{sigma!r},{exp_days!r},{samples},{';'.join(fixed)},"
                        + ",".join(repr(v) for v in truth) + ","
                        + ",".join(repr(v) for v in start) + "\n")
            for ti, fi in zip(t, flux):
                points.write(f"{name},{float(ti)!r},{float(fi)!r}\n")
            for label, kind, options in METHODS:
                model = kind(t, exposure)
                x, chi2, calls, seconds, status, errors = fit(
                    model, flux, sigma, start, fixed, options)
                methods.write(f"{name},{label},"
                              + ",".join(repr(float(v)) for v in x)
                              + f",{chi2!r},{calls},{seconds!r},{status},"
                              + ",".join(repr(float(v)) for v in errors) + "\n")
                print(f"{name:22} {label:26} chi2 {chi2:.10g} calls {calls:6} "
                      f"{seconds:.3f} s status {status}")


if __name__ == "__main__":
    main()
