"""Reference posteriors for rustellar-mcmc's transit posterior in two harder
regimes, by zeus 2.5.4 with batman 2.5.1, beside the hot Jupiter of
make_transit_posterior_reference.py (the same sampling and light curve
recipe: 500 points 0.0006 d apart about mid-transit, white noise from
SplitMix64 and Box-Muller, the period fixed at 3 d, quadratic limb
darkening (0.4, 0.25) as q1 and q2).

- grazing: k = 0.1, a/R* = 10, b = 1.0, so the planet's disc crosses the
  star's limb; noise 5e-4, seed 2027. Free: t0, k, a/R*, b, q1, q2 and the
  baseline; k uniform on (0, 0.3), the rest as the hot Jupiter's. k and b
  are degenerate along a curved ridge.
- eccentric: k = 0.1, a/R* = 10, b = 0.3, e = 0.3, w = 60 degrees; noise
  1e-3, seed 2028. Free: those of the hot Jupiter and sqrt(e) cos w and
  sqrt(e) sin w (uniform in both, e below 1 and the planet outside the
  star at periastron), with a normal prior on the star's density,
  3 pi (a/R*)^3 / (G P^2), at its true value with a 10% spread. b is the
  impact parameter at mid-transit: cos i = (b / a)(1 + e sin w) / (1 - e^2).

zeus's ensemble slice sampler, 40 walkers, independent chains pooled
(grazing: four of 40,000 steps, the first 8,000 discarded; eccentric:
eight of 120,000, the first 20,000 discarded). Not emcee: on the eccentric
posterior four chains of emcee's differential-evolution moves (DEMove 0.8,
DESnookerMove 0.2, 300,000 steps each) agreed with each other yet all left
out a tail (e sin w below zero, and q2 above 0.4) that zeus and
rustellar-mcmc both find, from any start, on the same log-probability;
its stretch move mixes too slowly here.

Writes crates/rustellar-mcmc/tests/reference/<case>_points.csv (the light
curve), <case>_posterior.csv (the 2.5, 16, 50, 84 and 97.5 percentiles of
each coordinate and of the density) and <case>_spread.csv (for each, how
far the chains' own percentiles stray from the pooled ones at most, in
the pooled sigma: the reference's own uncertainty).

    python crates/rustellar-benchmarks/references/make_transit_regimes_posterior_reference.py [case ...]
"""
import math
import os
import sys
import time
from multiprocessing import Pool

import batman
import numpy as np

from make_transit_posterior_reference import G, SplitMix

OUT = "crates/rustellar-mcmc/tests/reference"
N, STEP, PERIOD = 500, 0.0006, 3.0
CASES = {
    "grazing": dict(k=0.1, a=10.0, b=1.0, e=0.0, w=90.0, sigma=5e-4, seed=2027),
    "eccentric": dict(k=0.1, a=10.0, b=0.3, e=0.3, w=60.0, sigma=1e-3, seed=2028),
}


def density(a):
    return 3 * math.pi * a**3 / (G * (PERIOD * 86400.0) ** 2) / 1000.0


def params(t0, k, a, b, u1, u2, e=0.0, w=90.0):
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = t0, PERIOD, k, a
    cos_i = b / a * (1 + e * math.sin(math.radians(w))) / (1 - e**2)
    p.inc = math.degrees(math.acos(cos_i))
    p.ecc, p.w = e, w
    p.limb_dark, p.u = "quadratic", [u1, u2]
    return p


def light_curve(case):
    c = CASES[case]
    t = np.array([-0.5 * (N - 1) * STEP + STEP * i for i in range(N)])
    truth = params(0.0, c["k"], c["a"], c["b"], 0.4, 0.25, c["e"], c["w"])
    g = SplitMix(c["seed"])
    flux = batman.TransitModel(truth, t).light_curve(truth) + c["sigma"] * np.array(
        [g.normal() for _ in range(N)])
    return t, flux


def names_and_centre(case):
    c = CASES[case]
    centre = [0.0, c["k"], c["a"], c["b"], 0.4225, 0.3077]
    names = ["t0", "k", "a_rs", "b", "q1", "q2"]
    if case == "eccentric":
        sqrt_e = math.sqrt(c["e"])
        centre += [sqrt_e * math.cos(math.radians(c["w"])), sqrt_e * math.sin(math.radians(c["w"]))]
        names += ["sqrt_e_cos_w", "sqrt_e_sin_w"]
    return names + ["baseline"], centre + [1.0]


class Posterior:
    def __init__(self, case):
        self.case = case
        self.c = CASES[case]
        self.t, self.flux = light_curve(case)
        self.model = batman.TransitModel(params(0.0, 0.1, 10.0, 0.3, 0.4, 0.25), self.t)
        self.rho = density(self.c["a"])

    def __call__(self, x):
        eccentric = self.case == "eccentric"
        t0, k, a, b, q1, q2 = x[:6]
        base = x[-1]
        e, w = 0.0, 90.0
        if eccentric:
            sc, ss = x[6], x[7]
            e = sc * sc + ss * ss
            w = math.degrees(math.atan2(ss, sc))
            if e >= 1 or a * (1 - e) <= 1:
                return -np.inf
        k_max = 0.3 if self.case == "grazing" else 1.0
        if not (0 < k < k_max and a > 1 and 0 <= b < 1 + k and b < a and 0 <= q1 <= 1
                and 0 <= q2 <= 1 and base > 0):
            return -np.inf
        cos_i = b / a * (1 + e * math.sin(math.radians(w))) / (1 - e**2)
        if cos_i >= 1:
            return -np.inf
        r = math.sqrt(q1)
        m = self.model.light_curve(params(t0, k, a, b, 2 * r * q2, r * (1 - 2 * q2), e, w))
        lp = -0.5 * np.sum(((base * m - self.flux) / self.c["sigma"]) ** 2)
        if eccentric:
            lp += -0.5 * ((density(a) - self.rho) / (0.1 * self.rho)) ** 2
        return lp


def chain(job):
    """One independent chain: run 0 from seeds 1 and 2, run r from 1 + r and
    2 + 10 r."""
    case, run = job
    names, centre = names_and_centre(case)
    dim = len(centre)
    rng = np.random.default_rng(1 + run)
    p0 = np.array(centre) + 1e-4 * rng.standard_normal((WALKERS, dim))
    if case == "grazing":
        # Start inside the support: b below 1 + k for every walker.
        p0[:, 3] = np.minimum(p0[:, 3], p0[:, 1] + 0.99)
    import random

    import zeus
    # zeus draws from both of Python's generators.
    random.seed(2 + 10 * run)
    np.random.seed(2 + 10 * run)
    steps, discard = STEPS[case]
    sampler = zeus.EnsembleSampler(WALKERS, dim, Posterior(case), verbose=False)
    start = time.perf_counter()
    sampler.run_mcmc(p0, steps, progress=False)
    secs = time.perf_counter() - start
    tau = zeus.AutoCorrTime(sampler.get_chain(discard=discard))
    print(case, run, "seconds", round(secs), "tau", np.round(tau), flush=True)
    return sampler.get_chain(discard=discard, flat=True)


WALKERS = 40
STEPS = {"grazing": (40_000, 8_000), "eccentric": (120_000, 20_000)}
RUNS = {"grazing": 4, "eccentric": 8}
QUANTILES = [2.5, 16, 50, 84, 97.5]


def with_density(flat):
    return [flat[:, i] for i in range(flat.shape[1])] + [np.array([density(a) for a in flat[:, 2]])]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cases = sys.argv[1:] or list(CASES)
    jobs = [(c, r) for c in cases for r in range(RUNS[c])]
    with Pool(len(jobs)) as pool:
        chains = pool.map(chain, jobs)
    for case in cases:
        t, flux = light_curve(case)
        sigma = CASES[case]["sigma"]
        with open(f"{OUT}/{case}_points.csv", "w", newline="\n") as fh:
            fh.write("time,flux,flux_err\n")
            for ti, fi in zip(t, flux):
                fh.write(f"{float(ti)!r},{float(fi)!r},{sigma!r}\n")
        runs = [ch for (c, _), ch in zip(jobs, chains) if c == case]
        names, _ = names_and_centre(case)
        pooled = with_density(np.concatenate(runs))
        each = [with_density(r) for r in runs]
        header = "name,p2.5,p16,p50,p84,p97.5\n"
        with open(f"{OUT}/{case}_posterior.csv", "w", newline="\n") as fh, \
                open(f"{OUT}/{case}_spread.csv", "w", newline="\n") as sp:
            fh.write(header)
            sp.write(header)
            for j, (name, col) in enumerate(zip(names + ["density"], pooled)):
                q = np.percentile(col, QUANTILES)
                fh.write(name + "," + ",".join(repr(float(v)) for v in q) + "\n")
                # How far the runs stray from the pooled, in sigma.
                sigma_q = (q[3] - q[1]) / 2
                spread = np.array([np.percentile(r[j], QUANTILES) for r in each])
                worst = np.abs(spread - q).max(axis=0) / sigma_q
                sp.write(name + "," + ",".join(f"{v:.3f}" for v in worst) + "\n")
                print(case, name, "runs from the pooled, sigma:", np.round(worst, 2), flush=True)
