"""Reference posterior for rustellar-mcmc's transit posterior: emcee 3.1.6
with batman 2.5.1 on one hot Jupiter's transit.

The light curve (500 points, 0.0006 d apart about mid-transit, white noise
of 1e-3 from SplitMix64 and Box-Muller) is written out, so the Rust test
reads the same flux. Free: t0, k, a/R*, b, q1, q2 and the baseline, the
period fixed at 3 d; priors uniform (q1 and q2 on [0, 1], b on [0, 1 + k)).
emcee's stretch move, 40 walkers, 60,000 steps, the first 10,000 discarded;
the 2.5, 16, 50, 84 and 97.5 percentiles of each coordinate and of the
stellar density are written out.

    python crates/rustellar-benchmarks/references/make_transit_posterior_reference.py
"""
import math
import os
import time

import batman
import emcee
import numpy as np

OUT = "crates/rustellar-mcmc/tests/reference"
MASK = (1 << 64) - 1
G = 6.674_30e-11


class SplitMix:
    def __init__(self, seed):
        self.s = seed & MASK

    def unit(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
        return (z >> 11) * 2.0**-53

    def normal(self):
        u1 = 1.0 - self.unit()
        u2 = self.unit()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


TRUTH = dict(t0=0.0, per=3.0, rp=0.1, a=10.0, inc=None, b=0.3, u=[0.4, 0.25])
N, STEP, SIGMA = 500, 0.0006, 1e-3
NAMES = ["t0", "k", "a_rs", "b", "q1", "q2", "baseline"]


def params(t0, k, a, b, u1, u2):
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = t0, 3.0, k, a
    p.inc = math.degrees(math.acos(b / a))
    p.ecc, p.w = 0.0, 90.0
    p.limb_dark, p.u = "quadratic", [u1, u2]
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    t = np.array([-0.5 * (N - 1) * STEP + STEP * i for i in range(N)])
    truth = params(0.0, 0.1, 10.0, 0.3, 0.4, 0.25)
    model = batman.TransitModel(truth, t)
    g = SplitMix(2026)
    flux = model.light_curve(truth) + SIGMA * np.array([g.normal() for _ in range(N)])
    with open(f"{OUT}/transit_points.csv", "w", newline="\n") as fh:
        fh.write("time,flux,flux_err\n")
        for ti, fi in zip(t, flux):
            fh.write(f"{float(ti)!r},{float(fi)!r},{SIGMA!r}\n")

    def log_prob(x):
        t0, k, a, b, q1, q2, base = x
        if not (0 < k < 1 and a > 1 and 0 <= b < 1 + k and b < a and 0 <= q1 <= 1
                and 0 <= q2 <= 1 and base > 0):
            return -np.inf
        r = math.sqrt(q1)
        m = model.light_curve(params(t0, k, a, b, 2 * r * q2, r * (1 - 2 * q2)))
        return -0.5 * np.sum(((base * m - flux) / SIGMA) ** 2)

    walkers, steps, discard = 40, 60_000, 10_000
    rng = np.random.default_rng(1)
    centre = np.array([0.0, 0.1, 10.0, 0.3, 0.4225, 0.3077, 1.0])
    p0 = centre + 1e-4 * rng.standard_normal((walkers, 7))
    np.random.seed(2)
    sampler = emcee.EnsembleSampler(walkers, 7, log_prob)
    start = time.perf_counter()
    sampler.run_mcmc(p0, steps)
    secs = time.perf_counter() - start
    chain = sampler.get_chain(discard=discard, flat=True)
    tau = emcee.autocorr.integrated_time(sampler.get_chain(discard=discard), quiet=True)
    period = 3.0 * 86400.0
    density = 3 * math.pi * chain[:, 2] ** 3 / (G * period**2) / 1000.0
    cols = [chain[:, i] for i in range(7)] + [density]
    with open(f"{OUT}/transit_posterior.csv", "w", newline="\n") as fh:
        fh.write("name,p2.5,p16,p50,p84,p97.5\n")
        for name, c in zip(NAMES + ["density"], cols):
            q = np.percentile(c, [2.5, 16, 50, 84, 97.5])
            fh.write(name + "," + ",".join(repr(float(v)) for v in q) + "\n")
    print("seconds", secs, "tau", tau, "acceptance", sampler.acceptance_fraction.mean())


if __name__ == "__main__":
    main()
