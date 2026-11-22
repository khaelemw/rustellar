"""zeus 2.5.4's ensemble slice sampler beside rustellar-mcmc's slice moves.

The move study's targets (a 10-d standard normal, 40 walkers, 20,000
steps; the 2-d Rosenbrock density, 20 walkers, 50,000 steps), three seeds,
with zeus's differential and Gaussian moves; then the transit posterior of
the crate's reference light curve with batman 2.5.1 and zeus's
differential move, 40 walkers, 30,000 steps, the first 5,000 discarded.
tau is emcee's integrated_time (c = 5), the largest over the parameters;
log-probabilities per independent sample are zeus's calls over the
independent samples (for the transit, the calls after the discarded
steps, while the walkers spread from a small ball, are left out).

    python crates/rustellar-benchmarks/references/mcmc_zeus_study.py
"""
import math
import time
import warnings

import batman
import emcee
import numpy as np
import zeus

warnings.filterwarnings("ignore")


def gauss(x):
    return -0.5 * np.sum(x * x, axis=1)


def rosen(x):
    return -(100.0 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1.0 - x[:, 0]) ** 2) / 20.0


TARGETS = [("gauss10", gauss, 10, 40, 20000, 2000, np.zeros(10)),
           ("rosenbrock", rosen, 2, 20, 50000, 5000, np.array([1.0, 1.0]))]
MOVES = [("slice", lambda: zeus.moves.DifferentialMove()),
         ("slice gaussian", lambda: zeus.moves.GaussianMove())]

print("target,move,tau,sd,calls per walker-step,per independent sample,s per 1000 steps")
for name, lp, d, w, steps, discard, centre in TARGETS:
    for mname, mk in MOVES:
        taus, per, secs = [], 0.0, 0.0
        for seed in range(3):
            np.random.seed(seed)
            p0 = centre + 2.0 * (np.random.rand(w, d) - 0.5)
            s = zeus.EnsembleSampler(w, d, lp, moves=mk(), vectorize=True, verbose=False)
            t = time.perf_counter()
            s.run_mcmc(p0, steps, progress=False)
            secs += time.perf_counter() - t
            tau = emcee.autocorr.integrated_time(s.get_chain(discard=discard), quiet=True)
            taus.append(tau.max())
            per += s.ncall / (w * steps)
        mean, sd, per = np.mean(taus), np.std(taus), per / 3
        print(f"{name},{mname},{mean:.2f},{sd:.2f},{per:.2f},{mean * per:.1f},{1000 * secs / 3 / steps:.3f}",
              flush=True)

# The transit posterior, as mcmc_transit_speed.py's.
data = np.loadtxt("crates/rustellar-mcmc/tests/reference/transit_points.csv", delimiter=",", skiprows=1)
t, flux, err = data[:, 0], data[:, 1], data[:, 2]
p = batman.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.0, 0.1, 10.0, 88.28, 0.0, 90.0
p.limb_dark, p.u = "quadratic", [0.4, 0.25]
model = batman.TransitModel(p, t)


def log_prob(x):
    t0, k, a, b, q1, q2, base = x
    if not (0 < k < 1 and a > 1 and 0 <= b < 1 + k and b < a and 0 <= q1 <= 1
            and 0 <= q2 <= 1 and base > 0):
        return -np.inf
    r = math.sqrt(q1)
    p.t0, p.rp, p.a, p.inc = t0, k, a, math.degrees(math.acos(b / a))
    p.u = [2 * r * q2, r * (1 - 2 * q2)]
    m = model.light_curve(p)
    return -0.5 * np.sum(((base * m - flux) / err) ** 2)


walkers, steps, discard = 40, 30_000, 5_000
centre = np.array([0.0, 0.1, 10.0, 0.3, 0.4225, 0.3077, 1.0])
np.random.seed(3)
p0 = centre + 1e-4 * np.random.randn(walkers, 7)
s = zeus.EnsembleSampler(walkers, 7, log_prob, moves=zeus.moves.DifferentialMove(), verbose=False)
start = time.perf_counter()
s.run_mcmc(p0, steps, progress=False)
secs = time.perf_counter() - start
tau = emcee.autocorr.integrated_time(s.get_chain(discard=discard), quiet=True).max()
ess = walkers * (steps - discard) / tau
calls_per_step = s.neval[discard:].sum() / (walkers * (steps - discard))
print("zeus tuned width", s.mu, "(rustellar-mcmc units", 2 * s.mu, ")")
print("model,move,seconds,tau,independent samples,calls per independent sample,ms per independent sample")
print(f"batman,zeus slice,{secs:.1f},{tau:.0f},{ess:.0f},{calls_per_step * tau:.0f},{1000 * secs / ess:.2f}")
