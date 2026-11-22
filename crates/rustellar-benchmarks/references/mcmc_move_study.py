"""emcee 3.1.6's autocorrelation times per move, for rustellar-mcmc's
move study (crates/rustellar-mcmc/tests/study.rs prints the same table).

Targets: a 10-d standard normal (any Gaussian, for affine-invariant
moves) with 40 walkers, and Goodman & Weare's 2-d Rosenbrock density
exp(-(100 (y - x^2)^2 + (1 - x)^2) / 20) with 20 walkers. tau is emcee's
integrated_time (c = 5), the largest over the parameters, averaged over
three seeds; with one log-probability per walker per step it is also the
log-probabilities per independent sample.

    python crates/rustellar-benchmarks/references/mcmc_move_study.py
"""
import time
import numpy as np
import emcee


def gauss(x):
    return -0.5 * np.sum(x * x, axis=1)


def rosen(x):
    return -(100.0 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1.0 - x[:, 0]) ** 2) / 20.0


TARGETS = [("gauss10", gauss, 10, 40, 20000, 2000, np.zeros(10)),
           ("rosenbrock", rosen, 2, 20, 50000, 5000, np.array([1.0, 1.0]))]
MOVES = [("stretch", lambda: emcee.moves.StretchMove()),
         ("walk", lambda: emcee.moves.WalkMove()),
         ("de", lambda: emcee.moves.DEMove())]

for name, lp, d, w, steps, discard, centre in TARGETS:
    for mname, mk in MOVES:
        # emcee's walk move takes a covariance and an SVD per walker per
        # step (about 1 ms): one shorter run on the Gaussian only.
        seeds = 3
        if mname == "walk":
            if name != "gauss10":
                continue
            seeds, steps, discard = 1, 6000, 1000
        taus, secs = [], 0.0
        for seed in range(seeds):
            np.random.seed(seed)
            p0 = centre + 0.1 * np.random.randn(w, d)
            s = emcee.EnsembleSampler(w, d, lp, moves=mk(), vectorize=True)
            t = time.perf_counter()
            s.run_mcmc(p0, steps)
            secs += time.perf_counter() - t
            tau = emcee.autocorr.integrated_time(s.get_chain(discard=discard), quiet=True)
            taus.append(tau.max())
        print(f"{name},{mname},{np.mean(taus):.2f},{np.std(taus):.2f},{secs / seeds:.2f}", flush=True)
