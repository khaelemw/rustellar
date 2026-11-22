"""Wall time per independent sample of emcee 3.1.6 on the transit posterior
of rustellar-mcmc's reference light curve, with batman 2.5.1 and with
PyTransit 2.6.18 (QuadraticModel) as the model; the stretch and DE moves.
crates/rustellar-mcmc/tests/study.rs prints the same for this crate.

Free: t0, k, a/R*, b, q1, q2, baseline; uniform priors. 40 walkers, 30,000
steps, the first 5,000 discarded; tau the largest of emcee's
integrated_time over the coordinates. One thread.

    python crates/rustellar-benchmarks/references/mcmc_transit_speed.py
"""
import math
import time
import warnings

import batman
import emcee
import numpy as np
from pytransit import QuadraticModel

warnings.filterwarnings("ignore")
data = np.loadtxt("crates/rustellar-mcmc/tests/reference/transit_points.csv", delimiter=",", skiprows=1)
t, flux, err = data[:, 0], data[:, 1], data[:, 2]


def bat_params(t0, k, a, b, u1, u2):
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = t0, 3.0, k, a
    p.inc = math.degrees(math.acos(b / a))
    p.ecc, p.w = 0.0, 90.0
    p.limb_dark, p.u = "quadratic", [u1, u2]
    return p


bat_model = batman.TransitModel(bat_params(0.0, 0.1, 10.0, 0.3, 0.4, 0.25), t)
pt_model = QuadraticModel()
pt_model.set_data(t)


def batman_flux(t0, k, a, b, u1, u2):
    return bat_model.light_curve(bat_params(t0, k, a, b, u1, u2))


def pytransit_flux(t0, k, a, b, u1, u2):
    return pt_model.evaluate(k, np.array([u1, u2]), t0, 3.0, a, math.acos(b / a), 0.0, 0.0)


def make_log_prob(model):
    def log_prob(x):
        t0, k, a, b, q1, q2, base = x
        if not (0 < k < 1 and a > 1 and 0 <= b < 1 + k and b < a and 0 <= q1 <= 1
                and 0 <= q2 <= 1 and base > 0):
            return -np.inf
        r = math.sqrt(q1)
        m = model(t0, k, a, b, 2 * r * q2, r * (1 - 2 * q2))
        return -0.5 * np.sum(((base * m - flux) / err) ** 2)
    return log_prob


# The two models agree.
diff = np.max(np.abs(batman_flux(0.0, 0.1, 10.0, 0.3, 0.4, 0.25) - pytransit_flux(0.0, 0.1, 10.0, 0.3, 0.4, 0.25)))
print("max |batman - PyTransit|", diff)

walkers, steps, discard = 40, 30_000, 5_000
centre = np.array([0.0, 0.1, 10.0, 0.3, 0.4225, 0.3077, 1.0])
print("model,move,seconds,tau,independent samples,ms per independent sample")
for name, model in [("batman", batman_flux), ("PyTransit", pytransit_flux)]:
    lp = make_log_prob(model)
    lp(centre)  # compile numba
    for mname, mv in [("stretch", emcee.moves.StretchMove()), ("de", emcee.moves.DEMove())]:
        np.random.seed(3)
        p0 = centre + 1e-4 * np.random.randn(walkers, 7)
        s = emcee.EnsembleSampler(walkers, 7, lp, moves=mv)
        start = time.perf_counter()
        s.run_mcmc(p0, steps)
        secs = time.perf_counter() - start
        tau = emcee.autocorr.integrated_time(s.get_chain(discard=discard), quiet=True).max()
        ess = walkers * (steps - discard) / tau
        print(f"{name},{mname},{secs:.1f},{tau:.0f},{ess:.0f},{1000 * secs / ess:.2f}", flush=True)
