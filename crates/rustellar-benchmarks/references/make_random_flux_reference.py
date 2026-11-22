"""Write exact flux at random points over the whole valid parameter range.

Writes crates/rustellar-model/tests/reference/flux_random.csv: POINTS points
from a scrambled Sobol sequence (even coverage, no clusters or gaps, fixed
seed), each with

    exact   the defining integral by adaptive quadrature (scipy): the mean of
            the integral in r and in asin(r)
    spread  half the difference between those two, an estimate of the
            error in exact (every point is kept, however large)
    agol   exoplanet-core's quadratic model (Agol, Luger & Foreman-Mackey 2020)

The ranges:

    k       log-uniform from 1e-4 to 100 (RadiusRatio::MAX)
    z       uniform over the transit, from max(0, k - 1) to 1 + k, for 95% of
            points; the rest uniform over twice that range, to check the
            exact 0 and 1 outside it
    u1, u2  70% from Kipping (2013): q1, q2 uniform in [0, 1], u1 =
            2 sqrt(q1) q2, u2 = sqrt(q1) (1 - 2 q2), the physical laws (dark
            limb, intensity falling outwards); the rest uniform over
            u1 in [-1, 3], u2 in [-2, 2], kept if the intensity is never
            negative (limb brightening and the edges of the allowed range)

Run from the repository root:

    python crates/rustellar-benchmarks/references/make_random_flux_reference.py

Takes a few minutes. The output has no date in it, so running again with the
same versions gives identical files.
"""

import math
import platform
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import exoplanet_core
import numpy as np
import scipy
from exoplanet_core import quad_limbdark_light_curve
from scipy.integrate import IntegrationWarning
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_flux_reference import blocked_in_r, blocked_in_theta, disk_integral  # noqa: E402

OUT = (
    Path(__file__).resolve().parents[3]
    / "crates" / "rustellar-model" / "tests" / "reference" / "flux_random.csv"
)

POINTS = 32768  # a power of two, as Sobol sequences prefer
SEED = 20261115
K_MIN, K_MAX = 1e-4, 100.0
KIPPING_SHARE = 0.7
OUTSIDE_SHARE = 0.05


def allowed(u1, u2):
    """The same rule as LimbDarkening::quadratic: intensity never negative."""
    if u1 + u2 > 1.0:
        return False
    lowest_inside = u2 < 0.0 and 0.0 < u1 < -2.0 * u2
    return not (lowest_inside and u1 * u1 + 4.0 * u2 > 0.0)


def point(sample):
    """Map a point of the unit hypercube to (k, z, u1, u2)."""
    s_k, s_z, s_a, s_b, s_kind, s_out = sample
    k = math.exp(math.log(K_MIN) + s_k * (math.log(K_MAX) - math.log(K_MIN)))
    lo, hi = max(0.0, k - 1.0), 1.0 + k
    if s_out < OUTSIDE_SHARE:
        z = 2.0 * hi * s_z
    else:
        z = lo + (hi - lo) * s_z
    if s_kind < KIPPING_SHARE:
        q1, q2 = s_a, s_b
        u1, u2 = 2.0 * math.sqrt(q1) * q2, math.sqrt(q1) * (1.0 - 2.0 * q2)
    else:
        u1, u2 = -1.0 + 4.0 * s_a, -2.0 + 4.0 * s_b
    return k, z, u1, u2


def exact(args):
    """(exact flux, spread), with spread 0 where the flux is exactly 0 or 1."""
    warnings.simplefilter("ignore", IntegrationWarning)
    k, z, u1, u2 = args
    if z >= 1.0 + k:
        return 1.0, 0.0
    if k > 1.0 and z <= k - 1.0:
        return 0.0, 0.0
    d = disk_integral(u1, u2)
    in_r = blocked_in_r(z, k, u1, u2)
    in_theta = blocked_in_theta(z, k, u1, u2)
    return 1.0 - 0.5 * (in_r + in_theta) / d, 0.5 * abs(in_r - in_theta) / d


def main():
    sampler = qmc.Sobol(d=6, scramble=True, seed=SEED)
    candidates = [point(s) for s in sampler.random(POINTS)]
    rejected = sum(not allowed(u1, u2) for _, _, u1, u2 in candidates)
    candidates = [c for c in candidates if allowed(c[2], c[3])]

    with Pool() as pool:
        exacts = pool.map(exact, candidates, chunksize=64)

    rows = []
    for (k, z, u1, u2), (ex, spread) in zip(candidates, exacts):
        agol = 1.0 + float(quad_limbdark_light_curve(u1, u2, np.array([z]), np.array([k]))[0])
        rows.append((k, z, u1, u2, ex, spread, agol))

    header = [
        f"# {len(rows)} points: flux at random (k, z, u1, u2) over the whole valid range (Sobol, seed {SEED}).",
        f"# k log-uniform in [{K_MIN:g}, {K_MAX:g}]; z over the transit ({1 - OUTSIDE_SHARE:.0%}) or twice it;",
        f"# (u1, u2) {KIPPING_SHARE:.0%} Kipping (2013) q1, q2 uniform, the rest uniform in [-1, 3] x [-2, 2] where allowed.",
        f"# exact: the defining integral by adaptive quadrature (scipy {scipy.__version__}), in r and in asin(r), "
        f"the mean of the two; spread is half their difference ({rejected} disallowed laws not drawn).",
        f"# agol: exoplanet-core {exoplanet_core.__version__} quad_limbdark_light_curve (Agol, Luger & Foreman-Mackey 2020).",
        f"# numpy {np.__version__}, Python {platform.python_version()}. Lengths in stellar radii.",
        "k,z,u1,u2,exact,spread,agol",
    ]
    with OUT.open("w", newline="\n", encoding="ascii") as f:
        f.write("\n".join(header) + "\n")
        for row in rows:
            f.write(",".join(repr(float(x)) for x in row) + "\n")
    print(f"wrote {len(rows)} points to {OUT}, rejected {rejected}", file=sys.stderr)


if __name__ == "__main__":
    main()
