"""Write reference values of the transit flux at given separations.

For a grid of radius ratio k, separation z and quadratic limb darkening
(u1, u2), writes crates/rustellar-model/tests/reference/flux_points.csv with:

    exact      the defining integral, by adaptive quadrature (scipy), computed
               two ways (in r and in theta = asin(r)); a point is written only
               if the two agree to AGREEMENT
    batman     batman 2.5.1's analytic quadratic model (Kreidberg 2015)
    pytransit  PyTransit 2.6.18's analytic quadratic model (Parviainen 2015)
    agol       exoplanet-core 0.4.0's quadratic model (Agol, Luger &
               Foreman-Mackey 2020)

The grid covers k from 1e-4 to 20 and every line where the analytic formula
changes case (z = 0, k, 1 - k, 1 + k, k - 1), on the line and at distances
1e-12 to 1e-3 either side, including just inside and outside 1e-5.

Run from the repository root:

    python crates/rustellar-benchmarks/references/make_flux_reference.py

The output has no date in it, so running again with the same versions gives
identical files.
"""

import math
import platform
import sys
from pathlib import Path

import batman
import exoplanet_core
import numpy as np
import pytransit
import scipy
from batman import _quadratic_ld
from exoplanet_core import quad_limbdark_light_curve
from pytransit.models.numba.ma_quadratic_nb import eval_quad_z_s
from scipy.integrate import quad

OUT = (
    Path(__file__).resolve().parents[3]
    / "crates" / "rustellar-model" / "tests" / "reference" / "flux_points.csv"
)

RADIUS_RATIOS = [
    1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.45, 0.5, 0.55,
    0.7, 0.9, 0.99, 1.0, 1.01, 1.5, 2.0, 5.0, 10.0, 20.0,
]
SUN_LIKE = (0.4, 0.25)
# Uniform, linear with a dark limb, strong quadratic, the darkest allowed
# quadratic (intensity mu^2), and limb brightening.
OTHER_LAWS = [(0.0, 0.0), (1.0, 0.0), (0.1, 0.6), (2.0, -1.0), (-0.3, 0.5)]
OTHER_LAW_RADIUS_RATIOS = [0.01, 0.1, 0.5, 1.5]
GRID_POINTS = 13
OFFSETS = [0.0, 1e-12, 1e-9, 1e-6, 9.9e-6, 1.01e-5, 1e-3]
AGREEMENT = 2e-14


def intensity(mu, u1, u2):
    return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2


def disk_integral(u1, u2):
    return 1.0 - u1 / 3.0 - u2 / 6.0


def arc(r, z, k):
    """Angle of the ring of radius r that lies inside the planet."""
    if r <= 0.0:
        return math.pi if k > z else 0.0
    if r <= k - z:
        return 2.0 * math.pi
    if r <= z - k or r >= z + k:
        return 0.0
    c = (r * r + z * z - k * k) / (2.0 * r * z)
    return 2.0 * math.acos(min(1.0, max(-1.0, c)))


def breakpoints(z, k):
    lo = max(0.0, z - k)
    hi = min(1.0, z + k)
    inner = abs(k - z)
    points = sorted({p for p in (lo, inner, hi) if lo <= p <= hi})
    return lo, hi, points


def blocked_in_r(z, k, u1, u2):
    lo, hi, points = breakpoints(z, k)
    total = 0.0
    for a, b in zip(points, points[1:]):
        if b > a:
            value, _ = quad(
                lambda r: intensity(math.sqrt(max(0.0, 1.0 - r * r)), u1, u2) * r * arc(r, z, k),
                a, b, epsabs=1e-17, epsrel=1e-15, limit=2000,
            )
            total += value
    return total / math.pi


def blocked_in_theta(z, k, u1, u2):
    lo, hi, points = breakpoints(z, k)
    total = 0.0
    for a, b in zip(points, points[1:]):
        if b > a:
            value, _ = quad(
                lambda t: intensity(math.cos(t), u1, u2)
                * math.sin(t) * math.cos(t) * arc(math.sin(t), z, k),
                math.asin(a), math.asin(b), epsabs=1e-17, epsrel=1e-15, limit=2000,
            )
            total += value
    return total / math.pi


def exact_flux(z, k, u1, u2):
    """The exact flux, or None if the two quadratures disagree."""
    if z >= 1.0 + k:
        return 1.0
    if k > 1.0 and z <= k - 1.0:
        return 0.0
    in_r = blocked_in_r(z, k, u1, u2)
    in_theta = blocked_in_theta(z, k, u1, u2)
    if abs(in_r - in_theta) > AGREEMENT:
        return None
    return 1.0 - 0.5 * (in_r + in_theta) / disk_integral(u1, u2)


def separations(k):
    zs = {(1.0 + k) * i / (GRID_POINTS - 1) for i in range(GRID_POINTS)}
    lines = [0.0, k, 1.0 + k, abs(1.0 - k)]
    for line in lines:
        for d in OFFSETS:
            for z in (line - d, line + d):
                if z >= 0.0 and z <= 1.0 + k + 1e-3 and z >= k - 1.0 - 1e-3:
                    zs.add(z)
    return sorted(zs)


def main():
    rows = []
    skipped = 0
    cases = [(k, SUN_LIKE) for k in RADIUS_RATIOS]
    cases += [(k, law) for k in OTHER_LAW_RADIUS_RATIOS for law in OTHER_LAWS]
    for k, (u1, u2) in cases:
        for z in separations(k):
            exact = exact_flux(z, k, u1, u2)
            if exact is None:
                skipped += 1
                continue
            bat = float(_quadratic_ld._quadratic_ld(np.array([z]), k, u1, u2, 1)[0])
            try:
                pyt = float(eval_quad_z_s(z, k, np.array([u1, u2])))
            except ZeroDivisionError:
                # PyTransit divides by zero at some exact special cases; nan
                # marks "no value" and the tests skip it.
                pyt = math.nan
            agol = 1.0 + float(quad_limbdark_light_curve(u1, u2, np.array([z]), np.array([k]))[0])
            rows.append((k, z, u1, u2, exact, bat, pyt, agol))

    header = [
        f"# {len(rows)} points: flux at separation z for radius ratio k and quadratic limb darkening (u1, u2).",
        "# exact: the defining integral by adaptive quadrature (scipy "
        f"{scipy.__version__}), in r and in asin(r), kept where the two agree to {AGREEMENT:g}.",
        f"# batman: batman {batman.__version__} _quadratic_ld (Kreidberg 2015).",
        f"# pytransit: PyTransit {pytransit.__version__} eval_quad_z_s (Parviainen 2015); nan where it divides by zero.",
        f"# agol: exoplanet-core {exoplanet_core.__version__} quad_limbdark_light_curve (Agol, Luger & Foreman-Mackey 2020).",
        f"# numpy {np.__version__}, Python {platform.python_version()}. Lengths in stellar radii.",
        "k,z,u1,u2,exact,batman,pytransit,agol",
    ]
    with OUT.open("w", newline="\n") as f:
        f.write("\n".join(header) + "\n")
        for row in rows:
            f.write(",".join(repr(float(x)) for x in row) + "\n")
    print(f"wrote {len(rows)} points to {OUT}, skipped {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
