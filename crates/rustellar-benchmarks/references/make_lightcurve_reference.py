"""Write reference light curves: batman's orbit geometry with Agol's flux.

Writes two files into crates/rustellar-model/tests/reference/:

    lightcurve_sets.csv    one row per parameter set
    lightcurve_points.csv  one row per time: the set, the time, and
        z       batman 2.5.1's planet-star separation (its _rsky), or 100
                where batman puts the planet behind the star
        batman  batman's flux
        agol    exoplanet-core 0.4.0's flux at batman's z (Agol, Luger &
                Foreman-Mackey 2020), 1 behind the star

So the geometry and the flux come from two independent codes, and the flux
from the most precise published one.

The grid runs from small planets to eclipsing binaries (k up to 10), with
impact parameters set relative to k so every set has a full transit, inner
contact, a grazing one and a near miss. Half the sets put t0 about 4300 days
(a Kepler-era ephemeris) before the data (TESS era), to check the phase over
many orbits.

Run from the repository root:

    python crates/rustellar-benchmarks/references/make_lightcurve_reference.py

The output has no date in it, so running again with the same versions gives
identical files.
"""

import itertools
import math
import platform
import sys
from pathlib import Path

import batman
import exoplanet_core
import numpy as np
from batman import _rsky
from exoplanet_core import quad_limbdark_light_curve

OUT_DIR = Path(__file__).resolve().parents[3] / "crates" / "rustellar-model" / "tests" / "reference"

T_DATA = 3000.123456  # data near here (TESS era)
T0_FAR = -1300.654321  # a Kepler-era ephemeris
PERIOD = 3.0
BEHIND = 100.0  # batman's separation when the planet is behind the star

RADIUS_RATIOS = [0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0]
# Impact parameter as a function of k: central, typical, inner contact,
# limb at the centre, grazing, and a near miss.
IMPACTS = [
    ("0", lambda k: 0.0),
    ("0.3", lambda k: 0.3),
    ("0.6", lambda k: 0.6),
    ("0.9", lambda k: 0.9),
    ("|1-k|", lambda k: abs(1.0 - k)),
    ("1", lambda k: 1.0),
    ("1+k/2", lambda k: 1.0 + k / 2.0),
    ("1+k+0.01", lambda k: 1.0 + k + 0.01),
]
# Not 100, which is also batman's marker for "behind the star".
SEMI_MAJOR_AXES = [1.5, 3.0, 10.0, 30.0, 80.0, 300.0]
LIMB_DARKENING = [(0.0, 0.0), (0.4, 0.25), (0.6, 0.1), (0.05, 0.4)]
N_WINDOW = 29


def half_duration(edge, b, a_rs):
    if b >= edge:
        return None
    ratio = min((edge * edge - b * b) / (a_rs * a_rs - b * b), 1.0)
    return PERIOD / math.pi * math.asin(math.sqrt(ratio)) / 2.0


def times_for(k, b, a_rs, t0):
    # The transit nearest T_DATA.
    centre = t0 + round((T_DATA - t0) / PERIOD) * PERIOD
    half_t14 = half_duration(1.0 + k, b, a_rs)
    half_t23 = half_duration(abs(1.0 - k), b, a_rs) if k != 1.0 else None
    window = 1.2 * (half_t14 if half_t14 is not None else 0.05 * PERIOD)
    offsets = list(np.linspace(-window, window, N_WINDOW))
    for half in (half_t14, half_t23):
        if half is not None:
            offsets += [-half, half]
    offsets += [PERIOD / 4.0, PERIOD / 2.0, 0.6 * PERIOD]
    return np.array(sorted(centre + dt for dt in offsets))


def main():
    sets, points = [], []
    grid = itertools.product(RADIUS_RATIOS, IMPACTS, SEMI_MAJOR_AXES, LIMB_DARKENING)
    for k, (b_name, b_of), a_rs, (u1, u2) in grid:
        b = b_of(k)
        if a_rs <= 1.0 + k or b >= a_rs:
            continue
        index = len(sets)
        t0 = T0_FAR if index % 2 else T_DATA
        inc = math.acos(b / a_rs)
        t = times_for(k, b, a_rs, t0)

        z = _rsky._rsky(t, t0, PERIOD, a_rs, inc, 0.0, math.pi / 2.0, 1, 1)
        params = batman.TransitParams()
        params.t0, params.per, params.rp, params.a = t0, PERIOD, k, a_rs
        params.inc, params.ecc, params.w = math.degrees(inc), 0.0, 90.0
        params.u, params.limb_dark = [u1, u2], "quadratic"
        bat = batman.TransitModel(params, t).light_curve(params)
        behind = z == BEHIND
        agol = 1.0 + quad_limbdark_light_curve(u1, u2, np.where(behind, 1e3, z), np.full(z.size, k))
        agol[behind] = 1.0

        sets.append((index, k, b_name, b, a_rs, PERIOD, t0, u1, u2))
        for row in zip(t, z, bat, agol):
            points.append((index,) + tuple(float(x) for x in row))

    common = [
        f"# batman {batman.__version__} (separation and flux), exoplanet-core {exoplanet_core.__version__} (flux at batman's",
        f"# separation), numpy {np.__version__}, Python {platform.python_version()}. Circular orbits, quadratic limb darkening.",
        f"# Grid: k in {RADIUS_RATIOS}, b in {[name for name, _ in IMPACTS]}, a_rs in {SEMI_MAJOR_AXES} (a_rs > 1 + k),",
        f"# (u1, u2) in {LIMB_DARKENING}. t0 alternates {T_DATA} and {T0_FAR}; times around the transit nearest {T_DATA}.",
        f"# z = {BEHIND:g} where batman puts the planet behind the star. Times in days (BTJD), lengths in stellar radii.",
    ]
    files = {
        "lightcurve_sets.csv": [f"# {len(sets)} parameter sets for lightcurve_points.csv."]
        + common
        + ["set,k,b_name,b,a_rs,period,t0,u1,u2"]
        + [",".join([str(s[0]), repr(s[1]), s[2]] + [repr(float(x)) for x in s[3:]]) for s in sets],
        "lightcurve_points.csv": [f"# {len(points)} points over the sets in lightcurve_sets.csv."]
        + common
        + ["set,t,z,batman,agol"]
        + [",".join([str(p[0])] + [repr(x) for x in p[1:]]) for p in points],
    }
    for name, lines in files.items():
        path = OUT_DIR / name
        with open(path, "w", newline="\n", encoding="ascii") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)", file=sys.stderr)


if __name__ == "__main__":
    main()
