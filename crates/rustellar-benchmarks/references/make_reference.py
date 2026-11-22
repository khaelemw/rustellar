"""Write reference light curves from batman for the rustellar-model tests.

Runs batman (Kreidberg 2015) over a grid of circular-orbit transits with the
quadratic limb-darkening law and writes two CSV files into
crates/rustellar-model/tests/reference/:

    batman_sets.csv    one row per parameter set
    batman_points.csv  one row per time: the set it belongs to, the time, and
                       batman's flux

Run from the repository root, with the versions in tools/requirements.txt:

    python crates/rustellar-benchmarks/references/make_reference.py

The output has no date in it, so running the script again with the same
versions gives identical files, and `git diff` shows only real changes.

batman prints "Convergence failure in ellpic_bulirsch" for the sets with
b = 1 - k, where the planet touches the stellar limb from inside at
mid-transit. This is expected: its results there are still accurate to about
1e-8, and those sets are kept on purpose, because z = 1 - k is a special case
of the analytic formula.
"""

import itertools
import math
import platform
import sys
from pathlib import Path

import batman
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[3] / "crates" / "rustellar-model" / "tests" / "reference"

# A realistic TESS-era mid-transit time (BTJD), so the time arithmetic in the
# tests works on real-sized numbers. The period only rescales time, so one
# value is enough.
T0 = 3000.123456
PERIOD = 3.0

# Radius ratio, impact parameter, scaled semi-major axis and quadratic
# limb-darkening coefficients. Every combination has b < a_rs. b = 1.05 with
# a small k misses the star, which checks that the flux is exactly 1.
RADIUS_RATIOS = [0.005, 0.02, 0.05, 0.1, 0.2, 0.29]
IMPACT_PARAMETERS = [0.0, 0.3, 0.6, 0.9, 0.95, 1.0, 1.05]
SEMI_MAJOR_AXES = [1.5, 3.0, 8.0, 20.0, 100.0]
LIMB_DARKENING = [(0.0, 0.0), (0.3, 0.25), (0.6, 0.1), (0.05, 0.4)]

# Evenly spaced times across the transit window, which runs a little past
# first and fourth contact so it includes some points out of transit.
N_WINDOW = 25
WINDOW_SCALE = 1.3

# How far above 1 batman's flux may go before the script refuses to write it.
FLUX_OVERSHOOT = 1e-9


def contact_half_duration(edge, b, a_rs):
    """Half the time between the two points where the separation is `edge`.

    Returns None if the separation never gets that small. The argument of
    asin is capped at 1, which gives half a period's span when the orbit is
    so tight the planet never clears the star on the near side.
    """
    if b >= edge:
        return None
    ratio = min((edge * edge - b * b) / (a_rs * a_rs - b * b), 1.0)
    return PERIOD / math.pi * math.asin(math.sqrt(ratio)) / 2.0


def times_for(k, b, a_rs):
    """The times to evaluate for one parameter set, sorted."""
    half_t14 = contact_half_duration(1.0 + k, b, a_rs)
    half_t23 = contact_half_duration(1.0 - k, b, a_rs)

    # No transit: still sample a window around t0.
    window = WINDOW_SCALE * (half_t14 if half_t14 is not None else 0.05 * PERIOD)

    offsets = list(np.linspace(-window, window, N_WINDOW))
    for half in (half_t14, half_t23):
        if half is not None:
            offsets += [-half, half]
    offsets += [PERIOD / 4.0, PERIOD / 2.0]

    return np.array(sorted(T0 + dt for dt in offsets))


def batman_flux(k, b, a_rs, u1, u2, t):
    params = batman.TransitParams()
    params.t0 = T0
    params.per = PERIOD
    params.rp = k
    params.a = a_rs
    params.inc = math.degrees(math.acos(b / a_rs))
    params.ecc = 0.0
    params.w = 90.0
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    return batman.TransitModel(params, t).light_curve(params)


def header(description):
    return [
        f"# {description}",
        f"# Flux from batman {batman.__version__} (numpy {np.__version__}, "
        f"Python {platform.python_version()}), quadratic limb darkening,",
        "# circular orbits (ecc = 0, w = 90), inc = degrees(acos(b / a_rs)).",
        f"# Grid: k in {RADIUS_RATIOS}, b in {IMPACT_PARAMETERS},",
        f"# a_rs in {SEMI_MAJOR_AXES}, (u1, u2) in {LIMB_DARKENING}.",
        f"# Times per set: {N_WINDOW} evenly spaced over +-{WINDOW_SCALE} x T14/2 "
        "(+-0.05 P if no transit),",
        "# the contact times, t0 + P/4 and t0 + P/2.",
        "# Times in days (BTJD), lengths in stellar radii, flux relative to out of transit.",
    ]


def main():
    set_rows = []
    point_rows = []

    grid = itertools.product(RADIUS_RATIOS, IMPACT_PARAMETERS, SEMI_MAJOR_AXES, LIMB_DARKENING)
    for index, (k, b, a_rs, (u1, u2)) in enumerate(grid):
        inc_deg = math.degrees(math.acos(b / a_rs))
        t = times_for(k, b, a_rs)
        flux = batman_flux(k, b, a_rs, u1, u2, t)

        if np.isnan(flux).any():
            sys.exit(f"set {index}: batman returned NaN (k={k}, b={b}, a_rs={a_rs})")
        # batman returns up to about 4e-10 above 1 exactly at first and fourth
        # contact, where the planet only touches the star. That is batman's
        # rounding, kept as it is; anything larger means broken output.
        if (flux < 0.0).any() or (flux > 1.0 + FLUX_OVERSHOOT).any():
            sys.exit(f"set {index}: flux outside [0, 1] (k={k}, b={b}, a_rs={a_rs})")

        # repr() round-trips a float exactly, so the tests rebuild the very
        # inputs and times batman saw.
        set_rows.append(
            ",".join([str(index)] + [repr(float(x)) for x in (k, b, a_rs, inc_deg, PERIOD, T0, u1, u2)])
        )
        for ti, fi in zip(t, flux):
            point_rows.append(f"{index},{float(ti)!r},{float(fi):.15g}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "batman_sets.csv": header(f"{len(set_rows)} parameter sets for batman_points.csv.")
        + ["set,k,b,a_rs,inc_deg,period,t0,u1,u2"]
        + set_rows,
        "batman_points.csv": header(f"{len(point_rows)} points from batman, over the sets in batman_sets.csv.")
        + ["set,t,flux"]
        + point_rows,
    }
    for name, lines in files.items():
        path = OUT_DIR / name
        with open(path, "w", newline="\n", encoding="ascii") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)", file=sys.stderr)

    print(f"{len(set_rows)} sets, {len(point_rows)} points", file=sys.stderr)


if __name__ == "__main__":
    main()
