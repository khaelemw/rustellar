"""Reference data for rustellar-bls, from astropy's BoxLeastSquares.

First checks that this astropy reproduces two small worked examples (2 and
3 below), which later reference files build on, then writes the files
need, each recording the astropy and numpy versions in its header.

Run from the repository root:

    python crates/rustellar-benchmarks/references/make_bls_reference.py
"""

import math
import os
import platform
from fractions import Fraction
import sys
from pathlib import Path

import astropy
import numpy as np
import transitleastsquares.version as tls_version
from astropy.timeseries import BoxLeastSquares
from transitleastsquares import tls_constants
from transitleastsquares.grid import period_grid

OUT = Path(__file__).resolve().parents[3] / "crates" / "rustellar-bls" / "tests" / "reference"
# The Ofir grid is rustellar-core's.
CORE_OUT = Path(__file__).resolve().parents[3] / "crates" / "rustellar-core" / "tests" / "reference"

VERSIONS = (f"astropy {astropy.__version__}, numpy {np.__version__}, "
            f"Python {platform.python_version()}")
TLS_VERSIONS = (f"transitleastsquares {tls_version.TLS_VERSIONING}, "
                f"numpy {np.__version__}, Python {platform.python_version()}")


def example_2():
    """autoperiod for 1,000 even times over 27 d, durations 0.05, 0.1, 0.2 d."""
    t = np.linspace(0, 27, 1000)
    periods = BoxLeastSquares(t, np.ones_like(t)).autoperiod([0.05, 0.1, 0.2])
    assert len(periods) == 35371, len(periods)
    assert periods[0] == 0.4
    assert periods[1] == 0.4000109742379764
    assert periods[-1] == 13.500000000000021


def example_3():
    """A box away from phase zero, where astropy's fast and slow agree."""
    t, y, dy = example_3_data()
    bls = BoxLeastSquares(t, y, dy)
    for method in ["fast", "slow"]:
        r = bls.power([3.0], [0.12], method=method)
        assert r.power[0] == 348.1879707001639, (method, r.power[0])
        assert r.depth[0] == 0.005078553094469474, (method, r.depth[0])
        assert r.duration[0] == 0.12, (method, r.duration[0])
        assert r.transit_time[0] == 1.302942952053283, (method, r.transit_time[0])
        assert r.depth_snr[0] == 25.482750349124515, (method, r.depth_snr[0])


# (t_min, t_max, durations, minimum_period, maximum_period, minimum_n_transit,
# frequency_factor); None takes astropy's default. Times are BTJD-like, so
# the baseline is a difference of large numbers, as in real data.
AUTOPERIOD_CASES = [
    (0.0, 27.0, [0.05, 0.1, 0.2], None, None, 3, 1.0),            # worked example 2
    (1325.293, 1353.177, [0.04, 0.08, 0.12, 0.16], None, None, 3, 1.0),  # one TESS sector
    (1325.293, 1353.177, [0.01, 0.35], 0.5, 15.0, 3, 1.0),       # the pipeline's range
    (1325.293, 1353.177, [0.08], None, None, 2, 1.0),             # two transits
    (1325.293, 1353.177, [0.08], None, None, 5, 1.0),             # five transits
    (1325.293, 1353.177, [0.1, 0.2], None, None, 3, 0.37),        # finer
    (1325.293, 1353.177, [0.1, 0.2], None, None, 3, 7.5),         # coarser
    (1325.293, 2061.508, [0.08, 0.25], 1.0, 50.0, 3, 25.0),       # two years
    (2.5e-3, 3.1, [0.02, 0.05], None, None, 3, 1.0),              # short baseline
    (1325.293, 1353.177, [0.2, 0.05, 0.1], 0.7, 0.7, 3, 1.0),     # one period
    (1325.293, 1353.177, [0.05], 0.3, None, 4, 2.0),              # given minimum only
    (1325.293, 1353.177, [0.05], None, 9.0, 3, 2.0),              # given maximum only
    # (4 - 0.25) / 1.5 = 2.5 exactly: numpy rounds the tie to 2 (even), not 3.
    (0.0, 1.0, [0.25], 0.25, 4.0, 3, 6.0),
]


def autoperiod_case(case):
    t_min, t_max, durations, p_min, p_max, n_transit, factor = case
    t = np.array([t_min, 0.5 * (t_min + t_max), t_max])
    return BoxLeastSquares(t, np.ones_like(t)).autoperiod(
        durations, minimum_period=p_min, maximum_period=p_max,
        minimum_n_transit=n_transit, frequency_factor=factor)


def write_lines(path, lines):
    with path.open("w", newline="", encoding="ascii") as f:
        for line in lines:
            f.write(line + "\n")


def write_autoperiod():
    def opt(x):
        return "" if x is None else repr(float(x))
    cases, grids = [], []
    for i, case in enumerate(AUTOPERIOD_CASES):
        periods = autoperiod_case(case)
        t_min, t_max, durations, p_min, p_max, n_transit, factor = case
        cases.append(f"{i},{t_min!r},{t_max!r},{';'.join(repr(d) for d in durations)},"
                     f"{opt(p_min)},{opt(p_max)},{n_transit},{factor!r},{len(periods)}")
        n = len(periods)
        # Each period depends only on its index, so about 1,000 evenly spread
        # indices, with both ends, pin down the whole grid.
        keep = sorted(set(range(0, n, max(1, n // 1000))) | set(range(min(n, 25)))
                      | set(range(max(0, n - 25), n)))
        grids += [f"{i},{j},{float(periods[j])!r}" for j in keep]
    header = f"# astropy BoxLeastSquares.autoperiod ({VERSIONS})."
    write_lines(OUT / "autoperiod_cases.csv",
                [header, "# Empty minimum_period or maximum_period: astropy's default.",
                 "case,t_min,t_max,durations,minimum_period,maximum_period,"
                 "minimum_n_transit,frequency_factor,count"] + cases)
    write_lines(OUT / "autoperiod_grids.csv",
                [header + f" {len(grids)} sampled periods: about 1,000 per case, evenly spread, with the first and last 25.",
                 "case,index,period"] + grids)
    print(f"wrote {len(cases)} autoperiod cases, {len(grids)} periods", file=sys.stderr)


# IAU 2015 Resolution B3 nominal solar values, and the day in seconds.
GM_SUN = 1.3271244e20   # m^3 s^-2
R_SUN = 6.957e8         # m
DAY = 86400.0


def ofir_grid(baseline, radius, mass, p_min, p_max, oversampling, n_transit):
    """Ofir (2014) eqs 3, 5-7, written as rustellar-bls computes them.

    Unset edges: f_min = n_transit / S, f_max = the Roche limit at 3 R.
    Returns periods in days, shortest first, every x = 1..floor(N_opt).
    """
    r = radius * R_SUN
    gm = mass * GM_SUN
    s = baseline * DAY
    f_min = n_transit / s if p_max is None else 1.0 / (p_max * DAY)
    if p_min is None:
        r3 = 3.0 * r
        f_max = np.sqrt(gm / (r3 * r3 * r3)) / (2.0 * np.pi)
    else:
        f_max = 1.0 / (p_min * DAY)
    a = np.cbrt(4.0 * np.pi * np.pi) / np.pi * r / np.cbrt(gm) / (s * oversampling)
    c = np.cbrt(f_min) - a / 3.0
    n_opt = (np.cbrt(f_max) - np.cbrt(f_min) + a / 3.0) * 3.0 / a
    n = max(int(np.floor(n_opt)), 1)
    u = a / 3.0 * np.arange(n, 0, -1, dtype=float) + c
    return 1.0 / (u * u * u) / DAY


def check_ofir_against_tls(baseline, radius, mass, oversampling, n_transit):
    """TLS's period_grid, given the same constants, gives the same grid
    (to rounding), plus at most one period below the Roche limit: TLS takes
    x up to ceil(N_opt), the paper up to N_opt."""
    saved = tls_constants.G, tls_constants.R_sun, tls_constants.M_sun
    tls_constants.G, tls_constants.R_sun = 6.6743e-11, R_SUN
    tls_constants.M_sun = GM_SUN / tls_constants.G
    try:
        theirs = np.sort(period_grid(radius, mass, baseline,
                                     oversampling_factor=oversampling,
                                     n_transits_min=n_transit))
    finally:
        tls_constants.G, tls_constants.R_sun, tls_constants.M_sun = saved
    ours = ofir_grid(baseline, radius, mass, None, None, oversampling, n_transit)
    assert len(theirs) - len(ours) in (0, 1), (len(theirs), len(ours))
    np.testing.assert_allclose(theirs[len(theirs) - len(ours):], ours, rtol=1e-12, atol=0)
    return len(theirs) - len(ours)


# (t_min, t_max, stellar_radius, stellar_mass, minimum_period, maximum_period,
# oversampling_factor, minimum_n_transit); None takes the default edge.
OFIR_CASES = [
    (1325.293, 1353.177, 1.0, 1.0, None, None, 3.0, 2),       # one sector, the Sun
    (1325.293, 1353.177, 0.3, 0.25, None, None, 3.0, 2),      # M dwarf
    (1325.293, 1353.177, 2.5, 1.4, None, None, 3.0, 2),       # subgiant
    (1325.293, 1353.177, 1.0, 1.0, None, None, 5.0, 3),       # denser, three transits
    (1325.293, 1353.177, 1.0, 1.0, None, None, 1.5, 2),       # coarser
    (1325.293, 1353.177, 0.9, 0.95, 0.5, 15.0, 3.0, 2),       # the pipeline's range
    (1325.293, 1353.177, 1.0, 1.0, 0.5, None, 3.0, 2),        # given minimum only
    (1325.293, 1353.177, 1.0, 1.0, None, 6.0, 3.0, 2),        # given maximum only
    (1325.293, 2061.508, 1.0, 1.0, 1.0, 50.0, 3.0, 2),        # two years
    (2.5e-3, 3.1, 1.0, 1.0, None, None, 3.0, 2),              # short baseline
    (1325.293, 1353.177, 1.0, 1.0, 0.7, 0.7, 3.0, 2),         # one period
]


def write_ofir():
    def opt(x):
        return "" if x is None else repr(float(x))
    cases, grids, extra = [], [], []
    for i, case in enumerate(OFIR_CASES):
        t_min, t_max, radius, mass, p_min, p_max, oversampling, n_transit = case
        baseline = t_max - t_min
        if p_min is None and p_max is None:
            extra.append(check_ofir_against_tls(baseline, radius, mass, oversampling, n_transit))
        periods = ofir_grid(baseline, radius, mass, p_min, p_max, oversampling, n_transit)
        n = len(periods)
        cases.append(f"{i},{t_min!r},{t_max!r},{radius!r},{mass!r},{opt(p_min)},"
                     f"{opt(p_max)},{oversampling!r},{n_transit},{n}")
        keep = sorted(set(range(0, n, max(1, n // 1000))) | set(range(min(n, 25)))
                      | set(range(max(0, n - 25), n)))
        grids += [f"{i},{j},{float(periods[j])!r}" for j in keep]
    header = (f"# Ofir (2014) period grid, IAU 2015 nominal GM_sun and R_sun, from "
              f"numpy; checked against {TLS_VERSIONS} given the same constants.")
    write_lines(CORE_OUT / "ofir_cases.csv",
                [header, "# Empty minimum_period or maximum_period: the default edge.",
                 "case,t_min,t_max,stellar_radius,stellar_mass,minimum_period,"
                 "maximum_period,oversampling_factor,minimum_n_transit,count"] + cases)
    write_lines(CORE_OUT / "ofir_grids.csv",
                [header + f" {len(grids)} sampled periods: about 1,000 per case, evenly spread, with the first and last 25.",
                 "case,index,period"] + grids)
    print(f"wrote {len(cases)} Ofir cases, {len(grids)} periods; TLS's extra "
          f"sub-Roche periods per checked case: {extra}", file=sys.stderr)


def exact_in_transit(t, t0, p, d):
    """In transit, in exact rational arithmetic on the floats' values."""
    s = Fraction(t) - Fraction(t0)
    n = round(s / Fraction(p))
    return abs(s - n * Fraction(p)) < Fraction(d) / 2


def astropy_in_transit(t, t0, p, d):
    """astropy's own BoxLeastSquares.transit_mask, for one time."""
    times = np.array([t])
    bls = BoxLeastSquares(times, np.ones_like(times))
    return bool(bls.transit_mask(times, p, d, t0)[0])


def write_in_transit():
    """Times within 1e-12 d of a box edge, 1,000 to 3,000 orbits from t0."""
    rng = np.random.default_rng(23)
    rows, wrong = [], 0
    for case in range(20):
        p = float(rng.uniform(0.5, 1.5))
        d = float(rng.uniform(0.02, 0.3))
        t0 = float(rng.uniform(1.0, 2.0))
        for _ in range(100):
            n = int(rng.integers(1000, 3000))
            edge = float(rng.choice([-0.5, 0.5])) * d
            t = t0 + n * p + edge + float(rng.uniform(-1e-12, 1e-12))
            exact = exact_in_transit(t, t0, p, d)
            astropy = astropy_in_transit(t, t0, p, d)
            wrong += exact != astropy
            rows.append(f"{case},{p!r},{d!r},{t0!r},{t!r},{int(exact)},{int(astropy)}")
    header = (f"# In transit or not, for times within 1e-12 d of a box edge: exact (rational "
              f"arithmetic) and astropy's transit_mask ({VERSIONS}); astropy wrong on {wrong}.")
    write_lines(OUT / "in_transit.csv",
                [header, "case,period,duration,transit_time,time,exact,astropy"] + rows)
    print(f"wrote {len(rows)} in-transit times; astropy wrong on {wrong}", file=sys.stderr)


def exact_best_box(t, y, e, period, duration):
    """The best box (largest delta chi-squared with depth > 0) over every
    mid-transit time, in exact rational arithmetic on the floats' values.

    A point is in transit for a box whose left edge is at phase a iff
    0 < (phase - a) mod P < D, so the in-transit set changes only where a
    meets a point's phase or its phase less D. Every set is found by trying
    each such breakpoint and the middle of each gap between them.
    Returns (delta_chi2, mask) or None.
    """
    P, D = Fraction(period), Fraction(duration)
    t0 = Fraction(t[0])
    phase = [(Fraction(ti) - t0) % P for ti in t]
    w = [1 / (Fraction(ei) * Fraction(ei)) for ei in e]
    yy = [Fraction(yi) for yi in y]
    breaks = sorted(set(phase) | {(f - D) % P for f in phase})
    gaps = [(a + b) / 2 for a, b in zip(breaks, breaks[1:])]
    gaps.append(((breaks[-1] + breaks[0] + P) / 2) % P)
    best = None
    for a in breaks + gaps:
        mask = [0 < (f - a) % P < D for f in phase]
        n_in = sum(mask)
        if n_in == 0 or n_in == len(t):
            continue
        w_in = sum(wi for wi, m in zip(w, mask) if m)
        w_out = sum(wi for wi, m in zip(w, mask) if not m)
        y_in = sum(wi * yi for wi, yi, m in zip(w, yy, mask) if m) / w_in
        y_out = sum(wi * yi for wi, yi, m in zip(w, yy, mask) if not m) / w_out
        depth = y_out - y_in
        if depth <= 0:
            continue
        dchi2 = w_in * w_out / (w_in + w_out) * depth * depth
        if best is None or dchi2 > best[0]:
            best = (dchi2, mask)
    return best


def write_exact():
    """Small data sets, 6 to 20 points, and the exact best box for each."""
    rng = np.random.default_rng(31)
    cases, points = [], []
    for case in range(240):
        n = int(rng.integers(6, 21))
        if case < 200:
            # Random times, away from zero as real data is.
            t = np.sort(1325.0 + rng.uniform(0, 10, n))
            period = float(rng.uniform(0.5, 3.0))
            duration = float(rng.uniform(0.05, 0.45)) * period
            e = rng.uniform(0.5e-3, 2e-3, n)
        else:
            # Times on an exact grid, with the duration a whole number of
            # steps: points exactly D apart in phase, where events coincide.
            t = 1325.0 + 0.25 * np.sort(rng.choice(80, n, replace=False))
            period = float(rng.choice([1.0, 1.5, 2.0, 2.5]))
            duration = 0.25 * float(rng.integers(1, int(period / 0.25)))
            e = np.full(n, 1e-3)
        y = 1.0 + rng.normal(0, 1e-3, n)
        y[rng.uniform(size=n) < 0.3] -= 2e-3
        best = exact_best_box(t, y, e, period, duration)
        if best is None:
            dchi2, mask = "", ""
        else:
            dchi2 = repr(float(best[0]))
            mask = "".join("1" if m else "0" for m in best[1])
        cases.append(f"{case},{period!r},{duration!r},{dchi2},{mask}")
        points += [f"{case},{float(a)!r},{float(b)!r},{float(c)!r}" for a, b, c in zip(t, y, e)]
    header = (f"# The best box over every mid-transit time, in exact rational arithmetic "
              f"(Python {platform.python_version()} fractions); empty if no box has a dip.")
    write_lines(OUT / "exact_cases.csv",
                [header, "case,period,duration,delta_chi2,mask"] + cases)
    write_lines(OUT / "exact_points.csv",
                [header, "case,time,flux,flux_err"] + points)
    print(f"wrote {len(cases)} exact cases", file=sys.stderr)


def example_3_data():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 20, 400))
    y = 1 + 1e-3 * rng.standard_normal(400)
    y[np.abs((t - 1.3 + 1.5) % 3.0 - 1.5) < 0.06] -= 0.005
    return t, y, 1e-3 * np.ones_like(t)


def write_slow():
    """Worked example 3's data, and astropy's slow search on it, objective snr."""
    t, y, dy = example_3_data()
    write_lines(OUT / "example_3_points.csv",
                [f"# Worked example 3: numpy default_rng(1) ({VERSIONS}).", "time,flux,flux_err"]
                + [f"{a!r},{b!r},{c!r}" for a, b, c in zip(t.tolist(), y.tolist(), dy.tolist())])
    periods = np.linspace(1.0, 6.0, 300)
    durations = [0.06, 0.12, 0.2]
    r = BoxLeastSquares(t, y, dy).power(periods, durations, objective="snr", method="slow")
    rows = [f"{p!r},{s!r},{d!r},{t0!r},{dep!r}" for p, s, d, t0, dep in
            zip(periods.tolist(), r.power.tolist(), r.duration.tolist(),
                r.transit_time.tolist(), r.depth.tolist())]
    write_lines(OUT / "slow_power.csv",
                [f"# astropy BoxLeastSquares.power, method slow, objective snr, oversample 10, "
                 f"durations {durations} ({VERSIONS}).", "period,snr,duration,transit_time,depth"]
                + rows)
    print(f"wrote astropy slow on {len(rows)} periods", file=sys.stderr)


# ---------------------------------------------------------------------------
# Shared light-curve recipe. rustellar-bls's tests build the same light curves
# from the same seeds (src/tests/recipe.rs), bit for bit: only + - * / and
# floor, which IEEE 754 rounds the same everywhere, so only astropy's results
# need storing.

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def unit(self):
        return float(self.next_u64() >> 11) / 9007199254740992.0


def recipe(seed, n, t_start, cadence, noise, period, tc, duration, depth):
    """Times every `cadence`, jittered by up to half a cadence (the first
    exactly t_start); uniform noise of half-width `noise`; a box of `depth`
    within duration / 2 of tc + k period; errors from 0.75 to 1.25 noise."""
    rng = SplitMix64(seed)
    t, y, e = [], [], []
    for i in range(n):
        u_t, u_y, u_e = rng.unit(), rng.unit(), rng.unit()
        ti = t_start if i == 0 else t_start + cadence * i + 0.5 * cadence * u_t
        yi = noise * (2.0 * u_y - 1.0)
        if depth > 0.0:
            r = (ti - tc) / period
            f = r - math.floor(r)
            if min(f, 1.0 - f) * period < 0.5 * duration:
                yi -= depth
        t.append(ti)
        y.append(yi)
        e.append(noise * (0.75 + 0.5 * u_e))
    return np.array(t), np.array(y), np.array(e)


def samples(t, y, e):
    """Values the Rust recipe must reproduce bit for bit."""
    n = len(t)
    picks = [0, n // 3, 2 * n // 3, n - 1]
    return ";".join(f"{i}:{float(t[i])!r}:{float(y[i])!r}:{float(e[i])!r}" for i in picks)


def box_bins(transit_time, duration, period, t_min, bin_width, oversample):
    """astropy's start bin (0-based), length and bins round the orbit, from its
    reported box, where the period is a whole number of bins. astropy's box n
    holds its bins n + 1 .. n + k; bins m .. m + oversample - 1 are copies of
    1 .. oversample, the first on top of the last real bin, m. The box is
    clean, holding only real bins with their own sums, if n + k <= m - 1. Its
    reported time is (n b + k b / 2 + t_min) mod P, which gives n only mod m,
    so a box that could be n or n + m (k = oversample, n = 0) is not clean."""
    m = round(period / bin_width)
    assert m * bin_width == period
    k = round(duration / bin_width)
    assert k * bin_width == duration
    x = ((transit_time - t_min - 0.5 * duration) % period) / bin_width
    start = round(x)
    assert abs(x - start) < 1e-6, (transit_time, x)
    start %= m
    clean = start + k <= m - 1 and start + m > m + oversample - k
    return start, k, m, clean


# (seed, period, tc, duration, depth) on 1,350 points every 0.02 d from 1325.
FAST_CURVES = [
    (11, 3.1, 1326.2, 0.1, 3e-3),      # mid-phase at its own period
    (12, 2.3, 1325.0, 0.15, 2e-3),     # centred on the first point
    (13, 4.7, 1324.95, 0.12, 2.5e-3),  # just before the first point: the last bins
    (14, 1.0, 1325.0, 0.1, 0.0),       # noise only
]
# Bins of 1/256 d, and periods of a whole number of them, so astropy's bins
# and the crate's (which tile the orbit exactly) are the same bins.
FAST_DURATIONS = [0.03125, 0.08, 0.12, 0.16]
FAST_OVERSAMPLE = 8


def write_fast():
    """astropy's fast search, objective snr, on the shared recipe."""
    b = min(FAST_DURATIONS) / FAST_OVERSAMPLE
    assert b == 1 / 256
    periods = np.unique(np.round(np.geomspace(205, 2304, 250))) / 256
    curves, rows = [], []
    for c, (seed, period, tc, duration, depth) in enumerate(FAST_CURVES):
        t, y, e = recipe(seed, 1350, 1325.0, 0.02, 1e-3, period, tc, duration, depth)
        curves.append(f"{c},{seed},{period!r},{tc!r},{duration!r},{depth!r},{samples(t, y, e)}")
        r = BoxLeastSquares(t, y, e).power(periods, FAST_DURATIONS, objective="snr",
                                           oversample=FAST_OVERSAMPLE)
        for p, snr, dep, dur, t0 in zip(periods.tolist(), r.power.tolist(), r.depth.tolist(),
                                        r.duration.tolist(), r.transit_time.tolist()):
            if snr == -np.inf:
                # No box with a dip: astropy leaves the rest uninitialised.
                rows.append(f"{c},{p!r},-inf,,,,,,,")
                continue
            start, k, m, clean = box_bins(t0, dur, p, t.min(), b, FAST_OVERSAMPLE)
            rows.append(f"{c},{p!r},{snr!r},{dep!r},{dur!r},{t0!r},{start},{k},{m},{int(clean)}")
    header = (f"# astropy BoxLeastSquares.power, method fast, objective snr, oversample "
              f"{FAST_OVERSAMPLE}, durations {FAST_DURATIONS}, on the shared recipe ({VERSIONS}).")
    write_lines(OUT / "fast_curves.csv",
                [header, "curve,seed,period,tc,duration,depth,samples"] + curves)
    write_lines(OUT / "fast_power.csv",
                [header + " start and bins: astropy's box, 0-based; m bins round the orbit; "
                 "clean: its box holds only real bins.",
                 "curve,period,snr,depth,duration,transit_time,start,bins,m,clean"] + rows)
    print(f"wrote astropy fast on {len(curves)} curves, {len(rows)} periods", file=sys.stderr)


# (name, tc, true duration) at period 3 on 15,000 points every 0.002 d from
# 1325, depth 0.01, noise 1e-4, searched with durations 3/64, 3/32, 3/16 and
# oversample 8: the bin is 3/512 d, so the period is 512 bins exactly, bin 511
# the last, the longest box 32 bins against astropy's 8 padded. The cadence divides the period, so each orbit's points fall at
# the same phases (plus up to 0.001 d of jitter): the last bin holds two
# points an orbit, both in the last_bin transit.
WRAP_CASES = [
    ("mid_phase", 1326.5, 0.1),
    ("last_bin", 1325.0 + 2.9975, 0.004),
    ("phase_zero_short", 1325.0, 0.05),
    ("phase_zero_long", 1325.0, 0.2),
]
WRAP_DURATIONS = [3 / 64, 3 / 32, 3 / 16]
WRAP_OVERSAMPLE = 8


def write_wrap():
    """astropy fast and slow where fast's wrap-around goes wrong."""
    rows = []
    for c, (name, tc, duration) in enumerate(WRAP_CASES):
        t, y, e = recipe(100 + c, 15000, 1325.0, 0.002, 1e-4, 3.0, tc, duration, 0.01)
        for method in ["fast", "slow"]:
            r = BoxLeastSquares(t, y, e).power([3.0], WRAP_DURATIONS, objective="snr",
                                               method=method,
                                               oversample=WRAP_OVERSAMPLE)
            rows.append(f"{name},{100 + c},{tc!r},{duration!r},{method},{float(r.power[0])!r},"
                        f"{float(r.depth[0])!r},{float(r.duration[0])!r},"
                        f"{float(r.transit_time[0])!r},"
                        f"{samples(t, y, e)}")
    write_lines(OUT / "wrap_cases.csv",
                [f"# astropy fast and slow, objective snr, oversample {WRAP_OVERSAMPLE}, period 3, durations "
                 f"{WRAP_DURATIONS}, 15000 points every 0.002 d, depth 0.01, noise 1e-4, on the "
                 f"shared recipe ({VERSIONS}).",
                 "name,seed,tc,duration,method,snr,depth,box_duration,transit_time,samples"] + rows)
    for r in rows:
        print("  " + ",".join(r.split(",")[:9]), file=sys.stderr)


# Injected transits: 144 light curves of 2,000 points every 0.0135 d (27 d)
# from 1325, noise 1e-3, a box transit at a random period, mid-transit time,
# duration and snr. Three groups of 48: snr 8 to 20 at a random phase; snr
# 50 to 150 at a random phase; snr 8 to 20 within 0.03 d of the first
# point's phase.
# Each is searched at 201 periods within 2% of its own (the pipeline's zoom)
# with astropy's fast method, objective snr, oversample 10.
INJECTED_DURATIONS = [0.04, 0.06, 0.08, 0.11, 0.15, 0.2]


def injected_periods(period):
    """201 periods within 2% of `period`, with + - * / only."""
    return [period * (1.0 + 0.02 * (k - 100) / 100.0) for k in range(201)]


def write_injected():
    rng = SplitMix64(4242)
    rows = []
    for c in range(144):
        period = 1.5 + 6.5 * rng.unit()
        duration = 0.05 + 0.13 * rng.unit()
        snr = 8.0 + 12.0 * rng.unit()
        if 48 <= c < 96:
            snr = 50.0 + 100.0 * rng.unit()
        if c < 96:
            tc = 1325.0 + period * rng.unit()
        else:
            tc = 1325.0 + 0.06 * rng.unit() - 0.03
        n_in = duration / 0.0135 * (27.0 / period)
        depth = snr * 1e-3 / math.sqrt(n_in)
        t, y, e = recipe(5000 + c, 2000, 1325.0, 0.0135, 1e-3, period, tc, duration, depth)
        periods = injected_periods(period)
        r = BoxLeastSquares(t, y, e).power(periods, INJECTED_DURATIONS, objective="snr",
                                           oversample=10)
        k = int(np.argmax(r.power))
        rows.append(f"{c},{5000 + c},{period!r},{tc!r},{duration!r},{depth!r},"
                    f"{float(r.period[k])!r},{float(r.transit_time[k])!r},"
                    f"{float(r.depth[k])!r},{float(r.duration[k])!r},{float(r.power[k])!r},"
                    f"{samples(t, y, e)}")
    write_lines(OUT / "injected.csv",
                [f"# Injected box transits on the shared recipe (2000 points every 0.0135 d, noise "
                 f"1e-3), and astropy's best box: fast method, objective snr, oversample 10, "
                 f"durations {INJECTED_DURATIONS}, 201 periods within 2% of the true one "
                 f"({VERSIONS}).",
                 "case,seed,period,tc,duration,depth,astropy_period,astropy_transit_time,"
                 "astropy_depth,astropy_duration,astropy_snr,samples"] + rows)
    print(f"wrote {len(rows)} injected transits", file=sys.stderr)


# ---------------------------------------------------------------------------
# Survey light curves. A second shared recipe (src/tests/recipe.rs,
# `Survey`): segments of evenly spaced, jittered times, each point kept with
# probability `keep`, and the same noise, errors and box as `recipe`. Four
# draws a point, kept or not, so the stream does not depend on `keep`.

def survey(seed, segments, cadence, noise, keep, period, tc, duration, depth):
    rng = SplitMix64(seed)
    t, y, e = [], [], []
    for start, n in segments:
        for i in range(n):
            u_k, u_t, u_y, u_e = rng.unit(), rng.unit(), rng.unit(), rng.unit()
            if u_k >= keep:
                continue
            ti = start + cadence * i + 0.5 * cadence * u_t
            yi = noise * (2.0 * u_y - 1.0)
            if depth > 0.0 and in_box(ti, period, tc, duration):
                yi -= depth
            t.append(ti)
            y.append(yi)
            e.append(noise * (0.75 + 0.5 * u_e))
    return np.array(t), np.array(y), np.array(e)


def in_box(ti, period, tc, duration):
    r = (ti - tc) / period
    f = r - math.floor(r)
    return min(f, 1.0 - f) * period < 0.5 * duration


def sector(start, cadence, n):
    """A TESS sector: two orbits of n points, a day apart."""
    return [(start, n), (start + n * cadence + 1.0, n)]


def survey_families(rng):
    """(name, count, segments, cadence, noise, keep, durations, longest period)."""
    c2, c30 = 2.0 / 1440.0, 1.0 / 48.0
    nights = [(1325.0 + d, 86) for d in range(60) if rng.unit() < 0.7]
    return [
        ("tess_2min", 4, sector(1325.0, c2, 9000), c2, 1e-3, 1.0, [0.04, 0.09, 0.2], 12.0),
        ("tess_30min_three_sectors", 10,
         sector(1325.0, c30, 620) + sector(1690.25, c30, 620) + sector(2055.5, c30, 620),
         c30, 5e-4, 0.95, [0.08, 0.16, 0.3], 20.0),
        ("kepler_quarters", 6, [(1325.0 + 93.0 * q, 4150) for q in range(4)], 0.0204, 3e-4,
         0.3, [0.1, 0.2, 0.4], 40.0),
        ("ground_nights", 8, nights, 5.0 / 1440.0, 2e-3, 0.9, [0.04, 0.08, 0.15], 10.0),
        ("tess_2min_thinned", 8, sector(1325.0, c2, 9000), c2, 1e-3, 0.1,
         [0.04, 0.09, 0.2], 12.0),
    ]


def write_survey():
    rng = SplitMix64(8100)
    curves, rows = [], []
    seed = 8200
    for name, count, segments, cadence, noise, keep, durations, longest in survey_families(rng):
        for _ in range(count):
            seed += 1
            period = 0.6 + (min(longest, 15.0) - 0.6) * rng.unit()
            duration = durations[0] + (durations[2] - durations[0]) * rng.unit()
            tc = 1325.0 + period * rng.unit()
            snr = 7.0 + 18.0 * rng.unit()
            noise_only = rng.unit() < 0.25
            t, _, _ = survey(seed, segments, cadence, noise, keep, period, tc, duration, 0.0)
            n_in = sum(in_box(ti, period, tc, duration) for ti in t.tolist())
            depth = 0.0 if noise_only or n_in == 0 else snr * noise / math.sqrt(3.0 * n_in)
            t, y, e = survey(seed, segments, cadence, noise, keep, period, tc, duration, depth)
            c = len(curves)
            segs = "|".join(f"{a!r}:{n}" for a, n in segments)
            curves.append(f"{c},{name},{seed},{segs},{cadence!r},{noise!r},{keep!r},"
                          f"{period!r},{tc!r},{duration!r},{depth!r},"
                          f"{':'.join(repr(d) for d in durations)},{len(t)},{samples(t, y, e)}")
            near = [period * (1.0 + 0.01 * (k - 5.5) / 5.5) for k in range(12)]
            spread = [0.5 * (longest / 0.5) ** rng.unit() for _ in range(12)]
            periods = sorted(near + spread)
            bls = BoxLeastSquares(t, y, e)
            fast = bls.power(periods, durations, objective="snr", oversample=10)
            slow = bls.power(periods, durations, objective="snr", method="slow", oversample=10)
            for k, p in enumerate(periods):
                rows.append(f"{c},{p!r},{float(fast.power[k])!r},{float(fast.duration[k])!r},"
                            f"{float(fast.transit_time[k])!r},{float(slow.power[k])!r}")
            print(f"  {c} {name}: {len(t)} points, depth {depth:.2e}", file=sys.stderr)
    header = (f"# Survey light curves on the shared recipe, and astropy's fast and slow "
              f"searches, objective snr, oversample 10, at 24 periods each ({VERSIONS}).")
    write_lines(OUT / "survey_curves.csv",
                [header, "curve,family,seed,segments,cadence,noise,keep,period,tc,duration,"
                 "depth,durations,points,samples"] + curves)
    write_lines(OUT / "survey_power.csv",
                [header + " fast_duration and fast_transit_time: fast's box; -inf: no box "
                 "with a dip (astropy leaves the rest uninitialised).",
                 "curve,period,fast_snr,fast_duration,fast_transit_time,slow_snr"] + rows)
    print(f"wrote {len(curves)} survey curves, {len(rows)} periods", file=sys.stderr)


# ---------------------------------------------------------------------------
# Injection-recovery. Box transits injected into survey light
# curves with baselines a blind search can cover (27 to 60 days), searched
# over the same Ofir grid (a Sun-like star, oversampling 3, 0.6 days to half
# the baseline) by astropy's fast search (objective snr, oversample 10) and,
# on a tenth (every fortieth group of four, so each family equally), by
# transitleastsquares given the same grid constants.

RECOVERY_DURATIONS = [0.04, 0.06, 0.08, 0.11, 0.15, 0.2]
RECOVERY_COUNT = 1000
TLS_EVERY = 10


def recovery_families(rng):
    """(name, segments, cadence, noise, keep)."""
    c2, c30 = 2.0 / 1440.0, 1.0 / 48.0
    nights = [(1325.0 + d, 86) for d in range(60) if rng.unit() < 0.7]
    return [
        ("tess_30min_sector", sector(1325.0, c30, 620), c30, 5e-4, 0.95),
        ("tess_2min_thinned", sector(1325.0, c2, 9000), c2, 1e-3, 0.1),
        ("tess_30min_two_sectors", sector(1325.0, c30, 620) + sector(1352.5, c30, 620), c30,
         5e-4, 0.95),
        ("ground_nights", nights, 5.0 / 1440.0, 2e-3, 0.9),
    ]


def recovery_wide_families(rng):
    """The four families above, with fresh ground-night weather, plus a full
    2-minute sector and one Kepler-like quarter."""
    c2 = 2.0 / 1440.0
    return recovery_families(rng) + [
        ("tess_2min_sector", sector(1325.0, c2, 9000), c2, 1e-3, 1.0),
        ("kepler_quarter", [(1325.0, 4150)], 0.0204, 3e-4, 0.5),
    ]


def run_tls(t, y, e, baseline, threads=4):
    """transitleastsquares over the same Ofir grid: the same constants and
    range. Returns period, T0, depth (1 - its bottom flux), duration, SDE."""
    import contextlib, io
    from transitleastsquares import transitleastsquares
    saved = tls_constants.G, tls_constants.R_sun, tls_constants.M_sun
    tls_constants.G, tls_constants.R_sun = 6.6743e-11, R_SUN
    tls_constants.M_sun = GM_SUN / tls_constants.G
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = transitleastsquares(t, y + 1.0, e).power(
                period_min=0.6, period_max=baseline / 2.0, oversampling_factor=3.0,
                R_star=1.0, M_star=1.0, use_threads=threads, show_progress_bar=False)
    finally:
        tls_constants.G, tls_constants.R_sun, tls_constants.M_sun = saved
    return (float(r.period), float(r.T0), 1.0 - float(r.depth), float(r.duration),
            float(r.SDE))


def write_recovery():
    write_recovery_set("recovery", 9100, 9200, RECOVERY_COUNT, recovery_families, TLS_EVERY,
                       6.0, "a tenth")


def write_recovery_wide():
    """A second, independent and wider set: other seeds, six families, snr
    from 5."""
    write_recovery_set("recovery_wide", 9300, 19200, 3000, recovery_wide_families, 20, 5.0,
                       "a twentieth")


def write_recovery_set(stem, rng_seed, case_seed, count, make_families, tls_every, snr_low,
                       tls_share):
    rng = SplitMix64(rng_seed)
    families = make_families(rng)
    fam_rows = [f"{k},{name},{'|'.join(f'{a!r}:{n}' for a, n in segs)},{cad!r},{noise!r},{keep!r}"
                for k, (name, segs, cad, noise, keep) in enumerate(families)]
    rows, tls_rows = [], []
    for c in range(count):
        k = c % len(families)
        name, segments, cadence, noise, keep = families[k]
        seed = case_seed + c
        t, _, _ = survey(seed, segments, cadence, noise, keep, 1.0, 0.0, 0.0, 0.0)
        baseline = float(t.max() - t.min())
        while True:
            period = 0.7 * ((baseline / 2.2) / 0.7) ** rng.unit()
            duration = 0.04 + 0.16 * rng.unit()
            if rng.unit() < 0.15:
                tc = float(t.min()) + 0.06 * rng.unit() - 0.03
            else:
                tc = 1325.0 + period * rng.unit()
            snr = snr_low + (20.0 - snr_low) * rng.unit() if rng.unit() < 0.75 else 20.0 + 80.0 * rng.unit()
            n_in = sum(in_box(ti, period, tc, duration) for ti in t.tolist())
            if n_in > 0:
                break
        depth = snr * noise / math.sqrt(3.0 * n_in)
        t, y, e = survey(seed, segments, cadence, noise, keep, period, tc, duration, depth)
        grid = ofir_grid(baseline, 1.0, 1.0, 0.6, None, 3.0, 2)
        r = BoxLeastSquares(t, y, e).power(grid, RECOVERY_DURATIONS, objective="snr",
                                           oversample=10)
        b = int(np.argmax(r.power))
        rows.append(f"{c},{k},{seed},{period!r},{tc!r},{duration!r},{depth!r},{snr!r},{n_in},"
                    f"{len(grid)},{float(r.period[b])!r},{float(r.transit_time[b])!r},"
                    f"{float(r.depth[b])!r},{float(r.duration[b])!r},{float(r.power[b])!r},"
                    f"{samples(t, y, e)}")
        if (c // len(families)) % tls_every == 0:
            tp, tt0, tdep, tdur, sde = run_tls(t, y, e, baseline)
            tls_rows.append(f"{c},{tp!r},{tt0!r},{tdep!r},{tdur!r},{sde!r}")
        if c % 50 == 0:
            print(f"  {c} {name}: {len(t)} points, {len(grid)} periods", file=sys.stderr)
    header = (f"# Injection-recovery on survey light curves; astropy fast, objective snr, "
              f"oversample 10, durations {RECOVERY_DURATIONS}, over the Ofir grid (Sun-like "
              f"star, oversampling 3, 0.6 d to half the baseline) ({VERSIONS}).")
    write_lines(OUT / f"{stem}_families.csv",
                [header, "family,name,segments,cadence,noise,keep"] + fam_rows)
    write_lines(OUT / f"{stem}_cases.csv",
                [header, "case,family,seed,period,tc,duration,depth,snr,n_in,grid,"
                 "astropy_period,astropy_transit_time,astropy_depth,astropy_duration,"
                 "astropy_snr,samples"] + rows)
    write_lines(OUT / f"{stem}_tls.csv",
                [f"# transitleastsquares on the cases whose group of {len(families)} is a "
                 f"multiple of {tls_every} ({tls_share}, each family equally), the same Ofir grid "
                 f"constants and range ({TLS_VERSIONS}). depth: 1 - its bottom flux.",
                 "case,period,t0,depth,duration,sde"] + tls_rows)
    print(f"wrote {len(rows)} injections, {len(tls_rows)} with TLS", file=sys.stderr)


# (name, sectors, noise, offset, period, tc, duration, depth, searched periods):
# TESS 2-minute sectors, back to back, flux near 1 or in counts.
PRECISION_CURVES = [
    ("tess_13_sectors_flux", 13, 1e-3, 1.0, 3.0001, 1326.3, 0.1, 1e-4,
     [0.75, 3.0001, 7.5, 13.0]),
    ("tess_56_sectors_flux", 56, 1e-3, 1.0, 37.1, 1330.3, 0.2, 1e-4,
     [0.75, 3.0, 37.1, 150.0, 400.0]),
    ("tess_56_sectors_counts", 56, 30.0, 1e5, 37.1, 1330.3, 0.2, 3.0,
     [0.75, 3.0, 37.1, 150.0, 400.0]),
]
# Bins of 1/1024 d; every searched period is a whole number of them.
PRECISION_DURATIONS = [0.0625, 0.125, 0.25]
PRECISION_OVERSAMPLE = 64


def replay_fast(t, y, ivar, period, durations, oversample):
    """astropy's bls.c (sequential: no OpenMP on Windows) replayed in numpy
    floats, operation for operation. Returns the best snr and depth as it
    computes them, and its box: start n (it holds bins n + 1 .. n + k,
    1-based), length k, real bins m, and each point's bin."""
    b = min(durations) / oversample
    yc = y - np.median(y)
    min_t = t.min()
    sum_y = np.cumsum(yc * ivar)[-1]
    sum_ivar = np.cumsum(ivar)[-1]
    n_bins = int(math.ceil(period / b)) + oversample
    x = t - min_t
    ind = ((x - period * np.floor(x / period)) / b).astype(np.int64) + 1
    my, mi = np.zeros(n_bins + 1), np.zeros(n_bins + 1)
    np.add.at(my, ind, yc * ivar)
    np.add.at(mi, ind, ivar)
    for n, i in zip(range(1, oversample + 1), range(n_bins - oversample, n_bins)):
        my[i], mi[i] = my[n], mi[n]
    my, mi = np.cumsum(my), np.cumsum(mi)
    eps = np.finfo(float).eps
    best = (-math.inf, None)
    for d in durations:
        k = int(round(d / b))
        n = np.arange(0, n_bins - k + 1)
        yin, iin = my[n + k] - my[n], mi[n + k] - mi[n]
        yout, iout = sum_y - yin, sum_ivar - iin
        with np.errstate(divide="ignore", invalid="ignore"):
            yi, yo = yin / iin, yout / iout
            depth = yo - yi
            snr = depth / np.sqrt(1.0 / iin + 1.0 / iout)
        for j in np.nonzero((iin >= eps) & (iout >= eps) & (yo >= yi))[0]:
            if snr[j] > best[0]:
                best = (snr[j], (int(n[j]), k, depth[j]))
    snr, (start, k, depth) = best
    return snr, depth, start, k, n_bins - oversample, ind


def exact_stats(y, w, inside):
    """The box statistics of the points `inside`, in exact rationals from the
    float y and w, correctly rounded: depth, snr, delta_chi2, flux in, out."""
    def total(xs):
        return sum(map(Fraction, xs), Fraction(0))

    def weighted(ws, ys):
        return sum((Fraction(a) * Fraction(b) for a, b in zip(ws, ys)), Fraction(0))

    w_in, w_out = total(w[inside].tolist()), total(w[~inside].tolist())
    s_in = weighted(w[inside].tolist(), y[inside].tolist())
    s_out = weighted(w[~inside].tolist(), y[~inside].tolist())
    depth = s_out / w_out - s_in / w_in
    snr2 = depth * depth / (1 / w_in + 1 / w_out)
    # sqrt to 200 bits, then rounded once.
    snr = Fraction(math.isqrt(snr2.numerator * 4 ** 200 // snr2.denominator), 2 ** 200)
    snr = snr if depth >= 0 else -snr
    delta_chi2 = w_in * w_out / (w_in + w_out) * depth * depth
    return [float(v) for v in (depth, snr, delta_chi2, s_in / w_in, s_out / w_out)]


def write_precision():
    """astropy fast's best box at sampled periods of long light curves, with
    the box's statistics in exact arithmetic: its rounding error, and the
    exact values the crate's are checked against."""
    c2 = 2.0 / 1440.0
    b = min(PRECISION_DURATIONS) / PRECISION_OVERSAMPLE
    assert b == 1 / 1024
    curves, rows = [], []
    for c, (name, sectors, noise, offset, period, tc, duration, depth, searched) in \
            enumerate(PRECISION_CURVES):
        seed = 9500 + c
        segments = [s for k in range(sectors) for s in sector(1325.0 + 27.5 * k, c2, 9000)]
        t, y, e = survey(seed, segments, c2, noise, 1.0, period, tc, duration, depth)
        y = y + offset
        ivar = 1.0 / e ** 2
        curves.append(f"{c},{name},{seed},{sectors},{noise!r},{offset!r},{period!r},{tc!r},"
                      f"{duration!r},{depth!r},{len(t)},{samples(t, y, e)}")
        bls = BoxLeastSquares(t, y, e)
        for p in searched:
            p = round(p / b) * b
            r = bls.power([p], PRECISION_DURATIONS, objective="snr",
                          oversample=PRECISION_OVERSAMPLE)
            snr, dep, start, k, m, ind = replay_fast(t, y, ivar, p, PRECISION_DURATIONS,
                                                     PRECISION_OVERSAMPLE)
            assert snr == r.power[0] and dep == r.depth[0], "replay differs from astropy"
            assert start + k <= m - 1, "box through the padded bins"
            inside = (ind >= start + 1) & (ind <= start + k)
            exact = exact_stats(y, ivar, inside)
            rows.append(f"{c},{p!r},{float(r.power[0])!r},{float(r.depth[0])!r},{start},{k},{m},"
                        f"{int(inside.sum())}," + ",".join(repr(v) for v in exact))
            print(f"  {name} P {p}: astropy snr error {abs(r.power[0] / exact[1] - 1):.1e}",
                  file=sys.stderr)
    header = (f"# astropy BoxLeastSquares.power, method fast, objective snr, oversample "
              f"{PRECISION_OVERSAMPLE}, durations {PRECISION_DURATIONS}, replayed bit for bit to "
              f"find its box; that box's statistics in exact rationals, correctly rounded "
              f"({VERSIONS}).")
    write_lines(OUT / "precision_curves.csv",
                [header, "curve,name,seed,sectors,noise,offset,period,tc,duration,depth,points,"
                 "samples"] + curves)
    write_lines(OUT / "precision_boxes.csv",
                [header + " start: its box holds bins start .. start + bins - 1 (0-based) of "
                 "m round the orbit.",
                 "curve,period,astropy_snr,astropy_depth,start,bins,m,n_in,depth,snr,delta_chi2,"
                 "flux_in,flux_out"] + rows)
    print(f"wrote {len(rows)} boxes on {len(curves)} curves", file=sys.stderr)


# ---------------------------------------------------------------------------
# Limb-darkened injection-recovery at a fixed false-alarm rate.
# Transits from the Mandel & Agol model (batman; quadratic limb darkening),
# a Sun-like star, b uniform in [0, 1), into the 8.2 families' noise; as many
# noise-only curves, to set each method's SDE threshold at a 1% false-alarm
# rate. astropy fast (as in 8.2) on every curve, and transitleastsquares on
# every `LD_TLS_EVERY`th group.

LD_U = (0.4, 0.26)
LD_COUNT = 1000
LD_TLS_EVERY = 4


def ld_a_rs(period):
    """a / R for a Sun-like star (Kepler's third law), the IAU 2015 values."""
    return np.cbrt(GM_SUN * (period * DAY) ** 2 / (4.0 * np.pi * np.pi)) / R_SUN


def ld_model(t, period, tc, k, a_rs, b):
    """The flux drop (model - 1) at times t."""
    import batman
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = tc, period, k, a_rs
    params.inc = math.degrees(math.acos(b / a_rs))
    params.ecc, params.w = 0.0, 90.0
    params.u, params.limb_dark = list(LD_U), "quadratic"
    return batman.TransitModel(params, t).light_curve(params) - 1.0


def sde(power):
    """Signal detection efficiency: the peak's height above the mean, in
    standard deviations of the spectrum."""
    p = np.asarray(power, dtype=float)
    p = p[np.isfinite(p)]
    return float((p.max() - p.mean()) / p.std())


def write_limb_darkened():
    rng = SplitMix64(9500)
    families = recovery_families(rng)
    fam_rows = [f"{k},{name},{'|'.join(f'{a!r}:{n}' for a, n in segs)},{cad!r},{noise!r},{keep!r}"
                for k, (name, segs, cad, noise, keep) in enumerate(families)]
    rows, tls_rows = [], []
    for c in range(2 * LD_COUNT):
        k_fam = c % len(families)
        injected = (c // len(families)) % 2 == 0
        name, segments, cadence, noise, keep = families[k_fam]
        seed = 29200 + c
        t, y, e = survey(seed, segments, cadence, noise, keep, 1.0, 0.0, 0.0, 0.0)
        baseline = float(t.max() - t.min())
        period = tc = k = a_rs = b = snr = 0.0
        if injected:
            while True:
                period = 0.7 * ((baseline / 2.2) / 0.7) ** rng.unit()
                tc = 1325.0 + period * rng.unit()
                b = rng.unit()
                target = 5.0 + 15.0 * rng.unit() if rng.unit() < 0.75 else 20.0 + 80.0 * rng.unit()
                a_rs = float(ld_a_rs(period))
                k = 0.05
                for _ in range(3):
                    drop = ld_model(t, period, tc, k, a_rs, b)
                    snr = math.sqrt(float(np.sum(drop * drop))) / (noise / math.sqrt(3.0))
                    if snr == 0.0:
                        break
                    k *= math.sqrt(target / snr)
                if snr > 0.0 and k < 0.5:
                    break
            drop = ld_model(t, period, tc, k, a_rs, b)
            snr = math.sqrt(float(np.sum(drop * drop))) / (noise / math.sqrt(3.0))
            y = y + drop
        grid = ofir_grid(baseline, 1.0, 1.0, 0.6, None, 3.0, 2)
        r = BoxLeastSquares(t, y, e).power(grid, RECOVERY_DURATIONS, objective="snr",
                                           oversample=10)
        best = int(np.argmax(r.power))
        rows.append(f"{c},{k_fam},{seed},{int(injected)},{period!r},{tc!r},{k!r},{a_rs!r},{b!r},"
                    f"{snr!r},{len(grid)},{sde(r.power)!r},{float(r.period[best])!r},"
                    f"{float(r.transit_time[best])!r},{float(r.depth[best])!r},"
                    f"{samples(t, y, e)}")
        if (c // (2 * len(families))) % LD_TLS_EVERY == 0:
            tp, tt0, tdep, tdur, tsde = run_tls(t, y, e, baseline, os.cpu_count())
            tls_rows.append(f"{c},{tp!r},{tt0!r},{tdep!r},{tdur!r},{tsde!r}")
        if c % 50 == 0:
            print(f"  {c} {name}: {len(t)} points, {len(grid)} periods", file=sys.stderr)
    header = (f"# Limb-darkened injection-recovery: batman quadratic u = {LD_U}, Sun-like "
              f"star, b uniform in [0, 1), into the 8.2 families' noise; every other group "
              f"of {len(families)} noise only. astropy fast, objective snr, oversample 10, "
              f"durations {RECOVERY_DURATIONS}, over the Ofir grid (oversampling 3, 0.6 d to "
              f"half the baseline); sde = (max - mean) / std of its power ({VERSIONS}).")
    write_lines(OUT / "limb_families.csv",
                [header, "family,name,segments,cadence,noise,keep"] + fam_rows)
    write_lines(OUT / "limb_cases.csv",
                [header, "case,family,seed,injected,period,tc,k,a_rs,b,snr,grid,astropy_sde,"
                 "astropy_period,astropy_transit_time,astropy_depth,samples"] + rows)
    write_lines(OUT / "limb_tls.csv",
                [f"# transitleastsquares on every {LD_TLS_EVERY}th pair of groups, the same "
                 f"Ofir grid constants and range ({TLS_VERSIONS}). depth: 1 - its bottom flux.",
                 "case,period,t0,depth,duration,sde"] + tls_rows)
    print(f"wrote {len(rows)} curves, {len(tls_rows)} with TLS", file=sys.stderr)


def main():
    example_2()
    example_3()
    print(f"worked examples 2 and 3 reproduced ({VERSIONS})", file=sys.stderr)
    OUT.mkdir(parents=True, exist_ok=True)
    write_autoperiod()
    write_ofir()
    write_in_transit()
    write_exact()
    write_slow()
    write_fast()
    write_wrap()
    write_injected()
    write_survey()
    write_recovery()
    write_recovery_wide()
    write_precision()
    write_limb_darkened()


if __name__ == "__main__":
    main()
