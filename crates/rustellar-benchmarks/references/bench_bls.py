"""Time astropy's BoxLeastSquares (fast method, objective snr) on the shared
survey recipe, the same data and grids as the rustellar-bls timing run.

    python crates/rustellar-benchmarks/references/bench_bls.py [CSV]

With CSV (crates/rustellar-benchmarks/data/speed_bls.csv), also writes each
time as case,package,seconds (commas dropped from the case names).
"""

import importlib.util
import platform
import sys
import time
from pathlib import Path

import astropy
import numpy as np
from astropy.timeseries import BoxLeastSquares

spec = importlib.util.spec_from_file_location(
    "gen", Path(__file__).resolve().parent / "make_bls_reference.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

CADENCE = 2.0 / 1440.0
DURATIONS = [0.01 + 0.34 * k / 11.0 for k in range(12)]


def light_curve():
    """A TESS 2-minute sector, 18,000 points, a transit every 4.7 days."""
    return gen.survey(77, gen.sector(1325.0, CADENCE, 9000), CADENCE, 1e-3, 1.0,
                      4.7, 1326.3, 0.12, 1e-3)


def sectors(seed, count):
    """`count` sectors 27.5 days apart, as benches/search.rs builds them."""
    segments = [s for k in range(count) for s in gen.sector(1325.0 + 27.5 * k, CADENCE, 9000)]
    return gen.survey(seed, segments, CADENCE, 1e-3, 1.0, 4.7, 1326.3, 0.12, 1e-3)


def kepler():
    """Four years of 29.4-minute points: 16 quarters of 89 days, 5% lost."""
    segments = [(1325.0 + 90.0 * q, 4360) for q in range(16)]
    return gen.survey(79, segments, 0.0204, 3e-4, 0.95, 37.1, 1340.2, 0.25, 2e-4)


def ofir(t, longest=None):
    """Ofir's grid, a Sun-like star, oversampling 3, 0.6 d to `longest` or
    half the baseline."""
    return list(gen.ofir_grid(float(t.max() - t.min()), 1.0, 1.0, 0.6, longest, 3.0, 2))


def cases():
    one = light_curve()
    thirteen = sectors(78, 13)
    kep = kepler()
    log = [0.5 * 30.0 ** (k / 9999.0) for k in range(10000)]
    return [
        ("1 sector, 10,000 periods", one, log),
        ("1 sector, Ofir", one, ofir(one[0])),
        ("13 sectors, Ofir", thirteen, ofir(thirteen[0])),
        ("Kepler 4 years, Ofir to 100 d", kep, ofir(kep[0], 100.0)),
    ]


def grids():
    return {
        "2,000 periods, 1 to 10 d": [1.0 + 9.0 * k / 2000.0 for k in range(2000)],
        "10,000 periods, 0.5 to 15 d": [0.5 * 30.0 ** (k / 9999.0) for k in range(10000)],
    }


def main():
    print(f"astropy {astropy.__version__}, numpy {np.__version__}, Python "
          f"{platform.python_version()}, {platform.processor()}; 12 durations, oversample 10")
    rows = []
    for name, (t, y, e), periods in cases():
        bls = BoxLeastSquares(t, y, e)
        runs = []
        for _ in range(1 if len(t) * len(periods) > 1e9 else 3):
            start = time.perf_counter()
            r = bls.power(periods, DURATIONS, objective="snr", oversample=10)
            runs.append(time.perf_counter() - start)
        best = min(runs)
        rows.append((name.replace(",", ""), f"astropy {astropy.__version__}", best))
        k = int(np.argmax(r.power))
        print(f"  {name} ({len(t)} points, {len(periods)} periods): {best:.3f} s, "
              f"{best / len(periods) * 1e3:.3f} ms a period; "
              f"best P {r.period[k]:.5f}, depth {r.depth[k]:.3e}, duration {r.duration[k]:.3f}")
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", newline="\n") as out:
            out.write("case,package,seconds\n")
            for case, package, sec in rows:
                out.write(f"{case},{package},{sec!r}\n")


if __name__ == "__main__":
    main()
