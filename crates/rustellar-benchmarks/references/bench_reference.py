"""Time batman, PyTransit and exoplanet-core on the same planet and times as
crates/rustellar-model/benches/transit.rs, single-threaded.

    NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 python crates/rustellar-benchmarks/references/bench_reference.py [CSV]

With CSV (crates/rustellar-benchmarks/data/speed_model.csv), also writes
each time as case,package,seconds for the benchmarks crate's speed table.
"""

import math
import os
import platform
import sys
import timeit

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import batman
import numpy as np
import pytransit
from pytransit import QuadraticModel
import exoplanet_core

T0, P, K, A, B, U = 0.0, 3.0, 0.1, 10.0, 0.3, (0.4, 0.25)
INC = math.acos(B / A)
LONG_CADENCE = 29.4 / 60 / 24


def times(n):
    return np.linspace(-0.1, 0.1, n)


def best(stmt, number):
    return min(timeit.repeat(stmt, number=number, repeat=7)) / number


def batman_model(t, samples=1, exposure=0.0):
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, P, K, A, math.degrees(INC)
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, list(U), "quadratic"
    if samples > 1:
        m = batman.TransitModel(p, t, supersample_factor=samples, exp_time=exposure, nthreads=1)
    else:
        m = batman.TransitModel(p, t, nthreads=1)
    return lambda: m.light_curve(p)


def pytransit_model(t, samples=1, exposure=0.0):
    m = QuadraticModel(interpolate=False)
    if samples > 1:
        m.set_data(t, nsamples=[samples], exptimes=[exposure])
    else:
        m.set_data(t)
    ldc = np.array(U)
    f = lambda: m.evaluate(K, ldc, T0, P, A, INC)
    f()  # compile
    return f


def separation(t):
    phi = 2 * math.pi * (t - T0) / P
    return np.hypot(A * np.sin(phi), B * np.cos(phi))


def us(seconds):
    return f"{seconds * 1e6:10.1f} us"


def main():
    rows = []
    bat_name = f"batman {batman.__version__}"
    pyt_name = f"PyTransit {pytransit.__version__}"
    exo_name = f"exoplanet-core {exoplanet_core.__version__}"
    print(f"batman {batman.__version__}, PyTransit {pytransit.__version__}, "
          f"exoplanet-core {exoplanet_core.__version__}, numpy {np.__version__}, "
          f"Python {platform.python_version()}, {platform.processor()}")
    print("\nlight curve (times over +-0.1 d)")
    for n in [100, 1_000, 10_000, 100_000]:
        t = times(n)
        number = max(1, 100_000 // n)
        z = separation(t)
        u1, u2 = U
        bat = best(batman_model(t), number)
        pyt = best(pytransit_model(t), number)
        exo = best(lambda: exoplanet_core.quad_limbdark_light_curve(u1, u2, z, K), number)
        rows += [(f"curve {n}", bat_name, bat), (f"curve {n}", pyt_name, pyt),
                 (f"flux {n}", exo_name, exo)]
        print(f"  n = {n:>6}: batman {us(bat)}  PyTransit {us(pyt)}  exoplanet-core (flux only) {us(exo)}")
    print("\nKepler long cadence, 1000 times")
    t = times(1_000)
    for samples in [5, 15, 30]:
        bat = best(batman_model(t, samples, LONG_CADENCE), 20)
        pyt = best(pytransit_model(t, samples, LONG_CADENCE), 20)
        rows += [(f"exposure {samples}", bat_name, bat),
                 (f"exposure {samples}", pyt_name, pyt)]
        print(f"  {samples:>2} samples: batman {us(bat)}  PyTransit {us(pyt)}")

    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", newline="\n") as f:
            f.write("case,package,seconds\n")
            for case, name, sec in rows:
                f.write(f"{case},{name},{sec!r}\n")


if __name__ == "__main__":
    main()
