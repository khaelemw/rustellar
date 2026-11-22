"""astropy 7.2 LombScargle timings for rustellar-benchmarks's speed table: slow, cython and fast, floating mean, standard
normalisation, the best of three calls (one where points times frequencies
pass 1e11), on the same light curves as `speed::ls::curve`: n points over
30 days, each in its own 30 / n-day slot at the golden ratio's fraction, a
2.7-day sinusoid and even noise from the plastic number's multiples,
errors 1e-3; frequencies 0.01 + 0.005 k. slow and cython only up to 2e8
point-frequencies.

Writes crates/rustellar-benchmarks/data/speed_ls.csv.

Run: python crates/rustellar-benchmarks/references/bench_ls.py
"""

import os
import time

import astropy
import numpy as np
from astropy.timeseries import LombScargle

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT = os.path.join(ROOT, "crates", "rustellar-benchmarks", "data", "speed_ls.csv")
SIZES = [(1000, 10000), (10000, 10000), (10000, 100000), (100000, 100000), (1000000, 1000000)]


def curve(n):
    i = np.arange(n, dtype=float)
    slot = 30.0 / n
    t = slot * (i + ((i + 1) * 0.6180339887498949) % 1.0)
    noise = 3.464 * (((i + 1) * 0.7548776662466927) % 1.0 - 0.5)
    y = 1.0 + 1e-3 * np.sin(2 * np.pi * t / 2.7) + 1e-3 * noise
    return t, y, np.full(n, 1e-3)


def best(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    package = f"astropy {astropy.__version__}"
    with open(OUT, "w", newline="\n") as fp:
        fp.write("case,package,seconds\n")
        for n, nf in SIZES:
            t, y, dy = curve(n)
            freq = 0.01 + 0.005 * np.arange(nf)
            ls = LombScargle(t, y, dy)
            for method in ["slow", "cython", "fast"]:
                if method != "fast" and n * nf > 2e8:
                    continue
                repeat = 3 if n * nf < 1e11 else 1
                s = best(lambda: ls.power(freq, method=method, assume_regular_frequency=True), repeat)
                fp.write(f"{method} {n} x {nf},{package},{s!r}\n")
                print(method, n, nf, f"{s * 1e3:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
