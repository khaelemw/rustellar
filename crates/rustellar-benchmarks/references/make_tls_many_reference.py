"""TLS 1.32 on many injected transits, for comparing rustellar-tls over many
seeds (recovery, timing, depth, duration). Writes crates/rustellar-benchmarks/data/tls_many.csv.

Each curve: 27 d at 30 min (i / 48 d, i < 1296). Its parameters from
SplitMix64(seed + 1_000_000), in this order: k uniform in [0.015, 0.045),
period in [1, 9), t0 in [0, period), noise in [3e-4, 8e-4), sparse if a
uniform is below 0.3. Then, as in make_tls_injection_reference.py's
`harder` set, noise from SplitMix64(seed): per point the sum of four
uniforms less 2, times sqrt(3), times the noise; if sparse one more
uniform, and the point dropped below 0.25. Transit from batman, a = 12,
b = 0.2, quadratic (0.4804, 0.1867).

Run: python crates/rustellar-benchmarks/references/make_tls_many_reference.py [count] [processes]
"""
import csv
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

from make_tls_injection_reference import SplitMix64, harder

warnings.filterwarnings("ignore")


def params(seed):
    rng = SplitMix64(seed + 1_000_000)
    k = 0.015 + 0.03 * rng.uniform()
    period = 1.0 + 8.0 * rng.uniform()
    t0 = period * rng.uniform()
    noise = 3e-4 + 5e-4 * rng.uniform()
    sparse = rng.uniform() < 0.3
    return k, period, t0, noise, sparse


def one(seed):
    warnings.filterwarnings("ignore")
    from transitleastsquares import transitleastsquares
    k, period, t0, noise, sparse = params(seed)
    t, y = harder(seed, k, period, t0, noise, sparse)
    r = transitleastsquares(t, y, np.full(len(t), noise), verbose=False).power(
        use_threads=1, show_progress_bar=False, verbose=False)
    return [seed, repr(k), repr(period), repr(t0), repr(noise), int(sparse),
            repr(float(r.period)), repr(float(r.T0)), repr(1 - float(r.depth)),
            repr(float(r.duration)), repr(float(r.SDE))]


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "data", "tls_many.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with Pool(processes) as pool:
        rows = pool.map(one, range(1000, 1000 + count))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "k", "period", "t0", "noise", "sparse", "tls_period",
                    "tls_t0", "tls_depth", "tls_duration", "tls_sde"])
        w.writerows(rows)
    print(out, len(rows))
