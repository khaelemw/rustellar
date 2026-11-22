"""Time transitleastsquares 1.32 on the light curves that
crates/rustellar-tls/benches/search.rs times: the shared survey recipe
(make_bls_reference.survey, the same seeds) around 1, TLS's defaults.

    python crates/rustellar-benchmarks/references/bench_tls.py            all cases, every core
    python crates/rustellar-benchmarks/references/bench_tls.py --trial    the 30-minute case once, 4 threads
    python crates/rustellar-benchmarks/references/bench_tls.py CSV        all, and each time written to CSV

CSV is crates/rustellar-benchmarks/data/speed_tls.csv: case,package,seconds
(commas dropped from the case names), for the benchmarks crate's table.
"""

import importlib.metadata
import importlib.util
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import transitleastsquares
from transitleastsquares import transitleastsquares as Tls

warnings.filterwarnings("ignore")

spec = importlib.util.spec_from_file_location(
    "gen", Path(__file__).resolve().parent / "make_bls_reference.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

TWO = 2.0 / 1440.0
THIRTY = 1.0 / 48.0
NL = chr(10)


def sectors(seed, count, cadence, n):
    """As benches/search.rs: `count` sectors 27.5 days apart, flux around 1."""
    segments = [s for k in range(count) for s in gen.sector(1325.0 + 27.5 * k, cadence, n)]
    t, y, e = gen.survey(seed, segments, cadence, 1e-3, 1.0, 4.7, 1326.3, 0.12, 1e-3)
    return np.asarray(t), 1.0 + np.asarray(y), np.asarray(e)


def cases():
    return [
        ("1 sector, 30 min", lambda: sectors(80, 1, THIRTY, 620), 3),
        ("1 sector, 2 min", lambda: sectors(77, 1, TWO, 9000), 1),
        ("2 sectors, 2 min", lambda: sectors(78, 2, TWO, 9000), 1),
    ]


def main():
    trial = "--trial" in sys.argv
    threads = 4 if trial else os.cpu_count()
    print(f"transitleastsquares {transitleastsquares.tls_constants.TLS_VERSION}, "
          f"numpy {np.__version__}, Python {platform.python_version()}, "
          f"{platform.processor()}; {threads} threads")
    rows = []
    package = f"transitleastsquares {importlib.metadata.version('transitleastsquares')}"
    for name, make, repeats in cases()[:1] if trial else cases():
        t, y, e = make()
        runs = []
        for _ in range(1 if trial else repeats):
            start = time.perf_counter()
            r = Tls(t, y, e, verbose=False).power(
                use_threads=threads, show_progress_bar=False, verbose=False)
            runs.append(time.perf_counter() - start)
        best = min(runs)
        rows.append((name.replace(",", ""), package, best))
        rows.append(("periods " + name.replace(",", ""), package, float(len(r.periods))))
        print(f"  {name} ({len(t)} points, {len(r.periods)} periods): {best:.2f} s; "
              f"best P {r.period:.5f}, T14 {r.duration:.4f}, depth {1 - r.depth:.3e}, "
              f"SDE {r.SDE:.1f}", flush=True)
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if paths:
        with open(paths[0], "w", newline=NL) as out:
            out.write("case,package,seconds" + NL)
            for case, name, sec in rows:
                out.write(f"{case},{name},{sec!r}" + NL)


if __name__ == "__main__":
    main()
