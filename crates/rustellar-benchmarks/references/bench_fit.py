"""batman and PyTransit with scipy least_squares: fit timings for
rustellar-benchmarks's speed table. The same fits as
make_fit_reference.py (its methods and options), on the light curves it
wrote to crates/rustellar-fit/tests/reference: the best of seven repeats,
each the mean of a loop of fits lasting about 0.2 s.

Writes crates/rustellar-benchmarks/data/speed_fit.csv, cases
"<case> <method>" (method: lm 2-point, trf 2-point, trf 3-point, tight,
PyTransit), package "batman 2.5.1" or "PyTransit 2.6.18".

Run: python crates/rustellar-benchmarks/references/bench_fit.py
"""

import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import batman  # noqa: E402
import numpy as np  # noqa: E402
import pytransit  # noqa: E402

import make_fit_reference as ref  # noqa: E402

ROOT = os.path.join(HERE, "..", "..", "..")
DATA = os.path.join(ROOT, "crates", "rustellar-fit", "tests", "reference")
OUT = os.path.join(ROOT, "crates", "rustellar-benchmarks", "data", "speed_fit.csv")

SHORT = {
    "batman lm 2-point": "lm 2-point",
    "batman trf 2-point": "trf 2-point",
    "batman trf 3-point": "trf 3-point",
    "batman trf 3-point tight": "tight",
    "PyTransit trf 3-point": "PyTransit",
}


def cases():
    points = {}
    with open(os.path.join(DATA, "fit_points.csv")) as fp:
        for row in csv.DictReader(fp):
            points.setdefault(row["case"], ([], []))
            points[row["case"]][0].append(float(row["time"]))
            points[row["case"]][1].append(float(row["flux"]))
    out = []
    with open(os.path.join(DATA, "fit_cases.csv")) as fp:
        for row in csv.DictReader(fp):
            samples = int(row["samples"])
            exposure = (float(row["exposure_days"]), samples) if samples > 1 else None
            fixed = [f for f in row["fixed"].split(";") if f]
            start = [float(row["start_" + n]) for n in ref.NAMES]
            t, flux = points[row["case"]]
            out.append((row["case"], np.array(t), np.array(flux), float(row["sigma"]),
                        start, exposure, fixed))
    return out


def best(fn):
    began = time.perf_counter()
    fn()
    once = time.perf_counter() - began
    number = max(1, int(0.2 / max(once, 1e-6)))
    times = []
    for _ in range(7):
        began = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - began) / number)
    return min(times)


def main():
    packages = {
        "batman": f"batman {batman.__version__}",
        "PyTransit": f"PyTransit {pytransit.__version__}",
    }
    with open(OUT, "w", newline="\n") as fp:
        fp.write("case,package,seconds\n")
        for name, t, flux, sigma, start, exposure, fixed in cases():
            for label, kind, options in ref.METHODS:
                # The model set up once, as a user fitting would.
                model = kind(t, exposure)
                s = best(lambda: ref.fit(model, flux, sigma, start, fixed, options))
                package = packages[label.split(" ")[0]]
                fp.write(f"{name} {SHORT[label]},{package},{s!r}\n")
                print(f"{name:22} {SHORT[label]:12} {s * 1e3:9.2f} ms", flush=True)


if __name__ == "__main__":
    main()
