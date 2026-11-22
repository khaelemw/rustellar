"""statsmodels' autoregressive fit on the 2-minute TESS sector of
rustellar-detrend's benchmark (the survey recipe), for the README:
ar_select_order(maxlag=30, ic="aic", trend="c") and the refit at the order
chosen, as `Autoregressive::default().whiten`. statsmodels has no gaps, so
the sector is one series. Best of 3 after a warm-up call.

Run: python crates/rustellar-benchmarks/references/bench_detrend_ar.py
"""

import os
import sys
import time
import warnings

import numpy as np
from statsmodels.tsa.ar_model import ar_select_order

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_detrend_wotan import curves  # noqa: E402
from make_detrend_survey_reference import light_curve  # noqa: E402


def fit(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chosen = ar_select_order(f, 30, ic="aic", trend="c")
        return chosen.model.fit().resid


def main():
    p = curves()[1]
    t, f, _ = light_curve(p)
    f = np.array(f)
    fit(f)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        fit(f)
        times.append(time.perf_counter() - start)
    print(f"{p['name']} ({len(f)} points): statsmodels {min(times):.5f} s")


if __name__ == "__main__":
    main()
