"""Reference values for rustellar-mcmc's diagnostics.

emcee 3.1.6 `integrated_time` and ArviZ 1.0 (arviz_stats) rank-normalised
split-R-hat and bulk and tail ESS, on chains rebuilt in the Rust tests
from the same SplitMix64 stream (only +, *, / and floor, so bit-identical),
and scipy's `norm.ppf` for the inverse normal.

Run from the repository root:
    python crates/rustellar-benchmarks/references/make_mcmc_diagnostics_reference.py
"""
import math
import os

import numpy as np
import emcee
import arviz_stats
from arviz_stats.base import array_stats
from scipy.stats import norm

OUT = "crates/rustellar-mcmc/tests/reference"
MASK = (1 << 64) - 1


class SplitMix:
    def __init__(self, seed):
        self.s = seed & MASK

    def unit(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
        return (z >> 11) * 2.0**-53


def chains(kind, n_chain, n_draw, seed, phi, offset):
    out = []
    for k in range(n_chain):
        g = SplitMix(seed * 1000 + k)
        row = []
        y = 0.0
        for t in range(n_draw):
            u = g.unit()
            if kind == "ar":
                y = phi * y + (u - 0.5) if t > 0 else u - 0.5
                row.append(y + offset * k)
            elif kind == "ties":
                row.append(float(math.floor(5.0 * u)))
            elif kind == "heavy":
                row.append(1.0 / u)
            elif kind == "constant":
                row.append(1.0)
            elif kind == "trend":
                row.append(t / n_draw + 0.01 * (u - 0.5))
        out.append(row)
    return np.array(out)


CASES = [
    # name, kind, chains, draws, seed, phi, offset
    ("independent", "ar", 4, 1000, 1, 0.0, 0.0),
    ("ar_0.5", "ar", 4, 1000, 2, 0.5, 0.0),
    ("ar_0.9", "ar", 8, 2000, 3, 0.9, 0.0),
    ("ar_0.99", "ar", 4, 3000, 4, 0.99, 0.0),
    ("negative", "ar", 4, 777, 5, -0.6, 0.0),
    ("offset", "ar", 4, 400, 6, 0.3, 0.5),
    ("ties_odd", "ties", 4, 301, 7, 0.0, 0.0),
    ("one_chain", "ar", 1, 999, 8, 0.7, 0.0),
    ("heavy", "heavy", 6, 500, 9, 0.0, 0.0),
    ("trend", "trend", 4, 600, 10, 0.0, 0.0),
    ("two_chains_short", "ar", 2, 5, 11, 0.2, 0.0),
    ("three_draws", "ar", 4, 3, 12, 0.2, 0.0),
    ("constant", "constant", 4, 100, 13, 0.0, 0.0),
    ("too_short", "ar", 4, 40, 14, 0.999, 0.0),
]


def fmt(v):
    return repr(float(v))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = ["case,kind,chains,draws,seed,phi,offset,tau,window,rhat,ess_bulk,ess_tail"]
    for name, kind, nc, nd, seed, phi, off in CASES:
        x = chains(kind, nc, nd, seed, phi, off)
        # emcee: (steps, walkers, dims); its window is not returned, so
        # recompute it the same way.
        with np.errstate(all="ignore"):
            f = np.zeros(nd)
            for k in range(nc):
                f += emcee.autocorr.function_1d(x[k])
            f /= nc
            taus = 2.0 * np.cumsum(f) - 1.0
            window = emcee.autocorr.auto_window(taus, 5)
            tau = emcee.autocorr.integrated_time(x.T[:, :, None], quiet=True)[0]
            assert tau == taus[window] or (np.isnan(tau) and np.isnan(taus[window]))
        rhat = array_stats.rhat(x, method="rank", chain_axis=0, draw_axis=1)
        bulk = array_stats.ess(x, method="bulk", chain_axis=0, draw_axis=1)
        tail = array_stats.ess(x, method="tail", chain_axis=0, draw_axis=1, prob=(0.05, 0.95))
        rows.append(",".join([name, kind, str(nc), str(nd), str(seed), fmt(phi), fmt(off),
                              fmt(tau), str(int(window)), fmt(rhat), fmt(bulk), fmt(tail)]))
        print(rows[-1])
    with open(f"{OUT}/diagnostics.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")

    ps = [1e-300, 1e-100, 1e-20, 1e-10, 1e-5, 0.001, 0.02, 0.075, 0.0749, 0.075001,
          0.2, 0.37, 0.5, 0.5 + 1e-12, 0.63, 0.8, 0.924999, 0.925, 0.95, 0.999,
          1 - 1e-10, 1 - 2**-53]
    g = SplitMix(99)
    ps += [g.unit() for _ in range(200)]
    ps += [math.exp(-30.0 * g.unit()) for _ in range(100)]
    lines = ["p,quantile"] + [f"{fmt(p)},{fmt(norm.ppf(p))}" for p in ps]
    with open(f"{OUT}/normal_quantile.csv", "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("versions", emcee.__version__, arviz_stats.__version__)


if __name__ == "__main__":
    main()
