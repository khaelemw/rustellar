"""LEO-Vetter 1.2.0 and exovetter 0.0.16 on shared simulated light curves,
for rustellar-vet's comparison test.

Every curve is 27 days at 2 minutes (19,440 points), a 3-day period, mid-
transit at 1.5 days, a 0.1-day box transit, and noise of 1e-3 per point,
drawn from SplitMix64 and Box-Muller so the Rust test makes the very same
curves (seed = 1,000,003 * scenario + draw):

- planet (200): a 3e-3 transit.
- binary_odd_even (100): eclipses of 3e-3 and 2.7e-3 in turn (an eclipsing
  binary at twice the period), 10% apart.
- binary_secondary (100): a 3e-3 eclipse, a 1e-3 one at half the period,
  and ellipsoidal variation -2e-4 cos(4 pi phase).
- red_noise (200): the planet in white noise of 1e-3 and red noise of 1e-3
  (AR(1), time scale 0.05 days).
- gaps (20): the planet with three gaps of up to two days at random.

For each curve: LEO-Vetter's box odd-even depths, noise (sig_w, sig_r), depth
and error, ModShift-style secondary, SWEET and transit count; exovetter's
odd-even, ModShift secondary and SWEET at half the period.

Writes crates/rustellar-vet/tests/reference/vet_references.csv.

Run: python crates/rustellar-benchmarks/references/make_vet_reference.py
"""

import io
import math
import multiprocessing
import os
import sys
import contextlib

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "crates",
                   "rustellar-vet", "tests", "reference", "vet_references.csv")

N = 19440
STEP = 2.0 / 1440.0
P, T0, DUR = 3.0, 1.5, 0.1
SIGMA = 1e-3
MASK = (1 << 64) - 1
SCENARIOS = [("planet", 200), ("binary_odd_even", 100), ("binary_secondary", 100),
             ("red_noise", 200), ("gaps", 20)]


class SplitMix:
    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next() >> 11) * (1.0 / 9007199254740992.0)

    def normal(self):
        u1 = 1.0 - self.uniform()
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def curve(scenario, draw):
    """Time and flux, as the Rust test makes them."""
    index = [s for s, _ in SCENARIOS].index(scenario)
    rng = SplitMix(1_000_003 * index + draw)
    time = [i * STEP for i in range(N)]
    flux = []
    red = 0.0
    phi = math.exp(-STEP / 0.05)
    root = math.sqrt(1.0 - phi * phi)
    for i, t in enumerate(time):
        cycles = (t - T0) / P
        epoch = round(cycles)
        dt = (cycles - epoch) * P
        f = 1.0
        if abs(dt) < 0.5 * DUR:
            if scenario == "binary_odd_even" and epoch % 2 != 0:
                f -= 2.7e-3
            else:
                f -= 3e-3
        if scenario == "binary_secondary":
            half = (cycles - 0.5 - round(cycles - 0.5)) * P
            if abs(half) < 0.5 * DUR:
                f -= 1e-3
            f += -2e-4 * math.cos(4.0 * math.pi * cycles)
        f += SIGMA * rng.normal()
        if scenario == "red_noise":
            n = rng.normal()
            red = n if i == 0 else phi * red + root * n
            f += SIGMA * red
        flux.append(f)
    time, flux = np.array(time), np.array(flux)
    if scenario == "gaps":
        keep = np.ones(N, dtype=bool)
        for _ in range(3):
            start = 27.0 * rng.uniform()
            length = 2.0 * rng.uniform()
            keep &= ~((time >= start) & (time < start + length))
        time, flux = time[keep], flux[keep]
    return time, flux


def measure(job):
    scenario, draw = job
    from exovetter import odd_even, sweet
    from exovetter.modshift import modshift as ms
    from leo_vetter import fits, main, modshift, oddeven

    time, flux = curve(scenario, draw)
    err = np.full(time.size, SIGMA)
    row = {"scenario": scenario, "draw": draw}
    with contextlib.redirect_stdout(io.StringIO()):
        tlc = main.TCELightCurve(0, time, flux, flux, err, P, T0, DUR)
        tlc.get_SES_MES()
        oddeven.box(tlc)
        fits.sweet(tlc)
        modshift.uniqueness(tlc)
    m = tlc.metrics
    # LEO's "odd" transits are epochs 0, 2, ...; rustellar-vet calls those
    # even.
    row.update(leo_even=m["odd_dep"], leo_even_err=m["odd_dep_err"],
               leo_odd=m["even_dep"], leo_odd_err=m["even_dep_err"],
               leo_sig_dep=m["sig_dep"], leo_dep=m["dep"], leo_err=m["err"],
               leo_sig_w=m["sig_w"], leo_sig_r=m["sig_r"],
               leo_phs_sec=m["phs_sec"], leo_sig_sec=m["sig_sec"],
               leo_dep_sec=m["dep_sec"], leo_err_sec=m["err_sec"],
               leo_sine_amp=m["sine_amp"], leo_sine_sig=m["sine_sig"],
               leo_n_transit=m["N_transit"])
    sigma, _, _ = odd_even.calc_odd_even(time, flux.copy(), P, T0, DUR)
    sw = sweet.sweet(time, flux.copy() - 1.0, P, T0, DUR)
    box = -3e-3 * (np.abs(((time - T0) / P - np.round((time - T0) / P)) * P) < 0.5 * DUR)
    res, _ = ms.compute_modshift_metrics(time, flux - 1.0, box, P, T0, DUR * 24.0,
                                         show_plot=False)
    row.update(exo_oe_sigma=sigma, exo_sweet_half_amp=sw[0][0], exo_sweet_half_snr=sw[0][2],
               exo_ms_phase_sec=((res["phase_sec"] - res["phase_pri"]) / P) % 1.0,
               exo_ms_sigma_sec=res["sigma_sec"])
    return row


def main():
    jobs = [(s, d) for s, n in SCENARIOS for d in range(n)]
    with multiprocessing.Pool() as pool:
        rows = []
        for i, row in enumerate(pool.imap(measure, jobs, chunksize=4)):
            rows.append(row)
            if i % 50 == 0:
                print(i, "of", len(jobs), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    keys = list(rows[0].keys())
    with open(OUT, "w", newline="\n") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(v if isinstance(v, str) else repr(float(v)) if not isinstance(v, (int, np.integer)) else str(v)
                             for v in (row[k] for k in keys)) + "\n")
    print("wrote", len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
