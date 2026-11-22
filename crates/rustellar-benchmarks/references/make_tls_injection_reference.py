"""TLS 1.32 on injected transits, printed as Rust literals for rustellar-tls's
unit tests (tests/search.rs, `against_tls_on_injected_transits`).

The light curves are rebuilt in Rust from the same recipe: times i / 48 d
for i < 1296 (27 days at 30 minutes), less a gap when asked; noise from
SplitMix64 (seeded per curve), the sum of four uniforms less 2 times
sqrt(3), times the noise level; the transit from batman here and from
rustellar-model there (within 2e-8 of each other), a = 12, b = 0.2,
quadratic (0.4804, 0.1867).

A second set (`harder`) is shallower and noisier, some sparse: after a
point's noise, one more uniform, and the point is dropped below 0.25.

Run: python crates/rustellar-benchmarks/references/make_tls_injection_reference.py [harder]
"""
import sys
import warnings

import batman
import numpy as np
from transitleastsquares import transitleastsquares, tls_constants

warnings.filterwarnings("ignore")
MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))


# seed, k, period, t0, gap (days 12.5 to 14.5 dropped)
CASES = [
    (101, 0.04, 3.7, 1.3, False),
    (102, 0.03, 5.9, 4.4, False),
    (103, 0.05, 1.37, 0.2, False),
    (104, 0.025, 2.3, 0.9, False),
    (105, 0.035, 7.1, 6.0, False),
    (106, 0.04, 3.7, 1.3, True),
    (107, 0.03, 4.6, 2.2, True),
    (108, 0.045, 1.9, 0.5, True),
]
NOISE = 4e-4


# seed, k, period, t0, noise, sparse
HARDER = [
    (201, 0.02, 3.3, 1.1, 4e-4, False),
    (202, 0.025, 6.4, 2.0, 6e-4, False),
    (203, 0.03, 2.7, 0.4, 8e-4, False),
    (204, 0.03, 4.1, 3.0, 4e-4, True),
    (205, 0.02, 1.6, 0.7, 3e-4, True),
    (206, 0.04, 8.3, 5.1, 8e-4, True),
]


def model(k, period, t0, t):
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = t0, period, k, 12.0
    p.inc = float(np.degrees(np.arccos(0.2 / 12.0)))
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.4804, 0.1867], "quadratic"
    return batman.TransitModel(p, t).light_curve(p)


def harder(seed, k, period, t0, noise, sparse):
    t = np.arange(1296) / 48.0
    rng = SplitMix64(seed)
    keep_t, keep_f = [], []
    for ti, f in zip(t, model(k, period, t0, t)):
        n = 0.0
        for _ in range(4):
            n += rng.uniform() - 0.5
        if sparse and rng.uniform() < 0.25:
            continue
        keep_t.append(ti)
        keep_f.append(f + noise * (n * np.sqrt(3.0)))
    return np.array(keep_t), np.array(keep_f)


def curve(seed, k, period, t0, gap):
    t = np.arange(1296) / 48.0
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = t0, period, k, 12.0
    p.inc = float(np.degrees(np.arccos(0.2 / 12.0)))
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.4804, 0.1867], "quadratic"
    model = batman.TransitModel(p, t).light_curve(p)
    rng = SplitMix64(seed)
    flux = []
    for f in model:
        n = 0.0
        for _ in range(4):
            n += rng.uniform() - 0.5
        flux.append(f + NOISE * (n * np.sqrt(3.0)))
    flux = np.array(flux)
    if gap:
        keep = ~((t > 12.5) & (t < 14.5))
        t, flux = t[keep], flux[keep]
    return t, flux


def run(t, y, noise):
    return transitleastsquares(t, y, np.full(len(t), noise), verbose=False).power(
        use_threads=1, show_progress_bar=False, verbose=False)


if __name__ == "__main__" and sys.argv[1:] == ["harder"]:
    print("// " + tls_constants.TLS_VERSION + ", one thread, defaults")
    for seed, k, period, t0, noise, sparse in HARDER:
        t, y = harder(seed, k, period, t0, noise, sparse)
        r = run(t, y, noise)
        print(f"    ({seed}, {k!r}, {period!r}, {t0!r}, {noise!r}, {str(sparse).lower()}, "
              f"{float(r.period)!r}, {float(r.T0)!r}, {1 - float(r.depth)!r}, "
              f"{float(r.duration)!r}, {float(r.SDE)!r}),")
elif __name__ == "__main__":
    print("// " + tls_constants.TLS_VERSION + ", one thread, defaults")
    for seed, k, period, t0, gap in CASES:
        t, y = curve(seed, k, period, t0, gap)
        r = transitleastsquares(t, y, np.full(len(t), NOISE), verbose=False).power(
            use_threads=1, show_progress_bar=False, verbose=False)
        print(f"    ({seed}, {k!r}, {period!r}, {t0!r}, {str(gap).lower()}, "
              f"{float(r.period)!r}, {float(r.T0)!r}, {1 - float(r.depth)!r}, "
              f"{float(r.duration)!r}, {float(r.SDE)!r}),")
