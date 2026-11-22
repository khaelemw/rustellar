"""TLS 1.32's spectra and period uncertainty on fixed chi-squared arrays,
printed as Rust literals for rustellar-tls's unit tests (spectra.rs).

Run: python crates/rustellar-benchmarks/references/make_tls_spectra_reference.py
"""
import numpy as np
from transitleastsquares import stats, tls_constants


def chi2(n):
    c = np.array([1000 + ((k * 7919) % 1000) / 10.0 for k in range(n)])
    if n > 150:
        c[149], c[150], c[151] = 850.0, 700.0, 850.0
    else:
        c[n // 2] = 700.0
    return c


print("//", tls_constants.TLS_VERSION)
for n, over in ((400, 3), (100, 3), (400, 1)):
    c = chi2(n)
    sr, power_raw, power, sde_raw, sde = stats.spectra(c, over)
    periods = 1 + 0.01 * np.arange(n)
    unc = stats.period_uncertainty(periods, power)
    peak = int(np.argmax(power))
    print(f"// n {n}, oversampling {over}: sde_raw, sde, peak, power[0], power[peak], power[n-1], power_raw[peak], uncertainty")
    print(f"({n}, {over}, {sde_raw!r}, {sde!r}, {peak}, {power[0]!r}, {power[peak]!r}, {power[-1]!r}, {power_raw[peak]!r}, {unc!r}),")
