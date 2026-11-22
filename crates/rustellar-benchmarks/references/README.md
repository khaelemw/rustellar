# Reference generators

The Python scripts that made every reference file the rustellar crates are
tested against, and that timed the Python packages in the speed tables.
They live here, in a crate that is never published, so the science crates
stay free of Python: only a clone of the repository has them.

## Setting up

Python 3.12, with the packages at the versions the files were made with:

```text
python -m venv .venv
.venv/Scripts/activate            (Windows; source .venv/bin/activate elsewhere)
pip install -r crates/rustellar-benchmarks/references/requirements.txt
```

Run every script from the repository's root, as its first lines say:

```text
python crates/rustellar-benchmarks/references/make_ls_reference.py
```

Each writes its files where the tests read them (a crate's
`tests/reference/`, or `crates/rustellar-benchmarks/data/`) and overwrites
what is there; `git diff` then shows whether this machine reproduces them.
The light curves are made from fixed seeds, so a rerun with the same
versions gives the same files, to the last digit for the exact
(mpmath) values. Scratch files go in `work/`, which git ignores.

## Accuracy references

| Script | For | Writes |
|---|---|---|
| `make_reference.py` | model: batman's light curves | `batman_points.csv`, `batman_sets.csv` |
| `make_lightcurve_reference.py` | model: batman's orbits with Agol's flux | `lightcurve_points.csv`, `lightcurve_sets.csv` |
| `make_flux_reference.py` | model: the flux at given separations | `flux_points.csv` |
| `make_random_flux_reference.py` | model: exact flux over the whole parameter range | `flux_random.csv` |
| `make_exact_reference.py` | model: the blocked-light integrals at 60 digits | `exact_lines.csv`, `exact_random.csv` |
| `make_exposure_reference.py` | model: exposure-averaged curves, batman and PyTransit | `exposure_points.csv` |
| `make_eccentric_reference.py` | model: eccentric orbits | `eccentric_contacts.csv`, `eccentric_points.csv` |
| `make_detrend_location_reference.py` | detrend: median and mean of every window | `location_*.csv` |
| `make_detrend_biweight_reference.py` | detrend: biweight at 50 digits, with wotan's | `biweight_windows.csv` |
| `make_detrend_estimators_reference.py` | detrend: the other location estimators | `estimator_windows.csv` |
| `make_detrend_window_reference.py` | detrend: wotan's windows | `wotan_window_*.csv` |
| `make_detrend_spline_reference.py` | detrend: scipy's least-squares spline and an exact one | `spline_*.csv` |
| `make_detrend_robust_spline_reference.py` | detrend: the clipped spline, exact and scipy | `robust_spline_*.csv` |
| `make_detrend_huber_spline_reference.py` | detrend: the Huber spline, exact and statsmodels | `huber_spline_*.csv` |
| `make_detrend_huber_reference.py` | detrend: wotan's `huber` against proposal 2 solved exactly | printed |
| `make_detrend_ar_reference.py` | detrend: statsmodels' AutoReg and an exact fit | `ar_*.csv` |
| `make_detrend_survey_reference.py` | detrend: survey-scale curves with wotan | `survey_*.csv` |
| `make_detrend_survey_huber_reference.py` | detrend: Huber's M-estimator on the survey | `survey_huber.csv` |
| `make_detrend_survey_huber_joint_reference.py` | detrend: proposal 2 on the survey | `survey_huber_joint.csv` |
| `make_methods_wotan.py` | benchmarks: wotan on the window study's curves | `methods_*.csv` |
| `make_bls_reference.py` | bls: astropy's BoxLeastSquares, and the surveys and injections | many (see the script) |
| `make_tls_injection_reference.py` | tls: TLS on injected transits | printed as Rust literals |
| `make_tls_spectra_reference.py` | tls: TLS's spectra on fixed chi-squared arrays | printed as Rust literals |
| `make_tls_many_reference.py` | benchmarks: TLS over many seeds | `tls_many.csv` |
| `make_tls_uneven_reference.py` | benchmarks: TLS on uneven sampling, varying errors, 2-minute cadence | `tls_<kind>.csv` |
| `make_ls_reference.py` | ls: exact powers and astropy's | `ls_points.csv`, `ls_powers.csv` |
| `make_ls_models.py` | ls: exact peaks and best-fit sinusoids | `ls_models.csv` |
| `make_ls_fap.py` | ls: exact false-alarm probabilities | `ls_fap.csv` |
| `make_fit_reference.py` | fit: batman and PyTransit with scipy's least_squares | `fit_*.csv` |
| `make_vet_reference.py` | vet: LEO-Vetter and exovetter on shared curves | `vet_references.csv` |
| `make_mcmc_diagnostics_reference.py` | mcmc: emcee's and ArviZ's diagnostics | `crates/rustellar-mcmc/tests/reference/` |
| `make_transit_posterior_reference.py` | mcmc: emcee with batman on one transit | `crates/rustellar-mcmc/tests/reference/` |
| `verify_exovetter_odd_even.py` | vet: checks exovetter's odd/even test on its own functions | printed |

## Speed references

Each times a Python package on the same light curves as the Rust side and
writes or prints the rows of `crates/rustellar-benchmarks/data/speed_<crate>.csv`.
Time them on a quiet machine; the files in the repository were made on an
AMD Ryzen 7 5700G.

| Script | Times |
|---|---|
| `bench_reference.py` | batman, PyTransit and exoplanet-core |
| `bench_detrend_wotan.py` | wotan |
| `bench_detrend_spline.py` | scipy's splines |
| `bench_detrend_ar.py` | statsmodels' autoregressive fit |
| `bench_bls.py` | astropy's BoxLeastSquares |
| `bench_tls.py` | transitleastsquares |
| `bench_ls.py` | astropy's LombScargle |
| `bench_fit.py` | batman and PyTransit fits with scipy |
| `mcmc_transit_speed.py` | emcee with batman and with PyTransit |
| `mcmc_move_study.py` | emcee's autocorrelation times per move |
| `mcmc_zeus_study.py` | zeus's slice sampler |

Slow scripts (the TLS ones, `make_bls_reference.py`, the mcmc speed runs)
take from minutes to hours on 16 threads.
