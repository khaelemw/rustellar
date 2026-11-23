## rustellar-detrend speed

| Biweight | wotan 1.10 | rustellar-detrend, 1 thread | rustellar-detrend, 16 threads |
|---|--:|--:|--:|
| Kepler quarter | 0.0105 | 0.00430 | 0.00120 |
| TESS, 2 minutes | 0.180 | 0.0901 | 0.0109 |
| TESS, 20 seconds | 5.84 | 2.90 | 0.328 |

Each method on the 2-minute sector, one thread (wotan's default settings):

| | wotan 1.10 | rustellar-detrend |
|---|--:|--:|
| Median | 0.0582 | 0.00250 |
| Mean | 0.00958 | 0.00513 |
| Biweight | 0.180 | 0.0902 |
| Welsch | 0.321 | 0.206 |
| Andrews' sine wave | 0.323 | 0.198 |
| Huber (fixed scale) | - | 0.0727 |
| Huber's proposal 2 | 10.3 | 0.148 |
| Hodges-Lehmann | 9.19 | 0.872 |
| Trimmed mean | 0.196 | 0.0175 |
| Winsorized mean | 0.196 | 0.0215 |

The biweight on many light curves, seconds a light curve on 16 threads:

| | Curves | One after another | `flatten_many` |
|---|--:|--:|--:|
| Kepler quarter | 64 | 0.00136 | 0.000468 |
| TESS, 2 minutes | 32 | 0.0117 | 0.00934 |
| TESS, 20 seconds | 8 | 0.338 | 0.307 |

(52 s)

