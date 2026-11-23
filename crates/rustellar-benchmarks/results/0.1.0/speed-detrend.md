## rustellar-detrend speed

| Biweight | wotan 1.10 | rustellar-detrend, 1 thread | rustellar-detrend, 16 threads |
|---|--:|--:|--:|
| Kepler quarter | 0.0105 | 0.00441 | 0.00125 |
| TESS, 2 minutes | 0.180 | 0.0926 | 0.0109 |
| TESS, 20 seconds | 5.84 | 2.99 | 0.339 |

Each method on the 2-minute sector, one thread (wotan's default settings):

| | wotan 1.10 | rustellar-detrend |
|---|--:|--:|
| Median | 0.0582 | 0.00251 |
| Mean | 0.00958 | 0.00498 |
| Biweight | 0.180 | 0.0933 |
| Welsch | 0.321 | 0.209 |
| Andrews' sine wave | 0.323 | 0.201 |
| Huber (fixed scale) | - | 0.0742 |
| Huber's proposal 2 | 10.3 | 0.152 |
| Hodges-Lehmann | 9.19 | 0.913 |
| Trimmed mean | 0.196 | 0.0178 |
| Winsorized mean | 0.196 | 0.0207 |

The biweight on many light curves, seconds a light curve on 16 threads:

| | Curves | One after another | `flatten_many` |
|---|--:|--:|--:|
| Kepler quarter | 64 | 0.00163 | 0.000495 |
| TESS, 2 minutes | 32 | 0.0121 | 0.00951 |
| TESS, 20 seconds | 8 | 0.349 | 0.305 |

(53 s)

