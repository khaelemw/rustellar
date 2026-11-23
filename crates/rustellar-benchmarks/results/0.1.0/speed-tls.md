## rustellar-tls speed

Every search on 16 threads.

| | Points | Periods | transitleastsquares 1.32 | rustellar-tls, no bootstrap | default (bootstrap 100) | GPU, no bootstrap |
|---|--:|--:|--:|--:|--:|--:|
| 1 sector, 30 min | 1,240 | 2,436 | 8.82 | 0.797 | 0.981 | 0.160 |
| 1 sector, 2 min | 18,000 | 2,349 | 13.2 | 1.71 | 2.11 | 0.932 |
| 2 sectors, 2 min | 36,000 | 5,411 | 38.2 | 4.46 | 5.18 | 1.45 |

`Method::Fold`, transitleastsquares' own way: 1.75, 10.2, 34.4 seconds.

Seconds a light curve, sixteen curves searched one after another and with `Tls::search_many` (no bootstrap):

| | Loop, CPU | `search_many`, CPU | Loop, GPU | `search_many`, GPU |
|---|--:|--:|--:|--:|
| 16 curves, 1 sector at 30 min | 0.860 | 0.659 | 0.185 | 0.115 |
| 16 curves, 1 sector at 2 min | 1.66 | 1.20 | 0.848 | 0.523 |

(455 s)

