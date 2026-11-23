## rustellar-tls speed

Every search on 16 threads.

| | Points | Periods | transitleastsquares 1.32 | rustellar-tls, no bootstrap | default (bootstrap 100) | GPU, no bootstrap |
|---|--:|--:|--:|--:|--:|--:|
| 1 sector, 30 min | 1,240 | 2,436 | 8.82 | 0.708 | 0.873 | 0.152 |
| 1 sector, 2 min | 18,000 | 2,349 | 13.2 | 1.44 | 1.86 | 0.765 |
| 2 sectors, 2 min | 36,000 | 5,411 | 38.2 | 3.92 | 4.65 | 1.36 |

`Method::Fold`, transitleastsquares' own way: 1.65, 9.26, 30.4 seconds.

Seconds a light curve, sixteen curves searched one after another and with `Tls::search_many` (no bootstrap):

| | Loop, CPU | `search_many`, CPU | Loop, GPU | `search_many`, GPU |
|---|--:|--:|--:|--:|
| 16 curves, 1 sector at 30 min | 0.775 | 0.562 | 0.180 | 0.112 |
| 16 curves, 1 sector at 2 min | 1.44 | 1.05 | 0.739 | 0.442 |

(402 s)

